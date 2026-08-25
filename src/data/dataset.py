from pathlib import Path

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset

from src.data.tokenizer import TiktokenTokenizer


class PackedTokenDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        split: str = "train",
        shuffle: bool = True,
        dataset_path: str | Path | None = None,
    ):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.cached_sequences = None

        if dataset_path is not None:
            cached_sequences = torch.load(dataset_path, map_location="cpu", weights_only=True)
            if not isinstance(cached_sequences, torch.Tensor):
                raise TypeError("Cached dataset must contain a torch.Tensor")
            if cached_sequences.ndim != 2 or cached_sequences.shape[1] != seq_len + 1:
                raise ValueError(
                    f"Cached dataset must have shape [N, {seq_len + 1}], "
                    f"got {tuple(cached_sequences.shape)}"
                )
            if cached_sequences.dtype != torch.long:
                raise ValueError("Cached dataset tensor must have dtype torch.int64")
            self.cached_sequences = cached_sequences
            self.dataset = None
            return

        dataset = load_dataset(name, split=split, streaming=True)
        if shuffle:
            dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def __iter__(self):
        if self.cached_sequences is not None:
            for idx in range(self.rank, len(self.cached_sequences), self.world_size):
                sequence = self.cached_sequences[idx]
                yield sequence[:-1], sequence[1:]
            return

        tokenizer = TiktokenTokenizer("gpt2")
        tokens = []
        for example in self.dataset:
            tokens.extend(tokenizer.encode(example["text"]))
            while len(tokens) >= self.seq_len + 1:
                chunk = tokens[: self.seq_len + 1]
                del tokens[: self.seq_len]
                yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])
