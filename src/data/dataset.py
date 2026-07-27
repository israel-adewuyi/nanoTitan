import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset

from src.data.tokenizer import TiktokenTokenizer


class PackedTokenDataset(IterableDataset):
    def __init__(self, name: str, seq_len: int, seed: int, rank: int, world_size: int):
        self.seq_len = seq_len
        dataset = load_dataset(name, split="train", streaming=True).shuffle(
            seed=seed, buffer_size=10_000
        )
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def __iter__(self):
        tokenizer = TiktokenTokenizer("gpt2")
        tokens = []
        for example in self.dataset:
            tokens.extend(tokenizer.encode(example["text"]))
            while len(tokens) >= self.seq_len + 1:
                chunk = tokens[: self.seq_len + 1]
                del tokens[: self.seq_len]
                yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])
