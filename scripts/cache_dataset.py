from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import torch

from src.data.dataset import PackedTokenDataset


def materialize_dataset(
    dataset_name: str,
    output: Path,
    seq_len: int,
    num_sequences: int,
    seed: int = 42,
    split: str = "train",
    shuffle: bool = True,
) -> Path:
    if seq_len <= 0 or num_sequences <= 0:
        raise ValueError("seq_len and num_sequences must be positive")

    source = PackedTokenDataset(
        name=dataset_name,
        seq_len=seq_len,
        seed=seed,
        rank=0,
        world_size=1,
        split=split,
        shuffle=shuffle,
    )
    cached = torch.empty((num_sequences, seq_len + 1), dtype=torch.long)
    count = 0
    for count, (input_ids, target_ids) in enumerate(islice(source, num_sequences), start=1):
        cached[count - 1, :-1] = input_ids
        cached[count - 1, -1] = target_ids[-1]

    if count != num_sequences:
        raise RuntimeError(
            f"Dataset ended after {count} packed sequences; requested {num_sequences}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a packed streaming dataset locally.")
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--num-sequences", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--split", default="train")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = materialize_dataset(
        dataset_name=args.dataset_name,
        output=args.output,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        seed=args.seed,
        split=args.split,
        shuffle=not args.no_shuffle,
    )
    cached = torch.load(output, map_location="cpu", weights_only=True)
    print(f"Saved {tuple(cached.shape)} packed tokens to {output}")


if __name__ == "__main__":
    main()
