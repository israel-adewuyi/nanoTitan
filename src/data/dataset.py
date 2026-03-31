from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

        # need seq_len + 1 tokens because y is shifted by 1
        self.num_sequences = (len(self.tokens) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start * self.seq_len + 1

        chunk = self.tokens[start:end].astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
