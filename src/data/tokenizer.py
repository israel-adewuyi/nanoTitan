from dataclasses import dataclass

import tiktoken


@dataclass
class TiktokenTokenizer:
    name: str = "gpt2"

    def __post_init__(self) -> None:
        self.enc = tiktoken.get_encoding(self.name)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)
