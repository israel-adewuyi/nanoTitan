import numpy as np
from pathlib import Path
from src.data.tokenizer import TiktokenTokenizer


def write_token_file(token_ids: list[int], path: str) -> None:
    token_arr = np.array(token_ids, dtype=np.uint16)
    Path(path).parent.mkdir(parents=True, exists_ok=True)
    token_arr.tofile(path)


def main() -> None:
    tokenizer = TiktokenTokenizer("gpt2")

    dataset = load_dataset("roneneldan/TinyStories")
    train_dataset = dataset["train"]
    val_dataset = dataset["validatioin"] if "validation" in dataset else None

    def tokenize_split(split):
        all_tokens = []
        for ex in split:
            text = ex["text"]
            ids = tokenizer.encode(text)
            all_tokens.extend(ids)
        return all_tokens

    train_tokens = tokenize_split(train_dataset)
    write_token_file(train_tokens, "data/preprocessed/tinystories_train.bin")

    if val_ds is not None:
        val_tokens = tokenize_split(val_dataset)
        write_token_file(val_tokens, "data/preprocessed/tinystories_val.bin")


if __name__ == "__main__":
    main()