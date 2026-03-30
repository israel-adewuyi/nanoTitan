from __future__ import annotations

import argparse

from src.config import load_config
from src.model import NanoTitanModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instantiate NanoTitan from a TOML config")
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to a TOML config file. Relative paths can be given from the repo root or configs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config)
    model = NanoTitanModel(app_config.model)

    print(f"Loaded model config from {args.config}")
    print(app_config.model.model_dump())
    print(f"Instantiated {model.__class__.__name__}")


if __name__ == "__main__":
    main()
