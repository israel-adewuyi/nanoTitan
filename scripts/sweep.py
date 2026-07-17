from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from src.config import AppConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid sweep described by TOML.")
    parser.add_argument("sweep", type=Path, help="Path to the sweep TOML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print every run without launching training.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level forwarded to src.train (default: INFO).",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def safe_fragment(value: Any) -> str:
    if isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def make_run_name(base_name: str, overrides: dict[str, Any]) -> str:
    suffixes = []
    for key, value in overrides.items():
        key_fragment = safe_fragment(key.replace(".", "-"))
        suffixes.append(f"{key_fragment}-{safe_fragment(value)}")
    return "__".join((safe_fragment(base_name), *suffixes))


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def format_toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def dump_toml(config: dict[str, Any]) -> str:
    lines: list[str] = []

    def write_table(table: dict[str, Any], path: tuple[str, ...]) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        child_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]

        if path:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(format_toml_key(part) for part in path) + "]")
        for key, value in scalar_items:
            lines.append(f"{format_toml_key(key)} = {format_toml_value(value)}")
        for key, value in child_items:
            write_table(value, (*path, key))

    write_table(config, ())
    return "\n".join(lines) + "\n"


def build_runs(sweep_path: Path) -> tuple[Path, list[tuple[str, dict[str, Any]]]]:
    sweep = load_toml(sweep_path)
    base_path = Path(sweep["base_config"]).resolve()
    raw_base = load_toml(base_path)
    base = AppConfig.model_validate(raw_base).model_dump(mode="json")
    parameters = sweep["parameters"]
    keys = list(parameters)
    runs: list[tuple[str, dict[str, Any]]] = []
    for values in itertools.product(*(parameters[key] for key in keys)):
        overrides = dict(zip(keys, values, strict=True))
        run_config = copy.deepcopy(base)
        for key, value in overrides.items():
            set_dotted_value(run_config, key, value)
        run_config["run_name"] = make_run_name(str(base["run_name"]), overrides)
        AppConfig.model_validate(run_config)
        runs.append((run_config["run_name"], run_config))
    return base_path, runs


def main() -> int:
    args = parse_args()
    sweep_path = args.sweep.resolve()
    base_path, runs = build_runs(sweep_path)
    print(f"Loaded {len(runs)} run(s) from {sweep_path} using {base_path}")

    with tempfile.TemporaryDirectory(prefix="nanotitan-sweep-") as temp_dir:
        temp_path = Path(temp_dir)
        for index, (run_name, config) in enumerate(runs, start=1):
            config_path = temp_path / f"run-{index:04d}.toml"
            config_path.write_text(dump_toml(config), encoding="utf-8")
            world_size = config["runtime"]["dp_size"] * config["runtime"]["pp_size"]
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={world_size}",
                "-m",
                "src.train",
                f"@{config_path}",
                f"--log.level={args.log_level}",
            ]
            print(f"[{index}/{len(runs)}] {run_name}")
            if args.dry_run:
                print("  " + subprocess.list2cmdline(command))
                continue

            subprocess.run(command, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
