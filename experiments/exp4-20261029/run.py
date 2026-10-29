"""Exp4: fixed model_q architecture and 4096-token EP batch-size measurements."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tomllib
from pathlib import Path

from scripts.sweep import dump_toml
from src.config import AppConfig

# Bracket the largest complete optimizer step; refine only if time permits.
BATCH_CANDIDATES = [1, 2, 4, 8, 16, 32]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--master-port", type=int, default=29520)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or not all(x.isdigit() for x in gpus):
        parser.error("--gpus must contain distinct GPU ordinals")
    if args.batch_size < 1 or args.steps < 1 or (args.profile and args.steps < 5):
        parser.error("positive batch/steps required; profiling needs at least five steps")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True
    ).strip():
        parser.error("Commit tracked source changes before running")
    raw = tomllib.loads((repo / "examples/model_q.toml").read_text())
    raw["run_name"] = args.name or f"model-q-ep{len(gpus)}-b{args.batch_size}-s4096" + (
        "-profile" if args.profile else ""
    )
    raw["max_steps"] = args.steps
    raw["trainer"]["per_device_batch_size"] = args.batch_size
    raw["runtime"].update(dp_size=1, pp_size=1, ep_size=len(gpus), num_microbatches=1)
    raw["data"]["dataset_path"] = str(
        (args.dataset or repo / raw["data"]["dataset_path"]).resolve()
    )
    assert raw["model"]["max_seq_len"] == 4096
    if args.profile:
        raw["profiler"] = dict(
            enabled=True,
            wait_steps=2,
            warmup_steps=1,
            active_steps=1,
            record_shapes=True,
            profile_memory=True,
        )
    cfg = AppConfig.model_validate(raw)
    run_dir = args.output_dir.resolve() / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "resolved.toml"
    path.write_text(dump_toml(raw))
    if args.dry_run:
        cmd = [
            "uv",
            "--no-config",
            "run",
            "--no-project",
            "--active",
            "python",
            "-m",
            "src.train",
            str(path),
            "--dry-run",
        ]
    else:
        cmd = [
            "uv",
            "--no-config",
            "run",
            "--no-project",
            "--active",
            "torchrun",
            f"--nproc-per-node={len(gpus)}",
            f"--master-port={args.master_port}",
            "-m",
            "src.train",
            str(path),
        ]
    record = dict(commit=commit, config=raw, gpus=gpus, command=cmd)
    (run_dir / "launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(shlex.join(cmd), flush=True)
    env = dict(
        os.environ,
        HIP_VISIBLE_DEVICES=args.gpus,
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONPATH=os.pathsep.join(
            [str(repo), str(repo / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    )
    raise SystemExit(subprocess.call(cmd, cwd=run_dir, env=env))


if __name__ == "__main__":
    main()
