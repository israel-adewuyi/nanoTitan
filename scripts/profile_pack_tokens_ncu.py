from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "profiles" / "nsight_compute" / "pack_tokens"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile pack_tokens_kernel_cu with Nsight Compute."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    parser.add_argument("--set", dest="section_set", default="speed-of-light")
    parser.add_argument("--kernel-name", default="regex:.*pack_tokens_kernel_cu.*")
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--no-force-overwrite", action="store_true")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--check", action="store_true")

    return parser.parse_args()


def make_stem(args: argparse.Namespace) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"pack_tokens_{stamp}_tok{args.tokens}_hd{args.hidden_dim}_topk{args.top_k}_{args.dtype}"


def build_command(
    args: argparse.Namespace,
    ncu: str,
    report_path: Path,
) -> list[str]:
    benchmark_cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "bench_pack_tokens.py"),
        "--device",
        str(args.device),
        "--tokens",
        str(args.tokens),
        "--hidden-dim",
        str(args.hidden_dim),
        "--top-k",
        str(args.top_k),
        "--num-experts",
        str(args.num_experts),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(max(args.iters, args.profile_launches)),
    ]
    if args.check:
        benchmark_cmd.append("--check")

    ncu_cmd = [
        ncu,
        "--target-processes",
        "all",
        "--set",
        args.section_set,
        "--kernel-name",
        args.kernel_name,
        "--launch-skip",
        str(args.warmup),
        "--launch-count",
        str(args.profile_launches),
        "--export",
        str(report_path),
    ]
    if not args.no_force_overwrite:
        ncu_cmd.append("--force-overwrite")

    return [*ncu_cmd, *benchmark_cmd]


def run_and_log(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def main() -> int:
    args = parse_args()
    ncu = shutil.which(args.ncu)
    if ncu is None:
        print(
            f"Could not find Nsight Compute CLI '{args.ncu}'. "
            "Install Nsight Compute or set NCU=/path/to/ncu.",
            file=sys.stderr,
        )
        return 127

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = make_stem(args)
    report_path = results_dir / f"{stem}.ncu-rep"
    log_path = results_dir / f"{stem}.log"
    command_path = results_dir / f"{stem}.cmd.txt"

    command = build_command(args, ncu, report_path)
    command_path.write_text(format_command(command) + "\n", encoding="utf-8")

    print(f"Writing Nsight Compute report to: {report_path}")
    print(f"Writing command output log to:  {log_path}")
    return run_and_log(command, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
