import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "bench_pack_tokens.py"


def benchmark_args(args, iters, profile_only=False):
    command = [
        sys.executable,
        str(BENCHMARK),
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
        str(iters),
        "--seed",
        str(args.seed),
    ]
    if args.check and not profile_only:
        command.append("--check")
    if profile_only:
        command.append("--profile-only")
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pack_tokens, then profile it with Nsight Compute."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--set", default="detailed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("Nsight Compute CLI (ncu) was not found.")

    output = args.output
    if output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = (
            ROOT / "profiles" / "nsight_compute" / "pack_tokens" / (f"pack_tokens_{stamp}.ncu-rep")
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Running clean benchmark...")
    subprocess.run(benchmark_args(args, args.iters), cwd=ROOT, check=True)

    print(f"Profiling to {output}...")
    profile_command = [
        ncu,
        "--set",
        args.set,
        "--kernel-name",
        "regex:.*pack_tokens_kernel_cu.*",
        "--launch-skip",
        str(args.warmup),
        "--launch-count",
        str(args.profile_launches),
        "--export",
        str(output),
        "--force-overwrite",
        *benchmark_args(args, args.profile_launches, profile_only=True),
    ]
    subprocess.run(profile_command, cwd=ROOT, check=True)
    print(f"Saved Nsight report: {output}")


if __name__ == "__main__":
    main()
