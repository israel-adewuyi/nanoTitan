import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


def load_nanotitan_cuda():
    try:
        import nanotitan_cuda
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "nanotitan_cuda is not installed; build/install it before CUDA runs."
        ) from exc
    return nanotitan_cuda


def make_inputs(m, n, k, device, dtype):
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    return a, b


def run_gemm(nanotitan_cuda, implementation, a, b):
    if implementation == "cublas":
        return torch.mm(a, b)
    if implementation == "naive":
        return nanotitan_cuda.naive_gemm_kernel(a, b)
    if implementation == "tiled":
        return nanotitan_cuda.tiled_gemm_kernel(a, b)
    raise ValueError(f"Unknown GEMM implementation: {implementation}")


def time_ms(nanotitan_cuda, implementation, a, b, warmup, iters):
    for _ in range(warmup):
        run_gemm(nanotitan_cuda, implementation, a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        run_gemm(nanotitan_cuda, implementation, a, b)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=["naive", "tiled", "cublas", "both", "all"],
        default="all",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "gemm_benchmark.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch environment.")

    nanotitan_cuda = load_nanotitan_cuda()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    a, b = make_inputs(args.m, args.n, args.k, device, dtype)

    if args.implementation == "all":
        implementations = ["naive", "tiled", "cublas"]
    elif args.implementation == "both":
        implementations = ["naive", "tiled"]
    else:
        implementations = [args.implementation]

    if args.check:
        expected = a @ b
        for implementation in implementations:
            actual = run_gemm(nanotitan_cuda, implementation, a, b)
            torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])
        torch.cuda.synchronize()
        print("Correctness check passed.")

    print("GEMM BENCHMARK")
    print("==============")
    print(
        f"device=cuda:{args.device} dtype={args.dtype} m={args.m} n={args.n} "
        f"k={args.k} warmup={args.warmup} iters={args.iters}"
    )

    flops = 2 * args.m * args.n * args.k
    rows = []
    for implementation in implementations:
        avg_ms = time_ms(nanotitan_cuda, implementation, a, b, args.warmup, args.iters)
        tflops = flops / (avg_ms / 1000.0) / 1e12
        print(f"{implementation:>5}: {avg_ms:.4f} ms ({tflops:.2f} TFLOP/s)")
        rows.append([implementation, avg_ms, tflops])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["implementation", "latency_ms", "tflops"])
        writer.writerows(rows)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
