import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_inputs(tokens, hidden_dim, top_k, device, dtype):
    assignments = tokens * top_k
    expert_out = torch.randn(assignments, hidden_dim, device=device, dtype=dtype)
    token_ids = torch.arange(tokens, device=device, dtype=torch.int32).repeat_interleave(top_k)
    weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return expert_out, token_ids, weights.reshape(-1).contiguous()


def torch_combine(expert_out, token_ids, weights, tokens, hidden_dim):
    out = torch.zeros(tokens, hidden_dim, device=expert_out.device, dtype=expert_out.dtype)
    out.index_add_(0, token_ids.long(), expert_out * weights[:, None].to(expert_out.dtype))
    return out


def cuda_combine(expert_out, token_ids, weights, tokens, hidden_dim):
    import random_ext

    return random_ext.combine_tokens_kernel(expert_out, token_ids, weights, tokens, hidden_dim).to(
        expert_out.dtype
    )


def time_ms(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "cuda", "both"], default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch environment.")

    if args.backend in ("cuda", "both"):
        try:
            import random_ext  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "random_ext is not installed; build/install it before CUDA runs."
            ) from exc

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    inputs = make_inputs(args.tokens, args.hidden_dim, args.top_k, device, dtype)
    bench_args = (*inputs, args.tokens, args.hidden_dim)

    if args.check:
        if args.backend == "torch":
            torch_combine(*bench_args)
        else:
            torch.testing.assert_close(
                cuda_combine(*bench_args),
                torch_combine(*bench_args),
                rtol=1e-2,
                atol=1e-2,
            )
        print("Correctness check passed.")

    backends = ["torch", "cuda"] if args.backend == "both" else [args.backend]
    print("COMBINE BENCHMARK")
    print("=================")
    print(
        f"device=cuda:{args.device} dtype={args.dtype} tokens={args.tokens} "
        f"hidden_dim={args.hidden_dim} top_k={args.top_k} iters={args.iters}"
    )
    for backend in backends:
        fn = torch_combine if backend == "torch" else cuda_combine
        avg_ms = time_ms(fn, bench_args, args.warmup, args.iters)
        print(f"{backend:>5}: {avg_ms:.4f} ms")


if __name__ == "__main__":
    main()
