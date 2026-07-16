import argparse
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig  # noqa: E402


def make_cfg(args, backend, dtype):
    cfg = ModelConfig(
        vocab_size=1,
        d_model=args.d_model,
        d_head=1,
        n_heads=1,
        n_layers=1,
        max_seq_len=args.seq_len,
        ffn_in=args.ffn_in,
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_backend=backend,
        dtype=dtype,
        moe_router_dtype="float32",
    )
    cfg.dtype = getattr(torch, dtype)
    cfg.moe_router_dtype = torch.float32
    return cfg


def time_ms(moe, x, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            moe(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            moe(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "cuda", "both"], default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--ffn-in", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch environment.")

    try:
        import nanotitan_cuda  # noqa: F401
    except ModuleNotFoundError as exc:
        if args.backend in ("cuda", "both"):
            raise SystemExit(
                "nanotitan_cuda is not installed; build/install it before CUDA runs."
            ) from exc
        sys.modules["nanotitan_cuda"] = types.SimpleNamespace()

    from src.model.feed_fwd import MoE

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch_dtype = getattr(torch, args.dtype)
    x = torch.randn(args.batch, args.seq_len, args.d_model, device=device, dtype=torch_dtype)

    backends = ["torch", "cuda"] if args.backend == "both" else [args.backend]
    print("MOE BENCHMARK")
    print("=============")
    print(
        f"device=cuda:{args.device} dtype={args.dtype} batch={args.batch} "
        f"seq_len={args.seq_len} d_model={args.d_model} ffn_in={args.ffn_in} "
        f"num_experts={args.num_experts} top_k={args.top_k} iters={args.iters}"
    )
    for backend in backends:
        moe = MoE(make_cfg(args, backend, args.dtype)).to(device).eval()
        avg_ms = time_ms(moe, x, args.warmup, args.iters)
        print(f"{backend:>5}: {avg_ms:.4f} ms")


if __name__ == "__main__":
    main()
