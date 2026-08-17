import argparse

import torch


def make_inputs(args, device, dtype):
    torch.manual_seed(args.seed)
    x = torch.randn(args.tokens, args.hidden_dim, device=device, dtype=dtype)
    topk_experts = torch.multinomial(
        torch.ones(args.tokens, args.num_experts, device=device),
        args.top_k,
        replacement=False,
    ).to(torch.int32)
    topk_weights = torch.rand(args.tokens, args.top_k, device=device, dtype=torch.float32)

    counts = torch.bincount(topk_experts.flatten().long(), minlength=args.num_experts).to(
        torch.int32
    )
    offsets = torch.empty(args.num_experts + 1, device=device, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)
    return x, topk_weights, topk_experts, offsets


def check_result(inputs, outputs):
    x, topk_weights, topk_experts, offsets = inputs
    packed_x, token_ids, packed_experts, packed_weights = outputs
    token_ids = token_ids.long()

    torch.testing.assert_close(packed_x, x[token_ids])

    counts = offsets[1:] - offsets[:-1]
    expected_experts = torch.arange(
        counts.numel(), device=x.device, dtype=torch.int32
    ).repeat_interleave(counts.long())
    torch.testing.assert_close(packed_experts, expected_experts)

    matches = topk_experts[token_ids] == packed_experts[:, None]
    assert matches.any(dim=1).all()
    route_slots = matches.int().argmax(dim=1)
    torch.testing.assert_close(packed_weights, topk_weights[token_ids, route_slots])


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--profile-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if args.top_k > args.num_experts:
        raise SystemExit("--top-k cannot exceed --num-experts.")

    try:
        import nanotitan_cuda
    except ModuleNotFoundError as exc:
        raise SystemExit("Build nanotitan_cuda before running this benchmark.") from exc

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    inputs = make_inputs(args, device, dtype)
    x, topk_weights, topk_experts, offsets = inputs

    def run(offset_copy):
        return nanotitan_cuda.pack_tokens_kernel(x, topk_weights, topk_experts, offset_copy)

    # The kernel mutates offsets with atomicAdd. Make the copies before timing so
    # the measured interval contains only pack-tokens launches.
    for offset_copy in [offsets.clone() for _ in range(args.warmup)]:
        run(offset_copy)
    torch.cuda.synchronize()

    if args.check:
        check_result(inputs, run(offsets.clone()))
        torch.cuda.synchronize()
        print("Correctness check passed.")

    timed_offsets = [offsets.clone() for _ in range(args.iters)]
    if args.profile_only:
        for offset_copy in timed_offsets:
            run(offset_copy)
        torch.cuda.synchronize()
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for offset_copy in timed_offsets:
        run(offset_copy)
    end.record()
    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / args.iters
    assignments = args.tokens * args.top_k
    logical_bytes = assignments * (2 * args.hidden_dim * x.element_size() + 28)
    bandwidth = logical_bytes / (latency_ms / 1000) / 1e9

    print("PACK TOKENS BENCHMARK")
    print(
        f"{args.dtype} | tokens={args.tokens} | hidden={args.hidden_dim} | "
        f"top_k={args.top_k} | experts={args.num_experts}"
    )
    print(f"latency: {latency_ms:.4f} ms")
    print(f"logical bandwidth: {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    main()
