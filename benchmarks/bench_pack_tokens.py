import argparse

import torch

import random_ext


def make_inputs(tokens, hidden_dim, top_k, num_experts, device, dtype):
    X = torch.randn(tokens, hidden_dim, device=device, dtype=dtype)

    # Deterministic-ish random routing.
    topk_experts = torch.randint(
        low=0,
        high=num_experts,
        size=(tokens, top_k),
        device=device,
        dtype=torch.int32,
    )

    topk_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)

    # Count expert assignments.
    flat_experts = topk_experts.reshape(-1)
    counts = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)

    expert_offsets = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(counts, dim=0)

    return (
        X,
        topk_weights,
        topk_experts,
        expert_offsets,
    )


def run_pack(
    X,
    topk_weights,
    topk_experts,
    expert_offsets,
):
    return random_ext.pack_tokens_kernel(
        X,
        topk_weights,
        topk_experts,
        expert_offsets,
    )


def correctness_check(
    X,
    topk_experts,
    topk_weights,
    expert_offsets,
    packed_X,
    packed_tokenId,
    packed_expert,
    packed_topk_weights,
):
    total_assignments = packed_X.shape[0]
    num_experts = expert_offsets.numel() - 1

    assert packed_X.shape[0] == total_assignments
    assert packed_tokenId.shape == (total_assignments,)
    assert packed_expert.shape == (total_assignments,)
    assert packed_topk_weights.shape == (total_assignments,)

    # Check each packed row corresponds to its recorded original token.
    for idx in range(total_assignments):
        token_id = packed_tokenId[idx].item()
        torch.testing.assert_close(packed_X[idx], X[token_id])

    # Check expert-major layout follows offsets.
    for e in range(num_experts):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()

        if start != end:
            assert torch.all(packed_expert[start:end] == e), (
                f"Expert {e} range [{start}, {end}) contains wrong expert ids"
            )

    # Optional stronger check: every packed assignment should match original routing.
    # This assumes packed_topk_weights stores the corresponding top-k gate.
    for idx in range(total_assignments):
        token_id = packed_tokenId[idx].item()
        expert_id = packed_expert[idx].item()
        weight = packed_topk_weights[idx]

        matches = topk_experts[token_id] == expert_id
        assert torch.any(matches), (
            f"Packed assignment idx={idx} has token={token_id}, "
            f"expert={expert_id}, but token was not routed to that expert"
        )

        matched_weight = topk_weights[token_id][matches][0]
        torch.testing.assert_close(weight, matched_weight)


def benchmark(args):
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    (
        X,
        topk_weights,
        topk_experts,
        expert_offsets,
    ) = make_inputs(
        tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        num_experts=args.num_experts,
        device=device,
        dtype=dtype,
    )

    total_assignments = args.tokens * args.top_k
    warmup_expert_offsets = [expert_offsets.clone() for _ in range(args.warmup)]

    # Warmup
    for warmup_offsets in warmup_expert_offsets:
        run_pack(
            X,
            topk_weights,
            topk_experts,
            warmup_offsets,
        )

    torch.cuda.synchronize()

    if args.check:
        check_offsets = expert_offsets.clone()
        packed_X, packed_tokenId, packed_expert, packed_topk_weights = run_pack(
            X,
            topk_weights,
            topk_experts,
            check_offsets,
        )
        torch.cuda.synchronize()
        correctness_check(
            X,
            topk_experts,
            topk_weights,
            expert_offsets,
            packed_X,
            packed_tokenId,
            packed_expert,
            packed_topk_weights,
        )
        torch.cuda.synchronize()
        print("Correctness check passed.")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timed_expert_offsets = [expert_offsets.clone() for _ in range(args.iters)]
    torch.cuda.synchronize()

    start.record()

    for timed_offsets in timed_expert_offsets:
        packed_X, packed_tokenId, packed_expert, packed_topk_weights = run_pack(
            X,
            topk_weights,
            topk_experts,
            timed_offsets,
        )

    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / args.iters

    dtype_size = X.element_size()

    # Approx bytes:
    # Read X once per assignment and write packed_X once per assignment.
    # Read topk_experts/topk_weights and write packed_tokenId/packed_expert/packed_topk_weights.
    bytes_x = total_assignments * args.hidden_dim * dtype_size * 2
    bytes_metadata = total_assignments * (4 + dtype_size + 4 + 4 + dtype_size)
    approx_bytes = bytes_x + bytes_metadata

    gbps = approx_bytes / (avg_ms / 1000.0) / 1e9

    print()
    print("PACK TOKENS BENCHMARK")
    print("=====================")
    print(f"device:             cuda:{args.device}")
    print(f"dtype:              {args.dtype}")
    print(f"tokens:             {args.tokens}")
    print(f"hidden_dim:         {args.hidden_dim}")
    print(f"top_k:              {args.top_k}")
    print(f"num_experts:        {args.num_experts}")
    print(f"assignments:        {total_assignments}")
    print(f"warmup:             {args.warmup}")
    print(f"iters:              {args.iters}")
    print(f"avg latency:        {avg_ms:.4f} ms")
    print(f"approx bytes:       {approx_bytes / 1e6:.2f} MB")
    print(f"approx bandwidth:   {gbps:.2f} GB/s")


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
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
