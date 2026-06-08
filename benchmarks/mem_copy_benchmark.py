import torch
import random_ext


def bench_copy(dtype, numel, iters=1000, warmup=20):
    device = "cuda"
    src = torch.randn(numel, device=device, dtype=dtype)
    dst = torch.empty_like(src)

    # warmup
    for _ in range(warmup):
        random_ext.copy_scalar(src, dst, src.numel())

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        random_ext.copy_scalar(src, dst, src.numel())
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    ok = torch.allclose(src, dst)

    bytes_moved = src.numel() * src.element_size()
    gbps_one_way = bytes_moved / (avg_ms / 1000) / 1e9
    gbps_read_write = 2 * bytes_moved / (avg_ms / 1000) / 1e9

    print(
        f"dtype={dtype} "
        f"numel={numel} "
        f"size_mb={bytes_moved / 1e6:.1f} "
        f"ok={ok} "
        f"avg_us={avg_ms * 1000:.2f} "
        f"GB/s_oneway={gbps_one_way:.2f} "
        f"GB/s_readwrite={gbps_read_write:.2f}"
    )


def main():
    sizes_bytes = [
        1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
    ]

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for size_bytes in sizes_bytes:
            numel = size_bytes // torch.tensor([], dtype=dtype).element_size()
            bench_copy(dtype, numel)


if __name__ == "__main__":
    main()
