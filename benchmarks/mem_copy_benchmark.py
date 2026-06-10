import random_ext
import torch


def bench_copy_scalar(dtype, numel, iters=1000, warmup=20):
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

    ok = torch.equal(src, dst)

    bytes_moved = src.numel() * src.element_size() * 2
    tensor_mb = src.numel() * src.element_size() / 1e6
    bandwidth_GBs = bytes_moved / (avg_ms / 1000) / 1e9
    traffic_gb = bytes_moved / 1e9

    print(
        f"dtype={dtype} "
        f"numel={numel} "
        f"tensor_MB={tensor_mb:.1f} "
        f"traffic_GB={traffic_gb:.2f} "
        f"ok={ok} "
        f"avg_us={avg_ms * 1000:.2f} "
        f"bandwidth_GBps={bandwidth_GBs:.2f}"
    )


def bench_copy_vector(dtype, numel, iters=1000, warmup=20):
    device = "cuda"
    src = torch.randn(numel, device=device, dtype=dtype)
    dst = torch.empty_like(src)

    # warmup
    for _ in range(warmup):
        random_ext.copy_vector(src, dst)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        random_ext.copy_vector(src, dst)
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    ok = torch.equal(src, dst)

    bytes_moved = src.numel() * src.element_size() * 2
    tensor_mb = src.numel() * src.element_size() / 1e6
    bandwidth_GBs = bytes_moved / (avg_ms / 1000) / 1e9
    traffic_gb = bytes_moved / 1e9

    print(
        f"dtype={dtype} "
        f"numel={numel} "
        f"tensor_MB={tensor_mb:.1f} "
        f"traffic_GB={traffic_gb:.2f} "
        f"ok={ok} "
        f"avg_us={avg_ms * 1000:.2f} "
        f"bandwidth_GBps={bandwidth_GBs:.2f}"
    )


def bench_peer_copy_scalar(dtype, numel, iters=100, warmup=20):
    src_device, dest_device = 0, 1
    src = torch.randn(numel, dtype=dtype, device=f"cuda:{src_device}")
    dest = torch.empty_like(src, device=f"cuda:{dest_device}")

    count = numel * src.element_size()

    # warmup
    for _ in range(warmup):
        random_ext.peer_copy_scalar(src, src_device, dest, dest_device, count)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        random_ext.peer_copy_scalar(src, src_device, dest, dest_device, count)
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    ok = torch.equal(src, dest)

    bytes_moved = src.numel() * src.element_size()
    tensor_mb = src.numel() * src.element_size() / 1e6
    bandwidth_GBs = bytes_moved / (avg_ms / 1000) / 1e9
    traffic_gb = bytes_moved / 1e9

    print(
        f"dtype={dtype} "
        f"numel={numel} "
        f"tensor_MB={tensor_mb:.1f} "
        f"traffic_GB={traffic_gb:.2f} "
        f"ok={ok} "
        f"avg_us={avg_ms * 1000:.2f} "
        f"bandwidth_GBps={bandwidth_GBs:.2f}"
    )


def main():
    sizes_bytes = [
        1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
    ]
    print("=" * 32, "SCALAR COPY KERNEL", "=" * 32)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for size_bytes in sizes_bytes:
            numel = size_bytes // torch.tensor([], dtype=dtype).element_size()
            bench_copy_scalar(dtype, numel)

    print("=" * 32, "VECTOR COPY KERNEL", "=" * 32)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for size_bytes in sizes_bytes:
            numel = size_bytes // torch.tensor([], dtype=dtype).element_size()
            bench_copy_vector(dtype, numel)


if __name__ == "__main__":
    main()
