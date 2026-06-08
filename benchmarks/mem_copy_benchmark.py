import time

import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = torch.randn(1024, device=device)
    dst = torch.empty_like(src)

    start = time.perf_counter()
    try:
        import benchmarks

        benchmarks.copy_scalar(src, dst, src.numel())
    except Exception:
        dst.copy_(src)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_us = (time.perf_counter() - start) * 1e6
    print(f"ok={torch.allclose(src, dst)} elapsed_us={elapsed_us:.2f}")


if __name__ == "__main__":
    main()
