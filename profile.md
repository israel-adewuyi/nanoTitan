# Kernel Profiling

## Pack Tokens

Requires the CUDA environment, the built `nanotitan_cuda` extension, and
Nsight Compute (`ncu`) on `PATH`.

### Benchmark

```bash
uv run --no-sync python benchmarks/bench_pack_tokens.py --check
```

### Benchmark and profile

```bash
uv run --no-sync python scripts/profile_pack_tokens_ncu.py --check
```

The profile command first runs a clean benchmark, then saves an `.ncu-rep`
under `profiles/nsight_compute/pack_tokens/`. Use `--help` to change the shape,
dtype, device, iteration count, or profiler section set.
