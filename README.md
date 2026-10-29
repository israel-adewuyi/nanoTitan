# nanoTitan

**nanoTitan is a Mixture-of-Experts training stack for learning distributed training and CUDA kernel engineering from first principles.**

The repository currently contains a small autoregressive LM, 3D parallelism (DP + PP + EP), a CUDA-backed MoE dispatch path, autograd kernels, and basic grouped-GEMM implementations.

---

## What is implemented

### Distributed training

- DP, bucketed reducer with asynchronous all-reduce calls.
- Explicit process-group construction.
- GPipe, 1F1B pipeline parallelism schedules.
- Expert parallelism with all-to-all token dispatch across sharded experts.
- 3D DP × PP x EP composition.

### Mixture of Experts

- Top-k routing, expert-capacity load balancing.
- Load-balancing loss and per-layer routing statistics.
- CUDA kernels for expert counting, token packing, and weighted combine.
- Custom autograd wrappers and backward kernels for pack and combine.
- FP32, FP16, and BF16 correctness tests for dispatch operations.
- End-to-end pack → combine gradient parity tests.

### CUDA and profiling

- Expert-grouped GEMM kernels.
- Tests covering both up-projection and down-projection matrix shapes.
- Pack/combine microbenchmarks and an Nsight Compute launcher.

---

## Repository layout

```text
nanoTitan/
├── benchmarks/        # CUDA and training microbenchmarks
├── configs/           # Model and parallelism configurations
├── csrc/
│   ├── kernels/       # CUDA dispatch, combine, and GEMM kernels
│   └── runtime/       # C++/CUDA runtime experiments
├── scripts/           # Profiling utilities
├── src/
│   ├── data/          # Packed-token data pipeline
│   ├── model/         # Transformer and MoE implementations
│   └── parallel/      # Data and pipeline parallelism
└── tests/             # Model, CUDA, autograd, and distributed tests
```

---

## Getting Started

Clone repo

```
git clone https://github.com/israel-adewuyi/nanoTitan.git && cd nanoTitan
```

On your NVIDIA GPU machine, with a CUDA 13 development toolkit (`nvcc`) and a C++ compiler installed, install the GPU environment and build the CUDA extension:

```bash
uv sync --python 3.11 --locked --extra gpu
```

CUDA 13 is the default NVIDIA target: `gpu` selects PyTorch 2.13.0's CUDA 13.0 build (`cu130`) and uses the machine's installed toolkit to compile the extension. Select one accelerator extra per environment: `gpu`, `cpu`, or `rocm`. After installation, use `uv run --no-sync` to preserve the selected environment when running training or tests.

With the ROCm toolkit installed and `hipcc` on `PATH`, install the ROCm environment and build the extension:

```bash
uv sync --python 3.11 --locked --extra rocm --dev
```

PyTorch automatically translates the existing CUDA sources to HIP. For MI355X, set `PYTORCH_ROCM_ARCH=gfx950` before building.

### Cache dataset locally from the streaming dataset:

```bash
uv run --no-sync python -m scripts.cache_dataset \
  --dataset-name roneneldan/TinyStories \
  --seq-len 768 \
  --num-sequences 100000 \
  --output data/tinystories_seq768.pt
```

Then select it in a training configuration:

```toml
[data]
dataset_name = "roneneldan/TinyStories"
dataset_path = "data/tinystories_seq768.pt"
```

The cached sequence length must match `model.max_seq_len`.

Launch a distributed training job on 4 GPUs:

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=4 -m src.train configs/big_sabaka.toml
```

## Testing

### Config dry run (CPU)

Validate a configuration and count the full model's total and active (top-k) parameters:

```bash
uv run --no-sync python -m src.train configs/big_sabaka.toml --dry-run
```

Use the CPU environment below if dependencies are not installed. This constructs the real model on PyTorch's `meta` device, which stores tensor metadata without allocating weights. It works with CUDA configs and distributed layouts on a CPU machine, and counts all layers and experts once, regardless of parallelism. No CUDA extension, dataset, or `torchrun` launch is needed.

The check validates config values, routing settings, and layer/expert/microbatch divisibility. It does not execute forward/backward passes or verify datasets, CUDA kernels, distributed execution, or whether the model fits on your training hardware.

### CPU-compatible tests

Install the CPU environment:

```bash
uv sync --locked --extra cpu --dev --no-install-project
```

Run tests that do not require CUDA or multiple distributed ranks:

```bash
uv run --no-sync pytest -m "not cuda and not distributed"
```

### CUDA tests

Run only the CUDA tests:

```bash
uv run --no-sync pytest -m "cuda and not distributed"
```

Run all single-process tests, including CPU-compatible and CUDA tests:

```bash
uv run --no-sync pytest -m "not distributed"
```

### Distributed tests

Run all distributed tests on the same two GPUs:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=2 --module pytest -q tests/distributed/
```

All current distributed tests require exactly two ranks and run sequentially in this launch, reusing the same two GPUs. This includes DP training-step and PP forward/backward equivalence in FP32, FP16, and BF16, plus the FP32 torch EP correctness test.

The tests use NCCL when CUDA and NCCL are available, and otherwise fall back to Gloo on CPU. BF16 cases are skipped if either GPU lacks BF16 support.

### Complete test suite

On a machine with the CUDA extension built and two available ranks, run:

```bash
uv run --no-sync pytest -m "not distributed"
uv run --no-sync torchrun --standalone --nproc-per-node=2 --module pytest -q tests/distributed/
```

---

# Benchmark Results

## Model Specs

These specifications correspond to [`configs/big_sabaka.toml`](configs/big_sabaka.toml).

| Spec                   |                                  Value |
| ---------------------- | -------------------------------------: |
| Parameters             |                              1B-A-170M |
| Layers                 |                                     16 |
| Hidden size            |                                    512 |
| Attention heads        |                                      8 |
| Head dimension         |                                     64 |
| Experts per MoE layer  |                                     20 |
| Active experts/token   |                                      2 |
| Expert FFN size        |                                  2,048 |
| Expert capacity factor |                                   1.25 |
| Vocabulary size        |                                 50,257 |
| Sequence length        |                                    512 |
| Model precision        |                               bfloat16 |
| Router precision       |                                float32 |
| MoE backend            |                                   CUDA |
| Optimizer              |                                  AdamW |
| Learning rate          |                                   5e-4 |
| Gradient clipping      |                                    2.0 |
| Dataset                | TinyStories (`roneneldan/TinyStories`) |
| Seed                   |                                     42 |

## Benchmark Table

# Summary metrics

For the swept PP experiments, the selected runs use the default 1F1B schedule with activation checkpointing enabled. The 2-GPU result uses 7 microbatches, while the 4-GPU result uses 12 microbatches. Where an 8-GPU composition has two configurations, the configuration with the higher measured throughput is reported.

#### 2 GPUs

| Mode                    | GPUs |  DP |  PP |  EP | Total batch size | Tokens/sec | Step time (s) |
| ----------------------- | ---: | --: | --: | --: | ---------------: | ---------: | ------------: |
| DP                      |    2 |   2 |   1 |   1 |               42 |    2,665.9 |         8.066 |
| PP · 1F1B · AC on · M=7 |    2 |   1 |   2 |   1 |               42 |    2,155.1 |         9.979 |
| EP                      |    2 |   1 |   1 |   2 |               42 |    2,638.8 |         8.150 |

#### 4 GPUs

| Mode                     | GPUs |  DP |  PP |  EP | Total batch size | Tokens/sec | Step time (s) |
| ------------------------ | ---: | --: | --: | --: | ---------------: | ---------: | ------------: |
| DP                       |    4 |   4 |   1 |   1 |               84 |    6,046.5 |         7.114 |
| PP · 1F1B · AC on · M=12 |    4 |   1 |   4 |   1 |               84 |    3,343.6 |        12.877 |
| EP                       |    4 |   1 |   1 |   4 |               84 |    5,102.8 |         8.430 |
| DP + PP                  |    4 |   2 |   2 |   1 |               84 |    3,186.8 |        13.496 |
| DP + EP                  |    4 |   2 |   1 |   2 |               84 |    5,439.6 |         7.908 |
| PP + EP                  |    4 |   1 |   2 |   2 |               84 |    3,133.3 |        13.770 |

#### 8 GPUs

| Mode         | GPUs |  DP |  PP |  EP | Total batch size | Tokens/sec | Step time (s) |
| ------------ | ---: | --: | --: | --: | ---------------: | ---------: | ------------: |
| DP           |    8 |   8 |   1 |   1 |               40 |    3,674.2 |         5.574 |
| PP           |    8 |   1 |   8 |   1 |               40 |    2,228.2 |         9.191 |
| EP           |    8 |   1 |   1 |   8 |               40 |    3,203.4 |         6.393 |
| DP + PP      |    8 |   2 |   4 |   1 |               40 |    1,728.0 |        11.852 |
| DP + EP      |    8 |   4 |   1 |   2 |               40 |    3,546.9 |         5.774 |
| PP + EP      |    8 |   1 |   4 |   2 |               40 |    1,606.7 |        12.748 |
| DP + PP + EP |    8 |   2 |   2 |   2 |               40 |    1,578.4 |        12.976 |

The 2- and 4-GPU means cover steps 50–199. The 8-GPU means cover steps 50–99. The 8-GPU experiments used RTX 3060 GPUs and 16 experts; the 2- and 4-GPU experiments used RTX 3090 GPUs and 20 experts.
