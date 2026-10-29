# Exp4: model_q on AMD

Keep the full 38.96B-parameter, 2.72B-active model, BF16 and CUDA/HIP MoE backend.
Use EP=4 (GPUs 4-7 initially), DP=1, PP=1 and 4096 input/target tokens per sample.
The cached dataset must have shape [N,4097]; the last token supplies the shifted target.

Prepare it on CPU with `uv --no-config run --no-project --active python scripts/cache_dataset.py --seq-len 4096 --num-sequences 65536 --output data/tinystories_seq4096.pt` using the dedicated ROCm environment without syncing CUDA extras.

Run `experiments/exp4-20261029/run.py` with batch sizes 1,2,4,8,16,32 until a complete optimizer step fails or time runs short. Use six steps per successful pilot and compare later steps, not initialization. Refine between the last success and first failure if time permits. The largest tested batch is not a proven maximum unless an upper failure boundary was measured.

Use `--profile` for a separate five-step-or-longer run: two wait steps, one warmup, one recorded step. Continue an ordinary run at the selected batch for the remaining authorized rental. The launcher saves the exact source commit, resolved arguments and configuration next to metrics. Do not count profiled timings as ordinary throughput.