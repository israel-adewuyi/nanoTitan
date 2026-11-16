# Exp6 fixed precision comparison

One shared base config: 12 layers, width512, head64 x 8, FFN4096, 16 experts,
top2, EP2 on GPUs2/3, batch/rank1, 4096 tokens, 200 updates, LR5e-4, seed42.
The router remains FP32. Only model dtype changes; run names distinguish outputs.
No training is started by CPU preparation. The user launches the runs sequentially.

From the nanoTitan repository in the prepared environment:

```bash
for dtype in bfloat16 float16 float32; do
  uv --no-config run --no-project --active python experiments/exp6-20261116/run.py --dtype "$dtype" --output-dir /mnt/shared/.research-session/exp6-20261116/nanotitan-dry --dry-run
done
```

For training, remove `--dry-run` and use a new output root, e.g.
`/mnt/shared/.research-session/exp6-20261116/nanotitan-runs`.
TensorBoard logs are under each run directory's `runs/` directory.
Model construction validates shapes and parameter counts on meta tensors;
it does not validate GPU kernels, FP16 numerics, dataset loading or EP collectives.

The current parameter-precision AdamW fails a zero/small-gradient FP16 control.
Numeric repairs belong on the separate `fix/exp6-fp16-optimizer` branch;
this config commit does not contain them. Run the fix branch's checks before FP16 training.
