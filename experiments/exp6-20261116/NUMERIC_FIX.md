# Separate FP16 numeric repair

This branch (`fix/exp6-fp16-optimizer`) follows the one-commit experiment config
branch and keeps numeric changes separate. It is not merged into master.

Parameter-precision AdamW produces NaN/Inf on FP16 zero/small gradients: its
1e-8 epsilon and squared-gradient state underflow. Changing only epsilon does
not preserve small second moments. The repair keeps master weights and Adam
moments in FP32, then casts updated weights to the requested model dtype. The
FP32 router is unchanged. Optimizer saves include master weights for resume.
Gradient norm squares and LayerNorm/RMSNorm reductions use FP32 to avoid half
overflow/underflow; outputs retain the model dtype.

CPU tests cover zero/small-gradient updates against FP32 AdamW, optimizer resume,
zero/large normalization inputs and gradients, large gradient norms, and three
tiny Torch-MoE optimizer steps for each requested precision. They do not certify
the native GPU kernels, expert-parallel collectives, or long-run convergence.

```bash
uv --no-config run --no-project --active python -m pytest -q tests/test_precision_numerics.py
```
