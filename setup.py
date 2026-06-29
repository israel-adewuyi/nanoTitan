from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="nanotitan",
    ext_modules=[
        CUDAExtension(
            name="random_ext",
            sources=[
                "csrc/bindings.cpp",
                "csrc/random.cpp",
                "csrc/kernels/gemm.cu",
                "csrc/kernels/copy_kernel.cu",
                "csrc/kernels/pack_tokens.cu",
                "csrc/kernels/count_experts.cu",
                "csrc/kernels/combine_kernels.cu",
                "csrc/kernels/backward_combine_kernel.cu",
                "csrc/kernels/backward_pack_kernel.cu",
            ],
            include_dirs=[
                str(ROOT / "csrc"),
                str(ROOT / "csrc/runtime"),
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
