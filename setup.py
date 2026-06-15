from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="random_ext",
    ext_modules=[
        CUDAExtension(
            name="random_ext",
            sources=[
                "csrc/random.cpp",
                "csrc/bindings.cpp",
                "csrc/kernels/copy_kernel.cu",
                "csrc/kernels/pack_tokens.cu",
            ],
            include_dirs=["csrc", "csrc/runtime"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
