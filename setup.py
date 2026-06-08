from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="random_ext",
    ext_modules=[
        CUDAExtension(
            name="benchmarks",
            sources=["csrc/random.cpp", "csrc/bindings.cpp", "csrc/kernels/copy_kernels.cu"],
            include_dirs=["csrc", "csrc/runtime"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
