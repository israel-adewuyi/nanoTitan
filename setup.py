from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="random_ext",
    ext_modules=[
        CppExtension(
            name="random_ext",
            sources=[
                "csrc/random.cpp",
                "csrc/bindings.cpp",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
