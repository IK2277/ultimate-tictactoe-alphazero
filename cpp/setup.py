from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# C++17が必要
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2"]
else:
    extra_compile_args = ["-std=c++17", "-O3"]

ext_modules = [
    Pybind11Extension(
        "uttt_cpp",
        sources=[
            "uttt_game.cpp",
            "uttt_mcts.cpp",
            "python_bindings.cpp",
        ],
        include_dirs=["."],
        extra_compile_args=extra_compile_args,
        cxx_std=17,
    ),
]

setup(
    name="uttt_cpp",
    version="0.1.0",
    author="Your Name",
    description="C++ implementation of Ultimate Tic-Tac-Toe",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
