from __future__ import annotations

from setuptools import Extension, setup
import sys

import numpy

extra_compile_args = ["-O3"]
if sys.platform == "darwin":
    extra_compile_args += ["-std=c99"]

cpu_core_ext = Extension(
    "pychrome_native._cpu_core",
    sources=["pychrome_native/cpu_core.c"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    name="pychrome-native",
    version="0.1.0",
    description="Native CPU/GPU acceleration backends for Py-Chrome video",
    packages=["pychrome_native"],
    ext_modules=[cpu_core_ext],
)
