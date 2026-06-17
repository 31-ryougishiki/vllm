# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Setup script to compile vllm core modules to binary (.pyd/.so).

This compiles the split-attention-moe custom modules to protect core
algorithms for distributed topology management and Qwen3MoE model execution.

Usage:
    # Build all core modules in-place (from vllm/ root)
    python setup_cython.py build_ext --inplace

    # Clean build artifacts
    python setup_cython.py clean --all

Requirements:
    pip install cython setuptools
"""

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

# ---------------------------------------------------------------------------
# Platform-specific compiler settings
# ---------------------------------------------------------------------------
is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")

extra_compile_args = []
if is_windows:
    extra_compile_args = ["/EHsc", "/O2", "/MD"]
elif is_linux:
    extra_compile_args = ["-O3", "-march=native", "-fvisibility=hidden"]

# ---------------------------------------------------------------------------
# Extension definitions
# ---------------------------------------------------------------------------
extensions = [
    # Split-attn-moe: distributed group management (1938 lines, ~100 custom)
    Extension(
        "vllm.distributed.parallel_state_core",
        sources=["vllm/distributed/parallel_state_core.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    # Split-attn-moe: Qwen3MoE model with cross-group P2P (1061 lines, ~150 custom)
    Extension(
        "vllm.model_executor.models.qwen3_moe_core",
        sources=["vllm/model_executor/models/qwen3_moe_core.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

# ---------------------------------------------------------------------------
# Cython compiler directives
# ---------------------------------------------------------------------------
cython_directives = {
    "language_level": "3",          # Python 3 semantics
    "embedsignature": True,         # Preserve function signatures for debugging
    "boundscheck": False,           # Skip bounds checking (performance)
    "wraparound": False,            # Skip negative index wrapping
}

compiled_modules = cythonize(
    extensions,
    compiler_directives=cython_directives,
    nthreads=4,                     # Parallel compilation
)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
setup(
    name="vllm_core",
    version="1.0.0",
    description="Cython-compiled core modules for vllm (split-attn-moe)",
    ext_modules=compiled_modules,
    zip_safe=False,
)
