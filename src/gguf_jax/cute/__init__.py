"""Optional CuTe DSL dequantization kernels, loaded through cutejax.

Requires the ``cute`` dependency group (nvidia-cutlass-dsl, jax-tvm-ffi,
cutedsl-jax). Call :func:`register` to replace the pure-JAX kernels with the
cute implementations for the qtypes that have one.
"""
from .iq4_xs import matmul_iq4_xs
from .lowbit import LOWBIT_TYPES, matmul_lowbit
from .q4_k import dequantize_q4_k, matmul_q4_k, register
from .q4_k_gemm import gemm_q4_k
from .q5_k import matmul_q5_k
from .q6_k import matmul_q6_k

__all__ = [
    "LOWBIT_TYPES",
    "dequantize_q4_k",
    "gemm_q4_k",
    "matmul_iq4_xs",
    "matmul_lowbit",
    "matmul_q4_k",
    "matmul_q5_k",
    "matmul_q6_k",
    "register",
]
