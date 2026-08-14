"""Optional CuTe DSL dequantization kernels, loaded through cutejax.

Requires the ``cute`` dependency group (nvidia-cutlass-dsl, jax-tvm-ffi,
cutedsl-jax). Call :func:`register` to replace the pure-JAX kernels with the
cute implementations for the qtypes that have one.
"""
from .q4_k import dequantize_q4_k, register

__all__ = ["dequantize_q4_k", "register"]
