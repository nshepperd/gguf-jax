"""gguf-jax: load GGUF-quantized models in JAX and dequantize on the fly.

Quantized tensors are held as raw uint8 block data in :class:`QuantizedArray`
pytrees and decoded by pure-JAX kernels whose float32 output is bitwise
identical to the ``gguf.quants`` reference implementation from llama.cpp.
"""

from gguf.constants import GGMLQuantizationType

from .array import QuantizedArray, quantize
from .loader import GGUFFile, load_gguf
from ._matmul import dequant_matmul, fused_batch_limit, fused_types, matmul
from .quants import dequantize, dequantize_blocks, register_dequant, supported_types

__all__ = [
    "GGMLQuantizationType",
    "GGUFFile",
    "QuantizedArray",
    "dequant_matmul",
    "dequantize",
    "dequantize_blocks",
    "fused_batch_limit",
    "fused_types",
    "load_gguf",
    "matmul",
    "quantize",
    "register_dequant",
    "supported_types",
]
