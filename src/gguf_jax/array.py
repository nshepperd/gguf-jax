"""QuantizedArray: a JAX pytree holding a GGUF-quantized tensor."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

from . import quants

__all__ = ["QuantizedArray", "quantize"]

# Types kept as native (non-uint8) arrays: dequantize is a cast (or identity).
_NATIVE_TYPES = {
    GGMLQuantizationType.I8: jnp.int8,
    GGMLQuantizationType.I16: jnp.int16,
    GGMLQuantizationType.I32: jnp.int32,
    GGMLQuantizationType.I64: jnp.int64,
    GGMLQuantizationType.F64: jnp.float64,
}

_INTEGER_TYPES = {
    GGMLQuantizationType.I8,
    GGMLQuantizationType.I16,
    GGMLQuantizationType.I32,
    GGMLQuantizationType.I64,
}


@jax.tree_util.register_dataclass
@dataclass
class QuantizedArray:
    """A quantized tensor: raw GGUF block data plus the metadata to decode it.

    ``data`` is the quantized payload as stored in the GGUF file: a uint8
    array of shape ``(*shape[:-1], row_bytes)`` for quantized/float qtypes
    (F16/F32/BF16 included), or a native integer/float array for the plain
    I8/I16/I32/I64/F64 types.

    This is a pytree: it can be passed through ``jax.jit``, ``tree_map``,
    etc.; ``qtype``, ``shape`` and ``dtype`` are static metadata.
    """

    data: jax.Array
    qtype: GGMLQuantizationType = field(metadata=dict(static=True))
    shape: tuple[int, ...] = field(metadata=dict(static=True))
    dtype: Any = field(default=jnp.bfloat16, metadata=dict(static=True))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.data.size * self.data.dtype.itemsize

    def astype(self, dtype) -> QuantizedArray:
        """Return a copy whose default dequantization dtype is ``dtype``."""
        return replace(self, dtype=jnp.dtype(dtype))

    def dequantize(self, dtype=None) -> jax.Array:
        """Decode to a dense array of ``dtype`` (default: ``self.dtype``).

        Decoding computes in float32, bitwise identical to
        ``gguf.quants.dequantize``, then casts. Integer qtypes are returned
        as-is (``dtype`` is ignored).
        """
        dtype = self.dtype if dtype is None else dtype
        if self.qtype in _NATIVE_TYPES:
            out = self.data.reshape(self.shape)
            if self.qtype in _INTEGER_TYPES:
                return out
            return out.astype(dtype)
        out = quants.dequantize(self.data, self.qtype).reshape(self.shape)
        return out.astype(dtype)

    def __repr__(self) -> str:
        return (f"QuantizedArray({self.qtype.name}, shape={self.shape}, "
                f"dtype={jnp.dtype(self.dtype).name})")

    @classmethod
    def from_bytes(cls, data, qtype: GGMLQuantizationType,
                   shape: tuple[int, ...] | None = None,
                   dtype=jnp.bfloat16) -> QuantizedArray:
        """Wrap raw quantized bytes (numpy or JAX uint8, byte-shaped) as a QuantizedArray."""
        data = jnp.asarray(data)
        if qtype in _NATIVE_TYPES:
            shape = tuple(data.shape) if shape is None else tuple(shape)
            return cls(data=data.reshape(shape), qtype=qtype, shape=shape, dtype=jnp.dtype(dtype))
        if data.dtype != jnp.uint8:
            raise ValueError(f"expected uint8 data for {qtype.name}, got {data.dtype}")
        if shape is None:
            shape = quants._ref.quant_shape_from_byte_shape(data.shape, qtype)
        shape = tuple(int(s) for s in shape)
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        expected = shape[-1] // block_size * type_size
        data = data.reshape(*shape[:-1], expected)
        return cls(data=data, qtype=qtype, shape=shape, dtype=jnp.dtype(dtype))


def quantize(array, qtype: GGMLQuantizationType, dtype=jnp.bfloat16) -> QuantizedArray:
    """Quantize an array (on the host, via the gguf reference implementation).

    Only the qtypes for which gguf-py implements quantization are supported;
    this is mainly a convenience for tests and tools.
    """
    import gguf

    np_arr = np.asarray(array, dtype=np.float32)
    qdata = gguf.quants.quantize(np_arr, qtype)
    if qdata.dtype != np.uint8:
        qdata = np.ascontiguousarray(qdata).view(np.uint8)
    return QuantizedArray.from_bytes(qdata, qtype, shape=np_arr.shape, dtype=dtype)
