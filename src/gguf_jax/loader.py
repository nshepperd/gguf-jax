"""Load GGUF files into JAX-ready QuantizedArrays."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

from .array import _NATIVE_TYPES, QuantizedArray

__all__ = ["GGUFFile", "load_gguf"]


@dataclass
class GGUFFile:
    """Contents of a GGUF file: metadata key-values and named tensors.

    ``tensors`` is a plain dict of :class:`QuantizedArray` (itself a pytree),
    so it composes with ``jax.tree_util`` / ``jax.jit`` as usual.
    """

    metadata: dict[str, Any]
    tensors: dict[str, QuantizedArray]

    def __repr__(self) -> str:
        return f"GGUFFile({len(self.metadata)} metadata keys, {len(self.tensors)} tensors)"


def load_gguf(
    path: str,
    dtype=jnp.bfloat16,
    tensor_filter: Callable[[str], bool] | None = None,
) -> GGUFFile:
    """Load a GGUF file.

    Tensor payloads are uploaded to the default JAX device as raw quantized
    bytes (uint8), without dequantizing; call ``.dequantize()`` on a tensor
    (typically inside your jitted forward pass) to decode it.

    Args:
        path: path to the .gguf file.
        dtype: default dtype that ``dequantize()`` will produce.
        tensor_filter: optional predicate on tensor names; names for which it
            returns False are skipped (useful to load a subset of a model).
    """
    reader = GGUFReader(path, "r")

    metadata: dict[str, Any] = {}
    for name, fld in reader.fields.items():
        try:
            metadata[name] = fld.contents()
        # tolerate exotic field layouts rather than fail the load
        except Exception:  # noqa: BLE001
            metadata[name] = None

    tensors: dict[str, QuantizedArray] = {}
    for tensor in reader.tensors:
        if tensor_filter is not None and not tensor_filter(tensor.name):
            continue
        qtype = GGMLQuantizationType(tensor.tensor_type)
        logical_shape = tuple(reversed(tensor.shape.tolist()))
        data = tensor.data
        if qtype not in _NATIVE_TYPES:
            # normalize to byte view (F16/F32 come out of the reader as native dtypes)
            data = np.ascontiguousarray(data).view(np.uint8)
        tensors[tensor.name] = QuantizedArray.from_bytes(
            jnp.asarray(data), qtype, shape=logical_shape, dtype=dtype)

    return GGUFFile(metadata=metadata, tensors=tensors)
