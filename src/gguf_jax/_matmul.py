"""``x @ w.T`` for a quantized ``w``, picking the best available route.

There are three ways to multiply by a quantized weight and the right one
depends on the batch size, the qtype and which optional kernels are installed:

- a **fused kernel**, which keeps the weight quantized in HBM and never
  materializes it. Fastest, and the only route that does not allocate, but each
  kernel only covers a range of batch sizes;
- **dequantize-then-matmul**, which materializes the whole weight;
- the same thing **split across output rows**, when the whole weight is too
  large to materialize at once.

Callers should not have to know that, nor which kernels exist, nor what each
one's batch cap is. :func:`matmul` decides.

Getting the fallback right matters more than it looks. See
:func:`dequant_matmul` for the scheduling hazard it exists to avoid.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from gguf.constants import GGMLQuantizationType as QT

from .array import QuantizedArray

__all__ = ["dequant_matmul", "matmul"]

# How large a temporary a single dequantize may allocate before it is split
# across output rows. Sized as 4*N*K, i.e. as though the buffer were float32;
# it is not -- decoding is fused into the convert, so what reaches memory is
# bf16 at 2*N*K -- which makes this a 2x conservative budget on purpose.
DEQUANT_BLOCK_BYTES = int(
    os.environ.get("GGUF_JAX_DEQUANT_BLOCK_BYTES", 512 << 20))

# qtype -> (kernel, batch cap for a given weight). The cap belongs next to the
# kernel rather than at the call site: it is a property of how the kernel is
# written, and a caller that has to know it will get it wrong when it changes.
FusedEntry = tuple[Callable[..., jax.Array], Callable[[QuantizedArray], int]]


@functools.cache
def _fused() -> dict[QT, FusedEntry]:
    """Whichever cute kernels are importable. Empty without the cute extra."""
    try:
        from .cute import iq4_xs, q4_k, q5_k, q6_k
    except ImportError:
        return {}

    def q4_k_cap(w: QuantizedArray) -> int:
        # Q4_K is the only type with a tensor-core GEMM as well as a GEMV, and
        # the GEMM needs the output rows to tile evenly.
        from .cute.q4_k_gemm import _BN

        return max(q4_k._GEMV_MAX_M,
                   q4_k._GEMM_MAX_M if w.shape[0] % _BN == 0 else 0)

    return {
        QT.Q4_K: (q4_k.matmul_q4_k, q4_k_cap),
        QT.Q5_K: (q5_k.matmul_q5_k, lambda w: q5_k._GEMV_MAX_M),
        QT.Q6_K: (q6_k.matmul_q6_k, lambda w: q6_k._GEMV_MAX_M),
        QT.IQ4_XS: (iq4_xs.matmul_iq4_xs, lambda w: iq4_xs._GEMV_MAX_M),
    }


def fused_types() -> list[QT]:
    """The qtypes with a fused kernel in this installation."""
    return sorted(_fused().keys(), key=lambda t: t.name)


def fused_batch_limit(w: QuantizedArray) -> int:
    """Largest batch ``w`` can be multiplied at without materializing it.

    Zero when no fused kernel applies. Useful for deciding how to shape a
    call; :func:`matmul` consults it for you.
    """
    entry = _fused().get(w.qtype)
    return entry[1](w) if entry is not None else 0


def matmul(x: jax.Array, w: QuantizedArray, *, block_bytes: int | None = None,
           force_fused: bool = False) -> jax.Array:
    """``x @ w.T`` with ``w`` quantized, by whichever route fits.

    ``x`` is ``(..., K)`` and the result is ``(..., N)`` for ``w`` of shape
    ``(N, K)``. A fused kernel is used when one exists for the qtype, ``x`` is
    bfloat16, and the flattened batch is within that kernel's range; otherwise
    the weight is dequantized, in row blocks if it is too large to do at once.

    ``force_fused`` ignores the batch cap and calls the kernel anyway, which is
    for measuring where the crossover actually is -- above their range the
    kernels fall back internally to dequantizing the *whole* weight, so this
    can allocate a great deal. ``block_bytes`` overrides
    :data:`DEQUANT_BLOCK_BYTES` for this call.
    """
    if not isinstance(w, QuantizedArray):
        raise TypeError(f"expected a QuantizedArray weight, got {type(w).__name__}")
    if len(w.shape) != 2:
        raise ValueError(f"w must be 2D, got shape {w.shape}")
    if x.shape[-1] != w.shape[1]:
        raise ValueError(
            f"contraction mismatch: x has {x.shape[-1]}, w has {w.shape[1]}")

    entry = _fused().get(w.qtype)
    if entry is not None and x.dtype == jnp.bfloat16:
        batch = int(np.prod(x.shape[:-1], dtype=np.int64))
        if force_fused or batch <= entry[1](w):
            return entry[0](x, w)
    return dequant_matmul(x, w, block_bytes=block_bytes)


def dequant_matmul(x: jax.Array, w: QuantizedArray, *,
                   block_bytes: int | None = None) -> jax.Array:
    """``x @ w.T`` by materializing ``w``, without materializing all of them.

    Two scheduling hazards make this more than a one-liner, and both cost
    gigabytes when they bite.

    **Across weights.** A dequantize depends only on the weight, which is a
    parameter, so it is available from the executable's first instruction and
    XLA may hoist it arbitrarily early. It does. In a transformer prefill above
    every fused batch cap, every layer's matmul takes this path and none
    depends on any other, so the scheduler overlaps them: measured on a 36-layer
    8B model, 68 dequantized weights live at once and 3.66 GiB of scratch.
    Routing the weight bytes through an optimization barrier together with ``x``
    gives the dequantize an artificial dependency on the activation arriving at
    that layer, which chains them into the model's own sequential order and
    takes the same case to 193 MiB. It is semantically a no-op.

    **Within a weight.** Row blocks are independent too, so an unrolled loop
    over them has the same problem in miniature. ``fori_loop`` plus
    ``dynamic_update_slice`` makes the sequencing explicit and writes each block
    into the output in place. Stacking the blocks and transposing instead would
    be shorter, but it materializes the result twice and the transpose is large
    enough that XLA fails to find a config for it.
    """
    out_rows, in_features = w.shape
    blocks = _block_count(out_rows, in_features,
                          DEQUANT_BLOCK_BYTES if block_bytes is None else block_bytes)

    data, x = jax.lax.optimization_barrier((w.data, x))
    w = QuantizedArray(data=data, qtype=w.qtype, shape=w.shape, dtype=w.dtype)

    if blocks == 1:
        return x @ w.dequantize().T

    rows = out_rows // blocks
    flat = x.reshape(-1, in_features)
    data = w.data.reshape(blocks, rows, -1)
    out = jnp.zeros((flat.shape[0], out_rows), dtype=w.dtype)

    def body(i, out):
        sub = QuantizedArray(
            data=jax.lax.dynamic_index_in_dim(data, i, keepdims=False),
            qtype=w.qtype, shape=(rows, in_features), dtype=w.dtype,
        )
        return jax.lax.dynamic_update_slice(
            out, flat @ sub.dequantize().T, (0, i * rows))

    out = jax.lax.fori_loop(0, blocks, body, out)
    return out.reshape(*x.shape[:-1], out_rows)


def _block_count(out_rows: int, in_features: int, budget: int) -> int:
    """How many row blocks keep one dequantize under ``budget``.

    Constrained to divisors of ``out_rows`` so the blocks are uniform, which is
    what lets the split be a loop over a reshaped array.
    """
    needed = 4 * out_rows * in_features
    if needed <= budget:
        return 1
    target = -(-needed // budget)
    for n in range(target, out_rows + 1):
        if out_rows % n == 0:
            return n
    return out_rows
