"""Fused Q6_K dequant-matmul (warp-GEMV) as a CuTe DSL kernel, via cutejax.

Layout of one 210-byte Q6_K superblock (QK_K = 256 elements):

    bytes    0..127   ql      low 4 bits of each quant, 2 per byte
    bytes  128..191   qh      high 2 bits of each quant, 4 per byte
    bytes  192..207   scales  16 signed int8 sub-scales, one per 16 elements
    bytes  208..209   d       (f16) super-scale

Element e decodes as

    out[e] = (d * scales[e // 16]) * (q[e] - 32)

with float32 roundings in exactly that association, matching the numpy
reference in gguf.quants (and hence the pure-JAX kernel) bitwise.

The nibble/bit interleave follows the reference reshape order: for chunk
c = e // 32 and lane w = e % 32,

    ql byte  = 64 * (c // 4) + 32 * (c % 2) + w,   nibble shift 4 * ((c // 2) % 2)
    qh byte  = 128 + 32 * (c // 4) + w,            bit shift    2 * (c % 4)
    scale    = 2 * c + w // 16

Unlike Q4_K, the 210-byte superblock is not a multiple of 4, so there is no
uint32 row view: the scale bytes are read straight out of the uint8 tensor
(two distinct bytes per warp per chunk, so the loads stay coalesced). The
f16 ``d`` still comes from a float16 recast view — 210 is even — so the
hardware f16->f32 convert is used instead of manual bit widening.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

QK_K = 256
_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.Q6_K][1]  # 210
_D_HALF = (_BLOCK_BYTES - 2) // 2  # index of d in the float16 view of a block

_SB_PER_CTA = 8    # warps (== output rows) per CTA
_CTA = 256

# GEMV-style mapping for small M: one warp per output row n of W. Lane w walks
# the row's superblocks handling elements {32c + w}, dequantizes into registers
# (rounded through bf16, matching dequantize-then-matmul), multiplies with
# x[m, e] and accumulates in float32; a butterfly-shuffle reduction folds the
# 32 lanes. M is compile-time static (one specialization per batch size).
#
# There is no tensor-core Q6_K GEMM yet (q4_k_gemm.py has the Q4_K one), so
# above this cap matmul_q6_k falls back to dequantize-then-matmul. Measured on
# an RTX 5070 Ti (4096x14336 weight), the GEMV beats that fallback out to M=8
# (248us vs 349us) and loses by M=12 — a wider range than Q4_K's M<=2, where
# the tensor-core GEMM takes over instead.
_GEMV_MAX_M = 8


@cute.kernel
def _q6_k_matmul_kernel(gU: cute.Tensor, gH: cute.Tensor,
                        gX: cute.Tensor, gO: cute.Tensor,
                        shape_u: cute.Shape, shape_x: cute.Shape,
                        M: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    n = bid * _SB_PER_CTA + (tid >> 5)  # output row (row of W)
    lane = tid & 31

    if n < shape_u[0]:
        acc = cute.make_rmem_tensor(M, cutlass.Float32)
        for m in cutlass.range_constexpr(M):
            acc[m] = cutlass.Float32(0.0)

        n_sb = shape_x[1] >> 8
        for s in cutlass.range(n_sb):
            base = s * _BLOCK_BYTES
            d = cutlass.Float32(gH[n, s * (_BLOCK_BYTES // 2) + _D_HALF])
            for c in cutlass.range_constexpr(8):
                # scales are signed int8; lanes 0..15 and 16..31 differ
                sb = cutlass.Int32(gU[n, base + 192 + 2 * c + (lane >> 4)])
                sc = sb - ((sb & 0x80) << 1)
                dl = d * cutlass.Float32(sc)
                lo = cutlass.Int32(gU[n, base + 64 * (c >> 2) + 32 * (c & 1) + lane])
                hi = cutlass.Int32(gU[n, base + 128 + 32 * (c >> 2) + lane])
                q = (((lo >> (4 * ((c >> 1) & 1))) & 0x0F)
                     | (((hi >> (2 * (c & 3))) & 0x03) << 4)) - 32
                wgt = cutlass.Float32(cutlass.BFloat16(dl * cutlass.Float32(q)))
                e = s * QK_K + 32 * c + lane
                for m in cutlass.range_constexpr(M):
                    acc[m] = acc[m] + wgt * cutlass.Float32(gX[m, e])

        for m in cutlass.range_constexpr(M):
            v = acc[m]
            v = v + cute.arch.shuffle_sync_bfly(v, 16)
            v = v + cute.arch.shuffle_sync_bfly(v, 8)
            v = v + cute.arch.shuffle_sync_bfly(v, 4)
            v = v + cute.arch.shuffle_sync_bfly(v, 2)
            v = v + cute.arch.shuffle_sync_bfly(v, 1)
            if lane == 0:
                gO[m, n] = gO.element_type(v)


@cute.jit
def _q6_k_matmul_launch(stream, gU: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor):
    n_rows = gU.shape[0]
    row_halves = gU.shape[1] // 2
    hptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Float16)
    gH = cute.make_tensor(hptr, cute.make_layout((n_rows, row_halves), stride=(row_halves, 1)))
    n_cta = (n_rows + _SB_PER_CTA - 1) // _SB_PER_CTA
    _q6_k_matmul_kernel(gU, gH, gX, gO, gU.shape, gX.shape, gX.shape[0]).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def matmul_q6_k(x: jax.Array, w) -> jax.Array:
    """``x @ w.T`` with ``w`` a Q6_K :class:`~gguf_jax.QuantizedArray` (N, K).

    ``x`` is bfloat16 ``(..., K)``; the result is bfloat16 ``(..., N)``.
    Weights are dequantized in registers (rounded through bfloat16, so values
    match ``x @ w.dequantize(bfloat16).T`` up to f32 summation order) and
    never materialized. Dispatches on the flattened batch size M: warp-GEMV
    for decode shapes, dequantize-then-matmul above them.
    """
    from gguf_jax.array import QuantizedArray

    assert isinstance(w, QuantizedArray) and w.qtype == GGMLQuantizationType.Q6_K
    assert len(w.shape) == 2, "w must be a 2D weight"
    n_rows, k_dim = w.shape
    assert x.shape[-1] == k_dim, f"contraction mismatch: {x.shape[-1]} != {k_dim}"
    assert x.dtype == jnp.bfloat16, "x must be bfloat16"

    xm = x.reshape(-1, k_dim)
    m = xm.shape[0]
    if m <= _GEMV_MAX_M:
        out = cutejax.call(
            _q6_k_matmul_launch,
            jax.ShapeDtypeStruct((m, n_rows), jnp.bfloat16),
            w.data.reshape(n_rows, -1), xm,
            in_specs=[None, cutejax.ArraySpec(static_dims=(0,))],
            out_specs=cutejax.ArraySpec(static_dims=(0,)),
        )
    else:
        out = xm @ w.dequantize(jnp.bfloat16).T
    return out.reshape(*x.shape[:-1], n_rows)
