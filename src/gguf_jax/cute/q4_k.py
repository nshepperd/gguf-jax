"""Q4_K dequantization as a CuTe DSL kernel (via cutejax).

Layout of one 144-byte Q4_K superblock (QK_K = 256 elements):

    bytes   0..1    d      (f16) super-scale for the 6-bit sub-scales
    bytes   2..3    dmin   (f16) super-scale for the 6-bit sub-mins
    bytes   4..15   scales 8 x (6-bit scale, 6-bit min), packed
    bytes  16..143  qs     128 bytes of 4-bit quants

Element e of a superblock decodes as

    out[e] = (d * sc[e//32]) * q[e] - (dmin * mn[e//32])

with float32 roundings in exactly that association, matching the numpy
reference in gguf.quants (and hence the pure-JAX kernel) bitwise.

Grid mapping: one CTA per superblock, one thread per element. The f16 pair
(d, dmin) is read through a float16 bitcast view of the same buffer, so the
hardware f16->f32 convert is used instead of manual bit widening.
"""
from __future__ import annotations

import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute

import cutejax
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

QK_K = 256
_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][1]  # 144


_SB_PER_CTA = 8    # superblocks per CTA; 32 lanes per superblock
_CTA = 256


@cute.kernel
def _q4_k_kernel(gU: cute.Tensor, gW: cute.Tensor, gH: cute.Tensor,
                 gO: cute.Tensor, shape: cute.Shape):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    s = bid * _SB_PER_CTA + (tid >> 5)  # superblock index
    w = tid & 31                        # lane within the superblock

    if s < shape[0]:
        d = cutlass.Float32(gH[s, 0])
        dmin = cutlass.Float32(gH[s, 1])

        # scale words: bytes 4..15 = uint32 words 1..3 of the superblock
        w_d = cutlass.Int32(gW[s, 1])   # 4 "d row" bytes
        w_m = cutlass.Int32(gW[s, 2])   # 4 "m row" bytes
        w_md = cutlass.Int32(gW[s, 3])  # 4 "m_d row" bytes

        # Lane w handles elements {32k + w : k in 0..7}: every store
        # instruction below is warp-contiguous, the sub-block index k is a
        # compile-time constant, and the qs byte loads are warp-coalesced.
        for k in cutlass.range_constexpr(8):
            j = k & 3
            bd = (w_d >> (8 * j)) & 0xFF
            bm = (w_m >> (8 * j)) & 0xFF
            bmd = (w_md >> (8 * j)) & 0xFF
            if cutlass.const_expr(k < 4):
                sc = bd & 63
                mn = bm & 63
            else:
                sc = (bmd & 0x0F) | ((bd >> 6) << 4)
                mn = (bmd >> 4) | ((bm >> 6) << 4)
            dl = d * cutlass.Float32(sc)
            dm = dmin * cutlass.Float32(mn)
            qbyte = cutlass.Int32(gU[s, 16 + 32 * (k >> 1) + w])
            q = (qbyte >> (4 * (k & 1))) & 0x0F
            gO[s, 32 * k + w] = gO.element_type(dl * cutlass.Float32(q) - dm)


@cute.jit
def _q4_k_launch(stream, gU: cute.Tensor, gO: cute.Tensor):
    nb = gU.shape[0]
    # Same bytes, wider views: uint32 words for scales/quants, f16 for (d, dmin).
    wptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Int32)
    gW = cute.make_tensor(
        wptr, cute.make_layout((nb, _BLOCK_BYTES // 4), stride=(_BLOCK_BYTES // 4, 1)))
    hptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Float16)
    gH = cute.make_tensor(
        hptr, cute.make_layout((nb, _BLOCK_BYTES // 2), stride=(_BLOCK_BYTES // 2, 1)))
    n_cta = (nb + _SB_PER_CTA - 1) // _SB_PER_CTA
    _q4_k_kernel(gU, gW, gH, gO, gW.shape).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def dequantize_q4_k(data: jax.Array, dtype=jnp.float32) -> jax.Array:
    """Dequantize Q4_K bytes ``(..., row_bytes)`` to ``dtype`` ``(..., n)``."""
    assert data.dtype == jnp.uint8 and data.shape[-1] % _BLOCK_BYTES == 0
    blocks = data.reshape(-1, _BLOCK_BYTES)
    nb = blocks.shape[0]
    out = cutejax.call(
        _q4_k_launch, jax.ShapeDtypeStruct((nb, QK_K), dtype), blocks)
    out_shape = (*data.shape[:-1], data.shape[-1] // _BLOCK_BYTES * QK_K)
    return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# fused dequant-matmul: y = x @ W^T with W kept quantized in HBM
#
# GEMV-style mapping for small M (LLM decode/small-batch prefill): one warp
# per output row n of W. Lane w walks the row's superblocks handling elements
# {32k + w}, dequantizes into registers (rounded through bf16 so results
# match dequantize-then-matmul semantics), multiplies with x[m, e] and
# accumulates in float32; a butterfly-shuffle reduction folds the 32 lanes.
# M is compile-time static (one kernel specialization per batch size).

# Measured crossovers on RTX 5070 Ti (4096x14336 weight): the warp-GEMV
# wins at M <= 2 (73us vs the tensor-core GEMM's flat 91us); the
# tensor-core GEMM (q4_k_gemm.py) wins from there until ~M=160, where
# dequantize-then-matmul takes over because it reads the quantized weight
# once instead of once per 32-row M-tile.
_GEMV_MAX_M = 2
_GEMM_MAX_M = 128


@cute.kernel
def _q4_k_matmul_kernel(gU: cute.Tensor, gW: cute.Tensor, gH: cute.Tensor,
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
            d = cutlass.Float32(gH[n, s * (_BLOCK_BYTES // 2)])
            dmin = cutlass.Float32(gH[n, s * (_BLOCK_BYTES // 2) + 1])
            w_d = cutlass.Int32(gW[n, s * (_BLOCK_BYTES // 4) + 1])
            w_m = cutlass.Int32(gW[n, s * (_BLOCK_BYTES // 4) + 2])
            w_md = cutlass.Int32(gW[n, s * (_BLOCK_BYTES // 4) + 3])
            for k in cutlass.range_constexpr(8):
                j = k & 3
                bd = (w_d >> (8 * j)) & 0xFF
                bm = (w_m >> (8 * j)) & 0xFF
                bmd = (w_md >> (8 * j)) & 0xFF
                if cutlass.const_expr(k < 4):
                    sc = bd & 63
                    mn = bm & 63
                else:
                    sc = (bmd & 0x0F) | ((bd >> 6) << 4)
                    mn = (bmd >> 4) | ((bm >> 6) << 4)
                dl = d * cutlass.Float32(sc)
                dm = dmin * cutlass.Float32(mn)
                qbyte = cutlass.Int32(gU[n, s * _BLOCK_BYTES + 16 + 32 * (k >> 1) + lane])
                q = (qbyte >> (4 * (k & 1))) & 0x0F
                wgt = cutlass.Float32(cutlass.BFloat16(dl * cutlass.Float32(q) - dm))
                e = s * QK_K + 32 * k + lane
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
def _q4_k_matmul_launch(stream, gU: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor):
    n_rows = gU.shape[0]
    row_words = gU.shape[1] // 4
    row_halves = gU.shape[1] // 2
    wptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Int32)
    gW = cute.make_tensor(wptr, cute.make_layout((n_rows, row_words), stride=(row_words, 1)))
    hptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Float16)
    gH = cute.make_tensor(hptr, cute.make_layout((n_rows, row_halves), stride=(row_halves, 1)))
    n_cta = (n_rows + _SB_PER_CTA - 1) // _SB_PER_CTA
    _q4_k_matmul_kernel(gU, gW, gH, gX, gO, gU.shape, gX.shape, gX.shape[0]).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def matmul_q4_k(x: jax.Array, w, *, force_fused: bool = False) -> jax.Array:
    """``x @ w.T`` with ``w`` a Q4_K :class:`~gguf_jax.QuantizedArray` (N, K).

    ``x`` is bfloat16 ``(..., K)``; the result is bfloat16 ``(..., N)``.
    Weights are dequantized in registers (rounded through bfloat16, so values
    match ``x @ w.dequantize(bfloat16).T`` up to f32 summation order) and
    never materialized. Dispatches on the flattened batch size M: warp-GEMV
    for decode shapes, the tensor-core GEMM for small-batch/prefill, and
    dequantize-then-matmul for large M where re-reading the quantized weight
    per M-tile stops paying.

    ``force_fused=True`` forbids the dequantize fallback: the tensor-core
    GEMM handles every M, so the dense bf16 weight (2 * N * K bytes) is never
    materialized — the point when the weight is large. Costs speed at large M
    (measured ~1.5x slower than the fallback at M=512 on an RTX 5070 Ti,
    because each 32-row M-tile re-reads the quantized weight). Requires
    ``N % 64 == 0`` when M > 2.
    """
    from gguf_jax.array import QuantizedArray

    from .q4_k_gemm import _BN as _GEMM_BN
    from .q4_k_gemm import gemm_q4_k

    assert isinstance(w, QuantizedArray) and w.qtype == GGMLQuantizationType.Q4_K
    assert len(w.shape) == 2, "w must be a 2D weight"
    n_rows, k_dim = w.shape
    assert x.shape[-1] == k_dim, f"contraction mismatch: {x.shape[-1]} != {k_dim}"
    assert x.dtype == jnp.bfloat16, "x must be bfloat16"

    xm = x.reshape(-1, k_dim)
    m = xm.shape[0]
    if m <= _GEMV_MAX_M:
        out = cutejax.call(
            _q4_k_matmul_launch,
            jax.ShapeDtypeStruct((m, n_rows), jnp.bfloat16),
            w.data.reshape(n_rows, -1), xm,
            in_specs=[None, cutejax.ArraySpec(static_dims=(0,))],
            out_specs=cutejax.ArraySpec(static_dims=(0,)),
        )
    elif (m <= _GEMM_MAX_M or force_fused) and n_rows % _GEMM_BN == 0:
        out = gemm_q4_k(xm, w)
    elif force_fused:
        raise ValueError(
            f"force_fused matmul with M={m} > {_GEMV_MAX_M} needs the tensor-core "
            f"kernel, which requires N % {_GEMM_BN} == 0 (got N={n_rows})")
    else:
        out = xm @ w.dequantize(jnp.bfloat16).T
    return out.reshape(*x.shape[:-1], n_rows)


def register() -> None:
    """Replace the pure-JAX Q4_K kernel with the cute kernel.

    The kernel computes in float32 and writes the requested output dtype
    directly (single hardware round), so no float32 intermediate is
    materialized for bfloat16 dequantization.
    """
    from gguf_jax import quants

    quants.register_dequant(
        GGMLQuantizationType.Q4_K, dequantize_q4_k, override=True)
