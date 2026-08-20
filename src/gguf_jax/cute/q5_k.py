"""Fused Q5_K dequant-matmul (warp-GEMV) as a CuTe DSL kernel, via cutejax.

Layout of one 176-byte Q5_K superblock (QK_K = 256 elements):

    bytes   0..1    d      (f16) super-scale for the 6-bit sub-scales
    bytes   2..3    dmin   (f16) super-scale for the 6-bit sub-mins
    bytes   4..15   scales 8 x (6-bit scale, 6-bit min), packed
    bytes  16..47   qh     the 5th bit of each quant, 8 per byte
    bytes  48..175  qs     low 4 bits of each quant, 2 per byte

Element e decodes as

    out[e] = (d * sc[e//32]) * q[e] - (dmin * mn[e//32])

with float32 roundings in exactly that association, matching the numpy
reference in gguf.quants (and hence the pure-JAX kernel) bitwise. Q5_K is
Q4_K plus a fifth bit: for sub-block k = e // 32 and position t = e % 32,

    ql = (qs[32 * (k // 2) + t] >> (4 * (k % 2))) & 0x0F
    qh = (qh[t] >> k) & 1
    q  = ql | (qh << 4)

and the (scale, min) packing is byte-for-byte Q4_K's, so ``_get_scale_min``
carries over unchanged.

176 is a multiple of 4, so the whole superblock is addressable as uint32
words: word 0 is (d, dmin), words 1..3 the packed scales, words 4..11 qh,
words 12..43 qs. The kernel reads it that way throughout.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

QK_K = 256
_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.Q5_K][1]  # 176
_BLOCK_WORDS = _BLOCK_BYTES // 4                               # 44
_BLOCK_HALVES = _BLOCK_BYTES // 2                              # 88
_QH_WORD0 = 4                                                  # bytes 16..47
_QS_WORD0 = 12                                                 # bytes 48..175

_SB_PER_CTA = 8    # warps (== output rows) per CTA
_CTA = 256

# GEMV-style mapping for small M: one warp per output row n of W, using the
# "four consecutive bytes per lane" layout that Q4_K's v2 kernel settled on
# (see q4_k.py's _q4_k_matmul_kernel_v2 for the derivation). Lane w owns qs
# bytes 4w..4w+3, i.e. one uint32 load covering the whole 128-byte qs region
# per warp, exactly two of the eight sub-blocks, and four consecutive x
# elements per half.
#
# The fifth bits fall out of the same mapping for free: lane w's elements sit
# at positions t = 4*(w%8) + j within their sub-block, so the qh bits it needs
# live in qh bytes 4*(w%8)..+3 — one more uint32 load, again warp-coalesced
# (eight lanes share each word, four such groups covering all 32 bytes).
#
# There is no tensor-core Q5_K GEMM yet, so above this cap matmul_q5_k falls
# back to dequantize-then-matmul. Measured on an RTX 5070 Ti (4096x14336
# weight): the GEMV wins from M=1 (73us vs 386us, 5.3x) out to M=8 (377us vs
# 406us) and loses by M=9 (440us vs 397us). Same cap as Q6_K; Q4_K's is 2 only
# because its tensor-core GEMM takes over there.
_GEMV_MAX_M = 8


@cute.kernel
def _q5_k_matmul_kernel(gW: cute.Tensor, gH: cute.Tensor,
                        gX: cute.Tensor, gO: cute.Tensor,
                        n_rows: cutlass.Int32, shape_x: cute.Shape,
                        M: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    n = bid * _SB_PER_CTA + (tid >> 5)  # output row (row of W)
    lane = tid & 31
    grp = lane >> 3                     # which pair of sub-blocks this lane owns
    sub = lane & 7                      # which quarter of the 64-wide group
    off = sub << 2                      # first element within it

    if n < n_rows:
        acc = cute.make_rmem_tensor(M, cutlass.Float32)
        for m in cutlass.range_constexpr(M):
            acc[m] = cutlass.Float32(0.0)

        n_sb = shape_x[1] >> 8
        for s in cutlass.range(n_sb):
            base = s * _BLOCK_WORDS
            d = cutlass.Float32(gH[n, s * _BLOCK_HALVES])
            dmin = cutlass.Float32(gH[n, s * _BLOCK_HALVES + 1])
            w_d = cutlass.Int32(gW[n, base + 1])
            w_m = cutlass.Int32(gW[n, base + 2])
            w_md = cutlass.Int32(gW[n, base + 3])
            # low nibbles: qs bytes 4*lane..4*lane+3
            qw = cutlass.Int32(gW[n, base + _QS_WORD0 + lane])
            # fifth bits: qh bytes 4*sub..4*sub+3, bit kk of byte t
            qhw = cutlass.Int32(gW[n, base + _QH_WORD0 + sub])

            # Two sub-blocks per lane: kk = 2*grp (low nibbles) and 2*grp + 1
            # (high). kk < 4 for the first half of the warp and not the second,
            # so select branchlessly rather than diverge.
            for half in cutlass.range_constexpr(2):
                kk = 2 * grp + half
                jj = kk & 3
                bd = (w_d >> (8 * jj)) & 0xFF
                bm = (w_m >> (8 * jj)) & 0xFF
                bmd = (w_md >> (8 * jj)) & 0xFF
                hi_mask = -(kk >> 2)               # 0 for kk<4, -1 otherwise
                sc = (((bd & 63) & ~hi_mask)
                      | (((bmd & 0x0F) | ((bd >> 6) << 4)) & hi_mask))
                mn = (((bm & 63) & ~hi_mask)
                      | (((bmd >> 4) | ((bm >> 6) << 4)) & hi_mask))
                dl = d * cutlass.Float32(sc)
                dm = dmin * cutlass.Float32(mn)
                e0 = s * QK_K + 64 * grp + 32 * half + off
                for j in cutlass.range_constexpr(4):
                    q = (((qw >> (8 * j + 4 * half)) & 0x0F)
                         | (((qhw >> (8 * j + kk)) & 0x01) << 4))
                    wgt = cutlass.Float32(
                        cutlass.BFloat16(dl * cutlass.Float32(q) - dm))
                    for m in cutlass.range_constexpr(M):
                        acc[m] = acc[m] + wgt * cutlass.Float32(gX[m, e0 + j])

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
def _q5_k_matmul_launch(stream, gU: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor):
    n_rows = gU.shape[0]
    row_words = gU.shape[1] // 4
    row_halves = gU.shape[1] // 2
    wptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Int32)
    gW = cute.make_tensor(wptr, cute.make_layout((n_rows, row_words), stride=(row_words, 1)))
    hptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Float16)
    gH = cute.make_tensor(hptr, cute.make_layout((n_rows, row_halves), stride=(row_halves, 1)))
    n_cta = (n_rows + _SB_PER_CTA - 1) // _SB_PER_CTA
    _q5_k_matmul_kernel(gW, gH, gX, gO, n_rows, gX.shape, gX.shape[0]).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def matmul_q5_k(x: jax.Array, w) -> jax.Array:
    """``x @ w.T`` with ``w`` a Q5_K :class:`~gguf_jax.QuantizedArray` (N, K).

    ``x`` is bfloat16 ``(..., K)``; the result is bfloat16 ``(..., N)``.
    Weights are dequantized in registers (rounded through bfloat16, so values
    match ``x @ w.dequantize(bfloat16).T`` up to f32 summation order) and
    never materialized. Dispatches on the flattened batch size M: warp-GEMV
    for decode shapes, dequantize-then-matmul above them.
    """
    from gguf_jax.array import QuantizedArray

    assert isinstance(w, QuantizedArray) and w.qtype == GGMLQuantizationType.Q5_K
    assert len(w.shape) == 2, "w must be a 2D weight"
    n_rows, k_dim = w.shape
    assert x.shape[-1] == k_dim, f"contraction mismatch: {x.shape[-1]} != {k_dim}"
    assert x.dtype == jnp.bfloat16, "x must be bfloat16"

    xm = x.reshape(-1, k_dim)
    m = xm.shape[0]
    if m <= _GEMV_MAX_M:
        out = cutejax.call(
            _q5_k_matmul_launch,
            jax.ShapeDtypeStruct((m, n_rows), jnp.bfloat16),
            w.data.reshape(n_rows, -1), xm,
            in_specs=[None, cutejax.ArraySpec(static_dims=(0,))],
            out_specs=cutejax.ArraySpec(static_dims=(0,)),
        )
    else:
        out = xm @ w.dequantize(jnp.bfloat16).T
    return out.reshape(*x.shape[:-1], n_rows)
