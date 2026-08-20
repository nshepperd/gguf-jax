"""Fused IQ4_XS dequant-matmul (warp-GEMV) as a CuTe DSL kernel, via cutejax.

Layout of one 136-byte IQ4_XS superblock (QK_K = 256 elements):

    bytes   0..1    d         (f16) super-scale
    bytes   2..3    scales_h  (u16) high 2 bits of each of the 8 sub-scales
    bytes   4..7    scales_l  low 4 bits of each of the 8 sub-scales
    bytes   8..135  qs        128 bytes of 4-bit codebook indices

Element e decodes as, for sub-block j = e // 32 and position t = e % 32,

    sc  = (scales_l[j] | (scales_h[j] << 4)) - 32          (signed, -32..31)
    idx = (qs[16 * j + (t % 16)] >> (4 * (t // 16))) & 0x0F
    out = (d * sc) * kvalues[idx]

with float32 roundings in that association, matching the numpy reference in
gguf.quants (and hence the pure-JAX kernel) bitwise. ``kvalues`` is the
16-entry IQ4_NL codebook — the one thing that makes this type different in
kind from Q4_K/Q5_K/Q6_K: the quant is a codebook *index*, not a magnitude,
so decoding needs a table lookup rather than arithmetic.

136 is a multiple of 4, so the superblock is addressable as uint32 words:
word 0 packs (d, scales_h), word 1 is scales_l — and, little-endian, sub-scale
j's low nibble sits at bit 4*j of it — and words 2..33 are qs.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute, utils
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

QK_K = 256
_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.IQ4_XS][1]  # 136
_BLOCK_WORDS = _BLOCK_BYTES // 4                                 # 34
_BLOCK_HALVES = _BLOCK_BYTES // 2                                # 68
_QS_WORD0 = 2                                                    # bytes 8..135

_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113)

_SB_PER_CTA = 8    # warps (== output rows) per CTA
_CTA = 256

# GEMV-style mapping for small M, following Q4_K's v2 layout: lane w owns the
# four CONSECUTIVE qs bytes 4w..4w+3, one uint32 load per superblock covering
# the whole 128-byte qs region across the warp.
#
# IQ4_XS's 16-byte sub-blocks make that mapping land even better than on
# Q4_K/Q5_K: bytes 4w..4w+3 all sit inside sub-block w // 4, so each lane
# needs exactly ONE (scale, dl) pair per superblock instead of two, and its
# eight elements are two runs of four consecutive x values, at t = 4*(w%4) + j
# and t + 16.
#
# The codebook lookup goes through a 16-entry float32 table in shared memory,
# filled once per CTA. Entry i lands in bank i, so 16 distinct values across
# 32 lanes never collide and equal values broadcast: the lookup is a single
# conflict-free LDS however the indices fall.
#
# There is no tensor-core IQ4_XS GEMM, so above this cap matmul_iq4_xs falls
# back to dequantize-then-matmul. Measured on an RTX 5070 Ti (4096x14336
# weight): the GEMV wins from M=1 (65us vs 361us, 5.6x) out to M=7 (341us vs
# 362us), is a wash at M=8 (367us vs 362us) and loses from M=9. The cap sits at
# the wash rather than one below it because the fallback also materializes a
# dense bf16 copy of the whole weight, which the GEMV never does.
_GEMV_MAX_M = 8


@cute.kernel
def _iq4_xs_matmul_kernel(gW: cute.Tensor, gH: cute.Tensor,
                          gX: cute.Tensor, gO: cute.Tensor,
                          n_rows: cutlass.Int32, shape_x: cute.Shape,
                          M: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    @cute.struct
    class SharedStorage:
        kv: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16], 16]

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sKv = SharedStorage(storage).kv.get_tensor(cute.make_layout(16))
    for i in cutlass.range_constexpr(16):
        if tid == i:
            sKv[i] = cutlass.Float32(float(_KVALUES[i]))
    cute.arch.sync_threads()

    n = bid * _SB_PER_CTA + (tid >> 5)  # output row (row of W)
    lane = tid & 31
    sb = lane >> 2                      # the one sub-block this lane owns
    off = (lane & 3) << 2               # first element within it

    if n < n_rows:
        acc = cute.make_rmem_tensor(M, cutlass.Float32)
        for m in cutlass.range_constexpr(M):
            acc[m] = cutlass.Float32(0.0)

        n_sb = shape_x[1] >> 8
        for s in cutlass.range(n_sb):
            base = s * _BLOCK_WORDS
            d = cutlass.Float32(gH[n, s * _BLOCK_HALVES])
            w0 = cutlass.Int32(gW[n, base])          # (d, scales_h)
            w_sl = cutlass.Int32(gW[n, base + 1])    # scales_l
            qw = cutlass.Int32(gW[n, base + _QS_WORD0 + lane])

            sc = (((w_sl >> (4 * sb)) & 0x0F)
                  | ((((w0 >> 16) >> (2 * sb)) & 0x03) << 4)) - 32
            dl = d * cutlass.Float32(sc)

            for half in cutlass.range_constexpr(2):
                e0 = s * QK_K + 32 * sb + 16 * half + off
                for j in cutlass.range_constexpr(4):
                    idx = (qw >> (8 * j + 4 * half)) & 0x0F
                    wgt = cutlass.Float32(cutlass.BFloat16(dl * sKv[idx]))
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
def _iq4_xs_matmul_launch(stream, gU: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor):
    n_rows = gU.shape[0]
    row_words = gU.shape[1] // 4
    row_halves = gU.shape[1] // 2
    wptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Int32)
    gW = cute.make_tensor(wptr, cute.make_layout((n_rows, row_words), stride=(row_words, 1)))
    hptr = cute.recast_ptr(gU.iterator, dtype=cutlass.Float16)
    gH = cute.make_tensor(hptr, cute.make_layout((n_rows, row_halves), stride=(row_halves, 1)))
    n_cta = (n_rows + _SB_PER_CTA - 1) // _SB_PER_CTA
    _iq4_xs_matmul_kernel(gW, gH, gX, gO, n_rows, gX.shape, gX.shape[0]).launch(
        grid=[n_cta, 1, 1], block=[_CTA, 1, 1], stream=stream)


def matmul_iq4_xs(x: jax.Array, w) -> jax.Array:
    """``x @ w.T`` with ``w`` an IQ4_XS :class:`~gguf_jax.QuantizedArray` (N, K).

    ``x`` is bfloat16 ``(..., K)``; the result is bfloat16 ``(..., N)``.
    Weights are dequantized in registers (rounded through bfloat16, so values
    match ``x @ w.dequantize(bfloat16).T`` up to f32 summation order) and
    never materialized. Dispatches on the flattened batch size M: warp-GEMV
    for decode shapes, dequantize-then-matmul above them.
    """
    from gguf_jax.array import QuantizedArray

    assert isinstance(w, QuantizedArray) and w.qtype == GGMLQuantizationType.IQ4_XS
    assert len(w.shape) == 2, "w must be a 2D weight"
    n_rows, k_dim = w.shape
    assert x.shape[-1] == k_dim, f"contraction mismatch: {x.shape[-1]} != {k_dim}"
    assert x.dtype == jnp.bfloat16, "x must be bfloat16"

    xm = x.reshape(-1, k_dim)
    m = xm.shape[0]
    if m <= _GEMV_MAX_M:
        out = cutejax.call(
            _iq4_xs_matmul_launch,
            jax.ShapeDtypeStruct((m, n_rows), jnp.bfloat16),
            w.data.reshape(n_rows, -1), xm,
            in_specs=[None, cutejax.ArraySpec(static_dims=(0,))],
            out_specs=cutejax.ArraySpec(static_dims=(0,)),
        )
    else:
        out = xm @ w.dequantize(jnp.bfloat16).T
    return out.reshape(*x.shape[:-1], n_rows)
