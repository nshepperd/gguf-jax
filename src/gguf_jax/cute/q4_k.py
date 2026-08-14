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


def register() -> None:
    """Replace the pure-JAX Q4_K kernel with the cute kernel.

    The kernel computes in float32 and writes the requested output dtype
    directly (single hardware round), so no float32 intermediate is
    materialized for bfloat16 dequantization.
    """
    from gguf_jax import quants

    quants.register_dequant(
        GGMLQuantizationType.Q4_K, dequantize_q4_k, override=True)
