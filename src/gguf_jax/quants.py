"""Pure-JAX dequantization for GGUF/GGML quantized tensor formats.

Each kernel mirrors the numpy reference implementation in ``gguf.quants``
(gguf-py, from llama.cpp) operation-for-operation, so that the dequantized
float32 output is bitwise identical to the reference.

Layout convention (same as gguf-py): a quantized tensor is stored as uint8
bytes with shape ``(..., row_bytes)`` where the last axis packs
``row_bytes // type_size`` blocks of ``block_size`` elements each.

The kernels are registered in a dict keyed by ``GGMLQuantizationType``; a
faster implementation (e.g. a CuTe DSL kernel via cutejax) can replace an
entry with :func:`register_dequant`.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gguf import quants as _ref
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, QK_K

__all__ = [
    "dequantize",
    "dequantize_blocks",
    "register_dequant",
    "supported_types",
    "GGMLQuantizationType",
]

# fn(blocks: uint8[n_blocks, type_size], dtype) -> dtype[n_blocks, block_size]
# The result must equal the bitwise-exact float32 decode rounded once to
# ``dtype`` (a no-op for float32).
DequantFn = Callable[[jax.Array, Any], jax.Array]

_DEQUANT: dict[GGMLQuantizationType, DequantFn] = {}


def register_dequant(qtype: GGMLQuantizationType, fn: DequantFn, *, override: bool = False) -> None:
    """Register a block-dequantization kernel for ``qtype``.

    ``fn`` maps ``uint8[n_blocks, type_size]`` plus a target dtype to
    ``dtype[n_blocks, block_size]``; the values must be the bitwise-exact
    float32 decode rounded once to ``dtype``. A kernel that computes natively
    in the output dtype (e.g. a fused cutejax kernel writing bfloat16) avoids
    materializing the float32 intermediate. Pass ``override=True`` to replace
    the built-in pure-JAX kernel.
    """
    if qtype in _DEQUANT and not override:
        raise ValueError(f"dequant kernel for {qtype.name} already registered")
    _DEQUANT[qtype] = fn


def supported_types() -> list[GGMLQuantizationType]:
    return sorted(_DEQUANT.keys(), key=lambda t: t.name)


def _register(qtype: GGMLQuantizationType):
    """Register a float32-computing kernel, adding the output-dtype cast.

    The built-in pure-JAX kernels compute in float32 (that is what the
    bitwise contract is defined against); the cast to the requested dtype
    fuses into the surrounding XLA computation, so this costs nothing.
    """
    def deco(fn: Callable[[jax.Array], jax.Array]):
        def wrapper(blocks: jax.Array, dtype) -> jax.Array:
            return fn(blocks).astype(dtype)
        wrapper.__name__ = fn.__name__
        register_dequant(qtype, wrapper)
        return fn
    return deco


# ---------------------------------------------------------------------------
# helpers

def _view(x: jax.Array, dtype) -> jax.Array:
    """numpy-style ``.view(dtype)`` on the last axis of a uint8 array."""
    itemsize = np.dtype(dtype).itemsize
    if itemsize == 1:
        return lax.bitcast_convert_type(x, dtype)
    assert x.shape[-1] % itemsize == 0
    x = x.reshape(*x.shape[:-1], x.shape[-1] // itemsize, itemsize)
    return lax.bitcast_convert_type(x, dtype)


def _f16_to_f32(h: jax.Array) -> jax.Array:
    """float16 -> float32, bitwise identical to numpy's astype.

    XLA's convert does not preserve NaN payloads the way numpy does, so
    non-finite values are widened by explicit bit manipulation instead.
    """
    bits = lax.bitcast_convert_type(h, jnp.uint16).astype(jnp.uint32)
    naninf = ((bits & 0x8000) << 16) | 0x7F800000 | ((bits & 0x3FF) << 13)
    normal = lax.bitcast_convert_type(h.astype(jnp.float32), jnp.uint32)
    exp = (bits >> 10) & 0x1F
    return lax.bitcast_convert_type(jnp.where(exp == 31, naninf, normal), jnp.float32)


def _f16_bytes(x: jax.Array) -> jax.Array:
    """uint8 (..., 2n) -> float32 (..., n), via little-endian float16."""
    return _f16_to_f32(_view(x, jnp.float16))


def _u8_to_i8(x: jax.Array) -> jax.Array:
    return lax.bitcast_convert_type(x, jnp.int8)


def _f32(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32)


def _u8(vals) -> jax.Array:
    return jnp.array(vals, dtype=jnp.uint8)


# ---------------------------------------------------------------------------
# float passthrough "quants"

@_register(GGMLQuantizationType.F32)
def _dequant_f32(blocks: jax.Array) -> jax.Array:
    return _view(blocks, jnp.float32)


@_register(GGMLQuantizationType.F16)
def _dequant_f16(blocks: jax.Array) -> jax.Array:
    return _f16_bytes(blocks)


@_register(GGMLQuantizationType.BF16)
def _dequant_bf16(blocks: jax.Array) -> jax.Array:
    n = _view(blocks, jnp.int16).astype(jnp.int32) << 16
    return lax.bitcast_convert_type(n, jnp.float32)


# ---------------------------------------------------------------------------
# legacy quants

@_register(GGMLQuantizationType.Q4_0)
def _dequant_q4_0(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d, qs = blocks[:, :2], blocks[:, 2:]

    d = _f16_bytes(d)

    qs = qs.reshape(n_blocks, -1, 1, 16) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1).astype(jnp.int8) - jnp.int8(8)

    return d * _f32(qs)


@_register(GGMLQuantizationType.Q4_1)
def _dequant_q4_1(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d, m, qs = blocks[:, :2], blocks[:, 2:4], blocks[:, 4:]

    d = _f16_bytes(d)
    m = _f16_bytes(m)

    qs = qs.reshape(n_blocks, -1, 1, 16) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qs = _f32((qs & 0x0F).reshape(n_blocks, -1))

    return (d * qs) + m


@_register(GGMLQuantizationType.Q5_0)
def _dequant_q5_0(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d, qh, qs = blocks[:, :2], blocks[:, 2:6], blocks[:, 6:]

    d = _f16_bytes(d)
    qh = _view(qh, jnp.uint32)

    qh = qh.reshape(n_blocks, 1) >> jnp.arange(32, dtype=jnp.uint32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, 16) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qh = (qh & 0x01).astype(jnp.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).astype(jnp.int8) - jnp.int8(16)

    return d * _f32(qs)


@_register(GGMLQuantizationType.Q5_1)
def _dequant_q5_1(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d, m, qh, qs = blocks[:, :2], blocks[:, 2:4], blocks[:, 4:8], blocks[:, 8:]

    d = _f16_bytes(d)
    m = _f16_bytes(m)
    qh = _view(qh, jnp.uint32)

    qh = qh.reshape(n_blocks, 1) >> jnp.arange(32, dtype=jnp.uint32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, 16) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qh = (qh & 0x01).astype(jnp.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = _f32(ql | (qh << 4))

    return (d * qs) + m


@_register(GGMLQuantizationType.Q8_0)
def _dequant_q8_0(blocks: jax.Array) -> jax.Array:
    d, x = blocks[:, :2], blocks[:, 2:]
    d = _f16_bytes(d)
    x = _f32(_u8_to_i8(x))
    return x * d


# ---------------------------------------------------------------------------
# K-quants

@_register(GGMLQuantizationType.Q2_K)
def _dequant_q2_k(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    scales = blocks[:, : QK_K // 16]
    qs = blocks[:, QK_K // 16 : QK_K // 16 + QK_K // 4]
    d = blocks[:, QK_K // 16 + QK_K // 4 : QK_K // 16 + QK_K // 4 + 2]
    dmin = blocks[:, QK_K // 16 + QK_K // 4 + 2 :]

    d = _f16_bytes(d)
    dmin = _f16_bytes(dmin)

    dl = (d * _f32(scales & 0xF)).reshape(n_blocks, QK_K // 16, 1)
    ml = (dmin * _f32(scales >> 4)).reshape(n_blocks, QK_K // 16, 1)

    shift = _u8([0, 2, 4, 6]).reshape(1, 1, 4, 1)
    qs = (qs.reshape(n_blocks, -1, 1, 32) >> shift) & 3
    qs = _f32(qs.reshape(n_blocks, QK_K // 16, 16))

    qs = dl * qs - ml

    return qs.reshape(n_blocks, -1)


@_register(GGMLQuantizationType.Q3_K)
def _dequant_q3_k(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    hmask = blocks[:, : QK_K // 8]
    qs = blocks[:, QK_K // 8 : QK_K // 8 + QK_K // 4]
    scales = blocks[:, QK_K // 8 + QK_K // 4 : QK_K // 8 + QK_K // 4 + 12]
    d = blocks[:, QK_K // 8 + QK_K // 4 + 12 :]

    d = _f16_bytes(d)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape(n_blocks, 1, 8) >> _u8([0, 4]).reshape(1, 2, 1)
    lscales = lscales.reshape(n_blocks, 16)
    hscales = hscales.reshape(n_blocks, 1, 4) >> _u8([0, 2, 4, 6]).reshape(1, 4, 1)
    hscales = hscales.reshape(n_blocks, 16)
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = _f32(scales.astype(jnp.int8) - jnp.int8(32))

    dl = (d * scales).reshape(n_blocks, 16, 1)

    ql = qs.reshape(n_blocks, -1, 1, 32) >> _u8([0, 2, 4, 6]).reshape(1, 1, 4, 1)
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> _u8(list(range(8))).reshape(1, 1, 8, 1)
    ql = ql.reshape(n_blocks, 16, QK_K // 16) & 3
    qh = qh.reshape(n_blocks, 16, QK_K // 16) & 1
    qh = qh ^ 1  # strangely, the offset is zero when the bitmask is 1
    q = _f32(ql.astype(jnp.int8) - (qh << 2).astype(jnp.int8))

    return (dl * q).reshape(n_blocks, QK_K)


K_SCALE_SIZE = 12


def _get_scale_min(scales: jax.Array) -> tuple[jax.Array, jax.Array]:
    n_blocks = scales.shape[0]
    scales = scales.reshape(n_blocks, 3, 4)
    d, m, m_d = scales[:, 0:1], scales[:, 1:2], scales[:, 2:3]

    sc = jnp.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = jnp.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)

    return sc.reshape(n_blocks, 8), mn.reshape(n_blocks, 8)


@_register(GGMLQuantizationType.Q4_K)
def _dequant_q4_k(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    dmin = blocks[:, 2:4]
    scales = blocks[:, 4 : 4 + K_SCALE_SIZE]
    qs = blocks[:, 4 + K_SCALE_SIZE :]

    d = _f16_bytes(d)
    dmin = _f16_bytes(dmin)

    sc, m = _get_scale_min(scales)

    d = (d * _f32(sc)).reshape(n_blocks, -1, 1)
    dm = (dmin * _f32(m)).reshape(n_blocks, -1, 1)

    qs = qs.reshape(n_blocks, -1, 1, 32) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qs = _f32((qs & 0x0F).reshape(n_blocks, -1, 32))

    return (d * qs - dm).reshape(n_blocks, QK_K)


@_register(GGMLQuantizationType.Q5_K)
def _dequant_q5_k(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    dmin = blocks[:, 2:4]
    scales = blocks[:, 4 : 4 + K_SCALE_SIZE]
    qh = blocks[:, 4 + K_SCALE_SIZE : 4 + K_SCALE_SIZE + QK_K // 8]
    qs = blocks[:, 4 + K_SCALE_SIZE + QK_K // 8 :]

    d = _f16_bytes(d)
    dmin = _f16_bytes(dmin)

    sc, m = _get_scale_min(scales)

    d = (d * _f32(sc)).reshape(n_blocks, -1, 1)
    dm = (dmin * _f32(m)).reshape(n_blocks, -1, 1)

    ql = qs.reshape(n_blocks, -1, 1, 32) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qh = qh.reshape(n_blocks, -1, 1, 32) >> _u8(list(range(8))).reshape(1, 1, 8, 1)
    ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
    qh = (qh & 0x01).reshape(n_blocks, -1, 32)
    q = _f32(ql | (qh << 4))

    return (d * q - dm).reshape(n_blocks, QK_K)


@_register(GGMLQuantizationType.Q6_K)
def _dequant_q6_k(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    ql = blocks[:, : QK_K // 2]
    qh = blocks[:, QK_K // 2 : QK_K // 2 + QK_K // 4]
    scales = blocks[:, QK_K // 2 + QK_K // 4 : QK_K // 2 + QK_K // 4 + QK_K // 16]
    d = blocks[:, QK_K // 2 + QK_K // 4 + QK_K // 16 :]

    scales = _f32(_u8_to_i8(scales))
    d = _f16_bytes(d)
    d = (d * scales).reshape(n_blocks, QK_K // 16, 1)

    ql = ql.reshape(n_blocks, -1, 1, 64) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
    qh = qh.reshape(n_blocks, -1, 1, 32) >> _u8([0, 2, 4, 6]).reshape(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(n_blocks, -1, 32)
    q = (ql | (qh << 4)).astype(jnp.int8) - jnp.int8(32)
    q = _f32(q.reshape(n_blocks, QK_K // 16, -1))

    return (d * q).reshape(n_blocks, QK_K)


# ---------------------------------------------------------------------------
# ternary quants

@_register(GGMLQuantizationType.TQ1_0)
def _dequant_tq1_0(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    split1 = (QK_K - 4 * QK_K // 64) // 5
    split2 = split1 + QK_K // 64
    qs, qh, d = blocks[:, :split1], blocks[:, split1:split2], blocks[:, split2:]

    d = _f16_bytes(d)

    qs0, qs1 = qs[..., :32], qs[..., 32:]
    qs0 = qs0.reshape(n_blocks, -1, 1, 32) * _u8([1, 3, 9, 27, 81]).reshape(1, 1, 5, 1)
    qs0 = qs0.reshape(n_blocks, -1)
    qs1 = qs1.reshape(n_blocks, -1, 1, 16) * _u8([1, 3, 9, 27, 81]).reshape(1, 1, 5, 1)
    qs1 = qs1.reshape(n_blocks, -1)
    qh = qh.reshape(n_blocks, -1, 1, 4) * _u8([1, 3, 9, 27]).reshape(1, 1, 4, 1)
    qh = qh.reshape(n_blocks, -1)
    qs = jnp.concatenate([qs0, qs1, qh], axis=-1)
    qs = ((qs.astype(jnp.uint16) * 3) >> 8).astype(jnp.int8) - jnp.int8(1)

    return d * _f32(qs)


@_register(GGMLQuantizationType.TQ2_0)
def _dequant_tq2_0(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    qs, d = blocks[:, : QK_K // 4], blocks[:, QK_K // 4 :]

    d = _f16_bytes(d)

    qs = qs.reshape(n_blocks, -1, 1, 32) >> _u8([0, 2, 4, 6]).reshape(1, 1, 4, 1)
    qs = (qs & 0x03).reshape(n_blocks, -1).astype(jnp.int8) - jnp.int8(1)

    return d * _f32(qs)


# ---------------------------------------------------------------------------
# microscaling float quants

_MXFP4_KVALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def _e8m0_to_fp32_half(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.uint32)
    bits = jnp.where(x < 2, jnp.uint32(0x00200000) << x, (x - 1) << 23)
    return lax.bitcast_convert_type(bits, jnp.float32)


@_register(GGMLQuantizationType.MXFP4)
def _dequant_mxfp4(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    e, qs = blocks[:, :1], blocks[:, 1:]

    d = _e8m0_to_fp32_half(e)

    qs = qs.reshape(n_blocks, 1, -1) >> _u8([0, 4]).reshape(1, 2, 1)
    qs = _u8_to_i8(qs & 0x0F)

    kvalues = jnp.array(_MXFP4_KVALUES, dtype=jnp.int8)
    qs = jnp.take(kvalues, qs).reshape(n_blocks, -1)

    normal = d * _f32(qs)

    # For e < 2 the E8M0 scale is a float32 subnormal (2**(e-128)) and the
    # product can underflow into the subnormal range, which XLA:CPU's
    # jit-compiled code flushes to zero. The product |q| * 2**(e-128) has at
    # most 4 significant bits, so it is exactly representable: construct its
    # bits with integer ops, which FTZ cannot touch.
    e32 = e.astype(jnp.uint32)
    absq = jnp.abs(qs).astype(jnp.uint32)
    sign = (qs < 0).astype(jnp.uint32) << 31
    p = absq << (21 + jnp.minimum(e32, 1))  # product as a multiple of 2**-149
    s = (p >= 1 << 24).astype(jnp.uint32) + (p >= 1 << 25)
    bits = jnp.where(p < 1 << 23, p, ((s + 1) << 23) | ((p >> s) & 0x7FFFFF))
    subnormal_d = lax.bitcast_convert_type(sign | bits, jnp.float32)

    return jnp.where(e32 < 2, subnormal_d, normal)


def _exact_exp2(k: jax.Array) -> jax.Array:
    """2.0 ** k for integer k in the normal range, exact by bit construction."""
    return lax.bitcast_convert_type(((k + 127) << 23).astype(jnp.uint32), jnp.float32)


def _ue4m3_to_fp32(x: jax.Array) -> jax.Array:
    exp = (x >> 3).astype(jnp.int32) & 0xF
    man = _f32(x & 0x7)
    raw = jnp.where(
        exp == 0,
        man * jnp.float32(2**-9),
        (jnp.float32(1.0) + man / 8) * _exact_exp2(exp - 7))
    return jnp.where((x == 0) | (x == 0x7F), jnp.float32(0.0), raw * jnp.float32(0.5))


@_register(GGMLQuantizationType.NVFP4)
def _dequant_nvfp4(blocks: jax.Array) -> jax.Array:
    n_super = blocks.shape[0]
    d_bytes, qs = blocks[:, :4], blocks[:, 4:]

    d = _ue4m3_to_fp32(d_bytes).reshape(n_super, 4, 1)

    qs = qs.reshape(n_super, 4, 8)
    lo = _u8_to_i8(qs & 0x0F)
    hi = _u8_to_i8(qs >> 4)
    vals = jnp.concatenate([lo, hi], axis=-1)  # (n_super, 4, 16)

    kvalues = jnp.array(_MXFP4_KVALUES, dtype=jnp.int8)
    vals = jnp.take(kvalues, vals)

    return (d * _f32(vals)).reshape(n_super, 64)


# ---------------------------------------------------------------------------
# i-quants (grid codebook types)
#
# The decoded grids and sign tables are reused from the reference classes in
# gguf.quants rather than duplicating their hex dumps here.

def _grid_of(ref_cls) -> jax.Array:
    ref_cls.init_grid()
    assert ref_cls.grid is not None
    return jnp.asarray(ref_cls.grid.reshape(ref_cls.grid_shape))


def _ksigns() -> jax.Array:
    return jnp.asarray(np.frombuffer(_ref.IQ2_XXS.ksigns, dtype=np.uint8))


def _sign_bits(sign_bytes: jax.Array) -> jax.Array:
    """uint8 (...,) -> float32 (..., 8) of +-1 from the bits of each byte."""
    signs = sign_bytes[..., None] >> _u8(list(range(8)))
    signs = signs & 0x01
    return jnp.where(signs == 0, jnp.float32(1), jnp.float32(-1))


@_register(GGMLQuantizationType.IQ2_XXS)
def _dequant_iq2_xxs(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d, qs = blocks[:, :2], blocks[:, 2:]

    d = _f16_bytes(d)
    qs = _view(qs, jnp.uint32).reshape(n_blocks, -1, 2)

    db = d * (jnp.float32(0.5) + _f32(qs[..., 1] >> 28)) * jnp.float32(0.25)
    db = db.reshape(n_blocks, -1, 1, 1)

    signs = qs[..., 1].reshape(n_blocks, -1, 1) >> jnp.array([0, 7, 14, 21], dtype=jnp.uint32).reshape(1, 1, 4)
    signs = jnp.take(_ksigns(), signs & 0x7F)
    signs = _sign_bits(signs).reshape(n_blocks, -1, 4, 8)

    idx = _view(qs[..., 0], jnp.uint8).reshape(n_blocks, -1)
    grid = jnp.take(_grid_of(_ref.IQ2_XXS), idx, axis=0).reshape(n_blocks, -1, 4, 8)

    return (db * grid * signs).reshape(n_blocks, -1)


@_register(GGMLQuantizationType.IQ2_XS)
def _dequant_iq2_xs(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    qs = blocks[:, 2 : 2 + 2 * QK_K // 8]
    scales = blocks[:, 2 + 2 * QK_K // 8 :]

    d = _f16_bytes(d)
    qs = _view(qs, jnp.uint16)

    scales = scales.reshape(n_blocks, -1, 1) >> _u8([0, 4]).reshape(1, 1, 2)
    scales = _f32((scales & 0x0F).reshape(n_blocks, -1))
    db = d * (jnp.float32(0.5) + scales) * jnp.float32(0.25)
    db = db.reshape(n_blocks, -1, 1, 1)

    signs = jnp.take(_ksigns(), qs >> 9)
    signs = _sign_bits(signs).reshape(n_blocks, -1, 2, 8)

    grid = jnp.take(_grid_of(_ref.IQ2_XS), qs & 511, axis=0).reshape(n_blocks, -1, 2, 8)

    return (db * grid * signs).reshape(n_blocks, -1)


@_register(GGMLQuantizationType.IQ2_S)
def _dequant_iq2_s(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    qs = blocks[:, 2 : 2 + QK_K // 8]
    signs = blocks[:, 2 + QK_K // 8 : 2 + QK_K // 4]
    qh = blocks[:, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]
    scales = blocks[:, 2 + QK_K // 4 + QK_K // 32 :]

    d = _f16_bytes(d)

    scales = scales.reshape(n_blocks, -1, 1) >> _u8([0, 4]).reshape(1, 1, 2)
    scales = _f32((scales & 0x0F).reshape(n_blocks, -1))
    db = d * (jnp.float32(0.5) + scales) * jnp.float32(0.25)
    db = db.reshape(n_blocks, -1, 1, 1)

    signs = _sign_bits(signs).reshape(n_blocks, -1, 2, 8)

    qh = qh.reshape(n_blocks, -1, 1) >> _u8([0, 2, 4, 6]).reshape(1, 1, 4)
    qs = qs.astype(jnp.uint16) | ((qh & 0x03).astype(jnp.uint16) << 8).reshape(n_blocks, -1)

    grid = jnp.take(_grid_of(_ref.IQ2_S), qs, axis=0).reshape(n_blocks, -1, 2, 8)

    return (db * grid * signs).reshape(n_blocks, -1)


@_register(GGMLQuantizationType.IQ3_XXS)
def _dequant_iq3_xxs(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    qs = blocks[:, 2 : 2 + QK_K // 4]
    scales = blocks[:, 2 + QK_K // 4 :]

    d = _f16_bytes(d)
    scales = _view(scales, jnp.uint32)

    db = d * (jnp.float32(0.5) + _f32(scales >> 28)) * jnp.float32(0.5)
    db = db.reshape(n_blocks, -1, 1, 1)

    signs = scales.reshape(n_blocks, -1, 1) >> jnp.array([0, 7, 14, 21], dtype=jnp.uint32).reshape(1, 1, 4)
    signs = jnp.take(_ksigns(), signs & 0x7F)
    signs = _sign_bits(signs).reshape(n_blocks, -1, 4, 8)

    grid = jnp.take(_grid_of(_ref.IQ3_XXS), qs, axis=0).reshape(n_blocks, -1, 4, 8)

    return (db * grid * signs).reshape(n_blocks, -1)


@_register(GGMLQuantizationType.IQ3_S)
def _dequant_iq3_s(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    qs = blocks[:, 2 : 2 + QK_K // 4]
    qh = blocks[:, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]
    signs = blocks[:, 2 + QK_K // 4 + QK_K // 32 : 2 + QK_K // 4 + QK_K // 32 + QK_K // 8]
    scales = blocks[:, 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 :]

    d = _f16_bytes(d)

    scales = scales.reshape(n_blocks, -1, 1) >> _u8([0, 4]).reshape(1, 1, 2)
    scales = (scales & 0x0F).reshape(n_blocks, -1)
    db = d * _f32(1 + 2 * scales)
    db = db.reshape(n_blocks, -1, 1, 1)

    signs = _sign_bits(signs).reshape(n_blocks, -1, 4, 8)

    qh = qh.reshape(n_blocks, -1, 1) >> _u8(list(range(8)))
    qh = (qh & 0x01).astype(jnp.uint16).reshape(n_blocks, -1)
    qs = qs.astype(jnp.uint16) | (qh << 8)

    grid = jnp.take(_grid_of(_ref.IQ3_S), qs, axis=0).reshape(n_blocks, -1, 4, 8)

    return (db * grid * signs).reshape(n_blocks, -1)


_IQ1_DELTA = np.float32(0.125)


@_register(GGMLQuantizationType.IQ1_S)
def _dequant_iq1_s(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    qs = blocks[:, 2 : 2 + QK_K // 8]
    qh = blocks[:, 2 + QK_K // 8 :]

    d = _f16_bytes(d)
    qh = _view(qh, jnp.uint16)

    dl = d * _f32(2 * ((qh >> 12) & 7) + 1)
    dl = dl.reshape(n_blocks, -1, 1, 1)
    delta = jnp.where((qh & 0x8000) == 0, _IQ1_DELTA, -_IQ1_DELTA)
    delta = delta.reshape(n_blocks, -1, 1, 1)

    qh = qh.reshape(n_blocks, -1, 1) >> jnp.array([0, 3, 6, 9], dtype=jnp.uint16).reshape(1, 1, 4)
    qs = qs.astype(jnp.uint16) | ((qh & 7) << 8).reshape(n_blocks, -1)

    grid = jnp.take(_grid_of(_ref.IQ1_S), qs, axis=0).reshape(n_blocks, -1, 4, 8)

    return (dl * (grid + delta)).reshape(n_blocks, -1)


@_register(GGMLQuantizationType.IQ1_M)
def _dequant_iq1_m(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    qs = blocks[:, : QK_K // 8]
    qh = blocks[:, QK_K // 8 : QK_K // 8 + QK_K // 16]
    scales = blocks[:, QK_K // 8 + QK_K // 16 :]

    # The f16 scale is packed across multiple bytes
    scales = _view(scales, jnp.uint16)
    d = (scales.reshape(n_blocks, 4) & 0xF000) >> jnp.array([12, 8, 4, 0], dtype=jnp.uint16).reshape(1, 4)
    d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
    d = _f16_to_f32(lax.bitcast_convert_type(d, jnp.float16)).reshape(n_blocks, 1)

    scales = scales.reshape(n_blocks, -1, 1) >> jnp.array([0, 3, 6, 9], dtype=jnp.uint16).reshape(1, 1, 4)
    scales = (scales & 0x07).reshape(n_blocks, -1)
    dl = d * _f32(2 * scales + 1)
    dl = dl.reshape(n_blocks, -1, 2, 1, 1)

    qh = qh.reshape(n_blocks, -1, 1) >> _u8([0, 4]).reshape(1, 1, 2)
    qs = qs.astype(jnp.uint16) | ((qh & 0x07).astype(jnp.uint16) << 8).reshape(n_blocks, -1)

    delta = jnp.where((qh & 0x08) == 0, _IQ1_DELTA, -_IQ1_DELTA)
    delta = delta.reshape(n_blocks, -1, 2, 2, 1)

    grid = jnp.take(_grid_of(_ref.IQ1_S), qs, axis=0).reshape(n_blocks, -1, 2, 2, 8)

    return (dl * (grid + delta)).reshape(n_blocks, -1)


_IQ4_NL_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


@_register(GGMLQuantizationType.IQ4_NL)
def _dequant_iq4_nl(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d, qs = blocks[:, :2], blocks[:, 2:]

    d = _f16_bytes(d)

    qs = qs.reshape(n_blocks, -1, 1, 16) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    kvalues = jnp.array(_IQ4_NL_KVALUES, dtype=jnp.int8)
    qs = _f32(jnp.take(kvalues, qs))

    return d * qs


@_register(GGMLQuantizationType.IQ4_XS)
def _dequant_iq4_xs(blocks: jax.Array) -> jax.Array:
    n_blocks = blocks.shape[0]
    d = blocks[:, :2]
    scales_h = blocks[:, 2:4]
    scales_l = blocks[:, 4 : 4 + QK_K // 64]
    qs = blocks[:, 4 + QK_K // 64 :]

    d = _f16_bytes(d)
    scales_h = _view(scales_h, jnp.uint16)

    scales_l = scales_l.reshape(n_blocks, -1, 1) >> _u8([0, 4]).reshape(1, 1, 2)
    scales_h = scales_h.reshape(n_blocks, 1, -1) >> jnp.array(
        [2 * i for i in range(QK_K // 32)], dtype=jnp.uint16).reshape(1, -1, 1)
    scales_l = scales_l.reshape(n_blocks, -1) & 0x0F
    scales_h = scales_h.reshape(n_blocks, -1).astype(jnp.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).astype(jnp.int8) - jnp.int8(32)
    dl = (d * _f32(scales)).reshape(n_blocks, -1, 1)

    qs = qs.reshape(n_blocks, -1, 1, 16) >> _u8([0, 4]).reshape(1, 1, 2, 1)
    qs = qs.reshape(n_blocks, -1, 32) & 0x0F

    kvalues = jnp.array(_IQ4_NL_KVALUES, dtype=jnp.int8)
    qs = _f32(jnp.take(kvalues, qs))

    return (dl * qs).reshape(n_blocks, -1)


# ---------------------------------------------------------------------------
# top-level entry points

def dequantize_blocks(blocks: jax.Array, qtype: GGMLQuantizationType,
                      dtype=jnp.float32) -> jax.Array:
    """Dequantize ``uint8[n_blocks, type_size]`` to ``dtype[n_blocks, block_size]``."""
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if qtype not in _DEQUANT:
        raise NotImplementedError(f"Dequantization for {qtype.name} is not implemented")
    if blocks.dtype != jnp.uint8 or blocks.ndim != 2 or blocks.shape[-1] != type_size:
        raise ValueError(
            f"expected uint8 blocks of shape (n_blocks, {type_size}) for {qtype.name}, "
            f"got {blocks.dtype} {blocks.shape}")
    out = _DEQUANT[qtype](blocks, dtype)
    assert out.shape == (blocks.shape[0], block_size)
    return out


def dequantize(data: jax.Array, qtype: GGMLQuantizationType, dtype=jnp.float32) -> jax.Array:
    """Dequantize a byte-shaped uint8 array.

    ``data`` has shape ``(..., row_bytes)`` (the layout produced by
    ``gguf.quants.quantize`` and stored in GGUF files); the result has shape
    ``(..., row_bytes // type_size * block_size)``. With ``dtype=float32``
    (the default) the result is bitwise identical to
    ``gguf.quants.dequantize``; other dtypes are that result rounded once.
    """
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    shape = _ref.quant_shape_from_byte_shape(data.shape, qtype)
    blocks = data.reshape(-1, type_size)
    return dequantize_blocks(blocks, qtype, dtype).reshape(shape)
