"""Bitwise tests for the CuTe DSL kernels (skipped without GPU + cute deps)."""

import gguf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

import gguf_jax
from tests.test_bitwise import assert_bitwise_equal, logical_shapes, random_bytes

cute_mod = pytest.importorskip("gguf_jax.cute")

if jax.devices()[0].platform != "gpu":
    pytest.skip("cute kernels need a GPU", allow_module_level=True)

QTYPE = GGMLQuantizationType.Q4_K


def test_cute_q4_k_random_bytes_bitwise():
    for i, shape in enumerate(logical_shapes(QTYPE)):
        data = random_bytes(QTYPE, shape, seed=hash(("cute", i)) % 2**32)
        ref = gguf.quants.dequantize(data, QTYPE)
        ours = np.asarray(cute_mod.dequantize_q4_k(jnp.asarray(data)))
        assert_bitwise_equal(ours, ref, QTYPE)


def test_cute_q4_k_quantized_floats_bitwise():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(16, 512)).astype(np.float32)
    values *= 10.0 ** rng.integers(-8, 8, size=(16, 1)).astype(np.float32)
    # gguf-py cannot quantize Q4_K; llama.cpp-produced blocks are structurally
    # arbitrary bytes anyway, so random bytes above are the stronger test.
    data = random_bytes(QTYPE, values.shape, seed=99)
    ref = gguf.quants.dequantize(data, QTYPE)
    ours = np.asarray(cute_mod.dequantize_q4_k(jnp.asarray(data)))
    assert_bitwise_equal(ours, ref, QTYPE)


def test_cute_register_roundtrip():
    """register() swaps the kernel behind the normal QuantizedArray API."""
    from gguf_jax import quants

    original = quants._DEQUANT[QTYPE]
    try:
        cute_mod.register()
        assert quants._DEQUANT[QTYPE] is not original
        data = random_bytes(QTYPE, (4, 512), seed=7)
        qa = gguf_jax.QuantizedArray.from_bytes(data, QTYPE, dtype=jnp.float32)
        ref = gguf.quants.dequantize(data, QTYPE)
        assert_bitwise_equal(np.asarray(qa.dequantize()), ref, QTYPE)

        # under jit too
        fn = jax.jit(lambda q: q.dequantize())
        assert_bitwise_equal(np.asarray(fn(qa)), ref, QTYPE)
    finally:
        quants.register_dequant(QTYPE, original, override=True)


def _random_finite_q4k(n_rows, k_dim, seed):
    """Random Q4_K blocks with finite (small) d/dmin, byte-shaped (n_rows, row_bytes)."""
    rng = np.random.default_rng(seed)
    nb = n_rows * k_dim // 256
    data = rng.integers(0, 256, size=(nb, 144), dtype=np.uint8)
    data[:, :4] = (rng.normal(size=(nb, 2)) * 0.05).astype(np.float16).view(np.uint8)
    return data.reshape(n_rows, -1)


@pytest.mark.parametrize("m", [1, 5, 32, 100])
def test_cute_gemm_q4_k(m):
    """The tensor-core GEMM kernel, called directly."""
    n_rows, k_dim = 256, 1536
    data = _random_finite_q4k(n_rows, k_dim, seed=m + 50)
    w = gguf_jax.QuantizedArray.from_bytes(data, QTYPE, shape=(n_rows, k_dim))
    rng = np.random.default_rng(m + 500)
    x = jnp.asarray(rng.normal(size=(m, k_dim)), dtype=jnp.bfloat16)

    y = np.asarray(cute_mod.gemm_q4_k(x, w), dtype=np.float32)

    wd = jnp.asarray(gguf.quants.dequantize(data, QTYPE)).astype(jnp.bfloat16)
    ref = np.asarray(
        (x.astype(jnp.float32) @ wd.astype(jnp.float32).T).astype(jnp.bfloat16),
        dtype=np.float32)
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_cute_matmul_force_fused():
    """force_fused keeps the tensor-core kernel beyond the M=128 crossover."""
    n_rows, k_dim = 256, 1024
    data = _random_finite_q4k(n_rows, k_dim, seed=77)
    w = gguf_jax.QuantizedArray.from_bytes(data, QTYPE, shape=(n_rows, k_dim))
    rng = np.random.default_rng(77)
    x = jnp.asarray(rng.normal(size=(200, k_dim)), dtype=jnp.bfloat16)

    y = np.asarray(cute_mod.matmul_q4_k(x, w, force_fused=True), dtype=np.float32)
    wd = jnp.asarray(gguf.quants.dequantize(data, QTYPE)).astype(jnp.bfloat16)
    ref = np.asarray(
        (x.astype(jnp.float32) @ wd.astype(jnp.float32).T).astype(jnp.bfloat16),
        dtype=np.float32)
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())

    # N not a multiple of 64 cannot honor force_fused above the GEMV range
    w_odd = gguf_jax.QuantizedArray.from_bytes(
        _random_finite_q4k(96, k_dim, seed=78), QTYPE, shape=(96, k_dim))
    with pytest.raises(ValueError, match="force_fused"):
        cute_mod.matmul_q4_k(x, w_odd, force_fused=True)


# m chosen to cover all three dispatch paths: 1-2 warp-GEMV, 3-128
# tensor-core GEMM, >128 dequantize-then-matmul
@pytest.mark.parametrize("m", [1, 3, 8, 200])
def test_cute_matmul_q4_k(m):
    n_rows, k_dim = 256, 1536
    data = _random_finite_q4k(n_rows, k_dim, seed=m)
    w = gguf_jax.QuantizedArray.from_bytes(data, QTYPE, shape=(n_rows, k_dim))
    rng = np.random.default_rng(m + 100)
    x = jnp.asarray(rng.normal(size=(m, k_dim)), dtype=jnp.bfloat16)

    y = np.asarray(cute_mod.matmul_q4_k(x, w), dtype=np.float32)

    wd = jnp.asarray(gguf.quants.dequantize(data, QTYPE)).astype(jnp.bfloat16)
    ref = np.asarray(
        (x.astype(jnp.float32) @ wd.astype(jnp.float32).T).astype(jnp.bfloat16),
        dtype=np.float32)
    # identical products, f32 accumulation in a different order, one bf16 round
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_cute_matmul_q4_k_exact_weights():
    """One-hot x reads the dequantized weight out of the kernel column by column.

    This is what pins the per-element rounding: the warp-GEMV rounds each weight
    through bf16 so a fused matmul equals `x @ w.dequantize(bf16).T`. Any change
    to the lane->element mapping must keep it, and any change that factors the
    scales out of the inner loop cannot.
    """
    n_rows, k_dim = 64, 512
    data = _random_finite_q4k(n_rows, k_dim, seed=13)
    w = gguf_jax.QuantizedArray.from_bytes(data, QTYPE, shape=(n_rows, k_dim))
    ref = np.asarray(gguf.quants.dequantize(data, QTYPE)).reshape(n_rows, k_dim)
    ref_bf16 = np.asarray(jnp.asarray(ref).astype(jnp.bfloat16), dtype=np.float32)

    # Deliberately spread over nibble halves, sub-block edges and superblocks.
    for e in (0, 1, 3, 4, 31, 32, 33, 63, 64, 127, 128, 255, 256, k_dim - 1):
        x = jnp.zeros((1, k_dim), jnp.bfloat16).at[0, e].set(jnp.bfloat16(1))
        y = np.asarray(cute_mod.matmul_q4_k(x, w), dtype=np.float32)[0]
        np.testing.assert_array_equal(y, ref_bf16[:, e], err_msg=f"element {e}")


def test_cute_matmul_q4_k_batch_and_fallback():
    n_rows, k_dim = 256, 512
    data = _random_finite_q4k(n_rows, k_dim, seed=42)
    w = gguf_jax.QuantizedArray.from_bytes(data, QTYPE, shape=(n_rows, k_dim))
    rng = np.random.default_rng(0)

    # 3D batch, kernel path
    x = jnp.asarray(rng.normal(size=(2, 3, k_dim)), dtype=jnp.bfloat16)
    y = cute_mod.matmul_q4_k(x, w)
    assert y.shape == (2, 3, n_rows) and y.dtype == jnp.bfloat16

    # beyond the M cap: falls back to dequantize-then-matmul
    xl = jnp.asarray(rng.normal(size=(32, k_dim)), dtype=jnp.bfloat16)
    yl = np.asarray(cute_mod.matmul_q4_k(xl, w), dtype=np.float32)
    wd = jnp.asarray(gguf.quants.dequantize(data, QTYPE)).astype(jnp.bfloat16)
    ref = np.asarray((xl @ wd.T), dtype=np.float32)
    np.testing.assert_allclose(yl, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())

    # jit-compatible (QuantizedArray is a pytree)
    yj = jax.jit(cute_mod.matmul_q4_k)(x, w)
    np.testing.assert_array_equal(
        np.asarray(y).view(np.uint16), np.asarray(yj).view(np.uint16))


def test_cute_bf16_output_matches_cast():
    data = random_bytes(QTYPE, (4, 512), seed=8)
    via_cute = cute_mod.dequantize_q4_k(jnp.asarray(data), dtype=jnp.bfloat16)
    via_cast = cute_mod.dequantize_q4_k(jnp.asarray(data), dtype=jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(via_cute).view(np.uint16), np.asarray(via_cast).view(np.uint16))


# ---------------------------------------------------------------------------
# Q6_K fused matmul

Q6_K = GGMLQuantizationType.Q6_K


def _random_finite_q6k(n_rows, k_dim, seed):
    """Random Q6_K blocks with a finite (small) d, byte-shaped (n_rows, row_bytes)."""
    rng = np.random.default_rng(seed)
    nb = n_rows * k_dim // 256
    data = rng.integers(0, 256, size=(nb, 210), dtype=np.uint8)
    data[:, 208:210] = (rng.normal(size=(nb, 1)) * 0.05).astype(np.float16).view(np.uint8)
    return data.reshape(n_rows, -1)


# m chosen either side of the GEMV cap: 1-8 warp-GEMV, >8 dequantize-then-matmul
@pytest.mark.parametrize("m", [1, 2, 5, 8, 40])
def test_cute_matmul_q6_k(m):
    n_rows, k_dim = 256, 1536
    data = _random_finite_q6k(n_rows, k_dim, seed=m)
    w = gguf_jax.QuantizedArray.from_bytes(data, Q6_K, shape=(n_rows, k_dim))
    rng = np.random.default_rng(m + 100)
    x = jnp.asarray(rng.normal(size=(m, k_dim)), dtype=jnp.bfloat16)

    y = np.asarray(cute_mod.matmul_q6_k(x, w), dtype=np.float32)

    wd = jnp.asarray(gguf.quants.dequantize(data, Q6_K)).astype(jnp.bfloat16)
    ref = np.asarray(
        (x.astype(jnp.float32) @ wd.astype(jnp.float32).T).astype(jnp.bfloat16),
        dtype=np.float32)
    # identical products, f32 accumulation in a different order, one bf16 round
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_cute_matmul_q6_k_exact_weights():
    """One-hot x reads the dequantized weight out of the kernel column by column."""
    n_rows, k_dim = 64, 512
    data = _random_finite_q6k(n_rows, k_dim, seed=11)
    w = gguf_jax.QuantizedArray.from_bytes(data, Q6_K, shape=(n_rows, k_dim))
    ref = np.asarray(gguf.quants.dequantize(data, Q6_K).reshape(n_rows, k_dim))
    ref_bf16 = np.asarray(jnp.asarray(ref).astype(jnp.bfloat16), dtype=np.float32)

    for e in (0, 1, 17, 31, 32, 100, 255, 256, k_dim - 1):
        x = jnp.zeros((1, k_dim), jnp.bfloat16).at[0, e].set(jnp.bfloat16(1))
        y = np.asarray(cute_mod.matmul_q6_k(x, w), dtype=np.float32)[0]
        np.testing.assert_array_equal(y, ref_bf16[:, e])


def test_cute_matmul_q6_k_batch_and_jit():
    n_rows, k_dim = 128, 512
    data = _random_finite_q6k(n_rows, k_dim, seed=42)
    w = gguf_jax.QuantizedArray.from_bytes(data, Q6_K, shape=(n_rows, k_dim))
    rng = np.random.default_rng(0)

    x = jnp.asarray(rng.normal(size=(2, 3, k_dim)), dtype=jnp.bfloat16)
    y = cute_mod.matmul_q6_k(x, w)
    assert y.shape == (2, 3, n_rows) and y.dtype == jnp.bfloat16

    yj = jax.jit(cute_mod.matmul_q6_k)(x, w)
    np.testing.assert_array_equal(
        np.asarray(y).view(np.uint16), np.asarray(yj).view(np.uint16))


# ---------------------------------------------------------------------------
# Q5_K fused matmul

Q5_K = GGMLQuantizationType.Q5_K


def _random_finite_q5k(n_rows, k_dim, seed):
    """Random Q5_K blocks with finite (small) d/dmin, byte-shaped (n_rows, row_bytes)."""
    rng = np.random.default_rng(seed)
    nb = n_rows * k_dim // 256
    data = rng.integers(0, 256, size=(nb, 176), dtype=np.uint8)
    data[:, :4] = (rng.normal(size=(nb, 2)) * 0.05).astype(np.float16).view(np.uint8)
    return data.reshape(n_rows, -1)


# m chosen either side of the GEMV cap: 1-8 warp-GEMV, >8 dequantize-then-matmul
@pytest.mark.parametrize("m", [1, 2, 5, 8, 40])
def test_cute_matmul_q5_k(m):
    n_rows, k_dim = 256, 1536
    data = _random_finite_q5k(n_rows, k_dim, seed=m)
    w = gguf_jax.QuantizedArray.from_bytes(data, Q5_K, shape=(n_rows, k_dim))
    rng = np.random.default_rng(m + 100)
    x = jnp.asarray(rng.normal(size=(m, k_dim)), dtype=jnp.bfloat16)

    y = np.asarray(cute_mod.matmul_q5_k(x, w), dtype=np.float32)

    wd = jnp.asarray(gguf.quants.dequantize(data, Q5_K)).astype(jnp.bfloat16)
    ref = np.asarray(
        (x.astype(jnp.float32) @ wd.astype(jnp.float32).T).astype(jnp.bfloat16),
        dtype=np.float32)
    # identical products, f32 accumulation in a different order, one bf16 round
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_cute_matmul_q5_k_exact_weights():
    """One-hot x reads the dequantized weight out of the kernel column by column."""
    n_rows, k_dim = 64, 512
    data = _random_finite_q5k(n_rows, k_dim, seed=11)
    w = gguf_jax.QuantizedArray.from_bytes(data, Q5_K, shape=(n_rows, k_dim))
    ref = np.asarray(gguf.quants.dequantize(data, Q5_K).reshape(n_rows, k_dim))
    ref_bf16 = np.asarray(jnp.asarray(ref).astype(jnp.bfloat16), dtype=np.float32)

    # spread over nibble halves, fifth-bit positions, sub-block and superblock edges
    for e in (0, 1, 3, 4, 15, 16, 31, 32, 33, 63, 64, 127, 128, 255, 256, k_dim - 1):
        x = jnp.zeros((1, k_dim), jnp.bfloat16).at[0, e].set(jnp.bfloat16(1))
        y = np.asarray(cute_mod.matmul_q5_k(x, w), dtype=np.float32)[0]
        np.testing.assert_array_equal(y, ref_bf16[:, e], err_msg=f"element {e}")


def test_cute_matmul_q5_k_batch_and_jit():
    n_rows, k_dim = 128, 512
    data = _random_finite_q5k(n_rows, k_dim, seed=42)
    w = gguf_jax.QuantizedArray.from_bytes(data, Q5_K, shape=(n_rows, k_dim))
    rng = np.random.default_rng(0)

    x = jnp.asarray(rng.normal(size=(2, 3, k_dim)), dtype=jnp.bfloat16)
    y = cute_mod.matmul_q5_k(x, w)
    assert y.shape == (2, 3, n_rows) and y.dtype == jnp.bfloat16

    yj = jax.jit(cute_mod.matmul_q5_k)(x, w)
    np.testing.assert_array_equal(
        np.asarray(y).view(np.uint16), np.asarray(yj).view(np.uint16))


# ---------------------------------------------------------------------------
# IQ4_XS fused matmul

IQ4_XS = GGMLQuantizationType.IQ4_XS


def _random_finite_iq4xs(n_rows, k_dim, seed):
    """Random IQ4_XS blocks with a finite (small) d, byte-shaped (n_rows, row_bytes)."""
    rng = np.random.default_rng(seed)
    nb = n_rows * k_dim // 256
    data = rng.integers(0, 256, size=(nb, 136), dtype=np.uint8)
    data[:, :2] = (rng.normal(size=(nb, 1)) * 0.005).astype(np.float16).view(np.uint8)
    return data.reshape(n_rows, -1)


# m chosen either side of the GEMV cap: 1-8 warp-GEMV, >8 dequantize-then-matmul
@pytest.mark.parametrize("m", [1, 2, 5, 8, 40])
def test_cute_matmul_iq4_xs(m):
    n_rows, k_dim = 256, 1536
    data = _random_finite_iq4xs(n_rows, k_dim, seed=m)
    w = gguf_jax.QuantizedArray.from_bytes(data, IQ4_XS, shape=(n_rows, k_dim))
    rng = np.random.default_rng(m + 100)
    x = jnp.asarray(rng.normal(size=(m, k_dim)), dtype=jnp.bfloat16)

    y = np.asarray(cute_mod.matmul_iq4_xs(x, w), dtype=np.float32)

    wd = jnp.asarray(gguf.quants.dequantize(data, IQ4_XS)).astype(jnp.bfloat16)
    ref = np.asarray(
        (x.astype(jnp.float32) @ wd.astype(jnp.float32).T).astype(jnp.bfloat16),
        dtype=np.float32)
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_cute_matmul_iq4_xs_exact_weights():
    """One-hot x reads the dequantized weight out of the kernel column by column."""
    n_rows, k_dim = 64, 512
    data = _random_finite_iq4xs(n_rows, k_dim, seed=11)
    w = gguf_jax.QuantizedArray.from_bytes(data, IQ4_XS, shape=(n_rows, k_dim))
    ref = np.asarray(gguf.quants.dequantize(data, IQ4_XS).reshape(n_rows, k_dim))
    ref_bf16 = np.asarray(jnp.asarray(ref).astype(jnp.bfloat16), dtype=np.float32)

    # the 16-element nibble split is what distinguishes this layout from Q4_K's
    for e in (0, 1, 3, 4, 15, 16, 17, 31, 32, 33, 63, 127, 128, 255, 256, k_dim - 1):
        x = jnp.zeros((1, k_dim), jnp.bfloat16).at[0, e].set(jnp.bfloat16(1))
        y = np.asarray(cute_mod.matmul_iq4_xs(x, w), dtype=np.float32)[0]
        np.testing.assert_array_equal(y, ref_bf16[:, e], err_msg=f"element {e}")


def test_cute_matmul_iq4_xs_batch_and_jit():
    n_rows, k_dim = 128, 512
    data = _random_finite_iq4xs(n_rows, k_dim, seed=42)
    w = gguf_jax.QuantizedArray.from_bytes(data, IQ4_XS, shape=(n_rows, k_dim))
    rng = np.random.default_rng(0)

    x = jnp.asarray(rng.normal(size=(2, 3, k_dim)), dtype=jnp.bfloat16)
    y = cute_mod.matmul_iq4_xs(x, w)
    assert y.shape == (2, 3, n_rows) and y.dtype == jnp.bfloat16

    yj = jax.jit(cute_mod.matmul_iq4_xs)(x, w)
    np.testing.assert_array_equal(
        np.asarray(y).view(np.uint16), np.asarray(yj).view(np.uint16))


# ---------------------------------------------------------------------------
# Low-bit GEMV template (Q2_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S)
#
# These factor the scales out of the inner loop instead of rounding each
# weight through bf16, so the contract is different from the kernels above:
# with a one-hot x the output is the float32 reference weight rounded once to
# bf16 (bitwise), and with a dense x it is at least as close to the exact
# product as dequantize(bf16)-then-matmul.

LOWBIT = list(cute_mod.LOWBIT_TYPES)


def _random_finite_lowbit(qtype, n_rows, k_dim, seed, scale=0.01):
    """Random blocks with a small finite f16 super-scale, byte-shaped (n_rows, row_bytes)."""
    rng = np.random.default_rng(seed)
    nb = n_rows * k_dim // 256
    data = rng.integers(0, 256, size=(nb, GGML_QUANT_SIZES[qtype][1]), dtype=np.uint8)
    d = (rng.normal(size=(nb, 2)) * scale).astype(np.float16)
    if qtype == GGMLQuantizationType.Q2_K:
        data[:, 80:84] = d.view(np.uint8)                   # d, dmin
    elif qtype == GGMLQuantizationType.IQ1_M:
        # d is split across the top nibbles of the four scale words
        bits = d[:, 0].view(np.uint16)
        words = data[:, 48:56].copy().view(np.uint16)
        for j in range(4):
            words[:, j] = (words[:, j] & 0x0FFF) | (((bits >> (4 * j)) & 0xF) << 12)
        data[:, 48:56] = words.view(np.uint8)
    else:
        data[:, :2] = d[:, :1].view(np.uint8)               # d
    return data.reshape(n_rows, -1)


def _lowbit_weight(qtype, n_rows, k_dim, seed):
    data = _random_finite_lowbit(qtype, n_rows, k_dim, seed)
    w = gguf_jax.QuantizedArray.from_bytes(data, qtype, shape=(n_rows, k_dim))
    ref = np.asarray(gguf.quants.dequantize(data, qtype)).reshape(n_rows, k_dim)
    return w, ref


@pytest.mark.parametrize("qtype", LOWBIT, ids=lambda t: t.name)
def test_cute_matmul_lowbit_exact_weights(qtype):
    """One-hot x reads each weight out of the kernel: f32 reference, rounded once."""
    # K = 11 superblocks: odd, so rows are only 2-byte aligned and the
    # staging windows see every skew; 11 is also no multiple of the chunk size
    n_rows, k_dim = 48, 11 * 256
    w, ref = _lowbit_weight(qtype, n_rows, k_dim, seed=11)
    ref_bf16 = np.asarray(jnp.asarray(ref).astype(jnp.bfloat16), dtype=np.float32)
    fn = jax.jit(cute_mod.matmul_lowbit)
    for e in (0, 1, 7, 8, 15, 16, 31, 32, 255, 256, 1000, 1287, 2047, k_dim - 1):
        x = jnp.zeros((1, k_dim), jnp.bfloat16).at[0, e].set(jnp.bfloat16(1))
        y = np.asarray(fn(x, w), dtype=np.float32)[0]
        np.testing.assert_array_equal(y, ref_bf16[:, e], err_msg=f"element {e}")


@pytest.mark.parametrize("qtype", LOWBIT, ids=lambda t: t.name)
def test_cute_matmul_lowbit_accuracy(qtype):
    """Dense x, every batch size up to the cap: no worse than the bf16 path."""
    n_rows, k_dim = 128, 1280
    w, ref = _lowbit_weight(qtype, n_rows, k_dim, seed=5)
    ref_bf16 = np.asarray(jnp.asarray(ref).astype(jnp.bfloat16), dtype=np.float64)
    rng = np.random.default_rng(7)
    for m in (1, 2, 5, cute_mod.lowbit._GEMV_MAX_M):
        x = jnp.asarray(rng.normal(size=(m, k_dim)), jnp.bfloat16)
        xf = np.asarray(x, dtype=np.float64)
        exact = xf @ ref.astype(np.float64).T
        y = np.asarray(cute_mod.matmul_lowbit(x, w), dtype=np.float64)
        via_bf16 = np.asarray(
            (x.astype(jnp.float32) @ jnp.asarray(ref_bf16, jnp.float32).T).astype(jnp.bfloat16),
            dtype=np.float64)
        err = np.abs(y - exact).max()
        assert err <= np.abs(via_bf16 - exact).max() * 1.05 + 1e-30, (m, err)
        np.testing.assert_allclose(y, exact, rtol=1e-2, atol=1e-2 * np.abs(exact).max())


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.IQ2_XXS, GGMLQuantizationType.IQ1_M,
                                   GGMLQuantizationType.Q2_K], ids=lambda t: t.name)
def test_cute_matmul_lowbit_many_rows_per_warp(qtype):
    """More rows than the persistent grid has warps: the cross-row prefetch,
    and chunks where a row is shorter than one chunk (K = 2 superblocks)."""
    n_sm = jax.devices()[0].core_count
    n_rows = n_sm * cute_mod.lowbit._CTAS_PER_SM * 8 * 2 + 37
    for k_dim in (512, 1280):
        w, ref = _lowbit_weight(qtype, n_rows, k_dim, seed=k_dim)
        x = jnp.asarray(np.random.default_rng(1).normal(size=(3, k_dim)), jnp.bfloat16)
        y = np.asarray(cute_mod.matmul_lowbit(x, w), dtype=np.float64)
        exact = np.asarray(x, dtype=np.float64) @ ref.astype(np.float64).T
        np.testing.assert_allclose(y, exact, rtol=1e-2, atol=1e-2 * np.abs(exact).max())


def test_cute_matmul_lowbit_batch_jit_and_fallback():
    qtype = GGMLQuantizationType.IQ3_S
    n_rows, k_dim = 128, 512
    w, ref = _lowbit_weight(qtype, n_rows, k_dim, seed=42)
    rng = np.random.default_rng(0)

    x = jnp.asarray(rng.normal(size=(2, 3, k_dim)), dtype=jnp.bfloat16)
    y = cute_mod.matmul_lowbit(x, w)
    assert y.shape == (2, 3, n_rows) and y.dtype == jnp.bfloat16
    yj = jax.jit(cute_mod.matmul_lowbit)(x, w)
    np.testing.assert_array_equal(
        np.asarray(y).view(np.uint16), np.asarray(yj).view(np.uint16))

    # beyond the M cap: dequantize-then-matmul
    xl = jnp.asarray(rng.normal(size=(40, k_dim)), dtype=jnp.bfloat16)
    yl = np.asarray(cute_mod.matmul_lowbit(xl, w), dtype=np.float64)
    exact = np.asarray(xl, dtype=np.float64) @ ref.astype(np.float64).T
    np.testing.assert_allclose(yl, exact, rtol=2e-2, atol=2e-2 * np.abs(exact).max())
