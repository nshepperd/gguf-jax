"""The dispatching matmul wrapper.

Split in two: the parts that hold with no GPU and no cute extra (the
dequantize fallback, the blocking, the argument checking), and the parts that
need real kernels (that dispatch picks them, and agrees with the fallback).
"""
import gguf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gguf.constants import GGML_QUANT_SIZES
from gguf.constants import GGMLQuantizationType as QT

import gguf_jax
from gguf_jax import _matmul as mm

ON_GPU = jax.devices()[0].platform == "gpu"


def make_weight(n, k, qtype, seed=0):
    """Random blocks with small finite super-scales, so nothing overflows."""
    rng = np.random.default_rng(seed)
    bpb = GGML_QUANT_SIZES[qtype][1]
    nb = n * k // 256
    data = rng.integers(0, 256, size=(nb, bpb), dtype=np.uint8)
    scale = (rng.normal(size=(nb, 2)) * 0.05).astype(np.float16).view(np.uint8)
    if qtype in (QT.Q4_K, QT.Q5_K):
        data[:, :4] = scale                 # d, dmin
    elif qtype == QT.Q6_K:
        data[:, 208:210] = scale[:, :2]     # d
    elif qtype == QT.IQ4_XS:
        data[:, :2] = scale[:, :2]          # d
    return gguf_jax.QuantizedArray.from_bytes(
        data.reshape(n, -1), qtype, shape=(n, k))


def reference(x, w):
    """x @ w.T through the plain dequantize path, in float64."""
    wd = np.asarray(w.dequantize(jnp.bfloat16), dtype=np.float64)
    return np.asarray(x, dtype=np.float64) @ wd.T


# --- fallback behaviour, no kernels required ------------------------------

@pytest.mark.parametrize("m", [1, 4, 512])
def test_dequant_matmul_matches_plain(m):
    w = make_weight(128, 512, QT.Q4_K, seed=m)
    x = jnp.asarray(np.random.default_rng(m).normal(size=(m, 512)), jnp.bfloat16)
    got = np.asarray(mm.dequant_matmul(x, w), dtype=np.float64)
    ref = reference(x, w)
    np.testing.assert_allclose(got, ref, rtol=2e-2,
                               atol=2e-2 * np.abs(ref).max())


def test_blocking_does_not_change_the_result():
    """Splitting across output rows is an implementation detail, not a result."""
    w = make_weight(256, 512, QT.Q4_K, seed=3)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(4, 512)), jnp.bfloat16)
    whole = mm.dequant_matmul(x, w, block_bytes=1 << 30)
    assert mm._block_count(256, 512, 1 << 30) == 1
    for budget in (1 << 18, 1 << 16):
        assert mm._block_count(256, 512, budget) > 1
        split = mm.dequant_matmul(x, w, block_bytes=budget)
        np.testing.assert_array_equal(
            np.asarray(whole).view(np.uint16), np.asarray(split).view(np.uint16))


def test_block_count_divides_evenly():
    for out_rows in (128, 256, 1024, 12288, 151936):
        for budget in (1 << 20, 1 << 24, 1 << 28):
            n = mm._block_count(out_rows, 4096, budget)
            assert out_rows % n == 0, (out_rows, budget, n)
            assert 4 * (out_rows // n) * 4096 <= budget or n == out_rows


def test_argument_checking():
    w = make_weight(128, 512, QT.Q4_K)
    x = jnp.zeros((2, 512), jnp.bfloat16)
    with pytest.raises(TypeError, match="QuantizedArray"):
        mm.matmul(x, jnp.zeros((128, 512)))
    with pytest.raises(ValueError, match="contraction mismatch"):
        mm.matmul(jnp.zeros((2, 256), jnp.bfloat16), w)


def test_batched_leading_dims_are_preserved():
    w = make_weight(128, 512, QT.Q4_K, seed=9)
    x = jnp.asarray(np.random.default_rng(9).normal(size=(2, 3, 512)), jnp.bfloat16)
    assert mm.matmul(x, w).shape == (2, 3, 128)


# --- dispatch, needs the cute kernels -------------------------------------

pytestmark_gpu = pytest.mark.skipif(
    not ON_GPU, reason="fused kernels need a GPU")


@pytestmark_gpu
def test_fused_types_are_registered():
    pytest.importorskip("gguf_jax.cute")
    assert set(mm.fused_types()) >= {QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.IQ4_XS}


@pytestmark_gpu
@pytest.mark.parametrize("qtype", [QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.IQ4_XS])
def test_fused_agrees_with_fallback(qtype):
    """Whichever route dispatch picks, the answer is the same one."""
    pytest.importorskip("gguf_jax.cute")
    w = make_weight(256, 512, qtype, seed=hash(qtype) % 1000)
    for m in (1, 2, 8, 64, 512):
        x = jnp.asarray(np.random.default_rng(m).normal(size=(m, 512)), jnp.bfloat16)
        got = np.asarray(mm.matmul(x, w), dtype=np.float64)
        ref = reference(x, w)
        np.testing.assert_allclose(
            got, ref, rtol=2e-2, atol=2e-2 * np.abs(ref).max(),
            err_msg=f"{qtype.name} at M={m}")


@pytestmark_gpu
@pytest.mark.parametrize("qtype", [QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.IQ4_XS])
def test_every_fused_type_has_a_usable_batch_range(qtype):
    pytest.importorskip("gguf_jax.cute")
    w = make_weight(256, 512, qtype)
    assert mm.fused_batch_limit(w) >= 1, (
        f"{qtype.name} has a kernel but dispatch would never reach it")


def test_a_type_without_a_kernel_reports_no_fused_range():
    """F16 has no fused kernel, so the cap must be 0 and dispatch must not try."""
    w = gguf_jax.QuantizedArray.from_bytes(
        np.zeros((4, 8), np.uint8), QT.F16, shape=(4, 4))
    assert mm.fused_batch_limit(w) == 0


@pytestmark_gpu
def test_float32_x_takes_the_fallback():
    """The kernels are bfloat16-only; a float32 x must not be handed to them."""
    pytest.importorskip("gguf_jax.cute")
    w = make_weight(256, 512, QT.Q4_K, seed=77)
    x = jnp.asarray(np.random.default_rng(77).normal(size=(1, 512)), jnp.float32)
    got = np.asarray(mm.matmul(x, w), dtype=np.float64)
    ref = reference(x.astype(jnp.bfloat16), w)
    np.testing.assert_allclose(got, ref, rtol=5e-2, atol=5e-2 * np.abs(ref).max())
