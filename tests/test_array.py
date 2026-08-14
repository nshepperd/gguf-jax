"""Tests for the QuantizedArray pytree."""

import gguf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gguf.constants import GGMLQuantizationType

import gguf_jax
from tests.test_bitwise import assert_bitwise_equal, random_bytes


def test_quantize_roundtrip():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(16, 64)).astype(np.float32)
    qa = gguf_jax.quantize(values, GGMLQuantizationType.Q8_0, dtype=jnp.float32)
    assert qa.shape == (16, 64)
    assert qa.qtype == GGMLQuantizationType.Q8_0
    ref = gguf.quants.dequantize(
        gguf.quants.quantize(values, GGMLQuantizationType.Q8_0),
        GGMLQuantizationType.Q8_0)
    assert_bitwise_equal(np.asarray(qa.dequantize()), ref, qa.qtype)


def test_dequantize_dtype_cast():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(4, 256)).astype(np.float32)
    qa = gguf_jax.quantize(values, GGMLQuantizationType.Q4_0, dtype=jnp.bfloat16)
    out = qa.dequantize()
    assert out.dtype == jnp.bfloat16
    f32 = qa.dequantize(dtype=jnp.float32)
    # the bf16 result is exactly the rounded f32 result
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), np.asarray(f32.astype(jnp.bfloat16)).view(np.uint16))
    assert qa.astype(jnp.float16).dequantize().dtype == jnp.float16


def test_pytree_and_jit():
    data = random_bytes(GGMLQuantizationType.Q4_K, (2, 512), seed=3)
    qa = gguf_jax.QuantizedArray.from_bytes(data, GGMLQuantizationType.Q4_K, dtype=jnp.float32)
    assert qa.shape == (2, 512)

    # tree operations see exactly one leaf (the byte payload)
    leaves = jax.tree_util.tree_leaves(qa)
    assert len(leaves) == 1 and leaves[0].dtype == jnp.uint8
    qa2 = jax.tree_util.tree_map(lambda x: x, qa)
    assert qa2.qtype == qa.qtype and qa2.shape == qa.shape

    # QuantizedArray can be passed through jit as an argument
    @jax.jit
    def matvec(w: gguf_jax.QuantizedArray, x: jax.Array) -> jax.Array:
        return w.dequantize() @ x

    x = jnp.ones((512,), dtype=jnp.float32)
    ref = gguf.quants.dequantize(data, GGMLQuantizationType.Q4_K) @ np.ones(512, np.float32)
    np.testing.assert_allclose(np.asarray(matvec(qa, x)), ref, rtol=1e-6)


def test_from_bytes_infers_shape():
    data = random_bytes(GGMLQuantizationType.Q8_0, (4, 128), seed=4)
    qa = gguf_jax.QuantizedArray.from_bytes(data, GGMLQuantizationType.Q8_0)
    assert qa.shape == (4, 128)
    ref = gguf.quants.dequantize(data, GGMLQuantizationType.Q8_0)
    assert_bitwise_equal(np.asarray(qa.dequantize(dtype=jnp.float32)), ref, qa.qtype)


def test_native_integer_type():
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    qa = gguf_jax.QuantizedArray.from_bytes(values, GGMLQuantizationType.I32)
    out = qa.dequantize()
    assert out.dtype == jnp.int32  # dtype is ignored for integer tensors
    np.testing.assert_array_equal(np.asarray(out), values)


def test_bad_data_dtype_raises():
    with pytest.raises(ValueError):
        gguf_jax.QuantizedArray.from_bytes(
            np.zeros((4, 34), np.float32), GGMLQuantizationType.Q8_0)
