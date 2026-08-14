"""Bitwise tests for the CuTe DSL kernels (skipped without GPU + cute deps)."""

import gguf
import jax
import numpy as np
import pytest
from gguf.constants import GGMLQuantizationType

import gguf_jax
import jax.numpy as jnp
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


def test_cute_bf16_output_matches_cast():
    data = random_bytes(QTYPE, (4, 512), seed=8)
    via_cute = cute_mod.dequantize_q4_k(jnp.asarray(data), dtype=jnp.bfloat16)
    via_cast = cute_mod.dequantize_q4_k(jnp.asarray(data), dtype=jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(via_cute).view(np.uint16), np.asarray(via_cast).view(np.uint16))
