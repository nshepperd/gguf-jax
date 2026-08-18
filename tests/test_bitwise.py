"""Bitwise-exactness tests: gguf_jax dequantization vs the gguf-py reference.

Two data sources per qtype:

- random bytes: every block field takes arbitrary values (structurally valid
  by construction, since every bit pattern decodes). This exercises the full
  bit-manipulation pipeline, including non-finite f16 scales.
- realistic data: random floats round-tripped through the reference
  quantizer, for the qtypes gguf-py can quantize.

"Bitwise" means the raw float32 bit patterns match, which is stricter than
allclose: NaN payloads, signed zeros and subnormals must agree too.
"""

import gguf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

import gguf_jax

QTYPES = gguf_jax.supported_types()


def assert_bitwise_equal(ours: np.ndarray, ref: np.ndarray, qtype):
    """Assert identical float32 bit patterns.

    The one concession: where BOTH sides are NaN, payload/sign bits are not
    compared. Such NaNs only arise from non-finite f16 block scales, which
    valid GGUF files never contain, and IEEE 754 leaves NaN sign/payload
    propagation through arithmetic unspecified (XLA fusion and numpy disagree
    on the sign bit). Finite values, infinities, signed zeros and subnormals
    are all compared exactly.
    """
    assert ours.shape == ref.shape
    assert ours.dtype == ref.dtype == np.float32
    # np.dtype(...) rather than the bare scalar type: the latter picks a
    # `.view()` overload whose result type confuses type checkers.
    ours_bits = ours.view(np.dtype(np.uint32))
    ref_bits = ref.view(np.dtype(np.uint32))
    mismatch = (ours_bits != ref_bits) & ~(np.isnan(ours) & np.isnan(ref))
    if mismatch.any():
        idx = tuple(a[0] for a in np.nonzero(mismatch))
        raise AssertionError(
            f"{qtype.name}: {mismatch.sum()}/{mismatch.size} mismatched values; "
            f"first at {idx}: ours={ours[idx]!r} (0x{ours_bits[idx]:08x}) "
            f"ref={ref[idx]!r} (0x{ref_bits[idx]:08x})")


def random_bytes(qtype, shape, seed):
    """Random byte-shaped quantized data with the logical shape ``shape``."""
    byte_shape = gguf.quants.quant_shape_to_byte_shape(shape, qtype)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=byte_shape, dtype=np.uint8)


def logical_shapes(qtype):
    block_size, _ = GGML_QUANT_SIZES[qtype]
    n = max(block_size, 256)
    return [(n,), (4, 4 * n), (2, 3, n)]


@pytest.mark.parametrize("qtype", QTYPES, ids=lambda t: t.name)
def test_random_bytes_bitwise(qtype):
    for i, shape in enumerate(logical_shapes(qtype)):
        data = random_bytes(qtype, shape, seed=hash((qtype.name, i)) % 2**32)
        ref = gguf.quants.dequantize(data, qtype)
        ours = np.asarray(gguf_jax.dequantize(jnp.asarray(data), qtype))
        assert_bitwise_equal(ours, ref, qtype)


@pytest.mark.parametrize("qtype", QTYPES, ids=lambda t: t.name)
def test_random_bytes_bitwise_jit(qtype):
    shape = logical_shapes(qtype)[1]
    data = random_bytes(qtype, shape, seed=1234)
    ref = gguf.quants.dequantize(data, qtype)
    fn = jax.jit(lambda x: gguf_jax.dequantize(x, qtype))
    ours = np.asarray(fn(jnp.asarray(data)))
    assert_bitwise_equal(ours, ref, qtype)


@pytest.mark.parametrize("qtype", QTYPES, ids=lambda t: t.name)
def test_quantized_floats_bitwise(qtype):
    """Round realistic float data through the reference quantizer, then compare."""
    rng = np.random.default_rng(0)
    shape = (8, max(GGML_QUANT_SIZES[qtype][0], 256))
    # varied magnitudes so per-block scales differ widely
    values = rng.normal(size=shape).astype(np.float32)
    values *= 10.0 ** rng.integers(-8, 8, size=(shape[0], 1)).astype(np.float32)
    values[0, :] = 0.0  # all-zero block edge case
    try:
        data: np.ndarray = gguf.quants.quantize(values, qtype)
    except NotImplementedError:
        pytest.skip(f"gguf-py cannot quantize {qtype.name}")
    if data.dtype != np.uint8:  # F16/F32 come back as native floats
        data = np.ascontiguousarray(data).view(np.dtype(np.uint8))
    ref = gguf.quants.dequantize(data, qtype)
    ours = np.asarray(gguf_jax.dequantize(jnp.asarray(data), qtype))
    assert_bitwise_equal(ours, ref, qtype)


def test_supported_types_match_reference():
    """We implement exactly the set of types the reference can dequantize."""
    ref_types = set(gguf.quants._type_traits.keys()) | {
        GGMLQuantizationType.F16,
        GGMLQuantizationType.F32,
    }
    assert set(QTYPES) == ref_types


def test_bad_shape_raises():
    data = np.zeros((3,), dtype=np.uint8)  # not a multiple of Q8_0's 34-byte blocks
    with pytest.raises(ValueError):
        gguf_jax.dequantize(jnp.asarray(data), GGMLQuantizationType.Q8_0)
