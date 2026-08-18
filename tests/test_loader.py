"""Round-trip test: write a GGUF file with gguf-py, load it with gguf_jax."""

import gguf
import jax.numpy as jnp
import numpy as np
import pytest
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType

import gguf_jax
from tests.test_bitwise import assert_bitwise_equal, random_bytes

QUANTIZED_TENSORS = {
    "blk.0.attn_q.weight": (GGMLQuantizationType.Q4_K, (512, 256)),
    "blk.0.attn_k.weight": (GGMLQuantizationType.Q6_K, (256, 256)),
    "blk.0.ffn_up.weight": (GGMLQuantizationType.Q8_0, (128, 64)),
    "blk.0.ffn_down.weight": (GGMLQuantizationType.IQ4_XS, (2, 4, 256)),
}


@pytest.fixture(scope="module")
def gguf_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "tiny.gguf"
    writer = GGUFWriter(str(path), arch="llama")
    writer.add_name("tiny test model")
    writer.add_uint32("test.some_number", 42)
    writer.add_float32("test.some_float", 0.5)
    writer.add_array("test.some_list", [1, 2, 3])

    rng = np.random.default_rng(0)
    expected = {}

    f32 = rng.normal(size=(8, 16)).astype(np.float32)
    writer.add_tensor("output_norm.weight", f32)
    expected["output_norm.weight"] = (GGMLQuantizationType.F32, f32)

    f16 = rng.normal(size=(8, 16)).astype(np.float16)
    writer.add_tensor("blk.0.attn_norm.weight", f16)
    expected["blk.0.attn_norm.weight"] = (GGMLQuantizationType.F16, f16.astype(np.float32))

    for name, (qtype, shape) in QUANTIZED_TENSORS.items():
        data = random_bytes(qtype, shape, seed=hash(name) % 2**32)
        writer.add_tensor(name, data, raw_dtype=qtype)
        expected[name] = (qtype, gguf.quants.dequantize(data, qtype))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path, expected


def test_metadata(gguf_file):
    path, _ = gguf_file
    loaded = gguf_jax.load_gguf(str(path))
    assert loaded.metadata["general.architecture"] == "llama"
    assert loaded.metadata["general.name"] == "tiny test model"
    assert loaded.metadata["test.some_number"] == 42
    assert loaded.metadata["test.some_float"] == 0.5
    assert list(loaded.metadata["test.some_list"]) == [1, 2, 3]


def test_tensors_bitwise(gguf_file):
    path, expected = gguf_file
    loaded = gguf_jax.load_gguf(str(path), dtype=jnp.float32)
    assert set(loaded.tensors.keys()) == set(expected.keys())
    for name, (qtype, ref) in expected.items():
        qa = loaded.tensors[name]
        assert qa.qtype == qtype, name
        assert qa.shape == ref.shape, name
        assert qa.dtype == jnp.float32
        assert_bitwise_equal(np.asarray(qa.dequantize()), np.asarray(ref, dtype=np.float32), qtype)


def test_tensor_filter(gguf_file):
    path, _ = gguf_file
    loaded = gguf_jax.load_gguf(str(path), tensor_filter=lambda name: "ffn" in name)
    assert set(loaded.tensors.keys()) == {"blk.0.ffn_up.weight", "blk.0.ffn_down.weight"}


def test_default_dtype_is_bf16(gguf_file):
    path, _ = gguf_file
    loaded = gguf_jax.load_gguf(str(path))
    out = loaded.tensors["blk.0.attn_q.weight"].dequantize()
    assert out.dtype == jnp.bfloat16
