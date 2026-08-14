# gguf-jax

Load GGUF-quantized models (llama.cpp quants) directly in JAX and dequantize
on the fly during the forward pass — the same trick as
[bitsandbytes-jax](../bitsandbytes-jax) for bnb 4-bit, but for GGUF files.

Tensors are held as raw uint8 block data in a `QuantizedArray` pytree and
decoded by pure-JAX kernels whose float32 output is **bitwise identical** to
the `gguf.quants` numpy reference implementation from llama.cpp (gguf-py).

```python
import jax.numpy as jnp
import gguf_jax

model = gguf_jax.load_gguf("llama-3.2-1b-Q4_K_M.gguf", dtype=jnp.bfloat16)
print(model.metadata["general.architecture"])

w = model.tensors["blk.0.attn_q.weight"]   # QuantizedArray(Q4_K, shape=(2048, 2048), ...)

# inside your (jitted) forward pass:
y = x @ w.dequantize().T                   # bfloat16, decoded on the fly
```

`QuantizedArray` is a registered pytree (the uint8 payload is the only leaf;
qtype/shape/dtype are static), so it composes with `jax.jit`, `tree_map`,
checkpointing utilities, etc.

## Supported quant types

Everything the gguf-py reference can dequantize:

- passthrough: `F32`, `F16`, `BF16` (+ `F64`/`I8`/`I16`/`I32`/`I64` kept native)
- legacy: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- K-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- i-quants: `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`
- ternary: `TQ1_0`, `TQ2_0`
- microscaling: `MXFP4`, `NVFP4`

## API

- `load_gguf(path, dtype=jnp.bfloat16, tensor_filter=None) -> GGUFFile` —
  metadata dict + `dict[str, QuantizedArray]`. Payloads are uploaded to the
  default JAX device as-is (no dequantization at load time); use
  `jax.default_device(...)` to control placement.
- `QuantizedArray.dequantize(dtype=None)` — decode to a dense array
  (computes in float32, then casts to `dtype`, default `self.dtype`).
- `QuantizedArray.from_bytes(data, qtype, shape=None, dtype=...)` — wrap raw
  quantized bytes you got from somewhere else.
- `dequantize(data, qtype)` — functional form on byte-shaped uint8 arrays,
  drop-in equivalent of `gguf.quants.dequantize`.
- `quantize(array, qtype)` — host-side convenience wrapper around the gguf-py
  reference quantizer (only the types gguf-py can quantize).

## Bitwise correctness

The test suite (`tests/test_bitwise.py`) checks every supported qtype against
`gguf.quants.dequantize`, eagerly and under `jit`, on:

- fully random block bytes (every bit pattern decodes, so this covers all
  field encodings, including non-finite f16 scales), and
- realistic data round-tripped through the reference quantizer.

Equality is on raw float32 bit patterns — signed zeros, infinities and
subnormals included. The one carve-out: where *both* sides produce NaN
(possible only with non-finite block scales, which valid GGUF files never
contain), NaN sign/payload bits are not compared, since IEEE 754 leaves NaN
propagation through arithmetic unspecified and XLA fusion and numpy disagree.

Two implementation notes for exactness:

- f16→f32 widening uses explicit bit manipulation for the non-finite cases
  (XLA's convert canonicalizes NaN payloads; numpy preserves them).
- XLA:CPU compiles with flush-to-zero, so the MXFP4 kernel constructs exact
  subnormal results with integer ops when the E8M0 scale is subnormal.

## CuTe DSL kernels

The pure-JAX kernels are the reference path: dequantization is expressed as
plain XLA ops (bit twiddling + gathers + multiplies). Registered kernels have
the contract `fn(blocks_u8, dtype) -> dtype[n_blocks, block_size]` — the
bitwise-exact float32 decode rounded once to `dtype` — so a custom kernel can
write bfloat16 directly without materializing the float32 intermediate:

```python
gguf_jax.register_dequant(qtype, my_kernel, override=True)
```

`gguf_jax.cute` (optional `cute` dependency group: nvidia-cutlass-dsl,
jax-tvm-ffi, and [cutejax](../cutedsl-jax)) provides a Q4_K kernel written in
CuTe DSL as the demonstrator; `gguf_jax.cute.register()` swaps it in behind
the normal `QuantizedArray` API, and the same bitwise test battery applies to
it. Layout: one CTA per 8 superblocks, 32 lanes per superblock, 8 elements
per lane with warp-coalesced stores; the f16/uint32 fields are read through
`cute.recast_ptr` views of the one uint8 buffer.

Measured on an RTX 5070 Ti (`bench/bench_dequant.py`, (4096, 14336) weight,
dequantize only, effective bandwidth = bytes in + out over wall time):

| kernel      | f32 out            | bf16 out           |
|-------------|--------------------|--------------------|
| pure XLA    | 468 µs, 572 GB/s   | 274 µs, 549 GB/s   |
| cute Q4_K   | 433 µs, 619 GB/s   | 224 µs, 671 GB/s   |

The remaining headroom (~896 GB/s peak) is scattered small stores; fused
dequant-matmul, where the dense matrix never reaches HBM at all, is the real
prize and the intended next step.

## Development

```bash
uv sync
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run pytest
```

The test group pulls in `jax[cuda13]`; the allocator env vars keep JAX from
grabbing most of the VRAM up front.
