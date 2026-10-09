# gguf-jax

Load GGUF-quantized models (llama.cpp quants) directly in JAX and dequantize
on the fly during the forward pass — the same trick as bitsandbytes-jax for
bnb 4-bit, but for GGUF files.

```python
import jax.numpy as jnp
import gguf_jax

model = gguf_jax.load_gguf("llama-3.2-1b-Q4_K_M.gguf", dtype=jnp.bfloat16)
w = model.tensors["blk.0.attn_q.weight"]   # QuantizedArray(Q4_K, shape=(2048, 2048), ...)

# inside your (jitted) forward pass:
y = gguf_jax.matmul(x, w)                  # x @ w.T, fused kernel where one fits
y = x @ w.dequantize().T                   # or decode explicitly
```

## Features

- **Nothing is dequantized at load time.** Tensors stay as raw uint8 block data
  in a `QuantizedArray` and are decoded inside your forward pass, so only the
  quantized bytes live in device memory.
- **Bitwise-exact.** The float32 decode is bit-for-bit identical to the
  `gguf.quants` numpy reference from llama.cpp (gguf-py), for every supported
  qtype — see [docs/bitwise-correctness.md](docs/bitwise-correctness.md).
- **A pytree.** `QuantizedArray` is a registered dataclass with the uint8
  payload as its only leaf (qtype/shape/dtype are static), so it composes with
  `jax.jit`, `tree_map`, checkpointing utilities, etc.
- **Pluggable kernels.** `register_dequant` swaps in a faster decode for a
  qtype. `gguf_jax.cute` ships CuTe DSL kernels for Q4_K, Q5_K, Q6_K and
  IQ4_XS, including fused dequant-matmuls that keep the weights quantized in
  HBM, plus one fused GEMV template covering the low-bit types (Q2_K, IQ1_S,
  IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) — see
  [docs/cute-kernels.md](docs/cute-kernels.md).
- **One call to multiply by a quantized weight.** `gguf_jax.matmul(x, w)` picks
  the route: a fused kernel when one covers that qtype and batch size,
  otherwise dequantize-then-matmul, split across output rows if the weight is
  too big to materialize at once. Callers do not need to know which kernels
  exist or what each one's batch cap is.

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
  metadata dict + `dict[str, QuantizedArray]`. Payloads go to the default JAX
  device as-is; use `jax.default_device(...)` to control placement.
- `QuantizedArray.dequantize(dtype=None)` — decode to a dense array (computes
  in float32, then casts to `dtype`, default `self.dtype`).
- `QuantizedArray.from_bytes(data, qtype, shape=None, dtype=...)` — wrap raw
  quantized bytes from elsewhere.
- `dequantize(data, qtype)` — functional form on byte-shaped uint8 arrays,
  drop-in equivalent of `gguf.quants.dequantize`.
- `quantize(array, qtype)` — host-side wrapper around the gguf-py reference
  quantizer (only the types gguf-py can quantize).
- `register_dequant(qtype, fn, override=False)` — install a custom kernel.
- `matmul(x, w, block_bytes=None, force_fused=False) -> Array` — `x @ w.T`,
  dispatching over the available kernels. Prefer this to calling a
  `gguf_jax.cute.matmul_*` directly: above their batch range those fall back
  internally to dequantizing the *whole* weight, which is the allocation you
  were trying to avoid, whereas `matmul` blocks it instead.
- `dequant_matmul(x, w, block_bytes=None)` — the fallback on its own, for
  benchmarking against the fused path.
- `fused_types()` / `fused_batch_limit(w)` — which qtypes have a fused kernel
  in this installation, and the largest batch `w` can be multiplied at without
  being materialized (0 if none applies).
- `GGUF_JAX_DEQUANT_BLOCK_BYTES` — env override for how large a temporary one
  dequantize may allocate before it is split across output rows.
- `gguf_jax.cute.register()` / `gguf_jax.cute.matmul_q4_k(x, w)` / … — the
  optional CuTe DSL kernels, called directly. `matmul_lowbit(x, w)` takes any
  of `gguf_jax.cute.LOWBIT_TYPES`.

## Development

```bash
uv sync
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run pytest
```

The test group pulls in `jax[cuda13]`; the allocator env vars keep JAX from
grabbing most of the VRAM up front.
