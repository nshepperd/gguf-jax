# Bitwise correctness

The pure-JAX dequantization kernels mirror the numpy reference in
`gguf.quants` (gguf-py, from llama.cpp) operation-for-operation, so their
float32 output is bitwise identical to it.

## What the tests check

`tests/test_bitwise.py` runs every supported qtype against
`gguf.quants.dequantize`, both eagerly and under `jit`, on two data sources:

- **fully random block bytes** — every bit pattern decodes, so this covers all
  field encodings, including non-finite f16 block scales;
- **realistic data** — random floats round-tripped through the reference
  quantizer, for the qtypes gguf-py can quantize.

Equality is on raw float32 bit patterns, which is stricter than `allclose`:
signed zeros, infinities and subnormals must all agree.

The one carve-out is NaN. Where *both* sides produce NaN, sign and payload bits
are not compared. Such NaNs arise only from non-finite f16 block scales, which
valid GGUF files never contain, and IEEE 754 leaves NaN propagation through
arithmetic unspecified — XLA fusion and numpy disagree on the sign bit.

## Two places that need care

**f16 → f32 widening.** XLA's convert canonicalizes NaN payloads; numpy
preserves them. `_f16_to_f32` widens the non-finite cases by explicit bit
manipulation and uses the hardware convert for the rest.

**MXFP4 subnormals.** XLA:CPU compiles with flush-to-zero, so a product that
lands in the subnormal range is zeroed. For an E8M0 scale below 2 the product
`|q| * 2**(e-128)` has at most 4 significant bits and is exactly
representable, so the kernel constructs its bits with integer ops, which FTZ
cannot touch.

## Kernel contract

Registered dequantization kernels have the signature

```
fn(blocks_u8: uint8[n_blocks, type_size], dtype) -> dtype[n_blocks, block_size]
```

and must return the bitwise-exact float32 decode rounded *once* to `dtype`.
The built-in kernels compute in float32 and let the cast fuse into the
surrounding XLA computation; a custom kernel may instead compute the rounding
itself and write bfloat16 directly, skipping the float32 intermediate.

```python
gguf_jax.register_dequant(qtype, my_kernel, override=True)
```

The same test battery is applied to replacement kernels
(`tests/test_cute.py`).
