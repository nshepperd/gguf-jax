# CuTe DSL kernels

The pure-JAX kernels are the reference path: dequantization expressed as plain
XLA ops (bit twiddling + gathers + multiplies). `gguf_jax.cute` provides
hand-written CuTe DSL kernels for Q4_K (plus a fused Q6_K GEMV) as the
demonstrator that the [kernel contract](bitwise-correctness.md#kernel-contract)
is enough to swap in something faster.

It needs the optional `cute` dependency group: nvidia-cutlass-dsl, jax-tvm-ffi
and cutejax.

```python
from gguf_jax import cute
cute.register()      # Q4_K dequantize() now runs the cute kernel
```

`register()` swaps the kernel in behind the normal `QuantizedArray` API, and
the same bitwise test battery applies to it.

## Q4_K dequantization kernel

One CTA per 8 superblocks, 32 lanes per superblock, 8 elements per lane with
warp-coalesced stores. The f16 (`d`, `dmin`) pair and the packed scale words
are read through `cute.recast_ptr` views of the one uint8 buffer, so the
hardware f16→f32 convert is used instead of manual bit widening.

Because the kernel writes the requested dtype directly, bfloat16 output costs
one hardware round with no float32 intermediate.

Measured on an RTX 5070 Ti (`bench/bench_dequant.py`, (4096, 14336) weight,
dequantize only, effective bandwidth = bytes in + out over wall time):

| kernel      | f32 out            | bf16 out           |
|-------------|--------------------|--------------------|
| pure XLA    | 468 µs, 572 GB/s   | 274 µs, 549 GB/s   |
| cute Q4_K   | 433 µs, 619 GB/s   | 224 µs, 671 GB/s   |

The remaining headroom (~896 GB/s peak) is scattered small stores.

## Fused Q4_K dequant-matmul

`gguf_jax.cute.matmul_q4_k(x, w)` computes `x @ w.T` (`x` bfloat16 `(..., K)`,
`w` a Q4_K `QuantizedArray` `(N, K)`) with the weights dequantized on the fly —
the dense matrix never touches HBM. Weights are rounded through bfloat16
in-register, so values match `x @ w.dequantize(bfloat16).T` up to f32 summation
order (~1 ulp of bf16).

It dispatches on the flattened batch size M across three paths.

### M ≤ 2 — warp-GEMV (`q4_k.py`)

One warp per output row `n` of W. Lane `w` walks the row's superblocks handling
elements `{32k + w}`, dequantizes into registers, multiplies with `x[m, e]` and
accumulates in float32; a butterfly-shuffle reduction folds the 32 lanes. M is
compile-time static, so there is one specialization per batch size.

### 3 ≤ M ≤ 128 — tensor-core GEMM (`q4_k_gemm.py`)

A proper `mma.sync.m16n8k16` bf16 GEMM (SM80-class atoms, runs on SM120),
modeled on CuTeDSL's Ampere `TensorOpGemm`, with one structural change: the B
operand flows through the cp.async pipeline *quantized*.

- **A (activations)**: the standard path — cp.async into swizzled smem,
  `ldmatrix` into MMA fragments.
- **B (weights)**: cp.async copies the raw Q4_K words into smem (144 B per row
  per superblock, 3.5× less smem traffic than dense bf16); dequantization
  happens at the smem→register stage, writing bf16 straight into the MMA
  B-fragment. An identity-tensor partition supplies each fragment element's
  (n, k) coordinate, so the `mma.sync` thread-value layout never has to be
  spelled out.

Tile (bM, bN, bK) = (32, 64, 256) — bK is exactly one Q4_K superblock, so each
row's scale header is read once per k-tile. 128 threads, 3-stage cp.async
pipeline, f32 accumulate. M is dynamic (predicated loads and stores), so one
compile serves every batch size; N % 64 == 0 is required.

### M > 128 — dequantize-then-matmul

Re-reading the quantized weight once per 32-row M-tile stops paying; a single
dequant plus a dense GEMM wins.

`matmul_q4_k(x, w, force_fused=True)` forbids this fallback and keeps the
tensor-core kernel for every M, so the dense bf16 weight (2·N·K bytes) is never
materialized — useful when the weight is large and memory, not speed, is the
constraint. It costs ~1.5× at M=512 versus the fallback, and requires
N % 64 == 0 above the GEMV range.

### Benchmarks

`bench/bench_matmul.py`, same 4096×14336 weight (33MB quantized, 117MB dense):

| M   | fused (dispatch) | tc-gemm  | unfused  | dense bf16 matmul |
|-----|------------------|----------|----------|-------------------|
| 1   |  73 µs (gemv)    |  91 µs   | 379 µs   | 156 µs            |
| 4   |  91 µs           |  91 µs   | 405 µs   | 147 µs            |
| 16  |  91 µs           |  91 µs   | 389 µs   | 154 µs            |
| 32  |  92 µs           |  92 µs   | 399 µs   | 164 µs            |
| 64  | 188 µs           | 188 µs   | 398 µs   | 164 µs            |
| 128 | 366 µs           | 366 µs   | 444 µs   | 198 µs            |
| 512 | 883 µs (unfused) | 1365 µs  | 883 µs   | 643 µs            |

The tensor-core path is flat at ~91 µs through M = 32 (one M-tile): 4.3× the
unfused path, and faster than a dense bf16 matmul with resident weights up to
M ≈ 64 — the quantized bytes are simply less memory to read. Next lever: keep
W-tile reuse across M-tiles at large M (split-K / persistent CTAs) instead of
falling back.

## Fused Q6_K matmul (GEMV only)

`gguf_jax.cute.matmul_q6_k(x, w)` is the same contract for a Q6_K weight:
`x @ w.T`, weights dequantized in registers and rounded through bfloat16, the
dense matrix never materialized.

One 210-byte Q6_K superblock holds `ql` (128 B of low nibbles), `qh` (64 B of
high bit-pairs), 16 signed int8 sub-scales and an f16 `d`; element `e` decodes
as `(d * scales[e // 16]) * (q[e] - 32)`. The warp mapping is Q4_K's: one warp
per output row `n`, lane `w` walking the row's superblocks over elements
`{32c + w}` for `c` in 0..7, float32 accumulate, butterfly-shuffle reduction.
Unlike Q4_K there is no uint32 row view — 210 is not a multiple of 4 — so the
scale bytes are read straight out of the uint8 tensor (two distinct bytes per
warp per chunk, still coalesced); `d` comes from a float16 recast view.

There is no tensor-core Q6_K GEMM yet, so the dispatch is two-way: warp-GEMV
through M = 8, dequantize-then-matmul above it. The cap is higher than Q4_K's
M ≤ 2 precisely because nothing better competes in that range — the GEMV is
still ahead of the fallback at M = 8.

`bench/bench_matmul.py`, 4096×14336 Q6_K weight (48MB quantized, 117MB dense):

| M  | fused (dispatch) | unfused  | dense bf16 matmul |
|----|------------------|----------|-------------------|
| 1  |  66 µs (gemv)    | 357 µs   | 151 µs            |
| 2  |  92 µs (gemv)    | 347 µs   | 148 µs            |
| 4  | 163 µs (gemv)    | 350 µs   | 148 µs            |
| 8  | 244 µs (gemv)    | 342 µs   | 148 µs            |
| 16 | 346 µs (unfused) | 346 µs   | 150 µs            |

At M = 1 that is 676 GB/s of quantized weight read — 5.4× the unfused path,
and 2.3× faster than a dense bf16 matmul with the weights already resident.
The obvious next step is a Q6_K B-operand for the tensor-core GEMM, which
would flatten M = 3..128 the way it does for Q4_K.
