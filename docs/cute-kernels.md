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

One warp per output row `n` of W. Lane `w` walks the row's superblocks,
dequantizes into registers, multiplies with `x[m, e]` and accumulates in
float32; a butterfly-shuffle reduction folds the 32 lanes. M is compile-time
static, so there is one specialization per batch size.

Lane `w` owns qs bytes `4w..4w+3` — four *consecutive* bytes. Since byte `b`
decodes to elements `64*(b>>5) + (b&31)` and that plus 32, the lane's elements
are `e = 64*(w>>3) + 4*(w&7) + j` and `e+32` for `j` in 0..3, which is a
permutation of the superblock.

That mapping is chosen for instruction count, not for coalescing. Profiling put
the kernel at 78% compute speed-of-light against 49% DRAM with 59% of its
instructions on the ALU pipe: it is bound by integer bit-surgery, not by memory,
so the thing worth minimising is work per element. Consecutive bytes buy three
cuts at once — one `LDG.32` covers the whole 128-byte qs region (was four
`LDG.8` at stride 32), the lane touches exactly **two** of the eight 32-element
sub-blocks so it unpacks two `(scale, min)` pairs instead of eight, and its `x`
elements are contiguous runs of four. Measured 1.15× on the kernel and 1.08× on
end-to-end decode of Qwen3-VL-8B Q4_K_M, for 26% fewer instructions.

The per-element expression is unchanged, so the bf16 rounding guarantee holds
exactly (`test_cute_matmul_q4_k_exact_weights` pins it with a one-hot `x`); only
the f32 accumulation order moves, which the contract above already disclaims.
Set `GGUF_JAX_Q4K_GEMV_V1=1` to run the previous mapping for comparison.

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
as `(d * scales[e // 16]) * (q[e] - 32)`. One warp per output row `n`, lane `w`
walking the row's superblocks over elements `{32c + w}` for `c` in 0..7, float32
accumulate, butterfly-shuffle reduction — the mapping Q4_K used before its
consecutive-byte remap. Q6_K has not had the same treatment: its 210-byte block
is not 4-byte aligned, so the equivalent would be uint16 loads (six per
superblock down to three) rather than Q4_K's four down to one, and Q6_K is the
smaller share of a decode step.
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

## Low-bit fused GEMV (`lowbit.py`)

`gguf_jax.cute.matmul_lowbit(x, w)` covers the types a ~2-bit mixed GGUF is
built from — Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S
(`LOWBIT_TYPES`) — with **one kernel template and a small decoder per type**.
`gguf_jax.matmul` routes all eight to it. It was written for Underdog Saluki
27B (a GSQ-RCO IQ2-mix of Qwen3.8-27B, 7.89 GB), where each layer's tensors use
a different one of these types, and before this only 2% of the file's bytes
had a fused kernel.

### Numerics: a different contract

Unlike the kernels above, these do **not** round each weight through bf16.
Each lane computes `t = Σ qᵢ·xᵢ` over its eight elements with `qᵢ` the
*integer-valued* quant (signed grid value; `8·(grid ± 1/8)` for IQ1), then
`acc += scale · t` once per sub-block. Every `qᵢ·xᵢ` is exact in float32, so
the result is closer to the exact product than `x @ w.dequantize(bf16).T` —
`test_cute_matmul_lowbit_accuracy` asserts exactly that — but not bitwise
equal to it. With a one-hot `x` the output is the float32 reference weight
rounded once to bf16, bitwise (`test_cute_matmul_lowbit_exact_weights`).

### Template

- **Mapping.** One warp per output row; lane `w` owns elements `8w..8w+7` of
  each superblock. For every type these sit inside one scale group, so a lane
  needs one scale per superblock, one 16-byte `x` load per batch row, and one
  8-byte grid entry (IQ1/IQ2) or two 4-byte ones (IQ3).
- **Inner loop.** Quants are built as packed bf16 pairs and fed to
  `fma.rn.f32.bf16` (`FHFMA.BF16`, sm_100+), which multiplies bf16 *halves* —
  `.H1` operands included — into a float32. Nothing is unpacked to f32. The
  pairs come from a single `prmt` each: for IQ1/IQ2, whose grids use only three
  values, the shared-memory grid holds per element the `prmt` nibbles that pick
  a bf16 out of two value registers; for IQ3/Q2_K the table holds raw bytes,
  `prmt` puts them under a `0x43` high byte (bf16 `128 + v`), and one
  `sub.bf16x2` takes the 128 off exactly. Signs cost an `imad` + `lop3` per
  pair: `(signbyte · Cₚ) & 0x80008000` lands two sign bits on the two bf16 sign
  bits.
- **Staging.** A low-bit superblock is only 50–110 bytes, so one superblock in
  flight per warp leaves DRAM latency exposed — the first version's stalls
  were ~85% `long_scoreboard`. Each warp instead copies a chunk of 3–8
  superblocks into its own 512-byte shared-memory window with one 16-byte load
  and store per lane, prefetching the next chunk (across row boundaries) while
  decoding the current one. Rows are only 2-byte aligned, so the window starts
  at the aligned address below the chunk and the offset rides along. The chunk
  size divides the row's superblock count where it can, because the chunk body
  is straight-line code and an empty slot would still issue a full decode.
- **Persistence.** Grids (1–16 KB) are filled once per CTA, and the grid is a
  fixed, fully resident 4 CTAs/SM (`min_blocks_per_mp`) striding over row
  blocks — otherwise IQ1's 16 KB table would cost more L2 reads than the
  weight.

### Performance

`bench/gemv_shapes.py --saluki`, RTX 5070 Ti, M = 1, rotating weights so they
come from DRAM; µs per call, with GB/s of quantized weight:

| type    | (17408, 5120) ffn_up | (5120, 17408) ffn_down | (248320, 5120) output |
|---------|----------------------|------------------------|-----------------------|
| Q2_K    |  41 µs, 716 GB/s     |  46 µs, 636 GB/s       | 520 µs, 803 GB/s      |
| IQ2_XXS |  40 µs, 574 GB/s     |  46 µs, 499 GB/s       | 491 µs, 668 GB/s      |
| IQ2_XS  |  39 µs, 658 GB/s     |  39 µs, 656 GB/s       | 483 µs, 761 GB/s      |
| IQ2_S   |  43 µs, 669 GB/s     |  43 µs, 667 GB/s       | 508 µs, 801 GB/s      |
| IQ3_XXS |  46 µs, 747 GB/s     |  50 µs, 688 GB/s       | 616 µs, 790 GB/s      |
| IQ3_S   |  52 µs, 732 GB/s     |  53 µs, 721 GB/s       | 660 µs, 827 GB/s      |
| IQ1_S   |  38 µs, 457 GB/s     |  43 µs, 401 GB/s       | 475 µs, 523 GB/s      |
| IQ1_M   |  44 µs, 440 GB/s     |  50 µs, 388 GB/s       | 557 µs, 499 GB/s      |
| *IQ4_XS (`iq4_xs.py`)* | *90 µs, 524 GB/s* | *100 µs, 472 GB/s* | *1150 µs, 587 GB/s* |

Against dequantize-then-matmul at (17408, 5120) that is 14–17× at M = 1, and
still 1.8–2.7× at the M = 12 cap; Q2_K breaks even around M = 16.

The 3-bit types are at DRAM bandwidth. The 2-bit and especially the 1-bit
ones are not: per byte of weight they decode more elements, and ncu puts them
against the load/store pipe (~70%) and instruction issue (~75%) about equally,
with memory stalls gone. The per-superblock budget is ~6 LSU instructions
(four field reads, one grid lookup, one `x` load) and ~45 instructions in
all; getting the 2-bit types to DRAM speed means cutting both by roughly a
third — reusing each `x` load across two rows per warp, and fewer, wider
field reads, are the obvious candidates.

Two DSL details that cost time to find: converting a `Uint8`/`Uint16` element
to `Int32` *sign*-extends (`LDS.S8`), so every narrow field read is masked;
and bf16x2 arithmetic takes no immediates, so `sub.bf16x2`'s constant lives in
a register inside the inline asm.
