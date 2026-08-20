"""Q4_K vs Q6_K warp-GEMV at matched shapes, to separate two explanations.

Profiling Qwen3-VL-8B decode showed the Q6_K GEMV at 610 GB/s and the Q4_K one
at 442 on nominally comparable work, which reads as "the Q6_K kernel is better".
But GB/s is the wrong yardstick for these two: they share a structure -- one
warp per output row, one loop iteration per superblock, 8 chunks per iteration
-- so a superblock costs about the same either way, while Q6_K's superblock
carries 210 bytes against Q4_K's 144. A kernel bound by anything other than
bandwidth therefore posts a 1.46x higher GB/s for doing the same work.

This runs both types at IDENTICAL (N, K) so the two readings can be compared
directly. Both do N*K/256 warp-iterations at any shape, so ns/iter is the
like-for-like column and GB/s is the one that can mislead.

The card has 48 MB of L2 and most of these weights are smaller than that, so a
timing loop over one weight measures L2 rather than DRAM -- which is how an
earlier version of this reported 102% of achievable bandwidth. Each shape
therefore rotates over enough distinct weights to exceed L2 twice over, the way
a decode step does when it walks 4.7 GB of them.

    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
        uv run python bench/gemv_shapes.py
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as QT

import gguf_jax
from gguf_jax.cute import matmul_q4_k, matmul_q6_k

SMS = 70            # RTX 5070 Ti; only used to report wave count
L2_BYTES = 48 << 20  # GB203; weights below this would be timed out of cache

# The decode shapes of Qwen3-VL-8B, plus a large one to show the asymptote.
SHAPES = [
    (1024, 4096),     # k_proj, v_proj
    (4096, 4096),     # q_proj, o_proj
    (12288, 4096),    # gate_proj, up_proj
    (4096, 12288),    # down_proj
    (151936, 4096),   # lm_head
]


def make_weight(n: int, k: int, qtype: QT, rng) -> gguf_jax.QuantizedArray:
    bpb = GGML_QUANT_SIZES[qtype][1]
    nb = n * k // 256
    data = rng.integers(0, 256, size=(nb, bpb), dtype=np.uint8)
    # Keep the f16 super-scales small so nothing overflows into inf.
    scale = (rng.normal(size=(nb, 2)) * 0.05).astype(np.float16).view(np.uint8)
    if qtype == QT.Q4_K:
        data[:, :4] = scale              # d, dmin at bytes 0..3
    else:
        data[:, 208:210] = scale[:, :2]  # d at bytes 208..209
    return gguf_jax.QuantizedArray.from_bytes(
        data.reshape(n, -1), qtype, shape=(n, k))


def bench_rotating(fn, x, weights, *, iters: int, warmup: int = 5) -> float:
    """Seconds per call, cycling through `weights` so none stays L2-resident."""
    for w in weights:
        for _ in range(warmup):
            out = fn(x, w)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for i in range(iters):
        out = fn(x, weights[i % len(weights)])
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--bandwidth", type=float, default=790.0,
                   help="achievable GB/s, for the last column")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    print(f"{jax.devices()[0].device_kind}, M={args.batch}\n")
    print(f"{'shape':>16s} {'type':>5s} {'CTAs':>6s} {'waves':>6s} {'MB':>6s} "
          f"{'x':>4s} {'us':>8s} {'ns/iter':>8s} {'GB/s':>7s} {'% bw':>6s}")
    print("  (* = working set still fits L2; that row's GB/s is optimistic)\n")
    for n, k in SHAPES:
        iters = args.iters if n < 100_000 else args.iters // 8
        row = []
        for qtype, fn in ((QT.Q4_K, matmul_q4_k), (QT.Q6_K, matmul_q6_k)):
            nbytes = n * k // 256 * GGML_QUANT_SIZES[qtype][1]
            # Capped: past a point the rotation itself costs more than the
              # L2 residency it removes. A row that still fits is flagged.
            copies = min(16, max(1, -(-2 * L2_BYTES // nbytes)))
            weights = [make_weight(n, k, qtype, rng) for _ in range(copies)]
            x = jnp.asarray(rng.normal(size=(args.batch, k)), jnp.bfloat16)
            secs = bench_rotating(jax.jit(fn), x, weights, iters=iters)
            iters_done = n * k // 256          # warp-iterations, type-independent
            gbs = nbytes / secs / 1e9
            row.append((qtype, secs, nbytes, iters_done, gbs, copies))
            del weights
        for qtype, secs, nbytes, it, gbs, copies in row:
            ctas = (n + 7) // 8
            print(f"{f'({n},{k})':>16s} {str(qtype).split('.')[-1]:>5s} "
                  f"{ctas:6d} {ctas / SMS:6.1f} {nbytes / 1e6:6.0f} "
                  f"{copies:3d}{'*' if nbytes * copies < L2_BYTES else ' '}"
                  f"{secs * 1e6:8.2f} "
                  f"{secs * 1e9 / it:8.3f} {gbs:7.0f} "
                  f"{gbs / args.bandwidth * 100:5.0f}%")
        print()


if __name__ == "__main__":
    main()
