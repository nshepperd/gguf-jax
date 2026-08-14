"""Benchmark fused Q4_K matmul vs dequantize-then-matmul.

    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
        uv run python bench/bench_matmul.py

y = x @ W.T with W (N, K) Q4_K, x (M, K) bf16 — the decode/small-batch shape.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from gguf.constants import GGMLQuantizationType

import gguf_jax
from gguf_jax.cute import matmul_q4_k

N, K = 4096, 14336


def bench(fn, *args, iters=100, warmup=5):
    for _ in range(warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters


def main():
    rng = np.random.default_rng(0)
    nb = N * K // 256
    data = rng.integers(0, 256, size=(nb, 144), dtype=np.uint8)
    data[:, :4] = (rng.normal(size=(nb, 2)) * 0.05).astype(np.float16).view(np.uint8)
    w = gguf_jax.QuantizedArray.from_bytes(
        data.reshape(N, -1), GGMLQuantizationType.Q4_K, shape=(N, K))
    w_bytes = data.nbytes

    fused = jax.jit(matmul_q4_k)
    unfused = jax.jit(lambda x, w: x @ w.dequantize(jnp.bfloat16).T)
    dense_w = w.dequantize(jnp.bfloat16)
    dense = jax.jit(lambda x, wd: x @ wd.T)

    print(f"device: {jax.devices()[0].device_kind}, W: {N}x{K} Q4_K "
          f"({w_bytes/1e6:.0f}MB quantized, {N*K*2/1e6:.0f}MB as bf16)")
    print(f"{'M':>4} {'fused':>9} {'unfused':>9} {'dense bf16':>10}   speedup")
    for m in [1, 2, 4, 8, 16]:
        x = jnp.asarray(rng.normal(size=(m, K)), dtype=jnp.bfloat16)
        t_fused = bench(fused, x, w)
        t_unfused = bench(unfused, x, w)
        t_dense = bench(dense, x, dense_w)
        print(f"{m:>4} {t_fused*1e6:>7.0f}us {t_unfused*1e6:>7.0f}us "
              f"{t_dense*1e6:>8.0f}us   {t_unfused/t_fused:>5.1f}x")


if __name__ == "__main__":
    main()
