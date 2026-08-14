"""Benchmark dequantization throughput (pure-XLA kernels, and cute kernels if registered).

Usage:
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
        uv run python bench/bench_dequant.py [--cute]

Reports wall time per dequantize of a (4096, 14336) weight (llama-8B MLP size)
and the effective memory bandwidth (bytes read + bytes written) / time.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

import gguf_jax

SHAPE = (4096, 14336)
QTYPES = [
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.BF16,
]


def bench(fn, *args, iters=50, warmup=5):
    for _ in range(warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cute", action="store_true",
                        help="register cute kernels (gguf_jax.cute) before benchmarking")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    if args.cute:
        from gguf_jax import cute as gguf_cute
        gguf_cute.register()

    out_dtype = jnp.dtype(args.dtype)
    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}, out dtype: {out_dtype.name}, shape: {SHAPE}")
    numel = int(np.prod(SHAPE))
    out_bytes = numel * out_dtype.itemsize

    print(f"{'qtype':>8} {'bytes in':>10} {'time':>9} {'eff BW':>10}")
    for qtype in QTYPES:
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        rng = np.random.default_rng(0)
        data = rng.integers(0, 256, size=(SHAPE[0], SHAPE[1] // block_size * type_size),
                            dtype=np.uint8)
        qa = gguf_jax.QuantizedArray.from_bytes(jnp.asarray(data), qtype, shape=SHAPE,
                                                dtype=out_dtype)
        fn = jax.jit(lambda q: q.dequantize())
        dt = bench(fn, qa)
        in_bytes = data.nbytes
        bw = (in_bytes + out_bytes) / dt / 1e9
        print(f"{qtype.name:>8} {in_bytes/1e6:>8.1f}MB {dt*1e6:>7.0f}us {bw:>8.1f}GB/s")


if __name__ == "__main__":
    main()
