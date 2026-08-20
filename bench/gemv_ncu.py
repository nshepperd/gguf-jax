"""One Q4_K and one Q6_K warp-GEMV launch at a matched shape, for ncu.

Everything before cudaProfilerStart is warmup, so ncu with
``--profile-from-start off`` sees exactly two launches: q4_k then q6_k, same
(N, K), same M. ncu flushes caches between replay passes, so unlike the
wall-clock loop in gemv_shapes.py the weight is cold and the DRAM figures are
real rather than L2 hits.

    ncu --profile-from-start off -k regex:'q4_k_matmul|q6_k_matmul' \
        --section SpeedOfLight -- python bench/gemv_ncu.py -n 12288 -k 4096
"""
from __future__ import annotations

import argparse
import ctypes
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax
import jax.numpy as jnp
import numpy as np
from gguf.constants import GGMLQuantizationType as QT

from gguf_jax.cute import matmul_q4_k, matmul_q6_k

from gemv_shapes import make_weight


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=12288)
    p.add_argument("-k", type=int, default=4096)
    p.add_argument("-m", "--batch", type=int, default=1)
    args = p.parse_args()

    lib = ctypes.CDLL("libcudart.so")
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(args.batch, args.k)), jnp.bfloat16)

    calls = []
    for qtype, fn in ((QT.Q4_K, matmul_q4_k), (QT.Q6_K, matmul_q6_k)):
        w = make_weight(args.n, args.k, qtype, rng)
        jitted = jax.jit(fn)
        for _ in range(5):  # JIT, autotune, allocator warm
            jitted(x, w).block_until_ready()
        calls.append((jitted, w))

    lib.cudaProfilerStart()
    for jitted, w in calls:
        jitted(x, w).block_until_ready()
    lib.cudaProfilerStop()


if __name__ == "__main__":
    main()
