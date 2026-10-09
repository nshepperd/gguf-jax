"""One warp-GEMV launch per qtype at a matched shape, for ncu.

Everything before cudaProfilerStart is warmup, so ncu with
``--profile-from-start off`` sees exactly one launch per ``--types`` entry, in
order, all at the same (N, K) and M. ncu flushes caches between replay passes,
so unlike the wall-clock loop in gemv_shapes.py the weight is cold and the
DRAM figures are real rather than L2 hits.

    ncu --profile-from-start off -k regex:'matmul|kernel' \
        --section SpeedOfLight -- python bench/gemv_ncu.py -n 12288 -k 4096 \
        --types q4_k,q6_k
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
from gemv_shapes import KERNELS, make_weight
from gguf.constants import GGMLQuantizationType as QT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=12288)
    p.add_argument("-k", type=int, default=4096)
    p.add_argument("-m", "--batch", type=int, default=1)
    p.add_argument("--types", default="q4_k,q6_k")
    args = p.parse_args()

    lib = ctypes.CDLL("libcudart.so")
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(args.batch, args.k)), jnp.bfloat16)

    calls = []
    for name in args.types.split(","):
        qtype = QT[name.strip().upper()]
        fn = KERNELS[qtype][0]
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
