"""Fused Q4_K GEMM on tensor cores (SM80-class ``mma.sync``, runs on SM120).

Computes ``y = x @ W.T`` with ``x`` bf16 ``(M, K)`` and ``W`` Q4_K ``(N, K)``,
modeled on CuTeDSL's Ampere ``TensorOpGemm`` example, with one structural
change: the B operand flows through the pipeline *quantized*.

- A (activations): the standard path — cp.async into swizzled smem,
  ``ldmatrix`` into MMA fragments.
- B (weights): cp.async copies the raw Q4_K words into smem (144 B per row
  per superblock, 3.5x less smem traffic than dense bf16); dequantization
  happens at the smem->register stage, writing bf16 straight into the MMA
  B-fragment. An identity-tensor partition supplies each fragment element's
  (n, k) coordinate, so the ``mma.sync`` thread-value layout never has to be
  spelled out.

Tile: (bM, bN, bK) = (32, 64, 256) — bK is exactly one Q4_K superblock, so
each row's scale header is read once per k-tile. 128 threads, 3-stage
cp.async pipeline, m16n8k16 bf16 atoms in a (2, 2, 1) layout, f32
accumulate. M is dynamic (predicated loads/stores); N % 64 == 0 and
K % 256 == 0 are required (K is guaranteed by Q4_K itself).

Weight values are rounded through bf16 in-register (same as the GEMV kernel
and ``dequantize``-then-matmul), so all three paths agree up to f32
accumulation order.
"""
from __future__ import annotations

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute, utils
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

QK_K = 256
_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][1]  # 144
_BLOCK_WORDS = _BLOCK_BYTES // 4                               # 36
_BLOCK_HALVES = _BLOCK_BYTES // 2                              # 72

_BM, _BN, _BK = 32, 64, QK_K
_STAGES = 3
_THREADS = 128
_ATOM_LAYOUT = (2, 2, 1)
_MMA_INST = (16, 8, 16)


def _dequant_one(sBq, sBh, stage, n, k):
    """Decode element (n, k) of the current B k-tile from quantized smem.

    Same op order as the reference: fl32(d)*sc, fl32(dmin)*mn, dl*q - dm,
    rounded to bf16. ``k`` is the position within the superblock (0..255).
    """
    d = cutlass.Float32(sBh[n, 0, stage])
    dmin = cutlass.Float32(sBh[n, 1, stage])
    is_ = k >> 5
    j = is_ & 3
    w_d = cutlass.Int32(sBq[n, 1, stage])
    w_m = cutlass.Int32(sBq[n, 2, stage])
    w_md = cutlass.Int32(sBq[n, 3, stage])
    bd = (w_d >> (8 * j)) & 0xFF
    bm = (w_m >> (8 * j)) & 0xFF
    bmd = (w_md >> (8 * j)) & 0xFF
    # branchless select between the is_ < 4 and is_ >= 4 packings
    hi = (k >> 7) & 1  # == (is_ >= 4), since k in [0, 256)
    lo = 1 - hi
    sc = lo * (bd & 63) + hi * ((bmd & 0x0F) | ((bd >> 6) << 4))
    mn = lo * (bm & 63) + hi * ((bmd >> 4) | ((bm >> 6) << 4))
    dl = d * cutlass.Float32(sc)
    dm = dmin * cutlass.Float32(mn)
    # qs byte 16 + 32*(k//64) + (k%32), low nibble for k%64 < 32
    word = 4 + 8 * (k >> 6) + ((k & 31) >> 2)
    byte_in_word = k & 3
    nib = (k >> 5) & 1
    q32 = cutlass.Int32(sBq[n, word, stage])
    q = (q32 >> (8 * byte_in_word + 4 * nib)) & 0x0F
    return cutlass.BFloat16(dl * cutlass.Float32(q) - dm)


def _fill_b(tCrB, tCcB, sBq, sBh, k_block, stage):
    rb = tCrB[None, None, k_block]
    cb = tCcB[None, None, k_block]
    for i in range(cute.size(rb)):
        crd = cb[i]
        rb[i] = _dequant_one(sBq, sBh, stage, crd[0], crd[1])


@cute.kernel
def _q4k_gemm_kernel(
    mW: cute.Tensor,   # (N, W_words) int32 — quantized rows
    mX: cute.Tensor,   # (M, K) bf16
    mO: cute.Tensor,   # (M, N) bf16
    sA_layout: cute.ComposedLayout,
    sBq_layout: cute.Layout,
    sBh_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B1: cute.TiledCopy,
    tiled_copy_B2: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # gA: (bM, bK, kt), gW: (bN, 36, kt), gC: (bM, bN)
    gA = cute.local_tile(mX, tiler=(_BM, _BK), coord=(bidx, None))
    gW = cute.local_tile(mW, tiler=(_BN, _BLOCK_WORDS), coord=(bidy, None))
    gC = cute.local_tile(mO, tiler=(_BM, _BN), coord=(bidx, bidy))

    # identity tensor mirroring mX, for predication of A rows (m < M)
    mcA = cute.make_identity_tensor(mX.layout.shape)
    cA = cute.local_tile(mcA, tiler=(_BM, _BK), coord=(bidx, None))

    # shared memory
    @cute.struct
    class SharedStorage:
        a: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)], 16
        ]
        b: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, cute.cosize(sBq_layout)], 16
        ]

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sA = SharedStorage(storage).a.get_tensor(sA_layout)
    sBq = SharedStorage(storage).b.get_tensor(sBq_layout)
    # f16 view of the same B bytes: (d, dmin) at halves 0, 1 of each row
    sBh = cute.make_tensor(
        cute.recast_ptr(sBq.iterator, dtype=cutlass.Float16), sBh_layout)

    # sub-views of the B tiles for the split 128-bit copies
    kt_b = cute.size(gW, mode=[2])
    row_stride = cute.size(mW, mode=[1])
    gW1 = cute.make_tensor(
        gW.iterator,
        cute.make_layout((_BN, 32, kt_b), stride=(row_stride, 1, _BLOCK_WORDS)))
    gW2 = cute.domain_offset(
        (0, 32, 0),
        cute.make_tensor(
            gW.iterator,
            cute.make_layout((_BN, 4, kt_b), stride=(row_stride, 1, _BLOCK_WORDS))))
    stage_words = _BN * _BLOCK_WORDS
    sB1 = cute.make_tensor(
        sBq.iterator,
        cute.make_layout((_BN, 32, _STAGES), stride=(_BLOCK_WORDS, 1, stage_words)))
    sB2 = cute.domain_offset(
        (0, 32, 0),
        cute.make_tensor(
            sBq.iterator,
            cute.make_layout((_BN, 4, _STAGES), stride=(_BLOCK_WORDS, 1, stage_words))))

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B1 = tiled_copy_B1.get_slice(tidx)
    thr_copy_B2 = tiled_copy_B2.get_slice(tidx % 64)
    tAgA = thr_copy_A.partition_S(gA)    # (CPY, CPY_M, CPY_K, kt)
    tAsA = thr_copy_A.partition_D(sA)    # (CPY, CPY_M, CPY_K, PIPE)
    tB1gB = thr_copy_B1.partition_S(gW1)
    tB1sB = thr_copy_B1.partition_D(sB1)
    tB2gB = thr_copy_B2.partition_S(gW2)
    tB2sB = thr_copy_B2.partition_D(sB2)
    tAcA = thr_copy_A.partition_S(cA)

    # predicate for A along M (K always divides bK for Q4_K)
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
            stride=(cute.size(tAgA, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    for rest_v in range(tApA.shape[0]):
        for m in range(tApA.shape[1]):
            tApA[rest_v, m, 0] = cute.elem_less(
                tAcA[(0, rest_v), m, 0, 0][0], mX.shape[0])

    # prologue: zero A smem once (predicated-off rows then stay zero), then
    # prefetch the first (stages - 1) k-tiles
    tAsA.fill(0)
    cute.arch.sync_threads()

    k_tile_count = cute.size(tAgA, mode=[3])
    k_tile_index = cutlass.Int32(0)
    for k_tile in range(_STAGES - 1):
        if k_tile < k_tile_count:
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B1,
                tB1gB[None, None, None, k_tile_index],
                tB1sB[None, None, None, k_tile],
            )
            if tidx < 64:
                cute.copy(
                    tiled_copy_B2,
                    tB2gB[None, None, None, k_tile_index],
                    tB2sB[None, None, None, k_tile],
                )
            k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

    # MMA partitions
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)                       # (V, MMA_M, MMA_K, PIPE)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    # B: coordinates of each fragment element within the (bN, bK) tile
    cB = cute.make_identity_tensor((_BN, _BK))
    tCcB = thr_mma.partition_B(cB)                       # (V, MMA_N, MMA_K)
    tCrB = cute.make_fragment_like(tCcB, cutlass.BFloat16)

    # ldmatrix for A
    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16)
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    thr_copy_ldm_A = tiled_copy_s2r_A.get_slice(tidx)
    tCsA_copy_view = thr_copy_ldm_A.partition_S(sA)
    tCrA_copy_view = thr_copy_ldm_A.retile(tCrA)

    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(_STAGES - 1)

    # Snapshot of the stage the register pipeline reads from. Like tCsA_p in
    # the reference example, this advances only at the last k-block of a
    # tile — the live smem_pipe_read increments mid-tile (at k_block == 0).
    stage_read = cutlass.Int32(0)
    tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]

    num_k_block = cute.size(tCrA, mode=[2])


    # prefetch first k-block from the first k-tile
    cute.arch.cp_async_wait_group(_STAGES - 2)
    cute.arch.sync_threads()
    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
    _fill_b(tCrB, tCcB, sBq, sBh, 0, stage_read)

    for k_tile in range(k_tile_count):
        for k_block in cutlass.range(num_k_block, unroll_full=True):
            if k_block == num_k_block - 1:
                tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                stage_read = smem_pipe_read
                cute.arch.cp_async_wait_group(_STAGES - 2)
                cute.arch.sync_threads()

            # stage register pipeline for k_block + 1 (for the last k-block
            # this is k-block 0 of the next tile, from the new stage)
            k_block_next = (k_block + 1) % num_k_block
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, k_block_next],
                tCrA_copy_view[None, None, k_block_next],
            )
            _fill_b(tCrB, tCcB, sBq, sBh, k_block_next, stage_read)

            if k_block == 0 and k_tile + _STAGES - 1 < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, smem_pipe_write],
                    pred=tApA,
                )

            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_block],
                tCrB[None, None, k_block],
                tCrC,
            )

            if k_block == 0:
                if k_tile + _STAGES - 1 < k_tile_count:
                    cute.copy(
                        tiled_copy_B1,
                        tB1gB[None, None, None, k_tile_index],
                        tB1sB[None, None, None, smem_pipe_write],
                    )
                    if tidx < 64:
                        cute.copy(
                            tiled_copy_B2,
                            tB2gB[None, None, None, k_tile_index],
                            tB2sB[None, None, None, smem_pipe_write],
                        )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()
                smem_pipe_write = smem_pipe_read
                smem_pipe_read = smem_pipe_read + 1
                if smem_pipe_read == _STAGES:
                    smem_pipe_read = 0

    # epilogue: predicated direct stores
    mcC = cute.make_identity_tensor(mO.layout.shape)
    cC = cute.local_tile(mcC, tiler=(_BM, _BN), coord=(bidx, bidy))
    tCcC = thr_mma.partition_C(cC)
    for i in cutlass.range_constexpr(cute.size(tCrC)):
        if cute.elem_less(tCcC[i][0], mO.shape[0]):
            tCgC[i] = mO.element_type(tCrC[i])


@cute.jit
def _q4k_gemm_launch(stream, gU: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor):
    n_rows = gU.shape[0]
    row_words = gU.shape[1] // 4
    mW = cute.make_tensor(
        cute.recast_ptr(gU.iterator, dtype=cutlass.Int32),
        cute.make_layout((n_rows, row_words), stride=(row_words, 1)),
    )

    # A smem: swizzled bf16, (bM, bK, stages)
    a_atom = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3), 0,
        cute.make_layout((8, 64), stride=(64, 1)),
    )
    sA_layout = cute.tile_to_shape(a_atom, (_BM, _BK, _STAGES), (0, 1, 2))

    # B smem: plain int32 words (bN, 36, stages) + f16 view (bN, 72, stages)
    stage_words = _BN * _BLOCK_WORDS
    sBq_layout = cute.make_layout(
        (_BN, _BLOCK_WORDS, _STAGES), stride=(_BLOCK_WORDS, 1, stage_words))
    sBh_layout = cute.make_layout(
        (_BN, _BLOCK_HALVES, _STAGES), stride=(_BLOCK_HALVES, 1, 2 * stage_words))

    # gmem -> smem copies
    atom_async_a = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16, num_bits_per_copy=128)
    tiled_copy_A = cute.make_tiled_copy_tv(
        atom_async_a,
        cute.make_layout((_THREADS // 32, 32), stride=(32, 1)),
        cute.make_layout((1, 8)),
    )
    # cp.async supports only 128-bit copies: split each 36-word row into
    # words 0..31 (all 128 threads) and words 32..35 (threads < 64)
    atom_async_b = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
        cutlass.Int32, num_bits_per_copy=128)
    tiled_copy_B1 = cute.make_tiled_copy_tv(
        atom_async_b,
        cute.make_layout((16, 8), stride=(8, 1)),
        cute.make_layout((1, 4)),
    )
    tiled_copy_B2 = cute.make_tiled_copy_tv(
        atom_async_b,
        cute.make_layout((64, 1), stride=(1, 1)),
        cute.make_layout((1, 4)),
    )

    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, _MMA_INST)
    permutation_mnk = (
        _ATOM_LAYOUT[0] * _MMA_INST[0],
        _ATOM_LAYOUT[1] * _MMA_INST[1] * 2,
        _ATOM_LAYOUT[2] * _MMA_INST[2],
    )
    tiled_mma = cute.make_tiled_mma(
        op, cute.make_layout(_ATOM_LAYOUT), permutation_mnk=permutation_mnk)

    m_tiles = (gX.shape[0] + _BM - 1) // _BM
    n_tiles = n_rows // _BN
    _q4k_gemm_kernel(
        mW, gX, gO, sA_layout, sBq_layout, sBh_layout,
        tiled_copy_A, tiled_copy_B1, tiled_copy_B2, tiled_mma,
    ).launch(grid=[m_tiles, n_tiles, 1], block=[_THREADS, 1, 1], stream=stream)


def gemm_q4_k(x: jax.Array, w) -> jax.Array:
    """``x @ w.T`` on tensor cores; ``x`` bf16 ``(M, K)``, ``w`` Q4_K ``(N, K)``.

    Requires ``N % 64 == 0``. M is dynamic — one compile serves all M.
    """
    from gguf_jax.array import QuantizedArray

    assert isinstance(w, QuantizedArray) and w.qtype == GGMLQuantizationType.Q4_K
    n_rows, k_dim = w.shape
    assert n_rows % _BN == 0, f"N must be a multiple of {_BN}, got {n_rows}"
    assert x.shape[-1] == k_dim and x.dtype == jnp.bfloat16

    xm = x.reshape(-1, k_dim)
    out = cutejax.call(
        _q4k_gemm_launch,
        jax.ShapeDtypeStruct((xm.shape[0], n_rows), jnp.bfloat16),
        w.data.reshape(n_rows, -1), xm,
        # row_bytes % 144 == 0 makes every quantized row 16-byte aligned,
        # which the 128-bit cp.async copies require the compiler to know
        in_specs=[cutejax.ArraySpec(divisibility=_BLOCK_BYTES),
                  cutejax.ArraySpec(divisibility=QK_K)],
    )
    return out.reshape(*x.shape[:-1], n_rows)
