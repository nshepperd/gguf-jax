"""Fused warp-GEMV for the low-bit GGUF types: one kernel template, eight decoders.

Covers Q2_K and the grid-codebook i-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS and IQ3_S -- the types a ~2-bit mixed GGUF is built from. They
share one mapping and differ only in how a lane turns its bytes into quants,
so that part is a per-type ``decode`` hook and everything else is common.

**Mapping.** One warp per output row, walking the row one 256-element
superblock at a time. Lane ``w`` owns the eight consecutive elements
``8w..8w+7``, which for every type here sit inside a single scale group (16 or
32 elements). So a lane needs exactly one scale per superblock, its x values
are a single 16-byte load per batch row, and its quants are one 8-byte grid
entry (IQ1/IQ2) or two 4-byte ones (IQ3).

**Numerics.** The scales are factored out of the inner loop: each lane
computes ``t = sum_i q_i * x_i`` over its eight elements, with ``q_i`` the
*integer-valued* quant (grid value with sign applied, or 8*(grid + delta) for
IQ1), then ``acc += scale * t`` once per superblock. Every product ``q_i * x_i``
is exact in float32 (``q`` fits in bf16's mantissa), so this is more accurate
than rounding each dequantized weight to bf16 first -- and it does not match
``x @ w.dequantize(bf16).T`` bitwise, which the older k-quant GEMVs do. With a
one-hot ``x`` the result is the float32 reference weight rounded once to bf16.

**Inner loop.** The quants are built as packed bf16 pairs and fed straight to
``fma.rn.f32.bf16`` (sm_100+; ``FHFMA.BF16`` in SASS), which multiplies two
bf16 *halves* -- it takes the ``.H1`` half of a register directly -- and adds
into a float32. Neither x nor the quants are ever unpacked to float32. Two
ways of building the bf16 pairs, both a single ``prmt`` per pair:

- *selector tables* (IQ1, IQ2): those grids only use three distinct values,
  so the per-row-type value set fits in two registers as four bf16 slots, and
  the smem grid stores, per element, the ``prmt`` nibbles that pick its slot.
  The table word *is* the byte-permute selector; IQ1 picks between two value
  sets by the 8-group's delta sign, which also folds the delta into ``q``.
- *magic bytes* (IQ3, Q2_K): eight values per grid do not fit, so the table
  holds raw bytes ``v`` (< 128), ``prmt`` places each under a 0x43 high byte to
  form bf16 ``128 + v``, and one ``sub.bf16x2`` removes the 128 exactly.

Signs (IQ2/IQ3) are one ``imad`` + one ``lop3`` per pair: ``(sb * C_p) &
0x80008000`` lands sign bits 2p and 2p+1 on the two bf16 sign bits, because the
two shifted copies of the 8-bit sign byte never overlap.

**Memory.** A ~2-bit superblock is only 50-110 bytes, so one superblock per
warp in flight is nowhere near enough bytes outstanding to cover DRAM latency
(the first version spent most of its stalls waiting on exactly that). Weight
bytes are therefore staged: each warp copies a chunk of several superblocks
into its own shared-memory window with one 16-byte load and store per lane,
prefetching the next chunk while it decodes the current one, and the per-lane
field reads become short-latency shared loads. After that the kernels sit
against the load/store pipe and instruction issue about equally (ncu: LSU
~70%, issue ~75%), which is where further work would have to come from.

**Grids** live in shared memory (1-16 KB), filled once per CTA. CTAs are
persistent -- a fixed, fully resident grid striding over row blocks -- so that
fill is amortized over many rows instead of repeated per 8 rows, which for
IQ1's 16 KB table would otherwise read more bytes than the weight itself.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import cutejax
import cutlass
import jax
import jax.numpy as jnp
import numpy as np
from cutlass import cute
from cutlass.cute.arch import inline_ptx
from gguf import quants as _ref
from gguf.constants import GGML_QUANT_SIZES
from gguf.constants import GGMLQuantizationType as QT

QK_K = 256
_ROWS_PER_CTA = 8     # warps per CTA, one output row each
_CTA = 256
_CTAS_PER_SM = 4      # persistent grid size; 5 and 6 measured slower
_STAGE_BYTES = 512    # weight bytes staged per warp per chunk: 16 per lane

# Measured on an RTX 5070 Ti at (17408, 5120), against dequantize-then-matmul:
# 14-17x at M=1, still 1.8x (IQ3_S, Q2_K) to 2.7x (IQ2_XXS) at M=12, and a
# wash by M=16 for Q2_K -- past M=10 the per-batch-row accumulators start
# costing registers and the time per extra row roughly doubles.
_GEMV_MAX_M = 12


# ---------------------------------------------------------------------------
# Instruction helpers

def _prmt(a, b, sel):
    return inline_ptx("prmt.b32 {$w0}, {$r0}, {$r1}, {$r2};",
                      write_only_types=[cutlass.Int32], read_only_args=[a, b, sel])


def _sub_128(a):
    """bf16x2 (128 + v) -> v, exactly. bf16x2 arithmetic takes no immediates."""
    return inline_ptx(
        "{ .reg .b32 k; mov.b32 k, 0x43004300; sub.rn.bf16x2 {$w0}, {$r0}, k; }",
        write_only_types=[cutlass.Int32], read_only_args=[a])


def _selp(cond, a, b):
    """cond ? a : b for f32, branch-free. ``cond`` is an Int32 0/1."""
    return inline_ptx("{ .reg .pred p; setp.ne.s32 p, {$r0}, 0; selp.f32 {$w0}, {$r1}, {$r2}, p; }",
                      write_only_types=[cutlass.Float32], read_only_args=[cond, a, b])


def _f16_bits_to_f32(h):
    return inline_ptx("{ .reg .b16 h; cvt.u16.u32 h, {$r0}; cvt.f32.f16 {$w0}, h; }",
                      write_only_types=[cutlass.Float32], read_only_args=[h])


_DOT8_PTX = (
    "{ .reg .b16 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;"
    " .reg .f32 t;"
    " mov.b32 {a0, a1}, {$r0}; mov.b32 {a2, a3}, {$r1};"
    " mov.b32 {a4, a5}, {$r2}; mov.b32 {a6, a7}, {$r3};"
    " mov.b32 {b0, b1}, {$r4}; mov.b32 {b2, b3}, {$r5};"
    " mov.b32 {b4, b5}, {$r6}; mov.b32 {b6, b7}, {$r7};"
    " fma.rn.f32.bf16 t, a0, b0, 0f00000000;"
    " fma.rn.f32.bf16 t, a1, b1, t; fma.rn.f32.bf16 t, a2, b2, t;"
    " fma.rn.f32.bf16 t, a3, b3, t; fma.rn.f32.bf16 t, a4, b4, t;"
    " fma.rn.f32.bf16 t, a5, b5, t; fma.rn.f32.bf16 t, a6, b6, t;"
    " fma.rn.f32.bf16 {$w0}, a7, b7, t; }"
)


def _dot8(q, x):
    """sum of 8 bf16 products, q and x each four packed bf16x2 words, f32 out."""
    return inline_ptx(_DOT8_PTX, write_only_types=[cutlass.Float32],
                      read_only_args=[q[0], q[1], q[2], q[3], x[0], x[1], x[2], x[3]])


def _bf16_bits(v: float) -> int:
    return int(np.array(v, np.float32).view(np.uint32)) >> 16


def _pack2(lo: float, hi: float) -> int:
    """Two bf16 values in one int32 word, as a signed Python int."""
    word = _bf16_bits(lo) | (_bf16_bits(hi) << 16)
    return word - (1 << 32) if word >= 1 << 31 else word


_C43 = 0x43434343
_ONES = _pack2(1.0, 1.0)
_SIGN_MASK = 0x80008000 - (1 << 32)
_SIGN_MUL = (0x40008000, 0x10002000, 0x04000800, 0x01000200)


def _apply_signs(q, sb):
    """Flip q's eight bf16 lanes where the matching bit of the sign byte is set."""
    return [q[p] ^ ((sb * _SIGN_MUL[p]) & _SIGN_MASK) for p in range(4)]


def _ksigns(s7):
    """IQ2/IQ3 7-bit sign index -> 8 sign bits; the 8th is the parity of the 7."""
    return s7 | ((cute.arch.popc(s7) & 1) << 7)


def _select_pairs(v0, v1, w0, w1):
    """Selector-table entry (two words) -> four bf16 pairs from slots (v0, v1)."""
    return [_prmt(v0, v1, w0), _prmt(v0, v1, w0 >> 16),
            _prmt(v0, v1, w1), _prmt(v0, v1, w1 >> 16)]


def _magic_pairs(word):
    """Four byte values (< 128) -> two exact bf16 pairs."""
    return [_sub_128(_prmt(word, _C43, 0x4140)),
            _sub_128(_prmt(word, _C43, 0x4342))]


# ---------------------------------------------------------------------------
# Block field access

class _Block:
    """Typed reads of one staged superblock: byte ``base`` of staging slot
    ``slot`` (one per warp), through the slot's u8/u16/f16/u32 views.

    ``base`` must be even, and a multiple of 4 for word-aligned types.
    """

    def __init__(self, views, slot, base, nbytes: int):
        self.sB, self.sH, self.sF, self.sW = views
        self.slot = slot
        self.bbase = base
        self.hbase = base >> 1
        self.wbase = base >> 2 if nbytes % 4 == 0 else None

    # Uint8/Uint16 -> Int32 sign-extends (LDS.S8); the masks make the loads
    # zero-extending, which ptxas folds into LDS.U8/LDS.U16.
    def u8(self, off):
        return cutlass.Int32(self.sB[self.slot, self.bbase + off]) & 0xFF

    def u16(self, off):
        return cutlass.Int32(self.sH[self.slot, self.hbase + (off >> 1)]) & 0xFFFF

    def f16(self, off):
        return cutlass.Float32(self.sF[self.slot, self.hbase + (off >> 1)])

    def u32(self, off):
        """``off`` must be a multiple of 4 on word-aligned types, of 2 otherwise."""
        if self.wbase is not None:
            return cutlass.Int32(self.sW[self.slot, self.wbase + (off >> 2)])
        return self.u16(off) | (self.u16(off + 2) << 16)


# ---------------------------------------------------------------------------
# Per-type decoders. Each takes (block, lane, table) and returns
# (scale, four bf16x2 words of integer quants, min-scale or None) such that
# the lane's contribution is scale * sum(q * x) - min * sum(x).

def _scale_iq2(d, sc):
    return d * (cutlass.Float32(0.5) + cutlass.Float32(sc)) * cutlass.Float32(0.25)


# Value slots for the selector tables: codes 0, 1, 2 index grid_map.
_IQ2_SLOTS = (_pack2(8.0, 25.0), _pack2(43.0, 0.0))
# IQ1 stores 8 * (grid + delta), grid in {-1, 0, 1}, delta = +-1/8.
_IQ1_SLOTS_POS = (_pack2(-7.0, 1.0), _pack2(9.0, 0.0))
_IQ1_SLOTS_NEG = (_pack2(-9.0, -1.0), _pack2(7.0, 0.0))


def _decode_iq2_xxs(b, w, t):
    ib, l = w >> 2, w & 3
    idx = b.u8(2 + 8 * ib + l)
    aux = b.u32(6 + 8 * ib)
    db = _scale_iq2(b.f16(0), (aux >> 28) & 0xF)
    sb = _ksigns((aux >> (7 * l)) & 0x7F)
    e = t[idx]
    q = _select_pairs(_IQ2_SLOTS[0], _IQ2_SLOTS[1],
                      cutlass.Int32(e), cutlass.Int32(e >> 32))
    return db, _apply_signs(q, sb), None


def _decode_iq2_xs(b, w, t):
    qv = b.u16(2 + 2 * w)
    sc = (b.u8(66 + (w >> 2)) >> (4 * ((w >> 1) & 1))) & 0xF
    db = _scale_iq2(b.f16(0), sc)
    sb = _ksigns(qv >> 9)
    e = t[qv & 511]
    q = _select_pairs(_IQ2_SLOTS[0], _IQ2_SLOTS[1],
                      cutlass.Int32(e), cutlass.Int32(e >> 32))
    return db, _apply_signs(q, sb), None


def _decode_iq2_s(b, w, t):
    idx = b.u8(2 + w) | (((b.u8(66 + (w >> 2)) >> (2 * (w & 3))) & 3) << 8)
    sc = (b.u8(74 + (w >> 2)) >> (4 * ((w >> 1) & 1))) & 0xF
    db = _scale_iq2(b.f16(0), sc)
    e = t[idx]
    q = _select_pairs(_IQ2_SLOTS[0], _IQ2_SLOTS[1],
                      cutlass.Int32(e), cutlass.Int32(e >> 32))
    return db, _apply_signs(q, b.u8(34 + w)), None


def _decode_iq3_xxs(b, w, t):
    ib, l = w >> 2, w & 3
    iq = b.u16(2 + 2 * w)                 # the lane's two 4-element grid indices
    aux = b.u32(66 + 4 * ib)
    db = (b.f16(0) * (cutlass.Float32(0.5) + cutlass.Float32((aux >> 28) & 0xF))
          * cutlass.Float32(0.5))
    sb = _ksigns((aux >> (7 * l)) & 0x7F)
    q = _magic_pairs(t[iq & 0xFF]) + _magic_pairs(t[iq >> 8])
    return db, _apply_signs(q, sb), None


def _decode_iq3_s(b, w, t):
    iq = b.u16(2 + 2 * w)
    hb = b.u8(66 + (w >> 2)) >> (2 * (w & 3))
    i0 = (iq & 0xFF) | ((hb & 1) << 8)
    i1 = (iq >> 8) | (((hb >> 1) & 1) << 8)
    sc = (b.u8(106 + (w >> 3)) >> (4 * ((w >> 2) & 1))) & 0xF
    db = b.f16(0) * cutlass.Float32(1 + 2 * sc)
    q = _magic_pairs(t[i0]) + _magic_pairs(t[i1])
    return db, _apply_signs(q, b.u8(74 + w)), None


def _iq1_pairs(e, neg):
    """Selector entry -> pairs, choosing the value set by the delta sign."""
    sel = -neg                                       # 0 or all-ones
    v0 = _IQ1_SLOTS_POS[0] ^ ((_IQ1_SLOTS_POS[0] ^ _IQ1_SLOTS_NEG[0]) & sel)
    v1 = _IQ1_SLOTS_POS[1] ^ ((_IQ1_SLOTS_POS[1] ^ _IQ1_SLOTS_NEG[1]) & sel)
    return _select_pairs(v0, v1, cutlass.Int32(e), cutlass.Int32(e >> 32))


def _decode_iq1_s(b, w, t):
    qh = b.u16(34 + 2 * (w >> 2))
    idx = b.u8(2 + w) | (((qh >> (3 * (w & 3))) & 7) << 8)
    dl = b.f16(0) * cutlass.Float32(2 * ((qh >> 12) & 7) + 1)
    q = _iq1_pairs(t[idx], (qh >> 15) & 1)
    return dl * cutlass.Float32(0.125), q, None


def _decode_iq1_m(b, w, t):
    # the four scale words as two LDS.32 (IQ1_M blocks are word-aligned)
    lo, hi = b.u32(48), b.u32(52)
    # d is split across the top nibbles of the four scale words
    dbits = (((lo >> 12) & 0xF) | ((lo >> 24) & 0xF0)
             | ((hi >> 4) & 0xF00) | ((hi >> 16) & 0xF000))
    word = lo ^ ((lo ^ hi) & -((w >> 4) & 1))          # scale word w >> 3
    sraw = (word >> (16 * ((w >> 3) & 1))) & 0xFFFF
    sc = (sraw >> (3 * ((w >> 1) & 3))) & 7
    dl = _f16_bits_to_f32(dbits) * cutlass.Float32(2 * sc + 1)
    nib = (b.u8(32 + (w >> 1)) >> (4 * (w & 1))) & 0xF
    idx = b.u8(w) | ((nib & 7) << 8)
    q = _iq1_pairs(t[idx], nib >> 3)
    return dl * cutlass.Float32(0.125), q, None


def _decode_q2_k(b, w, t):
    # element 8w+j is (qs[32h + 8(w&3) + j] >> 2s) & 3, h = w>>4, s = (w>>2)&3
    off = 16 + 32 * (w >> 4) + 8 * (w & 3)
    sh = 2 * ((w >> 2) & 3)
    sc = b.u8(w >> 1)
    dd = b.u32(80)                                     # d, dmin in one LDS.32
    dl = _f16_bits_to_f32(dd & 0xFFFF) * cutlass.Float32(sc & 0xF)
    ml = _f16_bits_to_f32((dd >> 16) & 0xFFFF) * cutlass.Float32(sc >> 4)
    q = (_magic_pairs((b.u32(off) >> sh) & 0x03030303)
         + _magic_pairs((b.u32(off + 4) >> sh) & 0x03030303))
    return dl, q, ml


# ---------------------------------------------------------------------------
# Tables

def _ref_grid(ref_cls) -> np.ndarray:
    ref_cls.init_grid()
    assert ref_cls.grid is not None
    return np.asarray(ref_cls.grid).reshape(tuple(ref_cls.grid_shape))


def _grid_codes(ref_cls) -> np.ndarray:
    """(entries, values_per_entry) indices into ref_cls.grid_map."""
    grid = _ref_grid(ref_cls)
    return np.searchsorted(np.asarray(ref_cls.grid_map, np.float32), grid)


def _selector_table(ref_cls) -> np.ndarray:
    # prmt nibbles (2c, 2c+1) pick bf16 slot c out of the two value registers
    c = _grid_codes(ref_cls).astype(np.uint8)
    return ((2 * c) | ((2 * c + 1) << 4)).astype(np.uint8)


def _value_table(ref_cls) -> np.ndarray:
    return _ref_grid(ref_cls).astype(np.uint8)


@dataclass(frozen=True)
class _Spec:
    qtype: QT
    decode: Callable
    table: Callable[[], np.ndarray] | None   # () -> uint8 (entries, bytes/entry)
    entry_bytes: int = 0                     # 8: Int64 entries, 4: Int32

    @property
    def nbytes(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][1]

    @functools.cached_property
    def table_words(self) -> np.ndarray:
        if self.table is None:
            return np.zeros(1, np.int32)
        tab = self.table()
        assert tab.shape[1] == self.entry_bytes
        return np.frombuffer(np.ascontiguousarray(tab).tobytes(), dtype=np.int32)


_SPECS = {s.qtype: s for s in (
    _Spec(QT.IQ2_XXS, _decode_iq2_xxs, lambda: _selector_table(_ref.IQ2_XXS), 8),
    _Spec(QT.IQ2_XS, _decode_iq2_xs, lambda: _selector_table(_ref.IQ2_XS), 8),
    _Spec(QT.IQ2_S, _decode_iq2_s, lambda: _selector_table(_ref.IQ2_S), 8),
    _Spec(QT.IQ3_XXS, _decode_iq3_xxs, lambda: _value_table(_ref.IQ3_XXS), 4),
    _Spec(QT.IQ3_S, _decode_iq3_s, lambda: _value_table(_ref.IQ3_S), 4),
    _Spec(QT.IQ1_S, _decode_iq1_s, lambda: _selector_table(_ref.IQ1_S), 8),
    _Spec(QT.IQ1_M, _decode_iq1_m, lambda: _selector_table(_ref.IQ1_M), 8),
    _Spec(QT.Q2_K, _decode_q2_k, None),
)}

LOWBIT_TYPES = tuple(_SPECS)


# ---------------------------------------------------------------------------
# Kernel template

def _clamp_hi(i, hi):
    """min(i, hi) for Int32 (no integer min in the DSL's operator set)."""
    d = i - hi
    return i - (d & ~(d >> 31))


# Per-chunk overhead (staging, two warp syncs, prefetch addressing), in units
# of one superblock's decode. Picks between a few more chunks and a few empty
# slots in the last one.
_CHUNK_OVERHEAD = 0.35


def _chunk_size(nbytes: int, n_sb: int) -> int:
    """Superblocks per staged chunk for rows of ``n_sb`` superblocks.

    At most what fits in one 512-byte warp window after up to 14 bytes of
    misalignment (rows are only 2-byte aligned). The chunk body is
    straight-line code, so an empty slot in a row's last chunk still costs a
    full decode; this picks the size with the least total of empty slots and
    per-chunk overhead.
    """
    cap = min(8, (_STAGE_BYTES - 14) // nbytes)
    return min(range(cap, 0, -1),
               key=lambda c: -(-n_sb // c) * (c + _CHUNK_OVERHEAD))


@functools.cache
def _make_launch(qtype: QT, n_ctas: int, chunk: int, ragged: bool):
    """``ragged``: the row's superblock count is not a multiple of ``chunk``."""
    spec = _SPECS[qtype]
    nbytes = spec.nbytes
    table_words = int(spec.table_words.size) if spec.table is not None else 0
    assert table_words % _CTA == 0
    has_min = qtype == QT.Q2_K

    @cute.kernel
    def kernel(gW16: cute.Tensor, gT: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor,
               n_rows: cutlass.Int32, row_bytes: cutlass.Int32, last16: cutlass.Int32,
               n_sb: cutlass.Int32, M: cutlass.Constexpr[int]):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nct, _, _ = cute.arch.grid_dim()
        warp = tid >> 5
        lane = tid & 31

        smem = cutlass.memory.SmemAllocator()

        @cute.struct
        class Stage:
            b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _ROWS_PER_CTA * _STAGE_BYTES // 4], 16]

        stage = Stage(smem.allocate(Stage.size_in_bytes(), byte_alignment=16)).b
        sptr = stage.get_tensor(cute.make_layout(1)).iterator

        def sview(dtype, width):
            per = _STAGE_BYTES // width
            return cute.make_tensor(cute.recast_ptr(sptr, dtype=dtype),
                                    cute.make_layout((_ROWS_PER_CTA, per), stride=(per, 1)))

        # (warp, lane, word): each lane's 16 bytes, for one STS.128
        sStage = cute.make_tensor(
            cute.recast_ptr(sptr, dtype=cutlass.Int32),
            cute.make_layout((_ROWS_PER_CTA, 32, 4), stride=(_STAGE_BYTES // 4, 4, 1)))
        views = (sview(cutlass.Uint8, 1), sview(cutlass.Uint16, 2),
                 sview(cutlass.Float16, 2), sview(cutlass.Int32, 4))

        sT = None
        if cutlass.const_expr(table_words > 0):
            @cute.struct
            class Table:
                t: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, table_words], 16]

            sT32 = Table(smem.allocate(Table.size_in_bytes(), byte_alignment=16)).t.get_tensor(
                cute.make_layout(table_words))
            # unrolled, so every thread's loads are in flight at once
            for i in cutlass.range_constexpr(table_words // _CTA):
                sT32[tid + i * _CTA] = gT[tid + i * _CTA]
            cute.arch.sync_threads()
            if cutlass.const_expr(spec.entry_bytes == 8):
                sT = cute.make_tensor(cute.recast_ptr(sT32.iterator, dtype=cutlass.Int64),
                                      cute.make_layout(table_words // 2))
            else:
                sT = sT32

        n_blocks = (n_rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA
        n_chunks = (n_sb + chunk - 1) // chunk
        last_row = n_rows - 1
        row_step = nct * _ROWS_PER_CTA

        # Weight bytes go global -> registers -> this warp's smem window, one
        # chunk ahead: the next chunk's loads are in flight while this one
        # decodes out of shared memory. Each lane moves one aligned 16-byte
        # piece of a 512-byte window (one LDG.128 + one STS.128 per chunk);
        # the chunk's misalignment within the window rides along as ``skew``.
        # The prefetch runs on across rows (the last chunk of a row fetches
        # the first of the next), so the pipeline only fills once per warp --
        # rows are only a few chunks long. Rows past the end are clamped and
        # their results dropped, which keeps every warp of a CTA in step.
        # Windows are clamped to the last 16 bytes holding data; reading the
        # tail of that piece cannot fault.
        stg = cute.make_rmem_tensor(4, cutlass.Int32)
        skew = cute.make_rmem_tensor(2, cutlass.Int32)     # [current, next]
        start = _clamp_hi(bid * _ROWS_PER_CTA + warp, last_row) * row_bytes
        skew[1] = start & 15
        cute.autovec_copy(gW16[_clamp_hi((start >> 4) + lane, last16), None], stg)

        for rb in cutlass.range(bid, n_blocks, nct):
            n = rb * _ROWS_PER_CTA + warp
            acc = cute.make_rmem_tensor(M, cutlass.Float32)
            for m in cutlass.range_constexpr(M):
                acc[m] = cutlass.Float32(0.0)

            for c in cutlass.range(n_chunks):
                cute.autovec_copy(stg, sStage[warp, lane, None])
                skew[0] = skew[1]
                cute.arch.sync_warp()
                # next chunk of this row, or the first chunk of the next row
                more = cutlass.Int32(c + 1 < n_chunks)
                pn = _clamp_hi(n + (1 - more) * row_step, last_row)
                pstart = pn * row_bytes + more * (c + 1) * (chunk * nbytes)
                skew[1] = pstart & 15
                cute.autovec_copy(gW16[_clamp_hi((pstart >> 4) + lane, last16), None], stg)

                # straight-line, so the compiler can overlap one superblock's
                # dependent loads with the next one's
                for j in cutlass.range_constexpr(chunk):
                    s = c * chunk + j
                    valid = cutlass.Int32(1)
                    if cutlass.const_expr(ragged):
                        # an empty slot in the row's last chunk: decode whatever
                        # is staged there, against a valid x, and drop it
                        valid = cutlass.Int32(s < n_sb)
                        s = _clamp_hi(s, n_sb - 1)
                    blk = _Block(views, warp, skew[0] + j * nbytes, nbytes)
                    scale, q, ml = spec.decode(blk, lane, sT)
                    for m in cutlass.range_constexpr(M):
                        xv = cute.make_rmem_tensor(4, cutlass.Int32)
                        cute.autovec_copy(
                            cute.local_tile(gX, (1, 1, 4), (m, s, lane))[0, 0, None], xv)
                        xw = [xv[0], xv[1], xv[2], xv[3]]
                        part = scale * _dot8(q, xw)
                        if cutlass.const_expr(has_min):
                            part = part - ml * _dot8([_ONES] * 4, xw)
                        if cutlass.const_expr(ragged):
                            part = _selp(valid, part, cutlass.Float32(0.0))
                        acc[m] = acc[m] + part
                cute.arch.sync_warp()

            for m in cutlass.range_constexpr(M):
                v = acc[m]
                v = v + cute.arch.shuffle_sync_bfly(v, 16)
                v = v + cute.arch.shuffle_sync_bfly(v, 8)
                v = v + cute.arch.shuffle_sync_bfly(v, 4)
                v = v + cute.arch.shuffle_sync_bfly(v, 2)
                v = v + cute.arch.shuffle_sync_bfly(v, 1)
                # nested: the DSL stages each `if`, and `and` of two traced
                # booleans is not one it can stage
                if n < n_rows:  # noqa: SIM102
                    if lane == 0:
                        gO[m, n] = gO.element_type(v)

    @cute.jit
    def launch(stream, gU: cute.Tensor, gT: cute.Tensor, gX: cute.Tensor, gO: cute.Tensor):
        n_rows = gU.shape[0]
        row_bytes = gU.shape[1]
        total = n_rows * row_bytes
        n16 = (total + 15) // 16
        # the weight as 16-byte pieces; the static 4 makes each piece one LDG.128
        gW16 = cute.make_tensor(cute.recast_ptr(gU.iterator, dtype=cutlass.Int32),
                                cute.make_layout((n16, 4), stride=(4, 1)))
        m = gX.shape[0]
        n_sb = gX.shape[1] // QK_K
        # x as (M, superblock, word): the static 128 makes each lane's 16 bytes
        # provably aligned, so the four words load as one LDG.128
        gX3 = cute.make_tensor(cute.recast_ptr(gX.iterator, dtype=cutlass.Int32),
                               cute.make_layout((m, n_sb, 128), stride=(n_sb * 128, 128, 1)))
        # min_blocks_per_mp caps registers so the whole persistent grid is
        # resident at once; a CTA that has to wait for a slot would run its
        # share of the rows as a serial second wave.
        kernel(gW16, gT, gX3, gO, n_rows, row_bytes, n16 - 1, n_sb, m).launch(
            grid=[n_ctas, 1, 1], block=[_CTA, 1, 1], stream=stream,
            min_blocks_per_mp=_CTAS_PER_SM)

    return launch


@functools.cache
def _n_ctas() -> int:
    return jax.devices()[0].core_count * _CTAS_PER_SM


def _table(qtype: QT) -> jax.Array:
    # a fresh constant per call: a cached jax.Array would leak out of a jit trace
    return jnp.asarray(_SPECS[qtype].table_words)


def matmul_lowbit(x: jax.Array, w) -> jax.Array:
    """``x @ w.T`` with ``w`` a :class:`~gguf_jax.QuantizedArray` (N, K) of one
    of :data:`LOWBIT_TYPES`.

    ``x`` is bfloat16 ``(..., K)``; the result is bfloat16 ``(..., N)``.
    Quants are multiplied with x exactly and the scales applied once per
    sub-block in float32, so this is *more* accurate than
    ``x @ w.dequantize(bfloat16).T`` and not bitwise equal to it. Warp-GEMV
    for flattened batch sizes up to the cap, dequantize-then-matmul above.
    """
    from gguf_jax.array import QuantizedArray

    assert isinstance(w, QuantizedArray) and w.qtype in _SPECS, w.qtype
    assert len(w.shape) == 2, "w must be a 2D weight"
    n_rows, k_dim = w.shape
    assert x.shape[-1] == k_dim, f"contraction mismatch: {x.shape[-1]} != {k_dim}"
    assert x.dtype == jnp.bfloat16, "x must be bfloat16"

    # byte offsets into the weight are int32 in the kernel
    assert w.data.size < 2**31, "weight too large for int32 byte offsets"

    xm = x.reshape(-1, k_dim)
    m = xm.shape[0]
    if m <= _GEMV_MAX_M:
        n_sb = k_dim // QK_K
        chunk = _chunk_size(_SPECS[w.qtype].nbytes, n_sb)
        n_ctas = min(_n_ctas(), -(-n_rows // _ROWS_PER_CTA))
        out = cutejax.call(
            _make_launch(w.qtype, n_ctas, chunk, n_sb % chunk != 0),
            jax.ShapeDtypeStruct((m, n_rows), jnp.bfloat16),
            w.data.reshape(n_rows, -1), _table(w.qtype), xm,
            in_specs=[None, None, cutejax.ArraySpec(static_dims=(0,))],
            out_specs=cutejax.ArraySpec(static_dims=(0,)),
        )
    else:
        out = xm @ w.dequantize(jnp.bfloat16).T
    return out.reshape(*x.shape[:-1], n_rows)
