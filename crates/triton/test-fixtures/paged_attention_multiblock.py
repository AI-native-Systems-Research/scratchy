"""MULTI-BLOCK PAGED FLASH ATTENTION — the KV sweep reads the block table from device
memory, and the flash recurrence (m, l, acc) crosses blocks inside one program.

`paged_attention.py` (one-block, card-verified 2026-10-01, corr 0.999989) proved the
gathered V-leg contraction and the Kt-plane score leg reach the card. What it did NOT do
is the sweep: one launch = one KV block leaves the recurrence to a caller that does not
exist. THIS kernel is the paged form proper — `for b in range(NUM_BLOCKS)` — and the two
things that make it possible:

1. THE UNROLL PASS IS THE INTEGRATION. `unroll_constant_trip_loops`
   (triton-ktir/src/passes/to_ktir.rs:1212) turns the constant-trip KV sweep into
   straight-line trips, the same mechanism the DENSE 16-tile form card-verified
   (noncausal 0.999958, causal 0.999972). After the unroll, each trip's `load` of its
   block-table slice reads at a CONSTANT corner (`[t*64]`), and each trip's `gather` of
   its V rows reads a CONSTANT-trip ids vector — so the whole-function door's
   const-corner window guard (whole_function.rs:1336) is satisfied per trip. What is new
   relative to the dense form is the gather inside the unrolled body: multiple
   `construct_indirect_access_tile`s, one per trip, each consumed by its own materializing
   contraction.

2. THE BLOCK TABLE IS DEVICE-RESIDENT, SLICED PER BLOCK. `desc_table` is
   `[NUM_BLOCKS, BLOCK_N]` i32; trip t loads its 64 indices at row t. Each trip's gather
   is therefore over DIFFERENT indices — the paging itself — while the Kt plane is
   presented HOST-PRE-TRANSPOSED PER BLOCK (`[HEAD_DIM, NUM_BLOCKS*BLOCK_N]`, column
   block t is this block's K^T), which is the page-fill-time host obligation the one-block
   fixture already stated: the Kt plane is written ONCE when pages are filled, not per
   decode step.

THE RECURRENCE IS THE DENSE FORM'S OWN. m_i, l_i, acc are `scf.for` iter_args — the
loop-carried shape the dense 16-tile kernel already lowers through this door: the unroll
threads each trip's yield into the next trip's carry (`carries = yielded` under the trip
renaming, to_ktir.rs:1363), and `fold_splat_seeds` lowers the seeds to top-level
operands. Per block: m_ij = max(m_i, rowmax(qk_t)); p_t = exp2(qk_t - m_ij); alpha =
exp2(m_i - m_ij); acc = acc*alpha + p_t @ v_rows_t; l = l*alpha + rowsum(p_t).

THE SAME TWO-LEG LAW, PER BLOCK. The score leg is direct×direct (`tl.dot(q, kt_t)` —
kt_t is this block's K^T slice, loaded at a constant corner); the value leg is the rung-4
gather verbatim (`v_rows_t = v_desc.gather(ids_t, 0)`). The four score-leg gather
spellings that are refused by name (two gathers in one program before unrolling, a
transposed gather-B, a gather on A, a side-load of the table) are unchanged from the
one-block header — see `paged_attention.py`.

CONTROLS (the one-block contract, extended): the block table is a PERMUTATION of
block ids, so
  * a SORTED table must destroy the output against the permutation reference (the
    direct-read trap) while matching the sorted expectation (base-2 softmax over the
    table rows in sorted order);
  * a SWAPPED table (two block ids exchanged) must destroy it too — one-block's control
    could not distinguish "row i wrong" from "block b wrong", this one can;
  * per-input corruption controls (permq/permkt/permv/zeroq) as before.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def paged_attn_multiblock_fwd(desc_q, desc_kt, desc_v, desc_table, desc_o,
                              NUM_BLOCKS: tl.constexpr,
                              BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                              V: tl.constexpr, BLOCK_M: tl.constexpr):
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[BLOCK_M, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    # The Kt plane: block t's K^T is the column slice [t*BLOCK_N, (t+1)*BLOCK_N).
    # Host-pre-transposed at page fill; the kernel never transposes.
    kt_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[HEAD_DIM, NUM_BLOCKS * BLOCK_N],
        strides=[NUM_BLOCKS * BLOCK_N, 1],
        block_shape=[HEAD_DIM, BLOCK_N])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[V, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[1, HEAD_DIM])
    # THE BLOCK TABLE, DEVICE-RESIDENT AND SLICED PER BLOCK: trip t's 64 indices are
    # row t. `table_desc.load([t*BLOCK_N, 0])`'s corner is a constant after the unroll
    # materialises the induction variable — which is what makes a per-trip gather
    # addressable by the door's const-corner window guard.
    table_desc = tl.make_tensor_descriptor(
        desc_table, shape=[NUM_BLOCKS * BLOCK_N], strides=[1],
        block_shape=[BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[BLOCK_M, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])

    q = q_desc.load([0, 0])                       # [BLOCK_M, HEAD_DIM]

    # The flash recurrence — the dense form's own loop-carried shape. f16 accumulators
    # (the hardware fact, see attention_flash.py delta 2); l seeds at 1.0 so the epilogue
    # divide cannot be by zero; m seeds at -inf so the first block's max is unconditional.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float16) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float16) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16)

    for b in tl.range(0, NUM_BLOCKS, 1):
        # Trip-local slices, all at CONSTANT corners once the unroll materialises b:
        # the block-table row, the Kt column block, and (through ids_b) the V rows.
        ids_b = table_desc.load([b * BLOCK_N])    # [BLOCK_N] i32
        kt_b = kt_desc.load([0, b * BLOCK_N])     # [HEAD_DIM, BLOCK_N]
        v_rows_b = v_desc.gather(ids_b, 0)        # [BLOCK_N, HEAD_DIM] — rung 4 verbatim

        qk = tl.dot(q, kt_b, out_dtype=tl.float16)  # [BLOCK_M, BLOCK_N]
        m_ij = tl.maximum(m_i, tl.max(qk, 1).to(tl.float16))
        p = tl.math.exp2((qk - m_ij[:, None]).to(tl.float32)).to(tl.float16)
        alpha = tl.math.exp2((m_i - m_ij).to(tl.float32)).to(tl.float16)
        l_ij = tl.sum(p, 1).to(tl.float16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v_rows_b, acc)              # [BLOCK_M, HEAD_DIM]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    o = acc / l_i[:, None]
    o_desc.store([0, 0], o)


def inputs(seed: int = 0, m=64, num_blocks=2, block_n=64, v=1024, head_dim=64):
    """Randomized fp16 q, kt (pre-transposed per block), V table, in the kernel's
    geometry. The block table is NOT drawn here: it is pinned to `PAGED_ATTN_TABLE` (a
    permutation of block-row indices), the multi-block negative-control contract."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(m, head_dim, generator=g, dtype=torch.float16)
    kt = torch.randn(head_dim, num_blocks * block_n, generator=g,
                     dtype=torch.float16)
    table = torch.randn(v, head_dim, generator=g, dtype=torch.float16)
    return q, kt, table


def reference(q, kt, table, block_table, block_n=64):
    """The oracle: for each block b, the indices are block_table rows laid flat
    (`ids = block_table.flatten()` with one row per block), kt's column block b is this
    block's K^T, and the recurrence is the kernel's own base-2 form. The flattened
    spelling mirrors what the kernel sees: a `[NUM_BLOCKS, BLOCK_N]` table sliced per
    block IS a flat `[NUM_BLOCKS*BLOCK_N]` permutation, one 64-row group per block."""
    n = block_table.numel()
    ids = block_table.reshape(-1).long()          # [NUM_BLOCKS*BLOCK_N]
    qk = q.float() @ kt.float()                   # [m, NUM_BLOCKS*BLOCK_N]
    m = qk.max(dim=1).values
    p = torch.exp2(qk - m[:, None])
    l = p.sum(dim=1)
    o = (p @ table[ids].float()) / l[:, None]
    return o.to(torch.float16)


# THE BLOCK TABLE: a permutation of 0..127 arranged so that NO BLOCK'S rows are
# contiguous in the table — block 0's rows are scattered across both halves. A block-
# swap control on this table moves 64 rows at a time; a sorted-table control reads the
# table rows in order. Same 64-entry-per-block granularity as the one-block fixture's
# PAGED_ATTN_IDS (which is this table's first block at NUM_BLOCKS=1... it is not; this
# is a fresh permutation at the multi-block width).
PAGED_ATTN_TABLE = torch.tensor([
    # block 0: rows 37, 91, 12, ... (scattered)
    37, 91, 12, 60, 118, 5, 73, 44, 102, 26, 85, 9, 55, 127, 30, 68,
    14, 96, 47, 81, 3, 110, 23, 66, 89, 17, 100, 52, 77, 41, 8, 114,
    61, 20, 106, 34, 93, 2, 71, 49, 121, 15, 84, 38, 99, 11, 58, 79,
    27, 105, 6, 90, 46, 116, 32, 63, 95, 1, 74, 21, 108, 43, 87, 57,
    # block 1: the complement, also scattered
    0, 124, 16, 82, 40, 111, 29, 67, 98, 7, 53, 119, 35, 76, 25, 103,
    48, 13, 92, 59, 122, 4, 70, 39, 86, 18, 101, 54, 78, 42, 113, 22,
    64, 19, 107, 31, 94, 123, 10, 72, 120, 36, 83, 51, 97, 28, 62, 115,
    33, 104, 24, 88, 117, 45, 109, 56, 126, 50, 75, 69, 112, 65, 125, 80,
], dtype=torch.int32)


def _selfcheck():
    """The oracle vs an independently spelled base-2 block recurrence, plus the control
    discriminability, checked before anything ships. Run by the numeric-data generator
    and by any scorer that wants to re-derive the expectations."""
    q, kt, table = inputs(0, num_blocks=2)
    ref = reference(q, kt, table, PAGED_ATTN_TABLE[:128].reshape(2, 64))

    # Independent spelling: the explicit per-block recurrence, nothing shared with
    # `reference` but the function itself.
    m_i = torch.full((64,), float("-inf"))
    l_i = torch.ones(64)
    acc = torch.zeros(64, 64)
    for b in range(2):
        ids = PAGED_ATTN_TABLE[b * 64:(b + 1) * 64].long()
        qk = q.float() @ kt.float()[:, b * 64:(b + 1) * 64]
        m_ij = torch.maximum(m_i, qk.max(dim=1).values)
        p = torch.exp2(qk - m_ij[:, None])
        alpha = torch.exp2(m_i - m_ij)
        l_ij = p.sum(dim=1)
        acc = acc * alpha[:, None] + p @ table[ids].float()
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    indep = (acc / l_i[:, None]).to(torch.float16)
    # ONE f16 ULP, not zero: the two spellings are mathematically identical but round
    # differently (a single full-width max vs per-block maxima with rescale through
    # alpha), and measured disagreement is 2 of 4096 elements at 1.22e-4 -- f16's last
    # bit. The kernel itself carries f16 accumulators, so its error floor sits far
    # above this; anything larger than an ULP here means the oracle is wrong, not the
    # rounding.
    assert (ref.float() - indep.float()).abs().max() <= 2.5e-4, \
        "oracle disagrees with the independent block recurrence"

    # The sorted-table control must differ from the permutation reference...
    sorted_tbl = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    sorted_ref = reference(q, kt, table, sorted_tbl)
    assert (ref.float() - sorted_ref.float()).abs().max() > 1.0, \
        "sorted-table control does not discriminate"
    # ...and a block SWAP must differ too (one-block's control cannot see block-level
    # errors; this one exists to catch a trip that reads the wrong table row).
    swapped = PAGED_ATTN_TABLE[:128].reshape(2, 64).flip(0).contiguous()
    swapped_ref = reference(q, kt, table, swapped)
    assert (ref.float() - swapped_ref.float()).abs().max() > 1.0, \
        "block-swap control does not discriminate"
    return ref
