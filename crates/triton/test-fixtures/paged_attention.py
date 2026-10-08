"""RUNG 4 APPLIED TO ATTENTION — paged flash attention in Triton, one KV block per launch.

`paged_score.py` proved the gathered V-leg contraction (`tl.dot(p, v_rows)` with the rows
chosen at runtime by a device index vector). This kernel puts the FULL ONE-BLOCK flash
step around that contraction: the block table (`desc_ids`) picks the KV rows, the score
matmul runs over a HOST-PRESENTED Kt plane, softmax runs on the score tile, and the value
contraction is the rung-4 gather verbatim. One launch = one KV block = one page of rows;
the recurrence across blocks is the caller's loop (a multi-block sweep inside one program
is blocked by the whole-function door's computed-corner guard, which resolves window
corners through `index_constants` and refuses a loop-carried one by name).

WHY THE SCORE LEG READS A HOST-PRESENTED Kt PLANE, INSTEAD OF GATHERING K TOO
============================================================================
The two-leg asymmetry is the whole design, and it is not a choice:

* THE VALUE LEG IS RUNG 4 VERBATIM. `v_rows = v_desc.gather(ids, 0)` are `[N, HD]` rows in
  row-blocked order, which IS the matmul slot's `[k, n]` residency, so `tl.dot(p, v_rows)`
  contracts them where they lie — the one gathered-contraction shape the whole-function
  door materializes (`gathered_matmul_materializes`).

* THE SCORE LEG CANNOT GATHER. Four spellings, four refusals, each in the door's own words:
    - TWO GATHERS IN ONE PROGRAM: "Split the program into one node per gather" —
      `gather_of` refuses a second `ktdp.construct_indirect_access_tile` outright, because
      the index operand is paired with its tensor BY POSITION and two indices cannot both
      sit adjacent to their own operand.
    - THE GATHER ON THE SCORE MATMUL'S B (the natural `tl.dot(q, k_rows.T)`): a transposed
      gather is refused BY NAME by `dot_to_linalg`'s paged contract, with the arch reason
      (dxp refuses relayouts on MPW4, "Implicit syncs not available for architectures
      prior to RCUDD1A") — gathered rows are contracted where they lie, and K rows
      gathered as `[n, hd]` are the wrong axis for a score contraction.
    - THE GATHER ON A (the "swap the dot" idea, `tl.dot(k_rows, q)`): no door support —
      `gathered_matmul_materializes` exists for `in_values[1]` ONLY, and any other op
      reading a gathered load is refused ("this door declares an index operand only for
      `Program::ScalarMul`").
    - A DIRECT LOAD of the same table beside the gather: the gathered load may have
      exactly ONE consumer (the counts gate at the Triton door, the consumer seal at the
      emitter), so the score leg cannot read the V table directly either.

  SCRATCHY'S ANSWER, AND OURS: the shipped attention keeps a THIRD PHYSICAL Kt plane
  (`KvPlane::Kt`, written already-transposed by the host) and never transposes in the
  kernel. The cost is stated plainly: the KV cache grows by one more plane of K.
  `attention_flash.py`'s delta 6 records the argument the OTHER way for the dense form —
  there a `tt.trans` of a direct load is FREE (folded into the access-tile order), so a
  second plane would be pure waste; HERE the transpose is refused by the arch, so the
  plane is the only spelling. And the plane is written ONCE when the page is filled, not
  per decode step — which is exactly the answer to the task's own note that "you cannot
  pre-transpose a paged KV cache per decode step": the transposition is a page-fill-time
  host obligation, not a kernel op.

THE TILE ORIENTATION, AND WHY NO `tt.trans` SURVIVES IN THIS KERNEL
==================================================================
`desc_kt` is `[HEAD_DIM, N_CTX]` — this launch's K block, already transposed:
`kt[d, n] == k_page[n, d]`. The score contraction is then a PLAIN direct×direct dot:

    qk = tl.dot(q, kt)          # [BLOCK_M, N_CTX]

which is the tutorial's own `[M, N]` score tile — no `.T` on either operand, so nothing
asks the door for a transposed-B fold, a computed-value transpose, or a relayout. BOTH
softmax reductions then run along axis=1 (the KV axis, one value per query row), which is
the orientation `attention_flash.py` card-verified, and N_CTX == 64 == one f16 stick so
the reduce-MAX defect cell (`rows > 1` AND `cols > stick`) cannot arise — the same
BLOCK_N=64 law, restated for the same reason.

THE IDS ARE A PERMUTATION, AND THAT IS THE NEGATIVE CONTROL: with the ids SORTED the
gathered V rows are `v[:N]` in table order and the output must DIFFER from the
permutation reference — a match there is the direct-read trap (the table's first rows
read in table order), the exact wrong-rows failure the door's gather guard exists to
refuse. Same contract as `paged_score.py`.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def paged_attn_fwd(desc_q, desc_kt, desc_v, desc_ids, desc_o,  #
                   N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,  #
                   V: tl.constexpr, BLOCK_M: tl.constexpr):
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[BLOCK_M, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    kt_desc = tl.make_tensor_descriptor(desc_kt, shape=[HEAD_DIM, N_CTX],
                                        strides=[N_CTX, 1],
                                        block_shape=[HEAD_DIM, N_CTX])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[V, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[1, HEAD_DIM])
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[N_CTX], strides=[1],
                                         block_shape=[N_CTX])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[BLOCK_M, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])

    q = q_desc.load([0, 0])                       # [BLOCK_M, HEAD_DIM]
    kt = kt_desc.load([0, 0])                     # [HEAD_DIM, N_CTX], host-pre-transposed
    ids = ids_desc.load([0])                      # [N_CTX] i32 page-row indices
    v_rows = v_desc.gather(ids, 0)                # [N_CTX, HEAD_DIM] — rung 4 verbatim

    # The score tile [BLOCK_M, N_CTX]: the Kt plane is already transposed, so this is a
    # direct×direct dot with no trans anywhere (the only score spelling with a door —
    # see the header).
    qk = tl.dot(q, kt, out_dtype=tl.float16)      # [BLOCK_M, N_CTX]

    # Softmax over the KV axis, one value per query row — the tutorial's orientation.
    # Every reduce and transcendental truncates straight back to f16 so the widening
    # stays a local extf/op/truncf ISLAND (the shape LegalizeTypes collapses); N_CTX is
    # one f16 stick, so the reduce-MAX defect cell cannot arise. exp2, not exp, because
    # exp2 is the free transcendental on this device; the reference mirrors it exactly.
    m = tl.max(qk, 1).to(tl.float16)              # [BLOCK_M]
    p = tl.math.exp2((qk - m[:, None]).to(tl.float32)).to(tl.float16)
    l = tl.sum(p, 1).to(tl.float16)               # [BLOCK_M]

    # The value contraction — THE RUNG-4 GATHER, unchanged from `paged_score.py`:
    # gathered `[N_CTX, HEAD_DIM]` rows in row-blocked order ARE the slot's `[k, n]`
    # residency, contracted where they lie. p is already [BLOCK_M, N_CTX], so no
    # transpose is needed on this leg either.
    acc = tl.dot(p, v_rows, out_dtype=tl.float16)  # [BLOCK_M, HEAD_DIM]
    o = acc / l[:, None]
    o_desc.store([0, 0], o)


def inputs(seed: int = 0, m=64, n_ctx=64, v=1024, head_dim=64):
    """Randomized fp16 q, kt (already transposed), and V table, in the kernel's geometry.
    The ids are NOT drawn here: they are pinned to `PAGED_ATTN_IDS` (a permutation), the
    same contract as `paged_score.py` — the negative control needs a KNOWN permutation."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(m, head_dim, generator=g, dtype=torch.float16)
    kt = torch.randn(head_dim, n_ctx, generator=g, dtype=torch.float16)
    table = torch.randn(v, head_dim, generator=g, dtype=torch.float16)
    return q, kt, table


def reference(q, kt, table, ids):
    """The oracle, from the host-side facts the kernel states: the Kt plane is this
    launch's K block pre-transposed, the V rows are `table[ids]`, and the softmax is the
    kernel's own base-2 form (exp2, no scale) so the oracle shares no scaling with the
    kernel beyond the function itself. `q @ kt` IS the score tile — kt is k.T, so no
    transpose appears in the oracle either (checked against a textbook base-e SDPA in
    numpy before this shipped: identical to f16 rounding)."""
    qk = q.float() @ kt.float()                   # [m, n_ctx] — the kernel's tile
    m = qk.max(dim=1).values
    p = torch.exp2(qk - m[:, None])
    l = p.sum(dim=1)
    o = (p @ table[ids.long()].float()) / l[:, None]
    return o.to(torch.float16)


PAGED_ATTN_IDS = torch.tensor([
    53, 29, 32, 22, 39, 40, 28, 26, 50, 42, 49, 20, 55, 43, 58, 47,
    0, 57, 37, 2, 8, 15, 24, 31, 7, 38, 36, 21, 52, 60, 27, 56,
    45, 51, 62, 10, 12, 44, 41, 54, 13, 16, 61, 11, 17, 23, 34, 33,
    14, 3, 48, 18, 35, 59, 1, 4, 5, 9, 63, 25, 46, 6, 19, 30,
], dtype=torch.int32)
