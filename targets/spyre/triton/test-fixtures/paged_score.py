"""RUNG 4 — the paged V-leg contraction through the whole Triton ladder.

`tl.dot(p, v_rows)` where `v_rows = v_desc.gather(ids, 0)`: the contraction's K
axis reads rows of the V table chosen AT RUNTIME by an index vector from device
memory (rung 4 of the address-provenance ladder). This is the SCRATCHY FORM,
chosen because it is the only one the arch can run:

* the gathered rows `[K, HEAD_DIM]` in row-blocked order ARE the matmul slot's
  `[k, n]` residency ("RowBlocked vs Kernel is the identical stick-blocked
  formula"; scratchy's shipped attention contracts its V cache in place exactly
  so), so the contraction is DONE WHERE THE ROWS LIE -- no relayout, no
  transpose, nothing but the gather and the dot.
* the K leg (`tl.dot(q, k_rows.T)`) is deliberately NOT spelled here: it needs a
  transposed gather, and a relayout op is dxp-refused on this arch ("Implicit
  syncs not available for architectures prior to RCUDD1A", measured at 16 cores
  and at 1). scratchy's answer is the third physical Kt plane the host presents
  -- a buffer-layout decision, not a kernel op -- and `dot_to_linalg`'s paged
  contract refuses the transposed form by name rather than let the door meet it
  at its own guard.

`K = 64` gathered rows because two floors meet there: the Triton frontend
requires `tl.dot`'s K >= 16, and the vendor matmul emitter refuses a K that is
not a whole 64-element fp16 stick (`assemble_matmul` emits no
`coordinateMasking_`), so 64 is the smallest K both doors admit.

The ids are a PERMUTATION of 0..63, and that is the negative control: with the
ids SORTED, the kernel computes `p @ v[:64]` and the result must DIFFER from the
permutation reference -- a match there is the direct-read trap (the table's
first rows read in table order), which is exactly the wrong-rows failure the
whole-function door's gather guard exists to refuse.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def paged_vmatmul_fwd(desc_p, desc_v, desc_ids, desc_o,  #
                      M: tl.constexpr, K: tl.constexpr, V: tl.constexpr,  #
                      BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    p_desc = tl.make_tensor_descriptor(desc_p, shape=[M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, K])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[V, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[1, HEAD_DIM])
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[K], strides=[1],
                                         block_shape=[K])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    p = p_desc.load([0, 0])
    ids = ids_desc.load([0])
    rows = v_desc.gather(ids, 0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16)
    o = tl.dot(p, rows, acc)
    o_desc.store([0, 0], o)


def inputs(seed: int = 0, m=8, k=64, v=128, head_dim=64):
    """Randomized fp16 `p` and table, in the case's own geometry. The ids themselves
    are NOT drawn here: the case pins them to `PAGED_VMATHMUL_IDS` (a permutation),
    because the negative control needs a KNOWN permutation, not a random one."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(m, k, generator=g, dtype=torch.float16)
    table = torch.randn(v, head_dim, generator=g, dtype=torch.float16)
    return p, table


def reference(p, v, ids):
    return (p.float() @ v[ids.long()].float()).to(torch.float16)


PAGED_VMATHMUL_IDS = torch.tensor([
    53, 29, 32, 22, 39, 40, 28, 26, 50, 42, 49, 20, 55, 43, 58, 47,
    0, 57, 37, 2, 8, 15, 24, 31, 7, 38, 36, 21, 52, 60, 27, 56,
    45, 51, 62, 10, 12, 44, 41, 54, 13, 16, 61, 11, 17, 23, 34, 33,
    14, 3, 48, 18, 35, 59, 1, 4, 5, 9, 63, 25, 46, 6, 19, 30,
], dtype=torch.int32)
