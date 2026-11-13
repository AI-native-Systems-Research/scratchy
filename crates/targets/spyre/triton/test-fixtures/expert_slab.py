# SPDX-License-Identifier: Apache-2.0
"""ONE routed expert's projection: the slab gather a mixture-of-experts layer is built on.

    out[p, :] = x[p, :] @ W[expert_ids[p]].T          W : [E, N, K]

The upstream spelling, read from `triton_kernels/matmul_details/_p_matmul.py:423`
(`w = tl.reshape(W.load([off_w_z, off_w_n, off_k_w]), W.block_shape[1:]).T`), with the expert
id taken by a SCALAR pointer read (`_p_matmul.py`'s `tl.load(XSliceSizes + off_w_z)` /
`tl.load(WScale + expt_id1)` idiom). That is the whole point of this fixture: the expert axis
of a RANK-3 weight descriptor is offset by a value READ FROM MEMORY, so the address depends on
data. It is a different construct from `embedding.py`'s `tt.descriptor_gather`:

  * the gather selects ROWS by an index VECTOR (`block_shape[0] == 1`, >= 8 indices);
  * this selects ONE WHOLE SLAB by ONE index: a `tt.descriptor_load` over `[1, N, K]` whose
    leading offset is the scalar `tt.load` of `expert_ids[p]`.

THE DELTA, and the justification for each item
==============================================

1. EVERY SHAPE IS constexpr (SPYRE-SPECIFIC), for `embedding.py`'s delta 2 reason: the
   emitter derives the work division from the descriptor extents, and Spyre compiles one
   kernel per shape.

2. ONE PAIR PER PROGRAM, unrolled by `tl.static_range` (SPYRE-SPECIFIC). The pair index is
   then a Python int at trace time, so every corner except the expert offset is a constant --
   the ladder's own law (a `tl.range` induction variable is a runtime scalar and cannot
   splat). Upstream grids over pairs instead; the arithmetic per pair is the same.

3. THE DOT ACCUMULATES IN f16 (SPYRE-SPECIFIC), `matmul_fp8.py`'s spelling: the ladder's
   `dot_to_linalg` admits only an f16 dot result. This fixture isolates the ADDRESSING, so it
   carries no fp8 and no epilogue.

NOT CHANGED: the rank-3 descriptor, the scalar id read, the reshape that drops the expert
axis, and the transpose into `tl.dot` -- exactly upstream's four steps.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def expert_slab_fwd(desc_ids, desc_w, desc_x, desc_o,  #
                    P: tl.constexpr, E: tl.constexpr,  # SPYRE: delta 1
                    N: tl.constexpr, K: tl.constexpr,  #
                    ):
    # The stacked expert bank: one [N, K] slab per expert. The block is ONE slab.
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[E, N, K], strides=[N * K, K, 1],
                                       block_shape=[1, N, K])
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[P, K], strides=[K, 1],
                                       block_shape=[1, K])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[P, N], strides=[N, 1],
                                       block_shape=[1, N])
    for p in tl.static_range(P):  # SPYRE: delta 2
        # THE INDIRECTION: the routed expert of pair `p`, read from memory.
        expt = tl.load(desc_ids + p)
        w = tl.reshape(w_desc.load([expt, 0, 0]), [N, K])
        x = x_desc.load([p, 0])
        o = tl.dot(x, w.T, tl.zeros([1, N], dtype=tl.float16),
                   out_dtype=tl.float16)  # SPYRE: delta 3
        o_desc.store([p, 0], o)


# --- the configurations ----------------------------------------------------------
# gemma-4-26B-A4B's routed geometry: 128 experts, moe_intermediate_size 704, hidden 2816
# (`crates/models/arch/configs/gemma4-moe/gemma-4-26b-a4b-it.json`), one token's 8 picks.
GEMMA4_26B = dict(p=8, e=128, n=704, k=2816)

SIGNATURE = {
    "desc_ids": "*i32", "desc_w": "*fp16", "desc_x": "*fp16", "desc_o": "*fp16",
    "P": "constexpr", "E": "constexpr", "N": "constexpr", "K": "constexpr",
}


def constexprs(p=8, e=8, n=64, k=128):
    """One configuration. Shapes are constexpr (delta 1), so a shape change is a recompile."""
    return {"P": p, "E": e, "N": n, "K": k}


def inputs(seed: int = 0, p=8, e=8, n=64, k=128):
    """Routed ids in [0, E) (i32, as a scalar pointer read), an f16 bank and f16 rows."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, e, (p,), generator=g, dtype=torch.int32)
    w = torch.randn(e, n, k, generator=g, dtype=torch.float16)
    x = torch.randn(p, k, generator=g, dtype=torch.float16)
    return ids, w, x


def reference(ids: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """`out[p] = x[p] @ W[ids[p]].T`, the routed projection with no fusion."""
    return torch.stack([x[i] @ w[ids[i].item()].T for i in range(ids.shape[0])])
