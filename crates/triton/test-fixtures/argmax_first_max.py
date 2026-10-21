# SPDX-License-Identifier: Apache-2.0
"""argmax_first_max fixture: the greedy sampler's vocab scan, FIRST-MAX by
arithmetic, as one kernel.

This is the ladder form of loop closure's device sampler (#603/#500): the
algorithm that used to live in a hand-minted 8-link SuperDDC composite
(#698, deleted by #714) is stated HERE, in the kernel, over generic ops only
-- stick-local `tt.reduce` max, elementwise running combines, an equality
compare, the i32 widen, the staged ramp table's multiply. Nothing
argmax-specific exists below the kernel: whatever lowering the splice emits
for `tt.reduce`/`arith.cmpf`/`arith.muli` is the lowering this kernel gets,
which is the whole point.

THE ALGORITHM (two passes over the row, both online)
----------------------------------------------------
    pass 1:  m = running max of the logits row
    pass 2:  w = (x == m) * ramp,  ramp_i = V - i on real lanes, 0 on pad
                       (ramp LOADED from the staged table -- host contract)
             j = running max of w
    idx      = V - j

FIRST-MAX BY ARITHMETIC, not by a tie-break rule: the equality mask is true at
EVERY lane holding the maximum, and the ramp V - i is STRICTLY DECREASING in
i, so the max of w is attained exactly at the SMALLEST maximal lane. No
comparator, no select, no index-carrying reduce pair -- the tie law is the
ramp's monotonicity. This is #631's first-max law (host argmax, metal, and
the deleted composite all agree on it) made structural.

THE INDEX MATH IS i32, MANDATORILY: fp16 exact integers stop at 2048 and the
vocab is 49k-262k, and f32 is not an answer either -- the device is f16 and
`make_ktir`'s LegalizeTypes REFUSES genuine f32 compute by design (the
SP-E2-03 RED-stop: degrading it to f16 is an owner decision, and it would be
the wrong one, since f16 cannot hold 49k exactly either). So the ramp, the
mask multiply and the running combine all run in i32 lanes -- exact at any
vocab width, and the ingestion dtype for index tables is SENUINT32 anyway.
The LOGITS stay in their native fp16 domain end to end -- the max and the
equality compare never leave it.

THE SENTINEL LAW (all-NaN row), derived not patched: with maxnumf-style
combines a NaN lane never wins the running max, so an all-NaN row leaves m at
the -inf of the pad lanes; `x == -inf` is false on every NaN lane and true on
the pad lanes, whose ramp is 0, so w is identically zero, j = 0, and
idx = V - 0 = V -- one past the last real lane. With NaN-PROPAGATING combines
m itself is NaN, `x == NaN` is false everywhere, and w is zero again: idx = V
under EITHER NaN semantics, by construction. The caller treats idx == V as
the loud refusal (the host argmax answers token 0 over the same row; that
divergence is recorded, not reconciled here).

DEVICE LAWS THIS SHAPE OBEYS
----------------------------
* BLOCK IS ONE STICK (64) and every `tt.reduce` in the kernel reduces a
  [1, BLOCK] tile over its stick axis to [1] -- the golden rank-2 -> rank-1
  reduce form. attention_sdpa.py documents why: a Spyre reduce-MAX over more
  than one stick of columns mis-combines partial per-tile maxima, so
  cross-stick combines here are ELEMENTWISE (`tl.maximum` on the running
  [1]s), never wider reduces.
* V AND PAD_V ARE constexpr (SPYRE-SPECIFIC): affine loop bounds are static,
  the emitter derives work division from descriptor extents, and Spyre
  compiles one kernel per shape. Same law as every other fixture.
* ONE PROGRAM (grid [1]): the sampler serves the solo greedy decode step's
  single [1, V] row. Batched/sampled steps keep the host sampler -- they need
  the whole distribution, not one index.

HOST CONTRACT (the part that is not in this file)
-------------------------------------------------
TWO staged buffers, both host obligations for the same reason embedding.py pads
nothing in-kernel and attention_sdpa.py's causal mask is host-precomputed: the
buffers are staging rows the host already owns, and `other=`-masked loads are
not in this frontend's vocabulary.

* THE LOGITS ROW is presented as [PAD_V] fp16, PAD_V = the vocab rounded UP
  to a multiple of 64, with the pad lanes already holding -inf (the fp16
  floor, the same constant the running max initializes from). Real lanes are
  untouched: adding -inf past the row cannot change the max, and a pad lane
  can never win the equality against a finite m.
* THE RAMP TABLE is [PAD_V] i32 with ramp_i = V - i on the real lanes and 0
  on the pad lanes -- strictly decreasing, so the first-max law is its
  monotonicity. It is CONSTANT across decode steps (a function of V alone), so
  the host stages it ONCE; under the pooled-substrate regime it lives in a
  stable slot and every step's CB reads the same address. Computing it
  in-kernel would need `V - (start + tl.arange(...))` -- a runtime scalar
  added to an arange in the tile domain, a form no fixture carries (see the
  pass-2 comment).

The output is ONE i32 lane (the gather ingestion law: index tables are
SENUINT32 at wordLength 4), stored at word 0 of a [1, 64] i32 stick so the
store is a whole stick like every other fixture's.

`reference()` is native PyTorch with the tie law stated EXPLICITLY (mask,
then the first true index) rather than `torch.argmax`, whose tie behavior is
a backend promise rather than an arithmetic fact -- this fixture exists to
pin first-max, so the reference must derive it, not assume it.
"""

import torch

import triton
import triton.language as tl

V = 49155  # granite-3.1's vocab: NOT a multiple of 64 -- PAD_V carries the 5 pad lanes
PAD_V = 49216  # V rounded up to a multiple of 64
BLOCK = 64
NEG_INF = float("-inf")
DTYPE = torch.float16


@triton.jit
def argmax_first_max_kernel(logits_ptr, ramp_ptr, idx_ptr, V: tl.constexpr,
                            PAD_V: tl.constexpr, BLOCK: tl.constexpr):
    # THE SHAPES ARE THE GOLDEN FORMS, end to end. The row is presented as [1, PAD_V] and
    # every block is [1, BLOCK] (rank 2), so each per-stick `tl.max(x, axis=1)` is the
    # rank-2 -> rank-1 reduce every fixture's reduce takes. The accumulators `m` and `j`
    # are [1, BLOCK] vectors with EVERY LANE EQUAL: the reduce's [1] answer is expanded to
    # [1, 1] (`tl.expand_dims` -- Triton has no implicit rank promotion) and
    # `tl.maximum`'s own unit-dim broadcast fills the row, so the cross-stick combine is
    # pure elementwise (the attention fixture's law) and no scalar, no reduce-to-scalar,
    # and no splat of an interior value exists anywhere in the kernel.
    m = tl.full([1, BLOCK], float("-inf"), tl.float16)
    for start in range(0, PAD_V, BLOCK):
        l_desc = tl.make_tensor_descriptor(logits_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                           block_shape=[1, BLOCK])
        x = l_desc.load([0, start])
        # One-stick reduce over the stick axis -- the sanctioned reduce shape.
        blk_max = tl.max(x, axis=1)
        m = tl.maximum(m, tl.expand_dims(blk_max, 1))

    # Pass 2: w = (x == m) * ramp, running max -- ALL IN i32. The ramp is the STAGED
    # TABLE (host contract below): strictly decreasing, so max(w) lands on the FIRST
    # maximal lane; 0 on the pad lanes, which can never win. i32 because fp16 exact
    # integers stop at 2048 (V is 49k) and f32 is refused by the device's f16
    # legalization -- see the docstring's index-math law. Loading the ramp rather than
    # computing `V - (start + tl.arange(...))` follows the attention fixture's mask law:
    # no fixture adds a runtime scalar to an arange in the tile domain (attention's
    # aranges are dead code in every golden), so the precomputed table is the carried
    # form -- and it is CONSTANT across decode steps, a stage-once pool citizen.
    j = tl.full([1, BLOCK], 0, tl.int32)
    for start in range(0, PAD_V, BLOCK):
        l_desc = tl.make_tensor_descriptor(logits_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                           block_shape=[1, BLOCK])
        x = l_desc.load([0, start])
        r_desc = tl.make_tensor_descriptor(ramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                           block_shape=[1, BLOCK])
        ramp = r_desc.load([0, start])
        mask = (x == m).to(tl.int32)
        w = mask * ramp
        blk_w = tl.max(w, axis=1)
        j = tl.maximum(j, tl.expand_dims(blk_w, 1))

    # idx = V - j: every lane holds the same first-max index (or V itself on the
    # all-NaN/empty row -- the sentinel, one past the last real lane, never a valid
    # token). The store is the whole [1, BLOCK] i32 stick, like every other fixture's;
    # the consumer reads word 0.
    idx = V - j
    o_desc = tl.make_tensor_descriptor(idx_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                       block_shape=[1, BLOCK])
    o_desc.store([0, 0], idx)


def inputs(seed: int = 0):
    """One fp16 logits row, pad lanes -inf, with a deliberate TIE at the max:
    lanes 100 and 5000 both hold the row's largest value, so the reference
    pins FIRST-max against a real tie, not a vacuous one."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(V, generator=g, dtype=DTYPE)
    top = x.max()
    x[100] = top
    x[5000] = top
    row = torch.full((PAD_V,), NEG_INF, dtype=DTYPE)
    row[:V] = x
    return row


def reference(row: torch.Tensor) -> torch.Tensor:
    """First-max argmax, derived: the equality mask's first true index. Pad
    lanes are -inf and can never tie a finite max; an all-NaN real row leaves
    the mask empty and answers V (the sentinel), matching the kernel's
    arithmetic. Returned as a 1-lane int32 tensor (the ingestion dtype)."""
    x = row[:V]
    m = x.max()
    mask = x == m
    hit = torch.nonzero(mask)
    idx = hit[0].item() if hit.numel() > 0 else V
    return torch.tensor([idx], dtype=torch.int32)
