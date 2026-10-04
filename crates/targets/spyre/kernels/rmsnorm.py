# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton rmsnorm kernel — the FIRST SPLICED KERNEL.

BODY PROVENANCE: `crates/triton/test-fixtures/rmsnorm.py`'s `rmsnorm_fwd`, verbatim —
the body triton-spyre card-validated (0.999908 within_2pct, rmsnorm_granite
configuration) through the same KTIR→SuperDSC lowering this splice hands it to. The
deltas recorded there are the kernel's own design notes (the f32 island around
`tl.rsqrt`, the constexpr-folded reciprocal, the 1-D weight descriptor) and are not
restated here; read them there.

WHY THIS FILE EXISTS SEPARATELY from the fixture: the fixture is the LADDER's test
surface (its `constexprs()`/`inputs()`/`reference()` helpers are test code); this is
the KERNEL a model runs, compiled at `#[forward]` expansion time by
`crates/triton/splice` for every `SubOp::RmsNorm { gain: Scale }` node. The registry
row names this file; `test-fixtures/rmsnorm.py` stays the ladder's.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_w`, `desc_o` — the node's operand order
  (x, gamma) then the output. The registry does not permute.
* CONSTEXPRS: `M`, `D_MODEL`, `BLOCK_M`, `EPS`, `INV_D` — stated by the splice from
  the node's own region (`M = BLOCK_M =` the region's row count, `D_MODEL =` its
  width, `EPS =` the tape's epsilon, `INV_D = 1/D_MODEL` folded on the host).
* GRID: `[1]`.

⛔ THE SUM IS f32, MATCHING THE BUILDER'S OWN PROGRAM, AND THE DIFFERENCE IS
NUMERICALLY REAL ON BOTH HALVES. The card's `EXX2_ZEROMEAN` reduce and the builder's
hand-written KTIR both accumulate the mean of squares in f32; a Triton `tl.sum` over
an f16 block accumulates in f16, whose 65504 ceiling an RMS above ~4 at D_MODEL=4096
(or one element ≥256, at any width) blows through — an activation outlier in the
residual stream is enough. The kernel therefore widens the squares to f32 before the
reduce (`(x * x).to(tl.float32)`), which is also what the emulator's numeric oracle
compares against. The CARD is unaffected either way — it runs the descriptors, and
the descriptor's EXX2_ZEROMEAN reduce is the device's own precision — but the
spliced KTIR program must not state a DIFFERENT computation from the builder's, or
the emulator stops emulating the card for this op.
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd(desc_x, desc_w, desc_o,  #
                M: tl.constexpr, D_MODEL: tl.constexpr,  # SPYRE: delta 4
                BLOCK_M: tl.constexpr,  #
                EPS: tl.constexpr,  #
                INV_D: tl.constexpr,  # SPYRE: delta 2
                ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])
    # HF's weight is 1D (`nn.Parameter(torch.ones(hidden_size))`), so the descriptor is too.
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[D_MODEL], strides=[1],
                                       block_shape=[D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])

    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
    # EXX2_ZEROMEAN: the mean of squares, with NO mean subtracted. One reduce along the
    # hidden axis in f32 (the accumulator's own precision — see the module doc), one
    # multiply by the constexpr reciprocal (delta 2).
    ms = tl.sum((x * x).to(tl.float32), 1) * INV_D
    # RSQRT, inside the f32 island `tl.rsqrt`'s own dtype check forces (delta 1).
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    w = w_desc.load([0])
    # `x * r * weight`. HF spells the last multiply `self.weight * hidden_states`; the same
    # two operands in the other order, which for one IEEE multiply is the same result.
    o_desc.store([offs_m, 0], x * r[:, None] * w[None, :])
