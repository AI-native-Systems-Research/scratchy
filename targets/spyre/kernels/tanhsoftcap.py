# SPDX-License-Identifier: Apache-2.0
"""THE TANH-SOFTCAP SPLICE'S KERNEL — `out = cap · tanh(x / cap)`, gemma-4's
logit softcap.

BODY PROVENANCE: main's `KtirFunc::tanhsoftcap` — the splatted cap feeds ONE
`arith.divf`, the tanh runs on the quotient, and the SAME splat multiplies the
tanh's result back up: `splat(cap) → divf → tanh → mulf`. The cap reaching the
program as ONE splat feeding the divide and ONE feeding the multiply is not
stylistic — the consumer's `program_tanhsoftcap_cap` reads the cap OFF THE
PROGRAM (every `math.tanh` root's feeding `arith.divf` must carry exactly one
`splat(constant)` operand, and every root must agree), so folding either use
away, or spelling the cap a second way, would desync the program from the cap
the door reads.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's one input, then the
  output.
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N`, `CAP`, `N_BLOCKS`, `TAIL_H` —
  stated by the splice from the node's own region (`M =` rows, `N = BLOCK_N =`
  the region's width) and its `SubOp::TanhSoftCap { cap }` payload (the producer
  refuses `cap <= 0`).
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::tanhsoftcap`): `cols = out.c_len` — the
FIRST access tile's width — through `check_pointwise_cols`, so every block
states the region's full width and the door pads the device width itself,
exactly as it does for the builder's logical one.

⛔ THE TANH ISLAND, FORCED AND THEN COLLAPSED — `tl.tanh` is
`@_check_dtype(["fp32", "fp64"])` in `python/triton/language/math.py` (the same
family as `tl.exp`/`tl.rsqrt`), so the kernel widens the quotient, calls it, and
truncates straight back; LegalizeTypes collapses the island so the emitted KTIR
carries `math.tanh` on the f16 tile with no extf/truncf — the divf feeding the
tanh DIRECTLY, which is the shape `program_tanhsoftcap_cap` requires. `tl.fdiv`,
not `/`: `/` upcasts f16→f32 (the silumul delta-6 law).

⭐ ROW BLOCKING, THE BUILDER'S OWN LAW (`lower_tanhsoftcap_node`): a whole
`[rows, cols]` region whose three-tile live set (`rows × cols × 3`) exceeds the
`EW_LX_ELEMS` budget is ROW-BLOCKED, not refused — the softcap spans the
VOCAB-wide logits, wider than any activation, so prefill rungs exceed the budget
while decode (m=1) never does. `N_BLOCKS` full `[BLOCK_M, N]` tiles
(BLOCK_M = the builder's own `rows_per_block(cols, 3)`) plus a `[TAIL_H, N]`
tail tile when the region does not divide evenly. The trip count is constexpr,
so `to_ktir` unrolls it and the spliced program is straight-line with one
constant-corner tile per block — the door folds those through `Region::r_cover`
and ONE descriptor spans all of them.
"""

import triton
import triton.language as tl


@triton.jit
def tanhsoftcap_fwd(desc_x, desc_o,  #
                    M: tl.constexpr, N: tl.constexpr,  #
                    BLOCK_M: tl.constexpr,  #
                    BLOCK_N: tl.constexpr,  #
                    CAP: tl.constexpr,  #
                    N_BLOCKS: tl.constexpr,  #
                    TAIL_H: tl.constexpr,  #
                    ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                        block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                        block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        x = x_desc.load([offs_m, 0])
        # x / cap — the splatted cap as the divide's one constant operand.
        xc = tl.fdiv(x, CAP)
        # tanh, inside the f32 island the dtype check forces; LegalizeTypes
        # collapses it so the tanh's operand IS the divf.
        t = tl.tanh(xc.to(tl.float32)).to(tl.float16)
        # t * cap — the SAME splat, multiplied back up.
        o_desc.store([offs_m, 0], t * CAP)
    if TAIL_H > 0:
        t_x = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        x = t_x.load([offs_m, 0])
        xc = tl.fdiv(x, CAP)
        t = tl.tanh(xc.to(tl.float32)).to(tl.float16)
        t_o.store([offs_m, 0], t * CAP)
