# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton rmsnorm kernel — the FIRST SPLICED KERNEL.

BODY PROVENANCE: `crates/triton/test-fixtures/rmsnorm.py`'s `rmsnorm_fwd`, verbatim
EXCEPT the ceiling-safe pre-scale below (the fixture keeps the card-validated body
verbatim — it is the ladder's own test surface). The card's emission is unchanged by
the delta: the door re-assembles the exx2 sequence from the program's regions,
epsilon and gain offset, never from the mean's spelling (`program_rmsnorm_eps`'s own
discriminator is "any spelling of the mean").

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

⛔ THE SUM IS f16-TYPED, AND NO SPELLING CAN WIDEN IT — SO THE ACCUMULATOR IT
RUNS IN IS THE CARD'S. `LegalizeTypes` (SP-E2-03) erases EVERY `arith.extf`
f16→f32 — Triton's auto-widening and a kernel's own `.to(tl.float32)` are the
same op to it — and retypes the `tt.reduce` combiner to f16 with the rest of
the island. This is not our port's drift: the C++ chain's own oracle
(`crates/triton/test-experiment1/ktir/rmsnorm_granite.ktir.mlir`) records
exactly this form — `x·x` f16, an `arith.addf : f16` combiner, `math.rsqrt`
f16. A widening written into this file is dead on arrival. But an f16-TYPED
reduce does not mean an IEEE-f16 ACCUMULATOR: on the card this reduce lowers to
the FMA16 templates (`sum`/`mean`/`exx2` — `summeanmaxexx2.ddl`,
`exx2_fwd.smc`), whose accumulator is DLF16 (`SEN169_FP16`: 6-bit exponent,
bias 31, 9-bit mantissa, NO subnormals, magnitudes to ~8.58e9 — the vendor's
own `util/sendefs/numeric_convert.h`), and the emulator models exactly that
(`codec::round_dlf16` on the tree-fold's per-round partials — the fma-class
law in `fast_tree_fold`). The kernel therefore states the computation PLAIN:
`tl.sum(x·x, 1) · INV_D`, no pre-scale, no restore.

⛔ THE PRE-SCALE IS DEAD — TWO MEASURED FAILURES, BOTH EMULATOR ARTIFACTS OF
IEEE-f16 ACCUMULATOR MODELING. The first attempt dodged a 65504 "ceiling"
(Σx² > 65504 → inf → `rsqrt(inf)` = 0 → exact-zero norms at gemma-4-12b's
layers 18/22/48, Σx² = 89,771 / 94,669 / 91,950 at D=3840) by squaring
`x·2⁻³` and folding the restore into one `64·INV_D` constant. But the model's
norm rows span a 3e7 dynamic range of mean-of-squares (small-stream sandwich
norms at layers 2-4 down to ~2.5e-5, layer-18 rows up to ~25), and IEEE f16's
usable square-sum window cannot hold both ends: the pre-scale fixed the
ceiling and moved the damage to the small rows — the pre-scaled partials
(Σx²/64) landed in f16's SUBNORMAL band, and the emu gate's mid-band tensors
went 20-60× WORSE (first divergence t850→t824, count 678→696 of 738; the
layer-18 family improved only because its cascade feeds through the damaged
band). No power-of-two pre-scale serves a range that wide — the fix is the
accumulator model, and with DLF16 modeled, the plain form holds every band:
Σx² 95k is four orders inside the max, small-row partials (down to ~2e-4) are
seven orders above the 2⁻³⁰ flush, and the card itself ran this plain body
card-coherent (the W13 Paris pass) long before any of this.

The CARD was never affected by either band — its accumulator is DLF16, so
Σx² ≈ 9.5e4 sits five orders inside range and small-row squares flush only
below |x| = 2⁻¹⁵ (the model's smallest are ~3e-3). The ceiling and the crush
existed ONLY in the emulator's old IEEE-f16 accumulator rounding, and that
rounding is what the `codec::round_dlf16` law replaces. But the spliced KTIR
program still must not state a computation that diverges from the builder's
numerically — which is one more reason the plain form is the law: it is the
body the card was validated on, verbatim.

THE `(1 + w)` CONVENTION (`rmsnorm_oneplus_fwd`, a SECOND ENTRY in this same file):
the gemma class stores the gain zero-centred (`weight = gain − 1`, the HF
convention), so the applied gain is the loaded row PLUS `GAIN_OFFSET`. The offset
is stated as an `addf` over the load with a constant splat — the ONE shape the
door's `program_gain_offset` reader accepts — and the door resolves the SAME
value's `[1,1]` registry slot by bits, exactly the epsilon's law: the program
states the constant, the descriptor adds the registry's copy of it. The Scale
entry above states NO addf (the reader's `None`), which is why the conventions are
two entries and not one entry with a flag — neither program carries an op its
convention does not state, and the Scale program stays byte-identical to the form
the card was validated on.
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
    # EXX2_ZEROMEAN: the mean of squares, with NO mean subtracted. One reduce along
    # the hidden axis, stated PLAIN — the module doc's law: the reduce is f16-typed
    # (no widening survives LegalizeTypes) but its ACCUMULATOR is the card's DLF16,
    # which the emulator models, so no pre-scale is needed or allowed.
    ms = tl.sum(x * x, 1) * INV_D
    # RSQRT, inside the f32 island `tl.rsqrt`'s own dtype check forces (delta 1) — the
    # island collapses to f16 like every other, the C++ oracle's own form.
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    w = w_desc.load([0])
    # `x * r * weight`. HF spells the last multiply `self.weight * hidden_states`; the same
    # two operands in the other order, which for one IEEE multiply is the same result.
    o_desc.store([offs_m, 0], x * r[:, None] * w[None, :])


@triton.jit
def rmsnorm_oneplus_fwd(desc_x, desc_w, desc_o,  #
                        M: tl.constexpr, D_MODEL: tl.constexpr,  #
                        BLOCK_M: tl.constexpr,  #
                        EPS: tl.constexpr,  #
                        INV_D: tl.constexpr,  #
                        GAIN_OFFSET: tl.constexpr,  #
                        ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[D_MODEL], strides=[1],
                                       block_shape=[D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])

    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
    # The plain entry's own chain (module doc's law): the mean of squares stated
    # PLAIN — the reduce's accumulator is the card's DLF16, which the emulator
    # models — then the reciprocal.
    ms = tl.sum(x * x, 1) * INV_D
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    # ⭐ THE (1 + w) GAIN LEG — the loaded row plus the offset, as ONE `arith.addf`
    # over the load with the offset's constant splat. The f32 literal splat
    # island-collapses to f16 (it feeds an addf with an f16 operand — LegalizeTypes'
    # own law), so the applied gain is f16(1 + w) and the trailing multiplies stay
    # f16 throughout. `GAIN_OFFSET` is 1.0 by the convention's definition
    # (`GainConvention::from_offset` refuses every other value), and the door reads
    # THIS splat's value back off the program to resolve its registry slot — the
    # epsilon's law, one constant, two readers, one value.
    w = w_desc.load([0]) + GAIN_OFFSET
    o_desc.store([offs_m, 0], x * r[:, None] * w[None, :])
