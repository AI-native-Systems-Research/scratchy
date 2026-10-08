// SPDX-License-Identifier: Apache-2.0
//! THE PROVEN fp8 W8A8 MATMUL DESCRIPTORS, driven from the KTIR.
//!
//! ⭐⭐⭐ THE BODY IS `ibm/main`'s `lower_matmul_node` arity-3 branch, UNCHANGED — the whole W8A8 chain:
//! per-token activation quantize (`abs` → `amax` → scale → clamp → `qfp8ch`), `matmulfp8`
//! (fp8×fp8→fp16), then dequant by `a_scale`(per row) · `w_scale`(per channel). Only its DOOR changed:
//! it takes [`Fp8Facts`] recovered from the KTIR program instead of a `&SubtileNode`.
//!
//! ⛔ AND THAT CHAIN IS WHY fp8 NEEDS THIS. `KtirFunc::matmul_fp8` computes **fp16 activation × fp8
//! weight** — its own doc says outright that it does not quantize the activation — while this device's
//! fp8 is fp8×fp8. Lowering the KTIR op-for-op therefore cannot produce `matmulfp8` at all, and it
//! also addressed the weight as fp16: a `[2048, 2048]` fp8 weight is placed 4 MB and was being read
//! 8 MB. fp8 W8A8 runs on hardware today through this code, so the card keeps using it — reached
//! through the KTIR, from facts the KTIR itself states.

use super::{
    EmittedOp, In, assemble_convert, assemble_pointwise_broadcast_off, bmm_site, emit_sdsc_tiled,
    op_func_from_str, pw1, pw2, rb, rbo,
};
use crate::ir::bridge::tiled_op_sdsc_op::reduce::{
    assemble_reduce_off, assemble_reduce_seeded, reduce_opspec,
};
use crate::place::{PlaceId, SynthRole};
use crate::placement::{BundleLayout, syn};
use crate::reserved_tids::{FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{DataFormat, Df, Fp16, Role, SdscFoldSet};

/// What the KTIR states about one fp8 W8A8 contraction, recovered from its shapes and parameters.
///
/// ⛔ EVERY FIELD IS READ OUT OF THE KTIR. The fp8-ness itself is too: `KtirFunc::matmul_fp8` builds
/// its weight view through `view_fp8`, which writes `Dtype = Fp8E4m3` on the
/// `ktdp.construct_memory_view` — so the recogniser asks the view what its element type is rather than
/// guessing from arity.
pub(crate) struct Fp8Facts {
    /// The activation's tensor id — the quant chain's tensors are keyed off it so two matmuls on one
    /// activation share a single quantize.
    pub a_tid: u32,
    pub a_name: String,
    /// The fp8 weight's buffer name.
    pub w_name: String,
    /// The per-output-channel `w_scale`.
    pub ws_name: String,
    /// The output tensor id.
    pub out_tid: u32,
}

pub(crate) fn matmul_fp8_descriptors(
    facts: &Fp8Facts,
    m: u32,
    k: u32,
    n: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    {
        // fp8 W8A8 PREFILL (m>1). torch-spyre's `quantize_fp8_with_scale`
        // (decompositions.py:966) takes `scale` as an input computed by the CALLER via native `torch.abs` +
        // `torch.ops.aten.amax` (confirmed via grep — no L2-norm substitute anywhere in torch-spyre); the
        // `#33` belief that reduce-MAX "seeds 0 for rows>1" was disproved by the stable-softmax `block_max`
        // (attn.rs), itself a multi-row MAX reduce. ONE formula for any row count (m==1 decode or m>1
        // prefill), matching torch-spyre exactly — mirrors rmsnorm's unification (no scratchy-only approx,
        // no diagnostic-sweep scaffolding left in the live path).
        let stk = Fp16::ELEMS_PER_STICK; // 64
        // ⭐ THE 26b K-PAD — the contraction width rounded to a whole 128-elem SEN143_FP8 stick
        // (`work::fp8_k_pad`, the K-axis twin of `DeviceWidth`'s N law). gemma-4-26b's dense GeGLU
        // DOWN gemm contracts k=2112 = 16.5 fp8 sticks — the FIRST fp8 model with a non-legal K
        // (granite/g8b/g12b are all 128-multiples) — and `qfp8ch`'s convert guard refuses a
        // sub-stick tile outright. The fp16 READS of this chain stay at the LOGICAL `[m, k]` (the
        // activation's own reservation is k-wide; reading past it is out-of-footprint); only the
        // chain's TAIL widens: `cl` (the clamp output) and `afp8` (the convert output) are
        // reserved and addressed at `[m, k_pad]`, with the pad window ZERO-FILLED by this chain's
        // own emission. The zero is inert by the chain's own arithmetic — the amax never sees the
        // pad lanes (it reads `absx` at the logical width), and 0 clamps to 0 and contributes
        // 0·w to the contraction — but the FILL is not optional: segments zero-init once, while
        // intermediates RE-USE bytes across steps, so an unwritten pad lane is last step's
        // activation (or a NaN), and `maximum`/`minimum` pass a NaN through UNCHANGED.
        let k_pad = crate::work::fp8_k_pad(k);
        let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(m);
        let k_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(k);
        let n_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(n);
        let scale_cols =
            crate::sdsc_abstract::BlockCols::of_one_stick(crate::sdsc_abstract::Lanes::FP16);
        // ⭐ THE NAMES COME FROM THE KTIR'S OWN PARAMETERS — `args[i]` → `wiring::act_name(tid)`, the
        // same spelling `ds_name` produced for the SubtileIR path.
        let a_name = facts.a_name.clone();
        let w_name = facts.w_name.clone(); // fp8 weight `t{id}` — declared fp8 on the matmul operand (set_df), not the name
        let ws_name = facts.ws_name.clone(); // w_scale, read [1,n] (byte-identical to the loaded [n,1])
        let out = crate::place::act_name(facts.out_tid);
        use SynthRole as R;
        // SHARED quant tensors are a pure function of the ACTIVATION (abs→amax→scale→clamp→fp8) — derive
        // them from the ACTIVATION's id so a second matmul on the same activation resolves the IDENTICAL
        // fp8 tensor + a_scale (the layout `synth` is idempotent per id → one reservation). PER-MATMUL
        // tensors (the matmul output `raw` and its dequant `dqa`) derive from the OUTPUT's id.
        // ⭐ THE DOOR: both ids come from the KTIR-recovered facts, not from a `&SubtileNode`. They are
        // the same tensor ids either way — `Fp8Facts::a_tid`/`out_tid` are recorded off the program's
        // own views — so the placements these mint are the ones the proven path minted.
        let a_id = PlaceId::Act(facts.a_tid);
        let out_id = PlaceId::Act(facts.out_tid);
        // The operand SPELLING is a rendering of the id the layout is keyed by, so the op's name and
        // the allocator's key cannot drift — they have one preimage.
        let qn = |r: R| syn(layout, a_id.synth(r));
        let nm = |r: R| syn(layout, out_id.synth(r));
        // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed), so
        // they stay strings and take no `PlaceId`.
        let opn = |s: &str| format!("{out}_{s}");
        let (absx, amax, amaxfl, ascale, invs, sc, chi, cl) = (
            qn(R::FqAbsX),
            qn(R::FqAmax),
            qn(R::FqAmaxFl),
            qn(R::FqAscale),
            qn(R::FqInvS),
            qn(R::FqSc),
            qn(R::FqChi),
            qn(R::FqCl),
        );
        let (raw, dqa) = (nm(R::FqRaw), nm(R::FqDqA));
        let afp8 = qn(R::FqAfp8); // quantized fp8 activation — sized/typed fp8 via synth_df + the convert
        if let Some(l) = layout {
            // Footprints are `m` query rows (decode m=1 ⇒ byte-identical to the old `[1,·]`; prefill m>1
            // reserves the real multi-row extent so a per-row write can't clobber a neighbor row).
            for r in [R::FqAbsX, R::FqSc, R::FqChi] {
                l.synth(a_id.synth(r), &[m, k]);
            }
            // The chain's tail widens to the PAD: `cl` feeds the convert (whose 128-elem stick law
            // is the reason the pad exists) and `afp8` is the fp8 contraction operand — both are
            // reserved at `[m, k_pad]`, the one width the convert, the matmulfp8, and the packed
            // weight's K rows all share.
            l.synth(a_id.synth(R::FqCl), &[m, k_pad]);
            l.synth_df(a_id.synth(R::FqAfp8), &[m, k_pad], Df::Fp8); // 1-byte / 128-stick residency (½ fp16)
            for r in [R::FqAmax, R::FqAmaxFl, R::FqAscale, R::FqInvS] {
                l.synth(a_id.synth(r), &[m, stk]);
            }
            for r in [R::FqRaw, R::FqDqA] {
                l.synth(out_id.synth(r), &[m, n]);
            }
        }
        let pos448 = rbo(&crate::place::act_name(FP8_POS448_TID));
        let neg448 = rbo(&crate::place::act_name(FP8_NEG448_TID));
        let inv448 = rbo(&crate::place::act_name(FP8_INV448_TID));
        let mut ops: Vec<EmittedOp> = Vec::new();
        // SHARE the activation quantize across matmuls that read the SAME activation. `insert` returns false
        // when this activation (`a_name`) was already quantized earlier in the bundle (e.g. K/V after Q, or
        // Up after Gate) — in that case skip re-emitting the whole abs→amax→scale→clamp→qfp8ch chain and
        // reuse its `afp8`/`ascale` (both named by `a_name`); only `matmulfp8` + dequant stay per-matmul.
        // (The chain is a launch/op cost paid ONCE per distinct activation instead of once per matmul.)
        let already_quantized = !quantized.insert(a_name.clone());
        if !already_quantized {
            // `absx`'s ONLY consumer is the MAX-reduce right below (never a matmul) — same bug class as
            // rmsnorm's `sq16` (see rmsnorm.rs): a shape-driven RowBlocked handle would make it stickmajor-
            // eligible purely from shape (rows>1 && cols>64, true here for any real hidden/intermediate `k`),
            // and the native reduce cannot correctly walk a stick-major activation along `cols` at rows>1 (it
            // sums/maxes across the wrong rows). Emitted via the TYPED `pw1`/`Stk<FlatTag>` path (not a raw
            // `&str` one) so the producer and this reduce's consumer are the SAME compile-time-checked
            // handle — `fl()` forces both ends flat; a kind mismatch would be a `cargo build` type error.
            // `absx` is RowBlocked, NOT Flat (2026-07-28) -- same fix, same reason, as rmsnorm's
            // `sq16` (see the long note in ir/bridge/tiled_op_sdsc_op/rmsnorm.rs). A Flat OUTPUT
            // handle forces head_major=true for the WHOLE op, which per pointwise.rs "stays rank-3"
            // and therefore also reads the INPUT activation rank-3 flat. But `a_name` is the
            // stick-major residual stream, so at m=mq=31 with k=2048 that read is (pointwise.rs's own
            // words) "scrambled at rows>1 AND cols>64": the per-row amax was taken over a scrambled
            // element set, giving every row the wrong fp8 activation scale -- on EVERY fp8 matmul
            // (qkv, o_proj, gate/up, down), every layer, for batched prefill only. RowBlocked puts
            // both this op and the amax reduce on the stick-major rank-2 path, matching how the
            // activation was actually written. At m==1 (decode) `stickmajor` is false in both, so the
            // emission is byte-identical to the previous Flat form.
            let absx_h = rb(&absx, m, k);
            ops.push(pw1(
                &opn("fq_absx_op"),
                "abs",
                m_rows,
                k_cols,
                In::full(&rb(&a_name, m, k)),
                &absx_h,
                sym_id_base,
                layout,
            )); // absx = |act|
            // REVERTED (2026-07-28): a multi-stage blocked-reduce experiment lived here (three
            // iterations, all confirmed on real hardware to make ZERO difference to the actual bug —
            // the K-cache inf this was meant to fix turned out to be caused by cachewr's matmul
            // mechanism instead, see the cachewr plain-copy fix). Unlike the reduce-MAX-at-rows>1
            // theory (which was checked against the OLD proven code and definitively disproven — that
            // code batches rows=nqh fine), this width-blocking theory was never confirmed against any
            // proven reference at all, was pure speculation, and never demonstrated any real effect.
            // Keeping ~150 lines of never-validated-on-hardware blocking logic around after its own
            // premise (fixing K-cache inf) is already fixed elsewhere is unjustified risk with no
            // upside. Reverted to the simple, single unblocked reduce — decode already proves this form
            // works (m=1 always takes this path); prefill (m>1) now takes it too.
            //
            // ⛔⭐⭐⭐ EXCEPT WHERE THE REDUCE'S OWN TILER TIME-TILES IT — the gemma-4-12b fp8 prefill
            // blocker. A time-tiled MAX reduce is silently WRONG, not merely refused: the vendor DDL
            // (`summeanmaxexx2.ddl`) seeds LRF to -inf and writes the final value ONCE per invocation
            // (OVERWRITE semantics), while `rewrite_op_for_time_tile`'s tiled predicate exempts the
            // `[Active, RedStick]` accum from per-trip strides — every trip writes the SAME `[m,64]`
            // bytes, so trip t destroys trip t-1's partial and the surviving amax is the LAST slice's
            // max, not the global max. The bake's `tiled_trips_alias` guard correctly refuses it
            // (12 refusals on ktir_prefill_gemma_4_12b_it_fp8). So ASK THE TILER FIRST — the SAME
            // `reduce_opspec` the emission below builds, never a second LX formula — and when it would
            // time-tile, emit the remedy `reduce.rs`'s own doc names: PARTIALS + COMBINE, "tile the
            // reduction to one stick-width slice and combine the partials", one reduce per
            // k/time-slice into a DISJOINT `[m,64]` slot, then fold them with `maximum` into `amax`.
            let amax_h = rb(&amax, m, stk);
            // The TILER'S OWN ANSWER, not a second LX formula: build the SAME OpSpec the single
            // reduce below would emit and read its `time`. `head_major=false` is the fact
            // `assemble_reduce_seeded` derives from `absx_h`'s `Stk<RowBlockedTag>` kind.
            let single = reduce_opspec(op_func_from_str("max"), m, k, &absx, &amax, false)
                .map_err(|e| SuperDscError(format!("fq_amax_op: {e}")))?;
            let amax_time = single.time();
            if amax_time == 1 || single.iter.split_of("out") > 1 {
                // Non-tiled (every existing model: decode m=1, granite fp8 prefill) — the single
                // unblocked reduce, byte-identical to the pre-partials emission. (A reduction-core
                // split is left to the existing path too: the partials+combine remedy below is
                // derived for the pure time-tiling case, and this arm is unreachable at any geometry
                // the fp8 models actually bake.)
                ops.push(assemble_reduce_seeded(
                    &opn("fq_amax_op"),
                    "max",
                    m,
                    k,
                    &absx_h,
                    &amax_h,
                    sym_id_base,
                    layout,
                ));
            } else {
                // PARTIALS: `time` single-shot reduces, each over a `[m, k/time]` column slice of the
                // RowBlocked `absx`. Slice j starts at column `j·k/time` — a stick boundary by the
                // TimeTile invariant (`k/time` is a whole 64-stick) — whose device element offset is
                // `j·m·(k/time)`: the RowBlocked law `(c/64)·(m·64)` puts stick-group g of EVERY row
                // at `g·m·64`, so a whole-slice shift is exactly `j` blocks of the reduce's own
                // `[m, k/time]` slab (block-aligned, which is what keeps `reduce_opspec_off` on the
                // stick-major rank-2 path — a flat rank-3 read of a RowBlocked buffer scrambles at
                // rows>1 && cols>64). Partial 0 seeds `amax` itself (attn's bmax→run_m precedent: no
                // copy op for the only-ever-first slot); partials 1..time each land in their OWN
                // `[m, 64]` synth, so no two reduces share an output byte — the aliasing the bake
                // guard refuses is unrepresentable here, not merely avoided. Each partial is itself
                // single-shot BY CONSTRUCTION: its residency is exactly the per-trip residency the
                // tiler validated when it chose `time`.
                let k_part = k / amax_time;
                let partials: Vec<String> = (1..amax_time)
                    .map(|j| syn(layout, a_id.synth(R::FqAmaxP(j))))
                    .collect();
                if let Some(l) = layout {
                    for j in 1..amax_time {
                        l.synth(a_id.synth(R::FqAmaxP(j)), &[m, stk]);
                    }
                }
                for j in 0..amax_time {
                    let accum_h = if j == 0 {
                        rb(&amax, m, stk)
                    } else {
                        rb(&partials[(j - 1) as usize], m, stk)
                    };
                    let op_name = if j == 0 {
                        opn("fq_amax_op")
                    } else {
                        opn(&format!("fq_amax_p{j}_op"))
                    };
                    ops.push(assemble_reduce_off(
                        &op_name,
                        "max",
                        m_rows,
                        crate::sdsc_abstract::BlockCols::of_feature_cols(k_part),
                        &absx_h,
                        crate::addr::col_of(m, k, j * k_part, Df::Fp16),
                        &accum_h,
                        crate::addr::DevOff::ZERO,
                        sym_id_base,
                        layout,
                    ));
                }
                // COMBINE: amax = maximum(amax, partial_j) — running, in place on `amax` (the
                // tanhsoftcap `tscth` in-place precedent). Lands in the SAME `amax` name every
                // downstream op (`fq_amaxfl_op` on) already reads, so nothing below changes.
                for (i, p) in partials.iter().enumerate() {
                    let p_h = rb(p, m, stk);
                    ops.push(pw2(
                        &opn(&format!("fq_amax_c{}_op", i + 1)),
                        "maximum",
                        m_rows,
                        scale_cols,
                        In::full(&amax_h),
                        In::full(&p_h),
                        &amax_h,
                        sym_id_base,
                        layout,
                    ));
                }
            }
            // floor: amax = max(amax, 1/448) — a zero-amax (all-zero padded query row) would make
            // invs=recip(0)=inf → 0·inf=NaN downstream; every real (non-padded) row's amax is always >>
            // 1/448, so this leaves them byte-identical. A scratchy padding guard, not part of torch-spyre's
            // decomposition (which never sees a zero-amax row).
            let amaxfl_h = rb(&amaxfl, m, stk);
            ops.push(pw2(
                &opn("fq_amaxfl_op"),
                "maximum",
                m_rows,
                scale_cols,
                In::full(&amax_h),
                In::scalar(&inv448),
                &amaxfl_h,
                sym_id_base,
                layout,
            ));
            // a_scale = amax·(1/448); invs = reciprocal(a_scale) — torch-spyre's `quantize_fp8_with_scale`
            // takes `scale` as input and computes `reciprocal(scale)` itself.
            let ascale_h = rb(&ascale, m, stk);
            ops.push(pw2(
                &opn("fq_ascale_op"),
                "mul",
                m_rows,
                scale_cols,
                In::full(&amaxfl_h),
                In::scalar(&inv448),
                &ascale_h,
                sym_id_base,
                layout,
            ));
            let invs_h = rb(&invs, m, stk);
            ops.push(pw1(
                &opn("fq_invs_op"),
                "reciprocal",
                m_rows,
                scale_cols,
                In::full(&ascale_h),
                &invs_h,
                sym_id_base,
                layout,
            ));
            // scaled = act·invs; clamp to [-448,448] (torch-spyre's `quantize_fp8_with_scale` ALWAYS clamps).
            let sc_h = rb(&sc, m, k);
            ops.push(pw2(
                &opn("fq_sc_op"),
                "mul",
                m_rows,
                k_cols,
                In::full(&rb(&a_name, m, k)),
                In::col(&invs_h),
                &sc_h,
                sym_id_base,
                layout,
            ));
            let chi_h = rb(&chi, m, k);
            ops.push(pw2(
                &opn("fq_chi_op"),
                "minimum",
                m_rows,
                k_cols,
                In::full(&sc_h),
                In::scalar(&pos448),
                &chi_h,
                sym_id_base,
                layout,
            ));
            let cl_h = rb(&cl, m, k_pad);
            // ⭐ THE PAD WINDOW'S ZERO FILL. `cl`'s logical `[m, k]` window is written by the
            // clamp above; the `[k, k_pad)` window is written HERE, unconditionally, every
            // step — a `sub` of the 448 const with itself is a broadcast ZERO (no new const:
            // the fp8 consts are width-independent `[1,64]` scalars). `In::scalar` broadcasts
            // both axes, so the op's reads stay in the const's own footprint while the write
            // covers `[m, 64]` at `col_of(m, k_pad, k, Fp16)` — the RowBlocked stick-group
            // law (2112/64 = 33 stick-groups, whole, so the pad window starts at a stick
            // boundary BY ARITHMETIC; a non-whole k would need the fill re-aimed, which the
            // door's own 64-stick input guard excludes).
            if k_pad != k {
                let zf_h = rb(&cl, m, k_pad);
                ops.push(assemble_pointwise_broadcast_off(
                    &opn("fq_zfpad_op"),
                    "sub",
                    m_rows,
                    // ⭐ ONE STICK WIDE (the pad width, not `k_pad`): the write is aimed at the
                    // pad WINDOW via the output offset, so the op's own width is the window's
                    // 64 elements — a `[m, k_pad]` op would rewrite the logical window too
                    // (reading `sc`'s unpad-ded bytes at the pad position).
                    crate::sdsc_abstract::BlockCols::of_one_stick(
                        crate::sdsc_abstract::Lanes::FP16,
                    ),
                    &[In::scalar(&pos448).ew(), In::scalar(&pos448).ew()],
                    &zf_h,
                    // The RowBlocked stick-group law aims the write at the pad window: stick-
                    // group `k/64` of every row (2112/64 = 33, whole — the door's own 64-stick
                    // input guard keeps `k` a whole fp16 stick, so the window starts at a stick
                    // boundary by arithmetic).
                    crate::addr::col_of(m, k_pad, k, Df::Fp16),
                    sym_id_base,
                    layout,
                ));
            }
            ops.push(pw2(
                &opn("fq_cl_op"),
                "maximum",
                m_rows,
                k_cols,
                In::full(&chi_h),
                In::scalar(&neg448),
                &cl_h,
                sym_id_base,
                layout,
            ));
            // qfp8ch: f16 clamped act → SEN143_FP8. A dtype CONVERT (input 64-stick, output 128-stick), so it
            // goes through `assemble_convert` (two `primaryDsInfo_`), NOT a same-stick pointwise. The output's
            // `Df::Fp8` is intrinsic to the op (`convert_dtypes`). ⭐ AT `k_pad`: the convert's own
            // `assert_df_stick_multiple` (128-elem SEN143_FP8 law) is the guard that refused 2112 — the
            // pad exists so THIS op sees a whole stick count — and its input reads the zero-filled `cl`.
            ops.push(assemble_convert(
                &opn("fq_afp8_op"),
                "qfp8ch",
                m,
                k_pad,
                &rb(&cl, m, k_pad),
                &rb(&afp8, m, k_pad),
                sym_id_base,
                layout,
            ));
        } // end `if !already_quantized` — the shared activation-quantize chain (per distinct activation).
        // matmulfp8: raw = act_fp8 @ W_fp8 → fp16. Build the plain matmul opspec, then declare both operands
        // (A = quantized act, W = kernel) fp8 via `set_df` — that typed marker is what selects the `matmulfp8`
        // opFunc + the 128-stick / 1-byte operand residency. The output `raw` stays fp16 (the DDL `%ptsum_fp`).
        ops.push({
            // fp8 matmul GEOMETRY = fp16 (`::<Fp16>`): the OUTPUT is fp16 and the KERNEL's N-stick is 64,
            // so the `out` (N) work-division splits on the 64-stick — fp8-ness does NOT change the N-split.
            // (128 is ONLY the fp8 ACTIVATION's K-stick, which is the reduction axis — never split.) Then
            // stamp the activation + weight operands `Df::Fp8` (the ½-HBM 1-byte residency); `emit_sdsc`
            // detects the fp8 matmul and emits the 2-D PACKED KERNEL stick `[in:2, out:64]` + the
            // SEN143_FP8 compute marker (the DDL `batchmatmulfp8` operand contract). The OUTPUT stays fp16.
            // `Df::Fp8` twice, deliberately: `operand_df` is what A/W COST in LX and is the only one
            // available early enough to reach the work-division and time-tiling decision (both run
            // inside the opspec builder); the `set_df` loop below is what they ARE on the wire.
            let mut spec = crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands::<Fp16>(
                crate::sdsc_abstract::MatM::of_token_rows(m),
                crate::sdsc_abstract::MatN::of_out_features(n),
                // ⭐ `k_pad`, NOT `k`: the packed fp8 weight is staged with `fp8_k_pad` K rows
                // (the retile + the worker's staging both take it from the same rule), so the
                // contraction's K must match the staged weight or the tile walks past the
                // buffer. The activation operand is `[m, k_pad]` — the quantize chain's own
                // widened tail. No-op at every 128-aligned k.
                crate::sdsc_abstract::MatK::of_in_features(k_pad),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &afp8,
                &w_name,
                &raw,
                0,
                0,
                0,
                Df::Fp8,
            )
            .map_err(SuperDscError)?;
            for arg in &mut spec.args {
                if !matches!(arg.view().role, Role::Output) {
                    arg.set_df(Df::Fp8);
                }
            }
            let folds = SdscFoldSet::new(spec.iter.cores_used());
            emit_sdsc_tiled(&opn("fq_mm"), &spec, &folds, sym_id_base, layout)?
        });
        // dequant: out = raw · a_scale (per-row) · w_scale[n] (per-col). `ascale` is [m,stk] (one scale per
        // query row) read via `col` (out-broadcast over N, row r→row r) — per-row-correct for m>1,
        // byte-identical at decode m=1. `w_scale` is [1,n] per-CHANNEL; for m>1 it broadcasts over the m
        // query rows (`mb`, mirroring rmsnorm's gamma multiply) — a plain `full` would demand m distinct
        // rows from the 1-row w_scale.
        let ascale_h = rb(&ascale, m, stk);
        let raw_h = rb(&raw, m, n);
        let dqa_h = rb(&dqa, m, n);
        ops.push(pw2(
            &opn("fq_dqa_op"),
            "mul",
            m_rows,
            n_cols,
            In::full(&raw_h),
            In::col(&ascale_h),
            &dqa_h,
            sym_id_base,
            layout,
        ));
        let out_h = rb(&out, m, n);
        let ws_h = rb(&ws_name, 1, n);
        let w_scale_in = if m > 1 {
            In::mb(&ws_h)
        } else {
            In::full(&ws_h)
        };
        ops.push(pw2(
            &opn("fq_dqw_op"),
            "mul",
            m_rows,
            n_cols,
            In::full(&dqa_h),
            w_scale_in,
            &out_h,
            sym_id_base,
            layout,
        ));
        Ok(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::place::act_name;
    use crate::placement::{SegRole, TensorPlacement};

    /// A layout that places every tensor the fp8 chain names — the activation, the weight, the
    /// per-channel w_scale, the output, and the three worker-bound E4M3 consts — the same facts
    /// `lower_graph_to_superdsc`'s layout carries, stated here directly so the descriptors can be
    /// asked for in-crate. Ids are MINTED (`ids` map), because `resolve_seg_base` recovers a
    /// tensor's identity by lookup, never by parsing its spelling.
    fn fp8_layout(m: u32, k: u32, n: u32) -> BundleLayout {
        let mut l = BundleLayout::default();
        let mut place = |tid: u32, rows: u32, cols: u32, role: SegRole, l: &mut BundleLayout| {
            l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
            let seg = role.segment();
            let off = l.segment_bytes[seg];
            let size = crate::placement::synth_footprint_bytes(&[rows, cols], Df::Fp16);
            l.placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role,
                    segment: seg,
                    bank: 0,
                    offset: off,
                    size,
                },
            );
            l.segment_bytes[seg] = crate::placement::align128(off + size);
        };
        place(7, m, k, SegRole::Activation, &mut l); // a_tid: the activation
        place(8, n, k, SegRole::Weight, &mut l); // the fp8 weight (fp16-sized reservation is fine)
        place(9, 1, n, SegRole::Activation, &mut l); // w_scale
        place(10, m, n, SegRole::Activation, &mut l); // out_tid
        for tid in [FP8_POS448_TID, FP8_NEG448_TID, FP8_INV448_TID] {
            place(tid, 1, Fp16::ELEMS_PER_STICK, SegRole::Activation, &mut l);
        }
        l
    }

    fn facts() -> Fp8Facts {
        Fp8Facts {
            a_tid: 7,
            a_name: act_name(7),
            w_name: act_name(8),
            ws_name: act_name(9),
            out_tid: 10,
        }
    }

    /// ⭐⭐⭐⭐⭐ THE AMAX CHAIN AT AN LX-OVERFLOWING GEOMETRY EMITS NO TIME-TILED OP.
    ///
    /// This is the gemma-4-12b fp8 prefill blocker's own shape class: `m=1024, k=16384` puts the
    /// two-operand reduce at `2·(1024/32)·16384·2 = 2,097,152 B` of per-core LX — past the
    /// 1,677,721-B budget — so the single reduce WOULD time-tile (`time=2`), which the bake's
    /// `tiled_trips_alias` guard refuses (every trip writes the same `[m,64]` accum bytes — and
    /// with the vendor DDL's overwrite semantics the surviving amax would be the LAST slice's max,
    /// not the global max). The partials+combine path must replace it with single-shot ops only.
    #[test]
    fn the_amax_at_an_lx_overflowing_geometry_emits_no_time_tiled_op() {
        let (m, k, n) = (1024u32, 16384, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the wide-geometry fp8 chain emits");
        // The AMAX family must be single-shot. (The wide POINTWISE ops around it — absx/sc/chi/cl
        // over the same `[m,k]` — legitimately time-tile here; a pointwise output is `Active` over
        // `out` so `rewrite_op_for_time_tile` strides it per trip and the bake accepts it. A REDUCE
        // accum is `RedStick` — exempt from the stride — which is exactly the defect, so it is the
        // reduce family this pins.)
        let tiled: Vec<&str> = ops
            .iter()
            .filter(|o| o.time > 1 && o.op_name.contains("fq_amax"))
            .map(|o| o.op_name.as_str())
            .collect();
        assert!(
            tiled.is_empty(),
            "every op of the amax family must be single-shot (time=1); time-tiled: {tiled:?}"
        );
        // The partials actually fired: one seed reduce into `amax`, `time-1` slot reduces, and the
        // same number of combines. (A silent fall-back to the single reduce would leave this count
        // wrong AND the bake refusing.)
        let seed = ops.iter().filter(|o| o.op_name.ends_with("fq_amax_op")).count();
        let slots = ops
            .iter()
            .filter(|o| o.op_name.contains("fq_amax_p"))
            .count();
        let combines = ops
            .iter()
            .filter(|o| o.op_name.contains("fq_amax_c"))
            .count();
        assert_eq!((seed, slots, combines), (1, 1, 1), "at m=1024/k=16384 the tiler mints time=2 (resident 2,097,152 B > 1,677,721 B; at k/2 it is 1,048,576 B), so the chain is ONE seed reduce + ONE partial + ONE combine");
        // The combine is the LAST amax-family op before the floor — the downstream `fq_amaxfl_op`
        // still reads the same `amax` name, so the chain's tail is untouched.
        let amaxfl = ops
            .iter()
            .position(|o| o.op_name.ends_with("fq_amaxfl_op"))
            .expect("the floor op survives");
        let last_combine = ops
            .iter()
            .rposition(|o| o.op_name.contains("fq_amax_c"))
            .expect("a combine exists");
        assert!(
            last_combine < amaxfl,
            "the combines must fold into `amax` BEFORE the floor op reads it"
        );
    }

    /// ⛔ THE PARTIAL SLOTS ARE DISJOINT — each reduce's accum footprint is its own `[m,64]`
    /// reservation, never a shared byte. With one synth per slot this is a placement-level fact:
    /// distinct names at distinct offsets with non-overlapping [offset, offset+size) ranges, and
    /// the emitted AllocNode start addresses of the two reduces' outputs differ. That is exactly
    /// the property `tiled_trips_alias` refuses to see violated — here it cannot arise, and this
    /// pins it.
    #[test]
    fn the_amax_partial_slots_are_disjoint() {
        let (m, k, n) = (1024u32, 16384, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the wide-geometry fp8 chain emits");
        let s = l.synth.borrow();
        let a_id = PlaceId::Act(7);
        let amax_name = a_id.synth(SynthRole::FqAmax).to_string();
        let p1_name = a_id.synth(SynthRole::FqAmaxP(1)).to_string();
        let amax_off = *s.map.get(&amax_name).expect("amax is placed");
        let p1_off = *s.map.get(&p1_name).expect("partial 1 is placed");
        let amax_sz = s.sizes[&amax_name];
        let p1_sz = s.sizes[&p1_name];
        // Footprint sanity: both are the true `[m, 64]` fp16 extent.
        assert_eq!(amax_sz, m as u64 * 64 * 2, "amax owns [m, 64]");
        assert_eq!(p1_sz, m as u64 * 64 * 2, "the partial slot owns [m, 64]");
        // Disjoint byte ranges — the aliasing the bake refuses is unrepresentable here.
        let overlap = amax_off < p1_off + p1_sz && p1_off < amax_off + amax_sz;
        assert!(
            !overlap,
            "amax [{amax_off},{}) and its partial slot [{p1_off},{}) must not share a byte",
            amax_off + amax_sz,
            p1_off + p1_sz
        );
        // And the EMITTED output nodes agree: the two reduces' accum AllocNodes carry different
        // start addresses (the bake guard reads these same nodes; asserting them here keeps the
        // test honest against a future change that places slots apart but addresses them together).
        let start_of_output = |op: &EmittedOp| -> u64 {
            let dsc = op.dsc();
            for dsc_map in dsc.dscs_.iter() {
                for dsc in dsc_map.values() {
                    for node in dsc.scheduleTree_.iter() {
                        let binding = &op.arg_bindings[node.ldsIdx_ as usize];
                        if !binding.is_input
                            && binding.buffer.contains("fq_amax")
                        {
                            let v = node
                                .startAddressCoreCorelet_
                                .data_
                                .values()
                                .next()
                                .expect("a start address");
                            return v.parse().expect("addresses are numeric");
                        }
                    }
                }
            }
            panic!("{}: no output AllocNode", op.op_name);
        };
        let seed = ops
            .iter()
            .find(|o| o.op_name.ends_with("fq_amax_op"))
            .expect("the seed reduce");
        let slot = ops
            .iter()
            .find(|o| o.op_name.contains("fq_amax_p"))
            .expect("the partial reduce");
        let (seed_start, slot_start) = (start_of_output(seed), start_of_output(slot));
        assert_ne!(
            seed_start, slot_start,
            "the seed (amax) and the partial slot must start at different addresses"
        );
        assert!(
            (seed_start.max(slot_start) - seed_start.min(slot_start)) >= m as u64 * 64 * 2,
            "the two reduces' outputs are a whole [m,64] buffer apart, not interleaved"
        );
    }

    /// ⭐ THE NON-TILED CASE IS BYTE-IDENTICAL: at a geometry the tiler passes (decode m=1, and
    /// every granite fp8 prefill), the amax is ONE op named `fq_amax_op` and NO partial/combine op
    /// exists. Pins that the new path cannot fire where the old one was correct.
    #[test]
    fn the_amax_at_a_fitting_geometry_stays_one_single_shot_reduce() {
        let (m, k, n) = (32u32, 2048, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the fitting-geometry fp8 chain emits");
        assert!(
            ops.iter().all(|o| o.time == 1),
            "a fitting geometry has no time-tiled op at all"
        );
        assert_eq!(
            ops.iter().filter(|o| o.op_name.ends_with("fq_amax_op")).count(),
            1,
            "exactly one amax reduce"
        );
        assert!(
            !ops.iter()
                .any(|o| o.op_name.contains("fq_amax_p") || o.op_name.contains("fq_amax_c")),
            "no partial or combine op may exist at a fitting geometry"
        );
        assert!(
            !l.synth
                .borrow()
                .map
                .contains_key(&PlaceId::Act(7).synth(SynthRole::FqAmaxP(1)).to_string()),
            "no partial synth may be declared at a fitting geometry"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE 26b K-PAD (2112 → 2176): gemma-4-26b's dense GeGLU DOWN gemm is the FIRST fp8
    /// model whose contraction k is not a whole 128-elem SEN143_FP8 stick (2112 = 16.5 sticks —
    /// the panic that stopped the 26b phase-2 bake at `matmul_s20`). The chain must:
    ///   · emit at all (the `qfp8ch` convert's `assert_df_stick_multiple` refused the sub-stick
    ///     width — this test passing IS the wall gone);
    ///   · reserve `cl` and `afp8` at `[m, k_pad]` (the widened tail the convert and the
    ///     contraction share);
    ///   · ZERO-FILL the pad window (one `sub(448,448)` op) — stale pad lanes are last step's
    ///     activation, and a NaN there passes `maximum`/`minimum` unchanged into the product;
    ///   · keep the amax at the LOGICAL width (the pad never reaches the activation scale).
    #[test]
    fn the_26b_k2112_pads_to_a_whole_fp8_stick() {
        let (m, k, n) = (1u32, 2112, 2816);
        let k_pad = crate::work::fp8_k_pad(k);
        assert_eq!(k_pad, 2176, "2112 rounds up to the next whole 128-elem stick");
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the 2112-wide fp8 chain emits (the 26b wall is gone)");
        // The zero-fill exists (the `sub(448,448)` broadcast), and only ONE such op.
        assert_eq!(
            ops.iter().filter(|o| o.op_name.ends_with("fq_zfpad_op")).count(),
            1,
            "exactly one pad-window zero fill"
        );
        // The synth reservations for the widened tail: `cl` at [m, k_pad] fp16, `afp8` at
        // [m, k_pad] fp8 (1 byte/elem — the ½-fp16 residency).
        let s = l.synth.borrow();
        let cl_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(*cl_sz, m as u64 * k_pad as u64 * 2, "cl owns [m, k_pad] fp16");
        let afp8_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqAfp8).to_string())
            .expect("afp8 is placed");
        assert_eq!(
            *afp8_sz,
            m as u64 * k_pad as u64,
            "afp8 owns [m, k_pad] fp8 (1 B/elem)"
        );
        // The amax family is untouched by the pad: `absx` stays at the LOGICAL [m, k].
        let absx_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqAbsX).to_string())
            .expect("absx is placed");
        assert_eq!(*absx_sz, m as u64 * k as u64 * 2, "absx owns [m, k] fp16");
    }

    /// ⭐ THE ALIGNED CASE IS BYTE-IDENTICAL: at a 128-aligned k (every earlier fp8 model), no
    /// zero-fill op exists, no widened reservation exists, and the chain is exactly the ops the
    /// pre-pad emitter produced. Pins that the pad cannot fire where the width was already legal.
    #[test]
    fn an_aligned_k_emits_no_pad_op_and_no_widened_reservation() {
        let (m, k, n) = (32u32, 2048, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the aligned-geometry fp8 chain emits");
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_zfpad_op")),
            "no zero-fill op may exist at an aligned k"
        );
        let s = l.synth.borrow();
        let cl_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(*cl_sz, m as u64 * k as u64 * 2, "cl owns [m, k] fp16 — no widening");
    }

    /// ⭐⭐⭐⭐⭐ THE 26b UP/GATE GEMM: k=2816 (128-ALIGNED) but n=2112 (16.5 fp8 KERNEL sticks).
    /// The packed fp8 weight is `[n, k]` sticked on `out` (= n) — `view_stick_layout`'s
    /// `DeviceTileLayout::<Fp8>` guard (emit/mod.rs) refuses a non-128-multiple OUT the same way
    /// the convert refused a non-128-multiple K. This test REPRODUCES the pod panic locally:
    /// the phase-2 build died at `matmul_s20` (the 26b gate projection) with exactly this
    /// geometry, while the DOWN gemm (k=2112, n=2816) — the geometry the K-pad test covers —
    /// was already fixed by the K-pad commit.
    #[test]
    fn the_26b_n2112_packed_kernel_out_stick() {
        let (m, k, n) = (1u32, 2816, 2112);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the n=2112 packed-kernel fp8 chain emits (the 26b gate gemm wall is gone)");
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_zfpad_op")),
            "an aligned k needs no K-pad"
        );
        let _ = ops;
    }
}
