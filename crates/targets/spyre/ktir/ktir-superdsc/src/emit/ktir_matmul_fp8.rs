// SPDX-License-Identifier: Apache-2.0
//! THE PROVEN fp8 W8A8 MATMUL DESCRIPTORS, driven from the KTIR.
//!
//! ⭐⭐⭐ THE BODY IS `ibm/main`'s `lower_matmul_node` arity-3 branch, UNCHANGED — the whole W8A8 chain:
//! per-token activation quantize (`abs` → `amax` → scale → clamp → `qfp8ch`), `matmulfp8`
//! (fp8×fp8→fp16), then dequant by `a_scale`(per row) · `w_scale`(per channel). Only its DOOR changed:
//! it takes [`Fp8Facts`] recovered from the KTIR program instead of a `&SubtileNode`.
//!
//! ⛔ AND THAT CHAIN IS WHY fp8 NEEDS THIS. The builder's `KtirFunc::matmul_fp8` computed **fp16 activation × fp8
//! weight** — its own doc says outright that it does not quantize the activation — while this device's
//! fp8 is fp8×fp8. Lowering the KTIR op-for-op therefore cannot produce `matmulfp8` at all, and it
//! also addressed the weight as fp16: a `[2048, 2048]` fp8 weight is placed 4 MB and was being read
//! 8 MB. fp8 W8A8 runs on hardware today through this code, so the card keeps using it — reached
//! through the KTIR, from facts the KTIR itself states.

use super::{
    EmittedOp, In, PointwiseFacts, assemble_convert, assemble_convert_col,
    assemble_pointwise_broadcast_off, bmm_site, emit_sdsc_tiled, op_func_from_str, pw1, pw2, rb,
    rb_df, rbo,
};
use crate::ir::bridge::tiled_op_sdsc_op::reduce::{
    assemble_reduce_off, assemble_reduce_seeded, reduce_opspec, win_reduce_accum, win_reduce_data,
};
use crate::place::{PlaceId, SynthRole};
use crate::placement::{BundleLayout, syn};
use crate::reserved_tids::{FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID};
use crate::sdsc_abstract::{BlockCols, Lanes, RowCount};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{DataFormat, Df, Fp16, Role, SdscFoldSet};

/// What the KTIR states about one fp8 W8A8 contraction, recovered from its shapes and parameters.
///
/// ⛔ EVERY FIELD IS READ OUT OF THE KTIR. The fp8-ness itself is too: the builder's
/// `KtirFunc::matmul_fp8` built
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

/// ⭐⭐⭐ THE SHARED W8A8 ACTIVATION QUANTIZE — the first half of every fp8 matmul door
/// (`matmul_fp8_descriptors`'s own body, extracted): `abs → amax → floor → scale → clamp →
/// zero-pad → qfp8ch`, over the `[m, k]` activation, ONE scale per token row. The EXPERT door
/// calls this DIRECTLY (its per-pair matmuls and flat dequant legs replace the dense body), so
/// the two doors share one chain and no op is emitted only to be spliced out — the dense body's
/// own `matmulfp8` binds its W as a whole-tensor `[k_pad, n]` view, which against the expert
/// door's chunk scratch the synth footprint guard refuses at bake.
///
/// ⭐⭐ THE TAIL PADS PER PAIR. `pairs` is the K axis's pair count (dense = 1; the expert door
/// passes its top-k): the activation is `pairs` windows of `inn = k/pairs` lanes, and the tail
/// tensors (`cl`, `afp8`) are laid out per-pair PADDED — pair `p`'s window at `p·inn_pad` of a
/// `[m, pairs·inn_pad]` buffer. The reason is the fp8 STICK LAW: the matmul's A operand start
/// must land on a whole 128-elem fp8 stick (the HBM walk is stick-granular — the senulator
/// loads whole 128-B sticks by stick index), and at `inn ≡ 64 (mod 128)` (the 26b DOWN door's
/// 704) the LOGICAL pair stride is a HALF stick, so the padded layout is the only one whose
/// every window start is stick-aligned. The per-pair form fires ONLY at `pairs > 1 ∧
/// inn_pad ≠ inn`: at an aligned `inn` the padded layout IS the logical one and the whole-tensor
/// convert stands (byte-identical to the pre-pair emission), and at `pairs == 1` this is the
/// dense law verbatim (`cl_reach = k_pad = fp8_k_pad(k)`, one tail window, one convert).
pub(crate) fn fp8_quantize_descriptors(
    facts: &Fp8Facts,
    m: u32,
    k: u32,
    pairs: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    // fp8 W8A8 PREFILL (m>1). torch-spyre's `quantize_fp8_with_scale`
    // (decompositions.py:966) takes `scale` as an input computed by the CALLER via native `torch.abs` +
    // `torch.ops.aten.amax` (confirmed via grep — no L2-norm substitute anywhere in torch-spyre); the
    // `#33` belief that reduce-MAX "seeds 0 for rows>1" was disproved by the stable-softmax `block_max`
    // (attn.rs), itself a multi-row MAX reduce. ONE formula for any row count (m==1 decode or m>1
    // prefill), matching torch-spyre exactly — mirrors rmsnorm's unification (no scratchy-only approx,
    // no diagnostic-sweep scaffolding left in the live path).
    if pairs == 0 || !k.is_multiple_of(pairs) {
        return Err(SuperDscError(format!(
            "fp8 quantize t{}: the K axis is {k} lanes over {pairs} pairs — the per-pair tail law \
             needs a whole lane count per pair",
            facts.a_tid
        )));
    }
    let inn = k / pairs;
    let inn_pad = crate::work::fp8_k_pad(inn);
    // ⭐ THE HALF-STICK TAIL, m-GENERIC: at an ALIGNED `inn` (`inn_pad == inn` — the expert
    // door's gate/up at 2816, every dense caller at `pairs == 1`) the padded layout IS the
    // logical one and the whole-tensor convert stands at ANY row count (the dense prefill's
    // own card-proven form). At the half-stick tail (`inn_pad != inn`, the 26b DOWN's 704)
    // the per-pair arms below now split on `m`: m=1 keeps the FLAT stick-axis slice (the
    // decode form, byte-identical), m>1 takes [`assemble_convert_col`] at [`crate::addr::col_of`]
    // group corners — the per-slot rank-2 stick-major arms of design §4.5, landed.
    let stk = Fp16::ELEMS_PER_STICK; // 64
    // ⭐ THE 26b K-PAD — the contraction width rounded to a whole 128-elem SEN143_FP8 stick
    // (`work::fp8_k_pad`, the K-axis twin of `DeviceWidth`'s N law). gemma-4-26b's dense GeGLU
    // DOWN gemm contracts k=2112 = 16.5 fp8 sticks — the FIRST fp8 model with a non-legal K
    // (granite/g8b/g12b are all 128-multiples) — and `qfp8ch`'s convert guard refuses a
    // sub-stick tile outright. The fp16 READS of this chain stay at the LOGICAL `[m, k]` (the
    // activation's own reservation is k-wide; reading past it is out-of-footprint); only the
    // chain's TAIL widens — per pair at `pairs > 1`, one window at `pairs == 1`.
    // (`inn`/`inn_pad` are computed ABOVE, stated once for the whole tail law — the arms
    // below and the reservations all read them.)
    let k_pad = pairs * inn_pad; // the tail tensors' width (the per-pair padded layout)
    // `cl`'s own reach: the LAST pair's convert reads `[p·inn, p·inn + inn_pad)` — its pad
    // lanes overhang the logical width by exactly `inn_pad − inn` (one fp16 stick at the
    // door's whole-stick `inn`), so `cl` reserves `k + (inn_pad − inn)` and the fill below
    // covers the overhang. At `pairs == 1` this is `k_pad` itself (the dense law).
    let cl_reach = (pairs - 1) * inn + inn_pad;
    let ops = {
        let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(m);
        let k_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(k);
        let scale_cols =
            crate::sdsc_abstract::BlockCols::of_one_stick(crate::sdsc_abstract::Lanes::FP16);
        // ⭐ THE NAMES COME FROM THE KTIR'S OWN PARAMETERS — `args[i]` → `wiring::act_name(tid)`, the
        // same spelling `ds_name` produced for the SubtileIR path.
        let a_name = facts.a_name.clone();
        let out = crate::place::act_name(facts.out_tid);
        use SynthRole as R;
        // SHARED quant tensors are a pure function of the ACTIVATION (abs→amax→scale→clamp→fp8) — derive
        // them from the ACTIVATION's id so a second matmul on the same activation resolves the IDENTICAL
        // fp8 tensor + a_scale (the layout `synth` is idempotent per id → one reservation). PER-MATMUL
        // tensors (the matmul output `raw` and its dequant `dqa`) derive from the OUTPUT's id — those
        // belong to the dense body, not this half.
        // ⭐ THE DOOR: the ids come from the KTIR-recovered facts, not from a `&SubtileNode`. They are
        // the same tensor ids either way — `Fp8Facts::a_tid`/`out_tid` are recorded off the program's
        // own views — so the placements these mint are the ones the proven path minted.
        let a_id = PlaceId::Act(facts.a_tid);
        // The operand SPELLING is a rendering of the id the layout is keyed by, so the op's name and
        // the allocator's key cannot drift — they have one preimage.
        let qn = |r: R| syn(layout, a_id.synth(r));
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
        let afp8 = qn(R::FqAfp8); // quantized fp8 activation — sized/typed fp8 via synth_df + the convert
        if let Some(l) = layout {
            // Footprints are `m` query rows (decode m=1 ⇒ byte-identical to the old `[1,·]`; prefill m>1
            // reserves the real multi-row extent so a per-row write can't clobber a neighbor row).
            for r in [R::FqAbsX, R::FqSc, R::FqChi] {
                l.synth(a_id.synth(r), &[m, k]);
            }
            // The chain's tail widens to the PAD: `cl` feeds the convert (whose 128-elem stick law
            // is the reason the pad exists) and `afp8` is the fp8 contraction operand — both are
            // reserved at the PER-PAIR PADDED width, the one layout the convert, the per-pair
            // matmul A windows, and the packed weight's K rows all share (`cl` only to its last
            // convert's REACH — the one-stick overhang the fill below covers).
            l.synth(a_id.synth(R::FqCl), &[m, cl_reach]);
            l.synth_df(a_id.synth(R::FqAfp8), &[m, k_pad], Df::Fp8); // 1-byte / 128-stick residency (½ fp16)
            for r in [R::FqAmax, R::FqAmaxFl, R::FqAscale, R::FqInvS] {
                l.synth(a_id.synth(r), &[m, stk]);
            }
        }
        let pos448 = rbo(&crate::place::act_name(FP8_POS448_TID));
        let neg448 = rbo(&crate::place::act_name(FP8_NEG448_TID));
        let inv448 = rbo(&crate::place::act_name(FP8_INV448_TID));
        let mut ops: Vec<EmittedOp> = Vec::new();
        // SHARE the activation quantize across matmuls that read the SAME activation. `insert` returns false
        // when this activation (`a_name`) was already quantized earlier in the bundle (e.g. K/V after Q, or
        // Up after Gate — the expert door's gate/up pair is the same law) — in that case skip re-emitting
        // the whole abs→amax→scale→clamp→qfp8ch chain and reuse its `afp8`/`ascale` (both named by
        // `a_name`); only the per-matmul legs stay per-matmul. (The chain is a launch/op cost paid ONCE
        // per distinct activation instead of once per matmul.)
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
            // ⭐⭐⭐⭐⭐ ONE HARDWARE OP FOR THE FIRST FOUR STEPS — see [`OpFunc::QuantScalePerTokenFp8`].
            //
            // `abs → max-reduce → floor → ·(1/448)` IS `quantscalepertokenfp8`, whose DDL body is the
            // FABSMAX tree (`quant_scale_per_token.ddl:178-232`) with its partials in
            // `ddl.internal_tensor`, i.e. on-chip, and whose tail is exactly
            // `FMIN(FMAX(absmax·mulConst, clipMin), clipMax)` (`:259/:283/:287`) — the three constants
            // [`OpFunc::op_consts`] supplies are picked to make that identical to `fq_amaxfl` +
            // `fq_ascale`, floor included. Registered for dd2 (`multi_sentient_dd2_sysconfig.json:87`).
            //
            // ⛔ IT IS SHAPED EXACTLY LIKE THE REDUCE IT REPLACES, which is why it goes through the
            // reduce assembler and not a new one: input `[m, k]`, output `[m, stick]` — the DDL's
            // `%global_layout_output` drops the reduce dim its `%global_layout_input` carries, so the
            // output IS the per-token scale. `invs` below still reciprocates it (the DDL keeps its own
            // reciprocal INTERNAL, so it is not an operand we can read), and `sc/chi/cl/afp8` still
            // build the `[m,k]` fp8 tensor. 4 ops → 1, ×4 activations a layer = 12 of the 51 glue ops
            // that are 67% of a bs=1 decode layer.
            //
            // ⛔ IT IS AN UNSEEDED REDUCE, so it does NOT take the accumulator as a second input the way
            // `max`/`sum`/`mean` do — `OpFunc::ddl_input_arity` carries that fact and the emitter makes
            // a violation a build error. Passing the seed is not a diagnosable error in the compiler: it
            // is a silent template non-match (MEASURED — the same descriptor with two inputs is
            // `[DDC] DDL found but not suitable for op quantscalepertokenfp8`, with one input it is
            // accepted).
            //
            // ⛔ `absx`/`amax`/`amaxfl` are no longer materialised. Their `synth` reservations are left
            // in place deliberately for this first measurement so the bundle's ADDRESS ASSIGNMENT does
            // not move — `BundleLayout::synth` is a bump allocator, so dropping entries would shift
            // every later tensor and confound a coherence comparison with the baseline.
            let ascale_h = rb(
                &ascale,
                RowCount::of_token_rows(m),
                BlockCols::of_one_stick(Lanes::FP16),
            );
            let a_h = rb(
                &a_name,
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
            );
            // ⛔ ASK THE TILER FIRST — the SAME `reduce_opspec` the emission below builds, never a
            // second LX formula. A time-tiled reduce is UNEMITTABLE for this op class: the DDL's
            // accumulator has OVERWRITE semantics while `rewrite_op_for_time_tile`'s tiled predicate
            // exempts the `[Active, RedStick]` output from per-trip strides — every trip writes the
            // SAME `[m,64]` scale bytes, so the surviving scale would be the LAST slice's, not the
            // global one. The bake's `tiled_trips_alias` guard correctly refuses it (12b fp8 prefill
            // rung 71: at the true DDL arity the residency is `1·71·15360·2 = 2,181,120 B` vs the
            // 1,677,721-B budget — a PRIME rung, so `core_split` finds no divisor and the reduce
            // keeps the whole row count per core). So when the tiler would time-tile, emit the
            // remedy `reduce.rs`'s own doc names — "tile the reduction ... and combine the
            // partials": one fused reduce per k/time-slice into a DISJOINT `[m,64]` slot, then
            // fold them with `maximum`.
            let single = reduce_opspec(
                op_func_from_str("quantscalepertokenfp8"),
                m,
                k,
                &a_name,
                &ascale,
                false,
            )
            .map_err(|e| SuperDscError(format!("fq_qspt_op: {e}")))?;
            let qspt_time = single.time();
            if qspt_time == 1 || single.iter.split_of("out") > 1 {
                // Non-tiled (every existing geometry: decode m=1, granite fp8 prefill, the 12b's
                // split rungs) — the single fused reduce, byte-identical to the pre-partials
                // emission. (A reduction-core split is left to this path too: the partials+combine
                // remedy below is derived for the pure time-tiling case.)
                ops.push(assemble_reduce_seeded(
                    &opn("fq_qspt_op"),
                    "quantscalepertokenfp8",
                    m,
                    k,
                    &a_h,
                    &ascale_h,
                    sym_id_base,
                    layout,
                ));
            } else {
                // PARTIALS: `time` single-shot fused reduces, each over a `[m, k/time]` column slice
                // of the RowBlocked activation. Slice j starts at column `j·k/time` — a stick
                // boundary by the TimeTile invariant (`k/time` is a whole 64-stick) — whose device
                // element offset `col_of` renders through the RowBlocked law: stick-group g of every
                // row sits at `g·m·64`, so a whole-slice shift is exactly `j` blocks of the reduce's
                // own `[m, k/time]` slab (block-aligned, which is what keeps `reduce_opspec_off` on
                // the stick-major rank-2 path — a flat rank-3 read of a RowBlocked buffer scrambles
                // at rows>1 && cols>64). Partial 0 seeds `ascale` itself (attn's bmax→run_m
                // precedent: no copy op for the only-ever-first slot); partials 1..time each land in
                // their OWN `[m,64]` synth, so no two reduces share an output byte — the aliasing
                // the bake guard refuses is unrepresentable here, not merely avoided. Each partial
                // is itself single-shot BY CONSTRUCTION: its residency is exactly the per-trip
                // residency the tiler validated when it chose `time`.
                //
                // ⛔ THE COMBINE MAY FOLD SCALES, NOT AMAXES, AND STILL BE EXACT: the fused op's
                // tail `FMIN(FMAX(absmax·mulConst, clipMin), clipMax)` is MONOTONE NON-DECREASING
                // in the slice's amax (a positive scale, then a clamp), and a monotone
                // non-decreasing g commutes with max — `max_j g(amax_j) = g(max_j amax_j)` — so
                // `maximum` over the per-slice SCALES is bit-exact the single-shot scale, floor and
                // clip included. All-zero padded rows included: every slice's g(0) is the clipMin
                // floor, and the max of identical floors is that floor.
                let k_part = k / qspt_time;
                let partials: Vec<String> = (1..qspt_time)
                    .map(|j| syn(layout, a_id.synth(R::FqAmaxP(j))))
                    .collect();
                if let Some(l) = layout {
                    for j in 1..qspt_time {
                        l.synth(a_id.synth(R::FqAmaxP(j)), &[m, stk]);
                    }
                }
                for j in 0..qspt_time {
                    let accum_h = if j == 0 {
                        rb(
                            &ascale,
                            RowCount::of_token_rows(m),
                            BlockCols::of_one_stick(Lanes::FP16),
                        )
                    } else {
                        rb(
                            &partials[(j - 1) as usize],
                            RowCount::of_token_rows(m),
                            BlockCols::of_one_stick(Lanes::FP16),
                        )
                    };
                    let op_name = if j == 0 {
                        opn("fq_qspt_op")
                    } else {
                        opn(&format!("fq_qspt_p{j}_op"))
                    };
                    ops.push(assemble_reduce_off(
                        &op_name,
                        "quantscalepertokenfp8",
                        m_rows,
                        crate::sdsc_abstract::BlockCols::of_feature_cols(k_part),
                        // The data window is the j-th `[m, k/time]` slab — `col_of`'s whole-block
                        // shift (k/time is a whole 64-stick, the TimeTile invariant), minted
                        // against the activation's own `[m, k]` handle.
                        win_reduce_data(
                            layout,
                            &a_h,
                            crate::addr::col_of(m, k, j * k_part, Df::Fp16),
                            m_rows,
                            crate::sdsc_abstract::BlockCols::of_feature_cols(k_part),
                            &format!("{op_name}: activation"),
                        )
                        .map_err(SuperDscError)?,
                        win_reduce_accum(
                            layout,
                            &accum_h,
                            crate::addr::DevOff::ZERO,
                            m_rows,
                            &format!("{op_name}: scale"),
                        )
                        .map_err(SuperDscError)?,
                        sym_id_base,
                        layout,
                    ));
                }
                // COMBINE: ascale = maximum(ascale, partial_j) — running, in place on `ascale` (the
                // old amax chain's own law). Lands in the SAME `ascale` name every downstream op
                // (`fq_invs_op` on) already reads, so nothing below changes.
                for (i, p) in partials.iter().enumerate() {
                    let p_h = rb(
                        p,
                        RowCount::of_token_rows(m),
                        BlockCols::of_one_stick(Lanes::FP16),
                    );
                    ops.push(pw2(
                        &opn(&format!("fq_qspt_c{}_op", i + 1)),
                        "maximum",
                        m_rows,
                        scale_cols,
                        In::full(&ascale_h),
                        In::full(&p_h),
                        &ascale_h,
                        sym_id_base,
                        layout,
                    ));
                }
            }
            // The three tensors the two fused ops subsume, plus the `1/448` and ±448 they used to take as
            // scalar tensor operands — the constants now reach their own `mulConst`/`clipMin`/`clipMax`
            // through `ddl.get_external_constant` BY NAME out of the descriptor's `constantInfo_`.
            // (`chi` joins them: materialised by nothing, its reservation held for allocator stability.)
            let _ = (&absx, &amax, &amaxfl, &chi, &inv448, &pos448, &neg448);
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
            let invs_h = rb(
                &invs,
                RowCount::of_token_rows(m),
                BlockCols::of_one_stick(Lanes::FP16),
            );
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
            let sc_h = rb(
                &sc,
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
            );
            ops.push(pw2(
                &opn("fq_sc_op"),
                "mul",
                m_rows,
                k_cols,
                In::full(&rb(
                    &a_name,
                    RowCount::of_token_rows(m),
                    BlockCols::of_feature_cols(k),
                )),
                In::col(&invs_h),
                &sc_h,
                sym_id_base,
                layout,
            ));
            // ⭐⭐⭐⭐⭐ THE CLAMP IS THE DEVICE'S OWN `clip` OP — ONE input, computing
            // `FMIN(FMAX(x, clipMin), clipMax)` (`unary_pipeline.ddl:370-377` PE stage, `:472-479` SFP
            // stage) — the same min/max pair the two ops it replaces encoded, so the swap is
            // value-preserving by construction. The ±448 E4M3 bounds arrive as the external constants
            // `clipMin`/`clipMax` out of the descriptor's `constantInfo_` ([`OpFunc::op_consts`]),
            // NOT as the `[1, stick]` H2D tensor operands `fq_chi`/`fq_cl` read through
            // `In::scalar(&pos448)`/`In::scalar(&neg448)` — the constants dxp folds before the
            // constant tables, per the same contract as `quantscalepertokenfp8`'s three. 2 ops → 1,
            // ×4 activations ×40 layers = 40 more ops off a granite-2b fp8 decode step. `FP8_POS448`
            // keeps exactly ONE reader — the pad window's zero fill below (`sub(448,448)`, a
            // broadcast zero); `FP8_NEG448`'s bind loses its last reader.
            //
            // ⛔ `chi` IS NO LONGER MATERIALISED. Its `synth` reservation stays for the same reason
            // `absx`/`amax`/`amaxfl`'s do above: `BundleLayout::synth` is a bump allocator, so
            // dropping entries would shift every later tensor and confound a coherence comparison
            // with the baseline.
            let cl_h = rb(
                &cl,
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(cl_reach),
            );
            ops.push(pw1(
                &opn("fq_clip_op"),
                "clip",
                m_rows,
                k_cols,
                In::full(&sc_h),
                &cl_h,
                sym_id_base,
                layout,
            ));
            // ⭐ THE PAD WINDOW'S ZERO FILL. `cl`'s logical `[m, k]` window is written by the
            // clamp above; the `[k, cl_reach)` window — the LAST pair's convert overhang,
            // `inn_pad − inn` lanes — is written HERE, unconditionally, every step — a `sub`
            // of the 448 const with itself is a broadcast ZERO (no new const: the fp8 consts
            // are width-independent `[1,64]` scalars). `In::scalar` broadcasts both axes, so
            // the op's reads stay in the const's own footprint while the write covers `[m, 64]`
            // at `col_of(m, cl_reach, k, Fp16)` — the RowBlocked stick-group law. The door's
            // own 64-stick input guard keeps `inn` — hence `k = pairs·inn` — a whole fp16
            // stick, so the window starts at a stick boundary BY ARITHMETIC, and the overhang
            // is one stick exactly (`fp8_k_pad` adds 0 or 64 at a whole-64 `inn`), which is
            // why ONE one-stick op covers it. The NON-last pairs' pad lanes never read this
            // fill — their converts' `[p·inn, p·inn + inn_pad)` windows borrow the NEXT pair's
            // clamped lanes (finite, and inert × the packed weight's host-zero-filled pad K
            // rows); only the LAST pair's window reaches past `k` into the fill.
            if cl_reach != k {
                let zf_h = rb(
                    &cl,
                    RowCount::of_token_rows(m),
                    BlockCols::of_feature_cols(cl_reach),
                );
                ops.push(assemble_pointwise_broadcast_off(
                    PointwiseFacts {
                        op_name: &opn("fq_zfpad_op"),
                        op_func: "sub",
                        rows: m_rows,
                        // ⭐ ONE STICK WIDE (the pad width, not `cl_reach`): the write is aimed at the
                        // pad WINDOW via the output offset, so the op's own width is the window's
                        // 64 elements — a `[m, cl_reach]` op would rewrite the logical window too
                        // (reading `sc`'s unpad-ded bytes at the pad position).
                        cols: crate::sdsc_abstract::BlockCols::of_one_stick(
                            crate::sdsc_abstract::Lanes::FP16,
                        ),
                        inputs: &[In::scalar(&pos448).ew(), In::scalar(&pos448).ew()],
                        o: &zf_h,
                        // The RowBlocked stick-group law aims the write at the pad window: stick-
                        // group `k/64` of every row (whole — see the fill comment above).
                        out_offset: crate::addr::col_of(m, cl_reach, k, Df::Fp16).into(),
                    },
                    sym_id_base,
                    layout,
                ));
            }
            // qfp8ch: f16 clamped act → SEN143_FP8. A dtype CONVERT (input 64-stick, output 128-stick), so it
            // goes through `assemble_convert` (two `primaryDsInfo_`), NOT a same-stick pointwise. The output's
            // `Df::Fp8` is intrinsic to the op (`convert_dtypes`). ⭐ AT THE PADDED WIDTH: the convert's own
            // `assert_df_stick_multiple` (128-elem SEN143_FP8 law) is the guard that refused 2112 — the
            // pad exists so THIS op sees a whole stick count — and its input reads the zero-filled `cl`.
            if pairs == 1 || inn_pad == inn {
                // ONE whole-tensor convert — the padded layout IS the logical one: an aligned
                // `inn` (every existing model's geometry, and the expert door's gate/up at
                // inn=2816) or the dense `pairs == 1` law (`cl_reach == k_pad`). Byte-identical
                // to the pre-pair emission: same op, same name, same `[m, k_pad]` windows.
                ops.push(assemble_convert(
                    &opn("fq_afp8_op"),
                    "qfp8ch",
                    m,
                    k_pad,
                    &rb(
                        &cl,
                        RowCount::of_token_rows(m),
                        BlockCols::of_feature_cols(k_pad),
                    ),
                    &rb_df(
                        &afp8,
                        RowCount::of_token_rows(m),
                        BlockCols::of_feature_cols(k_pad),
                        Df::Fp8,
                    ),
                    sym_id_base,
                    layout,
                ));
            } else {
                // PER SLOT — the half-stick `inn` case (the 26b DOWN door's 704): slot `s`'s
                // convert reads `cl` at column `s·inn` (an fp16 stick boundary — `inn` is a
                // whole 64-stick) and writes `afp8` at column `s·inn_pad` (an fp8 stick
                // boundary — the whole reason the padded layout exists). ⭐ THE ARMS SPLIT ON
                // `m` (design §4.5): at m=1 the offsets are FLAT stick-axis slices — the
                // decode form, byte-identical, because `col_of` degenerates to the flat law
                // at one row; at m>1 the SAME corners go through [`assemble_convert_col`] as
                // [`crate::addr::col_of`] DevOffs, which carry the `(c/lanes)·(m·lanes)`
                // group-corner term of the stick-major layout — exactly where the expert
                // door's per-(slot, chunk) matmul A-windows read `afp8`
                // (`col_of(rows, k·inn_pad, s·inn_pad)` — one law at both ends of the seam).
                // The pad lanes stay inert by the pair law: a non-last slot's overhang
                // borrows the NEXT slot's clamped lanes (finite, × the packed weight's
                // host-zero-filled pad K rows), and the LAST slot's overhang reads the
                // `[k, cl_reach)` zero fill. Slot 0 keeps the shipped `fq_afp8_op` name so
                // an already-quantized activation's splice-free reuse (and every pinned
                // test) is unchanged.
                for s in 0..pairs {
                    let name = if s == 0 {
                        opn("fq_afp8_op")
                    } else {
                        opn(&format!("fq_afp8_p{s}_op"))
                    };
                    // ⭐ ONE DOOR FOR EVERY `m`. The m=1 arm used to route to a raw-`u32` flat-offset
                    // twin (`assemble_convert_off`, now deleted); at one row a `col_of` corner IS the
                    // flat stick-axis slice exactly — the `(c/lanes)·(m·lanes)` group-corner term is
                    // `m·c` and `m == 1` — so the decode arm emits the SAME offsets through the
                    // DevOff twin and its mint. THE HANDLES ARE TRUED TO THE FULL BUFFERS (`cl_reach`,
                    // `k_pad`) and the fp8 side carries its OWN df (`rb_df`): the mint's placement
                    // door prices bytes at the handle's word length, so the fp16 annotation the
                    // piece-wise handles used to carry would overstate every fp8 window 2× and refuse
                    // the last slot — whose window ends exactly at `cl_reach = (pairs−1)·inn +
                    // inn_pad` / `k_pad = pairs·inn_pad` BY CONSTRUCTION (the zero fill the last
                    // slot's overhang reads is the `[k, cl_reach)` fill the clamp stage wrote).
                    ops.push(assemble_convert_col(
                        &name,
                        "qfp8ch",
                        m,
                        inn_pad,
                        &rb(
                            &cl,
                            RowCount::of_token_rows(m),
                            BlockCols::of_feature_cols(cl_reach),
                        ),
                        crate::addr::col_of(m, cl_reach, s * inn, Df::Fp16),
                        &rb_df(
                            &afp8,
                            RowCount::of_token_rows(m),
                            BlockCols::of_feature_cols(k_pad),
                            Df::Fp8,
                        ),
                        crate::addr::col_of(m, k_pad, s * inn_pad, Df::Fp8),
                        sym_id_base,
                        layout,
                    ));
                }
            }
        } // end `if !already_quantized` — the shared activation-quantize chain (per distinct activation).
        ops
    };
    Ok(ops)
}

/// ⭐⭐⭐ THE PROVEN fp8 W8A8 MATMUL DESCRIPTORS, dense form — the quantize chain (via
/// [`fp8_quantize_descriptors`] at `pairs == 1`, the dense law verbatim) plus the dense
/// body: `matmulfp8` binding the WHOLE packed weight as one `[k_pad, n]` view, then the
/// two dequant legs. The signature is the pre-split one — every caller and test stands.
pub(crate) fn matmul_fp8_descriptors(
    facts: &Fp8Facts,
    m: u32,
    k: u32,
    n: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let mut ops = fp8_quantize_descriptors(facts, m, k, 1, sym_id_base, layout, quantized)?;
    // ⭐ THE 26b K-PAD (dense form) — the contraction width the packed weight was staged at.
    let k_pad = crate::work::fp8_k_pad(k);
    let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(m);
    let n_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(n);
    let out = crate::place::act_name(facts.out_tid);
    use SynthRole as R;
    let a_id = PlaceId::Act(facts.a_tid);
    let out_id = PlaceId::Act(facts.out_tid);
    let w_name = facts.w_name.clone(); // fp8 weight `t{id}` — declared fp8 on the matmul operand (set_df), not the name
    let ws_name = facts.ws_name.clone(); // w_scale, read [1,n] (byte-identical to the loaded [n,1])
    // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed).
    let opn = |s: &str| format!("{out}_{s}");
    // `afp8`/`ascale` resolve the SAME synth names the chain reserved (`syn` is idempotent per
    // id) — the already-quantized reuse path reads them back by name, exactly as before the
    // split; PER-MATMUL tensors (`raw`, `dqa`) derive from the OUTPUT's id.
    let afp8 = syn(layout, a_id.synth(R::FqAfp8));
    let ascale = syn(layout, a_id.synth(R::FqAscale));
    let (raw, dqa) = (
        syn(layout, out_id.synth(R::FqRaw)),
        syn(layout, out_id.synth(R::FqDqA)),
    );
    if let Some(l) = layout {
        for r in [R::FqRaw, R::FqDqA] {
            l.synth(out_id.synth(r), &[m, n]);
        }
    }
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
    let ascale_h = rb(
        &ascale,
        RowCount::of_token_rows(m),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let raw_h = rb(&raw, m_rows, n_cols);
    let dqa_h = rb(&dqa, m_rows, n_cols);
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
    let out_h = rb(&out, m_rows, n_cols);
    let ws_h = rb(&ws_name, RowCount::of_broadcast_row(), n_cols);
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
        let place = |tid: u32, rows: u32, cols: u32, role: SegRole, l: &mut BundleLayout| {
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
                    cover: crate::placement::BindCover::Full,
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

    /// ⭐⭐⭐⭐⭐ THE QUANT-CHAIN REDUCE AT AN LX-OVERFLOWING GEOMETRY STAYS SINGLE-SHOT.
    ///
    /// This is the gemma-4-12b fp8 prefill blocker's own shape class: `m=1024, k=16384`. The OLD
    /// two-operand amax reduce sat at `2·(1024/32)·16384·2 = 2,097,152 B` of per-core LX — past
    /// the 1,677,721-B budget — so it time-tiled (`time=2`), which the bake's `tiled_trips_alias`
    /// guard refuses (every trip writes the same `[m,64]` accum bytes; with the vendor DDL's
    /// overwrite semantics the surviving amax would be the LAST slice's max, not the global max).
    /// The fused `quantscalepertokenfp8` op is UNSEEDED — one `[m,k]` input at
    /// `(1024/32)·16384·2 = 1,048,576 B` per core, HALF the old residency — so it fits the budget
    /// single-shot and the blocker's geometry no longer tiles the reduce at all. (The wide
    /// POINTWISE ops over the same `[m,k]` legitimately time-tile: their outputs are `Active` over
    /// `out`, so `rewrite_op_for_time_tile` strides them per trip and the bake accepts them. A
    /// REDUCE accum is `RedStick` — exempt from the stride — which is the defect class, so the
    /// reduce is what this pins.)
    #[test]
    fn the_quant_reduce_at_an_lx_overflowing_geometry_stays_single_shot() {
        let (m, k, n) = (1024u32, 16384, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the wide-geometry fp8 chain emits");
        // The chain's ONE reduce — the fused quantscalepertokenfp8 — must be single-shot.
        let tiled: Vec<&str> = ops
            .iter()
            .filter(|o| o.time > 1 && o.op_name.ends_with("fq_qspt_op"))
            .map(|o| o.op_name.as_str())
            .collect();
        assert!(
            tiled.is_empty(),
            "the fused quant reduce must be single-shot (time=1); time-tiled: {tiled:?}"
        );
        // The explicit amax family is GONE — fused into the one op — and with it the partial
        // slots (nothing mints `FqAmaxP` while the fused op fits the budget).
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_qspt_op"))
                .count(),
            1,
            "the amax chain is ONE fused quantscalepertokenfp8 op"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.contains("fq_amax")),
            "the explicit amax family is fused away; a stray amax op is a partial re-introduction"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE FUSED QUANT REDUCE AT ITS OWN OVERFLOWING GEOMETRY SPLITS INTO PARTIALS.
    ///
    /// The 12b fp8 prefill's rung-71 wall: `m=71, k=15360` — a PRIME rung, so `core_split` finds
    /// no divisor ≤ 32 and the reduce keeps the whole row count per core. Even at the true DDL
    /// arity of ONE operand the residency is `1·71·15360·2 = 2,181,120 B` vs the 1,677,721-B
    /// budget, so the tiler legitimately answers `time=2` — and a time-tiled reduce is
    /// unemittable for this op class (the DDL's accumulator has overwrite semantics; every trip
    /// would write the same `[m,64]` scale bytes — the bake's `tiled_trips_alias` refusal). The
    /// emission must therefore never hand the tiler's `time>1` answer to the assembler: it
    /// splits into per-k-slice PARTIALS + a `maximum` COMBINE, every one of them single-shot.
    #[test]
    fn the_quant_reduce_at_a_fused_overflowing_geometry_emits_partials() {
        let (m, k, n) = (71u32, 15360, 3840);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(&facts(), m, k, 1, &mut 0, Some(&l), &mut quantized)
            .expect("the fused-overflowing fp8 quant chain emits");
        assert!(
            ops.iter()
                .all(|o| !o.op_name.contains("fq_qspt") || o.time == 1),
            "every fused-quant op (base, partials, combines) is single-shot; tiled: {:?}",
            ops.iter()
                .filter(|o| o.op_name.contains("fq_qspt") && o.time > 1)
                .map(|o| o.op_name.as_str())
                .collect::<Vec<_>>()
        );
        // The partial + combine family exists, folded into the same `ascale` the tail reads.
        assert!(
            ops.iter().any(|o| o.op_name.ends_with("fq_qspt_p1_op")),
            "the first partial is emitted"
        );
        assert!(
            ops.iter().any(|o| o.op_name.ends_with("fq_qspt_c1_op")),
            "the first combine is emitted"
        );
        assert!(
            l.synth
                .borrow()
                .map
                .contains_key(&PlaceId::Act(7).synth(SynthRole::FqAmaxP(1)).to_string()),
            "the first partial synth is declared"
        );
    }

    /// ⭐ THE FITTING CASE IS ONE FUSED OP: at a geometry the tiler passes (decode m=1, and
    /// every granite fp8 prefill), the quant chain's amax is the ONE `fq_qspt_op` and NO
    /// explicit amax-family op exists. Pins that no partial path can fire where the fused op
    /// was correct.
    #[test]
    fn the_quant_chain_at_a_fitting_geometry_is_one_fused_op() {
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
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_qspt_op"))
                .count(),
            1,
            "exactly one fused quant reduce"
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
        assert_eq!(
            k_pad, 2176,
            "2112 rounds up to the next whole 128-elem stick"
        );
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the 2112-wide fp8 chain emits (the 26b wall is gone)");
        // The zero-fill exists (the `sub(448,448)` broadcast), and only ONE such op.
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
                .count(),
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
        assert_eq!(
            *cl_sz,
            m as u64 * k_pad as u64 * 2,
            "cl owns [m, k_pad] fp16"
        );
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
        assert_eq!(
            *cl_sz,
            m as u64 * k as u64 * 2,
            "cl owns [m, k] fp16 — no widening"
        );
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

    /// An operand's own HBM start (core 0's allocate-node address for the binding whose buffer
    /// matches `needle`), in BYTES — `resolve_seg_base` folds each arg's `offset_elems` into the
    /// segment base, so the DELTAS between ops reading the same buffer at different offsets are
    /// the offsets themselves. The per-pair tail's laws read those deltas.
    fn operand_start_bytes(op: &EmittedOp, needle: &str, want_input: bool) -> u64 {
        let dsc = op.dsc();
        for dsc_map in dsc.dscs_.iter() {
            for dsc in dsc_map.values() {
                for node in dsc.scheduleTree_.iter() {
                    let binding = &op.arg_bindings[node.ldsIdx_ as usize];
                    if binding.is_input == want_input && binding.buffer.contains(needle) {
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
        panic!("{}: no {} AllocNode", op.op_name, needle);
    }

    /// ⭐⭐⭐⭐⭐ THE 26b DOWN TAIL PADS PER PAIR (704 → 768 × 8 pairs): the expert door's DOWN
    /// projection contracts inn=704 = 5.5 fp8 sticks, so the LOGICAL pair stride is a HALF fp8
    /// stick and the quantized activation is laid out per-pair PADDED — pair `p`'s window at
    /// `p·inn_pad` of a `[m, pairs·inn_pad]` buffer. This pins the whole tail law at the door's
    /// own geometry (m=1, pairs=8, k=5632):
    ///   · EIGHT `qfp8ch` converts — one per pair, pair 0 keeping the shipped `fq_afp8_op`
    ///     name and pairs 1..7 carrying `fq_afp8_p{p}_op`;
    ///   · the OUTPUT operand starts advance by the PADDED stride (768 B/pair) and every one
    ///     lands on a whole 128-B stick — the mid-stick start this layout exists to exclude;
    ///   · the INPUT (fp16 `cl`) starts advance by the LOGICAL stride (1408 B/pair);
    ///   · ONE zero-fill, aimed exactly one fp16 stick past the logical width (the last
    ///     pair's overhang `[k, cl_reach)`);
    ///   · the reservations: `cl` at `[m, cl_reach]` (5696·2 B), `afp8` at `[m, k_pad]`
    ///     (6144 B fp8), `absx` still at the LOGICAL `[m, k]`.
    #[test]
    fn the_26b_down_tail_pads_per_pair() {
        let (m, pairs, inn, n) = (1u32, 8u32, 704u32, 128u32);
        let k = pairs * inn; // 5632 — itself 128-aligned, so only the PER-PAIR law pads here
        let inn_pad = crate::work::fp8_k_pad(inn);
        assert_eq!(
            inn_pad, 768,
            "704 rounds up to the next whole 128-elem stick"
        );
        let (k_pad, cl_reach) = (pairs * inn_pad, (pairs - 1) * inn + inn_pad);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(&facts(), m, k, pairs, &mut 0, Some(&l), &mut quantized)
            .expect("the per-pair tail emits at the 26b DOWN geometry");
        // EIGHT converts, named per pair.
        let first = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op"))
            .count();
        let rest = ops
            .iter()
            .filter(|o| o.op_name.contains("fq_afp8_p"))
            .count();
        assert_eq!(
            (first, rest),
            (1, 7),
            "pair 0 keeps the shipped name; pairs 1..7 carry fq_afp8_p{{p}}_op"
        );
        // The OUTPUT windows advance by the PADDED stride and stay whole-stick — the
        // fp8 A operand starts the per-pair layout exists to keep legal.
        let afp8_starts: Vec<u64> = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
            .map(|o| operand_start_bytes(o, "fq_afp8", false))
            .collect();
        assert_eq!(afp8_starts.len(), 8);
        for s in &afp8_starts {
            assert_eq!(
                s % 128,
                0,
                "every afp8 window starts on a whole 128-B stick"
            );
        }
        assert_eq!(
            afp8_starts[7] - afp8_starts[0],
            7 * inn_pad as u64,
            "the fp8 windows advance by the PADDED stride ({} B/pair), not the logical half-stick one",
            inn_pad
        );
        // The INPUT (fp16 cl) windows advance by the LOGICAL stride.
        let cl_starts: Vec<u64> = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
            .map(|o| operand_start_bytes(o, "fq_cl", true))
            .collect();
        assert_eq!(
            cl_starts[7] - cl_starts[0],
            7 * inn as u64 * 2,
            "the fp16 cl windows advance by the LOGICAL stride ({} B/pair)",
            inn * 2
        );
        // ONE zero-fill, aimed exactly one fp16 stick past the logical width.
        let zf = ops
            .iter()
            .find(|o| o.op_name.ends_with("fq_zfpad_op"))
            .expect("the per-pair tail zero-fills the last pair's overhang");
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
                .count(),
            1,
            "exactly one pad-window zero fill"
        );
        assert_eq!(
            operand_start_bytes(zf, "fq_cl", false) - cl_starts[0],
            k as u64 * 2,
            "the fill writes [k, cl_reach) — one fp16 stick past the logical width"
        );
        // The reservations: cl to its REACH, afp8 at the PADDED width, absx at the LOGICAL one.
        let s = l.synth.borrow();
        let a7 = PlaceId::Act(7);
        let cl_sz = s
            .sizes
            .get(&a7.synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * cl_reach as u64 * 2,
            "cl owns [m, cl_reach] = [1, {cl_reach}] fp16"
        );
        let afp8_sz = s
            .sizes
            .get(&a7.synth(SynthRole::FqAfp8).to_string())
            .expect("afp8 is placed");
        assert_eq!(
            *afp8_sz,
            m as u64 * k_pad as u64,
            "afp8 owns [m, k_pad] = [1, {k_pad}] fp8 (1 B/elem)"
        );
        let absx_sz = s
            .sizes
            .get(&a7.synth(SynthRole::FqAbsX).to_string())
            .expect("absx is placed");
        assert_eq!(*absx_sz, m as u64 * k as u64 * 2, "absx owns [m, k] fp16");
    }

    /// ⭐ THE ALIGNED-`inn` CONTROL: at the expert door's gate/up geometry (inn=2816, itself a
    /// whole fp8 stick), the padded layout IS the logical one — ONE whole-tensor convert, no
    /// per-pair naming, no zero-fill, no widened `cl`. Byte-identical to the pre-pair emission;
    /// pins that the per-pair law cannot fire where the logical stride was already legal.
    #[test]
    fn an_aligned_inn_keeps_the_whole_tensor_convert() {
        let (m, pairs, inn, n) = (1u32, 8u32, 2816u32, 128u32);
        let k = pairs * inn;
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(&facts(), m, k, pairs, &mut 0, Some(&l), &mut quantized)
            .expect("the aligned-inn pair chain emits");
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_afp8_op"))
                .count(),
            1,
            "one whole-tensor convert"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.contains("fq_afp8_p")),
            "no per-pair convert at an aligned inn"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_zfpad_op")),
            "no zero-fill at an aligned inn (cl_reach == k)"
        );
        let s = l.synth.borrow();
        let cl_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * k as u64 * 2,
            "cl owns [m, k] fp16 — no widening"
        );
    }

    /// ⛔ THE PAIR LAW'S OWN REFUSAL: a K axis that does not divide into whole per-pair lane
    /// counts is a stated design error (typed `Err`, never a sub-stick bake) — the one refusal
    /// left in this fn. ⭐ THE HALF-STICK TAIL AT m>1 NOW EMITS (design §4.5 landed): the
    /// per-slot arms split on `m` — m=1 keeps the FLAT stick-axis slices (the decode form,
    /// byte-identical), m>1 takes [`assemble_convert_col`] at [`crate::addr::col_of`] group
    /// corners, whose `(c/lanes)·(m·lanes)` term carries the row count. This pins the m>1
    /// corner law at (m=2, pairs=2, inn=704): output windows advance by the PADDED
    /// group-corner stride `m·inn_pad` B (the `m` is the law — the flat form would say
    /// `inn_pad`), input windows by the LOGICAL group-corner stride `m·inn` fp16 elems, the
    /// zero-fill still aims one fp16 group past the logical width at the m=2 corner, and the
    /// reservations widen to the m-generic `[m, cl_reach]`/`[m, k_pad]`.
    #[test]
    fn the_per_pair_tail_refuses_a_non_whole_lane_count_and_pads_per_slot_at_prefill() {
        let l = fp8_layout(1, 9, 128);
        let mut quantized = std::collections::HashSet::new();
        let e = fp8_quantize_descriptors(&facts(), 1, 9, 2, &mut 0, Some(&l), &mut quantized);
        assert!(
            e.is_err(),
            "k=9 over pairs=2 has no whole lane count per pair — a typed refusal, not a bake"
        );
        // The 26b DOWN's own tail at prefill: inn=704 pads to 768, and the per-slot arms take
        // the GROUP-CORNER offsets — m=2 EMITS now, two rank-2 stick-major converts.
        let (m, pairs, inn) = (2u32, 2u32, 704u32);
        let k = pairs * inn; // 1408
        let inn_pad = crate::work::fp8_k_pad(inn);
        assert_eq!(
            inn_pad, 768,
            "704 rounds up to the next whole 128-elem stick"
        );
        let (k_pad, cl_reach) = (pairs * inn_pad, (pairs - 1) * inn + inn_pad);
        let l2 = fp8_layout(m, k, 128);
        let mut quantized2 = std::collections::HashSet::new();
        let ops2 =
            fp8_quantize_descriptors(&facts(), m, k, pairs, &mut 0, Some(&l2), &mut quantized2)
                .expect("the half-stick tail emits at m>1 — the per-slot §4.5 arms");
        assert_eq!(
            ops2.iter()
                .filter(|o| o.op_name.ends_with("fq_afp8_op"))
                .count(),
            1,
            "slot 0 keeps the shipped fq_afp8_op name"
        );
        assert_eq!(
            ops2.iter()
                .filter(|o| o.op_name.contains("fq_afp8_p"))
                .count(),
            (pairs - 1) as usize,
            "slots 1.. carry fq_afp8_p{{s}}_op"
        );
        let arms: Vec<&EmittedOp> = ops2
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
            .collect();
        assert_eq!(arms.len(), pairs as usize, "one arm per slot");
        let afp8_starts: Vec<u64> = arms
            .iter()
            .map(|o| operand_start_bytes(o, "fq_afp8", false))
            .collect();
        let cl_starts: Vec<u64> = arms
            .iter()
            .map(|o| operand_start_bytes(o, "fq_cl", true))
            .collect();
        for s in &afp8_starts {
            assert_eq!(
                s % 128,
                0,
                "every afp8 window starts on a whole 128-B stick at m>1 too"
            );
        }
        assert_eq!(
            afp8_starts[1] - afp8_starts[0],
            m as u64 * inn_pad as u64,
            "the fp8 windows advance by the PADDED GROUP-CORNER stride m·inn_pad = {} B — \
             the m term is the §4.5 law (the flat m=1 form would say {inn_pad})",
            m * inn_pad
        );
        assert_eq!(
            cl_starts[1] - cl_starts[0],
            m as u64 * inn as u64 * 2,
            "the fp16 cl windows advance by the LOGICAL GROUP-CORNER stride m·inn fp16 \
             elems = {} B (the flat m=1 form would say {})",
            m * inn * 2,
            inn * 2
        );
        // ONE zero-fill, aimed at the m=2 group corner one fp16 group past the logical width:
        // `col_of(2, cl_reach, k)` = (k/64)·(2·64) elems = m·k fp16 elems.
        let zf = ops2
            .iter()
            .find(|o| o.op_name.ends_with("fq_zfpad_op"))
            .expect("the per-slot tail zero-fills the last slot's overhang");
        assert_eq!(
            ops2.iter()
                .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
                .count(),
            1,
            "exactly one pad-window zero fill at m>1"
        );
        assert_eq!(
            operand_start_bytes(zf, "fq_cl", false) - cl_starts[0],
            m as u64 * k as u64 * 2,
            "the fill writes [k, cl_reach) at the m=2 group corner — m·k fp16 elems past base"
        );
        // The reservations, m-generic: cl [m, cl_reach], afp8 [m, k_pad], absx [m, k].
        let s2 = l2.synth.borrow();
        let a7 = PlaceId::Act(7);
        let cl_sz = s2
            .sizes
            .get(&a7.synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * cl_reach as u64 * 2,
            "cl owns [m, cl_reach] = [{m}, {cl_reach}] fp16"
        );
        let afp8_sz = s2
            .sizes
            .get(&a7.synth(SynthRole::FqAfp8).to_string())
            .expect("afp8 is placed");
        assert_eq!(
            *afp8_sz,
            m as u64 * k_pad as u64,
            "afp8 owns [m, k_pad] = [{m}, {k_pad}] fp8 (1 B/elem)"
        );
        let absx_sz = s2
            .sizes
            .get(&a7.synth(SynthRole::FqAbsX).to_string())
            .expect("absx is placed");
        assert_eq!(*absx_sz, m as u64 * k as u64 * 2, "absx owns [m, k] fp16");
        // And the positive control: the same m=2 at an ALIGNED inn (1024, a whole fp8 stick)
        // EMITS — one whole-tensor convert, the m-generic form the prefill door consumes.
        let l3 = fp8_layout(2, 2048, 128);
        let mut quantized3 = std::collections::HashSet::new();
        let ops =
            fp8_quantize_descriptors(&facts(), 2, 2048, 2, &mut 0, Some(&l3), &mut quantized3)
                .expect("the aligned-inn pair chain emits at m=2");
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_afp8_op"))
                .count(),
            1,
            "one whole-tensor convert at m=2 — no per-pair arms at an aligned inn"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.contains("fq_afp8_p")),
            "no per-pair convert at an aligned inn, prefill included"
        );
    }
}
