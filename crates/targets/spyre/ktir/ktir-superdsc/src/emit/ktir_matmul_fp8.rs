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

use super::{EmittedOp, In, assemble_convert, bmm_site, emit_sdsc_tiled, pw1, pw2, rb, rbo};
use crate::ir::bridge::tiled_op_sdsc_op::reduce::assemble_reduce_seeded;
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
        // The quant chain's typed extents: the activation's `m` is the matmul's token rows, `k`/`n`
        // its in/out feature columns, and the per-row scale tensors are ONE STICK wide.
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
            for r in [R::FqAbsX, R::FqSc, R::FqChi, R::FqCl] {
                l.synth(a_id.synth(r), &[m, k]);
            }
            l.synth_df(a_id.synth(R::FqAfp8), &[m, k], Df::Fp8); // 1-byte / 128-stick residency (½ fp16)
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
            let amax_h = rb(&amax, m, stk);
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
            let cl_h = rb(&cl, m, k);
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
            // `Df::Fp8` is intrinsic to the op (`convert_dtypes`).
            ops.push(assemble_convert(
                &opn("fq_afp8_op"),
                "qfp8ch",
                m,
                k,
                &rb(&cl, m, k),
                &rb(&afp8, m, k),
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
                crate::sdsc_abstract::MatK::of_in_features(k),
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
