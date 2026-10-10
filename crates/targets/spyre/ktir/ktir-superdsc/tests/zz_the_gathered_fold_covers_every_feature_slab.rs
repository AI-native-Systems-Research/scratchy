// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE GATHERED FOLD AT hd=128 — **BOTH FEATURE SLABS, MEASURED OFF THE EMITTED DESCRIPTORS**,
//! because the two defects that made granite-3.1-8b wrong from its first generated token were each
//! stated as a COMMENT and each invisible at hd=64.
//!
//! ## What was wrong, and why no test could see it
//! The collapsed (gathered) fold's two legs were written when `PageScratch::of_pass` refused two slabs,
//! so both carried a one-slab assumption in prose:
//!
//! * the VALUE leg declared `n = MatN::of_head_slab(SLAB_FEATS)` — ONE STICK — with **no slab loop**,
//!   under "ONE slab, because the gather's own precondition is `hd <= POOL_STICK`". At hd=64 one stick IS
//!   the whole head dim, so the missing loop was a no-op. At hd=128 the fold wrote only feature slab 0 of
//!   `run_o`: the upper 64 features of every head's attention output got NO prefix contribution at all.
//! * the SCORE leg declared `k = MatK::of_head_dim(hd)` under a `y`-batch, with "the gather's own
//!   precondition is `hd <= POOL_STICK`, so there is exactly one slab and no partial sums to accumulate".
//!   At hd=128 that is a TWO-STICK contraction under a `y`-batch — the shape `ScoreArm::choose` records
//!   as measured-twice incoherent inside dxp ("degenerate output at a FASTER ITL, which is the tell").
//!
//! Both bake clean. Both produce fluent wrong text. And the door above them refused hd=128, so every
//! green test in this suite was taken at the one head dim where neither defect exists — which is
//! precisely the shape of "a green test pins a divergence as correct".
//!
//! ## What this file measures
//! The DESCRIPTORS, not the call site. For each head dim it emits the whole attention body with the
//! gather on and asserts, from the emitted JSON:
//! 1. the gathered score leg contracts **one stick** (`N_["in_"] == POOL_STICK`) at every head dim, and
//!    there are `nslab` of them per (kv head, request);
//! 2. the gathered value leg's output is **one stick wide** (`N_["out_"] == POOL_STICK`) and there are
//!    `nslab` of them per (kv head, request), whose descriptors are DISTINCT — a second slab that
//!    emitted the same bytes would be the missing loop with a name;
//! 3. at hd=64 the op names are byte-for-byte what shipped (no `s` segment), so granite-3.1-2b's
//!    emission does not move.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, POOL_STICK, PagedKvPool, attn_bundle_rows};

const NQH: u32 = 32;
const NKVH: u32 = 8;
/// granite-3.1-2b: `hidden 2048 / nqh 32` — ONE slab, the only geometry the gather had card evidence at.
const HD_2B: u32 = 64;
/// granite-3.1-8b: `hidden 4096 / nqh 32` — TWO slabs, the geometry this file exists for.
const HD_8B: u32 = 128;
const CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;

/// One emitted op, reduced to what these assertions are about: its dsc name and its declared extents.
struct Op {
    name: String,
    /// `N_` — the declared iteration extents (`mb_`, `in_`, `out_`, `y_`).
    n: std::collections::BTreeMap<String, i64>,
    /// The whole dsc body, so two ops claiming to be different slabs can be shown to differ.
    body: String,
}

/// The whole attention body at one head dim and width, gather ON.
///
/// `handoff` is the caller-side statement of the head-major o_proj restructure
/// (`BundleAttnParams::headmajor_handoff` at the spyre door): every test that does not name it
/// passes `false`, which is the emission every bundle without the restructure still gets.
fn emit_at<const HD: u32>(mq: u32, handoff: bool) -> Vec<Op> {
    try_emit_at::<HD>(mq, handoff)
        .unwrap_or_else(|e| {
            panic!(
                "hd={HD} mq={mq} handoff={handoff}: assemble_attn refused: {}",
                e.0
            )
        })
        .iter()
        .map(|e| {
            let v = serde_json::to_value(&e.op).expect("serializes");
            let (name, body) = v["dscs_"][0]
                .as_object()
                .and_then(|m| m.iter().next())
                .map(|(k, b)| (k.clone(), b.clone()))
                .expect("one named dsc per emitted op");
            let n = body["N_"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, x)| x.as_i64().map(|i| (k.clone(), i)))
                        .collect()
                })
                .unwrap_or_default();
            Op {
                name,
                n,
                body: serde_json::to_string(&body).expect("serializes"),
            }
        })
        .collect()
}

/// [`emit_at`]'s refusal-preserving twin — the desync tests need the `Err`, not a panic.
fn try_emit_at<const HD: u32>(
    mq: u32,
    handoff: bool,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, ktir_superdsc::superdsc_error::SuperDscError> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows =
        attn_bundle_rows(geom, mq, true).unwrap_or_else(|| panic!("mq={mq} is not a baked rung"));
    let mut sym = 0i64;
    assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        // ⭐ THE GATHER IS ON, WHICH IS THE WHOLE POINT. With `None` here the fold takes the ungathered
        // arms, whose slab loops were never missing — so a test that forgot this argument would pass
        // against the code that was broken.
        Some("t_kv_idx"),
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        handoff,
        &mut sym,
        None,
    )
}

/// The PREFIX fold's gathered ops of one leg, keyed by their name. `p{b}` is a fold pass; `nsc`/`nov` are
/// the new-token block's, which are not folded and never gathered.
fn prefix_leg<'a>(ops: &'a [Op], leg: &str) -> Vec<&'a Op> {
    ops.iter()
        .filter(|o| {
            let bare = o.name.split('/').next_back().unwrap_or(&o.name);
            bare.contains("_p") && bare.contains(leg) && !bare.contains(&format!("n{leg}"))
        })
        .collect()
}

/// ⭐⭐⭐⭐⭐ THE VALUE LEG COVERS **EVERY** FEATURE SLAB, AND EACH OP'S OUTPUT IS ONE STICK WIDE.
///
/// ⛔ THE TWO HALVES ARE ONE FACT AND THIS IS WHY THE FIX IS A LOOP RATHER THAN A WIDER `out`. A
/// `y`-batched matmul reaches head `h` by striding `y` and derives that stride as `mb*out`; the
/// accumulators are head-major `[rows, hd]` whose real pitch is `mq*stick`, so `out = hd` derives `mq*hd`
/// and agrees only at one stick. So `out` must stay one stick AND the slabs must be swept by separate
/// ops — asserted together, because satisfying either alone is a silent wrong answer.
#[test]
fn the_gathered_value_leg_emits_one_op_per_feature_slab() {
    for (hd, nslab) in [(HD_2B, 1u32), (HD_8B, 2)] {
        for mq in [2u32, 8] {
            let ops = if hd == HD_2B {
                emit_at::<HD_2B>(mq, false)
            } else {
                emit_at::<HD_8B>(mq, false)
            };
            let leg = prefix_leg(&ops, "ov");
            // `nb = active_cap / 64` fold passes, each with `nkvh * mq * nslab` value ops.
            let nb = CAP / POOL_STICK;
            assert_eq!(
                leg.len() as u32,
                nb * NKVH * mq * nslab,
                "hd={hd} mq={mq}: the gathered value leg must emit nb*nkvh*mq*nslab ops. Got {}: {:?}",
                leg.len(),
                leg.iter().map(|o| o.name.as_str()).collect::<Vec<_>>()
            );
            for o in &leg {
                assert_eq!(
                    o.n.get("out_").copied(),
                    Some(POOL_STICK as i64),
                    "hd={hd} mq={mq} {}: a `y`-batched value op's `out` must be ONE STICK — at `out = \
                     hd` the derived y-stride is `mq*hd` where the head-major buffer's is `mq*stick`",
                    o.name
                );
                assert_eq!(
                    o.n.get("in_").copied(),
                    Some(POOL_STICK as i64),
                    "hd={hd} mq={mq} {}: the value leg contracts the 64-slot KV window",
                    o.name
                );
            }
            // ⭐ AND THE SLAB IS IN THE NAME AT hd=128 AND ABSENT AT hd=64 — the 2b's emission does not
            // move, which is the safety property for an address-moving change.
            let with_slab = leg.iter().filter(|o| o.name.contains("s1_r")).count();
            assert_eq!(
                with_slab as u32,
                if nslab == 1 { 0 } else { nb * NKVH * mq },
                "hd={hd} mq={mq}: slab 1's ops are named `s1` — at one slab there must be none, so the \
                 shipped 2b descriptor names are byte-identical"
            );
            // ⛔ AND TWO SLABS' OPS MUST NOT BE THE SAME DESCRIPTOR. A loop that emitted the same op
            // twice would satisfy every count above and still leave the upper half of `run_o` unwritten.
            if nslab > 1 {
                let s0 = leg
                    .iter()
                    .find(|o| o.name.contains("ov_g0s0_r0"))
                    .expect("slab 0 of (kv head 0, request 0)");
                let s1 = leg
                    .iter()
                    .find(|o| o.name.contains("ov_g0s1_r0"))
                    .expect("slab 1 of (kv head 0, request 0)");
                assert_ne!(
                    s0.body, s1.body,
                    "hd={hd} mq={mq}: the two slabs of (kv head 0, request 0) emit IDENTICAL \
                     descriptors, so slab 1 writes slab 0's bytes and the upper half of every head's \
                     output is never produced"
                );
            }
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE SCORE LEG CONTRACTS **ONE STICK** AT EVERY HEAD DIM — dxp's real precondition for a
/// `y`-batched op, and the one the head-dim gate was standing in for.
///
/// ⛔ `N_["in_"]` IS THE MEASUREMENT, NOT THE `MatK` AT THE CALL SITE. The call site passed
/// `MatK::of_head_dim(hd)`, which reads as "the contraction" and IS one stick at hd=64 — so reading the
/// source could not distinguish the correct form from the incoherent one. The descriptor can.
#[test]
fn the_gathered_score_leg_contracts_exactly_one_stick_per_op() {
    for (hd, nslab) in [(HD_2B, 1u32), (HD_8B, 2)] {
        for mq in [2u32, 8] {
            let ops = if hd == HD_2B {
                emit_at::<HD_2B>(mq, false)
            } else {
                emit_at::<HD_8B>(mq, false)
            };
            let leg = prefix_leg(&ops, "sc");
            let nb = CAP / POOL_STICK;
            assert_eq!(
                leg.len() as u32,
                nb * NKVH * mq * nslab,
                "hd={hd} mq={mq}: the gathered score leg must emit nb*nkvh*mq*nslab ops. Got {}: {:?}",
                leg.len(),
                leg.iter().map(|o| o.name.as_str()).collect::<Vec<_>>()
            );
            for o in &leg {
                assert_eq!(
                    o.n.get("in_").copied(),
                    Some(POOL_STICK as i64),
                    "hd={hd} mq={mq} {}: a `y`-batched score op MUST contract one stick. A two-stick \
                     contraction under a `y`-batch passes every stride check and is incoherent inside \
                     dxp — measured twice, and it is what hd=128 was emitting",
                    o.name
                );
                assert_eq!(
                    o.n.get("out_").copied(),
                    Some(POOL_STICK as i64),
                    "hd={hd} mq={mq} {}: the score leg writes one 64-slot window of columns",
                    o.name
                );
            }
            if nslab > 1 {
                let s0 = leg
                    .iter()
                    .find(|o| o.name.contains("sc_g0s0_r0"))
                    .expect("slab 0 of (kv head 0, request 0)");
                let s1 = leg
                    .iter()
                    .find(|o| o.name.contains("sc_g0s1_r0"))
                    .expect("slab 1 of (kv head 0, request 0)");
                assert_ne!(
                    s0.body, s1.body,
                    "hd={hd} mq={mq}: the two slab partials of one score row emit IDENTICAL \
                     descriptors, so the contraction covers half the head dim twice"
                );
            }
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE FOLD RUN'S OP COUNT **IS THE LAUNCH GROUP SIZE**, because a gathered fold run is never
/// chunked (`GroupKind::run_may_be_chunked` — every op of one pass must be in one group, and a pass IS the
/// whole run). So the slab split doubles a number that is already the largest group in the bundle, and
/// that number is the BAKE COST.
///
/// ⛔ MEASURED, and it is the practical price of hd=128: the granite-3.1-8b fp8 build spent **2470 s**,
/// of which ~40 minutes was ONE `dxp_standalone` on the fold group — against 334 s for a whole cold
/// granite-3.1-2b fp8 build whose fold group is half the size. It does bake, which is the thing that had
/// to be established; it is also superlinear, so a further doubling (hd=256, or a wider rung) needs this
/// number looked at before it is attempted rather than after.
///
/// ⭐ ASSERTED AS A COMPOSITION, NOT A CEILING. `GroupSize::CEILING` (512, the largest CHUNKED group
/// observed to bake) is already exceeded by the 2b's own exempt fold run (588), so a ceiling assertion
/// here would either fail on shipped code or pin the wrong bound. What is checkable is that the count is
/// exactly the two legs plus the per-block fixed cost plus the copies — so an accidental extra factor
/// (a slab loop nested inside a slab loop, say) shows up here as a number and not as a slow build.
#[test]
fn the_gathered_fold_runs_group_size_is_the_two_legs_plus_its_fixed_cost() {
    let nb = CAP / POOL_STICK;
    for (hd, nslab) in [(HD_2B, 1u32), (HD_8B, 2)] {
        for mq in [1u32, 2, 8] {
            let ops = if hd == HD_2B {
                emit_at::<HD_2B>(mq, false)
            } else {
                emit_at::<HD_8B>(mq, false)
            };
            let legs = (prefix_leg(&ops, "sc").len() + prefix_leg(&ops, "ov").len()) as u32;
            let copies = ops
                .iter()
                .filter(|o| o.name.contains("gkt") || o.name.contains("gv"))
                .count() as u32;
            assert_eq!(
                legs,
                2 * nb * NKVH * mq * nslab,
                "hd={hd} mq={mq}: the two gathered legs are nb*nkvh*mq*nslab ops each"
            );
            // TWO PLANES, one copy op per (request, entry cut) each — `ops_per_row` is 1 here.
            assert_eq!(
                copies,
                2 * mq,
                "hd={hd} mq={mq}: one Kᵗ and one V copy per request"
            );
            println!(
                "hd={hd} mq={mq} nslab={nslab}: legs={legs} copies={copies} total_attn_ops={}",
                ops.len()
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE HEAD-MAJOR HANDOFF'S FINALIZE — the single-row body's slab-major form, MEASURED OFF
/// THE EMITTED DESCRIPTORS like everything else in this file, because the restructure it belongs to
/// has TWO halves keyed on ONE bundle fact and each half alone is fluent wrong output:
///
/// * THIS form — at `mq == 1` over a multi-slab head dim, ONE op per feature slab (`nslab`, not
///   `nqh·nslab`), each ALL heads at once (`mb == nqh`) and ONE STICK wide (`out == POOL_STICK`),
///   writing `out` where the accumulator already holds it (slab-major). On granite-3.1-8b that is
///   the 64 → 2 op collapse per layer — 2,480 of ~15,400 descriptor executions per decode step
///   (the baked mq=1 body loses exactly 62 descriptors: 187 → 125 at the attention program).
/// * the o matmul's B-operand swap to the permuted o_proj copy, keyed on the SAME fact at the
///   matmul door (`lower_ktir_to_superdsc`'s `HeadmajorOproj`), whose staging walk the spyre
///   crate's own tests pin.
///
/// ⛔ AND EVERY DESYNC IS REFUSED, NOT CORRECTED. The handoff at `mq > 1` — where NO weight
/// permutation can fix the relayout, because `run_o` carries the head on its ROW axis and a matmul
/// contracts along its activation's COLUMN axis only — and at `nslab == 1` — where there is
/// nothing to restructure and the shipped one-op form already applies — each refuse here, so the
/// bundle never bakes half a restructure.
#[test]
fn the_headmajor_finalizes_one_op_per_slab_and_refuses_every_desync() {
    // ⭐ THE FORM, at the only geometry it is minted for: hd=128 (nslab=2), mq=1.
    let ops = emit_at::<HD_8B>(1, true);
    // (A nested fn, not a closure: the returned `&str` borrows the op's own name, and a
    // closure's elided input lifetime cannot say that.)
    fn bare(o: &Op) -> &str {
        o.name.rsplit('/').next().unwrap_or(&o.name)
    }
    let slab_ops: Vec<&Op> = ops
        .iter()
        .filter(|o| bare(o).starts_with("attn_o_s"))
        .collect();
    assert_eq!(
        slab_ops.len(),
        2,
        "hd=128 mq=1 handoff: the finalize must be ONE op per feature slab (2), got {}: {:?}",
        slab_ops.len(),
        slab_ops.iter().map(|o| o.name.as_str()).collect::<Vec<_>>()
    );
    // ⛔ AND THE PER-HEAD LOOP IS GONE — this IS the collapse being pinned. A handoff that still
    // paid `nqh·nslab` single-core relayout ops would satisfy every extent assertion below.
    let per_head = ops
        .iter()
        .filter(|o| bare(o).starts_with("attn_o_h"))
        .count();
    assert_eq!(
        per_head, 0,
        "hd=128 mq=1 handoff: the per-head finalize loop must not run — the handoff replaces it"
    );
    for o in &slab_ops {
        assert_eq!(
            o.n.get("mb_").copied(),
            Some(NQH as i64),
            "hd=128 mq=1 handoff {}: one slab's finalize carries ALL {NQH} heads at once — at \
             mq == 1 a row IS a head",
            o.name
        );
        assert_eq!(
            o.n.get("out_").copied(),
            Some(POOL_STICK as i64),
            "hd=128 mq=1 handoff {}: a slab's finalize is ONE STICK wide — the stick-blocked write \
             law's contiguous degeneracy over one slab of columns is what makes one op per slab possible",
            o.name
        );
    }
    // ⛔ AND THE TWO SLABS MUST NOT BE THE SAME DESCRIPTOR — a loop emitting the same op twice
    // would satisfy every count above and leave slab 1 of `out` unwritten.
    assert_ne!(
        slab_ops[0].body, slab_ops[1].body,
        "hd=128 mq=1 handoff: the two slabs' finalizes emit IDENTICAL descriptors, so slab 1 \
         writes slab 0's bytes and the upper half of every head's output is never produced"
    );

    // ⛔ THE DESYNCS — each is half a restructure, and each must refuse rather than bake.
    // (`.err().expect` and not `expect_err`: the Ok type is `Vec<EmittedOp>`, which carries no
    // `Debug`, and the refusal — not the success — is what this pass is about.)
    for mq in [2u32, 8] {
        let err = try_emit_at::<HD_8B>(mq, true).err().expect(
            "handoff at mq > 1 must refuse: no weight permutation can fix the relayout there",
        );
        assert!(
            err.0.contains("mq"),
            "hd=128 mq={mq} handoff refusal must name the width: {err}"
        );
    }
    let err = try_emit_at::<HD_2B>(1, true)
        .err()
        .expect("handoff at one slab must refuse: there is nothing to restructure");
    assert!(
        err.0.contains("slab"),
        "hd=64 mq=1 handoff refusal must name the slab count: {err}"
    );
}
