// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE WHOLE-DXP EAR ENVELOPE, WIRED AT THE LM HEADS THAT MEASURED IT.
//!
//! dxp refuses a whole-dxp descriptor whose per-core KERNEL slab passes its
//! mutable-address reach (`DtException: EAR overflow detected,
//! MutableAddrSplitting.cpp:814`). The boundary is not computable from this crate's
//! input schema, so it is pinned by MEASUREMENT as
//! [`WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT`] = 150 MiB: the gemma-4 12b's two-row head
//! (`1920 × 32768 × 2 B = 125.8 MB/core`, the same `{in:2, mb:2, out:8}` division)
//! is accepted whole-dxp and card-proven coherent; the 31b's (`2688 × 32768 × 2 B
//! = 176.2 MB/core`) is refused. Past the line the op takes the frontend's `out`
//! time trips — the staging that baked the 31b RC=0 and ran it coherent on card.
//!
//! ## The regression this file pins
//! The envelope landed as #598 (2026-10-17) and was DELETED by #714 one day after
//! the last gemma-4 fp8 bake, on the claim that "a dxp EAR-overflow refusal is the
//! loud generic refusal" — gated granite-only, whose 49k vocab keeps the slab 6×
//! under the line so the trips never fire there. The batch-decode rungs that
//! followed minted head shapes the envelope existed for: the gemma-4 12b fp8 card
//! build of 2026-11-17 died at `41_matmul_o2350` of `ktir_decode_cb_gemma_4_12b…`
//! — m=7, `{mb:7, out:4}`, per-core slab `3840 × 65536 × 2 B = 503 MB/core`, 3.4×
//! past the line. A refusal is not a fix; the law is restored, and this file is
//! its wiring pin: the over-line heads take the trips, the under-line heads stay
//! whole-dxp byte-identically, and every staging still covers the whole vocabulary.
//!
//! Run locally:
//!
//! ```sh
//! cargo test -p ktir-superdsc --test zz_the_lm_head_kernel_span_meets_the_ear
//! ```

use ktir_superdsc::addr::{col_of, rc_of};
use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::{EmittedOp, rb};
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    assemble_matmul_seeded, assemble_matmul_windowed,
};
use ktir_superdsc::sdsc_abstract::{KernelTag, MatK, MatM, MatN, Stk};
use ktir_superdsc::superdsc_opspec::Df;

/// The refused op's own geometry, read off the 2026-11-17 bake (`sdsc_41.json`):
/// `N_ {in: 3840, out: 262144, mb: 7}`, `numWkSlicesPerDim {in: 1, mb: 7, out: 4}`,
/// per-core `dataStageParam_.ss_ {in: 3840, out: 65536, mb: 1}` — a 503 MB/core
/// KERNEL slab, 3.4× past the 150 MiB envelope.
const M: u32 = 7;
const K: u32 = 3840;
const N: u32 = 262144;
const OUT_SPLIT: u32 = 4;
const PER_CORE_OUT: u32 = N / OUT_SPLIT;

/// granite-3.1-8b's lm_head — the under-line control: hidden 4096, vocab 49152,
/// per-core slab at one row `4096 × 1536 × 2 B ≈ 12.6 MB/core`, 12× under the
/// envelope. This is the shape that kept the envelope's deletion invisible.
const GRANITE_K: u32 = 4096;
const GRANITE_N: u32 = 49152;

type Op = EmittedOp<TpWorld<1>>;

/// The op's wire name, derived from its typed geometry — every form names itself,
/// so the dump's envelope keys cannot collide and no caller threads a string.
fn op_name(m: MatM, k: MatK, n: MatN) -> String {
    format!("matmul_o_m{}_k{}_n{}", m.get(), k.get(), n.get())
}

/// The whole-tensor form the bake's `MatmulTile` door emits for a head: one plain
/// fp16 matmul `[m, k]·[k, n] -> [m, n]` at batch 1. The extents arrive as the
/// typed [`MatM`]/[`MatK`]/[`MatN`] the windowed door takes; the seeded door's raw
/// parameters are unwrapped once, here at its own boundary.
fn whole_form(m: MatM, k: MatK, n: MatN) -> Op {
    let (mu, ku, nu) = (m.get(), k.get(), n.get());
    assemble_matmul_seeded(
        &op_name(m, k, n),
        mu,
        nu,
        ku,
        1,
        &rb("a", mu, ku),
        &Stk::<KernelTag>::kernel(ku as usize, nu as usize, "w"),
        &rb("o", mu, nu),
        &mut 0,
        None,
    )
    .expect("a head matmul emits")
}

/// One OUT-SLICE of the head at `slice` columns starting at column `col_start`,
/// through the shipped WINDOWED door — the per-trip shape the time tiler produces.
/// The corner arrives as a column index and is wrapped by the door's own typed
/// address constructors ([`rc_of`]/[`col_of`]), exactly as the `MatmulTile` door
/// wraps a region's corner.
fn slice_form(m: MatM, k: MatK, n: MatN, slice: MatN, col_start: u32) -> Op {
    let (mu, ku, nu, su) = (m.get(), k.get(), n.get(), slice.get());
    assemble_matmul_windowed(
        &format!("{}_c{col_start}", op_name(m, k, slice)),
        m,
        slice,
        k,
        &rb("a", mu, ku),
        rc_of(mu, ku, 0, 0, Df::Fp16),
        &Stk::<KernelTag>::kernel(ku as usize, su as usize, "w"),
        rc_of(ku, nu, 0, col_start, Df::Fp16),
        ku,
        &rb("o", mu, su),
        col_of(mu, nu, col_start, Df::Fp16),
        &mut 0,
        None,
    )
    .expect("an out-slice of the head emits — the windowed door's own form")
}

/// The refused form's facts, restated as numbers so a drift in the cost model moves
/// this file rather than silently pinning a different op.
#[test]
fn the_whole_form_is_the_refused_op() {
    let op = whole_form(MatM::of_token_rows(M), MatK::of_in_features(K), MatN::of_out_features(N));
    // ⛔ THE OVER-LINE FORM TAKES THE TRIPS — time > 1, and the trips sum to the
    // whole vocabulary across the per-core extent.
    assert!(
        op.time > 1,
        "the m={M} head (per-core slab {} B, over the {}-B envelope) must take the \
         frontend trips, not a whole-dxp descriptor dxp refuses",
        (K * PER_CORE_OUT * 2),
        ktir_superdsc::superdsc_opspec::WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT
    );
    let sp = op.op.as_ref().expect("a descriptor");
    assert_eq!(sp.numWkSlicesPerDim_["mb"], M, "mb splits per the bake");
    assert_eq!(
        sp.numWkSlicesPerDim_["out"], OUT_SPLIT,
        "out splits per the bake"
    );
    assert_eq!(sp.numWkSlicesPerDim_.get("in").copied().unwrap_or(1), 1);
    let dsc = op
        .op
        .as_ref()
        .and_then(|o| o.dscs_.first())
        .and_then(|m| m.values().next())
        .expect("a dsc");
    // The trips divide BOTH the iteration extent and the per-core stage — the
    // per-trip form `rewrite_op_for_time_tile` writes — and they must COVER the
    // vocabulary exactly: `time · N_.out_ == n`.
    assert_eq!(
        op.time as i64 * dsc.N_.out_,
        N as i64,
        "the {} trips of N_.out_ {} must sum to the whole vocabulary",
        op.time,
        dsc.N_.out_
    );
    assert_eq!(
        op.time as i64 * dsc.dataStageParam_["0"].ss_.out_,
        PER_CORE_OUT as i64,
        "the trips divide the per-core out extent exactly"
    );
    // The per-trip per-core KERNEL slab is under the envelope (and under LX).
    let per_trip = dsc.dataStageParam_["0"].ss_.in_ * dsc.dataStageParam_["0"].ss_.out_ * 2;
    assert!(
        per_trip as u64 <= ktir_superdsc::superdsc_opspec::WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT,
        "per-trip per-core slab {per_trip} B must be inside the envelope"
    );
    eprintln!(
        "M7 HEAD: {} trips, N_.out_ {}, per-core ss_ {{in: {}, out: {}}} — per-trip slab \
         {per_trip} B",
        op.time,
        dsc.N_.out_,
        dsc.dataStageParam_["0"].ss_.in_,
        dsc.dataStageParam_["0"].ss_.out_,
    );
}

/// The one-row 12b head — `62.9 MB/core`, INSIDE the envelope — stays whole-dxp:
/// ONE descriptor, no trips, byte-identical to the form that has always shipped.
#[test]
fn the_one_row_head_stays_whole_dxp() {
    let op = whole_form(MatM::of_token_rows(1), MatK::of_in_features(K), MatN::of_out_features(N));
    assert_eq!(
        op.time, 1,
        "the one-row 12b head (62.9 MB/core) is inside the envelope — whole-dxp"
    );
    let sp = op.op.as_ref().expect("a descriptor");
    assert_eq!(sp.numWkSlicesPerDim_["out"], 32, "the one-row fill split");
    let dsc = op
        .op
        .as_ref()
        .and_then(|o| o.dscs_.first())
        .and_then(|m| m.values().next())
        .expect("a dsc");
    assert_eq!(dsc.N_.out_, N as i64);
    assert_eq!(dsc.dataStageParam_["0"].ss_.out_, (N / 32) as i64);
}

/// granite's head — the control that kept the deletion invisible for 30 days:
/// 12× under the envelope, whole-dxp, unchanged.
#[test]
fn the_granite_head_stays_whole_dxp() {
    let op = whole_form(
        MatM::of_token_rows(1),
        MatK::of_in_features(GRANITE_K),
        MatN::of_out_features(GRANITE_N),
    );
    assert_eq!(op.time, 1, "granite's head is far inside the envelope");
}

/// The gemma-4 31b's two-row form — the shape that MEASURED the line's upper side
/// (176.2 MB/core, refused) — takes the trips.
#[test]
fn the_31b_two_row_head_takes_the_trips() {
    let op = whole_form(MatM::of_token_rows(2), MatK::of_in_features(5376), MatN::of_out_features(N));
    assert!(
        op.time > 1,
        "the 31b two-row head (176.2 MB/core) is past the envelope — trips"
    );
    let dsc = op
        .op
        .as_ref()
        .and_then(|o| o.dscs_.first())
        .and_then(|m| m.values().next())
        .expect("a dsc");
    assert_eq!(
        op.time as i64 * dsc.N_.out_,
        N as i64,
        "the trips cover the vocabulary exactly"
    );
}

/// Every trip-shaped form, written out for the pod's `dxp_standalone` acceptance run
/// — the `{name: dsc}` envelope the bundle stub reads, one json per form under
/// `$CARGO_TARGET_TMPDIR/lm_head_ear/`. The per-trip descriptor is what dxp compiles
/// (`concrete_trips` bakes exactly this shape per trip), so a green dxp run here is
/// the per-op half of the bake's proof.
#[test]
fn dump_the_trip_forms_for_the_pod_dxp_run() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("lm_head_ear");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    let (m7, k, n) = (
        MatM::of_token_rows(M),
        MatK::of_in_features(K),
        MatN::of_out_features(N),
    );
    let forms: Vec<Op> = vec![
        whole_form(m7, k, n),
        whole_form(MatM::of_token_rows(1), k, n),
        whole_form(MatM::of_token_rows(2), MatK::of_in_features(5376), n),
        slice_form(m7, k, n, MatN::of_out_features(N / 32), 0),
        slice_form(m7, k, n, MatN::of_out_features(N / 32), N - N / 32),
    ];
    for (i, op) in forms.iter().enumerate() {
        let d = op.op.as_ref().expect("a descriptor");
        let wrapped = serde_json::json!({ &op.op_name: serde_json::to_value(d).expect("json") });
        let p = dir.join(format!("sdsc_{i}.json"));
        std::fs::write(&p, serde_json::to_string_pretty(&wrapped).expect("json"))
            .unwrap_or_else(|e| panic!("writing {}: {e}", p.display()));
        eprintln!("WROTE {} (time {})", p.display(), op.time);
    }
}
