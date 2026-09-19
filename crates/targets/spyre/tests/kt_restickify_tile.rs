// SPDX-License-Identifier: Apache-2.0
//! THE Kᵀ RESTICKIFY'S TILE IS ONE STICK ON **BOTH** AXES, AND THE RUNTIME'S PER-STEP SHIFT IS THAT
//! SAME QUANTITY — both pinned off the EMITTED descriptor's own `N_`, not off a re-derivation of the
//! extents that were handed in.
//!
//! ⛔⛔⛔ A MULTI-STICK Kᵀ RESTICKIFY IS THE KNOWN-BAD ON-CARD SHAPE, and it has cost two separate bugs,
//! one per axis:
//! * SLOT axis: the on-card ReStickify wrote the 2nd Kᵀ stick wrong, which garbled decode past 64
//!   tokens — the fix was to stop asking it to, which is why the new-block leg sweeps row WINDOWS;
//! * FEATURE axis: the streamed prefix leg passed the whole head dim, so at `hd = 128` it asked for two
//!   feature sticks. granite-3.1-8b fp8 answered "France is a miswrite." repeatedly, DEGENERATE FROM THE
//!   FIRST TOKEN — and granite-3.1-2b (hd=64, one slab) is byte-identical either way, so the 2b card gate
//!   passed it straight through.
//!
//! ⭐ SO THE FEATURE AXIS IS NOT A PARAMETER: [`KtTile`] has one door per slot window and NO feature door
//! at all. That makes the bad shape unspellable, and the last test here is the reason that mattered —
//! it emits real attention AT `hd = 128` and reads every Kᵀ op's declared extents back out of the
//! descriptor, which is the check the 2b gate structurally could not be.
//!
//! ⛔ AND THE SHIFT: a decode step appends ONE slot, so exactly one of a page's `PAGE_SLOTS / STK`
//! stick-blocks has a stale Kᵀ. [`KtTile::of_write_slab`] is the tile that covers that block; the runtime
//! moves the op onto the live one by shifting its KV-segment base (`fold_plan::slab_delta`) — ONE number,
//! for an op that reads natural K and writes Kᵀ through that one base. The tile is what the emitter
//! declares (`N_`); the shift is what the pool's address law says a stick-block costs
//! (`PagedKvPool::slab_shift_elems`). A shift SMALLER than the tile re-transposes ground already covered
//! and leaves the block's tail stale; a shift LARGER skips slots outright, and either way the miss is a
//! partially-stale Kᵀ — the shape that reads back as fluent wrong output rather than as a crash. Nothing
//! in the type system relates those two, so it is asserted here, as an EQUATION rather than as `hd == 64`.

#![cfg(feature = "spyre")]

use ktir_superdsc::emit::assemble_restickify_kt_2d;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use scratchy_subtile::addr::DevOff;
use scratchy_subtile::sdsc_abstract::{
    AttnGeometry, KtTile, KvHead, POOL_STICK, PagedKvPool, RowWindow, attn_bundle_rows,
};
use std::num::NonZeroU32;

const NKVH: u32 = 8;

fn head0(nkvh: u32) -> KvHead {
    KvHead::new(0, NonZeroU32::new(nkvh).unwrap()).expect("head 0 is in range")
}

/// The emitted restickify's declared iteration extents, read out of the descriptor: `(out_, y_)` =
/// (feature width, slot width).
fn declared_extents(tile: KtTile) -> (u64, u64) {
    let mut sym = 0i64;
    let e = assemble_restickify_kt_2d(
        "kt_probe",
        tile,
        "t_kc",
        DevOff::ZERO,
        "t_kct",
        DevOff::ZERO,
        &mut sym,
        None,
    );
    extents_of(&e, "kt_probe")
}

/// `(out_, y_)` of an emitted op's declared iteration space.
fn extents_of(e: &ktir_superdsc::emit::EmittedOp, name: &str) -> (u64, u64) {
    let v = serde_json::to_value(&e.op).unwrap();
    let n = &v["dscs_"][0][name]["N_"];
    (
        n["out_"].as_u64().expect("declared feature extent"),
        n["y_"].as_u64().expect("declared slot extent"),
    )
}

/// ⭐ EVERY DOOR EMITS ONE STICK ON BOTH AXES, and the descriptor is EMITTABLE AT ALL.
///
/// That second half is not rhetorical: `restickify_kt_opspec_2d` puts both extents through
/// `StickExtent`, which refuses anything that is not a whole 64-element fp16 stick — and a sub-stick
/// Kᵀ extent is exactly what deleted the generic op walk this crate used to have. A 64-slot tile is
/// one whole stick, so it passes; this pins that it does, rather than assuming it.
#[test]
fn every_door_declares_one_stick_on_both_axes() {
    let one = (POOL_STICK as u64, POOL_STICK as u64);
    assert_eq!(
        declared_extents(KtTile::of_write_slab()),
        one,
        "the write-slab door must declare ONE stick-block of slots by ONE stick of features"
    );
    assert_eq!(
        declared_extents(KtTile::of_row_window(RowWindow::ROWS)),
        one,
        "the row-window door likewise — a window's rows are one stick by the window's own identity"
    );
}

/// ⛔⛔⛔ THE RUNTIME'S PER-STEP SHIFT IS EXACTLY THE TILE THE EMITTER DECLARED.
///
/// `slab_shift_elems` is what `fold_plan::slab_delta` multiplies the block index by; `out_ * y_` is
/// what the op covers. Equal means step `p+STK` starts precisely where step `p`'s tile ended.
#[test]
fn the_shift_is_exactly_the_tile() {
    let hd = POOL_STICK;
    let pool = PagedKvPool::new(NKVH as usize, hd as usize);
    let shift = pool
        .slab_shift_elems(head0(NKVH))
        .expect("at hd == STK one shift serves both planes");
    let (out, y) = declared_extents(KtTile::of_write_slab());
    assert_eq!(
        shift as u64,
        out * y,
        "the segment shift ({shift} elems) must equal the tile the op covers ({out} x {y}) — a \
         smaller shift leaves each block's tail stale, a larger one skips slots"
    );
}

/// ⛔ ABOVE ONE STICK NO SINGLE SHIFT SERVES BOTH PLANES.
///
/// Knat advances `STK` slots of `STK` elements each, INDEPENDENT of the head dim; Kᵀ advances one
/// slot-stick group of `hd * STK`. Those agree only at `hd == STK`, and a single segment shift cannot
/// serve two different strides — so `None` here is the emitter's instruction that the incremental
/// single-shift form has no answer at this head dim, and its absence would be a silently half-shifted
/// transpose at every head dim but one.
#[test]
fn no_single_shift_serves_both_planes_above_one_stick() {
    for hd in [POOL_STICK * 2, POOL_STICK * 4] {
        let pool = PagedKvPool::new(NKVH as usize, hd as usize);
        assert_eq!(
            pool.slab_shift_elems(head0(NKVH)),
            None,
            "hd={hd}: the two planes' stick-block strides differ, so the incremental form has no \
             single shift and must be refused"
        );
    }
    // And the refusal is about the RELATION, not about being large: at hd == STK it is granted.
    assert!(
        PagedKvPool::new(NKVH as usize, POOL_STICK as usize)
            .slab_shift_elems(head0(NKVH))
            .is_some()
    );
}

/// The grant does not depend on which kv head asks — a stick-block is a stick-block in every block.
#[test]
fn every_kv_head_gets_the_same_shift() {
    let pool = PagedKvPool::new(NKVH as usize, POOL_STICK as usize);
    let n = NonZeroU32::new(NKVH).unwrap();
    let first = pool.slab_shift_elems(head0(NKVH));
    for h in 0..NKVH {
        let head = KvHead::new(h, n).expect("in range");
        assert_eq!(
            pool.slab_shift_elems(head),
            first,
            "kv head {h} must take the same shift — a fused Slab group shares ONE stride"
        );
    }
}

/// Every Kᵀ restickify a real attention emission contains, with its declared `(out_, y_)`.
fn emitted_kt_ops<const HD: u32>(mq: u32, rar: bool) -> Vec<(String, (u64, u64))> {
    let mut sym = 0i64;
    let geom = AttnGeometry::<32, 8, HD>::minted();
    let width = attn_bundle_rows(geom, mq, rar).expect("a width the emitter can bake");
    let ops = assemble_attn(
        7,
        geom,
        width,
        PagedKvPool::PAGE_SLOTS as u32,
        PagedKvPool::PAGE_SLOTS as u32,
        "qs",
        "nks",
        "nvr",
        "kct",
        "kc",
        "vc",
        "pmask",
        "cmask",
        scratchy_target_spyre::bundle_code::PlaceId::Act(7),
        rar,
        &mut sym,
        None,
    )
    .expect("attention must emit");
    ops.iter()
        .filter(|e| e.op_name.contains("pfxkt") || e.op_name.contains("newkt"))
        .map(|e| (e.op_name.clone(), extents_of(e, &e.op_name)))
        .collect()
}

/// ⛔⛔⛔ THE LOCK, AT THE HEAD DIM THAT BROKE: `hd = 128`.
///
/// This is the test the 2b card gate structurally could not be. At hd=64 a whole-head-dim tile and a
/// per-slab tile are the SAME emission, so every 2b bundle was byte-identical whichever the emitter
/// asked for; at hd=128 the whole-head-dim ask is two feature sticks in one ReStickify, which the card
/// performs wrongly — granite-3.1-8b fp8 was degenerate from its first token.
///
/// So: emit real attention at hd=128 and read every Kᵀ op's declared extents back out of the emitted
/// descriptor. Each must be one stick on both axes, in both legs (`newkt`, the new block; `pfxkt`, the
/// streamed prefix window) and in both bundle kinds (a decode step, a prefill chunk).
#[test]
fn every_emitted_kt_restickify_is_one_stick_on_both_axes_at_hd_128() {
    for (mq, rar, kind) in [(1u32, true, "decode"), (31, false, "prefill chunk")] {
        let ops = emitted_kt_ops::<128>(mq, rar);
        assert!(
            !ops.is_empty(),
            "{kind}: an hd=128 emission must contain Kᵀ restickifies at all"
        );
        for (name, (out, y)) in &ops {
            assert_eq!(
                (*out, *y),
                (POOL_STICK as u64, POOL_STICK as u64),
                "{kind}: `{name}` declares a MULTI-STICK Kᵀ restickify — the on-card ReStickify \
                 writes its second stick wrong (this exact shape made granite-3.1-8b fp8 degenerate \
                 from the first token)"
            );
        }
    }
}

/// ⭐ AND THE SPLIT IS REAL WORK, NOT A RENAME: hd=128 emits TWICE the Kᵀ ops hd=64 does, because each
/// covers one feature slab. Without this the previous test could pass on an emission that silently
/// dropped the second slab instead of transposing it — one stick each, and half the features missing.
#[test]
fn hd_128_emits_one_kt_restickify_per_feature_slab() {
    let narrow = emitted_kt_ops::<64>(1, true);
    let wide = emitted_kt_ops::<128>(1, true);
    assert!(!narrow.is_empty(), "hd=64 emits Kᵀ restickifies");
    assert_eq!(
        wide.len(),
        narrow.len() * 2,
        "hd=128 is two feature slabs, so it needs two Kᵀ restickifies per (kv head, slot window) — \
         got {} against hd=64's {}",
        wide.len(),
        narrow.len()
    );
    // Every op is uniquely named: `nslab` ops sharing a name would collapse in the bundle, which is the
    // other way a slab can go missing.
    let mut names: Vec<&str> = wide.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "each slab's op must have its own name");
}
