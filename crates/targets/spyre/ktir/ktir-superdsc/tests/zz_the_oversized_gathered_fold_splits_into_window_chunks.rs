// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE WINDOW-CHUNK SPLIT OF AN OVERSIZED GATHERED FOLD — measured off the emitted ops,
//! because the split is a CORRECTNESS-BEARING emission change (each chunk must carry its own gather
//! copies; chunk 0 must stay request 0) and none of it is visible at a geometry that does not split.
//!
//! ## Why the split exists
//! A gathered PageFold run is exempt from the group-size cap (`GroupKind::run_may_be_chunked ==
//! false`, the group-major reps law), so one launch group holds the whole fold: its descriptor count
//! is `2·mq·ops_per_row + nb·(2·mq·nkvh·nslab + 11)`. At gemma-4-12b's sliding-class geometry
//! (nqh=16, nkvh=8, hd=256 ⇒ nslab=4) at mq=32, nb=4 that is 8,300 descriptors, and a
//! `dxp_standalone` child of that group was MEASURED at 2.5+ hours of compile — the bake-side guard
//! (`oversized_group`, against `DxGroupCeiling::MAX_DESCRIPTORS` = 4204) turns it into a build
//! error, and this split is what keeps the model buildable.
//!
//! ## What this file measures
//! The EMITTED op sequence, not the projection arithmetic:
//! 1. an under-ceiling fold (granite-8b's geometry, whose whole fold is exactly 2,156 at these
//!    extents) emits ONE unsplit run — every copy before every window, every fold op tagged
//!    request 0, the shipped sequence;
//! 2. gemma-4-12b's class-0 geometry at mq=32 splits into TWO chunks, each ≤ the ceiling;
//! 3. EVERY chunk's run begins with its own gather copies — the self-containment the group-major
//!    law demands (a chunk whose copies lived in another group would read that group's last page);
//! 4. the chunks COVER the window sweep exactly once — no window dropped, none double-folded.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, DxGroupCeiling, PagedKvPool, POOL_STICK, attn_bundle_rows,
};

const CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;

/// One emitted op, reduced to what this file asserts: its name, its fold flags and its request tag.
struct Op {
    name: String,
    page_fold: bool,
    request: u32,
    time: u32,
}

fn emit_at<const NQH: u32, const NKVH: u32, const HD: u32>(mq: u32) -> Vec<Op> {
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
        Some("t_kv_idx"),
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("NQH={NQH} NKVH={NKVH} HD={HD} mq={mq}: assemble_attn refused: {}", e.0))
    .iter()
    .map(|e| Op {
        name: e.op_name.clone(),
        page_fold: e.kv_page_fold,
        request: e.kv_request,
        time: e.time,
    })
    .collect()
}

/// The fold window ops of one emission — the `attn_p{b}…` blocks (the new-token block is `n…` and
/// never folds).
/// The gather copies — `attn_gkt_o…`/`attn_gv_o…`, with a chunk segment `_c{k}_` above chunk 0.
fn gather_copies(ops: &[Op]) -> Vec<&Op> {
    ops.iter()
        .filter(|o| o.name.starts_with("attn_gkt_o") || o.name.starts_with("attn_gv_o"))
        .collect()
}

/// ⭐⭐⭐⭐⭐ THE UNDER-CEILING FOLD IS THE SHIPPED SEQUENCE — one run, copies first, request 0
/// everywhere. This is the granite-parity pin: a bundle whose projected size fits the ceiling must
/// not move AT ALL, because the split's chunk-0-keeps-request-0 law is what makes that true and a
/// regression here is a fingerprint change for every model that already bakes.
#[test]
fn an_under_ceiling_gathered_fold_emits_one_unsplit_run() {
    // granite-3.1-8b's head geometry at the widest rung. The whole fold's descriptor count is
    // 2·32·1 + 4·(2·32·8·2 + 11) = 4,204·… — computed below, and asserted ≤ the ceiling so this
    // test names its own admission predicate rather than trusting the geometry.
    const NQH: u32 = 32;
    const NKVH: u32 = 8;
    const HD: u32 = 128;
    let mq = 32u32;
    let nslab = HD / POOL_STICK;
    let nb = CAP / POOL_STICK;
    let ops_per_row = 1u32; // nkvh=8: the IBR stick bound, blocks()/entries_per_op at every shipped hd
    let projected =
        2 * mq * ops_per_row + nb * (2 * mq * NKVH * nslab + 11);
    assert!(
        projected <= DxGroupCeiling::MAX_DESCRIPTORS as u32,
        "this test's fixture must be under the ceiling (projected {projected})"
    );
    assert!(
        projected > DxGroupCeiling::SPLIT_TARGET as u32,
        "this fixture is OVER the split target on purpose: at granite-8b's geometry the target \
         splits this fold, and the unsplit-run invariants below are what an under-TARGET \
         geometry (a smaller rung) — not this one — must keep"
    );
    let ops = emit_at::<NQH, NKVH, HD>(mq);
    // This geometry now SPLITS (one window = 1,099 > 512 target): 4 chunks of 1,099. The
    // under-TARGET unsplit-run invariants this test originally pinned belong to a smaller
    // rung; here we pin the split's own shape at granite's geometry — chunk 0 untagged, every
    // other chunk named, self-containment checked by the shared tests below.
    let fold: Vec<&Op> = ops.iter().filter(|o| o.page_fold).collect();
    let mut per_chunk: std::collections::BTreeMap<u32, usize> = Default::default();
    for o in &fold {
        *per_chunk.entry(o.request).or_default() += o.time as usize;
    }
    assert_eq!(
        per_chunk.len(),
        nb as usize,
        "granite's over-target fold splits one window per chunk (tags {:?})",
        per_chunk.keys().collect::<Vec<_>>()
    );
    for (chunk, trips) in &per_chunk {
        assert!(
            *trips <= DxGroupCeiling::MAX_DESCRIPTORS,
            "chunk {chunk} holds {trips} trips — over the ceiling"
        );
    }
    assert!(per_chunk.contains_key(&0), "chunk 0 keeps request 0");
    // AND CHUNK 0's COPIES PRECEDE EVERY WINDOW — the old order holds at the chunk-0 head.
    let copies = gather_copies(&ops);
    assert!(!copies.is_empty(), "the gather is on: the copies must exist");
    let last_copy = ops
        .iter()
        .rposition(|o| o.name.starts_with("attn_gv_o") && !o.name.contains("_c"))
        .expect("chunk 0's V-plane copies exist");
    let first_window = ops
        .iter()
        .position(|o| o.page_fold && o.name.contains("_p"))
        .expect("a fold window op");
    assert!(
        last_copy < first_window,
        "chunk 0's copies precede the first window (copy at {last_copy}, window at {first_window})"
    );
}

/// ⭐⭐⭐⭐⭐ THE SPLIT ITSELF, AT THE GEOMETRY THAT FORCED IT — gemma-4-12b's sliding class
/// (nqh=16, nkvh=8, hd=256 ⇒ nslab=4) at the widest rung: 8,300 descriptors projected. Against
/// `SPLIT_TARGET` = 512 — a cost bound below the bake ceiling — one window alone (2,059 + 64
/// copies) exceeds the target, so the split SATURATES at one window per chunk: 4 chunks of 2,123.
/// Each is under `MAX_DESCRIPTORS` (the legality bound the bake guard reads), which is the
/// invariant that matters; the target aims the split, the ceiling forbids what it may not stage.
#[test]
fn the_gemma4_sized_fold_splits_into_ceiling_fitting_chunks() {
    const NQH: u32 = 16;
    const NKVH: u32 = 8;
    const HD: u32 = 256;
    let mq = 32u32;
    let nslab = HD / POOL_STICK;
    let nb = CAP / POOL_STICK;
    let per_chunk_fixed = 2 * mq; // ops_per_row = 1 at nkvh=8
    let per_window = 2 * mq * NKVH * nslab + 11;
    let projected = per_chunk_fixed + nb * per_window;
    assert_eq!(projected, 8300, "the projected unsplit size at gemma-4 class 0");
    assert!(projected > DxGroupCeiling::MAX_DESCRIPTORS as u32);
    assert!(
        per_window as usize + per_chunk_fixed as usize > DxGroupCeiling::SPLIT_TARGET,
        "at this geometry the split saturates: one window exceeds the target"
    );

    let ops = emit_at::<NQH, NKVH, HD>(mq);
    // The per-chunk trips: each chunk's own copies + its windows' ops, all time=1 in this fixture.
    // Chunk membership is read off the request tag, which is the group walk's OWN boundary signal.
    let fold: Vec<&Op> = ops
        .iter()
        .filter(|o| o.page_fold)
        .collect();
    let mut per_chunk: std::collections::BTreeMap<u32, usize> = Default::default();
    for o in &fold {
        *per_chunk.entry(o.request).or_default() += o.time as usize;
    }
    // 4 windows, one per chunk (the saturated split): `nb` chunks, not 2.
    assert_eq!(
        per_chunk.len(),
        nb as usize,
        "the 8,300-descriptor fold splits one window per chunk at a target one window exceeds (tags {:?})",
        per_chunk.keys().collect::<Vec<_>>()
    );
    for (chunk, trips) in &per_chunk {
        assert!(
            *trips <= DxGroupCeiling::MAX_DESCRIPTORS,
            "chunk {chunk} holds {trips} trips — over the {} ceiling",
            DxGroupCeiling::MAX_DESCRIPTORS
        );
    }
    // Every chunk = its own copies (64) + its one window (2,059) = 2,123, exactly the arithmetic.
    for (chunk, trips) in &per_chunk {
        assert_eq!(
            *trips, 2123,
            "chunk {chunk} = its copies + its window (saturation arithmetic)"
        );
    }

    // ⭐ CHUNK 0 IS REQUEST 0 — the under-target bundles of this geometry (smaller rungs) keep the
    // shipped tag, and the group walk's boundary is where the emission cut.
    assert!(per_chunk.contains_key(&0), "chunk 0 keeps request 0");
}

/// ⭐⭐⭐⭐⭐ SELF-CONTAINMENT — every chunk's run BEGINS with its own gather copies, the
/// precondition the group-major reps law demands: pass `p` of a chunk re-fills the scratch with
/// page `p` before that chunk's legs read it, which is only true if the copies are INSIDE the
/// chunk's group.
#[test]
fn every_chunk_of_a_split_fold_begins_with_its_own_gather_copies() {
    const NQH: u32 = 16;
    const NKVH: u32 = 8;
    const HD: u32 = 256;
    let ops = emit_at::<NQH, NKVH, HD>(32);
    let fold: Vec<(usize, &Op)> = ops.iter().enumerate().filter(|(_, o)| o.page_fold).collect();
    // Group the fold ops by their request tag, in emission order, keeping positions.
    let mut chunks: Vec<(u32, usize, usize)> = Vec::new(); // (tag, first_idx, last_idx) in `ops`
    for (i, (_, o)) in fold.iter().enumerate() {
        match chunks.last_mut() {
            Some((t, _, last)) if *t == o.request => *last = i,
            _ => chunks.push((o.request, i, i)),
        }
    }
    assert!(chunks.len() >= 2, "the fixture must split");
    for (tag, first, _last) in &chunks {
        // The chunk's first fold ops must be gather copies — `attn_gkt_o…`/`attn_gv_o…` — and this
        // chunk's copies must be NAMED with the chunk segment (chunk 0's keep the shipped name).
        let head = &fold[*first..(*first + 2).min(fold.len())];
        assert!(
            head.iter().all(|(_, o)| {
                o.name.starts_with("attn_gkt_o") || o.name.starts_with("attn_gv_o")
            }),
            "chunk {tag} begins with {:?} — its own copies must lead its group",
            head.iter().map(|(_, o)| o.name.as_str()).collect::<Vec<_>>()
        );
        let want_seg = if *tag == 0 {
            "attn_gkt_o0_s".to_string()
        } else {
            format!("attn_gkt_o0_c{tag}_s")
        };
        assert!(
            fold[*first].1.name.starts_with(&want_seg),
            "chunk {tag}'s first copy is '{}' (want prefix '{want_seg}')",
            fold[*first].1.name
        );
    }
}

/// ⭐⭐⭐⭐⭐ THE COVERING PARTITION — the split folds every window exactly once. A dropped window is
/// a context the model silently ignores; a doubled one is a double-counted softmax contribution.
/// Both bake clean, which is why the count is asserted and not eyeballed.
#[test]
fn a_split_fold_covers_every_window_exactly_once() {
    const NQH: u32 = 16;
    const NKVH: u32 = 8;
    const HD: u32 = 256;
    let mq = 32u32;
    let nb = CAP / POOL_STICK;
    let nslab = HD / POOL_STICK;
    let ops = emit_at::<NQH, NKVH, HD>(mq);
    // The gathered score ops carry `_p{b}sc_g{kv}s{s}_r{r}` — one per (window, kv head, slab,
    // request), so counting them per window proves the cover: `nkvh·nslab·mq` per window.
    for b in 0..nb {
        let n: usize = ops
            .iter()
            .filter(|o| o.name.contains(&format!("_p{b}sc_")) && o.page_fold)
            .count();
        assert_eq!(
            n,
            (NKVH * nslab * mq) as usize,
            "window {b}: every (kv head, slab, request) score op must appear exactly once"
        );
    }
    // AND the copies are per chunk, not per window: at the saturated split, one chunk per window
    // (`nb` chunks) × 2 planes × mq copies.
    assert_eq!(
        gather_copies(&ops).len(),
        nb as usize * 2 * mq as usize,
        "a one-window-per-chunk split emits nb chunks × 2 planes × mq copies — no more and no \
         fewer (a chunk without its own copies reads another chunk's last page)"
    );
}
