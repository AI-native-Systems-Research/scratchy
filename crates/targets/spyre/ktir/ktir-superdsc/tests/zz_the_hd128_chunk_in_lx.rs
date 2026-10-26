// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ granite-3.1-8b's ONE-REQUEST DECODE CHUNK KEEPS ITS GATHERED Kᵗ AND V IN LX, at every rung the
//! frontend's LX window holds, from the shipped `assemble_attn` against a real layout whose attention
//! class registry names hd 128 (the query-group form the 8b decode runs).
//!
//! At head dim 128 a gather entry is one 16 KiB stick block. It is a 64-slot stick of Kᵗ's
//! `[128, slots]` (both slabs), and 128 slots of ONE slab of V's slab-major `[slab][slot][stick]`. So
//! the two legs have different parts:
//!   * Kᵗ: 8 pages a part, one copy per (part, kv head). The part's score splits its columns, one slot
//!     stick per core.
//!   * V: 16 pages a part, one copy per (part, kv head, slab). The part's per-slab PV splits its
//!     contraction, 128 slots per core.
//!
//! Every pin is one 16 KiB entry per core. All of them must fit `[LX_PIN_BASE, LX_PIN_END)`, the
//! frontend's share of LX (`FRONTEND_LX_END`), which refuses the 48-, 56- and 64-page rungs.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::attn::chunk_prob_pins;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    AttnFacts, RmsnormFacts, assemble_attn, assemble_rmsnorm,
};
use ktir_superdsc::place::{PlaceId, SynthRole};
use ktir_superdsc::placement::{BundleLayout, LX_PIN_BASE, LX_PIN_END, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, ChunkRowKind, ChunkScratch, KvExtent, PagedKvPool, QGroupBlock, SlotCount,
    attn_bundle_rows,
};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 128;
const NSLAB: u32 = 2;
const CAP: u32 = 256;
/// The storage/sweep pair on the boundary's one carrier — the sweep is the full cap (no rung).
const EXTENT: KvExtent = KvExtent::new(SlotCount::new(CAP), SlotCount::new(CAP));
const ENTRY_BYTES: u64 = HD as u64 * 64 * 2;

fn pool() -> PagedKvPool {
    PagedKvPool::new(NKVH, HD)
}

/// The names-and-placements half of [`emit`]: the layout every hd128 emission in this file
/// starts from, and the placed activation names `assemble_attn` binds. Factored out so the
/// declare-pass tests can declare op-local pins into the SAME layout before the chunk's
/// admission runs.
fn hd_layout() -> (BundleLayout, Vec<String>) {
    let mut layout = BundleLayout {
        attn_class_hds: vec![HD],
        ..BundleLayout::default()
    };
    let mut names = Vec::new();
    let ident = ktir_superdsc::reserved_tids::identity_class_tid(0);
    for (i, tid) in (1..=8u32).chain([900, ident]).enumerate() {
        let name = PlaceId::Act(tid).to_string();
        layout
            .ids
            .borrow_mut()
            .insert(name.clone(), PlaceId::Act(tid));
        names.push(name);
        layout.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: (i as u64) << 30,
                size: 1 << 30,
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    (layout, names)
}

/// The emission half of [`emit`]: one 40-page-class chunk against a caller-supplied layout,
/// whose op-local pin slots the caller may already have declared. The chunk's `lx_form`
/// admission reads the layout's FINAL slot sum (`lx_op_local_slot_bytes`), so what this
/// returns is the verdict of that budget.
fn emit_chunk(
    layout: &BundleLayout,
    names: &[String],
    pages: u32,
) -> Vec<(String, serde_json::Value)> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, 1, true).expect("the one-request decode rung");
    let chunk = ChunkScratch::of_chunk(pool(), pages).expect("a chunk granite-8b admits");
    let mut sym = 0i64;
    assemble_attn(
        AttnFacts {
            t: 0,
            geom,
            bundle_rows: rows,
            extent: EXTENT,
            qs: &names[0],
            new_k_scaled: &names[1],
            new_v: &names[2],
            kct: &names[3],
            vc: &names[4],
            pmask: &names[5],
            cmask: &names[6],
            kv_block_index: Some(&names[7]),
            chunk: Some(chunk),
            out_id: PlaceId::Act(900),
            rows_are_requests: false,
            headmajor_handoff: false,
        },
        &mut sym,
        Some(layout),
    )
    .unwrap_or_else(|e| panic!("pages={pages}: assemble_attn refused: {}", e.0))
    .iter()
    .map(|e| {
        let v = serde_json::to_value(&e.op).unwrap();
        let name = v["dscs_"][0]
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .expect("one named dsc per emitted op");
        (name, v)
    })
    .collect()
}

fn emit(pages: u32) -> Vec<(String, serde_json::Value)> {
    let (l, names) = hd_layout();
    emit_chunk(&l, &names, pages)
}

fn lx_only(v: &serde_json::Value) -> bool {
    v["memOrg_"]
        .as_object()
        .is_some_and(|m| m.len() == 1 && m.contains_key("lx"))
}

/// The one start address an operand's allocate node gives every core, or `None` if they differ.
fn pin(body: &serde_json::Value, lds: &serde_json::Value) -> Option<u64> {
    let idx = lds["ldsIdx_"].as_u64();
    let alloc = body["scheduleTree_"]
        .as_array()?
        .iter()
        .find(|n| n["nodeType_"] == "allocate" && n["ldsIdx_"].as_u64() == idx)?;
    assert_eq!(alloc["component_"], "lx");
    let starts: std::collections::BTreeSet<u64> = alloc["startAddressCoreCorelet_"]["data_"]
        .as_object()?
        .values()
        .map(|a| a.as_str().map_or_else(|| a.as_u64(), |s| s.parse().ok()))
        .collect::<Option<_>>()?;
    (starts.len() == 1).then(|| starts.into_iter().next().unwrap())
}

/// ⭐ WHICH RUNGS KEEP WHICH LEG IN LX, from the form's own law. Kᵗ is in LX on every rung that is a whole
/// number of 8-page parts, V on every rung with at least one 16-page part whose tail is whole sticks, and
/// both only while their pins fit the frontend's window. The 48-, 56- and 64-page rungs would pin 1.5,
/// 1.75 and 2 MiB; the window is 1,494,272 B. The 8-page rung keeps Kᵗ alone in LX (V is all tail, main's
/// HBM V leg); the 2-, 4- and 12-page rungs have neither leg whole.
#[test]
fn the_hd128_lx_form_covers_exactly_the_rungs_its_pins_fit() {
    let window = LX_PIN_END - LX_PIN_BASE;
    assert_eq!(window, 1_494_272);
    let mut admitted = Vec::new();
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        // Both budget terms 0: this ladder is the GATHER law alone (which rungs have a whole leg
        // and what the gather pins weigh). The full core-local budget — gathers plus the
        // probability-part pins — is `the_op_local_pin_slots_refuse_a_chunk_the_window_no_longer_holds`
        // below.
        assert!(
            chunk.lx_form(ChunkRowKind::Prompt, 0, 0).is_none(),
            "{j} page(s), prompt"
        );
        // A decode batch's chunk takes the same LX form: the law reads only the chunk's pages.
        assert_eq!(
            chunk.lx_form(ChunkRowKind::Batch, 0, 0),
            chunk.lx_form(ChunkRowKind::OneRequest, 0, 0),
            "{j} page(s), batch"
        );
        // Kᵗ: one 16 KiB tensor per (8-page part, kv head). V: one per (16-page part, kv head, slab).
        // Both come to `pages * nkvh / 8` entries a core.
        let k_pages = if j % 8 == 0 { j } else { 0 };
        let v_pages = j / 16 * 16;
        let pinned = u64::from(k_pages + v_pages) * u64::from(NKVH) * ENTRY_BYTES / 8;
        let fits = (v_pages > 0 || k_pages > 0) && pinned <= window;
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0) else {
            assert!(!fits, "{j} page(s): {pinned} B of pins fit, but no LX form");
            continue;
        };
        assert!(fits, "{j} page(s): an LX form pinning {pinned} B");
        admitted.push(j);
        assert_eq!(lx.part_pages(), 16, "{j} page(s): a V part");
        assert_eq!(lx.lx_parts() * 16 + lx.tail_pages(), j);
        assert_eq!(lx.lx_tensors(), lx.lx_parts() * NKVH * NSLAB);
        assert_eq!(lx.per_core_bytes(), ENTRY_BYTES);
        assert_eq!(lx.pinned_bytes_per_core(), pinned, "{j} page(s)");
        assert_eq!(lx.keys().is_some(), k_pages > 0, "{j} page(s)");
        if let Some(k) = lx.keys() {
            assert_eq!(k.part_pages(), 8);
            assert_eq!(k.tensors(), j / 8 * NKVH);
        }
    }
    assert_eq!(admitted, [8, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60]);
}

/// ⭐⭐ THE CORE-LOCAL TERMS OF THE ONE WINDOW BUDGET — `lx_window_fits`, the same function the pin
/// allocator's own guard asserts, from the WITNESS's side, with BOTH terms the emission mints: the
/// gather pins (`pinned_bytes_per_core`) and the probability-part pins beside them
/// (`chunk_prob_pins`, the `cesub` slice the mint declares). The widest admitted rung (40 pages,
/// 1,179,648 B of gather pins + 2,560 B of probability parts) is admitted while its op-local pin
/// slots leave room, admitted AT the boundary, and refused one byte past it. The bake reads both
/// terms off the layout (`lx_op_local_slot_bytes` + `lx_prob_pin_bytes`), the host off the baked
/// artifact — both meet at this one function, so neither arm can budget against another rung's
/// pins.
#[test]
fn the_op_local_pin_slots_refuse_a_chunk_the_window_no_longer_holds() {
    let chunk = ChunkScratch::of_chunk(pool(), 40).unwrap();
    // The emission's own derivation of the probability part's rows: the one-request decode's
    // re-laid query-group block (assemble_attn's `b_rows` at this rung, mq 1). Its division is
    // the law (`ChunkBufs::sm_division_of` of a solo chunk — only the batch runs sticks first).
    let q = QGroupBlock::of_decode(NKVH, NQH / NKVH, HD, 1).unwrap();
    let b_rows = chunk.row_blocks_of_query_groups(q).unwrap().rows();
    let (part_pages, pins) =
        chunk_prob_pins(chunk, b_rows, false, ktir_superdsc::work::EwDivision::Law);
    assert_eq!(
        part_pages, 40,
        "the whole 40-page chunk is one probability part at these rows"
    );
    let pins =
        pins.expect("the 40-page part is untiled at these rows, so it has a core-local slice");
    assert_eq!(
        pins,
        vec![(SynthRole::ChunkESub, 2, 2_560u64)],
        "the decode form's one probability-part pin: 4 rows x 10,240 cols fp16, one core's slice"
    );
    let prob_pin_bytes = pins.iter().map(|&(_, _, bytes)| bytes).sum::<u64>();
    let pinned = chunk
        .lx_form(ChunkRowKind::OneRequest, 0, 0)
        .expect("the 40-page rung fits with no op-local slots")
        .pinned_bytes_per_core();
    assert_eq!(pinned, 1_179_648);
    let headroom = LX_PIN_END - LX_PIN_BASE - pinned - prob_pin_bytes;
    assert_eq!(headroom, 312_064);
    // At the boundary exactly, the form still admits (`lx_window_fits` is `<=`); one byte of
    // op-local slots past it refuses.
    assert!(
        chunk
            .lx_form(ChunkRowKind::OneRequest, headroom, prob_pin_bytes)
            .is_some()
    );
    assert!(
        chunk
            .lx_form(ChunkRowKind::OneRequest, headroom + 1, prob_pin_bytes)
            .is_none(),
        "one byte of op-local slots past the headroom must refuse the rung"
    );
    // ⭐ AND THE PROBABILITY-PART TERM ALONE REFUSES — the gap band the gather-only witness used
    // to admit (op-local classes of 312,192..=314,624 B, norm rows 2439..=2458 at 128 B a row):
    // a class the gathers leave room for but the probability parts do not is the HBM fall now,
    // never a mint-time panic.
    let gap = headroom + 128;
    assert!(
        chunk
            .lx_form(ChunkRowKind::OneRequest, gap, prob_pin_bytes)
            .is_none(),
        "a class the probability parts do not leave room for must refuse the rung"
    );
    assert!(
        chunk.lx_form(ChunkRowKind::OneRequest, gap, 0).is_some(),
        "the same class with the probability-part term dropped would admit it — the term the \
         witness must not forget"
    );
    // A rung the core-local pins alone refuse stays refused at every op-local term — the terms
    // compose, never trade.
    let refused = ChunkScratch::of_chunk(pool(), 48).unwrap();
    assert!(refused.lx_form(ChunkRowKind::OneRequest, 0, 0).is_none());
    assert!(
        refused
            .lx_form(ChunkRowKind::OneRequest, headroom, prob_pin_bytes)
            .is_none()
    );
}

/// How many of the chunk's Kᵗ gather copies stayed in LX, and how many fell to the HBM
/// scratch — the emission-level verdict of the chunk's `lx_form` admission (an admitted
/// rung's copies write core-local pins; a refused rung's copies all land in HBM).
fn lx_kt_copies(ops: &[(String, serde_json::Value)]) -> (usize, usize) {
    let mut lx = 0;
    let mut hbm = 0;
    for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_cgkt_")) {
        let body = &v["dscs_"][0][name];
        let out = body["labeledDs_"].as_array().unwrap().last().unwrap();
        if lx_only(out) {
            lx += 1;
        } else {
            hbm += 1;
        }
    }
    (lx, hbm)
}

/// ⭐⭐ THE DECLARE PASS'S PROMISE, FROM THE CHUNK'S SIDE OF IT (#645 fix (a)'s commit 2
/// reliance): a norm whose program FOLLOWS the chunk in program order is ALREADY in the
/// layout when the chunk's `lx_form` admission runs. The declare pass lowers every op of the
/// rung against the shared layout BEFORE any real pass (`lower_subtile_tape_to_ktir`: "this
/// declares, it does not emit"), so the chunk budgets against the FINAL op-local slot sum,
/// never a mid-walk projection. Pinned at the widest admitted rung — 40 pages, 1,179,648 B of
/// gather pins plus a 2,560 B probability-part pin — where a trailing norm's `[2438, 64]`
/// rinv class is EXACTLY the rung's true remaining window: the chunk still admits and keeps
/// Kᵗ in LX, while the same norm at 2439 rows (one stick of class past that window) refuses
/// the rung to HBM — the GRACEFUL fall, at the witness, of the geometry the allocator alone
/// would have panicked on. Nothing differs between the two legs but the trailing norm's class
/// size, so the refusal IS the proof the chunk's admission read the trailing norm's slot. If
/// the declare pass were ever skipped for a rung, the chunk would admit against too small a sum —
/// the companion test below shows that world fails LOUDLY, in the pin allocator's own window
/// assert, at build.
///
/// ⭐ THE PROBABILITY-PART TERM IS BUDGETED BY THE SAME ADMISSION (`chunk_prob_pins`, the one
/// enumeration the mint also declares): before it, the witness counted the gathers only, so
/// classes of 312,192..=314,624 B (rows 2439..=2458) were admitted and the allocator's `cesub`
/// mint then PANICKED past the window — found by this test, closed in the same PR. The 2439
/// leg below is the first row of that old gap band, and it now refuses to HBM here.
#[test]
fn a_trailing_norms_slot_still_bounds_the_chunk_that_precedes_it() {
    // Control: with no op-local slots declared, the 40-page rung keeps Kᵗ whole in LX —
    // forty (part, kv head) copies (5 parts × 8 kv heads), no HBM leg.
    assert_eq!(
        lx_kt_copies(&emit(40)),
        (40, 0),
        "the control: no op-local term"
    );

    // One leg of the boundary: the trailing norm's whole public lowering against the shared
    // layout FIRST — the declare pass's own action, throwaway symbols, ops dropped (only the
    // layout is wanted) — then the chunk whose admission must budget against it.
    fn declared_chunk(rows: u32) -> (BundleLayout, Vec<String>, Vec<(String, serde_json::Value)>) {
        let (mut l, names) = hd_layout();
        // The trailing norm's operands: its out and its eps const, placed beside the chunk's.
        for (tid, size) in [(901u32, 2439 * 576 * 2), (902, 64 * 2)] {
            l.ids
                .borrow_mut()
                .insert(PlaceId::Act(tid).to_string(), PlaceId::Act(tid));
            l.placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role: SegRole::Activation,
                    segment: 3,
                    bank: 0,
                    offset: 40 << 30,
                    size,
                    cover: ktir_superdsc::placement::BindCover::Full,
                },
            );
        }
        let mut declare_syms = 0i64;
        let _ = assemble_rmsnorm(
            RmsnormFacts {
                prefix: "tail",
                rows,
                cols: 576,
                x: &names[0],
                gamma: &names[1],
                out_id: PlaceId::Act(901),
                eps_const: "t902",
                ones_const: None,
            },
            &mut declare_syms,
            Some(&l),
        );
        let ops = emit_chunk(&l, &names, 40);
        (l, names, ops)
    }

    // AT the allocator's boundary: the class is exactly the window left past the chunk's
    // FULL core-local footprint (1,179,648 B of gathers + the 2,560 B probability-part pin),
    // and the chunk's admission still holds — Kᵗ stays in LX with the trailing norm's slot
    // fully present in the budget.
    let (l_at, names_at, ops_at) = declared_chunk(2438);
    assert_eq!(
        l_at.lx_op_local_slot_bytes(),
        312_064,
        "the [2438, 64] rinv class is exactly the window past the chunk's full core-local \
         footprint"
    );
    assert_eq!(
        lx_kt_copies(&ops_at),
        (40, 0),
        "the chunk kept Kᵗ in LX with the trailing norm's slot in the budget"
    );
    // The trailing norm's REAL pass, after the chunk: its pin is already declared, so the
    // pin table is unchanged — first-one-wins, the same address the declare pass fixed.
    let pins_before = l_at.lx_op_local_pins();
    let mut real_syms = 0i64;
    let tail = assemble_rmsnorm(
        RmsnormFacts {
            prefix: "tail",
            rows: 2438,
            cols: 576,
            x: &names_at[0],
            gamma: &names_at[1],
            out_id: PlaceId::Act(901),
            eps_const: "t902",
            ones_const: None,
        },
        &mut real_syms,
        Some(&l_at),
    );
    assert_eq!(tail.len(), 5, "the trailing norm's real pass lowers");
    assert_eq!(
        l_at.lx_op_local_pins(),
        pins_before,
        "the real pass finds the pin already declared — same name, same class, same slot"
    );

    // PAST the window the admission budgets (one stick of class past the rung's true remaining
    // window — the FIRST row of the gap band the gather-only witness used to admit): the
    // chunk's admission refuses the rung — every Kᵗ copy falls to the HBM scratch, and the
    // probability-part mint that follows is quiet (its 2,560 B pins against the empty
    // core-local region the refused gathers left). This leg differs from the admitted one
    // only in the trailing norm's row count, so the refusal is the admission reading THAT
    // term — the graceful HBM fall where the allocator alone would have panicked.
    let (l_past, _, ops_past) = declared_chunk(2439);
    assert_eq!(l_past.lx_op_local_slot_bytes(), 312_192);
    assert_eq!(
        lx_kt_copies(&ops_past),
        (0, 40),
        "Kᵗ whole in the HBM scratch — the rung the trailing norm refused"
    );
}

/// The declare-pass-SKIPPED world, made loud: with the chunk's core-local pins minted first
/// (the chunk admitted against an EMPTY op-local term), the trailing norm's own pin mint
/// meets them inside the frontend's LX window. At the boundary the mint still fits — the
/// allocator's guard reads the same `lx_window_fits` `<=` the witness does — and one stick of
/// class past it PANICS, at build. `#[should_panic]` is the honest shape here (the same class
/// as `reserved_tids`' overflow guard): the assert is a `cargo build` gate the bake hits, so
/// a rung whose declare pass was skipped cannot ship silently and collide on card.
#[test]
#[should_panic(expected = "op-local LX slots meet the core-local pins")]
fn a_norm_minted_after_the_chunk_fails_loudly_not_on_card() {
    // The boundary's quiet side, from the allocator's own guard: with the chunk's full
    // core-local footprint minted first (1,179,648 B of gathers + the 2,560 B
    // probability-part pin — the chunk admitted against an EMPTY op-local term, the
    // declare-pass-skipped world), the [2438, 64] class fits EXACTLY past it — no panic,
    // the same `lx_window_fits` `<=` the witness reads.
    let (l, names) = hd_layout();
    let _ = emit_chunk(&l, &names, 40);
    l.synth_lx_op_local(PlaceId::Act(901).synth(SynthRole::Rinv), &[2438, 64]);
    // One stick of class past it, in the same world on a fresh layout (so the overage is the
    // class's own 128 B, not the first leg's): the op-local slot meets the core-local pins
    // inside the window and the mint PANICS here — the loud failure that makes the declare
    // pass's ordering a build-time law rather than a card-time hope.
    let (l2, names2) = hd_layout();
    let _ = emit_chunk(&l2, &names2, 40);
    l2.synth_lx_op_local(PlaceId::Act(903).synth(SynthRole::Rinv), &[2439, 64]);
}

/// ⭐⭐⭐ EVERY LX COPY AND ITS CONSUMER MEET AT ONE PIN, read off the emitted descriptors at every rung.
/// Each Kᵗ copy writes one (part, kv head) tensor that exactly one score part reads, and each V copy one
/// (part, kv head, slab) tensor that exactly one PV reads. Every pin is one 16 KiB entry per core, every
/// pin lies inside the frontend's window, and no two tensors share one. What stays in HBM is exactly
/// what the form says: Kᵗ whole on a rung with a Kᵗ tail, and V's tail pages. That every rung emits at
/// all is the build-time proof each consumer divides its work as its copy did (the core-local read
/// law).
#[test]
fn every_hd128_lx_copy_and_its_consumer_meet_at_one_pin() {
    for &j in &ChunkScratch::LADDER {
        let lx = ChunkScratch::of_chunk(pool(), j)
            .unwrap()
            .lx_form(ChunkRowKind::OneRequest, 0, 0);
        let ops = emit(j);
        let mut all_pins = std::collections::BTreeSet::new();
        for (copy, consumer) in [("attn_cgkt_", "attn_csc_g"), ("attn_cgv_", "attn_cov_g")] {
            let keys = copy == "attn_cgkt_";
            let (mut copy_pins, mut hbm_copies) = (Vec::new(), 0usize);
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(copy)) {
                let body = &v["dscs_"][0][name];
                let out = body["labeledDs_"].as_array().unwrap().last().unwrap();
                if lx_only(out) {
                    copy_pins.push(pin(body, out).expect("one pin on every core"));
                } else {
                    hbm_copies += 1;
                }
            }
            let (mut read_pins, mut hbm_parts) = (Vec::new(), 0usize);
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(consumer)) {
                let body = &v["dscs_"][0][name];
                let kernel = body["labeledDs_"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|l| l["dsType_"] == "KERNEL")
                    .expect("the consumer's kernel");
                if lx_only(kernel) {
                    read_pins.push(pin(body, kernel).expect("one pin on every core"));
                } else {
                    hbm_parts += 1;
                }
            }
            let (lx_tensors, want_hbm_copies, want_hbm_parts) = match (lx, keys) {
                (None, _) => (
                    0,
                    j as usize,
                    if keys { NKVH } else { NKVH * NSLAB } as usize,
                ),
                (Some(lx), true) => match lx.keys() {
                    Some(k) => (k.tensors() as usize, 0, 0),
                    None => (0, j as usize, NKVH as usize),
                },
                (Some(lx), false) => {
                    let tail = lx.tail_pages();
                    (
                        lx.lx_tensors() as usize,
                        (tail * 4 * NKVH / 32) as usize,
                        if tail > 0 { (NKVH * NSLAB) as usize } else { 0 },
                    )
                }
            };
            assert_eq!(copy_pins.len(), lx_tensors, "{copy} at {j} page(s)");
            assert_eq!(read_pins.len(), lx_tensors, "{consumer} at {j} page(s)");
            assert_eq!(hbm_copies, want_hbm_copies, "{copy} HBM at {j} page(s)");
            assert_eq!(hbm_parts, want_hbm_parts, "{consumer} HBM at {j} page(s)");
            let (mut c, mut r) = (copy_pins.clone(), read_pins.clone());
            c.sort();
            r.sort();
            assert_eq!(
                c, r,
                "{copy} at {j} page(s): copy and consumer meet at one pin"
            );
            for p in copy_pins {
                assert!(
                    (LX_PIN_BASE..LX_PIN_END - ENTRY_BYTES + 1).contains(&p),
                    "{copy} at {j} page(s): pin {p} leaves the window"
                );
                assert!(all_pins.insert(p), "{j} page(s): pin {p} shared");
            }
        }
        // The pins are bump-allocated back to back, 16 KiB each, so their span IS the form's byte count.
        if let (Some(lx), Some(&lo), Some(&hi)) =
            (lx, all_pins.iter().next(), all_pins.iter().next_back())
        {
            assert_eq!(
                hi + ENTRY_BYTES - lo,
                lx.pinned_bytes_per_core(),
                "{j} page(s)"
            );
        }
    }
}

/// ⭐ THE HOST's TABLE IS THE COPIES' RUNS. Kᵗ's run is part-major when Kᵗ stays in LX (stick
/// `p * nkvh + h`, word `c` = slot stick `c` of kv head `h`'s part `p`), head-major when it does not. V's
/// run follows it, part-major by (part, kv head, slab): word `c` of a whole part's stick is run `c` of
/// that slab, V's source block `slab * 2 + c % 2` of page `16 p + c / 2`. Both runs cover every (kv
/// head, page, block) exactly once, and each copy reads its own stick of the table.
#[test]
fn the_hd128_lx_table_is_each_legs_part_major_run() {
    // The pool's own ceiling — 64 pages of budget, the ladder's own top (PAGES_MAX), so every
    // rung below names only pages it holds.
    let ceiling = ktir_superdsc::sdsc_abstract::gather_entry_ceiling(
        64,
        32 * pool().stick_block_bytes(),
        pool(),
    )
    .unwrap();
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0) else {
            continue;
        };
        let n = chunk.entries();
        assert_eq!(
            lx.index_base(),
            chunk.index_sticks(),
            "{j} page(s): V's run follows Kᵗ's"
        );
        // V's run, entry by entry.
        let mut seen = std::collections::BTreeSet::new();
        for e in 0..n {
            let (kvh, page, block) = lx.entry_coord(e);
            assert!(
                kvh < NKVH && page < j && block < 4,
                "{j} page(s), V entry {e}"
            );
            assert!(
                seen.insert((kvh, page, block)),
                "{j} page(s): V entry {e} repeats"
            );
            if e < lx.lx_tensors() * 32 {
                let (op, c) = (e / 32, e % 32);
                let (head_part, slab) = (op / NSLAB, op % NSLAB);
                let (part, h) = (head_part / NKVH, head_part % NKVH);
                assert_eq!(lx.lx_tensor(part, h, slab), op);
                assert_eq!(
                    (kvh, page, block),
                    (h, part * 16 + c / 2, slab * 2 + c % 2),
                    "{j} page(s), V entry {e}"
                );
            }
        }
        assert_eq!(seen.len(), n as usize);
        // Kᵗ's run, where Kᵗ stays in LX.
        if let Some(k) = lx.keys() {
            let mut seen = std::collections::BTreeSet::new();
            for e in 0..n {
                let (kvh, page, block) = k.entry_coord(e);
                let (op, c) = (e / 32, e % 32);
                let (part, h) = (op / NKVH, op % NKVH);
                assert_eq!(k.tensor(part, h), op);
                assert_eq!(
                    (kvh, page, block),
                    (h, part * 8 + c / 4, c % 4),
                    "{j}: Kᵗ entry {e}"
                );
                assert!(seen.insert((kvh, page, block)));
            }
            assert_eq!(seen.len(), n as usize);
            let sticks: Vec<(u32, u32)> = k.copies().map(|(i, cp)| (i, cp.stick())).collect();
            assert_eq!(
                sticks,
                (0..chunk.index_sticks())
                    .map(|s| (s, s))
                    .collect::<Vec<_>>()
            );
        }
        let v_sticks: Vec<u32> = lx.copies().map(|(_, cp)| cp.stick()).collect();
        assert_eq!(
            v_sticks,
            (chunk.index_sticks()..2 * chunk.index_sticks()).collect::<Vec<_>>(),
            "{j} page(s)"
        );
        // The host's table is Kᵗ's run, then V's. Both budget terms 0: the run law in isolation.
        let pages: Vec<i64> = (0..j as i64).collect();
        let table = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::OneRequest,
            &pages,
            ceiling,
            0,
            0,
        )
        .unwrap();
        assert_eq!(table.len() as u32, 2 * n, "{j} page(s)");
        let word = |(kvh, page, block): (u32, u32, u32)| {
            ktir_superdsc::sdsc_abstract::GatherEntry::of_page_block(
                ceiling,
                i64::from(page),
                chunk.source_block(kvh, block).unwrap(),
            )
            .unwrap()
        };
        for e in 0..n {
            let key = match lx.keys() {
                Some(k) => k.entry_coord(e),
                None => chunk.entry_coord(e),
            };
            assert_eq!(table[e as usize], word(key), "{j} page(s): Kᵗ word {e}");
            assert_eq!(
                table[(n + e) as usize],
                word(lx.entry_coord(e)),
                "{j} page(s): V word {e}"
            );
        }
    }
}

/// ⭐ A V TAIL PART READS ITS OWN SLAB OF THE TAIL SCRATCH. The tail scratch holds each kv head's tail
/// pages slab-major, laid out by V's run (`ChunkLx::entry_coord` past the whole parts), and each tail
/// PV's kernel corner is that kv head's slab (`ChunkLx::tail_corner`). Read here as one law: the
/// destination element where tail entry `t` lands is the corner of its (kv head, slab) plus its run's
/// offset inside that slab.
#[test]
fn a_hd128_value_tail_part_reads_its_own_slab_of_the_tail_scratch() {
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0) else {
            continue;
        };
        if lx.tail_pages() == 0 {
            assert!(lx.tail_corner(0, 0).is_none());
            continue;
        }
        let tail_slots = lx.tail_pages() * 256;
        let per_head = lx.tail_pages() * 4;
        let entry_elems = HD * 64;
        for t in 0..NKVH * per_head {
            let e = lx.lx_tensors() * 32 + t;
            let (kvh, page, block) = lx.entry_coord(e);
            let (slab, run) = (block / 2, block % 2);
            // Where the copy writes entry `t`: the tail scratch's flat entry space.
            let dest = t * entry_elems;
            // Where the PV of (kvh, slab) reads that run: its corner, plus the run's slots in the slab.
            let corner = lx.tail_corner(kvh, slab).unwrap();
            let tail_page = page - lx.lx_parts() * 16;
            let slot = tail_page * 256 + run * 128;
            assert!(slot < tail_slots);
            assert_eq!(
                dest,
                corner.elems() + slot * 64,
                "{j} page(s), tail entry {t}"
            );
        }
    }
}
