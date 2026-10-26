// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE EMISSION-SIDE PINS OF THE CARRIED-FACTS SUPERSESSION — the door's
//! ONE resolution, stamped on the binding, judged where it is stamped.
//!
//! Records 53 and 54 were the same defect twice: the read-before-write walk kept a
//! private copy of `resolve_seg_base`'s address resolution and the copy drifted from
//! the door's at a different spot each time. The cure is structural — ONE resolution,
//! traveling on the [`ArgBinding`] — and these pins hold the stamp side of that law
//! against REAL emissions (an `OpSpec` through `emit_sdsc_tiled` with a placed
//! [`BundleLayout`], the same door the model bake runs):
//!
//! * THE WORD-LENGTH STAMPS: `base_bytes` is the placement offset PLUS the
//!   `offset_elems · word_length` slice, at the operand's OWN format — fp8 at 1 B,
//!   fp16 at 2 B, fp32 at 4 B. The walk multiplies nothing; the number it reads is
//!   these numbers.
//! * THE MINT ARMS: a synth-minted name (`BundleLayout::synth`) resolves through the
//!   synth map at its assigned offset; a PLACED name carries its placement's OWN
//!   role/segment/bank; an LX-pinned name carries the `lx` flag and its pin; and a
//!   name that is BOTH placed and synth-declared resolves through the PLACEMENT —
//!   the arm order IS the join law, so a stale synth entry can never shadow a real
//!   placement.
//! * THE TOUCHED BLOCKS: the binding's `touched` states WHICH bytes at the device's
//!   own stick geometry — a full-width fp16 sweep merges to ONE block of
//!   `rows · 16384` B (stick run × stick count × rows, every stride equal to the run
//!   so everything merges), and an fp8 operand's stick is the WHOLE 128 B with every
//!   row merging onto it (`rows · 128` B).
//!
//! The `df`/`offset_elems` mutations on a built opspec mirror the production
//! precedent (`gather_copy_opspec` sets both on its value args after the build), so
//! these are the door's real inputs, not a parallel fixture vocabulary.

use ktir_superdsc::emit::{EmittedOp, emit_sdsc_tiled};
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec;
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BindCover, BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::superdsc_opspec::{Df, OpSpec, SdscFoldSet};

/// The A operand's placement base — the number every word-length pin adds its slice to.
const A_BASE: u64 = 0x1_0000;

/// One placed name: the id registered in the mint table AND its placement row — the
/// two halves of `resolve_seg_base`'s arm 1, never one without the other.
fn place(
    l: &mut BundleLayout,
    tid: u32,
    role: SegRole,
    segment: usize,
    bank: u32,
    offset: u64,
    size: u64,
) {
    l.ids
        .borrow_mut()
        .insert(PlaceId::Act(tid).to_string(), PlaceId::Act(tid));
    l.placements.insert(
        tid,
        TensorPlacement {
            tid,
            role,
            segment,
            bank,
            offset,
            size,
            cover: BindCover::Full,
        },
    );
}

/// The standard three-name layout: A under test at `A_BASE`, W and O as plain
/// activations far away (sized for the widest geometry any pin here emits).
fn layout_a_under_test() -> BundleLayout {
    let mut l = BundleLayout::default();
    place(
        &mut l,
        401,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        A_BASE,
        0x4_0000,
    );
    place(
        &mut l,
        402,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x10_0000,
        0x40_0000,
    );
    place(
        &mut l,
        403,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x50_0000,
        0x1_0000,
    );
    l
}

/// A real emission: the opspec through the door with the layout threaded, exactly as
/// the model walk calls it.
fn emit(op: &OpSpec, l: &BundleLayout) -> EmittedOp {
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let mut sid = 0i64;
    emit_sdsc_tiled("mm_o", op, &folds, &mut sid, Some(l))
        .unwrap_or_else(|e| panic!("the matmul must emit, refused: {e}"))
}

/// The stamped binding of the named buffer — the door's answer for THAT operand.
fn binding_of<'a>(e: &'a EmittedOp, name: &str) -> &'a ktir_superdsc::emit::ArgBinding {
    e.arg_bindings
        .iter()
        .find(|b| b.buffer == name)
        .unwrap_or_else(|| panic!("a stamped binding for {name}"))
}

/// ⭐ THE FP16 WORD STAMP: an operand at `offset_elems = 8` carries
/// `base_bytes = placement + 8·2 = placement + 16`. The slice is folded at the door
/// (`resolve_seg_base`'s `offset_elems · df.word_length()`), so the walk reads a byte
/// base and never a word arithmetic — this pin holds the fold at fp16's 2 B.
#[test]
fn an_fp16_offset_stamp_folds_two_bytes_per_element() {
    let mut op = matmul_opspec(2, 64, 128, 1, "t401", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    op.args[0].set_offset_elems(8);
    let l = layout_a_under_test();
    let e = emit(&op, &l);
    let b = binding_of(&e, "t401");
    assert_eq!(b.offset_elems, 8, "the element offset rides the binding");
    assert_eq!(
        b.base_bytes,
        A_BASE + 16,
        "fp16's word is 2 B: the door folds 8 elems · 2 B into the byte base"
    );
}

/// ⭐ THE FP32 WORD STAMP: the same element offset at fp32's 4 B folds to `+32`. The
/// operand's OWN format is the multiplier (set on the arg the way
/// `gather_copy_opspec` sets its value args), never a global or a name-suffix guess.
#[test]
fn an_fp32_offset_stamp_folds_four_bytes_per_element() {
    let mut op = matmul_opspec(2, 64, 128, 1, "t401", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    op.args[0].set_df(Df::Fp32);
    op.args[0].set_offset_elems(8);
    let l = layout_a_under_test();
    let e = emit(&op, &l);
    let b = binding_of(&e, "t401");
    assert_eq!(
        b.base_bytes,
        A_BASE + 32,
        "fp32's word is 4 B: 8 elems · 4 B folded at the door"
    );
}

/// ⭐ THE FP8 WORD STAMP: at fp8's 1 B, `offset_elems = 64` folds to `+64` — the
/// packed format's whole point is that the element and the byte coincide, and the
/// stamp must not reuse fp16's multiplier here.
#[test]
fn an_fp8_offset_stamp_folds_one_byte_per_element() {
    let mut op = matmul_opspec(2, 64, 128, 1, "t401", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    op.args[0].set_df(Df::Fp8);
    op.args[0].set_offset_elems(64);
    let l = layout_a_under_test();
    let e = emit(&op, &l);
    let b = binding_of(&e, "t401");
    assert_eq!(
        b.base_bytes,
        A_BASE + 64,
        "fp8's word is 1 B: 64 elems · 1 B folded at the door"
    );
}

/// ⭐ THE SYNTH MINT: a name declared through `BundleLayout::synth` (the t1414 class)
/// resolves through the synth map — Intermediate role and segment, bank 0, at the
/// offset the mint assigned. The walk's Written table keys on exactly these facts.
#[test]
fn a_synth_minted_name_resolves_through_the_synth_map() {
    let mut l = BundleLayout::default();
    l.synth(PlaceId::Act(1414), &[2, 128]);
    let synth_off = *l
        .synth
        .borrow()
        .map
        .get("t1414")
        .expect("the mint assigned t1414 an offset");
    place(
        &mut l,
        402,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x10_0000,
        0x4_0000,
    );
    place(
        &mut l,
        403,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x50_0000,
        0x1_0000,
    );
    let op = matmul_opspec(2, 64, 128, 1, "t1414", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    let e = emit(&op, &l);
    let b = binding_of(&e, "t1414");
    assert_eq!(b.role, SegRole::Intermediate, "the synth arm's role");
    assert_eq!(
        b.segment,
        SegRole::Intermediate.segment(),
        "the synth arm's segment"
    );
    assert_eq!(b.bank, 0, "the synth arm's bank");
    assert_eq!(b.base_bytes, synth_off, "the mint's own assigned offset");
    assert!(!b.lx, "a synth-minted HBM slot is not LX");
}

/// ⭐ THE PLACEMENT'S OWN FACTS: a placed name carries its placement row verbatim —
/// role, segment, bank, offset — whatever the operand slot is. The stamp is the
/// placement's answer, not a re-derivation from the operand's direction or shape.
#[test]
fn a_placed_name_carries_its_placements_facts() {
    let mut l = BundleLayout::default();
    // t401 placed as a WEIGHT in bank 3 — every field distinctive, so a stamp that
    // guessed from the operand slot (an activation-shaped input) fails loudly.
    place(
        &mut l,
        401,
        SegRole::Weight,
        SegRole::Weight.segment(),
        3,
        A_BASE,
        0x4_0000,
    );
    place(
        &mut l,
        402,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x10_0000,
        0x4_0000,
    );
    place(
        &mut l,
        403,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x50_0000,
        0x1_0000,
    );
    let op = matmul_opspec(2, 64, 128, 1, "t401", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    let e = emit(&op, &l);
    let b = binding_of(&e, "t401");
    assert_eq!(b.role, SegRole::Weight, "the placement's role, carried");
    assert_eq!(b.segment, SegRole::Weight.segment(), "its segment");
    assert_eq!(b.bank, 3, "its bank — the Written-table key's second half");
    assert_eq!(b.base_bytes, A_BASE, "its intra-segment offset");
    assert!(!b.lx);
}

/// ⭐ THE ARM-1 JOIN: a name that is BOTH placed and synth-declared resolves through
/// the PLACEMENT. `resolve_seg_base` consults the placement row (arm 1) before the
/// synth map (arm 2), so a stale synth entry — a name the lowering once minted and
/// the caller later placed for real — can never shadow the real address. The two
/// offsets differ here by construction; the stamp must take the placement's.
#[test]
fn a_name_both_placed_and_synth_declared_resolves_through_the_placement() {
    let mut l = BundleLayout::default();
    // Mint the synth declaration FIRST, then place the same id — however the two
    // declarations were ordered in walk history, the door's arm order decides.
    l.synth(PlaceId::Act(9), &[2, 128]);
    let synth_off = *l
        .synth
        .borrow()
        .map
        .get("t9")
        .expect("the mint assigned t9 an offset");
    assert_ne!(
        synth_off, A_BASE,
        "the fixture premise: the two arms answer at different offsets"
    );
    place(
        &mut l,
        9,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        A_BASE,
        0x4_0000,
    );
    place(
        &mut l,
        402,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x10_0000,
        0x4_0000,
    );
    place(
        &mut l,
        403,
        SegRole::Activation,
        SegRole::Activation.segment(),
        0,
        0x50_0000,
        0x1_0000,
    );
    let op = matmul_opspec(2, 64, 128, 1, "t9", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    let e = emit(&op, &l);
    let b = binding_of(&e, "t9");
    assert_eq!(
        b.base_bytes, A_BASE,
        "arm 1 wins: the placement's offset, never the synth map's {synth_off}"
    );
    assert_eq!(
        b.segment,
        SegRole::Activation.segment(),
        "the placement's segment, not the synth arm's Intermediate"
    );
}

/// ⭐ THE LX PIN, BOTH FAMILIES: a name declared LX-resident — OP-LOCAL
/// (`synth_lx_op_local`, the shared per-footprint slots bumping down) or CORE-LOCAL
/// (`synth_lx_core_local`, every core its own slice bumping up) — carries the `lx`
/// flag and its PIN as the byte base. The flag is the walk's skip rule (core-local
/// bytes are the alternation walk's domain, never the HBM coverage walk's), and the
/// pin is the door's whole answer for the address: segment facts come back as the
/// Intermediate spelling but no HBM byte is ever judged.
#[test]
fn an_lx_pinned_name_carries_the_lx_flag_and_its_pin() {
    for (tid, core_local) in [(31u32, false), (32u32, true)] {
        let name = PlaceId::Act(tid).to_string();
        let mut l = BundleLayout::default();
        if core_local {
            l.synth_lx_core_local(PlaceId::Act(tid), 4096);
        } else {
            l.synth_lx_op_local(PlaceId::Act(tid), &[2, 64]);
        }
        let (pin, _fp) = l.lx_pin(&name).expect("the declared pin");
        place(
            &mut l,
            402,
            SegRole::Activation,
            SegRole::Activation.segment(),
            0,
            0x10_0000,
            0x4_0000,
        );
        place(
            &mut l,
            403,
            SegRole::Activation,
            SegRole::Activation.segment(),
            0,
            0x50_0000,
            0x1_0000,
        );
        let op = matmul_opspec(2, 64, 64, 1, &name, "t402", "t403")
            .expect("the small matmul is a legal OpSpec");
        let e = emit(&op, &l);
        let b = binding_of(&e, &name);
        assert!(
            b.lx,
            "{name}: the carried LX verdict — the walk's skip rule"
        );
        assert_eq!(
            b.base_bytes, pin,
            "{name}: the pin IS the byte base — no HBM offset is added"
        );
        assert_eq!(
            b.segment,
            SegRole::Intermediate.segment(),
            "{name}: the LX arm's segment spelling"
        );
    }
}

/// ⭐ THE FULL-WIDTH FP16 SWEEP — ONE MERGED BLOCK (touched pin A). A `[m, 8192]`
/// fp16 operand sweeps its stick axis whole: run = 64 lanes · 2 B = 128 B, 128
/// sticks at stride 128 B == run → MERGE (16,384 B), then m rows at stride 16,384 ==
/// run → MERGE again. The binding states ONE block of `m · 16,384` B with an empty
/// outer list — the shape the walk's union coverage credits as a single interval.
#[test]
fn a_full_width_fp16_sweep_stamps_one_merged_block() {
    let op = matmul_opspec(2, 64, 8192, 1, "t401", "t402", "t403")
        .expect("the wide-K matmul is a legal OpSpec");
    let l = layout_a_under_test();
    let e = emit(&op, &l);
    let t = &binding_of(&e, "t401").touched;
    assert_eq!(
        t.run_bytes,
        2 * 16384,
        "stick run 128 B × 128 sticks × 2 rows, all merged: one block of m·16,384 B"
    );
    assert!(
        t.outer.is_empty(),
        "every stride equals the run — nothing strided"
    );
    assert_eq!(t.offsets(), vec![0], "one block, at the base");
}

/// ⭐ THE FP8 STICK — ONE WHOLE 128-B STICK PER ROW, MERGED (touched pin C). At fp8
/// the stick IS 128 B (128 lanes · 1 B): a `[m, 128]` operand sweeps exactly one
/// stick per row, and the rows sit at stride 128 B == run → MERGE. The binding
/// states ONE block of `m · 128` B.
#[test]
fn an_fp8_sweep_stamps_one_whole_stick_per_row_merged() {
    let mut op = matmul_opspec(2, 64, 128, 1, "t401", "t402", "t403")
        .expect("the small matmul is a legal OpSpec");
    op.args[0].set_df(Df::Fp8);
    let l = layout_a_under_test();
    let e = emit(&op, &l);
    let t = &binding_of(&e, "t401").touched;
    assert_eq!(
        t.run_bytes,
        2 * 128,
        "the fp8 stick is the whole 128 B and both rows merge onto it: m·128 B"
    );
    assert!(t.outer.is_empty());
    assert_eq!(t.offsets(), vec![0]);
}
