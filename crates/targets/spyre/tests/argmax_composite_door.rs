// SPDX-License-Identifier: Apache-2.0
//! THE COMPOSITE ARGMAX DOOR — the thin-arm pin for #603 PR3a's wiring.
//!
//! `Program::Argmax`'s door arm resolves the three boundary names the walk-minted
//! program states ([`mint_argmax_node`]'s binding convention: `[logits, table,
//! row]`), names the two resident constants by their reserved tids, declares the
//! seven scratch synthetics at their full footprints, and hands the geometry to
//! [`try_assemble_argmax`] — the SAME assembly the probed chain came from
//! (`zz_the_argmax_composite_probes.rs`, in the mint's own crate, pins that side
//! at granite's own geometry).
//!
//! What is pinned HERE (the door's side):
//!   • the descriptors the door emits from a hand-built node are BYTE-IDENTICAL to
//!     the direct assembly's at the same geometry, over the SAME layout — the arm
//!     is provably thin, so the KTIR program and the descriptor group stay two
//!     expressions of ONE computation (the parity gate's structural half; the
//!     numeric half executes the program in the emu and lands with the walk);
//!   • the geometry is READ FROM THE PROGRAM's own views (the regions), never
//!     threaded beside it: a node minted at a different vocab emits that vocab's
//!     chain — pinned differentially, two geometries, plus the reduce's own `N_`
//!     stating each;
//!   • the door DECLARED the seven scratch synthetics — keyed off the ROW tid, at
//!     the chain's own formats — so the layout the declare pass shares is complete
//!     the moment the door has run (the seg0-overrun ordering law's precondition);
//!   • the ramp and the scalar V are named by their RESERVED tids (`argmax_const`),
//!     not minted as synthetics — a bundle without the composite never places or
//!     stages them.
//!
//! The geometry is small and stick-legal (vocab/hidden 128 = two fp16 sticks, and
//! 192 = three for the differential leg): the door reads the geometry from the
//! node's own views, so a small faithful node pins the arm as well as granite's
//! 49152-wide one (which the probes hold on the assembly side).

use ktir_superdsc::emit::argmax::{
    ArgmaxGeom, ArgmaxSlots, TiedLmHeadEmbedding, mint_argmax_node, try_assemble_argmax,
};
use ktir_superdsc::ktir_node::KtirNode;
use ktir_superdsc::place::{PlaceId, SynthRole, act_name};
use ktir_superdsc::placement::{
    BundleLayout, SegRole, TensorPlacement, align128, syn, synth_footprint_bytes,
};
use ktir_superdsc::reserved_tids::{argmax_ramp_tid, argmax_v_tid};
use ktir_superdsc::superdsc_opspec::{DataFormat, Df, Fp16};
use scratchy_target_spyre::ktir_superdsc_door;

/// Stick-legal small geometry — two fp16 sticks each way; 192 (three sticks) is
/// the differential leg's vocab.
const VOCAB: u32 = 128;
const WIDER_VOCAB: u32 = 192;
const HIDDEN: u32 = 128;
/// The three boundary tids. Distinct, as a real bundle's are; the table is a
/// weight id, the row an activation-source id — the door never asks.
const LOGITS_TID: u32 = 2;
const TABLE_TID: u32 = 1;
const ROW_TID: u32 = 3;

/// The eight links, in dependency order — the door's own naming.
const CHAIN: [&str; 8] = [
    "argmax_max16",
    "argmax_eq",
    "argmax_cvt",
    "argmax_mul",
    "argmax_max32",
    "argmax_sub",
    "argmax_cvt_idx",
    "argmax_gather",
];

/// The vendor kernel each link resolves to — the probes' own chain, so the door's
/// emission is THAT chain, not a same-shaped neighbor.
const CHAIN_FUNCS: [&str; 8] = [
    "max",
    "equal",
    "dl16tofp32",
    "mul",
    "max",
    "sub",
    "fp32toint32",
    "identity",
];

fn tied() -> TiedLmHeadEmbedding {
    TiedLmHeadEmbedding::from_tied_config(true).expect("granite ties")
}

/// The node, hand-built at the mint's own contract.
fn node_at(vocab: u32) -> KtirNode {
    mint_argmax_node(vocab, HIDDEN, LOGITS_TID, TABLE_TID, ROW_TID, tied())
}

/// The layout the declare pass would share: every boundary tensor and both
/// reserved-tid constants REGISTERED (name → identity, via `syn`) and PLACED
/// (segment, offset, true footprint), which is what the walk's layout pass owns
/// before any door runs — the router-const law: THE DOOR NAMES, THE LAYOUT
/// PLACES. The door itself never places a boundary tensor or a constant; a test
/// that skipped this setup would be testing the door against a layout no real
/// bundle can present.
fn layout_ready_for(vocab: u32) -> BundleLayout {
    let mut l = BundleLayout::default();
    let mut offs = [0u64; 7];
    let mut place = |tid: u32, role: SegRole, dims: &[u32], df: Df| {
        let size = synth_footprint_bytes(dims, df);
        let seg = role.segment();
        let offset = offs[seg];
        offs[seg] += align128(size.max(1));
        syn(Some(&l), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role,
                segment: seg,
                bank: 0,
                offset,
                size,
            },
        );
    };
    place(LOGITS_TID, SegRole::Activation, &[1, vocab], Df::Fp16);
    place(TABLE_TID, SegRole::Weight, &[vocab, HIDDEN], Df::Fp16);
    place(ROW_TID, SegRole::Activation, &[1, HIDDEN], Df::Fp16);
    place(
        argmax_ramp_tid(),
        SegRole::Activation,
        &[1, vocab],
        Df::Fp32,
    );
    place(
        argmax_v_tid(),
        SegRole::Activation,
        &[1, Df::Fp32.elems_per_stick()],
        Df::Fp32,
    );
    l
}

/// The door's emission from the hand-built node, with the layout the declare
/// pass would share. Rendered to strings up front — a failed equality then
/// prints the differing descriptor, not an unbounded JSON tree.
fn through_the_door_at(vocab: u32) -> (Vec<String>, Vec<String>, BundleLayout) {
    let layout = layout_ready_for(vocab);
    let mut syms = 0i64;
    let mut quantized = std::collections::HashSet::new();
    let ops = ktir_superdsc_door::lower(
        &node_at(vocab),
        &mut syms,
        Some(&layout),
        &mut quantized,
        None,
    )
    .unwrap_or_else(|e| panic!("the argmax door: {e:?}"));
    let names = ops.iter().map(|o| o.op_name.clone()).collect();
    let rendered = ops
        .iter()
        .map(|o| serde_json::to_string(&o.op).expect("serializable"))
        .collect();
    (rendered, names, layout)
}

/// The direct assembly at the same geometry, SHARING the door's layout: both
/// sides resolve the same names (the boundary tids, the reserved-tid constants,
/// the row-keyed synthetics), so the shared declarations are idempotent and any
/// divergence in the rendered descriptors is a divergence in the arm, not in the
/// allocator. The symbol base restarts at 0, as the door's did.
fn direct_at(vocab: u32, layout: &BundleLayout) -> Vec<String> {
    let lg = act_name(LOGITS_TID);
    let table = act_name(TABLE_TID);
    let row = act_name(ROW_TID);
    let ramp = act_name(ktir_superdsc::reserved_tids::argmax_ramp_tid());
    let v = act_name(ktir_superdsc::reserved_tids::argmax_v_tid());
    let row_id = PlaceId::Act(ROW_TID);
    let m = row_id.synth(SynthRole::AmMax).to_string();
    let mask = row_id.synth(SynthRole::AmMask).to_string();
    let mask32 = row_id.synth(SynthRole::AmMask32).to_string();
    let w = row_id.synth(SynthRole::AmW).to_string();
    let j = row_id.synth(SynthRole::AmJ).to_string();
    let idx32 = row_id.synth(SynthRole::AmIdx32).to_string();
    let idx_u32 = row_id.synth(SynthRole::AmIdxU32).to_string();
    let s = ArgmaxSlots {
        logits: &lg,
        m: &m,
        mask: &mask,
        mask32: &mask32,
        ramp: &ramp,
        v_const: &v,
        w: &w,
        j: &j,
        idx32: &idx32,
        idx_u32: &idx_u32,
        table: &table,
        row: &row,
    };
    let mut syms = 0i64;
    try_assemble_argmax(
        ArgmaxGeom::k1(vocab, HIDDEN),
        &s,
        tied(),
        &mut syms,
        Some(layout),
    )
    .unwrap_or_else(|e| panic!("the direct mint: {e}"))
    .into_iter()
    .map(|e| serde_json::to_string(&e.op).expect("serializable"))
    .collect()
}

/// The emitted dsc body (`dscs_[0]`'s first value — the op's own name is its
/// key) — the probes' own read pattern.
fn body(d: &str) -> serde_json::Value {
    let v: serde_json::Value = serde_json::from_str(d).expect("re-parse the rendered op");
    v.get("dscs_")
        .and_then(|d| d.as_array())
        .expect("dscs_ array")
        .first()
        .expect("one dsc")
        .as_object()
        .expect("dsc object")
        .values()
        .next()
        .expect("the op's own body")
        .clone()
}

/// The `opFuncName` row of one rendered op — the field the vendor's selection
/// table keys on.
fn op_func_name(d: &str) -> String {
    body(d)
        .pointer("/computeOp_/0/opFuncName")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("no opFuncName in {d}"))
        .to_string()
}

/// The arm is thin: the door's descriptors are byte-identical to the direct
/// assembly's — same eight links, same vendor kernels, same rendered bodies.
/// The KTIR program and the descriptor group are two expressions of ONE
/// computation; this is the structural half of that pin.
#[test]
fn the_door_mints_the_probes_chain_byte_identically() {
    let (door, names, layout) = through_the_door_at(VOCAB);
    let direct = direct_at(VOCAB, &layout);
    assert_eq!(
        names, CHAIN,
        "the door emits the chain's eight links in dependency order"
    );
    let funcs: Vec<String> = door.iter().map(|d| op_func_name(d)).collect();
    assert_eq!(
        funcs, CHAIN_FUNCS,
        "the door's links resolve the probes' own vendor kernels"
    );
    assert_eq!(door.len(), direct.len(), "eight ops both ways");
    for (i, (d, s)) in door.iter().zip(direct.iter()).enumerate() {
        assert_eq!(
            d, s,
            "link {i} differs between the door and the direct mint"
        );
    }
}

/// The geometry is the program's own: the door reads vocab off the node's views,
/// so a node minted at a wider vocab emits that vocab's chain. Pinned three
/// ways — the reduce's own `N_` stating each vocab, the two chains differing,
/// and the wider geometry's door staying byte-identical to ITS direct mint (no
/// cached geometry in the arm).
#[test]
fn the_geometry_comes_from_the_programs_own_views() {
    let (narrow, _, _) = through_the_door_at(VOCAB);
    let (wide, _, wide_layout) = through_the_door_at(WIDER_VOCAB);
    let out_of = |rendered: &str| {
        body(rendered)
            .get("N_")
            .and_then(|n| n.get("out_"))
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("out_ in N_ of {rendered}"))
    };
    assert_eq!(
        out_of(&narrow[0]),
        u64::from(VOCAB),
        "the max's stick axis is the minted vocab"
    );
    assert_eq!(
        out_of(&wide[0]),
        u64::from(WIDER_VOCAB),
        "the wider node's max states the wider vocab"
    );
    assert_ne!(
        narrow, wide,
        "two geometries must emit two chains — a cached geometry would emit one"
    );
    let wide_direct = direct_at(WIDER_VOCAB, &wide_layout);
    for (i, (d, s)) in wide.iter().zip(wide_direct.iter()).enumerate() {
        assert_eq!(d, s, "link {i} differs at the wider geometry");
    }
}

/// The door declared the seven scratch synthetics — keyed off the ROW tid, at
/// the chain's own formats — so the shared layout is complete once the door has
/// run (the declare pass's ordering law). The constants are NOT among them:
/// they are reserved-tid placements the layout owns, so a bundle without the
/// composite never stages them.
#[test]
fn the_door_declares_the_scratch_at_the_chains_own_formats() {
    let (_, _, layout) = through_the_door_at(VOCAB);
    let row_id = PlaceId::Act(ROW_TID);
    let fp = |role: SynthRole, dims: &[u32], df: Df| {
        let id = row_id.synth(role);
        let name = id.to_string();
        let want = align128(synth_footprint_bytes(dims, df).max(1));
        let got = layout
            .synth
            .borrow()
            .sizes
            .get(&name)
            .unwrap_or_else(|| panic!("'{name}' was not declared by the door"))
            .to_owned();
        assert_eq!(got, want, "'{name}' footprint");
        assert_eq!(layout.id_of(&name), Some(id), "'{name}' identity");
    };
    fp(SynthRole::AmMax, &[1, Fp16::ELEMS_PER_STICK], Df::Fp16);
    fp(SynthRole::AmMask, &[1, VOCAB], Df::Fp16);
    fp(SynthRole::AmMask32, &[1, VOCAB], Df::Fp32);
    fp(SynthRole::AmW, &[1, VOCAB], Df::Fp32);
    fp(SynthRole::AmJ, &[1, Df::Fp32.elems_per_stick()], Df::Fp32);
    fp(SynthRole::AmIdx32, &[1, Fp16::ELEMS_PER_STICK], Df::Fp32);
    fp(
        SynthRole::AmIdxU32,
        &[1, Fp16::ELEMS_PER_STICK],
        Df::SenUint32,
    );
    let ramp = act_name(argmax_ramp_tid());
    assert!(
        !layout.synth.borrow().map.contains_key(&ramp),
        "the ramp is a reserved-tid const, not a row-keyed synthetic"
    );
}

/// The mint's binding convention, read back through the door's own region walk:
/// three parameters, three bindings, in the order `[logits, table, row]` — and
/// the ROW is the one the program's store writes (the `written_args` chain),
/// which is what makes the emulator read it back as the output.
#[test]
fn the_mints_binding_order_is_logits_table_row() {
    let k = node_at(VOCAB);
    assert_eq!(k.func.arguments.len(), 3, "three pointer parameters");
    let tids: Vec<u32> = k.bindings.iter().map(|b| b.get()).collect();
    assert_eq!(tids, vec![LOGITS_TID, TABLE_TID, ROW_TID]);
    let regions = ktir_superdsc::emit::lower_ktir_to_superdsc::regions(&k)
        .unwrap_or_else(|e| panic!("regions: {e:?}"));
    assert_eq!(regions.len(), 3);
    assert!(!regions[0].is_out, "the logits are read, not written");
    assert!(!regions[1].is_out, "the table is read, not written");
    assert!(regions[2].is_out, "the row is the program's output");
    assert_eq!(regions[0].v_cols, VOCAB, "logits view states the vocab");
    assert_eq!(
        (regions[1].v_rows, regions[1].v_cols),
        (VOCAB, HIDDEN),
        "table view states [vocab, hidden]"
    );
    assert_eq!(
        (regions[2].v_rows, regions[2].v_cols),
        (1, HIDDEN),
        "row view states [1, hidden]"
    );
}
