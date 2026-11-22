// SPDX-License-Identifier: Apache-2.0
//! Lower a rerolled [`SubtileIR`] tape to a Spyre bundle. The producer mints one KTIR program per
//! node ([`crate::lower_subtile_tape_to_ktir`]), [`compute_bundle_layout`] places every buffer,
//! [`crate::superdsc_lower`] lowers each program to SuperDSC against the placement, and
//! [`emit_bundle`] partitions the descriptors into dxp launch groups and hands them to the bake
//! queue ([`crate::superdsc_bake`]).

pub use crate::lower_subtile_tape_to_ktir::{
    BundleWiring, NodeArgs, graph_wiring, lower_subtile_tape_to_ktir,
};
pub use crate::placement::{
    BundleLayout, KvBank, RetileDescriptor, SegRole, TensorPlacement, WEIGHT_SPILL_SEGS,
    WeightTail, align128,
};
use ktir_core::attrkey::AttrKey;
use ktir_core::ir::Attr;
use ktir_core::opkind::OpKind;
/// The baked bundle's types: this lowering's output is a `bundle::BundleCode`.
pub use scratchy_spyre_bundle as bundle;
pub use scratchy_spyre_bundle::program::ActiveCap;
pub use scratchy_spyre_bundle::reserved_tids::{
    ATTN_MASK_TID, LAST_HIDDEN_TID, PAGED_WRITE_ROWS_TID, RESERVED_REGIONS,
    RESERVED_REGIONS_ARE_DISJOINT, SENTINELS_ARE_INSIDE_THEIR_REGION, TidRegion, reserved_region,
};
pub use scratchy_spyre_bundle::rung::DxGroupCeiling;
use scratchy_subtile::subtile_ir::{RopeForm, SubOp, SubtileIR};
use scratchy_triton_splice::EmittedOp;
use spyre_superdsc::format::DataFormat;
use std::collections::BTreeMap;

/// Bytes in one stick: the generation's flit.
pub const STICK_BYTES: u64 =
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES;

/// fp16 elements in one stick.
pub const FP16_ELEMS_PER_STICK: u32 = scratchy_spyre_bundle::rung::FP16_LANES;

/// Why a bundle did not bake — a `cargo build` failure naming the defect, never a device fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperDscError(pub String);

impl std::fmt::Display for SuperDscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SuperDSC lowering: {}", self.0)
    }
}

impl std::error::Error for SuperDscError {}

/// ⛔⛔⛔ THE ADDRESS AUDIT — A LAYOUT THAT COULD FAULT MAY NOT COMPILE.
///
/// Everything up to the SuperDSC emission runs in the `#[forward]` procmacro, so every placement,
/// every segment extent and therefore every DMA window this bundle will ever generate is a
/// COMPILE-TIME CONSTANT. There is no reason for the card to be the thing that discovers a bad
/// address: it discovers them as `0xa35e RAS::PCI::BusFence`, which arrives with no host stack, no
/// statement of which transfer caused it, and a bus that needs resetting.
///
/// I had been asserting these at CB-build time, on the card, against numbers the bake computed —
/// checking at runtime a fact that was constant. This is the same audit, moved to where the
/// constants are, where the failure is a `cargo build` error naming the tensor.
///
/// Each check is a fault mechanism, not a tidiness rule:
///
///  * **Flit alignment.** A device address is written in FLITS (`>> 7`). An offset that is not a
///    multiple of 128 truncates DOWNWARD, so the transfer addresses a different location than its
///    length field describes. IBM's own validator for this is commented out in `qg.h`.
///  * **Inside its segment.** A placement running past `segment_bytes[seg]` addresses memory the
///    segment does not own — I1's inner half, which the device checks only on one path.
///  * **No overlap** is not checked here: synthetics are disjoint by construction
///    (`SynthAlloc::take`), and colored activations share slots by design.
///  * **Non-empty.** A zero-length field is read by the device as `1 << 27` flits = 16 GiB, in both
///    `handleHostDMA` and `handleXLATentry`. Zero does not mean nothing.
///  * **Allocatable.** A segment is ONE `FlexAllocator` allocation, served from ONE region, so it
///    cannot exceed [`bundle::MAX_SEGMENT_BYTES`] — see that constant for the granite-3.1-8b-fp16
///    weight segment this catches.
fn audit_layout_addresses(
    places: &[bundle::Placement],
    segment_bytes: &[u64; 7],
    // Extents of the weight and KV segments' banks 1..N (empty ⇒ one bank). A banked placement is
    // bounded by ITS OWN bank, not by the segment total — see [`bundle::Placement::bank`].
    weight_bank_bytes: &[u64],
    kv_bank_bytes: &[u64],
    fp: &str,
) {
    let mut fail: Vec<String> = Vec::new();
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    // The extent a placement is checked against: its bank's, which for bank 0 (everything in an
    // unbanked bundle) is the segment's own total.
    let extent = |seg: usize, bank: u32| -> Option<u64> {
        let extra = (bank as usize).checked_sub(1);
        match extra {
            None => segment_bytes.get(seg).copied(),
            Some(b) if seg == w_seg => weight_bank_bytes.get(b).copied(),
            Some(b) if seg == kv_seg => kv_bank_bytes.get(b).copied(),
            Some(_) => None,
        }
    };

    // ── ALLOCATABLE: each segment is one device region, and a region is capped ──
    //
    // ⛔ THIS WAS THE CARD'S JOB AND THE CARD IS BAD AT IT. `prepare` allocates a segment with
    // `DevAddr::alloc(segment_bytes[i])`, and flex serves that from a single region: over the cap it
    // returns an OOM whose numbers deny each other (`requested_bytes` well under `free_space_bytes`),
    // because the free bytes are in the regions the request cannot reach. Nothing in that message
    // names a segment, a tensor, or a model. All of it is decided here, by constants.
    // A BANK is a device region like any other, so it is bound by the same cap; checking the banks
    // here is what stops banking from trading one over-size allocation for another.
    for (what, banks) in [("weight", weight_bank_bytes), ("KV", kv_bank_bytes)] {
        for (b, &bytes) in banks.iter().enumerate() {
            if bytes > bundle::MAX_SEGMENT_BYTES {
                fail.push(format!(
                    "{what} bank {} packs {bytes} B, {} B past the {} B a single device region can \
                     hold",
                    b + 1,
                    bytes - bundle::MAX_SEGMENT_BYTES,
                    bundle::MAX_SEGMENT_BYTES,
                ));
            }
        }
    }
    for (seg, &bytes) in segment_bytes.iter().enumerate() {
        if bytes > bundle::MAX_SEGMENT_BYTES {
            let (biggest, share) = places
                .iter()
                .filter(|p| p.segment as usize == seg)
                .fold((0u64, 0u64), |(mx, sum), p| (mx.max(p.size), sum + p.size));
            fail.push(format!(
                "seg{seg} packs {bytes} B, which is {} B past the {} B a single device region — and \
                 therefore a single segment — can hold. A segment is ONE allocation; the other \
                 regions' free space cannot be reached from it. ({} placement(s) totalling {share} \
                 B, largest {biggest} B.) Shrink the segment: split it, or drop the device padding \
                 that grew it",
                bytes - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
                places.iter().filter(|p| p.segment as usize == seg).count(),
            ));
        }
    }

    for p in places {
        let seg = p.segment as usize;
        if seg >= segment_bytes.len() {
            fail.push(format!("{}: segment {seg} does not exist", p.id));
            continue;
        }
        if p.offset % 128 != 0 {
            fail.push(format!(
                "{} is at offset {} in seg{seg}, which is not 128-byte aligned — its device \
                 address is written in flits and would truncate to {}",
                p.id,
                p.offset,
                (p.offset / 128) * 128
            ));
        }
        if p.size == 0 {
            fail.push(format!(
                "{} has size 0 in seg{seg} — the device reads a zero length as 2^27 flits (16 GiB)",
                p.id
            ));
        }
        let Some(cap) = extent(seg, p.bank) else {
            fail.push(format!(
                "{} is in seg{seg} bank {}, which this bundle does not have ({} weight bank(s), \
                 {} KV bank(s))",
                p.id,
                p.bank,
                1 + weight_bank_bytes.len(),
                1 + kv_bank_bytes.len(),
            ));
            continue;
        };
        if p.offset + p.size > cap {
            fail.push(format!(
                "{} spans [{}, {}) of seg{seg} bank {}, which is only {cap} B",
                p.id,
                p.offset,
                p.offset + p.size,
                p.bank,
            ));
        }
    }

    // ── overlap: not checked here ──
    //
    // Colored activations overlap ON PURPOSE (two tensors whose lifetimes are disjoint share a slot:
    // the slot-coloring pass's optimisation, a fact this function does not have). Synthetics cannot:
    // `SynthAlloc::take` is their one writer and hands out disjoint ranges by construction.

    if !fail.is_empty() {
        let shown = fail.len().min(12);
        panic!(
            "\n⛔ [superdsc-layout] bundle {fp}: {} ADDRESS DEFECT(S). Refusing to bake.\n\n{}\n{}\n             Every number here is a compile-time constant, so this cannot be left for the card to \
             find — it finds them as `0xa35e RAS::PCI::BusFence`, with no host stack and no \
             statement of which transfer was responsible.\n",
            fail.len(),
            fail[..shown]
                .iter()
                .map(|f| format!("  • {f}"))
                .collect::<Vec<_>>()
                .join("\n"),
            if fail.len() > shown {
                format!("  … and {} more\n", fail.len() - shown)
            } else {
                String::new()
            },
        );
    }
}

/// ⛔⛔⛔ ONE SESSION STAGES ONE LAYOUT — every ladder rung must place every host-staged ACTIVATION
/// exactly where the ceiling does, the gather's own index excepted.
///
/// A decode session's host binds each step at the CEILING bundle's placements, whichever rung body it
/// then launches, while that body's descriptors were emitted against its OWN layout. A tensor the two
/// place apart — a staged input, or one of the constant sticks the session stages once at load — is
/// read by the body from bytes the host never wrote there: a clean bake, no fault, a wrong value. The
/// macro calls this on every rung it bakes, so a disagreement is a build failure.
pub fn ladder_stages_one_layout(ceiling: &BundleLayout, rung: &BundleLayout) -> Result<(), String> {
    let consts =
        |l: &BundleLayout| -> std::collections::BTreeMap<u32, spyre_superdsc::graph::ConstStick> {
            l.stick_consts.iter().copied().collect()
        };
    let (cc, rc) = (consts(ceiling), consts(rung));
    // A constant both place sits at one address holding one value.
    if let Some((tid, (a, b))) = cc
        .iter()
        .filter_map(|(t, a)| rc.get(t).map(|b| (t, (a, b))))
        .find(|(_, (a, b))| a != b)
    {
        return Err(format!(
            "a ladder rung places constant {b:?} at t{tid}, where its ceiling places {a:?} — the \
             session stages both at one address"
        ));
    }
    // The session stages its ceiling's constants: a rung reads only those.
    if let Some((tid, c)) = rc.iter().find(|(t, _)| !cc.contains_key(t)) {
        return Err(format!(
            "a ladder rung reads constant {c:?} at t{tid}, which its ceiling does not place — the \
             session stages the ceiling's constants alone"
        ));
    }
    let staged = |l: &BundleLayout,
                  own: &std::collections::BTreeMap<u32, _>|
     -> Vec<(u32, usize, u64, u64)> {
        l.placements
            .values()
            .filter(|p| matches!(p.role, SegRole::Activation) && !own.contains_key(&p.tid))
            .map(|p| (p.tid, p.segment, p.offset, p.size))
            .collect()
    };
    let (c, r) = (staged(ceiling, &cc), staged(rung, &rc));
    if c == r
        && ceiling.segment_bytes[SegRole::Activation.segment()]
            == rung.segment_bytes[SegRole::Activation.segment()]
    {
        return Ok(());
    }
    let first = c
        .iter()
        .zip(&r)
        .find(|(a, b)| a != b)
        .map(|(a, b)| {
            format!("first difference (tid, seg, offset, size): ceiling {a:?} vs rung {b:?}")
        })
        .unwrap_or_else(|| format!("{} vs {} staged activation(s)", c.len(), r.len()));
    Err(format!(
        "a ladder rung places a host-staged activation apart from its ceiling — the host stages the \
         ceiling's placements, so the rung would read bytes nobody wrote: {first}"
    ))
}

/// ⭐⭐⭐ ONE MODEL PACKS ONE RESIDENT WEIGHT SEGMENT — this bundle's packed size for it, as the
/// runtime's own alias guard reads it: bank 0's bytes ([`BundleLayout::segment_bytes`] at the
/// weight segment) plus the extra banks ([`BundleLayout::weight_bank_bytes`]).
///
/// The sessions a model serves share the weight segment OUTRIGHT (`Executor::alias_seg_from`
/// points a borrowing session's descriptor at the owner's region), and that guard refuses on a
/// packed-size mismatch BEFORE it ever compares a placement — so a bundle that reserves bytes a
/// sibling does not is a card-load refusal waiting to happen, whatever the bake thought it was
/// doing. This is the record the expansion's parity ledger ([`weight_segment_parity`]) holds one
/// of per baked bundle; factored so the macro never spells a segment index. The SHIPPED side's
/// twin — `BundleLayout::weight_seg_record` in `scratchy-spyre-bundle` — reads the same numbers
/// off the artifact the runtime loads, and the model-side test compares that one.
pub fn weight_seg_record(layout: &BundleLayout) -> (u64, Vec<u64>) {
    (
        layout.segment_bytes[SegRole::Weight.segment()],
        layout.weight_bank_bytes.clone(),
    )
}

/// ⛔⛔⛔ THE WEIGHT-SEGMENT PARITY LEDGER'S LAW — every bundle one model bakes must report the
/// SAME [`weight_seg_record`], or the build refuses naming both bundles.
///
/// The expansion records one entry per superdsc bundle it lowers — the decode ceiling, every
/// ladder rung, every batch rung, every prefill width — and calls this once, after the last
/// bake. The first entry is the reference; any later entry that disagrees is a BUILD error,
/// never a card failure: the runtime's alias guard compares exactly these numbers when a
/// session takes another's resident weights, so a skew here is a load-time refusal there.
/// This ledger is the guard that keeps the NEXT skew a build error instead of a card
/// failure.
///
/// An empty ledger is `Ok` — a bake that recorded nothing (KTIR-only, no superdsc section)
/// has no bundles to disagree.
pub fn weight_segment_parity(ledger: &[(String, u64, Vec<u64>)]) -> Result<(), String> {
    let Some((first_base, first_bank0, first_banks)) = ledger.first() else {
        return Ok(());
    };
    for (base, bank0, banks) in ledger.iter().skip(1) {
        if bank0 != first_bank0 || banks != first_banks {
            return Err(format!(
                "two bundles of one model pack different resident weight segments — every bundle \
                 shares ONE weight segment (the sessions alias it outright, and the alias guard \
                 refuses on these very numbers at card load): {first_base} packs bank0 {first_bank0} \
                 B + {} bank(s) {:?}, {base} packs bank0 {bank0} B + {} bank(s) {:?}",
                first_banks.len(),
                first_banks,
                banks.len(),
                banks,
            ));
        }
    }
    Ok(())
}

/// Build the GLOBAL [`BundleLayout`] for one fused SuperDSC bundle (task #55).
///
/// Classification by liveness over `ir.nodes` (already a valid eval order):
/// - a SOURCE tensor (`id < num_sources`) in `weight_ids` → [`SegRole::Weight`]
///   (seg1, packed contiguously, resident — all weights are simultaneously live so
///   no reuse, just dense packing);
/// - a SOURCE not in `weight_ids` → [`SegRole::Activation`] (seg0, per-step input);
/// - `ir.result` → [`SegRole::Logits`] (seg4);
/// - every other tensor (produced by some op AND consumed) → [`SegRole::Intermediate`]
///   (seg3), assigned by a lifetime-aware linear scan that REUSES the byte range of an
///   intermediate whose `last_use` precedes the new tensor's `first_def`.
///
/// All sizes are f16 bytes (`rows*cols*2`), 128 B aligned. A build-time disjointness
/// guard ([`bundle_layout_aliases`]) asserts no two SIMULTANEOUSLY-LIVE tensors in a
/// segment overlap — a silent on-card aliasing write is a `cargo build` Err.
/// THE KV POOLS — every attention's K and V cache tensors, each once, in layer order: what a card
/// bake places in the KV segment ([`compute_bundle_layout`]) and an emulator bake leaves to the
/// emulator, which holds them as the KTIR's own buffers.
/// A node's scalar constants as the card applies them: each is staged as an fp16 constant stick,
/// so the value the device multiplies or adds is `f16(v)`, never `v` — and the spliced programs
/// state the same scalars as f16 constexprs, so the emulator applies `f16(v)` too. Staging the
/// reference puts it on the device's side of both. At gemma-4's `eps = 1e-6`, `f16(1e-6)` is an
/// IEEE subnormal, a real precision fact of the card.
///
/// Returns whether anything was staged — `false` leaves the caller free to use the original node.
fn stage_registry_scalars<F: RopeForm>(op: &mut SubOp<F>) -> bool {
    let stage = |v: &mut f32| *v = half::f16::from_f32(*v).to_f32();
    match op {
        SubOp::ScalarMul { scale } => {
            stage(scale);
            true
        }
        SubOp::RmsNorm { eps, .. } => {
            stage(eps);
            true
        }
        SubOp::RmsNormUnit { eps } => {
            stage(eps);
            true
        }
        SubOp::TanhSoftCap { cap } => {
            stage(cap);
            true
        }
        _ => false,
    }
}

/// ⭐ THE EMITTER'S ROW PAD — `mq` rows padded up to whole fp16 sticks. THE one
/// home of the pad every row-count statement takes: the emitter's own placement
/// of the attention zero tensor (`compute_bundle_layout`, `mq_pad` off the
/// widest `AttnDecode`), and the worker's launch (`spyre_forward`'s
/// stick-padded row count for the constants' tables, `spyre_load`'s load-time
/// tape check). The stick is [`Fp16`]'s own — a row count that is not whole
/// fp16 sticks is not a shape this ladder places, and the launch's whole-stick
/// refusal on `hidden` checks the same fact.
pub fn mq_pad(mq: usize) -> usize {
    mq.next_multiple_of(FP16_ELEMS_PER_STICK as usize)
}

pub fn kv_pool_tids<F: RopeForm>(ir: &SubtileIR<F>) -> Vec<u32> {
    let mut pools: Vec<u32> = Vec::new();
    for n in &ir.nodes {
        if let SubOp::AttnDecode { layout: kv, .. } = &n.op {
            for t in [kv.cache_tensor(), kv.v_cache_tensor()] {
                let tid = t.index() as u32;
                if !pools.contains(&tid) {
                    pools.push(tid);
                }
            }
        }
    }
    pools
}

pub fn compute_bundle_layout<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // ⭐ THE LAYER STRUCTURE, so the weight segment can be split into BANKS at a LAYER boundary —
    // see [`bank_weight_segment`]. ONE MAP PER ATTENTION CLASS (a hybrid model's classes have
    // different layer counts, and banking needs the class boundaries), from
    // [`per_layer_external_tids`], a pre-pass over the re-rolled tape; EMPTY for an unrolled
    // bundle, which has no layer boundary and therefore cannot bank. A uniform model is one class,
    // so a one-element slice is exactly the pre-class single map.
    // ⭐ WHAT THE ROWS ARE — one sequence's, or a decode batch's: the paged step's sources differ.
    paged_rows: scratchy_triton_splice::PagedRows,
    per_layer_ext: &[std::collections::BTreeMap<u32, Vec<u32>>],
    // The graph's query extents, minted by the bake — threaded to the registry walk below,
    // never re-derived from graph rows (see [`bundle_registry`]'s own parameter note).
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> Result<BundleLayout, SuperDscError> {
    // ⭐ THE PER-TENSOR DEVICE FORMAT, FROM THE KERNEL SIGNATURES — a tensor a spliced kernel reads
    // is sized at the format that kernel's signature states for it
    // ([`scratchy_triton_splice::operand_dfs`]); every other tensor is fp16. A seat whose kernel
    // does not compile states no formats here — the refusal is the lowering's to raise.
    let kernel_dfs: std::collections::BTreeMap<u32, DataFormat> = {
        let mut m: std::collections::BTreeMap<u32, DataFormat> = Default::default();
        for n in &ir.nodes {
            let Some(dfs) = scratchy_triton_splice::operand_dfs(n, ir).ok().flatten() else {
                continue;
            };
            m.extend(dfs.into_iter().filter(|&(_, df)| df != DataFormat::Fp16));
        }
        m
    };
    // ⭐ EACH BUFFER HOLDS EVERY ELEMENT ITS PROGRAMS REACH — a card bake's division pads a stick
    // axis past the tensor's own extent ([`crate::superdsc_lower::rooms`]).
    let room = crate::superdsc_lower::rooms::<T, F>(
        ir,
        weight_ids,
        paged_rows == scratchy_triton_splice::PagedRows::Batch,
        query_rows,
    )
    .map_err(SuperDscError)?;
    // A `[rows, cols]` tensor is held row-major, each row whole sticks of its format.
    let nbytes = |tid: u32| -> u64 {
        let s = ir.tensors[tid as usize];
        let df = kernel_dfs.get(&tid).copied().unwrap_or(DataFormat::Fp16);
        let lanes = df.lanes::<ktir_core::ddl_binds::Baked>();
        let natural =
            u64::from(s.rows) * u64::from(s.cols).next_multiple_of(lanes) * df.word_bytes();
        natural.max(room.get(&tid).copied().unwrap_or(0))
    };
    // ── Liveness: first def (output of node i) and last use (input of node j). A
    //    source has no def (def = 0, live from the start); the result has no use
    //    (use = nodes.len(), live to the end). ──
    // ⭐ THE SCALARMUL WEIGHT FOLDS COME FIRST — every liveness and placement fact below must
    //    already know them. A folded ScalarMul emits NO op, so its output tensor is written by
    //    nobody and every reader of it is re-pointed at the PRODUCER's bytes (the placement
    //    alias in the intermediate scan). Computing the folds before liveness is what lets the
    //    producer's bytes stay live for the folded node's readers instead of being reclaimed
    //    under them.
    //
    // ⛔ TWO LISTS, BECAUSE THE TWO FOLD KINDS DIFFER IN WHAT HAPPENS TO THE NODE. The union
    // (`weight_scale_folds`) is everything the worker applies to staged bytes at load — the
    // ScalarMul folds AND the attention-scale folds — and it is what the census skips and the
    // layout carries. The ScalarMul folds ALONE leave the tape: their node is host-routed and
    // their output placement aliases the producer's bytes, which is why the liveness extension
    // and the placement alias below key on `scalarmul_folds` and never on the union — an
    // attention fold's node STAYS (at scale 1.0), its output is a real intermediate, and
    // aliasing it to the query's bytes would hand every attention consumer the query buffer.
    let weight_scale_folds = crate::lower_subtile_tape_to_ktir::weight_scale_folds(ir);
    let scalarmul_folds = crate::lower_subtile_tape_to_ktir::scalar_mul_weight_folds(ir);
    let n_nodes = ir.nodes.len();
    let mut first_def: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut last_use: std::collections::BTreeMap<u32, usize> = Default::default();
    for (i, node) in ir.nodes.iter().enumerate() {
        let o = node.output.tensor.index() as u32;
        first_def.entry(o).or_insert(i);
        for inp in &node.inputs {
            last_use.insert(inp.tensor.index() as u32, i);
        }
    }
    // ⛔ A FOLDED SCALARMUL'S READERS READ THE PRODUCER'S BYTES. The layout aliases the folded
    //    node's output placement to its producer's (in the intermediate scan below), so the
    //    producer's byte range must stay un-reused for as long as the FOLD'S OUTPUT is live —
    //    the scan would otherwise reclaim those bytes the moment the (now op-less) ScalarMul
    //    "consumes" the product, and the alias would surface as silent corruption: the fold's
    //    readers would read a later intermediate that the scan moved into the reclaimed range.
    for (fold_id, _, _) in &scalarmul_folds {
        let node = &ir.nodes[fold_id.index()];
        let producer_out = node.inputs[0].tensor.index() as u32;
        let fold_out = node.output.tensor.index() as u32;
        if let Some(&exp) = last_use.get(&fold_out)
            && exp > *last_use.get(&producer_out).unwrap_or(&0)
        {
            last_use.insert(producer_out, exp);
        }
    }

    let mut placements: std::collections::BTreeMap<u32, TensorPlacement> = Default::default();
    let mut seg_bytes = [0u64; 7];
    // Set by the paged-KV placement below, from the SAME `PagedKvPool` that sized the placement, so the
    // stride the runtime shifts by and the stride the addresses were baked with are one value.
    let kv_request_stride_bytes: u64 = 0;

    // ── Pack WEIGHTS (seg1) + ACTIVATIONS (seg0) + LOGITS (seg4): all simultaneously
    //    live within their role (weights resident the whole program; the single
    //    logits row lives to the end), so dense contiguous packing by tensor id. ──
    let pack = |tid: u32,
                role: SegRole,
                seg_bytes: &mut [u64; 7],
                place: &mut std::collections::BTreeMap<u32, TensorPlacement>| {
        let seg = role.segment();
        let off = seg_bytes[seg];
        let sz = nbytes(tid);
        place.insert(
            tid,
            TensorPlacement {
                tid,
                role,
                segment: seg,
                // Bank 0 — every placement is packed into one region here, and `bank_weight_segment`
                // is the ONE pass that ever moves a weight to another bank.
                bank: 0,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);
    };

    // The AttnDecode K/V cache tensors are SOURCES, but they are the KV POOLS, placed in seg2
    // (SegRole::Kv) below (search "THE KV POOLS"). If the source loop ALSO packs them into the
    // Activation segment, that Activation slot is never used — a DEAD HOLE that the per-step
    // whole-Activation-segment H2D re-ships every decode step (the O(cap) decode preamble cliff,
    // measured: 50 ms @cap=4096 vs 8 ms @cap=256). Skip them here; their only placement is seg2.
    let prefix_kv_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            SubOp::AttnDecode { layout: kv, .. } => Some([
                kv.cache_tensor().index() as u32,
                kv.v_cache_tensor().index() as u32,
            ]),
            _ => None,
        })
        .flatten()
        .collect();

    // Sources first (deterministic id order): weights → seg1, activations → seg3 (Activation role).
    for tid in 0..ir.num_sources {
        if prefix_kv_tids.contains(&tid) {
            continue; // re-placed into seg2 (Kv) below — no dead Activation slot / no per-step re-H2D
        }
        let role = if weight_ids.contains(&tid) {
            SegRole::Weight
        } else {
            SegRole::Activation
        };
        pack(tid, role, &mut seg_bytes, &mut placements);
    }
    // ── KSPLIT block WEIGHTS (seg1) ── the lm_head split's B block weights `ksplit_block_tid(b)` are added by
    // the `kernel0` loop (RetileDescriptors) + the worker (bytes), but NOT by the source loop above (they are
    // not manifest sources). Without a placement they get NO HBM address → staged to a default/aliased spot →
    // GARBAGE logits (on-card-observed: KSPLIT output empty/EOS while the bundle loaded fine). Place each in
    // seg1 (Weight) here — same condition as the emit branch + kernel0 (n_dev>16384 lm_head). Size = the
    // staged `[KB, n_dev]` f16 buffer (KB·n_dev·2, the retiled `[n_dev/64,KB,64]` footprint).
    // ── THE KV POOLS (seg2) ── every attention's K and V cache tensors, the pools the paged step's KV
    //    write scatters into and its walk gathers from, resident across steps — ONE DEVICE REGION PER
    //    LAYER ([`place_kv_pools`]). On a CARD build only: the emulator runs the paged step as KTIR
    //    over the cache tensors bound by tid, and never places them.
    let pools = if cfg!(feature = "spyre-lower") {
        kv_pool_tids(ir)
    } else {
        Vec::new()
    };
    let (kv_bank_bytes, kv_block_banks) = place_kv_pools(
        &pools,
        per_layer_ext,
        nbytes,
        &mut seg_bytes,
        &mut placements,
    );
    // Logits (seg4) — the graph result (it is also produced by an op, but its role
    // is OUTPUT; classify it before intermediates so it is not pool-coalesced).
    let result = ir.result.index() as u32;
    if result >= ir.num_sources {
        pack(result, SegRole::Logits, &mut seg_bytes, &mut placements);
    }
    // ⭐ THE SEGMENT BUDGET — every weight is packed by now, so this is the first point that knows
    //    whether seg1 fits one device region, and it runs BEFORE the intermediate coloring so a
    //    spilled tail lands at offset 0 of its slot. That offset is not cosmetic: the spill slot is
    //    NOT aliasable (it also holds mq-dependent intermediates, so its extent differs per rung),
    //    so every borrowing session has to be handed the tail's own device image — and an image is
    //    only portable between bundles of different seg6 extents if the tail starts at 0 in all of
    //    them. See `spill_weight_tail`.
    //
    // ⭐ THE PROVEN LEVER FIRST, AND BANKING ONLY WHERE THE SPILL CANNOT REACH. Two passes can bring
    //    an over-size weight segment under the cap, and they are NOT interchangeable — one of them
    //    keeps a segment number naming exactly one region and the other does not:
    //
    //    * [`spill_weight_tail`] moves the NON-PER-LAYER tail to a DISTINCT segment number. Every
    //      address it emits stays unambiguous, because `SEGMENT_OFFSETS[seg] + offset` still resolves
    //      to one region per segment. MEASURED on the card, granite-3.1-8b-instruct at fp16: coherent
    //      prose, TTFT 206.9 ms, ITL 189.2 ms, 5.3 tok/s, 0 WARN/SKIPPED/REFUSED.
    //    * [`bank_weight_segment`] backs ONE segment number with SEVERAL regions. It is the only lever
    //      on the PER-LAYER block (which the spill cannot touch) and its banks are aliasable where the
    //      spill slot is not — but it makes two regions share a segment number, and with it EVERY
    //      forward faulted `CB tag=ResponseTag(7341) state=Succeeded status=Error locator=0x2`,
    //      unchanged across three fixes that MOVED the addresses. An unchanged fault across an address
    //      change says the addresses are not what is wrong: dxp wires producer→consumer BY SEGMENT, so
    //      a shared segment number is suspected to be structurally inexpressible.
    //
    //    So take the spill wherever it has a lever at all — i.e. wherever the PER-LAYER BLOCK ALONE
    //    fits one region, which is the only part the spill cannot move — and fall back to banking only
    //    for a model whose per-layer block is itself over the cap (13B/30B/70B dense fp16), where the
    //    spill is powerless and banking is the only expressible answer. Both are no-ops for the
    //    overwhelming case of a weight segment that fits one region.
    //
    //    ⛔ Do NOT reorder these on the grounds that banking is more general. It is more general and
    //    it does not work yet; the spill is narrower and it is measured. Settle the stitcher question
    //    (`ModuleStitcher` in deeptools) before promoting banking.
    let per_layer_tids: std::collections::BTreeSet<u32> = per_layer_ext
        .iter()
        .flat_map(|m| m.values())
        .flatten()
        .copied()
        .collect();
    let per_layer_block_end = placements
        .values()
        .filter(|p| {
            p.segment == SegRole::Weight.segment()
                && matches!(p.role, SegRole::Weight)
                && per_layer_tids.contains(&p.tid)
        })
        .map(|p| align128(p.offset + p.size))
        .max()
        .unwrap_or(0);
    let (weight_bank_bytes, weight_tail) = if per_layer_block_end <= bundle::MAX_SEGMENT_BYTES {
        let tail = spill_weight_tail(&mut placements, &mut seg_bytes, &per_layer_tids)?;
        (Vec::new(), tail)
    } else {
        // A bank holds the tail like any other weight, so nothing is spilled.
        (
            bank_weight_segment(&mut placements, &mut seg_bytes, per_layer_ext)?,
            WeightTail::InPlace,
        )
    };
    // ⭐ THE TENSOR-PARALLEL RECEIVE WINDOW — placed iff the forward carries a collective, in the
    // weight segment's base region (bank 0) directly above the packed weights: `WINDOW_CUTS`
    // adjacent cuts, each one block per sending rank of the forward's world, each block
    // `RecvWindow::block_bytes_for` its widest collective operand, one placement per block's tid
    // (`WindowCut::block_tid`) — collective `k` lands in cut `WindowCut::of_collective(k)`.
    //
    // ⛔ UNIFORM BY CONSTRUCTION, NOT BY COORDINATION. Every bundle of a forward (prefix, rolled
    // body, suffix) is baked over THIS one layout, and `weight_segment_parity` holds every layout
    // of a session to one weight extent — so `(seg1, align128(extent), size)` is the same triple
    // wherever a send is baked and wherever its consume reads. The size reads only the operand's
    // COLUMNS and the target's row cap (`WINDOW_ROWS`), never this bundle's row count, which is
    // what keeps a decode layout and every prefill rung's equal; and `consume_fragments` cuts at
    // the same law, so no fragment can exceed it.
    let widest_cols = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::AllReduce | SubOp::AllGather { .. }))
        .map(|n| n.inputs[0].region.cols.len)
        .max();
    // A collective in a world of one is a forward the analysis never mints (its identity at
    // `W = 1`), so the window is placed for exactly the sharded worlds that carry collectives.
    if let Some(widest_cols) =
        widest_cols.filter(|_| <T as scratchy_spyre_bundle::collective::World>::SHARDED)
    {
        let w = SegRole::Weight.segment();
        let block =
            scratchy_spyre_bundle::collective::RecvWindow::<T>::block_bytes_for(widest_cols)
                .ok_or_else(|| {
                    SuperDscError(format!(
                        "a collective's operand has {widest_cols} columns — no receive window of a \
                     {}-rank world holds it",
                        <T as scratchy_spyre_bundle::collective::World>::RANKS
                    ))
                })?;
        let window =
            scratchy_spyre_bundle::collective::RecvWindow::<T>::new(align128(seg_bytes[w]), block)
                .ok_or_else(|| SuperDscError("an align128 offset is flit-aligned".to_string()))?;
        if window.end() > bundle::MAX_SEGMENT_BYTES {
            return Err(SuperDscError(format!(
                "the tensor-parallel receive window [{}, {}) does not fit the weight segment's \
                 {}-byte base region above the packed weights",
                window.offset(),
                window.end(),
                bundle::MAX_SEGMENT_BYTES,
            )));
        }
        // One placement per (cut, sending rank): rank `s`'s block of cut `c`, where `s`'s sends
        // into `c` land on every card and where every consume reads `s`'s share.
        for cut in (0..scratchy_spyre_bundle::collective::WINDOW_CUTS)
            .map(scratchy_spyre_bundle::collective::WindowCut::of_collective)
        {
            for from in scratchy_spyre_bundle::collective::ranks::<T>() {
                let tid = cut.block_tid(from);
                placements.insert(
                    tid,
                    TensorPlacement {
                        tid,
                        role: SegRole::Weight,
                        segment: w,
                        bank: 0,
                        offset: window.block_offset(cut, from),
                        size: window.block().get(),
                    },
                );
            }
        }
        seg_bytes[w] = window.end();
    }
    // ── INTERMEDIATES (seg3): lifetime-aware linear scan with byte-range reuse. ──
    // Collect every produced tensor that is NOT a source and NOT the logits, sorted
    // by first_def (def order == node order). Maintain `live` = currently-assigned
    // (offset, size, expiry=last_use) and `free` = reclaimed (offset, size) holes.
    // A FOLDED ScalarMul's output is placed by the ALIAS below, not by this pool — a colored
    // slot of its own would be a dead hole beside the producer's bytes its readers are
    // re-pointed at. (An attention-scale fold's node keeps running, so its output is an
    // ordinary intermediate and colors normally.)
    let folded_out: std::collections::HashSet<u32> = scalarmul_folds
        .iter()
        .map(|(id, _, _)| ir.nodes[id.index()].output.tensor.index() as u32)
        .collect();
    let mut inter: Vec<u32> = first_def
        .keys()
        .copied()
        .filter(|&t| t >= ir.num_sources && t != result && !folded_out.contains(&t))
        .collect();
    inter.sort_by_key(|t| (first_def[t], *t));
    // ── SEGMENT COLORING (the multi-op stitching fix): dxp's ModuleStitcher connects
    //    producer→consumer by SEGMENT (the dxp ref test_softmax_1core puts each
    //    inter-op tensor in its OWN segment: sub→seg2, exp reads seg2). scratchy used
    //    to cram ALL intermediates into seg3 (offset-distinguished) — but the stitcher
    //    can't tell t337 from t338 (both seg3) ⇒ the PT matmul's output is mis-wired
    //    (single-op MatMul_49 works; multi-op all-seg3 orphans). FIX: spread
    //    intermediates across the free segments {3,5,6} (lifetime-aware reuse) so no
    //    two SIMULTANEOUSLY-LIVE intermediates share a segment. ≤3 live fits; on
    //    overflow we fall back to seg3 offset-packing (the old behavior) for that tid.
    let inter_segs = [SegRole::Intermediate.segment(), 5usize, 6usize];
    // Per-segment: the live tensor's expiry (usize::MAX = free) + the running offset.
    let mut seg_live_exp = [0usize; 3]; // 0 = free (no live tensor)
    for &tid in &inter {
        let def = first_def[&tid];
        let exp = *last_use.get(&tid).unwrap_or(&n_nodes);
        let sz = align128(nbytes(tid));
        // Free any segment whose live tensor expired before this def.
        for exp in &mut seg_live_exp {
            if *exp != 0 && *exp < def {
                *exp = 0;
            }
        }
        // Pick the first free segment in {3,5,6}; else overflow into seg3 (offset-packed).
        let pick = (0..3).find(|&s| seg_live_exp[s] == 0);
        let seg = match pick {
            Some(s) => {
                seg_live_exp[s] = exp;
                inter_segs[s]
            }
            None => inter_segs[0], // overflow: reuse seg3 (offset-packed, may alias)
        };
        let off = seg_bytes[seg];
        seg_bytes[seg] = off + sz;
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Intermediate,
                segment: seg,
                bank: 0,
                offset: off,
                size: nbytes(tid),
            },
        );
    }

    // ⭐ THE FOLD'S PLACEMENT ALIAS. A folded ScalarMul writes nothing, so its output tensor's
    //    readers are pointed at the PRODUCER's bytes: same segment, same offset, same size — the
    //    recognition required the folded node's output region to MIRROR its input, and its input
    //    IS the producer's whole output, so the window is byte-for-byte the window the matmul
    //    wrote (which now holds `x·(W·s)` once the worker scales the staged weight). The scan
    //    above skipped these tids; this is their single placement. Same mechanism as the on-card
    //    residual threading (`hidden_out` ← `hidden_in`), and safe for the same reason the
    //    liveness extension above is: the producer's range is not reclaimable while the fold's
    //    readers live.
    for (fold_id, _, _) in &scalarmul_folds {
        let node = &ir.nodes[fold_id.index()];
        let producer_out = node.inputs[0].tensor.index() as u32;
        let fold_out = node.output.tensor.index() as u32;
        let p = placements[&producer_out];
        placements.insert(fold_out, TensorPlacement { tid: fold_out, ..p });
    }

    // ── THE PAGED STEP'S SOURCES ── spyre-inference's operands the host stages each step
    //    (`crate::paged_step`): the page index table, the mask stack, the KV write index and the
    //    query row index, at their reserved tids in the ACTIVATION segment like every per-step
    //    binding (the reserved-tid law above), and the decode kernel's `kv_row_pool` and int32 zero
    //    in the intermediate segment, which neither a layer's launch nor a walk's relaunch shifts
    //    (the KV segment moves with every layer). `PagedStep::card_placements` is
    //    the one statement of what they are and how big — the card stagers stage within it.
    for step in crate::lower_subtile_tape_to_ktir::paged_step_of_ir(
        ir,
        ActiveCap::FULL,
        paged_rows,
        query_rows,
    )?
    .iter()
    {
        for (tid, seat, size) in step.card_placements() {
            let role = match seat {
                crate::wiring::CardSeat::Step => SegRole::Activation,
                crate::wiring::CardSeat::Static => SegRole::Intermediate,
            };
            let seg = role.segment();
            let off = seg_bytes[seg];
            placements.insert(
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
            seg_bytes[seg] = align128(off + size);
        }
    }
    // ── AN OP'S OWN STAGING ── a buffer an op reaches that the graph does not own (the prefill
    //    lm-head tail's `LAST_HIDDEN_TID`, written and read inside one node's ops) is placed in the
    //    intermediate segment at the size its descriptors reach.
    let staging: Vec<(u32, u64)> = room
        .iter()
        .filter(|(t, _)| !placements.contains_key(t))
        .map(|(&t, &b)| (t, b))
        .collect();
    for (tid, size) in staging {
        let seg = SegRole::Intermediate.segment();
        let off = seg_bytes[seg];
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Intermediate,
                segment: seg,
                bank: 0,
                offset: off,
                size,
            },
        );
        seg_bytes[seg] = align128(off + size);
    }

    let mut resident_source_tids: Vec<u32> = prefix_kv_tids.iter().copied().collect();
    resident_source_tids.sort_unstable();
    Ok(BundleLayout {
        placements,
        segment_bytes: seg_bytes,
        weight_bank_bytes,
        kv_bank_bytes,
        kv_block_banks,
        weight_tail,
        kernel_weights: Default::default(),
        held: Default::default(),
        held_views: Default::default(),
        stick_consts: Vec::new(),
        weight_scale_folds: weight_scale_folds
            .into_iter()
            .map(|(_, tid, scale)| (tid, scale))
            .collect(),
        kv_request_stride_bytes,
        num_sources: ir.num_sources,
        resident_source_tids,
    })
}

/// ⭐ THE SEGMENT BUDGET: give the weight segment's TRAILING placements to [`WEIGHT_SPILL_SEGS`]
/// until what is left fits one device region.
///
/// A segment is one `FlexAllocator` allocation served from ONE 16 GiB region, and the same 16 GiB is
/// the addressing stride (`SEGMENT_OFFSETS`), so over the cap a bundle both fails to allocate AND
/// decomposes its own addresses into the next segment. Both are decided by constants here, which is
/// why this is a `cargo build` pass and not a load-time fallback.
///
/// 🛑 MEASURED, granite-3.1-8b at fp16 — and the arithmetic is the whole design:
///
/// ```text
///   40 layers × 423,641,088 = 16,945,643,520   the PER-LAYER block   ⎫ 15.78 GiB — FITS,
///                    + 8,192 =      final norm                       ⎬ with 223 MiB spare
///              + 419,430,400 = the tied embedding (51200 × 4096 × 2) ⎭
///                            = 17,365,082,112   177 MiB PAST the cap
/// ```
///
/// The overflow is caused ENTIRELY by the non-per-layer tail, and the per-layer block alone clears
/// the cap with 223 MiB to spare. So the fix is not to split the per-layer block (which the rolled
/// body could not address — see below) but to move the tail, which nothing strides.
///
/// ⛔ FROM THE END, AND ONLY THE END. Weights are packed in tid order and the non-per-layer ones
/// (final norm, then the tied embedding) sort LAST, so taking a trailing suffix leaves every
/// remaining weight at the offset it already had — which is what keeps `weight_stride` uniform, keeps
/// every model whose weights already fit BYTE-IDENTICAL, and keeps seg1's measurably load-bearing
/// placement untouched. Taking from the front, or repacking, would move all 362 of them.
///
/// ⛔ THIS DOES NOT RAISE THE CEILING ON THE PER-LAYER BLOCK, and it is worth being exact about why.
/// The rolled body reaches layer `v` by advancing ONE segment's base (`off[SEG_WEIGHT] = v·wstride`
/// in `superdsc_exec::launch_forward`), so every per-layer weight must sit in one strided segment.
/// Splitting the per-layer block IS expressible without touching any SDSC — `tensor_allocs` is
/// positional per launch and `DevAddr::shifted` exists, so layer `v` could be handed a different
/// region's base in slot 1 — but it needs the weight segment to become a *bank* of regions (staging,
/// H2D and `alias_seg_from` all per bank), and it is NOT built here. Dense fp16 therefore still tops
/// out where 40 layers of weights top out; this is what unblocks the tail, not a 30B.
///
/// ⭐ THE VERDICT IS RETURNED, AND BAKED — and [`weight_tail_of_placements`] is its proof. [`WeightTail::Spilled`] names the slot and the tail's
/// extent, which is the runtime's whole copy work list — so the decision travels in the bundle as a
/// typed fact instead of being re-derived from where the placements sit.
fn spill_weight_tail(
    placements: &mut std::collections::BTreeMap<u32, TensorPlacement>,
    seg_bytes: &mut [u64; 7],
    per_layer_tids: &std::collections::BTreeSet<u32>,
) -> Result<WeightTail, SuperDscError> {
    let w_seg = SegRole::Weight.segment();
    if seg_bytes[w_seg] <= bundle::MAX_SEGMENT_BYTES {
        return Ok(WeightTail::InPlace); // the overwhelming case: nothing moves, nothing is re-offset.
    }
    // ⛔ ONE SLOT, BY THE ARRAY'S TYPE. `WeightTail::Spilled` names one segment; a second declared
    // slot would let the tail split across two, which that verdict cannot state. This pattern is
    // irrefutable only while the table has exactly one row, so growing it is a compile error here.
    let [spill_slot] = WEIGHT_SPILL_SEGS;
    // The weight segment's own high-water, recomputed from what REMAINS after each move. The
    // original packing bumped `seg_bytes` cumulatively, so the high-water is the max end — taking
    // the trailing placement lowers it to the next one's end and never leaves a hole.
    let high_water = |pl: &std::collections::BTreeMap<u32, TensorPlacement>| -> u64 {
        pl.values()
            .filter(|p| p.segment == w_seg)
            .map(|p| align128(p.offset + p.size))
            .max()
            .unwrap_or(0)
    };
    // Trailing first. Only `Weight`-role placements are movable: seg1 also carries a tiny attention
    // `scale` const, and moving a non-weight would take it away from the role that binds it.
    let mut tail: Vec<u32> = placements
        .values()
        .filter(|p| p.segment == w_seg && matches!(p.role, SegRole::Weight))
        .map(|p| p.tid)
        .collect();
    tail.sort_by_key(|t| std::cmp::Reverse(placements[t].offset));

    let mut moved: Vec<(u32, usize, u64)> = Vec::new();
    for tid in tail {
        if seg_bytes[w_seg] <= bundle::MAX_SEGMENT_BYTES {
            break;
        }
        // ⛔⛔⛔ ROLE, NOT POSITION: A PER-LAYER WEIGHT MAY NEVER LEAVE THE STRIDED SEGMENT.
        // The rolled body reaches layer `v` by advancing ONE segment's base
        // (`off[SEG_WEIGHT] = v·weight_stride` in `superdsc_exec::launch_forward`), so a per-layer
        // weight moved out of `w_seg` is no longer strided by `v` — every layer would read LAYER 0's
        // copy of it. That is FLUENT GARBAGE: the model generates confident, well-formed text from
        // the wrong weights, and no load, no bake and no on-card fault reports it. It has to die at
        // `cargo build`.
        //
        // This holds today by CONSTRUCTION and not by luck — weights pack in tid order and the
        // non-per-layer ones (final norm, tied embedding) sort LAST, so a trailing suffix is exactly
        // the non-per-layer tail — but "by construction" is a fact about the CURRENT packing of the
        // CURRENT configs, not an invariant. A config whose last-packed weight is per-layer would
        // otherwise be silently mis-emitted, so the guard is keyed on the tensor's ROLE.
        //
        // ⚠️ And note what is NOT refused: a per-layer INTERMEDIATE coloured into a spill slot is
        // perfectly legal — intermediates are re-bound per launch and carry no `v·stride`. Refusing
        // on "is in a spill segment" instead of "is a per-layer WEIGHT" is what broke an earlier
        // attempt at this guard.
        if per_layer_tids.contains(&tid) {
            return Err(SuperDscError(format!(
                "seg{w_seg} packs {} B, {} B past the {} B one device region can hold, and the next \
                 tensor the tail spill would move (t{tid}) is a PER-LAYER weight. Moving it out of \
                 the strided weight segment would leave every layer reading layer 0's copy — fluent \
                 garbage that nothing downstream can detect — so this is a build refusal. The \
                 non-per-layer tail is already spilled; what remains over the cap is the per-layer \
                 block itself, and splitting THAT needs the weight segment to become a bank of \
                 regions (see `bank_weight_segment`).",
                seg_bytes[w_seg],
                seg_bytes[w_seg] - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
            )));
        }
        let sz = placements[&tid].size;
        // The slot must have room for it. `MAX_SEGMENT_BYTES` binds the spill slot too — it is a
        // device region like any other — so this cannot trade one over-size segment for another.
        let spill = spill_slot;
        if align128(seg_bytes[spill]) + sz > bundle::MAX_SEGMENT_BYTES {
            return Err(SuperDscError(format!(
                "seg{w_seg} packs {} B, {} B past the {} B one device region can hold, and the \
                 spill slot seg{spill} is too full to take t{tid} ({sz} B). The tail that CAN move \
                 is already moved: what is left is the per-layer block, and the rolled body \
                 advances exactly one segment's base per layer, so it cannot be split without \
                 making the weight segment a bank of regions (see `spill_weight_tail`).",
                seg_bytes[w_seg],
                seg_bytes[w_seg] - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
            )));
        }
        // Land above whatever the slot already holds — which is NOTHING, because this runs before
        // the intermediate coloring, so the first spilled tensor sits at offset 0 and the colored
        // intermediates pack above it. That is deliberate and load-bearing: the runtime hands each
        // borrowing session a COPY of `[0, tail_len)` of this segment (the slot cannot be aliased),
        // and one image is only portable to a bundle of a different extent if the tail starts at 0.
        let off = align128(seg_bytes[spill]);
        let p = placements.get_mut(&tid).expect("tid came from this map");
        p.segment = spill;
        p.offset = off;
        seg_bytes[spill] = align128(off + sz);
        seg_bytes[w_seg] = high_water(placements);
        moved.push((tid, spill, sz));
    }
    // Emit runs at cargo-build, so the spill is visible in the build log rather than inferred from a
    // segment total. Not gated: a bundle that had to re-budget its segments should say so once.
    eprintln!(
        "[superdsc-layout] SEGMENT BUDGET: spilled {} weight tensor(s) out of seg{w_seg}, leaving \
         {} B of the {} B cap; {}",
        moved.len(),
        seg_bytes[w_seg],
        bundle::MAX_SEGMENT_BYTES,
        moved
            .iter()
            .map(|(t, s, sz)| format!("t{t} → seg{s} ({sz} B)"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    // The slot held nothing before the first move (this runs before the intermediate coloring), so
    // its high-water now is exactly the tail's extent `[0, bytes)`.
    Ok(WeightTail::Spilled {
        segment: spill_slot as u32,
        bytes: seg_bytes[spill_slot],
    })
}

/// The weight-tail verdict as the PLACEMENTS state it: the spill slot some WEIGHT sits in, and the
/// high-water of the weights there. What [`spill_weight_tail`] returned must equal this once every
/// placement is final, which `bake_layout` proves before the verdict is baked.
fn weight_tail_of_placements(
    placements: &std::collections::BTreeMap<u32, TensorPlacement>,
) -> WeightTail {
    let [spill_slot] = WEIGHT_SPILL_SEGS;
    placements
        .values()
        .filter(|p| p.segment == spill_slot && matches!(p.role, SegRole::Weight))
        .map(|p| align128(p.offset + p.size))
        .max()
        .map_or(WeightTail::InPlace, |bytes| WeightTail::Spilled {
            segment: spill_slot as u32,
            bytes,
        })
}

#[cfg(test)]
mod weight_tail_tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// granite-3.1-8b's weight segment as the layout packs it: 40 per-layer blocks, then the final
    /// norm, then the 51200-wide tied embedding, in tid order.
    fn granite_8b(
        per_layer_block: u64,
    ) -> (BTreeMap<u32, TensorPlacement>, [u64; 7], BTreeSet<u32>) {
        let w = SegRole::Weight.segment();
        let mut placements = BTreeMap::new();
        let mut seg_bytes = [0u64; 7];
        let mut per_layer = BTreeSet::new();
        let sizes = (0..40u32)
            .map(|t| (t, per_layer_block))
            .chain([(40, 8_192), (41, 51_200 * 4_096 * 2)]);
        for (tid, size) in sizes {
            let offset = align128(seg_bytes[w]);
            placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role: SegRole::Weight,
                    segment: w,
                    bank: 0,
                    offset,
                    size,
                },
            );
            seg_bytes[w] = align128(offset + size);
            if tid < 40 {
                per_layer.insert(tid);
            }
        }
        (placements, seg_bytes, per_layer)
    }

    /// The padded 8b (gate/up at 14336) packs 177 MiB past one region: the embedding spills, and the
    /// verdict names the slot and the tail's extent — which the placements state too.
    #[test]
    fn the_padded_8b_spills_its_embedding_and_the_verdict_says_so() {
        let (mut placements, mut seg_bytes, per_layer) = granite_8b(423_641_088);
        assert!(seg_bytes[SegRole::Weight.segment()] > bundle::MAX_SEGMENT_BYTES);
        let tail = spill_weight_tail(&mut placements, &mut seg_bytes, &per_layer).unwrap();
        assert_eq!(
            tail,
            WeightTail::Spilled {
                segment: 6,
                bytes: 419_430_400
            }
        );
        assert_eq!((placements[&41].segment, placements[&41].offset), (6, 0));
        assert_eq!(placements[&40].segment, SegRole::Weight.segment());
        assert_eq!(weight_tail_of_placements(&placements), tail);
    }

    /// Unpadded (gate/up at 12800), the same model fits one region and nothing moves.
    #[test]
    fn the_unpadded_8b_keeps_its_tail_in_place() {
        let (mut placements, mut seg_bytes, per_layer) =
            granite_8b(423_641_088 - 2 * 4_096 * 1_536 * 2);
        let before = placements.clone();
        let tail = spill_weight_tail(&mut placements, &mut seg_bytes, &per_layer).unwrap();
        assert_eq!(tail, WeightTail::InPlace);
        assert_eq!(weight_tail_of_placements(&placements), WeightTail::InPlace);
        assert!(
            placements
                .iter()
                .all(|(t, p)| (p.segment, p.offset) == (before[t].segment, before[t].offset))
        );
    }
}

/// ⭐⭐ THE KV POOLS, ONE DEVICE REGION PER LAYER — every (class, iteration) launch block's pools
/// are their own bank of the KV segment ([`KvBank`]), at the SAME intra-block offsets for every
/// iteration of the class: each table of the class, in the map's own order, at the running sum of
/// the widest of its layers' aligned sizes. A layer's launch therefore re-bases the KV slot by
/// handing it that layer's region alone — the body bakes the class's intra-block offsets once —
/// and the pool is the model's whole context whatever its total, with no packing against a region
/// size: the largest allocation is one layer's pools.
///
/// The blocks are numbered class-major (bank 0 is the first class's first layer). Every table's
/// layers are placed through the table's own list, which `per_layer_external_tids` files only at
/// its loop's layer count. A pool no per-layer table names goes in bank 0 after that block, where
/// the unshifted prefix and suffix launches read it. Off the card `pools` is empty and nothing is
/// placed.
///
/// Returns the extra banks' bytes (banks `1..N`; bank 0's are written to `seg_bytes`) and each
/// pool table's bank per iteration, keyed by the table's layer-0 tid.
fn place_kv_pools(
    pools: &[u32],
    per_layer_ext: &[std::collections::BTreeMap<u32, Vec<u32>>],
    size_of: impl Fn(u32) -> u64,
    seg_bytes: &mut [u64; 7],
    placements: &mut std::collections::BTreeMap<u32, TensorPlacement>,
) -> (Vec<u64>, std::collections::BTreeMap<u32, Vec<KvBank>>) {
    let kv_seg = SegRole::Kv.segment();
    let is_pool: std::collections::BTreeSet<u32> = pools.iter().copied().collect();
    let mut place = |tid: u32, bank: KvBank, offset: u64| {
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Kv,
                segment: kv_seg,
                bank: bank.get(),
                offset,
                size: size_of(tid),
            },
        );
    };
    let mut bank_bytes: Vec<u64> = Vec::new();
    let mut table_banks: std::collections::BTreeMap<u32, Vec<KvBank>> = Default::default();
    for class in per_layer_ext {
        let tables: Vec<(u32, &Vec<u32>)> = class
            .iter()
            .filter(|(t0, _)| is_pool.contains(t0))
            .map(|(&t0, tids)| (t0, tids))
            .collect();
        let Some(&(_, first)) = tables.first() else {
            continue;
        };
        // Each table's slot is as wide as its widest layer, so no layer's tensor reaches into
        // the next table's slot.
        let mut intra: Vec<u64> = Vec::with_capacity(tables.len());
        let mut block = 0u64;
        for (_, tids) in &tables {
            intra.push(block);
            let widest = tids.iter().map(|&t| size_of(t)).max().unwrap_or(0);
            block = align128(block + widest);
        }
        let banks: Vec<KvBank> = (0..first.len())
            .map(|j| KvBank::of_placement((bank_bytes.len() + j) as u32))
            .collect();
        bank_bytes.extend(std::iter::repeat_n(block, banks.len()));
        for ((t0, tids), &off) in tables.iter().zip(&intra) {
            for (&tid, &bank) in tids.iter().zip(&banks) {
                place(tid, bank, off);
            }
            table_banks.insert(*t0, banks.clone());
        }
    }
    if bank_bytes.is_empty() {
        bank_bytes.push(0);
    }
    let owned: std::collections::BTreeSet<u32> = per_layer_ext
        .iter()
        .flat_map(|class| class.values().flatten().copied())
        .collect();
    let mut tail = bank_bytes[0];
    for &tid in pools.iter().filter(|t| !owned.contains(t)) {
        place(tid, KvBank::BASE, tail);
        tail = align128(tail + size_of(tid));
    }
    bank_bytes[0] = tail;
    seg_bytes[kv_seg] = bank_bytes[0];
    (bank_bytes.split_off(1), table_banks)
}

/// ⭐ THE PER-LAYER EXTERNAL TID CLASSES, READ OFF THE RE-ROLLED TAPE **BEFORE** ANY PLACEMENT
/// EXISTS — one `Vec<u32>` per repeating weight/KV tensor, indexed by layer.
///
/// ⛔ THIS IS WHY IT IS A PRE-PASS AND NOT PART OF THE WALK. The walk that emits the body collects
/// the same fact (`ComputeInput::External::per_layer`), but it collects it while BAKING addresses —
/// far too late for the layout to use it. And the layout is exactly what needs it: a weight BANK
/// boundary may only fall on a LAYER boundary (a launch has one base per segment, so a launch that
/// straddled two banks would be inexpressible), so `compute_bundle_layout` cannot decide banks
/// without knowing which tids are the same tensor in different layers.
///
/// The walk is fed FROM here rather than re-deriving it, so the fact is collected once: the
/// `External` arm downstream extends this map, it does not rebuild it.
/// `pub(crate)` for the SPLIT and nothing else: the caller is the rolled entry point, which now
/// lives in [`crate::lower_subtile_tape_to_ktir`]. Same visibility widening `lower_one_node` took.
pub(crate) fn per_layer_external_tids(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
) -> Vec<std::collections::BTreeMap<u32, Vec<u32>>> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    // ── ONE MAP PER LOOP (attention class), with the loop's OWN iteration count ──
    // A class-split roll's sibling loops carry different-length per-layer tables: gemma-4's
    // sliding body and its global body carry different layer counts, so a table's LENGTH names
    // the only loop it can belong to. Keying on ONE `iters` (the single-class shape this
    // pre-pass had before classes existed) drops every table of the "wrong" length — the layout
    // would then pack those layers as unshared weights, silently. The loop ranges are re-derived
    // here exactly as the lowering walk derives them, so the two never disagree about a class's
    // extent.
    let instrs = tape.instrs();
    let mut loops: Vec<(std::ops::Range<usize>, u32)> = Vec::new();
    {
        let mut open: Option<(usize, u32)> = None;
        for (i, instr) in instrs.iter().enumerate() {
            match instr {
                Instr::OpenLoop { bound, .. } => {
                    debug_assert!(
                        open.is_none(),
                        "the TapeBuilder typestate forbids nested loops"
                    );
                    if let LoopBound::Const(it) = bound {
                        open = Some((i, *it));
                    }
                }
                Instr::CloseLoop { .. } => {
                    if let Some((o, it)) = open.take() {
                        loops.push((o + 1..i, it));
                    }
                }
                _ => {}
            }
        }
    }
    let any_iters: u32 = loops.iter().map(|(_, it)| *it).max().unwrap_or(0);
    if any_iters < 2 {
        return Vec::new(); // nothing repeats: no layer structure, so no banking is expressible
    }
    // A loop's instrs may be indices into the WHOLE tape, so walk instrs once and file each
    // External table under the loop whose `iters` its length names — the only loop it can belong
    // to. (Two sibling loops of the SAME iters share per-layer tables by `tensor` id; the entry is
    // first-wins keyed on that id, and a table's owning body is re-derived downstream from the
    // walk's own body ranges, not from this map.)
    let mut out: Vec<std::collections::BTreeMap<u32, Vec<u32>>> =
        vec![Default::default(); loops.len()];
    for (li, (range, iters)) in loops.iter().enumerate() {
        if *iters < 2 {
            continue;
        }
        for instr in &instrs[range.clone()] {
            if let Instr::Compute { inputs, .. } = instr {
                for ci in inputs.iter() {
                    if let ComputeInput::External {
                        tensor,
                        per_layer: pl,
                        ..
                    } = ci
                        && pl.len() as u32 == *iters
                    {
                        out[li]
                            .entry(tensor.index() as u32)
                            .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
                    }
                }
            }
        }
    }
    out
}

/// ⭐⭐⭐ THE WEIGHT SEGMENT AS A **BANK** OF DEVICE REGIONS — what lifts the 16 GiB ceiling on a
/// model's weights, and the one pass that ever sets [`TensorPlacement::bank`].
///
/// A segment is ONE `FlexAllocator` allocation served from ONE region, and a region is
/// [`bundle::MAX_SEGMENT_BYTES`]; the same 16 GiB is the addressing stride (`SEGMENT_OFFSETS`). That
/// bounded the WEIGHTS of any model at 16 GiB — fp16 dense topped out near 8B — and no segment slot
/// is free to take a second one (0 intermediates, 1 weights, 2 KV, 3 activations, 4 logits, 5+6
/// intermediate COLORS, all occupied; taking a colour is what the `WeightOverflow = 5` attempt got
/// wrong).
///
/// ⭐ BUT 16 GiB IS A LIMIT ON BYTES PER REGION, NOT ON ADDRESSES PER SEGMENT. The rolled body bakes
/// only LAYER 0's offsets and reaches layer `v` by advancing the base it is handed
/// (`off[SEG_WEIGHT] = v·weight_stride`), and `tensor_allocs` is positional per launch with no
/// segment identity in a `DevAddr`. So layer `v` can be handed a DIFFERENT REGION in slot 1 with the
/// same descriptors — the weight segment becomes N regions, and the ceiling becomes N × 16 GiB.
///
/// ⛔ THREE RULES, EACH ONE A FAILURE MODE THAT IS SILENT IF IT IS NOT CHECKED HERE:
///
///  1. **A launch has ONE base per segment**, so a bank boundary may only fall on a LAYER boundary —
///     `lpb` whole layers per bank. A launch straddling two banks cannot be expressed at all.
///  2. **Every layer must keep the SAME intra-layer offsets**, because ONE baked body serves all of
///     them. This pass therefore only ever SUBTRACTS a whole number of layer strides from an
///     existing offset; it never repacks a layer, so `weight_stride` and every relative address
///     survive untouched.
///  3. **The non-per-layer weights go in ONE bank together.** The suffix reads the final norm AND
///     the lm_head; if those two landed in different banks the suffix would be inexpressible. That
///     is the defect the seg6 tail spill hid: it moved only as many trailing tensors as it took to
///     fit, which for granite-3.1-8b-fp16 was the embedding alone, leaving the final norm behind.
///
/// A bank is ALIASABLE, which is the second half of the payoff: it holds only weights, placed
/// identically in every bundle, so `alias_seg_from`'s size + placement equality checks pass and every
/// borrowing session SHARES the owner's regions. The spill slot could not be aliased (its co-tenant
/// intermediate colour is m-dependent), so each of granite-3.1-8b-fp16's 27–28 sessions needed its
/// own 419,430,400 B COPY of the tail — 10.5 GiB of device memory and ~7.6 s of the 25.2 s load.
///
/// Returns the extents of banks `1..N` (empty ⇒ one bank ⇒ nothing moved, and every model whose
/// weights already fit is byte-identical, including its `seg_bytes`).
fn bank_weight_segment(
    placements: &mut std::collections::BTreeMap<u32, TensorPlacement>,
    seg_bytes: &mut [u64; 7],
    per_layer_ext: &[std::collections::BTreeMap<u32, Vec<u32>>],
) -> Result<Vec<u64>, SuperDscError> {
    let w_seg = SegRole::Weight.segment();
    if seg_bytes[w_seg] <= bundle::MAX_SEGMENT_BYTES {
        return Ok(Vec::new()); // the overwhelming case: one region holds every weight.
    }
    // ── The per-layer WEIGHT classes, ONE LIST PER ATTENTION CLASS ──
    // Only classes whose layer-0 tid is a WEIGHT in this segment: `per_layer_ext` also carries the
    // KV caches (seg2), which have their own stride and their own segment.
    //
    // ⭐ THE LISTS KEEP THEIR CLASS BOUNDARIES. A hybrid model's classes have different layer
    // counts AND different per-layer strides (gemma-4's sliding and global layers pack different
    // weight blocks), so the old "every class agrees on the layer count / one stride" preconditions
    // are exactly what a hybrid model violates. The unit of banking is the (class, iteration)
    // launch, not the bare layer index: a bank boundary must fall on one, because one launch has
    // one base per segment. The launch-table proof downstream (`lower_subtile_tape_to_ktir`) holds
    // every placement to the table this pass produces, so a wrong split dies at cargo build.
    let classes: Vec<Vec<Vec<u32>>> = per_layer_ext
        .iter()
        .map(|m| {
            m.values()
                .filter(|tids| {
                    tids.first().is_some_and(|t0| {
                        placements.get(t0).is_some_and(|p| {
                            p.segment == w_seg && matches!(p.role, SegRole::Weight)
                        })
                    })
                })
                .cloned()
                .collect()
        })
        .collect();
    if classes.iter().all(|c| c.is_empty()) {
        return Err(SuperDscError(format!(
            "seg{w_seg} packs {} B, {} B past the {} B one device region can hold, and the tape has \
             NO per-layer weight classes — so there is no layer boundary to split the segment on. \
             Banking a weight segment needs the re-rolled layer loop; an unrolled bundle this large \
             cannot be addressed.",
            seg_bytes[w_seg],
            seg_bytes[w_seg] - bundle::MAX_SEGMENT_BYTES,
            bundle::MAX_SEGMENT_BYTES,
        )));
    }
    // ⭐ THE PACKING ORDER IS THE TRUE LAYER ORDER — classes were packed CONSECUTIVELY, class by
    // class, by the source loop. Banking walks every class's every layer in OFFSET order of that
    // class's layer-0 placements, packs banks greedily to just under the cap, and RE-BASES each
    // (class, iteration) so the launch table's per-launch shifts stay small. One constraint only:
    // all of a launch's weights land in ONE bank, which greedy whole-launch packing guarantees and
    // the launch-table proof re-checks.
    // ⭐ SPAN FACTS TAKE THE MAP AS A PARAMETER, not a capture: a capturing closure would hold the
    // immutable borrow across the re-placement's `get_mut`, and pre-copying every launch's span into
    // a side table would duplicate the walk. The map is threaded explicitly instead.
    let off_of = |m: &std::collections::BTreeMap<u32, TensorPlacement>, t: &u32| -> Option<u64> {
        m.get(t).map(|p| p.offset)
    };
    // Every launch, as the tid list it must keep together: one per (class, iteration), in the
    // class's own iteration order.
    let mut launches: Vec<Vec<u32>> = Vec::new();
    for class in &classes {
        let iters = class.iter().map(|c| c.len()).max().unwrap_or(0);
        if class.iter().any(|c| c.len() != iters) {
            return Err(SuperDscError(
                "bank_weight_segment: one class's per-layer weights disagree on that class's \
                 layer count — the class's own layer boundary is not well defined"
                    .into(),
            ));
        }
        for v in 0..iters {
            launches.push(class.iter().map(|c| c[v]).collect());
        }
    }
    // ⭐ THE REAL PRECONDITION IS *SHIFT AGREEMENT*, NOT A UNIFORM STRIDE. The baked body bakes
    // iteration-0 addresses and the launch table carries ONE `(bank, offset)` per launch, so what
    // must hold is: within a class, `placement_i(t) − placement_0(t)` is the SAME number for every
    // table `t` — every tensor of the class moves by the one shift the launch entry names. A hybrid
    // model's source pack interleaves the classes (gemma-4 packs each 6-layer cell as sliding block
    // + global block, so the GLOBAL class's inter-iteration stride alternates 473,992,192 /
    // 985,470,976 B), which is FINE: the stride varying across iterations is exactly what the
    // per-launch table absorbs. What is NOT fine — and is refused here — is two tables of one class
    // disagreeing about iteration i's shift, because no single launch entry could name both.
    for class in &classes {
        let Some(first) = class.first() else { continue };
        let shifts: Vec<u64> = first
            .iter()
            .map(|t| off_of(placements, t))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                SuperDscError("bank_weight_segment: a per-layer weight has no placement".into())
            })?;
        // `shifts[i] − shifts[0]` is iteration i's shift as the FIRST table lives it; every other
        // table must move its OWN layer-0 offset by exactly that same shift.
        for c in class {
            let Some(base) = off_of(placements, &c[0]) else {
                return Err(SuperDscError(
                    "bank_weight_segment: a per-layer weight has no placement".into(),
                ));
            };
            for (i, t) in c.iter().enumerate() {
                let Some(o) = off_of(placements, t) else {
                    return Err(SuperDscError(
                        "bank_weight_segment: a per-layer weight has no placement".into(),
                    ));
                };
                let want = shifts[i].wrapping_sub(shifts[0]);
                let got = o.wrapping_sub(base);
                if got != want {
                    return Err(SuperDscError(format!(
                        "bank_weight_segment: two tables of one weight class disagree on layer \
                         {i}'s shift ({want} vs {got} B). The baked body carries iteration-0 \
                         addresses and the launch table carries ONE shift per launch, so a class \
                         whose tables shift differently cannot be addressed by any launch table."
                    )));
                }
            }
        }
    }
    // ── Re-place: greedy whole-launch packing into banks, everything else into the tail bank ──
    // Each launch is a RIGID BLOCK: it keeps its intra-launch relative offsets (the baked body's
    // addresses are those relatives plus the launch entry's base) and is re-seated at `cur_end` of
    // the current bank, contiguously with the launches before it — no holes, so a bank's byte total
    // is just where `cur_end` stops.
    let mut bank_bytes: Vec<u64> = vec![0u64];
    let mut cur_bank = 0usize;
    let mut cur_end = 0u64;
    for launch in &launches {
        // Read every offset BEFORE any mutation: the reads borrow the map immutably and the
        // re-placement below needs it mutably.
        let offs: Vec<(u32, u64, u64)> = launch
            .iter()
            .map(|&t| {
                let p = placements.get(&t).expect("tid came from this map");
                (t, p.offset, p.size)
            })
            .collect();
        let lo = offs.iter().map(|(_, o, _)| *o).min().unwrap_or(0);
        let hi = offs
            .iter()
            .map(|(_, o, s)| align128(o + s))
            .max()
            .unwrap_or(0);
        let span = align128(hi.saturating_sub(lo));
        if span > bundle::MAX_SEGMENT_BYTES {
            return Err(SuperDscError(format!(
                "bank_weight_segment: ONE layer's weights span {span} B, past the {} B a single \
                 device region can hold. A bank boundary can only fall on a layer boundary, so \
                 this model cannot be addressed by advancing a per-layer base — it needs the layer \
                 itself split, which the rolled body cannot express.",
                bundle::MAX_SEGMENT_BYTES,
            )));
        }
        if cur_end + span > bundle::MAX_SEGMENT_BYTES {
            // Open the next bank at 0 — a launch never straddles.
            bank_bytes.push(0);
            cur_bank += 1;
            cur_end = 0;
        }
        for (t, o, _sz) in &offs {
            let p = placements.get_mut(t).expect("tid came from this map");
            p.bank = cur_bank as u32;
            p.offset = *o - lo + cur_end;
        }
        cur_end += span;
        bank_bytes[cur_bank] = cur_end;
    }
    // The non-per-layer weights (the head that sorted before layer 0, the final norm, the lm_head /
    // tied embedding) share ONE bank: the last layer bank if they fit in it, else a bank of their
    // own. Sharing costs nothing and saves a region; what matters is that they are TOGETHER, so the
    // suffix — which reads the norm and the lm_head in one launch — needs exactly one base.
    let launched: std::collections::BTreeSet<u32> = launches.iter().flatten().copied().collect();
    let mut tail: Vec<u32> = placements
        .values()
        .filter(|p| {
            p.segment == w_seg && matches!(p.role, SegRole::Weight) && !launched.contains(&p.tid)
        })
        .map(|p| p.tid)
        .collect();
    tail.sort_by_key(|t| placements[t].offset); // keep the packed order the tape produced
    let tail_len: u64 = tail
        .iter()
        .fold(0u64, |acc, t| align128(acc + placements[t].size));
    // An empty tail needs no room, so it "fits" the last layer bank trivially — one condition, not
    // two arms that happen to agree.
    let tail_bank = if tail.is_empty()
        || align128(bank_bytes[bank_bytes.len() - 1]) + tail_len <= bundle::MAX_SEGMENT_BYTES
    {
        bank_bytes.len() - 1
    } else {
        bank_bytes.push(0);
        bank_bytes.len() - 1
    };
    let mut cur = align128(bank_bytes[tail_bank]);
    for tid in &tail {
        let p = placements.get_mut(tid).expect("tid came from this map");
        p.bank = tail_bank as u32;
        p.offset = cur;
        cur = align128(cur + p.size);
        bank_bytes[tail_bank] = cur;
    }
    // ── Prove every bank is allocatable BEFORE anything is baked ──
    for (b, &bytes) in bank_bytes.iter().enumerate() {
        if bytes > bundle::MAX_SEGMENT_BYTES {
            return Err(SuperDscError(format!(
                "bank_weight_segment: weight bank {b} packs {bytes} B, {} B past the {} B one \
                 device region can hold ({} launch(es), {} bank(s)).",
                bytes - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
                launches.len(),
                bank_bytes.len(),
            )));
        }
    }
    seg_bytes[w_seg] = bank_bytes[0];
    Ok(bank_bytes[1..].to_vec())
}

/// The deterministic synthetic source value for BOTH legs of the numeric gate — the golden
/// buffers and the probe sources stage through this ONE function, so the two legs cannot
/// drift. `id` = source tid, `j` = element index.
///
/// ⭐ A HASH, NOT MODULUS ARITHMETIC — the probe network's RANK is the gate's own
/// signal-to-noise, and the old law had none. `((id·131 + j·7) % 197)/197 − 0.5` is a
/// phase-shifted period-197 sawtooth down every row (7 is invertible mod 197), so EVERY
/// weight it synthesizes has rank ≤ 197 — including the 3840×15360 MLP projections — and a
/// zero-mean sawtooth's branch outputs cancel toward zero, leaving rounding noise to
/// dominate what a sandwich-norm arch (gemma-4's post-attention / post-feedforward
/// layernorms) then renormalizes back to unit RMS before the residual add: the probe
/// amplifies its own rounding, measured on gemma-4-12b as a 0.02 floor doubling per layer
/// to O(1) by L23 while granite (pre-norm only) stays flat at 0.0152. A splitmix-style
/// avalanche over `(id, j)` holds no linear recurrence across rows or columns — the
/// synthesized weights are full-rank w.h.p., branch outputs keep real-activation magnitudes,
/// and the floor the gate measures is the EMULATOR's, not the probe's. The output range is
/// the old law's own ±0.05, so every magnitude-calibrated consumer (FP8_SYNTH_SCALE, the
/// payload normalizer, the fold scales) keeps its meaning; the rope sources still stage at
/// their period-`half` contract by calling with `j % half`, which this law satisfies
/// exactly (same `j % half` ⇒ same value).
#[inline]
pub fn dbg_synth_val(id: usize, j: usize) -> f32 {
    let mut z = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 24 bits — f32's own mantissa width — mapped to [−0.05, 0.05), the old law's range.
    ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
}

/// ⭐ THE ROPE COS/SIN SOURCES' STAGING PERIOD — the runtime contract BOTH legs of
/// the numeric gate must stage identically, or the gate convicts its own probe.
/// The bridge tiles the model's `[m, hd]` table across heads into the `[m, total]`
/// source (the splice kernel reads per-head rows `src[h·hd + d]`), and real tables
/// are half-duplicated (`cos[d+half] == cos[d]`, the reference's own second-half
/// read) — together a PERIOD-`head_dim/2` table, under which the reference's flat
/// read `src[d]` and the kernel's per-head read see the SAME values. Flat synth
/// noise breaks that equality: the first rope op then diverges over its whole
/// tensor (heads ≥ 1 read different entries) and every downstream observable
/// inherits it — measured on gemma-4-12b as 716/738 observables refusing at rel
/// 1.2-1.9, head 0's first half the only agreeing run (`h·hd + d == d` at h=0,
/// d<half — an index collision, not a pass). Keyed by source tid; a tid shared by
/// rope nodes of different `head_dim` would be an IR inconsistency (the bridge
/// memoizes tables by `(cols, local)`, so one tid is one `(width, hd)` table).
pub fn rope_table_periods<F: RopeForm>(ir: &SubtileIR<F>) -> std::collections::HashMap<u32, usize> {
    ir.nodes
        .iter()
        .filter_map(|n| n.op.rope_head_dim().map(|d| (n, d.get() as usize / 2)))
        .flat_map(|(n, half)| {
            n.inputs[1..3]
                .iter()
                .map(move |inp| (inp.tensor.index() as u32, half))
        })
        .collect()
}

/// The fp8 payload's synth normalizer — see the payload arm of the gate's weights
/// loop: codes at `synth × C` (e4m3 normals), the magnitude `1/C` riding the
/// ws row, so the dequant meets the reference product at the grid's own ~3%
/// floor instead of the 10-25% subnormal one. File scope because BOTH halves
/// state it: the gate's staging bytes and the golden's e4m3 round trip.
const FP8_SYNTH_SCALE: f32 = 512.0;

/// The prefill rope rows, by the runtime's own law: row r is position r's
/// table — `rope_cos_sin_kind((start + r))` at start == 0, the first chunk
/// this rung exists for — tiled across heads to the source's full width, the
/// same two calls the worker's emu forward binds (`forward_chunk`). ⭐ ONE
/// LAW, BOTH LEGS: `numeric_gate` feeds this product to the emulator's probe
/// binds AND to the golden's source buffers (`eval_golden_buffers`), so a
/// staging defect cannot hide on one side only (measured: staging real rows
/// into the emulator alone refused EVERY layer's RopeRotate/RopeAppend at
/// local ~25 — the golden was still rotating by the period-half synth while
/// the emulator rotated by the real tables). Real per-position rows are what
/// make a row-misaligned kernel DIVERGE; the periodic synth both legs share
/// at decode (one-row tables, where the two laws agree by construction)
/// would hide it on both sides.
pub fn staged_rope_rows<F: RopeForm>(
    ir: &SubtileIR<F>,
    rotary_cos: &[(u32, crate::wiring::RotaryKind)],
    rotary_sin: &[(u32, crate::wiring::RotaryKind)],
) -> Result<std::collections::HashMap<u32, Vec<f32>>, String> {
    let rope_table_half = rope_table_periods(ir);
    let mut out = std::collections::HashMap::new();
    for (classes, want_cos) in [(rotary_cos, true), (rotary_sin, false)] {
        for &(tid, kind) in classes {
            // The class the WIRING names must be the class the tape's rope ops
            // read, or the staged table is the wrong law entirely.
            if let Some(&half) = rope_table_half.get(&tid)
                && kind.head_dim() as usize != 2 * half
            {
                return Err(format!(
                    "prefill gate: cos/sin source t{tid} carries rotary class hd {} but the \
                     rope ops reading it state head_dim {} — the wiring and the tape disagree",
                    kind.head_dim(),
                    2 * half
                ));
            }
            let t = ir.tensors[tid as usize];
            let (rows, cols) = (t.rows as usize, t.cols as usize);
            let chd = kind.head_dim() as usize;
            let mut buf = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let (c, s) = crate::manifest::rope_cos_sin_kind(r as u32, kind);
                let row =
                    crate::forward_tape::tile_rotary_row(if want_cos { &c } else { &s }, cols, chd);
                buf[r * cols..(r + 1) * cols].copy_from_slice(&row);
            }
            out.insert(tid, buf);
        }
    }
    Ok(out)
}

/// The synth weight staging — ONE product, THREE consumers: the emulator's
/// resident bytes (`new_multi` moves them in) and the reference legs' per-node
/// materialization ([`StagedWeightView`]). The staged bytes ARE the round trip
/// the reference used to re-derive per node — an fp8 payload's codes are
/// `e4m3(synth · FP8_SYNTH_SCALE)`, so decoding a code through the e4m3 table
/// is bit-identical to round-tripping the synth value it was minted from; an
/// f16 weight's bytes are its folded image — so a staging defect cannot hide
/// on one leg only. `only` filters the staged set (the covering-depth slice
/// stages just the weights the kept launches read); `None` stages everything.
pub(crate) struct StagedWeights {
    pub bytes: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)>,
    /// The GEMM weights — staged in the on-disk `[cols, rows]` order the
    /// kernel's transpose-B indexing maps read (the manifest's own
    /// `gemm_weight` law; see `numeric_gate`'s staging block for the
    /// measured cost of binding the logical order instead).
    pub gemm: std::collections::HashSet<u32>,
    /// The fp8 scale rows — never materialized by the reference (`eval_node`
    /// ignores the ws row; its multiplier rides the fold lists' `ws_eff`).
    pub scale_rows: std::collections::HashSet<u32>,
}

impl StagedWeights {
    /// The reference legs' zero-copy window over this staging. The window owns
    /// no load-consts (`&[]`): the synth-weight staging never minted any, and
    /// the gate's own view — the one that judges a bundle with load-consts —
    /// states them separately (see [`StagedWeightView::load_consts`]).
    pub(crate) fn view(&self) -> StagedWeightView<'_> {
        StagedWeightView {
            bytes: &self.bytes,
            gemm: &self.gemm,
            scale_rows: &self.scale_rows,
            load_consts: &[],
        }
    }
}

/// The reference legs' window over the staged bytes: the golden and the
/// teacher-forced walk materialize a node's weight operands from the SAME
/// product the emulator binds, per node and transiently — each weight is read
/// by exactly one node, so the reference holds no weight f32 beyond the node
/// it is evaluating (the full-depth staging of every weight as f32 was the
/// gate's memory bomb: ~96 GB for the 12b).
pub(crate) struct StagedWeightView<'a> {
    pub bytes: &'a [(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)],
    pub gemm: &'a std::collections::HashSet<u32>,
    pub scale_rows: &'a std::collections::HashSet<u32>,
    /// The load-time constants — the SAME `(tid, kind)` list the worker's
    /// `bind_load_consts` stages. ⛔ DELIBERATELY NOT part of `bytes`: a weight
    /// is a source whose bytes ARE its value (the per-node `materialize`
    /// substitutes them at every read), while a load-const whose tid is a
    /// graph OUTPUT stages that output's INITIAL bytes — the ops overwrite
    /// them wherever they compute — so the reference legs take them through
    /// [`Self::stage_load_consts`] at buffer init instead, never through the
    /// read-time channel. Each entry is already BUILT (`LoadConst::build`) for
    /// the rank the probe stages, so the view carries no rank of its own.
    pub load_consts: &'a [BuiltConst],
}

/// One load-time constant, built for its staging rank: `(tid, bytes, dtype)`.
pub(crate) type BuiltConst = (u32, Vec<u8>, scratchy_tensors::DType);

impl StagedWeightView<'_> {
    /// The f32 image of staged weight `tid` in the LOGICAL `[rows, cols]`
    /// order `eval_node` reads — `None` for anything the reference does not
    /// materialize: a non-weight, a weight the covering filter dropped, or a
    /// scale row (never read; the fold lists carry its multiplier).
    fn materialize(&self, tid: usize, rows: usize, cols: usize) -> Option<Vec<f32>> {
        if self.scale_rows.contains(&(tid as u32)) {
            return None;
        }
        let (_, bytes, dt, _) = self.bytes.iter().find(|(t, ..)| *t == tid)?;
        let gemm = self.gemm.contains(&(tid as u32));
        let n = rows * cols;
        let mut out = Vec::with_capacity(n);
        match dt {
            scratchy_tensors::DType::Fp8E4m3 => {
                if gemm {
                    for l in 0..rows {
                        for j in 0..cols {
                            out.push(ktir_core::codec::e4m3_to_f32(bytes[j * rows + l]));
                        }
                    }
                } else {
                    out.extend(bytes.iter().map(|&c| ktir_core::codec::e4m3_to_f32(c)));
                }
            }
            _ => {
                let decode = |pair: &[u8]| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32();
                if gemm {
                    for l in 0..rows {
                        for j in 0..cols {
                            out.push(decode(&bytes[(j * rows + l) * 2..]));
                        }
                    }
                } else {
                    for e in 0..n {
                        out.push(decode(&bytes[e * 2..]));
                    }
                }
            }
        }
        Some(out)
    }

    /// ⭐ THE LOAD-CONSTS, STATED AS INITIAL BYTES — the reference legs'
    /// statement of the worker's own `bind_load_consts` staging: the worker
    /// writes each entry's built bytes at its tid's stick BEFORE the programs
    /// run, and the programs overwrite whatever they compute, so the reference
    /// writes the f32 image of the SAME builder's bytes (`LoadConst::build`,
    /// the one derivation both legs call) into `bufs[tid]` once at buffer init
    /// and lets the node walk overwrite it the same way. ⛔ NOT the
    /// `bytes`/`materialize` channel — that substitutes a weight's bytes at
    /// every READ, and an OUTPUT-tid entry reaching it would replace the
    /// computed product with its own initial bytes at every consumer
    /// (measured: the argmax mint's widened row read back as its staged
    /// zeros, golden idx 0 against a true argmax of 12358 on both legs).
    /// `floor` skips entries below it: the teacher-forced walk relays its
    /// source slots from the golden per node, so only its non-source buffers
    /// take initial bytes here; the golden passes 0. An entry whose buffer the
    /// covering slice left unallocated is skipped with it — nothing bound
    /// reads it.
    pub(crate) fn stage_load_consts(&self, bufs: &mut [Vec<f32>], floor: usize) {
        for (tid, bytes, dt) in self.load_consts {
            let (tid, dt) = (*tid, *dt);
            let i = tid as usize;
            if i < floor || i >= bufs.len() || bufs[i].is_empty() {
                continue;
            }
            let vals: Vec<f32> = match dt {
                scratchy_tensors::DType::F16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| half::f16::from_le_bytes(*p).to_f32())
                    .collect(),
                scratchy_tensors::DType::F32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| f32::from_le_bytes(*p))
                    .collect(),
                // The i32 hash/fold constants, as value images. INEXACT past
                // 2^24 by construction (f32 cannot hold the spellings) and
                // INERT: the host reference's Sample arm states its own
                // constants and reads only the row, the seed and the inv_t,
                // so nothing on a host leg consumes these images — the card
                // leg reads the verbatim LE words through `emu_weights`.
                scratchy_tensors::DType::I32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| i32::from_le_bytes(*p) as f32)
                    .collect(),
                other => panic!(
                    "the reference legs have no f32 image for the load-const at t{tid} \
                     staged as {other:?} — extend `stage_load_consts` with that dtype's decode"
                ),
            };
            assert_eq!(
                vals.len(),
                bufs[i].len(),
                "the load-const at t{tid} builds {} values against an IR shape of {} — \
                 the builder and the mint disagree",
                vals.len(),
                bufs[i].len()
            );
            bufs[i] = vals;
        }
    }
}

/// Stage the synth weights (the loop `numeric_gate` ran inline before the
/// truncation rework, extracted so the DBG oracle's golden states the SAME
/// law). `only` = the tids to stage (the covering bound set); `None` = all of
/// `weight_ids`.
fn stage_synth_weights<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    only: Option<&std::collections::HashSet<u64>>,
    // Each weight's synth stream (`dbg_synth_val`'s first argument) — its own tid, except where a
    // sharded gate gives a rank's slice of a sharded weight a stream of its own.
    seed: &dyn Fn(u32) -> usize,
) -> StagedWeights {
    use scratchy_subtile::lower::GemmWeight;
    // ── The fp8 operand classification, off the IR the lowering itself read ──
    let mut fp8_payloads: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut scale_rows: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for node in &ir.nodes {
        match &node.op {
            SubOp::MatmulTile { weight, .. }
                if matches!(weight, GemmWeight::Fp8Dynamic) && node.inputs.len() == 3 =>
            {
                fp8_payloads.insert(node.inputs[1].tensor.index() as u32);
                scale_rows.insert(node.inputs[2].tensor.index() as u32);
            }
            // `[rows, over, weight(, w_scale)]` — same law for the all-rows form.
            SubOp::AllRowsMatmul if node.inputs.len() == 4 => {
                fp8_payloads.insert(node.inputs[2].tensor.index() as u32);
                scale_rows.insert(node.inputs[3].tensor.index() as u32);
            }
            _ => {}
        }
    }
    // ── The fold pre-scales: the same lists the staging states ──
    let sm_folds = crate::lower_subtile_tape_to_ktir::scalar_mul_weight_folds(ir);
    let attn_folds = crate::lower_subtile_tape_to_ktir::attn_scale_weight_folds(ir);
    let fold_scale = |tid: u32| -> f32 {
        sm_folds
            .iter()
            .find(|(_, w, _)| *w == tid)
            .map(|(_, _, s)| *s)
            .or_else(|| {
                attn_folds
                    .iter()
                    .find(|(_, wq, wk, _)| *wq == tid || *wk == tid)
                    .map(|(_, _, _, s)| *s)
            })
            .unwrap_or(1.0)
    };
    // ── The GEMM weight classification — the SAME law the manifest's
    //    `gemm_weight` set stamps (`codegen.rs`: the source a MatmulTile binds
    //    as `inputs[1]`) and
    //    `spyre_load.rs`'s emulator branch stages by: a GEMM weight's Arg is
    //    bound with the SWAPPED shape `[cols, rows]` and bytes in the on-disk
    //    `[n, k]` order the kernel's transpose-B indexing maps read ("the KTIR
    //    emulator path keeps the buffer VERBATIM as `[n, k]`"); every other
    //    weight — norms, the fp8 scale rows — is verbatim at its own
    //    `[rows, cols]`. Binding the logical order instead is not a small
    //    error: the transpose-B maps resolve a DIFFERENT matrix and every GEMM
    //    output diverges at rel ~1 (measured: 518 of 522 granite tensors, all
    //    fp8-chained, the clean prefix passing). ──
    let mut gemm_weights: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for node in &ir.nodes {
        match &node.op {
            SubOp::MatmulTile { .. } => {
                gemm_weights.insert(node.inputs[1].tensor.index() as u32);
            }
            SubOp::AllRowsMatmul => {
                gemm_weights.insert(node.inputs[2].tensor.index() as u32);
            }
            _ => {}
        }
    }
    let mut bytes: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)> =
        Vec::with_capacity(weight_ids.len());
    for &tid in weight_ids {
        if let Some(only) = only
            && !only.contains(&(tid as u64))
        {
            continue;
        }
        let t = ir.tensors[tid as usize];
        let (rows, cols) = (t.rows as usize, t.cols as usize);
        let n = rows * cols;
        let gemm = gemm_weights.contains(&tid);
        let shape = if gemm {
            vec![cols, rows]
        } else {
            vec![rows, cols]
        };
        // The golden's flat order is the LOGICAL `[rows, cols]` row-major
        // (`eval_node` reads `w[l * n + j]`); a GEMM weight's device bytes are the
        // ON-DISK `[cols, rows]` order — `buffer[j * rows + l]` holds `W[l, j]` —
        // so the GEMM arms below walk `l` inner over `j` outer and everything else
        // walks the flat order straight through. The ws rows are ones
        // (permutation-invariant) and keep the verbatim shape, matching the
        // loader's non-GEMM branch.
        let b = if scale_rows.contains(&tid) {
            // fold / FP8_SYNTH_SCALE — the payload's normalizing constant comes
            // back out here (see the payload arm below): the golden never reads
            // this row, the emulator's dequant multiplies by it, so
            // (synth × C) × (fold / C) = the folded ideal. C = 512 keeps BOTH
            // sides out of subnormal range: the payload's smallest nonzero synth
            // value (0.1/197 ≈ 5.1e-4) lands at 0.26 in e4m3 normals, and
            // fold/C stays above f16's normal floor (6.1e-5) for every fold scale
            // the models ship.
            let s = half::f16::from_f32(fold_scale(tid) / FP8_SYNTH_SCALE);
            let mut b = Vec::with_capacity(n * 2);
            for _ in 0..n {
                b.extend_from_slice(&s.to_le_bytes());
            }
            b
        } else if fp8_payloads.contains(&tid) {
            // fp8 folds ride the ws row (a payload cannot absorb a scale — that is
            // a requantize), so the payload carries no fold factor. The codes are
            // the logical values × FP8_SYNTH_SCALE: at C=1 the ±0.05 synth values
            // are e4m3 SUBNORMAL (normals start at 2^-6 = 0.0156 — half the synth
            // range falls below), quantizing at 10-25% rel error instead of the
            // ~3% the 3-bit mantissa gives. The real path never hits that trap:
            // its codes are per-channel scaled into normal range with the
            // magnitude riding the ws row — C restores that shape for the probe,
            // and the ws arm above divides it back out.
            let mut b = Vec::with_capacity(n);
            if gemm {
                for j in 0..cols {
                    for l in 0..rows {
                        b.push(ktir_core::codec::f32_to_e4m3(
                            dbg_synth_val(seed(tid), l * cols + j) * FP8_SYNTH_SCALE,
                        ));
                    }
                }
            } else {
                for e in 0..n {
                    b.push(ktir_core::codec::f32_to_e4m3(
                        dbg_synth_val(seed(tid), e) * FP8_SYNTH_SCALE,
                    ));
                }
            }
            b
        } else {
            let mut b = Vec::with_capacity(n * 2);
            let push = |b: &mut Vec<u8>, v: f32| {
                let f = half::f16::from_f32(v);
                b.extend_from_slice(&f.to_le_bytes());
            };
            if gemm {
                for j in 0..cols {
                    for l in 0..rows {
                        push(
                            &mut b,
                            dbg_synth_val(seed(tid), l * cols + j) * fold_scale(tid),
                        );
                    }
                }
            } else {
                for e in 0..n {
                    push(&mut b, dbg_synth_val(seed(tid), e) * fold_scale(tid));
                }
            }
            b
        };
        let dt = if fp8_payloads.contains(&tid) {
            scratchy_tensors::DType::Fp8E4m3
        } else {
            scratchy_tensors::DType::F16
        };
        bytes.push((tid as usize, b, dt, shape));
    }
    StagedWeights {
        bytes,
        gemm: gemm_weights,
        scale_rows,
    }
}

/// DEBUG-ONLY numeric bisection oracle (codegen gates on `SCRATCHY_SUPERDSC_DBG`).
/// Runs `eval_dag` on the SAME deterministic synthetic sources the worker
/// self-test will bind, and writes, into `<body_bundle_dir>/dbg/`:
///   • `golden/t{tid}.bin` — eval_dag's f32 value for every produced tid,
///   • `order.json`        — `[{i,tid,op,rows,cols}]` in node order (so the
///                            self-test reports the FIRST op that diverges),
///   • `source_shapes.json`— `[{tid,rows,cols}]` for every source (the worker
///                            regenerates the synthetic value via `dbg_synth_val`),
///   • `weight_tids.json`  — the weight tids (the worker routes these into a
///                            synthetic SuperDscSession; the rest are activations).
/// Pre-attention tids (rmsnorm/qkv/rope) depend only on hidden+weights, so the
/// comparison needs NO mask/cache alignment — the bug localizes to one op.
/// A gated run's readback, as the teacher-forced walk restarts from it: the run's own tensors, and —
/// for one rank of a sharded gate — the world whose emulated operands its collectives read.
#[cfg(feature = "spyre-emu")]
#[derive(Clone, Copy)]
struct EmuReadback<'a> {
    own: &'a std::collections::HashMap<u64, Vec<f32>>,
    peer: Option<GatePeer<'a>>,
}

/// Whether `op` reads across ranks — the nodes a rank's reference evaluates against its peer.
fn is_collective<F: RopeForm>(op: &SubOp<F>) -> bool {
    matches!(op, SubOp::AllReduce | SubOp::AllGather { .. })
}

/// THE WORLD of a sharded gate, as one rank's reference reads it: which rank THIS one is and every
/// rank's buffers, indexed by rank then tid like the rank's own, that its collectives read every
/// operand from (in absolute rank order — `eval_collective`'s law). An empty rank's buffers stand
/// for a rank not yet evaluated — the first pass of the lockstep golden, whose collectives the
/// later passes correct — and read as this rank's own.
#[derive(Clone, Copy)]
pub(crate) struct GatePeer<'a> {
    pub rank: u32,
    pub ranks: &'a [Vec<Vec<f32>>],
}

impl GatePeer<'_> {
    fn collective<F: RopeForm>(
        self,
        node: &scratchy_subtile::subtile_ir::SubtileNode<F>,
        ir: &SubtileIR<F>,
        own: &[Vec<f32>],
    ) -> Vec<f32> {
        let ranks: Vec<&[Vec<f32>]> = self
            .ranks
            .iter()
            .enumerate()
            .map(|(r, b)| {
                if r as u32 == self.rank || b.is_empty() {
                    own
                } else {
                    b.as_slice()
                }
            })
            .collect();
        scratchy_subtile::tensor_parallel::eval_collective(node, ir, &ranks)
    }
}

/// ⭐ THE IN-MEMORY HALF OF THE GOLDEN — the reference semantics of the same
/// `dsl_math`, computed independently of the spyre lowering: every tensor's f32
/// values under the synthetic source law (`dbg_synth_val`), with the fold's WHOLE
/// contract stated (pre-scaled fold targets, skipped folded nodes, the reader-half
/// alias, attention folds at scale 1.0) and the fp8 matmul's QUANT MODEL stated
/// too (the e4m3 round trip and the ws mulf the bundle's own program computes —
/// see the closures inside), and the F16 STICK every exchanged value lives on
/// (sources and node outputs both, the fp8 payload excepted — see the staging
/// block). [`write_eval_golden`] writes these to disk
/// for the worker-side reader; the bake-time numeric gate
/// ([`numeric_gate`]) compares them against the emulator's own run IN-PROCESS —
/// the same values, no files — so the gate needs no environment variable and
/// cannot be skipped by forgetting to set one.
///
/// One rank of a sharded forward evaluates its collectives against its `peer`
/// ([`GatePeer`]): the reference walks the rank's own graph, and only a collective
/// reads across ranks.
pub(crate) fn eval_golden_buffers<F: RopeForm>(
    ir: &SubtileIR<F>,
    staged_rope: &std::collections::HashMap<u32, Vec<f32>>,
    // The staged bytes the per-node materialization decodes — the SAME
    // product the emulator binds (see [`StagedWeightView`]). The fold
    // pre-scales the reference used to apply to weight sources live in the
    // bytes now (they were minted with them), so there is no weight f32 in
    // `src_vals` at all: each node materializes its own operands, evals, and
    // drops them.
    weights: &StagedWeightView<'_>,
    // The covering bound set — `None` evaluates the whole graph (the DBG
    // oracle's full-depth golden); `Some` skips every node whose output the
    // truncated bundle does not bind, and leaves every non-bound tensor's
    // buffer unallocated.
    covering: Option<&std::collections::HashSet<u64>>,
    // One rank of a sharded forward: the peer whose buffers its collectives read. `None` for a
    // forward with no collective, where meeting one panics (`eval_node`'s own refusal).
    peer: Option<GatePeer<'_>>,
    // Each synth source's stream — `stage_gate_probe`'s own `seed`, so the reference reads exactly
    // the values the session was handed.
    seed: &dyn Fn(u32) -> usize,
) -> Vec<Vec<f32>> {
    let nsrc = ir.num_sources as usize;
    // ⭐ THE FOLD, STATED FOR THE GOLDEN TOO — BOTH KINDS. A weight-folded ScalarMul has no
    // device op, so the bundle under test never applies its multiplier as a node — the scale
    // enters through the staged weight bytes. The golden must model exactly that or every
    // comparison diverges at the first folded node: the bytes carry the pre-scaled product
    // (what `stage_bound_weights` does to the real bytes) and the eval walk SKIPS the folded
    // nodes (what `lower_one_node`'s HostRouted arm does to the tape). An attention-scale
    // fold's node is NOT skipped — it still runs, at the scale the splice's rewrite gave the
    // real program (1.0) — so the golden evaluates it at 1.0 over the pre-scaled W_q/W_k
    // (in the staged bytes), which is bit-for-bit the folded bundle's contract.
    let scalarmul_folds = crate::lower_subtile_tape_to_ktir::scalar_mul_weight_folds(ir);
    let attn_folds = crate::lower_subtile_tape_to_ktir::attn_scale_weight_folds(ir);
    let folded_node = |id: scratchy_subtile::subtile_ir::SubtileId| {
        scalarmul_folds.iter().any(|(fold_id, _, _)| *fold_id == id)
    };
    // ⭐⭐ THE FOLD'S READER HALF — the alias the card door's layout states and the bundle's
    // programs bind directly (the consumer rewrite in `lower_one_node`): a reader of the
    // folded node's output reads the PRODUCER'S bytes. Skipping the folded node alone leaves
    // its output buffer at the zero-init, and every comparison at the fold's consumers
    // diverges by exactly the folded term — a golden that models the bundle's bug as its
    // contract. The producer's product is scattered into the folded tid too (below), so the
    // golden states the fold's WHOLE contract, both halves.
    let fold_alias: Vec<_> = scalarmul_folds
        .iter()
        .map(|(id, _, _)| {
            let folded = &ir.nodes[id.index()];
            (
                folded.inputs[0].tensor.index() as u32,
                folded.output.tensor,
                folded.output.region,
            )
        })
        .collect();
    // The rope cos/sin sources stage at their runtime contract: a PREFILL
    // rung gets the REAL per-position rows (`staged_rope` — the same product
    // the probe binds, so the two legs cannot drift; see that fn's doc), and
    // decode keeps the period-`half` pattern per [`rope_table_periods`] —
    // there the tables are one row and the pattern agrees with the run-time
    // staging by construction.
    let rope_table_half = rope_table_periods(ir);
    // ⭐ WEIGHT SOURCES STAY EMPTY — materialized per node from the staged
    // bytes (see `weights`), never held as f32 beyond the node reading them;
    // a source the covering slice does not bind stays empty with its readers
    // skipped below. The fold pre-scales the reference used to apply to
    // weight sources live in the bytes now (they were minted with them), so
    // there is nothing to pre-scale here.
    let staged_weight = |id: usize| {
        weights.scale_rows.contains(&(id as u32)) || weights.bytes.iter().any(|(t, ..)| *t == id)
    };
    // ⭐ A SOURCE IS MATERIALIZED AS FAR AS ANY NODE READS IT. A paged KV pool is the model's whole
    // context, and the step under test reads a few pages of it; the rows past the last one any
    // node's region reaches are no reader's (`gather` addresses by region), so they are not built.
    let mut read_rows: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for n in &ir.nodes {
        for r in &n.inputs {
            let end = r.region.rows.start + r.region.rows.len;
            let e = read_rows.entry(r.tensor.index()).or_insert(0);
            *e = (*e).max(end);
        }
    }
    let mut src_vals: Vec<Vec<f32>> = (0..nsrc)
        .map(|id| {
            if staged_weight(id) || covering.is_some_and(|c| !c.contains(&(id as u64))) {
                Vec::new()
            } else if let Some(v) = staged_rope.get(&(id as u32)) {
                v.clone()
            } else {
                let t = ir.tensors[id];
                let rows = read_rows.get(&id).copied().unwrap_or(t.rows).min(t.rows);
                let n = (rows as usize) * (t.cols as usize);
                match rope_table_half.get(&(id as u32)) {
                    Some(&half) => (0..n)
                        .map(|j| dbg_synth_val(seed(id as u32), j % half))
                        .collect(),
                    None => (0..n).map(|j| dbg_synth_val(seed(id as u32), j)).collect(),
                }
            }
        })
        .collect();
    // ⭐⭐ THE F16 STICK, STATED FOR THE GOLDEN TOO. Every value the bundle's
    // programs exchange lives on an f16 stick — `run_step` binds each source
    // as `DType::F16` and every intermediate's segment is f16 — so the
    // emulator's arithmetic is f32 over f16-grid operands with an f16-grid
    // result. The golden walks f32 buffers, and the per-hop half-ulp residue
    // (~1.4e-4 relative) is invisible on any single tensor but ACCUMULATES
    // down the residual chain onto near-zero streams, where it lands as rel
    // ~1e-2 — measured on granite decode after the mask law: the last 10 of
    // 522 sat just over the fp8 floor with g and e equal to every printed
    // digit. Rounding the sources (the sticks the probe binds) and every
    // node output (the stick the program stores) puts the golden on the same
    // grid hop by hop. The fp8 payloads are EXEMPT: their stick is e4m3, and
    // the quant model's round trip below already states exactly that grid —
    // pre-rounding here would round-trip the wrong value.
    let fp8_payloads: std::collections::HashSet<usize> = ir
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            SubOp::MatmulTile {
                weight: scratchy_subtile::lower::GemmWeight::Fp8Dynamic,
                ..
            } if n.inputs.len() == 3 => Some(n.inputs[1].tensor.index()),
            _ => None,
        })
        .collect();
    for (id, vals) in src_vals.iter_mut().enumerate() {
        if !fp8_payloads.contains(&id) {
            for v in vals.iter_mut() {
                *v = half::f16::from_f32(*v).to_f32();
            }
        }
    }
    let src_refs: Vec<&[f32]> = src_vals.iter().map(|v| v.as_slice()).collect();
    // ⭐⭐ THE fp8 MATMUL'S QUANT MODEL — the emulator's own arithmetic, stated once so
    // the golden and the bundle agree at the GRID's precision instead of at a guessed
    // floor. The program computes `o[m,n] = (Σ_k a[m,k] · e4m3(w[n,k])) · ws[n]` — the
    // fp8 kernel's own law (`matmul_fp8.py`: the widened code read, the F32
    // wider-accumulate dot, the spelled `p * ws` epilogue; the emulator executes
    // exactly that — the CARD's per-token activation quantize is a door chain the
    // program never states, so the model needs no activation half). The probe stages
    // the codes as `e4m3(synth · FP8_SYNTH_SCALE)` and the ws row as
    // `f16(fold / FP8_SYNTH_SCALE)`, so the model is two round trips — the payload
    // through the e4m3 grid, the ws through f16 — and the FOLD rides the same mulf the
    // bundle's dequant applies (an sm-fold's residual scale, an attention fold's √s).
    // That is also what makes the PRODUCER's own tid carry the folded product: the
    // reader-half alias below then scatters it verbatim, matching the rewritten
    // consumer the bundle binds. `ws` is per output COLUMN; the probe's rows are one
    // constant, so the mulf is a scalar here.
    let ws_fold = |tid: u32| -> f32 {
        scalarmul_folds
            .iter()
            .find(|(_, w, _)| *w == tid)
            .map(|(_, _, s)| *s)
            .or_else(|| {
                attn_folds
                    .iter()
                    .find(|(_, wq, wk, _)| *wq == tid || *wk == tid)
                    .map(|(_, _, _, s)| *s)
            })
            .unwrap_or(1.0)
    };
    let e4m3_rt =
        |v: f32| ktir_core::codec::e4m3_to_f32(ktir_core::codec::f32_to_e4m3(v * FP8_SYNTH_SCALE));
    // eval_dag minus the folded nodes — the same per-node `eval_node`/`scatter` pair it runs,
    // with the fold's nodes skipped so the golden describes the FOLDED bundle's semantics.
    // A tensor the covering slice does not bind keeps NO buffer (its readers
    // are skipped with it); an empty source slot stays empty.
    let mut bufs: Vec<Vec<f32>> = ir
        .tensors
        .iter()
        .enumerate()
        .map(|(t, meta)| {
            if covering.is_some_and(|c| !c.contains(&(t as u64))) {
                Vec::new()
            } else {
                vec![0f32; (meta.rows * meta.cols) as usize]
            }
        })
        .collect();
    for (s, src) in src_refs.iter().enumerate() {
        if src.is_empty() {
            bufs[s] = Vec::new();
        } else {
            bufs[s].copy_from_slice(src);
        }
    }
    // ⭐ THE LOAD-CONSTS' INITIAL BYTES — the same pre-program staging the
    // resident performs (`stage_load_consts`' doc is the law): written once
    // at buffer init, overwritten by the node walk wherever ops compute. A
    // source-tid entry (the staged tables) lands here over its synth slot —
    // its bytes are its value, and no op ever writes it.
    weights.stage_load_consts(&mut bufs, 0);
    // ⭐ THE OFF-f16 STICKS — the outputs whose writer's own kernel signature
    // states a non-f16 stick (the argmax mint's i32 idx): the rounding hop
    // below is the F16 STICK's law, and pushing a whole-number index through
    // the f16 grid MOVES the value itself (12358 → 12360 at the grid's
    // spacing 8 in [8192, 16384) — measured). Stated once here so the loop's
    // guard is a set lookup, never a per-node signature read.
    let off_f16_sticks: std::collections::HashSet<u64> = ir
        .nodes
        .iter()
        .filter(|n| !crate::lower_subtile_tape_to_ktir::node_output_on_f16_stick(n, ir))
        .map(|n| n.output.tensor.index() as u64)
        .collect();
    for node in &ir.nodes {
        if folded_node(node.id) {
            continue;
        }
        // The covering slice: a node whose output the truncated bundle does
        // not bind is not part of the graph under test — skip it (its inputs
        // and outputs stay unallocated; nothing bound reads them).
        if covering.is_some_and(|c| !c.contains(&(node.output.tensor.index() as u64))) {
            continue;
        }
        // ⭐ THE PER-NODE WEIGHT MATERIALIZATION — the node's weight operands
        // decoded from the staged bytes (the emulator's own product) for
        // exactly this eval, then dropped: each weight is read by exactly one
        // node, so the reference never holds a weight's f32 beyond the node
        // it is evaluating. A staged weight's slot is empty by construction
        // (weights never enter the source staging), and `materialize` is
        // `None` for everything else — non-weights keep their staged values.
        let mut saved_weights: Vec<(usize, Vec<f32>)> = Vec::new();
        for r in &node.inputs {
            let w = r.tensor.index();
            let (rows, cols) = {
                let t = ir.tensors[w];
                (t.rows as usize, t.cols as usize)
            };
            if let Some(vals) = weights.materialize(w, rows, cols)
                && saved_weights.iter().all(|(t, _)| *t != w)
            {
                saved_weights.push((w, std::mem::take(&mut bufs[w])));
                bufs[w] = vals;
            }
        }
        // An attention-folded node evaluates at the REWRITTEN scale (1.0): the real program
        // the golden must track was minted from the rewritten node, and evaluating the
        // original scale over the pre-scaled sources would double-apply the multiplier.
        //
        // ⭐⭐ EVERY OTHER REGISTRY SCALAR IS STAGED AT ITS f16 IMAGE — one law with the
        // census, stated once in [`stage_registry_scalars`] (its doc is the law's own):
        // the eps, the ScalarMul multipliers, the attention score scale, the softcap,
        // the router's eps and route scale. The mean and meps stay exact f32 here — the
        // card's own DLF16 landings for those sticks are the EMULATOR's model
        // (`comm_sched::rmsnorm_scalar_sticks` → `Tile::compute_dlf16`), never the
        // reference's.
        let mut staged_node;
        // ⭐ THE SCALE AT ITS STATED SITE, the same typed [`AttnScaleSite`] the splice's SCALE
        // constexpr and the scalar registry read: a weight-folded attention evaluates at
        // 1.0 in-kernel (its √scale rides the staged W_q/W_k), and evaluating the model's
        // own scale over the pre-scaled sources would double-apply the multiplier. The
        // model's own `Attention` is passed through untouched.
        let (node_ref, scale_site) = if let SubOp::AttnDecode { .. } = &node.op
            && attn_folds.iter().any(|(id, ..)| *id == node.id)
        {
            (
                node,
                scratchy_subtile::subtile_ir::AttnScaleSite::FoldedIntoWeights,
            )
        } else {
            staged_node = node.clone();
            if stage_registry_scalars(&mut staged_node.op) {
                (
                    &staged_node,
                    scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
                )
            } else {
                (node, scratchy_subtile::subtile_ir::AttnScaleSite::InKernel)
            }
        };
        let mut out = match peer {
            Some(p) if is_collective(&node_ref.op) => p.collective(node_ref, ir, &bufs),
            _ => scratchy_subtile::subtile_ir::eval_node(node_ref, ir, scale_site, &bufs),
        };
        // The fp8 matmul's quant model (see the closures above): the ideal product over
        // the ROUND-TRIPPED payload, then the ws mulf — fold included — on the result.
        // The payload swap is temporary and per node: `eval_node` reads `inputs[1]` and
        // ignores the ws row, so the round trip is the only change it can see. A payload
        // materialized above IS the round trip already (the staged codes are
        // `e4m3(synth·C)`; decoding one through the table is bit-identical to
        // `e4m3_rt` on the raw synth) — re-tripping it would quantize the decoded
        // value a second time at the same scale, so only an UNstaged payload (one
        // the byte staging does not own) takes the round-trip re-eval.
        if let SubOp::MatmulTile {
            weight: scratchy_subtile::lower::GemmWeight::Fp8Dynamic,
            ..
        } = &node_ref.op
            && node_ref.inputs.len() == 3
        {
            let w_tid = node_ref.inputs[1].tensor.index();
            if !saved_weights.iter().any(|(t, _)| *t == w_tid) {
                let saved = std::mem::take(&mut bufs[w_tid]);
                bufs[w_tid] = saved.iter().map(|v| e4m3_rt(*v)).collect();
                out = scratchy_subtile::subtile_ir::eval_node(node_ref, ir, scale_site, &bufs);
                bufs[w_tid] = saved;
            }
            let ws_eff = half::f16::from_f32(
                ws_fold(node_ref.inputs[2].tensor.index() as u32) / FP8_SYNTH_SCALE,
            )
            .to_f32();
            for v in out.iter_mut() {
                *v *= ws_eff;
            }
        }
        // The f16 stick the program stores this node's output through (see
        // the staging block above for the law) — EXCEPT an output whose own
        // signature states a non-f16 stick (`off_f16_sticks`): the idx stick
        // is the kernel's i32 buffer, and its whole numbers are exact; the
        // f16 hop would MOVE them (12358 → 12360, measured). The fold's
        // reader-half alias below scatters the SAME product — the folded tid
        // holds the producer's stick, grid included.
        if !off_f16_sticks.contains(&(node.output.tensor.index() as u64)) {
            for v in out.iter_mut() {
                *v = half::f16::from_f32(*v).to_f32();
            }
        }
        let shape = ir.shape(node.output.tensor);
        scratchy_subtile::subtile_ir::scatter(
            &mut bufs,
            node.output.tensor,
            node.output.region,
            &out,
            shape,
        );
        // The producer of a folded ScalarMul: its product ALSO lands in the folded node's
        // output tid, the reader half of the fold's contract (`fold_alias` above). The fold
        // refuses a shared producer output, so at most one alias matches here.
        if let Some((_, folded_tid, folded_region)) = fold_alias
            .iter()
            .find(|(producer, _, _)| *producer == node.output.tensor.index() as u32)
        {
            // Verbatim: the producer's product already carries the fold (the dense
            // pre-scale on its weight, the quant model's `ws_eff` mulf on its output),
            // so the folded tid holds exactly the value the rewritten consumer binds.
            scratchy_subtile::subtile_ir::scatter(
                &mut bufs,
                *folded_tid,
                *folded_region,
                &out,
                shape,
            );
        }
        // The materialized weight slots go back to empty — the reference holds
        // no weight f32 past the node that read it.
        for (w, saved) in saved_weights {
            bufs[w] = saved;
        }
    }
    bufs
}

/// One row of the teacher-forced split the gate judges and a refusal dumps:
/// how much of a tensor's divergence its PRODUCING OP added itself, versus
/// what its inputs' drift alone predicts.
#[cfg(feature = "spyre-emu")]
pub(crate) struct TeacherForcedRow {
    pub tid: u64,
    /// The producing node's index in `ir.nodes` — a tid the reroll writes
    /// through several nodes carries one row per writer.
    pub node: usize,
    pub op: &'static str,
    pub fp8: bool,
    /// `rel(e_out, golden_op(e_in))` — the error this op's own execution
    /// added, its inputs pinned to the emulator's readbacks. An op class
    /// whose local error sits at its chain's floor while the TOTAL rel grows
    /// is propagating, not defecting.
    pub local: f32,
    /// `rel(golden_op(e_in), g_out)` — the divergence the inputs' drift
    /// alone predicts: what the op would have produced had it executed
    /// exactly. `local ≈ 0 ∧ inherited ≈ total` is the pure-propagation
    /// signature; anything else names a suspect.
    pub inherited: f32,
}

/// One restart verdict: the emulator's state is overlaid up to a boundary
/// (a fraction of the compare order), then the GOLDEN's semantics propagate
/// to the end untouched. `track` is the restart's distance from the emu at
/// the deep tail — small means the emu's later arithmetic is faithful and
/// the network itself amplifies the boundary difference; `contract` is its
/// distance from the plain golden — a restart that contracts back toward
/// the golden indicts the emu's later ops as the divergence's source.
#[cfg(feature = "spyre-emu")]
pub(crate) struct RestartVerdict {
    pub from_frac: f32,
    pub from_tid: u64,
    pub tail_n: usize,
    pub track_med: f32,
    pub contract_med: f32,
    pub track_last: f32,
    pub contract_last: f32,
    /// `rel(e, g)` at the last compared tid — the observed divergence the
    /// track/contract numbers are read against.
    pub direct_last: f32,
}

/// The split evidence a refusal carries, bundled so the dump writer takes
/// one argument for all of it: the per-op local/inherited rows and the
/// restart verdicts.
#[cfg(feature = "spyre-emu")]
pub(crate) struct RefusalEvidence<'a> {
    pub split: &'a [TeacherForcedRow],
    pub restarts: &'a [RestartVerdict],
}

/// The gate's own rel, stated once for the split: L1 over L1, identical to
/// the comparison loop's arithmetic so the split's numbers join the
/// verdicts' without a conversion.
#[cfg(feature = "spyre-emu")]
fn rel_l1(e: &[f32], g: &[f32]) -> f32 {
    let n = e.len().min(g.len());
    let dsum: f32 = e
        .iter()
        .zip(g.iter())
        .take(n)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let gsum: f32 = g.iter().take(n).map(|x| x.abs()).sum();
    dsum / gsum.max(1e-6)
}

/// The region-ordered slice of a full-tensor readback — [`scatter`]'s own
/// indexing, read instead of written, so an emulator readback (a whole
/// tensor) can stand in for a region-sized product at the same offsets.
#[cfg(feature = "spyre-emu")]
fn region_slice(
    full: &[f32],
    region: scratchy_subtile::subtile_ir::Region,
    shape: scratchy_subtile::subtile_ir::TensorShape,
) -> Vec<f32> {
    let (r, c) = (region.rows.len as usize, region.cols.len as usize);
    let mut out = Vec::with_capacity(r * c);
    for i in 0..r {
        let base =
            (region.rows.start as usize + i) * shape.cols as usize + region.cols.start as usize;
        out.extend_from_slice(&full[base..base + c]);
    }
    out
}

/// ⭐⭐ THE GOLDEN WALK'S TEACHER-FORCED TWIN — the gate's verdict instrument
/// (every gate, pass or refuse), run at the golden's OWN memory cost: the
/// walk holds no copy
/// of any source and recomputes no golden product. Each compared tensor's
/// producing op is evaluated over a buffer array (`tf`) whose every
/// teacher-forced tensor holds the EMULATOR'S readback instead of the
/// golden's product, and the split falls out against the golden's OWN
/// stored slots: `local` = the op's own error on the emulator's inputs,
/// `inherited` = the divergence the inputs' drift alone predicts, read
/// straight from `eval_golden_buffers`'s product for that node's region —
/// the plain side IS the golden, by construction, so the split cannot
/// drift from the values the gate judged. The sources are the golden's own
/// staged values, SHARED zero-copy: each node's eval pointer-swaps the
/// whole source block golden → tf → golden, so no second weight copy ever
/// exists — the 12b's base gate peaks near 111 GiB, and a walk that
/// regenerated or cloned its sources would push the card pod past its
/// 200 GiB container limit mid-refusal (the 31b's first gate run died
/// exactly there).
///
/// ⛔ KEEP IN STEP with `eval_golden_buffers`' walk: everything downstream
/// of the sources — the staging arms, the fold's skip and reader-half
/// alias, the fp8 quant model, the f16 stick — must stay its twin, because
/// the `inherited` column reads the golden's slots as this node's plain
/// product. The overlay applies per node AFTER the row is recorded, so the
/// row's `tf_out` is always the golden's semantics over the emulator's
/// PRIOR state — never over the emulator's own output for this tid.
#[cfg(feature = "spyre-emu")]
fn teacher_forced_walk<F: RopeForm>(
    ir: &SubtileIR<F>,
    golden: &mut [Vec<f32>],
    run: EmuReadback<'_>,
    tainted: &std::collections::HashSet<u32>,
    // The lm-head tail fold's tids (`is_prefill_lm_head_tail` — the splice's
    // own predicate): their program product is ONE row (row 0 of a stick the
    // IR sizes at [mq, cols] and the program never fills past row 0), so
    // their row states the fold's law — the walk's LAST row against the
    // emulator's row 0 — never the full-stick plain law.
    lm_head_folded: &std::collections::HashSet<u64>,
    // The staged bytes the walk materializes each node's weight operands
    // from (same view, same law as `eval_golden_buffers`), and the covering
    // slice — a node whose output the truncated bundle does not bind is not
    // part of the graph under test.
    weights: &StagedWeightView<'_>,
    covering: Option<&std::collections::HashSet<u64>>,
) -> (Vec<Vec<f32>>, Vec<TeacherForcedRow>) {
    let EmuReadback { own: emu, peer } = run;
    let nsrc = ir.num_sources as usize;
    let scalarmul_folds = crate::lower_subtile_tape_to_ktir::scalar_mul_weight_folds(ir);
    let attn_folds = crate::lower_subtile_tape_to_ktir::attn_scale_weight_folds(ir);
    let folded_node = |id: scratchy_subtile::subtile_ir::SubtileId| {
        scalarmul_folds.iter().any(|(fold_id, _, _)| *fold_id == id)
    };
    let fold_alias: Vec<_> = scalarmul_folds
        .iter()
        .map(|(id, _, _)| {
            let folded = &ir.nodes[id.index()];
            (
                folded.inputs[0].tensor.index() as u32,
                folded.output.tensor,
                folded.output.region,
            )
        })
        .collect();
    let ws_fold = |tid: u32| -> f32 {
        scalarmul_folds
            .iter()
            .find(|(_, w, _)| *w == tid)
            .map(|(_, _, s)| *s)
            .or_else(|| {
                attn_folds
                    .iter()
                    .find(|(_, wq, wk, _)| *wq == tid || *wk == tid)
                    .map(|(_, _, _, s)| *s)
            })
            .unwrap_or(1.0)
    };
    let e4m3_rt =
        |v: f32| ktir_core::codec::e4m3_to_f32(ktir_core::codec::f32_to_e4m3(v * FP8_SYNTH_SCALE));
    // The walk's own array holds ONLY the walk's products: every source slot
    // stays EMPTY and is pointer-swapped in from `golden` per node (see the
    // loop), and the golden's own products are read from `golden` itself. A
    // tensor the covering slice does not bind keeps no buffer either — its
    // producers are skipped below.
    let mut tf: Vec<Vec<f32>> = ir
        .tensors
        .iter()
        .enumerate()
        .map(|(t, meta)| {
            if t < nsrc || covering.is_some_and(|c| !c.contains(&(t as u64))) {
                Vec::new()
            } else {
                vec![0f32; (meta.rows * meta.cols) as usize]
            }
        })
        .collect();
    // The load-consts' initial bytes, same law as the golden's init call —
    // floored at `nsrc`: this walk's source slots are relayed from the golden
    // per node (they already carry their staged bytes there), so only its
    // non-source buffers take initial bytes here.
    weights.stage_load_consts(&mut tf, nsrc);
    // The off-f16 sticks, the golden's own law verbatim (`eval_golden_buffers`'s
    // set above is the statement): the rounding hop below must skip an output
    // whose signature states a non-f16 stick, or the walk's OWN local leg lies
    // about the op (the idx stick measured 12358 → 12360 through the grid).
    let off_f16_sticks: std::collections::HashSet<u64> = ir
        .nodes
        .iter()
        .filter(|n| !crate::lower_subtile_tape_to_ktir::node_output_on_f16_stick(n, ir))
        .map(|n| n.output.tensor.index() as u64)
        .collect();
    let mut rows: Vec<TeacherForcedRow> = Vec::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        if folded_node(node.id) {
            continue;
        }
        // The covering slice, same law as the golden's walk: a node whose
        // output the truncated bundle does not bind is not part of the graph
        // under test.
        if covering.is_some_and(|c| !c.contains(&(node.output.tensor.index() as u64))) {
            continue;
        }
        let mut staged_node;
        // A weight-folded attention evaluates at its stated site (1.0), the same typed law
        // the golden's own walk above states.
        let (node_ref, scale_site) = if let SubOp::AttnDecode { .. } = &node.op
            && attn_folds.iter().any(|(id, ..)| *id == node.id)
        {
            (
                node,
                scratchy_subtile::subtile_ir::AttnScaleSite::FoldedIntoWeights,
            )
        } else {
            staged_node = node.clone();
            if stage_registry_scalars(&mut staged_node.op) {
                (
                    &staged_node,
                    scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
                )
            } else {
                (node, scratchy_subtile::subtile_ir::AttnScaleSite::InKernel)
            }
        };
        // The source relay: the golden's staged sources move into `tf` for
        // this node's eval and move back before anything else can read
        // `golden` — pointer swaps, never copies, so the walk's weight
        // footprint is zero. The golden's WEIGHT slots arrive empty (its own
        // walk materialized per node and dropped), so this node's weight
        // operands materialize here the same way — the same staged bytes,
        // the same per-node transience.
        for s in 0..nsrc {
            std::mem::swap(&mut golden[s], &mut tf[s]);
        }
        let mut saved_weights: Vec<(usize, Vec<f32>)> = Vec::new();
        for r in &node.inputs {
            let w = r.tensor.index();
            let (rows, cols) = {
                let t = ir.tensors[w];
                (t.rows as usize, t.cols as usize)
            };
            if let Some(vals) = weights.materialize(w, rows, cols)
                && saved_weights.iter().all(|(t, _)| *t != w)
            {
                saved_weights.push((w, std::mem::take(&mut tf[w])));
                tf[w] = vals;
            }
        }
        let mut tf_out = match peer {
            Some(p) if is_collective(&node_ref.op) => p.collective(node_ref, ir, &tf),
            _ => scratchy_subtile::subtile_ir::eval_node(node_ref, ir, scale_site, &tf),
        };
        if let SubOp::MatmulTile {
            weight: scratchy_subtile::lower::GemmWeight::Fp8Dynamic,
            ..
        } = &node_ref.op
            && node_ref.inputs.len() == 3
        {
            let w_tid = node_ref.inputs[1].tensor.index();
            // Same law as the golden's arm: a materialized payload IS the
            // round trip already; only an unstaged payload re-evaluates.
            if !saved_weights.iter().any(|(t, _)| *t == w_tid) {
                let saved = std::mem::take(&mut tf[w_tid]);
                tf[w_tid] = saved.iter().map(|v| e4m3_rt(*v)).collect();
                tf_out = scratchy_subtile::subtile_ir::eval_node(node_ref, ir, scale_site, &tf);
                tf[w_tid] = saved;
            }
            let ws_eff = half::f16::from_f32(
                ws_fold(node_ref.inputs[2].tensor.index() as u32) / FP8_SYNTH_SCALE,
            )
            .to_f32();
            for v in tf_out.iter_mut() {
                *v *= ws_eff;
            }
        }
        // The materialized slots drop BEFORE the relay-out, so the swap moves
        // empty ↔ empty and no weight f32 outlives the node that read it.
        for (w, saved) in saved_weights {
            tf[w] = saved;
        }
        for s in 0..nsrc {
            std::mem::swap(&mut golden[s], &mut tf[s]);
        }
        // The f16 stick law, the golden's own hop verbatim — and the same
        // exemption: a non-f16 output stick skips the grid (the idx stick's
        // whole numbers are exact; the hop would move them).
        if !off_f16_sticks.contains(&(node.output.tensor.index() as u64)) {
            for v in tf_out.iter_mut() {
                *v = half::f16::from_f32(*v).to_f32();
            }
        }
        let shape = ir.shape(node.output.tensor);
        scratchy_subtile::subtile_ir::scatter(
            &mut tf,
            node.output.tensor,
            node.output.region,
            &tf_out,
            shape,
        );
        let tid = node.output.tensor.index() as u64;
        // A readback shorter than the full tensor (a truncated dump row) has no
        // region-true slice to offer — skip the row and the overlay rather than
        // panic on the region math.
        let full_len = shape.rows as usize * shape.cols as usize;
        if let Some(e) = emu.get(&tid)
            && e.len() == full_len
        {
            let e_region = region_slice(e, node.output.region, shape);
            if e_region.len() == tf_out.len() {
                // The golden's OWN product for this node's region — the plain
                // side of the split, read from the slots the gate judged.
                let g_region = region_slice(
                    &golden[node.output.tensor.index()],
                    node.output.region,
                    shape,
                );
                // ⭐ A FOLDED tail tid states the FOLD'S OWN LAW here too: the
                // program's product is S rows (rows 0..S of the readback —
                // the stick above them is sized by the IR and never written;
                // every golden today is one request, so S = 1 and the product
                // is row 0 alone), so the row compares the emulator's row 0
                // against this walk's LAST
                // row (the `selector_lastrow_col` extraction the lowering
                // pins). The teacher still pins — the fold's own pin: the
                // program wrote S rows starting at row 0, the rows the fold's
                // consumers read, so the emulator's row 0 is pinned there and
                // rows 1.. stay as the walk computed them (the program never
                // wrote them, and the fold law of every downstream folded tid
                // reads the walk's LAST row).
                let rcols = node.output.region.cols.len as usize;
                if lm_head_folded.contains(&tid)
                    && rcols > 0
                    && tf_out.len() % rcols == 0
                    && e_region.len() >= rcols
                {
                    let e_row = &e_region[..rcols];
                    let tf_row = &tf_out[tf_out.len() - rcols..];
                    let g_row = &g_region[g_region.len() - rcols..];
                    rows.push(TeacherForcedRow {
                        tid,
                        node: ni,
                        op: node.op.name(),
                        fp8: tainted.contains(&(tid as u32)),
                        local: rel_l1(e_row, tf_row),
                        inherited: rel_l1(tf_row, g_row),
                    });
                    let row0 = scratchy_subtile::subtile_ir::Region {
                        rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
                        cols: node.output.region.cols,
                    };
                    scratchy_subtile::subtile_ir::scatter(
                        &mut tf,
                        node.output.tensor,
                        row0,
                        e_row,
                        shape,
                    );
                } else {
                    rows.push(TeacherForcedRow {
                        tid,
                        node: ni,
                        op: node.op.name(),
                        fp8: tainted.contains(&(tid as u32)),
                        local: rel_l1(&e_region, &tf_out),
                        inherited: rel_l1(&tf_out, &g_region),
                    });
                    // The teacher's pin: downstream nodes see the EMULATOR's
                    // value, not the golden's product.
                    scratchy_subtile::subtile_ir::scatter(
                        &mut tf,
                        node.output.tensor,
                        node.output.region,
                        &e_region,
                        shape,
                    );
                }
            }
        }
        if let Some((_, folded_tid, folded_region)) = fold_alias
            .iter()
            .find(|(producer, _, _)| *producer == node.output.tensor.index() as u32)
        {
            scratchy_subtile::subtile_ir::scatter(
                &mut tf,
                *folded_tid,
                *folded_region,
                &tf_out,
                shape,
            );
            if let Some(e) = emu.get(&(folded_tid.index() as u64))
                && e.len() == full_len
            {
                let e_region = region_slice(e, *folded_region, shape);
                if e_region.len() == tf_out.len() {
                    scratchy_subtile::subtile_ir::scatter(
                        &mut tf,
                        *folded_tid,
                        *folded_region,
                        &e_region,
                        shape,
                    );
                }
            }
        }
    }
    (tf, rows)
}

pub fn write_eval_golden<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    dir: &std::path::Path,
    staged_rope: &std::collections::HashMap<u32, Vec<f32>>,
) -> std::io::Result<()> {
    let dbg = dir.join("dbg");
    std::fs::create_dir_all(dbg.join("golden"))?;
    let nsrc = ir.num_sources as usize;
    // The full-depth staging (`None` — the DBG oracle states the whole
    // graph's law), with the golden materializing each node's weights from
    // it per node: the same product, the same values, no all-weights f32.
    let staged = stage_synth_weights(ir, weight_ids, None, &|tid| tid as usize);
    let view = staged.view();
    let bufs = eval_golden_buffers(ir, staged_rope, &view, None, None, &|tid| tid as usize);
    // The scales manifest's skip set — the same folded-node predicate the walk used.
    let scalarmul_folds = crate::lower_subtile_tape_to_ktir::scalar_mul_weight_folds(ir);
    let folded_node = |id: scratchy_subtile::subtile_ir::SubtileId| {
        scalarmul_folds.iter().any(|(fold_id, _, _)| *fold_id == id)
    };
    let mut order = String::from("[");
    let mut seen = std::collections::HashSet::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        let tid = node.output.tensor.index() as u32 as usize;
        if ni > 0 {
            order.push(',');
        }
        let oplabel: String = format!("{:?}", node.op).chars().take(48).collect();
        let ins: Vec<String> = node
            .inputs
            .iter()
            .map(|i| i.tensor.index().to_string())
            .collect();
        order.push_str(&format!(
            "{{\"i\":{ni},\"tid\":{tid},\"op\":{:?},\"rows\":{},\"cols\":{},\"ins\":[{}]}}",
            oplabel,
            ir.tensors[tid].rows,
            ir.tensors[tid].cols,
            ins.join(",")
        ));
        if seen.insert(tid) {
            let b: Vec<u8> = bufs[tid].iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dbg.join(format!("golden/t{tid}.bin")), b)?;
        }
    }
    order.push(']');
    std::fs::write(dbg.join("order.json"), order)?;
    let mut shapes = String::from("[");
    for (id, t) in ir.tensors[..nsrc].iter().enumerate() {
        if id > 0 {
            shapes.push(',');
        }
        shapes.push_str(&format!(
            "{{\"tid\":{id},\"rows\":{},\"cols\":{}}}",
            t.rows, t.cols
        ));
    }
    shapes.push(']');
    std::fs::write(dbg.join("source_shapes.json"), shapes)?;
    let mut wt: Vec<u32> = weight_ids.iter().copied().collect();
    wt.sort_unstable();
    let wtj: Vec<String> = wt.iter().map(|w| w.to_string()).collect();
    std::fs::write(dbg.join("weight_tids.json"), format!("[{}]", wtj.join(",")))?;
    // granite ScalarMul scale VALUES (index i ↔ `scalarmul_scale_tid(i)`). SAME first-seen walk order as
    // `compute_bundle_layout` (both iterate `ir.nodes`, dedup by bits) ⇒ index↔TID consistent — including
    // the WEIGHT-FOLDED SKIP: a folded multiplier never enters the registry, so it must not enter this
    // list either, or every scale after it would bind one slot off. The reader binds each
    // `t{scalarmul_scale_tid(i)} = [scale_i]` so the on-device pointwise `mul` gets the real value
    // (unbound = 0 = wrong). Written alongside source_shapes.json (the run reads both).
    let mut sm: Vec<f32> = Vec::new();
    for node in &ir.nodes {
        if let SubOp::ScalarMul { scale } = &node.op
            && !folded_node(node.id)
            && !sm.iter().any(|s| s.to_bits() == scale.to_bits())
        {
            sm.push(*scale);
        }
    }
    let smj: Vec<String> = sm.iter().map(|s| format!("{s}")).collect();
    std::fs::write(
        dbg.join("scalarmul_scales.json"),
        format!("[{}]", smj.join(",")),
    )?;
    Ok(())
}

/// ⭐ THE GATE'S PROBE — everything one gated session binds and its reference reads: the staged
/// synth weights (the session's resident bytes, and the clone the reference view decodes), the
/// load-time constants, the seat, and every probe source (synth at the IR shape, scales as ones,
/// the rope rows, the compile-time scalars, the length mask). ONE staging for every gate: the
/// one-rank gate stages it once; the sharded gate once per rank, with `seed` naming each source's
/// synth stream — a rank's slice of a sharded source is its own, a replicated source the same on
/// every rank.
#[cfg(feature = "spyre-emu")]
struct GateProbe {
    emu_weights: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)>,
    view_bytes: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)>,
    gemm: std::collections::HashSet<u32>,
    scale_rows: std::collections::HashSet<u32>,
    load_consts: Vec<BuiltConst>,
    staged_rope: std::collections::HashMap<u32, Vec<f32>>,
    sources: Vec<(u64, Vec<f32>, Vec<usize>)>,
}

#[cfg(feature = "spyre-emu")]
fn stage_gate_probe<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    covering: Option<&std::collections::HashSet<u64>>,
    prefill: Option<&PrefillGate>,
    // The rung under test, whole — its sweep AND its query extents, one fact (clippy's
    // argument budget is the forcing function, but the rung was already the unit every
    // gate message names; the probe reads `rung.active_cap` / `rung.query_rows`).
    rung: &GateRung<'_>,
    seed: &dyn Fn(u32) -> usize,
) -> Result<GateProbe, String> {
    // ── Synth weights: staged ONCE for both legs — the emulator's resident
    //    bytes (moved into the session's HBM below) and the reference legs'
    //    per-node materialization (the clone the view borrows: `new_multi`
    //    moves the staging's own bytes, and the golden and the walk both
    //    outlive the session). The covering filter stages just the weights
    //    the kept launches read (measured shape: the 12b's full staging is
    //    ~12 GB of codes, its 7-of-48 covering slice ~1.75 GB); the clone is
    //    the one transient double, and the slice keeps both sides small. ──
    let StagedWeights {
        bytes: mut emu_weights,
        gemm,
        scale_rows,
    } = stage_synth_weights(ir, weight_ids, covering, seed);
    let weights_ref = emu_weights.clone();
    let load_consts: Vec<BuiltConst> = Vec::new();
    // ── The probe sources: synth at the IR shape, scales as ones, the mask
    //    keeping the prefix region's rows valid (the reference's own key set) ──
    // The rope cos/sin sources stage at their runtime contract — period-`half`,
    // per [`rope_table_periods`] — not as flat noise.
    let rope_table_half = rope_table_periods(ir);
    // ⭐ THE PREFILL ROPE ROWS — ONE LAW, BOTH LEGS (`staged_rope_rows`'s doc):
    // this product feeds the emulator's probe binds below AND the golden's
    // source buffers (`eval_golden_buffers`), so a staging defect cannot hide
    // on one side only. Decode stages nothing here and keeps the period-half
    // synth below — its tables are one row, where the periodic pattern and
    // the run-time staging agree by construction.
    let staged_rope: std::collections::HashMap<u32, Vec<f32>> = match prefill {
        Some(pg) => staged_rope_rows(ir, &pg.rotary_cos, &pg.rotary_sin)?,
        None => Default::default(),
    };
    // ⭐ THE PAGED STEP, STAGED BY THE WORKER'S OWN LAW (`crate::paged_step`). The gate's one step
    // is the reference's own: at decode, the rope-append prefix's rows cached and the new token at
    // the next position; at a first prompt chunk, nothing cached and `mq` rows from position 0. The
    // request's blocks run the pool's pages in REVERSE, so the walk reads every page through the
    // table rather than at its block's own index. The reference attends the cache's token-major
    // rows; the pools the paged walk reads are those SAME values laid into their pages
    // (`pool_of_rows`), so the two read one set of keys.
    // One request's step: the gate runs in an emulator bake, whose decode rungs are one request
    // wide (the batch rungs, `PagedKvPool::BATCH_RUNGS`, bake for the card alone) — a decode
    // bundle's one row is that request's, read by the batched kernel at one sequence, and a prompt
    // chunk's rows are one sequence's (the bake's `rows_are_requests`).
    let paged = crate::lower_subtile_tape_to_ktir::paged_step_of_ir(
        ir,
        rung.active_cap,
        scratchy_triton_splice::PagedRows::of_requests(prefill.is_none()),
        rung.query_rows,
    )
    .map_err(|e| e.0)?;
    // Each pool's class: its step, and its head's width.
    let pool_class: std::collections::HashMap<u32, (crate::wiring::PagedStep, usize)> = ir
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            SubOp::AttnDecode {
                mask, attention, ..
            } if n.inputs.len() >= 3 => {
                let step = paged.of(mask.class())?;
                let hd = attention.class(mask.class()).head_dim().get() as usize;
                Some([
                    (n.inputs[1].tensor.index() as u32, (step, hd)),
                    (n.inputs[2].tensor.index() as u32, (step, hd)),
                ])
            }
            _ => None,
        })
        .flatten()
        .collect();
    let step_rows = match (prefill, paged.first()) {
        (Some(pg), _) => crate::paged_step::StepRows::new(0, pg.mq as usize).expect(
            "the prefill wiring's rung holds no query rows — the bake cannot stage its step",
        ),
        (None, Some(p)) => {
            let cached = ir
                .nodes
                .iter()
                .filter_map(|n| match &n.op {
                    SubOp::AttnDecode { .. } if n.inputs.len() >= 5 => {
                        Some(n.inputs[1].region.rows.len as usize)
                    }
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            crate::paged_step::StepRows::one_at(cached.min(p.capacity().saturating_sub(1)))
        }
        (None, None) => crate::paged_step::StepRows::one_at(0),
    };
    // The request's blocks name the pages ONE ON — block `i` is page `i + 1` — so the walk reads
    // every page through the table rather than at its block's own index, and the pages the step
    // touches stay at the pool's start (the staging below writes only that prefix).
    let blocks: Vec<u32> = paged
        .first()
        .map(|p| {
            let need = step_rows.end().div_ceil(p.page_slots as usize) as u32;
            if need < p.pages {
                (1..=need).collect()
            } else {
                (0..need).collect()
            }
        })
        .unwrap_or_default();
    let mut sources: Vec<(u64, Vec<f32>, Vec<usize>)> = Vec::new();
    for tid in 0..ir.num_sources {
        // The covering slice: a source only the dropped launches read is not
        // part of the graph under test. The reserved-tid sources appended
        // below (the scalars, the mask) stay UNfiltered — the twin's programs
        // bind them exactly as the full bundle does.
        if weight_ids.contains(&tid) || covering.is_some_and(|c| !c.contains(&(tid as u64))) {
            continue;
        }
        let t = ir.tensors[tid as usize];
        // A pool's stimulus is its cached rows alone: the rows past them are no step's, and the
        // session's own allocation holds them at zero.
        let n = if pool_class.contains_key(&tid) {
            step_rows.start * t.cols as usize
        } else {
            (t.rows as usize) * (t.cols as usize)
        };
        let vals = if scale_rows.contains(&tid) {
            vec![1.0; n]
        } else if let Some(v) = staged_rope.get(&tid) {
            v.clone()
        } else if let Some(&half) = rope_table_half.get(&tid) {
            // The SAME stream the reference reads (`eval_golden_buffers`: `seed(id)`) — a rank's
            // slice of a column-split table is its own, so the stream is the rank's.
            (0..n).map(|j| dbg_synth_val(seed(tid), j % half)).collect()
        } else {
            (0..n).map(|j| dbg_synth_val(seed(tid), j)).collect()
        };
        match pool_class.get(&tid) {
            Some(&(p, hd)) => {
                let pool = crate::paged_step::pool_of_rows(&p, hd, &blocks, &vals, step_rows.start)
                    .map_err(|e| format!("the gate's pool t{tid}: {e}"))?;
                if !pool.is_empty() {
                    let rows = pool.len() / t.cols as usize;
                    sources.push((tid as u64, pool, vec![rows, t.cols as usize]));
                }
            }
            None => sources.push((tid as u64, vals, vec![t.rows as usize, t.cols as usize])),
        }
    }
    // Each class's paged step's sources, by the worker's law: the mask stack an f16 source like any
    // other, the page table and the write rows int32 words on the typed channel (the f16 source
    // channel would encode them into half the bytes their views span).
    for p in paged.iter() {
        let srcs = crate::paged_step::request_sources(&p, &blocks, &step_rows)
            .map_err(|e| format!("the gate's paged step: {e}"))?;
        for (tid, bytes, shape) in srcs.int32 {
            emu_weights.push((tid as usize, bytes, scratchy_tensors::DType::I32, shape));
        }
        let (mask_tid, mask, mask_shape) = srcs.mask;
        sources.push((u64::from(mask_tid), mask, mask_shape));
    }
    Ok(GateProbe {
        emu_weights,
        view_bytes: weights_ref,
        gemm,
        scale_rows,
        load_consts,
        staged_rope,
        sources,
    })
}

/// ⭐⭐⭐ THE BAKE-TIME NUMERIC GATE — "if there is garbage, it cannot compile."
///
/// Runs the JUST-BAKED bundle on the emulator with the synthetic probe (every
/// source at `dbg_synth_val`, the length mask keeping exactly the prefix
/// region's rows valid — the one state under which the emu's masked attention
/// equals the reference's own key set) and compares every OBSERVABLE tensor
/// against
/// [`eval_golden_buffers`] — the reference semantics of the same `dsl_math`,
/// computed independently of the spyre lowering. Divergence beyond the quant
/// floor is returned as `Err`, and the bake site panics on it: a divergent
/// bundle is a BUILD FAILURE, not an on-card discovery.
///
/// ⛔ WHY THE COMPARE SET IS THE OBSERVABLE SET: a tid fused away inside a
/// segment has a stick (it is referenced pre-fusion) but is never stored
/// post-fusion, so it reads the zero-init — comparing all golden tids would
/// false-positive on every bundle, and "emu exactly zero vs golden nonzero"
/// also conflates that skip class with the never-stored DEFECT class (a folded
/// tid nobody writes is exactly that shape). The sound set is the executor's
/// own dump law: fused boundary outputs ∪ native output bindings
/// ([`crate::runner::SpyreSession::observable_tids`]).
///
/// ⛔ THE fp8 FLOOR IS MODELED, NOT GUESSED. The reference states the program's
/// own quantization: for every fp8 matmul the golden round-trips the payload
/// through the e4m3 grid and multiplies the product by the ws row the bundle's
/// dequant applies — fold included — so the e4m3 noise is COMMON to both sides
/// and what remains is arithmetic-order and f16-stick residue. The probe binds
/// every fp8 payload as the e4m3 encoding of the same synthetic values × a
/// normalizer (`FP8_SYNTH_SCALE`) and every fp8 scale row as
/// `fold / FP8_SYNTH_SCALE` — the only binding under which the emulator's
/// dequant meets the model at the grid's own precision: the ±0.05 synth values
/// are e4m3 subnormal at unit scale, and the real path never quantizes them
/// there (its codes are per-channel scaled into normal range with the
/// magnitude riding the ws row — the probe restores that shape). `FP8_TOL`
/// is then a slack factor over `CLEAN_TOL` for the longer fp8 chains'
/// kernels-internal f16 islands — MEASURED at 0.0124 over granite's full
/// honest program (see the constant's own comment), not an order-of-magnitude
/// guess at a floor (the pre-model 8e-2 was: it let real 5% defects through
/// on granite).
///
/// SCOPE (v3): DECODE bundles (the m=1 probe) and PREFIX-FREE prefill rungs
/// (ActiveCap::NONE, the first-chunk programs): the gate stages the run-time
/// forward's OWN first-chunk law — the `[mq, mq]` clamped causal triangle and
/// the per-position rotary rows — and compares against the golden with the
/// rope tables read row-aligned and the lm-head tail's m=S fold stated as the
/// last-row law. A PREFIX-CAPABLE prefill bundle (ActiveCap::FULL — the
/// continuation chunks) is still refused by the CALLER with a loud skip line:
/// its resident-prefix mask law is the next rung. Graphs containing ops
/// `eval_node` has no host reference for (the MoE router/expert family, the KV
/// codec) are SKIPPED with a line naming the first such op and the fingerprint
/// — the reference for those families is the next rung, and a silent skip
/// would be a hole; a panicking one would break every MoE bake that works
/// today.
/// The context the gate's decode step attends: past ten page boundaries, ending mid-page — every
/// page-walk law a decode step reads (a table lookup per page, fully valid pages, a partly valid
/// one, masked pages past it) at a position a request reaches, and at least the 640 keys the
/// gate's softmax has always summed. See the decode golden's view in [`numeric_gate`].
#[cfg(feature = "spyre-emu")]
pub const GATE_DECODE_CONTEXT: u32 = 10
    * scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get()
    + scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get() / 2;

#[cfg(feature = "spyre-emu")]
pub struct PrefillGate {
    /// The bundle's query-row count — the rung it was baked at.
    pub mq: u32,
    /// `(tid, kind)` for every cos source the tape declares, then every sin
    /// source — the wiring's own rotary-class law (a hybrid arch's sliding and
    /// global classes carry different tables), so the gate stages exactly the
    /// tables the wiring names.
    pub rotary_cos: Vec<(u32, crate::wiring::RotaryKind)>,
    pub rotary_sin: Vec<(u32, crate::wiring::RotaryKind)>,
}

/// ⭐ THE RUNG UNDER TEST — the bundle's fingerprint and its KV capacity, ONE
/// param because they are one fact: every gate message names the rung by both,
/// and the refusal dump's own directory is `{fp}_cap{cap}` (the pair
/// [`write_refusal_dump`]'s signature already states). The gate sits at
/// clippy's argument budget, so the rung's identity crosses as this struct,
/// not as two loose parameters.
#[cfg(feature = "spyre-emu")]
pub struct GateRung<'a> {
    /// The bundle's fingerprint — the message half of the rung's identity.
    pub fp: &'a str,
    /// The rung's KV capacity — the decode length mask's width, and the dump's
    /// directory half.
    pub cap: u32,
    /// The rung's sweep — its paged walk ([`crate::lower_subtile_tape_to_ktir::paged_step_of_ir`]),
    /// so the gate stages the step against the reach the bundle was baked with.
    pub active_cap: ActiveCap,
    /// The rung's query extents — the pair the bake minted for the graph under test
    /// ([`scratchy_spyre_bundle::rung::QueryExtent`]), so the gate lowers the
    /// graph's nodes under the SAME extents the bundle was baked with.
    pub query_rows: scratchy_spyre_bundle::rung::QueryExtent,
    /// The slice law of every source, when the rung was baked for a sharded world (the analysis's
    /// `ShardedForward::sources`) — `None` for a world of one. A sharded rung is gated on both
    /// ranks at once ([`numeric_gate_sharded`]).
    pub shards: Option<&'a [scratchy_subtile::tensor_parallel::SourceShard]>,
}

#[cfg(feature = "spyre-emu")]
pub fn numeric_gate<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    groups: &[bundle::LaunchGroup<'static, T>],
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // ⭐ THE RUNG UNDER TEST — see [`GateRung`]: the fingerprint every message
    // names and the capacity the decode mask stages, one fact.
    rung: &GateRung,
    prefill: Option<&PrefillGate>,
    // The class-covering truncation this gate run judges (`None` = full
    // depth). `groups` must already BE the truncated bundle's groups when
    // this is `Some` — the twin `covering_twin_groups` mints.
    trunc: Option<&GateTrunc>,
) -> Result<(), String> {
    use scratchy_subtile::lower::GemmWeight;
    // The rung's identity, named for the body: the messages say `{fp}`, the
    // decode mask stages `cap` — both off the ONE param that carried them in.
    let (fp, cap) = (rung.fp, rung.cap);
    // ⛔ A SHARDED FORWARD HAS NO ONE-RANK RUN. Its consumes read the receive window the peer's
    // send fills, and its collectives have no one-rank reference, so it is gated on every rank at
    // once — below, once the covering slice is known. Which gate runs is the rung's WORLD, its
    // type; a sharded rung with no slice laws to stage its ranks by is a bake defect, refused by
    // name.
    use scratchy_spyre_bundle::collective::World;
    let sharded = match (<T as World>::SHARDED, rung.shards) {
        (false, _) => None,
        (true, Some(shards)) => Some(shards),
        (true, None) => {
            return Err(format!(
                "a {}-rank rung reached the gate with no slice laws to stage its ranks by",
                <T as World>::RANKS
            ));
        }
    };
    // ⛔ KEEP IN STEP with `eval_node`'s no-host-reference arms: an op listed
    // here that gains a reference makes the gate skip a graph it could check;
    // one missing here PANICS inside `eval_golden_buffers` with eval's own
    // message naming the op — both failures are loud, neither is silent.
    let unref_op = ir.nodes.iter().find(|n| {
        matches!(
            &n.op,
            SubOp::GateSplit { .. }
                | SubOp::LoadRows { .. }
                | SubOp::EmbeddingGather { .. }
                | SubOp::VisionRope
                | SubOp::VarlenAttention { .. }
                | SubOp::GatedDeltaNet
                | SubOp::Mamba2 { .. }
                | SubOp::KvEncode { .. }
                | SubOp::KvStage { .. }
                | SubOp::RotateRows { .. }
                | SubOp::AttnPackedKv
                | SubOp::SampleRowsGather
                | SubOp::SampleRowsScatter
        )
    });
    if let Some(n) = unref_op {
        eprintln!(
            "[spyre-superdsc] NUMERIC GATE SKIP {fp}: t{} is {} — eval_node has no host \
             reference for that family yet (the KV-codec reference arms are the next \
             rung); this bundle's numbers are UNCHECKED",
            n.output.tensor.index(),
            n.op.name()
        );
        return Ok(());
    }
    // ── The prefill golden: the splice's DEAD-PREFIX law, restated for eval ──
    // A prefix-free rung (ActiveCap::NONE) lowers its AttnDecode with SWEPT == 0:
    // the prefix segment "keeps its VIEW and drops its compute" (`lower_attn`),
    // so the program attends only the chunk's own new block through the one-pass
    // arm. eval_node derives the query positions from the CONCATENATED input
    // region rows, and `lower_region` pins a SameForwardRopeAppend prefix at
    // `valid_len - 1` rows (>= 1) — the DECODE contract, where the runtime length
    // mask bounds the sweep to that same extent. The rung consumes no length mask
    // for the prefix (the one-pass arm reads only the `[mq, mq]` triangle), so the
    // golden must see the prefix as the splice does: DEAD. The clone re-slices
    // those input regions to 0 rows — a gate-local VIEW of the same IR, never a
    // law change: every other consumer keeps the original, and a bundle whose
    // prefix is live (decode, a prefix-capable rung) never enters this arm.
    // ── The decode golden: a step at a REAL position, not the cache's last ──
    // `lower_region` pins a SameForwardRopeAppend prefix at `valid_len - 1` rows, and a paged
    // bundle's `valid_len` is its pool: the model's whole context. A step there attends every
    // slot of the pool, which no request reaches before its last token — and which the paged
    // walk's model-dtype softmax state cannot hold anyway when the keys score alike (the
    // denominator passes f16's 65504 past that many keys; spyre-inference's own state is the
    // model dtype too). The gate's decode step is at [`GATE_DECODE_CONTEXT`] instead — past ten
    // page boundaries, ending mid-page — the same kind of gate-local VIEW of the IR as the
    // prefill's dead prefix: the prefix regions are re-sliced, and the staging follows.
    let decode_reslice = prefill.is_none()
        && ir.nodes.iter().any(|n| {
            matches!(
                &n.op,
                SubOp::AttnDecode {
                    producer:
                        scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. },
                    ..
                }
            ) && n
                .inputs
                .get(1)
                .is_some_and(|r| r.region.rows.len > GATE_DECODE_CONTEXT)
        });
    let ir_owned;
    let ir: &SubtileIR<F> = if prefill.is_some() || trunc.is_some() || decode_reslice {
        ir_owned = {
            let mut g = ir.clone();
            if decode_reslice {
                for n in &mut g.nodes {
                    if matches!(
                        &n.op,
                        SubOp::AttnDecode {
                            producer:
                                scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. },
                            ..
                        }
                    ) {
                        for idx in [1usize, 2usize] {
                            if let Some(r) = n.inputs.get_mut(idx) {
                                r.region.rows = scratchy_subtile::subtile_ir::Range::new(
                                    r.region.rows.start,
                                    r.region.rows.len.min(GATE_DECODE_CONTEXT),
                                );
                            }
                        }
                    }
                }
            }
            if prefill.is_some() {
                for n in &mut g.nodes {
                    if matches!(
                        &n.op,
                        SubOp::AttnDecode {
                            producer:
                                scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. },
                            ..
                        }
                    ) {
                        for idx in [1usize, 2usize] {
                            if let Some(r) = n.inputs.get_mut(idx) {
                                r.region.rows =
                                    scratchy_subtile::subtile_ir::Range::new(r.region.rows.start, 0);
                            }
                        }
                    }
                }
            }
            if let Some(t) = trunc {
                // ⭐ THE TAIL SEAM RETARGET — the same view-law shape as the
                // dead-prefix re-slice above. The full-depth IR's suffix ops
                // read the LAST launch's hidden-out; the truncated bundle's
                // suffix reads the KEPT prefix's last hidden-out
                // (`unroll_layers`' own suffix binding for the truncated
                // twin). Every hidden-out of every launch shares one shape,
                // so the retarget rewrites only the input's tid and carries
                // the region verbatim — the golden then evaluates the tail
                // over exactly the product the twin's programs consume.
                for n in &mut g.nodes {
                    for r in &mut n.inputs {
                        if r.tensor.index() as u32 == t.suffix_rebind.0 {
                            r.tensor = scratchy_subtile::subtile_ir::TensorId::from_index(
                                t.suffix_rebind.1 as usize,
                            );
                        }
                    }
                }
            }
            g
        };
        &ir_owned
    } else {
        ir
    };
    // ── The taint map: which tensors' values cross an fp8 payload's grid ──
    let mut tainted: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for node in &ir.nodes {
        let crosses = matches!(&node.op,
            SubOp::MatmulTile { weight, .. } if matches!(weight, GemmWeight::Fp8Dynamic))
            || matches!(&node.op,
            SubOp::AllRowsMatmul if node.inputs.len() == 4)
            || node
                .inputs
                .iter()
                .any(|r| tainted.contains(&(r.tensor.index() as u32)));
        if crosses {
            tainted.insert(node.output.tensor.index() as u32);
        }
    }
    // ⭐ THE COVERING SLICE'S BOUND SET — every tid the (possibly truncated)
    // bundle's programs READ OR WRITE at the PLACE-ID level (`groups` here IS
    // the truncated twin's own groups when truncating; the full bundle's
    // otherwise). The readback declaration below filters the golden tids on
    // this set — a stick demands a shape derivable from the launch IR
    // (`resident::build`'s `ids` walk), but the golden set can name tensors
    // the reroll folded out of every launch (measured: parity-tiny states
    // 142 golden tids against 130 launch output bindings), and those ghosts
    // have no view anywhere, so declaring them refuses the session at
    // construction. They were never observable and stay uncompared, exactly
    // as before — `build_spec`'s own produced/consumed sets are computed
    // from the same bindings this filter reads.
    let bound: std::collections::HashSet<u64> = groups
        .iter()
        .flat_map(|g| g.programs.iter())
        .flat_map(|p| p.args.iter())
        .map(|(_, id)| u64::from(id.tid()))
        .collect();
    // ⭐⭐ AND THE NODE-LEVEL COVERING = THE BOUND SET'S BACKWARD CLOSURE over
    //    the (tail-retargeted) IR. The bound set alone is NOT closed at the
    //    node level: a value the fusion SSA-forwards inside a segment has no
    //    PlaceId anywhere (measured: the tiny's t149, a RmsNorm output whose
    //    only consumer is a Reshape — the reshape is pure addressing, so the
    //    norm's product rides SSA into the next program and t149 never
    //    becomes a stick), yet the reference MUST still compute it — the
    //    kept consumer gathers the slot. One reverse pass over the
    //    topologically ordered nodes closes it transitively: a consumer is
    //    visited before every one of its producers, so keeping an output
    //    keeps its inputs, all the way back to the sources. The staging, the
    //    probe sources and the golden/walk node skips all filter on this one
    //    set — a node the closure does not hold is not an ancestor of
    //    anything the twin can observe, and nothing kept reads it.
    let covering_owned: Option<std::collections::HashSet<u64>> = trunc.map(|_| {
        let mut keep = bound.clone();
        for node in ir.nodes.iter().rev() {
            if keep.contains(&(node.output.tensor.index() as u64)) {
                for r in &node.inputs {
                    keep.insert(r.tensor.index() as u64);
                }
            }
        }
        keep
    });
    let covering: Option<&std::collections::HashSet<u64>> = covering_owned.as_ref();
    // ⭐ A SHARDED RUNG IS GATED ON EVERY RANK AT ONCE — see [`numeric_gate_sharded`]: its
    // consumes read the window the peers' sends fill, so the ranks step collective by collective.
    if let Some(shards) = sharded {
        let mut scope = String::new();
        if let Some(p) = prefill {
            scope.push_str(&format!(" [prefill mq={}]", p.mq));
        }
        if let Some(t) = trunc {
            scope.push_str(&format!(
                " [class-covering {}/{} layers]",
                t.layers.0, t.layers.1
            ));
        }
        return numeric_gate_sharded(ShardedGate {
            groups,
            ir,
            weight_ids,
            covering,
            tainted: &tainted,
            prefill,
            fp,
            cap,
            active_cap: rung.active_cap,
            query_rows: rung.query_rows,
            shards,
            scope,
        });
    }
    // ── The probe: ONE staging for every gate (`stage_gate_probe`) — the one-rank gate's sources
    //    each draw their own tid's synth stream. ──
    let GateProbe {
        emu_weights,
        view_bytes,
        gemm,
        scale_rows,
        load_consts,
        staged_rope,
        sources,
    } = stage_gate_probe::<F>(ir, weight_ids, covering, prefill, rung, &|tid| tid as usize)?;
    let view = StagedWeightView {
        bytes: &view_bytes,
        gemm: &gemm,
        scale_rows: &scale_rows,
        load_consts: &load_consts,
    };
    // ── Run the bundle on the emulator ──
    // The golden tids, in the writer's own first-seen order — each with the
    // registry name of the op that first produces it (`SubOp::name`, so no
    // parallel string table can drift) and that node's INPUT tids, carried
    // into every verdict row: a refusal then names the CLASSES that diverge
    // and hands the reader the input buffers to judge magnitude-dependent
    // mechanisms (a reduce's landing grid is a function of its input scale).
    let GoldenIndex {
        tids: golden_tids,
        op: golden_op,
        ins: golden_ins,
    } = GoldenIndex::of(ir);
    // ⭐ THE GATE DECLARES ITS OWN READBACK SET — the golden tids are the program's
    // RESULTS, not just the compare list. `observable_tids` is the honest set of
    // what a run leaves in observable state, and a tensor only earns that by being
    // consumed OUTSIDE its fused segment or by being a declared result; a bundle
    // small enough to fuse into ONE segment (measured: gemma-4-12b parity-tiny —
    // 130 launch nodes, all 142 golden tids, 1 fused segment, 0 outputs) has no
    // boundaries to cross, so an empty results list starves the gate to zero
    // observables and the "no overlap" refusal fires on a bundle nothing is wrong
    // with. Declaring the golden tids as results keeps every produced golden tensor
    // an HBM edge (`fuse_program`'s `is_intermediate` excludes results, so the store
    // is never SSA-forwarded away and every consumer LOADS the stored tile) — read
    // back and threaded. ⚠ THAT IS AN EXCHANGE-MODELLING CHANGE, NOT PURE READBACK:
    // a cross-launch golden edge the fusion would have forwarded as raw f32 SSA now
    // rounds through the storage dtype at its store — the same rounding the per-node
    // golden reference applies at every node exchange (and every card descriptor's
    // DLF16 store), where the forwarded edge applied none. The gated emu and the
    // reference now agree on WHERE values round, instead of the emu under-rounding
    // forwarded edges; a value pair straddling an f16 grid boundary can still move
    // apart by up to one ulp, so this is stated as a modelling change and the
    // granite calibration is its control — leg 5 (granite-3.1-8b dense under
    // the worker's spyre-emu feature) must stay green under the
    // declared-results instrument. It also makes the compare set
    // COMPLETE for every bundle,
    // not just the ones the emulator's LX live-set budget happens to split into
    // segments. `run_step`'s per-pass `outputs` stays the real readback mechanism;
    // `results` here only states what this session exists to observe.
    //
    // ⚔ AND ONLY THE GOLDEN TIDS THE LAUNCHES ACTUALLY BIND — the `bound`
    // set computed above the staging (the covering slice's own filter).
    let results: Vec<u64> = golden_tids
        .iter()
        .copied()
        .filter(|t| bound.contains(t))
        .collect();
    let refs: Vec<(&[bundle::LaunchGroup<'static, T>], &[u64])> = vec![(groups, &results)];
    // ⭐ THE SHAPE FLOOR — the krg's own tensor shapes, the tape's statement
    // the resident sticks allocate at LEAST. The emulator's view walk derives a
    // tid's extent only from views its arg→tid mappings reach, and a widened
    // reader whose view sits on an interior edge of a fused function is
    // invisible to it (measured: the argmax mint's PAD_V-wide logits row over
    // the lm-head matmul's narrower device-width out view). The floor is the
    // same authority `compute_bundle_layout` reserves from — the minted krg —
    // and it is a MAX, so it can only raise an allocation.
    let shape_floor: std::collections::HashMap<u64, usize> = ir
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (i as u64, t.rows as usize * t.cols as usize))
        .collect();
    // The gate's weights are SYNTHESIZED bytes (random draws, built tables, word rows) —
    // built, so owned; stated by wrapping at the boundary the session's provenance type
    // demands.
    let emu_weights = emu_weights
        .into_iter()
        .map(|(id, b, dt, s)| (id, crate::allocator::HostBytes::Owned(b), dt, s))
        .collect::<Vec<_>>();
    let mut session =
        crate::runner::SpyreSession::new_multi_floored(&refs, emu_weights, &shape_floor)
            .map_err(|e| format!("gate session: {e}"))?;
    let observables = session.observable_tids(0);
    let compare: Vec<(u64, usize)> = golden_tids
        .iter()
        .filter(|t| observables.contains(t))
        .map(|t| (*t, 0))
        .collect();
    if compare.is_empty() {
        return Err(format!(
            "the numeric gate found no observable tensors to compare for {fp} — \
             {} golden tids, {} observable tids, no overlap: the bundle and the \
             reference disagree about what this program even is",
            golden_tids.len(),
            observables.len()
        ));
    }
    let out = session
        .run_step(0, sources, &compare)
        .map_err(|e| format!("gate run: {e}"))?;
    // ── Compare at the floor, against the in-memory reference ──
    // `mut` in the judge: the teacher-forced walk borrows the golden's staged sources through it
    // (pointer-swapped per node, restored before return).
    let eval_bufs =
        eval_golden_buffers(ir, &staged_rope, &view, covering, None, &|tid| tid as usize);
    let GateWorst {
        local_clean: worst_local_clean,
        local_fp8: worst_local_fp8,
        total_clean: worst_clean,
        total_fp8: worst_fp8,
    } = judge_gate_run(
        GateRun {
            ir,
            prefill,
            compare: &compare,
            out: &out,
            tainted: &tainted,
            golden_op: &golden_op,
            golden_ins: &golden_ins,
            view: &view,
            covering,
            fp,
            cap,
            peer: None,
        },
        eval_bufs,
    )?;
    let mut scope = String::new();
    if let Some(p) = prefill {
        scope.push_str(&format!(" [prefill mq={}]", p.mq));
    }
    if let Some(t) = trunc {
        scope.push_str(&format!(
            " [class-covering {}/{} layers]",
            t.layers.0, t.layers.1
        ));
    }
    // ⭐ THE RESIDENCY HIGH-WATER MARK, STATED ON EVERY PASS — a bundle that
    // fits must leave headroom, not fit by zero (the wall-5 law), and the
    // peak is the number that says so: the maximum live LX charge any core
    // reached across this gate's runs, over the per-core capacity.
    let (lx_peak, lx_cap) = (session.lx_peak_used(), session.lx_capacity());
    eprintln!(
        "[spyre-superdsc] NUMERIC GATE PASS {fp}: {} tensors at the floor (worst local \
         clean {worst_local_clean:.4}, worst local fp8-chained {worst_local_fp8:.4}; worst \
         total clean {worst_clean:.4}, worst total fp8-chained {worst_fp8:.4}; LX peak \
         {lx_peak} of {lx_cap} bytes, {:.1}% of capacity){scope}",
        compare.len(),
        if lx_cap > 0 {
            lx_peak as f64 * 100.0 / lx_cap as f64
        } else {
            f64::NAN
        }
    );
    Ok(())
}

/// One gated run, judged: its readback against its reference — the run's own inputs. One rank's
/// run of a sharded gate carries its world, whose buffers the reference's collectives read.
#[cfg(feature = "spyre-emu")]
struct GateRun<'a, F: RopeForm> {
    ir: &'a SubtileIR<F>,
    prefill: Option<&'a PrefillGate>,
    compare: &'a [(u64, usize)],
    out: &'a std::collections::HashMap<u64, Vec<f32>>,
    tainted: &'a std::collections::HashSet<u32>,
    golden_op: &'a std::collections::HashMap<u64, &'static str>,
    golden_ins: &'a std::collections::HashMap<u64, Vec<u32>>,
    view: &'a StagedWeightView<'a>,
    covering: Option<&'a std::collections::HashSet<u64>>,
    fp: &'a str,
    cap: u32,
    peer: Option<GatePeer<'a>>,
}

/// A passing run's worst errors: each compared tensor's producing op's own execution (`local`) and
/// the chain's accumulation (`total`), clean and fp8-chained.
#[cfg(feature = "spyre-emu")]
struct GateWorst {
    local_clean: f32,
    local_fp8: f32,
    total_clean: f32,
    total_fp8: f32,
}

/// ⭐ THE GATE'S VERDICT ON ONE RUN — every compared tensor at its floor, judged on its producing
/// op's own execution (the teacher-forced walk), the refusal carrying its evidence. The one-rank
/// gate judges its one run here; the sharded gate each rank's.
#[cfg(feature = "spyre-emu")]
fn judge_gate_run<F: RopeForm>(
    run: GateRun<'_, F>,
    mut eval_bufs: Vec<Vec<f32>>,
) -> Result<GateWorst, String> {
    let GateRun {
        ir,
        prefill,
        compare,
        out,
        tainted,
        golden_op,
        golden_ins,
        view,
        covering,
        fp,
        cap,
        peer,
    } = run;
    // ── Compare at the floor, against the in-memory reference ──
    const CLEAN_TOL: f32 = 5e-3;
    // The fp8 chains' slack over CLEAN_TOL: the quant model makes the e4m3 grid and
    // the ws mulf COMMON to both sides and the f16-stick model puts every exchanged
    // value on the stick grid, so what is left is the kernels' INTERNAL f16 islands
    // (rmsnorm's f16 inv-rms and pre-sum squares, rope's per-op muls, the attention's
    // f16 softmax tiles) — per-hop half-ulp terms that random-walk down forty-layer
    // chains onto collapsed streams. MEASURED, not guessed: after all three models
    // landed, granite's honest program tops out at rel 0.0124 over 522 tensors (the
    // six stragglers carry e == g to every printed digit), while the smallest defect
    // class the gate has convicted sits at 0.13 (the attention's o_proj consumers) —
    // 2e-2 holds 1.6x over the measured floor and 6.5x under the smallest garbage.
    // Transcribing each kernel's op order into the golden instead would couple the
    // reference to every kernel's internals — the wrong direction for a shared
    // `eval_node` law — so the slack IS the instrument for this class.
    const FP8_TOL: f32 = 2e-2;

    /// The tolerance one compared tensor grades at. ⭐ THE INDEX LAW IS ASKED
    /// FIRST (#813 review, item 3): the re-pointed result is an i32 stick read
    /// RAW, never fp8-quantized, so a stick that lands in a TAINTED (fp8)
    /// graph still grades at 0 — asking `is_fp8` first graded the stick at
    /// FP8_TOL on granite fp8-dynamic, where one whole wrong token is a
    /// 1-in-vocab rel that passes the fp8 bucket. Any divergence on the stick
    /// is a wrong token, whichever graph holds it.
    fn compare_tol(is_result_stick: bool, is_fp8: bool) -> f32 {
        if is_result_stick {
            0.0
        } else if is_fp8 {
            FP8_TOL
        } else {
            CLEAN_TOL
        }
    }

    let mut failures: Vec<String> = Vec::new();
    let mut worst_clean = 0f32;
    let mut worst_fp8 = 0f32;
    // The first divergences' opening elements — a refusal carries the SHAPE of
    // its garbage, not just the size: a constant g→e ratio indicts a scale law,
    // same-magnitude uncorrelated values indicts a layout, structured zeros
    // indict a binding that never landed.
    let mut first_fails: Vec<(u64, Vec<f32>, Vec<f32>)> = Vec::new();
    // Every judged tensor's verdict, kept for the refusal dump — the numbers the
    // loop already computed, so a refusal can carry its full per-layer curve.
    let mut verdicts: Vec<ComparedVerdict> = Vec::with_capacity(compare.len());
    // ⭐ THE LM-HEAD TAIL FOLD'S TIDS — by the splice's own predicate, the same
    // four conjuncts the lowering read (`is_prefill_lm_head_tail`), so the gate
    // states the fold the program actually carries, not a re-derived guess.
    // Prefill only: a decode bundle (rows-are-requests, m=1) never folds.
    let mut folded: std::collections::HashSet<u64> = if prefill.is_some() {
        ir.nodes
            .iter()
            .filter(|n| scratchy_triton_splice::is_prefill_lm_head_tail(n, ir, false))
            .map(|n| n.output.tensor.index() as u64)
            .collect()
    } else {
        Default::default()
    };
    // ⭐ EVERY TID ON THE ROW-LOCAL PATH FROM A FOLDED OUTPUT TO THE RESULT
    // RIDES THE FOLD — a forward closure to a fixpoint, keyed on the
    // structural facts, not set membership: a tid joins iff its producer is
    // one of the splice's own row-local tail ops (`row_local_tail_op` — the
    // same classifier that admits the fold), it consumes a fold-family
    // tensor, and it feeds the result through the same row-local walk
    // (`tensor_feeds_result_through_row_local_ops`) — so nothing un-folded
    // can ride in. The front end column-chunks a wide pointwise tail op past
    // the predicate's width conjunct (`n_blocks(out_cols, nb)` chunks
    // Silu/Mul/Add/ScalarMul/SiluMul — measured: granite's 49155-wide logits
    // ScalarMul is 7 region-slice nodes of 8192 cols, so none matches and the
    // tail op stays un-folded, the splice's documented cost-only shape:
    // "row 0 is still right ... the device pays mq−1 rows of reads of
    // unwritten logits for nothing"). Those chunks still scale the folded
    // matmul's row-0 (= LAST-row) product — row-locally, guaranteed by this
    // same walk — so each path tid's row 0 is the fold's product and its
    // rows 1+ are the dead stick: it takes the same last-row-vs-row-0 law
    // (measured on the granite refusal dump: last-row rel 0.00049 at the fp8
    // floor, rows 1-30 exactly zero; the plain law reads 1.0136). The
    // sampler mint re-points the rung's RESULT to the idx stick, so the
    // scaled logits between the fold and the draw are a compared
    // INTERMEDIATE, not the result — without the closure they took the
    // plain law and refused (measured: bundle 55d6eadfdf521a5c, t1155's
    // seven chunk locals 1.008-1.014). A no-op when the tail's own op
    // matched whole-region (gemma's TanhSoftCap already put the result in
    // the set). The result itself rides when its own producer walks in, and
    // the idx stick's compare stays EXACT (`result_is_index`, tolerance 0):
    // the idx readback is one row, `tail_fold`'s shape conjunct turns it
    // down and it falls to the plain exact compare below.
    loop {
        let mut grew = false;
        for n in &ir.nodes {
            let out = n.output.tensor.index() as u64;
            if folded.contains(&out) {
                continue;
            }
            if n.inputs
                .iter()
                .any(|i| folded.contains(&(i.tensor.index() as u64)))
                && scratchy_triton_splice::row_local_tail_op(&n.op)
                && scratchy_triton_splice::tensor_feeds_result_through_row_local_ops(
                    ir,
                    n.output.tensor,
                )
            {
                folded.insert(out);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    // ⭐ THE RESULT'S OWN DTYPE — the same fact the wiring ships
    // (`result_dtype_of_ir`). An INTEGER result is an INDEX stick, and a
    // tolerance over indices is a category error: one lane off is a WRONG
    // TOKEN, not accumulation noise, and its rel (1/40017 ≈ 2.5e-5 on
    // granite) sails under CLEAN_TOL — the gate would certify an argmax that
    // picked the wrong word. The numeric legs already pin the fixture's idx
    // exactly (64/64 lanes); this states the same law for the rung's ANSWER,
    // post-re-point: the result compares EXACTLY (tolerance 0).
    let result_is_index = matches!(
        crate::lower_subtile_tape_to_ktir::result_dtype_of_ir(ir),
        Ok(scratchy_tensors::DType::I32) | Ok(scratchy_tensors::DType::U32)
    );
    for &(tid, _) in compare {
        let g_full = &eval_bufs[tid as usize];
        let e = match out.get(&tid) {
            Some(v) => v,
            None => {
                failures.push(format!("t{tid}: no readback"));
                continue;
            }
        };
        // The fold is a documented transform, not a defect: the splice lowers
        // the result-width tail to m=S over the requests' answer rows (the
        // last-row extraction feeding the re-lowered ops; this golden's rung
        // is one request, so S = 1), so those tids' program product is S
        // rows — rows 0..S of a readback whose STICK the IR still sizes at
        // [mq, cols] (the program never writes past row S; measured: t302's
        // stick read back 19 rows with rows 1-18 all zero). The compare
        // states the fold's own
        // law — the golden's LAST row against the emu's row 0 (the
        // `selector_lastrow_col` extraction the lowering pins) — so a fold
        // that grabs the WRONG row still diverges. A tid that comes back
        // shorter than one row without matching the predicate takes the plain
        // law below, which refuses.
        let mq_rows = prefill.map(|p| p.mq as usize).unwrap_or(1);
        let tail_fold = folded.contains(&tid)
            && !g_full.is_empty()
            && !e.is_empty()
            && g_full.len().is_multiple_of(mq_rows)
            && g_full.len() / mq_rows <= e.len();
        let (g, e): (&[f32], &[f32]) = if tail_fold {
            let cols = g_full.len() / mq_rows;
            (&g_full[g_full.len() - cols..], &e[..cols])
        } else {
            (g_full, e)
        };
        let n = g.len().min(e.len());
        if n == 0 {
            continue;
        }
        let gsum: f32 = g.iter().take(n).map(|x| x.abs()).sum();
        let esum: f32 = e.iter().take(n).map(|x| x.abs()).sum();
        if gsum < 1e-6 && esum < 1e-6 {
            continue; // both dead — nothing to state
        }
        let dsum: f32 = g
            .iter()
            .zip(e.iter())
            .take(n)
            .map(|(a, b)| (a - b).abs())
            .sum();
        let rel = dsum / gsum.max(1e-6);
        let is_fp8 = tainted.contains(&(tid as u32));
        // The index stick's exact law above — and the index law is asked FIRST
        // (#813 review, item 3): see [`compare_tol`].
        let tol = compare_tol(result_is_index && tid == ir.result.index() as u64, is_fp8);
        verdicts.push(ComparedVerdict {
            tid,
            n,
            rel,
            tol,
            fp8: is_fp8,
            op: golden_op.get(&tid).copied().unwrap_or("?"),
            ins: golden_ins.get(&tid).cloned().unwrap_or_default(),
        });
        if is_fp8 {
            worst_fp8 = worst_fp8.max(rel);
        } else {
            worst_clean = worst_clean.max(rel);
        }
    }
    // ⭐ THE VERDICT IS THE OP'S OWN EXECUTION, NOT THE CHAIN'S ACCUMULATION —
    //    the teacher-forced walk runs on EVERY gate now (pass or refuse), and
    //    the gate refuses on each compared tensor's `local`: its producing
    //    op's own error over the EMULATOR'S OWN inputs, at the same
    //    tolerances. MEASURED (gemma-4-12b): every class's local sits at its
    //    floor with depth (worst clean 0.0009, worst fp8 0.016, matmuls
    //    bit-exact) while the total rel grows to 0.29 at the logits — the
    //    fp8 chain's floor noise amplified by the network itself (a golden
    //    restarted from the emu's state at 75% depth tracks the emu to 3% of
    //    the divergence), which the model's conditioning owns, not the
    //    emulator's execution; judging the total refused that model forever.
    //    The local still convicts every execution defect the total ever did
    //    — staging, wiring and readback position each show local ≈ the
    //    corruption at their site (smallest convicted class: rel 0.13; a
    //    wrong bind: local ≈ 1) — 8x and 60x over the 12b's worst local.
    //    The total stays a REPORTED statistic beside the verdict; a compared
    //    tensor with no split row (a folded-only producer, or a readback too
    //    short to region-slice) falls back to the total at the same
    //    tolerance — the pre-split rule — so no compared tensor escapes
    //    judgment.
    let (walk_tf, split) = teacher_forced_walk(
        ir,
        &mut eval_bufs,
        EmuReadback { own: out, peer },
        tainted,
        &folded,
        view,
        covering,
    );
    drop(walk_tf);
    let mut local_by_tid: std::collections::HashMap<u64, f32> =
        std::collections::HashMap::with_capacity(split.len());
    for row in &split {
        let worst = local_by_tid.entry(row.tid).or_insert(0f32);
        *worst = (*worst).max(row.local);
    }
    let mut worst_local_clean = 0f32;
    let mut worst_local_fp8 = 0f32;
    for v in &verdicts {
        let local = local_by_tid.get(&v.tid).copied();
        if let Some(l) = local {
            if v.fp8 {
                worst_local_fp8 = worst_local_fp8.max(l);
            } else {
                worst_local_clean = worst_local_clean.max(l);
            }
        }
        let judged = local.unwrap_or(v.rel);
        if judged > v.tol {
            let g = &eval_bufs[v.tid as usize];
            if first_fails.len() < 4
                && let Some(e) = out.get(&v.tid)
            {
                first_fails.push((
                    v.tid,
                    g.iter().take(8).copied().collect(),
                    e.iter().take(8).copied().collect(),
                ));
            }
            let gsum: f32 = g.iter().take(v.n).map(|x| x.abs()).sum();
            failures.push(format!(
                "t{}: {} {judged:.4} > {} ({}), total rel {:.4}, mean|g| {:.4}, {} elems",
                v.tid,
                if local.is_some() {
                    "local"
                } else {
                    "total, no split row"
                },
                v.tol,
                if v.fp8 { "fp8-chained" } else { "clean chain" },
                v.rel,
                gsum / v.n as f32,
                v.n
            ));
        }
    }
    if !failures.is_empty() {
        let samples: Vec<String> = first_fails
            .iter()
            .map(|(t, g, e)| {
                let gs: Vec<String> = g.iter().map(|v| format!("{v:.5}")).collect();
                let es: Vec<String> = e.iter().map(|v| format!("{v:.5}")).collect();
                format!("t{t}: g=[{}] e=[{}]", gs.join(","), es.join(","))
            })
            .collect();
        let mut restarts: Vec<RestartVerdict> = Vec::new();
        for frac in [0.25f32, 0.5, 0.75] {
            let cutoff = ((compare.len() as f32 * frac) as usize).clamp(1, compare.len());
            let overlay: std::collections::HashMap<u64, Vec<f32>> = compare[..cutoff]
                .iter()
                .filter_map(|(t, _)| out.get(t).map(|v| (*t, v.clone())))
                .collect();
            let (tf, _rows) = teacher_forced_walk(
                ir,
                &mut eval_bufs,
                EmuReadback {
                    own: &overlay,
                    peer,
                },
                tainted,
                &folded,
                view,
                covering,
            );
            let tail = compare.len() - (compare.len() / 10).clamp(1, compare.len() / 2);
            let mut tracks: Vec<f32> = Vec::new();
            let mut contracts: Vec<f32> = Vec::new();
            for (t, _) in &compare[tail..] {
                if let (Some(e), Some(g), Some(tv)) =
                    (out.get(t), eval_bufs.get(*t as usize), tf.get(*t as usize))
                {
                    tracks.push(rel_l1(tv, e));
                    contracts.push(rel_l1(tv, g));
                }
            }
            tracks.sort_by(|a, b| a.total_cmp(b));
            contracts.sort_by(|a, b| a.total_cmp(b));
            let med = |v: &[f32]| v[v.len() / 2];
            let (lt, _) = compare[compare.len() - 1];
            let (track_last, contract_last, direct_last) = match (
                out.get(&lt),
                eval_bufs.get(lt as usize),
                tf.get(lt as usize),
            ) {
                (Some(e), Some(g), Some(tv)) => (rel_l1(tv, e), rel_l1(tv, g), rel_l1(e, g)),
                (Some(e), Some(g), None) => (f32::NAN, f32::NAN, rel_l1(e, g)),
                _ => (f32::NAN, f32::NAN, f32::NAN),
            };
            restarts.push(RestartVerdict {
                from_frac: frac,
                from_tid: compare[cutoff - 1].0,
                tail_n: tracks.len(),
                track_med: if tracks.is_empty() {
                    f32::NAN
                } else {
                    med(&tracks)
                },
                contract_med: if contracts.is_empty() {
                    f32::NAN
                } else {
                    med(&contracts)
                },
                track_last,
                contract_last,
                direct_last,
            });
        }
        // ⭐ THE REFUSAL DUMPS ITS EVIDENCE — [`write_refusal_dump`]'s law. The note
        // names the dir when the write landed and the error when it did not; either
        // way the refusal itself is the same verdict it always was.
        let evidence = RefusalEvidence {
            split: &split,
            restarts: &restarts,
        };
        let dump_note =
            match write_refusal_dump(fp, cap, &failures, &verdicts, &eval_bufs, out, &evidence) {
                Ok(dir) => format!(" — full dump: {}", dir.display()),
                Err(e) => format!(" — dump failed: {e}"),
            };
        return Err(format!(
            "{} of {} compared tensors diverge beyond the quant floor in their producing \
             op's OWN execution: [{}] — first divergences: {}{dump_note}",
            failures.len(),
            compare.len(),
            failures.join("; "),
            samples.join(" | ")
        ));
    }
    Ok(GateWorst {
        local_clean: worst_local_clean,
        local_fp8: worst_local_fp8,
        total_clean: worst_clean,
        total_fp8: worst_fp8,
    })
}

/// Every tensor the reference computes, in the writer's own first-seen order — each with the
/// registry name of the op that first produces it (`SubOp::name`, so no parallel string table can
/// drift) and that node's INPUT tids, carried into every verdict row: a refusal then names the
/// CLASSES that diverge and hands the reader the input buffers to judge magnitude-dependent
/// mechanisms (a reduce's landing grid is a function of its input scale).
#[cfg(feature = "spyre-emu")]
struct GoldenIndex {
    tids: Vec<u64>,
    op: std::collections::HashMap<u64, &'static str>,
    ins: std::collections::HashMap<u64, Vec<u32>>,
}

#[cfg(feature = "spyre-emu")]
impl GoldenIndex {
    fn of<F: RopeForm>(ir: &SubtileIR<F>) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut index = Self {
            tids: Vec::new(),
            op: std::collections::HashMap::new(),
            ins: std::collections::HashMap::new(),
        };
        for n in &ir.nodes {
            let t = n.output.tensor.index() as u64;
            if seen.insert(t) {
                index.tids.push(t);
                index.op.insert(t, n.op.name());
                index.ins.insert(
                    t,
                    n.inputs.iter().map(|r| r.tensor.index() as u32).collect(),
                );
            }
        }
        index
    }
}

/// A sharded rung under the sharded gate — what [`numeric_gate`] hands [`numeric_gate_sharded`]
/// once its covering slice and taint map are known.
#[cfg(feature = "spyre-emu")]
struct ShardedGate<'a, T: scratchy_spyre_bundle::collective::Tp, F: RopeForm> {
    groups: &'a [bundle::LaunchGroup<'static, T>],
    ir: &'a SubtileIR<F>,
    weight_ids: &'a std::collections::HashSet<u32>,
    covering: Option<&'a std::collections::HashSet<u64>>,
    tainted: &'a std::collections::HashSet<u32>,
    prefill: Option<&'a PrefillGate>,
    // The rung's own facts, carried loose — fp, cap, the sweep, the query extents — plus
    // the shard law as its OWN typed field: `shards: &[SourceShard]`, not an Option to
    // recover from the rung. The sharded gate cannot be BUILT without its shards, so
    // nothing on this path unwraps.
    fp: &'a str,
    cap: u32,
    active_cap: ActiveCap,
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
    shards: &'a [scratchy_subtile::tensor_parallel::SourceShard],
    scope: String,
}

/// ⭐⭐ THE SHARDED GATE — a sharded rung, run on EVERY rank at once and judged against the
/// sharded reference. The review's law (#1009, item 3): the bundle's own consumes read the window
/// the peer's send filled, so the gate steps the ranks' emulator sessions collective by collective,
/// exactly as the cards step: each rank runs up to its next consume, the PEER's operand fragment —
/// what the peer's F-send would land — is staged into this rank's receive window, and the consume
/// runs over it. Then:
///
/// - each rank's run is judged against its OWN reference ([`judge_gate_run`], the one-rank gate's
///   verdict) — the rank's graph evaluated with its collectives read across ranks
///   ([`GatePeer`], `tensor_parallel::eval_collective`'s law), the lockstep golden settled to a
///   fixed point (each pass resolves one more collective);
/// - every value the reference holds bit-identical on every rank — a REPLICATED value — must come
///   back bit-identical from every emulated rank (the canonical reduction order's promise).
///
/// A rank's slice of a sharded source draws its own synth stream, so the ranks' partials differ
/// and a consume that read its own operand twice, or the window at the wrong offset, diverges;
/// replicated sources are the same on every rank, as they are on the cards.
#[cfg(feature = "spyre-emu")]
fn numeric_gate_sharded<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    gate: ShardedGate<'_, T, F>,
) -> Result<(), String> {
    use crate::bundle_code::collective::{Rank, plays_on, ranks as all_ranks};
    use scratchy_subtile::tensor_parallel::SourceShard;
    let ShardedGate {
        groups,
        ir,
        weight_ids,
        covering,
        tainted,
        prefill,
        fp,
        cap,
        active_cap,
        query_rows,
        shards,
        scope,
    } = gate;
    // The world the bake sharded the rung for is the rung's type.
    let ranks: Vec<Rank<T>> = all_ranks::<T>().collect();
    // ── Each rank's probe: its slices of the sharded sources on a stream of their own (ONE seed,
    //    read by the session's staging and by the reference alike) ──
    const RANK_STREAM: usize = 1 << 28;
    let seed = |r: usize| {
        move |tid: u32| match shards.get(tid as usize) {
            Some(SourceShard::Whole) | None => tid as usize,
            Some(_) => tid as usize + r * RANK_STREAM,
        }
    };
    // The probe lowers under the rung's sweep and query extents — the same GateRung
    // shape the one-rank gate hands it, wrapped (never unwrapped) with this gate's own
    // typed shard slice.
    let probe_rung = GateRung {
        fp,
        cap,
        active_cap,
        query_rows,
        shards: Some(shards),
    };
    let mut probes: Vec<GateProbe> = Vec::with_capacity(ranks.len());
    for me in &ranks {
        probes.push(stage_gate_probe::<F>(
            ir,
            weight_ids,
            covering,
            prefill,
            &probe_rung,
            &seed(me.get() as usize),
        )?);
    }
    let golden_index = GoldenIndex::of(ir);
    let shape_floor: std::collections::HashMap<u64, usize> = ir
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (i as u64, t.rows as usize * t.cols as usize))
        .collect();
    // ── Each rank's session: its played groups, cut before every consume ──
    let segments = |me: Rank<T>| -> Vec<Vec<bundle::LaunchGroup<'static, T>>> {
        let mut segs: Vec<Vec<bundle::LaunchGroup<'static, T>>> = vec![Vec::new()];
        for g in groups.iter().filter(|g| plays_on(g.kv.tp, me)) {
            if g.kv.tp.is_some() && segs.last().is_some_and(|s| !s.is_empty()) {
                segs.push(Vec::new());
            }
            if let Some(s) = segs.last_mut() {
                s.push(g.clone());
            }
        }
        segs
    };
    let segs: Vec<Vec<Vec<bundle::LaunchGroup<'static, T>>>> =
        ranks.iter().map(|&me| segments(me)).collect();
    let nseg = segs[0].len();
    if segs.iter().any(|s| s.len() != nseg) {
        return Err(format!(
            "the ranks cut into {:?} segments — every rank meets every collective",
            segs.iter().map(Vec::len).collect::<Vec<_>>()
        ));
    }
    let mut sessions = Vec::with_capacity(ranks.len());
    let mut observables: Vec<Vec<Vec<u64>>> = Vec::with_capacity(ranks.len());
    for (r, rank_segs) in segs.iter().enumerate() {
        let results: Vec<Vec<u64>> = rank_segs
            .iter()
            .map(|seg| {
                let bound: std::collections::HashSet<u64> = seg
                    .iter()
                    .flat_map(|g| g.programs.iter())
                    .flat_map(|p| p.args.iter())
                    .map(|(_, id)| u64::from(id.tid()))
                    .collect();
                golden_index
                    .tids
                    .iter()
                    .copied()
                    .filter(|t| bound.contains(t))
                    .collect()
            })
            .collect();
        let refs: Vec<(&[bundle::LaunchGroup<'static, T>], &[u64])> = rank_segs
            .iter()
            .zip(&results)
            .map(|(seg, res)| (seg.as_slice(), res.as_slice()))
            .collect();
        let session = crate::runner::SpyreSession::new_multi_floored(
            &refs,
            std::mem::take(&mut probes[r].emu_weights)
                .into_iter()
                .map(|(id, b, dt, s)| (id, crate::allocator::HostBytes::Owned(b), dt, s))
                .collect::<Vec<_>>(),
            &shape_floor,
        )
        .map_err(|e| format!("rank {r}'s gate session: {e}"))?;
        observables.push((0..nseg).map(|k| session.observable_tids(k)).collect());
        sessions.push(session);
    }
    // ── The lockstep run: segment k on every rank, then every peer's block of the window cut each
    //    rank's next consume reads gets that peer's fragment of the operand it names — what the
    //    peer's F-send lands ──
    // ⛔ THE OPERAND IS THE IR NODE'S, NOT THE BAKED CONSUME'S. A rolled body bakes ONE consume per
    // class and replays it per layer; its `TpConsume::operand` names the class's tid, and the card's
    // send reaches layer `l`'s copy through the launch's layer shift. The gate has no shift to
    // apply, so it reads each segment's collective off the graph it lowered: segment `k + 1` opens
    // with the consume of the `k`-th collective node, whose input IS that layer's operand.
    // The covering slice's own: a truncated bundle carries only the kept layers' consumes.
    let collective_nodes: Vec<&scratchy_subtile::subtile_ir::SubtileNode<F>> = ir
        .nodes
        .iter()
        .filter(|n| {
            is_collective(&n.op)
                && covering.is_none_or(|c| c.contains(&(n.output.tensor.index() as u64)))
        })
        .collect();
    if collective_nodes.len() + 1 != nseg {
        return Err(format!(
            "{} segment(s) for {} collective node(s) — a consume per collective",
            nseg,
            collective_nodes.len()
        ));
    }
    let mut outs: Vec<std::collections::HashMap<u64, Vec<f32>>> =
        vec![std::collections::HashMap::new(); ranks.len()];
    // A staged window block: `run_step`'s source — the block's tid, its values, their shape.
    type StagedWindow = (u64, Vec<f32>, Vec<usize>);
    let mut windows: Vec<Vec<StagedWindow>> = vec![Vec::new(); ranks.len()];
    for k in 0..nseg {
        for (r, session) in sessions.iter_mut().enumerate() {
            let sources = if k == 0 {
                std::mem::take(&mut probes[r].sources)
            } else {
                let staged = std::mem::take(&mut windows[r]);
                if staged.len() + 1 != ranks.len() {
                    return Err(format!(
                        "rank {r} reached segment {k} with {} of its {} peers' blocks staged",
                        staged.len(),
                        ranks.len() - 1
                    ));
                }
                staged
            };
            let mut want: Vec<(u64, usize)> = golden_index
                .tids
                .iter()
                .filter(|t| observables[r][k].contains(t))
                .map(|t| (*t, 0))
                .collect();
            if let Some(n) = collective_nodes.get(k) {
                let t = n.inputs[0].tensor.index() as u64;
                if !want.iter().any(|(w, _)| *w == t) {
                    want.push((t, 0));
                }
            }
            let o = session
                .run_step(k, sources, &want)
                .map_err(|e| format!("rank {r} segment {k}: {e}"))?;
            outs[r].extend(o);
        }
        if k + 1 == nseg {
            break;
        }
        for (r, window) in windows.iter_mut().enumerate() {
            // The cut is the RECEIVER's consume's — the one its programs read.
            let Some(mine) = segs[r][k + 1].first().and_then(|g| g.kv.tp) else {
                return Err(format!(
                    "rank {r}'s segment {} does not open with a consume",
                    k + 1
                ));
            };
            for peer in ranks[r].peers() {
                let p = peer.get() as usize;
                let Some(c) = segs[p][k + 1].first().and_then(|g| g.kv.tp) else {
                    return Err(format!(
                        "rank {p}'s segment {} does not open with a consume",
                        k + 1
                    ));
                };
                let x = collective_nodes[k].inputs[0];
                let tid = x.tensor.index();
                let t = ir.tensors[tid];
                let (rows, cols) = (t.rows as usize, t.cols as usize);
                let v = outs[p].get(&(tid as u64)).ok_or_else(|| {
                    format!(
                        "rank {p}'s operand t{tid} for consume {} was not read back",
                        k + 1
                    )
                })?;
                // The fragment's columns are the operand REGION's (`send_fragment` cuts the region).
                let f = c.fragment;
                let (r0, fr) = (x.region.rows.start as usize, f.rows() as usize);
                let (c0, fc) = (
                    (x.region.cols.start + f.first_col()) as usize,
                    f.cols() as usize,
                );
                if r0 + fr > rows || c0 + fc > cols || v.len() < rows * cols {
                    return Err(format!(
                        "rank {p}'s operand t{tid} [{rows}, {cols}] ({} read back) cannot hold \
                         the fragment rows {fr} cols [{c0}, {})",
                        v.len(),
                        c0 + fc
                    ));
                }
                let mut win = Vec::with_capacity(fr * fc);
                for i in r0..r0 + fr {
                    win.extend_from_slice(&v[i * cols + c0..i * cols + c0 + fc]);
                }
                window.push((u64::from(mine.cut.block_tid(peer)), win, vec![fr, fc]));
            }
        }
    }
    // ── The lockstep reference, to its fixed point ──
    let views: Vec<StagedWeightView<'_>> = probes
        .iter()
        .map(|p| StagedWeightView {
            bytes: &p.view_bytes,
            gemm: &p.gemm,
            scale_rows: &p.scale_rows,
            load_consts: &p.load_consts,
        })
        .collect();
    let collectives = ir.nodes.iter().filter(|n| is_collective(&n.op)).count();
    let mut golden: Vec<Vec<Vec<f32>>> = vec![Vec::new(); ranks.len()];
    let mut settled = false;
    for _ in 0..=collectives + 1 {
        let next: Vec<Vec<Vec<f32>>> = ranks
            .iter()
            .enumerate()
            .map(|(r, me)| {
                eval_golden_buffers(
                    ir,
                    &probes[r].staged_rope,
                    &views[r],
                    covering,
                    Some(GatePeer {
                        rank: me.get(),
                        ranks: &golden,
                    }),
                    &seed(r),
                )
            })
            .collect();
        settled = next == golden;
        golden = next;
        if settled {
            break;
        }
    }
    if !settled {
        return Err(format!(
            "the {}-rank reference did not settle in {} passes over {collectives} collective(s)",
            ranks.len(),
            collectives + 2
        ));
    }
    // ── Each rank judged against its own reference ──
    let mut worst: Vec<GateWorst> = Vec::with_capacity(ranks.len());
    let mut compared = 0;
    for (r, me) in ranks.iter().enumerate() {
        let compare: Vec<(u64, usize)> = golden_index
            .tids
            .iter()
            .filter(|t| observables[r].iter().any(|o| o.contains(t)))
            .map(|t| (*t, 0))
            .collect();
        compared += compare.len();
        let rank_bufs: Vec<Vec<Vec<f32>>> = outs
            .iter()
            .map(|o| {
                let mut b: Vec<Vec<f32>> = vec![Vec::new(); ir.tensors.len()];
                for (t, v) in o {
                    if let Some(slot) = b.get_mut(*t as usize) {
                        slot.clone_from(v);
                    }
                }
                b
            })
            .collect();
        worst.push(
            judge_gate_run(
                GateRun {
                    ir,
                    prefill,
                    compare: &compare,
                    out: &outs[r],
                    tainted,
                    golden_op: &golden_index.op,
                    golden_ins: &golden_index.ins,
                    view: &views[r],
                    covering,
                    fp,
                    cap,
                    peer: Some(GatePeer {
                        rank: me.get(),
                        ranks: &rank_bufs,
                    }),
                },
                golden[r].clone(),
            )
            .map_err(|e| format!("rank {r}: {e}"))?,
        );
    }
    // ── Replicated values: bit-identical on every emulated rank ──
    let same = |a: &[f32], b: &[f32]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    let mut replicated = 0;
    let mut split: Vec<String> = Vec::new();
    for &t in &golden_index.tids {
        let g0 = &golden[0][t as usize];
        if g0.is_empty() || !golden.iter().all(|g| same(g0, &g[t as usize])) {
            continue;
        }
        let emitted: Option<Vec<&Vec<f32>>> = outs.iter().map(|o| o.get(&t)).collect();
        if let Some(emitted) = emitted {
            replicated += 1;
            if !emitted.iter().all(|e| same(emitted[0], e)) {
                split.push(format!(
                    "t{t} ({})",
                    golden_index.op.get(&t).copied().unwrap_or("?")
                ));
            }
        }
    }
    if !split.is_empty() {
        return Err(format!(
            "{} of {replicated} replicated tensor(s) differ between the emulated ranks — every \
             rank must compute a replicated value bit-identically: [{}]",
            split.len(),
            split.join(", ")
        ));
    }
    let (lx_peak, lx_cap) = sessions
        .iter()
        .map(|s| (s.lx_peak_used(), s.lx_capacity()))
        .max()
        .unwrap_or((0, 0));
    let w = |f: fn(&GateWorst) -> f32| worst.iter().map(f).fold(0f32, f32::max);
    eprintln!(
        "[spyre-superdsc] NUMERIC GATE PASS {fp} ({} ranks, {} gated collective(s) of \
         {collectives}): {compared} rank-tensors at the floor, {replicated} replicated tensors \
         bit-identical across ranks (worst local clean {:.4}, worst local fp8-chained {:.4}; \
         worst total clean {:.4}, worst total fp8-chained {:.4}; LX peak {lx_peak} of {lx_cap} \
         bytes){scope}",
        ranks.len(),
        collective_nodes.len(),
        w(|g| g.local_clean),
        w(|g| g.local_fp8),
        w(|g| g.total_clean),
        w(|g| g.total_fp8),
    );
    Ok(())
}

/// One compared tensor's verdict, as the refusal dump states it — the numbers the
/// comparison loop already held, kept beside the buffers they judged so the dump is
/// one self-describing artifact.
#[cfg(feature = "spyre-emu")]
struct ComparedVerdict {
    tid: u64,
    n: usize,
    rel: f32,
    tol: f32,
    fp8: bool,
    /// The registry name of the op that first produced this tensor — a refusal
    /// names the CLASSES that diverge, not just their tids.
    op: &'static str,
    /// The producing node's input tids — a magnitude-dependent mechanism (a
    /// reduce's landing grid is a function of its input scale) is judged from
    /// the input buffers, which the dump already holds whenever the inputs are
    /// themselves compared tids.
    ins: Vec<u32>,
}

/// ⭐ A REFUSAL CARRIES ITS EVIDENCE. The emu and golden buffers the numeric gate
/// holds in process at the moment it refuses are the bisection material — the
/// per-layer rel curve, the onset, the garbage's shape — and a panic alone loses
/// them. So the refusal writes them out before returning: every judged tensor's
/// golden (`g_t{tid}.bin`) and emu (`e_t{tid}.bin`) values as LE f32, plus an
/// `order.json` (the verdicts with their tolerances, the failure lines, the worst
/// floors per class) that joins them, under the bake's own
/// [`crate::superdsc_bake::StageRoot`] — where files are cheap and nothing
/// outlives the build's temp dir. Best effort by necessity: a staging dir that
/// cannot be written must not mask the refusal it was going to document, so an IO
/// failure degrades to a note in the message, never a different error.
#[cfg(feature = "spyre-emu")]
fn write_refusal_dump(
    fp: &str,
    cap: u32,
    failures: &[String],
    verdicts: &[ComparedVerdict],
    golden: &[Vec<f32>],
    emu: &std::collections::HashMap<u64, Vec<f32>>,
    evidence: &RefusalEvidence<'_>,
) -> std::io::Result<std::path::PathBuf> {
    let dir = crate::superdsc_bake::StageRoot::resolve()
        .path()
        .join("gate_refusal")
        .join(format!("{fp}_cap{cap}"));
    std::fs::create_dir_all(&dir)?;
    let le = |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for v in verdicts {
        std::fs::write(
            dir.join(format!("g_t{}.bin", v.tid)),
            le(&golden[v.tid as usize]),
        )?;
        if let Some(e) = emu.get(&v.tid) {
            std::fs::write(dir.join(format!("e_t{}.bin", v.tid)), le(e))?;
        }
    }
    let worst = |fp8: bool| {
        verdicts
            .iter()
            .filter(|v| v.fp8 == fp8)
            .map(|v| v.rel)
            .fold(0f32, f32::max)
    };
    let order = serde_json::json!({
        "fp": fp,
        "cap": cap,
        "judged": verdicts.len(),
        "worst_clean": worst(false),
        "worst_fp8": worst(true),
        "failures": failures,
        "verdicts": verdicts
            .iter()
            .map(|v| serde_json::json!({
                "tid": v.tid, "op": v.op, "ins": v.ins, "n": v.n, "rel": v.rel, "tol": v.tol,
                "fp8": v.fp8, "failed": v.rel > v.tol,
            }))
            .collect::<Vec<_>>(),
        // The teacher-forced split: local = the producing op's own error over
        // the emulator's inputs; inherited = what the inputs' drift alone
        // predicts — read from the golden's own slots, shared not copied.
        "split": evidence.split
            .iter()
            .map(|r| serde_json::json!({
                "tid": r.tid, "node": r.node, "op": r.op, "fp8": r.fp8,
                "local": r.local, "inherited": r.inherited,
            }))
            .collect::<Vec<_>>(),
        "restarts": evidence.restarts
            .iter()
            .map(|r| serde_json::json!({
                "from_frac": r.from_frac, "from_tid": r.from_tid, "tail_n": r.tail_n,
                "track_med": r.track_med, "contract_med": r.contract_med,
                "track_last": r.track_last, "contract_last": r.contract_last,
                "direct_last": r.direct_last,
            }))
            .collect::<Vec<_>>(),
    });
    std::fs::write(
        dir.join("order.json"),
        serde_json::to_string_pretty(&order).map_err(std::io::Error::other)?,
    )?;
    Ok(dir)
}

/// ⭐⭐⭐⭐⭐ TRIPS PER CONCRETE dxp BUNDLE, AS A COMPILE-TIME CONSTANT — and the reason it stopped
/// being an env read is that **the knob moves CORRECTNESS, which no authority had measured.**
///
/// ⛔ WHAT THE ENV READ COST: three rounds of control experiments that never controlled anything.
/// `SCRATCHY_SUPERDSC_GROUP_SIZE` was read by `std::env::var` at macro-expansion time and forwarded by
/// NO `build.rs`, so changing it did not invalidate a single cargo unit. MEASURED: a build wrapper
/// exporting 512 against a tree last built at the default finished in 5.20 s with the binary's md5 and
/// mtime UNCHANGED — forcing the re-emit needs `cargo clean --release -p scratchy-models` (456 s), and
/// `cargo clean -p` alone cleans the dev profile and does nothing. The pod's build wrappers export 512
/// while a bare `cargo build` gets 128, so a "restored file-by-file to md5 equality, rebuilt" control
/// compared a 128 bundle against 512 headline numbers and read the difference as nondeterminism.
///
/// ⭐ THE TWO VALUES ARE FREE IN SPEED AND NOT FREE IN OUTPUT. Three separate measurements agree that
/// 128 and 512 cost the same: `spyre_exec.rs`'s 420-token probe (avg ITL 87.5 vs 87.5 ms, wall 45.5 vs
/// 45.4 s), the 8b TTFT/ITL comparison below, and a w8 ladder (ITL med 62.8–63.1 vs 62.6–62.8 ms). But
/// on granite-3.1-2b fp8 at width 8 against its own solo oracle, `solo_diff` is **8,8,8,8,8** at 128
/// (N=5) and **7,6,7** at 512 (N=3) — and the SURVIVING row moves between identical trials. The 8b
/// descriptor multiset is byte-identical across a 128 and a 512 bake (134 bundles, 94 distinct
/// descriptor md5s, multisets equal), and `dxp_standalone` is bit-deterministic on a fixed partition
/// (N=14 bakes across thread counts, `taskset` and `DT_PARALLEL_THREADS`, every output md5-identical).
/// **So the partition into compile groups is the only variable, and it changes the answer.** A quantity
/// like that cannot live in the environment of whoever happened to run the build.
///
/// The `2048` the canonical pod env sets is a THIRD value and the "it tanks throughput" warning attached
/// to it is about 2048, not about 512 — conflating the two is what kept this a read.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct GroupSize(usize);

impl GroupSize {
    /// The largest partition the bake survives. `dxp_standalone` sizes its pool from
    /// `hardware_concurrency()` (`dscglobal.h:56`), which reports the HOST's cores and ignores the
    /// cgroup quota, and its thread count tracks the GROUP it is handed (MEASURED ~85 threads on a
    /// 102-descriptor group against ~192 on a 300-descriptor one). Past the pod's ~3.1k–3.9k concurrent
    /// thread ceiling, `COMPILE_WIDTH` children of a large group die with
    /// `LLVM ERROR: pthread_create failed: Resource temporarily unavailable`. 2048 reaches it; 512 is
    /// the largest value observed to bake.
    const CEILING: usize = 512;

    /// The ONE value the emission uses. 128 is what an unset variable produced, so this is a pure
    /// constant-ification of a bare `cargo build` and of CI — no bundle fingerprint moves for them.
    /// ⛔ It is NOT what the pod wrappers were exporting, so pod headline numbers taken at 512 do not
    /// describe this build and must be re-measured, not carried over.
    pub const PRODUCTION: Self = Self::new(128);

    /// Const-evaluated, so an out-of-range group is `error[E0080]` at build time rather than a
    /// part-way dxp death or a silently different partition.
    const fn new(trips: usize) -> Self {
        assert!(trips >= 1, "a launch group holds at least one trip");
        assert!(
            trips <= Self::CEILING,
            "group size exceeds the measured dxp bake thread ceiling — the bake dies part-way with \
             `pthread_create failed`, see GroupSize::CEILING"
        );
        Self(trips)
    }

    /// Trips in one group. `1` = the historical per-op path (each trip its own program), which is the
    /// fault-isolation end of the range.
    pub const fn trips(self) -> usize {
        self.0
    }
}

/// ⛔⭐ THIS LINE IS WHAT MAKES THE CEILING A LOCK, AND WITHOUT IT THERE IS NO LOCK AT ALL.
/// `new`'s asserts alone do NOT fire for [`GroupSize::PRODUCTION`]: an associated const in an inherent
/// impl is evaluated ON DEMAND, and a read from a runtime expression (`group_size()`) does not demand it
/// — MEASURED, `Self::new(513)` passed `cargo check` clean and recompiled in 0.40 s with no diagnostic.
/// A module-level `const _: ()` is a required-const context, so it forces the evaluation and turns an
/// out-of-range group into `error[E0080]`. Mutation-checked in both directions: 513 is E0080 here, 128 is
/// clean. Same defect class as [`an-unused-associated-const-is-not-a-static-assert`] — do not remove
/// this in the belief that the `const fn` covers it.
const _: () = {
    assert!(GroupSize::PRODUCTION.trips() >= 1);
    assert!(GroupSize::PRODUCTION.trips() <= GroupSize::CEILING);
};

/// Medium-grain fusion group size (trips per concrete dxp bundle) — [`GroupSize::PRODUCTION`].
/// Stays a function so the pure `group_ranges` takes it as an argument and CBMC keeps its bound
/// (see `plan_capped`); it no longer reads the environment.
pub fn group_size() -> usize {
    // Decode is ~100% launch-count bound (measured ~3640 launches/tok @ ~41µs); fusing consecutive ops
    // into ≤g groups collapses that. 2048 was too aggressive — a fp8-dynamic granite prefill body fused
    // 512-trip groups fine but a dxp_standalone `vector::_M_range_check` crash surfaced once fusion pushed
    // past that. `GroupSize::new(1)` is the per-op fault-isolation end, reached by a source edit.
    //
    // ⛔ THE CLAIM THAT STOOD HERE — "forwarded by each arch build.rs (rerun-if-env-changed +
    // rustc-env) so a change recompiles" — WAS FALSE. No build.rs mentioned the variable; nothing
    // invalidated a cargo unit; the emitted bundle and the source went out of sync silently. See
    // `GroupSize`.
    //
    // ⭐ DEFAULT = 128, DOWN FROM 512, AND THE REASON IS THE BAKE'S THREAD BUDGET — not runtime.
    // `dxp_standalone` sizes its pool from `hardware_concurrency()` (`dscglobal.h:56`), which reports the
    // HOST's cores and ignores the cgroup quota, and its thread count then tracks the GROUP it is handed:
    // MEASURED ~85 threads on a 102-descriptor group against ~192 on a 300-descriptor one, saturating near
    // `nproc`. The empirical ceiling on the build pod is ~3.1k-3.9k concurrent threads, so
    // `COMPILE_WIDTH` children of a large group exceed it and dxp dies part-way with
    // `LLVM ERROR: pthread_create failed: Resource temporarily unavailable`. Smaller groups mean fewer
    // threads per child at the SAME `COMPILE_WIDTH` — the width stays at main's 32.
    //
    // ⛔ AND THE ENV CAP DOES NOT SUBSTITUTE FOR IT. `DT_PARALLEL_THREADS` is deeptools' only thread
    // variable (`strings dxp_standalone`: `DT_DEEPRT_VERBOSE`, `DT_PARALLEL_THREADS`,
    // `DT_PROG_CRITERIA_FILEPATH`) and it is inert here — MEASURED 88 threads with `=1` against 84
    // without on one group, and 192 on a 300-descriptor group WITH it set. Nor does CPU affinity:
    // libstdc++ implements `hardware_concurrency()` over `sysconf(_SC_NPROCESSORS_ONLN)`, which reports
    // ONLINE cpus and ignores affinity, so neither `taskset` nor `sched_setaffinity` reaches the pool.
    // Group size is the only lever that does.
    //
    // ⭐ AND IT IS FREE AT RUNTIME. MEASURED on granite-3.1-8b fp8: neither TTFT nor ITL moves between
    // 512 and 128 — consistent with the recorded bs=1 result that a launch is essentially free, and with
    // the earlier 128-vs-512 comparison that found slow-mode ITL identical to within 0.1 ms.
    GroupSize::PRODUCTION.trips()
}

/// One emitted bundle, awaiting the device code its groups are still being compiled into.
///
/// ⛔ WHY THE TWO HALVES ARE SEPARATE. The bake queue is deliberately ASYNCHRONOUS — group N of this
/// bundle compiles while the NEXT bundle is being written, which is what keeps `dxp_standalone` running
/// `COMPILE_WIDTH` wide across bundle boundaries instead of winding down to one at each of them. So the
/// metadata is finished here and the programs are collected by [`Self::into_code`], after the ONE drain
/// at the end of the emit.
pub struct EmittedBundle<T: scratchy_spyre_bundle::collective::Tp> {
    /// Content fingerprint — this bundle's identity, and how a sibling names it.
    pub fp: String,
    /// The memory plan. Empty (`Default`) when the caller supplied no layout.
    pub layout: bundle::BundleLayout<'static>,
    pub groups: PendingGroups<T>,
}

/// A bundle's launches, either already complete or waiting on the device compiler.
pub enum PendingGroups<T: scratchy_spyre_bundle::collective::Tp> {
    /// ⭐ EVERY LAUNCH, IN LAUNCH ORDER, COMPLETE — the `-Fspyre-emu` shape.
    ///
    /// A launch group carries the PROGRAMS it runs, so there is nothing left to resolve later:
    /// no compiler to wait on, no id to look a device image up by. A constructed program is
    /// finished the moment it is constructed.
    Ready(Vec<bundle::LaunchGroup<'static, T>>),
    /// `-Fspyre-hw`'s shape: each group's SuperDSC directory is staged and submitted
    /// ([`launch_groups`]), and its `(KvShifts, GroupId)` is all `into_code` needs to
    /// find the compiled artifact once `superdsc_bake::finish_global()` has drained the queue —
    /// exactly the id `emit_bundle_inner` "used to" carry before this branch pivoted to KTIR.
    Staged(Vec<(bundle::KvShifts<T>, crate::superdsc_bake::GroupId)>),
}

/// Every `scf.for` in `body` (recursing through nested regions) as `(which, lower, upper, step)`,
/// resolving each bound through the `arith.constant` that defines it. Bounds are compile-time
/// constants on this path — the tiling computes them — so a bound with no constant definition is
/// not a loop this audit can speak about, and is skipped rather than guessed at.
fn loop_bounds(body: &[ktir_core::ir::Operation<'_>]) -> Vec<(String, i64, i64, i64)> {
    fn walk(
        ops: &[ktir_core::ir::Operation<'_>],
        consts: &mut BTreeMap<u32, i64>,
        out: &mut Vec<(String, i64, i64, i64)>,
    ) {
        for op in ops {
            if op.op_type == OpKind::ArithConstant
                && let Some(r) = op.result
                && let Some((_, Attr::Int(v))) =
                    op.attributes.iter().find(|(k, _)| *k == AttrKey::Value)
            {
                consts.insert(r.0, *v);
            }
            if op.op_type == OpKind::ScfFor && op.operands.len() >= 3 {
                let g = |i: usize| consts.get(&op.operands[i].0).copied();
                if let (Some(lo), Some(hi), Some(st)) = (g(0), g(1), g(2)) {
                    let which = op.result.map_or_else(
                        || "an scf.for".to_string(),
                        |r| format!("the scf.for defining %{}", r.0),
                    );
                    out.push((which, lo, hi, st));
                }
            }
            for r in op.regions {
                walk(r, consts, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(body, &mut BTreeMap::new(), &mut out);
    out
}

/// ⛔⛔⛔ THE WORK AUDIT — A PROGRAM THAT COULD WAIT FOREVER MAY NOT COMPILE.
///
/// Every count in an emitted op is a COMPILE-TIME CONSTANT: the trip count, the grid, and every
/// `scf.for` bound are decided by the `#[forward]` procmacro before a single byte reaches the card.
/// So "will this program terminate, and will it do work" is a question the bake can answer, and the
/// card should never be the thing that discovers the answer is no — it discovers it by waiting,
/// with no host stack and no statement of which op is spinning.
///
/// ⛔ ZERO MEANS MAXIMUM ON THIS HARDWARE, NOT NOTHING. The device's own DMA and XLAT paths both
/// read `if (length == 0) length = 1 << 27` — a zero length field transfers 16 GiB. A zero count is
/// either a loop that runs never (silently wrong) or one that runs to a maximum nobody intended.
///
/// Checked per op:
///   * the func's own grid names at least one core, and — on `spyre-hw` only, where the limit
///     exists — no more than the generation's `CORES`
///   * every `scf.for` in the program runs a non-empty range with a positive step
fn audit_op_work<T: scratchy_spyre_bundle::collective::Tp>(ops: &[EmittedOp<T>], fp: &str) {
    let mut fail: Vec<String> = Vec::new();

    for op in ops {
        let name = &op.op_name;
        // ⭐ THE PROGRAM'S OWN GRID IS THE CORE COUNT. `numCoresUsed_` was the descriptor's answer
        // to the same question — how many units run this op — and reading a core count off a
        // descriptor has no KTIR counterpart. The func STATES its division, so the guard asks it
        // there: zero cores is an op no unit will ever run, and more than the machine has is a
        // schedule the hardware cannot satisfy.
        let (gx, gy, gz) = op.ktir.func.grid;
        let cores = (gx * gy * gz) as u32;
        if cores == 0 {
            fail.push(format!(
                "{name}: its grid names 0 cores — no unit will ever run this op"
            ));
        }
        // ⭐ THE CORE COUNT IS A CARD FACT — the build's generation's `DeviceGen::CORES`. A grid
        // wider than the card is a
        // schedule no card can satisfy, so on hardware it is fatal here rather than at submit. The
        // emulator has no such ceiling: it runs the tile grid the func declares, which is how a
        // prefill of more than 32 tokens gets one core per token, so this is checked where the
        // limit exists and nowhere else.
        #[cfg(feature = "spyre-lower")]
        if cores > <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::CORES {
            fail.push(format!(
                "{name}: its grid names {cores} cores but the card has {}",
                <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::CORES
            ));
        }
        // ⭐ EVERY LOOP BOUND THE PROGRAM STATES. The sdsc/core/corelet fold factors were the
        // descriptor's loop counts; KTIR states its counts as the `scf.for` bounds themselves, so
        // the SAME law is checked where KTIR keeps it. The law is unchanged: a count must be
        // positive, because a zero-valued count field on this device reads as its MAXIMUM — the op
        // does not fail, it runs never or it runs forever, and either way the host waits for a
        // completion that is not coming.
        {
            for (which, lo, hi, step) in loop_bounds(op.ktir.func.operations) {
                if hi <= lo {
                    fail.push(format!(
                        "{name}: {which} runs [{lo}, {hi}) — an empty loop, so the op does no work"
                    ));
                }
                if step <= 0 {
                    fail.push(format!(
                        "{name}: {which} has step {step} — a loop count must be positive, and a \
                         zero-valued count field on this device reads as its MAXIMUM"
                    ));
                }
            }
        }
    }

    if !fail.is_empty() {
        let shown = fail.len().min(12);
        panic!(
            "\n⛔ [superdsc-work] bundle {fp}: {} OP(S) WITH NO PROVABLE WORK. Refusing to bake.\n\n{}\n{}\n             Every count here is a compile-time constant. A program whose loop bound is zero does \
             not fail on the card — it runs never, or it runs to a maximum, and either way the \
             host waits for a completion that is not coming.\n",
            fail.len(),
            fail[..shown]
                .iter()
                .map(|f| format!("  • {f}"))
                .collect::<Vec<_>>()
                .join("\n"),
            if fail.len() > shown {
                format!("  … and {} more\n", fail.len() - shown)
            } else {
                String::new()
            },
        );
    }
}

/// Attach the layer-loop parameters to the re-rolled BODY named by `fp`.
///
/// Separate from [`emit_bundle`] because a body's meta names its prefix, suffix, fused twin and every
/// ladder rung — none of which exist yet when the body itself is emitted.
pub fn attach_reroll<T: scratchy_spyre_bundle::collective::Tp>(
    fp: &str,
    meta: bundle::RerollMeta<'static>,
) {
    if let Ok(mut c) = emitted::<T>().lock()
        && let Some(slot) = c.get_mut(fp)
    {
        slot.1 = Some(meta);
    }
}

/// ⛔⛔⛔ ONE SESSION ALLOCATES ONE MEMORY PLAN, AND THAT PLAN MUST HOLD EVERY BODY THE SESSION LAUNCHES.
///
/// A re-rolled body's session allocates each segment ONCE, at the extent ITS OWN bundle declares
/// (`Executor::load` reads `code.layout.segment_bytes`), and then launches every sibling against those
/// regions: the prefix, the suffix, the fused twin, and every ladder rung's body. Each sibling was
/// emitted against its OWN layout. [`ladder_stages_one_layout`] holds a rung's host-staged activations
/// to the ceiling's, but nothing held a rung's SEGMENT EXTENTS to the session's. They fit only because
/// the ceiling was the widest chunk, and the widest chunk had the widest HBM scratches.
///
/// ⛔ MEASURED ON CARD, granite-3.1-2b fp16 (pod nickm), when that stopped being true. With the split-KV
/// chunk's Kᵗ and V gathered into LX, the
/// 64-page ceiling keeps NO gather scratch in HBM, while a rung with a tail keeps HBM scratches the
/// ceiling lacks:
///   * with Kᵗ whole in HBM at the 12-page rung, its first step faults `RAS::PCI::BusFence` (0xa35e,
///     "PCIe bus master fence"). The rung's dxp input was byte-identical, op by op, to the V-only
///     build's, which runs that rung. Only the session's extents had changed.
///   * with only Kᵗ's and V's 4-page tails in HBM, the 12- and 20-page rungs faulted `HMI=STORE` at
///     `0x8dbf00`, `0x8dfb00` (12 pages) and `0x8ddb00` (20 pages). Those rungs lack the same extent;
///     that form has not been re-run under this law.
///
/// So the session's extent is the high-water mark of everything it launches, segment by segment
/// ([`session_segment_bytes`]), and it is decided here, at bake. A segment that would then outgrow one
/// device region is a build `Err`. Over-allocating a segment is inert for every body that addresses
/// less of it; under-allocating it is an address. Under this law the first case's 12-page rung runs,
/// with replies byte-identical to the V-only build's: a 2,413-token prompt that decodes in that rung
/// from its first step, and a 1,987-token one that crosses into it from the 8-page rung.
pub fn cover_ladder_allocation<T: scratchy_spyre_bundle::collective::Tp>(
    fp: &str,
) -> Result<(), String> {
    let mut c = emitted::<T>()
        .lock()
        .map_err(|_| "the emit collector was poisoned by a panic".to_string())?;
    let siblings: Vec<String> = c
        .get(fp)
        .ok_or_else(|| format!("bundle {fp} was never emitted"))?
        .1
        .as_ref()
        .ok_or_else(|| format!("bundle {fp} has no layer loop attached, so no siblings to cover"))?
        .siblings()
        .map(str::to_string)
        .collect();
    let bodies = siblings
        .iter()
        .map(|s| {
            c.get(s.as_str())
                .map(|(b, _)| b.layout.segment_bytes)
                .ok_or_else(|| format!("bundle {fp} names sibling {s}, which was never emitted"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let slot = c
        .get_mut(fp)
        .ok_or_else(|| format!("bundle {fp} was never emitted"))?;
    let covered = session_segment_bytes(slot.0.layout.segment_bytes, bodies);
    if let Some((seg, &bytes)) = covered
        .iter()
        .enumerate()
        .find(|(_, b)| **b > bundle::MAX_SEGMENT_BYTES)
    {
        return Err(format!(
            "bundle {fp}: covering its {} sibling(s) grows seg{seg} to {bytes} B, past the {} B one \
             device region can hold",
            siblings.len(),
            bundle::MAX_SEGMENT_BYTES
        ));
    }
    slot.0.layout.segment_bytes = covered;
    Ok(())
}

/// The extents one session must allocate: each segment at the largest extent the session's own bundle
/// or any body it launches declares. See [`cover_ladder_allocation`].
pub fn session_segment_bytes<const N: usize>(
    session: [u64; N],
    bodies: impl IntoIterator<Item = [u64; N]>,
) -> [u64; N] {
    bodies.into_iter().fold(session, |mut acc, b| {
        for (a, x) in acc.iter_mut().zip(b) {
            *a = (*a).max(x);
        }
        acc
    })
}

#[cfg(test)]
mod session_allocation_tests {
    use super::session_segment_bytes;

    /// The card's case: the 64-page ceiling keeps its chunk gather in LX, so its intermediate segment
    /// is SMALLER than a tail rung's, which keeps Kᵗ in HBM. The session must allocate the rung's.
    #[test]
    fn a_session_allocates_every_segment_at_the_widest_body_it_launches() {
        let mib = 1u64 << 20;
        let ceiling = [9 * mib, 2_500 * mib, 128, 800 * 1024, 100 * 1024, 0, 0];
        let tail_rung = [12 * mib, 2_500 * mib, 128, 790 * 1024, 100 * 1024, 0, 0];
        let narrow_rung = [8 * mib, 2_500 * mib, 128, 780 * 1024, 100 * 1024, 0, 4096];
        let s = session_segment_bytes(ceiling, [tail_rung, narrow_rung]);
        assert_eq!(
            s,
            [12 * mib, 2_500 * mib, 128, 800 * 1024, 100 * 1024, 0, 4096]
        );
        for body in [ceiling, tail_rung, narrow_rung] {
            assert!(
                body.iter().zip(&s).all(|(b, a)| b <= a),
                "a body addresses past its session's allocation"
            );
        }
    }

    #[test]
    fn a_session_with_no_siblings_allocates_its_own_extents() {
        let own = [1u64, 2, 3, 4, 5, 6, 7];
        assert_eq!(session_segment_bytes(own, []), own);
    }
}

/// Borrow the launch groups of the ALREADY-EMITTED bundle named `fp` — the
/// same groups [`drain_emitted_bundles`] will hand the macro's emission, run
/// under the registry lock so nothing is copied or taken. The bake-time
/// [`numeric_gate`] uses this to run the just-baked program; `None` when the
/// fp has no bundle (not emitted, or emitted but still waiting on a device
/// compiler — the `-Fspyre-hw` shape, where there is nothing to gate).
pub fn with_emitted_groups<T: scratchy_spyre_bundle::collective::Tp, R>(
    fp: &str,
    f: impl FnOnce(&[bundle::LaunchGroup<'static, T>]) -> R,
) -> Option<R> {
    let c = emitted::<T>().lock().ok()?;
    let (b, _) = c.get(fp)?;
    match &b.groups {
        PendingGroups::Ready(groups) => Some(f(groups)),
        PendingGroups::Staged(_) => None,
    }
}

/// EVERY BUNDLE THIS EMIT PRODUCED, by fingerprint — the channel the macro drains.
///
/// ⛔ A REGISTRATION, NOT A SEARCH. A bundle is in here because [`emit_bundle`] put it here, so
/// [`drain_emitted_bundles`] cannot silently return fewer bundles than were emitted.
///
/// ⭐ ONE COLLECTOR PER WORLD. A bundle's world is its type, so each world's bundles are collected
/// apart, keyed by the world's own `TypeId` — the macro drains the world it bakes
/// (`drain_emitted_bundles::<T>`), and a bundle of one world is never handed to another's drain.
fn emitted<T: scratchy_spyre_bundle::collective::Tp>() -> &'static Collector<T> {
    use std::any::{Any, TypeId};
    type Worlds = std::sync::Mutex<BTreeMap<TypeId, &'static (dyn Any + Send + Sync)>>;
    static WORLDS: std::sync::OnceLock<Worlds> = std::sync::OnceLock::new();
    let mut worlds = WORLDS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry: &'static (dyn Any + Send + Sync) = *worlds
        .entry(TypeId::of::<T>())
        .or_insert_with(|| Box::leak(Box::new(Collector::<T>::default())));
    // The entry under `TypeId::of::<T>()` is only ever this arm's own `Collector<T>`.
    match entry.downcast_ref::<Collector<T>>() {
        Some(c) => c,
        None => unreachable!("the collector keyed by T's TypeId is a Collector<T>"),
    }
}

/// One world's emitted bundles, by fingerprint.
type Collector<T> =
    std::sync::Mutex<BTreeMap<String, (EmittedBundle<T>, Option<bundle::RerollMeta<'static>>)>>;

/// The lock [`emit_bundle`] holds for one fingerprint while it checks [`emitted`] and emits.
fn emitting(fp: &str) -> std::sync::Arc<std::sync::Mutex<()>> {
    static L: std::sync::OnceLock<
        std::sync::Mutex<BTreeMap<String, std::sync::Arc<std::sync::Mutex<()>>>>,
    > = std::sync::OnceLock::new();
    let mut gates = L
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::sync::Arc::clone(gates.entry(fp.to_string()).or_default())
}

/// Take every emitted bundle, complete with its compiled device code — what the macro bakes.
///
/// Call ONCE, after `superdsc_bake::finish_global()` has drained the compilers.
pub fn drain_emitted_bundles<T: scratchy_spyre_bundle::collective::Tp>()
-> Result<Vec<bundle::BundleCode<'static, T>>, String> {
    let taken = match emitted::<T>().lock() {
        Ok(mut c) => std::mem::take(&mut *c),
        Err(_) => return Err("the emit collector was poisoned by a panic".to_string()),
    };
    taken
        .into_values()
        .map(|(b, reroll)| b.into_code(reroll))
        .collect()
}

impl<T: scratchy_spyre_bundle::collective::Tp> EmittedBundle<T> {
    /// Collect this bundle's compiled programs and produce the value the macro bakes.
    ///
    /// Call AFTER `superdsc_bake::finish_global()`. A group the queue has no result for is an `Err`
    /// naming it: after a successful drain that can only mean it was never submitted, and a bundle with
    /// a hole in its launch sequence must not reach the binary.
    pub fn into_code(
        self,
        reroll: Option<bundle::RerollMeta<'static>>,
    ) -> Result<bundle::BundleCode<'static, T>, String> {
        let groups = match self.groups {
            PendingGroups::Ready(g) => g,
            PendingGroups::Staged(staged) => {
                let bake = crate::superdsc_bake::global();
                let mut out = Vec::with_capacity(staged.len());
                for (kv, id) in &staged {
                    // `bake` is `None` on a `spyre-lower` build without `spyre-hw` (no device image
                    // is compiled) — `staged_groups` staged/submitted nothing FOR that
                    // same reason, so there is nothing to have lost here either. Empty
                    // `init_binary` matches `ktir_groups`'s own plan-only default.
                    let (init_binary, job_bin_ptr, correction) = match bake {
                        None => (Vec::new(), 0, Vec::new()),
                        Some(b) => {
                            let compiled = b.compiled_group(id).ok_or_else(|| {
                                format!(
                                    "{id}: submitted for compilation but no compiled program \
                                     came back — the bake queue drained without an error, so \
                                     this group was never submitted"
                                )
                            })?;
                            (
                                compiled.init_binary.clone(),
                                compiled.job_bin_ptr,
                                compiled.correction.clone(),
                            )
                        }
                    };
                    out.push(bundle::LaunchGroup {
                        kv: *kv,
                        // The hw runtime never reads `programs` — it executes the compiled
                        // `init_binary` below, not the KTIR that was lowered FROM. See
                        // `bundle::LaunchGroup::programs`'s own doc: "on a card the device image
                        // is compiled FROM these, and is their artifact rather than a second
                        // source of truth."
                        programs: std::borrow::Cow::Borrowed(&[]),
                        init_binary: bundle::BakedImage::Raw(std::borrow::Cow::Owned(init_binary)),
                        job_bin_ptr,
                        correction: std::borrow::Cow::Owned(correction),
                    });
                }
                out
            }
        };
        // ⭐ EVERY SEND NAMES BAKED BYTES, OR THE BUNDLE DOES NOT BAKE. A consume's send reads its
        // operand's placement and lands at the receive window's; the loader reads them through this
        // same resolution, so a bundle that passes here never meets an unplaced send at load.
        bundle::collective::sends(&self.layout, &groups).map_err(|u| {
            format!(
                "{}: group {}'s collective send names no baked bytes: {:?}",
                self.fp, u.group, u.why
            )
        })?;
        Ok(bundle::BundleCode {
            fp: std::borrow::Cow::Owned(self.fp),
            layout: self.layout,
            groups: std::borrow::Cow::Owned(groups),
            reroll,
        })
    }
}

/// ONE ATTENTION CLASS'S ROLLED BODY — its ops, its layer count, and its share of the
/// per-layer tid map.
///
/// A class-split roll (`reroll_layer_classes`) emits one of these per distinct layer shape; a
/// single-class roll (every model but the hybrid-attention ones) emits exactly one, which is the
/// case `RolledSuperDsc` was before classes existed. MEASURED on this ladder: the 26b MoE under
/// the WELDED single-body roll (the re-host's stopgap, since reverted) ran the whole model as 5
/// iterations of the 6-layer cell with the launch shifts derived from ONE class's placements —
/// 25 of 30 layers addressed through the wrong class's KV stride, faulting the card at the first
/// decode step. The class split is not an optimization; it is the only shape whose launch table
/// can address a hybrid model.
#[derive(Clone)]
pub struct RolledBody<T: scratchy_spyre_bundle::collective::Tp> {
    /// ONE layer's ops, baked at the class's layer-0 offsets.
    pub ops: Vec<EmittedOp<T>>,
    /// How many layers of THIS class the body serves.
    pub iters: u32,
    /// The ABSOLUTE layer ids this body serves, in class order — `[ids[iter]]` is the true
    /// layer run by iteration `iter`. Derived from the body's own `per_layer_out` rope table
    /// (`layer_index()`), so the true order cannot drift from what the tape says.
    pub layer_ids: Vec<u32>,
    /// This body's residual-stream INPUT/OUTPUT tids (first/last body node) — see
    /// [`RolledSuperDsc`] for the threading.
    pub hidden_in_tid: u32,
    pub hidden_out_tid: u32,
}

impl<T: scratchy_spyre_bundle::collective::Tp> RolledBody<T> {
    /// The absolute layer run at class-relative `iter`, or `None` past the end. Used by
    /// the executor's launch sequence derivation; `unroll_layers` reads `layer_ids` directly.
    pub fn layer_at(&self, iter: u32) -> Option<u32> {
        self.layer_ids.get(iter as usize).copied()
    }
}

/// One RE-ROLLED SuperDSC decode — the layer loop stays ROLLED (not 30× unrolled).
/// `prefix` = pre-loop ops (embed), `bodies[c]` = ONE layer's ops of attention class
/// `c` (the executor runs each for that class's layers, in [`Self::launches`] order),
/// `suffix` = post-loop ops (final norm + lm_head). `per_layer[t0]` = `[t0, t1, …]` —
/// the layer-v tensor id for each tensor a body references by its class's layer-0 id
/// `t0` (the node outputs + per-layer weights/KV). dxp then compiles SMALL bundles
/// (seconds) instead of one 2047-op unrolled monster (40 min). `layout` places EVERY
/// layer's tensors (the full resident set), so each `per_layer[t][v]` has a real
/// address.
///
/// ⭐ THE RESIDUAL THREADS IN-PLACE across EVERY body: the layout aliases each body's
/// `hidden_out_tid` and the next body's `hidden_in_tid` onto the prefix's hidden buffer,
/// so the host neither copies nor addresses them (see the pre-pass in
/// `lower_subtile_tape_to_ktir`).
#[derive(Clone)]
pub struct RolledSuperDsc<T: scratchy_spyre_bundle::collective::Tp> {
    pub prefix: Vec<EmittedOp<T>>,
    /// ONE entry per attention class, in the order [`bundle::RerollMeta::bodies`] bakes.
    pub bodies: Vec<RolledBody<T>>,
    pub suffix: Vec<EmittedOp<T>>,
    pub layout: BundleLayout,
    /// Per-layer tid map, keyed by each class's representative (layer-0-of-class) tid.
    /// A table's length is ITS class's `iters` — two classes of different layer counts
    /// carry different-length tables under the same map.
    pub per_layer: std::collections::BTreeMap<u32, Vec<u32>>,
    /// ⭐ THE LAYER SEQUENCE, IN TRUE ORDER — one entry per layer of the whole model,
    /// naming its body (index into `bodies`) and its class-relative iteration. The
    /// executor runs exactly this; a single-class roll degenerates to
    /// `[(0, 0), (0, 1), …]`, which is the sequence the old `for v in 0..iters` ran.
    pub launches: Vec<(u8, u32)>,
    /// Per-launch weight-segment `(bank, offset)` — `launches`' addressing half, proven
    /// against the placements at cargo-build (see [`bundle::LayerLaunch`]).
    pub launch_w: Vec<(u8, u64)>,
    /// Per-launch KV region — the layout's own bank for the launch's block
    /// ([`BundleLayout::kv_block_banks`]).
    pub launch_kv: Vec<KvBank>,
    /// The weight bank the PREFIX's weight operands live in, and the SUFFIX's — proven to be a single
    /// bank each (a launch has ONE base per segment). 0 when the group reads no weights.
    pub prefix_weight_bank: u32,
    pub suffix_weight_bank: u32,
    /// Bytes between two REQUESTS' KV within one page+layer — 0 (no request dimension in
    /// a page any more; kept for the wire shape `RerollMeta` does not carry).
    pub kv_request_stride: u64,
    /// The suffix's residual INPUT tensor id (the first suffix node's input[0], e.g.
    /// the last layer's post-attn residual t780). The rerolled body writes its output
    /// to its own `hidden_out_tid` (the representative-iteration tid), which differs from
    /// `suffix_in_tid`, so the placement aliases them — the body→suffix seam threads
    /// in place like the per-iter residual. `u32::MAX` if there is no suffix.
    pub suffix_in_tid: u32,
}

impl<T: scratchy_spyre_bundle::collective::Tp> RolledSuperDsc<T> {
    /// The total layer count — the sum over classes.
    pub fn iters(&self) -> u32 {
        self.bodies.iter().map(|b| b.iters).sum()
    }
}

/// ⭐⭐⭐ THE LAYER LOOP, UNROLLED INTO TENSOR IDS — the KTIR form of the executor's per-iteration
/// address bind.
///
/// ⛔ `RerollMeta`'S LAUNCH OFFSETS HAVE NO KTIR COUNTERPART. They are byte shifts into the
/// resident weight and KV SEGMENTS: at layer `v` the card's executor binds
/// `seg_base + launch.w_off` so one baked body addresses every layer. A KTIR launch binds no
/// segment base — it binds a TENSOR, and the emulator threads one buffer per tensor id — so there
/// is no offset to advance.
///
/// ⭐ BUT THE FACT ITSELF SURVIVES, BECAUSE IT WAS NEVER REALLY AN OFFSET. `per_layer[t0][v]` is
/// "the layer-`v` tensor that the body names by its class's layer-0 `t0`", and the offset is only
/// how the card reaches it. So the loop unrolls by REBINDING: launch `(c, j)` runs class `c`'s
/// programs against `per_layer[t][j]`. The layout already places every layer's tensors (see
/// [`RolledSuperDsc`]), so each rebound id has a real address.
///
/// ⭐ AND IT COSTS ALMOST NOTHING TO EMIT. A KTIR function is tensor-id-AGNOSTIC — its parameters
/// are `Ssa` indices and the tensor a parameter points at lives in `LaunchProgram::args` — so
/// `iters` copies of a body are `iters` copies of the ARGS over ONE shared program. That is why
/// unrolling here does not reintroduce the unrolled-bundle compile blowup the reroll was built to
/// avoid: the programs are interned once (`ktir_tokens::ProgramInterner`).
pub fn unroll_layers<T: scratchy_spyre_bundle::collective::Tp>(
    rolled: &RolledSuperDsc<T>,
) -> Vec<EmittedOp<T>> {
    let mut ops = rolled.prefix.clone();
    // ⭐⭐ THE LOOP-CARRIED RESIDUAL IS **NOT** IN `per_layer`, AND IT IS NOT AN OVERSIGHT.
    // [`RolledBody::hidden_in_tid`] documents the card's arrangement: the layout aliases every
    // body's hidden-out onto one resident buffer, so the residual threads in place. Aliasing is
    // addressing, so it has no KTIR counterpart — and its absence is not benign. Without the
    // rebinding below, every unrolled layer reads the SAME hidden-in (the prefix's output) and
    // only the last layer's write is ever read: the layers do not chain, and the model returns
    // one layer applied to the embedding.
    //
    // The KTIR form is a BINDING: launch `(c, j)` reads the PREVIOUS launch's hidden-out (as a
    // tensor id, `per_layer[body_c.hidden_out_tid][j]` — the same value the card's alias writes
    // in place), launch 0 reads the prefix's own output, and the suffix reads the last launch's.
    let hidden_of = |body: usize, iter: usize| -> Option<u32> {
        rolled
            .bodies
            .get(body)
            .and_then(|b| rolled.per_layer.get(&b.hidden_out_tid))
            .and_then(|ids| ids.get(iter))
            .copied()
    };
    for (li, &(body, iter)) in rolled.launches.iter().enumerate() {
        let (body, iter) = (body as usize, iter as usize);
        let prev_hidden = if li == 0 {
            None
        } else {
            let &(pb, pi) = &rolled.launches[li - 1];
            hidden_of(pb as usize, pi as usize)
        };
        let hidden_in = rolled.bodies[body].hidden_in_tid;
        for op in &rolled.bodies[body].ops {
            let mut op = op.clone();
            for b in op.ktir.bindings.iter_mut() {
                if b.get() == hidden_in {
                    // Launch 0's hidden-in is the prefix's own output; every later one is the
                    // previous launch's hidden-out.
                    if let Some(prev) = prev_hidden {
                        *b = scratchy_spyre_bundle::program::BufferId::new(prev);
                    }
                    continue;
                }
                // A tensor with no per-layer list is layer-INVARIANT (a shared constant, the
                // mask, a source): the body names the one tensor there is, at every iteration.
                if let Some(ids) = rolled.per_layer.get(&b.get())
                    && let Some(id) = ids.get(iter)
                {
                    *b = scratchy_spyre_bundle::program::BufferId::new(*id);
                }
            }
            ops.push(op);
        }
    }
    let last_hidden = rolled
        .launches
        .last()
        .and_then(|&(b, i)| hidden_of(b as usize, i as usize));
    for op in &rolled.suffix {
        let mut op = op.clone();
        if let Some(last) = last_hidden {
            for b in op.ktir.bindings.iter_mut() {
                if b.get() == rolled.suffix_in_tid {
                    *b = scratchy_spyre_bundle::program::BufferId::new(last);
                }
            }
        }
        ops.push(op);
    }
    ops
}

/// The gate's truncation law: how deep a slice of the roll the numeric gate
/// runs, and the one edge the slice retargets. `layers` = (kept, total) — the
/// PASS line names it. `suffix_rebind` = (the IR tid the full-depth suffix
/// reads as its residual input, the truncated graph's last kept hidden tid) —
/// the golden's IR clone rewires exactly this pair, mirroring the rebind
/// `unroll_layers` applied to the twin's own suffix bindings.
#[derive(Clone, Copy, Debug)]
pub struct GateTrunc {
    pub layers: (u32, u32),
    pub suffix_rebind: (u32, u32),
}

/// ⭐ THE CLASS-COVERING DEPTH — the smallest prefix of the launch sequence
/// that sees every body class AND every class transition the full sequence
/// contains, floored at 2 so a single-class roll still exercises the
/// body→body residual rebind seam, and capped at the sequence itself: a
/// transition the prefix cannot cover (measured: the parity-tiny's six
/// launches s,s,s,s,s,g never see g→s) runs at FULL depth — the identity
/// rebind, and a built-in equivalence test for the truncation machinery.
///
/// The residual seam is why transitions count at all: launch `li`'s
/// hidden-in binds launch `li-1`'s hidden-out (`unroll_layers`), and the
/// suffix binds the last kept launch's — a slice that never crosses a class
/// boundary would leave that rebinding law untested for every shape the
/// model actually runs. MEASURED anchors: gemma-4-12b (s×5,g repeating)
/// covers at 7/48; a single-class granite at 2/40; the parity-tiny at 6/6.
pub fn class_covering_depth(launches: &[(u8, u32)]) -> usize {
    let total = launches.len();
    if total <= 2 {
        return total;
    }
    let mut classes = std::collections::HashSet::new();
    let mut transitions = std::collections::HashSet::new();
    for (i, &(b, _)) in launches.iter().enumerate() {
        classes.insert(b);
        if i + 1 < launches.len() {
            transitions.insert((b, launches[i + 1].0));
        }
    }
    let mut seen_classes = std::collections::HashSet::new();
    let mut seen_transitions = std::collections::HashSet::new();
    for (i, &(b, _)) in launches.iter().enumerate() {
        seen_classes.insert(b);
        if i > 0 {
            seen_transitions.insert((launches[i - 1].0, b));
        }
        if seen_classes.len() == classes.len() && seen_transitions.len() == transitions.len() {
            return (i + 1).max(2).min(total);
        }
    }
    total
}

/// The class-covering twin's launch groups, and where its truncation stops ([`covering_twin_groups`]).
pub type CoveringTwin<T> = (Vec<bundle::LaunchGroup<'static, T>>, GateTrunc);

/// ⭐ THE CLASS-COVERING TWIN — the shipped bundle's per-op groups sliced to
/// the covering depth: the SAME roll, the SAME unroll law, the SAME grouping
/// (`FoldGrouping::Split`, the spelling every bake in `codegen.rs` uses), with
/// only the launch sequence truncated. The suffix rebind falls out of
/// `unroll_layers`' own loop — the twin's last kept launch IS its `last()` —
/// so the slice cannot drift from how the shipped bundle was constructed.
/// `None` when the roll is already covering at full depth: the caller runs
/// the shipped groups with no truncation (the identity rebind).
///
/// The twin is cheap by construction: the programs are interned
/// (`ktir_tokens::ProgramInterner`), so re-unrolling K launches is K copies
/// of the ARGS over one shared program per op — the same cost shape the bake
/// itself pays, at K/total of the depth.
pub fn covering_twin_groups<T: scratchy_spyre_bundle::collective::Tp>(
    rolled: &RolledSuperDsc<T>,
) -> Result<Option<CoveringTwin<T>>, SuperDscError> {
    let total = rolled.launches.len();
    let k = class_covering_depth(&rolled.launches);
    if k >= total {
        return Ok(None);
    }
    // The rebind pair, in the FULL IR's own tid domain — the domain the
    // golden's IR clone rewires. The full-depth suffix reads the LAST
    // launch's hidden-out (`unroll_layers`' own suffix rewrite, a per-layer
    // tid — not the rolled domain's abstract `suffix_in_tid`); the twin's
    // suffix reads the k-1'th launch's. Every launch's hidden-out shares
    // one shape, so the clone swaps the tid and carries the region.
    let hidden_of = |li: usize| -> Option<u32> {
        let &(body, iter) = rolled.launches.get(li)?;
        rolled
            .bodies
            .get(body as usize)
            .and_then(|b| rolled.per_layer.get(&b.hidden_out_tid))
            .and_then(|ids| ids.get(iter as usize))
            .copied()
    };
    let trunc = GateTrunc {
        layers: (k as u32, total as u32),
        suffix_rebind: (
            hidden_of(total - 1).unwrap_or(rolled.suffix_in_tid),
            hidden_of(k - 1).unwrap_or(rolled.suffix_in_tid),
        ),
    };
    let mut twin = rolled.clone();
    // The layout, per-layer map and body tables stay FULL — the kept
    // launches' lookups hit their real entries, and nothing in the twin
    // reads a launch past the cut. Only the sequence truncates.
    twin.launches.truncate(k);
    twin.launch_w.truncate(k);
    twin.launch_kv.truncate(k);
    let ops = unroll_layers(&twin);
    let groups = ktir_groups(&ops)?;
    Ok(Some((groups, trunc)))
}

/// ⭐⭐⭐ THIS BUNDLE'S LAUNCH GROUPS: ITS PROGRAMS, IN LAUNCH ORDER.
///
/// One node is one program and one program is one launch, in the order the tape runs them.
pub fn ktir_groups<T: scratchy_spyre_bundle::collective::Tp>(
    ops: &[EmittedOp<T>],
) -> Result<Vec<bundle::LaunchGroup<'static, T>>, SuperDscError> {
    let mut out: Vec<bundle::LaunchGroup<'static, T>> = Vec::with_capacity(ops.len());
    for e in ops {
        let k = &e.ktir;
        // ⭐ THE BINDING IS A ZIP, NOT A LOOKUP. `KtirNode::bindings` is the buffer each parameter
        // addresses, in parameter order, recorded by the construction that minted the parameter — so
        // pairing it with the func's own argument list is the whole binding. On THIS side of the door
        // a buffer number is a SubtileIR tensor index, which is exactly what `PlaceId::Act` names.
        if k.func.arguments.len() != k.bindings.len() {
            return Err(SuperDscError(format!(
                "{}: {} parameters against {} bound buffers — a launch binds one address per \
                 parameter, so the two must be the same length and in the same order",
                e.op_name,
                k.func.arguments.len(),
                k.bindings.len()
            )));
        }
        let args: Vec<(ktir_core::ir::Ssa, bundle::PlaceId)> = k
            .func
            .arguments
            .iter()
            .zip(k.bindings.iter())
            .map(|((ssa, _), b)| (*ssa, bundle::PlaceId::Act(b.get())))
            .collect();
        let program = bundle::LaunchProgram {
            func: k.func,
            program: k.program,
            args: std::borrow::Cow::Owned(args),
        };
        // ⭐ ONE GROUP PER OP — EXCEPT A COLLECTIVE'S CONSUME, whose ops are ONE group: the CB
        // pipeline puts the send and the wait in front of a consume group, so a consume split
        // over two groups would send twice and wait for a second send that never comes (the
        // descriptor path's `GroupKind::TpConsume` fuses them the same way).
        if let Some(last) = out.last_mut()
            && e.tp_consume.is_some()
            && last.kv.tp == e.tp_consume
        {
            last.programs.to_mut().push(program);
            continue;
        }
        out.push(bundle::LaunchGroup {
            kv: bundle::KvShifts {
                tp: e.tp_consume,
                ..bundle::KvShifts::default()
            },
            programs: std::borrow::Cow::Owned(vec![program]),
            init_binary: bundle::BakedImage::Raw(std::borrow::Cow::Borrowed(&[])),
            job_bin_ptr: 0,
            correction: std::borrow::Cow::Borrowed(&[]),
        });
    }
    Ok(out)
}

/// Where a bake writes each bundle's dxp input, when the build asks it to — the directory the BUILD
/// passes in (the build script's own `OUT_DIR`, the one place that knows it: `compile_carrier`'s
/// caller), never a path this crate reads from the environment. `None`: write nothing (every caller
/// outside a build script, and every build without the `superdsc-dump` feature).
#[derive(Clone, Copy, Debug)]
pub struct DumpRoot<'a>(Option<&'a std::path::Path>);

impl<'a> DumpRoot<'a> {
    /// No dump.
    pub const NONE: DumpRoot<'static> = DumpRoot(None);

    /// Dump under `<dir>/superdsc-dump/<fp>/` — `dir` is the build's own output directory.
    pub fn under(dir: &'a std::path::Path) -> Self {
        DumpRoot(Some(dir))
    }

    fn bundle_dir(self, fp: &str) -> Option<std::path::PathBuf> {
        self.0.map(|d| d.join("superdsc-dump").join(fp))
    }
}

/// RE-ROLLED tape-driven SuperDSC lowering — `SubtileIR → KTIR → SuperDSC`, and nothing but the
/// composition of its three steps:
///
///   1. [`lower_subtile_tape_to_ktir`] walks the rerolled tape and builds ONE KTIR program per node
///      (the producer half, in its own file — nothing there emits a descriptor);
///   2. [`crate::superdsc_lower`] places the rung and lowers each program to its SuperDSC
///      descriptors ([`launch_groups`]);
///   3. the bake — the groups rendered and handed to `superdsc_bake`'s stage/seal/submit
///      ([`emit_bundle`]) — compiles them and `inventory::submit`s the result.
///
/// Steps 2 and 3 run per BUNDLE, from [`emit_bundle`], because the tape walk's three segments
/// (prefix / body / suffix) are baked as three bundles over one layout; this entry is step 1 plus
/// the layout they share.
///
/// [`lower_subtile_tape_to_ktir`]: crate::lower_subtile_tape_to_ktir::lower_subtile_tape_to_ktir
pub fn lower_subtile_tape_to_superdsc<T: scratchy_spyre_bundle::collective::Tp, F: RopeForm>(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    active_cap: ActiveCap,
    rows_are_requests: bool,
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> Result<RolledSuperDsc<T>, SuperDscError> {
    crate::lower_subtile_tape_to_ktir::lower_subtile_tape_to_ktir(
        tape,
        ir,
        weight_ids,
        active_cap,
        rows_are_requests,
        query_rows,
    )
}

/// One launch group's dxp input, rendered: one `sdsc_<i>.json` per descriptor, then `bundle.mlir`
/// last (writing it completes the group, so the directory is compilable at exactly one point).
pub struct GroupInput {
    /// Launch order.
    pub group: u32,
    pub files: Vec<(String, String)>,
    /// Total bytes, so the staging budget's reservation is exact.
    pub bytes: usize,
    /// Content key over exactly what dxp reads, in read order: two groups with one key compile to
    /// one program.
    pub key: crate::dxp_cache::GroupDigest,
    /// Whether dxp runs its chunk-parameter explorer on this group.
    pub explore: crate::superdsc_bake::ChunkExplore,
}

/// One launch group of a bundle, lowered: consecutive programs' descriptors in launch order, and
/// the collective consume the group is, if it is one.
pub struct LoweredGroup<T: scratchy_spyre_bundle::collective::Tp> {
    pub tp: Option<bundle::collective::TpConsume<T>>,
    pub descriptors: crate::superdsc_lower::Descriptors,
}

/// `ops` lowered to SuperDSC against `layout` and cut into launch groups: each collective consume
/// one group of its own (the control-block pipeline sends and waits in front of it), every other
/// run of programs cut into groups of at most [`group_size`] descriptors. Past the bake budget the
/// refusal names the heaviest programs.
pub fn launch_groups<T: scratchy_spyre_bundle::collective::Tp>(
    ops: &[EmittedOp<T>],
    layout: &BundleLayout,
    fp: &str,
) -> Result<Vec<LoweredGroup<T>>, String> {
    let lowered: Vec<(&EmittedOp<T>, crate::superdsc_lower::Descriptors)> = ops
        .iter()
        .map(|e| Ok((e, crate::superdsc_lower::lower_op(e, layout)?)))
        .collect::<Result<_, String>>()?;
    let descriptors: usize = lowered.iter().map(|(_, d)| d.len()).sum();
    crate::superdsc_bake::BakeBudget::bundle_verdict(fp, descriptors, descriptors as u64).map_err(
        |e| {
            let mut heaviest: Vec<(&str, usize)> = lowered
                .iter()
                .map(|(e, d)| (e.op_name.as_str(), d.len()))
                .collect();
            heaviest.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            heaviest.truncate(8);
            format!("{e} The heaviest programs: {heaviest:?}")
        },
    )?;
    let mut groups: Vec<LoweredGroup<T>> = Vec::new();
    for (e, descriptors) in lowered {
        for d in descriptors {
            let joins = groups.last().is_some_and(|g| {
                g.tp == e.tp_consume && (g.tp.is_some() || g.descriptors.len() < group_size())
            });
            if !joins {
                groups.push(LoweredGroup {
                    tp: e.tp_consume,
                    descriptors: Vec::new(),
                });
            }
            if let Some(g) = groups.last_mut() {
                g.descriptors.push(d);
            }
        }
    }
    Ok(groups)
}

/// Render one launch group's dxp input.
fn render<T: scratchy_spyre_bundle::collective::Tp>(
    group: u32,
    g: &LoweredGroup<T>,
) -> std::io::Result<GroupInput> {
    let ops = &g.descriptors;
    let files = spyre_superdsc::emit::group_files(ops)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let explore = crate::superdsc_bake::ChunkExplore::of_group(ops.iter().map(|(_, s)| s));
    let key = crate::dxp_cache::GroupDigest::of_compile(&files, explore.digest_tag());
    Ok(GroupInput {
        group,
        bytes: files.iter().map(|(_, c)| c.len()).sum(),
        key,
        explore,
        files,
    })
}

/// Write rendered groups into `dir/group_<g>/`, for handing a bundle to `dxp_standalone` by hand.
pub fn write_rendered_dxp_input(
    dir: &std::path::Path,
    rendered: &[GroupInput],
) -> std::io::Result<()> {
    for g in rendered {
        let gdir = dir.join(format!("group_{}", g.group));
        std::fs::create_dir_all(&gdir)?;
        for (name, contents) in &g.files {
            std::fs::write(gdir.join(name), contents)?;
        }
    }
    Ok(())
}

/// The memory plan, projected into the baked artifact.
fn bake_layout(l: &BundleLayout) -> bundle::BundleLayout<'static> {
    let places: Vec<bundle::Placement> = l
        .placements
        .iter()
        .map(|(tid, p)| bundle::Placement {
            id: bundle::PlaceId::Act(*tid),
            segment: p.segment as u32,
            bank: p.bank,
            offset: p.offset,
            size: p.size,
            is_logits: p.role == SegRole::Logits,
        })
        .collect();
    audit_layout_addresses(
        &places,
        &l.segment_bytes,
        &l.weight_bank_bytes,
        &l.kv_bank_bytes,
        "<layout>",
    );
    assert_eq!(
        l.weight_tail,
        weight_tail_of_placements(&l.placements),
        "the segment budget's weight-tail verdict disagrees with the weight placements"
    );
    let kernel_weights = l
        .kernel_weights
        .iter()
        .map(|(tid, k)| {
            let walk = crate::stage_weight::RetileWalk { axes: &k.axes };
            bundle::KernelWeight {
                id: bundle::PlaceId::Act(*tid),
                axes: std::borrow::Cow::Owned(k.axes.clone()),
                arm: walk.arm(),
                stick_size: k.stick_size,
                word_length: k.word_length,
            }
        })
        .collect();
    let stick_consts = l
        .stick_consts
        .iter()
        .map(|(tid, c)| bundle::StickConst {
            id: bundle::PlaceId::Act(*tid),
            format: match c.format {
                spyre_superdsc::format::DataFormat::Fp16 => bundle::StickFormat::Fp16,
                spyre_superdsc::format::DataFormat::Fp32 => bundle::StickFormat::Fp32,
                spyre_superdsc::format::DataFormat::Fp8 => bundle::StickFormat::Fp8,
                spyre_superdsc::format::DataFormat::UInt32
                | spyre_superdsc::format::DataFormat::Int64 => bundle::StickFormat::UInt32,
            },
            value: c.bits,
        })
        .collect();
    bundle::BundleLayout {
        segment_bytes: l.segment_bytes,
        weight_bank_bytes: std::borrow::Cow::Owned(l.weight_bank_bytes.clone()),
        kv_bank_bytes: std::borrow::Cow::Owned(l.kv_bank_bytes.clone()),
        weight_tail: l.weight_tail,
        places: std::borrow::Cow::Owned(places),
        kernel_weights: std::borrow::Cow::Owned(kernel_weights),
        stick_consts: std::borrow::Cow::Owned(stick_consts),
        weight_scale_folds: std::borrow::Cow::Owned(l.weight_scale_folds.clone()),
        kv_request_stride_bytes: l.kv_request_stride_bytes,
        num_sources: l.num_sources,
        resident_source_tids: std::borrow::Cow::Owned(l.resident_source_tids.clone()),
    }
}

/// ⭐ EMIT ONE BUNDLE: audit its programs, then lower, partition and stage them.
///
/// Returns the bundle's fingerprint — its identity everywhere downstream: a sibling names it,
/// [`attach_reroll`] addresses it, and [`drain_emitted_bundles`] hands it to the macro. It is
/// content-addressed, so emitting the same programs over the same layout twice does nothing the
/// second time.
pub fn emit_bundle<T: scratchy_spyre_bundle::collective::Tp>(
    ops: &[EmittedOp<T>],
    layout: Option<&BundleLayout>,
    dump: DumpRoot<'_>,
) -> std::io::Result<String> {
    audit_op_work(ops, "<bundle>");
    let fp = bundle_fp(ops, layout);
    // One emit per fingerprint: two rungs can be one program, and both would stage the same group
    // directory, the second's cleanup deleting it under the first's compile.
    let gate = emitting(&fp);
    let _one_emit = gate
        .lock()
        .map_err(|_| std::io::Error::other("a bundle emit panicked holding its fingerprint"))?;
    if emitted::<T>().lock().is_ok_and(|c| c.contains_key(&fp)) {
        return Ok(fp);
    }
    let b = emit_bundle_inner(ops, layout, &fp, dump)?;
    if let Ok(mut c) = emitted::<T>().lock() {
        c.entry(fp.clone()).or_insert((b, None));
    }
    Ok(fp)
}

/// This bundle's fingerprint: every program (its name, its KTIR, its bindings and whether it is a
/// collective's consume) and the serialized layout the descriptors are lowered against — the
/// descriptors are a function of the two, so one fingerprint names one (program set, layout).
fn bundle_fp<T: scratchy_spyre_bundle::collective::Tp>(
    ops: &[EmittedOp<T>],
    layout: Option<&BundleLayout>,
) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    group_size().hash(&mut h);
    for e in ops {
        e.op_name.hash(&mut h);
        e.ktir.func.hash(&mut h);
        e.ktir.bindings.hash(&mut h);
        e.tp_consume.is_some().hash(&mut h);
    }
    let base = format!("{:016x}", h.finish());
    match layout {
        Some(l) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            serde_json::to_vec(l)
                .expect("a bundle layout serializes")
                .hash(&mut h);
            format!("{base}l{:016x}", h.finish())
        }
        None => base,
    }
}

fn emit_bundle_inner<T: scratchy_spyre_bundle::collective::Tp>(
    ops: &[EmittedOp<T>],
    layout: Option<&BundleLayout>,
    fp: &str,
    dump: DumpRoot<'_>,
) -> std::io::Result<EmittedBundle<T>> {
    let groups = match layout {
        Some(l) if cfg!(feature = "spyre-lower") => {
            PendingGroups::Staged(staged_groups(ops, l, fp, dump).map_err(std::io::Error::other)?)
        }
        _ => PendingGroups::Ready(ktir_groups(ops).map_err(|e| std::io::Error::other(e.0))?),
    };
    Ok(EmittedBundle {
        fp: fp.to_string(),
        layout: layout.map(bake_layout).unwrap_or_default(),
        groups,
    })
}

/// The card's launch groups: every program lowered to SuperDSC against `layout`, the descriptors
/// partitioned into launch groups, and each group rendered and handed to the bake queue — which a
/// `spyre-lower` build without `spyre-hw` does not have, so there every group is lowered and
/// rendered and none is compiled.
///
/// Returns the identity each group's compiled program will be found under, once
/// `superdsc_bake::finish_global()` has drained the queue ([`EmittedBundle::into_code`]).
fn staged_groups<T: scratchy_spyre_bundle::collective::Tp>(
    ops: &[EmittedOp<T>],
    layout: &BundleLayout,
    fp: &str,
    dump: DumpRoot<'_>,
) -> Result<Vec<(bundle::KvShifts<T>, crate::superdsc_bake::GroupId)>, String> {
    let groups = launch_groups(ops, layout, fp)?;
    let descriptors: usize = groups.iter().map(|g| g.descriptors.len()).sum();
    let rendered: Vec<GroupInput> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| render(i as u32, g))
        .collect::<std::io::Result<_>>()
        .map_err(|e| e.to_string())?;
    for g in &rendered {
        crate::superdsc_bake::BakeBudget::group_verdict(fp, g.group, g.files.len() - 1)?;
    }
    if let Some(dir) = dump.bundle_dir(fp) {
        write_rendered_dxp_input(&dir, &rendered).map_err(|e| e.to_string())?;
    }
    eprintln!(
        "[superdsc] BUNDLE {fp}: {} program(s) -> {descriptors} descriptor(s) in {} group(s)",
        ops.len(),
        rendered.len()
    );
    let mut out = Vec::with_capacity(rendered.len());
    let Some(bake) = crate::superdsc_bake::global() else {
        return Ok(out);
    };
    let ns = crate::superdsc_bake::EmitNs::mint();
    for (g, r) in groups.iter().zip(&rendered) {
        let id = crate::superdsc_bake::GroupId {
            fp: fp.to_string(),
            ns,
            group: r.group,
        };
        let kv = bundle::KvShifts {
            tp: g.tp,
            ..bundle::KvShifts::default()
        };
        if !bake.adopt(&id, r.key) {
            let gdir = bake.stage().group_dir(ns, fp, r.group as usize);
            bake.reserve(r.bytes);
            std::fs::create_dir_all(&gdir).map_err(|e| e.to_string())?;
            for (name, contents) in &r.files {
                std::fs::write(gdir.join(name), contents).map_err(|e| e.to_string())?;
            }
            bake.submit(crate::superdsc_bake::SealedGroup::of_rendered(
                gdir,
                id.clone(),
                r,
            ))?;
        }
        out.push((kv, id));
    }
    Ok(out)
}
