// SPDX-License-Identifier: Apache-2.0
//! Lower a [`SubtileIR`] to an **IBM Spyre SuperDSC (SDSC) bundle** — the
//! TILE-LEVEL, work-divided IR that DeepTools' backend (`dxp_standalone
//! --bundle`) compiles into a device program. This is the PERFORMANCE path:
//! unlike the high-level sengraph emitter ([`lower_subtile_tape_to_sengraph`]),
//! which hands DeepTools an op-graph and lets `CompileGraph` derive the
//! per-core split automatically (on-card PROVEN to leave the 32 cores idle —
//! `SENCORES 1→14.4ms vs 32→9.9ms`, only 1.45×), the SuperDSC emitter OWNS the
//! 32-core work-division itself (exactly what torch-spyre's Inductor front-end
//! does). SubtileIR is already tile-level, so this is a tile→tile lowering.
//!
//! GROUND TRUTH: the DeepTools scheduler test fixtures
//! `/project_src/deeptools/dcg/dcg_fe/scheduler/test/sdsc_{add,bmm_autoBuffer,
//! …}.json` (on the pod) are real, valid SuperDSC; this module mirrors them
//! field-for-field. Ingest PROVEN on-card 2026-06-25: `L3DlOpsScheduler_standalone
//! -s <sdsc>.json` → "Success!" EXIT 0. See memory `superdsc-emitter-design.md`.
//!
//! NAMED ITERATION DIMS (NOT generic c0/c1): the SuperDSC iteration space uses
//! fixed slots `in_/out_/mb_/i_/j_/ki_/kj_/x_/x1_/y_/r_/c_/ij_/…` (-1 = unused).
//! For a matmul A[mb,in] · W[in,out] → O[mb,out]: `in_`=K (reduction), `out_`=N,
//! `mb_`=M, `x_`=tiling factor. Pointwise add over [mb,out] uses `i_`/`j_`/`ij_`.
//!
//! HARDWARE: 32 cores, 2 corelets/core; STICK = 128 B = 64 fp16 elems; per-core
//! span limit 256 MiB. fp16 everywhere → dataFormat_ "SEN169_FP16", wordLength 2,
//! stickSize_ 64.

// The SuperDSC struct fields mirror the DeepTools fixture JSON keys VERBATIM
// (trailing-underscore / camelCase, e.g. `coreFoldProp_`, `numWkSlicesPerDim_`),
// so the serde field name IS the JSON key. snake_case renaming would break the
// wire format — silence the lint for the whole module.
#![allow(non_snake_case)]

pub use crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off;
// Re-exports: external crates reference these matmul-family functions directly through this
// module's path (`scratchy-forward-compiler-macro`'s codegen: `superdsc::assemble_matmul`;
// `scratchy-sdsc`'s emit.rs: `mono::assemble_matmul_split`) — keep those paths resolving after
// the matmul family's real home moved to `ir::bridge::tiled_op_sdsc_op::matmul`.
//
// `assemble_matmul_seeded` / `assemble_matmul_off` moved from the private `use` above to this
// `pub use` (2026-07-28): tests/element_torchspyre_parity.rs calls both through this module path,
// so a private import made that ENTIRE test target fail to compile with E0603 — it has not built
// since 57dfac42 (07-26), silently killing 6 tests. One of them,
// `per_core_start_derives_from_spyre_layout`, is the ONLY thing that pins mb-split per-core start
// bytes against torch-spyre, i.e. exactly the machinery under suspicion for the composite-m
// garble. Nothing caught it because CI never builds this crate with `--features superdsc`.
pub use crate::ir::bridge::tiled_op_sdsc_op::{
    assemble_attn, assemble_matmul, assemble_matmul_off, assemble_matmul_seeded,
    assemble_matmul_split,
};
use ktir_core::attrkey::AttrKey;
use ktir_core::ir::Attr;
use ktir_core::opkind::OpKind;
/// ⭐ THE BAKED BUNDLE'S TYPES. This lowering's OUTPUT is a `bundle::BundleCode` — the value the
/// `#[forward]` macro puts in the binary and the runtime reads back.
pub use scratchy_spyre_bundle as bundle;
// The PRODUCER half's surface, re-exported so external callers (`scratchy-forward-compiler-macro`'s
// codegen: `superdsc::graph_wiring`, `superdsc::unroll_layers`) keep resolving through this module's
// path after the SubtileIR → KTIR construction moved to its own file.
pub use crate::lower_subtile_tape_to_ktir::{
    BundleWiring, NodeArgs, graph_wiring, lower_graph_to_ktir, lower_graph_to_superdsc,
    lower_subtile_tape_to_ktir,
};
// ⭐⭐⭐ THE REQUEST TYPE LIVES IN `ktir_superdsc::ktir_node` — `KtirNode` and the decode ladder rung
// `ActiveCap`. Not a device fact and not the caller's own plan: what a KTIR producer hands in.
//
// ⚠️ `ActiveCap` IS RE-EXPORTED, AND IT IS THE ONE NAME IN THIS MOVE THAT HAS TO BE. Every other
// caller was re-pointed at the crate; this one cannot be, because its call sites are TOKENS the
// `#[forward]` macro emits (`codegen.rs`, 9 sites) which expand inside `scratchy-models` and the other
// crates that run the spyre bake — and none of them depends on `ktir-superdsc`, only on this crate. So
// the generated path must resolve through here. `KtirNode` is NOT re-exported: its only callers are in
// this crate and they name the leaf directly.
pub use ktir_superdsc::ktir_node::ActiveCap;
use scratchy_subtile::subtile_ir::{RopeForm, SubOp, SubtileIR};
use scratchy_subtile::superdsc_opspec::DataFormat;
use std::collections::BTreeMap;

// Re-export the hardware constants from the typed core so the whole emitter
// shares ONE source of truth (the typed witnesses live in `superdsc_opspec`).
// (`DataFormat` itself is brought into scope by the main `use` block above.)
pub use scratchy_subtile::superdsc_opspec::{ACTIVE_CORELETS, MAX_CORES};
/// fp16 data format alias (the default). `Fp32` is used only by the fp32-SFP-merge split-K path;
/// `SenInt8` only by the packed-int8 (SENINT8) quant weight path.
pub use scratchy_subtile::superdsc_opspec::{Df, Fp8, Fp16, Fp32, SenInt8};

// Per-tensor dtype (`wordLength` / `stickSize_` / `dataFormat_`) is now the arg's typed [`Df`]
// (`ArgView::df`), NOT a `_fp8`/`_fp32`/`_senint8` name substring. The old `word_length_for` /
// `stick_elems_for` / `dataformat_for` string-sniffers are DELETED — see `Df` in superdsc_opspec.

// ⭐ THE WORK DIVISION LIVES IN `ktir_superdsc::work` — the card constants, `core_split`,
// `CoreSplit`, `DeviceWidth`, `distribute_cores`, `core_to_wk_slice` and the matmul cost search.
// Re-exported so every in-file call site and every
// `crate::lower_subtile_tape_to_superdsc::{core_split, …}` path in the tree resolves unchanged.
pub use ktir_superdsc::work::{
    CORELETS_PER_CORE, CoreSplit, DeviceWidth, FP16_ELEMS_PER_STICK, MAX_SPAN_BYTES, MatmulSplit,
    STICK_BYTES, bump_sticks_to_splittable, core_split, core_to_wk_slice, distribute_cores,
    matmul_cost_split, matmul_split_plan, stick_count,
};

// ── EMIT-TIME CONSTANTS THAT USED TO BE ENVIRONMENT READS ───────────────────
//
// ⛔⛔⛔ THE EMITTER MUST NOT READ THE SHELL. `#[forward]` compiles
// `(config.json, math DSL)` into an artifact. An `env::var` in here makes it
// `(config.json, math DSL, whoever's shell ran cargo)` — two builds of identical
// source emit different bundles, and nothing downstream can be called a constant.
//
// ⛔ THE COST IS ON THE RECORD, in `spyre_load.rs`'s own comment: a bundle baked
// WITHOUT a K-split flag and served by a worker that read the flag staged blocks
// the bundle had no room for — every completion came back EMPTY, RC=0, no error,
// and it "cost four runs and one wrong conclusion". The WORKER was then fixed to
// ask the bundle instead of the environment. The EMITTER was not, so the same
// two-processes-agreeing-by-convention hazard survived one level up.
//
// ⭐ EACH VALUE BELOW IS WHAT THE UNSET VARIABLE PRODUCED. None of these is set
// by the canonical pod env or its build wrapper, so the unset case IS the
// production case and this is a pure constant-ification: the bundle fingerprints
// do not move. Changing one is now a source edit that recompiles, which is what
// makes it a constant rather than a configuration.
//
// 🛑 `SCRATCHY_SUPERDSC_GROUP_SIZE` IS DELIBERATELY NOT HERE. It is the one knob
// the canonical env DOES set (2048), and the two authorities disagree: this file
// says "2048 was too aggressive … 512 is the real ceiling, not 2048", while the
// canonical pod env says "G=2048 = whole-body = ~14.6 tok/s … pod DEBUG scripts use
// 512; DON'T — it tanks throughput". Picking either silently is a crash or a 2.2×
// throughput regression, so it stays a read until someone settles it on a card.

// ⭐⭐⭐ THE WIRE LAYER LIVES IN `ktir_superdsc::wire` — `Dsc`, `SdscOp` and the whole
// `#[derive(Serialize)]` family, the three fp8 nested-fold generators, and the HBM segment
// constants. The extraction plan claimed these could only move WITH `emit_sdsc`; they could not,
// because every one is pure data with `pub` fields and the generators take extents plus a
// `StickLayout`. So the crate owns the SuperDSC TYPE while `emit_sdsc` is still on this side
// constructing it. Re-exported so every in-file and cross-crate path resolves unchanged.
pub use ktir_superdsc::wire::{
    AddrFold, AllocNode, ComputeAttrs, ComputeOp, Dsc, FoldFunc, FoldProp, IterSpace, LabeledDs,
    LayoutInfo, MemOrg, MemPresence, SEGMENT_OFFSETS, SEGMENT_SIZE, SdscFolds, SdscOp, StageParam,
    build_coordinates, core_dsc_schedule, gen_coord_info_value, hbm_seg_off, matmul_iter_space,
    segment_base,
};
// `ex_unit` was in the list above and is GONE from the crate, not moved: a stringly `&str -> &str`
// helper with no caller anywhere, a silent `_ => "sfp"` default, and an answer that CONTRADICTED the
// typed `OpFunc::ex_unit` on `Transpose`. The re-export is why a grep for callers found none while the
// name still resolved — it was reachable, just never called. `OpFunc::ex_unit` is the sole producer.

// ⛔⛔⛔ AND IT IS THE SAME 16 GiB THE ALLOCATOR CAPS A REGION AT — for two INDEPENDENT reasons, which
// is why an over-size segment is worse than a failed allocation.
//
//   * flex serves a segment's allocation from ONE region, capped at `MAX_REGION_BYTES` ⇒ it fails.
//   * this ADDRESSING scheme puts segment `s` at `s · SEGMENT_SIZE` and `hbm_seg_off` recovers
//     `(s, intra)` by dividing — which is exact ONLY while `intra < SEGMENT_SIZE`. A placement past
//     that decomposes into the NEXT segment: granite-3.1-8b-fp16's tied embedding starts at
//     16,945,651,712 in seg1, so `SEGMENT_OFFSETS[1] + off` lands beyond `SEGMENT_OFFSETS[2]` and the
//     lm_head weight would have been addressed inside the KV segment. Not a fault — wrong logits.
//
// So the `audit_layout_addresses` ceiling check is an ADDRESSING guard as much as an allocation one,
// and this assert is what keeps the number it uses tied to this scheme.
const _: () = assert!(bundle::MAX_SEGMENT_BYTES == SEGMENT_SIZE);

// ───────────────────────────────────────────────────────────────────────────
// GLOBAL bundle memory layout (task #55) — the fix for the multi-op address bug.
//
// The per-op `segment_base(arg_index)` above places each op's args in segments by
// their PER-OP-LOCAL position (arg 0→seg0, arg 1→seg1, …). That is correct for a
// SINGLE op but WRONG for the 697-op fused bundle: every op reuses seg0..6, so
// op-5's weight and op-300's activation both bake to seg1 and ALIAS. The fix is a
// GLOBAL layout the emitter owns: every distinct tensor (by SubtileIR id) gets ONE
// (segment, byte-offset) for the whole bundle, keyed by ROLE not arg position. A
// single fused program then addresses (segment, offset); the executor binds ≤7
// segment regions (one per role) — defeating the 7-tensor-segment HW cap by packing
// many tensors into one region at interior offsets (flex `defines.hpp` segment_id =
// top 3 bits of the operand DMVA; one region = one segment_id).
// ───────────────────────────────────────────────────────────────────────────

// ⭐⭐⭐ THE RESERVED TID SPACE LIVES IN `ktir_superdsc::reserved_tids` — every `*_TID` sentinel, the
// `TidRegion`/`RESERVED_REGIONS` table with both of its compile-time disjointness proofs,
// `scalarmul_scale_tid` and `kct_resident_tid`.
//
// ⭐ IT IS THE PRODUCER↔LOWERING ABI, not scratchy's plan: each id names a tensor the LOWERING
// invents or requires and the caller binds, so a third-party KTIR producer has to agree with these
// numbers to bind anything at all. Re-exported so every path in the tree resolves unchanged.
pub use ktir_superdsc::reserved_tids::{
    ATTN_CAUSAL_TID, ATTN_MASK_TID, ATTN_SCALE_TID, ATTN_ZERO_TID, FP8_INV448_TID, FP8_NEG448_TID,
    FP8_POS448_TID, IDENTITY_TID, KCT_RESIDENT_BASE, LAST_HIDDEN_TID, NEW_V_PROBE_TID,
    ONES_REDUCE_TID, RESERVED_REGIONS, RESERVED_REGIONS_ARE_DISJOINT, RMS_HALF_TID,
    RMS_INVCOLS_TID, RMS_RSQRT_PROBE_TID, RMS_SEED_TID, RMS_VAR_PROBE_TID, ROPE_P_TID,
    SCALARMUL_SCALE_BASE, SEL_HEADMAJOR_TID, SEL_KV_HEADMAJOR_TID, SELT_HEADMAJOR_TID,
    SENTINELS_ARE_INSIDE_THEIR_REGION, TidRegion, kct_resident_tid, scalarmul_scale_tid,
};

// ⭐⭐⭐ THE MEMORY PLAN LIVES IN `ktir_superdsc::placement` — `SegRole`, `TensorPlacement`,
// `RetileDescriptor`, `BundleLayout` and its whole impl, `SynthAlloc`, `align128`, `syn`.
//
// ⛔ IT IS A STRUCT, NOT A TRAIT, AND THE EXTRACTION PLAN WAS WRONG THAT IT SHOULD BE ONE. A
// `BundleLayout` is a DEVICE memory plan; the only scratchy-shaped names the whole region touched
// were `bundle::PlaceId` and `sdsc_abstract::StickLayout`, and both were already in that crate. What
// is scratchy's is the CONSTRUCTION — `compute_bundle_layout` (SubtileIR liveness),
// `audit_layout_addresses` (audits `bundle::Placement`) and `bake_layout` — all three of which
// stayed right here. That is the seam: a third-party producer builds the same struct from its own
// facts.
pub use ktir_superdsc::placement::{
    BundleLayout, MAX_FOLD_PASSES, MAX_PAGES_PER_REQUEST, RetileDescriptor, SegRole, SynthAlloc,
    TensorPlacement, WEIGHT_SPILL_SEGS, align128, syn, synth_footprint_bytes,
};
// ⭐⭐⭐ AND THE EMITTER ITSELF NOW LIVES IN `ktir_superdsc::emit` — `emit_sdsc`/`emit_sdsc_tiled`, the
// whole `assemble_*` family, `pw1`/`pw2`, `ArgBinding`/`EmittedOp`, `bmm_site` and the SFP constant
// table. What is left in THIS file is the half that reads a `SubtileIR`: `compute_bundle_layout`
// (liveness → a `BundleLayout`), `audit_layout_addresses`, `bake_layout`, the tape walk, the group
// classifier and the bake queue.
//
// ⚠️ A PRIVATE `use`, NOT A `pub use`. Every other caller in the tree was RE-POINTED at
// `ktir_superdsc::emit::…` rather than kept resolving through here — a re-export would have hidden
// which of them still believe the emitter is in this crate. (`ActiveCap` above is the one exception,
// and only because its call sites are tokens the `#[forward]` macro emits.)
use ktir_superdsc::emit::EmittedOp;
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
///  * **No overlap.** Two placements sharing bytes means one tensor's producer overwrites the
///    other's, which is silent: wrong output, or a fault when the second is a kernel.
///  * **Non-empty.** A zero-length field is read by the device as `1 << 27` flits = 16 GiB, in both
///    `handleHostDMA` and `handleXLATentry`. Zero does not mean nothing.
///  * **Allocatable.** A segment is ONE `FlexAllocator` allocation, served from ONE region, so it
///    cannot exceed [`bundle::MAX_SEGMENT_BYTES`] — see that constant for the granite-3.1-8b-fp16
///    weight segment this catches.
fn audit_layout_addresses(
    places: &[bundle::Placement],
    segment_bytes: &[u64; 7],
    // Extents of weight banks 1..N (empty ⇒ one bank). A banked placement is bounded by ITS OWN
    // bank, not by the segment total — see [`bundle::Placement::bank`].
    weight_bank_bytes: &[u64],
    fp: &str,
) {
    let mut fail: Vec<String> = Vec::new();
    let w_seg = SegRole::Weight.segment();
    // The extent a placement is checked against: its bank's, which for bank 0 (everything in an
    // unbanked bundle) is the segment's own total.
    let extent = |seg: usize, bank: u32| -> Option<u64> {
        match (seg == w_seg, bank) {
            (_, 0) => segment_bytes.get(seg).copied(),
            (true, b) => weight_bank_bytes.get(b as usize - 1).copied(),
            (false, _) => None,
        }
    };

    // ── ALLOCATABLE: each segment is one device region, and a region is capped ──
    //
    // ⛔ THIS WAS THE CARD'S JOB AND THE CARD IS BAD AT IT. `prepare` allocates a segment with
    // `DevAddr::alloc(segment_bytes[i])`, and flex serves that from a single region: over the cap it
    // returns an OOM whose numbers deny each other (`requested_bytes` well under `free_space_bytes`),
    // because the free bytes are in the regions the request cannot reach. Nothing in that message
    // names a segment, a tensor, or a model. All of it is decided here, by constants.
    // A weight BANK is a device region like any other, so it is bound by the same cap; checking the
    // banks here is what stops banking from trading one over-size allocation for another.
    for (b, &bytes) in weight_bank_bytes.iter().enumerate() {
        if bytes > bundle::MAX_SEGMENT_BYTES {
            fail.push(format!(
                "weight bank {} packs {bytes} B, {} B past the {} B a single device region can hold",
                b + 1,
                bytes - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
            ));
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
                "{} is in seg{seg} bank {}, which this bundle does not have ({} weight bank(s))",
                p.id,
                p.bank,
                1 + weight_bank_bytes.len(),
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

    // ── overlap, per segment — SYNTHETICS ONLY ──
    //
    // ⛔ COLORED ACTIVATIONS OVERLAP ON PURPOSE. The slot-coloring pass gives two tensors whose
    // LIFETIMES are disjoint the same address; that is the optimisation, not a defect, and
    // granite's `t447`/`t464`/`t1127` legitimately share one 4096 B slot. Liveness is not a fact
    // this function has, so it does not get to judge them.
    //
    // Synthetics are different: they come from a BUMP allocator, so disjointness IS their
    // invariant and an overlap means the allocator handed the same address out twice — which is
    // exactly what it did, by never advancing `next` for an undeclared name.
    for seg in 0..segment_bytes.len() {
        let mut in_seg: Vec<&bundle::Placement> = places
            .iter()
            .filter(|p| {
                p.segment as usize == seg
                    && p.size > 0
                    && matches!(p.id, bundle::PlaceId::Synth { .. })
            })
            .collect();
        in_seg.sort_by_key(|p| p.offset);
        for w in in_seg.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a.offset + a.size > b.offset {
                fail.push(format!(
                    "{} spans [{}, {}) and {} starts at {} in seg{seg} — they SHARE {} B, so one \
                     tensor's producer writes over the other's",
                    a.id,
                    a.offset,
                    a.offset + a.size,
                    b.id,
                    b.offset,
                    a.offset + a.size - b.offset
                ));
            }
        }
    }

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
pub fn compute_bundle_layout<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // TRUE when this bundle's query rows are separate requests. The prefix mask's reservation
    // depends on it (a prompt reads one broadcast validity row, a batch reads one per query row),
    // and so does the attention pad's door: a decode width must be a baked ladder rung
    // (`PaddedMq::of_bundle`), which is the `Err` this returns.
    rows_are_requests: bool,
    // ⭐ THE LAYER STRUCTURE, so the weight segment can be split into BANKS at a LAYER boundary —
    // see [`bank_weight_segment`]. From [`per_layer_external_tids`], a pre-pass over the re-rolled
    // tape; EMPTY for an unrolled bundle, which has no layer boundary and therefore cannot bank.
    per_layer_ext: &std::collections::BTreeMap<u32, Vec<u32>>,
) -> Result<BundleLayout, SuperDscError> {
    // fp8 W8A8 weights (SEN143_FP8: 1-byte / 128-elem stick) are `input[1]` of any arity-3 MatmulTile.
    // Their device footprint is HALF the fp16 weight — this is the unfakeable 1-byte-read proxy: it is
    // what shrinks `seg_bytes[1]` (the weight segment). A dequant-to-f16 transient would NOT shrink it.
    let fp8_weight_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
        .map(|n| n.inputs[1].tensor.index() as u32)
        .collect();
    let nbytes = |tid: u32| -> u64 {
        let s = ir.tensors[tid as usize];
        // Reserve the DEVICE footprint: a stick-last tensor pads its innermost stick dim up to a whole
        // stick on-device — fp16 is a 64-elem / 2-byte stick, fp8 is a 128-elem / 1-byte stick (HALF the
        // bytes). Width is DERIVED from the operand's `Df` (rung 1), never hardcoded. (granite lm_head
        // vocab 49159 → 49216 padding stays in-bounds for the on-card write / weight staging.)
        let df = if fp8_weight_tids.contains(&tid) {
            Df::Fp8
        } else {
            Df::Fp16
        };
        s.rows as u64
            * bump_sticks_to_splittable(s.cols.next_multiple_of(df.elems_per_stick())) as u64
            * df.word_length() as u64
    };
    // ── Liveness: first def (output of node i) and last use (input of node j). A
    //    source has no def (def = 0, live from the start); the result has no use
    //    (use = nodes.len(), live to the end). ──
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

    // The AttnDecode K/V cache tensors are SOURCES, but they get RE-PLACED into seg2 (SegRole::Kv) at
    // the batched cache size further below (search "Prefix K/V caches → seg2"). If the source loop ALSO
    // packs them into the Activation segment at their source size, that Activation slot is never used
    // (the seg2 placement overrides it) — a DEAD HOLE (80 tensors × [cap,kv_dim]·2 = 335 MB @cap=4096
    // for granite) that the per-step whole-Activation-segment H2D re-ships every decode step. That dead
    // re-upload IS the O(cap) decode preamble cliff (measured: 50 ms @cap=4096 vs 8 ms @cap=256). Skip
    // them here; their only live placement is seg2 (what the worker binds + attention reads).
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
    let per_layer_tids: std::collections::BTreeSet<u32> =
        per_layer_ext.values().flatten().copied().collect();
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
    let weight_bank_bytes = if per_layer_block_end <= bundle::MAX_SEGMENT_BYTES {
        spill_weight_tail(&mut placements, &mut seg_bytes, &per_layer_tids)?;
        Vec::new()
    } else {
        bank_weight_segment(&mut placements, &mut seg_bytes, per_layer_ext)?
    };
    // ── ROPE permutation matrix P [hd,hd] (task: in-bundle RoPE) ── If the tape has
    // any RopeRotate/RopeAppend, `lower_rope_node` emits `rot = matmul(x, P)` (the
    // rotate-half as a 64-stick-aligned matmul, avoiding the 32-half sub-stick). P is
    // a FIXED permutation-sign matrix the worker synthesizes + binds (it is NOT a
    // model/safetensors weight, so it gets the reserved id ROPE_P_TID). Place it in
    // seg0 (ACTIVATION, re-bound per step like RMS_SEED/ATTN_SCALE) — NOT seg1 (WEIGHT):
    // a seg1 weight is only H2D'd at PrepareModel, where the worker binds ONLY the
    // manifest's model weights, so a synthetic seg1 P stays ZERO ⇒ rot=matmul(x,0)=0 ⇒
    // RoPE collapses to `x·cos` (rotate-half/sin term DROPPED) ⇒ wrong positional
    // encoding ⇒ wrong content. Same P for every head/position/layer.
    if let Some(hd) = ir.nodes.iter().find_map(|n| match &n.op {
        SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
            Some(head_dim.get() as u64)
        }
        _ => None,
    }) {
        let seg = SegRole::Activation.segment();
        let off = seg_bytes[seg];
        let sz = hd * hd * 2; // [hd,hd] fp16
        placements.insert(
            ROPE_P_TID,
            TensorPlacement {
                tid: ROPE_P_TID,
                bank: 0,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);
    }

    // ── DEAD seg0 CONSTS REMOVED (2026-07-28, TTFT) ── This block used to place, for every mq>1
    // (batched-prefill) bundle: the head-major one-hot selectors Sel_q [nqh·nqh·hd,hd],
    // Sel_kv [nkvh·nkvh·hd,hd], SelT [nqh·hd,nqh·hd], and the matmul-by-ones row-sum weight
    // ONES_REDUCE [max(hidden, max fp8-K), stick]. NOTHING reads any of them any more: the unified
    // emitter (ir::bridge::tiled_op_sdsc_op::attn) is head-contiguous and needs no selector matmul,
    // and rmsnorm/fp8-amax both use NATIVE reduces (assemble_reduce_seeded "mean"/"max"), not the
    // matmul-by-ones substitute. Verified: no op anywhere names these tids; only the const decls,
    // this placement, and doc comments referenced them.
    // They were not free. The shim re-uploads the whole dirty seg0 every forward, so each prefill
    // chunk was paying ~18.3 MB of H2D for never-written, never-read bytes:
    //   Sel_q 8.4 MB + SelT 8.4 MB + Sel_kv 0.5 MB + ONES_REDUCE 1.0 MB.
    // With the op count already cut 5x (RoPE row-batching), TTFT was dominated by a ~218 ms FIXED
    // cost that op-count work cannot touch; this dead H2D is the bulk of it.
    // The ONES_REDUCE placement also drove the worker's `uses_ones_reduce` binding (read back out of
    // bundle_layout.json), so unplacing it also stops the worker from staging the all-ones buffer.
    // Its qk-norm build guard (cols==hidden assert) went with it: that guard existed only to protect
    // the matmul-by-ones reduce's reuse of RMS_INVCOLS=1/hidden as a 1/cols scale, and the native
    // reduce it was replaced by folds 1/N itself, so the constraint no longer applies.
    // ── KV-cache-write / GQA-replicate MATMUL-BY-IDENTITY weight (#3, SCRATCHY_SUPERDSC_KV_MATMUL) ──
    // A [hd,hd] identity fp16 seg0 ACTIVATION, worker-bound flat: for hd==64 the retile is
    // row-major identity (single stick) ⇒ NO RetileDescriptor (like the all-ones). The mq>1 cachewr
    // becomes matmul(kh[mqu,hd], I[hd,hd]) = kh, replacing the mqu·nqh·2 SlotSolo copies.
    //
    // BUG #1 (found via a real pod trace, 20cf993a): `assemble_attn`'s GQA new_k/new_v replication
    // (ir::bridge::tiled_op_sdsc_op::attn.rs, `attn_krep{h}`/`attn_vrep{h}`) references this SAME
    // `ident` tensor UNCONDITIONALLY — every model with attention, every mq, no env-var check at
    // all (the on-card multi-row plain copy is proven broken, so this IS the copy mechanism, not
    // an optional #3 optimization for it). But this placement was gated ONLY on the opt-in
    // SCRATCHY_SUPERDSC_KV_MATMUL env var — unset in a normal build, IDENTITY_TID was never placed
    // at all, so the always-referenced `ident` fell through to an unrelated synthetic offset with
    // whatever garbage happened to be there.
    //
    // BUG #2 (found via a real pod HARD-FAIL, 2026-07-27): the fix for BUG #1 was placed INSIDE the
    // `if let Some((nqh,nkvh,hd)) = ir.nodes.find_map(... n.output.region.rows.len > 1)` block right
    // above — an mq>1-ONLY guard (the batched-prefill selector/ones-reduce consts). Decode's tape only
    // ever has mq=1 AttnDecode nodes, so that ENTIRE enclosing block — including this placement — never
    // ran for decode's own bundle at all. Confirmed by the worker's own hard-fail: decode's baked
    // bundle_layout.json genuinely never had IDENTITY_TID, exactly as this predicts. Moved OUTSIDE that
    // mq>1 guard so it runs for ANY AttnDecode node regardless of mq, matching the consumer's real,
    // unconditional need (assemble_attn_head references `ident` at every mq, decode included).
    if let Some(hd) = ir.nodes.iter().find_map(|n| match &n.op {
        SubOp::AttnDecode { geom, .. } => Some(geom.hd().get() as u64),
        _ => None,
    }) {
        let seg = SegRole::Activation.segment();
        let off = seg_bytes[seg];
        let sz = hd * hd * 2; // [hd,hd] fp16 identity
        placements.insert(
            IDENTITY_TID,
            TensorPlacement {
                tid: IDENTITY_TID,
                bank: 0,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);

        // ATTN_ZERO (BUG #3, same class as #1/#2 above, found by re-auditing this exact area 2026-07-28):
        // `new_k`/`new_v` (AttnDecode's inputs[3]/[4]) are declared "[mq_pad, nkvh·hd] (worker zero-pads
        // rows)" — but nothing ever actually zeroed rows [mq..mq_pad) at ANY mq. This tid's placement was
        // ALSO stuck inside the old mq>1-only selector guard (moved out alongside IDENTITY_TID above), and
        // even there it was placed but never CONSUMED anywhere in this file — dead. `new_k`/`new_v` are a
        // lifetime-reused seg3 intermediate (compute_bundle_layout's linear-scan reuse), never explicitly
        // re-zeroed between steps/layers, so the padding rows can alias whatever UNRELATED tensor last
        // lived at that byte range — potentially large magnitude, not bounded "stale K-vector" data. The
        // causal mask (mask_neg, a moderate ~-32752 fp16 constant) only reliably neutralizes BOUNDED
        // garbage; it does not guarantee correctness against arbitrary aliased memory. Placed here
        // (unconditional, any mq with attention) so `lower_attn_node` can emit a real zero-copy into the
        // padding rows before GQA-replicate ever reads them, instead of relying on masking alone.
        // SIZE IS `mq_pad`, NOT ONE STICK (2026-07-29). The worker binds this as `[mq_pad, hd]` zeros
        // (`vec![0.0f32; mq_pad*hd]`, narrowed to 2-byte f16), and `mq_pad = mq.div_ceil(64)*64` — so a
        // hardcoded one-stick reservation matched the bind ONLY while every chunk fit in 64 query rows.
        // At mq_pad=128 the bind is 16384 B into an 8192 B placement and spills 8192 B into the seg3
        // tensors that follow. Nothing catches it: the shim's refill only checks the bind against the
        // whole SEGMENT size, never against `p.size` (see the `src.bytes.size() > p.size` guard added
        // alongside this in sdsc_shim.cpp).
        //
        // WHICH NEIGHBOURS ACTUALLY STAY CORRUPTED is decided by bind ORDER, because the shim refills
        // `std::map<std::string, HostBuf> bound` in LEXICOGRAPHIC tid order. Reserved tids are
        // `u32::MAX - k`, so a LARGER k sorts EARLIER and is overwritten by this tensor's spill with no
        // chance to be rewritten. Measured at mq=127 (spill `[1845248, 1853440)`):
        //   MAX-11 RMS_INVCOLS, MAX-13 FP8_POS448, MAX-14 FP8_NEG448, MAX-15 FP8_INV448 → sort BEFORE
        //     ATTN_ZERO (MAX-10) ⇒ left ZEROED.
        //   MAX-6 RMS_HALF and MAX-3 pmask → sort AFTER ⇒ rebound, harmless.
        // So the fp8 activation-quant clamp constants are the real victims. `inv448 = 0` makes
        // `ascale = amaxfl · inv448 = 0`, and `fq_dqa = mul(raw, col(ascale))` is then exactly 0, so all
        // seven fp8 projections emit zeros in all 40 layers: prefill writes an ALL-ZERO KV cache while
        // the residual carries the embedding through untouched. Decode (its own seg3 is not aliased,
        // and its mq_pad is 64, so its own bind fits) stays numerically healthy but attends a zero
        // prefix — leaving it conditioned on the last prompt token alone. On hardware that read as
        // FLUENT, on-grammar output about entirely the wrong subject (asked about the history of
        // France, answered about fractions of an inch).
        // ⚠ An earlier version of this comment blamed pmask being zeroed to "valid". That is WRONG —
        // pmask is rebound after this tensor, and the only pmask bytes any op reads are the 512 B its
        // own bind restores. Recorded because that false story invites a "fix" that merely reorders
        // binds, which would leave the over-bind in place.
        //
        // Derive the extent from the WIDEST AttnDecode in the tape. `div_ceil` keeps it at exactly one
        // stick for every mq<=64 (decode's mq=1 included), so previously-baked bundles are unchanged.
        let mqp_z = ir
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                SubOp::AttnDecode { .. } => Some(n.output.region.rows.len as u64),
                _ => None,
            })
            .max()
            .unwrap_or(1)
            .div_ceil(Fp16::ELEMS_PER_STICK as u64)
            * Fp16::ELEMS_PER_STICK as u64;
        let off = seg_bytes[seg];
        let sz = mqp_z * hd * 2;
        placements.insert(
            ATTN_ZERO_TID,
            TensorPlacement {
                tid: ATTN_ZERO_TID,
                bank: 0,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);
    }
    // (No AttnDecode node at all means no consumer ever references `ident`/ATTN_ZERO, so
    // SCRATCHY_SUPERDSC_KV_MATMUL has nothing to do for such a model — already a no-op, not a case that
    // needs handling here.)

    // ── RMSNorm Newton const [1,stick] ── If the tape has any RmsNorm, place `RMS_HALF_TID` in seg0
    // (ACTIVATION, re-bound per step like ATTN_SCALE — a seg1 weight would only be H2D'd at prepare).
    // The worker binds 0.5 (EXACT in fp16); the rmsnorm derives 1.0/1.5/−1.0 on-card from it. The old
    // RMS_SEED_TID seed-floor placement was REMOVED: the amax-normalized reciprocal seed never
    // underflows, so no floor const is needed. See [`RMS_HALF_TID`].
    if ir
        .nodes
        .iter()
        .any(|n| matches!(n.op, SubOp::RmsNorm { .. }))
    {
        let a = SegRole::Activation.segment();
        // Bind as a [1, stick] row (64 copies) so every derived-constant op is a uniform [1, stick]
        // elementwise op (no [1,1]→stick broadcast bookkeeping).
        let sz = Fp16::ELEMS_PER_STICK as u64 * 2;
        let hoff = seg_bytes[a];
        placements.insert(
            RMS_HALF_TID,
            TensorPlacement {
                tid: RMS_HALF_TID,
                bank: 0,
                role: SegRole::Activation,
                segment: a,
                offset: hoff,
                size: sz,
            },
        );
        seg_bytes[a] = align128(hoff + sz);
        // RMS_INVCOLS `[1,stick]` = 1/cols (worker-bound), for the mq>1 sum-based amax pre-scale + un-scale.
        let ioff = seg_bytes[a];
        placements.insert(
            RMS_INVCOLS_TID,
            TensorPlacement {
                tid: RMS_INVCOLS_TID,
                bank: 0,
                role: SegRole::Activation,
                segment: a,
                offset: ioff,
                size: sz,
            },
        );
        seg_bytes[a] = align128(ioff + sz);
    }

    // ── fp8 W8A8 activation-quant consts [1,stick] (seg0, worker-bound like RMS_HALF) ── E4M3 clamp
    // bounds ±448 + 1/448 for `qfp8ch` (per-token amax → a_scale). Placed iff the tape has an fp8
    // (arity-3) MatmulTile. Unplaced (and thus value-0) was the "clamp consts default 0 → wrong quant"
    // gap; placing them here + binding in the worker gives the real E4M3 bounds.
    if ir
        .nodes
        .iter()
        .any(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
    {
        let a = SegRole::Activation.segment();
        let sz = Fp16::ELEMS_PER_STICK as u64 * 2; // [1, stick] fp16
        for tid in [FP8_POS448_TID, FP8_NEG448_TID, FP8_INV448_TID] {
            let off = seg_bytes[a];
            placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role: SegRole::Activation,
                    segment: a,
                    bank: 0,
                    offset: off,
                    size: sz,
                },
            );
            seg_bytes[a] = align128(off + sz);
        }
    }

    // ── INTERMEDIATES (seg3): lifetime-aware linear scan with byte-range reuse. ──
    // Collect every produced tensor that is NOT a source and NOT the logits, sorted
    // by first_def (def order == node order). Maintain `live` = currently-assigned
    // (offset, size, expiry=last_use) and `free` = reclaimed (offset, size) holes.
    // AttnDecode's new_k/new_v (inputs[3]/[4]) are RE-placed at the PADDED [mq_pad, nkvh·hd] size below,
    // so they must NOT also be placed by this general colored pool: a double-placement reserves a wasted
    // colored slot AND records a stale (unpadded) placement in the map that the later re-placement
    // overwrites — the coloring's live-set then reflects the wrong offset/size. Exclude them here; the
    // AttnDecode loop is their single placement.
    let replaced_kv: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            SubOp::AttnDecode { .. } if n.inputs.len() >= 5 => Some([
                n.inputs[3].tensor.index() as u32,
                n.inputs[4].tensor.index() as u32,
            ]),
            _ => None,
        })
        .flatten()
        .collect();
    let mut inter: Vec<u32> = first_def
        .keys()
        .copied()
        .filter(|&t| t >= ir.num_sources && t != result && !replaced_kv.contains(&t))
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

    // ── Attention KV cache (seg2) + scale const (seg1) for in-bundle attention ──
    // The IR's AttnDecode K/V cache tensors are reused as the per-layer IDENTITY but
    // RE-placed (override) at the BATCHED size `[nqh,hd,cap]·2` (the worker fills them
    // transposed-K + GQA-replicated V each step — they are read-only INPUTS in-bundle,
    // never written by a SuperDSC op). One shared `scale` const `[1,1]` (worker
    // synthesizes `1/sqrt(hd)`). Placed LAST so it overrides any earlier source/
    // intermediate classification of the cache ids.
    let mut scale_placed = false;
    for node in &ir.nodes {
        if let SubOp::AttnDecode {
            geom, layout: kv, ..
        } = &node.op
        {
            let nqh = geom.nqh().get() as u64;
            let nkvh = geom.nkvh().get() as u64;
            let hd = geom.hd().get() as u64;
            let k_id = kv.cache_tensor().index() as u32;
            let v_id = kv.v_cache_tensor().index() as u32;
            let cap = ir.tensors[k_id as usize].rows as u64; // prefix cache capacity (rows)
            let mq32 = node.output.region.rows.len; // chunk query rows (1=decode, >1=prefill)
            let mq = mq32 as u64;
            // THE SAME DOOR AS THE EMIT — `PaddedMq::of_bundle`, the one parse boundary from a
            // bundle's runtime width to the pad law, so the placements and the ops they hold cannot
            // be sized by different pads (a decode width must be a baked ladder rung here exactly as
            // it must be in `lower_attn_node`). The placements below spend it in TWO roles and each
            // names its own: the staging tensors' ROW extent and the causal mask's SCORE width.
            let mq_pad = scratchy_subtile::sdsc_abstract::PaddedMq::of_bundle(
                mq32,
                rows_are_requests,
            )
            .ok_or_else(|| {
                SuperDscError(format!(
                    "AttnDecode t{}: a decode bundle at {mq32} query rows — not a width the \
                         decode ladder bakes (1, or PagedKvPool::BATCH_RUNGS), so its staging \
                         tensors and causal mask have no placeable pad.",
                    node.output.tensor.index() as u32
                ))
            })?
            .pad();
            // ── PAGED K/V POOL (`sdsc_abstract::PagedKvPool`) ──
            // This layer's slice of ONE page: Kᵀ, then V, then natural K. NOTHING here is sized by
            // the pool — the placement covers a single page+layer and the runtime adds
            // `layer + physical page` per launch, which is what takes the servable context out of
            // the baked program. All three planes sit in the KV segment because the re-roll executor
            // advances only that segment (and the weights) per layer.
            // A PAGE WIDER THAN THE SWEEP IS A WIN, NOT A REGRESSION — the reverse of what this
            // guard used to assert. It read: "a fold covers one page per launch, so a page wider than
            // the pre-paged capacity would sweep MORE KV for the same context — a regression, not a
            // feature." That is an argument about the sweep, and the sweep is not what a pass costs.
            //
            // Measured, eight requests, one page, varying ONLY the swept width through the ladder's
            // own rungs: 64 slots 39.5 ms, 128 slots 39.0 ms, 256 slots 42.0 ms. Four times the sweep
            // costs six percent. A pass is likewise flat in the ROWS it computes (256 rows to 32:
            // no measurable change). A fold pass is a fixed cost.
            //
            // What it is not flat in is the NUMBER of passes, and `reps = pages * requests`, so the
            // page width sets the pass count: at 576 tokens a 256-slot page is three passes per
            // request, and that is 137 ms of a bs=8 step against 39 ms at one page. Making the page
            // wider than the sweep is how the pass count comes down.
            //
            // The real ceiling is MEMORY: a page is `nkvh * hd * slots * 2 * layers`, so the pool has
            // to be sized for the batch width actually run. That is the constraint a 4-bit KV cache
            // lifts, by making a page denser instead of bigger.
            let pool =
                scratchy_subtile::sdsc_abstract::PagedKvPool::new(nkvh as usize, hd as usize);
            let _ = cap;
            let seg = SegRole::Kv.segment();
            let plane_bytes = pool.plane_stride() as u64 * 2;
            // THE REQUEST STRIDE THIS PLACEMENT WAS SIZED FOR. `plane_stride` already carries the
            // `ROWS` factor, so the pool grew ×ROWS per page here and nowhere else; publishing the
            // stride from the same `pool` is what stops the runtime's shift from disagreeing with the
            // addresses the ops baked.
            // ⛔ NO REQUEST STRIDE TO PUBLISH. A page holds slots, not requests, so there is no
            // "bytes between two requests' KV" for the runtime to shift by — a request is reached by its
            // PAGE, through the host's block table. Left at 0, which is what an unbatched bundle always
            // published and what `LaunchPages` reads as "no request dimension".
            let layer_base = seg_bytes[seg];
            // Kᵀ at 0, V at `v_plane_base`, natural K at `knat_plane_base` — the offsets
            // `PagedKvPool` bakes into the ops' plane-relative bases.
            for (tid, plane_off) in [
                // natural K, V, transposed K — the pre-paged cache's own order.
                (k_id, pool.knat_plane_base() as u64 * 2),
                (v_id, pool.v_plane_base() as u64 * 2),
                (kct_resident_tid(k_id), pool.kt_plane_base() as u64 * 2),
            ] {
                placements.insert(
                    tid,
                    TensorPlacement {
                        tid,
                        role: SegRole::Kv,
                        segment: seg,
                        bank: 0,
                        offset: layer_base + plane_off,
                        size: plane_bytes,
                    },
                );
            }
            // The per-layer stride the executor advances by MUST be exactly the three planes, with
            // no alignment padding creeping in between layers (hd and page_slots are 64-multiples,
            // so a plane is already 128-aligned — assert rather than trust).
            let layer_stride_bytes = pool.layer_stride() as u64 * 2;
            debug_assert_eq!(
                align128(layer_base + layer_stride_bytes),
                layer_base + layer_stride_bytes,
                "paged KV layer slice must be 128-aligned by construction"
            );
            seg_bytes[seg] = layer_base + layer_stride_bytes;
            let _ = nqh;
            // new_k/new_v (inputs[3]/[4]) are RE-placed at the PADDED size `[mq_pad,
            // nkvh·hd]·2` (fresh seg3 high-water) so the in-bundle transpose's
            // `[mq_pad,…]` read never aliases the next tensor (pad rows are masked).
            //
            // ⛔⛔⛔ THE WIDTH IS THE **DEVICE** WIDTH, NOT `nkvh·hd`. new_k/new_v are written by the
            // k_proj/v_proj MATMULS, and a matmul writes `DeviceWidth::for_output(m, n, k)` columns —
            // which for a kv width under 8 sticks is WIDER than the logical one
            // (`bump_sticks_to_splittable`: 192 = 3 sticks → 8 sticks = 512, so the proj fills ≥8
            // cores). Sizing the placement at the LOGICAL width made the producer address past its own
            // tensor into whatever seg3 placed next. It is invisible on granite (nkvh·hd = 512 is
            // already 8 sticks, so the bump is a no-op and both widths agree) and refuses at build time
            // on SmolLM2-135M (nkvh·hd = 192): `t340: access offset 0B + 31744B exceeds its placement
            // footprint 24576B`. `for_pointwise` is documented as EQUAL to `for_output` for any
            // producer matmul above the util floor — every real proj — so this is the SAME rule the
            // write uses, asked once, which is what that refusal's own last sentence demands.
            if node.inputs.len() >= 5 {
                let seg3i = SegRole::Intermediate.segment();
                for in_idx in [3usize, 4usize] {
                    let nid = node.inputs[in_idx].tensor.index() as u32;
                    // nkvh·hd, at the width the producing proj matmul actually writes.
                    let cols =
                        DeviceWidth::for_pointwise(ir.tensors[nid as usize].cols).get() as u64;
                    let _ = nkvh;
                    let pbytes = mq_pad.rows().row_axis_extent() as u64 * cols * 2;
                    let off = seg_bytes[seg3i];
                    placements.insert(
                        nid,
                        TensorPlacement {
                            tid: nid,
                            bank: 0,
                            role: SegRole::Intermediate,
                            segment: seg3i,
                            offset: off,
                            size: pbytes,
                        },
                    );
                    seg_bytes[seg3i] = align128(off + pbytes);
                }
            }
            if !scale_placed {
                // Attention MASKS → seg0 (activation, re-bound per step), worker-TILED to [nqh·mq, *]:
                // pmask=[nqh·mq, cap] (prefix validity), cmask=[nqh·mq, mq_pad] (causal triu). Shared across
                // layers (same shape every layer). The attention SOFTMAX SCALE is NOT placed here: it flows
                // through `scalarmul_scales` (config `attention_multiplier`, via `AttnDecode.scale`), placed
                // below with the other config scales and bound by the worker's scale loop — the emitter
                // NEVER recomputes 1/sqrt(hd), which would ignore the model's real attention_multiplier.
                //
                // NOTE (2026-07-28): pmask is OVER-RESERVED. Prefix validity is head- and
                // query-row-independent, the emitter reads it via In::mb_at (one-row mb-broadcast), and
                // BOTH workers now stage exactly one `[cap]` row — so `cap*2` would suffice and
                // [nqh·mq, cap] is ~507 KB of dead seg0 re-H2D'd per prefill forward. Deliberately NOT
                // shrunk here: pmask is placed for EVERY attention bundle, so resizing it shifts the
                // seg0 offsets of the DECODE bundle too (verified: all decode hashes move). Decode was
                // only just made coherent, and 507 KB is 2.7% of the ~19 MB of dead seg0 this pass
                // removes, so it is not worth perturbing a known-good bundle. Revisit once prefill TTFT
                // has settled and decode can be re-validated in the same run.
                // PREFIX-VALIDITY MASK stays in the ACTIVATION segment, ONE PAGE wide, describing
                // the TAIL page — the only page that is partially valid. Its own segment would let a
                // per-page shift select each page's row, but a bound tensor marks its WHOLE segment
                // for re-upload every step, and the only spare segments hold ~100-150 MB of
                // intermediates: that cost 1.8 ms per token. Full pages need no mask at all (see the
                // zero-masked fold variant), so no shift is needed and this can stay small.
                // ONE ROW PER PAGE, in the ACTIVATION segment: the fold re-launch walks along it,
                // so page `i`'s validity sits `i * PAGE_SLOTS` elements in.
                //
                // It lives here, and not in a segment of its own, because BINDING a tensor marks its
                // WHOLE segment for re-upload every step. The spare segments carry a couple of
                // hundred intermediates (~1.6 MB in the decode bundle), which at the preamble's
                // measured ~2.5 GB/s is ~0.64 ms per token of pure upload — most of the paged decode
                // regression. The activation segment is 0.04 MB and is uploaded every step anyway,
                // so the mask rides along for free.
                //
                // Shifting it per page is safe HERE and nowhere else: dumped from the baked bundle,
                // a fold group addresses exactly two things — the KV segment, and the mask. It
                // touches nothing else in the activation segment, so the shift moves nothing it reads.
                let mseg = SegRole::Activation.segment();
                let pmoff = seg_bytes[mseg];
                // ONE validity row over the max context, or one PER QUERY ROW when the rows are
                // separate requests and each carries its own history. The multiplier is the op row
                // count the attention sweeps (`nqh*mq`), which is what the per-row read addresses.
                // It stays 1 for a prompt chunk — at prefill's mq=96 the widened form would be
                // ~50 MB of mask re-uploaded every forward, and a prompt does not need it.
                // ONE BLOCK PER FOLD PASS when the rows are requests. A pass reads ONE request's
                // page, so every other row must be masked off for it — the mask is `[pass][row][page]`
                // and the runtime steps it by a whole `rows * page` block. A prompt keeps one
                // broadcast row per page, which is all its rows can need.
                let (pm_rows, pm_blocks) = if rows_are_requests {
                    (nqh * mq, MAX_FOLD_PASSES)
                } else {
                    (1, MAX_PAGES_PER_REQUEST)
                };
                let pmbytes = pm_rows
                    * pm_blocks
                    * scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u64
                    * 2;
                placements.insert(
                    ATTN_MASK_TID,
                    TensorPlacement {
                        tid: ATTN_MASK_TID,
                        bank: 0,
                        role: SegRole::Activation,
                        segment: mseg,
                        offset: pmoff,
                        size: pmbytes,
                    },
                );
                seg_bytes[mseg] = align128(pmoff + pmbytes);
                let a = SegRole::Activation.segment();
                let cmoff = seg_bytes[a];
                // The mask is `[nqh·mq, mq_pad]`: `mq_pad` is its COLUMN count, the score width —
                // the row extent answers here only through the new-block identity.
                let cmbytes = nqh * mq * mq_pad.cols().score_axis_extent() as u64 * 2;
                placements.insert(
                    ATTN_CAUSAL_TID,
                    TensorPlacement {
                        tid: ATTN_CAUSAL_TID,
                        bank: 0,
                        role: SegRole::Activation,
                        segment: a,
                        offset: cmoff,
                        size: cmbytes,
                    },
                );
                seg_bytes[a] = align128(cmoff + cmbytes);
                // DIAGNOSTIC probe (mq>1 only): persistent seg0 buffer for layer-0's pre-selector new_v
                // [mq_pad, nkvh·hd]. lower_attn_node copies layer-0 new_v here; the worker reads it to split
                // the structural inf (matmul vs selector). Never reused ⇒ survives to post-prefill readback.
                if mq > 1 {
                    let npoff = seg_bytes[a];
                    let npbytes = mq_pad.rows().row_axis_extent() as u64 * nkvh * hd * 2;
                    placements.insert(
                        NEW_V_PROBE_TID,
                        TensorPlacement {
                            tid: NEW_V_PROBE_TID,
                            bank: 0,
                            role: SegRole::Activation,
                            segment: a,
                            offset: npoff,
                            size: npbytes,
                        },
                    );
                    seg_bytes[a] = align128(npoff + npbytes);
                }
                scale_placed = true;
            }
        }
    }

    // ⛔ BUILD GUARD (guard-every-crash-at-build-time, structural class): every RESERVED
    // synthetic constant (ROPE_P / ATTN_SCALE / ATTN_MASK / ATTN_CAUSAL / RMS_SEED /
    // RMS_HALF) is bound by the worker PER STEP via `acts`, so it MUST be placed in seg0
    // (ACTIVATION). A reserved tid placed in seg1 (WEIGHT) is only H2D'd at PrepareModel —
    // which binds ONLY the manifest's model weights — so it would stay ZERO on-card,
    // producing silent wrong numerics with NO crash to trace (this is exactly the RoPE-P
    // bug: P in seg1 ⇒ rot=matmul(x,0)=0 ⇒ RoPE collapsed to x·cos). A misplacement is a
    // pure-data emitter mistake; turn it into a `cargo build` panic instead of garbage out.
    {
        let act_seg = SegRole::Activation.segment();
        // NOTE: ATTN_SCALE_TID is intentionally absent — the attention scale is no longer a reserved
        // per-step const; it flows through `scalarmul_scales` (config attention_multiplier) whose
        // placements are proven seg0 by the scale loop above.
        for &rtid in &[
            ROPE_P_TID,
            ATTN_MASK_TID,
            ATTN_CAUSAL_TID,
            RMS_HALF_TID,
            RMS_INVCOLS_TID,
            ONES_REDUCE_TID,
        ] {
            if let Some(pl) = placements.get(&rtid) {
                assert!(
                    pl.segment == act_seg && matches!(pl.role, SegRole::Activation),
                    "reserved synthetic tid t{rtid} placed in segment {} (role {:?}) — it MUST be \
                     seg{act_seg} ACTIVATION: the worker binds it PER STEP via `acts`, so a \
                     WEIGHT-segment reserved tid is never bound (PrepareModel binds only manifest \
                     weights) ⇒ stays ZERO ⇒ silent wrong numerics",
                    pl.segment,
                    pl.role,
                );
            }
        }
    }

    // ── granite ScalarMul scale constants ── collect the DISTINCT scale values (embedding / residual /
    //    attention / logits multipliers) and place a `[1,1]` worker-bound const per scale in seg0
    //    (ACTIVATION, exactly like ATTN_SCALE). `lower_scalarmul_node` reads the index here → the const TID
    //    the pointwise `mul` multiplies by; the worker binds each `t{tid}=[scale]`. NO weight-fold, NO
    //    host-route — a real on-device pointwise multiply (the ATTN_SCALE mechanism).
    let mut scalarmul_scales: Vec<f32> = Vec::new();
    let push_scale = |scale: f32, scalarmul_scales: &mut Vec<f32>| {
        if !scalarmul_scales
            .iter()
            .any(|s| s.to_bits() == scale.to_bits())
        {
            scalarmul_scales.push(scale);
        }
    };
    // ⛔ NOTHING IS SEEDED HERE. This registry holds exactly the model's own multipliers, in exactly
    // the order the node walk below registers them — the same contents and the same indices
    // `subtile→superdsc` produces, because the index IS the device tid
    // (`SCALARMUL_SCALE_BASE - i`). Two algebraic identities (`0.0`, `1.0`) were once pushed FIRST
    // for the KTIR construction's `linalg.*` `outs` seeds, which shifted every model scale by two
    // slots and so changed the address every constant reaches the card at. Those seeds are
    // immediates in the KTIR now (`KtirFunc::splat_zero`/`splat_one`) and never reach a descriptor,
    // because the ported bodies fold the accumulator seed into the contraction exactly as
    // `subtile→superdsc` does.
    for node in &ir.nodes {
        // EVERY config-derived on-device scalar flows through this ONE registry → a `[1,1]` worker-bound
        // const: the muP ScalarMul multipliers (embedding/residual/logits) AND the attention softmax scale
        // (`AttnDecode.scale` == config `attention_multiplier`, set by `attention_scale_for`). NO recompute
        // (the worker must never invent `1/sqrt(hd)` — that ignored the model's real attention_multiplier).
        match &node.op {
            SubOp::ScalarMul { scale } => push_scale(*scale, &mut scalarmul_scales),
            // torch-spyre `spyre__sdpa_overrideable`: scaling_factor = sqrt(scale), applied to BOTH q and K
            // (`query * scaling_factor`, `key * scaling_factor`). Register √scale for the prefill split; the
            // un-split `scale` stays for the decode qs.
            SubOp::AttnDecode { scale, .. } => {
                push_scale(*scale, &mut scalarmul_scales);
                push_scale(scale.sqrt(), &mut scalarmul_scales);
            }
            // RMSNorm epsilon (config `rms_norm_eps`) flows through the SAME registry — a `[1,1]`
            // worker-bound const the rmsnorm adds to the mean-of-squares (config value, not dropped).
            // ⛔ THE EPSILON ONLY — the mean-of-squares divisor is NOT a registry scale. `1/cols` is
            // bound at the reserved `RMS_INVCOLS_TID` as a `[1, stick]` row with its own placement,
            // which is where `subtile→superdsc` reads it from; pushing it here as well added one
            // registry slot per rmsnorm node, shifting the tid of every constant registered after it.
            SubOp::RmsNorm { eps, .. } => push_scale(*eps, &mut scalarmul_scales),
            _ => {}
        }
    }
    for i in 0..scalarmul_scales.len() {
        let a = SegRole::Activation.segment();
        let off = seg_bytes[a];
        let tid = scalarmul_scale_tid(i);
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: a,
                bank: 0,
                offset: off,
                size: 2,
            },
        );
        seg_bytes[a] = align128(off + 2);
    }

    // Synthetic intermediates (assigned lazily during lowering) start ABOVE the
    // colored intermediates in seg3, so they never overlap a real intermediate.
    let synth = std::cell::RefCell::new(SynthAlloc {
        next: seg_bytes[SegRole::Intermediate.segment()],
        map: std::collections::BTreeMap::new(),
        sizes: std::collections::BTreeMap::new(),
    });
    // Every placed tensor's spelling, registered at the one site that knows the whole set.
    let ids = std::cell::RefCell::new(
        placements
            .keys()
            .map(|&tid| {
                (
                    ktir_superdsc::place::act_name(tid),
                    bundle::PlaceId::Act(tid),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>(),
    );
    Ok(BundleLayout {
        placements,
        ids,
        segment_bytes: seg_bytes,
        weight_bank_bytes,
        kernel_weights: std::collections::BTreeMap::new(),
        scalarmul_scales,
        synth,
        arrangements: std::cell::RefCell::new(std::collections::BTreeMap::new()),
        kv_request_stride_bytes,
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
fn spill_weight_tail(
    placements: &mut std::collections::BTreeMap<u32, TensorPlacement>,
    seg_bytes: &mut [u64; 7],
    per_layer_tids: &std::collections::BTreeSet<u32>,
) -> Result<(), SuperDscError> {
    let w_seg = SegRole::Weight.segment();
    if seg_bytes[w_seg] <= bundle::MAX_SEGMENT_BYTES {
        return Ok(()); // the overwhelming case: nothing moves, nothing is re-offset.
    }
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
        // First declared slot with room for it. `MAX_SEGMENT_BYTES` binds the spill slot too — it is
        // a device region like any other — so this cannot trade one over-size segment for another.
        let Some(&spill) = WEIGHT_SPILL_SEGS
            .iter()
            .find(|&&s| align128(seg_bytes[s]) + sz <= bundle::MAX_SEGMENT_BYTES)
        else {
            return Err(SuperDscError(format!(
                "seg{w_seg} packs {} B, {} B past the {} B one device region can hold, and every \
                 declared spill slot {WEIGHT_SPILL_SEGS:?} is too full to take t{tid} ({sz} B). The \
                 tail that CAN move is already moved: what is left is the per-layer block, and the \
                 rolled body advances exactly one segment's base per layer, so it cannot be split \
                 without making the weight segment a bank of regions (see `spill_weight_tail`).",
                seg_bytes[w_seg],
                seg_bytes[w_seg] - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
            )));
        };
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
    Ok(())
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
) -> std::collections::BTreeMap<u32, Vec<u32>> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    let mut iters: u32 = 0;
    for instr in tape.instrs() {
        if let Instr::OpenLoop {
            bound: LoopBound::Const(it),
            ..
        } = instr
        {
            iters = *it;
        }
    }
    let mut out: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    if iters < 2 {
        return out; // nothing repeats: no layer structure, so no banking is expressible
    }
    for instr in tape.instrs() {
        if let Instr::Compute { inputs, .. } = instr {
            for ci in inputs.iter() {
                if let ComputeInput::External {
                    tensor,
                    per_layer: pl,
                    ..
                } = ci
                    && pl.len() as u32 == iters
                {
                    out.entry(tensor.index() as u32)
                        .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
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
    per_layer_ext: &std::collections::BTreeMap<u32, Vec<u32>>,
) -> Result<Vec<u64>, SuperDscError> {
    let w_seg = SegRole::Weight.segment();
    if seg_bytes[w_seg] <= bundle::MAX_SEGMENT_BYTES {
        return Ok(Vec::new()); // the overwhelming case: one region holds every weight.
    }
    // ── The per-layer WEIGHT classes, as layer-indexed tid lists ──
    // Only classes whose layer-0 tid is a WEIGHT in this segment: `per_layer_ext` also carries the
    // KV caches (seg2), which have their own stride and their own segment.
    let classes: Vec<&Vec<u32>> = per_layer_ext
        .values()
        .filter(|tids| {
            tids.first().is_some_and(|t0| {
                placements
                    .get(t0)
                    .is_some_and(|p| p.segment == w_seg && matches!(p.role, SegRole::Weight))
            })
        })
        .collect();
    let Some(layers) = classes.iter().map(|c| c.len()).max() else {
        return Err(SuperDscError(format!(
            "seg{w_seg} packs {} B, {} B past the {} B one device region can hold, and the tape has \
             NO per-layer weight classes — so there is no layer boundary to split the segment on. \
             Banking a weight segment needs the re-rolled layer loop; an unrolled bundle this large \
             cannot be addressed.",
            seg_bytes[w_seg],
            seg_bytes[w_seg] - bundle::MAX_SEGMENT_BYTES,
            bundle::MAX_SEGMENT_BYTES,
        )));
    };
    if classes.iter().any(|c| c.len() != layers) {
        return Err(SuperDscError(
            "bank_weight_segment: per-layer weight classes disagree on the layer count — one \
             tensor repeats fewer times than another, so no layer boundary is well defined"
                .into(),
        ));
    }
    // ── The per-layer stride, from the packing that already exists ──
    // Layer v's tids all sit at `their layer-0 offset + v·stride`; that uniformity is what the
    // rolled body needs and what `reroll` re-verifies. Derive it here from layer 0 → layer 1 and
    // hold every class to it, because banking DIVIDES by it.
    let off_of = |t: &u32| -> Option<u64> { placements.get(t).map(|p| p.offset) };
    let mut stride: u64 = 0;
    for c in &classes {
        let (Some(a), Some(b)) = (off_of(&c[0]), off_of(&c[1])) else {
            return Err(SuperDscError(
                "bank_weight_segment: a per-layer weight has no placement".into(),
            ));
        };
        let d = b.wrapping_sub(a);
        if stride == 0 {
            stride = d;
        } else if stride != d {
            return Err(SuperDscError(format!(
                "bank_weight_segment: NON-UNIFORM per-layer weight stride ({stride} vs {d} B). \
                 Banking splits the segment at a layer boundary, which needs every layer packed at \
                 one stride."
            )));
        }
    }
    if stride == 0 {
        return Err(SuperDscError(
            "bank_weight_segment: per-layer weight stride is 0 — every layer would read layer 0's \
             weights"
                .into(),
        ));
    }
    if stride > bundle::MAX_SEGMENT_BYTES {
        return Err(SuperDscError(format!(
            "bank_weight_segment: ONE layer is {stride} B, past the {} B a single device region can \
             hold. A bank boundary can only fall on a layer boundary, so this model cannot be \
             addressed by advancing a per-layer base — it needs the layer itself split, which the \
             rolled body cannot express.",
            bundle::MAX_SEGMENT_BYTES,
        )));
    }
    // ── How many whole layers one region holds, and therefore how many banks ──
    let lpb = (bundle::MAX_SEGMENT_BYTES / stride) as usize;
    let layer_banks = layers.div_ceil(lpb);
    // Which layer each per-layer tid belongs to, and the base every layer's offsets are measured
    // from (the first per-layer weight's offset). Subtracting `pl_base` puts layer 0 at offset 0 in
    // bank 0, so banks ≥ 1 have no leading hole where the non-per-layer head used to sit.
    let mut layer_of: std::collections::BTreeMap<u32, usize> = Default::default();
    for c in &classes {
        for (v, t) in c.iter().enumerate() {
            layer_of.insert(*t, v);
        }
    }
    let pl_base = layer_of
        .keys()
        .filter_map(off_of)
        .min()
        .expect("a class exists, so a placement exists");
    // ── Re-place: per-layer weights by formula, everything else into the tail bank ──
    let mut bank_bytes = vec![0u64; layer_banks];
    let mut tail: Vec<u32> = placements
        .values()
        .filter(|p| {
            p.segment == w_seg
                && matches!(p.role, SegRole::Weight)
                && !layer_of.contains_key(&p.tid)
        })
        .map(|p| p.tid)
        .collect();
    tail.sort_by_key(|t| placements[t].offset); // keep the packed order the tape produced
    for (tid, v) in layer_of.clone() {
        let b = v / lpb;
        let p = placements
            .get_mut(&tid)
            .expect("layer_of was built from placements");
        p.bank = b as u32;
        p.offset = p.offset - pl_base - (b * lpb) as u64 * stride;
        bank_bytes[b] = bank_bytes[b].max(align128(p.offset + p.size));
    }
    // The non-per-layer weights (the head that sorted before layer 0, the final norm, the lm_head /
    // tied embedding) share ONE bank: the last layer bank if they fit in it, else a bank of their
    // own. Sharing costs nothing and saves a region; what matters is that they are TOGETHER, so the
    // suffix — which reads the norm and the lm_head in one launch — needs exactly one base.
    let tail_len: u64 = tail
        .iter()
        .fold(0u64, |acc, t| align128(acc + placements[t].size));
    // An empty tail needs no room, so it "fits" the last layer bank trivially — one condition, not
    // two arms that happen to agree.
    let tail_bank = if tail.is_empty()
        || align128(bank_bytes[layer_banks - 1]) + tail_len <= bundle::MAX_SEGMENT_BYTES
    {
        layer_banks - 1
    } else {
        bank_bytes.push(0);
        layer_banks
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
                 device region can hold (stride {stride} B/layer, {lpb} layer(s)/bank, {layers} \
                 layers, {} bank(s)).",
                bytes - bundle::MAX_SEGMENT_BYTES,
                bundle::MAX_SEGMENT_BYTES,
                bank_bytes.len(),
            )));
        }
    }
    seg_bytes[w_seg] = bank_bytes[0];
    Ok(bank_bytes[1..].to_vec())
}

/// The deterministic synthetic source value (MUST match the worker self-test in
/// `spyre_worker.rs::superdsc_selftest`). `id` = source tid, `j` = element index.
#[inline]
pub fn dbg_synth_val(id: usize, j: usize) -> f32 {
    (((id * 131 + j * 7) % 197) as f32 / 197.0 - 0.5) * 0.1
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
pub fn write_eval_golden<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    dir: &std::path::Path,
) -> std::io::Result<()> {
    let dbg = dir.join("dbg");
    std::fs::create_dir_all(dbg.join("golden"))?;
    let nsrc = ir.num_sources as usize;
    let src_vals: Vec<Vec<f32>> = (0..nsrc)
        .map(|id| {
            let t = ir.tensors[id];
            let n = (t.rows as usize) * (t.cols as usize);
            (0..n).map(|j| dbg_synth_val(id, j)).collect()
        })
        .collect();
    let src_refs: Vec<&[f32]> = src_vals.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(ir, &src_refs);
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
    for id in 0..nsrc {
        if id > 0 {
            shapes.push(',');
        }
        shapes.push_str(&format!(
            "{{\"tid\":{id},\"rows\":{},\"cols\":{}}}",
            ir.tensors[id].rows, ir.tensors[id].cols
        ));
    }
    shapes.push(']');
    std::fs::write(dbg.join("source_shapes.json"), shapes)?;
    let mut wt: Vec<u32> = weight_ids.iter().copied().collect();
    wt.sort_unstable();
    let wtj: Vec<String> = wt.iter().map(|w| w.to_string()).collect();
    std::fs::write(dbg.join("weight_tids.json"), format!("[{}]", wtj.join(",")))?;
    // granite ScalarMul scale VALUES (index i ↔ `scalarmul_scale_tid(i)`). SAME first-seen walk order as
    // `compute_bundle_layout` (both iterate `ir.nodes`, dedup by bits) ⇒ index↔TID consistent. The worker
    // reads this + binds each `t{scalarmul_scale_tid(i)} = [scale_i]` so the on-device pointwise `mul` gets
    // the real value (unbound = 0 = wrong). Written alongside source_shapes.json (the run reads both).
    let mut sm: Vec<f32> = Vec::new();
    for node in &ir.nodes {
        if let SubOp::ScalarMul { scale } = &node.op
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

/// Op kind for MEDIUM-GRAIN fusion grouping (torch-spyre's Inductor-kernel
/// granularity, mirrored SDSC-native): a run of `Pure` ops (rmsnorm/matmul/
/// elementwise — no per-entry shim interception) FUSES into ONE concrete dxp
/// bundle of up to `g` trips; ops the shim intercepts per-manifest-entry must
/// stay SINGLETON so the interception lands on the right entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupKind {
    /// Fusable on-card trip (no per-entry shim handling).
    Pure,
    /// `host_kv_write`: the shim host-scatters KV then does `oi += skip` to skip
    /// the on-card cachewr copies it replaced. This entry AND those `skip`
    /// entries MUST be singletons so `oi += skip` skips exactly them.
    HostKv { skip: usize },
    /// `slot_write` DECODE cachewr: every fused copy writes the SAME slot (baked at slot 0,
    /// shim shifts the group base by `slot_pos·stride` ONCE). Consecutive `Slot` trips FUSE
    /// ≤ g (the launch reduction). MUST all share one slot — enforce via [`GroupKind::SlotSolo`]
    /// for the distinct-slot case, so a fused `Slot` group is uniform-slot by construction.
    ///
    /// CARRIES ITS REQUEST, and a run BREAKS where the request changes. A group is one launch and a
    /// launch resolves ONE request's page table and write cursor, so two requests cannot share a
    /// group — but a request's own kv-head copies still can, and they must: a batched decode step's
    /// cache writes are `nkvh · requests · 2` per layer, and making each its own singleton (which is
    /// what treating them as distinct-slot did) cost ~5,300 extra launches per forward and ran a
    /// batch of 8 twenty times slower than one request. 0 for an unbatched bundle, so its grouping
    /// is exactly what it was.
    Slot { req: u32 },
    /// `slot_write` PREFILL cachewr: writes a DISTINCT per-entry slot (baked at `s·hd`). The
    /// shim's single per-group slot-shift CANNOT express distinct slots, so each is its OWN
    /// singleton group (launched with its baked slot doff; prefill `slot_pos==0` ⇒ no shift).
    /// This makes "distinct-slot copy fused into a uniform-shift group" UNCONSTRUCTABLE.
    SlotSolo,
    /// PAGE FOLD (`kv_page_fold`): folds one page of resident prefix. Consecutive fold trips FUSE
    /// (they share a page base) but must never fuse with anything else — the shim re-launches this
    /// group per page, and a stray op swept in would re-run per page too, re-seeding the new-token
    /// block or re-adding the residual once per page.
    PageFold,
    /// `slab_write` DECODE incremental Kᵀ restickify (kill-restickify Stage 2, hd==64): re-transposes
    /// ONLY the current 64-slot slab; the shim shifts the group's seg2 base by
    /// `(slot_pos/64)·slab_stride_bytes` ONCE (each op baked at slab 0). Consecutive `Slab` trips FUSE
    /// ≤ g (the nkvh per-kv-head restickifies, all one `slab_stride_bytes`). DISTINCT kind from `Slot`:
    /// its stride (STICK_BYTES·stick = 8192) ≠ the cachewr's `slot_stride_bytes` (128), so fusing the two
    /// would trip the per-group uniform-stride assert (mixed-stride group → wrong base shift).
    Slab,
}

/// WHICH REQUEST A TRIP'S ADDRESSING BELONGS TO.
///
/// Wraps the op's `kv_request` so the group walk cannot silently compare it against something else
/// (a slot index, a page, a `rep`) — every one of those is also a small integer, and this file's
/// history is a list of exactly that mistake. `0` is both "request 0" and "untagged", deliberately:
/// an unbatched bundle must stay byte-identical, so the two are the same value. That conflation is
/// SAFE HERE and nowhere else — for a group BOUNDARY, treating an untagged trip as request 0 can
/// only break a run that would otherwise have fused, and over-breaking costs launches while
/// under-breaking costs correctness. (Where the conflation is NOT safe is the page base and write
/// cursor; those live behind `fold_plan::RequestSlot`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripRequest(pub u32);

/// A body trip: WHAT it is plus WHOSE it is. The pair travels together so the fusion walk cannot
/// consult one without the other.
///
/// Before this type the walk saw only `GroupKind`, and only the `Slot` variant happened to carry a
/// request. So every OTHER kind fused across requests — and a group is ONE launch, which resolves
/// ONE request's page table, so a `Pure` run spanning two requests gave both of them the FIRST
/// request's KV page. That is the same class of bug as the six before it: correctness depended on
/// each variant separately remembering to check the request, and a new variant (or an op newly
/// tagged with a request, like the per-page Kᵀ re-transpose) got it wrong by default.
/// Now the request is checked ONCE, for all kinds, in [`Trip::fusable_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trip {
    pub kind: GroupKind,
    pub req: TripRequest,
}

impl GroupKind {
    /// Whether two trips of these kinds may share one concrete bundle, IGNORING the request.
    ///
    /// Note `SlotSolo` and `HostKv` are not fusable even with THEMSELVES — that is how "this trip is
    /// a singleton by construction" is expressed, and it is what makes the walk's advance-by-one
    /// fallback unnecessary: a run is `[start, start+1)` unless the kind opts in.
    fn fusable_kind_with(self, other: GroupKind) -> bool {
        match (self, other) {
            (GroupKind::Pure, GroupKind::Pure) => true,
            (GroupKind::Slab, GroupKind::Slab) => true,
            (GroupKind::PageFold, GroupKind::PageFold) => true,
            (GroupKind::Slot { req: a }, GroupKind::Slot { req: b }) => a == b,
            // SlotSolo: distinct baked slot per entry — one per-group shift cannot express two.
            // HostKv: the shim's `oi += skip` must land on this exact entry.
            _ => false,
        }
    }
}

impl Trip {
    pub fn new(kind: GroupKind, req: TripRequest) -> Trip {
        Trip { kind, req }
    }

    /// THE ONE RULE THAT DECIDES EVERY GROUP BOUNDARY: same request, and kinds that fuse.
    ///
    /// A group is one launch; a launch resolves one request's page table and one write cursor and
    /// shifts the segment base ONCE. Two requests therefore cannot share a group whatever their
    /// kinds — so the request test lives HERE, above the kind test, and applies to kinds that do not
    /// exist yet.
    fn fusable_with(&self, other: &Trip) -> bool {
        self.req == other.req && self.kind.fusable_kind_with(other.kind)
    }
}

/// Partition trip indices `0..kinds.len()` into CONTIGUOUS groups for medium-grain
/// fusion. Contiguous ⇒ emission (dataflow) order preserved ⇒ every producer
/// precedes its consumer, so cross-group edges thread through the shared global
/// HBM placement (addresses are grouping-invariant — baked at lowering, not here).
/// A run of `Pure` trips is chunked ≤ `g`; each `HostKv` (with the `skip` entries
/// it replaces) and each `Slot` is its OWN singleton. `g==1` reproduces the per-op
/// partition EXACTLY. PURE (no env/IO) so Kani verifies the covering-partition
/// invariant (no trip dropped or double-emitted → no dropped/duplicated compute).
pub fn group_ranges(trips: &[Trip], g: usize) -> Vec<core::ops::Range<usize>> {
    let g = if g == 0 { 1 } else { g };
    let n = trips.len();
    let mut groups: Vec<core::ops::Range<usize>> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if let GroupKind::HostKv { skip } = trips[i].kind {
            // this entry + the `skip` on-card copies it replaced: each its own singleton, so the
            // shim's per-entry `oi += skip` advances over exactly them.
            let end = i.saturating_add(1).saturating_add(skip).min(n);
            let mut k = i;
            while k < end {
                groups.push(k..k + 1);
                k += 1;
            }
            i = end;
            continue;
        }
        // EVERY other kind takes the same walk. `end` starts one past `start`, so the walk always
        // advances (termination is structural, not a special case) and a kind that fuses with
        // nothing — SlotSolo — comes out a singleton without an arm of its own.
        let start = i;
        let mut end = start + 1;
        let mut cnt = 1usize;
        while end < n && cnt < g && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        groups.push(start..end);
        i = end;
    }
    groups
}

/// CBMC-tractable SCALAR twin of [`group_ranges`]'s partition walk (a `Vec<Range>`
/// is CBMC-hostile — dynamic alloc over symbolic control flow blows up). Runs the
/// IDENTICAL i-advancing control flow but tracks only scalars: the covered-prefix
/// end and whether every emitted group is contiguous + correctly sized (Pure ≤ g,
/// any non-Pure group == singleton). Returns `(covered_end, ok)`. Because it shares
/// [`group_ranges`]'s exact walk, proving `(covered_end==n && ok)` proves the Vec
/// version is an EXACT covering partition (no trip dropped/double-emitted) with the
/// size/singleton invariant. Keep byte-identical to `group_ranges`'s match arms.
pub fn group_ranges_cover_ok(trips: &[Trip], g: usize) -> (usize, bool) {
    let g = if g == 0 { 1 } else { g };
    let n = trips.len();
    let mut i = 0usize;
    let mut expect = 0usize; // start of the next group must equal the covered-prefix end
    let mut ok = true;
    while i < n {
        if let GroupKind::HostKv { skip } = trips[i].kind {
            let end = i.saturating_add(1).saturating_add(skip).min(n);
            let mut k = i;
            while k < end {
                ok &= k == expect; // contiguous; each is a [k,k+1) singleton (len 1 ≤ g)
                expect = k + 1;
                k += 1;
            }
            i = end;
            continue;
        }
        let start = i;
        let mut end = start + 1;
        let mut cnt = 1usize;
        while end < n && cnt < g && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        ok &= start == expect;
        ok &= (end - start) <= g; // group ≤ g
        ok &= end > start; // non-empty ⇒ the walk advances ⇒ terminates
        expect = end;
        i = end;
    }
    (expect, ok)
}

/// Whether ANY group of this partition spans two requests — the bug-#7 predicate.
///
/// A group is one launch, and a launch shifts the KV base by ONE request's page/cursor. So a group
/// holding trips of two requests gives the second request the first's KV, silently: fluent output,
/// wrong tokens. Scalar (Vec-free) so CBMC can prove it never happens for ANY trip sequence.
pub fn group_spans_two_requests(trips: &[Trip], g: usize) -> bool {
    let g = if g == 0 { 1 } else { g };
    let n = trips.len();
    let mut i = 0usize;
    while i < n {
        if let GroupKind::HostKv { skip } = trips[i].kind {
            i = i.saturating_add(1).saturating_add(skip).min(n);
            continue;
        }
        let start = i;
        let mut end = start + 1;
        let mut cnt = 1usize;
        while end < n && cnt < g && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        let mut k = start;
        while k < end {
            if trips[k].req != trips[start].req {
                return true;
            }
            k += 1;
        }
        i = end;
    }
    false
}

/// Medium-grain fusion group size (trips per concrete dxp bundle). `1` = the
/// historical per-op path (each trip its own program). Env `SCRATCHY_SUPERDSC_GROUP_SIZE`.
/// Hoisted OUT of the pure `group_ranges` (env reads unbound CBMC — see `plan_capped`).
pub fn group_size() -> usize {
    // Decode is ~100% launch-count bound (measured ~3640 launches/tok @ ~41µs); fusing consecutive ops
    // into ≤g groups collapses that. 2048 was too aggressive — a fp8-dynamic granite prefill body fused
    // 512-trip groups fine but a dxp_standalone `vector::_M_range_check` crash surfaced once fusion pushed
    // past that. `SCRATCHY_SUPERDSC_GROUP_SIZE=1` is the explicit DEBUG opt-out (per-op fault isolation).
    // Read at EMIT time + forwarded by each arch build.rs (rerun-if-env-changed + rustc-env) so a change
    // recompiles.
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
    std::env::var("SCRATCHY_SUPERDSC_GROUP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&g| g >= 1)
        .unwrap_or(128)
}

/// The FLAT (no-loop) `bundle.mlir` body for the all-time=1 case: one
/// `sdscbundle.sdsc_execute` per op, addresses baked concrete in the json. Kept
/// BYTE-IDENTICAL to the historical emission so the green single-shot tests do
/// not change. `emit_bundle_mlir` delegates here when no op has time>1.
pub fn bundle_mlir(sdsc_filenames: &[String]) -> String {
    let mut body = String::new();
    for f in sdsc_filenames {
        body.push_str(&format!(
            "    sdscbundle.sdsc_execute () {{sdsc_filename=\"{f}\"}}\n"
        ));
    }
    format!("module {{\n  func.func @sdsc_bundle() {{\n{body}    return\n  }}\n}}\n")
}

/// Expand one [`EmittedOp`] into its CONCRETE per-trip [`SdscOp`]s. A time=1 op
/// yields `[op.clone()]`. A time=N op yields N copies: in trip `t`, every tiled
/// tensor's AllocNode start addresses are bumped by `t · stride_bytes` (the
/// `affine_strides[ti]["out"]` advance); addresses stay CONCRETE. This PRE-UNROLL
/// replaces a symbolic `scf.for` — dxp's always-on `LoopUnroll` would otherwise
/// clone an in-loop `sdsc_execute` with IDENTICAL `symbol_ids` and double-reserve
/// them (DtException "Symbol already reserved", VariableDefinition.cpp:629; #53).
pub fn concrete_trips(e: &EmittedOp) -> Vec<SdscOp> {
    if e.time <= 1 {
        return vec![e.dsc().clone()];
    }
    (0..e.time)
        .map(|t| {
            let mut op = e.dsc().clone();
            for dsc_map in op.dscs_.iter_mut() {
                for dsc in dsc_map.values_mut() {
                    for node in dsc.scheduleTree_.iter_mut() {
                        let ti = node.ldsIdx_ as usize;
                        let Some(stride) = e.affine_strides.get(ti).and_then(|m| m.get("out"))
                        else {
                            continue; // non-tiled tensor — base address unchanged.
                        };
                        let bump = t as i64 * *stride;
                        for v in node.startAddressCoreCorelet_.data_.values_mut() {
                            let base: i64 = v.parse().unwrap_or(0);
                            *v = (base + bump).to_string();
                        }
                    }
                }
            }
            op
        })
        .collect()
}

/// The `bundle.mlir` orchestration `dxp_standalone --bundle -d <dir>` consumes:
/// one FLAT `sdscbundle.sdsc_execute () {sdsc_filename="sdsc_{i}.json"}` per
/// EXPANDED trip — concrete addresses baked into each json, NO `scf.for`, NO
/// symbols. A tiled op (time=N) contributes N consecutive flat executes (its
/// pre-unrolled trips from [`concrete_trips`]); a time=1 op contributes one. The
/// trip ordering is `ops` in order, trips `0..time` — IDENTICAL to [`write_bundle`]
/// so `sdsc_{i}.json` names line up. Delegates to [`bundle_mlir`].
pub fn emit_bundle_mlir(ops: &[EmittedOp]) -> String {
    let total: usize = ops.iter().map(|e| e.time.max(1) as usize).sum();
    let filenames: Vec<String> = (0..total).map(|i| format!("sdsc_{i}.json")).collect();
    bundle_mlir(&filenames)
}

/// Write a SuperDSC bundle directory: one `sdsc_{idx}.json` per op (serialized
/// `{ "{op_name}": SdscOp }`) + `bundle.mlir`. This is the artifact the
/// in-scratchy AoT bake feeds to `dxp_standalone --bundle -d <dir>` (driven by
/// `scr chat`, never a hand CLI — see the design memory). `ops` = [`EmittedOp`]
/// in topological order. Every op's HBM start addresses are SYMBOLIZED (see
/// [`symbolize_op`]) and the segment-base VALUES passed via `bundle.mlir`
/// symbol_ids so dxp's ModuleStitcher wires the multi-op dataflow.
/// Build the per-op manifest entries (idx/subdir/op_name/slot_stride_bytes + the `host_kv_write`
/// metadata) in the SAME idx/trip order [`write_bundle`] lays out the `op_{idx}/` subdirs. Factored
/// out so [`write_bundle_cached`] can refresh `op_manifest.json` UNCONDITIONALLY (even on a
/// device-artifact cache hit) — the manifest is a runtime-read artifact (like `bundle_layout.json`)
/// and MUST always reflect the current emitter, else a host-routing change (e.g. `host_kv_write`) is
/// silently dropped when the `sdsc_*.json` device ops are byte-identical (the cache reuses the old
/// dir, so the old manifest survives). Pure: no file IO; mirrors `write_bundle`'s trip iteration.
/// Per-trip [`GroupKind`] + owning-op index for the body's flat trip sequence.
/// `write_bundle` and `build_manifest_ops` BOTH call this so their grouping is
/// IDENTICAL (same partition, same subdir names). `host_kv_write`'s first trip
/// carries `HostKv{skip=kv_n_skip}` (kv_n_skip counts the on-card cachewr TRIPS
/// it replaces — the same unit the shim's `oi += kv_n_skip` advances over);
/// `slot_stride_bytes>0` → `Slot`; everything else `Pure` (fusable).
fn trip_kinds_and_owner(ops: &[EmittedOp]) -> (Vec<Trip>, Vec<usize>) {
    let mut kinds = Vec::new();
    let mut owner = Vec::new();
    for (oi, e) in ops.iter().enumerate() {
        let nt = (e.time.max(1)) as usize; // == concrete_trips(e).len()
        for ti in 0..nt {
            let k = if e.host_kv_write && ti == 0 {
                GroupKind::HostKv {
                    skip: e.kv_n_skip as usize,
                }
            } else if e.kv_page_fold {
                GroupKind::PageFold
            } else if e.slab_write {
                // Incremental Kᵀ restickify (kill-restickify Stage 2) → Slab (fusable, one
                // `slab_stride_bytes`). A DISTINCT kind from Slot: its 8192-byte slab stride ≠ the
                // cachewr's 128, so fusing them would trip the per-group uniform-stride assert.
                GroupKind::Slab
            } else if e.slot_stride_bytes > 0 {
                // Distinct per-slot cachewr (mq>1 prefill) → SlotSolo (singleton); uniform-slot
                // cachewr (decode) → Slot (fusable). A distinct-slot copy MUST NOT be fused into
                // a uniform-shift Slot group (would collapse the per-slot writes).
                if e.slot_no_fuse {
                    GroupKind::SlotSolo
                } else {
                    GroupKind::Slot { req: e.kv_request }
                }
            } else {
                GroupKind::Pure
            };
            kinds.push(Trip::new(k, TripRequest(e.kv_request)));
            owner.push(oi);
        }
    }
    (kinds, owner)
}

/// Whether the per-page fold gets a group of its own.
///
/// Splitting it costs ONE EXTRA LAUNCH PER LAYER, and this path is launch-bound: measured, the split
/// body is 5 groups where the baseline is 4, which is 40 more launches per token. That is only worth
/// paying when the fold is actually re-launched — i.e. when the context spans more than one page.
/// So the body is emitted BOTH ways and the runtime picks: `Fused` is the baseline's launch count
/// for any context that fits a page, `Split` is what makes longer contexts expressible at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FoldGrouping {
    /// Fold fused into the surrounding body — baseline launch count, single-page contexts only.
    Fused,
    /// Fold in its own re-launchable group — one extra launch per layer, any context length.
    Split,
}

// The FOLD-ROW REGIME a bundle declares travels as `bundle::FoldRows` on every `bundle::OpEntry`, and
// `bundle::BundleCode::fold_rows()` is its one reader — including the "fold groups disagree" refusal,
// since a session carries ONE intermediate stride.

/// Trip kinds under a fold grouping: `Fused` folds the per-page fold back into the ordinary work so
/// it merges with the neighbouring run. THE one place the two variants differ.
/// The ops a bundle variant actually contains. `Fused` serves single-page contexts, which have no
/// FULL pages at all, so their zero-masked fold is dropped outright — keeping it would add back the
/// very launches this variant exists to avoid.
pub fn ops_for_grouping(ops: &[EmittedOp], _fold: FoldGrouping) -> Vec<&EmittedOp> {
    ops.iter().collect()
}

fn trip_kinds_for(ops: &[EmittedOp], fold: FoldGrouping) -> (Vec<Trip>, Vec<usize>) {
    let (mut kinds, owner) = trip_kinds_and_owner(ops);
    if fold == FoldGrouping::Fused {
        for t in kinds.iter_mut() {
            if matches!(t.kind, GroupKind::PageFold) {
                // Reclassified, but it KEEPS ITS REQUEST — so it still cannot fuse across requests.
                t.kind = GroupKind::Pure;
            }
        }
    }
    (kinds, owner)
}

/// This op's own single `Dsc` (every `EmittedOp` wraps exactly one `dscs_` entry — `emit_sdsc`/
/// `emit_sdsc_tiled` always produce ONE `Dsc` keyed by `op_name`) and its primary `ComputeOp` (index
/// 0 — the matmul/pointwise itself; a fused epilogue's own 2nd entry is looked at separately by
/// `print_fusion_candidates`, which is exactly the case a scan for NEW candidates must not double
/// count as still-unfused).
fn primary_compute_op(e: &EmittedOp) -> Option<(&Dsc, &ComputeOp)> {
    let dsc = e.dsc().dscs_.first()?.values().next()?;
    let cop = dsc.computeOp_.first()?;
    Some((dsc, cop))
}

/// Every `opFuncName` the DDL's matmul family answers to (fp16/fp8/int8 × plain/batched — see the
/// dtype-suffix match in this file's own `computeOp_` construction). A closed list, not a substring
/// guess, so a future op named e.g. `"batchnormfwd"` is never mistaken for a matmul.
const MATMUL_OP_FUNCS: &[&str] = &[
    "matmul",
    "matmulfp8",
    "matmulint8",
    "batchmatmul",
    "batchmatmulfp8",
    "batchmatmulint8",
];

/// FUSION CANDIDATES, measured from the real emitted tape (`SCRATCHY_SDSC_FUSION_SCAN=1`).
/// For every adjacent `(ops[i], ops[i+1])` where `ops[i]` is a matmul,
/// reports whether `ops[i+1]` reads `ops[i]`'s own output buffer (a real producer/consumer link, via
/// `arg_bindings` — the SAME buffer-identity mechanism the runtime uses, not a name-string guess) and
/// whether it writes BACK to that same buffer (the in-place "epilogue" shape this codebase's fusion
/// mechanism can already express). For each of `ops[i+1]`'s OTHER inputs (the epilogue's would-be
/// extra operand), prints that buffer's own `scale_` from `labeledDs_` — a `-1`/non-1 entry is this
/// codebase's existing broadcast marker (see `Scale`), the signal that decides admissibility,
/// not an assumption made without looking.
fn print_fusion_candidates(ops: &[EmittedOp]) {
    let mut n = 0usize;
    for i in 0..ops.len().saturating_sub(1) {
        let (a, b) = (&ops[i], &ops[i + 1]);
        let Some((_, a_cop)) = primary_compute_op(a) else {
            continue;
        };
        if !MATMUL_OP_FUNCS.contains(&a_cop.opFuncName.as_str()) {
            continue;
        }
        let Some((b_dsc, b_cop)) = primary_compute_op(b) else {
            continue;
        };
        let a_outputs: Vec<&str> = a
            .arg_bindings
            .iter()
            .filter(|ab| !ab.is_input)
            .map(|ab| ab.buffer.as_str())
            .collect();
        let b_inputs: Vec<&str> = b
            .arg_bindings
            .iter()
            .filter(|ab| ab.is_input)
            .map(|ab| ab.buffer.as_str())
            .collect();
        let b_outputs: Vec<&str> = b
            .arg_bindings
            .iter()
            .filter(|ab| !ab.is_input)
            .map(|ab| ab.buffer.as_str())
            .collect();
        let Some(&shared) = a_outputs.iter().find(|o| b_inputs.contains(o)) else {
            continue;
        };
        let in_place = b_outputs.contains(&shared);
        n += 1;
        eprintln!(
            "[fusion-scan] {:>3}. {} (matmul:{}) -> {} ({}:{}) shares '{shared}', in_place={in_place}",
            n, a.op_name, a_cop.opFuncName, b.op_name, b_cop.exUnit, b_cop.opFuncName,
        );
        for extra in b_inputs.iter().filter(|&&x| x != shared) {
            // `labeledDs_`/`arg_bindings` are BOTH parallel to `op.args` (ldsIdx_ == position ==
            // arg_bindings index — see `emit_sdsc`'s `labeled.push` loop), so the extra operand's
            // OWN position in `b.arg_bindings` is its index into `b_dsc.labeledDs_` too. `dsName_`
            // there is the generic `"Tensor{i}"` placeholder, never the real buffer name, so
            // matching by NAME (as opposed to by shared position) would silently find nothing.
            let idx = b.arg_bindings.iter().position(|ab| ab.buffer == *extra);
            let scale = idx
                .and_then(|i| b_dsc.labeledDs_.get(i))
                .map(|l| l.scale_.clone());
            eprintln!("[fusion-scan]      extra operand '{extra}' scale_={scale:?}");
        }
    }
    eprintln!("[fusion-scan] {n} candidate(s) found");
}

/// THE PER-GROUP ADDRESS SHIFTS: one [`bundle::KvShifts`] per launch group, in launch order.
///
/// The partition here MUST match [`render_dxp_input`]'s exactly — shifts partitioned one way against
/// programs partitioned another shift the wrong program. Both call `trip_kinds_for` + `group_ranges`
/// with the same `group_size()`, and `group_ranges` is Kani-proven an exact covering partition; the two
/// are then ZIPPED into one [`bundle::LaunchGroup`] each, so the pairing exists at one site instead of
/// being an index the reader has to trust.
pub fn launch_index(ops: &[EmittedOp], fold: FoldGrouping) -> Vec<bundle::KvShifts> {
    // MEDIUM-GRAIN: one manifest entry per fusion GROUP (a contiguous run of ≤ g Pure
    // trips = one concrete dxp bundle; every host_kv_write/slot trip a singleton). g==1
    // reproduces the historical per-trip manifest exactly. The shim reads one entry →
    // one program; a pure group needs no per-entry handling, a host_kv/slot singleton
    // carries its metadata (the shim's `oi += kv_n_skip` then skips exactly the singleton
    // groups that hold the replaced copies).
    let (kinds, owner) = trip_kinds_for(ops, fold);
    let groups = group_ranges(&kinds, group_size());
    // LAUNCHES, not ops. A group IS a launch, and fusion means the two numbers are far apart — the
    // decode body is ~271 ops in 5 groups. An op-count gate therefore cannot see a fusion regression:
    // marking each batched cache write as its own singleton left the op count untouched (+16 per
    // request, as designed) while multiplying launches ~26x, which ran a batch of 8 twenty times
    // slower than a single request. Print it so the emit fingerprint can carry it.
    //
    // ⛔⛔⛔ AND YET AT bs=1 A LAUNCH IS ESSENTIALLY FREE — MEASURED 2026-08-16, granite-8b fp8, by
    // baking the SAME tree at two group sizes and interleaving them run-for-run:
    //   GROUP_SIZE=128 → 5 groups/layer = 200 launches/token → slow-mode ITL 82.2-83.0 ms
    //   GROUP_SIZE=512 → 3 groups/layer = 120 launches/token → slow-mode ITL 82.3-83.1 ms
    // +80 launches a token moved the slow mode by NOTHING, so the per-launch overhead is <= ~0.02 ms
    // and the 0.134 ms/launch that `SCRATCHY_SDSC_GROUP_TIME` reports at GROUP_SIZE=1 is almost all the
    // per-op DRAIN that timer needs, not launch cost.
    //
    // ⚖️ BOTH FACTS ARE TRUE AND THEY ARE ABOUT DIFFERENT THINGS: the bs=8 regression above was
    // launches that each carry a SLOT/PAGE SHIFT and serialize the fold, while these are Pure groups
    // over one row. So do not price a bs=1 launch reduction (merging the Slot/PageFold kinds, i.e. the
    // host-routing project) as an ITL win — at bs=1 the decode critical path is the WEIGHT STREAM
    // (8.477 GB/token at ~103 GB/s against IBM's own 150 GB/s measured peak), and op count and launch
    // count have now BOTH been measured not to move it.
    eprintln!(
        "[spyre-superdsc] GROUPS — {} launch group(s) for {} trip(s)",
        groups.len(),
        kinds.len(),
    );
    // WHAT THE OPS ACTUALLY ARE. A transformer layer is ~7 matmuls, 2 norms, RoPE and attention; the
    // decode body emits 271 at ONE row. Counting them by family says where that number comes from,
    // which no other diagnostic here reports — op_manifest.json has the names but is only written on
    // a forge run, so it cannot be read from a local bake.
    if std::env::var_os("SCRATCHY_SDSC_OPHIST").is_some() {
        let mut hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for e in ops.iter() {
            // Collapse the indices a family varies over (head, block, request, tensor id).
            let fam: String = e
                .op_name
                .chars()
                .map(|c| if c.is_ascii_digit() { '#' } else { c })
                .collect();
            let fam = fam.replace("##", "#").replace("##", "#").replace("##", "#");
            *hist.entry(fam).or_default() += 1;
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        eprintln!("[sdsc-ophist] {} ops:", ops.len());
        for (k, n) in v.iter().take(24) {
            eprintln!("[sdsc-ophist]   {n:5}  {k}");
        }
        // ⭐⭐⭐ TRIPS, NOT OPS — and the two are FAR apart, which is why every op-count reading of this
        // backend's cost has been wrong. An op with `time = t` streams its weight tile `t` times, so
        // TRIPS are what the weight stream is billed in: the decode body's 635 trips at granite-8b are
        // 4 launch groups, and one group holds 512 of them. MEASURED against the card:
        //   2b  248 trips/layer x 40 layers = 9,920/token, wstride 60.9 MB/layer => 246 KB/trip, ITL 26.0 ms
        //   8b  635 trips/layer x 40 layers = 25,400/token, wstride 211.9 MB/layer => 389 KB/trip, ITL 82.5 ms
        // i.e. 93.6 vs 102.7 GB/s of achieved weight bandwidth — the two models sit on ONE line, and
        // 8b is NOT over-split relative to 2b. Fitting the pair as `fixed + bytes/bw` gives ~227 GB/s
        // marginal and ~1.5 us FIXED per trip, so roughly half of a trip's time is overhead that a
        // BIGGER TILE would amortize. This histogram is where that per-family tile size is read off.
        let mut trips: std::collections::BTreeMap<String, (usize, u32)> = Default::default();
        for e in ops.iter() {
            let fam: String = e
                .op_name
                .chars()
                .map(|c| if c.is_ascii_digit() { '#' } else { c })
                .collect();
            let fam = fam.replace("##", "#").replace("##", "#").replace("##", "#");
            let slot = trips.entry(fam).or_default();
            slot.0 += 1;
            slot.1 += e.time;
        }
        let mut tv: Vec<_> = trips.into_iter().collect();
        tv.sort_by_key(|e| std::cmp::Reverse(e.1.1));
        let total: u32 = tv.iter().map(|(_, (_, t))| *t).sum();
        eprintln!(
            "[sdsc-trips] {total} trips over {} ops (trips = ops x time):",
            ops.len()
        );
        for (k, (n, t)) in tv.iter().take(24) {
            eprintln!(
                "[sdsc-trips]   {t:6} trips  {n:5} ops  time={:<4} {k}",
                *t as usize / n.max(&1)
            );
        }
    }
    // FUSION CANDIDATES, MEASURED, not guessed from reading `attn.rs`. Every prior candidate list
    // came from reading model-arch source; this walks the ACTUAL emitted
    // tape for whatever model/config is baking, generic across architectures (no attention-specific or
    // granite-specific logic below) — a `(matmul, next-op)` pair is a candidate iff they share a buffer
    // name via `arg_bindings` (the SAME mechanism the runtime uses to know which buffer is which), with
    // no assumption about which op families exist upstream.
    if std::env::var_os("SCRATCHY_SDSC_FUSION_SCAN").is_some() {
        print_fusion_candidates(ops);
    }
    // WHICH GROUP IS WHICH. The shim's per-group timer (`SCRATCHY_SDSC_GROUP_TIME`) reports by group
    // INDEX, because a launched group has no name on the runtime side — so its histogram is a list of
    // anonymous numbers unless the emitter says what index `gi` holds. It is the same index here and
    // there (both walk `group_ranges` output in order), so printing owner+extent+kind here is what
    // turns "op 19 is 24% of compute" into a statement about a specific matmul.
    if std::env::var_os("SCRATCHY_SDSC_OPHIST").is_some() {
        eprintln!(
            "[sdsc-groups] {} groups (index = the shim's `op N`):",
            groups.len()
        );
        for (gi, r) in groups.iter().enumerate() {
            let e = &ops[owner[r.start]];
            eprintln!(
                "[sdsc-groups]   op {gi:<3} {:3} trip(s)  req {}  {:?}  {}",
                r.end - r.start,
                kinds[r.start].req.0,
                kinds[r.start].kind,
                e.op_name,
            );
        }
    }
    let mut shifts: Vec<bundle::KvShifts> = Vec::new();
    for (gi, r) in groups.iter().enumerate() {
        // The group's first trip's owning op supplies name + any per-entry metadata.
        let e = &ops[owner[r.start]];
        // SLOT-group stride: the owner (first trip) carries it; a Slot group's shim slot-shift is
        // applied ONCE to the whole group, so EVERY op in the group MUST share one stride (else some
        // head lands at the wrong row). Guard it at build (an un-catchable `cargo build` panic — the
        // re-roll path swallows Err). Pure groups have stride 0 (their owner is a Pure op). HostKv
        // singleton stride is 0 (it carries kv_* metadata instead).
        let slot = if matches!(
            kinds[r.start].kind,
            GroupKind::Slot { .. } | GroupKind::SlotSolo
        ) {
            let stride = e.slot_stride_bytes;
            // A SlotSolo group is a singleton (r.len()==1) by construction, so the uniform-stride
            // check is trivially true; a fused Slot group must share one stride (uniform shift).
            for t in r.clone() {
                let o = &ops[owner[t]];
                assert!(
                    o.slot_stride_bytes == stride,
                    "fused Slot group {gi}: op '{}' stride {} != group stride {stride} — a fused \
                     Slot group's shim slot-shift is uniform, so all cachewr copies must share one \
                     stride (else a head writes the wrong cache row)",
                    o.op_name,
                    o.slot_stride_bytes
                );
                // AND one REQUEST. The shim resolves a group's page table and write cursor ONCE,
                // from the owner's `kv_request`, so two requests in one group would put one
                // request's token into the other's history — the one cache-write error no later
                // step can detect. `group_ranges` breaks a Slot run where the request changes, so
                // this cannot fire; it is here because the consequence is undetectable downstream.
                assert!(
                    o.kv_request == e.kv_request,
                    "fused Slot group {gi}: op '{}' names request {} but the group resolves request \
                     {} — a group is ONE launch and one page table, so a fused cachewr group must \
                     be single-request (else a request's token is written into another's history)",
                    o.op_name,
                    o.kv_request,
                    e.kv_request
                );
            }
            stride
        } else {
            0
        };
        // SLAB-group stride (kill-restickify Stage 2): mirror the Slot guard — the shim shifts the fused
        // Slab group's seg2 base by `(slot_pos/64)·slab_stride` ONCE, so every op in the group MUST share
        // one slab_stride (else a kv-head restickifies the wrong slab). Non-Slab groups: 0.
        let slab = if matches!(kinds[r.start].kind, GroupKind::Slab) {
            let stride = e.slab_stride_bytes;
            for t in r.clone() {
                let o = &ops[owner[t]];
                assert!(
                    o.slab_stride_bytes == stride,
                    "fused Slab group {gi}: op '{}' slab_stride {} != group stride {stride} — a fused \
                     Slab group's shim shift is uniform, so all restickifies must share one slab_stride \
                     (else a kv-head re-transposes the wrong slab)",
                    o.op_name,
                    o.slab_stride_bytes
                );
            }
            stride
        } else {
            0
        };
        let fold_group = matches!(kinds[r.start].kind, GroupKind::PageFold);
        let _ = gi; // launch order is the slice position now, not a field
        shifts.push(bundle::KvShifts {
            slot_stride_bytes: slot,
            slab_stride_bytes: slab,
            // PAGED: `page_slots` is the write-slot modulus AND the declaration that this bundle is
            // paged — a runtime that does not know the key must refuse rather than apply an
            // absolute-position shift and land past the page.
            page_slots: e.kv_page_slots,
            request: e.kv_request,
            // `page_fold` marks the group the runtime re-launches once per page.
            page_fold: fold_group,
            // ONE PASS PER PAGE INSTEAD OF PER (REQUEST, PAGE) — only when the fold's kernels actually
            // carry the request axis.
            batched_requests: fold_group && e.kv_batched_requests,
            fold_rows: match e.kv_fold_rows {
                scratchy_subtile::sdsc_abstract::FoldRowRegime::PerRequest => {
                    bundle::FoldRows::PerRequest
                }
                scratchy_subtile::sdsc_abstract::FoldRowRegime::WholeBatch => {
                    bundle::FoldRows::WholeBatch
                }
            },
        });
    }
    shifts
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  dxp's INPUT — the only files this emitter still writes, and they outlive one compile by nothing
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One launch group's dxp input, RENDERED but not yet anywhere.
///
/// dxp takes a DIRECTORY and requires one json per device op (merging a group's ops into one file is
/// `DtException: Expected empty FoldManager when importing from json`), so these files must exist —
/// but only between being written and being compiled. Rendering is separate from writing so ONE
/// renderer serves both consumers: the bake queue's staging dir, and [`write_dxp_input`] for feeding
/// dxp by hand.
pub struct GroupInput {
    /// Launch order — the `group` of the [`bundle::OpEntry`] that names this program.
    pub group: u32,
    /// `(file name, contents)`: one `sdsc_{i}.json` per trip, then `bundle.mlir` LAST.
    ///
    /// Last is not cosmetic — `bundle.mlir` completes the group, so writing it last is what makes the
    /// directory safe to compile at exactly one point.
    pub files: Vec<(String, String)>,
    /// Total bytes, so the staging budget's reservation is exact rather than an estimate.
    pub bytes: usize,
    /// ⭐ CONTENT KEY over exactly what dxp will read, in read order — the bake queue's memo key.
    /// Two groups with the same key compile to the same bytes, which 42% of them do.
    pub key: u64,
}

/// Render every launch group's dxp input.
///
/// MEDIUM-GRAIN FUSION (the torch-spyre per-kernel model, mirrored SDSC-native): the flat trip
/// sequence is partitioned into contiguous GROUPS (≤ `group_size()` Pure trips; each
/// host_kv_write/slot trip a singleton) and each group is ONE flat multi-op bundle — the same shape
/// dxp compiles for the whole model, just smaller, so it compiles to ONE program. dxp wires the
/// intra-group dataflow by shared byte address (the addresses are the SAME global placements,
/// grouping-invariant — producer-out == consumer-in by construction).
///
/// The partition here MUST match [`build_manifest_ops_grouped`]'s exactly: a manifest partitioned one
/// way against programs partitioned another launches a different set of groups than the index
/// describes, silently dropping whole groups of every layer. Both call `trip_kinds_for` +
/// `group_ranges` with the same `group_size()`, and `group_ranges` is Kani-proven an exact covering
/// partition (`group_ranges_is_exact_covering_partition`).
pub fn render_dxp_input(ops: &[EmittedOp], fold: FoldGrouping) -> std::io::Result<Vec<GroupInput>> {
    // Pre-unroll every op into its concrete trips and render one json per trip, in `ops`-order then
    // trip-order.
    let mut trip_json: Vec<String> = Vec::new();
    for (idx, (e, trip)) in ops
        .iter()
        .flat_map(|e| concrete_trips(e).into_iter().map(move |t| (e, t)))
        .enumerate()
    {
        // Top-level key carries the GLOBAL program-step index `{idx}_{op_name}` (mirroring
        // torch-spyre's `{idx}_{opfunc}`) so dxp's ModuleStitcher can order it — without it:
        // `DtException: expecting a valid entry in core schedule` (ModuleStitcher.cpp:216).
        let key = format!("{idx}_{}", e.op_name);
        let one: BTreeMap<&str, &SdscOp> = BTreeMap::from([(key.as_str(), &trip)]);
        // COMPACT, not pretty. Nothing reads these by eye — dxp parses them — and a 12B model emits
        // ~230k of them across the rung ladder. MEASURED on a real gemma-4 op: 32073 B pretty vs
        // ~9.7 KB compact, so indentation was ~2/3 of every byte written and ~2/3 of the per-file time.
        trip_json.push(
            serde_json::to_string(&one)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
    }
    let (kinds, _owner) = trip_kinds_for(ops, fold);
    let mut out = Vec::new();
    for (gi, r) in group_ranges(&kinds, group_size()).iter().enumerate() {
        let mut files: Vec<(String, String)> = Vec::with_capacity(r.end - r.start + 1);
        let mut names: Vec<String> = Vec::with_capacity(r.end - r.start);
        for (li, ti) in (r.start..r.end).enumerate() {
            let name = format!("sdsc_{li}.json");
            files.push((name.clone(), trip_json[ti].clone()));
            names.push(name);
        }
        // ⭐ THE KEY HASHES THE COMPILER'S INPUT, in the order `bundle.mlir` lists it — hashing
        // anything else would let two different inputs collide, and hashing the directory afterwards
        // would re-read what we just wrote.
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for j in &trip_json[r.start..r.end] {
                j.hash(&mut h);
            }
            h.finish()
        };
        files.push(("bundle.mlir".to_string(), bundle_mlir(&names)));
        out.push(GroupInput {
            group: gi as u32,
            bytes: files.iter().map(|(_, c)| c.len()).sum(),
            key,
            files,
        });
    }
    Ok(out)
}

/// Write the rendered dxp input into `dir/group_{gi}/` — the ARTIFACT DUMP.
///
/// Not part of the build (the bake queue stages its own copy and reclaims it): this is for handing a
/// bundle to `dxp_standalone --bundle -d <dir>` by hand, which is how a scheduler refusal gets
/// diagnosed and how the on-card gate bundles are produced.
pub fn write_dxp_input(
    dir: &std::path::Path,
    ops: &[EmittedOp],
    fold: FoldGrouping,
) -> std::io::Result<()> {
    for g in render_dxp_input(ops, fold)? {
        let gdir = dir.join(format!("group_{}", g.group));
        std::fs::create_dir_all(&gdir)?;
        for (name, contents) in &g.files {
            std::fs::write(gdir.join(name), contents)?;
        }
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Emitting a bundle
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ THE MEMORY PLAN, PROJECTED INTO THE BAKED ARTIFACT.
///
/// ⛔ THE EXHAUSTIVE DESTRUCTURE IS THE GUARD, and it is the whole reason this is one function. There
/// is no `..` below: adding a field to [`BundleLayout`] fails to compile HERE until it is either baked
/// or explicitly named as emit-only.
fn bake_layout(l: &BundleLayout) -> bundle::BundleLayout<'static> {
    let BundleLayout {
        ids,
        placements,
        segment_bytes,
        weight_bank_bytes,
        kernel_weights,
        scalarmul_scales,
        synth,
        // EMIT-ONLY, deliberately not baked: the arrangement authority is the build-time check that a
        // tensor has ONE device layout (`declare_arrangement` returns a build `Err` naming the tensor
        // otherwise). Its verdict is that the bundle compiled; there is nothing for the runtime to do
        // with the table.
        arrangements: _,
        kv_request_stride_bytes,
    } = l;

    let mut places: Vec<bundle::Placement> = placements
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
    // SYNTHETIC intermediates are not SubtileIR tensors, so they have no `t{id}` — they are named
    // directly and always live in the Intermediate segment.
    let synth = synth.borrow();
    places.extend(synth.map.iter().map(|(name, off)| {
        bundle::Placement {
            // ⛔ REFUSES rather than defaulting. Every synthetic is minted through `PlaceId::synth`,
            // which records the id; a name in `map` with no id means something allocated an
            // intermediate by SPELLING it, which is the round-trip this replaced. A fabricated id
            // here would place a real tensor under a wrong identity and the host would bind past it.
            id: *ids
                .borrow()
                .get(name)
                .unwrap_or_else(|| panic!("synthetic '{name}' was allocated without an identity")),
            segment: SegRole::Intermediate as u32,
            // Synthetics are intermediates; only the WEIGHT segment is ever banked.
            bank: 0,
            offset: *off,
            // ⛔ NO `unwrap_or(0)`. A zero-length field is read by the device as `1 << 27` flits —
            // 16 GiB — in both `handleHostDMA` and `handleXLATentry`. Every synthetic now has a size,
            // declared or reserved at first reference, so a missing one is a bug in the allocator and
            // says so.
            size: *synth
                .sizes
                .get(name)
                .unwrap_or_else(|| panic!("synthetic '{name}' has an offset but no reserved size")),
            is_logits: false,
        }
    }));

    // ⛔ THE INTERMEDIATE SEGMENT MUST COVER EVERY SYNTH, WHENEVER IT WAS DECLARED. The SubtileIR
    // walk grows this itself, right after its own node loop — but a synth can also be minted LATER,
    // by `ktir_to_superdsc::lower` running inside `emit_bundle_inner`, long after that walk returned
    // and its `segment_bytes` were final. Those intermediates then sat past the segment's end and
    // every one of them was an address defect (MEASURED: 1777 of them, `t0_kt0 spans
    // [6799360, 6803456) of seg0, which is only 6799360 B`).
    //
    // ⭐ SO THE GROW BELONGS HERE, not at either producer. This is the ONE place the final
    // `segment_bytes` are produced, and it already reads `synth` for the placements above — so
    // covering the allocator's high-water mark is a fact about the allocation rather than a step a
    // caller can forget. A no-op for the SubtileIR path, which already grew it.
    let mut segment_bytes = *segment_bytes;
    let seg_i = SegRole::Intermediate.segment();
    if synth.next > segment_bytes[seg_i] {
        segment_bytes[seg_i] = synth.next;
    }

    // ⭐ EVERY ADDRESS THIS BUNDLE WILL USE IS NOW DECIDED. Prove them before emitting.
    audit_layout_addresses(&places, &segment_bytes, weight_bank_bytes, "<layout>");

    bundle::BundleLayout {
        segment_bytes,
        weight_bank_bytes: std::borrow::Cow::Owned(weight_bank_bytes.clone()),
        places: std::borrow::Cow::Owned(places),
        kernel_weights: std::borrow::Cow::Owned(
            kernel_weights
                .iter()
                .map(|(tid, k)| bundle::KernelWeight {
                    id: bundle::PlaceId::Act(*tid),
                    device_size: std::borrow::Cow::Owned(k.device_size.clone()),
                    stride_map: std::borrow::Cow::Owned(k.stride_map.clone()),
                    stick_size: k.stick_size,
                    word_length: k.word_length,
                })
                .collect(),
        ),
        scalarmul_scales: std::borrow::Cow::Owned(scalarmul_scales.clone()),
        kv_request_stride_bytes: *kv_request_stride_bytes,
    }
}

/// One emitted bundle, awaiting the device code its groups are still being compiled into.
///
/// ⛔ WHY THE TWO HALVES ARE SEPARATE. The bake queue is deliberately ASYNCHRONOUS — group N of this
/// bundle compiles while the NEXT bundle is being written, which is what keeps `dxp_standalone` running
/// `COMPILE_WIDTH` wide across bundle boundaries instead of winding down to one at each of them. So the
/// metadata is finished here and the programs are collected by [`Self::into_code`], after the ONE drain
/// at the end of the emit.
pub struct EmittedBundle {
    /// Content fingerprint — this bundle's identity, and how a sibling names it.
    pub fp: String,
    /// The memory plan. Empty (`Default`) when the caller supplied no layout.
    pub layout: bundle::BundleLayout<'static>,
    pub groups: PendingGroups,
}

/// A bundle's launches, either already complete or waiting on the device compiler.
pub enum PendingGroups {
    /// ⭐ EVERY LAUNCH, IN LAUNCH ORDER, COMPLETE — the `-Fspyre-emu` shape.
    ///
    /// A launch group carries the PROGRAMS it runs, so there is nothing left to resolve later:
    /// no compiler to wait on, no id to look a device image up by. A constructed program is
    /// finished the moment it is constructed.
    Ready(Vec<bundle::LaunchGroup<'static>>),
    /// `-Fspyre-hw`'s shape: each group's SuperDSC directory is staged and submitted
    /// ([`ktir_groups_via_superdsc`]), and its `(KvShifts, GroupId)` is all `into_code` needs to
    /// find the compiled artifact once `superdsc_bake::finish_global()` has drained the queue —
    /// exactly the id `emit_bundle_inner` "used to" carry before this branch pivoted to KTIR.
    Staged(Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)>),
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
///   * `time`, the trip count, is `> 0`
///   * the func's own grid names at least one core, and — on `spyre-hw` only, where the limit
///     exists — no more than `MAX_CORES`
///   * every `scf.for` in the program runs a non-empty range with a positive step
fn audit_op_work(ops: &[EmittedOp], fp: &str) {
    let mut fail: Vec<String> = Vec::new();

    for op in ops {
        let name = &op.op_name;
        if op.time == 0 {
            fail.push(format!("{name}: time (trip count) is 0"));
        }
        // ⭐ THE PROGRAM'S OWN GRID IS THE CORE COUNT. `numCoresUsed_` was the descriptor's answer
        // to the same question — how many units run this op — and reading a core count off a
        // descriptor has no KTIR counterpart. The func STATES its division, so the guard asks it
        // there: zero cores is an op no unit will ever run, and more than the machine has is a
        // schedule the hardware cannot satisfy.
        let cores = op.ktir.as_ref().map_or(1, |k| {
            let (gx, gy, gz) = k.func.grid;
            (gx * gy * gz) as u32
        });
        if cores == 0 {
            fail.push(format!(
                "{name}: its grid names 0 cores — no unit will ever run this op"
            ));
        }
        // ⭐ `MAX_CORES` IS A CARD FACT — "Spyre dd2: 32 cores"
        // (`scratchy_subtile::superdsc_opspec::MAX_CORES`). A grid wider than the card is a
        // schedule no card can satisfy, so on hardware it is fatal here rather than at submit. The
        // emulator has no such ceiling: it runs the tile grid the func declares, which is how a
        // prefill of more than 32 tokens gets one core per token, so this is checked where the
        // limit exists and nowhere else.
        #[cfg(feature = "spyre-hw")]
        if cores > MAX_CORES {
            fail.push(format!(
                "{name}: its grid names {cores} cores but the card has {MAX_CORES}"
            ));
        }
        // ⭐ EVERY LOOP BOUND THE PROGRAM STATES. The sdsc/core/corelet fold factors were the
        // descriptor's loop counts; KTIR states its counts as the `scf.for` bounds themselves, so
        // the SAME law is checked where KTIR keeps it. The law is unchanged: a count must be
        // positive, because a zero-valued count field on this device reads as its MAXIMUM — the op
        // does not fail, it runs never or it runs forever, and either way the host waits for a
        // completion that is not coming.
        if let Some(k) = op.ktir.as_ref() {
            for (which, lo, hi, step) in loop_bounds(k.func.operations) {
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

/// ⭐ EMIT ONE BUNDLE: audit its ops, then build its launch index and memory plan.
///
/// ⛔ NOTHING PERSISTS AND NOTHING IS SUBMITTED — see [`emit_bundle_inner`]. The bundle is collected
/// in memory for the macro to bake, because a launch group now carries its own KTIR instead of an
/// id naming device code a compiler still owes us.
///
/// Returns the bundle's FINGERPRINT, which is its identity everywhere downstream: a sibling names
/// it, [`attach_reroll`] addresses it, and [`drain_emitted_bundles`] hands it to the macro. It is
/// content-addressed, so emitting the same ops at the same grouping twice (the ladder's ceiling
/// rung IS the body) does nothing the second time.
///
/// `attn_params` is [`RolledSuperDsc::attn_params`] — the four attention facts no KTIR states, which
/// main's own walk held as parameters. It is `None` for a bundle with no attention node.
pub fn emit_bundle(
    ops: &[EmittedOp],
    layout: Option<&BundleLayout>,
    fold: FoldGrouping,
    attn_params: Option<crate::ktir_superdsc_door::BundleAttnParams>,
) -> std::io::Result<String> {
    // ⭐ EVERY LOOP COUNT IN THESE OPS IS DECIDED. Prove they terminate and do work before emitting.
    audit_op_work(ops, "<bundle>");
    // Content-addressed: the same ops at the same grouping ARE the same bundle, so the second emit has
    // nothing to do. The ladder's ceiling rung IS the body, so this fires on every rolled model.
    let fp = bundle_fp(ops, fold);
    if emitted().lock().is_ok_and(|c| c.contains_key(&fp)) {
        return Ok(fp);
    }
    let b = emit_bundle_inner(ops, layout, fold, attn_params)?;
    debug_assert_eq!(
        b.fp, fp,
        "bundle_fp and emit_bundle_inner disagree on the fingerprint"
    );
    if let Ok(mut c) = emitted().lock() {
        c.entry(fp.clone()).or_insert((b, None));
    }
    Ok(fp)
}

/// This bundle's fingerprint — the SPLIT hash, with an `f` suffix marking the fused variant.
///
/// ⛔ THE FUSED TWIN'S NAME KEYS OFF THE UNFILTERED OPS. The runtime finds a body's fused twin as
/// `<split fp>f`, so the base hash must be the split body's, computed before `ops_for_grouping` drops
/// anything. Hashing the trimmed list is what made the twin unfindable.
fn bundle_fp(ops: &[EmittedOp], fold: FoldGrouping) -> String {
    let base = bundle_fingerprint_grouped(ops, FoldGrouping::Split);
    match fold {
        FoldGrouping::Split => base,
        FoldGrouping::Fused => format!("{base}f"),
    }
}

/// Attach the layer-loop parameters to the re-rolled BODY named by `fp`.
///
/// Separate from [`emit_bundle`] because a body's meta names its prefix, suffix, fused twin and every
/// ladder rung — none of which exist yet when the body itself is emitted.
pub fn attach_reroll(fp: &str, meta: bundle::RerollMeta<'static>) {
    if let Ok(mut c) = emitted().lock()
        && let Some(slot) = c.get_mut(fp)
    {
        slot.1 = Some(meta);
    }
}

/// EVERY BUNDLE THIS EMIT PRODUCED, by fingerprint — the channel the macro drains.
///
/// ⛔ A REGISTRATION, NOT A SEARCH. A bundle is in here because [`emit_bundle`] put it here, so
/// [`drain_emitted_bundles`] cannot silently return fewer bundles than were emitted.
#[allow(clippy::type_complexity)]
fn emitted()
-> &'static std::sync::Mutex<BTreeMap<String, (EmittedBundle, Option<bundle::RerollMeta<'static>>)>>
{
    static C: std::sync::OnceLock<
        std::sync::Mutex<BTreeMap<String, (EmittedBundle, Option<bundle::RerollMeta<'static>>)>>,
    > = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
}

/// Take every emitted bundle, complete with its compiled device code — what the macro bakes.
///
/// Call ONCE, after `superdsc_bake::finish_global()` has drained the compilers.
pub fn drain_emitted_bundles() -> Result<Vec<bundle::BundleCode<'static>>, String> {
    let taken = match emitted().lock() {
        Ok(mut c) => std::mem::take(&mut *c),
        Err(_) => return Err("the emit collector was poisoned by a panic".to_string()),
    };
    taken
        .into_values()
        .map(|(b, reroll)| b.into_code(reroll))
        .collect()
}

fn emit_bundle_inner(
    ops: &[EmittedOp],
    layout: Option<&BundleLayout>,
    fold: FoldGrouping,
    attn_params: Option<crate::ktir_superdsc_door::BundleAttnParams>,
) -> std::io::Result<EmittedBundle> {
    // ── GUARD #14 (a VERIFICATION since task #50, not a refusal of time-tiling): across an op's
    //    trips, no two writes to the SAME OUTPUT tensor may land on the same byte address. With real
    //    per-core HBM segment addressing each tiled tensor's trips advance from a real per-core base,
    //    so they cannot alias by construction — a residual stride bug would be silently wrong, so it
    //    is checked rather than assumed. ──
    if let Some(e) = ops.iter().find(|e| e.time > 1 && tiled_trips_alias(e)) {
        return Err(std::io::Error::other(format!(
            "[spyre-superdsc] time-tiled op '{}' (time={}) emits ALIASING per-trip OUTPUT addresses \
             — two trips would write the same HBM byte (silently-wrong). The #50 per-core stride must \
             advance each trip past the previous; this is an internal stride/segment bug. Refusing to \
             bake.",
            e.op_name, e.time
        )));
    }

    let fp = bundle_fp(ops, fold);
    let kept: Vec<&EmittedOp> = ops_for_grouping(ops, fold);
    let filtered: Vec<EmittedOp>;
    let ops: &[EmittedOp] = if kept.len() == ops.len() {
        ops
    } else {
        // Rebuild the trimmed list by value only when something was actually dropped.
        filtered = kept.into_iter().map(EmittedOp::shallow_copy).collect();
        &filtered
    };

    // ⭐ ONE SHARED KTIR, TWO CONSUMERS. `-Fspyre-emu` runs the group's own programs directly, so
    // the group IS the result — nothing is staged and nothing is submitted. `-Fspyre-hw` lowers
    // that SAME KTIR to a real SuperDSC directory and stages + submits it to the device compiler:
    // this is the write-files/reserve-disk/pair-with-an-id step `emit_bundle_inner` used to do
    // directly from SubtileIR, revived here over KTIR instead (`ktir_groups_via_superdsc`).
    let groups = if cfg!(feature = "spyre-hw") {
        PendingGroups::Staged(
            ktir_groups_via_superdsc(ops, &fp, fold, layout, attn_params)
                .map_err(|e| std::io::Error::other(e.0))?,
        )
    } else {
        PendingGroups::Ready(ktir_groups(ops, fold).map_err(|e| std::io::Error::other(e.0))?)
    };

    Ok(EmittedBundle {
        fp,
        layout: layout.map(bake_layout).unwrap_or_default(),
        groups,
    })
}

impl EmittedBundle {
    /// Collect this bundle's compiled programs and produce the value the macro bakes.
    ///
    /// Call AFTER `superdsc_bake::finish_global()`. A group the queue has no result for is an `Err`
    /// naming it: after a successful drain that can only mean it was never submitted, and a bundle with
    /// a hole in its launch sequence must not reach the binary.
    pub fn into_code(
        self,
        reroll: Option<bundle::RerollMeta<'static>>,
    ) -> Result<bundle::BundleCode<'static>, String> {
        let groups = match self.groups {
            PendingGroups::Ready(g) => g,
            PendingGroups::Staged(staged) => {
                let bake = crate::superdsc_bake::global();
                let mut out = Vec::with_capacity(staged.len());
                for (kv, id) in &staged {
                    // `bake` is `None` on a cardless build with `SCRATCHY_PLAN_ONLY_BAKE=1` (the
                    // one case `global()` doesn't panic for having no device compiler) —
                    // `ktir_groups_via_superdsc` staged/submitted nothing FOR that same reason,
                    // so there is nothing to have lost here either. Empty `init_binary` matches
                    // `ktir_groups`'s own plan-only default.
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
                        init_binary: std::borrow::Cow::Owned(init_binary),
                        job_bin_ptr,
                        correction: std::borrow::Cow::Owned(correction),
                    });
                }
                out
            }
        };
        Ok(bundle::BundleCode {
            fp: std::borrow::Cow::Owned(self.fp),
            layout: self.layout,
            groups: std::borrow::Cow::Owned(groups),
            reroll,
        })
    }
}

/// Deterministic content fingerprint of a SuperDSC bundle — its identity everywhere downstream, and
/// the bake queue's cue that a bundle it has already emitted is this one again.
///
/// Stable across builds (`DefaultHasher` has a fixed seed). Hashes the rendered `bundle.mlir` as well
/// as the per-op json, because the loop structure varies with `time`/strides while the byte-faithful
/// per-op json does not.
pub fn bundle_fingerprint(ops: &[EmittedOp]) -> String {
    bundle_fingerprint_grouped(ops, FoldGrouping::Split)
}

/// As [`bundle_fingerprint`], keyed by the fold grouping as well — the two variants share their ops and
/// differ only in group boundaries.
pub fn bundle_fingerprint_grouped(ops: &[EmittedOp], fold: FoldGrouping) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // MEDIUM-GRAIN: fold the group size in, since flipping G repartitions the body into different
    // launch groups and therefore different dxp programs.
    group_size().hash(&mut h);
    // ⛔ AND THE ACTUAL PARTITION, NOT JUST G. A `GroupKind` change (e.g. distinct-slot cachewr
    // Slot→SlotSolo) repartitions the body into a DIFFERENT set of groups even when every per-op json
    // and G are unchanged, so two bundles that launch differently would otherwise share a fingerprint —
    // and one would silently answer for the other.
    {
        let (kinds, _owner) = trip_kinds_for(ops, fold);
        for r in group_ranges(&kinds, group_size()) {
            (r.start, r.end).hash(&mut h);
        }
    }
    // ⭐ THE FINGERPRINT IS OVER WHAT GETS BAKED, AND WHAT GETS BAKED IS THE KTIR. Hashing the
    // per-trip SuperDSC descriptors said nothing about a program whose ops, tiling or grid changed
    // while its descriptors happened to agree. `Operation` hashes structurally — kind, operands,
    // attributes, result type, nested regions — so any edit to the emitted program is an edit to
    // the fingerprint, and a bundle can never silently answer for a different one.
    for e in ops {
        e.op_name.hash(&mut h);
        e.time.hash(&mut h);
        if let Some(k) = e.ktir.as_ref() {
            k.func.hash(&mut h);
            k.bindings.hash(&mut h);
        }
    }
    format!("{:016x}", h.finish())
}

/// Build-time aliasing check for a TIME-TILED op (#50 verification, GUARD #14):
/// expand the op's concrete trips and assert that no two trips write the SAME
/// HBM byte address for the SAME OUTPUT tensor (ldsIdx). Returns `true` if any
/// alias is found (→ a build `Err` in [`write_bundle_cached`]). A time=1 op never
/// aliases (one trip). Only OUTPUT/non-input HBM allocate nodes are checked —
/// inputs/LX legitimately repeat addresses (they are read, never written).
fn tiled_trips_alias(e: &EmittedOp) -> bool {
    if e.time <= 1 {
        return false;
    }
    // ldsIdx of every OUTPUT (write) tensor: the computeOp_ outputLabeledDs names
    // are `Tensor{i}-idx{i}`, so the output ldsIdx set is derivable, but the simpler
    // robust signal is the LabeledDs dsType_ == "OUTPUT". Collect those ldsIdx.
    //
    // A K-SPLIT (reduction/`in`-dim) matmul legitimately has MULTIPLE cores accumulate a partial
    // product into the SAME shared output address WITHIN one trip (dxp PSUM-accumulates them) —
    // the exact case the #50 disjoint-output guard already relaxes elsewhere (`reduction_split` at
    // line ~4082: `op.iter.split_of("in") > 1 && !v.layout.contains(&"in")`). Comparing addresses
    // GLOBALLY across every trip (the old `seen` set spanning all trips) flagged those intra-trip,
    // same-address accumulation writes as if they were a cross-TRIP collision — a false positive:
    // matmul_o325-class ops (m=31, K-split, time=2) legitimately repeat an address several times
    // within trip 0 (one per K-split core) and AGAIN within trip 1 (bumped by exactly one trip's
    // stride) — never actually reusing a byte a DIFFERENT trip already wrote. The real invariant is
    // per-trip: dedupe addresses WITHIN each trip first (collapsing the expected K-split repeats),
    // THEN check that address does not ALSO appear in a DIFFERENT trip's deduped set — that is the
    // only condition GUARD #14 is meant to catch (trip N re-writing a byte trip M already wrote).
    let trips = concrete_trips(e);
    let mut seen_by_trip: Vec<std::collections::BTreeSet<(u32, u64)>> =
        Vec::with_capacity(trips.len());
    for trip in &trips {
        let mut this_trip: std::collections::BTreeSet<(u32, u64)> =
            std::collections::BTreeSet::new();
        for dsc_map in &trip.dscs_ {
            for dsc in dsc_map.values() {
                let out_lds: std::collections::BTreeSet<u32> = dsc
                    .labeledDs_
                    .iter()
                    .filter(|l| l.dsType_ == "OUTPUT")
                    .map(|l| l.ldsIdx_)
                    .collect();
                for node in &dsc.scheduleTree_ {
                    if node.component_ != "hbm" || !out_lds.contains(&node.ldsIdx_) {
                        continue;
                    }
                    for v in node.startAddressCoreCorelet_.data_.values() {
                        if let Ok(a) = v.parse::<u64>() {
                            this_trip.insert((node.ldsIdx_, a));
                        }
                    }
                }
            }
        }
        seen_by_trip.push(this_trip);
    }
    for i in 0..seen_by_trip.len() {
        for j in (i + 1)..seen_by_trip.len() {
            if !seen_by_trip[i].is_disjoint(&seen_by_trip[j]) {
                return true; // trip j re-writes a byte trip i already wrote.
            }
        }
    }
    false
}

// ───────────────────────────────────────────────────────────────────────────
// The SubtileIR walk — the integration that turns the model's fused tape into a
// SuperDSC bundle. Peer of the sengraph lowering off the SAME `SubtileIR`, but
// tile→tile (SubtileIR is ALREADY tile-level) into the work-divided IR. Driven
// from `scr chat` via the AoT bake — never a hand CLI.
// ───────────────────────────────────────────────────────────────────────────

// `SuperDscError` now lives in the shared leaf module [`scratchy_subtile::superdsc_error`]
// so the typed core (`superdsc_opspec::WorkPlan::time_tile_for_lx`) and this wire
// module raise the SAME build-time error without a module cycle. Re-exported here
// so every existing `superdsc::SuperDscError` path keeps resolving.
pub use scratchy_subtile::superdsc_error::SuperDscError;

// ⛔ `ds_name` IS GONE, AND ITS ABSENCE IS THE ONE-PATH INVARIANT MADE VISIBLE. It was the stable
// per-tensor SuperDSC dataspace name taken from a `TensorRegion` (`t{id}`, collapsing the region), and
// the only thing that ever needed it was a body naming a SubtileIR region as a device operand — i.e.
// a node lowered straight to a descriptor. Every operand name now comes from
// `crate::wiring::act_name(tid)` off `KtirNode::args`, which is the same spelling; nothing in the
// producer names a device operand at all. If this function comes back, a side path came back with it.

/// One RE-ROLLED SuperDSC decode — the layer loop stays ROLLED (not 30× unrolled).
/// `prefix` = pre-loop ops (embed), `body` = ONE layer's ops (the executor runs it
/// `iters`×), `suffix` = post-loop ops (final norm + lm_head). `per_layer[t0]` =
/// `[t0, t1, …, t_{iters-1}]` — the layer-v tensor id for each tensor the body
/// references by layer-0's id `t0` (the node outputs + per-layer weights/KV); the
/// executor binds `per_layer[t][v]`'s (already-placed) address at iteration `v` — the
/// ivar weight/KV/hidden threading. dxp then compiles THREE SMALL bundles (seconds)
/// instead of one 2047-op unrolled monster (40 min). `layout` places EVERY layer's
/// tensors (the full resident set), so each `per_layer[t][v]` has a real address.
pub struct RolledSuperDsc {
    pub prefix: Vec<EmittedOp>,
    pub body: Vec<EmittedOp>,
    pub suffix: Vec<EmittedOp>,
    pub iters: u32,
    pub layout: BundleLayout,
    pub per_layer: std::collections::BTreeMap<u32, Vec<u32>>,
    /// Per-layer byte stride of the WEIGHT segment (seg1): the executor binds layer
    /// `v`'s weights by passing `seg1_base + v·weight_stride` (the body's baked
    /// layer-0 offsets shift to layer-v). UNIFORM across all per-layer weights (a
    /// build guard enforces it). 0 if no per-layer weights.
    pub weight_stride: u64,
    /// ⭐ LAYERS PER WEIGHT BANK — the divisor that turns layer `v` into `(bank, offset)`:
    /// `bank = v / layers_per_bank`, `offset = (v % layers_per_bank) · weight_stride`.
    ///
    /// READ OFF THE PLACEMENTS the banking pass wrote, never recomputed from the policy — the
    /// placements ARE the decision, and a second copy of `MAX_SEGMENT_BYTES / stride` here could
    /// disagree with the addresses that were actually baked. Equal to `iters` for an unbanked bundle,
    /// which makes the division a no-op and the launch sequence byte-identical.
    pub layers_per_bank: u32,
    /// The weight bank the PREFIX's weight operands live in, and the SUFFIX's — proven to be a single
    /// bank each (a launch has ONE base per segment). 0 when the group reads no weights.
    pub prefix_weight_bank: u32,
    pub suffix_weight_bank: u32,
    /// Per-layer byte stride of the KV segment (seg2), same contract. 0 if none.
    pub kv_stride: u64,
    /// Bytes between two REQUESTS' KV within one page+layer — the launch shifts seg2 by
    /// `request * this` on top of the layer and page terms. 0 if this bundle has no paged KV.
    /// Straight from [`BundleLayout::kv_request_stride_bytes`]; see there for why it is published
    /// rather than re-derived.
    pub kv_request_stride: u64,
    /// Requests one page holds — `PagedKvPool::ROWS`, so the runtime can refuse a pool row this bundle
    /// cannot address. 0 when there is no paged KV.
    pub kv_request_rows: u32,
    /// The body's residual-stream INPUT tensor id (the first body node's input[0] —
    /// the layer's hidden-in). OUTPUT id (the last body node's output — hidden-out).
    /// The executor threads `hidden_out → hidden_in` between iterations (the
    /// loop-carried residual). `u32::MAX` if the body is empty.
    pub hidden_in_tid: u32,
    pub hidden_out_tid: u32,
    /// The suffix's residual INPUT tensor id (the first suffix node's input[0], e.g.
    /// the last layer's post-attn residual t780). The rerolled body writes its output
    /// to `hidden_out_tid` (the representative-iteration tid t360), which differs from
    /// `suffix_in_tid`, so the executor copies `hidden_out → suffix_in` ONCE after the
    /// loop — the body→suffix seam (analogous to the per-iter `hidden_out → hidden_in`).
    /// `u32::MAX` if there is no suffix. When it equals `hidden_out_tid` the copy is a
    /// no-op (placements coincide, as the prefix→body seam does).
    pub suffix_in_tid: u32,
    /// ⭐ THE FOUR ATTENTION FACTS NO KTIR PROGRAM STATES — `ibm/main`'s own `lower_one_node`
    /// parameters, published from the walk that had them to the bundle emit that needs them.
    ///
    /// main lowered nodes DURING this walk, so `SubOp::AttnDecode`'s geometry and multiplier were in
    /// its hand and the rung and row kind were its own parameters. The KTIR split lowers one pass
    /// later, per bundle, so the facts ride here — beside `kv_request_stride` and for the same stated
    /// reason: it is the emitter's number travelling to the consumer rather than being re-derived
    /// there, where a second derivation would be free to disagree. `None` when the graph has no
    /// attention node. See [`crate::ktir_superdsc_door::BundleAttnParams`].
    ///
    /// ⛔ NOT A RECORD ON THE PROGRAM. `AttnFacts` was that, and it let the emulator and the card
    /// compute different attention; these are arguments of [`emit_bundle`], stated once per bundle.
    pub attn_params: Option<crate::ktir_superdsc_door::BundleAttnParams>,
}

/// ⭐⭐⭐ THE LAYER LOOP, UNROLLED INTO TENSOR IDS — the KTIR form of the executor's per-iteration
/// address bind.
///
/// ⛔ `RerollMeta`'s STRIDES HAVE NO KTIR COUNTERPART. `weight_stride` / `kv_stride` are byte
/// offsets into the resident weight and KV SEGMENTS: at iteration `v` the card's executor binds
/// `seg_base + v·stride` so one baked body addresses every layer. A KTIR launch binds no segment
/// base — it binds a TENSOR, and the emulator threads one buffer per tensor id — so there is no
/// offset to advance.
///
/// ⭐ BUT THE FACT ITSELF SURVIVES, BECAUSE IT WAS NEVER REALLY AN OFFSET. `per_layer[t0][v]` is
/// "the layer-`v` tensor that the body names by layer-0's `t0`", and the stride is only how the
/// card reaches it. So the loop unrolls by REBINDING: iteration `v` runs the same programs against
/// `per_layer[t][v]`. The layout already places every layer's tensors (see [`RolledSuperDsc`]), so
/// each rebound id has a real address.
///
/// ⭐ AND IT COSTS ALMOST NOTHING TO EMIT. A KTIR function is tensor-id-AGNOSTIC — its parameters
/// are `Ssa` indices and the tensor a parameter points at lives in `LaunchProgram::args` — so
/// `iters` copies of a body are `iters` copies of the ARGS over ONE shared program. That is why
/// unrolling here does not reintroduce the unrolled-bundle compile blowup the reroll was built to
/// avoid: the programs are interned once (`ktir_tokens::ProgramInterner`).
pub fn unroll_layers(rolled: &RolledSuperDsc) -> Vec<EmittedOp> {
    let mut ops = rolled.prefix.clone();
    // ⭐⭐ THE LOOP-CARRIED RESIDUAL IS **NOT** IN `per_layer`, AND IT IS NOT AN OVERSIGHT.
    // [`RolledSuperDsc::hidden_in_tid`] documents the card's arrangement: the executor threads
    // `hidden_out → hidden_in` between iterations and copies `hidden_out → suffix_in` after the
    // loop, because the two placements ALIAS. Aliasing is addressing, so it has no KTIR
    // counterpart — and its absence is not benign. Without this, every unrolled layer reads the
    // SAME hidden-in (the prefix's output) and only the last layer's write is ever read: the
    // layers do not chain, and the model returns one layer applied to the embedding.
    //
    // The KTIR form is a BINDING: iteration `v` reads iteration `v-1`'s hidden-out, iteration 0
    // reads the prefix's output, and the suffix reads the last iteration's.
    let per_layer_out: Option<&Vec<u32>> = rolled.per_layer.get(&rolled.hidden_out_tid);
    let hidden_of = |v: usize| -> Option<usize> {
        per_layer_out
            .and_then(|ids| ids.get(v))
            .map(|t| *t as usize)
    };
    for v in 0..rolled.iters as usize {
        for op in &rolled.body {
            let mut op = op.clone();
            if let Some(k) = op.ktir.as_mut() {
                for b in k.bindings.iter_mut() {
                    if b.get() == rolled.hidden_in_tid {
                        // Iteration 0's hidden-in is the prefix's own output; every later one is
                        // the previous iteration's hidden-out.
                        if v > 0
                            && let Some(prev) = hidden_of(v - 1)
                        {
                            *b = ktir_superdsc::ktir_node::BufferId::new(prev as u32);
                        }
                        continue;
                    }
                    // A tensor with no per-layer list is layer-INVARIANT (a shared constant, the
                    // mask, a source): the body names the one tensor there is, at every iteration.
                    if let Some(ids) = rolled.per_layer.get(&b.get())
                        && let Some(id) = ids.get(v)
                    {
                        *b = ktir_superdsc::ktir_node::BufferId::new(*id);
                    }
                }
            }
            ops.push(op);
        }
    }
    let last_hidden = rolled
        .iters
        .checked_sub(1)
        .and_then(|v| hidden_of(v as usize));
    for op in &rolled.suffix {
        let mut op = op.clone();
        if let Some(k) = op.ktir.as_mut()
            && let Some(last) = last_hidden
        {
            for b in k.bindings.iter_mut() {
                if b.get() == rolled.suffix_in_tid {
                    *b = ktir_superdsc::ktir_node::BufferId::new(last as u32);
                }
            }
        }
        ops.push(op);
    }
    ops
}

// NOTE — STILL TO MIRROR FROM THE FIXTURE (the dsc tile-schedule internals, the
// hard tail of the emitter): the per-`dscs_` entry fields T_/Tel_/P_/Pel_/B_/
// ChipD_/CoreD_/CoreletD_/loopOrder_/loopProperties_/dataStageParam_/
// scheduleTree_ (per-core HBM start addresses via affine folds)/primaryDsInfo_/
// labeledDs_ (memOrg_ hbm/lx/l0)/computeOp_/pdsRelation_/pcfg_/target_. These
// come from torch-spyre scheduler.py + scratchpad planning; emitted next,
// matmul-first (mirror sdsc_bmm_autoBuffer.json), then add/mul/silu/rmsnorm/
// softmax, then the SubtileIR walk. Each on-card dxp failure during the in-
// scratchy AoT bake becomes a build-time guard (guard_superdsc_crash_patterns).

#[cfg(test)]
mod tests {
    // `KernelTag` names a WEIGHT-tile stick layout, which only the SuperDSC assemblers take — and
    // after the retarget those assemblers are reached from tests alone, so the import belongs here
    // rather than in the lowering's own prelude.
    use scratchy_subtile::sdsc_abstract::{KernelTag, Stk};
    // ⭐ SAME REASON, WIDER: the whole SuperDSC EMITTER is now `ktir_superdsc::emit`, so every name
    // below is reached from these tests alone and none of them belongs in the lowering's prelude.
    // The tests stayed here deliberately — they pin laws of the emitter as this crate CALLS it (the
    // bundle layout it hands in, the folds it asks for), and moving them would have moved two test
    // rosters at once.
    use ktir_superdsc::emit::{emit_sdsc, rb, scaling_factor_const_fp32, sen169_bits};
    use scratchy_subtile::superdsc_opspec::{
        ItDim, MaxCores, OpFunc, OpInfo, Scale, SdscFoldSet, StickExtent, WorkPlan,
    };

    /// ⭐⭐⭐ AN INDEX PAST ITS REGION IS A REFUSAL, NOT AN ALIAS. Every reserved tid used to be
    /// raw `u32` subtraction from a base — `SCALARMUL_SCALE_BASE - idx`, unchecked — so the 100th
    /// scale silently took `ksplit_block_tid(0)`'s slot. Two tensors, one tid, one placement: the
    /// second producer overwrites the first's bytes and the first's consumer reads them. On a
    /// device with no stack to attach to that is wrong output or a hang, reported by nothing.
    #[test]
    #[should_panic(expected = "reserved tid region overflow")]
    fn an_index_past_its_region_refuses_instead_of_aliasing_the_next_one() {
        // scalarmul owns 100 slots; the 101st is past its floor.
        let _ = scalarmul_scale_tid(RESERVED_REGIONS[1].slots as usize);
    }

    /// The regions tile the space downward without gaps between the ones that abut, and every
    /// declared base IS its region's base — the constants are derived from the table, so there is
    /// no second copy to drift.
    #[test]
    fn the_declared_bases_are_the_regions_bases() {
        assert_eq!(SCALARMUL_SCALE_BASE, RESERVED_REGIONS[1].base);
        assert_eq!(KCT_RESIDENT_BASE, RESERVED_REGIONS[2].base);
        // Each region sits strictly below the one before it. They need NOT abut: the K-split
        // regions were removed and their span is deliberately left as a hole, because a reserved
        // id is a number that has been baked into artifacts and sliding one up to close a gap
        // would silently repoint it.
        for w in RESERVED_REGIONS.windows(2) {
            assert!(
                w[1].base < w[0].floor(),
                "{} overlaps {} or sits above it",
                w[1].name,
                w[0].name
            );
        }
    }

    /// The last slot of each region is still ITS OWN, and the first slot of the next is not.
    #[test]
    fn a_regions_last_slot_belongs_to_it_and_the_next_tid_does_not() {
        for r in RESERVED_REGIONS {
            assert_eq!(r.at(r.slots - 1), r.floor(), "{}", r.name);
        }
        assert_eq!(scalarmul_scale_tid(99), RESERVED_REGIONS[1].floor());
    }

    /// ⭐⭐⭐ THE ARGUMENT ORDER OF `for_output` IS LOAD-BEARING, AND THIS IS THE MODEL THAT PROVES
    /// IT. granite-3.x-2b: lm_head is `[n=49155, k=2048]` on disk. Padding the OUTPUT axis `n` is
    /// the whole point — 49155 rounds to 769 sticks, which is PRIME, so the full-occupancy bump
    /// takes it to 800 sticks = 51200. Padding `k` instead is a NO-OP, because 2048 is already 32
    /// sticks and fills the machine.
    ///
    /// So a caller that passes `(k, n)` where `(n, k)` is meant gets a PLAUSIBLE answer — the
    /// weight's own contraction width, unpadded — and stages 49155 columns into a placement sized
    /// for 51200. The executor catches it as
    /// `host size 201338880 B != prod(device_size)*word_length (209715200 B)`, which is exactly
    /// `2048 * 49155 * 2` against `2048 * 51200 * 2`. That is a real failure this repo shipped:
    /// the staged `[rows, cols]` pair moved from a JSON manifest (which recorded the LOGICAL
    /// `[k, n]`) to `BoundWeight::staged_shape` (which returned the ON-DISK `[n, k]`) under the
    /// same field names, so every downstream use read `n` where it meant `k`.
    #[test]
    fn the_output_axis_is_what_gets_padded_and_swapping_k_and_n_is_silent() {
        const N: u32 = 49155; // granite vocab
        const K: u32 = 2048; // granite hidden
        let right = DeviceWidth::for_output(1, N, K).get();
        assert_eq!(right, 51200, "the output axis pads to full occupancy");
        assert_eq!(right % 64, 0);
        assert_eq!(
            right / 64 % MAX_CORES,
            0,
            "800 sticks = 25 per core on 32 cores"
        );

        // The swapped call is not an ERROR — it is a different, plausible number.
        let swapped = DeviceWidth::for_output(1, K, N).get();
        assert_eq!(
            swapped, K,
            "k is already core-splittable, so padding it is a no-op"
        );
        assert_ne!(swapped, right);

        // And the byte sizes are the two in the failure message: the host staged the weight
        // UNPADDED (`n * k`, because the no-op bump left the width alone) into a placement the
        // emitter had sized at the padded width (`k * n_dev`).
        assert_eq!(
            N as usize * K as usize * 2,
            201_338_880,
            "what the host staged"
        );
        assert_eq!(
            K as usize * right as usize * 2,
            209_715_200,
            "what the device wanted"
        );
    }
    use super::*;
    use scratchy_subtile::lower::GemmWeight;
    // ⛔ THIS WHOLE MODULE WAS DEAD. `matmul_opspec`/`matmul_dims`/`matmul_split_map`/
    // `reduce_opspec_df` moved out of this file into `ir::bridge::tiled_op_sdsc_op::{matmul,reduce}`
    // and the `use super::*` no longer reached them, so `cargo test -p scratchy-subtile --lib` failed
    // to COMPILE — which means every assertion below has been reporting nothing, including the three
    // fused-epilogue tests, while two epilogue fusions were built on that mechanism.
    //
    // ⭐ AN UNCOMPILABLE TEST TARGET IS INDISTINGUISHABLE FROM A PASSING ONE unless you read past the
    // integration-test results, and this crate's `tests/` directory is green — so `cargo test -p …`
    // printed dozens of `ok` lines with this target's `error:` above them. Same shape as the
    // `#![cfg(kani)]` proofs that were vacuous for ~100 commits.
    use crate::ir::bridge::tiled_op_sdsc_op::matmul::dims::{matmul_dims, matmul_split_map};
    use crate::ir::bridge::tiled_op_sdsc_op::matmul::opspec::matmul_opspec;
    use crate::ir::bridge::tiled_op_sdsc_op::reduce::reduce_opspec_df;

    /// Guard the SEN169_FP16 (1-6-9) encoding the reduce `scaling_factor` const needs.
    /// The bit patterns are pinned against the SFP-constant table (`plus1=0x3E00`,
    /// `minus1=0xBE00`, `fastSigmoidConst=0x3C00`=0.5) — those are SEN169, NOT IEEE f16.
    /// A regression here re-introduces the silent reduce mis-scale (mean 4600× too small,
    /// sum halved) that produced inf attention scores / garbage output on-card.
    #[test]
    fn sen169_encoding_pinned_to_sfp_table() {
        assert_eq!(sen169_bits(1.0), 0x3E00, "SEN169 1.0 (cf. SFP plus1)");
        assert_eq!(sen169_bits(-1.0), 0xBE00, "SEN169 -1.0 (cf. SFP minus1)");
        assert_eq!(
            sen169_bits(0.5),
            0x3C00,
            "SEN169 0.5 (cf. SFP fastSigmoidConst)"
        );
        assert_eq!(sen169_bits(0.0), 0x0000, "SEN169 +0");
        // The reduce scale that was the bug: must NOT equal the IEEE f16 bits.
        let inv576 = 1.0f32 / 576.0;
        assert_ne!(
            sen169_bits(inv576),
            half::f16::from_f32(inv576).to_bits(),
            "1/576 SEN169 must differ from IEEE f16 (the on-card mis-read)"
        );
        // Decode-round-trip within fp16 precision (1-6-9, bias 31).
        for v in [1.0f32 / 576.0, 1.0 / 9.0, 1.0 / 256.0, 1.0 / 2048.0] {
            let b = sen169_bits(v);
            let exp = ((b >> 9) & 0x3F) as i32 - 31;
            let mant = (b & 0x1FF) as f32 / 512.0;
            let decoded = (1.0 + mant) * 2f32.powi(exp);
            assert!(
                (decoded - v).abs() / v < 0.005,
                "SEN169 round-trip {v} → {decoded}"
            );
        }
    }

    /// The fp32 reduce (mq>1 rmsnorm mean(x²)) must encode its `scaling_factor`
    /// external const in the OP's data_format — IEEE_FP32 with the RAW f32 bits,
    /// NOT a SEN169_FP16 word. A fp16 const feeding an fp32 op is a mixed
    /// [fp16,fp32] op → DD2 "Unsupported result precision conversion"
    /// (SentientToProgIR/Utils.cpp:34). Mirrors torch-spyre `encodeConstant` →
    /// `BinaryConvert<uint32_t>(float)` for IEEE_FP32 (module.cpp:126). The fp16
    /// reduce path (`sen169_encoding_pinned_to_sfp_table`) is unchanged.
    #[test]
    fn fp32_reduce_scale_const_is_ieee_fp32() {
        // fp32 reduce → ReduceScalingFp32 with raw f32 bits of 1/N.
        let f32_spec = reduce_opspec_df(OpFunc::Mean, 64, 576, "r_x", "r_acc", true, Df::Fp32)
            .expect("fp32 mean reduce opspec");
        assert_eq!(
            f32_spec.op_info,
            OpInfo::ReduceScalingFp32((1.0f32 / 576.0).to_bits()),
            "fp32 reduce const must be raw IEEE f32 bits of 1/N"
        );
        // The serialized const carries dataFormat_ = IEEE_FP32 and the 32-bit word.
        let ci = scaling_factor_const_fp32((1.0f32 / 576.0).to_bits());
        assert_eq!(ci["0"]["dataFormat_"], "IEEE_FP32");
        let got = ci["0"]["data_"][0].as_u64().unwrap();
        assert_eq!(got, (1.0f32 / 576.0).to_bits() as u64);
        assert!(
            got > 0xFFFF,
            "a true fp32 word exceeds 16 bits; got {got:#x}"
        );

        // fp16 reduce path is untouched: still SEN169_FP16, 16-bit word.
        let f16_spec = reduce_opspec_df(OpFunc::Mean, 64, 576, "r_x", "r_acc", true, Df::Fp16)
            .expect("fp16 mean reduce opspec");
        assert_eq!(
            f16_spec.op_info,
            OpInfo::ReduceScaling(sen169_bits(1.0f32 / 576.0) as u32),
            "fp16 reduce const must stay SEN169_FP16"
        );
    }

    #[test]
    fn core_split_matches_fixture() {
        assert_eq!(core_split(384, 32), 32); // 384=2^7·3 → 32 (÷12)
        assert_eq!(core_split(320, 32), 32); // 320=2^6·5 → 32 (÷10)
        assert_eq!(core_split(64, 32), 32);
        assert_eq!(core_split(320, 9), 8); // not ÷9; largest divisor ≤9 is 8
        assert_eq!(core_split(48, 32), 24); // not ÷32; 48=16·3, largest ≤32 is 24
        assert_eq!(core_split(7, 32), 7); // prime ≤ max
        assert_eq!(core_split(13, 32), 13);
    }

    #[test]
    fn distribute_beats_single_core() {
        // The bmm case: out=384, mb=384 (outputs), in=64 (reduction).
        let dims = vec![
            ItDim {
                name: "mb",
                size: 384,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "out",
                size: 384,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            },
            ItDim {
                name: "in",
                size: 64,
                is_reduction: true,
                is_stick: true,
                df: Df::Fp16,
            },
        ];
        let splits = distribute_cores(&dims, MAX_CORES);
        let cores: u32 = splits.values().product::<u32>().max(1);
        // Must use all 32 cores — the entire point (sengraph auto-split wastes 31).
        assert_eq!(
            cores, 32,
            "work-division must fill all 32 cores: {splits:?}"
        );
        // And the validated WorkPlan accepts it (≤32 by type).
        let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores).unwrap();
        assert_eq!(plan.cores_used().get(), 32);
    }

    #[test]
    fn bundle_layout_roles_and_packing() {
        // GLOBAL layout (task #55): weight source → seg1, activation source → seg0,
        // result → seg4 (logits), all 128 B aligned. A two-op chain produces one
        // seg3 intermediate; with a third op consuming nothing of the first, the
        // freed slot is REUSED (lifetime coloring).
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        let tensors = vec![
            TensorShape { rows: 4, cols: 8 },  // t0 = activation source
            TensorShape { rows: 8, cols: 16 }, // t1 = weight source
            TensorShape { rows: 4, cols: 16 }, // t2 = result (logits)
        ];
        let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
            tensor: TensorId::from_index(t as usize),
            region: ts[t].whole(),
        };
        let nodes = vec![SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::MatmulTile {
                weight: GemmWeight::Dense,
            },
            inputs: vec![whole(0, &tensors), whole(1, &tensors)],
            output: whole(2, &tensors),
        }];
        let ir: SubtileIR = SubtileIR {
            tensors,
            num_sources: 2,
            nodes,
            result: TensorId::from_index(2),
            // Hand-authored fixture: there is no source op list to be the
            // provenance of, so the map is empty.
            op_output: Vec::new(),
        };
        let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        let layout = compute_bundle_layout(&ir, &weight_ids, false, &Default::default())
            .expect("a layout for a plain matmul bundle");

        // ⭐ ROLES AND PACKING, ASSERTED AS THE RULES — not as magic numbers. This test pinned
        // `(Activation, seg 0, off 0, 64 B)` and both of those had since changed BY DESIGN, silently,
        // because the target would not compile:
        //   · seg 0 → 3: the dated `Activation = 3` / `Intermediate = 0` diagnostic swap (see `SegRole`),
        //     so a segment INDEX literal here restates a decision that already moved once.
        //   · 64 B → 4096 B: `nbytes` reserves the DEVICE footprint, and
        //     `bump_sticks_to_splittable` pads a 1-stick width up to 8 sticks so the work division can
        //     split it — 4 rows x 512 elems x 2 B.
        // So the segment comes from the role itself, and the size is checked against the invariants the
        // rule guarantees (at least the flat footprint, a whole number of sticks) rather than one
        // padding policy's current output. A role mix-up or an aliasing regression still fails; a
        // deliberate re-tune of the padding no longer reports a defect that is not there.
        let flat = |rows: u64, cols: u64| rows * cols * 2;
        for (tid, role, rows, cols) in [
            (0u32, SegRole::Activation, 4u64, 8u64),
            (1, SegRole::Weight, 8, 16),
            (2, SegRole::Logits, 4, 16),
        ] {
            let p = layout.placements[&tid];
            assert_eq!(p.role, role, "t{tid}'s role");
            assert_eq!(
                p.segment,
                role.segment(),
                "t{tid} lands in ITS ROLE's segment"
            );
            assert_eq!(
                p.offset, 0,
                "t{tid} is the only tensor in that segment, so it packs at 0"
            );
            assert!(
                p.size >= flat(rows, cols),
                "t{tid}'s reservation ({}) covers its flat footprint ({})",
                p.size,
                flat(rows, cols)
            );
            assert_eq!(
                p.size % (64 * 2),
                0,
                "t{tid}'s reservation is whole 64-elem fp16 sticks"
            );
            assert_eq!(
                layout.segment_bytes[p.segment],
                align128(p.offset + p.size),
                "segment {} is sized to its packed contents, 128 B aligned",
                p.segment
            );
        }
        // No intermediates in a single-op bundle — whichever segment that role currently owns.
        assert_eq!(layout.segment_bytes[SegRole::Intermediate.segment()], 0);

        // No two placements in the SAME segment overlap (build-time safety twin).
        let mut by_seg: std::collections::BTreeMap<usize, Vec<(u64, u64)>> = Default::default();
        for p in layout.placements.values() {
            by_seg
                .entry(p.segment)
                .or_default()
                .push((p.offset, p.size));
        }
        for ranges in by_seg.values() {
            for (i, &(o1, s1)) in ranges.iter().enumerate() {
                for &(o2, s2) in &ranges[i + 1..] {
                    assert!(
                        o1 + s1 <= o2 || o2 + s2 <= o1,
                        "segment alias {o1}+{s1} vs {o2}+{s2}"
                    );
                }
            }
        }
    }

    #[test]
    fn emit_sdsc_matmul_minimal_fields() {
        // The matmul OpSpec lowers to the frontend-minimal field set.
        let op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").unwrap();
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
        let j = serde_json::to_string(&sdsc).unwrap();
        // Frontend-minimal memOrg: hbm+lx only, NO register file.
        assert!(j.contains("\"memOrg_\":{\"hbm\":{\"isPresent\":1},\"lx\":{\"isPresent\":1}}"));
        assert!(!j.contains("pelrf") && !j.contains("ptxrf"));
        // startAddr data keys carry SPACES "[c, 0, 0]".
        assert!(j.contains("[0, 0, 0]"));
        // Dropped scheduler fields are absent.
        assert!(!j.contains("gtrIdsUsed_") && !j.contains("pdsRelation_"));
        assert!(!j.contains("hbmStartAddress_") && !j.contains("lxBufferSize_"));
        assert!(!j.contains("stickRepl_") && !j.contains("unpadN_"));
        // exUnit is pt (sealed from OpFunc), and the fold factors agree.
        assert_eq!(sdsc.dscs_[0]["MatMul_0"].computeOp_[0].exUnit, "pt");
        assert_eq!(sdsc.coreFoldProp_.factor_, 32);
        assert_eq!(sdsc.coreletFoldProp_.factor_, 1);
    }

    #[test]
    fn emit_sdsc_matmul_fused_epilogue_broadcast_batch_scale_correct() {
        // The bug this test exists to pin: the pmask fusion's FIRST landing marked only "mb"
        // broadcast and forgot "y" (the GQA-group batch axis a batched score matmul ALSO needs
        // broadcast for a head-independent mask) — `set_scale_for_dim` silently no-op'd on the
        // missing marker at the time, so the emitter built a plausible-looking WRONG SdscOp with no
        // signal, and garbled generation on the card was the first anyone noticed. `broadcast_batch`
        // exists so the CALLER never has to name "y" (or re-derive whether it exists) at all — this
        // asserts a batched (`batch>1`, so `batch_dim_name()` returns `Some("y")`) matmul's fused
        // epilogue operand shows RedNonStick on BOTH "mb" and "y", Active on "out".
        let mut op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").unwrap();
        assert_eq!(op.time(), 1);
        assert_eq!(
            op.batch_dim_name(),
            Some("y"),
            "a batch=16 matmul must carry a real y dim"
        );
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
            scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[("mb", scratchy_subtile::superdsc_opspec::Scale::RedNonStick)],
                broadcast_batch: true,
            },
        ));
        // The epilogue operand is the one inserted BEFORE the real output (see
        // `attach_fused_epilogue`'s own doc) — for this 3-arg-input matmul (a,w,o) that is index 2.
        let epi_idx = 2;
        let view = op.args[epi_idx].view();
        // ⭐ ASSERTED BY DIM NAME, NOT BY POSITION. This test asserted `layout == ["mb","out","y"]` and
        // a scale array indexed against it; the emitter's order is `["mb","y","out"]` (the
        // on-hardware-proven batch-inner walk `batched_decode_walk_order.rs` pins), so the positional
        // form went stale the moment the walk work landed — and, because this whole test target failed
        // to COMPILE, said nothing about it for as long as it was wrong. The SUBJECT is which dims are
        // broadcast, which does not depend on their order, so it is now stated that way.
        let scale_of = |d: &str| {
            view.layout
                .iter()
                .position(|&l| l == d)
                .map(|i| view.scale[i])
                .unwrap_or_else(|| panic!("epilogue operand has no `{d}` dim: {:?}", view.layout))
        };
        assert_eq!(scale_of("mb"), Scale::RedNonStick, "mb must be broadcast");
        assert_eq!(
            scale_of("y"),
            Scale::RedNonStick,
            "the GQA-group batch axis must be broadcast too"
        );
        assert_eq!(scale_of("out"), Scale::Active, "out must stay Active");

        // And the FULL emit succeeds (would have panicked pre-fix if "y" were absent from this
        // shape's layout, or produced a silently-wrong scale_ if the marker were dropped).
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
        let labeled = &sdsc.dscs_[0]["MatMul_0"].labeledDs_;
        let mask_lds = labeled
            .iter()
            .find(|l| l.dsName_ == format!("Tensor{epi_idx}"))
            .expect("mask labeledDs_ entry");
        // Wire `scale_` is parallel to the operand's own layout, so it is read the same way: -1 for a
        // broadcast dim, 1 for an active one. Two broadcast dims and one active, whatever the order.
        let wire = |d: &str| {
            view.layout
                .iter()
                .position(|&l| l == d)
                .map(|i| mask_lds.scale_[i])
                .expect("dim")
        };
        assert_eq!(wire("mb"), -1, "wire scale_ mb: {:?}", mask_lds.scale_);
        assert_eq!(wire("y"), -1, "wire scale_ y: {:?}", mask_lds.scale_);
        assert_eq!(wire("out"), 1, "wire scale_ out: {:?}", mask_lds.scale_);
    }

    #[test]
    fn emit_sdsc_matmul_fused_epilogue_broadcast_batch_is_noop_when_unbatched() {
        // The other half of the SAME fix: `broadcast_batch=true` on an UNBATCHED (batch==1) matmul
        // must be a genuine no-op, never a panic — `matmul_dims` omits "y" entirely at batch==1, and
        // `batch_dim_name()` reporting `None` there is exactly what lets `attach_fused_epilogue` skip
        // marking it instead of reaching for a dim that does not exist (the caller-side bug class
        // this mechanism replaces: attn.rs no longer computes "is this op batched" itself to decide
        // whether "y" is safe to name).
        let mut op = matmul_opspec(384, 384, 64, 1, "Tensor0", "Tensor1", "Tensor2").unwrap();
        assert_eq!(
            op.batch_dim_name(),
            None,
            "a batch=1 matmul must carry no y dim at all"
        );
        // broadcast_batch: true must NOT panic despite there being no "y" to mark.
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
            scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[("mb", scratchy_subtile::superdsc_opspec::Scale::RedNonStick)],
                broadcast_batch: true,
            },
        ));
        let epi_idx = 2;
        let view = op.args[epi_idx].view();
        assert_eq!(
            view.layout,
            ["mb", "out"],
            "unbatched output stays rank-2: {:?}",
            view.layout
        );
        assert_eq!(view.scale, [Scale::RedNonStick, Scale::Active]);
    }

    #[test]
    fn emit_sdsc_matmul_fused_epilogue_matches_golden_shape() {
        // Mirrors `ddc/ddl_templates/test/sdsc_bmm_lxopt.json`'s `MatMul_122`: computeOp_ is a
        // 2-element array — the matmul (unchanged inputLabeledDs, no mention of the epilogue operand),
        // then a second entry whose inputLabeledDs/outputLabeledDs both alias the MATMUL'S OWN output
        // by name (in place), with the epilogue's extra tensor as its second input.
        let mut op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").unwrap();
        assert_eq!(
            op.time(),
            1,
            "test assumes an untiled matmul (attach_fused_epilogue's precondition)"
        );
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
            scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[],
                broadcast_batch: false,
            },
        ));
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
        let ops = &sdsc.dscs_[0]["MatMul_0"].computeOp_;
        assert_eq!(
            ops.len(),
            2,
            "fused epilogue must add exactly one computeOp_ entry: {ops:?}"
        );

        // computeOp_[0]: the matmul itself, UNCHANGED — no trace of the mask operand.
        assert_eq!(ops[0].opFuncName, "batchmatmul");
        assert_eq!(ops[0].inputLabeledDs, vec!["Tensor0-idx0", "Tensor1-idx1"]);
        assert_eq!(ops[0].outputLabeledDs, vec!["Tensor3-idx3"]);

        // computeOp_[1]: the epilogue, reading+writing the MATMUL'S OWN output in place, plus the
        // mask as its second input — exactly the golden's `biasadd` shape.
        assert_eq!(ops[1].exUnit, "sfp");
        assert_eq!(ops[1].opFuncName, "stridedadd");
        assert_eq!(ops[1].inputLabeledDs, vec!["Tensor3-idx3", "Tensor2-idx2"]);
        assert_eq!(ops[1].outputLabeledDs, vec!["Tensor3-idx3"]);

        // The mask tensor still gets its OWN labeledDs_ entry (the on-card walk must address it), typed
        // OUTPUT to match `bmm.ddl`'s bias/bnA/bnB/resadd convention (same layout bucket as the real
        // output, not INPUT).
        let labeled = &sdsc.dscs_[0]["MatMul_0"].labeledDs_;
        assert_eq!(
            labeled.len(),
            4,
            "activation, kernel, mask, output: {labeled:?}"
        );
        let mask_lds = labeled
            .iter()
            .find(|l| l.dsName_ == "Tensor2")
            .expect("mask labeledDs_ entry");
        assert_eq!(mask_lds.dsType_, "OUTPUT");
    }

    #[test]
    fn sub_stick_matmul_is_rejected() {
        // N=65 is not a multiple of the 64-fp16 stick → builder Err (witness a).
        assert!(matmul_opspec(384, 65, 64, 1, "a", "w", "o").is_err());
    }

    #[test]
    fn prefill_m_gt_1_lm_head_folds_to_m1_decode_unchanged() {
        // The mq>1 (prefill) bundle CANNOT run the vocab-wide lm_head at m>1 (it time-tiles, and
        // per-row time-tiling is design-risk-4). It runs it at m=1 over the LAST prompt row instead,
        // which is what lets prefill produce the first generated token's logits itself. This guards
        // both halves of the fold: the per-stick extraction copies, and the m=1 re-lowering. The m=1
        // DECODE bundle must stay a single bare matmul — no copies, no extra ops.
        //
        // ⛔ THROUGH THE WHOLE LOWERING, BECAUSE THE COPIES ARE DESCRIPTORS. The counted ops are
        // SuperDSC descriptors — `hidden/64` `lmlast{j}_o2` copies + `matmul_o2`, main's own names —
        // and after the SubtileIR → KTIR → SuperDSC split the producer emits ONE `lmlast_s0`
        // PROGRAM whose consumer arm (`lower_ktir_to_superdsc::lmlast`) materializes those copies.
        // Asking `lower_one_node` alone (which is what this used to do) counts programs, so a
        // one-program-two-copies fold reads as "one op short" while the emission is exactly main's.
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // hidden[m, H] @ W_lmhead[H, vocab] -> logits[m, vocab] (t2 = the result). Stick-aligned
        // H=128, vocab=256 so the m=1 path is a clean single matmul (no time-tile).
        let (h, vocab) = (128u32, 256u32);
        let build = |m: u32| {
            let tensors = vec![
                TensorShape { rows: m, cols: h }, // t0 = hidden (activation source)
                TensorShape {
                    rows: h,
                    cols: vocab,
                }, // t1 = lm_head weight source
                TensorShape {
                    rows: m,
                    cols: vocab,
                }, // t2 = logits (result)
            ];
            let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
                tensor: TensorId::from_index(t as usize),
                region: ts[t].whole(),
            };
            let node = SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(0, &tensors), whole(1, &tensors)],
                output: whole(2, &tensors),
            };
            let ir: SubtileIR = SubtileIR {
                tensors,
                num_sources: 2,
                nodes: vec![node.clone()],
                result: TensorId::from_index(2),
                // Hand-authored fixture: there is no source op list to be the
                // provenance of, so the map is empty.
                op_output: Vec::new(),
            };
            (node, ir)
        };
        let lower = |m: u32| {
            let (_, ir) = build(m);
            // t1 is the lm_head weight; t0 is the activation source.
            let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
            lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
                .unwrap_or_else(|e| panic!("lm_head at m={m} unexpectedly refused: {}", e.0))
                .0
                .into_iter()
                .map(|e| e.op_name)
                .collect::<Vec<_>>()
        };
        // m>1 (prefill): H/64 extraction copies, THEN the matmul re-lowered at m=1.
        let mq = 8u32;
        let names = lower(mq);
        let copies = (h / Fp16::ELEMS_PER_STICK) as usize;
        assert_eq!(
            names.len(),
            copies + 1,
            "prefill (m>1) lm_head must fold to {copies} extraction copies + 1 matmul, got {names:?}"
        );
        for (j, name) in names.iter().take(copies).enumerate() {
            assert_eq!(
                name,
                &format!("lmlast{j}_o2"),
                "copy {j} misnamed in {names:?}"
            );
        }
        assert_eq!(
            names[copies], "matmul_o2",
            "the folded tail must end in the lm_head matmul"
        );
        // m==1 (decode): the SAME node lowers to exactly one bare matmul — the fold never fires, so
        // the decode bundle is byte-identical to the pre-fold emitter.
        assert_eq!(lower(1), vec!["matmul_o2".to_string()]);

        // ⛔ NEGATIVE CONTROL FOR THE NAME'S OWN FACT. `lmlast` names its copies from
        // `KtirNode::node_out_tid`, and a MISSING one must REFUSE — not fall back to the program's own
        // output, which is the reserved `LAST_HIDDEN_TID` and is exactly the wrong name this test
        // pins. Strip the fact off the extraction program and the lowering must Err naming it.
        let (_, ir) = build(mq);
        let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        let (mut programs, layout) = lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false)
            .expect("KTIR for the tail");
        let extract = programs
            .iter_mut()
            .find(|e| e.op_name.starts_with("lmlast"))
            .expect("the tail emits an extraction program");
        extract.ktir.as_mut().expect("its KTIR").node_out_tid = None;
        let mut sym = 0i64;
        let mut fp8q = std::collections::HashSet::new();
        let stripped = crate::ktir_superdsc_door::lower(
            extract.ktir.as_ref().unwrap(),
            &mut sym,
            Some(&layout),
            &mut fp8q,
            None,
        );
        let why = stripped
            .err()
            .expect("a nameless extraction must refuse")
            .message;
        assert!(
            why.contains("node_out_tid"),
            "the refusal must name the missing fact, got {why:?}"
        );
    }

    #[test]
    fn fp8_shared_activation_quantizes_once() {
        // granite decode emits q/k/v = gemm(normed, ·) — THREE arity-3 fp8 matmuls reading the SAME
        // activation (and gate/up = gemm(normed2, ·) — two more). The per-token activation quantize
        // (square→amax→scale→clamp→qfp8ch) is a PURE function of the activation, independent of the weight,
        // so it must be emitted ONCE and shared — not re-run per matmul. Lock that: two arity-3 fp8 matmuls
        // sharing t0 emit exactly ONE `fq_afp8_op` (qfp8ch) yet still TWO `fq_mm` (per-matmul matmulfp8).
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // ⛔ `n` WAS 64, WHICH IS SUB-STICK FOR fp8. An fp8 stick is 128 elems (fp16's is 64), and the
        // dxp scheduler rejects a sub-stick tile — a guard this crate enforces by construction, so the
        // lowering `Err`s before it can emit anything and this test asserted nothing about its actual
        // subject. That guard landed after the test was written, and the dead target hid it. The subject
        // — ONE shared activation quantize across two matmuls — does not depend on `n`, so `n` becomes a
        // legal fp8 width and the test measures what it is named for.
        let (k, n) = (128u32, 128u32);
        let tensors = vec![
            TensorShape { rows: 1, cols: k }, // t0 = activation (m=1 decode)
            TensorShape { rows: k, cols: n }, // t1 = W1 (fp8)
            TensorShape { rows: 1, cols: n }, // t2 = w_scale1
            TensorShape { rows: k, cols: n }, // t3 = W2 (fp8)
            TensorShape { rows: 1, cols: n }, // t4 = w_scale2
            TensorShape { rows: 1, cols: n }, // t5 = out1
            TensorShape { rows: 1, cols: n }, // t6 = out2
        ];
        let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
            tensor: TensorId::from_index(t as usize),
            region: ts[t].whole(),
        };
        let nodes = vec![
            SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    weight: GemmWeight::Fp8Dynamic,
                },
                inputs: vec![whole(0, &tensors), whole(1, &tensors), whole(2, &tensors)],
                output: whole(5, &tensors),
            },
            SubtileNode {
                id: SubtileId::from_index(1),
                op: SubOp::MatmulTile {
                    weight: GemmWeight::Fp8Dynamic,
                },
                // SAME activation t0, DIFFERENT weight/scale/out → the quantize of t0 must be reused.
                inputs: vec![whole(0, &tensors), whole(3, &tensors), whole(4, &tensors)],
                output: whole(6, &tensors),
            },
        ];
        let ir: SubtileIR = SubtileIR {
            tensors,
            num_sources: 5, // t0 activation + t1..t4 (weights + w_scales)
            nodes,
            result: TensorId::from_index(6),
            // Hand-authored fixture: there is no source op list to be the
            // provenance of, so the map is empty.
            op_output: Vec::new(),
        };
        let weight_ids: std::collections::HashSet<u32> = [1u32, 3u32].into_iter().collect();
        let (ops, _layout) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
            .expect("two-fp8-matmul lowering");
        let quantizes = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op"))
            .count();
        assert_eq!(
            quantizes, 1,
            "two matmuls sharing an activation must quantize it ONCE (shared), got {quantizes}"
        );
        let matmuls = ops.iter().filter(|o| o.op_name.ends_with("fq_mm")).count();
        assert_eq!(
            matmuls, 2,
            "each fp8 matmul still emits its OWN matmulfp8 (weight differs), got {matmuls}"
        );
        // The FIRST chain op is likewise shared: one, not two. It is `abs` (`fq_absx_op`), not the
        // `square` (`fq_sq_op`) this test named — the quantize chain became abs→max, and no op by the
        // old name has existed for as long as this target failed to compile, so the assert was looking
        // for zero of something and would have passed only by finding nothing.
        let first_chain_op = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_absx_op"))
            .count();
        assert_eq!(
            first_chain_op, 1,
            "the activation |x| must be shared too, got {first_chain_op}"
        );
    }

    #[test]
    fn matmul_cost_split_fills_cores() {
        // sdsc_bmm_autoBuffer.json: M=384, N=384, K=64, batch=16 → must use 32 cores.
        let s = matmul_cost_split(16, 384, 384, 64, 32);
        assert_eq!(s.cores(), 32, "matmul split must fill 32 cores: {s:?}");
        // and the iteration space maps M→mb, N→out, K→in, batch→x.
        let it = matmul_iter_space(384, 384, 64, 16);
        assert_eq!((it.mb_, it.out_, it.in_, it.x_), (384, 384, 64, 16));
    }

    #[test]
    fn assemble_matmul_serializes_and_fills_cores() {
        // bmm 384×384×64 batch16 fits LX (576 KiB < 1.6 MiB) → time=1 EmittedOp.
        let emitted = assemble_matmul(
            "MatMul_0",
            384,
            384,
            64,
            16,
            &rb("act", 384, 64),
            &Stk::<KernelTag>::kernel(64, 384, "wt"),
            &rb("out", 384, 384),
            None,
        );
        assert_eq!(emitted.time, 1, "bmm must NOT time-tile (fits LX)");
        let op = emitted.dsc();
        // numWkSlices product = 32 cores.
        let prod: u32 = op.numWkSlicesPerDim_.values().product();
        assert_eq!(
            prod, 32,
            "matmul must use 32 cores: {:?}",
            op.numWkSlicesPerDim_
        );
        // serializes to JSON with the mandatory trailing-underscore keys.
        let j = serde_json::to_string(op).expect("SuperDSC serializes");
        assert!(j.contains("\"coreFoldProp_\""), "coreFoldProp_ present");
        assert!(j.contains("\"numWkSlicesPerDim_\""));
        assert!(j.contains("\"batchmatmul\""), "batch>1 → batchmatmul");
        assert!(j.contains("\"SEN169_FP16\""));
        // time=1 op is NOT symbolic — no isStartAddrSymbolic_, addresses concrete.
        assert!(
            !j.contains("isStartAddrSymbolic_"),
            "time=1 op stays concrete-addr"
        );
        // per-core stage dims = full / split.
        let dsc = &op.dscs_[0]["MatMul_0"];
        assert_eq!(dsc.numCoresUsed_, 32);
        assert_eq!(dsc.computeOp_[0].exUnit, "pt");
    }

    #[test]
    fn bundle_mlir_emits_execute() {
        // The flat (all-time=1) bundle.mlir is byte-identical to the historical form.
        let b = bundle_mlir(&["sdsc_0.json".to_string()]);
        assert!(b.contains("func.func @sdsc_bundle()"));
        assert!(b.contains("sdscbundle.sdsc_execute () {sdsc_filename=\"sdsc_0.json\"}"));
        // A time=1 EmittedOp routes through emit_bundle_mlir → the SAME flat body.
        let emitted = assemble_matmul(
            "MatMul_0",
            384,
            384,
            64,
            16,
            &rb("act", 384, 64),
            &Stk::<KernelTag>::kernel(64, 384, "wt"),
            &rb("out", 384, 384),
            None,
        );
        let via_emitted = emit_bundle_mlir(&[emitted]);
        assert_eq!(
            via_emitted,
            bundle_mlir(&["sdsc_0.json".to_string()]),
            "an all-time=1 bundle.mlir must be byte-identical to the historical flat form"
        );
        assert!(!via_emitted.contains("scf.for"));
    }

    #[test]
    fn tiled_matmul_concrete_unrolls() {
        // 64×16384×2048 batch1 overflows LX (2.42 MiB > 1.68 MiB) → time-tiled.
        // The A term is only 256 KiB so out-tiling brings it under LX (a wide-K
        // shape would Err instead); the cost split fills 32 cores via out×32.
        let emitted = assemble_matmul(
            "matmul_o7",
            64,
            16384,
            2048,
            1,
            &rb("a", 64, 2048),
            &Stk::<KernelTag>::kernel(2048, 16384, "w"),
            &rb("o", 64, 16384),
            None,
        );
        let n = emitted.time;
        assert!(n > 1, "this matmul must time-tile, got time={n}");
        // (ii) the divided per-time `out` shows up in N_ / ss_ (per-core out_per_time).
        let dsc = &emitted.dsc().dscs_[0]["matmul_o7"];
        let split_out = emitted
            .dsc()
            .numWkSlicesPerDim_
            .get("out")
            .copied()
            .unwrap_or(1);
        let per_core_out_per_time = (dsc.N_.out_ as u32) / split_out;
        assert_eq!(
            per_core_out_per_time % 64,
            0,
            "per-time per-core out must be 64-aligned"
        );
        // (iv) CONCRETE-UNROLL: bundle.mlir has N flat executes, NO scf.for / symbols;
        //      the SdscOp JSON is concrete (NOT isStartAddrSymbolic_), and trips differ.
        let mlir = emit_bundle_mlir(&[emitted.shallow_copy()]);
        assert!(
            !mlir.contains("scf.for"),
            "concrete-unroll has no scf.for:\n{mlir}"
        );
        assert!(
            !mlir.contains("affine.apply"),
            "concrete-unroll has no affine.apply"
        );
        assert!(
            !mlir.contains("symbol_ids"),
            "concrete-unroll has no symbol_ids"
        );
        assert_eq!(
            mlir.matches("sdscbundle.sdsc_execute").count() as u32,
            n,
            "one flat execute per trip"
        );
        let trips = concrete_trips(&emitted);
        assert_eq!(trips.len() as u32, n);
        let j0 = serde_json::to_string(&trips[0]).unwrap();
        assert!(
            !j0.contains("isStartAddrSymbolic_"),
            "trip json is concrete, not symbolic"
        );
        assert!(
            j0.contains("{\"factor_\":1,\"label_\":\"time\"}"),
            "sdscFoldProps_ time stays 1"
        );
        assert_ne!(
            j0,
            serde_json::to_string(&trips[1]).unwrap(),
            "trips differ (bumped addrs)"
        );
    }

    #[test]
    fn stick_count_ceils() {
        assert_eq!(stick_count(64), 1);
        assert_eq!(stick_count(65), 2);
        assert_eq!(stick_count(384), 6);
    }

    // ── RUNG-2 LOCK: the matmul work-division is df-aware. An fp8 (128-lane) matmul
    // MUST split its N/K by the 128-stick basis, NOT fp16's 64 — a 64-granular split
    // hands a core a sub-128 slice, the exact `L3DlOpsScheduler:1070 multiple-of-stick`
    // DtException the fp8 bake used to hit. This guard is fail-first: reverting
    // `stick_basis`/`matmul_split_map` to a hardcoded 64 makes it RED. ──
    #[test]
    fn matmul_split_is_df_aware_fp8_128() {
        // N=512: fp16 ⇒ 512/64 = 8 sticks (can split ≤8 ways); fp8 ⇒ 512/128 = 4 sticks.
        // The fp8 split must therefore be COARSER (≤4), never the fp16 8. The `::<Fp8>` type
        // param — not a runtime flag — is what sources the 128 basis onto the dims.
        let dims_f16 = matmul_dims::<Fp16>(
            1,
            &StickExtent::<Fp16>::new(512).unwrap(),
            &StickExtent::<Fp16>::new(256).unwrap(),
            1,
        );
        let dims_f8 = matmul_dims::<Fp8>(
            1,
            &StickExtent::<Fp8>::new(512).unwrap(),
            &StickExtent::<Fp8>::new(256).unwrap(),
            1,
        );
        let s_f16 = matmul_split_map(&dims_f16, MAX_CORES);
        let s_f8 = matmul_split_map(&dims_f8, MAX_CORES);
        let out16 = s_f16.get("out").copied().unwrap_or(1);
        let out8 = s_f8.get("out").copied().unwrap_or(1);
        assert!(
            out16 <= 8,
            "fp16 out split bounded by 8 sticks, got {out16}"
        );
        assert!(
            out8 <= 4,
            "fp8 out split MUST be bounded by 4 (128-)sticks, got {out8}"
        );
        // The fp8 per-core `out` extent is a whole 128-stick multiple (never sub-stick).
        assert_eq!(
            512u32 / out8.max(1) % 128,
            0,
            "fp8 per-core out must be 128-aligned"
        );
    }

    #[test]
    fn workplan_rejects_substick_fp8_split() {
        // A hand-crafted over-split of an fp8 stick dim (2 sticks, split 4 ways) must be a
        // typed `Err` at emit (WorkPlan::divide stick clause, stick_basis=128), NOT an
        // on-card DtException. The SAME split of a fp16 dim (more 64-sticks) is legal.
        let over = |name: &'static str, _: u32| {
            let mut m = std::collections::BTreeMap::new();
            m.insert(name, 4u32);
            m
        };
        let fp8_dim = vec![ItDim {
            name: "out",
            size: 256,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp8,
        }];
        let err = WorkPlan::divide(&fp8_dim, MaxCores::<MAX_CORES>, |d, c| over(d[0].name, c));
        assert!(
            err.is_err(),
            "fp8 256 (=2×128 sticks) split 4 ways must be Err (sub-stick), got {err:?}"
        );
        // fp16 256 = 4×64 sticks ⇒ split 4 ways is exactly 1 stick/core ⇒ Ok.
        let fp16_dim = vec![ItDim {
            name: "out",
            size: 256,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        }];
        let ok = WorkPlan::divide(&fp16_dim, MaxCores::<MAX_CORES>, |d, c| over(d[0].name, c));
        assert!(
            ok.is_ok(),
            "fp16 256 (=4×64 sticks) split 4 ways is 1 stick/core, must be Ok, got {ok:?}"
        );
    }

    #[test]
    fn matmul_split_fp16_byte_identical_to_stick_count() {
        // The dense (fp16) path must be UNCHANGED by the df-aware refactor: for any N the
        // `out` split equals the pre-refactor `stick_count`(÷64)-based split. (Inert-at-fp16
        // is the rung invariant — only fp8 emission changes.)
        for &n in &[64u32, 128, 256, 384, 512, 2048, 5504] {
            let dims = matmul_dims::<Fp16>(
                1,
                &StickExtent::<Fp16>::new(n).unwrap(),
                &StickExtent::<Fp16>::new(64).unwrap(),
                1,
            );
            let split = matmul_split_map(&dims, MAX_CORES)
                .get("out")
                .copied()
                .unwrap_or(1);
            let expected = core_split(stick_count(n), MAX_CORES);
            assert_eq!(
                split, expected,
                "fp16 out split for N={n} must match stick_count-based split"
            );
        }
    }

    // SEN169_FP16 (1-6-9, bias 31) encode — the device's NATIVE fp16, NOT IEEE (1-5-10). Feeding IEEE
    // bits is silently mis-read by the device (1.0→IEEE 0x3C00→SEN169 0.5; 1/576→SEN169 ≈1e-6) — the
    // ~14× rmsnorm scale bug. Anchored to the SFP const table's ground truth (plus1=0x3E00,
    // minus1=0xBE00) + the exp-field/bias/packing points that were wrong in that bug. Concrete
    // machine-check (the float encoder is the ALU leaf; Kani/CBMC over-approximates its libm log2).
    #[test]
    fn sen169_encode_anchors() {
        assert_eq!(sen169_bits(1.0), 0x3E00); // 2^0 ⇒ exp field 31 (bias-31), mantissa 0
        assert_eq!(sen169_bits(-1.0), 0xBE00); // sign bit + plus1
        assert_eq!(sen169_bits(2.0), 0x4000); // 2^1 ⇒ exp field 32
        assert_eq!(sen169_bits(0.5), 0x3C00); // 2^-1 ⇒ exp field 30
        assert_eq!(sen169_bits(0.0), 0); // zero
        assert_ne!(sen169_bits(1.0), 0x3C00); // NOT IEEE-f16 1.0 (0x3C00) — the mismatch that WAS the bug
    }
}

/// ⭐⭐⭐ THIS BUNDLE'S LAUNCH GROUPS: ITS PROGRAMS, IN LAUNCH ORDER.
///
/// One node is one program and one program is one launch. Nothing here partitions by trip kind or
/// carries a shift, and that is not a simplification — it is what is left once the parts with no
/// counterpart are removed.
///
/// ⛔ `KvShifts` HAS NO KTIR COUNTERPART. Every field of it is baked KV addressing or a fold:
/// `slot_stride_bytes` and `slab_stride_bytes` are seg2 shifts, `page_slots` the paged write
/// modulus, `request` the batched request axis, and `page_fold` / `batched_requests` / `fold_rows`
/// describe the fold pass (`bundle/src/lib.rs:384-403`). The KTIR path threads KV through the
/// graph's OWN tensors — the prefix cache is a source and the new K/V a result — so there is no
/// device-side slot to shift to.
///
/// ⛔ AND NEITHER DOES THE TRIP PARTITION. `GroupKind::HostKv` exists so the shim can host-scatter
/// and then skip the on-card copies it replaced; `GroupKind::Slot` fuses cache writes sharing one
/// baked slot. Both describe the shim's fold machinery, not the program.
pub fn ktir_groups(
    ops: &[EmittedOp],
    _fold: FoldGrouping,
) -> Result<Vec<bundle::LaunchGroup<'static>>, SuperDscError> {
    let mut out = Vec::with_capacity(ops.len());
    for e in ops {
        let k = e.ktir.as_ref().ok_or_else(|| {
            SuperDscError(format!(
                "{}: no KTIR program — the node lowering declined this op, and a bundle short a \
                 program computes something else",
                e.op_name
            ))
        })?;
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
        out.push(bundle::LaunchGroup {
            kv: bundle::KvShifts::default(),
            programs: std::borrow::Cow::Owned(vec![bundle::LaunchProgram {
                func: k.func,
                args: std::borrow::Cow::Owned(args),
            }]),
            init_binary: std::borrow::Cow::Borrowed(&[]),
            job_bin_ptr: 0,
            correction: std::borrow::Cow::Borrowed(&[]),
        });
    }
    Ok(out)
}

/// `-Fspyre-hw`'s counterpart to [`ktir_groups`]: the SAME per-op KTIR, lowered to real SuperDSC
/// descriptors and handed to the bake queue.
///
/// ⛔⛔⛔ THIS RENDERS NOTHING AND WRITES NOTHING OF ITS OWN. `lower_ktir_to_superdsc::lower` returns
/// [`EmittedOp`]s carrying in-memory [`SdscOp`]s — built by `pw1`/`pw2`/`assemble_matmul`, the SAME
/// builders the SubtileIR path uses — and everything after that is main's mechanism, unchanged:
/// [`render_dxp_input`] renders, [`launch_index`] partitions, and the `superdsc_bake` queue stages,
/// seals and compiles. The chain is `SubtileIR (memory) → KTIR (memory) → SuperDSC (memory) → this
/// bake → inventory::submit → this launch`.
///
/// ⛔ AND THAT IS A CORRECTION. This function used to call a SECOND emitter — a vendored port of the
/// sibling repo's `triton-superdsc-lower`, with its own `Dsc` model, its own JSON writer, its own
/// `bundle.mlir` writer — and `std::fs::write` the bytes itself. Two emitters for one format meant
/// every layout decision was made twice and the copy got them wrong (a 64-element stick declared on
/// an axis of extent 1, which dxp refuses from inside its scheduler). The copy is deleted; the law
/// lives once, where it is proven on the card.
///
/// Returns the identity each group's compiled artifact will be found under — nothing here reads a
/// compiled result back; that happens once for the whole emit, in [`EmittedBundle::into_code`], after
/// `superdsc_bake::finish_global()` has drained the queue.
fn ktir_groups_via_superdsc(
    ops: &[EmittedOp],
    fp: &str,
    fold: FoldGrouping,
    layout: Option<&BundleLayout>,
    attn_params: Option<crate::ktir_superdsc_door::BundleAttnParams>,
) -> Result<Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)>, SuperDscError> {
    // KTIR → SuperDSC, in memory. One symbol counter for the WHOLE bundle, exactly as the SubtileIR
    // walk threads it: a per-op restart would alias addresses across ops in one bundle.
    let mut sym_id_base: i64 = 0;
    let mut sdsc_ops: Vec<EmittedOp> = Vec::new();
    // fp8 activation-quantize dedup, per BUNDLE — main's own set, threaded the way its tape walk
    // threaded it: q/k/v share one rms output and gate/up another, so each distinct activation is
    // quantized ONCE and the later matmuls emit only `matmulfp8` + dequant.
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    // (program name, descriptors) — written into the dump's `bundle_id.json`, since the eprintln below
    // is invisible on a successful build.
    let mut per_prog: Vec<(String, usize)> = Vec::new();
    for e in ops {
        // ⛔ ONE PATH, AND IT IS CHECKED HERE. An op arriving with a descriptor ALREADY BUILT means
        // some producer arm lowered a `SubtileNode` straight to SuperDSC, which is the `*_sdsc`
        // side-path class. There used to be a pass-through here for exactly one such arm
        // (`lower_sumreduce_node`); that arm is deleted, so the pass-through is a refusal now — the
        // only way to SuperDSC is through a KTIR program.
        if e.op.is_some() {
            return Err(SuperDscError(format!(
                "{}: arrived carrying a SuperDSC descriptor built by the PRODUCER. The only path to \
                 SuperDSC is `SubtileIR → KTIR → SuperDSC`; a node lowered straight to a descriptor \
                 is a second path to the same format. Build its KTIR in \
                 `lower_subtile_tape_to_ktir` and port main's body in `lower_ktir_to_superdsc`.",
                e.op_name
            )));
        }
        let k = e.ktir.as_ref().ok_or_else(|| {
            SuperDscError(format!(
                "{}: no KTIR program — the node lowering declined this op, and a bundle short a \
                 program computes something else",
                e.op_name
            ))
        })?;
        let lowered = crate::ktir_superdsc_door::lower(
            k,
            &mut sym_id_base,
            layout,
            &mut fp8_quantized,
            attn_params,
        )
        .map_err(|err| {
            SuperDscError(format!("{}: KTIR -> SuperDSC: {}", e.op_name, err.message))
        })?;
        // ⭐ PER-PROGRAM ATTRIBUTION, because a TRIP IS NOT A DESCRIPTOR. `render_dxp_input`
        // pre-unrolls each op into `concrete_trips(e)` — `EmittedOp::time` trips, set by
        // `WorkPlan::time_tile_for_lx` when a per-core tile does not fit the scratchpad. So a bundle's
        // trip total is `sum(descriptors × time)`, and a handful of descriptors with a large `time` is
        // indistinguishable from many descriptors in the `GROUPS` line alone. Printed per program so
        // the two causes can be told apart — a body of 17 nodes reporting 8748 trips is one or the
        // other, and only this says which.
        let trips: u32 = lowered.iter().map(|o| o.time.max(1)).sum();
        if lowered.len() as u32 != trips {
            eprintln!(
                "  [ktir->sdsc] {}: {} descriptor(s), {trips} trip(s) — TIME-TILED",
                e.op_name,
                lowered.len(),
            );
        } else {
            eprintln!(
                "  [ktir->sdsc] {}: {} descriptor(s)",
                e.op_name,
                lowered.len()
            );
        }
        per_prog.push((e.op_name.clone(), lowered.len()));
        sdsc_ops.extend(lowered);
    }

    // ⛔⛔⛔ main's FOUR POST-LOWERING GUARDS, AT THE GRAIN WHERE THE DESCRIPTORS NOW EXIST.
    //
    // All four are `lower_graph_to_superdsc`/`emit_bundle_inner`'s own text, and all four went SILENT
    // when the lowering moved here: they read `EmittedOp::op` and `EmittedOp::time`, which main's walk
    // had already filled by the time they ran. On this path the ops reaching `emit_bundle_inner` carry
    // a KTIR program and NOTHING ELSE — `EmittedOp::bare` sets `op: None` and `time: 1` — so the
    // aliasing check's `e.time > 1` was false for every op in every bundle, and the three dxp guards
    // had no `dscs_` to walk at all. A guard that cannot fire is not a guard; `sdsc_ops` below is the
    // list main's guards were written against.
    if let Some(e) = sdsc_ops.iter().find(|e| e.time > 1 && tiled_trips_alias(e)) {
        return Err(SuperDscError(format!(
            "[spyre-superdsc] time-tiled op '{}' (time={}) emits ALIASING per-trip OUTPUT addresses \
             — two trips would write the same HBM byte (silently-wrong). The #50 per-core stride must \
             advance each trip past the previous; this is an internal stride/segment bug. Refusing to \
             bake.",
            e.op_name, e.time
        )));
    }
    // GUARD (priority-1, from OBSERVED on-card failure 2026-06-25): dxp_standalone
    // throws `DtException: allocNode (dsc2.cpp:3999)` when a dsc has an empty
    // scheduleTree_ — the per-tensor HBM allocate nodes are REQUIRED. Refuse to
    // bake a scheduleTree_-less op at BUILD time rather than dxp-crash on-card.
    for e in &sdsc_ops {
        let name = &e.op_name;
        let op = e.dsc();
        for dsc_map in &op.dscs_ {
            for (dname, dsc) in dsc_map {
                if dsc.scheduleTree_.is_empty() {
                    return Err(SuperDscError(format!(
                        "op {name} (dsc {dname}): empty scheduleTree_ — dxp_standalone \
                         DtException's at allocNode (dsc2.cpp:3999) without the per-tensor HBM \
                         allocate nodes. Emit them via the page_coalesce HBM coloring (task #50) \
                         before baking."
                    )));
                }
                // GUARD (priority-1, OBSERVED on-card 2026-06-25): an AllocNode with
                // an empty `coordinates_.coordInfo` makes the scheduler throw
                // `DtException: "There must be at least one valid candidate."`
                // (L3DlOpsScheduler.cpp:1195) — the per-dim tile-folds are required.
                for an in &dsc.scheduleTree_ {
                    let empty_coord = an
                        .coordinates_
                        .get("coordInfo")
                        .and_then(|c| c.as_object())
                        .map(|o| o.is_empty())
                        .unwrap_or(true);
                    if empty_coord {
                        return Err(SuperDscError(format!(
                            "op {name} (dsc {dname}, alloc {}): empty coordinates_.coordInfo — \
                             dxp's scheduler DtException's ('at least one valid candidate', \
                             L3DlOpsScheduler.cpp:1195) without the per-dim tile-folds. Emit the \
                             coordInfo (core_fold/corelet_fold/elem_arr from the work-division) \
                             before baking.",
                            an.name_
                        )));
                    }
                    // GUARD (priority-1, from OBSERVED on-card crash 2026-06-25):
                    // the startAddressCoreCorelet_ fold's [core,corelet] factors
                    // MUST equal the SDSC's declared coreFoldProp_/coreletFoldProp_
                    // sizes, else the FoldManager import throws "Different
                    // cardinality between json and caller" (foldInfrastructure.h:2775).
                    let attrs = &an.startAddressCoreCorelet_.dim_prop_attr;
                    let core_ok =
                        attrs.first().map(|f| f.factor_) == Some(op.coreFoldProp_.factor_);
                    let corelet_ok =
                        attrs.get(1).map(|f| f.factor_) == Some(op.coreletFoldProp_.factor_);
                    if !core_ok || !corelet_ok {
                        return Err(SuperDscError(format!(
                            "op {name} (alloc {}): startAddressCoreCorelet_ fold factors \
                             [{:?}] disagree with coreFoldProp_={} / coreletFoldProp_={} — dxp \
                             FoldManager throws 'Different cardinality' (foldInfrastructure.h:2775).",
                            an.name_,
                            attrs.iter().map(|f| f.factor_).collect::<Vec<_>>(),
                            op.coreFoldProp_.factor_,
                            op.coreletFoldProp_.factor_
                        )));
                    }
                }
            }
        }
    }

    // ⭐ WHICH BUNDLE THE NEXT `GROUPS` LINE BELONGS TO. `launch_index` prints the group/trip totals
    // but takes no bundle identity, so six `GROUPS` lines in a build log are unattributable on their
    // own — and the question that matters is which of them main bakes too. Printed here, immediately
    // before that call, so the pairing is adjacent in the log rather than inferred from order.
    // ⛔ NO DESCRIPTOR-FOOTPRINT AUDIT HERE, AND THAT IS DELIBERATE. One lived here and refused
    // layouts `subtile→superdsc` BAKES AND RUNS: it flagged every `scalarmul` whose bound `[1,1]`
    // scale placement is 2 B while the descriptor reads a stick-collapsed 128 B, which is fine on the
    // card because `align128(off + 2)` pads the cursor to 128 B and the read lands on mapped padding.
    // Same class as our `StickExtent` guard being stronger than the vendor's own baking fixtures.
    // It also never caught the failure it was written for — it was SILENT while the card refused the
    // submission — so it bought a false negative and a false positive and no diagnosis. An addressing
    // invariant belongs in the TYPES that mint addresses, never in a check beside the emitter.
    // ⭐ THE BUNDLE AS dxp SEES IT, ON DISK — for DIFFING AGAINST A KNOWN-GOOD DUMP. `write_dxp_input`
    // exists for exactly this ("how a scheduler refusal gets diagnosed"), and the comparison it enables
    // is the only fence-free way to answer what differs between our prefill descriptors and main's,
    // which run on this same card. Byte-equality against a known-good dump is what localised a layout
    // defect in this repo before; reasoning about paged addressing is what has not.
    //
    // ⚠️ DIAGNOSTIC, AND UNCONDITIONAL BY CHOICE — there is no env gate to hide it behind (this repo
    // forbids behaviour toggles), so it either belongs here or it does not. Remove it deliberately once
    // the prefill fence is closed, or keep it and say so.
    // ⛔ `fold`, NOT A HARDCODED `Split`. This read `FoldGrouping::Split` unconditionally, so a FUSED
    // twin's dump directory was its split's partition under the twin's name — the one thing about a
    // twin that differs (where the group boundaries fall) was the one thing the dump could not show,
    // and a diff of the two dirs came back "identical" no matter what the bake actually did.
    let dump = std::path::Path::new("/tmp/superdsc-dump").join(fp);
    if let Err(e) = write_dxp_input(&dump, &sdsc_ops, fold) {
        eprintln!(
            "  [superdsc-dump] {fp}: could not write {}: {e}",
            dump.display()
        );
    }
    // ⛔ THE IDENTITY GOES IN THE DUMP, BECAUSE THE LOG DOES NOT SURVIVE. cargo swallows a build
    // script's stderr on SUCCESS, so every `BUNDLE`/`GROUPS` line this file prints is visible only when
    // the build FAILS — which means a green build produces 106 dump directories named by fingerprint
    // and nothing anywhere says which is the `mq=19, start=0` prefill the card refuses. Correlating a
    // dump against a log that was never printed is not a comparison; writing the identity beside the
    // descriptors is.
    //
    // `mq` is the discriminator that names the rung, and it is READ FROM THE PROGRAM: the attention
    // program's q view is built first and is `[mq, nqh·hd]`, so its row count is `mq` — the same fact
    // `recognize_attn` recovers, not a new one threaded in.
    let mut progs: Vec<(String, usize)> = Vec::new();
    for (name, n) in &per_prog {
        progs.push((name.clone(), *n));
    }
    let mq = ops
        .iter()
        .filter(|e| e.op_name.starts_with("attn"))
        .find_map(|e| {
            let k = e.ktir.as_ref()?;
            k.func
                .operations
                .iter()
                .find(|o| o.op_type == OpKind::KtdpConstructMemoryView)
                .and_then(|o| {
                    o.attributes
                        .iter()
                        .find(|(key, _)| *key == AttrKey::Shape)
                        .and_then(|(_, v)| match v {
                            Attr::IntList(l) if l.len() == 2 => l.first().copied(),
                            _ => None,
                        })
                })
        })
        .unwrap_or(-1);
    let id_json = format!(
        "{{\"fp\":\"{fp}\",\"mq\":{mq},\"programs\":{},\"descriptors\":{},\"per_program\":[{}]}}\n",
        ops.len(),
        sdsc_ops.len(),
        progs
            .iter()
            .map(|(n, c)| format!("{{\"op\":\"{n}\",\"descriptors\":{c}}}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    if let Err(e) = std::fs::write(dump.join("bundle_id.json"), &id_json) {
        eprintln!("  [superdsc-dump] {fp}: could not write bundle_id.json: {e}");
    }
    eprintln!(
        "[ktir->sdsc] BUNDLE {fp}: {} KTIR program(s) -> {} descriptor(s)",
        ops.len(),
        sdsc_ops.len(),
    );

    // ── from here down this is main's block, verbatim ──
    let shifts = launch_index(&sdsc_ops, fold);
    let mut groups: Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)> = Vec::new();
    // No dxp (a Mac / cardless `cargo check`) ⇒ nothing to stage, and nothing to stage it FOR. This
    // is a CAPABILITY probe, not a behaviour flag.
    if let Some(bake) = crate::superdsc_bake::global() {
        for g in render_dxp_input(&sdsc_ops, fold).map_err(|e| SuperDscError(e.to_string()))? {
            let id = crate::superdsc_bake::GroupId {
                fp: fp.to_string(),
                group: g.group,
            };
            let gdir = bake.stage().group_dir(fp, g.group as usize);
            bake.reserve(g.bytes);
            std::fs::create_dir_all(&gdir).map_err(|e| SuperDscError(e.to_string()))?;
            for (name, contents) in &g.files {
                std::fs::write(gdir.join(name), contents)
                    .map_err(|e| SuperDscError(e.to_string()))?;
            }
            bake.submit(crate::superdsc_bake::SealedGroup::sealed(
                gdir,
                id.clone(),
                g.bytes,
                g.key,
            ))
            .map_err(SuperDscError)?;
            let kv = *shifts.get(g.group as usize).ok_or_else(|| {
                SuperDscError(format!(
                    "{fp}: group {} has dxp input but no entry in the launch index ({} entries) — \
                     `render_dxp_input` and `launch_index` disagree about the group partition",
                    g.group,
                    shifts.len(),
                ))
            })?;
            groups.push((kv, id));
        }
    }
    Ok(groups)
}

/// RE-ROLLED tape-driven SuperDSC lowering — `SubtileIR → KTIR → SuperDSC`, and nothing but the
/// composition of its three steps:
///
///   1. [`lower_subtile_tape_to_ktir`] walks the rerolled tape and builds ONE KTIR program per node
///      (the producer half, in its own file — nothing there emits a descriptor);
///   2. [`crate::ktir_superdsc_door::lower`] turns each of those programs into the real SuperDSC
///      descriptors, reached from [`ktir_groups_via_superdsc`] below (the consumer half, in its own
///      file — main's proven bodies with their input door changed);
///   3. the bake — [`render_dxp_input`] + `superdsc_bake`'s stage/seal/submit, also in
///      [`ktir_groups_via_superdsc`] — compiles them and `inventory::submit`s the result.
///
/// Steps 2 and 3 run per BUNDLE, from [`emit_bundle`], because the tape walk's three segments
/// (prefix / body / suffix) are baked as three bundles over one layout; this entry is step 1 plus
/// the layout they share.
///
/// [`lower_subtile_tape_to_ktir`]: crate::lower_subtile_tape_to_ktir::lower_subtile_tape_to_ktir
pub fn lower_subtile_tape_to_superdsc<F: RopeForm>(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    active_cap: ActiveCap,
    rows_are_requests: bool,
) -> Result<RolledSuperDsc, SuperDscError> {
    crate::lower_subtile_tape_to_ktir::lower_subtile_tape_to_ktir(
        tape,
        ir,
        weight_ids,
        active_cap,
        rows_are_requests,
    )
}
