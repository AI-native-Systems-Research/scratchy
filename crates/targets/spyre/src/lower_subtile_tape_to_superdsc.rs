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
    AttnWiring, BundleWiring, NodeArgs, graph_wiring, lower_graph_to_ktir, lower_graph_to_superdsc,
    lower_subtile_tape_to_ktir, place_step_inputs,
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
    fp8_k_pad, matmul_cost_split, matmul_split_plan, stick_count,
};

// ⭐ THE DXP-GROUP DESCRIPTOR CEILING LIVES IN THE LEAF CRATE — in `sdsc_abstract`, beside the
// emitter's window-chunk split that targets it — so the split decision and the bake-side
// oversized-group guard (in `ktir_groups_via_superdsc` below) cannot each grow their own
// constant. Re-exported here under the leaf's own name: a second divergent value would have to
// be a NEW name in a file that already resolves this one. See `DxGroupCeiling`'s own doc for
// the measured pedigree of 4204.
pub use ktir_superdsc::sdsc_abstract::DxGroupCeiling;
// ⭐ THE POOL'S TYPED PLANE KINDS — brought in beside the ceiling re-export because this file
// places the pool's planes (`pool.plane::<KtPlane>().base_bytes()`): the placement's byte fact
// and the pool's element fact must be one derivation, stated through the pool's own handles.
use ktir_superdsc::sdsc_abstract::{KnatPlane, KtPlane, VPlane};

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
// ✅ `SCRATCHY_SUPERDSC_GROUP_SIZE` IS NOW HERE TOO — see [`GroupSize`]. It was held
// back "until someone settles it on a card" because two authorities looked like they
// disagreed on throughput; they did not. The "it tanks throughput" warning is about
// 2048, while 128-vs-512 is the same speed in three independent measurements. What
// the card DID settle is worse than a throughput question: the two values give
// DIFFERENT OUTPUT at width 8, so the partition can never be ambient.

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
// `scalarmul_scale_tid` and the `KctResident` plane-name type.
//
// ⭐ IT IS THE PRODUCER↔LOWERING ABI, not scratchy's plan: each id names a tensor the LOWERING
// invents or requires and the caller binds, so a third-party KTIR producer has to agree with these
// numbers to bind anything at all. Re-exported so every path in the tree resolves unchanged.
pub use ktir_superdsc::reserved_tids::{
    ATTN_CAUSAL_TID, ATTN_MASK_TID, ATTN_SCALE_TID, ATTN_ZERO_TID, FP8_INV448_TID, FP8_NEG448_TID,
    FP8_POS448_TID, IDENTITY_TID, KCT_RESIDENT_BASE, KV_BLOCK_INDEX_TID, KctResident,
    LAST_HIDDEN_TID, NEW_V_PROBE_TID, ONES_REDUCE_TID, OPROJ_HEADMAJOR_BASE, RESERVED_REGIONS,
    RESERVED_REGIONS_ARE_DISJOINT, RMS_HALF_TID, RMS_INVCOLS_TID, RMS_RSQRT_PROBE_TID,
    RMS_SEED_TID, RMS_VAR_PROBE_TID, ROPE_P_TID, SCALARMUL_SCALE_BASE, SEL_HEADMAJOR_TID,
    SEL_KV_HEADMAJOR_TID, SELT_HEADMAJOR_TID, SENTINELS_ARE_INSIDE_THEIR_REGION, TidRegion,
    identity_class_tid, is_kernel_table_class_tid, oproj_headmajor_tid, reserved_region,
    rope_p_class_hds, rope_p_class_tid, scalarmul_scale_tid,
};

// The rung facts the walk takes as one explicit struct — the `#[forward]` expansion
// (which aliases this module as `superdsc`) constructs it, so it spells through this
// module's path.
pub use crate::lower_subtile_tape_to_ktir::{PagedFoldStatement, RungFacts};

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
    BundleLayout, RetileDescriptor, SegRole, SynthAlloc, TensorPlacement, WEIGHT_SPILL_SEGS,
    WeightTail, align128, syn, synth_footprint_bytes,
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
///  * **Cover arithmetic.** A `FoldBlocks` write-cover whose block does not divide the placement's
///    size describes no legal bind — the doors would refuse every producer forever. See
///    [`ktir_superdsc::placement::BindCover`].
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
        // ── COVER: the write-cover class's arithmetic, proven against the placement's own size ──
        //
        // The class is a law the bind doors enforce at runtime; here its NUMBERS are constants,
        // so a class that cannot describe any legal bind of this placement is a `cargo build`
        // error and not a launch-time refusal. A `FoldBlocks` whose block does not divide the
        // size leaves a tail no whole number of blocks can ever fill; a block of 0 admits only
        // 0-byte binds (the device reads a zero length as 16 GiB); a block past the size admits
        // nothing at all. `Full` states `bytes == size` and needs no arithmetic.
        if let ktir_superdsc::placement::BindCover::FoldBlocks { block_bytes } = p.cover {
            if block_bytes == 0 {
                fail.push(format!(
                    "{} declares a FoldBlocks cover with a 0 B block — no nonzero bind can be a \
                     whole number of zero-byte blocks",
                    p.id
                ));
            } else if block_bytes > p.size {
                fail.push(format!(
                    "{} declares a FoldBlocks cover of {block_bytes} B blocks over a {} B \
                     placement — one block already over-binds it",
                    p.id, p.size
                ));
            } else if p.size % block_bytes != 0 {
                fail.push(format!(
                    "{} declares a FoldBlocks cover of {block_bytes} B blocks over a {} B \
                     placement, which is not a whole number of blocks — the {} B tail can never \
                     be filled by any legal bind",
                    p.id,
                    p.size,
                    p.size % block_bytes
                ));
            }
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

/// ⛔⛔⛔ ONE SESSION STAGES ONE LAYOUT — every ladder rung must place every host-staged ACTIVATION
/// exactly where the ceiling does, the gather's own index excepted.
///
/// A decode session's host binds each step at the CEILING bundle's placements, whichever rung body it
/// then launches, while that body's descriptors were emitted against its OWN layout. A tensor the two
/// place apart is read by the body from bytes the host never wrote there — a clean bake, no fault, a
/// wrong constant. The one tensor allowed to differ is [`KV_BLOCK_INDEX_TID`]: only a gathering body
/// places it, only a gathering step stages it, and [`compute_bundle_layout`] places it LAST so it moves
/// nothing else. The macro calls this on every rung it bakes, so a disagreement is a build failure.
pub fn ladder_stages_one_layout(ceiling: &BundleLayout, rung: &BundleLayout) -> Result<(), String> {
    let staged = |l: &BundleLayout| -> Vec<(u32, usize, u64, u64)> {
        l.placements
            .values()
            .filter(|p| matches!(p.role, SegRole::Activation) && p.tid != KV_BLOCK_INDEX_TID)
            .map(|p| (p.tid, p.segment, p.offset, p.size))
            .collect()
    };
    let (c, r) = (staged(ceiling), staged(rung));
    if c == r {
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
    // ⭐ THE REFUSAL CARRIES ITS OWN EVIDENCE. The lists are the whole story — a positional
    // zip over tid-sorted tuples can only say WHICH entry moved, and the walk order that
    // assigned the offsets is not recoverable from the first difference alone. Every staged
    // activation of BOTH layouts, plus the step-input discovery table (the slot identities
    // behind the tids) and the activation segment's high-water: one refusal, everything a
    // fix needs, no second build to gather it.
    let show = |rows: &[(u32, usize, u64, u64)]| {
        rows.iter()
            .map(|(tid, seg, off, size)| format!("t{tid}:s{seg}@{off}+{size}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let steps = |l: &BundleLayout| {
        l.step_inputs
            .iter()
            .map(|(tid, row)| format!("t{tid}=slot{}:{:?}", row.slot, row.df))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let act_seg = SegRole::Activation.segment();
    Err(format!(
        "a ladder rung places a host-staged activation apart from its ceiling — the host stages the \
         ceiling's placements, so the rung would read bytes nobody wrote: {first}\n  ceiling \
         staged ({}): {}\n  rung staged ({}): {}\n  ceiling step inputs: {}\n  rung step inputs: \
         {}\n  activation-segment high-water: ceiling {} B vs rung {} B",
        c.len(),
        show(&c),
        r.len(),
        show(&r),
        steps(ceiling),
        steps(rung),
        ceiling.segment_bytes[act_seg],
        rung.segment_bytes[act_seg],
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

/// ⭐ THE SCALAR REGISTRY, factored out of [`compute_bundle_layout`] so a pass that needs only
/// the registry can build it WITHOUT a placement: the distinct config-derived on-device scalar
/// VALUES, content-dedup'd by BITS, in node-walk order. Index `i` ↔ reserved const TID
/// `scalarmul_scale_tid(i)` — the registry IS the address law, which is why the bake and the
/// wiring must both compute it from THIS function and never from two walks that can drift.
///
/// ⛔ A PURE FUNCTION OF `ir.nodes`, AND ONLY THAT. [`graph_wiring`] builds a registry-only
/// [`BundleLayout`] from this because the placement pass [`compute_bundle_layout`] would be a
/// THROWAWAY there (the shipped layout is the re-rolled bake's), and it is the CARD's budget
/// pass: it refuses a weight segment past one 16 GiB region with no layer classes to bank on.
/// MEASURED on the old ladder: the 26b MoE under the fp8 preset wired fine through the banked
/// re-rolled bake and then panicked in the wiring on a layout nobody reads. Any fact this
/// needs must stay a pure walk — a placement-derived input would rebuild that refusal one
/// field at a time.
pub(crate) fn scalar_registry<F: RopeForm>(ir: &SubtileIR<F>) -> Vec<f32> {
    let weight_scale_folds = crate::lower_subtile_tape_to_ktir::weight_scale_folds(ir);
    // ⭐ EXCEPT THE WEIGHT-FOLDED ONES: a multiplier that commutes into its matmul's weight
    //    never becomes a device op, so its scale never enters this registry either — the
    //    worker applies it to the staged weight bytes instead. The skip is by NODE, not by
    //    value: the same VALUE pushed by an `AttnDecode` or `RmsNorm` below still lands (the
    //    registry is content-dedup'd, and its consumers look values up BY BITS), so a shared
    //    value keeps its slot and the registry stays the one list both the bake and the bind
    //    read.
    let folded_node = |id: scratchy_subtile::subtile_ir::SubtileId| {
        weight_scale_folds
            .iter()
            .any(|(fold_id, _, _)| *fold_id == id)
    };
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
    // immediates in the KTIR now and never reach a descriptor,
    // because the ported bodies fold the accumulator seed into the contraction exactly as
    // `subtile→superdsc` does.
    for node in &ir.nodes {
        // EVERY config-derived on-device scalar flows through this ONE registry → a `[1,1]` worker-bound
        // const: the muP ScalarMul multipliers (embedding/residual/logits) AND the attention softmax scale
        // (`AttnDecode.scale` == config `attention_multiplier`, set by `attention_scale_for`). NO recompute
        // (the worker must never invent `1/sqrt(hd)` — that ignored the model's real attention_multiplier).
        match &node.op {
            SubOp::ScalarMul { scale } => {
                if !folded_node(node.id) {
                    push_scale(*scale, &mut scalarmul_scales);
                }
            }
            // torch-spyre `spyre__sdpa_overrideable`: scaling_factor = sqrt(scale), applied to BOTH q and K
            // (`query * scaling_factor`, `key * scaling_factor`). Register √scale for the door's split: the
            // decode door emits `qs = Q·√s` / `new_k_scaled = K·√s` reading the `[1,1]` consts, and the weight
            // fold commutes the same √s into W_q/W_k when it applies. The un-split `scale` is the door's
            // lookup anchor (it derives √s from the registry's own entry) and the program's `qk · SCALE`
            // spelling — the emulator runs that mulf; the card's split replaces it.
            //
            // ⭐ EXCEPT THE WEIGHT-FOLDED ONES (the same skip the ScalarMul arm takes): an
            // attention whose √scale commuted into W_q/W_k runs at scale 1.0 and multiplies
            // NOTHING, so neither it nor its square root may occupy a registry slot — and the
            // 1.0 itself most of all, which is why the skip is by NODE and not by value (a
            // pushed identity would shift every scale after it by a slot).
            SubOp::AttnDecode { scale, .. } => {
                if !folded_node(node.id) {
                    push_scale(*scale, &mut scalarmul_scales);
                    push_scale(scale.sqrt(), &mut scalarmul_scales);
                }
            }
            // RMSNorm epsilon (config `rms_norm_eps`) flows through the SAME registry — a `[1,1]`
            // worker-bound const the rmsnorm adds to the mean-of-squares (config value, not dropped).
            // ⛔ THE EPSILON ONLY — the mean-of-squares divisor is NOT a registry scale. `1/cols` is
            // bound at the reserved `RMS_INVCOLS_TID` as a `[1, stick]` row with its own placement,
            // which is where `subtile→superdsc` reads it from; pushing it here as well added one
            // registry slot per rmsnorm node, shifting the tid of every constant registered after it.
            //
            // ⭐ THE (1+w) CONVENTION'S OFFSET RIDES THE SAME LAW: a OnePlusScale norm's gain row
            // needs 1.0 as a bound `[1,1]` const (the door's `rmga_` add reads it exactly like the
            // epsilon), so the census collects `GainConvention::offset()` — 1.0, the convention's
            // own constant, never a second spelling. Pushed ONLY when a real node states the
            // convention, so models without OnePlusScale norms keep the byte-identical registry
            // (the NOTHING-IS-SEEDED law above is a content-dedup'd push of a value a node needs,
            // not a seed: a Scale-only model pushes no 1.0 here at all).
            SubOp::RmsNorm { eps, gain } => {
                push_scale(*eps, &mut scalarmul_scales);
                if matches!(
                    gain,
                    scratchy_subtile::subtile_ir::GainConvention::OnePlusScale
                ) {
                    push_scale(gain.offset(), &mut scalarmul_scales);
                }
            }
            // The gemma-4 vocabulary registers through the SAME registry, each a `[1,1]`
            // worker-bound const its door reads by `scale_slot` exactly like the arms above:
            // RmsNormUnit's epsilon, TanhSoftCap's cap.
            SubOp::RmsNormUnit { eps } => push_scale(*eps, &mut scalarmul_scales),
            SubOp::TanhSoftCap { cap } => push_scale(*cap, &mut scalarmul_scales),
            _ => {}
        }
    }
    scalarmul_scales
}

/// ⭐⭐ THE CENSUS'S STAGING TWIN — the golden's statement of the registry's own
/// binding fact, arm-for-arm with the census above. Every scalar the census routes
/// to the worker-bound `[1,1]` registry is bound as ONE IEEE fp16 per slot
/// (`SCALE_BYTES=2`), and the H2D codec's IEEE→SEN conversion re-normalizes an f16
/// subnormal EXACTLY into a DLF16 normal — so the VALUE the card applies is `f16(v)`,
/// never `v` itself. The spliced programs state the same scalars as f16-typed
/// constexpr multiplies (`qk * SCALE`, `ms + EPS`, `x * SCALE`), and LegalizeTypes
/// collapses those splats onto the f16 grid, so the EMULATOR applies `f16(v)` too —
/// staging the golden puts the reference on the card's side of both bindings at once.
/// The staging is a real precision fact, not a modelling convenience: at gemma-4's
/// `eps=1e-6`, `f16(1e-6) = 17·2⁻²⁴` is an IEEE SUBNORMAL (6% relative steps), and
/// the +1.33% it costs the reciprocal is the card's own loss from binding a
/// subnormal scalar through an f16 slot (a registry slot written in SEN bits
/// directly would keep it ≤0.1%). On real weights the mean-of-squares is O(1) —
/// four orders above eps — and the staging never surfaces (measured on the card
/// dump: meps − mean = 0.0 exactly at O(1) means).
///
/// ARM-FOR-ARM WITH THE CENSUS, and that is the law: a variant the census
/// registers is staged here; a variant it does not is left exact. Two omissions
/// are the census's own: the OnePlusScale offset is 1.0 (f16-exact — staging it is
/// the identity), and the attention `√scale` is not a node field — it is a
/// DERIVED registry slot the door reads: the decode door (`attn_at`) multiplies
/// BOTH Q and the roped new-K by it, so the card's effective decode score scale
/// is `f16(√s)²` — two f16-slot multiplies — while the emulator's program
/// (`qk · SCALE` on the f16 dot) and this golden both apply `f16(s)`. That
/// difference is the KNOWN card-vs-emu attention residual (≈2e-4 relative on the
/// scores at granite-8b's 2⁻⁷), deliberately NOT modeled here: the gate compares
/// the golden against the emulator, and staging `f16(√s)²` would separate the
/// two for nothing. It is the first fact to subtract in any card-vs-emu parity
/// run. The SPLIT norm pair
/// (`RmsNormReduce`/`RmsNormApply`) is absent for the same reason it is absent
/// from the census: it is host-routed on spyre (metal's `decompose_rmsnorm` is the
/// only producer), and a host-routed eps never becomes a device scalar — staging
/// it would mis-model host-exact glue.
///
/// ⛔ THE CALLER'S WALK MUST HAVE SKIPPED THE WEIGHT-FOLDED NODES (as
/// `eval_golden_buffers` does): a folded multiplier reaches the card inside the
/// staged weight bytes as `f16(ws·s)`, and staging its node on top would apply the
/// f16 image twice.
///
/// Returns whether anything was staged — `false` leaves the caller free to use the
/// original node.
fn stage_registry_scalars<F: RopeForm>(op: &mut SubOp<F>) -> bool {
    let stage = |v: &mut f32| *v = half::f16::from_f32(*v).to_f32();
    match op {
        SubOp::ScalarMul { scale } => {
            stage(scale);
            true
        }
        SubOp::AttnDecode { scale, .. } => {
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
    // ⭐ TRUE when this bundle's attention reads the paged KV through the page index — the
    // reference's law: every DECODE gathers its pages (batched rungs and the one-request rung
    // alike), prefill sweeps. Decides the `KV_BLOCK_INDEX_TID` placement alone; every other
    // `rows_are_requests` consumer above keeps its own fact. Same value the emitter admits the
    // gather under — one predicate, both halves.
    gathers_kv: bool,
    // ⭐ THE LAYER STRUCTURE, so the weight segment can be split into BANKS at a LAYER boundary —
    // see [`bank_weight_segment`]. ONE MAP PER ATTENTION CLASS (a hybrid model's classes have
    // different layer counts, and banking needs the class boundaries), from
    // [`per_layer_external_tids`], a pre-pass over the re-rolled tape; EMPTY for an unrolled
    // bundle, which has no layer boundary and therefore cannot bank. A uniform model is one class,
    // so a one-element slice is exactly the pre-class single map.
    per_layer_ext: &[std::collections::BTreeMap<u32, Vec<u32>>],
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
        // ⭐ THE fp8 K-PAD (`work::fp8_k_pad`), PRODUCER-KEYED by the same set above (an arity-3
        // MatmulTile's `inputs[1]`, never a width census): a dense fp8 weight whose logical k is
        // not a whole 128-elem stick (gemma-4-26b's dense GeGLU DOWN: k=2112 = 16.5 sticks) must
        // reserve `fp8_k_pad(k)` ROWS — the packed retile and the worker's staged rebuild both
        // widen to k_pad (the whole-stick law the activation quantize's own tail takes), so a
        // raw-`rows` placement is smaller than the bytes the matmul addresses and the footprint
        // guard refuses the op (26b decode t35: access 6,127,616 B vs placement 5,947,392 B —
        // the pad stick would alias whatever is placed next). Over-reserved is safe (the retile's
        // staging guard reads the WORKER's bytes); under-reserved aliases. Inert at every
        // 128-aligned k (granite, g8b, g12b — byte-identical placement).
        let rows = if fp8_weight_tids.contains(&tid) {
            fp8_k_pad(s.rows).max(s.rows)
        } else {
            s.rows
        };
        rows as u64
            * bump_sticks_to_splittable(s.cols.next_multiple_of(df.elems_per_stick())) as u64
            * df.word_length() as u64
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
    // ⭐ THE PAGED-FOLD REGION'S PAGE CEILING — None until the attention placement below derives it
    // (the SAME `max_pages` that sizes the prefix-validity mask's rows), then carried on the layout so
    // the tape walk states the fold family's tables and mask at the region they actually describe —
    // ONE derivation, one field, no re-computation in the walk.
    let mut paged_fold_region_pages: Option<u32> = None;
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
                // ⭐ FULL — a source's binds are its whole extent: weights stage exactly, and an
                // activation the host binds short is a tail the consumers read unwritten. The
                // fold's per-pass tensors are the one family with a block law, and each of those
                // mints its cover at its own site with its geometry in hand.
                cover: ktir_superdsc::placement::BindCover::Full,
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
    // ⭐⭐⭐ THE HEAD-MAJOR o_proj PERMUTED COPIES — the layout half of the restructure whose
    //    emission halves are the attention finalize's slab-major form and the o matmul's B-operand
    //    swap. Where the head dim is multi-slab and the o matmul is fp8 W8A8, every o_proj weight
    //    gains a SECOND, k-shuffled copy under `oproj_headmajor_tid(w)` — placed in THIS loop,
    //    immediately after its companion, so each layer's block grows by exactly one copy and the
    //    per-layer stride stays UNIFORM at S+E (what the executor's one-base-per-segment advance
    //    needs; the stride guard proves it). ⛔ MINTED BUNDLE-INVARIANTLY: the gate is the GEOMETRY
    //    (`OprojEdges::geometry_applies`), never this bundle's row count — every bundle of a model
    //    shares ONE staged weight segment, so a rows-gated mint would hand the single-row body
    //    addresses no other bundle staged. Only the door's emission gates on rows
    //    (`BundleAttnParams::headmajor_handoff`), and a bundle that never states the handoff
    //    simply never names its permuted copy — staged and unread, like any other resident weight.
    //    ⛔ AND ONE MINT SERVES THE WIRING TOO: [`crate::lower_subtile_tape_to_ktir::oproj_perm_copies`]
    //    is the shared implementation, because a list the placement pass derives separately from
    //    the wiring's is a list that can silently disagree — #299's registry rewrite of
    //    `graph_wiring` dropped the wiring's copy while this pass kept placing, and every copy
    //    sat never-staged while the m=1 fp8 o matmul read it.
    let oproj_perm_after = crate::lower_subtile_tape_to_ktir::oproj_perm_copies(ir)?;
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
        // The companion insert — see `oproj_perm_after` above. Hand-built (NOT through `pack`,
        // which would index `ir.tensors[perm]` for a size a reserved tid has no tensor for): same
        // role, same segment, same BANK, same SIZE as the weight it permutes, one 128-aligned
        // block later. The companion is COPIED out (it is `Copy`) so the insert below does not
        // borrow against it.
        if let Some(&perm) = oproj_perm_after.get(&tid) {
            let seg = SegRole::Weight.segment();
            let off = seg_bytes[seg];
            let companion = placements[&tid];
            placements.insert(
                perm,
                TensorPlacement {
                    tid: perm,
                    role: SegRole::Weight,
                    segment: seg,
                    bank: companion.bank,
                    offset: off,
                    size: companion.size,
                    cover: ktir_superdsc::placement::BindCover::Full,
                },
            );
            seg_bytes[seg] = align128(off + companion.size);
        }
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
    let mut per_layer_tids: std::collections::BTreeSet<u32> = per_layer_ext
        .iter()
        .flat_map(|m| m.values())
        .flatten()
        .copied()
        .collect();
    // ⭐ THE PERMUTED COPIES ARE PER-LAYER WEIGHTS TOO. They ride the same per-layer stride as
    // their companions (the insert placed one after each layer's o_proj), so they must be SPARED
    // by the tail spill exactly as the companions are — a perm copy moved to a spill segment is a
    // per-layer WEIGHT outside the strided segment, which the stride guard below exists to refuse.
    per_layer_tids.extend(oproj_perm_after.values().copied());
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
    // ── ROPE permutation matrix P [hd,hd] (task: in-bundle RoPE) ── If the tape has
    // any RopeRotate/RopeAppend, main's `lower_rope_node` emits `rot = matmul(x, P)` (the
    // rotate-half as a 64-stick-aligned matmul, avoiding the 32-half sub-stick). P is
    // a FIXED permutation-sign matrix the worker synthesizes + binds (it is NOT a
    // model/safetensors weight, so it gets the reserved id ROPE_P_TID). Place it in
    // seg0 (ACTIVATION, re-bound per step like RMS_SEED/ATTN_SCALE) — NOT seg1 (WEIGHT):
    // a seg1 weight is only H2D'd at PrepareModel, where the worker binds ONLY the
    // manifest's model weights, so a synthetic seg1 P stays ZERO ⇒ rot=matmul(x,0)=0 ⇒
    // RoPE collapses to `x·cos` (rotate-half/sin term DROPPED) ⇒ wrong positional
    // encoding ⇒ wrong content. Same P for every head/position/layer of one CLASS.
    //
    // ⭐⭐ ONE P PER DISTINCT ROPE HEAD DIM, NOT ONE AT THE MAX. The max-fold this replaced was
    // itself a half-fix: it stopped a global layer's P from addressing past its footprint, but the
    // VALUE the macro emitted was still staged at the BASE head dim — so on gemma-4
    // tiny-allglobal a `[64,64]` table sat in a `[128,128]` placement and the global class's rope
    // rotated pairs ±32 (half=32 for a 128-wide head) with lanes 64..127 left at cos-only
    // (dump-proven: first divergence `t107` at op2, the roped K). A composite single P is
    // IMPOSSIBLE — the big class's `[hd,hd]` contraction sweeps every input lane, so it would read
    // the small class's nonzero block as spurious ±1 terms — so each class gets its OWN table,
    // OWN placement, OWN tid (`rope_p_class_tid`). Class 0 (the widest, set sorted descending)
    // keeps `ROPE_P_TID`, so a UNIFORM model's ops name the same tid and its bundle is
    // byte-identical.
    //
    // ⛔ THE CLASS SET IS THE SHARED REGISTRY, NOT A LOCAL DERIVATION. The door resolves a rope
    // program's class off `BundleLayout::rope_class_hds` and the worker's load-time bind builds
    // each class's table from the same list the macro bakes into the wiring — so THIS loop's set
    // (via the shared `rope_p_class_hds` helper) is the one source of truth for all three.
    let rope_class_hds = ktir_superdsc::reserved_tids::rope_p_class_hds(
        ir.nodes.iter().filter_map(|n| match &n.op {
            SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
                Some(head_dim.get())
            }
            _ => None,
        }),
    );
    for (class, &hd) in rope_class_hds.iter().enumerate() {
        let tid = ktir_superdsc::reserved_tids::rope_p_class_tid(class);
        let seg = SegRole::Activation.segment();
        let off = seg_bytes[seg];
        let sz = hd as u64 * hd as u64 * 2; // [hd,hd] fp16
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                bank: 0,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
                cover: ktir_superdsc::placement::BindCover::Full,
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
    // ⭐⭐ ONE IDENTITY PER DISTINCT ATTENTION HEAD DIM, NOT ONE AT THE MAX. The max-fold this
    // replaced sized the placement at the widest `AttnDecode` head dim while the VALUE was staged at
    // the base head dim — so on gemma-4 tiny-allglobal a `[64,64]` identity sat in a `[128,128]`
    // placement and the global class's GQA krep/vrep matmuls read slab 1 as ALL ZERO (dump-proven:
    // `newkrep` lanes 64..127 zero, lanes 0..63 exactly the roped K). Each class gets its OWN
    // `[hd,hd]` table, OWN placement, OWN tid (`identity_class_tid`); class 0 (the widest, set
    // sorted descending) keeps `IDENTITY_TID`, so a uniform model's bundle is byte-identical.
    // `attn_class_hds` is the SHARED registry the door resolves a program's class through and the
    // worker's load-time bind builds each class's table from.
    let attn_class_hds = ktir_superdsc::reserved_tids::rope_p_class_hds(
        ir.nodes.iter().filter_map(|n| match &n.op {
            SubOp::AttnDecode { geom, .. } => Some(geom.hd().get()),
            _ => None,
        }),
    );
    if let Some((&hd_max, _)) = attn_class_hds.split_first() {
        for (class, &hd) in attn_class_hds.iter().enumerate() {
            let tid = ktir_superdsc::reserved_tids::identity_class_tid(class);
            let seg = SegRole::Activation.segment();
            let off = seg_bytes[seg];
            let sz = hd as u64 * hd as u64 * 2; // [hd,hd] fp16 identity
            placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    bank: 0,
                    role: SegRole::Activation,
                    segment: seg,
                    offset: off,
                    size: sz,
                    // ⭐ FULL — an identity class is a synthetic constant, bound whole at load
                    // (`synthetic_constants`); a short bind would leave identity rows the
                    // class-split copies read as whatever the shadow held.
                    cover: ktir_superdsc::placement::BindCover::Full,
                },
            );
            seg_bytes[seg] = align128(off + sz);
        }
        // `hd` below (ATTN_ZERO's per-row width) is the WIDEST class's, matching the max-fold this
        // block's placement used to make: ATTN_ZERO is one zero buffer shared by every class's
        // pad-row zero-copies, and `synthetic_constants` binds `mq_pad * head_dim` zeros at the
        // base head dim the wiring carries — unchanged by the class split.
        let hd = hd_max as u64;
        let seg = SegRole::Activation.segment();

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
        // (unconditional, any mq with attention) so main's `lower_attn_node` can emit a real zero-copy into the
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
                cover: ktir_superdsc::placement::BindCover::Full,
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
                cover: ktir_superdsc::placement::BindCover::Full,
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
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
        seg_bytes[a] = align128(ioff + sz);
    }

    // ── fp8 W8A8 activation-quant consts [1,stick] (seg0, worker-bound like RMS_HALF) ── E4M3 clamp
    // bounds ±448 + 1/448 for `qfp8ch` (per-token amax → a_scale). Placed iff the tape has an fp8
    // (arity-3) MatmulTile. Unplaced (and thus value-0) was the "clamp consts default 0 → wrong
    // quant" gap; placing them here + binding in the worker gives the real E4M3 bounds.
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
                    cover: ktir_superdsc::placement::BindCover::Full,
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
        .filter(|&t| {
            t >= ir.num_sources
                && t != result
                && !replaced_kv.contains(&t)
                && !folded_out.contains(&t)
        })
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
                cover: ktir_superdsc::placement::BindCover::Full,
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

    // ── Attention KV cache (seg2) + scale const (seg1) for in-bundle attention ──
    // The IR's AttnDecode K/V cache tensors are reused as the per-layer IDENTITY but
    // RE-placed (override) at the BATCHED size `[nqh,hd,cap]·2` (the worker fills them
    // transposed-K + GQA-replicated V each step — they are read-only INPUTS in-bundle,
    // never written by a SuperDSC op). One shared `scale` const `[1,1]` (worker
    // synthesizes `1/sqrt(hd)`). Placed LAST so it overrides any earlier source/
    // intermediate classification of the cache ids.
    let mut scale_placed = false;
    // The KV block index's bytes when this bundle gathers — sized below beside the mask whose pass
    // grid it shares, CARRIED UNPLACED on the layout (see the comment at its hand-off below) and
    // placed by `BundleLayout::place_kv_block_index_last` after the last host-staged activation,
    // so a body whose gathering differs moves nothing the host stages.
    let mut kv_block_index_reservation: Option<ktir_superdsc::placement::KvBlockIndexReservation> =
        None;
    // ⭐ THE GATHER REACH per pooled source tid — stated beside each attention node's KV plane
    // placements below (the pool those planes' gathers name), consumed by the gather door's
    // entry-unit check.
    let mut gather_reach =
        std::collections::BTreeMap::<u32, ktir_superdsc::sdsc_abstract::GatherReach>::new();
    // ⭐ THE LAYER COUNT A PAGE SPANS — hoisted above the node loop because TWO consumers read
    // it: the gather reach stated at every attention node's plane placements, and the page
    // ceiling the mask's pass grid derives inside the first node's block. ONE derivation, both
    // consumers — the reach and the pass grid cannot disagree about how many layers a page spans.
    //
    // ⭐ THE TRIP COUNT, NEVER THE CONTAINER LENGTH. `per_layer_ext` is ONE MAP PER
    // ATTENTION CLASS, and every value in a class's map is that class's per-layer tid
    // list — filed by [`per_layer_external_tids`] under the law `pl.len() == iters`, so
    // a value's LENGTH is the class loop's iteration count. A page spans EVERY layer of
    // EVERY class, so the count is the SUM of the class trip counts. `.len()` on the
    // slice is the CLASS COUNT — 1 for any uniform model — which understates the page
    // 40× on granite, multiplies `max_pages` by the same factor, and reserves two
    // 10.67 GiB mask/index tensors that seg3 cannot hold. (The merge that brought the
    // derived ceiling here met the per-class slice type and kept `.len()` silently —
    // it type-checks against both the single-map and per-class shapes.)
    let kv_layers = if per_layer_ext.is_empty() {
        ir.nodes
            .iter()
            .filter(|n| matches!(n.op, SubOp::AttnDecode { .. }))
            .count()
    } else {
        per_layer_ext
            .iter()
            .map(|class| class.values().next().map_or(0, Vec::len))
            .sum::<usize>()
    }
    .max(1) as u64;
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
            // it must be in main's `lower_attn_node`). The placements below spend it in TWO roles and each
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
            // The pool from the GEOMETRY itself — the pair at its source, not the u64 memory-math
            // locals above re-narrowed into the seam.
            let pool = scratchy_subtile::sdsc_abstract::PagedKvPool::new(
                geom.nkvh().get(),
                geom.hd().get(),
            );
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
            // Natural K at 0 (`knat_plane_base`), V at `v_plane_base`, Kᵀ at `kt_plane_base` —
            // the offsets `PagedKvPool` bakes into the ops' plane-relative bases, stated here
            // through the pool's TYPED plane handles so the placement's byte fact and the
            // pool's element fact are one derivation.
            for (tid, plane_off) in [
                // natural K, V, transposed K — the pre-paged cache's own order.
                (k_id, pool.plane::<KnatPlane>().base_bytes()),
                (v_id, pool.plane::<VPlane>().base_bytes()),
                (
                    ktir_superdsc::reserved_tids::KctResident::of(k_id).wire_tid(),
                    pool.plane::<KtPlane>().base_bytes(),
                ),
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
                        cover: ktir_superdsc::placement::BindCover::Full,
                    },
                );
            }
            // ⭐⭐⭐⭐⭐ THE GATHER REACH — the POOL a gathered read of these planes names, stated
            // where the slice is placed so the two are ONE fact at one site. The placement above
            // covers this layer's slice of ONE page; a gathered read's index names stick-blocks
            // RELATIVE TO THE PLANE'S OWN BASE across the whole pool — every page, in the
            // uniform `hd · 64`-element block grid `GatherEntry::of_page_block` counts in — so
            // the gather door checks the read's entry stride against THIS reach instead of the
            // slice footprint. The reach is the TYPED pair
            // (`GatherReach::of_plane` — the entry ceiling and the entry unit), minted from the
            // SAME page ceiling the mask's pass grid derives in the first node's block below
            // (one flex region, `bundle::MAX_SEGMENT_BYTES`, over this bundle's own page bytes)
            // and the pool's own geometry, and the host-side index composition mints through
            // the SAME door (`gather_entry_ceiling`), so a value above the ceiling is
            // unconstructible at both halves and the two cannot disagree. Nothing else widens:
            // a plain sweep of these tids is still bounded by the placement alone.
            let page_elems = pool.layer_stride() as u64 * kv_layers;
            let page_bytes = page_elems * 2;
            let page_ceiling = (bundle::MAX_SEGMENT_BYTES / page_bytes).max(1);
            let reach = ktir_superdsc::sdsc_abstract::GatherReach::of_plane(
                page_ceiling,
                page_bytes,
                pool,
            )
            .ok_or_else(|| {
                SuperDscError(format!(
                    "attn t{k_id}: the paged KV pool's own geometry admits no gather reach — a \
                     page ({page_bytes} B over {kv_layers} layer(s)) is not a whole number of the \
                     pool's stick blocks, so no integer index can name a page boundary and the \
                     gathered read this bundle emits cannot address the pool"
                ))
            })?;
            // ⛔ ONE SPELLING OF THE TID SET — `pool_reach_tids`, the same derivation the
            // region stamp's premise guard asks, so the set this insertion covers and the set
            // the stamp checks can never name different pools.
            for tid in ktir_superdsc::sdsc_abstract::pool_reach_tids(k_id, v_id) {
                gather_reach.insert(tid, reach);
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
                            cover: ktir_superdsc::placement::BindCover::Full,
                        },
                    );
                    seg_bytes[seg3i] = align128(off + pbytes);
                }
            }
            if !scale_placed {
                // Attention MASKS → seg0 (activation, re-bound per step), worker-TILED to [nqh·mq, *]:
                // pmask=[nqh·mq, max_pages] (prefix validity), cmask=[nqh·mq, mq_pad] (causal triu).
                // Shared across layers (same shape every layer). The attention SOFTMAX SCALE is NOT
                // placed here: it flows through `scalarmul_scales` (config `attention_multiplier`,
                // via `AttnDecode.scale`), placed below with the other config scales and bound by the
                // worker's scale loop — the emitter NEVER recomputes 1/sqrt(hd), which would ignore
                // the model's real attention_multiplier.
                //
                // (The 2026-07 note that pmask was "over-reserved" described the pre-windowing
                // upload: `refill_activations` now sends the BOUND extent as merged aligned runs,
                // never the reservation, and the host binds only the live blocks — so the tail of a
                // deep reservation is dead ADDRESS SPACE, not dead H2D.)
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
                // regression. The activation segment is small and is uploaded every step anyway,
                // so the mask rides along for free.
                //
                // Shifting it per page is safe HERE and nowhere else: dumped from the baked bundle,
                // a fold group addresses exactly two things — the KV segment, and the mask. It
                // touches nothing else in the activation segment, so the shift moves nothing it reads.
                let mseg = SegRole::Activation.segment();
                let pmoff = seg_bytes[mseg];
                // ⭐⭐⭐⭐⭐ THE CONTEXT CEILING IS DERIVED FROM THE DEVICE, NOT CHOSEN.
                //
                // A row may hold as many pages as ONE POOL can, and a pool is a single flex region
                // (`bundle::MAX_SEGMENT_BYTES` — the same cap the runtime budget takes from
                // `card_bytes.max_single_allocation()`, which `runtime` guards equal to it).
                // The reservation is that cap over the model's own page bytes, so it covers every
                // pool any card can budget for and MEMORY binds at runtime, exactly as it does on
                // metal: a bigger budget, a wider card or a denser page (fp8 KV) raises the served
                // context by arithmetic, not by editing a constant. The number is per-bundle fact —
                // read back at load as `BakedMaskBlocks` — not a shared const, because the page
                // stride is a MODEL fact and no model-independent constant can be its ceiling.
                //
                // The page bytes are this bundle's own placement facts: `layer_stride` (three planes
                // of `nkvh · hd · PAGE_SLOTS` elements — the same deltas the runtime derives the
                // pool stride from) × 2 B × the layer count. The layer count is the roll's trip
                // count for a re-rolled tape; an unrolled graph states its own, one AttnDecode node
                // per layer.
                //
                // ⭐ THE TRIP COUNT IS `kv_layers`, hoisted above the node loop — the gather
                // reach stated at every node's plane placements and this page ceiling read ONE
                // derivation of the layer count (the trip-count law lives at the hoist).
                let layers = kv_layers;
                let page_bytes = pool.layer_stride() as u64 * 2 * layers;
                let max_pages = (bundle::MAX_SEGMENT_BYTES / page_bytes).max(1);
                // ⭐ THE FOLD FAMILY'S REGION — the SAME ceiling, bounded by the pool THIS graph
                // laid. `max_pages` is the DEVICE-side bound (one flex region over this bundle's
                // page bytes — the number the mask reservation above keeps, covering every pool a
                // card could budget); the fold's region is a REACH fact: its tables and mask are
                // walked one page per pass by entries that gather from THIS pool, and no context
                // can out-reach the pool's own rows (the cache tensor's row extent, `cap`). The
                // min is belt-and-braces in the direction that cannot strand a context: a pool
                // larger than one region cannot be placed, so `cap <= max_pages` in every baked
                // bundle, and a smaller pool (this 256-page one against the 546-page device
                // ceiling) takes its own page count as the region — the fold's plane descriptors
                // state the region, and a region past the pool's placements sweeps past them
                // (the splice refuses exactly that).
                paged_fold_region_pages = Some(max_pages.min(cap).max(1) as u32);
                // ONE validity row over the max context, or one PER QUERY ROW when the rows are
                // separate requests and each carries its own history. The multiplier is the op row
                // count the attention sweeps (`nqh*mq`), which is what the per-row read addresses.
                // It stays 1 for a prompt chunk — at prefill's mq=96 the widened form would be
                // tens of MB of mask re-uploaded every forward, and a prompt does not need it.
                // ONE BLOCK PER FOLD PASS when the rows are requests. A pass reads ONE request's
                // page, so every other row must be masked off for it — the mask is `[pass][row][page]`
                // and the runtime steps it by a whole `rows * page` block. A prompt keeps one
                // broadcast row per page, which is all its rows can need.
                // ⭐⭐⭐ AND A GATHERED BUNDLE BLOCKS THE MASK **PER PAGE**, SO ITS PASSES ARE THE PAGES
                // ONE ROW CAN HOLD — `max_pages` here — where an uncollapsed fold's are `pages × requests`
                // over the shared pool. That is a SMALLER reservation per page, and since the host stages
                // only the live blocks it is also `mq`x less mask uploaded every step: at bs=8 and three
                // pages, 3 blocks instead of 24.
                //
                // ⛔ WHICH FORM THIS BUNDLE USES IS THE SAME `rows_are_requests` CONDITION THE INDEX
                // PLACEMENT AND `assemble_attn`'s gather are gated on — one predicate, three consumers.
                // Getting it wrong is silent in the dangerous direction: too few blocks against the fold
                // is a fold reading mask bytes nobody staged, which read as ZERO and therefore VALID.
                let (pm_rows, pm_blocks) = if rows_are_requests {
                    (nqh * mq, max_pages)
                } else {
                    (1, max_pages)
                };
                let pmbytes = pm_rows
                    * pm_blocks
                    * scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u64
                    * 2;
                // ⭐⭐⭐ THE MASK'S WRITE-COVER — the same `rows_are_requests` predicate the
                // reservation above gates on, because it is the same fact: a PROMPT CHUNK's one
                // row is bound WHOLE ("the extent is the bake's reserve … every swept column
                // gets an explicit value" — `Full`), while a decode BATCH's host stages only
                // the LIVE blocks of the `[pass][row][page]` grid, one `pm_block_bytes` block
                // per live page. `pm_block_bytes` is `pmbytes / pm_blocks` by construction —
                // the mask's own pass stride, the ONE number the KV block index's reservation
                // takes for its pitch too. There is no second spelling of this number.
                let pm_block_bytes =
                    pm_rows * scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u64 * 2;
                placements.insert(
                    ATTN_MASK_TID,
                    TensorPlacement {
                        tid: ATTN_MASK_TID,
                        bank: 0,
                        role: SegRole::Activation,
                        segment: mseg,
                        offset: pmoff,
                        size: pmbytes,
                        cover: if rows_are_requests {
                            ktir_superdsc::placement::BindCover::FoldBlocks {
                                block_bytes: pm_block_bytes,
                            }
                        } else {
                            ktir_superdsc::placement::BindCover::Full
                        },
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
                        // ⭐ FULL — the causal mask is bound whole, keyed, once per causal
                        // shape (`CausalKey`); a short bind would leave swept columns
                        // unmasked, i.e. VALID.
                        cover: ktir_superdsc::placement::BindCover::Full,
                    },
                );
                seg_bytes[a] = align128(cmoff + cmbytes);
                // ⭐⭐⭐⭐⭐ THE KV BLOCK INDEX — the index tensor a GATHERED KV read is addressed through.
                //
                // A seg0 ACTIVATION, exactly like the two masks above and for the same reason: it is
                // per-STEP host data (the block table changes as requests grow pages), and a seg1 weight
                // is only H2D'd at PrepareModel, so a synthetic seg1 index would stay ZERO — which for
                // an index tensor means every row gathering block 0, i.e. row 0's keys for everyone.
                //
                // ⭐⭐⭐ PLACED IFF THE BUNDLE GATHERS, WHICH IS WHAT MAKES THE PLACEMENT THE ANSWER.
                // It used to be unconditional, and that cost the runtime its only artifact-derived way
                // to ask the question: `BakeFacts::gathers_kv` is this placement's presence, exactly as
                // `uses_identity` is `IDENTITY_TID`'s, and it decides whether the forward tape carries a
                // `KvBlockIndex` step. Unconditional, every bundle claimed to gather, so every launch
                // would have had to stage a table — including the prompt-chunk and solo-decode paths
                // that emit no gather at all — and the one condition that must govern both halves would
                // have had to be re-derived somewhere else.
                //
                // ⭐⭐⭐ `gathers_kv` is that condition, and it is the SAME fact
                // `lower_ktir_to_superdsc` passes `assemble_attn`'s `kv_block_index` from. The
                // reference gathers its pages on EVERY decode — batched and one-request alike — so
                // both rungs place the index; prefill's rows are consecutive positions of one
                // sequence sharing ONE resident history, so there is nothing for an index to
                // distinguish and its emission stays byte-identical.
                //
                // ⛔⛔⛔⛔⛔ AND IT IS **STILL** THE BARE ROW-KIND-ADJACENT PREDICATE, ALONE, AGAINST THE
                // OBVIOUS FIX, BECAUSE THE 8b BATCHED PATH **READS THESE BYTES** — MEASURED, TWICE, ON
                // THE CARD.
                //
                // The split is real and it is stated on [`GatherScratch::admits`]: `assemble_attn` may
                // DECLINE the gather on the geometry (`of_fold_pass` refuses a head dim its flat copy
                // cannot express) and this placement does not ask, so an hd=128 bundle reserves this
                // activation, `BakeFacts::gathers_kv` answers `true` and the forward tape stages a table
                // that NO descriptor gathers through — while `GathersKv::of_launch_groups`, the per-BODY
                // door, answers `false` for the same bundle. Three doors, two answers. At hd=64
                // `of_fold_pass` never refuses, which is why it was invisible.
                //
                // ⛔ NARROWING THIS WITH `GatherScratch::admits` — the one-line fix, which reserves only for
                // the bundles that really gather — CANNOT BE VERIFIED TODAY, BECAUSE THE 8b BATCHED ORACLE
                // DOES NOT HOLD. Measured on pod `nickm-7db9667cdd-z2jc6`, `scr batch` over the `c` probe
                // (420-token generations, one distinct integer sequence per row), each rung against its OWN
                // `--max-num-seqs 1` run of the identical file:
                //
                // | tree | rung 2 | rung 4 | rung 8 |
                // |---|---|---|---|
                // | this one, narrowed | `solo_diff=1` | `solo_diff=3` | `solo_diff=7`, 3 DEGEN |
                // | this one, NOT narrowed (the code below) | `solo_diff=1` | `solo_diff=3` | `solo_diff=7` |
                // | **`f080bb60a` REBUILT, md5-verified byte-equal** | `solo_diff=1` ×3 trials | — | — |
                // | `f080bb60a` as recorded in its own commit message | `solo_diff=0` | `solo_diff=0` | `solo_diff=0` |
                //
                // ⭐ THE CONTROL IS THE WHOLE POINT. Narrowing this reservation moves every activation after
                // it, so the first reading — "the hole was load-bearing padding and I removed it" — was
                // entirely plausible and it is WRONG: the parent commit, restored file-by-file to md5
                // equality and rebuilt on this same pod, is equally non-solo-exact. `rc=0`, no fault, ITL
                // and wall unchanged across every variant (rung 8: 173.4 / 173.9 / 175.4 ms), and each bad
                // row diverges from its own solo at a common prefix of 87-118 of 420 tokens. The SOLO
                // oracles are byte-identical across all of it (`solo=EXACT` 8/8 against the baseline's own
                // solo jsonl), so the oracle is not what moved either.
                //
                // ⛔ SO "granite-3.1-8b fp8 RUNS EVERY RUNG CLEAN AT 3.66x" IS NOT REPRODUCIBLE FROM THE
                // COMMIT THAT RECORDS IT, and the 3.66x is a THROUGHPUT number whose correctness column was
                // one sample. What differs between this morning's binary and the same source rebuilt is not
                // in the source; the untested candidate is the BUNDLE BUILD (dxp's work division is
                // thread-count sensitive and these builds are pinned to `taskset -c 0-19`), which would make
                // the 8b's batched numerics a property of the compile rather than of the emission.
                //
                // ⛔⛔⛔ AND IT IS NOT AN hd=128 STORY: **granite-3.1-2b fp8 — the ONE geometry the gather
                // actually runs on — is WORSE.** Same pod, same probe, same tree (whose 2b emission is
                // byte-identical to `f080bb60a` by construction: at hd=64 `admits` is `true`, so the name
                // gate reduces to `rows_are_requests` and this placement is untouched code), two trials per
                // rung:
                //
                // | rung | solo | batched | `solo_diff` |
                // |---|---|---|---|
                // | 2 | ITL 27.8 ms | 33.9 / 33.7 ms | **2 of 2**, both trials |
                // | 4 | ITL 27.7 ms | 39.3 / 40.5 ms | **4 of 4**, both trials |
                // | 8 | ITL 28.1 ms | 63.5 / 62.4 ms | **8 of 8**, both trials |
                //
                // NOT ONE ROW matches its own solo at any width, and rung 4 finished 1385 and 1307 of the
                // solo's 1680 tokens — rows terminating early, which is the collapse symptom, not noise. The
                // pre-gather 2b record in `/work/bw/out.pre-gather` has the same probe at width 8 with
                // `solo_diff=0`. So the gather's own 4.11x rests on a correctness column that does not hold
                // today either, and the gather is ON in every one of those runs.
                //
                // ⛔ AND THAT IS WHY THIS STAYS AS IT IS. Not because the narrowing is wrong — it is right,
                // and [`GatherScratch::admits`] states the rule — but because landing an address-moving
                // change against an oracle that fails 1-of-2 rows on the unmodified parent would pin
                // whatever it produced as "no regression". The order is: give the 8b batched path a STABLE
                // oracle first (find why the same source rebuilt is not the same numerics), then narrow
                // this, then measure. Until then only the emitter-side half of the one rule lands:
                // `lower_ktir_to_superdsc` gates the tensor's NAME on `admits`, which is provably
                // non-observable (`assemble_attn`'s `kv_block_index.zip(of_fold_pass(..))` is already `None`
                // at hd=128 whichever way the name goes), so the coupling is stated where it costs no
                // addresses and this placement keeps every byte it had.
                if gathers_kv {
                    // ⛔⛔⛔ SIZE = THE ENTRIES THE **DESCRIPTOR DECLARES**, NOT THE ENTRIES THE HOST
                    // MEANS TO FILL. This was `WIDEST_BATCH_RUNG * MAX_PAGES_PER_ROW` — 32 x 16 = 512
                    // entries, "one per (launch row, logical page), the same shape as the host's
                    // `block_tables`". MEASURED against the real emission
                    // (`the_declared_index_holds_no_more_entries_than_the_bundle_reserves`), the
                    // descriptor declares **2048**: the index's dims are the value's PAGED dims and its
                    // extents come from the op's own iteration space, so along the slot axis it is the KV
                    // WINDOW WIDTH (256) and not a page count — `["out","mb"] = [256, 8]`.
                    //
                    // 512 reserved against 2048 declared is 1536 entries read PAST the placement, and past
                    // it is the next seg0 tensor's bytes read as block numbers. Block numbers are valid
                    // addresses, so that bakes clean, faults nothing, and gathers from wherever those
                    // bytes happen to point.
                    //
                    // So the bound is the descriptor's own worst case: the slot axis can declare at most a
                    // whole page (`PAGE_SLOTS`, the pre-paged capacity every rung's `active_cap` is a
                    // divisor of) and the batch axis at most the widest rung. Over-reserving is inert — the
                    // tail is simply unread — while under-reserving is an address, so the asymmetry decides
                    // which way to round. At `PAGE_SLOTS x WIDEST_BATCH_RUNG x 4 B` this is 32 KB of an
                    // activation segment that is uploaded every step anyway.
                    //
                    // ⛔ ENTRIES ARE GLOBAL STICK-BLOCK INDICES, NOT PAGE NUMBERS. dxp computes
                    // `addr = idx * skip_addr + base_addr`, and `skip_addr` is what the score leg's Kᵗ
                    // operand emits as its entry: ONE STICK BLOCK, `hd * ELEMS_PER_STICK` elements —
                    // MEASURED at 4096 at head_dim 64, and 8192 at head_dim 128. That unit works because
                    // the whole pool is a uniform array of stick blocks (`plane_block_elems` is four of
                    // them; `plane_stride` / `layer_stride` / `page_stride` all multiples), so one index
                    // reaches any cell with no relayout. An entry written as a PAGE number is short by
                    // `page_stride / stick_block` — hundreds — and lands inside a different layer:
                    // `sdsc_abstract::gather_entries_per_page` is the factor, and it is NOT a constant.
                    //
                    // ⭐⭐⭐⭐⭐ THE SIZE IS **THE MASK'S OWN PASS GRID**, AND THAT PITCH IS FORCED BY THE LAUNCH
                    // ABI, NOT CHOSEN.
                    //
                    // The gather now lives on the KERNEL-less copy op, whose gather dim is `mb` with no
                    // second pinned axis — so the index is a FLAT rank-1 table of `nkvh * nb * mq` entries
                    // per fold pass, and dxp's own formula reduces to `ceil_to_stick(mb)` with no per-row
                    // stick to pad to. What sizes this placement is therefore not the entry layout but the
                    // PASS layout: pass `p` needs its own entries, and the only per-pass shifts a launch has
                    // are `kv`, `mask` and `intermediate` (`fold_plan::SegDeltas`). KV is resident and never
                    // uploaded; the intermediate segment carries `qs` and the whole online-softmax state, so
                    // shifting it would move every operand the pass reads. The index is an ACTIVATION —
                    // `SegRole::Activation` IS `SEG_MASK` — so it rides the MASK's shift, and its per-pass
                    // pitch must BE the mask's block stride.
                    //
                    // ⛔ SO THE TWO PLACEMENTS DERIVE THEIR PITCH FROM ONE NUMBER, `pmbytes / pm_blocks`,
                    // computed just above for the mask itself. A pitch that disagrees puts pass `p`'s gather
                    // on some other pass's entries — a clean bake reading real block numbers from the wrong
                    // page. `gather_index_table` takes the SAME value as `pass_stride_entries`.
                    //
                    // ⛔ AND IT IS THE MASK'S BLOCK COUNT TOO — the derived page ceiling, whatever
                    // it is: a gathered fold's passes are the pages ONE ROW holds. Over-reserving
                    // would be inert here, but agreeing with the mask's own grid is what keeps the
                    // shift meaningful for both tensors.
                    kv_block_index_reservation =
                        Some(ktir_superdsc::placement::KvBlockIndexReservation {
                            bytes: pmbytes,
                            block_bytes: pm_block_bytes,
                        });
                }
                // DIAGNOSTIC probe (mq>1 only): persistent seg0 buffer for layer-0's pre-selector new_v
                // [mq_pad, nkvh·hd]. main's `lower_attn_node` copies layer-0 new_v here; the worker reads it to split
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
                            cover: ktir_superdsc::placement::BindCover::Full,
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
        let mut checked: Vec<u32> = vec![
            ROPE_P_TID,
            ATTN_MASK_TID,
            ATTN_CAUSAL_TID,
            RMS_HALF_TID,
            RMS_INVCOLS_TID,
            ONES_REDUCE_TID,
        ];
        // The rope-P and identity CLASS tids ride the same check: every class table is worker-bound
        // per step exactly like the class-0 sentinels, so a class tid in seg1 is the same silent-zero
        // defect. (The class lists are the registries this same pass built, so the tids here are the
        // ones the door and the load-time bind resolve.)
        checked
            .extend((0..rope_class_hds.len()).map(ktir_superdsc::reserved_tids::rope_p_class_tid));
        checked.extend(
            (0..attn_class_hds.len()).map(ktir_superdsc::reserved_tids::identity_class_tid),
        );
        for &rtid in &checked {
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

    // ── granite ScalarMul scale constants ── the registry itself is [`scalar_registry`] above
    //    (the factored pure walk — the same function the wiring pass builds its registry-only
    //    layout from, so the bake's list and the wiring's cannot drift); what stays HERE is
    //    only the placement: a `[1,1]` worker-bound const per scale in seg0 (ACTIVATION,
    //    exactly like ATTN_SCALE). The pointwise `mul` the door lowers reads the index here →
    //    the const TID it multiplies by; the worker binds each `t{tid}=[scale]`. The PAYLOAD
    //    half is all `Fp16` — the staging-twin law, the value the card applies is `f16(v)` —
    //    so every census placement is 2 B and the format column is one flat Fp16 list. The
    //    program-stated half (which can state `Fp32`) is appended LATER, by the completion
    //    pass, at each slot's own word length.
    let scalarmul_scales = scalar_registry(ir);
    let scalarmul_scales_df =
        vec![ktir_superdsc::superdsc_opspec::Df::Fp16; scalarmul_scales.len()];
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
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
        seg_bytes[a] = align128(off + 2);
    }

    // ⛔⛔⛔ THE KV BLOCK INDEX IS NOT PLACED HERE — ITS RESERVATION IS CARRIED, AND THE WALK
    // PLACES IT LAST OF ALL. One session runs bodies that gather and bodies that do not — the
    // one-request ladder's per-page bodies under a split-KV chunk ceiling gather through the
    // page table while its unchunked sweep rungs sweep the resident KV — and the host stages
    // every input at the CEILING's placements, while each body's descriptors read its OWN
    // layout. Placed where it was — first beside the mask, then at the tail of THIS pass —
    // the index still sat before the two placements that come after the pass
    // (`place_step_inputs`, `complete_program_scales`), so a body whose gathering differed
    // pushed every step input and program scale down by its reservation (279,552 B on
    // granite-3.1-2b) and `ladder_stages_one_layout` refused the first such rung: the ceiling
    // staged `t1412` at 603,392 while the rung's own layout read it at 323,840. The
    // reservation therefore rides the layout as a typed value
    // ([`BundleLayout::kv_block_index`]), and the ONE door
    // [`BundleLayout::place_kv_block_index_last`] — called by each walk after the last
    // host-staged placement — puts it past every offset the guard compares, so the two
    // layouts agree on every other tensor by construction, which is what
    // `ladder_stages_one_layout` holds the macro to. A reservation still pending at the bake
    // is a build panic, never a silently unplaced index.
    let kv_block_index = kv_block_index_reservation;

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
        // The intermediate-tid cursor seeds LAZILY, on the first mint, above the highest
        // non-reserved placement of the whole bundle — the placement map is complete here,
        // so the seed sees every tensor (see `BundleLayout::take_inter_tid`).
        inter_cursor: Default::default(),
        segment_bytes: seg_bytes,
        weight_bank_bytes,
        weight_tail,
        kernel_weights: std::collections::BTreeMap::new(),
        scalarmul_scales,
        scalarmul_scales_df,
        weight_scale_folds: weight_scale_folds
            .into_iter()
            .map(|(_, tid, scale)| (tid, scale))
            .collect(),
        weight_copies: oproj_perm_after
            .iter()
            .map(|(w, perm)| (*w, *perm))
            .collect(),
        synth,
        // Pinned-LX registry starts empty (stick 1024 base): it fills only when a
        // decomposition declares an intermediate via `BundleLayout::synth_lx_op_local` (the
        // exx2 rmsnorm `rinv` stick) — every other bundle leaves it untouched.
        lx: Default::default(),
        arrangements: std::cell::RefCell::new(std::collections::BTreeMap::new()),
        kv_request_stride_bytes,
        // ⭐ THE GATHER REACH — filled beside each attention node's KV plane placements above.
        // Empty for every bundle with no paged KV, keeping those byte-identical.
        gather_reach,
        // ⛔ THE CLASS REGISTRIES, filled where their tables are placed above — the door resolves
        // a rope/attention program's class off these lists and the worker's load-time bind builds
        // each class's table from the same lists the macro bakes into the wiring, so this is the
        // ONE source of truth for all three. Empty for a tape with no rope/attention nodes.
        rope_class_hds,
        attn_class_hds,
        // ⭐ THE HOST-STAGED STEP INPUTS — empty here (the layout walk places graph tensors and
        // pools only); `place_step_inputs` fills it after the splices state their bindings.
        step_inputs: std::collections::BTreeMap::new(),
        // ⭐ THE KV BLOCK INDEX'S PENDING RESERVATION — unplaced by design (the comment at its
        // derivation above); the walk's `place_kv_block_index_last` spends it after the last
        // host-staged placement.
        kv_block_index,
        paged_fold_region_pages,
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
                    cover: ktir_superdsc::placement::BindCover::Full,
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
        .filter_map(|n| match &n.op {
            SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
                Some((n, head_dim.get() as usize / 2))
            }
            _ => None,
        })
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
    /// The reference legs' zero-copy window over this staging.
    pub(crate) fn view(&self) -> StagedWeightView<'_> {
        StagedWeightView {
            bytes: &self.bytes,
            gemm: &self.gemm,
            scale_rows: &self.scale_rows,
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
}

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
}

/// Stage the synth weights (the loop `numeric_gate` ran inline before the
/// truncation rework, extracted so the DBG oracle's golden states the SAME
/// law). `only` = the tids to stage (the covering bound set); `None` = all of
/// `weight_ids`.
fn stage_synth_weights<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    only: Option<&std::collections::HashSet<u64>>,
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
                            dbg_synth_val(tid as usize, l * cols + j) * FP8_SYNTH_SCALE,
                        ));
                    }
                }
            } else {
                for e in 0..n {
                    b.push(ktir_core::codec::f32_to_e4m3(
                        dbg_synth_val(tid as usize, e) * FP8_SYNTH_SCALE,
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
                            dbg_synth_val(tid as usize, l * cols + j) * fold_scale(tid),
                        );
                    }
                }
            } else {
                for e in 0..n {
                    push(&mut b, dbg_synth_val(tid as usize, e) * fold_scale(tid));
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
    let mut src_vals: Vec<Vec<f32>> = (0..nsrc)
        .map(|id| {
            if staged_weight(id) || covering.is_some_and(|c| !c.contains(&(id as u64))) {
                Vec::new()
            } else if let Some(v) = staged_rope.get(&(id as u32)) {
                v.clone()
            } else {
                let t = ir.tensors[id];
                let n = (t.rows as usize) * (t.cols as usize);
                match rope_table_half.get(&(id as u32)) {
                    Some(&half) => (0..n).map(|j| dbg_synth_val(id, j % half)).collect(),
                    None => (0..n).map(|j| dbg_synth_val(id, j)).collect(),
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
        let mut attn_node;
        let mut staged_node;
        let node_ref = if let SubOp::AttnDecode { .. } = &node.op
            && attn_folds.iter().any(|(id, ..)| *id == node.id)
        {
            attn_node = node.clone();
            if let SubOp::AttnDecode { scale, .. } = &mut attn_node.op {
                *scale = 1.0;
            }
            &attn_node
        } else {
            staged_node = node.clone();
            if stage_registry_scalars(&mut staged_node.op) {
                &staged_node
            } else {
                node
            }
        };
        let mut out = scratchy_subtile::subtile_ir::eval_node(node_ref, ir, &bufs);
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
                out = scratchy_subtile::subtile_ir::eval_node(node_ref, ir, &bufs);
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
        // the staging block above for the law). The fold's reader-half alias
        // below scatters the SAME rounded product — the folded tid holds the
        // producer's stick, grid included.
        for v in out.iter_mut() {
            *v = half::f16::from_f32(*v).to_f32();
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
    emu: &std::collections::HashMap<u64, Vec<f32>>,
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
        let mut attn_node;
        let mut staged_node;
        let node_ref = if let SubOp::AttnDecode { .. } = &node.op
            && attn_folds.iter().any(|(id, ..)| *id == node.id)
        {
            attn_node = node.clone();
            if let SubOp::AttnDecode { scale, .. } = &mut attn_node.op {
                *scale = 1.0;
            }
            &attn_node
        } else {
            staged_node = node.clone();
            if stage_registry_scalars(&mut staged_node.op) {
                &staged_node
            } else {
                node
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
        let mut tf_out = scratchy_subtile::subtile_ir::eval_node(node_ref, ir, &tf);
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
                tf_out = scratchy_subtile::subtile_ir::eval_node(node_ref, ir, &tf);
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
        for v in tf_out.iter_mut() {
            *v = half::f16::from_f32(*v).to_f32();
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
                // program's product is ONE row (row 0 of the readback — the
                // stick above it is sized by the IR and never written), so the
                // row compares the emulator's row 0 against this walk's LAST
                // row (the `selector_lastrow_col` extraction the lowering
                // pins) and the teacher's pin is skipped: every row-bearing
                // consumer of a folded tid is itself folded (row-less by the
                // readback-length check above), so there is nothing
                // downstream to teacher-force with the emu's value.
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
    let staged = stage_synth_weights(ir, weight_ids, None);
    let view = staged.view();
    let bufs = eval_golden_buffers(ir, staged_rope, &view, None);
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
/// rope tables read row-aligned and the lm-head tail's m=1 fold stated as the
/// last-row law. A PREFIX-CAPABLE prefill bundle (ActiveCap::FULL — the
/// continuation chunks) is still refused by the CALLER with a loud skip line:
/// its resident-prefix mask law is the next rung. Graphs containing ops
/// `eval_node` has no host reference for (the MoE router/expert family, the KV
/// codec) are SKIPPED with a line naming the first such op and the fingerprint
/// — the reference for those families is the next rung, and a silent skip
/// would be a hole; a panicking one would break every MoE bake that works
/// today.
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
}

#[cfg(feature = "spyre-emu")]
pub fn numeric_gate<F: RopeForm>(
    groups: &[bundle::LaunchGroup<'static>],
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
    let ir_owned;
    let ir: &SubtileIR<F> = if prefill.is_some() || trunc.is_some() {
        ir_owned = {
            let mut g = ir.clone();
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
    // ── The fold list the scales manifest's skip set states ──
    let sm_folds = crate::lower_subtile_tape_to_ktir::scalar_mul_weight_folds(ir);
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
    // ── Synth weights: staged ONCE for both legs — the emulator's resident
    //    bytes (moved into the session's HBM below) and the reference legs'
    //    per-node materialization (the clone the view borrows: `new_multi`
    //    moves the staging's own bytes, and the golden and the walk both
    //    outlive the session). The covering filter stages just the weights
    //    the kept launches read (measured shape: the 12b's full staging is
    //    ~12 GB of codes, its 7-of-48 covering slice ~1.75 GB); the clone is
    //    the one transient double, and the slice keeps both sides small. ──
    let StagedWeights {
        bytes: emu_weights,
        gemm,
        scale_rows,
    } = stage_synth_weights(ir, weight_ids, covering);
    let weights_ref = emu_weights.clone();
    let view = StagedWeightView {
        bytes: &weights_ref,
        gemm: &gemm,
        scale_rows: &scale_rows,
    };
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
        let n = (t.rows as usize) * (t.cols as usize);
        let vals = if scale_rows.contains(&tid) {
            vec![1.0; n]
        } else if let Some(v) = staged_rope.get(&tid) {
            v.clone()
        } else if let Some(&half) = rope_table_half.get(&tid) {
            (0..n)
                .map(|j| dbg_synth_val(tid as usize, j % half))
                .collect()
        } else {
            (0..n).map(|j| dbg_synth_val(tid as usize, j)).collect()
        };
        sources.push((tid as u64, vals, vec![t.rows as usize, t.cols as usize]));
    }
    // The compile-time scalars, at their reserved tids — the reader-side law
    // `scalarmul_scales.json` states (same first-seen non-folded walk).
    let folded =
        |id: scratchy_subtile::subtile_ir::SubtileId| sm_folds.iter().any(|(fid, _, _)| *fid == id);
    let mut sm: Vec<f32> = Vec::new();
    for node in &ir.nodes {
        if let SubOp::ScalarMul { scale } = &node.op
            && !folded(node.id)
            && !sm.iter().any(|s| s.to_bits() == scale.to_bits())
        {
            sm.push(*scale);
        }
    }
    for (i, v) in sm.iter().enumerate() {
        sources.push((u64::from(scalarmul_scale_tid(i)), vec![*v], vec![1, 1]));
    }
    // The length mask: the synthetic one-past-the-graph tid (`BundleWiring`'s
    // own placement law). ⭐ NOT ALL-VALID — the append slot must stay masked.
    // A rope-append prefix's IR region is sliced to `valid_len - 1` rows (the
    // keys BEFORE this token) while the spliced program sweeps the rung and
    // lets THIS mask bound it; all-valid unmasks the append slot itself and
    // hands the program one extra key the reference does not attend (its
    // v-cache row is never written — the fresh v rides the new-v segment —
    // measured: granite decode diverged at every AttnDecode output by exactly
    // one synth row's v-mix term at weight 1/257). The reference's own key set
    // is the region, so the mask keeps exactly its rows valid. Pre-populated
    // prefixes read their region directly (the mask parameter is truncated
    // away for them), so a tape with no rope-append prefix keeps the
    // all-valid default — the bind is then simply unused.
    if let Some(pg) = prefill {
        // ⭐ THE PREFILL MASK, BY THE RUNTIME'S OWN LAW: the forward's emu
        // staging binds a `[mq, mq]` clamped causal triangle — the same two
        // calls the worker's `forward_chunk` composes (`ChunkRows::row_logical_pos`
        // over `prefill_causal_col_valid`, the SSOT the one-pass kernel's own
        // `[MQ, MQ]` mask descriptor states) — at a FULL chunk: every row
        // real, so the clamp is the identity and the golden, which has no
        // pad-row law, sees exactly the rows the program reads. ⛔ NOT
        // `CausalKey::mask`: that is the card's `[mq, mq_pad]` stick-padded
        // layout — a different tensor than the one-pass kernel consumes.
        let mq = pg.mq as usize;
        let chunk = scratchy_subtile::sdsc_abstract::ChunkRows::chunk(
            std::num::NonZeroU32::new(pg.mq).expect("a prefill rung bakes mq > 1"),
        );
        let mut m = vec![0.0f32; mq * mq];
        for r in 0..mq {
            let pos = chunk.row_logical_pos(r, mq);
            for c in 0..mq {
                if !scratchy_subtile::sdsc_abstract::prefill_causal_col_valid(c, pos) {
                    m[r * mq + c] = -1.0e38;
                }
            }
        }
        sources.push((u64::from(ir.tensors.len() as u32), m, vec![mq, mq]));
    } else {
        let valid_prefix = ir
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                SubOp::AttnDecode {
                    producer:
                        scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. },
                    ..
                } if n.inputs.len() >= 5 => Some(n.inputs[1].region.rows.len as usize),
                _ => None,
            })
            .max()
            .unwrap_or(cap as usize);
        sources.push((
            u64::from(ir.tensors.len() as u32),
            crate::manifest::attn_mask_fill(cap as usize, valid_prefix),
            vec![1, cap as usize],
        ));
    }
    // ── Run the bundle on the emulator ──
    // The golden tids, in the writer's own first-seen order — each with the
    // registry name of the op that first produces it (`SubOp::name`, so no
    // parallel string table can drift) and that node's INPUT tids, carried
    // into every verdict row: a refusal then names the CLASSES that diverge
    // and hands the reader the input buffers to judge magnitude-dependent
    // mechanisms (a reduce's landing grid is a function of its input scale).
    let mut seen = std::collections::HashSet::new();
    let mut golden_tids: Vec<u64> = Vec::new();
    let mut golden_op: std::collections::HashMap<u64, &'static str> =
        std::collections::HashMap::new();
    let mut golden_ins: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
    for n in &ir.nodes {
        let t = n.output.tensor.index() as u64;
        if seen.insert(t) {
            golden_tids.push(t);
            golden_op.insert(t, n.op.name());
            golden_ins.insert(
                t,
                n.inputs.iter().map(|r| r.tensor.index() as u32).collect(),
            );
        }
    }
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
    let refs: Vec<(&[bundle::LaunchGroup<'static>], &[u64])> = vec![(groups, &results)];
    // ⭐ SEAT (i), THE EMPTY TABLE ON PURPOSE: this gate's own scale staging (above) is the
    // census half — `SubOp::ScalarMul` multipliers, every one staged at the incumbent f16 —
    // so there is no f32 slot to declare and the session stays byte-identical. A program
    // stating an f32 scale (the triton m-merge, once its l-line lands) hands the REAL table
    // here, and a bundle whose f32 slot goes undeclared would fall back to the f16 fill —
    // which is why the worker's one real call site passes the baked column, not this default.
    let mut session = crate::runner::SpyreSession::new_multi(&refs, emu_weights, &[])
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
    // `mut`: the teacher-forced walk borrows the golden's staged sources
    // through it (pointer-swapped per node, restored before return).
    let mut eval_bufs = eval_golden_buffers(ir, &staged_rope, &view, covering);
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
    // ⭐ THE RESULT RIDES THE FOLD — keyed on the structural fact, not set
    // membership: the result joins the folded law iff the splice's own
    // `tensor_feeds_result_through_row_local_ops` holds from a FOLDED node's
    // output to the result — the same predicate that admitted the fold, so
    // nothing un-folded can ride in. The front end column-chunks every wide
    // pointwise tail op (`n_blocks(out_cols, nb)` chunks Silu/Mul/Add/
    // ScalarMul/SiluMul — measured: granite's 49155-wide logits ScalarMul is
    // SEVEN region-slice nodes, six of 8192 + one of 3), and the predicate's
    // width conjunct reads the output TENSOR's cols, not the chunk region's,
    // so the chunking does not change the fold's decision: EVERY region-slice
    // piece of the result-feeding tail tensor folds to the m=1 tail. (This is
    // the leg-55 fix: region-keyed, every piece's 8192 failed the equality,
    // the pieces stayed un-folded and read mq−1 rows the folded m=1 matmul
    // never wrote — "row 0 is still right ... the device pays mq−1 rows of
    // reads of unwritten logits for nothing" — a cost-only shape the tree
    // carried on main too, until the read-before-write walk refused it at
    // prefill leg 55.) The folded pieces each scale one row-0 column slice of
    // the folded matmul's (= LAST-row) product — row-locally, guaranteed by
    // this same walk — so the RESULT's row 0 is the fold's product and its
    // rows 1+ are the dead stick: it takes the same last-row-vs-row-0 law
    // (measured on the granite refusal dump: last-row rel 0.00049 at the fp8
    // floor, rows 1-30 exactly zero; the plain law reads 1.0136). A no-op
    // when the tail's own op matched whole-region (gemma's TanhSoftCap is
    // never chunked and already put the result in the set).
    if ir.nodes.iter().any(|n| {
        folded.contains(&(n.output.tensor.index() as u64))
            && scratchy_triton_splice::tensor_feeds_result_through_row_local_ops(
                ir,
                n.output.tensor,
            )
    }) {
        folded.insert(ir.result.index() as u64);
    }
    for &(tid, _) in &compare {
        let g_full = &eval_bufs[tid as usize];
        let e = match out.get(&tid) {
            Some(v) => v,
            None => {
                failures.push(format!("t{tid}: no readback"));
                continue;
            }
        };
        // The fold is a documented transform, not a defect: the splice lowers
        // the result-width tail to m=1 (the last-row extraction feeding the
        // re-lowered ops), so those tids' program product is ONE row — row 0
        // of a readback whose STICK the IR still sizes at [mq, cols] (the
        // program never writes past row 0; measured: t302's stick read back
        // 19 rows with rows 1-18 all zero). The compare states the fold's own
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
        let tol = if is_fp8 { FP8_TOL } else { CLEAN_TOL };
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
    let (walk_tf, split) =
        teacher_forced_walk(ir, &mut eval_bufs, &out, &tainted, &folded, &view, covering);
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
                &overlay,
                &tainted,
                &folded,
                &view,
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
            match write_refusal_dump(fp, cap, &failures, &verdicts, &eval_bufs, &out, &evidence) {
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
    eprintln!(
        "[spyre-superdsc] NUMERIC GATE PASS {fp}: {} tensors at the floor (worst local \
         clean {worst_local_clean:.4}, worst local fp8-chained {worst_local_fp8:.4}; worst \
         total clean {worst_clean:.4}, worst total fp8-chained {worst_fp8:.4}){scope}",
        compare.len()
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
    /// is exactly what it was. The request rides as a [`TripRequest`] — the same wrapper [`Trip`]
    /// carries — so the kind's own comparison (`fusable_kind_with`) cannot match it against a
    /// bare small integer that happens to equal a request.
    Slot { req: TripRequest },
    /// `slot_write` PREFILL cachewr: writes a DISTINCT per-entry slot (baked at `s·hd`). The
    /// shim's single per-group slot-shift CANNOT express distinct slots, so each is its OWN
    /// singleton group (launched with its baked slot doff; prefill `slot_pos==0` ⇒ no shift).
    /// This makes "distinct-slot copy fused into a uniform-shift group" UNCONSTRUCTABLE.
    SlotSolo,
    /// PAGE FOLD (`kv_page_fold`): folds one page of resident prefix. Consecutive fold trips FUSE
    /// (they share a page base) but must never fuse with anything else — the shim re-launches this
    /// group per page, and a stray op swept in would re-run per page too, re-seeding the new-token
    /// block or re-adding the residual once per page.
    ///
    /// ⭐⭐⭐⭐⭐ `gathered` IS THE OP'S OWN [`EmittedOp::kv_gathered`], AND IT DECIDES THE SIZE CAP —
    /// see [`GroupKind::run_may_be_chunked`]. It lives in the KIND because the chunking rule is a
    /// property of the kind and must stay in ONE place; a gathered fold's passes communicate through a
    /// scratch they rewrite, an ungathered fold's do not, and that is the whole difference.
    PageFold { gathered: bool },
    /// `slab_write` DECODE incremental Kᵀ restickify (kill-restickify Stage 2, hd==64): re-transposes
    /// ONLY the current 64-slot slab; the shim shifts the group's seg2 base by
    /// `(slot_pos/64)·slab_stride_bytes` ONCE (each op baked at slab 0). Consecutive `Slab` trips FUSE
    /// ≤ g (the nkvh per-kv-head restickifies, all one `slab_stride_bytes`). DISTINCT kind from `Slot`:
    /// its stride (STICK_BYTES·stick = 8192) ≠ the cachewr's `slot_stride_bytes` (128), so fusing the two
    /// would trip the per-group uniform-stride assert (mixed-stride group → wrong base shift).
    Slab,
    /// A GATHERED read outside any fold — the split-KV chunk's copies (`attn.rs`'s
    /// `assemble_attn_chunk`), which pull every resident page through the index ONCE per launch.
    /// Consecutive `Gathered` trips FUSE, but never with anything else: the group's `KvShifts::gathered`
    /// is what zeroes its KV page shift (the entries are absolute) and what tells the host this body
    /// needs its index staged, so a `Pure` op swept in would lose its page base, and a copy swept into
    /// a `Pure` group would gain one.
    Gathered,
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
    /// ⭐ THE OP'S [`EmittedOp::kv_part`] — which page of a multi-page window it addresses. A
    /// [`PagePart`](ktir_superdsc::page_part::PagePart), minted only by the cutting law
    /// (`PagedKvPool::write_parts`) and copied op → trip → launch, so the value the walk
    /// compares and the value the launch shifts by are ONE value. Compared beside the request,
    /// for every kind, and for the same reason: a launch resolves ONE page per group, so a
    /// cache-write part (a `Slot`) and a Kᵀ re-transpose part (a `Pure`) of the window's second
    /// page cannot share a group with anything aimed at its first. WHAT THE PART CARRIES IS THE
    /// ARGUMENT'S TYPE, NOT THIS COMMENT — a bare `u32` here is what made the walk's comparison
    /// and the launch's re-read two derivations that could disagree (and did, guarded only by an
    /// assert at bake).
    pub part: ktir_superdsc::page_part::PagePart,
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
            (GroupKind::Gathered, GroupKind::Gathered) => true,
            // AND THE SAME GATHER STATE. They move together in today's emitter (one `gather` decision
            // serves a whole attention block), so this arm is `true` in practice — but a group takes
            // its SIZE CAP from its first trip's kind, so a run mixing the two would silently give one
            // half the other's cap. Breaking at that boundary can only cost a launch.
            (GroupKind::PageFold { gathered: a }, GroupKind::PageFold { gathered: b }) => a == b,
            (GroupKind::Slot { req: a }, GroupKind::Slot { req: b }) => a == b,
            // SlotSolo: distinct baked slot per entry — one per-group shift cannot express two.
            // HostKv: the shim's `oi += skip` must land on this exact entry.
            _ => false,
        }
    }

    /// ⭐⭐⭐⭐⭐ WHETHER A RUN OF THIS KIND MAY BE **CHUNKED BY THE GROUP SIZE** — false for
    /// `PageFold { gathered: true }` ALONE, and that is a CORRECTNESS law, not a tuning choice.
    ///
    /// ⛔⛔⛔ A GROUP IS THE UNIT OF THE `reps` RELAUNCH, AND THE LOOP IS GROUP-MAJOR.
    /// `superdsc_exec::launch_ops_inner` is `for op in ops { for rep in 0..reps(op) { … } }`, so a fold
    /// cut into groups A then B runs **A(pass 0..n), then B(pass 0..n)** — never A(0),B(0),A(1),B(1).
    /// Every op inside one pass therefore has to be inside one group.
    ///
    /// For an UNGATHERED fold the split is survivable by accident: each pass reads the pool through its
    /// own KV segment shift, and the online softmax's running max/sum/output is accumulate-only, so
    /// applying the (window, pass) contributions group-major gives the same answer. **For a GATHERED
    /// fold it is wrong**, because the passes communicate through a buffer they REWRITE: group A's copy
    /// leaves the gathered scratch holding page `n-1`, and every one of group B's passes then reads page
    /// `n-1`. With one pass that is invisible — A writes page 0, B reads page 0 — which is exactly why
    /// this survived every single-page test and broke at the first page crossing, at ANY batch width
    /// (MEASURED on granite-3.1-2b fp8: first divergence at absolute slot 256, offset 0 into page 1, at
    /// width 2 and width 8 alike; `SCRATCHY_SDSC_PEROP_SYNC=1` does not move it, because the order is
    /// semantically wrong rather than racy).
    ///
    /// ⭐ AND [`GroupSize`]'s OWN DOC ALREADY RECORDED THE SYMPTOM WITHOUT NAMING IT: `solo_diff` is
    /// 8,8,8,8,8 at `g = 128` and 7,6,7 at `g = 512`, "so the partition into compile groups is the only
    /// variable, and it changes the answer". A partition that changes the answer is this: at 512 more of
    /// the fold fits one group. The fix is not a bigger `g` — a bigger `g` only moves which contexts are
    /// wrong — it is that this run is never cut.
    ///
    /// ⛔⛔⛔⛔⛔ AND IT IS THE **GATHERED** FOLD ONLY — "one rule for both folds" WAS SHIPPED AND IS A
    /// MEASURED 8b REGRESSION.
    ///
    /// What stood here claimed two things, and both were false. It said gating on the gather "would put
    /// the decision in a second place (the trip carries no gather bit, so it would have to be threaded)":
    /// [`EmittedOp::kv_gathered`] already exists, `attn.rs` already sets it from the same `gather` the
    /// fold's own [`GatheredFold`] is built from, and [`launch_index`] already reads it one function
    /// away — so the bit was never absent, and carrying it in the KIND keeps the rule in exactly the one
    /// place that comment wanted. It also said "the ungathered fold loses nothing by it: fewer groups is
    /// fewer launches". That is the assumption granite-3.1-8b falsifies.
    ///
    /// ⛔ MEASURED, `RedHatAI/granite-3.1-8b-instruct-FP8-dynamic` (hd = 128) **AT A TREE WHERE
    /// [`PageScratch::of_pass`] STILL REFUSED TWO SLABS** — confirmed with `SCRATCHY_GATHER_DIAG=1` SET
    /// and 0 gather steps, against 421 at 2b — so that model's fold was UNGATHERED and these three rows
    /// are three partitions of the SAME ungathered fold. That refusal is gone and the 8b's fold is now
    /// gathered (419 diag steps, `own_bad 0,0,0`, `solo_diff 0,0,0`, N=3), so the table below is evidence
    /// about the UNGATHERED fold — which is what the rule is about. 420-token `c` probe at width 8, each
    /// binary scored against its OWN `--max-num-seqs 1` run, N = 3:
    /// ```text
    ///   origin/main  (cap = g)     own_bad 8,7,7  degen 4,5,3  first divergence 87-118 chars (term 14-19)
    ///   cap lifted for BOTH folds  own_bad 8,8,8  degen 1,1,2  first divergence 1-4 chars (term 0):
    ///                                                          " 11111111…", " / / / / /…"
    ///   THIS RULE (gathered only)  own_bad 7,8,8  degen 3,2,1  first divergence 87-118 chars (term 14-19)
    /// ```
    /// Every 8b width-8 cell is broken — that defect predates all of this and is not what changed — but
    /// lifting the cap on the ungathered fold turns a prefix coherent for fifteen terms into garbage from
    /// the first token, and restoring the cap restores main's per-row divergence offsets EXACTLY
    /// (99/106/118/93/106/112/87 chars, row for row, on both trees).
    ///
    /// ⛔ WHY it damages the ungathered fold is NOT established, and this comment does not guess: the
    /// only thing measured is that the partition is the variable. Which is the same shape as the bug the
    /// exemption exists for, and the reason the exemption is now no wider than its evidence.
    ///
    /// ⭐ THE NECESSITY ARGUMENT ONLY EVER COVERED THE GATHERED FOLD. Its passes read a scratch the
    /// gather REWRITES per pass, so group-major execution makes group B read the page group A left. An
    /// ungathered pass rebases the KV segment instead and holds no state between groups — the online
    /// softmax is accumulate-only — so it needs no exemption and is not given one.
    fn run_may_be_chunked(self) -> bool {
        !matches!(self, GroupKind::PageFold { gathered: true })
    }
}

impl Trip {
    /// A trip addressing its request's page at the write cursor — every trip but a page part's.
    pub fn new(kind: GroupKind, req: TripRequest) -> Trip {
        Trip::at_part(kind, req, ktir_superdsc::page_part::PagePart::at_cursor())
    }

    /// A trip addressing the page `part` names past the write cursor — the part the cutting law
    /// minted for the op this trip was built from (see [`Trip::part`]).
    pub fn at_part(
        kind: GroupKind,
        req: TripRequest,
        part: ktir_superdsc::page_part::PagePart,
    ) -> Trip {
        Trip { kind, req, part }
    }

    /// THE ONE RULE THAT DECIDES EVERY GROUP BOUNDARY: same request, same page part, and kinds that fuse.
    ///
    /// A group is one launch; a launch resolves one request's page table and one write cursor and
    /// shifts the segment base ONCE. Two requests therefore cannot share a group whatever their
    /// kinds — so the request test lives HERE, above the kind test, and applies to kinds that do not
    /// exist yet. Two page parts cannot either: their pages differ, and the shift is one page — the
    /// part comparison is over the OPAQUE [`PagePart`](ktir_superdsc::page_part::PagePart) type,
    /// which the cutting law alone mints, so it cannot be confused with a slot index, a page count
    /// or a `rep`, and the value compared here is the value the launch later shifts by (there is no
    /// second derivation to disagree with it).
    fn fusable_with(&self, other: &Trip) -> bool {
        self.req == other.req && self.part == other.part && self.kind.fusable_kind_with(other.kind)
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
        // ⛔ THE CAP IS PER KIND — see [`GroupKind::run_may_be_chunked`]. A GATHERED `PageFold` run is ONE
        // group however long it is, because a group is the unit of the `reps` relaunch and the launch loop
        // is group-major: cutting it makes every pass of the second group read the scratch the first group
        // left at its LAST pass. An UNGATHERED fold holds no such state, so it chunks like everything
        // else — exempting it too is a measured 8b regression.
        let cap = if trips[start].kind.run_may_be_chunked() {
            g
        } else {
            usize::MAX
        };
        while end < n && cnt < cap && trips[start].fusable_with(&trips[end]) {
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
        // Byte-identical to `group_ranges`: the cap is per kind.
        let cap = if trips[start].kind.run_may_be_chunked() {
            g
        } else {
            usize::MAX
        };
        while end < n && cnt < cap && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        ok &= start == expect;
        // ⛔ THE SIZE INVARIANT IS NOW PER KIND TOO, and a GATHERED `PageFold` run has NO size bound — it
        // must be one group at any length (`GroupKind::run_may_be_chunked`). Keeping `<= g` here would make
        // the proof contradict the walk it is the twin of.
        ok &= (end - start) <= cap; // group ≤ its kind's cap
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
        // THE SAME PER-KIND CAP AS THE WALK. It cannot change this predicate's answer — the run only
        // extends while `fusable_with` holds, and that requires an equal request — but a third copy of
        // the walk that drifts from the other two is how the cap came to be missing from one of them.
        let cap = if trips[start].kind.run_may_be_chunked() {
            g
        } else {
            usize::MAX
        };
        while end < n && cnt < cap && trips[start].fusable_with(&trips[end]) {
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
                // ⛔ THE GATHER STATE TRAVELS IN THE KIND, TAKEN FROM THE OP THAT DECLARED IT — the same
                // field `launch_index` puts on the group's `KvShifts`, so the size cap and the runtime's
                // `gathered` flag cannot come to disagree about which fold this is.
                GroupKind::PageFold {
                    gathered: e.kv_gathered,
                }
            } else if e.kv_gathered {
                GroupKind::Gathered
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
                    GroupKind::Slot {
                        req: TripRequest(e.kv_request),
                    }
                }
            } else {
                GroupKind::Pure
            };
            kinds.push(Trip::at_part(k, TripRequest(e.kv_request), e.kv_part));
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
            if let GroupKind::PageFold { gathered } = t.kind {
                // Reclassified, but it KEEPS ITS REQUEST — so it still cannot fuse across requests.
                // ⭐⭐⭐⭐⭐ AND THE FUSED VARIANT FOLDS AWAY THE PER-PAGE RELAUNCH, NOT THE ENTRY
                // SEMANTICS. A fold op whose gather reads ABSOLUTE stick-blocks (`kv_gathered`,
                // carried here in the kind by `trip_kinds_and_owner` from the paged-fold stamp's
                // chain of custody) is a `Gathered` run: `launch_index` derives the group's
                // `KvShifts.gathered` from this kind, and the runtime's page-base composition
                // asks that bit — a gathered fold rewritten to `Pure` composes the page base on
                // top of the operand's own pool base and reads one page past its keys, the same
                // two-bases defect the copy-leg stamp cures, through the fused twin. `Gathered`
                // fuses only with `Gathered` (`fusable_kind_with`), so the fold's gathered ops
                // leave the neighbouring `Pure` run — the launch-count cost the `FoldGrouping`
                // doc already records for the split variant, paid here for correctness. An
                // UNGATHERED fold stays `Pure` exactly as before, and `page_fold` stays `false`
                // for either: the fused variant still never re-launches.
                t.kind = if gathered {
                    GroupKind::Gathered
                } else {
                    GroupKind::Pure
                };
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
        // ⭐ ONE PAGE PART per group, of ANY kind — AS A TYPE, NOT AN ASSERT. The runtime resolves the
        // group's page once, at the group's part, so an op of another part would land on the wrong
        // page: a cache write nothing downstream can detect. That guard used to be an `assert!` here,
        // re-reading the owner op's field against every member's — two derivations that could only be
        // proven to agree by running them. There is ONE derivation now: [`Trip::fusable_with`] breaks
        // every run where the part changes, and the shift below reads the group's own first trip's
        // part — the same value the walk compared — so "the members disagree with the shift" is not a
        // state the emitter can construct. What the part carries is [`Trip::part`]'s type.
        let fold_group = matches!(kinds[r.start].kind, GroupKind::PageFold { .. });
        let _ = gi; // launch order is the slice position now, not a field
        // ⭐ THE GROUP'S GATHERED ANSWER, derived ONCE — the KV shift's zero and the staging
        // channel below both key off it, so there is one derivation, not two that could
        // disagree.
        let gathered =
            (fold_group && e.kv_gathered) || matches!(kinds[r.start].kind, GroupKind::Gathered);
        shifts.push(bundle::KvShifts {
            slot_stride_bytes: slot,
            slab_stride_bytes: slab,
            // PAGED: `page_slots` is the write-slot modulus AND the declaration that this bundle is
            // paged — a runtime that does not know the key must refuse rather than apply an
            // absolute-position shift and land past the page.
            page_slots: e.kv_page_slots,
            request: e.kv_request,
            // ⭐ THE GROUP'S PART, OFF THE WALK'S OWN TRIP — the first trip of the run `group_ranges`
            // formed, whose part every other member's was compared against by `fusable_with`. Never
            // the owner op's field: before this was a second read of a second derivation, and the
            // assert that proved the two agreed was a refusal at bake. One derivation, one source.
            part: kinds[r.start].part,
            // `page_fold` marks the group the runtime re-launches once per page.
            page_fold: fold_group,
            // ONE PASS PER PAGE INSTEAD OF PER (REQUEST, PAGE) — only when the fold's kernels actually
            // carry the request axis.
            batched_requests: fold_group && e.kv_batched_requests,
            // ⭐ AND WHETHER THAT AXIS IS OVER A GATHERED SCRATCH — the other half of `fold_plan::collapsed`,
            // and what sends the fold's KV shift to zero. A non-fold group gathers only as a `Gathered`
            // run (the split-KV chunk's copies), which takes the same zero shift and the same staged index.
            // Derived ONCE here: the staging channel below keys off the same answer.
            gathered,
            // ⭐ THE STAGING CHANNEL — whether the table those absolute entries read through is
            // the FILL-STAGED REGION FORM, not the composer's page-list table. Both channels'
            // entries are ABSOLUTE (both take the zero shift above); this bit only discriminates
            // WHO stages the table, so the composer and the forward's index-step door key their
            // page-list answer on `gathered && !region_form`. Stamped by the walk on a region
            // carrier's ops and on fold carriers (folds stage through the same host fill).
            region_form: gathered && e.kv_region_form,
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
//  The op-local LX slot alternation walk — #645 fix (a)'s guard on sharing
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One op-local pin's row in the alternation walk's ledger — what the dump's `bundle_id.json`
/// carries beside the descriptors, and what a same-class pair hunt reads to name a bundle's
/// independent norms.
#[derive(Clone, Debug)]
pub struct LxSlotRow {
    /// The pin's synth name (`{tensor}_{role}`, minted by `BundleLayout::synth_lx_op_local`).
    pub name: String,
    /// The pin's footprint class in bytes — the class key, so two rows of one class share a slot.
    pub class_bytes: u64,
    /// The shared slot's start address (top-down from `LX_PIN_END`).
    pub slot: u64,
    /// Every program that touches the pin, in walk order: `(program name, read or wrote)`.
    /// The owner is the first `false` (a write) — the op whose decomposition minted the pin.
    pub touched_by: Vec<(String, bool)>,
}

/// ⭐⭐⭐ THE OP-LOCAL SLOT ALTERNATION WALK — the build-time guard that same-class LX pins
/// SHARE a slot only where execution order makes the sharing sound (#645's fix (a), step 3 of 3).
///
/// Every op-local pin of one footprint class resolves to ONE slot address
/// (`BundleLayout::synth_lx_op_local`), so a name's bytes survive exactly until the next
/// same-class name's program runs. Sharing is therefore sound iff, per class, the touching
/// names form CONTIGUOUS RUNS over the bundle's execution order — `x…x y…y` alternates the
/// slot cleanly, `x…y…x` reads `y`'s bytes as `x`'s. This walk proves that, at build, per
/// launch group over its programs in order:
///
///   * **one program, one name per class** — a program touching two same-class pins is
///     refused outright. One program's descriptors are ONE SDSC phase, and the citation
///     pack schedules per node, one phase each — so two same-class pins in one program
///     live in the SAME phase, with no phase boundary between their touches for a shared
///     slot to alternate across. (Impossible in every emission today — each rmsnorm's
///     `rinv` is touched only by its own program — guarded generically so a future op
///     that reads two norms' pins in one program refuses here, not on card.)
///   * **per-name contiguity** — once a name's run has been closed by a same-class sibling,
///     touching it again is a refusal naming both names and the class. The run state carries
///     ACROSS the bundle's launch groups: the groups launch in order on one stream, so a
///     sibling in an earlier group poisons a later group's re-read exactly as an intra-group
///     one does.
///   * **both directions, by name** — every pin this bundle TOUCHES is written by ≥1 program
///     and read by ≥1 program, both through named `arg_bindings`. A read with no named write
///     is a producer the emission lost, or one that addresses the pin through a RAW OFFSET no
///     walk over `arg_bindings` can see; a write with no read is dead scratch against the
///     window. Scoped to touched pins — the layout is rung-shared, so other bundles' pins sit
///     untouched in this bundle's table and are not this walk's to judge.
///
/// The ORDER this walk guards is the EMITTED order — the order of `sdsc_ops` as lowered, cut
/// into programs by `per_prog` and into launch groups by the SAME partition [`launch_index`]
/// and [`render_dxp_input`] use (`trip_kinds_for` + `group_ranges`; `group_ranges` is
/// Kani-proven an exact covering partition, and contiguous, so emission order is group order).
/// ⛔ IT DOES NOT GUARD DEVICE ORDER, AND CANNOT: whether the device really executes programs
/// complete-before-start in declaration order rests on the #645 citation pack (dxp compiles a
/// group's programs one at a time, `SchedulerStages.cpp:29-38`; program order = declaration
/// order = combined-packet order, `ProgramLayout.cpp:126-127`, `PlacePrograms.cpp:78-79`; the
/// vendor stream contract is complete-before-start, `runtime_stream_mode.hpp:25-33`) — and the
/// CARD LEG is the measurement of that premise, not this walk.
///
/// Touches are derived PURELY from each op's `arg_bindings` — the same name-keyed consult
/// `mark_lx_residency` resolves allocations with — so the walk adds no emission state, keeps
/// every emitted byte identical, and has no log for the declare pass to drain. A bundle with
/// no op-local pins (the overwhelmingly common case) returns an empty ledger unchanged.
///
/// On a refusal the fallback is to SPLIT THE CONFLICTING CLASS — give the returning name its
/// own slot — never to un-share globally; the walk itself only refuses.
pub fn walk_lx_slot_alternation(
    sdsc_ops: &[EmittedOp],
    per_prog: &[(String, usize)],
    fold: FoldGrouping,
    layout: Option<&BundleLayout>,
) -> Result<Vec<LxSlotRow>, String> {
    // The pin table: name → (class bytes, slot address). Empty ⇒ nothing shares ⇒ Ok, no cost.
    // Bound to a local so the borrowed keys outlive the collect — the accessor hands back owned
    // rows (`Vec<(String, u64, u64)>`), and the walk reads them by reference from here on.
    let table = layout.map(|l| l.lx_op_local_pins()).unwrap_or_default();
    let pins: std::collections::BTreeMap<&str, (u64, u64)> = table
        .iter()
        .map(|(name, fp, slot)| (name.as_str(), (*fp, *slot)))
        .collect();
    if pins.is_empty() {
        return Ok(Vec::new());
    }
    // The op-local names each op touches, straight off its arg bindings.
    let touches: Vec<std::collections::BTreeSet<&str>> = sdsc_ops
        .iter()
        .map(|e| {
            e.arg_bindings
                .iter()
                .map(|b| b.buffer.as_str())
                .filter(|n| pins.contains_key(n))
                .collect()
        })
        .collect();
    // Program index per op, off the SAME per-program table the dump writes (`per_prog` is
    // (program name, descriptor count) in lowering order, and `sdsc_ops` is exactly its
    // concatenation — see `ktir_groups_via_superdsc`'s loop).
    let mut prog_of: Vec<usize> = Vec::with_capacity(sdsc_ops.len());
    for (pi, (_, n)) in per_prog.iter().enumerate() {
        prog_of.extend(std::iter::repeat_n(pi, *n));
    }
    debug_assert_eq!(
        prog_of.len(),
        sdsc_ops.len(),
        "per_prog's descriptor counts must sum to sdsc_ops"
    );
    // THE partition launch_index and render_dxp_input both use — same kinds, same group walk.
    let (kinds, owner) = trip_kinds_for(sdsc_ops, fold);
    let groups = group_ranges(&kinds, group_size());
    // Run state per class: the name currently holding the slot, and the names whose runs have
    // been closed by a sibling. Carried ACROSS groups: the bundle's launches are ordered.
    let mut run: std::collections::BTreeMap<u64, &str> = std::collections::BTreeMap::new();
    let mut closed: std::collections::BTreeMap<u64, std::collections::BTreeSet<&str>> =
        std::collections::BTreeMap::new();
    // The ledger, accumulated in walk order.
    let mut touched_by: std::collections::BTreeMap<&str, Vec<(String, bool)>> =
        std::collections::BTreeMap::new();
    for r in &groups {
        // This group's programs, in order: maximal runs of one program over the group's trips.
        // A program cut across a group boundary is two programs — one per compile unit — which
        // is exactly how dxp sees it (a group is one directory, one compile invocation).
        let mut seq: Vec<(usize, std::collections::BTreeSet<&str>)> = Vec::new();
        for ti in r.clone() {
            let o = owner[ti];
            let p = prog_of[o];
            if !seq.last().is_some_and(|(q, _)| *q == p) {
                seq.push((p, Default::default()));
            }
            for &n in &touches[o] {
                let is_input = sdsc_ops[o]
                    .arg_bindings
                    .iter()
                    .any(|b| b.buffer.as_str() == n && b.is_input);
                // Ledger row, deduplicated: one entry per (program, direction) per pin.
                let v = touched_by.entry(n).or_default();
                if !v
                    .last()
                    .is_some_and(|(s, i)| s == &per_prog[p].0 && *i == is_input)
                {
                    v.push((per_prog[p].0.clone(), is_input));
                }
                seq.last_mut().expect("pushed above").1.insert(n);
            }
        }
        for (p, names) in &seq {
            let prog = &per_prog[*p].0;
            // ONE PROGRAM, ONE NAME PER CLASS: two same-class pins in one program have no
            // phase boundary between them that any citation covers.
            let mut by_class: std::collections::BTreeMap<u64, Vec<&str>> = Default::default();
            for &n in names {
                by_class.entry(pins[n].0).or_default().push(n);
            }
            for (fp, same) in &by_class {
                if same.len() > 1 {
                    return Err(format!(
                        "program '{prog}' touches {} same-class op-local LX pins — {} — of the \
                         {fp}-byte class at slot {}: within one program there is no program-order \
                         boundary for the shared slot, so the later name would read the earlier's \
                         bytes. Split the conflicting class (own slot per name) or keep each \
                         class's names in separate programs; never un-share globally.",
                        same.len(),
                        same.iter()
                            .map(|n| format!("'{n}' (slot {})", pins[n].1))
                            .collect::<Vec<_>>()
                            .join(", "),
                        pins[same[0]].1,
                    ));
                }
            }
            // PER-NAME CONTIGUITY: a name re-touched after a sibling closed its run reads the
            // sibling's bytes — refuse, naming both.
            for &n in names {
                let (fp, slot) = pins[n];
                let closed_names = closed.entry(fp).or_default();
                if closed_names.contains(n) {
                    let sibling = run.get(&fp).copied().unwrap_or("<closed>");
                    return Err(format!(
                        "op-local LX pin '{n}' of the {fp}-byte class (slot {slot}) is touched \
                         again by program '{prog}' after '{sibling}' of the same class took the \
                         slot — the touches of one class must form contiguous runs over the \
                         bundle's launch groups, or the returning name reads the sibling's bytes. \
                         Split the conflicting class (own slot per name) or fold '{n}'s uses into \
                         its own run; never un-share globally."
                    ));
                }
                match run.get(&fp) {
                    // The name's own run continues — nothing to close.
                    Some(&m) if m == n => {}
                    // A sibling's run opens: close the old name's, start this one's.
                    Some(&m) => {
                        closed_names.insert(m);
                        run.insert(fp, n);
                    }
                    None => {
                        run.insert(fp, n);
                    }
                }
            }
        }
    }
    // ⭐ THE DIRECTION GUARD — every op-local pin THIS BUNDLE touches is touched in BOTH
    // directions: at least one program WRITES it and at least one READS it, both through named
    // arg bindings (the only consult this walk has). A pin read but never written is a producer
    // the emission lost — or one that addresses the pin through a RAW OFFSET instead of a named
    // binding, which no walk over `arg_bindings` can see; this refusal turns both into a build
    // failure instead of a card-time read of stale slot bytes. A pin written but never read is
    // dead scratch minted against the window budget. Scoped to the pins this bundle TOUCHES,
    // not every pin in the table: the layout is RUNG-shared (the declare pass pre-declares
    // prefix, bodies and suffix alike), so other bundles' pins sit untouched in this bundle's
    // table and are not this walk's to judge.
    for (&name, touches) in &touched_by {
        let (fp, slot) = pins[name];
        if !touches.iter().any(|(_, is_input)| !is_input) {
            return Err(format!(
                "op-local LX pin '{name}' of the {fp}-byte class (slot {slot}) is READ by this \
                 bundle but no program WRITES it: its producer is missing, or it addresses the pin \
                 through a raw offset instead of a named arg binding this walk can see. Every pin \
                 this bundle touches must have one producer and one consumer, both by name."
            ));
        }
        if !touches.iter().any(|(_, is_input)| *is_input) {
            return Err(format!(
                "op-local LX pin '{name}' of the {fp}-byte class (slot {slot}) is WRITTEN by this \
                 bundle but no program READS it: its consumer is missing, or it addresses the pin \
                 through a raw offset instead of a named arg binding this walk can see — dead \
                 scratch minted against the LX window. Every pin this bundle touches must have one \
                 producer and one consumer, both by name."
            ));
        }
    }
    Ok(pins
        .into_iter()
        .map(|(name, (fp, slot))| LxSlotRow {
            name: name.to_string(),
            class_bytes: fp,
            slot,
            touched_by: touched_by.remove(name).unwrap_or_default(),
        })
        .collect())
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
    /// ⭐ CONTENT KEY over exactly what dxp will read, in read order — the bake queue's memo key and
    /// the persistent cache's. Two groups with the same key compile to the same bytes. It covers
    /// [`Self::explore`] too: the same json compiled with and without the explorer is different
    /// output.
    pub key: crate::dxp_cache::GroupDigest,
    /// Whether dxp runs its chunk-parameter explorer on this group — decided by the group's own trips.
    pub explore: crate::superdsc_bake::ChunkExplore,
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
    // Pre-unroll every op into its concrete trips, in `ops`-order then trip-order. The json is
    // rendered per GROUP below, because the key's step index is the WITHIN-GROUP position.
    let mut trips: Vec<(&str, SdscOp)> = Vec::new();
    for e in ops {
        for t in concrete_trips(e) {
            trips.push((&e.op_name, t));
        }
    }
    let (kinds, _owner) = trip_kinds_for(ops, fold);
    let mut out = Vec::new();
    for (gi, r) in group_ranges(&kinds, group_size()).iter().enumerate() {
        // Top-level key carries the step index `{li}_{op_name}` (mirroring torch-spyre's
        // `{idx}_{opfunc}`), which is what dxp's ModuleStitcher orders by — without an index:
        // `DtException: expecting a valid entry in core schedule` (ModuleStitcher.cpp:216).
        //
        // ⭐ THE INDEX IS THE WITHIN-GROUP POSITION, NOT THE BUNDLE'S GLOBAL ONE. A group is dxp's
        // whole compile unit (one directory per invocation), so the within-group order is the
        // complete ordering information — MEASURED: renumbering a group's keys to 0-based compiles
        // to BYTE-IDENTICAL `init_binary.bin`/`spyrecode.json` (group sizes 1/24/128/331, same
        // pod, same compiler), and the full-bake binary-sha gate covers every group shape after.
        //
        // ⛔ AND THE GLOBAL INDEX WAS FORCING REDUNDANT COMPILES. It made the memo's content key
        // differ for groups whose input was identical but sat at a different position in the
        // bundle, so the rung ladder's repeated projections recompiled per position. MEASURED on
        // gemma-4-26b-a4b-it-parity-tiny: 783 groups, 431 unique under the global index, 357 under
        // the per-group one — 17% of the surviving compiles were position-shifted duplicates.
        let mut group_json: Vec<String> = Vec::with_capacity(r.end - r.start);
        for (li, (name, trip)) in (r.start..r.end).map(|ti| &trips[ti]).enumerate() {
            let key = format!("{li}_{name}");
            let one: BTreeMap<&str, &SdscOp> = BTreeMap::from([(key.as_str(), trip)]);
            // COMPACT, not pretty. Nothing reads these by eye — dxp parses them — and a 12B model emits
            // ~230k of them across the rung ladder. MEASURED on a real gemma-4 op: 32073 B pretty vs
            // ~9.7 KB compact, so indentation was ~2/3 of every byte written and ~2/3 of the per-file time.
            group_json.push(
                serde_json::to_string(&one)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            );
        }
        let explore =
            crate::superdsc_bake::ChunkExplore::of_group((r.start..r.end).map(|ti| &trips[ti].1));
        let mut files: Vec<(String, String)> = Vec::with_capacity(group_json.len() + 1);
        let mut names: Vec<String> = Vec::with_capacity(group_json.len());
        for (li, j) in group_json.into_iter().enumerate() {
            let name = format!("sdsc_{li}.json");
            files.push((name.clone(), j));
            names.push(name);
        }
        files.push(("bundle.mlir".to_string(), bundle_mlir(&names)));
        // ⭐ THE KEY HASHES THE COMPILER'S INPUT — every file of the group, names included, in the
        // order dxp reads them. Hashing anything else would let two different inputs collide, and
        // hashing the directory afterwards would re-read what we just wrote. SHA-256, because the
        // key also names a compiled group on disk across builds ([`crate::dxp_cache`]).
        //
        // ⛔ AND THE EXPLORER SETTING IS COMPILER INPUT — the same json compiles to a different
        // program with and without it, and it is no longer a process-wide setting the cache's
        // `DxpIdentity` can see. So it enters the group's own digest, for EVERY group: a store
        // written before the setting was per group holds explorer-off programs, and a `Run` group
        // keyed as before would read one back and keep the regression on every warm build.
        let key = crate::dxp_cache::GroupDigest::of_compile(&files, explore.digest_tag());
        out.push(GroupInput {
            group: gi as u32,
            bytes: files.iter().map(|(_, c)| c.len()).sum(),
            key,
            explore,
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
    write_rendered_dxp_input(dir, &render_dxp_input(ops, fold)?)
}

/// Write ALREADY-RENDERED groups into `dir/group_{gi}/` — the write half of [`write_dxp_input`],
/// split out because the bake has TWO consumers of one render: this artifact dump and the staging
/// loop. Rendering twice doubled the serializer's share of every bake for a dump that is I/O-cheap
/// (MEASURED on the bake pod: 0.05 ms per 10 KB file — the render, not the write, was the cost).
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
        weight_tail,
        kernel_weights,
        scalarmul_scales,
        scalarmul_scales_df,
        weight_scale_folds,
        weight_copies,
        synth,
        // EMIT-ONLY, deliberately not baked: the arrangement authority is the build-time check that a
        // tensor has ONE device layout (`declare_arrangement` returns a build `Err` naming the tensor
        // otherwise). Its verdict is that the bundle compiled; there is nothing for the runtime to do
        // with the table.
        arrangements: _,
        // EMIT-ONLY, deliberately not baked AS PLACEMENTS: pinned-LX intermediates have no HBM
        // placement by construction (their `memOrg_` is lx-only and no L3 movement is generated for
        // them), so there is no address for the runtime to stage — the pin lives in the compiled
        // sdsc allocates. The pin SUM is a different fact and IS baked (`lx_op_local_bytes` below):
        // it is the host's term of the one `lx_window_fits` budget, and the host cannot re-derive
        // it — the slot map is this layout's, gone when the bake returns.
        lx,
        kv_request_stride_bytes,
        // ⭐ THE HOST-STAGED STEP INPUTS' DISCOVERY TABLE + STAGING LAW, BAKED. The placement pass
        // recorded each stated input's slot and the element format its program's own view states;
        // here that format becomes the staging channel — WHICH runtime narrowing the host's bytes
        // take — and a format with no channel is a `cargo build` panic (this runs at bake), not a
        // launch-time guess. The runtime resolves slots through the baked table and never
        // recomputes the producer's `tensors.len()+k` numbering.
        step_inputs,
        // EMIT-ONLY: the gather reach is the gather door's compile-time bound — the pool a pooled
        // table's entry run is refused past at build. The runtime's own pool budget and the host's
        // `GatherEntry` composition are the value-side authorities; the baked artifact carries the
        // placements those entries address, not the bound that checked them.
        gather_reach: _,
        // EMIT-ONLY too: the class registries are build-time facts the door and the load-time bind
        // resolve through; the baked artifact carries each class's PLACEMENT (in `places` above),
        // which is all the runtime needs.
        rope_class_hds: _,
        attn_class_hds: _,
        // EMIT-ONLY: the intermediate-tid cursor is spent by the time any bundle bakes — its whole
        // job is done at mint time (bundle-unique tids, above every placement). The baked artifact
        // carries the minted RESULTS (the synth placements above), not the minter.
        inter_cursor: _,
        // ⛔ THE PENDING KV BLOCK INDEX — SPENT OR REFUSED, never silently dropped (checked just
        // below). The reservation is placed (or this bundle never gathered) by the time any
        // bundle bakes; a `Some` here means a walk forgot
        // [`BundleLayout::place_kv_block_index_last`]. The placement's PRESENCE is
        // `BakeFacts::gathers_kv`, and a gather that finds no table reads entry 0 — a real
        // address — so an unspent reservation is a `cargo build` panic (this runs at bake),
        // never a silent `gathers_kv: false`.
        kv_block_index,
        // EMIT-ONLY: the paged-fold region's page ceiling is the WALK's input — the fact the
        // fold family's tables and mask state their extents against. The baked artifact carries
        // those extents themselves (the step-input placements below), and the runtime's fold
        // passes answer "how many" from the launch group's own declaration, never by re-reading
        // a ceiling — so there is nothing for the bake to carry.
        paged_fold_region_pages: _,
    } = l;
    assert!(
        kv_block_index.is_none(),
        "the KV block index reservation reached the bake unplaced — the walk forgot \
         BundleLayout::place_kv_block_index_last, and the placement's presence is \
         BakeFacts::gathers_kv: a gather with no table reads entry 0"
    );

    let mut places: Vec<bundle::Placement> = placements
        .iter()
        .map(|(tid, p)| bundle::Placement {
            id: bundle::PlaceId::Act(*tid),
            segment: p.segment as u32,
            bank: p.bank,
            offset: p.offset,
            size: p.size,
            is_logits: p.role == SegRole::Logits,
            // ⭐ THE WRITE-COVER RIDES ALONG — minted at the placement's own construction
            // site, where its geometry was in hand; the runtime's bind doors read it here.
            cover: p.cover,
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
            // ⭐ FULL — a synthetic is device-written by the op that minted it; a host bind
            // short of its extent would leave the op's readers on unwritten bytes.
            cover: ktir_superdsc::placement::BindCover::Full,
        }
    }));

    // ⛔ EVERY PLACE ONE ID — the bake-time twin of the runtime Executor::load's own refusal, fired
    // here so the defect is a `cargo build` panic rather than a card incident one hour later. The
    // runtime's check (same text, same numbers) was the ONLY guard until 2026-10-20, when granite-2b
    // fp16 hit it at `load_model`: `1310 placements collapse to 1214 distinct ids` — an intermediate
    // minted per NODE from that node's own bindings had landed on other nodes' placements, 96 of
    // them. Duplicates are worse than a count mismatch: the map keeps ONE entry per id, so the other
    // tensor has NO placement at all — the host binds nothing past it and the device reads whatever
    // shares its address. `BundleLayout::take_inter_tid` makes the mint bundle-unique so this cannot
    // fire from a mint again; this check stands for any other route onto a used id.
    {
        let mut seen = std::collections::BTreeSet::new();
        let dupes: Vec<String> = places
            .iter()
            .filter(|p| !seen.insert(p.id))
            .map(|p| p.id.to_string())
            .collect();
        assert!(
            dupes.is_empty(),
            "{} placements collapse to {} distinct ids — {:?} appear more than once. Two \
             tensors share an identity, so one of them has no placement. An intermediate was \
             minted onto a bound tensor's tid — the mint is BundleLayout::take_inter_tid's to \
             make impossible, and the runtime's load check is the same refusal one hour later.",
            places.len(),
            places.len() - dupes.len(),
            dupes
        );
    }

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
    // ⭐ AND THE TAIL VERDICT IS THE PLACEMENTS' OWN. The runtime copies `[0, bytes)` of the named
    // slot into every borrowing session and nothing else, so a verdict that disagreed with where the
    // weights actually sit would hand a session a zero-filled embedding (argmax 0 every token) or
    // copy scratch over its own. Both come from this one pass; this is the proof they agree.
    let placed_tail = weight_tail_of_placements(placements);
    assert_eq!(
        *weight_tail, placed_tail,
        "the segment budget's weight-tail verdict disagrees with the weight placements"
    );

    bundle::BundleLayout {
        segment_bytes,
        weight_bank_bytes: std::borrow::Cow::Owned(weight_bank_bytes.clone()),
        weight_tail: *weight_tail,
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
        // Each scale slot's staging channel, off the registry's own df column — the same
        // fail-closed projection `step_inputs` below runs: a format with no channel is a BUILD
        // PANIC naming the slot, not a launch-time guess. The columns enter parallel (the
        // completion pass keeps them one length), and the mapping is total over the registry.
        scalarmul_scales_stage: {
            assert_eq!(
                scalarmul_scales.len(),
                scalarmul_scales_df.len(),
                "the scale registry's value and format columns are out of parallel — the layout \
                 was built with one column missing"
            );
            std::borrow::Cow::Owned(
                scalarmul_scales
                    .iter()
                    .zip(scalarmul_scales_df.iter())
                    .enumerate()
                    .map(|(i, (v, df))| {
                        crate::wiring::scale_stage_of(
                            *df,
                            &format!(
                                "scale slot {} (value {})",
                                ktir_superdsc::reserved_tids::scalarmul_scale_tid(i),
                                v
                            ),
                        )
                    })
                    .collect(),
            )
        },
        weight_scale_folds: std::borrow::Cow::Owned(weight_scale_folds.clone()),
        weight_copies: std::borrow::Cow::Owned(weight_copies.clone()),
        kv_request_stride_bytes: *kv_request_stride_bytes,
        // The rung's op-local pin slots — final by now (the declare pass minted every pin before
        // any bundle was emitted), and shared by every bundle of the rung, so whichever bundle the
        // host reads it from carries the same budget.
        lx_op_local_bytes: lx.borrow().op_local_bytes,
        // The probability-part pins the chunk form minted beside its gathers — the witness's
        // second core-local term, recorded at the chunk's admission. The host's
        // `chunk_gather_index_table` read passes it beside `lx_op_local_bytes`, so it refuses
        // exactly what the bake refused.
        lx_prob_pin_bytes: lx.borrow().prob_pin_bytes,
        // The discovery table + staging law, tid-sorted (the placement pass kept a BTreeMap): one
        // row per stated input, its element format mapped to the runtime channel that narrows it.
        // ⛔ A FORMAT WITH NO CHANNEL IS A BUILD PANIC, not a silent default: this runs at bake, so
        // a program stating a step input the launch could not stage fails `cargo build` naming the
        // tid — the same fail-closed shape as every other bake-time guard here.
        step_inputs: std::borrow::Cow::Owned(
            step_inputs
                .iter()
                .map(|(&tid, row)| {
                    let stage = match row.df {
                        Df::Fp16 => bundle::StepInputStage::IeeeF16,
                        Df::SenUint32 => bundle::StepInputStage::RawU32,
                        other => panic!(
                            "t{tid}: a host-staged step input whose program states {other:?} has \
                             no staging channel — the runtime narrows IEEE f16 to SEN169 or copies \
                             raw u32 entries, and nothing else. REFUSING to bake a buffer the \
                             launch could not stage."
                        ),
                    };
                    bundle::StepInput {
                        tid,
                        slot: row.slot,
                        stage,
                    }
                })
                .collect(),
        ),
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
    {
        let _pc = crate::pc::Pc::start("audit");
        audit_op_work(ops, "<bundle>");
    }
    // Content-addressed: the same ops at the same grouping ARE the same bundle, so the second emit has
    // nothing to do. The ladder's ceiling rung IS the body, so this fires on every rolled model.
    let fp = {
        let _pc = crate::pc::Pc::start("fp");
        bundle_fp(ops, fold, attn_params)
    };
    if emitted().lock().is_ok_and(|c| c.contains_key(&fp)) {
        return Ok(fp);
    }
    let b = {
        let _pc = crate::pc::Pc::start("inner");
        emit_bundle_inner(ops, layout, fold, attn_params)?
    };
    debug_assert_eq!(
        b.fp, fp,
        "bundle_fp and emit_bundle_inner disagree on the fingerprint"
    );
    if let Ok(mut c) = emitted().lock() {
        c.entry(fp.clone()).or_insert((b, None));
    }
    {
        let _pc = crate::pc::Pc::start("registry");
        crate::pc::count("registry", ops.len() as u64);
    }
    Ok(fp)
}

/// This bundle's fingerprint — the SPLIT hash, with an `f` suffix marking the fused variant.
///
/// ⛔ THE FUSED TWIN'S NAME KEYS OFF THE UNFILTERED OPS. The runtime finds a body's fused twin as
/// `<split fp>f`, so the base hash must be the split body's, computed before `ops_for_grouping` drops
/// anything. Hashing the trimmed list is what made the twin unfindable.
///
/// ⛔⛔⛔ AND IT HASHES THE SPLIT-KV CHUNK, BECAUSE THE OPS CANNOT. `ops` are the pre-expansion programs,
/// and a chunk rung's programs are BIT-IDENTICAL to every other rung's — the attention program sweeps one
/// page, and the chunk width is a bake fact (`BundleAttnParams::chunk_pages`) the door expands. Hashed
/// without it, every rung of the chunk ladder fingerprints alike, the content-addressed early-out in
/// [`emit_bundle`] returns the FIRST rung's body for all of them, and a 16-page context runs a 1-page
/// chunk: clean bake, no fault, every key past slot 256 silently unread. Appended only when present, so
/// every bundle without a chunk keeps its fingerprint byte for byte.
fn bundle_fp(
    ops: &[EmittedOp],
    fold: FoldGrouping,
    attn_params: Option<crate::ktir_superdsc_door::BundleAttnParams>,
) -> String {
    let split = bundle_fingerprint_grouped(ops, FoldGrouping::Split);
    let base = match attn_params.and_then(|p| p.chunk_pages) {
        Some(pages) => format!("{split}c{pages}"),
        None => split,
    };
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
/// chunk's Kᵗ and V gathered into LX (the chunk's LX form in `ktir_superdsc::sdsc_abstract`), the
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
pub fn cover_ladder_allocation(fp: &str) -> Result<(), String> {
    let mut c = emitted()
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

/// Whether the EMITTED bundle `fp` carries a page-fold launch group — the very `KvShifts::page_fold`
/// the runtime loads. Asked of the emitted bundle, not of the program it was emitted from: the
/// attention expands inside [`emit_bundle`], and the fold flag is set by that expansion. `None` when
/// no bundle by that fingerprint was emitted.
pub fn emitted_bundle_folds(fp: &str) -> Option<bool> {
    let c = emitted().lock().ok()?;
    let (b, _) = c.get(fp)?;
    Some(match &b.groups {
        PendingGroups::Ready(groups) => groups.iter().any(|g| g.kv.page_fold),
        PendingGroups::Staged(groups) => groups.iter().any(|(kv, _)| kv.page_fold),
    })
}

/// Borrow the launch groups of the ALREADY-EMITTED bundle named `fp` — the
/// same groups [`drain_emitted_bundles`] will hand the macro's emission, run
/// under the registry lock so nothing is copied or taken. The bake-time
/// [`numeric_gate`] uses this to run the just-baked program; `None` when the
/// fp has no bundle (not emitted, or emitted but still waiting on a device
/// compiler — the `-Fspyre-hw` shape, where there is nothing to gate).
pub fn with_emitted_groups<R>(
    fp: &str,
    f: impl FnOnce(&[bundle::LaunchGroup<'static>]) -> R,
) -> Option<R> {
    let c = emitted().lock().ok()?;
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

    let fp = bundle_fp(ops, fold, attn_params);
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
pub struct RolledBody {
    /// ONE layer's ops, baked at the class's layer-0 offsets.
    pub ops: Vec<EmittedOp>,
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

impl RolledBody {
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
pub struct RolledSuperDsc {
    pub prefix: Vec<EmittedOp>,
    /// ONE entry per attention class, in the order [`bundle::RerollMeta::bodies`] bakes.
    pub bodies: Vec<RolledBody>,
    pub suffix: Vec<EmittedOp>,
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
    /// Per-launch KV-segment offset, same contract.
    pub launch_kv: Vec<u64>,
    /// Bytes of ONE pool page — every layer of every class (`Σ_c n_c·stride_c`). 0 when
    /// there is no paged KV.
    pub kv_page_stride: u64,
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

impl RolledSuperDsc {
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
pub fn unroll_layers(rolled: &RolledSuperDsc) -> Vec<EmittedOp> {
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
            if let Some(k) = op.ktir.as_mut() {
                for b in k.bindings.iter_mut() {
                    if b.get() == hidden_in {
                        // Launch 0's hidden-in is the prefix's own output; every later one is the
                        // previous launch's hidden-out.
                        if let Some(prev) = prev_hidden {
                            *b = ktir_superdsc::ktir_node::BufferId::new(prev);
                        }
                        continue;
                    }
                    // A tensor with no per-layer list is layer-INVARIANT (a shared constant, the
                    // mask, a source): the body names the one tensor there is, at every iteration.
                    if let Some(ids) = rolled.per_layer.get(&b.get())
                        && let Some(id) = ids.get(iter)
                    {
                        *b = ktir_superdsc::ktir_node::BufferId::new(*id);
                    }
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
        if let Some(k) = op.ktir.as_mut()
            && let Some(last) = last_hidden
        {
            for b in k.bindings.iter_mut() {
                if b.get() == rolled.suffix_in_tid {
                    *b = ktir_superdsc::ktir_node::BufferId::new(last);
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
pub fn covering_twin_groups(
    rolled: &RolledSuperDsc,
) -> Result<Option<(Vec<bundle::LaunchGroup<'static>>, GateTrunc)>, SuperDscError> {
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
    let groups = ktir_groups(&ops, FoldGrouping::Split)?;
    Ok(Some((groups, trunc)))
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

/// ⭐ THE STAMP — ONE FUNCTION, and the walk's call is the production path. Everything the
/// splice states about a carrier's KV (`kv_paged_fold`, `kv_paged_region` — each the ONE door
/// for its fact) reaches the descriptors the bake actually groups HERE, copied verbatim onto
/// every op the door minted from that carrier's program. One function so there is one
/// stamping site to audit; [`mint_bundle_ops`]'s call below is the only production
/// caller, and the pin (`prefix_free_prefill_rung_stamps_its_carrier_at`) calls it
/// directly on a door-lowered carrier to check the stamp's own contract — every op, no
/// exceptions, non-empty.
/// Fatal refusals — no rung can survive a broken stamp premise.
fn stamp_carrier_kv_facts(
    carrier: &EmittedOp,
    lowered: &mut Vec<EmittedOp>,
    layout: Option<&BundleLayout>,
) -> Result<(), SuperDscError> {
    // ⭐ THE PAGED-FOLD STAMP — the CHAIN-OF-CUSTODY step. The splice states the fold on the
    // program's EmittedOp (`kv_paged_fold`, the ONE door: no other site may set it); THIS is
    // where it reaches the descriptors the bake actually groups, copied VERBATIM onto every
    // op the door minted from that program — the walk derives nothing, the stamp recomputes
    // nothing, and `emitted_bundle_folds`'s proof reads what this wrote. A program that
    // states no fold stamps nothing, so every non-fold bundle stays byte-identical.
    // The `kv_region_form` rider: a fold's step inputs are staged by the SAME host fill
    // (`fill_attn_step_inputs`) a region carrier's are, so its groups answer the staging
    // channel `region_form = true` — folds never read the composer's page-list table.
    if let Some(facts) = carrier.kv_paged_fold {
        for o in lowered {
            o.kv_page_fold = true;
            o.kv_batched_requests = facts.batched_requests;
            o.kv_gathered = facts.gathered;
            o.kv_region_form = true;
            o.kv_fold_rows = facts.fold_rows;
            o.kv_page_slots = facts.page_slots;
        }
    }
    // ⭐ THE PAGED-REGION STAMP — the one-shot attention chain of custody, the same shape as
    // the fold's. The splice states the fill-form region on the program's EmittedOp
    // (`kv_paged_region`, the ONE door); THIS is where it reaches the descriptors the bake
    // actually groups. The addressing fact (`kv_gathered` — the entries are ABSOLUTE
    // stick-blocks, the op takes the ZERO KV shift, never the page base on top) and the
    // staging channel (`kv_region_form` — the table is host-fill-staged, not
    // composer-composed) are stamped together onto EVERY op the door minted from that
    // carrier's program: the door materializes the gather as legs + dots + softmax inside
    // the one carrier, the KV shift only touches seg2 operands, and the whole attention
    // group takes the zero shift — the per-leg granularity the retired door stamps carried
    // was a grouping optimization, not a correctness requirement. A program that states no
    // region stamps nothing.
    //
    // ⛔ THE SLOT/SLAB GUARD — the stamp's soundness, ENFORCED, not assumed. The region
    // stamp is sound because the attention program carries no runtime-composed seg2 shift
    // of its own: it has no slot writes (the cache write is the door's RopeAppend arm, a
    // SEPARATE program that self-stamps its slot facts) and its only SEG_KV readers are
    // the pool gathers (absolute entries — the second half of the soundness statement is
    // what `kv_gathered` means). An op with a runtime-composed slot/slab shift inside this
    // carrier would have that shift zeroed too — every new key landing on the wrong slot,
    // silently — so a future kernel that fuses the cache write into attention must hit
    // THIS refusal, not the card. (The else-if order above already makes a mixed
    // fold+region carrier impossible: `kv_paged_fold` is checked first.)
    else if let Some(region) = carrier.kv_paged_region {
        // ⛔⛔ (a') THE REACH GUARD — the stamp's premise, CHECKED, not assumed. The stamp
        // is sound because the entries the carrier gathers are ABSOLUTE stick-blocks, and
        // "absolute" is a fact of a POOL the layout placed a `gather_reach` for — the same
        // reach the placement loop inserts for these tids (`pool_reach_tids`, the ONE
        // spelling both sites derive through). A carrier whose pool carries no reach is a
        // graph that is not paged the way the mint assumed — a future non-paged attention
        // form reaching the one-shot arm — and stamping it would zero the KV shift over an
        // UNPAGED read, silently. It refuses here, at the build, instead.
        let reach_tids = region.reach_tids();
        match layout {
            Some(l)
                if reach_tids
                    .iter()
                    .all(|tid| l.gather_reach.contains_key(tid)) => {}
            Some(_) => {
                return Err(SuperDscError(format!(
                    "[spyre-superdsc] {}: the paged-region carrier's pool (k=t{}, v=t{}) \
                     carries no gather reach in this bundle's layout — the region stamp gives \
                     every op of the carrier the ZERO KV page shift because the entries are \
                     ABSOLUTE stick-blocks, and absolute entries are a fact of a pool the \
                     placement covered. A graph that placed no reach for these tids is not \
                     paged; stamping it would zero the shift over an unpaged read. Refusing.",
                    carrier.op_name, region.k_tid, region.v_tid
                )));
            }
            None => {
                return Err(SuperDscError(format!(
                    "[spyre-superdsc] {}: a paged-region carrier arrived with no bundle \
                     layout to check its pool's gather reach against — the region stamp's \
                     premise (a placed pool) has no artifact behind it on this path. \
                     Refusing.",
                    carrier.op_name
                )));
            }
        }
        if let Some(o) = lowered
            .iter()
            .find(|o| o.slot_stride_bytes > 0 || o.slab_write)
        {
            return Err(SuperDscError(format!(
                "[spyre-superdsc] {}: a paged-region carrier may not carry the \
                 slot/slab-shifted op '{}' (slot_stride_bytes={}, slab_write={}): the \
                 region stamp gives every op of the carrier the ZERO KV page shift, which \
                 would corrupt a runtime-composed slot/slab shift. The cache write is a \
                 separate program; keep it there.",
                carrier.op_name, o.op_name, o.slot_stride_bytes, o.slab_write
            )));
        }
        for o in lowered {
            o.kv_gathered = true;
            o.kv_region_form = true;
        }
    }
    Ok(())
}

/// What the walk's mint hands the rest of the walk: every descriptor it lowered, in program
/// order, and each program's name with its descriptor count (the dump's `per_program` and the
/// LX slot walk read the second).
struct MintedBundle {
    ops: Vec<EmittedOp>,
    per_prog: Vec<(String, usize)>,
}

/// The walk's mint, extracted verbatim so the stamp's custody is testable WITHOUT the bake: a test
/// can run this, then the walk's own pure consumers (`launch_index`, `render_dxp_input`), and count
/// trips — the same counting the walk itself groups by — while never touching the bake queue, the
/// staging root, or dxp. The walk (`ktir_groups_via_superdsc`) calls this and then runs its guards,
/// render, and staging exactly as before.
fn mint_bundle_ops(
    ops: &[EmittedOp],
    layout: Option<&BundleLayout>,
    attn_params: Option<crate::ktir_superdsc_door::BundleAttnParams>,
) -> Result<MintedBundle, SuperDscError> {
    // KTIR → SuperDSC, in memory. One symbol counter for the WHOLE bundle, exactly as the SubtileIR
    // walk threads it: a per-op restart would alias addresses across ops in one bundle.
    let mut sym_id_base: i64 = 0;
    let mut sdsc_ops: Vec<EmittedOp> = Vec::new();
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
        let _pc_cls = crate::pc::Pc::start(crate::pc::mint_class(&e.op_name));
        let mut lowered =
            crate::ktir_superdsc_door::lower(k, &mut sym_id_base, layout, attn_params).map_err(
                |err| SuperDscError(format!("{}: KTIR -> SuperDSC: {}", e.op_name, err.message)),
            )?;
        drop(_pc_cls);
        crate::pc::count(crate::pc::mint_class(&e.op_name), lowered.len() as u64);
        stamp_carrier_kv_facts(e, &mut lowered, layout)?;
        // ⭐ THE CACHE-WRITE FACTS NEED NO STAMP SITE — the append's recipe
        // (`kv_cache_write`, called from the door's `Program::RopeAppend` arm)
        // SELF-STAMPS every op it mints (`kv_page_slots`, `kv_request`,
        // `kv_part`, the slot stride), which is what `latch_paged_geometry`
        // reads off the final ops. A carrier-borne stamp copied here would be a second
        // stamping site for one fact; there is none.
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
    Ok(MintedBundle {
        ops: sdsc_ops,
        per_prog,
    })
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
    let _pc_mint = crate::pc::Pc::start("mint");
    let MintedBundle {
        ops: sdsc_ops,
        per_prog,
    } = mint_bundle_ops(ops, layout, attn_params)?;

    // ⛔⛔⛔ main's FOUR POST-LOWERING GUARDS, AT THE GRAIN WHERE THE DESCRIPTORS NOW EXIST.
    //
    // All four are `lower_graph_to_superdsc`/`emit_bundle_inner`'s own text, and all four went SILENT
    // when the lowering moved here: they read `EmittedOp::op` and `EmittedOp::time`, which main's walk
    // had already filled by the time they ran. On this path the ops reaching `emit_bundle_inner` carry
    // a KTIR program and NOTHING ELSE — `EmittedOp::bare` sets `op: None` and `time: 1` — so the
    // aliasing check's `e.time > 1` was false for every op in every bundle, and the three dxp guards
    // had no `dscs_` to walk at all. A guard that cannot fire is not a guard; `sdsc_ops` below is the
    // list main's guards were written against.
    {
        let _pc = crate::pc::Pc::start("guards");
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
    } // pc:guards

    // ⭐ THE OP-LOCAL SLOT ALTERNATION WALK — #645 fix (a)'s guard on same-class sharing, at the
    //    grain where programs and their descriptors now exist. Refuses a bundle whose same-class
    //    op-local pins would interleave over one shared slot (the walk's own doc states the two
    //    rules and the exact premise split: emitted order here, device order on the citations,
    //    the card leg as the measurement). Also THE LEDGER's one computation — the dump below
    //    writes it beside the descriptors, and the same-class pair hunts read it from there.
    let lx_ledger =
        walk_lx_slot_alternation(&sdsc_ops, &per_prog, fold, layout).map_err(SuperDscError)?;

    // ── THE BAKE BUDGET, PER BUNDLE (`BakeBudget::bundle_verdict`) ─────────────────────────
    //    A bundle is the unit the emitter fans out, so its counts bound BOTH the wave's critical
    //    path and every serial path's per-bundle cost. Refused here, after the correctness guards
    //    but before the render: a bundle past the budget costs nothing to refuse, and a bundle
    //    that is also incorrect should name the correctness defect, not the size.
    {
        let trips: u64 = sdsc_ops.iter().map(|o| o.time.max(1) as u64).sum();
        crate::superdsc_bake::BakeBudget::bundle_verdict(fp, sdsc_ops.len(), trips)
            .map_err(SuperDscError)?;
    }

    // ⭐ ONE RENDER, TWO CONSUMERS. `render_dxp_input` serialises every trip of the bundle, and it
    // is the dearest per-op emitter cost after the mint: the artifact dump below and the staging
    // loop below THAT each used to pay a FULL pass of it (the dump via `write_dxp_input`'s internal
    // render), while the writes themselves are I/O-cheap — MEASURED on the bake pod: 0.05 ms per
    // 10 KB file, so the serializer was effectively the whole cost. Both consumers take this one
    // pass. A cardless host still pays it exactly once, for the dump, as it did before; a dxp host
    // pays it once instead of twice.
    let rendered = {
        let _pc = crate::pc::Pc::start("render");
        render_dxp_input(&sdsc_ops, fold).map_err(|e| SuperDscError(e.to_string()))?
    };

    // ── THE BAKE BUDGET, PER GROUP (`BakeBudget::group_verdict`) ── the bake-side
    //    oversized-dxp-group guard the leaf crate's `DxGroupCeiling` doc promises for THIS
    //    function. A group is what dxp compiles and its cost is superlinear in size (8,300
    //    descriptors ≈ 2.5 h, 4,183 ≈ 57 min), the emitter's window-chunk split already
    //    re-partitions over-ceiling gathered folds, and the granite-8b baseline's own largest
    //    group (4,204) sits AT the ceiling — so a group past it here is either a non-fold kind
    //    the split cannot touch or a split that stopped short, and the bake refuses instead of
    //    compiling for hours behind a build that looks hung. One `sdsc_{li}.json` file per
    //    trip; `bundle.mlir` is the +1 that is not a descriptor.
    for g in &rendered {
        let descriptors = g.files.len() - 1;
        crate::superdsc_bake::BakeBudget::group_verdict(fp, g.group, descriptors)
            .map_err(SuperDscError)?;
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
    {
        let _pc = crate::pc::Pc::start("dump");
        let dump = std::path::Path::new("/tmp/superdsc-dump").join(fp);
        if let Err(e) = write_rendered_dxp_input(&dump, &rendered) {
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
        // The segment budget's verdict rides the identity for the same reason `mq` does: on a green
        // build the bake's own SEGMENT BUDGET line is never printed, and a spilled tail is a load-time
        // and device-memory cost that nothing else at build time names.
        let weight_tail = match layout.map(|l| l.weight_tail).unwrap_or_default() {
            WeightTail::InPlace => "\"in_place\"".to_string(),
            WeightTail::Spilled { segment, bytes } => {
                format!("{{\"spilled\":{{\"segment\":{segment},\"bytes\":{bytes}}}}}")
            }
        };
        // ⭐ THE OP-LOCAL PIN LEDGER rides the identity for the same reason everything above does:
        // a green build prints nothing, and the #645 same-class pair hunts (which bundle holds two
        // independent same-class norms, and which programs touch them) have nowhere else to read
        // the walk's view from. One row per op-local pin — name, class bytes, slot address, and
        // the programs touching it with their direction — straight off the alternation walk's
        // ledger, so the dump cannot disagree with the guard that ran.
        let lx_pins = lx_ledger
            .iter()
            .map(|r| {
                format!(
                    "{{\"name\":\"{}\",\"class_bytes\":{},\"slot\":{},\"touched_by\":[{}]}}",
                    r.name,
                    r.class_bytes,
                    r.slot,
                    r.touched_by
                        .iter()
                        .map(|(p, i)| format!("{{\"program\":\"{p}\",\"input\":{i}}}"))
                        .collect::<Vec<_>>()
                        .join(","),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let id_json = format!(
            "{{\"fp\":\"{fp}\",\"mq\":{mq},\"weight_tail\":{weight_tail},\"programs\":{},\
             \"descriptors\":{},\"per_program\":[{}],\"lx_pins\":[{}]}}\n",
            ops.len(),
            sdsc_ops.len(),
            progs
                .iter()
                .map(|(n, c)| format!("{{\"op\":\"{n}\",\"descriptors\":{c}}}"))
                .collect::<Vec<_>>()
                .join(","),
            lx_pins,
        );
        if let Err(e) = std::fs::write(dump.join("bundle_id.json"), &id_json) {
            eprintln!("  [superdsc-dump] {fp}: could not write bundle_id.json: {e}");
        }
        eprintln!(
            "[ktir->sdsc] BUNDLE {fp}: {} KTIR program(s) -> {} descriptor(s)",
            ops.len(),
            sdsc_ops.len(),
        );
    } // pc:dump

    // ── from here down this is main's block, verbatim ──
    let shifts = {
        let _pc = crate::pc::Pc::start("launch");
        launch_index(&sdsc_ops, fold)
    };
    let mut groups: Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)> = Vec::new();
    let mut staged: u64 = 0;
    // No dxp (a Mac / cardless `cargo check`) ⇒ nothing to stage, and nothing to stage it FOR. This
    // is a CAPABILITY probe, not a behaviour flag.
    if let Some(bake) = crate::superdsc_bake::global() {
        for g in &rendered {
            let id = crate::superdsc_bake::GroupId {
                fp: fp.to_string(),
                group: g.group,
            };
            let kv = *shifts.get(g.group as usize).ok_or_else(|| {
                SuperDscError(format!(
                    "{fp}: group {} has dxp input but no entry in the launch index ({} entries) — \
                     `render_dxp_input` and `launch_index` disagree about the group partition",
                    g.group,
                    shifts.len(),
                ))
            })?;
            // ⭐ ALREADY IN HAND — compiled earlier in this build, mid-compile for an identical group,
            // or by a previous build of the same emission ([`crate::dxp_cache`]): the group never
            // reaches the staging root, the budget, or a `dxp_standalone` child, and its program is
            // filed under `id` by the same content law that lets the memo skip a twin.
            if bake.adopt(&id, g.key) {
                crate::pc::count("cachit", 1);
                groups.push((kv, id));
                continue;
            }
            {
                let _pc = crate::pc::Pc::start("stage");
                let gdir = bake.stage().group_dir(fp, g.group as usize);
                bake.reserve(g.bytes);
                std::fs::create_dir_all(&gdir).map_err(|e| SuperDscError(e.to_string()))?;
                for (name, contents) in &g.files {
                    std::fs::write(gdir.join(name), contents)
                        .map_err(|e| SuperDscError(e.to_string()))?;
                }
                bake.submit(crate::superdsc_bake::SealedGroup::of_rendered(
                    gdir,
                    id.clone(),
                    g,
                ))
                .map_err(SuperDscError)?;
            }
            staged += g.bytes as u64;
            groups.push((kv, id));
        }
    }
    crate::pc::bundle_line(fp, ops.len(), sdsc_ops.len(), staged);
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
    // The rung's own facts — the one-request flag and the chunk it bakes.
    // See [`RungFacts`].
    rung: RungFacts,
) -> Result<RolledSuperDsc, SuperDscError> {
    crate::lower_subtile_tape_to_ktir::lower_subtile_tape_to_ktir(
        tape,
        ir,
        weight_ids,
        active_cap,
        rows_are_requests,
        rung,
    )
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

    /// ⭐⭐⭐ THE PROBE'S WEIGHTS ARE FULL ROW RANK — the gate measures the EMULATOR's rounding
    /// floor, and a rank-deficient probe would measure its own. The old law
    /// (`(id·131 + j·7) % 197`) is a period-197 sawtooth in the FLAT index `j = r·cols + c`,
    /// so every synthesized weight — the 3840×15360 MLP projections included — had rank ≤ 197:
    /// branch outputs cancelled toward zero, and a sandwich-norm arch (gemma-4's
    /// post-feedforward layernorm) renormalized the rounding-dominated remainder back to unit
    /// RMS before the residual add, doubling a 0.02 floor per layer to O(1) by L23 while
    /// granite (pre-norm) stayed flat at 0.0152 — the probe amplifying the very error it
    /// exists to measure. 256 rows is past the old law's 197-row ceiling, so this test fails
    /// under any modulus-arithmetic regression and passes under the splitmix avalanche.
    #[test]
    fn the_probe_synthesizes_full_row_rank_weights() {
        let rows = 256usize; // > 197: the old law's rank ceiling
        let cols = 512usize;
        let tid = 7usize; // any source id; the law must not depend on it
        let mut m: Vec<Vec<f64>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| dbg_synth_val(tid, r * cols + c) as f64)
                    .collect()
            })
            .collect();
        // Rank by Gaussian elimination with partial pivoting.
        let mut rank = 0usize;
        for c in 0..cols {
            let piv = (rank..rows)
                .max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))
                .expect("rank < rows inside the loop, so the range is non-empty");
            if m[piv][c].abs() < 1e-9 {
                continue;
            }
            m.swap(rank, piv);
            let (head, tail) = m.split_at_mut(rank + 1);
            let pivot = &head[rank];
            for row in tail {
                let f = row[c] / pivot[c];
                for (d, s) in row[c..].iter_mut().zip(&pivot[c..]) {
                    *d -= f * s;
                }
            }
            rank += 1;
            if rank == rows {
                break;
            }
        }
        assert_eq!(
            rank, rows,
            "the probe weight is rank {rank} of {rows} — a rank-deficient probe cancels its own \
             branch outputs and the gate measures the probe, not the emulator"
        );
    }

    /// ⭐⭐⭐ AN INDEX PAST ITS REGION IS A REFUSAL, NOT AN ALIAS. Every reserved tid used to be
    /// raw `u32` subtraction from a base — `SCALARMUL_SCALE_BASE - idx`, unchecked — so the 100th
    /// scale silently took `ksplit_block_tid(0)`'s slot. Two tensors, one tid, one placement: the
    /// second producer overwrites the first's bytes and the first's consumer reads them. On a
    /// device with no stack to attach to that is wrong output or a hang, reported by nothing.
    #[test]
    #[should_panic(expected = "reserved tid region overflow")]
    fn an_index_past_its_region_refuses_instead_of_aliasing_the_next_one() {
        // The 101st scale is past scalarmul's floor. Asked BY NAME — see
        // `the_declared_bases_are_the_regions_bases` for what a positional index cost here.
        let _ = scalarmul_scale_tid(reserved_region("scalarmul_scale").slots as usize);
    }

    /// ⛔⛔⛔ EVERY DECLARED BASE IS ITS OWN **NAMED** REGION'S BASE — and this test previously
    /// asserted the opposite, which is why the defect it now catches was invisible.
    ///
    /// It read `assert_eq!(KCT_RESIDENT_BASE, RESERVED_REGIONS[2].base)`. Both sides were the same
    /// POSITION, so inserting the `kv_block_index` region at 2 moved the constant AND the expectation
    /// together and the test stayed green — while `kct_resident_tid(0)` had begun answering
    /// `KV_BLOCK_INDEX_TID`, i.e. layer 0's resident Kᵗ kernel and the gather's index table sharing one
    /// placement. A test whose expectation is derived the same wrong way as the code cannot see the
    /// code being wrong (`a-green-test-can-pin-a-port-divergence-as-correct`).
    ///
    /// The bake DID fail — `slots: 1` made `k_id >= 1` overflow — so nothing was silently shipped. That
    /// was luck, not this test.
    #[test]
    fn the_declared_bases_are_the_regions_bases() {
        assert_eq!(
            SCALARMUL_SCALE_BASE,
            reserved_region("scalarmul_scale").base
        );
        assert_eq!(KCT_RESIDENT_BASE, reserved_region("kct_resident").base);
        // ⭐ AND THE TWO DOORS THAT SHARE A NEIGHBOURHOOD MUST NOT MEET. This is the assertion the
        // positional form could not make: it compares two INDEPENDENTLY derived tids rather than one
        // constant against its own definition.
        assert_ne!(
            KctResident::of(0).wire_tid(),
            KV_BLOCK_INDEX_TID,
            "the resident Kᵗ kernel of layer 0 and the gather's index tensor are ONE tid"
        );
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
        // ⛔ #197 MOVED THE WORD INSIDE THE FOLD MANAGER: `data_` is now
        // `fold_manager_const(bits, folds)` — an object whose value lands at
        // `["[0, 0, 0]"][0]` as a DECIMAL STRING (the FoldManager deque dtype), not
        // the bare array this test read before. The call site below missed that
        // change (#197 updated the emitter, not this test), which is why this
        // module had not compiled since.
        let folds = SdscFoldSet::new(f32_spec.iter.cores_used());
        let ci = scaling_factor_const_fp32((1.0f32 / 576.0).to_bits(), &folds);
        assert_eq!(ci["0"]["dataFormat_"], "IEEE_FP32");
        let got = ci["0"]["data_"]["data_"]["[0, 0, 0]"][0]
            .as_str()
            .and_then(|s| s.parse::<u64>().ok())
            .expect("the fold manager holds the word as a decimal string");
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
            tensor: TensorId::from_index(t),
            region: ts[t].whole(),
        };
        let nodes = vec![SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::MatmulTile {
                n: 16,
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
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
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
        let (sdsc, _) = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
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
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs {
            one: scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[("mb", scratchy_subtile::superdsc_opspec::Scale::RedNonStick)],
                broadcast_batch: true,
            },
        });
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
        let (sdsc, _) = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
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
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs {
            one: scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[("mb", scratchy_subtile::superdsc_opspec::Scale::RedNonStick)],
                broadcast_batch: true,
            },
        });
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
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs {
            one: scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[],
                broadcast_batch: false,
            },
        });
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let (sdsc, _) = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
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
        // OUTPUT to match `bmm.ddl`'s bias/resadd convention (same layout bucket as the real
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
        // which is what lets prefill produce the first generated token's logits itself.
        //
        // ⭐ THE PRODUCER HALF IS THE SPLICE'S, AND IT HAS LANDED: `lower_all` routes an m>1
        // vocab-wide MatmulTile through the Triton `lmlast.py` extraction + the re-lowered m=1
        // matmul. This test pins three things: (1) the m>1 tail LOWERS to the fold's two programs
        // (an extraction named `lmlast_s{id}` and the m=1 matmul after it — never a silent m>1
        // lowering of a vocab-wide matmul); (2) the m=1 DECODE bundle stays a single bare matmul —
        // the fold never fires at m=1; (3) the CONSUMER half of the fold
        // (`lower_ktir_to_superdsc::lmlast`, main's copy loop verbatim) still refuses a nameless
        // extraction — pinned through the splice's own fold program below.
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
                tensor: TensorId::from_index(t),
                region: ts[t].whole(),
            };
            let node = SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    n: vocab,
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
                .map(|(ops, _)| ops.into_iter().map(|e| e.op_name).collect::<Vec<_>>())
        };
        // m>1 (prefill): the tail FOLDS — the splice's `lmlast.py` extraction (its own program,
        // over the reserved `LAST_HIDDEN_TID` staging) plus the re-lowered m=1 matmul, never a
        // silent m>1 lowering of a vocab-wide matmul (the wrong-token defect the fold exists to
        // prevent). Through the door the extraction is `hidden/64` per-stick copies named after
        // the tail's own output tid (`lmlast{j}_o2`), then the m=1 matmul `matmul_o2`.
        let mq = 8u32;
        let names = lower(mq).expect("the m>1 lm-head tail folds to m=1");
        assert_eq!(
            names.len() as u32,
            h / 64 + 1,
            "the fold is the per-stick copies plus the m=1 matmul: {names:?}"
        );
        for (j, name) in names.iter().take(h as usize / 64).enumerate() {
            assert_eq!(
                name,
                &format!("lmlast{j}_o2"),
                "copy {j} is named after the tail's own output tid"
            );
        }
        assert_eq!(
            names.last().map(String::as_str),
            Some("matmul_o2"),
            "the fold's last program is the m=1 matmul"
        );
        // m==1 (decode): the SAME node lowers to exactly one bare matmul — the fold never fires, so
        // the decode bundle is byte-identical to the pre-fold emitter.
        assert_eq!(
            lower(1).expect("the m=1 decode lm_head lowers"),
            vec!["matmul_o2".to_string()]
        );

        // ⛔ NEGATIVE CONTROL FOR THE NAME'S OWN FACT, through the splice's own fold.
        // `lmlast` names its copies from `KtirNode::node_out_tid`, and a MISSING one must
        // REFUSE — not fall back to the program's own output, which is the reserved
        // `LAST_HIDDEN_TID` and is exactly the wrong name this test pins. Strip the fact off
        // the extraction program and the door must Err naming it.
        let (node, ir) = build(mq);
        let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .expect("the tail's layout mints");
        let mut programs = scratchy_triton_splice::lower_all(&node, &ir, false)
            .expect("the splice's fold mints the tail");
        let extract = programs
            .iter_mut()
            .find(|e| e.op_name.starts_with("lmlast"))
            .expect("the tail emits an extraction program");
        extract.ktir.as_mut().expect("its KTIR").node_out_tid = None;
        let mut sym = 0i64;
        let stripped = crate::ktir_superdsc_door::lower(
            extract.ktir.as_ref().unwrap(),
            &mut sym,
            Some(&layout),
            None,
        );
        let why = match stripped {
            Err(e) => e.message,
            Ok(_) => panic!("a nameless extraction must refuse"),
        };
        assert!(
            why.contains("node_out_tid"),
            "the refusal must name the missing fact, got {why:?}"
        );
    }

    /// ⭐ THE fp8 K-PAD RESERVES THE PADDED ROWS IN THE WEIGHT PLACEMENT. A dense fp8 weight
    /// whose logical k is not a whole 128-elem stick (gemma-4-26b's dense GeGLU DOWN: k=2112 =
    /// 16.5 sticks) has its packed retile and its worker-staged rebuild both widened to
    /// k_pad=2176 — the placement must reserve the SAME rows or the matmul's k_pad-wide tile
    /// read overruns it into the next tensor's bytes. This is the 26b decode bake's first
    /// refusal, verbatim: `t35: access offset 0B + 6127616B exceeds its placement footprint
    /// 5947392B (seg1)` — the access IS the padded extent; the footprint was the raw one.
    /// An fp16 weight of the same shape stays raw (2112 = 33 whole fp16 sticks — the pad is
    /// fp8-keyed by the producing arity-3 MatmulTile, never a width census).
    #[test]
    fn fp8_half_stick_weight_placement_reserves_the_k_pad() {
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // The 26b dense GeGLU DOWN's own geometry: k=2112 (16.5 fp8 sticks → k_pad=2176), n=2816
        // (22 sticks, `bump_sticks_to_splittable`-inert: core_split(44,32)=22 ≥ the util floor).
        let (m, k, n) = (1u32, 2112, 2816);
        let build = |weight: GemmWeight| -> SubtileIR {
            let tensors = vec![
                TensorShape { rows: m, cols: k }, // t0 = activation
                TensorShape { rows: k, cols: n }, // t1 = W
                TensorShape { rows: 1, cols: n }, // t2 = w_scale (fp8 arity-3 only)
                TensorShape { rows: m, cols: n }, // t3 = out
            ];
            let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
                tensor: TensorId::from_index(t),
                region: ts[t].whole(),
            };
            let mut inputs = vec![whole(0, &tensors), whole(1, &tensors)];
            if matches!(weight, GemmWeight::Fp8Dynamic) {
                inputs.push(whole(2, &tensors));
            }
            let node = SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile { n, weight },
                inputs,
                output: whole(3, &tensors),
            };
            SubtileIR {
                tensors,
                num_sources: 3,
                nodes: vec![node],
                result: TensorId::from_index(3),
                // Hand-authored fixture: there is no source op list to be the
                // provenance of, so the map is empty.
                op_output: Vec::new(),
            }
        };
        let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        // fp8 (arity-3): the placement reserves the PADDED extent — 2176·2816·1 B, the exact
        // bytes the matmul's tile read addresses (the refusal's access side).
        let ir = build(GemmWeight::Fp8Dynamic);
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .expect("the fp8 layout mints");
        let p = layout.placements.get(&1).expect("the fp8 weight is placed");
        assert_eq!(
            p.size, 6_127_616,
            "k=2112 fp8 reserves k_pad=2176 rows × 2816 cols × 1 B"
        );
        // fp16 (arity-2), SAME shape: raw rows — 2112 = 33 whole fp16 sticks, no pad.
        let ir = build(GemmWeight::Dense);
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .expect("the fp16 layout mints");
        let p = layout
            .placements
            .get(&1)
            .expect("the fp16 weight is placed");
        assert_eq!(
            p.size, 11_894_784,
            "the same shape at fp16 stays raw: 2112 rows × 2816 cols × 2 B"
        );
    }

    #[test]
    fn fp8_shared_activation_quantizes_per_matmul_until_the_dedup_is_threaded() {
        // granite decode emits q/k/v = gemm(normed, ·) — THREE arity-3 fp8 matmuls reading the SAME
        // activation (and gate/up = gemm(normed2, ·) — two more). The per-token activation quantize
        // (square→amax→scale→clamp→qfp8ch) is a PURE function of the activation, independent of the
        // weight, so it is worth emitting ONCE and sharing — and the per-`Program` door did exactly
        // that, threading one bundle-wide `quantized` set through every fp8 matmul it lowered.
        //
        // ⛔ PR-A'S REROUTE DROPPED THAT DEDUP, AND THE RULING STATES IT RATHER THAN SILENTLY
        // LOSING IT: the six easy families now lower through `whole_function::lower_function`,
        // which lowers ONE op at a time and owns no bundle-wide set — so each fp8 matmul quantizes
        // the shared activation itself. The door's own doc carries the same statement
        // ("⛔ THE fp8 ACTIVATION-QUANTIZE DEDUP IS NOT THREADED THROUGH THE WALK, AND THAT IS
        // STATED RATHER THAN SILENT"), and the follow-up threads the set where it is consumed.
        // Until then this test pins WHAT IS: per-matmul chains (fq_afp8_op/fq_qspt_op/fq_clip_op
        // == the matmul count), each still the FUSED per-op chain its name states — a silent
        // revert to the unfused 4-op/2-op forms must still fail below.
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
            tensor: TensorId::from_index(t),
            region: ts[t].whole(),
        };
        let nodes = vec![
            SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    n,
                    weight: GemmWeight::Fp8Dynamic,
                },
                inputs: vec![whole(0, &tensors), whole(1, &tensors), whole(2, &tensors)],
                output: whole(5, &tensors),
            },
            SubtileNode {
                id: SubtileId::from_index(1),
                op: SubOp::MatmulTile {
                    n,
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
            quantizes, 2,
            "until the bundle-wide dedup is threaded through the walk, each of the two matmuls \
             quantizes the shared activation itself — and this count is the STATED regression the \
             door's doc names, not a silent one; got {quantizes}"
        );
        let matmuls = ops.iter().filter(|o| o.op_name.ends_with("fq_mm")).count();
        assert_eq!(
            matmuls, 2,
            "each fp8 matmul still emits its OWN matmulfp8 (weight differs), got {matmuls}"
        );
        // The FIRST chain op likewise rides per-matmul now (the same stated regression): the fused
        // per-token scale (`fq_qspt_op` — `quantscalepertokenfp8`), which subsumed the `abs`
        // (`fq_absx_op`) that subsumed the `square` (`fq_sq_op`) this test originally named.
        //
        // ⛔ THE NAME IS THE ONLY THING THAT MOVES HERE, AND AN OUT-OF-DATE NAME MAKES THIS ASSERT
        // VACUOUS IN THE PASSING DIRECTION — it would be counting zero of something that no longer
        // exists, which is exactly how the `fq_sq_op` spelling survived. So it is pinned to the op the
        // chain ACTUALLY starts with, and the count stays the matmul count (per-op, until the dedup
        // returns).
        let first_chain_op = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_qspt_op"))
            .count();
        assert_eq!(
            first_chain_op, 2,
            "the fused per-token activation scale rides each matmul's own chain until the dedup is \
             threaded, got {first_chain_op}"
        );
        // ⭐ AND THE OP IT REPLACED MUST BE GONE, so a silent revert to the 4-op chain cannot leave this
        // test green: `fq_absx_op` counting 1 and `fq_qspt_op` counting 1 would both hold if BOTH were
        // emitted, and the assert above cannot see that.
        let replaced = ops
            .iter()
            .filter(|o| {
                o.op_name.ends_with("fq_absx_op")
                    || o.op_name.ends_with("fq_amax_op")
                    || o.op_name.ends_with("fq_amaxfl_op")
                    || o.op_name.ends_with("fq_ascale_op")
            })
            .count();
        assert_eq!(
            replaced, 0,
            "the 4 ops `quantscalepertokenfp8` subsumes must no longer be emitted, got {replaced}"
        );
        // ⭐ SAME LAW FOR THE CLAMP TAIL: the fused `clip` rides each matmul's own chain (per-op,
        // the same stated regression), and the `minimum`/`maximum` pair it subsumes must be gone —
        // the same vacuous-green trap, so a silent revert to the 2-op clamp cannot leave this green
        // either.
        let clip = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_clip_op"))
            .count();
        assert_eq!(
            clip, 2,
            "the fused ±448 clamp rides each matmul's own chain until the dedup is threaded, got \
             {clip}"
        );
        let clamp_pair = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_chi_op") || o.op_name.ends_with("fq_cl_op"))
            .count();
        assert_eq!(
            clamp_pair, 0,
            "the 2 ops `clip` subsumes must no longer be emitted, got {clamp_pair}"
        );
    }

    /// ⭐ THE SCALARMUL WEIGHT FOLD, END TO END through the real walk — granite's own residual
    /// shape (`add(oproj · residual_multiplier, hidden)`), at the decode dims that shape runs,
    /// at an UP-multiplier: the one class the f16 floor gate permits, because `|s| >= 1` can
    /// only move a staged value away from the subnormal floor (the companion test below pins
    /// the refusal side). One lowering answers four questions at once:
    ///   * the ScalarMul is GONE from the tape (no `scalarmul` program, no registry slot);
    ///   * the fold names the PRODUCER MATMUL'S WEIGHT (t1), the recognition's own target;
    ///   * the folded node's output placement ALIASES the producer's bytes — the reader of the
    ///     scaled product reads the matmul's output buffer, which is the whole mechanism by
    ///     which an op-less node's consumers still find their data;
    ///   * the two matmuls themselves are untouched (the fold deletes exactly one program).
    #[test]
    fn the_residual_scalarmul_folds_into_the_weight_and_off_the_tape() {
        let scale = 2.0f32; // an UP-multiplier — the f16 floor gate's permitted class
        let (ir, weight_ids) = residual_chain_fixture(scale);
        let (ops, layout) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
            .expect("the folded residual shape lowers");

        // (a) the ScalarMul is GONE — no program carries its name, and the walk emitted the two
        // matmuls and nothing else. ⛔ THE COUNT IS THE ANTI-VACUOUS HALF: a name filter alone
        // stays green if the splice renames the op; the total pins that exactly one program was
        // deleted by the fold.
        let scalarmul_programs = ops
            .iter()
            .filter(|o| o.op_name.contains("scalarmul"))
            .count();
        assert_eq!(
            scalarmul_programs,
            0,
            "the folded ScalarMul must not become a device program: {:?}",
            ops.iter().map(|o| &o.op_name).collect::<Vec<_>>()
        );
        assert_eq!(
            ops.len(),
            2,
            "the fold deletes exactly the ScalarMul's program — two matmuls remain: {:?}",
            ops.iter().map(|o| &o.op_name).collect::<Vec<_>>()
        );

        // (b) the fold is in the LAYOUT, naming the producer's weight (t1).
        assert_eq!(
            layout.weight_scale_folds,
            vec![(1u32, scale)],
            "the fold targets W (t1), the producer matmul's last input"
        );

        // (c) the folded node's output placement ALIASES the producer's bytes — same segment,
        // same offset, same size — which is how the consumer of the scaled product reads data
        // no op wrote. The producer placement itself is what the alias must equal, read from
        // the layout rather than restated here.
        let producer = &layout.placements[&3u32];
        let folded = &layout.placements[&4u32];
        assert_eq!(
            (folded.segment, folded.offset, folded.size),
            (producer.segment, producer.offset, producer.size),
            "the folded node's output (t3) must read the producer's bytes (t2)"
        );

        // (d) the folded scale LEFT the registry — no `[1,1]` const bind exists for it, so the
        // registry this graph's programs read holds nothing for a program that no longer is.
        assert!(
            !layout
                .scalarmul_scales
                .iter()
                .any(|s| s.to_bits() == scale.to_bits()),
            "the folded multiplier must not occupy a registry slot: {:?}",
            layout.scalarmul_scales
        );
    }

    /// ⛔ THE F16 FLOOR GATE — the SAME graph at granite's actual DOWN-multiplier refuses the
    /// fold and keeps the multiplier on the tape. The landing `f16(ws·s)` is unprovable at
    /// bake (staged ws is checkpoint data, unbounded below), and a subnormal landing sheds
    /// mantissa bits the device codec does not restore — measured on granite fp8 as a 0.29%
    /// o-side seed amplifying to 10.2% at the logits and a different answer. The refusal is
    /// the mirror of every assertion the folded test above makes:
    ///   * the ScalarMul STAYS a program — three ops, one of them the pointwise mul;
    ///   * the fold list is EMPTY — no fold the staged bytes cannot carry at full precision;
    ///   * the node's output placement is its OWN — no alias onto the producer's bytes, so
    ///     the op that exists writes the buffer its reader reads;
    ///   * the multiplier is BACK in the registry — the surviving program reads its const.
    #[test]
    fn a_downscale_fold_refuses_at_the_f16_floor_and_stays_on_the_tape() {
        let scale = 0.130_417_02f32; // granite's residual_multiplier, as it appears in the dsl
        let (ir, weight_ids) = residual_chain_fixture(scale);
        let (ops, layout) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
            .expect("the refused residual shape still lowers");

        // (a) the ScalarMul STAYS — three programs, exactly one of them the pointwise mul.
        // ⛔ THE COUNT IS THE ANTI-VACUOUS HALF, same law as the folded test's.
        let scalarmul_programs = ops
            .iter()
            .filter(|o| o.op_name.contains("scalarmul"))
            .count();
        assert_eq!(
            scalarmul_programs,
            1,
            "the refused fold keeps the ScalarMul program: {:?}",
            ops.iter().map(|o| &o.op_name).collect::<Vec<_>>()
        );
        assert_eq!(
            ops.len(),
            3,
            "nothing else is deleted by a refused fold: {:?}",
            ops.iter().map(|o| &o.op_name).collect::<Vec<_>>()
        );

        // (b) the fold list is EMPTY — the gate refused the only candidate.
        assert_eq!(
            layout.weight_scale_folds,
            Vec::<(u32, f32)>::new(),
            "a down-scale fold is not admitted: {:?}",
            layout.weight_scale_folds
        );

        // (c) the node's output placement is its OWN — the reader reads the op's buffer, not
        // an alias of the producer's (the producer placement read from the layout, not
        // restated, same law as the folded test's alias assertion).
        let producer = &layout.placements[&3u32];
        let scaled = &layout.placements[&4u32];
        assert_ne!(
            (scaled.segment, scaled.offset),
            (producer.segment, producer.offset),
            "an unfolded ScalarMul must own its output placement — no alias onto the \
             producer's bytes"
        );

        // (d) the multiplier is BACK in the registry — the surviving program reads its const.
        assert!(
            layout
                .scalarmul_scales
                .iter()
                .any(|s| s.to_bits() == scale.to_bits()),
            "the refused fold's multiplier must occupy a registry slot: {:?}",
            layout.scalarmul_scales
        );
    }

    /// granite's residual chain at decode dims, at whatever multiplier the caller pins:
    /// `t5 = (t0·t1)·s + …` — a matmul into a ScalarMul into a second matmul, with the two
    /// weights as sources. Both f16-floor tests lower this one fixture; only the scale
    /// differs, so the folded and refused shapes cannot drift apart.
    fn residual_chain_fixture(
        scale: f32,
    ) -> (
        scratchy_subtile::subtile_ir::SubtileIR,
        std::collections::HashSet<u32>,
    ) {
        use scratchy_subtile::lower::GemmWeight;
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // granite-3.1-2b decode: hidden 2048, o_proj [2048, 2048] at m=1.
        let (m, k, n) = (1u32, 2048u32, 2048u32);
        // ⛔ SOURCES ARE `0..num_sources` — the caller-filled prefix, so the two weights sit at
        // t1/t2 and every produced tensor follows them.
        let tensors = vec![
            TensorShape { rows: m, cols: k }, // t0 = x (activation source)
            TensorShape { rows: k, cols: n }, // t1 = W (weight source)
            TensorShape { rows: k, cols: n }, // t2 = W2 (the consumer's own weight)
            TensorShape { rows: m, cols: n }, // t3 = x·W
            TensorShape { rows: m, cols: n }, // t4 = (x·W)·s — folded away, or kept by the gate
            TensorShape { rows: m, cols: n }, // t5 = the residual add's output
        ];
        let whole = |t: usize| TensorRegion {
            tensor: TensorId::from_index(t),
            region: tensors[t].whole(),
        };
        let nodes = vec![
            SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    n,
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(0), whole(1)],
                output: whole(3),
            },
            SubtileNode {
                id: SubtileId::from_index(1),
                op: SubOp::ScalarMul { scale },
                inputs: vec![whole(3)],
                output: whole(4),
            },
            SubtileNode {
                id: SubtileId::from_index(2),
                op: SubOp::MatmulTile {
                    n,
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(4), whole(2)],
                output: whole(5),
            },
        ];
        let ir = scratchy_subtile::subtile_ir::SubtileIR {
            tensors,
            num_sources: 3,
            nodes,
            result: TensorId::from_index(5),
            // Hand-authored fixture: there is no source op list to be the provenance of.
            op_output: Vec::new(),
        };
        (ir, [1u32, 2u32].into_iter().collect())
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
            &rb(
                "act",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(384),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(64),
            ),
            &Stk::<KernelTag>::kernel(64, 384, "wt"),
            &rb(
                "out",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(384),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(384),
            ),
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
            &rb(
                "act",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(384),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(64),
            ),
            &Stk::<KernelTag>::kernel(64, 384, "wt"),
            &rb(
                "out",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(384),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(384),
            ),
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
        // A `y`-batched matmul keeps the frontend's LX-fit time trips, and the
        // only one the build admits contracts ONE stick (`matmul_opspec_split`'s refusal). So this is
        // the card-validated batched form of the (ii) bmm, `in = 64` at 16 kernel batches, widened in N
        // until its per-core kernel slab (`in · out · batch`) overflows LX: at most 32 cores leave
        // every core ≥ 1024 `out` columns, 2 MiB of kernel. The same shape UNBATCHED is handed to dxp
        // whole, in one descriptor, which this asserts first.
        let (m, n, k, batch) = (64, 32768, 64, 16);
        let whole = assemble_matmul(
            "matmul_o7",
            m,
            n,
            k,
            1,
            &rb(
                "a",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(m),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(k),
            ),
            &Stk::<KernelTag>::kernel(k as usize, n as usize, "w"),
            &rb(
                "o",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(m),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(n),
            ),
            None,
        );
        assert_eq!(whole.time, 1, "an unbatched fp16 matmul is staged by dxp");
        // The shape this test drove before the refusal, 64×16384×2048 at two kernel batches: a
        // 32-stick contraction under a `y` batch, the form measured incoherent on granite-3.1-8b.
        // It is a build error now, and the error names why.
        let refused = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::try_assemble_matmul_seeded(
            "matmul_o7",
            64,
            16384,
            2048,
            2,
            &rb(
                "a",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(64),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(2048),
            ),
            &Stk::<KernelTag>::kernel(2048, 16384, "w"),
            &rb(
                "o",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(64),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(16384),
            ),
            &mut 0,
            None,
        )
        .err()
        .expect("a y-batched multi-stick contraction must be refused");
        assert!(
            refused.contains(
                "a y-batched (batch 2) shared-kernel matmul contracts 2048 elements, 32 sticks"
            ) && refused.contains("card-validated only at a one-stick contraction"),
            "the refusal names the shape and the reason: {refused}"
        );
        let emitted = assemble_matmul(
            "matmul_o7",
            m,
            n,
            k,
            batch,
            &rb(
                "a",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(m),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(k),
            ),
            &Stk::<KernelTag>::kernel(k as usize, n as usize, "w"),
            &rb(
                "o",
                ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(m),
                ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(n),
            ),
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

    // ⭐ THE WIRING IS A REGISTRY, NOT A PLACEMENT. The old ladder MEASURED this exact wall: the
    // 26b MoE under the fp8 preset wired fine through the BANKED re-rolled bake, then panicked in
    // `graph_wiring` on a throwaway `compute_bundle_layout(&[])` layout nobody read — the unrolled
    // graph has no layer classes, so its budget pass can only refuse any weight segment past one
    // 16 GiB region. This pins BOTH sides of the law on one fixture: the placement pass REFUSES
    // (the budget law lives there and stays), and the wiring SUCCEEDS with the registry intact.
    #[test]
    fn the_wiring_is_a_registry_not_a_placement() {
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // One weight at 32 GiB — 2× the one-region cap, un-movable by the tail spill (no slot is
        // 16 GiB wide): 268,435,456 rows × 64 cols × 2 B = 34,359,738,368 B.
        let tensors = vec![
            TensorShape {
                rows: 268_435_456,
                cols: 64,
            }, // t0 = the over-cap weight (an unused source still packs)
            TensorShape { rows: 1, cols: 64 }, // t1 = activation
            TensorShape { rows: 1, cols: 64 }, // t2 = out
        ];
        let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
            tensor: TensorId::from_index(t),
            region: ts[t].whole(),
        };
        let node = SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::ScalarMul { scale: 2.0 },
            inputs: vec![whole(1, &tensors)],
            output: whole(2, &tensors),
        };
        let ir: SubtileIR = SubtileIR {
            tensors,
            num_sources: 2,
            nodes: vec![node],
            result: TensorId::from_index(2),
            op_output: Vec::new(),
        };
        let weight_ids: std::collections::HashSet<u32> = [0u32].into_iter().collect();
        // The budget pass refuses the same fixture the wiring accepts — on the UNROLLED graph,
        // with no layer classes, an over-cap weight segment is un-bankable and un-spillable.
        let err = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .expect_err("the placement pass is the budget law and must refuse 32 GiB in seg1");
        assert!(
            err.0.contains("one device region can hold"),
            "the refusal is the region-cap law, not some other defect: {err:?}"
        );
        // The wiring builds NO placement — it reads only the registries — so the same fixture
        // wires, and the scalar registry it carries is the pure walk's content.
        let wiring = crate::lower_subtile_tape_to_ktir::graph_wiring(&ir, &weight_ids)
            .expect("the wiring is a registry and never runs the budget pass");
        assert_eq!(
            wiring.scalarmul_scales,
            vec![2.0],
            "the registry-only layout carries the node walk's scales"
        );
        assert_eq!(wiring.nodes.len(), 1, "the ScalarMul node still lowers");
    }

    /// ⭐ THE MINTING BODY, hand-stated — the walk-unit miniature of the spliced bodies that
    /// mint: two parameter loads, one UNSTORED `arith.addf` whose result the terminal consumes
    /// (an intermediate by the walk's own definition — this is exactly mint site 1's shape),
    /// and one stored `arith.mulf` terminal.
    ///
    /// Hand-stated rather than spliced because no family the splice lowers at main leaves an
    /// unstored classified result: the fused chains (silu, gelu, rmsnorm) fold their interiors
    /// into one terminal, so their door arms mint nothing. The mint is reached by the bodies
    /// that carry cross-op results — #717's attention through the whole-function door, and the
    /// parked rope program that hit this on the card — and this body is their two-op
    /// miniature, stated from the same op kinds the walk really reads (the `zz_` roster's law).
    fn minting_program(
        a: u32,
        b: u32,
        out: u32,
        name: &'static str,
    ) -> ktir_superdsc::ktir_node::KtirNode {
        use ktir_core::arena::Arena;
        use ktir_core::attrkey::AttrKey;
        use ktir_core::dtypes::DType;
        use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
        use ktir_core::irtype::IrType;
        use ktir_core::opkind::OpKind;
        use ktir_superdsc::ktir_node::{BufferId, Elementwise, KtirNode, Program};
        let arena: &'static Arena = Arena::global();
        let shape = |op: Operation<'static>, r: i64, c: i64| {
            op.with_attr(arena, AttrKey::Shape, Attr::IntList(arena.ints(vec![r, c])))
                .with_attr(arena, AttrKey::Dtype, Attr::Dtype(DType::F16))
        };
        // The unstored add's result is a MINTED intermediate, so it states a 2-D tensor
        // type — the footprint the walk's minted buffer is sized from (the same law the
        // `zz_` roster's unstored matmuls follow).
        let ty = |r: i64, c: i64| {
            Some(IrType::Tensor {
                dims: arena.ints(vec![r, c]),
                elem: DType::F16,
            })
        };
        // Parameters: a, b, out — the node's operand prefix, output LAST (the splice's own
        // `default_bindings` law, so the fixture states the real binding order).
        let (p_a, p_b, p_out) = (Ssa(0), Ssa(1), Ssa(2));
        let zero = Ssa(3);
        let (v_a, t_a, val_a) = (Ssa(4), Ssa(5), Ssa(6));
        let (v_b, t_b, val_b) = (Ssa(7), Ssa(8), Ssa(9));
        let (sum, prod, v_o, t_o) = (Ssa(10), Ssa(11), Ssa(12), Ssa(13));
        let ops = vec![
            Operation::new(arena, Some(zero), OpKind::ArithConstant, &[]).with_attr(
                arena,
                AttrKey::Value,
                Attr::Int(0),
            ),
            shape(
                Operation::new(arena, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_a]),
                8,
                64,
            ),
            shape(
                Operation::new(
                    arena,
                    Some(t_a),
                    OpKind::KtdpConstructAccessTile,
                    &[v_a, zero, zero],
                ),
                8,
                64,
            ),
            shape(
                Operation::new(arena, Some(val_a), OpKind::KtdpLoad, &[t_a]),
                8,
                64,
            ),
            shape(
                Operation::new(arena, Some(v_b), OpKind::KtdpConstructMemoryView, &[p_b]),
                8,
                64,
            ),
            shape(
                Operation::new(
                    arena,
                    Some(t_b),
                    OpKind::KtdpConstructAccessTile,
                    &[v_b, zero, zero],
                ),
                8,
                64,
            ),
            shape(
                Operation::new(arena, Some(val_b), OpKind::KtdpLoad, &[t_b]),
                8,
                64,
            ),
            // ⭐ THE INTERMEDIATE: `a + b` is consumed by the mul below and stored NOWHERE.
            {
                let mut op = shape(
                    Operation::new(arena, Some(sum), OpKind::ArithAddf, &[val_a, val_b]),
                    8,
                    64,
                );
                op.result_type = ty(8, 64);
                op
            },
            shape(
                Operation::new(arena, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
                8,
                64,
            ),
            shape(
                Operation::new(
                    arena,
                    Some(t_o),
                    OpKind::KtdpConstructAccessTile,
                    &[v_o, zero, zero],
                ),
                8,
                64,
            ),
            shape(
                Operation::new(arena, Some(prod), OpKind::ArithMulf, &[sum, val_b]),
                8,
                64,
            ),
            Operation::new(arena, None, OpKind::KtdpStore, &[prod, t_o]),
        ];
        KtirNode {
            func: IRFunction {
                name,
                arguments: arena.args(vec![
                    (p_a, IrType::Index),
                    (p_b, IrType::Index),
                    (p_out, IrType::Index),
                ]),
                operations: arena.ops(ops),
                grid: (1, 1, 1),
                return_type: None,
            },
            program: Program::Elementwise(Elementwise::Mul),
            bindings: vec![BufferId::new(a), BufferId::new(b), BufferId::new(out)],
            mask: None,
            step_inputs: Vec::new(),
            node_out_tid: None,
            minted: Default::default(),
        }
    }

    /// The two-node bundle's shared layout: six placements `t0..t5` — node A binds
    /// `t0, t1` and writes `t4`, node B binds `t2, t3` and writes `t5`. The `ids` row per
    /// placement is what `compute_bundle_layout` derives; stated here the same way so the
    /// bake's synthetic-identity lookup sees the same table the real layout builds.
    fn two_node_layout() -> ktir_superdsc::placement::BundleLayout {
        use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
        let mut l = BundleLayout::default();
        for tid in 0..6u32 {
            l.ids.borrow_mut().insert(
                ktir_superdsc::place::act_name(tid),
                ktir_superdsc::place::PlaceId::Act(tid),
            );
            l.placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role: SegRole::Activation,
                    segment: 0,
                    bank: 0,
                    offset: u64::from(tid) * 0x10000,
                    size: 8 * 64 * 2,
                    cover: ktir_superdsc::placement::BindCover::Full,
                },
            );
        }
        // The placement high-water of seg0, as `compute_bundle_layout` states it — the bake's
        // address check reads this, so a hand-stated layout states it too.
        l.segment_bytes[0] = 6 * 0x10000;
        l
    }

    /// ⭐⭐ THE CARD P0 OF 2026-10-20, AS A TEST — two whole-function nodes in ONE bundle, and
    /// every intermediate mint must stay above every placement in it.
    ///
    /// The mint used to be seeded per NODE off that node's own bindings. Node A below
    /// (bindings `t0, t1, t4`) therefore minted from `t5` — node B's output, a real placement —
    /// and on the card, granite-2b fp16 refused at `load_model` with "1310 placements collapse
    /// to 1214 distinct ids — t454 … t549 appear more than once" (96 silent aliases per rope
    /// program, each landing on another node's placement exactly as node A's first mint lands
    /// on `t5` here). The bundle-unique cursor (`BundleLayout::take_inter_tid`) mints strictly
    /// above every placement instead, and this is the two-node shape that made the per-node
    /// seed collide.
    ///
    /// ⛔ FAILS WITHOUT THE FIX BY CONSTRUCTION: node A's first mint lands on `t5` — a real
    /// placement — so `bake_layout`'s duplicate-id refusal fires (and the distinctness assert
    /// below would fire even without it).
    #[test]
    fn two_whole_function_nodes_in_one_bundle_mint_above_every_placement() {
        let layout = two_node_layout();
        // The seed's own filter: reserved tids (the const tables sit near `u32::MAX`) are
        // skipped by the max, never walked into.
        let max_real = layout
            .placements
            .keys()
            .copied()
            .filter(|t| !ktir_superdsc::reserved_tids::is_reserved(*t))
            .max()
            .expect("the layout places its tensors");

        // THE NODES the walk lowers — one instance each, because each pass borrows the same
        // emitted program's KTIR (the splice-runs-once shape; a fresh instance per pass is
        // the defect the replay assert below refuses).
        let programs = [
            minting_program(0, 1, 4, "mint_a"),
            minting_program(2, 3, 5, "mint_b"),
        ];
        // One full walk pass over the shared layout, in node order — the declare pass's own
        // shape, which the real pass repeats verbatim.
        let pass = |layout: &ktir_superdsc::placement::BundleLayout| {
            for k in &programs {
                let mut syms = 0i64;
                ktir_superdsc::emit::whole_function::lower_function(
                    k,
                    Some(layout),
                    &mut syms,
                    None,
                )
                .expect("the walk lowers the minting program");
            }
        };

        // THE DECLARE PASS, then the real one — the same two lowerings every rung runs over
        // the ONE shared plan.
        pass(&layout);
        let synth_before = layout.synth.borrow().map.len();
        assert!(
            synth_before >= 2,
            "each node minted at least one intermediate — the fixture no longer exercises \
             the mint ({} minted)",
            synth_before
        );
        // THE REAL PASS REPLAYS: no fresh synthetics, or the body's snapshot would lack them
        // (the DMA-past-the-segment defect the declare pass exists to prevent). This is the
        // replay record's own law — a mint keyed on the SSA value, confirmed by the layout
        // before it is trusted — without which the second pass would mint beside the first
        // pass's buffers instead of on them.
        pass(&layout);
        assert_eq!(
            layout.synth.borrow().map.len(),
            synth_before,
            "the second lowering minted fresh synthetics — the declare/real name agreement broke"
        );
        // And every mint sat strictly above every real placement (max_real = t5 here): the
        // cursor moved past the placements, never through them.
        assert!(
            layout.inter_cursor.get().is_some_and(|c| c > max_real),
            "the mint cursor {:?} did not advance past the highest placement t{max_real}",
            layout.inter_cursor.get()
        );

        // THE BAKE — the runtime's own law (`Executor::load`'s duplicate refusal), at build
        // time: all place ids distinct.
        let baked = bake_layout(&layout);
        let mut seen = std::collections::BTreeSet::new();
        assert!(
            baked.places.iter().all(|p| seen.insert(p.id)),
            "two placements share an identity — the exact card refusal this test pins"
        );
    }

    /// ⛔ THE REFUSAL ITSELF: a layout whose synthetics collide with a real placement is a BUILD
    /// failure, not a card one. The duplicate is injected by hand — declaring a synthetic at an
    /// already-placed tid, exactly what a colliding mint does — because the mint path that used
    /// to produce it is the one the test above pins as fixed.
    #[test]
    #[should_panic(expected = "collapse to")]
    fn bake_layout_refuses_duplicate_placement_ids() {
        let layout = two_node_layout();
        // A synthetic at a REAL placement's tid — t5, node B's output, the tid the per-node
        // seed minted first for node A (A's bindings max at t4, so the old seed was t5).
        let victim = 5u32;
        assert!(
            layout.placements.contains_key(&victim),
            "t{victim} must be a real placement for the collision to be the card's shape"
        );
        layout.synth(ktir_superdsc::place::PlaceId::Act(victim), &[1, 64]);
        let _ = bake_layout(&layout);
    }

    /// ⭐⭐⭐ THE FILL HALF'S BAKE: the discovery table crosses into the runtime artifact, each
    /// row's element format mapped to the staging channel that narrows its bytes. The two formats
    /// today's programs state — an fp16 buffer (the SEN169 narrowing) and a u32 index table (the
    /// raw copy) — bake to their channels, tid-sorted (the placement pass kept a `BTreeMap`, so
    /// the artifact's byte order cannot depend on the walk's program order).
    #[test]
    fn the_bake_maps_each_step_inputs_format_to_its_staging_channel() {
        let mut layout = two_node_layout();
        layout.step_inputs.insert(
            5,
            ktir_superdsc::placement::StepInputRow {
                slot: 0,
                df: Df::Fp16,
            },
        );
        layout.step_inputs.insert(
            2,
            ktir_superdsc::placement::StepInputRow {
                slot: 1,
                df: Df::SenUint32,
            },
        );
        let baked = bake_layout(&layout);
        assert_eq!(
            baked.step_inputs.iter().copied().collect::<Vec<_>>(),
            vec![
                bundle::StepInput {
                    tid: 2,
                    slot: 1,
                    stage: bundle::StepInputStage::RawU32,
                },
                bundle::StepInput {
                    tid: 5,
                    slot: 0,
                    stage: bundle::StepInputStage::IeeeF16,
                },
            ],
            "tid-sorted, each row's format mapped to its channel"
        );
        // And the fill answers through the baked table: slot 1 is t2 at its own placement extent.
        assert_eq!(
            baked.step_input_fill(1, 8 * 64 * 2),
            Ok(bundle::StepInputFill {
                place: bundle::PlaceId::Act(2),
                size: 8 * 64 * 2,
                stage: bundle::StepInputStage::RawU32,
            }),
            "the baked table answers the fill's own question end to end"
        );
    }

    /// ⛔ A FORMAT WITH NO STAGING CHANNEL IS A BUILD PANIC, NOT A SILENT DEFAULT. This runs at
    /// bake, so a program stating a step input the launch could not stage fails `cargo build`
    /// naming the tid — fail-closed, exactly like every other bake-time guard here. A new format
    /// gains a channel and a variant in the same commit, never a fallback.
    #[test]
    #[should_panic(expected = "no staging channel")]
    fn the_bake_refuses_a_step_input_format_with_no_channel() {
        let mut layout = two_node_layout();
        layout.step_inputs.insert(
            5,
            ktir_superdsc::placement::StepInputRow {
                slot: 0,
                df: Df::Fp8,
            },
        );
        let _ = bake_layout(&layout);
    }

    /// ⛔ (b) THE PREFIX-FREE PREFILL RUNG STAMPS ITS CARRIER — the trigger fix's anti-regression
    /// pin, the WALK-STAMP half (the carrier half is `triton_splice_golden`'s door test, which
    /// asserts the mint over every arm and shape). A fresh prompt (start == 0) bakes a
    /// PREFIX-FREE rung — `ActiveCap::NONE`, a chunk ceiling the ladder never stated — so its walk
    /// threads NO fold region, and that is exactly the shape whose carrier the old region-keyed
    /// mint left BARE: the fill-staged absolute entries read with the page base composed on top
    /// (token 0's double-base defect). This test runs the production chain up to the walk —
    /// [`super::lower_graph_to_ktir`] (which threads `paged_fold = None` to every node) → the
    /// step-input placements → the walk's own bundle params — then, for the wiring half, the
    /// walk's own mint ([`super::mint_bundle_ops`]) and the walk's own PURE consumers
    /// ([`super::launch_index`], [`super::render_dxp_input`]), NEVER the walk itself: the walk's
    /// group Vec is filled only inside the bake, so a cardless test through the walk would assert
    /// over nothing, and a bake-side test would start a real bake from inside `cargo test`. The
    /// render is machine-independent — no dxp, no staging, nothing written — so the test runs
    /// cardless exactly where the walk's mint does. It asserts BOTH halves of the fix: the
    /// carrier arrives at the walk stating its region (the prefill arm's own extent contract),
    /// and the walk's minted gathered groups account for EXACTLY the carrier's own trips — the
    /// TRIP-ARITHMETIC DIFFERENTIAL against a carrier-alone run of the same mint (program
    /// identity is not name-readable off the trips: the walk composes op names from roles
    /// and regions, a different name space from the program's own) — the zero-KV-shift
    /// contract whose absence double-based the read.
    ///
    /// The pin's body, parameterized over the rung's WIDTH: `valid_len` states the
    /// one-pass cell's dead prefix (`prefix = valid_len − 1`), the golden cell's own
    /// prefix-free law scaled to the width.
    fn prefix_free_prefill_rung_stamps_its_carrier_at(mq: u32, valid_len: u32) {
        // The judgment's GEOMETRY: granite 2b (32 q-heads, 8 kv-heads, hd 64), a
        // one-page pool (`cap` 256), rope-appended caches — the golden's rope-appended
        // prefix-free prefill cell, restated here through the same `lower_region`
        // entrance so the graph is the production shape, rope tables included (the
        // cos/sin sources below state the K stream's own width, the same real-path
        // staging the golden's `attn_ir` states since its own fix). This fixture was
        // BORN RED on that very shape — its cos/sin first stated `q_width` against
        // the K stream's `kv_width`, and the committed windowing law refused it at
        // the mint (§9 of the campaign proposal carries the record).
        let (nqh, nkvh, hd) = (32u32, 8u32, 64u32);
        let cap = 256u32;
        let scale = 0.015625f32; // granite 2b's attention_multiplier (hd 64, f16-exact)
        let q_width = nqh * hd;
        let kv_width = nkvh * hd;
        let sources = |cache_rows: u32| {
            vec![
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: mq,
                    cols: q_width,
                }, // 0: q
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: cache_rows,
                    cols: kv_width,
                }, // 1: prefix_k
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: cache_rows,
                    cols: kv_width,
                }, // 2: prefix_v
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: mq,
                    cols: kv_width,
                }, // 3: new_k
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: mq,
                    cols: kv_width,
                }, // 4: new_v
                // ⭐ 5/6: cos/sin AT THE K STREAM'S OWN WIDTH — the REAL construction. The
                // real path stages each rope's tables at THAT rope's own full column width —
                // GQA's Q and K differ (`to_wavefront.rs`: "Cos/sin pre-tiled to each rope's
                // full width (GQA: Q and K differ)", minted through `bx.cos_sin(k_cols)` over
                // the K projection's own width) — and the worker fills them head-replicated
                // at exactly that width (`spyre_forward.rs`'s `tile`: a `[rows, width]`
                // source, `width` a multiple of the class's head dim, the per-position
                // rotary row repeated `width / head_dim` times per row). The splice's
                // windowing law demands precisely this staging: x, cos, sin and v share ONE
                // row window and take WHOLE `total` columns over tensors of one height,
                // `total` being the append's output width — `kv_width` here. This fixture
                // was BORN RED on that law (its cos/sin stated `q_width`, copied from the
                // golden `attn_ir`'s then-inert spelling, and the mint refused it); both
                // fixtures now state the real staging.
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: mq,
                    cols: kv_width,
                }, // 5: cos
                scratchy_subtile::subtile_ir::SourceShape {
                    rows: mq,
                    cols: kv_width,
                }, // 6: sin
            ]
        };
        let geom = scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
            scratchy_subtile::model_geometry::QueryHeads::new(nqh),
            scratchy_subtile::model_geometry::KvHeads::new(nkvh),
            scratchy_subtile::model_geometry::HeadDim::new(hd),
        )
        .expect("granite's geometry mints");
        let attn_op = scratchy_subtile::lower::OpDesc {
            op: scratchy_subtile::subtile_ir::SubOp::attn_decode(
                geom,
                scale,
                valid_len,
                scratchy_subtile::subtile_ir::AttnMask::Causal,
            ),
            m: mq,
            inputs: vec![
                scratchy_subtile::lower::InputRef::Ext(0),
                scratchy_subtile::lower::InputRef::Ext(1),
                scratchy_subtile::lower::InputRef::Ext(2),
                scratchy_subtile::lower::InputRef::Ext(3),
                scratchy_subtile::lower::InputRef::Ext(4),
            ],
        };
        let input = scratchy_subtile::lower::LoweringInput {
            sources: sources(cap),
            ops: vec![
                // 0: k' = rope_append(k, cos, sin, v, prefix_k, prefix_v) — the cache write that
                // binds `KvCacheProducer::SameForwardRopeAppend` for the attention below.
                scratchy_subtile::lower::OpDesc {
                    op: scratchy_subtile::subtile_ir::SubOp::rope_append(
                        scratchy_subtile::model_geometry::HeadDim::new(hd),
                        0,
                        scratchy_subtile::subtile_ir::AttnMask::Causal,
                        scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                    ),
                    m: mq,
                    inputs: vec![
                        scratchy_subtile::lower::InputRef::Ext(3),
                        scratchy_subtile::lower::InputRef::Ext(5),
                        scratchy_subtile::lower::InputRef::Ext(6),
                        scratchy_subtile::lower::InputRef::Ext(4),
                        scratchy_subtile::lower::InputRef::Ext(1),
                        scratchy_subtile::lower::InputRef::Ext(2),
                    ],
                },
                // 1: attn = AttnDecode(q, prefix_k, prefix_v, new_k, new_v)
                attn_op,
            ],
            result: 1,
            exported: None,
        };
        let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
        let ir = scratchy_subtile::subtile_ir::lower_region(&input, nb);

        // ⛔ THE PREFIX-FREE WALK: `ActiveCap::NONE` (a start==0 chunk has nothing resident to
        // attend) and prefill rows, through the production unrolled entrance — which threads
        // `paged_fold = None` to every node, the exact condition the judgment prefill's bake
        // runs under.
        let weight_ids = std::collections::HashSet::new();
        let (programs, mut layout) =
            super::lower_graph_to_ktir(&ir, &weight_ids, super::ActiveCap::new(u32::MAX), false)
                .unwrap_or_else(|e| panic!("the prefix-free prefill walk lowered: {e}"));

        // ── THE CARRIER HALF: exactly one op arrives stating its region, and it states the
        // prefill arm's own extent contract. ──
        let carriers: Vec<_> = programs
            .iter()
            .filter(|e| e.kv_paged_region.is_some())
            .collect();
        assert_eq!(
            carriers.len(),
            1,
            "the walk's one attention carrier — and only it — states a paged region"
        );
        let carrier = carriers[0];
        let region = carrier.kv_paged_region.expect("filtered on Some");
        let node = ir.nodes.last().expect("the attention node");
        let prefix_len = node.inputs[1].region.rows.len;
        let new_len = node.inputs[3].region.rows.len;
        let expected_pages = (prefix_len + new_len)
            .div_ceil(ktir_superdsc::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32);
        assert_eq!(
            region.region_pages, expected_pages,
            "the prefix-free carrier's region defaults to the walk's own extent (the spec's \
             `region_pages.unwrap_or(pages)` law)"
        );
        // And the layout this walk returned PLACED the pool the region names — the reach the
        // stamp's premise guard checks is present, not just assumed.
        for tid in region.reach_tids() {
            assert!(
                layout.gather_reach.contains_key(&tid),
                "the layout placed no gather reach for t{tid} — the stamp's premise guard \
                 would refuse this walk"
            );
        }

        // ── THE WALK-STAMP HALF: the same chain the unrolled entrance runs, landing in the
        // walk that stamps. ──
        super::place_step_inputs(&mut layout, &[&programs], ir.tensors.len() as u32)
            .unwrap_or_else(|e| panic!("step inputs placed: {e}"));
        layout.place_kv_block_index_last();
        let attn_params =
            crate::lower_subtile_tape_to_ktir::attn_bundle_params(&ir, false, false, None)
                .unwrap_or_else(|e| panic!("bundle params minted: {e}"))
                .expect("an attention graph mints attention params");
        // ── THE DIRECT-STAMP PIN (the child): the stamp's OWN contract, checked where it
        // happens. Lower the carrier through the door directly — a fresh sym-id base is
        // immaterial here, the stamped bits are constants and only per-op coverage and
        // non-emptiness matter — and call the walk's own stamp function
        // (`stamp_carrier_kv_facts`, the ONE stamping site) on the result. ──
        let k = carrier
            .ktir
            .as_ref()
            .expect("the carrier carries its KTIR program");
        let mut solo_base: i64 = 0;
        let mut solo_lowered =
            crate::ktir_superdsc_door::lower(k, &mut solo_base, Some(&layout), Some(attn_params))
                .unwrap_or_else(|err| {
                    panic!("the carrier lowered through the door: {}", err.message)
                });
        super::stamp_carrier_kv_facts(carrier, &mut solo_lowered, Some(&layout))
            .unwrap_or_else(|e| panic!("the stamp function accepted the carrier: {e}"));
        assert!(
            !solo_lowered.is_empty(),
            "the door minted no ops from the carrier — the stamp would cover nothing"
        );
        assert!(
            solo_lowered
                .iter()
                .all(|o| o.kv_gathered && o.kv_region_form),
            "the stamp left an op of the carrier unstamped — a partial stamp is exactly \
             the defect class this pin exists for"
        );

        // ── THE WIRING HALF: the walk's own mint and the walk's own PURE consumers — launch_index
        // and render_dxp_input — NEVER the walk itself. The walk's group Vec is filled only inside
        // the bake (`if let Some(bake) = superdsc_bake::global()`), so a cardless test through the
        // walk would assert over NOTHING, and a bake-side test would start a REAL bake from inside
        // `cargo test`. The mint is the walk's own extraction — same loop, same stamp call — so
        // what it stamps is what the walk stamps. ──
        let sdsc = super::mint_bundle_ops(&programs, Some(&layout), Some(attn_params))
            .unwrap_or_else(|e| panic!("the walk's mint rendered: {e}"))
            .ops;
        let shifts = super::launch_index(&sdsc, super::FoldGrouping::Split);
        let rendered = super::render_dxp_input(&sdsc, super::FoldGrouping::Split)
            .unwrap_or_else(|e| panic!("the walk's render ran cardless: {e}"));
        // ⛔ THE TRIP-ARITHMETIC DIFFERENTIAL (the wiring half) — the walk's mint APPLIED the
        // stamp and the groups FAITHFULLY CARRY the bits, proven by COUNTING, not naming.
        // Program identity is NOT readable off the rendered trips: the walk composes op
        // names from roles and regions (`{act_name(tid)}_g{leg}`, `{name}_scatter_t{tid}`,
        // …) — a different name space from the program's own `attn_s{id}` — so no
        // name-keyed membership test can find the carrier's trips. Instead the mint runs a
        // SECOND TIME on the carrier ALONE, and the full run's gathered groups must account
        // for EXACTLY the carrier's own trip total. The per-group trip count is the walk's
        // OWN derivation, not a new one: `files.len() − 1`, the same formula the group
        // budget uses for its `descriptors` count, resting on the render law "one
        // `sdsc_{i}.json` per trip, then `bundle.mlir` LAST" (`render_dxp_input`
        // constructs exactly one file per trip). TWO PREMISES, both stated:
        // (i) `group_ranges` is an EXACT covering partition (Kani-proven via the scalar
        //     twin `group_ranges_cover_ok`), so Σ trips over groups answering `gathered`
        //     is exactly the Gathered-kind trip count — a partial stamp leaves carrier
        //     trips Pure, a strict shortfall, never an invisible shuffle;
        // (ii) the carrier's trip count is bundle-context-free (the door's time-tiling is
        //      a per-op decision, sym-id-independent) — if that ever breaks, the equality
        //      fails LOUDLY here, which is the pin doing its job. The emptiness case — a
        //      deleted stamp leaving no gathered group at all — is caught by the alone
        //      run's own trip count.
        let carrier_idx = programs
            .iter()
            .position(|e| e.kv_paged_region.is_some())
            .expect("the carrier asserted above");
        let alone_sdsc = super::mint_bundle_ops(
            &programs[carrier_idx..carrier_idx + 1],
            Some(&layout),
            Some(attn_params),
        )
        .unwrap_or_else(|e| panic!("the carrier-alone mint rendered: {e}"))
        .ops;
        let alone_rendered = super::render_dxp_input(&alone_sdsc, super::FoldGrouping::Split)
            .unwrap_or_else(|e| panic!("the carrier-alone render ran cardless: {e}"));
        // Summed over ALL of the alone run's groups — not its gathered ones — so a deleted
        // stamp leaves `alone_trips` intact and the differential below fails at the
        // EQUALITY, not vacuously at this assert.
        let alone_trips = alone_rendered
            .iter()
            .map(|g| (g.files.len() - 1) as u64)
            .sum::<u64>();
        assert!(
            alone_trips > 0,
            "the carrier-alone run minted no trips — the carrier lowered to nothing, so \
             the differential below would pass vacuously"
        );
        let mut gathered_trips = 0u64;
        let mut region_form_trips = 0u64;
        for g in &rendered {
            // One `sdsc_{li}.json` per trip, then `bundle.mlir` last — a group's trip
            // count is its file count minus one. The group's channel bits come from the
            // walk's own launch index, keyed by the group's own id — the same pairing the
            // walk's staging loop uses.
            let group_trips = (g.files.len() - 1) as u64;
            let kv = *shifts.get(g.group as usize).unwrap_or_else(|| {
                panic!(
                    "group {} has dxp input but no entry in the launch index ({} entries) — \
                     `render_dxp_input` and `launch_index` disagree about the group partition",
                    g.group,
                    shifts.len()
                )
            });
            if kv.gathered {
                gathered_trips += group_trips;
            }
            if kv.gathered && kv.region_form {
                region_form_trips += group_trips;
            }
        }
        assert_eq!(
            gathered_trips, alone_trips,
            "the full run's gathered groups hold {gathered_trips} trips but the carrier \
             alone mints {alone_trips} — the stamp lost part of the carrier mid-body"
        );
        assert_eq!(
            region_form_trips, alone_trips,
            "the full run's gathered-and-region-form groups hold {region_form_trips} trips \
             but the carrier alone mints {alone_trips} — part of the carrier read the region \
             form unstaged"
        );
    }

    #[test]
    fn the_prefix_free_prefill_rung_stamps_its_carrier() {
        // The judgment's GEOMETRY at the m=31 one-pass rung — NOT the judgment prefill's
        // own shape: the judgment prefill's BODY is the mq=256 primary top
        // (4ae78fbfabfa3098, 4 groups/257 trips), whose width the variant below pins.
        prefix_free_prefill_rung_stamps_its_carrier_at(31, 33);
    }

    #[test]
    fn the_judgment_prefill_width_stamps_its_split_carrier() {
        // The judgment prefill's own WIDTH (mq=256, a one-page dead prefix over a
        // two-page gathered region) — the split-carrier case: the real body's carrier
        // is 161 counted ops spanning the 128-cut into two Gathered groups. Whether
        // this fixture's synthetic carrier crosses the cut is a counted fact the pod
        // run's dump will show; the stamp assert holds over every carved group either
        // way.
        prefix_free_prefill_rung_stamps_its_carrier_at(256, 257);
    }
}
