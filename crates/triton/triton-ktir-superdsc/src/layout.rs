// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! A [`BundleLayout`] DERIVED FROM THE PROGRAM — the device memory plan for one Triton kernel.
//!
//! # WHY A LAYOUT IS NOT OPTIONAL
//!
//! `layout: None` is `ktir-superdsc`'s unit-test arm, and two of its bodies cannot take it. MEASURED:
//!
//! ```text
//! $ cargo run --offline --example drive -- rmsnorm_granite 1 rmsnorm      # layout = None
//! REFUSED  rmsnorm_granite  rmsnorm  ktir-superdsc  "RmsNorm rmsnorm_fwd: epsilon 0.00001001358,
//!   read off the program, is absent from `BundleLayout::scalarmul_scales` …"
//! ```
//!
//! `emit::lower_ktir_to_superdsc::rmsnorm` reads its epsilon off the program and then resolves it to
//! a registry slot with `scale_slot(layout, eps)`, whose body is `layout.and_then(…)` — so with
//! `None` the answer is always "absent", whatever the program says. `Program::ScalarMul` has the
//! identical door. The registry lives on the layout, so the layout is the only way through it.
//!
//! # THE TRAP THIS MODULE IS BUILT AROUND
//!
//! `emit::resolve_seg_base` (vendor/ktir-superdsc/src/emit/mod.rs:1635) branches on whether a
//! layout is present, and its layout arm has THREE outcomes, not two:
//!
//! 1. the name is in `ids` as `PlaceId::Act(tid)` AND `placements` has that tid → its global address;
//! 2. the name is in the `synth` allocator's map → an intermediate's stable offset;
//! 3. **anything else → `panic!("synthetic '…' is accessed but was never declared to the layout")`**.
//!
//! A real parameter missing EITHER its `ids` entry or its `placements` entry falls into (3) and takes
//! the whole build down. So a half-built layout is strictly worse than `None`, and every tensor the
//! chosen body will name has to be in both maps before that body runs. That is why this module is
//! driven by [`regions`] — the parameter list the body itself reads — and not by a guess.
//!
//! # WHAT THIS MODULE DOES NOT DO
//!
//! It places exactly one RESERVED constant — rope's rotate matrix P, whose extent the PROGRAM
//! states (`[hd, hd]` at the head dim its own output view declares). It does not place the
//! constants whose extent nothing in the program says: `ATTN_MASK_TID` and `ATTN_CAUSAL_TID` are
//! the worker's per-step tables sized by a prefix CAPACITY, `IDENTITY_TID` by the head dim of a
//! geometry the program does not carry, `kct_resident_tid` is resident scratch. A placement with an
//! invented extent is the footprint guard's false negative — the guard compares an access against
//! `p.size`, so a size this module made up turns the one check that catches an over-run into a
//! rubber stamp. [`for_node`] therefore REFUSES a program whose entry point needs those, by name —
//! see [`reserved_consts_for`].

use ktir_core::ir::IRFunction;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, regions};
use ktir_superdsc::ktir_node::{KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{
    BundleLayout, SegRole, SynthAlloc, TensorPlacement, align128, synth_footprint_bytes,
};
use ktir_superdsc::reserved_tids::{ROPE_P_TID, scalarmul_scale_tid};
use ktir_superdsc::superdsc_opspec::Df;
use ktir_superdsc::wire::SEGMENT_SIZE;

use crate::Error;

/// One 128-B granule — the size the device READS in, and the floor `resolve_seg_base`'s footprint
/// guard compares a `[1,1]` broadcast const against (`align128(p.size).max(128)`).
const GRANULE: u64 = 128;

/// A bound `[1,1]` fp16 scale const's byte size. Two bytes, exactly as
/// `compute_bundle_layout`'s scale loop states it (scratchy
/// `lower_subtile_tape_to_superdsc.rs`, the `size: 2` placement) — the WORKER binds one fp16 per
/// registry slot, so the tensor really is two bytes; the 128-B granule is what the guard rounds it
/// up to, not what the tensor is.
const SCALE_BYTES: u64 = 2;

/// The layout for one node, and the refusal when the program cannot have one.
///
/// Calls [`regions`] itself so the driver needs nothing but the node. A `regions` failure is
/// reported under the SAME stage label [`crate::emit_node`] uses for it, so adding a layout does not
/// move where a program that never had a readable parameter list is refused.
pub fn for_node(node: &KtirNode) -> Result<BundleLayout, Error> {
    let r = regions(node).map_err(|e| Error {
        stage: "regions",
        message: format!("{e:?}"),
    })?;
    for_regions(node, &r)
}

/// [`for_node`] with the parameter walk already done — the form [`crate::emit_node`] uses, so the
/// node's `regions` are read ONCE and the layout cannot be built from a different reading than the
/// body gets.
pub fn for_regions(node: &KtirNode, r: &[Region]) -> Result<BundleLayout, Error> {
    let name = node.func.name;
    // ⛔ REFUSE BEFORE PLACING ANYTHING, because a body that names a device constant this module
    // did not place does not get a diagnostic — it gets `resolve_seg_base`'s panic.
    //
    // ⭐ AND THE REFUSAL IS ABOUT THE **BODY THAT WILL RUN**, WHICH IS WHY IT IS GATED ON
    // [`is_per_program_node`]. The reserved constants below are named by the per-`Program` entry
    // points — `attn_at`'s `ATTN_MASK_TID`/`IDENTITY_TID`/`kct_resident_tid(k)` are that hand-written
    // body's device state, and it is the body that looks them up. The WHOLE-FUNCTION door never
    // calls it: it walks the function's own ops, every operand of which is a parameter or an
    // in-function value, so nothing in that emission can reach a name this module did not place.
    //
    // ⛔ MEASURED, and this is why the gate exists: `attention_flash_noncausal` walks 8 matmuls, 8
    // reduces and an epilogue — no mask (STAGE 1 loads none), no rotate matrix, no resident KV — and
    // was refused for naming four constants NOTHING IN IT NAMES, purely because its `Program` field
    // spells `Attn`. `Program` is stated on a whole-function case only because the constructor
    // demands one, which the case table says in terms; reading it as a claim about the emission is
    // reading the wrong field.
    //
    // The predicate is [`scales_for_program_shape`]'s own, shared rather than restated: that
    // function already decides per-`Program`-versus-whole-function this way, and two statements of
    // which door a node goes through is how the two come to disagree.
    if let Some(need) = reserved_consts_for(node.program).filter(|_| is_per_program_node(&node.func))
    {
        return Err(Error {
            stage: "layout",
            message: format!(
                "{name}: {:?} names the reserved device constant(s) {need:?}, which are not \
                 parameters of any Triton kernel and which this adapter will not invent a placement \
                 for — `resolve_seg_base` PANICS on a name that is in neither `ids` nor the synth \
                 map, so an incomplete layout is worse than none. Place them from whatever owns the \
                 device state (the rotate matrix P, the attention masks, the identity) before \
                 driving this program.",
                node.program
            ),
        });
    }

    let scalarmul_scales = scales_for_program_shape(node.program, &node.func)?;

    // WHICH PARAMETER IS THE GATHER'S INDEX VECTOR, read off the program by the SAME function family
    // the lowering body reads it with — so the placement and the descriptor cannot disagree about
    // which buffer is the index. `None` for every affinely-addressed program. The MULTI-TILE reading
    // (`gathers_of`) accepts an unrolled sweep's per-trip tiles — every one over the SAME index/value
    // pair, which is all the placement needs: the index buffer is ONE buffer however many trips read
    // it. Different pairs are refused by `gathers_of` itself, by name.
    let gathers = ktir_superdsc::emit::lower_ktir_to_superdsc::gathers_of(node).map_err(|e| Error {
        stage: "layout",
        message: format!("{e:?}"),
    })?;
    // ⛔ AND THE ELEMENT TYPE IS CHECKED, NOT ASSUMED — for EVERY tile's index, since a sweep's
    // tiles share one parameter and one check would be a no-op for the rest only by coincidence.
    // dbo's `GatherIndexConversion.cpp:133` DT_CHECKs a 4-byte SENUINT32 index, and the
    // `idx32toaddr` SFP program splits each entry into two 16-bit halves — so a narrower index has
    // no form here at all, and sizing its placement at 4 B/elem while the program means 2 would
    // OVER-reserve and hide a real over-run from the footprint guard. The Triton fixture's own
    // contract agrees independently (`semantic.py`'s `descriptor_gather` asserts an int16/int32
    // index vector; `embedding.py` declares `desc_ids` as `*i32` for exactly this reason).
    for g in &gathers {
        let idx_dtype = index_view_dtype(node, g.index_tid);
        if idx_dtype != Some(ktir_core::dtypes::DType::I32) {
            return Err(Error {
                stage: "layout",
                message: format!(
                    "{name}: the gather's index buffer t{} states element type {idx_dtype:?}. dbo's \
                     index→address conversion DT_CHECKs a 4-byte SENUINT32 index \
                     (`GatherIndexConversion.cpp:133`) and its `idx32toaddr` program splits each entry \
                     into two 16-bit halves, so a narrower index has no lowering — it is refused rather \
                     than widened behind the caller's back.",
                    g.index_tid,
                ),
            });
        }
    }
    let index_tid = gathers.first().map(|g| g.index_tid);

    let mut placements: std::collections::BTreeMap<u32, TensorPlacement> = Default::default();
    // Bytes placed per segment so far — the running high-water [`pack`] appends at.
    let mut segment_bytes = [0u64; 7];

    // ── SEGMENTS ─────────────────────────────────────────────────────────────────────────────────
    // PACKED BY ROLE, AT DISJOINT INTRA-SEGMENT OFFSETS — one segment for every parameter the
    // program READS, one for every parameter it STORES THROUGH, and the synthetics alone in a third.
    //
    // ⛔⛔⛔ IT USED TO BE ONE SEGMENT PER PARAMETER (`segment_base(arg_index)`, offset 0), AND THAT
    // WAS A CEILING THE DEVICE DOES NOT HAVE. MEASURED: `decoder_layer_one_flat` is 14 parameters and
    // `decoder_two_layers_flat` is 23, so both were refused for wanting 14/23 of 7 segments — the
    // whole decoder block, unreachable on an arithmetic of ours. The 7-segment doctrine is stated PER
    // OP: `wire::SEGMENT_OFFSETS`'s own doc says "each I/O arg lives in its OWN 16 GiB segment keyed
    // by its `arg_index` (== `ldsIdx_` == the tensor's position in `op.args`)" and `segment_base`'s
    // says "an op with >7 distinct dataspaces". `op.args` is ONE EMITTED DESCRIPTOR's operand list —
    // three or four tensors for every body here — not the function's parameter list.
    //
    // ⭐ WHAT MAKES PACKING SAFE, AND WHERE THE PROOF IS. `resolve_seg_base` resolves a placed tensor
    // to `(segment_base(p.segment), p.offset + slice)`: the segment is a BASE and the offset is free
    // inside it, so a segment holds as many tensors as fit and `SEGMENT_SIZE` is 16 GiB against
    // Triton buffers of KiB to MiB. Non-aliasing is therefore a property of the OFFSETS rather than
    // of the segment numbering, and it is PROVED rather than asserted: [`overlaps_none`] compares
    // every pair of placements over their 128-B GRANULE spans — the same granule
    // `resolve_seg_base`'s footprint guard measures an access against (`align128(p.size).max(128)`)
    // — so a pair this module proves disjoint cannot be brought back together by a rounding
    // disagreement. That is the justification the old refusal said it did not have.
    //
    // ⭐ AND IT IS THE PRODUCER'S OWN CONSTRUCTION, NOT A SCHEME INVENTED HERE. scratchy's
    // `compute_bundle_layout` (`lower_subtile_tape_to_superdsc.rs`, the `pack` closure) places every
    // source at `seg_bytes[role.segment()]` and bumps that high-water by `align128(off + size)`;
    // that is how a granite-3.1-8b bundle puts hundreds of weights in seg1 and every per-step input
    // in seg3, on card, at 40 layers. This is the same three lines, over `regions()` instead of over
    // a tape.
    //
    // ⭐⭐ WHY THE STORED-THROUGH PARAMETERS GET THEIR OWN SEGMENT, which is the one decision here
    // that is not forced. dxp's ModuleStitcher wires producer→consumer BY SEGMENT — `placement.rs`'s
    // note on the intermediate colours records what ignoring that cost ("cramming them together was
    // the multi-op all-seg3 orphans bug") — and the positional scheme kept a program's output in a
    // different segment from its inputs BY ACCIDENT of parameter order. Packing everything into one
    // segment would quietly give that up, so the split is kept deliberately, and it costs 2 of the 7
    // slots however many parameters the function has. `Region::is_out` is the program's own statement
    // of which parameters those are — a `ktdp.store` writes through the view — and NOT "the last
    // parameter", which its own doc explains is false when a constant is first used after the output.
    //
    // ⭐ SO `role` AND `segment` NOW AGREE, and the warning that used to stand here is gone: the role
    // CHOOSES the segment again ([`pack`] takes only the role), so a downstream worker deriving a
    // binding region from `role.segment()` reads the same segment the placement states.
    for reg in r.iter() {
        // SIZE IS THE TENSOR'S TRUE FOOTPRINT, VIA THEIR OWN FUNCTION. `synth_footprint_bytes`
        // pads the inner (stick) axis up to the dtype's stick-elem multiple and multiplies the
        // rest — the same rule `materialized_bytes` measures an access against on the other side of
        // the footprint guard, so a guard hit means a real over-run and never a rounding
        // disagreement between two paddings. Hand-rolling `rows*cols*2` and 128-aligning it would
        // UNDER-state every tensor whose cols are not a stick multiple (padded logits: 49159 →
        // 49664), and the guard would then refuse the correct descriptor.
        //
        // The extent is the parameter's own VIEW (`v_rows × v_cols`, off its
        // `ktdp.construct_memory_view`). A view is a reinterpretation of the whole buffer, so its
        // element product is the buffer's; `Region`'s own doc is explicit that the view shape is
        // not always the tensor's shape (rope views `x` as `[rows·heads, hd]`), and it is the
        // shape every descriptor here addresses through.
        // ⭐⭐⭐ A GATHER'S INDEX BUFFER IS FOUR BYTES AN ELEMENT, AND THE FOOTPRINT GUARD IS WHY THIS
        // CANNOT BE LEFT AT THE fp16 DEFAULT. `Region` states a view's fp8-ness and nothing else, so an
        // `*i32` parameter was sized as if it were fp16 — 512 B for `embedding.py`'s `[256]` id vector
        // against the 1024 B it really is. `resolve_seg_base`'s guard then compares the descriptor's
        // 1024 B access against that 512 B placement and refuses the CORRECT descriptor:
        //   `t0: access offset 0B + 1024B exceeds its placement footprint 512B (seg3)`
        // — measured, and the reason this arm exists. Had the numbers happened to fit, the other
        // direction would have been worse: the guard would have waved through an access reaching into
        // whatever is packed next.
        //
        // The format is [`Df::Uint32`], which is the format the DESCRIPTOR declares for that operand
        // (`Role::KernelIdx` ⇒ SENUINT32, `wordLength` 4) — so the placement and the access are sized in
        // ONE format rather than in two that have to agree.
        let df = if Some(reg.tid) == index_tid {
            Df::SenUint32
        } else if reg.is_fp8 {
            Df::Fp8
        } else {
            Df::Fp16
        };
        // ⛔⛔⛔ A RUNG-3 BOUND SCALE — a `[1,1]` PARAMETER — IS 2 B, NOT ITS STICK-PADDED
        // FOOTPRINT. `synth_footprint_bytes(&[1,1], Fp16)` pads the inner axis to the
        // 64-elem stick (128 B), which is right for a tensor the descriptors address and
        // WRONG for a scalar the launcher binds: the runner's `const:t<id>=<value>` arm
        // demands an EXACT payload match against the placement (`bundle_run.rs`, "A short
        // buffer would leave the tail whatever the device had there"), and the payload is
        // one IEEE fp16 — 2 B. A 128 B placement refuses the correct bind; and the other
        // direction is the recorded unbound-scale identity trap, a clean exit 0 whose
        // readback is the INPUT at within_2pct=1.0000, so exactness here is a correctness
        // matter and not a convenience.
        //
        // ⭐ THE SAME SIZE THE SCALE REGISTRY ITSELF USES — `SCALE_BYTES`, the size
        // `compute_bundle_layout`'s scale loop packs every reserved scale at. The
        // distinction is the BINDING KEY, not the size: a registry scale is reached at
        // `scalarmul_scale_tid(i)` and a rung-3 scale at its own parameter tid, which is
        // what `scalarmul_bound` passes `split_out_excluding` as a skip tid so the region
        // never counts as tensor arity.
        let is_bound_scale = reg.v_rows == 1 && reg.v_cols == 1;
        let size = if is_bound_scale {
            SCALE_BYTES
        } else {
            synth_footprint_bytes(&[reg.v_rows, reg.v_cols], df)
        };
        // ⭐ THE ROLE IS THE TENSOR'S LIFECYCLE, AND EVERY ONE OF THESE IS BOUND BY THE LAUNCH: the
        // host binds an address per parameter per launch, which is what `SegRole::Activation` ("per-
        // step graph inputs … DMA'd each step") states, and `SegRole::Logits` ("the single graph
        // output — D2H'd each step") is the same fact about the buffer the launch reads back.
        //
        // Nothing here is a resident model weight or a paged KV plane, so no other variant would be
        // true — a Triton kernel's weight is an argument bound per launch like the rest.
        //
        // ⛔ AND `SegRole::Weight` IS NOT THE FIX FOR THE mb-BROADCAST DEFECT, MEASURED. `rmsnorm_
        // granite` on card computes every stage of the fused norm correctly (sq16 within_2pct 1.0000,
        // rinv/xn to DL16 precision) and then reads gamma ROW-INDEXED in the final `out = xn·gamma`:
        // a stick-index ramp staged as gamma shows row r fetching gamma's stick r at rows=64, and
        // sticks 2r/2r+1 at rows=32 — one law, `idx = (r·cols + c)/rows`, so the [1,cols] vector is
        // SPLIT across the rows instead of broadcast to them. Placing gamma in `SegRole::Weight`
        // (segment 1, granite's own segment for it) was tried at BOTH row counts and moved nothing:
        // within_2pct 0.0193 at 64 rows and 0.0200 at 32, byte-identical to the `Activation`
        // placement, with the ramp still reading `stick == row` at 0.7854. So the per-core mapping of
        // the segment is NOT what distinguishes us from granite, and this arm stays as it was.
        let role = if reg.is_out { SegRole::Logits } else { SegRole::Activation };
        pack(name, reg.tid, role, size, &mut placements, &mut segment_bytes)?;
    }

    // ── THE SCALE REGISTRY ───────────────────────────────────────────────────────────────────────
    // Index `i` ↔ the reserved tid `scalarmul_scale_tid(i)` (`u32::MAX - 20 - i`), which is what the
    // bodies render as their `[1,1]` const operand — so each one needs a placement or
    // `resolve_seg_base` panics on it. Packed after the inputs in the SAME segment, which is where
    // `compute_bundle_layout`'s scale loop puts them (`SegRole::Activation`, `size: 2`, high-water
    // bumped per const): the tensor is 2 B (the worker binds one fp16) and [`pack`] still advances a
    // whole granule, so one const's read can never reach the next.
    for i in 0..scalarmul_scales.len() {
        // Worker-bound per step, exactly like the parameters — and this is the ONE role scratchy's
        // own build guard asserts for a reserved const ("a WEIGHT-segment reserved tid is never
        // bound ⇒ stays ZERO ⇒ silent wrong numerics").
        let tid = scalarmul_scale_tid(i);
        pack(name, tid, SegRole::Activation, SCALE_BYTES, &mut placements, &mut segment_bytes)?;
    }

    // ── THE ROTATE MATRIX P ──────────────────────────────────────────────────────────────────────
    // `rope_at` opens with `crate::place::act_name(ROPE_P_TID)` and reaches it from BOTH its forms —
    // the head-major collapse as one `[hd, hd]` kernel matmul, the slab path as a per-slab ±I block
    // — so a rope driven without this placement is not a wrong address, it is `resolve_seg_base`'s
    // panic.
    //
    // ⭐ AND ITS EXTENT IS READ, NOT INVENTED, WHICH IS WHY THIS ONE RESERVED CONST CAN BE PLACED
    // HERE AT ALL. P is `[hd, hd]` fp16 — `rope_at` addresses it as
    // `Stk::<KernelTag>::kernel(hd, hd, &p)` and scratchy places it at `hd * hd * 2` — and `hd` is
    // stated by the program: `Region`'s own doc says the rope views its plane as `[rows·heads, hd]`,
    // so the OUTPUT view's column count IS the head dim. Nothing is guessed, so the footprint guard
    // keeps its teeth for P as well.
    //
    // ⛔ IT IS A DEVICE CONSTANT AND NOT A PARAMETER, so it is placed with the per-launch inputs
    // (`SegRole::Activation`) and not in the output segment: scratchy's own comment at this
    // placement records the alternative's cost — a seg1/WEIGHT P is H2D'd only at PrepareModel,
    // where the worker binds only the manifest's model weights, so it "stays ZERO ⇒ rot=matmul(x,0)
    // ⇒ RoPE collapses to x·cos ⇒ wrong positional encoding".
    if matches!(node.program, Program::Rope) {
        let out = r.iter().find(|reg| reg.is_out).ok_or_else(|| Error {
            stage: "layout",
            message: format!(
                "{name}: Rope states no parameter its program stores through, so there is no output \
                 view to read the head dim off and P's `[hd, hd]` extent cannot be stated. A rope \
                 that writes nothing is a malformed program, not a placement to invent."
            ),
        })?;
        let hd = out.v_cols;
        if hd == 0 {
            return Err(Error {
                stage: "layout",
                message: format!(
                    "{name}: the output t{} states a `[{}, 0]` view, so its head dim is zero and P \
                     would be a zero-byte placement — which `resolve_seg_base`'s synth arm records \
                     the device reads as `1 << 27` flits = 16 GiB.",
                    out.tid, out.v_rows,
                ),
            });
        }
        let size = synth_footprint_bytes(&[hd, hd], Df::Fp16);
        pack(name, ROPE_P_TID, SegRole::Activation, size, &mut placements, &mut segment_bytes)?;
    }

    // ── THE fp8 W8A8 CLAMP CONSTS ─────────────────────────────────────────────────────────────────
    // `matmul_fp8_descriptors`'s activation-quantize chain names three reserved consts by name —
    // `rb(&act_name(FP8_{POS,NEG,INV}448_TID))`, the E4M3 clamp bounds ±448 and 1/448 for the
    // per-token amax→a_scale — and a layout that does not place them is not a wrong address, it is
    // `resolve_seg_base`'s panic on `t4294967281`-and-friends (MEASURED: the first fp8 bake died
    // exactly there). Scratchy's own producer places them under the same rule this arm copies
    // (`lower_subtile_tape_to_superdsc.rs:809`): iff the tape has an fp8 (arity-3) MatmulTile —
    // here, iff any region states `is_fp8`, which `regions()` stamps iff the view's dtype is
    // `Fp8E4m3`.
    //
    // Each is `[1, stick]` fp16 (64 elems × 2 B = 128 B, one granule), worker-bound in the
    // ACTIVATION segment exactly like the scalarmul scales — the same placement role scratchy's
    // own build uses (`SegRole::Activation`). The VALUES are the device's own E4M3 bounds; bake
    // records them beside `scalarmul_scales` so the launcher binds them rather than leaving them
    // at whatever the segment held (scratchy's own comment on the unbound state: "clamp consts
    // never bound ⇒ garbage quant").
    if r.iter().any(|reg| reg.is_fp8) {
        for tid in [
            ktir_superdsc::reserved_tids::FP8_POS448_TID,
            ktir_superdsc::reserved_tids::FP8_NEG448_TID,
            ktir_superdsc::reserved_tids::FP8_INV448_TID,
        ] {
            let size = synth_footprint_bytes(&[1, 64], Df::Fp16);
            pack(name, tid, SegRole::Activation, size, &mut placements, &mut segment_bytes)?;
        }
    }

    // ⛔⛔⛔ THE GATHER'S INDEX BUFFER CANNOT END FLUSH AT ITS SEGMENT'S HIGH-WATER, AND THE DEVICE
    // ITSELF ENFORCES IT — MEASURED ON CARD, `paged_score_small`: the ids (256 B, the segment's last
    // placement) ended exactly at `segment_bytes[3]`, and the gather leg's HBM fetch of the FINAL
    // 128-B stick FAULTED (`HMI=FETCH addr=0xc00004480`, `run_prefix_only sync rc=-1`). One 128-B
    // granule of slack past the ids' last byte (segment_bytes 17664→17792) makes the identical
    // bundle run clean with byte-identical output — the fetch over-reads past the tensor by up to
    // one granule, and a segment boundary is not a legal place for that over-read to land. The
    // fault is scoped, not general: the SAME embedding configs run card-verified with their ids at
    // offset 0 (followed by the table, never flush), and attention's flush-at-end OUTPUT in seg4
    // writes clean — it is the gather's FETCH path alone that pays this.
    //
    // The slack is added ONLY when the ids is the segment's last placement (its end flush at the
    // high-water), because that is the only placement whose over-read can reach the boundary — an
    // ids packed anywhere else has a neighbour above it to absorb the read, and a blanket segment
    // round-up would move `segment_bytes` for every config and break the byte-identity the 12-config
    // regression pins. `pack`'s own arithmetic already guarantees `segment_bytes[seg]` is a whole
    // 128-B multiple, so one granule of slack past the high-water covers the measured over-read
    // exactly.
    if let Some(index_tid) = index_tid {
        if let Some(&p) = placements.get(&index_tid) {
            if p.offset + align128(p.size).max(GRANULE) == segment_bytes[p.segment] {
                segment_bytes[p.segment] += GRANULE;
            }
        }
    }

    // ── THE CEILING THAT IS REAL: BYTES, PER SEGMENT ─────────────────────────────────────────────
    // A segment is one 16 GiB region AND the addressing stride is that same 16 GiB
    // (`wire::hbm_seg_off`: `seg = addr / SEGMENT_SIZE`), so a placement past the end of its segment
    // does not merely overflow a region — its address DECOMPOSES INTO THE NEXT SEGMENT, which is
    // precisely the aliasing every line above exists to prevent, and [`overlaps_none`] would not see
    // it because the two tensors' `(segment, offset)` pairs still differ. Refused by name.
    if let Some((seg, bytes)) = segment_bytes.iter().enumerate().find(|(_, &b)| b > SEGMENT_SIZE) {
        return Err(Error {
            stage: "layout",
            message: format!(
                "{name}: seg{seg} packs {bytes} B and a segment is {SEGMENT_SIZE} B (`SEGMENT_SIZE`, \
                 which is also the addressing stride) — every placement past that point resolves \
                 into the NEXT segment's addresses, so it would alias another tensor with nothing \
                 here able to see it. This program's tensors do not fit one HBM segment."
            ),
        });
    }

    // ── THE OVERLAP PROOF ────────────────────────────────────────────────────────────────────────
    // Everything above is arithmetic this module wrote, and an overlap inside one segment is the
    // failure mode with no symptom: two tensors share bytes, the producer of one writes the other's
    // and every guard downstream sees a legal access. Cheap to prove, so proved.
    overlaps_none(name, &placements)?;

    // ── SYNTHETICS ───────────────────────────────────────────────────────────────────────────────
    // `resolve_seg_base`'s synth arm hardcodes `SegRole::Intermediate.segment()`, so every
    // intermediate a body invents lands in that segment — and under the role-packed scheme above
    // NOTHING ELSE IS THERE: every parameter is `Activation` or `Logits` and P and the scale consts
    // are `Activation`, so the intermediate segment belongs to the synthetics alone and this seed is
    // 0 today. It is still read off `segment_bytes` rather than written as `0`, because that is what
    // makes it stay true: `compute_bundle_layout` seeds it the same way (`next:
    // seg_bytes[Intermediate]`, synthetics packed ABOVE the colored intermediates in one segment),
    // and the day a role here becomes `Intermediate` the allocator must start above it rather than
    // on top of it.
    let synth = std::cell::RefCell::new(SynthAlloc {
        next: segment_bytes[SegRole::Intermediate.segment()],
        map: Default::default(),
        sizes: Default::default(),
    });

    // IDENTITY BY OPERAND SPELLING, for every placed tensor, from the one site that knows the whole
    // set — `compute_bundle_layout`'s construction verbatim. A placement without its `ids` entry is
    // invisible to `resolve_seg_base`, which then treats the parameter as an undeclared synthetic
    // and panics; deriving the map from `placements.keys()` is what makes the two impossible to
    // disagree.
    let ids = std::cell::RefCell::new(
        placements
            .keys()
            .map(|&tid| (act_name(tid), PlaceId::Act(tid)))
            .collect::<std::collections::BTreeMap<_, _>>(),
    );

    Ok(BundleLayout {
        placements,
        ids,
        // The PLACEMENT high-water per segment. It does not include the synthetics, which are
        // bump-allocated during lowering through the `RefCell` above and so do not exist yet — the
        // same order `compute_bundle_layout` builds in.
        segment_bytes,
        // No banking: banks exist for a weight segment too big for one device region, and one
        // Triton kernel's parameters are one region each.
        weight_bank_bytes: Vec::new(),
        // No matmul KERNEL re-tile descriptors. These are built ONLY from a `DeviceTileLayout`
        // witness so the shim's re-tile and the emitter's per-core address read one layout; an
        // entry invented here would be a second, unwitnessed layout. `Program::Matmul` on a
        // Triton kernel therefore stages its weight FLAT, which is what an empty map states.
        kernel_weights: Default::default(),
        scalarmul_scales,
        synth,
        // The arrangement authority starts empty and every op declares into it — that is the point:
        // the FIRST op to address a tensor fixes its device layout and a later disagreement is a
        // build error. Pre-seeding it would pre-decide arrangements no descriptor has stated.
        arrangements: Default::default(),
        // No paged KV in a single Triton kernel's parameter list; 0 is the documented "no paged KV"
        // value, not a placeholder.
        kv_request_stride_bytes: 0,
    })
}

/// The ELEMENT TYPE the program declares for the buffer bound to `tid` — its parameter's
/// `ktdp.construct_memory_view` `Dtype`, which is the only statement of that buffer's element width
/// anywhere in the program.
///
/// The tid→parameter join is [`regions`]' own (`bindings[i]` is the tensor `arguments[i]` points at), so
/// this reaches the same parameter that walk does rather than keeping a second copy of the mapping.
///
/// `None` when nothing is bound to `tid` or its view carries no dtype. The caller turns that into a
/// refusal rather than a default: "not stated" and "stated as i32" are different facts.
fn index_view_dtype(node: &KtirNode, tid: u32) -> Option<ktir_core::dtypes::DType> {
    use ktir_core::attrkey::AttrKey;
    use ktir_core::ir::Attr;
    use ktir_core::opkind::OpKind;
    let f = &node.func;
    let i = node.bindings.iter().position(|b| b.get() == tid)?;
    let (ptr, _) = f.arguments.get(i)?;
    let view = f.operations.iter().find(|o| {
        o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr)
    })?;
    view.attributes.iter().find_map(|(k, v)| match (k, v) {
        (AttrKey::Dtype, Attr::Dtype(d)) => Some(*d),
        _ => None,
    })
}

/// The reserved device constants a `Program`'s entry point names without them being parameters, or
/// `None` when it names none.
///
/// ⛔ READ OFF THE BODIES, one by one. `rope_at` opens with `crate::place::act_name(ROPE_P_TID)`
/// (the rotate matrix) and `attn_at` names `ATTN_MASK_TID`, `ATTN_CAUSAL_TID`, `IDENTITY_TID` and
/// `kct_resident_tid(k_id)`. None of those is a buffer a Triton kernel declares. Every other body
/// names only its parameters and its own declared synthetics.
///
/// ⭐ THE LINE IS NOT "IS IT A PARAMETER", IT IS "DOES THE PROGRAM STATE ITS EXTENT" — and that is
/// what moved `Rope` off this list. The footprint guard compares an access against `p.size`, so a
/// placement whose size this module invented converts the one check that catches an over-run into a
/// rubber stamp; a placement whose size the PROGRAM states keeps it. Rope's P is `[hd, hd]` and `hd`
/// is the rope output view's own column count (see [`for_regions`]'s P placement), so it is read.
/// Attention's are not, and none of them is close: `ATTN_MASK_TID` is `[nqh·mq, cap]` over a prefix
/// CAPACITY the worker chooses per step, `ATTN_CAUSAL_TID` the same, `IDENTITY_TID` is `[hd, hd]` at
/// the head dim of a GQA geometry that reaches `attn_at` as a const generic and not as a view, and
/// `kct_resident_tid` is resident scratch whose extent is the K cache's. So `Attn` stays refused BY
/// NAME, which is the whole point of this function: `resolve_seg_base` PANICS on a name in neither
/// `ids` nor the synth map, so the choice is a diagnostic here or a build crash there.
/// Does this function go through the PER-`Program` door, or the whole-function walk?
///
/// ONE statement of the question, because two callers ask it: the reserved-constant refusal in
/// [`for_regions`] (whose named constants belong to the per-`Program` bodies) and
/// [`scales_for_program_shape`] (whose fixed per-`Program` scale order only describes those bodies).
///
/// The test is how many ops the whole-function walk would lower. ONE means the function IS the
/// single compute op its `Program` names, so the hand-written entry point is what runs. MORE than
/// one means the walk runs and `Program` is inert — which the case table states for every
/// whole-function configuration it carries.
fn is_per_program_node(f: &IRFunction<'static>) -> bool {
    use ktir_superdsc::emit::whole_function as W;
    f.operations
        .iter()
        .filter(|o| {
            W::program_of(o.op_type).is_some()
                || o.op_type == ktir_core::opkind::OpKind::LinalgReduce
        })
        .count()
        <= 1
}

fn reserved_consts_for(p: Program) -> Option<&'static [&'static str]> {
    match p {
        Program::Attn => Some(&[
            "ATTN_MASK_TID",
            "ATTN_CAUSAL_TID",
            "IDENTITY_TID",
            "kct_resident_tid(k)",
        ]),
        // `Rope` is HERE and not above: its one reserved constant is placed by [`for_regions`],
        // from an extent the program states.
        Program::Rope
        | Program::RmsNorm
        | Program::ScalarMul
        | Program::Elementwise(_)
        | Program::SiluMul
        | Program::LmLast
        | Program::Transpose
        | Program::Matmul => None,
    }
}

/// Place one tensor at the next free 128-B granule of its role's segment, and bump that segment's
/// high-water.
///
/// ⭐ THE OFFSET IS NOT A CHOICE, WHICH IS WHY PACKING NEEDS NO LIFECYCLE. The old one-segment-per-
/// parameter scheme was defended on the grounds that "packing several tensors into one segment needs
/// a reason to put THIS tensor at THAT offset"; append-at-the-high-water is that reason, and it is
/// the only rule `compute_bundle_layout`'s own `pack` closure uses. Every offset is 128-B aligned and
/// every tensor advances the water by at least one GRANULE, so consecutive placements cannot share a
/// granule — which is the property [`overlaps_none`] then PROVES over the whole set rather than
/// trusting it of this arithmetic.
fn pack(
    name: &str,
    tid: u32,
    role: SegRole,
    size: u64,
    placements: &mut std::collections::BTreeMap<u32, TensorPlacement>,
    segment_bytes: &mut [u64; 7],
) -> Result<(), Error> {
    let segment = role.segment();
    let offset = segment_bytes[segment];
    if let Some(prev) = placements.insert(
        tid,
        TensorPlacement { tid, role, segment, bank: 0, offset, size },
    ) {
        // A repeated tid means one buffer reached this walk twice — two parameters bound to it, or a
        // reserved const colliding with a parameter. Their `bindings` are ours (parameter position),
        // so this is our bug, not the program's, and the second placement would WIN: every op
        // addressing the first would read the second's offset, at the second's extent.
        return Err(Error {
            stage: "layout",
            message: format!(
                "{name}: t{tid} is placed twice — already at {} B of seg{} ({} B), now at {offset} B \
                 of seg{segment} ({size} B). One buffer cannot hold two placements: the second wins \
                 and every descriptor built against the first addresses the wrong bytes.",
                prev.offset, prev.segment, prev.size,
            ),
        });
    }
    // ⛔ AT LEAST ONE GRANULE PER TENSOR, WHATEVER ITS LOGICAL SIZE. The device reads in 128-B
    // granules and `resolve_seg_base`'s footprint guard admits an access up to
    // `align128(size).max(128)`, so a 2-B scale const OCCUPIES a granule; advancing the water by
    // `align128(offset + size)` alone would be right for every tensor at or above the granule and
    // would let two sub-granule consts share one on-card read.
    segment_bytes[segment] = align128(offset + size).max(offset + GRANULE);
    Ok(())
}

/// No two placements share a byte in one segment.
fn overlaps_none(
    name: &str,
    placements: &std::collections::BTreeMap<u32, TensorPlacement>,
) -> Result<(), Error> {
    let mut spans: Vec<(usize, u64, u64, u32)> = placements
        .values()
        // The device reads in 128-B granules and the footprint guard compares against
        // `align128(size).max(128)`, so a tensor OCCUPIES its rounded-up granule span whatever its
        // logical size. Two 2-B consts 2 B apart do not overlap logically and DO overlap on card.
        .map(|p| {
            (
                p.segment,
                p.offset,
                p.offset + align128(p.size).max(GRANULE),
                p.tid,
            )
        })
        .collect();
    spans.sort();
    for w in spans.windows(2) {
        let (s0, _, e0, t0) = w[0];
        let (s1, b1, _, t1) = w[1];
        if s0 == s1 && b1 < e0 {
            return Err(Error {
                stage: "layout",
                message: format!(
                    "{name}: t{t1} starts at {b1} B of seg{s1} while t{t0} still occupies it \
                     (through {e0} B, its 128-B-granule span) — two tensors on one buffer is the \
                     failure with no symptom: the second's producer writes the first's bytes and \
                     every guard downstream sees a legal access."
                ),
            });
        }
    }
    Ok(())
}

/// THE REGISTRY VALUES, CHOSEN BY THE PROGRAM'S OWN SHAPE rather than by a caller's flag.
///
/// A function that is ONE compute op takes [`scales_for`] — the per-`Program` rule, unchanged, so
/// every configuration that baked before bakes byte-identically. A WHOLE-KERNEL function takes the
/// whole-function rule, because the per-`Program` readers cannot serve it: `program_rmsnorm_eps`
/// enforces one root `math.rsqrt` per FUNCTION and a decoder layer has two; `program_scalarmul_scale`
/// requires every `arith.mulf` to agree and a decoder layer has 21 of them carrying four different
/// constants.
///
/// ⭐ DERIVED, NOT FLAGGED, AND THAT IS DELIBERATE. Which door a node will be driven through is the
/// caller's choice and is made AFTER this function runs, so a flag would have to be threaded from
/// `bake_py` through `for_node` for a fact the program already states. Counting the function's compute
/// ops asks the question directly: a one-op function IS a per-`Program` node, whatever door is used,
/// and its registry must not change.
///
/// ⛔⛔⛔ AND THE WHOLE-FUNCTION RULE IS NOT "EVERY FLOAT THE PROGRAM SPLATS". That was tried and is
/// the divergence this module's [`scales_for`] doc records in the vendor's own words: the
/// mean-of-squares divisor `1/cols` is NOT a registry scale — it is bound at the reserved
/// `RMS_INVCOLS_TID` with its own placement — and pushing it here adds a slot per rmsnorm, shifting
/// the device tid (`SCALARMUL_SCALE_BASE - i`) of every constant registered after it. So the
/// whole-function rule registers exactly two kinds of value:
///
/// 1. **one epsilon per recognised rmsnorm chain**, read where that chain's own `arith.addf` states it;
/// 2. **every splat multiplier that is NOT chain-interior** — the genuine scalar multipliers, which for
///    a decoder are the score scale and the two residual multipliers.
///
/// The `1/cols` of each rmsnorm falls into NEITHER, because the recogniser makes it chain-interior.
/// That is the whole reason the chain has to be recognised before the registry can be filled, and it
/// is why this function calls THE SAME recogniser the emitter dispatches on — agreement is not
/// something to maintain, it is the same call.
fn scales_for_program_shape(
    p: Program,
    f: &IRFunction<'static>,
) -> Result<Vec<f32>, crate::Error> {
    use ktir_superdsc::emit::whole_function as W;

    if is_per_program_node(f) {
        return Ok(scales_for(p, f));
    }

    let chains = W::program_rmsnorm_chains(f).map_err(|e| crate::Error {
        stage: "layout",
        message: e.message,
    })?;
    let mut out: Vec<f32> = Vec::new();
    let push = |v: f32, out: &mut Vec<f32>| {
        if !out.iter().any(|x| x.to_bits() == v.to_bits()) {
            out.push(v);
        }
    };
    // (1) the epsilons, one per chain.
    for c in &chains {
        push(c.eps, &mut out);
    }
    // (2) the splat multipliers that survive the fusions. A chain-interior multiply is not a scalar
    // multiply at all — the fused body owns its constants — so its value must not take a slot.
    let interior: Vec<ktir_core::ir::Ssa> =
        chains.iter().flat_map(|c| c.consumed.iter().copied()).collect();
    for op in f.operations.iter() {
        if op.op_type != ktir_core::opkind::OpKind::ArithMulf {
            continue;
        }
        if op.result.is_some_and(|r| interior.contains(&r)) {
            continue;
        }
        if let Some((v, _)) = W::splat_scale_of(f, op) {
            push(v, &mut out);
        }
    }
    Ok(out)
}

/// THE VALUES THE CHOSEN ENTRY POINT WILL LOOK UP — exactly those, in a fixed order, per `Program`.
///
/// ⛔⛔⛔ THIS USED TO BE "EVERY FLOAT THE PROGRAM SPLATS", AND THAT WAS A REGISTRY DIVERGENCE.
/// `rmsnorm_granite` splats three: the epsilon, `1/D_MODEL` and the reduce seed `0.0`. Registering
/// the first two put `1/4096` in slot 1, and `compute_bundle_layout`'s scale walk refuses exactly
/// that, in terms:
///
/// > ⛔ THE EPSILON ONLY — the mean-of-squares divisor is NOT a registry scale. `1/cols` is bound at
/// > the reserved `RMS_INVCOLS_TID` as a `[1, stick]` row with its own placement, which is where
/// > `subtile→superdsc` reads it from; pushing it here as well added one registry slot per rmsnorm
/// > node, shifting the tid of every constant registered after it.
///
/// The index IS the device tid (`SCALARMUL_SCALE_BASE - i`), so a spurious slot moves the ADDRESS
/// every later constant reaches the card at. It happened to be harmless for a one-node rmsnorm — the
/// epsilon landed on slot 0 and nothing reads a later slot — but only by accident of the order the
/// ttir emitted its constants in, which is not a property anything should depend on.
///
/// ⭐⭐⭐ AND IT CALLS THEIR READERS RATHER THAN RESTATING THEM. `program_rmsnorm_eps`,
/// `program_scalarmul_scale` and `program_score_scale` were private, so the first version of this
/// function reproduced all three structurally — two matchers for one fact, and the copy on this side
/// is the one that would go stale unnoticed. They are `pub` now, so the registry is populated by the
/// SAME function the body looks its value up with: agreement is not something to maintain, it is the
/// same call. Nothing else here can drift either — the walk (`f.operations`, not `ops_deep()`), the
/// `arith.constant`-behind-`tensor.splat` shape, the one-root rule and the all-`mulf`-agree rule are
/// all theirs by construction now.
fn scales_for(p: Program, f: &IRFunction<'static>) -> Vec<f32> {
    use ktir_superdsc::emit::lower_ktir_to_superdsc as L;
    match p {
        // `rmsnorm` resolves ONE value: `scale_slot(layout, program_rmsnorm_eps(&k.func))`.
        Program::RmsNorm => L::program_rmsnorm_eps(f).into_iter().collect(),
        // `scalarmul` resolves ONE value: `scale_slot(layout, program_scalarmul_scale(&k.func))`.
        Program::ScalarMul => L::program_scalarmul_scale(f).into_iter().collect(),
        // `attn_at` resolves TWO — the score scale AND its square root, in that order:
        // `scale_idx` on `scale_val`, then `sqrt_scale_idx` on `scale_val.sqrt()`, because
        // "torch-spyre splits into sqrt_scale on both Q and K". `compute_bundle_layout` pushes the
        // same pair for `SubOp::AttnDecode` (`push_scale(*scale)` then `push_scale(scale.sqrt())`).
        // Unreachable today — `for_regions` refuses `Attn` at the reserved-const door above — but
        // stated here so the registry is not a second place the pair could go missing.
        Program::Attn => match L::program_score_scale(f) {
            Some(v) => vec![v, v.sqrt()],
            None => Vec::new(),
        },
        // ⭐ EMPTY, AND THAT IS THE ANSWER FOR THESE. None of these bodies calls `scale_slot` or
        // indexes `scalarmul_scales` — grep over the file: the only readers are `rmsnorm`,
        // `scalarmul`, `attn_at` and `scale_idx_of` (which only recognises a RESERVED tid among the
        // parameters and finds none when the registry is empty). A value registered for them would
        // occupy a slot nothing asks for.
        Program::Elementwise(_)
        | Program::SiluMul
        | Program::LmLast
        | Program::Transpose
        | Program::Matmul
        | Program::Rope => Vec::new(),
    }
}
