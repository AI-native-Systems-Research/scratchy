// SPDX-License-Identifier: Apache-2.0
//! THE PREFIX-VALIDITY RESERVATION vs THE CONTINUATION'S SCORE MASK — one synthetic
//! mask tid, two shapes, and the footprint law that keeps them from colliding.
//!
//! MEASURED ON THE POD (job 19, `runhelper-63988699`, granite-3.1-2b spyre-hw CLI
//! build at 63988699f): the width-512 prefix-capable continuation bake panicked at
//! the door with
//!
//! ```text
//!   assemble_pointwise_broadcast add_o1137: t1131: access window end 786432B
//!   (offset 262144B, operand 524288B) exceeds its placement footprint 279552B (seg3)
//! ```
//!
//! The two sides of that refusal come from two different authorities, and this file
//! pins BOTH:
//!
//! * **The window** is `attn.py`'s `_attn_cont_head` mask load — the program's own
//!   contract, correct as stated. `mask_d` is a `[MQ, KTN_ROWS]` block of the
//!   `[MQ, SWEPT + KTN_ROWS]` mask view at column `SWEPT`: at granite-2b's width-512
//!   prefix rung (MQ=512, SWEPT=256, KTN_ROWS=512) that is a `[512, 512]` window at
//!   column 256 — offset 131072 elems, end 393216 elems = 786432 B, exactly the
//!   view's full extent.
//! * **The footprint** is the placement pass's prefix-VALIDITY reservation
//!   (`lower_subtile_tape_to_superdsc`, the `ATTN_MASK_TID` block): one broadcast
//!   row per page over the derived context ceiling, `max_pages · PAGE_SLOTS · 2`
//!   bytes. At granite-3.1-2b that is
//!   `⌊16 GiB / (3 · 8 · 64 · 256 · 2 · 40 layers)⌋ = 546` pages →
//!   `546 · 256 · 2 = 279552 B` — the panic's own number, derived here from the
//!   model's geometry (nkvh=8, hd=64, 40 layers) and the one-region ceiling.
//!
//! The splice binds the attention's `desc_mask` parameter at the graph's next-free
//! tid (`mint_attn`'s law, `ir.tensors.len()`). The placement used to alias exactly
//! that tid onto the validity reservation's bytes — fusing the continuation's
//! `[MQ, SWEPT + KTN_ROWS]` score mask and the paged-decode `[1, max_pages·256]`
//! validity mask into ONE footprint — and the reservation covered only the decode
//! form. The alias is DELETED (#717): a prompt chunk's score mask now gets its OWN
//! placement sized from the program's declared view (`AttnArm::mask_view`, the one
//! statement the splice's own arm derivation makes), and only a decode bundle
//! (mq == 1) reads the runtime-staged validity bytes. This test pins the law at
//! BOTH prefill arms — the binding's placement must cover whichever form the
//! bundle's arm declares:
//!
//! * decode: `[1, swept]` (the staged validity buffer);
//! * one-pass prefill (start==0): `[MQ, KTN_ROWS]`;
//! * prefix-capable continuation: `[MQ, SWEPT + KTN_ROWS]`.
//!
//! and the third leg pins the DECODE side's own constraint: a BATCHED-decode rung
//! (rows are REQUESTS) must keep the alias at every rung, because the decode ladder
//! stages ONE layout — the macro's `ladder_stages_one_layout` guard, the pod's job-20
//! refusal.
//!
//! The width-256 prefix rung passed by arithmetic luck (its mask_d end
//! `256 · 512 · 2 = 262144 B` fit under 279552 B with 17088 B of slack); the
//! width-512 rung needed `512 · 768 · 2 = 786432 B` and refused. The refusal was
//! the LOUD half; the one-pass arm's silently capped `[512,256]` mask add (the
//! width the plain width-512 rung baked) was the quiet one, pinned by the wire
//! assert below. Both legs run at the exact granite-3.1-2b facts the pod baked,
//! through the production chain (region → validate → tape → reroll → the rolled
//! walk → the door under the walk's own layout).

use std::collections::HashSet;

use scratchy_subtile::fixtures;
use scratchy_subtile::lower::{InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    NeoX, SourceShape, SubOp, SubtileIR, ValidatedGraph, lower_region,
};
use scratchy_subtile::subtile_tape::{lower_dag_to_tape, reroll_subtile_tape};
use scratchy_target_spyre::ktir_superdsc_door::lower as door_lower;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    ATTN_MASK_TID, ActiveCap, RolledSuperDsc, RungFacts, ladder_stages_one_layout,
    lower_subtile_tape_to_superdsc,
};

/// granite-3.1-2b's attention-block geometry: hidden 2048, kv_dim 512 (8 kv-heads ×
/// head_dim 64), intermediate 8192 — the exact trio the pod's failing build baked
/// (`configs/granite/granite-3.1-2b-instruct.json`).
const H: u32 = 2048;
const KV: u32 = 512;
const I: u32 = 8192;
const HD: u32 = 64;
/// The model's true layer count — LOAD-BEARING for the footprint, not set dressing:
/// `max_pages` divides the one-region ceiling by `layer_stride · 2 · layers`, and
/// granite-3.1-2b's 40 layers is what makes the validity reservation 546 pages
/// (279552 B) rather than the 780 a 28-layer model would reserve. A two-layer
/// fixture would reserve 10922 pages and the overrun would NOT reproduce.
const LAYERS: u32 = 40;
/// The width-512 prefix rung's chunk: MQ query rows over a 256-slot resident prefix
/// (`PrefixCapacity` 256 = `prefix_cap_default`, the value the pod's env-free build
/// derived from `min(max_position_embeddings, 256)`).
const MQ: u32 = 512;
const PREFIX: u32 = 256;
/// A decode batch width — one of `PagedKvPool::BATCH_RUNGS` `{2, 4, 8, 16, 32}`: the
/// widest rung the pod's job-20 bake lowered (rows=32, `[32, swept + 64]` mask view,
/// 20480 B at the ceiling vs 8192 B at active_cap=64, tid 1131).
const BATCH: u32 = 8;

/// The one-layer fixture at granite-3.1-2b's geometry, restated as a CHUNK at an
/// arbitrary row count: every op at `m = rows`, per-position rope tables at the
/// production widths (the q-site pair `[rows, H]`, the K-site append's own-width pair
/// `[rows, KV]` — the head-major test's per-width memoization law), and the two cache
/// sources at the resident prefix capacity `[PREFIX, KV]` — the capacity the splice
/// resolves `SWEPT` from under `ActiveCap::FULL`. `rows = MQ` is the prefill chunk;
/// `rows = BATCH` (a request-batch width) is the batched-decode leg.
fn chunk_layer_input(rows: u32) -> LoweringInput {
    let mut input = fixtures::one_layer_input_shaped(H, KV, I, HD);
    input.sources[0] = SourceShape { rows, cols: H }; // 0  x — the chunk's rows
    input.sources[5] = SourceShape { rows, cols: H }; // 5  cos, q-site [rows, heads·hd]
    input.sources[6] = SourceShape { rows, cols: H }; // 6  sin, q-site
    input.sources[7] = SourceShape {
        rows: PREFIX,
        cols: KV,
    }; // 7  prefix_k cache [cap=PREFIX, kv]
    input.sources[8] = SourceShape {
        rows: PREFIX,
        cols: KV,
    }; // 8  prefix_v cache [cap=PREFIX, kv]
    input.sources.push(SourceShape { rows, cols: KV }); // 14 cos, k-site [rows, kv]
    input.sources.push(SourceShape { rows, cols: KV }); // 15 sin, k-site
    // The K-side append (op 5) reads its own-width pair — the per-width law.
    input.ops[5].inputs[1] = InputRef::Ext(14);
    input.ops[5].inputs[2] = InputRef::Ext(15);
    for op in &mut input.ops {
        op.m = rows;
    }
    // The attention's valid_len is the modeled prefix capacity — the bridge's own
    // stamp (`valid_len = bx.prefix_len`), decoupled from the cache TENSOR's rows
    // only in the fixture's decode rigging, which this restatement removes.
    for op in &mut input.ops {
        if let SubOp::AttnDecode { valid_len, .. } = &mut op.op {
            *valid_len = PREFIX;
        }
    }
    input
}

/// `n` copies of the chunk-shaped layer, each with its OWN weights/KV sources
/// (the `n_layer_input_distinct` law), under the one-op embedding stand-in the
/// re-roll needs (every layer's input Computed — the head-major test's prologue
/// trick). Source ids: 0=x, 1=cos_q, 2=sin_q, 3=cos_k, 4=sin_k (all five SHARED),
/// then `n` blocks of eleven per-layer sources; layer `L`'s block starts at
/// `5 + L·11`. Returns the input with the staged-WEIGHT source id set — every
/// per-layer source except the two KV caches. `rows` is the chunk's row count
/// (MQ for the prefill legs, a batch width for the batched-decode leg).
fn n_layer_of(base: &LoweringInput, rows: u32, n: u32) -> (LoweringInput, HashSet<u32>) {
    let body = base.ops.clone();
    let ops_per_layer = body.len(); // 15
    // Base source ids: 0=x, 1=rms_w0, 2=q_w, 3=k_w, 4=v_w, 5=cos_q, 6=sin_q,
    // 7=prefix_k, 8=prefix_v, 9=o_w, 10=rms_w1, 11=gate_w, 12=up_w, 13=down_w,
    // 14=cos_k, 15=sin_k. SHARED: 0 and the four rotary-table ids. PER-LAYER: the
    // other eleven.
    let per_layer_src_ids: Vec<usize> = (1..base.sources.len())
        .filter(|e| !matches!(e, 5 | 6 | 14 | 15))
        .collect();
    let n_per = per_layer_src_ids.len(); // 11
    let mut sources = vec![
        base.sources[0],
        base.sources[5],
        base.sources[6],
        base.sources[14],
        base.sources[15],
    ];
    for _ in 0..n {
        for &e in &per_layer_src_ids {
            sources.push(base.sources[e]);
        }
    }
    let map_ext = |e: usize, layer: usize| -> usize {
        match e {
            5 => 1,  // cos_q (shared)
            6 => 2,  // sin_q (shared)
            14 => 3, // cos_k (shared)
            15 => 4, // sin_k (shared)
            _ => {
                5 + layer * n_per
                    + per_layer_src_ids
                        .iter()
                        .position(|&x| x == e)
                        .expect("per-layer src")
            }
        }
    };
    let mut ops: Vec<OpDesc> = Vec::with_capacity(1 + n as usize * ops_per_layer);
    // Op 0 — the embedding stand-in (granite's `embedding_multiplier`, the head-major
    // test's prologue): makes every layer's input Computed so the re-roll finds one
    // repeating body. It reads NO weight, so the staged-weight set is untouched.
    ops.push(OpDesc {
        op: SubOp::ScalarMul { scale: 12.0 },
        m: rows,
        inputs: vec![InputRef::Ext(0)],
    });
    let mut weight_ids = HashSet::new();
    for layer in 0..n as usize {
        let base_op = 1 + layer * ops_per_layer;
        for op in body.iter() {
            let inputs = op
                .inputs
                .iter()
                .map(|inp| match inp {
                    InputRef::Op(j) => InputRef::Op(base_op + *j),
                    // Ext(0) = the layer-input x: the prologue's output for layer 0,
                    // the previous layer's residual output for L>0 — ONE rule.
                    InputRef::Ext(0) => InputRef::Op(base_op - 1),
                    InputRef::Ext(e) => InputRef::Ext(map_ext(*e, layer)),
                })
                .collect();
            let mut op = OpDesc {
                op: op.op,
                m: rows,
                inputs,
            };
            // The KV writer's layer id is per-copy state — production graphs stamp
            // each layer's rope append with its TRUE index.
            let true_layer = layer as u32;
            if let SubOp::RopeAppend { layer, .. } | SubOp::KvAppend { layer, .. } = &mut op.op {
                *layer = true_layer;
            }
            ops.push(op);
        }
        // This layer's staged weights: the eleven per-layer sources minus the caches.
        for &e in &per_layer_src_ids {
            if e != 7 && e != 8 {
                weight_ids.insert(map_ext(e, layer) as u32);
            }
        }
    }
    (
        LoweringInput {
            sources,
            result: ops.len() - 1,
            exported: None,
            ops,
        },
        weight_ids,
    )
}

/// The production chain over the fixture — the codegen's own four steps (region →
/// validate → tape → reroll) and the rolled walk, at the PREFIX bake's own argument
/// values: `ActiveCap::FULL` (the continuation sweep), `rows_are_requests = false`
/// (a prompt chunk's rows are one request's positions — `!is_prefill && decode_rows
/// > 1` is the emission's own derivation), `one_request = false` (a prefill bake).
fn walk(active_cap: ActiveCap) -> (RolledSuperDsc, SubtileIR<NeoX>) {
    walk_shaped(active_cap, MQ, false, LAYERS)
}

/// [`walk`] at an arbitrary row kind: `rows_are_requests = true` is the BATCH
/// ladder's own bake fact (a batch rung's rows are REQUESTS), the shape the
/// decode-ladder legs below lower at.
fn walk_shaped(
    active_cap: ActiveCap,
    rows: u32,
    rows_are_requests: bool,
    layers: u32,
) -> (RolledSuperDsc, SubtileIR<NeoX>) {
    let (input, weight_ids) = n_layer_of(&chunk_layer_input(rows), rows, layers);
    let rg = lower_region(&input, std::num::NonZeroU32::new(8192).unwrap());
    let valid = ValidatedGraph::new(&rg).expect("the granite-2b chunk graph validates");
    let tape = lower_dag_to_tape(&valid);
    let rolled_tape = reroll_subtile_tape(&tape, &rg);
    let rolled = lower_subtile_tape_to_superdsc(
        &rolled_tape,
        &rg,
        &weight_ids,
        active_cap,
        rows_are_requests,
        RungFacts {
            one_request: false,
            chunk_pages: None,
        },
    )
    .expect("the granite-2b prefix-capable chunk lowers through the rolled walk");
    (rolled, rg)
}

/// The mask's binding tid — `mint_attn`'s own law (`ir.tensors.len()`), restated
/// here so the test names the same tensor the splice binds and the placement
/// aliases. One buffer, two names: the reserved const and the binding tid.
fn mask_binding_tid(rg: &SubtileIR<NeoX>) -> u32 {
    rg.tensors.len() as u32
}

/// The body's attention op — the splice's `attn_s{id}` row, the program whose mask
/// load is the law under test.
fn attention_op(rolled: &RolledSuperDsc) -> &ktir_superdsc::emit::EmittedOp {
    rolled
        .bodies
        .iter()
        .flat_map(|b| b.ops.iter())
        .find(|e| e.op_name.starts_with("attn_s"))
        .expect("the body carries the attention's spliced row")
}

/// ⭐ THE PREFIX-CAPABLE CONTINUATION — the width-512 prefix rung's own facts. The
/// reservation must cover the continuation's `[MQ, SWEPT + KTN_ROWS]` score mask
/// (786432 B at these extents), and the door must lower the attention's program
/// under the walk's own layout — the exact bake that panicked on the pod.
/// ⛔ RED-BY-DESIGN, `#[ignore]`d until #717's placement work lands (flex-rs's branch):
/// the mask binding tid's placement alias — one buffer, two names, the reserved const
/// and the splice's binding tid — and the reservation's `[MQ, SWEPT + KTN_ROWS]` sizing
/// are that branch's laws. This file rides the pointwise Win PR verbatim (its driver is
/// the rolled walk the mint gates) so the law is pinned where the mint landed; un-ignore
/// when the placement does.
#[test]
#[ignore = "red-by-design pending #717's placement fix (the binding-tid alias and the reservation sizing)"]
fn the_prefix_continuation_mask_reservation_covers_the_score_mask() {
    let (rolled, rg) = walk(ActiveCap::FULL);

    // THE RESERVATION LAW — the validity-shaped reservation must cover the
    // continuation's score mask, whatever the derived page ceiling made it.
    let mask_tid = mask_binding_tid(&rg);
    let placed = rolled.layout.placements.get(&mask_tid).expect(
        "the mask's binding tid is placed (the splice's and the placement's mint laws agree)",
    );
    let need = u64::from(MQ) * u64::from(PREFIX + MQ) * 2;
    assert!(
        placed.size >= need,
        "the mask reservation is {} B but the width-512 continuation's score mask \
         [MQ={MQ}, SWEPT={PREFIX} + KTN_ROWS={MQ}] needs {need} B — the pod's own \
         refusal (add_o1137: end 786432B vs footprint 279552B, runhelper-63988699)",
        placed.size
    );

    // THE DOOR — the descriptors that reach the card, under the SAME layout the
    // tape walk computed. This is the bake that panicked; a reservation that covers
    // the window is what lets it lower.
    let attn = attention_op(&rolled);
    let k = attn
        .ktir
        .as_ref()
        .expect("the attention row carries its program");
    let mut sym = 0i64;
    door_lower(k, &mut sym, Some(&rolled.layout), rolled.attn_params)
        .unwrap_or_else(|e| panic!("the continuation's attention lowered: {}", e.message));
}

/// The ONE-PASS arm at the same rung — a start==0 chunk's `[MQ, KTN_ROWS]` causal
/// triangle over the SAME synthetic mask tid. The reservation must cover it too:
/// 524288 B at these extents, against the same 279552 B validity shape.
/// ⛔ RED-BY-DESIGN, `#[ignore]`d until #717's emission+placement work lands (flex-rs's
/// branch): the ONE-PASS whole-view mask add (`add_o{t}` sweeping `[MQ, KTN_ROWS]`) is
/// that branch's emission — main's one-pass arm masks inside the attention's own chunk
/// legs and emits no whole-view add for this rung to narrow. The cap law this leg pins
/// (a too-small reservation must refuse, never silently narrow) is the pointwise PR's;
/// the add it reads is not main's to emit.
#[test]
#[ignore = "red-by-design pending #717's one-pass mask-add emission (flex-rs's branch)"]
fn the_one_pass_mask_reservation_covers_the_causal_triangle() {
    let (rolled, rg) = walk(ActiveCap::NONE);

    let mask_tid = mask_binding_tid(&rg);
    // THE DOOR — the pod's PLAIN width-512 prefill rung (this arm) baked clean, so the
    // measurement here is what the door EMITS, not whether it refuses: the whole-view
    // mask add never hit the footprint guard, and the cap law below was the silent
    // defect instead.
    let attn = attention_op(&rolled);
    let k = attn
        .ktir
        .as_ref()
        .expect("the attention row carries its program");
    let mut sym = 0i64;
    let emitted = door_lower(k, &mut sym, Some(&rolled.layout), rolled.attn_params)
        .unwrap_or_else(|e| panic!("the one-pass attention lowered: {}", e.message));
    // THE WIRE — the add's own descriptor states the emitted sweep. The cap law
    // (`pointwise_width_the_output_holds`) narrows it when the mask's placement
    // can't hold the asked width, and its reproduction proof passes VACUOUSLY on
    // a reservation of the wrong SHAPE (the [1, 139776] validity form under a
    // [512,512] handle): held = 279552/512 B/row → 273 elems → floor-stick 256,
    // and 256 re-pads to the asked 512 — so the pod's PLAIN width-512 prefill
    // rung baked an add at [512,256], sweeping half the causal triangle and
    // leaving the other half of the output unwritten (downstream max/sub/exp
    // read all 512). No refusal. This is the reservation's SECOND symptom.
    let add = emitted
        .iter()
        .find(|o| o.op_name == "add_o1049")
        .expect("the one-pass mask add is emitted");
    let v = serde_json::to_value(&add.op).expect("the add's descriptor serializes");
    let dsc = v["dscs_"][0]
        .as_object()
        .expect("the add's per-core dsc map");
    let (_, dsc) = dsc.iter().next().expect("the add's dsc");
    let iter = &dsc["N_"];
    let (mb, out) = (
        iter["mb_"].as_i64().expect("the add's row extent"),
        iter["out_"].as_i64().expect("the add's column extent"),
    );
    assert_eq!(
        (mb, out),
        (i64::from(MQ), i64::from(MQ)),
        "the one-pass mask add sweeps [{mb},{out}] but the causal triangle is \
         [{MQ},{MQ}] — the width the pod's plain width-512 prefill rung baked"
    );

    let placed = rolled.layout.placements.get(&mask_tid).expect(
        "the mask's binding tid is placed (the splice's and the placement's mint laws agree)",
    );
    let need = u64::from(MQ) * u64::from(MQ) * 2;
    assert!(
        placed.size >= need,
        "the mask reservation is {} B but the one-pass causal triangle \
         [MQ={MQ}, KTN_ROWS={MQ}] needs {need} B over the same synthetic tid",
        placed.size
    );
}

/// ⭐ THE BATCHED-DECODE LADDER — the pod's job-20 refusal, pinned as the macro's own
/// guard states it. A batch rung's rows are REQUESTS (`rows_are_requests = true`, a
/// `PagedKvPool::BATCH_RUNGS` width), and every rung of a decode session shares ONE
/// host-staged layout with the ceiling (`ladder_stages_one_layout`, the codegen's
/// rung-vs-ceiling check): the mask binding must alias the runtime validity buffer at
/// EVERY rung — never the prompt chunk's own-placement arm, whose `[rows, swept +
/// ktn_rows]` size varies with the rung's swept extent (20480 B at the ceiling vs
/// 8192 B at active_cap=64 on the pod, tid 1131 — bytes no rung body staged).
///
/// Two layers, not forty: this leg pins the LAYOUT LAW (placement equality across
/// rungs), which is layer-count-invariant — the prefill legs above carry the forty
/// because the 546-page reservation arithmetic is load-bearing there.
/// ⛔ RED-BY-DESIGN, `#[ignore]`d with the two legs above (the same binding-tid
/// placement alias, at the batched-decode ladder's rungs).
#[test]
#[ignore = "red-by-design pending #717's placement fix (the binding-tid alias at every rung)"]
fn a_batched_decode_rung_stages_the_ceilings_mask_layout() {
    let (ceiling, rg) = walk_shaped(ActiveCap::FULL, BATCH, true, 2);
    let (rung, _) = walk_shaped(ActiveCap::new(64), BATCH, true, 2);

    // THE ONE-LAYOUT LAW — the macro's own guard, verbatim: the exact check the pod's
    // job-20 bake panicked on, over every host-staged activation the two layouts place.
    ladder_stages_one_layout(&ceiling.layout, &rung.layout).unwrap_or_else(|e| {
        panic!("the batched-decode rung stages the ceiling's layout (job 20): {e}")
    });

    // THE STAGING CONTRACT — at BOTH rungs the binding reads the validity bytes: same
    // segment, same offset, same size as `ATTN_MASK_TID`'s reservation.
    let mask_tid = mask_binding_tid(&rg);
    for (name, layout) in [("ceiling", &ceiling.layout), ("rung 64", &rung.layout)] {
        let bound = layout.placements.get(&mask_tid).expect(
            "the mask's binding tid is placed (the splice's and the placement's mint laws agree)",
        );
        let staged = layout
            .placements
            .get(&ATTN_MASK_TID)
            .expect("the validity reservation is placed");
        assert_eq!(
            (bound.segment, bound.offset, bound.size),
            (staged.segment, staged.offset, staged.size),
            "the {name} batched-decode binding must alias the runtime validity buffer"
        );
    }
}
