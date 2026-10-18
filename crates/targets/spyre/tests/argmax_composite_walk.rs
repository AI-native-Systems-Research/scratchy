// SPDX-License-Identifier: Apache-2.0
//! THE COMPOSITE ARGMAX'S WALK MINT — #603 PR3a's loop closure, emission side.
//!
//! `zz_the_argmax_composite_probes.rs` (the mint's own crate) pins the chain's assembly;
//! `argmax_composite_door.rs` (this crate) pins the door arm that turns a minted node
//! into descriptors byte-identical to that assembly. What is pinned HERE is the THIRD
//! seam — the WALK (`lower_subtile_tape_to_ktir`) deciding, from the graph's own result
//! tail, that a tied one-request rung gets the composite, and minting it into the
//! suffix as ONE KTIR-first op:
//!
//!   • the mint FIRES behind granite's own tail shape (the Dense lm-head `MatmulTile`
//!     under a logits-scaling `ScalarMul`) and behind the tail-less form (qwen's:
//!     result straight off the matmul) — the recognition walks the tail it finds;
//!   • the minted op is KTIR-first (`op: None, ktir: Some` — the only legal bundle
//!     op), appended LAST in the suffix (WAR-safe: everything that reads the residual
//!     buffer has already run), named by the ROW tid (rung-stable), and bound over the
//!     graph's OWN slots: `[logits = ir.result, table = the lm-head weight, row = the
//!     first body's hidden-in, which IS the embedding-source slot]`;
//!   • the layout PLACED the two resident fp32 constants (the `[1, vocab]` ramp and the
//!     scalar-V stick, activation-segment, `size = vocab·4`) and REGISTERED their names
//!     — the router-const law: the door names, the layout places — and the walk's own
//!     declare pass declared the seven scratch synthetics at the row (the seg0-overrun
//!     ordering law's precondition, exercised through the real walk);
//!   • `None` (untied, batched, prefill) mints NOTHING and places NOTHING — the host
//!     readback path is untouched for every model that does not declare tied;
//!   • a tied graph the walk cannot recognize REFUSES LOUDLY: a staged (fp8 W8A8)
//!     lm-head and a multi-row result are build errors naming the fact, never a silent
//!     host fallback — the model declared tied, so a quiet host round-trip would hide a
//!     broken emission behind a working one.
//!
//! The fixture is the head-major handoff file's own replication law — the shared
//! `one_layer_input_shaped` fixture, rope tables widened the door's way, replicated
//! per layer with distinct weights under a one-op prologue standing in for the
//! embedding — with the lm-head tail APPENDED after the last layer: the Dense
//! `MatmulTile { n: VOCAB }` over the last layer's residual, then (granite's leg) a
//! logits-scaling `ScalarMul`, with the graph's result on the SCALED logits. Two
//! layers is the minimum that both rerolls and proves the per-layer family; the
//! geometry is small and stick-legal (the door test holds the descriptor side at the
//! same vocab).

use std::collections::HashSet;

use ktir_superdsc::emit::argmax::TiedLmHeadEmbedding;
use ktir_superdsc::ktir_node::Program;
use ktir_superdsc::place::{PlaceId, SynthRole, act_name};
use ktir_superdsc::placement::SegRole;
use ktir_superdsc::reserved_tids::{argmax_ramp_tid, argmax_v_tid};
use scratchy_subtile::fixtures;
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    NeoX, SourceShape, SubOp, SubtileIR, ValidatedGraph, lower_region,
};
use scratchy_subtile::subtile_tape::{lower_dag_to_tape, reroll_subtile_tape};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    ActiveCap, RolledSuperDsc, RungFacts, lower_subtile_tape_to_superdsc,
};

/// Small stick-legal geometry: hidden/kv/intermediate 256 (four sticks), head_dim 64
/// (one slab), vocab 128 (two sticks — the door test's own small vocab; the probes hold
/// granite's 49152-wide geometry on the assembly side).
const H: u32 = 256;
const KV: u32 = 256;
const I: u32 = 256;
const HD: u32 = 64;
const VOCAB: u32 = 128;
/// Two layers: the reroll needs a repeating body, and `per_layer` needs a second entry
/// to prove the family (not just one placement) — the head-major file's own minimum.
const LAYERS: u32 = 2;
/// The prologue's embedding multiplier — granite-3.1's own stand-in, and ⛔ NEVER 1.0
/// (a multiply by 1.0 is exactly what constant folding erases, which breaks the reroll
/// the head-major file documents).
const EMBED_SCALE: f32 = 12.0;
/// granite's logits-scaling stand-in — the tail op the recognition must walk THROUGH.
/// f16-exact, so the scalarmul const bind carries it without registry subtleties.
const LOGITS_SCALE: f32 = 0.5;

fn tied() -> TiedLmHeadEmbedding {
    TiedLmHeadEmbedding::from_tied_config(true).expect("granite ties")
}

/// The one-layer fixture at the small geometry, rope tables widened the door's way (the
/// shared fixture's `[1, hd]` rows suit subtile-level tests; the walk's declare pass
/// runs the door, whose rope reads `[1, heads·hd]` — q-site `[1, H]`, k-site `[1, KV]`).
fn one_layer_roped() -> LoweringInput {
    let mut input = fixtures::one_layer_input_shaped(H, KV, I, HD);
    input.sources[5] = SourceShape { rows: 1, cols: H }; // 5: cos, q-site tiled [1, 4·64]
    input.sources[6] = SourceShape { rows: 1, cols: H }; // 6: sin, q-site tiled
    input.sources.push(SourceShape { rows: 1, cols: KV }); // 14: cos, k-site tiled [1, 4·64]
    input.sources.push(SourceShape { rows: 1, cols: KV }); // 15: sin, k-site tiled
    // The K-side append (op 5) reads its own-width pair — inputs 1, 2 are its cos, sin.
    input.ops[5].inputs[1] = InputRef::Ext(14);
    input.ops[5].inputs[2] = InputRef::Ext(15);
    // The prefix caches at ONE PAGE (256 rows) — production's own resident-cache cap
    // (`prefix_cap_default`, the golden file's law). The shared fixture's stock 8 rows
    // would clamp every legal rung to the same sweep (`swept.min(tensor rows)` with the
    // 64-row rung stick), making the ladder assert below vacuous; at a page the
    // interior rungs 64/128 genuinely sweep less than the ceiling.
    input.sources[7] = SourceShape {
        rows: 256,
        cols: KV,
    };
    input.sources[8] = SourceShape {
        rows: 256,
        cols: KV,
    };
    input
}

/// `LAYERS` copies of the layer, each with its OWN weights/KV (the
/// `n_layer_input_distinct` law), under the one-op prologue standing in for the
/// embedding, with the lm-head tail appended: the Dense `MatmulTile { n: VOCAB }` over
/// the last layer's residual, then — when `scale_tail` — granite's logits-scaling
/// `ScalarMul`, with the result on the SCALED logits. The lm-head table is a SHARED
/// source (one table, read once — the tied embedding the composite gathers), and it
/// joins the staged-weight set.
///
/// Source ids: 0=x, 1=cos_q, 2=sin_q, 3=cos_k, 4=sin_k (all five SHARED — x is the
/// global input, the rotary tables are layer-invariant), then `n` blocks of eleven
/// per-layer sources, then the lm-head table LAST.
fn n_layer_tied(n: u32, scale_tail: bool) -> (LoweringInput, HashSet<u32>) {
    let base = one_layer_roped();
    let body = base.ops.clone();
    let ops_per_layer = body.len();
    // Base source ids 0..=15: the five shared above, plus per-layer 1,2,3,4 (rms_w0, q_w,
    // k_w, v_w), 7,8 (prefix_k, prefix_v), 9 (o_w), 10 (rms_w1), 11,12,13 (gate, up, down).
    let per_layer_src_ids: Vec<usize> = (1..base.sources.len())
        .filter(|e| !matches!(e, 5 | 6 | 14 | 15))
        .collect();
    let n_per = per_layer_src_ids.len();
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
    // The lm-head table, in the IR's own `[K, N]` spelling — `[hidden, vocab]`. The
    // mint's `[vocab, hidden]` view is the tied bank's out-major staging form over the
    // same placement (the embedding orientation, `emit::argmax`'s module law).
    sources.push(SourceShape {
        rows: H,
        cols: VOCAB,
    });
    let lm_table = sources.len() - 1;
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
    let mut ops: Vec<OpDesc> = Vec::with_capacity(1 + n as usize * ops_per_layer + 2);
    // Op 0 — the embedding stand-in (the head-major file's own law: it makes every
    // layer's input Computed, which is the production shape the reroll needs).
    ops.push(OpDesc {
        op: SubOp::ScalarMul { scale: EMBED_SCALE },
        m: 1,
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
                    // the previous layer's residual output for L>0 — ONE rule, which is
                    // what makes every layer a copy of one repeating body.
                    InputRef::Ext(0) => InputRef::Op(base_op - 1),
                    InputRef::Ext(e) => InputRef::Ext(map_ext(*e, layer)),
                })
                .collect();
            let mut op = OpDesc {
                op: op.op,
                m: op.m,
                inputs,
            };
            // The KV writer's layer id is per-copy state (the head-major file's own law:
            // a replicated layer-0 id would name one layer twice).
            let true_layer = layer as u32;
            if let SubOp::RopeAppend { layer, .. } | SubOp::KvAppend { layer, .. } = &mut op.op {
                *layer = true_layer;
            }
            ops.push(op);
        }
        for &e in &per_layer_src_ids {
            if e != 7 && e != 8 {
                weight_ids.insert(map_ext(e, layer) as u32);
            }
        }
    }
    // ⭐ THE LM-HEAD TAIL — the suffix the composite closes. The matmul reads the last
    // layer's residual and the shared table; the tail op (when present) scales the
    // logits the way granite's config does. The result rides the TAIL's output when it
    // is there and the matmul's when it is not — both are the tail the recognition
    // walks.
    let last_out = ops.len() - 1;
    let lm = ops.len();
    ops.push(OpDesc {
        op: SubOp::MatmulTile {
            n: VOCAB,
            weight: GemmWeight::Dense,
        },
        m: 1,
        inputs: vec![InputRef::Op(last_out), InputRef::Ext(lm_table)],
    });
    let result = if scale_tail {
        let scaled = ops.len();
        ops.push(OpDesc {
            op: SubOp::ScalarMul {
                scale: LOGITS_SCALE,
            },
            m: 1,
            inputs: vec![InputRef::Op(lm)],
        });
        scaled
    } else {
        lm
    };
    weight_ids.insert(lm_table as u32);
    (
        LoweringInput {
            sources,
            result,
            exported: None,
            ops,
        },
        weight_ids,
    )
}

/// The production chain over the fixture — the codegen's own four steps (region →
/// validate → tape → reroll) and the rolled walk, at the ONE-REQUEST rung's own
/// argument values (`rows_are_requests = false`, no chunk), with the tie token the
/// expansion would mint — at the cap rung given (FULL for the single-bundle legs; the
/// ladder assert passes the interior rungs).
fn walk_tied(
    scale_tail: bool,
    argmax: Option<TiedLmHeadEmbedding>,
    cap: ActiveCap,
) -> (RolledSuperDsc, SubtileIR<NeoX>) {
    let (input, weight_ids) = n_layer_tied(LAYERS, scale_tail);
    let rg = lower_region(&input, std::num::NonZeroU32::new(8192).unwrap());
    let valid = ValidatedGraph::new(&rg).expect("the tied two-layer graph validates");
    let tape = lower_dag_to_tape(&valid);
    let rolled_tape = reroll_subtile_tape(&tape, &rg);
    let rolled = lower_subtile_tape_to_superdsc(
        &rolled_tape,
        &rg,
        &weight_ids,
        cap,
        false,
        RungFacts {
            one_request: true,
            chunk_pages: None,
            argmax,
        },
    )
    .unwrap_or_else(|e| panic!("the tied one-request walk: {e:?}"));
    (rolled, rg)
}

/// The lm-head matmul node — the ONLY matmul at the vocab width (every layer
/// projection is 256-wide), found by DATA, never by position.
fn lm_head_node(rg: &SubtileIR<NeoX>) -> &scratchy_subtile::subtile_ir::SubtileNode<NeoX> {
    rg.nodes
        .iter()
        .find(|n| matches!(&n.op, SubOp::MatmulTile { n, .. } if *n == VOCAB))
        .expect("the lm-head matmul is in the graph")
}

/// The prologue node — the embedding stand-in, the ONLY scalarmul at the embedding
/// multiplier (the tail's scale differs).
fn prologue_node(rg: &SubtileIR<NeoX>) -> &scratchy_subtile::subtile_ir::SubtileNode<NeoX> {
    rg.nodes
        .iter()
        .find(|n| matches!(&n.op, SubOp::ScalarMul { scale } if *scale == EMBED_SCALE))
        .expect("the embedding prologue is in the graph")
}

/// The mint FIRES behind granite's own tail shape, and every fact it states is read
/// back off the graph's own slots: the op is KTIR-first and LAST in the suffix, its
/// bindings are `[result, table, row]` over the graph's own tids, the layout placed
/// and registered the two resident fp32 constants, and the walk's own declare pass
/// declared the seven scratch synthetics at the row.
#[test]
fn the_walk_mints_the_composite_behind_the_granite_tail() {
    let (rolled, rg) = walk_tied(true, Some(tied()), ActiveCap::FULL);
    let last = rolled
        .suffix
        .last()
        .expect("the suffix holds the composite");
    assert!(
        last.op_name.starts_with("argmax_o"),
        "the composite is named by its row tid: {}",
        last.op_name
    );
    // KTIR-first: the program rides, and no producer-built descriptor does — the only
    // legal bundle op (`ktir_groups` / `ktir_groups_via_superdsc` both refuse the rest).
    let k = last
        .ktir
        .as_ref()
        .expect("the composite carries its KTIR program");
    assert!(last.op.is_none(), "no producer-built descriptor");
    assert!(
        matches!(k.program, Program::Argmax(_)),
        "the program is the composite's own kind"
    );
    // THE BINDING CONVENTION over the graph's own slots: [logits = ir.result, table =
    // the lm-head weight, row = the first body's hidden-in = the embedding source].
    let tids: Vec<u32> = k.bindings.iter().map(|b| b.get()).collect();
    let table_tid = lm_head_node(&rg).inputs[1].tensor.index() as u32;
    let row_tid = prologue_node(&rg).output.tensor.index() as u32;
    assert_eq!(
        tids,
        vec![rg.result.index() as u32, table_tid, row_tid],
        "the composite binds [logits, table, row] over the graph's own tids"
    );
    // The name keys on the ROW (the door's op-name law) — rung-stable, and the walk's
    // row IS the embedding-source slot the prologue writes.
    assert_eq!(last.op_name, format!("argmax_o{row_tid}"));
    // The layout placed the two resident fp32 constants at the vocab the table states,
    // and registered their names (the router-const law: the door names, the layout
    // places). fp32 — 4 bytes/elem, never the f16 the const bind narrows to.
    let ramp = rolled
        .layout
        .placements
        .get(&argmax_ramp_tid())
        .expect("the ramp is placed");
    assert_eq!(
        ramp.size,
        u64::from(VOCAB) * 4,
        "the ramp is [1, vocab] fp32"
    );
    assert_eq!(
        ramp.role,
        SegRole::Activation,
        "the ramp is an activation const"
    );
    assert_eq!(
        rolled.layout.id_of(&act_name(argmax_ramp_tid())),
        Some(PlaceId::Act(argmax_ramp_tid())),
        "the ramp's name is registered at its own identity"
    );
    let v = rolled
        .layout
        .placements
        .get(&argmax_v_tid())
        .expect("the scalar-V stick is placed");
    assert_eq!(v.size, 128, "the V stick is [1, 32] fp32");
    assert_eq!(
        v.role,
        SegRole::Activation,
        "the V stick is an activation const"
    );
    // The walk's own declare pass declared the seven scratch synthetics at the row —
    // the seg0-overrun ordering law's precondition, exercised through the REAL walk
    // (not a hand-built layout). Two formats checked: the fp32 mask (vocab-wide) and
    // the fp16 max (one stick).
    let row_id = PlaceId::Act(row_tid);
    let declared = |role: SynthRole, want: u64| {
        let name = row_id.synth(role).to_string();
        let got = rolled
            .layout
            .synth
            .borrow()
            .sizes
            .get(&name)
            .unwrap_or_else(|| panic!("'{name}' was not declared by the walk"))
            .to_owned();
        assert_eq!(got, want, "'{name}' footprint");
    };
    declared(SynthRole::AmMask32, u64::from(VOCAB) * 4);
    declared(SynthRole::AmMax, 128); // [1, 64] fp16 — one stick
}

/// The tail-less form — qwen's: the result straight off the matmul — mints too. The
/// recognition walks the tail it finds, and zero tail ops is a tail.
#[test]
fn the_walk_mints_the_composite_for_a_tailless_tied_head() {
    let (rolled, rg) = walk_tied(false, Some(tied()), ActiveCap::FULL);
    let last = rolled
        .suffix
        .last()
        .expect("the suffix holds the composite");
    let k = last
        .ktir
        .as_ref()
        .expect("the composite carries its program");
    let tids: Vec<u32> = k.bindings.iter().map(|b| b.get()).collect();
    let table_tid = lm_head_node(&rg).inputs[1].tensor.index() as u32;
    let row_tid = prologue_node(&rg).output.tensor.index() as u32;
    assert_eq!(
        tids,
        vec![rg.result.index() as u32, table_tid, row_tid],
        "the tail-less result binds as the logits slot"
    );
    assert!(rolled.layout.placements.contains_key(&argmax_ramp_tid()));
}

/// `None` — untied, batched, prefill — mints NOTHING and places NOTHING: the host
/// readback path is untouched for every model that does not declare tied, and the
/// suffix is exactly the tail the graph stated (the matmul and, when present, the
/// scale), never one op more.
#[test]
fn an_untied_rung_keeps_the_host_path() {
    let (tied_rolled, _) = walk_tied(true, Some(tied()), ActiveCap::FULL);
    let (plain_rolled, _) = walk_tied(true, None, ActiveCap::FULL);
    assert_eq!(
        plain_rolled.suffix.len() + 1,
        tied_rolled.suffix.len(),
        "the tie token adds exactly one op — the composite — and nothing else"
    );
    assert!(
        plain_rolled
            .suffix
            .iter()
            .all(|e| !e.op_name.starts_with("argmax_o")),
        "no composite op is minted without the tie token"
    );
    assert!(
        !plain_rolled
            .layout
            .placements
            .contains_key(&argmax_ramp_tid()),
        "the ramp is not placed without the tie token"
    );
    assert!(
        !plain_rolled.layout.placements.contains_key(&argmax_v_tid()),
        "the V stick is not placed without the tie token"
    );
}

/// ⭐⭐⭐ THE NaN SENTINEL ROW — the composite answers an all-NaN (or all-−inf) logits row
/// with idx = V, ONE PAST the table's last row, and the gather then reads
/// `[V·H·2, (V+1)·H·2)` of the table's staging. The layout extends the table's OWN
/// placement by exactly one fp16 row (`hidden · 2` B): the staging writes only `V·H·2`
/// bytes (its bind's own length — the over-bind law keeps the tail free) into a
/// zero-filled shadow, so row V reads ZEROS — in-bounds by construction, never by hoping
/// the next placement is harmless. The pinned fact is the DELTA against the SAME fixture
/// walked untied (not a fixture-specific byte count), the extension never MOVES the table,
/// and a layout that RETURNS has already passed `audit_layout_addresses` (it runs inside
/// `compute_bundle_layout`), so the extended placement is proven inside its segment.
#[test]
fn the_table_carries_one_sentinel_row_when_the_composite_mints() {
    let (tied_rolled, rg) = walk_tied(true, Some(tied()), ActiveCap::FULL);
    let (plain_rolled, _) = walk_tied(true, None, ActiveCap::FULL);
    let table_tid = lm_head_node(&rg).inputs[1].tensor.index() as u32;
    let bare = plain_rolled
        .layout
        .placements
        .get(&table_tid)
        .expect("the table is placed on the untied walk too — the extension is the only delta");
    let ext = tied_rolled
        .layout
        .placements
        .get(&table_tid)
        .expect("the table is placed");
    assert_eq!(
        ext.size,
        bare.size + u64::from(H) * 2,
        "the composite extends the table by EXACTLY one fp16 row — the idx=V gather's read"
    );
    assert_eq!(
        ext.offset, bare.offset,
        "the extension grows the placement's TAIL; it never moves the table"
    );

    // RUNG-STABLE, like the consts the composite states: every rung of the ladder mints
    // the composite, so every rung carries the same sentinel row — a rung that forgot it
    // would gather past its own table while the ceiling rung's bake stayed clean.
    let mut rungs = ActiveCap::decode_ladder(256);
    assert!(
        rungs.len() >= 2,
        "the fixture's cap admits interior rungs — else this assert is vacuous"
    );
    rungs.push(ActiveCap::FULL);
    for cap in rungs {
        let (rolled, _) = walk_tied(true, Some(tied()), cap);
        let p = rolled
            .layout
            .placements
            .get(&table_tid)
            .unwrap_or_else(|| panic!("the table is placed at {cap:?}"));
        assert_eq!(
            (p.offset, p.size),
            (ext.offset, ext.size),
            "the sentinel row is rung-stable at {cap:?}"
        );
    }
}

/// A one-matmul graph — the lm-head ALONE. The recognition runs before the walk's loop
/// scan, so the smallest graph that reaches it is the result's own producer; `weight`
/// states the lm-head's staging form, and `m` its row count.
fn lm_head_only(m: u32, weight: GemmWeight, with_scale: bool) -> SubtileIR<NeoX> {
    let mut sources = vec![
        SourceShape { rows: m, cols: H }, // x
        SourceShape {
            rows: H,
            cols: VOCAB,
        }, // the lm-head table, IR spelling [K, N]
    ];
    let mut inputs = vec![InputRef::Ext(0), InputRef::Ext(1)];
    if with_scale {
        // The arity-3 fp8 form's per-channel scale — the third operand that makes a
        // staged lm-head recognizable by its ARITY, never by a shape.
        sources.push(SourceShape {
            rows: 1,
            cols: VOCAB,
        });
        inputs.push(InputRef::Ext(2));
    }
    let input = LoweringInput {
        sources,
        result: 0,
        exported: None,
        ops: vec![OpDesc {
            op: SubOp::MatmulTile { n: VOCAB, weight },
            m,
            inputs,
        }],
    };
    lower_region(&input, std::num::NonZeroU32::new(8192).unwrap())
}

fn walk_refuses(ir: &SubtileIR<NeoX>) -> String {
    let valid = ValidatedGraph::new(ir).expect("the lm-head-only graph validates");
    let tape = lower_dag_to_tape(&valid);
    let rolled_tape = reroll_subtile_tape(&tape, ir);
    let weight_ids: HashSet<u32> = [1u32].into_iter().collect();
    match lower_subtile_tape_to_superdsc(
        &rolled_tape,
        ir,
        &weight_ids,
        ActiveCap::FULL,
        false,
        RungFacts {
            one_request: true,
            chunk_pages: None,
            argmax: Some(tied()),
        },
    ) {
        Err(e) => e.0,
        // `RolledSuperDsc` is not `Debug`, so the Ok arm states its own refusal.
        Ok(_) => panic!("a tied graph the walk cannot recognize must refuse, not fall back"),
    }
}

/// A STAGED lm-head (fp8 W8A8 — the scale rides as the matmul's third operand) on a
/// tied one-request rung REFUSES LOUDLY, naming the staging fact: the tied gather
/// reads a Dense `[vocab, hidden]` table in place, and this first cut has no staging
/// fallback. This is the boundary granite-fp8 sits on (its lm-head is W8A8 when the
/// preset replaces the dense emission) — surfaced, never papered over.
#[test]
fn a_staged_lm_head_refuses_loudly() {
    let err = walk_refuses(&lm_head_only(1, GemmWeight::Fp8Dynamic, true));
    assert!(
        err.contains("lm-head matmul at arity 3"),
        "the refusal names the arity/staging fact: {err}"
    );
    assert!(
        err.contains("Dense [vocab, hidden] table"),
        "the refusal names what the tied gather reads: {err}"
    );
}

/// A multi-row result REFUSES the k=1 composite, naming the row count: the expansion
/// gates the tie token on `one_request`, so a multi-row result reaching the walk means
/// the gate was dropped, and the composite has no batched form to fall back to.
#[test]
fn a_multi_row_result_refuses_the_k1_composite() {
    let err = walk_refuses(&lm_head_only(3, GemmWeight::Dense, false));
    assert!(
        err.contains("3-row result"),
        "the refusal names the row count: {err}"
    );
}

/// ⭐ THE LADDER-SIDE CONTENT-KEY ASSERT (#603 PR3a) — every cap rung is its own
/// lowering of the same tape, and the composite must mint IDENTICALLY at every one:
/// same op name, same program, same bindings, and its two resident consts at the same
/// segment/bank/offset/size. That is the content-key law made testable — "rung-
/// independent by construction, asserted not conventions" (`emit::argmax`): the bake's
/// content key ("a hash of exactly what dxp will read") collapses the composite across
/// the whole ladder, so the group costs ONE dxp compile, not one per rung. The
/// emission-side half of the guard is the door's byte-identity test (a descriptor for
/// this op is a pure function of exactly the content compared here); this is the
/// ladder-side half.
///
/// The ladder is the macro's own construction — the interior decode rungs
/// `ActiveCap::decode_ladder` admits below the storage cap, with the FULL ceiling
/// appended (the rung any context ≤ cap falls back to) — and NON-VACUITY is pinned
/// alongside: the lowest interior rung's attention program is NOT the ceiling's (the
/// swept extent is baked into it), so the rungs are genuinely different lowerings and
/// the composite's identity is a real property of the composite, not of a fixture that
/// lowers the same bytes at every rung.
#[test]
fn every_cap_rung_mints_the_identical_composite() {
    // The ladder is the macro's own construction at the fixture's own resident-cache
    // cap — one page, `decode_ladder(256)`'s interior rungs {64, 128}, the rungs the
    // golden file names as production's bake set.
    let mut rungs = ActiveCap::decode_ladder(256);
    assert!(
        rungs.len() >= 2,
        "the fixture's cap admits interior rungs — else this assert is vacuous"
    );
    // The ceiling LAST: every interior rung compares against it, as the design doc
    // states ("each rung's rendered composite against the ceiling rung's").
    rungs.push(ActiveCap::FULL);

    let rolled_at: Vec<RolledSuperDsc> = rungs
        .iter()
        .map(|&cap| walk_tied(true, Some(tied()), cap).0)
        .collect();

    // The composite's whole content: the mint (op name + KTIR node) and the two
    // resident consts' placements. `TensorPlacement` carries no `PartialEq`, so the
    // placement compares as the tuple a descriptor bakes: role, segment, bank, offset,
    // size — offset included, because a moved const is a moved address.
    let composite_of = |rolled: &RolledSuperDsc| {
        let last = rolled
            .suffix
            .last()
            .expect("the suffix holds the composite at every rung");
        assert!(
            last.op_name.starts_with("argmax_o"),
            "the suffix's last op is the composite at every rung: {}",
            last.op_name
        );
        let placement = |tid: u32| {
            let p = rolled
                .layout
                .placements
                .get(&tid)
                .unwrap_or_else(|| panic!("tid {tid} is placed at every rung"));
            (p.role, p.segment, p.bank, p.offset, p.size)
        };
        (
            last.op_name.clone(),
            last.ktir
                .clone()
                .expect("the composite carries its program"),
            placement(argmax_ramp_tid()),
            placement(argmax_v_tid()),
        )
    };

    let ceiling = composite_of(rolled_at.last().expect("the ceiling rung lowered"));
    for (rung, rolled) in rungs.iter().zip(&rolled_at).take(rungs.len() - 1) {
        let here = composite_of(rolled);
        assert_eq!(here.0, ceiling.0, "the op name is rung-stable at {rung:?}");
        assert_eq!(
            here.1.func, ceiling.1.func,
            "the func is rung-independent at {rung:?} — no per-rung offset may enter it"
        );
        assert_eq!(here.1.program, ceiling.1.program, "the program at {rung:?}");
        assert_eq!(
            here.1.bindings, ceiling.1.bindings,
            "the bindings at {rung:?}"
        );
        assert_eq!(here.1.mask, ceiling.1.mask, "the mask binding at {rung:?}");
        assert_eq!(
            here.1.node_out_tid, ceiling.1.node_out_tid,
            "the node output binding at {rung:?}"
        );
        assert_eq!(
            here.2, ceiling.2,
            "the ramp's placement is rung-independent at {rung:?}"
        );
        assert_eq!(
            here.3, ceiling.3,
            "the V stick's placement is rung-independent at {rung:?}"
        );
    }

    // NON-VACUITY — the rungs ARE different lowerings of the same tape: some body
    // program at the lowest interior rung differs from the ceiling's (the attention's
    // swept extent enters its program — `lower_attn`'s `param_read_rows` law).
    let body_funcs = |rolled: &RolledSuperDsc| {
        rolled
            .bodies
            .iter()
            .flat_map(|b| b.ops.iter().filter_map(|e| e.ktir.as_ref().map(|k| k.func)))
            .collect::<Vec<_>>()
    };
    let low = body_funcs(&rolled_at[0]);
    let ceil = body_funcs(rolled_at.last().expect("the ceiling rung lowered"));
    assert_eq!(low.len(), ceil.len(), "the ladder lowers the same tape");
    assert!(
        low.iter().zip(&ceil).any(|(a, b)| a != b),
        "the lowest interior rung's attention program differs from the ceiling's — the \
         rungs are genuinely different lowerings, or this assert pins nothing"
    );
}
