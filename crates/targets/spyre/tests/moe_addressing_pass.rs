// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE ADDRESSING WELL-FORMEDNESS PASS'S OWN GATES — the general
//! compile-time lockdown's refusals, each through the REAL bake entry
//! (`lower_graph_to_superdsc`), so what is tested is the pass's WIRING under the
//! bake and not just its arithmetic.
//!
//! The pass is the graph half of the lockdown (the splice's `addressing` module):
//! the binding half is `bindings_for`'s M6 provenance walk and the door half is
//! `expert_matmul`'s `k <= e_count`. What only THIS pass catches:
//!
//! - **THE ID RANGE vs THE BANK BUDGET** — `E <= e_count`. The landed guards bound
//!   the id COUNT (`k <= E`, `k <= e_count`); at `E > e_count` an id in
//!   `[e_count, E)` composes a weight-gather address past the bank's end — a clean
//!   bake that faults the card's FETCH (the wall-10 class). Pinned here with the
//!   numbers in the refusal.
//! - **THE CLOSED CONSUMER TABLE** — a `SubOp` reading an index-minted tensor (a
//!   `RouteTopK`'s or `ExpertSort`'s output) at a position with no stated read law
//!   is an E0004-style refusal: a new index-consumer class needs a bound proof in
//!   the pass before it can bake. Pinned here with an elementwise read of the
//!   routing — a node nothing else in the pipeline would refuse.
//! - **THE BAKE-HEAD k <= E** — the pass states the mint's own law at the bake
//!   head, so a top-k wider than the expert count refuses BEFORE a program is
//!   minted (the binding-time twin fires later, at mint).
//! - **THE ARITY HARDENING** — a short `ExpertMatmul` is a named refusal instead
//!   of `bindings_for`'s unguarded `inputs[2]`/`inputs[3]` PANIC.
//!
//! And the honest boundary: the 26b gate/up geometry (the chunk-cut harness's own
//! numbers) lowers UNCHANGED under the pass — the tabled chain is exactly what the
//! front end's every MoE arm builds.

use scratchy_subtile::subtile_ir::{
    EwKind, ExpertBundle, ExpertProj, NeoX, NumExperts, SubOp, SubtileIR, SubtileId, SubtileNode,
    TensorId, TensorRegion, TensorShape, TopK,
};

/// The 26b gate/up door's own numbers (the chunk-cut harness's): one token row,
/// eight routed experts, a 128-expert bank of `[704, 2816]` slabs.
const M: u32 = 1;
const K: u32 = 8;
const E: u32 = 128;
const N_OUT: u32 = 704;
const IN: u32 = 2816;

/// The honest chain at the given bank row count — the wavefront's own wiring:
/// an honest router chain — `RouteArgsort` over the raw logits, `RouteTopK`
/// inverting the ranks into the `[m, k]` indices, a `RouteGatherScores`
/// closing the chain so the pair's binding walk binds (the argsort and the
/// gather both FOLD into the fused pair, so only the topk program mints) —
/// then an `ExpertSort` produces the pair rows (which ARE the routing tile
/// the matmul's builder binds), and the `ExpertMatmul` reads `[pairs, pairs,
/// w, s]`. Tensors: `t0` x `[m, in]`, `t1` the router logits `[m, e]`, `t2`
/// the fp8 bank `[bank_rows, in]`, `t3` the scales `[bank_rows, 1]`, `t4`
/// the sort's pair rows `[m, k·in]`, `t5` the matmul out `[m, k·n]`, `t6`
/// the top-k indices `[m, k]`, `t7` the argsort's rank vector `[m, e]`,
/// `t8` the gather's scores `[m, k]` (t6–t8 appended after the result so
/// every `_o{tid}` pin keeps its id).
fn chain_ir(bank_rows: u32, k: u32, e: u32) -> SubtileIR<NeoX> {
    let nz = |n: u32| std::num::NonZeroU32::new(n).unwrap();
    let tensors = vec![
        TensorShape { rows: M, cols: IN }, // 0: x (source)
        TensorShape { rows: M, cols: e },  // 1: the router logits (source)
        TensorShape {
            rows: bank_rows,
            cols: IN,
        }, // 2: the fp8 bank (source)
        TensorShape {
            rows: bank_rows,
            cols: 1,
        }, // 3: the scale bank (source)
        TensorShape {
            rows: M,
            cols: IN * k,
        }, // 4: the sort's pair rows
        TensorShape {
            rows: M,
            cols: N_OUT * k,
        }, // 5: the matmul's out
        TensorShape { rows: M, cols: k },  // 6: the top-k indices (the sort's routing)
        TensorShape { rows: M, cols: e },  // 7: the argsort's rank vector (folds)
        TensorShape { rows: M, cols: k },  // 8: the gather's scores (folds)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // The honest router chain's two folding legs: the argsort (its only
    // consumer is the topk) and the gather (its ids come from the topk over
    // the same logits). Both fold into the fused pair, so the pair's binding
    // walk binds and only the topk program mints.
    let argsort = SubtileNode {
        id: SubtileId::from_index(3),
        op: SubOp::RouteArgsort,
        inputs: vec![tr(1)],
        output: tr(7),
    };
    let topk = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RouteTopK {
            k: TopK::new(nz(k)),
        },
        inputs: vec![tr(7)],
        output: tr(6),
    };
    let gather = SubtileNode {
        id: SubtileId::from_index(4),
        op: SubOp::RouteGatherScores,
        inputs: vec![tr(1), tr(6)],
        output: tr(8),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(1),
        op: SubOp::ExpertSort {
            experts: NumExperts::new(nz(e)),
            k: TopK::new(nz(k)),
            bundle: ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(6)],
        output: tr(4),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(2),
        op: SubOp::ExpertMatmul {
            proj: ExpertProj::Gate,
            n: N_OUT,
            k: TopK::new(nz(k)),
            quant: scratchy_subtile::lower::ExpertQuant::declared(64, 8),
            bundle: ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(4), tr(4), tr(2), tr(3)],
        output: tr(5),
    };
    SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 4,
        nodes: vec![argsort, topk, gather, sort, matmul],
        op_output: Vec::new(),
    }
}

/// The bake's refusal for `ir`, as text.
fn refusal_of(ir: &SubtileIR<NeoX>) -> String {
    let weight_ids = std::collections::HashSet::new();
    match scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
        ir,
        &weight_ids,
        ktir_superdsc::ktir_node::ActiveCap::FULL,
        false,
    ) {
        Err(e) => format!("{e:?}"),
        Ok(_) => panic!("this graph must refuse under the addressing pass"),
    }
}

/// ⭐ THE HONEST BOUNDARY: the 26b gate/up geometry lowers UNCHANGED under the
/// pass — the graph-half sweep runs at the bake head and finds every read in the
/// consumer table and the composition in range. The chain is exactly what the
/// front end's MoE arms build, so this is the wiring proof, not a new shape.
#[test]
fn the_honest_chain_lowers_under_the_pass() {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            &chain_ir(E * N_OUT, K, E),
            &weight_ids,
            ktir_superdsc::ktir_node::ActiveCap::FULL,
            false,
        )
        .unwrap_or_else(|e| panic!("the honest chain must lower under the pass: {e:?}"));
    assert!(
        ops.iter()
            .any(|o| o.op_name.starts_with("emmm_") || o.op_name.starts_with("emwg_")),
        "the expert matmul's gathered body must still be emitted: {}",
        ops.iter()
            .map(|o| o.op_name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// ⛔ THE ID RANGE vs THE BANK BUDGET — the composition the two landed guards
/// never tied. A router over 8 experts whose bank packs only 3 (rows `3·n`
/// against `n=704` per expert) bakes cleanly under the landed guards (`k=2 <= E`
/// and `k=2 <= e_count`) while an id in `[3, 8)` composes a weight-gather address
/// past the bank's end. The refusal names the op, both counts, and the law.
#[test]
fn a_bank_narrower_than_the_router_refuses_with_the_numbers() {
    let msg = refusal_of(&chain_ir(3 * N_OUT, 2, 8));
    assert!(
        msg.contains("over 8 experts") && msg.contains("packs only 3"),
        "the refusal must state both counts: {msg}"
    );
    assert!(
        msg.contains("t2 is [2112, 2816]") && msg.contains("n=704"),
        "the refusal must name the bank and the per-expert width: {msg}"
    );
    assert!(
        msg.contains("id in [3, 8)") && msg.contains("router's expert count <= the bank's"),
        "the refusal must state the law: {msg}"
    );
}

/// The mint's own law at the bake head: a top-k of 9 over 8 experts would have to
/// include the argsort's `+inf` pad lanes, and a pad lane's id runs past the bank.
/// The pass fires BEFORE a program is minted; the binding-time twin (M6) fires at
/// mint with its own name.
#[test]
fn a_top_k_wider_than_the_experts_refuses_at_the_bake_head() {
    let msg = refusal_of(&chain_ir(8 * N_OUT, 9, 8));
    assert!(
        msg.contains("top-k of 9 over 8 experts") && msg.contains("pad lanes"),
        "the refusal must state the count and the pad-lane law: {msg}"
    );
}

/// ⛔ THE CLOSED CONSUMER TABLE — the E0004 discipline. An elementwise multiply
/// reading the sort's output is a node NOTHING ELSE in the pipeline would refuse
/// (a plain elementwise over pair rows is a legal lowering); the pass refuses it
/// because `Elementwise` has no stated read law for an index-minted tensor. A new
/// index-consumer class states its law in the pass first — that is the whole point
/// of the table.
#[test]
fn an_untabled_consumer_of_the_routing_refuses_unproven() {
    let ir = chain_ir(E * N_OUT, K, E);
    // Replace the matmul with an elementwise multiply over the pair rows — the
    // sort's output, an INDEX MINT, read by an untabled class.
    let mut nodes = ir.nodes.clone();
    nodes.pop();
    let mul = SubtileNode {
        id: SubtileId::from_index(2),
        op: SubOp::Elementwise(EwKind::Mul),
        inputs: vec![
            TensorRegion {
                tensor: TensorId::from_index(4),
                region: ir.tensors[4].whole(),
            },
            TensorRegion {
                tensor: TensorId::from_index(4),
                region: ir.tensors[4].whole(),
            },
        ],
        output: TensorRegion {
            tensor: TensorId::from_index(5),
            region: ir.tensors[5].whole(),
        },
    };
    nodes.push(mul);
    let ir = SubtileIR { nodes, ..ir };
    let msg = refusal_of(&ir);
    assert!(
        msg.contains("Elementwise(Mul) t5 reads t4 at input 0"),
        "the refusal must name the reader, the tensor and the position: {msg}"
    );
    assert!(
        msg.contains("no bound proof in the consumer table"),
        "the refusal must state the E0004 law: {msg}"
    );
    assert!(
        msg.contains("RouteGatherScores[1]/RouteExpertScale[1]"),
        "the refusal must list the tabled readers so the author knows where the proof goes: {msg}"
    );
}

/// ⛔ THE ARITY HARDENING — a three-input `ExpertMatmul` is a NAMED refusal here,
/// not the `bindings_for` `inputs[2]`/`inputs[3]` PANIC it would otherwise be (the
/// ABI table admits arity 3; nothing in this repository constructs it).
#[test]
fn a_short_arity_matmul_refuses_instead_of_panicking() {
    let ir = chain_ir(E * N_OUT, K, E);
    let mut nodes = ir.nodes.clone();
    let mut matmul = nodes.pop().unwrap();
    matmul.inputs.truncate(3);
    nodes.push(matmul);
    let ir = SubtileIR { nodes, ..ir };
    let msg = refusal_of(&ir);
    assert!(
        msg.contains("takes 3 input(s)") && msg.contains("exactly four"),
        "the refusal must state the arity and the bank form's law: {msg}"
    );
}
