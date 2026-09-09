// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `SyncSendRecvFusion.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e229_isIn` | 229 | 0 | 5 | `dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:54` |
//! | `e230_FuseSyncSendRecv` | 230 | 0 | 32 | `dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:70` |
//! | `e231_getNextEligableNode` | 231 | 0 | 14 | `dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:104` |
//! | `e388_isSubSet` | 388 | 1 | 5 | `dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:60` |
//! | `e389_runOnOperation` | 389 | 1 | 26 | `dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:120` |

// ⛔ THE PASS DRIVER IS `e389_runOnOperation`, STILL AN OPEN ANCHOR BELOW, so nothing outside this
// file calls any of it yet and CI's `-D warnings` would fail on the first ported leaf.
// ⭐ REMOVE THIS WITH `e389_runOnOperation`.
#![allow(dead_code)]

use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    DbgNamePrefix, new_dbg_name_from_list,
};
use crate::islands::sentient::dialects::sentient::SyncHalf;
use crate::islands::sentient::dialects::{Op, dataflow, sentient, symbol, uniform};

/// WHERE AN OP SITS IN ITS BLOCK — the `Operation *` the pass carries in `to_be_deleted` and hands to
/// [`fuse_sync_send_recv`] as the pair to fuse (`:123`, `:137-139`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBlock(pub usize);

/// Replaces: e229_isIn
///
/// Whether `a` is one of the peers in `b` — the `for (auto b : B) if (a == b)` of `:55-56`.
#[must_use]
pub fn is_in(a: SyncHalf, b: &[SyncHalf]) -> bool {
    b.contains(&a)
}

/// `isEqualSets` (`:66`) — mutual containment of the two syncs' peer lists.
///
/// ⛔ ITS `isSubSet` HALF IS `e388_isSubSet`, WHOSE ANCHOR IS STILL OPEN BELOW, so each direction is
/// written here as the one `all(is_in)` line `:60` is. e388's porter should collapse both onto it.
/// `isEqualSets` itself is no ledger unit at all — one body line, and it is private here for that
/// reason.
fn is_equal_sets(a: &[SyncHalf], b: &[SyncHalf]) -> bool {
    a.iter().all(|a| is_in(*a, b)) && b.iter().all(|b| is_in(*b, a))
}

/// THE FIVE OPS THE ELIGIBILITY WALK STEPS OVER — `dataflow.get_unit`, `sentient.scalar_constant`,
/// `symbol.create_symbol`, `uniform.def_immutable_mapping` and `uniform.query_map` (`:109-113`).
///
/// ⛔ THE SAME FIVE AS `LoopMerging.cpp:67-69`, and deliberately a second predicate: the two lists are
/// independently maintained in the reference. See [`super::loop_merging::get_next_eligible_op`].
fn skipped_between_syncs(op: &Op) -> bool {
    matches!(
        op,
        Op::Dataflow(dataflow::Op::GetUnit { .. })
            | Op::Sentient(sentient::Op::ScalarConstant { .. })
            | Op::Symbol(symbol::Op::CreateSymbol { .. })
            | Op::Uniform(uniform::Op::DefImmutableMapping { .. } | uniform::Op::QueryMap { .. })
    )
}

/// Replaces: e231_getNextEligableNode
///
/// The next op after `op` in its own block, walking past the bookkeeping ops of
/// [`skipped_between_syncs`].
#[must_use]
pub fn get_next_eligable_node(block: &[Op], op: InBlock) -> Option<InBlock> {
    block
        .iter()
        .enumerate()
        .skip(op.0 + 1)
        .find(|(_, next)| !skipped_between_syncs(next))
        .map(|(at, _)| InBlock(at))
}

/// WHAT ONE FUSION PRODUCED — the new `sentient.sync` and the two the driver must then erase, at
/// their positions AFTER the new one was inserted before them (`:138-139`, `:144`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fused {
    /// The `sentient.sync sendrecv` that now stands where the first of the pair stood.
    pub fused: InBlock,
    /// The send and the recv it replaces, in the order they were given.
    pub consumed: (InBlock, InBlock),
}

/// The four attributes of a `sentient.sync` the fusion reads.
type SyncParts<'a> = (sentient::SyncMode, &'a [SyncHalf], bool, Option<&'a str>);

/// `llvm::dyn_cast<sentient::SyncOp>(op)` — the op at `at`, if it is a sync.
fn sync_at(block: &[Op], at: InBlock) -> Option<SyncParts<'_>> {
    match block.get(at.0) {
        Some(Op::Sentient(sentient::Op::Sync {
            mode,
            peers,
            soft,
            dbg_name,
            ..
        })) => Some((*mode, peers.as_slice(), *soft, dbg_name.as_deref())),
        _ => None,
    }
}

/// Replaces: e230_FuseSyncSendRecv
///
/// Inserts a `sentient.sync sendrecv` before `op` when `op` and `op_next` are a send/recv pair
/// agreeing on `soft` and on their peer set, and names the two the driver then erases.
///
/// ⭐ THE FIRST SYNC'S PEERS AND `soft` ARE THE NEW OP'S, whichever way round the pair is (`:97`),
/// and `implicit_sync_memory_boundary` is `None` — the `getSI32IntegerAttr(-1)` of `:98`.
///
/// ⛔ TRAP: `None` FOR `op_next` IS THE REFERENCE'S NULL `Operation *`, which it hands straight to
/// `llvm::dyn_cast` (`:75`) — the end of a block is simply not fusible here.
pub fn fuse_sync_send_recv(
    block: &mut Vec<Op>,
    op: InBlock,
    op_next: Option<InBlock>,
) -> Option<Fused> {
    let op_next = op_next?;
    let (mode, peers, soft, dbg_name) = sync_at(block, op)?;
    let (mode_next, peers_next, soft_next, dbg_name_next) = sync_at(block, op_next)?;

    if !matches!(
        (mode, mode_next),
        (sentient::SyncMode::Send, sentient::SyncMode::Recv)
            | (sentient::SyncMode::Recv, sentient::SyncMode::Send)
    ) {
        return None;
    }
    if soft != soft_next {
        return None;
    }
    if !is_equal_sets(peers, peers_next) {
        return None;
    }

    let fused = Op::Sentient(sentient::Op::Sync {
        mode: sentient::SyncMode::SendRecv,
        peers: peers.to_vec(),
        soft,
        implicit_sync_memory_boundary: None,
        dbg_name: new_dbg_name_from_list(DbgNamePrefix::Ssrf, dbg_name, &[dbg_name_next]),
    });
    block.insert(op.0, fused);
    Some(Fused {
        fused: op,
        consumed: (InBlock(op.0 + 1), InBlock(op_next.0 + 1)),
    })
}

// crustify:todo: e388_isSubSet
//   authority : dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:60  (5 body lines, level 1)
//   original  : static bool isSubSet(mlir::ArrayAttr A, mlir::ArrayAttr B)
//   calls     : e229_isIn

// crustify:todo: e389_runOnOperation
//   authority : dcc/src/Transform/Sentient/SyncSendRecvFusion.cpp:120  (26 body lines, level 1)
//   original  : void SyncSendRecvFusionPass::runOnOperation()
//   calls     : e230_FuseSyncSendRecv, e231_getNextEligableNode

#[cfg(test)]
mod unit_tests {
    use super::{Fused, InBlock, fuse_sync_send_recv, get_next_eligable_node, is_in};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, sentient, symbol};

    /// One half of a rendezvous, from the arm that names a lowered destination.
    fn peer(dst: sentient::Consumer) -> sentient::SyncHalf {
        sentient::SyncHalf::of_lowered_destination(dst)
    }

    /// A `sentient.sync` with the four attributes the fusion reads.
    fn sync(
        mode: sentient::SyncMode,
        peers: Vec<sentient::SyncHalf>,
        soft: bool,
        dbg_name: &str,
    ) -> Op {
        Op::Sentient(sentient::Op::Sync {
            mode,
            peers,
            soft,
            implicit_sync_memory_boundary: Some(4),
            dbg_name: Some(dbg_name.to_owned()),
        })
    }

    /// 🎯 e229 — a peer set is searched by identity, and a peer that is not in it is not found.
    #[test]
    fn a_peer_is_in_a_set_only_when_the_set_names_it() {
        let units = [
            peer(sentient::Consumer::L3lu),
            peer(sentient::Consumer::L0su),
        ];
        assert!(is_in(peer(sentient::Consumer::L0su), &units));
        assert!(!is_in(peer(sentient::Consumer::L3su), &units));
    }

    /// 🎯 e231 — the walk steps over the five bookkeeping ops and stops at the next real op.
    #[test]
    fn the_eligible_node_is_the_next_op_past_the_bookkeeping_five() {
        let block = vec![
            sync(sentient::SyncMode::Send, Vec::new(), false, "s"),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(1),
                symbol_id: 0,
                max_value: None,
            }),
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: Val(2),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            sync(sentient::SyncMode::Recv, Vec::new(), false, "r"),
        ];
        assert_eq!(get_next_eligable_node(&block, InBlock(0)), Some(InBlock(3)));
        assert_eq!(get_next_eligable_node(&block, InBlock(3)), None);
    }

    /// 🎯 e230 — the vendor's own shape: a send and a recv over the same peers fuse into one
    /// `sendrecv` inserted before the pair, named from both.
    #[test]
    fn a_send_and_a_recv_over_the_same_peers_fuse_into_one_sendrecv() {
        let units = vec![
            peer(sentient::Consumer::L3lu),
            peer(sentient::Consumer::L0su),
        ];
        let mut block = vec![
            sync(sentient::SyncMode::Send, units.clone(), true, "a"),
            sync(
                sentient::SyncMode::Recv,
                vec![units[1], units[0]],
                true,
                "b",
            ),
        ];
        let fused = fuse_sync_send_recv(&mut block, InBlock(0), Some(InBlock(1)));
        assert_eq!(
            fused,
            Some(Fused {
                fused: InBlock(0),
                consumed: (InBlock(1), InBlock(2)),
            })
        );
        assert_eq!(
            block[0],
            Op::Sentient(sentient::Op::Sync {
                mode: sentient::SyncMode::SendRecv,
                peers: units,
                soft: true,
                implicit_sync_memory_boundary: None,
                dbg_name: Some("SSRF(a, b)".to_owned()),
            })
        );
    }

    /// 🎯 e230 — and each of the three disagreements refuses, leaving the block alone.
    #[test]
    fn a_disagreeing_pair_is_not_fused() {
        let units = vec![peer(sentient::Consumer::L3lu)];
        let other = vec![peer(sentient::Consumer::L0su)];
        for pair in [
            // Two sends: neither `(send, recv)` nor `(recv, send)`.
            (
                sync(sentient::SyncMode::Send, units.clone(), false, "a"),
                sync(sentient::SyncMode::Send, units.clone(), false, "b"),
            ),
            // Disagreeing `soft`.
            (
                sync(sentient::SyncMode::Send, units.clone(), false, "a"),
                sync(sentient::SyncMode::Recv, units.clone(), true, "b"),
            ),
            // Disagreeing peer sets.
            (
                sync(sentient::SyncMode::Send, units.clone(), false, "a"),
                sync(sentient::SyncMode::Recv, other.clone(), false, "b"),
            ),
        ] {
            let mut block = vec![pair.0, pair.1];
            let before = block.clone();
            assert_eq!(
                fuse_sync_send_recv(&mut block, InBlock(0), Some(InBlock(1))),
                None
            );
            assert_eq!(block, before);
        }
        // And the end of a block fuses with nothing.
        let mut block = vec![sync(sentient::SyncMode::Send, units, false, "a")];
        assert_eq!(fuse_sync_send_recv(&mut block, InBlock(0), None), None);
    }
}
