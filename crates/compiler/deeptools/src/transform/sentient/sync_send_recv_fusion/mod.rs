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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET: `e389_runOnOperation` is the pass ENTRY and is now
// ported, but nothing calls the entry either, so this is still what keeps the module warning-free
// under CI's `-D warnings`.
// ⭐ REMOVE THIS WHEN THE PASS IS WIRED — the precedent is
// [`super::nop_insertion_for_back_to_back_syncs`], whose entry is ported and equally uncalled.
#![allow(dead_code)]

use super::OptLevel;
use super::analyses::InstructionEstimator;
use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    DbgNamePrefix, new_dbg_name_from_list,
};
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::sentient::SyncHalf;
use crate::islands::sentient::dialects::{Op, dataflow, regions_mut, sentient, symbol, uniform};
use crate::model::Model;
use crate::workload::Workload;

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
/// ⭐ BOTH DIRECTIONS ARE [`is_subset`] (e388), which is the whole of the reference's own body; it is
/// no ledger unit itself, and private here for that reason.
fn is_equal_sets(a: &[SyncHalf], b: &[SyncHalf]) -> bool {
    is_subset(a, b) && is_subset(b, a)
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

/// Replaces: e388_isSubSet
///
/// Whether every peer of `a` is also a peer of `b` (`:60-64`).
///
/// ⭐ AN EMPTY `a` IS A SUBSET OF ANYTHING, which is the reference's loop not running — and it is
/// reached: a `sentient.sync` with no peers is what [`fuse_sync_send_recv`]'s own tests fuse.
#[must_use]
pub fn is_subset(a: &[SyncHalf], b: &[SyncHalf]) -> bool {
    a.iter().all(|&a| is_in(a, b))
}

/// `-dcc-sync-send-recv-fusion-disable`, `cl::init(false)` (`:33-36`) — a `dcc-opt` command-line flag
/// this crate has no command line for, so it is the constant it initialises to.
const DISABLE_THIS_PASS: bool = false;

/// ONE BLOCK'S FUSIONS AND THEIR DEFERRED ERASE — the body of the `unit_op.walk` (`:134-142`) plus its
/// share of the module-wide `to_be_deleted` (`:145`).
///
/// ⛔⛔ THE ERASE IS WHAT MUST STAY DEFERRED, not merely what the reference happens to defer: the ops
/// queued for deletion are still IN the block while the walk runs, and membership in the queue is the
/// walk's only guard against fusing one of them again (`:135-136`).
/// ⭐ ONE QUEUE PER BLOCK IS THE REFERENCE'S ONE QUEUE PER MODULE: [`fuse_sync_send_recv`] inserts and
/// consumes within a single block, so no recorded position ever names an op of another one.
/// ⭐ INNERMOST REGIONS FIRST is MLIR's default `WalkOrder::PostOrder` (`:134`); a fusion inside a
/// region cannot change its parent block, so descending before the block is the same order that walk
/// visits in.
fn fuse_syncs_in_block(block: &mut Vec<Op>) {
    for op in block.iter_mut() {
        for region in regions_mut(op) {
            fuse_syncs_in_block(region);
        }
    }
    let mut to_be_deleted: Vec<InBlock> = Vec::new();
    let mut at = 0;
    while at < block.len() {
        let op = InBlock(at);
        if !to_be_deleted.contains(&op) {
            let op_next = get_next_eligable_node(block, op);
            if let Some(fused) = fuse_sync_send_recv(block, op, op_next) {
                // The `sendrecv` went in AT `op`, so every position already queued from here on names
                // one op later than it did.
                for deleted in &mut to_be_deleted {
                    if deleted.0 >= fused.fused.0 {
                        deleted.0 += 1;
                    }
                }
                to_be_deleted.push(fused.consumed.0);
                to_be_deleted.push(fused.consumed.1);
                // ⭐ THE WALK RESUMES AT THE FUSED-OVER OP'S SUCCESSOR, which is where `getNextNode()`
                // stood before the insert — the reference's iterator holds it across the fusion.
                at = fused.consumed.0.0;
            }
        }
        at += 1;
    }
    to_be_deleted.sort_unstable();
    to_be_deleted.dedup();
    for deleted in to_be_deleted.iter().rev() {
        block.remove(deleted.0);
    }
}

/// Replaces: e389_runOnOperation
///
/// The pass entry: fuses the send/recv pairs of every program unit of one module, skipping a unit that
/// still fits its instruction buffer when the driver was given `-O0` (`:120-146`).
///
/// ⛔ `getChildAnalysis<InstructionEstimator>(unit_op)` IS THE `E` PARAMETER — the estimator is
/// per-unit there and asked about `unit_op` here, and `haveIbuffSpace` is out of campaign scope, so a
/// caller at [`OptLevel::O0`] gets its `todo!`.
/// ⭐ THE SKIP IS `continue`, NOT A RETURN: the walk's lambda returns per unit (`:130`).
pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload, E: InstructionEstimator>(
    program: &mut Program<A, M, W>,
    instruction_estimator: &mut E,
    opt_level: OptLevel,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    for unit in program.units.iter_mut() {
        if opt_level == OptLevel::O0 && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        fuse_syncs_in_block(&mut unit.body);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Fused, InBlock, InstructionEstimator, OptLevel, fuse_sync_send_recv,
        get_next_eligable_node, is_in, is_subset, run_on_operation,
    };
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::{Op, Val, sentient, symbol};
    use crate::islands::sentient::{Program, ProgramUnit, ProgramUnits};
    use crate::model::Model;
    use crate::transform::sentient::analyses::InstructionCount;
    use crate::units::DfirUnit;
    use crate::workload::Workload;

    /// A model, so the program is typed; nothing here reads it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// AN ESTIMATOR THAT ANSWERS THE ONE QUESTION e389 ASKS. ⛔ `InstructionEstimatorImpl` is out of
    /// campaign scope, so a test states its answer rather than computing one.
    struct StatedIbuffSpace(bool);

    impl InstructionEstimator for StatedIbuffSpace {
        fn recalculate(&mut self, _unit: &[Op]) {
            todo!("e389 never recalculates")
        }

        fn estimated_instruction_count_of_op(&mut self, _op: &Op) -> InstructionCount {
            todo!("e389 never counts instructions")
        }

        fn estimated_instruction_count_of_region(&mut self, _region: &[Op]) -> InstructionCount {
            todo!("e389 never counts instructions")
        }

        fn remaining_ibuff_space(&mut self, _unit: &[Op]) -> InstructionCount {
            todo!("e389 never asks for the remaining space, only whether there is any")
        }

        fn have_ibuff_space(&mut self, _unit: &[Op]) -> bool {
            self.0
        }
    }

    /// A two-unit program, both units running the same body.
    fn program_of(body: Vec<Op>) -> Program<Dd2, AnyModel, AnyRung> {
        let unit = |body: Vec<Op>| ProgramUnit {
            iter_arg: None,
            on: Units::one(DfirUnit::L0su, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit(body.clone()), vec![unit(body)]),
            bound: core::marker::PhantomData,
        }
    }

    /// `sentient.for` around `body`.
    fn for_op(body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(7),
            iv_reg: sentient::Reg::UNALLOCATED,
            bound: Val(8),
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

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

    /// 🎯 e388 — every peer of a subset is in the superset, one that is not refuses, and the empty
    /// list is a subset of anything.
    #[test]
    fn a_peer_list_is_a_subset_only_when_every_peer_of_it_is_in_the_other() {
        let units = [
            peer(sentient::Consumer::L3lu),
            peer(sentient::Consumer::L0su),
        ];
        assert!(is_subset(&[peer(sentient::Consumer::L0su)], &units));
        assert!(!is_subset(&[peer(sentient::Consumer::L3su)], &units));
        assert!(is_subset(&[], &units));
    }

    /// 🎯 e389 — the pass fuses every pair of every unit, at the top of a block and inside a
    /// `sentient.for`, and erases the two halves; at `-O0` a unit that still fits its instruction
    /// buffer is left untouched.
    #[test]
    fn the_pass_fuses_and_erases_in_every_block_and_skips_a_unit_with_ibuff_space_at_o0() {
        let pair = || {
            vec![
                sync(sentient::SyncMode::Send, Vec::new(), false, "a"),
                sync(sentient::SyncMode::Recv, Vec::new(), false, "b"),
            ]
        };
        let body = || {
            let mut body = pair();
            body.push(for_op(pair()));
            body
        };
        // ⭐ `implicit_sync_memory_boundary` IS `None` ON THE FUSED OP, which is why this is not the
        // `sync` helper's op.
        let fused = Op::Sentient(sentient::Op::Sync {
            mode: sentient::SyncMode::SendRecv,
            peers: Vec::new(),
            soft: false,
            implicit_sync_memory_boundary: None,
            dbg_name: Some("SSRF(a, b)".to_owned()),
        });

        let mut program = program_of(body());
        run_on_operation(&mut program, &mut StatedIbuffSpace(true), OptLevel::O1);
        for unit in program.units.iter() {
            assert_eq!(unit.body, vec![fused.clone(), for_op(vec![fused.clone()])]);
        }

        let mut at_o0 = program_of(body());
        run_on_operation(&mut at_o0, &mut StatedIbuffSpace(true), OptLevel::O0);
        for unit in at_o0.units.iter() {
            assert_eq!(unit.body, body());
        }
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
