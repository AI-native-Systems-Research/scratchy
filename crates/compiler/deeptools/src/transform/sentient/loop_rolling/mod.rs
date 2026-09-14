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

//! `LoopRolling.cpp` — 18 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5, 6, 7, 8]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e077_isIncrementField` | 077 | 0 | 12 | `dcc/src/Transform/Sentient/LoopRolling.cpp:67` |
//! | `e078_setEffectiveStart` | 078 | 0 | 4 | `dcc/src/Transform/Sentient/LoopRolling.cpp:134` |
//! | `e079_checkOpUsage` | 079 | 0 | 11 | `dcc/src/Transform/Sentient/LoopRolling.cpp:144` |
//! | `e080_getOperandKind` | 080 | 0 | 6 | `dcc/src/Transform/Sentient/LoopRolling.cpp:189` |
//! | `e081_deleteMatchedOps` | 081 | 0 | 4 | `dcc/src/Transform/Sentient/LoopRolling.cpp:233` |
//! | `e082_updateEndOpsOfMatchedOps` | 082 | 0 | 7 | `dcc/src/Transform/Sentient/LoopRolling.cpp:240` |
//! | `e083_computeValsIfDifferent` | 083 | 0 | 62 | `dcc/src/Transform/Sentient/LoopRolling.cpp:254` |
//! | `e084_updateBody` | 084 | 0 | 70 | `dcc/src/Transform/Sentient/LoopRolling.cpp:684` |
//! | `e317_checkDeltasOfOperands` | 317 | 1 | 79 | `dcc/src/Transform/Sentient/LoopRolling.cpp:505` |
//! | `e509_insertNextInstr` | 509 | 3 | 15 | `dcc/src/Transform/Sentient/LoopRolling.cpp:114` |
//! | `e510_insertOperandKind` | 510 | 3 | 3 | `dcc/src/Transform/Sentient/LoopRolling.cpp:183` |
//! | `e511_insertOperandDelta` | 511 | 3 | 3 | `dcc/src/Transform/Sentient/LoopRolling.cpp:186` |
//! | `e512_collectStartingValIterArgs` | 512 | 3 | 86 | `dcc/src/Transform/Sentient/LoopRolling.cpp:592` |
//! | `e565_collectDeltasOfOperands` | 565 | 4 | 109 | `dcc/src/Transform/Sentient/LoopRolling.cpp:392` |
//! | `e602_compareToNextWindow` | 602 | 5 | 67 | `dcc/src/Transform/Sentient/LoopRolling.cpp:321` |
//! | `e628_matchAndRoll` | 628 | 6 | 104 | `dcc/src/Transform/Sentient/LoopRolling.cpp:758` |
//! | `e642_rollInstrsInBlock` | 642 | 7 | 77 | `dcc/src/Transform/Sentient/LoopRolling.cpp:886` |
//! | `e652_runOnOperation` | 652 | 8 | 53 | `dcc/src/Transform/Sentient/LoopRolling.cpp:966` |


use std::collections::BTreeMap;

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, Val, dataflow, sentient, symbol, uniform,
};
use crate::model::Model;
use crate::transform::sentient::analyses::{InstructionEstimator, OutOfScopeInstructionEstimator};
use crate::transform::sentient::loop_tree::{LoopNodeId, LoopTree};
use crate::transform::sentient::skeleton;
use crate::transform::sentient::{ForRef, UnitFilter};
use crate::units::DfirUnit;
use crate::workload::Workload;

/// WHERE AN INSTRUCTION SITS IN THE BLOCK BEING ROLLED — `Block::iterator`.
///
/// ⛔⛔ AN IDENTITY, AND IT CANNOT BE A [`Val`] LIKE THE REST OF THIS MODULE TREE'S HANDLES.
/// `Window` keys `op_to_index_` by `Operation *` and its `start_`/`effective_start_`/`end_` are all
/// `Block::iterator`s (`dcc/src/Transform/Sentient/LoopRolling.cpp:93-96`), and the ops a window
/// holds need not bind anything at all — a `sentient.sync` is the very op that ENDS one in the L3
/// case — so [`crate::transform::sentient::ForRef`]'s "name the op by the value it binds" is not
/// available here.
///
/// ⭐ ONE ORDINAL, NOT A PATH LIKE THE BRIDGES' `OpId`: loop rolling is handed one block and never
/// descends. A nested reader is reported as absence instead; see [`Window::check_op_usage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct InstrPos(pub(crate) usize);

/// HOW MANY INSTRUCTIONS A WINDOW ROLLS — `Window::size_`, which `setEffectiveStart` rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowSize(pub(crate) usize);

/// WHICH WINDOW OF THE LIST `rollInstrsInBlock` BUILDS — `std::vector<Window *>::iterator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WindowIndex(pub(crate) usize);

/// WHICH OPERAND SLOT — the `i` of `getOperand(i)`, and the key every delta and kind is filed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OperandIdx(pub(crate) usize);

/// WHICH ITERATION ARGUMENT the rolled loop carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct IterArgIdx(pub(crate) usize);

/// WHICH RESULT of a matched op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ResultNum(pub(crate) usize);

/// THE CONSTANT STRIDE ONE OPERAND ADVANCES BY BETWEEN TWO WINDOWS.
///
/// ⚠️ `i64`, WHERE THE REFERENCE'S MAP IS `std::map<unsigned, int>` (`LoopRolling.cpp:171`) —
/// deliberate: `computeValsIfDifferent` computes the difference as `int64_t` from two `getValue()`s
/// that are themselves `int64_t`, so a stride wide enough to need it is truncated on the way into
/// that map and nothing downstream can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Delta(pub(crate) i64);

/// HOW MANY LOOPS THIS PASS HAS BUILT — `int &new_loop_count_`, which names them `"LR loop <n>"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NewLoopCount(pub(crate) u32);

/// WHICH OF THE TWO CASES IS BEING ROLLED — `bool is_L3_case_`.
///
/// ⭐ AN ENUM BECAUSE THE FILE HEADER ENUMERATES EXACTLY TWO (`LoopRolling.cpp:9-13`) and a third
/// would need a name, not a second boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RollingCase {
    /// L3 units, windows of instructions delimited by soft syncs at the OUTERMOST level.
    L3SoftSyncWindows,
    /// Generic units, windows of size 1 at the INNERMOST level.
    SingleInstrWindows,
}

/// HOW CORRESPONDING OPERANDS OF ONE MATCHED OP DIFFER ACROSS THE WINDOWS — `enum OperandKind`
/// (`LoopRolling.cpp:163`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OperandKind {
    /// `kNoDelta` — the same value in every window.
    NoDelta,
    /// `kFirstLastOpUsage` — the last rollable op's result read by the next window's first.
    FirstLastOpUsage,
    /// `kConstValueDelta` — a constant stride, recorded in `operand_deltas_`.
    ConstValueDelta,
}

/// WHAT `computeValsIfDifferent` ANSWERS — its three documented outcomes, as one value.
///
/// ⛔⛔ A `bool` PLUS AN `int64_t &` CANNOT SAY IT. The reference documents three
/// (`LoopRolling.cpp:249-253`): not computable, computable and different, computable and the same
/// with *"delta untouched"*. Both callers pre-set `delta = 0`, so "untouched" and "zero" are the
/// same thing to them — but they are not the same fact, and the caller that eventually cares should
/// not have to reconstruct it from an out-parameter it cannot see the history of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OperandDifference {
    /// `return false` — one or both values are not a constant or a uniform mapping.
    Incomputable,
    /// `return true`, delta untouched.
    Same,
    /// `return true`, delta written.
    Delta(Delta),
}

/// WHICH SIDE OF THE COMPARISON A WINDOW IS ON — the reference's `bool is_start_window`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowRole {
    /// `is_start_window == true` — the matches are CREATED here and the deltas collected.
    Start,
    /// `is_start_window == false` — the deltas are CHECKED against the ones the start window filed.
    Follower,
}

/// ONE WINDOW OF INSTRUCTIONS — the reference's `Window` (`LoopRolling.cpp:86`).
///
/// ⚠️ `dcc::CommonPassOptions &opts_` IS DROPPED, AND THAT IS A MEASUREMENT: the only two mentions of
/// `opts_` in the class are its declaration (`:87`) and the constructor's initialiser (`:107`).
/// Nothing reads it.
#[derive(Debug)]
pub(crate) struct Window {
    /// `is_L3_case_`.
    pub(crate) case: RollingCase,
    /// `size_`.
    pub(crate) size: WindowSize,
    /// `start_`.
    pub(crate) start: InstrPos,
    /// `effective_start_` — where the FIRST window to be rolled begins; `start_` for all others.
    pub(crate) effective_start: InstrPos,
    /// `end_` — a soft sync in the L3 case.
    pub(crate) end: InstrPos,
    /// `op_to_index_` — ⛔ COMPUTED FROM `start_`, NOT `effective_start_`, which the reference says
    /// out loud (`:97-98`) and [`Self::set_effective_start`] therefore leaves alone.
    pub(crate) op_to_index: BTreeMap<InstrPos, usize>,
    /// `first_rollable_op_`.
    pub(crate) first_rollable_op: Option<InstrPos>,
    /// `last_rollable_op_`.
    pub(crate) last_rollable_op: Option<InstrPos>,
}

impl Window {
    /// A WINDOW OPENING AT `start`, all three of its iterators there — the reference's constructor
    /// (`LoopRolling.cpp:105-110`).
    ///
    /// ⛔ NOT AN ANCHORED UNIT: the campaign excludes trivial constructors, and no anchor in this
    /// file can be filled without one.
    pub(crate) fn opening(case: RollingCase, start: InstrPos) -> Window {
        Window {
            case,
            size: WindowSize(0),
            start,
            effective_start: start,
            end: start,
            op_to_index: BTreeMap::new(),
            first_rollable_op: None,
            last_rollable_op: None,
        }
    }

    /// `isOpInThisWindow` — an excluded one-line accessor, and the guard both rewrites below read.
    pub(crate) fn is_op_in_this_window(&self, op: InstrPos) -> bool {
        self.op_to_index.contains_key(&op)
    }

    /// Replaces: e509_insertNextInstr
    ///
    /// Extends the window to `instr`, indexing it and remembering it as the first/last rollable op
    /// when the case admits it.
    ///
    /// ⭐ THE L3 CASE ADMITS ONLY THE THREE TRANSFERS, the generic case anything that is not a
    /// constant or a uniform map lookup — and the reference's note excludes the two loads that
    /// *"should never provide rolling opportunities"* (`LoopRolling.cpp:118-119`).
    pub(crate) fn insert_next_instr(&mut self, instr: InstrPos, block: &[Op]) {
        // `std::map::insert` keeps the first index for a repeated position while `size_` still grows.
        self.op_to_index.entry(instr).or_insert(self.size.0);
        self.size = WindowSize(self.size.0 + 1);
        self.end = instr;
        let op = &block[instr.0];
        let generic_case = self.case == RollingCase::SingleInstrWindows
            && !matches!(
                op,
                Op::Sentient(sentient::Op::ScalarConstant { .. })
                    | Op::Uniform(uniform::Op::QueryMap { .. })
                    | Op::Uniform(uniform::Op::DefImmutableMapping { .. })
            );
        let transfer = matches!(
            op,
            Op::Sentient(
                sentient::Op::LoadAndSend { .. }
                    | sentient::Op::ReceiveAndStore { .. }
                    | sentient::Op::LoadAndStore { .. }
            )
        );
        if generic_case || transfer {
            if self.first_rollable_op.is_none() {
                self.first_rollable_op = Some(instr);
            }
            self.last_rollable_op = Some(instr);
        }
    }

    /// Replaces: e078_setEffectiveStart
    ///
    /// Points the window at where rolling actually begins, with the matching size — the pair
    /// `compareToNextWindow` writes once it knows how many leading instructions of the start window
    /// will not be rolled.
    ///
    /// ⛔ `op_to_index` IS DELIBERATELY NOT REBUILT: the map is computed from `start_`, so
    /// [`Self::is_op_in_this_window`] keeps admitting the skipped leading ops — which is what makes
    /// `checkOpUsage` tolerate a rolled op reading one of them.
    pub(crate) fn set_effective_start(&mut self, effective_start: InstrPos, size: WindowSize) {
        self.effective_start = effective_start;
        self.size = size;
    }

    /// Replaces: e079_checkOpUsage
    ///
    /// Refuses `at` when any result of it is read outside this window, excusing only the last
    /// rollable op's use in `next_window`'s first rollable op.
    ///
    /// ⭐ A NESTED READER IS OUTSIDE BY CONSTRUCTION, so it needs no identity: `op_to_index_` only
    /// ever holds top-level ops of the block, and neither it nor `getFirstRollableOp()` can name an
    /// op inside another op's region. [`users_of`] reports one as `None`.
    pub(crate) fn check_op_usage(&self, at: InstrPos, block: &[Op], next_window: &Window) -> bool {
        for result in dialects::results(&block[at.0]) {
            for user in users_of(result, block) {
                let carried_into_next = Some(at) == self.last_rollable_op
                    && user.is_some()
                    && user == next_window.first_rollable_op;
                if !carried_into_next && !user.is_some_and(|user| self.is_op_in_this_window(user)) {
                    return false;
                }
            }
        }
        true
    }
}

/// ONE SET OF MATCHED OPS ACROSS THE WINDOWS TO BE ROLLED — the reference's `MatchedOp` (`:159`).
#[derive(Debug)]
pub(crate) struct MatchedOp {
    /// `start_op_` — this op's instance in the start window.
    pub(crate) start_op: InstrPos,
    /// `end_op_` — its instance in the end window, written by
    /// [`LoopRollingManager::update_end_ops_of_matched_ops`].
    pub(crate) end_op: InstrPos,
    /// `operand_kinds_`.
    pub(crate) operand_kinds: BTreeMap<OperandIdx, OperandKind>,
    /// `operand_deltas_` — NONZERO deltas only (`collectDeltasOfOperands` files a zero as
    /// `kNoDelta` instead).
    pub(crate) operand_deltas: BTreeMap<OperandIdx, Delta>,
    /// `iter_arg_to_res_num_and_operand_` — ⭐ `Option<OperandIdx>` IS THE REFERENCE'S `-1`, which
    /// means this op is not the last rollable op used in the next window's first rollable op.
    pub(crate) iter_arg_to_res_num_and_operand:
        BTreeMap<IterArgIdx, (ResultNum, Option<OperandIdx>)>,
}

impl MatchedOp {
    /// ONE MATCH, both instances starting as the same op — the reference's constructor (`:179`).
    ///
    /// ⛔ NOT AN ANCHORED UNIT (an excluded trivial constructor).
    pub(crate) fn of(start_op: InstrPos) -> MatchedOp {
        MatchedOp {
            start_op,
            end_op: start_op,
            operand_kinds: BTreeMap::new(),
            operand_deltas: BTreeMap::new(),
            iter_arg_to_res_num_and_operand: BTreeMap::new(),
        }
    }

    /// Replaces: e510_insertOperandKind
    ///
    /// Files how operand `i` differs across the windows — ⛔ THE FIRST CLASSIFICATION WINS, because
    /// `std::map::insert` does not overwrite an existing key.
    pub(crate) fn insert_operand_kind(&mut self, i: OperandIdx, kind: OperandKind) {
        self.operand_kinds.entry(i).or_insert(kind);
    }

    /// Replaces: e511_insertOperandDelta
    ///
    /// Files the nonzero stride operand `i` advances by — first write wins, as for
    /// [`Self::insert_operand_kind`].
    pub(crate) fn insert_operand_delta(&mut self, i: OperandIdx, delta: Delta) {
        self.operand_deltas.entry(i).or_insert(delta);
    }

    /// Replaces: e080_getOperandKind
    ///
    /// The kind recorded for operand `i`, which the reference requires to be present.
    ///
    /// ⛔ THE `DT_CHECK_MSG` BECOMES A LOUD `todo!` AND NOT A DEFAULT: an absent kind means
    /// `collectDeltasOfOperands` never classified this slot, and answering `kNoDelta` for it would
    /// roll a loop that leaves the operand pointing at the start window's value.
    pub(crate) fn operand_kind(&self, i: OperandIdx) -> OperandKind {
        match self.operand_kinds.get(&i) {
            Some(kind) => *kind,
            None => todo!(
                "MatchedOp::getOperandKind: \"Operand must have an OperandKind.\" \
                 (LoopRolling.cpp:191-192), asked for operand {i:?}"
            ),
        }
    }
}

/// EVERY READER OF A VALUE IN THIS BLOCK — `Value::getUses()`, answered by a walk.
///
/// ⛔ NOT AN ANCHORED UNIT — the *mechanism* for reaching operands, which the campaign brief names as
/// the one thing a port may supply for itself. MLIR keeps a use list on the value; this island's ops
/// are a tree, so the list is recomputed.
///
/// ⭐ `None` IS A READER INSIDE SOME OP'S REGION, reported once per top-level op rather than
/// identified: no window and no `first_rollable_op_` can name a nested op, so the only fact any
/// caller here needs is that such a reader exists.
///
/// ⭐ THE WALK IS COMPLETE FOR THIS ISLAND: a region is one block, and SSA dominance forbids a use
/// outside the region its definition sits in, so a value defined at the top level of `block` has all
/// its readers in `block` or in a region of one of `block`'s ops.
fn users_of(val: Val, block: &[Op]) -> Vec<Option<InstrPos>> {
    let mut users: Vec<Option<InstrPos>> = Vec::new();
    for (at, op) in block.iter().enumerate() {
        if dialects::operands(op).contains(&val) {
            users.push(Some(InstrPos(at)));
        }
        if reads_nested(val, op) {
            users.push(None);
        }
    }
    users
}

/// Whether anything inside one op's regions reads `val`.
fn reads_nested(val: Val, op: &Op) -> bool {
    dialects::regions(op).iter().any(|region| {
        region
            .iter()
            .any(|inner| dialects::operands(inner).contains(&val) || reads_nested(val, inner))
    })
}

/// WHETHER TWO OPS ARE EQUIVALENT — `dcc::OperationEquivalence::operationsAreEquivalent`.
///
/// ⛔ NOT AN ANCHORED UNIT — it is `dcc/src/Analysis/OperationEquivalence.cpp`, not this campaign's
/// file list, and `e083_computeValsIfDifferent` is unportable without it. The sibling
/// [`crate::bridges::dataflow_ir_to_sentient::vc_vector_chain_helper::ops_are_equivalent`] is the
/// same walk one rung down; this one speaks the mixed rung.
///
/// ⭐ NO OPTIONS, BECAUSE LOOP ROLLING'S `oe_` HAS NONE THAT VARY: it is built with a null preference
/// functor, `all_block_args_are_equiv = true`, `do_recursive_compare = true` and
/// `use_equiv_classes = true` (`LoopRolling.cpp:987-989`) — so every block argument matches every
/// other, operand definitions are compared recursively, and the equivalence classes are a cache.
///
/// ⭐ BLANKING THE BLOCK ARGUMENTS IS CORRECT, NOT A SHORTCUT: MLIR keeps a region's arguments on the
/// BLOCK, so `operationsAreEquivalent` never compares them at all. This island stores them in the
/// variant, so leaving them in the skeleton would make every pair of `sentient.for`s inequivalent.
fn ops_are_equivalent(a: &Op, b: &Op, defs: Definitions<'_>) -> bool {
    if skeleton(a) != skeleton(b) {
        return false;
    }
    for (read_a, read_b) in dialects::operands(a)
        .into_iter()
        .zip(dialects::operands(b))
    {
        if read_a == read_b {
            continue;
        }
        match (defs.of(read_a), defs.of(read_b)) {
            (Some(def_a), Some(def_b)) => {
                if !core::ptr::eq(def_a, def_b) && !ops_are_equivalent(def_a, def_b, defs) {
                    return false;
                }
            }
            // `all_block_args_are_equiv = true`: nothing defines either, and that is a match.
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => return false,
        }
    }
    for (region_a, region_b) in dialects::regions(a).into_iter().zip(dialects::regions(b)) {
        if region_a.len() != region_b.len() {
            return false;
        }
        for (inner_a, inner_b) in region_a.iter().zip(region_b.iter()) {
            if !ops_are_equivalent(inner_a, inner_b, defs) {
                return false;
            }
        }
    }
    true
}

/// `oe_.operationsAreEquivalent(.., /*allow_different_operands*/ true)` — the same walk with the
/// operand comparison switched off (`OperationEquivalence.cpp:232-315`).
///
/// ⛔ NOT AN ANCHORED UNIT, for the reason [`ops_are_equivalent`] is not.
/// ⛔ THE FLAG IS TOP-LEVEL ONLY: `regionsAreEquivalent` is re-entered with `allow_different_operands`
/// HARDCODED to `false` (`:317-325`), so a nested pair still has to read the same values.
fn ops_are_equivalent_ignoring_operands(a: &Op, b: &Op, defs: Definitions<'_>) -> bool {
    if skeleton(a) != skeleton(b) {
        return false;
    }
    for (region_a, region_b) in dialects::regions(a).into_iter().zip(dialects::regions(b)) {
        if region_a.len() != region_b.len() {
            return false;
        }
        for (inner_a, inner_b) in region_a.iter().zip(region_b.iter()) {
            if !ops_are_equivalent(inner_a, inner_b, defs) {
                return false;
            }
        }
    }
    true
}

/// THE `uniform.def_immutable_mapping` A `uniform.query_map`'S `$map` NAMES, as its `(key, value)`
/// pairs.
///
/// ⛔ NOT AN ANCHORED UNIT — `dcc/src/Dialect/Uniform/Utils.cpp`, outside this campaign's file list.
/// The reference `dyn_cast`s and then dereferences without a check, so anything else is a crash there
/// and a named `todo!` here.
fn immutable_mapping_pairs(map: Val, defs: Definitions<'_>) -> Vec<(Val, Val)> {
    match defs.of(map) {
        Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs.clone(),
        other => todo!(
            "a uniform.query_map's $map is not a uniform.def_immutable_mapping \
             (Dialect/Uniform/Utils.cpp:403-404): {other:?}"
        ),
    }
}

/// `getConstantOrSymbolTargetValues` (`Dialect/Uniform/Utils.cpp:400`) — every target value of a
/// query map's mapping, as a number.
///
/// ⛔ NOT AN ANCHORED UNIT (see [`immutable_mapping_pairs`]). ⭐ AN EMPTY ANSWER MEANS "not all of
/// them are constants or symbols", which is the reference's own signal and what e083 tests for.
fn constant_or_symbol_target_values(map: Val, defs: Definitions<'_>) -> Vec<i64> {
    let pairs = immutable_mapping_pairs(map, defs);
    if pairs.is_empty() {
        todo!(
            "getConstantOrSymbolTargetValues: DT_CHECK(!immutable_map.getValues().empty()) \
             (Dialect/Uniform/Utils.cpp:405)"
        );
    }
    let mut values: Vec<i64> = Vec::new();
    for (_key, value) in pairs {
        match defs.of(value) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => values.push(*value),
            Some(Op::Symbol(symbol::Op::CreateSymbol { symbol_id, .. })) => values.push(*symbol_id),
            _ => return Vec::new(),
        }
    }
    values
}

/// `getDeltasBetweenValuesOfUniformMappings` (`Dialect/Uniform/Utils.cpp:204`) — the per-unit
/// difference between two mappings' target constants.
///
/// ⛔ NOT AN ANCHORED UNIT (see [`immutable_mapping_pairs`]).
///
/// ⭐ ONE KEY LIST, SUPPLIED, AND THAT IS WHAT THE REFERENCE EFFECTIVELY USES: it derives a list per
/// query map, refuses when the two differ in length, and then looks up BOTH mappings with A's list
/// (`values_b = target_map_b.getValuesFromKeys(key_vals_a)`). Deriving it is
/// `getListOfKeyOpsFromUniformMapping` — the enclosing region's units — which is this campaign's own
/// `e392_getQueryKeyAndUnitsFromParentRegion` (`Transform/Sentient/Utils.cpp:40`), still unfilled.
fn deltas_between_values_of_uniform_mappings(
    map_a: Val,
    map_b: Val,
    key_vals: &[Val],
    defs: Definitions<'_>,
) -> Option<Vec<Delta>> {
    let pairs_a = immutable_mapping_pairs(map_a, defs);
    let pairs_b = immutable_mapping_pairs(map_b, defs);
    let mut deltas: Vec<Delta> = Vec::new();
    for key in key_vals {
        let value_a = value_for_key(&pairs_a, *key)?;
        let value_b = value_for_key(&pairs_b, *key)?;
        let a = scalar_constant_value(defs.of(value_a)?)?;
        let b = scalar_constant_value(defs.of(value_b)?)?;
        deltas.push(Delta(b - a));
    }
    Some(deltas)
}

/// `DefImmutableMappingOp::getValuesFromKeys` for one key — `None` when the mapping has no entry.
fn value_for_key(pairs: &[(Val, Val)], key: Val) -> Option<Val> {
    pairs.iter().find(|(k, _)| *k == key).map(|(_, value)| *value)
}

/// `dyn_cast<sentient::ConstantOp>(..).getValue()`.
fn scalar_constant_value(op: &Op) -> Option<i64> {
    match op {
        Op::Sentient(sentient::Op::ScalarConstant { value, .. }) => Some(*value),
        _ => None,
    }
}

/// `MaxIterArgsCreated` — `-loop-rolling-max-iter-args-created`, whose default is the only value
/// this crate has (`LoopRolling.cpp:59-63`).
const MAX_ITER_ARGS_CREATED: usize = 4;

/// `zero.getDefiningOp()->erase()`, in the block the constant builder is anchored at.
///
/// ⛔ NOT AN ANCHORED UNIT — the *mechanism* for reaching a definition.
fn erase_definition(val: Val, consts: &mut Vec<Op>) {
    if let Some(at) = consts
        .iter()
        .position(|op| dialects::results(op).contains(&val))
    {
        consts.remove(at);
    }
}

/// `sentient::ConstantOp::create(const_builder_, .., getIndexType(), value)`.
fn scalar_constant(result: Val, value: i64) -> Op {
    Op::Sentient(sentient::Op::ScalarConstant {
        value,
        result,
        // `ConstantOp`'s own default, which the reference never overrides here.
        reg_locale: sentient::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    })
}

/// `sentient::AddOp::create(builder, .., getIndexType(), iter_arg, const_op)`.
fn scalar_add(result: Val, lhs: Val, rhs: Val) -> Op {
    Op::Sentient(sentient::Op::ScalarAdd {
        lhs,
        rhs,
        result,
        // `AddOp::create` sets no `regLocale`, so the `.td`'s absent register stands.
        reg: None,
        ty: ScalarTy::Index,
        element_size: None,
    })
}

/// `replaceUsesWithIf(.., owner != yield_op && !cur_window->isOpInThisWindow(owner))`.
///
/// ⛔ NOT AN ANCHORED UNIT — the predicate half of `updateBody`'s rewrite.
///
/// ⭐ THE `yield_op` HALF NEEDS NO TEST: the yield lives in [`RolledLoop::body`], which this walk
/// never enters. The reference has to name it because it built the yield into `bb_`'s new loop
/// before the windows were re-examined, and its own comment says so (`LoopRolling.cpp:744-746`).
fn replace_uses_outside_window(block: &mut [Op], window: &Window, of: Val, with: Val) {
    for (at, op) in block.iter_mut().enumerate() {
        if window.is_op_in_this_window(InstrPos(at)) {
            continue;
        }
        dialects::replace_all_uses_with(core::slice::from_mut(op), of, with);
    }
}

/// Replaces: e077_isIncrementField
///
/// Whether operand `i` of `op` is a transfer's increment field — the one operand loop rolling may
/// rewrite to an `iter_arg` even when its delta is zero.
///
/// ⛔ `LoadAndExtractScalar` AND `LoadComputeAndSend` ARE EXCLUDED BY THE REFERENCE'S OWN NOTE
/// (`LoopRolling.cpp:68-69`): they *"should never provide rolling opportunities"*.
/// ⚠️ TRAP: the positions are `SentientOps.td:507-509`, `:550-552`, `:722-729`, and this island's
/// list matches them only up to slot 2 — it models `$consumer`/`$producer` as a wire end, so
/// `load_and_send` and `receive_and_store` are one slot short from position 3 on.
/// ⚠️ `OpBuilder &const_builder` IS UNUSED IN THE REFERENCE'S BODY and is dropped.
pub(crate) fn is_increment_field(op: &Op, i: OperandIdx) -> bool {
    match op {
        Op::Sentient(sentient::Op::ReceiveAndStore { .. } | sentient::Op::LoadAndSend { .. }) => {
            i == OperandIdx(2)
        }
        Op::Sentient(sentient::Op::LoadAndStore { .. }) => {
            i == OperandIdx(4) || i == OperandIdx(7)
        }
        _ => false,
    }
}

/// `isImmutableAddressLXAddress(op, i, const_builder)` — `Analyses/Utils.cpp:307-326`: is operand `i`
/// an `$immutable_addr` whose base address is on the LX?
///
/// ⛔ NOT AN ANCHORED UNIT. It lives in `Transform/Sentient/Analyses/`, outside the campaign's 656
/// definitions, but it is small and pure — the precedent is `loop_merging`'s
/// `is_sum_less_than_lccr_max_value`. ⭐ A `load_and_store` HAS TWO, and each asks about ITS OWN END.
fn is_immutable_address_lx_address(op: &Op, i: OperandIdx, defs: Definitions<'_>) -> bool {
    match op {
        // Operand 1 of all four, whose base address is the unit the op itself runs on.
        Op::Sentient(
            sentient::Op::ReceiveAndStore { .. }
            | sentient::Op::LoadAndSend { .. }
            | sentient::Op::LoadAndExtractScalar { .. }
            | sentient::Op::LoadComputeAndSend { .. },
        ) => i == OperandIdx(1),
        Op::Sentient(sentient::Op::LoadAndStore { src, dst, .. }) => match i {
            OperandIdx(3) => is_lx_unit(*src, defs),
            OperandIdx(6) => is_lx_unit(*dst, defs),
            _ => false,
        },
        _ => false,
    }
}

/// `isLXUnit(src)` — `Analyses/Utils.cpp:328-343`, through
/// `getUnitTypeFromUniformMappingAsString` (`Dialect/Uniform/Utils.cpp:258-283`).
///
/// ⛔ THE MAPPING ARM READS `getValues()[0]` AND ASKS NO AGREEMENT: it is the mapping's FIRST target,
/// not the targets the key selects, which is why this is not `address_pinning_and_toggle`'s
/// `unit_type_of`. The reference's `GetLocalUnitOp` arm has no island variant to match
/// (`islands/dataflow_ir/dialects/dataflow.rs:78` — the lowering emits `get_unit`).
fn is_lx_unit(val: Val, defs: Definitions<'_>) -> bool {
    match defs.of(val) {
        Some(Op::Dataflow(dataflow::Op::GetUnit { unit, .. })) => *unit == DfirUnit::Lx,
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(*map)
            else {
                return false;
            };
            matches!(
                pairs.first().and_then(|(_, value)| defs.of(*value)),
                Some(Op::Dataflow(dataflow::Op::GetUnit { unit, .. })) if *unit == DfirUnit::Lx
            )
        }
        _ => false,
    }
}

/// WHICH RESULT OF WHICH OP DEFINES A VALUE — `getDefiningOp()` paired with
/// `cast<OpResult>(..).getResultNumber()`.
///
/// ⛔ NOT AN ANCHORED UNIT — the *mechanism* for reaching operands. [`Definitions`] answers with the
/// `&Op`, and every handle a [`Window`] holds is an [`InstrPos`], so the walk is redone here to
/// speak positions. `None` is a block argument.
fn defining_result(val: Val, block: &[Op]) -> Option<(InstrPos, ResultNum)> {
    block.iter().enumerate().find_map(|(at, op)| {
        dialects::results(op)
            .iter()
            .position(|result| *result == val)
            .map(|num| (InstrPos(at), ResultNum(num)))
    })
}

/// THE `sentient.for` BEING BUILT — what `matchAndRoll` creates and [`LoopRollingManager::update_body`]
/// fills.
///
/// ⛔⛔ NOT YET AN OP OF THE BLOCK, AND THAT IS DELIBERATE. The reference inserts the `ForOp` into
/// `bb_` and then moves ops out of `bb_` INTO it (`:822-856`), which here would be one `&mut` into a
/// vector reaching another element of the same vector. Building the loop beside the block and
/// inserting it once, after `cur_window`'s end, is the same IR.
#[derive(Debug)]
pub(crate) struct RolledLoop {
    /// The loop's `iter_args`: `getRegionIterArgs()[i]` is `carried[i].arg` and `getResult(i)` is
    /// `carried[i].result`.
    pub(crate) carried: Vec<sentient::Carried>,
    /// The body.
    pub(crate) body: Vec<Op>,
    /// Where the `sentient.yield` sits in [`Self::body`] — `matchAndRoll` creates it FIRST
    /// (`:837-838`) and moves it to the end only after `updateBody` has run (`:844`), which is what
    /// puts the created adds after the moved ops.
    pub(crate) yield_at: usize,
}

/// ONE ATTEMPT TO ROLL FROM A GIVEN START WINDOW — the reference's `LoopRollingManager` (`:201`).
///
/// ⛔⛔ IT HOLDS NO BORROW INTO THE IR, unlike the reference's `Block &bb_` and its two
/// `OpBuilder &`s. These passes rewrite SentientIR in place, and a field holding `&mut Vec<Op>`
/// cannot coexist with the `getDefiningOp` lookups the same methods need over the same block — the
/// convention [`crate::transform::sentient::ForRef`] already states for this module tree. The block,
/// the constant builder's block, the definitions and the value minter are parameters instead.
///
/// ⚠️ `oe_` IS GONE WITH THEM: it is constructed with a null preference functor,
/// `all_block_args_are_equiv = true` and `do_recursive_compare = true` (`LoopRolling.cpp:987-989`),
/// so once the equivalence classes — a cache — are dropped, what is left is the pure
/// [`ops_are_equivalent`].
#[derive(Debug)]
pub(crate) struct LoopRollingManager {
    /// `is_L3_case_`.
    pub(crate) case: RollingCase,
    /// `new_loop_count_`, which the reference shares by reference across managers.
    pub(crate) new_loop_count: NewLoopCount,
    /// `start_` — the window rolling is attempted from.
    pub(crate) start: WindowIndex,
    /// `window_list_end_`.
    pub(crate) window_list_end: WindowIndex,
    /// `operand_to_result_num_` — every usage of the last rollable op in the next window's first.
    pub(crate) operand_to_result_num: BTreeMap<OperandIdx, ResultNum>,
    /// `matched_ops_` — ⛔ POPULATED IN REVERSE ORDER (`:217`), which is why
    /// [`Self::update_end_ops_of_matched_ops`] walks it forwards against a backwards block walk and
    /// [`Self::update_body`] walks it backwards.
    pub(crate) matched_ops: Vec<MatchedOp>,
}

impl LoopRollingManager {
    /// A MANAGER FOR ONE ATTEMPT — the reference's constructor (`:221-231`).
    ///
    /// ⛔ NOT AN ANCHORED UNIT (an excluded trivial constructor).
    pub(crate) fn over(
        case: RollingCase,
        new_loop_count: NewLoopCount,
        start: WindowIndex,
        window_list_end: WindowIndex,
    ) -> LoopRollingManager {
        LoopRollingManager {
            case,
            new_loop_count,
            start,
            window_list_end,
            operand_to_result_num: BTreeMap::new(),
            matched_ops: Vec::new(),
        }
    }

    /// Replaces: e081_deleteMatchedOps
    ///
    /// Drops every match collected for this attempt, so the manager can be reused for the next one.
    ///
    /// ⭐ THE `delete` IS THE `Vec`: the reference holds `MatchedOp *` and frees each before
    /// clearing, and owning them by value makes the two statements one.
    pub(crate) fn delete_matched_ops(&mut self) {
        self.matched_ops.clear();
    }

    /// Replaces: e082_updateEndOpsOfMatchedOps
    ///
    /// Points each match at its instance in `end_window`, walking that window backwards from its end
    /// as the matches were collected forwards.
    ///
    /// ⛔ `std::prev(begin())` IS INEXPRESSIBLE RATHER THAN CHECKED: the reference steps the iterator
    /// once per match with nothing stopping it at the block's start, and a reversed range that simply
    /// runs out is the type guard for that instead of an assertion.
    pub(crate) fn update_end_ops_of_matched_ops(&mut self, end_window: &Window) {
        for (at, matched_op) in (0..=end_window.end.0).rev().zip(self.matched_ops.iter_mut()) {
            matched_op.end_op = InstrPos(at);
        }
    }

    /// Replaces: e083_computeValsIfDifferent
    ///
    /// The difference between two windows' instances of one operand, when both are constants or
    /// uniform mappings; `Incomputable` when either is not, or when exactly one is a block argument.
    ///
    /// ⛔ TWO EQUAL VALUES ARE NOT COMPARED AT ALL and neither are two block arguments — both are
    /// `Same`, which is what lets a rolled op keep reading a loop-invariant value.
    /// ⭐ ASSOCIATED, NOT A METHOD: the only member it used was `oe_`; see the type's note.
    pub(crate) fn compute_vals_if_different(
        a_operand: Val,
        b_operand: Val,
        key_vals: &[Val],
        defs: Definitions<'_>,
    ) -> OperandDifference {
        let a_def = defs.of(a_operand);
        let b_def = defs.of(b_operand);
        let mut found_mismatch = false;
        if a_operand != b_operand {
            match (a_def, b_def) {
                // Both block arguments: `found_mismatch = false`.
                (None, None) => {}
                (Some(a_def), Some(b_def)) => {
                    found_mismatch = !ops_are_equivalent(a_def, b_def, defs);
                }
                // One is a block argument and the other is not, so no delta can be computed.
                (Some(_), None) | (None, Some(_)) => return OperandDifference::Incomputable,
            }
        }
        if !found_mismatch {
            return OperandDifference::Same;
        }
        // ⭐ REACHABLE ONLY THROUGH THE `(Some, Some)` ARM ABOVE, which is what makes the reference's
        // unchecked `*a_operand_op` safe there.
        let (Some(a_def), Some(b_def)) = (a_def, b_def) else {
            return OperandDifference::Incomputable;
        };
        let a_const = scalar_constant_value(a_def);
        let b_const = scalar_constant_value(b_def);
        let query_a = query_map_of(a_def);
        let query_b = query_map_of(b_def);
        match (a_const, b_const, query_a, query_b) {
            (Some(a), Some(b), _, _) => OperandDifference::Delta(Delta(b - a)),
            (Some(a), None, _, Some(map_b)) => {
                match uniform_value(&constant_or_symbol_target_values(map_b, defs)) {
                    Some(b) => OperandDifference::Delta(Delta(b - a)),
                    None => OperandDifference::Incomputable,
                }
            }
            (None, Some(b), Some(map_a), _) => {
                match uniform_value(&constant_or_symbol_target_values(map_a, defs)) {
                    Some(a) => OperandDifference::Delta(Delta(b - a)),
                    None => OperandDifference::Incomputable,
                }
            }
            (None, None, Some(map_a), Some(map_b)) => {
                match deltas_between_values_of_uniform_mappings(map_a, map_b, key_vals, defs) {
                    Some(deltas) => match uniform_delta(&deltas) {
                        Some(delta) => OperandDifference::Delta(delta),
                        None => OperandDifference::Incomputable,
                    },
                    None => OperandDifference::Incomputable,
                }
            }
            _ => OperandDifference::Incomputable,
        }
    }

    /// Replaces: e565_collectDeltasOfOperands
    ///
    /// Classifies every operand of `op_b` against `op_a`'s, recording the kind — and for a constant
    /// stride the delta — on `matched_op`, and refusing the match when one cannot be classified.
    ///
    /// ⛔ FOUR OPERANDS REFUSE THE WHOLE MATCH WHEN THEY DIFFER (`:414-459`): a transfer's
    /// `$increment`, a `sentient.set_mask`'s mask, a loop's `$bound`, and — in the L3 case only — an
    /// `$immutable_addr` on an LX base. ⭐ THE PAIRED READER IS [`Self::check_deltas_of_operands`],
    /// which re-asks for exactly the kinds this one wrote.
    pub(crate) fn collect_deltas_of_operands(
        &mut self,
        start_window: &Window,
        op_a: InstrPos,
        op_b: InstrPos,
        matched_op: &mut MatchedOp,
        block: &[Op],
        key_vals: &[Val],
        defs: Definitions<'_>,
    ) -> bool {
        let a_operands = dialects::operands(&block[op_a.0]);
        let b_operands = dialects::operands(&block[op_b.0]);
        for i in (0..a_operands.len()).rev() {
            let i = OperandIdx(i);
            let a_operand = a_operands[i.0];
            // See [`Self::check_deltas_of_operands`] on the reference's unchecked `op_b` index.
            let Some(b_operand) = b_operands.get(i.0).copied() else {
                return false;
            };

            if a_operand == b_operand {
                matched_op.insert_operand_kind(i, OperandKind::NoDelta);
                continue;
            }

            if is_increment_field(&block[op_a.0], i)
                || matches!(&block[op_a.0], Op::Sentient(sentient::Op::SetMask { .. }))
            {
                return false;
            }
            // `dyn_cast<sentient::ForOp>(op_a)` and `a_operand == for_op_a.getBound()`: a loop whose
            // bound is a block argument is not one this pass rolls.
            if let Op::Sentient(sentient::Op::For { bound, .. }) = &block[op_a.0] {
                if a_operand == *bound {
                    return false;
                }
            }
            if self.case == RollingCase::L3SoftSyncWindows
                && is_immutable_address_lx_address(&block[op_a.0], i, defs)
            {
                return false;
            }

            // `op_b` (the next window's first rollable op) reading one of the start window's last
            // rollable op's results.
            if start_window.first_rollable_op == Some(op_a) {
                if let Some((at, res_num)) = defining_result(b_operand, block) {
                    if start_window.last_rollable_op == Some(at) {
                        self.operand_to_result_num.insert(i, res_num);
                        matched_op.insert_operand_kind(i, OperandKind::FirstLastOpUsage);
                        continue;
                    }
                }
            }

            match LoopRollingManager::compute_vals_if_different(a_operand, b_operand, key_vals, defs)
            {
                OperandDifference::Delta(delta) if delta != Delta(0) => {
                    matched_op.insert_operand_delta(i, delta);
                    matched_op.insert_operand_kind(i, OperandKind::ConstValueDelta);
                }
                // `delta == 0`, and `Same` is the reference's untouched pre-set 0.
                OperandDifference::Delta(_) | OperandDifference::Same => {
                    matched_op.insert_operand_kind(i, OperandKind::NoDelta);
                }
                OperandDifference::Incomputable => return false,
            }
        }
        true
    }

    /// Replaces: e317_checkDeltasOfOperands
    ///
    /// Whether every operand of `op_b` differs from `op_a`'s in exactly the way `matched_op` recorded
    /// — the same kind, and for a constant stride the same delta.
    ///
    /// ⛔ THE WALK IS BACKWARDS (`:514`) and each slot is asked for its kind whether or not it
    /// differs, so an unclassified operand is the loud `todo!` of [`MatchedOp::operand_kind`] rather
    /// than a silent pass.
    /// ⭐ THE `kFirstLastOpUsage` ARM IS THE ONLY ONE THAT NEEDS POSITIONS: it asks whether `op_a` IS
    /// the window's first rollable op and whether `op_b`'s operand comes out of its last.
    pub(crate) fn check_deltas_of_operands(
        &self,
        cur_window: &Window,
        op_a: InstrPos,
        op_b: InstrPos,
        matched_op: &MatchedOp,
        block: &[Op],
        key_vals: &[Val],
        defs: Definitions<'_>,
    ) -> bool {
        let a_operands = dialects::operands(&block[op_a.0]);
        let b_operands = dialects::operands(&block[op_b.0]);
        for i in (0..a_operands.len()).rev() {
            let i = OperandIdx(i);
            let a_operand = a_operands[i.0];
            // The reference indexes `op_b` with `op_a`'s count unchecked; the two are equivalent ops
            // by the time `compareToNextWindow` gets here, so a shorter `op_b` is not a state to
            // classify.
            let Some(b_operand) = b_operands.get(i.0).copied() else {
                return false;
            };
            let operand_kind = matched_op.operand_kind(i);

            if a_operand == b_operand {
                if operand_kind != OperandKind::NoDelta {
                    return false;
                }
                continue;
            }

            // `op_b` (the next window's first rollable op) reading one of this window's last
            // rollable op's results.
            if cur_window.first_rollable_op == Some(op_a) {
                if let Some((at, res_num)) = defining_result(b_operand, block) {
                    if cur_window.last_rollable_op == Some(at) {
                        if operand_kind != OperandKind::FirstLastOpUsage
                            || self.operand_to_result_num.get(&i) != Some(&res_num)
                        {
                            return false;
                        }
                        continue;
                    }
                }
            }

            match LoopRollingManager::compute_vals_if_different(a_operand, b_operand, key_vals, defs)
            {
                OperandDifference::Delta(delta) if delta != Delta(0) => {
                    if operand_kind != OperandKind::ConstValueDelta
                        || matched_op.operand_deltas.get(&i) != Some(&delta)
                    {
                        return false;
                    }
                }
                // `delta == 0`, and `Same` is the reference's "delta untouched" over its pre-set 0.
                OperandDifference::Delta(_) | OperandDifference::Same => {
                    if operand_kind != OperandKind::NoDelta {
                        return false;
                    }
                }
                OperandDifference::Incomputable => return false,
            }
        }
        true
    }

    /// Replaces: e602_compareToNextWindow
    ///
    /// Whether `next_window` repeats `cur_window`: the two are walked BACKWARDS in step from their
    /// ends, and a start window files a match per pair while a follower re-checks the filed deltas
    /// (`:321-386`).
    ///
    /// ⛔ THE OPERANDS ARE ALLOWED TO DIFFER AT THE TOP LEVEL AND ONLY THERE — see
    /// [`ops_are_equivalent_ignoring_operands`]; comparing them is this function's own job.
    /// ⛔ THE EFFECTIVE START IS THE LAST PAIR MATCHED (`:381-384`), which `++it_cur` recovers after
    /// the third clause has already stepped past it — an empty walk therefore leaves it PAST the end.
    /// ⭐ THE PAIRED RANGES ARE THE TYPE GUARD for `std::prev` off the front of the block, the same
    /// way [`Self::update_end_ops_of_matched_ops`] bounds its own backwards step.
    pub(crate) fn compare_to_next_window(
        &mut self,
        role: WindowRole,
        cur_window: &mut Window,
        next_window: &Window,
        block: &[Op],
        key_vals: &[Val],
        defs: Definitions<'_>,
    ) -> bool {
        // `it_next != cur_window->getEnd()` (`:354-355`) — one pair per instruction between the ends.
        let pairs = next_window.end.0.saturating_sub(cur_window.end.0);
        // `++it_cur` on an untouched iterator (`:384`).
        let mut effective_start = InstrPos(cur_window.end.0 + 1);
        for (at, (cur, next)) in (0..=cur_window.end.0)
            .rev()
            .zip((0..=next_window.end.0).rev())
            .take(pairs)
            .enumerate()
        {
            let (cur_op, next_op) = (InstrPos(cur), InstrPos(next));
            if !ops_are_equivalent_ignoring_operands(&block[cur], &block[next], defs) {
                return false;
            }
            if !cur_window.check_op_usage(cur_op, block, next_window) {
                return false;
            }
            effective_start = cur_op;
            match role {
                WindowRole::Start => {
                    let mut matched_op = MatchedOp::of(cur_op);
                    // `delete matched_op` on the refusal (`:388-391`) IS THE DROP: it is never filed.
                    if !self.collect_deltas_of_operands(
                        cur_window,
                        cur_op,
                        next_op,
                        &mut matched_op,
                        block,
                        key_vals,
                        defs,
                    ) {
                        return false;
                    }
                    self.matched_ops.push(matched_op);
                }
                // `*it_matched_op` — the reference walks the list alongside and does not check its
                // end, for the reason [`Self::check_deltas_of_operands`] gives about `op_b`.
                WindowRole::Follower => {
                    let Some(matched_op) = self.matched_ops.get(at) else {
                        return false;
                    };
                    if !self.check_deltas_of_operands(
                        cur_window, cur_op, next_op, matched_op, block, key_vals, defs,
                    ) {
                        return false;
                    }
                }
            }
        }
        if role == WindowRole::Start {
            cur_window.set_effective_start(effective_start, WindowSize(self.matched_ops.len()));
        }
        true
    }

    /// Replaces: e512_collectStartingValIterArgs
    ///
    /// Collects the starting value of every iteration argument the rolled loop needs — one per
    /// recorded delta, one per usage of the last rollable op in the next window's first rollable op,
    /// and `zero` for each result read past the last window rolled — refusing when more than
    /// [`MAX_ITER_ARGS_CREATED`] would be built.
    ///
    /// ⛔ `zero` IS ERASED ON EVERY PATH THAT DOES NOT PLACE IT, the refusal included: the caller
    /// mints it into `consts` before asking (`LoopRolling.cpp:806-812`).
    pub(crate) fn collect_starting_val_iter_args(
        &mut self,
        start_window: &Window,
        cur_window: &Window,
        starting_vals: &mut Vec<Val>,
        zero: Val,
        block: &[Op],
        consts: &mut Vec<Op>,
    ) -> bool {
        let mut used_zero = false;
        // `operand_to_result_num` is read while the matches are written, so the two are split.
        let LoopRollingManager {
            matched_ops,
            operand_to_result_num,
            ..
        } = self;
        // `matched_ops_` is populated in reverse, so `rbegin()` walks the block FORWARDS.
        for matched_op in matched_ops.iter_mut().rev() {
            let op_in_start_window = matched_op.start_op;
            // Iterator argument(s) for deltas.
            for operand_idx in matched_op.operand_deltas.keys() {
                starting_vals.push(dialects::operands(&block[op_in_start_window.0])[operand_idx.0]);
            }

            // Iterator argument(s) for usages of the last rollable op in the next window's first
            // rollable op.
            if start_window.last_rollable_op == Some(op_in_start_window) {
                let Some(first_rollable_op) = start_window.first_rollable_op else {
                    todo!(
                        "collectStartingValIterArgs: the start window has a last rollable op but no                          first one (LoopRolling.cpp:615-616)"
                    )
                };
                for (operand_idx, result_num) in operand_to_result_num.iter() {
                    // Several iteration arguments may name the same result when it is read more than
                    // once, to allow for different starting values.
                    let iter_arg_num = IterArgIdx(starting_vals.len());
                    matched_op
                        .iter_arg_to_res_num_and_operand
                        .insert(iter_arg_num, (*result_num, Some(*operand_idx)));
                    starting_vals
                        .push(dialects::operands(&block[first_rollable_op.0])[operand_idx.0]);
                }
            }

            // Iterator argument(s) for usages of ops past the last window rolled. These get the
            // arbitrary starting value 0.
            let op_in_cur_window = matched_op.end_op;
            for (i, result) in dialects::results(&block[op_in_cur_window.0])
                .into_iter()
                .enumerate()
            {
                for user in users_of(result, block) {
                    if user.is_some_and(|user| cur_window.is_op_in_this_window(user)) {
                        continue;
                    }
                    // Try reusing an existing iterator argument.
                    let used_previously = matched_op
                        .iter_arg_to_res_num_and_operand
                        .values()
                        .any(|(result_num, _)| *result_num == ResultNum(i));
                    if !used_previously {
                        matched_op
                            .iter_arg_to_res_num_and_operand
                            .insert(IterArgIdx(starting_vals.len()), (ResultNum(i), None));
                        used_zero = true;
                        starting_vals.push(zero);
                    }
                }
            }
        }
        let within_allowance = starting_vals.len() <= MAX_ITER_ARGS_CREATED;
        if !within_allowance || !used_zero {
            erase_definition(zero, consts);
        }
        within_allowance
    }

    /// Replaces: e084_updateBody
    ///
    /// Moves each match's end instance into the loop, re-points its delta operands at the matching
    /// `iter_arg`, yields `iter_arg + delta` for each, and hands every reader past the last rolled
    /// window the loop's own result instead.
    ///
    /// ⛔ THE CONSTANTS GO TO `consts`, NOT THE LOOP BODY: `const_builder_` is anchored at the start
    /// of the block CONTAINING the program unit (`:990-991`), which is a different block from `bb_`.
    /// ⭐ THE MOVES ARE ONE SPLICE AT THE END, so every [`InstrPos`] stays valid while it is read.
    /// ⚠️ `start_window` IS UNUSED IN THE REFERENCE'S BODY and is dropped.
    pub(crate) fn update_body(
        &mut self,
        cur_window: &Window,
        rolled: &mut RolledLoop,
        block: &mut Vec<Op>,
        consts: &mut Vec<Op>,
        values: &mut Values,
    ) {
        let mut iter_arg_idx = 0usize;
        let mut moved: Vec<InstrPos> = Vec::new();
        // `matched_ops_` is populated in reverse, so `rbegin()` walks the block FORWARDS.
        for matched_op in self.matched_ops.iter().rev() {
            let at = matched_op.end_op;
            moved.push(at);
            // Iterator arguments corresponding to deltas.
            for (operand, delta) in &matched_op.operand_deltas {
                let arg = rolled.carried[iter_arg_idx].arg;
                let constant = values.mint();
                consts.push(scalar_constant(constant, delta.0));
                let sum = values.mint();
                rolled.body.push(scalar_add(sum, arg, constant));
                dialects::set_operand(&mut rolled.body[rolled.yield_at], iter_arg_idx, sum);
                dialects::set_operand(&mut block[at.0], operand.0, arg);
                iter_arg_idx += 1;
            }
            // Iterator arguments corresponding to results.
            for (iter_arg, (result_num, operand)) in &matched_op.iter_arg_to_res_num_and_operand {
                if iter_arg.0 != iter_arg_idx {
                    todo!(
                        "updateBody: \"Iterator argument should correspond to result number from \
                         op usage.\" (LoopRolling.cpp:718-721): {iter_arg:?} against {iter_arg_idx}"
                    );
                }
                if let Some(operand) = operand {
                    let Some(first) = cur_window.first_rollable_op else {
                        todo!(
                            "updateBody: an operand index was recorded for a window with no first \
                             rollable op (LoopRolling.cpp:726-729)"
                        )
                    };
                    dialects::set_operand(
                        &mut block[first.0],
                        operand.0,
                        rolled.carried[iter_arg_idx].arg,
                    );
                }
                let result = dialects::results(&block[at.0])[result_num.0];
                replace_uses_outside_window(
                    block,
                    cur_window,
                    result,
                    rolled.carried[iter_arg_idx].result,
                );
                dialects::set_operand(&mut rolled.body[rolled.yield_at], iter_arg_idx, result);
                iter_arg_idx += 1;
            }
        }
        // `op.moveBefore(yield_op)`, deferred: descending removal keeps the earlier positions valid,
        // and the ops land ahead of the yield in block order.
        let mut taken: Vec<Op> = Vec::new();
        for at in moved.iter().rev() {
            taken.push(block.remove(at.0));
        }
        taken.reverse();
        let inserted = taken.len();
        for (n, op) in taken.into_iter().enumerate() {
            rolled.body.insert(rolled.yield_at + n, op);
        }
        rolled.yield_at += inserted;
    }

    /// Replaces: e628_matchAndRoll
    ///
    /// Matches windows forward from `start` for as long as each repeats the last, and — once enough of
    /// them have — rolls them into one `sentient.for` that replaces the originals (`:758-861`).
    ///
    /// ⛔ `next` IS AN IN/OUT PARAMETER, exactly as the reference's `iterator &`: on a mismatch it is
    /// left ON the window that failed, which is the next attempt's start.
    /// ⛔ THE ERASED RANGE IS `[start.effective_start, second_last.end]` — the LAST window's ops are
    /// not erased, they were MOVED into the body by [`Self::update_body`], and the loop takes their
    /// place. ⭐ `oe_.clearCache()` (`:857`) is dropped with the cache it clears.
    #[expect(
        clippy::too_many_arguments,
        reason = "the reference reaches the block, the constant block, the window list and the value \
                  minter through `this` and two `OpBuilder &`s, and neither is a thing this crate has"
    )]
    pub(crate) fn match_and_roll(
        &mut self,
        next: &mut WindowIndex,
        windows: &mut [Window],
        block: &mut Vec<Op>,
        consts: &mut Vec<Op>,
        key_vals: &[Val],
        values: &mut Values,
    ) {
        let (Some(start_size), Some(start_end)) = (
            windows.get(self.start.0).map(|window| window.size),
            windows.get(self.start.0).map(|window| window.end),
        ) else {
            return;
        };
        // "Cannot match if next window is larger."
        let Some(next_size) = windows.get(next.0).map(|window| window.size) else {
            return;
        };
        if next_size.0 > start_size.0 {
            return;
        }
        // "Cannot match certain instructions in the single instruction case."
        if self.case == RollingCase::SingleInstrWindows
            && matches!(
                block.get(start_end.0),
                Some(
                    Op::Sentient(sentient::Op::ScalarConstant { .. })
                        | Op::Dataflow(dataflow::Op::GetUnit { .. })
                )
            )
        {
            return;
        }

        // `auto cur = start_;` and its predecessor, carried rather than recomputed by `std::prev`:
        // the profitability test below is what makes `second_last` a window at all.
        let mut cur = self.start;
        let mut second_last = self.start;
        let mut count: u32 = 1;
        while *next != self.window_list_end {
            // "No hope to match windows if their sizes are different."
            let (Some(cur_size), Some(next_size)) = (
                windows.get(cur.0).map(|window| window.size),
                windows.get(next.0).map(|window| window.size),
            ) else {
                break;
            };
            if count > 1 && next_size != cur_size {
                break;
            }
            let role = if cur == self.start {
                WindowRole::Start
            } else {
                WindowRole::Follower
            };
            let (before, from_next) = windows.split_at_mut(next.0);
            let (Some(cur_window), Some(next_window)) = (before.get_mut(cur.0), from_next.first())
            else {
                break;
            };
            // The definitions are read-only and only this phase needs them, so they are built here
            // rather than held: `consts` is the enclosing block a hoisted constant is defined in.
            let regions: [&[Op]; 2] = [block, consts];
            let matched = self.compare_to_next_window(
                role,
                cur_window,
                next_window,
                block,
                key_vals,
                Definitions::from_innermost(&regions),
            );
            if !matched {
                break;
            }
            count += 1;
            second_last = cur;
            cur = *next;
            *next = WindowIndex(next.0 + 1);
        }

        // "In the single instruction case, it's profitable to roll only if at least 4 matching
        // windows are found."
        if count > 3 || (self.case == RollingCase::L3SoftSyncWindows && count > 1) {
            let (Some(cur_end), Some(effective_start), Some(second_last_end)) = (
                windows.get(cur.0).map(|window| window.end),
                windows.get(self.start.0).map(|window| window.effective_start),
                windows.get(second_last.0).map(|window| window.end),
            ) else {
                self.delete_matched_ops();
                return;
            };
            // Everything after the last rolled window is touched by neither the moves nor the
            // erasure, so its length is where the new loop goes.
            let after_cur_window = block.len().saturating_sub(cur_end.0 + 1);
            if let Some(end_window) = windows.get(cur.0) {
                self.update_end_ops_of_matched_ops(end_window);
            }

            // "Determine the starting values of the new iterator arguments. If the number of iterator
            // arguments exceeds the allowance, do not roll." `zero` is minted first because e512 is
            // the one that erases it again on every path that does not place it.
            let zero = values.mint();
            consts.push(scalar_constant(zero, 0));
            let mut starting_vals: Vec<Val> = Vec::new();
            let collected = match (windows.get(self.start.0), windows.get(cur.0)) {
                (Some(start_window), Some(cur_window)) => self.collect_starting_val_iter_args(
                    start_window,
                    cur_window,
                    &mut starting_vals,
                    zero,
                    block,
                    consts,
                ),
                _ => false,
            };
            if collected {
                let bound = values.mint();
                consts.push(scalar_constant(bound, i64::from(count)));
                self.new_loop_count = NewLoopCount(self.new_loop_count.0 + 1);
                // `body_block.addArgument` once for the induction variable, then one per starting
                // value; `regLocales` is created from an EMPTY vector (`:816-819`).
                let iv = values.mint();
                let carried: Vec<sentient::Carried> = starting_vals
                    .iter()
                    .map(|init| sentient::Carried {
                        init: *init,
                        arg: values.mint(),
                        result: values.mint(),
                        reg: sentient::Reg {
                            locale: sentient::RegType::Unknown,
                            index: None,
                        },
                        program_header: false,
                        element_size: None,
                    })
                    .collect();
                let mut rolled = RolledLoop {
                    carried,
                    body: vec![Op::Sentient(sentient::Op::Yield {
                        results: starting_vals.clone(),
                    })],
                    yield_at: 0,
                };
                if let Some(cur_window) = windows.get(cur.0) {
                    self.update_body(cur_window, &mut rolled, block, consts, values);
                }
                // `yield_op->moveBefore(&body_block, body_block.end())` (`:844`) — the adds e084 built
                // after it now precede it.
                let yielded = rolled.body.remove(rolled.yield_at);
                rolled.body.push(yielded);
                rolled.yield_at = rolled.body.len() - 1;

                // "Remove the original instructions, proceeding backwards from the second last
                // window's end", the trailing `start_of_first_window->erase()` included — one closed
                // range, since a backwards walk that erases as it goes is exactly a drain.
                if effective_start.0 <= second_last_end.0 && second_last_end.0 < block.len() {
                    block.drain(effective_start.0..=second_last_end.0);
                }
                let at = block.len().saturating_sub(after_cur_window);
                block.insert(
                    at,
                    Op::Sentient(sentient::Op::For {
                        iv,
                        bound,
                        bound_reg: None,
                        carried: rolled.carried,
                        dbg_name: Some(format!("LR loop #{}", self.new_loop_count.0)),
                        body: rolled.body,
                    }),
                );
            }
        }
        self.delete_matched_ops();
    }
}

/// Replaces: e642_rollInstrsInBlock
///
/// Collects the block's windows — soft-sync delimited in the L3 case, one rollable instruction each
/// otherwise — and rolls every run of matching consecutive windows (`:886-964`).
///
/// ⛔ EVERY WINDOW PAST `next` IS REMAPPED AFTER A ROLL: the reference's `Block::iterator`s survive
/// another op's erasure and an [`InstrPos`] ordinal does not, and they all sit in the untouched tail.
/// ⭐ THE L3 RECURSION IS HOISTED ahead of collection; it rewrites only nested bodies.
pub(crate) fn roll_instrs_in_block(
    case: RollingCase,
    block: &mut Vec<Op>,
    consts: &mut Vec<Op>,
    key_vals: &[Val],
    new_loop_count: &mut NewLoopCount,
    values: &mut Values,
) {
    // "In the L3 case, we need to recurse inside uniformize regions and equalize patterns as their
    // contents are still at the outermost level so LoopRolling may be applied."
    if case == RollingCase::L3SoftSyncWindows {
        for at in 0..block.len() {
            if !matches!(&block[at], Op::UniformRegions(_)) {
                continue;
            }
            for region in dialects::regions_mut(&mut block[at]) {
                roll_instrs_in_block(case, region, consts, key_vals, new_loop_count, values);
            }
        }
    }

    // "Collect the windows delimited by soft sync's in the L3 case, or size 1 windows in the single
    // instruction case" — a window that closes on neither is the reference's `delete w`, which here
    // is the binding going out of scope.
    let mut windows: Vec<Window> = Vec::new();
    let mut at = 0;
    while at < block.len() {
        let mut window = Window::opening(case, InstrPos(at));
        while at < block.len() {
            let this = InstrPos(at);
            window.insert_next_instr(this, block);
            at += 1;
            let closes = match case {
                // "Only push windows terminating in a soft sync."
                RollingCase::L3SoftSyncWindows => matches!(
                    &block[this.0],
                    Op::Sentient(sentient::Op::Sync { soft: true, .. })
                ),
                // "Greedily include SetMaskOps/ConstantOps/QueryMapOps/DefImmutableMappingOps at the
                // start of the current window and IncrMask ops at the end of the current window."
                RollingCase::SingleInstrWindows => {
                    !matches!(
                        &block[this.0],
                        Op::Sentient(
                            sentient::Op::SetMask { .. } | sentient::Op::ScalarConstant { .. }
                        ) | Op::Uniform(
                            uniform::Op::DefImmutableMapping { .. } | uniform::Op::QueryMap { .. }
                        )
                    ) && !matches!(
                        block.get(at),
                        Some(Op::Sentient(sentient::Op::IncrMask { .. }))
                    )
                }
            };
            if closes {
                windows.push(window);
                break;
            }
        }
    }

    // "Try to roll sequences of consecutive equivalent windows": `next` is left on the window that
    // failed to match, and that window is the next instance's start.
    let mut start = WindowIndex(0);
    while start.0 + 1 < windows.len() {
        let mut next = WindowIndex(start.0 + 1);
        let end = WindowIndex(windows.len());
        let mut manager = LoopRollingManager::over(case, *new_loop_count, start, end);
        let length_before = block.len();
        manager.match_and_roll(&mut next, &mut windows, block, consts, key_vals, values);
        // `int &new_loop_count_`, shared across every manager this block builds.
        *new_loop_count = manager.new_loop_count;
        let moved_by = block.len() as isize - length_before as isize;
        if moved_by != 0 {
            for window in windows.iter_mut().skip(next.0) {
                shift_window(window, moved_by);
            }
        }
        start = next;
    }
}

/// EVERY POSITION ONE WINDOW HOLDS, MOVED BY `by` — what a `Block::iterator` gets for free from the
/// erasure of another op and an [`InstrPos`] ordinal does not.
///
/// ⛔ NOT AN ANCHORED UNIT: it exists because this module names an instruction by its ordinal, which
/// [`InstrPos`] says out loud, and the reference has nothing for it to replace.
fn shift_window(window: &mut Window, by: isize) {
    let moved = |at: InstrPos| InstrPos(at.0.saturating_add_signed(by));
    window.start = moved(window.start);
    window.effective_start = moved(window.effective_start);
    window.end = moved(window.end);
    let op_to_index = window
        .op_to_index
        .iter()
        .map(|(at, index)| (moved(*at), *index))
        .collect();
    window.op_to_index = op_to_index;
    window.first_rollable_op = window.first_rollable_op.map(moved);
    window.last_rollable_op = window.last_rollable_op.map(moved);
}

/// The `uniform.query_map`'s `$map`, when that is what defines a value.
fn query_map_of(op: &Op) -> Option<Val> {
    match op {
        Op::Uniform(uniform::Op::QueryMap { map, .. }) => Some(*map),
        _ => None,
    }
}

/// The one value a mapping's targets all share — `vals.front()` once every entry equals it.
fn uniform_value(values: &[i64]) -> Option<i64> {
    let first = *values.first()?;
    values.iter().all(|value| *value == first).then_some(first)
}

/// The one delta a pair of mappings shares — `deltas.front()` once every entry equals it.
fn uniform_delta(deltas: &[Delta]) -> Option<Delta> {
    let first = *deltas.first()?;
    deltas.iter().all(|delta| *delta == first).then_some(first)
}

/// The body of the `sentient.for` whose induction variable is `iv` — `*for_op.getBody(0)`, the loop
/// being named by the value it binds ([`ForRef`], which is how the tree holds it).
fn for_body_mut(block: &mut Vec<Op>, iv: Val) -> Option<&mut Vec<Op>> {
    for op in block.iter_mut() {
        let regions: Vec<&mut Vec<Op>> = match op {
            Op::Sentient(sentient::Op::For { iv: at, body, .. }) => {
                if *at == iv {
                    return Some(body);
                }
                vec![body]
            }
            other => dialects::regions_mut(other),
        };
        for region in regions {
            if let Some(found) = for_body_mut(region, iv) {
                return Some(found);
            }
        }
    }
    None
}

/// `tree.walk(rollSingleInstrsInInnermostLoop, kPreOrder)` REDUCED TO THE NODES ITS ACTION ACTS ON —
/// `if (!n->isInnermostLoop()) return nullptr;`.
///
/// ⭐ A FIXED LIST IS THE SAME WALK HERE, unlike [`super::loop_absorption`]'s: this action removes no
/// node, so `preOrderWalk`'s re-reading of `getFirstChild()`/`getNextSibling()` sees what it saw.
/// ⭐ THE SYNTHETIC ROOT IS FILTERED BY [`LoopTree::loop_of`], which is `getOpAs<sentient::ForOp>()`.
fn innermost_loops(tree: &LoopTree<false>, n: LoopNodeId, out: &mut Vec<ForRef>) {
    if tree.is_innermost_loop(n) {
        out.extend(tree.loop_of(n));
        return;
    }
    let mut child = tree.first_child(n);
    while let Some(c) = child {
        innermost_loops(tree, c, out);
        child = tree.next_sibling(c);
    }
}

/// `-dcc-loop-rolling-disable`, `cl::init(false)` (`:47-49`) — a `dcc-opt` command-line flag, not a
/// program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `-l3-loop-rolling-only`, `cl::init(false)` (`:50-54`) — *"Only roll windows delimited by soft syncs
/// at the outermost level in L3 units."*
const L3_ONLY_LOOP_ROLLING: bool = false;

/// `-single-instr-loop-rolling-only`, `cl::init(false)` (`:55-58`) — *"Only roll windows of size 1 at
/// the innermost level."*
const SINGLE_INSTR_ONLY_LOOP_ROLLING: bool = false;

/// `opts_.OptLevel == 0` (`:971`) — the PIPELINE's optimisation level, which is `2` by default
/// (`dcc/tools/Options/dcc-pass-option.h:63-65`), so the shipped pipeline never reaches the
/// `haveIbuffSpace` half of the `&&`.
const OPT_LEVEL_ZERO: bool = false;

/// Replaces: e652_runOnOperation
///
/// The pass entry: for every program unit the filter admits and whose instruction buffer is not
/// already roomy, roll the soft-sync windows of its outermost blocks — or, when only single-instruction
/// rolling is on, the single instructions of each of its innermost loops (`:966-1018`).
///
/// ⛔ THE TWO CASES ARE EXCLUSIVE AND `L3_rolling_` WINS: with a filter that names nothing both flags
/// come out true (`:873-879`), so the shipped standalone invocation takes the L3 arm only.
/// ⛔ `new_loop_count_` IS A PASS MEMBER: the `LR loop #n` numbering runs across the whole module.
/// ⭐ `key_vals` IS THE UNIT'S OWN LIST, from `getListOfKeyOpsFromUniformMapping`'s
/// `dataflow.program_unit` arm (`Dialect/Uniform/Utils.cpp:180-186`).
/// ⭐ `oe`/`const_builder` ARE THE MECHANISM FOR REACHING OPERANDS, which the campaign names
/// droppable — the builder's block is the one the unit sits in, which is `preamble`.
/// ⭐ `unit_list` COLLAPSES for [`super::loop_absorption::run_on`]'s reason: rolling in one unit
/// reaches no other, and the gate that selects a unit rewrites nothing.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    opts: &UnitFilter,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    // The constructor's two flags (`:873-879`): *"the include/exclude list for this pass is either
    // empty (if the pass is called with --dcc-loop-rolling) or {L3SU, L3LU} (if the pass is called in
    // the pipeline)"*.
    let is_invoked_by_option = opts.is_empty();
    let l3_rolling =
        (is_invoked_by_option || opts.is_include_list()) && !SINGLE_INSTR_ONLY_LOOP_ROLLING;
    let single_instr_loop_rolling =
        (is_invoked_by_option || !opts.is_include_list()) && !L3_ONLY_LOOP_ROLLING;

    let Program {
        preamble, units, ..
    } = program;
    let mut new_loop_count = NewLoopCount(0);
    for unit in units.iter_mut() {
        // `getChildAnalysis<InstructionEstimator>(unit_op)` IS CONSTRUCTED PER UNIT (`:970`), even for
        // one the `&&` never asks anything of.
        let mut instruction_estimator = OutOfScopeInstructionEstimator;
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        // `DT_CHECK(unit_op.getUnits().size() >= 1)` IS DISCHARGED BY THE TYPE: a
        // [`Units`](crate::islands::dataflow_ir::Units) always has a head, and `getUnitType` of it is
        // the GENERIC component, which is [`UnitFilter::names_component`].
        let unit_is_in_incl_excl_list = opts.names_component(unit.on.kind());
        let skip_loop_rolling = if opts.is_include_list() {
            !unit_is_in_incl_excl_list
        } else {
            unit_is_in_incl_excl_list
        };
        if skip_loop_rolling {
            continue;
        }
        let key_vals = {
            let scope: [&[Op]; 1] = [preamble.as_slice()];
            dialects::collect_unit_ops(&unit.on.vals(), Definitions::from_innermost(&scope))
        };
        if l3_rolling {
            roll_instrs_in_block(
                RollingCase::L3SoftSyncWindows,
                &mut unit.body,
                preamble,
                &key_vals,
                &mut new_loop_count,
                values,
            );
        } else if single_instr_loop_rolling {
            // ⛔ THE TREE IS BUILT ONCE, BEFORE ANY ROLL: its nodes are the loops the unit came in
            // with, so a loop this pass creates is never itself descended into.
            let tree: LoopTree<false> = LoopTree::of(&unit.body);
            if tree.empty() {
                continue;
            }
            let mut innermost = Vec::new();
            innermost_loops(&tree, tree.root(), &mut innermost);
            for for_op in innermost {
                if let Some(body) = for_body_mut(&mut unit.body, for_op.0) {
                    roll_instrs_in_block(
                        RollingCase::SingleInstrWindows,
                        body,
                        preamble,
                        &key_vals,
                        &mut new_loop_count,
                        values,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Elements};
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};

    /// A `sentient.scalar_add` reading `lhs`/`rhs` and binding `result`.
    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        scalar_add(result, lhs, rhs)
    }

    /// A `sentient.receive_and_store`, whose operand 2 is `$increment`.
    fn receive_and_store() -> Op {
        Op::Sentient(sentient::Op::ReceiveAndStore {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            producer: sentient::StoreSource::Constant(Val(4)),
            result: Val(5),
            dst: None,
            drop_first: None,
            multicast_info: None,
            extent: sentient::Extent::of(Elements(8), Bits(16)),
            interleaved_group: Elements(0),
            coalesce: false,
            subword_length: 0,
            stride: 0,
            permute: false,
            shuffle_mode: None,
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// `e077_isIncrementField` — the increment slot of a `receive_and_store` and nothing else.
    #[test]
    fn is_increment_field_names_only_the_transfer_increment() {
        let transfer = receive_and_store();
        assert!(is_increment_field(&transfer, OperandIdx(2)));
        assert!(!is_increment_field(&transfer, OperandIdx(0)));
        assert!(!is_increment_field(
            &add(Val(1), Val(2), Val(3)),
            OperandIdx(2)
        ));
    }

    /// `e078_setEffectiveStart` — ⭐ `op_to_index` SURVIVES, which is the reference's own note.
    #[test]
    fn set_effective_start_moves_the_start_and_leaves_the_index_map() {
        let mut window = Window::opening(RollingCase::L3SoftSyncWindows, InstrPos(3));
        window.op_to_index.insert(InstrPos(3), 0);
        window.op_to_index.insert(InstrPos(4), 1);
        window.set_effective_start(InstrPos(4), WindowSize(1));
        assert_eq!(window.effective_start, InstrPos(4));
        assert_eq!(window.size, WindowSize(1));
        assert_eq!(window.start, InstrPos(3));
        assert!(window.is_op_in_this_window(InstrPos(3)));
    }

    /// `e079_checkOpUsage` — the next window's first rollable op is excused; a third reader is not.
    #[test]
    fn check_op_usage_excuses_only_the_next_windows_first_rollable_op() {
        let mut block = vec![
            scalar_constant(Val(0), 0),
            add(Val(0), Val(0), Val(1)),
            add(Val(1), Val(0), Val(2)),
            add(Val(1), Val(0), Val(3)),
        ];
        let mut window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(0));
        window.op_to_index.insert(InstrPos(0), 0);
        window.op_to_index.insert(InstrPos(1), 1);
        window.end = InstrPos(1);
        window.first_rollable_op = Some(InstrPos(1));
        window.last_rollable_op = Some(InstrPos(1));
        let mut next = Window::opening(RollingCase::SingleInstrWindows, InstrPos(2));
        next.first_rollable_op = Some(InstrPos(2));
        assert!(!window.check_op_usage(InstrPos(1), &block, &next));
        block.pop();
        assert!(window.check_op_usage(InstrPos(1), &block, &next));
    }

    /// `e509_insertNextInstr` — the L3 case rolls only the transfers, the generic case anything that
    /// is not a constant or a map lookup.
    #[test]
    fn insert_next_instr_indexes_the_op_and_names_the_rollable_ones() {
        let block = vec![
            scalar_constant(Val(40), 7),
            receive_and_store(),
            add(Val(41), Val(42), Val(43)),
        ];
        let mut l3 = Window::opening(RollingCase::L3SoftSyncWindows, InstrPos(0));
        l3.insert_next_instr(InstrPos(0), &block);
        l3.insert_next_instr(InstrPos(1), &block);
        assert_eq!(l3.size, WindowSize(2));
        assert_eq!(l3.end, InstrPos(1));
        assert_eq!(l3.op_to_index.get(&InstrPos(1)), Some(&1));
        assert_eq!(l3.first_rollable_op, Some(InstrPos(1)));
        assert_eq!(l3.last_rollable_op, Some(InstrPos(1)));

        let mut generic = Window::opening(RollingCase::SingleInstrWindows, InstrPos(0));
        generic.insert_next_instr(InstrPos(0), &block);
        assert_eq!(generic.first_rollable_op, None);
        generic.insert_next_instr(InstrPos(2), &block);
        assert_eq!(generic.first_rollable_op, Some(InstrPos(2)));
    }

    /// `e510_insertOperandKind` — `std::map::insert` does not overwrite.
    #[test]
    fn insert_operand_kind_keeps_the_first_classification() {
        let mut matched_op = MatchedOp::of(InstrPos(0));
        matched_op.insert_operand_kind(OperandIdx(1), OperandKind::ConstValueDelta);
        matched_op.insert_operand_kind(OperandIdx(1), OperandKind::NoDelta);
        assert_eq!(
            matched_op.operand_kind(OperandIdx(1)),
            OperandKind::ConstValueDelta
        );
    }

    /// `e511_insertOperandDelta` — likewise the first stride wins.
    #[test]
    fn insert_operand_delta_keeps_the_first_stride() {
        let mut matched_op = MatchedOp::of(InstrPos(0));
        matched_op.insert_operand_delta(OperandIdx(1), Delta(4));
        matched_op.insert_operand_delta(OperandIdx(1), Delta(9));
        assert_eq!(
            matched_op.operand_deltas.get(&OperandIdx(1)),
            Some(&Delta(4))
        );
    }

    /// `e080_getOperandKind` — what `collectDeltasOfOperands` filed.
    #[test]
    fn operand_kind_answers_what_was_recorded() {
        let mut matched_op = MatchedOp::of(InstrPos(2));
        matched_op
            .operand_kinds
            .insert(OperandIdx(4), OperandKind::ConstValueDelta);
        assert_eq!(
            matched_op.operand_kind(OperandIdx(4)),
            OperandKind::ConstValueDelta
        );
    }

    /// `e081_deleteMatchedOps` — the attempt's matches are gone.
    #[test]
    fn delete_matched_ops_empties_the_list() {
        let mut manager = LoopRollingManager::over(
            RollingCase::L3SoftSyncWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        manager.matched_ops.push(MatchedOp::of(InstrPos(0)));
        manager.matched_ops.push(MatchedOp::of(InstrPos(1)));
        manager.delete_matched_ops();
        assert!(manager.matched_ops.is_empty());
    }

    /// `e082_updateEndOpsOfMatchedOps` — the end window is walked BACKWARDS from its end.
    #[test]
    fn update_end_ops_of_matched_ops_walks_the_end_window_backwards() {
        let mut manager = LoopRollingManager::over(
            RollingCase::L3SoftSyncWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        manager.matched_ops.push(MatchedOp::of(InstrPos(0)));
        manager.matched_ops.push(MatchedOp::of(InstrPos(1)));
        let mut end_window = Window::opening(RollingCase::L3SoftSyncWindows, InstrPos(6));
        end_window.end = InstrPos(7);
        manager.update_end_ops_of_matched_ops(&end_window);
        assert_eq!(manager.matched_ops[0].end_op, InstrPos(7));
        assert_eq!(manager.matched_ops[1].end_op, InstrPos(6));
    }

    /// `e083_computeValsIfDifferent` — two constants give their difference; a block argument facing a
    /// defined value gives nothing.
    #[test]
    fn compute_vals_if_different_reads_two_constants() {
        let block = vec![scalar_constant(Val(0), 3), scalar_constant(Val(1), 7)];
        let regions: [&[Op]; 1] = [&block];
        let defs = Definitions::from_innermost(&regions);
        assert_eq!(
            LoopRollingManager::compute_vals_if_different(Val(0), Val(1), &[], defs),
            OperandDifference::Delta(Delta(4))
        );
        assert_eq!(
            LoopRollingManager::compute_vals_if_different(Val(9), Val(0), &[], defs),
            OperandDifference::Incomputable
        );
    }

    /// `e317_checkDeltasOfOperands` — the recorded stride is confirmed operand by operand, and one
    /// wrong delta refuses the whole op.
    #[test]
    fn check_deltas_of_operands_confirms_the_recorded_stride() {
        let block = vec![
            scalar_constant(Val(0), 4),
            scalar_constant(Val(1), 10),
            add(Val(0), Val(9), Val(2)),
            add(Val(1), Val(9), Val(3)),
        ];
        let regions: [&[Op]; 1] = [&block];
        let defs = Definitions::from_innermost(&regions);
        let manager = LoopRollingManager::over(
            RollingCase::SingleInstrWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        let window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(2));
        let mut matched_op = MatchedOp::of(InstrPos(2));
        matched_op
            .operand_kinds
            .insert(OperandIdx(0), OperandKind::ConstValueDelta);
        matched_op
            .operand_kinds
            .insert(OperandIdx(1), OperandKind::NoDelta);
        matched_op.operand_deltas.insert(OperandIdx(0), Delta(6));
        assert!(manager.check_deltas_of_operands(
            &window,
            InstrPos(2),
            InstrPos(3),
            &matched_op,
            &block,
            &[],
            defs
        ));
        // The same pair against a stride of 7 is not the pair that was matched.
        matched_op.operand_deltas.insert(OperandIdx(0), Delta(7));
        assert!(!manager.check_deltas_of_operands(
            &window,
            InstrPos(2),
            InstrPos(3),
            &matched_op,
            &block,
            &[],
            defs
        ));
    }

    /// `e565_collectDeltasOfOperands` — the differing constant operand is recorded as a stride and the
    /// shared one as no delta; ⛔ a differing `$increment` refuses the whole pair.
    #[test]
    fn collect_deltas_of_operands_records_the_stride_and_refuses_an_increment() {
        let block = vec![
            scalar_constant(Val(0), 4),
            scalar_constant(Val(1), 10),
            add(Val(0), Val(9), Val(2)),
            add(Val(1), Val(9), Val(3)),
        ];
        let regions: [&[Op]; 1] = [&block];
        let defs = Definitions::from_innermost(&regions);
        let mut manager = LoopRollingManager::over(
            RollingCase::SingleInstrWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        let window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(2));
        let mut matched_op = MatchedOp::of(InstrPos(2));
        assert!(manager.collect_deltas_of_operands(
            &window,
            InstrPos(2),
            InstrPos(3),
            &mut matched_op,
            &block,
            &[],
            defs
        ));
        assert_eq!(
            matched_op.operand_kind(OperandIdx(0)),
            OperandKind::ConstValueDelta
        );
        assert_eq!(
            matched_op.operand_deltas.get(&OperandIdx(0)),
            Some(&Delta(6))
        );
        assert_eq!(matched_op.operand_kind(OperandIdx(1)), OperandKind::NoDelta);

        // A pair of transfers whose `$increment` differs, which is operand 2 of both.
        let mut second = receive_and_store();
        dialects::set_operand(&mut second, 2, Val(30));
        let transfers = vec![receive_and_store(), second];
        let regions: [&[Op]; 1] = [&transfers];
        let defs = Definitions::from_innermost(&regions);
        assert!(!manager.collect_deltas_of_operands(
            &Window::opening(RollingCase::SingleInstrWindows, InstrPos(0)),
            InstrPos(0),
            InstrPos(1),
            &mut MatchedOp::of(InstrPos(0)),
            &transfers,
            &[],
            defs
        ));
    }

    /// `e512_collectStartingValIterArgs` — the delta's start-window operand, then `zero` for the
    /// result a reader outside the window picks up; `zero` survives because it was placed.
    #[test]
    fn collect_starting_val_iter_args_takes_the_delta_operand_then_zero() {
        let block = vec![
            add(Val(10), Val(11), Val(12)),
            add(Val(13), Val(11), Val(14)),
            add(Val(14), Val(15), Val(16)),
        ];
        let mut consts = vec![scalar_constant(Val(99), 0)];
        let start_window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(0));
        let mut cur_window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(1));
        cur_window.op_to_index.insert(InstrPos(1), 0);
        let mut matched_op = MatchedOp::of(InstrPos(0));
        matched_op.end_op = InstrPos(1);
        matched_op.operand_deltas.insert(OperandIdx(0), Delta(3));
        let mut manager = LoopRollingManager::over(
            RollingCase::SingleInstrWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        manager.matched_ops.push(matched_op);
        let mut starting_vals: Vec<Val> = Vec::new();

        assert!(manager.collect_starting_val_iter_args(
            &start_window,
            &cur_window,
            &mut starting_vals,
            Val(99),
            &block,
            &mut consts,
        ));
        assert_eq!(starting_vals, vec![Val(10), Val(99)]);
        assert_eq!(
            manager.matched_ops[0]
                .iter_arg_to_res_num_and_operand
                .get(&IterArgIdx(1)),
            Some(&(ResultNum(0), None))
        );
        assert_eq!(consts.len(), 1);
    }

    /// `e512_collectStartingValIterArgs` — past the allowance nothing rolls, and the unplaced `zero`
    /// leaves the constant block.
    #[test]
    fn collect_starting_val_iter_args_refuses_past_the_allowance() {
        let block = vec![add(Val(10), Val(11), Val(12)), scalar_constant(Val(50), 0)];
        let mut consts = vec![scalar_constant(Val(99), 0)];
        let start_window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(0));
        let cur_window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(1));
        let mut manager = LoopRollingManager::over(
            RollingCase::SingleInstrWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        for _ in 0..=MAX_ITER_ARGS_CREATED {
            let mut matched_op = MatchedOp::of(InstrPos(0));
            matched_op.end_op = InstrPos(1);
            matched_op.operand_deltas.insert(OperandIdx(0), Delta(1));
            manager.matched_ops.push(matched_op);
        }
        let mut starting_vals: Vec<Val> = Vec::new();

        assert!(!manager.collect_starting_val_iter_args(
            &start_window,
            &cur_window,
            &mut starting_vals,
            Val(99),
            &block,
            &mut consts,
        ));
        assert_eq!(starting_vals.len(), MAX_ITER_ARGS_CREATED + 1);
        assert!(consts.is_empty());
    }

    /// `e084_updateBody` — the end op moves into the loop reading `iter_arg`, the yield carries
    /// `iter_arg + delta` and the op's own result, and the reader past the window reads the loop.
    #[test]
    fn update_body_moves_the_op_and_carries_its_delta() {
        let reg = sentient::Reg {
            locale: sentient::RegType::Unknown,
            index: None,
        };
        let carried_of = |init: Val, arg: Val, result: Val| sentient::Carried {
            init,
            arg,
            result,
            reg,
            program_header: false,
            element_size: None,
        };
        let mut rolled = RolledLoop {
            carried: vec![
                carried_of(Val(20), Val(21), Val(22)),
                carried_of(Val(30), Val(31), Val(32)),
            ],
            body: vec![Op::Sentient(sentient::Op::Yield {
                results: vec![Val(20), Val(30)],
            })],
            yield_at: 0,
        };
        let mut block = vec![
            add(Val(10), Val(11), Val(12)),
            add(Val(12), Val(12), Val(13)),
        ];
        let mut window = Window::opening(RollingCase::SingleInstrWindows, InstrPos(0));
        window.op_to_index.insert(InstrPos(0), 0);
        window.first_rollable_op = Some(InstrPos(0));
        let mut matched_op = MatchedOp::of(InstrPos(0));
        matched_op.operand_deltas.insert(OperandIdx(0), Delta(4));
        matched_op
            .iter_arg_to_res_num_and_operand
            .insert(IterArgIdx(1), (ResultNum(0), None));
        let mut manager = LoopRollingManager::over(
            RollingCase::SingleInstrWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(2),
        );
        manager.matched_ops.push(matched_op);
        let mut consts: Vec<Op> = Vec::new();
        let mut values = Values::default();

        manager.update_body(&window, &mut rolled, &mut block, &mut consts, &mut values);

        assert_eq!(consts, vec![scalar_constant(Val(0), 4)]);
        assert_eq!(
            rolled.body,
            vec![
                // The end op, moved, its delta operand now the iteration argument.
                add(Val(21), Val(11), Val(12)),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(1), Val(12)],
                }),
                // `iter_arg + 4`, built after the yield and moved ahead of it by `matchAndRoll`.
                add(Val(21), Val(0), Val(1)),
            ]
        );
        assert_eq!(rolled.yield_at, 1);
        // The reader outside the window now reads the loop's result.
        assert_eq!(block, vec![add(Val(32), Val(32), Val(13))]);
    }

    /// A window of one op at `at`, indexed as the reference's walk would leave it.
    fn window_at(at: InstrPos, block: &[Op]) -> Window {
        let mut window = Window::opening(RollingCase::SingleInstrWindows, at);
        window.insert_next_instr(at, block);
        window
    }

    /// e602 — three windows of one `sentient.scalar_add` each: the start window files the stride it
    /// found and moves its effective start onto the matched op, and the follower confirms the same
    /// stride. ⛔ A THIRD CONSTANT AT A DIFFERENT STRIDE REFUSES THE FOLLOWER.
    #[test]
    fn e602_files_the_start_windows_deltas_then_rechecks_them() {
        // `%3 = add %0, %9` / `%4 = add %1, %9` / `%5 = add %2, %9`, one per window.
        let compare = |third: i64| {
            let block = vec![
                scalar_constant(Val(0), 4),
                scalar_constant(Val(1), 10),
                scalar_constant(Val(2), third),
                add(Val(0), Val(9), Val(3)),
                add(Val(1), Val(9), Val(4)),
                add(Val(2), Val(9), Val(5)),
            ];
            let regions: [&[Op]; 1] = [&block];
            let defs = Definitions::from_innermost(&regions);
            let mut manager = LoopRollingManager::over(
                RollingCase::SingleInstrWindows,
                NewLoopCount(0),
                WindowIndex(0),
                WindowIndex(3),
            );
            let mut start = window_at(InstrPos(3), &block);
            let mut second = window_at(InstrPos(4), &block);
            let third_window = window_at(InstrPos(5), &block);

            let matched = manager.compare_to_next_window(
                WindowRole::Start,
                &mut start,
                &second,
                &block,
                &[],
                defs,
            );
            let followed = manager.compare_to_next_window(
                WindowRole::Follower,
                &mut second,
                &third_window,
                &block,
                &[],
                defs,
            );
            let deltas = manager.matched_ops[0].operand_deltas.clone();
            (matched, followed, deltas, start.effective_start, start.size)
        };

        let (matched, followed, deltas, effective_start, size) = compare(16);
        assert!(matched && followed);
        assert_eq!(deltas.get(&OperandIdx(0)), Some(&Delta(6)));
        // ⭐ THE FOLLOWER LEAVES THE START WINDOW'S OWN EFFECTIVE START ALONE.
        assert_eq!(effective_start, InstrPos(3));
        assert_eq!(size, WindowSize(1));

        let (matched, followed, ..) = compare(20);
        assert!(matched);
        assert!(!followed);
    }

    /// e628 — four single-instruction windows at a constant stride roll into ONE `sentient.for`: the
    /// originals are gone, the loop sits where the last window ended, the reader past it takes the
    /// loop's result, and `next` is left at the end of the window list.
    #[test]
    fn e628_rolls_four_matching_windows_into_one_loop() {
        let mut values = Values::default();
        let shared = values.mint();
        // `%c = constant 4 + 6n` / `%r = add %c, %shared`, four times, then a reader of the last.
        let steps: Vec<(Val, Val)> = (0..4).map(|_| (values.mint(), values.mint())).collect();
        let mut block: Vec<Op> = steps
            .iter()
            .enumerate()
            .map(|(n, (constant, _))| scalar_constant(*constant, 4 + 6 * n as i64))
            .collect();
        block.extend(steps.iter().map(|(constant, result)| add(*constant, shared, *result)));
        let last_result = steps[3].1;
        let tail = values.mint();
        block.push(add(last_result, last_result, tail));
        let mut windows: Vec<Window> = (4..8).map(|at| window_at(InstrPos(at), &block)).collect();
        let mut manager = LoopRollingManager::over(
            RollingCase::SingleInstrWindows,
            NewLoopCount(0),
            WindowIndex(0),
            WindowIndex(4),
        );
        let mut consts: Vec<Op> = Vec::new();
        let mut next = WindowIndex(1);

        manager.match_and_roll(
            &mut next,
            &mut windows,
            &mut block,
            &mut consts,
            &[],
            &mut values,
        );

        // Every window was consumed, and the attempt's matches were dropped after it.
        assert_eq!(next, WindowIndex(4));
        assert!(manager.matched_ops.is_empty());
        assert_eq!(manager.new_loop_count, NewLoopCount(1));
        // The four constants stay, the four adds are replaced by the loop, the reader follows it.
        assert_eq!(block.len(), 6);
        let Op::Sentient(sentient::Op::For {
            bound,
            carried,
            dbg_name,
            body,
            ..
        }) = &block[4]
        else {
            panic!("the rolled loop sits where the last window ended: {:?}", block)
        };
        assert_eq!(dbg_name.as_deref(), Some("LR loop #1"));
        // One iteration argument for the stride, one for the result read past the last window.
        assert_eq!(carried.len(), 2);
        assert_eq!(carried[0].init, steps[0].0);
        // The bound is `count`, and the stride constant is the one e084 built.
        assert!(consts.contains(&scalar_constant(*bound, 4)));
        assert!(consts.iter().any(|op| matches!(
            op,
            Op::Sentient(sentient::Op::ScalarConstant { value: 6, .. })
        )));
        // The moved op, the `iter_arg + delta` it yields, then the yield.
        assert_eq!(body.len(), 3);
        assert!(matches!(
            body[2],
            Op::Sentient(sentient::Op::Yield { .. })
        ));
        // The reader past the last window now reads the loop's own result.
        assert_eq!(
            block[5],
            add(carried[1].result, carried[1].result, tail)
        );
    }

    /// e642 — the driver over one block: it collects the windows itself, greedily folding the leading
    /// constants into the first, and rolls the two runs of four matching adds into two `sentient.for`s.
    /// ⛔ THE SECOND ROLL IS WHAT PROVES THE REMAP: its windows name ops by ordinals that the first
    /// roll shortened the block under.
    #[test]
    fn e642_collects_the_windows_and_rolls_both_runs() {
        let mut values = Values::default();
        let shared = values.mint();
        // `%c = constant 4 + 6n` four times and `%c = constant 100 + 5n` four times, then one
        // `%r = add %c, %shared` per constant: two runs a single delta apart, at different deltas.
        let runs: Vec<Vec<(Val, Val)>> = (0..2)
            .map(|_| (0..4).map(|_| (values.mint(), values.mint())).collect())
            .collect();
        let mut block: Vec<Op> = Vec::new();
        for (run, (base, stride)) in runs.iter().zip([(4, 6), (100, 5)]) {
            for (n, (constant, _)) in run.iter().enumerate() {
                block.push(scalar_constant(*constant, base + stride * n as i64));
            }
        }
        for run in &runs {
            for (constant, result) in run {
                block.push(add(*constant, shared, *result));
            }
        }
        let mut consts: Vec<Op> = Vec::new();
        let mut new_loop_count = NewLoopCount(0);

        roll_instrs_in_block(
            RollingCase::SingleInstrWindows,
            &mut block,
            &mut consts,
            &[],
            &mut new_loop_count,
            &mut values,
        );

        assert_eq!(new_loop_count, NewLoopCount(2));
        // The eight constants stand; the eight adds are gone into the two loops that took their place.
        assert_eq!(block.len(), 10);
        let names: Vec<&str> = block
            .iter()
            .filter_map(|op| match op {
                Op::Sentient(sentient::Op::For { dbg_name, .. }) => dbg_name.as_deref(),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["LR loop #1", "LR loop #2"]);
    }

    /// A model and a rung, so the program is typed; nothing here reads either.
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// e652 — the pass entry through the invocation that reaches the single-instruction arm, which is
    /// the PIPELINE's: an exclude list naming the two L3 halves leaves `L3_rolling_` false, the LXLU
    /// unit it does not name is admitted, and the four adds inside the unit's one innermost loop are
    /// rolled into a loop NESTED IN THAT LOOP'S BODY rather than beside it.
    #[test]
    fn e652_rolls_the_single_instructions_of_each_innermost_loop() {
        let mut values = Values::default();
        let (unit_val, iv, bound, shared) =
            (values.mint(), values.mint(), values.mint(), values.mint());
        let pairs: Vec<(Val, Val)> = (0..4).map(|_| (values.mint(), values.mint())).collect();
        // `%c = constant 4 + 6n` four times then one `%r = add %c, %shared` per constant, all inside
        // the loop: e642's own single-instruction case, moved one region in.
        let mut inner: Vec<Op> = pairs
            .iter()
            .enumerate()
            .map(|(n, (constant, _))| scalar_constant(*constant, 4 + 6 * n as i64))
            .collect();
        inner.extend(
            pairs
                .iter()
                .map(|(constant, result)| add(*constant, shared, *result)),
        );
        inner.push(Op::Sentient(sentient::Op::Yield {
            results: Vec::new(),
        }));
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, unit_val),
                    precision: None,
                    body: vec![Op::Sentient(sentient::Op::For {
                        iv,
                        bound,
                        bound_reg: None,
                        carried: Vec::new(),
                        dbg_name: None,
                        body: inner,
                    })],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };

        run_on_operation(
            &mut program,
            &UnitFilter::Exclude(vec![DfirUnit::L3su, DfirUnit::L3lu]),
            &mut values,
        );

        let unit = program.units.iter().next().expect("the one unit");
        let Op::Sentient(sentient::Op::For { body, .. }) = &unit.body[0] else {
            panic!("the loop the tree found is still the unit's only op: {:?}", unit.body)
        };
        let names: Vec<&str> = body
            .iter()
            .filter_map(|op| match op {
                Op::Sentient(sentient::Op::For { dbg_name, .. }) => dbg_name.as_deref(),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["LR loop #1"]);
    }
}
