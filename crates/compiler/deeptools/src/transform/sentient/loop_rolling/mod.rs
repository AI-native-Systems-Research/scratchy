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


// ── STILL SCHEDULED IN THIS FILE (levels 1..8) — anchors, not dead comments. ⛔ Do not delete one
// you did not port; on bridge 2 that silently lost 149 of 384 functions.

// crustify:todo: e317_checkDeltasOfOperands
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:505  (79 body lines, level 1)
//   original  : bool checkDeltasOfOperands(Window *cur_window, Operation &op_a, Operation &op_b, MatchedOp *matched_op)
//   calls     : e080_getOperandKind, e083_computeValsIfDifferent

// crustify:todo: e509_insertNextInstr
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:114  (15 body lines, level 3)
//   original  : void insertNextInstr(Block::iterator instr)
//   calls     : e422_insert

// crustify:todo: e510_insertOperandKind
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:183  (3 body lines, level 3)
//   original  : void insertOperandKind(unsigned i, OperandKind kind)
//   calls     : e422_insert

// crustify:todo: e511_insertOperandDelta
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:186  (3 body lines, level 3)
//   original  : void insertOperandDelta(unsigned i, int delta)
//   calls     : e422_insert

// crustify:todo: e512_collectStartingValIterArgs
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:592  (86 body lines, level 3)
//   original  : bool collectStartingValIterArgs(Window *start_window, Window *cur_window, SmallVector<Value, 2> &starting_vals, Value &zero)
//   calls     : e252_size, e422_insert

// crustify:todo: e565_collectDeltasOfOperands
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:392  (109 body lines, level 4)
//   original  : bool collectDeltasOfOperands(Window *start_window, Operation &op_a, Operation &op_b, MatchedOp *matched_op)
//   calls     : e077_isIncrementField, e083_computeValsIfDifferent, e510_insertOperandKind, e511_insertOperandDelta

// crustify:todo: e602_compareToNextWindow
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:321  (67 body lines, level 5)
//   original  : bool compareToNextWindow(bool is_start_window, Window *cur_window, Window *next_window)
//   calls     : e078_setEffectiveStart, e079_checkOpUsage, e252_size, e317_checkDeltasOfOperands, e565_collectDeltasOfOperands

// crustify:todo: e628_matchAndRoll
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:758  (104 body lines, level 6)
//   original  : void matchAndRoll(std::vector<Window *>::iterator &next)
//   calls     : e081_deleteMatchedOps, e082_updateEndOpsOfMatchedOps, e084_updateBody, e252_size, e512_collectStartingValIterArgs, e602_compareToNextWindow

// crustify:todo: e642_rollInstrsInBlock
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:886  (77 body lines, level 7)
//   original  : void rollInstrsInBlock(dcc::OperationEquivalence &oe, dcc::CommonPassOptions &opts, OpBuilder &const_builder, Block &bb, bool is_L3_case)
//   calls     : e509_insertNextInstr, e628_matchAndRoll

// crustify:todo: e652_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopRolling.cpp:966  (53 body lines, level 8)
//   original  : void runOnOperation()
//   calls     : e252_size, e642_rollInstrsInBlock


// ⛔ THE REMOVAL TRIGGER IS `e652_runOnOperation`. Every type and method below is reached only from
// the pass entry, which is this file's last unfilled anchor (level 8): until it lands, the whole
// module is unreachable from the crate and `dead_code` would fire on all of it. ⭐ DELETE THIS LINE
// WHEN THAT ANCHOR IS FILLED — a warning that survives it is a unit nothing calls, which the
// campaign's own note names as the failure mode to catch.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, Val, affine, sentient, symbol, uniform,
};

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
/// (`LoopRolling.cpp:162`).
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

/// ONE WINDOW OF INSTRUCTIONS — the reference's `Window` (`LoopRolling.cpp:83`).
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

/// ONE SET OF MATCHED OPS ACROSS THE WINDOWS TO BE ROLLED — the reference's `MatchedOp` (`:158`).
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
    /// ONE MATCH, both instances starting as the same op — the reference's constructor (`:180`).
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
/// `use_equiv_classes = true` (`LoopRolling.cpp:970-972`) — so every block argument matches every
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

/// ONE OP WITH EVERY VALUE AND EVERY REGION BLANKED — what is left is its NAME AND ATTRIBUTES.
///
/// ⛔ NOT AN ANCHORED UNIT — the structural half of [`ops_are_equivalent`], which is
/// `OperationEquivalence`'s own first test.
fn skeleton(op: &Op) -> Op {
    let mut bare = op.clone();
    match &mut bare {
        Op::Sentient(inner) => {
            for val in sentient::operands_mut(inner) {
                *val = Val(0);
            }
            for val in sentient::results_mut(inner) {
                *val = Val(0);
            }
            for val in sentient::block_args_mut(inner) {
                *val = Val(0);
            }
            for region in sentient::regions_mut(inner) {
                region.clear();
            }
        }
        Op::AffineFor(loop_op) => {
            loop_op.iv = Val(0);
            for bound in [&mut loop_op.lo, &mut loop_op.hi] {
                if let affine::Bound::Val(bound) = bound {
                    *bound = Val(0);
                }
            }
            for carried in &mut loop_op.carried {
                carried.init = Val(0);
                carried.arg = Val(0);
                carried.result = Val(0);
            }
            loop_op.body.clear();
        }
        other => {
            if let Some(mut inner) = dialects::lowered(other) {
                for (_role, val) in lower::vals_mut(&mut inner) {
                    *val = Val(0);
                }
                for region in lower::regions_mut(&mut inner) {
                    region.clear();
                }
                *other = dialects::raised(inner);
            }
        }
    }
    bare
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
             (Dialect/Uniform/Utils.cpp:402): {other:?}"
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
             (Dialect/Uniform/Utils.cpp:404)"
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
    })
}

/// `replaceUsesWithIf(.., owner != yield_op && !cur_window->isOpInThisWindow(owner))`.
///
/// ⛔ NOT AN ANCHORED UNIT — the predicate half of `updateBody`'s rewrite.
///
/// ⭐ THE `yield_op` HALF NEEDS NO TEST: the yield lives in [`RolledLoop::body`], which this walk
/// never enters. The reference has to name it because it built the yield into `bb_`'s new loop
/// before the windows were re-examined, and its own comment says so (`LoopRolling.cpp:733-735`).
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
/// ⚠️ TRAP: the positions are `SentientOps.td`'s (`:507-509`, `:550-552`, `:719-723`) and this
/// island's operand list matches them only up to slot 2 — it models `$consumer`/`$producer` as a wire
/// end, so `load_and_send` and `receive_and_store` are one slot short from position 3 on.
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

/// THE `sentient.for` BEING BUILT — what `matchAndRoll` creates and [`LoopRollingManager::update_body`]
/// fills.
///
/// ⛔⛔ NOT YET AN OP OF THE BLOCK, AND THAT IS DELIBERATE. The reference inserts the `ForOp` into
/// `bb_` and then moves ops out of `bb_` INTO it (`:761-856`), which here would be one `&mut` into a
/// vector reaching another element of the same vector. Building the loop beside the block and
/// inserting it once, after `cur_window`'s end, is the same IR.
#[derive(Debug)]
pub(crate) struct RolledLoop {
    /// The loop's `iter_args`: `getRegionIterArgs()[i]` is `carried[i].arg` and `getResult(i)` is
    /// `carried[i].result`.
    pub(crate) carried: Vec<sentient::Carried>,
    /// The body.
    pub(crate) body: Vec<Op>,
    /// Where the `sentient.yield` sits in [`Self::body`] — `matchAndRoll` creates it FIRST and moves
    /// it to the end only after `updateBody` has run (`:846`, `:856`), which is what puts the created
    /// adds after the moved ops.
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
/// `all_block_args_are_equiv = true` and `do_recursive_compare = true` (`LoopRolling.cpp:970-972`),
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
    /// `matched_ops_` — ⛔ POPULATED IN REVERSE ORDER (`:236`), which is why
    /// [`Self::update_end_ops_of_matched_ops`] walks it forwards against a backwards block walk and
    /// [`Self::update_body`] walks it backwards.
    pub(crate) matched_ops: Vec<MatchedOp>,
}

impl LoopRollingManager {
    /// A MANAGER FOR ONE ATTEMPT — the reference's constructor (`:219-231`).
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

    /// Replaces: e084_updateBody
    ///
    /// Moves each match's end instance into the loop, re-points its delta operands at the matching
    /// `iter_arg`, yields `iter_arg + delta` for each, and hands every reader past the last rolled
    /// window the loop's own result instead.
    ///
    /// ⛔ THE CONSTANTS GO TO `consts`, NOT THE LOOP BODY: `const_builder_` is anchored at the start
    /// of the block CONTAINING the program unit (`:967-968`), which is a different block from `bb_`.
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;

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
}
