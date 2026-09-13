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

//! `CFGSimplificationSentientLevel.cpp` — 20 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e031_populateTable` | 031 | 0 | 90 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:924` |
//! | `e032_findExistingIterArg` | 032 | 0 | 78 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1455` |
//! | `e033_replaceResultByIterArg` | 033 | 0 | 8 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1541` |
//! | `e034_replaceIfOpByIterArg` | 034 | 0 | 8 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1552` |
//! | `e035_replacePredicateIVByIterArg` | 035 | 0 | 8 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1563` |
//! | `e036_checkCascadingArgUses` | 036 | 0 | 10 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1576` |
//! | `e289_computeTemplateSeqAndUpdateMonotoneSeq` | 289 | 1 | 75 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2096` |
//! | `e290_insertSequence` | 290 | 1 | 24 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2360` |
//! | `e291_padTableSlice` | 291 | 1 | 49 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2398` |
//! | `e292_generateEncodingTypes` | 292 | 1 | 26 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2450` |
//! | `e293_generateEncodingTuple` | 293 | 1 | 52 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2485` |
//! | `e294_codeGenMonotoneTableSlice` | 294 | 1 | 137 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2654` |
//! | `e295_codeGenGenericTableSlice` | 295 | 1 | 58 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2802` |
//! | `e431_processLeaf` | 431 | 2 | 64 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1023` |
//! | `e432_transformInContiguousSequenceCase` | 432 | 2 | 321 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1597` |
//! | `e433_transformInMonotoneSequenceCase` | 433 | 2 | 91 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1923` |
//! | `e434_parseSequence` | 434 | 2 | 176 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2177` |
//! | `e435_codeGenGenericNestedIf` | 435 | 2 | 71 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2556` |
//! | `e496_parseFixedDims` | 496 | 3 | 59 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2021` |
//! | `e555_findPatternsAndSimplify` | 555 | 4 | 340 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1101` |

use super::{
    Builders, ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS, Entries, EvaluatedValueId,
    ExpressionEvaluator, IndexStride, IntervalMarker, IntervalStride, Leaf, Sequence, SequenceKind,
    Table, TableIndex, TableSlice,
};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{self as ir, Op, Val, affine, arith, sentient};
use core::num::NonZeroI64;
use std::collections::{BTreeMap, BTreeSet};

/// WHERE AN OP SITS IN THE NESTED REGIONS — one `(region, position)` step per level, outermost
/// first. The stand-in for `mlir::Operation *`.
///
/// ⭐ TWO NUMBERS PER LEVEL, not one: `sentient.if` has TWO regions (`then_body`, `else_body`), so a
/// path of bare ordinals could not tell a then-body op from an else-body one.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpPath(Vec<(u32, u32)>);

impl OpPath {
    /// The op those steps name, relative to the root body.
    #[must_use]
    pub fn at(path: &[(u32, u32)]) -> Self {
        Self(path.to_vec())
    }

    /// The steps, outermost first.
    #[must_use]
    pub fn path(&self) -> &[(u32, u32)] {
        &self.0
    }

    /// The op at position `index` of this op's region `region`.
    #[must_use]
    pub fn child(&self, region: u32, index: u32) -> Self {
        let mut steps = self.0.clone();
        steps.push((region, index));
        Self(steps)
    }

    /// The op that encloses this one, `None` at the top level — `Operation::getParentOp`.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        let mut steps = self.0.clone();
        steps.pop();
        (!steps.is_empty()).then_some(Self(steps))
    }
}

/// THE PASS'S OWN OP MARKERS (`dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:42-50`)
/// — pass-local, and deliberately NOT island attributes: each exists only to name an op again later
/// in the same pass, which an [`OpPath`] already does. Bridge 2's `e285_replaceIfOpByIterArg` reached
/// the same conclusion for the same marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mark {
    /// `DEAD_THEN_BRANCH` — "dead-then-branch".
    DeadThenBranch,
    /// `DEAD_ELSE_BRANCH` — "dead-else-branch".
    DeadElseBranch,
    /// `IF_OP_TO_BE_REPLACED_BY_ITER_ARG` — "if-op-replaced-by-iter-arg".
    IfOpReplacedByIterArg,
    /// `RESULT_TO_BE_REPLACED_BY_ITER_ARG` — "result-replaced-by-iter-arg".
    ResultReplacedByIterArg,
    /// `LHS_IN_PRED_TO_BE_REPLACED_BY_ITER_ARG` — "lhs-replaced-by-iter-arg".
    LhsInPredReplacedByIterArg,
    /// `TO_DELETE` — "to-delete".
    ToDelete,
    /// `PREVIOUS_FOR_OP_CREATED` — "previous-for-op-created".
    PreviousForOpCreated,
}

/// Which ops carry which [`Mark`] — `setAttr`, `hasAttr` and `removeAttr` over paths.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Marks {
    on: BTreeSet<(Mark, OpPath)>,
}

impl Marks {
    /// `op->setAttr(<mark>, ..)`.
    pub fn set(&mut self, mark: Mark, at: &[(u32, u32)]) {
        self.on.insert((mark, OpPath::at(at)));
    }

    /// `op->hasAttr(<mark>)`.
    #[must_use]
    pub fn has(&self, mark: Mark, at: &[(u32, u32)]) -> bool {
        self.on.contains(&(mark, OpPath::at(at)))
    }

    /// `hasAttr` then `removeAttr` in one call — the order and the pairing the rewriters below use.
    pub fn take(&mut self, mark: Mark, at: &[(u32, u32)]) -> bool {
        self.on.remove(&(mark, OpPath::at(at)))
    }
}

/// One entry of `lhs_to_for_op_or_null_` (`:540-543`): the loop a predicate's LHS is the IV of, with
/// the four bounds `getOrCreateTupleForIV` records
/// (`Analyses/CFGSSentientLevelConditionalTree.hpp:363-368`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopInfo {
    /// `std::get<0>` — the `sentient.for`, or the null the map's own name allows.
    pub for_op: Option<OpPath>,
    /// `std::get<1>` — the lower bound.
    pub lb: i64,
    /// `std::get<2>` — the upper bound.
    pub ub: i64,
    /// `std::get<3>` — ⭐ NON-ZERO AS A TYPE, which is what `DT_CHECK_MSG(step != 0, ..)` (`:933`)
    /// asks for at runtime.
    pub step: NonZeroI64,
    /// `std::get<4>` — the number of iterations.
    pub iterations: i64,
}

/// One entry of `ivs_dimensions_multipliers_` (`:550-551`): an IV, the size of the table dimension it
/// spans and the multiplier that dimension contributes (`:780-830`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvDim {
    /// `std::get<0>` — the induction variable.
    pub iv: Val,
    /// `std::get<1>` — this dimension's extent.
    pub dimension: i64,
    /// `std::get<2>` — this dimension's multiplier into the flattened table.
    pub multiplier: i64,
}

/// A TABLE INDEX THAT IS IN RANGE BY CONSTRUCTION — `is_idx_valid` (`:936`) as a type, so a filter
/// bit cannot be addressed outside the array the IV's iteration count sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableIdx(usize);

impl TableIdx {
    /// `(rhs_val - lb) / step`, `None` when it falls outside `[0, num_iterations)`.
    #[must_use]
    pub fn of(rhs_val: i64, loop_info: &LoopInfo) -> Option<Self> {
        let idx = (rhs_val - loop_info.lb) / loop_info.step.get();
        if idx >= loop_info.iterations {
            return None;
        }
        usize::try_from(idx).ok().map(Self)
    }
}

/// `ivs_to_values_and_filters` (`:780-830`): per IV, the value an ancestor predicate fixed and one
/// filter bit per iteration, initialised to `(std::nullopt, an array of FALSE)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IvValuesAndFilters {
    entries: Vec<IvEntry>,
}

/// One IV's `(std::optional<WidestIntType>, bool *)` pair.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IvEntry {
    iv: Val,
    chosen: Option<i64>,
    filter: Vec<bool>,
}

impl IvValuesAndFilters {
    /// One entry per IV, each with that IV's `num_iterations` filter bits — the reference's
    /// `new bool[num_iterations]` (`:805`).
    #[must_use]
    pub fn of(ivs: impl IntoIterator<Item = (Val, i64)>) -> Self {
        let mut table = Self::default();
        for (iv, iterations) in ivs {
            table.entry(iv, iterations);
        }
        table
    }

    /// The value an ancestor predicate fixed this IV to — `.first`.
    #[must_use]
    pub fn chosen(&self, iv: Val) -> Option<i64> {
        self.entries
            .iter()
            .find(|entry| entry.iv == iv)
            .and_then(|entry| entry.chosen)
    }

    /// `it->first` at position `at` — the map iterator [`LeafSink::process_leaf`] advances, whose
    /// `end()` is `None`.
    #[must_use]
    pub fn iv_at(&self, at: usize) -> Option<Val> {
        self.entries.get(at).map(|entry| entry.iv)
    }

    /// `.second[iteration]`, READ-ONLY: an IV with no entry, or an iteration past the array its trip
    /// count sized, reads the `false` the reference's `new bool[num_iterations]` holds there.
    #[must_use]
    pub fn filter_bit(&self, iv: Val, iteration: i64) -> bool {
        self.entries
            .iter()
            .find(|entry| entry.iv == iv)
            .and_then(|entry| {
                usize::try_from(iteration)
                    .ok()
                    .and_then(|at| entry.filter.get(at))
            })
            .copied()
            .unwrap_or_default()
    }

    /// `.first = value`.
    pub fn set_chosen(&mut self, iv: Val, value: Option<i64>, iterations: i64) {
        self.entry(iv, iterations).chosen = value;
    }

    /// `.second[idx]`.
    pub fn filter(&mut self, iv: Val, at: TableIdx, iterations: i64) -> bool {
        self.entry(iv, iterations)
            .filter
            .get(at.0)
            .copied()
            .unwrap_or_default()
    }

    /// `.second[idx] = on`, answering what was there before.
    pub fn set_filter(&mut self, iv: Val, at: TableIdx, on: bool, iterations: i64) -> bool {
        let entry = self.entry(iv, iterations);
        match entry.filter.get_mut(at.0) {
            Some(bit) => {
                let previous = *bit;
                *bit = on;
                previous
            }
            None => false,
        }
    }

    /// `ivs_to_values_and_filters[iv]` — `std::map::operator[]`, which inserts the default.
    fn entry(&mut self, iv: Val, iterations: i64) -> &mut IvEntry {
        if let Some(position) = self.entries.iter().position(|entry| entry.iv == iv) {
            return &mut self.entries[position];
        }
        let position = self.entries.len();
        self.entries.push(IvEntry {
            iv,
            chosen: None,
            filter: vec![false; usize::try_from(iterations).unwrap_or_default()],
        });
        &mut self.entries[position]
    }
}

/// One `dcc::CFGSCondNode` (`Analyses/CFGSSentientLevelConditionalTree.hpp:31`), as much of it as
/// these units read.
///
/// ⛔ `Analyses/` IS OUT OF CAMPAIGN SCOPE: this is the projection its caller fills, not a port of
/// the tree — nothing here computes the tree's own answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondNode {
    /// `getLhs()` — the predicate's left operand, an IV.
    pub lhs: Val,
    /// `getRhsVal()` — the constant the predicate compares it against.
    pub rhs_val: i64,
    /// `getOperation()` — the `sentient.if` this node describes.
    pub op: OpPath,
    /// `getThenNode()`.
    pub then_node: Branch,
    /// `getElseNode()`.
    pub else_node: Branch,
}

/// One `CFGSCondThenNode`/`CFGSCondElseNode`: at most one child in a candidate subtree, so
/// `isLeaf()` IS `child.is_none()` — the header's own note (`:28-30`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Branch {
    /// `isDead()`, and `setAsDead()` sets it.
    pub dead: bool,
    /// `getFirstChild()`.
    pub child: Option<Box<CondNode>>,
    /// `getResults()` and `getBlock()` — what [`LeafSink::process_leaf`] hands to
    /// [`Table::create_table_entry_at_idx`] when this branch IS a leaf.
    pub leaf: Leaf,
}

/// `PatternSimplificationManager` (`:537-695`) — the members the units in this file read.
///
/// ⛔ `cur_root_` (`:594`) IS NOT DECLARED: no unit ported so far reads it, and every use of it is
/// the `getLoc()` of a builder call — SentientIR carries no `Loc`, so there is nothing for the island
/// to hold.
#[derive(Debug, Clone)]
pub struct PatternSimplificationManager {
    /// `lhs_to_for_op_or_null_`.
    pub lhs_to_for_op_or_null: BTreeMap<Val, LoopInfo>,
    /// `ivs_dimensions_multipliers_`.
    pub ivs_dimensions_multipliers: Vec<IvDim>,
    /// `table_` (`:553`) — `None` is the `nullptr` it starts as.
    pub table: Option<Table>,
    /// `template_sequences_` (`:578`) — OWNING (`std::vector<Sequence *>` deleted in the destructor),
    /// so the template monotone sequence is named by INDEX rather than by the reference's raw pointer
    /// into this vector.
    pub template_sequences: Vec<Sequence>,
    /// `table_follows_contiguous_pattern_` (`:559`) — ⭐ STARTS TRUE.
    pub table_follows_contiguous_pattern: bool,
    /// `table_slices_follow_montone_pattern_` (`:563`) — ⭐ STARTS TRUE, and the reference's own
    /// spelling of "montone" is kept so that a grep for the member lands here.
    pub table_slices_follow_montone_pattern: bool,
    /// `table_slices_are_consistent_` (`:570`) — ⭐ STARTS TRUE, and only
    /// [`PatternSimplificationManager::parse_fixed_dims`] ever clears it.
    pub table_slices_are_consistent: bool,
    /// `abort_pattern_` (`:591`).
    pub abort_pattern: bool,
    /// `monotone_seq_start_val_` (`:583`).
    pub monotone_seq_start_val: Option<Val>,
    /// `monotone_seq_int_step_` (`:586`) — ⭐ THE ONE MEMBER THE REFERENCE LEAVES UNINITIALISED, so
    /// `None` is a state it cannot distinguish and this one can.
    pub monotone_seq_int_step: Option<EvaluatedValueId>,
    /// `monotone_seq_val_step_` (`:587`).
    pub monotone_seq_val_step: Option<Val>,
    /// The pass-local op markers of `:42-50`.
    pub marks: Marks,
    /// `int &encoding_if_op_count_` (`:600`) — ⭐ A REFERENCE to the PASS's own counter (`:710`), so
    /// it outlives one manager and names every encoding conditional the whole pass generates.
    pub encoding_if_op_count: EncodingIfOpCount,
}

/// How many encoding conditionals the pass has named — `encoding_if_op_count_` (`:710`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodingIfOpCount(pub u32);

/// ⭐ THREE FLAGS START TRUE (`:559`, `:563`, `:570`) — a derived `Default` would start the pass
/// having already failed to match every pattern, and nothing but a mismatch ever clears one.
impl Default for PatternSimplificationManager {
    fn default() -> PatternSimplificationManager {
        PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::new(),
            ivs_dimensions_multipliers: Vec::new(),
            table: None,
            template_sequences: Vec::new(),
            table_follows_contiguous_pattern: true,
            table_slices_follow_montone_pattern: true,
            table_slices_are_consistent: true,
            abort_pattern: false,
            monotone_seq_start_val: None,
            monotone_seq_int_step: None,
            monotone_seq_val_step: None,
            marks: Marks::default(),
            encoding_if_op_count: EncodingIfOpCount(0),
        }
    }
}

/// The iter arg [`PatternSimplificationManager::find_existing_iter_arg`] is looking for.
#[derive(Debug, Clone, Copy)]
pub struct IterArgTarget {
    /// `target_lb`.
    pub lb: Val,
    /// `target_step`; `None` is the case where `monotone_seq_val_step_` stands in for it (`:1522-1523`).
    ///
    /// ⭐ AN `EvaluatedValueId`, NOT A `Val`: every caller's target step comes from the evaluator —
    /// `getConstant(table_step)` (`:1638`) or `*monotone_seq_int_step_` (`:1650`).
    pub step: Option<EvaluatedValueId>,
    /// `type`.
    pub ty: ScalarTy,
}

/// `*step_as_ev == evaluator_.evaluateMultiplyByConst(*target_step, multiplier)` (`:1519-1520`).
fn step_matches_target(step: Val, target_step: EvaluatedValueId, multiplier: i64) -> bool {
    let _ = (step, target_step, multiplier);
    todo!(
        "ExpressionEvaluator::evaluateValue / evaluateMultiplyByConst — out of campaign scope \
         (Analyses/ExpressionEvaluatorUtils)"
    )
}

/// `Operation::walk`, pre-order, giving each op its absolute [`OpPath`] steps.
///
/// ⛔ THE DESCENT IS THE ISLAND'S OWN — `sentient` regions and an `affine.for` body, exactly what
/// `ir::defining_op` and `ir::replace_all_uses_with` descend.
fn walk(
    body: &[Op],
    at: &mut Vec<(u32, u32)>,
    region: u32,
    visit: &mut impl FnMut(&[(u32, u32)], &Op),
) {
    for (index, op) in body.iter().enumerate() {
        at.push((region, index as u32));
        visit(at, op);
        match op {
            Op::Sentient(inner) => {
                for (sub_region, sub) in sentient::regions(inner).into_iter().enumerate() {
                    walk(sub, at, sub_region as u32, visit);
                }
            }
            Op::AffineFor(loop_op) => walk(&loop_op.body, at, 0, visit),
            Op::UniformRegions(regions) => {
                for (sub_region, region) in regions.regions().iter().enumerate() {
                    walk(&region.body, at, sub_region as u32, visit);
                }
            }
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
        at.pop();
    }
}

/// [`walk`], with the ops handed over mutably.
fn walk_mut(
    body: &mut [Op],
    at: &mut Vec<(u32, u32)>,
    region: u32,
    visit: &mut impl FnMut(&[(u32, u32)], &mut Op),
) {
    for (index, op) in body.iter_mut().enumerate() {
        at.push((region, index as u32));
        visit(at, op);
        match op {
            Op::Sentient(inner) => {
                for (sub_region, sub) in sentient::regions_mut(inner).into_iter().enumerate() {
                    walk_mut(sub, at, sub_region as u32, visit);
                }
            }
            Op::AffineFor(loop_op) => walk_mut(&mut loop_op.body, at, 0, visit),
            Op::UniformRegions(regions) => {
                for (sub_region, region) in regions.regions_mut().iter_mut().enumerate() {
                    walk_mut(&mut region.body, at, sub_region as u32, visit);
                }
            }
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
        at.pop();
    }
}

/// [`walk_mut`] restricted to one op and its subtree — `for_op->walk(..)`, which visits `for_op`
/// itself too.
fn walk_mut_under(root: &mut [Op], base: &OpPath, visit: &mut impl FnMut(&[(u32, u32)], &mut Op)) {
    let prefix = base.path().to_vec();
    walk_mut(root, &mut Vec::new(), 0, &mut |at, op| {
        if at.starts_with(&prefix) {
            visit(at, op);
        }
    });
}

/// The op those steps name, if they still name one.
fn op_at<'a>(root: &'a [Op], path: &[(u32, u32)]) -> Option<&'a Op> {
    let (&(_, index), rest) = path.split_first()?;
    let op = root.get(index as usize)?;
    if rest.is_empty() {
        return Some(op);
    }
    let (region, _) = rest[0];
    match op {
        Op::Sentient(inner) => op_at(
            sentient::regions(inner).into_iter().nth(region as usize)?,
            rest,
        ),
        Op::AffineFor(loop_op) => op_at(&loop_op.body, rest),
        // The path [`walk`] hands out numbers a local region, so this resolves one.
        Op::UniformRegions(regions) => op_at(&regions.regions().get(region as usize)?.body, rest),
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => None,
    }
}

/// [`op_at`] FOR A REWRITE — the same descent over a mutable tree, which e432's cleanup needs to set
/// an already-built loop's `init` and its terminator's operands.
fn op_at_mut<'a>(root: &'a mut [Op], path: &[(u32, u32)]) -> Option<&'a mut Op> {
    let (&(_, index), rest) = path.split_first()?;
    let op = root.get_mut(index as usize)?;
    if rest.is_empty() {
        return Some(op);
    }
    let (region, _) = rest[0];
    match op {
        Op::Sentient(inner) => op_at_mut(
            sentient::regions_mut(inner)
                .into_iter()
                .nth(region as usize)?,
            rest,
        ),
        Op::AffineFor(loop_op) => op_at_mut(&mut loop_op.body, rest),
        Op::UniformRegions(regions) => op_at_mut(
            &mut regions.regions_mut().get_mut(region as usize)?.body,
            rest,
        ),
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => None,
    }
}

/// One op's operands, read-only — `sentient::operands_mut` is the island's only accessor, so a clone
/// answers it.
fn operands_of(op: &Op) -> Vec<Val> {
    match op {
        Op::Sentient(inner) => {
            let mut copy = inner.clone();
            sentient::operands_mut(&mut copy)
                .into_iter()
                .map(|operand| *operand)
                .collect()
        }
        Op::AffineFor(loop_op) => {
            let mut operands: Vec<Val> = loop_op.carried.iter().map(|entry| entry.init).collect();
            for bound in [&loop_op.lo, &loop_op.hi] {
                if let affine::Bound::Val(val) = bound {
                    operands.push(*val);
                }
            }
            operands
        }
        other => ir::lowered(other)
            .map(|lower| crate::islands::dataflow_ir::dialects::operands(&lower))
            .unwrap_or_default(),
    }
}

/// Every op that reads `val`, by path — `Value::getUses`.
fn uses_of(val: Val, root: &[Op]) -> Vec<OpPath> {
    let mut found = Vec::new();
    walk(root, &mut Vec::new(), 0, &mut |at, op| {
        if operands_of(op).contains(&val) {
            found.push(OpPath::at(at));
        }
    });
    found
}

/// `Operation::use_empty` — no result of this op is read anywhere.
fn use_empty(op: &Op, root: &[Op]) -> bool {
    ir::results(op)
        .into_iter()
        .all(|result| uses_of(result, root).is_empty())
}

/// The path of the op that binds `val` — `Value::getDefiningOp`, which is `None` for a region
/// argument.
fn defining_path(val: Val, root: &[Op]) -> Option<OpPath> {
    let mut found = None;
    walk(root, &mut Vec::new(), 0, &mut |at, op| {
        if found.is_none() && ir::results(op).contains(&val) {
            found = Some(OpPath::at(at));
        }
    });
    found
}

/// The literal a value is, if it is one — what `dcc::utils::isConstant<sentient::ConstantOp>` answers
/// and what `isTargetConstant` compares (`Analyses/Utils.cpp:156`).
///
/// ⛔ The `uniform.query_map` arm of those two needs `getListOfValueOpsFromUniformMapping`, which no
/// unit of this batch reaches.
fn const_value(val: Val, root: &[Op]) -> Option<i64> {
    match ir::defining_op(val, root)? {
        Op::Sentient(inner) => match inner {
            sentient::Op::ScalarConstant { value, .. } => Some(*value),
            _ => None,
        },
        Op::Arith(inner) => match inner {
            arith::Op::Constant { value, .. } => Some(*value),
            arith::Op::ConstantInt { value, .. } => Some(match value {
                arith::IntConst::Bool(bit) => i64::from(*bit),
                arith::IntConst::Int { value, .. } => *value,
            }),
            _ => None,
        },
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::AffineFor(_)
        | Op::Vector(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_)
        | Op::UniformRegions(_) => None,
    }
}

/// `dcc::utils::isSameConstant` (`Analyses/Utils.cpp:185`) — a constant on one side and the same
/// literal on the other; a region argument on either side is never one.
fn is_same_constant(val: Val, target: Val, root: &[Op]) -> bool {
    match (const_value(val, root), const_value(target, root)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

/// The `init` a loop carries at the position that binds `val` as its region argument or its result —
/// how a `sentient.for`'s interface answers `getType()` (`dcc/src/Dialect/Sentient/SentientOps.cpp:951`).
fn carried_init_of(val: Val, root: &[Op]) -> Option<Val> {
    let mut found = None;
    walk(root, &mut Vec::new(), 0, &mut |_, op| {
        if let Op::Sentient(sentient::Op::For { carried, .. }) = op {
            for entry in carried {
                if entry.arg == val || entry.result == val {
                    found = Some(entry.init);
                }
            }
        }
    });
    found
}

/// The scalar type a value carries — `Value::getType`, for the scalar ops this pass compares.
///
/// ⭐ An op that binds no scalar type answers `None`, which SKIPS an iter-arg candidate rather than
/// accepting it on an unknown type.
fn scalar_ty_of(val: Val, root: &[Op]) -> Option<ScalarTy> {
    if let Some(init) = carried_init_of(val, root) {
        return scalar_ty_of(init, root);
    }
    match ir::defining_op(val, root)? {
        Op::Sentient(inner) => match inner {
            sentient::Op::ScalarConstant { ty, .. }
            | sentient::Op::ScalarAdd { ty, .. }
            | sentient::Op::ScalarSub { ty, .. }
            | sentient::Op::ScalarMul { ty, .. } => Some(*ty),
            sentient::Op::ScalarCopy { input, .. } => scalar_ty_of(*input, root),
            _ => None,
        },
        Op::Arith(inner) => match inner {
            arith::Op::Constant { .. } => Some(ScalarTy::Index),
            arith::Op::ConstantInt { value, .. } => Some(match value {
                arith::IntConst::Bool(_) => ScalarTy::Int(1),
                arith::IntConst::Int { bits, .. } => ScalarTy::Int(*bits),
            }),
            _ => None,
        },
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::AffineFor(_)
        | Op::Vector(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_)
        | Op::UniformRegions(_) => None,
    }
}

/// `dcc::utils::getIterArgIncrementer<sentient::ForOp, sentient::AddOp>` (`dcc/src/Utils/Utils.cpp:342`)
/// reduced to its answer here: the `$inp2` of the `sentient.scalar_add` that advances iter arg
/// `index`, which is that iter arg's step.
fn iter_arg_step(
    carried: &[sentient::Carried],
    yielded: &[Val],
    index: usize,
    root: &[Op],
) -> Option<Val> {
    let arg = carried.get(index)?.arg;
    let yield_operand = *yielded.get(index)?;
    let Op::Sentient(sentient::Op::ScalarAdd { lhs, rhs, .. }) =
        ir::defining_op(yield_operand, root)?
    else {
        return None;
    };
    (*lhs == arg).then_some(*rhs)
}

/// `CFGSSentientLevelConditionalTree::findClosestParent` — the closest enclosing op the tree has a
/// node for (`Analyses/RedundantDefinitionEliminationTree.hpp:276`), and `isOperationSelected` there
/// is `isa<sentient::IfOp>` (`Analyses/CFGSSentientLevelConditionalTree.cpp:639-641`).
fn find_closest_parent(root: &[Op], op: &OpPath) -> Option<OpPath> {
    let mut at = op.parent();
    while let Some(candidate) = at {
        if let Some(Op::Sentient(sentient::Op::If { .. })) = op_at(root, candidate.path()) {
            return Some(candidate);
        }
        at = candidate.parent();
    }
    None
}

// ── CODE GENERATION SUPPORT ─────────────────────────────────────────────────────────────────────

/// `SentientRegType::unknown` — the `locale_attr` every conditional generated in this file carries
/// (`:2769-2771`, `:2831-2833`), no register chosen yet.
const UNKNOWN_LOCALE: sentient::Reg = sentient::Reg {
    locale: sentient::RegType::Unknown,
    index: None,
};

/// `mlir::sentient::ConstantOp::create(const_builder_, unit_op_.getLoc(), <type>, <value>)`.
fn scalar_constant(result: Val, value: i64, ty: ScalarTy) -> Op {
    Op::Sentient(sentient::Op::ScalarConstant {
        value,
        result,
        // `ConstantOp`'s own default, which the reference never overrides at these call sites.
        reg_locale: sentient::RegType::Imm,
        ty,
        is_symbol: false,
    })
}

/// `sentient::YieldOp::create(builder, loc, results)`.
fn yield_of(results: &[Val]) -> Op {
    Op::Sentient(sentient::Op::Yield {
        results: results.to_vec(),
    })
}

/// `*it_encoding; ++it_encoding` — the encoding tuple is read strictly in order, and running off its
/// end is the reference's read past `end()`.
fn next_encoding(encoding_tuple: &[Val], next: &mut usize) -> Option<Val> {
    let value = encoding_tuple.get(*next).copied();
    *next += 1;
    value
}

/// `new Sequence(evaluator_.getConstant(0), evaluator_.getConstant(0), 0, prev_interval_marker,
/// interval_stride, next_expected, evaluator_)` (`:2417-2419`, `:2424-2426`) — the dummy whose
/// predicate is always false because it repeats the previous sequence's interval marker.
fn dummy_sequence<E: ExpressionEvaluator + ?Sized>(
    evaluator: &mut E,
    interval_marker: IntervalMarker,
    interval_stride: IntervalStride,
    kind: SequenceKind,
) -> Sequence {
    let lb = evaluator.get_constant(0);
    let stride = evaluator.get_constant(0);
    Sequence::new(
        lb,
        stride,
        Entries(0),
        interval_marker,
        interval_stride,
        kind,
    )
}

/// A VALUE AND THE TYPE IT CARRIES — `Value` plus the `iv.getType()` of `:2737`, which the island
/// keeps on the op that BINDS the value rather than on the value itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedVal {
    /// The value.
    pub val: Val,
    /// Its type.
    pub ty: ScalarTy,
}

/// WHERE A FRESHLY BUILT CONDITIONAL LANDS — the `OpBuilder builder_new_if(if_op)` of `:1180`/`:1330`,
/// which inserts BEFORE `if_op`, as the block plus the path that op occupies.
///
/// ⛔ THE PATH IS NEEDED AS WELL AS THE BLOCK, because a [`Mark`] is keyed by absolute [`OpPath`]: an
/// op built here cannot be marked without knowing where the tree will sit.
pub struct Site<'a> {
    /// The block the builder inserts into.
    pub block: &'a mut Vec<Op>,
    /// The absolute path of the insertion point.
    pub at: OpPath,
}

impl Site<'_> {
    /// `builder.insert(op)` — appended when the path names no position in this block.
    fn insert(&mut self, op: Op) {
        let index = self
            .at
            .path()
            .last()
            .map_or(self.block.len(), |&(_, index)| index as usize);
        self.block.insert(index.min(self.block.len()), op);
    }
}

/// ONE LEVEL OF A GENERATED `if / else if / ... / else` CHAIN.
struct IfLevel {
    /// `comparison` (`:2752-2758`), from the sign of the sequence's interval stride.
    predicate: sentient::CmpPredicate,
    /// `rhs_val` — the interval marker this level tests against.
    rhs: Val,
    /// What this level's `then` region yields.
    then_result: Val,
    /// The value this level's `sentient.if` binds.
    result: Val,
    /// `RESULT_TO_BE_REPLACED_BY_ITER_ARG` on the `then` yield (`:2784-2786`).
    mark_then_result: bool,
    /// `LHS_IN_PRED_TO_BE_REPLACED_BY_ITER_ARG` on the conditional itself (`:2775-2777`).
    mark_lhs_in_pred: bool,
}

/// The `if (iv P m0) .. else if (iv P m1) .. else <else_result>` those levels describe, outermost
/// first — each level's ELSE region holding the next level and a yield of its result (`:2761-2789`,
/// `:2836-2853`).
fn if_chain(levels: &[IfLevel], iv: Val, else_result: Val) -> Option<Op> {
    let mut built: Option<Op> = None;
    for (depth, level) in levels.iter().enumerate().rev() {
        let else_body = match built.take() {
            // `YieldOp::create(else_builder, .., sentient_ifop_tmp.getResults()[0])` (`:2772-2773`).
            Some(inner) => {
                let inner_result = levels
                    .get(depth + 1)
                    .map_or(else_result, |next| next.result);
                vec![inner, yield_of(&[inner_result])]
            }
            None => vec![yield_of(&[else_result])],
        };
        built = Some(Op::Sentient(sentient::Op::If {
            predicate: level.predicate,
            lhs: iv,
            rhs: level.rhs,
            yielded: vec![sentient::Yielded {
                result: level.result,
                reg: UNKNOWN_LOCALE,
                element_size: None,
            }],
            dbg_name: None,
            then_body: vec![yield_of(&[level.then_result])],
            else_body,
        }));
    }
    built
}

/// The absolute path of level `depth` of a chain built at `at` — every level below the first sits at
/// position 0 of its parent's ELSE region, which [`sentient::regions`] numbers 1.
fn level_path(at: &OpPath, depth: usize) -> OpPath {
    let mut path = at.clone();
    for _ in 0..depth {
        path = path.child(1, 0);
    }
    path
}

impl PatternSimplificationManager {
    /// Replaces: e031_populateTable
    ///
    /// Walks the conditional subtree, fixing each IV to the value its predicate tests, marking the
    /// branches an ancestor already decided as dead, and handing every leaf to `process_leaf`.
    ///
    /// TRAP: `process_leaf` IS `e431_processLeaf` (level 2, not this batch) — taken as a parameter so
    /// that the whole effect, `setAsDead` on a leaf it rejects included, is complete here.
    pub fn populate_table(
        &mut self,
        n: &mut CondNode,
        ivs: &mut IvValuesAndFilters,
        process_leaf: &mut impl FnMut(&Branch, &mut IvValuesAndFilters) -> bool,
    ) {
        let iv = n.lhs;
        let rhs_val = n.rhs_val;
        // `DT_CHECK_MSG(step != 0, ..)` (`:933`) is [`LoopInfo::step`]'s type; an IV with no recorded
        // loop is the state that check catches, and there is no table dimension to populate for it.
        let (iterations, idx) = {
            let Some(info) = self.lhs_to_for_op_or_null.get(&iv) else {
                return;
            };
            (info.iterations, TableIdx::of(rhs_val, info))
        };
        // `DT_CHECK_MSG(is_idx_valid || n->getThenNode()->isDead(), ..)` (`:937`) is a pure
        // assertion about the tree the caller handed over: nothing here refuses at runtime.
        let chosen = ivs.chosen(iv);
        let disallowed = idx.is_some_and(|at| ivs.filter(iv, at, iterations));
        let skip_true_branch =
            chosen.is_some_and(|value| value != rhs_val) || disallowed || n.then_node.dead;
        let skip_false_branch = chosen.is_some_and(|value| value == rhs_val) || n.else_node.dead;

        if skip_true_branch {
            self.marks.set(Mark::DeadThenBranch, n.op.path());
        } else {
            ivs.set_chosen(iv, Some(rhs_val), iterations);
            if n.then_node.child.is_some() {
                if let Some(child) = n.then_node.child.as_deref_mut() {
                    self.populate_table(child, ivs, process_leaf);
                }
            } else if !process_leaf(&n.then_node, ivs) {
                n.then_node.dead = true;
                self.marks.set(Mark::DeadThenBranch, n.op.path());
            }
            ivs.set_chosen(iv, chosen, iterations);
        }

        if skip_false_branch {
            self.marks.set(Mark::DeadElseBranch, n.op.path());
        } else {
            let previous = idx.map(|at| ivs.set_filter(iv, at, true, iterations));
            if n.else_node.child.is_some() {
                if let Some(child) = n.else_node.child.as_deref_mut() {
                    self.populate_table(child, ivs, process_leaf);
                }
            } else if !process_leaf(&n.else_node, ivs) {
                n.else_node.dead = true;
                self.marks.set(Mark::DeadElseBranch, n.op.path());
            }
            if let (Some(at), Some(bit)) = (idx, previous) {
                ivs.set_filter(iv, at, bit, iterations);
            }
        }
    }

    /// Replaces: e032_findExistingIterArg
    ///
    /// The index of an iter arg this IV's loop ALREADY carries with the sought type, lower bound and
    /// stride, cascading outwards through `rest` so every enclosing loop matches too.
    ///
    /// TRAP: the stride comparison against a target `EvaluatedValue` is `ExpressionEvaluatorUtils`,
    /// out of campaign scope — see [`step_matches_target`].
    pub fn find_existing_iter_arg(
        &self,
        root: &[Op],
        cur: &IvDim,
        rest: &[IvDim],
        inner_for_op: Option<&OpPath>,
        target: &IterArgTarget,
        inner_lb: Option<Val>,
    ) -> Option<usize> {
        // `DT_CHECK_MSG(for_op, ..)` (`:1465`) and `DT_CHECK_MSG(sentient_for, ..)` (`:1467`): only a
        // `sentient.for` recorded for this IV has iter args to offer.
        let for_op_path = self.lhs_to_for_op_or_null.get(&cur.iv)?.for_op.clone()?;
        let Op::Sentient(sentient::Op::For { carried, body, .. }) =
            op_at(root, for_op_path.path())?
        else {
            return None;
        };
        // `sentient_for.getBody()->getTerminator()`.
        let yield_index = body.len().checked_sub(1)?;
        let Op::Sentient(sentient::Op::Yield { results: yielded }) = body.get(yield_index)? else {
            return None;
        };
        let yield_path = for_op_path.child(0, u32::try_from(yield_index).ok()?);

        for index in 0..carried.len() {
            let candidate = carried[index].arg;
            // Not the innermost loop: the iter arg must be the one the next inner loop starts from.
            if inner_lb.is_some_and(|lb| lb != candidate) {
                continue;
            }
            // The candidate may only be read by conditionals, by the next inner loop, by the yield,
            // by a dead op, or by a `scalar_add` that feeds nothing but the yield (`:1477-1499`) —
            // otherwise reusing it would mismatch register types.
            let mut is_candidate = true;
            for use_path in uses_of(candidate, root) {
                let Some(use_owner) = op_at(root, use_path.path()) else {
                    continue;
                };
                if matches!(use_owner, Op::Sentient(sentient::Op::If { .. }))
                    || inner_for_op.is_some_and(|inner| *inner == use_path)
                    || use_path == yield_path
                    || use_empty(use_owner, root)
                {
                    continue;
                }
                let Op::Sentient(sentient::Op::ScalarAdd { result, .. }) = use_owner else {
                    is_candidate = false;
                    break;
                };
                if uses_of(*result, root)
                    .into_iter()
                    .any(|add_use| add_use != yield_path)
                {
                    is_candidate = false;
                    break;
                }
            }
            if !is_candidate {
                continue;
            }

            let Some(yielded_val) = yielded.get(index).copied() else {
                continue;
            };
            if scalar_ty_of(yielded_val, root) != Some(target.ty) {
                continue;
            }
            // The yielded value is either the inner loop's own result or has to carry the stride.
            if defining_path(yielded_val, root).as_ref() != inner_for_op {
                let Some(step) = iter_arg_step(carried, yielded, index, root) else {
                    continue;
                };
                match target.step {
                    Some(target_step) => {
                        if const_value(step, root).is_none()
                            || !step_matches_target(step, target_step, cur.multiplier)
                        {
                            continue;
                        }
                    }
                    None => {
                        if Some(step) != self.monotone_seq_val_step {
                            continue;
                        }
                    }
                }
            }
            // `DT_CHECK_MSG(lower_bound, ..)` (`:1527`): a loop has one init per carried value.
            let lower_bound = carried[index].init;
            if rest.is_empty() {
                if is_same_constant(lower_bound, target.lb, root) {
                    return Some(index);
                }
            } else if self
                .find_existing_iter_arg(
                    root,
                    &rest[0],
                    &rest[1..],
                    Some(&for_op_path),
                    target,
                    Some(lower_bound),
                )
                .is_some()
            {
                return Some(index);
            }
        }
        None
    }

    /// Replaces: e033_replaceResultByIterArg
    ///
    /// Points every marked `sentient.yield` in the loop at the new iter arg instead of the value it
    /// was yielding.
    pub fn replace_result_by_iter_arg(
        &mut self,
        root: &mut Vec<Op>,
        for_op: &OpPath,
        replacement: Val,
    ) {
        let marks = &mut self.marks;
        walk_mut_under(root, for_op, &mut |at, op| {
            if let Op::Sentient(sentient::Op::Yield { results }) = op
                && marks.take(Mark::ResultReplacedByIterArg, at)
                && let Some(first) = results.first_mut()
            {
                *first = replacement;
            }
        });
    }

    /// Replaces: e034_replaceIfOpByIterArg
    ///
    /// Rewires every reader of a marked conditional's first result onto the new iter arg.
    ///
    /// TRAP: `getResult(0)` of a `sentient.if` is `yielded[0].result` — what `sentient::results`
    /// answers first.
    pub fn replace_if_op_by_iter_arg(
        &mut self,
        root: &mut Vec<Op>,
        for_op: &OpPath,
        replacement: Val,
    ) {
        let marks = &mut self.marks;
        let mut replaced = Vec::new();
        walk_mut_under(root, for_op, &mut |at, op| {
            if let Op::Sentient(sentient::Op::If { yielded, .. }) = op
                && marks.take(Mark::IfOpReplacedByIterArg, at)
                && let Some(first) = yielded.first()
            {
                replaced.push(first.result);
            }
        });
        for result in replaced {
            ir::replace_all_uses_with(root, result, replacement);
        }
    }

    /// Replaces: e035_replacePredicateIVByIterArg
    ///
    /// Replaces the IV in every marked conditional's predicate with the new iter arg.
    ///
    /// TRAP: `setOperand(0, ..)` is the predicate's `$lhs` — `sentient::operands_mut` on an `If`
    /// answers `[lhs, rhs]`.
    pub fn replace_predicate_iv_by_iter_arg(
        &mut self,
        root: &mut Vec<Op>,
        for_op: &OpPath,
        replacement: Val,
    ) {
        let marks = &mut self.marks;
        walk_mut_under(root, for_op, &mut |at, op| {
            if let Op::Sentient(sentient::Op::If { lhs, .. }) = op
                && marks.take(Mark::LhsInPredReplacedByIterArg, at)
            {
                *lhs = replacement;
            }
        });
    }

    /// Replaces: e036_checkCascadingArgUses
    ///
    /// Whether each conditional outwards from `cur_if_op` tests the matching IV of `cur` then `rest`,
    /// so one nest of predicates cascades over the whole IV list.
    ///
    /// TRAP: a path that names something other than a `sentient.if` is the null `dyn_cast` the
    /// reference dereferences unchecked (`:1582`); no predicate there names the IV.
    pub fn check_cascading_arg_uses(
        &self,
        root: &[Op],
        cur_if_op: &OpPath,
        cur: &IvDim,
        rest: &[IvDim],
    ) -> bool {
        let Some(Op::Sentient(sentient::Op::If { lhs, rhs, .. })) = op_at(root, cur_if_op.path())
        else {
            return false;
        };
        if *lhs != cur.iv && *rhs != cur.iv {
            return false;
        }
        if rest.is_empty() {
            return true;
        }
        let Some(parent_if_op) = find_closest_parent(root, cur_if_op) else {
            return false;
        };
        self.check_cascading_arg_uses(root, &parent_if_op, &rest[0], &rest[1..])
    }
}

/// The seven code-generation units of level 1 — `:2096`-`:2860`.
///
/// ⛔ `ExpressionEvaluator` AND THE TWO BUILDERS ARE PARAMETERS, NOT MEMBERS
/// ([`super::ExpressionEvaluator`], [`Builders`]): the analysis is out of campaign scope, and a
/// `&mut Vec<Op>` held by the manager could not coexist with the lookups its own methods make.
impl PatternSimplificationManager {
    /// Replaces: e289_computeTemplateSeqAndUpdateMonotoneSeq
    ///
    /// Copies the first slice's sequences into the template, shifts every monotone sequence's LB back
    /// by the entries before it, clears whatever the other slices disagree about, and — when the
    /// monotone step is zero and there is no third sequence — turns every monotone sequence into a
    /// default value.
    ///
    /// TRAP: A MONOTONE SEQUENCE WITH NO LB OR NO STRIDE gets no shifted LB rather than the
    /// reference's null deref (`:2109-2112`); only a TEMPLATE copy ever has its LB cleared (`:2115`).
    pub fn compute_template_seq_and_update_monotone_seq(
        &mut self,
        table_slices: &mut [TableSlice],
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        // `table_slices.front()` (`:2103`) — there is nothing to template from without a slice.
        let Some(first) = table_slices.first() else {
            return;
        };
        let first_sequences = first.sequences.clone();
        let mut entries_seen_so_far = Entries::default();
        let mut template_monotone_seq = None;
        for s in first_sequences {
            let mut copy = s;
            if s.kind == SequenceKind::MonotoneSequence {
                template_monotone_seq = Some(self.template_sequences.len());
                // shifted_lb = lb - stride * entries_seen_so_far (`:2109-2112`).
                if let (Some(lb), Some(stride)) = (s.lb, s.stride) {
                    let scaled = evaluator.evaluate_multiply_by_const(stride, entries_seen_so_far);
                    copy.shifted_lb = Some(evaluator.evaluate_sub(lb, scaled));
                }
                // The shifted LB is used from here on rather than the LB (`:2115`).
                copy.lb = None;
            }
            self.template_sequences.push(copy);
            entries_seen_so_far += s.length;
        }
        // `:2120-2153` — mark every field the slices disagree about as inconsistent.
        for table_slice in table_slices.iter_mut() {
            let mut entries_seen_so_far = Entries::default();
            // `DT_CHECK(sequences.size() == template_sequences_.size())` (`:2123-2125`) is what makes
            // the zip total; a shorter slice simply has no template sequence to compare against.
            for (index, cur_seq) in table_slice.sequences.iter_mut().enumerate() {
                let Some(template_seq) = self.template_sequences.get_mut(index) else {
                    break;
                };
                if !matches!((cur_seq.lb, template_seq.lb), (Some(cur), Some(template)) if evaluator.equal(cur, template))
                {
                    template_seq.lb = None;
                }
                if !matches!((cur_seq.stride, template_seq.stride), (Some(cur), Some(template)) if evaluator.equal(cur, template))
                {
                    template_seq.stride = None;
                }
                if cur_seq.interval_marker != template_seq.interval_marker {
                    template_seq.interval_marker = None;
                }
                // `:2143-2151` — each slice's own monotone sequence gets its shifted LB too.
                if cur_seq.kind == SequenceKind::MonotoneSequence
                    && let (Some(lb), Some(stride)) = (cur_seq.lb, cur_seq.stride)
                {
                    let scaled = evaluator.evaluate_multiply_by_const(stride, entries_seen_so_far);
                    let shifted_val = evaluator.evaluate_sub(lb, scaled);
                    cur_seq.shifted_lb = Some(shifted_val);
                    if let Some(template_shifted) = template_seq.shifted_lb
                        && !evaluator.equal(template_shifted, shifted_val)
                    {
                        template_seq.shifted_lb = None;
                    }
                }
                entries_seen_so_far += cur_seq.length;
            }
        }
        // `:2157-2169` — a zero step with no third sequence is a default value in disguise.
        let candidate = template_monotone_seq
            .filter(|_| self.template_sequences.len() < 3)
            .and_then(|index| Some((index, self.template_sequences.get(index)?.stride?)));
        let Some((index, stride)) = candidate else {
            return;
        };
        let zero = evaluator.get_constant(0);
        if !evaluator.equal(stride, zero) {
            return;
        }
        if let Some(monotone) = self
            .template_sequences
            .get_mut(index)
            .and_then(Sequence::as_monotone_mut)
        {
            monotone.change_to_default_value();
        }
        for table_slice in table_slices.iter_mut() {
            for s in &mut table_slice.sequences {
                if let Some(monotone) = s.as_monotone_mut() {
                    monotone.change_to_default_value();
                }
            }
        }
    }

    /// Replaces: e290_insertSequence
    ///
    /// Appends `seq` to the slice and, past the cap of three sequences, records which pattern has
    /// failed — a 1-D slice aborts the whole match, a monotone one does not, since every slice's
    /// sequences are still needed (`:2382-2384`).
    pub fn insert_sequence(&mut self, seq: Sequence, table_slice: &mut TableSlice) {
        if self.abort_pattern {
            return;
        }
        if table_slice
            .sequences
            .last()
            .is_some_and(|last| last.kind == SequenceKind::MonotoneSequence)
        {
            table_slice.has_inserted_monotone_seq_before_last = true;
        }
        table_slice.sequences.push(seq);
        if table_slice.sequences.len() > 3 {
            if table_slice.is_1d_rep_of_table {
                self.table_follows_contiguous_pattern = false;
                self.abort_pattern = true;
            } else {
                self.table_slices_follow_montone_pattern = false;
            }
        }
    }

    /// Replaces: e291_padTableSlice
    ///
    /// Pads the slice with dummy sequences — each repeating the previous interval marker, so its
    /// predicate is always false — until it has `max_num_seq` of them in the expected kind order.
    ///
    /// TRAP: WITH [`ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS`] OFF the expected kind is always
    /// `kDefaultValue`, so a slice holding a monotone sequence cannot reach `target_count` and hits
    /// the reference's own `DT_CHECK` (`:2445-2446`); an empty slice is its `front()` on empty.
    pub fn pad_table_slice(
        table_slice: &mut TableSlice,
        max_num_seq: usize,
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        let sequences = &mut table_slice.sequences;
        let mut next_expected = SequenceKind::DefaultValue;
        let Some(first_seq) = sequences.first() else {
            return;
        };
        let interval_stride = first_seq.interval_stride();
        // m0 = m1 - first sequence stride * first sequence length (`:2405-2407`).
        let Some(mut prev_interval_marker) = first_seq.interval_marker else {
            return;
        };
        prev_interval_marker -= interval_stride * first_seq.length;
        let target_count = if ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS {
            3
        } else {
            max_num_seq
        };
        let mut at = 0usize;
        for _ in 0..target_count {
            if at == sequences.len() {
                // Pad after the current sequences (`:2415-2421`).
                sequences.push(dummy_sequence(
                    evaluator,
                    prev_interval_marker,
                    interval_stride,
                    next_expected,
                ));
                at = sequences.len();
            } else if sequences[at].kind == next_expected {
                // The next sequence is of the expected kind (`:2429-2432`).
                if let Some(marker) = sequences[at].interval_marker {
                    prev_interval_marker = marker;
                }
                at += 1;
            } else {
                // Pad in the middle, leaving `at` on the unexpected sequence (`:2422-2428`).
                sequences.insert(
                    at,
                    dummy_sequence(
                        evaluator,
                        prev_interval_marker,
                        interval_stride,
                        next_expected,
                    ),
                );
                at += 1;
            }
            next_expected = if next_expected == SequenceKind::DefaultValue
                && ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS
            {
                SequenceKind::MonotoneSequence
            } else {
                SequenceKind::DefaultValue
            };
        }
    }

    /// Replaces: e292_generateEncodingTypes
    ///
    /// The type of every field the encoding has to carry: one table-entry type per field the template
    /// does NOT fix, plus a predicate type for each interval marker that varies — the last sequence's
    /// marker being the `else` and never encoded.
    pub fn generate_encoding_types(&self) -> Vec<ScalarTy> {
        let mut types = Vec::new();
        let Some(table) = self.table.as_ref() else {
            return types;
        };
        let table_entry_type = table.table_entry_type();
        for (seq_seen_so_far, template_s) in self.template_sequences.iter().enumerate() {
            let skip_interval_marker = seq_seen_so_far + 1 == self.template_sequences.len();
            match template_s.kind {
                // A monotone sequence keeps both the starting value and the step (`:2463-2467`).
                SequenceKind::MonotoneSequence => {
                    if template_s.shifted_lb.is_none() {
                        types.push(table_entry_type);
                    }
                    if template_s.stride.is_none() {
                        types.push(table_entry_type);
                    }
                }
                SequenceKind::DefaultValue => {
                    if template_s.lb.is_none() {
                        types.push(table_entry_type);
                    }
                }
            }
            if !skip_interval_marker && template_s.interval_marker.is_none() {
                types.push(table.predicate_type());
            }
        }
        types
    }

    /// Replaces: e293_generateEncodingTuple
    ///
    /// The values for those types, read off this slice's own sequences: a materialised offset value
    /// per field the template leaves open, and a `sentient.scalar_constant` per varying marker.
    ///
    /// TRAP: `Value &iv` AND `OpBuilder &builder` ARE UNREAD IN THE REFERENCE (`:2485`) — everything is
    /// built through `const_builder_`/`query_map_builder_`, so neither is a parameter here. A field the
    /// template leaves open that this sequence has not got is its null deref (`:2510`).
    pub fn generate_encoding_tuple(
        &self,
        cur_table_slice: &TableSlice,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> Vec<Val> {
        let mut encoding_tuple = Vec::new();
        let Some(table) = self.table.as_ref() else {
            return encoding_tuple;
        };
        let table_entry_type = table.table_entry_type();
        let sequences = &cur_table_slice.sequences;
        for (seq_seen_so_far, (s, template_s)) in
            sequences.iter().zip(&self.template_sequences).enumerate()
        {
            let skip_interval_marker = seq_seen_so_far + 1 == sequences.len();
            match s.kind {
                SequenceKind::MonotoneSequence => {
                    if let (None, Some(shifted_lb)) = (template_s.shifted_lb, s.shifted_lb) {
                        encoding_tuple.push(evaluator.build_offset_value(
                            shifted_lb,
                            table_entry_type,
                            builders,
                        ));
                    }
                    if let (None, Some(stride)) = (template_s.stride, s.stride) {
                        encoding_tuple.push(evaluator.build_offset_value(
                            stride,
                            table_entry_type,
                            builders,
                        ));
                    }
                }
                SequenceKind::DefaultValue => {
                    if let (None, Some(lb)) = (template_s.lb, s.lb) {
                        encoding_tuple.push(evaluator.build_offset_value(
                            lb,
                            table_entry_type,
                            builders,
                        ));
                    }
                }
            }
            if !skip_interval_marker
                && template_s.interval_marker.is_none()
                && let Some(interval_marker) = s.interval_marker
            {
                let result = builders.values.mint();
                builders.consts.push(scalar_constant(
                    result,
                    interval_marker.0,
                    table.predicate_type(),
                ));
                encoding_tuple.push(result);
            }
        }
        encoding_tuple
    }

    /// `create_simple_two_branch_conditional` (`:2666-2678`) — three template sequences whose two
    /// default values are the same known value and whose first two markers are exactly one interval
    /// apart, so `if (iv == m1)` covers the middle sequence on its own.
    fn two_branch_special_case(&self, evaluator: &impl ExpressionEvaluator) -> bool {
        let [first, middle, last] = self.template_sequences.as_slice() else {
            return false;
        };
        if first.kind != SequenceKind::DefaultValue || last.kind != SequenceKind::DefaultValue {
            return false;
        }
        if !matches!((first.lb, last.lb), (Some(lhs), Some(rhs)) if evaluator.equal(lhs, rhs)) {
            return false;
        }
        match (first.interval_marker, middle.interval_marker) {
            (Some(m0), Some(m1)) => m1 - m0 == middle.interval_stride().one_entry(),
            _ => false,
        }
    }

    /// Replaces: e294_codeGenMonotoneTableSlice
    ///
    /// Builds `if (iv <= m0) yield D0 else if (iv <= m1) yield a1 else yield D1` from the template and
    /// `encoding_tuple`, records the monotone sequence's start value and step, marks the yields that
    /// become an iterator argument and — with `mark_new_cmp` over a table that is not 1-D — each new
    /// comparison; answers the outermost conditional's result.
    ///
    /// TRAP: THE 2-BRANCH SPECIAL CASE STILL BUILDS SEQUENCE 0's MARKER CONSTANT before skipping its
    /// branch (`:2735-2746`), and an empty template is the reference's `new_if.getResults()` on null.
    pub fn code_gen_monotone_table_slice(
        &mut self,
        mark_new_cmp: bool,
        iv: TypedVal,
        encoding_tuple: &[Val],
        site: &mut Site<'_>,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> Option<Val> {
        let table_entry_type = self.table.as_ref()?.table_entry_type();
        let create_simple_two_branch_conditional = self.two_branch_special_case(evaluator);
        let count = self.template_sequences.len();
        let mut next = 0usize;
        let mut levels: Vec<IfLevel> = Vec::new();
        let mut else_result = None;
        let mut mark_else_result = false;
        for i in 0..count {
            let s = *self.template_sequences.get(i)?;
            // The value this sequence yields, from the template when every slice agrees on it and
            // from the encoding otherwise (`:2688-2712`).
            let result = match s.kind {
                SequenceKind::MonotoneSequence => {
                    let result = match s.shifted_lb {
                        Some(shifted_lb) => {
                            evaluator.build_offset_value(shifted_lb, table_entry_type, builders)
                        }
                        None => next_encoding(encoding_tuple, &mut next)?,
                    };
                    match s.stride {
                        Some(stride) => self.monotone_seq_int_step = Some(stride),
                        None => {
                            self.monotone_seq_val_step =
                                Some(next_encoding(encoding_tuple, &mut next)?);
                        }
                    }
                    self.monotone_seq_start_val = Some(result);
                    result
                }
                SequenceKind::DefaultValue => match s.lb {
                    Some(lb) => evaluator.build_offset_value(lb, table_entry_type, builders),
                    None => next_encoding(encoding_tuple, &mut next)?,
                },
            };
            // One sequence needs no conditional at all (`:2714-2717`).
            if count == 1 {
                return Some(result);
            }
            // The last sequence corresponds to the "else", so it needs no predicate (`:2718-2730`).
            if i + 1 == count {
                else_result = Some(result);
                mark_else_result = s.kind == SequenceKind::MonotoneSequence;
                break;
            }
            // The interval marker, again from the template when it is common (`:2733-2743`).
            let rhs = match s.interval_marker {
                Some(interval_marker) => {
                    let rhs = builders.values.mint();
                    builders
                        .consts
                        .push(scalar_constant(rhs, interval_marker.0, iv.ty));
                    rhs
                }
                None => next_encoding(encoding_tuple, &mut next)?,
            };
            // The special case has no branch for the first default value (`:2745-2746`).
            if create_simple_two_branch_conditional && i == 0 {
                continue;
            }
            let predicate = if create_simple_two_branch_conditional && i == 1 {
                // The middle sequence is a single interval marker value (`:2755-2758`).
                sentient::CmpPredicate::Eq
            } else if s.interval_stride().get().get() > 0 {
                sentient::CmpPredicate::Sle
            } else {
                sentient::CmpPredicate::Sge
            };
            levels.push(IfLevel {
                predicate,
                rhs,
                then_result: result,
                result: builders.values.mint(),
                mark_then_result: s.kind == SequenceKind::MonotoneSequence,
                mark_lhs_in_pred: mark_new_cmp && self.ivs_dimensions_multipliers.len() > 1,
            });
        }
        site.insert(if_chain(&levels, iv.val, else_result?)?);
        for (depth, level) in levels.iter().enumerate() {
            let path = level_path(&site.at, depth);
            if level.mark_lhs_in_pred {
                self.marks
                    .set(Mark::LhsInPredReplacedByIterArg, path.path());
            }
            if level.mark_then_result {
                self.marks
                    .set(Mark::ResultReplacedByIterArg, path.child(0, 0).path());
            }
        }
        if mark_else_result {
            let innermost = level_path(&site.at, levels.len().saturating_sub(1));
            self.marks
                .set(Mark::ResultReplacedByIterArg, innermost.child(1, 0).path());
        }
        levels.first().map(|level| level.result)
    }

    /// Replaces: e295_codeGenGenericTableSlice
    ///
    /// Builds `if (iv <= m0) yield D0 ... else yield Dk` over a slice of default-value sequences, the
    /// last one falling under the innermost `else`; answers the outermost conditional's result, or the
    /// single sequence's own value when there is nothing to branch on.
    ///
    /// TRAP: A SEQUENCE THAT IS NOT `kDefaultValue` is the reference's `DT_CHECK` (`:2821-2822`) and
    /// one missing its LB or marker is its null deref (`:2823-2830`) — here either stops the build.
    pub fn code_gen_generic_table_slice(
        &self,
        iv: Val,
        table_slice: &TableSlice,
        site: &mut Site<'_>,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> Option<Val> {
        let table = self.table.as_ref()?;
        let table_entry_type = table.table_entry_type();
        let (last, leading) = table_slice.sequences.split_last()?;
        // The whole slice yields the same value, so no `sentient.if` is needed (`:2808-2813`).
        if leading.is_empty() {
            return Some(evaluator.build_offset_value(last.lb?, table_entry_type, builders));
        }
        let mut levels = Vec::new();
        for seq in leading {
            let then_result = evaluator.build_offset_value(seq.lb?, table_entry_type, builders);
            let rhs = builders.values.mint();
            builders.consts.push(scalar_constant(
                rhs,
                seq.interval_marker?.0,
                table.predicate_type(),
            ));
            levels.push(IfLevel {
                predicate: if seq.interval_stride().get().get() > 0 {
                    sentient::CmpPredicate::Sle
                } else {
                    sentient::CmpPredicate::Sge
                },
                rhs,
                then_result,
                result: builders.values.mint(),
                mark_then_result: false,
                mark_lhs_in_pred: false,
            });
        }
        // The last sequence falls under the "else" (`:2855-2858`).
        let else_result = evaluator.build_offset_value(last.lb?, table_entry_type, builders);
        site.insert(if_chain(&levels, iv, else_result)?);
        levels.first().map(|level| level.result)
    }
}

// ── THE FOUR LEVEL-2 REWRITERS AND THEIR CALL VOCABULARY ────────────────────────────────────────

/// `cl::opt<int> MaximumIterArgs("dcc-cfg-sentient-level-max-num-iter-args", .., cl::init(6))`
/// (`:70-74`) — how many iterator arguments one loop may carry.
const MAXIMUM_ITER_ARGS: usize = 6;

/// THE THREE THINGS ONE LEAF TOUCHES, SPLIT OUT OF THE MANAGER —
/// [`PatternSimplificationManager::populate_table`] already holds `&mut self` while it calls the leaf
/// handler, so the handler borrows the two maps it reads plus the table it writes, not the manager.
pub struct LeafSink<'a> {
    /// `ivs_dimensions_multipliers_`, read positionally against the accumulated tuple.
    pub ivs_dimensions_multipliers: &'a [IvDim],
    /// `lhs_to_for_op_or_null_`.
    pub lhs_to_for_op_or_null: &'a BTreeMap<Val, LoopInfo>,
    /// `table_` — a `&mut`, because the entry it writes IS the leaf's effect. Also discharges the
    /// reference's unchecked `table_->` on a null table.
    pub table: &'a mut Table,
}

impl LeafSink<'_> {
    /// Replaces: e431_processLeaf
    ///
    /// Expands one leaf over every iteration tuple its IVs still allow, writing the leaf into the
    /// table at each tuple's flattened index, and answers whether any tuple survived.
    ///
    /// TRAP: THE RECURSIVE ANSWER IS DISCARDED (`:1069`, `:1082`) — only a filter allowing NO
    /// iteration of the IV AT THIS LEVEL kills the leaf. An IV with no recorded loop takes
    /// `lhs_to_for_op_or_null_[iv]`'s default of zero iterations, which is that same dead leaf.
    pub fn process_leaf(
        &mut self,
        leaf: &Branch,
        ivs: &IvValuesAndFilters,
        at: usize,
        tuple_so_far: &mut Vec<i64>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> bool {
        let Some(iv) = ivs.iv_at(at) else {
            // `c1 (i2*..*in) + .. + cn (1)` — each IV's multiplier already holds its tail product.
            let idx: i64 = self
                .ivs_dimensions_multipliers
                .iter()
                .zip(tuple_so_far.iter().copied())
                .map(|(dim, iteration)| dim.multiplier * iteration)
                .sum();
            self.table
                .create_table_entry_at_idx(TableIndex(idx), &leaf.leaf, evaluator);
            return true;
        };
        let info = self.lhs_to_for_op_or_null.get(&iv);
        match ivs.chosen(iv) {
            Some(value) => {
                // `(value - lb) / step`, which without a loop would be the reference's divide by zero.
                let Some(info) = info else { return true };
                tuple_so_far.push((value - info.lb) / info.step.get());
                self.process_leaf(leaf, ivs, at + 1, tuple_so_far, evaluator);
                tuple_so_far.pop();
            }
            None => {
                let mut at_least_one_allowed_value = false;
                for iteration in 0..info.map_or(0, |info| info.iterations) {
                    if !ivs.filter_bit(iv, iteration) {
                        at_least_one_allowed_value = true;
                        tuple_so_far.push(iteration);
                        self.process_leaf(leaf, ivs, at + 1, tuple_so_far, evaluator);
                        tuple_so_far.pop();
                    }
                }
                if !at_least_one_allowed_value {
                    return false;
                }
            }
        }
        true
    }
}

/// THE LOOP CLONE BOTH TRANSFORMS NEED — `createSentientForOpWithAdditionalIterArgs(for_op,
/// start_vals_and_steps)` (`Transform/Sentient/Utils.cpp:195`), plus the loops the tree refuses to grow.
///
/// ⛔ e393 IS NOT PORTED, so the clone arrives as a closure exactly as the leaf handler does for
/// [`PatternSimplificationManager::populate_table`]: given the tree, the loop and one
/// `(start_val, step)` per new iterator argument, it answers where the clone landed.
pub struct LoopCloning<'a> {
    /// `tree.isForOpToAvoidNewIterArgs(for_op)` — the loops loop splitting has claimed.
    pub for_ops_to_avoid_new_iter_args: &'a [OpPath],
    /// `createSentientForOpWithAdditionalIterArgs`.
    pub clone: &'a mut dyn FnMut(&mut Vec<Op>, &OpPath, &[(Val, Val)]) -> OpPath,
}

/// THE CONTIGUOUS-SEQUENCE CALL (`:1597-1601`) — the conditional being replaced, what replaces it,
/// and the two facts `findPatternsAndSimplify` already established about the table.
pub struct ContiguousCase<'a> {
    /// `if_op`.
    pub if_op: &'a OpPath,
    /// `new_if` — the nested conditional built over the table slice.
    pub new_if: Val,
    /// `num_new_iter_args` — up to two per IV, one fewer for each existing iter arg reused.
    pub num_new_iter_args: usize,
    /// `has_positive_step`.
    pub has_positive_step: bool,
}

/// THE MONOTONE-SEQUENCE CALL (`:1922`) — the free IV's loop, the conditional being replaced and
/// the value replacing it.
pub struct MonotoneCase<'a> {
    /// `for_op`, the free IV's `sentient.for`.
    pub for_op: &'a OpPath,
    /// `if_op`.
    pub if_op: &'a OpPath,
    /// `new_if`.
    pub new_if: Val,
}

/// THE RECURSION STATE `codeGenGenericNestedIf` THREADS (`:2557-2564`).
pub struct GenericNest<'a, 'slices> {
    /// `generate_encoding`.
    pub generate_encoding: bool,
    /// `types` — REPRESENTED BY ITS LENGTH ALONE: a `sentient.if` in this island carries a register
    /// per yielded value rather than a type, and the reference's own `locale_attr` holds exactly one.
    pub types: &'a [ScalarTy],
    /// `iter_table_slice`, advanced once per leaf.
    pub slices: core::slice::Iter<'slices, TableSlice>,
}

/// ONE LEVEL OF THE CHAIN e435 BUILDS, kept as a struct so the assembly pass carries a named record
/// rather than a tuple of four vectors.
struct NestedLevel {
    /// The constant this level's predicate tests the IV against.
    rhs: Val,
    /// The values this level's `sentient.if` binds, one per result type.
    results: Vec<Val>,
    /// The ops the `then` recursion produced.
    then_ops: Vec<Op>,
    /// What that recursion answers, which the `then` region yields.
    then_results: Vec<Val>,
}

/// WHICH OF e434's THREE CONTINUATIONS THE NEXT ENTRY TAKES (`:2306-2350`).
enum SequenceStep {
    /// The entry's difference matches the last sequence's stride, so it joins it.
    Extend,
    /// The entry starts a new sequence.
    Start,
    /// The last sequence has one entry, so it swallows this one and adopts the difference.
    Greedy {
        /// Whether that difference is zero, which decides the sequence's kind.
        difference_is_zero: bool,
    },
}

/// `iv_cur_val + by * iv_step`.
fn advanced(marker: IntervalMarker, stride: IntervalStride, by: i64) -> IntervalMarker {
    let mut moved = marker;
    moved += stride * Entries(by);
    moved
}

/// `new Sequence(lb, <int> stride, ..)` (`:269-280`) — the overload the [`Sequence`] type leaves to
/// its callers, since interning the constant belongs to the out-of-scope evaluator.
fn const_stride_sequence(
    evaluator: &mut impl ExpressionEvaluator,
    lb: EvaluatedValueId,
    stride: i64,
    length: Entries,
    interval_marker: IntervalMarker,
    interval_stride: IntervalStride,
    kind: SequenceKind,
) -> Sequence {
    let stride = evaluator.get_constant(stride);
    Sequence::new(lb, stride, length, interval_marker, interval_stride, kind)
}

/// `cur_seq->incrementLength(1); cur_seq->incrementIntervalMarker(1)` — the pair that always travels
/// together, on a slice whose last sequence may have no marker to step.
fn extend_last_sequence(table_slice: &mut TableSlice) {
    if let Some(seq) = table_slice.sequences.last_mut() {
        seq.increment_length(Entries(1));
        if let Some(mut marked) = seq.marked_mut() {
            marked.increment_interval_marker(Entries(1));
        }
    }
}

impl PatternSimplificationManager {
    /// Replaces: e432_transformInContiguousSequenceCase
    ///
    /// Secures one synthetic-IV and one monotone-sequence iterator argument across the whole loop
    /// nest — reusing an existing one or cloning each loop to add it — then retires the original
    /// conditional in favour of `new_if` and points its placeholders at those arguments.
    ///
    /// TRAP: EVERY REFUSAL IS GATED ON `num_new_iter_args > 0` (`:1683`, `:1692`), including the
    /// capacity check, so a nest that needs nothing is never rejected. The outermost loop's veto is
    /// `isa<BlockArgument>(init)`, i.e. an initial value with no defining op.
    pub fn transform_in_contiguous_sequence_case(
        &mut self,
        root: &mut Vec<Op>,
        case: &ContiguousCase<'_>,
        cloning: &mut LoopCloning<'_>,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> bool {
        let Some(table) = self.table.as_ref() else {
            return false;
        };
        let predicate_type = table.predicate_type();
        let table_entry_type = table.table_entry_type();
        let table_size = i64::try_from(table.size().0).unwrap_or(i64::MAX);
        let mut num_new_iter_args = case.num_new_iter_args;
        let mut create_mono_seq_iter_arg = self.monotone_seq_start_val.is_some();
        let mut create_synthetic_iv = self.ivs_dimensions_multipliers.len() > 1;
        let mut synthetic_iv_idx = None;
        let mut seq_iter_arg_idx = None;
        let mut start_vals_and_steps: Vec<(Val, Val)> = Vec::new();
        // `rbegin()`..`rend()`, innermost IV first; `rest` runs to and includes the outermost, which
        // is the reference's `std::prev(rend(), 1)`.
        let ivs: Vec<IvDim> = self
            .ivs_dimensions_multipliers
            .iter()
            .rev()
            .copied()
            .collect();
        let Some((cur, rest)) = ivs.split_first() else {
            return false;
        };

        if create_synthetic_iv {
            let table_lb = if case.has_positive_step {
                0
            } else {
                table_size.saturating_sub(1)
            };
            let table_step = if case.has_positive_step { 1 } else { -1 };
            let table_lb_val = builders.values.mint();
            builders
                .consts
                .push(scalar_constant(table_lb_val, table_lb, predicate_type));
            let step = evaluator.get_constant(table_step);
            synthetic_iv_idx = self.find_existing_iter_arg(
                root,
                cur,
                rest,
                None,
                &IterArgTarget {
                    lb: table_lb_val,
                    step: Some(step),
                    ty: predicate_type,
                },
                None,
            );
            if synthetic_iv_idx.is_some() {
                create_synthetic_iv = false;
                num_new_iter_args = num_new_iter_args.saturating_sub(1);
            } else {
                // `std::make_pair(table_lb_val, table_lb_val)` — a dummy step, replaced per loop.
                start_vals_and_steps.push((table_lb_val, table_lb_val));
            }
        }
        if create_mono_seq_iter_arg {
            // `DT_CHECK(monotone_seq_int_step_ && monotone_seq_start_val_.has_value())` (`:1648-1649`).
            let (Some(start_val), Some(step_val)) =
                (self.monotone_seq_start_val, self.monotone_seq_int_step)
            else {
                return false;
            };
            seq_iter_arg_idx = self.find_existing_iter_arg(
                root,
                cur,
                rest,
                None,
                &IterArgTarget {
                    lb: start_val,
                    step: Some(step_val),
                    ty: table_entry_type,
                },
                None,
            );
            if seq_iter_arg_idx.is_some() {
                create_mono_seq_iter_arg = false;
                num_new_iter_args = num_new_iter_args.saturating_sub(1);
            } else {
                start_vals_and_steps.push((start_val, start_val));
            }
        }

        for (level, dim) in ivs.iter().enumerate() {
            let Some(for_op) = self
                .lhs_to_for_op_or_null
                .get(&dim.iv)
                .and_then(|info| info.for_op.clone())
            else {
                return false;
            };
            if num_new_iter_args > 0 && cloning.for_ops_to_avoid_new_iter_args.contains(&for_op) {
                return false;
            }
            // `DT_CHECK_MSG(sentient_for, "Expect sentient.for op.")` (`:1690`) — NOT gated.
            let Some(Op::Sentient(sentient::Op::For { carried, .. })) = op_at(root, for_op.path())
            else {
                return false;
            };
            if num_new_iter_args == 0 {
                continue;
            }
            if carried.len() + num_new_iter_args > MAXIMUM_ITER_ARGS {
                return false;
            }
            if level + 1 == ivs.len()
                && carried
                    .iter()
                    .any(|entry| defining_path(entry.init, root).is_none())
            {
                return false;
            }
            if level == 0 && !self.check_free_iv_uses(root, cur, rest) {
                return false;
            }
        }

        // `if_op->getResult(0)` read before the mark, since the rewrite below consumes `root`.
        let original_result = first_result_of_if(root, case.if_op);
        self.marks.set(Mark::ToDelete, case.if_op.path());
        let is_one_monotone_seq = matches!(
            self.template_sequences.as_slice(),
            [only] if only.kind == SequenceKind::MonotoneSequence
        );
        if is_one_monotone_seq {
            self.marks
                .set(Mark::IfOpReplacedByIterArg, case.if_op.path());
        } else if let Some(result) = original_result {
            ir::replace_all_uses_with(root, result, case.new_if);
        }

        if !create_synthetic_iv && !create_mono_seq_iter_arg {
            // Nothing to add: the iterator arguments already found do the job (`:1883-1917`).
            let Some(for_op) = self
                .lhs_to_for_op_or_null
                .get(&cur.iv)
                .and_then(|info| info.for_op.clone())
            else {
                return false;
            };
            let Some(carried) = carried_of_for(root, &for_op) else {
                return false;
            };
            let synthetic_iv_replacement =
                synthetic_iv_idx.and_then(|at| carried.get(at).map(|entry| entry.arg));
            let seq_replacement =
                seq_iter_arg_idx.and_then(|at| carried.get(at).map(|entry| entry.arg));
            self.finish_placeholders(
                root,
                &for_op,
                is_one_monotone_seq,
                synthetic_iv_replacement,
                seq_replacement,
            );
            return true;
        }

        let mut previous_for_op: Option<OpPath> = None;
        for (level, dim) in ivs.iter().enumerate() {
            if create_synthetic_iv {
                let step_as_int = if case.has_positive_step {
                    dim.multiplier
                } else {
                    0 - dim.multiplier
                };
                let step = builders.values.mint();
                builders
                    .consts
                    .push(scalar_constant(step, step_as_int, predicate_type));
                if let Some(first) = start_vals_and_steps.first_mut() {
                    first.1 = step;
                }
            }
            if create_mono_seq_iter_arg && let Some(int_step) = self.monotone_seq_int_step {
                let scaled =
                    evaluator.evaluate_multiply_by_const(int_step, Entries(dim.multiplier));
                let step = evaluator.build_offset_value(scaled, table_entry_type, builders);
                if let Some(last) = start_vals_and_steps.last_mut() {
                    last.1 = step;
                }
            }
            let Some(for_op) = self
                .lhs_to_for_op_or_null
                .get(&dim.iv)
                .and_then(|info| info.for_op.clone())
            else {
                return false;
            };
            let new_for_op = (cloning.clone)(root, &for_op, &start_vals_and_steps);
            let Some((carried, body_len)) = carried_and_body_len(root, &new_for_op) else {
                return false;
            };
            let first_new = carried.len().saturating_sub(num_new_iter_args);
            let new_iter_arg: Vec<Val> =
                carried[first_new..].iter().map(|entry| entry.arg).collect();
            let synthetic_iv_replacement = match synthetic_iv_idx {
                Some(at) => carried.get(at).map(|entry| entry.arg),
                None => new_iter_arg.first().copied(),
            };
            let seq_replacement = match seq_iter_arg_idx {
                Some(at) => carried.get(at).map(|entry| entry.arg),
                None => new_iter_arg.last().copied(),
            };
            if level == 0 {
                self.finish_placeholders(
                    root,
                    &new_for_op,
                    is_one_monotone_seq,
                    synthetic_iv_replacement,
                    seq_replacement,
                );
            } else if let Some(prev) = previous_for_op.take() {
                self.rewire_previous_loop(root, &prev, &new_for_op, &new_iter_arg, body_len);
            }
            self.marks
                .set(Mark::PreviousForOpCreated, new_for_op.path());
            previous_for_op = Some(new_for_op);
        }
        if let Some(last) = previous_for_op {
            self.marks.take(Mark::PreviousForOpCreated, last.path());
        }
        true
    }

    /// The innermost loop's IV, and any `sentient.sub` over it, may only be read where turning it into
    /// an iterator argument still leaves loop coalescing possible (`:1710-1750`).
    fn check_free_iv_uses(&self, root: &[Op], cur: &IvDim, rest: &[IvDim]) -> bool {
        let mut sub_result = None;
        for use_path in uses_of(cur.iv, root) {
            let Some(use_owner) = op_at(root, use_path.path()) else {
                continue;
            };
            if let Op::Sentient(sentient::Op::ScalarSub { result, .. }) = use_owner {
                sub_result = Some(*result);
            } else if !self
                .marks
                .has(Mark::LhsInPredReplacedByIterArg, use_path.path())
                && !self.marks.has(Mark::ToDelete, use_path.path())
                && !self.check_cascading_arg_uses(root, &use_path, cur, rest)
            {
                return false;
            }
        }
        let Some(sub_result) = sub_result else {
            return true;
        };
        uses_of(sub_result, root).into_iter().all(|use_path| {
            self.marks.has(Mark::ToDelete, use_path.path())
                || self.check_cascading_arg_uses(root, &use_path, cur, rest)
        })
    }

    /// The cleanup both transforms share (`:1832-1847`, `:1904-1916`): the marked conditional, yield
    /// and predicate inside `new_if` take the iterator arguments that were just secured.
    fn finish_placeholders(
        &mut self,
        root: &mut Vec<Op>,
        for_op: &OpPath,
        is_one_monotone_seq: bool,
        synthetic_iv_replacement: Option<Val>,
        seq_replacement: Option<Val>,
    ) {
        if is_one_monotone_seq {
            if let Some(replacement) = seq_replacement {
                self.replace_if_op_by_iter_arg(root, for_op, replacement);
            }
            return;
        }
        if self.monotone_seq_start_val.is_some()
            && let Some(replacement) = seq_replacement
        {
            self.replace_result_by_iter_arg(root, for_op, replacement);
        }
        if self.ivs_dimensions_multipliers.len() > 1
            && let Some(replacement) = synthetic_iv_replacement
        {
            self.replace_predicate_iv_by_iter_arg(root, for_op, replacement);
        }
    }

    /// `new_for_op->walk(..)` (`:1854-1877`) — the clone built one level in starts its new iterator
    /// arguments from this clone's, and when the two loops are directly nested this clone's terminator
    /// yields the inner loop's matching results.
    ///
    /// ⭐ THE REFERENCE'S WALK IS A TRACKED PATH HERE: exactly one `PREVIOUS_FOR_OP_CREATED` is ever
    /// live, so searching for it and remembering where it was put are the same thing.
    fn rewire_previous_loop(
        &mut self,
        root: &mut Vec<Op>,
        prev: &OpPath,
        new_for_op: &OpPath,
        new_iter_arg: &[Val],
        body_len: usize,
    ) {
        if !self.marks.take(Mark::PreviousForOpCreated, prev.path()) {
            return;
        }
        let are_loops_consecutive = prev.parent().as_ref() == Some(new_for_op);
        let mut prev_results = Vec::new();
        if let Some(Op::Sentient(sentient::Op::For { carried, .. })) = op_at_mut(root, prev.path())
        {
            let first_new = carried.len().saturating_sub(new_iter_arg.len());
            for (entry, arg) in carried[first_new..].iter_mut().zip(new_iter_arg) {
                entry.init = *arg;
                prev_results.push(entry.result);
            }
        }
        if !are_loops_consecutive {
            return;
        }
        let Some(index) = body_len
            .checked_sub(1)
            .and_then(|last| u32::try_from(last).ok())
        else {
            return;
        };
        if let Some(Op::Sentient(sentient::Op::Yield { results })) =
            op_at_mut(root, new_for_op.child(0, index).path())
        {
            let first_new = results.len().saturating_sub(prev_results.len());
            for (slot, value) in results[first_new..].iter_mut().zip(&prev_results) {
                *slot = *value;
            }
        }
    }

    /// Replaces: e433_transformInMonotoneSequenceCase
    ///
    /// Secures one iterator argument for the monotone sequence on the free IV's loop — reusing an
    /// existing one or cloning the loop — retires the conditional in favour of `new_if`, and points
    /// the marked placeholder at that argument.
    ///
    /// TRAP: THE CAPACITY CHECK IS `+ 1`, NOT `+ num_new_iter_args` (`:1954`), and it is reached only
    /// when no existing iterator argument matched.
    pub fn transform_in_monotone_sequence_case(
        &mut self,
        root: &mut Vec<Op>,
        case: &MonotoneCase<'_>,
        cloning: &mut LoopCloning<'_>,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> bool {
        let Some(table_entry_type) = self.table.as_ref().map(Table::table_entry_type) else {
            return false;
        };
        let is_one_monotone_seq = matches!(
            self.template_sequences.as_slice(),
            [only] if only.kind == SequenceKind::MonotoneSequence
        );
        let create_new_iter_arg = self.monotone_seq_start_val.is_some();
        let mut seq_iter_arg = None;
        if let Some(start_val) = self.monotone_seq_start_val {
            // `findExistingIterArg(rbegin(), rbegin(), ..)`: the free IV alone, so `rest` is empty.
            let Some(innermost) = self.ivs_dimensions_multipliers.last().copied() else {
                return false;
            };
            let found = self.find_existing_iter_arg(
                root,
                &innermost,
                &[],
                None,
                &IterArgTarget {
                    lb: start_val,
                    step: None,
                    ty: table_entry_type,
                },
                None,
            );
            let Some(carried) = carried_of_for(root, case.for_op) else {
                return false;
            };
            match found {
                Some(index) => seq_iter_arg = carried.get(index).map(|entry| entry.arg),
                None if carried.len() + 1 > MAXIMUM_ITER_ARGS => return false,
                None => {}
            }
        }

        let original_result = first_result_of_if(root, case.if_op);
        self.marks.set(Mark::ToDelete, case.if_op.path());
        if is_one_monotone_seq {
            self.marks
                .set(Mark::IfOpReplacedByIterArg, case.if_op.path());
        } else if let Some(result) = original_result {
            ir::replace_all_uses_with(root, result, case.new_if);
        }
        if !create_new_iter_arg {
            return true;
        }

        let mut for_op_containing_iter_arg = case.for_op.clone();
        if seq_iter_arg.is_none() {
            let Some(start_val) = self.monotone_seq_start_val else {
                return true;
            };
            // The step is either a value a previous nested conditional built, or an evaluated one.
            let int_step = self.monotone_seq_int_step;
            let Some(step) = self.monotone_seq_val_step.or_else(|| {
                int_step.map(|step| evaluator.build_offset_value(step, table_entry_type, builders))
            }) else {
                return false;
            };
            for_op_containing_iter_arg = (cloning.clone)(root, case.for_op, &[(start_val, step)]);
            seq_iter_arg = carried_of_for(root, &for_op_containing_iter_arg)
                .and_then(|carried| carried.last().map(|entry| entry.arg));
        }
        if let Some(replacement) = seq_iter_arg {
            if is_one_monotone_seq {
                self.replace_if_op_by_iter_arg(root, &for_op_containing_iter_arg, replacement);
            } else {
                self.replace_result_by_iter_arg(root, &for_op_containing_iter_arg, replacement);
            }
        }
        true
    }

    /// `table_->getTableEntryAtIdx(index)->getEV()`, absent where the reference's `DT_CHECK_MSG`
    /// (`:2187`) or `getEV`'s own on a leaf that yields nothing (`:474`, `:469-471`) would fire.
    fn table_ev(&self, at: TableIndex) -> Option<EvaluatedValueId> {
        self.table.as_ref()?.table_entry_at_idx(at)?.ev()
    }

    /// Replaces: e434_parseSequence
    ///
    /// Walks one table slice front to back, opening the first sequence from the first two or three
    /// entries and then folding each later entry into the last sequence or starting a new one.
    ///
    /// TRAP: `Value &iv` IS UNUSED IN THE REFERENCE BODY and is dropped. Every `EvaluatedValue`
    /// comparison is CONTENT equality, hence [`ExpressionEvaluator::equal`] and not `==`.
    ///
    /// TRAP: THE TWO DIFFERENCES ARE INTERNED IN THE REFERENCE'S OWN ORDER (`:2256-2257`) and inside
    /// the arm that needs them — hoisting them would change what the evaluator has seen.
    pub fn parse_sequence(
        &mut self,
        iv_cur_val: IntervalMarker,
        iv_step: IntervalStride,
        index: TableIndex,
        length_remaining: Entries,
        table_slice: &mut TableSlice,
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        if length_remaining.0 == 0 || self.abort_pattern {
            return;
        }
        let idx_stride = table_slice.idx_stride.0;
        let next_index = TableIndex(index.0 + idx_stride);
        let Some(val) = self.table_ev(index) else {
            return;
        };
        let one_on = advanced(iv_cur_val, iv_step, 1);
        if table_slice.sequences.is_empty() {
            if length_remaining.0 == 1 {
                // Pattern: D
                let seq = const_stride_sequence(
                    evaluator,
                    val,
                    0,
                    Entries(1),
                    iv_cur_val,
                    iv_step,
                    SequenceKind::DefaultValue,
                );
                self.insert_sequence(seq, table_slice);
                return;
            }
            let Some(next_val) = self.table_ev(next_index) else {
                return;
            };
            if length_remaining.0 == 2 {
                if evaluator.equal(next_val, val) {
                    // Pattern: D D
                    let seq = const_stride_sequence(
                        evaluator,
                        val,
                        0,
                        Entries(2),
                        one_on,
                        iv_step,
                        SequenceKind::DefaultValue,
                    );
                    self.insert_sequence(seq, table_slice);
                } else if ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS {
                    // Pattern: a1 a2
                    let stride = evaluator.evaluate_sub(next_val, val);
                    let seq = Sequence::new(
                        val,
                        stride,
                        Entries(2),
                        one_on,
                        iv_step,
                        SequenceKind::MonotoneSequence,
                    );
                    self.insert_sequence(seq, table_slice);
                } else {
                    // Pattern: D D'
                    let head = const_stride_sequence(
                        evaluator,
                        val,
                        0,
                        Entries(1),
                        iv_cur_val,
                        iv_step,
                        SequenceKind::DefaultValue,
                    );
                    self.insert_sequence(head, table_slice);
                    let tail = const_stride_sequence(
                        evaluator,
                        next_val,
                        0,
                        Entries(1),
                        one_on,
                        iv_step,
                        SequenceKind::DefaultValue,
                    );
                    self.insert_sequence(tail, table_slice);
                }
                return;
            }
            let next_next_index = TableIndex(next_index.0 + idx_stride);
            let Some(next_next_val) = self.table_ev(next_next_index) else {
                return;
            };
            let two_on = advanced(iv_cur_val, iv_step, 2);
            if evaluator.equal(val, next_val) && evaluator.equal(next_val, next_next_val) {
                // Pattern: D D D
                let seq = const_stride_sequence(
                    evaluator,
                    val,
                    0,
                    Entries(3),
                    two_on,
                    iv_step,
                    SequenceKind::DefaultValue,
                );
                self.insert_sequence(seq, table_slice);
            } else if evaluator.equal(val, next_val) {
                // Pattern: D D a1, or D D D'
                let head = const_stride_sequence(
                    evaluator,
                    val,
                    0,
                    Entries(2),
                    one_on,
                    iv_step,
                    SequenceKind::DefaultValue,
                );
                self.insert_sequence(head, table_slice);
                let kind = if ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS {
                    SequenceKind::MonotoneSequence
                } else {
                    SequenceKind::DefaultValue
                };
                let tail = const_stride_sequence(
                    evaluator,
                    next_next_val,
                    0,
                    Entries(1),
                    two_on,
                    iv_step,
                    kind,
                );
                self.insert_sequence(tail, table_slice);
            } else {
                let second_difference = evaluator.evaluate_sub(next_next_val, next_val);
                let first_difference = evaluator.evaluate_sub(next_val, val);
                if evaluator.equal(second_difference, first_difference)
                    && ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS
                {
                    // Pattern: a1 a2 a3
                    let seq = Sequence::new(
                        val,
                        first_difference,
                        Entries(3),
                        two_on,
                        iv_step,
                        SequenceKind::MonotoneSequence,
                    );
                    self.insert_sequence(seq, table_slice);
                } else {
                    // Conservatively, the pattern: D a1 a2
                    let head = const_stride_sequence(
                        evaluator,
                        val,
                        0,
                        Entries(1),
                        iv_cur_val,
                        iv_step,
                        SequenceKind::DefaultValue,
                    );
                    self.insert_sequence(head, table_slice);
                    if ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS {
                        let seq = Sequence::new(
                            next_val,
                            second_difference,
                            Entries(2),
                            two_on,
                            iv_step,
                            SequenceKind::MonotoneSequence,
                        );
                        self.insert_sequence(seq, table_slice);
                    } else if evaluator.equal(next_val, next_next_val) {
                        // Pattern: D D' D'
                        let seq = const_stride_sequence(
                            evaluator,
                            next_val,
                            0,
                            Entries(2),
                            two_on,
                            iv_step,
                            SequenceKind::DefaultValue,
                        );
                        self.insert_sequence(seq, table_slice);
                    } else {
                        // Pattern: D D' D''
                        let first = const_stride_sequence(
                            evaluator,
                            next_val,
                            0,
                            Entries(1),
                            one_on,
                            iv_step,
                            SequenceKind::DefaultValue,
                        );
                        self.insert_sequence(first, table_slice);
                        let second = const_stride_sequence(
                            evaluator,
                            next_next_val,
                            0,
                            Entries(1),
                            two_on,
                            iv_step,
                            SequenceKind::DefaultValue,
                        );
                        self.insert_sequence(second, table_slice);
                    }
                }
            }
            table_slice.set_prev_val(next_next_val);
            self.parse_sequence(
                two_on,
                iv_step,
                TableIndex(next_next_index.0 + idx_stride),
                Entries(length_remaining.0 - 3),
                table_slice,
                evaluator,
            );
            return;
        }

        let prev_val = table_slice.prev_val();
        let difference = evaluator.evaluate_sub(val, prev_val);
        let Some(cur_seq) = table_slice.sequences.last() else {
            return;
        };
        let cur_seq_length = cur_seq.length;
        let cur_seq_kind = cur_seq.kind;
        let stride_matches = cur_seq
            .stride
            .is_some_and(|stride| evaluator.equal(stride, difference));
        let has_monotone_before_last = table_slice.has_inserted_monotone_seq_before_last;
        let step = if cur_seq_length.0 > 1 {
            if stride_matches {
                SequenceStep::Extend
            } else {
                SequenceStep::Start
            }
        } else {
            let zero = evaluator.get_constant(0);
            let difference_is_zero = evaluator.equal(difference, zero);
            if !difference_is_zero
                && (!ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS || has_monotone_before_last)
            {
                SequenceStep::Start
            } else {
                SequenceStep::Greedy { difference_is_zero }
            }
        };
        match step {
            SequenceStep::Extend => extend_last_sequence(table_slice),
            SequenceStep::Start => {
                // A value after a monotone sequence is always treated as a default value.
                let new_kind = if cur_seq_kind == SequenceKind::MonotoneSequence
                    || has_monotone_before_last
                    || !ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS
                {
                    SequenceKind::DefaultValue
                } else {
                    SequenceKind::MonotoneSequence
                };
                let seq =
                    const_stride_sequence(evaluator, val, 0, Entries(1), one_on, iv_step, new_kind);
                self.insert_sequence(seq, table_slice);
            }
            SequenceStep::Greedy { difference_is_zero } => {
                extend_last_sequence(table_slice);
                if let Some(seq) = table_slice.sequences.last_mut() {
                    seq.stride = Some(difference);
                    seq.kind = if difference_is_zero {
                        SequenceKind::DefaultValue
                    } else {
                        SequenceKind::MonotoneSequence
                    };
                }
            }
        }
        table_slice.set_prev_val(val);
        self.parse_sequence(
            one_on,
            iv_step,
            next_index,
            Entries(length_remaining.0 - 1),
            table_slice,
            evaluator,
        );
    }

    /// Replaces: e435_codeGenGenericNestedIf
    ///
    /// Builds `if (i1 == c1) {..} else if (i1 == c2) {..} else {..}` over every fixed IV, each leaf
    /// holding one table slice's code, and answers the outermost chain's results.
    ///
    /// TRAP: `is_start`, `index` AND `multiplier` ARE ALL DEAD in the reference — `!is_start && i != 0`
    /// is `i != 0` at every call site, and `index` is only ever advanced, never read.
    ///
    /// TRAP: A DIMENSION OF ONE leaves the reference's `result` null, so this answers no results while
    /// still emitting the `else` recursion's ops and the yield over them.
    pub fn code_gen_generic_nested_if(
        &self,
        nest: &mut GenericNest<'_, '_>,
        ivs: &[IvDim],
        at: &OpPath,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> (Vec<Op>, Vec<Val>) {
        // `it_cur == it_free_dimension`: the free IV is the last, so one IV left IS the leaf.
        if ivs.len() <= 1 {
            let (Some(free), Some(slice)) = (ivs.last(), nest.slices.next()) else {
                return (Vec::new(), Vec::new());
            };
            if nest.generate_encoding {
                return (
                    Vec::new(),
                    self.generate_encoding_tuple(slice, builders, evaluator),
                );
            }
            let mut ops = Vec::new();
            let mut site = Site {
                block: &mut ops,
                at: at.clone(),
            };
            let result =
                self.code_gen_generic_table_slice(free.iv, slice, &mut site, builders, evaluator);
            return (ops, result.into_iter().collect());
        }
        let Some((cur, rest)) = ivs.split_first() else {
            return (Vec::new(), Vec::new());
        };
        // `DT_CHECK_MSG(step != 0, "Expect non-zero stride.")` (`:2586`) is the loop's own type.
        let Some(info) = self.lhs_to_for_op_or_null.get(&cur.iv) else {
            return (Vec::new(), Vec::new());
        };
        let (lb, step) = (info.lb, info.step.get());
        let Some(predicate_type) = self.table.as_ref().map(Table::predicate_type) else {
            return (Vec::new(), Vec::new());
        };

        let level_count = usize::try_from(cur.dimension.saturating_sub(1)).unwrap_or(0);
        let mut levels: Vec<NestedLevel> = Vec::new();
        for depth in 0..level_count {
            let rhs = builders.values.mint();
            let value = lb + i64::try_from(depth).unwrap_or(0) * step;
            builders
                .consts
                .push(scalar_constant(rhs, value, predicate_type));
            let results: Vec<Val> = nest.types.iter().map(|_| builders.values.mint()).collect();
            let (then_ops, then_results) = self.code_gen_generic_nested_if(
                nest,
                rest,
                &level_path(at, depth).child(0, 0),
                builders,
                evaluator,
            );
            levels.push(NestedLevel {
                rhs,
                results,
                then_ops,
                then_results,
            });
        }
        // The last value of this IV falls under the innermost "else" (`:2628-2633`).
        let (else_ops, else_results) = self.code_gen_generic_nested_if(
            nest,
            rest,
            &level_path(at, level_count),
            builders,
            evaluator,
        );
        let mut built = else_ops;
        built.push(yield_of(&else_results));
        let mut outer_results = Vec::new();
        for (depth, level) in levels.into_iter().enumerate().rev() {
            let mut then_body = level.then_ops;
            then_body.push(yield_of(&level.then_results));
            let if_op = Op::Sentient(sentient::Op::If {
                predicate: sentient::CmpPredicate::Eq,
                lhs: cur.iv,
                rhs: level.rhs,
                yielded: level
                    .results
                    .iter()
                    .map(|&result| sentient::Yielded {
                        result,
                        reg: UNKNOWN_LOCALE,
                        element_size: None,
                    })
                    .collect(),
                dbg_name: None,
                then_body,
                else_body: built,
            });
            if depth == 0 {
                // The caller creates the yield over the outermost conditional (`:2611-2613`).
                built = vec![if_op];
                outer_results = level.results;
            } else {
                built = vec![if_op, yield_of(&level.results)];
            }
        }
        (built, outer_results)
    }
}

/// `if_op->getResult(0)` — the value a `sentient.if` binds first, if it binds one.
fn first_result_of_if(root: &[Op], if_op: &OpPath) -> Option<Val> {
    let Some(Op::Sentient(sentient::Op::If { yielded, .. })) = op_at(root, if_op.path()) else {
        return None;
    };
    yielded.first().map(|entry| entry.result)
}

/// `sentient_for.getRegionIterArgs()` and its inits, cloned because the callers below go on to mutate
/// the tree they came from.
fn carried_of_for(root: &[Op], for_op: &OpPath) -> Option<Vec<sentient::Carried>> {
    let Some(Op::Sentient(sentient::Op::For { carried, .. })) = op_at(root, for_op.path()) else {
        return None;
    };
    Some(carried.clone())
}

/// [`carried_of_for`] plus the body length, which names the loop's terminator.
fn carried_and_body_len(root: &[Op], for_op: &OpPath) -> Option<(Vec<sentient::Carried>, usize)> {
    let Some(Op::Sentient(sentient::Op::For { carried, body, .. })) = op_at(root, for_op.path())
    else {
        return None;
    };
    Some((carried.clone(), body.len()))
}

impl PatternSimplificationManager {
    /// Replaces: e496_parseFixedDims
    ///
    /// Enumerates every tuple of the FIXED IVs and parses one [`TableSlice`] per leaf, recording
    /// whether the slices agree in sequence count and kind (`:2021-2084`).
    ///
    /// TRAP: THE COUNT MISMATCH DOES NOT `break` AND THE KIND MISMATCH DOES (`:2053`, `:2069`).
    /// TRAP: A MISSING `lhs_to_for_op_or_null_` ENTRY IS `DT_CHECK_MSG(step != 0, ..)` (`:2038-2039`) —
    /// the reference reads the map with `operator[]`, which default-constructs step 0.
    pub fn parse_fixed_dims(
        &mut self,
        fixed: &[IvDim],
        free_dimension: IvDim,
        index: TableIndex,
        table_slices: &mut Vec<TableSlice>,
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        let Some((cur, rest)) = fixed.split_first() else {
            // `it_cur == it_free_dimension` — reached a leaf.
            let Some((lb, step)) = self
                .lhs_to_for_op_or_null
                .get(&free_dimension.iv)
                .map(|info| (info.lb, info.step))
            else {
                panic!(
                    "DT_CHECK(step != 0) Expect non-zero stride. \
                     (`CFGSimplificationSentientLevel.cpp:2038`): the free dimension \
                     {free_dimension:?} is the IV of no recorded loop"
                )
            };
            let zero = evaluator.get_constant(0);
            let length = Entries(free_dimension.dimension);
            let mut table_slice = TableSlice::new(
                index,
                IndexStride(free_dimension.multiplier),
                length,
                zero,
                /* is_1d_rep_of_table */ false,
            );
            self.parse_sequence(
                IntervalMarker(lb),
                IntervalStride::new(step),
                index,
                length,
                &mut table_slice,
                evaluator,
            );
            if self.table_slices_are_consistent
                && let Some(first) = table_slices.first()
            {
                if table_slice.sequences.len() != first.sequences.len() {
                    self.table_slices_are_consistent = false;
                }
                for (cur_seq, first_seq) in table_slice.sequences.iter().zip(&first.sequences) {
                    if cur_seq.kind != first_seq.kind {
                        self.table_slices_are_consistent = false;
                        break;
                    }
                }
            }
            table_slices.push(table_slice);
            return;
        };
        let mut index = index;
        for _ in 0..cur.dimension {
            self.parse_fixed_dims(rest, free_dimension, index, table_slices, evaluator);
            index = TableIndex(index.0 + cur.multiplier);
        }
    }
}

// ── THE LEVEL-4 DRIVER, AND WHAT AN INSERTION COSTS A TREE NAMED BY POSITION ─────────────────────

/// WHAT `findPatternsAndSimplify` ASKS OF THE CONDITIONAL TREE — `CFGSSentientLevelConditionalTree
/// &tree` and the three questions it puts to `n` (`:1101-1102`).
///
/// ⛔ `Analyses/CFGSSentientLevelConditionalTree` IS OUT OF CAMPAIGN SCOPE, so each answer arrives as
/// a parameter, exactly as [`LeafSink`] and [`LoopCloning`] do.
///
/// ⛔ `simplify_subtree` MUST CONFINE ITS REWRITE TO THAT OP'S OWN REGIONS. It is
/// `subtree(*op); compute(); removeConditionWhenThenElseBranchesMatch()` followed by
/// `getRoot()->getNumLeavesInSubtree()` (`:1312-1315`, `:1401-1404`), and a rewrite reaching the op's
/// SIBLINGS would move ops whose position this driver has already recorded.
pub struct TreeSeam<'a> {
    /// `n->getNumLeavesInSubtree()`, the cost every case is compared against.
    pub num_leaves_in_subtree: i64,
    /// `n->setNoCandidatesForAllSubtreeNodes()`.
    pub set_no_candidates: &'a mut dyn FnMut(),
    /// The subtree simplification above, answering the leaves its root has left.
    pub simplify_subtree: &'a mut dyn FnMut(&mut Vec<Op>, &OpPath) -> i64,
}

/// WHAT AN INSERTION DOES TO EVERY OTHER POSITION IN ITS BLOCK.
///
/// ⭐ THE REFERENCE PAYS NOTHING HERE: an `Operation *` survives a sibling insertion, so `:1177`'s
/// `OpBuilder builder_new_if(if_op)` leaves `if_op`, `for_op` and every marked op still named. In this
/// island each of those is a POSITION, and every position at or after the insertion point has moved.
struct Shift {
    /// The op whose region was inserted into, empty for the top-level body.
    under: OpPath,
    /// Which of its regions.
    region: u32,
    /// The position inserted at — every sibling from here on moves.
    from: i64,
    /// How far, negative for an erase.
    by: i64,
}

impl Shift {
    /// What inserting `by` ops in front of the op at `at` does — `OpBuilder builder(op)`.
    fn before(at: &OpPath, by: i64) -> Option<Shift> {
        let (&(region, index), under) = at.path().split_last()?;
        Some(Shift {
            under: OpPath::at(under),
            region,
            from: i64::from(index),
            by,
        })
    }

    /// The shift that undoes it — what erasing those ops again does (`:1242`, `:1438`).
    fn undo(&self) -> Shift {
        Shift {
            under: self.under.clone(),
            region: self.region,
            from: self.from + self.by,
            by: -self.by,
        }
    }

    /// One recorded position, after that insertion.
    fn apply(&self, path: &OpPath) -> OpPath {
        if !path.path().starts_with(self.under.path()) {
            return path.clone();
        }
        let depth = self.under.path().len();
        let mut steps = path.path().to_vec();
        let Some(step) = steps.get_mut(depth) else {
            return path.clone();
        };
        if step.0 != self.region || i64::from(step.1) < self.from {
            return path.clone();
        }
        step.1 = u32::try_from(i64::from(step.1) + self.by).unwrap_or(step.1);
        OpPath::at(&steps)
    }
}

impl Marks {
    /// Every mark re-keyed for an insertion, the ops they name having moved.
    fn shift(&mut self, shift: &Shift) {
        self.on = core::mem::take(&mut self.on)
            .into_iter()
            .map(|(mark, at)| (mark, shift.apply(&at)))
            .collect();
    }

    /// Every mark on `at` and on the ops inside it forgotten — what `erase()` costs the pass-local
    /// attributes of the ops it deletes.
    fn drop_under(&mut self, at: &OpPath) {
        self.on
            .retain(|(_, marked)| !marked.path().starts_with(at.path()));
    }
}

/// THE BLOCK ONE OP SITS IN, AND WHERE IN IT — what an `OpBuilder(op)` insertion point IS here, since
/// this island's blocks are `Vec<Op>` and not a linked list an iterator can name a place in.
fn block_of_mut<'a>(root: &'a mut Vec<Op>, at: &OpPath) -> Option<(&'a mut Vec<Op>, usize)> {
    let (&(region, index), under) = at.path().split_last()?;
    if under.is_empty() {
        return Some((root, index as usize));
    }
    let parent = op_at_mut(root, under)?;
    let block = ir::regions_mut(parent).into_iter().nth(region as usize)?;
    Some((block, index as usize))
}

/// `builder.insert(..)` for a run of ops in front of the op at `at`, answering what it moved.
fn insert_before(root: &mut Vec<Op>, at: &OpPath, ops: Vec<Op>) -> Option<Shift> {
    let shift = Shift::before(at, i64::try_from(ops.len()).unwrap_or(0))?;
    let (block, index) = block_of_mut(root, at)?;
    let index = index.min(block.len());
    for (offset, op) in ops.into_iter().enumerate() {
        block.insert(index + offset, op);
    }
    Some(shift)
}

/// `DT_CHECK_MSG(new_if.getDefiningOp()->use_empty(), "Expect new conditional to have no uses
/// yet."); new_if.getDefiningOp()->erase()` (`:1240-1242`, `:1436-1438`) over the run
/// [`insert_before`] put there.
fn erase_unused(root: &mut Vec<Op>, at: &OpPath, count: usize, cite: &str) {
    if let Some(op) = op_at(root, at.path())
        && !use_empty(op, root)
    {
        panic!(
            "DT_CHECK(new_if.getDefiningOp()->use_empty()) Expect new conditional to have no uses \
             yet. (`CFGSimplificationSentientLevel.cpp:{cite}`): {at:?} is still read"
        )
    }
    if let Some((block, index)) = block_of_mut(root, at) {
        for _ in 0..count.min(block.len().saturating_sub(index)) {
            block.remove(index);
        }
    }
}

/// `getNewDbgNameFromOp(prefix, op, suffix)` (`Utils/Utils.cpp:492-501`) — `None` when that op carries
/// no name of its own, which is what makes every caller's `if` skip the rename.
fn new_dbg_name_from_op(root: &[Op], prefix: &str, at: &OpPath, suffix: &str) -> Option<String> {
    let Some(Op::Sentient(sentient::Op::If { dbg_name, .. })) = op_at(root, at.path()) else {
        return None;
    };
    dbg_name
        .as_ref()
        .map(|name| format!("{prefix}{name}{suffix}"))
}

/// `dataflow::setDbgNameAttr(op, name)`, on an op with a place to keep one.
fn set_dbg_name(root: &mut Vec<Op>, at: &OpPath, name: String) {
    if let Some(Op::Sentient(inner)) = op_at_mut(root, at.path())
        && let Some(slot) = sentient::dbg_name_mut(inner)
    {
        *slot = Some(name);
    }
}

impl PatternSimplificationManager {
    /// Every position this manager has recorded, moved by one insertion — the marks, and the loops
    /// `lhs_to_for_op_or_null_` names.
    fn shift_records(&mut self, shift: &Shift) {
        self.marks.shift(shift);
        for info in self.lhs_to_for_op_or_null.values_mut() {
            if let Some(for_op) = info.for_op.as_ref() {
                info.for_op = Some(shift.apply(for_op));
            }
        }
    }

    /// [`Self::shift_records`] for an insertion a code-gen call made: the marks that call has just set
    /// name the ops it BUILT, which are already where they landed, so only the inherited ones move.
    fn shift_over_generated(&mut self, before: &Marks, shift: &Shift) {
        let added: Vec<(Mark, OpPath)> = self.marks.on.difference(&before.on).cloned().collect();
        self.marks = before.clone();
        self.shift_records(shift);
        for (mark, at) in added {
            self.marks.set(mark, at.path());
        }
    }

    /// Replaces: e555_findPatternsAndSimplify
    ///
    /// Reads the table as one contiguous 1-D slice and, in the reference's order, replaces the
    /// conditional by the single value every branch yields, by a chain over the free IV (case 1), by
    /// an encoding conditional outside the free IV's loop plus that chain (case 2), or by the same
    /// nest with consecutive duplicates compressed (case 3) — each kept only if it leaves fewer leaves.
    ///
    /// TRAP: `generateEncodingTypes()`'s RESULT IS UNUSED IN CASE 1 (`:1181`) and the call is made all
    /// the same. Every insertion moves its siblings, hence [`Shift`]; a code-gen call that builds
    /// nothing leaves the reference's `results[0].getDefiningOp()` on null, which stops that case.
    pub fn find_patterns_and_simplify(
        &mut self,
        root: &mut Vec<Op>,
        n: &CondNode,
        tree: &mut TreeSeam<'_>,
        cloning: &mut LoopCloning<'_>,
        builders: &mut Builders<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> bool {
        self.abort_pattern = false;
        let Some(table) = self.table.as_ref() else {
            return false;
        };
        let table_size = i64::try_from(table.size().0).unwrap_or(i64::MAX);
        let table_entry_type = table.table_entry_type();
        // `iv.getType()` (`:2737`) — the table's predicate type IS `n->getLhs().getType()` (`:839`),
        // and the free IV is that LHS.
        let predicate_type = table.predicate_type();
        let ivs = self.ivs_dimensions_multipliers.clone();
        // `std::prev(ivs_dimensions_multipliers_.end(), 1)` — the free IV (`:1111-1112`).
        let Some(&free) = ivs.last() else {
            return false;
        };
        let free_iv = TypedVal {
            val: free.iv,
            ty: predicate_type,
        };
        let zero = evaluator.get_constant(0);
        let mut contiguous = TableSlice::new(
            TableIndex(0),
            IndexStride(1),
            Entries(table_size),
            zero,
            /* is_1d_rep_of_table */ true,
        );
        // `DT_CHECK_MSG(free_iv_step != 0, ..)` (`:1117`): the reference reads the map with
        // `operator[]`, which default-constructs step 0 for an IV it has no loop for.
        let Some(free_loop) = self.lhs_to_for_op_or_null.get(&free.iv).cloned() else {
            panic!(
                "DT_CHECK(free_iv_step != 0) Expect non-zero stride. \
                 (`CFGSimplificationSentientLevel.cpp:1117`): the free IV {:?} is the IV of no \
                 recorded loop",
                free.iv
            )
        };
        for record in &ivs {
            let Some(info) = self.lhs_to_for_op_or_null.get(&record.iv) else {
                panic!(
                    "DT_CHECK(iv_step != 0) Expect non-zero stride. \
                     (`CFGSimplificationSentientLevel.cpp:1126`): the IV {:?} is the IV of no \
                     recorded loop",
                    record.iv
                )
            };
            if free_loop.step != info.step {
                self.table_follows_contiguous_pattern = false;
                break;
            }
        }
        let has_positive_step = free_loop.step.get() > 0;
        let table_lb = if ivs.len() == 1 {
            free_loop.lb
        } else if has_positive_step {
            0
        } else {
            table_size - 1
        };
        self.parse_sequence(
            IntervalMarker(table_lb),
            IntervalStride::new(free_loop.step),
            TableIndex(0),
            Entries(table_size),
            &mut contiguous,
            evaluator,
        );
        let mut if_op = n.op.clone();

        // "If all the branches yield the same value" (`:1153-1171`).
        if let [only] = contiguous.sequences.as_slice()
            && only.kind == SequenceKind::DefaultValue
        {
            // `*getSequences().front()->getLB()` (`:1157`) is a null deref without an LB.
            let Some(common_value) = only.lb else {
                return false;
            };
            let replacement =
                evaluator.build_offset_value(common_value, table_entry_type, builders);
            if let Some(result) = first_result_of_if(root, &if_op) {
                ir::replace_all_uses_with(root, result, replacement);
            }
            self.marks.set(Mark::ToDelete, if_op.path());
            (tree.set_no_candidates)();
            return false;
        }

        // There is only one table slice in case 1 (`:1150`).
        let mut table_slices = vec![contiguous];
        if self.table_follows_contiguous_pattern {
            // CASE 1: contiguous pattern (`:1172-1242`).
            self.compute_template_seq_and_update_monotone_seq(&mut table_slices, evaluator);
            let _unused_types = self.generate_encoding_types();
            let sequence_encoding = match table_slices.first() {
                Some(slice) => self.generate_encoding_tuple(slice, builders, evaluator),
                None => Vec::new(),
            };
            let before = self.marks.clone();
            let new_if = match block_of_mut(root, &if_op) {
                Some((block, _)) => {
                    let mut site = Site {
                        block,
                        at: if_op.clone(),
                    };
                    self.code_gen_monotone_table_slice(
                        /* mark_new_cmp */ true,
                        free_iv,
                        &sequence_encoding,
                        &mut site,
                        builders,
                        evaluator,
                    )
                }
                None => None,
            };
            if let Some(new_if) = new_if
                && let Some(shift) = Shift::before(&if_op, 1)
            {
                // The chain landed AT `if_op`, so the conditional it replaces moved down one.
                let new_if_op = if_op.clone();
                self.shift_over_generated(&before, &shift);
                if_op = shift.apply(&if_op);
                let viable = tree.num_leaves_in_subtree
                    > i64::try_from(self.template_sequences.len()).unwrap_or(i64::MAX);
                // One new iterator argument for the synthetic 1-D variable unless the table already
                // is 1-D, and one for the monotone sequence if there is one (`:1215-1222`).
                let num_new_iter_args = usize::from(ivs.len() > 1)
                    + usize::from(self.monotone_seq_start_val.is_some());
                if viable && !self.abort_pattern {
                    // Named before the clone below moves `if_op` again (`:1226-1231`).
                    if let Some(name) = new_dbg_name_from_op(root, "CFGSimpl(", &if_op, ", case1)")
                    {
                        set_dbg_name(root, &new_if_op, name);
                    }
                    if self.transform_in_contiguous_sequence_case(
                        root,
                        &ContiguousCase {
                            if_op: &if_op,
                            new_if,
                            num_new_iter_args,
                            has_positive_step,
                        },
                        cloning,
                        builders,
                        evaluator,
                    ) {
                        (tree.set_no_candidates)();
                        return true;
                    }
                }
                erase_unused(root, &new_if_op, 1, "1240");
                self.marks.drop_under(&new_if_op);
                let undo = shift.undo();
                self.shift_records(&undo);
                if_op = undo.apply(&if_op);
            }
        }

        // Recompute the table slices, no longer treating the table as one contiguous slice, if the
        // table is not 1-D (`:1244-1255`).
        self.abort_pattern = false;
        self.template_sequences.clear();
        if ivs.len() > 1 {
            table_slices.clear();
            self.parse_fixed_dims(
                &ivs[..ivs.len() - 1],
                free,
                TableIndex(0),
                &mut table_slices,
                evaluator,
            );
        }

        // CASE 2: monotone pattern — one slice per tuple of the fixed IVs (`:1261-1370`).
        if ivs.len() > 1 && self.table_slices_follow_montone_pattern {
            if !self.table_slices_are_consistent {
                let max_num_seq = table_slices
                    .iter()
                    .map(|slice| slice.sequences.len())
                    .max()
                    .unwrap_or(0);
                for slice in &mut table_slices {
                    PatternSimplificationManager::pad_table_slice(slice, max_num_seq, evaluator);
                }
            }
            self.compute_template_seq_and_update_monotone_seq(&mut table_slices, evaluator);
            let Some(mut for_op) = self
                .lhs_to_for_op_or_null
                .get(&free.iv)
                .and_then(|info| info.for_op.clone())
            else {
                panic!(
                    "DT_CHECK(for_op) Expect valid for op. \
                     (`CFGSimplificationSentientLevel.cpp:1291`): the free IV {:?} has no recorded \
                     loop",
                    free.iv
                )
            };
            let generated_if_types = self.generate_encoding_types();
            // The (n-1)-dimensional conditional yielding each slice's encoding, placed OUTSIDE the
            // free IV's loop because its results are read inside it (`:1292-1305`).
            let (ops, results_of_reduced_if) = {
                let mut nest = GenericNest {
                    generate_encoding: true,
                    types: &generated_if_types,
                    slices: table_slices.iter(),
                };
                self.code_gen_generic_nested_if(&mut nest, &ivs, &for_op, builders, evaluator)
            };
            // `results_of_reduced_if[0].getDefiningOp()` (`:1306`) is a null deref where the nest
            // built no conditional at all.
            if !ops.is_empty()
                && !results_of_reduced_if.is_empty()
                && let Some(shift) = insert_before(root, &for_op, ops)
            {
                let mut reduced_if_op = for_op.clone();
                self.shift_records(&shift);
                if_op = shift.apply(&if_op);
                for_op = shift.apply(&for_op);
                self.encoding_if_op_count =
                    EncodingIfOpCount(self.encoding_if_op_count.0.saturating_add(1));
                set_dbg_name(
                    root,
                    &reduced_if_op,
                    format!("CFGSimpl enc #{}", self.encoding_if_op_count.0),
                );
                let subtree_leaves = (tree.simplify_subtree)(root, &reduced_if_op);
                let before = self.marks.clone();
                let new_if = match block_of_mut(root, &if_op) {
                    Some((block, _)) => {
                        let mut site = Site {
                            block,
                            at: if_op.clone(),
                        };
                        self.code_gen_monotone_table_slice(
                            /* mark_new_cmp */ false,
                            free_iv,
                            &results_of_reduced_if,
                            &mut site,
                            builders,
                            evaluator,
                        )
                    }
                    None => None,
                };
                if let Some(new_if) = new_if
                    && let Some(shift) = Shift::before(&if_op, 1)
                {
                    let new_if_op = if_op.clone();
                    self.shift_over_generated(&before, &shift);
                    if_op = shift.apply(&if_op);
                    for_op = shift.apply(&for_op);
                    reduced_if_op = shift.apply(&reduced_if_op);
                    let new_num_leaves = subtree_leaves
                        + i64::try_from(self.template_sequences.len()).unwrap_or(i64::MAX);
                    if tree.num_leaves_in_subtree > new_num_leaves && !self.abort_pattern {
                        if let Some(name) =
                            new_dbg_name_from_op(root, "CFGSimpl(", &if_op, ", case2)")
                        {
                            set_dbg_name(root, &new_if_op, name);
                        }
                        if self.transform_in_monotone_sequence_case(
                            root,
                            &MonotoneCase {
                                for_op: &for_op,
                                if_op: &if_op,
                                new_if,
                            },
                            cloning,
                            builders,
                            evaluator,
                        ) {
                            (tree.set_no_candidates)();
                            return true;
                        }
                    }
                    // Cleanup: both generated conditionals are retired (`:1366-1369`).
                    self.marks.set(Mark::ToDelete, new_if_op.path());
                    self.marks.set(Mark::ToDelete, reduced_if_op.path());
                }
            }
        }

        // CASE 3 (DEFAULT): compress consecutive duplicates in each slice (`:1372-1440`).
        for slice in &mut table_slices {
            slice.recompute_as_default_vals(evaluator);
        }
        let type_vector = [table_entry_type];
        let (ops, results_of_reduced_if) = {
            let mut nest = GenericNest {
                generate_encoding: false,
                types: &type_vector,
                slices: table_slices.iter(),
            };
            self.code_gen_generic_nested_if(&mut nest, &ivs, &if_op, builders, evaluator)
        };
        let count = ops.len();
        let (Some(&reduced), false) = (results_of_reduced_if.first(), ops.is_empty()) else {
            return false;
        };
        let Some(shift) = insert_before(root, &if_op, ops) else {
            return false;
        };
        let new_if_op = if_op.clone();
        self.shift_records(&shift);
        if_op = shift.apply(&if_op);
        let subtree_leaves = (tree.simplify_subtree)(root, &new_if_op);
        if tree.num_leaves_in_subtree > subtree_leaves {
            if let Some(name) = new_dbg_name_from_op(root, "CFGSimpl(", &if_op, ", case3)") {
                set_dbg_name(root, &new_if_op, name);
            }
            if let Some(result) = first_result_of_if(root, &if_op) {
                ir::replace_all_uses_with(root, result, reduced);
            }
            self.marks.set(Mark::ToDelete, if_op.path());
            (tree.set_no_candidates)();
        } else {
            erase_unused(root, &new_if_op, count, "1437");
            self.marks.drop_under(&new_if_op);
            self.shift_records(&shift.undo());
        }
        false
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::{IndexStride, TableIndex, TableSize};
    use super::*;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::sentient::dialects::sentient::{
        Carried, CmpPredicate, Reg, RegType, Yielded,
    };

    /// THE OUT-OF-SCOPE `ExpressionEvaluator` AS A TEST DOUBLE — an id stands for the TEXT of the
    /// expression it names, so `getConstant` is memoised the way the analysis's arena is and `equal`
    /// is the content equality `EvaluatedValue::operator==` provides.
    ///
    /// ⛔ TEST ONLY. `Analyses/ExpressionEvaluatorUtils` is outside this campaign and no
    /// implementation of [`ExpressionEvaluator`] may ship in it.
    #[derive(Default)]
    struct FakeEvaluator {
        next: u32,
        content: Vec<(EvaluatedValueId, String)>,
        /// Every id `build_offset_value` was asked to materialise, in order.
        built: Vec<EvaluatedValueId>,
    }

    impl FakeEvaluator {
        fn intern(&mut self, text: String) -> EvaluatedValueId {
            if let Some((id, _)) = self.content.iter().find(|(_, seen)| *seen == text) {
                return *id;
            }
            let id = EvaluatedValueId::new(self.next);
            self.next += 1;
            self.content.push((id, text));
            id
        }

        fn text(&self, id: EvaluatedValueId) -> String {
            self.content
                .iter()
                .find(|(seen, _)| *seen == id)
                .map_or_else(String::new, |(_, text)| text.clone())
        }
    }

    impl ExpressionEvaluator for FakeEvaluator {
        fn get_constant(&mut self, value: i64) -> EvaluatedValueId {
            self.intern(value.to_string())
        }

        fn evaluate_sub(
            &mut self,
            lhs: EvaluatedValueId,
            rhs: EvaluatedValueId,
        ) -> EvaluatedValueId {
            let text = format!("({} - {})", self.text(lhs), self.text(rhs));
            self.intern(text)
        }

        fn evaluate_multiply_by_const(
            &mut self,
            value: EvaluatedValueId,
            factor: Entries,
        ) -> EvaluatedValueId {
            let text = format!("({} * {})", self.text(value), factor.0);
            self.intern(text)
        }

        fn evaluate_sum(
            &mut self,
            lhs: EvaluatedValueId,
            rhs: EvaluatedValueId,
        ) -> EvaluatedValueId {
            let text = format!("({} + {})", self.text(lhs), self.text(rhs));
            self.intern(text)
        }

        fn evaluate_value(&mut self, value: Val) -> EvaluatedValueId {
            self.intern(format!("%{}", value.0))
        }

        fn equal(&self, lhs: EvaluatedValueId, rhs: EvaluatedValueId) -> bool {
            self.text(lhs) == self.text(rhs)
        }

        fn build_offset_value(
            &mut self,
            value: EvaluatedValueId,
            ty: ScalarTy,
            builders: &mut Builders<'_>,
        ) -> Val {
            self.built.push(value);
            let result = builders.values.mint();
            builders.query_maps.push(scalar_constant(result, 0, ty));
            result
        }
    }

    /// One sequence, its interval stride fixed at one entry per table entry.
    fn sequence(
        kind: SequenceKind,
        lb: EvaluatedValueId,
        stride: EvaluatedValueId,
        length: i64,
        marker: i64,
    ) -> Sequence {
        Sequence::new(
            lb,
            stride,
            Entries(length),
            IntervalMarker(marker),
            IntervalStride::new(ONE),
            kind,
        )
    }

    fn table_slice(
        zero: EvaluatedValueId,
        sequences: Vec<Sequence>,
        is_1d_rep_of_table: bool,
    ) -> TableSlice {
        let mut slice = TableSlice::new(
            TableIndex(0),
            IndexStride(1),
            Entries(0),
            zero,
            is_1d_rep_of_table,
        );
        slice.sequences = sequences;
        slice
    }

    /// A table whose entries are `index` and whose predicates are `i1`.
    fn table() -> Table {
        Table::new(TableSize(3), ScalarTy::Int(1), ScalarTy::Index)
    }

    /// `regIndex = -1`, the locale unassigned — what every op in these fixtures carries.
    const UNASSIGNED: Reg = Reg {
        locale: RegType::Unknown,
        index: None,
    };
    const ONE: NonZeroI64 = NonZeroI64::new(1).unwrap();

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn scalar_add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    fn for_op(carried: Vec<Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(1),
            bound: Val(20),
            bound_reg: None,
            carried,
            dbg_name: None,
            body,
        })
    }

    fn if_op(lhs: Val, rhs: Val, yielded: Vec<Yielded>, then_body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Eq,
            lhs,
            rhs,
            yielded,
            dbg_name: None,
            then_body,
            else_body: Vec::new(),
        })
    }

    fn loop_info(for_op: Option<OpPath>, iterations: i64) -> LoopInfo {
        LoopInfo {
            for_op,
            lb: 0,
            ub: iterations,
            step: ONE,
            iterations,
        }
    }

    /// A leaf the caller rejects marks its branch dead and the enclosing `if`'s ELSE side with it,
    /// and the IV's fixed value is restored on the way out.
    #[test]
    fn populate_table_marks_the_rejected_leaf_dead_and_restores_the_iv() {
        let iv = Val(0);
        let mut manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(iv, loop_info(None, 4))]),
            ..PatternSimplificationManager::default()
        };
        let mut node = CondNode {
            lhs: iv,
            rhs_val: 1,
            op: OpPath::at(&[(0, 0)]),
            then_node: Branch::default(),
            else_node: Branch::default(),
        };
        let mut ivs = IvValuesAndFilters::of([(iv, 4)]);
        let mut seen = Vec::new();
        // The then leaf sees `iv == 1` and is accepted; the else leaf sees no fixed value and is not.
        let mut process_leaf = |_: &Branch, table: &mut IvValuesAndFilters| {
            let chosen = table.chosen(iv);
            seen.push(chosen);
            chosen == Some(1)
        };

        manager.populate_table(&mut node, &mut ivs, &mut process_leaf);

        assert_eq!(seen, vec![Some(1), None]);
        assert!(!node.then_node.dead);
        assert!(node.else_node.dead);
        assert!(!manager.marks.has(Mark::DeadThenBranch, &[(0, 0)]));
        assert!(manager.marks.has(Mark::DeadElseBranch, &[(0, 0)]));
        assert_eq!(ivs.chosen(iv), None);
    }

    /// The loop's one iter arg already starts at the target lower bound and advances by the monotone
    /// sequence's step, so it is reused; a different step finds nothing.
    #[test]
    fn find_existing_iter_arg_matches_the_monotone_step_and_the_lower_bound() {
        let iv = Val(0);
        let root = vec![
            constant(0, Val(10)),
            constant(2, Val(11)),
            constant(0, Val(12)),
            for_op(
                vec![Carried {
                    init: Val(10),
                    arg: Val(2),
                    result: Val(3),
                    reg: UNASSIGNED,
                    program_header: false,
                    element_size: None,
                }],
                vec![scalar_add(Val(2), Val(11), Val(4)), yield_op(vec![Val(4)])],
            ),
        ];
        let mut manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(
                iv,
                loop_info(Some(OpPath::at(&[(0, 3)])), 4),
            )]),
            monotone_seq_val_step: Some(Val(11)),
            ..PatternSimplificationManager::default()
        };
        let cur = IvDim {
            iv,
            dimension: 4,
            multiplier: 1,
        };
        let target = IterArgTarget {
            lb: Val(12),
            step: None,
            ty: ScalarTy::Index,
        };

        assert_eq!(
            manager.find_existing_iter_arg(&root, &cur, &[], None, &target, None),
            Some(0)
        );

        manager.monotone_seq_val_step = Some(Val(10));
        assert_eq!(
            manager.find_existing_iter_arg(&root, &cur, &[], None, &target, None),
            None
        );
    }

    /// The marked yield hands back the iter arg, and the mark is gone.
    #[test]
    fn replace_result_by_iter_arg_rewrites_the_marked_yield() {
        let mut root = vec![for_op(Vec::new(), vec![yield_op(vec![Val(4)])])];
        let mut manager = PatternSimplificationManager::default();
        manager
            .marks
            .set(Mark::ResultReplacedByIterArg, &[(0, 0), (0, 0)]);

        manager.replace_result_by_iter_arg(&mut root, &OpPath::at(&[(0, 0)]), Val(9));

        assert_eq!(
            op_at(&root, &[(0, 0), (0, 0)]),
            Some(&yield_op(vec![Val(9)]))
        );
        assert!(
            !manager
                .marks
                .has(Mark::ResultReplacedByIterArg, &[(0, 0), (0, 0)])
        );
    }

    /// Every reader of the marked conditional's first result moves onto the iter arg.
    #[test]
    fn replace_if_op_by_iter_arg_rewires_the_readers_of_result_zero() {
        let mut root = vec![for_op(
            Vec::new(),
            vec![
                if_op(
                    Val(1),
                    Val(2),
                    vec![Yielded {
                        result: Val(5),
                        reg: UNASSIGNED,
                        element_size: None,
                    }],
                    vec![yield_op(vec![Val(6)])],
                ),
                scalar_add(Val(5), Val(6), Val(8)),
                yield_op(vec![Val(8)]),
            ],
        )];
        let mut manager = PatternSimplificationManager::default();
        manager
            .marks
            .set(Mark::IfOpReplacedByIterArg, &[(0, 0), (0, 0)]);

        manager.replace_if_op_by_iter_arg(&mut root, &OpPath::at(&[(0, 0)]), Val(9));

        assert_eq!(
            op_at(&root, &[(0, 0), (0, 1)]),
            Some(&scalar_add(Val(9), Val(6), Val(8)))
        );
    }

    /// The marked predicate tests the iter arg instead of the IV.
    #[test]
    fn replace_predicate_iv_by_iter_arg_rewrites_the_lhs() {
        let mut root = vec![for_op(
            Vec::new(),
            vec![if_op(Val(1), Val(2), Vec::new(), Vec::new())],
        )];
        let mut manager = PatternSimplificationManager::default();
        manager
            .marks
            .set(Mark::LhsInPredReplacedByIterArg, &[(0, 0), (0, 0)]);

        manager.replace_predicate_iv_by_iter_arg(&mut root, &OpPath::at(&[(0, 0)]), Val(9));

        assert_eq!(
            op_at(&root, &[(0, 0), (0, 0)]),
            Some(&if_op(Val(9), Val(2), Vec::new(), Vec::new()))
        );
    }

    /// Nested conditionals testing IV0 inside IV1 cascade; an IV no ancestor tests does not.
    #[test]
    fn check_cascading_arg_uses_walks_out_to_the_enclosing_conditional() {
        let inner = if_op(Val(0), Val(98), Vec::new(), Vec::new());
        let root = vec![if_op(Val(1), Val(99), Vec::new(), vec![inner])];
        let manager = PatternSimplificationManager::default();
        let cur = IvDim {
            iv: Val(0),
            dimension: 4,
            multiplier: 1,
        };
        let outer = IvDim {
            iv: Val(1),
            dimension: 4,
            multiplier: 4,
        };
        let absent = IvDim {
            iv: Val(42),
            dimension: 4,
            multiplier: 4,
        };
        let inner_path = OpPath::at(&[(0, 0), (0, 0)]);

        assert!(manager.check_cascading_arg_uses(&root, &inner_path, &cur, &[outer]));
        assert!(!manager.check_cascading_arg_uses(&root, &inner_path, &cur, &[absent]));
    }

    /// The template keeps only what every slice agrees on, each monotone sequence gains its shifted
    /// LB, and a step that is a constant zero over fewer than three sequences turns every monotone
    /// sequence into a default value.
    #[test]
    fn compute_template_seq_shifts_the_monotone_lb_and_collapses_a_zero_step() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let ten = evaluator.get_constant(10);
        let eleven = evaluator.get_constant(11);
        let twenty = evaluator.get_constant(20);
        let mut slices = vec![
            table_slice(
                zero,
                vec![
                    sequence(SequenceKind::DefaultValue, ten, zero, 2, 3),
                    sequence(SequenceKind::MonotoneSequence, twenty, zero, 3, 9),
                ],
                false,
            ),
            table_slice(
                zero,
                vec![
                    sequence(SequenceKind::DefaultValue, eleven, zero, 2, 3),
                    sequence(SequenceKind::MonotoneSequence, twenty, zero, 3, 9),
                ],
                false,
            ),
        ];
        let mut manager = PatternSimplificationManager::default();

        manager.compute_template_seq_and_update_monotone_seq(&mut slices, &mut evaluator);

        let scaled = evaluator.evaluate_multiply_by_const(zero, Entries(2));
        let shifted = evaluator.evaluate_sub(twenty, scaled);
        // The slices disagree about the first sequence's value, so the template forgets it and keeps
        // the interval marker they share.
        assert_eq!(manager.template_sequences[0].lb, None);
        assert_eq!(
            manager.template_sequences[0].interval_marker,
            Some(IntervalMarker(3))
        );
        // The monotone sequence became a default value whose LB is the shifted one, in the template
        // and in every slice.
        assert_eq!(
            manager.template_sequences[1].kind,
            SequenceKind::DefaultValue
        );
        assert_eq!(manager.template_sequences[1].lb, Some(shifted));
        assert_eq!(manager.template_sequences[1].shifted_lb, None);
        assert_eq!(slices[1].sequences[1].kind, SequenceKind::DefaultValue);
        assert_eq!(slices[1].sequences[1].lb, Some(shifted));
    }

    /// A fourth sequence fails the monotone pattern for a sliced table and the contiguous pattern for
    /// a 1-D one; only the 1-D failure aborts, and after it nothing is inserted at all.
    #[test]
    fn insert_sequence_caps_each_slice_at_three_and_only_a_1d_slice_aborts() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let default = sequence(SequenceKind::DefaultValue, zero, zero, 1, 0);
        let monotone = sequence(SequenceKind::MonotoneSequence, zero, zero, 1, 0);
        let mut sliced = table_slice(zero, vec![default, default, monotone], false);
        let mut one_d = table_slice(zero, vec![default, default, default], true);
        let mut manager = PatternSimplificationManager::default();

        manager.insert_sequence(default, &mut sliced);

        assert!(sliced.has_inserted_monotone_seq_before_last);
        assert!(!manager.table_slices_follow_montone_pattern);
        assert!(manager.table_follows_contiguous_pattern);
        assert!(!manager.abort_pattern);

        manager.insert_sequence(default, &mut one_d);

        assert!(!manager.table_follows_contiguous_pattern);
        assert!(manager.abort_pattern);

        manager.insert_sequence(default, &mut one_d);

        assert_eq!(one_d.sequences.len(), 4);
    }

    /// The slice is padded to `max_num_seq` default-value dummies of length zero, each repeating the
    /// last real interval marker so that its predicate is always false.
    #[test]
    fn pad_table_slice_appends_dummies_repeating_the_last_interval_marker() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let mut slice = table_slice(
            zero,
            vec![sequence(SequenceKind::DefaultValue, zero, zero, 3, 5)],
            false,
        );

        PatternSimplificationManager::pad_table_slice(&mut slice, 3, &mut evaluator);

        assert_eq!(
            slice
                .sequences
                .iter()
                .map(|s| (s.kind, s.interval_marker, s.length))
                .collect::<Vec<_>>(),
            vec![
                (
                    SequenceKind::DefaultValue,
                    Some(IntervalMarker(5)),
                    Entries(3)
                ),
                (
                    SequenceKind::DefaultValue,
                    Some(IntervalMarker(5)),
                    Entries(0)
                ),
                (
                    SequenceKind::DefaultValue,
                    Some(IntervalMarker(5)),
                    Entries(0)
                ),
            ]
        );
        assert_eq!(slice.sequences[1].lb, Some(zero));
    }

    /// Each field the template leaves open costs a table-entry type and each varying interval marker
    /// a predicate type; the last sequence's marker is the `else` and is never encoded.
    #[test]
    fn generate_encoding_types_covers_the_open_fields_and_skips_the_last_marker() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let mut open_monotone = sequence(SequenceKind::MonotoneSequence, zero, zero, 1, 0);
        open_monotone.stride = None;
        open_monotone.interval_marker = None;
        let mut open_default = sequence(SequenceKind::DefaultValue, zero, zero, 1, 7);
        open_default.lb = None;
        let manager = PatternSimplificationManager {
            table: Some(table()),
            template_sequences: vec![open_monotone, open_default],
            ..PatternSimplificationManager::default()
        };

        assert_eq!(
            manager.generate_encoding_types(),
            vec![
                ScalarTy::Index,
                ScalarTy::Index,
                ScalarTy::Int(1),
                ScalarTy::Index
            ]
        );
    }

    /// A field the template leaves open is materialised from this slice's own sequence, and its
    /// varying interval marker becomes a `sentient.scalar_constant` of the predicate type.
    #[test]
    fn generate_encoding_tuple_materialises_the_open_fields_and_the_varying_marker() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let four = evaluator.get_constant(4);
        let nine = evaluator.get_constant(9);
        let mut open = sequence(SequenceKind::DefaultValue, zero, zero, 1, 4);
        open.lb = None;
        open.interval_marker = None;
        let manager = PatternSimplificationManager {
            table: Some(table()),
            template_sequences: vec![open, sequence(SequenceKind::DefaultValue, zero, zero, 1, 9)],
            ..PatternSimplificationManager::default()
        };
        let slice = table_slice(
            zero,
            vec![
                sequence(SequenceKind::DefaultValue, four, zero, 1, 4),
                sequence(SequenceKind::DefaultValue, nine, zero, 1, 9),
            ],
            false,
        );
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();

        let tuple = {
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            manager.generate_encoding_tuple(&slice, &mut builders, &mut evaluator)
        };

        assert_eq!(tuple, vec![Val(0), Val(1)]);
        assert_eq!(evaluator.built, vec![four]);
        assert_eq!(consts, vec![scalar_constant(Val(1), 4, ScalarTy::Int(1))]);
    }

    /// The template's known value is materialised and its open ones read off the encoding in order,
    /// the monotone sequence's start value and step are recorded, and the chain marks the monotone
    /// yield and both new comparisons.
    #[test]
    fn code_gen_monotone_table_slice_builds_the_chain_and_marks_the_monotone_yield() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let ten = evaluator.get_constant(10);
        let twenty = evaluator.get_constant(20);
        let mut open_monotone = sequence(SequenceKind::MonotoneSequence, zero, zero, 1, 4);
        open_monotone.stride = None;
        let iv_dim = IvDim {
            iv: Val(50),
            dimension: 4,
            multiplier: 1,
        };
        let mut manager = PatternSimplificationManager {
            table: Some(table()),
            template_sequences: vec![
                sequence(SequenceKind::DefaultValue, ten, zero, 1, 0),
                open_monotone,
                sequence(SequenceKind::DefaultValue, twenty, zero, 1, 9),
            ],
            ivs_dimensions_multipliers: vec![iv_dim, iv_dim],
            ..PatternSimplificationManager::default()
        };
        let iv = TypedVal {
            val: Val(50),
            ty: ScalarTy::Index,
        };
        let mut block = Vec::new();
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();

        let result = {
            let mut site = Site {
                block: &mut block,
                at: OpPath::at(&[(0, 0)]),
            };
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            manager.code_gen_monotone_table_slice(
                true,
                iv,
                &[Val(100), Val(101)],
                &mut site,
                &mut builders,
                &mut evaluator,
            )
        };

        let inner = Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Sle,
            lhs: Val(50),
            rhs: Val(3),
            yielded: vec![Yielded {
                result: Val(4),
                reg: UNKNOWN_LOCALE,
                element_size: None,
            }],
            dbg_name: None,
            then_body: vec![yield_op(vec![Val(100)])],
            else_body: vec![yield_op(vec![Val(5)])],
        });
        let outer = Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Sle,
            lhs: Val(50),
            rhs: Val(1),
            yielded: vec![Yielded {
                result: Val(2),
                reg: UNKNOWN_LOCALE,
                element_size: None,
            }],
            dbg_name: None,
            then_body: vec![yield_op(vec![Val(0)])],
            else_body: vec![inner, yield_op(vec![Val(4)])],
        });
        assert_eq!(result, Some(Val(2)));
        assert_eq!(block, vec![outer]);
        assert_eq!(manager.monotone_seq_start_val, Some(Val(100)));
        assert_eq!(manager.monotone_seq_val_step, Some(Val(101)));
        assert!(
            manager
                .marks
                .has(Mark::ResultReplacedByIterArg, &[(0, 0), (1, 0), (0, 0)])
        );
        assert!(
            manager
                .marks
                .has(Mark::LhsInPredReplacedByIterArg, &[(0, 0)])
        );
        assert!(
            manager
                .marks
                .has(Mark::LhsInPredReplacedByIterArg, &[(0, 0), (1, 0)])
        );
    }

    /// Every sequence but the last gets a branch testing its own interval marker, the last one
    /// becoming the innermost `else`.
    #[test]
    fn code_gen_generic_table_slice_chains_one_branch_per_sequence_but_the_last() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let first = evaluator.get_constant(1);
        let second = evaluator.get_constant(2);
        let third = evaluator.get_constant(3);
        let slice = table_slice(
            zero,
            vec![
                sequence(SequenceKind::DefaultValue, first, zero, 1, 3),
                sequence(SequenceKind::DefaultValue, second, zero, 1, 7),
                sequence(SequenceKind::DefaultValue, third, zero, 1, 9),
            ],
            false,
        );
        let manager = PatternSimplificationManager {
            table: Some(table()),
            ..PatternSimplificationManager::default()
        };
        let mut block = Vec::new();
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();

        let result = {
            let mut site = Site {
                block: &mut block,
                at: OpPath::at(&[(0, 0)]),
            };
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            manager.code_gen_generic_table_slice(
                Val(50),
                &slice,
                &mut site,
                &mut builders,
                &mut evaluator,
            )
        };

        let inner = Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Sle,
            lhs: Val(50),
            rhs: Val(4),
            yielded: vec![Yielded {
                result: Val(5),
                reg: UNKNOWN_LOCALE,
                element_size: None,
            }],
            dbg_name: None,
            then_body: vec![yield_op(vec![Val(3)])],
            else_body: vec![yield_op(vec![Val(6)])],
        });
        let outer = Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Sle,
            lhs: Val(50),
            rhs: Val(1),
            yielded: vec![Yielded {
                result: Val(2),
                reg: UNKNOWN_LOCALE,
                element_size: None,
            }],
            dbg_name: None,
            then_body: vec![yield_op(vec![Val(0)])],
            else_body: vec![inner, yield_op(vec![Val(5)])],
        });
        assert_eq!(result, Some(Val(2)));
        assert_eq!(block, vec![outer]);
        assert_eq!(evaluator.built, vec![first, second, third]);
        assert_eq!(
            consts,
            vec![
                scalar_constant(Val(1), 3, ScalarTy::Int(1)),
                scalar_constant(Val(4), 7, ScalarTy::Int(1)),
            ]
        );
    }

    /// e431 — the fixed IV contributes its one chosen iteration and the free IV every iteration its
    /// filter still allows, so the leaf lands at the flattened index of each surviving tuple; an IV
    /// with no allowed iteration at all kills the leaf.
    #[test]
    fn process_leaf_writes_one_entry_per_allowed_iteration_tuple() {
        let (fixed, free) = (Val(0), Val(1));
        let dims = [
            IvDim {
                iv: fixed,
                dimension: 2,
                multiplier: 2,
            },
            IvDim {
                iv: free,
                dimension: 2,
                multiplier: 1,
            },
        ];
        let loops = BTreeMap::from([(fixed, loop_info(None, 2)), (free, loop_info(None, 2))]);
        let mut table = Table::new(TableSize(4), ScalarTy::Int(1), ScalarTy::Index);
        let mut evaluator = FakeEvaluator::default();
        let leaf = Branch {
            leaf: Leaf {
                results: vec![Val(7)],
                block: None,
            },
            ..Branch::default()
        };
        let mut ivs = IvValuesAndFilters::of([(fixed, 2), (free, 2)]);
        ivs.set_chosen(fixed, Some(1), 2);
        // Iteration 0 of the free IV is filtered out, so only the tuple `(1, 1)` survives.
        ivs.set_filter(
            free,
            TableIdx::of(0, &loop_info(None, 2)).expect("iteration 0"),
            true,
            2,
        );

        let alive = {
            let mut sink = LeafSink {
                ivs_dimensions_multipliers: &dims,
                lhs_to_for_op_or_null: &loops,
                table: &mut table,
            };
            sink.process_leaf(&leaf, &ivs, 0, &mut Vec::new(), &mut evaluator)
        };

        assert!(alive);
        // `2 * 1 + 1 * 1`, and nothing at the filtered `2 * 1 + 1 * 0`.
        let entry = table
            .table_entry_at_idx(TableIndex(3))
            .expect("the tuple (1, 1)");
        assert_eq!(entry.ev(), Some(evaluator.evaluate_value(Val(7))));
        assert!(table.table_entry_at_idx(TableIndex(2)).is_none());

        let mut all_filtered = IvValuesAndFilters::of([(free, 2)]);
        for iteration in 0..2 {
            all_filtered.set_filter(
                free,
                TableIdx::of(iteration, &loop_info(None, 2)).expect("an iteration"),
                true,
                2,
            );
        }
        let mut sink = LeafSink {
            ivs_dimensions_multipliers: &dims,
            lhs_to_for_op_or_null: &loops,
            table: &mut table,
        };
        assert!(!sink.process_leaf(&leaf, &all_filtered, 0, &mut Vec::new(), &mut evaluator));
    }

    /// e432 — a nest needing no new iterator argument is never refused: the conditional is marked for
    /// deletion and its readers move onto `new_if`. Asking for one against a full loop is refused.
    #[test]
    fn transform_in_contiguous_sequence_case_needs_no_iter_arg_but_respects_the_cap() {
        let iv = Val(1);
        let if_path = OpPath::at(&[(0, 1), (0, 0)]);
        let body = vec![
            if_op(
                iv,
                Val(4),
                vec![Yielded {
                    result: Val(20),
                    reg: UNASSIGNED,
                    element_size: None,
                }],
                vec![yield_op(vec![Val(21)])],
            ),
            scalar_add(Val(20), Val(5), Val(22)),
            yield_op(Vec::new()),
        ];
        let manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(
                iv,
                loop_info(Some(OpPath::at(&[(0, 1)])), 4),
            )]),
            ivs_dimensions_multipliers: vec![IvDim {
                iv,
                dimension: 4,
                multiplier: 1,
            }],
            table: Some(table()),
            ..PatternSimplificationManager::default()
        };
        let mut evaluator = FakeEvaluator::default();
        let mut clone = |_: &mut Vec<Op>, at: &OpPath, _: &[(Val, Val)]| at.clone();

        let mut nothing_to_add = manager.clone();
        let mut root = vec![constant(5, Val(5)), for_op(Vec::new(), body.clone())];
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();
        let case = ContiguousCase {
            if_op: &if_path,
            new_if: Val(30),
            num_new_iter_args: 0,
            has_positive_step: true,
        };
        let transformed = {
            let mut cloning = LoopCloning {
                for_ops_to_avoid_new_iter_args: &[],
                clone: &mut clone,
            };
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            nothing_to_add.transform_in_contiguous_sequence_case(
                &mut root,
                &case,
                &mut cloning,
                &mut builders,
                &mut evaluator,
            )
        };

        assert!(transformed);
        assert!(nothing_to_add.marks.has(Mark::ToDelete, if_path.path()));
        assert_eq!(
            op_at(&root, &[(0, 1), (0, 1)]),
            Some(&scalar_add(Val(30), Val(5), Val(22)))
        );

        // Six iterator arguments already, so the one this nest asks for does not fit (`:1691`).
        let full: Vec<Carried> = (0..6)
            .map(|i| Carried {
                init: Val(5),
                arg: Val(40 + i),
                result: Val(50 + i),
                reg: UNASSIGNED,
                program_header: false,
                element_size: None,
            })
            .collect();
        let mut at_capacity = manager;
        let mut root = vec![constant(5, Val(5)), for_op(full, body)];
        let case = ContiguousCase {
            num_new_iter_args: 1,
            ..case
        };
        let mut cloning = LoopCloning {
            for_ops_to_avoid_new_iter_args: &[],
            clone: &mut clone,
        };
        let mut builders = Builders {
            consts: &mut consts,
            query_maps: &mut query_maps,
            values: &mut values,
        };
        assert!(!at_capacity.transform_in_contiguous_sequence_case(
            &mut root,
            &case,
            &mut cloning,
            &mut builders,
            &mut evaluator,
        ));
        assert!(!at_capacity.marks.has(Mark::ToDelete, if_path.path()));
    }

    /// e433 — with no iterator argument to reuse, the free IV's loop is cloned for the monotone
    /// sequence's `(start, step)` pair, the conditional is retired in favour of `new_if`, and the
    /// marked yield hands back the clone's new argument.
    #[test]
    fn transform_in_monotone_sequence_case_clones_the_loop_for_the_sequence_arg() {
        let iv = Val(1);
        let (for_path, if_path) = (OpPath::at(&[(0, 0)]), OpPath::at(&[(0, 0), (0, 0)]));
        let mut root = vec![for_op(
            Vec::new(),
            vec![
                if_op(
                    iv,
                    Val(4),
                    vec![Yielded {
                        result: Val(20),
                        reg: UNASSIGNED,
                        element_size: None,
                    }],
                    vec![yield_op(vec![Val(21)])],
                ),
                scalar_add(Val(20), Val(5), Val(22)),
                yield_op(vec![Val(22)]),
            ],
        )];
        let mut manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(iv, loop_info(Some(for_path.clone()), 4))]),
            ivs_dimensions_multipliers: vec![IvDim {
                iv,
                dimension: 4,
                multiplier: 1,
            }],
            table: Some(table()),
            monotone_seq_start_val: Some(Val(10)),
            monotone_seq_val_step: Some(Val(11)),
            ..PatternSimplificationManager::default()
        };
        manager
            .marks
            .set(Mark::ResultReplacedByIterArg, &[(0, 0), (0, 2)]);
        let mut evaluator = FakeEvaluator::default();
        let mut requests: Vec<(Val, Val)> = Vec::new();
        let mut clone = |root: &mut Vec<Op>, at: &OpPath, pairs: &[(Val, Val)]| {
            requests.extend_from_slice(pairs);
            if let Some(Op::Sentient(sentient::Op::For { carried, .. })) =
                op_at_mut(root, at.path())
            {
                for &(init, _) in pairs {
                    carried.push(Carried {
                        init,
                        arg: Val(60),
                        result: Val(61),
                        reg: UNASSIGNED,
                        program_header: false,
                        element_size: None,
                    });
                }
            }
            at.clone()
        };
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();

        let transformed = {
            let mut cloning = LoopCloning {
                for_ops_to_avoid_new_iter_args: &[],
                clone: &mut clone,
            };
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            manager.transform_in_monotone_sequence_case(
                &mut root,
                &MonotoneCase {
                    for_op: &for_path,
                    if_op: &if_path,
                    new_if: Val(30),
                },
                &mut cloning,
                &mut builders,
                &mut evaluator,
            )
        };

        assert!(transformed);
        assert_eq!(requests, vec![(Val(10), Val(11))]);
        assert!(manager.marks.has(Mark::ToDelete, if_path.path()));
        assert_eq!(
            op_at(&root, &[(0, 0), (0, 1)]),
            Some(&scalar_add(Val(30), Val(5), Val(22)))
        );
        assert_eq!(
            op_at(&root, &[(0, 0), (0, 2)]),
            Some(&yield_op(vec![Val(60)]))
        );
    }

    /// e434 — three equal entries open one default-value sequence of length three, and the fourth,
    /// whose difference matches that sequence's zero stride, starts a second sequence of its own.
    #[test]
    fn parse_sequence_opens_a_run_of_three_then_starts_a_new_one() {
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let mut table = Table::new(TableSize(4), ScalarTy::Int(1), ScalarTy::Index);
        for at in 0..3 {
            table.create_table_entry_at_idx(
                TableIndex(at),
                &Leaf {
                    results: vec![Val(7)],
                    block: None,
                },
                &mut evaluator,
            );
        }
        table.create_table_entry_at_idx(
            TableIndex(3),
            &Leaf {
                results: vec![Val(8)],
                block: None,
            },
            &mut evaluator,
        );
        let mut manager = PatternSimplificationManager {
            table: Some(table),
            ..PatternSimplificationManager::default()
        };
        let mut slice = table_slice(zero, Vec::new(), false);

        manager.parse_sequence(
            IntervalMarker(0),
            IntervalStride::new(ONE),
            TableIndex(0),
            Entries(4),
            &mut slice,
            &mut evaluator,
        );

        assert_eq!(
            slice
                .sequences
                .iter()
                .map(|s| (s.kind, s.length, s.interval_marker, s.lb))
                .collect::<Vec<_>>(),
            vec![
                (
                    SequenceKind::DefaultValue,
                    Entries(3),
                    Some(IntervalMarker(2)),
                    Some(evaluator.evaluate_value(Val(7))),
                ),
                (
                    SequenceKind::DefaultValue,
                    Entries(1),
                    Some(IntervalMarker(3)),
                    Some(evaluator.evaluate_value(Val(8))),
                ),
            ]
        );
        assert!(!manager.abort_pattern);
    }

    /// e435 — a fixed IV of dimension two becomes one `if (iv == lb)` whose two arms each hold one
    /// table slice's value, and the chain answers the conditional's own result.
    #[test]
    fn code_gen_generic_nested_if_builds_one_branch_per_fixed_iteration() {
        let (fixed, free) = (Val(1), Val(2));
        let mut evaluator = FakeEvaluator::default();
        let zero = evaluator.get_constant(0);
        let first = evaluator.get_constant(11);
        let second = evaluator.get_constant(22);
        let slices = vec![
            table_slice(
                zero,
                vec![sequence(SequenceKind::DefaultValue, first, zero, 1, 0)],
                false,
            ),
            table_slice(
                zero,
                vec![sequence(SequenceKind::DefaultValue, second, zero, 1, 0)],
                false,
            ),
        ];
        let manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(fixed, loop_info(None, 2))]),
            table: Some(table()),
            ..PatternSimplificationManager::default()
        };
        let ivs = [
            IvDim {
                iv: fixed,
                dimension: 2,
                multiplier: 1,
            },
            IvDim {
                iv: free,
                dimension: 2,
                multiplier: 1,
            },
        ];
        let types = [ScalarTy::Index];
        let mut nest = GenericNest {
            generate_encoding: false,
            types: &types,
            slices: slices.iter(),
        };
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();

        let (ops, results) = {
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            manager.code_gen_generic_nested_if(
                &mut nest,
                &ivs,
                &OpPath::at(&[(0, 0)]),
                &mut builders,
                &mut evaluator,
            )
        };

        assert_eq!(results, vec![Val(1)]);
        assert_eq!(
            ops,
            vec![Op::Sentient(sentient::Op::If {
                predicate: CmpPredicate::Eq,
                lhs: fixed,
                rhs: Val(0),
                yielded: vec![Yielded {
                    result: Val(1),
                    reg: UNKNOWN_LOCALE,
                    element_size: None,
                }],
                dbg_name: None,
                then_body: vec![yield_op(vec![Val(2)])],
                else_body: vec![yield_op(vec![Val(3)])],
            })]
        );
        assert_eq!(consts, vec![scalar_constant(Val(0), 0, ScalarTy::Int(1))]);
        // Both slices are consumed, the `then` one first.
        assert_eq!(evaluator.built, vec![first, second]);
        assert!(nest.slices.next().is_none());
    }

    /// 496/656 — one fixed IV of dimension two parses one slice per iteration, stepping the table
    /// index by that dimension's multiplier, and the two slices disagreeing in sequence count clears
    /// the consistency flag.
    #[test]
    fn parse_fixed_dims_parses_one_slice_per_fixed_iteration_and_notices_they_disagree() {
        let (fixed, free) = (Val(1), Val(2));
        let mut evaluator = FakeEvaluator::default();
        let mut table = Table::new(TableSize(4), ScalarTy::Int(1), ScalarTy::Index);
        // `[7, 7]` is one sequence and `[7, 8]` is two, so the slices cannot agree.
        for (at, result) in [(0, 7), (1, 7), (2, 7), (3, 8)] {
            table.create_table_entry_at_idx(
                TableIndex(at),
                &Leaf {
                    results: vec![Val(result)],
                    block: None,
                },
                &mut evaluator,
            );
        }
        let mut manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(free, loop_info(None, 2))]),
            table: Some(table),
            ..PatternSimplificationManager::default()
        };
        let mut slices = Vec::new();

        manager.parse_fixed_dims(
            &[IvDim {
                iv: fixed,
                dimension: 2,
                multiplier: 2,
            }],
            IvDim {
                iv: free,
                dimension: 2,
                multiplier: 1,
            },
            TableIndex(0),
            &mut slices,
            &mut evaluator,
        );

        assert_eq!(
            slices
                .iter()
                .map(|slice| (
                    slice.start_idx,
                    slice.idx_stride,
                    slice.length,
                    slice.sequences.len()
                ))
                .collect::<Vec<_>>(),
            vec![
                (TableIndex(0), IndexStride(1), Entries(2), 1),
                (TableIndex(2), IndexStride(1), Entries(2), 2),
            ]
        );
        assert!(!manager.table_slices_are_consistent);
    }

    /// 555/656 — the whole driver over the 1-D table `[D0 D0 D0 D1]`: case 1 builds
    /// `if (iv <= 2) yield D0 else yield D1` in front of the conditional it replaces, names it after
    /// that conditional, retires it and hands its reader the new result.
    #[test]
    fn find_patterns_and_simplify_replaces_the_conditional_with_the_contiguous_chain() {
        let free_iv = Val(50);
        let mut evaluator = FakeEvaluator::default();
        let mut table = Table::new(TableSize(4), ScalarTy::Int(1), ScalarTy::Index);
        for at in 0..3 {
            table.create_table_entry_at_idx(
                TableIndex(at),
                &Leaf {
                    results: vec![Val(70)],
                    block: None,
                },
                &mut evaluator,
            );
        }
        table.create_table_entry_at_idx(
            TableIndex(3),
            &Leaf {
                results: vec![Val(71)],
                block: None,
            },
            &mut evaluator,
        );
        let if_path = OpPath::at(&[(0, 0), (0, 0)]);
        let mut manager = PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::from([(
                free_iv,
                loop_info(Some(OpPath::at(&[(0, 0)])), 4),
            )]),
            ivs_dimensions_multipliers: vec![IvDim {
                iv: free_iv,
                dimension: 4,
                multiplier: 1,
            }],
            table: Some(table),
            ..PatternSimplificationManager::default()
        };
        let original = Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Eq,
            lhs: free_iv,
            rhs: Val(60),
            yielded: vec![Yielded {
                result: Val(61),
                reg: UNASSIGNED,
                element_size: None,
            }],
            dbg_name: Some("orig".to_string()),
            then_body: vec![yield_op(vec![Val(70)])],
            else_body: vec![yield_op(vec![Val(71)])],
        });
        let mut root = vec![Op::Sentient(sentient::Op::For {
            iv: free_iv,
            bound: Val(62),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: None,
            body: vec![
                original,
                scalar_add(Val(61), Val(63), Val(64)),
                yield_op(Vec::new()),
            ],
        })];
        let n = CondNode {
            lhs: free_iv,
            rhs_val: 0,
            op: if_path,
            then_node: Branch::default(),
            else_node: Branch::default(),
        };
        let mut no_candidates = false;
        let mut consts = Vec::new();
        let mut query_maps = Vec::new();
        let mut values = Values::default();

        let simplified = {
            let mut set_no_candidates = || no_candidates = true;
            let mut simplify_subtree = |_: &mut Vec<Op>, _: &OpPath| 0;
            let mut tree = TreeSeam {
                num_leaves_in_subtree: 4,
                set_no_candidates: &mut set_no_candidates,
                simplify_subtree: &mut simplify_subtree,
            };
            let mut clone = |_: &mut Vec<Op>, at: &OpPath, _: &[(Val, Val)]| at.clone();
            let mut cloning = LoopCloning {
                for_ops_to_avoid_new_iter_args: &[],
                clone: &mut clone,
            };
            let mut builders = Builders {
                consts: &mut consts,
                query_maps: &mut query_maps,
                values: &mut values,
            };
            manager.find_patterns_and_simplify(
                &mut root,
                &n,
                &mut tree,
                &mut cloning,
                &mut builders,
                &mut evaluator,
            )
        };

        assert!(simplified);
        assert!(no_candidates);
        let Some(Op::Sentient(sentient::Op::For { body, .. })) = root.first() else {
            panic!("the fixture's loop")
        };
        // The chain landed at the conditional's position, which pushed the conditional down one.
        assert_eq!(
            body.first(),
            Some(&Op::Sentient(sentient::Op::If {
                predicate: CmpPredicate::Sle,
                lhs: free_iv,
                rhs: Val(1),
                yielded: vec![Yielded {
                    result: Val(2),
                    reg: UNKNOWN_LOCALE,
                    element_size: None,
                }],
                dbg_name: Some("CFGSimpl(orig, case1)".to_string()),
                then_body: vec![yield_op(vec![Val(0)])],
                else_body: vec![yield_op(vec![Val(3)])],
            }))
        );
        assert!(matches!(
            body.get(1),
            Some(Op::Sentient(sentient::Op::If {
                dbg_name: Some(name),
                ..
            })) if name == "orig"
        ));
        assert!(manager.marks.has(Mark::ToDelete, &[(0, 0), (0, 1)]));
        assert_eq!(body.get(2), Some(&scalar_add(Val(2), Val(63), Val(64))));
        assert_eq!(consts, vec![scalar_constant(Val(1), 2, ScalarTy::Int(1))]);
    }
}
