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
    ExpressionEvaluator, IntervalMarker, IntervalStride, Sequence, SequenceKind, Table, TableSlice,
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
}

/// `PatternSimplificationManager` (`:537-695`) — the members the units in this file read.
///
/// ⛔ `table_slices_are_consistent_` (`:570`) AND `cur_root_` (`:594`) ARE NOT DECLARED YET: no unit
/// ported so far reads either, and every use of `cur_root_` is the `getLoc()` of a builder call —
/// SentientIR carries no `Loc`, so there is nothing for the island to hold.
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
}

/// ⭐ TWO FLAGS START TRUE (`:559`, `:563`) — a derived `Default` would start the pass having already
/// failed to match both patterns, and `insertSequence` only ever clears them.
impl Default for PatternSimplificationManager {
    fn default() -> PatternSimplificationManager {
        PatternSimplificationManager {
            lhs_to_for_op_or_null: BTreeMap::new(),
            ivs_dimensions_multipliers: Vec::new(),
            table: None,
            template_sequences: Vec::new(),
            table_follows_contiguous_pattern: true,
            table_slices_follow_montone_pattern: true,
            abort_pattern: false,
            monotone_seq_start_val: None,
            monotone_seq_int_step: None,
            monotone_seq_val_step: None,
            marks: Marks::default(),
        }
    }
}

/// The iter arg [`PatternSimplificationManager::find_existing_iter_arg`] is looking for.
#[derive(Debug, Clone, Copy)]
pub struct IterArgTarget {
    /// `target_lb`.
    pub lb: Val,
    /// `target_step`; `None` is the case where `monotone_seq_val_step_` stands in for it (`:1522-1523`).
    pub step: Option<EvaluatedValue>,
    /// `type`.
    pub ty: ScalarTy,
}

/// One `EvaluatedValue` — the expression evaluator's answer for a value.
///
/// ⛔ `Analyses/ExpressionEvaluatorUtils` IS OUT OF CAMPAIGN SCOPE. This is the NAME and the value it
/// was asked about, nothing more: every question about one goes through [`step_matches_target`],
/// which is a `todo!` rather than a guess at what the evaluator would have said.
#[derive(Debug, Clone, Copy)]
pub struct EvaluatedValue(pub Val);

/// `*step_as_ev == evaluator_.evaluateMultiplyByConst(*target_step, multiplier)` (`:1519-1520`).
fn step_matches_target(step: Val, target_step: EvaluatedValue, multiplier: i64) -> bool {
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

// crustify:todo: e431_processLeaf
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1023  (64 body lines, level 2)
//   original  : template <typename MapTy> bool PatternSimplificationManager::processLeaf( dcc::CFGSCondNode *leaf, MapTy &ivs_to_values_and_filters, typename MapTy::iterator it, std::vector<dcc::WidestIntType> &tuple_so_far)
//   calls     : e288_createTableEntryAtIdx

// crustify:todo: e432_transformInContiguousSequenceCase
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1597  (321 body lines, level 2)
//   original  : bool PatternSimplificationManager::transformInContiguousSequenceCase( dcc::CFGSSentientLevelConditionalTree &tree, Operation *if_op, Value new_if, int num_new_iter_args, bool has_positive_step)
//   calls     : e032_findExistingIterArg, e033_replaceResultByIterArg, e034_replaceIfOpByIterArg, e035_replacePredicateIVByIterArg, e036_checkCascadingArgUses, e252_size, e393_createSentientForOpWithAdditionalIterArgs

// crustify:todo: e433_transformInMonotoneSequenceCase
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1923  (91 body lines, level 2)
//   original  : bool PatternSimplificationManager::transformInMonotoneSequenceCase( Operation *for_op, Operation *if_op, Value new_if)
//   calls     : e032_findExistingIterArg, e033_replaceResultByIterArg, e034_replaceIfOpByIterArg, e252_size, e393_createSentientForOpWithAdditionalIterArgs

// crustify:todo: e434_parseSequence
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2177  (176 body lines, level 2)
//   original  : void PatternSimplificationManager::parseSequence( dcc::WidestIntType iv_cur_val, dcc::WidestIntType iv_step, Value &iv, dcc::WidestIntType index, dcc::WidestIntType length_remaining, TableSlice *table_slice)
//   calls     : e023_incrementLength, e024_incrementIntervalMarker, e028_getPrevVal, e029_getEV, e287_getTableEntryAtIdx, e290_insertSequence

// crustify:todo: e435_codeGenGenericNestedIf
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2556  (71 body lines, level 2)
//   original  : llvm::SmallVector<Value> PatternSimplificationManager::codeGenGenericNestedIf( bool is_start, bool generate_encoding, std::vector<std::tuple<Value, dcc::WidestIntType, dcc::WidestIntType>>::iterator it_cur, std::vector<std::tuple<Value, dcc::WidestIntType, dcc::WidestIntType>>::iterator it_free_dime
//   calls     : e293_generateEncodingTuple, e295_codeGenGenericTableSlice

// crustify:todo: e496_parseFixedDims
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2021  (59 body lines, level 3)
//   original  : void PatternSimplificationManager::parseFixedDims( std::vector<std::tuple<Value, dcc::WidestIntType, dcc::WidestIntType>>::iterator it_cur, std::vector<std::tuple<Value, dcc::WidestIntType, dcc::WidestIntType>>::iterator it_free_dimension, dcc::WidestIntType index, std::vector<TableSlice *> &table_s
//   calls     : e027_print, e252_size, e434_parseSequence

// crustify:todo: e555_findPatternsAndSimplify
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:1101  (340 body lines, level 4)
//   original  : bool PatternSimplificationManager::findPatternsAndSimplify( dcc::CFGSCondNode *n, dcc::CFGSSentientLevelConditionalTree &tree)
//   calls     : e027_print, e252_size, e286_recomputeAsDefaultVals, e289_computeTemplateSeqAndUpdateMonotoneSeq, e291_padTableSlice, e292_generateEncodingTypes, e293_generateEncodingTuple, e294_codeGenMonotoneTableSlice, e432_transformInContiguousSequenceCase, e433_transformInMonotoneSequenceCase, e434_parseSequence, e435_codeGenGenericNestedIf, e496_parseFixedDims

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
}
