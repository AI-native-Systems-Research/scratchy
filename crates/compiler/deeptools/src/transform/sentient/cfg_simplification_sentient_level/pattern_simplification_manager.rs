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

/// One entry of `lhs_to_for_op_or_null_` (`:537-690`): the loop a predicate's LHS is the IV of, with
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
    /// `std::get<3>` — ⭐ NON-ZERO AS A TYPE, which is what `DT_CHECK_MSG(step != 0, ..)` (`:931`)
    /// asks for at runtime.
    pub step: NonZeroI64,
    /// `std::get<4>` — the number of iterations.
    pub iterations: i64,
}

/// One entry of `ivs_dimensions_multipliers_` (`:537-690`): an IV, the size of the table dimension it
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

/// A TABLE INDEX THAT IS IN RANGE BY CONSTRUCTION — `is_idx_valid` (`:934`) as a type, so a filter
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
    /// `new bool[num_iterations]` (`:806`).
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

/// `PatternSimplificationManager` (`:537-690`) — the members the units in this file read.
#[derive(Debug, Clone, Default)]
pub struct PatternSimplificationManager {
    /// `lhs_to_for_op_or_null_`.
    pub lhs_to_for_op_or_null: BTreeMap<Val, LoopInfo>,
    /// `ivs_dimensions_multipliers_`.
    pub ivs_dimensions_multipliers: Vec<IvDim>,
    /// `monotone_seq_val_step_`.
    pub monotone_seq_val_step: Option<Val>,
    /// The pass-local op markers of `:42-50`.
    pub marks: Marks,
}

/// The iter arg [`PatternSimplificationManager::find_existing_iter_arg`] is looking for.
#[derive(Debug, Clone, Copy)]
pub struct IterArgTarget {
    /// `target_lb`.
    pub lb: Val,
    /// `target_step`; `None` is the case where `monotone_seq_val_step_` stands in for it (`:1519`).
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

/// `*step_as_ev == evaluator_.evaluateMultiplyByConst(*target_step, multiplier)` (`:1508-1515`).
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
        | Op::Uniform(_) => None,
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
        | Op::Uniform(_) => None,
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
        // `DT_CHECK_MSG(step != 0, ..)` (`:931`) is [`LoopInfo::step`]'s type; an IV with no recorded
        // loop is the state that check catches, and there is no table dimension to populate for it.
        let (iterations, idx) = {
            let Some(info) = self.lhs_to_for_op_or_null.get(&iv) else {
                return;
            };
            (info.iterations, TableIdx::of(rhs_val, info))
        };
        // `DT_CHECK_MSG(is_idx_valid || n->getThenNode()->isDead(), ..)` (`:935`) is a pure
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
            // by a dead op, or by a `scalar_add` that feeds nothing but the yield (`:1478-1503`) —
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
            // `DT_CHECK_MSG(lower_bound, ..)` (`:1523`): a loop has one init per carried value.
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

// crustify:todo: e289_computeTemplateSeqAndUpdateMonotoneSeq
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2096  (75 body lines, level 1)
//   original  : void PatternSimplificationManager::computeTemplateSeqAndUpdateMonotoneSeq( std::vector<TableSlice *> &table_slices)
//   calls     : e026_changeToDefaultValue, e252_size

// crustify:todo: e290_insertSequence
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2360  (24 body lines, level 1)
//   original  : void PatternSimplificationManager::insertSequence(Sequence *seq, TableSlice *table_slice)
//   calls     : e252_size

// crustify:todo: e291_padTableSlice
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2398  (49 body lines, level 1)
//   original  : void PatternSimplificationManager::padTableSlice(TableSlice *table_slice, unsigned max_num_seq)
//   calls     : e025_getIntervalStride, e252_size

// crustify:todo: e292_generateEncodingTypes
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2450  (26 body lines, level 1)
//   original  : llvm::SmallVector<Type> PatternSimplificationManager::generateEncodingTypes()
//   calls     : e252_size

// crustify:todo: e293_generateEncodingTuple
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2485  (52 body lines, level 1)
//   original  : llvm::SmallVector<Value> PatternSimplificationManager::generateEncodingTuple( Value &iv, TableSlice *cur_table_slice, OpBuilder &builder)
//   calls     : e252_size

// crustify:todo: e294_codeGenMonotoneTableSlice
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2654  (137 body lines, level 1)
//   original  : Value PatternSimplificationManager::codeGenMonotoneTableSlice( bool mark_new_cmp, Value &iv, llvm::SmallVector<Value> &encoding_tuple, OpBuilder &builder)
//   calls     : e025_getIntervalStride, e252_size

// crustify:todo: e295_codeGenGenericTableSlice
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:2802  (58 body lines, level 1)
//   original  : Value PatternSimplificationManager::codeGenGenericTableSlice( const Value &iv, TableSlice *table_slice, OpBuilder &builder)
//   calls     : e025_getIntervalStride, e252_size

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
    use super::*;
    use crate::islands::sentient::dialects::sentient::{
        Carried, CmpPredicate, Reg, RegType, Yielded,
    };

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
}
