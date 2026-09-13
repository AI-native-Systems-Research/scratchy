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

//! `LoopMerging.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e075_getNextEligibleOp` | 075 | 0 | 11 | `dcc/src/Transform/Sentient/LoopMerging.cpp:63` |
//! | `e076_getNewDbgName` | 076 | 0 | 6 | `dcc/src/Transform/Sentient/LoopMerging.cpp:75` |
//! | `e316_mergeLoops` | 316 | 1 | 100 | `dcc/src/Transform/Sentient/LoopMerging.cpp:179` |
//! | `e508_loopsAreMergeable` | 508 | 3 | 94 | `dcc/src/Transform/Sentient/LoopMerging.cpp:83` |
//! | `e564_runLoopMerging` | 564 | 4 | 39 | `dcc/src/Transform/Sentient/LoopMerging.cpp:281` |
//! | `e601_runOn` | 601 | 5 | 13 | `dcc/src/Transform/Sentient/LoopMerging.cpp:321` |
//! | `e627_runOnOperation` | 627 | 6 | 5 | `dcc/src/Transform/Sentient/LoopMerging.cpp:335` |

use std::collections::BTreeMap;

use crate::arch::Arch;
use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    BlockArgEquivalence, DbgNamePrefix, EquivalenceTag, OperationEquivalence, SubregionCompare,
    new_dbg_name_from_list,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, dataflow, erase_defining_op, regions_mut, regions_ref,
    replace_all_uses_with, sentient, symbol, uniform, use_count,
};
use crate::model::Model;
use crate::transform::sentient::ForRef;
use crate::transform::sentient::analyses::{InstructionEstimator, OutOfScopeInstructionEstimator};
use crate::transform::sentient::loop_coalescing::TripLimit;
use crate::transform::sentient::loop_tree::LoopTree;
use crate::transform::sentient::regions_are_equivalent;
use crate::transform::sentient::utils::{ConstKind, ConstantImm, constant_imm, is_constant};
use crate::workload::Workload;

/// `oe_` (`LoopMerging.cpp:58-61`) — this pass's one comparison configuration.
///
/// ⛔ `all_block_args_are_equiv` IS `false`, so a pair of iter args the per-call checker declines is a
/// MISMATCH and not a match. It is built identically to loop absorption's (`LoopAbsorption.cpp:43-46`)
/// and differs from loop rolling's, which passes `true` (`LoopRolling.cpp:987-989`).
const OE: OperationEquivalence = OperationEquivalence::tagged(
    EquivalenceTag::LoopMerging,
    SubregionCompare::Recursive,
    BlockArgEquivalence::SameOwnerAndIndex,
);

/// `cl::opt<bool> EnableDynamicLoopMerging("dcc-loop-merging-dynamic-loops", .., cl::init(false))`
/// (`LoopMerging.cpp:36-39`) — off, so a loop whose trip count is not a constant never merges.
pub const ENABLE_DYNAMIC_LOOP_MERGING: bool = false;

/// WHERE AN OP SITS IN ITS BLOCK — the `Operation *` identity `loopsAreMergeable` compares against
/// the second loop (`:85`).
///
/// ⛔ A POSITION AND NOT A BORROW, because `getNextEligibleOp`'s answer is used for IDENTITY
/// (`getNextEligibleOp(loop_a) != loop_b`), and this island's ops are a tree with no addresses to
/// compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBlock(pub usize);

/// THE FIVE OPS MERGING STEPS OVER — `dataflow.get_unit`, `sentient.scalar_constant`,
/// `symbol.create_symbol`, `uniform.def_immutable_mapping` and `uniform.query_map` (`:66-69`).
///
/// ⛔ IT IS `uniform.query_map`, NOT `symbol.query_map`: the symbol dialect declares one of the same
/// name and the reference names the uniform one, so a symbol query between two loops keeps them
/// apart.
fn skipped_between_loops(op: &Op) -> bool {
    matches!(
        op,
        Op::Dataflow(dataflow::Op::GetUnit { .. })
            | Op::Sentient(sentient::Op::ScalarConstant { .. })
            | Op::Symbol(symbol::Op::CreateSymbol { .. })
            | Op::Uniform(uniform::Op::DefImmutableMapping { .. } | uniform::Op::QueryMap { .. })
    )
}

/// Replaces: e075_getNextEligibleOp
///
/// The next op after `op` in its own block, walking past the bookkeeping ops of
/// [`skipped_between_loops`].
///
/// ⛔ TRAP: `None` IS THE REFERENCE'S NULL AND ITS CALLER TREATS IT AS "NOT ADJACENT" — the last op
/// of a block is a merge candidate with nothing, so returning the block's end here would make two
/// loops in different blocks look adjacent.
#[must_use]
pub fn get_next_eligible_op(block: &[Op], op: InBlock) -> Option<InBlock> {
    block
        .iter()
        .enumerate()
        .skip(op.0 + 1)
        .find(|(_, next)| !skipped_between_loops(next))
        .map(|(at, _)| InBlock(at))
}

/// A LOOP'S `dbgName`, PRESENT — `for_op.getDbgName().value()` (`:77-78`) as a type.
///
/// ⛔ `.value()` ON AN ABSENT NAME IS `std::bad_optional_access`, so the reference merges two
/// unnamed loops by throwing. Here a nameless loop cannot reach [`get_new_dbg_name`] at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbgName(String);

impl DbgName {
    /// The name a `sentient.for` carries, or nothing when it is not a loop or carries none.
    #[must_use]
    pub fn of(op: &Op) -> Option<DbgName> {
        let Op::Sentient(sentient::Op::For { dbg_name, .. }) = op else {
            return None;
        };
        dbg_name.clone().map(DbgName)
    }

    /// The text it carries.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.0
    }
}

/// Replaces: e076_getNewDbgName
///
/// `LM(<first>, <second>)` — the merged loop's `dbgName`, built from both sources' own.
#[must_use]
pub fn get_new_dbg_name(for_op1: &DbgName, for_op2: &DbgName) -> DbgName {
    DbgName(format!("LM({}, {})", for_op1.text(), for_op2.text()))
}

/// TWO LOOPS THE PASS MAY MERGE — the pair's own two `DT_CHECK`s (`:181`, `:184-185`) as a type.
///
/// ⛔⛔ *"loop pointers are identical"* AND *"expected second loop to come right after the first one"*
/// ARE BOTH UNWRITABLE HERE, because the second position is not given: it is DERIVED by
/// [`get_next_eligible_op`], which never answers with its own argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeablePair {
    /// `loops.first`.
    first: InBlock,
    /// `loops.second`.
    second: InBlock,
}

impl MergeablePair {
    /// The pair `first` opens, or nothing when the next eligible op is not a second `sentient.for`.
    #[must_use]
    pub fn of(block: &[Op], first: InBlock) -> Option<MergeablePair> {
        let is_loop = |at: InBlock| {
            matches!(
                block.get(at.0),
                Some(Op::Sentient(sentient::Op::For { .. }))
            )
        };
        let second = get_next_eligible_op(block, first)?;
        (is_loop(first) && is_loop(second)).then_some(MergeablePair { first, second })
    }
}

/// WHERE A MERGE HAPPENS — the `dataflow.program_unit`'s block, and the nest of `sentient.for` bodies
/// from it down to the block the two loops share.
///
/// ⛔⛔ `DT_CHECK_MSG(unit, "expected loop to appear in dataflow.program_unit")` (`:203`) IS THIS TYPE,
/// and so are the reference's TWO builders: `OpBuilder const_builder(unit.getRegion())` (`:205`)
/// inserts at the START of `unit_body` — it is passed `.getRegion()` and so starts INSIDE, unlike
/// `LoopAbsorption.cpp:71` — while `OpBuilder builder(parent->getFirstChild()..)` (`:243`) inserts into
/// the block `nest` names. ⭐ NO ARM USES BOTH.
#[derive(Debug)]
pub struct MergeSite<'u> {
    /// The `dataflow.program_unit`'s own block.
    pub unit_body: &'u mut Vec<Op>,
    /// The region path from `unit_body` down to the loops' shared block, outermost first. EMPTY means
    /// the two loops are top-level, which is `getParentLoop() == getRoot()`.
    pub nest: Vec<NestStep>,
}

/// ONE STEP FROM A BLOCK INTO A REGION NESTED IN IT — `(the op's position, which of its regions)`.
///
/// ⛔ NOT A CHAIN OF `sentient.for` BODIES. `LoopTree` makes two loops inside one `sentient.if` region
/// SIBLINGS — *"a loop inside a `sentient.if` is a child of the loop AROUND the `if`"* — and the pairs
/// this pass considers are pairs of tree siblings, so their shared block is any region on the way down.
pub type NestStep = (usize, usize);

/// WHETHER THE PAIR BECAME ONE LOOP — the `++loops_merged_count` the pass keeps (`:277`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merged {
    /// The two loops are now one.
    Pair,
    /// The site names no such pair.
    ///
    /// ⛔ NOT `DT_CHECK_MSG(parent, "did not expect root here")` (`:239`), which CANNOT FIRE: the root
    /// is synthetic and non-null (`new LoopNode(nullptr)`, `LoopTree.hpp:150`), so a top-level pair
    /// reaches that arm and builds its `sentient.scalar_add` before the first top-level loop.
    NotMerged,
}

/// The block `nest` names — one region per step. `&[]` is `unit_body` itself.
fn block_at<'a>(unit_body: &'a [Op], nest: &[NestStep]) -> Option<&'a [Op]> {
    let mut block: &[Op] = unit_body;
    for (at, region) in nest {
        block = *regions_ref(block.get(*at)?).get(*region)?;
    }
    Some(block)
}

/// [`block_at`], for the rewrite.
fn block_at_mut<'a>(unit_body: &'a mut Vec<Op>, nest: &[NestStep]) -> Option<&'a mut Vec<Op>> {
    let mut block: &mut Vec<Op> = unit_body;
    for (at, region) in nest {
        block = regions_mut(block.get_mut(*at)?).into_iter().nth(*region)?;
    }
    Some(block)
}

/// WHERE A LOOP SITS IN THE UNIT — the region path down to its block, and its position in that block.
///
/// ⭐ THE MECHANISM `LoopNode *` REPLACES: the C++ node holds an `Operation *` that knows its own
/// block, so nothing has to be searched for. [`ForRef`] is an identity, so this finds it.
fn site_of(unit_body: &[Op], loop_op: ForRef) -> Option<(Vec<NestStep>, InBlock)> {
    fn search(block: &[Op], iv: Val, path: &mut Vec<NestStep>) -> Option<(Vec<NestStep>, InBlock)> {
        for (at, op) in block.iter().enumerate() {
            if matches!(op, Op::Sentient(sentient::Op::For { iv: this, .. }) if *this == iv) {
                return Some((path.clone(), InBlock(at)));
            }
            for (region, body) in regions_ref(op).into_iter().enumerate() {
                path.push((at, region));
                let found = search(body, iv, path);
                path.pop();
                if found.is_some() {
                    return found;
                }
            }
        }
        None
    }
    search(unit_body, loop_op.0, &mut Vec::new())
}

/// The `sentient.for` at `at`, as the four fields the merge reads.
fn loop_at(block: &[Op], at: InBlock) -> Option<(Val, &[sentient::Carried], Option<&str>)> {
    let Some(Op::Sentient(sentient::Op::For {
        bound,
        carried,
        dbg_name,
        ..
    })) = block.get(at.0)
    else {
        return None;
    };
    Some((*bound, carried, dbg_name.as_deref()))
}

/// Replaces: e316_mergeLoops
///
/// Merges the second loop into the first by summing their trip counts, rewiring the second's results
/// to the first's, renaming the survivor `LM(..)` and erasing the second loop and its dead bound.
///
/// ⛔ SIX OF THE EIGHT BOUND ARMS ARE `dcc::utils::updateBoundToValuePlusMap` /
/// `updateBoundToMapPlusMap` (`Analyses/Utils.cpp:421`, `:462`), which are OUT OF CAMPAIGN SCOPE.
///
/// ⛔ TRAP: `if (first_loop.getBound().getDefiningOp()->use_empty()) ..->erase()` (`:271-272`) READS THE
/// NEW BOUND, which the loop it was just built for reads — so that erase cannot fire. The one that
/// does is `saved_def_op`, captured from the SECOND bound BEFORE the loop goes (`:273-275`).
pub fn merge_loops(site: MergeSite<'_>, pair: MergeablePair, values: &mut Values) -> Merged {
    let MergeSite { unit_body, nest } = site;

    // ── everything the arm choice needs, before anything moves ──
    let (new_bound, new_bound_op, second_bound, results, dbg_name) = {
        let scope: &[Op] = unit_body;
        let Some(siblings) = block_at(scope, &nest) else {
            return Merged::NotMerged;
        };
        let (Some((first_bound, first_carried, first_name)), Some((second_bound, second_carried, second_name))) =
            (loop_at(siblings, pair.first), loop_at(siblings, pair.second))
        else {
            return Merged::NotMerged;
        };
        let defs = Definitions::from_innermost(core::slice::from_ref(&scope));
        let constant_of = |bound: Val| match defs.of(bound) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, ty, .. })) => Some((*value, *ty)),
            _ => None,
        };
        let is_query_map = |bound: Val| {
            matches!(defs.of(bound), Some(Op::Uniform(uniform::Op::QueryMap { .. })))
        };
        // `first_query_map && isConstant<sentient::ConstantOp>(first_query_map.getResult())`
        // (`:208-213`).
        let is_constant_query_map =
            |bound: Val| is_query_map(bound) && is_constant(bound, ConstKind::ScalarConstant, defs);

        let new_bound = values.mint();
        let new_bound_op = match (constant_of(first_bound), constant_of(second_bound)) {
            // `first_const_bound && second_const_bound` (`:214-218`) — the sum, at the START of the
            // unit's own block.
            (Some((first, ty)), Some((second, _))) => BuiltBound::AtUnitStart(Op::Sentient(
                sentient::Op::ScalarConstant {
                    value: first + second,
                    result: new_bound,
                    reg_locale: sentient::RegType::Imm,
                    ty,
                    is_symbol: false,
                },
            )),
            (Some(_), None) if is_constant_query_map(second_bound) => {
                todo!("dcc::utils::updateBoundToValuePlusMap — out of campaign scope (`:219-223`)")
            }
            (None, Some(_)) if is_constant_query_map(first_bound) => {
                todo!("dcc::utils::updateBoundToValuePlusMap — out of campaign scope (`:224-229`)")
            }
            _ if is_constant_query_map(first_bound) && is_constant_query_map(second_bound) => {
                todo!("dcc::utils::updateBoundToMapPlusMap — out of campaign scope (`:230-233`)")
            }
            // `!first_query_map && !second_query_map` (`:235-247`) — a `sentient.scalar_add` BEFORE the
            // parent's first child, so no instruction lands between two otherwise mergeable loops.
            _ if !is_query_map(first_bound) && !is_query_map(second_bound) => {
                BuiltBound::BeforeTheParentsFirstChild(Op::Sentient(sentient::Op::ScalarAdd {
                    lhs: first_bound,
                    rhs: second_bound,
                    result: new_bound,
                    // `AddOp::create` sets no `regLocale`, so the `.td`'s absent register stands, and
                    // `bound_type` (`:200`) is a trip count.
                    reg: None,
                    ty: ScalarTy::Index,
                    element_size: None,
                }))
            }
            _ if is_query_map(first_bound) != is_query_map(second_bound) => {
                todo!("dcc::utils::updateBoundToValuePlusMap — out of campaign scope (`:248-255`)")
            }
            _ => todo!("dcc::utils::updateBoundToMapPlusMap — out of campaign scope (`:256-260`)"),
        };
        // `DT_CHECK_MSG(.., "expected equal number of results")` (`:263-265`) is discharged by
        // `loopsAreMergeable`, which proved the two bodies equivalent; `llvm::zip` pairs what it has.
        let results: Vec<(Val, Val)> = second_carried
            .iter()
            .zip(first_carried)
            .map(|(second, first)| (second.result, first.result))
            .collect();
        (
            new_bound,
            new_bound_op,
            second_bound,
            results,
            new_dbg_name_from_list(DbgNamePrefix::Lm, first_name, &[second_name]),
        )
    };

    // ── the rewrite, innermost block first ──
    {
        let Some(siblings) = block_at_mut(unit_body, &nest) else {
            return Merged::NotMerged;
        };
        if let Some(Op::Sentient(sentient::Op::For {
            bound,
            dbg_name: name,
            ..
        })) = siblings.get_mut(pair.first.0)
        {
            *bound = new_bound;
            // ⭐ ONLY WHEN BOTH SOURCES HAD ONE (`:266-268`) — `setDbgNameAttr(op, nullptr)` REMOVES
            // the attribute, so the reference's `if` keeps the survivor's own name instead.
            if let Some(merged) = dbg_name {
                *name = Some(merged);
            }
        }
        // `std::get<1>(pair).replaceAllUsesWith(std::get<0>(pair))` (`:265`).
        for (of, with) in results {
            replace_all_uses_with(siblings, of, with);
        }
        // `if (first_loop.getBound().getDefiningOp()->use_empty()) ..->erase()` (`:271-272`).
        if use_count(new_bound, siblings) == 0 {
            erase_defining_op(siblings, new_bound);
        }
        // `second_loop.getOperation()->erase()` (`:274`).
        siblings.remove(pair.second.0);
    }
    // `if (saved_def_op->use_empty()) saved_def_op->erase();` (`:275`) — the second bound's definer
    // outlived the loop that read it.
    if use_count(second_bound, unit_body) == 0 {
        erase_defining_op(unit_body, second_bound);
    }
    match new_bound_op {
        BuiltBound::AtUnitStart(op) => unit_body.insert(0, op),
        BuiltBound::BeforeTheParentsFirstChild(op) => {
            let Some(siblings) = block_at_mut(unit_body, &nest) else {
                return Merged::NotMerged;
            };
            let at = siblings
                .iter()
                .position(|op| matches!(op, Op::Sentient(sentient::Op::For { .. })))
                .unwrap_or(0);
            siblings.insert(at, op);
        }
    }
    Merged::Pair
}

/// THE MERGED BOUND AND WHICH BUILDER MAKES IT — see [`MergeSite`].
#[derive(Debug)]
enum BuiltBound {
    /// `const_builder`, at the start of the unit's own block.
    AtUnitStart(Op),
    /// `builder`, before the first `sentient.for` of the loops' own parent.
    BeforeTheParentsFirstChild(Op),
}

/// `dcc::utils::isSumLessThanLCCRMaxValue` (`Analyses/Utils.cpp:343` and `:379`) — BOTH overloads, as
/// one question about two trip counts.
///
/// ⛔ NOT AN ANCHORED UNIT — `Transform/Sentient/Analyses/` is out of campaign scope, and this is the
/// arithmetic half, which [`ConstantImm`] already carries. The reference's `DT_CHECK` on two query
/// maps of different lengths is a refusal here.
fn is_sum_less_than_lccr_max_value(a: &ConstantImm, b: &ConstantImm) -> bool {
    let fits = |sum: i64| sum <= TripLimit::LCCR.0;
    match (a, b) {
        (ConstantImm::Scalar(a), ConstantImm::Scalar(b)) => fits(a.saturating_add(*b)),
        // The `(op, int64_t)` overload: one side folds to a scalar the other's every unit adds.
        (ConstantImm::Scalar(scalar), ConstantImm::PerUnit(values))
        | (ConstantImm::PerUnit(values), ConstantImm::Scalar(scalar)) => values
            .iter()
            .all(|value| fits(value.saturating_add(*scalar))),
        (ConstantImm::PerUnit(a), ConstantImm::PerUnit(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| fits(a.saturating_add(*b)))
        }
    }
}

/// The `sentient.for` body at `at` — `for_op.getLoopBody()`.
fn loop_body_at(block: &[Op], at: InBlock) -> Option<&[Op]> {
    match block.get(at.0) {
        Some(Op::Sentient(sentient::Op::For { body, .. })) => Some(body.as_slice()),
        _ => None,
    }
}

/// Replaces: e508_loopsAreMergeable
///
/// Two adjacent `sentient.for`s merge when their trip counts still fit one LCCR, the second picks its
/// iter args up where the first left them, and their bodies are the same computation modulo that.
///
/// ⛔ `mergeable_block_args` IS A LOCAL, not a field: written and read only here (`:130`, `:154-156`).
/// ⭐ A LOOP CARRYING NOTHING SKIPS THE PICK-UP TEST (`:109`) — effect-only loops merge on bodies alone.
#[must_use]
pub fn loops_are_mergeable(
    unit_body: &[Op],
    nest: &[NestStep],
    loop_a: InBlock,
    loop_b: InBlock,
) -> bool {
    let Some(siblings) = block_at(unit_body, nest) else {
        return false;
    };
    // `if (getNextEligibleOp(loop_a) != loop_b) return false;` (`:85-90`).
    if get_next_eligible_op(siblings, loop_a) != Some(loop_b) {
        return false;
    }
    let (Some((bound_a, carried_a, _)), Some((bound_b, carried_b, _))) =
        (loop_at(siblings, loop_a), loop_at(siblings, loop_b))
    else {
        return false;
    };
    let defs = Definitions::from_innermost(core::slice::from_ref(&unit_body));

    // `isConstant<sentient::ConstantOp>` on both bounds, then their sum against the LCCR (`:91-107`).
    match (constant_imm(bound_a, defs), constant_imm(bound_b, defs)) {
        (Some(a), Some(b)) => {
            if !is_sum_less_than_lccr_max_value(&a, &b) {
                return false;
            }
        }
        // `else if (!EnableDynamicLoopMerging)` — it is off, so this is always a refusal.
        _ if !ENABLE_DYNAMIC_LOOP_MERGING => return false,
        _ => {}
    }

    // `mergeable_block_args` — a's region iter arg to b's, for the checker below.
    let mut mergeable_block_args: BTreeMap<Val, Val> = BTreeMap::new();
    if !carried_a.is_empty() {
        // *"the iter_args of the second loop pick up where the previous loop left"* (`:111-126`).
        let picks_up = carried_a.len() == carried_b.len()
            && carried_a.iter().zip(carried_b).all(|(a, b)| {
                // `hasOneUse()` AND that one use being `loop_b` IS `result == init`: the init is a
                // use, so one use that is also the init leaves no other.
                use_count(a.result, unit_body) == 1 && a.result == b.init
            });
        if picks_up {
            for (a, b) in carried_a.iter().zip(carried_b) {
                mergeable_block_args.insert(a.arg, b.arg);
            }
        } else if carried_a
            .iter()
            .map(|a| a.result)
            .chain(carried_b.iter().map(|b| b.arg))
            .any(|val| use_count(val, unit_body) != 0)
        {
            // *"results or iter_args with non-empty use sets are not constrained"* (`:135-142`).
            return false;
        }
    }

    let (Some(body_a), Some(body_b)) = (
        loop_body_at(siblings, loop_a),
        loop_body_at(siblings, loop_b),
    ) else {
        return false;
    };
    // `operands_are_equiv` (`:146-171`) — a pair of iter args matches only through the map.
    let mut check = |a_operand: Val, b_operand: Val| -> bool {
        defs.of(a_operand).is_none()
            && defs.of(b_operand).is_none()
            && mergeable_block_args.get(&a_operand) == Some(&b_operand)
    };
    let mut checker: Option<&mut dyn FnMut(Val, Val) -> bool> = Some(&mut check);
    // ⭐ THE BLOCK ARGUMENT COUNT IS `1 + carried` — the induction variable then the iter args.
    regions_are_equivalent(
        1 + carried_a.len(),
        body_a,
        1 + carried_b.len(),
        body_b,
        defs,
        OE.block_args,
        &mut checker,
    )
}

/// `++loops_merged_count` (`:277`) — how many pairs this pass turned into one loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct LoopsMerged(pub usize);

/// Replaces: e564_runLoopMerging
///
/// Walks the unit's loop forest in reverse BFS, merging each node with its next sibling when
/// [`loops_are_mergeable`] accepts the pair, and removing the absorbed node from the tree.
///
/// ⛔⛔ `checkAndMerge`'s `resume` STEERING IS DEAD under the default `-dcc-loop-merging-walk-order=rbfs`:
/// [`LoopTree::walk_reverse_bfs`] precomputes the visit list and `(void)action(..)` discards the node
/// returned, so the condition is exactly "has a next sibling, is not the root, and is mergeable".
pub fn run_loop_merging(unit_body: &mut Vec<Op>, values: &mut Values) -> LoopsMerged {
    let mut tree = LoopTree::<false>::of(unit_body);
    // `if (!tree.empty()) tree.walk(..)` (`:315`).
    if tree.empty() {
        return LoopsMerged::default();
    }
    let mut merged = LoopsMerged::default();
    for n in tree.walk_reverse_bfs() {
        // `if (n == tree.getRoot() || !next_node) return resume;` (`:290`) — the root names no loop, so
        // `loop_of` answering `None` IS that test.
        let (Some(curr), Some(next)) = (tree.loop_of(n), tree.next_sibling(n)) else {
            continue;
        };
        // ⭐ A NODE ALREADY REMOVED IS NEVER ACTED ON: `tree.remove` only ever takes a LATER sibling,
        // and reverse BFS visited those FIRST — and its `sentient.for` is gone from the IR by then, so
        // `site_of` no longer finds it either.
        let (Some((nest, first)), Some((next_nest, second))) = (
            site_of(unit_body, curr),
            tree.loop_of(next).and_then(|l| site_of(unit_body, l)),
        ) else {
            continue;
        };
        // Two tree siblings in DIFFERENT blocks are never adjacent, which is `getNextEligibleOp`'s own
        // answer inside `loopsAreMergeable` (`:85`).
        if nest != next_nest {
            continue;
        }
        let Some(pair) = block_at(unit_body, &nest).and_then(|b| MergeablePair::of(b, first)) else {
            continue;
        };
        if pair.second != second || !loops_are_mergeable(unit_body, &nest, first, second) {
            continue;
        }
        // *"We are merging iff we are returning curr_node to the guided walk"* (`:305-309`).
        if merge_loops(MergeSite { unit_body, nest }, pair, values) == Merged::Pair {
            merged.0 += 1;
            tree.remove(next);
        }
    }
    merged
}

/// `opts_.OptLevel == 0` (`:326`) — the PIPELINE's optimisation level, which is `2` by default
/// (`dcc/tools/Options/dcc-pass-option.h:63-65`), so the shipped pipeline never reaches the
/// `haveIbuffSpace` half of the `&&`.
const OPT_LEVEL_ZERO: bool = false;

/// Replaces: e601_runOn
///
/// The module walk: merges the adjacent equivalent loops of every program unit whose instruction
/// buffer is not already roomy (`:321-333`).
///
/// ⛔ `haveIbuffSpace` STAYS A `todo!` BEHIND [`OPT_LEVEL_ZERO`]: the pipeline fixes that const's
/// value, not this pass, so flipping it reaches the estimator rather than quietly skipping a unit.
/// ⭐ `unit_list` COLLAPSES — [`crate::islands::sentient::ProgramUnits::iter_mut`] hands out one unit
/// at a time in walk order and merging one reaches no other, so collect-then-merge is one pass.
pub fn run_on<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>, values: &mut Values) {
    for unit in program.units.iter_mut() {
        // `getChildAnalysis<InstructionEstimator>(unit)` IS CONSTRUCTED PER UNIT (`:324-325`), even
        // for one the `&&` never asks anything of.
        let mut instruction_estimator = OutOfScopeInstructionEstimator;
        if OPT_LEVEL_ZERO && instruction_estimator.have_ibuff_space(&unit.body) {
            continue;
        }
        // `runLoopMerging` answers with a count nothing above it reads (`:332`).
        let _ = run_loop_merging(&mut unit.body, values);
    }
}

/// `cl::opt<bool> DisableThisPass("dcc-loop-merging-disable", .., cl::init(false))`
/// (`LoopMerging.cpp:33-35`) — off, so the shipped pipeline runs the pass.
const DISABLE_THIS_PASS: bool = false;

/// Replaces: e627_runOnOperation
///
/// The pass entry: unless the flag turns the whole pass off, merge every unit's adjacent loops
/// (`:335-339`).
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    values: &mut Values,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    run_on(program, values);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::Val;
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::{DfirUnit, Residency};

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

    /// `sentient.for` carrying `dbg_name`.
    fn sentient_for(dbg_name: Option<&str>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound: Val(1),
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: dbg_name.map(str::to_owned),
            body: Vec::new(),
        })
    }

    /// A `sentient.scalar_constant` a loop bound reads.
    fn constant_of(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// A `sentient.for` over `bound`, binding `iv`.
    fn loop_over(bound: Val, iv: Val, dbg_name: Option<&str>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: Vec::new(),
            dbg_name: dbg_name.map(str::to_owned),
            body: Vec::new(),
        })
    }

    /// The four non-`sentient` bookkeeping ops the walk steps over, plus the constant.
    fn skipped_ops() -> Vec<Op> {
        vec![
            Op::Dataflow(dataflow::Op::GetUnit {
                result: Val(2),
                residency: Residency::Global,
                unit: DfirUnit::L3lu,
                num_folds: None,
                reg_locale: None,
            }),
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: Val(3),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(4),
                symbol_id: 0,
                max_value: None,
            }),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(5),
                pairs: Vec::new(),
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(6),
                map: Val(5),
                key: Val(3),
            }),
        ]
    }

    /// Two constant-bounded loops whose counts fit one LCCR are mergeable; the same pair whose counts
    /// overflow it is not — `isSumLessThanLCCRMaxValue` is the only difference between the two.
    #[test]
    fn e508_refuses_a_pair_whose_trip_counts_overflow_one_lccr() {
        fn mergeable(first: i64, second: i64) -> bool {
            let block = vec![
                constant_of(Val(10), first),
                constant_of(Val(11), second),
                loop_over(Val(10), Val(20), None),
                loop_over(Val(11), Val(21), None),
            ];
            loops_are_mergeable(&block, &[], InBlock(2), InBlock(3))
        }

        assert!(mergeable(5, 7));
        assert!(!mergeable(60_000, 60_000));
    }

    /// Two loops separated by all five skipped ops are adjacent; one separated by anything else is
    /// not, and the last op of a block is adjacent to nothing.
    #[test]
    fn e075_walks_past_the_five_bookkeeping_ops_and_stops_at_the_block_end() {
        let mut block = vec![sentient_for(None)];
        block.extend(skipped_ops());
        block.push(sentient_for(None));
        assert_eq!(get_next_eligible_op(&block, InBlock(0)), Some(InBlock(6)));
        assert_eq!(get_next_eligible_op(&block, InBlock(6)), None);

        let blocked = vec![
            sentient_for(None),
            Op::Sentient(sentient::Op::Nop { dbg_name: None }),
            sentient_for(None),
        ];
        assert_eq!(get_next_eligible_op(&blocked, InBlock(0)), Some(InBlock(1)));
    }

    /// The merged name, and the absent name the reference's `.value()` throws on.
    #[test]
    fn e076_wraps_both_source_names_in_lm() {
        let first = sentient_for(Some("L0"));
        let second = sentient_for(Some("LM(L1, L2)"));
        let first = DbgName::of(&first).expect("a named loop");
        let second = DbgName::of(&second).expect("a named loop");
        assert_eq!(
            get_new_dbg_name(&first, &second).text(),
            "LM(L0, LM(L1, L2))"
        );
        assert!(DbgName::of(&sentient_for(None)).is_none());
        assert!(DbgName::of(&Op::Sentient(sentient::Op::Nop { dbg_name: None })).is_none());
    }
    /// Two adjacent constant-bound loops become one over the SUM, named from both, with the second
    /// loop and the second bound gone — and ⭐ THE FIRST LOOP'S OLD BOUND STILL STANDING, which is
    /// the reference's own dead erase (`:271-272`).
    #[test]
    fn e316_merges_two_constant_bound_loops_into_one_over_the_sum() {
        let mut values = Values::default();
        for _ in 0..20 {
            let _ = values.mint();
        }
        let mut unit_body = vec![
            constant_of(Val(0), 4),
            constant_of(Val(1), 6),
            loop_over(Val(0), Val(10), Some("a")),
            loop_over(Val(1), Val(11), Some("b")),
        ];
        let pair = MergeablePair::of(&unit_body, InBlock(2)).expect("two adjacent loops");
        let merged = merge_loops(
            MergeSite {
                unit_body: &mut unit_body,
                nest: Vec::new(),
            },
            pair,
            &mut values,
        );
        assert_eq!(merged, Merged::Pair);
        assert_eq!(
            unit_body,
            vec![
                constant_of(Val(20), 10),
                constant_of(Val(0), 4),
                loop_over(Val(20), Val(10), Some("LM(a, b)")),
            ]
        );
    }

    /// The reverse-BFS walk merges a pair inside a `sentient.if` region as readily as a top-level one —
    /// they are all root siblings — and ⛔ REFUSES THE CROSS-BLOCK PAIR the flattening puts beside it,
    /// which is `getNextEligibleOp` seeing the `sentient.if` and not the loop inside it.
    #[test]
    fn e564_merges_both_sibling_pairs_and_not_the_pair_that_spans_two_blocks() {
        let mut values = Values::default();
        for _ in 0..30 {
            let _ = values.mint();
        }
        let mut unit_body = vec![
            constant_of(Val(0), 1),
            constant_of(Val(1), 2),
            constant_of(Val(2), 4),
            constant_of(Val(3), 8),
            loop_over(Val(0), Val(10), None),
            loop_over(Val(1), Val(11), None),
            Op::Sentient(sentient::Op::If {
                predicate: sentient::CmpPredicate::Eq,
                lhs: Val(0),
                rhs: Val(1),
                yielded: Vec::new(),
                dbg_name: None,
                then_body: vec![
                    loop_over(Val(2), Val(12), None),
                    loop_over(Val(3), Val(13), None),
                ],
                else_body: Vec::new(),
            }),
        ];

        assert_eq!(
            run_loop_merging(&mut unit_body, &mut values),
            LoopsMerged(2)
        );
        let only_loop = |block: &[Op]| -> Val {
            let bounds: Vec<Val> = block
                .iter()
                .filter_map(|op| match op {
                    Op::Sentient(sentient::Op::For { bound, .. }) => Some(*bound),
                    _ => None,
                })
                .collect();
            let [only] = bounds[..] else {
                panic!("expected exactly one surviving loop, got {bounds:?}");
            };
            only
        };
        let Some(Op::Sentient(sentient::Op::If { then_body, .. })) = unit_body.last().cloned() else {
            panic!("the `sentient.if` survives");
        };
        let (top, nested) = (only_loop(&unit_body), only_loop(&then_body));
        // ⭐ BOTH MERGED BOUNDS LAND IN THE UNIT'S OWN BLOCK — `const_builder` is built from the
        // `dataflow.program_unit`'s region, not from the block the loops share.
        let scope: &[Op] = &unit_body;
        let defs = Definitions::from_innermost(core::slice::from_ref(&scope));
        let value_of = |bound: Val| match defs.of(bound) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => *value,
            other => panic!("expected a constant bound, got {other:?}"),
        };
        // The top-level pair summed to 1 + 2 and the `then` region's to 4 + 8: neither pair borrowed
        // the other's trip count, and no cross-block pair merged.
        assert_eq!((value_of(top), value_of(nested)), (3, 12));
    }

    /// e601 — the pass entry reaches every unit of the module: both units' adjacent pairs merge.
    #[test]
    fn e601_merges_the_loops_of_every_program_unit() {
        let mut values = Values::default();
        for _ in 0..40 {
            let _ = values.mint();
        }
        let body = || {
            vec![
                constant_of(Val(0), 4),
                constant_of(Val(1), 6),
                loop_over(Val(0), Val(10), None),
                loop_over(Val(1), Val(11), None),
            ]
        };
        let unit = |body: Vec<Op>| ProgramUnit {
            on: Units::one(DfirUnit::Lxlu, Val(99)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(unit(body()), vec![unit(body())]),
            bound: core::marker::PhantomData,
        };

        run_on(&mut program, &mut values);

        for unit in program.units.iter() {
            let loops = unit
                .body
                .iter()
                .filter(|op| matches!(op, Op::Sentient(sentient::Op::For { .. })))
                .count();
            assert_eq!(loops, 1, "each unit's pair merged: {:?}", unit.body);
        }
    }
    /// e627 — the pass entry with the flag off does what e601 does: the unit's pair merges.
    #[test]
    fn e627_runs_the_pass_when_the_flag_is_off() {
        let mut values = Values::default();
        for _ in 0..40 {
            let _ = values.mint();
        }
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(99)),
                    precision: None,
                    body: vec![
                        constant_of(Val(0), 4),
                        constant_of(Val(1), 6),
                        loop_over(Val(0), Val(10), None),
                        loop_over(Val(1), Val(11), None),
                    ],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };

        run_on_operation(&mut program, &mut values);

        let loops = program
            .units
            .iter()
            .flat_map(|unit| unit.body.iter())
            .filter(|op| matches!(op, Op::Sentient(sentient::Op::For { .. })))
            .count();
        assert_eq!(loops, 1);
    }
}
