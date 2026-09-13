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

//! `ScalarOpMergingAndHoisting.cpp` — 11 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e165_calculateIBuffRequired` | 165 | 0 | 24 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:591` |
//! | `e166_isProfitableForHoisting` | 166 | 0 | 29 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:837` |
//! | `e167_sortBlocks` | 167 | 0 | 19 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:910` |
//! | `e168_unrollBurstAndIL` | 168 | 0 | 51 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1011` |
//! | `e169_findFieldUnrollCandidate` | 169 | 0 | 8 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1294` |
//! | `e362_isFieldUnrollCandidate` | 362 | 1 | 124 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1080` |
//! | `e363_markFieldUnrollingCandidates` | 363 | 1 | 87 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1206` |
//! | `e364_doScalarOpMerging` | 364 | 1 | 135 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1334` |
//! | `e528_buildBlock` | 528 | 3 | 179 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:644` |
//! | `e577_collectBlocks` | 577 | 4 | 26 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:869` |
//! | `e611_runScalarOpMerging` | 611 | 5 | 10 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:569` |

use core::cmp::Ordering;
use std::collections::BTreeSet;

use super::scalar_op_hoisting::{
    IbuffSpace, first_const_operand_index, is_immutable_value_in_range, is_sentient_constant,
};
use super::{
    AddressScale, BurstAndIl, FieldUnrollData, MemoryOpInfo, OperationData, ScalarOpComp,
    ScalarOpMergingBlock, UnrollTarget, does_value_exceed_lrf_range,
};
use crate::arch::{Arch, Elements};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, erase_defining_op, operands, replace_all_uses_with, results, use_count,
};
use crate::transform::sentient::analyses::{
    EvaluatedValue, Evaluation, ExpressionEvaluator, InstructionCount, OffsetSites,
};

/// `DisableFieldUnrolling` (`:110-113`) — `cl::init(false)`, so the field-unroll arm is live.
const DISABLE_FIELD_UNROLLING: bool = false;

/// Replaces: e165_calculateIBuffRequired
///
/// The ADDITIONAL IBuff entries one transfer's burst and interleaved group cost — their product,
/// less the one entry the op itself already consumes (`:591-615`).
///
/// ⭐ `max(_, 1)` GIVES 0, 1 AND THE REFERENCE'S `il_group = -1` ONE ANSWER, which is why
/// [`BurstAndIl`] can carry the island's unsigned counts without losing a case.
#[must_use]
pub(crate) fn calculate_ibuff_required(transfer: BurstAndIl) -> InstructionCount {
    let entries = transfer
        .burst
        .0
        .max(1)
        .saturating_mul(transfer.interleaved_group.0.max(1))
        - 1;
    // `(int)` on both counts, and `int` is what `required_ibuff_` accumulates. ⭐ SATURATING WHERE THE
    // REFERENCE'S NARROWING CAST IS IMPLEMENTATION-DEFINED: a burst past `INT_MAX` is not an IBuff
    // budget any unit could meet on either side of the conversion.
    InstructionCount(i32::try_from(entries).unwrap_or(i32::MAX))
}

/// Replaces: e166_isProfitableForHoisting
///
/// Whether merging `block` would also let scalar op hoisting fire: the block's bottom op must feed
/// the enclosing `sentient.for`'s `sentient.yield` at the very iter-arg position the block reads
/// (`:837-865`).
///
/// ⭐ `parent_op` IS `region->getParentOp()`, which a `&mut Vec<Op>` region cannot answer — `None`,
/// and any op that is not a `sentient.for`, are both the reference's failed `dyn_cast_or_null`.
#[must_use]
pub(crate) fn is_profitable_for_hoisting(
    block: &ScalarOpMergingBlock,
    parent_op: Option<&Op>,
) -> bool {
    // "Blocks are formed in reverse order so the front of the block is closest to the bottom of the
    // containing Region" — and an empty block has no such op, which `front()` does not survive.
    let Some(&OperationData { op: bottom_op, .. }) = block.block_ops.first() else {
        return false;
    };
    let Some(Op::Sentient(ops::Op::For { carried, body, .. })) = parent_op else {
        return false;
    };
    let Some(input_val) = block.input_value_to_block else {
        return false;
    };
    // `!input_val.hasOneUse()`. ⭐ COUNTING INSIDE THE BODY LOSES NOTHING: the count only decides the
    // answer for a value that IS one of `carried`'s args, and a region argument has no use outside.
    if use_count(input_val, body) != 1 {
        return false;
    }
    let Some(yielded) = body.iter().find_map(|op| match op {
        Op::Sentient(ops::Op::Yield { results }) => Some(results),
        _ => None,
    }) else {
        return false;
    };
    // ⭐ `*last_op_in_block->getUsers().begin()` IS THAT YIELD, OR THE ANSWER IS NO: every op of a
    // merging block has exactly one use, so a yield reading `bottom_op` at `idx` is that one use.
    carried
        .iter()
        .enumerate()
        .any(|(idx, slot)| slot.arg == input_val && yielded.get(idx) == Some(&bottom_op))
}

/// HOW MUCH TRANSFORMING ONE BLOCK IS WORTH — `sortBlocks`'s `GetBlockRatio` lambda (`:911-923`).
///
/// ⛔ NEITHER THE FLOAT NOR THE `1000.0` SENTINEL SURVIVES, BECAUSE BOTH SPELL AN ORDER. The ratio is
/// `num_scalar_ops / required_ibuff` and the sentinel is there so a block needing no IBuff "will
/// always beat blocks that do require additional IBuff" — the reference's own words. Comparing by
/// cross-multiplication says the first exactly, and a greatest element says the second including the
/// one case the sentinel gets wrong: 1001 eliminated scalar ops against a single IBuff slot.
#[derive(Debug, Clone, Copy)]
enum BlockRatio {
    /// `(float)getNumScalarOpsInBlock() / (float)required_ibuff`, kept as the pair.
    PerSlot { ops: u32, slots: i32 },
    /// `required_ibuff == 0` — highest priority, and transformed wherever it lands.
    NoSlotsNeeded,
}

impl BlockRatio {
    fn of(block: &ScalarOpMergingBlock) -> BlockRatio {
        if block.required_ibuff == InstructionCount(0) {
            return BlockRatio::NoSlotsNeeded;
        }
        BlockRatio::PerSlot {
            ops: block.num_scalar_ops_in_block.0,
            slots: block.required_ibuff.0,
        }
    }
}

impl Ord for BlockRatio {
    fn cmp(&self, other: &BlockRatio) -> Ordering {
        match (self, other) {
            (BlockRatio::NoSlotsNeeded, BlockRatio::NoSlotsNeeded) => Ordering::Equal,
            (BlockRatio::NoSlotsNeeded, BlockRatio::PerSlot { .. }) => Ordering::Greater,
            (BlockRatio::PerSlot { .. }, BlockRatio::NoSlotsNeeded) => Ordering::Less,
            // `a.ops / a.slots` against `b.ops / b.slots`, without the division. ⭐ THE DIVISOR IS
            // POSITIVE HERE — `required_ibuff_` is a sum of `calculateIBuffRequired` answers, none of
            // them negative — so cross-multiplication preserves the order the float division gives.
            // `i128` because `u32 * i32` is the one place this could wrap, and a wrap reorders the pass.
            (
                BlockRatio::PerSlot {
                    ops: a_ops,
                    slots: a_slots,
                },
                BlockRatio::PerSlot {
                    ops: b_ops,
                    slots: b_slots,
                },
            ) => (i128::from(*a_ops) * i128::from(*b_slots))
                .cmp(&(i128::from(*b_ops) * i128::from(*a_slots))),
        }
    }
}

impl PartialOrd for BlockRatio {
    fn partial_cmp(&self, other: &BlockRatio) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BlockRatio {
    fn eq(&self, other: &BlockRatio) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for BlockRatio {}

/// Replaces: e167_sortBlocks
///
/// Orders the candidate blocks highest benefit first, so that running out of IBuff drops the least
/// valuable of them (`:910-928`).
///
/// ⭐ STABLE WHERE `std::sort` IS NOT: the C++ order among equal ratios is unspecified, and two
/// equally valuable blocks still have to be transformed in SOME order — collection order is the one
/// that is reproducible.
pub(crate) fn sort_blocks(blocks: &mut [ScalarOpMergingBlock]) {
    // `GetBlockRatio(a) > GetBlockRatio(b)` — descending.
    blocks.sort_by(|a, b| BlockRatio::of(b).cmp(&BlockRatio::of(a)));
}

/// Replaces: e168_unrollBurstAndIL
///
/// FIELD UNROLLS one transfer: replaces it with one copy per speculative immutable, chained through
/// `mutable_addr`, each unbursted, un-interleaved and incrementing by zero (`:1011-1062`). Answers
/// the values the created ops bind, in creation order.
///
/// ⛔ THE EXTRA ADDS THE IL CASE NEEDS ARE NOT EMITTED HERE and that is the reference: its own note
/// says the merging candidate absorbs them, and `mutable_addr` chaining is all it creates.
pub(crate) fn unroll_burst_and_il(
    target: UnrollTarget,
    region: &mut Vec<Op>,
    evaluator: &mut dyn ExpressionEvaluator,
    sites: &mut OffsetSites<'_>,
) -> Vec<Val> {
    // `zero_const` — ONE for the whole unroll, in the const builder's block (`:1025-1027`).
    let zero = sites.values.mint();
    sites.consts.push(Op::Sentient(ops::Op::ScalarConstant {
        value: 0,
        result: zero,
        reg_locale: ops::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    }));

    // ⭐ EVERY `buildOffsetValue` BEFORE THE FIRST CLONE, because it can insert into `region` too
    // (see [`OffsetSites::query_maps`]) and a position taken before it would drift. Its ops go to the
    // START of whichever block it uses, so hoisting them out of the loop reorders nothing.
    // "Ops are unrolled top down but the speculative immutables are in reverse order."
    let mut immutables: Vec<Val> = Vec::with_capacity(target.immutables.len());
    for immutable in target.immutables.iter().rev() {
        immutables.push(evaluator.build_offset_value_of(
            *immutable,
            sites,
            region,
            ScalarTy::Index,
        ));
    }

    // `OpBuilder builder(op); builder.setInsertionPointAfter(op);` — every clone lands directly after
    // the original, in creation order, so one index taken here serves the whole loop.
    let Some(at) = region
        .iter()
        .position(|op| results(op).contains(&target.result))
    else {
        // ⭐ UNREACHABLE, AND A NO-OP RATHER THAN A REFUSAL: [`UnrollTarget::of`] found this op in
        // this region, and building an offset value only ever creates ops. Nothing was inserted, so
        // nothing is reported as created.
        return Vec::new();
    };
    let template = region[at].clone();

    let mut input_to_next_op = target.mutable_addr;
    let mut created: Vec<Val> = Vec::with_capacity(immutables.len());
    for (offset, immutable) in immutables.into_iter().enumerate() {
        let result = sites.values.mint();
        let Some(clone) = unrolled_clone(&template, input_to_next_op, immutable, zero, result)
        else {
            // ⭐ UNREACHABLE for the same reason as above: [`UnrollTarget::of`] admitted only the two
            // transfers [`unrolled_clone`] matches.
            break;
        };
        region.insert(at + 1 + offset, clone);
        input_to_next_op = result;
        created.push(result);
    }

    if let Some(&last) = created.last() {
        // ⛔ THE ORDER IS LOAD-BEARING: erasing first would take the uses with the op.
        replace_all_uses_with(region, target.result, last);
        erase_defining_op(region, target.result);
    }
    created
}

/// One `builder.clone(*op)` with its three addresses re-pointed, a fresh result, and burst and
/// interleaving cleared — `None` for anything but the two transfers [`UnrollTarget::of`] admits.
fn unrolled_clone(
    template: &Op,
    mutable_addr: Val,
    immutable_addr: Val,
    increment: Val,
    result: Val,
) -> Option<Op> {
    let mut clone = template.clone();
    let Op::Sentient(
        ops::Op::LoadAndSend {
            mutable_addr: into_mutable,
            immutable_addr: into_immutable,
            increment: into_increment,
            result: into_result,
            extent,
            interleaved_group,
            ..
        }
        | ops::Op::ReceiveAndStore {
            mutable_addr: into_mutable,
            immutable_addr: into_immutable,
            increment: into_increment,
            result: into_result,
            extent,
            interleaved_group,
            ..
        },
    ) = &mut clone
    else {
        return None;
    };
    *into_mutable = mutable_addr;
    *into_immutable = immutable_addr;
    *into_increment = increment;
    // ⭐ A CLONE BINDS A FRESH VALUE, which is the identity the created list is made of.
    *into_result = result;
    extent.burst_size = Elements(0);
    *interleaved_group = Elements(0);
    Some(clone)
}

/// Replaces: e169_findFieldUnrollCandidate
///
/// The block's field unroll candidate for `op`, and only once it has been marked for unrolling
/// (`:1294-1302`).
#[must_use]
pub(crate) fn find_field_unroll_candidate(
    op: Val,
    block: &ScalarOpMergingBlock,
) -> Option<FieldUnrollData> {
    block
        .unroll_candidates
        .iter()
        .find(|candidate| candidate.op == Some(op) && candidate.marked_for_unrolling)
        .cloned()
}

// crustify:todo: e362_isFieldUnrollCandidate
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1080  (124 body lines, level 1)
//   original  : bool ScalarOpMerging::isFieldUnrollCandidate( FieldUnrollData &candidate, const EvaluatedValue *&merging_increment)
//   calls     : e157_doesImmutableImmExceedRange, e161_addSpeculativeImmutable

// crustify:todo: e363_markFieldUnrollingCandidates
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1206  (87 body lines, level 1)
//   original  : void ScalarOpMerging::markFieldUnrollingCandidates()
//   calls     : e157_doesImmutableImmExceedRange, e166_isProfitableForHoisting, e167_sortBlocks

// crustify:todo: e364_doScalarOpMerging
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1334  (135 body lines, level 1)
//   original  : void ScalarOpMerging::doScalarOpMerging(ScalarOpMergingBlock &block)
//   calls     : e166_isProfitableForHoisting, e169_findFieldUnrollCandidate

/// `ScalarOpMerging::isFieldUnrollCandidate` (`:1080`) — whether the burst/IL transfer's speculative
/// immutables all fit, advancing the merging increment past them if they do.
///
/// ⛔ IT IS e362 AND IT IS NOT PORTED YET.
/// ⛔ IT TAKES BOTH FLAVOURS OF THE INCREMENT where the reference takes one
/// `const EvaluatedValue *&`: updating only the handle would leave the decoded reading stale for the
/// next arm's LRF test, and decoding a handle is out of campaign scope (see
/// [`MergingIncrement`](super::scalar_op_hoisting::MergingIncrement)).
fn is_field_unroll_candidate(
    candidate: &mut FieldUnrollData,
    merging_increment: &mut EvaluatedValue,
    merging_increment_ev: &mut Evaluation,
) -> bool {
    let _ = (candidate, merging_increment, merging_increment_ev);
    todo!(
        "isFieldUnrollCandidate (senpass e362, ScalarOpMergingAndHoisting.cpp:1080) is not ported \
         yet — whether the transfer's speculative immutables all stay in range"
    )
}

/// `Value::getDefiningOp()` RESTRICTED TO ONE REGION — `analyze_op->getParentRegion() !=
/// parent_region` (`:659`) expressed as a lookup, because [`defining_op`] recurses into nested ones.
///
/// [`defining_op`]: crate::islands::sentient::dialects::defining_op
fn defining_op_in_region(val: Val, region: &[Op]) -> Option<&Op> {
    region.iter().find(|op| results(op).contains(&val))
}

/// Replaces: e528_buildBlock
///
/// Walks the operand chain up from one merge candidate collecting every op that can absorb the
/// running merging increment, and keeps the block only where it reached more than one op.
///
/// ⛔ THE SKIP SET GAINS THE OP THE WALK IS ABOUT TO DECLINE (`:653`), breaks included, so e577
/// never re-analyses one — and here it also gains the chain's final input value, which defines no op
/// at all and so can never collide with a candidate.
/// ⛔ TWO USES END THE CHAIN AND NO USES DOES NOT (`:654`): a dead op is still merged through.
/// ⛔ `setInputValueToBlock` IS OVERWRITTEN EVERY STEP — what survives is the last step's input.
/// ⛔ THE ADD/SUB ARMS DECLINE ON A MISSING `element_size` AFTER SUMMING (`:673-679`), so the
/// increment the evaluator memoised advances even when the block ends there.
/// ⭐ THE UNROLL ARM COMMITS ITS INCREMENT ONLY IF e362 ACCEPTS (`:769-773`), and the in-range arm's
/// new increment is the DIFFERENCE the new immutable absorbed rather than a further sum (`:806`).
pub(crate) fn build_block<A: Arch, E: ExpressionEvaluator>(
    merge_candidate: Val,
    region: &[Op],
    ibuff_space: IbuffSpace,
    comp: ScalarOpComp,
    scale: AddressScale,
    evaluator: &mut E,
    analyzed_ops: &mut BTreeSet<Val>,
    blocks: &mut Vec<ScalarOpMergingBlock>,
) {
    let mut remaining_ibuff = InstructionCount(ibuff_space.0);
    let mut increment = evaluator.constant(0);
    let mut increment_ev = evaluator.constant_evaluation(0);
    let mut block = ScalarOpMergingBlock::default();
    let regions: [&[Op]; 1] = [region];
    let defs = Definitions::from_innermost(&regions);
    let mut analyze = Some(merge_candidate);
    while let Some(op_val) = analyze {
        analyzed_ops.insert(op_val);
        if use_count(op_val, region) > 1 {
            break;
        }
        let Some(op) = defining_op_in_region(op_val, region) else {
            break;
        };
        let input_value = match op {
            Op::Sentient(ops::Op::ScalarAdd {
                lhs,
                rhs,
                element_size,
                ..
            }) => {
                let Some(const_operand_idx) = first_const_operand_index(op, defs) else {
                    break;
                };
                let operand = operands(op)[const_operand_idx];
                let operand_ev = evaluator.evaluate_value(operand);
                increment_ev = evaluator.evaluate_sum(&increment_ev, &operand_ev);
                let operand_handle = evaluator.evaluate_value_handle(operand);
                increment = evaluator.evaluate_sum_handle(increment, operand_handle);
                let Some(element_size) = *element_size else {
                    break;
                };
                if does_value_exceed_lrf_range::<A>(&increment_ev, element_size, comp, scale) {
                    break;
                }
                record(&mut block, op_val, merge_candidate, evaluator, increment);
                if const_operand_idx == 0 { *rhs } else { *lhs }
            }
            Op::Sentient(ops::Op::ScalarSub {
                lhs,
                rhs,
                element_size,
                ..
            }) => {
                // "Subs are treated as adds so the increment value will be equal to the opposite
                // constant" — which is only expressible when the SUBTRAHEND is the constant.
                if !is_sentient_constant(*rhs, defs) {
                    break;
                }
                let rhs_ev = evaluator.evaluate_value(*rhs);
                increment_ev = evaluator.evaluate_sub(&increment_ev, &rhs_ev);
                let rhs_handle = evaluator.evaluate_value_handle(*rhs);
                increment = evaluator.evaluate_sub_handle(increment, rhs_handle);
                let Some(element_size) = *element_size else {
                    break;
                };
                if does_value_exceed_lrf_range::<A>(&increment_ev, element_size, comp, scale) {
                    break;
                }
                record(&mut block, op_val, merge_candidate, evaluator, increment);
                *lhs
            }
            Op::Sentient(
                ops::Op::LoadAndSend { .. }
                | ops::Op::ReceiveAndStore { .. }
                | ops::Op::LoadComputeAndSend { .. },
            ) => {
                let Some(mem_info) = MemoryOpInfo::of(op) else {
                    break;
                };
                // `calculateIBuffRequired<LAS|RAS>`, and 0 for LCAS — which is what
                // [`BurstAndIl::of`]'s own `isa<>` pair already says.
                let required_ibuff =
                    BurstAndIl::of(op).map_or(InstructionCount(0), calculate_ibuff_required);
                let Some(immutable_addr_op) = defs.of(mem_info.immutable_addr) else {
                    break;
                };
                if mem_info.burst.0 > 1 || mem_info.il.0 > 0 {
                    if DISABLE_FIELD_UNROLLING {
                        break;
                    }
                    if !matches!(
                        immutable_addr_op,
                        Op::Sentient(ops::Op::ScalarConstant { .. } | ops::Op::ScalarCopy { .. })
                    ) {
                        break;
                    }
                    if required_ibuff > remaining_ibuff {
                        break;
                    }
                    let mut unroll_candidate = FieldUnrollData {
                        op: Some(op_val),
                        cost: required_ibuff,
                        ..FieldUnrollData::default()
                    };
                    let mut unroll_increment = increment;
                    let mut unroll_increment_ev = increment_ev.clone();
                    if !is_field_unroll_candidate(
                        &mut unroll_candidate,
                        &mut unroll_increment,
                        &mut unroll_increment_ev,
                    ) {
                        break;
                    }
                    increment = unroll_increment;
                    increment_ev = unroll_increment_ev;
                    remaining_ibuff.0 -= required_ibuff.0;
                    let zero = evaluator.constant(0);
                    block.add_op_to_block(
                        OperationData {
                            op: op_val,
                            mod_by: zero,
                            merging_increment: increment,
                            replace_with_mod: false,
                        },
                        false,
                    );
                    block.add_unroll_candidate(unroll_candidate);
                } else {
                    if !is_sentient_constant(mem_info.immutable_addr, defs) {
                        todo!(
                            "buildBlock: DT_CHECK_MSG(isConstant<ConstantOp>(immutable_addr_), \
                             \"Expect constant immutable address\") \
                             (ScalarOpMergingAndHoisting.cpp:788)"
                        )
                    }
                    if !is_sentient_constant(mem_info.increment, defs) {
                        todo!(
                            "buildBlock: DT_CHECK_MSG(isConstant<ConstantOp>(increment_), \"Expect \
                             constant increment\") (ScalarOpMergingAndHoisting.cpp:791)"
                        )
                    }
                    let immutable_addr_ev = evaluator.evaluate_value(mem_info.immutable_addr);
                    let op_increment_ev = evaluator.evaluate_value(mem_info.increment);
                    // `immutable - increment - merging_increment`.
                    let new_immutable_addr_ev = {
                        let rebased = evaluator.evaluate_sub(&immutable_addr_ev, &op_increment_ev);
                        evaluator.evaluate_sub(&rebased, &increment_ev)
                    };
                    let immutable_addr = evaluator.evaluate_value_handle(mem_info.immutable_addr);
                    let op_increment = evaluator.evaluate_value_handle(mem_info.increment);
                    let new_immutable_addr = {
                        let rebased = evaluator.evaluate_sub_handle(immutable_addr, op_increment);
                        evaluator.evaluate_sub_handle(rebased, increment)
                    };
                    if !is_immutable_value_in_range(
                        &new_immutable_addr_ev,
                        &immutable_addr_ev,
                        mem_info.element_size,
                        op,
                    ) {
                        break;
                    }
                    // `immutable - (immutable - increment - merging_increment)`.
                    increment_ev =
                        evaluator.evaluate_sub(&immutable_addr_ev, &new_immutable_addr_ev);
                    increment = evaluator.evaluate_sub_handle(immutable_addr, new_immutable_addr);
                    block.add_op_to_block(
                        OperationData {
                            op: op_val,
                            mod_by: new_immutable_addr,
                            merging_increment: increment,
                            replace_with_mod: false,
                        },
                        false,
                    );
                }
                mem_info.mutable_addr
            }
            _ => break,
        };
        block.input_value_to_block = Some(input_value);
        analyze = Some(input_value);
    }

    if block.block_ops.len() > 1 {
        blocks.push(block);
    }
}

/// The add/sub arms' shared tail (`:692-699`, `:729-736`) — the op is recorded with a modification of
/// ZERO because merging deletes it, and anything above the candidate is what enables merging.
fn record<E: ExpressionEvaluator>(
    block: &mut ScalarOpMergingBlock,
    op: Val,
    merge_candidate: Val,
    evaluator: &mut E,
    merging_increment: EvaluatedValue,
) {
    let zero = evaluator.constant(0);
    block.add_op_to_block(
        OperationData {
            op,
            mod_by: zero,
            merging_increment,
            replace_with_mod: false,
        },
        true,
    );
    if op != merge_candidate {
        block.set_enables_merging();
    }
}

/// Replaces: e577_collectBlocks
///
/// Searches the region bottom-up for a scalar add/sub merging candidate and builds a block from each
/// one the operand chains walked so far have not already absorbed (`:869-895`).
///
/// ⭐ ONE LOOP FOR THE REFERENCE'S TWO: a region at this rung is a single flat block.
/// ⛔ THE SKIP SET IS KEYED BY THE RESULT AN OP BINDS, which is all [`build_block`] ever writes into
/// it; an op that binds nothing takes the same `continue` the reference's `else` does.
pub(crate) fn collect_blocks<A: Arch, E: ExpressionEvaluator>(
    region: &[Op],
    ibuff_space: IbuffSpace,
    comp: ScalarOpComp,
    scale: AddressScale,
    evaluator: &mut E,
    blocks: &mut Vec<ScalarOpMergingBlock>,
) {
    let mut analyzed_ops: BTreeSet<Val> = BTreeSet::new();
    for op in region.iter().rev() {
        let Some(op_val) = results(op).first().copied() else {
            continue;
        };
        if !analyzed_ops.insert(op_val) {
            continue;
        }
        if matches!(
            op,
            Op::Sentient(ops::Op::ScalarAdd { .. } | ops::Op::ScalarSub { .. })
        ) {
            build_block::<A, E>(
                op_val,
                region,
                ibuff_space,
                comp,
                scale,
                evaluator,
                &mut analyzed_ops,
                blocks,
            );
        }
    }
}

/// WHERE THE PASS IS RUN — `ScalarOpMerging(opt_context, region, ibuff_space)` is constructed on a
/// loop body (`:2320`) and on the program unit's own region (`:2376`), and `region_.getParentOp()` is
/// the difference [`is_profitable_for_hoisting`] asks about.
///
/// ⛔ IT IS ONE PARAMETER BECAUSE THE TWO READINGS OVERLAP: the parent op OWNS the region it hands
/// out, so a caller cannot pass `&mut Vec<Op>` and `Option<&Op>` for the same loop separately.
pub(crate) enum MergingRegion<'a> {
    /// `for_op.getLoopBody()` — the `sentient.for` whose body is the region.
    LoopBody(&'a mut Op),
    /// `unit.getRegion()`, which is no loop body, so hoisting is never profitable for its blocks.
    UnitRegion(&'a mut Vec<Op>),
}

impl MergingRegion<'_> {
    /// `region_`, read.
    ///
    /// ⭐ AN EMPTY REGION FOR A [`MergingRegion::LoopBody`] THAT IS NOT A LOOP: the reference's own
    /// `DT_CHECK_MSG(for_op, "Node is not a for_op as expected.")` (`:2303`) makes it unreachable, and
    /// a region with no op in it collects no block and transforms nothing.
    fn region(&self) -> &[Op] {
        match self {
            Self::LoopBody(for_op) => match &**for_op {
                Op::Sentient(ops::Op::For { body, .. }) => body,
                _ => &[],
            },
            Self::UnitRegion(region) => region,
        }
    }

    /// `region_.getParentOp()`, which only a loop body has.
    fn parent_op(&self) -> Option<&Op> {
        match self {
            Self::LoopBody(for_op) => Some(for_op),
            Self::UnitRegion(_) => None,
        }
    }

    /// The same region, borrowed for shorter — one pass member handed to two calls in turn.
    fn reborrow(&mut self) -> MergingRegion<'_> {
        match self {
            Self::LoopBody(for_op) => MergingRegion::LoopBody(for_op),
            Self::UnitRegion(region) => MergingRegion::UnitRegion(region),
        }
    }
}

/// `ScalarOpMerging::markFieldUnrollingCandidates` (`:1206`) — spends the unit's remaining IBuff on
/// the sorted blocks' unroll candidates, whole block first and candidate by candidate otherwise.
///
/// ⛔ IT IS e363 AND IT IS NOT PORTED YET. [`sort_blocks`] and [`is_profitable_for_hoisting`], the two
/// units it opens with, are ported; the LDSTI imm window its corner case measures against arrives
/// with e361's `OptimizationContext`, as does the decoding of a stored merging increment.
fn mark_field_unrolling_candidates(
    blocks: &mut [ScalarOpMergingBlock],
    parent_op: Option<&Op>,
    ibuff_space: &mut IbuffSpace,
    scale: AddressScale,
    evaluator: &mut dyn ExpressionEvaluator,
) {
    let _ = (blocks, parent_op, ibuff_space, scale, evaluator);
    todo!(
        "markFieldUnrollingCandidates (senpass e363, ScalarOpMergingAndHoisting.cpp:1206) is not \
         ported yet — which unroll candidates the remaining IBuff pays for"
    )
}

/// `ScalarOpMerging::doScalarOpMerging` (`:1334`) — merges one block: the chain's scalar ops collapse
/// into the composite transfers that absorb them, and each field-unroll candidate is unrolled first.
///
/// ⛔ IT IS e364 AND IT IS NOT PORTED YET. [`is_profitable_for_hoisting`], [`unroll_burst_and_il`] and
/// [`find_field_unroll_candidate`], the three units it delegates to, are ported and waiting for it.
fn do_scalar_op_merging(
    block: &ScalarOpMergingBlock,
    region: MergingRegion<'_>,
    evaluator: &mut dyn ExpressionEvaluator,
    sites: &mut OffsetSites<'_>,
) {
    let _ = (block, region, evaluator, sites);
    todo!(
        "doScalarOpMerging (senpass e364, ScalarOpMergingAndHoisting.cpp:1334) is not ported yet — \
         the merge of one block's chain into the transfers that absorb it"
    )
}

/// Replaces: e611_runScalarOpMerging
///
/// The whole `ScalarOpMerging` pass, which the reference's constructor runs (`:462-464`): collect the
/// region's merging blocks, spend the unit's IBuff marking field-unroll candidates, then merge each
/// block in the order that spending left them in (`:569-577`).
///
/// ⭐ NOTHING BUT THE REGION SURVIVES THE PASS: `getBlocks()` and `getRegion()` (`:467-468`) have no
/// reader in the file, so `blocks_` and the spent `ibuff_space_` are the pass's own bookkeeping.
pub(crate) fn run_scalar_op_merging<A: Arch, E: ExpressionEvaluator>(
    mut region: MergingRegion<'_>,
    ibuff_space: IbuffSpace,
    comp: ScalarOpComp,
    scale: AddressScale,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
) {
    let mut ibuff_space = ibuff_space;
    let mut blocks: Vec<ScalarOpMergingBlock> = Vec::new();
    collect_blocks::<A, E>(
        region.region(),
        ibuff_space,
        comp,
        scale,
        evaluator,
        &mut blocks,
    );
    if blocks.is_empty() {
        return;
    }
    // Order blocks with respect to their benefit vs IBuff impact and mark operations in the blocks
    // for Field Unrolling.
    mark_field_unrolling_candidates(
        &mut blocks,
        region.parent_op(),
        &mut ibuff_space,
        scale,
        evaluator,
    );
    for block in &blocks {
        do_scalar_op_merging(block, region.reborrow(), evaluator, sites);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::ScalarOpCount;
    use super::*;
    use crate::arch::Dd2;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::transform::sentient::analyses::{EvaluatedValue, Evaluation, Offsets, ScalarOffset};

    /// The out-of-scope evaluator, stating the ONE answer these units consume.
    struct StatedEvaluator {
        offsets: Vec<(EvaluatedValue, Val)>,
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            todo!("no unit of this file evaluates a value")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("no unit of this file evaluates a sum")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("no unit of this file builds an offset from a fresh evaluation")
        }

        fn build_offset_value_of(
            &mut self,
            immutable: EvaluatedValue,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            self.offsets
                .iter()
                .find(|(of, _)| *of == immutable)
                .map_or(Val(0), |(_, val)| *val)
        }
    }

    fn values_after(issued: u32) -> Values {
        let mut values = Values::default();
        for _ in 0..issued {
            values.mint();
        }
        values
    }

    fn load_and_send(
        mutable_addr: Val,
        immutable_addr: Val,
        increment: Val,
        result: Val,
        burst: u64,
        interleaved_group: u64,
    ) -> Op {
        Op::Sentient(ops::Op::LoadAndSend {
            mutable_addr,
            immutable_addr,
            increment,
            consumer: SendEnd::to_self(Val(9)),
            result,
            extent: ops::Extent {
                total_elements: Elements(4),
                element_size: Bits(32),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(burst),
            },
            interleaved_group: Elements(interleaved_group),
            rotate_val: None,
            dir: None,
            shuffle_mode: ops::ShuffleMode::NoShuffle,
            reg: ops::Reg {
                locale: ops::RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    fn carried(init: Val, arg: Val, result: Val) -> ops::Carried {
        ops::Carried {
            init,
            arg,
            result,
            reg: ops::Reg {
                locale: ops::RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    fn for_loop(carried: Vec<ops::Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(ops::Op::For {
            iv: Val(30),
            bound: Val(31),
            bound_reg: None,
            carried,
            dbg_name: None,
            body,
        })
    }

    fn bottom_op(op: Val) -> OperationData {
        OperationData {
            op,
            mod_by: EvaluatedValue(0),
            merging_increment: EvaluatedValue(0),
            replace_with_mod: false,
        }
    }

    fn block_of(ops: u32, slots: i32) -> ScalarOpMergingBlock {
        ScalarOpMergingBlock {
            num_scalar_ops_in_block: ScalarOpCount(ops),
            required_ibuff: InstructionCount(slots),
            ..ScalarOpMergingBlock::default()
        }
    }

    /// AN EVALUATOR THAT ACTUALLY SUMS — `ExpressionEvaluator` is out of campaign scope, and the
    /// running increment is the one answer e528 consumes. Every handle is a fresh arena entry, so the
    /// numbering below IS the order the walk asked its questions in.
    #[derive(Default)]
    struct SummingEvaluator {
        interned: u32,
    }

    impl SummingEvaluator {
        fn intern(&mut self) -> EvaluatedValue {
            self.interned += 1;
            EvaluatedValue(self.interned - 1)
        }
    }

    impl ExpressionEvaluator for SummingEvaluator {
        /// The fixture's one constant.
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            absolute(8)
        }

        fn evaluate_sum(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation {
            absolute(offset_of(lhs) + offset_of(rhs))
        }

        fn constant_evaluation(&mut self, value: i64) -> Evaluation {
            absolute(value)
        }

        fn constant(&mut self, _value: i64) -> EvaluatedValue {
            self.intern()
        }

        fn evaluate_value_handle(&mut self, _value: Val) -> EvaluatedValue {
            self.intern()
        }

        fn evaluate_sum_handle(
            &mut self,
            _lhs: EvaluatedValue,
            _rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            self.intern()
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e528 builds no offset value")
        }
    }

    fn absolute(offset: i64) -> Evaluation {
        Evaluation {
            known_absolute: true,
            base: None,
            offsets: Offsets::AllUnit(ScalarOffset(offset)),
        }
    }

    fn offset_of(evaluation: &Evaluation) -> i64 {
        evaluation
            .all_unit_offset()
            .map_or(0, |offset: ScalarOffset| offset.0)
    }

    fn add_sized(lhs: Val, rhs: Val, result: Val, element_size: Bits) -> Op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: Some(element_size),
        })
    }

    fn scalar_constant(value: i64, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: ops::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn candidate_of(op: Val, cost: i32, immutables: Vec<EvaluatedValue>) -> FieldUnrollData {
        FieldUnrollData {
            op: Some(op),
            cost: InstructionCount(cost),
            speculative_immutables: immutables,
            ..FieldUnrollData::default()
        }
    }

    /// The vendor's own worked example (`:936-940`): burst 2 × IL 2 is four transfers, three of them
    /// beyond the IBuff entry the op already holds.
    #[test]
    fn a_burst_two_il_two_transfer_costs_three_further_ibuff_entries() {
        let transfer = BurstAndIl::of(&load_and_send(Val(1), Val(2), Val(3), Val(5), 2, 2));
        assert_eq!(
            transfer,
            Some(BurstAndIl {
                burst: Elements(2),
                interleaved_group: Elements(2),
            })
        );
        assert_eq!(
            calculate_ibuff_required(transfer.expect("a load_and_send has both counts")),
            InstructionCount(3)
        );
        // Unbursted and un-interleaved: the op's own entry is the whole cost.
        assert_eq!(
            calculate_ibuff_required(BurstAndIl {
                burst: Elements(0),
                interleaved_group: Elements(0),
            }),
            InstructionCount(0)
        );
        // ⛔ AND THE `isa<>` PAIR IS THE CONSTRUCTOR: a scalar add has no burst to cost anything.
        assert_eq!(BurstAndIl::of(&add(Val(1), Val(2), Val(3))), None);
    }

    #[test]
    fn a_block_whose_bottom_op_yields_the_slot_it_reads_is_worth_hoisting() {
        let body = vec![
            add(Val(3), Val(9), Val(5)),
            Op::Sentient(ops::Op::Yield {
                results: vec![Val(5)],
            }),
        ];
        let block = ScalarOpMergingBlock {
            input_value_to_block: Some(Val(3)),
            block_ops: vec![bottom_op(Val(5))],
            ..ScalarOpMergingBlock::default()
        };
        let loop_op = for_loop(vec![carried(Val(1), Val(3), Val(4))], body.clone());
        assert!(is_profitable_for_hoisting(&block, Some(&loop_op)));

        // ⛔ THE NEGATIVE THE INDEX PAIRING EXISTS FOR: the same chain, yielded at ANOTHER slot. The
        // block reads iter arg 1 and the yield hands its result back at position 0.
        let mispaired = for_loop(
            vec![
                carried(Val(0), Val(2), Val(7)),
                carried(Val(1), Val(3), Val(4)),
            ],
            vec![
                add(Val(3), Val(9), Val(5)),
                Op::Sentient(ops::Op::Yield {
                    results: vec![Val(5), Val(8)],
                }),
            ],
        );
        assert!(!is_profitable_for_hoisting(&block, Some(&mispaired)));
    }

    #[test]
    fn blocks_sort_by_benefit_and_needing_no_ibuff_wins_outright() {
        let mut blocks = vec![
            block_of(5, 3),
            // ⛔ THE SENTINEL'S OWN CASE: 2000 : 1 is above `1000.0`, and it must still lose.
            block_of(2000, 1),
            block_of(1, 0),
            block_of(5, 2),
        ];
        sort_blocks(&mut blocks);
        assert_eq!(
            blocks
                .iter()
                .map(|block| (block.num_scalar_ops_in_block.0, block.required_ibuff.0))
                .collect::<Vec<_>>(),
            vec![(1, 0), (2000, 1), (5, 2), (5, 3)]
        );
    }

    /// The vendor's worked example again (`:936-968`), unrolled: one `load_and_send` with burst 2 and
    /// IL 2 becomes four, chained through `mutable_addr`, and its reader follows the last of them.
    #[test]
    fn a_burst_two_il_two_load_unrolls_into_four_chained_loads() {
        let mut region = vec![
            load_and_send(Val(1), Val(2), Val(3), Val(5), 2, 2),
            add(Val(5), Val(9), Val(6)),
        ];
        // The immutables come in REVERSE program order, as `isFieldUnrollCandidate` collects them.
        let candidate = candidate_of(
            Val(5),
            3,
            vec![
                EvaluatedValue(3),
                EvaluatedValue(2),
                EvaluatedValue(1),
                EvaluatedValue(0),
            ],
        );
        let target = UnrollTarget::of(&candidate, &region).expect("the op is in the region");

        let mut evaluator = StatedEvaluator {
            offsets: (0..4).map(|i| (EvaluatedValue(i), Val(100 + i))).collect(),
        };
        let mut consts: Vec<Op> = Vec::new();
        let mut values = values_after(20);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let created = unroll_burst_and_il(target, &mut region, &mut evaluator, &mut sites);

        assert_eq!(created, vec![Val(21), Val(22), Val(23), Val(24)]);
        assert_eq!(
            region,
            vec![
                load_and_send(Val(1), Val(100), Val(20), Val(21), 0, 0),
                load_and_send(Val(21), Val(101), Val(20), Val(22), 0, 0),
                load_and_send(Val(22), Val(102), Val(20), Val(23), 0, 0),
                load_and_send(Val(23), Val(103), Val(20), Val(24), 0, 0),
                add(Val(24), Val(9), Val(6)),
            ]
        );
        // The zero increment is created once, in the const builder's block.
        assert_eq!(
            consts,
            vec![Op::Sentient(ops::Op::ScalarConstant {
                value: 0,
                result: Val(20),
                reg_locale: ops::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            })]
        );
    }

    #[test]
    fn only_a_candidate_marked_for_unrolling_is_found() {
        let mut marked = candidate_of(Val(5), 3, Vec::new());
        marked.marked_for_unrolling = true;
        let block = ScalarOpMergingBlock {
            unroll_candidates: vec![candidate_of(Val(4), 1, Vec::new()), marked.clone()],
            ..ScalarOpMergingBlock::default()
        };
        assert_eq!(find_field_unroll_candidate(Val(5), &block), Some(marked));
        // ⛔ THE UNMARKED ONE IS NOT AN ANSWER: `markFieldUnrollingCandidates` has not chosen it, so
        // there is no IBuff reserved for it.
        assert_eq!(find_field_unroll_candidate(Val(4), &block), None);
    }

    /// e528 — a chain of two adds over the same constant becomes ONE block whose second op carries the
    /// summed increment, ending at the value from outside it.
    #[test]
    fn a_chain_of_two_adds_is_one_block_ending_at_the_value_from_outside_it() {
        let region = vec![
            scalar_constant(8, Val(2)),
            add_sized(Val(1), Val(2), Val(4), Bits(16)),
            add_sized(Val(4), Val(2), Val(6), Bits(16)),
        ];
        let mut analyzed = BTreeSet::new();
        let mut blocks = Vec::new();
        build_block::<Dd2, _>(
            Val(6),
            &region,
            IbuffSpace(8),
            ScalarOpComp::Lxlu,
            AddressScale::ONE,
            &mut SummingEvaluator::default(),
            &mut analyzed,
            &mut blocks,
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].input_value_to_block, Some(Val(1)));
        assert_eq!(blocks[0].num_scalar_ops_in_block, ScalarOpCount(2));
        assert_eq!(
            blocks[0].block_ops,
            vec![
                OperationData {
                    op: Val(6),
                    mod_by: EvaluatedValue(3),
                    merging_increment: EvaluatedValue(2),
                    replace_with_mod: false,
                },
                OperationData {
                    op: Val(4),
                    mod_by: EvaluatedValue(6),
                    merging_increment: EvaluatedValue(5),
                    replace_with_mod: false,
                },
            ]
        );
        // ⛔ `%1` IS IN THE SKIP SET although it defines no op — see [`build_block`]'s first trap.
        assert_eq!(analyzed, BTreeSet::from([Val(1), Val(4), Val(6)]));
    }
    /// 577/656 — the bottom add of the chain is the only candidate the walk builds from: the add above
    /// it is already in the skip set, and the constant they share is not an add at all.
    #[test]
    fn e577_collects_one_block_from_the_bottom_add_of_a_chain() {
        let region = vec![
            scalar_constant(8, Val(2)),
            add_sized(Val(1), Val(2), Val(4), Bits(16)),
            add_sized(Val(4), Val(2), Val(6), Bits(16)),
        ];
        let mut blocks = Vec::new();

        collect_blocks::<Dd2, _>(
            &region,
            IbuffSpace(8),
            ScalarOpComp::Lxlu,
            AddressScale::ONE,
            &mut SummingEvaluator::default(),
            &mut blocks,
        );

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].input_value_to_block, Some(Val(1)));
        assert_eq!(blocks[0].num_scalar_ops_in_block, ScalarOpCount(2));
    }
    /// e611 — the pass collects the chain's one block and then reaches the candidate marking, which is
    /// e363 and not ported.
    #[test]
    #[should_panic(expected = "senpass e363")]
    fn a_region_with_one_merging_block_reaches_the_unported_candidate_marking() {
        let mut region = vec![
            scalar_constant(8, Val(2)),
            add_sized(Val(1), Val(2), Val(4), Bits(16)),
            add_sized(Val(4), Val(2), Val(6), Bits(16)),
        ];
        let mut consts = Vec::new();
        let mut values = values_after(10);
        run_scalar_op_merging::<Dd2, _>(
            MergingRegion::UnitRegion(&mut region),
            IbuffSpace(8),
            ScalarOpComp::Lxlu,
            AddressScale::ONE,
            &mut SummingEvaluator::default(),
            &mut OffsetSites {
                consts: &mut consts,
                query_maps: None,
                values: &mut values,
            },
        );
    }

    /// ⭐ `if (blocks_.empty()) return;` — a region with no mergeable chain in it is left exactly as it
    /// was, and neither unported unit is reached.
    #[test]
    fn a_region_with_no_merging_candidate_is_left_alone() {
        let mut region = vec![
            scalar_constant(8, Val(2)),
            load_and_send(Val(1), Val(2), Val(3), Val(5), 0, 0),
        ];
        let untouched = region.clone();
        let mut consts = Vec::new();
        let mut values = values_after(10);
        run_scalar_op_merging::<Dd2, _>(
            MergingRegion::UnitRegion(&mut region),
            IbuffSpace(8),
            ScalarOpComp::Lxlu,
            AddressScale::ONE,
            &mut SummingEvaluator::default(),
            &mut OffsetSites {
                consts: &mut consts,
                query_maps: None,
                values: &mut values,
            },
        );
        assert_eq!(region, untouched);
        assert!(consts.is_empty());
    }
}
