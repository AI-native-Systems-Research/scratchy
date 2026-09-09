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

use super::{
    AddressScale, BurstAndIl, FieldUnrollData, ImmRange, MemoryOpInfo, OperationData, ScalarOpComp,
    ScalarOpCount, ScalarOpMergingBlock, UnrollTarget, does_immutable_imm_exceed_range,
};
use crate::arch::Elements;
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, defining_op, element_size, erase_defining_op, operands,
    replace_all_uses_with, results, set_operand, use_count,
};
use crate::transform::sentient::analyses::{
    EvaluatedValue, ExpressionEvaluator, InstructionCount, OffsetSites,
};

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

/// ONE ELEMENT COUNT AS THE REFERENCE'S `int` — every count below is multiplied and divided in `int`
/// arithmetic that [`Elements`]'s `u64` cannot underflow. ⭐ SATURATING: see
/// [`calculate_ibuff_required`].
fn count(elements: Elements) -> i64 {
    i64::try_from(elements.0).unwrap_or(i64::MAX)
}

/// `getLdTypeOrStTypeInElements` (`dcc/src/Dialect/Sentient/Utils.cpp:370-416`) — how many elements one
/// full IL iteration of this transfer advances by.
///
/// ⛔ THE LX ANSWER IS A WINDOW WIDTH IN BYTES DIVIDED BY THE ELEMENT WIDTH, and the shuffle mode is
/// what narrows the window: 128 bytes normally, 2 for a 2-byte splat or mask, 16 for the 16-byte ones
/// (`:388-406`). The L0 answer is the transfer's own `total_elements`.
/// ⭐ `llvm_unreachable` FOR EVERY OTHER SHUFFLE MODE AND OP, kept as the panic it is: `splat`,
/// `rotate` and `splat4b` have no ldtype and this pass may not guess one.
fn ld_type_or_st_type_in_elements(op: &Op, comp: ScalarOpComp) -> Elements {
    let (total_elements, element_size, shuffle_mode) = match op {
        Op::Sentient(ops::Op::LoadAndSend {
            extent,
            shuffle_mode,
            ..
        }) => (
            extent.total_elements,
            extent.element_size,
            Some(*shuffle_mode),
        ),
        Op::Sentient(ops::Op::ReceiveAndStore {
            extent,
            shuffle_mode,
            ..
        }) => (extent.total_elements, extent.element_size, *shuffle_mode),
        _ => panic!("only a transfer that supports IL has an ldtype/sttype (Utils.cpp:384)"),
    };
    if matches!(comp, ScalarOpComp::L0lu | ScalarOpComp::L0su) {
        return total_elements;
    }
    let ldsttype_in_bytes = match shuffle_mode {
        None | Some(ops::ShuffleMode::NoShuffle) => 128,
        Some(ops::ShuffleMode::Splat2B | ops::ShuffleMode::Masked2B) => 2,
        Some(
            ops::ShuffleMode::Splat16B | ops::ShuffleMode::ZeroPad16B | ops::ShuffleMode::Masked16B,
        ) => 16,
        Some(
            mode @ (ops::ShuffleMode::Splat | ops::ShuffleMode::Rotate | ops::ShuffleMode::Splat4B),
        ) => {
            panic!("unexpected shuffle_mode {mode:?} for ldtype/sttype calculation (Utils.cpp:404)")
        }
    };
    // `(float)ldsttype_in_bytes / ((float)element_size / 8.0)` truncated to `int` — the `8.0` promotes
    // the divisor to double, as in [`super::scaled_imm`].
    let ldsttype = (f64::from(ldsttype_in_bytes) / (f64::from(element_size.0) / 8.0)) as i64;
    // `DT_CHECK_MSG(ldsttype > 0, "ldsttype should be greater than 0")` (`:409`) — kept as the abort it
    // is, because an element wider than the window has no ldtype to guess.
    if ldsttype <= 0 {
        panic!("an element wider than this unit's ldtype window has no ldtype (Utils.cpp:409)");
    }
    Elements(ldsttype.unsigned_abs())
}

/// Replaces: e362_isFieldUnrollCandidate
///
/// Whether one bursted or interleaved transfer can be field unrolled: appends the immutable each op the
/// unrolling would create needs, and advances `merging_increment` past the whole unroll (`:1080-1200`).
///
/// ⛔ IT COUNTS THE BURST **DOWN** because merging runs bottom up, so `speculative_immutables` comes out
/// in REVERSE program order — [`UnrollTarget`] is where that is read back.
/// ⛔ THE TWO `DT_CHECK`s ON A CONSTANT IMMUTABLE AND INCREMENT ARE THE CALLER'S `isa<ConstantOp,
/// CopyOp>` GUARD (`:761-762`); `false` here is only ever "the immutable left the IMM range".
/// ⭐ `merging_increment` IS IN/OUT (`const EvaluatedValue *&`) and the caller keeps the advanced value
/// only when the answer is `true` (`:772-774`).
pub(crate) fn is_field_unroll_candidate(
    candidate: &mut FieldUnrollData,
    merging_increment: &mut EvaluatedValue,
    op: &Op,
    region: &[Op],
    comp: ScalarOpComp,
    ldsti_imm_range: ImmRange,
    scale: AddressScale,
    evaluator: &mut dyn ExpressionEvaluator,
) -> bool {
    // ⭐ `MemoryOpInfo` ADMITS EXACTLY WHAT THE CALLER HAS ALREADY MATCHED (`:739`), and an LCAS cannot
    // reach here: its burst is 1 and its IL 0, which is the caller's own `burst > 1 || il > 0` gate.
    let Some(mem_info) = MemoryOpInfo::of(op) else {
        return false;
    };
    // `dyn_cast<sentient::CopyOp>(...getDefiningOp())` — a copied immutable is evaluated at its source
    // (`:1091-1093`).
    let mut immutable_addr = mem_info.immutable_addr;
    if let Some(Op::Sentient(ops::Op::ScalarCopy { input, .. })) =
        defining_op(immutable_addr, region)
    {
        immutable_addr = *input;
    }
    let immutable_addr_ev = evaluator.evaluate_value_handle(immutable_addr);
    let increment_ev = evaluator.evaluate_value_handle(mem_info.increment);

    // The value one full IL iteration adds: the chunk size on an L0 with chunks, the ldtype otherwise
    // (`:1106-1125`). The last burst's own chunk calculation is done inside the loop.
    let ldsttype = count(ld_type_or_st_type_in_elements(op, comp));
    let is_l0 = matches!(comp, ScalarOpComp::L0lu | ScalarOpComp::L0su);
    let modifier = if mem_info.il.0 > 0 && mem_info.chunk_size.0 > 0 && is_l0 {
        count(mem_info.chunk_size)
    } else {
        ldsttype
    };
    let modifier_ev = evaluator.constant(modifier);

    let burst = count(mem_info.burst);
    for b in (1..=burst).rev() {
        if mem_info.il.0 > 0 {
            if mem_info.chunk_size.0 != 0 && b == burst && is_l0 {
                let chunk_mod = ((ldsttype / count(mem_info.chunk_size) - 1)
                    * count(mem_info.chunk_stride))
                    + count(mem_info.chunk_size);
                let chunk_mod_ev = evaluator.constant(chunk_mod);
                *merging_increment =
                    evaluator.evaluate_sum_handle(*merging_increment, chunk_mod_ev);
            } else {
                *merging_increment = evaluator.evaluate_sum_handle(*merging_increment, modifier_ev);
            }

            for i in (0..count(mem_info.il)).rev() {
                let scaled_increment = evaluator.evaluate_multiply_by_const(increment_ev, i);
                let unrolled_immutable_ev =
                    evaluator.evaluate_sum_handle(immutable_addr_ev, scaled_increment);
                // The op is unrolled into no-update ops, so the increment is folded out here.
                let speculative_immutable_ev =
                    evaluator.evaluate_sub_handle(unrolled_immutable_ev, *merging_increment);
                let evaluation = evaluator.evaluation_of(speculative_immutable_ev);
                if does_immutable_imm_exceed_range(
                    &evaluation,
                    mem_info.element_size,
                    ldsti_imm_range,
                    scale,
                ) {
                    return false;
                }
                candidate.add_speculative_immutable(speculative_immutable_ev);
            }
        } else {
            // Without IL the op is repeated `burst` times, each unbursted:
            // `speculative = immutable - increment - merging_increment`, and the increment becomes
            // `immutable - speculative` (`:1177-1198`).
            let without_increment = evaluator.evaluate_sub_handle(immutable_addr_ev, increment_ev);
            let speculative_immutable_ev =
                evaluator.evaluate_sub_handle(without_increment, *merging_increment);
            let evaluation = evaluator.evaluation_of(speculative_immutable_ev);
            if does_immutable_imm_exceed_range(
                &evaluation,
                mem_info.element_size,
                ldsti_imm_range,
                scale,
            ) {
                return false;
            }
            *merging_increment =
                evaluator.evaluate_sub_handle(immutable_addr_ev, speculative_immutable_ev);
            candidate.add_speculative_immutable(speculative_immutable_ev);
        }
    }
    true
}
/// Replaces: e363_markFieldUnrollingCandidates
///
/// Spends the region's remaining IBuff on the most valuable blocks: a block whose whole cost fits and
/// which either enables hoisting or ends in a merging enabler has EVERY candidate marked, and any other
/// block gets the longest prefix of candidates that fits and ends in an enabler (`:1206-1291`).
///
/// ⛔ THE PREFIX IS ONLY BANKED AT AN ENABLER (`:1279-1284`): candidates that fit but are followed by no
/// enabler cost nothing and are not marked, which is why `remaining_ibuff` is a local.
/// ⭐ THE CORNER CASE (`:1225-1257`) IS ABOUT THE **FIRST** OP OF THE CHAIN, which is `block_ops.back()`
/// — a one-add block whose add is that op cannot absorb an increment that leaves the IMM range.
pub(crate) fn mark_field_unrolling_candidates(
    blocks: &mut [ScalarOpMergingBlock],
    ibuff_space: &mut InstructionCount,
    region: &[Op],
    parent_op: Option<&Op>,
    ldsti_imm_range: ImmRange,
    scale: AddressScale,
    evaluator: &mut dyn ExpressionEvaluator,
) {
    sort_blocks(blocks);

    for block in blocks.iter_mut() {
        if ibuff_space.0 <= 0 {
            break;
        }
        if block.unroll_candidates.is_empty() {
            continue;
        }
        let whole_block_fits = ibuff_space.0 >= block.required_ibuff.0
            && (is_profitable_for_hoisting(block, parent_op)
                || block
                    .unroll_candidates
                    .last()
                    .is_some_and(|last| last.enables_merging));

        if whole_block_fits {
            // "block_ops are stored bottom-up, so back() is the first op in the chain".
            let first_block_op = (block.num_scalar_ops_in_block == ScalarOpCount(1))
                .then(|| block.block_ops.last().copied())
                .flatten();
            for candidate in block.unroll_candidates.iter_mut() {
                if let Some(first) = first_block_op {
                    if candidate.op == Some(first.op) {
                        // `llvm_unreachable("Unexpected operation type in unroll candidates")`
                        // (`:1246-1248`) and `DT_CHECK(element_size > 0)`: every candidate is a
                        // transfer, so both are the reference's own aborts.
                        let element_size = defining_op(first.op, region)
                            .and_then(MemoryOpInfo::of)
                            .map_or_else(
                                || {
                                    panic!(
                                        "an unroll candidate is a transfer and has an element_size"
                                    )
                                },
                                |info| info.element_size,
                            );
                        let evaluation = evaluator.evaluation_of(first.merging_increment);
                        if does_immutable_imm_exceed_range(
                            &evaluation,
                            element_size,
                            ldsti_imm_range,
                            scale,
                        ) {
                            continue;
                        }
                    }
                }
                ibuff_space.0 -= candidate.cost.0;
                candidate.marked_for_unrolling = true;
            }
        } else {
            // ⭐ THE `unsigned cost > int remaining_ibuff` PROMOTION (`:1272`) CANNOT BITE: the loop is
            // entered with `ibuff_space_ > 0` and only ever subtracts a cost it has just admitted, so
            // `remaining_ibuff` never goes negative and the comparison is the signed one written here.
            let mut remaining_ibuff = *ibuff_space;
            let mut to_unroll: Vec<usize> = Vec::new();
            let mut marked: Vec<usize> = Vec::new();
            for (at, candidate) in block.unroll_candidates.iter().enumerate() {
                if candidate.cost.0 > remaining_ibuff.0 {
                    break;
                }
                to_unroll.push(at);
                remaining_ibuff.0 -= candidate.cost.0;
                if candidate.enables_merging {
                    *ibuff_space = remaining_ibuff;
                    marked.append(&mut to_unroll);
                }
            }
            for at in marked {
                block.unroll_candidates[at].marked_for_unrolling = true;
            }
        }
    }
}
/// WHERE ONE CREATED OP GOES RELATIVE TO ITS ANCHOR — `OpBuilder builder(op)` against
/// `builder.setInsertionPointAfter(op)` (`:1379`, `:1425`), which a `bool` cannot keep apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// `OpBuilder builder(op)` / `setInsertionPoint(op)` — directly above the anchor.
    Before,
    /// `setInsertionPointAfter(op)` — directly below it.
    After,
}

/// `createRemainingAdd` (`:1358-1372`) — the `sentient.add` carrying whatever merging increment is left,
/// placed relative to the op binding `anchor` and, when `last_mem_op` is given, spliced into that op's
/// operand list in place of `inp`.
///
/// ⭐ THE OFFSET VALUE IS BUILT FIRST, because building it can insert into `region` too and a position
/// taken before it would drift — see [`unroll_burst_and_il`].
/// ⛔ `element_size` IS COPIED FROM THE BLOCK'S ANCHOR OP, and the reference says why: it "is essential
/// for range checks" on the add this leaves behind (`:1352`, `:1369`).
fn create_remaining_add(
    inp: Val,
    merging_increment: EvaluatedValue,
    anchor_element_size: Option<Bits>,
    anchor: Val,
    place: Place,
    last_mem_op: Option<Val>,
    result: Val,
    region: &mut Vec<Op>,
    evaluator: &mut dyn ExpressionEvaluator,
    sites: &mut OffsetSites<'_>,
) {
    let merging_increment_const =
        evaluator.build_offset_value_of(merging_increment, sites, region, ScalarTy::Index);
    let new_add = Op::Sentient(ops::Op::ScalarAdd {
        lhs: inp,
        rhs: merging_increment_const,
        result,
        reg: None,
        element_size: anchor_element_size,
        ty: ScalarTy::Index,
    });
    match region.iter().position(|op| results(op).contains(&anchor)) {
        Some(at) => region.insert(
            match place {
                Place::Before => at,
                Place::After => at + 1,
            },
            new_add,
        ),
        // ⭐ UNREACHABLE, AND APPENDED RATHER THAN DROPPED: every anchor here is an op this walk has
        // just found in this region, or one [`unroll_burst_and_il`] has just created in it.
        None => region.push(new_add),
    }
    if let Some(mem_op) = last_mem_op {
        replace_uses_of_with(region, mem_op, inp, result);
    }
}

/// `last_mem_op->replaceUsesOfWith(inp, new_add)` (`:1370`) — ONE op's operand list, ⛔ NOT the value's
/// users: every other reader of `of` keeps reading it.
fn replace_uses_of_with(region: &mut [Op], in_op: Val, of: Val, with: Val) {
    let Some(op) = region.iter_mut().find(|op| results(op).contains(&in_op)) else {
        return;
    };
    for (at, operand) in operands(op).into_iter().enumerate() {
        if operand == of {
            set_operand(op, at, with);
        }
    }
}

/// The three address operands of one transfer, written by NAME where the reference writes them by
/// `getOperandNumber()` (`:1400-1418`) — the same three slots, and no index to get wrong.
fn set_transfer_addresses(
    region: &mut [Op],
    of: Val,
    mutable_addr: Option<Val>,
    immutable_addr: Option<Val>,
    increment: Option<Val>,
) {
    let Some(Op::Sentient(
        ops::Op::LoadAndSend {
            mutable_addr: into_mutable,
            immutable_addr: into_immutable,
            increment: into_increment,
            ..
        }
        | ops::Op::ReceiveAndStore {
            mutable_addr: into_mutable,
            immutable_addr: into_immutable,
            increment: into_increment,
            ..
        }
        | ops::Op::LoadComputeAndSend {
            mutable_addr: into_mutable,
            immutable_addr: into_immutable,
            increment: into_increment,
            ..
        },
    )) = region.iter_mut().find(|op| results(op).contains(&of))
    else {
        return;
    };
    if let Some(val) = mutable_addr {
        *into_mutable = val;
    }
    if let Some(val) = immutable_addr {
        *into_immutable = val;
    }
    if let Some(val) = increment {
        *into_increment = val;
    }
}

/// WHICH ARM OF `doScalarOpMerging`'S `isa<>` CHAIN AN OP TAKES (`:1387`, `:1397`); the third arm is the
/// `llvm_unreachable("Unsupported op found in block!")` (`:1464`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergedOp {
    /// `sentient.add` / `sentient.sub`, which merging ERASES.
    AddOrSub,
    /// A transfer, with the burst and IL the arm reads RAW — ⛔ NOT [`MemoryOpInfo`]'s defaulted pair:
    /// an LCAS is given `burst = 0, il = 0` outright (`:1414-1418`), and a `burst_size` of 0 must stay 0
    /// or an unbursted transfer would take the field-unrolling path.
    Transfer(BurstAndIl),
}

impl MergedOp {
    fn of(op: &Op) -> Option<MergedOp> {
        match op {
            Op::Sentient(ops::Op::ScalarAdd { .. } | ops::Op::ScalarSub { .. }) => {
                Some(MergedOp::AddOrSub)
            }
            Op::Sentient(ops::Op::LoadComputeAndSend { .. }) => {
                Some(MergedOp::Transfer(BurstAndIl {
                    burst: Elements(0),
                    interleaved_group: Elements(0),
                }))
            }
            _ => BurstAndIl::of(op).map(MergedOp::Transfer),
        }
    }
}

/// Replaces: e364_doScalarOpMerging
///
/// APPLIES the analysis to one block: every add/sub is erased, every plain transfer takes its new
/// immutable and a zero increment, marked candidates are field unrolled, and whatever increment is left
/// becomes one `sentient.add` at the top of the chain (`:1334-1465`).
///
/// ⛔ MERGING STOPS AT THE FIRST BURSTED TRANSFER THAT WAS NOT MARKED FOR UNROLLING (`:1421-1431`): the
/// remaining increment is added BELOW it, every other reader of its result goes through that add, and
/// the rest of the block is left exactly as it was — which is the early `return`.
/// ⭐ `first_op` (`:1341`) AND `unroll_candidates` (`:1347`) ARE DEAD LOCALS in the reference.
pub(crate) fn do_scalar_op_merging(
    block: &ScalarOpMergingBlock,
    region: &mut Vec<Op>,
    parent_op: Option<&Op>,
    evaluator: &mut dyn ExpressionEvaluator,
    sites: &mut OffsetSites<'_>,
) {
    // A block with one scalar op is worth merging only if that makes it a hoisting candidate
    // (`:1339-1340`).
    if block.num_scalar_ops_in_block <= ScalarOpCount(1)
        && !is_profitable_for_hoisting(block, parent_op)
    {
        return;
    }
    // `block_data.back()` and `input_value_to_block_` are both read unconditionally (`:1351`, `:1361`);
    // an empty block and a block that never called `setInputValueToBlock` are the two reads that would
    // not survive, and `buildBlock` reaches neither.
    let (Some(anchor), Some(input_value_to_block)) =
        (block.block_ops.last().copied(), block.input_value_to_block)
    else {
        return;
    };

    let mut merging_increment = evaluator.constant(0);
    // `zero_const` — ONE for the whole block, in the const builder's block (`:1345-1346`).
    let zero = sites.values.mint();
    sites.consts.push(Op::Sentient(ops::Op::ScalarConstant {
        value: 0,
        result: zero,
        reg_locale: ops::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    }));
    let anchor_element_size = {
        let scope: [&[Op]; 1] = [region.as_slice()];
        element_size(anchor.op, Definitions::from_innermost(&scope))
    };

    for (at, data) in block.block_ops.iter().enumerate() {
        let op_val = data.op;
        let next = block.block_ops.get(at + 1).map(|next| next.op);
        let is_last_op_in_block = next.is_none();
        let Some(merged) = defining_op(op_val, region).and_then(MergedOp::of) else {
            panic!("unsupported op found in block (ScalarOpMergingAndHoisting.cpp:1464)")
        };

        match merged {
            MergedOp::AddOrSub => {
                merging_increment = data.merging_increment;
                match next {
                    // The chain's next op takes over this one's readers, and this op goes away.
                    Some(next_op) => replace_all_uses_with(region, op_val, next_op),
                    None => {
                        let new_add = sites.values.mint();
                        create_remaining_add(
                            input_value_to_block,
                            merging_increment,
                            anchor_element_size,
                            op_val,
                            Place::Before,
                            None,
                            new_add,
                            region,
                            evaluator,
                            sites,
                        );
                        replace_all_uses_with(region, op_val, new_add);
                    }
                }
                erase_defining_op(region, op_val);
            }
            MergedOp::Transfer(transfer) => {
                if transfer.burst.0 > 1 || transfer.interleaved_group.0 > 0 {
                    let Some(candidate) = find_field_unroll_candidate(op_val, block) else {
                        // `replaceAllUsesExcept(new_add, {new_add})` (`:1428-1430`), as a replacement
                        // done BEFORE the add exists — the add's own read of `op_val` is the exception.
                        let new_add = sites.values.mint();
                        replace_all_uses_with(region, op_val, new_add);
                        create_remaining_add(
                            op_val,
                            merging_increment,
                            anchor_element_size,
                            op_val,
                            Place::After,
                            None,
                            new_add,
                            region,
                            evaluator,
                            sites,
                        );
                        return;
                    };
                    merging_increment = data.merging_increment;
                    let Some(target) = UnrollTarget::of(&candidate, region) else {
                        // ⭐ UNREACHABLE: the marked candidate carries the immutables and the op is the
                        // transfer this walk has just matched — see [`UnrollTarget`]'s three checks.
                        return;
                    };
                    let unrolled = unroll_burst_and_il(target, region, evaluator, sites);
                    if is_last_op_in_block {
                        if let Some(&first) = unrolled.first() {
                            let new_add = sites.values.mint();
                            create_remaining_add(
                                input_value_to_block,
                                merging_increment,
                                anchor_element_size,
                                first,
                                Place::Before,
                                Some(first),
                                new_add,
                                region,
                                evaluator,
                                sites,
                            );
                        }
                        return;
                    }
                } else {
                    merging_increment = data.merging_increment;
                    let immutable_addr = evaluator.build_offset_value_of(
                        data.mod_by,
                        sites,
                        region,
                        ScalarTy::Index,
                    );
                    match next {
                        Some(next_op) => {
                            set_transfer_addresses(region, op_val, Some(next_op), None, None);
                        }
                        None => {
                            let new_add = sites.values.mint();
                            create_remaining_add(
                                input_value_to_block,
                                merging_increment,
                                anchor_element_size,
                                op_val,
                                Place::Before,
                                Some(op_val),
                                new_add,
                                region,
                                evaluator,
                                sites,
                            );
                        }
                    }
                    set_transfer_addresses(region, op_val, None, Some(immutable_addr), Some(zero));
                }
            }
        }
    }
}

// crustify:todo: e528_buildBlock
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:644  (179 body lines, level 3)
//   original  : void ScalarOpMerging::buildBlock( Operation *merge_candidate, std::unordered_set<Operation *> &analyzed_ops)
//   calls     : e158_doesValueExceedLRFRange, e162_addOpToBlock, e163_setEnablesMerging, e164_addUnrollCandidate, e252_size, e361_isImmutableValueInRange, e362_isFieldUnrollCandidate, e422_insert

// crustify:todo: e577_collectBlocks
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:869  (26 body lines, level 4)
//   original  : void ScalarOpMerging::collectBlocks()
//   calls     : e422_insert, e528_buildBlock

// crustify:todo: e611_runScalarOpMerging
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:569  (10 body lines, level 5)
//   original  : void ScalarOpMerging::runScalarOpMerging()
//   calls     : e363_markFieldUnrollingCandidates, e364_doScalarOpMerging, e577_collectBlocks

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::transform::sentient::analyses::{Evaluation, Offsets, ScalarOffset};

    /// The out-of-scope evaluator, stating the answers these units consume: a handle arena holding one
    /// `AllUnit` offset per entry, which is all any range test here reads back.
    #[derive(Default)]
    struct StatedEvaluator {
        /// What `buildOffsetValue` materialises one handle into.
        offsets: Vec<(EvaluatedValue, Val)>,
        /// The arena itself — a handle is an index into it.
        arena: Vec<i64>,
        /// What `evaluateValue` answers for a value; anything unstated evaluates to 0.
        stated: Vec<(Val, i64)>,
    }

    impl StatedEvaluator {
        /// A fresh arena entry holding `offset`.
        fn intern(&mut self, offset: i64) -> EvaluatedValue {
            self.arena.push(offset);
            EvaluatedValue(u32::try_from(self.arena.len() - 1).expect("a small arena"))
        }

        /// The offset one handle names.
        fn offset(&self, ev: EvaluatedValue) -> i64 {
            self.arena[ev.0 as usize]
        }
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

        fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
            let offset = self
                .stated
                .iter()
                .find(|(val, _)| *val == value)
                .map_or(0, |(_, offset)| *offset);
            self.intern(offset)
        }

        fn constant(&mut self, value: i64) -> EvaluatedValue {
            self.intern(value)
        }

        fn evaluate_sum_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let sum = self.offset(lhs) + self.offset(rhs);
            self.intern(sum)
        }

        fn evaluate_sub_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let difference = self.offset(lhs) - self.offset(rhs);
            self.intern(difference)
        }

        fn evaluate_multiply_by_const(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
            let product = self.offset(ev) * by;
            self.intern(product)
        }

        fn evaluation_of(&mut self, ev: EvaluatedValue) -> Evaluation {
            Evaluation {
                known_absolute: true,
                base: None,
                offsets: Offsets::AllUnit(ScalarOffset(self.offset(ev))),
            }
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
            ..StatedEvaluator::default()
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

    /// e362 — the vendor's own no-IL case (`:1177-1198`): a burst of 2 speculates two immutables, each
    /// the previous one less the increment, and the merging increment ends up at the total.
    #[test]
    fn a_burst_two_transfer_speculates_one_immutable_per_burst_iteration() {
        let region = vec![load_and_send(Val(1), Val(2), Val(3), Val(5), 2, 0)];
        let mut evaluator = StatedEvaluator {
            stated: vec![(Val(2), 1024), (Val(3), 8)],
            ..StatedEvaluator::default()
        };
        let mut candidate = candidate_of(Val(5), 1, Vec::new());
        let mut merging_increment = evaluator.constant(0);
        // 1016 and 1008 scale to 4064 and 4032 at 32 bits, both inside this window.
        let wide = ImmRange {
            min: -4096,
            max: 4096,
        };
        assert!(is_field_unroll_candidate(
            &mut candidate,
            &mut merging_increment,
            &region[0],
            &region,
            ScalarOpComp::Lxlu,
            wide,
            AddressScale::ONE,
            &mut evaluator,
        ));
        assert_eq!(
            candidate
                .speculative_immutables
                .iter()
                .map(|ev| evaluator.offset(*ev))
                .collect::<Vec<_>>(),
            vec![1016, 1008]
        );
        assert_eq!(evaluator.offset(merging_increment), 16);

        // ⛔ THE NEGATIVE THE WHOLE FUNCTION EXISTS FOR: the same op against a window the first
        // speculated immutable already leaves — no candidate, and the increment is left alone.
        let mut refused = candidate_of(Val(5), 1, Vec::new());
        let mut untouched = evaluator.constant(0);
        assert!(!is_field_unroll_candidate(
            &mut refused,
            &mut untouched,
            &region[0],
            &region,
            ScalarOpComp::Lxlu,
            ImmRange {
                min: -100,
                max: 100
            },
            AddressScale::ONE,
            &mut evaluator,
        ));
        assert!(refused.speculative_immutables.is_empty());
        assert_eq!(evaluator.offset(untouched), 0);
    }

    /// e363 — the whole of the best block is marked and charged, and the block that no longer fits gets
    /// only the prefix ending at its merging enabler.
    #[test]
    fn the_best_block_is_marked_whole_and_the_next_only_up_to_its_enabler() {
        let mut enabler = candidate_of(Val(11), 1, Vec::new());
        enabler.enables_merging = true;
        let mut second_enabler = candidate_of(Val(20), 1, Vec::new());
        second_enabler.enables_merging = true;
        let mut blocks = vec![
            // 2 ops : 4 slots — the worse ratio, and sorted second.
            ScalarOpMergingBlock {
                block_ops: vec![bottom_op(Val(20))],
                unroll_candidates: vec![second_enabler, candidate_of(Val(21), 3, Vec::new())],
                num_scalar_ops_in_block: ScalarOpCount(2),
                required_ibuff: InstructionCount(4),
                ..ScalarOpMergingBlock::default()
            },
            // 4 ops : 3 slots.
            ScalarOpMergingBlock {
                block_ops: vec![bottom_op(Val(10))],
                unroll_candidates: vec![candidate_of(Val(10), 2, Vec::new()), enabler],
                num_scalar_ops_in_block: ScalarOpCount(4),
                required_ibuff: InstructionCount(3),
                ..ScalarOpMergingBlock::default()
            },
        ];
        let mut ibuff_space = InstructionCount(4);
        let mut evaluator = StatedEvaluator::default();
        mark_field_unrolling_candidates(
            &mut blocks,
            &mut ibuff_space,
            &[],
            None,
            ImmRange {
                min: -4096,
                max: 4096,
            },
            AddressScale::ONE,
            &mut evaluator,
        );
        let marked: Vec<Vec<bool>> = blocks
            .iter()
            .map(|block| {
                block
                    .unroll_candidates
                    .iter()
                    .map(|candidate| candidate.marked_for_unrolling)
                    .collect()
            })
            .collect();
        // The 4 : 3 block sorts first and takes three of the four slots; the other then affords only
        // its first candidate, which is the enabler the prefix is banked at.
        assert_eq!(marked, vec![vec![true, true], vec![true, false]]);
        assert_eq!(ibuff_space, InstructionCount(0));
    }

    /// e364 — a two-add chain feeding one plain transfer: both adds go away, the transfer takes the new
    /// immutable and a zero increment, and one add carrying the block's remaining increment is left at
    /// the top of the chain.
    #[test]
    fn merging_a_block_erases_its_adds_and_leaves_one_at_the_top() {
        let mut region = vec![
            add(Val(1), Val(9), Val(5)),
            add(Val(5), Val(10), Val(6)),
            load_and_send(Val(6), Val(2), Val(3), Val(7), 0, 0),
        ];
        // Bottom-up, as `buildBlock` collects them; the anchor is the top add and carries the width.
        let top_add = Op::Sentient(ops::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(9),
            result: Val(5),
            reg: None,
            ty: ScalarTy::Index,
            element_size: Some(Bits(32)),
        });
        region[0] = top_add;
        let block = ScalarOpMergingBlock {
            input_value_to_block: Some(Val(1)),
            block_ops: vec![
                OperationData {
                    op: Val(7),
                    mod_by: EvaluatedValue(1),
                    merging_increment: EvaluatedValue(2),
                    replace_with_mod: false,
                },
                bottom_op(Val(6)),
                OperationData {
                    op: Val(5),
                    mod_by: EvaluatedValue(0),
                    merging_increment: EvaluatedValue(3),
                    replace_with_mod: false,
                },
            ],
            num_scalar_ops_in_block: ScalarOpCount(2),
            ..ScalarOpMergingBlock::default()
        };
        let mut evaluator = StatedEvaluator {
            // The transfer's new immutable, and the increment the remaining add is built from.
            offsets: vec![(EvaluatedValue(1), Val(50)), (EvaluatedValue(3), Val(51))],
            arena: vec![0, 0, 0, 0],
            ..StatedEvaluator::default()
        };
        let mut consts: Vec<Op> = Vec::new();
        let mut values = values_after(60);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        do_scalar_op_merging(&block, &mut region, None, &mut evaluator, &mut sites);

        let remaining_add = Op::Sentient(ops::Op::ScalarAdd {
            lhs: Val(1),
            rhs: Val(51),
            result: Val(61),
            reg: None,
            ty: ScalarTy::Index,
            element_size: Some(Bits(32)),
        });
        assert_eq!(
            region,
            vec![
                remaining_add,
                // `mutable_addr` follows the chain up to the new add; the immutable is the analysis's
                // and the increment is the one zero the block created.
                load_and_send(Val(61), Val(50), Val(60), Val(7), 0, 0),
            ]
        );
        assert_eq!(
            consts,
            vec![Op::Sentient(ops::Op::ScalarConstant {
                value: 0,
                result: Val(60),
                reg_locale: ops::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            })]
        );
    }
}
