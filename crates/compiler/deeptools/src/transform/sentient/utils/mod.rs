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

//! `Utils.cpp` — 19 of the campaign's 656 units (dependency level(s) [0, 1, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e239_getOperationOfBlock` | 239 | 0 | 8 | `dcc/src/Transform/Sentient/Utils.cpp:31` |
//! | `e240_selectIndicesForUnits` | 240 | 0 | 15 | `dcc/src/Transform/Sentient/Utils.cpp:66` |
//! | `e241_moveToCommonDominator` | 241 | 0 | 71 | `dcc/src/Transform/Sentient/Utils.cpp:84` |
//! | `e242_getForLoopInfoIfIV` | 242 | 0 | 37 | `dcc/src/Transform/Sentient/Utils.cpp:156` |
//! | `e243_roundDownUnrollFactor` | 243 | 0 | 32 | `dcc/src/Transform/Sentient/Utils.cpp:397` |
//! | `e244_negatePredicate` | 244 | 0 | 18 | `dcc/src/Transform/Sentient/Utils.cpp:431` |
//! | `e245_reversePredicate` | 245 | 0 | 18 | `dcc/src/Transform/Sentient/Utils.cpp:450` |
//! | `e246_getOutermostConstInitialization` | 246 | 0 | 55 | `dcc/src/Transform/Sentient/Utils.cpp:469` |
//! | `e247_memoryOpRequiresImmutAddrScalarCopy` | 247 | 0 | 64 | `dcc/src/Transform/Sentient/Utils.cpp:526` |
//! | `e248_getFoldModeAttributeIfExists` | 248 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.cpp:593` |
//! | `e250_hasUniformizeRegion` | 250 | 0 | 12 | `dcc/src/Transform/Sentient/Utils.cpp:622` |
//! | `e251_add` | 251 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:165` |
//! | `e252_size` | 252 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:171` |
//! | `e253_areAllValuesEqual` | 253 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.hpp:180` |
//! | `e392_getQueryKeyAndUnitsFromParentRegion` | 392 | 1 | 24 | `dcc/src/Transform/Sentient/Utils.cpp:40` |
//! | `e393_createSentientForOpWithAdditionalIterArgs` | 393 | 1 | 59 | `dcc/src/Transform/Sentient/Utils.cpp:195` |
//! | `e395_pruneOutOfScopeEntries` | 395 | 1 | 55 | `dcc/src/Transform/Sentient/Utils.cpp:635` |
//! | `e396_replaceValue` | 396 | 1 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:175` |
//! | `e546_findAndReplaceRedundantIterArgsUsedInConditions` | 546 | 3 | 138 | `dcc/src/Transform/Sentient/Utils.cpp:257` |

#![allow(dead_code)]
// ⛔ NOTHING CALLS THIS FILE YET — `Utils.cpp` is the campaign's shared leaf library, and its
// consumers (`e392`, `e393`, `e395`, `e546` here, plus the passes) are not in this batch. CI runs
// clippy with `-D warnings`, so without this the first ported leaf fails the gate.
// ⭐ REMOVE THIS WITH THE FIRST CONSUMER: at that point an unused item here is a real defect again.

use core::num::{NonZeroU32, NonZeroU64};

use super::analyses::{UnitIndex, UnitIndexMap};
use super::register_packing::constant_target_values;
use super::scalar_op_merging_and_hoisting::{ScalarOpComp, compute_address_scale};
use super::{ForRef, IterArgIndex};
use crate::arch::{Arch, Elements};
use crate::formats::Bits;
use sys_arch_spec::fields::{self, ImmSpec, ImmWidth, Sign};
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, dataflow, operands, regions_mut, regions_ref, results, sentient, symbol,
    uniform, use_count,
};

pub(crate) mod units_and_their_values;

/// A POSITION IN ONE BLOCK — what `int op_num` indexes and what every `(op, region)` step of an
/// [`OpAt`] is expressed in.
///
/// ⛔ `usize`, SO A NEGATIVE `op_num` IS INEXPRESSIBLE: `while (op_num > 0)` (`:33`) treats every
/// negative as zero and hands back the FIRST op of the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBlock(pub usize);

/// Replaces: e239_getOperationOfBlock
///
/// The op at position `op_num` of `block` — `std::next(block.getOperations().begin(), op_num)`
/// (`:31-38`).
///
/// ⛔ TRAP: THE REFERENCE WALKS PAST THE END and dereferences the list sentinel. Every call site
/// bounds `op_num` by the block it indexes — `LiveRangeReduction.cpp:1180` scans that very block, and
/// `:1039`/`:1076` index a clone built one op for one (`:1024-1026`) — so `None` names that
/// unreachable case.
#[must_use]
pub fn operation_of_block(block: &[Op], op_num: InBlock) -> Option<&Op> {
    block.get(op_num.0)
}

/// Replaces: e240_selectIndicesForUnits
///
/// Flattens every `dataflow.create_group` in `units` to its members and appends each surviving
/// unit's index to `indices` (`:66-82`).
///
/// ⛔ TRAP: `units` COMES BACK REVERSED — the walk is a LIFO pop-back that pushes onto
/// `cleaned_units` in pop order, and a group's members go onto the same stack, so nested groups
/// expand transitively and in reverse.
/// ⛔ TRAP: `indices` IS APPENDED TO, NOT CLEARED — the caller's earlier entries survive.
pub fn select_indices_for_units(
    units: &mut Vec<Val>,
    indices: &mut Vec<UnitIndex>,
    map: &impl UnitIndexMap,
    defs: Definitions<'_>,
) {
    let mut cleaned = Vec::new();
    while let Some(unit) = units.pop() {
        if let Some(Op::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) = defs.of(unit) {
            units.extend(unit_ids.iter().copied());
            continue;
        }
        cleaned.push(unit);
        indices.push(map.index_of(unit));
    }
    *units = cleaned;
}

/// WHERE ONE OP SITS INSIDE A `dataflow.program_unit` BODY — the `(op, region)` steps that open each
/// enclosing block, outermost first, then its own position in the innermost one.
///
/// ⭐⭐ THIS IS THE WHOLE OF WHAT `DominanceInfo` (`:85`) ANSWERS HERE, AND THAT IS PROVABLE: every
/// region of this island holds exactly one block, so `dominates(a, b)` reduces to *a's block
/// encloses b's, and a comes no later than b's ancestor in it* — see [`OpAt::dominates`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpAt {
    /// One `(position of the enclosing op, which of its regions)` step per level, outermost first.
    enclosing: Vec<(InBlock, usize)>,
    /// The position in the block those steps open.
    index: InBlock,
}

impl OpAt {
    /// An op at `index` of the block `enclosing` opens.
    #[must_use]
    pub fn at(enclosing: &[(InBlock, usize)], index: InBlock) -> OpAt {
        OpAt {
            enclosing: enclosing.to_vec(),
            index,
        }
    }

    /// An op at `index` of the program-unit body itself.
    #[must_use]
    pub fn top(index: InBlock) -> OpAt {
        OpAt {
            enclosing: Vec::new(),
            index,
        }
    }

    /// `Operation::getParentOp()`, and `None` when that parent is the `dataflow.program_unit` — which
    /// is the `dyn_cast<dataflow::ProgramUnitOp>` arm at `:91`.
    #[must_use]
    pub fn parent(&self) -> Option<OpAt> {
        let (index, _) = *self.enclosing.last()?;
        Some(OpAt {
            enclosing: self.enclosing[..self.enclosing.len() - 1].to_vec(),
            index,
        })
    }

    /// The op this path names.
    #[must_use]
    pub fn op<'a>(&self, unit_body: &'a [Op]) -> Option<&'a Op> {
        block_of(unit_body, &self.enclosing)?.get(self.index.0)
    }

    /// `DominanceInfo::dominates(self, other)` for two ops of one program unit — reflexive, as MLIR's
    /// `dominates(Operation *, Operation *)` is.
    #[must_use]
    pub fn dominates(&self, other: &OpAt) -> bool {
        let depth = self.enclosing.len();
        depth <= other.enclosing.len()
            && self.enclosing[..] == other.enclosing[..depth]
            && if depth == other.enclosing.len() {
                self.index <= other.index
            } else {
                self.index <= other.enclosing[depth].0
            }
    }
}

/// WHERE THE NEW USE IS — `new_use`, which the reference only ever asks `DominanceInfo` about.
///
/// ⛔⛔ [`NewUse::OtherUnit`] IS THE ONLY WAY `failure()` HAPPENS, AND THAT IS PROVED: the nop the
/// `ProgramUnitOp` arm inserts sits at position 0 of the unit body and therefore dominates every op
/// in it, so `!dom_info.dominates(insert_point, new_use)` at `:98` can hold only when `new_use` lies
/// outside the unit — which is exactly the comment the reference puts on that line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NewUse {
    /// Inside the same `dataflow.program_unit`, at this path.
    SameUnit(OpAt),
    /// Outside it.
    OtherUnit,
}

/// `mlir::LogicalResult` FROM A HOIST — the one `failure()` at `:99` and `success()` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Hoisted {
    /// `success()` — `op` now dominates `new_use`.
    Done,
    /// `failure()` — the common dominator would be outside the program unit.
    OutsideProgramUnit,
}

/// Replaces: e241_moveToCommonDominator
///
/// Climbs out of the enclosing regions until the insert point dominates `new_use`, rehoists every
/// operand definition that would stop dominating it, then moves `op` there (`:84-154`).
///
/// ⭐ THE DUMMY `sentient.nop` IS DROPPED POSITIONING MECHANISM, erased again at `:152`; [`NewUse`]
/// records why an insert POSITION still reaches the one `failure()`.
/// ⭐ `op` MOVES FIRST HERE AND LAST THERE (`:151`) — it changes no verdict below, because every
/// operand of `op` is defined outside `op` and the removal is in a strictly nested block.
pub fn move_to_common_dominator(unit_body: &mut Vec<Op>, op: &OpAt, new_use: &NewUse) -> Hoisted {
    let mut point = op.clone();
    loop {
        if dominates_use(&point, new_use) {
            break;
        }
        match point.parent() {
            Some(parent) => {
                // ⚠️ ISLAND GAP, NOT A CHOICE: this arm (`:101-125`) cannot fire, because
                // `uniform::LocalRegion::body` holds ops of the rung BELOW and no op of this island
                // is ever nested in a `uniform.uniformize_regions`. The sentient-rung op that would
                // change that is the extension `e444_analyze`/`e505_transform` already need — see
                // `transform/sentient/local_region_splitting_for_value_commoning/local_region.rs`.
                if let Some(Op::Uniform(uniform::Op::UniformizeRegions { .. })) =
                    parent.op(unit_body)
                    && !promotes_above_uniform_region(op, unit_body)
                {
                    todo!(
                        "dcc::uniform::utils::getRegionOpAndIndex (dcc/src/Dialect/Uniform/Utils.cpp:343) — \
                         a sentient-rung uniform.uniformize_regions is the island extension this arm \
                         needs (Transform/Sentient/Utils.cpp:114-124)"
                    )
                }
                point = parent;
            }
            None => {
                // The nop the `ProgramUnitOp` arm builds goes at the front of the unit body, and
                // `break` because the climb cannot go farther than the unit (`:91-100`).
                point = OpAt::top(InBlock(0));
                if !dominates_use(&point, new_use) {
                    return Hoisted::OutsideProgramUnit;
                }
                break;
            }
        }
    }
    // `if (insert_point == op) return success();` (`:127`)
    if point == *op {
        return Hoisted::Done;
    }
    let Some(moved) = remove_at(unit_body, op) else {
        return Hoisted::Done;
    };
    let mut worklist = defined_operands(unit_body, &moved);
    let mut hoisted = 0;
    while let Some(curr) = worklist.pop() {
        let Some(curr_at) = path_of(unit_body, curr) else {
            continue;
        };
        if curr_at.dominates(&point) {
            continue;
        }
        let Some(def) = remove_at(unit_body, &curr_at) else {
            continue;
        };
        worklist.extend(defined_operands(unit_body, &def));
        insert_at(unit_body, &point, def);
        hoisted += 1;
    }
    // The moves left `[m_k, .., m_1, insert_point]`, so `final_insert_point` has drifted by `hoisted`
    // and `op->moveBefore(final_insert_point)` (`:151`) lands immediately after the last of them.
    insert_at(
        unit_body,
        &OpAt::at(&point.enclosing, InBlock(point.index.0 + hoisted)),
        moved,
    );
    Hoisted::Done
}

/// `dom_info.dominates(insert_point, new_use)` — see [`NewUse`] for why the second arm is `false`.
fn dominates_use(point: &OpAt, new_use: &NewUse) -> bool {
    match new_use {
        NewUse::SameUnit(at) => point.dominates(at),
        NewUse::OtherUnit => false,
    }
}

/// `llvm::copy_if(op->getOperands(), .., [](Value v) { return v.getDefiningOp(); })` (`:136-137`),
/// which is also the `DT_CHECK(!isa<BlockArgument>(curr))` at `:141`.
fn defined_operands(unit_body: &[Op], op: &Op) -> Vec<Val> {
    operands(op)
        .into_iter()
        .filter(|val| path_of(unit_body, *val).is_some())
        .collect()
}

/// `promote_above_uniform_region` (`:104-112`) — `op` is a `sentient.scalar_copy` of a
/// `dataflow.create_multicast_group`, a `sentient.scalar_constant` or a `symbol.create_symbol`.
///
/// ⚠️ DIVERGENCE: A COPY OF A BLOCK ARGUMENT ANSWERS `false` HERE AND ASSERTS THERE —
/// `isa<CreateMulticastGroupOp>(copy_inp_op)` (`:107`) is a bare `isa<>` on a possibly-null defining
/// op, which is an assertion failure and not an answer of false.
fn promotes_above_uniform_region(op: &OpAt, unit_body: &[Op]) -> bool {
    let Some(Op::Sentient(sentient::Op::ScalarCopy { input, .. })) = op.op(unit_body) else {
        return false;
    };
    let scopes = visible_from(unit_body, op);
    let defs = Definitions::from_innermost(&scopes);
    matches!(
        defs.of(*input),
        Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }))
    ) || is_constant(*input, ConstKind::ScalarConstant, defs)
        || is_constant(*input, ConstKind::CreateSymbol, defs)
}

/// The blocks an op at `at` can see definitions in, innermost first — MLIR's outward visibility.
fn visible_from<'a>(unit_body: &'a [Op], at: &OpAt) -> Vec<&'a [Op]> {
    (0..=at.enclosing.len())
        .rev()
        .filter_map(|depth| block_of(unit_body, &at.enclosing[..depth]))
        .collect()
}

/// The block `enclosing` opens, borrowed.
fn block_of<'a>(unit_body: &'a [Op], enclosing: &[(InBlock, usize)]) -> Option<&'a [Op]> {
    match enclosing.split_first() {
        None => Some(unit_body),
        Some((&(InBlock(index), region), rest)) => {
            let inner = *regions_ref(unit_body.get(index)?).get(region)?;
            block_of(inner, rest)
        }
    }
}

/// The block `enclosing` opens, mutably.
fn block_of_mut<'a>(
    unit_body: &'a mut Vec<Op>,
    enclosing: &[(InBlock, usize)],
) -> Option<&'a mut Vec<Op>> {
    match enclosing.split_first() {
        None => Some(unit_body),
        Some((&(InBlock(index), region), rest)) => {
            let inner = regions_mut(unit_body.get_mut(index)?)
                .into_iter()
                .nth(region)?;
            block_of_mut(inner, rest)
        }
    }
}

/// WHERE THE OP BINDING `val` SITS — `Value::getDefiningOp()` plus the position the moves index by.
///
/// ⭐ THE WHOLE UNIT IS SEARCHED RATHER THAN THE VISIBLE SCOPES, because a [`Val`] is bound once
/// ([`crate::islands::dataflow_ir::Values`]) and the answer must stay right as ops move between
/// blocks.
fn path_of(unit_body: &[Op], val: Val) -> Option<OpAt> {
    fn walk(block: &[Op], val: Val, enclosing: &mut Vec<(InBlock, usize)>) -> Option<OpAt> {
        for (index, op) in block.iter().enumerate() {
            if results(op).contains(&val) {
                return Some(OpAt::at(enclosing, InBlock(index)));
            }
            for (region, inner) in regions_ref(op).into_iter().enumerate() {
                enclosing.push((InBlock(index), region));
                let found = walk(inner, val, enclosing);
                enclosing.pop();
                if found.is_some() {
                    return found;
                }
            }
        }
        None
    }
    walk(unit_body, val, &mut Vec::new())
}

/// `*val.user_begin()` — THE OP THAT READS `val` ITSELF, at whatever depth, and never the enclosing
/// op whose region merely holds that read. Only meaningful where `val` has exactly one use.
fn immediate_user<'a>(val: Val, scope: &'a [Op]) -> Option<&'a Op> {
    scope.iter().find_map(|op| {
        if operands(op).contains(&val) {
            return Some(op);
        }
        regions_ref(op)
            .into_iter()
            .find_map(|region| immediate_user(val, region))
    })
}

/// `Operation::remove()` — the first half of a `moveBefore`.
fn remove_at(unit_body: &mut Vec<Op>, at: &OpAt) -> Option<Op> {
    let block = block_of_mut(unit_body, &at.enclosing)?;
    (at.index.0 < block.len()).then(|| block.remove(at.index.0))
}

/// `Operation::moveBefore(at)` — the second half, `at` naming the op to land in front of.
fn insert_at(unit_body: &mut Vec<Op>, at: &OpAt, op: Op) {
    if let Some(block) = block_of_mut(unit_body, &at.enclosing) {
        let index = at.index.0.min(block.len());
        block.insert(index, op);
    }
}

/// WHICH OP `isConstant<ConstTy>` IS INSTANTIATED FOR (`dcc/src/Utils/Utils.cpp:445-447`); the
/// `arith::ConstantOp` instantiation has no reader in this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstKind {
    /// `mlir::sentient::ConstantOp`.
    ScalarConstant,
    /// `mlir::symbol::CreateSymbolOp`.
    CreateSymbol,
}

impl ConstKind {
    /// `isa<ConstTy>(op)`.
    pub(crate) fn matches(self, op: &Op) -> bool {
        match self {
            ConstKind::ScalarConstant => {
                matches!(op, Op::Sentient(sentient::Op::ScalarConstant { .. }))
            }
            ConstKind::CreateSymbol => matches!(op, Op::Symbol(symbol::Op::CreateSymbol { .. })),
        }
    }
}

/// `isConstant<ConstTy>` (`dcc/src/Utils/Utils.cpp:424-442`) — the op itself, or a
/// `uniform.query_map` every value of whose immutable mapping is one.
///
/// ⚠️ DIVERGENCE: AN EMPTY IMMUTABLE MAPPING ANSWERS `true` HERE AND THROWS THERE —
/// `DT_CHECK(!immutable_map.getValues().empty())` (`Utils/Utils.cpp:434`), and `.all()` over no pairs
/// is vacuously true.
pub(crate) fn is_constant(val: Val, kind: ConstKind, defs: Definitions<'_>) -> bool {
    let Some(def) = defs.of(val) else {
        return false;
    };
    if kind.matches(def) {
        return true;
    }
    match def {
        Op::Uniform(uniform::Op::QueryMap { map, .. }) => match defs.of(*map) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs
                .iter()
                .all(|(_, value)| defs.of(*value).is_some_and(|def| kind.matches(def))),
            _ => false,
        },
        _ => false,
    }
}

/// WHETHER A NORMALIZED INDUCTION VARIABLE IS ACCEPTED — `allow_normalized_iv`, whose declaration
/// defaults it to `true` (`Utils.hpp:73`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizedIv {
    /// The default: `const - iv` reads as the same loop, ascending.
    Accepted,
    /// `MultiDimLoopPeeling.cpp:153` passes `false` — only a bare induction variable counts.
    Rejected,
}

/// WHAT A `sentient.for` INDUCTION VARIABLE RESOLVES TO — the reference's five-tuple, whose fields
/// its own declaration names *"the loop, lower and upper bounds of this (normalized) IV, the step and
/// the number of iterations"* (`Utils.hpp:69-70`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForLoopInfo {
    /// The loop, named by its induction variable.
    pub loop_op: ForRef,
    /// `std::get<1>` — the lower bound.
    pub lower_bound: i64,
    /// `std::get<2>` — the upper bound.
    pub upper_bound: i64,
    /// `std::get<3>` — the step.
    pub step: i64,
    /// `std::get<4>` — the iteration count, always the loop's own constant bound.
    pub iterations: i64,
}

/// Replaces: e242_getForLoopInfoIfIV
///
/// The `sentient.for` and bounds `val` names as an induction variable, bare or normalized as
/// `const - iv`, and `None` for the reference's `{nullptr, 0, 0, 0, 0}` (`:156-193`).
///
/// ⛔ TRAP: A BARE `sentient.for` IV COUNTS DOWN — the tuple is `(bound, 1, -1, bound)`, so
/// `lower_bound` is the HIGH end; only the normalized form is ascending.
/// ⛔ TRAP: `MultiDimLoopPeeling.cpp:148` DOCUMENTS THE FAILURE TUPLE AS `{nullptr,-1,-1,-1,-1}`
/// and it is all zeros — nothing reads it, which is why `None` loses nothing.
#[must_use]
pub fn for_loop_info_if_iv(
    val: Val,
    allow_normalized_iv: NormalizedIv,
    defs: Definitions<'_>,
) -> Option<ForLoopInfo> {
    let (iv_candidate, subtracted_from, val_is_sub_op) = match defs.of(val) {
        // `isa<BlockArgument>(val)` — nothing in scope binds it (`:161`).
        None => (val, 0, false),
        Some(_) if allow_normalized_iv == NormalizedIv::Rejected => return None,
        Some(Op::Sentient(sentient::Op::ScalarSub { lhs, rhs, .. })) => {
            let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) = defs.of(*lhs)
            else {
                return None;
            };
            (*rhs, *value, true)
        }
        Some(_) => return None,
    };
    let (for_op, arg_number) = defs.for_arg_of(iv_candidate)?;
    // `sentient_for_op.getInductionVar() == iv_candidate` — argument 0 is the induction variable.
    if arg_number != 0 {
        return None;
    }
    // `Sentient For-op can have non-constant bounds.` (`:181`)
    let Op::Sentient(sentient::Op::For { bound, .. }) = for_op else {
        return None;
    };
    let Some(Op::Sentient(sentient::Op::ScalarConstant { value: bound, .. })) = defs.of(*bound)
    else {
        return None;
    };
    let loop_op = ForRef(iv_candidate);
    Some(if val_is_sub_op {
        ForLoopInfo {
            loop_op,
            lower_bound: subtracted_from - bound,
            upper_bound: subtracted_from,
            step: 1,
            iterations: *bound,
        }
    } else {
        ForLoopInfo {
            loop_op,
            lower_bound: *bound,
            upper_bound: 1,
            step: -1,
            iterations: *bound,
        }
    })
}

/// WHICH BACKEND A PROGRAM IS BUILT FOR — `enum class SenTargets` (`util/sendefs/sendefs.h:177`),
/// read from `dccExtContext().dsc_global_->backend` (`OpRerolling.cpp:1053`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenTarget {
    /// `UNDEFINED`.
    Undefined,
    /// `SENTIENT` — the card, and the only target a reduction is ever unrolled for.
    Sentient,
    /// `SENULATOR`.
    Senulator,
    /// `SENPCFG`.
    SenPcfg,
    /// `SENTF`.
    SenTf,
    /// `SYSTEMC`.
    SystemC,
    /// `R5SS`.
    R5ss,
    /// `HOST`.
    Host,
    /// `INVALID`.
    Invalid,
    /// `NOP`.
    Nop,
}

/// Replaces: e243_roundDownUnrollFactor
///
/// The largest legal unroll factor at or below `n` — 1, 2, 3 or 4 for a reduction on
/// [`SenTarget::Sentient`], and 1, 2, 4 or 8 otherwise (`:397-428`).
///
/// ⛔ TRAP: A REDUCTION ON ANY OTHER TARGET IS NEVER UNROLLED, whatever `n` is (`:416`).
/// ⭐ `DT_CHECK_MSG(n != 0, "Expect a positive target unroll value")` IS [`NonZeroU32`], and the
/// return set is [`sentient::UnrollFactor`] exactly — 3 reduction-only, 8 non-reduction-only.
#[must_use]
pub fn round_down_unroll_factor(
    n: NonZeroU32,
    compute_op: &Op,
    sen_target: SenTarget,
) -> sentient::UnrollFactor {
    let reduction = matches!(
        compute_op,
        Op::Sentient(sentient::Op::VectorUnary {
            unary_op: sentient::UnaryOp::ReductionAbsMax
                | sentient::UnaryOp::ReductionAbsMin
                | sentient::UnaryOp::ReductionAdd
                | sentient::UnaryOp::ReductionMax
                | sentient::UnaryOp::ReductionMin,
            ..
        })
    );
    match (reduction, sen_target, n.get()) {
        (true, SenTarget::Sentient, 1) => sentient::UnrollFactor::X1,
        (true, SenTarget::Sentient, 2) => sentient::UnrollFactor::X2,
        (true, SenTarget::Sentient, 3) => sentient::UnrollFactor::X3,
        (true, SenTarget::Sentient, _) => sentient::UnrollFactor::X4,
        (true, _, _) => sentient::UnrollFactor::X1,
        (false, _, 1) => sentient::UnrollFactor::X1,
        (false, _, 2 | 3) => sentient::UnrollFactor::X2,
        (false, _, 4..=7) => sentient::UnrollFactor::X4,
        (false, _, _) => sentient::UnrollFactor::X8,
    }
}

/// Replaces: e244_negatePredicate
///
/// The predicate that holds exactly when `pred` does not — [`sentient::CmpPredicate::negated`],
/// which is total over the six a `sentient.if` can carry, so the `llvm_unreachable` is unwritable.
#[must_use]
pub fn negate_predicate(pred: sentient::CmpPredicate) -> sentient::CmpPredicate {
    pred.negated()
}

/// Replaces: e245_reversePredicate
///
/// The predicate that holds with the operands swapped — [`sentient::CmpPredicate::reversed`], which
/// the island already carries because `RemoveStaticCondition` asks it of every condition.
#[must_use]
pub fn reverse_predicate(pred: sentient::CmpPredicate) -> sentient::CmpPredicate {
    pred.reversed()
}

/// HOW MANY ITERATIONS OF THE INNERMOST LOOP THE UNROLLED PROGRAM RUNS — the third element of
/// `getOutermostConstInitialization`'s tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainSize {
    /// The product of the chain's bounds; a `bound <= 0` is refused outright (`:496-498`).
    Iterations(NonZeroU64),
    /// `size = -1` — some loop in the chain has a non-constant bound (`Utils.hpp:139-140`).
    Unknown,
}

/// THE OUTERMOST CONSTANT INITIALIZATION OF AN ITER-ARG CHAIN — the reference's triple
/// (`Utils.hpp:134-144`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutermostConstInit {
    /// The outermost loop the walk reached, named by its induction variable.
    pub loop_op: ForRef,
    /// Which iter arg of it a constant initialises — `None` is the reference's `-1`.
    pub iter_arg: Option<IterArgIndex>,
    /// The unrolled iteration count.
    pub size: ChainSize,
}

/// Replaces: e246_getOutermostConstInitialization
///
/// Walks the chain of iter-arg initializations inner to outer and answers the outermost loop, which
/// of its iter args a constant initialises, and the unrolled iteration count (`:469-524`).
///
/// ⛔ TRAP: A VALID LOOP COMES BACK WITH `iter_arg: None` when the walk ends without a constant —
/// the reference returns `{outer_loop, -1, size}` there, and callers test the index
/// (`AddressPinningAndToggle.cpp:2677`).
/// ⛔ TRAP: ONCE `size` IS `-1` THE REFERENCE KEEPS MULTIPLYING IT, printing `-4` for a constant 4
/// outside a non-constant loop; the ONE caller that reads `size` at all tests `size <= 0`
/// (`AddressPinningAndToggle.cpp:2830`), which [`ChainSize`] answers; the other three discard it.
#[must_use]
pub fn outermost_const_initialization(
    iter_arg: Val,
    defs: Definitions<'_>,
) -> Option<OutermostConstInit> {
    let mut iter_arg_index: Option<IterArgIndex> = None;
    let mut curr: Option<Val> = Some(iter_arg);
    let mut outer_loop: Option<&Op> = None;
    let mut prev: Option<(&Op, usize)> = None;
    let mut size: u64 = 1;
    let mut unknown = false;

    // `while (iter_arg_index < 0 && curr_iter_arg)` (`:488`)
    while iter_arg_index.is_none() {
        let Some(curr_arg) = curr else { break };
        let (for_op, arg_number) = defs.for_arg_of(curr_arg)?;
        // `getRegionIterArgs()` is `getBody()->getArguments().drop_front(1)`, and `None` is the `-1`
        // `getIndexOfLoopRegionIterArgs` answers for an induction variable or a non-`for` block
        // argument (`Analyses/Utils.cpp:257-271`) — which the reference then indexes lists with.
        let curr_it_index = arg_number.checked_sub(1)?;
        outer_loop = Some(for_op);
        let Op::Sentient(sentient::Op::For {
            bound,
            carried,
            body,
            ..
        }) = for_op
        else {
            return None;
        };
        match defs.of(*bound) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => {
                if *value <= 0 {
                    return None;
                }
                size = size.saturating_mul(value.unsigned_abs());
            }
            _ => unknown = true,
        }
        if let Some((prev_loop, prev_it_index)) = prev {
            // `!curr_iter_arg.hasOneUse() || !isa<ForOp>(*curr_iter_arg.user_begin())` (`:509-510`).
            // Every use of a block argument is inside the region that binds it, so `body` holds all
            // of them and `hasOneUse()` is this count.
            if use_count(curr_arg, body) != 1 {
                return None;
            }
            if !matches!(
                immediate_user(curr_arg, body)?,
                Op::Sentient(sentient::Op::For { .. })
            ) {
                return None;
            }
            // `DT_CHECK_MSG(yield, "expected terminator of previous loop body to be a yield")`
            let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last() else {
                return None;
            };
            let Op::Sentient(sentient::Op::For {
                carried: prev_carried,
                ..
            }) = prev_loop
            else {
                return None;
            };
            if results.get(curr_it_index) != prev_carried.get(prev_it_index).map(|it| &it.result) {
                break;
            }
        }
        let curr_init = carried.get(curr_it_index)?.init;
        if defs.for_arg_of(curr_init).is_some() {
            curr = Some(curr_init);
            prev = Some((for_op, curr_it_index));
        } else {
            if is_constant(curr_init, ConstKind::ScalarConstant, defs) {
                iter_arg_index = u32::try_from(curr_it_index).ok().map(IterArgIndex);
            }
            curr = None;
        }
    }

    let Op::Sentient(sentient::Op::For { iv, .. }) = outer_loop? else {
        return None;
    };
    Some(OutermostConstInit {
        loop_op: ForRef(*iv),
        iter_arg: iter_arg_index,
        size: if unknown {
            ChainSize::Unknown
        } else {
            NonZeroU64::new(size).map_or(ChainSize::Unknown, ChainSize::Iterations)
        },
    })
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The types e247 is stated in — its two `llvm_unreachable`s and its one `DT_CHECK`.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// THE SIX MEMORY UNITS `memoryOpRequiresImmutAddrScalarCopy` DECIDES FOR — the units its own arms
/// name, so the trailing `llvm_unreachable("unexpected operation or unit type")` (`Utils.cpp:589`) is
/// only ever about a MISMATCHED op/unit pair and never about an unknown unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUnit {
    /// `L0LU` — the L0 load unit.
    L0lu,
    /// `L0SU` — the L0 store unit.
    L0su,
    /// `LXLU` — the LX load unit.
    Lxlu,
    /// `LXSU` — the LX store unit.
    Lxsu,
    /// `L3LU` — the L3 load half.
    L3lu,
    /// `L3SU` — the L3 store half.
    L3su,
}

impl MemoryUnit {
    /// `DT_CHECK(!is_any_of(unit, L3LU, L3SU))` (`Utils/DccExtContext.cpp:36`) AS A CONVERSION —
    /// `None` for the two L3 units, which have no IMM for these operations (`Utils.cpp:557-558`), and
    /// the four that remain are exactly [`ScalarOpComp`]'s.
    #[must_use]
    pub(crate) const fn with_imm(self) -> Option<ScalarOpComp> {
        match self {
            MemoryUnit::L0lu => Some(ScalarOpComp::L0lu),
            MemoryUnit::L0su => Some(ScalarOpComp::L0su),
            MemoryUnit::Lxlu => Some(ScalarOpComp::Lxlu),
            MemoryUnit::Lxsu => Some(ScalarOpComp::Lxsu),
            MemoryUnit::L3lu | MemoryUnit::L3su => None,
        }
    }
}

/// `do_range_check` (`Utils.hpp:154`, defaulted `true`) — whether the constant is also checked against
/// the unit's immediate width, or only tested for being constant at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeCheck {
    /// The default: check the immediate fits.
    Check,
    /// Skip the width check — the caller is asking only whether the address is constant.
    Skip,
}

/// WHICH OF THE FOUR OPS THIS IS, AND THE ATTRIBUTES THAT OP CONTRIBUTES — the `dyn_cast` chain
/// (`Utils.cpp:532-554`) with its `llvm_unreachable("unexpected op type")` as a type.
///
/// ⛔ EACH ARM CARRIES ONLY WHAT ITS ARM READS. The reference leaves `burst_size`/`il` at their
/// initialised 0 for the two scalar/compute ops and reads `getChunkStride()` on the load alone, so a
/// chunk stride is not a fact the other three have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmutAddrMemoryOp {
    /// `sentient.receive_and_store`.
    ReceiveAndStore {
        /// `getBurstSize()`.
        burst_size: Elements,
        /// `getInterleavedGroup()`.
        interleaved_group: Elements,
    },
    /// `sentient.load_and_send`.
    LoadAndSend {
        /// `getBurstSize()`.
        burst_size: Elements,
        /// `getInterleavedGroup()`.
        interleaved_group: Elements,
        /// `getChunkStride()` — read by no other arm.
        chunk_stride: Elements,
    },
    /// `sentient.load_and_extract_scalar`.
    LoadAndExtractScalar,
    /// `sentient.load_compute_and_send`.
    LoadComputeAndSend,
}

/// ONE MEMORY OP AS e247 READS IT — the kind, the immutable address and the element size the address
/// is computed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmutAddrMemoryOpInfo {
    /// Which op, and its own attributes.
    pub kind: ImmutAddrMemoryOp,
    /// `getImmutableAddr()` — the operand every test below is about.
    pub immutable_addr: Val,
    /// `getElementSize()`, or `getDstElementSize()` for an LCAS (`:550`) — a width in BITS.
    pub element_size: Bits,
}

impl ImmutAddrMemoryOpInfo {
    /// The `dyn_cast` chain (`Utils.cpp:532-554`) — `None` where the reference reaches
    /// `llvm_unreachable("unexpected op type")`.
    #[must_use]
    pub fn of(op: &Op) -> Option<ImmutAddrMemoryOpInfo> {
        match op {
            Op::Sentient(ops::Op::ReceiveAndStore {
                immutable_addr,
                extent,
                interleaved_group,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::ReceiveAndStore {
                    burst_size: extent.burst_size,
                    interleaved_group: *interleaved_group,
                },
                immutable_addr: *immutable_addr,
                element_size: extent.element_size,
            }),
            Op::Sentient(ops::Op::LoadAndSend {
                immutable_addr,
                extent,
                interleaved_group,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadAndSend {
                    burst_size: extent.burst_size,
                    interleaved_group: *interleaved_group,
                    chunk_stride: extent.chunk_stride,
                },
                immutable_addr: *immutable_addr,
                element_size: extent.element_size,
            }),
            Op::Sentient(ops::Op::LoadAndExtractScalar {
                immutable_addr,
                element_size,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadAndExtractScalar,
                immutable_addr: *immutable_addr,
                element_size: *element_size,
            }),
            Op::Sentient(ops::Op::LoadComputeAndSend {
                immutable_addr,
                dst_element_size,
                ..
            }) => Some(ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadComputeAndSend,
                immutable_addr: *immutable_addr,
                element_size: *dst_element_size,
            }),
            _ => None,
        }
    }
}

/// `isConstant<mlir::sentient::ConstantOp>` (`Utils/Utils.cpp:424-442`) AND THE CONSTANTS IT FOUND —
/// so `is_imm_size_valid`'s opening `DT_CHECK_MSG(isConstant(imm), …)` (`DccExtContext.cpp:34`) is
/// this value's existence rather than a check the range test repeats.
///
/// ⛔ NOT THE NARROWER PRIVATE COPIES: `rematerialization_pass/mod.rs:173` and
/// `scalar_copy_insertion_for_symbols/mod.rs:356` each hold one that omits the query-map arm, and
/// they should collapse into this when their units are reviewed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstantImm {
    /// A `sentient.scalar_constant` — one value for every unit.
    Scalar(i64),
    /// A `uniform.query_map` whose mapping targets are all constants — one value per unit.
    PerUnit(Vec<i64>),
}

/// `isConstant<mlir::sentient::ConstantOp>(val)` (`Utils/Utils.cpp:424-442`), keeping the values.
///
/// ⭐ `None` COVERS ALL THREE OF THE REFERENCE'S FALSE PATHS: a block argument (`defs.of` answers
/// `None`), an op that is neither a constant nor a query map, and a query map with a non-constant
/// target — which is also the empty list `getConstantTargetValues` returns for the last of those.
#[must_use]
pub fn constant_imm(val: Val, defs: Definitions<'_>) -> Option<ConstantImm> {
    match defs.of(val) {
        Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => {
            Some(ConstantImm::Scalar(*value))
        }
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let values = constant_target_values(*map, defs);
            if values.is_empty() {
                None
            } else {
                Some(ConstantImm::PerUnit(values))
            }
        }
        _ => None,
    }
}

/// `Isa::typeToImmInfo.at(isa.getOpcodeType(OpCodeT::LDSTIU))` for one unit
/// (`Utils/DccExtContext.cpp:40-45`).
///
/// ⭐ `DT_CHECK(sizeSignMap.size() == 1)` (`:42`) FAILS THE BUILD HERE, not the run: this is a
/// `const fn` and the four call sites below are `const` items, so a table with two immediate fields on
/// the LDSTIU type is a compile error.
const fn ldstiu_imm_info(comp: fields::Comp) -> (ImmWidth, Sign) {
    let opcodes = comp.opcodes();
    let mut i = 0;
    let mut ldstiu = None;
    while i < opcodes.len() {
        if str_eq(opcodes[i].op, "LDSTIU") {
            ldstiu = Some(opcodes[i].ty);
        }
        i += 1;
    }
    let ty = match ldstiu {
        Some(ty) => ty.get(),
        None => panic!("every memory unit defines LDSTIU (`isa.cpp:1132`, `:1198`, `:1267`, `:1359`)"),
    };

    let all = comp.fields();
    let mut j = 0;
    let mut info = None;
    let mut found = 0;
    while j < all.len() {
        if all[j].ty.get() == ty
            && let ImmSpec::Imm { bits, sign } = all[j].imm
        {
            info = Some((bits, sign));
            found += 1;
        }
        j += 1;
    }
    match (info, found) {
        (Some(info), 1) => info,
        _ => panic!(
            "the LDSTIU type must have exactly one immediate field — \
             DT_CHECK(sizeSignMap.size() == 1) (`Utils/DccExtContext.cpp:42`)"
        ),
    }
}

/// `str::eq` is not `const`, and [`ldstiu_imm_info`] — like `SetMaskRE`'s e377 — needs the opcode
/// spelling compared at build time.
pub(crate) const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const LDSTIU_IMM_L0LU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::L0lu);
const LDSTIU_IMM_L0SU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::L0su);
const LDSTIU_IMM_LXLU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::Lxlu);
const LDSTIU_IMM_LXSU: (ImmWidth, Sign) = ldstiu_imm_info(fields::Comp::Lxsu);

/// `DccExtContext::is_imm_size_valid` (`Utils/DccExtContext.cpp:32-75`) — whether the constant
/// immutable address still fits the unit's LDSTIU immediate field once scaled.
///
/// ⛔ NOT AN ANCHORED UNIT: `dcc/src/Utils/` is outside this campaign's file list, and this is the
/// same in-scope-because-it-is-needed case as
/// [`machine_max_reg_num`](super::local_region_splitting_for_value_commoning).
///
/// TRAP: the reference computes `(imm * element_size) / 8 / scale` in a 32-bit `int` and compares
/// against `pow(2, immSize)` in `double`; the width is at most 14 bits, so `i64` throughout is the
/// same answer without the overflow.
#[must_use]
pub(crate) fn imm_size_valid<A: Arch>(
    unit: ScalarOpComp,
    imm: &ConstantImm,
    element_size: Bits,
) -> bool {
    let (bits, sign) = match unit {
        ScalarOpComp::L0lu => LDSTIU_IMM_L0LU,
        ScalarOpComp::L0su => LDSTIU_IMM_L0SU,
        ScalarOpComp::Lxlu => LDSTIU_IMM_LXLU,
        ScalarOpComp::Lxsu => LDSTIU_IMM_LXSU,
    };
    let scale = i64::from(compute_address_scale::<A>(unit).get());
    let width = i64::from(bits.get());
    let valid = |value: i64| -> bool {
        let val = (value * i64::from(element_size.0)) / 8 / scale;
        match sign {
            Sign::Unsigned => val < (1 << width) && val >= 0,
            Sign::Signed => val < (1 << (width - 1)) && val >= -(1 << (width - 1)),
            Sign::ModuloUnsigned => val <= (1 << width) && val >= -(1 << width),
        }
    };
    match imm {
        ConstantImm::Scalar(value) => valid(*value),
        // ⭐ EVERY UNIT'S TARGET MUST FIT, and `llvm::all_of` over the empty list is true — which is
        // the symbol case the reference notes performs no range check (`:65-67`).
        ConstantImm::PerUnit(values) => values.iter().all(|value| valid(*value)),
    }
}

/// Replaces: e247_memoryOpRequiresImmutAddrScalarCopy
///
/// Whether a memory op's constant immutable address must be copied into a scalar register rather than
/// ridden as an LDSTIU immediate (`Utils.cpp:526-590`).
///
/// TRAP: the LCAS arm can only ever return `false` — `DT_CHECK_MSG(is_constant && is_in_range, …)`
/// (`:581`) throws whenever `res` would be true, and `DT_CHECK_MSG` is not debug-gated
/// (`util/dt_exception.hpp:110-118`). TRAP: `load_and_send`'s `chunk_stride` DEFAULTS TO **1**
/// (`SentientOps.td:517`), so an L0LU load's `chunk_stride_present` is normally TRUE.
#[must_use]
pub fn memory_op_requires_immut_addr_scalar_copy<A: Arch>(
    unit_type: MemoryUnit,
    op: &ImmutAddrMemoryOpInfo,
    do_range_check: RangeCheck,
    defs: Definitions<'_>,
) -> bool {
    // ⛔ A NON-CONSTANT ADDRESS IS NOT AN EARLY RETURN. It makes every arm's ANSWER `false`, but the
    // LCAS `DT_CHECK`s (`:579`, `:581`) and the trailing `llvm_unreachable` (`:589`) are reached
    // whatever `is_constant` is, so it stays a flag.
    let imm = constant_imm(op.immutable_addr, defs);
    let is_constant = imm.is_some();
    let is_in_range = match (&imm, do_range_check, unit_type.with_imm()) {
        (Some(imm), RangeCheck::Check, Some(unit)) => {
            imm_size_valid::<A>(unit, imm, op.element_size)
        }
        // Not constant, `do_range_check` off, or an L3 unit with no IMM for these operations
        // (`:559-562`).
        _ => true,
    };

    // ⛔ THE ORDER IS THE `else if` CHAIN'S: the two scalar/compute ops answer before the L3 arm, so an
    // LAE or LCAS on an L3 unit reaches its own arm with `is_in_range` forced true (`:576-585`).
    match (op.kind, unit_type) {
        (
            ImmutAddrMemoryOp::ReceiveAndStore {
                burst_size,
                interleaved_group,
            },
            MemoryUnit::L0su | MemoryUnit::Lxsu,
        ) => {
            is_constant
                && (burst_size > Elements(1) || interleaved_group > Elements(0) || !is_in_range)
        }
        (
            ImmutAddrMemoryOp::LoadAndSend {
                burst_size,
                interleaved_group,
                chunk_stride,
            },
            MemoryUnit::L0lu | MemoryUnit::Lxlu,
        ) => {
            // `chunk_stride_present` IS AN L0LU-ONLY TERM: *"In case of chunk strides, we should use
            // LRF for offset"* (`:542`), and the test the reference writes is `unit_type == L0LU`
            // (`:543`).
            let chunk_stride_present =
                unit_type == MemoryUnit::L0lu && chunk_stride > Elements(0);
            is_constant
                && (burst_size > Elements(1)
                    || interleaved_group > Elements(0)
                    || chunk_stride_present
                    || !is_in_range)
        }
        (ImmutAddrMemoryOp::LoadAndExtractScalar, _) => is_constant && !is_in_range,
        (ImmutAddrMemoryOp::LoadComputeAndSend, MemoryUnit::Lxlu) => {
            if is_constant && is_in_range {
                false
            } else {
                panic!(
                    "sentient::LoadComputeAndSendOp requires immediate immutable address in range \
                     (`Utils.cpp:581`)"
                )
            }
        }
        (ImmutAddrMemoryOp::LoadComputeAndSend, _) => {
            panic!("DT_CHECK(unit_type == LXLU) (`Utils.cpp:579`): {unit_type:?}")
        }
        // No imm in L3LU/L3SU, so being constant at all is the whole answer (`:585-587`).
        (
            ImmutAddrMemoryOp::ReceiveAndStore { .. } | ImmutAddrMemoryOp::LoadAndSend { .. },
            MemoryUnit::L3lu | MemoryUnit::L3su,
        ) => is_constant,
        (ImmutAddrMemoryOp::ReceiveAndStore { .. } | ImmutAddrMemoryOp::LoadAndSend { .. }, _) => {
            panic!(
                "llvm_unreachable(\"unexpected operation or unit type\") (`Utils.cpp:589`): \
                 {:?} on {unit_type:?}",
                op.kind
            )
        }
    }
}

/// Replaces: e248_getFoldModeAttributeIfExists
///
/// An op's `fold_mode` attribute, `None` when it carries none (`Utils.cpp:593-598`).
///
/// ⭐ THE `if (!op)` NULL CHECK IS THE `&Op` PARAMETER, and `hasAttr("fold_mode")` is WHICH VARIANT:
/// the four compute ops are the only ones the `.td` gives the attribute to, and on those it is already
/// an `Option` because it is an `OptionalAttr` (`SentientOps.td:248`, `:303`, `:335`, `:375`).
#[must_use]
pub fn fold_mode_attribute_if_exists(op: &Op) -> Option<ops::FoldMode> {
    match op {
        Op::Sentient(
            ops::Op::VectorMac { fold_mode, .. }
            | ops::Op::VectorBinary { fold_mode, .. }
            | ops::Op::VectorUnary { fold_mode, .. }
            | ops::Op::VectorTernary { fold_mode, .. },
        ) => *fold_mode,
        _ => None,
    }
}

/// Replaces: e250_hasUniformizeRegion
///
/// Whether a program unit holds a `uniform.uniformize_regions` or a `uniform.equalize_pattern`
/// anywhere inside it (`Utils.cpp:622-632`).
///
/// ⭐ ONE FUNCTION, TWO DECLARATIONS: this body and `OldRegisterInitialization.cpp:536-546` are the
/// same walk twice — they differ only in namespace qualification and line wrapping — and that one is
/// already ported as `e107_hasUniformizeRegion`, so this FORWARDS to it rather than walking twice.
#[must_use]
pub fn has_uniformize_region(unit_body: &[Op]) -> bool {
    super::old_register_initialization::register_init_info::has_uniformize_region(unit_body)
}

// e251_add, e252_size and e253_areAllValuesEqual are ported ONE MODULE DOWN, on the
// `UnitsAndTheirValues` that already carries e249_normalizeNullValues:
// `units_and_their_values.rs`. The scheduler split one C++ class (`Utils.hpp:161-190`) across two
// homes; a second Rust type with the same fields would be the split made real.

// crustify:todo: e392_getQueryKeyAndUnitsFromParentRegion
//   authority : dcc/src/Transform/Sentient/Utils.cpp:40  (24 body lines, level 1)
//   original  : mlir::LogicalResult getQueryKeyAndUnitsFromParentRegion( Operation *op, Value &key, SmallVector<Value> &units)
//   calls     : e252_size

// crustify:todo: e393_createSentientForOpWithAdditionalIterArgs
//   authority : dcc/src/Transform/Sentient/Utils.cpp:195  (59 body lines, level 1)
//   original  : Operation *createSentientForOpWithAdditionalIterArgs( Operation *loop_op, std::vector<std::pair<Value, Value>> &start_vals_and_steps)
//   calls     : e252_size

// crustify:todo: e395_pruneOutOfScopeEntries
//   authority : dcc/src/Transform/Sentient/Utils.cpp:635  (55 body lines, level 1)
//   original  : void pruneOutOfScopeEntries(mlir::uniform::DefImmutableMappingOp map_op, SmallVectorImpl<Operation *> &to_be_deleted)
//   calls     : e252_size

// crustify:todo: e396_replaceValue
//   authority : dcc/src/Transform/Sentient/Utils.hpp:175  (4 body lines, level 1)
//   original  : void replaceValue(size_t index, mlir::Value new_val)
//   calls     : e252_size

// crustify:todo: e546_findAndReplaceRedundantIterArgsUsedInConditions
//   authority : dcc/src/Transform/Sentient/Utils.cpp:257  (138 body lines, level 3)
//   original  : LogicalResult findAndReplaceRedundantIterArgsUsedInConditions( mlir::sentient::ForOp loop, std::set<int> &iter_arg_indices_to_delete)
//   calls     : e422_insert

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{LocalRegion, UniformRegions};

    /// `sentient.scalar_constant` — the constant an immutable address resolves to.
    fn scalar_constant(result: Val, value: i64) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: ops::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.load_and_send` reading `immutable_addr`, with the `.td`'s own extent defaults.
    fn load_and_send(immutable_addr: Val, chunk_stride: Elements) -> Op {
        Op::Sentient(ops::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr,
            increment: Val(3),
            consumer: SendEnd::to_self(Val(4)),
            result: Val(5),
            extent: ops::Extent {
                chunk_stride,
                ..ops::Extent::of(Elements(64), Bits(32))
            },
            interleaved_group: Elements(0),
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

    /// `sentient.vector_unary`, carrying `fold_mode` or not.
    fn vector_unary(fold_mode: Option<ops::FoldMode>) -> Op {
        Op::Sentient(ops::Op::VectorUnary {
            mask: Val(2),
            op_a: ops::Operand::from(ops::Port::Zero),
            unary_op: ops::UnaryOp::Floor,
            result: ops::ResultPorts::default(),
            compute_precision: ops::Precision::Fp16,
            fold_mode,
            unroll_factor: ops::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// AN L0LU LOAD NEEDS THE COPY FOR ITS CHUNK STRIDE ALONE, the same load on LXLU does not, and a
    /// non-constant address needs none — the three ways the `is_constant && (…)` arm resolves.
    #[test]
    fn e247_memory_op_requires_immut_addr_scalar_copy() {
        let scope = vec![
            scalar_constant(Val(2), 0),
            load_and_send(Val(2), Elements(1)),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);
        let op = ImmutAddrMemoryOpInfo::of(&scope[1]).expect("a load_and_send is one of the four");

        assert!(memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::L0lu,
            &op,
            RangeCheck::Check,
            defs
        ));
        assert!(!memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &op,
            RangeCheck::Check,
            defs
        ));

        // A block argument is not a constant, so no arm can ask for the copy.
        let unbound = ImmutAddrMemoryOpInfo {
            immutable_addr: Val(99),
            ..op
        };
        assert!(!memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::L0lu,
            &unbound,
            RangeCheck::Check,
            defs
        ));
    }

    /// A NON-CONSTANT ADDRESS STILL REACHES THE LCAS `DT_CHECK_MSG`: `is_constant` is a term OF that
    /// check (`Utils.cpp:581`), not a guard in front of it, so a block-argument address throws where
    /// the load arms merely answer `false`.
    #[test]
    #[should_panic(expected = "Utils.cpp:581")]
    fn e247_a_non_constant_address_still_throws_on_an_lcas() {
        let scope: Vec<Op> = Vec::new();
        let regions: [&[Op]; 1] = [&scope];
        let _ = memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &ImmutAddrMemoryOpInfo {
                kind: ImmutAddrMemoryOp::LoadComputeAndSend,
                immutable_addr: Val(99),
                element_size: Bits(32),
            },
            RangeCheck::Check,
            Definitions::from_innermost(&regions),
        );
    }

    /// AN ADDRESS THAT DOES NOT FIT THE UNIT'S IMMEDIATE FORCES THE COPY on the unit whose field is
    /// narrower, and the wider one takes the same address as an immediate.
    #[test]
    fn e247_an_out_of_range_constant_needs_the_copy() {
        let scope = vec![
            scalar_constant(Val(2), 8_000),
            load_and_send(Val(2), Elements(0)),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);
        let op = ImmutAddrMemoryOpInfo::of(&scope[1]).expect("a load_and_send is one of the four");

        // 8000 * 32 bits / 8 = 32,000, past LXLU's 14-bit signed field but not past `Skip`.
        assert!(memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &op,
            RangeCheck::Check,
            defs
        ));
        assert!(!memory_op_requires_immut_addr_scalar_copy::<Dd2>(
            MemoryUnit::Lxlu,
            &op,
            RangeCheck::Skip,
            defs
        ));
    }

    /// THE ATTRIBUTE IS RETURNED WHERE IT EXISTS, and an op the `.td` gives no `fold_mode` answers
    /// `None` rather than `FoldMode::None`.
    #[test]
    fn e248_get_fold_mode_attribute_if_exists() {
        assert_eq!(
            fold_mode_attribute_if_exists(&vector_unary(Some(ops::FoldMode::FoldAbBoth))),
            Some(ops::FoldMode::FoldAbBoth)
        );
        assert_eq!(fold_mode_attribute_if_exists(&vector_unary(None)), None);
        assert_eq!(
            fold_mode_attribute_if_exists(&scalar_constant(Val(2), 0)),
            None
        );
    }

    /// THE RE-EXPORT ANSWERS FOR BOTH MEMBERS OF THE `isa<>`, and `false` for a unit holding neither.
    #[test]
    fn e250_has_uniformize_region() {
        assert!(has_uniformize_region(&[Op::Uniform(
            uniform::Op::UniformizeRegions {
                regions: Vec::new(),
                results: Vec::new(),
            }
        )]));
        assert!(!has_uniformize_region(&[scalar_constant(Val(2), 0)]));
    }

    use core::num::NonZeroU32;

    /// `%r = sentient.scalar_constant {value} : index`.
    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `%out = sentient.scalar_copy %input : index`.
    fn copy(input: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Lrf,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    /// `%out = sentient.scalar_sub %lhs, %rhs : index`.
    fn sub(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarSub {
            lhs,
            rhs,
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// `sentient.nop`.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// `sentient.yield`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    /// One carried value, with nothing assigned to it yet.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `%r = sentient.for %iv = 0 to %bound iter_args(..) { body }`.
    fn for_op(iv: Val, bound: Val, carried: Vec<sentient::Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            carried,
            dbg_name: None,
            body,
        })
    }

    /// `sentient.vector_unary` computing `unary_op`.
    fn reduction_unary(unary_op: sentient::UnaryOp) -> Op {
        Op::Sentient(sentient::Op::VectorUnary {
            mask: Val(90),
            op_a: sentient::Operand::from(sentient::Port::North),
            unary_op,
            result: sentient::ResultPorts::default(),
            compute_precision: sentient::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// Which op each position of a block holds, for asserting a layout.
    fn kind(op: &Op) -> &'static str {
        match op {
            Op::Sentient(sentient::Op::ScalarConstant { .. }) => "constant",
            Op::Sentient(sentient::Op::ScalarCopy { .. }) => "copy",
            Op::Sentient(sentient::Op::For { .. }) => "for",
            Op::Sentient(sentient::Op::Nop { .. }) => "nop",
            _ => "other",
        }
    }

    /// A target unroll count.
    fn n(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("the test's own literal is nonzero")
    }

    /// e239: the op at the index, and the walk past the end that the reference dereferences.
    #[test]
    fn operation_of_block_indexes_the_block_and_stops_at_its_end() {
        let block = vec![
            constant(Val(0), 1),
            constant(Val(1), 2),
            constant(Val(2), 3),
        ];
        assert!(matches!(
            operation_of_block(&block, InBlock(1)),
            Some(Op::Sentient(sentient::Op::ScalarConstant { value: 2, .. }))
        ));
        assert!(operation_of_block(&block, InBlock(3)).is_none());
    }

    /// e240: a group expands transitively, `units` comes back reversed and `indices` is appended to.
    #[test]
    fn select_indices_for_units_flattens_groups_and_reverses_the_units() {
        /// Ten times the value number, standing in for an out-of-scope index map.
        struct TenTimes;
        impl UnitIndexMap for TenTimes {
            fn index_of(&self, unit: Val) -> UnitIndex {
                UnitIndex(unit.0 * 10)
            }
        }
        let scope = vec![Op::Dataflow(dataflow::Op::CreateGroup {
            result: Val(1),
            unit_ids: vec![Val(2), Val(3)],
        })];
        let scopes: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&scopes);
        let mut units = vec![Val(0), Val(1)];
        let mut indices = vec![UnitIndex(99)];
        select_indices_for_units(&mut units, &mut indices, &TenTimes, defs);
        assert_eq!(units, vec![Val(3), Val(2), Val(0)]);
        assert_eq!(
            indices,
            vec![UnitIndex(99), UnitIndex(30), UnitIndex(20), UnitIndex(0)]
        );
    }

    /// e241: the copy and the operand it reads both leave the loop, landing in front of it in that
    /// order, and a use in another program unit is the one failure.
    #[test]
    fn move_to_common_dominator_hoists_the_operand_chain_ahead_of_the_insert_point() {
        let body = vec![constant(Val(2), 7), copy(Val(2), Val(3))];
        let mut unit = vec![
            constant(Val(0), 4),
            for_op(Val(1), Val(0), Vec::new(), body.clone()),
            nop(),
        ];
        let copy_at = OpAt::at(&[(InBlock(1), 0)], InBlock(1));
        let landed = move_to_common_dominator(
            &mut unit,
            &copy_at,
            &NewUse::SameUnit(OpAt::top(InBlock(2))),
        );
        assert_eq!(landed, Hoisted::Done);
        let layout: Vec<&str> = unit.iter().map(kind).collect();
        assert_eq!(layout, vec!["constant", "constant", "copy", "for", "nop"]);
        let Op::Sentient(sentient::Op::For { body: emptied, .. }) = &unit[3] else {
            panic!("position 3 is still the loop")
        };
        assert!(emptied.is_empty());

        let mut other = vec![
            constant(Val(0), 4),
            for_op(Val(1), Val(0), Vec::new(), body),
            nop(),
        ];
        assert_eq!(
            move_to_common_dominator(&mut other, &copy_at, &NewUse::OtherUnit),
            Hoisted::OutsideProgramUnit
        );
    }

    /// e242: a bare induction variable counts DOWN, `const - iv` counts up, and a normalized IV is
    /// refused when the caller says so.
    #[test]
    fn for_loop_info_if_iv_reads_both_the_bare_and_the_normalized_induction_variable() {
        let body = vec![constant(Val(2), 8), sub(Val(2), Val(1), Val(3))];
        let top = vec![
            constant(Val(0), 8),
            for_op(Val(1), Val(0), Vec::new(), body.clone()),
        ];
        let scopes: [&[Op]; 2] = [&body, &top];
        let defs = Definitions::from_innermost(&scopes);
        assert_eq!(
            for_loop_info_if_iv(Val(1), NormalizedIv::Accepted, defs),
            Some(ForLoopInfo {
                loop_op: ForRef(Val(1)),
                lower_bound: 8,
                upper_bound: 1,
                step: -1,
                iterations: 8,
            })
        );
        assert_eq!(
            for_loop_info_if_iv(Val(3), NormalizedIv::Accepted, defs),
            Some(ForLoopInfo {
                loop_op: ForRef(Val(1)),
                lower_bound: 0,
                upper_bound: 8,
                step: 1,
                iterations: 8,
            })
        );
        assert_eq!(
            for_loop_info_if_iv(Val(3), NormalizedIv::Rejected, defs),
            None
        );
        // A constant is neither, whatever `allow_normalized_iv` says.
        assert_eq!(
            for_loop_info_if_iv(Val(2), NormalizedIv::Accepted, defs),
            None
        );
    }

    /// e243: a reduction caps at 4 on the card and at 1 everywhere else; anything else rounds down to
    /// a power of two up to 8.
    #[test]
    fn round_down_unroll_factor_caps_reductions_at_four_and_only_on_the_card() {
        let reduction = reduction_unary(sentient::UnaryOp::ReductionAdd);
        let plain = nop();
        assert_eq!(
            round_down_unroll_factor(n(3), &reduction, SenTarget::Sentient),
            sentient::UnrollFactor::X3
        );
        assert_eq!(
            round_down_unroll_factor(n(9), &reduction, SenTarget::Sentient),
            sentient::UnrollFactor::X4
        );
        assert_eq!(
            round_down_unroll_factor(n(9), &reduction, SenTarget::Senulator),
            sentient::UnrollFactor::X1
        );
        assert_eq!(
            round_down_unroll_factor(n(3), &plain, SenTarget::Sentient),
            sentient::UnrollFactor::X2
        );
        assert_eq!(
            round_down_unroll_factor(n(7), &plain, SenTarget::Host),
            sentient::UnrollFactor::X4
        );
        assert_eq!(
            round_down_unroll_factor(n(99), &plain, SenTarget::Host),
            sentient::UnrollFactor::X8
        );
    }

    /// e244: all six, and negating twice is the identity.
    #[test]
    fn negate_predicate_is_total_and_an_involution() {
        use sentient::CmpPredicate::{Eq, Ne, Sge, Sgt, Sle, Slt};
        let negated: Vec<sentient::CmpPredicate> = [Eq, Ne, Slt, Sle, Sgt, Sge]
            .into_iter()
            .map(negate_predicate)
            .collect();
        assert_eq!(negated, vec![Ne, Eq, Sge, Sgt, Sle, Slt]);
        for pred in [Eq, Ne, Slt, Sle, Sgt, Sge] {
            assert_eq!(negate_predicate(negate_predicate(pred)), pred);
        }
    }

    /// e245: all six — equality is symmetric, so only the four orderings swap.
    #[test]
    fn reverse_predicate_leaves_the_symmetric_predicates_alone() {
        use sentient::CmpPredicate::{Eq, Ne, Sge, Sgt, Sle, Slt};
        let reversed: Vec<sentient::CmpPredicate> = [Eq, Ne, Slt, Sle, Sgt, Sge]
            .into_iter()
            .map(reverse_predicate)
            .collect();
        assert_eq!(reversed, vec![Eq, Ne, Sgt, Sge, Slt, Sle]);
    }

    /// e246: `*curr_iter_arg.user_begin()` IS THE OP THAT READS THE ARG, so an inner `sentient.for`
    /// sitting inside a local region of the outer loop's body still passes the `isa<ForOp>` test —
    /// the enclosing `uniform.equalize_pattern` is not the user.
    #[test]
    fn outermost_const_initialization_reads_the_user_and_not_its_enclosing_op() {
        let inner_body = vec![yield_op(vec![Val(7)])];
        let inner_loop = for_op(
            Val(6),
            Val(1),
            vec![carried(Val(4), Val(7), Val(8))],
            inner_body.clone(),
        );
        let outer_body = vec![
            Op::UniformRegions(UniformRegions::EqualizePattern {
                regions: vec![LocalRegion {
                    arg: Val(20),
                    units: Vec::new(),
                    body: vec![inner_loop],
                }],
            }),
            yield_op(vec![Val(8)]),
        ];
        let top = vec![
            constant(Val(0), 4),
            constant(Val(1), 2),
            constant(Val(2), 0),
            for_op(
                Val(3),
                Val(0),
                vec![carried(Val(2), Val(4), Val(5))],
                outer_body.clone(),
            ),
        ];
        let scopes: [&[Op]; 3] = [&inner_body, &outer_body, &top];
        assert_eq!(
            outermost_const_initialization(Val(7), Definitions::from_innermost(&scopes)),
            Some(OutermostConstInit {
                loop_op: ForRef(Val(3)),
                iter_arg: Some(IterArgIndex(0)),
                size: ChainSize::Iterations(
                    core::num::NonZeroU64::new(8).expect("the product of 2 and 4")
                ),
            })
        );
    }

    /// e246: the reference's own chained-initialization example — an inner iter arg initialised from an
    /// outer one, whose own initializer is a constant, with the sizes multiplied along the way.
    #[test]
    fn outermost_const_initialization_walks_the_chain_to_the_constant() {
        let inner_body = vec![yield_op(vec![Val(7)])];
        let outer_body = vec![
            for_op(
                Val(6),
                Val(1),
                vec![carried(Val(4), Val(7), Val(8))],
                inner_body.clone(),
            ),
            yield_op(vec![Val(8)]),
        ];
        let top = vec![
            constant(Val(0), 4),
            constant(Val(1), 2),
            constant(Val(2), 0),
            for_op(
                Val(3),
                Val(0),
                vec![carried(Val(2), Val(4), Val(5))],
                outer_body.clone(),
            ),
        ];
        let scopes: [&[Op]; 3] = [&inner_body, &outer_body, &top];
        let defs = Definitions::from_innermost(&scopes);
        assert_eq!(
            outermost_const_initialization(Val(7), defs),
            Some(OutermostConstInit {
                loop_op: ForRef(Val(3)),
                iter_arg: Some(IterArgIndex(0)),
                size: ChainSize::Iterations(
                    core::num::NonZeroU64::new(8).expect("the product of 2 and 4")
                ),
            })
        );
        // The outer loop's own iter arg reaches the constant in one step, and only its own bound.
        assert_eq!(
            outermost_const_initialization(Val(4), defs),
            Some(OutermostConstInit {
                loop_op: ForRef(Val(3)),
                iter_arg: Some(IterArgIndex(0)),
                size: ChainSize::Iterations(
                    core::num::NonZeroU64::new(4).expect("the outer bound")
                ),
            })
        );
    }
}
