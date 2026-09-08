// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `UnitFiltering.cpp` — 8 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e140_removeCoresCoreletsFoldsFromProgramUnit` | 140/384 | 20 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240` |
//! | `e141_isDataTransfer` | 141/384 | 11 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314` |
//! | `e203_removeCoresCoreletsFoldsFromDefImmutMap` | 203/384 | 37 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99` |
//! | `e204_cleanup` | 204/384 | 49 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263` |
//! | `e205_isDataTransferToKeep` | 205/384 | 10 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327` |
//! | `e262_removeCoresCoreletsFoldsFromUniformizeRegion` | 262/384 | 98 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139` |
//! | `e263_removeAncestors` | 263/384 | 18 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339` |
//! | `e296_runOnOperation` | 296/384 | 127 | `dcc/src/Transform/Dataflow/UnitFiltering.cpp:362` |

use std::collections::BTreeSet;

use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, agen, dataflow, dbg_name, defining_op, operands, regions, regions_mut,
    uniform, uses,
};

use crate::islands::dataflow_ir::dialects::uniform::MappedTy;

use super::vc_vector_operands::{OpId, defining_position, op_at, remove_at, use_positions};
use crate::units::{Core, Corelet, NumFolds, Residency};

/// WHICH FOLD A UNIT HANDLE IS THE INSTANCE OF — `dyn_cast<OpResult>(unit).getResultNumber()`
/// (`UnitFiltering.cpp:245`).
///
/// ⭐ A `get_unit`'s RESULT NUMBER *IS* THE FOLD. `get_unit` is `Variadic<Index>:$units` with "each
/// return value corresponding to an instance of program time steps" (`Dataflow.td:48,56-58`), so the
/// i-th result is fold i and nothing else records it.
///
/// ⛔ ONLY [`Self::ZERO`] IS CONSTRUCTIBLE, BECAUSE THIS ISLAND BINDS ONE RESULT PER `get_unit`
/// ([`dataflow::Op::GetUnit`] has a single `result: Val`). Folding is
/// [`NumFolds`](crate::units::NumFolds) elsewhere in the crate and no emitter uses it yet; when a
/// variadic `get_unit` lands, this gains a constructor taking the result index and every reader below
/// keeps working. A `pub u32` field would have let a caller invent a fold the IR does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FoldId(u32);

impl FoldId {
    /// The first fold — the only one a single-result `get_unit` can bind.
    pub const ZERO: FoldId = FoldId(0);

    /// The index, for the comparison against a filter.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A NON-EMPTY "KEEP ONLY THESE" SET.
///
/// ⛔⛔ EMPTY MEANS "NO FILTER" IN THE REFERENCE AND THAT IS WHY THIS IS A TYPE. Every clause of
/// entry 140 reads `!filter_x_except_.empty() && filter_x_except_.count(id) == 0`
/// (`UnitFiltering.cpp:250-255`): an empty set keeps everything, a non-empty one keeps only its
/// members. Those are two different modes stored in one `std::set<int>`, and forgetting the
/// `!empty()` guard turns "filter nothing" into "erase everything". Here the mode is
/// `Option<Only<_>>` — [`None`] is no filter, [`Some`] is a set that cannot be empty — so the guard
/// cannot be forgotten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Only<T: Ord>(BTreeSet<T>);

impl<T: Ord> Only<T> {
    /// The filter these ids describe, or [`None`] for "no filtering" — the reference's empty set.
    #[must_use]
    pub fn these(ids: BTreeSet<T>) -> Option<Only<T>> {
        if ids.is_empty() {
            None
        } else {
            Some(Only(ids))
        }
    }

    /// `filter_x_except_.count(id) != 0` — whether this filter keeps `id`.
    #[must_use]
    pub fn keeps(&self, id: &T) -> bool {
        self.0.contains(id)
    }

    /// The ids it names, ascending — `for (auto fold_id : filter_folds_except_)` (`:285`), which
    /// entry 204 counts rather than tests. ⭐ A `std::set` iterates sorted; so does a [`BTreeSet`].
    pub fn ids(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

/// THE PASS'S THREE "EXCEPT" SETS — `filter_folds_except_`, `filter_cores_except_`,
/// `filter_corelets_except_` (`UnitFiltering.cpp:92-94`).
///
/// ⭐ ONE STRUCT BECAUSE THREE ENTRIES READ THE SAME TRIPLE, character for character: entry 140
/// (`:250-255`), entry 262 (`:155-160`) and entry 203 (`:186-191`). The pass member `opts_` that
/// supplies them is an excluded data member (`docs/bridge2-porting-order.md:1120`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnitFilters {
    /// Keep only these folds. [`None`] keeps every fold.
    ///
    /// ⛔ FOLD 0 IS ALWAYS IN IT WHEN IT IS SET:
    /// `DT_CHECK_MSG(filter_folds_except_.empty() || filter_folds_except_.count(0) > 0, "Should
    /// always keep fold #0 if filtering folds.")` (`:383-385`, entry 296). With this island's
    /// single-result `get_unit` every handle is [`FoldId::ZERO`], so the fold clause below is
    /// faithful and cannot fire — see [`FoldId`].
    pub folds: Option<Only<FoldId>>,
    /// Keep only these cores. [`None`] keeps every core.
    pub cores: Option<Only<Core>>,
    /// Keep only these corelets. [`None`] keeps every corelet.
    pub corelets: Option<Only<Corelet>>,
}

/// WHICH FOLD AND WHERE — what the reference reads off a `program_unit` operand before filtering it.
///
/// ⛔ THE REFERENCE DEREFERENCES A NULL HERE IF THE OPERAND IS NOT A `get_unit` RESULT.
/// `llvm::dyn_cast<OpResult>(unit)` yields a null `OpResult` for a block argument and
/// `getResultNumber()` is then called on it unguarded (`UnitFiltering.cpp:244-245`); one line later
/// `dyn_cast<GetUnitOp>(unit.getDefiningOp())` may also be null, which `getCoreId`/`getCoreletId`
/// answer -1 for (`DccExtContext.cpp:78-88`, `:116-124`). So the reference assumes every operand of
/// a `program_unit` is a `get_unit` result — which entry 296 arranges — and this returns [`None`]
/// rather than guessing an id for anything else. The caller KEEPS such an operand; see
/// [`remove_cores_corelets_folds_from_program_unit`].
fn unit_bound_by(unit: Val, scope: &[DfirOp]) -> Option<(FoldId, Residency)> {
    match defining_op(unit, scope) {
        Some(DfirOp::Dataflow(dataflow::Op::GetUnit { residency, .. })) => {
            Some((FoldId::ZERO, *residency))
        }
        Some(
            DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::Arith(_)
            | DfirOp::Affine(_)
            | DfirOp::Scf(_)
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ `symbol` HERE TOO: a `symbol.create_symbol` result is an `index`, not the `!ddl.unit`
            // a `program_unit` operand is, so it is one more `dyn_cast<GetUnitOp>` null.
            // ⭐ AND `uniform`: a `program_unit` operand defined by a `uniform.query_map` is an
            // `index` the schedule resolves later, not the `get_unit` result this `dyn_cast` wants,
            // so the operand is KEPT rather than filtered — see the note above.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_),
        )
        | None => None,
    }
}

/// `dcc::getCoreletId(get_unit_op)` — the `corelet` attribute, or [`None`] for the reference's -1
/// (`DccExtContext.cpp:116-124`).
///
/// ⛔⛔ A CORE-WIDE UNIT'S CORELET IS **0**, NOT ABSENT, and that is the whole reason
/// [`Residency`] separates [`Residency::CoreWide`] from [`Residency::Scratchpad`]. The scheduler's
/// own materializer writes `core` AND `corelet = 0` for a unit declared in a
/// `group { kind = "core" }` — the L3 halves (`UnitMaterializer.cpp:62-80`) — while a per-core
/// scratchpad carries `core` and no `corelet` at all (`:142-152`). So a `--filter-corelets=1` run
/// erases the L3 units, and a port that answered "absent" for them would keep units the reference
/// drops.
const fn corelet_id(residency: Residency) -> Option<Corelet> {
    match residency {
        // No `corelet` attribute at all: `getCoreletId` returns -1.
        Residency::Global | Residency::Scratchpad { .. } => None,
        // `corelet = 0`, written explicitly.
        Residency::CoreWide { .. } => Corelet::checked(0),
        Residency::Corelet { corelet, .. } => Some(corelet),
    }
}

/// THE THREE-CLAUSE FILTER TEST — `UnitFiltering.cpp:250-255` (entry 140) and `:186-191`
/// (entry 203), character for character in the reference and once here.
///
/// ⛔ -1 IS NOT A CORE AND IT IS IN NO FILTER SET, so a non-empty core filter erases a core-less
/// unit; the corelet clause's `corelet_id != -1` guard is the OPPOSITE — an absent corelet survives.
/// See [`UnitFilters`] and [`corelet_id`].
fn unit_is_filtered_out(unit: Val, scope: &[DfirOp], filters: &UnitFilters) -> bool {
    // `dyn_cast<OpResult>(unit).getResultNumber()` and
    // `dyn_cast<GetUnitOp>(unit.getDefiningOp())`, together.
    let Some((fold_id, residency)) = unit_bound_by(unit, scope) else {
        // Not a `get_unit` result — see [`unit_bound_by`]. Kept.
        return false;
    };

    // `!filter_folds_except_.empty() && filter_folds_except_.count(fold_id) == 0`.
    let wrong_fold = filters
        .folds
        .as_ref()
        .is_some_and(|only| !only.keeps(&fold_id));
    // `!filter_cores_except_.empty() && filter_cores_except_.count(core_id) == 0`, where a core-less
    // unit's `core_id` is -1 and no set holds it.
    let wrong_core = filters
        .cores
        .as_ref()
        .is_some_and(|only| match residency.core() {
            Some(core) => !only.keeps(&core),
            None => true,
        });
    // `!filter_corelets_except_.empty() && corelet_id != -1 &&
    //  filter_corelets_except_.count(corelet_id) == 0`.
    let wrong_corelet = filters
        .corelets
        .as_ref()
        .is_some_and(|only| match corelet_id(residency) {
            Some(corelet) => !only.keeps(&corelet),
            None => false,
        });

    wrong_fold || wrong_core || wrong_corelet
}

/// Replaces: e140_removeCoresCoreletsFoldsFromProgramUnit
///
/// **140/384** `UnitFilteringPass::removeCoresCoreletsFoldsFromProgramUnit` —
/// `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240` (20L).
///
/// ```cpp
/// void UnitFilteringPass::removeCoresCoreletsFoldsFromProgramUnit(
///     dataflow::ProgramUnitOp unit_op) {
///   for (int i = unit_op.getNumOperands() - 1; i >= 0; --i) {
///     Value unit = unit_op.getUnits()[i];
///     auto get_unit_op_result = llvm::dyn_cast<OpResult>(unit);
///     unsigned fold_id = get_unit_op_result.getResultNumber();
///     auto get_unit_op =
///         llvm::dyn_cast<mlir::dataflow::GetUnitOp>(unit.getDefiningOp());
///     unsigned core_id = dcc::getCoreId(get_unit_op);
///     int corelet_id = dcc::getCoreletId(get_unit_op);
///     if ((!filter_folds_except_.empty() &&
///          filter_folds_except_.count(fold_id) == 0) ||
///         (!filter_cores_except_.empty() &&
///          filter_cores_except_.count(core_id) == 0) ||
///         (!filter_corelets_except_.empty() && corelet_id != -1 &&
///          filter_corelets_except_.count(corelet_id) == 0)) {
///       unit_op->eraseOperand(i);
///     }
///   }
///   DT_CHECK_MSG(unit_op.getNumOperands() >= 1,
///                "Should not filter out all units");
/// }
/// ```
///
/// # ⛔ THE OPERAND LIST *IS* THE UNIT LIST, WHICH IS WHY THE SIGNATURE IS THIS
///
/// `program_unit` declares `(ins Variadic<Index>:$units, OptionalAttr<StrAttr>:$precision)`
/// (`Dataflow.td:107`) — one variadic operand group and an attribute — so `getNumOperands()` and
/// `getUnits().size()` are the same number and `eraseOperand(i)` erases a unit. The `iter_arg :`
/// spelling in its syntax is printed sugar over those same operands (`:99-104`), not a second group.
/// Taking `&mut Vec<Val>` is therefore the whole mutation this function performs; the enclosing
/// [`dataflow::Op::ProgramUnit`] is not otherwise touched.
///
/// # ⭐ THE REVERSE LOOP IS MECHANISM, AND THE DECISION IS PER UNIT
///
/// `for (int i = getNumOperands() - 1; i >= 0; --i)` counts DOWN because `eraseOperand(i)` shifts
/// every later index. No clause reads another operand, so the surviving set is the same either way
/// and [`Vec::retain`] expresses it in one pass — the brief's "mechanism for reaching operands" that
/// a port may drop (`AGENT-BRIEF.md:53-56`).
///
/// # ⛔ -1 IS NOT A CORE AND IT IS IN NO FILTER SET
///
/// `getCoreId` answers -1 for a unit with no `core` attribute (`DccExtContext.cpp:80-87`), the sets
/// are `std::set<int>` filled from non-negative command-line ids (`:92-94`, `:387-400`), so
/// `count(-1)` is 0 and a non-empty core filter ERASES every core-less unit — the HBM handle among
/// them ([`Residency::Global`]). That is modelled by `residency.core()` being [`None`], not
/// papered over. The corelet clause is the opposite: its `corelet_id != -1` guard means an absent
/// corelet SURVIVES any corelet filter. Two -1s, two different answers, and mixing them up would
/// silently drop or keep whole units.
///
/// # ⭐ THE CLOSING `DT_CHECK_MSG` HAS NO COUNTERPART
///
/// `DT_CHECK_MSG(unit_op.getNumOperands() >= 1, "Should not filter out all units")` (`:259-260`) is a
/// hard stop on the *user's filter options* — it fires only when the filters name no surviving core
/// at all — and it is a stop in every build (a `DtException` throw, `util/dt_exception.hpp:110-118`,
/// or an `assert`, `dataflow-scheduler/include/dataflow-scheduler/dt_exception.h:34`). This crate
/// never runtime-refuses (`CLAUDE.md`), and there is nothing here to refuse about: the function
/// faithfully removes what the filters exclude, and an empty result is the caller's option set being
/// contradictory. Entry 296, which builds the filters, is where that belongs — it already validates
/// the fold set the same way (`:383-385`).
pub fn remove_cores_corelets_folds_from_program_unit(
    units: &mut Vec<Val>,
    scope: &[DfirOp],
    filters: &UnitFilters,
) {
    // The three clauses are [`unit_is_filtered_out`]; `unit_op->eraseOperand(i)` is what this drops.
    units.retain(|unit| !unit_is_filtered_out(*unit, scope, filters));
}

/// Replaces: e141_isDataTransfer
///
/// **141/384** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314` (11L).
///
/// ```cpp
/// static bool isDataTransfer(Operation *op) {
///   return isa<dataflow::SyncSendOp, dataflow::SyncRecvOp,
///              dataflow::ImplicitSyncOnStreamingBufferOp, dataflow::SendOp,
///              dataflow::ReceiveOp, dataflow::OpaqueOp, agen::VectorLoadOp,
///              agen::VectorStoreOp, agen::CompositeLoadOp, agen::CompositeStoreOp,
///              agen::CompositeLoadAndStoreOp,
///              agen::CompositeIndirectLoadAndStoreOp,
///              agen::CompositeIndirectLoadOp, agen::CompositeIndirectStoreOp,
///              agen::IndirectVectorLoadOp, agen::IndirectVectorStoreOp,
///              agen::CompositeMemoryInterleaveOp, agen::SymbolicVectorLoadOp,
///              agen::SymbolicVectorStoreOp>(op);
/// }
/// ```
///
/// # ⭐ NINETEEN CLASSES, NINE OF WHICH THIS ISLAND DECLARES
///
/// The six `dataflow` ones all exist here. Of the thirteen `agen` ones, the island declares
/// `vector_load`, `vector_store` and `composite_load_and_store` — "the four here are the ones an
/// emitted program contains" ([`agen`]) — and the other ten (`composite_load`, `composite_store`,
/// the four indirect composites, the two indirect vectors, `composite_memory_interleave` and the two
/// symbolic vectors) are not declared. They are listed in the match below as comments rather than
/// invented: the brief's rule to grow the island is about a function's *input*
/// (`AGENT-BRIEF.md:57`), and this function's input is any op at all.
///
/// # ⛔ NO WILDCARD, SO A NEW ISLAND OP CANNOT DEFAULT TO "NOT A TRANSFER"
///
/// A `_ => false` would make the tenth `agen` op silently non-transferring the day it is declared,
/// and entry 205 (`isDataTransferToKeep`, `:327`) is the only caller: it decides whether a transfer
/// SURVIVES unit filtering. Getting a `false` for a real transfer deletes data movement from the
/// program. Every arm below is spelled out, so declaring an op is a compile error until this
/// function has an answer for it.
#[must_use]
pub fn is_data_transfer(op: &DfirOp) -> bool {
    match op {
        // The six `dataflow` classes, all of them declared here.
        DfirOp::Dataflow(
            dataflow::Op::SyncSend { .. }
            | dataflow::Op::SyncRecv { .. }
            | dataflow::Op::ImplicitSync { .. }
            | dataflow::Op::Send { .. }
            | dataflow::Op::Receive { .. }
            | dataflow::Op::Opaque(_),
        ) => true,
        // Four of the thirteen `agen` classes. The nine the island does not declare —
        // `composite_store`, `composite_indirect_load_and_store`,
        // `composite_indirect_load`, `composite_indirect_store`, `indirect_vector_load`,
        // `indirect_vector_store`, `composite_memory_interleave`, `symbolic_vector_load`,
        // `symbolic_vector_store` — belong on this side of the answer when they land.
        DfirOp::Agen(
            agen::Op::VectorLoad { .. }
            | agen::Op::VectorStore { .. }
            | agen::Op::CompositeLoad(_)
            | agen::Op::CompositeLoadAndStore(_),
        ) => true,
        // `agen.yield` terminates a transfer's region and is not one; the view and unit binders,
        // `program_unit` and every arith/affine/scf/vectorchain op are not in the `isa<>` list.
        DfirOp::Agen(agen::Op::Yield | agen::Op::SetTransferMaskState { .. })
        | DfirOp::Dataflow(
            dataflow::Op::GetUnit { .. }
            | dataflow::Op::GetLocalUnit { .. }
            | dataflow::Op::CreateGroup { .. }
            | dataflow::Op::GetLogicalMemoryView { .. }
            | dataflow::Op::GetPagedLogicalMemoryView { .. }
            | dataflow::Op::ProgramUnit { .. },
        )
        | DfirOp::Arith(_)
        | DfirOp::Affine(_)
        | DfirOp::Scf(_)
        // ⛔⛔ AND NEITHER IS `vector.load`/`vector.store`, WHICH IS THE ANSWER TO WATCH HERE. The
        // `isa<>` list is nineteen `dataflow` and `agen` classes; upstream's plain accesses are in
        // none of them, so unit filtering does not treat one as data movement. They are also not a
        // form the scheduler produces (see
        // [`vector`](crate::islands::dataflow_ir::dialects::vector)), so no transfer of an emitted
        // program is dropped by this answer.
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        // ⭐ `symbol.create_symbol` IS NOT IN THE NINETEEN-CLASS `isa<>` LIST EITHER. It binds a
        // scalar the schedule fixes later; nothing crosses a datapath because of it.
        // ⭐ AND NO `uniform.` OP IS IN THE NINETEEN-CLASS LIST. `uniform.uniformize_regions` holds
        // the transfers rather than being one; the transfers inside it answer for themselves.
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => false,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::generated::{OpaqueFunc, SyncSignal};
    use crate::islands::dataflow_ir::dialects::{Index, vectorchain};
    use crate::islands::dataflow_ir::link::{Link, Lxsu, Sfp};
    use crate::islands::dataflow_ir::ty::{ElemType, MemRef, Vector};
    use crate::units::DfirUnit;

    /// `vector<64xf16>`.
    const LANES: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };

    /// A core, by index — the arch has 32 (`arch.rs:211`).
    fn core(index: u32) -> Core {
        Core::checked(index).expect("the arch has 32 cores")
    }

    /// A corelet, by index — the arch has 2 per core (`arch.rs:212`).
    fn corelet(index: u32) -> Corelet {
        Corelet::checked(index).expect("the arch has 2 corelets per core")
    }

    /// `%n = dataflow.get_unit {core = c, corelet = 0, name = "lxlu-CL0", type = "lxlu"} : index` —
    /// the unit the vendor's edge case binds 32 of (`core_filtering_edge_case.mlir:710-741`).
    fn lxlu_of_core(result: Val, index: u32) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result,
            residency: Residency::Corelet {
                core: core(index),
                corelet: corelet(0),
            },
            unit: DfirUnit::Lxlu,
            num_folds: None,
        })
    }

    /// The filters, none set — `--dcc-filter-units` with no option at all.
    fn no_filters() -> UnitFilters {
        UnitFilters::default()
    }

    /// 🎯 140/384 — THE VENDOR'S OWN EDGE CASE, ANSWER FOR ANSWER.
    /// `dcc-opt --dcc-filter-units="filter-cores-except=0"` (`core_filtering_edge_case.mlir:2`) over a
    /// `dataflow.program_unit iter_arg : %arg0 -> (%514#0, .., %545#0)` binding the `lxlu-CL0` unit of
    /// every one of the 32 cores (`:710-741`, `:968`). The expectation keeps ONE:
    /// `dataflow.program_unit iter_arg : %[[VAL_134]] -> (%[[VAL_130]]#0)` where `%[[VAL_130]]` is
    /// `{core = 0 : i32, corelet = 0 : i32, name = "lxlu-CL0", .., type = "lxlu"}` (`:143`, `:147`).
    #[test]
    fn a_core_filter_keeps_only_that_cores_unit() {
        let scope: Vec<DfirOp> = (0..32)
            .map(|index| lxlu_of_core(Val(100 + index), index))
            .collect();
        let mut units: Vec<Val> = (0..32).map(|index| Val(100 + index)).collect();
        let filters = UnitFilters {
            cores: Only::these(BTreeSet::from([core(0)])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(units, vec![Val(100)]);
    }

    /// ⛔ AN UNSET FILTER KEEPS EVERYTHING. That is the `!filter_x_except_.empty()` half of every
    /// clause, and it is the case every ordinary compile takes — the pass runs on every module and
    /// erases nothing unless asked.
    #[test]
    fn no_filter_keeps_every_unit() {
        let scope: Vec<DfirOp> = (0..4)
            .map(|index| lxlu_of_core(Val(100 + index), index))
            .collect();
        let mut units: Vec<Val> = (0..4).map(|index| Val(100 + index)).collect();

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &no_filters());

        assert_eq!(units, vec![Val(100), Val(101), Val(102), Val(103)]);
    }

    /// ⛔⛔ A UNIT WITH NO `core` ATTRIBUTE IS ERASED BY ANY CORE FILTER. `getCoreId` answers -1 for
    /// it (`DccExtContext.cpp:80-87`) and no filter set holds -1, so the clause fires. The vendor's
    /// own module binds two such units — `{name = "hbm", ..}` and `{name = "lx", ..}`, neither
    /// carrying `core` (`core_filtering_edge_case.mlir:770-771`).
    #[test]
    fn a_core_less_unit_is_erased_by_any_core_filter() {
        let scope = vec![
            lxlu_of_core(Val(100), 0),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(101),
                residency: Residency::Global,
                unit: DfirUnit::Hbm,
                num_folds: None,
            }),
        ];
        let mut units = vec![Val(100), Val(101)];
        let filters = UnitFilters {
            cores: Only::these(BTreeSet::from([core(0)])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(units, vec![Val(100)], "the HBM handle goes");
    }

    /// ⛔⛔ AND THE OTHER -1 GOES THE OTHER WAY. The corelet clause carries `corelet_id != -1`
    /// (`UnitFiltering.cpp:254`), so a unit with no `corelet` attribute — a per-core scratchpad,
    /// `C0-lx` (`UnitMaterializer.cpp:142-152`) — SURVIVES a corelet filter it is not a member of.
    /// Reading the two -1s the same way would erase every scratchpad from a `filter-corelets` run.
    #[test]
    fn an_absent_corelet_survives_a_corelet_filter() {
        let scope = vec![DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(100),
            residency: Residency::Scratchpad { core: core(0) },
            unit: DfirUnit::Lx,
            num_folds: None,
        })];
        let mut units = vec![Val(100)];
        let filters = UnitFilters {
            corelets: Only::these(BTreeSet::from([corelet(1)])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(units, vec![Val(100)]);
    }

    /// ⛔⛔ BUT A CORE-WIDE UNIT IS CORELET **0**, NOT ABSENT, so the same filter erases it. The
    /// materializer writes `corelet = 0` for a unit declared `group { kind = "core" }` — the L3
    /// halves (`UnitMaterializer.cpp:62-80`) — and `getCoreletId` reads the attribute that is there.
    /// This is the distinction [`Residency`] exists to keep, and the vendor exercises the filter it
    /// turns on: `filter-corelets-except=0` (`corelet_filtering.mlir:2`).
    #[test]
    fn a_core_wide_unit_is_filtered_as_corelet_zero() {
        let scope = vec![DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(100),
            residency: Residency::CoreWide { core: core(0) },
            unit: DfirUnit::L3lu,
            num_folds: None,
        })];
        let mut units = vec![Val(100)];
        let filters = UnitFilters {
            corelets: Only::these(BTreeSet::from([corelet(1)])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(units, Vec::<Val>::new(), "corelet 0 is not corelet 1");
    }

    /// ⛔ THE THREE CLAUSES ARE AN `||`. A unit whose core is kept and whose corelet is not is still
    /// erased — `filter-cores-except=0,1 filter-corelets-except=1`
    /// (`basic_kEmitProgIR_2.mlir:2`) is the vendor's combined run.
    #[test]
    fn passing_the_core_clause_does_not_save_a_filtered_corelet() {
        let scope = vec![
            lxlu_of_core(Val(100), 0),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(101),
                residency: Residency::Corelet {
                    core: core(1),
                    corelet: corelet(1),
                },
                unit: DfirUnit::Lxlu,
                num_folds: None,
            }),
        ];
        let mut units = vec![Val(100), Val(101)];
        let filters = UnitFilters {
            cores: Only::these(BTreeSet::from([core(0), core(1)])),
            corelets: Only::these(BTreeSet::from([corelet(1)])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(
            units,
            vec![Val(101)],
            "core 0's corelet 0 fails the second clause"
        );
    }

    /// ⭐ THE FOLD CLAUSE IS FAITHFUL AND CANNOT FIRE HERE. Every handle this island can bind is
    /// [`FoldId::ZERO`], and entry 296 refuses a fold filter that does not contain 0
    /// (`UnitFiltering.cpp:383-385`), so a legal fold filter always keeps every unit this island has.
    /// The test states that rather than leaving it to be inferred.
    #[test]
    fn a_legal_fold_filter_keeps_every_unit_of_this_island() {
        let scope = vec![lxlu_of_core(Val(100), 0), lxlu_of_core(Val(101), 1)];
        let mut units = vec![Val(100), Val(101)];
        let filters = UnitFilters {
            folds: Only::these(BTreeSet::from([FoldId::ZERO])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(units, vec![Val(100), Val(101)]);
        assert_eq!(FoldId::ZERO.get(), 0);
    }

    /// ⛔ AN OPERAND THAT IS NOT A `get_unit` RESULT IS KEPT, where the reference reads through a
    /// null `OpResult` (`:244-245`). Keeping it cannot lose a unit; erasing on a failed lookup could.
    #[test]
    fn an_operand_with_no_get_unit_is_kept() {
        let scope = vec![lxlu_of_core(Val(100), 0)];
        let mut units = vec![Val(100), Val(999)];
        let filters = UnitFilters {
            cores: Only::these(BTreeSet::from([core(0)])),
            ..no_filters()
        };

        remove_cores_corelets_folds_from_program_unit(&mut units, &scope, &filters);

        assert_eq!(units, vec![Val(100), Val(999)]);
    }

    /// ⛔ AND AN EMPTY SET IS NOT A FILTER. [`Only::these`] maps it to [`None`], which is the
    /// reference's `!filter_x_except_.empty()` guard made structural — a `Some(empty set)` would
    /// erase every unit.
    #[test]
    fn an_empty_filter_set_is_no_filter() {
        assert_eq!(Only::these(BTreeSet::<Core>::new()), None);
        assert!(Only::these(BTreeSet::from([core(3)])).is_some_and(|only| only.keeps(&core(3))));
    }

    /// 🎯 141/384 — the six `dataflow` classes of the `isa<>` list (`UnitFiltering.cpp:315-317`).
    #[test]
    fn the_dataflow_transfers_are_data_transfers() {
        let (send_end, recv_end) = Link::<Sfp, Lxsu>::between(Val(4), Val(5)).ends();
        let transfers = [
            DfirOp::Dataflow(dataflow::Op::SyncSend {
                to: Val(5),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            }),
            DfirOp::Dataflow(dataflow::Op::SyncRecv {
                from: Val(4),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            }),
            DfirOp::Dataflow(dataflow::Op::ImplicitSync {
                view: Val(10),
                dst: Val(5),
                size: Val(6),
                view_ty: MemRef {
                    shape: vec![1, 64],
                    elem: ElemType::F16,
                },
                dbg_name: None,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to: send_end,
                data: Val(20),
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(21),
                from: recv_end,
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Opaque(dataflow::Opaque {
                func: OpaqueFunc::Reciprocal,
                read_write: Vec::new(),
                read_only: Vec::new(),
                params: Vec::new(),
                dbg_name: None,
            })),
        ];

        for op in &transfers {
            assert!(is_data_transfer(op), "{op:?} is in the isa<> list");
        }
    }

    /// 🎯 141/384 — the three `agen` classes the island declares, of the thirteen listed
    /// (`UnitFiltering.cpp:317-324`).
    #[test]
    fn the_agen_accesses_are_data_transfers() {
        let view_ty = MemRef {
            shape: vec![1, 64],
            elem: ElemType::F16,
        };
        let load = DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            access: agen::Access::OfView,
            result: Val(20),
            view: Val(10),
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty: view_ty.clone(),
            ty: LANES,
        });
        let store = DfirOp::Agen(agen::Op::VectorStore {
            value: Val(20),
            view: Val(11),
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty,
            ty: LANES,
        });

        assert!(is_data_transfer(&load));
        assert!(is_data_transfer(&store));
    }

    /// ⛔ AND WHAT IS NOT IN THE LIST. `get_unit` binds a unit, `program_unit` holds a program, a
    /// rotate computes and `agen.yield` terminates a transfer's region — none of them MOVES data, and
    /// entry 205 uses this answer to decide what unit filtering may delete.
    #[test]
    fn binders_terminators_and_compute_are_not_data_transfers() {
        let not_transfers = [
            lxlu_of_core(Val(100), 0),
            DfirOp::Dataflow(dataflow::Op::ProgramUnit {
                units: vec![Val(100)],
                precision: None,
                body: Vec::new(),
            }),
            DfirOp::Agen(agen::Op::Yield),
            DfirOp::VectorChain(vectorchain::Op::Rotate {
                result: Val(21),
                input: Val(20),
                position: Val(3),
                right_shift: true,
                input_ty: LANES,
                ty: LANES,
            }),
        ];

        for op in &not_transfers {
            assert!(!is_data_transfer(op), "{op:?} is not in the isa<> list");
        }
    }

    /// 🎯 203/384 — A REDUCED MAP IS REBUILT; A MAP NOTHING SURVIVES IS LEFT ALONE.
    ///
    /// ⛔ BOTH BRANCHES OF `if (!unit_list_is_reduced || new_keys.empty()) return;` ARE THE
    /// ASSERTION. Rebuilding an empty map would leave a `uniform.def_immutable_mapping` with no keys
    /// behind for a region this pass is about to delete whole (`:126-128`).
    #[test]
    fn a_reduced_map_is_rebuilt_and_an_emptied_one_is_left_alone() {
        let scope = vec![lxlu_of_core(Val(100), 0), lxlu_of_core(Val(101), 1)];
        let pairs = vec![(Val(100), Val(200)), (Val(101), Val(201))];
        let mut vals = Values::default();
        let filters = UnitFilters {
            cores: Only::these(BTreeSet::from([core(0)])),
            ..no_filters()
        };

        let reduced = remove_cores_corelets_folds_from_def_immut_map(
            &mut vals,
            Val(50),
            &pairs,
            MappedTy::Index,
            &scope,
            &filters,
        )
        .expect("core 1's key is filtered out, core 0's survives");
        assert_eq!(Val(50), reduced.to_delete);
        assert_eq!(
            DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                result: reduced.map,
                pairs: vec![(Val(100), Val(200))],
                values_ty: MappedTy::Index,
            }),
            reduced.op
        );

        // ⛔ AND THE TWO CASES THAT MUST NOT REBUILD: nothing filtered, and everything filtered.
        assert!(
            remove_cores_corelets_folds_from_def_immut_map(
                &mut vals,
                Val(50),
                &pairs,
                MappedTy::Index,
                &scope,
                &no_filters()
            )
            .is_none()
        );
        let no_core = UnitFilters {
            cores: Only::these(BTreeSet::from([core(7)])),
            ..no_filters()
        };
        assert!(
            remove_cores_corelets_folds_from_def_immut_map(
                &mut vals,
                Val(50),
                &pairs,
                MappedTy::Index,
                &scope,
                &no_core
            )
            .is_none()
        );
    }

    /// 🎯 204/384 — THE USE-EMPTY BINDERS GO, THE USED `get_unit` IS REBUILT WITH `num_folds`.
    ///
    /// ⛔ `num_folds = 1 : i32` IS THE WHOLE OBSERVABLE REBUILD in this island, so a port that
    /// erased-and-recreated the op without setting the attribute would print the input back
    /// unchanged and pass any test that only counted ops. ⭐ The group that is read survives; the one
    /// nothing reads does not.
    #[test]
    fn use_empty_binders_are_erased_and_a_used_unit_is_rebuilt() {
        let mut module = vec![
            lxlu_of_core(Val(100), 0),
            lxlu_of_core(Val(101), 1),
            DfirOp::Dataflow(dataflow::Op::CreateGroup {
                result: Val(102),
                unit_ids: vec![Val(100)],
            }),
            DfirOp::Dataflow(dataflow::Op::CreateGroup {
                result: Val(103),
                unit_ids: vec![Val(100)],
            }),
            DfirOp::Dataflow(dataflow::Op::ProgramUnit {
                units: vec![Val(102)],
                precision: None,
                body: Vec::new(),
            }),
        ];
        let filters = UnitFilters {
            folds: Only::these(BTreeSet::from([FoldId::ZERO])),
            ..no_filters()
        };

        cleanup(&mut module, &filters);

        assert_eq!(
            vec![
                // ⭐ REBUILT: the same handle, name, type and residency, plus the attribute.
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(100),
                    residency: Residency::Corelet {
                        core: core(0),
                        corelet: corelet(0),
                    },
                    unit: DfirUnit::Lxlu,
                    num_folds: Some(NumFolds::ONE),
                }),
                // `%101` was read by nothing, `%103` by nothing; `%102` is read by the program unit.
                DfirOp::Dataflow(dataflow::Op::CreateGroup {
                    result: Val(102),
                    unit_ids: vec![Val(100)],
                }),
                DfirOp::Dataflow(dataflow::Op::ProgramUnit {
                    units: vec![Val(102)],
                    precision: None,
                    body: Vec::new(),
                }),
            ],
            module
        );
    }

    /// 🎯 205/384 — AN EMPTY TRANSFER FILTER KEEPS **NOTHING**, WHICH IS THE OPPOSITE OF THE OTHERS.
    ///
    /// ⛔ THE MISSING `!empty()` GUARD IS THE ASSERTION (`:327-337` has none). Reading this filter
    /// like the fold/core/corelet ones would make every transfer in the module "to keep" on an
    /// ordinary compile, and entry 263 would then preserve the whole program.
    #[test]
    fn only_a_named_transfer_in_a_non_empty_filter_is_kept() {
        let named = DfirOp::Dataflow(dataflow::Op::Opaque(dataflow::Opaque {
            func: OpaqueFunc::Reciprocal,
            read_write: Vec::new(),
            read_only: Vec::new(),
            params: Vec::new(),
            dbg_name: Some("copy-in".to_owned()),
        }));
        let filter = Only::these(BTreeSet::from(["copy-in".to_owned()]));

        assert!(is_data_transfer_to_keep(&named, filter.as_ref()));
        // ⛔ NO FILTER AT ALL KEEPS IT NO MORE THAN A FILTER THAT DOES NOT NAME IT.
        assert!(!is_data_transfer_to_keep(&named, None));
        // ⛔ AND A TRANSFER WITH NO `dbgName` IS NEVER KEPT — every `agen` access in this island.
        let load = DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            access: agen::Access::OfView,
            result: Val(30),
            view: Val(10),
            indices: vec![Index::Const(0)],
            view_ty: MemRef {
                shape: vec![64],
                elem: ElemType::F16,
            },
            ty: LANES,
        });
        assert!(is_data_transfer(&load) && !is_data_transfer_to_keep(&load, filter.as_ref()));
    }

    /// 🎯 262/384 — THE VENDOR'S OWN EDGE CASE: *"filtering by cores results in first region of
    /// uniformize_region to be removed"* (`core_filtering_edge_case.mlir:190-191`). Its
    /// `%788 = uniform.uniformize_regions -> index` binds two 16-unit regions, one over the odd cores
    /// and one over the even (`:987-1000`); under `filter-cores-except=0` (`:2`) the expectation is
    /// `(%[[VAL_155]] -> %[[VAL_130]]#0){ uniform.yield %[[VAL_150]] }` — ONE region, ONE unit
    /// (`%[[VAL_130]]` is `{core = 0 : i32, .., type = "lxlu"}`), and a yield of a value from OUTSIDE
    /// the region, unrenamed (`:166-170`, `:143`).
    #[test]
    fn a_core_filter_removes_a_whole_region_and_keeps_one_unit() {
        let scope: Vec<DfirOp> = (0..32)
            .map(|index| lxlu_of_core(Val(100 + index), index))
            .collect();
        let odd_cores = uniform::LocalRegion {
            arg: Val(60),
            units: (0..16).map(|k| Val(101 + 2 * k)).collect(),
            body: vec![DfirOp::Uniform(uniform::Op::Yield {
                operands: vec![Val(51)],
            })],
        };
        let even_cores = uniform::LocalRegion {
            arg: Val(61),
            units: (0..16).map(|k| Val(100 + 2 * k)).collect(),
            body: vec![DfirOp::Uniform(uniform::Op::Yield {
                operands: vec![Val(50)],
            })],
        };
        let filters = UnitFilters {
            cores: Only::these(BTreeSet::from([core(0)])),
            ..no_filters()
        };
        let mut vals = Values::default();

        let reduced = remove_cores_corelets_folds_from_uniformize_region(
            &mut vals,
            &[odd_cores, even_cores],
            &[Val(70)],
            &scope,
            &filters,
        );

        let FilteredUniformizeRegions::Rebuilt { op, replacements } = reduced else {
            panic!("core 0 survives in the second region, so the op is rebuilt");
        };
        // `getResult(0).replaceAllUsesWith(new_uniformize_op.getResult(0))`, and the result is minted
        // before any region argument — the order the op prints in.
        assert_eq!(replacements, vec![(Val(70), Val(0))]);
        assert_eq!(
            DfirOp::Uniform(uniform::Op::UniformizeRegions {
                regions: vec![uniform::LocalRegion {
                    arg: Val(1),
                    units: vec![Val(100)],
                    body: vec![DfirOp::Uniform(uniform::Op::Yield {
                        operands: vec![Val(50)],
                    })],
                }],
                results: vec![Val(0)],
            }),
            op
        );
    }

    /// 🎯 263/384 — THE VENDOR'S OWN FILTERED SEND, ANCESTORS AND ALL.
    /// `filter-transfers-except=c2-l3lu-sync-recv-lxlu0-lxlu1`
    /// (`transfer_core_fold_filtering.mlir:2`) keeps the recv's
    /// `def_immutable_mapping` → `query_map` → `dataflow.sync_recv` chain (`:155-157`) and the send's
    /// identical chain at `:818-820` is gone: the send binds nothing, so the `query_map` and then the
    /// mapping lose their only reader in turn. ⛔ AND THE CASCADE STOPS WHERE SOMETHING ELSE READS —
    /// `if (!has_uses)` keeps the `create_group` the surviving recv reads.
    #[test]
    fn a_filtered_send_takes_its_ancestors_and_stops_at_a_shared_value() {
        let group = DfirOp::Dataflow(dataflow::Op::CreateGroup {
            result: Val(430),
            unit_ids: vec![Val(300), Val(301)],
        });
        let recv = DfirOp::Dataflow(dataflow::Op::SyncRecv {
            from: Val(430),
            signal: SyncSignal::InputToLxsuToLxluToSync,
        });
        let mut module = vec![
            group.clone(),
            DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(469),
                pairs: vec![(Val(260), Val(430))],
                values_ty: MappedTy::Index,
            }),
            DfirOp::Uniform(uniform::Op::QueryMap {
                result: Val(470),
                map: Val(469),
                key: Val(5),
                ty: MappedTy::Index,
            }),
            DfirOp::Dataflow(dataflow::Op::SyncSend {
                to: Val(470),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            }),
            recv.clone(),
        ];
        let filter = Only::these(BTreeSet::from([
            "c2-l3lu-sync-recv-lxlu0-lxlu1".to_owned(),
        ]));

        remove_ancestors(&OpId::at(&[3]), &mut module, filter.as_ref());

        assert_eq!(module, vec![group, recv]);
    }
}

/// THE MAP THIS PASS PUTS IN PLACE OF A FILTERED ONE — the `DefImmutableMappingOp::create` at
/// `UnitFiltering.cpp:130-135` together with what the reference does to the old op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducedDefImmutMap {
    /// The new `uniform.def_immutable_mapping`, carrying the surviving pairs.
    pub op: DfirOp,
    /// Its handle — `immutable_map.replaceAllUsesWith(new_immutable_map)` reads every use of the old
    /// one through this.
    pub map: Val,
    /// `to_delete_.push_back(immutable_map)` — the old map's handle, erased at the end of the pass
    /// (entry 296) rather than here.
    pub to_delete: Val,
}

/// Replaces: e203_removeCoresCoreletsFoldsFromDefImmutMap
///
/// **203/384** `UnitFilteringPass::removeCoresCoreletsFoldsFromDefImmutMap` —
/// `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99` (37L).
///
/// ⛔ [`None`] IS "DO NOT MODIFY THE MAP" AND IT COVERS TWO CASES: nothing was filtered, or
/// EVERYTHING was — "If all keys are to be filtered out, do not modify the map. Instead, the entire
/// region should be deleted at the last step of this pass" (`:126-128`). ⚠️ `DT_CHECK(!values.empty())`
/// has no counterpart: an empty map reduces nothing and takes the same [`None`].
///
/// ⚠️ `values_ty` IS CARRIED THROUGH, NOT RE-DERIVED. Filtering drops pairs; it does not change what
/// the surviving values are, and the reference's `builder.getIndexType()` (`:132`) is the mapping's
/// RESULT type, which is `index` for a bitstream map too.
#[must_use]
pub fn remove_cores_corelets_folds_from_def_immut_map(
    vals: &mut Values,
    map: Val,
    pairs: &[(Val, Val)],
    values_ty: MappedTy,
    scope: &[DfirOp],
    filters: &UnitFilters,
) -> Option<ReducedDefImmutMap> {
    // `std::vector<mlir::Value> new_keys; new_values;` / `bool unit_list_is_reduced = false;` — one
    // list here, as [`uniform::Op::DefImmutableMapping`] holds them.
    let mut new_pairs: Vec<(Val, Val)> = Vec::new();
    let mut unit_list_is_reduced = false;

    // `for (int i = 0; i < values.size(); ++i)`, whose key is the unit the three clauses judge.
    for &(key, value) in pairs {
        if unit_is_filtered_out(key, scope, filters) {
            unit_list_is_reduced = true;
            continue;
        }
        new_pairs.push((key, value));
    }

    // `if (!unit_list_is_reduced || new_keys.empty()) return;`
    if !unit_list_is_reduced || new_pairs.is_empty() {
        return None;
    }

    // `DefImmutableMappingOp::create(builder, loc, builder.getIndexType(), new_keys, new_values)`,
    // then `replaceAllUsesWith` and `to_delete_.push_back(immutable_map)`.
    let new_map = vals.mint();
    Some(ReducedDefImmutMap {
        op: DfirOp::Uniform(uniform::Op::DefImmutableMapping {
            result: new_map,
            pairs: new_pairs,
            values_ty,
        }),
        map: new_map,
        to_delete: map,
    })
}

/// Replaces: e204_cleanup
///
/// **204/384** `UnitFilteringPass::cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`
/// (49L).
///
/// ⛔ THE REBUILD'S ONLY OBSERVABLE EFFECT HERE IS `num_folds`: this island's `get_unit` binds ONE
/// result ([`FoldId`]) and a non-empty fold filter always holds fold 0 (`:383-385`, entry 296), so
/// `new_num_folds` is 1 and the new op carries the same name, type, residency and handle — only the
/// attribute the reference `setAttr`s is new. ⚠️ See the body for the one attribute this cannot write.
pub fn cleanup(module: &mut Vec<DfirOp>, filters: &UnitFilters) {
    // `module_op.walk(..)`: `use_empty()` is a MODULE-WIDE question — a top-level `get_unit` is read
    // inside a `program_unit`'s region — so the erasures are planned over the whole module first.
    //
    // ⭐ WHICH IS THE SAME ANSWER THE REFERENCE'S IN-ORDER WALK GIVES. The only erasure that removes
    // a use is a `create_group`'s, and a group is defined after the units it names, so the walk has
    // already decided about those units when it reaches the group. Nothing cascades within one pass.
    let mut to_erase: BTreeSet<Val> = BTreeSet::new();
    plan_cleanup(module, module, &mut to_erase);
    apply_cleanup(module, filters, &to_erase);
}

/// WHICH BINDERS NOTHING READS — the `op->use_empty()` half of [`cleanup`], over the whole module.
fn plan_cleanup(scope: &[DfirOp], module: &[DfirOp], to_erase: &mut BTreeSet<Val>) {
    for op in scope {
        for region in regions(op) {
            plan_cleanup(region, module, to_erase);
        }

        // `isa<dataflow::CreateGroupOp, dataflow::CreateMulticastGroupOp>(op) && op->use_empty()`,
        // and the `get_unit_op.use_empty()` arm below it. ⚠️ `create_multicast_group` is not declared
        // in this island (as entry 141 records for ten `agen` classes); it belongs in this pattern
        // the day it is.
        if let DfirOp::Dataflow(
            dataflow::Op::CreateGroup { result, .. } | dataflow::Op::GetUnit { result, .. },
        ) = op
            && uses(*result, module).is_empty()
        {
            to_erase.insert(*result);
        }
    }
}

/// THE ERASURES AND THE ONE REBUILD — `op->erase()` and the `GetUnitOp::create` beside it.
fn apply_cleanup(scope: &mut Vec<DfirOp>, filters: &UnitFilters, to_erase: &BTreeSet<Val>) {
    for op in scope.iter_mut() {
        for region in regions_mut(op) {
            apply_cleanup(region, filters, to_erase);
        }
    }

    scope.retain_mut(|op| match op {
        DfirOp::Dataflow(dataflow::Op::CreateGroup { result, .. }) => !to_erase.contains(result),
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result, num_folds, ..
        }) => {
            // `if (get_unit_op.use_empty()) { erase(); return advance(); }`
            if to_erase.contains(result) {
                return false;
            }

            // `if (filter_folds_except_.empty()) return WalkResult::advance();`
            if let Some(folds) = &filters.folds {
                // `int orig_num_folds = get_unit_op.getNumResults();` — ONE here ([`FoldId`]).
                let orig_num_folds = 1;
                // "Cannot assume all elements of filter_folds_except_ are existing fold IDs."
                let new_num_folds: u32 = folds
                    .ids()
                    .filter(|fold_id| fold_id.get() < orig_num_folds)
                    .map(|_| 1)
                    .sum();

                // The rebuilt op: `getNameAttr()`/`getTypeAttr()` and `setAttr("core", core_id)` are
                // the residency this one already carries, `setAttr("num_folds", new_num_folds)` is
                // the new attribute, and the fold mapping plus `dropAllUses()` are the identity for a
                // single result whose id the filter holds.
                //
                // ⚠️ THE ONE ATTRIBUTE THIS CANNOT WRITE: the reference `setAttr`s `core` even when
                // `getCoreId` answered -1 (`:293`, an `unsigned` holding -1), so a surviving HBM
                // handle comes out of the reference carrying `core = -1 : i32`. A [`Core`] in this
                // crate is a validated non-negative id and [`Residency::Global`] prints no `core` at
                // all, so that unit is printed bare here.
                *num_folds = Some(NumFolds(new_num_folds));
            }

            true
        }
        _ => true,
    });
}

/// Replaces: e205_isDataTransferToKeep
///
/// **205/384** `UnitFilteringPass::isDataTransferToKeep` —
/// `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327` (10L).
///
/// ⛔ NO `!empty()` GUARD, WHICH INVERTS EVERY OTHER FILTER IN THIS FILE: an empty
/// `filter_transfers_except_` keeps NOTHING, where an empty fold/core/corelet set keeps everything
/// ([`Only`]). ⛔ And only `dataflow.opaque` carries a `dbgName` in this island
/// ([`dbg_name`]), so every `agen` access answers [`None`] here and no filter can keep one.
#[must_use]
pub fn is_data_transfer_to_keep(op: &DfirOp, filter_transfers: Option<&Only<String>>) -> bool {
    // `if (isDataTransfer(op))` — entry 141.
    if !is_data_transfer(op) {
        return false;
    }

    // `if (auto dbg_name = getDbgNameAttr(op))`.
    let Some(name) = dbg_name(op) else {
        return false;
    };

    // `std::find(filter_transfers_except_.begin(), .., dbg_name.str()) != .end()`.
    filter_transfers.is_some_and(|only| only.keeps(&name.to_owned()))
}

/// WHAT FILTERING LEAVES OF A `uniform.uniformize_regions` — the three exits of entry 262.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilteredUniformizeRegions {
    /// `if (!unit_list_is_reduced) return;` — no unit was filtered and the op stands as it is.
    Unchanged,
    /// `new_unit_list.empty()`: every unit went, so the op is deleted with nothing in its place
    /// (`:168-174`).
    ///
    /// ⚠️ `DT_CHECK_MSG(getResult(i).use_empty(), "Cannot filter out uniformize region with uses")`
    /// has no counterpart — this crate never runtime-refuses, and a used result here is the caller's
    /// option set being contradictory, exactly as entry 140 records for its own `DT_CHECK_MSG`.
    Deleted,
    /// The rebuilt op, whose old handles the caller redirects.
    Rebuilt {
        /// `UniformizeRegionsOp::create(builder, .., new_unit_list, new_list_sizes, .., new_num_regions)`.
        op: DfirOp,
        /// `getResult(i).replaceAllUsesWith(new_uniformize_op.getResult(i))`, old → new.
        replacements: Vec<(Val, Val)>,
    },
}

/// Replaces: e262_removeCoresCoreletsFoldsFromUniformizeRegion
///
/// **262/384** `UnitFilteringPass::removeCoresCoreletsFoldsFromUniformizeRegion` —
/// `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139` (98L).
///
/// ⛔ THE EXTRACT DROPS THE TAIL `to_delete_.push_back(uniformize_op);` (`:236`), so a REBUILD deletes
/// the old op too and not only the total-filter case. ⚠️ `regIndices`/`regLocales` (`Uniform.td:88-89`)
/// are undeclared in this island; the reference passes both through unchanged
/// (`getRegIndicesIfExist()`, `getRegLocalesIfExist()`), which is the identity here.
#[must_use]
pub fn remove_cores_corelets_folds_from_uniformize_region(
    vals: &mut Values,
    local_regions: &[uniform::LocalRegion],
    results: &[Val],
    scope: &[DfirOp],
    filters: &UnitFilters,
) -> FilteredUniformizeRegions {
    // ⭐ THE REFERENCE'S TWO FILTER LOOPS ARE ONE PASS HERE. `getUnits()` is the FLAT operand list and
    // `getRegionUnitList(i)` its region-i slice — the op's own verifier ties the two
    // ([`uniform::Op::UniformizeRegions`]) — so the flat loop that sets `unit_list_is_reduced` and the
    // per-region loop that rebuilds `new_unit_list_i` apply the same three clauses to the same units
    // in the same order. Both are [`unit_is_filtered_out`], and neither clause reads another unit.
    let mut unit_list_is_reduced = false;
    let mut new_lists: Vec<Vec<Val>> = Vec::with_capacity(local_regions.len());
    for region in local_regions {
        let mut new_unit_list_i: Vec<Val> = Vec::new();
        for &unit in &region.units {
            if unit_is_filtered_out(unit, scope, filters) {
                unit_list_is_reduced = true;
                continue;
            }
            new_unit_list_i.push(unit);
        }
        new_lists.push(new_unit_list_i);
    }

    // `if (!unit_list_is_reduced) return;`
    if !unit_list_is_reduced {
        return FilteredUniformizeRegions::Unchanged;
    }

    // `if (new_unit_list.empty())` — the flat list is empty exactly when every region's is.
    if new_lists.iter().all(Vec::is_empty) {
        return FilteredUniformizeRegions::Deleted;
    }

    // `getResultTypes()` is passed through, so the new op binds one result per old result.
    let new_results: Vec<Val> = results.iter().map(|_| vals.mint()).collect();

    let mut new_regions: Vec<uniform::LocalRegion> = Vec::with_capacity(new_lists.len());
    for (region, new_unit_list_i) in local_regions.iter().zip(new_lists) {
        // `empty_region_indices.push_back(i); --new_num_regions;` — a region whose units all went is
        // not rebuilt, and the surviving regions' own lengths ARE `new_list_sizes`.
        if new_unit_list_i.is_empty() {
            continue;
        }

        // `curr_block.addArgument(builder.getIndexType(), ..)` then `IRMapping bv_map;
        // bv_map.map(orig_block.getArguments(), curr_block.getArguments());` — ⛔ A FRESH MAPPING PER
        // REGION, because each region binds its own argument ([`uniform::LocalRegion::arg`]).
        let arg = vals.mint();
        let mut bv_map = ValueMapping::new();
        bv_map.map(region.arg, arg);

        // `for (auto &it : orig_block.getOperations()) builder.clone(it, bv_map);` — the terminator
        // included, since `getOperations()` is the whole block and [`uniform::Op::Yield`] is one of
        // them. A body operand defined outside the region comes through unchanged
        // ([`ValueMapping::lookup_or_default`]).
        let body = vals.clone_ops(&region.body, &mut bv_map);

        new_regions.push(uniform::LocalRegion {
            arg,
            units: new_unit_list_i,
            body,
        });
    }

    FilteredUniformizeRegions::Rebuilt {
        op: DfirOp::Uniform(uniform::Op::UniformizeRegions {
            regions: new_regions,
            results: new_results.clone(),
        }),
        replacements: results.iter().copied().zip(new_results).collect(),
    }
}

/// Replaces: e263_removeAncestors
///
/// **263/384** `UnitFilteringPass::removeAncestors` —
/// `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339` (18L).
///
/// ⛔ THE ERASURE CASCADES AND POSITIONS MUST NOT. The reference erases INSIDE the loop, so an
/// ancestor popped later sees its consumer already gone; a POSITION is invalidated by an earlier
/// sibling's removal ([`erase_op`](super::vc_vector_operands::erase_op)), so this walk mutates
/// nothing and erases in descending order at the end.
pub fn remove_ancestors(
    op: &OpId,
    module: &mut Vec<DfirOp>,
    filter_transfers: Option<&Only<String>>,
) {
    // `llvm::SmallVector<Operation *> worklist = {op};`
    let mut worklist: Vec<OpId> = vec![op.clone()];
    // What the reference has `erase()`d by this point in ITS walk. Nothing leaves `module` until the
    // walk is over, so every position pushed below stays the position it names.
    let mut erased: Vec<OpId> = Vec::new();

    // `while (!worklist.empty()) { Operation *oper = worklist.back(); worklist.pop_back(); .. }`
    while let Some(oper) = worklist.pop() {
        let Some(current) = op_at(&oper, module) else {
            continue;
        };

        // `for (auto operand : oper->getOperands())`
        for operand in operands(current) {
            // `Operation *operand_op = operand.getDefiningOp(); if (operand_op && ..)` — a block
            // argument has no defining op, and that is where the walk up stops.
            let Some(ancestor) = defining_position(operand, module) else {
                continue;
            };

            // `!isDataTransferToKeep(operand_op)` — entry 205. A transfer the options name by
            // `dbgName` is neither walked through nor erased.
            let keep = op_at(&ancestor, module)
                .is_some_and(|def| is_data_transfer_to_keep(def, filter_transfers));
            if keep {
                continue;
            }

            // `if (std::find(worklist.begin(), .., operand_op) == worklist.end())` — ⭐ THE DEDUP IS
            // OVER THE PENDING WORKLIST ONLY, so an op already popped IS pushed again. That is the
            // cascade: it is reconsidered now that a consumer of it has gone.
            if !worklist.contains(&ancestor) {
                worklist.push(ancestor);
            }
        }

        // `bool has_uses = ..; if (!has_uses) oper->erase();` over `getResult(i).use_empty()`.
        // ⛔ A USE BY AN OP THIS WALK HAS ALREADY ERASED IS NOT A USE — the reference gets that for
        // free by erasing as it goes, and without it a chain would stop at its first link.
        let has_uses = use_positions(&oper, module)
            .into_iter()
            .any(|user| !erased.contains(&user));
        if !has_uses && !erased.contains(&oper) {
            erased.push(oper);
        }
    }

    // ⛔ DESCENDING LEXICOGRAPHIC ORDER: removing `[3]` renumbers `[4]`, and every position a
    // removal invalidates is greater than it (see `erase_op`).
    erased.sort_unstable();
    for position in erased.iter().rev() {
        remove_at(position.path(), module);
    }
}
