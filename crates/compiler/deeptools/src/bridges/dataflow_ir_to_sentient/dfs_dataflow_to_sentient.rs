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

//! `DataflowToSentient.cpp` — 21 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e039_isSenComponentL0LU` | 039/384 | 2 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96` |
//! | `e040_isSenComponentL0SU` | 040/384 | 2 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100` |
//! | `e041_ExtendUnitNameToCorelet` | 041/384 | 11 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104` |
//! | `e042_isSameListOfUnits` | 042/384 | 10 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175` |
//! | `e043_isTargetL3` | 043/384 | 6 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720` |
//! | `e044_lowerOpaqueOperation` | 044/384 | 27 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984` |
//! | `e159_getUnitNameFromAListOfGetUnitOp` | 159/384 | 10 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119` |
//! | `e160_areCoreletsDifferent` | 160/384 | 6 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132` |
//! | `e161_separateBasedOnDestinationUnits` | 161/384 | 18 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761` |
//! | `e221_pushBackTheUnitToListIfDoesnotExist` | 221/384 | 5 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143` |
//! | `e222_createUniformRegionsWithTwoRegionsNoResult` | 222/384 | 18 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153` |
//! | `e223_lowerL3SyncOperationForAUnit` | 223/384 | 57 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375` |
//! | `e224_lowerL3SyncOperationForAGroupOfUnits` | 224/384 | 61 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667` |
//! | `e273_lowerL0LXSyncOperationForAUnit` | 273/384 | 180 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189` |
//! | `e274_lowerL0LXSyncOperationForAGroupOfUnits` | 274/384 | 222 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438` |
//! | `e300_lowerSyncForAUnit` | 300/384 | 7 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733` |
//! | `e301_lowerSyncForAGroup` | 301/384 | 8 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746` |
//! | `e302_lowerSyncLXL3ToLXL3` | 302/384 | 928 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787` |
//! | `e319_lowerSyncForAQueryMap` | 319/384 | 166 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728` |
//! | `e337_lowerSyncOperation` | 337/384 | 80 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901` |
//! | `e361_runOnOperation` | 361/384 | 33 | `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014` |

use super::vc_vector_operands::defining_position;
use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, arith, dataflow, defining_op, regions, uniform, uses,
};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::dataflow_ir::{Program, Values};
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::{Corelet, DfirUnit, Residency};

/// Replaces: e039_isSenComponentL0LU
///
/// **039/384** `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96` (2L).
///
/// ```cpp
/// static inline bool isSenComponentL0LU(SenComponents comp) {
///   return EnumsConversion::senCompToGenericComp.at(comp) == SenComponents::L0LU;
/// }
/// ```
///
/// ⭐⭐ IT IS THE **GENERIC** COMPONENT THAT IS TESTED, NOT THE SPELLING. `senCompToGenericComp`
/// (`sys-arch-spec/arch_enums.cpp:124-211`) maps every per-core, per-corelet spelling onto the one
/// image the ISA names, so this answers `true` for `L0LU` and for nothing else — and in particular
/// **not** for `L0SU`. A port that had folded the two halves together would answer `true` for both
/// and this predicate would select every L0 unit in the program.
///
/// ⛔ THE `.at()` CAN THROW AND OURS CANNOT. `L0`, `CONSTANT` and `SFPRING` are not keys of that map,
/// so the reference aborts on them; [`DfirUnit::generic`] is total, and each of the three has its own
/// image — none of which is `L0LU`, so those units answer `false` here.
#[must_use]
pub const fn is_sen_component_l0lu(unit: DfirUnit) -> bool {
    matches!(unit.generic(), GenericComp::L0lu)
}

/// Replaces: e040_isSenComponentL0SU
///
/// **040/384** `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100` (2L).
///
/// ```cpp
/// static inline bool isSenComponentL0SU(SenComponents comp) {
///   return EnumsConversion::senCompToGenericComp.at(comp) == SenComponents::L0SU;
/// }
/// ```
///
/// ⭐ THE STORE HALF, AND ONLY IT — see [`is_sen_component_l0lu`] for why the two are separate
/// images rather than one `L0`.
#[must_use]
pub const fn is_sen_component_l0su(unit: DfirUnit) -> bool {
    matches!(unit.generic(), GenericComp::L0su)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 041/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH HALF OF THE LX A SYNC NAMES — the only two components either caller extends.
///
/// ⛔⛔ TWO CASES, BECAUSE BOTH CALL SITES GUARD ON EXACTLY TWO. `ExtendUnitNameToCorelet` is reached
/// only under `(dst_comp == SenComponents::LXLU || dst_comp == SenComponents::LXSU)`
/// (`DataflowToSentient.cpp:409-411` and `:709-713`) — the already-numbered `LXLU0`/`LXSU0`/`LXLU1`/
/// `LXSU1` spellings and every L3 component take the sibling branch and are never extended. Taking a
/// whole [`sen::Consumer`] here would offer fourteen inputs the reference cannot present, and each
/// one would need a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LxHalf {
    /// `lxlu` — the LX **load** unit, `SenComponents::LXLU`.
    Load,
    /// `lxsu` — the LX **store** unit, `SenComponents::LXSU`.
    Store,
}

/// Replaces: e041_ExtendUnitNameToCorelet
///
/// **041/384** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104` (11L).
///
/// ```cpp
/// static inline LogicalResult ExtendUnitNameToCorelet(std::string &name,
///                                                     dataflow::GetUnitOp unit,
///                                                     OpBuilder builder) {
///   if (!unit->hasAttr("corelet")) {
///     unit->emitError("Unknown corelet information for sentient");
///     return LogicalResult::failure();
///   }
///   if (unit->getAttr("corelet") == builder.getI32IntegerAttr(0)) {
///     name += "0";
///   } else {
///     name += "1";
///   }
///   return LogicalResult::success();
/// }
/// ```
///
/// # ⭐⭐ THE NAME IT EXTENDS BECOMES A `SentientLoadConsumer`, WHICH IS WHY THE RESULT IS ONE
///
/// Both callers feed the extended string straight into
/// `symbolizeSentientLoadConsumer(dst_unit_name).value()` and wrap it in a
/// `SentientLoadConsumerAttr` (`:419-421` and `:715-718`). So the function's real output is not text:
/// it is the choice between `lxlu0` and `lxlu1` (or `lxsu0`/`lxsu1`) that the sync op carries. Naming
/// it [`sen::Consumer`] is what makes `.value()` — an `std::optional` unwrap that aborts on a
/// spelling the enum has no case for — unreachable.
///
/// # ⛔ ANY NON-ZERO CORELET BECOMES `1`, AND THAT IS THE REFERENCE'S OWN CHOICE
///
/// The test is `== builder.getI32IntegerAttr(0)`, with a bare `else`. There is no `lxlu2` in
/// `SentientLoadConsumer` (`SentientTypes.td:556-596`), so a third corelet could not be named even if
/// the arm existed — the vocabulary tops out at two. `SenComponents::LXLU1` is what corelet 1 gets and
/// what anything above it would get.
///
/// # ⛔⛔ AND THE ERROR ARM HAS NO INPUT HERE, BY CONSTRUCTION
///
/// The refusal is *"Unknown corelet information for sentient"* — a `get_unit` with no `corelet`
/// attribute. In this crate a unit's attributes come from [`crate::units::residency_of`], which sends
/// **both** LX halves to `Residency::Corelet { core, corelet }` unconditionally
/// (`src/units.rs:583-594`, following `UnitMaterializer.cpp:82-115`): an `lxlu` that carries no
/// corelet is not constructible. Taking a [`Corelet`] rather than a whole
/// [`crate::units::Residency`] is that guard — the three residencies without a corelet cannot be
/// passed, so the failure is a build error at the call site instead of a run-time refusal.
#[must_use]
pub const fn extend_unit_name_to_corelet(half: LxHalf, corelet: Corelet) -> sen::Consumer {
    match (half, corelet.get()) {
        (LxHalf::Load, 0) => sen::Consumer::Lxlu0,
        (LxHalf::Load, _) => sen::Consumer::Lxlu1,
        (LxHalf::Store, 0) => sen::Consumer::Lxsu0,
        (LxHalf::Store, _) => sen::Consumer::Lxsu1,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 042/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e042_isSameListOfUnits
///
/// **042/384** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175` (10L).
///
/// ```cpp
/// static bool isSameListOfUnits(
///     std::vector<mlir::Operation *> key_units,
///     std::vector<mlir::dataflow::GetUnitOp> src_unit_ops) {
///   std::vector<mlir::dataflow::GetUnitOp> key_unit_ops;
///   for (auto key : key_units) {
///     if (auto unit = llvm::dyn_cast<dataflow::GetUnitOp>(key)) {
///       key_unit_ops.push_back(unit);
///     } else {
///       return false;
///     }
///   }
///   return src_unit_ops == key_unit_ops;
/// }
/// ```
///
/// # ⭐⭐ IT COMPARES OP **IDENTITY**, NOT UNIT KIND
///
/// `std::vector<GetUnitOp> == std::vector<GetUnitOp>` compares element-wise, and an `OpState`'s
/// `operator==` is *"the same operation"* — the underlying `Operation *`. So two distinct `get_unit`
/// ops that both bind `C0-lxlu-CL0` are **not** the same list. Comparing kinds instead would answer
/// `true` for two different bindings of one unit, and the caller uses this to decide whether a
/// memoised lowering may be reused.
///
/// Here a `get_unit`'s identity is the [`Val`] it defines: values are minted once
/// ([`crate::islands::dataflow_ir::Values::mint`]) and no two ops share one, so `result` equality
/// *is* pointer equality. That is why the source side is a list of [`Val`]s and not of units.
///
/// # ⭐ LENGTH IS PART OF IT
///
/// `std::vector::operator==` compares sizes first, so a prefix is not a match. Slice equality says
/// the same.
///
/// # ⛔ THE FIRST NON-`get_unit` KEY DECIDES, AND WHAT WAS PUSHED BEFORE IT IS DISCARDED
///
/// The reference returns from inside the loop, so a key list of `[get_unit, send]` is `false` however
/// long the source list is — it never reaches the comparison.
///
/// # ⚠️ NO CALLER AT `a0d29abbed`
///
/// A grep of every `.cpp`/`.hpp`/`.h` in the authority tree finds this symbol exactly once, at its
/// own definition: the memoising caller it was written for
/// (`lowerL0LXSyncOperationForAUnit`, `:189`, entry 273) now keys its map another way. It is ported
/// anyway — the campaign's rule is that a scheduled function gets its port and its audit, and a
/// predicate the reference kept is not this port's to delete.
#[must_use]
pub fn is_same_list_of_units(key_units: &[DfirOp], src_unit_ops: &[Val]) -> bool {
    let mut key_unit_ops: Vec<Val> = Vec::with_capacity(key_units.len());
    for key in key_units {
        match key {
            // `dyn_cast<dataflow::GetUnitOp>(key)` succeeded.
            DfirOp::Dataflow(dataflow::Op::GetUnit { result, .. }) => key_unit_ops.push(*result),
            // ⛔ IT DID NOT — and the dialects are spelled out rather than wildcarded so that a new
            // op cannot silently join the `false` side without being looked at.
            DfirOp::Dataflow(_)
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Agen(_)
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ `uniform` JOINS THE `false` SIDE: a `uniformize_regions` binds its own results and a
            // `query_map` binds an `index`, so neither is the `get_unit` this `dyn_cast` wants.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => return false,
        }
    }
    src_unit_ops == key_unit_ops.as_slice()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 043/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e043_isTargetL3
///
/// **043/384** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720` (6L).
///
/// ```cpp
/// static bool isTargetL3(uniform::QueryMapOp query_map) {
///   auto unit_type =
///       dcc::uniform::utils::getUnitTypeFromUniformMappingAsString(query_map);
///   if (unit_type.has_value())
///     return (unit_type.value().substr(0, 2) == "l3" ? true : false);
///   return true;
/// }
/// ```
///
/// # ⭐⭐ ONLY THE **FIRST** QUERIED VALUE IS READ
///
/// `getUnitTypeFromUniformMappingAsString` (`dcc/src/Dialect/Uniform/Utils.cpp:258-284`, itself
/// outside the 384) takes `def_map_op.getValues()[0]` and returns that one unit's `type` (for a
/// `get_unit`) or its `name` (for a `get_local_unit`), lower-cased. It never looks at the rest — a
/// query naming `[l3lu, lxlu]` answers on the `l3lu` alone. Hence [`slice::first`] and not `all` or
/// `any`.
///
/// # ⛔⛔ AND ABSENT MEANS **TRUE**, WHICH IS THE OPPOSITE DEFAULT FROM THE ONE IT LOOKS LIKE
///
/// `std::nullopt` — no `DefImmutableMappingOp` behind the map, or a mapping with no values at all —
/// returns `true`, i.e. *treat the target as L3*. In the caller (`:1838`) that picks the single merged
/// `lowerSyncLXL3ToLXL3(..., -1, false)` over the two-region `uniform::UniformizeRegionsOp` split, so
/// defaulting the other way would emit two per-corelet regions for a sync the reference emits once.
/// An empty list here is that case.
///
/// # ⛔ THE THIRD OUTCOME OF THE C++ HELPER IS UNREPRESENTABLE HERE, AND THAT IS THE GUARD
///
/// If `values[0]` is defined by neither a `get_unit` nor a `get_local_unit`, the helper falls out of
/// both branches and returns a **present but empty** string — so `substr(0, 2)` is `""` and the answer
/// is `false`, not the `true` of the absent case. Two very different defaults, told apart by whether
/// the string exists. A query map here names units by type ([`DfirUnit`]), so "a queried value that is
/// not a unit" has no spelling; the two cases that remain are the two this function distinguishes.
///
/// # ⛔ `l3` IS A PREFIX TEST OVER THE LOWER-CASED SPELLING, AND ONLY THE TWO L3 HALVES PASS IT
///
/// `l3lu` and `l3su` are the only unit spellings in the vocabulary beginning `l3` — `l0`, `lx`,
/// `lxlu`, `lxsu` and `lxvirtualibr` all fail it, and so does every `get_local_unit` name. Matching
/// the variants states that without putting a string comparison in the compiler.
#[must_use]
pub fn is_target_l3(queried_units: &[DfirUnit]) -> bool {
    match queried_units.first() {
        // `!def_map_op` or `getValues().empty()` — `std::nullopt`, and the default is `true`.
        None => true,
        // `unit_type.value().substr(0, 2) == "l3"`.
        Some(unit) => matches!(unit, DfirUnit::L3lu | DfirUnit::L3su),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 044/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e044_lowerOpaqueOperation
///
/// **044/384** `DataflowToSentientLoweringPass::lowerOpaqueOperation` —
/// `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984` (27L).
///
/// ```cpp
/// LogicalResult DataflowToSentientLoweringPass::lowerOpaqueOperation(
///     dataflow::OpaqueOp opaque_op) {
///   for (auto itr : opaque_op.getReadWriteRegisterDictionary()) {
///     if (!mlir::dyn_cast<StringAttr>(itr.getValue())) {
///       opaque_op->emitError("Registers parameters should have values.");
///       return LogicalResult::failure();
///     }
///   }
///   for (auto itr : opaque_op.getReadOnlyRegisterDictionary()) {
///     if (!mlir::dyn_cast<StringAttr>(itr.getValue())) {
///       opaque_op->emitError("Registers parameters should have values.");
///       return LogicalResult::failure();
///     }
///   }
///   for (auto itr : opaque_op.getParameterDictionary()) {
///     if (!mlir::dyn_cast<StringAttr>(itr.getValue())) {
///       opaque_op->emitError("Parameters should have values.");
///       return LogicalResult::failure();
///     }
///   }
///   OpBuilder builder(opaque_op);
///   auto dofunc = opaque_op.getFuncName();
///   StringAttr dbg_name_attr = getDbgNameAttr(opaque_op);
///   sentient::OpaqueOp::create(builder, opaque_op->getLoc(), dbg_name_attr,
///                              dofunc, opaque_op.getReadWriteRegisterDictionary(),
///                              opaque_op.getReadOnlyRegisterDictionary(),
///                              opaque_op.getParameterDictionary());
///   return LogicalResult::success();
/// }
/// ```
///
/// # ⭐⭐ EVERY FIELD CROSSES UNCHANGED, INCLUDING THE ORDER WITHIN EACH DICTIONARY
///
/// The three dictionaries are handed over as whole `DictionaryAttr`s, so the lowered op's registers
/// and parameters are byte-for-byte the ones the rung below wrote. IBM's own answer key is one line
/// (`dcc/test/Conversion/DataflowToSentient/opaque.mlir:18`):
///
/// ```text
/// sentient.opaque {dbgName = "opaque_op #1", func_name = "reciprocal", parameter_dictionary = {a = "A", b = "B", c = "C"}, read_only_register_dictionary = {}, read_write_register_dictionary = {P0 = "R0", P1 = "R1"}}
/// ```
///
/// from the input `dataflow.opaque {dbgName="opaque_op #1", func_name= "reciprocal",
/// read_write_register_dictionary = {"P0" = "R0", "P1" = "R1"}, read_only_register_dictionary = {},
/// parameter_dictionary = {"a" = "A", "b" = "B","c" = "C"}}` (`:32`). Note that the EMPTY dictionary
/// is still printed, and that `read_only`/`read_write` do not swap.
///
/// ⚠️ THAT ANSWER KEY IS WHY THE PRINTER CHANGED IN THIS CHANGESET. `sentient.opaque` was rendering
/// `func_name = "RECIPROCAL"` (the generated enum's own spelling) and `{P0 = "0"}` (the register
/// address with no `R`), neither of which is what the reference forwards. Both are fixed in
/// [`crate::islands::sentient::dialects::sentient`]; the dictionaries are also key-sorted there now,
/// as MLIR stores them and as the rung below already printed them.
///
/// # ⛔⛔ ALL THREE VALIDATION LOOPS HAVE NO INPUT, BY CONSTRUCTION
///
/// Each loop asks only whether a dictionary's value is a `StringAttr` — the reference's dictionaries
/// are `DictionaryAttr`s that could hold an integer, an array or a nested dictionary. Ours cannot: a
/// register binds to a [`dataflow::RegAddr`] and a parameter to a
/// [`crate::generated::ParamValue`], both by type. So *"Registers parameters should have values."* and
/// *"Parameters should have values."* are unreachable rather than unchecked — and the newtype exists
/// **because** of this check: an empty `String` once satisfied it and then substituted an empty
/// operand into the instruction (see [`dataflow::RegAddr`]).
///
/// # ⛔ WHAT THE PORT DROPS
///
/// `OpBuilder builder(opaque_op)` (`:2004`) positions the insertion point AT the `dataflow.opaque`,
/// and the `dataflow.opaque` is erased afterwards via `to_be_deleted`. Placement is the one mechanism
/// this campaign's ports may drop — the op itself is what this function decides.
///
/// ⚠️ THE CALLER'S OWN `setInsertionPointToStart` IS DEAD, AND AN EARLIER NOTE HERE SAID OTHERWISE.
/// `runOnOperation` (`:2028-2029`, entry 361) builds `OpBuilder builder(unit_op)` and moves it to the
/// front of the unit's region — but this function takes the op alone and makes the builder above, so
/// nothing carries that insertion point in. The `sentient.opaque` lands where the `dataflow.opaque`
/// was, not at the top of the region.
#[must_use]
pub fn lower_opaque_operation(opaque: &dataflow::Opaque) -> sen::Op {
    sen::Op::Opaque {
        // `opaque_op.getFuncName()`.
        func: opaque.func,
        // `opaque_op.getReadWriteRegisterDictionary()` — ⛔ FIRST OF THE TWO, and the reference
        // passes read-write before read-only (`:2007-2009`).
        read_write: opaque.read_write.clone(),
        // `opaque_op.getReadOnlyRegisterDictionary()`.
        read_only: opaque.read_only.clone(),
        // `opaque_op.getParameterDictionary()`.
        params: opaque.params.clone(),
        // `getDbgNameAttr(opaque_op)` — absent stays absent: `getDbgNameAttr` returns a null
        // `StringAttr` when the op has no `dbgName`, and `sentient::OpaqueOp::create` takes it as the
        // optional attribute it is declared to be.
        dbg_name: opaque.dbg_name.clone(),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::generated::{OpaqueFunc, ParamKey, ParamValue, RegName};
    use crate::islands::dataflow_ir::dialects::dataflow::RegAddr;
    use crate::islands::sentient::dialects::Op as SenOp;
    use crate::units::{Core, Row};

    /// 🎯 039/384 + 040/384 — THE LOAD HALF AND THE STORE HALF ARE TOLD APART.
    ///
    /// ⛔ THE POINT OF THE PAIR. The two predicates exist to route a sync onto one half of the L0, so
    /// a mapping that collapsed `l0lu` and `l0su` onto one generic component would make both answer
    /// `true` for both units and every L0 sync would be emitted twice.
    #[test]
    fn the_l0_halves_are_distinct_generic_components() {
        assert!(is_sen_component_l0lu(DfirUnit::L0lu));
        assert!(!is_sen_component_l0su(DfirUnit::L0lu));

        assert!(is_sen_component_l0su(DfirUnit::L0su));
        assert!(!is_sen_component_l0lu(DfirUnit::L0su));
    }

    /// 🎯 039/384 + 040/384 — AND NO OTHER UNIT IS AN L0 HALF.
    ///
    /// ⛔ INCLUDING THE THREE THE REFERENCE'S MAP HAS NO KEY FOR. `senCompToGenericComp.at(L0)`,
    /// `.at(CONSTANT)` and `.at(SFPRING)` throw (`arch_enums.cpp:124-211` has no entry for them);
    /// ours answer `false`, which is the routing decision those units need.
    #[test]
    fn nothing_else_is_an_l0_half() {
        for unit in [
            DfirUnit::PtRow(Row::checked(0).expect("row 0 exists on every arch")),
            DfirUnit::Pe,
            DfirUnit::Sfp,
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::Lx,
            DfirUnit::L3lu,
            DfirUnit::L3su,
            DfirUnit::Hbm,
            DfirUnit::CrossPtnLink,
            DfirUnit::SfpState,
            DfirUnit::PeState,
            DfirUnit::L0,
            DfirUnit::Constant,
            DfirUnit::SfpRing,
        ] {
            assert!(
                !is_sen_component_l0lu(unit),
                "{unit:?} is not the L0 load unit"
            );
            assert!(
                !is_sen_component_l0su(unit),
                "{unit:?} is not the L0 store unit"
            );
        }
    }
    /// Corelet 0 of this build's arch.
    fn corelet0() -> Corelet {
        Corelet::checked(0).expect("every arch has corelet 0")
    }

    /// Corelet 1 of this build's arch.
    fn corelet1() -> Corelet {
        Corelet::checked(1).expect("every arch this crate builds for has two corelets")
    }

    /// 🎯 041/384 — THE CORELET PICKS THE NUMBERED CONSUMER, AND THE HALVES DO NOT CROSS.
    ///
    /// ⛔ `lxlu` + corelet 1 IS `lxlu1`, which is the reference's own worked case: `type = "lxlu"`
    /// with `corelet = 1` becomes `SentientLoadConsumer::lxlu1` (`DataflowToSentient.cpp:104-117`
    /// feeding `symbolizeSentientLoadConsumer` at `:419-421`). Extending the STORE half's name with
    /// the LOAD half's number would name a unit that is not on the other end of the sync.
    #[test]
    fn the_corelet_numbers_the_lx_consumer() {
        assert_eq!(
            extend_unit_name_to_corelet(LxHalf::Load, corelet0()),
            sen::Consumer::Lxlu0
        );
        assert_eq!(
            extend_unit_name_to_corelet(LxHalf::Load, corelet1()),
            sen::Consumer::Lxlu1
        );
        assert_eq!(
            extend_unit_name_to_corelet(LxHalf::Store, corelet0()),
            sen::Consumer::Lxsu0
        );
        assert_eq!(
            extend_unit_name_to_corelet(LxHalf::Store, corelet1()),
            sen::Consumer::Lxsu1
        );
    }

    /// 🎯 041/384 — AND THE EXTENDED NAME IS THE SPELLING `symbolizeSentientLoadConsumer` TAKES.
    ///
    /// ⭐ THE REFERENCE APPENDS A DIGIT TO A NAME AND THEN LOOKS THE WHOLE STRING UP. So the port is
    /// only right if the consumer it returns spells `"lxlu"` + `"0"`; a variant whose spelling were
    /// `lxlu_0` would round-trip through nothing.
    #[test]
    fn the_numbered_consumer_spells_the_extended_name() {
        for (half, corelet, spelling) in [
            (LxHalf::Load, corelet0(), "lxlu0"),
            (LxHalf::Load, corelet1(), "lxlu1"),
            (LxHalf::Store, corelet0(), "lxsu0"),
            (LxHalf::Store, corelet1(), "lxsu1"),
        ] {
            assert_eq!(
                extend_unit_name_to_corelet(half, corelet).spelling(),
                spelling
            );
        }
    }

    /// A `dataflow.get_unit` binding `unit` on core 0, corelet 0.
    fn get_unit(result: u32, unit: DfirUnit) -> DfirOp {
        let core = Core::checked(0).expect("every arch has core 0");
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency: crate::units::residency_of(unit, core, corelet0()),
            unit,
            num_folds: None,
        })
    }

    /// 🎯 042/384 — THE SAME OPS IN THE SAME ORDER, AND NOTHING ELSE IS THE SAME LIST.
    ///
    /// ⛔ IDENTITY, NOT KIND. The last case is two DIFFERENT bindings of the same two units: the
    /// reference compares `Operation *`s, so that is `false`. A port that compared unit kinds would
    /// reuse a memoised lowering keyed on somebody else's `get_unit`.
    #[test]
    fn the_same_list_of_units_is_the_same_ops() {
        let keys = vec![get_unit(3, DfirUnit::Lxlu), get_unit(4, DfirUnit::L3lu)];

        assert!(is_same_list_of_units(&keys, &[Val(3), Val(4)]));
        // Order matters.
        assert!(!is_same_list_of_units(&keys, &[Val(4), Val(3)]));
        // Length matters — a prefix is not a match.
        assert!(!is_same_list_of_units(&keys, &[Val(3)]));
        assert!(!is_same_list_of_units(&keys, &[Val(3), Val(4), Val(5)]));
        // Different bindings of the same units are different ops.
        assert!(!is_same_list_of_units(&keys, &[Val(7), Val(8)]));
    }

    /// 🎯 042/384 — A KEY THAT IS NOT A `get_unit` DECIDES ON ITS OWN.
    ///
    /// ⛔ AND IT DECIDES EVEN THOUGH THE `get_unit`s BEFORE IT MATCHED. The reference returns from
    /// inside the loop (`DataflowToSentient.cpp:180-184`), so the comparison never runs.
    #[test]
    fn a_key_that_is_not_a_get_unit_refuses_the_whole_list() {
        let keys = vec![
            get_unit(3, DfirUnit::Lxlu),
            DfirOp::Dataflow(dataflow::Op::SyncSend {
                to: Val(3),
                signal: crate::generated::SyncSignal::InputToLxsuToLxluToSync,
                wait: dataflow::AsyncTransferWait::Immediately,
            }),
        ];
        assert!(!is_same_list_of_units(&keys, &[Val(3), Val(4)]));
        // Not even against the one value it did collect.
        assert!(!is_same_list_of_units(&keys, &[Val(3)]));
    }

    /// 🎯 042/384 — TWO EMPTY LISTS ARE THE SAME LIST.
    ///
    /// ⭐ THE LOOP BODY NEVER RUNS AND `{} == {}`. No arm of the reference excludes it.
    #[test]
    fn two_empty_lists_are_the_same_list() {
        assert!(is_same_list_of_units(&[], &[]));
        assert!(!is_same_list_of_units(&[], &[Val(0)]));
    }

    /// 🎯 043/384 — ONLY THE FIRST QUERIED UNIT IS READ.
    ///
    /// ⛔ `getValues()[0]` (`dcc/src/Dialect/Uniform/Utils.cpp:268`). A query naming an L3 half first
    /// is an L3 target however the rest of the list reads, and one naming an LX half first is not —
    /// which is the difference between one merged `lowerSyncLXL3ToLXL3(..., -1, false)` and a
    /// two-region uniformize split (`DataflowToSentient.cpp:1838-1860`).
    #[test]
    fn only_the_first_queried_unit_decides_the_l3_target() {
        assert!(is_target_l3(&[DfirUnit::L3lu]));
        assert!(is_target_l3(&[DfirUnit::L3su]));
        assert!(is_target_l3(&[DfirUnit::L3lu, DfirUnit::Lxlu]));
        assert!(!is_target_l3(&[DfirUnit::Lxlu, DfirUnit::L3lu]));
    }

    /// 🎯 043/384 — AND NO QUERIED UNIT AT ALL IS **TRUE**.
    ///
    /// ⛔⛔ THE DEFAULT IS THE L3 SIDE. `std::nullopt` — no `DefImmutableMappingOp`, or a mapping with
    /// no values — reaches `return true` (`DataflowToSentient.cpp:1725`). Defaulting to `false` would
    /// split a sync into two corelet regions the reference emits as one.
    #[test]
    fn a_query_naming_nothing_is_an_l3_target() {
        assert!(is_target_l3(&[]));
    }

    /// 🎯 043/384 — AND `l3` IS A PREFIX NO OTHER UNIT SPELLING HAS.
    ///
    /// ⛔ `l0` AND `lx` BOTH BEGIN WITH `l`. The reference tests `substr(0, 2) == "l3"`, so the L0 and
    /// LX halves — the units this arm exists to tell the L3 apart from — are not L3 targets.
    #[test]
    fn no_other_unit_spelling_begins_l3() {
        for unit in [
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::Lx,
            DfirUnit::L0lu,
            DfirUnit::L0su,
            DfirUnit::L0,
            DfirUnit::LxVirtualIbr,
            DfirUnit::Hbm,
            DfirUnit::Pe,
            DfirUnit::Sfp,
        ] {
            assert!(
                !is_target_l3(&[unit]),
                "{unit:?} does not spell an l3 unit type"
            );
            assert!(
                !unit.spelling().starts_with("l3"),
                "{unit:?} must also fail the reference's own prefix test"
            );
        }
        for unit in [DfirUnit::L3lu, DfirUnit::L3su] {
            assert!(unit.spelling().starts_with("l3"));
        }
    }

    /// IBM's own opaque body, typed — `dcc/test/Conversion/DataflowToSentient/opaque.mlir:32`.
    ///
    /// ⚠️ WITH THIS CRATE'S OWN VOCABULARY, NOT THE TEST FILE'S STRINGS. `func_name = "reciprocal"` is
    /// [`OpaqueFunc::Reciprocal`]; the vendored file's `P0`/`a`/`A` are hand-written names that no
    /// template in this crate's census declares, so the registers and parameters here are real
    /// [`RegName`]/[`ParamKey`]/[`ParamValue`] cases. What is under test is the FORWARDING, and the
    /// shape — two read-write registers, an empty read-only dictionary, three parameters, a `dbgName`
    /// — is the vendored one.
    fn ibms_opaque() -> dataflow::Opaque {
        dataflow::Opaque {
            func: OpaqueFunc::Reciprocal,
            read_write: vec![(RegName::A00, RegAddr(0)), (RegName::A01, RegAddr(1))],
            read_only: Vec::new(),
            params: vec![
                (ParamKey::Prec, ParamValue::Fp16),
                (ParamKey::Unroll, ParamValue::N4),
                (ParamKey::Out0, ParamValue::Result),
            ],
            dbg_name: Some("opaque_op #1".to_owned()),
        }
    }

    /// 🎯 044/384 — EVERY FIELD CROSSES THE RUNG UNCHANGED.
    ///
    /// ⛔ INCLUDING THE EMPTY DICTIONARY AND THE `dbgName`. The reference hands all three
    /// `DictionaryAttr`s and `getDbgNameAttr(opaque_op)` to `sentient::OpaqueOp::create`
    /// (`DataflowToSentient.cpp:2005-2010`); dropping the empty one would change the op's attribute
    /// set, and dropping the name loses the only handle a debugger has on a spliced `.smc` body.
    #[test]
    fn the_opaque_body_crosses_the_rung_unchanged() {
        let dfir = ibms_opaque();
        assert_eq!(
            lower_opaque_operation(&dfir),
            sen::Op::Opaque {
                func: OpaqueFunc::Reciprocal,
                read_write: vec![(RegName::A00, RegAddr(0)), (RegName::A01, RegAddr(1))],
                read_only: Vec::new(),
                params: vec![
                    (ParamKey::Prec, ParamValue::Fp16),
                    (ParamKey::Unroll, ParamValue::N4),
                    (ParamKey::Out0, ParamValue::Result),
                ],
                dbg_name: Some("opaque_op #1".to_owned()),
            }
        );
    }

    /// 🎯 044/384 — AND THE TWO DICTIONARIES DO NOT SWAP.
    ///
    /// ⛔⛔ THE ARGUMENT ORDER IS `read_write` THEN `read_only` (`:2007-2008`), and the two mean
    /// opposite things: `read_write` is the body's INTERNAL scratch, `read_only` the caller-bound
    /// input/output registers (`ddcv1.cpp:3369-3391`). A port that crossed them would bind a kernel's
    /// scratch registers to its caller's operands.
    #[test]
    fn the_register_dictionaries_do_not_swap() {
        let dfir = dataflow::Opaque {
            func: OpaqueFunc::Exp,
            read_write: vec![(RegName::A00, RegAddr(0))],
            read_only: vec![(RegName::A01, RegAddr(8))],
            params: Vec::new(),
            dbg_name: None,
        };
        let sen::Op::Opaque {
            read_write,
            read_only,
            dbg_name,
            ..
        } = lower_opaque_operation(&dfir)
        else {
            panic!("lowering an opaque yields an opaque");
        };
        assert_eq!(read_write, vec![(RegName::A00, RegAddr(0))]);
        assert_eq!(read_only, vec![(RegName::A01, RegAddr(8))]);
        // ⭐ ABSENT STAYS ABSENT — `getDbgNameAttr` returns a null attribute for an op without one.
        assert_eq!(dbg_name, None);
    }

    /// 🎯 044/384 — AND THE LOWERED OP PRINTS AS IBM'S ANSWER KEY WRITES IT.
    ///
    /// ⛔⛔ THIS IS THE TEST THAT FOUND THE PRINTER DEFECTS. The expectation is
    /// `dcc/test/Conversion/DataflowToSentient/opaque.mlir:18` — key-sorted attributes, a lower-cased
    /// `func_name`, and register addresses carrying the `R` that
    /// `dcc/src/Dialect/Sentient/Utils.cpp:157` strips back off by position. `sentient.opaque` was
    /// printing `func_name = "RECIPROCAL"` and `{a0_0 = "0"}`, which is the wrong port, silently.
    #[test]
    fn the_lowered_opaque_prints_as_the_reference_writes_it() {
        let mut out = String::new();
        crate::islands::sentient::print::emit(
            &mut out,
            &SenOp::Sentient(lower_opaque_operation(&ibms_opaque())),
            0,
        );
        assert_eq!(
            out.trim(),
            "sentient.opaque {dbgName = \"opaque_op #1\", func_name = \"reciprocal\", \
             parameter_dictionary = {out0 = \"result\", prec = \"fp16\", unroll = \"4\"}, \
             read_only_register_dictionary = {}, \
             read_write_register_dictionary = {a0_0 = \"R0\", a0_1 = \"R1\"}}"
        );
    }

    /// 🎯 044/384 — AND THE RUNG BELOW PRINTS THE SAME BODY, WHICH IS WHERE IT CAME FROM.
    ///
    /// ⭐ THE INPUT SIDE OF THE ANSWER KEY (`opaque.mlir:32`). `dataflow.opaque` gained its `dbgName`
    /// in this changeset precisely so that [`lower_opaque_operation`] has one to forward.
    #[test]
    fn the_dataflow_opaque_prints_its_debug_name() {
        let mut out = String::new();
        crate::islands::dataflow_ir::print::emit(
            &mut out,
            &DfirOp::Dataflow(dataflow::Op::Opaque(ibms_opaque())),
            0,
        );
        assert_eq!(
            out.trim(),
            "dataflow.opaque {dbgName = \"opaque_op #1\", func_name = \"reciprocal\", \
             parameter_dictionary = {out0 = \"result\", prec = \"fp16\", unroll = \"4\"}, \
             read_only_register_dictionary = {}, \
             read_write_register_dictionary = {a0_0 = \"R0\", a0_1 = \"R1\"}}"
        );
    }

    /// 🎯 159/384 — ONE KIND ON THE LIST IS THE ANSWER, AND TWO KINDS ARE NO ANSWER.
    ///
    /// ⛔ THE MIXED LIST IS THE CASE THE FUNCTION EXISTS FOR. `"Src unit types has to be the same."`
    /// (`DataflowToSentient.cpp:124`) — a sync whose sources straddle an `lxlu` and an `l3lu` has no
    /// single generic component to route on, so there is nothing to hand
    /// `stringToSenComponents.find(...)` and the answer must be absent rather than one of the two.
    #[test]
    fn one_unit_kind_across_the_list_is_the_name() {
        assert_eq!(
            unit_name_from_a_list_of_get_unit_op(&[DfirUnit::Lxlu]),
            Some(DfirUnit::Lxlu)
        );
        assert_eq!(
            unit_name_from_a_list_of_get_unit_op(&[DfirUnit::Lxlu, DfirUnit::Lxlu, DfirUnit::Lxlu]),
            Some(DfirUnit::Lxlu)
        );
        assert_eq!(
            unit_name_from_a_list_of_get_unit_op(&[DfirUnit::Lxlu, DfirUnit::L3lu]),
            None
        );
        // And the mismatch is refused wherever on the list it sits, not just next to the head.
        assert_eq!(
            unit_name_from_a_list_of_get_unit_op(&[DfirUnit::L3su, DfirUnit::L3su, DfirUnit::Lxsu]),
            None
        );
    }

    /// 🎯 159/384 — AND THE EMPTY LIST IS THE SAME ABSENCE AS THE MISMATCH.
    ///
    /// ⛔⛔ BOTH OF THE REFERENCE'S FAILURES COLLAPSE HERE. `DT_CHECK(units.size() > 0)` aborts and
    /// the mismatch returns `""`, and `""` is fed to `find(...)->second` — see the item's own note.
    /// A port that answered `Some(_)` for the empty list would have to invent a unit kind.
    #[test]
    fn no_units_is_no_name() {
        assert_eq!(unit_name_from_a_list_of_get_unit_op(&[]), None);
    }

    /// 🎯 159/384 — TWO CORELETS OF ONE KIND ARE ONE NAME.
    ///
    /// ⛔ BECAUSE `getType()` READS THE `type` ATTRIBUTE AND NOT THE NAME. `C0-lxlu-CL0` and
    /// `C0-lxlu-CL1` are two `get_unit`s with `type = "lxlu"`, and the reference accepts them as one
    /// name; the corelet split is [`are_corelets_different`]'s and
    /// [`separate_based_on_destination_units`]'s job, not this one's. A port that keyed on the
    /// PRINTED NAME would refuse every real two-corelet sync.
    #[test]
    fn the_two_corelets_of_one_kind_share_a_name() {
        let core = Core::checked(0).expect("every arch has core 0");
        // The two bindings this list stands for, spelled out to show they differ only in corelet.
        assert_ne!(
            crate::units::residency_of(DfirUnit::Lxlu, core, corelet0()),
            crate::units::residency_of(DfirUnit::Lxlu, core, corelet1())
        );
        assert_eq!(
            unit_name_from_a_list_of_get_unit_op(&[DfirUnit::Lxlu, DfirUnit::Lxlu]),
            Some(DfirUnit::Lxlu)
        );
    }

    /// 🎯 160/384 — THE SUBJECT IS THE UNIT, AND IT IS ASKED ABOUT THE **OTHER** CORELET.
    ///
    /// ⛔ NOT WHETHER THE TWO LISTS DIFFER. A corelet-0 unit whose peers are all on corelet 0
    /// answers `false`; the same unit with a peer on corelet 1 answers `true`. Reading the predicate
    /// as "are these two lists different" would answer `true` for the first case and split a region
    /// the reference keeps whole.
    #[test]
    fn a_unit_is_different_from_the_corelet_it_is_not_on() {
        let core = Core::checked(0).expect("every arch has core 0");
        let on_0 = crate::units::residency_of(DfirUnit::Lxlu, core, corelet0());
        let on_1 = crate::units::residency_of(DfirUnit::Lxlu, core, corelet1());

        assert!(!are_corelets_different(on_0, OccupiedCorelets::Corelet0));
        assert!(are_corelets_different(on_0, OccupiedCorelets::Corelet1));
        assert!(are_corelets_different(on_0, OccupiedCorelets::Both));

        assert!(are_corelets_different(on_1, OccupiedCorelets::Corelet0));
        assert!(!are_corelets_different(on_1, OccupiedCorelets::Corelet1));
        assert!(are_corelets_different(on_1, OccupiedCorelets::Both));
    }

    /// 🎯 160/384 — A UNIT WITH NO `corelet` ATTRIBUTE IS DIFFERENT FROM NOBODY.
    ///
    /// ⛔⛔ THE NULL ATTRIBUTE EQUALS NEITHER LITERAL. `getAttr("corelet")` is null for the LX
    /// scratchpad and the HBM, so both disjuncts fail and the answer is `false` whatever the lists
    /// hold — including `Both`, where a port that treated "absent" as "corelet 1" would say `true`.
    #[test]
    fn a_unit_with_no_corelet_attribute_is_never_different() {
        for occupied in [
            OccupiedCorelets::Corelet0,
            OccupiedCorelets::Corelet1,
            OccupiedCorelets::Both,
        ] {
            for residency in [
                Residency::Global,
                Residency::Scratchpad {
                    core: Core::checked(0).expect("every arch has core 0"),
                },
            ] {
                assert!(
                    !are_corelets_different(residency, occupied),
                    "{residency:?} carries no corelet attribute"
                );
            }
        }
    }

    /// 🎯 160/384 — AND AN L3 HALF **IS** ON CORELET 0.
    ///
    /// ⛔⛔ `CoreWide` PRINTS `corelet = 0 : i32`. `C0-l3lu` is bound with `core` AND `corelet = 0`
    /// while `C0-lx` is bound with `core` alone, so `getAttr("corelet")` answers the literal `0` for
    /// an L3 half and null for the scratchpad. Folding the two residencies together — they are both
    /// "not per corelet" — would make the L3 side answer `false` here.
    #[test]
    fn the_l3_halves_answer_as_corelet_zero() {
        let core = Core::checked(0).expect("every arch has core 0");
        let l3 = crate::units::residency_of(DfirUnit::L3lu, core, corelet0());
        assert_eq!(l3, Residency::CoreWide { core });

        assert!(!are_corelets_different(l3, OccupiedCorelets::Corelet0));
        assert!(are_corelets_different(l3, OccupiedCorelets::Corelet1));
    }

    /// 🎯 160/384 — AND THE PAIR THE `DT_CHECK` RULES OUT IS UNSPELLABLE.
    ///
    /// ⛔ `DT_CHECK(units_with_corelet_0.size() > 0 || units_with_corelet_1.size() > 0)`
    /// (`DataflowToSentient.cpp:137-138`) — two empty lists abort, so the domain is three cases and
    /// [`OccupiedCorelets::of`] is the one place that says so. This is a type guard standing in for a
    /// runtime abort, which is why there is no fourth variant to test.
    #[test]
    fn two_empty_lists_mint_no_occupancy() {
        assert_eq!(OccupiedCorelets::of(&[], &[]), None);
        assert_eq!(
            OccupiedCorelets::of(&[Val(1)], &[]),
            Some(OccupiedCorelets::Corelet0)
        );
        assert_eq!(
            OccupiedCorelets::of(&[], &[Val(2)]),
            Some(OccupiedCorelets::Corelet1)
        );
        assert_eq!(
            OccupiedCorelets::of(&[Val(1)], &[Val(2)]),
            Some(OccupiedCorelets::Both)
        );
    }

    /// A `dataflow.get_unit` binding `unit` on core 0 and the given corelet.
    fn get_unit_on(result: u32, unit: DfirUnit, corelet: Corelet) -> DfirOp {
        let core = Core::checked(0).expect("every arch has core 0");
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(result),
            residency: crate::units::residency_of(unit, core, corelet),
            unit,
            num_folds: None,
        })
    }

    /// 🎯 161/384 — THE FOUR BUCKETS, EACH REACHED BY A DESTINATION THAT BELONGS IN IT.
    ///
    /// ⛔ AND THE `create_group` BUCKET IS ONE OF THEM. `src_dst_group` is the reason
    /// [`dataflow::Op::CreateGroup`] exists in the island at all: without it that list could never be
    /// non-empty and `lowerSyncLXL3ToLXL3`'s collective arm would be dead.
    #[test]
    fn each_destination_kind_reaches_its_own_bucket() {
        let scope = vec![
            get_unit_on(10, DfirUnit::Lxlu, corelet0()),
            get_unit_on(11, DfirUnit::Lxlu, corelet1()),
            get_unit_on(12, DfirUnit::L3lu, corelet0()),
            DfirOp::Dataflow(dataflow::Op::CreateGroup {
                result: Val(13),
                unit_ids: vec![Val(10), Val(11)],
            }),
        ];
        let separated = separate_based_on_destination_units(
            &[
                (Val(0), Val(10)),
                (Val(1), Val(11)),
                (Val(2), Val(12)),
                (Val(3), Val(13)),
            ],
            &scope,
        );
        assert_eq!(
            separated,
            SeparatedDestinations {
                lx_corelet0: vec![(Val(0), Val(10))],
                lx_corelet1: vec![(Val(1), Val(11))],
                l3: vec![(Val(2), Val(12))],
                group: vec![(Val(3), Val(13))],
            }
        );
    }

    /// 🎯 161/384 — AN L3 DESTINATION GOES TO THE L3 LIST EVEN THOUGH ITS `corelet` IS `0`.
    ///
    /// ⛔⛔ THE PREFIX TEST COMES FIRST. `dst_unit.getType().str().substr(0, 2) != "l3"` guards the
    /// whole corelet split (`DataflowToSentient.cpp:768-777`), and an L3 half carries
    /// `corelet = 0 : i32` — so testing the corelet first would put every L3 destination in
    /// `src_dst_lx_corelet0` and lose the merged L3 lowering. Both halves are checked because the
    /// prefix, not the half, is what the reference reads.
    #[test]
    fn the_l3_prefix_outranks_the_corelet_attribute() {
        for (id, unit) in [(20, DfirUnit::L3lu), (21, DfirUnit::L3su)] {
            let scope = vec![get_unit_on(id, unit, corelet0())];
            let separated = separate_based_on_destination_units(&[(Val(0), Val(id))], &scope);
            assert_eq!(
                separated,
                SeparatedDestinations {
                    l3: vec![(Val(0), Val(id))],
                    ..SeparatedDestinations::default()
                },
                "{unit:?} is an L3 destination"
            );
            // And the residency it was bound with really is the one that prints `corelet = 0`.
            assert!(is_corelet_0_attribute(crate::units::residency_of(
                unit,
                Core::checked(0).expect("every arch has core 0"),
                corelet0()
            )));
        }
    }

    /// 🎯 161/384 — A NON-L3 DESTINATION WITH **NO** `corelet` LANDS ON THE CORELET-1 LIST.
    ///
    /// ⛔⛔ THE C++'s `// corelet = 1` COMMENT IS WRONG AND THE CODE IS WHAT IS PORTED. The test is
    /// `== getI32IntegerAttr(0)`, so the LX scratchpad — bound with `core` and no `corelet` — takes
    /// the `else`. This test pins the divergence so that an audit reading the comment cannot "fix"
    /// the port into disagreeing with the reference.
    #[test]
    fn a_destination_with_no_corelet_takes_the_else() {
        let scope = vec![get_unit_on(30, DfirUnit::Lx, corelet0())];
        let separated = separate_based_on_destination_units(&[(Val(0), Val(30))], &scope);
        assert_eq!(
            separated,
            SeparatedDestinations {
                lx_corelet1: vec![(Val(0), Val(30))],
                ..SeparatedDestinations::default()
            }
        );
    }

    /// 🎯 161/384 — A DESTINATION THAT IS NEITHER OP IS DROPPED WITHOUT A WORD.
    ///
    /// ⛔ BOTH `dyn_cast`s FAILING MEANS NO `push_back` AT ALL. There is no fifth bucket and no
    /// diagnostic; the pair vanishes and the loop continues to the next destination — which this test
    /// shows by keeping a good pair behind the dropped one. A value with no defining op at all (a
    /// region argument, the reference's null pointer) is the same silence.
    #[test]
    fn a_destination_that_is_neither_op_is_dropped() {
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                result: Val(40),
                of: Val(41),
                which: dataflow::LocalUnit::PeLrf,
            }),
            get_unit_on(41, DfirUnit::Lxsu, corelet0()),
        ];
        let separated = separate_based_on_destination_units(
            &[
                (Val(0), Val(40)),
                // No op defines `%99` — the reference's `getDefiningOp()` is null here.
                (Val(1), Val(99)),
                (Val(2), Val(41)),
            ],
            &scope,
        );
        assert_eq!(
            separated,
            SeparatedDestinations {
                lx_corelet0: vec![(Val(2), Val(41))],
                ..SeparatedDestinations::default()
            }
        );
    }

    /// 🎯 161/384 — AND THE ORDER WITHIN A BUCKET IS THE DESTINATION ORDER.
    ///
    /// ⛔ BECAUSE THE CALLER INDEXES IT. `lowerSyncLXL3ToLXL3` reads `src_dst_l3[0]` and walks the
    /// lists in order, so a port that sorted or grouped them would rename which sync is emitted
    /// first.
    #[test]
    fn the_pairs_keep_their_destination_order() {
        let scope = vec![
            get_unit_on(50, DfirUnit::Lxlu, corelet0()),
            get_unit_on(51, DfirUnit::Lxsu, corelet0()),
        ];
        let separated = separate_based_on_destination_units(
            &[(Val(7), Val(51)), (Val(8), Val(50)), (Val(9), Val(51))],
            &scope,
        );
        assert_eq!(
            separated.lx_corelet0,
            vec![(Val(7), Val(51)), (Val(8), Val(50)), (Val(9), Val(51))]
        );
    }

    /// 🎯 222/384 — TWO REGIONS, TWO DISTINCT `index` ARGUMENTS, AND A YIELD IN EACH.
    ///
    /// ⛔ THE ARGUMENTS MUST DIFFER: `getRegionArg(i)` is `getRegion(i).getArgument(0)`, so a shared
    /// value would make entry 182's per-region remapping read the wrong region's unit.
    #[test]
    fn the_two_uniform_regions_split_the_units_and_bind_one_argument_each() {
        let mut values = Values::default();
        let op = create_uniform_regions_with_two_regions_no_result(
            &[Val(0), Val(1)],
            &[Val(2)],
            &mut values,
        );
        let uniform::Op::UniformizeRegions { regions, results } = &op else {
            panic!("entry 222 builds a uniformize_regions: {op:?}");
        };
        assert!(results.is_empty(), "created with mlir::TypeRange()");
        assert_eq!(regions.len(), 2);
        assert_ne!(regions[0].arg, regions[1].arg);
        assert_eq!(regions[0].units, vec![Val(0), Val(1)]);
        assert_eq!(regions[1].units, vec![Val(2)]);
        for region in regions {
            assert_eq!(
                region.body,
                vec![DfirOp::Uniform(uniform::Op::Yield {
                    operands: Vec::new()
                })]
            );
        }
    }

    /// 🎯 223/384 — `sentient.sync {.., mode = send, soft = false, units = [lxlu0]}`.
    ///
    /// The vendor's own line: `dcc/test/L3SU/sync-op-l3su.mlir:24`, an L3SU send naming one extended
    /// LX load unit.
    #[test]
    fn an_l3_sync_for_one_unit_names_the_extended_destination() {
        let op = L3Half::Store.lower_l3_sync_operation_for_a_unit(
            SyncToLower::Send(dataflow::AsyncTransferWait::Immediately),
            L3SyncDst::Lx(LxHalf::Load, corelet0()),
            None,
        );
        let sen::Op::Sync {
            mode,
            peers,
            soft,
            implicit_sync_memory_boundary,
            dbg_name,
        } = op
        else {
            unreachable!("just built one")
        };
        assert_eq!(mode, sen::SyncMode::Send);
        assert_eq!(
            peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            vec![sen::Consumer::Lxlu0]
        );
        assert!(!soft, "`wait_immediately_for_async_transfers = true`");
        assert_eq!(
            implicit_sync_memory_boundary, None,
            "`getSI32IntegerAttr(-1)`"
        );
        assert_eq!(dbg_name, None);
    }

    /// 🎯 224/384 — the six-unit group of `sync-op-l3su.mlir:14`, soft, and deduplicated.
    ///
    /// ⛔ ORDER IS FIRST OCCURRENCE, NOT SORTED: the key prints
    /// `[lxlu0, lxlu1, lxsu0, lxsu1, l3lu, l3su]`, which is the group's own order.
    #[test]
    fn an_l3_sync_for_a_group_dedups_in_first_occurrence_order() {
        let op = L3Half::Store.lower_l3_sync_operation_for_a_group_of_units(
            SyncToLower::Send(dataflow::AsyncTransferWait::Deferred),
            &[
                L3SyncDst::Lx(LxHalf::Load, corelet0()),
                L3SyncDst::Lx(LxHalf::Load, corelet1()),
                L3SyncDst::Lx(LxHalf::Store, corelet0()),
                L3SyncDst::Lx(LxHalf::Store, corelet1()),
                L3SyncDst::L3lu,
                L3SyncDst::L3su,
                L3SyncDst::Lx(LxHalf::Load, corelet0()),
            ],
            None,
        );
        let sen::Op::Sync {
            mode, peers, soft, ..
        } = op
        else {
            unreachable!("just built one")
        };
        assert_eq!(mode, sen::SyncMode::Send);
        assert!(soft, "`wait_immediately_for_async_transfers = false`");
        assert_eq!(
            peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            vec![
                sen::Consumer::Lxlu0,
                sen::Consumer::Lxlu1,
                sen::Consumer::Lxsu0,
                sen::Consumer::Lxsu1,
                sen::Consumer::L3lu,
                sen::Consumer::L3su,
            ]
        );
    }

    /// 🎯 221/384 — THE LIST IS A SET, AND IT KEEPS INSERTION ORDER.
    ///
    /// ⛔ THE DUPLICATE IS THE WHOLE FUNCTION. Its callers push the same consumer once per sync pair
    /// they walk, and a list naming `lxlu0` twice emits the sync to it twice.
    #[test]
    fn a_consumer_is_appended_once_and_in_order() {
        let mut list = Vec::new();
        push_back_the_unit_to_list_if_doesnot_exist(sen::Consumer::Lxlu0, &mut list);
        push_back_the_unit_to_list_if_doesnot_exist(sen::Consumer::Sfp, &mut list);
        push_back_the_unit_to_list_if_doesnot_exist(sen::Consumer::Lxlu0, &mut list);
        assert_eq!(list, vec![sen::Consumer::Lxlu0, sen::Consumer::Sfp]);
    }
    /// 🎯 273/384 — THE `N` SAYS THE DESTINATION IS ON THE OTHER CORELET, AND BOTH CORELETS SPLIT.
    ///
    /// ⛔ THE VENDOR'S OWN PAIR: `@same_corelet` expects `units = [lxsu]` and `@diff_corelet`
    /// `units = [lxsuN]` for the same `lxlu`→`lxsu` sync
    /// (`Conversion/DataflowToSentient/uniform_sync.mlir:25` and `:43`). With a source on EACH
    /// corelet the sync uniformizes and the two regions name opposite spellings.
    #[test]
    fn an_lx_sync_names_the_neighbour_only_across_corelets() {
        let corelet0 = Corelet::checked(0).expect("every arch has corelet 0");
        let corelet1 = Corelet::checked(1).expect("Target::CORELETS_PER_CORE is 2");
        let src = L0LxSrc::Lx(LxHalf::Load);
        let mut values = Values::default();

        let peers = |lowered: Option<L0LxLowering>| -> Vec<sen::Consumer> {
            match lowered {
                Some(L0LxLowering::One(sen::Op::Sync { peers, .. })) => {
                    peers.iter().copied().map(sen::SyncHalf::peer).collect()
                }
                other => panic!("entry 273 emits one sentient.sync here: {other:?}"),
            }
        };

        // `@same_corelet` — src on corelet 0, dst `lxsu` on corelet 0.
        assert_eq!(
            peers(src.lower_l0lx_sync_operation_for_a_unit(
                L0LxSyncToLower::Send(dataflow::AsyncTransferWait::Immediately),
                &[Val(0)],
                &[],
                L0LxSyncDst::Lx(LxHalf::Store, corelet0),
                None,
                &mut values,
            )),
            vec![sen::Consumer::Lxsu]
        );
        // `@diff_corelet` — the same source, the destination now on corelet 1.
        assert_eq!(
            peers(src.lower_l0lx_sync_operation_for_a_unit(
                L0LxSyncToLower::Recv,
                &[Val(0)],
                &[],
                L0LxSyncDst::Lx(LxHalf::Store, corelet1),
                None,
                &mut values,
            )),
            vec![sen::Consumer::LxsuN]
        );

        // Both corelets hold a source: two regions, and region 1 sees the same destination as its
        // OWN unit where region 0 saw the neighbour (`:337-340`).
        let Some(L0LxLowering::TwoRegions {
            regions,
            region0,
            region1,
            l3,
        }) = src.lower_l0lx_sync_operation_for_a_unit(
            L0LxSyncToLower::Recv,
            &[Val(0)],
            &[Val(1)],
            L0LxSyncDst::Lx(LxHalf::Store, corelet1),
            None,
            &mut values,
        )
        else {
            panic!("both corelets occupied uniformizes the sync");
        };
        assert!(l3.is_none(), "the per-unit lowering has no third create");
        let uniform::Op::UniformizeRegions { regions, .. } = &regions else {
            panic!("entry 222 builds a uniformize_regions: {regions:?}");
        };
        assert_eq!(regions[0].units, vec![Val(0)]);
        assert_eq!(regions[1].units, vec![Val(1)]);
        assert_eq!(
            peers(Some(L0LxLowering::One(region0))),
            vec![sen::Consumer::LxsuN]
        );
        assert_eq!(
            peers(Some(L0LxLowering::One(region1))),
            vec![sen::Consumer::Lxsu]
        );
    }

    /// 🎯 274/384 — THE GROUP'S TWO LISTS ARE REVERSED IMAGES, AND REGION 1 GETS ITS OWN.
    ///
    /// ⛔⛔ THIS PINS DELIBERATE DIVERGENCE (1). The vendor's `@same_group` syncs one `lxlu` source at
    /// a time to the group `{lxsu@corelet0, lxsu@corelet1}` and prints
    /// `[lxsu, lxsuN]` for a corelet-0 source and `[lxsuN, lxsu]` for a corelet-1 one
    /// (`Conversion/DataflowToSentient/uniform_sync.mlir:247` against `:255`). `:645` hands region 1
    /// the FIRST of those two lists; region 1 here gets the second.
    #[test]
    fn a_group_sync_gives_each_corelet_its_own_reversed_list() {
        let corelet0 = Corelet::checked(0).expect("every arch has corelet 0");
        let corelet1 = Corelet::checked(1).expect("Target::CORELETS_PER_CORE is 2");
        let src = L0LxSrc::Lx(LxHalf::Load);
        let group = [
            L0LxSyncDst::Lx(LxHalf::Store, corelet0),
            L0LxSyncDst::Lx(LxHalf::Store, corelet1),
        ];
        let mut values = Values::default();
        let peers = |op: &sen::Op| match op {
            sen::Op::Sync { peers, .. } => peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            other => panic!("entry 274 emits sentient.sync: {other:?}"),
        };

        // `@same_group`, source on corelet 0 — the list the vendor prints at `:247`.
        let Some(L0LxLowering::One(from_corelet0)) = src
            .lower_l0lx_sync_operation_for_a_group_of_units(
                L0LxSyncToLower::Send(dataflow::AsyncTransferWait::Immediately),
                &[Val(0)],
                &[],
                &group,
                None,
                &mut values,
            )
        else {
            panic!("a single-corelet source emits one sync");
        };
        assert_eq!(
            peers(&from_corelet0),
            vec![sen::Consumer::Lxsu, sen::Consumer::LxsuN]
        );

        // Both corelets occupied: region 0 gets that list and region 1 the reversed one.
        let Some(L0LxLowering::TwoRegions {
            region0,
            region1,
            l3,
            ..
        }) = src.lower_l0lx_sync_operation_for_a_group_of_units(
            L0LxSyncToLower::Send(dataflow::AsyncTransferWait::Immediately),
            &[Val(0)],
            &[Val(1)],
            &group,
            None,
            &mut values,
        )
        else {
            panic!("both corelets occupied uniformizes the group sync");
        };
        assert_eq!(
            peers(&region0),
            vec![sen::Consumer::Lxsu, sen::Consumer::LxsuN]
        );
        assert_eq!(
            peers(&region1),
            vec![sen::Consumer::LxsuN, sen::Consumer::Lxsu]
        );
        // Divergence (2): no L3 destination in the group, so no third sync naming nobody.
        assert!(l3.is_none());
    }

    /// 🎯 300/384 — THE SOURCE'S FIRST TWO CHARACTERS PICK THE LOWERING, AND THE TWO NAME ONE
    /// DESTINATION DIFFERENTLY.
    ///
    /// ⛔ An `l3su` source syncing to `lxlu` on corelet 1 names `lxlu1`, the EXTENDED spelling; an
    /// `lxlu` source whose own units sit on corelet 0 names `lxluN`, the NEIGHBOUR. A dispatch on the
    /// wrong prefix emits a peer that is on neither end of the sync.
    #[test]
    fn the_source_prefix_picks_the_l3_lowering_or_the_lx_one() {
        let corelet1 = Corelet::checked(1).expect("Target::CORELETS_PER_CORE is 2");
        let dst = L0LxSyncDst::Lx(LxHalf::Load, corelet1);
        let mut values = Values::default();
        let peers = |lowered: Option<L0LxLowering>| match lowered {
            Some(L0LxLowering::One(sen::Op::Sync { peers, .. })) => peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            other => panic!("entry 300 emits one sentient.sync: {other:?}"),
        };

        assert_eq!(
            peers(lower_sync_for_a_unit(
                SyncSrc::L3(L3Half::Store),
                L0LxSyncToLower::Recv,
                &[],
                &[],
                dst,
                None,
                &mut values,
            )),
            vec![sen::Consumer::Lxlu1]
        );
        assert_eq!(
            peers(lower_sync_for_a_unit(
                SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Load)),
                L0LxSyncToLower::Recv,
                &[Val(0)],
                &[],
                dst,
                None,
                &mut values,
            )),
            vec![sen::Consumer::LxluN]
        );
    }

    /// 🎯 301/384 — THE SAME DISPATCH OVER A GROUP, WHERE THE TWO ARMS DEDUPLICATE ON DIFFERENT
    /// SPELLINGS.
    ///
    /// ⛔ The L3 arm names the group's two corelets `lxsu0`/`lxsu1` — the corelet is in the NAME —
    /// while the LX arm names them `lxsu`/`lxsuN`, relative to its own corelet. Both lists have two
    /// entries, so a dispatch that picked the wrong one would still emit a two-peer sync.
    #[test]
    fn the_group_dispatch_deduplicates_on_the_arms_own_spelling() {
        let corelet0 = Corelet::checked(0).expect("every arch has corelet 0");
        let corelet1 = Corelet::checked(1).expect("Target::CORELETS_PER_CORE is 2");
        let group = [
            L0LxSyncDst::Lx(LxHalf::Store, corelet0),
            L0LxSyncDst::Lx(LxHalf::Store, corelet1),
        ];
        let mut values = Values::default();
        let peers = |lowered: Option<L0LxLowering>| match lowered {
            Some(L0LxLowering::One(sen::Op::Sync { peers, .. })) => peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            other => panic!("entry 301 emits one sentient.sync: {other:?}"),
        };

        assert_eq!(
            peers(lower_sync_for_a_group(
                SyncSrc::L3(L3Half::Load),
                L0LxSyncToLower::Recv,
                &[],
                &[],
                &group,
                None,
                &mut values,
            )),
            vec![sen::Consumer::Lxsu0, sen::Consumer::Lxsu1]
        );
        assert_eq!(
            peers(lower_sync_for_a_group(
                SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Load)),
                L0LxSyncToLower::Recv,
                &[Val(0)],
                &[],
                &group,
                None,
                &mut values,
            )),
            vec![sen::Consumer::Lxsu, sen::Consumer::LxsuN]
        );
    }

    /// 🎯 302/384 — TWO DESTINATION CORELETS ARE TWO REGIONS, EACH CARRYING ITS BUCKET'S SOURCES AND
    /// ONE SYNC.
    ///
    /// ⛔ REGION ORDER IS BUCKET ORDER AND THE NEIGHBOUR SUFFIX IS PER REGION: with the source on
    /// corelet 0, region 0's corelet-0 destination is the local `lxsu` and region 1's corelet-1
    /// destination is `lxsuN`. One region per DESTINATION corelet, not per source.
    #[test]
    fn the_two_destination_corelets_take_a_region_each() {
        let corelet0 = Corelet::checked(0).expect("every arch has corelet 0");
        let corelet1 = Corelet::checked(1).expect("Target::CORELETS_PER_CORE is 2");
        let unit = |result: u32, unit: DfirUnit, corelet: Corelet| {
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(result),
                residency: Residency::Corelet {
                    core: Core::checked(0).expect("every arch has core 0"),
                    corelet,
                },
                unit,
                num_folds: None,
            })
        };
        let scope = vec![
            unit(0, DfirUnit::Lxlu, corelet0),
            unit(1, DfirUnit::Lxlu, corelet1),
            unit(2, DfirUnit::Lxsu, corelet0),
            unit(3, DfirUnit::Lxsu, corelet1),
        ];
        let mut values = Values::default();

        let lowered = lower_sync_lx_l3_to_lx_l3(
            SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Load)),
            L0LxSyncToLower::Recv,
            &[(Val(0), Val(2)), (Val(1), Val(3))],
            corelet0,
            &scope,
            None,
            &mut values,
        );

        let Some(LxL3Lowering::Regions { regions, syncs }) = lowered else {
            panic!("two destination corelets uniformize: {lowered:?}");
        };
        let uniform::Op::UniformizeRegions { regions, results } = regions else {
            panic!("entry 302 emits uniform.uniformize_regions: {regions:?}");
        };
        assert!(results.is_empty());
        assert_eq!(
            regions.iter().map(|r| r.units.clone()).collect::<Vec<_>>(),
            vec![vec![Val(0)], vec![Val(1)]]
        );
        let peers = |sync: &sen::Op| match sync {
            sen::Op::Sync { peers, .. } => peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            other => panic!("a region holds one sentient.sync: {other:?}"),
        };
        assert_eq!(peers(&syncs[0]), vec![sen::Consumer::Lxsu]);
        assert_eq!(peers(&syncs[1]), vec![sen::Consumer::LxsuN]);
    }

    /// 🎯 319/384 — BOTH CORELETS HOLD A SOURCE, SO THE SYNC IS UNIFORMIZED: region 0 names `lxsu`
    /// and region 1 names `lxsuN` for the SAME destination, because `corelet_id` is the region's.
    ///
    /// ⛔ THE UNITS THE REGIONS BIND ARE THE MAP'S KEYS, split by the key's own corelet.
    #[test]
    fn a_query_map_over_both_corelets_uniformizes_into_two_regions() {
        let scope = vec![
            get_unit_on(0, DfirUnit::Lxsu, corelet0()),
            get_unit_on(1, DfirUnit::Lxsu, corelet1()),
            get_unit_on(10, DfirUnit::Lxsu, corelet0()),
        ];
        let mut values = Values::default();
        let lowered = lower_sync_for_a_query_map(
            SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Store)),
            L0LxSyncToLower::Recv,
            &[(Val(0), Val(10)), (Val(1), Val(10))],
            &scope,
            None,
            &mut values,
        )
        .expect("both corelets hold a source and the target is not the L3");

        let peers = |sync: &sen::Op| match sync {
            sen::Op::Sync { peers, .. } => peers
                .iter()
                .copied()
                .map(sen::SyncHalf::peer)
                .collect::<Vec<_>>(),
            other => panic!("a region holds one sentient.sync: {other:?}"),
        };
        let QueryMapLowering::TwoRegions {
            regions,
            region0,
            region1,
        } = lowered
        else {
            panic!("two corelets are two regions");
        };
        let uniform::Op::UniformizeRegions { regions, results } = regions else {
            panic!("`createUniformRegionsWithTwoRegionsNoResult` builds one op");
        };
        assert!(results.is_empty());
        assert_eq!(
            regions.iter().map(|r| r.units.clone()).collect::<Vec<_>>(),
            vec![vec![Val(0)], vec![Val(1)]]
        );
        let LxL3Lowering::One(ref sync0) = region0 else {
            panic!("one bucket is one sync");
        };
        let LxL3Lowering::One(ref sync1) = region1 else {
            panic!("one bucket is one sync");
        };
        assert_eq!(peers(sync0), vec![sen::Consumer::Lxsu]);
        assert_eq!(
            peers(sync1),
            vec![sen::Consumer::LxsuN],
            "corelet 1's sources reach a corelet-0 destination through the neighbour"
        );
    }

    /// 🎯 319/384 — THE L0 ARM NAMES THE DESTINATION'S HALF AND CARRIES THE IMPLICIT SYNC'S TILE
    /// SIZE; a destination on the SAME half is `emitError("Unsupported dst unit for L0 unit.")`.
    #[test]
    fn an_l0_query_map_syncs_with_the_other_half_and_refuses_its_own() {
        let scope = vec![
            get_unit_on(0, DfirUnit::L0lu, corelet0()),
            get_unit_on(10, DfirUnit::L0su, corelet0()),
            get_unit_on(11, DfirUnit::L0lu, corelet1()),
        ];
        let mut values = Values::default();
        let lowered = lower_sync_for_a_query_map(
            SyncSrc::L0Lx(L0LxSrc::L0(L0Half::Load)),
            L0LxSyncToLower::ImplicitSync(64),
            &[(Val(0), Val(10))],
            &scope,
            None,
            &mut values,
        );
        assert_eq!(
            lowered,
            Some(QueryMapLowering::One(sen::Op::Sync {
                mode: sen::SyncMode::SendRecv,
                peers: vec![sen::SyncHalf::of_lowered_destination(sen::Consumer::L0su)],
                soft: false,
                implicit_sync_memory_boundary: Some(64),
                dbg_name: None,
            }))
        );

        assert_eq!(
            lower_sync_for_a_query_map(
                SyncSrc::L0Lx(L0LxSrc::L0(L0Half::Load)),
                L0LxSyncToLower::ImplicitSync(64),
                &[(Val(0), Val(11))],
                &scope,
                None,
                &mut values,
            ),
            None,
            "an l0lu source cannot sync with another l0lu"
        );
    }
    /// 🎯 337/384 — THE SOURCES SPLIT BY CORELET AND THE DESTINATION'S OP PICKS THE LOWERING; a list
    /// mixing two unit kinds is *"Src unit types has to be the same."*
    #[test]
    fn a_sync_dispatches_on_its_destination_over_corelet_split_sources() {
        let scope = vec![
            get_unit_on(0, DfirUnit::Lxsu, corelet0()),
            get_unit_on(1, DfirUnit::Lxsu, corelet1()),
            get_unit_on(2, DfirUnit::Lxlu, corelet0()),
            get_unit_on(3, DfirUnit::L3lu, corelet0()),
        ];
        let mut values = Values::default();
        let lowered = lower_sync_operation(
            &[Val(0), Val(1)],
            L0LxSyncToLower::Recv,
            Val(2),
            &scope,
            None,
            &mut values,
        );
        let mut expected_values = Values::default();
        assert_eq!(
            lowered,
            lower_sync_for_a_unit(
                SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Store)),
                L0LxSyncToLower::Recv,
                &[Val(0)],
                &[Val(1)],
                L0LxSyncDst::Lx(LxHalf::Load, corelet0()),
                None,
                &mut expected_values,
            )
            .map(SyncLowering::L0Lx),
            "`lowerSyncForAUnit(src_unit_name, op, .., corelet0, corelet1, dst_unit)`"
        );

        // ⛔ THE MIXED LIST IS THE REFUSAL, and an L3 source reaches the lowering with both corelet
        // lists empty.
        assert_eq!(
            lower_sync_operation(
                &[Val(0), Val(3)],
                L0LxSyncToLower::Recv,
                Val(2),
                &scope,
                None,
                &mut values,
            ),
            None
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 159/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e159_getUnitNameFromAListOfGetUnitOp
///
/// **159/384** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119` (10L).
///
/// ```cpp
/// static std::string getUnitNameFromAListOfGetUnitOp(
///     std::vector<mlir::dataflow::GetUnitOp> &units) {
///   DT_CHECK(units.size() > 0);
///   std::string unit_name = units[0].getType().str();
///   for (auto src_unit : units) {
///     if (src_unit.getType().str() != unit_name) {
///       units[0]->emitError("Src unit types has to be the same.");
///       return "";
///     }
///   }
///   return unit_name;
/// }
/// ```
///
/// # ⭐⭐ `getType()` IS THE `type` **ATTRIBUTE**, NOT THE VALUE'S MLIR TYPE
///
/// `dataflow.get_unit` declares `(ins StrAttr:$name, StrAttr:$type)` and returns
/// `Variadic<Index>:$units` (`Dataflow.td:54-58`), so the generated `getType()` accessor hands back
/// the `type` STRING — `"lxlu"`, `"l3su"`, `"pe"` — and `.str()` copies it. Every result of the op is
/// an `index`, so reading the *value's* type would answer `index` for all six units and this function
/// would never find a difference. That is why the answer here is a [`DfirUnit`] and the list is one
/// of unit kinds: the `get_unit` ops' identities are not read, only their `type`.
///
/// # ⛔⛔ THE MISMATCH'S `""` AND THE EMPTY LIST'S ABORT ARE ONE ANSWER, BECAUSE NEITHER IS A NAME
///
/// The reference has two ways to fail and no way to report either. `DT_CHECK(units.size() > 0)`
/// aborts; the mismatch returns the empty string — and every one of the four callers feeds the result
/// straight into `EnumsConversion::stringToSenComponents.find(src_unit_name)->second`
/// (`:245-246`, `:397-398`, `:487-488`, `:686-687`), which for `""` dereferences `end()`. So the
/// empty string is not a value a caller can act on; it is UB one line later. [`None`] is the single
/// answer that covers both, and it forces the caller to have the arm the reference does not.
///
/// # ⭐ THE FIRST ELEMENT IS COMPARED AGAINST ITSELF, AND THAT IS NOT A WASTED PASS
///
/// The loop starts at `units[0]`, whose comparison is trivially equal — so a one-element list always
/// answers with that element. Skipping the head would give the same answer, and the port keeps the
/// reference's shape so the pass count matches when the audit reads them side by side.
///
/// # ⭐ IT IS THE **KIND** THAT MUST AGREE, NOT THE CORELET
///
/// `C0-lxlu-CL0` and `C0-lxlu-CL1` are two `get_unit` ops with the same `type` and different `corelet`
/// attributes, and this function accepts them as one name — deliberately: its callers use the answer
/// to pick a lowering by COMPONENT (`senCompToGenericComp.at(src_comp)`, `:247`) and split the
/// corelets separately, with [`are_corelets_different`] and
/// [`separate_based_on_destination_units`]. A list mixing `lxlu` with `l3lu` is the case it refuses.
#[must_use]
pub fn unit_name_from_a_list_of_get_unit_op(units: &[DfirUnit]) -> Option<DfirUnit> {
    // `DT_CHECK(units.size() > 0);` and `units[0].getType().str()` in one read.
    let unit_name = *units.first()?;
    for src_unit in units {
        // `if (src_unit.getType().str() != unit_name)` — "Src unit types has to be the same."
        if *src_unit != unit_name {
            return None;
        }
    }
    Some(unit_name)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 160/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF A CORE'S TWO CORELETS CARRY A SOURCE UNIT — the pair of lists
/// `areCoreletsDifferent` reads, with the reference's own `DT_CHECK` discharged.
///
/// ⛔⛔ THREE CASES BECAUSE THE FOURTH IS THE ABORT. `DT_CHECK(units_with_corelet_0.size() > 0 ||
/// units_with_corelet_1.size() > 0)` (`DataflowToSentient.cpp:136`) says both lists empty is not
/// an input, and this is the only fact about the two lists that the body reads — everything else is
/// `.size() > 0`. Stating the domain as a type moves that abort to the one place a value of this type
/// is minted ([`OccupiedCorelets::of`]), which is how [`crate::bridges::dataflow_ir_to_sentient::vc_helper`]
/// discharged the same shape for `PtLanes`.
///
/// ⭐ AND IT IS TWO CORELETS FOR THE SAME REASON THE REFERENCE HARD-CODES `0` AND `1`: the lists are
/// built per corelet of one core, and `Target::CORELETS_PER_CORE` is two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupiedCorelets {
    /// Only `units_with_corelet_0` is non-empty.
    Corelet0,
    /// Only `units_with_corelet_1` is non-empty.
    Corelet1,
    /// Both lists have a unit on them.
    Both,
}

impl OccupiedCorelets {
    /// WHICH CORELETS THE TWO LISTS OCCUPY, or [`None`] for the pair the reference's `DT_CHECK`
    /// rules out.
    ///
    /// ⭐ THE ELEMENTS ARE NEVER LOOKED AT. `areCoreletsDifferent` reads its two vectors only through
    /// `.size() > 0`, so this takes the `get_unit` results a caller already has and asks nothing else
    /// of them.
    #[must_use]
    pub fn of(
        units_with_corelet_0: &[Val],
        units_with_corelet_1: &[Val],
    ) -> Option<OccupiedCorelets> {
        match (
            units_with_corelet_0.is_empty(),
            units_with_corelet_1.is_empty(),
        ) {
            (false, false) => Some(OccupiedCorelets::Both),
            (false, true) => Some(OccupiedCorelets::Corelet0),
            (true, false) => Some(OccupiedCorelets::Corelet1),
            // `DT_CHECK(units_with_corelet_0.size() > 0 || units_with_corelet_1.size() > 0)`.
            (true, true) => None,
        }
    }

    /// Whether `units_with_corelet_0.size() > 0`.
    #[must_use]
    const fn holds_corelet_0(self) -> bool {
        match self {
            OccupiedCorelets::Corelet0 | OccupiedCorelets::Both => true,
            OccupiedCorelets::Corelet1 => false,
        }
    }

    /// Whether `units_with_corelet_1.size() > 0`.
    #[must_use]
    const fn holds_corelet_1(self) -> bool {
        match self {
            OccupiedCorelets::Corelet1 | OccupiedCorelets::Both => true,
            OccupiedCorelets::Corelet0 => false,
        }
    }
}

/// Replaces: e160_areCoreletsDifferent
///
/// **160/384** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132` (6L).
///
/// ```cpp
/// static bool areCoreletsDifferent(
///     OpBuilder builder, dataflow::GetUnitOp unit,
///     std::vector<dataflow::GetUnitOp> units_with_corelet_0,
///     std::vector<dataflow::GetUnitOp> units_with_corelet_1) {
///   DT_CHECK(units_with_corelet_0.size() > 0 || units_with_corelet_1.size() > 0);
///   return (unit->getAttr("corelet") == builder.getI32IntegerAttr(0) &&
///           units_with_corelet_1.size() > 0) ||
///          (unit->getAttr("corelet") == builder.getI32IntegerAttr(1) &&
///           units_with_corelet_0.size() > 0);
/// }
/// ```
///
/// # ⭐⭐ IT ASKS WHETHER THIS UNIT IS ON THE **OTHER** CORELET FROM SOMEBODY
///
/// Not whether the two lists differ from each other: the subject is `unit`, and each disjunct pairs
/// *its* corelet with the presence of a unit on the opposite one. A unit on corelet 0 with peers only
/// on corelet 0 answers `false`; the same unit with any peer on corelet 1 answers `true`. That is why
/// the two lists collapse to [`OccupiedCorelets`] — their contents never matter, only which corelets
/// are occupied.
///
/// # ⛔⛔ A UNIT WITH **NO** `corelet` ATTRIBUTE ANSWERS `false`, AND THE NULL IS HOW
///
/// `Operation::getAttr` returns a null `Attribute` for an absent name, and a null attribute equals no
/// `IntegerAttr` — so both disjuncts' first conjunct is false and the answer is `false` whatever the
/// lists hold. That is not an accident of the C++: a unit that carries no `corelet` is one the L3
/// path handles, and `lowerL0LXSyncOperationForAUnit` refuses it by name ninety-four lines later
/// (*"Unknown corelet information for sentient"*, `:226-229`). Here that null is
/// [`crate::units::Residency::Scratchpad`] and [`crate::units::Residency::Global`], the two
/// residencies that print no `corelet`.
///
/// # ⛔ AND `CoreWide` IS CORELET **ZERO**, NOT ABSENT
///
/// `C0-l3lu` carries `core` AND `corelet = 0` while `C0-lx` carries `core` alone
/// (`UnitMaterializer.cpp:62-80` against `:142-152`) — the distinction [`crate::units::Residency`]
/// exists to keep. So an L3 half reaching this function IS on corelet 0 as far as `getAttr` is
/// concerned, and answers `true` whenever anything sits on corelet 1.
///
/// # ⛔ `builder` IS ONLY THERE TO MINT THE TWO LITERALS
///
/// `builder.getI32IntegerAttr(0)` and `...(1)` are how the C++ writes `0` and `1` as attributes it
/// can compare against; the builder is not positioned and nothing is emitted. That is the mechanism
/// for reaching an operand, which the campaign brief permits dropping.
///
/// # ⚠️ NO CALLER AT `a0d29abbed`
///
/// A grep of the authority tree finds this symbol exactly once, at its own definition. Its
/// neighbours in the file — `lowerL0LXSyncOperationForAUnit` (`:189`) and `lowerSyncLXL3ToLXL3`
/// (`:787`) — decide the corelet split with their own inline tests and
/// [`separate_based_on_destination_units`] instead. It is ported anyway: a scheduled function gets
/// its port and its audit, and a predicate the reference kept is not this port's to delete.
#[must_use]
pub fn are_corelets_different(unit: Residency, occupied: OccupiedCorelets) -> bool {
    // `unit->getAttr("corelet")` — the attribute, or nothing. Spelled out over the four residencies
    // so that a fifth cannot join the null side without being looked at.
    let corelet = match unit {
        Residency::Corelet { corelet, .. } => Some(corelet.get()),
        // ⭐ `corelet = 0 : i32` IS PRINTED FOR A CORE-WIDE UNIT — see the note above.
        Residency::CoreWide { .. } => Some(0),
        // No `corelet` attribute at all: `getAttr` is null and equals neither literal.
        Residency::Scratchpad { .. } | Residency::Global => None,
    };

    (corelet == Some(0) && occupied.holds_corelet_1())
        || (corelet == Some(1) && occupied.holds_corelet_0())
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 161/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A SYNC'S SOURCE/DESTINATION PAIRS SORTED BY WHERE THE DESTINATION LIVES — the four out-parameters
/// of `separateBasedOnDestinationUnits`.
///
/// ⛔ FOUR LISTS AND NOT A MAP, BECAUSE THE CALLER TESTS THEM AGAINST EACH OTHER. `lowerSyncLXL3ToLXL3`
/// asks three times over whether a named three of the four are empty (`DataflowToSentient.cpp:798-803`,
/// and `:825-826` for the fourth); that the remaining one is non-empty is not tested there but comes
/// from the `DT_CHECK` that at least one of the four is (`:796-797`). So each list is a named field.
///
/// ⭐ THE ORDER WITHIN EACH LIST IS THE DESTINATION ORDER, which is what makes `src_dst_l3[0]` mean
/// anything: the reference walks `dst_vs` by index and pushes as it goes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeparatedDestinations {
    /// `src_dst_lx_corelet0` — pairs whose destination is a non-L3 unit on corelet 0.
    pub lx_corelet0: Vec<(Val, Val)>,
    /// `src_dst_lx_corelet1` — pairs whose destination is a non-L3 unit NOT on corelet 0.
    pub lx_corelet1: Vec<(Val, Val)>,
    /// `src_dst_l3` — pairs whose destination is an `l3lu` or `l3su`.
    pub l3: Vec<(Val, Val)>,
    /// `src_dst_group` — pairs whose destination is a `dataflow.create_group` handle.
    pub group: Vec<(Val, Val)>,
}

/// Replaces: e161_separateBasedOnDestinationUnits
///
/// **161/384** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761` (18L).
///
/// ```cpp
/// static void separateBasedOnDestinationUnits(
///     mlir::OpBuilder builder, std::vector<mlir::Value> src_vs,
///     std::vector<mlir::Value> dst_vs,
///     std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_lx_corelet0,
///     std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_lx_corelet1,
///     std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_l3,
///     std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_group) {
///   for (int i = 0; i < dst_vs.size(); i++) {
///     if (auto dst_unit =
///             llvm::dyn_cast<dataflow::GetUnitOp>(dst_vs[i].getDefiningOp())) {
///       if (dst_unit.getType().str().substr(0, 2) != "l3") {  // LX
///         if (dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(0)) {
///           src_dst_lx_corelet0.push_back(std::make_pair(src_vs[i], dst_vs[i]));
///         } else {  // corelet = 1
///           src_dst_lx_corelet1.push_back(std::make_pair(src_vs[i], dst_vs[i]));
///         }
///       } else {  // L3
///         src_dst_l3.push_back(std::make_pair(src_vs[i], dst_vs[i]));
///       }
///     } else if (auto dst_unit = llvm::dyn_cast<dataflow::CreateGroupOp>(
///                    dst_vs[i].getDefiningOp())) {
///       src_dst_group.push_back(std::make_pair(src_vs[i], dst_vs[i]));
///     }
///   }
/// }
/// ```
///
/// # ⛔⛔ THE `else` BRANCH IS NOT "CORELET 1", IT IS "NOT CORELET 0", AND THE COMMENT LIES
///
/// The C++ writes `} else {  // corelet = 1`, but the test above it is
/// `getAttr("corelet") == getI32IntegerAttr(0)` — so a destination with **no** `corelet` attribute at
/// all lands in `src_dst_lx_corelet1` along with the genuine corelet-1 units, because a null
/// `Attribute` equals no `IntegerAttr`. The only non-L3 unit that can carry no `corelet` is a
/// scratchpad memory ([`crate::units::Residency::Scratchpad`], `C0-lx`), and `lowerSyncLXL3ToLXL3`
/// then treats `src_dst_lx_corelet1` as a corelet-1 region. The port reproduces the CODE and this
/// note records the divergence between it and the comment; see [`are_corelets_different`], which has
/// the same null and where it means `false` instead.
///
/// # ⛔⛔ AND `substr(0, 2) != "l3"` IS DECIDED **BEFORE** THE CORELET, SO AN L3 HALF NEVER SPLITS
///
/// `l3lu` and `l3su` are the only unit spellings beginning `l3` — the same prefix test
/// [`is_target_l3`] makes — and they are also the units that carry `corelet = 0` while belonging to
/// the whole core ([`crate::units::Residency::CoreWide`]). Testing the corelet first would put every
/// L3 destination in `src_dst_lx_corelet0` and lose the merged L3 lowering entirely.
///
/// # ⭐ A DESTINATION THAT IS NEITHER A `get_unit` NOR A `create_group` IS **SILENTLY DROPPED**
///
/// Both `dyn_cast`s failing means no `push_back` on any of the four lists, and the loop moves on.
/// The pair vanishes — no diagnostic, no fifth bucket. The caller's
/// `DT_CHECK(src_dst_lx_corelet0.size() != 0 || …)` (`:796-797`) is the only thing that notices, and
/// only when *every* destination was dropped. `None` from
/// [`defining_op`](crate::islands::dataflow_ir::dialects::defining_op) — a destination that is a
/// region argument, which is the reference's null pointer — is the same silence.
///
/// # ⛔ THE PAIRS ARE INDEXED IN LOCKSTEP AND THE REFERENCE DOES NOT CHECK THE LENGTHS
///
/// The loop bounds on `dst_vs.size()` and indexes `src_vs[i]`, so a source list shorter than the
/// destination list is an out-of-bounds read. Taking the two as ONE list of pairs makes that
/// unspellable rather than checked, which is this crate's standing preference; a caller with two
/// vectors zips them and the zip is where a length difference becomes visible.
///
/// # ⛔ `builder` MINTS THE LITERAL `0` AND NOTHING ELSE
///
/// As in [`are_corelets_different`] — no insertion point, no emission. The four lists ARE the
/// function's output, so they are returned rather than filled through references; the reference's
/// caller declares all four empty immediately before the call (`:791-792`), so appending and
/// returning are the same thing here.
#[must_use]
pub fn separate_based_on_destination_units(
    src_dst: &[(Val, Val)],
    scope: &[DfirOp],
) -> SeparatedDestinations {
    let mut separated = SeparatedDestinations::default();

    for (src_v, dst_v) in src_dst {
        match defining_op(*dst_v, scope) {
            // `llvm::dyn_cast<dataflow::GetUnitOp>(dst_vs[i].getDefiningOp())`.
            Some(DfirOp::Dataflow(dataflow::Op::GetUnit {
                residency, unit, ..
            })) => {
                // `} else {  // L3` (`:777`) — the reference's negated `substr(0, 2) != "l3"` arm.
                if matches!(unit, DfirUnit::L3lu | DfirUnit::L3su) {
                    separated.l3.push((*src_v, *dst_v));
                } else if is_corelet_0_attribute(*residency) {
                    separated.lx_corelet0.push((*src_v, *dst_v));
                } else {
                    // `} else {  // corelet = 1` — and everything the attribute is not 0 for.
                    separated.lx_corelet1.push((*src_v, *dst_v));
                }
            }
            // `llvm::dyn_cast<dataflow::CreateGroupOp>(dst_vs[i].getDefiningOp())`.
            Some(DfirOp::Dataflow(dataflow::Op::CreateGroup { .. })) => {
                separated.group.push((*src_v, *dst_v));
            }
            // ⭐ BOTH CASTS FAILED, OR THERE IS NO DEFINING OP — the pair is dropped, exactly as the
            // reference drops it. The dialects are named rather than wildcarded so that a new
            // destination-binding op cannot join this side unnoticed.
            Some(
                DfirOp::Dataflow(_)
                | DfirOp::Arith(_)
                | DfirOp::Scf(_)
                | DfirOp::Affine(_)
                | DfirOp::Agen(_)
                | DfirOp::VectorChain(_)
                | DfirOp::Vector(_)
                | DfirOp::Symbol(_)
                // ⭐ AND `uniform` JOINS THE DROP SIDE — not vacuously, which is why it is worth a
                // line. A destination reached INSIDE a local region is a `uniform.query_map` result
                // (`flatten_local_region4.mlir:355-356`), and neither `dyn_cast` accepts one, so the
                // reference drops that pair too: the unit such a value stands for is chosen per
                // region, and this sort is over units named at the definition site.
                | DfirOp::Uniform(_),
            )
            | None => {}
        }
    }

    separated
}

/// `unit->getAttr("corelet") == builder.getI32IntegerAttr(0)` — whether the printed `corelet`
/// attribute is present AND zero.
///
/// ⭐ SHARED BY THE TWO UNITS THAT ASK IT, so the null-attribute rule is written once. See
/// [`are_corelets_different`] for why an absent attribute is `false` and why
/// [`crate::units::Residency::CoreWide`] is zero rather than absent.
const fn is_corelet_0_attribute(residency: Residency) -> bool {
    match residency {
        Residency::Corelet { corelet, .. } => corelet.get() == 0,
        Residency::CoreWide { .. } => true,
        Residency::Scratchpad { .. } | Residency::Global => false,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 222/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e222_createUniformRegionsWithTwoRegionsNoResult
///
/// **222/384** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:152` (22L).
///
/// A result-less `uniform.uniformize_regions` with two regions, each binding one `index` argument
/// (`emplaceBlock()` + `addArgument`) and holding a bare `uniform.yield`. The returned op IS the pair
/// of `OpBuilder`s: *"insert into region i"* is a push onto `regions[i].body` before that yield.
/// ⛔ TWO SLICES, NOT `units` + `list_sizes` — the op's three prefix-sum `assert`s
/// (`Uniform.cpp:125-136`) are unwritable once the split is the argument.
#[must_use]
pub fn create_uniform_regions_with_two_regions_no_result(
    region0_units: &[Val],
    region1_units: &[Val],
    values: &mut Values,
) -> uniform::Op {
    let mut region = |units: &[Val]| uniform::LocalRegion {
        arg: values.mint(),
        units: units.to_vec(),
        body: vec![DfirOp::Uniform(uniform::Op::Yield {
            operands: Vec::new(),
        })],
    };
    uniform::Op::UniformizeRegions {
        regions: vec![region(region0_units), region(region1_units)],
        // `mlir::TypeRange()`.
        results: Vec::new(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 223/384 + 224/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH HALF OF THE L3 A SYNC LEAVES FROM — the source guard of both L3 sync lowerings, as a type.
///
/// ⛔ `gen_comp == L3LU || gen_comp == L3SU` (`DataflowToSentient.cpp:400` and `:689`) is the ONLY
/// thing either function reads the source unit for; every other component leaves through
/// `emitError("Unknown lowering of the sync operation")`. Carried as the RECEIVER of the two
/// lowerings because an unused `self` is silent where an unused named parameter is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum L3Half {
    /// `l3lu` — `SenComponents::L3LU`.
    Load,
    /// `l3su` — `SenComponents::L3SU`.
    Store,
}

/// WHICH `dataflow` SYNC IS BEING LOWERED — `DT_CHECK(send_op || recv_op)` as a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncToLower {
    /// `dataflow.sync_send`, carrying its `$wait_immediately_for_async_transfers`.
    ///
    /// ⛔ THE ATTRIBUTE'S ABSENCE IS THE OTHER REFUSAL (`:385-388`, `:677-680`). It is a mandatory
    /// field of [`dataflow::Op::SyncSend`], so `!has_value()` has no input here.
    Send(dataflow::AsyncTransferWait),
    /// `dataflow.sync_recv` — never soft, and `wait_immediately_for_async_transfers` is not its
    /// attribute.
    Recv,
}

impl SyncToLower {
    /// `sync_tag` — `send`, and `recv` only for a `sync_recv`.
    #[must_use]
    pub const fn mode(self) -> sen::SyncMode {
        match self {
            Self::Send(_) => sen::SyncMode::Send,
            Self::Recv => sen::SyncMode::Recv,
        }
    }

    /// `soft = send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value()`.
    #[must_use]
    pub const fn soft(self) -> bool {
        matches!(self, Self::Send(dataflow::AsyncTransferWait::Deferred))
    }
}

/// A DESTINATION AN L3 SYNC MAY NAME — the eight accepted components, with `LXLU`/`LXSU` already
/// carrying the corelet that extends them.
///
/// ⛔ EIGHT COMPONENTS, THREE CASES. `is_any_of(dst_comp, L3LU, L3SU, LXLU, LXSU, LXLU0, LXSU0,
/// LXLU1, LXSU1)` (`:405-408`, negated at `:699-704`) is the whole accepted set, and `LXLU0` is
/// exactly `Lx(Load, corelet 0)` once [`extend_unit_name_to_corelet`] has run — so the two spellings
/// collapse, and both `emitError("Unknown lowering of the L3 sync …")` arms lose their input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3SyncDst {
    /// `l3lu`.
    L3lu,
    /// `l3su`.
    L3su,
    /// `lxlu`/`lxsu`, with the corelet whose number is appended to the name.
    Lx(LxHalf, Corelet),
}

impl L3SyncDst {
    /// `symbolizeSentientLoadConsumer(dst_unit_name).value()`.
    #[must_use]
    pub const fn consumer(self) -> sen::Consumer {
        match self {
            Self::L3lu => sen::Consumer::L3lu,
            Self::L3su => sen::Consumer::L3su,
            Self::Lx(half, corelet) => extend_unit_name_to_corelet(half, corelet),
        }
    }
}

impl L3Half {
    /// Replaces: e223_lowerL3SyncOperationForAUnit
    ///
    /// **223/384** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375` (60L).
    ///
    /// One `sentient.sync` naming one destination.
    ///
    /// ⛔ `implicit_sync_memory_boundary: None` IS `builder.getSI32IntegerAttr(-1)`, which the
    /// vendor's key prints on every one of these (`dcc/test/L3SU/sync-op-l3su.mlir:24`).
    #[must_use]
    pub fn lower_l3_sync_operation_for_a_unit(
        self,
        op: SyncToLower,
        dst: L3SyncDst,
        dbg_name: Option<String>,
    ) -> sen::Op {
        sen::Op::Sync {
            mode: op.mode(),
            peers: vec![sen::SyncHalf::of_lowered_destination(dst.consumer())],
            soft: op.soft(),
            implicit_sync_memory_boundary: None,
            dbg_name,
        }
    }

    /// Replaces: e224_lowerL3SyncOperationForAGroupOfUnits
    ///
    /// **224/384** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:666` (65L).
    ///
    /// The same op over a whole `dataflow.create_group`, deduplicated in first-occurrence order
    /// (`std::find(..) == end()`, `:718-720`).
    ///
    /// ⛔ THE REFERENCE'S `break` ON A BAD DESTINATION STILL EMITS THE SYNC, silently dropping every
    /// remaining unit of the group; [`L3SyncDst`] makes that input unrepresentable instead.
    #[must_use]
    pub fn lower_l3_sync_operation_for_a_group_of_units(
        self,
        op: SyncToLower,
        dst_unit_group: &[L3SyncDst],
        dbg_name: Option<String>,
    ) -> sen::Op {
        let mut peers: Vec<sen::SyncHalf> = Vec::new();
        for dst in dst_unit_group {
            let peer = sen::SyncHalf::of_lowered_destination(dst.consumer());
            if !peers.contains(&peer) {
                peers.push(peer);
            }
        }
        sen::Op::Sync {
            mode: op.mode(),
            peers,
            soft: op.soft(),
            implicit_sync_memory_boundary: None,
            dbg_name,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// 221/384
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e221_pushBackTheUnitToListIfDoesnotExist
///
/// `pushBackTheUnitToListIfDoesnotExist` (`DataflowToSentient.cpp:143`) — appends one consumer to a
/// sync's unit list unless the list already names it.
///
/// ⛔ THE REFERENCE ABORTS ON AN UNKNOWN NAME, and that is what the enum parameter removes: it takes
/// a `std::string` and calls `symbolizeSentientLoadConsumer(unit_name).value()` (`:146-147`), which
/// throws for anything outside the sixteen. There is no name here to fail to symbolize.
pub fn push_back_the_unit_to_list_if_doesnot_exist(
    unit: sen::Consumer,
    list: &mut Vec<sen::Consumer>,
) {
    // `:148-149` — `std::find(list.begin(), list.end(), unit_attr) == list.end()`.
    if !list.contains(&unit) {
        list.push(unit);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 273/384 + 274/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH HALF OF THE L0 A SYNC NAMES — the two components [`is_sen_component_l0lu`] and
/// [`is_sen_component_l0su`] tell apart, as one type.
///
/// ⛔ THE CROSS-HALF GUARD IS THE ONLY THING EITHER FUNCTION ASKS OF IT. Both lowerings accept a
/// destination only when `(isSenComponentL0LU(src) && isSenComponentL0SU(dst)) ||
/// (isSenComponentL0SU(src) && isSenComponentL0LU(dst))` (`DataflowToSentient.cpp:253-255`,
/// negated per destination at `:507-510`) — an L0 load unit syncs with an L0 store unit and with
/// nothing else — and the emitted name is then read off the DESTINATION's half alone (`:256-257`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum L0Half {
    /// `l0lu` — `SenComponents::L0LU`.
    Load,
    /// `l0su` — `SenComponents::L0SU`.
    Store,
}

/// WHICH UNIT AN L0/LX SYNC LEAVES FROM — the `switch (gen_comp)`'s four accepted cases.
///
/// ⛔ THE `default:` ARM IS `emitError("Unknown lowering of the sync operation")` (`:365-367`,
/// `:657-659`), so naming the four is what removes it. The source's own corelet is NOT here: which
/// corelets hold a source is the pair of lists, i.e. [`OccupiedCorelets`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L0LxSrc {
    /// Generic `L0LU`/`L0SU`.
    L0(L0Half),
    /// Generic `LXLU`/`LXSU` — ⭐ WHICH HALF IS NEVER READ on this side: the LX arms derive the
    /// emitted name from the destination, so the source half only picks the `case` label.
    Lx(LxHalf),
}

/// WHICH `dataflow` SYNC AN L0/LX LOWERING IS GIVEN — `DT_CHECK(send_op || recv_op ||
/// implicit_sync_op)` (`:196`, `:445`) as a type.
///
/// ⛔ NOT [`SyncToLower`], AND THE THIRD CASE IS WHY. The L3 twins accept only a send or a recv
/// (`DT_CHECK(send_op || recv_op)`, `:381`), so giving them an implicit-sync input would offer them
/// the one op they abort on. ⭐ AND `soft` IS NOT A FIELD OF THE ANSWER HERE: all nine
/// `sentient.sync` creates in these two functions pass `false` (`:262`, `:283`, `:309`, `:350`,
/// `:361`, `:523`, `:603`, `:618`, `:640`, `:646`, `:652`) — a deferred send is a REFUSAL on the L0
/// and LX→LX paths rather than a soft sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L0LxSyncToLower {
    /// `dataflow.sync_send`, carrying its `$wait_immediately_for_async_transfers`.
    Send(dataflow::AsyncTransferWait),
    /// `dataflow.sync_recv`.
    Recv,
    /// `dataflow.implicit_sync_on_streaming_buffer` — lowered as `sendrecv` (`:454`), carrying its
    /// already-resolved `$buffer_size`.
    ///
    /// ⛔⛔ RESOLVED BY THE CALLER, BECAUSE NO SINGLE-ISLAND WALK CAN ANSWER IT. The reference reads
    /// the size off the defining op and accepts TWO (`:206-215`, `:451-462`): an
    /// `arith.constant_index` — a [`DfirOp`] — or a `sentient.scalar_constant`, which is not one.
    /// [`crate::islands::dataflow_ir::dialects::defining_op`] walks a `&[DfirOp]` scope and so cannot
    /// see the second, and the `DT_ERROR("sync buffer size has to be a constant op")` on anything
    /// else is an abort. Reaching the operand is the mechanism a port may drop; the size is the input.
    ImplicitSync(i32),
}

impl L0LxSyncToLower {
    /// `sync_tag` — `send`, `recv` for a `sync_recv`, and `sendrecv` for an implicit sync.
    #[must_use]
    pub const fn mode(self) -> sen::SyncMode {
        match self {
            Self::Send(_) => sen::SyncMode::Send,
            Self::Recv => sen::SyncMode::Recv,
            Self::ImplicitSync(_) => sen::SyncMode::SendRecv,
        }
    }

    /// `send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value()` — *"Unsuported sync
    /// operation for L0"* / *"for LX"*.
    #[must_use]
    pub const fn deferred(self) -> bool {
        matches!(self, Self::Send(dataflow::AsyncTransferWait::Deferred))
    }

    /// `implicit_sync_tile_size` — ⛔ [`None`] IS THE REFERENCE'S `-1`, both as its initial value
    /// (`:203`, `:450`) and as the `si32` every non-implicit sync writes.
    #[must_use]
    pub const fn tile_size(self) -> Option<i32> {
        match self {
            Self::Send(_) | Self::Recv => None,
            Self::ImplicitSync(size) => Some(size),
        }
    }
}

/// A DESTINATION AN L0/LX SYNC MAY NAME — the `type` and the `corelet` of the destination
/// `dataflow.get_unit`, which are the only two things either function reads it for.
///
/// ⛔⛔ THE CORELET IS MANDATORY ON EVERYTHING BUT L3, AND THAT GUARD IS THIS TYPE.
/// `dst_comp != L3LU && dst_comp != L3SU && !dst_unit->hasAttr("corelet")` is
/// *"Unknown corelet information for sentient"* (`:224-229`, per destination at `:500-505` and
/// `:557-560`), so [`Corelet`] sits on the two non-L3 cases and on neither L3 one — the refusal has
/// no input rather than an arm.
///
/// ⛔ AND THE NUMBERED SPELLINGS COLLAPSE INTO THE CORELET. `SenComponents::LXLU0`/`LXLU1`/`LXSU0`/
/// `LXSU1` reach `dst_unit_name.pop_back()` in the group lowering (`:565-567`) to strip the digit
/// back off; this island's [`DfirUnit`] has no numbered LX unit at all — the corelet is
/// [`crate::units::Residency`]'s, never the name's — so the digit is never on. In the per-unit
/// lowering those four spellings instead fall through to
/// `emitError("Unknown lowering of the LXLU/LXSU sync operation")` (`:369`), which is the same
/// answer this type gives for a destination that is neither an L0 half, an LX half nor an L3 half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L0LxSyncDst {
    /// An `l0lu`/`l0su` destination, with the `corelet` `:224` demands it carry — ⭐ WHOSE VALUE IS
    /// NEVER READ: only its presence is, and the emitted consumer is the half.
    L0(L0Half, Corelet),
    /// An `lxlu`/`lxsu` destination, with the corelet whose difference from the source's is what
    /// appends the `N`.
    Lx(LxHalf, Corelet),
    /// An `l3lu`/`l3su` destination — the one case exempt from carrying a corelet.
    L3(L3Half),
}

/// WHAT ONE L0/LX SYNC LOWERING EMITS.
///
/// ⛔⛔ THE TWO-REGION FORM HANDS ITS SYNCS BACK BESIDE THE REGIONS, NOT INSIDE THEM.
/// [`uniform::LocalRegion::body`](crate::islands::dataflow_ir::dialects::uniform::LocalRegion) is a
/// `Vec<`[`DfirOp`]`>` — the shared dialects are re-exported onto the Sentient rung, so one
/// `uniform.uniformize_regions` value is a DataflowIR op whichever rung holds it — and a
/// `sentient.sync` is not a [`DfirOp`]. The reference reaches the same place with two `OpBuilder`s
/// aimed into the regions (`:330-334`, `:632-636`); positioning a builder is exactly the mechanism a
/// port may drop, so the placement is the caller's and the ops are all here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L0LxLowering {
    /// One `sentient.sync` where the lowered op stood.
    One(sen::Op),
    /// A result-less `uniform.uniformize_regions` and the syncs its two regions hold.
    TwoRegions {
        /// [`create_uniform_regions_with_two_regions_no_result`]'s op, carrying only its two yields.
        regions: uniform::Op,
        /// The sync `builder_region0` writes — corelet 0's sources.
        region0: sen::Op,
        /// The sync `builder_region1` writes — corelet 1's sources.
        region1: sen::Op,
        /// The sync the OUTER builder writes for the group's L3 destinations (`:648-653`) — ⭐ ALWAYS
        /// [`None`] from the per-unit lowering, which has no third create.
        l3: Option<sen::Op>,
    },
}

/// `symbolizeSentientLoadConsumer("l0lu"/"l0su")` — `:257-258` and `:516-517`.
const fn l0_consumer(half: L0Half) -> sen::Consumer {
    match half {
        L0Half::Load => sen::Consumer::L0lu,
        L0Half::Store => sen::Consumer::L0su,
    }
}

/// `symbolizeSentientLoadConsumer("l3lu"/"l3su")` — the unextended L3 name both arms pass through.
const fn l3_consumer(half: L3Half) -> sen::Consumer {
    match half {
        L3Half::Load => sen::Consumer::L3lu,
        L3Half::Store => sen::Consumer::L3su,
    }
}

/// `dst_unit_name` for an LX destination, with the `"N"` a cross-corelet sync appends (`:305`,
/// `:336-339`, `:566`).
const fn lx_consumer(half: LxHalf, neighbour: bool) -> sen::Consumer {
    match (half, neighbour) {
        (LxHalf::Load, false) => sen::Consumer::Lxlu,
        (LxHalf::Load, true) => sen::Consumer::LxluN,
        (LxHalf::Store, false) => sen::Consumer::Lxsu,
        (LxHalf::Store, true) => sen::Consumer::LxsuN,
    }
}

/// `sentient::SyncOp::create(..)` — ⭐ `soft` IS `false` AT EVERY CREATE IN BOTH UNITS.
fn sync(
    mode: sen::SyncMode,
    peers: Vec<sen::Consumer>,
    implicit_sync_memory_boundary: Option<i32>,
    dbg_name: Option<String>,
) -> sen::Op {
    sen::Op::Sync {
        mode,
        peers: peers
            .into_iter()
            .map(sen::SyncHalf::of_lowered_destination)
            .collect(),
        soft: false,
        implicit_sync_memory_boundary,
        dbg_name,
    }
}

impl L0LxSrc {
    /// Replaces: e273_lowerL0LXSyncOperationForAUnit
    ///
    /// **273/384** `lowerL0LXSyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189` (180L).
    ///
    /// One `sentient.sync` naming one destination — two of them, split into a
    /// `uniform.uniformize_regions`, when both of the source core's corelets hold a source unit.
    ///
    /// ⛔ EVERY REFUSAL IS AN `emitError`, WHICH FAILS THE PASS, so [`None`] is what each of the
    /// reference's diagnostic arms means — including the ones that `break` past a create.
    /// ⛔ TRAP: THE LX→L3 ARM NEVER READS THE WAIT FLAG (`:279-292` sits before the `:294` check), so
    /// a DEFERRED send from an LX to an L3 still emits `soft = false`; only the L0 and LX→LX arms
    /// refuse one.
    #[must_use]
    pub fn lower_l0lx_sync_operation_for_a_unit(
        self,
        op: L0LxSyncToLower,
        src_units_corelet0: &[Val],
        src_units_corelet1: &[Val],
        dst: L0LxSyncDst,
        dbg_name: Option<String>,
        values: &mut Values,
    ) -> Option<L0LxLowering> {
        // `:237-247` — "List of src units cannot be empty", and the `DT_CHECK` that the two lists
        // name one component, which taking `self` instead of the lists discharges.
        let occupied = OccupiedCorelets::of(src_units_corelet0, src_units_corelet1)?;
        let mode = op.mode();

        match self {
            L0LxSrc::L0(src_half) => {
                // `:249-252` — "Unsuported sync operation for L0".
                if op.deferred() {
                    return None;
                }
                // `:255-258` — the cross-half guard, then the name off the DESTINATION's half.
                // Anything else is `emitError("Unknown lowering of the L0 sync operation")` (`:273`).
                let L0LxSyncDst::L0(dst_half, _) = dst else {
                    return None;
                };
                if dst_half == src_half {
                    return None;
                }
                Some(L0LxLowering::One(sync(
                    mode,
                    vec![l0_consumer(dst_half)],
                    op.tile_size(),
                    dbg_name,
                )))
            }
            L0LxSrc::Lx(_) => {
                // `DT_CHECK_MSG(implicit_sync_tile_size == -1, "LX doesn't have implicit sync")`
                // (`:277-278`).
                if op.tile_size().is_some() {
                    return None;
                }
                match dst {
                    // `:279-288` — the L3 destination keeps its bare name and its `-1` boundary.
                    L0LxSyncDst::L3(half) => Some(L0LxLowering::One(sync(
                        mode,
                        vec![l3_consumer(half)],
                        None,
                        dbg_name,
                    ))),
                    L0LxSyncDst::Lx(dst_half, dst_corelet) => {
                        // `:293-296` — "Unsuported sync operation for LX".
                        if op.deferred() {
                            return None;
                        }
                        match occupied {
                            // `:299-317` — one corelet holds every source, so the sync is one op and
                            // the `N` says whether the destination sits on the OTHER corelet.
                            OccupiedCorelets::Corelet0 | OccupiedCorelets::Corelet1 => {
                                let neighbour = (occupied.holds_corelet_0()
                                    && dst_corelet.get() == 1)
                                    || (occupied.holds_corelet_1() && dst_corelet.get() == 0);
                                Some(L0LxLowering::One(sync(
                                    mode,
                                    vec![lx_consumer(dst_half, neighbour)],
                                    None,
                                    dbg_name,
                                )))
                            }
                            // `:320-363` — both corelets have sources, so the sync is uniformized
                            // and each region names the destination from ITS corelet's point of view.
                            OccupiedCorelets::Both => {
                                let regions = create_uniform_regions_with_two_regions_no_result(
                                    src_units_corelet0,
                                    src_units_corelet1,
                                    values,
                                );
                                // `:336-339` — a corelet-1 destination is the neighbour of region 0's
                                // sources and the local unit of region 1's; a corelet-0 destination
                                // is the mirror (the reference's bare `else`).
                                let dst_on_corelet1 = dst_corelet.get() == 1;
                                Some(L0LxLowering::TwoRegions {
                                    regions,
                                    region0: sync(
                                        mode,
                                        vec![lx_consumer(dst_half, dst_on_corelet1)],
                                        None,
                                        dbg_name.clone(),
                                    ),
                                    region1: sync(
                                        mode,
                                        vec![lx_consumer(dst_half, !dst_on_corelet1)],
                                        None,
                                        dbg_name,
                                    ),
                                    l3: None,
                                })
                            }
                        }
                    }
                    // `:366` — "Unknown lowering of the LXLU/LXSU sync operation".
                    L0LxSyncDst::L0(..) => None,
                }
            }
        }
    }

    /// Replaces: e274_lowerL0LXSyncOperationForAGroupOfUnits
    ///
    /// **274/384** `lowerL0LXSyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438` (222L).
    ///
    /// The same lowering over a whole `dataflow.create_group`, deduplicated in first-occurrence order
    /// ([`push_back_the_unit_to_list_if_doesnot_exist`]) into one peer list per source corelet.
    ///
    /// ⛔⛔ DELIBERATE DIVERGENCE (1): `:645` GIVES REGION 1 CORELET **0**'S LIST. The two-region arm
    /// builds `dst_unit_names_for_src_corelet1` at `:572`/`:580` and then never reads it — both
    /// `SyncOp::create`s pass `dst_unit_names_for_src_corelet0` — so every corelet-1 source syncs with
    /// the units its NEIGHBOUR should. The per-unit twin does this correctly (`:342-362`), and the
    /// vendor's own key prints the two lists as distinct and reversed
    /// (`Conversion/DataflowToSentient/uniform_sync.mlir:247` against `:255`). Region 1 gets its own
    /// list here; the test pins both.
    ///
    /// ⛔ DELIBERATE DIVERGENCE (2): the third create (`:648-653`) is unconditional, so a group with
    /// no L3 destination emits a `sentient.sync` naming NOBODY. It is [`None`] here.
    #[must_use]
    pub fn lower_l0lx_sync_operation_for_a_group_of_units(
        self,
        op: L0LxSyncToLower,
        src_units_corelet0: &[Val],
        src_units_corelet1: &[Val],
        dst_unit_group: &[L0LxSyncDst],
        dbg_name: Option<String>,
        values: &mut Values,
    ) -> Option<L0LxLowering> {
        // `:473-485` — "List of src units cannot be empty" and the two-list `DT_CHECK`.
        let occupied = OccupiedCorelets::of(src_units_corelet0, src_units_corelet1)?;
        let mode = op.mode();

        match self {
            L0LxSrc::L0(src_half) => {
                // `:493-497` — "Unsuported sync operation for L0".
                if op.deferred() {
                    return None;
                }
                let mut peers = Vec::new();
                for dst in dst_unit_group {
                    // `:510-513` — "Unknown lowering of the L0 sync operation", which an L3 or LX
                    // destination takes too: neither is the other L0 half.
                    let L0LxSyncDst::L0(dst_half, _) = dst else {
                        return None;
                    };
                    if *dst_half == src_half {
                        return None;
                    }
                    // `:515-523`.
                    push_back_the_unit_to_list_if_doesnot_exist(l0_consumer(*dst_half), &mut peers);
                }
                // `:525-531` — ⭐ ONE sync for the whole group, carrying the implicit-sync boundary.
                Some(L0LxLowering::One(sync(
                    mode,
                    peers,
                    op.tile_size(),
                    dbg_name,
                )))
            }
            L0LxSrc::Lx(_) => {
                // `:539-544` — the `DT_CHECK_MSG` then "Unsuported sync operation for LX".
                if op.tile_size().is_some() || op.deferred() {
                    return None;
                }
                let mut for_corelet0 = Vec::new();
                let mut for_corelet1 = Vec::new();
                let mut for_l3 = Vec::new();
                for dst in dst_unit_group {
                    match dst {
                        // `:554-582` — each LX destination joins BOTH lists, local on its own
                        // corelet's and neighbour on the other's.
                        L0LxSyncDst::Lx(half, corelet) => {
                            let local = lx_consumer(*half, false);
                            let neighbour = lx_consumer(*half, true);
                            let (in_corelet0, in_corelet1) = if corelet.get() == 0 {
                                (local, neighbour)
                            } else {
                                (neighbour, local)
                            };
                            push_back_the_unit_to_list_if_doesnot_exist(
                                in_corelet0,
                                &mut for_corelet0,
                            );
                            push_back_the_unit_to_list_if_doesnot_exist(
                                in_corelet1,
                                &mut for_corelet1,
                            );
                        }
                        // `:583-586`.
                        L0LxSyncDst::L3(half) => push_back_the_unit_to_list_if_doesnot_exist(
                            l3_consumer(*half),
                            &mut for_l3,
                        ),
                        // `:588-590` — "Unknown lowering of the LX sync operation".
                        L0LxSyncDst::L0(..) => return None,
                    }
                }
                match occupied {
                    // `:593-607` and `:608-622` — one corelet's list, then the L3 destinations, in
                    // that order.
                    OccupiedCorelets::Corelet0 | OccupiedCorelets::Corelet1 => {
                        let mut peers = if occupied.holds_corelet_0() {
                            for_corelet0
                        } else {
                            for_corelet1
                        };
                        peers.extend(for_l3);
                        Some(L0LxLowering::One(sync(mode, peers, None, dbg_name)))
                    }
                    // `:623-654` — two regions plus one outer sync for the L3 destinations, which
                    // stay OUTSIDE the uniformization because they are the same unit from both
                    // corelets.
                    OccupiedCorelets::Both => {
                        let regions = create_uniform_regions_with_two_regions_no_result(
                            src_units_corelet0,
                            src_units_corelet1,
                            values,
                        );
                        Some(L0LxLowering::TwoRegions {
                            regions,
                            region0: sync(mode, for_corelet0, None, dbg_name.clone()),
                            region1: sync(mode, for_corelet1, None, dbg_name.clone()),
                            l3: (!for_l3.is_empty()).then(|| sync(mode, for_l3, None, dbg_name)),
                        })
                    }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 300/384 + 301/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH FAMILY OF UNIT A SYNC LEAVES FROM — `src_unit_name.substr(0, 2) != "l3"`, which is the whole
/// body of both dispatchers.
///
/// ⛔ THE NAME IS NOT CARRIED, THE CHOICE IS. `src_unit_name` reaches them from
/// [`unit_name_from_a_list_of_get_unit_op`] and is read for nothing but those two characters; WHICH
/// half of the L3 (resp. which L0/LX component) the source is, the callee derives from `src_unit_ops`
/// for itself (`DataflowToSentient.cpp:400`, `:247`). Both facts come off one `get_unit`, so they are
/// one value here — and `src_unit_ops` disappears with the name, being the list [`L3Half`] already
/// stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncSrc {
    /// A source whose `type` does not begin `l3`.
    L0Lx(L0LxSrc),
    /// `l3lu`/`l3su`.
    L3(L3Half),
}

/// The L3 twins' view of the sync being lowered — ⛔ [`None`] IS THEIR `DT_CHECK(send_op || recv_op)`:
/// an implicit sync on a streaming buffer is an input the L0/LX pair has and they abort on.
const fn l3_sync_to_lower(op: L0LxSyncToLower) -> Option<SyncToLower> {
    match op {
        L0LxSyncToLower::Send(wait) => Some(SyncToLower::Send(wait)),
        L0LxSyncToLower::Recv => Some(SyncToLower::Recv),
        L0LxSyncToLower::ImplicitSync(_) => None,
    }
}

/// The L3 twins' view of one destination — ⛔ [`None`] IS `emitError("Unknown lowering of the L3 sync
/// operation")`, whose input is an L0 destination; and for a group it is the `break` [`L3SyncDst`]
/// made unrepresentable, so the caller refuses where the reference emits a truncated peer list.
const fn l3_sync_dst(dst: L0LxSyncDst) -> Option<L3SyncDst> {
    match dst {
        L0LxSyncDst::L3(L3Half::Load) => Some(L3SyncDst::L3lu),
        L0LxSyncDst::L3(L3Half::Store) => Some(L3SyncDst::L3su),
        L0LxSyncDst::Lx(half, corelet) => Some(L3SyncDst::Lx(half, corelet)),
        L0LxSyncDst::L0(..) => None,
    }
}

/// Replaces: e300_lowerSyncForAUnit
///
/// **300/384** `lowerSyncForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733` (7L).
///
/// The per-unit L3 or L0/LX sync lowering, chosen by the source's family.
///
/// ⛔ THE L3 ARM TAKES A NARROWER INPUT THAN THE L0/LX ONE — an implicit sync and an L0 destination
/// are aborts there — so the one [`Option`] covers both arms' refusals.
#[must_use]
pub fn lower_sync_for_a_unit(
    src: SyncSrc,
    op: L0LxSyncToLower,
    src_units_corelet0: &[Val],
    src_units_corelet1: &[Val],
    dst: L0LxSyncDst,
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<L0LxLowering> {
    match src {
        // `:740-742`.
        SyncSrc::L0Lx(src) => src.lower_l0lx_sync_operation_for_a_unit(
            op,
            src_units_corelet0,
            src_units_corelet1,
            dst,
            dbg_name,
            values,
        ),
        // `:744` — ⭐ WHICH CORELETS HOLD A SOURCE IS NOT READ on this side: an L3 sync names its
        // destination outright.
        SyncSrc::L3(half) => Some(L0LxLowering::One(half.lower_l3_sync_operation_for_a_unit(
            l3_sync_to_lower(op)?,
            l3_sync_dst(dst)?,
            dbg_name,
        ))),
    }
}

/// Replaces: e301_lowerSyncForAGroup
///
/// **301/384** `lowerSyncForAGroup` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746` (8L).
///
/// [`lower_sync_for_a_unit`]'s dispatch over a whole `dataflow.create_group`.
///
/// ⛔ THE GROUP IS RESOLVED ONCE FOR BOTH ARMS. The reference hands each callee the `CreateGroupOp`
/// and each walks `getUnitIds()` itself, so a destination neither accepts is refused per arm.
#[must_use]
pub fn lower_sync_for_a_group(
    src: SyncSrc,
    op: L0LxSyncToLower,
    src_units_corelet0: &[Val],
    src_units_corelet1: &[Val],
    dst_unit_group: &[L0LxSyncDst],
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<L0LxLowering> {
    match src {
        // `:753-756`.
        SyncSrc::L0Lx(src) => src.lower_l0lx_sync_operation_for_a_group_of_units(
            op,
            src_units_corelet0,
            src_units_corelet1,
            dst_unit_group,
            dbg_name,
            values,
        ),
        // `:757-758`.
        SyncSrc::L3(half) => {
            let mut group = Vec::with_capacity(dst_unit_group.len());
            for dst in dst_unit_group {
                group.push(l3_sync_dst(*dst)?);
            }
            Some(L0LxLowering::One(
                half.lower_l3_sync_operation_for_a_group_of_units(
                    l3_sync_to_lower(op)?,
                    &group,
                    dbg_name,
                ),
            ))
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 302/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT `lowerSyncLXL3ToLXL3` EMITS.
///
/// ⛔ N REGIONS, NOT [`L0LxLowering::TwoRegions`]'S TWO: the group arms open one region per
/// `create_group` destination, so the count is an input's length. ⭐ AND THE SYNCS SIT BESIDE THE
/// REGIONS for the reason [`L0LxLowering`] gives — a `sentient.sync` is not a [`DfirOp`], so region
/// `i`'s sync is `syncs[i]` and [`uniformized`] is the only thing that builds the pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LxL3Lowering {
    /// One `sentient.sync` where the lowered op stood, with nothing uniformized.
    One(sen::Op),
    /// One region per bucket, each holding exactly one sync.
    Regions {
        /// The result-less `uniform.uniformize_regions`, whose regions carry only their yields.
        regions: uniform::Op,
        /// Region `i`'s `sentient.sync`.
        syncs: Vec<sen::Op>,
    },
}

/// `UniformizeRegionsOp::create(.., sorted_units, list_sizes, .., num_of_regions)` — the N-region
/// form of [`create_uniform_regions_with_two_regions_no_result`], whose `sorted_units` is the
/// concatenation of the per-region unit lists and whose `list_sizes` are their lengths.
fn uniformized(per_region: Vec<(Vec<Val>, sen::Op)>, values: &mut Values) -> LxL3Lowering {
    let mut regions = Vec::with_capacity(per_region.len());
    let mut syncs = Vec::with_capacity(per_region.len());
    for (units, sync) in per_region {
        regions.push(uniform::LocalRegion {
            arg: values.mint(),
            units,
            body: vec![DfirOp::Uniform(uniform::Op::Yield {
                operands: Vec::new(),
            })],
        });
        syncs.push(sync);
    }
    LxL3Lowering::Regions {
        regions: uniform::Op::UniformizeRegions {
            regions,
            results: Vec::new(),
        },
        syncs,
    }
}

/// `llvm::dyn_cast<dataflow::GetUnitOp>(v.getDefiningOp())` and the `type`/`corelet` the sync
/// lowerings read off it. ⛔ [`None`] IS BOTH `DT_CHECK(dst_op)` — a destination that is not a
/// `get_unit` — and *"Unknown corelet information for sentient"*.
fn sync_destination(v: Val, scope: &[DfirOp]) -> Option<L0LxSyncDst> {
    let Some(DfirOp::Dataflow(dataflow::Op::GetUnit {
        residency, unit, ..
    })) = defining_op(v, scope)
    else {
        return None;
    };
    // `dst_unit->hasAttr("corelet")` — and `CoreWide` prints `corelet = 0`, as
    // [`are_corelets_different`] says.
    let corelet = match residency {
        Residency::Corelet { corelet, .. } => Some(*corelet),
        Residency::CoreWide { .. } => Corelet::checked(0),
        Residency::Scratchpad { .. } | Residency::Global => None,
    };
    match unit {
        DfirUnit::L0lu => Some(L0LxSyncDst::L0(L0Half::Load, corelet?)),
        DfirUnit::L0su => Some(L0LxSyncDst::L0(L0Half::Store, corelet?)),
        DfirUnit::Lxlu => Some(L0LxSyncDst::Lx(LxHalf::Load, corelet?)),
        DfirUnit::Lxsu => Some(L0LxSyncDst::Lx(LxHalf::Store, corelet?)),
        DfirUnit::L3lu => Some(L0LxSyncDst::L3(L3Half::Load)),
        DfirUnit::L3su => Some(L0LxSyncDst::L3(L3Half::Store)),
        // Every other spelling is `emitError("Unknown lowering of the … sync operation")`, spelled
        // out so that a new unit cannot join the refusal side unlooked-at.
        DfirUnit::Sfp
        | DfirUnit::Pe
        | DfirUnit::PtRow(_)
        | DfirUnit::Lx
        | DfirUnit::Hbm
        | DfirUnit::L0
        | DfirUnit::Constant
        | DfirUnit::SfpState
        | DfirUnit::PeState
        | DfirUnit::SfpRing
        | DfirUnit::LxVirtualIbr
        | DfirUnit::L3Ibr
        | DfirUnit::CrossPtnLink
        | DfirUnit::LxluScaleReg => None,
    }
}

/// `group_op.getUnitIds()` with every member's `dyn_cast` resolved — ⛔ [`None`] IS THE NULL
/// `group_op` THE REFERENCE PASSES ON WITHOUT A CHECK, which is a destination the bucketing sorted as
/// a group and this walk finds is not one.
fn sync_group_destinations(v: Val, scope: &[DfirOp]) -> Option<Vec<L0LxSyncDst>> {
    let Some(DfirOp::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) = defining_op(v, scope)
    else {
        return None;
    };
    let mut group = Vec::with_capacity(unit_ids.len());
    for member in unit_ids {
        group.push(sync_destination(*member, scope)?);
    }
    Some(group)
}

/// `sameGroup` — ⛔⛔ THE TEST IS INVERTED, SO A NON-EMPTY LIST ALWAYS ANSWERS `false`:
/// `if (u.getDefiningOp() == dst_vs[0].getDefiningOp()) sameGroup = false;` compares the first
/// destination with ITSELF on the first iteration where `!=` was plainly meant. Reproduced — its arm
/// is reachable only through [`all_keys_folds`], and in the two mixed-with-group arms, which have no
/// such override, not at all.
fn same_group(dsts: &[Val], scope: &[DfirOp]) -> bool {
    let mut same = true;
    // An empty list never enters the loop; the reference would read `dst_vs[0]` out of bounds.
    let Some(first) = dsts.first() else {
        return same;
    };
    let first = defining_position(*first, scope);
    for dst in dsts {
        if defining_position(*dst, scope) == first {
            same = false;
        }
    }
    same
}

/// `all_keys_folds` — every source is a result of ONE `get_unit`, so the map's keys are folds of one
/// unit and one instance of the fold stands for all of them.
fn all_keys_folds(srcs: &[Val], scope: &[DfirOp]) -> bool {
    let Some(first) = srcs.first() else {
        return true;
    };
    let first = defining_position(*first, scope);
    srcs.iter()
        .all(|src| defining_position(*src, scope) == first)
}

/// The reference's `is_src_l3`/`corelet_id` three-way at each of this function's per-unit call sites,
/// which is [`lower_sync_for_a_unit`] with the source list picked by `corelet_id`. ⛔ ONE SOURCE SITS
/// ON ONE CORELET, so [`OccupiedCorelets::Both`] and its two regions have no input here.
fn one_unit_sync(
    src: SyncSrc,
    op: L0LxSyncToLower,
    corelet_id: Corelet,
    pair: (Val, Val),
    scope: &[DfirOp],
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<sen::Op> {
    let (src_v, dst_v) = pair;
    let one = [src_v];
    let (corelet0, corelet1): (&[Val], &[Val]) = if corelet_id.get() == 0 {
        (&one, &[])
    } else {
        (&[], &one)
    };
    match lower_sync_for_a_unit(
        src,
        op,
        corelet0,
        corelet1,
        sync_destination(dst_v, scope)?,
        dbg_name,
        values,
    )? {
        L0LxLowering::One(sync) => Some(sync),
        L0LxLowering::TwoRegions { .. } => None,
    }
}

/// [`one_unit_sync`] for a `create_group` destination — the same dispatch, through
/// [`lower_sync_for_a_group`].
fn one_group_sync(
    src: SyncSrc,
    op: L0LxSyncToLower,
    corelet_id: Corelet,
    pair: (Val, Val),
    scope: &[DfirOp],
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<sen::Op> {
    let (src_v, dst_v) = pair;
    let one = [src_v];
    let (corelet0, corelet1): (&[Val], &[Val]) = if corelet_id.get() == 0 {
        (&one, &[])
    } else {
        (&[], &one)
    };
    match lower_sync_for_a_group(
        src,
        op,
        corelet0,
        corelet1,
        &sync_group_destinations(dst_v, scope)?,
        dbg_name,
        values,
    )? {
        L0LxLowering::One(sync) => Some(sync),
        L0LxLowering::TwoRegions { .. } => None,
    }
}

/// Replaces: e302_lowerSyncLXL3ToLXL3
///
/// **302/384** `lowerSyncLXL3ToLXL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787` (928L).
///
/// One sync when the destinations share a bucket of [`separate_based_on_destination_units`], else one
/// `uniform.uniformize_regions` region per non-empty bucket — LX-corelet-0, LX-corelet-1, L3, then
/// the groups — each holding ONE sync lowered from that bucket's first pair alone.
///
/// ⛔ TRAP: `corelet_id` IS THE SOURCE'S CORELET AND IS THE SAME IN EVERY REGION, so whether a region
/// names its destination `lxlu` or `lxluN` is that destination's corelet against `corelet_id`.
#[must_use]
pub fn lower_sync_lx_l3_to_lx_l3(
    src: SyncSrc,
    op: L0LxSyncToLower,
    src_dst: &[(Val, Val)],
    corelet_id: Corelet,
    scope: &[DfirOp],
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<LxL3Lowering> {
    let separated = separate_based_on_destination_units(src_dst, scope);
    let buckets: Vec<&[(Val, Val)]> = [
        separated.lx_corelet0.as_slice(),
        separated.lx_corelet1.as_slice(),
        separated.l3.as_slice(),
    ]
    .into_iter()
    .filter(|bucket| !bucket.is_empty())
    .collect();
    let group = separated.group.as_slice();

    // `DT_CHECK(src_dst_lx_corelet0.size() != 0 || …)` — every destination was dropped.
    if buckets.is_empty() && group.is_empty() {
        return None;
    }

    // The three-way disjunction that is "exactly one unit bucket, and no group". ⛔ IT LOWERS
    // `src_vs[0]`/`dst_vs[0]`, THE GLOBAL FIRST PAIR — the bucket's own first only because a dropped
    // pair would have left the reference's `dst_op` null under a `DT_CHECK`. So a bucket of eight
    // destinations emits ONE sync naming the first of them.
    if group.is_empty() && buckets.len() == 1 {
        let pair = *src_dst.first()?;
        return Some(LxL3Lowering::One(one_unit_sync(
            src, op, corelet_id, pair, scope, dbg_name, values,
        )?));
    }

    // The chain's final `return LogicalResult::failure()`: all four buckets non-empty is the one
    // arrangement no arm claims.
    if !group.is_empty() && buckets.len() == 3 {
        return None;
    }

    // ⭐ THE GROUP-ONLY ARM READS THE GLOBAL LISTS, both for `sameGroup` and for the region it
    // lowers, where the mixed arms read the group bucket's own pairs.
    let group_only = buckets.is_empty();
    let group_pairs = if group_only { src_dst } else { group };
    let group_dsts: Vec<Val> = group_pairs.iter().map(|(_, dst_v)| *dst_v).collect();
    let mut same = same_group(&group_dsts, scope);
    // `if (all_keys_folds) sameGroup = true;` — the override exists in the group-only arm alone, and
    // is the only path into a same-group lowering anywhere in this function.
    if group_only {
        let srcs: Vec<Val> = src_dst.iter().map(|(src_v, _)| *src_v).collect();
        if all_keys_folds(&srcs, scope) {
            same = true;
        }
    }

    // One group sync where the op stood: one fold instance stands for all of them, so nothing is
    // uniformized and no nested uniform regions are introduced.
    if group_only && same {
        let pair = *src_dst.first()?;
        return Some(LxL3Lowering::One(one_group_sync(
            src, op, corelet_id, pair, scope, dbg_name, values,
        )?));
    }

    let mut per_region: Vec<(Vec<Val>, sen::Op)> = Vec::new();
    for bucket in buckets {
        let pair = *bucket.first()?;
        per_region.push((
            bucket.iter().map(|(src_v, _)| *src_v).collect(),
            one_unit_sync(src, op, corelet_id, pair, scope, dbg_name.clone(), values)?,
        ));
    }
    if !group.is_empty() {
        if same {
            // ⚠️ THE REFERENCE'S DEAD ARM, KEPT: `sameGroup` gives every group destination ONE
            // shared region lowered from the first pair, and no arm that reaches here can set it.
            let pair = *group_pairs.first()?;
            per_region.push((
                group_pairs.iter().map(|(src_v, _)| *src_v).collect(),
                one_group_sync(src, op, corelet_id, pair, scope, dbg_name.clone(), values)?,
            ));
        } else {
            for pair in group_pairs {
                per_region.push((
                    vec![pair.0],
                    one_group_sync(src, op, corelet_id, *pair, scope, dbg_name.clone(), values)?,
                ));
            }
        }
    }

    Some(uniformized(per_region, values))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 319/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ONE QUERY-MAP SYNC LOWERING EMITS.
///
/// ⛔ THE TWO-REGION FORM HOLDS A WHOLE [`LxL3Lowering`] PER REGION, which may itself be a nested
/// `uniform.uniformize_regions` — and its syncs sit beside the regions for the reason
/// [`L0LxLowering`] gives: a `sentient.sync` is not a [`DfirOp`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryMapLowering {
    /// The L0 arm's one `sentient.sync` (`:1813-1822`).
    One(sen::Op),
    /// Every arm that delegates once: one corelet holds every source, the target is the L3, or the
    /// source is an L3 half.
    Delegated(LxL3Lowering),
    /// `:1847-1886` — a result-less two-region `uniform.uniformize_regions`, region *i* lowering
    /// corelet *i*'s pairs with `corelet_id = i`.
    TwoRegions {
        /// [`create_uniform_regions_with_two_regions_no_result`]'s op, carrying only its two yields.
        regions: uniform::Op,
        /// What `builder_region0` writes.
        region0: LxL3Lowering,
        /// What `builder_region1` writes.
        region1: LxL3Lowering,
    },
}

/// `separateKeyValuesBasedOnKeysCorelet`'s per-key corelet (`Dialect/Uniform/Utils.cpp:459-466`) —
/// ⛔ [`None`] IS BOTH `emitError("Unknown corelet information for sentient")` and the null
/// dereference a key that is not a `get_unit` walks into. `DT_CHECK(corelet_id == 1)` on the else
/// side is [`Corelet`]'s own bound.
fn key_corelet(v: Val, scope: &[DfirOp]) -> Option<Corelet> {
    let Some(DfirOp::Dataflow(dataflow::Op::GetUnit { residency, .. })) = defining_op(v, scope)
    else {
        return None;
    };
    match residency {
        Residency::Corelet { corelet, .. } => Some(*corelet),
        // `CoreWide` prints `corelet = 0`, as [`are_corelets_different`] says.
        Residency::CoreWide { .. } => Corelet::checked(0),
        Residency::Scratchpad { .. } | Residency::Global => None,
    }
}

/// `dyn_cast<dataflow::GetUnitOp>(v.getDefiningOp())` — the unit type alone.
fn queried_unit(v: Val, scope: &[DfirOp]) -> Option<DfirUnit> {
    let Some(DfirOp::Dataflow(dataflow::Op::GetUnit { unit, .. })) = defining_op(v, scope) else {
        return None;
    };
    Some(*unit)
}

/// One L0 destination (`:1786-1802`) — a `get_unit`, or the sole member of a `dataflow.create_group`.
///
/// ⛔ [`None`] IS BOTH `DT_CHECK_MSG(getUnitIds().size() == 1, "For L0 units the target cannot be a
/// group of multiple units")` and `DT_CHECK_MSG(target_unit, "…has to be a single unit.")`.
fn l0_target_unit(v: Val, scope: &[DfirOp]) -> Option<DfirUnit> {
    if let Some(unit) = queried_unit(v, scope) {
        return Some(unit);
    }
    let Some(DfirOp::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) = defining_op(v, scope)
    else {
        return None;
    };
    let [member] = unit_ids.as_slice() else {
        return None;
    };
    queried_unit(*member, scope)
}

/// Replaces: e319_lowerSyncForAQueryMap
///
/// **319/384** `DataflowToSentient.cpp:1728` (166L) — one `sentient.sync` for an L0 source, else
/// [`lower_sync_lx_l3_to_lx_l3`] over the map's pairs split by the KEY's corelet: once, or twice
/// inside a two-region `uniform.uniformize_regions` when both corelets hold a source.
///
/// ⛔ TRAP: `corelet_id = -1` IS NOT A THIRD CASE — `lowerSyncLXL3ToLXL3` tests `corelet_id == 0`
/// and takes its `else`, so the merged target-L3 arm lowers as corelet **1** (`:815-823`). ⭐ And
/// the three `src_unit_ops*` parameters are never read; the split is re-derived from the map.
#[must_use]
pub fn lower_sync_for_a_query_map(
    src: SyncSrc,
    op: L0LxSyncToLower,
    src_dst: &[(Val, Val)],
    scope: &[DfirOp],
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<QueryMapLowering> {
    // `if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().has_value())` — "Unknown async
    // transfers modes for sentient". [`L0LxSyncToLower::Send`] carries the value, so the ABSENT
    // attribute has no spelling and the refusal has no input.
    let l0lx = match src {
        // `else` (`:1892-1898`) — `DT_CHECK_MSG(implicit_sync_tile_size == -1, "L3 doesn't have
        // implicit sync")`, then one merged lowering with `is_src_l3 = true`.
        SyncSrc::L3(_) => {
            if op.tile_size().is_some() {
                return None;
            }
            return Some(QueryMapLowering::Delegated(lower_sync_lx_l3_to_lx_l3(
                src,
                op,
                src_dst,
                // `corelet_id = -1`, unread on the path `is_src_l3` takes first.
                Corelet::checked(1)?,
                scope,
                dbg_name,
                values,
            )?));
        }
        SyncSrc::L0Lx(l0lx) => l0lx,
    };

    // `separateKeyValuesBasedOnKeysCorelet(query_map, …)` — the pairs split by the KEY's corelet.
    let mut corelet0: Vec<(Val, Val)> = Vec::new();
    let mut corelet1: Vec<(Val, Val)> = Vec::new();
    for pair in src_dst {
        if key_corelet(pair.0, scope)?.get() == 0 {
            corelet0.push(*pair);
        } else {
            corelet1.push(*pair);
        }
    }
    // `DT_CHECK(src_corelet0_vs.size() + src_corelet1_vs.size() != 0)`.
    if corelet0.is_empty() && corelet1.is_empty() {
        return None;
    }

    match l0lx {
        L0LxSrc::L0(src_half) => {
            // `for (int i = 0; i < dst_vs.size(); i++)` — ⛔ THE LOOP ONLY GUARDS. `dst_unit_name` is
            // overwritten per destination and the last one wins, which is harmless because the
            // cross-half test forces every destination onto the half the source is not.
            let mut dst_half = None;
            for (_, dst_v) in src_dst {
                let unit = l0_target_unit(*dst_v, scope)?;
                // `if (!((isSenComponentL0LU(src_comp) && isSenComponentL0SU(dst_comp)) ||
                // (isSenComponentL0SU(src_comp) && isSenComponentL0LU(dst_comp))))` — "Unsupported
                // dst unit for L0 unit.". Then the name is the DESTINATION's half (`:1810-1811`).
                dst_half = Some(match src_half {
                    L0Half::Load if is_sen_component_l0su(unit) => L0Half::Store,
                    L0Half::Store if is_sen_component_l0lu(unit) => L0Half::Load,
                    L0Half::Load | L0Half::Store => return None,
                });
            }
            // ⛔ AN EMPTY VALUE LIST LEAVES `dst_unit_name` `""`, which
            // `symbolizeSentientLoadConsumer` has no case for and the reference `.value()`s.
            // ⭐ AND THIS ARM NEVER READS THE WAIT FLAG: a deferred send that entry 273 refuses
            // (`:249-252`) emits `soft = false` here.
            // ⚠️ `if (SyncOp::create(…)) return success();` HAS NO `break`, so a falsy create would
            // fall through into the LX case below; `create` never answers null.
            Some(QueryMapLowering::One(sync(
                op.mode(),
                vec![l0_consumer(dst_half?)],
                op.tile_size(),
                dbg_name,
            )))
        }
        L0LxSrc::Lx(_) => {
            // `DT_CHECK_MSG(implicit_sync_tile_size == -1, "LX doesn't have implicit sync")`.
            if op.tile_size().is_some() {
                return None;
            }
            // `if (src_corelet1_vs.size() == 0)` — every source sits on corelet 0.
            if corelet1.is_empty() {
                return Some(QueryMapLowering::Delegated(lower_sync_lx_l3_to_lx_l3(
                    src,
                    op,
                    &corelet0,
                    Corelet::checked(0)?,
                    scope,
                    dbg_name,
                    values,
                )?));
            }
            // `else if (src_corelet0_vs.size() == 0)`.
            if corelet0.is_empty() {
                return Some(QueryMapLowering::Delegated(lower_sync_lx_l3_to_lx_l3(
                    src,
                    op,
                    &corelet1,
                    Corelet::checked(1)?,
                    scope,
                    dbg_name,
                    values,
                )?));
            }
            // `else if (isTargetL3(query_map))` — entry 043, which reads the mapping's FIRST value
            // alone. ⛔ A `create_group` there leaves the helper's string PRESENT AND EMPTY, so
            // `substr(0, 2)` is `""`: not L3 (`Dialect/Uniform/Utils.cpp:258`).
            let target_l3 = match src_dst.first() {
                None => is_target_l3(&[]),
                Some((_, dst_v)) => match queried_unit(*dst_v, scope) {
                    Some(unit) => is_target_l3(&[unit]),
                    None => false,
                },
            };
            if target_l3 {
                // `src_unit_res`/`dst_unit_res` — corelet 0's pairs, then corelet 1's.
                let merged: Vec<(Val, Val)> = corelet0.iter().chain(&corelet1).copied().collect();
                return Some(QueryMapLowering::Delegated(lower_sync_lx_l3_to_lx_l3(
                    src,
                    op,
                    &merged,
                    Corelet::checked(1)?,
                    scope,
                    dbg_name,
                    values,
                )?));
            }
            // `:1847-1886` — both corelets hold a source and the target is not the L3.
            let keys = |pairs: &[(Val, Val)]| -> Vec<Val> {
                pairs.iter().map(|(src_v, _)| *src_v).collect()
            };
            let regions = create_uniform_regions_with_two_regions_no_result(
                &keys(&corelet0),
                &keys(&corelet1),
                values,
            );
            Some(QueryMapLowering::TwoRegions {
                regions,
                region0: lower_sync_lx_l3_to_lx_l3(
                    src,
                    op,
                    &corelet0,
                    Corelet::checked(0)?,
                    scope,
                    dbg_name.clone(),
                    values,
                )?,
                region1: lower_sync_lx_l3_to_lx_l3(
                    src,
                    op,
                    &corelet1,
                    Corelet::checked(1)?,
                    scope,
                    dbg_name,
                    values,
                )?,
            })
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 337/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ONE `dataflow` SYNC LOWERED TO — the two answer shapes its three dispatch targets have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncLowering {
    /// A `get_unit` or a `create_group` destination, through [`lower_sync_for_a_unit`] and
    /// [`lower_sync_for_a_group`].
    L0Lx(L0LxLowering),
    /// A `uniform.query_map` destination, through [`lower_sync_for_a_query_map`].
    QueryMap(QueryMapLowering),
}

/// The source's family, off the one `type` every source agreed on — ⛔ [`None`] IS
/// `stringToSenComponents.find(src_unit_name)->second` DEREFERENCING `end()`, which is what a source
/// that is no sync-capable unit reaches (`:245`, `:397`, `:487`, `:685`).
const fn sync_source(unit: DfirUnit) -> Option<SyncSrc> {
    match unit {
        DfirUnit::L0lu => Some(SyncSrc::L0Lx(L0LxSrc::L0(L0Half::Load))),
        DfirUnit::L0su => Some(SyncSrc::L0Lx(L0LxSrc::L0(L0Half::Store))),
        DfirUnit::Lxlu => Some(SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Load))),
        DfirUnit::Lxsu => Some(SyncSrc::L0Lx(L0LxSrc::Lx(LxHalf::Store))),
        // `src_unit_name.substr(0, 2) != "l3"` — the two spellings that fail it.
        DfirUnit::L3lu => Some(SyncSrc::L3(L3Half::Load)),
        DfirUnit::L3su => Some(SyncSrc::L3(L3Half::Store)),
        DfirUnit::Sfp
        | DfirUnit::Pe
        | DfirUnit::PtRow(_)
        | DfirUnit::Lx
        | DfirUnit::Hbm
        | DfirUnit::L0
        | DfirUnit::Constant
        | DfirUnit::SfpState
        | DfirUnit::PeState
        | DfirUnit::SfpRing
        | DfirUnit::LxVirtualIbr
        | DfirUnit::L3Ibr
        | DfirUnit::CrossPtnLink
        | DfirUnit::LxluScaleReg => None,
    }
}

/// Replaces: e337_lowerSyncOperation
///
/// **337/384** `DataflowToSentient.cpp:1901` (80L): the sync dispatcher — one source family for the
/// whole list, the sources split by corelet, and the destination's defining op picking the lowering.
///
/// ⛔ THE CORELET SPLIT IS NOT MADE FOR AN L3 SOURCE (`:1935`), so both lists reach an L3 lowering
/// empty — which is exactly what it reads them for: nothing.
/// ⛔⛔ `src_units` IS ALREADY NARROWED BY THE CALLER. `:1912-1925` replaces the operand list with the
/// units of the enclosing `uniform.uniformize_regions` region when the op sits in one; the enclosing
/// op is not in the `&[DfirOp]` scope holding a region's body, so no walk here can find it.
#[must_use]
pub fn lower_sync_operation(
    src_units: &[Val],
    op: L0LxSyncToLower,
    dst_units: Val,
    scope: &[DfirOp],
    dbg_name: Option<String>,
    values: &mut Values,
) -> Option<SyncLowering> {
    // `getUnitNameFromAListOfGetUnitOp` over `src_unit_ops`, whose `DT_CHECK(src_unit_op)` and
    // `DT_CHECK(src_units.size() > 0)` are the walk and the empty list.
    // ⭐ `getDirectUnitOpOrgetFirstIndirectUnitOpViaQueryMap(src_units[0])` (`:1905`) reads the name
    // off the FIRST source, resolving one behind a query map; every source is then checked against
    // it, so a list this walk resolves at all agrees on one kind either way.
    let mut units = Vec::with_capacity(src_units.len());
    for src_unit in src_units {
        units.push(queried_unit(*src_unit, scope)?);
    }
    let src = sync_source(unit_name_from_a_list_of_get_unit_op(&units)?)?;

    // `:1935-1946` — `corelet == 0` or, `DT_CHECK`ed by [`Corelet`], 1.
    let (mut corelet0, mut corelet1) = (Vec::new(), Vec::new());
    if !matches!(src, SyncSrc::L3(_)) {
        for src_unit in src_units {
            if key_corelet(*src_unit, scope)?.get() == 0 {
                corelet0.push(*src_unit);
            } else {
                corelet1.push(*src_unit);
            }
        }
    }

    // ⭐ WHICH OPERAND NAMES THE DESTINATION IS THE OP'S OWN (`:1949-1961`): `$to_unit` on a send,
    // `$from_unit` on a recv, `$dst_unit` on an implicit sync — one value once resolved.
    match defining_op(dst_units, scope)? {
        // `:1963-1967`.
        DfirOp::Dataflow(dataflow::Op::GetUnit { .. }) => lower_sync_for_a_unit(
            src,
            op,
            &corelet0,
            &corelet1,
            sync_destination(dst_units, scope)?,
            dbg_name,
            values,
        )
        .map(SyncLowering::L0Lx),
        // `:1969-1973`.
        DfirOp::Dataflow(dataflow::Op::CreateGroup { .. }) => lower_sync_for_a_group(
            src,
            op,
            &corelet0,
            &corelet1,
            &sync_group_destinations(dst_units, scope)?,
            dbg_name,
            values,
        )
        .map(SyncLowering::L0Lx),
        // `:1975-1979` — the map's pairs, which the reference's callee reads off the op itself.
        DfirOp::Uniform(uniform::Op::QueryMap { map, .. }) => {
            let Some(DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) =
                defining_op(*map, scope)
            else {
                return None;
            };
            lower_sync_for_a_query_map(src, op, pairs, scope, dbg_name, values)
                .map(SyncLowering::QueryMap)
        }
        // `return LogicalResult::failure();` (`:1981`) — a destination that is none of the three.
        _ => None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 361/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE `dataflow` OPERATION THE PASS REPLACED, IN WALK ORDER.
#[derive(Debug, Clone, PartialEq)]
pub enum LoweredDataflowOp<'a> {
    /// A `sync_send`, `sync_recv` or `implicit_sync_on_streaming_buffer` that lowered, and is
    /// therefore queued for erasure (`:2026`).
    Sync {
        /// The op replaced.
        op: &'a DfirOp,
        /// Entry 337's answer.
        lowering: SyncLowering,
    },
    /// `signalPassFailure()` (`:2027`) — entry 337 refused, and the op is NOT queued.
    SyncFailed {
        /// The op left standing.
        op: &'a DfirOp,
    },
    /// A `dataflow.opaque` that lowered (`:2031`).
    Opaque {
        /// The op replaced.
        op: &'a DfirOp,
        /// Entry 044's `sentient.opaque`.
        lowered: sen::Op,
    },
}

/// WHAT ONE `dataflow.program_unit` CAME OUT OF THE PASS AS.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitSyncLowering<'a> {
    /// The FIRST walk's sites, in walk order (`:2019-2034`).
    pub lowered: Vec<LoweredDataflowOp<'a>>,
    /// The SECOND walk's own list (`:2035-2038`) — `dataflow.create_group`s nothing reads.
    pub dead_groups: Vec<&'a DfirOp>,
}

/// Replaces: e361_runOnOperation
///
/// **361/384** `DataflowToSentientLoweringPass::runOnOperation` —
/// `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014` (33L). Entry 337 over every
/// sync and entry 044 over every opaque of every program unit, then the groups nothing reads.
///
/// ⛔⛔ THE SECOND WALK RUNS BEFORE ANY ERASE, so `op->use_empty()` (`:2036`) is asked while the syncs
/// that name a group are all still standing. A group addressed by a sync this pass just lowered is
/// therefore NOT dead here — only a group nothing ever referenced is.
/// ⛔ `auto unit = unit_op.getUnits()[0].getDefiningOp<GetUnitOp>();` (`:2018`) IS DEAD — bound and
/// never read.
/// ⛔ AND SO IS THE OPAQUE BRANCH'S BUILDER (`:2028-2029`): `lowerOpaqueOperation` takes the op alone
/// and makes its OWN `OpBuilder builder(opaque_op)` (`:2004`), so `setInsertionPointToStart` acts on a
/// builder that is never passed anywhere. The `sentient.opaque` lands where the `dataflow.opaque` was,
/// not at the top of the region — see [`lower_opaque_operation`].
/// ⛔ THE `CODEGEN_DUMP_IRS` DUMP (`:2042-2046`) IS NOT PORTED: an env-gated debug artefact.
#[must_use]
pub fn run_on_operation<'a, A: Arch>(
    program: &'a Program<A>,
    values: &mut Values,
) -> Vec<UnitSyncLowering<'a>> {
    let mut per_unit: Vec<UnitSyncLowering<'_>> = Vec::new();
    // `module_op.walk([&](dataflow::ProgramUnitOp unit_op) { .. })`.
    for unit in program.units.iter() {
        // ⛔ THE RESOLUTION SCOPE IS THE PREAMBLE **AND** THE BODY. A sync's sources are the
        // module-level `dataflow.get_unit`s the program unit takes as operands, while the group a
        // collective sync addresses is inside the unit — which is why the second walk looks for it
        // there (`:2035`). Entry 337 reads one `&[DfirOp]`, so the two are joined for it; the delete
        // lists below still name ops of `unit.body` itself.
        let scope: Vec<DfirOp> = program
            .preamble
            .iter()
            .chain(unit.body.iter())
            .cloned()
            .collect();
        let src_units = unit.on.vals();

        let mut sites: Vec<&DfirOp> = Vec::new();
        sync_and_opaque_sites(&unit.body, &mut sites);
        let mut lowered: Vec<LoweredDataflowOp<'_>> = Vec::new();
        for site in sites {
            // ⭐ WHICH OPERAND NAMES THE DESTINATION IS THE OP'S OWN, and the debug name is the
            // signal it carries — see [`L0LxSyncToLower`].
            let sync = match site {
                DfirOp::Dataflow(dataflow::Op::SyncSend { to, signal, wait }) => Some((
                    *to,
                    L0LxSyncToLower::Send(*wait),
                    Some(signal.spelling().to_owned()),
                )),
                DfirOp::Dataflow(dataflow::Op::SyncRecv { from, signal }) => Some((
                    *from,
                    L0LxSyncToLower::Recv,
                    Some(signal.spelling().to_owned()),
                )),
                // ⛔ THE TILE SIZE IS RESOLVED HERE because entry 337 takes it already resolved: a
                // size that is no `arith.constant` is `DT_ERROR("sync buffer size has to be a
                // constant op")`, which is this walk's own refusal and not a lowering.
                DfirOp::Dataflow(dataflow::Op::ImplicitSync { dst, size, .. }) => {
                    match defining_op(*size, &scope) {
                        Some(DfirOp::Arith(arith::Op::Constant { value, .. })) => {
                            i32::try_from(*value)
                                .ok()
                                .map(|size| (*dst, L0LxSyncToLower::ImplicitSync(size), None))
                        }
                        _ => None,
                    }
                }
                DfirOp::Dataflow(dataflow::Op::Opaque(opaque)) => {
                    lowered.push(LoweredDataflowOp::Opaque {
                        op: site,
                        lowered: lower_opaque_operation(opaque),
                    });
                    continue;
                }
                _ => continue,
            };
            let lowering = sync.and_then(|(dst_units, op, dbg_name)| {
                lower_sync_operation(&src_units, op, dst_units, &scope, dbg_name, values)
            });
            lowered.push(match lowering {
                Some(lowering) => LoweredDataflowOp::Sync { op: site, lowering },
                None => LoweredDataflowOp::SyncFailed { op: site },
            });
        }

        // `:2035-2038` — a second walk of the same unit, for groups only.
        let mut dead_groups: Vec<&DfirOp> = Vec::new();
        dead_create_groups(&unit.body, &unit.body, &mut dead_groups);
        per_unit.push(UnitSyncLowering {
            lowered,
            dead_groups,
        });
    }
    per_unit
}

/// `unit_op.walk([&](Operation* op) { if (isa<SyncSendOp, SyncRecvOp,
/// ImplicitSyncOnStreamingBufferOp>(op)) .. else if (dyn_cast<OpaqueOp>(op)) .. })`
/// (`:2019-2034`) — the four kinds the first walk stops on, regions included.
fn sync_and_opaque_sites<'a>(body: &'a [DfirOp], found: &mut Vec<&'a DfirOp>) {
    for op in body {
        if matches!(
            op,
            DfirOp::Dataflow(
                dataflow::Op::SyncSend { .. }
                    | dataflow::Op::SyncRecv { .. }
                    | dataflow::Op::ImplicitSync { .. }
                    | dataflow::Op::Opaque(_)
            )
        ) {
            found.push(op);
        }
        for region in regions(op) {
            sync_and_opaque_sites(region, found);
        }
    }
}

/// `if (isa<CreateGroupOp>(op) && op->use_empty()) to_be_deleted.push_back(op);` (`:2036-2037`).
///
/// ⭐ `unit` IS THE WHOLE UNIT BODY WHILE `body` DESCENDS, because a group bound in a loop can be
/// named by a sync outside it — `use_empty` is a census over the operation, not over the region.
fn dead_create_groups<'a>(body: &'a [DfirOp], unit: &[DfirOp], found: &mut Vec<&'a DfirOp>) {
    for op in body {
        if let DfirOp::Dataflow(dataflow::Op::CreateGroup { result, .. }) = op
            && uses(*result, unit).is_empty()
        {
            found.push(op);
        }
        for region in regions(op) {
            dead_create_groups(region, unit, found);
        }
    }
}

#[cfg(test)]
mod pass_unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::{OpFunc, SyncSignal};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::Core;

    /// 🎯 361/384 — THE TWO WALKS ARE NOT ONE: a sync lowers and is queued, while the group it
    /// addresses survives the second walk BECAUSE that sync has not been erased yet.
    #[test]
    fn a_lowered_sync_leaves_the_group_it_names_alive_and_an_unnamed_one_dead() {
        let core = Core::checked(0).expect("every arch has core 0");
        let corelet = Corelet::checked(0).expect("every arch has corelet 0");
        let preamble = vec![
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(0),
                residency: crate::units::residency_of(DfirUnit::Lxsu, core, corelet),
                unit: DfirUnit::Lxsu,
                num_folds: None,
            }),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(1),
                residency: crate::units::residency_of(DfirUnit::Lxlu, core, corelet),
                unit: DfirUnit::Lxlu,
                num_folds: None,
            }),
        ];
        let body = vec![
            DfirOp::Dataflow(dataflow::Op::CreateGroup {
                result: Val(2),
                unit_ids: vec![Val(1)],
            }),
            DfirOp::Dataflow(dataflow::Op::CreateGroup {
                result: Val(3),
                unit_ids: vec![Val(1)],
            }),
            DfirOp::Dataflow(dataflow::Op::SyncRecv {
                from: Val(2),
                signal: SyncSignal::InputToLxsuToLxluToSync,
            }),
        ];
        let program = Program::<Dd2> {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxsu, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            arch: core::marker::PhantomData,
        };

        let mut values = Values::default();
        let per_unit = run_on_operation(&program, &mut values);
        let [unit] = per_unit.as_slice() else {
            panic!("one program unit in, one record out, got {per_unit:?}");
        };
        assert!(
            matches!(unit.lowered.as_slice(), [LoweredDataflowOp::Sync { .. }]),
            "the recv over a group lowered, got {:?}",
            unit.lowered
        );
        assert_eq!(
            unit.dead_groups,
            [&program.units.iter().next().expect("one unit").body[1]],
            "only the group no sync names is dead — the addressed one still has its user"
        );
    }
}
