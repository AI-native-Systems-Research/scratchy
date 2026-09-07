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

use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, dataflow, defining_op};
use crate::islands::dataflow_ir::ty::GenericComp;
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
/// `OpBuilder builder(opaque_op)` positions the insertion point; the caller
/// (`runOnOperation`, `:2027-2029`, entry 361) overrides it with
/// `builder.setInsertionPointToStart(&unit_op.getRegion().front())` before this runs, so the
/// `sentient.opaque` lands at the TOP of the unit's region and the `dataflow.opaque` is erased
/// afterwards via `to_be_deleted`. Both are placement, which is the one mechanism this campaign's
/// ports may drop — the op itself is what this function decides.
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
            read_write: vec![
                (RegName::A00, RegAddr(0)),
                (RegName::A01, RegAddr(1)),
            ],
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
/// units_with_corelet_1.size() > 0)` (`DataflowToSentient.cpp:137-138`) says both lists empty is not
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
/// path handles, and `lowerL0LXSyncOperationForAUnit` refuses it by name two hundred lines earlier
/// (*"Unknown corelet information for sentient"*, `:227-229`). Here that null is
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
/// asks whether three of the four are empty while the fourth is not, twice over
/// (`DataflowToSentient.cpp:796-800`), so each list is a named field rather than a bucket to look up.
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
/// `DT_CHECK(src_dst_lx_corelet0.size() != 0 || …)` (`:794-795`) is the only thing that notices, and
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
/// caller declares all four empty immediately before the call (`:791-793`), so appending and
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
                // `if (dst_unit.getType().str().substr(0, 2) != "l3")` — the LX side.
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
