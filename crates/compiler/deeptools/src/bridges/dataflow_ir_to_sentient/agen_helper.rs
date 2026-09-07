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

//! `Helper.cpp` — 60 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e030_getLoopNestLevel` | 030/384 | 6 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:43` |
//! | `e031_checkIndirectMemViewForExtractOp` | 031/384 | 42 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:388` |
//! | `e032_findExtractScalarOp` | 032/384 | 20 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:514` |
//! | `e033_getLoadConsumer` | 033/384 | 36 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242` |
//! | `e034_setldtype` | 034/384 | 60 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647` |
//! | `e035_generateSetSendDestinationStmts` | 035/384 | 49 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731` |
//! | `e036_getStoreOpFromLoadStorePattern` | 036/384 | 8 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872` |
//! | `e037_findCandidateForLowering` | 037/384 | 12 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884` |
//! | `e038_addLoadChainToDeleteList` | 038/384 | 9 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975` |
//! | `e151_checkCompositeRegion` | 151/384 | 89 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:206` |
//! | `e152_checkStoreOpFromExtractPattern` | 152/384 | 27 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:433` |
//! | `e153_isLoadAndExtractScalarPattern` | 153/384 | 27 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:463` |
//! | `e154_isReceiveAndExtractScalarPattern` | 154/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:493` |
//! | `e155_updateSymbolicAccessDetails` | 155/384 | 35 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013` |
//! | `e156_getStoreProducer` | 156/384 | 159 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285` |
//! | `e157_constructSetActiveMaskValueOp` | 157/384 | 161 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567` |
//! | `e158_createUniformizeRegionsOp` | 158/384 | 57 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880` |
//! | `e210_checkBasicConditions` | 210/384 | 133 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:58` |
//! | `e211_processInterleaveOp` | 211/384 | 80 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:305` |
//! | `e212_gatherAffineLoadStoreDetails` | 212/384 | 74 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:538` |
//! | `e213_constructImmutableAddress` | 213/384 | 16 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217` |
//! | `e214_setImmutableAddrAndIncrements` | 214/384 | 43 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581` |
//! | `e215_setsttype` | 215/384 | 50 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731` |
//! | `e216_constructReceiveAndExtractScalarOp` | 216/384 | 91 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471` |
//! | `e217_lowerVectorLoadHelper` | 217/384 | 46 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899` |
//! | `e218_lowerSetTransferMaskStateOp` | 218/384 | 9 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815` |
//! | `e219_cloneStartAddrOutsideLoop` | 219/384 | 79 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942` |
//! | `e220_cleanupTriviallyRedundantSetSendDestination` | 220/384 | 38 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084` |
//! | `e267_constructTimeLoopsAndVectorOperations` | 267/384 | 109 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789` |
//! | `e268_constructLoadAndStoreStmt` | 268/384 | 177 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167` |
//! | `e269_constructLoadAndExtractScalarOp` | 269/384 | 114 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351` |
//! | `e270_addStoreInputToDeleteList` | 270/384 | 8 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987` |
//! | `e271_lowerCompositeMemoryInterleaveOp` | 271/384 | 37 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775` |
//! | `e272_insertInitializationStmt` | 272/384 | 13 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834` |
//! | `e298_constructAffineDetailsAndAddrs` | 298/384 | 16 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787` |
//! | `e299_lowerAffineCompositeHelper` | 299/384 | 15 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953` |
//! | `e311_constructAffineCompDetailsAndAddrs` | 311/384 | 33 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809` |
//! | `e312_lowerExtractVectorLoadOp` | 312/384 | 21 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001` |
//! | `e313_lowerExtractVectorStoreOp` | 313/384 | 20 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026` |
//! | `e314_lowerVectorLoadOp` | 314/384 | 20 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050` |
//! | `e315_lowerVectorStoreOp` | 315/384 | 28 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074` |
//! | `e316_lowerIndirectVectorLoadOp` | 316/384 | 44 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169` |
//! | `e317_lowerIndirectVectorStoreOp` | 317/384 | 46 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217` |
//! | `e318_lowerLDCVTIPattern` | 318/384 | 326 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444` |
//! | `e327_gatherSymbolicLoadStoreDetails` | 327/384 | 153 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051` |
//! | `e328_adjustMutableAddrInitForIndirect` | 328/384 | 127 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447` |
//! | `e329_lowerCompositeLoadOp` | 329/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106` |
//! | `e330_lowerCompositeStoreOp` | 330/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127` |
//! | `e331_lowerCompositeLoadAndStoreOp` | 331/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148` |
//! | `e332_lowerCompositeIndirectLoadOp` | 332/384 | 42 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267` |
//! | `e333_lowerCompositeIndirectStoreOp` | 333/384 | 42 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313` |
//! | `e334_lowerCompositeIndirectLoadAndStoreOp` | 334/384 | 17 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359` |
//! | `e335_insertCopyAndAddStmts` | 335/384 | 13 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861` |
//! | `e336_adjustMutableAddrInitForStride` | 336/384 | 47 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034` |
//! | `e357_generateAffineAddressManipulationStmts` | 357/384 | 381 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:625` |
//! | `e358_constructLoadAndSendStmt` | 358/384 | 104 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910` |
//! | `e359_constructReceiveAndStoreStmt` | 359/384 | 131 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025` |
//! | `e360_constructSymbolicDetailsAndAddrs` | 360/384 | 16 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849` |
//! | `e374_lowerSymbolicVectorLoadOp` | 374/384 | 26 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380` |
//! | `e375_lowerSymbolicVectorStoreOp` | 375/384 | 30 | `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410` |

use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, arith, dataflow};
use crate::islands::dataflow_ir::ty::AffineMap;
use crate::units::DfirUnit;

/// WHICH LOOP DIALECT A LOOP BELONGS TO — the two `getLoopNestLevel` dispatches on
/// (`Helper.cpp:43-49`).
///
/// ⛔⛔ THE KIND IS NOT DECORATION, IT IS THE COUNTING RULE. `getParentOfType<LoopTy>` walks up to
/// the nearest ancestor **of that type**, skipping everything else — so an `affine.for` directly
/// inside an `scf.for` is at level 0, not level 1. Both dialects appear in one nest here
/// (`e264_createForOpWithAdditionalReturnValue` builds either), which is exactly when getting this
/// wrong stops being harmless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// `affine.for` — [`crate::islands::dataflow_ir::dialects::affine::Op::For`].
    AffineFor,
    /// `scf.for`. Not in the island yet: nothing this bridge emits builds one, and a printed variant
    /// no emitter constructs is the dead arm this crate keeps out. It is a KIND a loop can have,
    /// which is all `getLoopNestLevel` asks.
    ScfFor,
}

/// HOW MANY LOOPS OF ITS OWN KIND ENCLOSE A LOOP — 0 for an outermost one.
///
/// ⛔ ZERO-BASED, AND THE `-1` IS WHY. `dcc::utils::getLoopNestLevel` starts at `level = -1` and
/// increments once per iteration INCLUDING the one that looks at the loop itself
/// (`Utils.cpp:211-221`), so an unnested loop comes out at 0 and the level is a COUNT OF ANCESTORS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LoopNestLevel(pub u32);

/// Replaces: e030_getLoopNestLevel
///
/// The file-static dispatcher (`Helper.cpp:43-49`) over the template (`Utils.cpp:211-221`):
///
/// ```c++
/// static inline int getLoopNestLevel(Operation* op) {
///   if (isa<affine::AffineForOp>(op))
///     return dcc::utils::getLoopNestLevel<affine::AffineForOp>(op);
///   if (isa<scf::ForOp>(op)) return dcc::utils::getLoopNestLevel<scf::ForOp>(op);
///   op->emitError("Expected a loop");
///   return -1;
/// }
///
/// template <class LoopTy> int getLoopNestLevel(Operation *loop_op) {
///   DT_CHECK_MSG(loop_op && isa<LoopTy>(loop_op), "expected valid loop");
///   LoopTy loop = dyn_cast<LoopTy>(loop_op);
///   int level = -1;
///   while (loop) { loop = loop->template getParentOfType<LoopTy>(); ++level; }
///   return level;
/// }
/// ```
///
/// ⛔⛔ IT COUNTS ONLY LOOPS OF THE **SAME** KIND. `getParentOfType<LoopTy>` is not "the parent" — it
/// is "the nearest enclosing op of this type", so every intervening op, an `scf.for` included, is
/// stepped over rather than counted. `enclosing` is therefore filtered by kind and its ORDER does
/// not matter; only how many of them share the loop's dialect.
///
/// ⭐ `-1` IS UNREACHABLE HERE, AND NOT BECAUSE IT IS HANDLED. The dispatcher's third branch —
/// `emitError("Expected a loop")` then `return -1` — exists because its argument is an
/// `Operation*` that might be anything; the argument here is a [`LoopKind`], so "not a loop" is
/// not a value that can be passed. The same goes for the template's `DT_CHECK_MSG(loop_op && ..)`.
#[must_use]
pub fn loop_nest_level(of: LoopKind, enclosing: &[LoopKind]) -> LoopNestLevel {
    let ancestors = enclosing.iter().filter(|kind| **kind == of).count();
    LoopNestLevel(u32::try_from(ancestors).unwrap_or(u32::MAX))
}

/// WHAT AN INDIRECT MEMORY VIEW'S `from_unit` OPERAND RESOLVED TO — `dyn_cast_or_null` on its
/// defining op (`Helper.cpp:391-393`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewedUnit {
    /// It is a `dataflow.get_unit` of this kind.
    GetUnit(DfirUnit),
    /// Its defining op is something else — the `dyn_cast_or_null` returned null.
    NotAGetUnit,
}

/// WHAT AN INDIRECT MEMORY VIEW'S `start_address` OPERAND RESOLVED TO — `Helper.cpp:409-424`.
///
/// ⭐ THE C++ HAS A THIRD OUTCOME THAT CANNOT HAPPEN. Its second test is
/// `!ind_mem_view_start_addr_int || getInt() != 0` — an `arith.constant` whose value is not an
/// `IntegerAttr` is reported as *"start address is not 0"*. A `get_logical_memory_view`'s
/// `start_address` operand is typed `index`, so a constant defining it always carries an integer,
/// and the `dyn_cast<IntegerAttr>` cannot fail. Hence two variants here, not three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewStart {
    /// It is an `arith.constant` of this value.
    Constant(i64),
    /// Its defining op is not an `arith.constant` — the `dyn_cast_or_null` returned null.
    NotAConstant,
}

/// THE THREE THINGS `checkIndirectMemViewForExtractOp` ASKS OF A `dataflow.get_logical_memory_view`,
/// already resolved.
///
/// ⭐ RESOLVING OPERANDS IS THE MECHANISM, NOT THE RULE. The C++ walks each operand back to its
/// defining op because it holds an IR at run time; the rule it then applies is the four checks
/// below. [`Self::resolve`] does the walk for a caller that has the op list, so both halves are
/// available and neither is assumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndirectMemView<'a> {
    /// `ind_mem_view.getFromUnit().getDefiningOp()`.
    pub from_unit: ViewedUnit,
    /// `ind_mem_view.getStartAddress().getDefiningOp()`.
    pub start_address: ViewStart,
    /// `ind_mem_view.getLayoutMap()`.
    pub layout_map: &'a AffineMap,
}

impl<'a> IndirectMemView<'a> {
    /// RESOLVE A VIEW'S TWO OPERANDS AGAINST THE OPS IN SCOPE.
    ///
    /// ⭐ FLAT, AND THAT IS WHAT SCOPE MEANS. A value is defined by exactly one op, and only ops at
    /// or above the view's own level can define one it names — so the caller passes the list it is
    /// walking (the program's preamble, or a unit's body) rather than a whole tree.
    #[must_use]
    pub fn resolve(
        ops: &[DfirOp],
        from_unit: Val,
        start_address: Val,
        layout_map: &'a AffineMap,
    ) -> IndirectMemView<'a> {
        let mut resolved_unit = ViewedUnit::NotAGetUnit;
        let mut resolved_start = ViewStart::NotAConstant;
        for op in ops {
            match op {
                DfirOp::Dataflow(dataflow::Op::GetUnit { result, unit, .. })
                    if *result == from_unit =>
                {
                    resolved_unit = ViewedUnit::GetUnit(*unit);
                }
                DfirOp::Arith(arith::Op::Constant { result, value })
                    if *result == start_address =>
                {
                    resolved_start = ViewStart::Constant(*value);
                }
                _ => {}
            }
        }
        IndirectMemView {
            from_unit: resolved_unit,
            start_address: resolved_start,
            layout_map,
        }
    }
}

/// THE OUTCOME OF [`check_indirect_mem_view_for_extract_op`] — admissible, or WHICH check refused.
///
/// ⛔⛔ THE C++ RETURNS A BARE `bool` AND EMITS THE REASON AS A DIAGNOSTIC, WHICH IS A REASON THAT
/// ONLY EXISTS IN A LOG. Five different malformed views all come back as `false`
/// (`Helper.cpp:388-431`), and the caller — `e269_constructLoadAndExtractScalarOp` — can only say
/// "not an extract pattern". Naming the outcome keeps the vendor's five messages, in the vendor's
/// order, as VALUES; [`Self::admissible`] is the `bool` the callers branch on.
///
/// ⛔ NOT AN ERROR TYPE. Nothing here is a `Result` and nothing stops: an inadmissible view means
/// this op is not the extract pattern, which is an ordinary answer to an ordinary question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndirectMemViewCheck {
    /// A 1D identity view of a virtual IBR from address 0 — the shape the extract pattern needs.
    Admissible,
    /// The `from_unit` operand is not defined by a `dataflow.get_unit`.
    FromUnitIsNotAGetUnit,
    /// It is a `get_unit`, but not of [`DfirUnit::LxVirtualIbr`].
    NotOnAVirtualIbr,
    /// The `start_address` operand is not an `arith.constant`.
    StartAddressIsNotConstant,
    /// It is a constant, but not zero.
    StartAddressIsNotZero,
    /// The `layout_map` is not the one-dimensional identity.
    LayoutMapIsNot1DIdentity,
}

impl IndirectMemViewCheck {
    /// The C++'s own return value: `true` only for [`Self::Admissible`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, IndirectMemViewCheck::Admissible)
    }

    /// The diagnostic the C++ emits for this outcome, verbatim (`Helper.cpp:394-430`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            IndirectMemViewCheck::Admissible => None,
            IndirectMemViewCheck::FromUnitIsNotAGetUnit => {
                Some("from unit of indirect memory view is not a get_unit operation")
            }
            IndirectMemViewCheck::NotOnAVirtualIbr => {
                Some("indirect memory view is not operating on a virtual IBR")
            }
            IndirectMemViewCheck::StartAddressIsNotConstant => {
                Some("indirect memory view does not have a constant start address")
            }
            IndirectMemViewCheck::StartAddressIsNotZero => {
                Some("indirect memory view start address is not 0")
            }
            IndirectMemViewCheck::LayoutMapIsNot1DIdentity => {
                Some("expecting 1D identity map for layout_map")
            }
        }
    }
}

/// Replaces: e031_checkIndirectMemViewForExtractOp
///
/// `checkIndirectMemViewForExtractOp` (`Helper.cpp:388-431`) — *"The mem_view should be a 1D space
/// with start address of 0 operating on a virtual IBR unit."*
///
/// ⛔⛔ FOUR CHECKS IN THIS ORDER, AND THE ORDER IS THE PORT. Each one is what makes the next
/// answerable: a unit that is not a `get_unit` has no `type=` to compare, a start address that is
/// not a constant has no integer to test against zero. Reordering them turns a diagnosis into a
/// crash on the C++ side and a wrong answer here.
///
/// ⭐ WHY A VIRTUAL IBR AT ALL: the index vector of a gather lives in the LX's Index Buffer Region,
/// and the extract pattern is *load one index, then use it as the address of the real load*. The view
/// being flat, unpermuted and based at 0 is what makes the extracted scalar usable as an address
/// directly — `e328_adjustMutableAddrInitForIndirect` adds it to the mutable address with nothing in
/// between.
#[must_use]
pub fn check_indirect_mem_view_for_extract_op(view: &IndirectMemView<'_>) -> IndirectMemViewCheck {
    // `:391-397` — the from-unit must be a get_unit before its type can be read.
    let unit = match view.from_unit {
        ViewedUnit::NotAGetUnit => return IndirectMemViewCheck::FromUnitIsNotAGetUnit,
        ViewedUnit::GetUnit(unit) => unit,
    };
    // `:398-407` — and the unit it names must be the virtual IBR. The C++ compares
    // `stringToSenComponents.find(getType().str())->second` against `SenComponents::LXVIRTUALIBR`;
    // here the `type=` string and the enum are the same value (`DfirUnit::spelling`).
    if unit != DfirUnit::LxVirtualIbr {
        return IndirectMemViewCheck::NotOnAVirtualIbr;
    }
    // `:409-424` — a constant start address, and it must be zero. TWO checks, not one: the C++
    // reports "does not have a constant start address" and "start address is not 0" separately.
    match view.start_address {
        ViewStart::NotAConstant => return IndirectMemViewCheck::StartAddressIsNotConstant,
        ViewStart::Constant(start) if start != 0 => {
            return IndirectMemViewCheck::StartAddressIsNotZero;
        }
        ViewStart::Constant(_) => {}
    }
    // `:426-431` — `getNumDims() != 1 || !isIdentity()`.
    if view.layout_map.dims != 1 || !view.layout_map.is_identity() {
        return IndirectMemViewCheck::LayoutMapIsNot1DIdentity;
    }
    // `:433` — `return true;`
    IndirectMemViewCheck::Admissible
}

/// WHICH EXTRACT STATEMENT — the two `findExtractScalarOp` is instantiated with
/// (`Helper.cpp:3195, 3243, 3293, 3339`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractScalarKind {
    /// `sentient.load_and_extract_scalar` — the load side, paired with an indirect LOAD.
    LoadAndExtractScalar,
    /// `sentient.receive_and_extract_scalar` — the store side, paired with an indirect STORE.
    ReceiveAndExtractScalar,
}

/// THE `extract_idx` THAT PAIRS ONE EXTRACT STATEMENT WITH THE INDIRECT ACCESS THAT USES IT.
///
/// ⛔⛔ AN `i8` ATTRIBUTE, MINTED PER PROGRAM UNIT. `fuseLoadOrStoreChainOps` opens with
/// `unsigned extract_idx = 0` for each unit (`AgenToSentient.cpp:22-27`) and every extract statement
/// created stamps the SAME value on itself and on the indirect access it feeds
/// (`Helper.cpp:2456-2463`, `:2553-2558`), then increments. The vendor's output shows the attribute:
/// `extract_idx = 0 : i8`
/// (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores_composite.mlir:34`).
///
/// ⭐ MINTED, NOT WRITTEN — the only way to get one is [`ExtractScalarOps::mint`], for the reason
/// [`crate::islands::dataflow_ir::Values`] gives about SSA names: two extract statements sharing an
/// index is not a diagnosis anyone would enjoy making from a wrong address at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExtractIndex(u8);

impl ExtractIndex {
    /// The value the `extract_idx` attribute carries.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// ONE EXTRACT STATEMENT, AS THE INDIRECT ACCESS THAT USES IT NEEDS TO SEE IT.
///
/// ⭐ IT IS THE OP'S TWO RESULTS THAT MATTER. `sentient.load_and_extract_scalar` binds an ADDRESS and
/// a DATUM (`%21, %22 = sentient.load_and_extract_scalar ..`,
/// `lx_indirect_loads_stores_composite.mlir:34`), and what the indirect access does with the op it
/// finds is take the datum — `e328_adjustMutableAddrInitForIndirect` adds it to the mutable address
/// (`%34 = sentient.scalar_add %33, %22`, `:43`). Carrying the two results is what makes the found
/// op usable without a second lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractScalarOp {
    /// Its `extract_idx`.
    pub index: ExtractIndex,
    /// Which of the two statements it is.
    pub kind: ExtractScalarKind,
    /// Its first result — the address it advanced.
    pub addr: Val,
    /// Its second result — the extracted scalar, which is the indirect access's own address.
    pub data: Val,
}

/// THE EXTRACT STATEMENTS ONE PROGRAM UNIT HAS BOUND, in mint order.
///
/// ⭐ PER UNIT, LIKE THE COUNTER. `fuseLoadOrStoreChainOps` resets `extract_idx` to 0 for every
/// program unit and `findExtractScalarOp` walks `unit` alone (`Helper.cpp:520`) — so this is the
/// whole search space, and an index from one unit means nothing in another.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtractScalarOps {
    /// In mint order, so an [`ExtractIndex`] is also a position. Private: see [`ExtractIndex`].
    minted: Vec<ExtractScalarOp>,
}

impl ExtractScalarOps {
    /// STAMP THE NEXT `extract_idx` ON A NEW EXTRACT STATEMENT — the counter half of
    /// `Helper.cpp:2456-2463`.
    ///
    /// ⭐ THE COUNTER IS SHARED BETWEEN THE TWO KINDS. One `extract_idx` per unit serves both the
    /// load and the store side (`AgenToSentient.cpp:27` feeds `lowerExtractVectorLoadOp` at `:57`
    /// and `lowerExtractVectorStoreOp` at `:75`), so indices are unique across kinds and the kind a
    /// lookup names is a consistency check rather than a disambiguator.
    ///
    /// The op this stamps is built by `e269_constructLoadAndExtractScalarOp` and its store twin,
    /// which are other entries; this is the pairing they record.
    pub fn mint(&mut self, kind: ExtractScalarKind, addr: Val, data: Val) -> ExtractScalarOp {
        let op = ExtractScalarOp {
            index: ExtractIndex(u8::try_from(self.minted.len()).unwrap_or(u8::MAX)),
            kind,
            addr,
            data,
        };
        self.minted.push(op);
        op
    }

    /// Replaces: e032_findExtractScalarOp
    ///
    /// `findExtractScalarOp<ExtractOpTy>` (`Helper.cpp:513-535`):
    ///
    /// ```c++
    /// DT_CHECK_MSG(op->hasAttr("extract_idx"), "input operation does not have extract_idx attribute");
    /// auto extract_idx = cast<IntegerAttr>(op->getAttr("extract_idx")).getInt();
    /// Operation* extract_op = nullptr;
    /// unit.walk<WalkOrder::PreOrder>([&](ExtractOpTy curr_op) {
    ///   DT_CHECK_MSG(curr_op->hasAttr("extract_idx"), "load_and_extract_scalar_op found without extract_idx");
    ///   auto curr_extract_idx = cast<IntegerAttr>(curr_op->getAttr("extract_idx")).getInt();
    ///   if (curr_extract_idx == extract_idx) { extract_op = curr_op; return WalkResult::interrupt(); }
    ///   return WalkResult::advance();
    /// });
    /// DT_CHECK(extract_op);
    /// return extract_op;
    /// ```
    ///
    /// ⭐ `None` IS THE CALLER'S OWN BRANCH, NOT A REFUSAL. `DT_CHECK(extract_op)` looks like the
    /// only outcome, but every call site tests the result anyway and reports it —
    /// *"could not locate a load_and_extract_scalar operation matching the extract_idx used by op"*
    /// (`Helper.cpp:3196-3199`, `:3244-3247`, `:3294-3297`, `:3340-3343`) — so absence is an answer
    /// this function is expected to give.
    ///
    /// ⭐ BOTH `DT_CHECK_MSG`s ARE UNREPRESENTABLE. *"input operation does not have extract_idx"* is
    /// the caller passing an op with no attribute — here the index is a typed argument, so there is
    /// nothing to look up and fail. *"load_and_extract_scalar_op found without extract_idx"* is an
    /// extract statement that was never stamped — here every entry gets its index from
    /// [`Self::mint`], so an unstamped one cannot be in the list.
    ///
    /// ⛔ THE WALK IS THE MECHANISM. Pre-order with an interrupt on the first match is how the C++
    /// searches an op tree; what it MEANS is "the extract statement of this kind with this index",
    /// and the first match is the only match because [`Self::mint`] never repeats an index.
    #[must_use]
    pub fn find(&self, of: ExtractScalarKind, index: ExtractIndex) -> Option<ExtractScalarOp> {
        self.minted
            .iter()
            .copied()
            .find(|op| op.kind == of && op.index == index)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::{AffineExpr, ElemType, MemRef};
    use crate::units::Residency;

    /// ⭐ AN OUTERMOST LOOP IS LEVEL 0 — the `level = -1` start, which is easy to lose.
    #[test]
    fn an_unnested_loop_is_level_zero() {
        assert_eq!(loop_nest_level(LoopKind::AffineFor, &[]), LoopNestLevel(0));
        assert_eq!(loop_nest_level(LoopKind::ScfFor, &[]), LoopNestLevel(0));
    }

    /// The level is the number of enclosing loops of the same kind.
    #[test]
    fn the_level_counts_the_enclosing_loops() {
        let nest = [
            LoopKind::AffineFor,
            LoopKind::AffineFor,
            LoopKind::AffineFor,
        ];
        assert_eq!(
            loop_nest_level(LoopKind::AffineFor, &nest),
            LoopNestLevel(3)
        );
    }

    /// ⛔⛔ ONLY LOOPS OF THE SAME KIND COUNT. `getParentOfType<AffineForOp>` steps over an
    /// `scf.for`, so an `affine.for` inside three of them is still level 0 — and an `scf.for` in the
    /// same place is level 3.
    #[test]
    fn a_loop_of_the_other_kind_is_stepped_over() {
        let nest = [LoopKind::ScfFor, LoopKind::ScfFor, LoopKind::ScfFor];
        assert_eq!(
            loop_nest_level(LoopKind::AffineFor, &nest),
            LoopNestLevel(0)
        );
        assert_eq!(loop_nest_level(LoopKind::ScfFor, &nest), LoopNestLevel(3));
    }

    /// A mixed nest counts each kind separately, at the same position.
    #[test]
    fn a_mixed_nest_counts_each_kind_separately() {
        let nest = [
            LoopKind::AffineFor,
            LoopKind::ScfFor,
            LoopKind::AffineFor,
            LoopKind::ScfFor,
            LoopKind::ScfFor,
        ];
        assert_eq!(
            loop_nest_level(LoopKind::AffineFor, &nest),
            LoopNestLevel(2)
        );
        assert_eq!(loop_nest_level(LoopKind::ScfFor, &nest), LoopNestLevel(3));
    }

    /// The identity over one dimension — what an admissible view's `layout_map` is.
    fn identity_1d() -> AffineMap {
        AffineMap::identity(1)
    }

    /// ⭐ THE SHAPE THE VENDOR ACTUALLY EMITS. `%13 = dataflow.get_unit {name = "lxvirtualibr",
    /// type = "lxvirtualibr"}` then `%23 = dataflow.get_logical_memory_view %13, %3
    /// {layout_map = #map<(d0) -> (d0)>} : index, index, memref<32xindex>` with `%3 = 0`
    /// (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores_composite.mlir:30,36`).
    #[test]
    fn the_vendors_own_indirect_view_is_admissible() {
        let map = identity_1d();
        let view = IndirectMemView {
            from_unit: ViewedUnit::GetUnit(DfirUnit::LxVirtualIbr),
            start_address: ViewStart::Constant(0),
            layout_map: &map,
        };
        let outcome = check_indirect_mem_view_for_extract_op(&view);
        assert_eq!(outcome, IndirectMemViewCheck::Admissible);
        assert!(outcome.admissible());
        assert_eq!(outcome.diagnostic(), None);
    }

    /// ⛔ EACH CHECK, AND THE MESSAGE IT CARRIES.
    #[test]
    fn each_refusal_names_itself() {
        let map = identity_1d();
        let cases = [
            (
                ViewedUnit::NotAGetUnit,
                ViewStart::Constant(0),
                IndirectMemViewCheck::FromUnitIsNotAGetUnit,
            ),
            (
                ViewedUnit::GetUnit(DfirUnit::Lx),
                ViewStart::Constant(0),
                IndirectMemViewCheck::NotOnAVirtualIbr,
            ),
            (
                ViewedUnit::GetUnit(DfirUnit::LxVirtualIbr),
                ViewStart::NotAConstant,
                IndirectMemViewCheck::StartAddressIsNotConstant,
            ),
            (
                ViewedUnit::GetUnit(DfirUnit::LxVirtualIbr),
                ViewStart::Constant(64),
                IndirectMemViewCheck::StartAddressIsNotZero,
            ),
        ];
        for (from_unit, start_address, expected) in cases {
            let view = IndirectMemView {
                from_unit,
                start_address,
                layout_map: &map,
            };
            let outcome = check_indirect_mem_view_for_extract_op(&view);
            assert_eq!(outcome, expected);
            assert!(!outcome.admissible());
            assert!(outcome.diagnostic().is_some(), "{expected:?} has a message");
        }
    }

    /// ⛔ `getNumDims() != 1 || !isIdentity()` — BOTH halves refuse.
    #[test]
    fn a_layout_map_that_is_not_the_one_dimensional_identity_is_refused() {
        // Two dimensions, identity: refused on the arity.
        let two_d = AffineMap::identity(2);
        // One dimension, not the identity: `(d0) -> (d0 * 4)` is the strided view of a gather's
        // data, not of its index vector.
        let strided = AffineMap::unary(AffineExpr::dim(0).times(4));
        for map in [&two_d, &strided] {
            let view = IndirectMemView {
                from_unit: ViewedUnit::GetUnit(DfirUnit::LxVirtualIbr),
                start_address: ViewStart::Constant(0),
                layout_map: map,
            };
            assert_eq!(
                check_indirect_mem_view_for_extract_op(&view),
                IndirectMemViewCheck::LayoutMapIsNot1DIdentity
            );
        }
    }

    /// ⭐ THE ORDER IS LOAD-BEARING: a view that fails the FIRST check reports that one even though
    /// the later ones would fail too. The C++ returns at the first refusal.
    #[test]
    fn the_first_failing_check_is_the_one_reported() {
        let two_d = AffineMap::identity(2);
        let view = IndirectMemView {
            from_unit: ViewedUnit::NotAGetUnit,
            start_address: ViewStart::Constant(64),
            layout_map: &two_d,
        };
        assert_eq!(
            check_indirect_mem_view_for_extract_op(&view),
            IndirectMemViewCheck::FromUnitIsNotAGetUnit
        );
    }

    /// ⭐ AND THE OPERAND WALK FINDS THEM — the mechanism, over the ops the vendor's unit declares.
    #[test]
    fn resolve_reads_the_defining_ops() {
        let map = identity_1d();
        let ops = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(3),
                value: 0,
            }),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(13),
                residency: Residency::Global,
                unit: DfirUnit::LxVirtualIbr,
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(23),
                from: Val(13),
                start: Val(3),
                layout: identity_1d(),
                ty: MemRef {
                    shape: vec![32],
                    elem: ElemType::Int(64),
                },
            }),
        ];
        let view = IndirectMemView::resolve(&ops, Val(13), Val(3), &map);
        assert_eq!(view.from_unit, ViewedUnit::GetUnit(DfirUnit::LxVirtualIbr));
        assert_eq!(view.start_address, ViewStart::Constant(0));
        assert!(check_indirect_mem_view_for_extract_op(&view).admissible());
    }

    /// An operand nothing in scope defines resolves to neither a `get_unit` nor a constant — which
    /// is exactly what `dyn_cast_or_null` answers for a null defining op.
    #[test]
    fn resolve_answers_for_an_unbound_operand() {
        let map = identity_1d();
        let view = IndirectMemView::resolve(&[], Val(13), Val(3), &map);
        assert_eq!(view.from_unit, ViewedUnit::NotAGetUnit);
        assert_eq!(view.start_address, ViewStart::NotAConstant);
        assert_eq!(
            check_indirect_mem_view_for_extract_op(&view),
            IndirectMemViewCheck::FromUnitIsNotAGetUnit
        );
    }

    /// ⭐ THE PAIRING ROUND-TRIPS: the index a mint stamps is the one a lookup finds, and it carries
    /// the two results the indirect access needs.
    #[test]
    fn a_minted_extract_statement_is_found_by_its_index() {
        let mut ops = ExtractScalarOps::default();
        let first = ops.mint(ExtractScalarKind::LoadAndExtractScalar, Val(21), Val(22));
        assert_eq!(first.index.get(), 0, "the counter opens at 0 per unit");

        let found = ops.find(ExtractScalarKind::LoadAndExtractScalar, first.index);
        assert_eq!(found, Some(first));
        assert_eq!(found.map(|op| op.data), Some(Val(22)));
    }

    /// ⭐ THE COUNTER IS SHARED BETWEEN THE TWO KINDS, so the second statement is index 1 whichever
    /// side it is on — and a lookup for the wrong kind finds nothing.
    #[test]
    fn the_counter_is_shared_and_the_kind_is_checked() {
        let mut ops = ExtractScalarOps::default();
        let load = ops.mint(ExtractScalarKind::LoadAndExtractScalar, Val(21), Val(22));
        let store = ops.mint(ExtractScalarKind::ReceiveAndExtractScalar, Val(31), Val(32));
        assert_eq!(store.index.get(), 1);

        assert_eq!(
            ops.find(ExtractScalarKind::ReceiveAndExtractScalar, store.index),
            Some(store)
        );
        assert_eq!(
            ops.find(ExtractScalarKind::ReceiveAndExtractScalar, load.index),
            None,
            "index 0 was stamped on the LOAD side"
        );
    }

    /// ⛔ AN EMPTY UNIT HAS NOTHING TO FIND, and that is the `None` every call site already reports.
    #[test]
    fn a_unit_with_no_extract_statement_finds_none() {
        let empty = ExtractScalarOps::default();
        let mut other = ExtractScalarOps::default();
        let elsewhere = other.mint(ExtractScalarKind::LoadAndExtractScalar, Val(1), Val(2));
        assert_eq!(
            empty.find(ExtractScalarKind::LoadAndExtractScalar, elsewhere.index),
            None
        );
    }
}
