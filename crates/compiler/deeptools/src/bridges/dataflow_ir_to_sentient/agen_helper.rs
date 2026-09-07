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

use crate::arch::{Arch, Bytes, Elements, IsaGen};
use crate::islands::dataflow_ir::dialects::{
    self as dfir_op, Op as DfirOp, Val, arith, dataflow, defining_op, results, uses,
};
use crate::islands::dataflow_ir::link::SendEnd;
use crate::islands::dataflow_ir::ty::{AffineMap, GenericComp, Vector};
use crate::islands::sentient::dialects::{Op as SenOp, sentient as sen};
use crate::units::DfirUnit;
use core::num::NonZeroU32;
use std::collections::BTreeSet;

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

// ══════════════════════════════════════════════════════════════════════════════════════════════
// THE AGEN OP CLASSES — the reference's template parameters, as values.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH `agen` OP CLASS — what `findCandidateForLowering<OpTy>` and
/// `getStoreOpFromLoadStorePattern<VectorStoreTy>` are instantiated with.
///
/// ⭐⭐ THE TEMPLATE PARAMETER IS AN INPUT, SO IT HAS TO BE NAMABLE. `findCandidateForLowering` is
/// instantiated with **eleven** different classes (`Helper.cpp:3013, 3038, 3063, 3086, 3120, 3141,
/// 3162, 3184, 3232, 3282, 3328, 3373`) and `getStoreOpFromLoadStorePattern` with two
/// (`VectorStoreOp` at `:3053, :3067`, `SymbolicVectorStoreOp` at `:3384`). A port that hard-coded
/// one class would be a port of one instantiation, not of the function.
///
/// ⛔ THIS ISLAND HOLDS THREE OF THE TWELVE TODAY — `agen.vector_load`, `agen.vector_store` and
/// `agen.composite_load_and_store`. The other nine are the indirect (gather/scatter) and symbolic
/// addressing families, which nothing in this crate emits yet; [`agen_op_kind`] is the one place
/// they attach when an op for them is added, and until then a search for one finds nothing rather
/// than finding the wrong thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AgenOpKind {
    /// `agen.vector_load`.
    VectorLoad,
    /// `agen.vector_store`.
    VectorStore,
    /// `agen.indirect_vector_load`.
    IndirectVectorLoad,
    /// `agen.indirect_vector_store`.
    IndirectVectorStore,
    /// `agen.symbolic_vector_load`.
    SymbolicVectorLoad,
    /// `agen.symbolic_vector_store`.
    SymbolicVectorStore,
    /// `agen.composite_load`.
    CompositeLoad,
    /// `agen.composite_store`.
    CompositeStore,
    /// `agen.composite_load_and_store`.
    CompositeLoadAndStore,
    /// `agen.composite_indirect_load`.
    CompositeIndirectLoad,
    /// `agen.composite_indirect_store`.
    CompositeIndirectStore,
    /// `agen.composite_indirect_load_and_store`.
    CompositeIndirectLoadAndStore,
}

/// WHICH CLASS ONE STATEMENT IS, or `None` for a statement that is not an `agen` op at all.
///
/// ⛔ TOTAL OVER THE ISLAND'S OPS, NO WILDCARD. An `agen` op added to the island must say which of
/// the reference's classes it is, rather than silently being invisible to every search below.
#[must_use]
pub fn agen_op_kind(op: &DfirOp) -> Option<AgenOpKind> {
    match op {
        DfirOp::Agen(op) => match op {
            dfir_op::agen::Op::VectorLoad { .. } => Some(AgenOpKind::VectorLoad),
            dfir_op::agen::Op::VectorStore { .. } => Some(AgenOpKind::VectorStore),
            dfir_op::agen::Op::CompositeLoadAndStore(_) => Some(AgenOpKind::CompositeLoadAndStore),
            // The region terminator is not a transfer.
            dfir_op::agen::Op::Yield => None,
        },
        DfirOp::Arith(_)
        | DfirOp::Scf(_)
        | DfirOp::Affine(_)
        | DfirOp::Dataflow(_)
        | DfirOp::VectorChain(_) => None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 033/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A LOAD, AS THE FIVE CLASSES `getLoadConsumer` DISTINGUISHES — each carrying the value the search
/// starts from.
///
/// ⭐⭐ THE CLASS DECIDES WHERE THE SEARCH ROOTS, AND THAT IS THE FIRST HALF OF THE FUNCTION. The
/// three vector loads root at their RESULT; the two composite loads root at the region argument
/// `getLoadInductionVar()`, because a composite load's data never becomes a top-level SSA value at
/// all (`Helper.cpp:1245-1251`).
///
/// ⛔ AND A NON-LOAD IS NOT CONSTRUCTIBLE. The reference's `DT_CHECK_MSG(consumer_root, ...)` fires
/// when `load_op` is none of the five; here that op cannot be handed to [`load_consumer`], so the
/// abort has no representation. [`AgenLoad::of`] is the door.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgenLoad {
    /// `VectorLoadOp` — roots at `getResult(0)`.
    Vector {
        /// The vector it binds.
        result: Val,
    },
    /// `IndirectVectorLoadOp` — roots at `getResult(0)`.
    IndirectVector {
        /// The vector it binds.
        result: Val,
    },
    /// `SymbolicVectorLoadOp` — roots at `getResult(0)`.
    SymbolicVector {
        /// The vector it binds.
        result: Val,
    },
    /// `CompositeLoadOp` — roots at `getLoadInductionVar()`.
    Composite {
        /// The region argument the loaded vector arrives on.
        load_induction_var: Val,
    },
    /// `CompositeIndirectLoadOp` — roots at `getLoadInductionVar()`.
    CompositeIndirect {
        /// The region argument the loaded vector arrives on.
        load_induction_var: Val,
    },
}

impl AgenLoad {
    /// THE VALUE THE CONSUMER SEARCH STARTS FROM — `consumer_root` (`Helper.cpp:1244-1251`).
    #[must_use]
    pub const fn consumer_root(self) -> Val {
        match self {
            Self::Vector { result }
            | Self::IndirectVector { result }
            | Self::SymbolicVector { result } => result,
            Self::Composite { load_induction_var }
            | Self::CompositeIndirect { load_induction_var } => load_induction_var,
        }
    }

    /// WHICH LOAD CLASS ONE STATEMENT IS, or `None` if it is not one of the five.
    ///
    /// ⛔ THREE OF THE FIVE HAVE NO ISLAND OP YET (the indirect and symbolic families), and
    /// `agen.composite_load_and_store` is **not** `CompositeLoadOp` — the reference's `isa<>` list
    /// does not include it, so it answers `None` here too rather than borrowing the composite arm.
    #[must_use]
    pub fn of(op: &DfirOp) -> Option<AgenLoad> {
        match op {
            DfirOp::Agen(dfir_op::agen::Op::VectorLoad { result, .. }) => {
                Some(AgenLoad::Vector { result: *result })
            }
            DfirOp::Agen(
                dfir_op::agen::Op::VectorStore { .. }
                | dfir_op::agen::Op::CompositeLoadAndStore(_)
                | dfir_op::agen::Op::Yield,
            )
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            | DfirOp::VectorChain(_) => None,
        }
    }
}

/// THE THREE OPS THAT MAY SIT BETWEEN A LOAD AND ITS SEND.
///
/// ⛔ EXACTLY THREE, AND BOTH FUNCTIONS THAT NAME THEM NAME THE SAME THREE: `getLoadConsumer`
/// accepts them ahead of the send (`Helper.cpp:1268-1270`) and `addLoadChainToDeleteList` deletes
/// them (`Helper.cpp:2977-2980`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rearrangement {
    /// `vectorchain.select`.
    Select,
    /// `vectorchain.shuffle`.
    Shuffle,
    /// `vectorchain.rotate`.
    Rotate,
}

impl Rearrangement {
    /// WHETHER ONE STATEMENT IS ONE OF THE THREE — `isa<SelectOp, ShuffleOp, RotateOp>`.
    #[must_use]
    pub fn of(op: &DfirOp) -> Option<Rearrangement> {
        match op {
            DfirOp::VectorChain(op) => match op {
                dfir_op::vectorchain::Op::Select { .. } => Some(Rearrangement::Select),
                dfir_op::vectorchain::Op::Shuffle { .. } => Some(Rearrangement::Shuffle),
                dfir_op::vectorchain::Op::Rotate { .. } => Some(Rearrangement::Rotate),
                dfir_op::vectorchain::Op::Estimate { .. }
                | dfir_op::vectorchain::Op::FastExp { .. }
                | dfir_op::vectorchain::Op::Floor { .. }
                | dfir_op::vectorchain::Op::ScanWithGap { .. }
                | dfir_op::vectorchain::Op::Multiply { .. }
                | dfir_op::vectorchain::Op::MultiplyAccumulate { .. }
                | dfir_op::vectorchain::Op::ElementWiseCompare { .. }
                | dfir_op::vectorchain::Op::ElementWiseSelection { .. }
                | dfir_op::vectorchain::Op::Binary { .. }
                | dfir_op::vectorchain::Op::ConstantBitstream { .. }
                | dfir_op::vectorchain::Op::Cast { .. }
                // ⛔ A PACK IS NOT A REARRANGEMENT, however much it looks like one: `getLoadConsumer`
                // names `SelectOp`, `ShuffleOp` and `RotateOp` and stops (`Helper.cpp:1268-1270`), so
                // a load feeding a pack has no consumer by this rule.
                | dfir_op::vectorchain::Op::Pack { .. }
                | dfir_op::vectorchain::Op::Merge { .. }
                | dfir_op::vectorchain::Op::CreateAffineMask { .. } => None,
            },
            DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_) => None,
        }
    }
}

/// WHAT A LOAD'S CONSUMER SEARCH FOUND — the reference's `std::pair<Operation*, Operation*>` with
/// its two null cases told apart.
///
/// ⛔ TWO DISTINCT `{nullptr, nullptr}` RETURNS IN THE REFERENCE, AND THEY CARRY DIFFERENT
/// DIAGNOSTICS: *"loadOp should have one consumer"* (`Helper.cpp:1257`) and *"unsupported loadOp
/// consumer!"* (`Helper.cpp:1279`). A single `Option` would collapse them, and the caller decides
/// what to say.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum LoadConsumer<'a> {
    /// The pair: the `dataflow.send` this load feeds, and the op defining the unit it sends to.
    Found {
        /// `consumer_info.first` — the send.
        send: &'a DfirOp,
        /// `consumer_info.second` = `send_op.getToUnit().getDefiningOp()`.
        ///
        /// ⛔ `None` IS THE REFERENCE'S NULL, NOT AN OMISSION. `getDefiningOp()` answers null when
        /// the to-unit is a region argument rather than an op's result, and the reference returns
        /// that null as the second half of a SUCCESSFUL pair — `generateSetSendDestinationStmts`
        /// then finds neither a `get_unit` nor a `query_map` and emits nothing.
        consumer: Option<&'a DfirOp>,
    },
    /// `emitError("loadOp should have one consumer")` — the root's use count is not one.
    NotOneUse,
    /// `emitError("unsupported loadOp consumer!")` — the single user is neither a send nor a
    /// single-used rearrangement feeding one.
    Unsupported,
}

/// Replaces: e033_getLoadConsumer
///
/// **033/384** `AgenToSentientLoweringPass::getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242` (36L).
///
/// ```cpp
/// std::pair<Operation*, Operation*> AgenToSentientLoweringPass::getLoadConsumer(Operation* load_op) {
///   Value consumer_root = nullptr;
///   if (isa<VectorLoadOp, IndirectVectorLoadOp, SymbolicVectorLoadOp>(load_op))
///     consumer_root = load_op->getResult(0);
///   else if (auto comp_load_op = dyn_cast<CompositeLoadOp>(load_op))
///     consumer_root = comp_load_op.getLoadInductionVar();
///   else if (auto comp_ind_load_op = dyn_cast<CompositeIndirectLoadOp>(load_op))
///     consumer_root = comp_ind_load_op.getLoadInductionVar();
///   DT_CHECK_MSG(consumer_root, "could not determine root of the loadOp consumer");
///   // Assume single consumer per loadOp.
///   if (!consumer_root.hasOneUse()) {
///     load_op->emitError("loadOp should have one consumer");
///     return std::make_pair(nullptr, nullptr);
///   }
///   // Assume loadOp is followed by only:
///   //   - a sendOp, or
///   //   - a selectOp/shuffleOp followed by a sendOp, or
///   //   - a storeOp
///   auto* user = *consumer_root.getUsers().begin();
///   if (auto send_op = dyn_cast<dataflow::SendOp>(user)) {
///     Operation* consumer = send_op.getToUnit().getDefiningOp();
///     return std::make_pair(send_op, consumer);
///   } else if (isa<vectorchain::SelectOp, vectorchain::ShuffleOp, vectorchain::RotateOp>(user) &&
///              user->hasOneUse()) {
///     if (auto send_op = dyn_cast<dataflow::SendOp>(*user->getUsers().begin())) {
///       Operation* consumer = send_op.getToUnit().getDefiningOp();
///       return std::make_pair(send_op, consumer);
///     }
///   }
///   load_op->emitError("unsupported loadOp consumer!");
///   return std::make_pair(nullptr, nullptr);
/// }
/// ```
///
/// ⛔⛔ `hasOneUse()` COUNTS USES, AND THAT IS THE WHOLE VALUE OF THE FUNCTION. A previous attempt
/// at this entry inspected the shapes it expected and reported "one consumer" for a load whose
/// result a compute also read — a check that confirms its own assumption. [`uses`] is total over
/// every op in the island and descends into regions, so a use this search does not expect still
/// counts (see its own note).
///
/// ⛔ THE COMMENT PROMISES A STORE ARM THAT THE CODE DOES NOT HAVE. "Assume loadOp is followed by
/// only: a sendOp, or a selectOp/shuffleOp followed by a sendOp, **or a storeOp**" — and then no
/// branch tests for a store, so a load consumed by an `agen.vector_store` reaches
/// `emitError("unsupported loadOp consumer!")`. The port follows the CODE, which is what runs; the
/// store pattern is recognised elsewhere, by [`store_op_from_load_store_pattern`], and its caller
/// asks that question before this one (`Helper.cpp:2905-2918`).
///
/// ⭐ THE SCOPE IS THE WHOLE FUNCTION BODY, NOT THE LOAD'S BLOCK. The send lives in the same
/// `affine.for` body as the load, but the `get_unit` its `to` resolves to is at the top level
/// (`lx-to-sfp-bypass-1.mlir:126,131-136`), so the search walks down and the resolution walks up.
pub fn load_consumer<'a>(load: AgenLoad, scope: &'a [DfirOp]) -> LoadConsumer<'a> {
    let root = load.consumer_root();

    // `if (!consumer_root.hasOneUse())` — one entry per USE, so a value read twice by one op is not
    // single-used, and a value read nowhere is not either.
    let users = uses(root, scope);
    let [user] = users.as_slice() else {
        return LoadConsumer::NotOneUse;
    };

    // `if (auto send_op = dyn_cast<dataflow::SendOp>(user))`.
    if let DfirOp::Dataflow(dfir_op::dataflow::Op::Send { to, .. }) = user {
        return LoadConsumer::Found {
            send: user,
            consumer: defining_op(to.val(), scope),
        };
    }

    // `else if (isa<SelectOp, ShuffleOp, RotateOp>(user) && user->hasOneUse())`, then
    // `dyn_cast<SendOp>(*user->getUsers().begin())`.
    if Rearrangement::of(user).is_some()
        && let [rearranged] = results(user).as_slice()
        && let [send] = uses(*rearranged, scope).as_slice()
        && let DfirOp::Dataflow(dfir_op::dataflow::Op::Send { to, .. }) = send
    {
        return LoadConsumer::Found {
            send,
            consumer: defining_op(to.val(), scope),
        };
    }

    LoadConsumer::Unsupported
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 034/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT `setldtype` DECIDED — one variant per outcome the reference has.
///
/// ⛔ NOT A `Result`, AND NOT AN `Option`. The reference has FOUR outcomes: success having written
/// nothing, success having written a mode and rewritten `total_elements`, and two DIFFERENT
/// `emitOpError`s. A `Result<Option<..>>` would collapse the two diagnostics, and this crate has no
/// error type at all — so the function's own return names its outcomes and `#[must_use]` stops a
/// caller dropping one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum LdType {
    /// `LogicalResult::success()` with nothing written: the component is not an LX half, or the send
    /// is already at stick granularity and no shuffle asked for anything else.
    Default,
    /// The non-default `ldtype`, and the `total_elements` the reference rewrote to a FULL STICK.
    NonDefault {
        /// What the `shuffle_mode` attribute becomes.
        shuffle_mode: sen::ShuffleMode,
        /// `total_elements = sysDef.bytesPerStick * 8 / element_width`.
        total_elements: Elements,
    },
    /// `emitOpError("LX loads involving explicit padding should be at stick granularity")`
    /// (`Helper.cpp:1678-1681`).
    PaddingNotStickGranular,
    /// `emitOpError("unsupported ldtype")` — either the explicit arm (`:1688`) or the implicit one
    /// (`:1705`).
    UnsupportedLdType,
}

/// Replaces: e034_setldtype
///
/// **034/384** `AgenToSentientLoweringPass::setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647` (60L).
///
/// ```cpp
/// LogicalResult AgenToSentientLoweringPass::setldtype(SenComponents comp, Operation* consumer,
///     int& total_elements, unsigned& element_width, std::string& shuffle_mode) {
///   // non-default ldtypes currently support for LX only
///   if (!is_any_of(comp, LXLU, LXSU)) return LogicalResult::success();
///   DT_CHECK_MSG(isa<dataflow::SendOp>(consumer), "consumer must be SendOp");
///   auto send_op = cast<dataflow::SendOp>(consumer);
///   SenSystemDef sysDef;
///   if (auto shuffle_op = dyn_cast_or_null<vectorchain::ShuffleOp>(send_op.getSendData().getDefiningOp())) {
///     total_elements = sysDef.bytesPerStick * 8 / element_width;
///     int repetitions = shuffle_op.getRepetition();
///     if (isSplatFromFirstElem(shuffle_op, 2) && repetitions == 64) {
///       shuffle_mode = "splat2b";
///     } else if (isRightZeroPadFromFirstElem(shuffle_op, 16) && repetitions == 1) {
///       int num_elems = dataflow::utils::getNumElements(shuffle_op.getResult().getType());
///       unsigned result_bitwidth = dataflow::utils::getElementTypeBitWidth(shuffle_op.getResult().getType());
///       if (result_bitwidth * num_elems / 8 != sysDef.bytesPerStick) {
///         return shuffle_op->emitOpError("LX loads involving explicit padding should be at stick granularity");
///       }
///       shuffle_mode = "zpad16b";
///     } else if (isSplatFromFirstElem(shuffle_op, 16) && repetitions == 8) {
///       shuffle_mode = "splat16b";
///     } else {
///       return shuffle_op->emitOpError("unsupported ldtype");
///     }
///   } else if (element_width * total_elements / 8 != sysDef.bytesPerStick) {
///     int num_bytes = element_width * total_elements / 8;
///     if (num_bytes == 2)        { shuffle_mode = "splat2b"; }
///     else if (num_bytes == 16)  { shuffle_mode = "zpad16b"; }
///     else { return consumer->emitOpError("unsupported ldtype"); }
///     total_elements = sysDef.bytesPerStick * 8 / element_width;
///   }
///   return LogicalResult::success();
/// }
/// ```
///
/// ⛔⛔ THE ZERO-PAD ARM READS THE SHUFFLE'S **RESULT**, NOT ITS INPUT, and that is the check a
/// previous attempt documented as prose and did not write: an explicit right-zero-pad is only legal
/// when the shuffle's own result is exactly one stick wide. `isRightZeroPadFromFirstElem` looked at
/// the INPUT's element width; this looks at the RESULT's, and the two differ whenever the pad widens.
///
/// ⛔ THE `DT_CHECK_MSG(isa<SendOp>(consumer))` IS UNREPRESENTABLE HERE. The parameter is the send's
/// `getSendData()` value, which only a send has, so a non-send cannot be passed.
///
/// ⛔ AND `element_width` IS `NonZeroU32` BECAUSE IT IS A DIVISOR. `bytesPerStick * 8 / element_width`
/// with a zero width is the reference's own division by zero; here it is not expressible.
///
/// ⭐ THE REFERENCE REWRITES ITS CALLER'S `total_elements` BEFORE it classifies, so on the two
/// explicit failures the caller's variable is already clobbered. That is invisible: both failures
/// propagate and the caller stops. The rewritten value is carried only on [`LdType::NonDefault`].
pub fn setldtype<A: Arch>(
    comp: GenericComp,
    send_data: Val,
    scope: &[DfirOp],
    total_elements: Elements,
    element_width: NonZeroU32,
) -> LdType {
    // non-default ldtypes currently support for LX only
    if !matches!(comp, GenericComp::Lxlu | GenericComp::Lxsu) {
        return LdType::Default;
    }

    // `total_elements should reflect full stick. element_width is in bits.`
    let full_stick = Elements(A::BYTES_PER_STICK.get() * 8 / u64::from(element_width.get()));

    // `dyn_cast_or_null<vectorchain::ShuffleOp>(send_op.getSendData().getDefiningOp())` — the null
    // is the send's data being a region argument, which is the same arm as it being any other op.
    if let Some(DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle {
        indices,
        repetition,
        input_ty,
        ty,
        ..
    })) = defining_op(send_data, scope)
    {
        let repetitions = *repetition;
        if is_splat_from_first_elem(indices, *input_ty, Bytes(2)) && repetitions == 64 {
            // 2B 64way splat ldtype (mode 1)
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::Splat2B,
                total_elements: full_stick,
            }
        } else if is_right_zero_pad_from_first_elem(indices, *input_ty, Bytes(16))
            && repetitions == 1
        {
            // Return type of shuffle should reflect a stick.
            if u64::from(ty.elem.bits()) * ty.len / 8 == A::BYTES_PER_STICK.get() {
                // 16B 0 pad ldtype (mode 2)
                LdType::NonDefault {
                    shuffle_mode: sen::ShuffleMode::ZeroPad16B,
                    total_elements: full_stick,
                }
            } else {
                LdType::PaddingNotStickGranular
            }
        } else if is_splat_from_first_elem(indices, *input_ty, Bytes(16)) && repetitions == 8 {
            // 16B 8way splat ldtype (mode 3)
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::Splat16B,
                total_elements: full_stick,
            }
        } else {
            LdType::UnsupportedLdType
        }
    } else {
        // If sendOp isn't at stick granularity, set ldtype appropriately. If a shuffleOp wasn't used
        // to explicitly state a non-default ldtype, assume user doesn't care how data is arranged
        // (pad versus splat). In this case for 16B data transfers, use zpad16b (mode 2). 2B data
        // transfers only support splat.
        let num_bytes = u64::from(element_width.get()) * total_elements.0 / 8;
        if num_bytes == A::BYTES_PER_STICK.get() {
            return LdType::Default;
        }
        match num_bytes {
            // 2B 64way splat ldtype (mode 1)
            2 => LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::Splat2B,
                total_elements: full_stick,
            },
            // 16B zero pad ldtype (mode 2)
            16 => LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::ZeroPad16B,
                total_elements: full_stick,
            },
            _ => LdType::UnsupportedLdType,
        }
    }
}

/// `vectorchain::utils::isSplatFromFirstElem` (`dialect_utils/VectorChain/Utils.cpp:181`) — a
/// private helper of [`setldtype`], not a scheduled unit.
///
/// ```cpp
/// Type elem_type = dataflow::utils::getElementType(shuffle_op.getInput().getType());
/// if (!dataflow::isIntOrFloatType(elem_type)) return false;
/// unsigned input_element_bitwidth = dataflow::getIntOrFloatBitWidth(elem_type);
/// unsigned num_bits = num_bytes * 8;
/// unsigned num_elements = num_bits / input_element_bitwidth;
/// if (num_elements == 0 || (num_bits % input_element_bitwidth != 0) ||
///     indices.size() != num_elements) return false;
/// for (unsigned i = 0; i < num_elements; ++i) { if (indices[i] != i) return false; }
/// return true;
/// ```
///
/// ⛔ `isIntOrFloatType` IS A TYPE GUARD HERE, NOT A TEST. It *"handles both builtin and custom
/// dataflow types such as CustomMXFloatType"* (`Dialect/Dataflow/Utils.h:30-32`), and every variant
/// of our `ElemType` is an integer, a float or an MX float — so the early `false` has no input that
/// reaches it. It is the element WIDTH that can still be degenerate, which the next line covers.
fn is_splat_from_first_elem(indices: &[i32], input_ty: Vector, num_bytes: Bytes) -> bool {
    let input_element_bitwidth = u64::from(input_ty.elem.bits());
    // ⛔ THE REFERENCE DIVIDES BY THIS WITHOUT GUARDING IT. MLIR has no zero-width element type, but
    // `ElemType::Int(0)` is spellable in this island, so the guard is here rather than a panic.
    if input_element_bitwidth == 0 {
        return false;
    }
    let num_bits = num_bytes.0 * 8;
    let num_elements = num_bits / input_element_bitwidth;

    if num_elements == 0
        || !num_bits.is_multiple_of(input_element_bitwidth)
        || indices.len() as u64 != num_elements
    {
        return false;
    }

    // `indices[i] != i` — the first `num_elements` lanes read the input's first `num_elements`
    // lanes, in order.
    indices
        .iter()
        .enumerate()
        .take(num_elements as usize)
        .all(|(i, index)| i64::from(*index) == i as i64)
}

/// `vectorchain::utils::isRightZeroPadFromFirstElem` (`dialect_utils/VectorChain/Utils.cpp:217`) — a
/// private helper of [`setldtype`], not a scheduled unit.
///
/// ```cpp
/// ... same prologue as isSplatFromFirstElem ...
/// if ((num_elements == 0) || (num_bits % input_element_bitwidth != 0)) return false;
/// for (unsigned i = 0; i < num_elements; ++i) if (indices[i] != i) return false;
/// int indices_size = indices.size();
/// for (unsigned i = num_elements; i < indices_size; ++i) if (indices[i] != -1) return false;
/// return true;
/// ```
///
/// ⛔ NO SIZE EQUALITY, WHICH IS THE POINT — a pad is LONGER than the lanes it keeps, and the tail
/// is `-1`. The reference then indexes `indices[i]` for `i < num_elements` without checking the
/// vector is that long; a shorter list is a right zero pad of something it does not contain, so
/// here it is `false` rather than an out-of-bounds read.
fn is_right_zero_pad_from_first_elem(indices: &[i32], input_ty: Vector, num_bytes: Bytes) -> bool {
    let input_element_bitwidth = u64::from(input_ty.elem.bits());
    if input_element_bitwidth == 0 {
        return false;
    }
    let num_bits = num_bytes.0 * 8;
    let num_elements = num_bits / input_element_bitwidth;

    if num_elements == 0 || !num_bits.is_multiple_of(input_element_bitwidth) {
        return false;
    }
    let kept = num_elements as usize;
    if indices.len() < kept {
        return false;
    }

    indices[..kept]
        .iter()
        .enumerate()
        .all(|(i, index)| i64::from(*index) == i as i64)
        && indices[kept..].iter().all(|index| *index == -1)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 035/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE UNITS A LOAD'S CONSUMER NAMES — the two shapes `consumer_op` can have, and neither.
///
/// ⛔⛔ THE `uniform::QueryMapOp` ARM IS A LOOKUP THIS CRATE RESOLVES AT COMPILE TIME. The reference
/// walks `getAllQueriedValues` and refuses with *"vector_loadOp's consumer is not a getUnitOp."*
/// when one of them is not a `get_unit` (`Helper.cpp:2755`). Here the queried values ARE units by
/// type, so that refusal has no input — what the port keeps is the RULE the walk feeds: collect
/// every consumer's component, then test the set.
///
/// ⛔ AND [`Self::Neither`] IS A REAL CASE, NOT A PLACEHOLDER. `consumer_comps` stays empty when
/// `consumer_op` is neither op — including when it is NULL, which is exactly what
/// [`LoadConsumer::Found`] carries for a send whose `to` is a region argument. The reference then
/// falls through both `areAnyOf` tests and returns success having emitted nothing; there is no
/// `else` at `Helper.cpp:2745-2760`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsumerUnits {
    /// `dyn_cast<dataflow::GetUnitOp>(consumer_op)` — one unit.
    Bound(DfirUnit),
    /// `dyn_cast<uniform::QueryMapOp>(consumer_op)` — every unit the query names.
    Queried(Vec<DfirUnit>),
    /// Neither, or no consumer op at all.
    Neither,
}

impl ConsumerUnits {
    /// THE GENERIC COMPONENTS — `dcc::getUnitType(consumer_get_unit)`, which is
    /// `senCompToGenericComp` applied to the `type=` (`Utils/DccExtContext.cpp:126-131`).
    #[must_use]
    pub fn comps(&self) -> Vec<GenericComp> {
        match self {
            Self::Bound(unit) => vec![unit.generic()],
            Self::Queried(units) => units.iter().map(|unit| unit.generic()).collect(),
            Self::Neither => Vec::new(),
        }
    }
}

/// WHAT `generateSetSendDestinationStmts` DECIDED.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum SetSendDestination {
    /// `LogicalResult::success()` with nothing inserted.
    Nothing,
    /// The `sentient.set_send_dst` to insert before the load.
    ///
    /// ⭐ BOXED, for the reason [`crate::islands::dataflow_ir::dialects::agen::Op::CompositeLoadAndStore`]
    /// is: an enum is as wide as its largest variant, and a `SenOp` is many words while the other two
    /// outcomes are none.
    Emit(Box<SenOp>),
    /// `emitError("load consumer requires generating SETDSTMASK instruction which is not supported
    /// at the current arch level")`.
    NoSetDstMaskAtThisArchLevel,
}

/// `dccExtContext().getArch() >= RCUDD1A_ISA` (`Helper.cpp:2768`).
///
/// ⛔⛔ TRUE ON EVERY ARCH THIS CRATE BUILDS FOR, AND THAT IS A COMPILE-TIME FACT RATHER THAN AN
/// ASSUMPTION. `IsaCoreGen` is ordered `MPW2 < MPW3 < MPW4 < RCUDD1A < SEN1P5` with
/// `DEFAULT_ISA = RCUDD1A_ISA` (`sys-arch-spec/isa/isa.hpp:24-35`), and [`IsaGen`] models the last
/// two only. The match is exhaustive, so adding an older generation to [`IsaGen`] stops the build
/// here instead of silently answering `true` for it.
const fn supports_set_dst_mask(isa: IsaGen) -> bool {
    match isa {
        IsaGen::Rcudd1a | IsaGen::Sen1p5 => true,
    }
}

/// `areAnyOf` (`Helper.cpp:2739-2745`) — is any consumer component one of `reference`?
fn are_any_of(consumer_comps: &[GenericComp], reference: &[GenericComp]) -> bool {
    consumer_comps.iter().any(|comp| reference.contains(comp))
}

/// Replaces: e035_generateSetSendDestinationStmts
///
/// **035/384** `AgenToSentientLoweringPass::generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731` (49L).
///
/// ```cpp
/// auto comp = dcc::getUnitType(unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
/// if (comp != LXLU) return LogicalResult::success();
/// auto areAnyOf = [](std::vector<SenComponents>& consumer_comps, std::vector<SenComponents> reference) {
///   for (auto consumer_comp : consumer_comps) if (is_any_of(consumer_comp, reference)) return true;
///   return false;
/// };
/// std::vector<SenComponents> consumer_comps;
/// if (auto consumer_get_unit = dyn_cast<dataflow::GetUnitOp>(consumer_op)) {
///   consumer_comps.push_back(dcc::getUnitType(consumer_get_unit));
/// } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(consumer_op)) {
///   auto map_op = query_op.getMap().getDefiningOp<uniform::DefImmutableMappingOp>();
///   llvm::SmallVector<Value> units;
///   query_op.getAllQueriedValues(units);
///   for (auto unit : units) {
///     if (auto consumer_get_unit = unit.getDefiningOp<dataflow::GetUnitOp>()) {
///       consumer_comps.push_back(dcc::getUnitType(consumer_get_unit));
///     } else {
///       return op->emitError("vector_loadOp's consumer is not a getUnitOp.");
///     }
///   }
/// }
/// // If consumer is bypassing SFP, we need to insert set_send_dst operations
/// // to make the routing explicit. Redundant set_send_dst operations will be
/// // eliminated by a later transform.
/// if (areAnyOf(consumer_comps, {PT, SFP, L0SU, CROSSPTNLINK})) {
///   if (dccExtContext().getArch() >= RCUDD1A_ISA) {
///     (void)sentient::SetSendDestinationOp::create(builder, op->getLoc(), consumer_op->getResult(0));
///   } else if (areAnyOf(consumer_comps, {PT, L0SU, CROSSPTNLINK})) {
///     return op->emitError("load consumer requires generating SETDSTMASK instruction which is not supported at the current arch level");
///   }
/// }
/// return LogicalResult::success();
/// ```
///
/// ⛔⛔ THE OP IT EMITS **IS** THE FUNCTION. A previous attempt at this entry documented the bypass
/// condition and emitted nothing, so no `sentient.set_send_dst` ever reached a program — while the
/// reference's own golden expects one before every load that bypasses the SFP:
/// `sentient.set_send_dst(%[[VAL_10]])` on the `ptrow0` handle and `(%[[VAL_9]])` on the `l0su` one
/// (`lx-to-sfp-bypass-1.mlir:47,55,63`).
///
/// ⛔ `map_op` IS DEAD IN THE REFERENCE. `query_op.getMap().getDefiningOp<DefImmutableMappingOp>()`
/// is bound and never read — the walk uses `getAllQueriedValues` instead — so it carries no
/// behaviour to port.
///
/// ⭐ `comp` IS THE **GENERIC** COMPONENT, so the `!= LXLU` gate is `lxlu` and nothing else — an
/// `lxsu` program never gets one of these, and neither does an L3 half.
///
/// ⭐ AND THE FOUR-WAY TEST IS OVER GENERIC COMPONENTS TOO, which is why one arm covers all eight PT
/// rows: `senCompToGenericComp` sends every `PTROW*` spelling to `PT` (`arch_enums.cpp:127-152`).
///
/// ⛔ THE OPERAND IS THE SEND'S OWN `to`. `consumer_op->getResult(0)` is the value the send names,
/// so a [`SendEnd`] is what this takes — the set_send_dst cannot be pointed at a unit the send does
/// not go to.
pub fn generate_set_send_destination_stmts<A: Arch>(
    on: GenericComp,
    consumer: &ConsumerUnits,
    to: SendEnd,
) -> SetSendDestination {
    if !matches!(on, GenericComp::Lxlu) {
        return SetSendDestination::Nothing;
    }

    let consumer_comps = consumer.comps();

    // If consumer is bypassing SFP, we need to insert set_send_dst operations to make the routing
    // explicit. Redundant set_send_dst operations will be eliminated by a later transform.
    let bypasses_sfp = are_any_of(
        &consumer_comps,
        &[
            GenericComp::Pt,
            GenericComp::Sfp,
            GenericComp::L0su,
            GenericComp::CrossPtnLink,
        ],
    );
    if !bypasses_sfp {
        return SetSendDestination::Nothing;
    }

    if supports_set_dst_mask(A::GEN) {
        SetSendDestination::Emit(Box::new(SenOp::Sentient(sen::Op::SetSendDst { units: to })))
    } else if are_any_of(
        &consumer_comps,
        // ⛔ THE SFP IS ABSENT FROM THIS SECOND LIST ON PURPOSE. An SFP consumer needs no
        // SETDSTMASK, so on an arch without one it is silently fine; the other three are not.
        &[
            GenericComp::Pt,
            GenericComp::L0su,
            GenericComp::CrossPtnLink,
        ],
    ) {
        // ⛔ UNREACHABLE ON EVERY ARCH THIS CRATE BUILDS FOR — see [`supports_set_dst_mask`]. It is
        // written because it is the function, and it is where an older generation lands the day one
        // is added to `IsaGen`.
        SetSendDestination::NoSetDstMaskAtThisArchLevel
    } else {
        SetSendDestination::Nothing
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 036/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e036_getStoreOpFromLoadStorePattern
///
/// **036/384** `AgenToSentientLoweringPass::getStoreOpFromLoadStorePattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872` (8L).
///
/// ```cpp
/// template <typename VectorStoreTy>
/// Operation* AgenToSentientLoweringPass::getStoreOpFromLoadStorePattern(Operation* op) {
///   Operation* store_op = nullptr;
///   DT_CHECK(op->getNumResults() == 1);
///   auto result = op->getResult(0);
///   if (result.hasOneUse())
///     store_op = dyn_cast<VectorStoreTy>(*result.user_begin());
///   return store_op;
/// }
/// ```
///
/// ⭐ THE NULL RETURN IS THE ANSWER, NOT A REFUSAL. "This load does not feed a store of that class"
/// is the question the callers ask — `lowerVectorLoadOp` uses it to tell a load-and-store pattern
/// from a plain load (`Helper.cpp:3053,3067`) — so `Option` is the exact translation.
///
/// ⛔ TWO INSTANTIATIONS, AND THE CLASS IS A PARAMETER: `VectorStoreOp` (`:3053, :3067`) and
/// `SymbolicVectorStoreOp` (`:3384`). Passing the wrong one finds nothing rather than the wrong op.
///
/// ⛔ `DT_CHECK(op->getNumResults() == 1)` IS ANSWERED BY THE USE CENSUS. A load with any other
/// arity has no single result to trace, which is the same outcome as having no store — minus the
/// abort.
#[must_use]
pub fn store_op_from_load_store_pattern<'a>(
    store_kind: AgenOpKind,
    op: &DfirOp,
    scope: &'a [DfirOp],
) -> Option<&'a DfirOp> {
    let bound = results(op);
    let [result] = bound.as_slice() else {
        return None;
    };
    // `if (result.hasOneUse()) store_op = dyn_cast<VectorStoreTy>(*result.user_begin());`
    let users = uses(*result, scope);
    let [user] = users.as_slice() else {
        return None;
    };
    if agen_op_kind(user) == Some(store_kind) {
        Some(user)
    } else {
        None
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 037/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH STATEMENTS CARRY THE `"marked"` ATTRIBUTE, BY PRE-ORDER POSITION.
///
/// ⛔⛔ THE MARK IS A REAL PART OF THE ALGORITHM, NOT BOOKKEEPING. `gatherAffineLoadStoreDetails`
/// sets it on the transfer it processed (`Helper.cpp:546`, and `:848` on the outermost loop), and
/// `lowerAffineCompositeHelper` re-finds the marked op after building the loops *because* "The op
/// may have changed due to loop cloning" (`Helper.cpp:2963`) — the mark is how an op survives its
/// own pointer being invalidated.
///
/// ⭐ SO THE IDENTITY IS THE PRE-ORDER POSITION, which is the identity the walk itself uses: the
/// statements of a body numbered `0..n` in `WalkOrder::PreOrder`, each op counted before the
/// statements of its regions. Our ops carry no attribute dictionary, and hanging one on them to
/// store a transient mark would put pass state into the IR.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Marked(BTreeSet<usize>);

impl Marked {
    /// The statements at these pre-order positions carry the mark.
    #[must_use]
    pub fn at(positions: impl IntoIterator<Item = usize>) -> Marked {
        Marked(positions.into_iter().collect())
    }

    /// `op->hasAttr("marked")`.
    #[must_use]
    pub fn holds(&self, position: usize) -> bool {
        self.0.contains(&position)
    }
}

/// THE OP `findCandidateForLowering` SETTLED ON, and where it sits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate<'a> {
    /// Its pre-order position, which is its identity for [`Marked`].
    pub position: usize,
    /// The op itself.
    pub op: &'a DfirOp,
}

/// Replaces: e037_findCandidateForLowering
///
/// **037/384** `AgenToSentientLoweringPass::findCandidateForLowering` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884` (12L).
///
/// ```cpp
/// template <typename OpTy>
/// OpTy AgenToSentientLoweringPass::findCandidateForLowering(ProgramUnitOp& unit) {
///   OpTy candidate_op = nullptr;
///   unit.walk<WalkOrder::PreOrder>([&](OpTy op) {
///     if (op->hasAttr("marked")) {
///       candidate_op = op;
///       return WalkResult::interrupt();
///     }
///     return WalkResult::advance();
///   });
///   DT_CHECK(candidate_op);
///   return candidate_op;
/// }
/// ```
///
/// ⭐ THE FIRST MARKED OP OF THAT CLASS IN PRE-ORDER, AND `interrupt()` MEANS IT STOPS THERE. A
/// second marked op of the same class is not an error and is not visited — so the walk order is
/// load-bearing, not incidental.
///
/// ⛔ THE CLASS FILTER IS THE TEMPLATE PARAMETER. `unit.walk<OpTy>` visits only ops of that class,
/// which is why the enclosing `program_unit` never matches itself.
///
/// ⛔ `DT_CHECK(candidate_op)` IS AN ABORT WHEN NOTHING IS MARKED. `None` is that outcome: the
/// callers reach this only after `gatherAffineLoadStoreDetails` marked the op they are lowering, so
/// it is unreachable from the reference's own call sites.
#[must_use]
pub fn find_candidate_for_lowering<'a>(
    kind: AgenOpKind,
    marked: &Marked,
    unit: &'a [DfirOp],
) -> Option<Candidate<'a>> {
    let mut position = 0usize;
    walk_pre_order(unit, &mut position, &mut |position, op| {
        if agen_op_kind(op) == Some(kind) && marked.holds(position) {
            Some(Candidate { position, op })
        } else {
            None
        }
    })
}

/// `WalkOrder::PreOrder` over a body — each op numbered before its own regions' statements, and the
/// first `Some` interrupts the walk.
fn walk_pre_order<'a, T>(
    body: &'a [DfirOp],
    position: &mut usize,
    found: &mut impl FnMut(usize, &'a DfirOp) -> Option<T>,
) -> Option<T> {
    for op in body {
        let here = *position;
        *position += 1;
        if let Some(hit) = found(here, op) {
            return Some(hit);
        }
        for region in dfir_op::regions(op) {
            if let Some(hit) = walk_pre_order(region, position, found) {
                return Some(hit);
            }
        }
    }
    None
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 038/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e038_addLoadChainToDeleteList
///
/// **038/384** `AgenToSentientLoweringPass::addLoadChainToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975` (9L).
///
/// ```cpp
/// void AgenToSentientLoweringPass::addLoadChainToDeleteList(
///     Operation* op, SmallVectorImpl<Operation*>& to_be_deleted) {
///   // TODO: Proper model the handling of select op, for, e.g., sizes.
///   for (auto* user : op->getUsers()) {
///     if (isa<vectorchain::SelectOp, vectorchain::ShuffleOp, vectorchain::RotateOp>(user))
///       to_be_deleted.push_back(*user->getUsers().begin());
///     to_be_deleted.push_back(user);
///   }
/// }
/// ```
///
/// ⭐⭐ THE ORDER IS THE REFERENCE'S: a rearrangement's own user (the send) goes on the list BEFORE
/// the rearrangement itself, so the chain is torn down consumer-first. Reversing it would leave a
/// deleted value with a live use for as long as the list is walked.
///
/// ⛔ IT APPENDS AND DOES NOT CLEAR. The parameter is `SmallVectorImpl&` and several callers add to
/// one list across a whole lowering (`Helper.cpp:2917, 3210`).
///
/// ⛔ ONE ENTRY PER **USE**. `getUsers()` is a mapped range over the use list, so an op that reads
/// the load twice is pushed twice — and the reference does not deduplicate.
///
/// ⛔ THE REFERENCE TAKES THE OP AND WALKS THE USERS OF **ALL** ITS RESULTS; this takes the one value
/// whose chain is being torn down. Both call sites pass a single-result load — `load_op`
/// (`Helper.cpp:2917`) and `candidate_op` (`:3210`) — so the two agree, and naming the value makes
/// which chain is meant explicit rather than implied by the op's arity.
///
/// ⛔ AND `*user->getUsers().begin()` ON A REARRANGEMENT WITH NO USERS IS THE REFERENCE
/// DEREFERENCING THE END OF A RANGE. A rearrangement whose result nothing reads contributes only
/// itself here, rather than an unspecified pointer.
pub fn add_load_chain_to_delete_list<'a>(
    root: Val,
    scope: &'a [DfirOp],
    to_be_deleted: &mut Vec<&'a DfirOp>,
) {
    // TODO: Proper model the handling of select op, for, e.g., sizes.
    for user in uses(root, scope) {
        if Rearrangement::of(user).is_some()
            // `*user->getUsers().begin()` — a rearrangement binds one result, so the users of the OP
            // are the users of that result, and the reference takes the first of them.
            && let Some(rearranged) = results(user).first()
            && let Some(onward) = uses(*rearranged, scope).first()
        {
            to_be_deleted.push(onward);
        }
        to_be_deleted.push(user);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::SyncSignal;
    use crate::islands::dataflow_ir::dialects::{
        Index, affine, agen, arith, dataflow, vectorchain,
    };
    use crate::islands::dataflow_ir::link::{
        CrossPtnLink as CrossPtnLinkUnit, L0su as L0suUnit, Link, Lxlu as LxluUnit, PtRowUnit,
        Sfp as SfpUnit,
    };
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef};
    use crate::units::{Core, Corelet, Residency, Row};

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

    /// `%1 = dataflow.get_unit {core = 0, corelet = 0, name = "C0-CL0-LX-LU", type = "lxlu"}`.
    const LXLU: Val = Val(4);
    /// `%l0su = dataflow.get_unit {..., type = "l0su"}` — `%[[VAL_9]]` in the expectation.
    const L0SU: Val = Val(7);
    /// `%pt = dataflow.get_unit {..., name = "C0-CL0-PT-0", type = "ptrow0"}` — `%[[VAL_10]]`.
    const PT: Val = Val(8);
    /// `%8 = dataflow.get_logical_memory_view %3, %c0_0 {layout_map = #map1}`.
    const VIEW: Val = Val(10);

    /// `vector<1x1x1x128xi8>`, flat: 128 lanes of `i8`.
    const LANES: Vector = Vector {
        len: 128,
        elem: ElemType::Int(8),
    };

    /// `0 : i32` core and corelet, which every unit in the golden carries.
    fn at_corelet_zero() -> Residency {
        Residency::Corelet {
            core: Core::checked(0).expect("core 0 exists"),
            corelet: Corelet::checked(0).expect("corelet 0 exists"),
        }
    }

    /// `%8`'s type. ⛔ THE GOLDEN WRITES `memref<?x?x?x128xi8>` AND OUR SHAPE IS STATIC, so the three
    /// dynamic extents are written as their trip count of 1. Nothing under test reads the shape; the
    /// lane count, which every one of them does read, is the same 128.
    fn view_ty() -> MemRef {
        MemRef {
            shape: vec![1, 1, 1, 128],
            elem: ElemType::Int(8),
        }
    }

    /// `#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 128 + d1 * 4096 + d0 * 8192)>`.
    fn layout() -> AffineMap {
        AffineMap {
            dims: 4,
            results: vec![
                AffineExpr::dim(3)
                    .plus(AffineExpr::dim(2).times(128))
                    .plus(AffineExpr::dim(1).times(4096))
                    .plus(AffineExpr::dim(0).times(8192)),
            ],
        }
    }

    /// `%8[%arg0, %arg1, %arg2, %c0_0]`.
    fn indices(iv: Val) -> Vec<Index> {
        vec![
            Index::Val(Val(20)),
            Index::Val(Val(21)),
            Index::Val(iv),
            Index::Val(Val(1)),
        ]
    }

    /// ONE OF THE GOLDEN'S THREE INNERMOST LOOPS — `affine.for %arg2 = 0 to %c1 { vector_load; send }`.
    fn load_and_send(iv: Val, loaded: Val, to: SendEnd) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Val(Val(0)),
            // The golden's loops are plain counted ones: they carry nothing.
            carried: Vec::new(),
            body: vec![
                DfirOp::Agen(agen::Op::VectorLoad {
                    result: loaded,
                    view: VIEW,
                    indices: indices(iv),
                    view_ty: view_ty(),
                    ty: LANES,
                }),
                DfirOp::Dataflow(dataflow::Op::Send {
                    to,
                    data: loaded,
                    ty: LANES,
                }),
            ],
        })
    }

    /// ⭐⭐ THE REFERENCE'S OWN TEST PROGRAM, TYPED — the input half of
    /// `dcc/test/Conversion/AgenToSentient/lx-to-sfp-bypass-1.mlir:98-141`, whose `RUN` line is
    /// `SENARCH=rcudd1a dcc-opt --dcc-agen-to-sentient`.
    ///
    /// ⛔ THE UNITS ARE AT THE TOP LEVEL AND THE SENDS ARE THREE LOOPS DEEP, which is the whole reason
    /// the search takes a scope rather than a block: `defining_op` walks UP to the `get_unit` while
    /// `uses` walks DOWN to the send.
    fn lx_to_sfp_bypass_1() -> Vec<DfirOp> {
        let unit = |result: Val, unit: DfirUnit| {
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result,
                residency: at_corelet_zero(),
                unit,
            })
        };
        // `dataflow.send %pt, %9` — one end of an LXLU-to-PT-row-0 link, which is the only way a send
        // can name a destination in this island.
        let (to_pt, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let (to_l0su, _) = Link::<LxluUnit, L0suUnit>::between(LXLU, L0SU).ends();

        vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 1,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 0,
            }),
            unit(Val(3), DfirUnit::Sfp),
            unit(LXLU, DfirUnit::Lxlu),
            unit(Val(5), DfirUnit::Lxsu),
            unit(Val(6), DfirUnit::Lx),
            unit(L0SU, DfirUnit::L0su),
            unit(PT, DfirUnit::PtRow(Row::checked(0).expect("row 0 exists"))),
            unit(Val(9), DfirUnit::Pe),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: VIEW,
                from: Val(6),
                start: Val(1),
                layout: layout(),
                ty: view_ty(),
            }),
            DfirOp::Dataflow(dataflow::Op::ProgramUnit {
                units: vec![LXLU],
                precision: None,
                body: vec![DfirOp::Affine(affine::Op::For {
                    iv: Val(20),
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Val(Val(0)),
                    // The golden's loops are plain counted ones: they carry nothing.
                    carried: Vec::new(),
                    body: vec![DfirOp::Affine(affine::Op::For {
                        iv: Val(21),
                        lo: affine::Bound::Const(0),
                        hi: affine::Bound::Val(Val(0)),
                        // The golden's loops are plain counted ones: they carry nothing.
                        carried: Vec::new(),
                        body: vec![
                            DfirOp::Dataflow(dataflow::Op::SyncRecv {
                                from: Val(5),
                                signal: SyncSignal::InputToLxsuToLxluToSync,
                            }),
                            load_and_send(Val(30), Val(31), to_pt),
                            DfirOp::Dataflow(dataflow::Op::SyncRecv {
                                from: Val(5),
                                signal: SyncSignal::InputToLxsuToLxluToSync,
                            }),
                            load_and_send(Val(32), Val(33), to_pt),
                            DfirOp::Dataflow(dataflow::Op::SyncRecv {
                                from: Val(5),
                                signal: SyncSignal::InputToLxsuToLxluToSync,
                            }),
                            load_and_send(Val(34), Val(35), to_l0su),
                        ],
                    })],
                })],
            }),
        ]
    }

    /// The units a send's resolved consumer op names — `dyn_cast<GetUnitOp>(consumer_op)`.
    fn consumer_units(consumer: Option<&DfirOp>) -> ConsumerUnits {
        match consumer {
            Some(DfirOp::Dataflow(dataflow::Op::GetUnit { unit, .. })) => {
                ConsumerUnits::Bound(*unit)
            }
            _ => ConsumerUnits::Neither,
        }
    }

    /// 🎯 033/384 + 035/384 — THE REFERENCE'S OWN GOLDEN: EVERY LX LOAD THAT BYPASSES THE SFP GETS A
    /// `sentient.set_send_dst` NAMING ITS CONSUMER'S HANDLE.
    ///
    /// ⛔⛔ THE EXPECTATION IS THE VENDOR'S, VERBATIM. `lx-to-sfp-bypass-1.mlir:47,55,63` are
    /// `sentient.set_send_dst(%[[VAL_10]])`, `(%[[VAL_10]])` and `(%[[VAL_9]])` — `VAL_10` being the
    /// `type = "ptrow0"` handle and `VAL_9` the `type = "l0su"` one. Two PT loads and one L0SU load,
    /// each with its own set_send_dst, in program order.
    ///
    /// ⭐ AND IT RUNS THE PAIR END TO END: `load_consumer` finds each send and resolves its
    /// destination handle to the `get_unit` nine levels up, and
    /// `generate_set_send_destination_stmts` turns that into the op.
    #[test]
    fn every_sfp_bypassing_lx_load_gets_a_set_send_dst() {
        let program = lx_to_sfp_bypass_1();
        // The program unit is bound to `%1` (`lxlu`), which is what gates the whole function.
        let on = DfirUnit::Lxlu.generic();

        let mut emitted = String::new();
        for loaded in [Val(31), Val(33), Val(35)] {
            let consumer = load_consumer(AgenLoad::Vector { result: loaded }, &program);
            let LoadConsumer::Found { send, consumer } = consumer else {
                panic!("{loaded:?} is sent exactly once, so its consumer is found: {consumer:?}");
            };
            let DfirOp::Dataflow(dataflow::Op::Send { to, .. }) = send else {
                panic!("the consumer of a load in this program is a send");
            };

            match generate_set_send_destination_stmts::<Dd2>(on, &consumer_units(consumer), *to) {
                SetSendDestination::Emit(op) => {
                    crate::islands::sentient::print::emit(&mut emitted, &op, 0);
                }
                other => {
                    panic!("{loaded:?} bypasses the SFP, so it gets a set_send_dst: {other:?}")
                }
            }
        }

        assert_eq!(
            emitted,
            "sentient.set_send_dst(%8)\n\
             sentient.set_send_dst(%8)\n\
             sentient.set_send_dst(%7)\n",
            "the vendor's expectation is set_send_dst on the ptrow0 handle twice and the l0su handle \
             once (lx-to-sfp-bypass-1.mlir:47,55,63)"
        );
    }

    /// 🎯 033/384 — A LOAD READ TWICE HAS NO SINGLE CONSUMER, AND THAT IS COUNTED, NOT ASSUMED.
    ///
    /// ⛔⛔ THIS IS THE CHECK A PREVIOUS ATTEMPT FAKED. `hasOneUse()` counts USES; a search that only
    /// looked for the send it expected would find one here and report a consumer. The second reader
    /// below is a `vectorchain.multiply` — an op no part of `getLoadConsumer` mentions — and it still
    /// has to make the answer [`LoadConsumer::NotOneUse`].
    #[test]
    fn a_load_read_by_something_else_as_well_has_no_single_consumer() {
        let (to_pt, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let program = vec![
            DfirOp::Agen(agen::Op::VectorLoad {
                result: Val(31),
                view: VIEW,
                indices: indices(Val(30)),
                view_ty: view_ty(),
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to: to_pt,
                data: Val(31),
                ty: LANES,
            }),
        ];
        assert!(matches!(
            load_consumer(AgenLoad::Vector { result: Val(31) }, &program),
            LoadConsumer::Found { .. }
        ));

        let mut also_computed = program.clone();
        also_computed.push(DfirOp::VectorChain(vectorchain::Op::Multiply {
            result: Val(40),
            a: Val(31),
            b: Val(31),
            reduction_map: AffineMap::identity(1),
            operand_ty: LANES,
            ty: LANES,
        }));
        assert_eq!(
            load_consumer(AgenLoad::Vector { result: Val(31) }, &also_computed),
            LoadConsumer::NotOneUse,
            "the load now has three uses — the send and two operands of the multiply"
        );

        // And a load nothing reads is not single-used either.
        assert_eq!(
            load_consumer(AgenLoad::Vector { result: Val(31) }, &program[..1]),
            LoadConsumer::NotOneUse
        );
    }

    /// 🎯 033/384 — A REARRANGEMENT MAY SIT BETWEEN THE LOAD AND THE SEND, AND ONLY IF IT IS
    /// SINGLE-USED ITSELF.
    ///
    /// ⭐ `isa<SelectOp, ShuffleOp, RotateOp>(user) && user->hasOneUse()` (`Helper.cpp:1268-1269`):
    /// the chain is load -> shuffle -> send, and the pair returned is the SEND's, not the shuffle's.
    #[test]
    fn a_shuffle_between_the_load_and_the_send_is_walked_through() {
        let (to_pt, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let shuffled = || {
            DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(41),
                input: Val(31),
                indices: vec![0, 1],
                repetition: 64,
                input_ty: Vector {
                    len: 2,
                    elem: ElemType::Int(8),
                },
                ty: LANES,
            })
        };
        let load = DfirOp::Agen(agen::Op::VectorLoad {
            result: Val(31),
            view: VIEW,
            indices: indices(Val(30)),
            view_ty: view_ty(),
            ty: LANES,
        });
        let pt = DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: PT,
            residency: at_corelet_zero(),
            unit: DfirUnit::PtRow(Row::checked(0).expect("row 0 exists")),
        });

        let through = vec![
            pt.clone(),
            load.clone(),
            shuffled(),
            DfirOp::Dataflow(dataflow::Op::Send {
                to: to_pt,
                data: Val(41),
                ty: LANES,
            }),
        ];
        let found = load_consumer(AgenLoad::Vector { result: Val(31) }, &through);
        let LoadConsumer::Found { send, consumer } = found else {
            panic!("load -> shuffle -> send is the supported chain: {found:?}");
        };
        assert!(matches!(
            send,
            DfirOp::Dataflow(dataflow::Op::Send { data, .. }) if *data == Val(41)
        ));
        assert_eq!(
            consumer,
            Some(&pt),
            "the pair's second half is the get_unit"
        );

        // A shuffle whose result nothing reads is not single-used, so the chain is unsupported.
        assert_eq!(
            load_consumer(
                AgenLoad::Vector { result: Val(31) },
                &[load.clone(), shuffled()]
            ),
            LoadConsumer::Unsupported
        );

        // Nor is a chain whose single onward user is not a send.
        assert_eq!(
            load_consumer(
                AgenLoad::Vector { result: Val(31) },
                &[
                    load,
                    shuffled(),
                    DfirOp::Agen(agen::Op::VectorStore {
                        value: Val(41),
                        view: VIEW,
                        indices: indices(Val(30)),
                        view_ty: view_ty(),
                        ty: LANES,
                    }),
                ]
            ),
            LoadConsumer::Unsupported
        );
    }

    /// 🎯 033/384 — A SEND TO A REGION ARGUMENT HAS NO DEFINING OP, AND THAT IS A SUCCESSFUL PAIR.
    ///
    /// ⛔ `getDefiningOp()` ANSWERS NULL RATHER THAN FAILING, and the reference returns that null as
    /// `consumer_info.second`. [`generate_set_send_destination_stmts`] then sees neither a `get_unit`
    /// nor a `query_map` and emits nothing — which is what [`ConsumerUnits::Neither`] produces.
    #[test]
    fn a_send_to_a_value_no_op_defines_pairs_with_none() {
        // The link's `from`/`to` are values; nothing requires an op in this scope to define them.
        let (to_pt, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, Val(99)).ends();
        let program = vec![
            DfirOp::Agen(agen::Op::VectorLoad {
                result: Val(31),
                view: VIEW,
                indices: indices(Val(30)),
                view_ty: view_ty(),
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to: to_pt,
                data: Val(31),
                ty: LANES,
            }),
        ];
        let found = load_consumer(AgenLoad::Vector { result: Val(31) }, &program);
        assert!(matches!(found, LoadConsumer::Found { consumer: None, .. }));

        assert_eq!(
            generate_set_send_destination_stmts::<Dd2>(
                GenericComp::Lxlu,
                &ConsumerUnits::Neither,
                to_pt
            ),
            SetSendDestination::Nothing,
            "no consumer component is in the bypass set, so there is nothing to make explicit"
        );
    }

    /// 🎯 033/384 — THE COMPOSITE LOADS ROOT AT THEIR INDUCTION VARIABLE, NOT AT A RESULT.
    ///
    /// ⭐ A composite transfer's loaded vector never becomes a top-level SSA value — it arrives on
    /// the region argument `getLoadInductionVar()` (`Helper.cpp:1247-1251`), and the send that
    /// consumes it is INSIDE that region. So the root differs by class and the use is nested.
    #[test]
    fn a_composite_load_roots_at_its_induction_variable() {
        assert_eq!(
            AgenLoad::Composite {
                load_induction_var: Val(70)
            }
            .consumer_root(),
            Val(70)
        );
        assert_eq!(
            AgenLoad::Vector { result: Val(31) }.consumer_root(),
            Val(31)
        );
        assert_eq!(
            AgenLoad::SymbolicVector { result: Val(31) }.consumer_root(),
            Val(31)
        );
    }

    /// 🎯 034/384 — THE THREE EXPLICIT `ldtype`s, EACH WITH ITS OWN SHUFFLE SHAPE AND REPETITION.
    ///
    /// ⭐ THE REPETITION IS PART OF THE PATTERN, NOT A HINT: `splat2b` is 2 bytes repeated 64 times,
    /// `splat16b` is 16 bytes repeated 8, `zpad16b` is 16 bytes and no repetition — and all three
    /// come to one 128-byte stick. A shape with the wrong repetition is not a different mode, it is
    /// `emitOpError("unsupported ldtype")`.
    #[test]
    fn the_three_explicit_ldtypes_are_recognised_by_shape_and_repetition() {
        let bytes = |n: u64| Vector {
            len: n,
            elem: ElemType::Int(8),
        };
        let shuffle = |indices: Vec<i32>, repetition: u32, input_ty: Vector, ty: Vector| {
            vec![DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(41),
                input: Val(31),
                indices,
                repetition,
                input_ty,
                ty,
            })]
        };
        let width = NonZeroU32::new(8).expect("8 is not zero");
        let stick = Elements(128);

        // 2B 64-way splat.
        let splat2b = shuffle(vec![0, 1], 64, bytes(2), bytes(128));
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(41), &splat2b, stick, width),
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::Splat2B,
                total_elements: stick,
            }
        );

        // 16B 8-way splat.
        let splat16b = shuffle((0..16).collect(), 8, bytes(16), bytes(128));
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(41), &splat16b, stick, width),
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::Splat16B,
                total_elements: stick,
            }
        );

        // 16B right zero pad, no repetition — 16 kept lanes then 112 of `-1`.
        let mut padded: Vec<i32> = (0..16).collect();
        padded.extend(std::iter::repeat_n(-1, 112));
        let zpad16b = shuffle(padded.clone(), 1, bytes(16), bytes(128));
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(41), &zpad16b, stick, width),
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::ZeroPad16B,
                total_elements: stick,
            }
        );

        // ⭐ THE RIGHT REPETITION ON THE WRONG PATTERN IS NOT A MODE.
        let confused = shuffle(vec![0, 1], 8, bytes(2), bytes(128));
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(41), &confused, stick, width),
            LdType::UnsupportedLdType
        );

        // ⭐ AND NEITHER IS A PATTERN THAT DOES NOT START AT THE FIRST ELEMENT.
        let offset = shuffle(vec![1, 2], 64, bytes(2), bytes(128));
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(41), &offset, stick, width),
            LdType::UnsupportedLdType
        );
    }

    /// 🎯 034/384 — THE ZERO-PAD ARM READS THE SHUFFLE'S **RESULT** TYPE, WHICH IS ITS OWN CHECK.
    ///
    /// ⛔⛔ THE ONE LINE OF THIS FUNCTION THAT LOOKS AT SOMETHING ELSE. Both `isSplat…` and
    /// `isRightZeroPad…` classify from the shuffle's INPUT; the pad arm then reads
    /// `shuffle_op.getResult().getType()` and refuses unless the RESULT is exactly one stick
    /// (`Helper.cpp:1675-1682`). A 64-byte result passes every pattern test and is still an error.
    #[test]
    fn an_explicit_pad_to_less_than_a_stick_is_an_error() {
        let mut padded: Vec<i32> = (0..16).collect();
        padded.extend(std::iter::repeat_n(-1, 48));
        let half_stick = vec![DfirOp::VectorChain(vectorchain::Op::Shuffle {
            result: Val(41),
            input: Val(31),
            indices: padded,
            repetition: 1,
            input_ty: Vector {
                len: 16,
                elem: ElemType::Int(8),
            },
            // 64 lanes of i8 is 64 bytes — half of DD2's 128-byte stick.
            ty: Vector {
                len: 64,
                elem: ElemType::Int(8),
            },
        })];
        assert_eq!(
            setldtype::<Dd2>(
                GenericComp::Lxlu,
                Val(41),
                &half_stick,
                Elements(128),
                NonZeroU32::new(8).expect("8 is not zero"),
            ),
            LdType::PaddingNotStickGranular
        );
    }

    /// 🎯 034/384 — WITH NO SHUFFLE, THE WIDTH OF THE SEND DECIDES, AND A FULL STICK DECIDES NOTHING.
    ///
    /// ⭐ THE IMPLICIT ROUTE IS AN `else if` ON THE WIDTH (`Helper.cpp:1690`): a send already at stick
    /// granularity keeps the default `noshuffle`, a 2-byte one must splat and a 16-byte one is padded
    /// because "user doesn't care how data is arranged".
    #[test]
    fn without_a_shuffle_the_transfer_width_picks_the_mode() {
        let width = NonZeroU32::new(8).expect("8 is not zero");
        // No op defines the send's data in this scope, which is the `dyn_cast_or_null` null.
        let none: [DfirOp; 0] = [];

        // 128 bytes — a whole stick, so nothing is written.
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(31), &none, Elements(128), width),
            LdType::Default
        );
        // 2 bytes — splat, and `total_elements` becomes a full stick.
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(31), &none, Elements(2), width),
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::Splat2B,
                total_elements: Elements(128),
            }
        );
        // 16 bytes — zero pad.
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(31), &none, Elements(16), width),
            LdType::NonDefault {
                shuffle_mode: sen::ShuffleMode::ZeroPad16B,
                total_elements: Elements(128),
            }
        );
        // Any other short width has no mode.
        assert_eq!(
            setldtype::<Dd2>(GenericComp::Lxlu, Val(31), &none, Elements(8), width),
            LdType::UnsupportedLdType
        );
    }

    /// 🎯 034/384 — NON-DEFAULT `ldtype`s ARE THE LX'S ALONE, AND THE TWO HALVES ARE BOTH IT.
    ///
    /// ⛔⛔ `is_any_of(comp, LXLU, LXSU)` NEEDS THE HALVES TO BE DISTINGUISHABLE FROM EVERY OTHER
    /// COMPONENT AND FROM THE WHOLE LX. `GenericComp::Lx` — the `type = "lx"` handle the golden's
    /// `%3` binds — is NOT in the set, so a memory-level component never gets one.
    #[test]
    fn only_the_lx_halves_get_a_non_default_ldtype() {
        let width = NonZeroU32::new(8).expect("8 is not zero");
        let none: [DfirOp; 0] = [];
        // A 2-byte send: the one width that would otherwise produce a mode.
        let short = Elements(2);

        for comp in [GenericComp::Lxlu, GenericComp::Lxsu] {
            assert!(
                matches!(
                    setldtype::<Dd2>(comp, Val(31), &none, short, width),
                    LdType::NonDefault { .. }
                ),
                "{comp:?} is an LX half"
            );
        }
        for comp in [
            GenericComp::Lx,
            GenericComp::L0lu,
            GenericComp::L0su,
            GenericComp::L3lu,
            GenericComp::L3su,
            GenericComp::Pt,
            GenericComp::Sfp,
            GenericComp::Hbm,
        ] {
            assert_eq!(
                setldtype::<Dd2>(comp, Val(31), &none, short, width),
                LdType::Default,
                "{comp:?} is not an LX half, so the early return fires before anything is read"
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

    /// 🎯 035/384 — THE FOUR BYPASS COMPONENTS AND NOTHING ELSE.
    ///
    /// ⭐ `areAnyOf(consumer_comps, {PT, SFP, L0SU, CROSSPTNLINK})` (`Helper.cpp:2765`). A consumer
    /// that is the SFP's own input path needs no explicit routing, and neither does an LX or an L3.
    #[test]
    fn only_the_four_bypass_consumers_get_a_set_send_dst() {
        let (to, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let emits = |unit: DfirUnit| {
            matches!(
                generate_set_send_destination_stmts::<Dd2>(
                    GenericComp::Lxlu,
                    &ConsumerUnits::Bound(unit),
                    to
                ),
                SetSendDestination::Emit(_)
            )
        };

        for row in 0..8u32 {
            let unit = DfirUnit::PtRow(Row::checked(row).expect("this arch has eight PT rows"));
            // ⭐ ALL EIGHT ROWS ARE ONE GENERIC COMPONENT — `senCompToGenericComp` sends every
            // `PTROW*` spelling to `PT` (`arch_enums.cpp:127-152`), so one arm covers the column.
            assert!(emits(unit), "{unit:?} bypasses the SFP");
        }
        assert!(emits(DfirUnit::Sfp));
        assert!(emits(DfirUnit::L0su));
        assert!(emits(DfirUnit::CrossPtnLink));

        for unit in [
            DfirUnit::L0lu,
            DfirUnit::Lx,
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::L3lu,
            DfirUnit::L3su,
            DfirUnit::Pe,
            DfirUnit::Hbm,
        ] {
            assert!(!emits(unit), "{unit:?} needs no explicit routing");
        }
    }

    /// 🎯 035/384 — THE GATE IS THE LOAD UNIT, SO AN LXSU PROGRAM GETS NONE OF THESE.
    ///
    /// ⛔ `if (comp != LXLU) return success();` (`Helper.cpp:2736`) — and `comp` is the GENERIC
    /// component of the program unit's first handle, so the store half of the very same LX is
    /// excluded. Folding the two halves onto one `GenericComp::Lx` would emit a set_send_dst into
    /// every LX store program.
    #[test]
    fn only_an_lxlu_program_emits_a_set_send_dst() {
        let (to, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let consumer =
            ConsumerUnits::Bound(DfirUnit::PtRow(Row::checked(0).expect("row 0 exists")));

        assert!(matches!(
            generate_set_send_destination_stmts::<Dd2>(GenericComp::Lxlu, &consumer, to),
            SetSendDestination::Emit(_)
        ));
        for on in [
            GenericComp::Lxsu,
            GenericComp::Lx,
            GenericComp::L0lu,
            GenericComp::L3lu,
            GenericComp::Pt,
        ] {
            assert_eq!(
                generate_set_send_destination_stmts::<Dd2>(on, &consumer, to),
                SetSendDestination::Nothing,
                "{on:?} is not the LX load unit"
            );
        }
    }

    /// 🎯 035/384 — A QUERIED CONSUMER IS A SET, AND ONE BYPASSING MEMBER IS ENOUGH.
    ///
    /// ⭐ THE `uniform::QueryMapOp` ARM PUSHES ONE COMPONENT PER QUERIED UNIT (`Helper.cpp:2748-2758`)
    /// and `areAnyOf` is an ANY, not an ALL: a map whose units are mostly harmless still needs the
    /// routing made explicit if one of them bypasses the SFP.
    #[test]
    fn one_bypassing_member_of_a_queried_set_is_enough() {
        let (to, _) = Link::<LxluUnit, CrossPtnLinkUnit>::between(LXLU, Val(99)).ends();

        assert!(matches!(
            generate_set_send_destination_stmts::<Dd2>(
                GenericComp::Lxlu,
                &ConsumerUnits::Queried(
                    vec![DfirUnit::L0lu, DfirUnit::Lx, DfirUnit::CrossPtnLink,]
                ),
                to
            ),
            SetSendDestination::Emit(_)
        ));
        assert_eq!(
            generate_set_send_destination_stmts::<Dd2>(
                GenericComp::Lxlu,
                &ConsumerUnits::Queried(vec![DfirUnit::L0lu, DfirUnit::Lx]),
                to
            ),
            SetSendDestination::Nothing
        );
        // An empty query names no component, which is `consumer_comps` staying empty.
        assert_eq!(
            generate_set_send_destination_stmts::<Dd2>(
                GenericComp::Lxlu,
                &ConsumerUnits::Queried(Vec::new()),
                to
            ),
            SetSendDestination::Nothing
        );
    }

    /// 🎯 035/384 — AND EVERY ARCH THIS CRATE BUILDS FOR SUPPORTS THE INSTRUCTION.
    ///
    /// ⛔⛔ THE ARCH TEST IS A COMPILE-TIME TRUTH, NOT A RUNTIME BRANCH. `IsaCoreGen` is ordered
    /// `MPW2 < MPW3 < MPW4 < RCUDD1A < SEN1P5` (`sys-arch-spec/isa/isa.hpp:24-35`) and [`IsaGen`]
    /// models only the last two, so `>= RCUDD1A_ISA` holds for both and
    /// [`SetSendDestination::NoSetDstMaskAtThisArchLevel`] is unreachable. This asserts that, so the
    /// day an older generation is added the claim in the doc fails with the code.
    #[test]
    fn no_arch_this_crate_targets_lacks_setdstmask() {
        for isa in [IsaGen::Rcudd1a, IsaGen::Sen1p5] {
            assert!(
                supports_set_dst_mask(isa),
                "{isa:?} is at or past RCUDD1A, so SETDSTMASK exists"
            );
        }

        let (to, _) = Link::<LxluUnit, SfpUnit>::between(LXLU, Val(3)).ends();
        // Both arch traits, through the const generic, on the one consumer that bypasses.
        assert!(matches!(
            generate_set_send_destination_stmts::<Dd2>(
                GenericComp::Lxlu,
                &ConsumerUnits::Bound(DfirUnit::Sfp),
                to
            ),
            SetSendDestination::Emit(_)
        ));
        assert!(matches!(
            generate_set_send_destination_stmts::<crate::arch::Sen1p5>(
                GenericComp::Lxlu,
                &ConsumerUnits::Bound(DfirUnit::Sfp),
                to
            ),
            SetSendDestination::Emit(_)
        ));
    }

    /// 🎯 036/384 — A LOAD FEEDING A STORE IS THE PATTERN; ANYTHING ELSE IS NOT.
    ///
    /// ⭐ THE CALLERS ASK THIS TO TELL A LOAD-AND-STORE PATTERN FROM A PLAIN LOAD
    /// (`Helper.cpp:3053,3067`), so the null return is the answer and not a failure.
    #[test]
    fn a_load_whose_single_user_is_a_store_of_that_class_is_the_pattern() {
        let load = DfirOp::Agen(agen::Op::VectorLoad {
            result: Val(31),
            view: VIEW,
            indices: indices(Val(30)),
            view_ty: view_ty(),
            ty: LANES,
        });
        let store = DfirOp::Agen(agen::Op::VectorStore {
            value: Val(31),
            view: Val(11),
            indices: indices(Val(30)),
            view_ty: view_ty(),
            ty: LANES,
        });
        let pattern = vec![load.clone(), store.clone()];

        assert_eq!(
            store_op_from_load_store_pattern(AgenOpKind::VectorStore, &load, &pattern),
            Some(&pattern[1])
        );

        // ⭐ THE CLASS IS THE TEMPLATE PARAMETER, AND ASKING FOR THE WRONG ONE FINDS NOTHING RATHER
        // THAN THE WRONG OP — `dyn_cast<SymbolicVectorStoreOp>` on a plain store is null.
        assert_eq!(
            store_op_from_load_store_pattern(AgenOpKind::SymbolicVectorStore, &load, &pattern),
            None
        );

        // A load that is sent rather than stored is not the pattern.
        let (to, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let sent = vec![
            load.clone(),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(31),
                ty: LANES,
            }),
        ];
        assert_eq!(
            store_op_from_load_store_pattern(AgenOpKind::VectorStore, &load, &sent),
            None
        );

        // ⛔ AND `hasOneUse()` AGAIN: a load the store reads AND a send reads is not the pattern.
        let both = vec![
            load.clone(),
            store,
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(31),
                ty: LANES,
            }),
        ];
        assert_eq!(
            store_op_from_load_store_pattern(AgenOpKind::VectorStore, &load, &both),
            None
        );

        // An op with no result has nothing to trace.
        assert_eq!(
            store_op_from_load_store_pattern(
                AgenOpKind::VectorStore,
                &DfirOp::Dataflow(dataflow::Op::Send {
                    to,
                    data: Val(31),
                    ty: LANES,
                }),
                &pattern
            ),
            None
        );
    }

    /// 🎯 037/384 — THE FIRST MARKED OP OF THAT CLASS IN PRE-ORDER, ACROSS THE WHOLE NEST.
    ///
    /// ⭐⭐ PRE-ORDER MEANS AN ENCLOSING OP IS NUMBERED BEFORE ITS BODY, and `interrupt()` means the
    /// walk stops at the first hit — so a second marked op of the same class is never visited. The
    /// golden's program has three `agen.vector_load`s at positions 3, 6 and 9 of the program unit's
    /// body; marking the last two must find the middle one.
    #[test]
    fn the_walk_finds_the_first_marked_op_of_that_class_in_pre_order() {
        let program = lx_to_sfp_bypass_1();

        // The pre-order positions of the three loads, discovered by the walk itself rather than
        // asserted from a count.
        let mut loads = Vec::new();
        let mut position = 0usize;
        walk_pre_order(&program, &mut position, &mut |position, op| {
            if agen_op_kind(op) == Some(AgenOpKind::VectorLoad) {
                loads.push(position);
            }
            None::<()>
        });
        assert_eq!(loads.len(), 3, "the golden holds three vector loads");

        // Marking the last two finds the middle one — the FIRST marked, not the last marked.
        let candidate = find_candidate_for_lowering(
            AgenOpKind::VectorLoad,
            &Marked::at([loads[1], loads[2]]),
            &program,
        )
        .expect("two of the three loads are marked");
        assert_eq!(candidate.position, loads[1]);
        assert!(matches!(
            candidate.op,
            DfirOp::Agen(agen::Op::VectorLoad { result, .. }) if *result == Val(33)
        ));

        // ⛔ THE CLASS FILTERS BEFORE THE MARK. A marked load is not a candidate store.
        assert_eq!(
            find_candidate_for_lowering(AgenOpKind::VectorStore, &Marked::at([loads[0]]), &program),
            None
        );

        // ⛔ AND `DT_CHECK(candidate_op)` — nothing marked is the abort, which here is `None`.
        assert_eq!(
            find_candidate_for_lowering(AgenOpKind::VectorLoad, &Marked::default(), &program),
            None
        );
    }

    /// 🎯 038/384 — THE CHAIN IS LISTED CONSUMER-FIRST, AND THE ORDER IS THE REFERENCE'S.
    ///
    /// ⛔⛔ THE SEND GOES ON THE LIST BEFORE THE SHUFFLE THAT FEEDS IT (`Helper.cpp:2978-2981`), so
    /// walking the list deletes the reader before the value it reads. Reversing the two would leave a
    /// deleted shuffle with a live use.
    #[test]
    fn the_rearrangements_user_is_listed_before_the_rearrangement() {
        let (to, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let program = vec![
            DfirOp::Agen(agen::Op::VectorLoad {
                result: Val(31),
                view: VIEW,
                indices: indices(Val(30)),
                view_ty: view_ty(),
                ty: LANES,
            }),
            DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(41),
                input: Val(31),
                indices: vec![0, 1],
                repetition: 64,
                input_ty: Vector {
                    len: 2,
                    elem: ElemType::Int(8),
                },
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(41),
                ty: LANES,
            }),
        ];

        let mut to_be_deleted = Vec::new();
        add_load_chain_to_delete_list(Val(31), &program, &mut to_be_deleted);
        assert_eq!(
            to_be_deleted,
            vec![&program[2], &program[1]],
            "the send is listed first, then the shuffle that feeds it"
        );

        // ⛔ IT APPENDS. Several callers build one list across a whole lowering.
        add_load_chain_to_delete_list(Val(31), &program, &mut to_be_deleted);
        assert_eq!(to_be_deleted.len(), 4);

        // A load sent directly contributes only the send.
        let direct = vec![
            program[0].clone(),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(31),
                ty: LANES,
            }),
        ];
        let mut only_the_send = Vec::new();
        add_load_chain_to_delete_list(Val(31), &direct, &mut only_the_send);
        assert_eq!(only_the_send, vec![&direct[1]]);

        // ⛔ AND A REARRANGEMENT NOTHING READS CONTRIBUTES ONLY ITSELF, where the reference
        // dereferences the end of an empty range.
        let dangling = vec![program[0].clone(), program[1].clone()];
        let mut listed = Vec::new();
        add_load_chain_to_delete_list(Val(31), &dangling, &mut listed);
        assert_eq!(listed, vec![&dangling[1]]);
    }
}
