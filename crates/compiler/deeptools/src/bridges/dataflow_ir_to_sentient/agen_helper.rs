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

use super::agen_access_details::{
    AccessContainer, AccessDetailsAffine, AccessDetailsAffineComposite, AccessDetailsSymbolic,
    IndicesCoeffDict, MemoryOperandIndex,
};
use crate::arch::{Arch, Bytes, Elements, IsaGen};
use crate::islands::dataflow_ir::dialects::{
    self as dfir_op, Index, Op as DfirOp, Val, arith, dataflow, defining_op, results, uses,
};
use crate::islands::dataflow_ir::link::SendEnd;
use crate::islands::dataflow_ir::ty::{
    AffineMap, FlatConstraints, GenericComp, IntegerSet, MemRef, Vector,
};
use crate::islands::dataflow_ir::{ProgramUnit, ValueMapping, Values};
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
    /// `scf.for` — [`crate::islands::dataflow_ir::dialects::scf::Op::For`].
    ///
    /// ⭐ AN **INPUT** LOOP, WHICH IS WHY IT IS HERE. Nothing this bridge emits builds one; the pass
    /// input contains them (`scf_loop_with_result.mlir:32`) and
    /// [`super::tf_transform_loop_to_legalize_for_sentient_lowering::transform_scf_to_affine_loop`]
    /// is what turns the legal ones into `affine.for`. Until it has run, a nest can hold both kinds,
    /// which is exactly when the counting rule above matters.
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
/// defining op (`Helper.cpp:392-393`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewedUnit {
    /// It is a `dataflow.get_unit` of this kind.
    GetUnit(DfirUnit),
    /// Its defining op is something else — the `dyn_cast_or_null` returned null.
    NotAGetUnit,
}

/// WHAT AN INDIRECT MEMORY VIEW'S `start_address` OPERAND RESOLVED TO — `Helper.cpp:409-422`.
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

    /// The diagnostic the C++ emits for this outcome, verbatim (`Helper.cpp:395-426`).
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
/// being flat, unpermuted and based at 0 is what makes the extracted scalar usable as an address with
/// no scaling and no offset of its own — `e328_adjustMutableAddrInitForIndirect` adds it straight into
/// the address chain with one `sentient.scalar_add`. WHERE that add goes is not one place: to the
/// mutable address itself (`Helper.cpp:1496`), or to a loop's iter_arg INITIALISER when the extract
/// sits outside that loop (`:1522`, `:1533`), or after whichever of the two ops comes last when the
/// address is not an iter_arg (`:1569`).
#[must_use]
pub fn check_indirect_mem_view_for_extract_op(view: &IndirectMemView<'_>) -> IndirectMemViewCheck {
    // `:392-398` — the from-unit must be a get_unit before its type can be read.
    let unit = match view.from_unit {
        ViewedUnit::NotAGetUnit => return IndirectMemViewCheck::FromUnitIsNotAGetUnit,
        ViewedUnit::GetUnit(unit) => unit,
    };
    // `:399-407` — and the unit it names must be the virtual IBR. The C++ compares
    // `stringToSenComponents.find(getType().str())->second` against `SenComponents::LXVIRTUALIBR`;
    // here the `type=` string and the enum are the same value (`DfirUnit::spelling`).
    if unit != DfirUnit::LxVirtualIbr {
        return IndirectMemViewCheck::NotOnAVirtualIbr;
    }
    // `:409-422` — a constant start address, and it must be zero. TWO checks, not one: the C++
    // reports "does not have a constant start address" and "start address is not 0" separately.
    match view.start_address {
        ViewStart::NotAConstant => return IndirectMemViewCheck::StartAddressIsNotConstant,
        ViewStart::Constant(start) if start != 0 => {
            return IndirectMemViewCheck::StartAddressIsNotZero;
        }
        ViewStart::Constant(_) => {}
    }
    // `:424-428` — `getNumDims() != 1 || !isIdentity()`.
    if view.layout_map.dims != 1 || !view.layout_map.is_identity() {
        return IndirectMemViewCheck::LayoutMapIsNot1DIdentity;
    }
    // `:430` — `return true;`
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

/// WHAT AN EXTRACT STATEMENT BINDS — **the two kinds do not bind the same number of values.**
///
/// ⛔⛔ TWO RESULTS ON THE LOAD SIDE, ONE ON THE STORE SIDE. `Sentient_LoadAndExtractScalarOp` is
/// `let results = (outs Index:$addr, Index:$data);` (`SentientOps.td:622`, with `getAddrResult()` =
/// result 0 and `getDataResult()` = result 1); `Sentient_ReceiveAndExtractScalarOp` is
/// `let results = (outs Index:$result);` (`:684`) — one value, and it is the DATUM. A shape with a
/// mandatory address field makes a receive's result 0 an address, which is exactly the value the
/// indirect access wants to ADD to an address.
///
/// ⭐ AND THE REFERENCE READS THEM BY DIFFERENT INDICES. `e328_adjustMutableAddrInitForIndirect` picks
/// `load_and_extract_op.getDataResult()` for the load and `extract_op->getResult(0)` for the receive
/// (`Helper.cpp:1467-1470`) — result 1 in one case, result 0 in the other. [`Self::data`] is that
/// choice, made once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractScalarResults {
    /// `sentient.load_and_extract_scalar` — `addr` then `data`
    /// (`%21, %22 = sentient.load_and_extract_scalar ..`,
    /// `lx_indirect_loads_stores_composite.mlir:34`).
    LoadAndExtractScalar {
        /// Result 0 — the address it advanced (`mutable_addr + increment`).
        addr: Val,
        /// Result 1 — the extracted scalar.
        data: Val,
    },
    /// `sentient.receive_and_extract_scalar` — one result, the extracted scalar. There is no address
    /// result to name.
    ReceiveAndExtractScalar {
        /// The sole result.
        data: Val,
    },
}

impl ExtractScalarResults {
    /// Which of the two statements bound these.
    #[must_use]
    pub const fn kind(self) -> ExtractScalarKind {
        match self {
            Self::LoadAndExtractScalar { .. } => ExtractScalarKind::LoadAndExtractScalar,
            Self::ReceiveAndExtractScalar { .. } => ExtractScalarKind::ReceiveAndExtractScalar,
        }
    }

    /// THE EXTRACTED SCALAR — result 1 of a load, the sole result of a receive
    /// (`Helper.cpp:1467-1470`).
    #[must_use]
    pub const fn data(self) -> Val {
        match self {
            Self::LoadAndExtractScalar { data, .. } | Self::ReceiveAndExtractScalar { data } => {
                data
            }
        }
    }
}

/// ONE EXTRACT STATEMENT, AS THE INDIRECT ACCESS THAT USES IT NEEDS TO SEE IT.
///
/// ⭐ IT IS THE OP'S RESULTS THAT MATTER, AND [`ExtractScalarResults`] IS HOW MANY THERE ARE. What the
/// indirect access does with the op it finds is take the extracted scalar and add it to an address —
/// `e328_adjustMutableAddrInitForIndirect` at `Helper.cpp:1496`, `:1522`, `:1533` or `:1569`
/// (`%34 = sentient.scalar_add %33, %22`, `lx_indirect_loads_stores_composite.mlir:43`). Carrying the
/// results is what makes the found op usable without a second lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractScalarOp {
    /// Its `extract_idx`.
    pub index: ExtractIndex,
    /// What it bound — and which of the two statements it is, since only one of them has an address
    /// result.
    pub results: ExtractScalarResults,
}

impl ExtractScalarOp {
    /// Which of the two statements it is.
    #[must_use]
    pub const fn kind(self) -> ExtractScalarKind {
        self.results.kind()
    }

    /// The extracted scalar — the value the indirect access adds to its address.
    #[must_use]
    pub const fn data(self) -> Val {
        self.results.data()
    }
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
    pub fn mint(&mut self, results: ExtractScalarResults) -> ExtractScalarOp {
        let op = ExtractScalarOp {
            index: ExtractIndex(u8::try_from(self.minted.len()).unwrap_or(u8::MAX)),
            results,
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
            .find(|op| op.kind() == of && op.index == index)
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// THE AGEN OP CLASSES — the reference's template parameters, as values.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH `agen` OP CLASS — what `findCandidateForLowering<OpTy>` and
/// `getStoreOpFromLoadStorePattern<VectorStoreTy>` are instantiated with.
///
/// ⭐⭐ THE TEMPLATE PARAMETER IS AN INPUT, SO IT HAS TO BE NAMABLE. `findCandidateForLowering` is
/// instantiated at TWELVE sites with **ten** different classes (`Helper.cpp:3013, 3038, 3063, 3086,
/// 3120, 3141, 3162, 3184, 3232, 3282, 3328, 3373` — `VectorLoadOp` at `:3013` and `:3063`,
/// `VectorStoreOp` at `:3038` and `:3086`) and `getStoreOpFromLoadStorePattern` with two
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
            dfir_op::agen::Op::SymbolicVectorLoad { .. } => Some(AgenOpKind::SymbolicVectorLoad),
            dfir_op::agen::Op::SymbolicVectorStore { .. } => Some(AgenOpKind::SymbolicVectorStore),
            dfir_op::agen::Op::CompositeLoadAndStore(_) => Some(AgenOpKind::CompositeLoadAndStore),
            // The region terminator is not a transfer, and neither the interleave nor the mask
            // state is one of the twelve classes `fuseLoadOrStoreChainOps`'s candidate walk looks
            // for (`AgenToSentient.cpp:33-37`) — `e384_runOnOperation` finds those two itself.
            dfir_op::agen::Op::Yield
            | dfir_op::agen::Op::CompositeMemoryInterleave { .. }
            | dfir_op::agen::Op::SetTransferMaskState { .. } => None,
        },
        DfirOp::Arith(_)
        | DfirOp::Scf(_)
        | DfirOp::Affine(_)
        | DfirOp::Dataflow(_)
        // ⭐ A `vector.load` IS NOT AN `agen` OP. Upstream's plain access is a different dialect's
        // operation however much it reads like `agen.vector_load`, and every `isa<>` in
        // `AgenToSentient` names the `agen` classes only.
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        // ⭐ AND `uniform` WITH THEM: none of its four ops is in the `agen` dialect, so
        // `dyn_cast<agen::…>` is null for every one of them.
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
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
    /// ⛔ TWO OF THE FIVE HAVE NO ISLAND OP YET (the indirect family), and
    /// `agen.composite_load_and_store` is **not** `CompositeLoadOp` — the reference's `isa<>` list
    /// does not include it, so it answers `None` here too rather than borrowing the composite arm.
    #[must_use]
    pub fn of(op: &DfirOp) -> Option<AgenLoad> {
        match op {
            DfirOp::Agen(dfir_op::agen::Op::VectorLoad { result, .. }) => {
                Some(AgenLoad::Vector { result: *result })
            }
            // ⭐ AND THE SYMBOLIC LOAD ROOTS AT ITS RESULT LIKE THE OTHER TWO VECTOR LOADS — one
            // `isa<>` arm, one root (`Helper.cpp:1245-1247`).
            DfirOp::Agen(dfir_op::agen::Op::SymbolicVectorLoad { result, .. }) => {
                Some(AgenLoad::SymbolicVector { result: *result })
            }
            DfirOp::Agen(
                dfir_op::agen::Op::VectorStore { .. }
                | dfir_op::agen::Op::SymbolicVectorStore { .. }
                | dfir_op::agen::Op::CompositeLoadAndStore(_)
                | dfir_op::agen::Op::CompositeMemoryInterleave { .. }
                | dfir_op::agen::Op::SetTransferMaskState { .. }
                | dfir_op::agen::Op::Yield,
            )
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            // ⛔ AND `vector.load` IS NOT ONE OF THE FIVE EITHER. `getLoadConsumer`'s roots are
            // `isa<VectorLoadOp, IndirectVectorLoadOp, SymbolicVectorLoadOp>` plus the two composite
            // loads (`Conversion/AgenToSentient/Helper.cpp:1245-1250`) — all `agen`, no upstream
            // `vector::LoadOp` anywhere in the chain.
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ AND `uniform`: no `uniform.` op is one of the five load classes.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
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
                // ⛔ AND NEITHER IS A NEGATION, THOUGH THE REFERENCE DOES PAIR IT WITH A SELECT
                // ELSEWHERE: `isa<…, vectorchain::NegOp, vectorchain::SelectOp>(user)` is a DIFFERENT
                // test in a DIFFERENT function (`VectorChainToSentientPT.cpp:302`,
                // `VectorOperands.cpp:706` and `:761`). `getLoadConsumer`'s three are Select, Shuffle
                // and Rotate and it stops there.
                | dfir_op::vectorchain::Op::Neg { .. }
                | dfir_op::vectorchain::Op::Pack { .. }
                | dfir_op::vectorchain::Op::Merge { .. }
                | dfir_op::vectorchain::Op::CreateAffineMask { .. }
                | dfir_op::vectorchain::Op::CreateAffineMaskSet { .. } => None,
            },
            DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            // ⭐ NOT A REARRANGEMENT: the three are `vectorchain` ops.
            | DfirOp::Vector(_)
            // ⭐ AND NOR IS `uniform`: `isa<SelectOp, ShuffleOp, RotateOp>` names three
            // `vectorchain` ops and nothing else.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
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
/// may have changed due to loop cloning" (`Helper.cpp:2962-2963`) — the mark is how an op survives its
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

    /// `op->setAttr("marked", builder.getI8IntegerAttr(1))` — the mark
    /// `gatherAffineLoadStoreDetails` leaves so the op can be found again (`Helper.cpp:545-547`).
    ///
    /// ⛔ THE VALUE IS NEVER READ, only the attribute's presence, which is why this takes none.
    pub fn mark(&mut self, position: usize) {
        self.0.insert(position);
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
    use crate::bridges::dataflow_ir_to_sentient::agen_access_details::MemoryOperandIndex;
    use crate::generated::SyncSignal;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::dataflow_ir::dialects::{
        Index, affine, agen, arith, dataflow, uniform, vectorchain,
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
            syms: 0,
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
            dbg_name: None,
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
                num_folds: None,
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
                    dbg_name: None,
                    carried: Vec::new(),
                    body: vec![DfirOp::Affine(affine::Op::For {
                        iv: Val(21),
                        lo: affine::Bound::Const(0),
                        hi: affine::Bound::Val(Val(0)),
                        // The golden's loops are plain counted ones: they carry nothing.
                        dbg_name: None,
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
            num_folds: None,
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
                num_folds: None,
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

    /// ⭐ THE PAIRING ROUND-TRIPS: the index a mint stamps is the one a lookup finds, and the datum it
    /// carries is result 1 of the load — `%21, %22 = sentient.load_and_extract_scalar`
    /// (`lx_indirect_loads_stores_composite.mlir:34`), consumed as `%22` at `:43`.
    #[test]
    fn a_minted_extract_statement_is_found_by_its_index() {
        let mut ops = ExtractScalarOps::default();
        let first = ops.mint(ExtractScalarResults::LoadAndExtractScalar {
            addr: Val(21),
            data: Val(22),
        });
        assert_eq!(first.index.get(), 0, "the counter opens at 0 per unit");

        let found = ops.find(ExtractScalarKind::LoadAndExtractScalar, first.index);
        assert_eq!(found, Some(first));
        assert_eq!(found.map(ExtractScalarOp::data), Some(Val(22)));
    }

    /// ⭐ THE COUNTER IS SHARED BETWEEN THE TWO KINDS, so the second statement is index 1 whichever
    /// side it is on — and a lookup for the wrong kind finds nothing.
    #[test]
    fn the_counter_is_shared_and_the_kind_is_checked() {
        let mut ops = ExtractScalarOps::default();
        let load = ops.mint(ExtractScalarResults::LoadAndExtractScalar {
            addr: Val(21),
            data: Val(22),
        });
        let store = ops.mint(ExtractScalarResults::ReceiveAndExtractScalar { data: Val(31) });
        assert_eq!(store.index.get(), 1);
        assert_eq!(
            store.data(),
            Val(31),
            "a receive binds ONE result and it is the datum (`SentientOps.td:684`)"
        );

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
        let elsewhere = other.mint(ExtractScalarResults::LoadAndExtractScalar {
            addr: Val(1),
            data: Val(2),
        });
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

    /// ⭐ THE TWO COMPOSITE FAMILIES: a load region sends before it yields, a store region yields what
    /// its receive bound.
    #[test]
    fn a_composite_region_is_checked_per_family() {
        let (to, _) = Link::<LxluUnit, PtRowUnit<0>>::between(LXLU, PT).ends();
        let (_, from) = Link::<PtRowUnit<0>, LxluUnit>::between(PT, LXLU).ends();
        let load_body = vec![
            DfirOp::Agen(agen::Op::VectorLoad {
                result: Val(31),
                view: VIEW,
                indices: vec![Index::Const(0)],
                view_ty: stick_view(),
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(31),
                ty: LANES,
            }),
            DfirOp::Agen(agen::Op::Yield),
        ];
        assert!(check_composite_region(CompositeFamily::Load, &load_body, &[]).admissible());

        // ⛔ THE SEND MUST BE THE STATEMENT BEFORE THE YIELD — a two-statement region whose front is
        // the load has none.
        assert_eq!(
            check_composite_region(CompositeFamily::Load, &load_body[..2], &[]),
            CompositeRegionCheck::LoadSendDoesNotPrecedeYield
        );

        let store_body = vec![
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(41),
                from,
                ty: LANES,
            }),
            DfirOp::Agen(agen::Op::Yield),
        ];
        assert!(
            check_composite_region(CompositeFamily::Store, &store_body, &[Val(41)]).admissible()
        );

        // ⛔ AND A STORE REGION THAT YIELDS SOMETHING ELSE leaves the received value unread.
        assert_eq!(
            check_composite_region(CompositeFamily::Store, &store_body, &[]),
            CompositeRegionCheck::StoreYieldsIncorrectValue
        );
    }

    /// ⭐ ONE INDEX, RESOLVING TO ZERO, OVER A ONE-DIMENSIONAL VIEW — and each refusal names itself.
    #[test]
    fn the_extract_store_wants_one_zero_index_over_one_dimension() {
        let one_d = MemRef {
            shape: vec![32],
            elem: ElemType::Int(64),
        };
        let scope = [DfirOp::Arith(arith::Op::Constant {
            result: Val(3),
            value: 0,
        })];

        assert!(
            check_store_op_from_extract_pattern(&[Index::Val(Val(3))], &one_d, &scope).admissible()
        );
        assert!(
            check_store_op_from_extract_pattern(&[Index::Const(0)], &one_d, &scope).admissible()
        );

        assert_eq!(
            check_store_op_from_extract_pattern(
                &[Index::Const(0), Index::Const(0)],
                &one_d,
                &scope
            ),
            StoreFromExtractCheck::ExpectingIndicesSizeOne
        );
        assert_eq!(
            check_store_op_from_extract_pattern(&[Index::Const(1)], &one_d, &scope),
            StoreFromExtractCheck::ExpectingStartOffsetZero
        );
        // ⛔ AND A LOOP-VARIABLE INDEX, which is where the reference dereferences null.
        assert_eq!(
            check_store_op_from_extract_pattern(&[Index::Val(Val(9))], &one_d, &scope),
            StoreFromExtractCheck::ExpectingStartOffsetZero
        );
        assert_eq!(
            check_store_op_from_extract_pattern(
                &[Index::Const(0)],
                &MemRef {
                    shape: vec![2, 32],
                    elem: ElemType::Int(64)
                },
                &scope
            ),
            StoreFromExtractCheck::ExpectingOneDimStoreSet
        );
    }

    /// `vector<128xi8>`'s view — one stick of `i8`.
    fn stick_view() -> MemRef {
        MemRef {
            shape: vec![128],
            elem: ElemType::Int(8),
        }
    }

    /// ⭐ THE GATHER INDEX'S PATH: an `agen.vector_load` off an LX view whose one consumer stores into
    /// a virtual-IBR view (`lx_indirect_loads_stores_composite.mlir:30,36`).
    fn extract_scalar_scope() -> Vec<DfirOp> {
        let ty = MemRef {
            shape: vec![32],
            elem: ElemType::Int(64),
        };
        vec![
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(11),
                residency: at_corelet_zero(),
                unit: DfirUnit::Lx,
                num_folds: None,
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(21),
                from: Val(11),
                start: Val(3),
                layout: identity_1d(),
                ty: ty.clone(),
            }),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(13),
                residency: Residency::Global,
                unit: DfirUnit::LxVirtualIbr,
                num_folds: None,
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(23),
                from: Val(13),
                start: Val(3),
                layout: identity_1d(),
                ty: ty.clone(),
            }),
            DfirOp::Agen(agen::Op::VectorLoad {
                result: Val(31),
                view: Val(21),
                indices: vec![Index::Const(0)],
                view_ty: ty.clone(),
                ty: LANES,
            }),
            DfirOp::Agen(agen::Op::VectorStore {
                value: Val(31),
                view: Val(23),
                indices: vec![Index::Const(0)],
                view_ty: ty,
                ty: LANES,
            }),
        ]
    }

    /// ⛔ THE ORDER IS THE PATTERN: LX for the load, virtual IBR for the store.
    #[test]
    fn a_load_of_the_lx_stored_into_the_virtual_ibr_is_the_extract_pattern() {
        let scope = extract_scalar_scope();
        assert!(is_load_and_extract_scalar_pattern(&scope[4], &scope));

        // The store is not a load, and neither is a load whose view is the IBR's.
        assert!(!is_load_and_extract_scalar_pattern(&scope[5], &scope));
        let mut from_the_ibr = scope.clone();
        from_the_ibr[4] = DfirOp::Agen(agen::Op::VectorLoad {
            result: Val(31),
            view: Val(23),
            indices: vec![Index::Const(0)],
            view_ty: stick_view(),
            ty: LANES,
        });
        assert!(!is_load_and_extract_scalar_pattern(
            &from_the_ibr[4],
            &from_the_ibr
        ));
    }

    /// ⛔ THE STORE SIDE ASKS ONE QUESTION — is the view cut from the virtual IBR?
    #[test]
    fn a_store_into_the_virtual_ibr_is_the_receive_and_extract_pattern() {
        let scope = extract_scalar_scope();
        assert!(is_receive_and_extract_scalar_pattern(&scope[5], &scope));

        let mut into_the_lx = scope.clone();
        into_the_lx[5] = DfirOp::Agen(agen::Op::VectorStore {
            value: Val(31),
            view: Val(21),
            indices: vec![Index::Const(0)],
            view_ty: stick_view(),
            ty: LANES,
        });
        assert!(!is_receive_and_extract_scalar_pattern(
            &into_the_lx[5],
            &into_the_lx
        ));
    }

    /// ⭐ EVERY HANDLE A SYMBOLIC ACCESS HOLDS IS READ THROUGH THE CLONE'S MAPPING — the five values
    /// AND the operation.
    #[test]
    fn a_symbolic_access_follows_the_clone() {
        let original = agen::Op::VectorLoad {
            result: Val(31),
            view: Val(21),
            indices: vec![Index::Val(Val(5))],
            view_ty: stick_view(),
            ty: LANES,
        };
        let cloned = agen::Op::VectorLoad {
            result: Val(131),
            view: Val(121),
            indices: vec![Index::Val(Val(105))],
            view_ty: stick_view(),
            ty: LANES,
        };

        let mut details = AccessDetailsSymbolic::new(&original, DfirUnit::Lxlu);
        details.base.set_indices(&[Val(5)]);
        details.set_strides(&[Val(6)]);
        details.base.mem_ref = Some(Val(7));
        details.base.set_mem_view_start_addr(Val(8));
        details.base.memory = Some(Val(9));

        let mut container = AccessContainer::default();
        container
            .vacancy(MemoryOperandIndex::DirSrc)
            .expect("a fresh container has every slot empty")
            .fill(details);

        let mut ir_map = ValueMapping::new();
        for (from, to) in [
            (Val(5), Val(105)),
            (Val(6), Val(106)),
            (Val(7), Val(107)),
            (Val(8), Val(108)),
            (Val(9), Val(109)),
        ] {
            ir_map.map(from, to);
        }
        let mut op_map = OpMapping::new();
        op_map.map(&original, &cloned);

        update_symbolic_access_details(&mut container, &ir_map, &op_map);

        let updated = &container.entries()[0];
        assert!(
            core::ptr::eq(updated.base.op, &cloned),
            "the operation is remapped too"
        );
        assert_eq!(updated.base.indices, vec![Val(105)]);
        assert_eq!(updated.strides(), &[Val(106)]);
        assert_eq!(updated.base.mem_ref, Some(Val(107)));
        assert_eq!(updated.base.mem_view_start_addr, Some(Val(108)));
        assert_eq!(updated.base.memory, Some(Val(109)));
    }

    /// ⭐ THE TWO PRODUCERS A STORE MAY HAVE: a receive off a `get_unit`, and a splat of a one-value
    /// bitstream.
    #[test]
    fn a_store_producer_is_the_receive_or_the_splat_behind_it() {
        let (_, from) = Link::<PtRowUnit<0>, LxluUnit>::between(PT, LXLU).ends();
        let received = vec![
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: PT,
                residency: at_corelet_zero(),
                unit: DfirUnit::PtRow(Row::checked(0).expect("row 0 exists")),
                num_folds: None,
            }),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(41),
                from,
                ty: LANES,
            }),
        ];
        assert_eq!(
            get_store_producer(&AgenStore::Vector { value: Val(41) }, &received),
            StoreProducer::Found {
                inp_op: &received[1],
                producer: &received[0],
            }
        );

        let splat = vec![
            DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
                result: Val(51),
                value: vec![0],
                ty: LANES,
                is_symbol: false,
            }),
            DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(52),
                input: Val(51),
                indices: vec![0],
                repetition: 128,
                input_ty: LANES,
                ty: LANES,
            }),
        ];
        assert_eq!(
            get_store_producer(&AgenStore::Vector { value: Val(52) }, &splat),
            StoreProducer::Found {
                inp_op: &splat[1],
                producer: &splat[0],
            }
        );
        // ⛔ AND THE COMPOSITE'S REGION IS READ THE SAME WAY — a bitstream in front of the shuffle.
        assert_eq!(
            get_store_producer(
                &AgenStore::Composite {
                    input_vector: None,
                    body: &splat,
                },
                &splat
            ),
            StoreProducer::Found {
                inp_op: &splat[1],
                producer: &splat[0],
            }
        );

        // ⛔ ONE VALUE ONLY, and a stored value from neither op is not a producer at all.
        let mut two_values = splat.clone();
        two_values[0] = DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
            result: Val(51),
            value: vec![0, 1],
            ty: LANES,
            is_symbol: false,
        });
        assert_eq!(
            get_store_producer(&AgenStore::Vector { value: Val(52) }, &two_values),
            StoreProducer::BitstreamNotOneValue
        );
        assert_eq!(
            get_store_producer(&AgenStore::Vector { value: Val(51) }, &splat),
            StoreProducer::StoreValueNotReceiveOrShuffle
        );
    }

    /// ⭐⭐ THE VENDOR'S OWN ANSWER KEY — all five `slice_mask_map`s of
    /// `dcc/test/Conversion/AgenToSentient/set_transfer_mask_state.mlir` with the `numvalidentry`,
    /// `sliceid_xsl`, `xslinner` and `wsllen` its `CHECK-SENT-IR` lines require, over `vector<128xi8>`
    /// at eight slices.
    #[test]
    fn the_vendors_five_mask_maps_encode_as_they_check() {
        let of = |map: SliceMaskMap| SetTransferMaskState {
            mask_value: Val(5),
            num_slices: NonZeroU32::new(8).expect("eight slices"),
            slice_mask_map: map,
            ty: LANES,
            dbg_name: None,
        };
        let generic = |slice: u32, wsl: (u64, u64), xsl: (u64, u64)| {
            of(SliceMaskMap::Generic {
                slice_id_xsl: sen::SliceId(slice),
                wsl: MaskRun {
                    unmasked: Elements(wsl.0),
                    masked: Elements(wsl.1),
                },
                xsl: MaskRun {
                    unmasked: Elements(xsl.0),
                    masked: Elements(xsl.1),
                },
            })
        };
        // `(numvalidentry, sliceid_xsl, xslinner, wsllen, maskall, precision)`.
        let fields = |mask: &SetTransferMaskState| match construct_set_active_mask_value_op::<Dd2>(
            DfirUnit::Lxlu,
            mask,
        ) {
            ActiveMaskValue::Samv(SenOp::Sentient(sen::Op::Samv {
                num_valid_entry,
                slice_id_xsl,
                xsl_inner,
                wsl_len,
                mask_all,
                precision,
                ..
            })) => Some((
                num_valid_entry.0,
                slice_id_xsl.0,
                xsl_inner,
                wsl_len.0,
                mask_all,
                precision.0,
            )),
            _ => None,
        };

        // `(A)(A)(A)(A)(A)(A|B)(1)(1)`, maskA (8, 8) and maskB (1, 1).
        assert_eq!(
            fields(&generic(5, (8, 8), (1, 1))),
            Some((12, 5, true, 8, false, 8))
        );
        // `(A)(A)(A)(A)(A)(A)(A)(A|B)`, maskA (4, 12) and maskB (2, 2).
        assert_eq!(
            fields(&generic(7, (4, 12), (2, 2))),
            Some((9, 7, true, 4, false, 8))
        );
        // `(A|B)(1)(1)(1)(1)(1)(1)(1)`, maskA (3, 1) and maskB (12, 4) — the outer mask is the XSL's,
        // so `xslinner` is false and `wsllen` is the inner length.
        assert_eq!(
            fields(&generic(0, (3, 1), (12, 4))),
            Some((15, 0, false, 4, false, 8))
        );
        // `(0)(0)(0)(0)(0)(0)(0)(0)` and `(1)(1)(1)(1)(1)(1)(1)(1)`.
        assert_eq!(
            fields(&of(SliceMaskMap::Unmask)),
            Some((0, 7, false, 1, false, 8))
        );
        assert_eq!(
            fields(&of(SliceMaskMap::FullMask)),
            Some((0, 0, false, 1, true, 8))
        );

        // ⛔ AND SAMV IS AN LXLU OP.
        assert_eq!(
            construct_set_active_mask_value_op::<Dd2>(DfirUnit::Lxsu, &of(SliceMaskMap::FullMask)),
            ActiveMaskValue::NotInLxlu
        );
    }

    /// ⭐ THE FIRST REGION BUILDS THE OP AND HOISTS THE IN-LOOP UNIT; THE LAST REGION FINDS IT AND
    /// CLEARS THE FLAG.
    #[test]
    fn the_first_region_builds_the_op_and_the_last_clears_the_flag() {
        let in_loop = DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(92),
            residency: at_corelet_zero(),
            unit: DfirUnit::Lxlu,
            num_folds: None,
        });
        let source = UniformizeSource {
            kind: RegionOpKind::UniformizeRegions,
            regions: vec![
                vec![
                    UnitOperand::Outside(Val(90)),
                    UnitOperand::InsideLoop {
                        unit: Val(92),
                        def: &in_loop,
                    },
                ],
                vec![UnitOperand::Outside(Val(91))],
            ],
        };

        let mut values = Values::default();
        let yield_of = |arg: Val| {
            vec![DfirOp::Uniform(uniform::Op::Yield {
                operands: vec![arg],
            })]
        };
        // The clone is minted first, then the op's one `index` result, then each region's argument.
        let expected = Uniformized {
            op: DfirOp::Uniform(uniform::Op::UniformizeRegions {
                regions: vec![
                    uniform::LocalRegion {
                        arg: Val(2),
                        units: vec![Val(90), Val(0)],
                        body: yield_of(Val(2)),
                    },
                    uniform::LocalRegion {
                        arg: Val(3),
                        units: vec![Val(91)],
                        body: yield_of(Val(3)),
                    },
                ],
                results: vec![Val(1)],
            }),
            hoisted: vec![DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(0),
                residency: at_corelet_zero(),
                unit: DfirUnit::Lxlu,
                num_folds: None,
            })],
            active: true,
        };
        assert_eq!(
            create_uniformize_regions_op(&mut values, &source, 0, &mut []),
            CreatedUniformize::Created(Box::new(expected.clone()))
        );

        // The second region is also the last, so it finds the op and removes the attribute.
        let mut preceding = vec![expected];
        assert_eq!(
            create_uniformize_regions_op(&mut values, &source, 1, &mut preceding),
            CreatedUniformize::Existing(0)
        );
        assert!(!preceding[0].active, "the last region clears it");

        // ⛔ NOTHING ACTIVE BEFORE THE LOOP is where the reference walks off the front of the block.
        assert_eq!(
            create_uniformize_regions_op(&mut values, &source, 1, &mut preceding),
            CreatedUniformize::NoActiveOp
        );
    }

    // ─────────────────────────────── 210/384 ───────────────────────────────

    /// The vendor's own unsplit 64-lane composite (`lx_indirect_loads_stores_composite.mlir`), whose
    /// sets and orders the transfer planner derives.
    fn composite_transfer() -> agen::Op {
        use crate::bridges::subtile_to_dataflow_ir::transfer::{Lanes, plan};

        let planned = plan(&[1, 1, 64], &[1, 1, 1, 1, 64], 64, Lanes::F16)
            .expect("one 64-lane vector is the unsplit case");
        agen::Op::CompositeLoadAndStore(Box::new(agen::CompositeTransfer {
            src: Val(21),
            src_indices: vec![Index::Val(Val(1)), Index::Val(Val(4)), Index::Const(0)],
            src_ty: MemRef {
                shape: vec![12, 64, 64],
                elem: ElemType::F16,
            },
            dst: Val(27),
            dst_indices: vec![Index::Const(0); 5],
            dst_ty: MemRef {
                shape: vec![2, 2, 1, 1, 64],
                elem: ElemType::F16,
            },
            load_iv: Val(9),
            load_iv_ty: Vector {
                len: planned.vector_lanes,
                elem: ElemType::F16,
            },
            load_set: planned.load_set,
            load_order: planned.load_order,
            store_set: planned.store_set,
            store_order: planned.store_order,
            time_set: planned.time_set,
            time_order: planned.time_order,
            load_time_addr_map: planned.load_time_addr_map,
            store_time_addr_map: planned.store_time_addr_map,
            body: vec![DfirOp::Agen(agen::Op::Yield)],
        }))
    }

    /// ⛔ A COMPOSITE IS ASKED ABOUT ITS TIME ORDER TOO, and a constant result names no dimension —
    /// `inversePermutation` returns null and the transfer is refused.
    #[test]
    fn the_composites_orders_are_permutations_and_a_constant_time_order_is_not() {
        let op = DfirOp::Agen(composite_transfer());
        assert_eq!(
            check_basic_conditions(CheckedOp::Dfir(&op)),
            BasicConditions::Admissible
        );

        let mut bent = composite_transfer();
        if let agen::Op::CompositeLoadAndStore(transfer) = &mut bent {
            transfer.time_order.results = vec![AffineExpr::Const(0)];
        }
        let bent = DfirOp::Agen(bent);
        assert_eq!(
            check_basic_conditions(CheckedOp::Dfir(&bent)),
            BasicConditions::TimeOrderNotAPermutation
        );

        // `:137-141` — an already-lowered store answers on the mark alone.
        assert_eq!(
            check_basic_conditions(CheckedOp::ReceiveAndStore { marked: false }),
            BasicConditions::NotMarked
        );
    }

    // ─────────────────────────────── 211/384 ───────────────────────────────

    /// One `sentient.load_and_send` of `extent`, which is all the interleave check reads off it.
    fn sent_load_and_send(extent: sen::Extent) -> SenOp {
        SenOp::Sentient(sen::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: Link::<LxluUnit, L0suUnit>::between(Val(4), Val(5)).ends().0,
            result: Val(6),
            extent,
            interleaved_group: 0,
            rotate_val: None,
            dir: None,
            shuffle_mode: sen::ShuffleMode::NoShuffle,
            reg: sen::Reg {
                locale: sen::RegType::Unknown,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// ⛔ THE REGION MUST AGREE ON THE BURST, so two sends of the same extent pass and one raised
    /// burst refuses the whole interleave.
    #[test]
    fn an_interleaved_region_agrees_on_the_burst_or_is_refused() {
        let mut extent = sen::Extent::of(Elements(64), Bytes(2));
        extent.burst_size = Elements(8);
        let region = [
            sent_load_and_send(extent),
            sent_load_and_send(extent),
            SenOp::Agen(agen::Op::Yield),
        ];
        let interleave = MemoryInterleave {
            granularity: Some(Elements(8)),
            region: &region,
        };
        assert_eq!(
            process_interleave_op::<Dd2>(DfirUnit::L3lu, &interleave),
            InterleaveCheck::Admissible
        );

        // ⛔ AN L3 UNIT IS THE FIRST GATE, before the granularity or the region.
        assert_eq!(
            process_interleave_op::<Dd2>(DfirUnit::Lxlu, &interleave),
            InterleaveCheck::NotAnL3Unit
        );

        let mut louder = extent;
        louder.burst_size = Elements(16);
        let region = [
            sent_load_and_send(extent),
            sent_load_and_send(louder),
            SenOp::Agen(agen::Op::Yield),
        ];
        assert_eq!(
            process_interleave_op::<Dd2>(
                DfirUnit::L3lu,
                &MemoryInterleave {
                    granularity: None,
                    region: &region,
                }
            ),
            InterleaveCheck::DifferentBurst
        );
    }

    // ────────────────────────── 212/384 and 213/384 ──────────────────────────

    /// An access record as entries 212 and 213 read one. ⛔ THE TRAIT'S TWO REAL INSTANTIATIONS ARE
    /// BUILT BY UNITS THIS BATCH DOES NOT OWN, so the test states the three fields directly.
    struct Record {
        moi: MemoryOperandIndex,
        start_addr: Option<Val>,
        dict: IndicesCoeffDict,
    }

    impl AccessRecord for Record {
        fn memory_index(&self) -> Option<MemoryOperandIndex> {
            Some(self.moi)
        }
        fn mem_view_start_addr(&self) -> Option<Val> {
            self.start_addr
        }
        fn indices_coeff_dict(&self) -> &IndicesCoeffDict {
            &self.dict
        }
    }

    /// A record owning `moi`, whose view starts at `start_addr`.
    fn record(moi: MemoryOperandIndex, start_addr: u32) -> Record {
        Record {
            moi,
            start_addr: Some(Val(start_addr)),
            dict: IndicesCoeffDict::default(),
        }
    }

    /// ⛔ NO ACCESS DETAILS IS AN INPUT ENTRY 357 PROVABLY LEAVES ALONE — it takes the `else` at
    /// `Helper.cpp:761` with nothing to iterate — and the gather still marks the op and still runs the
    /// immutable addresses, which have nothing to place.
    #[test]
    fn the_empty_gather_marks_the_op_and_places_no_address() {
        let mut marked = Marked::at([]);
        let details = AccessContainer::<Record>::default();
        let mut mutable_addrs = AccessContainer::<Val>::default();
        let mut immutable_addrs = AccessContainer::<Val>::default();

        assert_eq!(
            gather_affine_load_store_details(
                7,
                &mut marked,
                DfirUnit::Lxlu,
                &details,
                &mut mutable_addrs,
                &mut immutable_addrs
            ),
            GatheredDetails::Gathered
        );
        assert!(marked.holds(7), "`:546` sets the attribute");
        assert!(immutable_addrs.entries().is_empty());
        assert!(mutable_addrs.entries().is_empty());
    }

    /// ⛔⛔ THE L3 ARM READS THE RECORD AND THE OTHER ARM READS BY POSITION, so the same two records
    /// give the view's own start addresses on an L3 half and the updated ones anywhere else.
    #[test]
    fn the_l3_immutable_address_is_the_views_own_and_elsewhere_it_is_the_updated_one() {
        let mut details = AccessContainer::<Record>::default();
        for (moi, addr) in [
            (MemoryOperandIndex::DirSrc, 11),
            (MemoryOperandIndex::DirDst, 12),
        ] {
            if let Some(slot) = details.vacancy(moi) {
                slot.fill(record(moi, addr));
            }
        }
        let mut updated = AccessContainer::<Val>::default();
        for (moi, addr) in [
            (MemoryOperandIndex::DirSrc, 101),
            (MemoryOperandIndex::DirDst, 102),
        ] {
            if let Some(slot) = updated.vacancy(moi) {
                slot.fill(Val(addr));
            }
        }

        let mut elsewhere = AccessContainer::<Val>::default();
        assert_eq!(
            construct_immutable_address(DfirUnit::Lxlu, &details, &updated, &mut elsewhere),
            ImmutableAddresses::Constructed
        );
        assert_eq!(elsewhere.entries(), &[Val(101), Val(102)]);

        let mut on_l3 = AccessContainer::<Val>::default();
        assert_eq!(
            construct_immutable_address(DfirUnit::L3lu, &details, &updated, &mut on_l3),
            ImmutableAddresses::Constructed
        );
        assert_eq!(on_l3.entries(), &[Val(11), Val(12)]);

        // `:1221` — the bare `failure()` when the two containers disagree.
        assert_eq!(
            construct_immutable_address(
                DfirUnit::L3lu,
                &details,
                &AccessContainer::<Val>::default(),
                &mut AccessContainer::<Val>::default()
            ),
            ImmutableAddresses::SizeMismatch
        );
    }

    /// One `dataflow.program_unit` on `comp`, holding `body`.
    fn unit_holding(comp: DfirUnit, body: Vec<DfirOp>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(comp, VIEW),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// ⭐ 374's REACHABLE HALF IS THE STORE SEARCH, and this is the load-and-store pattern it finds:
    /// the symbolic store is the load result's ONE use, so entry 036 answers with it and the walk
    /// stops where entry 360 is missing.
    #[test]
    #[should_panic(expected = "e360_constructSymbolicDetailsAndAddrs")]
    fn a_symbolic_load_that_feeds_a_symbolic_store_reaches_the_details_gap() {
        let load = DfirOp::Agen(agen::Op::SymbolicVectorLoad {
            result: Val(31),
            view: VIEW,
            indices: indices(Val(30)),
            strides: vec![Val(40)],
            multicast: None,
            view_ty: view_ty(),
            ty: LANES,
        });
        let store = DfirOp::Agen(agen::Op::SymbolicVectorStore {
            value: Val(31),
            view: VIEW,
            indices: indices(Val(30)),
            strides: vec![Val(40)],
            view_ty: view_ty(),
            ty: LANES,
        });
        let body = vec![load, store];
        let unit = unit_holding(DfirUnit::Lxlu, body.clone());
        lower_symbolic_vector_load_op(&body[0], &unit, DfirUnit::Lxlu, &body);
    }

    /// A lone symbolic store passes `nullptr` for the load, and stops at the same gap.
    #[test]
    #[should_panic(expected = "e360_constructSymbolicDetailsAndAddrs")]
    fn a_lone_symbolic_store_reaches_the_details_gap() {
        let unit = unit_holding(DfirUnit::Lxsu, Vec::new());
        lower_symbolic_vector_store_op(&unit, DfirUnit::Lxsu);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 151/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH COMPOSITE FAMILY A REGION BELONGS TO — the two `isa<>` groups `checkCompositeRegion`
/// dispatches on (`Helper.cpp:207`, `:241`).
///
/// ⛔ AN OP IN NEITHER GROUP IS NOT REPRESENTABLE, and that is the reference's own fall-through:
/// `checkCompositeRegion` returns `success()` for anything else without looking at a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeFamily {
    /// `agen.composite_load`.
    Load,
    /// `agen.composite_indirect_load`.
    IndirectLoad,
    /// `agen.composite_store`.
    Store,
    /// `agen.composite_indirect_store`.
    IndirectStore,
}

/// THE OUTCOME OF [`check_composite_region`] — admissible, or WHICH of the reference's diagnostics
/// refused the region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum CompositeRegionCheck {
    /// `LogicalResult::success()`.
    Admissible,
    /// The load's region holds neither 2 nor 3 operations.
    LoadRegionSizeNotTwoOrThree,
    /// The operation before the load's yield is not a `dataflow.send`.
    LoadSendDoesNotPrecedeYield,
    /// Size 2: the operation before the store's yield is not a `dataflow.receive`.
    StoreReceiveDoesNotPrecedeYield,
    /// Size 2: nothing reads the receive's result.
    StoreYieldsIncorrectValue,
    /// Size 3: the front operation is neither a `vectorchain.constant_bitstream` nor a
    /// `dataflow.receive`.
    StoreRegionSizeThreeWrongFront,
    /// Size 3: the operation after the front one is not a `vectorchain.shuffle`.
    StoreRegionSizeThreeWithoutShuffle,
    /// Size 3: front, shuffle and terminator are not one chain.
    StoreRegionIsNotALinearChain,
    /// The store's region holds neither 2 nor 3 operations.
    StoreRegionSizeNotTwoOrThree,
}

impl CompositeRegionCheck {
    /// The C++'s own return value: `success()` only for [`Self::Admissible`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, CompositeRegionCheck::Admissible)
    }

    /// The diagnostic the C++ emits for this outcome, verbatim (`Helper.cpp:219-292`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            CompositeRegionCheck::Admissible => None,
            CompositeRegionCheck::LoadRegionSizeNotTwoOrThree => Some(
                "composite_load's region size must be 2 or 3 and only dataflow.send, \
                 vectorchain.select + dataflow.send, or vectorchain.shuffle + dataflow.send are \
                 allowed!\n",
            ),
            CompositeRegionCheck::LoadSendDoesNotPrecedeYield => {
                Some("A sendOp must precede yieldOp.\n")
            }
            CompositeRegionCheck::StoreReceiveDoesNotPrecedeYield => {
                Some("A ReceiveOp must precede yieldOp.\n")
            }
            CompositeRegionCheck::StoreYieldsIncorrectValue => {
                Some("composite_store region yielding incorrect value!")
            }
            CompositeRegionCheck::StoreRegionSizeThreeWrongFront => Some(
                "composite_store with region size 3 must contain only dataflow.receive and one \
                 vectorchain.shuffle or vectorchain.constant_bitstream and one \
                 vectorchain.shuffle!\n",
            ),
            CompositeRegionCheck::StoreRegionSizeThreeWithoutShuffle => {
                Some("composite_store with region size 3 must contain a vectorchain.shuffle!\n")
            }
            CompositeRegionCheck::StoreRegionIsNotALinearChain => {
                Some("composite_store region is not a linear chain to the terminator!\n")
            }
            CompositeRegionCheck::StoreRegionSizeNotTwoOrThree => Some(
                "composite_store's region size must be 2 or 3 and only 1 dataflow.receive OR 1 \
                 dataflow.receive/vectorchain.constant_bitstream and 1 vectorchain.shuffle is \
                 allowed!\n",
            ),
        }
    }
}

/// EVERY USE OF `bound` INSIDE ONE COMPOSITE REGION — the body's reads PLUS the terminator's.
///
/// ⛔ THE ISLAND'S `agen.yield` CARRIES NO OPERANDS (`dialects/agen.rs:105`), so a region's
/// terminator reads nothing and its operand list has to be counted separately. That is what makes
/// `hasOneUse()` answerable for a value the yield is the only reader of.
fn uses_in_region(bound: Val, body: &[DfirOp], yielded: &[Val]) -> usize {
    uses(bound, body).len() + yielded.iter().filter(|operand| **operand == bound).count()
}

/// Replaces: e151_checkCompositeRegion
///
/// `checkCompositeRegion` (`Helper.cpp:206`) — a composite transfer's region must be one of the
/// shapes the lowering knows how to read, and `body` holds it INCLUDING its `agen.yield`.
///
/// ⛔ *"Wrong send_data's def op."* IS UNREACHABLE IN THE REFERENCE: the two conjuncts need
/// `curr_op` to be both at and not at `rend` (`:236-240`), so no region is ever refused by it and it
/// gets no outcome here.
/// ⛔ `yielded` IS THE TERMINATOR'S OPERAND LIST — see [`uses_in_region`].
#[must_use]
pub fn check_composite_region(
    family: CompositeFamily,
    body: &[DfirOp],
    yielded: &[Val],
) -> CompositeRegionCheck {
    match family {
        CompositeFamily::Load | CompositeFamily::IndirectLoad => {
            // `:217-224` — `region_size != 3 && region_size != 2`.
            if body.len() != 2 && body.len() != 3 {
                return CompositeRegionCheck::LoadRegionSizeNotTwoOrThree;
            }
            // `:226-230` — `rbegin()`, one `++` past the yield, then `dyn_cast<dataflow::SendOp>`.
            let before_yield = &body[body.len() - 2];
            if !matches!(
                before_yield,
                DfirOp::Dataflow(dfir_op::dataflow::Op::Send { .. })
            ) {
                return CompositeRegionCheck::LoadSendDoesNotPrecedeYield;
            }
            // `:233-240` — the send_data check, dead for the reason in this function's own note.
            CompositeRegionCheck::Admissible
        }
        CompositeFamily::Store | CompositeFamily::IndirectStore => match body.len() {
            // `:253-262`.
            2 => {
                let front = &body[0];
                if !matches!(
                    front,
                    DfirOp::Dataflow(dfir_op::dataflow::Op::Receive { .. })
                ) {
                    return CompositeRegionCheck::StoreReceiveDoesNotPrecedeYield;
                }
                // `:259-262` — `user_begin() == user_end()`: nothing reads what the receive bound.
                if results(front)
                    .iter()
                    .all(|bound| uses_in_region(*bound, body, yielded) == 0)
                {
                    return CompositeRegionCheck::StoreYieldsIncorrectValue;
                }
                CompositeRegionCheck::Admissible
            }
            // `:263-285`.
            3 => {
                let front = &body[0];
                if !matches!(
                    front,
                    DfirOp::VectorChain(dfir_op::vectorchain::Op::ConstantBitstream { .. })
                        | DfirOp::Dataflow(dfir_op::dataflow::Op::Receive { .. })
                ) {
                    return CompositeRegionCheck::StoreRegionSizeThreeWrongFront;
                }
                // `:274` — `dyn_cast<ShuffleOp>(curr_op.getNextNode())`.
                let DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle {
                    result: shuffled, ..
                }) = &body[1]
                else {
                    return CompositeRegionCheck::StoreRegionSizeThreeWithoutShuffle;
                };
                // `:281-282` — `!curr_op.hasOneUse() || !shuffle_op->hasOneUse() ||
                // *shuffle_op->getUsers().begin() != yield_op`.
                let front_uses: usize = results(front)
                    .iter()
                    .map(|bound| uses_in_region(*bound, body, yielded))
                    .sum();
                if front_uses != 1
                    || uses_in_region(*shuffled, body, yielded) != 1
                    || !yielded.contains(shuffled)
                {
                    return CompositeRegionCheck::StoreRegionIsNotALinearChain;
                }
                CompositeRegionCheck::Admissible
            }
            _ => CompositeRegionCheck::StoreRegionSizeNotTwoOrThree,
        },
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 152/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE OUTCOME OF [`check_store_op_from_extract_pattern`] — the reference's four diagnostics as
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum StoreFromExtractCheck {
    /// A 1D identity store from offset 0 — `return true`.
    Admissible,
    /// The subscript is not one index long.
    ExpectingIndicesSizeOne,
    /// That index is not the constant 0.
    ExpectingStartOffsetZero,
    /// The `store_set` is not one-dimensional.
    ExpectingOneDimStoreSet,
    /// The `store_order` is not the one-dimensional identity.
    ExpectingOneDimIdentityStoreOrder,
}

impl StoreFromExtractCheck {
    /// The C++'s own return value: `true` only for [`Self::Admissible`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, StoreFromExtractCheck::Admissible)
    }

    /// The diagnostic the C++ emits for this outcome, verbatim (`Helper.cpp:437-456`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            StoreFromExtractCheck::Admissible => None,
            StoreFromExtractCheck::ExpectingIndicesSizeOne => Some("expecting indices size 1"),
            StoreFromExtractCheck::ExpectingStartOffsetZero => {
                Some("expecting start offset of 0 from vector_store")
            }
            StoreFromExtractCheck::ExpectingOneDimStoreSet => Some("expecting 1D store_set"),
            StoreFromExtractCheck::ExpectingOneDimIdentityStoreOrder => {
                Some("expecting 1D identity map for store_map")
            }
        }
    }
}

/// Replaces: e152_checkStoreOpFromExtractPattern
///
/// `checkStoreOpFromExtractPattern` (`Helper.cpp:433`) — *"StoreOp should be for a 1d space with 0
/// start address"*, taking the `agen.vector_store`'s own subscript and view type.
///
/// ⛔ THE REFERENCE NULL-DEREFERENCES A NON-CONSTANT INDEX: `dyn_cast<arith::ConstantOp>` is
/// unguarded and `.getValue()` follows (`:440-443`), so an index that is a loop variable crashes it.
/// Here that answers [`StoreFromExtractCheck::ExpectingStartOffsetZero`], which is what the guarded
/// read would have said.
#[must_use]
pub fn check_store_op_from_extract_pattern(
    indices: &[Index],
    view_ty: &MemRef,
    scope: &[DfirOp],
) -> StoreFromExtractCheck {
    // `:435-439` — `getMapOperands().size() != 1`.
    let [start] = indices else {
        return StoreFromExtractCheck::ExpectingIndicesSizeOne;
    };
    // `:440-447` — the operand's defining `arith.constant`, and its integer must be 0. A map
    // operand is typed `index`, so `arith.constant` here is the island's index form.
    let start_addr = match start {
        Index::Const(value) => Some(*value),
        Index::Val(val) => match defining_op(*val, scope) {
            Some(DfirOp::Arith(dfir_op::arith::Op::Constant { value, .. })) => Some(*value),
            _ => None,
        },
        Index::Strided(..) => None,
    };
    if start_addr != Some(0) {
        return StoreFromExtractCheck::ExpectingStartOffsetZero;
    }
    // `:449-452` — `getStoreSet().getValue().getNumDims() != 1`. The island synthesises the set over
    // the view's dimensions (`dialects/agen.rs:258-270`), so its dim count IS the view's rank.
    if view_ty.shape.len() != 1 {
        return StoreFromExtractCheck::ExpectingOneDimStoreSet;
    }
    // `:454-458` — `getStoreOrder()`'s `getNumDims() != 1 || !isIdentity()`. ⭐ THE ISLAND'S
    // `store_order` IS `identity_map(rank)` (`:241-244`), so it is always an identity and only the
    // rank can refuse — the same question the set answered, kept as its own check because the
    // reference reports it with its own message.
    if view_ty.shape.len() != 1 {
        return StoreFromExtractCheck::ExpectingOneDimIdentityStoreOrder;
    }
    // `:460`
    StoreFromExtractCheck::Admissible
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 153/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE UNIT A VIEW IS CUT FROM — `getMemRef()` → `dataflow.get_logical_memory_view` →
/// `getFromUnit()` → `dataflow.get_unit`, the walk entries 153 and 154 both open with.
///
/// ⛔ `None` IS EVERY ONE OF THE REFERENCE'S THREE NULL `dyn_cast`s: a view that is the PAGED op, a
/// from-unit that is not a `get_unit`, and an operand no op in scope defines.
fn viewed_unit(view: Val, scope: &[DfirOp]) -> Option<DfirUnit> {
    let DfirOp::Dataflow(dfir_op::dataflow::Op::GetLogicalMemoryView { from, .. }) =
        defining_op(view, scope)?
    else {
        return None;
    };
    let DfirOp::Dataflow(dfir_op::dataflow::Op::GetUnit { unit, .. }) = defining_op(*from, scope)?
    else {
        return None;
    };
    Some(*unit)
}

/// Replaces: e153_isLoadAndExtractScalarPattern
///
/// `isLoadAndExtractScalarPattern` (`Helper.cpp:463`) — an `agen.vector_load` OF THE LX whose one
/// consumer stores INTO THE VIRTUAL IBR: the gather's index arriving where an address can read it.
///
/// ⛔ THE TWO COMPONENTS ARE DIFFERENT UNITS AND THE ORDER IS THE PATTERN — LX for the load, and
/// [`DfirUnit::LxVirtualIbr`] for the store. A load out of the IBR is not this pattern.
/// ⭐ THE REFERENCE TAKES A `VectorLoadOp&`; anything else answers `false` here rather than being
/// unrepresentable, because the callers walk a body and ask of each statement.
#[must_use]
pub fn is_load_and_extract_scalar_pattern(load: &DfirOp, scope: &[DfirOp]) -> bool {
    let DfirOp::Agen(dfir_op::agen::Op::VectorLoad { result, view, .. }) = load else {
        return false;
    };
    // `:465-474` — the load's view must be cut from the LX itself.
    if viewed_unit(*view, scope) != Some(DfirUnit::Lx) {
        return false;
    }
    // `:476-478` — `!load_op->hasOneUse()`, then the user must be an `agen.vector_store`.
    let users = uses(*result, scope);
    let [
        DfirOp::Agen(dfir_op::agen::Op::VectorStore {
            view: store_view, ..
        }),
    ] = users.as_slice()
    else {
        return false;
    };
    // `:479-488` — and that store's view must be cut from the virtual IBR.
    viewed_unit(*store_view, scope) == Some(DfirUnit::LxVirtualIbr)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 154/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e154_isReceiveAndExtractScalarPattern
///
/// `isReceiveAndExtractScalarPattern` (`Helper.cpp:493`) — an `agen.vector_store` INTO THE VIRTUAL
/// IBR, which is a received index landing where an address can read it.
///
/// ⛔ `dcc::getUnitType()` IS NOT USABLE HERE and the reference says why (`:502-504`): a memory
/// view's `type=` spells L0/LX/PE components that the generic-component enum does not carry. The
/// comparison is against the view's own unit, which is what [`viewed_unit`] answers.
/// ⭐ `DT_CHECK(store_op)` has no representation: a null op cannot be passed.
#[must_use]
pub fn is_receive_and_extract_scalar_pattern(store: &DfirOp, scope: &[DfirOp]) -> bool {
    let DfirOp::Agen(dfir_op::agen::Op::VectorStore { view, .. }) = store else {
        return false;
    };
    // `:496-509`
    viewed_unit(*view, scope) == Some(DfirUnit::LxVirtualIbr)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 155/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE **OPERATION** STANDING FOR ANOTHER — the half of MLIR's `IRMapping` that
/// [`ValueMapping`] does not carry.
///
/// ⛔⛔ `Operation::clone(IRMapping&)` RECORDS `mapper.map(this, newOp)`, which is what makes
/// `ir_map.lookupOrDefault(ad.getOp())` answer with the clone: `copyLoopBody` clones every statement
/// of the body through one map and hands it back (`Utils.cpp:357-381`). Without an op half, entry 155
/// could remap an access's values but not the access's own operation.
///
/// ⭐ KEYED BY ADDRESS, as the C++ keys on `Operation*` — two structurally equal ops are two ops.
#[derive(Debug, Default, Clone)]
pub struct OpMapping<'a> {
    pairs: Vec<(&'a dfir_op::agen::Op, &'a dfir_op::agen::Op)>,
}

impl<'a> OpMapping<'a> {
    /// An empty mapping: every operation stands for itself.
    #[must_use]
    pub fn new() -> OpMapping<'a> {
        OpMapping { pairs: Vec::new() }
    }

    /// `from` now stands for `to`.
    pub fn map(&mut self, from: &'a dfir_op::agen::Op, to: &'a dfir_op::agen::Op) {
        self.pairs.push((from, to));
    }

    /// What `op` stands for — ITSELF where nothing was mapped, which is `lookupOrDefault`. The search
    /// is from the back, so the newest entry wins, as in [`ValueMapping::lookup_or_default`].
    #[must_use]
    pub fn lookup_or_default(&self, op: &'a dfir_op::agen::Op) -> &'a dfir_op::agen::Op {
        self.pairs
            .iter()
            .rev()
            .find(|(from, _)| core::ptr::eq(*from, op))
            .map_or(op, |(_, to)| *to)
    }
}

/// Replaces: e155_updateSymbolicAccessDetails
///
/// `updateSymbolicAccessDetails` (`Helper.cpp:1013`) — a symbolic access built against a loop body
/// that has since been CLONED names the original's values; this reads all six of its handles through
/// the clone's mapping, in the reference's order.
///
/// ⛔ THE OP REMAP IS BEHAVIOUR, NOT BOOKKEEPING, and the reference says why: *"This will always be
/// in the innermost loop so it will always be cloned"* (`:1016-1017`). See [`OpMapping`].
/// ⛔ A NULL `Value` MAPS TO ITSELF — `lookupOrDefault` on the three optional handles leaves [`None`].
pub fn update_symbolic_access_details<'a>(
    access_details: &mut AccessContainer<AccessDetailsSymbolic<'a>>,
    ir_map: &ValueMapping,
    op_map: &OpMapping<'a>,
) {
    // `:1015` — `for (auto& ad : access_details)`, over the container's own insertion order.
    for ad in access_details.entries_mut() {
        // `:1018-1019` — the operation.
        ad.base.op = op_map.lookup_or_default(ad.base.op);

        // `:1021-1027` — every index, then `setIndices`.
        let indices: Vec<Val> = ad
            .base
            .indices
            .iter()
            .map(|index| ir_map.lookup_or_default(*index))
            .collect();
        ad.base.set_indices(&indices);

        // `:1029-1035` — every stride, then `setStrides`.
        let strides: Vec<Val> = ad
            .strides()
            .iter()
            .map(|stride| ir_map.lookup_or_default(*stride))
            .collect();
        ad.set_strides(&strides);

        // `:1037-1039` — the mem_ref.
        ad.base.mem_ref = ad
            .base
            .mem_ref
            .map(|mem_ref| ir_map.lookup_or_default(mem_ref));

        // `:1041-1043` — the mem_view_start_addr.
        if let Some(start) = ad.base.mem_view_start_addr {
            ad.base
                .set_mem_view_start_addr(ir_map.lookup_or_default(start));
        }

        // `:1046-1047` — the memory operand.
        ad.base.memory = ad
            .base
            .memory
            .map(|memory| ir_map.lookup_or_default(memory));
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 156/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A STORE AS `getStoreProducer` CLASSES IT — five stores, each naming where the search starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgenStore<'a> {
    /// `agen.vector_store` — the search starts at the stored value.
    Vector {
        /// The stored vector.
        value: Val,
    },
    /// `agen.indirect_vector_store`.
    IndirectVector {
        /// The stored vector.
        value: Val,
    },
    /// `agen.symbolic_vector_store`.
    SymbolicVector {
        /// The stored vector.
        value: Val,
    },
    /// `agen.composite_load_and_store` — the input vector when it has one, else the region's front.
    Composite {
        /// The `input_vector` operand.
        input_vector: Option<Val>,
        /// The store region's statements, terminator last.
        body: &'a [DfirOp],
    },
    /// `agen.composite_indirect_load_and_store`, read through its region only.
    CompositeIndirect {
        /// The store region's statements, terminator last.
        body: &'a [DfirOp],
    },
}

/// WHAT FEEDS A STORE: the op holding the value, and the op behind THAT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreProducer<'a> {
    /// The pair the reference returns.
    Found {
        /// The receive or shuffle holding the stored value.
        inp_op: &'a DfirOp,
        /// What feeds `inp_op`.
        producer: &'a DfirOp,
    },
    /// The stored value came from neither a receive nor a shuffle.
    StoreValueNotReceiveOrShuffle,
    /// A `vectorchain.coalesce` input that no receive precedes.
    CoalesceNotPrecededByReceive,
    /// A composite's `input_vector` came from none of the three accepted ops.
    CompositeInputUnsupported,
    /// No first stage matched at all.
    UnsupportedStoreProducer1,
    /// A receive fed by none of the four accepted producers.
    ReceiveProducerUnsupported,
    /// A shuffle over a receive whose own producer is not a `get_unit`.
    ShuffleReceiveProducerNotAGetUnit,
    /// A constant bitstream carrying more or fewer than one value.
    BitstreamNotOneValue,
    /// A constant bitstream of an unsupported width.
    BitstreamWidthNotSupported,
    /// A shuffle over a bitstream that is not a first-element splat.
    UnsupportedSplatMode,
    /// A splat whose result shape is none of the three the hardware has.
    UnsupportedSplatResult,
    /// A shuffle fed by neither a receive nor a bitstream.
    ShuffleInputNotReceiveOrBitstream,
    /// The value's op was neither a receive nor a shuffle after the first stage.
    UnsupportedStoreProducer2,
}

impl<'a> StoreProducer<'a> {
    /// The reference's message for a rejection, verbatim.
    #[must_use]
    pub fn diagnostic(self) -> Option<&'static str> {
        match self {
            StoreProducer::Found { .. } => None,
            StoreProducer::StoreValueNotReceiveOrShuffle => {
                Some("expecting a ReceiveOp or a ShuffleOp as input to VectorStoreOp")
            }
            StoreProducer::CoalesceNotPrecededByReceive => {
                Some("must be preceded by dataflow.receiveOp.")
            }
            StoreProducer::CompositeInputUnsupported => Some(
                "must be preceded by vectorchain.coalesce, vectorchain.shuffle, or dataflow.receive.",
            ),
            StoreProducer::UnsupportedStoreProducer1 => Some("unsupported storeOp producer! (1)"),
            StoreProducer::ReceiveProducerUnsupported => Some(
                "expected a get_unit, create_multicast_group, or ifOp producer for receive inputs",
            ),
            StoreProducer::ShuffleReceiveProducerNotAGetUnit => {
                Some("producers for ReceiveOps feeding ShuffleOps should be a GetUnitOp")
            }
            StoreProducer::BitstreamNotOneValue => {
                Some("ConstantBitstreamOp producers should contain 1 value")
            }
            StoreProducer::BitstreamWidthNotSupported => {
                Some("ConstantBitstreamOp producers should be 8 or 16 bits")
            }
            StoreProducer::UnsupportedSplatMode => {
                Some("unsupported splat mode for ShuffleOp input")
            }
            StoreProducer::UnsupportedSplatResult => {
                Some("unsupported splat result for ShuffleOp input")
            }
            StoreProducer::ShuffleInputNotReceiveOrBitstream => {
                Some("ShuffleOp inp_op producers can only be a ReceiveOp or a ConstantBitstreamOp")
            }
            StoreProducer::UnsupportedStoreProducer2 => Some("unsupported storeOp producer! (2)"),
        }
    }
}

/// EVERY EXPANDED SHUFFLE INDEX IS 0 — `isFirstElemSplat` (`VectorChain/Utils.cpp:80-88`).
///
/// ⭐ The reference expands each index `repetition` times before testing; a repeated 0 is still 0, so
/// the repetition count cannot change the answer and is not read here.
fn is_first_elem_splat(indices: &[i32]) -> bool {
    indices.iter().all(|index| *index == 0)
}

/// `findYieldsResolvingTo<dataflow::GetUnitOp, scf::IfOp>` (`dcc/src/Utils/Utils.cpp:183-208`) — does
/// either arm of this conditional yield something that resolves to a `dataflow.get_unit`?
///
/// ⛔ A BLOCK ARGUMENT IS SKIPPED, NOT REJECTED (`:188`): here it is an operand with no defining op.
fn yields_resolving_to_get_unit(if_op: &DfirOp, result_index: usize, scope: &[DfirOp]) -> bool {
    let DfirOp::Scf(dfir_op::scf::Op::If {
        body, else_body, ..
    }) = if_op
    else {
        return false;
    };
    [body, else_body].into_iter().any(|region| {
        // `getRegion(i).front().getTerminator()->getOperand(result_index)`.
        let Some(DfirOp::Scf(dfir_op::scf::Op::Yield { operands })) = region.last() else {
            return false;
        };
        let Some(yielded) = operands.get(result_index) else {
            return false;
        };
        match defining_op(*yielded, scope) {
            Some(DfirOp::Dataflow(dataflow::Op::GetUnit { .. })) => true,
            // `:196-201` — a query over a mapping every one of whose values is a `get_unit`.
            Some(DfirOp::Uniform(dfir_op::uniform::Op::QueryMap { map, .. })) => matches!(
                defining_op(*map, scope),
                Some(DfirOp::Uniform(dfir_op::uniform::Op::DefImmutableMapping { pairs, .. }))
                    if pairs.iter().all(|(_, value)| matches!(
                        defining_op(*value, scope),
                        Some(DfirOp::Dataflow(dataflow::Op::GetUnit { .. }))
                    ))
            ),
            // `:202-205` — a nested conditional, asked about the result the yield actually names.
            Some(nested @ DfirOp::Scf(dfir_op::scf::Op::If { results, .. })) => results
                .iter()
                .position(|result| *result == *yielded)
                .is_some_and(|at| yields_resolving_to_get_unit(nested, at, scope)),
            _ => false,
        }
    })
}

/// The four producers the `DT_CHECK` at `Helper.cpp:1375-1387` accepts behind a `dataflow.receive`.
///
/// ⛔ `dataflow.create_multicast_group` HAS NO ISLAND OP, so two of the four arms — the direct one and
/// `findYieldsResolvingTo<CreateMulticastGroupOp, IfOp>` — have nothing here to match. The gap is
/// recorded rather than stood in for.
fn receive_producer_is_supported(producer: &DfirOp, scope: &[DfirOp]) -> bool {
    match producer {
        DfirOp::Dataflow(dataflow::Op::GetUnit { .. })
        | DfirOp::Uniform(dfir_op::uniform::Op::QueryMap { .. }) => true,
        DfirOp::Scf(dfir_op::scf::Op::If { .. }) => {
            yields_resolving_to_get_unit(producer, 0, scope)
        }
        _ => false,
    }
}

/// Replaces: e156_getStoreProducer
///
/// `getStoreProducer` (`Helper.cpp:1285`) — the op holding a store's value, and the op behind THAT,
/// which is what tells the destination lowering where the data came from.
///
/// ⛔ TWO STAGES, AND THE SECOND RE-EXAMINES THE FIRST'S ANSWER: stage one finds `inp_op` per store
/// class, stage two accepts it only as a receive (four producer classes) or a shuffle (a receive over
/// a `get_unit`, or a splat of a one-value bitstream at 4/8/16 bits).
/// ⛔ THE SPLAT'S RESULT SHAPE IS CHECKED AGAINST THE STICK, not against the bitstream: 256×4b,
/// 128×8b or 64×16b and nothing else (`:1427-1429`).
#[must_use]
pub fn get_store_producer<'a>(store: &AgenStore<'a>, scope: &'a [DfirOp]) -> StoreProducer<'a> {
    // Stage one — `:1287-1366`. Each class reaches its own `inp_op`.
    let inp_op: &'a DfirOp = match store {
        // `:1288-1301` — the three plain stores are read identically.
        AgenStore::Vector { value }
        | AgenStore::IndirectVector { value }
        | AgenStore::SymbolicVector { value } => {
            let Some(def) = defining_op(*value, scope) else {
                return StoreProducer::UnsupportedStoreProducer1;
            };
            match def {
                DfirOp::Dataflow(dataflow::Op::Receive { .. })
                | DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle { .. }) => def,
                _ => return StoreProducer::StoreValueNotReceiveOrShuffle,
            }
        }
        // `:1302-1328` — a composite with an input vector. The reference's middle arm is a
        // `vectorchain.coalesce`, which this island does not have; see
        // [`StoreProducer::CoalesceNotPrecededByReceive`].
        AgenStore::Composite {
            input_vector: Some(input),
            ..
        } => {
            let Some(def) = defining_op(*input, scope) else {
                return StoreProducer::CompositeInputUnsupported;
            };
            match def {
                DfirOp::Dataflow(dataflow::Op::Receive { .. })
                | DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle { .. }) => def,
                _ => return StoreProducer::CompositeInputUnsupported,
            }
        }
        // `:1330-1348` and `:1350-1366` — no input vector: the store region's first two statements
        // decide. A receive that yields straight away IS the input op; anything else must be the
        // shuffle behind it.
        AgenStore::Composite { body, .. } | AgenStore::CompositeIndirect { body } => {
            let Some(front) = body.first() else {
                return StoreProducer::UnsupportedStoreProducer1;
            };
            let next = body.get(1);
            match front {
                DfirOp::Dataflow(dataflow::Op::Receive { .. }) => match next {
                    Some(DfirOp::Agen(dfir_op::agen::Op::Yield)) => front,
                    Some(
                        shuffle @ DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle { .. }),
                    ) => shuffle,
                    _ => return StoreProducer::UnsupportedStoreProducer1,
                },
                DfirOp::VectorChain(dfir_op::vectorchain::Op::ConstantBitstream { .. }) => {
                    match next {
                        Some(
                            shuffle @ DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle {
                                ..
                            }),
                        ) => shuffle,
                        _ => return StoreProducer::UnsupportedStoreProducer1,
                    }
                }
                _ => return StoreProducer::UnsupportedStoreProducer1,
            }
        }
    };

    // Stage two — `:1369-1442`.
    let producer: &'a DfirOp = match inp_op {
        // `:1375-1388` — the receive's own producer.
        DfirOp::Dataflow(dataflow::Op::Receive { from, .. }) => {
            let Some(producer) = defining_op(from.val(), scope) else {
                return StoreProducer::ReceiveProducerUnsupported;
            };
            if !receive_producer_is_supported(producer, scope) {
                return StoreProducer::ReceiveProducerUnsupported;
            }
            producer
        }
        // `:1390-1436` — the shuffle's input, which is a receive or a splat source.
        DfirOp::VectorChain(dfir_op::vectorchain::Op::Shuffle {
            input, indices, ty, ..
        }) => match defining_op(*input, scope) {
            Some(DfirOp::Dataflow(dataflow::Op::Receive { from, .. })) => {
                let Some(producer) = defining_op(from.val(), scope) else {
                    return StoreProducer::ShuffleReceiveProducerNotAGetUnit;
                };
                if !matches!(producer, DfirOp::Dataflow(dataflow::Op::GetUnit { .. })) {
                    return StoreProducer::ShuffleReceiveProducerNotAGetUnit;
                }
                producer
            }
            Some(
                bitstream @ DfirOp::VectorChain(dfir_op::vectorchain::Op::ConstantBitstream {
                    value,
                    ty: bitstream_ty,
                    ..
                }),
            ) => {
                if value.len() != 1 {
                    return StoreProducer::BitstreamNotOneValue;
                }
                let bitstream_bits = bitstream_ty.elem.bits();
                if bitstream_bits != 4 && bitstream_bits != 8 && bitstream_bits != 16 {
                    return StoreProducer::BitstreamWidthNotSupported;
                }
                if !is_first_elem_splat(indices) {
                    return StoreProducer::UnsupportedSplatMode;
                }
                let splat_bits = ty.elem.bits();
                let stick_wide = (ty.len == 256 && splat_bits == 4)
                    || (ty.len == 128 && splat_bits == 8)
                    || (ty.len == 64 && splat_bits == 16);
                if !stick_wide {
                    return StoreProducer::UnsupportedSplatResult;
                }
                bitstream
            }
            _ => return StoreProducer::ShuffleInputNotReceiveOrBitstream,
        },
        _ => return StoreProducer::UnsupportedStoreProducer2,
    };

    StoreProducer::Found { inp_op, producer }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 157/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE OF `set_transfer_mask_state`'s MASK ATTRIBUTE PAIRS — `num_unmasked_elements[i]` beside
/// `num_masked_elements[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskRun {
    /// How many elements the mask leaves live.
    pub unmasked: Elements,
    /// How many it masks off.
    pub masked: Elements,
}

impl MaskRun {
    /// The run's whole length. The verifier has already made the two attributes the same length.
    #[must_use]
    pub const fn elems(self) -> Elements {
        Elements(self.unmasked.0 + self.masked.0)
    }
}

/// A `slice_mask_map` AS THE THREE SAMV PATTERNS, so that `isUnmask`, `isFullMask`, `isGenericSAMV`
/// and `getSliceIDXsl` are all answered by the variant instead of re-parsed from the string.
///
/// ⛔ THE TWO DEGENERATE PATTERNS CARRY NO MASK, which is the reference's two `DT_CHECK_MSG`s —
/// *"SAMVs resetting the mask should not have mask attributes"* and *"SAMVs fully masking should not
/// contain a mask"* — held by the type instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceMaskMap {
    /// `(0)(0)(0)(0)(0)(0)(0)(0)` — the mask is being reset.
    Unmask,
    /// `(1)(1)(1)(1)(1)(1)(1)(1)` — everything is masked.
    FullMask,
    /// 0-7 `(A)`, then one `(A|B)`, then 0-7 `(1)`: two masks, and the index of the `(A|B)` slice.
    Generic {
        /// `getSliceIDXsl` — which slice carries the crossover.
        slice_id_xsl: sen::SliceId,
        /// maskA, the within-slice mask.
        wsl: MaskRun,
        /// maskB, the cross-slice mask.
        xsl: MaskRun,
    },
}

/// `agen.set_transfer_mask_state`'s operands and attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetTransferMaskState {
    /// `$mask_value`.
    pub mask_value: Val,
    /// `$num_slices`.
    pub num_slices: NonZeroU32,
    /// `$slice_mask_map`.
    pub slice_mask_map: SliceMaskMap,
    /// The result vector's type — its element count and precision are both read.
    pub ty: Vector,
    /// `$dbgName`.
    pub dbg_name: Option<String>,
}

/// The `sentient.samv` a `set_transfer_mask_state` becomes, or why it cannot become one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActiveMaskValue {
    /// The emitted op.
    Samv(SenOp),
    /// The unit is not an LXLU.
    NotInLxlu,
    /// SAMV is defined for eight slices only.
    NotEightSlices,
    /// The mask does not span a stick.
    NotStickLength,
    /// Eight slices share fewer than eight elements, so no slice holds one.
    SliceHoldsNoElements,
    /// The inner mask does not tile the slice.
    InnerDimNotDivisible,
    /// The outer mask covers less or more than a slice.
    OuterDimDoesNotSpanSlice,
}

impl ActiveMaskValue {
    /// The reference's message for a rejection, verbatim.
    ///
    /// ⛔ [`SliceHoldsNoElements`](ActiveMaskValue::SliceHoldsNoElements) HAS NO MESSAGE because the
    /// reference has no check: `elems_per_slice == 0` reaches `(int)std::log2(0)`
    /// (`Helper.cpp:2716-2720`), and the stick-length check (`:2589-2590`) makes that need
    /// `precision > 128` — so it is unreachable there.
    #[must_use]
    pub fn diagnostic(&self) -> Option<&'static str> {
        match self {
            ActiveMaskValue::Samv(_) | ActiveMaskValue::SliceHoldsNoElements => None,
            ActiveMaskValue::NotInLxlu => Some("only supported in LXLU unit"),
            ActiveMaskValue::NotEightSlices => Some("SAMV requires 8 slices"),
            ActiveMaskValue::NotStickLength => Some("SAMV requires masks of stick length"),
            ActiveMaskValue::InnerDimNotDivisible => {
                Some("inner dim mask should be evenly divisible into the slice")
            }
            ActiveMaskValue::OuterDimDoesNotSpanSlice => {
                Some("outer dim masked and unmasked should span a whole slice")
            }
        }
    }
}

/// Replaces: e157_constructSetActiveMaskValueOp
///
/// `constructSetActiveMaskValueOp` (`Helper.cpp:2566`) — an `agen.set_transfer_mask_state` becomes
/// one `sentient.samv`, whose `numvalidentry` PACKS TWO COUNTS: the outer dimension's valid entries
/// shifted by the bits the inner dimension needs, plus the inner's, swapped when the cross-slice mask
/// is the inner one.
///
/// ⛔ A FULL COUNT ENCODES AS ZERO, twice (`:2680`, `:2700`): `inner == inner_dim_len` and
/// `outer == outer_dim_len` are both written back as 0, so the field never has to hold the width.
/// ⛔ `maskall` IS TRUE FOR THE FULL-MASK PATTERN ONLY; the generic path always writes false.
/// ⛔ THE REFERENCE COMPUTES `mask_wsl_elems` AND NEVER READS IT (`:2648`) — only maskB's total
/// decides `xslinner`.
#[must_use]
pub fn construct_set_active_mask_value_op<A: Arch>(
    unit: DfirUnit,
    mask_op: &SetTransferMaskState,
) -> ActiveMaskValue {
    // `:2569-2573` — LXLU only.
    if unit != DfirUnit::Lxlu {
        return ActiveMaskValue::NotInLxlu;
    }

    // `:2575-2577`.
    let num_slices = u64::from(mask_op.num_slices.get());
    if num_slices != 8 {
        return ActiveMaskValue::NotEightSlices;
    }

    // `:2579-2584` and `:2586-2591` — the slice's share, and the mask's own width.
    let elems_per_slice = Elements(mask_op.ty.len / num_slices);
    let precision = u64::from(mask_op.ty.elem.bits());
    if mask_op.ty.len * precision != A::BYTES_PER_STICK.get() * 8 {
        return ActiveMaskValue::NotStickLength;
    }
    let raw_precision = sen::RawPrecision(u32::try_from(precision).unwrap_or(u32::MAX));

    let (mask_all, num_valid_entry, slice_id_xsl, xsl_inner, wsl_len) = match &mask_op
        .slice_mask_map
    {
        // `:2595-2606`.
        SliceMaskMap::Unmask => (false, 0, sen::SliceId(7), false, 1),
        // `:2607-2618`.
        SliceMaskMap::FullMask => (true, 0, sen::SliceId(0), false, 1),
        SliceMaskMap::Generic {
            slice_id_xsl,
            wsl,
            xsl,
        } => {
            if elems_per_slice.0 == 0 {
                return ActiveMaskValue::SliceHoldsNoElements;
            }

            // `:2651` — maskB spanning the slice means the cross-slice mask is the OUTER one.
            let xsl_inner = xsl.elems() != elems_per_slice;
            // `:2657-2671` — which mask is which, then the inner dimension's length.
            let (inner, outer) = if xsl_inner { (xsl, wsl) } else { (wsl, xsl) };
            let inner_dim_len = inner.elems().0;
            // A zero-length inner mask divides nothing, which is this same rejection rather than the
            // reference's division by zero.
            if elems_per_slice.0.checked_rem(inner_dim_len) != Some(0) {
                return ActiveMaskValue::InnerDimNotDivisible;
            }

            // `:2679-2680`.
            let mut inner_dim_valid = inner.unmasked.0;
            if inner_dim_valid == inner_dim_len {
                inner_dim_valid = 0;
            }

            // `:2684`.
            let outer_dim_len = elems_per_slice.0 / inner_dim_len;

            // `:2693-2700`.
            if outer.elems() != elems_per_slice {
                return ActiveMaskValue::OuterDimDoesNotSpanSlice;
            }
            let mut outer_dim_valid = outer.unmasked.0 / inner_dim_len;
            if outer_dim_valid == outer_dim_len {
                outer_dim_valid = 0;
            }

            // `:2716-2720` — the packing.
            let num_valid_entry = if xsl_inner {
                (inner_dim_valid << outer_dim_len.checked_ilog2().unwrap_or(0)) + outer_dim_valid
            } else {
                (outer_dim_valid << inner_dim_len.checked_ilog2().unwrap_or(0)) + inner_dim_valid
            };
            let wsl_len = if xsl_inner {
                outer_dim_len
            } else {
                inner_dim_len
            };
            (false, num_valid_entry, *slice_id_xsl, xsl_inner, wsl_len)
        }
    };

    // `:2724-2728`.
    ActiveMaskValue::Samv(SenOp::Sentient(sen::Op::Samv {
        mask_value: mask_op.mask_value,
        mask_all,
        num_valid_entry: sen::ValidEntries(u32::try_from(num_valid_entry).unwrap_or(u32::MAX)),
        slice_id_xsl,
        xsl_inner,
        wsl_len: sen::WslLen(u32::try_from(wsl_len).unwrap_or(u32::MAX)),
        precision: raw_precision,
        dbg_name: mask_op.dbg_name.clone(),
    }))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 158/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH REGION-BEARING OP the units and list sizes are read from — the reference's two accepted
/// classes, so that its `DT_ERROR("num_of_regions was not set")` third case cannot be spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionOpKind {
    /// `uniform.uniformize_regions`.
    UniformizeRegions,
    /// `uniform.equalize_pattern`.
    EqualizePattern,
}

/// ONE UNIT OPERAND, PLACED RELATIVE TO THE LOOP the new op is being lifted out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitOperand<'a> {
    /// Defined outside the loop already — usable as it stands.
    Outside(Val),
    /// Defined inside the loop by a `dataflow.create_group` or `dataflow.get_unit`, which is cloned.
    InsideLoop {
        /// The operand as it stands.
        unit: Val,
        /// The op to clone.
        def: &'a DfirOp,
    },
    /// ⛔ DEFINED INSIDE THE LOOP BY SOMETHING ELSE, and the reference leaves it there (`:3906`) — the
    /// new op then reads a value its own position dominates nothing of.
    InsideLoopOther(Val),
}

/// The `$units`/`$list_sizes` pair of the op being rebuilt, zipped per region as [`LocalRegion`] takes
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformizeSource<'a> {
    /// Which op they came from.
    pub kind: RegionOpKind,
    /// One unit list per region; `regions.len()` is `getNumRegions()`.
    pub regions: Vec<Vec<UnitOperand<'a>>>,
}

/// A NEWLY BUILT `uniform.uniformize_regions`, with the clones that must precede it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uniformized {
    /// The op.
    pub op: DfirOp,
    /// The unit definitions hoisted out of the loop, in the order they were cloned.
    pub hoisted: Vec<DfirOp>,
    /// The `"active"` attribute — bookkeeping between one call and the next, never emitted.
    pub active: bool,
}

/// Which op a region got: a new one, or the one an earlier region of the same op already made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreatedUniformize {
    /// The first region built this.
    Created(Box<Uniformized>),
    /// A later region reuses the op at this position among the preceding ops.
    Existing(usize),
    /// ⛔ NOTHING ACTIVE PRECEDES THE LOOP. The reference's `while (!prev_node->hasAttr("active"))`
    /// walks off the front of the block and dereferences null (`:3931-3933`).
    NoActiveOp,
}

/// Replaces: e158_createUniformizeRegionsOp
///
/// `createUniformizeRegionsOp` (`Helper.cpp:3880`) — the FIRST region of a region-bearing op builds one
/// `uniform.uniformize_regions` outside the loop, whose every region is an empty block yielding its own
/// argument; every later region of that same op finds and reuses it.
///
/// ⛔ THE HANDOFF IS AN ATTRIBUTE ON THE IR: `"active"` is set only when there is more than one region,
/// searched for backwards from the loop, and removed by the LAST region (`:3923-3937`).
/// ⛔ ONLY `create_group` AND `get_unit` ARE HOISTED; any other in-loop definition is left where it is.
#[must_use]
pub fn create_uniformize_regions_op(
    values: &mut Values,
    source: &UniformizeSource<'_>,
    region_idx: usize,
    preceding: &mut [Uniformized],
) -> CreatedUniformize {
    let num_of_regions = source.regions.len();

    // `:3926-3938` — the later regions do not build anything.
    if region_idx != 0 {
        let Some(at) = preceding.iter().rposition(|earlier| earlier.active) else {
            return CreatedUniformize::NoActiveOp;
        };
        if region_idx == num_of_regions.saturating_sub(1) {
            preceding[at].active = false;
        }
        return CreatedUniformize::Existing(at);
    }

    // `:3902-3911` — make sure the units are outside the loop.
    let mut hoisted: Vec<DfirOp> = Vec::new();
    let mut mapping = ValueMapping::new();
    let mut units_per_region: Vec<Vec<Val>> = Vec::with_capacity(num_of_regions);
    for region in &source.regions {
        let mut units: Vec<Val> = Vec::with_capacity(region.len());
        for operand in region {
            units.push(match operand {
                UnitOperand::Outside(unit) | UnitOperand::InsideLoopOther(unit) => *unit,
                UnitOperand::InsideLoop { unit, def } => {
                    let clone = values.clone_without_regions(def, &mut mapping);
                    let cloned_unit = results(&clone).first().copied().unwrap_or(*unit);
                    hoisted.push(clone);
                    cloned_unit
                }
            });
        }
        units_per_region.push(units);
    }

    // `:3913-3915` — one `index` result, then `:3916-3922` one block per region, each binding an
    // `index` argument and yielding it straight back.
    let result = values.mint();
    let regions: Vec<dfir_op::uniform::LocalRegion> = units_per_region
        .into_iter()
        .map(|units| {
            let arg = values.mint();
            dfir_op::uniform::LocalRegion {
                arg,
                units,
                body: vec![DfirOp::Uniform(dfir_op::uniform::Op::Yield {
                    operands: vec![arg],
                })],
            }
        })
        .collect();

    CreatedUniformize::Created(Box::new(Uniformized {
        op: DfirOp::Uniform(dfir_op::uniform::Op::UniformizeRegions {
            regions,
            results: vec![result],
        }),
        hoisted,
        // `:3923-3925`.
        active: num_of_regions > 1,
    }))
}

/// WHAT `checkBasicConditions` IS HANDED — any DataflowIR op, or the one SentientIR op its
/// `dyn_cast` chain also accepts.
///
/// ⭐ `sentient::ReceiveAndStoreOp` IS IN THAT CHAIN (`Helper.cpp:139-142`) because the pass revisits
/// ops it has already lowered, and the ONLY thing it reads off one is `hasAttr("marked")` — so that
/// bool is the whole arm. See [`Marked`] for where the attribute lives here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedOp<'a> {
    /// An op of the DataflowIR being lowered.
    Dfir(&'a DfirOp),
    /// An already-lowered `sentient.receive_and_store`, with whether it carries the mark.
    ReceiveAndStore {
        /// `recv_and_send_op->hasAttr("marked")`.
        marked: bool,
    },
}

/// THE OUTCOME OF [`check_basic_conditions`] — admissible, or WHICH of the reference's three
/// diagnostics refused the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum BasicConditions {
    /// `LogicalResult::success()` (`:190`).
    Admissible,
    /// An unmarked `sentient.receive_and_store` — a bare `failure()` with no diagnostic (`:139-140`).
    NotMarked,
    /// *"Only permutations are allowed in the load order for lowering into sentient"* (`:148-150`).
    DataOrderNotAPermutation,
    /// *"The load set needs to be hyper rectangular for lowering into sentient"* (`:160-162`).
    DataSetNotHyperRectangular,
    /// *"Only permutations are allowed in the time order for lowering into sentient"* (`:172-174`).
    TimeOrderNotAPermutation,
}

impl BasicConditions {
    /// `success()` only for [`Self::Admissible`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, BasicConditions::Admissible)
    }

    /// The diagnostic the C++ emits, verbatim (`Helper.cpp:148-174`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            BasicConditions::Admissible | BasicConditions::NotMarked => None,
            BasicConditions::DataOrderNotAPermutation => {
                Some("Only permutations are allowed in the load order for lowering into sentient")
            }
            BasicConditions::DataSetNotHyperRectangular => {
                Some("The load set needs to be hyper rectangular for lowering into sentient")
            }
            BasicConditions::TimeOrderNotAPermutation => {
                Some("Only permutations are allowed in the time order for lowering into sentient")
            }
        }
    }
}

/// Replaces: e210_checkBasicConditions
///
/// **210/384** `AgenToSentientLoweringPass::checkBasicConditions` —
/// `dcc/src/Conversion/AgenToSentient/Helper.cpp:58` (133L). Every transfer's data order must be a
/// permutation and its data set a hyper rectangle; a composite's TIME order must be a permutation too.
///
/// ⛔ TWELVE `dyn_cast` ARMS COLLECT THE SAME TWO LISTS, and the class only decides how many go in:
/// the load/store pair contribute one set and one order, the load-AND-store composites two of each.
/// ⛔ THE TIME-ORDER CHECK IS NOT FOR EVERY COMPOSITE (`:166-167`): its `isa<>` names composite load,
/// composite store and the two load-and-stores — the two INDIRECT single-ended composites set a
/// `time_order` and are never asked about it.
/// ⛔ AND THE TIME **SET** IS NOT CHECKED AT ALL — the reference's rectangularity test is commented out
/// under *"TODO find out why time_set is not rectangle."* (`:177-187`), so it is not ported.
/// ⛔ THE VECTOR PAIR'S SET AND ORDER ARE THE ISLAND'S DERIVED ONES ([`agen::access_set`],
/// [`agen::access_order`]), so both checks pass by construction for them; a composite carries its own.
/// ⛔ `signalPassFailure()` PRECEDES EVERY FAILURE AND IS NOT AN OUTCOME: the pass fails because the
/// caller propagates this answer.
pub fn check_basic_conditions(op: CheckedOp<'_>) -> BasicConditions {
    let mut data_sets: Vec<IntegerSet> = Vec::new();
    let mut data_orders: Vec<AffineMap> = Vec::new();
    let mut time_order: Option<&AffineMap> = None;

    match op {
        // `:137-141` — the one arm that answers on its own.
        CheckedOp::ReceiveAndStore { marked } => {
            return if marked {
                BasicConditions::Admissible
            } else {
                BasicConditions::NotMarked
            };
        }
        // `:64-65`, `:70-71` — one set and one order each.
        CheckedOp::Dfir(
            DfirOp::Agen(dfir_op::agen::Op::VectorLoad { view_ty, ty, .. })
            | DfirOp::Agen(dfir_op::agen::Op::VectorStore { view_ty, ty, .. }),
        ) => {
            data_sets.push(dfir_op::agen::access_set(view_ty, ty.len));
            data_orders.push(dfir_op::agen::access_order(view_ty.shape.len()));
        }
        // `:121-128` — `CompositeLoadAndStoreOp`: two of each, a time order, and NO region check.
        CheckedOp::Dfir(DfirOp::Agen(dfir_op::agen::Op::CompositeLoadAndStore(transfer))) => {
            data_sets.push(transfer.load_set.clone());
            data_sets.push(transfer.store_set.clone());
            data_orders.push(transfer.load_order.clone());
            data_orders.push(transfer.store_order.clone());
            time_order = Some(&transfer.time_order);
        }
        // Every other op falls out of the chain with both lists empty and is admissible (`:190`).
        CheckedOp::Dfir(_) => {}
    }

    // `:144-152` — `if (!inversePermutation(data_order))`.
    if data_orders
        .iter()
        .any(|data_order| data_order.inverse_permutation().is_none())
    {
        return BasicConditions::DataOrderNotAPermutation;
    }

    // `:154-164` — `constraints.isHyperRectangular(0, constraints.getNumCols() - 1)`.
    for data_set in &data_sets {
        let constraints = FlatConstraints::from_integer_set(data_set);
        if !constraints.is_hyper_rectangular(0, constraints.num_cols().saturating_sub(1)) {
            return BasicConditions::DataSetNotHyperRectangular;
        }
    }

    // `:166-175` — the composite time order, for the four classes named there.
    if let Some(time_order) = time_order
        && time_order.inverse_permutation().is_none()
    {
        return BasicConditions::TimeOrderNotAPermutation;
    }
    BasicConditions::Admissible
}

/// AN `agen.composite_memory_interleave`, AS THE CHECK READS IT — its `granularity` attribute and the
/// operations in its region.
///
/// ⭐ A STRUCT FOR THE SAME REASON AS [`IndirectMemView`]: the region it carries holds SentientIR
/// transfers by the time this runs, while the DataflowIR op
/// ([`agen::Op::CompositeMemoryInterleave`](crate::islands::dataflow_ir::dialects::agen::Op::CompositeMemoryInterleave))
/// holds DataflowIR ones — and what the check reads off it is exactly these two things.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInterleave<'a> {
    /// `getGranularity()` — [`None`] when the op carries no `granularity` attribute, which is the
    /// reference's `hasAttr` test (`:323`).
    pub granularity: Option<Elements>,
    /// The region's operations in order, terminator included.
    pub region: &'a [SenOp],
}

/// THE OUTCOME OF [`process_interleave_op`] — admissible, or WHICH of the reference's eight
/// diagnostics refused the op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum InterleaveCheck {
    /// `LogicalResult::success()` (`:385`).
    Admissible,
    /// *"only supported in L3 units"* (`:313-315`).
    NotAnL3Unit,
    /// *"illegal granularity setting"* (`:325-328`) — zero, or above the unit's burst size.
    IllegalGranularity,
    /// *"region does not contain enough operations"* (`:335-339`) — two transfers plus the yield.
    RegionTooSmall,
    /// *"invalid operation in region (no burst_size attribute)"* (`:344-348`).
    ///
    /// ⛔ THE FIRST REGION OP IS ASKED FOR ITS ATTRIBUTES BEFORE ANYTHING ASKS WHAT IT IS, which is
    /// why an op outside the transfer family lands here rather than on
    /// [`Self::InvalidOperationInRegion`].
    NoBurstSizeAttribute,
    /// *"invalid operation in region (no total_elements attribute)"* (`:352-356`).
    ///
    /// ⛔ STRUCTURALLY UNREACHABLE HERE: the two attributes are one [`sen::Extent`] on the island, so
    /// an op that carries `burst_size` carries `total_elements` with it.
    NoTotalElementsAttribute,
    /// *"invalid operation found in op region"* (`:364-368`) — not one of the three transfers.
    InvalidOperationInRegion,
    /// *"operation with different burst found"* (`:369-373`).
    DifferentBurst,
    /// *"operation with different total_elements found"* (`:374-378`).
    DifferentTotalElements,
    /// *"operation with unexpected operation type found"* (`:379-382`) — a region mixing two of the
    /// three transfer classes, which agree on burst and count.
    UnexpectedOperationType,
}

impl InterleaveCheck {
    /// `success()` only for [`Self::Admissible`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, InterleaveCheck::Admissible)
    }

    /// The diagnostic the C++ emits, verbatim (`Helper.cpp:315-381`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            InterleaveCheck::Admissible => None,
            InterleaveCheck::NotAnL3Unit => Some("only supported in L3 units"),
            InterleaveCheck::IllegalGranularity => Some("illegal granularity setting"),
            InterleaveCheck::RegionTooSmall => Some("region does not contain enough operations"),
            InterleaveCheck::NoBurstSizeAttribute => {
                Some("invalid operation in region (no burst_size attribute)")
            }
            InterleaveCheck::NoTotalElementsAttribute => {
                Some("invalid operation in region (no total_elements attribute)")
            }
            InterleaveCheck::InvalidOperationInRegion => {
                Some("invalid operation found in op region")
            }
            InterleaveCheck::DifferentBurst => Some("operation with different burst found"),
            InterleaveCheck::DifferentTotalElements => {
                Some("operation with different total_elements found")
            }
            InterleaveCheck::UnexpectedOperationType => {
                Some("operation with unexpected operation type found")
            }
        }
    }
}

/// Replaces: e211_processInterleaveOp
///
/// **211/384** `AgenToSentientLoweringPass::processInterleaveOp` —
/// `dcc/src/Conversion/AgenToSentient/Helper.cpp:305` (80L). An interleaved region is lowerable only
/// on an L3 unit, at a legal granularity, and only when every transfer in it is the SAME class with
/// the SAME burst and the SAME element count.
///
/// ⛔ THE GRANULARITY DEFAULT IS THE MAXIMUM AND IS NEVER CHECKED (`:322-323`): the bounds test runs
/// only when the attribute is present, so `l3BurstSize` itself is legal and 0 is not.
/// ⛔ `region_ops.size() < 3` COUNTS THE TERMINATOR (`:335`), so it means two transfers.
/// ⛔ THE IDENTITY TEST IS THE OP **NAME** (`:361`, `:379-381`), not the attributes: a region holding a
/// `load_and_send` beside a `receive_and_store` with matching burst and count is still refused.
/// ⛔ THE YIELD IS SKIPPED FIRST (`:363`) — it carries neither attribute and is not a transfer.
/// ⛔ `DT_CHECK_MSG(interleave_op, ...)` (`:308`) IS THE PARAMETER TYPE, and the unit walk that reaches
/// `comp` (`:311-312`) is the operand mechanism this port drops.
pub fn process_interleave_op<A: Arch>(
    comp: DfirUnit,
    interleave: &MemoryInterleave<'_>,
) -> InterleaveCheck {
    // `:313-316` — `is_any_of(comp, L3LU, L3SU)`.
    if !matches!(comp, DfirUnit::L3lu | DfirUnit::L3su) {
        return InterleaveCheck::NotAnL3Unit;
    }

    // `:320-329` — `sysDef.l3BurstSize`, and the attribute is bounded by it only when present.
    let max_burst = Elements(u64::from(A::L3_BURST));
    if let Some(granularity) = interleave.granularity
        && (granularity == Elements(0) || granularity > max_burst)
    {
        return InterleaveCheck::IllegalGranularity;
    }

    // `:334-339` — two transfers plus the yield.
    if interleave.region.len() < 3 {
        return InterleaveCheck::RegionTooSmall;
    }
    let Some(first) = interleave.region.first() else {
        return InterleaveCheck::RegionTooSmall;
    };
    // `:343-359` — the front operation's two attributes, in the reference's order.
    let Some(first_extent) = transfer_extent(first) else {
        return InterleaveCheck::NoBurstSizeAttribute;
    };

    for region_op in interleave.region {
        // `:363` — `if (isa<agen::YieldOp>(region_op)) continue;`.
        if matches!(region_op, SenOp::Agen(dfir_op::agen::Op::Yield)) {
            continue;
        }
        // `:364-368` — `isa<LoadAndSendOp, ReceiveAndStoreOp, LoadAndStoreOp>`.
        let Some(extent) = interleaved_transfer_extent(region_op) else {
            return InterleaveCheck::InvalidOperationInRegion;
        };
        if extent.burst_size != first_extent.burst_size {
            return InterleaveCheck::DifferentBurst;
        }
        if extent.total_elements != first_extent.total_elements {
            return InterleaveCheck::DifferentTotalElements;
        }
        // `:379-382` — `region_op.getName().getStringRef() != op_type`.
        if sentient_op_name(region_op) != sentient_op_name(first) {
            return InterleaveCheck::UnexpectedOperationType;
        }
    }
    InterleaveCheck::Admissible
}

/// THE `burst_size`/`total_elements` PAIR ANY OP CARRYING A [`sen::Extent`] HAS —
/// `first_op->hasAttr("burst_size")` (`Helper.cpp:344`) over the whole rung.
fn transfer_extent(op: &SenOp) -> Option<&sen::Extent> {
    match op {
        SenOp::Sentient(
            sen::Op::Load { extent, .. }
            | sen::Op::LoadAndSend { extent, .. }
            | sen::Op::ReceiveAndStore { extent, .. }
            | sen::Op::LoadAndStore { extent, .. },
        ) => Some(extent),
        _ => None,
    }
}

/// THE SAME PAIR, BUT ONLY FROM THE THREE CLASSES AN INTERLEAVED REGION MAY HOLD —
/// `isa<sentient::LoadAndSendOp, ReceiveAndStoreOp, LoadAndStoreOp>` (`Helper.cpp:364-365`).
///
/// ⛔ `sentient.load` IS DELIBERATELY OUT: it carries both attributes, so the reference reads them off
/// it as the front op and then refuses it in the loop.
fn interleaved_transfer_extent(op: &SenOp) -> Option<&sen::Extent> {
    match op {
        SenOp::Sentient(
            sen::Op::LoadAndSend { extent, .. }
            | sen::Op::ReceiveAndStore { extent, .. }
            | sen::Op::LoadAndStore { extent, .. },
        ) => Some(extent),
        _ => None,
    }
}

/// WHICH OF THE THREE TRANSFERS AN OP IS — the reference's `getName().getStringRef()` (`:361`),
/// narrowed to the classes that reach the comparison.
fn sentient_op_name(op: &SenOp) -> Option<&'static str> {
    match op {
        SenOp::Sentient(sen::Op::LoadAndSend { .. }) => Some("sentient.load_and_send"),
        SenOp::Sentient(sen::Op::ReceiveAndStore { .. }) => Some("sentient.receive_and_store"),
        SenOp::Sentient(sen::Op::LoadAndStore { .. }) => Some("sentient.load_and_store"),
        _ => None,
    }
}

/// WHAT `gatherAffineLoadStoreDetails` AND `constructImmutableAddress` READ OFF ONE ACCESS RECORD.
///
/// ⭐⭐ A TRAIT BECAUSE THE C++ IS A TEMPLATE, and it has exactly two instantiations —
/// `AccessDetailsAffine` and `AccessDetailsAffineComposite` (`Helper.cpp:2805`, `:2845`). The
/// precedent is `LoopBodyOp` in this module.
pub trait AccessRecord {
    /// `getMemoryIndex()` — [`None`] is the reference's `kMax`, i.e. never set.
    fn memory_index(&self) -> Option<MemoryOperandIndex>;
    /// `getMemViewStartAddr()` — [`None`] is a null `Value`, i.e. no view resolved.
    fn mem_view_start_addr(&self) -> Option<Val>;
    /// `getIndicesCoeffDict()`.
    fn indices_coeff_dict(&self) -> &IndicesCoeffDict;
}

impl AccessRecord for AccessDetailsAffine<'_> {
    fn memory_index(&self) -> Option<MemoryOperandIndex> {
        self.base.memory_index
    }
    fn mem_view_start_addr(&self) -> Option<Val> {
        self.base.mem_view_start_addr
    }
    fn indices_coeff_dict(&self) -> &IndicesCoeffDict {
        &self.indices_coeff_dict
    }
}

impl AccessRecord for AccessDetailsAffineComposite<'_> {
    fn memory_index(&self) -> Option<MemoryOperandIndex> {
        self.affine.base.memory_index
    }
    fn mem_view_start_addr(&self) -> Option<Val> {
        self.affine.base.mem_view_start_addr
    }
    fn indices_coeff_dict(&self) -> &IndicesCoeffDict {
        &self.affine.indices_coeff_dict
    }
}

/// EVERY LOOP ITERATOR ACROSS ALL THE ACCESS RECORDS, WITH ONE COEFFICIENT PER RECORD — the
/// reference's `llvm::DenseMap<Value, std::vector<int64_t>> indices_coeff_dict` (`Helper.cpp:553`).
///
/// ⛔ EVERY ROW IS AS LONG AS THERE ARE RECORDS, zero-filled on first sight (`:559-564`), *"to make
/// lowering simple"* — an iterator only one access uses still has a column for the other.
/// ⛔ AND THE CONSTANT ROW IS ONE OF THEM. The reference keys it under `nullptr`
/// ([`IndicesCoeffDict::constant`]), and entry 357 is what drops it again by skipping the null key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatheredCoefficients {
    /// One row per distinct iterator, in first-seen order.
    pub per_index: Vec<(Val, Vec<i64>)>,
    /// The constant offsets, one per record.
    pub constant: Vec<i64>,
}

/// THE OUTCOME OF [`gather_affine_load_store_details`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum GatheredDetails {
    /// `success()` (`:615`).
    Gathered,
    /// *"Unable to generate address manipulation statements"* (`:601-602`) — entry 357's failure.
    AddressManipulationFailed,
    /// *"Unable to construct immutable addresses"* (`:612`) — entry 213's failure.
    ImmutableAddressesFailed,
}

impl GatheredDetails {
    /// `success()` only for [`Self::Gathered`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, GatheredDetails::Gathered)
    }

    /// The diagnostic the C++ emits, verbatim (`Helper.cpp:601`, `:612`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            GatheredDetails::Gathered => None,
            GatheredDetails::AddressManipulationFailed => {
                Some("Unable to generate address manipulation statements")
            }
            GatheredDetails::ImmutableAddressesFailed => {
                Some("Unable to construct immutable addresses")
            }
        }
    }
}

/// Replaces: e212_gatherAffineLoadStoreDetails
///
/// **212/384** `AgenToSentientLoweringPass::gatherAffineLoadStoreDetails` —
/// `dcc/src/Conversion/AgenToSentient/Helper.cpp:538` (74L). Mark the op, transpose every record's
/// iterator coefficients into one table per iterator, collect the view start addresses under their
/// operand, and hand both on to the address manipulation and the immutable addresses.
///
/// ⛔ THE START ADDRESS GOES INTO **TWO** CONTAINERS (`:568-570`): `mutable_addrs`, which entry 357
/// then rewrites, and a local copy that entry 213 reads as `updated_mem_view_start_addrs`.
/// ⛔ THE ROWS ARE ZERO-FILLED IN A FIRST PASS AND WRITTEN IN A SECOND (`:559-572`), which is what
/// makes column `i` mean "record `i`" even for an iterator the earlier records never mentioned.
/// ⛔ `insert` ABORTS ON A FILLED SLOT; here the slot is a capability ([`AccessContainer::vacancy`]), so
/// a record whose operand is already taken — or which never resolved a view — is passed over.
/// ⛔ THE `LLVM_DEBUG` BLOCK IS DROPPED (`:574-589`), and with it the only call to
/// [`loop_nest_level`] in this function.
pub fn gather_affine_load_store_details<T: AccessRecord>(
    position: usize,
    marked: &mut Marked,
    comp: DfirUnit,
    access_details: &AccessContainer<T>,
    mutable_addrs: &mut AccessContainer<Val>,
    immutable_addrs: &mut AccessContainer<Val>,
) -> GatheredDetails {
    // `:546` — the mark this pass recovers the op by.
    marked.mark(position);

    let records = access_details.entries();
    let mut coefficients = GatheredCoefficients {
        per_index: Vec::new(),
        constant: vec![0; records.len()],
    };
    let mut mem_view_start_addrs = AccessContainer::<Val>::default();

    for (i, record) in records.iter().enumerate() {
        let dict = record.indices_coeff_dict();
        for (index, coefficient) in &dict.per_index {
            // `:560-564` — the row appears the first time its iterator does, zero-filled.
            let row = match coefficients
                .per_index
                .iter_mut()
                .find(|(known, _)| known == index)
            {
                Some((_, row)) => row,
                None => {
                    coefficients
                        .per_index
                        .push((*index, vec![0; records.len()]));
                    match coefficients.per_index.last_mut() {
                        Some((_, row)) => row,
                        None => continue,
                    }
                }
            };
            // `:571-572` — column `i` is record `i`.
            if let Some(slot) = row.get_mut(i) {
                *slot = *coefficient;
            }
        }
        if let Some(slot) = coefficients.constant.get_mut(i) {
            *slot = dict.constant;
        }

        // `:568-570` — the same address under the same operand, in both containers.
        if let Some(moi) = record.memory_index()
            && let Some(start_addr) = record.mem_view_start_addr()
        {
            if let Some(slot) = mutable_addrs.vacancy(moi) {
                slot.fill(start_addr);
            }
            if let Some(slot) = mem_view_start_addrs.vacancy(moi) {
                slot.fill(start_addr);
            }
        }
    }

    // `generateAffineAddressManipulationStmts(...)` (`:597-603`) — entry 357/384, unported.
    //
    // ⛔ THE GATE IS AN INPUT ON WHICH IT PROVABLY WRITES NOTHING: with no access details its
    // `DT_CHECK` on the three sizes holds trivially (`:631-632`), the coefficient table is empty so it
    // takes the `else` at `:761`, whose loop over `mutable_addrs_base` — empty too — runs zero times,
    // and it returns `success()` (`:827`). ⛔ THE `:698` LOOP IS THE OTHER BRANCH's AND IS NOT REACHED,
    // and this is not the only such input: a non-L3 unit whose constant row is all zeros takes neither
    // arm at `:788`/`:795` and writes nothing either.
    if !records.is_empty() {
        todo!(
            "e357_generateAffineAddressManipulationStmts not ported: {} access record(s) need their \
             mutable address registers assigned and advanced across the loop nest \
             (`Helper.cpp:625`)",
            records.len()
        );
    }

    // `:610-613` — entry 213, on the local copy entry 357 has just updated.
    if !construct_immutable_address(comp, access_details, &mem_view_start_addrs, immutable_addrs)
        .admissible()
    {
        return GatheredDetails::ImmutableAddressesFailed;
    }
    GatheredDetails::Gathered
}

/// THE OUTCOME OF [`construct_immutable_address`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum ImmutableAddresses {
    /// `success()` (`:1235`) — every record's immutable address is placed.
    Constructed,
    /// `failure()` with no diagnostic (`:1221-1223`): the two containers hold different counts.
    SizeMismatch,
}

impl ImmutableAddresses {
    /// `success()` only for [`Self::Constructed`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, ImmutableAddresses::Constructed)
    }
}

/// Replaces: e213_constructImmutableAddress
///
/// **213/384** `AgenToSentientLoweringPass::constructImmutableAddress` —
/// `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217` (16L). On L3 the immutable address is the
/// memory view's own start address; everywhere else it is the one the address manipulation left.
///
/// ⭐ THE REASON IS THE REFERENCE'S OWN (`:1212-1215`): an L3 offset is handled in an LBR/EBR, so the
/// register keeps the view's start and the offset rides elsewhere.
/// ⛔ THE L3 ARM READS THE RECORD, THE OTHER ARM READS BY **POSITION** — `updated_mem_view_start_addrs[i]`
/// (`:1231`) is the i-th entry in insertion order, not `get(moi)`, and the two orders agree only
/// because entry 212 filled both containers from the same walk.
/// ⛔ THE SIZE TEST IS WHY THAT INDEX IS IN RANGE (`:1221`), and it is a bare `failure()` with no
/// message.
pub fn construct_immutable_address<T: AccessRecord>(
    comp: DfirUnit,
    access_details: &AccessContainer<T>,
    updated_mem_view_start_addrs: &AccessContainer<Val>,
    immutable_addrs: &mut AccessContainer<Val>,
) -> ImmutableAddresses {
    let records = access_details.entries();
    let updated = updated_mem_view_start_addrs.entries();
    if records.len() != updated.len() {
        return ImmutableAddresses::SizeMismatch;
    }

    for (i, record) in records.iter().enumerate() {
        // `:1226-1232` — `is_any_of(comp, L3LU, L3SU)`.
        let address = if matches!(comp, DfirUnit::L3lu | DfirUnit::L3su) {
            record.mem_view_start_addr()
        } else {
            updated.get(i).copied()
        };
        if let Some(moi) = record.memory_index()
            && let Some(address) = address
            && let Some(slot) = immutable_addrs.vacancy(moi)
        {
            slot.fill(address);
        }
    }
    ImmutableAddresses::Constructed
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// 374/384
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e374_lowerSymbolicVectorLoadOp
///
/// **374/384** `AgenToSentientLoweringPass::lowerSymbolicVectorLoadOp` —
/// `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380` (26L). A symbolic load, and the symbolic store
/// it feeds when it feeds one, lowered as one load-and-send pair.
///
/// ⭐ THE STORE IS FOUND TWICE ON PURPOSE (`:3384`, then `:3399-3402`): entry 360 may CLONE the ops it
/// gathers, so the second read takes the store out of `kDirDst` rather than trusting the first.
/// ⛔ `has(kDirDst)` IS THE PATTERN TEST AFTER THE FACT — a load with no store reaches
/// `lowerVectorLoadHelper` with a null `store_op`, which is the plain-load half of that helper.
/// ⛔ THE RETURN TYPE IS `!` BECAUSE NO OUTCOME EXISTS YET. `LogicalResult` has no counterpart here
/// (the crate forbids `Result`) and the statement count its caller advances by cannot be answered
/// before entry 360 says whether the store came with it. It becomes a real type with 360.
pub fn lower_symbolic_vector_load_op<A: Arch>(
    op: &DfirOp,
    unit: &ProgramUnit<A>,
    comp: DfirUnit,
    scope: &[DfirOp],
) -> ! {
    // `:3383-3384` — entry 036 at its `SymbolicVectorStoreOp` instantiation.
    let store_op = store_op_from_load_store_pattern(AgenOpKind::SymbolicVectorStore, op, scope);

    // `:3386-3391` — `constructSymbolicDetailsAndAddrs` fills all three containers, and every line
    // below reads one of them: `:3393-3394` takes the candidate out of `kDirSrc`, `:3399` asks
    // `has(kDirDst)`, and `:3405-3407` hands all three to entry 217.
    todo!(
        "e360_constructSymbolicDetailsAndAddrs is unported, so e217_lowerVectorLoadHelper cannot \
         lower the agen.symbolic_vector_load on {:?} of {:?} (store {:?})",
        comp,
        unit.on.kind(),
        store_op
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// 375/384
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e375_lowerSymbolicVectorStoreOp
///
/// **375/384** `AgenToSentientLoweringPass::lowerSymbolicVectorStoreOp` —
/// `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410` (30L). A symbolic store on its own, lowered to
/// one `sentient.receive_and_store`.
///
/// ⛔ IT PASSES `nullptr` FOR THE STORE (`:3415`), which is what makes entry 360 gather ONE record —
/// hence `DT_CHECK(size == 1)` on all three containers (`:3419-3420`) and the `[0]` indexing at
/// `:3429-3430`.
/// ⛔ THE ELEMENT TYPE COMES FROM THE **MEMREF**, not from the stored vector (`:3427`), unlike
/// `AccessDetailsSymbolic::initialize`, which takes the WIDTH from the value
/// (`AccessDetails.cpp:879-880`).
/// ⛔ AND THE DELETE LIST GETS TWO THINGS: the store itself (`:3436`) and, through entry 270, the op
/// that produced the value it stored (`:3438-3439`).
/// ⛔ `!` FOR THE SAME REASON AS ENTRY 374 — see its note.
pub fn lower_symbolic_vector_store_op<A: Arch>(unit: &ProgramUnit<A>, comp: DfirUnit) -> ! {
    // `:3413-3418` — `constructSymbolicDetailsAndAddrs(op, nullptr, ...)`. The single record it
    // gathers is the only route to `access_details[0]`, `mutable_addrs[0]` and `immutable_addrs[0]`,
    // which are exactly the three arguments entry 028 needs at `:3428-3431`.
    todo!(
        "e360_constructSymbolicDetailsAndAddrs is unported, so e028_constructReceiveAndStoreStmt \
         cannot lower the agen.symbolic_vector_store on {:?} of {:?} (e270_addStoreInputToDeleteList \
         is unported too)",
        comp,
        unit.on.kind()
    );
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e214_setImmutableAddrAndIncrements
// crustify:todo: e215_setsttype
// crustify:todo: e216_constructReceiveAndExtractScalarOp
// crustify:todo: e217_lowerVectorLoadHelper
// crustify:todo: e218_lowerSetTransferMaskStateOp
// crustify:todo: e219_cloneStartAddrOutsideLoop
// crustify:todo: e220_cleanupTriviallyRedundantSetSendDestination
// crustify:todo: e267_constructTimeLoopsAndVectorOperations
// crustify:todo: e268_constructLoadAndStoreStmt
// crustify:todo: e269_constructLoadAndExtractScalarOp
// crustify:todo: e270_addStoreInputToDeleteList
// crustify:todo: e271_lowerCompositeMemoryInterleaveOp
// crustify:todo: e272_insertInitializationStmt
// crustify:todo: e298_constructAffineDetailsAndAddrs
// crustify:todo: e299_lowerAffineCompositeHelper
// crustify:todo: e311_constructAffineCompDetailsAndAddrs
// crustify:todo: e312_lowerExtractVectorLoadOp
// crustify:todo: e313_lowerExtractVectorStoreOp
// crustify:todo: e314_lowerVectorLoadOp
// crustify:todo: e315_lowerVectorStoreOp
// crustify:todo: e316_lowerIndirectVectorLoadOp
// crustify:todo: e317_lowerIndirectVectorStoreOp
// crustify:todo: e318_lowerLDCVTIPattern
// crustify:todo: e327_gatherSymbolicLoadStoreDetails
// crustify:todo: e328_adjustMutableAddrInitForIndirect
// crustify:todo: e329_lowerCompositeLoadOp
// crustify:todo: e330_lowerCompositeStoreOp
// crustify:todo: e331_lowerCompositeLoadAndStoreOp
// crustify:todo: e332_lowerCompositeIndirectLoadOp
// crustify:todo: e333_lowerCompositeIndirectStoreOp
// crustify:todo: e334_lowerCompositeIndirectLoadAndStoreOp
// crustify:todo: e335_insertCopyAndAddStmts
// crustify:todo: e336_adjustMutableAddrInitForStride
// crustify:todo: e357_generateAffineAddressManipulationStmts
// crustify:todo: e358_constructLoadAndSendStmt
// crustify:todo: e359_constructReceiveAndStoreStmt
// crustify:todo: e360_constructSymbolicDetailsAndAddrs
