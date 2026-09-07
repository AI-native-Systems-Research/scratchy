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

//! `VectorOperands.cpp` — 17 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e071_getOperandFromReceiveOp` | 071/384 | 54 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34` |
//! | `e072_getOperandFromSendOp` | 072/384 | 50 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95` |
//! | `e073_constValToField` | 073/384 | 12 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250` |
//! | `e074_sameBlock` | 074/384 | 11 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652` |
//! | `e075_eraseOp` | 075/384 | 16 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806` |
//! | `e166_getOperandFromConstantOp` | 166/384 | 26 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267` |
//! | `e167_getOperandFromConstantBitstreamOp` | 167/384 | 5 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299` |
//! | `e168_getOperandFromNegOp` | 168/384 | 5 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366` |
//! | `e169_getName` | 169/384 | 11 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866` |
//! | `e170_getLayoutMapAndIndices` | 170/384 | 40 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879` |
//! | `e232_getOperandFromLoadOrStoreOp` | 232/384 | 94 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153` |
//! | `e233_eraseOperands` | 233/384 | 46 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690` |
//! | `e234_setValue` | 234/384 | 3 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72` |
//! | `e278_getOperandFromShuffleOp` | 278/384 | 36 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311` |
//! | `e304_getOperandWithPrecision` | 304/384 | 254 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389` |
//! | `e320_getOperand` | 320/384 | 4 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378` |
//! | `e343_getOperandFromCastOp` | 343/384 | 5 | `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354` |
//!
//! Original files homed here: `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp`, `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp`


use crate::islands::sentient::dialects::sentient as sen;

/// AN OPERATION'S IDENTITY — the stand-in for `mlir::Operation *`.
///
/// ⭐⭐ THE VALUE IS ITS PLACE IN THE REGION TREE, NOT A POINTER AND NOT A COUNTER. The reference
/// keys `OperandReuse::data_origins_` by `Operation *` and asks `DominanceInfo` whether one op
/// dominates another; both questions are about WHERE the op sits, so the identity carries the
/// position and both answers fall out of it. A flat index would answer the first and lose the
/// second the moment a loop body appears — an op inside `affine.for` #1 comes *later* in a flat
/// walk than one inside `affine.for` #0 and dominates neither.
///
/// ⭐ ONE ORDINAL PER REGION LEVEL, OUTERMOST FIRST. `[3]` is the fourth op of the program unit's
/// body; `[3, 0]` is the first op of that op's region. `mlir::Operation *` is a pointer, so nothing
/// in the C++ names this structure — but every use of it in `OperandReuse` is one of the two
/// questions above.
///
/// ⛔ NOT AN EXTENT. The ordinals index positions within a block; they are never lane counts,
/// addresses or bounds, and nothing here does arithmetic on them beyond comparing two at the same
/// level.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId {
    /// The ordinals, outermost first.
    path: Vec<u32>,
}

impl OpId {
    /// THE OP AT THIS PATH.
    #[must_use]
    pub fn at(path: &[u32]) -> OpId {
        OpId {
            path: path.to_vec(),
        }
    }

    /// ITS PATH, OUTERMOST FIRST.
    #[must_use]
    pub fn path(&self) -> &[u32] {
        &self.path
    }
}

/// WHERE AN OPERAND COMES FROM — `VectorOperandType` (`VectorOperands.hpp:28-36`).
///
/// ⛔ EIGHT CASES AND NO NINTH. The reference switches on this to decide whether an operand is a
/// register file, a link, an immediate or the internal state a compare/select forwards, and
/// `OperandReuse::setReuseInformation` treats `Constant` and `LRF` specially by name
/// (`OperandReuse.cpp:26,36`) — so a wildcard here would silently absorb a new source into the
/// wrong rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VectorOperandType {
    /// A value arriving over a link — a `dataflow.send`/`receive` pair's end.
    Link,
    /// The local register file.
    Lrf,
    /// The indirect register file.
    Irf,
    /// The cross register file.
    Xrf,
    /// An immediate.
    Constant,
    /// A `vectorchain.constant_bitstream`.
    ConstantBitstream,
    /// A neighbour forward.
    Nfwd,
    /// ⭐ THE INTERNAL STATE a `SELECT`/`FCMP`/`FMINMAX` forwards — the C++ says so in a trailing
    /// comment on the enumerator itself (`VectorOperands.hpp:35`).
    IState,
}

/// ONE OPERAND OF A COMPUTE, AS THE VECTORCHAIN LOWERING SEES IT — `VectorOperand`
/// (`VectorOperands.hpp:38-113`).
///
/// # ⚠️ PARTIAL BY DESIGN — three of the six data members are not here yet
///
/// ⭐ THE MEMBERS ARRIVE WITH THE UNITS THAT READ THEM. `values_` (a uniformized value per
/// core/corelet/fold) and `splat_` are only ever touched by `e169_getName`, `e234_setValue` and
/// `e304_getOperandWithPrecision` — none of which is scheduled in this wave — and `values_` in
/// particular holds a mix of a compute-port name (`symbolizeSentientComputePort` consumes it at
/// `VectorChainToSentientPESFP.cpp:722-726`), the literal `"latch"`, and a slice index printed as
/// decimal (`VectorOperands.cpp:240`). Choosing between one enum, three fields and an index newtype
/// is a decision that belongs to whoever ports those units against their own callers, not a guess
/// made here for a field this wave never reads.
///
/// ⛔ THE TWO PRECISIONS ARE `Option`, AND THE EMPTY STRING IS WHY. The constructor
/// (`VectorOperands.hpp:76-79`) sets only `type_`, `op_` and the value, so both precisions start
/// EMPTY, and `getInputPrecisionFromOperand`'s absent overload returns `""` for a missing operand
/// (`VectorChainHelper.cpp:43-49`). The consumers test for it by name —
/// `if (result_forwarding.empty() && result_precision == "") result_precision = compute_precision;`
/// (`VectorChainToSentientPESFP.cpp:257-263` and again at `:1131-1136`) — so "unset" is a value this
/// type has to be able to hold, and `Precision::None` is NOT it (that one spells `none` on the wire).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorOperand {
    /// `type_` — where it comes from.
    pub kind: VectorOperandType,
    /// `op_` — the operation that produced it.
    pub op: OpId,
    /// `orig_precision_` — the element precision of the value as it was produced. `None` is the
    /// reference's empty string; see the type's note.
    pub orig_precision: Option<sen::Precision>,
    /// `on_the_fly_conv_precision_` — the precision it is converted to on the way in, which starts
    /// equal to [`Self::orig_precision`] (`VectorOperands.cpp:399-400`) and only differs where a
    /// `vectorchain.cast` folded into the operand.
    pub on_the_fly_conv_precision: Option<sen::Precision>,
}
