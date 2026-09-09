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

use crate::arch::{Arch, IsaGen};
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{
    Index, Op as DfirOp, Val, operands, regions, regions_mut, results, uses,
};
use crate::islands::dataflow_ir::dialects::{agen, arith, dataflow, uniform, vector};
use crate::islands::dataflow_ir::link;
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

use super::vc_vector_chain_helper::precision_in_string;

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

    /// THE BLOCK IT SITS IN — `Operation::getBlock()`.
    ///
    /// ⭐⭐ A BLOCK IS A PATH PREFIX. A position is its enclosing block's position followed by the
    /// op's own ordinal, so dropping the last ordinal names the block, and two ops are in the same
    /// block exactly when their prefixes are equal. Every top-level op of a program unit's body has
    /// the EMPTY prefix, which is that body's single block — the answer `sameBlock`
    /// (`VectorOperands.cpp:652`) needs for the common case of a compute and its operands sitting
    /// side by side.
    ///
    /// ⛔ IT IS THE PARENT BLOCK, NOT THE PARENT OP. `[3, 1]`'s block is `[3]`, which is the
    /// position of the op OWNING that block; the reference's `getBlock()` returns the block and
    /// `getParentOp()` the op, and this crate's positions cannot tell one from the other. Nothing
    /// here needs to — the only use is the equality above.
    ///
    /// ⛔ A MULTI-REGION OP'S TWO BLOCKS ARE ONE PREFIX HERE, which is why [`op_at`] flattens
    /// regions: `scf.if`'s `then` and `else` bodies both answer `[3]`. ⛔ AND THIS COMPILER DOES EMIT
    /// `scf.if` — [`Condition::wrap`](super::tf_transform_paged_mem_view_impl::Condition::wrap) builds
    /// one per page guard — but every one it builds is ONE-ARMED, its `else_body` an empty `Vec`,
    /// which in [`scf::Op::If`](crate::islands::dataflow_ir::dialects::scf::Op::If) means *no block*
    /// at all. One non-empty region has nothing to conflate. A two-armed `scf.if` or `affine.if`
    /// arriving on the input side would conflate, and
    /// [`OperandReuse::dominates`](super::vc_operand_reuse::OperandReuse::dominates) says what that
    /// would and would not cost.
    #[must_use]
    pub fn block(&self) -> &[u32] {
        &self.path[..self.path.len().saturating_sub(1)]
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
/// # ⭐ `splat_` ARRIVED WITH ENTRY 304, WHICH IS ITS ONLY WRITER
///
/// `splat_` is *"to capture the select semantics"* (`VectorOperands.hpp:70`), and
/// [`VectorOperand::with_precision`] writes exactly one value into it — `"east"`, in its `SelectOp`
/// arm (`VectorOperands.cpp:582`). Its only reader is `e340_analyzeAndFillOperandForwarding`, which
/// hands it to `symbolizeSentientComputePort` and keeps the answer as a `SentientComputePortAttr`
/// (`VectorChainHelper.cpp:544-547`) — so the field is an `Option<`[`sen::Port`]`>` rather than a
/// string, and `None` is the empty string that reader tests against.
///
/// ⭐ `values_` IS HERE NOW, because `e071_getOperandFromReceiveOp` and `e072_getOperandFromSendOp`
/// exist to WRITE it — the link a compute reads over is the operand's value and nothing else. See
/// [`Self::values`] and [`OperandValue`] for what one entry holds and what it deliberately does not
/// yet spell.
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
    /// `values_` — ⭐ THE OPERAND'S VALUE, ONE ENTRY PER UNIFORMIZED CORE/CORELET/FOLD.
    ///
    /// The C++ is a `std::vector<std::string>` *"to handle uniformized values across
    /// cores/corelets/folds"* (`VectorOperands.hpp:44-46`), and the three-argument constructor pushes
    /// exactly one through `setValue` (`:72-79`) — which is what [`VectorOperand::new`] does.
    ///
    /// ⛔ A LIST AND NOT ONE VALUE, even though every constructor in the file writes exactly one.
    /// `setValue` *clears* before pushing, so the vector is the member's own shape and a later unit
    /// filling one entry per fold is not a change to this type.
    pub values: Vec<OperandValue>,
    /// `orig_precision_` — the element precision of the value as it was produced. `None` is the
    /// reference's empty string; see the type's note.
    pub orig_precision: Option<sen::Precision>,
    /// `on_the_fly_conv_precision_` — the precision it is converted to on the way in, which starts
    /// equal to [`Self::orig_precision`] (`VectorOperands.cpp:399-400`) and only differs where a
    /// `vectorchain.cast` folded into the operand.
    pub on_the_fly_conv_precision: Option<sen::Precision>,
    /// `splat_` — see the type's own note. [`sen::Port::East`] where a `vectorchain.select` folded
    /// into the operand from below, `None` everywhere else.
    pub splat: Option<sen::Port>,
}

/// ONE ENTRY OF `values_` — an operand's value (`VectorOperands.hpp:44-46`).
///
/// # ⚠️ THREE CASES, ONE PER THING THE C++ STRING HOLDS
///
/// ⛔⛔ THE C++ STRING HOLDS AT LEAST THREE DIFFERENT THINGS, AND SPELLING THEM AS ONE WOULD BE WRONG
/// THREE WAYS OVER. `getName` (`VectorOperands.cpp:866-877`) shows the split by `type_`:
///
/// - a `LINK` operand's value is a **compute port**, handed straight to
///   `symbolizeSentientComputePort` (`VectorChainToSentientPESFP.cpp:722-726`) — this variant, and
///   what `e071_getOperandFromReceiveOp` and `e072_getOperandFromSendOp` write;
/// - an `LRF`/`IRF`/`ISTATE` operand's value is a **decimal slice index**, computed as
///   `(start_address + layout_map.getSingleConstantResult()) * bit_width / 1024` and printed with
///   `std::to_string` (`VectorOperands.cpp:222-241`), which `getName` prefixes with `lrf`/`irf`/
///   `istate` — [`Self::Slice`], which arrives with `e169_getName` because that is the unit that
///   READS the prefix decision. `e232_getOperandFromLoadOrStoreOp` is what will WRITE it;
/// - a `CONSTANT` operand's value comes from `constValToField` (`VectorOperands.cpp:250`), which
///   entry 073 ported in this same file — and its answer is a [`sen::Port`] too, one of the four
///   pseudo-units, so [`Self::Port`] already covers that case. See [`const_val_to_field`].
///
/// ⭐ SO THE ENUM STATES WHAT IT COVERS AND A NEW CASE IS AN ADDITION RATHER THAN A REINTERPRETATION.
/// A `Vec<sen::Port>` would have to be replaced outright by the second unit that touches the field; a
/// `Vec<String>` would put a closed set back into a string, which this crate does not do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperandValue {
    /// A COMPUTE PORT — the link a value arrives over or leaves by.
    Port(sen::Port),
    /// A REGISTER-FILE SLICE — the index `getName` prefixes with its file's name.
    Slice(RegisterSlice),
    /// A RAW IMMEDIATE — `std::to_string(const_val)` on a constant bitstream the caller did NOT ask
    /// to read as a splatted vector (`VectorOperands.cpp:302-303`).
    ///
    /// ⛔⛔ THIS ONE HAS NO COMPUTE-PORT SPELLING AND THAT IS THE REFERENCE'S OWN GAP, not a
    /// restriction added here. `SentientComputePort` is a closed sixty-three-case enum
    /// (`SentientTypes.td:98-160`) with no decimal case, so `symbolizeSentientComputePort("7")`
    /// answers `std::nullopt` and every one of `getName`'s call sites then calls `.value()` on it.
    /// [`VectorOperand::name`] answers `None` here, which is that `nullopt` and nothing more.
    ///
    /// ⭐ THE NUMBER IS STILL READ, JUST NOT THROUGH THE VALUE. `Splat::createSentientConstants`
    /// re-reads the immediate off `op_`'s own `vectorchain.constant_bitstream`
    /// (`VectorChainToSentientPESFP/Splat.cpp:41-42`, inside `createSentientConstants` at `:34-67`),
    /// which is why a literal never has to become a port.
    Literal(i64),
}

/// WHICH REGISTER FILE AND WHICH SLICE OF IT — one entry of `values_` for an `LRF`, `IRF` or
/// `ISTATE` operand (`VectorOperands.cpp:222-241`).
///
/// ⛔⛔ THE FILE IS PART OF THE VALUE HERE AND IN THE REFERENCE IT IS NOT — the C++ value is the bare
/// decimal and `getName` reads the file off `type_`. **THE BOUND IS WHY.** A slice index is
/// `(start_address + layout_map.getSingleConstantResult()) * bit_width / 1024`, and how large it may
/// be is a property of the FILE: thirty-two for the LRF ([`sen::LrfIndex`]), two for the IRF
/// ([`IrfIndex`]), four for the state file ([`sen::IStateIndex`]). A single unbounded `SliceIndex`
/// would have to be narrowed at [`VectorOperand::name`] instead, and a checked narrowing there is a
/// runtime refusal — precisely the one `LrfIndex` was rewritten as thirty-two variants to delete
/// (see its note). Minting the file and its bounded index together is what makes `name` total.
///
/// ⭐ AND THE REDUNDANCY WITH [`VectorOperandType`] IS LOAD-BEARING, NOT AN OVERSIGHT. See
/// [`VectorOperand::name`]: the reference reads the file from `type_` and the index from the value,
/// and that is exactly how it produces `"lrflatch"` for an operand one of
/// `OperandReuse.cpp:28`, `:30`, `:32`, `:40` or `:43` re-valued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterSlice {
    /// A slice of the local register file — `getName` spells it `lrf<n>`.
    Lrf(sen::LrfIndex),
    /// A slice of the indirect register file — `irf<n>`.
    Irf(IrfIndex),
    /// A slice of the state file — `istate<n>`.
    IState(sen::IStateIndex),
}

/// WHICH `irf<n>` — ⛔ TWO, BECAUSE `SentientTypes.td:108-109` DECLARES TWO.
///
/// ⛔ THERE IS NO `sen::IrfIndex` TO REUSE, and that is because [`sen::Port`] spells the two files as
/// separate cases ([`sen::Port::Irf0`], [`sen::Port::Irf1`]) rather than as an indexed one — the same
/// shape the `.td` has. This type is the *operand side* of that pair, so that a slice can be carried
/// before it is turned into a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IrfIndex {
    /// `irf0`.
    I0,
    /// `irf1`.
    I1,
}

/// WHICH COMPUTE UNIT IS ASKING — the `comp` argument of `getOperandFromReceiveOp` and
/// `getOperandFromSendOp` (`VectorOperands.cpp:36`, `:97`).
///
/// # 🛑 THREE OF `SenComponents`' FIFTY, AND THE FUNCTIONS' OWN STRUCTURE PROVES IT
///
/// ⛔⛔ BOTH FUNCTIONS ARE WRITTEN AS `if (comp == PT) { … } else { /* PE/SFP */ … }` — the `else`
/// carries that comment in the reference itself (`VectorOperands.cpp:64`, `:113`) and its body tests
/// `comp == SFP` to decide the SFP ring. A fourth component reaching either of them would silently
/// take the PE/SFP branch and be told to send over `pt`.
///
/// ⭐ AND THE CALL SITES AGREE. `getOperandWithPrecision` reaches these two from
/// `VectorOperands.cpp:433` and `:443`, inside the vectorchain lowering that runs once for the PT
/// (`VectorChainToSentientPT.cpp`) and once for `is_any_of(unit_comp, PE, SFP)`
/// (`VectorChainToSentientPESFP.cpp:1385-1389`). Nothing else asks.
///
/// ⛔ SO IT IS ITS OWN THREE-CASE TYPE AND NOT [`crate::islands::dataflow_ir::ty::GenericComp`]:
/// eighteen components can be a *peer* of one of these operations, and only three can be the one
/// asking. Passing the peer where the asker goes must be an E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeComp {
    /// `PT` — the matrix unit, whose links are compass directions.
    Pt,
    /// `PE`.
    Pe,
    /// `SFP`.
    Sfp,
}

/// `dcc_ext_ctx.getArch() >= RCUDD1A_ISA` (`VectorOperands.cpp:70`, `:127`).
///
/// ⛔⛔ TRUE ON EVERY ARCH THIS CRATE BUILDS FOR, AND THAT IS A COMPILE-TIME FACT RATHER THAN AN
/// ASSUMPTION. `IsaCoreGen` is ordered `MPW2 < MPW3 < MPW4 < RCUDD1A < SEN1P5` with
/// `DEFAULT_ISA = RCUDD1A_ISA` (`sys-arch-spec/isa/isa.hpp:24-35`), and [`IsaGen`] models the last
/// two only. The match is exhaustive, so adding an older generation to [`IsaGen`] stops the build
/// here instead of silently answering `true` for it.
const fn supports_sfp_ring(isa: IsaGen) -> bool {
    match isa {
        IsaGen::Rcudd1a | IsaGen::Sen1p5 => true,
    }
}

impl VectorOperand {
    /// AN OPERAND OF ONE KIND, CARRYING ONE VALUE — the three-argument constructor
    /// (`VectorOperands.hpp:76-79`).
    ///
    /// ⭐ THE TWO PRECISIONS START UNSET, because the constructor initialises only `type_`, `op_` and
    /// the value; see the type's own note on why that is an `Option` and not `Precision::None`.
    ///
    /// ⛔ NOT `e234_setValue`. The constructor's body IS a `setValue` call, and clearing before
    /// pushing is trivially the same thing when the list starts empty — but `setValue` is a public
    /// mutator with its own callers and its own entry (234/384), so it is not anchored here.
    #[must_use]
    pub fn new(kind: VectorOperandType, value: OperandValue, op: OpId) -> VectorOperand {
        VectorOperand {
            kind,
            op,
            values: vec![value],
            orig_precision: None,
            on_the_fly_conv_precision: None,
            splat: None,
        }
    }

    /// Replaces: e234_setValue
    ///
    /// THE OPERAND'S ONE VALUE, REPLACING WHATEVER IT HELD — `values_.clear();
    /// values_.emplace_back(val);` (`VectorOperands.hpp:72-75`).
    ///
    /// ⛔ IT CLEARS FIRST, which is what makes `OperandReuse`'s re-valuing to [`sen::Port::Latch`]
    /// (`OperandReuse.cpp:28-43`) REPLACE a slice index rather than append a second entry to a member
    /// whose reader is `values_.front()`.
    pub fn set_value(&mut self, val: OperandValue) {
        self.values.clear();
        self.values.push(val);
    }

    /// Replaces: e071_getOperandFromReceiveOp
    ///
    /// **071/384** `VectorOperand::getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34` (54L).
    ///
    /// ```cpp
    /// std::string link;
    /// std::string unit_str;
    /// std::optional<std::string> unit_str_optional = dcc::uniform::utils::findUnitType(receive_op.getFromUnit());
    /// if (unit_str_optional.has_value()) { unit_str = unit_str_optional.value(); }
    /// else { receive_op.emitOpError("Unit type is inconsistent in ReceiveOp."); return std::nullopt; }
    /// auto record = EnumsConversion::stringToSenComponents.find(unit_str);
    /// if (record != EnumsConversion::stringToSenComponents.end()) {
    ///   auto generic = EnumsConversion::senCompToGenericComp.at(record->second);
    ///   if (comp == PT) {
    ///     if (record->second == L0LU) { link = "west"; }
    ///     else if (generic == CROSSPTNLINK) { link = "crossptnlink"; }
    ///     else if (generic == PT || record->second == SFP || record->second == LXLU) { link = "north"; }
    ///     else { receive_op->emitError("PT cannot expect data other than L0-LU, N-link, CROSS-PT-N-LINK"); return std::nullopt; }
    ///   } else {  // PE/SFP
    ///     if (record->second == LXLU || record->second == LXSU) { link = "lx"; }
    ///     else if (record->second == PE) { link = "pe"; }
    ///     else if (record->second == SFP) {
    ///       link = "sfp";
    ///       if (dcc_ext_ctx.getArch() >= RCUDD1A_ISA && comp == SFP) {
    ///         if (EnumsConversion::stringToSenComponents.at(unit_str) == SFP) { link += "ring"; }
    ///       }
    ///     }
    ///     else if (generic == PT) { link = "pt"; }
    ///     else { receive_op->emitError("Unsupported receive unit for PE/SFP"); return std::nullopt; }
    ///   }
    ///   return VectorOperand(Link, link, receive_op.getOperation());
    /// } else { receive_op->emitError("Unknown receiver"); return std::nullopt; }
    /// ```
    ///
    /// ⛔⛔ THE OPERAND IT RETURNS **IS** THE FUNCTION, and its whole payload is the link name. A
    /// version of this that decided the port and returned nothing would leave every PE/SFP compute
    /// without an `opA`.
    ///
    /// ⭐⭐ WHICH PORT, MEASURED AGAINST THE VENDOR'S OWN GOLDENS.
    /// `Conversion/VectorChainToSentientPT/loweringXRF_with_if_branch.mlir:130` is a
    /// `dataflow.get_unit` with `type = "l0lu"`, three `dataflow.receive`s read it (`:145`, `:154`,
    /// `:164`), and the expectation is `opA = #sentient<compute_port west>` (`:49`, `:58`, `:68`) —
    /// the `L0LU` arm.
    /// `.../xrf_increments.mlir:413-457` receives from an `lxlu` on a PT unit and expects
    /// `opC = #sentient<compute_port north>` (`:71`, `:81`, `:88`) — the `LXLU` arm.
    ///
    /// ⛔ THE PEER IS A RESOLVED UNIT, NOT A STRING, so two of the reference's four failure paths
    /// cannot be reached from here and are not written:
    ///
    /// - `findUnitType`'s empty optional (*"Unit type is inconsistent in ReceiveOp."*) is a
    ///   disagreement between the `core`/`corelet` attributes of the units a `query_map` names — a
    ///   question about the IR the caller resolved before it had a [`DfirUnit`] at all;
    /// - *"Unknown receiver"* is `stringToSenComponents.find` missing. That map is
    ///   `flipMap(senComponentsToString)` (`arch_enums.cpp`), and [`DfirUnit::spelling`] is the same
    ///   table — so a spelling that came OUT of it cannot fail to go back IN.
    ///
    /// ⛔ AND THE `stringToSenComponents.at(unit_str) == SFP` RE-CHECK IS A TAUTOLOGY IN THE
    /// REFERENCE ITSELF. It sits inside `else if (record->second == SFP)`, where `record->second` is
    /// that exact lookup; the second `at()` asks a question already answered one line above.
    ///
    /// ⭐ THE IF-CHAIN IS A TOTAL MATCH HERE, AND THE ORDER SURVIVES IT. The reference's arms are
    /// disjoint — `L0LU`, then the only unit whose generic is `CROSSPTNLINK`, then the PT rows plus
    /// `SFP` and `LXLU` — so no unit reaches two of them and precedence carries no information. What
    /// a match buys is that a nineteenth [`DfirUnit`] has to say which arm it belongs to.
    #[must_use]
    pub fn from_receive_op<A: Arch>(
        from_unit: DfirUnit,
        comp: ComputeComp,
        receive_op: OpId,
    ) -> VectorOperand {
        let link = match comp {
            ComputeComp::Pt => match from_unit {
                DfirUnit::L0lu => sen::Port::West,
                // `generic == CROSSPTNLINK` — one unit maps there.
                DfirUnit::CrossPtnLink => sen::Port::CrossPtNorthLink,
                // `generic == PT` is every row of the matrix unit; the other two are named exactly.
                DfirUnit::PtRow(_) | DfirUnit::Sfp | DfirUnit::Lxlu => sen::Port::North,
                DfirUnit::Pe
                | DfirUnit::Lxsu
                | DfirUnit::Lx
                | DfirUnit::Hbm
                | DfirUnit::L0su
                | DfirUnit::L0
                | DfirUnit::L3lu
                | DfirUnit::L3su
                | DfirUnit::Constant
                | DfirUnit::SfpState
                | DfirUnit::PeState
                | DfirUnit::SfpRing
                | DfirUnit::LxVirtualIbr
                | DfirUnit::L3Ibr
                | DfirUnit::LxluScaleReg => todo!(
                    "PT cannot expect data other than L0-LU, N-link, CROSS-PT-N-LINK (VectorOperands.cpp:59-63)"
                ),
            },
            // PE/SFP.
            ComputeComp::Pe | ComputeComp::Sfp => match from_unit {
                DfirUnit::Lxlu | DfirUnit::Lxsu => sen::Port::Lx,
                DfirUnit::Pe => sen::Port::Pe,
                // ⭐ THE RING IS THE SFP TALKING TO ITSELF. A PE receiving from an SFP gets plain
                // `sfp`; only an SFP asking gets `sfpring`, and only from DD1 up — which is every
                // arch here, see [`supports_sfp_ring`].
                DfirUnit::Sfp => {
                    if supports_sfp_ring(A::GEN) && matches!(comp, ComputeComp::Sfp) {
                        sen::Port::SfpRing
                    } else {
                        sen::Port::Sfp
                    }
                }
                DfirUnit::PtRow(_) => sen::Port::Pt,
                DfirUnit::Lx
                | DfirUnit::Hbm
                | DfirUnit::L0lu
                | DfirUnit::L0su
                | DfirUnit::L0
                | DfirUnit::L3lu
                | DfirUnit::L3su
                | DfirUnit::Constant
                | DfirUnit::SfpState
                | DfirUnit::PeState
                | DfirUnit::SfpRing
                | DfirUnit::LxVirtualIbr
                | DfirUnit::L3Ibr
                | DfirUnit::CrossPtnLink
                | DfirUnit::LxluScaleReg => {
                    todo!("Unsupported receive unit for PE/SFP (VectorOperands.cpp:76)")
                }
            },
        };

        VectorOperand::new(
            VectorOperandType::Link,
            OperandValue::Port(link),
            receive_op,
        )
    }

    /// Replaces: e072_getOperandFromSendOp
    ///
    /// **072/384** `VectorOperand::getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95` (50L).
    ///
    /// ```cpp
    /// std::string link;
    /// std::string unit_str;
    /// std::optional<std::string> unit_str_optional = dcc::uniform::utils::findUnitType(send_op.getToUnit());
    /// if (unit_str_optional.has_value()) { unit_str = unit_str_optional.value(); }
    /// else { send_op.emitOpError("Unit type is inconsistent in SendOp."); return std::nullopt; }
    /// auto record = EnumsConversion::stringToSenComponents.find(unit_str);
    /// if (record != EnumsConversion::stringToSenComponents.end()) {
    ///   auto generic = EnumsConversion::senCompToGenericComp.at(record->second);
    ///   if (comp == PT) {
    ///     if (generic == PT || record->second == PE) { link = "south"; }
    ///   } else {  // PE/SFP
    ///     if (generic == PT) { link = "pt"; }
    ///     else if (record->second == PE) { link = "pe"; }
    ///     else if (record->second == SFP) {
    ///       link = "sfp";
    ///       if (comp == SFP) {
    ///         if (dcc_ext_ctx.getArch() >= RCUDD1A_ISA) link += "ring";
    ///         else send_op.emitWarning("SFP to SFP communication requires target arch DD1 and above");
    ///       }
    ///     }
    ///     else if (record->second == L0LU || record->second == L0SU) { link = "l0"; }
    ///     else if (record->second == LXLU || record->second == LXSU) { link = "lx"; }
    ///   }
    ///   if (link.empty()) { send_op->emitError("Unsupported destination for PE/SFP FMA: " + unit_str); return std::nullopt; }
    ///   return VectorOperand(Link, link, send_op.getOperation());
    /// } else { send_op->emitError("Unknown destination"); return std::nullopt; }
    /// ```
    ///
    /// ⭐⭐ MEASURED AGAINST THE VENDOR'S OWN GOLDEN.
    /// `Conversion/VectorChainToSentientPT/xrf_increments.mlir:422` is `dataflow.send %5, %44`, and
    /// the expectation is `ResultForwarding = [#sentient<compute_port south>]` (`:81`, `:88`) — the
    /// `generic == PT` arm of the PT branch. ⭐ `%5` IS NOT A UNIT BUT A `uniform.query_map` (`:378`)
    /// over a mapping (`:377`) whose two targets are both `type = "ptrow1"` (`:375`, `:376`).
    /// Resolving that is precisely what `findUnitType` does, and its empty optional — the reference's
    /// *"Unit type is inconsistent in SendOp."* — is those two targets disagreeing.
    ///
    /// ⛔⛔ THE PT BRANCH HAS NO `else`, AND THAT IS WHY THE REFUSAL IS AT THE BOTTOM. An unmatched
    /// destination on a PT leaves `link` empty and falls into `if (link.empty())` — whose message says
    /// *"Unsupported destination for PE/SFP FMA"* even though the unit asking is the PT. Both branches
    /// are written out here, both reach that same refusal, and the message is reproduced as the
    /// reference words it.
    ///
    /// ⛔ THE `emitWarning` ARM IS UNREACHABLE ON EVERY ARCH THIS CRATE BUILDS FOR — see
    /// [`supports_sfp_ring`] — so SFP→SFP always spells `sfpring` here. It is written because it is
    /// the function, and it is where an older generation lands the day one is added to [`IsaGen`].
    ///
    /// ⛔ TWO FAILURE PATHS ARE UNREACHABLE FROM A RESOLVED [`DfirUnit`] — the same two
    /// [`Self::from_receive_op`] documents, with *"Unknown destination"* in place of
    /// *"Unknown receiver"*.
    ///
    /// ⭐ NOTE THE ASYMMETRY WITH THE RECEIVE SIDE, WHICH IS THE REFERENCE'S: a PE/SFP may SEND to
    /// the L0 and to either LX half, and may not RECEIVE from the L0 at all.
    #[must_use]
    pub fn from_send_op<A: Arch>(
        to_unit: DfirUnit,
        comp: ComputeComp,
        send_op: OpId,
    ) -> VectorOperand {
        let link = match comp {
            ComputeComp::Pt => match to_unit {
                DfirUnit::PtRow(_) | DfirUnit::Pe => sen::Port::South,
                DfirUnit::Sfp
                | DfirUnit::Lxlu
                | DfirUnit::Lxsu
                | DfirUnit::Lx
                | DfirUnit::Hbm
                | DfirUnit::L0lu
                | DfirUnit::L0su
                | DfirUnit::L0
                | DfirUnit::L3lu
                | DfirUnit::L3su
                | DfirUnit::Constant
                | DfirUnit::SfpState
                | DfirUnit::PeState
                | DfirUnit::SfpRing
                | DfirUnit::LxVirtualIbr
                | DfirUnit::L3Ibr
                | DfirUnit::CrossPtnLink
                | DfirUnit::LxluScaleReg => todo!(
                    "Unsupported destination for PE/SFP FMA (VectorOperands.cpp:136-139, reached from the PT branch)"
                ),
            },
            // PE/SFP.
            ComputeComp::Pe | ComputeComp::Sfp => match to_unit {
                DfirUnit::PtRow(_) => sen::Port::Pt,
                DfirUnit::Pe => sen::Port::Pe,
                DfirUnit::Sfp => {
                    if matches!(comp, ComputeComp::Sfp) && supports_sfp_ring(A::GEN) {
                        sen::Port::SfpRing
                    } else {
                        sen::Port::Sfp
                    }
                }
                DfirUnit::L0lu | DfirUnit::L0su => sen::Port::L0,
                DfirUnit::Lxlu | DfirUnit::Lxsu => sen::Port::Lx,
                DfirUnit::Lx
                | DfirUnit::Hbm
                | DfirUnit::L0
                | DfirUnit::L3lu
                | DfirUnit::L3su
                | DfirUnit::Constant
                | DfirUnit::SfpState
                | DfirUnit::PeState
                | DfirUnit::SfpRing
                | DfirUnit::LxVirtualIbr
                | DfirUnit::L3Ibr
                | DfirUnit::CrossPtnLink
                | DfirUnit::LxluScaleReg => {
                    todo!("Unsupported destination for PE/SFP FMA (VectorOperands.cpp:136-139)")
                }
            },
        };

        VectorOperand::new(VectorOperandType::Link, OperandValue::Port(link), send_op)
    }
}

/// THE VALUE A SPLATTED CONSTANT OPERAND CARRIES — the domain `constValToField` accepts.
///
/// ⛔⛔ FOUR CASES BECAUSE THE FIFTH THROWS. The reference takes a `double` and ends its
/// comparison chain with `DT_ERROR("Only 0, 1, 2, or 3 are supported values")`
/// (`VectorOperands.cpp:260`), which is an unconditional `throw` — so the `return ""` on the next
/// line is dead, and so are both callers' `if (value == "")` arms (`:286-289`). The accepted domain
/// is exactly these four, and naming them makes the invariant a TYPE rather than a runtime refusal,
/// the way entry 048 handled `getSentientCmpIPredicate`'s six predicates.
///
/// ⭐ AND THE HARDWARE AGREES THAT FOUR IS THE SET. `zero`, `one`, `two` and `three` are four
/// PSEUDO-UNITS of the compute port attribute (`SentientTypes.td:100-106`), not four numbers —
/// there is no `four` port for a fifth case to name.
///
/// ⛔ NOT A FLOAT, AND NOT A NUMBER AT ALL HERE. The reference's parameter is a `double` only
/// because its two callers pass either `IntegerAttr::getInt()` or `FloatAttr::getValueAsDouble()`
/// (`:277-281`); what the chain of `const_val == N` tests actually decides is WHICH OF FOUR PORTS
/// the splat is read from. Keeping the double would put a `clippy::float_cmp` equality on the one
/// decision in this file that has a closed set for an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstantOperandValue {
    /// A splat of 0 — `zero`.
    Zero,
    /// A splat of 1 — `one`.
    One,
    /// A splat of 2 — `two`.
    Two,
    /// A splat of 3 — `three`.
    Three,
}

impl ConstantOperandValue {
    /// THE SPLAT THIS IS, or `None` for a value no pseudo-port names.
    ///
    /// ⛔⛔ THIS IS THE HALF OF `constValToField` THAT NAMING ITS DOMAIN PUSHED OUT TO THE CALLER.
    /// The reference's chain is `if (const_val == 0) return "zero"; else if (const_val == 1) …` with
    /// a `DT_ERROR` under it (`VectorOperands.cpp:250-262`) — recognition and spelling in one
    /// function. Entry 073 ported the spelling over this enum, so the recognition lives here, at the
    /// one place a value from the IR becomes one of the four.
    ///
    /// ⭐ AND `None` IS THE REFERENCE'S `DT_ERROR`, NOT ITS `return ""`. Both of its callers
    /// (entries 166 and 167) test the returned string for emptiness and never see it, because the
    /// throw happens first; declining here is the same refusal reached the way this crate reaches
    /// one.
    #[must_use]
    pub const fn of(splat: i64) -> Option<ConstantOperandValue> {
        match splat {
            0 => Some(ConstantOperandValue::Zero),
            1 => Some(ConstantOperandValue::One),
            2 => Some(ConstantOperandValue::Two),
            3 => Some(ConstantOperandValue::Three),
            _ => None,
        }
    }
}

/// Replaces: e073_constValToField
///
/// THE COMPUTE PORT A SPLATTED CONSTANT OPERAND IS READ FROM — `constValToField`
/// (`VectorOperands.cpp:250`).
///
/// ⭐ THE RESULT IS A PORT, NOT A NAME. The reference's `std::string` goes straight into
/// `VectorOperand::values_` and comes back out through `symbolizeSentientComputePort`
/// (`VectorChainToSentientPESFP.cpp:722-726`) — the string round-trips into this very enumeration,
/// so the port is what the function computes and the spelling is [`sen::Port`]'s business.
///
/// ⭐ TOTAL, so there is nothing for a caller to check. `getOperandFromConstantOp` (entry 166) and
/// `getOperandFromConstantBitstreamOp` (entry 167) each test the returned string against `""`
/// before using it; with the domain named, both tests are statically false and both branches go.
#[must_use]
pub const fn const_val_to_field(const_val: ConstantOperandValue) -> sen::Port {
    match const_val {
        ConstantOperandValue::Zero => sen::Port::Zero,
        ConstantOperandValue::One => sen::Port::One,
        ConstantOperandValue::Two => sen::Port::Two,
        ConstantOperandValue::Three => sen::Port::Three,
    }
}

/// THE OP AT A POSITION, or `None` where the path names nothing in `scope`.
///
/// ⭐ ONE ORDINAL PER LEVEL, AND A MULTI-REGION OP'S REGIONS ARE CONCATENATED in [`regions`]
/// order. [`OpId`]'s ordinals are per LEVEL, not per region, so an op with two regions needs a rule;
/// this is it, and [`remove_at`] flattens the same way through [`regions_mut`]. `scf.if` is the only
/// op in this island with two regions and nothing this compiler emits constructs one — a fact
/// `tf_cfg_simplification_dataflow_level.rs` records against its own conditional walk — so the
/// flattening is unobservable today.
#[must_use]
pub(super) fn op_at<'a>(id: &OpId, scope: &'a [DfirOp]) -> Option<&'a DfirOp> {
    let (first, rest) = id.path().split_first()?;
    let mut op = scope.get(*first as usize)?;
    for ordinal in rest {
        op = regions(op).into_iter().flatten().nth(*ordinal as usize)?;
    }
    Some(op)
}

/// WHETHER A POSITION HOLDS AN `arith.constant` — `isa<mlir::arith::ConstantOp>(op)`.
///
/// ⛔⛔ NOT `kind == VectorOperandType::Constant`, WHICH IS A DIFFERENT QUESTION, and the two
/// disagree in BOTH directions. `getOperandFromConstantBitstreamOp` builds a `Constant`-kinded
/// operand over a `vectorchain.constant_bitstream` (`VectorOperands.cpp:305`) — kind `Constant`,
/// not an `arith.constant`; and the trivial-shuffle path re-tags an operand built over its parent as
/// `NFWD` or `ConstantBitstream` after the fact (`:317-336`) — so an operand whose op IS an
/// `arith.constant` can carry any of three kinds. The reference asks the OP, and so does this.
///
/// ⛔ ALL THREE OF THIS ISLAND'S CONSTANT VARIANTS ARE THE ONE C++ OP CLASS. `arith::ConstantOp`
/// covers the index, integer and dense-vector forms; [`arith::Op::Constant`],
/// [`arith::Op::ConstantInt`] and [`arith::Op::DenseConstant`] are split here because they PRINT
/// differently (see `ConstantInt`'s note), so the `isa<>` is a match on all three.
///
/// ⭐ A PATH THAT RESOLVES TO NOTHING IS NOT A CONSTANT, and [`same_block`] then falls through to
/// its block comparison. An `Operation *` cannot dangle in the reference; a position can name an op
/// in a scope it was not given, and the total answer to "is that a constant" is no.
pub(super) fn is_arith_constant(op: &OpId, scope: &[DfirOp]) -> bool {
    matches!(
        op_at(op, scope),
        Some(DfirOp::Arith(
            arith::Op::Constant { .. }
                | arith::Op::ConstantInt { .. }
                | arith::Op::DenseConstant { .. }
        ))
    )
}

/// Replaces: e074_sameBlock
///
/// WHETHER AN OPERAND IS DEFINED IN THE SAME BLOCK AS THE OP USING IT, THE CONSTANT EXCEPTED —
/// `VectorOperand::sameBlock` (`VectorOperands.cpp:652`), the single-operand overload.
///
/// ```text
///   if (operand.has_value()) {
///     if (!isa<mlir::arith::ConstantOp>(operand.value().op_) &&
///         operand.value().op_->getBlock() != this_op->getBlock()) {
///       return LogicalResult::failure();
///     }
///     return LogicalResult::success();
///   } else {
///     return LogicalResult::failure();
///   }
/// ```
///
/// ⛔⛔ AN ABSENT OPERAND IS A FAILURE, NOT A SUCCESS. `false` here is `LogicalResult::failure()`,
/// and the sole caller — `OperandReuse::setReuseInformation` (`OperandReuse.cpp:31`) — turns a
/// failure into `operand_i.setValue("latch")`, i.e. "re-read it, do not reuse the register". Getting
/// the empty case backwards would have an unknown operand claim reuse.
///
/// ⛔ WHY THE CONSTANT IS EXEMPT: MLIR canonicalisation hoists `arith.constant` out of the block
/// that uses it, so a constant operand is *expected* to be defined elsewhere and that is not a
/// reason to latch. ⭐ THE EXEMPTION IS UNREACHABLE FROM THE ONE CALLER, which guards the call with
/// `if (operand_i.type_ != Constant)` — but "unreachable at today's only call site" is not the same
/// claim as "not part of the function", and the second overload (see the scope note below) is called
/// from two more places.
///
/// ⭐ `scope` IS HOW A POSITION ANSWERS `isa<>`. `Operation *` carries its class; [`OpId`] carries
/// only its place, and the ops it is a place in are the rest of the answer. `agen_helper.rs`'s
/// entry 038 takes the same `scope: &[DfirOp]` for the same reason.
#[must_use]
pub fn same_block(this_op: &OpId, operand: Option<&VectorOperand>, scope: &[DfirOp]) -> bool {
    // `if (operand.has_value())` … `} else { return LogicalResult::failure(); }`
    let Some(operand) = operand else {
        return false;
    };

    // `if (!isa<mlir::arith::ConstantOp>(operand.value().op_) &&
    //      operand.value().op_->getBlock() != this_op->getBlock()) return failure();`
    if !is_arith_constant(&operand.op, scope) && operand.op.block() != this_op.block() {
        return false;
    }

    // `return LogicalResult::success();`
    true
}

/// WHETHER NOTHING IN `scope` READS ANY RESULT OF THE OP AT A POSITION — `user->getUses().empty()`.
///
/// ⛔ `false` FOR A POSITION THAT NAMES NO OP, which is the reference's `user &&` guard
/// (`VectorOperands.cpp:810`). A null user is not erasable there and leaves `all_uses_deleted`
/// false; an unresolvable position gets the same answer here, so an op whose users cannot all be
/// accounted for is never erased.
fn has_no_uses(id: &OpId, scope: &[DfirOp]) -> bool {
    match op_at(id, scope) {
        Some(op) => results(op)
            .into_iter()
            .all(|result| uses(result, scope).is_empty()),
        None => false,
    }
}

/// EVERY USE OF THE RESULTS OF THE OP AT `of`, AS POSITIONS — `op->getUses()` with the owner of
/// each use resolved.
///
/// ⛔ ONE ENTRY PER **USE**, NOT PER USER, matching [`uses`] and `Value::use_begin()`: an op that
/// reads the same result twice appears twice, and that is what makes the reference's
/// `all_uses_deleted` loop count iterations the way MLIR does.
///
/// ⛔ AND IT DESCENDS INTO REGIONS, because [`uses`] does. A result read by an op inside an
/// `affine.for` body has a user, and a walk that stopped at the top level would erase the definition
/// out from under it.
pub(super) fn use_positions(of: &OpId, scope: &[DfirOp]) -> Vec<OpId> {
    let Some(op) = op_at(of, scope) else {
        return Vec::new();
    };
    let produced = results(op);
    let mut found: Vec<OpId> = Vec::new();
    collect_use_positions(&produced, scope, &[], 0, &mut found);
    found
}

/// [`use_positions`]'s recursion.
///
/// `prefix` is the position of the op OWNING the block `scope` is, and `base` is where that block
/// starts in the owner's flattened region sequence — the numbering [`op_at`] reads back. The
/// top-level call passes the empty prefix and zero, which is the program unit body's own block.
fn collect_use_positions(
    of: &[Val],
    scope: &[DfirOp],
    prefix: &[u32],
    base: u32,
    found: &mut Vec<OpId>,
) {
    for (ordinal, op) in scope.iter().enumerate() {
        let mut path: Vec<u32> = prefix.to_vec();
        path.push(base + ordinal as u32);

        for read in operands(op) {
            if of.contains(&read) {
                found.push(OpId::at(&path));
            }
        }

        let mut child = 0u32;
        for region in regions(op) {
            collect_use_positions(of, region, &path, child, found);
            child += region.len() as u32;
        }
    }
}

/// REMOVE THE OP AT A POSITION — `Operation::erase()`.
///
/// ⭐ IT DESCENDS WITH [`regions_mut`], ARM FOR ARM WITH [`op_at`]'s [`regions`] and with the same
/// flattening of a multi-region op, so a position read one way is removed the other.
///
/// ⛔ A PATH THAT NAMES NOTHING REMOVES NOTHING. There is no other total answer, and the reference
/// cannot reach the case — `Operation::erase()` takes a live pointer.
pub(super) fn remove_at(path: &[u32], scope: &mut Vec<DfirOp>) {
    let Some((first, rest)) = path.split_first() else {
        return;
    };
    let first = *first as usize;

    if rest.is_empty() {
        if first < scope.len() {
            scope.remove(first);
        }
        return;
    }

    let Some(op) = scope.get_mut(first) else {
        return;
    };

    // Descend one level, re-basing the next ordinal onto the region that actually holds it.
    let mut wanted = rest[0] as usize;
    for region in regions_mut(op) {
        if wanted < region.len() {
            let mut rebased: Vec<u32> = vec![wanted as u32];
            rebased.extend_from_slice(&rest[1..]);
            remove_at(&rebased, region);
            return;
        }
        wanted -= region.len();
    }
}

/// Replaces: e075_eraseOp
///
/// ERASE AN OP AND THE USERS THAT NOTHING ELSE READS — `VectorOperand::eraseOp`
/// (`VectorOperands.cpp:806`), the one-argument overload.
///
/// ```text
///   bool all_uses_deleted = true;
///   std::vector<mlir::Operation *> to_be_erased;
///   for (auto &use : op->getUses()) {
///     Operation *user = use.getOwner();
///     if (user && user->getUses().empty()) {
///       to_be_erased.push_back(user);
///     } else {
///       all_uses_deleted = false;
///     }
///   }
///   for (auto e : to_be_erased) e->erase();
///   if (all_uses_deleted) op->erase();
/// ```
///
/// ⛔⛔ ONE LEVEL OF USERS, NOT A TRANSITIVE SWEEP. A user is erased only when NOTHING reads it, and
/// the users of *that* user are never examined — the reference walks exactly one edge. A recursive
/// version would delete a chain whose head this pass has not decided to lower, which is why
/// `eraseOperands` (entry 233) exists separately with its own `intermediate_ops` list.
///
/// ⛔⛔ AND `op` GOES ONLY IF **EVERY** USE WAS ERASABLE. One surviving reader keeps the definition,
/// so the two loops are not independent: a partially-erased use list leaves `op` in place, still
/// feeding whatever survived.
///
/// ⛔ THE ERASURE ORDER IS DESCENDING, AND THAT IS THIS PORT'S OBLIGATION, NOT THE REFERENCE'S. An
/// `Operation *` stays valid while its siblings are erased; a POSITION does not — removing `[3]`
/// renumbers `[4]` to `[3]`. Every position invalidated by removing `p` (`p`'s later siblings, and
/// everything under them) is lexicographically GREATER than `p`, so removing in descending
/// lexicographic order removes each op before anything that could renumber it.
///
/// ⭐ AND `op`'s OWN POSITION SURVIVES THAT LOOP BY DOMINANCE. A user of a result comes after the
/// op that defines it, so every position in `to_be_erased` is lexicographically greater than `op`'s
/// and none of them renumbers it.
///
/// ⛔ THE REFERENCE CAN PUSH ONE USER TWICE — an op reading the same result twice appears twice in
/// `getUses()`, and `to_be_erased` is not deduplicated, so `e->erase()` runs twice on it. That is a
/// double free there; here it would remove a *different, innocent* op at the same ordinal, so this
/// port deduplicates. The behaviour the reference intends is a single erase per op.
pub fn erase_op(op: &OpId, scope: &mut Vec<DfirOp>) {
    let mut all_uses_deleted = true;
    let mut to_be_erased: Vec<OpId> = Vec::new();

    // `for (auto &use : op->getUses()) { Operation *user = use.getOwner(); … }`
    for user in use_positions(op, scope) {
        // `if (user && user->getUses().empty())`
        if has_no_uses(&user, scope) {
            if !to_be_erased.contains(&user) {
                to_be_erased.push(user);
            }
        } else {
            all_uses_deleted = false;
        }
    }

    // `for (auto e : to_be_erased) e->erase();` — ⛔ descending, see the note above.
    to_be_erased.sort_unstable();
    for position in to_be_erased.iter().rev() {
        remove_at(position.path(), scope);
    }

    // `if (all_uses_deleted) op->erase();`
    if all_uses_deleted {
        remove_at(op.path(), scope);
    }
}

/// WHICH READING A CALLER WANTS OF A `vectorchain.constant_bitstream`'S FIRST ELEMENT — the
/// `is_constant_splatted_vector` argument of `getOperandFromConstantBitstreamOp`
/// (`VectorOperands.cpp:299-301`).
///
/// ⛔⛔ THE FLAG DECIDES WHAT KIND OF THING THE VALUE IS, so it is not a `bool` beside an `i64` but
/// the integer's own tag. `true` runs the element through `constValToField` and the answer is one of
/// four PSEUDO-UNITS; `false` runs it through `std::to_string` and the answer is an IMMEDIATE with no
/// compute-port spelling at all (see [`OperandValue::Literal`]). A `bool` and an `i64` in the same
/// signature can be transposed at a call site; these cannot.
///
/// ⭐ AND THE CLASSIFICATION IS THE CALLER'S, WHICH IS WHY THE SPLATTED CASE CARRIES A
/// [`ConstantOperandValue`]. `constValToField` has no answer for a fifth value — its four `==` tests
/// fall through to an unconditional throw (`VectorOperands.cpp:260`, see [`const_val_to_field`]) —
/// and entry 073 already made that domain a type rather than a runtime refusal. The only caller that
/// passes `true` is `getOperandFromShuffleOp`'s trivial-shuffle branch (`VectorOperands.cpp:333-334`,
/// entry 278), which is exactly the one that has established the shuffle is a recognised splat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BitstreamConstant {
    /// `is_constant_splatted_vector == true` — the element names a pseudo-unit.
    SplattedVector(ConstantOperandValue),
    /// `is_constant_splatted_vector == false` — the element IS the immediate.
    Immediate(i64),
}

/// WHAT `getLayoutMapAndIndices` HANDS BACK — the reference's three out-parameters
/// (`VectorOperands.cpp:879-881`).
///
/// ⛔ THREE OUT-PARAMETERS PLUS A `LogicalResult` IS ONE `Option<Self>`. The reference writes
/// `layout_map`, `operands` and `logical_view_op` through references and returns `success()`/
/// `failure()`; every caller checks the result before reading any of them, so "all three or none" is
/// the actual contract and a struct states it. ⛔ AND `Option` HERE IS A CLASSIFICATION, NOT A RUNTIME
/// REFUSAL: `None` is the reference's `else` arm, which is a diagnostic saying the op is not one of
/// the four memory accesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutAndIndices {
    /// `layout_map` — the view's layout, composed with the access's own subscripts for an `agen`
    /// access and left alone for a plain one. See [`layout_map_and_indices`] on why that differs.
    pub layout_map: AffineMap,
    /// `operands` — the SSA values the subscripts are computed from, in the order the access lists
    /// them.
    pub operands: Vec<Val>,
    /// `logical_view_op` — where the `dataflow.get_logical_memory_view` the access indexes lives.
    pub logical_view_op: OpId,
}

impl VectorOperand {
    /// Replaces: e167_getOperandFromConstantBitstreamOp
    ///
    /// **167/384** `VectorOperand::getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299` (5L).
    ///
    /// ```cpp
    /// auto const_val = mlir::cast<IntegerAttr>(op.getValue()[0]).getInt();
    /// std::string value = is_constant_splatted_vector ? constValToField(const_val)
    ///                                                 : std::to_string(const_val);
    /// return VectorOperand(Constant, value, op.getOperation());
    /// ```
    ///
    /// ⛔⛔ THE KIND IS `Constant`, NOT `ConstantBitstream`, EVEN THOUGH THE OP IS ONE. Both callers
    /// then disagree about that: the trivial-shuffle path overwrites it with `ConstantBitstream`
    /// immediately (`VectorOperands.cpp:335`) and `getOperandWithPrecision`'s own arm leaves it
    /// `Constant` (`:522-530`). The difference is load-bearing downstream — `OperandReuse` skips a
    /// `Constant` operand entirely (`OperandReuse.cpp:26`, `:57`) and would latch a
    /// `ConstantBitstream` one — so this function's answer is the `Constant` the reference writes, and
    /// re-tagging is the caller's line, not a correction to make here.
    ///
    /// ⛔ THE FIRST ELEMENT AND ONLY THE FIRST. `getValue()` is the op's whole `ArrayAttr`: a splat
    /// carries one element and a vector constant carries `128 / bitwidth` of them
    /// (`VectorChainToSentientPESFP/Splat.cpp:46-56`), and this function indexes `[0]` unconditionally with no arity check. That
    /// is sound for the splatted caller — a trivial shuffle IS the one-element case — and the other
    /// caller reaches it for any constant bitstream at all, taking element zero as the whole value.
    /// ⭐ SO THE PARAMETER IS THE ELEMENT, NOT THE OP'S VALUE LIST: nothing here can use a second
    /// element, and handing this function the list would invite a porter to think it could.
    ///
    /// ⛔ THE `std::optional` NEVER HOLDS `nullopt` AND BOTH CALLERS PROVE IT, calling `.value()` on
    /// the result with no `has_value()` test at all (`VectorOperands.cpp:335`, `:526-529`). There is
    /// no `return std::nullopt` in the body. So the port returns a [`VectorOperand`] outright rather
    /// than an `Option` no caller could ever see empty.
    ///
    /// ⭐ AND BOTH CALLERS OVERWRITE SOMETHING THE MOMENT THEY GET IT: the shuffle path the kind, and
    /// `getOperandWithPrecision` both precisions, from
    /// `getElementType(const_bit_op.getType())` (`:525-529`) — which is why the two precisions start
    /// unset here (see [`VectorOperand::new`]).
    #[must_use]
    pub fn from_constant_bitstream_op(element: BitstreamConstant, op: OpId) -> VectorOperand {
        // `std::string value = is_constant_splatted_vector ? constValToField(const_val)
        //                                                  : std::to_string(const_val);`
        let value = match element {
            BitstreamConstant::SplattedVector(field) => {
                OperandValue::Port(const_val_to_field(field))
            }
            BitstreamConstant::Immediate(const_val) => OperandValue::Literal(const_val),
        };
        // `return VectorOperand(Constant, value, op.getOperation());`
        VectorOperand::new(VectorOperandType::Constant, value, op)
    }

    /// Replaces: e168_getOperandFromNegOp
    ///
    /// **168/384** `VectorOperand::getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366` (5L).
    ///
    /// ```cpp
    /// DT_CHECK(isa<vectorchain::NegOp>(op));
    /// auto *parent = op.getOperand(0).getDefiningOp();
    /// auto operand = getOperand(dcc_ext_ctx, parent, comp);
    /// return operand;
    /// ```
    ///
    /// ⛔⛔ A NEGATION IS TRANSPARENT TO THIS QUESTION — that is the whole function. The operand of a
    /// compute that reads a `vectorchain.neg` is the operand of whatever the NEGATION reads, with the
    /// negation itself contributing nothing to the answer. It can do that because the PE/SFP lowering
    /// never converts a `NegOp`: it FOLDS one into the FMA it feeds, by `dyn_cast`ing both inputs of
    /// the multiply (`VectorChainToSentientPESFP.cpp:534-539`), so the sign lives on the consuming
    /// instruction and the operand chain must skip straight past it. ⭐ ITS SOLE CALLER SAYS SO IN A
    /// COMMENT: *"The NegOp doesn't change the original precision"* (`VectorOperands.cpp:568`), and
    /// unlike every neighbouring arm it does NOT overwrite either precision afterwards.
    ///
    /// ⛔ OPERAND 0 IS `$op` AND OPERAND 1 IS THE OPTIONAL `$mask` (`VectorChain.td:359-373`), so
    /// `getOperand(0)` is the value being negated whether or not a mask is present. Reading the mask's
    /// producer instead would give the compute a predicate as its data source.
    ///
    /// ⛔ `DT_CHECK(isa<NegOp>(op))` IS NOT AN `assert!` HERE. A position that does not hold a
    /// `vectorchain.neg` has no operand to report, so the total answer is `None` — the same answer the
    /// reference's own `parent == nullptr` case degenerates to. This crate never runtime-refuses; see
    /// the file banner.
    ///
    /// ⛔ `get_operand` IS AN SCC CUT AND IT IS SPELLED AS ONE. `getOperand` (entry 320) tail-calls
    /// `getOperandWithPrecision` (entry 304, 254 lines), which reaches this function back through its
    /// `NegOp` arm (`:566-569`) — a genuine cycle. Taking the recursion as a caller-supplied closure
    /// is the same seam `std_scf_to_sentient.rs:305` and `agen_helper.rs:1405` already use, and it
    /// keeps this unit's own content — "skip the negation, ask about its input" — testable on its own.
    ///
    /// ⭐ AND THE CUT CARRIES `traverse_upwards = true`, the declaration's default
    /// (`VectorOperands.hpp:85`), because this call passes only three arguments. That is not
    /// cosmetic: it selects the *upward* branch of `getOperandWithPrecision`'s cast and select arms
    /// (`:544` versus `:556`), so a closure that hard-coded `false` would resolve a negated
    /// cast through the cast's USER instead of its input.
    #[must_use]
    pub fn from_neg_op(
        neg_op: &OpId,
        comp: ComputeComp,
        scope: &[DfirOp],
        get_operand: &mut impl FnMut(&OpId, ComputeComp) -> Option<VectorOperand>,
    ) -> Option<VectorOperand> {
        // `DT_CHECK(isa<vectorchain::NegOp>(op));`
        let Some(DfirOp::VectorChain(vc::Op::Neg { input, .. })) = op_at(neg_op, scope) else {
            return None;
        };
        // `auto *parent = op.getOperand(0).getDefiningOp();`
        let parent = defining_position(*input, scope)?;
        // `auto operand = getOperand(dcc_ext_ctx, parent, comp); return operand;`
        get_operand(&parent, comp)
    }

    /// Replaces: e343_getOperandFromCastOp
    ///
    /// **343/384** `VectorOperand::getOperandFromCastOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354` (5L).
    ///
    /// ```cpp
    /// DT_CHECK(isa<vectorchain::CastOp>(op));
    /// auto *parent = op.getOperand(0).getDefiningOp();
    /// auto operand = getOperand(dcc_ext_ctx, parent, comp);
    /// return operand;
    /// ```
    ///
    /// ⛔⛔ A CAST IS TRANSPARENT AND THIS FORWARD IS UNCONDITIONAL — where entry 304's own `CastOp`
    /// arm asks the same question of the same input it guards it three ways and then rewrites the
    /// answer: `hasOneUse()`, the input's defining op being in the SAME BLOCK, and
    /// `on_the_fly_conv_precision_ = getPrecisionInString(getElementType(cast_op.getType()))`
    /// (`:541-565`). This function keeps neither guard and touches neither precision, so a caller that
    /// swapped one for the other would report the producer's precision through a multi-use cast.
    /// ⛔ NO CALL SITE IN THE REFERENCE — only the declaration at `VectorOperands.hpp:56`; entry 304
    /// spells the same three lines out inline. Ported for the seam it names, not for a caller.
    /// ⛔ `DT_CHECK(isa<CastOp>(op))` IS `None` HERE, and `get_operand` is the SCC cut
    /// [`Self::from_neg_op`] documents, carrying the same `traverse_upwards = true` default.
    #[must_use]
    pub fn from_cast_op(
        cast_op: &OpId,
        comp: ComputeComp,
        scope: &[DfirOp],
        get_operand: &mut impl FnMut(&OpId, ComputeComp) -> Option<VectorOperand>,
    ) -> Option<VectorOperand> {
        // `DT_CHECK(isa<vectorchain::CastOp>(op));`
        let Some(DfirOp::VectorChain(vc::Op::Cast { input, .. })) = op_at(cast_op, scope) else {
            return None;
        };
        // `auto *parent = op.getOperand(0).getDefiningOp();`
        let parent = defining_position(*input, scope)?;
        // `auto operand = getOperand(dcc_ext_ctx, parent, comp); return operand;`
        get_operand(&parent, comp)
    }

    /// Replaces: e169_getName
    ///
    /// **169/384** `VectorOperand::getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866` (11L).
    ///
    /// ```cpp
    /// if (this->type_ == LRF) return "lrf" + this->getFirstValue();
    /// else if (this->type_ == IRF) return "irf" + this->getFirstValue();
    /// else if (this->type_ == XRF) return "xrf";
    /// else if (this->type_ == ISTATE) return "istate" + this->getFirstValue();
    /// else return this->getFirstValue();
    /// ```
    ///
    /// THE OPERAND'S NAME AS THE SENTIENT INSTRUCTION SPELLS IT — a register file's slice gets its
    /// file's prefix, everything else is its value verbatim.
    ///
    /// ⛔⛔ IT MATCHES THE **VALUE** AND THE REFERENCE MATCHES `type_`, AND THAT DIFFERENCE IS A
    /// DELIBERATE DIVERGENCE THAT FIXES A DEFECT. `OperandReuse::setReuseInformation` re-values an
    /// operand to `"latch"` for every kind except `Constant` (`OperandReuse.cpp:26-43`) — **including
    /// `LRF`** — and never touches `type_`. So on the reference a latched LRF operand answers
    /// `"lrflatch"`, which is not a `SentientComputePort` at all. ⭐ AND THE SAME FUNCTION'S OWN
    /// SECOND LOOP IS THE EVIDENCE THAT `latch` WAS THE INTENDED ANSWER: `:57` tests
    /// `from.getName() != "latch"` **unprefixed**, so a latched LRF passes a test written to exclude
    /// it and gets `setReuseFlag` called on the very op that was just told to re-read. Here the value
    /// carries which register file it is a slice OF (see [`RegisterSlice`]), so `latch` answers
    /// `latch` and there is no spelling to concatenate.
    ///
    /// ⭐ THE `XRF` ARM IS REDUNDANT WITH THE `else`, AND SAYING SO IS THE POINT. An `XRF` operand is
    /// constructed as `VectorOperand(operand_type, "xrf", op)` (`VectorOperands.cpp:219-220`) — the
    /// only place one is built — so its first value already IS `"xrf"` and the `else` would return the
    /// same string. It is also the only kind with no slice index, because the XRF is one register and
    /// the arm above it skips the whole slice computation for exactly that reason (`:209`, `:219`).
    /// [`OperandValue::Port`] covers it with no arm of its own.
    ///
    /// ⛔ THE RESULT IS A [`sen::Port`], NOT A `String`, BECAUSE ITS READERS ARE A CLOSED SET.
    /// Every consumer either hands it to `symbolizeSentientComputePort(...).value()` — some thirty
    /// sites across `VectorChainToSentientPT.cpp:398-856` — or COMPARES two of them for equality
    /// (`OperandReuse.cpp:37-38`). Both survive typing; only the string does not.
    ///
    /// ⛔ AND `None` IS THAT `symbolize` RETURNING `std::nullopt`, NOT A REFUSAL ADDED HERE. It is
    /// reachable one way: an [`OperandValue::Literal`], whose decimal has no case in the sixty-three
    /// the `.td` declares (`SentientTypes.td:98-160`). The reference calls `.value()` on the empty
    /// optional there and dies; this hands the caller the same fact as a value.
    /// ⭐ AND IT CANNOT REACH THE EQUALITY READER, so `None == None` conflating two distinct
    /// immediates is not a behaviour this introduces: `OperandReuse.cpp:37-38` compares only inside
    /// `if (operand_i.type_ != Constant)`, and a literal value is written by exactly one constructor —
    /// [`Self::from_constant_bitstream_op`], which tags the operand `Constant`.
    ///
    /// ⛔ AN EMPTY VALUE LIST IS ALSO `None`, AND THE REFERENCE CANNOT GET THERE: `getFirstValue()` is
    /// `values_.front()` (`VectorOperands.hpp:76`) on a vector every constructor pushes one entry
    /// into, so an empty one would be undefined behaviour rather than a case. `None` is the only total
    /// answer this port can give it.
    #[must_use]
    pub fn name(&self) -> Option<sen::Port> {
        match self.values.first() {
            // `if (this->type_ == LRF) return "lrf" + this->getFirstValue();`
            Some(OperandValue::Slice(RegisterSlice::Lrf(slice))) => Some(sen::Port::Lrf(*slice)),
            // `else if (this->type_ == IRF) return "irf" + this->getFirstValue();`
            Some(OperandValue::Slice(RegisterSlice::Irf(IrfIndex::I0))) => Some(sen::Port::Irf0),
            Some(OperandValue::Slice(RegisterSlice::Irf(IrfIndex::I1))) => Some(sen::Port::Irf1),
            // `else if (this->type_ == ISTATE) return "istate" + this->getFirstValue();`
            Some(OperandValue::Slice(RegisterSlice::IState(slice))) => {
                Some(sen::Port::IState(*slice))
            }
            // `else if (this->type_ == XRF) return "xrf";` — and `else return this->getFirstValue();`,
            // which answers the same thing for it. See the note on the redundant arm.
            Some(OperandValue::Port(port)) => Some(*port),
            // The `else` for a value with no port spelling, and for a list the reference cannot have.
            Some(OperandValue::Literal(_)) | None => None,
        }
    }
}

/// Replaces: e170_getLayoutMapAndIndices
///
/// **170/384** `vectorchain::getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879` (40L).
///
/// WHERE A MEMORY ACCESS LANDS IN ITS VIEW'S LINEAR REGION, AND WHICH VALUES ITS SUBSCRIPTS ARE
/// COMPUTED FROM — the four accesses the vectorchain lowering can read, and nothing else.
///
/// ```cpp
/// if (auto tmp_op = dyn_cast<agen::VectorStoreOp>(op)) {
///   auto indices = tmp_op.getMapOperands();
///   operands = {indices.begin(), indices.end()};
///   logical_view_op = tmp_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///   layout_map = logical_view_op.getLayoutMap();
///   auto indices_map = tmp_op.getAffineMap();
///   auto order_map = tmp_op.getStoreOrder();
///   indices_map = order_map.compose(indices_map);
///   layout_map = layout_map.compose(indices_map);
///   layout_map = compressUnusedSymbols(layout_map);
/// } else if (auto tmp_op = dyn_cast<vector::StoreOp>(op)) {
///   auto indices = tmp_op.getIndices();
///   operands = {indices.begin(), indices.end()};
///   logical_view_op = tmp_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///   layout_map = logical_view_op.getLayoutMap();
/// } else if (auto tmp_op = dyn_cast<agen::VectorLoadOp>(op)) {   // as the agen store, with
///   ...                                                          // getMapIndices/getLoadOrder
/// } else if (auto tmp_op = dyn_cast<vector::LoadOp>(op)) {       // as the vector store
///   ...
/// } else {
///   op->emitOpError("can't extract memory layout map or indices.");
///   return failure();
/// }
/// return success();
/// ```
///
/// ⛔⛔ THE `agen` PAIR COMPOSE AND THE `vector` PAIR DO NOT, AND THAT ASYMMETRY IS THE REFERENCE'S
/// OWN. It is not an omission to repair: an `agen` access carries its subscripts as an
/// `AffineMapAttr` plus operands and a separate `load_order`/`store_order` permutation, so the address
/// it names is only known after both are folded into the view's layout; a `vector.load`/`vector.store`
/// has neither — `Vector_LoadOp`'s arguments are `$base` and `Variadic<Index>:$indices` and that is
/// all (see [`vector::Op`]) — so its subscripts are already the view's own dimensions and the layout
/// map alone answers the question. ⛔ THE CONSEQUENCE IS VISIBLE TO BOTH CALLERS: for a plain access
/// the returned map is indexed by the VIEW's dimensions, and for an `agen` one by the LOOP NEST's.
/// `getOperandFromLoadOrStoreOp` (entry 232) then requires one result and a single constant
/// (`VectorOperands.cpp:165`, `:209`), and `LoweringXRF::getLayoutExpr` (entry 240) feeds the map and
/// the operand list to `fullyComposeAffineMapAndOperands` and requires
/// `getNumInputs() == operands.size()` (`LoweringXRF.cpp:37-38`) — so both the map's arity and its
/// dimension space are contracts, not incidentals.
///
/// ⛔ THE `order_map` IS THE IDENTITY HERE BECAUSE THIS ISLAND EMITS NO OTHER. `agen.vector_load`'s
/// `load_order` is a real attribute that CAN permute — `Agen.td`'s own example writes
/// `affine_map<(d0, d1) -> (d1, d0)>` — but [`agen::Op::VectorLoad`] does not carry the field, and its
/// printer derives `load_order = identity_map(view_ty.shape.len())`. So the composition runs for real
/// against the order map this bridge actually writes, and a permuting order is an island field to add
/// on the day something needs one, not a silently dropped step. ⭐ AND MLIR'S OWN
/// `replaceDimsAndSymbols` MAKES THE IDENTITY CASE EXACT rather than approximately right: composing
/// with it rebuilds no node at all (see [`AffineExpr::replace_dims_and_symbols`]), so
/// `order_map.compose(indices_map)` gives back `indices_map` unchanged.
///
/// ⛔ AND `getAffineMap()` IS RECOVERED FROM THE INDEX LIST, WHICH IS WHERE THIS ISLAND KEEPS IT. The
/// vendor writes `agen.vector_store %48, %49[0, %arg9 + %arg8 * 8, 0]`
/// (`dcc/test/PT/bf16-pt.mlir:161`) — an `AffineMapAttr` `(d0, d1) -> (0, d0 + d1 * 8, 0)` printed in
/// MLIR's fused form beside its two operands — and [`Index`] is that same fusion. Splitting it back
/// out is [`access_map`], and it is the map attribute that the reference reads, not a new one.
///
/// ⛔ `None` IS THE REFERENCE'S TWO WAYS OF NOT ANSWERING, AND ONE OF THEM IT DOES NOT SURVIVE. The
/// `else` arm is a real diagnostic over every other op class; the `getDefiningOp<...>()` casts are
/// NOT — they return null for a base defined by anything other than a
/// `dataflow.get_logical_memory_view` (a paged view, say, before `TransformPagedMemView` has run) and
/// the very next line dereferences it. Both become `None`.
///
/// ⛔ AND THE `_` ARM IS RIGHT HERE, unlike in the islands' classification tables where a new dialect
/// must be a build error. The reference's `else` IS the open case — it reports the op class it was
/// handed — so there is no arm to add when the IR grows a memory access this pass does not read.
#[must_use]
pub fn layout_map_and_indices(op: &OpId, scope: &[DfirOp]) -> Option<LayoutAndIndices> {
    match op_at(op, scope)? {
        // `if (auto tmp_op = dyn_cast<agen::VectorStoreOp>(op))` — `getMapOperands`/`getStoreOrder`.
        DfirOp::Agen(agen::Op::VectorStore {
            view,
            indices,
            view_ty,
            ..
        })
        // `} else if (auto tmp_op = dyn_cast<agen::VectorLoadOp>(op)) {` — `getMapIndices`/
        // `getLoadOrder`. ⭐ TWO ACCESSORS SPELLED DIFFERENTLY FOR THE SAME THING: the store's
        // operand list is `getMapOperands()` and the load's is `getMapIndices()`, and the bodies are
        // otherwise identical.
        | DfirOp::Agen(agen::Op::VectorLoad {
            view,
            indices,
            view_ty,
            ..
        }) => mapped_access(*view, indices, view_ty, scope),
        // `} else if (auto tmp_op = dyn_cast<vector::StoreOp>(op)) {` and the `vector::LoadOp` arm —
        // `getIndices()`, `getBase()`, and NO composition.
        DfirOp::Vector(vector::Op::Store { base, indices, .. })
        | DfirOp::Vector(vector::Op::Load { base, indices, .. }) => {
            plain_access(*base, indices, scope)
        }
        // `} else { op->emitOpError("can't extract memory layout map or indices."); return failure(); }`
        _ => None,
    }
}

/// THE `agen` ARMS OF [`layout_map_and_indices`] — the two that compose.
fn mapped_access(
    view: Val,
    indices: &[Index],
    view_ty: &MemRef,
    scope: &[DfirOp],
) -> Option<LayoutAndIndices> {
    // `auto indices = tmp_op.getMapOperands(); operands = {indices.begin(), indices.end()};` and
    // `auto indices_map = tmp_op.getAffineMap();` — one field here, see the anchor's note.
    let (indices_map, operands) = access_map(indices);
    // `logical_view_op = tmp_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();`
    // `layout_map = logical_view_op.getLayoutMap();`
    let (logical_view_op, layout_map) = logical_view(view, scope)?;
    // `auto order_map = tmp_op.getStoreOrder();` / `getLoadOrder()`.
    let order_map = AffineMap::identity(view_ty.shape.len() as u32);
    // `indices_map = order_map.compose(indices_map);`
    let indices_map = order_map.compose(&indices_map);
    // `layout_map = layout_map.compose(indices_map);`
    let layout_map = layout_map.compose(&indices_map);
    Some(LayoutAndIndices {
        // `layout_map = compressUnusedSymbols(layout_map);`
        layout_map: layout_map.compress_unused_symbols(),
        operands,
        logical_view_op,
    })
}

/// THE `vector` ARMS OF [`layout_map_and_indices`] — the two that do not.
fn plain_access(base: Val, indices: &[Index], scope: &[DfirOp]) -> Option<LayoutAndIndices> {
    // `auto indices = tmp_op.getIndices(); operands = {indices.begin(), indices.end()};`
    //
    // ⛔ THE SUBSCRIPTS ARE THE OPERANDS THEMSELVES HERE, with no map to split them out of:
    // `Variadic<Index>:$indices` is an SSA list. ⭐ A LITERAL INDEX CONTRIBUTES NONE, because in
    // vendor MLIR it would be an `arith.constant` result and [`Index::Const`] is this island's
    // folded form of exactly that — so it is an operand the reference has and this island does not
    // spell, not a subscript being dropped.
    let (_, operands) = access_map(indices);
    // `logical_view_op = tmp_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();`
    // `layout_map = logical_view_op.getLayoutMap();` — and that is the whole arm.
    let (logical_view_op, layout_map) = logical_view(base, scope)?;
    Some(LayoutAndIndices {
        layout_map,
        operands,
        logical_view_op,
    })
}

/// THE VIEW A MEMORY ACCESS INDEXES — `getMemRef()`/`getBase()` followed by
/// `getDefiningOp<dataflow::GetLogicalMemoryViewOp>()`, and its `layout_map`.
///
/// ⛔ THE TEMPLATED `getDefiningOp<T>()` IS A `dyn_cast`, SO A BASE DEFINED BY ANYTHING ELSE IS NULL —
/// and the reference then calls `getLayoutMap()` on it. `None` is what this port answers instead; see
/// [`layout_map_and_indices`].
fn logical_view(base: Val, scope: &[DfirOp]) -> Option<(OpId, AffineMap)> {
    let at = defining_position(base, scope)?;
    match op_at(&at, scope)? {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { layout, .. }) => {
            Some((at, layout.clone()))
        }
        _ => None,
    }
}

/// SPLIT AN INDEX LIST BACK INTO THE `AffineMapAttr` AND THE OPERANDS MLIR KEEPS IT AS —
/// `getAffineMap()` beside `getMapOperands()`.
///
/// ⛔⛔ THE MAP IS BUILT WITH THE **SIMPLIFYING** OPERATORS, and that is not a shortcut. An
/// `affine_map` attribute in MLIR cannot be anything but canonical — the only ways to make one are the
/// parser and `AffineExpr`'s operators, both of which run `simplifyAdd`/`simplifyMul` — so
/// reconstructing `[%arg9]` as `d0 * 1 + 0` would hand [`AffineMap::compose`] a map the vendor op
/// does not carry. See [`AffineExpr::added`].
///
/// ⛔ ONE DIMENSION PER DISTINCT OPERAND, NUMBERED BY FIRST APPEARANCE, because that is what
/// `getMapOperands()` returns beside the map: a value used by two subscripts is one operand and one
/// `d<i>`. Numbering per *subscript* instead would declare a map with more dimensions than the access
/// has operands, and `compose`'s arity precondition would then be wrong for every caller. ⭐ AND ONE
/// CALLER TESTS EXACTLY THIS: `LoweringXRF::getLayoutExpr` guards its whole simplification on
/// `logical_view_map.getNumInputs() == operands.size()` (`LoweringXRF.cpp:38`) and silently skips it
/// otherwise, so a map with a dimension per subscript would take the untaken branch.
pub(super) fn access_map(indices: &[Index]) -> (AffineMap, Vec<Val>) {
    let mut operands: Vec<Val> = Vec::new();
    let mut results: Vec<AffineExpr> = Vec::new();

    for index in indices {
        results.push(match index {
            // `%arg1` — the subscript is the operand.
            Index::Val(val) => AffineExpr::Dim(dim_of(*val, &mut operands)),
            // `4` — a literal, which is a result and not an operand.
            Index::Const(constant) => AffineExpr::Const(*constant),
            // `%arg9 + %arg8 * 8 + 4` — the sum [`Index::Strided`] holds, term by term, with the
            // constant addend added LAST so `fold_add`'s `x + 0` rule can drop a zero one.
            Index::Strided(terms, addend) => {
                let mut sum: Option<AffineExpr> = None;
                for (val, stride) in terms {
                    let term = AffineExpr::Dim(dim_of(*val, &mut operands)).scaled(*stride);
                    sum = Some(match sum {
                        Some(so_far) => so_far.added(term),
                        None => term,
                    });
                }
                match sum {
                    Some(so_far) => so_far.added(AffineExpr::Const(*addend)),
                    None => AffineExpr::Const(*addend),
                }
            }
        });
    }

    let map = AffineMap {
        dims: operands.len() as u32,
        // ⭐ NONE, AND [`AffineMap::syms`] SAYS WHY: an access this bridge emits has no symbol, which
        // is also why `compressUnusedSymbols` has nothing to do at the end of an `agen` arm.
        syms: 0,
        results,
    };
    (map, operands)
}

/// WHICH `d<i>` A VALUE IS, ADDING IT TO THE OPERAND LIST THE FIRST TIME IT APPEARS.
fn dim_of(val: Val, operands: &mut Vec<Val>) -> u32 {
    match operands.iter().position(|held| *held == val) {
        Some(already) => already as u32,
        None => {
            operands.push(val);
            (operands.len() - 1) as u32
        }
    }
}

/// THE `Val` THE REUSE TABLE KEYS AN OPERAND'S PRODUCER BY — [`defining_position`]'s inverse.
///
/// ⭐ `OperandReuse::data_origins_` is keyed by `Operation *` in the reference and by [`Val`] here
/// (see [`OperandReuse::id`](super::vc_operand_reuse::OperandReuse::id)), so a [`VectorOperand`]'s
/// [`OpId`] has to be resolved to the value its op defines before the table can be asked about it.
///
/// ⛔ `None` FOR A POSITION THAT NAMES NO OP, OR AN OP THAT DEFINES NOTHING. The reference's `op_` is
/// always a live `Operation *` with a result, so neither is a case it has.
pub(super) fn origin_val(op: &OpId, scope: &[DfirOp]) -> Option<Val> {
    results(op_at(op, scope)?).first().copied()
}

/// WHERE THE OP THAT PRODUCES A VALUE LIVES — `Value::getDefiningOp()`.
///
/// ⛔ IT DESCENDS INTO REGIONS, for the reason [`use_positions`] gives in the other direction: a
/// value defined inside an `affine.for` body has a defining op, and a walk that stopped at the top
/// level would report none. ⭐ THE FIRST MATCH WINS because an SSA value has exactly one definition;
/// a second match would mean the scope handed in is not SSA.
///
/// ⛔ A BLOCK ARGUMENT HAS NO DEFINING OP AND ANSWERS `None`, exactly as MLIR's own accessor does —
/// `getDefiningOp()` returns null for one. A loop induction variable is the case that reaches this.
pub(super) fn defining_position(val: Val, scope: &[DfirOp]) -> Option<OpId> {
    find_defining_position(val, scope, &[], 0)
}

/// [`defining_position`]'s recursion, numbered the way [`op_at`] reads a path back.
fn find_defining_position(val: Val, scope: &[DfirOp], prefix: &[u32], base: u32) -> Option<OpId> {
    for (ordinal, op) in scope.iter().enumerate() {
        let mut path: Vec<u32> = prefix.to_vec();
        path.push(base + ordinal as u32);

        if results(op).contains(&val) {
            return Some(OpId::at(&path));
        }

        let mut child = 0u32;
        for region in regions(op) {
            if let Some(found) = find_defining_position(val, region, &path, child) {
                return Some(found);
            }
            child += region.len() as u32;
        }
    }
    None
}

impl VectorOperand {
    /// Replaces: e166_getOperandFromConstantOp
    ///
    /// # WHICH PSEUDO-PORT A SPLATTED VECTOR CONSTANT IS READ FROM
    ///
    /// ```cpp
    /// std::optional<VectorOperand> VectorOperand::getOperandFromConstantOp(
    ///     mlir::arith::ConstantOp &op) {
    ///   std::string value;
    ///   auto splat_attr = mlir::cast<SplatElementsAttr>(op.getValue());
    ///   if (splat_attr) {
    ///     double const_val;
    ///     auto splat_value = splat_attr.getSplatValue<Attribute>();
    ///     if (mlir::isa<IntegerAttr>(splat_value)) {
    ///       const_val = mlir::cast<IntegerAttr>(splat_value).getInt();
    ///     } else if (mlir::isa<FloatAttr>(splat_value)) {
    ///       const_val = mlir::cast<FloatAttr>(splat_value).getValueAsDouble();
    ///     } else {
    ///       op->emitError("Only integer or float vectors are supported");
    ///       return std::nullopt;
    ///     }
    ///
    ///     value = constValToField(const_val);
    ///     if (value == "") {
    ///       op->emitError("Only 0, 1, 2, or 3 are supported values");
    ///       return std::nullopt;
    ///     }
    ///   } else {
    ///     op->emitError("Only constant splatted vectors are supported");
    ///     return std::nullopt;
    ///   }
    ///
    ///   return VectorOperand(Constant, value, op.getOperation());
    /// }
    /// ```
    /// (`dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267-294`)
    ///
    /// # ⭐⭐ A CONSTANT OPERAND COSTS NO PORT — THE HARDWARE HAS FOUR OF THEM WIRED IN
    ///
    /// `dcc/test/PE/test1.mlir:38-39` declares `%cst_0 = arith.constant dense<1.000000e+00> :
    /// vector<64xf16>` and `%cst_1 = arith.constant dense<0.000000e+00>`, hands both to a
    /// `vectorchain.multiply_and_accumulate` (`:60`), and the reference's own `CHECK-SENT-IR` reads
    /// `opB = #sentient<compute_port one>, opC = #sentient<compute_port zero>` (`:18`). No load, no
    /// link, no register — the operand IS the port, which is why this function returns an operand
    /// rather than emitting anything.
    ///
    /// # ⛔ THE INTEGER/FLOAT SPLIT COLLAPSES, AND THE ISLAND IS WHY
    ///
    /// `IntegerAttr::getInt()` and `FloatAttr::getValueAsDouble()` (`:275-281`) exist because MLIR
    /// keeps two attribute kinds; both feed ONE `double` and one chain of `const_val == N` tests.
    /// [`arith::Op::DenseConstant::splat`] is a single `i64` for exactly this reason — see its own
    /// note for the census that every `arith.constant dense<…>` in the authority tree is integral —
    /// so *"Only integer or float vectors are supported"* names no third case here.
    ///
    /// ⛔ AND `if (value == "")` IS STATICALLY FALSE. `constValToField` ends in an unconditional
    /// `DT_ERROR` (`:260`), so its `return ""` and this arm are both dead in the reference; the
    /// domain is [`ConstantOperandValue`] and the refusal happens where the value is recognised. See
    /// [`ConstantOperandValue::of`].
    ///
    /// # THE TWO ABORTS, EACH DECLINED HERE
    ///
    /// * ⛔ A SPLAT OUTSIDE `0..=3` THROWS IN THE REFERENCE. `DT_ERROR` is not a diagnostic
    ///   (`util/dt_exception.hpp:110-121`); a `dense<4>` vector operand takes the compiler down. This
    ///   port answers `None`, which is the same refusal without the crash — and it is reachable, not
    ///   theoretical: `dense<4.000000e+00>` appears twice in the authority's tests.
    /// * ⛔ A **SCALAR** `arith.constant` IS `mlir::cast`'s OWN ASSERT (`:270`), and the caller
    ///   reaches this for any `isa<mlir::arith::ConstantOp>` (`VectorOperands.cpp:513-515`) — an
    ///   `arith.constant 3 : index` included. It then asks `getElementType` of that scalar type
    ///   (`:516`), which aborts as well. So the input is ill-formed twice over and `None` is the only
    ///   answer this crate can give; [`arith::Op::Constant`] and [`arith::Op::ConstantInt`] are named
    ///   rather than wildcarded so a fourth constant form has to decide.
    ///
    /// ⭐ THE TWO PRECISIONS ARE THE CALLER'S. `getOperand` sets `orig_precision_` and
    /// `on_the_fly_conv_precision_` from `getElementType(const_op.getType())` immediately after this
    /// returns (`:516-521`) — which is why `test1.mlir`'s golden also carries
    /// `opBPrecision = #sentient<precision fp16>` — and [`VectorOperand::new`] leaves both unset.
    #[must_use]
    pub fn from_constant_op(op: &arith::Op, at: OpId) -> Option<VectorOperand> {
        // `auto splat_attr = mlir::cast<SplatElementsAttr>(op.getValue());` — and the `else` arm
        // *"Only constant splatted vectors are supported"* that this cast makes unreachable.
        let splat = match op {
            arith::Op::DenseConstant { splat, .. } => *splat,
            arith::Op::Constant { .. } | arith::Op::ConstantInt { .. } => return None,
            // ⛔ NOT AN `arith.constant` AT ALL, and the reference could not be handed one: its
            // parameter is an `mlir::arith::ConstantOp&`. This island's [`arith::Op`] is one enum
            // over the whole dialect, so the six arithmetic forms are written out rather than
            // wildcarded — [`is_arith_constant`] draws the same line for [`same_block`].
            arith::Op::AddI(_)
            | arith::Op::SubI(_)
            | arith::Op::MulI(_)
            | arith::Op::DivSI(_)
            | arith::Op::RemSI(_)
            | arith::Op::Compare { .. }
            | arith::Op::Select { .. }
            // ⭐ THE TWO CONVERSIONS ARE HERE TOO: entry 343's `getOperandFromCastOp` is what reads
            // one, and it forwards to its INPUT's operand rather than reading a splat.
            | arith::Op::SiToFp(_)
            | arith::Op::FpToSi(_)
            | arith::Op::Logic { .. } => return None,
        };

        // `const_val` — either attribute kind reaches the same number — and
        // `value = constValToField(const_val);`.
        let value = const_val_to_field(ConstantOperandValue::of(splat)?);

        // `return VectorOperand(Constant, value, op.getOperation());`
        Some(VectorOperand::new(
            VectorOperandType::Constant,
            OperandValue::Port(value),
            at,
        ))
    }
}

/// WHICH UNIT A LOGICAL MEMORY VIEW IS TAKEN OVER — `findUnitType(logical_view_op.getFromUnit())`
/// (`VectorOperands.cpp:174-175`), whose answer the reference keeps as a string.
///
/// ⛔ TWO OPS DEFINE ONE, and entry 232 reads both: a register file is a `dataflow.get_local_unit`
/// ([`dataflow::LocalUnit`]) while `pestate` and `sfpstate` are `dataflow.get_unit`s in their own
/// right ([`DfirUnit`]) — the split [`dataflow::LocalUnit`]'s closing note records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewUnit {
    /// A `dataflow.get_local_unit` — one of the register files.
    Local(dataflow::LocalUnit),
    /// A `dataflow.get_unit` — a unit a transfer can address.
    Unit(DfirUnit),
}

/// [`ViewUnit`] FOR THE VALUE A VIEW IS TAKEN FROM. `None` where the value is defined by neither op,
/// which is the reference's *"Unit type is inconsistent in memory view"*.
fn view_unit(from: Val, scope: &[DfirOp]) -> Option<ViewUnit> {
    match op_at(&defining_position(from, scope)?, scope)? {
        DfirOp::Dataflow(dataflow::Op::GetLocalUnit { which, .. }) => Some(ViewUnit::Local(*which)),
        DfirOp::Dataflow(dataflow::Op::GetUnit { unit, .. }) => Some(ViewUnit::Unit(*unit)),
        _ => None,
    }
}

/// WHICH FILE AN ACCESS'S UNIT IS — entry 232's four `operand_type` assignments
/// (`VectorOperands.cpp:186-198`).
///
/// ⛔ THE XRF IS SPLIT OFF BECAUSE IT HAS NO SLICE INDEX: the reference returns
/// `VectorOperand(XRF, "xrf", op)` before the slice arithmetic and skips the single-constant check
/// with it (`:209`, `:219`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryFile {
    /// `PTXRF → XRF`.
    Xrf,
    /// The three files whose value is a slice index.
    Sliced(SliceFile),
}

/// A FILE WHOSE OPERAND VALUE IS A SLICE INDEX — see [`RegisterSlice`], which is one of its slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SliceFile {
    /// Any of the three `*_lrfreg`s.
    Lrf,
    /// `ptirf`.
    Irf,
    /// `pestate` or `sfpstate`.
    IState,
}

impl SliceFile {
    /// The `operand_type` the reference assigns for it.
    const fn kind(self) -> VectorOperandType {
        match self {
            SliceFile::Lrf => VectorOperandType::Lrf,
            SliceFile::Irf => VectorOperandType::Irf,
            SliceFile::IState => VectorOperandType::IState,
        }
    }
}

/// THE `stringToSenComponents` LOOKUP AND ITS FOUR ARMS. `None` is both *"Unknown memory type"*
/// exits — the name that is in the table but is none of the four (`ptarf`, `l0scale`), and the one
/// that is not in it at all.
fn memory_file(unit: ViewUnit) -> Option<MemoryFile> {
    match unit {
        // `if (record->second == PTXRF) operand_type = XRF;`
        ViewUnit::Local(dataflow::LocalUnit::PtXrf) => Some(MemoryFile::Xrf),
        // `else if (record->second == PTIRF) operand_type = IRF;`
        ViewUnit::Local(dataflow::LocalUnit::PtIrf) => Some(MemoryFile::Sliced(SliceFile::Irf)),
        // `else if (dcc::utils::isLRFReg(memory_unit_name)) operand_type = LRF;` — ⭐ THE ONLY
        // PREDICATE HERE THAT READS THE NAME AND NOT THE ENUMERATOR, and these three are what spell
        // `lrf` in it (`pe_lrfreg`, `sfp_lrfreg`, `pt_lrfreg`).
        ViewUnit::Local(
            dataflow::LocalUnit::PeLrf | dataflow::LocalUnit::SfpLrf | dataflow::LocalUnit::PtLrf,
        ) => Some(MemoryFile::Sliced(SliceFile::Lrf)),
        // `else if (is_any_of(record->second, PESTATE, SFPSTATE)) operand_type = ISTATE;`
        ViewUnit::Unit(DfirUnit::PeState | DfirUnit::SfpState) => {
            Some(MemoryFile::Sliced(SliceFile::IState))
        }
        // `else { op->emitError("Unknown memory type"); return std::nullopt; }`
        ViewUnit::Local(dataflow::LocalUnit::PtArf | dataflow::LocalUnit::L0Scale)
        | ViewUnit::Unit(_) => None,
    }
}

/// WHICH SLICE OF WHICH FILE A COMPUTED INDEX IS.
///
/// ⛔⛔ `None` IS THE REFERENCE'S OWN DEATH AND NOT A CHECK ADDED HERE. The value it builds is a
/// decimal string, and every reader of it calls `symbolizeSentientComputePort("lrf" + value).value()`
/// — which is `std::nullopt` for `lrf32` because the `.td` declares thirty-two cases
/// (`SentientTypes.td:98-160`). ⭐ AND IT IS NOT [`sen::LrfIndex`]`::checked` REBORN: the bound
/// belongs to the FILE, so it is stated once here, at the single site that computes an index, rather
/// than on the type where every holder of one would have to re-handle it (see [`RegisterSlice`]).
fn register_slice(file: SliceFile, access: i64) -> Option<RegisterSlice> {
    match file {
        SliceFile::Lrf => Some(RegisterSlice::Lrf(match access {
            0 => sen::LrfIndex::L0,
            1 => sen::LrfIndex::L1,
            2 => sen::LrfIndex::L2,
            3 => sen::LrfIndex::L3,
            4 => sen::LrfIndex::L4,
            5 => sen::LrfIndex::L5,
            6 => sen::LrfIndex::L6,
            7 => sen::LrfIndex::L7,
            8 => sen::LrfIndex::L8,
            9 => sen::LrfIndex::L9,
            10 => sen::LrfIndex::L10,
            11 => sen::LrfIndex::L11,
            12 => sen::LrfIndex::L12,
            13 => sen::LrfIndex::L13,
            14 => sen::LrfIndex::L14,
            15 => sen::LrfIndex::L15,
            16 => sen::LrfIndex::L16,
            17 => sen::LrfIndex::L17,
            18 => sen::LrfIndex::L18,
            19 => sen::LrfIndex::L19,
            20 => sen::LrfIndex::L20,
            21 => sen::LrfIndex::L21,
            22 => sen::LrfIndex::L22,
            23 => sen::LrfIndex::L23,
            24 => sen::LrfIndex::L24,
            25 => sen::LrfIndex::L25,
            26 => sen::LrfIndex::L26,
            27 => sen::LrfIndex::L27,
            28 => sen::LrfIndex::L28,
            29 => sen::LrfIndex::L29,
            30 => sen::LrfIndex::L30,
            31 => sen::LrfIndex::L31,
            _ => return None,
        })),
        SliceFile::Irf => Some(RegisterSlice::Irf(match access {
            0 => IrfIndex::I0,
            1 => IrfIndex::I1,
            _ => return None,
        })),
        SliceFile::IState => Some(RegisterSlice::IState(match access {
            0 => sen::IStateIndex::S0,
            1 => sen::IStateIndex::S1,
            2 => sen::IStateIndex::S2,
            3 => sen::IStateIndex::S3,
            _ => return None,
        })),
    }
}

/// THE ACCESS'S OWN SUBSCRIPTS AS A MAP — what `fullyComposeAffineMapAndOperands` substitutes into
/// the layout, and `None` for the two `agen` arms, where [`layout_map_and_indices`] has already
/// composed them.
///
/// ⛔⛔ WITHOUT IT A LITERAL-SUBSCRIPTED `vector.load` HAS NO OFFSET AT ALL. [`plain_access`] hands
/// back the VIEW's map and drops [`Index::Const`] subscripts, because the view's dimensions are entry
/// 170's contract; in the reference those subscripts are `arith.constant` results and folding them in
/// is exactly what makes `isSingleConstant()` true for `%lrf_memory_fp16[%c4, %c0]`
/// (`sfp-to-sfp-ring.mlir:156`, `:171`).
///
/// ⭐ AND `canonicalizeMapAndOperands` NEEDS NOTHING BEHIND IT HERE: it drops unused dimensions and
/// folds, and [`AffineMap::compose`] has already folded what this caller reads out of the map.
fn subscript_map(op: &OpId, scope: &[DfirOp]) -> Option<AffineMap> {
    match op_at(op, scope)? {
        DfirOp::Vector(vector::Op::Load { indices, .. } | vector::Op::Store { indices, .. }) => {
            Some(access_map(indices).0)
        }
        _ => None,
    }
}

/// A CONSTANT INDEX'S VALUE — `dyn_cast<mlir::arith::ConstantIndexOp>(val.getDefiningOp()).value()`,
/// with `None` for the null cast the reference dereferences.
fn constant_index(val: Val, scope: &[DfirOp]) -> Option<i64> {
    match op_at(&defining_position(val, scope)?, scope)? {
        DfirOp::Arith(arith::Op::Constant { value, .. }) => Some(*value),
        _ => None,
    }
}

/// HOW MANY BITS ONE SLICE OF A REGISTER FILE HOLDS — `int offset_per_slice = 1024; // bits`
/// (`VectorOperands.cpp:223`).
const SLICE_BITS: i64 = 1024;

impl VectorOperand {
    /// Replaces: e232_getOperandFromLoadOrStoreOp
    ///
    /// WHICH REGISTER-FILE SLICE A LOAD OR STORE TOUCHES — the access's constant-folded 1-D offset
    /// scaled into 1024-bit slices, or `xrf`, which has no index.
    ///
    /// ⛔ TRAP: the two `bit_width` workarounds are part of the address (80 → 8, 24 → 16), and the
    /// START ADDRESS IS IN BYTES yet is added to an ELEMENT offset before the scaling — the
    /// reference's own arithmetic, comment and all.
    /// ⛔ `None` COVERS ITS SEVEN `return std::nullopt` SITES AND ONE MORE: a slice index the file has
    /// no case for, which is where the reference dies instead (see [`register_slice`]).
    #[must_use]
    pub fn from_load_or_store_op(op: &OpId, scope: &[DfirOp]) -> Option<VectorOperand> {
        // `if (failed(getLayoutMapAndIndices(op, layout_map, indices, logical_view_op)))` — "op needs
        // to be either vector load or store".
        let access = layout_map_and_indices(op, scope)?;
        // `if (layout_map.getNumResults() != 1)` — "should map to a 1D view of memory for LRF/XRF/IRF".
        if access.layout_map.results.len() != 1 {
            return None;
        }
        let DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            from, start, ty, ..
        }) = op_at(&access.logical_view_op, scope)?
        else {
            return None;
        };
        // `findUnitType(logical_view_op.getFromUnit())`, else "Unit type is inconsistent".
        let unit = view_unit(*from, scope)?;
        // `affine::fullyComposeAffineMapAndOperands(&layout_map, &indices);` and the guarded
        // `canonicalizeMapAndOperands` — see [`subscript_map`] for both.
        let layout_map = match subscript_map(op, scope) {
            Some(subscripts) => access.layout_map.compose(&subscripts),
            None => access.layout_map,
        };

        // `// find regfile type` — the lookup and its four arms.
        match memory_file(unit)? {
            // `if (operand_type == XRF) return VectorOperand(operand_type, "xrf", op);`
            MemoryFile::Xrf => Some(VectorOperand::new(
                VectorOperandType::Xrf,
                OperandValue::Port(sen::Port::Xrf),
                op.clone(),
            )),
            MemoryFile::Sliced(file) => {
                // `if (operand_type != XRF && !layout_map.isSingleConstant())` — "Store indices
                // should lead to constant indices in the 1D memory for LRF/IRF".
                let offset = layout_map.single_constant()?;
                // `auto bit_width = getElementTypeBitWidth(logical_view_op.getResult().getType());`
                let bit_width = match i64::from(ty.elem.bits()) {
                    // `// TODO: this if statement must be dropped due to F80Type workaround`
                    80 => 8,
                    24 => 16,
                    bits => bits,
                };
                // `if (auto const_start_address = dyn_cast<arith::ConstantIndexOp>(
                //        logical_view_op.getStartAddress().getDefiningOp()))`, else "Only constant
                // start address are supported for LRF/XRF/IRF/ISTATE in PT".
                let start_address = constant_index(*start, scope)?;
                // `auto total_address = (start_address + layout_map.getSingleConstantResult()) *
                //  bit_width; auto access = (total_address) / offset_per_slice;`
                let access = (start_address + offset) * bit_width / SLICE_BITS;
                // `return VectorOperand(operand_type, std::to_string(access), op);`
                Some(VectorOperand::new(
                    file.kind(),
                    OperandValue::Slice(register_slice(file, access)?),
                    op.clone(),
                ))
            }
        }
    }
}

/// WHETHER AN OP IS ONE OF THE FIVE THAT CAN SIT BETWEEN AN OPERAND AND THE COMPUTE — the `isa<>`
/// list of [`erase_operands`] (`VectorOperands.cpp:705-707`).
pub(super) fn is_intermediate(op: &OpId, scope: &[DfirOp]) -> bool {
    matches!(
        op_at(op, scope),
        Some(
            DfirOp::Arith(arith::Op::SiToFp(_) | arith::Op::FpToSi(_))
                | DfirOp::VectorChain(
                    vc::Op::Cast { .. } | vc::Op::Neg { .. } | vc::Op::Select { .. }
                )
        )
    )
}

/// Replaces: e233_eraseOperands
///
/// ERASE THE OPS FEEDING EVERY OPERAND WHOSE CHAIN THIS LOWERING CONSUMED, STEPPING OVER THE
/// CONVERSION, CAST, NEGATION OR SELECT THAT CAN SIT IN BETWEEN (see [`is_intermediate`]).
///
/// ⛔ A `latch` OPERAND IS NEVER ERASED (`getName() != "latch"`), one surviving reader anywhere keeps
/// the definition, and an op already erased is skipped — so one op feeding two operands goes once.
/// ⛔ THE REFERENCE READS `operand.value()` BEFORE ITS OWN `has_value()` TEST (`:693-696`), which is
/// a null dereference; the `Option` is taken first here.
/// ⛔ AND EVERY REMOVAL WAITS FOR THE END, DESCENDING: an [`OpId`] is a POSITION, so erasing one op
/// renumbers the siblings that the remaining decisions name (see [`erase_op`]).
pub fn erase_operands(operands: &[Option<VectorOperand>], scope: &mut Vec<DfirOp>) {
    let mut erased_list: Vec<OpId> = Vec::new();

    // `for (auto &operand : operands)`, with `if (!operand.has_value()) continue;` first.
    for operand in operands.iter().flatten() {
        // `if (std::find(erased_list.begin(), erased_list.end(), operand.value().op_) == end())`
        if erased_list.contains(&operand.op) {
            continue;
        }

        let mut all_uses_deleted = true;
        let mut to_be_erased: Vec<OpId> = Vec::new();

        // `for (auto user : operand.value().op_->getUsers())`
        for mut user in use_positions(&operand.op, scope) {
            // ⚠️ DELIBERATE DIVERGENCE: the reference declares `intermediate_ops` OUTSIDE this loop
            // (`:699`) and never clears it, so one user's chain is re-erased for every later user and
            // the chain of a user that still has readers is erased anyway. Per-user is the intent.
            let mut intermediate_ops: Vec<OpId> = Vec::new();

            // `while (user && isa<..>(user) && user->hasOneUse()) { intermediate_ops.push_back(user);
            //  user = *user->getUsers().begin(); }`
            while is_intermediate(&user, scope) {
                let mut users = use_positions(&user, scope);
                if users.len() != 1 {
                    break;
                }
                intermediate_ops.push(user);
                user = users.remove(0);
            }

            // `if (user && user->getUses().empty()) { user->dropAllUses(); to_be_erased.push_back(
            //  user); for (auto op : intermediate_ops) { .. } } else all_uses_deleted = false;`
            if has_no_uses(&user, scope) {
                to_be_erased.push(user);
                to_be_erased.extend(intermediate_ops);
            } else {
                all_uses_deleted = false;
            }
        }

        // `for (auto e : to_be_erased) { erased_list.push_back(e); e->erase(); }`
        erased_list.extend(to_be_erased);

        // `if (all_uses_deleted) { if (operand.value().getName() != "latch") { erased_list.push_back(
        //  operand.value().op_); operand.value().op_->erase(); } }`
        if all_uses_deleted && operand.name() != Some(sen::Port::Latch) {
            erased_list.push(operand.op.clone());
        }
    }

    // ⚠️ DELIBERATE DIVERGENCE: the reference erases inside the loop, so a later operand's
    // `getUsers()` no longer sees a use from an already-erased op; here the scheduled positions are
    // still visible, which can only make `has_no_uses` stricter — never erase more.
    erased_list.sort_unstable();
    erased_list.dedup();
    for position in erased_list.iter().rev() {
        remove_at(position.path(), scope);
    }
}

/// EVERY USE OF AN OP'S RESULTS THAT AN ALREADY-CLAIMED OP DOES NOT ACCOUNT FOR — `setOperands({})`
/// on each op the rewriter is about to erase (`VectorOperands.cpp:782-789`, `:848-852`).
///
/// ⛔⛔ THIS IS WHY THE REFERENCE CLEARS OPERANDS AND NOT AN OPTIMISATION OF IT. Its own note says
/// so: *"Since the rewriter does not immediately erase operations, then the uses of the operands in
/// an operation to be deleted persist. We circumvent this problem by deleting the operands of any
/// operation to be deleted."* A phase that could still see a claimed op's uses would decide the op
/// feeding it still has readers and leave the whole source chain in the unit.
fn live_use_positions(of: &OpId, scope: &[DfirOp], erased: &[OpId]) -> Vec<OpId> {
    use_positions(of, scope)
        .into_iter()
        .filter(|user| !erased.contains(user))
        .collect()
}

/// [`has_no_uses`] over the uses [`live_use_positions`] leaves — `user->getUses().empty()` after the
/// claimed ops dropped theirs. ⛔ `false` FOR A POSITION THAT NAMES NO OP, as [`has_no_uses`].
fn has_no_live_uses(id: &OpId, scope: &[DfirOp], erased: &[OpId]) -> bool {
    op_at(id, scope).is_some() && live_use_positions(id, scope, erased).is_empty()
}

/// THE `erased_list` OVERLOAD OF `VectorOperand::eraseOperands` (`VectorOperands.cpp:742-800`) —
/// [`erase_operands`]'s walk, with every skip test taken against a list its CALLER owns.
///
/// ⛔ NOT AN ANCHORED UNIT: this is one of the scope holes `docs/bridge2-porting-order.md` records,
/// and the overload `e280_cleanup` (`VectorChainToSentientPESFP.cpp:1062`) actually calls.
/// ⛔ IT RECORDS AND DOES NOT REMOVE, because an [`OpId`] is a POSITION: a caller running three
/// phases over one scope cannot let phase one renumber what phases two and three name. The claimed
/// ops stop being readers all the same — see [`live_use_positions`].
pub(super) fn erase_operands_recording(
    operands: &[Option<VectorOperand>],
    scope: &[DfirOp],
    erased_list: &mut Vec<OpId>,
) {
    // `for (auto &operand : operands)` — ⛔ the reference reads `operand.value().op_` in the
    // `std::find` BEFORE its own `has_value()` test (`:747-749`); the `Option` is taken first here.
    for operand in operands.iter().flatten() {
        // `if (std::find(erased_list.begin(), erased_list.end(), operand.value().op_) == end())`
        if erased_list.contains(&operand.op) {
            continue;
        }

        let mut all_uses_deleted = true;
        let mut to_be_erased: Vec<OpId> = Vec::new();

        for mut user in live_use_positions(&operand.op, scope, erased_list) {
            // ⚠️ DELIBERATE DIVERGENCE, as in [`erase_operands`]: the reference declares
            // `intermediate_ops` outside this loop (`:756`) and never clears it.
            let mut intermediate_ops: Vec<OpId> = Vec::new();

            // `while (user && isa<..>(user) && user->hasOneUse()) { .. }`
            while is_intermediate(&user, scope) {
                let mut users = live_use_positions(&user, scope, erased_list);
                if users.len() != 1 {
                    break;
                }
                intermediate_ops.push(user);
                user = users.remove(0);
            }

            if has_no_live_uses(&user, scope, erased_list) {
                // ⛔ THE EXTRA TEST THIS OVERLOAD HAS: a user another phase already claimed is not
                // claimed again (`:768-769`) — ⭐ and its intermediates are then not claimed either,
                // which is the reference's behaviour and not a shortcut. `all_uses_deleted` stays
                // true for it, since the `else` is only for a user that still has readers.
                if !erased_list.contains(&user) {
                    to_be_erased.push(user);
                    to_be_erased.extend(intermediate_ops);
                }
            } else {
                all_uses_deleted = false;
            }
        }

        // `for (auto e : to_be_erased) { e->setOperands({}); erased_list.push_back(e); .. }` — ⭐
        // AFTER the user loop, so a user decided earlier in this operand's loop did not see them.
        erased_list.extend(to_be_erased);

        // `if (all_uses_deleted) { if (operand.value().getName() != "latch") { .. } }`
        if all_uses_deleted && operand.name() != Some(sen::Port::Latch) {
            erased_list.push(operand.op.clone());
        }
    }
}

/// THE `erased_list` OVERLOAD OF `VectorOperand::eraseOp` (`VectorOperands.cpp:825-857`) —
/// [`erase_op`]'s walk, recording instead of removing, for the reason above.
///
/// ⛔ NOT AN ANCHORED UNIT, same scope hole. ⭐ THIS ONE IS WHERE THE REFERENCE DEDUPLICATES — the
/// `std::find` against BOTH `to_be_erased` and `erased_list` (`:837-841`) that [`erase_op`] had to
/// decide for itself. ⛔ AND `op` GOES IN UNCONDITIONALLY WHEN EVERY USE WAS ERASABLE: no latch test
/// and no dedup test on `op` itself, unlike the operand case.
pub(super) fn erase_op_recording(op: &OpId, scope: &[DfirOp], erased_list: &mut Vec<OpId>) {
    let mut all_uses_deleted = true;
    let mut to_be_erased: Vec<OpId> = Vec::new();

    for user in live_use_positions(op, scope, erased_list) {
        if has_no_live_uses(&user, scope, erased_list) {
            if !to_be_erased.contains(&user) && !erased_list.contains(&user) {
                to_be_erased.push(user);
            }
        } else {
            all_uses_deleted = false;
        }
    }

    erased_list.extend(to_be_erased);

    if all_uses_deleted {
        erased_list.push(op.clone());
    }
}

/// `getExpandedVector(getShuffleIndicesAsVector(op), repetition)` — the index list repeated
/// `repetition` times (`dialect_utils/VectorChain/Utils.cpp:50-73`).
///
/// ⛔ NOT ONE OF THE 384, and the three classifiers below are not either — they are
/// `dialect_utils`, which bridge 2's list excludes. They are here because entry 278 is only its three
/// branch tests, so a port without them would be the predicate-shaped non-port `TASK.md` forbids.
///
/// ⛔ AN INVALID INDEX LIST IS **EMPTY**, NOT REJECTED (`:52`), which makes every classifier's loop
/// vacuous and answers `true`. This island's `indices` is a `Vec<i32>` and `isShuffleIndicesValid`'s
/// only real test is `index >= -(vars + pads)` (`:28-31`) — ⛔ BOTH SEGMENTS, which is why entry 279
/// adding [`vc::Op::Shuffle::pad`](vc::Op::Shuffle) widens the bound here rather than leaving it.
pub(super) fn expanded_shuffle_indices(
    indices: &[i32],
    repetition: u32,
    variables: usize,
    pads: usize,
) -> Vec<i32> {
    let max_negative = -i32::try_from(variables + pads).unwrap_or(i32::MAX);
    if indices.iter().any(|index| *index < max_negative) {
        return Vec::new();
    }
    indices.repeat(repetition as usize)
}

/// `isShuffleNFWDVersion0` (`dialect_utils/VectorChain/Utils.cpp:135-152`) — ⛔ THE 32-LANE LIST IS
/// THE FP32 PATTERN AND EVERY OTHER LENGTH IS TREATED AS FP16, including an empty one, which passes.
fn is_shuffle_nfwd_version_0(expanded: &[i32]) -> bool {
    let pattern: &[i32] = if expanded.len() == 32 {
        &[1, 0, 3, 3]
    } else {
        &[2, 3, 0, 1, 6, 7, 6, 7]
    };
    expanded
        .iter()
        .enumerate()
        .all(|(i, index)| *index == pattern[i % pattern.len()])
}

/// `isShuffleNFWDVersion2` (`dialect_utils/VectorChain/Utils.cpp:155-172`) — the same shape with the
/// other two patterns.
fn is_shuffle_nfwd_version_2(expanded: &[i32]) -> bool {
    let pattern: &[i32] = if expanded.len() == 32 {
        &[2, 0, 1, 3]
    } else {
        &[4, 5, 3, 3, 5, 5, 6, 7]
    };
    expanded
        .iter()
        .enumerate()
        .all(|(i, index)| *index == pattern[i % pattern.len()])
}

/// `isCustomVectorTrivialShuffle` (`dialect_utils/VectorChain/Utils.cpp:126-133`) — one index, zero,
/// a custom-vector result, and a constant bitstream feeding it.
///
/// ⛔ `isa<dataflow::CustomVectorType>` IS THE ELEMENT TYPE HERE. This island states a vector as
/// [`crate::islands::dataflow_ir::ty::Vector`] and carries the custom-ness on the ELEMENT
/// ([`ElemType::MxFloat`], [`ElemType::MxInt`]) rather than as a distinct type constructor; every
/// `!dataflow.custom_vector` in the authority tree's 825 cases has an MX element, and MLIR's builtin
/// `VectorType` cannot hold one (`DataflowTypes.td:30`), so the two spellings coincide.
fn is_custom_vector_trivial_shuffle(
    indices: &[i32],
    ty: Vector,
    input: Val,
    scope: &[DfirOp],
) -> Option<OpId> {
    // `if (indices.size() != 1 || indices[0] != 0) return false;` — ⛔ THE UNEXPANDED LIST.
    if indices != [0] {
        return None;
    }
    if !matches!(ty.elem, ElemType::MxFloat(_) | ElemType::MxInt(_)) {
        return None;
    }
    // `return isa<vectorchain::ConstantBitstreamOp>(parent);` — and the caller's `DT_CHECK` with it.
    let parent = defining_position(input, scope)?;
    match op_at(&parent, scope)? {
        DfirOp::VectorChain(vc::Op::ConstantBitstream { .. }) => Some(parent),
        _ => None,
    }
}

/// WHICH UNIT A `dataflow.send` GOES TO — `findUnitType(send_op.getToUnit())`, which is what
/// [`VectorOperand::from_send_op`] left to its caller. `None` is *"Unit type is inconsistent in
/// SendOp."*
fn send_destination(to: link::SendEnd, scope: &[DfirOp]) -> Option<DfirUnit> {
    unit_behind(to.val(), scope)
}

/// WHICH UNIT A `dataflow.receive` COMES FROM — the twin [`VectorOperand::from_receive_op`] left to
/// ITS caller, and how [`VectorOperand::with_precision`] reaches it. `None` is *"Unit type is
/// inconsistent in ReceiveOp."*
fn recv_source(from: link::RecvEnd, scope: &[DfirOp]) -> Option<DfirUnit> {
    unit_behind(from.val(), scope)
}

/// `findUnitType` (`dcc/src/Dialect/Uniform/Utils.cpp:286-301`) — WHICH UNIT A VALUE NAMING A
/// TRANSFER'S PEER STANDS FOR.
///
/// # ⛔⛔ THE `uniform.query_map` ARM IS THE UNIFORMIZED CASE AND DECLINING IT ANSWERS NOTHING
///
/// An earlier note here said that arm was unreachable from this island. It is not: a uniformized
/// transfer's peer IS a `query_map`, and `mixed_precision.mlir:745` receives from
/// `%195 = uniform.query_map(map:%194, key:%arg0)` rather than from a `get_unit`
/// (`dcc/test/Conversion/VectorChainToSentientPESFP/mixed_precision.mlir:628-629`, `:745`). With the
/// arm missing, every operand of every compute in that fixture — the vendor's own — was *"unit type
/// is inconsistent"*.
///
/// ⛔ `getUnitTypeFromUniformMappingAsString` READS THE MAPPING'S **VALUES**, NOT ITS KEYS
/// (`Utils.cpp:268`): the keys are the core handles a `query_map` is keyed BY, and the values are the
/// peer units. And it reads `getValues()[0]` ONLY, on the strength of its own `TODO` that agreement
/// between the entries *"should be left to canonicalization"* (`:260-261`) — so one entry decides the
/// type for every core, and this port does not check the rest either.
///
/// ⛔ ITS `if (unit_type.empty())` GUARDS A STRING ASSIGNED `""` TWO LINES ABOVE — always true. What
/// that makes reachable is the trailing `return unit_type` for a first value defined by NEITHER op,
/// where the reference hands back an EMPTY name instead of `std::nullopt`; both miss
/// `stringToSenComponents.find`, so `None` is the same answer.
///
/// ⛔ A REGISTER FILE IS NOT A TRANSFER PEER. The `get_local_unit` arm answers a name like `ptxrf`
/// and no [`DfirUnit`] spells one, which is the reference's *"Unknown receiver"*/*"Unknown
/// destination"* — see [`ViewUnit`], whose split is the same one from the memory-view side.
fn unit_behind(val: Val, scope: &[DfirOp]) -> Option<DfirUnit> {
    match op_at(&defining_position(val, scope)?, scope)? {
        // `if (auto unit = getDefiningOp<dataflow::GetUnitOp>()) return unit.getType().str();`
        DfirOp::Dataflow(dataflow::Op::GetUnit { unit, .. }) => Some(*unit),
        DfirOp::Uniform(uniform::Op::QueryMap { map, .. }) => {
            // `if (!def_map_op) return std::nullopt; if (def_map_op.getValues().empty()) ..`
            let mapping = op_at(&defining_position(*map, scope)?, scope)?;
            let DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. }) = mapping else {
                return None;
            };
            let (_, first_value) = pairs.first()?;
            // ⛔ ONE LEVEL ONLY: the reference resolves that value as a `get_unit` or a
            // `get_local_unit` and does not ask a second `query_map`.
            match op_at(&defining_position(*first_value, scope)?, scope)? {
                DfirOp::Dataflow(dataflow::Op::GetUnit { unit, .. }) => Some(*unit),
                _ => None,
            }
        }
        _ => None,
    }
}

impl VectorOperand {
    /// Replaces: e278_getOperandFromShuffleOp
    ///
    /// **278/384** `VectorOperand::getOperandFromShuffleOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311` (36L).
    ///
    /// Three recognised shuffle shapes answer with the operand of what the shuffle READS, re-tagged;
    /// anything else answers with the operand of the shuffle's single USER.
    ///
    /// ⛔ THE RE-TAG IS AN OVERWRITE OF AN ANSWER ALREADY BUILT (`:317-319`), so the NFWD operand
    /// keeps the parent's position and precisions and loses only its value and kind.
    /// ⛔ THE TRIVIAL-SHUFFLE ARM TAKES `is_constant_splatted_vector = true` — the only caller that
    /// does — and then overwrites `Constant` with `ConstantBitstream` (`:333-335`).
    /// ⛔ ONLY THE **FIRST** USER IS INSPECTED (`:339-340`); program order is the order here.
    /// ⭐ `get_operand` IS THE SAME SCC CUT [`Self::from_neg_op`] takes, `traverse_upwards = true`.
    #[must_use]
    pub fn from_shuffle_op<A: Arch>(
        op: &OpId,
        comp: ComputeComp,
        scope: &[DfirOp],
        get_operand: &mut impl FnMut(&OpId, ComputeComp) -> Option<VectorOperand>,
    ) -> Option<VectorOperand> {
        // `DT_CHECK(isa<vectorchain::ShuffleOp>(op));`
        let Some(DfirOp::VectorChain(vc::Op::Shuffle {
            input,
            variable,
            pad,
            indices,
            repetition,
            ty,
            ..
        })) = op_at(op, scope)
        else {
            return None;
        };
        let expanded = expanded_shuffle_indices(indices, *repetition, variable.len(), pad.len());

        // The two NFWD arms, which differ only in the port they re-value to.
        let nfwd = if is_shuffle_nfwd_version_0(&expanded) {
            Some(sen::Port::Nfwd0)
        } else if is_shuffle_nfwd_version_2(&expanded) {
            Some(sen::Port::Nfwd2)
        } else {
            None
        };
        if let Some(port) = nfwd {
            // `auto *parent = op.getOperand(0).getDefiningOp();`
            let parent = defining_position(*input, scope)?;
            let mut operand = get_operand(&parent, comp)?;
            operand.kind = VectorOperandType::Nfwd;
            operand.set_value(OperandValue::Port(port));
            return Some(operand);
        }

        // `} else if (vectorchain::utils::isCustomVectorTrivialShuffle(op)) {`
        if let Some(parent) = is_custom_vector_trivial_shuffle(indices, *ty, *input, scope) {
            let Some(DfirOp::VectorChain(vc::Op::ConstantBitstream { value, .. })) =
                op_at(&parent, scope)
            else {
                return None;
            };
            // `getOperandFromConstantBitstreamOp(const_bit_op, /*splatted*/ true)`, whose
            // `getValue()[0]` is element zero.
            let splat = ConstantOperandValue::of(*value.first()?)?;
            let mut operand = VectorOperand::from_constant_bitstream_op(
                BitstreamConstant::SplattedVector(splat),
                parent,
            );
            operand.kind = VectorOperandType::ConstantBitstream;
            return Some(operand);
        }

        // `if (op.getOperation()->user_begin() != op.getOperation()->user_end())`
        let user = use_positions(op, scope).into_iter().next()?;
        match op_at(&user, scope)? {
            // `if (auto send_op = llvm::dyn_cast<dataflow::SendOp>(use))`
            DfirOp::Dataflow(dataflow::Op::Send { to, .. }) => Some(
                VectorOperand::from_send_op::<A>(send_destination(*to, scope)?, comp, user),
            ),
            // `else if (auto store_op = llvm::dyn_cast<agen::VectorStoreOp>(use))`
            DfirOp::Agen(agen::Op::VectorStore { .. }) => {
                VectorOperand::from_load_or_store_op(&user, scope)
            }
            // `return std::nullopt;` — both of them.
            _ => None,
        }
    }
}

/// `operand.orig_precision_ = getPrecisionInString(elem_type); operand.on_the_fly_conv_precision_ =
/// operand.orig_precision_;` — the pair eleven of [`VectorOperand::with_precision`]'s arms write.
fn set_both_precisions(operand: &mut VectorOperand, elem: ElemType) {
    operand.orig_precision = Some(precision_in_string(elem));
    operand.on_the_fly_conv_precision = operand.orig_precision;
}

/// THE PRECISION DATA OFF THE `pt` PORT ARRIVES AT — *"Input from PT is int16/fp16 in DD2 and
/// int24/fp24 in Sen1p5"* (`VectorOperands.cpp:446-453`), which OVERRIDES the element type the
/// receive itself carries.
///
/// ⛔ `isIntOrIndex()` IS TRUE FOR AN `IntegerType` ONLY, so an MX-int element takes the FLOAT arm:
/// `CustomMXIntType` is not an integer type to MLIR and the reference's `else` is unconditional.
const fn pt_receive_precision(elem: ElemType, isa: IsaGen) -> sen::Precision {
    match (elem, isa) {
        (ElemType::Int(_), IsaGen::Sen1p5) => sen::Precision::Int24,
        (ElemType::Int(_), IsaGen::Rcudd1a) => sen::Precision::Int16,
        (_, IsaGen::Sen1p5) => sen::Precision::Fp24,
        (_, IsaGen::Rcudd1a) => sen::Precision::Fp16,
    }
}

/// WHAT [`VectorOperand::with_precision`] ANSWERS — the operand AND the `bool&` beside it.
///
/// ⛔ THE OUT-PARAMETER IS NOT AN IMPLEMENTATION DETAIL OF THE CALL. Every caller copies it into an
/// `is_precision_converted_global` and hands that to `patternAgnosticFuseNonComputeOpsHelper`
/// (`VectorChainToSentientPESFP.cpp:47-54`, `:1213`), which is what erases the folded cast — so an
/// answer that did not say whether a cast folded in would leave the cast in the program.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct OperandWithPrecision {
    /// The operand, or the reference's `std::nullopt`.
    pub operand: Option<VectorOperand>,
    /// `is_precision_converted` — set by the three arms that FOLD A CAST into their answer.
    pub precision_converted: bool,
}

impl VectorOperand {
    /// Replaces: e304_getOperandWithPrecision
    ///
    /// **304/384** `VectorOperand::getOperandWithPrecision` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389` (254L).
    ///
    /// WHICH OPERAND A COMPUTE READS AND AT WHAT PRECISION — the dispatch every other `from_*` in
    /// this file is one arm of, plus the `bool&` beside its answer.
    ///
    /// ⛔ THE THREE CAST ARMS ANSWER WITH SOMEONE ELSE'S OPERAND and overwrite only
    /// `on_the_fly_conv_precision_`; every other arm writes BOTH, and `NegOp` writes neither (`:566`).
    /// ⛔ `traverse_upwards` PICKS WHICH END OF A FOLDED OP IS ASKED — its input's definition, same
    /// block only (`:466`), or its single USER — but the RECURSION always takes the declaration's
    /// default `true` (`:468`, `:547`), so a chain resolves upwards from the first hop on.
    /// ⛔ THE `pt` ARM OVERRIDES THE TYPE THE IR CARRIES (`:446-453`), which is how an `f16` receive
    /// becomes the vendor's `opAPrecision = #sentient<precision fp24>`.
    /// ⛔ `Multiply`/`MultiplyAccumulate`/`Binary` ARE AN EXPLICIT `std::nullopt` (`:641-643`): a
    /// compute is not an operand of a compute, and falling through would be right for a wrong reason.
    pub fn with_precision<A: Arch>(
        op: &OpId,
        comp: ComputeComp,
        traverse_upwards: bool,
        scope: &[DfirOp],
        get_operand: &mut impl FnMut(&OpId, ComputeComp) -> Option<VectorOperand>,
    ) -> OperandWithPrecision {
        // `is_precision_converted = false;`
        let mut converted = false;
        let Some(at) = op_at(op, scope) else {
            return OperandWithPrecision {
                operand: None,
                precision_converted: converted,
            };
        };
        let operand = match at {
            // The four memory accesses. ⭐ EACH READS THE **VECTOR** IT MOVES, not the view: the
            // loads take their result's type and the stores `getValueToStore()`'s (`:397`, `:405`,
            // `:415`, `:425`) — one field here either way.
            DfirOp::Vector(vector::Op::Load { ty, .. } | vector::Op::Store { ty, .. })
            | DfirOp::Agen(agen::Op::VectorLoad { ty, .. } | agen::Op::VectorStore { ty, .. }) => {
                VectorOperand::from_load_or_store_op(op, scope).map(|mut operand| {
                    set_both_precisions(&mut operand, ty.elem);
                    operand
                })
            }
            // `getOperandFromSendOp(dcc_ext_ctx, send_op, comp)`.
            DfirOp::Dataflow(dataflow::Op::Send { to, ty, .. }) => send_destination(*to, scope)
                .map(|unit| {
                    let mut operand = VectorOperand::from_send_op::<A>(unit, comp, op.clone());
                    set_both_precisions(&mut operand, ty.elem);
                    operand
                }),
            // `getOperandFromReceiveOp(dcc_ext_ctx, recv_op, comp)`, then the PT override.
            DfirOp::Dataflow(dataflow::Op::Receive { from, ty, .. }) => {
                recv_source(*from, scope).map(|unit| {
                    let mut operand = VectorOperand::from_receive_op::<A>(unit, comp, op.clone());
                    // `if (operand.value().getFirstValue() == "pt")`.
                    if operand.values.first() == Some(&OperandValue::Port(sen::Port::Pt)) {
                        operand.orig_precision = Some(pt_receive_precision(ty.elem, A::GEN));
                        operand.on_the_fly_conv_precision = operand.orig_precision;
                    } else {
                        set_both_precisions(&mut operand, ty.elem);
                    }
                    operand
                })
            }
            // `mlir::arith::FPToSIOp` and `mlir::arith::SIToFPOp` — ⛔ THE FLAG IS SET BEFORE THE
            // `hasOneUse()` TEST (`:462`, `:488`), so a multiply-used conversion reports one anyway.
            DfirOp::Arith(arith::Op::FpToSi(conv) | arith::Op::SiToFp(conv)) => {
                converted = true;
                folded_cast(
                    op,
                    conv.input,
                    conv.to.elem,
                    comp,
                    traverse_upwards,
                    scope,
                    get_operand,
                )
            }
            // `getOperandFromConstantOp(const_op)` — ⭐ the scalar `arith.constant`s the reference
            // also reaches here abort inside it and inside `getElementType`; see entry 166's note.
            DfirOp::Arith(const_op @ arith::Op::DenseConstant { ty, .. }) => {
                VectorOperand::from_constant_op(const_op, op.clone()).map(|mut operand| {
                    set_both_precisions(&mut operand, ty.elem);
                    operand
                })
            }
            // `getOperandFromConstantBitstreamOp(const_bit_op)` — ⛔ WITH ITS DEFAULTED
            // `is_constant_splatted_vector = false` (`:301`), so the value is the raw immediate and
            // NOT the pseudo-port spelling entry 278's trivial-shuffle arm asks for.
            DfirOp::VectorChain(vc::Op::ConstantBitstream { value, ty, .. }) => {
                // `op.getValue()[0]` — an unchecked index; an empty list is `None` here.
                value.first().map(|const_val| {
                    let mut operand = VectorOperand::from_constant_bitstream_op(
                        BitstreamConstant::Immediate(*const_val),
                        op.clone(),
                    );
                    set_both_precisions(&mut operand, ty.elem);
                    operand
                })
            }
            // `getOperandFromShuffleOp(dcc_ext_ctx, shuffle_op, comp)` — ⭐ THE PRECISIONS ARE SET
            // ONLY IF IT ANSWERED (`:533-540`), and the `nullopt` is returned as it stands.
            DfirOp::VectorChain(vc::Op::Shuffle { ty, .. }) => {
                VectorOperand::from_shuffle_op::<A>(op, comp, scope, get_operand).map(
                    |mut operand| {
                        set_both_precisions(&mut operand, ty.elem);
                        operand
                    },
                )
            }
            // `vectorchain::CastOp` — ⛔ HERE THE FLAG IS SET **INSIDE** `hasOneUse()` (`:542-543`),
            // unlike the two `arith` conversions above. Entry 343 is this body reached from
            // `getOperandFromCastOp`.
            DfirOp::VectorChain(vc::Op::Cast { input, ty, .. }) => {
                if use_positions(op, scope).len() == 1 {
                    converted = true;
                    folded_cast(
                        op,
                        *input,
                        ty.elem,
                        comp,
                        traverse_upwards,
                        scope,
                        get_operand,
                    )
                } else {
                    None
                }
            }
            // `getOperandFromNegOp(..)` — *"The NegOp doesn't change the original precision"*.
            DfirOp::VectorChain(vc::Op::Neg { .. }) => {
                VectorOperand::from_neg_op(op, comp, scope, get_operand)
            }
            // `vectorchain::SelectOp` — the only writer of [`VectorOperand::splat`].
            DfirOp::VectorChain(vc::Op::Select {
                input,
                input_ty,
                ty,
                ..
            }) => {
                if use_positions(op, scope).len() != 1 {
                    None
                } else if traverse_upwards {
                    // ⛔ NO `has_value()` GUARD ON THIS BRANCH (`:574-583`): the reference
                    // dereferences whatever came back, so a select whose input has no operand is a
                    // null deref there and `None` here.
                    defining_position(*input, scope)
                        .and_then(|parent| get_operand(&parent, comp))
                        .map(|mut operand| {
                            // ⭐ THE INPUT'S ELEMENT TYPE, `getData().getType()` (`:576-577`).
                            set_both_precisions(&mut operand, input_ty.elem);
                            // `operand.value().splat_ = "east";`
                            operand.splat = Some(sen::Port::East);
                            operand
                        })
                } else {
                    // ⭐ AND THE DOWNWARD BRANCH TAKES THE SELECT'S **OWN** TYPE (`:588`), not its
                    // input's, and sets no splat.
                    use_positions(op, scope)
                        .into_iter()
                        .next()
                        .and_then(|user| get_operand(&user, comp))
                        .map(|mut operand| {
                            set_both_precisions(&mut operand, ty.elem);
                            operand
                        })
                }
            }
            // `vectorchain::ElementWiseCompareOp` — ⛔ UPWARDS ONLY (`:599`); asked downwards it
            // falls past every arm to the trailing `nullopt`.
            DfirOp::VectorChain(vc::Op::ElementWiseCompare { ty, .. }) if traverse_upwards => {
                // `for (auto user : ..getUsers()) if (isa<agen::VectorStoreOp>(user))`.
                let store = use_positions(op, scope).into_iter().find(|user| {
                    matches!(
                        op_at(user, scope),
                        Some(DfirOp::Agen(agen::Op::VectorStore { .. }))
                    )
                });
                match store {
                    Some(user) => {
                        // ⛔ THE RECURSION SHARES THE ONE `bool&`, and it re-initialises it to
                        // `false` on entry — so the store's answer REPLACES this call's flag rather
                        // than adding to it. Reaching here it was still false either way.
                        let answer = VectorOperand::with_precision::<A>(
                            &user,
                            comp,
                            traverse_upwards,
                            scope,
                            get_operand,
                        );
                        converted = answer.precision_converted;
                        answer.operand.map(|mut operand| {
                            set_both_precisions(&mut operand, ty.elem);
                            operand
                        })
                    }
                    // `VectorOperand(ISTATE, "0", ew_compare_op)` — ⚠️ CARRYING THE REFERENCE'S OWN
                    // `TODO`: *"set appropriate istate number once translator changes are
                    // implemented"*, which is why the index is [`sen::IStateIndex::S0`] and not a
                    // number this port chose.
                    None => {
                        let mut operand = VectorOperand::new(
                            VectorOperandType::IState,
                            OperandValue::Slice(RegisterSlice::IState(sen::IStateIndex::S0)),
                            op.clone(),
                        );
                        set_both_precisions(&mut operand, ty.elem);
                        Some(operand)
                    }
                }
            }
            // `mlir::uniform::QueryMapOp` — ONE OPERAND PER MAPPED VALUE, FOLDED INTO THE FIRST.
            //
            // ⛔ THE FOLD KEEPS ONLY `values_.front()` OF EACH LATER OPERAND (`:634-635`), so the
            // uniformized operand is one kind and one position with one value per core — which is
            // what [`VectorOperand::values`] is a list for.
            // ⛔ AND THE REFERENCE DEREFERENCES EVERY ANSWER UNCHECKED (`:627-640`), the null
            // `map_op`, each `operand.value()`, its `values_.front()` and the final
            // `folded_operand.value()` among them; each is a `None` here.
            DfirOp::Uniform(uniform::Op::QueryMap { map, .. }) => {
                let mapping = defining_position(*map, scope).and_then(|at| op_at(&at, scope));
                let Some(DfirOp::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = mapping
                else {
                    return OperandWithPrecision {
                        operand: None,
                        precision_converted: converted,
                    };
                };
                let mut folded: Option<VectorOperand> = None;
                for (_, value) in pairs {
                    let operand = defining_position(*value, scope)
                        .and_then(|position| get_operand(&position, comp));
                    match (&mut folded, operand) {
                        // `if (!folded_operand.has_value()) folded_operand = operand;` — ⭐ WHICH
                        // RETRIES: a first value with no operand leaves the fold empty and the next
                        // entry becomes the base.
                        (None, operand) => folded = operand,
                        (Some(folded), Some(operand)) => {
                            folded.values.extend(operand.values.first().copied());
                        }
                        (Some(_), None) => {}
                    }
                }
                // `folded_operand.value().op_ = query_op;` — the fold answers for the QUERY, not for
                // whichever core's value seeded it.
                folded.map(|mut operand| {
                    operand.op = op.clone();
                    operand
                })
            }
            // `isa<MultiplyOp, MultiplyAndAccumulateOp, BinaryOp>(op) -> std::nullopt`, and the
            // function's own trailing `return std::nullopt` for everything else.
            _ => None,
        };
        OperandWithPrecision {
            operand,
            precision_converted: converted,
        }
    }
}

/// THE BODY THE THREE CAST ARMS SHARE — `arith.fptosi` (`:461-486`), `arith.sitofp` (`:487-512`) and
/// `vectorchain.cast` (`:541-565`), which differ only in where the flag is set.
///
/// ⛔ ONLY `on_the_fly_conv_precision_` IS OVERWRITTEN: the cast folds INTO an operand that keeps its
/// own origin, its own position and its own original precision.
/// ⛔ THE UPWARD BRANCH REQUIRES THE SAME BLOCK AND THE DOWNWARD ONE DOES NOT — see [`same_block`],
/// which is the same question asked of an operand already built.
fn folded_cast(
    op: &OpId,
    input: Val,
    result_elem: ElemType,
    comp: ComputeComp,
    traverse_upwards: bool,
    scope: &[DfirOp],
    get_operand: &mut impl FnMut(&OpId, ComputeComp) -> Option<VectorOperand>,
) -> Option<VectorOperand> {
    // `if (cast_op->hasOneUse())`.
    if use_positions(op, scope).len() != 1 {
        return None;
    }
    let asked = if traverse_upwards {
        // `if (cast_op.getIn().getDefiningOp()->getBlock() == op->getBlock())`.
        let parent = defining_position(input, scope)?;
        if parent.block() != op.block() {
            return None;
        }
        parent
    } else {
        // `Operation *user = (*cast_op->getUses().begin()).getOwner();`
        use_positions(op, scope).into_iter().next()?
    };
    let mut operand = get_operand(&asked, comp)?;
    operand.on_the_fly_conv_precision = Some(precision_in_string(result_elem));
    Some(operand)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 320/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

impl VectorOperand {
    /// Replaces: e320_getOperand
    ///
    /// **320/384** `VectorOperand::getOperand` — `VectorOperands.cpp:378` (4L): the overload that
    /// asks [`Self::with_precision`] and throws the `bool&` away, and the one that CLOSES the
    /// recursion every other unit in this file cuts open with a `get_operand` parameter.
    ///
    /// ⛔ THE RECURSION IT HANDS DOWN IS `traverse_upwards = true`, NOT THIS CALL'S ARGUMENT — the
    /// reference's inner calls pass three arguments and take the declaration's default
    /// (`VectorOperands.hpp:85`), so only the FIRST hop honours a caller's `false`.
    #[must_use]
    pub fn operand<A: Arch>(
        op: &OpId,
        comp: ComputeComp,
        traverse_upwards: bool,
        scope: &[DfirOp],
    ) -> Option<VectorOperand> {
        // `bool is_precision_converted;` — declared UNINITIALISED and never read by this overload;
        // `getOperandWithPrecision` writes it on entry, so nothing observes the indeterminate value.
        Self::with_precision::<A>(
            op,
            comp,
            traverse_upwards,
            scope,
            &mut |inner, inner_comp| Self::operand::<A>(inner, inner_comp, true, scope),
        )
        .operand
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{BitstreamConstant, IrfIndex, LayoutAndIndices, RegisterSlice};
    use super::{ComputeComp, ConstantOperandValue, OpId, OperandValue, VectorOperand};
    use super::{
        VectorOperandType, const_val_to_field, erase_op, erase_operands, layout_map_and_indices,
        same_block,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{Index, Op as DfirOp, Val};
    use crate::islands::dataflow_ir::dialects::{affine, agen, arith, dataflow, uniform, vector};
    use crate::islands::dataflow_ir::link::{Link, Lxsu, Pe, PtRowUnit, Sfp};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
    use crate::islands::sentient::dialects::sentient as sen;
    use crate::units::{Core, Corelet, DfirUnit, Residency, Row};

    /// The vector every op in these fixtures is typed at — 128 lanes of bf16, the width
    /// `dcc/test/PESFP/*.mlir` computes at.
    const V: Vector = Vector {
        len: 128,
        elem: ElemType::Bf16,
    };

    /// The vector the PT row's transfers move — 64 lanes of bf16, one stick
    /// (`dcc/test/PT/bf16-pt.mlir:161`, `:214`).
    const V64: Vector = Vector {
        len: 64,
        elem: ElemType::Bf16,
    };

    /// `arith.constant dense<0> : vector<128xbf16>` binding `result`.
    fn dense(result: Val) -> DfirOp {
        DfirOp::Arith(arith::Op::DenseConstant {
            result,
            splat: 0,
            ty: V,
        })
    }

    /// `vectorchain.fast_exp %input : vector<128xbf16>` binding `result`.
    fn fast_exp(result: Val, input: Val) -> DfirOp {
        DfirOp::VectorChain(vc::Op::FastExp {
            result,
            input,
            input_ty: V,
            ty: V,
        })
    }

    /// `vectorchain.floor %input` binding `result`.
    fn floor(result: Val, input: Val) -> DfirOp {
        DfirOp::VectorChain(vc::Op::Floor {
            result,
            input,
            input_ty: V,
            ty: V,
        })
    }

    /// `affine.for %iv = 0 to 8 { body }`, carrying nothing.
    fn for_loop(iv: Val, body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(8),
            carried: Vec::new(),
            body,
            dbg_name: None,
        })
    }

    /// An operand of `kind` defined at `path`, with both precisions unset.
    fn operand(kind: VectorOperandType, path: &[u32]) -> VectorOperand {
        VectorOperand {
            kind,
            op: OpId::at(path),
            values: Vec::new(),
            orig_precision: None,
            on_the_fly_conv_precision: None,
            splat: None,
        }
    }

    fn row(index: u32) -> DfirUnit {
        DfirUnit::PtRow(Row::checked(index).expect("this arch has a row zero"))
    }

    fn port(operand: &VectorOperand) -> sen::Port {
        assert_eq!(operand.kind, VectorOperandType::Link);
        let [OperandValue::Port(port)] = operand.values.as_slice() else {
            unreachable!("a link operand carries exactly one port")
        };
        *port
    }

    /// ⭐ `loweringXRF_with_if_branch.mlir:130` is a `type = "l0lu"` unit, three receives read it
    /// (`:145`, `:154`, `:164`), and `:49`/`:58`/`:68` expect `opA = #sentient<compute_port west>`.
    #[test]
    fn a_pt_receiving_from_the_l0_load_unit_reads_west() {
        let operand =
            VectorOperand::from_receive_op::<Dd2>(DfirUnit::L0lu, ComputeComp::Pt, OpId::at(&[0]));
        assert_eq!(port(&operand), sen::Port::West);
        assert_eq!(operand.op, OpId::at(&[0]));
        assert_eq!(operand.orig_precision, None);
        assert_eq!(operand.on_the_fly_conv_precision, None);
    }

    /// ⭐ `xrf_increments.mlir:413-457` receives from an `lxlu` on a PT unit; `:71`/`:81`/`:88` expect
    /// `opC = #sentient<compute_port north>`. The row above it and the SFP take the same arm.
    #[test]
    fn a_pt_receives_from_the_lx_the_sfp_and_the_rows_over_north() {
        for peer in [DfirUnit::Lxlu, DfirUnit::Sfp, row(0), row(1)] {
            let operand =
                VectorOperand::from_receive_op::<Dd2>(peer, ComputeComp::Pt, OpId::at(&[1]));
            assert_eq!(port(&operand), sen::Port::North, "{peer:?}");
        }
    }

    #[test]
    fn a_pt_receiving_over_the_cross_pt_link_names_it() {
        let operand = VectorOperand::from_receive_op::<Dd2>(
            DfirUnit::CrossPtnLink,
            ComputeComp::Pt,
            OpId::at(&[2]),
        );
        assert_eq!(port(&operand), sen::Port::CrossPtNorthLink);
    }

    /// ⭐ THE PE/SFP BRANCH, ARM BY ARM. Both LX halves collapse to `lx`; the PT rows to `pt`.
    #[test]
    fn a_pe_receives_from_the_lx_halves_the_pt_and_the_sfp() {
        for (peer, expected) in [
            (DfirUnit::Lxlu, sen::Port::Lx),
            (DfirUnit::Lxsu, sen::Port::Lx),
            (DfirUnit::Pe, sen::Port::Pe),
            (row(0), sen::Port::Pt),
            (DfirUnit::Sfp, sen::Port::Sfp),
        ] {
            let operand =
                VectorOperand::from_receive_op::<Dd2>(peer, ComputeComp::Pe, OpId::at(&[3]));
            assert_eq!(port(&operand), expected, "{peer:?}");
        }
    }

    /// ⛔ THE RING IS THE SFP ASKING, NOT THE SFP ANSWERING. Same peer, two askers, two ports.
    #[test]
    fn only_an_sfp_receiving_from_an_sfp_reads_the_ring() {
        let asked_by_sfp =
            VectorOperand::from_receive_op::<Dd2>(DfirUnit::Sfp, ComputeComp::Sfp, OpId::at(&[4]));
        let asked_by_pe =
            VectorOperand::from_receive_op::<Dd2>(DfirUnit::Sfp, ComputeComp::Pe, OpId::at(&[4]));
        assert_eq!(port(&asked_by_sfp), sen::Port::SfpRing);
        assert_eq!(port(&asked_by_pe), sen::Port::Sfp);
        // ⭐ AND IT IS THE RING ON THE NEWER GENERATION TOO — `supports_sfp_ring` is total.
        let on_sen1p5 = VectorOperand::from_receive_op::<Sen1p5>(
            DfirUnit::Sfp,
            ComputeComp::Sfp,
            OpId::at(&[4]),
        );
        assert_eq!(port(&on_sen1p5), sen::Port::SfpRing);
    }

    /// ⭐ `xrf_increments.mlir:422` sends to `%5`, a `type = "ptrow1"` unit, and `:81`/`:88` expect
    /// `ResultForwarding = [#sentient<compute_port south>]`. A send to the PE takes the same arm.
    #[test]
    fn a_pt_sends_to_the_rows_and_the_pe_over_south() {
        for peer in [row(1), row(0), DfirUnit::Pe] {
            let operand = VectorOperand::from_send_op::<Dd2>(peer, ComputeComp::Pt, OpId::at(&[5]));
            assert_eq!(port(&operand), sen::Port::South, "{peer:?}");
        }
    }

    /// ⭐ THE SEND SIDE REACHES THE MEMORIES THE RECEIVE SIDE DOES NOT — both L0 halves and both LX
    /// halves (`VectorOperands.cpp:128-133`).
    #[test]
    fn a_pe_sends_to_the_l0_and_lx_halves() {
        for (peer, expected) in [
            (DfirUnit::L0lu, sen::Port::L0),
            (DfirUnit::L0su, sen::Port::L0),
            (DfirUnit::Lxlu, sen::Port::Lx),
            (DfirUnit::Lxsu, sen::Port::Lx),
            (row(0), sen::Port::Pt),
            (DfirUnit::Pe, sen::Port::Pe),
        ] {
            let operand = VectorOperand::from_send_op::<Dd2>(peer, ComputeComp::Pe, OpId::at(&[6]));
            assert_eq!(port(&operand), expected, "{peer:?}");
        }
    }

    #[test]
    fn only_an_sfp_sending_to_an_sfp_uses_the_ring() {
        let asked_by_sfp =
            VectorOperand::from_send_op::<Dd2>(DfirUnit::Sfp, ComputeComp::Sfp, OpId::at(&[7]));
        let asked_by_pe =
            VectorOperand::from_send_op::<Dd2>(DfirUnit::Sfp, ComputeComp::Pe, OpId::at(&[7]));
        assert_eq!(port(&asked_by_sfp), sen::Port::SfpRing);
        assert_eq!(port(&asked_by_pe), sen::Port::Sfp);
    }

    /// ⛔ ONE VALUE, AND THE CONSTRUCTOR CLEARS FIRST. `setValue` is `values_.clear()` then
    /// `emplace_back` (`VectorOperands.hpp:72-75`), so a freshly built operand has exactly one entry
    /// however many folds it may later carry.
    #[test]
    fn a_new_operand_carries_exactly_one_value() {
        let operand = VectorOperand::new(
            VectorOperandType::Lrf,
            OperandValue::Port(sen::Port::Latch),
            OpId::at(&[8, 1]),
        );
        assert_eq!(operand.values, vec![OperandValue::Port(sen::Port::Latch)]);
        assert_eq!(operand.kind, VectorOperandType::Lrf);
        assert_eq!(operand.op.path(), &[8, 1]);
    }

    // ── e073_constValToField ──────────────────────────────────────────────────────────────────

    /// ⭐ THE VENDOR CASE IS THE PORT NAME ITSELF. `dcc/test` reaches this only through whole-program
    /// `CHECK-SENT-IR` lines, where its answer appears as the operand field of a compute —
    /// `dcc/test/PESFP/exp_bf16.mlir` checks `operand_b = #sentient<compute_port zero>` for a
    /// `dense<0.0>` splat. That mapping is what this asserts.
    #[test]
    fn the_four_splat_values_name_the_four_pseudo_unit_ports() {
        assert_eq!(
            const_val_to_field(ConstantOperandValue::Zero),
            sen::Port::Zero
        );
        assert_eq!(
            const_val_to_field(ConstantOperandValue::One),
            sen::Port::One
        );
        assert_eq!(
            const_val_to_field(ConstantOperandValue::Two),
            sen::Port::Two
        );
        assert_eq!(
            const_val_to_field(ConstantOperandValue::Three),
            sen::Port::Three
        );
    }

    /// ⛔ FOUR DISTINCT PORTS, not four names for one. A mapping that collapsed two would make two
    /// different splats read the same pseudo-unit.
    #[test]
    fn the_four_ports_are_distinct() {
        let ports = [
            const_val_to_field(ConstantOperandValue::Zero),
            const_val_to_field(ConstantOperandValue::One),
            const_val_to_field(ConstantOperandValue::Two),
            const_val_to_field(ConstantOperandValue::Three),
        ];
        for (i, a) in ports.iter().enumerate() {
            for b in &ports[i + 1..] {
                assert_ne!(a, b, "{ports:?}");
            }
        }
    }

    // ── OpId::block ───────────────────────────────────────────────────────────────────────────

    /// ⭐ A BLOCK IS THE PATH WITHOUT THE LAST ORDINAL, and the top level's block is the empty one.
    #[test]
    fn a_block_is_the_position_without_the_op_s_own_ordinal() {
        assert_eq!(OpId::at(&[3]).block(), &[] as &[u32]);
        assert_eq!(OpId::at(&[4]).block(), &[] as &[u32]);
        assert_eq!(OpId::at(&[3, 1]).block(), &[3]);
        assert_eq!(OpId::at(&[3, 1, 2]).block(), &[3, 1]);
    }

    // ── e074_sameBlock ────────────────────────────────────────────────────────────────────────

    /// ⛔⛔ AN ABSENT OPERAND FAILS. `OperandReuse.cpp:31` reads the failure as "latch it"; a `true`
    /// here would let an operand nobody could resolve claim register reuse.
    #[test]
    fn an_absent_operand_is_not_in_the_same_block() {
        let scope = vec![dense(Val(0)), fast_exp(Val(1), Val(0))];
        assert!(!same_block(&OpId::at(&[1]), None, &scope));
    }

    /// The ordinary case: a compute and the op feeding it, side by side at the top level.
    #[test]
    fn a_top_level_operand_shares_the_top_level_block() {
        let scope = vec![
            dense(Val(0)),
            fast_exp(Val(1), Val(0)),
            floor(Val(2), Val(1)),
        ];
        let from_a_link = operand(VectorOperandType::Link, &[1]);
        assert!(same_block(&OpId::at(&[2]), Some(&from_a_link), &scope));
    }

    /// ⛔ A NON-CONSTANT OPERAND FROM ANOTHER BLOCK FAILS — the whole point of the function.
    #[test]
    fn an_operand_from_a_loop_body_is_a_different_block() {
        let scope = vec![
            dense(Val(0)),
            for_loop(Val(9), vec![fast_exp(Val(1), Val(0))]),
            floor(Val(2), Val(1)),
        ];
        // `[1, 0]` is the `fast_exp` inside the loop; its block is `[1]`, the user's is `[]`.
        let inside_the_loop = operand(VectorOperandType::Link, &[1, 0]);
        assert!(!same_block(&OpId::at(&[2]), Some(&inside_the_loop), &scope));
    }

    /// ⛔⛔ THE EXEMPTION IS ON THE OP'S CLASS, NOT ON THE OPERAND'S KIND. The same position, the
    /// same blocks, and the answer flips because the defining op is an `arith.constant` — and note
    /// the kind here is `Link`, so a port that had tested `kind == Constant` would answer `false`.
    #[test]
    fn a_constant_from_another_block_is_exempt() {
        let scope = vec![
            fast_exp(Val(1), Val(0)),
            for_loop(Val(9), vec![dense(Val(0))]),
            floor(Val(2), Val(1)),
        ];
        let hoisted_constant = operand(VectorOperandType::Link, &[1, 0]);
        assert!(same_block(&OpId::at(&[2]), Some(&hoisted_constant), &scope));
    }

    // ── e075_eraseOp ──────────────────────────────────────────────────────────────────────────

    /// An op with no uses at all: `all_uses_deleted` never falsifies, so it goes.
    #[test]
    fn an_unused_op_is_erased_on_its_own() {
        let mut scope = vec![dense(Val(0)), fast_exp(Val(1), Val(7))];
        erase_op(&OpId::at(&[0]), &mut scope);
        assert_eq!(scope, vec![fast_exp(Val(1), Val(7))]);
    }

    /// ⭐ THE ONE USER GOES TOO, because nothing reads it — the reference's single edge.
    #[test]
    fn the_op_and_its_only_dead_user_both_go() {
        let mut scope = vec![dense(Val(0)), fast_exp(Val(1), Val(0))];
        erase_op(&OpId::at(&[0]), &mut scope);
        assert!(scope.is_empty(), "{scope:?}");
    }

    /// ⛔⛔ ONE LIVE READER KEEPS EVERYTHING. `fast_exp` is read by `floor`, so it is not erasable,
    /// `all_uses_deleted` is false, and the constant survives feeding it.
    #[test]
    fn a_user_that_is_itself_read_keeps_the_definition() {
        let before = vec![
            dense(Val(0)),
            fast_exp(Val(1), Val(0)),
            floor(Val(2), Val(1)),
        ];
        let mut scope = before.clone();
        erase_op(&OpId::at(&[0]), &mut scope);
        assert_eq!(scope, before);
    }

    /// ⛔ ONE LEVEL, NOT TRANSITIVE. Erasing the head of the chain above from its middle takes
    /// `floor` (nothing reads it) and `fast_exp`, and leaves the constant — a transitive sweep would
    /// have taken all three.
    #[test]
    fn the_walk_stops_after_one_edge() {
        let mut scope = vec![
            dense(Val(0)),
            fast_exp(Val(1), Val(0)),
            floor(Val(2), Val(1)),
        ];
        erase_op(&OpId::at(&[1]), &mut scope);
        assert_eq!(scope, vec![dense(Val(0))]);
    }

    /// ⛔ A USER INSIDE A REGION IS STILL A USER, and it is removed from that region rather than
    /// from the top level.
    #[test]
    fn a_user_nested_in_a_loop_is_erased_in_place() {
        let mut scope = vec![
            dense(Val(0)),
            for_loop(Val(9), vec![fast_exp(Val(1), Val(0))]),
        ];
        erase_op(&OpId::at(&[0]), &mut scope);
        assert_eq!(scope, vec![for_loop(Val(9), Vec::new())]);
    }

    /// ⛔⛔ DESCENDING ORDER, WHICH IS THIS PORT'S OBLIGATION. Three dead users at `[1]`, `[2]` and
    /// `[3]`: removing them front-first would renumber the survivors and delete the wrong ops. Only
    /// the trailing `floor`, which reads nothing of `%0`, is left.
    #[test]
    fn several_dead_users_are_removed_without_renumbering_each_other() {
        let mut scope = vec![
            dense(Val(0)),
            fast_exp(Val(1), Val(0)),
            fast_exp(Val(2), Val(0)),
            fast_exp(Val(3), Val(0)),
            floor(Val(4), Val(8)),
        ];
        erase_op(&OpId::at(&[0]), &mut scope);
        assert_eq!(scope, vec![floor(Val(4), Val(8))]);
    }

    /// ⛔ ONE ERASE PER OP EVEN WHEN ONE OP READS THE VALUE TWICE. `getUses()` yields two uses for
    /// this `binary`, and the reference would push it into `to_be_erased` twice and erase it twice;
    /// here a second removal at the same ordinal would take an innocent op instead.
    #[test]
    fn an_op_reading_the_value_twice_is_erased_once() {
        let mut scope = vec![
            dense(Val(0)),
            DfirOp::VectorChain(vc::Op::Binary {
                result: Val(1),
                op1: Val(0),
                op2: Val(0),
                mask: None,
                binary_op: vc::BinaryOp::Add,
                op_specific_map: AffineMap::unary(AffineExpr::Dim(0)),
                operand_ty: V,
                ty: V,
            }),
            floor(Val(2), Val(8)),
        ];
        erase_op(&OpId::at(&[0]), &mut scope);
        assert_eq!(scope, vec![floor(Val(2), Val(8))]);
    }

    // ═══════════════════════════════════════ 166 ═══════════════════════════════════════

    /// `arith.constant dense<splat> : vector<128xbf16>` at position `[0]`.
    fn dense_splat(splat: i64) -> arith::Op {
        arith::Op::DenseConstant {
            result: Val(30),
            splat,
            ty: V,
        }
    }

    /// ⭐ 166/384 — THE VENDOR'S OWN PAIR: `dcc/test/PE/test1.mlir:38-39` splats one and zero into a
    /// MAC's `opB` and `opC`, and its `CHECK-SENT-IR` reads `opB = #sentient<compute_port one>,
    /// opC = #sentient<compute_port zero>` (`:18`).
    ///
    /// ⭐ AND THE OPERAND IS `Constant`-KINDED OVER THE CONSTANT'S OWN OP, carrying exactly one value
    /// with both precisions unset — the caller fills those (`VectorOperands.cpp:516-521`).
    #[test]
    fn a_splat_of_one_is_the_one_port() {
        let at = OpId::at(&[3]);
        let one =
            VectorOperand::from_constant_op(&dense_splat(1), at.clone()).expect("one is a port");

        assert_eq!(one.kind, VectorOperandType::Constant);
        assert_eq!(one.op, at);
        assert_eq!(one.values, vec![OperandValue::Port(sen::Port::One)]);
        assert_eq!(one.orig_precision, None);
        assert_eq!(one.on_the_fly_conv_precision, None);

        let zero = VectorOperand::from_constant_op(&dense_splat(0), at).expect("zero is a port");
        assert_eq!(zero.values, vec![OperandValue::Port(sen::Port::Zero)]);
    }

    /// 🎯 DERIVED — TWO AND THREE ARE PORTS TOO, and no test in the authority tree reaches them:
    /// `compute_port` never spells anything but `zero` or `one` across all 825 files, because the
    /// `dense<2>` and `dense<3.000000e+00>` constants they do contain feed a `sentient.splat`
    /// instead. Their acceptance is `constValToField`'s own domain (`VectorOperands.cpp:250-262`).
    #[test]
    fn the_four_splats_are_the_four_pseudo_ports() {
        let at = OpId::at(&[0]);
        let ports = [
            sen::Port::Zero,
            sen::Port::One,
            sen::Port::Two,
            sen::Port::Three,
        ];

        for (splat, port) in (0..4).zip(ports) {
            let operand = VectorOperand::from_constant_op(&dense_splat(splat), at.clone())
                .unwrap_or_else(|| panic!("dense<{splat}> is a port"));
            assert_eq!(operand.values, vec![OperandValue::Port(port)]);
        }
    }

    /// 🎯 THE TWO ABORTS — a splat past three, and a constant that is not a vector at all.
    ///
    /// ⚠️ DERIVED: the reference throws for the first (`DT_ERROR`) and asserts inside `mlir::cast`
    /// for the second, so neither has a fixture. ⭐ `dense<4.000000e+00>` is real input, though —
    /// twice, in the authority's tests — which is why [`arith::Op::DenseConstant`] can spell it.
    #[test]
    fn a_splat_no_port_names_is_declined() {
        let at = OpId::at(&[0]);
        assert_eq!(ConstantOperandValue::of(4), None);
        assert_eq!(ConstantOperandValue::of(-1), None);
        assert!(VectorOperand::from_constant_op(&dense_splat(4), at.clone()).is_none());

        // `arith.constant 3 : index` — the caller reaches this for any `arith.constant`.
        assert!(
            VectorOperand::from_constant_op(
                &arith::Op::Constant {
                    result: Val(31),
                    value: 3,
                },
                at
            )
            .is_none()
        );
    }
    // ── e167_getOperandFromConstantBitstreamOp ────────────────────────────────────────────────

    /// ⭐ THE SPLATTED READING IS A PSEUDO-UNIT PORT, NOT A NUMBER. `getOperandFromShuffleOp`'s
    /// trivial-shuffle branch is the only caller that passes `is_constant_splatted_vector = true`
    /// (`VectorOperands.cpp:333-334`), and it re-tags the answer `ConstantBitstream` on the very next
    /// line (`:335`) — so the only thing THIS function decides for that caller is the value.
    #[test]
    fn a_splatted_bitstream_constant_reads_as_its_pseudo_unit_port() {
        let operand = VectorOperand::from_constant_bitstream_op(
            BitstreamConstant::SplattedVector(ConstantOperandValue::Two),
            OpId::at(&[4]),
        );
        assert_eq!(operand.kind, VectorOperandType::Constant);
        assert_eq!(operand.values, vec![OperandValue::Port(sen::Port::Two)]);
        assert_eq!(operand.op, OpId::at(&[4]));
        assert_eq!(operand.name(), Some(sen::Port::Two));
    }

    /// ⛔ WITHOUT THE FLAG IT IS A DECIMAL WITH NO COMPUTE-PORT SPELLING AT ALL. `:522-530` is the
    /// caller that takes the default, and `symbolizeSentientComputePort("7")` has no case to answer
    /// with — see [`OperandValue::Literal`]. ⭐ AND BOTH PRECISIONS STAY UNSET, because that caller
    /// writes them itself off the bitstream's element type (`:525-529`), not this function.
    #[test]
    fn an_unsplatted_bitstream_constant_stays_an_immediate_with_no_port() {
        let operand = VectorOperand::from_constant_bitstream_op(
            BitstreamConstant::Immediate(7),
            OpId::at(&[4]),
        );
        assert_eq!(operand.kind, VectorOperandType::Constant);
        assert_eq!(operand.values, vec![OperandValue::Literal(7)]);
        assert_eq!(operand.name(), None);
        assert_eq!(operand.orig_precision, None);
        assert_eq!(operand.on_the_fly_conv_precision, None);
    }

    // ── e168_getOperandFromNegOp ──────────────────────────────────────────────────────────────

    /// `vectorchain.neg %input : vector<128xbf16>` binding `result`.
    fn neg(result: Val, input: Val) -> DfirOp {
        DfirOp::VectorChain(vc::Op::Neg {
            result,
            input,
            mask: None,
            input_ty: V,
            ty: V,
        })
    }

    /// ⭐⭐ THE QUESTION IS FORWARDED ABOUT THE INPUT'S DEFINER, NEVER ABOUT THE NEGATION. Here
    /// `%2 = vectorchain.neg %1` reads the `fast_exp` at `[1]`, so the recursion is asked exactly once
    /// and about `[1]`, and its answer comes back untouched — precisions included, because
    /// `getOperandFromNegOp` writes none (`VectorOperands.cpp:366-373`) and its sole caller says why:
    /// *"The NegOp doesn't change the original precision"* (`:568`).
    #[test]
    fn a_negation_forwards_the_operand_of_what_it_reads() {
        let scope = vec![dense(Val(0)), fast_exp(Val(1), Val(0)), neg(Val(2), Val(1))];
        let mut asked: Vec<(OpId, ComputeComp)> = Vec::new();
        let answer = {
            let mut recurse = |at: &OpId, comp: ComputeComp| {
                asked.push((at.clone(), comp));
                Some(operand(VectorOperandType::Nfwd, at.path()))
            };
            VectorOperand::from_neg_op(&OpId::at(&[2]), ComputeComp::Sfp, &scope, &mut recurse)
        };
        assert_eq!(asked, vec![(OpId::at(&[1]), ComputeComp::Sfp)]);
        assert_eq!(answer, Some(operand(VectorOperandType::Nfwd, &[1])));
    }

    /// ⛔ `DT_CHECK(isa<vectorchain::NegOp>(op))` IS A CLASSIFICATION HERE, NOT AN ABORT. A caller
    /// that arrives with anything else gets `None`, and the recursion is never consulted at all.
    #[test]
    fn a_position_holding_no_negation_answers_none_without_recursing() {
        let scope = vec![dense(Val(0)), fast_exp(Val(1), Val(0))];
        let mut asked = 0_usize;
        let answer = {
            let mut recurse = |_: &OpId, _: ComputeComp| {
                asked += 1;
                Some(operand(VectorOperandType::Nfwd, &[0]))
            };
            VectorOperand::from_neg_op(&OpId::at(&[1]), ComputeComp::Pe, &scope, &mut recurse)
        };
        assert_eq!(answer, None);
        assert_eq!(asked, 0);
    }

    /// ⛔ A NEGATION READING A BLOCK ARGUMENT HAS NO DEFINER, and the reference dereferences the
    /// null: `getOperand(…, nullptr, comp)` tail-calls `getOperandWithPrecision`, whose first
    /// statement past the out-parameter is `isa<vector::LoadOp>(op)` (`VectorOperands.cpp:394`).
    /// `None` is the deliberate divergence, and the recursion is not asked about a position that
    /// does not exist.
    #[test]
    fn a_negation_of_a_loop_induction_variable_answers_none() {
        let scope = vec![for_loop(Val(9), vec![neg(Val(2), Val(9))])];
        let mut asked = 0_usize;
        let answer = {
            let mut recurse = |_: &OpId, _: ComputeComp| {
                asked += 1;
                Some(operand(VectorOperandType::Nfwd, &[0]))
            };
            VectorOperand::from_neg_op(&OpId::at(&[0, 0]), ComputeComp::Pt, &scope, &mut recurse)
        };
        assert_eq!(answer, None);
        assert_eq!(asked, 0);
    }

    // ── e169_getName ──────────────────────────────────────────────────────────────────────────

    /// ⭐ THE FOUR PREFIXED ARMS, ONE ASSERTION EACH — `"lrf" + value`, `"irf" + value`,
    /// `"istate" + value` (`VectorOperands.cpp:866-877`). In this port the file and its bounded index
    /// travel together in the VALUE, so there is no decimal to concatenate a prefix onto: the prefix
    /// is which [`RegisterSlice`] case the value is.
    #[test]
    fn a_register_file_slice_names_itself_with_its_files_prefix() {
        let named = |kind: VectorOperandType, value: OperandValue| {
            VectorOperand::new(kind, value, OpId::at(&[0])).name()
        };
        assert_eq!(
            named(
                VectorOperandType::Lrf,
                OperandValue::Slice(RegisterSlice::Lrf(sen::LrfIndex::L3))
            ),
            Some(sen::Port::Lrf(sen::LrfIndex::L3))
        );
        assert_eq!(
            named(
                VectorOperandType::Irf,
                OperandValue::Slice(RegisterSlice::Irf(IrfIndex::I0))
            ),
            Some(sen::Port::Irf0)
        );
        assert_eq!(
            named(
                VectorOperandType::Irf,
                OperandValue::Slice(RegisterSlice::Irf(IrfIndex::I1))
            ),
            Some(sen::Port::Irf1)
        );
        assert_eq!(
            named(
                VectorOperandType::IState,
                OperandValue::Slice(RegisterSlice::IState(sen::IStateIndex::S2))
            ),
            Some(sen::Port::IState(sen::IStateIndex::S2))
        );
    }

    /// ⛔⛔ THE `latch` DIVERGENCE, AND IT IS THE WHOLE REASON THIS PORT READS THE VALUE AND NOT
    /// `type_`. `OperandReuse.cpp` re-values a reused operand `"latch"` and leaves `type_` alone
    /// (`:28`, `:30`, `:32`, `:40`, `:43`), so the reference's `getName()` answers `"lrflatch"` for a
    /// latched LRF operand — which is not a `SentientComputePort` at all, and every call site then
    /// calls `.value()` on the `nullopt`. ⭐ `:57` IS THE EVIDENCE FOR WHICH ANSWER WAS MEANT: its
    /// own test is the unprefixed `from.getName() != "latch"`.
    #[test]
    fn a_latched_lrf_operand_names_the_latch_and_not_lrflatch() {
        let operand = VectorOperand::new(
            VectorOperandType::Lrf,
            OperandValue::Port(sen::Port::Latch),
            OpId::at(&[0]),
        );
        assert_eq!(operand.name(), Some(sen::Port::Latch));
    }

    /// ⭐ THE `XRF` ARM IS REDUNDANT WITH THE `else`, AND THIS SHOWS IT RATHER THAN ASSERTING IT. An
    /// XRF operand is built exactly once in the whole reference, as
    /// `VectorOperand(operand_type, "xrf", op)` (`VectorOperands.cpp:219-220`), so its value already
    /// IS the string that arm returns — and the `else` therefore answers identically.
    #[test]
    fn an_xrf_operand_names_the_xrf_from_its_value_alone() {
        let named = |kind: VectorOperandType| {
            VectorOperand::new(kind, OperandValue::Port(sen::Port::Xrf), OpId::at(&[0])).name()
        };
        assert_eq!(named(VectorOperandType::Xrf), Some(sen::Port::Xrf));
        assert_eq!(named(VectorOperandType::Link), Some(sen::Port::Xrf));
    }

    /// ⛔ AN OPERAND CARRYING NO VALUE HAS NO NAME, and in the reference it has no defined behaviour:
    /// `getFirstValue()` is `values_.front()` on an empty `std::vector` (`VectorOperands.hpp:76`).
    #[test]
    fn an_operand_with_no_value_has_no_name() {
        assert_eq!(operand(VectorOperandType::Link, &[0]).name(), None);
    }

    // ── e170_getLayoutMapAndIndices ───────────────────────────────────────────────────────────

    /// `memref<64x16x1xbf16>` — the PT row's own view throughout `dcc/test/PT/bf16-pt.mlir`.
    fn row_view_ty() -> MemRef {
        MemRef {
            shape: vec![64, 16, 1],
            elem: ElemType::Bf16,
        }
    }

    /// `#map4 = affine_map<(d0, d1, d2) -> (d2 * 1024 + d1 * 64 + d0)>` (`bf16-pt.mlir:69`) — built
    /// with the VERBATIM operators, because a printed map is transcribed exactly as the program
    /// spells it.
    fn layout_map4() -> AffineMap {
        AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(1024)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        }
    }

    /// `%52 = dataflow.get_logical_memory_view %47, %c0_0 {layout_map = #map4} : index, index,
    /// memref<64x16x1xbf16>` (`bf16-pt.mlir:211`).
    fn row_view() -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: Val(52),
            from: Val(47),
            start: Val(1),
            layout: layout_map4(),
            ty: row_view_ty(),
        })
    }

    /// `[0, %arg12 + %arg11 * 2 + %arg10 * 8, 0]` — the load's subscripts at `bf16-pt.mlir:214`.
    fn row_subscripts() -> Vec<Index> {
        vec![
            Index::Const(0),
            Index::Strided(vec![(Val(12), 1), (Val(11), 2), (Val(10), 8)], 0),
            Index::Const(0),
        ]
    }

    /// `d0 + d1 * 2 + d2 * 8` — the flattened stick index those subscripts compute, one dimension per
    /// distinct operand in the order the access lists them.
    fn flat_stick_index() -> AffineExpr {
        AffineExpr::dim(0)
            .plus(AffineExpr::dim(1).times(2))
            .plus(AffineExpr::dim(2).times(8))
    }

    /// ⭐⭐ THE `agen` ARM COMPOSES, AND ON THE VENDOR'S OWN PROGRAM THE ANSWER IS THE VECTOR WIDTH.
    /// `bf16-pt.mlir:214` loads `%52[0, %arg12 + %arg11 * 2 + %arg10 * 8, 0]` from a view whose
    /// `layout_map` is `#map4 = (d0, d1, d2) -> (d2 * 1024 + d1 * 64 + d0)` (`:69`): the lane axis
    /// `d0` and the page axis `d2` are both literal zero, so the composite collapses to `64 *` the
    /// flattened stick index — sixty-four elements per stick, which is exactly the `vector<64xbf16>`
    /// the load binds. ⛔ AND THE ORDER MAP LEAVES IT ALONE: the program writes
    /// `load_order = #map5` (`:70`), the identity over three dims, which is precisely what
    /// [`AffineMap::identity`] over `view_ty.shape.len()` derives.
    #[test]
    fn an_agen_load_composes_the_views_layout_with_its_own_subscripts() {
        let scope = vec![
            row_view(),
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: Val(53),
                view: Val(52),
                indices: row_subscripts(),
                view_ty: row_view_ty(),
                ty: V64,
                multicast_info: None,
            }),
        ];
        assert_eq!(
            layout_map_and_indices(&OpId::at(&[1]), &scope),
            Some(LayoutAndIndices {
                layout_map: AffineMap {
                    dims: 3,
                    syms: 0,
                    results: vec![flat_stick_index().times(64)],
                },
                operands: vec![Val(12), Val(11), Val(10)],
                logical_view_op: OpId::at(&[0]),
            })
        );
    }

    /// ⭐ THE STORE ARM IS THE SAME BODY UNDER A DIFFERENT ACCESSOR NAME — the store's operand list is
    /// `getMapOperands()` and the load's is `getMapIndices()`. `bf16-pt.mlir:161` stores through the
    /// same `#map4` view (`:158`) at `[0, %arg9 + %arg8 * 8, 0]`, two operands instead of three, so
    /// the composite takes TWO dimensions: the arity of the answer follows the ACCESS, not the view.
    #[test]
    fn an_agen_store_composes_the_same_way_over_its_own_operand_count() {
        let scope = vec![
            row_view(),
            DfirOp::Agen(agen::Op::VectorStore {
                value: Val(48),
                view: Val(52),
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(Val(9), 1), (Val(8), 8)], 0),
                    Index::Const(0),
                ],
                view_ty: row_view_ty(),
                ty: V64,
                dbg_name: None,
            }),
        ];
        assert_eq!(
            layout_map_and_indices(&OpId::at(&[1]), &scope),
            Some(LayoutAndIndices {
                layout_map: AffineMap {
                    dims: 2,
                    syms: 0,
                    results: vec![
                        AffineExpr::dim(0)
                            .plus(AffineExpr::dim(1).times(8))
                            .times(64)
                    ],
                },
                operands: vec![Val(9), Val(8)],
                logical_view_op: OpId::at(&[0]),
            })
        );
    }

    /// ⛔⛔ THE SAME VIEW AND THE SAME SUBSCRIPTS THROUGH A `vector.load` COMPOSE NOTHING — the
    /// layout comes back VERBATIM, `#map4` and not `#map4 ∘ anything`. That asymmetry is the
    /// reference's own: its two `vector` arms take `getIndices()` and the view's `getLayoutMap()` and
    /// stop (`VectorOperands.cpp:893-898` and `:910-915`), because `Vector_LoadOp` carries no order map and
    /// no access
    /// map to compose with. A port that composed here "for consistency" would address a different
    /// element of every view.
    #[test]
    fn a_plain_vector_load_takes_the_views_layout_verbatim() {
        let scope = vec![
            row_view(),
            DfirOp::Vector(vector::Op::Load {
                result: Val(53),
                base: Val(52),
                indices: row_subscripts(),
                base_ty: row_view_ty(),
                ty: V64,
            }),
        ];
        assert_eq!(
            layout_map_and_indices(&OpId::at(&[1]), &scope),
            Some(LayoutAndIndices {
                layout_map: layout_map4(),
                operands: vec![Val(12), Val(11), Val(10)],
                logical_view_op: OpId::at(&[0]),
            })
        );
    }

    /// `%lrf_memory_fp16 = dataflow.get_logical_memory_view %lrf_memory_unit, %c0
    /// {layout_map = affine_map<(i, j) -> (64 * i + j)>} : index, index, memref<8x64xf16>`
    /// (`sfp-to-sfp-ring.mlir:168-170`).
    ///
    /// ⭐ THE CONSTANT MOVES TO THE RIGHT AND THAT IS THE PARSER, NOT US: MLIR builds `64 * i`
    /// through `AffineExpr::operator*`, whose `simplifyMul` canonicalises the constant term to the
    /// RHS, so the map the attribute holds — and the one this island prints — is `d0 * 64 + d1`. It is
    /// the reason [`AffineExpr::added`] and [`AffineExpr::scaled`] exist beside the verbatim pair.
    fn lrf_view() -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: Val(80),
            from: Val(81),
            start: Val(82),
            layout: AffineMap {
                dims: 2,
                syms: 0,
                results: vec![AffineExpr::dim(0).times(64).plus(AffineExpr::dim(1))],
            },
            ty: MemRef {
                shape: vec![8, 64],
                elem: ElemType::F16,
            },
        })
    }

    /// ⭐ AND A CONSTANT-SUBSCRIPTED ACCESS CONTRIBUTES NO OPERANDS AT ALL.
    /// `sfp-to-sfp-ring.mlir:171` is `%data1 = vector.load %lrf_memory_fp16[%c4, %c0] :
    /// memref<8x64xf16>, vector<64xf16>`: in vendor MLIR those subscripts are two `arith.constant`
    /// results and so two entries of `getIndices()`, and [`Index::Const`] is this island's folded form
    /// of exactly that. The layout still comes back as the view's, untouched.
    #[test]
    fn a_constant_subscripted_vector_load_has_no_operands() {
        let DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            layout: lrf_layout,
            ty: lrf_ty,
            ..
        }) = lrf_view()
        else {
            unreachable!("`lrf_view` is a logical memory view")
        };
        let scope = vec![
            lrf_view(),
            DfirOp::Vector(vector::Op::Load {
                result: Val(83),
                base: Val(80),
                indices: vec![Index::Const(4), Index::Const(0)],
                base_ty: lrf_ty,
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            }),
        ];
        assert_eq!(
            layout_map_and_indices(&OpId::at(&[1]), &scope),
            Some(LayoutAndIndices {
                layout_map: lrf_layout,
                operands: Vec::new(),
                logical_view_op: OpId::at(&[0]),
            })
        );
    }

    /// ⛔ A BASE THAT IS NOT A `dataflow.get_logical_memory_view` ANSWERS NOTHING. The reference's
    /// `getDefiningOp<dataflow::GetLogicalMemoryViewOp>()` is a `dyn_cast`, so it yields null there
    /// and the very next line calls `getLayoutMap()` on it (`VectorOperands.cpp:902-904`). `None` is
    /// the classification that replaces the dereference.
    #[test]
    fn an_access_whose_base_is_not_a_logical_view_answers_none() {
        let scope = vec![
            dense(Val(52)),
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: Val(53),
                view: Val(52),
                indices: row_subscripts(),
                view_ty: row_view_ty(),
                ty: V64,
                multicast_info: None,
            }),
        ];
        assert_eq!(layout_map_and_indices(&OpId::at(&[1]), &scope), None);
    }

    /// ⛔ AND SO DOES AN OP THAT IS NONE OF THE FOUR MEMORY ACCESSES — the reference's `else` arm,
    /// `op->emitOpError("can't extract memory layout map or indices.")`. Asked about the VIEW itself,
    /// which defines a memref but reads none.
    #[test]
    fn an_op_that_is_not_a_memory_access_answers_none() {
        let scope = vec![row_view()];
        assert_eq!(layout_map_and_indices(&OpId::at(&[0]), &scope), None);
    }
    // ── e232_getOperandFromLoadOrStoreOp / e233_eraseOperands / e234_setValue ──────────────────

    /// `%c0 = arith.constant 0 : index` binding `result` — a view's start address.
    fn const_index(result: Val, value: i64) -> DfirOp {
        DfirOp::Arith(arith::Op::Constant { result, value })
    }

    /// ⭐ THE VENDOR'S TWO CASES, ONE PER FILE, AND THE SLICE INDEX IS COMPUTED NOT COPIED.
    /// `sfp-to-sfp-ring.mlir:156`,`:167-171` loads `%lrf_memory_fp16[%c4, %c0]` where `%c4` holds
    /// **3** and the view is `(i,j)->(64*i+j)` over `memref<8x64xf16>`: `(0 + 192) * 16 / 1024 = 3`,
    /// and its CHECK is `opA = #sentient<compute_port lrf3>` (`:84`). `int8-genkg3-pt.mlir:123-133`
    /// stores at `[1, 0]` of `(i,j)->(128*i+j)` over `memref<2x128xi8>`: `(0 + 128) * 8 / 1024 = 1`,
    /// CHECKed as `ResultForwarding = [#sentient<compute_port irf1>]` (`:15`).
    #[test]
    fn a_load_and_a_store_name_the_slice_their_folded_address_lands_in() {
        let lrf = vec![
            const_index(Val(82), 0),
            DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                result: Val(81),
                of: Val(90),
                which: dataflow::LocalUnit::SfpLrf,
            }),
            lrf_view(),
            DfirOp::Vector(vector::Op::Load {
                result: Val(83),
                base: Val(80),
                indices: vec![Index::Const(3), Index::Const(0)],
                base_ty: MemRef {
                    shape: vec![8, 64],
                    elem: ElemType::F16,
                },
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            }),
        ];
        let loaded = VectorOperand::from_load_or_store_op(&OpId::at(&[3]), &lrf)
            .expect("the vendor's LRF load lowers");
        assert_eq!(loaded.kind, VectorOperandType::Lrf);
        assert_eq!(
            loaded.values,
            vec![OperandValue::Slice(RegisterSlice::Lrf(sen::LrfIndex::L3))]
        );

        let irf_ty = MemRef {
            shape: vec![2, 128],
            elem: ElemType::Int(8),
        };
        let irf = vec![
            const_index(Val(82), 0),
            DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                result: Val(81),
                of: Val(90),
                which: dataflow::LocalUnit::PtIrf,
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(80),
                from: Val(81),
                start: Val(82),
                layout: AffineMap {
                    dims: 2,
                    syms: 0,
                    results: vec![AffineExpr::dim(0).times(128).plus(AffineExpr::dim(1))],
                },
                ty: irf_ty.clone(),
            }),
            DfirOp::Agen(agen::Op::VectorStore {
                value: Val(83),
                view: Val(80),
                indices: vec![Index::Const(1), Index::Const(0)],
                view_ty: irf_ty,
                ty: Vector {
                    len: 128,
                    elem: ElemType::Int(8),
                },
                dbg_name: None,
            }),
        ];
        let stored = VectorOperand::from_load_or_store_op(&OpId::at(&[3]), &irf)
            .expect("the vendor's IRF store lowers");
        assert_eq!(stored.kind, VectorOperandType::Irf);
        assert_eq!(
            stored.values,
            vec![OperandValue::Slice(RegisterSlice::Irf(IrfIndex::I1))]
        );
    }

    /// ⭐ IT REPLACES, WHICH IS WHY `OperandReuse` RE-VALUING TO `latch` LOSES THE SLICE INDEX.
    #[test]
    fn setting_a_value_drops_the_one_the_operand_held() {
        let mut held = VectorOperand::new(
            VectorOperandType::Lrf,
            OperandValue::Slice(RegisterSlice::Lrf(sen::LrfIndex::L3)),
            OpId::at(&[0]),
        );
        held.set_value(OperandValue::Port(sen::Port::Latch));
        assert_eq!(held.values, vec![OperandValue::Port(sen::Port::Latch)]);
    }

    /// ⭐⭐ THE WHOLE CHAIN GOES, AND A `latch` KEEPS ITS DEFINITION. The dense constant at `[0]` is
    /// read only through a `neg` whose own reader is an unused `fast_exp`, so all three are erased;
    /// re-valued to [`sen::Port::Latch`] the same operand keeps `[0]` and loses only its consumers.
    #[test]
    fn an_operand_whose_only_readers_died_is_erased_unless_it_is_a_latch() {
        let chain = || vec![dense(Val(0)), neg(Val(1), Val(0)), fast_exp(Val(2), Val(1))];

        let mut scope = chain();
        erase_operands(
            &[Some(operand(VectorOperandType::Constant, &[0]))],
            &mut scope,
        );
        assert_eq!(scope, Vec::new());

        let mut latched = operand(VectorOperandType::Link, &[0]);
        latched.set_value(OperandValue::Port(sen::Port::Latch));
        let mut scope = chain();
        erase_operands(&[Some(latched)], &mut scope);
        assert_eq!(scope, vec![dense(Val(0))]);
    }

    /// 🎯 278/384 — THE VENDOR'S OWN TWO NFWD SHUFFLES, RE-TAGGED OVER THE PARENT'S ANSWER.
    /// `Conversion/VectorChainToSentientPESFP/nfwd_for_binary_op.mlir:155-159` is
    /// `shuffle {indices = [2,3,0,1,6,7,6,7]}` and `{indices = [4,5,3,3,5,5,6,7]}` feeding one binary,
    /// whose expectation is `opA = nfwd0, opADataID = 1` beside `opB = nfwd2, opBDataID = 2` (`:48`) —
    /// the ports come from here and the data IDs from the parent, which is why the parent's position
    /// has to survive the re-tag. The third shuffle is neither pattern and falls through to its send.
    #[test]
    fn the_two_vendor_nfwd_patterns_re_tag_the_parents_operand() {
        let (to, from) = Link::<Sfp, Lxsu>::between(Val(20), Val(21)).ends();
        let shuffle = |result: Val, indices: Vec<i32>| {
            DfirOp::VectorChain(vc::Op::Shuffle {
                result,
                input: Val(1),
                variable: Vec::new(),
                pad: Vec::new(),
                mask: None,
                indices,
                repetition: 8,
                input_ty: V64,
                ty: V64,
            })
        };
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(1),
                from,
                ty: V64,
            }),
            shuffle(Val(2), vec![2, 3, 0, 1, 6, 7, 6, 7]),
            shuffle(Val(3), vec![4, 5, 3, 3, 5, 5, 6, 7]),
            shuffle(Val(4), vec![0, 1, 2, 3, 4, 5, 6, 7]),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(4),
                ty: V64,
                dir: None,
            }),
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(21),
                residency: Residency::Corelet {
                    core: Core::checked(0).expect("the arch has core 0"),
                    corelet: Corelet::checked(0).expect("the arch has corelet 0"),
                },
                unit: DfirUnit::Lxsu,
                num_folds: None,
            }),
        ];
        // The parent's answer, which the reference builds and then overwrites two fields of.
        let mut parent_operand = |at: &OpId, _comp: ComputeComp| {
            Some(VectorOperand::new(
                VectorOperandType::Lrf,
                OperandValue::Slice(RegisterSlice::Lrf(sen::LrfIndex::L0)),
                at.clone(),
            ))
        };

        for (path, port) in [(1u32, sen::Port::Nfwd0), (2, sen::Port::Nfwd2)] {
            let operand = VectorOperand::from_shuffle_op::<Dd2>(
                &OpId::at(&[path]),
                ComputeComp::Pe,
                &scope,
                &mut parent_operand,
            )
            .expect("the parent resolves");
            assert_eq!(operand.kind, VectorOperandType::Nfwd);
            assert_eq!(operand.name(), Some(port));
            // ⭐ THE PARENT'S POSITION, NOT THE SHUFFLE'S — this is `opADataID = 1`.
            assert_eq!(operand.op, OpId::at(&[0]));
        }

        // Neither pattern: the answer comes from the shuffle's single user, a send to the LXSU.
        let operand = VectorOperand::from_shuffle_op::<Dd2>(
            &OpId::at(&[3]),
            ComputeComp::Pe,
            &scope,
            &mut parent_operand,
        )
        .expect("the send resolves");
        assert_eq!(operand.kind, VectorOperandType::Link);
        assert_eq!(operand.name(), Some(sen::Port::Lx));
        assert_eq!(operand.op, OpId::at(&[4]));
    }

    /// 🎯 320/384 — `getOperand` IS THE SCC CUT, and the two tests below drive the whole dispatch
    /// through it: a cast over a uniformized receive resolves only because the recursion closes.
    fn resolved(op: &OpId, comp: ComputeComp, scope: &[DfirOp]) -> Option<VectorOperand> {
        VectorOperand::operand::<Sen1p5>(op, comp, true, scope)
    }

    /// 🎯 320/384 — THE KNOT TIES: `%1 = vectorchain.neg %0` over an `arith.constant dense` resolves
    /// in two hops, so the recursion `getOperandFromNegOp` needs is really reachable and terminates.
    ///
    /// ⛔ AND THE ANSWER IS THE CONSTANT'S OWN POSITION, not the negation's: entry 168 forwards the
    /// operand untouched, precisions and all.
    #[test]
    fn a_negated_constant_resolves_through_the_closed_recursion() {
        let scope = vec![dense(Val(0)), neg(Val(1), Val(0))];
        let answer =
            VectorOperand::operand::<Sen1p5>(&OpId::at(&[1]), ComputeComp::Sfp, true, &scope)
                .expect("the negation forwards the constant");
        assert_eq!(answer.kind, VectorOperandType::Constant);
        assert_eq!(answer.op, OpId::at(&[0]));
        assert!(answer.orig_precision.is_some());
        assert_eq!(answer.orig_precision, answer.on_the_fly_conv_precision);
    }

    /// 🎯 304/384 — THE VENDOR'S OWN CAST OVER A UNIFORMIZED PT RECEIVE, on `SENARCH=sen1p5`.
    /// `Conversion/VectorChainToSentientPESFP/mixed_precision.mlir:628-629` maps 64 core handles to
    /// `ptrow7` `get_unit`s and reads one back with `uniform.query_map`; `:745-747` is
    /// `%270 = dataflow.receive %195 : vector<64xf16>`, `%271 = vectorchain.cast %270 :
    /// vector<64xf16>, vector<64xf32>` and a MAC over `%271`. The PE's expectation is
    /// `opA = #sentient<compute_port pt>, opAPrecision = #sentient<precision fp24>` (`:332`).
    ///
    /// ⛔ SO THE ANSWER CONTRADICTS THE IR TWICE: the original precision is `fp24` where the receive
    /// says `f16`, and the on-the-fly precision is the CAST's `fp32` — while the operand's position
    /// stays the RECEIVE's, which is what the golden's `opADataID` is counted from.
    #[test]
    fn a_cast_over_a_uniformized_pt_receive_reads_fp24_and_converts_to_fp32() {
        let f16 = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let f32 = Vector {
            len: 64,
            elem: ElemType::F32,
        };
        let (_, from) = Link::<PtRowUnit<7>, Pe>::between(Val(195), Val(196)).ends();
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(193),
                residency: Residency::Corelet {
                    core: Core::checked(31).expect("the arch has core 31"),
                    corelet: Corelet::checked(1).expect("the arch has corelet 1"),
                },
                unit: DfirUnit::PtRow(Row::checked(7).expect("this arch's PT has row seven")),
                num_folds: None,
            }),
            DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(194),
                pairs: vec![(Val(0), Val(193))],
            }),
            DfirOp::Uniform(uniform::Op::QueryMap {
                result: Val(195),
                map: Val(194),
                key: Val(0),
            }),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(270),
                from,
                ty: f16,
            }),
            DfirOp::VectorChain(vc::Op::Cast {
                result: Val(271),
                input: Val(270),
                input_ty: f16,
                ty: f32,
            }),
            // `%273 = vectorchain.multiply_and_accumulate %271, %cst, %cst_0[..]` — the cast's ONE
            // use, and the compute that asks. Its reduction map is not consulted from here.
            DfirOp::VectorChain(vc::Op::MultiplyAccumulate {
                result: Val(273),
                a: Val(271),
                b: Val(300),
                acc: Val(301),
                reduction_map: AffineMap {
                    dims: 1,
                    syms: 0,
                    results: vec![AffineExpr::dim(0)],
                },
                operand_ty: f32,
                ty: f32,
            }),
        ];

        let answer = VectorOperand::with_precision::<Sen1p5>(
            &OpId::at(&[4]),
            ComputeComp::Pe,
            true,
            &scope,
            &mut |op: &OpId, comp| resolved(op, comp, &scope),
        );

        assert!(answer.precision_converted);
        let operand = answer
            .operand
            .expect("the receive behind the cast resolves");
        assert_eq!(operand.kind, VectorOperandType::Link);
        assert_eq!(operand.name(), Some(sen::Port::Pt));
        assert_eq!(operand.orig_precision, Some(sen::Precision::Fp24));
        assert_eq!(
            operand.on_the_fly_conv_precision,
            Some(sen::Precision::Fp32)
        );
        assert_eq!(operand.op, OpId::at(&[3]));
        assert_eq!(operand.splat, None);

        // ⭐ AND THE COMPUTE ITSELF IS NOT AN OPERAND: `MultiplyAndAccumulateOp` is the explicit
        // `std::nullopt` (`VectorOperands.cpp:641-643`).
        let compute = VectorOperand::with_precision::<Sen1p5>(
            &OpId::at(&[5]),
            ComputeComp::Pe,
            true,
            &scope,
            &mut |op: &OpId, comp| resolved(op, comp, &scope),
        );
        assert_eq!(compute.operand, None);
        assert!(!compute.precision_converted);
    }

    /// 🎯 343/384 — A CAST FORWARDS ITS INPUT'S OPERAND UNTOUCHED, AND THAT IS EXACTLY WHAT SEPARATES
    /// IT FROM ENTRY 304'S OWN `CastOp` ARM: two users decline there (`hasOneUse()`, `:542`) and
    /// forward here, and neither precision is rewritten.
    #[test]
    fn a_cast_forwards_its_inputs_operand_untouched() {
        let scope = vec![
            dense(Val(0)),
            DfirOp::VectorChain(vc::Op::Cast {
                result: Val(1),
                input: Val(0),
                input_ty: V,
                ty: V,
            }),
            // TWO users of the cast, which is the one-use guard's negative case.
            fast_exp(Val(2), Val(1)),
            fast_exp(Val(3), Val(1)),
        ];
        let answer = VectorOperand::from_cast_op(
            &OpId::at(&[1]),
            ComputeComp::Sfp,
            &scope,
            &mut |op: &OpId, comp| resolved(op, comp, &scope),
        )
        .expect("the cast forwards the constant behind it");
        assert_eq!(answer.kind, VectorOperandType::Constant);
        assert_eq!(answer.op, OpId::at(&[0]));

        // ⛔ THE GUARDED ARM DECLINES THE SAME POSITION, and does not even set the conversion flag —
        // it is set INSIDE `hasOneUse()`.
        let guarded = VectorOperand::with_precision::<Sen1p5>(
            &OpId::at(&[1]),
            ComputeComp::Sfp,
            true,
            &scope,
            &mut |op: &OpId, comp| resolved(op, comp, &scope),
        );
        assert_eq!(guarded.operand, None);
        assert!(!guarded.precision_converted);

        // ⛔ `DT_CHECK(isa<CastOp>(op))` — a position holding anything else has no operand to report.
        assert_eq!(
            VectorOperand::from_cast_op(
                &OpId::at(&[0]),
                ComputeComp::Sfp,
                &scope,
                &mut |op: &OpId, comp| resolved(op, comp, &scope),
            ),
            None
        );
    }
}
