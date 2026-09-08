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
use crate::islands::dataflow_ir::dialects::{agen, arith, dataflow, vector};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, MemRef};
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

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
/// # ⚠️ PARTIAL BY DESIGN — `splat_` IS NOT HERE YET
///
/// ⭐ THE MEMBER ARRIVES WITH THE UNIT THAT READS IT. `splat_` is *"to capture the select
/// semantics"* (`VectorOperands.hpp:112`) and only `e304_getOperandWithPrecision` ever touches it;
/// that entry is not this wave's, and what shape the field wants is a decision for its porter against
/// its own callers rather than a guess made here.
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
        }
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
                | DfirUnit::LxVirtualIbr => todo!(
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
                | DfirUnit::CrossPtnLink => {
                    todo!("Unsupported receive unit for PE/SFP (VectorOperands.cpp:76)")
                }
            },
        };

        VectorOperand::new(VectorOperandType::Link, OperandValue::Port(link), receive_op)
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
                | DfirUnit::CrossPtnLink => todo!(
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
                | DfirUnit::CrossPtnLink => todo!(
                    "Unsupported destination for PE/SFP FMA (VectorOperands.cpp:136-139)"
                ),
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
fn is_arith_constant(op: &OpId, scope: &[DfirOp]) -> bool {
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
/// [`ConstantOperandValue`]. `constValToField` has no answer for a fifth value — its `default:` is an
/// unconditional throw (see [`const_val_to_field`]) — and entry 073 already made that domain a type
/// rather than a runtime refusal. The only caller that passes `true` is
/// `getOperandFromShuffleOp`'s trivial-shuffle branch (`VectorOperands.cpp:333-334`, entry 278), which
/// is exactly the one that has established the shuffle is a recognised splat.
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
    /// constructed as `VectorOperand(operand_type, "xrf", op)` (`VectorOperands.cpp:218-220`) — the
    /// only place one is built — so its first value already IS `"xrf"` and the `else` would return the
    /// same string. It is also the only kind with no slice index, because the XRF is one register and
    /// the arm above it skips the whole slice computation for exactly that reason (`:208`, `:218`).
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
    /// (`dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:264-293`)
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
            | arith::Op::Logic { .. }
            // ⛔ `arith.sitofp`/`arith.fptosi` BIND A VECTOR AND ARE STILL NOT CONSTANTS — the
            // reference's parameter is an `mlir::arith::ConstantOp&`, which neither is.
            | arith::Op::Convert { .. } => return None,
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

#[cfg(test)]
mod unit_tests {
    use super::{BitstreamConstant, IrfIndex, LayoutAndIndices, RegisterSlice};
    use super::{ComputeComp, ConstantOperandValue, OpId, OperandValue, VectorOperand};
    use super::{
        VectorOperandType, const_val_to_field, erase_op, layout_map_and_indices, same_block,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::units::{DfirUnit, Row};
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{Index, Op as DfirOp, Val};
    use crate::islands::dataflow_ir::dialects::{affine, agen, arith, dataflow, vector};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
    use crate::islands::sentient::dialects::sentient as sen;

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
            mask: None,
            dbg_name: None,
            result,
            input,
            input_ty: V,
            ty: V,
        })
    }

    /// `vectorchain.floor %input` binding `result`.
    fn floor(result: Val, input: Val) -> DfirOp {
        DfirOp::VectorChain(vc::Op::Floor {
            mask: None,
            dbg_name: None,
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
        let on_sen1p5 =
            VectorOperand::from_receive_op::<Sen1p5>(DfirUnit::Sfp, ComputeComp::Sfp, OpId::at(&[4]));
        assert_eq!(port(&on_sen1p5), sen::Port::SfpRing);
    }

    /// ⭐ `xrf_increments.mlir:422` sends to `%5`, a `type = "ptrow1"` unit, and `:81`/`:88` expect
    /// `ResultForwarding = [#sentient<compute_port south>]`. A send to the PE takes the same arm.
    #[test]
    fn a_pt_sends_to_the_rows_and_the_pe_over_south() {
        for peer in [row(1), row(0), DfirUnit::Pe] {
            let operand =
                VectorOperand::from_send_op::<Dd2>(peer, ComputeComp::Pt, OpId::at(&[5]));
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
            let operand =
                VectorOperand::from_send_op::<Dd2>(peer, ComputeComp::Pe, OpId::at(&[6]));
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
        assert_eq!(const_val_to_field(ConstantOperandValue::One), sen::Port::One);
        assert_eq!(const_val_to_field(ConstantOperandValue::Two), sen::Port::Two);
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
                dbg_name: None,
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
    /// `VectorOperand(operand_type, "xrf", op)` (`VectorOperands.cpp:218-220`), so its value already
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
                access: agen::Access::OfView,
                result: Val(53),
                view: Val(52),
                indices: row_subscripts(),
                view_ty: row_view_ty(),
                ty: V64,
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
                dbg_name: None,
                access: agen::Access::OfView,
                value: Val(48),
                view: Val(52),
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(Val(9), 1), (Val(8), 8)], 0),
                    Index::Const(0),
                ],
                view_ty: row_view_ty(),
                ty: V64,
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
                access: agen::Access::OfView,
                result: Val(53),
                view: Val(52),
                indices: row_subscripts(),
                view_ty: row_view_ty(),
                ty: V64,
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
}
