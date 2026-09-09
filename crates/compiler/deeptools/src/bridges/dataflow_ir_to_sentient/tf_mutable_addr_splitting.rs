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

//! `MutableAddrSplitting.cpp` — 23 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e111_getMaxMutableRange` | 111/384 | 8 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673` |
//! | `e112_getMaxImmutableRange` | 112/384 | 8 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683` |
//! | `e113_isEligibleForSplitting` | 113/384 | 20 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832` |
//! | `e114_sortDataBasedOnWeight` | 114/384 | 4 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855` |
//! | `e115_createNewMemViewWithMod` | 115/384 | 12 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966` |
//! | `e184_getLoopTripCount` | 184/384 | 56 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743` |
//! | `e185_hasMutableAddrOverflow` | 185/384 | 14 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801` |
//! | `e186_calculatePartitionSizes` | 186/384 | 97 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862` |
//! | `e187_constructConditionals` | 187/384 | 59 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000` |
//! | `e188_calculateSubscriptsCoefficients` | 188/384 | 11 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188` |
//! | `e189_synthesizeTimeInfo` | 189/384 | 22 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256` |
//! | `e190_createExplicitTimeLoops` | 190/384 | 57 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282` |
//! | `e250_initMASData` | 250/384 | 31 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707` |
//! | `e251_setupForPartitioning` | 251/384 | 6 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819` |
//! | `e252_fillPartitions` | 252/384 | 117 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063` |
//! | `e253_adjustForEvenImmutableAddr` | 253/384 | 47 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204` |
//! | `e289_initialize` | 289/384 | 8 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694` |
//! | `e290_createPartitions` | 290/384 | 10 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983` |
//! | `e306_transformVectorLoad` | 306/384 | 75 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298` |
//! | `e307_transformVectorStore` | 307/384 | 74 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375` |
//! | `e321_transformCompLoadAndStore` | 321/384 | 92 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451` |
//! | `e322_transformCompIndLoadAndStore` | 322/384 | 124 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546` |
//! | `e351_runOnOperation` | 351/384 | 70 | `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226` |

use std::collections::VecDeque;
use std::num::NonZeroU64;

use crate::arch::{Arch, Bounded, Elements, IsaGen, Sticks};
use crate::formats::Bits;
use crate::generated::DataType;
use crate::islands::dataflow_ir::dialects::{
    self, Index, Op as DfirOp, Val, affine, agen, arith, dataflow, defining_op, scf, symbol, uses,
};
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef,
};
use crate::islands::dataflow_ir::{Program, ValueMapping, Values};
use crate::units::DfirUnit;

use super::agen_access_details::{
    AccessContainer, AccessDetailsAffine, AccessDetailsAffineComposite, ConstructedDetails,
    MemoryOperandIndex, TimeBound, TimeStepsInfo, construct_time_steps_info,
};
use super::tf_transform_paged_mem_view_impl::{
    UseChain, VectorLoadOp, VectorStoreOp, clone_store_with_new_access_info,
    clone_use_chain_to_new_store, create_new_mem_op, erase_vector_load_and_use_chain,
    erase_vector_store_and_use_chain, get_use_chain, indices_from_map, vector_store_use_chain,
};
use super::tf_utils::{LoopBound, LoopStep, constant_index, owner_of_block_arg};

/// WHICH HALF OF THE L3 — the `DT_CHECK(is_any_of(comp, L3LU, L3SU))` both range queries open with.
///
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:674` and `:685`.
///
/// ⛔⛔ A TYPE, NOT A CHECK. `DT_CHECK` aborts the compiler; the fact it is asserting is that these
/// two queries are only ever asked about an external memory unit, and an external address register is
/// something only the L3 halves have. Making that the parameter means the abort has no caller left to
/// have — a `SenComponents` argument can be `PE`, an [`L3Half`] cannot.
///
/// ⭐⭐ AND THE HALF DOES NOT CHANGE THE ANSWER, WHICH IS WORTH KNOWING. `regInfoPerUnit` declares
/// `L3LU`'s and `L3SU`'s registers in two separate blocks, and for `EAR` and `EBR` the two blocks are
/// identical: `{16, 21, 32, UNSIGNED, true}` at `sysdef.cpp:313-314` and `:336-337`, and the same
/// arch-split `EBR` at `:320-321`/`:326-327` and `:343-344`/`:349-350`. So the component selects a
/// table row whose contents are the same either way, and its ONLY function in these two functions is
/// the `DT_CHECK`. The parameter stays because the caller must still prove which unit it is asking
/// about — see [`max_mutable_range`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3Half {
    /// `SenComponents::L3LU` — the load half.
    Load,
    /// `SenComponents::L3SU` — the store half.
    Store,
}

impl L3Half {
    /// WHICH HALF A UNIT KIND IS, OR NONE FOR A COMPONENT THE `DT_CHECK` WOULD ABORT ON.
    ///
    /// ⛔ EXHAUSTIVE, NO WILDCARD, for the same reason
    /// [`moves_memory`](crate::islands::dataflow_ir::Units::moves_memory) is: a new unit
    /// kind must state whether it is an L3 half rather than silently inherit "no" and take a
    /// splitting decision meant for external memory.
    #[must_use]
    pub const fn of(unit: DfirUnit) -> Option<Self> {
        match unit {
            DfirUnit::L3lu => Some(L3Half::Load),
            DfirUnit::L3su => Some(L3Half::Store),
            DfirUnit::Sfp
            | DfirUnit::Pe
            | DfirUnit::PtRow(_)
            | DfirUnit::Lxlu
            | DfirUnit::Lxsu
            | DfirUnit::Lx
            | DfirUnit::Hbm
            | DfirUnit::L0lu
            | DfirUnit::L0su
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
}

/// AN EXTERNAL ADDRESS RANGE, IN BITS.
///
/// ⛔⛔ BITS, BECAUSE THAT IS THE UNIT BOTH OVERRIDES ARE DOCUMENTED IN AND THE ONE THE CALLERS
/// DIVIDE. The two `cl::opt`s say *"Measured in bits"* (`MutableAddrSplitting.cpp:56-68`) and every
/// consumer immediately does `range / elem_size_in_bits` to get an element count
/// (`:804`, `:874-876`, `:927`). A range held in sticks or bytes would put that conversion at each of
/// those three sites instead of one — see [`AddrRange::elements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddrRange(u64);

impl AddrRange {
    /// THE RANGE A `bits`-WIDE ADDRESS REGISTER SPANS — `pow(2, bitSize) * bytesPerStick * 8`.
    ///
    /// ⭐ ONE REGISTER VALUE PER STICK. The register counts sticks, so its span in bits is
    /// `2^bitSize` sticks times the stick's bytes times eight — the arithmetic both range queries
    /// share (`:676-679` and `:687-690`).
    ///
    /// ⛔ THE REFERENCE COMPUTES THIS IN FLOATING POINT AND GETS AN EXACT ANSWER. `pow(2, 21)` is a
    /// `double`; every value it can return here is a power of two, which a `double` represents
    /// exactly, so the `int64_t` the function returns is not rounded. Shifting instead is the same
    /// number without the round trip — and it cannot overflow because [`Arch::L3_EAR_BITS`] is a
    /// [`Bounded<53>`], whose bound is the reference's own return type.
    #[must_use]
    pub fn of_register<A: Arch>(bits: Bounded<53>) -> Self {
        let sticks = Sticks(1u64 << bits.get());
        Self(A::sticks_to_bytes(sticks).0 * 8)
    }

    /// AN OVERRIDE GIVEN ON THE COMMAND LINE, WHICH IS ALREADY IN BITS.
    #[must_use]
    pub const fn of_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// THE RANGE ITSELF, in bits.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// HOW MANY ELEMENTS OF ONE FORMAT FIT IN IT — `getMaxMutableRange(comp) / elem_size_in_bits`.
    ///
    /// ⭐ THE DIVISION ALL THREE CONSUMERS DO (`:804`, `:876`, `:927`), once. DataflowIR addresses are
    /// in element granularity (`Dataflow.td:250`), so a range in bits is not comparable with an
    /// address until it has crossed this.
    ///
    /// ⛔ THE FORMAT, NOT ITS WIDTH, SO THE DIVISOR CANNOT BE ZERO. `elem_size_in_bits` is a bare
    /// `int` in the reference and a zero-width element would divide by it; every
    /// [`DataType`] answers at least four bits ([`DataType::bits`]), which makes that unrepresentable
    /// rather than unlikely.
    #[must_use]
    pub fn elements(self, elem: DataType) -> Elements {
        Elements(self.0 / u64::from(elem.bits().0))
    }
}

/// `-dcc-mutable-addr-splitting-max-mutable-size`, `cl::init(-1)` — `MutableAddrSplitting.cpp:56-61`.
///
/// ⛔⛔ `-1` IS `None`, NOT A NEGATIVE SIZE. The reference stores the flag as an `int64_t` and reads
/// the sentinel back as `MaxMutableSize < 0` (`:676`), so "unset" and "set to a size" share one
/// variable and every reader has to know which comparison means which. An `Option` says it once.
///
/// ⭐ AND IT IS A CONST BECAUSE IT IS NOT A RUNTIME VALUE HERE. Nothing in this crate parses
/// `dcc-opt`'s command line; the flag exists so a person can override the register table by hand
/// while debugging, and its default is the whole of its behaviour in a compiled pipeline. Changing it
/// is a code change, which is the visibility this crate wants for a value that decides how a program
/// is split.
pub const MAX_MUTABLE_SIZE: Option<AddrRange> = None;

/// `-dcc-mutable-addr-splitting-max-immutable-size`, `cl::init(-1)` — `MutableAddrSplitting.cpp:63-68`.
///
/// See [`MAX_MUTABLE_SIZE`].
pub const MAX_IMMUTABLE_SIZE: Option<AddrRange> = None;

/// Replaces: e111_getMaxMutableRange
///
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673-681`:
///
/// ```text
/// DT_CHECK(is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU));
/// auto &sys_def = dcc_ext_ctx_.dsc_global_->sysDef;
/// return MaxMutableSize < 0
///            ? pow(2, sys_def.regInfoPerUnit.at(comp).at(RegType::EAR).bitSize) *
///                  sys_def.bytesPerStick * 8
///            : MaxMutableSize;
/// ```
///
/// ⭐ THE **MUTABLE** HALF IS THE `EAR`. An external address is an immutable base plus a mutable
/// offset; the offset lives in the External Address Register, so how far a transfer's address may
/// travel before the pass must split it is exactly what that register can hold
/// ([`Arch::L3_EAR_BITS`], 21 bits on every arch).
///
/// ⛔ THE OVERRIDE WINS WHEN IT IS SET, AND IT IS TAKEN AS GIVEN. `MaxMutableSize` is already in bits
/// and bypasses the register table entirely — including the arch — which is why it exists: it is the
/// hand-hold for a machine whose table is wrong. See [`MAX_MUTABLE_SIZE`].
///
/// # Arguments
///
/// * `_half` — which L3 half is asking. Unread, because the two table rows it selects between are
///   identical; present because the caller must still prove it is asking about one of them. See
///   [`L3Half`].
#[must_use]
pub fn max_mutable_range<A: Arch>(_half: L3Half) -> AddrRange {
    match MAX_MUTABLE_SIZE {
        Some(given) => given,
        None => AddrRange::of_register::<A>(A::L3_EAR_BITS),
    }
}

/// Replaces: e112_getMaxImmutableRange
///
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683-692` — the same function over the other
/// register:
///
/// ```text
/// DT_CHECK(is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU));
/// auto &sys_def = dcc_ext_ctx_.dsc_global_->sysDef;
/// return MaxImmutableSize < 0
///            ? pow(2, sys_def.regInfoPerUnit.at(comp).at(RegType::EBR).bitSize) *
///                  sys_def.bytesPerStick * 8
///            : MaxImmutableSize;
/// ```
///
/// ⛔⛔ TWO FUNCTIONS BECAUSE THE REGISTER IS DIFFERENT, AND THE DIFFERENCE IS NOT A CONSTANT FACTOR.
/// The mutable range reads the `EAR` (21 bits, every arch) and the immutable range reads the `EBR`
/// (30 bits on RCUDD1A, **32** from SEN1P5 — `sysdef.cpp:320-327`). So the immutable space is 512×
/// the mutable one on DD2 and 2048× on SEN1P5, and a port that shared one query between the two
/// callers would silently pick one arch's ratio for both.
///
/// ⭐ WHAT THE CALLER DOES WITH IT: `immutable_space = (getMaxImmutableRange(comp) /
/// elem_size_in_bits) - max_immutable`, checked against the mutable it wants to shift (`:925-930`) —
/// the reason the immutable range matters at all is that splitting moves address out of the mutable
/// half into the immutable one, and this is the room left there.
///
/// # Arguments
///
/// * `_half` — see [`max_mutable_range`].
#[must_use]
pub fn max_immutable_range<A: Arch>(_half: L3Half) -> AddrRange {
    match MAX_IMMUTABLE_SIZE {
        Some(given) => given,
        None => AddrRange::of_register::<A>(A::L3_EBR_BITS),
    }
}

/// A `dataflow.get_logical_memory_view` WHOSE START ADDRESS IS A CONSTANT — everything a clone of it
/// needs, and the PROOF that [`is_eligible_for_splitting`] admitted it.
///
/// ⭐⭐ THIS TYPE IS THE GATE'S OUTPUT AND [`create_new_mem_view_with_mod`]'s INPUT, which is what
/// makes the pass's own precondition checkable. `setupForPartitioning` runs
/// `DT_CHECK(isEligibleForSplitting(all_mem_views))` first (`MutableAddrSplitting.cpp:826`) and every
/// view the partitioning then clones is one of the views that check passed over; carrying the
/// constant start as an `i64` rather than a `Val` is that fact in the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstStartMemView<'a> {
    /// `mem_view_op.getMemory()` — the memory unit viewed.
    pub from: Val,
    /// The value of the `arith.constant` bound to `mem_view_op.getStartAddress()`, in elements.
    pub start: i64,
    /// `layout_map`.
    pub layout: &'a AffineMap,
    /// The view's type.
    pub ty: &'a MemRef,
}

/// WHAT ONE MEMORY VIEW OF A LOAD/STORE CHAIN LOOKS LIKE TO THE SPLITTING GATE.
///
/// ⛔ NOT AN ERROR TYPE. Nothing here is a `Result` and nothing stops: these are the three shapes
/// `isEligibleForSplitting`'s two `dyn_cast`/`isConstant` tests distinguish, and two of them make the
/// gate answer `false` — which is a fact about the program, reported by [`SplittingEligibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitCandidateView<'a> {
    /// A `dataflow.get_logical_memory_view` whose start address is an `arith.constant`.
    ConstantStart(ConstStartMemView<'a>),
    /// A `dataflow.get_logical_memory_view` whose start address is bound by something else — a
    /// toggle (`arith.subi` over an `iter_args` chain), a conditional, or a region argument.
    NonConstantStart,
    /// Bound by an operation that is not a `get_logical_memory_view`, or by no operation at all.
    NotAMemoryView,
}

impl<'a> SplitCandidateView<'a> {
    /// RESOLVES ONE VIEW VALUE AGAINST THE OPS IN SCOPE — the two `dyn_cast`s, in the gate's order.
    ///
    /// ⚠️ `scope` IS THE DEF-USE WALK, WHICH IS MECHANISM. `mem_view.getDefiningOp()` is MLIR asking
    /// a value which operation bound it; this island has no use lists, so the caller passes the ops
    /// in scope and [`defining_op`] scans them. An SSA value is bound exactly once, so the two agree
    /// on every well-formed program.
    ///
    /// ⛔ A REGION ARGUMENT IS [`SplitCandidateView::NotAMemoryView`], NOT A MISSING CASE.
    /// `mem_view.getDefiningOp()` is null for one and `dyn_cast<GetLogicalMemoryViewOp>(nullptr)` is
    /// null too, so the reference takes the `return false` — the same answer, reached the same way.
    #[must_use]
    pub fn resolve(mem_view: Val, scope: &'a [DfirOp]) -> SplitCandidateView<'a> {
        // `dyn_cast<dataflow::GetLogicalMemoryViewOp>(mem_view.getDefiningOp())`.
        let Some(DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            from,
            start,
            layout,
            ty,
            ..
        })) = defining_op(mem_view, scope)
        else {
            return SplitCandidateView::NotAMemoryView;
        };
        // `dcc::utils::isConstant<arith::ConstantOp>(mem_view_op.getStartAddress())`.
        //
        // ⛔ THE `uniform::QueryMapOp` ARM OF `isConstant` COLLAPSES HERE. `Utils.cpp:428-441` also
        // answers true for a query into an immutable mapping whose every value is a constant; there
        // is no `uniform` dialect in this island (a query map is a UNIFORMIZATION artefact, and this
        // crate emits programs already specialised per unit), so the only true case is the
        // `arith.constant` one. If the island ever grows one, this is the arm that grows with it.
        //
        // ⛔ AND `isa<BlockArgument>(val) → false` (`:426`) IS THE `None` ARM below: a start address
        // that is a region argument is not a constant, which is the whole toggle pattern
        // `AddressPinningAndToggle` leaves behind.
        match defining_op(*start, scope) {
            Some(DfirOp::Arith(arith::Op::Constant { value, .. })) => {
                SplitCandidateView::ConstantStart(ConstStartMemView {
                    from: *from,
                    start: *value,
                    layout,
                    ty,
                })
            }
            _ => SplitCandidateView::NonConstantStart,
        }
    }
}

/// WHETHER THE CHAIN'S MEMORY VIEWS MAY BE SPLIT — and, where not, WHICH ONE SAID NO.
///
/// ⛔ NOT AN ERROR TYPE, AND NOT A DIAGNOSTIC EITHER. The reference's caller wraps this in
/// `DT_CHECK(isEligibleForSplitting(all_mem_views))` (`:826`), which aborts the compiler with an
/// assertion and no message. Naming the offending view is strictly more than `false` carried and
/// costs nothing; [`Self::eligible`] is the reference's own boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplittingEligibility {
    /// Every view is a `get_logical_memory_view` with a constant start address.
    Eligible,
    /// This view is bound by something that is not a `get_logical_memory_view`.
    ViewIsNotAMemoryView(Val),
    /// This view's start address is not an `arith.constant`.
    ViewStartAddressIsNotConstant(Val),
}

impl SplittingEligibility {
    /// The reference's own `bool`.
    #[must_use]
    pub const fn eligible(self) -> bool {
        matches!(self, SplittingEligibility::Eligible)
    }
}

/// Replaces: e113_isEligibleForSplitting
///
/// **113/384** `MutableAddrSplittingPass::isEligibleForSplitting` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832` (20L).
///
/// ```cpp
/// bool MutableAddrSplittingPass::isEligibleForSplitting(
///     const SmallVectorImpl<Value> &all_mem_views) const {
///   // Other passes are responsible to remove variability from immutable
///   // addresses. Currently, AddressPinningAndToggle is one such pass. However, it
///   // expects a toggle or conditional immutable address to be used only in one
///   // memory operation and a yield operation. At this time, MutableAddrSplitting
///   // will not support any memory view used in the load/store chain where
///   // splitting is required that does not have a constant start address.
///   for (auto &mem_view : all_mem_views) {
///     // Ineligible for splitting if any of the mem views are not a
///     // GetLogicalMemoryViewOp.
///     auto mem_view_op =
///         dyn_cast<dataflow::GetLogicalMemoryViewOp>(mem_view.getDefiningOp());
///     if (!mem_view_op) return false;
///
///     // Start address must be a constant.
///     if (!dcc::utils::isConstant<arith::ConstantOp>(
///             mem_view_op.getStartAddress()))
///       return false;
///   }
///   return true;
/// }
/// ```
///
/// # ⭐ WHY A CONSTANT START IS THE PRICE OF SPLITTING
///
/// Splitting a transfer whose mutable address has overflowed means emitting the SAME transfer several
/// times over disjoint partitions of its iteration space, each reading a view whose start address is
/// the original PLUS a partition offset ([`create_new_mem_view_with_mod`]). Where the start address
/// is a constant that offset is another constant and the clone is free. Where it is a toggle or a
/// conditional it is a VALUE, computed inside the unit, and moving it means rewriting the chain that
/// computes it — which is what the leading comment declines: `AddressPinningAndToggle` has already
/// arranged for a toggled address to be used by exactly one memory operation and one yield, and a
/// second user would break that.
///
/// # ⛔ EVERY VIEW, NOT THE ONE BEING SPLIT
///
/// `all_mem_views` is the whole load/store chain — a composite transfer names two, source and
/// destination — and ONE non-constant start makes the whole chain ineligible. Checking only the view
/// about to be cloned would admit exactly the case the comment rules out.
///
/// # ⚠️ EMPTY IS ELIGIBLE
///
/// A `for` over nothing falls through to `return true`. Kept, because `initialize` collects the views
/// before this runs and a chain with none has no view to fail the test.
#[must_use]
pub fn is_eligible_for_splitting(all_mem_views: &[Val], scope: &[DfirOp]) -> SplittingEligibility {
    for mem_view in all_mem_views {
        match SplitCandidateView::resolve(*mem_view, scope) {
            SplitCandidateView::ConstantStart(_) => {}
            SplitCandidateView::NotAMemoryView => {
                return SplittingEligibility::ViewIsNotAMemoryView(*mem_view);
            }
            SplitCandidateView::NonConstantStart => {
                return SplittingEligibility::ViewStartAddressIsNotConstant(*mem_view);
            }
        }
    }
    SplittingEligibility::Eligible
}

/// ONE LOOP ITERATOR OF A MEMORY OPERATION'S SUBSCRIPTS — `MutableAddrSplitting.cpp:104-125`.
///
/// ```cpp
/// /// @brief This data structure contains information about a loop iterator used
/// /// in memory operation subscripts. May include implicit loop iterators.
/// struct MASData {
///   // If the loop is an implicit loop, the iter_arg_ value will be null.
///   Value iter_arg_ = nullptr;
///   int dim_;
///   // The coefficient taking into account the layout map.
///   int64_t composed_coeff_;
///   int64_t num_iters_;
///   int64_t weight_;
/// };
/// ```
///
/// ⭐ THE TWO CONSTRUCTORS ARE ONE TYPE WITH AN `Option`. The four-argument one leaves `iter_arg_`
/// null for an IMPLICIT loop — a time dimension the transfer's `time_set` describes and no
/// `affine.for` binds — so the field is genuinely absent rather than zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasData {
    /// `iter_arg_` — the loop's induction variable, or `None` for an implicit loop.
    pub iter_arg: Option<Val>,
    /// `dim_` — which subscript dimension it is, in the order `getIndices()` lists them.
    pub dim: u32,
    /// `composed_coeff_` — *"The coefficient taking into account the layout map."*
    pub composed_coeff: i64,
    /// `num_iters_` — the loop's trip count (entry 184, `getLoopTripCount`).
    pub num_iters: i64,
    /// `weight_` — how much mutable address this iterator is responsible for.
    ///
    /// ⭐ `num_iters < 2 ? 0 : (num_iters - 2) * composed_coeff` (entry 250, `initMASData`,
    /// `:736`), under the comment *"The iter_arg will be the number of iterations - 1 at maximum and
    /// the last iteration of every loop can overflow the mutable as no data transfer will occur
    /// after it. So really, the number of iterations - 2 is the last utilized mutable address."*
    /// So the weight is NOT the span of the loop: it is the span of the addresses the loop actually
    /// transfers from, which is two iterations short of it.
    pub weight: i64,
}

/// Replaces: e114_sortDataBasedOnWeight
///
/// **114/384** `MutableAddrSplittingPass::sortDataBasedOnWeight` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855` (4L).
///
/// ```cpp
/// void MutableAddrSplittingPass::sortDataBasedOnWeight(
///     SmallVectorImpl<MASData> &mas_data) const {
///   llvm::sort(mas_data, [](MASData &a, MASData &b) -> bool {
///     return a.weight_ > b.weight_;
///   });
/// }
/// ```
///
/// # ⭐ HEAVIEST FIRST, BECAUSE THAT IS THE ORDER THE SPLIT IS SEARCHED IN
///
/// `calculatePartitionSizes` walks the sorted list and, per dimension, asks whether splitting THIS
/// one brings the mutable address back in range; if it does not, it splits the dimension fully and
/// moves to the next (`:862-905`, under *"Note: This is not optimal at all"*). Descending weight is
/// what makes that greedy walk reach a decision in as few splits as possible — a different order
/// would still produce a correct partitioning, but a different, larger one, and the partition count
/// is charged against `MaxNumConditionals` (`:957-962`).
///
/// # ⛔ STABLE, WHERE `llvm::sort` IS NOT — AND THAT IS A DELIBERATE NARROWING
///
/// `llvm::sort` is `std::sort` (plus a shuffle under `EXPENSIVE_CHECKS`): the relative order of two
/// dimensions of EQUAL weight is unspecified, so every tie order is a conformant answer and the
/// reference itself does not promise one. This crate's whole emission is byte-reproducible by
/// construction ([`crate::islands::dataflow_ir::print`]), so it takes the one tie order that keeps it
/// that way — `slice::sort_by`, which is stable, leaving equal weights in the order `initMASData`
/// filled them, i.e. by subscript dimension.
///
/// # ⚠️ `&mut [MasData]`, NOT `&mut Vec<MasData>`
///
/// A sort permutes; it does not add or remove. `SmallVectorImpl<MASData>&` is the reference's only
/// way to say "some vector", and the slice says what this function actually needs.
pub fn sort_data_based_on_weight(mas_data: &mut [MasData]) {
    // `return a.weight_ > b.weight_` — descending, so `b` is the left operand of the comparison.
    mas_data.sort_by(|a, b| b.weight.cmp(&a.weight));
}

/// WHAT `createNewMemViewWithMod` LEAVES BEHIND — a fresh start address and the view that reads it.
///
/// ⛔ TWO OPS, AND THEY DO NOT GO IN THE SAME PLACE. The reference builds the constant with
/// `OpBuilder const_builder(unit)` — at the `dataflow.program_unit`, i.e. OUTSIDE the unit's body —
/// and the cloned view with the caller's own builder, wherever the original view sat. That placement
/// is the caller's (it owns the two op lists); the fields say which is which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewMemViewWithMod {
    /// `arith.constant <start + modifier> : index`, for the preamble.
    pub start_address: DfirOp,
    /// The cloned `dataflow.get_logical_memory_view`, reading that constant.
    pub mem_view: DfirOp,
    /// The value the cloned view binds — what the new memory operation is built against.
    pub result: Val,
}

/// Replaces: e115_createNewMemViewWithMod
///
/// **115/384** `MutableAddrSplittingPass::createNewMemViewWithMod` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966` (12L).
///
/// ```cpp
/// dataflow::GetLogicalMemoryViewOp
/// MutableAddrSplittingPass::createNewMemViewWithMod(
///     ExpressionEvaluator &evaluator, OpBuilder &builder,
///     dataflow::ProgramUnitOp unit, Operation *op,
///     dataflow::GetLogicalMemoryViewOp &mem_view_op, int64_t modifier) const {
///   auto start_addr_op = mem_view_op.getStartAddress().getDefiningOp();
///   DT_CHECK(start_addr_op);
///
///   auto new_mem_view_op =
///       cast<dataflow::GetLogicalMemoryViewOp>(builder.clone(*mem_view_op));
///   OpBuilder const_builder(unit);
///   OpBuilder query_map_builder(dcc::uniform::utils::getLocalOrGlobalRegion(op));
///   dcc::agen::utils::updateMemViewStartAddress(
///       evaluator, const_builder, query_map_builder, new_mem_view_op, modifier);
///
///   return new_mem_view_op;
/// }
/// ```
///
/// # ⭐⭐ THE THREE-ARMED HELPER COLLAPSES TO ONE ARM, BY THE TYPE
///
/// `updateMemViewStartAddress` (`dcc/src/Dialect/Agen/Utils.cpp:337-420`) branches on what binds the
/// start address:
///
/// | binder | what it does |
/// |---|---|
/// | `arith.constant` or `uniform.query_map` | evaluate, add `modifier`, rebuild the start value and assign it |
/// | `arith.subi` (a toggle) | operand 0 gets `2 * modifier`, the `iter_args` chain's init gets `modifier` |
/// | `scf.if` / `affine.if` | walk the conditional tree and update each yielded constant |
/// | anything else | `llvm_unreachable` |
///
/// ⛔ AND ONLY THE FIRST IS REACHABLE FROM HERE. Every view this pass clones has passed
/// [`is_eligible_for_splitting`], which returns false for a start address that is not an
/// `arith.constant` — so the toggle and conditional arms belong to the helper's OTHER callers, and
/// taking a [`ConstStartMemView`] rather than a `Val` is what makes that argument checkable instead
/// of a comment. `updateMemViewStartAddress` is not a bridge-2 unit and is not presented as one here.
///
/// ⭐ SO THE WHOLE OF ARM ONE IS: `evaluateValue(start)` reads the constant,
/// `evaluateAddWithConst(ev, modifier)` adds the offset, `buildOffsetValue` materialises the sum with
/// `createArithConstant` — `arith::ConstantOp` of `builder.getIndexType()` — and
/// `getStartAddressMutable().assign(new_start_addr)` points the cloned view at it. Two ops, and the
/// arithmetic is `start + modifier`.
///
/// ⚠️ `DT_CHECK(start_addr_op)` and the `cast<>` of the clone are both discharged by the types: a
/// [`ConstStartMemView`] exists only where an `arith.constant` was found bound to the start address,
/// and cloning a view in this island cannot produce anything but a view.
///
/// ⚠️ `evaluator`, `unit`, `op` and the two extra builders are MECHANISM — an insertion point apiece
/// and a memoised evaluation of an SSA value this crate reads as a literal. See
/// [`NewMemViewWithMod`] for where the two ops belong.
#[must_use]
pub fn create_new_mem_view_with_mod(
    vals: &mut Values,
    view: &ConstStartMemView<'_>,
    modifier: i64,
) -> NewMemViewWithMod {
    // `createArithConstant(const_builder, .., getIndexType(), start + modifier)`.
    let start = vals.mint();
    let result = vals.mint();
    NewMemViewWithMod {
        start_address: DfirOp::Arith(arith::Op::Constant {
            result: start,
            value: view.start + modifier,
        }),
        // `builder.clone(*mem_view_op)`, with the start address assigned. Everything else — the
        // memory it views, its layout map and its type — is the original's.
        mem_view: DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result,
            from: view.from,
            start,
            layout: view.layout.clone(),
            ty: view.ty.clone(),
        }),
        result,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::agen_access_details::{IndicesCoeffDict, TimeOffsets};
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::{Index, agen};
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::{ElemType, ScalarTy, Vector};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::{Core, Corelet, Residency, Row};

    /// 🎯 111/384 — THE MUTABLE RANGE IS THE EAR'S 21 BITS OF STICKS, IN BITS.
    ///
    /// `pow(2, 21) * 128 * 8` = 2^31 (`MutableAddrSplitting.cpp:676-679` over `sysdef.cpp:313`,
    /// `:206`). ⭐ AND IT IS THE SAME ON BOTH ARCHES, because the `EAR` row is outside the
    /// `coreArch <= RCUDD1A_ISA` branch — the one L3 register that is.
    #[test]
    fn the_mutable_range_is_two_to_the_thirty_first_bits_on_every_arch() {
        assert_eq!(
            max_mutable_range::<Dd2>(L3Half::Load).bits(),
            2_147_483_648,
            "2^21 sticks x 128 bytes x 8"
        );
        assert_eq!(
            max_mutable_range::<Sen1p5>(L3Half::Load).bits(),
            max_mutable_range::<Dd2>(L3Half::Load).bits(),
            "the EAR is 21 bits on both arches"
        );
    }

    /// 🎯 112/384 — AND THE IMMUTABLE RANGE IS THE EBR'S, WHICH IS NOT.
    ///
    /// ⛔⛔ 2^40 ON DD2 AND 2^42 FROM SEN1P5. The `EBR` row sits INSIDE the arch branch — 30 bits at
    /// `sysdef.cpp:320-321`, 32 at `:326-327` — so the immutable space a program may occupy is four
    /// times larger on SEN1P5. Reading one arch's number on the other is how a splitting decision
    /// comes out wrong while every line of the pass looks right.
    #[test]
    fn the_immutable_range_is_four_times_larger_from_sen1p5() {
        assert_eq!(
            max_immutable_range::<Dd2>(L3Half::Load).bits(),
            1_099_511_627_776,
            "2^30 sticks x 128 bytes x 8 = 2^40"
        );
        assert_eq!(
            max_immutable_range::<Sen1p5>(L3Half::Load).bits(),
            4_398_046_511_104,
            "2^32 sticks x 128 bytes x 8 = 2^42"
        );
        assert_eq!(
            max_immutable_range::<Sen1p5>(L3Half::Load).bits()
                / max_immutable_range::<Dd2>(L3Half::Load).bits(),
            4
        );
    }

    /// 🎯 112/384 — AND THE IMMUTABLE RANGE DOES NOT FIT IN 32 BITS, WHILE THE MUTABLE ONE JUST DOES.
    ///
    /// ⛔⛔ THE REASON THE RANGE IS A `u64` AND NOT [`crate::formats::Bits`], WHICH IS A `u32`. The
    /// mutable range is 2^31 — half of `u32::MAX`, so a 32-bit range type would pass every test
    /// written against the `EAR` and then truncate the `EBR`'s 2^40 to zero on the SAME arch. A
    /// maximum of zero makes every address an overflow, and the only test that would have caught it
    /// is the one that asks the immutable question.
    #[test]
    fn the_immutable_range_does_not_fit_in_a_thirty_two_bit_width() {
        assert!(
            max_mutable_range::<Dd2>(L3Half::Load).bits() < u64::from(u32::MAX),
            "2^31 fits, with one bit to spare"
        );
        assert!(max_immutable_range::<Dd2>(L3Half::Load).bits() > u64::from(u32::MAX));
        assert!(max_immutable_range::<Sen1p5>(L3Half::Load).bits() > u64::from(u32::MAX));
    }

    /// 🎯 111/384 + 112/384 — THE HALF NEVER CHANGES THE ANSWER.
    ///
    /// ⭐ THE FINDING BEHIND [`L3Half`]'s NOTE. `regInfoPerUnit` declares the two halves in separate
    /// blocks (`sysdef.cpp:313` vs `:336`, `:320` vs `:343`, `:326` vs `:349`) with identical `EAR`
    /// and `EBR` rows, so the component argument's only function in these two queries is the
    /// `DT_CHECK` — which is why it is a type here and not a value to test against.
    #[test]
    fn the_two_l3_halves_declare_the_same_registers() {
        for half in [L3Half::Load, L3Half::Store] {
            assert_eq!(
                max_mutable_range::<Sen1p5>(half),
                max_mutable_range::<Sen1p5>(L3Half::Load),
                "{half:?}"
            );
            assert_eq!(
                max_immutable_range::<Sen1p5>(half),
                max_immutable_range::<Sen1p5>(L3Half::Load),
                "{half:?}"
            );
        }
    }

    /// 🎯 111/384 — AND ONLY AN L3 HALF CAN ASK.
    ///
    /// ⛔ THIS IS WHERE `DT_CHECK(is_any_of(comp, L3LU, L3SU))` WENT (`:674`, `:685`). The reference
    /// aborts the compiler for a `PE` or an `LXLU`; here such a unit cannot produce the argument, so
    /// there is no abort left to reach.
    #[test]
    fn only_the_l3_halves_have_an_external_address_register() {
        assert_eq!(L3Half::of(DfirUnit::L3lu), Some(L3Half::Load));
        assert_eq!(L3Half::of(DfirUnit::L3su), Some(L3Half::Store));
        for not_l3 in [
            DfirUnit::Pe,
            DfirUnit::Sfp,
            DfirUnit::PtRow(Row::checked(0).expect("every PT has a row 0")),
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::Lx,
            DfirUnit::Hbm,
            DfirUnit::L0lu,
            DfirUnit::L0su,
            DfirUnit::L0,
            DfirUnit::Constant,
            DfirUnit::SfpState,
            DfirUnit::PeState,
            DfirUnit::SfpRing,
            DfirUnit::LxVirtualIbr,
            DfirUnit::CrossPtnLink,
        ] {
            assert_eq!(L3Half::of(not_l3), None, "{not_l3:?} has no EAR");
        }
    }

    /// 🎯 111/384 — AND THE RANGE BECOMES AN ELEMENT COUNT BY THE ELEMENT'S WIDTH.
    ///
    /// `getMaxMutableRange(comp) / elem_size_in_bits` (`:804`, `:876`, `:927`) — so the SAME machine
    /// admits half as many fp16 addresses as int8 ones, which is the whole reason the range is kept
    /// in bits.
    #[test]
    fn the_range_divides_by_the_element_width() {
        let mutable = max_mutable_range::<Dd2>(L3Half::Load);
        assert_eq!(mutable.elements(DataType::Senint8), Elements(268_435_456));
        assert_eq!(
            mutable.elements(DataType::Sen169Fp16),
            Elements(134_217_728)
        );
        assert_eq!(
            mutable.elements(DataType::Senint4),
            Elements(536_870_912),
            "a sub-byte format gets more of them, not fewer"
        );
    }

    /// 🎯 111/384 — AN OVERRIDE IS TAKEN AS GIVEN, TABLE AND ARCH BYPASSED.
    ///
    /// ⛔ `MaxMutableSize < 0 ? computed : MaxMutableSize` (`:676-680`) — the flag is already in bits
    /// and nothing scales it. This exercises the branch the constant selects, since
    /// [`super::MAX_MUTABLE_SIZE`] is `None` in a compiled pipeline.
    #[test]
    fn an_override_replaces_the_computed_range_entirely() {
        let given = AddrRange::of_bits(4096);
        assert_eq!(given.bits(), 4096);
        assert_ne!(given, max_mutable_range::<Dd2>(L3Half::Load));
        assert_eq!(given.elements(DataType::Senint8), Elements(512));
    }
    /// One `dataflow.get_logical_memory_view` over a constant start, and the ops that bind it.
    fn view_with_start(vals: &mut Values, start: DfirOp) -> (Vec<DfirOp>, Val) {
        let result = vals.mint();
        let start_val = match &start {
            DfirOp::Arith(arith::Op::Constant { result, .. }) => *result,
            _ => unreachable!("the fixtures bind the start with a constant"),
        };
        let view = DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result,
            from: Val(0),
            start: start_val,
            layout: AffineMap::linear(&[128, 1]),
            ty: MemRef {
                shape: vec![4, 128],
                elem: ElemType::F16,
            },
        });
        (vec![start, view], result)
    }

    /// 🎯 113/384 — A CONSTANT START IS ELIGIBLE, AND THE GATE HANDS THE CLONE ITS INPUT.
    #[test]
    fn a_constant_start_address_is_eligible() {
        let mut vals = Values::default();
        let _memory = vals.mint();
        let start = vals.mint();
        let (scope, view) = view_with_start(
            &mut vals,
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 2048,
            }),
        );

        assert_eq!(
            is_eligible_for_splitting(&[view], &scope),
            SplittingEligibility::Eligible
        );
        assert!(is_eligible_for_splitting(&[view], &scope).eligible());

        let SplitCandidateView::ConstantStart(candidate) =
            SplitCandidateView::resolve(view, &scope)
        else {
            unreachable!("a constant start resolves to ConstantStart");
        };
        assert_eq!(candidate.start, 2048);
        assert_eq!(candidate.from, Val(0));
    }

    /// 🎯 113/384 — A START ADDRESS BOUND BY A TOGGLE IS NOT.
    ///
    /// `AddressPinningAndToggle` leaves `%addr = arith.subi %arg, %step`, and the leading comment at
    /// `MutableAddrSplitting.cpp:832-839` is about exactly that shape: the toggled address may be used
    /// by one memory operation and one yield, so splitting — which would add a second user — declines.
    #[test]
    fn a_toggled_start_address_is_not_eligible() {
        let mut vals = Values::default();
        let _memory = vals.mint();
        let iter_arg = vals.mint();
        let step = vals.mint();
        let start = vals.mint();
        let result = vals.mint();
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: step,
                value: 16,
            }),
            DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                result: start,
                lhs: iter_arg,
                rhs: step,
                ty: ScalarTy::Index,
            })),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from: Val(0),
                start,
                layout: AffineMap::linear(&[1]),
                ty: MemRef {
                    shape: vec![512],
                    elem: ElemType::F16,
                },
            }),
        ];

        assert_eq!(
            SplitCandidateView::resolve(result, &scope),
            SplitCandidateView::NonConstantStart
        );
        assert_eq!(
            is_eligible_for_splitting(&[result], &scope),
            SplittingEligibility::ViewStartAddressIsNotConstant(result)
        );
    }

    /// 🎯 113/384 — A VALUE NO `get_logical_memory_view` BINDS IS THE FIRST `dyn_cast`'s `false`, AND
    /// ONE INELIGIBLE VIEW CONDEMNS THE WHOLE CHAIN.
    #[test]
    fn one_non_view_condemns_every_view() {
        let mut vals = Values::default();
        let _memory = vals.mint();
        let start = vals.mint();
        let (mut scope, good) = view_with_start(
            &mut vals,
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 0,
            }),
        );
        // A region argument: nothing in scope binds it.
        let unbound = Val(4096);
        assert_eq!(
            SplitCandidateView::resolve(unbound, &scope),
            SplitCandidateView::NotAMemoryView
        );

        assert_eq!(
            is_eligible_for_splitting(&[good, unbound], &scope),
            SplittingEligibility::ViewIsNotAMemoryView(unbound)
        );

        // ⭐ AND THE OFFENDER IS THE FIRST IN LOOP ORDER, WHICH IS WHERE THE `for` RETURNS.
        let other_start = vals.mint();
        scope.push(DfirOp::Arith(arith::Op::Constant {
            result: other_start,
            value: 1,
        }));
        assert_eq!(
            is_eligible_for_splitting(&[unbound, good], &scope),
            SplittingEligibility::ViewIsNotAMemoryView(unbound)
        );
    }

    /// 🎯 113/384 — NO VIEWS AT ALL FALLS THROUGH TO `return true`.
    #[test]
    fn an_empty_chain_is_eligible() {
        assert_eq!(
            is_eligible_for_splitting(&[], &[]),
            SplittingEligibility::Eligible
        );
    }

    /// 🎯 114/384 — HEAVIEST FIRST, AND EQUAL WEIGHTS KEEP THEIR DIMENSION ORDER.
    #[test]
    fn the_sort_is_descending_by_weight_and_stable() {
        let entry = |dim: u32, weight: i64| MasData {
            iter_arg: None,
            dim,
            composed_coeff: 1,
            num_iters: 4,
            weight,
        };
        let mut data = vec![
            entry(0, 16),
            entry(1, 4096),
            entry(2, 16),
            entry(3, 0),
            entry(4, 4096),
        ];
        sort_data_based_on_weight(&mut data);
        assert_eq!(
            data.iter().map(|d| (d.dim, d.weight)).collect::<Vec<_>>(),
            vec![(1, 4096), (4, 4096), (0, 16), (2, 16), (3, 0)]
        );
    }

    /// 🎯 115/384 — THE CLONE IS THE ORIGINAL WITH `start + modifier`, AND EVERYTHING ELSE UNTOUCHED.
    #[test]
    fn the_clone_shifts_only_the_start_address() {
        let mut vals = Values::default();
        let memory = vals.mint();
        let layout = AffineMap::linear(&[128, 1]);
        let ty = MemRef {
            shape: vec![4, 128],
            elem: ElemType::F16,
        };
        let view = ConstStartMemView {
            from: memory,
            start: 2048,
            layout: &layout,
            ty: &ty,
        };

        let new = create_new_mem_view_with_mod(&mut vals, &view, 512);
        assert_eq!(
            new.start_address,
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 2560,
            })
        );
        assert_eq!(
            new.mem_view,
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(2),
                from: memory,
                start: Val(1),
                layout: layout.clone(),
                ty: ty.clone(),
            })
        );
        assert_eq!(new.result, Val(2));

        // ⭐ A NEGATIVE MODIFIER SHIFTS THE OTHER WAY — `evaluateAddWithConst` is signed addition, and
        // `MutableStartAddrShifting` is the caller that passes one.
        let back = create_new_mem_view_with_mod(&mut vals, &view, -1024);
        assert_eq!(
            back.start_address,
            DfirOp::Arith(arith::Op::Constant {
                result: Val(3),
                value: 1024,
            })
        );
    }

    /// One `arith.constant N : index`, and the value it binds.
    fn index_const(vals: &mut Values, value: i64) -> (DfirOp, Val) {
        let result = vals.mint();
        (DfirOp::Arith(arith::Op::Constant { result, value }), result)
    }

    /// An `scf.for` over the three bound values, with an empty body.
    fn scf_for(vals: &mut Values, lo: Val, hi: Val, step: Val) -> DfirOp {
        DfirOp::Scf(scf::Op::For {
            iv: vals.mint(),
            lo,
            hi,
            step,
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// 🎯 184/384 — AN `affine.for` WITH CONSTANT BOUNDS COUNTS `ub - lb`, AND ITS STEP IS ALWAYS 1.
    ///
    /// ⭐ THE VENDOR'S OWN OUTER LOOP. `affine.for %arg1 = 0 to 4` wraps the split transfer in
    /// `constant_start_addr_1`
    /// (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:247`), and the 4
    /// it answers here is the `num_iters_` that dimension's weight is computed from.
    #[test]
    fn an_affine_for_with_constant_bounds_counts_its_span() {
        let mut vals = Values::default();
        let loop_op = DfirOp::Affine(affine::Op::For {
            iv: vals.mint(),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });

        assert_eq!(
            get_loop_trip_count(&loop_op, &[]),
            LoopTripCount::Iterations(4)
        );
        assert_eq!(get_loop_trip_count(&loop_op, &[]).iterations(), Some(4));
    }

    /// 🎯 184/384 — A NON-ZERO LOWER BOUND IS FINE ON ITS OWN, BECAUSE THE STEP IS 1.
    ///
    /// `DT_CHECK_MSG(lb == 0 || step == 1, …)` (`:794`) is a disjunction, and
    /// [`affine::Op::For`] carries no step — so the affine arm satisfies the second disjunct for every
    /// loop this island can hold and `lb` is unconstrained. `(ub - lb) / 1`.
    #[test]
    fn an_affine_lower_bound_above_zero_is_still_counted() {
        let mut vals = Values::default();
        let loop_op = DfirOp::Affine(affine::Op::For {
            iv: vals.mint(),
            lo: affine::Bound::Const(3),
            hi: affine::Bound::Const(11),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });

        assert_eq!(
            get_loop_trip_count(&loop_op, &[]),
            LoopTripCount::Iterations(8)
        );
    }

    /// 🎯 184/384 — A SYMBOLIC AFFINE BOUND IS `-1`, WHICH ITS CALLER'S `DT_CHECK` REJECTS.
    ///
    /// `if (!affine_for.hasConstantBounds()) return -1;` (`:752`), and
    /// `DT_CHECK(num_iters >= 0)` in `initMASData` (`:730`) is what that `-1` reaches — so
    /// [`LoopTripCount::iterations`] answers [`None`] rather than a number.
    #[test]
    fn an_affine_for_with_a_value_bound_has_no_constant_count() {
        let mut vals = Values::default();
        let bound = vals.mint();
        let loop_op = DfirOp::Affine(affine::Op::For {
            iv: vals.mint(),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Val(bound),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });

        assert_eq!(
            get_loop_trip_count(&loop_op, &[]),
            LoopTripCount::AffineBoundsAreNotConstant
        );
        assert_eq!(get_loop_trip_count(&loop_op, &[]).iterations(), None);
    }

    /// 🎯 184/384 — AN `scf.for` READS ITS THREE OPERANDS THROUGH THE DEF-USE WALK.
    #[test]
    fn an_scf_for_over_constants_counts_its_span() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (hi_op, hi) = index_const(&mut vals, 12);
        let (step_op, step) = index_const(&mut vals, 4);
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, hi_op, step_op];

        // `(12 - 0) / 4` — and `lb == 0` satisfies the first disjunct, so the step of 4 is allowed.
        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::Iterations(3)
        );
    }

    /// 🎯 184/384 — ⭐⭐ THE VENDOR'S SYMBOLIC BOUND COUNTS **8**, FROM ITS `maxValue`.
    ///
    /// `constant_start_addr_1` gives its inner `scf.for` the upper bound
    /// `symbol.create_symbol {SymbolId = -1476 : i64, granularity = 8 : i64, maxValue = 8 : i64}`
    /// (`mutable_addr_splitting_one_dim.mlir:251`, the loop at `:253`) and the pass partitions that
    /// dimension as if it ran 8 times — the trip count the schedule COULD fix it to, which is what
    /// bounds the mutable address before any symbol has a value.
    ///
    /// ⛔ AND [`constant_index`] SAYS NO TO THE SAME VALUE, which is why entry 142 declines this loop
    /// while this function counts it.
    #[test]
    fn a_symbol_upper_bound_counts_its_max_value() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let hi = vals.mint();
        let sym = DfirOp::Symbol(symbol::Op::CreateSymbol {
            result: hi,
            symbol_id: -1476,
            max_value: Some(8),
        });
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op, sym];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::Iterations(8)
        );
        assert_eq!(constant_index(hi, &scope), None);
    }

    /// 🎯 184/384 — ⛔⛔ AND WITHOUT A `maxValue` THE ANSWER IS **ZERO**, NOT A REFUSAL.
    ///
    /// `else return false;` (`:777`) in an `int64_t` function. That 0 passes
    /// `DT_CHECK(num_iters >= 0)` and gives the dimension weight 0, where the two `return -1`s abort
    /// — one return type, two outcomes, which is the whole reason [`LoopTripCount`] is an enum.
    #[test]
    fn a_symbol_upper_bound_without_a_max_value_counts_zero() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let hi = vals.mint();
        let sym = DfirOp::Symbol(symbol::Op::CreateSymbol {
            result: hi,
            symbol_id: -1476,
            max_value: None,
        });
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op, sym];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::SymbolUpperBoundHasNoMaxValue(hi)
        );
        // ⛔ THE ASYMMETRY, STATED: a zero that its caller accepts, beside a `-1` that aborts.
        assert_eq!(get_loop_trip_count(&loop_op, &scope).iterations(), Some(0));
        assert_eq!(LoopTripCount::AffineBoundsAreNotConstant.iterations(), None);
    }

    /// 🎯 184/384 — ⭐⭐ THE VENDOR'S `arith.select` BOUND COUNTS THE **LARGER ARM**, NOT THE PICKED
    /// ONE.
    ///
    /// `select_ub` writes
    ///
    /// ```text
    /// %904 = arith.cmpi slt, %arg1, %c2 : index
    /// %905 = arith.select %904, %c8, %c4 : index
    /// scf.for %arg2 = %c0 to %905 step %c1 {
    /// ```
    ///
    /// (`mutable_addr_splitting_one_dim.mlir:288-291`) and the pass partitions that dimension against
    /// **8** — the same trip count `constant_start_addr_1` reaches through a `maxValue`, which is why
    /// the two cases split identically. ⛔ The condition is never read: an address bound has to hold
    /// for every iteration of the outer loop, including the ones that pick 4.
    #[test]
    fn a_select_upper_bound_counts_its_larger_arm() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let (true_op, true_value) = index_const(&mut vals, 8);
        let (false_op, false_value) = index_const(&mut vals, 4);
        let condition = vals.mint();
        let hi = vals.mint();
        let select = DfirOp::Arith(arith::Op::Select {
            result: hi,
            condition,
            true_value,
            false_value,
            ty: ScalarTy::Index,
            dbg_name: None,
        });
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op, true_op, false_op, select];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::Iterations(8)
        );
    }

    /// 🎯 184/384 — AND THE ARMS THE OTHER WAY ROUND GIVE THE SAME 8.
    ///
    /// `ub = true_val > false_val ? true_val : false_val` (`:788`) is a maximum, so the arm order does
    /// not matter — which is what makes the strict `>` on equal arms unobservable.
    #[test]
    fn a_select_upper_bound_is_order_independent() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let (true_op, true_value) = index_const(&mut vals, 4);
        let (false_op, false_value) = index_const(&mut vals, 8);
        let condition = vals.mint();
        let hi = vals.mint();
        let select = DfirOp::Arith(arith::Op::Select {
            result: hi,
            condition,
            true_value,
            false_value,
            ty: ScalarTy::Index,
            dbg_name: None,
        });
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op, true_op, false_op, select];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::Iterations(8)
        );
    }

    /// 🎯 184/384 — A SELECT WITH A NON-CONSTANT ARM IS THE `DT_CHECK_MSG` (`:783-785`).
    #[test]
    fn a_select_upper_bound_needs_both_arms_constant() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let (true_op, true_value) = index_const(&mut vals, 8);
        let false_value = vals.mint();
        let condition = vals.mint();
        let hi = vals.mint();
        let select = DfirOp::Arith(arith::Op::Select {
            result: hi,
            condition,
            true_value,
            false_value,
            ty: ScalarTy::Index,
            dbg_name: None,
        });
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op, true_op, select];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::SelectUpperBoundArmsAreNotConstant(hi)
        );
    }

    /// 🎯 184/384 — THE `arith.divsi` BOUND THE SCHEDULER WRITES IS UNSUPPORTED HERE.
    ///
    /// `llvm_unreachable("unsupported upper loop bound operation")` (`:790`). The scheduler spells a
    /// dynamic trip count `(%hi - %lo) / %step`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:226-228`), and this pass
    /// has no worst case for it at all — unlike a symbol's `maxValue` or a select's arms.
    #[test]
    fn a_divided_upper_bound_is_unsupported() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let (num_op, num) = index_const(&mut vals, 64);
        let (den_op, den) = index_const(&mut vals, 8);
        let hi = vals.mint();
        let div = DfirOp::Arith(arith::Op::DivSI(arith::IntBinary {
            result: hi,
            lhs: num,
            rhs: den,
            ty: ScalarTy::Index,
        }));
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op, num_op, den_op, div];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::UnsupportedScfUpperBound(hi)
        );
    }

    /// 🎯 184/384 — AN UPPER BOUND NOTHING BINDS IS THE `DT_CHECK(ub_op)` (`:769`), AND IT IS NOT THE
    /// `llvm_unreachable`.
    ///
    /// A region argument has no defining op, so `getDefiningOp()` is null and the reference rejects it
    /// BEFORE the three casts — which is why this is its own variant.
    #[test]
    fn an_upper_bound_bound_by_nothing_is_its_own_answer() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let hi = vals.mint();
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, step_op];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::ScfUpperBoundIsNotAnOperation(hi)
        );
    }

    /// 🎯 184/384 — A NON-CONSTANT `scf` LOWER BOUND IS THE `DT_CHECK(lb_op)` (`:765`).
    #[test]
    fn an_scf_lower_bound_must_be_a_constant() {
        let mut vals = Values::default();
        let (hi_op, hi) = index_const(&mut vals, 8);
        let (step_op, step) = index_const(&mut vals, 1);
        let lo = vals.mint();
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![hi_op, step_op];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::ScfLowerBoundIsNotConstant(lo)
        );
    }

    /// 🎯 184/384 — A NON-CONSTANT STEP IS `-1`, AND IT IS TESTED BEFORE EITHER BOUND.
    ///
    /// `if (!const_step.has_value()) return -1;` (`:759`) precedes both `getDefiningOp()` walks, so a
    /// loop with a symbolic step and a symbolic lower bound reports the STEP.
    #[test]
    fn a_non_constant_step_is_reported_before_the_bounds() {
        let mut vals = Values::default();
        let (hi_op, hi) = index_const(&mut vals, 8);
        let lo = vals.mint();
        let step = vals.mint();
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![hi_op];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::ScfStepIsNotConstant(step)
        );
    }

    /// 🎯 184/384 — ⛔ A ZERO STEP IS DECLINED RATHER THAN DIVIDED BY.
    ///
    /// `scf::ForOp::verify` does not check the step (`SCF.cpp:379-386`; the positivity verifier is
    /// `scf::ParallelOp`'s, `:2804-2808`), so this loop is verifiable DataflowIR and
    /// `(ub - lb) / 0` is undefined behaviour in the reference — not an answer to reproduce. See
    /// [`LoopTripCount::ScfStepIsNotPositive`].
    #[test]
    fn a_zero_step_is_declined_not_divided_by() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (hi_op, hi) = index_const(&mut vals, 8);
        let (step_op, step) = index_const(&mut vals, 0);
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, hi_op, step_op];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::ScfStepIsNotPositive(0)
        );
    }

    /// 🎯 184/384 — AND SO IS A NEGATIVE ONE, WHOSE QUOTIENT ITS CALLER WOULD REJECT ANYWAY.
    #[test]
    fn a_negative_step_is_declined() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (hi_op, hi) = index_const(&mut vals, 8);
        let (step_op, step) = index_const(&mut vals, -1);
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, hi_op, step_op];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::ScfStepIsNotPositive(-1)
        );
    }

    /// 🎯 184/384 — A STRIDED LOOP THAT DOES NOT START AT ZERO IS THE `DT_CHECK_MSG` (`:795-796`).
    ///
    /// ⚠️ IT TAKES BOTH. `lb = 4, step = 8` fails; either alone is fine — see
    /// [`an_scf_for_over_constants_counts_its_span`] for `lb = 0` with a step of 4, and
    /// [`an_affine_lower_bound_above_zero_is_still_counted`] for `lb = 3` with a step of 1.
    #[test]
    fn a_strided_loop_from_a_non_zero_lower_bound_is_declined() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 4);
        let (hi_op, hi) = index_const(&mut vals, 68);
        let (step_op, step) = index_const(&mut vals, 8);
        let loop_op = scf_for(&mut vals, lo, hi, step);
        let scope = vec![lo_op, hi_op, step_op];

        assert_eq!(
            get_loop_trip_count(&loop_op, &scope),
            LoopTripCount::StridedLoopWithNonZeroLowerBound {
                lo: LoopBound(4),
                step: LoopStep::checked(8).expect("8 is a positive step"),
            }
        );
    }

    /// 🎯 184/384 — AN OPERATION THAT IS NOT A LOOP IS THE OUTER `llvm_unreachable` (`:793`).
    ///
    /// The `// TODO: Remove the DT_CHECK safely and make this a broader utility` above the function
    /// (`:742`) is about exactly this arm.
    #[test]
    fn an_operation_that_is_not_a_loop_has_no_trip_count() {
        let mut vals = Values::default();
        let (const_op, _) = index_const(&mut vals, 8);

        assert_eq!(get_loop_trip_count(&const_op, &[]), LoopTripCount::NotALoop);
        assert_eq!(get_loop_trip_count(&const_op, &[]).iterations(), None);
    }

    /// 🎯 184/384 — ⛔ A REVERSED RANGE COUNTS NEGATIVE, AND THAT IS KEPT.
    ///
    /// `(0 - 8) / 1` is `-8`. Unlike [`Iterations`](super::super::tf_utils::Iterations), which
    /// saturates at zero, this count is the reference's own signed quotient — because
    /// `DT_CHECK(num_iters >= 0)` in `initMASData` (`:730`) is the thing that rejects it, and
    /// saturating here would turn that abort into a silently empty dimension.
    #[test]
    fn a_reversed_affine_range_keeps_its_negative_count() {
        let mut vals = Values::default();
        let loop_op = DfirOp::Affine(affine::Op::For {
            iv: vals.mint(),
            lo: affine::Bound::Const(8),
            hi: affine::Bound::Const(0),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });

        assert_eq!(
            get_loop_trip_count(&loop_op, &[]),
            LoopTripCount::Iterations(-8)
        );
        assert_eq!(get_loop_trip_count(&loop_op, &[]).iterations(), Some(-8));
    }

    /// One iterator, enough to make `mas_data` non-empty.
    fn one_iterator(vals: &mut Values) -> MasData {
        MasData {
            iter_arg: Some(vals.mint()),
            dim: 0,
            composed_coeff: 2048,
            num_iters: 8,
            weight: 12_288,
        }
    }

    /// 🎯 185/384 — AN ADDRESS INSIDE THE `EAR`'S RANGE IS NOT AN OVERFLOW, AND NOTHING ELSE IS ASKED.
    ///
    /// `if (max_mutable > (getMaxMutableRange(comp) / elem_size_in_bits))` (`:804`) — below the limit
    /// the function returns `false` before it ever looks at the correction flag or at `mas_data`, which
    /// is why an empty `mas_data` is fine here.
    #[test]
    fn an_address_within_the_ear_range_does_not_overflow() {
        // `2^31` bits of `EAR` at 16 bits an element — see
        // [`the_mutable_range_is_two_to_the_thirty_first_bits_on_every_arch`].
        let limit = Elements(134_217_728);
        assert_eq!(
            max_mutable_range::<Dd2>(L3Half::Load).elements(DataType::Sen169Fp16),
            limit
        );

        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &[],
                L3Half::Load,
                MutableAddr(134_217_728),
                DataType::Sen169Fp16,
                EarOverflowCorrection::Forbidden,
            ),
            MutableAddrOverflow::InRange
        );
        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &[],
                L3Half::Load,
                MutableAddr(134_217_728),
                DataType::Sen169Fp16,
                EarOverflowCorrection::Forbidden,
            )
            .overflowed(),
            Some(false)
        );
    }

    /// 🎯 185/384 — ⛔ IT IS A STRICT `>`, SO ONE ELEMENT PAST THE LIMIT IS THE OVERFLOW.
    #[test]
    fn one_element_past_the_range_overflows() {
        let mut vals = Values::default();
        let mas_data = [one_iterator(&mut vals)];

        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &mas_data,
                L3Half::Load,
                MutableAddr(134_217_729),
                DataType::Sen169Fp16,
                EarOverflowCorrection::Allowed,
            ),
            MutableAddrOverflow::Overflow
        );
    }

    /// 🎯 185/384 — ⛔⛔ AND WITHOUT THE OVERRIDE THAT SAME PROGRAM STOPS THE COMPILER.
    ///
    /// `if (!dcc_ext_ctx_.dsc_global_->dcc_correct_ear_overflow) DT_CHECK_MSG(false, "EAR overflow
    /// detected");` (`:807-808`). [`CORRECT_EAR_OVERFLOW`] is [`EarOverflowCorrection::Forbidden`], so
    /// this — not [`MutableAddrOverflow::Overflow`] — is what a shipped `dcc` answers, and the four
    /// vendor tests only reach the split because every one of their `RUN` lines says
    /// `DT_OPT=correctearoverflow=1`.
    #[test]
    fn the_shipped_default_forbids_correcting_an_overflow_at_all() {
        let mut vals = Values::default();
        let mas_data = [one_iterator(&mut vals)];

        assert_eq!(CORRECT_EAR_OVERFLOW, EarOverflowCorrection::Forbidden);
        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &mas_data,
                L3Half::Load,
                MutableAddr(134_217_729),
                DataType::Sen169Fp16,
                CORRECT_EAR_OVERFLOW,
            ),
            MutableAddrOverflow::OverflowCorrectionForbidden
        );
    }

    /// 🎯 185/384 — AN OVERFLOW WITH NO ITERATORS HAS NOTHING TO PARTITION.
    ///
    /// `DT_CHECK_MSG(!mas_data.empty(), …)` (`:810-813`) — `initMASData` returns early on
    /// `indices.size() == 0` (`:713`), so a transfer whose subscripts name no loop reaches here with an
    /// empty list and its overflow is a constant offset no cut can reduce.
    #[test]
    fn an_overflow_with_no_iterators_cannot_be_split() {
        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &[],
                L3Half::Load,
                MutableAddr(134_217_729),
                DataType::Sen169Fp16,
                EarOverflowCorrection::Allowed,
            ),
            MutableAddrOverflow::NoLoopsToSplit
        );
    }

    /// 🎯 185/384 — ⛔ THE CORRECTION FLAG IS TESTED FIRST, WHICH IS OBSERVABLE.
    ///
    /// The same program — an overflow with no iterators — answers
    /// [`MutableAddrOverflow::OverflowCorrectionForbidden`] under the shipped default and
    /// [`MutableAddrOverflow::NoLoopsToSplit`] with the override. Reversing the two checks would report
    /// the more specific fact and be wrong.
    #[test]
    fn the_correction_flag_is_read_before_the_iterator_list() {
        let forbidden = has_mutable_addr_overflow::<Dd2>(
            &[],
            L3Half::Load,
            MutableAddr(134_217_729),
            DataType::Sen169Fp16,
            EarOverflowCorrection::Forbidden,
        );
        let allowed = has_mutable_addr_overflow::<Dd2>(
            &[],
            L3Half::Load,
            MutableAddr(134_217_729),
            DataType::Sen169Fp16,
            EarOverflowCorrection::Allowed,
        );

        assert_eq!(forbidden, MutableAddrOverflow::OverflowCorrectionForbidden);
        assert_eq!(allowed, MutableAddrOverflow::NoLoopsToSplit);
        assert_eq!(forbidden.overflowed(), None);
        assert_eq!(allowed.overflowed(), None);
    }

    /// 🎯 185/384 — A NARROWER ELEMENT MAKES THE SAME REGISTER HOLD MORE, SO THE SAME SPAN FITS.
    ///
    /// The limit is `range / elem_size_in_bits`, so `senint4` doubles it against `f16` and a span that
    /// overflowed one is inside the other — the element format is half the gate.
    #[test]
    fn the_element_format_moves_the_limit() {
        let mut vals = Values::default();
        let mas_data = [one_iterator(&mut vals)];
        let span = MutableAddr(134_217_729);

        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &mas_data,
                L3Half::Load,
                span,
                DataType::Sen169Fp16,
                EarOverflowCorrection::Allowed,
            ),
            MutableAddrOverflow::Overflow
        );
        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &mas_data,
                L3Half::Load,
                span,
                DataType::Senint4,
                EarOverflowCorrection::Allowed,
            ),
            MutableAddrOverflow::InRange
        );
    }

    /// 🎯 185/384 — ⭐⭐ THE VENDOR'S OWN OVERFLOW, WITH THEIR OWN OVERRIDE.
    ///
    /// All four answer keys run `--dcc-mutable-addr-splitting-max-mutable-size=<N>`, which replaces the
    /// register-derived range wholesale ([`MAX_MUTABLE_SIZE`]). `mutable_addr_splitting_one_dim.mlir`
    /// passes `91200` bits — 5700 `f16` elements — and `constant_start_addr_1`'s address span is
    /// `12672`, which is what makes it overflow by `6972` and split.
    ///
    /// ⚠️ THE OVERRIDE IS A CONST HERE, SO THIS EXERCISES THE ARITHMETIC, NOT THE FLAG. The comparison
    /// [`has_mutable_addr_overflow`] performs is [`MutableAddr::exceeds`], and this is that call over
    /// the vendor's two numbers; see [`an_override_replaces_the_computed_range_entirely`].
    #[test]
    fn the_vendors_override_and_span_overflow_by_their_own_numbers() {
        let limit = AddrRange::of_bits(91_200).elements(DataType::Sen169Fp16);
        assert_eq!(limit, Elements(5700));

        // `12672 = 384 (arg1) + 12288 (arg2)` — the two weights of `constant_start_addr_1`.
        let span = MutableAddr(12_672);
        assert!(span.exceeds(limit));
        assert_eq!(span.beyond(limit), 6972);

        // And the same numbers one element lower are exactly in range: `>` is strict.
        assert!(!MutableAddr(5700).exceeds(limit));
        assert_eq!(MutableAddr(5700).beyond(limit), 0);
    }

    /// 🎯 185/384 — ⛔ A NEGATIVE SPAN EXCEEDS NOTHING, AND ITS DIFFERENCE STAYS NEGATIVE.
    ///
    /// `max_mutable` is a sum of weights over layout coefficients and starts at the subscript map's
    /// constant offset, so the negative half of [`MutableAddr`] is reachable — see the type. The
    /// signed comparison against an unsigned limit is written out in [`MutableAddr::exceeds`] rather
    /// than left to a cast.
    #[test]
    fn a_negative_span_is_inside_every_limit() {
        let limit = Elements(5700);
        assert!(!MutableAddr(-1).exceeds(limit));
        assert!(!MutableAddr(0).exceeds(limit));
        assert_eq!(MutableAddr(-1).beyond(limit), -5701);

        assert_eq!(
            has_mutable_addr_overflow::<Dd2>(
                &[],
                L3Half::Load,
                MutableAddr(-1),
                DataType::Sen169Fp16,
                EarOverflowCorrection::Forbidden,
            ),
            MutableAddrOverflow::InRange
        );
    }

    /// 🎯 185/384 — `max_mutable += weight` ACCUMULATES, WHICH IS HOW BOTH ITS PRODUCERS FILL IT.
    ///
    /// `initMASData` (`:738`) and [`synthesize_time_info`] (`:1278`) each add one weight per
    /// dimension to the same running total.
    #[test]
    fn the_span_accumulates_one_weight_at_a_time() {
        let mut span = MutableAddr(0);
        // `constant_start_addr_1`: `arg1` at 384 and `arg2` at 12288.
        span.add_weight(384);
        span.add_weight(12_288);
        assert_eq!(span, MutableAddr(12_672));
        assert_eq!(span.get(), 12_672);
    }

    /// THE LAYOUT AND ELEMENT TYPE ALL FIVE `MutableAddrSplitting` ANSWER KEYS SHARE.
    ///
    /// `affine_map<(d0, d1, d2) -> (d2 * 256 + d1 * 64 + d0)>` over `memref<?x64x4xf16>` — identical in
    /// `mutable_addr_splitting_one_dim.mlir:5`, `_one_dim_sen1p5.mlir:9`, `_multi_dim.mlir:4` and
    /// `_time_dims.mlir:5`. ⭐ WHICH IS WHERE EVERY `composed_coeff_` BELOW COMES FROM: a subscript of
    /// `%arg * 8` on `d2` is a coefficient of `8 * 256 = 2048`, and one of `%arg * 16` on `d1` is
    /// `16 * 64 = 1024`.
    ///
    /// ⚠️ THE LEADING `?` IS WRITTEN AS `1`. [`MemRef::shape`] is a `Vec<u64>` with no dynamic extent,
    /// and [`calculate_partition_sizes`] reads neither the shape nor the map — it takes a whole
    /// [`ConstStartMemView`] because the reference takes the op, and the start address is the only
    /// field it touches.
    fn answer_key_memref() -> (AffineMap, MemRef) {
        (
            AffineMap::linear(&[1, 64, 256]),
            MemRef {
                shape: vec![1, 64, 4],
                elem: ElemType::F16,
            },
        )
    }

    /// ONE `mas_data` ENTRY, spelled out.
    ///
    /// ⚠️ `iter_arg` IS `None` AND NOTHING NOTICES. [`calculate_partition_sizes`] reads `weight_`,
    /// `composed_coeff_`, `num_iters_` and — only to report a division it will not do — `dim_`. The
    /// induction variable is what [`construct_conditionals`] compares against the sizes this returns.
    fn iterator(dim: u32, composed_coeff: i64, num_iters: i64, weight: i64) -> MasData {
        MasData {
            iter_arg: None,
            dim,
            composed_coeff,
            num_iters,
            weight,
        }
    }

    /// A SPAN THAT OVERFLOWS THE COMPILED-IN `EAR` RANGE BY EXACTLY `overflow` ELEMENTS.
    ///
    /// # ⛔⛔ WHY THE ANSWER KEYS' OWN `max_mutable` NUMBERS ARE NOT PASSED DIRECTLY
    ///
    /// All five keys run `--dcc-mutable-addr-splitting-max-mutable-size=<N>` and so compare against a
    /// range of `N / 16` elements rather than the `EAR`'s 134217728 — but that flag is a `const`
    /// [`MAX_MUTABLE_SIZE`] here, deliberately (see the type). Everything
    /// [`calculate_partition_sizes`] computes depends on `max_mutable - limit` and on nothing else
    /// about either number, so each key is reproduced by re-expressing its pair against the range this
    /// crate actually compiles with. The overflow — the one quantity the arithmetic reads — is
    /// unchanged, and each test states the vendor's original pair beside it.
    fn span_overflowing_by<A: Arch>(overflow: i64, elem: DataType) -> MutableAddr {
        let limit = max_mutable_range::<A>(L3Half::Load).elements(elem);
        let span = MutableAddr(0i64.saturating_add_unsigned(limit.0) + overflow);
        assert_eq!(
            span.beyond(limit),
            overflow,
            "the overflow is the whole input"
        );
        span
    }

    /// A CONSTANT START ADDRESS THAT LEAVES `space` ELEMENTS OF `EBR` ROOM ABOVE IT.
    ///
    /// `immutable_space = (getMaxImmutableRange(comp) / elem_size_in_bits) - max_immutable` (`:926`),
    /// solved for the address — the other side of [`max_immutable_range`], used by the three tests
    /// about running out of immutable room.
    fn start_leaving_immutable_space<A: Arch>(space: i64, elem: DataType) -> i64 {
        let room = max_immutable_range::<A>(L3Half::Load).elements(elem);
        let start = 0i64.saturating_add_unsigned(room.0) - space;
        assert_eq!(elements_less_address(room, start), space);
        start
    }

    /// 🎯 186/384 — ⭐⭐ IBM'S `one_dim` ANSWER KEY: EIGHT ITERATIONS SPLIT INTO PARTITIONS OF **4**.
    ///
    /// `mutable_addr_splitting_one_dim.mlir:200-225` (`constant_start_addr_1`), run with
    /// `--dcc-mutable-addr-splitting-max-mutable-size=91200` — 5700 `f16` elements.
    ///
    /// | input | value | from |
    /// |---|---|---|
    /// | start address | 3072 | `%c3072`, the `hbm` view's |
    /// | `%arg1` | `0 to 4`, subscript `%arg1 * 3` on `d1` | coeff `3 * 64 = 192`, weight `(4-2) * 192 = 384` |
    /// | `%arg2` | `0 to %719` where `maxValue = 8`, subscript `%arg2 * 8` on `d2` | coeff `8 * 256 = 2048`, weight `(8-2) * 2048 = 12288` |
    /// | `max_mutable` | `0 + 384 + 12288 = 12672` | |
    /// | overflow | `12672 - 5700 = 6972` | |
    ///
    /// ⭐ AND THE VENDOR'S OUTPUT IS THE ARITHMETIC READ BACK: `%[[VAL_15]] = arith.constant 4` with
    /// `cmpi slt %[[VAL_12]], %[[VAL_15]]` is the partition size, the single `scf.if` is
    /// `total_partitions - 1` conditionals, and the second view's `arith.constant 11264` is
    /// `3072 + 8192` — the start plus `shifted_mutable`.
    ///
    /// ⛔ THE SYMBOLIC BOUND IS WHY [`get_loop_trip_count`] HAS TO READ `maxValue`: without it
    /// `num_iters_` is 0 and this case has no dimension to split at all.
    #[test]
    fn the_one_dim_answer_key_splits_the_symbolic_dim_into_partitions_of_four() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 3072,
            layout: &layout,
            ty: &ty,
        };
        // Already sorted by [`sort_data_based_on_weight`]: 12288 before 384.
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 192, 4, 384)];
        let mut conditionals = Conditionals::default();

        let split = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span_overflowing_by::<Dd2>(6972, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                // `extra_iters = ceil(6972 / 2048) = 4`, so `8 - 4 = 4`.
                sizes: vec![4],
                total_partitions: 2,
                shifted_mutable: 8192,
                // `6972 - 4 * 2048 = -1220`.
                remaining_overflow: -1220,
            })
        );
        // One `scf.if`, which is what the answer key contains.
        assert_eq!(conditionals, Conditionals(1));
        // `%[[VAL_11]] = arith.constant 11264` — the second partition's view.
        assert_eq!(view.start + 8192, 11_264);
    }

    /// 🎯 186/384 — ⭐⭐ IBM'S `select_ub` ANSWER KEY: EIGHT ITERATIONS INTO **3**, WHICH IS **THREE**
    /// PARTITIONS.
    ///
    /// `mutable_addr_splitting_one_dim.mlir:250-300` (`select_ub`), same 5700-element limit.
    ///
    /// ⭐⭐ THE CASE WHERE `num_iters % partition_size != 0`, AND SO THE ONE THAT PROVES THE SECOND
    /// CEILING DIVISION. `8 / 3` is 2 with a remainder, so `num_partitions` is **3** and the answer key
    /// carries two nested `scf.if`s — against `arith.constant 3` and `arith.constant 6`, the partition
    /// boundaries. A floor division here would emit one conditional and leave the last two iterations
    /// reading the wrong address.
    ///
    /// | input | value |
    /// |---|---|
    /// | start address | 2048 |
    /// | `%arg1` `0 to 4`, `%arg1 * 16` on `d1` | coeff 1024, weight 2048 |
    /// | `%arg2` `0 to select(8, 4)`, `%arg2 * 8` on `d2` | coeff 2048, `num_iters` **8** — the larger arm — weight 12288 |
    /// | overflow | `14336 - 5700 = 8636` |
    ///
    /// ⭐ `shifted_mutable = 2 * 3 * 2048 = 12288`, and the three views start at 2048, 8192 and 14336 —
    /// a step of `partition_size * coeff = 6144`.
    #[test]
    fn the_select_upper_bound_answer_key_needs_three_partitions() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 2048,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 1024, 4, 2048)];
        let mut conditionals = Conditionals::default();

        let split = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span_overflowing_by::<Dd2>(8636, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                // `extra_iters = ceil(8636 / 2048) = 5`, so `8 - 5 = 3`.
                sizes: vec![3],
                total_partitions: 3,
                shifted_mutable: 12_288,
                remaining_overflow: -1604,
            })
        );
        // Two nested `scf.if`s.
        assert_eq!(conditionals, Conditionals(2));
        // The two boundaries the answer key compares `%arg2` against, and the three view starts.
        assert_eq!([3, 6], [3, 2 * 3]);
        assert_eq!(
            [view.start, view.start + 6144, view.start + 12_288],
            [2048, 8192, 14_336]
        );
    }

    /// 🎯 186/384 — ⭐⭐ IBM'S `multi_dim` ANSWER KEY: THE **HEAVIER** DIMENSION IS THE ONLY ONE TOUCHED.
    ///
    /// `mutable_addr_splitting_multi_dim.mlir:180-205` (`constant_start_addr_1`), run with
    /// `--dcc-mutable-addr-splitting-max-mutable-size=75000` — `75000 / 16 = 4687` elements, a
    /// TRUNCATING division.
    ///
    /// | input | value |
    /// |---|---|
    /// | start address | 2048 |
    /// | `%arg1` `0 to 8`, `%arg1 * 16 + 1` on `d1` | coeff 1024, weight 6144, **and a constant offset of 64** |
    /// | `%arg2` `0 to 4`, `%arg2 * 8` on `d2` | coeff 2048, weight 4096 |
    /// | `max_mutable` | `64 + 6144 + 4096 = 10304` |
    /// | overflow | `10304 - 4687 = 5617` |
    ///
    /// ⛔⛔ THE SORT PUTS THE **SMALLER-COEFFICIENT** DIMENSION FIRST HERE, BECAUSE WEIGHT IS
    /// `(num_iters - 2) * coeff` AND `%arg1` HAS TWICE THE ITERATIONS. Splitting by coefficient instead
    /// would cut `%arg2` (2048) and produce a different program: this is the case where
    /// [`sort_data_based_on_weight`]'s key is observable.
    ///
    /// ⭐ FOUR PARTITIONS FROM ONE DIMENSION — `8 / 2` — so three nested `scf.if`s and four views at
    /// 2048, 4096, 6144 and 8192.
    #[test]
    fn the_multi_dim_answer_key_splits_the_heavier_dim_only() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 2048,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(1, 1024, 8, 6144), iterator(2, 2048, 4, 4096)];
        let mut conditionals = Conditionals::default();

        let split = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span_overflowing_by::<Dd2>(5617, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                // `extra_iters = ceil(5617 / 1024) = 6`, so `8 - 6 = 2`.
                sizes: vec![2],
                total_partitions: 4,
                // `(4 - 1) * 2 * 1024`.
                shifted_mutable: 6144,
                remaining_overflow: -527,
            })
        );
        assert_eq!(conditionals, Conditionals(3));
        assert_eq!(
            [
                view.start,
                view.start + 2048,
                view.start + 4096,
                view.start + 6144
            ],
            [2048, 4096, 6144, 8192]
        );
        // ⭐ AND THE UNTOUCHED DIMENSION IS NOT IN THE RESULT AT ALL — the walk `break`s, so a consumer
        // that iterated `mas_data` rather than `sizes` would compare `%arg2` against nothing.
        assert_eq!(mas_data.len(), 2);
    }

    /// 🎯 186/384 — ⭐⭐ IBM'S `time_dims` ANSWER KEY: SIXTEEN ITERATIONS INTO **13**.
    ///
    /// `mutable_addr_splitting_time_dims.mlir` (`constant_start_addr`), run with
    /// `--dcc-mutable-addr-splitting-max-mutable-size=600000` — 37500 `f16` elements.
    ///
    /// ⭐ THE `mas_data` HERE IS [`synthesize_time_info`]'s, NOT `initMASData`'s: the time bounds
    /// `[16, 3, 3]` composed through the time order carry weights `[28672, 64, 64]` and a
    /// `max_mutable` of 43136, which overflows by **5636**.
    ///
    /// ⭐ A START ADDRESS OF **ZERO**, so `shifted_mutable` IS the second partition's view address —
    /// `26624 = 13 * 2048`, which is what the answer key's second `arith.constant` says.
    #[test]
    fn the_time_dims_answer_key_splits_sixteen_into_thirteen() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [
            iterator(2, 2048, 16, 28_672),
            iterator(1, 64, 3, 64),
            iterator(0, 64, 3, 64),
        ];
        let mut conditionals = Conditionals::default();

        let split = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span_overflowing_by::<Dd2>(5636, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                // `extra_iters = ceil(5636 / 2048) = 3`, so `16 - 3 = 13`.
                sizes: vec![13],
                // `16 % 13 != 0`, so `16 / 13 + 1`.
                total_partitions: 2,
                shifted_mutable: 26_624,
                remaining_overflow: -508,
            })
        );
        assert_eq!(conditionals, Conditionals(1));
        assert_eq!(view.start + 26_624, 26_624);
    }

    /// 🎯 186/384 — ⭐⭐ IBM'S SEN1P5 ANSWER KEY, WHICH EXISTS **ONLY** TO EXERCISE THE PARITY RULE.
    ///
    /// `mutable_addr_splitting_one_dim_sen1p5.mlir` — *"Sentient 1.5 test separated as it cannot run on
    /// Z due to lack of SENARCH env var. Tests that odd EBRs are properly handled in
    /// MutableAddrSplittingPass."* (`:5-6`), run with `SENARCH=sen1p5` and
    /// `--dcc-mutable-addr-splitting-max-mutable-size=100000` — 6250 `f16` elements.
    ///
    /// | input | value |
    /// |---|---|
    /// | start address | **2112** — `2112 / 64 = 33`, an **ODD** number of sticks |
    /// | `%arg1` `0 to 4`, `%arg1 * 16` on `d1` | coeff 1024, weight 2048 |
    /// | `%arg2` `0 to 8`, `%arg2 * 8` on `d2` | coeff 2048, weight 12288 |
    /// | `max_mutable` | `64 + 2048 + 12288 = 14400` — the `64` is the `d0` subscript |
    /// | overflow | `14400 - 6250 = 8150` |
    ///
    /// `extra_iters = ceil(8150 / 2048) = 4`, `partition_size = 4` — the answer key's
    /// `%[[VAL_15]] = arith.constant 4` — `num_partitions = 8 / 4 = 2`, `shifted_mutable = 8192`.
    ///
    /// # ⛔⛔ AN ODD STICK PLUS AN EVEN SHIFT IS THE PATH, AND IT COSTS ONE STICK
    ///
    /// `is_even` is false and `is_shift_even` is true, so the two differ and both `DT_CHECK`s in the
    /// SEN1P5 block are evaluated. The all-odd one holds trivially here — a single constant is odd or
    /// it is not — and the room check needs 64 elements, which the `EBR` has.
    ///
    /// ⭐⭐ AND THE STICK IS ACTUALLY SPENT, BY A LATER FUNCTION: the answer key's first partition
    /// starts at `arith.constant 2048`, which is `2112 - 64` — one stick BELOW the view's own start —
    /// and the second at `10240 = 2048 + 8192`. This function only checks that the room exists; nothing
    /// it returns carries the shift-back, which is why `shifted_mutable` is 8192 and not 8256.
    #[test]
    fn the_sen1p5_answer_key_takes_the_parity_path_and_passes() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 2112,
            layout: &layout,
            ty: &ty,
        };
        // `2112 / 64 = 33` — odd, which is the whole point of the file.
        assert_eq!(2112 / 64, 33);
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 1024, 4, 2048)];
        let mut conditionals = Conditionals::default();

        let split = calculate_partition_sizes::<Sen1p5>(
            &mas_data,
            L3Half::Load,
            &view,
            span_overflowing_by::<Sen1p5>(8150, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                sizes: vec![4],
                total_partitions: 2,
                shifted_mutable: 8192,
                // `8150 - 4 * 2048 = -42`.
                remaining_overflow: -42,
            })
        );
        assert_eq!(conditionals, Conditionals(1));
        // The two view starts the answer key writes — the first one a stick below `view.start`.
        assert_eq!([2112 - 64, 2112 - 64 + 8192], [2048, 10_240]);
    }

    /// 🎯 186/384 — A DIMENSION LIGHT ENOUGH TO SPEND ENTIRELY GETS A PARTITION SIZE OF **1**, AND THE
    /// WALK GOES ON.
    ///
    /// ⭐ THE `if` BRANCH (`:889-895`), WHICH NONE OF THE FIVE ANSWER KEYS REACHES — every one of them
    /// has a heaviest dimension that absorbs the whole overflow on its own. It is reached when the
    /// overflow exceeds even the heaviest weight: that dimension is cut to one iteration per partition,
    /// its whole weight is subtracted, `total_partitions` takes ALL of its iterations, and the next
    /// dimension is asked the same question.
    ///
    /// Here the 12288-weight dimension is spent whole (overflow 12400 → 112, eight partitions), and the
    /// 128-weight one then takes the `else` branch for the remainder — so `sizes` has TWO entries, in
    /// the sorted order, and `total_partitions` is a product.
    #[test]
    fn a_light_dimension_is_split_to_one_iteration_and_the_walk_continues() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 64, 4, 128)];
        let mut conditionals = Conditionals::default();

        let split = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span_overflowing_by::<Dd2>(12_400, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                // `1` for the dimension spent whole, then `4 - ceil(112 / 64) = 2`.
                sizes: vec![1, 2],
                // `8 * 2`.
                total_partitions: 16,
                // `12288 + (2 - 1) * 2 * 64`.
                shifted_mutable: 12_416,
                // `112 - 2 * 64`.
                remaining_overflow: -16,
            })
        );
        // ⛔ FIFTEEN CONDITIONALS FROM ONE TRANSFER — one under the shipped budget. See
        // [`the_conditional_budget_is_shared_across_the_program_unit`].
        assert_eq!(conditionals, Conditionals(15));
    }

    /// 🎯 186/384 — ⛔ SPLITTING EVERY DIMENSION AS FAR AS IT GOES CAN STILL NOT BE ENOUGH.
    ///
    /// `DT_CHECK_MSG(mutable_overflow <= 0, "Cannot split enough to bring mutable address back in
    /// range.")` (`:922-923`) — the walk ran out of dimensions with the `if` branch taken every time,
    /// so every dimension is down to one iteration per partition and the address still does not fit.
    ///
    /// ⭐ THE SAME TWO DIMENSIONS AS
    /// [`a_light_dimension_is_split_to_one_iteration_and_the_walk_continues`], 100 elements further
    /// out: `12500 - 12288 - 128 = 84` left over.
    #[test]
    fn splitting_every_dimension_to_one_iteration_can_still_be_not_enough() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 64, 4, 128)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Dd2>(
                &mas_data,
                L3Half::Load,
                &view,
                span_overflowing_by::<Dd2>(12_500, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::CannotSplitEnough {
                remaining_overflow: 84
            }
        );
        // ⭐ AND NOTHING WAS CHARGED — the count advances only past every check but the last.
        assert_eq!(conditionals, Conditionals::default());
    }

    /// 🎯 186/384 — ⛔ AN EMPTY ITERATOR LIST IDENTIFIES NO PARTITIONS, AND ONLY A NON-OVERFLOWING SPAN
    /// GETS THERE.
    ///
    /// `DT_CHECK_MSG(!partition_sizes.empty(), "No partitions were identified.")` (`:924`).
    ///
    /// ⚠️ THE CHECK ABOVE IT SHADOWS THIS ONE FOR EVERY OTHER INPUT. With no dimensions the overflow is
    /// untouched, so a positive one reports
    /// [`Partitioning::CannotSplitEnough`] first — which is why this test has to pass a span exactly AT
    /// the limit. In the pass itself the case is already gone:
    /// [`has_mutable_addr_overflow`] answers
    /// [`MutableAddrOverflow::NoLoopsToSplit`] for an empty `mas_data` before `setupForPartitioning`
    /// runs (`:806-812`).
    #[test]
    fn an_empty_iterator_list_identifies_no_partitions() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Dd2>(
                &[],
                L3Half::Load,
                &view,
                span_overflowing_by::<Dd2>(0, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::NoPartitionsIdentified
        );
        assert_eq!(conditionals, Conditionals::default());
    }

    /// 🎯 186/384 — ⛔⛔ THE SHIFT HAS TO **FIT IN THE IMMUTABLE HALF**, WHICH IS WHY THE PASS ASKS FOR
    /// BOTH RANGES.
    ///
    /// `DT_CHECK_MSG(shifted_mutable <= immutable_space, …)` (`:928-930`). Splitting does not shrink an
    /// address, it moves it from the `EAR` into the `EBR` — so a transfer whose immutable base already
    /// sits one element below the top of the `EBR` has nowhere to put the 8192 elements the split wants
    /// to shift, no matter how the iteration space is cut.
    ///
    /// ⭐ THE SAME `one_dim` SPLIT AS
    /// [`the_one_dim_answer_key_splits_the_symbolic_dim_into_partitions_of_four`], with the start
    /// address moved to the top of the range instead of 3072.
    #[test]
    fn the_shift_cannot_exceed_the_room_left_in_the_immutable_half() {
        let (layout, ty) = answer_key_memref();
        let start = start_leaving_immutable_space::<Dd2>(1, DataType::Sen169Fp16);
        let view = ConstStartMemView {
            from: Val(0),
            start,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 192, 4, 384)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Dd2>(
                &mas_data,
                L3Half::Load,
                &view,
                span_overflowing_by::<Dd2>(6972, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::ShiftExceedsImmutableRange {
                shifted_mutable: 8192,
                immutable_space: 1,
            }
        );
        assert_eq!(conditionals, Conditionals::default());
    }

    /// 🎯 186/384 — ⛔⛔ AN **ODD** IMMUTABLE STICK AND AN **EVEN** SHIFT NEED ONE SPARE STICK, AND
    /// 32 ELEMENTS IS NOT ONE.
    ///
    /// `DT_CHECK_MSG(immutable_space >= num_elems_in_stick, "No mutable space to shift back one stick to
    /// maintain even immutable address.")` (`:948-950`) — the failing half of the path
    /// [`the_sen1p5_answer_key_takes_the_parity_path_and_passes`] takes and passes.
    ///
    /// ⭐ THE INPUT IS TUNED TO THE PARITY, NOT TO THE SIZE. A start address 32 elements below the top
    /// of the SEN1P5 `EBR` happens to sit at an ODD number of 64-element sticks, and the split's
    /// `shifted_mutable` of 4 is EVEN — so the two disagree, the block runs, and the 32 elements left
    /// are half of what realigning would cost.
    #[test]
    fn an_odd_immutable_stick_and_an_even_shift_need_a_stick_of_room() {
        let (layout, ty) = answer_key_memref();
        let start = start_leaving_immutable_space::<Sen1p5>(32, DataType::Sen169Fp16);
        // An odd number of sticks, and 64 elements to a stick at `f16`.
        assert_eq!(elems_in_stick::<Sen1p5>(DataType::Sen169Fp16).get(), 64);
        assert_eq!((start / 64) % 2, 1);
        let view = ConstStartMemView {
            from: Val(0),
            start,
            layout: &layout,
            ty: &ty,
        };
        // `extra_iters = ceil(2 / 1) = 2`, size `4`, `6 % 4 != 0` so two partitions,
        // `shifted_mutable = 1 * 4 * 1 = 4` — even.
        let mas_data = [iterator(0, 1, 6, 4)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Sen1p5>(
                &mas_data,
                L3Half::Load,
                &view,
                span_overflowing_by::<Sen1p5>(2, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::NoRoomToRealignTheImmutableStick {
                immutable_space: 32,
                elems_in_stick: 64,
            }
        );
        assert_eq!(conditionals, Conditionals::default());
    }

    /// 🎯 186/384 — ⭐⭐ AND AN **ODD** SHIFT ON THE SAME ODD STICK NEEDS NO ROOM AT ALL.
    ///
    /// `if (is_even != is_shift_even)` (`:945`) — the room check is inside the parity mismatch, so
    /// exactly the same 32 elements of `EBR` are enough when the shift preserves the parity it found.
    /// One element of difference in the overflow changes `shifted_mutable` from 4 to 3 and turns a
    /// refusal into a split.
    ///
    /// ⛔ WHICH IS WHY THE PARITY IS A COMPARISON OF TWO BOOLEANS AND NOT A TEST ON EITHER ONE. A port
    /// that checked `!is_even` — or `!is_shift_even` — would refuse this program and admit the previous
    /// one.
    #[test]
    fn an_odd_shift_on_an_odd_stick_needs_no_room_at_all() {
        let (layout, ty) = answer_key_memref();
        let start = start_leaving_immutable_space::<Sen1p5>(32, DataType::Sen169Fp16);
        let view = ConstStartMemView {
            from: Val(0),
            start,
            layout: &layout,
            ty: &ty,
        };
        // `extra_iters = 3`, size `3`, `6 % 3 == 0` so two partitions,
        // `shifted_mutable = 1 * 3 * 1 = 3` — odd, matching the odd stick.
        let mas_data = [iterator(0, 1, 6, 4)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Sen1p5>(
                &mas_data,
                L3Half::Load,
                &view,
                span_overflowing_by::<Sen1p5>(3, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::Split(PartitionSizes {
                sizes: vec![3],
                total_partitions: 2,
                shifted_mutable: 3,
                remaining_overflow: 0,
            })
        );
        assert_eq!(conditionals, Conditionals(1));
    }

    /// 🎯 186/384 — ⛔⛔ AND THE PARITY RULE DOES NOT EXIST BEFORE SEN1P5.
    ///
    /// `if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA)` (`:932`) — the guard the whole block sits
    /// behind, and the reason IBM keeps a separate answer key for it. The same odd-stick address with
    /// the same even shift and the same 32 elements of room SPLITS on DD2, where nothing asks about
    /// sticks.
    ///
    /// ⭐ AN [`IsaGen`] COMPARISON, NOT A FLAG. `IsaGen` derives `Ord` with `Rcudd1a` before `Sen1p5`,
    /// so `A::GEN >= IsaGen::Sen1p5` is the reference's own `>=` — and a future generation inherits the
    /// rule rather than losing it.
    #[test]
    fn the_parity_rule_does_not_exist_before_sen1p5() {
        let (layout, ty) = answer_key_memref();
        // DD2's `EBR` is 2^30 sticks and SEN1P5's is 2^32, so "32 elements from the top" is a different
        // address on each — see [`the_immutable_range_is_four_times_larger_from_sen1p5`].
        let start = start_leaving_immutable_space::<Dd2>(32, DataType::Sen169Fp16);
        assert_eq!((start / 64) % 2, 1, "an odd number of sticks, as before");
        let view = ConstStartMemView {
            from: Val(0),
            start,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(0, 1, 6, 4)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Dd2>(
                &mas_data,
                L3Half::Load,
                &view,
                // The even shift that SEN1P5 refuses.
                span_overflowing_by::<Dd2>(2, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::Split(PartitionSizes {
                sizes: vec![4],
                total_partitions: 2,
                shifted_mutable: 4,
                remaining_overflow: 0,
            })
        );
        assert!(Dd2::GEN < IsaGen::Sen1p5);
    }

    /// 🎯 186/384 — ⛔ A DIMENSION WITH NO COEFFICIENT IS NOT DIVIDED BY.
    ///
    /// `int64_t extra_iters = mutable_overflow / mas_data[i].composed_coeff_;` (`:905`) — a division by
    /// a field nothing in the reference constrains. See
    /// [`Partitioning::DimensionHasNoCoefficient`] for why it is unreachable from the pass and guarded
    /// anyway.
    #[test]
    fn a_dimension_with_no_coefficient_is_not_divided_by() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        // A weight above the overflow, which is what selects the dividing branch.
        let mas_data = [iterator(7, 0, 4, 6)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Dd2>(
                &mas_data,
                L3Half::Load,
                &view,
                span_overflowing_by::<Dd2>(5, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::DimensionHasNoCoefficient { dim: 7 }
        );
    }

    /// 🎯 186/384 — ⛔ AND NEITHER IS A PARTITION SIZE OF ZERO.
    ///
    /// `mas_data[i].num_iters_ % partition_sizes.back()` (`:911`) — the second division, by a number
    /// the function itself just computed. `ceil(8 / 4) = 2` extra iterations out of 2 leaves nothing.
    /// See [`Partitioning::PartitionSizeIsZero`].
    #[test]
    fn a_partition_size_of_zero_is_not_divided_by() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(5, 4, 2, 9)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            calculate_partition_sizes::<Dd2>(
                &mas_data,
                L3Half::Load,
                &view,
                span_overflowing_by::<Dd2>(8, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
            ),
            Partitioning::PartitionSizeIsZero { dim: 5 }
        );
    }

    /// 🎯 186/384 — ⛔⛔ THE CONDITIONAL BUDGET IS **SHARED ACROSS THE PROGRAM UNIT**, SO THE SECOND
    /// TRANSFER IS THE ONE THAT BLOWS IT.
    ///
    /// `num_conditionals_` is a pass member (`:222`) reset per `dataflow::ProgramUnitOp` (`:244-246`),
    /// not per transfer — so two identical transfers in one unit are NOT two identical decisions. Here
    /// the first splits into 16 partitions for 15 conditionals, one under the shipped budget, and the
    /// second asks for 15 more and is refused.
    ///
    /// ⭐ WHICH IS WHY [`Conditionals`] IS A `&mut` ARGUMENT RATHER THAN A RETURN VALUE. The caller owns
    /// the reset, and a port that recomputed the count per transfer would accept a program the
    /// reference rejects.
    #[test]
    fn the_conditional_budget_is_shared_across_the_program_unit() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let mas_data = [iterator(2, 2048, 8, 12_288), iterator(1, 64, 4, 128)];
        let span = span_overflowing_by::<Dd2>(12_400, DataType::Sen169Fp16);
        let mut conditionals = Conditionals::default();

        let first = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span,
            DataType::Sen169Fp16,
            &mut conditionals,
        );
        assert!(matches!(first, Partitioning::Split(_)));
        assert_eq!(conditionals, Conditionals(15));

        let second = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Load,
            &view,
            span,
            DataType::Sen169Fp16,
            &mut conditionals,
        );
        assert_eq!(
            second,
            Partitioning::TooManyConditionals {
                total: Conditionals(30),
                limit: 16,
            }
        );
        // ⛔ AND THE COUNTER IS ALREADY ADVANCED, because the reference adds before it checks (`:958`).
        assert_eq!(conditionals, Conditionals(30));

        // ⭐ THE NEXT PROGRAM UNIT STARTS OVER, which is the reset at `:244-246`.
        let mut next_unit = Conditionals::default();
        assert!(matches!(
            calculate_partition_sizes::<Dd2>(
                &mas_data,
                L3Half::Load,
                &view,
                span,
                DataType::Sen169Fp16,
                &mut next_unit,
            ),
            Partitioning::Split(_)
        ));
        assert_eq!(next_unit, Conditionals(15));
    }

    /// 🎯 186/384 — ⛔⛔ AND THE SHIPPED BUDGET IS **16**, NOT UNLIMITED.
    ///
    /// `cl::init(16)` (`:70-75`). The two size overrides beside it default to their `-1` sentinel and so
    /// never bind ([`MAX_MUTABLE_SIZE`], [`MAX_IMMUTABLE_SIZE`]); this one binds in every compile, and
    /// `-1` is the way OUT of it rather than the default. All five answer keys need one, two or three
    /// conditionals and stay well inside it.
    #[test]
    fn the_shipped_conditional_budget_is_sixteen() {
        assert_eq!(MAX_NUM_CONDITIONALS, Some(16));
        assert_eq!(MAX_MUTABLE_SIZE, None, "-1: unset");
        assert_eq!(MAX_IMMUTABLE_SIZE, None, "-1: unset");
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  187 — constructConditionals
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// THE TREE AS MLIR TEXT, from the region the reference built it into.
    fn printed(ops: &[DfirOp]) -> String {
        let mut out = String::new();
        for op in ops {
            print::emit(&mut out, op, 0);
        }
        out
    }

    /// EVERY `scf.if` IN A REGION AND EVERYTHING INSIDE THEM.
    fn conditionals_in(ops: &[DfirOp]) -> usize {
        ops.iter()
            .map(|op| match op {
                DfirOp::Scf(scf::Op::If {
                    body, else_body, ..
                }) => 1 + conditionals_in(body) + conditionals_in(else_body),
                _ => 0,
            })
            .sum()
    }

    /// THE TWO REGIONS OF THE `scf.if` A CHAIN ENDS WITH.
    ///
    /// ⭐ THE LAST OP OF A REGION IS THE TERMINATOR, NOT THE CONDITIONAL — see [`with_terminator`].
    fn branches(ops: &[DfirOp]) -> (&[DfirOp], &[DfirOp]) {
        ops.iter()
            .rev()
            .find_map(|op| match op {
                DfirOp::Scf(scf::Op::If {
                    body, else_body, ..
                }) => Some((body.as_slice(), else_body.as_slice())),
                _ => None,
            })
            .unwrap_or((&[], &[]))
    }

    /// A REGION HOLDING NOTHING BUT THE TERMINATOR — the partition `fillPartitions` fills.
    fn empty_partition() -> Vec<DfirOp> {
        vec![DfirOp::Scf(scf::Op::Yield {
            operands: Vec::new(),
        })]
    }

    /// ONE ITERATOR OF A TRANSFER THAT IS BEING SPLIT, WITH THE INDUCTION VARIABLE IT COMPARES.
    ///
    /// ⭐ [`iterator`] LEAVES `iter_arg` UNSET, because [`calculate_partition_sizes`] never reads it;
    /// every conditional [`construct_conditionals`] emits does.
    fn split_dim(iter_arg: Val, dim: u32, composed_coeff: i64, num_iters: i64) -> MasData {
        MasData {
            iter_arg: Some(iter_arg),
            dim,
            composed_coeff,
            num_iters,
            // ⭐ `(num_iters - 2) * composed_coeff`, which nothing below this line reads.
            weight: (num_iters - 2) * composed_coeff,
        }
    }

    /// 🎯 187/384 — ⭐⭐ IBM'S `select_ub` ANSWER KEY, BOUNDARY FOR BOUNDARY AND REGION FOR REGION.
    ///
    /// Eight iterations in partitions of three (`sizes: [3]`, three partitions) put boundaries at **3**
    /// and **6**, with the second conditional inside the first one's `else` —
    /// `mutable_addr_splitting_one_dim.mlir:210-230`:
    ///
    /// ```text
    /// %[[VAL_21:.*]] = arith.constant 3 : index
    /// %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_21]] : index
    /// scf.if %[[VAL_22]] {
    ///   ..
    /// } else {
    ///   %[[VAL_26:.*]] = arith.constant 6 : index
    ///   %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_26]] : index
    ///   scf.if %[[VAL_27]] { .. } else { .. }
    /// }
    /// ```
    ///
    /// ⭐ AND THE VENDOR'S SSA ORDER IS THE MINTING ORDER: the outer boundary is `%21` and the inner
    /// one `%26`, so the constants are minted from the outside in. The three partitions are the three
    /// regions this leaves empty.
    #[test]
    fn the_select_ub_answer_key_nests_the_second_boundary_in_the_first_ones_else() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let tree = construct_conditionals(&mut vals, &[split_dim(iv, 2, 2048, 8)], &[3]);

        assert_eq!(
            printed(tree.ops().unwrap_or_default()),
            concat!(
                "%1 = arith.constant 3 : index\n",
                "%2 = arith.cmpi slt, %0, %1 : index\n",
                "scf.if %2 {\n",
                "} else {\n",
                "  %3 = arith.constant 6 : index\n",
                "  %4 = arith.cmpi slt, %0, %3 : index\n",
                "  scf.if %4 {\n",
                "  } else {\n",
                "  }\n",
                "}\n",
            )
        );
        assert_eq!(iv, Val(0), "the induction variable both conditionals read");
    }

    /// 🎯 187/384 — AND ITS SIBLING KEY, WHOSE ONE BOUNDARY IS **4**.
    ///
    /// `constant_start_addr_1` splits the same eight iterations into partitions of four, so there is one
    /// conditional and two partitions — `mutable_addr_splitting_one_dim.mlir:30-32`:
    ///
    /// ```text
    /// %[[VAL_17:.*]] = arith.constant 4 : index
    /// %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_17]] : index
    /// scf.if %[[VAL_18]] {
    /// ```
    ///
    /// ⛔ `8 / 4 = 2` PARTITIONS NEEDS ONE CONDITIONAL, NOT TWO. `while (ub < num_iters_)` stops at
    /// `ub == 8` because the last partition is the `else` of everything above it — a `<=` there would
    /// emit a comparison no iteration can fail.
    #[test]
    fn the_one_dim_answer_key_has_a_single_boundary_at_four() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let tree = construct_conditionals(&mut vals, &[split_dim(iv, 2, 2048, 8)], &[4]);

        assert_eq!(
            printed(tree.ops().unwrap_or_default()),
            concat!(
                "%1 = arith.constant 4 : index\n",
                "%2 = arith.cmpi slt, %0, %1 : index\n",
                "scf.if %2 {\n",
                "} else {\n",
                "}\n",
            )
        );
    }

    /// 🎯 187/384 — ⭐⭐ AND THE `multi_dim` KEY'S THREE, WHICH IS THE DEEPEST CHAIN IBM SHIPS.
    ///
    /// Eight iterations in partitions of two: boundaries **2**, **4**, **6** and four partitions
    /// (`mutable_addr_splitting_multi_dim.mlir:30-56`), each nested in the previous one's `else` and
    /// each comparing the SAME induction variable `%[[VAL_13]]` — one dimension split four ways, not
    /// three dimensions.
    ///
    /// ⭐ THE CONDITIONAL COUNT IS THE PARTITION COUNT LESS ONE, which is exactly what
    /// [`calculate_partition_sizes`] charged to the [`Conditionals`] budget for this key: `4 - 1 = 3`.
    #[test]
    fn the_multi_dim_answer_key_chains_three_boundaries_against_one_iterator() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let tree = construct_conditionals(&mut vals, &[split_dim(iv, 1, 1024, 8)], &[2]);
        let ops = tree.ops().unwrap_or_default();

        assert_eq!(
            printed(ops),
            concat!(
                "%1 = arith.constant 2 : index\n",
                "%2 = arith.cmpi slt, %0, %1 : index\n",
                "scf.if %2 {\n",
                "} else {\n",
                "  %3 = arith.constant 4 : index\n",
                "  %4 = arith.cmpi slt, %0, %3 : index\n",
                "  scf.if %4 {\n",
                "  } else {\n",
                "    %5 = arith.constant 6 : index\n",
                "    %6 = arith.cmpi slt, %0, %5 : index\n",
                "    scf.if %6 {\n",
                "    } else {\n",
                "    }\n",
                "  }\n",
                "}\n",
            )
        );
        assert_eq!(
            conditionals_in(ops),
            3,
            "one fewer than the four partitions"
        );
    }

    /// 🎯 187/384 — AND THE `time_dims` KEY COMPARES A TIME DIMENSION'S INDUCTION VARIABLE.
    ///
    /// Sixteen iterations in partitions of thirteen — `mutable_addr_splitting_time_dims.mlir:33-36`:
    ///
    /// ```text
    /// affine.for %[[VAL_13:.*]] = 0 to 16 {
    ///   %[[VAL_14:.*]] = arith.constant 13 : index
    ///   %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_14]] : index
    /// ```
    ///
    /// ⭐⭐ `%[[VAL_13]]` IS A LOOP [`create_explicit_time_loops`] WROTE. That iterator does not exist
    /// in the input — the transfer's time dimension is implicit in its `time_order` map — so the
    /// `iter_arg_` this function compares was filled in by that function at `:491`, three lines before
    /// `createPartitions` runs.
    #[test]
    fn the_time_dims_answer_key_compares_the_explicit_time_loops_iterator() {
        let mut vals = Values::default();
        let time_iv = vals.mint();
        let tree = construct_conditionals(
            &mut vals,
            &[MasData {
                iter_arg: Some(time_iv),
                dim: 3,
                composed_coeff: 2048,
                num_iters: 16,
                weight: 28672,
            }],
            &[13],
        );

        assert_eq!(
            printed(tree.ops().unwrap_or_default()),
            concat!(
                "%1 = arith.constant 13 : index\n",
                "%2 = arith.cmpi slt, %0, %1 : index\n",
                "scf.if %2 {\n",
                "} else {\n",
                "}\n",
            )
        );
    }

    /// 🎯 187/384 — ⭐⭐ EVERY REGION IS PRESENT AND EVERY LEAF IS EMPTY, WHICH IS WHAT MAKES IT A
    /// PARTITION.
    ///
    /// `scf::IfOp::create(cond_builder, op->getLoc(), ub_cond.getResult(), true)` — the trailing `true`
    /// is `withElseRegion` (`:1023-1024`), and an [`scf::Op::If`] whose `else_body` is an empty `Vec`
    /// has NO `else` BLOCK AT ALL in this island, a different operation. So both regions of both
    /// conditionals hold exactly the terminator MLIR's builder puts there, and `fillPartitions` has
    /// three places to copy the transfer into.
    #[test]
    fn every_region_exists_and_every_leaf_holds_only_its_terminator() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let tree = construct_conditionals(&mut vals, &[split_dim(iv, 2, 2048, 8)], &[3]);
        let ops = tree.ops().unwrap_or_default();

        let (first_then, first_else) = branches(ops);
        assert_eq!(first_then, empty_partition(), "iterations 0..3");
        let (second_then, second_else) = branches(first_else);
        assert_eq!(second_then, empty_partition(), "iterations 3..6");
        assert_eq!(second_else, empty_partition(), "iterations 6..8");
    }

    /// 🎯 187/384 — ⛔ NO PARTITION SIZES, NO TREE.
    ///
    /// `DT_CHECK(!partition_sizes.empty())` in the caller (`createPartitions`, `:989`) — and this
    /// function would take `prev_partitions.front()` of a chain it never built.
    #[test]
    fn no_partition_sizes_builds_no_tree() {
        let mut vals = Values::default();
        let iv = vals.mint();
        assert_eq!(
            construct_conditionals(&mut vals, &[split_dim(iv, 2, 2048, 8)], &[]),
            ConditionalTree::NoPartitionsToBuild
        );
        assert_eq!(
            vals.issued(),
            iv.0 + 1,
            "a refusal mints nothing: the SSA numbering is untouched"
        );
    }

    /// 🎯 187/384 — ⛔⛔ A PARTITIONED DIMENSION WITH NO ITERATOR HAS NOTHING TO COMPARE.
    ///
    /// `DT_CHECK(mas_data[partition_dim].iter_arg_ != nullptr)` (`:1015`) — the reference's own guard,
    /// which is [`MasData::iter_arg`] being `None`. It is a real shape: a time dimension carries no
    /// induction variable until [`create_explicit_time_loops`] has written its loop.
    #[test]
    fn a_partitioned_dimension_with_no_iterator_has_nothing_to_compare() {
        let mut vals = Values::default();
        assert_eq!(
            construct_conditionals(
                &mut vals,
                &[MasData {
                    iter_arg: None,
                    dim: 3,
                    composed_coeff: 2048,
                    num_iters: 16,
                    weight: 28672,
                }],
                &[13],
            ),
            ConditionalTree::DimensionHasNoIterator { dim: 3 }
        );
    }

    /// 🎯 187/384 — ⛔⛔ A PARTITION SIZE THAT COVERS THE FIRST DIMENSION LEAVES `front()` WITH NOTHING
    /// TO READ.
    ///
    /// `auto root_if_op = prev_partitions.front();` (`:1039`) on the empty chain a one-iteration
    /// dimension produces. See [`ConditionalTree::DimensionNeedsNoConditional`] for why
    /// [`calculate_partition_sizes`] can only reach it through such a dimension.
    #[test]
    fn a_partition_size_that_covers_the_first_dimension_leaves_no_root() {
        let mut vals = Values::default();
        let iv = vals.mint();
        assert_eq!(
            construct_conditionals(
                &mut vals,
                &[MasData {
                    iter_arg: Some(iv),
                    dim: 5,
                    composed_coeff: 64,
                    num_iters: 1,
                    weight: -64,
                }],
                &[1],
            ),
            ConditionalTree::DimensionNeedsNoConditional { dim: 5 }
        );
    }

    /// 🎯 187/384 — ⭐ BUT THE INNERMOST DIMENSION MAY COVER ITSELF, AND THE TREE IS ONE LEVEL SHORT.
    ///
    /// Nothing reads the empty `prev_partitions` an innermost dimension leaves, so the two boundaries
    /// of the outer dimension are the whole tree — and the partitions inside them are undivided.
    #[test]
    fn the_innermost_dimension_may_cover_itself_without_a_boundary() {
        let mut vals = Values::default();
        let outer = vals.mint();
        let inner = vals.mint();
        let tree = construct_conditionals(
            &mut vals,
            &[
                MasData {
                    iter_arg: Some(outer),
                    dim: 0,
                    composed_coeff: 2048,
                    num_iters: 3,
                    weight: 2048,
                },
                MasData {
                    iter_arg: Some(inner),
                    dim: 1,
                    composed_coeff: 64,
                    num_iters: 4,
                    weight: 128,
                },
            ],
            &[1, 4],
        );

        assert_eq!(
            conditionals_in(tree.ops().unwrap_or_default()),
            2,
            "the outer dimension's two boundaries, and nothing inside them"
        );
    }

    /// 🎯 187/384 — ⛔⛔ A PARTITION SIZE OF ZERO WOULD NEVER STOP.
    ///
    /// `ub += partition_sizes[partition_dim]` with a step of zero never reaches `num_iters_`, so
    /// `while (ub < mas_data[partition_dim].num_iters_)` appends `scf.if`s until the process dies. ⭐
    /// [`calculate_partition_sizes`] cannot hand one over — it answers
    /// [`Partitioning::PartitionSizeIsZero`] first — and this function takes the sizes as a slice.
    #[test]
    fn a_partition_size_of_zero_would_never_stop() {
        let mut vals = Values::default();
        let iv = vals.mint();
        assert_eq!(
            construct_conditionals(
                &mut vals,
                &[MasData {
                    iter_arg: Some(iv),
                    dim: 2,
                    composed_coeff: 2048,
                    num_iters: 8,
                    weight: 12288,
                }],
                &[0],
            ),
            ConditionalTree::PartitionSizeIsNotPositive { dim: 2, size: 0 }
        );
    }

    /// 🎯 187/384 — ⛔ AND MORE SIZES THAN ITERATORS IS NOT INDEXED.
    ///
    /// `mas_data[partition_dim]` indexed by a `partition_sizes` position (`:1010-1011`).
    #[test]
    fn more_partition_sizes_than_iterators_is_not_indexed() {
        let mut vals = Values::default();
        let iv = vals.mint();
        assert_eq!(
            construct_conditionals(&mut vals, &[split_dim(iv, 2, 2048, 8)], &[4, 2]),
            ConditionalTree::MorePartitionsThanIterators {
                sizes: 2,
                iterators: 1,
            }
        );
        assert_eq!(vals.issued(), iv.0 + 1, "a refusal mints nothing");
    }

    /// 🎯 187/384 — ⭐⭐ A SECOND DIMENSION IS BUILT INTO **EVERY** PREVIOUS PARTITION.
    ///
    /// `for (int p = 0; p < num_prev_partitions; ++p) { cond_builder = prev_partitions[p]
    /// .getThenBodyBuilder(); .. }` plus one more chain in `prev_partitions.back()
    /// .getElseBodyBuilder()` (`:1044-1053`) — so an outer dimension with two boundaries gets THREE
    /// chains of the inner dimension: one in each `then`, one in the last `else`.
    ///
    /// ⭐ AND THE MINTING IS LEVEL BY LEVEL, NOT DEPTH FIRST: `createConditionalsForPartition(0)` runs
    /// to completion before the loop descends, so the outer dimension's `%2..%5` all precede the inner
    /// dimension's `%6..%11` — the outer `else` chain's constant `%4` is minted BEFORE the inner chain
    /// `%6` that sits inside the outer `then` above it.
    #[test]
    fn a_second_dimension_is_built_into_every_previous_partition() {
        let mut vals = Values::default();
        let outer = vals.mint();
        let inner = vals.mint();
        let tree = construct_conditionals(
            &mut vals,
            &[
                MasData {
                    iter_arg: Some(outer),
                    dim: 0,
                    composed_coeff: 2048,
                    num_iters: 3,
                    weight: 2048,
                },
                MasData {
                    iter_arg: Some(inner),
                    dim: 1,
                    composed_coeff: 64,
                    num_iters: 4,
                    weight: 128,
                },
            ],
            &[1, 2],
        );

        assert_eq!(
            printed(tree.ops().unwrap_or_default()),
            concat!(
                "%2 = arith.constant 1 : index\n",
                "%3 = arith.cmpi slt, %0, %2 : index\n",
                "scf.if %3 {\n",
                "  %6 = arith.constant 2 : index\n",
                "  %7 = arith.cmpi slt, %1, %6 : index\n",
                "  scf.if %7 {\n",
                "  } else {\n",
                "  }\n",
                "} else {\n",
                "  %4 = arith.constant 2 : index\n",
                "  %5 = arith.cmpi slt, %0, %4 : index\n",
                "  scf.if %5 {\n",
                "    %8 = arith.constant 2 : index\n",
                "    %9 = arith.cmpi slt, %1, %8 : index\n",
                "    scf.if %9 {\n",
                "    } else {\n",
                "    }\n",
                "  } else {\n",
                "    %10 = arith.constant 2 : index\n",
                "    %11 = arith.cmpi slt, %1, %10 : index\n",
                "    scf.if %11 {\n",
                "    } else {\n",
                "    }\n",
                "  }\n",
                "}\n",
            )
        );
    }

    /// 🎯 187/384 — ⛔⛔ BUT ONLY THE **LAST** PREVIOUS PARTITION CARRIES THE THIRD DIMENSION, AND THAT
    /// IS THE REFERENCE'S DEFECT.
    ///
    /// `curr_partitions = createConditionalsForPartition(d);` ASSIGNS inside the loop over
    /// `prev_partitions` (`:1049`), so every chain but the last one's is dropped from the list the next
    /// dimension descends into. Three dimensions, an outer pair of boundaries, and the first branch
    /// keeps the second dimension's two conditionals with NOTHING inside them — while the last branch
    /// carries all three levels.
    ///
    /// | region | conditionals inside it | dimensions guarded |
    /// |---|---|---|
    /// | root `then` | 2 | 0 and 1 — ⛔ the third is missing |
    /// | second `then` | 4 | all three |
    /// | second `else` | 5 | all three |
    ///
    /// ⛔ A TRANSFER COPIED INTO A LEAF OF THAT FIRST BRANCH IS SHIFTED FOR THE DIMENSIONS ABOVE IT
    /// AND UNSHIFTED FOR THE ONES BELOW — the wrong address for every iteration of the innermost
    /// dimension but the first. ⭐ IT IS UNREACHABLE FROM THE FIVE ANSWER KEYS, all of which partition
    /// one dimension; see [`Level`] for why `fillPartitions` agrees with the defect and the two cannot
    /// be corrected apart.
    #[test]
    fn only_the_last_previous_partition_carries_the_third_dimension() {
        let mut vals = Values::default();
        let first = vals.mint();
        let second = vals.mint();
        let third = vals.mint();
        let dim = |iter_arg: Val, at: u32, num_iters: i64| MasData {
            iter_arg: Some(iter_arg),
            dim: at,
            composed_coeff: 2048,
            num_iters,
            weight: 2048,
        };
        let tree = construct_conditionals(
            &mut vals,
            &[dim(first, 0, 3), dim(second, 1, 3), dim(third, 2, 4)],
            &[1, 1, 2],
        );
        let ops = tree.ops().unwrap_or_default();

        let (root_then, root_else) = branches(ops);
        assert_eq!(
            conditionals_in(root_then),
            2,
            "the second dimension's chain, with the third dimension missing under it"
        );
        let (second_then, second_else) = branches(root_else);
        assert_eq!(conditionals_in(second_then), 4, "two levels, both filled");
        assert_eq!(conditionals_in(second_else), 5, "two levels and the tail");
        assert_eq!(conditionals_in(ops), 13);
    }

    // ---------------------------------------------------------------- 188/384

    /// ONE SPLIT DIMENSION, spelled out for [`calculate_subscripts_coefficients`].
    ///
    /// ⭐ ONLY `dim_` IS READ. The weights and coefficients that got this dimension chosen belong to
    /// [`calculate_partition_sizes`]; this function asks the subscripts map one question per result,
    /// about one position.
    fn split_at(dim: u32) -> MasData {
        MasData {
            iter_arg: None,
            dim,
            composed_coeff: 0,
            num_iters: 0,
            weight: 0,
        }
    }

    /// 🎯 188/384 — ⭐⭐ IBM'S `one_dim` KEY: THE SPLIT DIMENSION'S OWN SUBSCRIPT MOVES BY **8** AND
    /// THE OTHER TWO DO NOT MOVE AT ALL.
    ///
    /// The transfer loads at `[0, %arg1 * 3, %arg2 * 8]`
    /// (`mutable_addr_splitting_one_dim.mlir:259`) and `%arg2` — dimension **1** of the two-loop nest
    /// — is what [`calculate_partition_sizes`] splits into partitions of four. So the row is
    /// `[0, 0, 8]`, and the vendor's second partition reads
    /// `[0, %13 * 3, %14 * 8 - 32]` (`:40` against `:35`) — `8 * 4`, the coefficient times the
    /// iterations the earlier partitions consumed.
    ///
    /// ⛔ THE FIRST TWO ZEROS ARE THE POINT. A constant subscript and a subscript belonging to the
    /// OTHER loop are both `isFunctionOfDim` failures, and a nonzero there would shift a partition
    /// sideways through the tensor.
    #[test]
    fn the_one_dim_answer_key_moves_only_the_split_subscript() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(3),
                AffineExpr::dim(1).times(8),
            ],
        };
        let coeffs = calculate_subscripts_coefficients(&[split_at(1)], &[4], &subscripts);

        assert_eq!(
            coeffs.rows(),
            Some(
                &[DimCoefficients {
                    dim: 1,
                    coeffs: vec![0, 0, 8],
                }][..]
            )
        );
        let row = &coeffs.rows().unwrap()[0].coeffs;
        assert_eq!(row[2] * 4, 32, "the vendor's own `%14 * 8 - 32`");
    }

    /// 🎯 188/384 — ⭐⭐ AND ITS `select_ub` SIBLING, WHOSE **THREE** PARTITIONS SUBTRACT THE SAME
    /// COEFFICIENT TWICE.
    ///
    /// `[0, %arg1 * 16, %arg2 * 8]` (`mutable_addr_splitting_one_dim.mlir:393`) split on dimension 1
    /// into partitions of three: the row is again `[0, 0, 8]`, and the key writes
    /// `%18 * 8 - 24` then `%18 * 8 - 48` (`:223`, `:228` against `:215`) — `8 * 3` and `8 * 6`.
    ///
    /// ⭐ THE OUTER SUBSCRIPT'S `* 16` IS NOT THIS ROW'S BUSINESS, which is what makes this key worth
    /// a second test beside its sibling: a scaling of ANOTHER dimension sits right there in the map
    /// and still answers 0.
    #[test]
    fn the_select_ub_answer_key_subtracts_the_same_coefficient_per_partition() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(16),
                AffineExpr::dim(1).times(8),
            ],
        };
        let coeffs = calculate_subscripts_coefficients(&[split_at(1)], &[3], &subscripts);

        assert_eq!(
            coeffs.rows().map(|rows| rows[0].coeffs.as_slice()),
            Some(&[0, 0, 8][..])
        );
        let row = &coeffs.rows().unwrap()[0].coeffs;
        assert_eq!(
            (row[2] * 3, row[2] * 6),
            (24, 48),
            "the vendor's own `- 24` and `- 48`"
        );
    }

    /// 🎯 188/384 — ⭐⭐ AND THE `multi_dim` KEY, WHOSE SPLIT SUBSCRIPT CARRIES A CONSTANT THE
    /// COEFFICIENT MUST IGNORE.
    ///
    /// `[0, %arg1 * 16 + 1, %arg2 * 8]` (`mutable_addr_splitting_multi_dim.mlir:309`) split on
    /// dimension **0** into partitions of two. The `+ 1` is part of the subscript, not of its stride,
    /// so the row is `[0, 16, 0]` — and the vendor's three later partitions read
    /// `%13 * 16 - 31`, `- 63`, `- 95` (`:43`, `:51`, `:56` against `:35`), which is `1 - 32`,
    /// `1 - 64` and `1 - 96`.
    ///
    /// ⛔⛔ A COEFFICIENT TAKEN FROM THE WHOLE EXPRESSION WOULD BE **17** HERE, and every partition
    /// after the first would land one stick early. The walk stops at the `Mul` and reads only its
    /// right operand.
    #[test]
    fn the_multi_dim_answer_key_ignores_the_constant_beside_the_stride() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(16).plus(AffineExpr::Const(1)),
                AffineExpr::dim(1).times(8),
            ],
        };
        let coeffs = calculate_subscripts_coefficients(&[split_at(0)], &[2], &subscripts);

        assert_eq!(
            coeffs.rows().map(|rows| rows[0].coeffs.as_slice()),
            Some(&[0, 16, 0][..])
        );
        let row = &coeffs.rows().unwrap()[0].coeffs;
        assert_eq!(
            [1 - row[1] * 2, 1 - row[1] * 4, 1 - row[1] * 6],
            [-31, -63, -95],
            "the vendor's own three partitions"
        );
    }

    /// 🎯 188/384 — ⭐⭐ A DIMENSION THAT APPEARS UNSCALED HAS A COEFFICIENT OF **1**, WHICH IS THE
    /// INITIAL VALUE THE WALK NEVER OVERRIDES.
    ///
    /// `dim_coeffs.push_back(1);` then a walk for a `Mul` (`Dialect/Agen/Utils.cpp:526-540`) — a bare
    /// `d0`, and a `d0` under a `mod` or a `floordiv`, both leave the 1 standing.
    ///
    /// ⚠️ NOT ZERO, WHICH IS THE OTHER PLAUSIBLE ANSWER. `[.., %arg1, ..]` moves by exactly one
    /// subscript per iteration, and a 0 here would leave every partition reading the first one's data.
    #[test]
    fn an_unscaled_dimension_moves_by_one_per_iteration() {
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0),
                AffineExpr::dim(0).modulo(4),
                AffineExpr::dim(0).floordiv(4),
                AffineExpr::dim(0).plus(AffineExpr::Const(7)),
            ],
        };
        let coeffs = calculate_subscripts_coefficients(&[split_at(0)], &[1], &subscripts);

        assert_eq!(
            coeffs.rows().map(|rows| rows[0].coeffs.as_slice()),
            Some(&[1, 1, 1, 1][..]),
            "involved, and never multiplied"
        );
    }

    /// 🎯 188/384 — ⛔ A SCALING OF **ANOTHER** DIMENSION IS NOT THIS DIMENSION'S COEFFICIENT.
    ///
    /// `bin_expr.isFunctionOfDim(dim)` guards the override (`Dialect/Agen/Utils.cpp:534`), so the
    /// `Mul` in `d0 + d1 * 8` is walked past when the question is about `d0` — leaving the 1 — and
    /// answers 8 when it is about `d1`.
    #[test]
    fn a_scaling_of_another_dimension_is_walked_past() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(0).plus(AffineExpr::dim(1).times(8))],
        };

        assert_eq!(
            calculate_subscripts_coefficients(&[split_at(0)], &[1], &subscripts)
                .rows()
                .map(|rows| rows[0].coeffs.as_slice()),
            Some(&[1][..]),
            "`d0` is involved and unscaled"
        );
        assert_eq!(
            calculate_subscripts_coefficients(&[split_at(1)], &[1], &subscripts)
                .rows()
                .map(|rows| rows[0].coeffs.as_slice()),
            Some(&[8][..]),
            "`d1` is the one being scaled"
        );
    }

    /// 🎯 188/384 — ⛔⛔ AND THE WALK STOPS AT THE **FIRST** `Mul` IT REACHES, IN POST-ORDER.
    ///
    /// `return WalkResult::interrupt();` (`Dialect/Agen/Utils.cpp:539`) over a post-order visitor
    /// (`llvm-project/mlir/lib/IR/AffineExpr.cpp:58` — left subtree, right subtree, then the node), so
    /// `d0 * 4 + d0 * 8` answers **4**: the left subtree's `Mul` is reached before the right's, and
    /// before the `Add` that holds them.
    ///
    /// ⚠️ NEITHER OPERAND IS THE WHOLE TRUTH HERE — the subscript really does move by 12 per
    /// iteration — and this test pins the reference's answer rather than the arithmetic's. Nothing in
    /// the authority tree's answer keys writes a dimension twice in one subscript; if one ever does,
    /// this is the line that says what the vendor's compiler emits for it.
    #[test]
    fn the_first_scaling_in_post_order_wins() {
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0)
                    .times(4)
                    .plus(AffineExpr::dim(0).times(8)),
            ],
        };
        let coeffs = calculate_subscripts_coefficients(&[split_at(0)], &[1], &subscripts);

        assert_eq!(
            coeffs.rows().map(|rows| rows[0].coeffs.as_slice()),
            Some(&[4][..]),
            "the leftmost `Mul`, not the sum and not the last"
        );
    }

    /// 🎯 188/384 — ⛔⛔ AND A SEMI-AFFINE SCALE IS THE UNCHECKED `cast` — REPORTED, NOT GUESSED AT.
    ///
    /// `cast<AffineConstantExpr>(bin_expr.getRHS()).getValue()` (`Dialect/Agen/Utils.cpp:536-537`)
    /// aborts on `d0 * s0`: legal MLIR, expressible in this island, and a stride no
    /// `new_exprs[r] - (coeff * prev_iters)` can be written with.
    ///
    /// ⭐ IT NAMES THE RESULT, so the caller can point at the subscript rather than the transfer.
    #[test]
    fn a_symbolic_scale_is_not_a_coefficient() {
        let subscripts = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::Mul(Box::new(AffineExpr::dim(0)), Box::new(AffineExpr::sym(0))),
            ],
        };
        let coeffs = calculate_subscripts_coefficients(&[split_at(0)], &[1], &subscripts);

        assert_eq!(
            coeffs,
            SubscriptsCoefficients::CoefficientIsNotConstant { dim: 0, result: 1 }
        );
        assert_eq!(coeffs.rows(), None, "no partition can be written from it");
    }

    /// 🎯 188/384 — ⭐ ONE ROW PER SPLIT DIMENSION, IN `partition_sizes` ORDER, EACH CARRYING ITS OWN
    /// DIMENSION.
    ///
    /// `for (int p = 0, e = partition_sizes.size(); p < e; ++p)` (`:1188`) — the LENGTH of the sizes
    /// chooses how far down the weight-sorted `mas_data` the rows go, and the sizes themselves are
    /// never read. Two sizes over three iterators is two rows, for the two heaviest dimensions.
    #[test]
    fn the_rows_follow_the_partition_sizes_length_and_not_their_values() {
        let subscripts = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).times(16),
                AffineExpr::dim(1).times(8),
                AffineExpr::dim(2).times(2),
            ],
        };
        let mas_data = [split_at(0), split_at(1), split_at(2)];

        assert_eq!(
            calculate_subscripts_coefficients(&mas_data, &[2, 1], &subscripts).rows(),
            Some(
                &[
                    DimCoefficients {
                        dim: 0,
                        coeffs: vec![16, 0, 0],
                    },
                    DimCoefficients {
                        dim: 1,
                        coeffs: vec![0, 8, 0],
                    },
                ][..]
            ),
            "two sizes, the two heaviest dimensions, the third untouched"
        );
        assert_eq!(
            calculate_subscripts_coefficients(&mas_data, &[], &subscripts).rows(),
            Some(&[][..]),
            "nothing split, nothing to shift"
        );
    }

    // ── 189/384 and 190/384 — the time dimensions, and the loops that make them explicit ─────────

    /// THE VENDOR'S OWN TRANSFER FROM `mutable_addr_splitting_time_dims.mlir:126-142`
    /// (`constant_start_addr`), transcribed.
    ///
    /// ⭐ SSA NAMES CHOSEN TO MATCH THE ANSWER KEY'S OWN CAPTURES: `%[[VAL_9]]` is `%arg1`,
    /// `%[[VAL_10]]` is `%arg3`, `%[[VAL_11]]`/`%[[VAL_12]]` are the two views and `%[[VAL_18]]` is the
    /// load iv — so the loop [`create_explicit_time_loops`] mints below is `%13`, which is the key's
    /// `%[[VAL_13]]`.
    ///
    /// ⚠️ THE LEADING `?` IS WRITTEN AS `1`, as in [`answer_key_memref`] — [`MemRef::shape`] has no
    /// dynamic extent.
    ///
    /// ⭐ NEITHER e189 NOR e190 READS THE OP: they read `time_bounds`, `time_offsets`,
    /// `time_addr_map`, `time_order` and `indices`. It is transcribed anyway because e190 MOVES this op
    /// into the loop nest it builds, so the value that nest holds has to be the vendor's transfer.
    fn time_dims_transfer() -> agen::Op {
        let (_, ty) = answer_key_memref();
        // `affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 == 0, d2 == 0)>`
        let transfer_set = IntegerSet {
            dims: 3,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(1),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(2),
                    is_equality: true,
                },
            ],
        };

        agen::Op::CompositeLoadAndStore(Box::new(agen::CompositeTransfer {
            src: Val(11),
            src_indices: vec![Index::Const(0); 3],
            src_ty: ty.clone(),
            dst: Val(12),
            // `dst:%dst_mem_view[0, %arg1 * 16, %arg3 * 8]`
            dst_indices: vec![
                Index::Const(0),
                Index::Strided(vec![(Val(9), 16)], 0),
                Index::Strided(vec![(Val(10), 8)], 0),
            ],
            dst_ty: ty,
            load_iv: Val(18),
            load_iv_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            load_set: transfer_set.clone(),
            load_order: AffineMap::identity(3),
            store_set: transfer_set,
            store_order: AffineMap::identity(3),
            // `time_symbols(%c3)` (`:130`) — `%c3` is the key's own `%[[VAL_1]]`.
            time_symbols: vec![Val(1)],
            time_set: time_dims_time_set(),
            time_order: time_dims_time_order(),
            load_time_addr_map: AffineMap::identity(3),
            // `store_time_addr_map = affine_map<(d0, d1, d2) -> (d0*64, d1, d2*8)>`
            store_time_addr_map: AffineMap {
                dims: 3,
                syms: 0,
                results: vec![
                    AffineExpr::dim(0).times(64),
                    AffineExpr::dim(1),
                    AffineExpr::dim(2).times(8),
                ],
            },
            body: vec![DfirOp::Agen(agen::Op::Yield)],
            dir: None,
            multicast_info: None,
            dbg_name: None,
        }))
    }

    /// `time_set = affine_set<(d0, d1, d2)[s0]:(d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0,
    /// d2 >= 0, -d2 + 15 >= 0)>` — `mutable_addr_splitting_time_dims.mlir:136`. `s0` is `%c3`.
    fn time_dims_time_set() -> IntegerSet {
        let spanning = |dim: u32, upper: AffineExpr| {
            vec![
                Constraint {
                    expr: AffineExpr::dim(dim),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(dim).times(-1).plus(upper),
                    is_equality: false,
                },
            ]
        };
        IntegerSet {
            dims: 3,
            symbols: 1,
            constraints: spanning(0, AffineExpr::sym(0).plus(AffineExpr::Const(-1)))
                .into_iter()
                .chain(spanning(1, AffineExpr::sym(0).plus(AffineExpr::Const(-1))))
                .chain(spanning(2, AffineExpr::Const(15)))
                .collect(),
        }
    }

    /// `time_order = affine_map<(d0, d1, d2) -> (d2, d1, d0)>` — a REVERSAL, which is what makes this
    /// key the one that shows the indexing defect in
    /// `updateSubscriptsAndIndicesForExplicitTimeLoops`.
    fn time_dims_time_order() -> AffineMap {
        AffineMap {
            dims: 3,
            syms: 0,
            results: vec![AffineExpr::dim(2), AffineExpr::dim(1), AffineExpr::dim(0)],
        }
    }

    /// THE VENDOR'S ACCESS DETAILS FOR THAT TRANSFER'S **dst** OPERAND — the one being split.
    ///
    /// `time_bounds` and `time_offsets` are what `constructTimeStepsInfo` leaves behind, and BOTH are
    /// indexed by `time_order` RESULT POSITION — outermost first
    /// (`dialect_utils/Agen/Utils.cpp:115`, `:254`):
    ///
    /// | `time_order` result | time dim | trip count, from `time_set` | offset, from `layout ∘ store_time_addr_map` |
    /// |---|---|---|---|
    /// | 0 | `d2` | `-d2 + 15 >= 0` → 16 | `d2 * 8` on the layout's `d2` → `8 * 256 = 2048` |
    /// | 1 | `d1` | `-d1 + s0 - 1 >= 0`, `s0 = 3` → 3 | `d1` on the layout's `d1` → `1 * 64 = 64` |
    /// | 2 | `d0` | `-d0 + s0 - 1 >= 0` → 3 | `d0 * 64` on the layout's `d0` → `64 * 1 = 64` |
    ///
    /// ⭐⭐ AND THE VENDOR'S OWN OUTPUT CONFIRMS THE 2048: `%[[VAL_6]] = arith.constant 26624` is the
    /// second partition's view start, and `26624 = 13 * 2048` — the partition size times this offset.
    fn time_dims_details(op: &agen::Op) -> AccessDetailsAffineComposite<'_> {
        let mut ad = AccessDetailsAffineComposite::new(op, DfirUnit::L3su);
        // `dst:...[0, %arg1 * 16, %arg3 * 8]` — two indices, so two non-time dimensions.
        ad.affine.base.indices = vec![Val(9), Val(10)];
        ad.time_bounds = vec![
            TimeBound::Steps(16),
            TimeBound::Steps(3),
            TimeBound::Steps(3),
        ];
        ad.time_offsets = TimeOffsets {
            per_dim: vec![2048, 64, 64],
            constant: 0,
        };
        ad.time_order = Some(time_dims_time_order());
        ad.time_set = Some(time_dims_time_set());
        ad.time_addr_map = match op {
            agen::Op::CompositeLoadAndStore(transfer) => Some(transfer.store_time_addr_map.clone()),
            _ => None,
        };
        ad
    }

    /// THE `dst` SUBSCRIPTS AND THE TIME SET, AS `transformCompLoadAndStore` READS THEM OFF THE DETAILS
    /// (`:488-490`).
    ///
    /// `dst:%dst_mem_view[0, %arg1 * 16, %arg3 * 8]` is the map `(d0, d1) -> (0, d0 * 16, d1 * 8)` over
    /// the indices `[%arg1, %arg3]`.
    fn time_dims_access() -> SubscriptsAndTime {
        SubscriptsAndTime {
            subscripts_map: AffineMap {
                dims: 2,
                syms: 0,
                results: vec![
                    AffineExpr::Const(0),
                    AffineExpr::dim(0).times(16),
                    AffineExpr::dim(1).times(8),
                ],
            },
            indices: vec![Val(9), Val(10)],
            time_set: time_dims_time_set(),
        }
    }

    /// ONE `mas_data` ENTRY FOR AN EXPLICIT LOOP — the five-argument constructor (`:127-135`), which
    /// is what makes it NOT a time dimension as far as [`create_explicit_time_loops`] is concerned.
    fn loop_iterator(iter_arg: Val, dim: u32, composed_coeff: i64, num_iters: i64) -> MasData {
        MasData {
            iter_arg: Some(iter_arg),
            dim,
            composed_coeff,
            num_iters,
            weight: if num_iters < 2 {
                0
            } else {
                (num_iters - 2) * composed_coeff
            },
        }
    }

    /// `initMASData`'S OWN TWO ENTRIES FOR THE `time_dims` KEY — entry 250 is not in this worklist, so
    /// they are derived here from the input's two `affine.for`s and the shared layout map.
    ///
    /// | loop | subscript | layout dim | `composed_coeff` | `num_iters` | weight |
    /// |---|---|---|---|---|---|
    /// | `%arg1 = 0 to 4` | `%arg1 * 16` on `d1` | `d1 * 64` | `16 * 64 = 1024` | 4 | `(4-2) * 1024 = 2048` |
    /// | `%arg3 = 0 to 8` | `%arg3 * 8` on `d2` | `d2 * 256` | `8 * 256 = 2048` | 8 | `(8-2) * 2048 = 12288` |
    ///
    /// ⭐ CHECKABLE DOWNSTREAM RATHER THAN ASSERTED: with the three time weights this batch adds, the
    /// total is what makes the vendor's `arith.constant 13` and `arith.constant 26624` come out.
    fn time_dims_loop_iterators() -> Vec<MasData> {
        vec![
            loop_iterator(Val(9), 0, 1024, 4),
            loop_iterator(Val(10), 1, 2048, 8),
        ]
    }

    /// 🎯 189/384 — ⭐⭐ IBM'S `time_dims` KEY: THREE TIME DIMENSIONS BECOME THREE MORE ITERATORS AND
    /// THE SPAN REACHES **43136**.
    ///
    /// `mutable_addr_splitting_time_dims.mlir`, run with
    /// `--dcc-mutable-addr-splitting-max-mutable-size=600000` — 37500 `f16` elements.
    ///
    /// | dimension | source | `composed_coeff` | `num_iters` | weight |
    /// |---|---|---|---|---|
    /// | 0 | `%arg1`, explicit | 1024 | 4 | 2048 |
    /// | 1 | `%arg3`, explicit | 2048 | 8 | 12288 |
    /// | 2 | time 0, `d2` | 2048 | 16 | `(16-2) * 2048 = 28672` |
    /// | 3 | time 1, `d1` | 64 | 3 | `(3-2) * 64 = 64` |
    /// | 4 | time 2, `d0` | 64 | 3 | 64 |
    ///
    /// `max_mutable = 2048 + 12288 + 28672 + 64 + 64 = 43136`, and `43136 - 37500 = 5636` is the
    /// overflow the split is computed against — see
    /// [`the_time_dims_key_splits_the_outermost_time_dimension_into_partitions_of_thirteen`].
    ///
    /// ⭐ THE DIMENSION NUMBERS CONTINUE THE INDICES': two indices, so the time dimensions are 2, 3
    /// and 4.
    #[test]
    fn the_time_dims_key_appends_three_time_iterators_and_their_weights() {
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        let mut mas_data = time_dims_loop_iterators();
        let mut max_mutable = MutableAddr(0);
        // `initMASData` reached 2048 + 12288 before this function runs.
        for data in &mas_data {
            max_mutable.add_weight(data.weight);
        }
        assert_eq!(max_mutable, MutableAddr(14_336));

        synthesize_time_info(&mut mas_data, &ad, &mut max_mutable);

        assert_eq!(
            &mas_data[2..],
            &[
                MasData {
                    iter_arg: None,
                    dim: 2,
                    composed_coeff: 2048,
                    num_iters: 16,
                    weight: 28_672,
                },
                MasData {
                    iter_arg: None,
                    dim: 3,
                    composed_coeff: 64,
                    num_iters: 3,
                    weight: 64,
                },
                MasData {
                    iter_arg: None,
                    dim: 4,
                    composed_coeff: 64,
                    num_iters: 3,
                    weight: 64,
                },
            ]
        );
        assert_eq!(max_mutable, MutableAddr(43_136));
        // 600000 bits of `f16` is 37500 elements, and this is the vendor's own overflow.
        assert_eq!(
            max_mutable.beyond(AddrRange::of_bits(600_000).elements(DataType::Sen169Fp16)),
            5636
        );
    }

    /// 🎯 189/384 — ⛔ A DIMENSION OF FEWER THAN TWO STEPS WEIGHS **NOTHING**, NOT ONE OFFSET.
    ///
    /// `time_bounds[i] < 2 ? 0 : (time_bounds[i] - 2) * time_offsets[i]` — the reference's own reason
    /// is that the last iteration of any loop may overflow the mutable address because no transfer
    /// follows it, so `num_iters - 2` is the last address actually used. ⭐ The ENTRY still appears,
    /// with its coefficient and trip count intact; only the weight is zero.
    #[test]
    fn a_time_dimension_of_one_step_appends_an_iterator_that_weighs_nothing() {
        let op = time_dims_transfer();
        let mut ad = time_dims_details(&op);
        ad.time_bounds = vec![TimeBound::Steps(1), TimeBound::Steps(2)];
        ad.time_offsets = TimeOffsets {
            per_dim: vec![512, 512],
            constant: 0,
        };
        let mut mas_data = Vec::new();
        let mut max_mutable = MutableAddr(7);

        synthesize_time_info(&mut mas_data, &ad, &mut max_mutable);

        assert_eq!(
            mas_data.iter().map(|data| data.weight).collect::<Vec<_>>(),
            vec![0, 0],
            "one step weighs 0, and two steps weigh (2-2) * 512 = 0 as well"
        );
        assert_eq!(mas_data[0].num_iters, 1);
        assert_eq!(mas_data[0].composed_coeff, 512);
        assert_eq!(max_mutable, MutableAddr(7), "the span is untouched");
    }

    /// 🎯 189/384 — ⛔ A SENTINEL BOUND IS TRANSCRIBED AS THE REFERENCE'S OWN INTEGER AND WEIGHS ZERO.
    ///
    /// `time_bounds` is a vector of `int64_t` in the C++ and carries `kCoalesced = -2` and
    /// `kInvalid = -1` (`AccessDetails.hpp:252-255`) straight into `num_iters_`. Both are below 2, so
    /// both take the `? 0` arm and neither moves `max_mutable` — which is why the reference can store
    /// them without checking. ⭐ [`create_explicit_time_loops`] is where a sentinel is caught, because
    /// it needs a LOOP BOUND rather than a weight.
    #[test]
    fn a_sentinel_time_bound_becomes_a_negative_trip_count_that_weighs_nothing() {
        let op = time_dims_transfer();
        let mut ad = time_dims_details(&op);
        ad.time_bounds = vec![TimeBound::Coalesced, TimeBound::Variable];
        ad.time_offsets = TimeOffsets {
            per_dim: vec![2048, 2048],
            constant: 0,
        };
        let mut mas_data = Vec::new();
        let mut max_mutable = MutableAddr(100);

        synthesize_time_info(&mut mas_data, &ad, &mut max_mutable);

        assert_eq!(
            mas_data
                .iter()
                .map(|data| (data.num_iters, data.weight))
                .collect::<Vec<_>>(),
            vec![(-2, 0), (-1, 0)],
            "`kCoalesced` and `kInvalid`, as the reference stores them"
        );
        assert_eq!(max_mutable, MutableAddr(100));
    }

    /// 🎯 189/384 — ⛔ THE TRAILING `time_offsets` ENTRY IS THE CONSTANT COEFFICIENT AND IS IGNORED.
    ///
    /// `DT_CHECK(time_offsets.size() >= time_bounds.size())` guards a `time_offsets[i]` indexed by a
    /// `time_bounds` position, under the reference's own note that the extra entry *"isn't used for
    /// anything currently so it is ignored"*. [`TimeOffsets`](super::agen_access_details::TimeOffsets)
    /// names it `constant` instead of leaving it on the end of the vector, so this is a fact about the
    /// type — and a `per_dim` SHORTER than `time_bounds` stops where the failed `DT_CHECK` would have.
    #[test]
    fn the_constant_time_offset_is_ignored_and_a_short_list_stops_the_walk() {
        let op = time_dims_transfer();
        let mut ad = time_dims_details(&op);
        ad.time_offsets = TimeOffsets {
            per_dim: vec![2048, 64],
            constant: 999_999,
        };
        let mut mas_data = Vec::new();
        let mut max_mutable = MutableAddr(0);

        synthesize_time_info(&mut mas_data, &ad, &mut max_mutable);

        assert_eq!(
            mas_data.len(),
            2,
            "three bounds, two offsets — the third dimension has no coefficient to charge"
        );
        assert_eq!(max_mutable, MutableAddr(28_672 + 64));
    }

    /// 🎯 186/384, 189/384 — ⭐⭐ AND THE VENDOR'S `arith.constant 13` AND `arith.constant 26624` COME
    /// OUT OF THAT SPAN.
    ///
    /// The overflow [`the_time_dims_key_appends_three_time_iterators_and_their_weights`] reaches is
    /// 5636 elements. The heaviest dimension is time dimension 0 at `composed_coeff = 2048` over 16
    /// iterations, so `extra_iters = ceil(5636 / 2048) = 3` and the partition size is `16 - 3 = 13` —
    /// which is the constant the key compares the new loop's induction variable against
    /// (`mutable_addr_splitting_time_dims.mlir:34-35`). `shifted_mutable = 13 * 2048 = 26624` is the
    /// second partition's view start address (`:26`, `:44`).
    ///
    /// ⚠️ THE OVERRIDE IS RE-EXPRESSED AGAINST THE COMPILED-IN RANGE — see [`span_overflowing_by`].
    #[test]
    fn the_time_dims_key_splits_the_outermost_time_dimension_into_partitions_of_thirteen() {
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(3),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        let mut mas_data = time_dims_loop_iterators();
        let mut max_mutable = MutableAddr(14_336);
        synthesize_time_info(&mut mas_data, &ad, &mut max_mutable);
        sort_data_based_on_weight(&mut mas_data);

        // Heaviest first: time dim 0, then `%arg3`, then `%arg1`, then the two 64s in dimension order.
        assert_eq!(
            mas_data.iter().map(|data| data.dim).collect::<Vec<_>>(),
            vec![2, 1, 0, 3, 4]
        );

        let mut conditionals = Conditionals::default();
        let split = calculate_partition_sizes::<Dd2>(
            &mas_data,
            L3Half::Store,
            &view,
            span_overflowing_by::<Dd2>(5636, DataType::Sen169Fp16),
            DataType::Sen169Fp16,
            &mut conditionals,
        );

        assert_eq!(
            split,
            Partitioning::Split(PartitionSizes {
                sizes: vec![13],
                total_partitions: 2,
                shifted_mutable: 26_624,
                // `5636 - 3 * 2048 = -508`.
                remaining_overflow: -508,
            })
        );
        // One `scf.if`, which is what the answer key contains.
        assert_eq!(conditionals, Conditionals(1));
    }

    /// 🎯 190/384 — ⭐⭐ IBM'S `time_dims` KEY, WHOLE: ONE `affine.for %13 = 0 to 16` AROUND THE
    /// TRANSFER, THE TIME SET PINNED TO `d2 == 0`, AND THE SPLIT DIMENSION'S ITERATOR FILLED IN.
    ///
    /// `mutable_addr_splitting_time_dims.mlir` — `affine.for %[[VAL_13]] = 0 to 16` (`:33`) and
    /// `#[[$ATTR_9]] = affine_set<(d0, d1, d2)[s0] : (d2 == 0, d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0,
    /// -d1 + s0 - 1 >= 0)>` (`:14`), which is the `time_set` both partitions carry.
    ///
    /// | step | value |
    /// |---|---|
    /// | `innermost_time_dim` | 2 — `mas_data[0]`, the only split entry, and it has no iterator |
    /// | `num_non_time_dims` | 2 — `%arg1` and `%arg3` |
    /// | `time_dim_idx` | `2 - 2 = 0`, so ONE loop, bound `time_bounds[0] = 16` |
    /// | `subscripts_map_time` | `(d0, d1, d2, d3, d4) -> (d2 * 64, d0 * 16 + d3, d1 * 8 + d4 * 8)` |
    /// | rewritten subscripts | `(d0, d1, d2) -> (d2 * 64, d0 * 16, d1 * 8)` |
    /// | rewritten indices | `[%arg1, %arg3, %13]` |
    ///
    /// ⛔ THE REWRITTEN SUBSCRIPTS MAP NEVER REACHES THE OUTPUT, WHICH IS WHY THE KEY STILL PRINTS
    /// `dst:%[[VAL_16]][0, %[[VAL_9]] * 16, %[[VAL_10]] * 8]` — the two-dimensional map — beside an
    /// indices list that has grown a third entry. See [`TimeLoopNest::access`].
    #[test]
    fn the_time_dims_key_wraps_the_transfer_in_one_explicit_time_loop() {
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        let mut mas_data = time_dims_loop_iterators();
        let mut max_mutable = MutableAddr(14_336);
        synthesize_time_info(&mut mas_data, &ad, &mut max_mutable);
        sort_data_based_on_weight(&mut mas_data);

        let mut vals = Values::default();
        // The thirteen values the key names before the loop, so the loop's iv is `%13`.
        for _ in 0..13 {
            vals.mint();
        }

        let built = create_explicit_time_loops(
            &mut vals,
            &mut mas_data,
            &[13],
            DfirOp::Agen(time_dims_transfer()),
            &ad,
            &time_dims_access(),
        );

        let nest = built.nest().expect("one time dimension is split");
        assert_eq!(nest.ivs, vec![Val(13)]);

        // `affine.for %[[VAL_13]] = 0 to 16 { <the transfer> }`
        let mut out = String::new();
        print::emit(&mut out, &nest.nest, 0);
        assert_eq!(
            out.lines().next(),
            Some("affine.for %13 = 0 to 16 {"),
            "one loop, the outermost time dimension's own trip count"
        );
        assert!(
            out.contains("agen.composite_load_and_store"),
            "and the transfer is inside it"
        );

        // `#[[$ATTR_9]]`, byte for byte.
        assert_eq!(
            print::integer_set(&nest.access.time_set),
            "affine_set<(d0, d1, d2)[s0] : (d2 == 0, d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, \
             -d1 + s0 - 1 >= 0)>"
        );
        assert_eq!(
            print::affine_map(&nest.access.subscripts_map),
            "affine_map<(d0, d1, d2) -> (d2 * 64, d0 * 16, d1 * 8)>"
        );
        assert_eq!(nest.access.indices, vec![Val(9), Val(10), Val(13)]);

        // The split dimension now has an iterator; the two inner time dimensions still do not.
        assert_eq!(
            mas_data
                .iter()
                .map(|data| (data.dim, data.iter_arg))
                .collect::<Vec<_>>(),
            vec![
                (2, Some(Val(13))),
                (1, Some(Val(10))),
                (0, Some(Val(9))),
                (3, None),
                (4, None),
            ]
        );
    }

    /// 🎯 190/384 — ⭐ THE CONCATENATED MAP SPANS BOTH DIMENSION LISTS, AND MLIR'S OWN `+` FOLDS THE
    /// ZERO SUBSCRIPT AWAY.
    ///
    /// `concatenateMaps` (`dialect_utils/Agen/Utils.cpp:258`) shifts the time address map above the
    /// subscripts' dimensions and adds the two result-wise, so the vendor's `(0, d0 * 16, d1 * 8)` and
    /// `(d0 * 64, d1, d2 * 8)` become one five-dimensional map — with `0 + d2 * 64` printed as
    /// `d2 * 64`, which is [`AffineExpr::sum`] and not the verbatim builder.
    #[test]
    fn the_subscripts_and_the_time_address_map_concatenate_into_one() {
        let access = time_dims_access();
        let time_addr_map = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).times(64),
                AffineExpr::dim(1),
                AffineExpr::dim(2).times(8),
            ],
        };

        assert_eq!(
            print::affine_map(&concatenate_maps(&access.subscripts_map, &time_addr_map)),
            "affine_map<(d0, d1, d2, d3, d4) -> (d2 * 64, d0 * 16 + d3, d1 * 8 + d4 * 8)>"
        );
    }

    /// 🎯 190/384 — ⭐ NO SPLIT TIME DIMENSION MEANS THE TRANSFER STAYS EXACTLY WHERE IT WAS.
    ///
    /// `if (innermost_time_dim < 0) return;` (`:1301`) — the common case, and the one four of the
    /// pass's five answer keys take. The op comes back out because it went in by value.
    #[test]
    fn a_split_on_an_explicit_loop_alone_creates_no_time_loops() {
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        let mut mas_data = time_dims_loop_iterators();
        let mut vals = Values::default();

        let built = create_explicit_time_loops(
            &mut vals,
            &mut mas_data,
            &[4],
            DfirOp::Agen(time_dims_transfer()),
            &ad,
            &time_dims_access(),
        );

        assert_eq!(
            built,
            ExplicitTimeLoops::NotSplitOnATimeDimension(DfirOp::Agen(time_dims_transfer()))
        );
        assert_eq!(built.nest(), None);
        assert_eq!(mas_data, time_dims_loop_iterators(), "nothing was touched");
        assert_eq!(vals.issued(), 0, "and no name was minted");
    }

    /// 🎯 190/384 — ⛔⛔ A SENTINEL TIME BOUND CANNOT BE A LOOP BOUND, AND THE REPORT LEAVES
    /// `mas_data` ALONE.
    ///
    /// `AffineForOp::create(builder, loc, 0, time_bounds[dim])` (`Agen/Utils.cpp:430-431`) with
    /// `kCoalesced = -2` builds `affine.for %i = 0 to -2` — legal MLIR that runs zero times, so the
    /// transfer moved into it would never happen. ⭐ [`synthesize_time_info`] stores the same value
    /// happily, because a weight of 0 is a correct answer for it.
    #[test]
    fn a_coalesced_time_bound_is_not_a_loop_bound() {
        let op = time_dims_transfer();
        let mut ad = time_dims_details(&op);
        ad.time_bounds = vec![
            TimeBound::Coalesced,
            TimeBound::Steps(3),
            TimeBound::Steps(3),
        ];
        let mut mas_data = time_dims_loop_iterators();
        mas_data.push(MasData {
            iter_arg: None,
            dim: 2,
            composed_coeff: 2048,
            num_iters: -2,
            weight: 0,
        });
        let before = mas_data.clone();

        assert_eq!(
            create_explicit_time_loops(
                &mut Values::default(),
                &mut mas_data,
                &[1, 1, 1],
                DfirOp::Agen(time_dims_transfer()),
                &ad,
                &time_dims_access(),
            ),
            ExplicitTimeLoops::TimeBoundIsNotALoopTripCount {
                time_dim: 0,
                bound: TimeBound::Coalesced,
            }
        );
        assert_eq!(mas_data, before);
    }

    /// 🎯 190/384 — ⛔ MORE PARTITION SIZES THAN ITERATORS IS `mas_data[i]` READ PAST THE END.
    ///
    /// `for (int i = 0, e = partition_sizes.size(); i < e; ++i) { .. mas_data[i] .. }` (`:1294-1297`),
    /// the same pairing [`ConditionalTree::MorePartitionsThanIterators`] guards one function later.
    #[test]
    fn more_partition_sizes_than_iterators_is_reported() {
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        let mut mas_data = time_dims_loop_iterators();

        assert_eq!(
            create_explicit_time_loops(
                &mut Values::default(),
                &mut mas_data,
                &[4, 8, 16],
                DfirOp::Agen(time_dims_transfer()),
                &ad,
                &time_dims_access(),
            ),
            ExplicitTimeLoops::MorePartitionsThanIterators {
                sizes: 3,
                iterators: 2,
            }
        );
    }

    /// 🎯 190/384 — ⛔⛔ TWO LOOPS EXPOSE THE INDEXING DEFECT: `for_ops` IS BUILT IN `time_order`
    /// ORDER AND READ BY RAW TIME-DIMENSION INDEX.
    ///
    /// With `time_dim_idx = 1` the reference creates `for_ops[0] = 0 to time_bounds[0]` and
    /// `for_ops[1] = 0 to time_bounds[1]` — outermost first, i.e. in `time_order` RESULT order, which
    /// for this key is `(d2, d1, d0)`. Then
    /// `updateSubscriptsAndIndicesForExplicitTimeLoops` binds the time-address dimension `d0` to
    /// `for_ops[2 - 2] = for_ops[0]` and `d1` to `for_ops[1]` (`Agen/Utils.cpp:472`) — by the
    /// TRANSFER's own dimension numbering.
    ///
    /// ⛔ SO `d0`, whose trip count is 3, is bound to the loop of bound **16**, and `d1`, whose trip
    /// count is 3, to the loop of bound 3. The single-loop case above cannot show this because
    /// `for_ops[0]` is the only entry there is. ⭐ PORTED AS WRITTEN — this is the reference's
    /// numbering, and permuting here would emit a different `indices` list from `dcc`.
    #[test]
    fn the_for_ops_are_indexed_by_the_raw_time_dimension_and_not_by_time_order() {
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        // The split reaches time dimension 1, so `innermost_time_dim` is `2 + 1 = 3`.
        let mut mas_data = vec![
            MasData {
                iter_arg: None,
                dim: 3,
                composed_coeff: 64,
                num_iters: 3,
                weight: 64,
            },
            MasData {
                iter_arg: None,
                dim: 2,
                composed_coeff: 2048,
                num_iters: 16,
                weight: 28_672,
            },
        ];
        let mut vals = Values::default();

        let built = create_explicit_time_loops(
            &mut vals,
            &mut mas_data,
            &[1, 1],
            DfirOp::Agen(time_dims_transfer()),
            &ad,
            &time_dims_access(),
        );
        let nest = built.nest().expect("two time dimensions are split");

        let mut out = String::new();
        print::emit(&mut out, &nest.nest, 0);
        let bounds: Vec<&str> = out
            .lines()
            .filter(|line| line.contains("affine.for"))
            .collect();
        assert_eq!(
            bounds,
            vec!["affine.for %0 = 0 to 16 {", "  affine.for %1 = 0 to 3 {"],
            "outermost first, so `time_bounds[0]` is the outer loop"
        );

        // `d0` — the time-address dimension whose own trip count is 3 — takes `for_ops[0]`, the loop
        // of bound 16. `d1` takes `for_ops[1]`, the loop of bound 3.
        assert_eq!(nest.access.indices, vec![Val(9), Val(10), Val(0), Val(1)]);
        assert_eq!(
            print::affine_map(&nest.access.subscripts_map),
            "affine_map<(d0, d1, d2, d3) -> (d2 * 64, d0 * 16 + d3, d1 * 8)>"
        );
        // Both explicit dimensions are pinned, in `time_order` result order: `d2` then `d1`.
        assert_eq!(
            print::integer_set(&nest.access.time_set),
            "affine_set<(d0, d1, d2)[s0] : (d2 == 0, d1 == 0, d0 >= 0, -d0 + s0 - 1 >= 0)>"
        );
    }

    /// 🎯 190/384 — ⛔ A `time_order` RESULT THAT IS NOT A BARE DIMENSION IS THE UNCHECKED `cast`.
    ///
    /// `time_order.getDimPosition(dim)` is `cast<AffineDimExpr>(getResult(dim)).getPosition()`
    /// (`Agen/Utils.cpp:497`) — not `dyn_cast`, so a non-permutation aborts rather than answering.
    #[test]
    fn a_time_order_that_is_not_a_permutation_is_reported() {
        let op = time_dims_transfer();
        let mut ad = time_dims_details(&op);
        ad.time_order = Some(AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2).times(2),
                AffineExpr::dim(1),
                AffineExpr::dim(0),
            ],
        });
        let mut mas_data = vec![MasData {
            iter_arg: None,
            dim: 2,
            composed_coeff: 2048,
            num_iters: 16,
            weight: 28_672,
        }];

        assert_eq!(
            create_explicit_time_loops(
                &mut Values::default(),
                &mut mas_data,
                &[13],
                DfirOp::Agen(time_dims_transfer()),
                &ad,
                &time_dims_access(),
            ),
            ExplicitTimeLoops::TimeOrderIsNotAPermutation { time_dim: 0 }
        );
    }

    /// 🎯 190/384 — ⛔ A SPLIT DIMENSION WITH NO ITERATOR THAT IS NOT A TIME DIMENSION EITHER MAKES
    /// `time_dim_idx` NEGATIVE.
    ///
    /// `DT_CHECK(time_dim >= 0 && ..)` (`Agen/Utils.cpp:425`). Every entry `initMASData` pushes carries
    /// its loop's induction variable and every entry [`synthesize_time_info`] pushes is numbered at or
    /// above `indices.len()`, so this is a `mas_data` neither of them built.
    #[test]
    fn a_non_time_dimension_without_an_iterator_is_reported() {
        let op = time_dims_transfer();
        let ad = time_dims_details(&op);
        let mut mas_data = vec![MasData {
            iter_arg: None,
            dim: 1,
            composed_coeff: 2048,
            num_iters: 8,
            weight: 12_288,
        }];

        assert_eq!(
            create_explicit_time_loops(
                &mut Values::default(),
                &mut mas_data,
                &[4],
                DfirOp::Agen(time_dims_transfer()),
                &ad,
                &time_dims_access(),
            ),
            ExplicitTimeLoops::NonTimeDimensionHasNoIterator { dim: 1 }
        );
    }
    /// 🎯 250/384 — ⭐⭐ IBM'S `one_dim` ANSWER KEY, FROM THE OTHER END: THE `max_mutable` OF **12672**
    /// THAT ENTRY 186 IS HANDED.
    ///
    /// `constant_start_addr_1` (`mutable_addr_splitting_one_dim.mlir:239-263`) —
    /// `affine.for %arg1 = 0 to 4` around `scf.for %arg2 = %c0 to %719 step %c1` (`maxValue = 8`),
    /// subscripts `[0, %arg1 * 3, %arg2 * 8]` over `d2 * 256 + d1 * 64 + d0`, so coefficients 192 and
    /// 2048. ⛔ WEIGHT IS `num_iters - 2`, NOT `- 1`: `(4-2)*192 = 384` and `(8-2)*2048 = 12288`, and
    /// `0 + 384 + 12288` is the span [`the_one_dim_answer_key_splits_the_symbolic_dim_into_partitions_of_four`]
    /// splits. The `agen::Op` is not read by this function — only the indices and their coefficients.
    #[test]
    fn the_one_dim_answer_key_weighs_both_loops_at_num_iters_less_two() {
        let mut vals = Values::default();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let hi = vals.mint();
        let arg2 = vals.mint();
        let arg1 = vals.mint();
        let scope = vec![
            lo_op,
            step_op,
            DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: hi,
                symbol_id: -1476,
                max_value: Some(8),
            }),
            DfirOp::Affine(affine::Op::For {
                iv: arg1,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: vec![DfirOp::Scf(scf::Op::For {
                    iv: arg2,
                    lo,
                    hi,
                    step,
                    carried: Vec::new(),
                    body: Vec::new(),
                    dbg_name: None,
                })],
                dbg_name: None,
            }),
        ];

        let op = time_dims_transfer();
        let mut ad = AccessDetailsAffine::new(&op, DfirUnit::L3lu);
        ad.base.indices = vec![arg1, arg2];
        ad.indices_coeff_dict = IndicesCoeffDict {
            per_index: vec![(arg1, 192), (arg2, 2048)],
            constant: 0,
        };

        assert_eq!(
            init_mas_data(&ad, &scope),
            MasDataInit::Initialized {
                mas_data: vec![
                    MasData {
                        iter_arg: Some(arg1),
                        dim: 0,
                        composed_coeff: 192,
                        num_iters: 4,
                        weight: 384,
                    },
                    MasData {
                        iter_arg: Some(arg2),
                        dim: 1,
                        composed_coeff: 2048,
                        num_iters: 8,
                        weight: 12_288,
                    },
                ],
                max_mutable: MutableAddr(12_672),
            }
        );
    }
    /// 🎯 251/384 — ⛔ THE MIDDLE STATEMENT IS LOAD-BEARING: THE `one_dim` KEY IS UNREACHABLE FROM
    /// `initMASData` ORDER.
    ///
    /// The same case fed in the order entry 250 builds it — 384 before 12288 — cuts the LIGHT
    /// dimension to 1 first and then sizes the heavy one against a reduced overflow, `[1, 4]`.
    /// `sortDataBasedOnWeight` (`:827`) is what turns it back into the vendor's single
    /// `arith.constant 4`.
    #[test]
    fn the_setup_sorts_before_it_sizes_and_the_one_dim_key_needs_that() {
        let mut vals = Values::default();
        let start = vals.mint();
        let (scope, mem_view) = view_with_start(
            &mut vals,
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 3072,
            }),
        );
        let (layout, ty) = answer_key_memref();
        let view = ConstStartMemView {
            from: Val(0),
            start: 3072,
            layout: &layout,
            ty: &ty,
        };
        let mut mas_data = [iterator(0, 192, 4, 384), iterator(1, 2048, 8, 12_288)];
        let mut conditionals = Conditionals::default();

        assert_eq!(
            setup_for_partitioning::<Dd2>(
                &mut mas_data,
                &[mem_view],
                L3Half::Load,
                &view,
                span_overflowing_by::<Dd2>(6972, DataType::Sen169Fp16),
                DataType::Sen169Fp16,
                &mut conditionals,
                &scope,
            ),
            SetupForPartitioning::Sized(Partitioning::Split(PartitionSizes {
                sizes: vec![4],
                total_partitions: 2,
                shifted_mutable: 8192,
                remaining_overflow: -1220,
            }))
        );
        // The sort ran in place, so `partition_sizes[0]` belongs to the heavy dimension.
        assert_eq!(mas_data[0].dim, 1);
    }
    /// 🎯 307/384 — ⭐⭐ IBM'S `constant_start_addr_2` KEY END TO END: ONE `agen.vector_store`
    /// BECOMES TWO PARTITIONS, EACH WITH ITS OWN CLONE OF THE **LOAD'S** VIEW, WHICH THE CLONED LOAD
    /// READS.
    ///
    /// `mutable_addr_splitting_one_dim.mlir:267-288` in, `:49-82` out — the cut at `%arg2 < 6`, the
    /// `else` arm's `* 2 - 4`, `dbgName = ""` on the rebuilt store and none on the cloned load, and
    /// the per-arm order view/load-view/load/store. ⛔ THE LAYOUT IS THE VENDOR'S TRANSPOSED AND
    /// SCALED BY 24000, for the two reasons the 306/384 test records; `4 + 6528 * 24000` overflows the
    /// compiled EAR by 22454276, which two of `%arg2`'s eight iterations cover and one does not.
    #[test]
    fn the_one_dim_answer_keys_store_becomes_two_partitions_each_with_its_own_load_view() {
        const K: i64 = 24000;
        let mut vals = Values::default();
        let hbm = vals.mint();
        let lx = vals.mint();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (start_op, start) = index_const(&mut vals, 2048);
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let src_view = vals.mint();
        let dst_view = vals.mint();
        let data = vals.mint();

        let layout = AffineMap::linear(&[256 * K, 64 * K, 1]);
        let ty = MemRef {
            shape: vec![4, 64, 1],
            elem: ElemType::F16,
        };
        let vec_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let view = |result: Val, from: Val, start: Val| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                layout: layout.clone(),
                ty: ty.clone(),
            })
        };
        let scope = vec![
            lo_op,
            start_op,
            // `affine.for %arg1 = 0 to 4 { affine.for %arg2 = 0 to 8 { .. } }`.
            DfirOp::Affine(affine::Op::For {
                iv: arg1,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: vec![DfirOp::Affine(affine::Op::For {
                    iv: arg2,
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Const(8),
                    carried: Vec::new(),
                    body: Vec::new(),
                    dbg_name: None,
                })],
                dbg_name: None,
            }),
            view(src_view, lx, lo),
            view(dst_view, hbm, start),
            // `%data = agen.vector_load %src_mem_view[0, 0, 0]`.
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: data,
                view: src_view,
                indices: vec![Index::Const(0); 3],
                view_ty: ty.clone(),
                ty: vec_ty,
                multicast_info: None,
            }),
            // `agen.vector_store %data, %dst_mem_view[%c4, %arg1 * 3 + %c16, %arg2 * 2 + 8]`,
            // transposed.
            DfirOp::Agen(agen::Op::VectorStore {
                dbg_name: None,
                value: data,
                view: dst_view,
                indices: vec![
                    Index::Strided(vec![(arg2, 2)], 8),
                    Index::Strided(vec![(arg1, 3)], 16),
                    Index::Const(4),
                ],
                view_ty: ty.clone(),
                ty: vec_ty,
            }),
        ];

        let candidate = MasCandidate {
            op: &scope[6],
            comp: DfirUnit::L3su,
            mem_index: MemoryOperandIndex::DirSrc,
        };
        let mut conditionals = Conditionals::default();
        let split = transform_vector_store::<Dd2>(
            &mut vals,
            &candidate,
            DataType::Sen169Fp16,
            EarOverflowCorrection::Allowed,
            &mut conditionals,
            &scope,
        );
        let TransferSplit::Split {
            ops,
            hoisted,
            erased,
            partitions,
            adjustments,
        } = split
        else {
            panic!("the key overflows the EAR by 22454276 elements: {split:?}")
        };
        assert_eq!(partitions, 2);
        assert_eq!(
            adjustments,
            [
                EvenImmutableAdjustment::NotOnThisArch,
                EvenImmutableAdjustment::NotOnThisArch
            ]
        );
        // `op.eraseOpAndUseChain()` walks a producer-ward chain FORWARDS, which is the store before
        // the load that feeds it — the same consumer-first order the load's reversed walk produces.
        assert_eq!(erased, [&scope[6], &scope[5]]);
        // The `else` partition is filled first, so its constant is minted first; both sit at
        // function scope, as the vendor's `%c5120` and `%c2048` do.
        assert_eq!(
            printed(&hoisted),
            concat!(
                "%11 = arith.constant 73730048 : index\n",
                "%15 = arith.constant 2048 : index\n",
            ),
            "2048 + 6 * 12288000, and the vendor's own 2048 + 6 * 512 = 5120"
        );
        let arm = |view: &str, start: &str, load_view: &str, load: &str, subscript: &str| {
            format!(
                concat!(
                    "  {view} = dataflow.get_logical_memory_view %0, {start} {{layout_map = #MAP}} : index, index, memref<4x64x1xf16>\n",
                    "  {load_view} = dataflow.get_logical_memory_view %1, %2 {{layout_map = #MAP}} : index, index, memref<4x64x1xf16>\n",
                    "  {load} = agen.vector_load {load_view}[0, 0, 0] {{load_order = #ORDER, load_set = #SET}} : memref<4x64x1xf16>, vector<64xf16>\n",
                    "  agen.vector_store {load}, {view}[{subscript}, %4 * 3 + 16, 4] {{dbgName = \"\", store_order = #ORDER, store_set = #SET}} : memref<4x64x1xf16>, vector<64xf16>\n",
                ),
                view = view,
                start = start,
                load_view = load_view,
                load = load,
                subscript = subscript,
            )
        };
        let emitted = printed(&ops)
            .replace(
                "affine_map<(d0, d1, d2) -> (d0 * 6144000 + d1 * 1536000 + d2)>",
                "#MAP",
            )
            .replace("affine_map<(d0, d1, d2) -> (d0, d1, d2)>", "#ORDER")
            .replace(
                "affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>",
                "#SET",
            );
        assert_eq!(
            emitted,
            format!(
                concat!(
                    "%9 = arith.constant 6 : index\n",
                    "%10 = arith.cmpi slt, %5, %9 : index\n",
                    "scf.if %10 {{\n{then}}} else {{\n{else_}}}\n",
                ),
                // `agen.vector_store %VAL_19, %VAL_17[4, %VAL_11 * 3 + 16, %VAL_12 * 2 + 8]`.
                then = arm("%16", "%15", "%17", "%18", "%5 * 2 + 8"),
                // `agen.vector_store %VAL_22, %VAL_20[4, %VAL_11 * 3 + 16, %VAL_12 * 2 - 4]`.
                else_ = arm("%12", "%11", "%13", "%14", "%5 * 2 + -4"),
            ),
            "four ops per arm, the load's view cloned into each and read by the cloned load, and \
             `dbgName` on the rebuilt store where the plain clone above it has none"
        );
    }

    /// 🎯 306/384 — ⭐⭐ IBM'S `one_dim` KEY END TO END: ONE `agen.vector_load` BECOMES TWO
    /// PARTITIONS, EACH WITH ITS OWN CLONE OF THE **STORE'S** VIEW.
    ///
    /// `mutable_addr_splitting_one_dim.mlir:240-264` in, `:12-47` out — the cut at `%arg2 < 4`, the
    /// `else` arm's `* 8 - 32`, four ops per arm in the vendor's order, and both start constants
    /// HOISTED out of the unit. ⛔ THE LAYOUT IS THE VENDOR'S TRANSPOSED AND SCALED BY 24576: it runs
    /// `--max-mutable-size=91200`, which is [`MAX_MUTABLE_SIZE`] = [`None`] here (see
    /// [`span_overflowing_by`]), and [`agen::access_set`] derives the lane run on the LAST axis where
    /// the key's `d2 * 256 + d1 * 64 + d0` puts it on `d0`. The start addresses scale; nothing else does.
    #[test]
    fn the_one_dim_answer_keys_load_becomes_two_partitions_each_with_its_own_store_view() {
        const K: i64 = 24576;
        let mut vals = Values::default();
        let hbm = vals.mint();
        let lx = vals.mint();
        let (lo_op, lo) = index_const(&mut vals, 0);
        let (step_op, step) = index_const(&mut vals, 1);
        let (start_op, start) = index_const(&mut vals, 3072);
        let hi = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let src_view = vals.mint();
        let dst_view = vals.mint();
        let data = vals.mint();

        let layout = AffineMap::linear(&[256 * K, 64 * K, 1]);
        let ty = MemRef {
            shape: vec![4, 64, 1],
            elem: ElemType::F16,
        };
        let vec_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let view = |result: Val, from: Val, start: Val| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                layout: layout.clone(),
                ty: ty.clone(),
            })
        };
        let scope = vec![
            lo_op,
            step_op,
            start_op,
            DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: hi,
                symbol_id: -1476,
                max_value: Some(8),
            }),
            // `affine.for %arg1 = 0 to 4 { scf.for %arg2 = %c0 to %719 step %c1 { .. } }`.
            DfirOp::Affine(affine::Op::For {
                iv: arg1,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: vec![DfirOp::Scf(scf::Op::For {
                    iv: arg2,
                    lo,
                    hi,
                    step,
                    carried: Vec::new(),
                    body: Vec::new(),
                    dbg_name: None,
                })],
                dbg_name: None,
            }),
            view(src_view, hbm, start),
            view(dst_view, lx, lo),
            // `%data = agen.vector_load %src_mem_view[0, %arg1 * 3, %arg2 * 8]`, transposed.
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: data,
                view: src_view,
                indices: vec![
                    Index::Strided(vec![(arg2, 8)], 0),
                    Index::Strided(vec![(arg1, 3)], 0),
                    Index::Const(0),
                ],
                view_ty: ty.clone(),
                ty: vec_ty.clone(),
                multicast_info: None,
            }),
            // `agen.vector_store %data, %dst_mem_view[0, 0, 0]`.
            DfirOp::Agen(agen::Op::VectorStore {
                dbg_name: None,
                value: data,
                view: dst_view,
                indices: vec![Index::Const(0); 3],
                view_ty: ty.clone(),
                ty: vec_ty,
            }),
        ];

        let candidate = MasCandidate {
            op: &scope[7],
            comp: DfirUnit::L3lu,
            mem_index: MemoryOperandIndex::DirSrc,
        };
        let mut conditionals = Conditionals::default();
        let split = transform_vector_load::<Dd2>(
            &mut vals,
            &candidate,
            DataType::Sen169Fp16,
            EarOverflowCorrection::Allowed,
            &mut conditionals,
            &scope,
        );
        let TransferSplit::Split {
            ops,
            hoisted,
            erased,
            partitions,
            adjustments,
        } = split
        else {
            panic!("the key overflows the EAR by 177209344 elements: {split:?}")
        };
        assert_eq!(partitions, 2);
        // Dd2 asks nothing of the parity, so no partition's subscripts moved for it.
        assert_eq!(
            adjustments,
            [
                EvenImmutableAdjustment::NotOnThisArch,
                EvenImmutableAdjustment::NotOnThisArch
            ]
        );
        // `op.eraseOpAndUseChain()` — the store BEFORE the load it consumes, which is the only order
        // that erases nothing still in use.
        assert_eq!(erased, [&scope[8], &scope[7]]);
        // `%VAL_9` before `%VAL_10`: the `else` partition is filled first, so its constant is minted
        // first, and both sit at function scope in the vendor's output.
        assert_eq!(
            printed(&hoisted),
            concat!(
                "%13 = arith.constant 201329664 : index\n",
                "%17 = arith.constant 3072 : index\n",
            ),
            "3072 + 4 * 50331648, and the vendor's own 3072 + 4 * 2048 = 11264"
        );
        let arm = |view: &str, start: &str, store_view: &str, load: &str, subscript: &str| {
            format!(
                concat!(
                    "  {view} = dataflow.get_logical_memory_view %0, {start} {{layout_map = #MAP}} : index, index, memref<4x64x1xf16>\n",
                    "  {store_view} = dataflow.get_logical_memory_view %1, %2 {{layout_map = #MAP}} : index, index, memref<4x64x1xf16>\n",
                    "  {load} = agen.vector_load {view}[{subscript}, %6 * 3, 0] {{dbgName = \"\", load_order = #ORDER, load_set = #SET}} : memref<4x64x1xf16>, vector<64xf16>\n",
                    "  agen.vector_store {load}, {store_view}[0, 0, 0] {{store_order = #ORDER, store_set = #SET}} : memref<4x64x1xf16>, vector<64xf16>\n",
                ),
                view = view,
                start = start,
                store_view = store_view,
                load = load,
                subscript = subscript,
            )
        };
        let emitted = printed(&ops)
            .replace(
                "affine_map<(d0, d1, d2) -> (d0 * 6291456 + d1 * 1572864 + d2)>",
                "#MAP",
            )
            .replace("affine_map<(d0, d1, d2) -> (d0, d1, d2)>", "#ORDER")
            .replace(
                "affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>",
                "#SET",
            );
        assert_eq!(
            emitted,
            format!(
                concat!(
                    "%11 = arith.constant 4 : index\n",
                    "%12 = arith.cmpi slt, %7, %11 : index\n",
                    "scf.if %12 {{\n{then}}} else {{\n{else_}}}\n",
                ),
                // `%VAL_21 = agen.vector_load %VAL_19[0, %VAL_13 * 3, %VAL_14 * 8]`.
                then = arm("%18", "%17", "%20", "%19", "%7 * 8"),
                // `%VAL_24 = agen.vector_load %VAL_22[0, %VAL_13 * 3, %VAL_14 * 8 - 32]`.
                else_ = arm("%14", "%13", "%16", "%15", "%7 * 8 + -32"),
            ),
            "four ops per arm, the destination's view cloned into each, and only the shifted subscript \
             and the start address differ"
        );
    }

    /// 🎯 253/384 — ⭐⭐ IBM'S `one_dim_sen1p5` KEY: THE `[0, …]` SUBSCRIPT COMES BACK AS `[64, …]`.
    ///
    /// `mutable_addr_splitting_one_dim_sen1p5.mlir:39` — the start address 2112 is an ODD 33 sticks and
    /// the shift is even, so one stick of `f16` (128 bytes = 64 elements) leaves the immutable address
    /// and lands on `d0`'s subscript. ⛔ AND DD2 NEVER ASKS: the same input is untouched there, which is
    /// why the vendor keeps two copies of this file.
    #[test]
    fn the_sen1p5_answer_key_moves_a_stick_into_the_innermost_subscript() {
        let op = time_dims_transfer();
        let mut ad = AccessDetailsAffine::new(&op, DfirUnit::L3lu);
        ad.base.transfer_order = AffineMap::identity(3);
        ad.base.extents = vec![Elements(64), Elements(1), Elements(1)];

        // `%src_mem_view[0, %arg1 * 16, %arg2 * 8]`.
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(16),
                AffineExpr::dim(1).times(8),
            ],
        };

        let mut map = subscripts.clone();
        let mut addr_mod = 0;
        assert_eq!(
            adjust_for_even_immutable_addr::<Sen1p5>(
                2112,
                &mut map,
                &ad,
                &mut addr_mod,
                DataType::Sen169Fp16,
            ),
            EvenImmutableAdjustment::ShiftedByAStick {
                result: 0,
                elems_in_stick: 64,
            }
        );
        assert_eq!(addr_mod, -64, "the stick came out of the immutable address");
        assert_eq!(
            map.results[0],
            AffineExpr::Const(64),
            "the vendor's `[64, ..`"
        );
        assert_eq!(&map.results[1..], &subscripts.results[1..]);
        assert_eq!(
            (map.dims, map.syms),
            (2, 0),
            "the symbol count is preserved"
        );

        let mut untouched = subscripts.clone();
        let mut no_mod = 0;
        assert_eq!(
            adjust_for_even_immutable_addr::<Dd2>(
                2112,
                &mut untouched,
                &ad,
                &mut no_mod,
                DataType::Sen169Fp16,
            ),
            EvenImmutableAdjustment::NotOnThisArch
        );
        assert_eq!((untouched, no_mod), (subscripts, 0));
    }

    /// 🎯 252/384 — ⭐⭐ THE `sen1p5` KEY'S TWO PARTITIONS, IN THE ORDER THE WALK FILLS THEM.
    ///
    /// `mutable_addr_splitting_one_dim_sen1p5.mlir:19-50` splits eight iterations of `d1` in halves,
    /// so the boundary is **4** and the `else` arm's load is `[64, %VAL_11 * 16, %VAL_12 * 8 - 32]`
    /// against the `then` arm's `[64, %VAL_11 * 16, %VAL_12 * 8]` — `8 * prev_iters(4)`, with
    /// `4 * 2048` of start address to match.
    ///
    /// ⛔ AND THE `else` IS FILLED FIRST: `kReverseBFS` pops a breadth-first stack, so the deepest
    /// level comes first and `else` precedes `then` within one.
    #[test]
    fn the_sen1p5_answer_key_fills_the_else_partition_first_and_shifts_its_subscript() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let mas_data = [split_dim(iv, 1, 2048, 8)];
        let mut tree = construct_conditionals(&mut vals, &mas_data, &[4])
            .ops()
            .unwrap_or_default()
            .to_vec();
        let subscripts_map = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).scaled(16),
                AffineExpr::dim(1).scaled(8),
            ],
        };

        let mut filled: Vec<(String, i64)> = Vec::new();
        let mut creator = |vals: &mut Values, map: &AffineMap, start_addr_mod: i64| {
            filled.push((print::affine_map(map), start_addr_mod));
            vec![DfirOp::Arith(arith::Op::Constant {
                result: vals.mint(),
                value: start_addr_mod,
            })]
        };
        let done = fill_partitions(
            &mut vals,
            &mut tree,
            &mas_data,
            &[4],
            &subscripts_map,
            &mut creator,
        );

        assert_eq!(done, FilledPartitions::Filled { partitions: 2 });
        assert_eq!(
            filled,
            vec![
                (
                    "affine_map<(d0, d1) -> (0, d0 * 16, d1 * 8 - 32)>".to_owned(),
                    8192,
                ),
                ("affine_map<(d0, d1) -> (0, d0 * 16, d1 * 8)>".to_owned(), 0),
            ],
            "the else partition first, carrying 4 iterations of both the address and the subscript"
        );
        assert_eq!(
            printed(&tree),
            concat!(
                "%1 = arith.constant 4 : index\n",
                "%2 = arith.cmpi slt, %0, %1 : index\n",
                "scf.if %2 {\n",
                "  %4 = arith.constant 0 : index\n",
                "} else {\n",
                "  %3 = arith.constant 8192 : index\n",
                "}\n",
            ),
            "each partition's ops land inside its own arm, before that region's terminator"
        );

        // ⛔ AND THE REBUILT MAP DROPS THE SYMBOLS THE ORIGINAL DECLARED (`:1171`).
        let symbolic = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::dim(0).added(AffineExpr::sym(0))],
        };
        let mut shapes: Vec<(u32, u32)> = Vec::new();
        let dim0 = [split_dim(iv, 0, 2048, 8)];
        fill_partitions(
            &mut vals,
            &mut tree,
            &dim0,
            &[4],
            &symbolic,
            &mut |_: &mut Values, map: &AffineMap, _| {
                shapes.push((map.dims, map.syms));
                Vec::new()
            },
        );
        assert_eq!(shapes, vec![(1, 0), (1, 0)], "`s0` is still in the results");
    }

    /// 🎯 289/384 — THE VENDOR'S `constant_start_addr_1`: **3072 IS 48 WHOLE STICKS OF fp16**.
    ///
    /// `mutable_addr_splitting_one_dim.mlir:239-257` — `%c3072` is the immutable start address of the
    /// candidate's view and the program unit's `precision = "fp16"` makes `num_elems_in_stick`
    /// `128 * 8 / 16 = 64`. ⛔ AND THE CHECK IS ONLY HALF THE FUNCTION: the initialized list is what
    /// the truncated tail produces.
    #[test]
    fn the_one_dim_answer_keys_immutable_address_is_a_whole_number_of_fp16_sticks() {
        let (layout, ty) = answer_key_memref();
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let scope = vec![DfirOp::Affine(affine::Op::For {
            iv: arg1,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })];

        let op = time_dims_transfer();
        let mut ad = AccessDetailsAffine::new(&op, DfirUnit::L3lu);
        ad.base.element_width = Bits(16);
        ad.base.indices = vec![arg1];
        ad.indices_coeff_dict = IndicesCoeffDict {
            per_index: vec![(arg1, 192)],
            constant: 0,
        };

        let view = ConstStartMemView {
            from: vals.mint(),
            start: 3072,
            layout: &layout,
            ty: &ty,
        };
        assert_eq!(
            initialize::<Dd2>(&ad, &view, &scope),
            MasInitialization::Initialized(MasDataInit::Initialized {
                mas_data: vec![MasData {
                    iter_arg: Some(arg1),
                    dim: 0,
                    composed_coeff: 192,
                    num_iters: 4,
                    // `(4 - 2) * 192`.
                    weight: 384,
                }],
                max_mutable: MutableAddr(384),
            })
        );

        // ⛔ AND EIGHT ELEMENTS PAST IT IS NOT A STICK BOUNDARY, which is what the `DT_CHECK` aborts
        // on: an immutable address the EBR cannot name in sticks.
        assert_eq!(
            initialize::<Dd2>(
                &ad,
                &ConstStartMemView {
                    start: 3080,
                    ..view
                },
                &scope,
            ),
            MasInitialization::ImmutableAddrIsNotAWholeNumberOfSticks {
                start: 3080,
                elems_in_stick: NonZeroU64::new(64).expect("64 fp16 elements fill a stick"),
            }
        );

        // ⛔ AND AN UNPOPULATED RECORD IS THE DIVISION BY ZERO (`element_width_ = 0`).
        ad.base.element_width = Bits(0);
        assert_eq!(
            initialize::<Dd2>(&ad, &view, &scope),
            MasInitialization::NoElementsFitInAStick { width: Bits(0) }
        );
    }

    /// 🎯 290/384 — THE `sen1p5` ANSWER KEY, BUILT AND FILLED BY THE ONE CALL THAT DOES BOTH.
    ///
    /// `mutable_addr_splitting_one_dim_sen1p5.mlir:19-50` — entry 187's tree and entry 252's two
    /// partitions, `else` first, from one `createPartitions`.
    #[test]
    fn the_sen1p5_answer_key_is_built_and_filled_by_one_call() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let mas_data = [split_dim(iv, 1, 2048, 8)];
        let subscripts_map = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).scaled(16),
                AffineExpr::dim(1).scaled(8),
            ],
        };

        // Entry 187's tree, from a `Values` seeded exactly as `create_partitions` finds it.
        let expected_tree = {
            let mut same = Values::default();
            let _ = same.mint();
            construct_conditionals(&mut same, &mas_data, &[4])
                .ops()
                .unwrap_or_default()
                .to_vec()
        };

        let mut mods: Vec<i64> = Vec::new();
        let created = {
            let mut creator = |_: &mut Values, _: &AffineMap, start_addr_mod: i64| {
                mods.push(start_addr_mod);
                Vec::new()
            };
            create_partitions(&mut vals, &mas_data, &[4], &subscripts_map, &mut creator)
        };
        assert_eq!(
            created,
            Partitions::Created {
                ops: expected_tree,
                partitions: 2,
            },
            "the tree is entry 187's, and both of its leaves were filled"
        );
        assert_eq!(mods, vec![8192, 0], "the else partition first");

        // ⛔ AND NO PLAN IS NO TREE: `DT_CHECK(!partition_sizes.empty())` (`:989`).
        assert_eq!(
            create_partitions(
                &mut vals,
                &mas_data,
                &[],
                &subscripts_map,
                &mut |_: &mut Values, _: &AffineMap, _| Vec::new()
            ),
            Partitions::NoConditionalTree(ConditionalTree::NoPartitionsToBuild)
        );
    }

    // ── 321/384 and 322/384 — the two composite transfers, each on its own answer key ────────────

    /// THE `time_dims` KEY'S LAYOUT SCALED BY 4096, WHICH IS WHAT MAKES IT OVERFLOW HERE.
    ///
    /// [`MAX_MUTABLE_SIZE`] is [`None`] in this crate, so the vendor's
    /// `--dcc-mutable-addr-splitting-max-mutable-size=600000` (37500 `f16` elements) becomes the
    /// `EAR`'s own 134217728, and the key's `43136`-element span has to grow to reach it. Scaling the
    /// layout is entry 306's precedent: `43072 * 4096 + 64 = 176422976` against 134217728 leaves an
    /// overflow of 42205248, so the heaviest dimension's cut falls at 10 of its 16 iterations where
    /// the vendor's falls at 13, and the shift is `10 * 2048 * 4096` where the vendor's is `13 * 2048`.
    const TIME_DIMS_K: i64 = 4096;

    /// The two arms' start addresses, in the fill order [`TransferSplit::hoisted`] records.
    fn hoisted_starts(hoisted: &[DfirOp]) -> Vec<i64> {
        hoisted
            .iter()
            .map(|op| match op {
                DfirOp::Arith(arith::Op::Constant { value, .. }) => *value,
                _ => unreachable!("a partition's start address is an `arith.constant`"),
            })
            .collect()
    }

    /// `#[[$ATTR_9]]`/`#[[$ATTR_10]]` — the split time dimension PINNED and its own two constraints
    /// dropped, which is what `updateTimeSetForExplicitDims` leaves on every rebuilt transfer.
    fn time_set_with_dim_pinned(set: &IntegerSet, dim: u32, dropped: usize) -> IntegerSet {
        IntegerSet {
            dims: set.dims,
            symbols: set.symbols,
            constraints: std::iter::once(Constraint {
                expr: AffineExpr::dim(dim),
                is_equality: true,
            })
            .chain(
                set.constraints
                    .iter()
                    .enumerate()
                    .filter(|&(c, _)| c / 2 != dropped)
                    .map(|(_, constraint)| constraint.clone()),
            )
            .collect(),
        }
    }

    /// `affine.for %arg1 = 0 to 4 { affine.for %arg2 = 0 to 8 { } }` — the key's own two iterators.
    fn time_dims_loop_nest(outer: Val, inner: Val) -> DfirOp {
        let for_op = |iv: Val, hi: i64, body: Vec<DfirOp>| {
            DfirOp::Affine(affine::Op::For {
                iv,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(hi),
                carried: Vec::new(),
                body,
                dbg_name: None,
            })
        };
        for_op(outer, 4, vec![for_op(inner, 8, Vec::new())])
    }

    /// 🎯 321/384 — ⭐⭐ IBM'S `constant_start_addr` KEY END TO END: THE OUTERMOST TIME DIMENSION
    /// BECOMES AN `affine.for 0 to 16` AND THE TRANSFER IS REBUILT IN BOTH ARMS OF ONE `scf.if`.
    ///
    /// `mutable_addr_splitting_time_dims.mlir:116-146` in, `:19-60` out — one loop of 16 around the
    /// cut, three ops per arm in the vendor's order (the split dst view, the cloned src view, the
    /// transfer), `#[[$ATTR_9]]`'s pinned `d2`, and `dst:[0, %arg1 * 16, %arg3 * 8]` UNCHANGED by the
    /// split. ⛔ THE CUT IS 10 AND THE SHIFT 83886080 — see [`TIME_DIMS_K`].
    #[test]
    fn the_time_dims_key_splits_its_outermost_time_dimension_end_to_end() {
        let mut vals = Values::default();
        let (c0_op, c0) = index_const(&mut vals, 0);
        let (c3_op, _c3) = index_const(&mut vals, 3);
        let lx = vals.mint();
        let hbm = vals.mint();
        // [`time_dims_transfer`] is written in the answer key's own SSA numbering, so the scope mints
        // into it: `%9`/`%10` are the two iterators, `%11`/`%12` the two views, `%18` the load iv.
        while vals.issued() < 9 {
            vals.mint();
        }
        let arg1 = vals.mint();
        let arg3 = vals.mint();
        let src_view = vals.mint();
        let dst_view = vals.mint();
        while vals.issued() < 19 {
            vals.mint();
        }

        let (_, ty) = answer_key_memref();
        let layout = AffineMap::linear(&[1, 64 * TIME_DIMS_K, 256 * TIME_DIMS_K]);
        let view = |result: Val, from: Val| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start: c0,
                layout: layout.clone(),
                ty: ty.clone(),
            })
        };
        let scope = vec![
            c0_op,
            c3_op,
            time_dims_loop_nest(arg1, arg3),
            view(src_view, lx),
            view(dst_view, hbm),
            DfirOp::Agen(time_dims_transfer()),
        ];
        let candidate = MasCandidate {
            op: &scope[5],
            comp: DfirUnit::L3su,
            mem_index: MemoryOperandIndex::DirDst,
        };
        let mut conditionals = Conditionals::default();
        let split = transform_comp_load_and_store::<Dd2>(
            &mut vals,
            &candidate,
            DataType::Sen169Fp16,
            EarOverflowCorrection::Allowed,
            &mut conditionals,
            &scope,
        );
        let TransferSplit::Split {
            ops,
            hoisted,
            erased,
            partitions,
            adjustments,
        } = split
        else {
            panic!("the scaled key overflows the EAR by 42205248 elements: {split:?}")
        };
        assert_eq!(partitions, 2);
        assert_eq!(
            erased,
            [&scope[5]],
            "the transfer alone — a composite carries its own body"
        );
        assert_eq!(
            adjustments,
            [
                EvenImmutableAdjustment::NotOnThisArch,
                EvenImmutableAdjustment::NotOnThisArch
            ]
        );
        // `%[[VAL_6]] = arith.constant 26624` then `%[[VAL_7]] = arith.constant 0` — the `else`
        // partition is filled first, and 26624 is `13 * 2048` where this is `10 * 2048 * 4096`.
        let shift = 10 * 2048 * TIME_DIMS_K;
        assert_eq!(hoisted_starts(&hoisted), [shift, 0]);

        // `affine.for %[[VAL_13]] = 0 to 16 { %c13; cmpi slt; scf.if .. else .. }`.
        let [DfirOp::Affine(affine::Op::For { lo, hi, body, .. })] = &ops[..] else {
            panic!("one explicit loop for the split time dimension: {ops:?}")
        };
        assert_eq!(
            (lo, hi),
            (&affine::Bound::Const(0), &affine::Bound::Const(16))
        );
        let [
            DfirOp::Arith(arith::Op::Constant { value: cut, .. }),
            DfirOp::Arith(arith::Op::Compare { predicate, .. }),
            DfirOp::Scf(scf::Op::If {
                body: then_arm,
                else_body,
                ..
            }),
        ] = &body[..]
        else {
            panic!("the cut, its comparison and the two arms: {body:?}")
        };
        assert_eq!(
            (*cut, *predicate),
            (10, arith::CmpIPredicate::Slt),
            "the vendor's `arith.constant 13` at 4096x"
        );

        let narrowed = time_set_with_dim_pinned(&time_dims_time_set(), 2, 2);
        let original = match &scope[5] {
            DfirOp::Agen(agen::Op::CompositeLoadAndStore(transfer)) => transfer,
            _ => unreachable!("the candidate is the key's transfer"),
        };
        for (arm, start) in [(then_arm, &hoisted[1]), (else_body, &hoisted[0])] {
            let [
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: split_view,
                    start: split_start,
                    ..
                }),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: cloned_src,
                    from: cloned_from,
                    ..
                }),
                DfirOp::Agen(agen::Op::CompositeLoadAndStore(rebuilt)),
                DfirOp::Scf(scf::Op::Yield { .. }),
            ] = &arm[..]
            else {
                panic!("the two views, the transfer and the region's terminator: {arm:?}")
            };
            let DfirOp::Arith(arith::Op::Constant { result: from, .. }) = start else {
                unreachable!("a partition's start address is an `arith.constant`")
            };
            assert_eq!(split_start, from, "the arm reads its own hoisted constant");
            assert_eq!(cloned_from, &lx, "the unsplit side is cloned off `%lx`");
            assert_eq!((rebuilt.dst, rebuilt.src), (*split_view, *cloned_src));
            assert_eq!(
                rebuilt.dst_indices, original.dst_indices,
                "`createPartitions` reads the ORIGINAL two-dimensional subscripts map"
            );
            assert_eq!(rebuilt.time_set, narrowed);
            assert_eq!(
                rebuilt.time_symbols, original.time_symbols,
                "`time_symbols(%c3)`"
            );
            assert_eq!(rebuilt.dbg_name, Some(String::new()), "`dbgName = \"\"`");
            assert_eq!(rebuilt.dir, Some(agen::RoutingDirection::PseudoRandom));
        }
    }

    /// THE `query_start_addr` KEY'S TRANSFER (`mutable_addr_splitting_time_dims.mlir:169-186`), with
    /// the split side chosen by `mem_index`.
    ///
    /// ⛔ `direct_src` GATHERS ON `kDirSrc` AND `direct_dst` SCATTERS ON `kDirDst`, and the maps swap
    /// with them so both arrangements reach the SAME cut: the split side always carries
    /// `[0, %arg1 * 16 + 1, %arg2 * 8]` and the `(d2 * 64, d1, d0 * 8)` time map.
    fn query_transfer(
        mem_index: MemoryOperandIndex,
        hbm_view: Val,
        lx_view: Val,
        ibr_view: Val,
        ibr_ty: MemRef,
        args: (Val, Val),
    ) -> agen::Op {
        let (_, ty) = answer_key_memref();
        let (arg1, arg2) = args;
        // `load_set`/`store_set` — `(d0 >= 0, -d0 + 63 >= 0, d1 == 0, d2 == 0)`, as the first key's.
        let transfer_set = match time_dims_transfer() {
            agen::Op::CompositeLoadAndStore(transfer) => transfer.load_set.clone(),
            _ => unreachable!("the first key's transfer is a composite load and store"),
        };
        // `time_set = (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0, d2 >= 0, -d2 + 2 >= 0)`.
        let spanning = |dim: u32, upper: AffineExpr| {
            vec![
                Constraint {
                    expr: AffineExpr::dim(dim),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(dim).times(-1).plus(upper),
                    is_equality: false,
                },
            ]
        };
        let time_set = IntegerSet {
            dims: 3,
            symbols: 1,
            constraints: spanning(0, AffineExpr::Const(15))
                .into_iter()
                .chain(spanning(1, AffineExpr::sym(0).plus(AffineExpr::Const(-1))))
                .chain(spanning(2, AffineExpr::Const(2)))
                .collect(),
        };
        // `(d2 * 64, d1, d0 * 8)` — the split side's offset at each time step.
        let split_time_addr_map = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2).times(64),
                AffineExpr::dim(1),
                AffineExpr::dim(0).times(8),
            ],
        };
        // `[0, %arg1 * 16 + 1, %arg2 * 8]` and `[0, 0, %arg1]`.
        let split_indices = vec![
            Index::Const(0),
            Index::Strided(vec![(arg1, 16)], 1),
            Index::Strided(vec![(arg2, 8)], 0),
        ];
        let peer_indices = vec![Index::Const(0), Index::Const(0), Index::Val(arg1)];
        let address = agen::IndirectAccess {
            view: ibr_view,
            indices: vec![Index::Val(arg2)],
            ty: ibr_ty,
        };
        let on_dir_src = mem_index == MemoryOperandIndex::DirSrc;
        agen::Op::CompositeIndirectLoadAndStore(Box::new(agen::CompositeIndirectTransfer {
            // The address view sits on the side that is NOT split, which is the `DT_CHECK`.
            indirect_src: (!on_dir_src).then(|| address.clone()),
            direct_src: if on_dir_src { hbm_view } else { lx_view },
            direct_src_indices: if on_dir_src {
                split_indices.clone()
            } else {
                peer_indices.clone()
            },
            direct_src_ty: ty.clone(),
            indirect_dst: on_dir_src.then_some(address),
            direct_dst: if on_dir_src { lx_view } else { hbm_view },
            direct_dst_indices: if on_dir_src {
                peer_indices
            } else {
                split_indices
            },
            direct_dst_ty: ty,
            load_iv: Val(27),
            load_iv_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            load_set: transfer_set.clone(),
            load_order: AffineMap::identity(3),
            store_set: transfer_set,
            store_order: AffineMap::identity(3),
            time_symbols: vec![Val(1)],
            time_set,
            time_order: AffineMap::identity(3),
            load_indirect_time_addr_map: None,
            load_direct_time_addr_map: if on_dir_src {
                split_time_addr_map.clone()
            } else {
                AffineMap::identity(3)
            },
            store_indirect_time_addr_map: Some(AffineMap {
                dims: 3,
                syms: 0,
                results: vec![AffineExpr::Const(0); 3],
            }),
            store_direct_time_addr_map: if on_dir_src {
                AffineMap::identity(3)
            } else {
                split_time_addr_map
            },
            multicast_info: None,
            dbg_name: None,
            body: vec![DfirOp::Agen(agen::Op::Yield)],
        }))
    }

    /// THE `query_start_addr` KEY'S SCOPE, with its three views and the transfer last.
    ///
    /// ⛔ `%query` IS AN `arith.constant 0` HERE. `uniform.query_map(map:%def_map, key:%arg0)` over a
    /// mapping whose every destination is `%c0` is exactly the constant `isConstant`'s second arm
    /// answers `true` for, and this island has no `uniform` dialect — see
    /// [`SplitCandidateView::resolve`], which records that collapse.
    fn query_scope(vals: &mut Values, mem_index: MemoryOperandIndex) -> (Vec<DfirOp>, Val) {
        let (c0_op, c0) = index_const(vals, 0);
        let (c3_op, _c3) = index_const(vals, 3);
        let lx = vals.mint();
        let hbm = vals.mint();
        let ibr = vals.mint();
        while vals.issued() < 14 {
            vals.mint();
        }
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let hbm_view = vals.mint();
        let ibr_view = vals.mint();
        let lx_view = vals.mint();
        while vals.issued() < 28 {
            vals.mint();
        }

        let (_, ty) = answer_key_memref();
        let layout = AffineMap::linear(&[1, 64 * TIME_DIMS_K, 256 * TIME_DIMS_K]);
        // `memref<32xi32>` under `affine_map<(d0) -> (d0)>` — the address view.
        let ibr_ty = MemRef {
            shape: vec![32],
            elem: ElemType::Int(32),
        };
        let view = |result: Val, from: Val, layout: AffineMap, ty: MemRef| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start: c0,
                layout,
                ty,
            })
        };
        let scope = vec![
            c0_op,
            c3_op,
            time_dims_loop_nest(arg1, arg2),
            view(hbm_view, hbm, layout.clone(), ty.clone()),
            view(ibr_view, ibr, AffineMap::identity(1), ibr_ty.clone()),
            view(lx_view, lx, layout, ty),
            DfirOp::Agen(query_transfer(
                mem_index,
                hbm_view,
                lx_view,
                ibr_view,
                ibr_ty,
                (arg1, arg2),
            )),
        ];
        (scope, ibr_view)
    }

    /// 🎯 322/384 — ⭐⭐ IBM'S `query_start_addr` KEY END TO END: THE GATHER'S DIRECT SOURCE IS SPLIT
    /// AND ITS DIRECT **AND** INDIRECT DESTINATIONS ARE BOTH CLONED PER PARTITION.
    ///
    /// `mutable_addr_splitting_time_dims.mlir:148-186` in, `:63-113` out — one loop of 16, four ops
    /// per arm in the vendor's order (split src view, direct dst clone, address view clone, the
    /// transfer), `#[[$ATTR_10]]`'s pinned `d0`, and `indirect_dst:[%arg2]` carried through with its
    /// index. ⛔ THE CUT IS 10 AND THE SHIFT 83886080 — see [`TIME_DIMS_K`].
    #[test]
    fn the_query_start_addr_key_splits_the_direct_source_and_clones_both_destinations() {
        let mut vals = Values::default();
        let (scope, ibr_view) = query_scope(&mut vals, MemoryOperandIndex::DirSrc);
        let candidate = MasCandidate {
            op: &scope[6],
            comp: DfirUnit::L3lu,
            mem_index: MemoryOperandIndex::DirSrc,
        };
        let mut conditionals = Conditionals::default();
        let split = transform_comp_ind_load_and_store::<Dd2>(
            &mut vals,
            &candidate,
            DataType::Sen169Fp16,
            EarOverflowCorrection::Allowed,
            &mut conditionals,
            &scope,
        );
        let TransferSplit::Split {
            ops,
            hoisted,
            partitions,
            ..
        } = split
        else {
            panic!("the scaled key overflows the EAR by 42205248 elements: {split:?}")
        };
        assert_eq!(partitions, 2);
        assert_eq!(hoisted_starts(&hoisted), [10 * 2048 * TIME_DIMS_K, 0]);

        let [DfirOp::Affine(affine::Op::For { hi, body, .. })] = &ops[..] else {
            panic!("one explicit loop for the split time dimension: {ops:?}")
        };
        assert_eq!(hi, &affine::Bound::Const(16));
        let [_, _, DfirOp::Scf(scf::Op::If { body: then_arm, .. })] = &body[..] else {
            panic!("the cut, its comparison and the two arms: {body:?}")
        };
        let [
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: split_view, ..
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: direct_dst, ..
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: address, ..
            }),
            DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(rebuilt)),
            DfirOp::Scf(scf::Op::Yield { .. }),
        ] = &then_arm[..]
        else {
            panic!(
                "the split src view, both cloned destinations, the transfer and the yield: {then_arm:?}"
            )
        };
        let original = match &scope[6] {
            DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(transfer)) => transfer,
            _ => unreachable!("the candidate is the key's transfer"),
        };
        assert_eq!(rebuilt.direct_src, *split_view);
        assert_eq!(rebuilt.direct_dst, *direct_dst);
        assert_eq!(
            rebuilt.direct_src_indices, original.direct_src_indices,
            "`createPartitions` reads the ORIGINAL subscripts map"
        );
        assert_eq!(
            rebuilt.indirect_dst,
            Some(agen::IndirectAccess {
                view: *address,
                indices: original
                    .indirect_dst
                    .as_ref()
                    .map(|indirect| indirect.indices.clone())
                    .unwrap_or_default(),
                ty: MemRef {
                    shape: vec![32],
                    elem: ElemType::Int(32),
                },
            }),
            "`indirect_dst:%[[VAL_26]][%[[VAL_15]]]` — the address view cloned, its index kept"
        );
        assert_ne!(
            *address, ibr_view,
            "each partition gets its own address view"
        );
        assert_eq!(rebuilt.indirect_src, None);
        assert_eq!(
            rebuilt.time_set,
            time_set_with_dim_pinned(&original.time_set, 0, 0)
        );
    }

    /// 🎯 322/384 — ⛔⛔ AND THE `kDirDst` BRANCH REBUILDS A SCATTER CLAIMING AN INDIRECT
    /// **DESTINATION** THAT IS REALLY ITS INDIRECT SOURCE, WITH NO INDICES.
    ///
    /// `MutableAddrSplitting.cpp:653`, `:657` — `op.getIndirectSrcMemref()` goes into the indirect
    /// destination slot with `getEmptyAffineMap()`, and `num_ind_dst_indices` is recounted off the
    /// original's empty one. The vendor's key never reaches this branch; the same transfer with the
    /// address on the LOAD side does, and the two indirect slots come out pointing at one view.
    #[test]
    fn the_dir_dst_branch_puts_the_indirect_source_in_the_destination_slot_with_no_indices() {
        let mut vals = Values::default();
        let (scope, _) = query_scope(&mut vals, MemoryOperandIndex::DirDst);
        let candidate = MasCandidate {
            op: &scope[6],
            comp: DfirUnit::L3su,
            mem_index: MemoryOperandIndex::DirDst,
        };
        let mut conditionals = Conditionals::default();
        let split = transform_comp_ind_load_and_store::<Dd2>(
            &mut vals,
            &candidate,
            DataType::Sen169Fp16,
            EarOverflowCorrection::Allowed,
            &mut conditionals,
            &scope,
        );
        let TransferSplit::Split { ops, .. } = split else {
            panic!("the scaled key overflows the EAR by 42205248 elements: {split:?}")
        };
        let [DfirOp::Affine(affine::Op::For { body, .. })] = &ops[..] else {
            panic!("one explicit loop for the split time dimension: {ops:?}")
        };
        let [_, _, DfirOp::Scf(scf::Op::If { body: then_arm, .. })] = &body[..] else {
            panic!("the cut, its comparison and the two arms: {body:?}")
        };
        let [
            ..,
            DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(rebuilt)),
            DfirOp::Scf(scf::Op::Yield { .. }),
        ] = &then_arm[..]
        else {
            panic!("the transfer sits last, ahead of the region's terminator: {then_arm:?}")
        };
        let (Some(src), Some(dst)) = (&rebuilt.indirect_src, &rebuilt.indirect_dst) else {
            panic!("both indirect slots are filled: {rebuilt:?}")
        };
        assert_eq!(src.view, dst.view, "ONE cloned address view in both slots");
        assert!(
            dst.indices.is_empty(),
            "`getEmptyAffineMap()` has no operands"
        );
        let original = match &scope[6] {
            DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(transfer)) => transfer,
            _ => unreachable!("the candidate is the key's transfer"),
        };
        assert_eq!(
            Some(&src.indices),
            original
                .indirect_src
                .as_ref()
                .map(|indirect| &indirect.indices),
            "the indirect SOURCE keeps its own `[%arg2]`"
        );
    }

    /// 🎯 351/384 — THE TWO FILTERS ARE THE PASS: an HBM view in an L3 half is a candidate, and the
    /// LX view beside it, the symbol-started view above it and the whole `lxlu` unit are not.
    #[test]
    fn only_an_hbm_view_in_an_l3_unit_reaches_a_transform() {
        let mut vals = Values::default();
        let get_unit = |result: Val, unit: DfirUnit, residency: Residency| {
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result,
                residency,
                unit,
                num_folds: None,
            })
        };
        let core0 = Core::checked(0).expect("core 0 exists");
        let hbm = vals.mint();
        let lx = vals.mint();
        let l3lu = vals.mint();
        let lxlu = vals.mint();
        let preamble = vec![
            get_unit(hbm, DfirUnit::Hbm, Residency::Global),
            get_unit(lx, DfirUnit::Lx, Residency::Scratchpad { core: core0 }),
            get_unit(l3lu, DfirUnit::L3lu, Residency::CoreWide { core: core0 }),
            get_unit(
                lxlu,
                DfirUnit::Lxlu,
                Residency::Corelet {
                    core: core0,
                    corelet: Corelet::checked(0).expect("corelet 0 exists"),
                },
            ),
        ];

        let layout = AffineMap::linear(&[2048, 1]);
        let ty = MemRef {
            shape: vec![1, 2048],
            elem: ElemType::F16,
        };
        let vec_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let view = |vals: &mut Values, from: Val, start: Val| {
            let result = vals.mint();
            (
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result,
                    from,
                    start,
                    layout: layout.clone(),
                    ty: ty.clone(),
                }),
                result,
            )
        };
        let (start_op, start) = index_const(&mut vals, 0);
        let sym = vals.mint();
        let (hbm_view_op, hbm_view) = view(&mut vals, hbm, start);
        let (lx_view_op, lx_view) = view(&mut vals, lx, start);
        let (sym_view_op, _) = view(&mut vals, hbm, sym);
        let data = vals.mint();
        let l3lu_body = vec![
            start_op.clone(),
            DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: sym,
                symbol_id: -1476,
                max_value: Some(8),
            }),
            hbm_view_op,
            lx_view_op,
            // ⛔ AN HBM VIEW STARTED BY A SYMBOL — `isa_and_nonnull<symbol::CreateSymbolOp, ..>`.
            sym_view_op,
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: data,
                view: hbm_view,
                indices: vec![Index::Const(0); 2],
                view_ty: ty.clone(),
                ty: vec_ty.clone(),
                multicast_info: None,
            }),
            DfirOp::Agen(agen::Op::VectorStore {
                dbg_name: None,
                value: data,
                view: lx_view,
                indices: vec![Index::Const(0); 2],
                view_ty: ty.clone(),
                ty: vec_ty.clone(),
            }),
        ];

        // The same HBM load again, on a unit the component gate never walks into.
        let (other_view_op, other_view) = view(&mut vals, hbm, start);
        let lxlu_body = vec![
            start_op,
            other_view_op,
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: vals.mint(),
                view: other_view,
                indices: vec![Index::Const(0); 2],
                view_ty: ty,
                ty: vec_ty,
                multicast_info: None,
            }),
        ];

        let a_unit = |on: DfirUnit, val: Val, body: Vec<DfirOp>| ProgramUnit::<Dd2> {
            on: Units::one(on, val),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        let program = Program::<Dd2> {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble,
            units: ProgramUnits::of(
                a_unit(DfirUnit::L3lu, l3lu, l3lu_body),
                vec![a_unit(DfirUnit::Lxlu, lxlu, lxlu_body)],
            ),
            arch: core::marker::PhantomData,
        };

        let splitting =
            run_on_operation::<Dd2>(&program, &mut vals, EarOverflowCorrection::Allowed);
        let MutableAddrSplitting::Split(split) = splitting else {
            panic!("every candidate has one user and one direct memory operand: {splitting:?}")
        };
        assert_eq!(
            split.len(),
            1,
            "one candidate: the LX view is not HBM, the symbol-started view is excluded, and the \
             `lxlu` unit is not walked"
        );
        let MasSplit {
            candidate,
            dispatch,
        } = &split[0];
        assert_eq!(candidate.comp, DfirUnit::L3lu);
        assert_eq!(candidate.mem_index, MemoryOperandIndex::DirSrc);
        assert!(
            matches!(candidate.op, DfirOp::Agen(agen::Op::VectorLoad { view, .. }) if *view == hbm_view),
            "the load of the HBM view, reached as its one user"
        );
        // `[0, 0]` involves no loop, so `initialize` leaves `mas_data` empty and `max_mutable` at 0.
        assert_eq!(
            *dispatch,
            MasDispatch::VectorLoad(TransferSplit::AddressFits(MutableAddrOverflow::InRange))
        );
    }
}

/// WHAT A LOOP ANSWERS WHEN ASKED HOW MANY TIMES IT RUNS — every value the reference's one `int64_t`
/// carries, and the four aborts it carries none for.
///
/// # ⛔⛔ ONE `int64_t` WITH THREE SENTINELS AND FOUR ABORTS
///
/// `getLoopTripCount` returns a count, `-1` twice, `false` once, and stops the compiler four times
/// (`MutableAddrSplitting.cpp:743-798`). Its only caller reads the number back through
/// `DT_CHECK(num_iters >= 0)` (`initMASData`, `:730`) — so `-1` aborts THERE while `return false`,
/// which is **0** in an `int64_t` function, walks straight past it and becomes a dimension of zero
/// iterations. That asymmetry is not a rounding of the same idea: it is two different outcomes from
/// one return type, and an enum is where it can be seen.
///
/// ⭐ THE ABORTS ARE STATES, NOT PANICS. Three `DT_CHECK`s and two `llvm_unreachable`s say "the
/// scheduler does not write this"; each is a shape of DataflowIR that is perfectly expressible in
/// this island, so each gets a variant and the caller decides. Nothing here refuses at run time.
///
/// ⚠️ `LoopTripCount`, NOT `TripCount`, BECAUSE A SIBLING MODULE ALREADY HAS THAT NAME FOR A
/// DIFFERENT DOMAIN.
/// [`TripCount`](super::tf_loop_unroll_for_shuffle_op::TripCount) is `getConstantTripCount`'s answer
/// (entry 182): a strictly positive unroll factor, narrowed because
/// `loopUnrollByFactor` asserts on zero. This one is a count that may be zero or negative, because
/// its caller's own `DT_CHECK` is what rejects those. Two functions, two domains — and one `use`
/// list in this module's parent would otherwise pick one of them silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopTripCount {
    /// `(ub - lb) / step` — the count.
    ///
    /// ⛔ SIGNED, BECAUSE THE REFERENCE'S DIVISION IS. An `affine.for` from 8 to 0 has constant
    /// bounds and yields `-8`, which is a count its caller's `DT_CHECK(num_iters >= 0)` rejects —
    /// unlike [`Iterations`](super::tf_utils::Iterations), which saturates. Saturating here would turn that abort into a silently
    /// empty dimension, so the quotient is kept as it comes out.
    Iterations(i64),

    /// `if (!affine_for.hasConstantBounds()) return -1;` (`:752`) — an `affine.for` whose lower or
    /// upper bound is an [`affine::Bound::Val`].
    AffineBoundsAreNotConstant,

    /// `if (!const_step.has_value()) return -1;` (`:759`) — an `scf.for` whose step operand is not an
    /// `arith.constant`.
    ScfStepIsNotConstant(Val),

    /// `else return false;` (`:777`) — the upper bound is a `symbol.create_symbol` with no `maxValue`.
    ///
    /// ⛔⛔ THIS IS A TRIP COUNT OF **ZERO**, NOT A REFUSAL, AND THAT IS THE REFERENCE'S OWN ANSWER.
    /// `return false` in an `int64_t` function returns 0, which passes
    /// `DT_CHECK(num_iters >= 0)` and gives the dimension weight 0 (`num_iters < 2 ? 0 : …`). See
    /// [`LoopTripCount::iterations`]: this variant answers `Some(0)` and
    /// [`LoopTripCount::AffineBoundsAreNotConstant`] answers [`None`], which is the whole difference
    /// between the two sentinels.
    SymbolUpperBoundHasNoMaxValue(Val),

    /// `DT_CHECK(lb_op)` (`:766`) — an `scf.for` whose lower bound is not an `arith.constant`.
    ///
    /// ⚠️ `dyn_cast_or_null`, SO A REGION ARGUMENT LANDS HERE TOO: the reference tolerates a null
    /// defining op in the cast and then rejects the null result, which is one state, not two.
    ScfLowerBoundIsNotConstant(Val),

    /// `DT_CHECK(ub_op)` (`:770`) — the upper bound is bound by no operation at all.
    ///
    /// ⭐ SEPARATE FROM [`LoopTripCount::UnsupportedScfUpperBound`] BECAUSE THE REFERENCE SEPARATES THEM:
    /// a plain `dyn_cast` on the null would have fallen into the `llvm_unreachable`, and the
    /// `DT_CHECK` in front of the three arms is what makes "nothing binds it" its own answer.
    ScfUpperBoundIsNotAnOperation(Val),

    /// `DT_CHECK_MSG(true_op && false_op, "Expecting SelectOp upper bound to contain constant
    /// values.")` (`:783-785`) — an `arith.select` upper bound with a non-constant arm.
    SelectUpperBoundArmsAreNotConstant(Val),

    /// `llvm_unreachable("unsupported upper loop bound operation")` (`:790`) — the upper bound is
    /// bound by an operation that is none of the three the function knows.
    ///
    /// ⭐ THE `arith.divsi` BOUND THE SCHEDULER ACTUALLY WRITES IS THIS ONE.
    /// `dynamic_pt_masking.mlir:226-228` spells a trip count `(%hi - %lo) / %step`; that op is
    /// neither a constant, a symbol nor a select, so this pass cannot count such a loop at all.
    UnsupportedScfUpperBound(Val),

    /// `llvm_unreachable("Unsupported operation.")` (`:793`) — the operation is neither an
    /// `affine.for` nor an `scf.for`.
    NotALoop,

    /// `DT_CHECK_MSG(lb == 0 || step == 1, "Expecting a lower bound of 0 and a step of 1 for the
    /// loop.")` (`:795-796`).
    ///
    /// ⚠️ AN `||`, SO IT TAKES BOTH TO FAIL: a lower bound of 4 with a step of 1 is fine, and so is a
    /// lower bound of 0 with a step of 8. What the reference will not divide is a strided loop that
    /// also starts somewhere other than zero, because `(ub - lb) / step` is only the trip count of
    /// such a loop when the span happens to divide.
    ///
    /// ⭐ UNREACHABLE FROM THE AFFINE ARM BY CONSTRUCTION. [`affine::Op::For`] carries no step, so
    /// `getStepAsInt()` is 1 for every `affine.for` this island can hold and the disjunct is already
    /// satisfied — this variant belongs to the `scf` arm alone.
    StridedLoopWithNonZeroLowerBound {
        /// `lb`, which is not 0.
        lo: LoopBound,
        /// `step`, which is not 1.
        step: LoopStep,
    },

    /// A step of zero or below, which the reference divides by.
    ///
    /// # ⛔⛔ NOT A NARROWING OF THE REFERENCE — A NARROWING OF UNDEFINED BEHAVIOUR
    ///
    /// `scf::ForOp::verify` checks only that the init args match the results (`SCF.cpp:379-386`);
    /// the "constant step operand must be positive" verifier belongs to `scf::ParallelOp`
    /// (`SCF.cpp:2804-2808`). So `scf.for %i = %c0 to %c8 step %c0` is a VERIFIABLE `scf.for`, and
    /// `(ub - lb) / step` on it is an integer division by zero — undefined behaviour in the
    /// reference, not an answer to reproduce. A negative step gives a negative quotient that its
    /// caller's `DT_CHECK(num_iters >= 0)` rejects. [`LoopStep::checked`] declines both, and the
    /// value is carried so the decision is legible.
    ScfStepIsNotPositive(i64),
}

impl LoopTripCount {
    /// THE COUNT, WHERE THERE IS ONE — the `int64_t` the reference hands back to
    /// `DT_CHECK(num_iters >= 0)`.
    ///
    /// ⭐ `Some(0)` FOR [`LoopTripCount::SymbolUpperBoundHasNoMaxValue`], because `return false` is a
    /// zero that passes that check. Every other non-count is `-1` or an abort, and both are [`None`].
    #[must_use]
    pub const fn iterations(self) -> Option<i64> {
        match self {
            LoopTripCount::Iterations(count) => Some(count),
            LoopTripCount::SymbolUpperBoundHasNoMaxValue(_) => Some(0),
            LoopTripCount::AffineBoundsAreNotConstant
            | LoopTripCount::ScfStepIsNotConstant(_)
            | LoopTripCount::ScfLowerBoundIsNotConstant(_)
            | LoopTripCount::ScfUpperBoundIsNotAnOperation(_)
            | LoopTripCount::SelectUpperBoundArmsAreNotConstant(_)
            | LoopTripCount::UnsupportedScfUpperBound(_)
            | LoopTripCount::NotALoop
            | LoopTripCount::StridedLoopWithNonZeroLowerBound { .. }
            | LoopTripCount::ScfStepIsNotPositive(_) => None,
        }
    }
}

/// Replaces: e184_getLoopTripCount
///
/// **184/384** `MutableAddrSplittingPass::getLoopTripCount` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743` (57L).
///
/// ```cpp
/// // TODO: Remove the DT_CHECK safely and make this a broader utility.
/// int64_t MutableAddrSplittingPass::getLoopTripCount(Operation *op) const {
///   DT_CHECK(op);
///   int step = 1;
///   int64_t lb = 0, ub = 0;
///   if (auto affine_for = dyn_cast<affine::AffineForOp>(op)) {
///     step = affine_for.getStepAsInt();
///     if (!affine_for.hasConstantBounds()) return -1;
///     lb = affine_for.getConstantLowerBound();
///     ub = affine_for.getConstantUpperBound();
///   } else if (auto scf_for = dyn_cast<scf::ForOp>(op)) {
///     auto const_step = scf_for.getConstantStep();
///     if (!const_step.has_value()) return -1;
///     step = const_step.value().getSExtValue();
///     auto lb_op = dyn_cast_or_null<arith::ConstantOp>(
///         scf_for.getLowerBound().getDefiningOp());
///     DT_CHECK(lb_op);
///     lb = cast<IntegerAttr>(lb_op.getValue()).getInt();
///     auto ub_op = scf_for.getUpperBound().getDefiningOp();
///     DT_CHECK(ub_op);
///     if (auto ub_const_op = dyn_cast<arith::ConstantOp>(ub_op)) {
///       ub = cast<IntegerAttr>(ub_const_op.getValue()).getInt();
///     } else if (auto ub_sym_op = dyn_cast<symbol::CreateSymbolOp>(ub_op)) {
///       if (ub_sym_op->hasAttr("maxValue"))
///         ub = cast<IntegerAttr>(ub_sym_op->getAttr("maxValue")).getInt();
///       else
///         return false;
///     } else if (auto select_ub_op = dyn_cast<arith::SelectOp>(ub_op)) {
///       auto true_op = dyn_cast_or_null<arith::ConstantOp>(
///           select_ub_op.getTrueValue().getDefiningOp());
///       auto false_op = dyn_cast_or_null<arith::ConstantOp>(
///           select_ub_op.getFalseValue().getDefiningOp());
///       DT_CHECK_MSG(
///           true_op && false_op,
///           "Expecting SelectOp upper bound to contain constant values.");
///       auto true_val = cast<IntegerAttr>(true_op.getValue()).getInt();
///       auto false_val = cast<IntegerAttr>(false_op.getValue()).getInt();
///       ub = true_val > false_val ? true_val : false_val;
///     } else {
///       llvm_unreachable("unsupported upper loop bound operation");
///     }
///   } else {
///     llvm_unreachable("Unsupported operation.");
///   }
///   DT_CHECK_MSG(lb == 0 || step == 1,
///                "Expecting a lower bound of 0 and a step of 1 for the loop.");
///
///   return (ub - lb) / step;
/// }
/// ```
///
/// # ⭐⭐ THE SELECT ARM IS THE WHOLE POINT: A SYMBOLIC BOUND MEASURED AT ITS WORST CASE
///
/// The mutable address a transfer reaches depends on how far its loops travel, and this pass has to
/// bound that BEFORE the schedule fixes any symbol. So the three upper-bound arms are three ways of
/// saying "the largest value this bound can take":
///
/// | upper bound | worst case |
/// |---|---|
/// | `arith.constant N` | `N` |
/// | `symbol.create_symbol {maxValue = N}` | `N` — the scheduler's own promise ([`symbol::Op::CreateSymbol`]) |
/// | `arith.select %c, %a, %b` | `max(a, b)`, ⛔ **not** the value the condition picks |
///
/// ⭐ AND THE VENDOR EXERCISES TWO OF THE THREE.
/// `mutable_addr_splitting_one_dim.mlir` has `constant_start_addr_1` bound by
/// `symbol.create_symbol {SymbolId = -1476 : i64, granularity = 8 : i64, maxValue = 8 : i64}`
/// (`:251`) and `select_ub` bound by `arith.select %904, %c8, %c4 : index` (`:387`) — and expects a
/// trip count of **8** from each, which is what makes both cases partition identically on that
/// dimension. ⛔ `ub = true_val > false_val ? true_val : false_val` is a strict `>`, so equal arms
/// take the true one; the same number either way, which is why the tie is not observable.
///
/// # ⛔ NEITHER ARM IS A CONSTANT-FOLD, AND `constant_index` STILL SAYS NO TO BOTH
///
/// [`constant_index`] is `dyn_cast<arith::ConstantIndexOp>` and
/// answers [`None`] for a `symbol.create_symbol` and for an `arith.select` with two constant arms —
/// which is why [`get_dataflow_for_loop_info_if_iv`](super::tf_utils::get_dataflow_for_loop_info_if_iv)
/// (entry 142) declines such a loop while this function counts it. Two functions read the same three
/// operands of the same op and give different answers on purpose: one wants the trip count the
/// program WILL have, this one wants the trip count it COULD have.
///
/// # ⚠️ `scope` IS THE DEF-USE WALK
///
/// `getDefiningOp()` is MLIR asking a value what bound it. This island has no use lists, so the
/// caller passes the ops in scope and [`defining_op`] scans them; an SSA value is bound once, so the
/// two agree on every well-formed program. See [`SplitCandidateView::resolve`].
///
/// ⚠️ `DT_CHECK(op)` IS DISCHARGED BY THE SIGNATURE — a `&DfirOp` is not null.
///
/// # Arguments
///
/// * `loop_op` — the operation to count. Not necessarily a loop: see [`LoopTripCount::NotALoop`].
/// * `scope` — the operations whose results are in scope for `loop_op`'s bound operands.
#[must_use]
pub fn get_loop_trip_count(loop_op: &DfirOp, scope: &[DfirOp]) -> LoopTripCount {
    let (lo, step, ub) = match loop_op {
        // `if (auto affine_for = dyn_cast<affine::AffineForOp>(op))` — TESTED FIRST, and the affine
        // bounds are attributes rather than operands, so nothing is walked.
        DfirOp::Affine(affine::Op::For { lo, hi, .. }) => {
            // `step = affine_for.getStepAsInt();` — 1, and only 1: see
            // [`LoopTripCount::StridedLoopWithNonZeroLowerBound`].
            //
            // `if (!affine_for.hasConstantBounds()) return -1;` — which is
            // `hasConstantLowerBound() && hasConstantUpperBound()` (`AffineOps.td:313-315`), so ONE
            // pattern for both.
            let (affine::Bound::Const(lo), affine::Bound::Const(hi)) = (*lo, *hi) else {
                return LoopTripCount::AffineBoundsAreNotConstant;
            };
            (LoopBound(lo), LoopStep::ONE, hi)
        }
        // `else if (auto scf_for = dyn_cast<scf::ForOp>(op))` — three SSA operands, three walks.
        DfirOp::Scf(scf::Op::For { lo, hi, step, .. }) => {
            // `auto const_step = scf_for.getConstantStep(); if (!const_step.has_value()) return -1;`
            let Some(step_value) = constant_index(*step, scope) else {
                return LoopTripCount::ScfStepIsNotConstant(*step);
            };
            let Some(step) = LoopStep::checked(step_value) else {
                return LoopTripCount::ScfStepIsNotPositive(step_value);
            };

            // `dyn_cast_or_null<arith::ConstantOp>(getLowerBound().getDefiningOp())`, then
            // `DT_CHECK(lb_op)`.
            let Some(lo_value) = constant_index(*lo, scope) else {
                return LoopTripCount::ScfLowerBoundIsNotConstant(*lo);
            };

            // `auto ub_op = scf_for.getUpperBound().getDefiningOp(); DT_CHECK(ub_op);` — the raw
            // defining op, checked before any of the three casts.
            let Some(ub_op) = defining_op(*hi, scope) else {
                return LoopTripCount::ScfUpperBoundIsNotAnOperation(*hi);
            };
            let ub = match ub_op {
                // `if (auto ub_const_op = dyn_cast<arith::ConstantOp>(ub_op))`.
                DfirOp::Arith(arith::Op::Constant { value, .. }) => *value,
                // `else if (auto ub_sym_op = dyn_cast<symbol::CreateSymbolOp>(ub_op))` — the
                // `hasAttr("maxValue")` question, which is this island's [`Option`].
                DfirOp::Symbol(symbol::Op::CreateSymbol { max_value, .. }) => match max_value {
                    Some(max) => *max,
                    None => return LoopTripCount::SymbolUpperBoundHasNoMaxValue(*hi),
                },
                // `else if (auto select_ub_op = dyn_cast<arith::SelectOp>(ub_op))` — ⛔ THE LARGER
                // ARM, not the selected one. The condition is deliberately not read.
                DfirOp::Arith(arith::Op::Select {
                    true_value,
                    false_value,
                    ..
                }) => {
                    // Two more `dyn_cast_or_null<arith::ConstantOp>`s, then
                    // `DT_CHECK_MSG(true_op && false_op, …)` over both at once.
                    let (Some(true_val), Some(false_val)) = (
                        constant_index(*true_value, scope),
                        constant_index(*false_value, scope),
                    ) else {
                        return LoopTripCount::SelectUpperBoundArmsAreNotConstant(*hi);
                    };
                    // `ub = true_val > false_val ? true_val : false_val;`
                    true_val.max(false_val)
                }
                // `else llvm_unreachable("unsupported upper loop bound operation")`.
                //
                // ⛔ SPELLED OUT BY DIALECT, not a wildcard: a fourth upper-bound shape the scheduler
                // starts writing must be classified here rather than inherit an abort.
                DfirOp::Arith(_)
                | DfirOp::Affine(_)
                | DfirOp::Scf(_)
                | DfirOp::Dataflow(_)
                | DfirOp::Agen(_)
                | DfirOp::VectorChain(_)
                | DfirOp::Vector(_)
                | DfirOp::Uniform(_)
                | DfirOp::Symbol(
                    symbol::Op::ImmutableMapping { .. } | symbol::Op::QueryMap { .. },
                ) => return LoopTripCount::UnsupportedScfUpperBound(*hi),
            };
            (LoopBound(lo_value), step, ub)
        }
        // `else llvm_unreachable("Unsupported operation.")` — the TODO above the function ("Remove
        // the DT_CHECK safely and make this a broader utility") is about exactly this arm.
        DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Vector(_)
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => return LoopTripCount::NotALoop,
    };

    // `DT_CHECK_MSG(lb == 0 || step == 1, "Expecting a lower bound of 0 and a step of 1 for the
    // loop.");`
    if lo.0 != 0 && step.get() != 1 {
        return LoopTripCount::StridedLoopWithNonZeroLowerBound { lo, step };
    }

    // `return (ub - lb) / step;` — a truncating signed division, as the reference's is. The step is
    // positive by construction, so the divisor is not zero.
    LoopTripCount::Iterations((ub - lo.0) / step.get())
}

/// THE MUTABLE ADDRESS A TRANSFER REACHES, IN ELEMENTS — the `max_mutable` threaded through
/// `initMASData`, [`has_mutable_addr_overflow`], `setupForPartitioning`, `calculatePartitionSizes`
/// and `synthesizeTimeInfo`.
///
/// # ⛔⛔ SIGNED, WHERE [`Elements`] IS NOT, AND THAT IS NOT A CHOICE
///
/// `max_mutable` is a SUM OF WEIGHTS (`:738`, `:1278`) and a weight is
/// `(num_iters - 2) * composed_coeff` over a coefficient the layout map supplies. It starts at the
/// subscript map's constant offset (`iter_coeff_dict[nullptr]`, `:724`), and
/// `calculatePartitionSizes` immediately subtracts a limit from it and requires the difference to be
/// `<= 0` (`:875-876`, `:920`) — so both the value and its difference live in the negative half. An
/// [`Elements`] would make the reference's own `int64_t` unrepresentable.
///
/// ⭐ SO THE NEWTYPE IS WHAT KEEPS IT FROM MIXING WITH THE LIMIT IT IS COMPARED AGAINST. The limit is
/// a RANGE divided by an element width — an unsigned count of elements the register can address
/// ([`AddrRange::elements`]) — and the two are compared, not added. [`MutableAddr::exceeds`] and
/// [`MutableAddr::beyond`] are the only two ways across, and each states which side is signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MutableAddr(pub i64);

impl MutableAddr {
    /// `max_mutable > (getMaxMutableRange(comp) / elem_size_in_bits)` (`:804`).
    ///
    /// ⭐ A NEGATIVE SPAN EXCEEDS NOTHING, which is what the reference's signed comparison against a
    /// non-negative quotient says too — written out because the widths differ here.
    #[must_use]
    pub const fn exceeds(self, limit: Elements) -> bool {
        self.0 > 0 && self.0.unsigned_abs() > limit.0
    }

    /// `max_mutable - (max_mutable_range / elem_size_in_bits)` (`:875-876`) — how far past the limit it
    /// reaches, negative when it is inside.
    ///
    /// ⛔ SATURATING, BECAUSE THE LIMIT IS UNSIGNED AND THE DIFFERENCE IS NOT. A limit larger than
    /// [`i64::MAX`] cannot be reached by any register this crate models — `EBR` at 32 bits over 128
    /// bytes per stick is 2^42 bits, 2^38 elements at `f16` — so the saturation is unobservable and
    /// is here to make the conversion total rather than to change an answer.
    #[must_use]
    pub const fn beyond(self, limit: Elements) -> i64 {
        self.0.saturating_sub_unsigned(limit.0)
    }

    /// `max_mutable += weight` (`:738`, `:1278`).
    pub const fn add_weight(&mut self, weight: i64) {
        self.0 = self.0.saturating_add(weight);
    }

    /// The span, for the arithmetic that reads it.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// `dsc_global_->dcc_correct_ear_overflow` — WHETHER `dcc` MAY SPLIT AN OVERFLOWING TRANSFER AT ALL.
///
/// # ⛔⛔ OFF BY DEFAULT, WHICH MAKES THIS WHOLE PASS OPT-IN
///
/// `bool dcc_correct_ear_overflow = false;` under the comment *"Allows DCC to attempt to correct an
/// EAR overflow if encountered"* (`sys-arch-spec/dscglobal/dscglobal.h:92-93`), set from
/// `DT_OPT=correctearoverflow=1` (`dscglobal.cpp:280-281`). With it clear,
/// [`has_mutable_addr_overflow`] does not decline the split — it stops the compiler:
/// `DT_CHECK_MSG(false, "EAR overflow detected")` (`:807-808`), under *"Currently preventing
/// transformation of any EAR overflow cases and addressing them upstream unless manual override
/// specified."*
///
/// ⭐⭐ AND ALL FOUR OF THE VENDOR'S OWN TESTS PASS IT. Every `RUN` line under
/// `dcc/test/Transform/MutableAddrSplitting/` is `DT_OPT=correctearoverflow=1 dcc-opt
/// --dcc-mutable-addr-splitting …`, so the answer keys this port is checked against are all
/// [`EarOverflowCorrection::Allowed`] — and the shipped default is the other one.
///
/// ⛔ A PARAMETER, NOT A CONST, BECAUSE IT IS THE ONE THING A TEST MUST BE ABLE TO SAY. The two
/// `cl::opt` overrides above ([`MAX_MUTABLE_SIZE`]) are consts because nothing in this crate parses a
/// command line and their default IS their behaviour; this flag's non-default is what the entire
/// answer key runs under, so a caller states it. [`CORRECT_EAR_OVERFLOW`] is the shipped value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarOverflowCorrection {
    /// `dcc_correct_ear_overflow == false` — an overflow stops the compiler.
    Forbidden,
    /// `dcc_correct_ear_overflow == true`, i.e. `DT_OPT=correctearoverflow=1` — an overflow is split.
    Allowed,
}

/// THE SHIPPED DEFAULT — `dscglobal.h:93`.
pub const CORRECT_EAR_OVERFLOW: EarOverflowCorrection = EarOverflowCorrection::Forbidden;

/// WHETHER A TRANSFER'S MUTABLE ADDRESS HAS RUN PAST THE `EAR`, AND WHICH OF THE TWO `DT_CHECK`s
/// STANDS IN THE WAY OF SPLITTING IT.
///
/// ⛔ NOT AN ERROR TYPE. The reference returns `bool` and aborts on two paths; the aborts are facts
/// about the program — *"no loops involved in address calculation"*, *"EAR overflow detected"* — and
/// each is a shape of DataflowIR this island can hold. [`Self::overflowed`] is the reference's own
/// boolean, where it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutableAddrOverflow {
    /// `return false` — the address fits in the `EAR` and nothing is split.
    InRange,
    /// `return true` — it does not fit, and the pass will partition the transfer.
    Overflow,
    /// `DT_CHECK_MSG(false, "EAR overflow detected")` (`:808`) — it does not fit and
    /// [`EarOverflowCorrection::Forbidden`] says `dcc` may not try. See [`CORRECT_EAR_OVERFLOW`]: this
    /// is the SHIPPED answer to an overflow.
    OverflowCorrectionForbidden,
    /// `DT_CHECK_MSG(!mas_data.empty(), "Mutable address overflow detected, no loops involved in
    /// address calculation - cannot split data transfer.")` (`:810-813`).
    ///
    /// ⭐ SPLITTING MEANS PARTITIONING AN ITERATION SPACE, so a transfer with no iterators has no
    /// space to partition — the address it overflows by is a constant offset and there is nothing to
    /// cut. `initMASData` returns early on `indices.size() == 0` (`:713`), which is exactly how an
    /// empty `mas_data` reaches here.
    NoLoopsToSplit,
}

impl MutableAddrOverflow {
    /// THE REFERENCE'S OWN `bool`, where it returns one — [`None`] on the two aborts.
    #[must_use]
    pub const fn overflowed(self) -> Option<bool> {
        match self {
            MutableAddrOverflow::InRange => Some(false),
            MutableAddrOverflow::Overflow => Some(true),
            MutableAddrOverflow::OverflowCorrectionForbidden
            | MutableAddrOverflow::NoLoopsToSplit => None,
        }
    }
}

/// Replaces: e185_hasMutableAddrOverflow
///
/// **185/384** `MutableAddrSplittingPass::hasMutableAddrOverflow` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801` (17L).
///
/// ```cpp
/// bool MutableAddrSplittingPass::hasMutableAddrOverflow(
///     const SmallVectorImpl<MASData> &mas_data, const SenComponents comp,
///     const int64_t max_mutable, const int elem_size_in_bits) const {
///   if (max_mutable > (getMaxMutableRange(comp) / elem_size_in_bits)) {
///     // Currently preventing transformation of any EAR overflow cases and
///     // addressing them upstream unless manual override specified.
///     if (!dcc_ext_ctx_.dsc_global_->dcc_correct_ear_overflow)
///       DT_CHECK_MSG(false, "EAR overflow detected");
///
///     DT_CHECK_MSG(
///         !mas_data.empty(),
///         "Mutable address overflow detected, no loops involved in address "
///         "calculation - cannot split data transfer.");
///     return true;
///   }
///   return false;
/// }
/// ```
///
/// # ⭐⭐ THE GATE FOR THE WHOLE PASS, AND IT IS ASKED ONCE PER TRANSFER
///
/// `transformCompLoadAndStore` runs `initialize`, [`synthesize_time_info`], then
/// `if (!hasMutableAddrOverflow(...)) return;` (`:475-479`) — so a transfer whose address fits is left
/// exactly as it was, and everything downstream of here (the sort, the partition sizes, the
/// conditional tree, the explicit time loops) happens only for one that does not.
///
/// # ⛔ THE ORDER OF THE TWO CHECKS IS OBSERVABLE
///
/// The correction flag is tested FIRST. So a transfer that overflows with no loops at all reports
/// [`MutableAddrOverflow::OverflowCorrectionForbidden`] under the shipped default and
/// [`MutableAddrOverflow::NoLoopsToSplit`] only once correction is allowed — two different answers to
/// the same program, decided by a flag. Reversing them would report the more specific fact and be
/// wrong.
///
/// # Arguments
///
/// * `mas_data` — the transfer's loop iterators. Only its emptiness is read.
/// * `half` — which L3 half owns the `EAR` (see [`L3Half`]).
/// * `max_mutable` — the address span the transfer reaches, in elements.
/// * `elem` — the transfer's element format, which divides the register's range.
///   ⚠️ THE FORMAT RATHER THAN `elem_size_in_bits`: see [`AddrRange::elements`] for why the divisor
///   is named by a [`DataType`]. Bridging `getElementWidth()` to one is the caller's, and its caller
///   (`transformCompLoadAndStore`, entry 321) is not in this worklist.
/// * `correction` — `dcc_correct_ear_overflow`. See [`EarOverflowCorrection`].
#[must_use]
pub fn has_mutable_addr_overflow<A: Arch>(
    mas_data: &[MasData],
    half: L3Half,
    max_mutable: MutableAddr,
    elem: DataType,
    correction: EarOverflowCorrection,
) -> MutableAddrOverflow {
    // `if (max_mutable > (getMaxMutableRange(comp) / elem_size_in_bits))`.
    if !max_mutable.exceeds(max_mutable_range::<A>(half).elements(elem)) {
        return MutableAddrOverflow::InRange;
    }

    // `if (!dcc_ext_ctx_.dsc_global_->dcc_correct_ear_overflow) DT_CHECK_MSG(false, "EAR overflow
    // detected");` — first, and unconditional once the address has overflowed.
    if correction == EarOverflowCorrection::Forbidden {
        return MutableAddrOverflow::OverflowCorrectionForbidden;
    }

    // `DT_CHECK_MSG(!mas_data.empty(), "Mutable address overflow detected, no loops involved in
    // address calculation - cannot split data transfer.");`
    if mas_data.is_empty() {
        return MutableAddrOverflow::NoLoopsToSplit;
    }

    MutableAddrOverflow::Overflow
}

/// `-dcc-mutable-addr-splitting-max-conditionals`, `cl::init(16)` — `MutableAddrSplitting.cpp:70-75`.
///
/// ⛔⛔ ITS DEFAULT IS **16**, NOT `-1`. The two size overrides above default to their sentinel and so
/// never bind ([`MAX_MUTABLE_SIZE`]); this one is a real budget in every compile, and
/// *"use -1 for unlimited"* is the escape rather than the norm. A port that read the three flags as
/// one family would drop the only limit of the three that fires.
///
/// ⭐ `None` IS `-1`, as with the other two: *"A MaxNumConditionals value of -1 indicates no maximum"*
/// (`:956`).
pub const MAX_NUM_CONDITIONALS: Option<i64> = Some(16);

/// HOW MANY `scf.if`s THE PASS HAS ADDED TO THE PROGRAM SO FAR.
///
/// # ⛔⛔ PER PROGRAM, DESPITE ITS RESET SAYING PER UNIT
///
/// `runOnOperation` sets it to zero at the top of every unit — *"Reset num_conditionals_ so every unit
/// is not exceeding the max conditionals"* (`:244-246`) — but that reset is in the walk that only
/// COLLECTS candidates, and the counter's one writer, [`calculate_partition_sizes`] (`:958`), is
/// reached from the dispatch loop that runs after the whole walk. So the resets are inert and the
/// budget of [`MAX_NUM_CONDITIONALS`] is one total for the program, shared by every overflowing
/// transfer in it — see [`run_on_operation`].
///
/// ⚠️ AN `i64` WHERE THE REFERENCE'S COUNTER IS AN `int`. `num_conditionals_` is declared `int`
/// (`:222`) and compared against an `int64_t` flag (`:960`), so the comparison already promotes; the
/// wider type is the one both sides of it are read at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Conditionals(pub i64);

/// WHAT `calculatePartitionSizes` LEAVES BEHIND — its out-parameter and the three running totals its
/// own checks are stated in terms of.
///
/// ⭐ THE REFERENCE KEEPS THE LAST THREE LOCAL. `partition_sizes` is the only thing its callers read
/// ([`construct_conditionals`], [`calculate_subscripts_coefficients`],
/// [`create_explicit_time_loops`] all index it), and `shifted_mutable`, `total_partitions` and the
/// remaining `mutable_overflow` exist only to be checked. They are carried here because they are
/// exactly what the vendor's answer keys pin — a second view's start address IS `shifted_mutable`,
/// and the number of `scf.if`s IS `total_partitions - 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionSizes {
    /// `partition_sizes` — one entry per split dimension, in the SORTED order
    /// [`sort_data_based_on_weight`] left `mas_data` in, so `partition_sizes[i]` belongs to
    /// `mas_data[i]`.
    ///
    /// ⛔ SHORTER THAN `mas_data` IN THE NORMAL CASE. The walk `break`s at the first dimension heavy
    /// enough to absorb the whole overflow, so a two-dimensional transfer usually gets ONE entry —
    /// which is why every consumer bounds its own loop by `partition_sizes.size()` rather than by
    /// `mas_data.size()`.
    pub sizes: Vec<i64>,
    /// `total_partitions` — the product over the split dimensions. One conditional fewer than this is
    /// charged to [`Conditionals`].
    pub total_partitions: i64,
    /// `shifted_mutable` — how much address the split moves out of the mutable half into the
    /// immutable one, in elements. This is the offset the LAST partition's cloned memory view carries
    /// ([`create_new_mem_view_with_mod`]).
    pub shifted_mutable: i64,
    /// `mutable_overflow` as the walk left it — non-positive, which is the reference's own check.
    pub remaining_overflow: i64,
}

/// WHETHER THE TRANSFER CAN BE PARTITIONED, AND WHICH OF THE REFERENCE'S FIVE `DT_CHECK`s SAYS NOT.
///
/// ⛔ NOT AN ERROR TYPE — five aborts, each a fact about a program this island can hold. See
/// [`SplittingEligibility`] for the same treatment of the same pass's other gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partitioning {
    /// The split, and everything the reference computed to reach it.
    Split(PartitionSizes),

    /// `DT_CHECK_MSG(mutable_overflow <= 0, "Cannot split enough to bring mutable address back in
    /// range.")` (`:922-923`).
    ///
    /// ⭐ THE WALK RAN OUT OF DIMENSIONS. Every dimension was split to a partition size of 1 — one
    /// iteration each, the finest cut there is — and the address still does not fit. Carrying what is
    /// left says how much by.
    CannotSplitEnough {
        /// `mutable_overflow`, still positive.
        remaining_overflow: i64,
    },

    /// `DT_CHECK_MSG(!partition_sizes.empty(), "No partitions were identified.")` (`:924`).
    ///
    /// ⚠️ ONLY AN EMPTY `mas_data` REACHES THIS, and
    /// [`has_mutable_addr_overflow`] has already refused that case
    /// ([`MutableAddrOverflow::NoLoopsToSplit`]) before `setupForPartitioning` runs. The reference
    /// keeps the check anyway, and so does this: the argument is a slice, and a slice can be empty.
    NoPartitionsIdentified,

    /// `DT_CHECK_MSG(shifted_mutable <= immutable_space, "Shifting the required mutable would exceed
    /// the maximum immutable range.")` (`:928-930`).
    ///
    /// ⭐⭐ THIS IS WHY THERE ARE TWO RANGE QUERIES. Splitting does not remove address, it MOVES it:
    /// each partition's view starts further along, so what leaves the `EAR` arrives in the `EBR`
    /// ([`max_immutable_range`]). If the immutable half has no room, the split has nowhere to put it.
    ShiftExceedsImmutableRange {
        /// `shifted_mutable`.
        shifted_mutable: i64,
        /// `(getMaxImmutableRange(comp) / elem_size_in_bits) - max_immutable`.
        immutable_space: i64,
    },

    /// `DT_CHECK_MSG(immutable_space >= num_elems_in_stick, "No mutable space to shift back one stick
    /// to maintain even immutable address.")` (`:948-950`) — the SEN1P5-only parity rule.
    ///
    /// ⭐ FROM SEN1P5 AN L3 IMMUTABLE ADDRESS MUST BE AN EVEN NUMBER OF STICKS. The reference says so
    /// in the comment above the block (`:933-937`): if the shift would flip that parity, filling the
    /// partitions has to walk one stick back, and there must be room for the extra stick.
    NoRoomToRealignTheImmutableStick {
        /// `immutable_space`.
        immutable_space: i64,
        /// `num_elems_in_stick`.
        elems_in_stick: u64,
    },

    /// `DT_CHECK_MSG(MaxNumConditionals == -1 || num_conditionals_ <= MaxNumConditionals, "Required
    /// number of conditionals to partition exceeds max allowed conditionals.")` (`:959-962`).
    ///
    /// ⚠️ THE COUNTER IS ALREADY ADVANCED WHEN THIS IS REPORTED, because the reference adds before it
    /// checks (`:958`). Faithful, and harmless there because the check aborts; stated because here it
    /// does not.
    TooManyConditionals {
        /// `num_conditionals_` after this transfer's `req_conditionals` were added.
        total: Conditionals,
        /// `MaxNumConditionals`.
        limit: i64,
    },

    /// A dimension the walk had to divide by whose `composed_coeff_` is ZERO.
    ///
    /// # ⛔⛔ A NARROWING OF UNDEFINED BEHAVIOUR, NOT OF THE REFERENCE
    ///
    /// `extra_iters = mutable_overflow / mas_data[i].composed_coeff_` (`:905`) divides by a field
    /// nothing has constrained. It cannot be zero on the path the pass actually takes — the `else`
    /// branch is reached only when `weight_ > mutable_overflow >= 0`, and
    /// `weight_ = (num_iters - 2) * composed_coeff_`, so a zero coefficient gives a zero weight that
    /// takes the `if` instead — but the argument is a slice of [`MasData`] and the field is an `i64`,
    /// so the division is guarded rather than assumed. Compare
    /// [`LoopTripCount::ScfStepIsNotPositive`].
    DimensionHasNoCoefficient {
        /// `mas_data[i].dim_`.
        dim: u32,
    },

    /// A partition size of ZERO, which `num_iters_ % partition_sizes.back()` (`:911`) divides by.
    ///
    /// ⛔ ALSO UNREACHABLE ON THE PASS'S OWN PATH, for the same reason: with a positive coefficient
    /// and a non-negative overflow, `extra_iters <= num_iters - 2`, so the size is at least 2. Guarded
    /// because [`MasData`] is a plain struct. See [`Partitioning::DimensionHasNoCoefficient`].
    PartitionSizeIsZero {
        /// `mas_data[i].dim_`.
        dim: u32,
    },
}

/// `getBytesPerStick() * 8 / elem_size_in_bits` — HOW MANY ELEMENTS FILL ONE STICK.
///
/// ⭐ NEVER ZERO FOR ANY [`DataType`]. A stick is 1024 bits ([`Arch::BYTES_PER_STICK`] is 128 on every
/// generation and is a [`NonZeroU64`]) and an element is 4 to 32 bits (`DataType::bits`), so the
/// quotient is 32 to 256. The [`NonZeroU64::MIN`] fallback is what makes this total; no format can
/// reach it.
fn elems_in_stick<A: Arch>(elem: DataType) -> NonZeroU64 {
    NonZeroU64::new(A::BYTES_PER_STICK.get() * 8 / u64::from(elem.bits().0))
        .unwrap_or(NonZeroU64::MIN)
}

/// `<a count of elements> - <an address in elements>`, as the reference's `int64_t` subtraction.
///
/// ⛔ SATURATING FOR THE SAME REASON [`MutableAddr::beyond`] IS: the left operand is an unsigned count
/// and the difference is signed. No register this crate models makes the saturation reachable.
const fn elements_less_address(limit: Elements, addr: i64) -> i64 {
    0i64.saturating_add_unsigned(limit.0).saturating_sub(addr)
}

/// Replaces: e186_calculatePartitionSizes
///
/// **186/384** `MutableAddrSplittingPass::calculatePartitionSizes` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862` (102L).
///
/// ```cpp
/// // Note: This is not optimal at all but a more optimal solution requires a
/// // much more complex algorithm or an enumeration method. Use of external
/// // math solvers could help.
///
/// // For each dim, check if the weight of the dim is enough to bring the mutable
/// // back in range. If it is, calculate the split. If it is not, fully split the
/// // current dim and go to the next one.
/// int64_t max_mutable_range = getMaxMutableRange(comp);
/// int64_t mutable_overflow =
///     max_mutable - (max_mutable_range / elem_size_in_bits);
///
/// int64_t shifted_mutable = 0;
/// int64_t total_partitions = 1;
/// for (int i = 0, e = mas_data.size(); i < e; ++i) {
///   if (mas_data[i].weight_ <= mutable_overflow) {
///     partition_sizes.push_back(1);
///     mutable_overflow -= mas_data[i].weight_;
///     shifted_mutable += mas_data[i].weight_;
///     total_partitions *= mas_data[i].num_iters_;
///   } else {
///     int64_t extra_iters = mutable_overflow / mas_data[i].composed_coeff_;
///     if (mutable_overflow % mas_data[i].composed_coeff_ != 0) extra_iters += 1;
///     partition_sizes.push_back(mas_data[i].num_iters_ - extra_iters);
///     mutable_overflow -= extra_iters * mas_data[i].composed_coeff_;
///     int64_t num_partitions =
///         mas_data[i].num_iters_ % partition_sizes.back() == 0
///             ? mas_data[i].num_iters_ / partition_sizes.back()
///             : mas_data[i].num_iters_ / partition_sizes.back() + 1;
///     shifted_mutable += (num_partitions - 1) * partition_sizes.back() *
///                        mas_data[i].composed_coeff_;
///     total_partitions *= num_partitions;
///     break;
///   }
/// }
///
/// DT_CHECK_MSG(mutable_overflow <= 0,
///              "Cannot split enough to bring mutable address back in range.");
/// DT_CHECK_MSG(!partition_sizes.empty(), "No partitions were identified.");
///
/// auto max_immutable =
///     dcc::agen::utils::getMaxImmutableAddress(evaluator, mem_view_op);
/// auto immutable_space =
///     (getMaxImmutableRange(comp) / elem_size_in_bits) - max_immutable;
/// DT_CHECK_MSG(shifted_mutable <= immutable_space,
///              "Shifting the required mutable would exceed the maximum "
///              "immutable range.");
///
/// // For sen1p5, immutable addresses need to be in an even number of sticks
/// // only. If the immutable would become an odd number of sticks as result of
/// // splitting, there needs to be enough mutable space remaining to shift a
/// // stick back when filling the partitions.
/// if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA) {
///   int64_t num_elems_in_stick =
///       dcc_ext_ctx_.getBytesPerStick() * 8 / elem_size_in_bits;
///   bool is_even = dcc::agen::utils::isL3ImmutableAddrEven(
///       evaluator, mem_view_op.getStartAddress(), num_elems_in_stick);
///   bool is_shift_even = shifted_mutable % 2 == 0;
///   if (is_even != is_shift_even) {
///     DT_CHECK_MSG(is_even ? true
///                          : dcc::agen::utils::isL3ImmutableAddrAllOdd(
///                                evaluator, mem_view_op.getStartAddress(),
///                                num_elems_in_stick),
///                  "All immutable addrs must be all even or all odd to execute "
///                  "even shift.");
///     DT_CHECK_MSG(immutable_space >= num_elems_in_stick,
///                  "No mutable space to shift back one stick to maintain even "
///                  "immutable address.");
///   }
/// }
///
/// // The number of conditionals required will be one less than the total number
/// // of partitions. A MaxNumConditionals value of -1 indicates no maximum.
/// int req_conditionals = total_partitions - 1;
/// num_conditionals_ += req_conditionals;
/// DT_CHECK_MSG(
///     MaxNumConditionals == -1 || num_conditionals_ <= MaxNumConditionals,
///     "Required number of conditionals to partition exceeds max allowed "
///     "conditionals.");
/// ```
///
/// # ⭐⭐ THE GREEDY WALK, AND WHY IT LOOKS AT ONE DIMENSION IN PRACTICE
///
/// The overflow is a number of elements the address runs past the `EAR`. Cutting a dimension into
/// partitions of `p` iterations removes `(num_iters - p) * coeff` from the span it reaches, so the
/// question per dimension is *"is this dimension's whole weight enough?"*:
///
/// | branch | condition | what it does |
/// |---|---|---|
/// | `if` | `weight <= mutable_overflow` | split it as far as it goes — partition size **1**, one iteration each — subtract the whole weight and CONTINUE |
/// | `else` | `weight > mutable_overflow` | compute the smallest cut that absorbs what is left, and **`break`** |
///
/// ⭐ SO THE HEAVIEST DIMENSION USUALLY ENDS THE WALK ON ITS OWN — which is why
/// [`sort_data_based_on_weight`] runs first, and why `partition_sizes` is one entry long in all four
/// of the vendor's answer keys. `mas_data[i].dim_` is what ties entry `i` back to a subscript.
///
/// # ⭐ `extra_iters` IS A CEILING DIVISION, WRITTEN AS TWO STATEMENTS
///
/// `mutable_overflow / coeff`, plus one if the division left a remainder — the smallest number of
/// iterations whose address contribution covers the overflow. `partition_size = num_iters -
/// extra_iters`, and `num_partitions` is a second ceiling division of `num_iters` by that.
///
/// # ⭐⭐ THE FIVE ANSWER KEYS, EACH ONE LINE OF ARITHMETIC
///
/// | test | limit | `max_mutable` | overflow | heaviest dim | `sizes` | `total_partitions` | `shifted_mutable` |
/// |---|---|---|---|---|---|---|---|
/// | `one_dim/constant_start_addr_1` | 5700 | 12672 | 6972 | 2048 × 8 | `[4]` | 2 | 8192 |
/// | `one_dim/select_ub` | 5700 | 14336 | 8636 | 2048 × 8 | `[3]` | 3 | 12288 |
/// | `multi_dim/constant_start_addr_1` | 4687 | 10304 | 5617 | 1024 × 8 | `[2]` | 4 | 6144 |
/// | `time_dims/constant_start_addr` | 37500 | 43136 | 5636 | 2048 × 16 | `[13]` | 2 | 26624 |
/// | `one_dim_sen1p5/constant_start_addr_1` | 6250 | 14400 | 8150 | 2048 × 8 | `[4]` | 2 | 8192 |
///
/// ⭐ THE LAST ONE IS THE ONLY TEST OF THE SEN1P5 PARITY BLOCK, and IBM keeps it in a separate file for
/// exactly that reason — *"Tests that odd EBRs are properly handled in MutableAddrSplittingPass"*
/// (`mutable_addr_splitting_one_dim_sen1p5.mlir:6`). Its start address of 2112 is an ODD 33 sticks and
/// its shift is even, so the block runs; see
/// `the_sen1p5_answer_key_takes_the_parity_path_and_passes` below.
///
/// Each `sizes` is the `arith.constant` the vendor's `CHECK-SENT-IR` compares an induction variable
/// against, each `total_partitions - 1` is the number of nested `scf.if`s, and each
/// `shifted_mutable` divided by `total_partitions - 1` is the step between the partitions' view start
/// addresses.
///
/// # ⛔ THE SEN1P5 PARITY `DT_CHECK` CANNOT FIRE ON A CONSTANT START, AND IT IS STILL PORTED
///
/// `isL3ImmutableAddrEven` and `isL3ImmutableAddrAllOdd` are the same walk under two predicates —
/// `isDivisibleBy(2)` and `isAllOdd()` — over `evaluateDivideByConst(ev, num_elems_in_stick)`
/// (`Dialect/Agen/Utils.cpp:218-288`). For an `arith.constant` start that evaluation is a SINGLE
/// value, so the two predicates are exact complements: `!is_even` implies all-odd, and
/// `is_even ? true : isL3ImmutableAddrAllOdd(...)` is true either way. The check exists for the
/// toggle and conditional-tree start addresses the OTHER callers of those helpers pass, which
/// [`is_eligible_for_splitting`] has already excluded here — so there is no variant for it and this
/// is the record of why.
///
/// # ⚠️ WHAT IS MECHANISM
///
/// `evaluator` is a memoised evaluation of an SSA value this crate reads as a literal, and
/// `getMaxImmutableAddress(evaluator, mem_view_op)` collapses to `view.start` for an `arith.constant`
/// start: its first arm is `getMinMax({&const_ev}, false)` over a single `AllUnitEvaluatedValue`,
/// which is that constant (`Utils.cpp:295-298`, `ExpressionEvaluatorUtils.cpp:631-669`). Taking a
/// [`ConstStartMemView`] is what makes that collapse checkable rather than assumed.
///
/// # Arguments
///
/// * `mas_data` — the transfer's iterators, ALREADY SORTED by [`sort_data_based_on_weight`]. The
///   reference takes it mutably and does not modify it here.
/// * `half` — which L3 half owns the two registers (see [`L3Half`]).
/// * `view` — the memory view being split, whose start address is `max_immutable`.
/// * `max_mutable` — the span the transfer reaches, in elements.
/// * `elem` — the element format, which divides both ranges. See [`AddrRange::elements`].
/// * `num_conditionals` — the per-unit running count, advanced by `total_partitions - 1`. See
///   [`Conditionals`].
pub fn calculate_partition_sizes<A: Arch>(
    mas_data: &[MasData],
    half: L3Half,
    view: &ConstStartMemView<'_>,
    max_mutable: MutableAddr,
    elem: DataType,
    num_conditionals: &mut Conditionals,
) -> Partitioning {
    // `int64_t max_mutable_range = getMaxMutableRange(comp);`
    // `int64_t mutable_overflow = max_mutable - (max_mutable_range / elem_size_in_bits);`
    let mut mutable_overflow = max_mutable.beyond(max_mutable_range::<A>(half).elements(elem));

    let mut sizes: Vec<i64> = Vec::new();
    let mut shifted_mutable: i64 = 0;
    let mut total_partitions: i64 = 1;
    for dim in mas_data {
        if dim.weight <= mutable_overflow {
            // *"This dim doesn't have enough weight to fully resolve the required shift. Set
            // partition size 1 for this dim and continue splitting."*
            sizes.push(1);
            mutable_overflow -= dim.weight;
            // *"Entire weight is shifted."*
            shifted_mutable += dim.weight;
            // *"Each iteration is a partition."*
            total_partitions = total_partitions.saturating_mul(dim.num_iters);
            continue;
        }

        // *"This dim has enough weight to fully resolve the required shift. Calculate how many
        // partitions are required and then stop splitting."*
        if dim.composed_coeff == 0 {
            return Partitioning::DimensionHasNoCoefficient { dim: dim.dim };
        }
        // `extra_iters = mutable_overflow / coeff`, `+ 1` on a remainder — a ceiling division.
        let mut extra_iters = mutable_overflow / dim.composed_coeff;
        if mutable_overflow % dim.composed_coeff != 0 {
            extra_iters += 1;
        }
        let partition_size = dim.num_iters - extra_iters;
        sizes.push(partition_size);
        mutable_overflow -= extra_iters * dim.composed_coeff;

        if partition_size == 0 {
            return Partitioning::PartitionSizeIsZero { dim: dim.dim };
        }
        // `num_iters_ % partition_sizes.back() == 0 ? num_iters_ / back : num_iters_ / back + 1`.
        let num_partitions = if dim.num_iters % partition_size == 0 {
            dim.num_iters / partition_size
        } else {
            dim.num_iters / partition_size + 1
        };
        // *"When splitting a dim into x partitions, (x-1) partitions will need adjustment. Each
        // adjusted partition will shift the mutable over by <partition size> * <coeff>."*
        shifted_mutable += (num_partitions - 1) * partition_size * dim.composed_coeff;
        total_partitions = total_partitions.saturating_mul(num_partitions);
        break;
    }

    // `DT_CHECK_MSG(mutable_overflow <= 0, "Cannot split enough to bring mutable address back in
    // range.");`
    if mutable_overflow > 0 {
        return Partitioning::CannotSplitEnough {
            remaining_overflow: mutable_overflow,
        };
    }
    // `DT_CHECK_MSG(!partition_sizes.empty(), "No partitions were identified.");`
    if sizes.is_empty() {
        return Partitioning::NoPartitionsIdentified;
    }

    // `auto max_immutable = getMaxImmutableAddress(evaluator, mem_view_op);` — the constant itself.
    // `auto immutable_space = (getMaxImmutableRange(comp) / elem_size_in_bits) - max_immutable;`
    let immutable_space =
        elements_less_address(max_immutable_range::<A>(half).elements(elem), view.start);
    // `DT_CHECK_MSG(shifted_mutable <= immutable_space, …);`
    if shifted_mutable > immutable_space {
        return Partitioning::ShiftExceedsImmutableRange {
            shifted_mutable,
            immutable_space,
        };
    }

    // `if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA)` — [`IsaGen`] is ordered, so this is the
    // same comparison.
    if A::GEN >= IsaGen::Sen1p5 {
        let stick = elems_in_stick::<A>(elem);
        // `isL3ImmutableAddrEven(evaluator, getStartAddress(), num_elems_in_stick)` —
        // `evaluateDivideByConst(ev, n).isDivisibleBy(2)` over the single constant start.
        let is_even = (view.start / stick.get().cast_signed()) % 2 == 0;
        let is_shift_even = shifted_mutable % 2 == 0;
        if is_even != is_shift_even {
            // The `is_even ? true : isL3ImmutableAddrAllOdd(...)` check sits here and cannot fire on a
            // constant start — see this function's own doc for the reason it is absent.
            //
            // `DT_CHECK_MSG(immutable_space >= num_elems_in_stick, …);`
            if immutable_space < 0i64.saturating_add_unsigned(stick.get()) {
                return Partitioning::NoRoomToRealignTheImmutableStick {
                    immutable_space,
                    elems_in_stick: stick.get(),
                };
            }
        }
    }

    // `int req_conditionals = total_partitions - 1; num_conditionals_ += req_conditionals;` — the
    // counter advances BEFORE the check, which is the order kept here.
    num_conditionals.0 = num_conditionals.0.saturating_add(total_partitions - 1);
    // `DT_CHECK_MSG(MaxNumConditionals == -1 || num_conditionals_ <= MaxNumConditionals, …);`
    if let Some(limit) = MAX_NUM_CONDITIONALS {
        if num_conditionals.0 > limit {
            return Partitioning::TooManyConditionals {
                total: *num_conditionals,
                limit,
            };
        }
    }

    Partitioning::Split(PartitionSizes {
        sizes,
        total_partitions,
        shifted_mutable,
        remaining_overflow: mutable_overflow,
    })
}

/// WHAT `constructConditionals` BUILT, OR WHY IT COULD NOT.
///
/// ⛔ NOT AN ERROR TYPE — see [`Partitioning`]. Two of the four answers below are shapes the reference
/// reaches by reading past the end of an empty `SmallVector`, which is why they are stated rather than
/// assumed away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionalTree {
    /// The ops that go where the transfer was, outermost first — an `arith.constant`, an
    /// `arith.cmpi slt` and the root `scf.if`, whose regions hold the rest of the tree.
    ///
    /// ⭐⭐ THE WHOLE TREE IS IN THE LAST OP, BY VALUE. The reference returns one `scf::IfOp` handle
    /// and everything else is reachable from it through the region it was built into;
    /// [`scf::Op::If`]'s two `Vec<Op>` regions hold their nested conditionals directly, so the root op
    /// IS the tree and there is nothing to hand back beside it.
    Built(Vec<DfirOp>),

    /// `DT_CHECK(!partition_sizes.empty())` — `createPartitions`, `:989`.
    NoPartitionsToBuild,

    /// `DT_CHECK(mas_data[partition_dim].iter_arg_ != nullptr)` (`:1015`).
    ///
    /// ⭐⭐ AND THIS IS WHY [`create_explicit_time_loops`] RUNS FIRST. A time dimension has no
    /// induction variable to compare against until that function has written the `affine.for` that
    /// carries it (`:491` runs before `:533`'s `createPartitions`); an entry still holding `None` here
    /// is a partitioned time dimension whose loop was never made explicit.
    DimensionHasNoIterator {
        /// `mas_data[i].dim_`.
        dim: u32,
    },

    /// A DIMENSION WHOSE PARTITION SIZE COVERS ALL OF ITS ITERATIONS, so `while (ub < num_iters_)`
    /// never runs and its chain of conditionals is empty.
    ///
    /// # ⛔⛔ TWO READS PAST THE END OF AN EMPTY `SmallVector`
    ///
    /// `prev_partitions.front()` takes the root from the FIRST dimension's chain (`:1039`) and
    /// `prev_partitions.back()` takes the lineage that continues (`:1052`) — both undefined on an
    /// empty one. ⭐ Only the INNERMOST dimension may have an empty chain safely, and only because
    /// nothing looks at it afterwards; that shape is [`ConditionalTree::Built`], with one fewer level
    /// than there are entries in `partition_sizes`.
    ///
    /// ⭐ REACHABLE ONLY THROUGH A DIMENSION OF ONE ITERATION. Every partition size
    /// [`calculate_partition_sizes`] returns is strictly below its `num_iters_` — `extra_iters` is at
    /// least 1 in the dividing branch — except the flat `1` the whole-weight branch pushes, which
    /// equals `num_iters_` when the dimension runs once. A one-iteration dimension has a NEGATIVE
    /// weight (`(1 - 2) * coeff`), so it sorts last and is reached only when the overflow outlasts
    /// every dimension above it.
    DimensionNeedsNoConditional {
        /// `mas_data[i].dim_`.
        dim: u32,
    },

    /// A PARTITION SIZE OF ZERO OR LESS, WHICH `ub += partition_sizes[partition_dim]` NEVER ESCAPES.
    ///
    /// ⛔⛔ A NON-TERMINATING LOOP IN THE REFERENCE, NOT A WRONG ANSWER. `while (ub < num_iters_)`
    /// with a step of zero appends `scf.if`s until the process dies. [`calculate_partition_sizes`]
    /// cannot return one ([`Partitioning::PartitionSizeIsZero`]), and this function takes the sizes as
    /// a slice — so the bound is stated here rather than trusted across the two.
    PartitionSizeIsNotPositive {
        /// `mas_data[i].dim_`.
        dim: u32,
        /// `partition_sizes[i]`.
        size: i64,
    },

    /// MORE PARTITION SIZES THAN ITERATORS.
    ///
    /// ⛔ `mas_data[partition_dim]` INDEXED BY A `partition_sizes` POSITION (`:1010-1011`), where the
    /// reference's own construction keeps the two aligned: [`calculate_partition_sizes`] pushes one
    /// size per `mas_data` entry it walks and stops. A caller that pairs the wrong two lists reads a
    /// neighbouring iterator in C++ and gets this instead.
    MorePartitionsThanIterators {
        /// `partition_sizes.len()`.
        sizes: usize,
        /// `mas_data.size()`.
        iterators: usize,
    },
}

impl ConditionalTree {
    /// THE TREE, WHERE THERE IS ONE.
    #[must_use]
    pub fn ops(&self) -> Option<&[DfirOp]> {
        match self {
            ConditionalTree::Built(ops) => Some(ops),
            ConditionalTree::NoPartitionsToBuild
            | ConditionalTree::DimensionHasNoIterator { .. }
            | ConditionalTree::DimensionNeedsNoConditional { .. }
            | ConditionalTree::PartitionSizeIsNotPositive { .. }
            | ConditionalTree::MorePartitionsThanIterators { .. } => None,
        }
    }
}

/// THE TWO VALUES ONE CONDITIONAL MINTS: the boundary constant and the comparison against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cond {
    /// `arith::ConstantIndexOp::create(cond_builder, op->getLoc(), ub)`.
    bound: Val,
    /// `arith::CmpIOp::create(.., slt, iter_arg_, ub_const)` — the `scf.if`'s predicate.
    cond: Val,
}

/// ONE DIMENSION'S CONDITIONALS, EVERY CHAIN OF THEM, AND WHICH CHAINS THE NEXT DIMENSION DESCENDS
/// INTO.
///
/// # ⛔⛔ THE REFERENCE'S `prev_partitions` BOOKKEEPING, WHICH IS NOT WHAT ITS COMMENT SAYS
///
/// `constructConditionals` walks the dimensions outermost first, building each one's chain inside the
/// PREVIOUS dimension's partitions. Which partitions receive a chain, and which of those go on to
/// receive the NEXT dimension's, is decided by four lines (`:1044-1058`):
///
/// ```cpp
/// for (int p = 0; p < num_prev_partitions; ++p) {
///   auto curr_partition = prev_partitions[p];
///   cond_builder = curr_partition.getThenBodyBuilder();
///   curr_partitions = createConditionalsForPartition(d);   // ⛔ ASSIGNED, NOT APPENDED
/// }
/// cond_builder = prev_partitions.back().getElseBodyBuilder();
/// auto last_partition = createConditionalsForPartition(d);
/// prev_partitions = curr_partitions;
/// for (auto &o : last_partition) prev_partitions.push_back(o);
/// ```
///
/// A chain IS built into every previous partition's `then` body — but `curr_partitions` keeps only the
/// LAST one's, so the next dimension descends into the last previous partition alone. ⛔ Every other
/// branch stops one dimension short: the transfer copied into it is guarded on the outer dimensions
/// and unguarded on the inner ones, which is the wrong address for every iteration but the first.
///
/// ⛔⛔ PORTED AS WRITTEN. Two or more entries in `partition_sizes` means the whole-weight branch of
/// [`calculate_partition_sizes`] ran, which sets the outer sizes to **1** — so those chains are
/// `num_iters - 1` long and the branches that lose the inner dimensions are the majority. ⭐ NONE OF
/// THE FIVE ANSWER KEYS REACHES IT: every one partitions exactly ONE dimension, where
/// `prev_partitions` is used once and the bookkeeping cannot be observed. The reference is the
/// specification; a port that quietly appended instead of assigning would emit a different program
/// from `dcc` for inputs neither of them is tested on.
///
/// ⛔ AND `fillPartitions` AGREES WITH THE DEFECT, WHICH IS WHY IT CANNOT BE FIXED HERE ALONE. Its
/// walk from a leaf takes exactly `partition_sizes.size()` steps up the tree (`:1119-1167`), two
/// `getParentNode()` calls per partitioned dimension — so it reads one conditional per dimension in
/// the order it expects to find them. Appending here would deepen the branches it walks and the two
/// would disagree about which dimension a conditional belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Level {
    /// `ub`, `2 * ub`, … — the boundaries below this dimension's trip count, one conditional each. The
    /// same for every chain of the dimension, because they all split it the same way.
    bounds: Vec<i64>,
    /// `mas_data[d].iter_arg_` — the induction variable every one of them compares.
    iter_arg: Val,
    /// Every chain built for this dimension, in construction order: chain `i` sits in the `then` body
    /// of the previous level's `prev[i]`, and the last chain sits in `prev.back()`'s `else` body.
    chains: Vec<Vec<Cond>>,
    /// `prev_partitions` AS THIS LEVEL LEAVES IT — `(chain, position)` for each conditional, in order.
    /// The next level builds one chain per entry, plus one in the `else` of the last.
    prev: Vec<(usize, usize)>,
}

/// Replaces: e187_constructConditionals
///
/// **187/384** `MutableAddrSplittingPass::constructConditionals` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000` (62L).
///
/// ```cpp
/// SmallVector<scf::IfOp, 32> previous_leaves;
/// scf::IfOp root_op = nullptr;
/// OpBuilder cond_builder(op);
///
/// auto createConditionalsForPartition =
///     [&](int partition_dim) -> SmallVector<scf::IfOp, 32> {
///   SmallVector<scf::IfOp, 32> if_ops;
///   int64_t ub = partition_sizes[partition_dim];
///   while (ub < mas_data[partition_dim].num_iters_) {
///     // Create the cmpi operation for this partition.
///     auto ub_const =
///         arith::ConstantIndexOp::create(cond_builder, op->getLoc(), ub);
///     DT_CHECK(mas_data[partition_dim].iter_arg_ != nullptr);
///     auto ub_cond = arith::CmpIOp::create(
///         cond_builder, op->getLoc(), arith::CmpIPredicate::slt,
///         mas_data[partition_dim].iter_arg_, ub_const);
///
///     // Create the if operation for this partition.
///     if_ops.emplace_back(scf::IfOp::create(cond_builder, op->getLoc(),
///                                           ub_cond.getResult(), true));
///
///     // Set the builder to the else block of the new ifOp.
///     cond_builder = if_ops.back().getElseBodyBuilder();
///
///     // Set the ub for the next iteration.
///     ub += partition_sizes[partition_dim];
///   }
///   return if_ops;
/// };
///
/// // Create the partitions for the first dim.
/// auto prev_partitions = createConditionalsForPartition(0);
///
/// // Store the root operation for the conditional tree. It will be used to
/// // generate a ConditionalTree later to insert the new operations for each
/// // partition.
/// auto root_if_op = prev_partitions.front();
///
/// // Create the partitions for the remaining dims in each previous partition.
/// for (int d = 1, e = partition_sizes.size(); d < e; ++d) {
///   SmallVector<scf::IfOp, 32> curr_partitions;
///   for (int p = 0, num_prev_partitions = prev_partitions.size();
///        p < num_prev_partitions; ++p) {
///     auto curr_partition = prev_partitions[p];
///     cond_builder = curr_partition.getThenBodyBuilder();
///     curr_partitions = createConditionalsForPartition(d);   // ⛔ ASSIGNS, NOT APPENDS
///   }
///   cond_builder = prev_partitions.back().getElseBodyBuilder();
///   auto last_partition = createConditionalsForPartition(d);
///   prev_partitions = curr_partitions;
///   for (auto &o : last_partition) prev_partitions.push_back(o);
/// }
///
/// return root_if_op;
/// ```
///
/// # ⭐⭐ A CHAIN OF `slt` COMPARISONS, EACH NESTED IN THE PREVIOUS ONE'S `else`
///
/// One dimension split into partitions of `p` iterations needs a conditional at every boundary below
/// its trip count — `p`, `2p`, `3p`, … — and the last partition is what is left when all of them are
/// false. So `while (ub < num_iters_)` produces `ceil(num_iters / p) - 1` conditionals for
/// `ceil(num_iters / p)` partitions, which is exactly the `total_partitions - 1` that
/// [`calculate_partition_sizes`] charged to the [`Conditionals`] budget.
///
/// ⭐⭐ IBM'S `select_ub` ANSWER KEY IS THIS SHAPE, PRINTED: eight iterations in partitions of three
/// give boundaries at **3** and **6**, and the second conditional sits inside the first one's `else`.
///
/// ```text
/// %21 = arith.constant 3 : index
/// %22 = arith.cmpi slt, %18, %21 : index
/// scf.if %22 {
///   ..                                    // iterations 0, 1, 2
/// } else {
///   %26 = arith.constant 6 : index
///   %27 = arith.cmpi slt, %18, %26 : index
///   scf.if %27 {
///     ..                                  // iterations 3, 4, 5
///   } else {
///     ..                                  // iterations 6, 7
///   }
/// }
/// ```
/// (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:210-231`)
///
/// ⛔ THE BOUND IS AN OPERAND, NOT A LITERAL. `arith.cmpi` takes two SSA values, so each boundary is
/// an `arith.constant` of its own — see [`arith::Op::Compare`]. Both ops land in the same region as
/// the `scf.if` they guard, which is why they are in the returned list and inside each `else` body.
///
/// ⭐ AND THE ORDER IS THE REFERENCE'S: a whole dimension's chain is minted before any dimension
/// inside it, because `createConditionalsForPartition` runs to completion before the loop over
/// `prev_partitions` descends. That is level order, not depth-first — see [`Level`] — and it is what
/// makes the printed SSA names line up with the vendor's, where the outer conditional's constant is
/// `%21` and the inner one's is `%26`.
///
/// # ⭐ AN EMPTY LEAF IS A PARTITION, AND THAT IS THE WHOLE OUTPUT
///
/// Every conditional is created with both regions (`scf::IfOp::create(.., /*withElseRegion=*/true)`),
/// and the regions this function leaves EMPTY are the partitions `fillPartitions` then fills with a
/// copy of the transfer at a shifted address. A region holding nothing but its own elided
/// `scf.yield` — see [`scf::Op::If::else_body`], where an empty `Vec` means NO BLOCK, a different op —
/// is that leaf.
///
/// ⚠️ `dcc::ConditionalTree` IS MECHANISM. The `CondNode` graph `createPartitions` computes over the
/// built region (`:993-994`) is how a caller finds those leaves through an MLIR region's parent and
/// child pointers; a [`scf::Op::If`] holds its regions by value, so a consumer reaches a leaf by
/// matching the tree it was handed.
///
/// # ⛔ WHAT ELSE IS MECHANISM
///
/// `OpBuilder cond_builder(op)` and the `getElseBodyBuilder()`/`getThenBodyBuilder()` retargeting are
/// an insertion point walking a region under construction; assembling the value tree says the same
/// thing. `previous_leaves` and `root_op` are declared at the top of the reference and never written
/// or read — `root_if_op` is a third, separate local.
///
/// # Arguments
///
/// * `vals` — the SSA namer, for the boundary constants and the comparisons.
/// * `mas_data` — the transfer's iterators, sorted by weight, with the time dimensions' induction
///   variables already filled in by [`create_explicit_time_loops`].
/// * `partition_sizes` — [`PartitionSizes::sizes`], one entry per split dimension, aligned with
///   `mas_data`.
pub fn construct_conditionals(
    vals: &mut Values,
    mas_data: &[MasData],
    partition_sizes: &[i64],
) -> ConditionalTree {
    if partition_sizes.is_empty() {
        return ConditionalTree::NoPartitionsToBuild;
    }
    if partition_sizes.len() > mas_data.len() {
        return ConditionalTree::MorePartitionsThanIterators {
            sizes: partition_sizes.len(),
            iterators: mas_data.len(),
        };
    }

    // ── The conditionals, dimension by dimension, in the order the reference mints them ──────────
    let mut levels: Vec<Level> = Vec::new();
    for (index, (dim, &size)) in mas_data.iter().zip(partition_sizes).enumerate() {
        if size <= 0 {
            return ConditionalTree::PartitionSizeIsNotPositive { dim: dim.dim, size };
        }
        let Some(iter_arg) = dim.iter_arg else {
            return ConditionalTree::DimensionHasNoIterator { dim: dim.dim };
        };

        // `int64_t ub = partition_sizes[..]; while (ub < num_iters_) { .. ub += partition_sizes[..] }`
        let bounds: Vec<i64> = std::iter::successors(Some(size), |ub| Some(ub + size))
            .take_while(|ub| *ub < dim.num_iters)
            .collect();
        let is_innermost = index + 1 == partition_sizes.len();
        if bounds.is_empty() && (index == 0 || !is_innermost) {
            // `prev_partitions.front()` (`:1039`) and `.back()` (`:1052`) on an empty vector.
            return ConditionalTree::DimensionNeedsNoConditional { dim: dim.dim };
        }

        // One chain per previous partition, plus one in the last previous partition's `else` body.
        let count = levels.last().map_or(1, |prev| prev.prev.len() + 1);
        let chains: Vec<Vec<Cond>> = (0..count)
            .map(|_| {
                bounds
                    .iter()
                    .map(|_| Cond {
                        bound: vals.mint(),
                        cond: vals.mint(),
                    })
                    .collect()
            })
            .collect();

        // `prev_partitions = curr_partitions; for (auto &o : last_partition) push_back(o);` — the
        // chain in the last previous partition's `then`, then the one in its `else`. See [`Level`].
        let live: Vec<usize> = match count {
            1 => vec![0],
            _ => vec![count - 2, count - 1],
        };
        let prev = live
            .iter()
            .flat_map(|&chain| (0..chains[chain].len()).map(move |at| (chain, at)))
            .collect();
        let empty = bounds.is_empty();
        levels.push(Level {
            bounds,
            iter_arg,
            chains,
            prev,
        });
        if empty {
            // An innermost dimension whose partition size covers it contributes no conditional, and
            // nothing reads the empty `prev_partitions` it leaves behind.
            break;
        }
    }

    ConditionalTree::Built(assemble_conditionals(&levels, 0, 0))
}

/// ONE CHAIN, ASSEMBLED FROM ITS INNERMOST CONDITIONAL OUTWARD, WITH THE DIMENSIONS INSIDE IT IN
/// PLACE.
///
/// The ops of chain `chain` at level `level`, without the enclosing region's terminator. Each
/// conditional's `then` body holds the next dimension's chain if this one is in `prev_partitions`, and
/// the LAST conditional of the last such chain holds one in its `else` body too — the reference's
/// `prev_partitions.back().getElseBodyBuilder()`.
fn assemble_conditionals(levels: &[Level], level: usize, chain: usize) -> Vec<DfirOp> {
    let (Some(here), Some(conds)) = (
        levels.get(level),
        levels.get(level).and_then(|here| here.chains.get(chain)),
    ) else {
        return Vec::new();
    };

    // The chain is built inside out, because each conditional holds the next in its `else` region.
    let mut nested: Vec<DfirOp> = Vec::new();
    for (at, cond) in conds.iter().enumerate().rev() {
        // ⭐ THE NEXT DIMENSION GOES IN THE `then` BODY OF EVERY CONDITIONAL IN `prev_partitions` —
        // and in the `else` BODY of the last one only.
        let position = here.prev.iter().position(|&slot| slot == (chain, at));
        let then_body = match position {
            Some(slot) => assemble_conditionals(levels, level + 1, slot),
            None => Vec::new(),
        };
        let else_body = match (at + 1 == conds.len(), here.prev.last()) {
            (true, Some(&last)) if last == (chain, at) => {
                assemble_conditionals(levels, level + 1, here.prev.len())
            }
            _ => nested,
        };

        nested = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: cond.bound,
                value: here.bounds.get(at).copied().unwrap_or_default(),
            }),
            DfirOp::Arith(arith::Op::Compare {
                result: cond.cond,
                predicate: arith::CmpIPredicate::Slt,
                lhs: here.iter_arg,
                rhs: cond.bound,
            }),
            DfirOp::Scf(scf::Op::If {
                cond: cond.cond,
                // `scf::IfOp::create(builder, loc, cond, /*withElseRegion=*/true)` (`:1197`) — the
                // three-argument overload, which is the RESULTLESS one. A partition's two arms
                // rewrite the transfer's operands in place; neither yields a value.
                results: Vec::new(),
                body: with_terminator(then_body),
                else_body: with_terminator(else_body),
                // ⭐ `dbg_name` IS SET BY THE CONDITIONAL-TREE PASS, NOT BY THIS ONE.
                // `constructConditionals` attaches no attribute to the `scf.if` it builds.
                dbg_name: None,
            }),
        ];
    }
    nested
}

/// A REGION HOLDING `ops` AND THE `scf.yield` MLIR'S OWN BUILDER PUTS THERE.
///
/// ⛔⛔ WHAT MAKES THE `else` REGION EXIST. `scf::IfOp::create(.., /*withElseRegion=*/true)` builds
/// both regions with a terminator in each, and an empty `Vec` in this island means the block is
/// ABSENT — see [`scf::Op::If::else_body`]. So the leaf of a conditional tree is a region whose only
/// op is this terminator, which is elided when printed because the op binds nothing.
fn with_terminator(mut ops: Vec<DfirOp>) -> Vec<DfirOp> {
    ops.push(DfirOp::Scf(scf::Op::Yield {
        operands: Vec::new(),
    }));
    ops
}

/// THE COEFFICIENT OF ONE SPLIT DIMENSION IN EVERY SUBSCRIPT OF A TRANSFER.
///
/// One row of the reference's `SmallVector<SmallVector<int64_t>>`, with the dimension it belongs to
/// attached. ⭐ THE `dim` IS NOT DECORATION: the reference recovers it by re-indexing
/// `mas_data[p].dim_` at every use, and `fillPartitions` pairs a row with a `mas_data` entry
/// (`:1119`, `:1148`, `:1151`) — a row carrying its own dimension cannot be paired with the wrong one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimCoefficients {
    /// `mas_data[p].dim_` — the position in the subscripts map this row is about.
    pub dim: u32,
    /// One coefficient per RESULT of the subscripts map, in order: how far that subscript moves per
    /// iteration of `dim`.
    pub coeffs: Vec<i64>,
}

/// WHAT `calculateSubscriptsCoefficients` EXTRACTED, OR WHY IT COULD NOT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscriptsCoefficients {
    /// One row per split dimension, in `partition_sizes` order.
    Extracted(Vec<DimCoefficients>),

    /// ⛔⛔ A SCALED DIMENSION WHOSE SCALE IS NOT A CONSTANT.
    ///
    /// `dim_coeffs.back() = cast<AffineConstantExpr>(bin_expr.getRHS()).getValue();`
    /// (`Dialect/Agen/Utils.cpp:536-537`) — an unchecked `cast`, which aborts on a `Mul` by anything
    /// else.
    ///
    /// ⭐ MLIR'S OWN CANONICALISATION IS WHY THE REFERENCE GETS AWAY WITH IT: `simplifyMul` puts the
    /// constant operand on the RIGHT, and a product of two dimensions is not an affine expression at
    /// all. What is left is the SEMI-AFFINE product `d0 * s0` — legal, expressible in this island (see
    /// [`AffineExpr::Sym`]), and a coefficient no `fillPartitions` subtraction can use.
    CoefficientIsNotConstant {
        /// `mas_data[p].dim_`.
        dim: u32,
        /// Which RESULT of the subscripts map holds it.
        result: usize,
    },
}

impl SubscriptsCoefficients {
    /// THE ROWS, WHERE THEY ARE ALL CONSTANT.
    #[must_use]
    pub fn rows(&self) -> Option<&[DimCoefficients]> {
        match self {
            SubscriptsCoefficients::Extracted(rows) => Some(rows),
            SubscriptsCoefficients::CoefficientIsNotConstant { .. } => None,
        }
    }
}

/// Replaces: e188_calculateSubscriptsCoefficients
///
/// **188/384** `MutableAddrSplittingPass::calculateSubscriptsCoefficients` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188` (11L).
///
/// ```cpp
/// SmallVector<SmallVector<int64_t>> subscripts_coeffs;
/// for (int p = 0, e = partition_sizes.size(); p < e; ++p) {
///   SmallVector<int64_t> dim_coeffs =
///       dcc::agen::utils::extractConstantOffsetsFromMapForDim(subscripts_map,
///                                                            mas_data[p].dim_);
///   // Add the vector to the map.
///   subscripts_coeffs.push_back(dim_coeffs);
/// }
/// return subscripts_coeffs;
/// ```
///
/// # ⭐⭐ WHAT THE ROWS ARE FOR: THE SUBSCRIPT `fillPartitions` SUBTRACTS FROM
///
/// A partition is the same transfer with its start address shifted forward and its subscripts shifted
/// BACK, so that partition `n`'s first iteration reads what iteration `n * partition_size` of the
/// original read. `fillPartitions` does the second half with these coefficients (`:1151-1157`):
///
/// ```cpp
/// auto coeffs = subscripts_coeffs[p];
/// new_exprs[r] = new_exprs[r] - (coeffs[r] * prev_iters);
/// ```
///
/// ⭐⭐ AND IBM'S OWN KEYS SHOW THE ARITHMETIC. `constant_start_addr_1` loads at
/// `[0, %13 * 3, %14 * 8]` and splits `%14` into partitions of four, so the coefficient row is
/// `[0, 0, 8]` and the second partition reads `[0, %13 * 3, %14 * 8 - 32]`
/// (`mutable_addr_splitting_one_dim.mlir:35`, `:40` — `8 * 4 = 32`). ⭐ The `0` rows are what keeps
/// the other two subscripts still: only the split dimension's own stride moves.
///
/// # ⛔ THE COEFFICIENT IS READ OFF THE MAP, NOT TAKEN FROM `composed_coeff_`
///
/// [`MasData::composed_coeff`] is the dimension's stride in ELEMENTS OF THE FLAT ADDRESS — the
/// layout map applied to the subscript stride (`2048` for a dimension whose subscript stride is 8 over
/// a `d2 * 256` layout). These coefficients are the SUBSCRIPT strides, one per result of the
/// unflattened map, and they are what an `agen` index list is written in. Using one where the other
/// belongs is an address off by the whole layout.
///
/// # Arguments
///
/// * `mas_data` — the transfer's iterators, sorted by weight.
/// * `partition_sizes` — [`PartitionSizes::sizes`]. ⭐ ONLY ITS LENGTH IS READ, exactly as the
///   reference reads it (`partition_sizes.size()`): the sizes say WHICH dimensions were split, and
///   passing them rather than a count keeps the pairing with `mas_data` visible at the call.
/// * `subscripts_map` — the transfer's index map, one result per subscript.
#[must_use]
pub fn calculate_subscripts_coefficients(
    mas_data: &[MasData],
    partition_sizes: &[i64],
    subscripts_map: &AffineMap,
) -> SubscriptsCoefficients {
    let mut rows = Vec::with_capacity(partition_sizes.len());
    // ⭐ `zip` RATHER THAN `mas_data[p]`: one row per split dimension that HAS an iterator, where the
    // reference indexes `mas_data` by a `partition_sizes` position. Its own construction keeps the two
    // aligned — [`calculate_partition_sizes`] pushes one size per entry it walks — so the two agree
    // wherever the reference is defined, and this asks nothing about a dimension that is not there.
    for (dim, _size) in mas_data.iter().zip(partition_sizes) {
        let mut coeffs = Vec::with_capacity(subscripts_map.results.len());
        for (result, expr) in subscripts_map.results.iter().enumerate() {
            let Some(coeff) = constant_offset_for_dim(expr, dim.dim) else {
                return SubscriptsCoefficients::CoefficientIsNotConstant {
                    dim: dim.dim,
                    result,
                };
            };
            coeffs.push(coeff);
        }
        rows.push(DimCoefficients {
            dim: dim.dim,
            coeffs,
        });
    }
    SubscriptsCoefficients::Extracted(rows)
}

/// HOW FAR ONE SUBSCRIPT MOVES PER ITERATION OF `d<dim>` — `None` when the scale is not a constant.
///
/// `dcc::agen::utils::extractConstantOffsetsFromMapForDim`, per result
/// (`dcc/src/Dialect/Agen/Utils.cpp:516-545`):
///
/// ```cpp
/// auto expr = map.getResult(i);
/// // No coefficient exists
/// if (!expr.isFunctionOfDim(dim)) {
///   dim_coeffs.push_back(0);
///   continue;
/// }
/// // Dim is involved, initialize with 1 to start.
/// dim_coeffs.push_back(1);
/// // Look for a multiplication expression in the result that contains the
/// // dim on the left side. If it exists, override the coeff and move to the
/// // next result for analysis.
/// expr.walk([&](AffineExpr e) {
///   if (auto bin_expr = dyn_cast<AffineBinaryOpExpr>(e)) {
///     if (bin_expr.getKind() == AffineExprKind::Mul &&
///         bin_expr.isFunctionOfDim(dim)) {
///       dim_coeffs.back() =
///           cast<AffineConstantExpr>(bin_expr.getRHS()).getValue();
///       return WalkResult::interrupt();
///     }
///   }
///   return WalkResult::advance();
/// });
/// ```
///
/// ⭐ THREE ANSWERS: **0** for a subscript that does not mention the dimension, **1** for one that
/// mentions it unscaled (`d0` alone, or `d0 + 1`), and the multiplier for one that scales it. ⛔ THE
/// `1` IS NOT A DEFAULT — it is the answer for `[0, %13 * 3, %14]`-shaped subscripts, where the split
/// dimension advances the subscript by one per iteration.
///
/// ⚠️ NOT A COEFFICIENT SUM: `d0 * 8 + d0 * 4` answers **8**, not 12, because the walk stops at the
/// first product. MLIR's `simplifyAdd` folds such a sum into `d0 * 12` before it can be asked, which
/// is why the reference can interrupt.
fn constant_offset_for_dim(expr: &AffineExpr, dim: u32) -> Option<i64> {
    if !expr.is_function_of_dim(dim) {
        return Some(0);
    }
    match first_scaling_of_dim(expr, dim) {
        None => Some(1),
        Some(AffineExpr::Const(scale)) => Some(*scale),
        // `cast<AffineConstantExpr>` on a semi-affine product — see
        // [`SubscriptsCoefficients::CoefficientIsNotConstant`].
        Some(_) => None,
    }
}

/// THE RIGHT-HAND SIDE OF THE FIRST `Mul` INVOLVING `d<dim>`, IN THE ORDER MLIR WALKS.
///
/// ⛔⛔ POST ORDER, LEFT OPERAND FIRST — `AffineExpr::walk` is `walkPostOrder`
/// (`llvm-project/mlir/lib/IR/AffineExpr.cpp:58`, `AffineExprVisitor.h:150-231`), which visits the
/// left subtree, then the right, then the node. So the product found is the DEEPEST-LEFTMOST one, and
/// `(d0 * 8) floordiv 4` answers with the inner `8` rather than with the division. Reading the tree
/// outside in would answer differently for every nested product.
///
/// ⭐ `getRHS()` ALONE, BECAUSE MLIR PUTS THE CONSTANT THERE. `simplifyMul` orders a product's
/// operands so that a constant is the right-hand side; this island's [`AffineExpr::times`] takes an
/// `i64` and does the same. A `Mul` reached here whose right side is not a constant is the semi-affine
/// `d0 * s0`.
fn first_scaling_of_dim<'a>(expr: &'a AffineExpr, dim: u32) -> Option<&'a AffineExpr> {
    let (lhs, rhs) = match expr {
        AffineExpr::Add(a, b)
        | AffineExpr::Mul(a, b)
        | AffineExpr::Mod(a, b)
        | AffineExpr::FloorDiv(a, b) => (a.as_ref(), b.as_ref()),
        AffineExpr::Dim(_) | AffineExpr::Sym(_) | AffineExpr::Const(_) => return None,
    };
    if let Some(found) = first_scaling_of_dim(lhs, dim) {
        return Some(found);
    }
    if let Some(found) = first_scaling_of_dim(rhs, dim) {
        return Some(found);
    }
    // `WalkResult::interrupt()` — the node itself, after both operands.
    if matches!(expr, AffineExpr::Mul(..)) && expr.is_function_of_dim(dim) {
        return Some(rhs);
    }
    None
}

/// Replaces: e189_synthesizeTimeInfo
///
/// **189/384** `MutableAddrSplittingPass::synthesizeTimeInfo` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256` (22L).
///
/// ```cpp
/// void MutableAddrSplittingPass::synthesizeTimeInfo(
///     SmallVectorImpl<MASData> &mas_data, agen::AccessDetailsAffineComposite &ad,
///     int64_t &max_mutable) const {
///   auto layout_map = ad.getMemViewLayoutMap();
///   auto time_offsets = ad.getTimeOffsets();
///   auto time_bounds = ad.getTimeBounds();
///   DT_CHECK(time_offsets.size() >= time_bounds.size());
///
///   // Note: time_offsets may contain one entry more than time_bounds to
///   //       represent the constant coefficient. However, that constant
///   //       coefficient isn't used for anything currently so it is
///   //       ignored.
///   int num_non_time_dims = ad.getIndices().size();
///   for (int i = 0, e = time_bounds.size(); i < e; ++i) {
///     // The iter_arg will be the time_bounds - 1 at maximum and the
///     // last iteration of every loop can overflow the mutable as no data transfer
///     // will occur after it. So really, the time_bounds - 2 is the last
///     // utilized mutable address.
///     int64_t weight =
///         time_bounds[i] < 2 ? 0 : (time_bounds[i] - 2) * time_offsets[i];
///     mas_data.emplace_back(num_non_time_dims + i, time_offsets[i],
///                           time_bounds[i], weight);
///     max_mutable += weight;
///   }
/// }
/// ```
///
/// # ⭐⭐ TIME DIMENSIONS ARE ITERATORS TOO, THEY JUST HAVE NO LOOP YET
///
/// `initMASData` (entry 250, `:707`) describes the `affine.for` nest the transfer already sits in —
/// one [`MasData`] per surrounding loop, each with its induction variable. A composite transfer's
/// TIME dimensions are the walk the hardware performs *inside* one op: the `time_set` bounds them
/// and the `*_time_addr_map` says how far each step moves the address, but no loop in the program
/// binds them. This function appends one entry per time dimension so that
/// [`calculate_partition_sizes`] can charge them for the mutable address they consume and pick one
/// to split — and the entries it appends are exactly the ones whose `iter_arg` is `None`, which is
/// how [`create_explicit_time_loops`] later finds them again.
///
/// ⭐ THE DIMENSION NUMBERS CONTINUE THE INDICES' OWN. `num_non_time_dims + i` puts time dimension 0
/// immediately after the last real index, which is the numbering every later reader assumes:
/// `create_explicit_time_loops` recovers `i` as `dim - indices.len()` (`:1308`, `:1337`).
///
/// # ⛔⛔ THE WEIGHT IS TWO ITERATIONS SHORT OF THE SPAN, AND SO IS `initMASData`'S
///
/// `time_bounds[i] < 2 ? 0 : (time_bounds[i] - 2) * time_offsets[i]`, under the reference's own
/// reason: *"The iter_arg will be the time_bounds - 1 at maximum and the last iteration of every loop
/// can overflow the mutable as no data transfer will occur after it. So really, the time_bounds - 2
/// is the last utilized mutable address."* A one- or zero-step dimension therefore weighs NOTHING,
/// not one offset — see [`MasData::weight`], which carries the identical formula for the explicit
/// loops.
///
/// # ⛔⛔ THE `DT_CHECK` IS STRUCTURAL HERE, AND THE EXTRA OFFSET IS DELIBERATELY IGNORED
///
/// `DT_CHECK(time_offsets.size() >= time_bounds.size())` guards a `time_offsets[i]` indexed by a
/// `time_bounds` position. [`TimeOffsets`](super::agen_access_details::TimeOffsets) names the trailing entry `constant` instead of leaving it
/// on the end of the vector, so `per_dim` indexes in lockstep with `time_bounds` by construction and
/// the reference's own note — *"time_offsets may contain one entry more than time_bounds to represent
/// the constant coefficient. However, that constant coefficient isn't used for anything currently so
/// it is ignored"* — is a fact about the type rather than a comment. ⚠️ A `zip` still stops at the
/// shorter of the two, which is what a failed `DT_CHECK` would have done: no entry is appended for a
/// dimension whose offset is missing.
///
/// # ⚠️ `layout_map` IS FETCHED AND NEVER READ
///
/// `ad.getMemViewLayoutMap()` is the reference's first line and nothing in the body mentions it —
/// the layout is already folded into `time_offsets` by `calculateTimeOffsets`, which flattens
/// `mem_view_layout_map.compose(time_addr_map)` (`dialect_utils/Agen/Utils.cpp:103-109`). Reading it
/// again here would be the second answer to a question already answered.
///
/// # ⛔ A SENTINEL BOUND IS TRANSCRIBED, NOT REFUSED — AND IT WEIGHS ZERO
///
/// The reference's `time_bounds` is a vector of `int64_t` carrying [`TimeBound::Coalesced`] as **-2**
/// and [`TimeBound::Variable`] as **-1**, and it stores that number straight into `num_iters_`. Both
/// are below 2, so both weigh 0 and neither moves `max_mutable` — which is why the reference can be
/// this careless with them. This port keeps the same numbers in [`MasData::num_iters`] (see
/// [`time_bound_as_num_iters`]) rather than inventing a refusal the reference does not have, and
/// [`create_explicit_time_loops`] is where a sentinel is actually caught, because it reads
/// `ad.time_bounds` — the typed vector — and not the transcribed integer.
///
/// # Arguments
///
/// * `mas_data` — the iterator list, APPENDED to; `&mut Vec` because this function grows it.
/// * `ad` — the transfer's access details, for `time_bounds`, `time_offsets` and `indices`.
/// * `max_mutable` — the running mutable-address span, `+=`'d by every weight
///   ([`MutableAddr::add_weight`]).
pub fn synthesize_time_info(
    mas_data: &mut Vec<MasData>,
    ad: &AccessDetailsAffineComposite<'_>,
    max_mutable: &mut MutableAddr,
) {
    // `int num_non_time_dims = ad.getIndices().size();`
    let num_non_time_dims = ad.affine.base.indices.len();

    for (i, (&bound, &offset)) in ad
        .time_bounds
        .iter()
        .zip(&ad.time_offsets.per_dim)
        .enumerate()
    {
        let num_iters = time_bound_as_num_iters(bound);
        // `time_bounds[i] < 2 ? 0 : (time_bounds[i] - 2) * time_offsets[i]`
        let weight = if num_iters < 2 {
            0
        } else {
            num_iters.saturating_sub(2).saturating_mul(offset)
        };
        mas_data.push(MasData {
            // The four-argument constructor (`:112-116`) — an IMPLICIT loop, so no `iter_arg`.
            iter_arg: None,
            dim: u32::try_from(num_non_time_dims.saturating_add(i)).unwrap_or(u32::MAX),
            composed_coeff: offset,
            num_iters,
            weight,
        });
        max_mutable.add_weight(weight);
    }
}

/// THE INTEGER THE REFERENCE'S `time_bounds` VECTOR HOLDS — a trip count, or a sentinel below it.
///
/// ⛔ THE SENTINELS ARE THE REFERENCE'S OWN NUMBERS: `kInvalid = -1`, `kCoalesced = -2`
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:252-255`). Both are `< 2`, so
/// [`synthesize_time_info`]'s weight is 0 for either and `max_mutable` is unchanged — the reference
/// never distinguishes them here.
///
/// ⚠️ SATURATING AT [`i64::MAX`], where the reference's `int64_t` could not have held the count at
/// all. [`TimeBound::Steps`] is a `u64` because a bound is a count; every bound in the authority
/// tree's tests is under 4096.
fn time_bound_as_num_iters(bound: TimeBound) -> i64 {
    match bound {
        TimeBound::Steps(steps) => match i64::try_from(steps) {
            Ok(steps) => steps,
            Err(_) => i64::MAX,
        },
        TimeBound::Variable => -1,
        TimeBound::Coalesced => -2,
    }
}

/// A TRANSFER'S ACCESS TRIPLE, AS THIS PASS REWRITES IT — the reference's three in-out parameters.
///
/// ⭐ THREE `&`-PARAMETERS THAT ARE ALWAYS PASSED TOGETHER. `transformCompLoadAndStore` reads them off the
/// access details into locals (`:488-490`), hands all three to
/// [`create_explicit_time_loops`] to be rewritten, and then hands
/// the SAME three to `cloneWithNewAccessInfo` for every partition (`:512-515`, `:526-529`) — they are
/// one value with one lifetime, so they are one struct here.
///
/// ⛔⛔ AND ONE OF THE THREE IS WRITTEN AND NEVER READ, DELIBERATELY. See
/// [`TimeLoopNest::access`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubscriptsAndTime {
    /// `ad.getSubscriptsMap()` — the transfer's subscripts as a map over its indices.
    pub subscripts_map: AffineMap,
    /// `ad.getIndices()` — the values the map's dimensions stand for, in order.
    pub indices: Vec<Val>,
    /// `ad.getTimeSet()` — the bounds on the walk the hardware performs inside the transfer.
    pub time_set: IntegerSet,
}

/// WHAT `createExplicitTimeLoops` LEAVES BEHIND — the loop nest, its induction variables, and the
/// rewritten access triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeLoopNest {
    /// THE OUTERMOST `affine.for`, WITH EVERYTHING ELSE INSIDE IT — including the transfer, which the
    /// reference moves into the innermost body with `op->moveBefore(for_body, for_body->begin())`
    /// (`:1327`).
    ///
    /// ⭐ THE OP GOES IN BY VALUE, WHICH IS WHAT A MOVE IS. The reference's `Operation*` keeps its
    /// identity and changes parent; this island's [`affine::Op::For::body`] holds its ops directly, so
    /// re-parenting is handing the op to the loop that now owns it — and the transfer is no longer
    /// anywhere else, which is the same guarantee `moveBefore` gives.
    pub nest: DfirOp,

    /// One induction variable per loop, OUTERMOST FIRST — `for_ops[i].getInductionVar()`.
    ///
    /// ⭐ THE LIST IS `time_dim_idx + 1` LONG, and it is indexed by the RAW time-dimension index: the
    /// reference's own two readers are `for_ops[i - num_orig_dims]` (`Agen/Utils.cpp:472`) and
    /// `for_ops[actual_time_dim]` (`:1341`). ⛔ Which is not the same index as the position within
    /// `time_order` — see [`create_explicit_time_loops`].
    pub ivs: Vec<Val>,

    /// The access triple with the now-explicit time dimensions folded into it.
    ///
    /// # ⛔⛔ `subscripts_map` IS COMPUTED, RETURNED, AND READ BY NOBODY
    ///
    /// The reference rewrites its `subscripts_map` out-parameter to a map over the explicit loops
    /// (`Agen/Utils.cpp:476-477`) — and then `transformCompLoadAndStore` calls
    /// `createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(), createOps)` (`:533`),
    /// passing the ORIGINAL map off the access details, and it is that one every partition's
    /// `new_subscripts_map` descends from. The local the pass just had rewritten is dead.
    ///
    /// ⭐ WHICH IS EXACTLY WHAT THE VENDOR'S OWN ANSWER KEY SHOWS. The partitioned transfer prints
    /// `dst:%[[VAL_16]][0, %[[VAL_9]] * 16, %[[VAL_10]] * 8]` — the two-dimensional map it started
    /// with — while `indices` has grown a third entry, the new loop's induction variable
    /// (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_time_dims.mlir:39`). The extra
    /// operand is invisible because `printAffineMapOfSSAIds` prints only `numDims + numSymbols` of
    /// them and `AgenOps.cpp`'s verifier does not count them either.
    ///
    /// ⛔ CARRIED ANYWAY, BECAUSE THE FUNCTION'S JOB IS TO PRODUCE IT. Dropping it would make this
    /// port a different function from the reference on the day a caller starts reading it; the two
    /// members beside it — `indices` and `time_set` — ARE live, and both reach the cloned transfer.
    pub access: SubscriptsAndTime,
}

/// WHETHER THE TIME LOOPS WERE MADE EXPLICIT, AND WHICH OF THE REFERENCE'S ABORTS SAYS NOT.
///
/// ⛔ NOT AN ERROR TYPE — one plain `return`, and eleven states this island can hold that the
/// reference either `DT_CHECK`s or reads past. See [`ConditionalTree`] and [`Partitioning`] for the
/// same treatment of the same pass's other two out-parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplicitTimeLoops {
    /// The nest, its induction variables and the rewritten access.
    Created(TimeLoopNest),

    /// `if (innermost_time_dim < 0) return;` (`:1301`) — NO SPLIT DIMENSION IS A TIME DIMENSION, so
    /// there is nothing to make explicit and the transfer stays exactly where it was.
    ///
    /// ⭐ THE COMMON CASE, NOT AN EDGE. Four of the pass's five answer keys split an explicit
    /// `affine.for` and land here; only `mutable_addr_splitting_time_dims.mlir` reaches
    /// [`ExplicitTimeLoops::Created`].
    ///
    /// ⭐ THE OP COMES BACK OUT, because it went in by value and the reference left it untouched.
    NotSplitOnATimeDimension(DfirOp),

    /// MORE PARTITION SIZES THAN ITERATORS — `mas_data[i]` indexed by a `partition_sizes` position
    /// (`:1294-1297`), the same pairing [`ConditionalTree::MorePartitionsThanIterators`] guards.
    MorePartitionsThanIterators {
        /// `partition_sizes.size()`.
        sizes: usize,
        /// `mas_data.size()`.
        iterators: usize,
    },

    /// A SPLIT DIMENSION WITH NO ITERATOR THAT IS NOT A TIME DIMENSION EITHER.
    ///
    /// ⛔ `time_dim_idx = innermost_time_dim - num_non_time_dims` IS NEGATIVE (`:1308`), and
    /// `constructExplicitTimeLoops` opens with `DT_CHECK(time_dim >= 0 && ..)`
    /// (`Agen/Utils.cpp:425`). Every entry `initMASData` pushes (entry 250, `:707`) carries the enclosing
    /// loop's induction variable and every entry [`synthesize_time_info`] pushes is numbered at or
    /// above `indices.len()`, so this is a `mas_data` neither of them built.
    NonTimeDimensionHasNoIterator {
        /// `mas_data[i].dim_`.
        dim: u32,
    },

    /// THE TIME DIMENSION TO SPLIT HAS NO ENTRY IN `time_bounds`.
    ///
    /// ⛔ The other half of `DT_CHECK(time_dim >= 0 && time_dim < time_bounds.size())`
    /// (`Agen/Utils.cpp:425`). [`synthesize_time_info`] numbers the time dimensions from
    /// `time_bounds`, so reaching this means the two were built from different access details.
    TimeDimensionHasNoBound {
        /// `time_dim`.
        time_dim: usize,
        /// `time_bounds.size()`.
        bounds: usize,
    },

    /// A TIME BOUND THAT IS NOT A TRIP COUNT, WHERE A LOOP NEEDS ONE.
    ///
    /// ⛔⛔ `affine::AffineForOp::create(builder, loc, 0, time_bounds[dim])`
    /// (`Agen/Utils.cpp:430-431`) with a SENTINEL — [`TimeBound::Coalesced`] is `-2` and
    /// [`TimeBound::Variable`] is `-1` — builds `affine.for %i = 0 to -2`, a loop MLIR accepts and
    /// that runs zero times. The transfer moved into it would never happen.
    ///
    /// ⭐ THE TYPE IS WHAT CATCHES IT, WHICH IS WHY [`synthesize_time_info`] DOES NOT HAVE TO. That
    /// function transcribes both sentinels into `num_iters` as the reference's own integers and they
    /// weigh 0 there; here the same value has to be a loop bound and cannot be one.
    TimeBoundIsNotALoopTripCount {
        /// Which time dimension, raw index.
        time_dim: usize,
        /// `time_bounds[time_dim]`.
        bound: TimeBound,
    },

    /// `ad.getTimeAddrMap()` IS NULL — `concatenateMaps` reads `getNumResults()` off it
    /// (`dialect_utils/Agen/Utils.cpp:259`).
    TimeAddrMapIsAbsent,

    /// `assert(map_A.getNumResults() == map_B.getNumResults())` — `concatenateMaps`
    /// (`dialect_utils/Agen/Utils.cpp:259`).
    ///
    /// ⛔ AND AN `assert`, NOT A `DT_CHECK`: in a release build the loop that follows walks
    /// `map_A`'s results and reads `shifted_map_B.getResult(i)` past the end of `map_B`'s.
    SubscriptsAndTimeAddrDisagreeOnRank {
        /// `subscripts_map.getNumResults()`.
        subscripts: usize,
        /// `time_addr_map.getNumResults()`.
        time_addr: usize,
    },

    /// `ad.getTimeOrder()` IS NULL — `updateTimeSetForExplicitDims` calls `getResult` and
    /// `getDimPosition` on it (`Agen/Utils.cpp:489`, `:497`).
    TimeOrderIsAbsent,

    /// `time_order.getResult(dim)` PAST THE END, for a `dim` in `0..=time_dim`
    /// (`Agen/Utils.cpp:489`).
    TimeOrderHasNoDimension {
        /// `time_dim`.
        time_dim: usize,
        /// `time_order.getNumResults()`.
        results: usize,
    },

    /// `time_order.getDimPosition(dim)` ON A RESULT THAT IS NOT A BARE DIMENSION
    /// (`Agen/Utils.cpp:497`).
    ///
    /// ⛔ `cast<AffineDimExpr>` — not `dyn_cast`, so this is an abort and not a `nullptr`. A
    /// `time_order` is the composite op's own attribute (`op.getTimeOrder()`), and it is a permutation
    /// only because `checkBasicConditions` gates on `inversePermutation` succeeding, for four of the
    /// six composite kinds (`AgenToSentient/Helper.cpp:166-175`); nothing constructs it.
    TimeOrderIsNotAPermutation {
        /// The `dim` whose result is not a dimension.
        time_dim: usize,
    },

    /// FEWER INDICES THAN THE SUBSCRIPTS MAP HAS DIMENSIONS — `indices[i]` for
    /// `i < subscripts_map.getNumDims()` (`Agen/Utils.cpp:454`).
    ///
    /// ⛔ THE TWO ARE ONE VALUE IN THE ACCESS DETAILS and disagreeing is a program neither
    /// `constructDetails` nor this pass can produce; it is stated because [`SubscriptsAndTime`] holds
    /// them side by side and the reference indexes one by the other.
    SubscriptsMapHasMoreDimensionsThanIndices {
        /// `subscripts_map.getNumDims()`.
        dims: u32,
        /// `indices.size()`.
        indices: usize,
    },

    /// A TIME DIMENSION USED BY THE SUBSCRIPTS WITH NO LOOP CREATED FOR IT.
    ///
    /// ⛔ `DT_CHECK((i - num_orig_dims) < for_ops.size())` (`Agen/Utils.cpp:470`).
    ///
    /// ⭐ STRUCTURALLY UNREACHABLE WHEN THE TWO ARE PAIRED AS THE REFERENCE PAIRS THEM: the branch
    /// that indexes `for_ops` is guarded by `i <= actual_preserve_dim`, i.e.
    /// `i - num_orig_dims <= time_dim`, and `constructExplicitTimeLoops` returns exactly
    /// `time_dim + 1` loops. It is stated because this port takes the induction variables as a slice
    /// and the reference's own `DT_CHECK` is the only thing that says so.
    TimeDimensionHasNoLoop {
        /// The raw time-dimension index the subscripts use.
        time_dim: usize,
        /// `for_ops.size()`.
        loops: usize,
    },
}

impl ExplicitTimeLoops {
    /// THE NEST, WHERE ONE WAS BUILT.
    #[must_use]
    pub fn nest(&self) -> Option<&TimeLoopNest> {
        match self {
            ExplicitTimeLoops::Created(nest) => Some(nest),
            ExplicitTimeLoops::NotSplitOnATimeDimension(_)
            | ExplicitTimeLoops::MorePartitionsThanIterators { .. }
            | ExplicitTimeLoops::NonTimeDimensionHasNoIterator { .. }
            | ExplicitTimeLoops::TimeDimensionHasNoBound { .. }
            | ExplicitTimeLoops::TimeBoundIsNotALoopTripCount { .. }
            | ExplicitTimeLoops::TimeAddrMapIsAbsent
            | ExplicitTimeLoops::SubscriptsAndTimeAddrDisagreeOnRank { .. }
            | ExplicitTimeLoops::TimeOrderIsAbsent
            | ExplicitTimeLoops::TimeOrderHasNoDimension { .. }
            | ExplicitTimeLoops::TimeOrderIsNotAPermutation { .. }
            | ExplicitTimeLoops::SubscriptsMapHasMoreDimensionsThanIndices { .. }
            | ExplicitTimeLoops::TimeDimensionHasNoLoop { .. } => None,
        }
    }
}

/// Replaces: e190_createExplicitTimeLoops
///
/// **190/384** `MutableAddrSplittingPass::createExplicitTimeLoops` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282` (57L).
///
/// ```cpp
/// int innermost_time_dim = -1;
/// for (int i = 0, e = partition_sizes.size(); i < e; ++i) {
///   if (mas_data[i].iter_arg_ != nullptr) continue;
///   if (mas_data[i].dim_ > innermost_time_dim)
///     innermost_time_dim = mas_data[i].dim_;
/// }
///
/// if (innermost_time_dim < 0) return;
///
/// int num_non_time_dims = indices.size();
///
/// int time_dim_idx = innermost_time_dim - num_non_time_dims;
/// auto time_bounds = ad.getTimeBounds();
/// auto for_ops = dcc::agen::utils::constructExplicitTimeLoops(op, time_bounds, time_dim_idx);
///
/// auto time_addr_map = ad.getTimeAddrMap();
/// AffineMap subscripts_map_time = agen::utils::concatenateMaps(subscripts_map, time_addr_map);
///
/// auto time_order = ad.getTimeOrder();
/// dcc::agen::utils::updateTimeSetForExplicitDims(time_dim_idx, time_order, time_set);
/// dcc::agen::utils::updateSubscriptsAndIndicesForExplicitTimeLoops(
///     for_ops, time_dim_idx, subscripts_map, subscripts_map_time, indices);
///
/// auto for_body = cast<affine::AffineForOp>(for_ops.back()).getBody();
/// op->moveBefore(for_body, for_body->begin());
///
/// for (int i = 0, e = mas_data.size(); i < e; ++i) {
///   if (mas_data[i].iter_arg_ != nullptr) continue;
///   int actual_time_dim = mas_data[i].dim_ - num_non_time_dims;
///   if (actual_time_dim > time_dim_idx) continue;
///   DT_CHECK(actual_time_dim < for_ops.size());
///   mas_data[i].iter_arg_ =
///       cast<affine::AffineForOp>(for_ops[actual_time_dim]).getInductionVar();
/// }
/// ```
///
/// # ⭐⭐ A TIME DIMENSION CANNOT BE COMPARED AGAINST UNTIL A LOOP BINDS IT
///
/// [`calculate_partition_sizes`] may pick a TIME dimension to split — [`synthesize_time_info`] put
/// them in the same list and gave them the heaviest weights. But partitioning means emitting
/// `arith.cmpi slt, <iterator>, <boundary>`, and a time dimension has no iterator: it is a walk the
/// `time_set` describes and the hardware performs inside one op. So this function writes the
/// `affine.for` that binds it, from the OUTERMOST time dimension down to the one being split, moves
/// the transfer inside, and fills in the induction variables [`construct_conditionals`] will compare
/// (`:491` runs before `:533`'s `createPartitions`). ⛔ Without it, that function's
/// [`ConditionalTree::DimensionHasNoIterator`] is the only possible answer.
///
/// ⭐ ONLY DOWN TO THE SPLIT DIMENSION, under the reference's own note: *"Time loops are created
/// outermost to innermost, preserving the innermost time loops as much as possible."* Every time
/// dimension inner to it stays implicit and keeps being walked by the hardware, which is the whole
/// point — an explicit loop is one `agen` op per iteration.
///
/// # ⛔⛔ TWO INDEXINGS OF THE SAME NUMBER THAT DISAGREE WHENEVER `time_order` IS NOT THE IDENTITY
///
/// `time_bounds` and `time_offsets` are ordered by `time_order` RESULT POSITION — outermost first —
/// because `calculateTimeOffsets` and `calculateTimeBounds` walk `time_order.getResults()`
/// (`dialect_utils/Agen/Utils.cpp:115`, `:254`). `constructExplicitTimeLoops` therefore builds
/// `for_ops[dim]` with `time_bounds[dim]`, i.e. in that same outermost-first order. ⛔ But
/// `updateSubscriptsAndIndicesForExplicitTimeLoops` reaches for `for_ops[i - num_orig_dims]`
/// (`dcc/src/Dialect/Agen/Utils.cpp:472`) where `i` is a dimension of the TIME ADDRESS MAP — its own
/// dimension numbering, which `time_order` exists precisely to permute. The two agree only when
/// `time_order` is the identity.
///
/// ⛔ THE VENDOR'S OWN KEY IS A CASE WHERE THEY DISAGREE AND IT DOES NOT SHOW.
/// `time_order = affine_map<(d0, d1, d2) -> (d2, d1, d0)>` is a reversal
/// (`mutable_addr_splitting_time_dims.mlir:137`), so outermost is `d2` while `for_ops[0]` gets looked
/// up for time-address dimension `d0`. It goes unnoticed because exactly ONE loop is created:
/// `for_ops[0]` is the only entry there is, and the `i > actual_preserve_dim` test
/// drops every other dimension before it can index anything. ⭐ PORTED AS WRITTEN — the numbering is
/// the reference's, and a port that permuted here would emit a different `indices` list from `dcc`
/// for a transfer neither of them is tested on.
///
/// # ⛔ THE `mas_data` WRITES ARE ALL-OR-NOTHING
///
/// The reference walks `mas_data` and assigns as it goes, so a `DT_CHECK` in the middle leaves the
/// list half-filled. This port collects the assignments and applies them once every check has passed,
/// which is what a report instead of an abort has to mean: on anything but
/// [`ExplicitTimeLoops::Created`] the caller's `mas_data` is exactly as it handed it over.
///
/// # Arguments
///
/// * `vals` — the SSA namer, for the loops' induction variables.
/// * `mas_data` — the transfer's iterators, sorted by weight; the time entries' `iter_arg`s are
///   filled in here.
/// * `partition_sizes` — [`PartitionSizes::sizes`], which is what says WHICH dimensions are split.
/// * `op` — the transfer, BY VALUE, because it ends up inside the innermost loop.
/// * `ad` — the access details, for `time_bounds`, `time_addr_map` and `time_order`.
/// * `access` — the triple to rewrite; [`SubscriptsAndTime`].
pub fn create_explicit_time_loops(
    vals: &mut Values,
    mas_data: &mut [MasData],
    partition_sizes: &[i64],
    op: DfirOp,
    ad: &AccessDetailsAffineComposite<'_>,
    access: &SubscriptsAndTime,
) -> ExplicitTimeLoops {
    if partition_sizes.len() > mas_data.len() {
        return ExplicitTimeLoops::MorePartitionsThanIterators {
            sizes: partition_sizes.len(),
            iterators: mas_data.len(),
        };
    }

    // `if (mas_data[i].iter_arg_ != nullptr) continue; if (mas_data[i].dim_ > innermost_time_dim) ..`
    // — the highest dimension among the split ones that no loop binds. `-1` is [`None`].
    let innermost_time_dim = mas_data[..partition_sizes.len()]
        .iter()
        .filter(|data| data.iter_arg.is_none())
        .map(|data| data.dim)
        .max();
    let Some(innermost_time_dim) = innermost_time_dim else {
        // `if (innermost_time_dim < 0) return;`
        return ExplicitTimeLoops::NotSplitOnATimeDimension(op);
    };

    // `int num_non_time_dims = indices.size();`
    let num_non_time_dims = access.indices.len();
    // `int time_dim_idx = innermost_time_dim - num_non_time_dims;`
    let Some(time_dim_idx) = (innermost_time_dim as usize).checked_sub(num_non_time_dims) else {
        return ExplicitTimeLoops::NonTimeDimensionHasNoIterator {
            dim: innermost_time_dim,
        };
    };

    // ── `constructExplicitTimeLoops(op, time_bounds, time_dim_idx)`, bounds first ────────────────
    if time_dim_idx >= ad.time_bounds.len() {
        return ExplicitTimeLoops::TimeDimensionHasNoBound {
            time_dim: time_dim_idx,
            bounds: ad.time_bounds.len(),
        };
    }
    let mut trip_counts: Vec<i64> = Vec::with_capacity(time_dim_idx + 1);
    for (time_dim, &bound) in ad.time_bounds[..=time_dim_idx].iter().enumerate() {
        match bound {
            TimeBound::Steps(steps) => match i64::try_from(steps) {
                Ok(steps) => trip_counts.push(steps),
                Err(_) => {
                    return ExplicitTimeLoops::TimeBoundIsNotALoopTripCount { time_dim, bound };
                }
            },
            TimeBound::Coalesced | TimeBound::Variable => {
                return ExplicitTimeLoops::TimeBoundIsNotALoopTripCount { time_dim, bound };
            }
        }
    }
    // The reference has created the loops by this point, so the names are minted here too — the nest
    // itself is assembled at the end, once the transfer is the only thing left to put inside it.
    let ivs: Vec<Val> = trip_counts.iter().map(|_| vals.mint()).collect();

    // ── `concatenateMaps(subscripts_map, ad.getTimeAddrMap())` ───────────────────────────────────
    let Some(time_addr_map) = ad.time_addr_map.as_ref() else {
        return ExplicitTimeLoops::TimeAddrMapIsAbsent;
    };
    if access.subscripts_map.results.len() != time_addr_map.results.len() {
        return ExplicitTimeLoops::SubscriptsAndTimeAddrDisagreeOnRank {
            subscripts: access.subscripts_map.results.len(),
            time_addr: time_addr_map.results.len(),
        };
    }
    let subscripts_map_time = concatenate_maps(&access.subscripts_map, time_addr_map);

    // ── `updateTimeSetForExplicitDims(time_dim_idx, time_order, time_set)` ───────────────────────
    let Some(time_order) = ad.time_order.as_ref() else {
        return ExplicitTimeLoops::TimeOrderIsAbsent;
    };
    // `time_order.getResult(dim)` and `time_order.getDimPosition(dim)` for every dim being made
    // explicit, resolved here so the helper below is total. Both are aborts in the reference.
    let mut explicit_dims: Vec<(AffineExpr, u32)> = Vec::with_capacity(time_dim_idx + 1);
    for time_dim in 0..=time_dim_idx {
        let Some(result) = time_order.results.get(time_dim) else {
            return ExplicitTimeLoops::TimeOrderHasNoDimension {
                time_dim,
                results: time_order.results.len(),
            };
        };
        let Some(position) = time_order.dim_position(time_dim) else {
            return ExplicitTimeLoops::TimeOrderIsNotAPermutation { time_dim };
        };
        explicit_dims.push((result.clone(), position));
    }
    let time_set = update_time_set_for_explicit_dims(&explicit_dims, &access.time_set, time_order);

    // ── `updateSubscriptsAndIndicesForExplicitTimeLoops(..)` ─────────────────────────────────────
    let num_orig_dims = access.subscripts_map.dims as usize;
    if access.indices.len() < num_orig_dims {
        return ExplicitTimeLoops::SubscriptsMapHasMoreDimensionsThanIndices {
            dims: access.subscripts_map.dims,
            indices: access.indices.len(),
        };
    }
    let Some((subscripts_map, indices)) = update_subscripts_and_indices_for_explicit_time_loops(
        &ivs,
        time_dim_idx,
        num_orig_dims,
        &subscripts_map_time,
        &access.indices,
    ) else {
        return ExplicitTimeLoops::TimeDimensionHasNoLoop {
            time_dim: time_dim_idx,
            loops: ivs.len(),
        };
    };

    // ── The induction variables the now-explicit time dimensions are bound by ────────────────────
    //
    // `int actual_time_dim = mas_data[i].dim_ - num_non_time_dims; if (actual_time_dim >
    // time_dim_idx) continue; DT_CHECK(actual_time_dim < for_ops.size());`
    for data in mas_data.iter_mut().filter(|data| data.iter_arg.is_none()) {
        let Some(actual_time_dim) = (data.dim as usize).checked_sub(num_non_time_dims) else {
            continue;
        };
        if actual_time_dim > time_dim_idx {
            continue;
        }
        // Structurally within range: `time_dim_idx + 1` loops were created. See
        // [`ExplicitTimeLoops::TimeDimensionHasNoLoop`], which the earlier `?` already answered for.
        data.iter_arg = ivs.get(actual_time_dim).copied();
    }

    ExplicitTimeLoops::Created(TimeLoopNest {
        nest: nest_explicit_time_loops(op, &ivs, &trip_counts),
        ivs,
        access: SubscriptsAndTime {
            subscripts_map,
            indices,
            time_set,
        },
    })
}

/// [`nest_explicit_time_loops`] WITH MORE THAN ONE OP IN THE INNERMOST BODY.
///
/// ⛔ `TPMVCompositeLoadStore::createNewMemOp` NEEDS IT: its `cloneMemViewIfNonPaged` builds the other
/// side's view at the SAME insertion point as the new transfer
/// (`TransformPagedMemView/TransformPagedMemViewImpl.cpp:1252`, `:1259`), which is
/// `for_ops.back().getBody()` by the time entry 325 calls it. The single-op form above cannot say
/// that, and hoisting the view out of the nest would put it where the loop iterators are not in scope.
pub(super) fn nest_explicit_time_loops_over(
    body: Vec<DfirOp>,
    ivs: &[Val],
    trip_counts: &[i64],
) -> Vec<DfirOp> {
    let mut nest = body;
    for (&iv, &hi) in ivs.iter().zip(trip_counts).rev() {
        nest = vec![DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(hi),
            carried: Vec::new(),
            body: nest,
            dbg_name: None,
        })];
    }
    nest
}

/// `constructExplicitTimeLoops` — `dcc/src/Dialect/Agen/Utils.cpp:422` (16L).
///
/// ```cpp
/// SmallVector<Operation *, 16> constructExplicitTimeLoops(
///     Operation *mem_op, const SmallVectorImpl<int64_t> &time_bounds, int time_dim) {
///   DT_CHECK(time_dim >= 0 && time_dim < time_bounds.size());
///   OpBuilder builder(mem_op);
///   SmallVector<Operation *, 16> for_ops;
///   for (int dim = 0; dim <= time_dim; ++dim) {
///     auto for_op = affine::AffineForOp::create(builder, mem_op->getLoc(), 0, time_bounds[dim]);
///     for_ops.push_back(for_op);
///     builder.setInsertionPointToStart(for_op.getBody());
///   }
///   return for_ops;
/// }
/// ```
///
/// ⚠️ NOT A SCHEDULED UNIT — `dcc::agen::utils` is not in this campaign's manifest, so this carries a
/// citation and no anchor. Its caller is [`create_explicit_time_loops`].
///
/// # ⭐ ONE NEST, BUILT FROM THE INSIDE OUT
///
/// `builder.setInsertionPointToStart(for_op.getBody())` after each loop is what makes the next one
/// NESTED rather than a sibling, and `OpBuilder builder(mem_op)` puts the outermost where the transfer
/// was. This island's [`affine::Op::For::body`] holds its ops directly, so the same nest is one value
/// assembled innermost first — and `op` goes in at the bottom, which is the reference's separate
/// `op->moveBefore(for_body, for_body->begin())` (`MutableAddrSplitting.cpp:1326-1327`).
///
/// ⛔ TOTAL, WITH THE `DT_CHECK` HOISTED. The caller has already turned `time_bounds[0..=time_dim]`
/// into trip counts and reported anything that is not one, so `trip_counts` IS the loop list: one
/// bound per loop, outermost first. An empty one yields `op` itself, which is the nest of no loops.
///
/// ⭐ `carried: Vec::new()`, WHICH PRINTS NO `iter_args`, NO RESULTS AND NO TERMINATOR — the plain
/// counted loop `AffineForOp::create(builder, loc, 0, ub)` builds. ⭐ AND `dbg_name: None`: the
/// reference passes no name, and an empty dictionary is not the same text as no dictionary.
pub(super) fn nest_explicit_time_loops(op: DfirOp, ivs: &[Val], trip_counts: &[i64]) -> DfirOp {
    let mut nest = op;
    for (&iv, &hi) in ivs.iter().zip(trip_counts).rev() {
        nest = DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(hi),
            carried: Vec::new(),
            body: vec![nest],
            dbg_name: None,
        });
    }
    nest
}

/// `concatenateMaps` — `dialect_utils/Agen/Utils.cpp:258` (9L).
///
/// ```cpp
/// AffineMap concatenateMaps(const AffineMap &map_A, const AffineMap &map_B) {
///   assert(map_A.getNumResults() == map_B.getNumResults());
///   auto shifted_map_B = map_B.shiftDims(map_A.getNumDims());
///   SmallVector<AffineExpr, 16> exprs;
///   for (int i = 0; i < map_A.getNumResults(); ++i)
///     exprs.emplace_back(map_A.getResult(i) + shifted_map_B.getResult(i));
///   return AffineMap::get(shifted_map_B.getNumDims(), 0, exprs, shifted_map_B.getContext());
/// }
/// ```
///
/// ⚠️ NOT A SCHEDULED UNIT — a `dcc::agen::utils` dependency, so no anchor.
///
/// # ⭐ TWO MAPS OVER DISJOINT DIMENSIONS, ADDED RESULT BY RESULT
///
/// `shiftDims(map_A.getNumDims())` renumbers `map_B`'s dimensions to sit ABOVE `map_A`'s, so the sum
/// is a map over both dimension lists at once: the subscripts' own indices first, then the time
/// dimensions. That is what makes `subscripts_map_time` addressable by a single
/// `replaceDimsAndSymbols` in [`update_subscripts_and_indices_for_explicit_time_loops`].
///
/// ⛔ `+` IS MLIR'S SIMPLIFYING OPERATOR, NOT A BARE NODE — [`AffineExpr::sum`], which is why the
/// vendor's `0 + d2 * 64` comes out as `d2 * 64`.
///
/// ⛔⛔ AND THE RESULT HAS **ZERO** SYMBOLS, `AffineMap::get(.., 0, ..)`. Both inputs' symbols are
/// dropped, not merged — a `subscripts_map` with a symbol in it would lose it here. Every subscripts
/// map in the authority tree's own keys is symbol-free (the transfer's symbols live on `time_set`
/// instead, as `time_symbols`), which is why the reference can do this.
pub(super) fn concatenate_maps(map_a: &AffineMap, map_b: &AffineMap) -> AffineMap {
    let shifted_map_b = map_b.shift_dims(map_a.dims);
    let results = map_a
        .results
        .iter()
        .zip(&shifted_map_b.results)
        .map(|(a, b)| a.clone().added(b.clone()))
        .collect();
    AffineMap {
        dims: shifted_map_b.dims,
        syms: 0,
        results,
    }
}

/// `updateTimeSetForExplicitDims` — `dcc/src/Dialect/Agen/Utils.cpp:482` (33L).
///
/// ```cpp
/// void updateTimeSetForExplicitDims(int time_dim, AffineMap &time_order, IntegerSet &time_set) {
///   DT_CHECK(time_dim >= 0);
///   SmallVector<AffineExpr, 16> new_exprs;
///   SmallVector<bool, 16> eq_exprs;
///   for (int dim = 0; dim <= time_dim; ++dim) {
///     new_exprs.emplace_back(time_order.getResult(dim));
///     eq_exprs.push_back(true);
///   }
///   for (int c = 0; c < time_set.getNumConstraints(); ++c) {
///     auto constraint = time_set.getConstraint(c);
///     bool copy_constraint = true;
///     for (int dim = 0; dim <= time_dim; ++dim) {
///       auto dim_pos = time_order.getDimPosition(dim);
///       if (constraint.isFunctionOfDim(dim_pos)) { copy_constraint = false; break; }
///     }
///     if (copy_constraint) {
///       new_exprs.push_back(constraint);
///       if (time_set.isEq(c)) eq_exprs.push_back(true);
///       else eq_exprs.push_back(false);
///     }
///   }
///   time_set = IntegerSet::get(time_order.getNumDims(), time_set.getNumSymbols(), new_exprs,
///                              eq_exprs);
/// }
/// ```
///
/// ⚠️ NOT A SCHEDULED UNIT — a `dcc::agen::utils` dependency, so no anchor.
///
/// # ⭐⭐ AN EXPLICIT DIMENSION IS PINNED TO ZERO, NOT DELETED
///
/// The rewritten set opens with `time_order.getResult(dim) == 0` for every dimension a loop now binds,
/// and then copies only the constraints that say nothing about those dimensions. So the transfer no
/// longer walks them — each iteration of the new `affine.for` performs a transfer whose time set is
/// fixed at step 0 of the explicit dimensions and still walks the implicit ones. ⛔ The equality goes
/// FIRST, ahead of every copied constraint, which is the order the vendor's own key prints:
/// `affine_set<(d0, d1, d2)[s0] : (d2 == 0, d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>`
/// (`mutable_addr_splitting_time_dims.mlir:14`).
///
/// ⛔ AND THE DROPPED CONSTRAINTS ARE FOUND BY `time_order.getDimPosition(dim)`, NOT BY THE RESULT
/// EXPRESSION. The equality is written in terms of `getResult(dim)` — which for a permutation IS the
/// dimension — while the filter asks about the position. The two coincide exactly when `time_order`
/// is a permutation, which is what [`ExplicitTimeLoops::TimeOrderIsNotAPermutation`] is about.
///
/// ⛔ THE DIMENSION COUNT COMES FROM `time_order`, THE SYMBOL COUNT FROM THE OLD SET. A `time_set`
/// with more dimensions than `time_order` has would be narrowed here, silently; the reference does not
/// check and neither does this.
///
/// ⚠️ TOTAL, WITH BOTH REFERENCE LOOKUPS HOISTED: `explicit_dims` is `(getResult(dim),
/// getDimPosition(dim))` for `dim` in `0..=time_dim`, resolved by the caller so the two aborts can be
/// reported rather than taken.
pub(super) fn update_time_set_for_explicit_dims(
    explicit_dims: &[(AffineExpr, u32)],
    time_set: &IntegerSet,
    time_order: &AffineMap,
) -> IntegerSet {
    let mut constraints: Vec<Constraint> = explicit_dims
        .iter()
        .map(|(result, _)| Constraint {
            expr: result.clone(),
            is_equality: true,
        })
        .collect();

    // `if (constraint.isFunctionOfDim(dim_pos)) { copy_constraint = false; break; }`
    constraints.extend(
        time_set
            .constraints
            .iter()
            .filter(|constraint| {
                !explicit_dims
                    .iter()
                    .any(|&(_, position)| constraint.expr.is_function_of_dim(position))
            })
            .cloned(),
    );

    IntegerSet {
        dims: time_order.dims,
        symbols: time_set.symbols,
        constraints,
    }
}

/// `updateSubscriptsAndIndicesForExplicitTimeLoops` — `dcc/src/Dialect/Agen/Utils.cpp:439` (41L).
///
/// ```cpp
/// void updateSubscriptsAndIndicesForExplicitTimeLoops(
///     SmallVectorImpl<Operation *> &for_ops, int time_dim, AffineMap &subscripts_map,
///     AffineMap &subscripts_map_time, SmallVector<Value> &indices) {
///   SmallVector<AffineExpr, 16> replace_dims;
///   SmallVector<Value> new_indices;
///   int dim_num = 0;
///   int num_orig_dims = subscripts_map.getNumDims();
///   for (int i = 0, e = num_orig_dims; i < e; ++i) {
///     replace_dims.emplace_back(getAffineDimExpr(dim_num++, subscripts_map.getContext()));
///     new_indices.push_back(indices[i]);
///   }
///   int actual_preserve_dim = time_dim + num_orig_dims;
///   for (int i = num_orig_dims, e = subscripts_map_time.getNumDims(); i < e; ++i) {
///     if (i > actual_preserve_dim || !subscripts_map_time.isFunctionOfDim(i)) {
///       replace_dims.emplace_back(getAffineConstantExpr(0, subscripts_map.getContext()));
///       continue;
///     }
///     replace_dims.emplace_back(getAffineDimExpr(dim_num++, subscripts_map.getContext()));
///     DT_CHECK((i - num_orig_dims) < for_ops.size());
///     new_indices.emplace_back(cast<affine::AffineForOp>(for_ops[i - num_orig_dims])
///                                  .getInductionVar());
///   }
///   subscripts_map = subscripts_map_time.replaceDimsAndSymbols(replace_dims, {}, dim_num, 0);
///   indices = new_indices;
/// }
/// ```
///
/// ⚠️ NOT A SCHEDULED UNIT — a `dcc::agen::utils` dependency, so no anchor.
///
/// # ⭐⭐ EVERY DIMENSION THE LOOPS DO NOT BIND IS SUBSTITUTED WITH **ZERO**
///
/// `subscripts_map_time` spans the original indices AND every time dimension. Of the time half, only
/// the ones a new `affine.for` binds survive as dimensions — the rest are replaced by the constant 0,
/// under the reference's own reason: *"If the dim isn't used in the time_addr_map, or it's below the
/// preserved time dim index, it doesn't impact the subscripts."* ⛔ Which is the SAME statement
/// [`update_time_set_for_explicit_dims`] makes about the set: an explicit iteration transfers at step
/// 0 of everything still implicit, and the implicit walk supplies the rest of the address itself.
///
/// ⭐ THE SURVIVORS ARE RENUMBERED CONSECUTIVELY. `dim_num++` only advances on a kept dimension, so
/// the result is a dense map over `new_indices` — for the vendor's key,
/// `(d0, d1, d2, d3, d4) -> (d2 * 64, d0 * 16 + d3, d1 * 8 + d4 * 8)` becomes
/// `(d0, d1, d2) -> (d2 * 64, d0 * 16, d1 * 8)` with the new loop's induction variable appended to the
/// indices.
///
/// ⛔ `i > actual_preserve_dim` IS A ONE-SIDED TEST, so a dimension INNER to the preserved one is
/// dropped and one OUTER to it is kept even if no loop was created for it — which is exactly right,
/// because `constructExplicitTimeLoops` creates a loop for every dimension from 0 up to `time_dim`.
/// ⛔ And `isFunctionOfDim` is asked of `subscripts_map_time`, not of the time address map, so a time
/// dimension whose coefficient is zero costs nothing.
///
/// ⚠️ `Option` IS THE `DT_CHECK` — see [`ExplicitTimeLoops::TimeDimensionHasNoLoop`], which is
/// structurally unreachable when `ivs` comes from [`nest_explicit_time_loops`]'s own bound list.
/// ⚠️ `subscripts_map` IS NOT A PARAMETER: the reference reads only its `getNumDims()` and its
/// context off it, and the caller has that number already.
pub(super) fn update_subscripts_and_indices_for_explicit_time_loops(
    ivs: &[Val],
    time_dim: usize,
    num_orig_dims: usize,
    subscripts_map_time: &AffineMap,
    indices: &[Val],
) -> Option<(AffineMap, Vec<Val>)> {
    let mut replace_dims: Vec<AffineExpr> = Vec::new();
    let mut new_indices: Vec<Val> = Vec::new();
    let mut dim_num = 0;

    // The original map's dimensions and indices come through unchanged.
    for &index in indices.iter().take(num_orig_dims) {
        replace_dims.push(AffineExpr::dim(dim_num));
        dim_num += 1;
        new_indices.push(index);
    }

    // `int actual_preserve_dim = time_dim + num_orig_dims;`
    let actual_preserve_dim = time_dim + num_orig_dims;
    for i in num_orig_dims..subscripts_map_time.dims as usize {
        let position = u32::try_from(i).unwrap_or(u32::MAX);
        if i > actual_preserve_dim || !subscripts_map_time.is_function_of_dim(position) {
            replace_dims.push(AffineExpr::Const(0));
            continue;
        }
        replace_dims.push(AffineExpr::dim(dim_num));
        dim_num += 1;
        // `DT_CHECK((i - num_orig_dims) < for_ops.size());`
        new_indices.push(*ivs.get(i - num_orig_dims)?);
    }

    let subscripts_map =
        subscripts_map_time.replace_dims_and_symbols(&replace_dims, &[], dim_num, 0);
    Some((subscripts_map, new_indices))
}

/// WHAT [`init_mas_data`] LEAVES ITS CALLER — the filled list and the span, or the reason there is
/// neither. The two `DT_CHECK`s are variants: nothing here refuses at run time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MasDataInit {
    /// One entry per index, outermost first, plus `max_mutable`.
    Initialized {
        /// `mas_data` — the reference's out-parameter, in `ad.getIndices()` order.
        mas_data: Vec<MasData>,
        /// `max_mutable` — `iter_coeff_dict[nullptr]` plus every weight.
        max_mutable: MutableAddr,
    },
    /// `if (indices.size() == 0) return;` — the caller keeps the `max_mutable` it already had.
    NoIndices,
    /// `DT_CHECK(iter_coeff_dict.size() == indices.size() + 1)` (`:717`).
    CoefficientCountDoesNotMatchTheIndices {
        /// `indices.size()`.
        indices: usize,
        /// `iter_coeff_dict.size() - 1`, which is the coefficient vector's length.
        coefficients: usize,
    },
    /// `cast<BlockArgument>(index)` plus `DT_CHECK(loop)` (`:727-728`) — no region binds this index.
    IndexIsNotALoopIterator(Val),
    /// `DT_CHECK(num_iters >= 0)` (`:730`).
    LoopTripCountIsNotUsable {
        /// The index whose loop was asked.
        index: Val,
        /// What [`get_loop_trip_count`] answered.
        count: LoopTripCount,
    },
}

/// Replaces: e250_initMASData
///
/// **250/384** `MutableAddrSplittingPass::initMASData` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707` (31L): one [`MasData`] per subscript
/// index, and `max_mutable` as the subscripts' constant offset plus every weight.
///
/// ⛔ `weight = num_iters < 2 ? 0 : (num_iters - 2) * composed_coeff` (`:736`) — MINUS TWO, because
/// the iterator peaks at `num_iters - 1` and no transfer follows the last iteration, so
/// `num_iters - 2` is the last mutable address actually used.
#[must_use]
pub fn init_mas_data(ad: &AccessDetailsAffine<'_>, scope: &[DfirOp]) -> MasDataInit {
    // `auto indices = ad.getIndices(); if (indices.size() == 0) return;`
    let indices = &ad.base.indices;
    if indices.is_empty() {
        return MasDataInit::NoIndices;
    }

    // `auto iter_coeff_dict = ad.getIndicesCoeffDict();` — `per_index[i]` pairs `indices[i]`, so the
    // `+ 1` of the reference's length check is the dict's named constant and this is the rest.
    let dict = &ad.indices_coeff_dict;
    if dict.per_index.len() != indices.len() {
        return MasDataInit::CoefficientCountDoesNotMatchTheIndices {
            indices: indices.len(),
            coefficients: dict.per_index.len(),
        };
    }

    // `max_mutable = iter_coeff_dict[nullptr];` — the span starts at the constant offset.
    let mut max_mutable = MutableAddr(dict.constant);
    let mut mas_data = Vec::with_capacity(indices.len());

    for (i, (&index, &(_, composed_coeff))) in indices.iter().zip(&dict.per_index).enumerate() {
        // `auto loop = cast<BlockArgument>(index).getOwner()->getParentOp(); DT_CHECK(loop);`
        let Some(loop_op) = owner_of_block_arg(index, scope) else {
            return MasDataInit::IndexIsNotALoopIterator(index);
        };

        // `auto num_iters = getLoopTripCount(loop); DT_CHECK(num_iters >= 0);`
        let count = get_loop_trip_count(loop_op, scope);
        let Some(num_iters) = count.iterations().filter(|iters| *iters >= 0) else {
            return MasDataInit::LoopTripCountIsNotUsable { index, count };
        };

        let weight = if num_iters < 2 {
            0
        } else {
            (num_iters - 2).saturating_mul(composed_coeff)
        };

        // `mas_data.emplace_back(index, i, composed_coeff, num_iters, weight);`
        mas_data.push(MasData {
            iter_arg: Some(index),
            dim: u32::try_from(i).unwrap_or(u32::MAX),
            composed_coeff,
            num_iters,
            weight,
        });
        // `max_mutable += weight;`
        max_mutable.add_weight(weight);
    }

    MasDataInit::Initialized {
        mas_data,
        max_mutable,
    }
}

/// WHAT THE THREE-STATEMENT SETUP ANSWERS — the sizes, or the eligibility check that stopped it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupForPartitioning {
    /// `calculatePartitionSizes(...)`'s answer, over a `mas_data` now sorted by weight.
    Sized(Partitioning),
    /// `DT_CHECK(isEligibleForSplitting(all_mem_views))` (`:826`) — and nothing is sorted.
    NotEligibleForSplitting(SplittingEligibility),
}

/// Replaces: e251_setupForPartitioning
///
/// **251/384** `MutableAddrSplittingPass::setupForPartitioning` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819` (6L): the three statements that stand
/// between an overflow and a partition plan — eligibility, sort, size.
///
/// ⛔ THE SORT IS THE MIDDLE STATEMENT AND IT MUTATES THE CALLER'S `mas_data` (`:827`), which is what
/// lets [`calculate_partition_sizes`] stop at the heaviest dimension; the ineligible arm returns
/// having sorted nothing.
pub fn setup_for_partitioning<A: Arch>(
    mas_data: &mut [MasData],
    all_mem_views: &[Val],
    half: L3Half,
    view: &ConstStartMemView<'_>,
    max_mutable: MutableAddr,
    elem: DataType,
    num_conditionals: &mut Conditionals,
    scope: &[DfirOp],
) -> SetupForPartitioning {
    // `DT_CHECK(isEligibleForSplitting(all_mem_views));`
    let eligibility = is_eligible_for_splitting(all_mem_views, scope);
    if !eligibility.eligible() {
        return SetupForPartitioning::NotEligibleForSplitting(eligibility);
    }

    // `sortDataBasedOnWeight(mas_data);`
    sort_data_based_on_weight(mas_data);

    // `calculatePartitionSizes(evaluator, mas_data, partition_sizes, comp, mem_view_op,
    //                          max_mutable, elem_size_in_bits);`
    SetupForPartitioning::Sized(calculate_partition_sizes::<A>(
        mas_data,
        half,
        view,
        max_mutable,
        elem,
        num_conditionals,
    ))
}

/// WHAT THE PARITY REALIGNMENT DID — and every state the reference reaches instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvenImmutableAdjustment {
    /// `if (dcc_ext_ctx_.getArch() < IsaCoreGen::SEN1P5_ISA) return;` (`:1208`).
    NotOnThisArch,
    /// `is_even == is_even_mod` — nothing to move.
    ParitiesAlreadyAgree,
    /// A stick left `immutable_addr_mod` and joined the subscript at this result.
    ShiftedByAStick {
        /// The `transfer_order` result that is a function of `d0`.
        result: usize,
        /// `num_elems_in_stick`.
        elems_in_stick: u64,
    },
    /// `DT_CHECK(res < num_res)` (`:1232`) — no `transfer_order` result mentions `d0`.
    NoTransferOrderResultUsesTheInnermostDim,
    /// `DT_CHECK_MSG(ad.getExtents()[res] >= num_elems_in_stick, "Innermost dimension extend must fit
    /// a full stick.")` (`:1241-1242`).
    InnermostExtentIsSmallerThanAStick {
        /// `ad.getExtents()[res]`.
        extent: Elements,
        /// `num_elems_in_stick`.
        elems_in_stick: u64,
    },
    /// The same subscript, with no extent recorded for it — `getExtents()` and `getTransferOrder()`
    /// are built together (`AccessDetailsBase::constructExtentAndTotalElements`) and disagree here.
    ExtentsDoNotCoverTheTransferOrder {
        /// `transfer_order.getNumResults()`.
        results: usize,
        /// `ad.getExtents().size()`.
        extents: usize,
    },
}

/// Replaces: e253_adjustForEvenImmutableAddr
///
/// **253/384** `MutableAddrSplittingPass::adjustForEvenImmutableAddr` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204` (47L): from SEN1P5 on, a partition whose
/// shift has the opposite stick parity to the immutable address gives one stick back and adds those
/// elements to the `d0` subscript instead.
///
/// ⛔ THE STICK MOVES TO THE FIRST `transfer_order` RESULT THAT IS A FUNCTION OF `d0`, NOT TO RESULT 0
/// (`:1229-1232`) — `d0`'s coefficient is always 1, which is what makes the addition a plain one.
/// ⭐ AND THIS ONE KEEPS `getNumSymbols()` (`:1251`), unlike [`fill_partitions`].
/// ⚠️ The `is_even ? true : isL3ImmutableAddrAllOdd(...)` check cannot fire on a constant start — the
/// two predicates are exact complements there; see [`calculate_partition_sizes`] for the full reason.
pub fn adjust_for_even_immutable_addr<A: Arch>(
    immutable_addr: i64,
    subscripts_map: &mut AffineMap,
    ad: &AccessDetailsAffine<'_>,
    immutable_addr_mod: &mut i64,
    elem: DataType,
) -> EvenImmutableAdjustment {
    if A::GEN < IsaGen::Sen1p5 {
        return EvenImmutableAdjustment::NotOnThisArch;
    }

    // `int num_elems_in_stick = dcc_ext_ctx_.getBytesPerStick() * 8 / elem_size_in_bits;`
    let stick = elems_in_stick::<A>(elem);
    let elems_in_stick = stick.get();

    // `isL3ImmutableAddrEven(evaluator, immutable_addr, num_elems_in_stick)` —
    // `evaluateDivideByConst(ev, n).isDivisibleBy(2)` over the single constant address.
    let is_even = (immutable_addr / elems_in_stick.cast_signed()) % 2 == 0;
    // `bool is_even_mod = immutable_addr_mod % 2 == 0;`
    let is_even_mod = *immutable_addr_mod % 2 == 0;
    if is_even == is_even_mod {
        return EvenImmutableAdjustment::ParitiesAlreadyAgree;
    }

    // `auto transfer_order = ad.getTransferOrder();` then the first result naming `d0`.
    let transfer_order = &ad.base.transfer_order;
    let num_res = transfer_order.results.len();
    let Some(res) = (0..num_res).find(|&res| transfer_order.results[res].is_function_of_dim(0))
    else {
        // `DT_CHECK(res < num_res);`
        return EvenImmutableAdjustment::NoTransferOrderResultUsesTheInnermostDim;
    };

    // `DT_CHECK_MSG(ad.getExtents()[res] >= num_elems_in_stick, …);`
    let Some(&extent) = ad.base.extents.get(res) else {
        return EvenImmutableAdjustment::ExtentsDoNotCoverTheTransferOrder {
            results: num_res,
            extents: ad.base.extents.len(),
        };
    };
    if extent.0 < elems_in_stick {
        return EvenImmutableAdjustment::InnermostExtentIsSmallerThanAStick {
            extent,
            elems_in_stick,
        };
    }

    // `immutable_addr_mod -= num_elems_in_stick;` — ⚠️ DELIBERATE DIVERGENCE: the reference subtracts
    // at `:1222`, BEFORE the `res` search and before both `DT_CHECK`s (`:1232`, `:1241-1242`), so it
    // leaves the caller's `immutable_addr_mod` short of a stick on a path it then aborts. Moved after.
    *immutable_addr_mod = immutable_addr_mod.saturating_sub(elems_in_stick.cast_signed());

    // `if (i == res) expr = expr + num_elems_in_stick;` over every result, then
    // `AffineMap::get(getNumDims(), getNumSymbols(), new_exprs, getContext())`.
    let stick_elems = elems_in_stick.cast_signed();
    for (i, expr) in subscripts_map.results.iter_mut().enumerate() {
        if i == res {
            *expr = expr.clone().added(AffineExpr::Const(stick_elems));
        }
    }

    EvenImmutableAdjustment::ShiftedByAStick {
        result: res,
        elems_in_stick,
    }
}

/// WHICH ARM OF AN `scf.if` A NODE IS — `CondNode::isThenNode()` / `isElseNode()`
/// (`dcc/src/Analysis/ConditionalTree.hpp:89-90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// The `then` region.
    Then,
    /// The `else` region.
    Else,
}

/// ONE `CondNode` HOP: the `scf.if` a node hangs under, and which of its two arms.
///
/// ⭐ THE PAIR IS THE REFERENCE'S TWO `getParentNode()` CALLS. `curr_node->getParentNode()` is the
/// `if` node and its parent again is the arm node above it (`MutableAddrSplitting.cpp:1166`), so one
/// [`Step`] is one partitioned dimension's worth of walking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Step {
    /// Where the `scf.if` sits in the region that holds it.
    at: usize,
    /// Which of its regions this node is.
    arm: Arm,
    /// `curr_if_op.getCondition()` — the `arith.cmpi` result.
    cond: Val,
}

/// A `CondNode` DURING THE BREADTH-FIRST WALK — an `scf.if` node or one of its two arm nodes.
#[derive(Debug, Clone)]
enum CondNode {
    /// An `if` node: the path to the region holding it, and where in that region it is.
    If {
        /// The arms walked through to reach it.
        path: Vec<Step>,
        /// Its index in that region.
        at: usize,
        /// Its condition.
        cond: Val,
    },
    /// A `then` or `else` node, addressed by the whole path to it.
    Arm(Vec<Step>),
}

/// THE REGION A PATH OF ARMS LEADS TO.
fn region_at<'a>(tree: &'a [DfirOp], path: &[Step]) -> Option<&'a [DfirOp]> {
    let mut region = tree;
    for step in path {
        let DfirOp::Scf(scf::Op::If {
            body, else_body, ..
        }) = region.get(step.at)?
        else {
            return None;
        };
        region = match step.arm {
            Arm::Then => body,
            Arm::Else => else_body,
        };
    }
    Some(region)
}

/// The same walk, for the region a partition's ops are inserted into.
fn region_at_mut<'a>(tree: &'a mut Vec<DfirOp>, path: &[Step]) -> Option<&'a mut Vec<DfirOp>> {
    let mut region = tree;
    for step in path {
        let DfirOp::Scf(scf::Op::If {
            body, else_body, ..
        }) = region.get_mut(step.at)?
        else {
            return None;
        };
        region = match step.arm {
            Arm::Then => body,
            Arm::Else => else_body,
        };
    }
    Some(region)
}

/// The `scf.if`s a region holds, in syntactic order — one arm node's children.
fn conditionals_in(region: &[DfirOp]) -> Vec<(usize, Val)> {
    region
        .iter()
        .enumerate()
        .filter_map(|(at, op)| match op {
            DfirOp::Scf(scf::Op::If { cond, .. }) => Some((at, *cond)),
            _ => None,
        })
        .collect()
}

/// `CondNode::walk<kReverseBFS>` OVER THE LEAVES — `isLeafNode()` is an arm node with no `scf.if` in
/// it (`ConditionalTree.hpp:91`).
///
/// ⛔⛔ BREADTH-FIRST, COLLECTED INTO A STACK AND THEN POPPED (`OperationTree.cpp:200-236`, with
/// `keep_order = false`), so the DEEPEST partition is filled FIRST and same-depth siblings run
/// `else` before `then`. The order is what decides which partition's ops are built first, so it is
/// part of the answer rather than a detail of the walk.
fn reverse_bfs_leaves(tree: &[DfirOp], root: (usize, Val)) -> Vec<Vec<Step>> {
    let mut order: Vec<CondNode> = Vec::new();
    let mut queue: VecDeque<CondNode> = VecDeque::new();
    queue.push_back(CondNode::If {
        path: Vec::new(),
        at: root.0,
        cond: root.1,
    });

    while let Some(node) = queue.pop_front() {
        match &node {
            // `Every CondNode corresponding to an If op has two children - a ThenNode and an
            // ElseNode` (`ConditionalTree.hpp:28-29`), in that order.
            CondNode::If { path, at, cond } => {
                for arm in [Arm::Then, Arm::Else] {
                    let mut child = path.clone();
                    child.push(Step {
                        at: *at,
                        arm,
                        cond: *cond,
                    });
                    queue.push_back(CondNode::Arm(child));
                }
            }
            CondNode::Arm(path) => {
                let children = region_at(tree, path)
                    .map(conditionals_in)
                    .unwrap_or_default();
                for (at, cond) in children {
                    queue.push_back(CondNode::If {
                        path: path.clone(),
                        at,
                        cond,
                    });
                }
            }
        }
        order.push(node);
    }

    order.reverse();
    order
        .into_iter()
        .filter_map(|node| match node {
            CondNode::Arm(path) => region_at(tree, &path)
                .filter(|region| conditionals_in(region).is_empty())
                .map(|_| path),
            CondNode::If { .. } => None,
        })
        .collect()
}

/// WHAT `fillPartitions` DID TO THE TREE, OR THE `DT_CHECK` THAT STOPPED IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilledPartitions {
    /// Every leaf received `op_creator`'s ops.
    Filled {
        /// How many leaves were filled — the tree's partitions.
        partitions: usize,
    },
    /// `calculateSubscriptsCoefficients`'s refusal, passed through unchanged.
    CoefficientsNotExtracted(SubscriptsCoefficients),
    /// `DT_CHECK(coeffs.size() == subscripts_map.getNumResults())` (`:1152`) — ⭐ hoisted out of the
    /// leaf walk, which reads the same rows for every partition, so it fires before any leaf is
    /// filled rather than part-way through.
    CoefficientRowDoesNotCoverTheSubscripts {
        /// `mas_data[p].dim_`.
        dim: u32,
        /// `coeffs.size()`.
        coefficients: usize,
        /// `subscripts_map.getNumResults()`.
        results: usize,
    },
    /// `mas_data[p]` for a `partition_sizes` position with no iterator.
    MorePartitionsThanIterators {
        /// `partition_sizes.size()`.
        sizes: usize,
        /// `mas_data.size()`.
        iterators: usize,
    },
    /// A LEAF NESTED LESS DEEPLY THAN THERE ARE PARTITIONED DIMENSIONS.
    ///
    /// ⛔⛔ THE OTHER HALF OF [`construct_conditionals`]'S `prev_partitions` DEFECT. The walk takes
    /// exactly `partition_sizes.size()` steps up from every leaf (`:1119-1167`), and the branches
    /// that lost their inner dimensions are shallower than that — where the reference calls
    /// `getParentNode()` on the root and dereferences null.
    PartitionIsShallowerThanThePlan {
        /// How many arms this leaf sits under.
        depth: usize,
        /// `partition_sizes.size()`.
        dimensions: usize,
    },
    /// `cast<arith::CmpIOp>` on the condition, or `cast<arith::ConstantOp>` on its right-hand side
    /// (`:1127-1136`).
    ConditionIsNotACompareAgainstAConstant(Val),
    /// The tree holds no `scf.if` at all, so `cond_tree.getRoot()` is not an `if` node.
    TreeHasNoRootConditional,
}

/// Replaces: e252_fillPartitions
///
/// **252/384** `MutableAddrSplittingPass::fillPartitions` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063` (117L): every leaf of the conditional
/// tree is one partition, and its address is recovered by reading the conditionals above it — one per
/// partitioned dimension, innermost first.
///
/// ⛔ `prev_iters` IS `demarkation - partition_sizes[p]` IN A `then` ARM AND `demarkation` IN AN
/// `else` (`:1142-1143`); it scales BOTH the start address (`* composed_coeff_`) and, negated, every
/// subscript (`* coeffs[r]`). ⛔⛔ AND THE REBUILT MAP HARD-CODES **ZERO SYMBOLS** (`:1171-1172`),
/// dropping any the original declared — unlike [`adjust_for_even_immutable_addr`], which preserves
/// them at `:1251`.
pub fn fill_partitions(
    vals: &mut Values,
    cond_tree: &mut Vec<DfirOp>,
    mas_data: &[MasData],
    partition_sizes: &[i64],
    subscripts_map: &AffineMap,
    op_creator: &mut impl FnMut(&mut Values, &AffineMap, i64) -> Vec<DfirOp>,
) -> FilledPartitions {
    if partition_sizes.len() > mas_data.len() {
        return FilledPartitions::MorePartitionsThanIterators {
            sizes: partition_sizes.len(),
            iterators: mas_data.len(),
        };
    }

    // `auto subscripts_coeffs = calculateSubscriptsCoefficients(mas_data, partition_sizes,
    //                                                           subscripts_map);`
    let rows = match calculate_subscripts_coefficients(mas_data, partition_sizes, subscripts_map) {
        SubscriptsCoefficients::Extracted(rows) => rows,
        refusal => return FilledPartitions::CoefficientsNotExtracted(refusal),
    };
    let results = subscripts_map.results.len();
    for row in &rows {
        // `DT_CHECK(coeffs.size() == subscripts_map.getNumResults());`
        if row.coeffs.len() != results {
            return FilledPartitions::CoefficientRowDoesNotCoverTheSubscripts {
                dim: row.dim,
                coefficients: row.coeffs.len(),
                results,
            };
        }
    }

    let Some(&root) = conditionals_in(cond_tree).first() else {
        return FilledPartitions::TreeHasNoRootConditional;
    };
    let leaves = reverse_bfs_leaves(cond_tree, root);
    let mut partitions = 0usize;

    for path in leaves {
        // The walk takes one [`Step`] per partitioned dimension, from the leaf outward.
        if path.len() < partition_sizes.len() {
            return FilledPartitions::PartitionIsShallowerThanThePlan {
                depth: path.len(),
                dimensions: partition_sizes.len(),
            };
        }

        // `SmallVector<AffineExpr> new_exprs; for (auto &res : subscripts_map.getResults()) ..`
        let mut new_exprs = subscripts_map.results.clone();
        let mut start_addr_mod = 0i64;

        // `for (int p = partition_sizes.size() - 1; p >= 0; --p)` — and `curr_node` climbs two
        // `CondNode`s per dimension, which is one [`Step`] per dimension here.
        for (up, p) in (0..partition_sizes.len()).rev().enumerate() {
            let step = path[path.len() - 1 - up];

            // `cast<arith::CmpIOp>(curr_if_op.getCondition().getDefiningOp())`. ⚠️ `cond.getLhs()` is
            // bound to `iter` there and never read; the predicate is not checked either.
            let Some(DfirOp::Arith(arith::Op::Compare { rhs, .. })) =
                defining_op(step.cond, cond_tree)
            else {
                return FilledPartitions::ConditionIsNotACompareAgainstAConstant(step.cond);
            };
            // `cast<IntegerAttr>(cast<arith::ConstantOp>(cond.getRhs().getDefiningOp()).getValue())`.
            let Some(DfirOp::Arith(arith::Op::Constant { value, .. })) =
                defining_op(*rhs, cond_tree)
            else {
                return FilledPartitions::ConditionIsNotACompareAgainstAConstant(step.cond);
            };
            let demarkation = *value;

            // `prev_iters = is_then_node ? demarkation - partition_sizes[p] : demarkation;`
            let prev_iters = match step.arm {
                Arm::Then => demarkation.saturating_sub(partition_sizes[p]),
                Arm::Else => demarkation,
            };

            // `start_addr_mod += prev_iters * mas_data[p].composed_coeff_;`
            start_addr_mod = start_addr_mod
                .saturating_add(prev_iters.saturating_mul(mas_data[p].composed_coeff));

            // `new_exprs[r] = new_exprs[r] - (coeffs[r] * prev_iters);` — MLIR's `operator-(int64_t)`
            // is `*this + (-v)`, so this is the same `simplifyAdd` [`AffineExpr::added`] performs.
            for (expr, &coeff) in new_exprs.iter_mut().zip(&rows[p].coeffs) {
                let shift = coeff.saturating_mul(prev_iters);
                *expr = expr.clone().added(AffineExpr::Const(-shift));
            }
        }

        // `AffineMap::get(subscripts_map.getNumDims(), 0, new_exprs, ..)` — the symbol count is a
        // literal `0`, not `subscripts_map.getNumSymbols()`.
        let new_subscripts_map = AffineMap {
            dims: subscripts_map.dims,
            syms: 0,
            results: new_exprs,
        };

        // `op_creator(partition_builder, new_subscripts_map, start_addr_mod)`, where the builder is
        // `getThenBodyBuilder()` or `getElseBodyBuilder()` — `atBlockTerminator` for a resultless
        // `scf.if`, so the ops land before the region's `scf.yield`.
        // ⚠️ `vals` IS THE `partition_builder` — the reference hands the creator a builder because
        // that is what makes ops; here what a creator needs from its caller is the value minter.
        let created = op_creator(vals, &new_subscripts_map, start_addr_mod);
        let Some(region) = region_at_mut(cond_tree, &path) else {
            return FilledPartitions::TreeHasNoRootConditional;
        };
        let at = region
            .iter()
            .position(|op| matches!(op, DfirOp::Scf(scf::Op::Yield { .. })))
            .unwrap_or(region.len());
        region.splice(at..at, created);
        partitions += 1;
    }

    FilledPartitions::Filled { partitions }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 289/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT THE PASS'S PER-CANDIDATE SETUP LEAVES BEHIND — the initialized list, or the precondition it
/// stopped on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MasInitialization {
    /// [`init_mas_data`]'s answer, passed through unchanged.
    Initialized(MasDataInit),
    /// `DT_CHECK(dcc::agen::utils::hasValidL3ImmutableAddr(evaluator, mem_view_op,
    /// num_elems_in_stick))` (`:700-701`).
    ImmutableAddrIsNotAWholeNumberOfSticks {
        /// The view's constant start address, in elements.
        start: i64,
        /// `num_elems_in_stick`.
        elems_in_stick: NonZeroU64,
    },
    /// ⛔ THE WIDTH IS A DIVISOR TWICE OVER: `getBytesPerStick() * 8 / ad.getElementWidth()` traps on
    /// the `element_width_ = 0` an unpopulated record still carries, and a width wider than a stick
    /// makes the quotient zero — which `isDivisibleBy` then divides by
    /// (`ExpressionEvaluatorUtils.cpp:115`).
    NoElementsFitInAStick {
        /// `ad.getElementWidth()`.
        width: Bits,
    },
}

/// Replaces: e289_initialize
///
/// **289/384** `MutableAddrSplittingPass::initialize` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694` (8L): the immutable address must be a
/// whole number of sticks, and then [`init_mas_data`] collects the iterators and the mutable span.
///
/// ⛔ THE `initMASData` CALL IS THE TAIL `bridge2.cpp` TRUNCATED (AGENT-BRIEF §1); without it
/// `mas_data` reaches `synthesizeTimeInfo` empty and no candidate is ever split.
/// ⭐ A [`ConstStartMemView`] PINS ARM ONE of `hasValidL3ImmutableAddr`
/// (`Dialect/Agen/Utils.cpp:147-151`), whose whole content there is the divisibility.
#[must_use]
pub fn initialize<A: Arch>(
    ad: &AccessDetailsAffine<'_>,
    view: &ConstStartMemView<'_>,
    scope: &[DfirOp],
) -> MasInitialization {
    // `int64_t num_elems_in_stick = dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();`
    let width = ad.base.element_width;
    let Some(elems_in_stick) = (width.0 > 0)
        .then(|| A::BYTES_PER_STICK.get() * 8 / u64::from(width.0))
        .and_then(NonZeroU64::new)
    else {
        return MasInitialization::NoElementsFitInAStick { width };
    };

    // `DT_CHECK(hasValidL3ImmutableAddr(evaluator, mem_view_op, num_elems_in_stick));` — the helper's
    // constant-start arm is `ev.isDivisibleBy(num_elems_in_stick)`, i.e. `offsetValue() % divisor == 0`.
    let stick = i64::try_from(elems_in_stick.get()).unwrap_or(i64::MAX);
    if view.start.rem_euclid(stick) != 0 {
        return MasInitialization::ImmutableAddrIsNotAWholeNumberOfSticks {
            start: view.start,
            elems_in_stick,
        };
    }

    // *"Initialize MASData which also collects the max mutable address."*
    MasInitialization::Initialized(init_mas_data(ad, scope))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 290/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT GOES WHERE THE TRANSFER WAS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partitions {
    /// The conditional tree with every leaf filled.
    Created {
        /// [`construct_conditionals`]'s ops, now holding one partition per leaf.
        ops: Vec<DfirOp>,
        /// How many leaves [`fill_partitions`] filled.
        partitions: usize,
    },
    /// `DT_CHECK(!partition_sizes.empty())` (`:989`) and `DT_CHECK_MSG(root_op, "Root op of
    /// conditional tree could not be determined")` (`:991`) — both are the tree not being built.
    NoConditionalTree(ConditionalTree),
    /// [`fill_partitions`]'s refusal, with the partly filled tree dropped as the abort drops it.
    NotFilled(FilledPartitions),
}

/// Replaces: e290_createPartitions
///
/// **290/384** `MutableAddrSplittingPass::createPartitions` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983` (10L): build the conditional tree, then
/// fill each of its leaves with one partition of the transfer.
///
/// ⛔ `dcc::ConditionalTree cond_tree(*root_op); cond_tree.compute();` (`:993-994`) HAS NOTHING TO DO
/// HERE: it caches the parent/child links of ops already built, and [`ConditionalTree::Built`] holds
/// those ops by value — [`fill_partitions`] walks them and reads the conditionals above each leaf.
#[must_use]
pub fn create_partitions(
    vals: &mut Values,
    mas_data: &[MasData],
    partition_sizes: &[i64],
    subscripts_map: &AffineMap,
    op_creator: &mut impl FnMut(&mut Values, &AffineMap, i64) -> Vec<DfirOp>,
) -> Partitions {
    // `DT_CHECK(!partition_sizes.empty()); auto root_op = constructConditionals(..);
    //  DT_CHECK_MSG(root_op, ..);` — [`construct_conditionals`] asks the emptiness itself, so both
    // checks are one absent tree.
    let tree = construct_conditionals(vals, mas_data, partition_sizes);
    let Some(ops) = tree.ops() else {
        return Partitions::NoConditionalTree(tree);
    };
    let mut ops = ops.to_vec();

    // `fillPartitions(cond_tree, mas_data, partition_sizes, subscripts_map, op_creator);`
    match fill_partitions(
        vals,
        &mut ops,
        mas_data,
        partition_sizes,
        subscripts_map,
        op_creator,
    ) {
        FilledPartitions::Filled { partitions } => Partitions::Created { ops, partitions },
        refusal => Partitions::NotFilled(refusal),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 306/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE TRANSFER THE PASS DECIDED TO LOOK AT — `MutableAddrSplittingPass::MASCandidate`
/// (`dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:91-101`).
///
/// ⚠️ UNANCHORED: the pass's nested data members are among the 106 excluded entries, and entry 306 is
/// the first thing that needs one. ⭐ `dataflow::ProgramUnitOp unit_` IS ABSENT: its only use is
/// `OpBuilder const_builder(unit)` (`:970`), an insertion point, which is mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasCandidate<'a> {
    /// `op_` — the memory operation.
    pub op: &'a DfirOp,
    /// `comp_` — the unit it runs on. The two L3 halves are the only ones [`L3Half::of`] admits.
    pub comp: DfirUnit,
    /// `mem_index_` — which memory operand of it is under consideration.
    pub mem_index: MemoryOperandIndex,
}

/// WHAT THE PASS LEFT WHERE THE TRANSFER WAS — and every precondition it stopped on instead.
///
/// ⛔ NOT AN ERROR TYPE. Six of these are `DT_CHECK`s or `cast<>`s that abort the compiler, and
/// [`Self::AddressFits`] is the reference's own `return` for a transfer whose address does not
/// overflow — by far the common case, and NOT a refusal.
///
/// ⭐ ONE ENUM FOR ENTRIES 306 AND 307: `transformVectorLoad` and `transformVectorStore` differ in
/// their opening `dyn_cast` and in nothing else they can stop on, so the two casts are two variants
/// and every other refusal is shared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSplit<'a> {
    /// The conditional tree, one partition per leaf, and the chain it replaces.
    Split {
        /// [`create_partitions`]'s ops, to be placed where the load was.
        ops: Vec<DfirOp>,
        /// ⛔ THE PARTITIONS' START-ADDRESS CONSTANTS, WHICH DO **NOT** GO IN `ops`: `const_builder`
        /// is seated at the `dataflow.program_unit` (`:970`), so they land OUTSIDE the unit's body —
        /// in fill order, which is the vendor's `%c10240` before `%c2048`. See [`NewMemViewWithMod`].
        hoisted: Vec<DfirOp>,
        /// `op.eraseOpAndUseChain()` (`:369`) — the delete list, consumer first.
        erased: Vec<&'a DfirOp>,
        /// How many leaves were filled.
        partitions: usize,
        /// One [`adjust_for_even_immutable_addr`] answer per partition, in fill order. ⛔ THE
        /// REFERENCE DISCARDS THESE (the call is `void`, `:344-346`) and its two `DT_CHECK`s abort
        /// from inside the lambda; carried so the caller can see what the abort would have been.
        adjustments: Vec<EvenImmutableAdjustment>,
    },
    /// `if (!hasMutableAddrOverflow(...)) return;` (`:317-319`) — the load is left exactly as it was.
    ///
    /// ⚠️ AND IT CARRIES THAT CALL'S TWO ABORTS TOO ([`MutableAddrOverflow::NoLoopsToSplit`],
    /// [`MutableAddrOverflow::OverflowCorrectionForbidden`]), because the reference's `!` cannot tell
    /// them from [`MutableAddrOverflow::InRange`] either. The payload is what distinguishes them.
    AddressFits(MutableAddrOverflow),
    /// `auto op = dyn_cast<agen::VectorLoadOp>(candidate.op_); DT_CHECK(op);` (`:299-300`).
    NotAVectorLoad,
    /// `auto op = dyn_cast<agen::VectorStoreOp>(candidate.op_); DT_CHECK(op);` (`:376-377`).
    NotAVectorStore,
    /// `DT_CHECK(succeeded(ad.constructDetails(candidate.mem_index_)));` (`:307-308`).
    DetailsNotConstructed(ConstructedDetails),
    /// `initialize()` left `subscripts_map_` null, which `ad.getSubscriptsMap()` (`:367`) dereferences.
    ///
    /// ⚠️ UNREACHABLE AFTER [`ConstructedDetails::Complete`] — `AccessDetails.cpp:305` sets it for
    /// every vector load — which is why reading the map above the overflow gate rather than at
    /// `:367` cannot turn the reference's plain `return` into this.
    SubscriptsMapIsNotSet,
    /// `cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp())` (`:313-314`).
    MemRefIsNotAMemoryView,
    /// The view's start address is not an `arith.constant`: `hasValidL3ImmutableAddr`'s other two arms
    /// (`Dialect/Agen/Utils.cpp:152-163`), and `DT_CHECK(isEligibleForSplitting(..))` (`:826`).
    MemViewStartIsNotConstant,
    /// `DT_CHECK(is_any_of(comp, L3LU, L3SU))`, which both range queries open with.
    CandidateIsNotOnAnL3Half(DfirUnit),
    /// [`initialize`]'s refusal (`:316`).
    NotInitialized(MasInitialization),
    /// [`setup_for_partitioning`]'s refusal (`:334-336`).
    NotSetUpForPartitioning(SetupForPartitioning),
    /// [`create_partitions`]'s refusal (`:367-368`).
    NotPartitioned(Partitions),
    /// `auto op = dyn_cast<agen::CompositeLoadAndStoreOp>(candidate.op_); DT_CHECK(op);` (`:453-454`).
    NotACompositeLoadAndStore,
    /// `dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_)` with its `DT_CHECK` (`:548`).
    NotACompositeIndirectLoadAndStore,
    /// THE OPERAND UNDER CONSIDERATION ALREADY HAS AN ENTRY — `insert`'s first `DT_CHECK_MSG`
    /// (`AccessDetails.hpp:389-390`), which a container built one line earlier cannot fail.
    AccessSlotUnavailable(MemoryOperandIndex),
    /// The `result &= ..constructTimeStepsInfo(access_details, false, false).succeeded()` half of
    /// `DT_CHECK(result)` (`:462-466`).
    TimeStepsNotConstructed(TimeStepsInfo),
    /// `initialize()` left `time_set_` unset, which `ad.getTimeSet()` (`:490`) hands to every clone.
    ///
    /// ⚠️ UNREACHABLE AFTER [`TimeStepsInfo::Constructed`], whose first act is to read it.
    TimeSetIsNotSet,
    /// `DT_CHECK(candidate.mem_index_ == kDirSrc ? !op.hasIndirectSrc() : !op.hasIndirectDst())`
    /// (`:581-585`) — *"The immutable address must be zero for the side of the transfer involving the
    /// indirect."*
    IndirectIsOnTheSplitSide(MemoryOperandIndex),
    /// [`create_explicit_time_loops`]'s refusal (`:492-493`) — boxed, being the largest variant.
    TimeLoopsNotCreated(Box<ExplicitTimeLoops>),
}

/// Replaces: e306_transformVectorLoad
///
/// **306/384** `MutableAddrSplittingPass::transformVectorLoad` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298` (75L): split one overflowing
/// `agen.vector_load` into one partition per branch of a conditional tree.
///
/// ⛔ THE STORE'S VIEW IS CLONED **PER PARTITION** AND NOT SHARED (`:351-360`) — *"Every memory
/// operand should have it's own unique mem view"* — and it is collected from the LOAD's use chain
/// (`:325-330`) because a load/store pattern is merged into one transfer later, so both halves must be
/// eligible before either is split.
/// ⛔ AND THE ADJUSTMENT RUNS INSIDE THE LAMBDA (`:339-341`), once per partition, on that partition's own
/// `start_addr_mod` — hoisting it would apply one leaf's parity correction to every leaf.
#[must_use]
pub fn transform_vector_load<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    elem: DataType,
    correction: EarOverflowCorrection,
    num_conditionals: &mut Conditionals,
    scope: &'s [DfirOp],
) -> TransferSplit<'s> {
    // `auto op = dyn_cast<agen::VectorLoadOp>(candidate.op_); DT_CHECK(op);`
    let (Some(op), DfirOp::Agen(agen_op)) = (VectorLoadOp::of(candidate.op), candidate.op) else {
        return TransferSplit::NotAVectorLoad;
    };

    // "Collect the relevant access details." — `access_details.emplace_insert(mem_index, op, comp)`
    // then `DT_CHECK(succeeded(ad.constructDetails(mem_index)))`. ⚠️ THE CONTAINER IS MECHANISM: it
    // memoises one record per memory operand, and a vector load has exactly one.
    let mut ad = AccessDetailsAffine::new(agen_op, candidate.comp);
    let details = ad.construct_details(candidate.mem_index, scope);
    if details != ConstructedDetails::Complete {
        return TransferSplit::DetailsNotConstructed(details);
    }
    let Some(subscripts_map) = ad.subscripts_map.clone() else {
        return TransferSplit::SubscriptsMapIsNotSet;
    };

    // `cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp())`.
    let view = match SplitCandidateView::resolve(op.view, scope) {
        SplitCandidateView::ConstantStart(view) => view,
        SplitCandidateView::NonConstantStart => return TransferSplit::MemViewStartIsNotConstant,
        SplitCandidateView::NotAMemoryView => return TransferSplit::MemRefIsNotAMemoryView,
    };

    // `int64_t max_mutable = 0; initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);`
    let initialized = initialize::<A>(&ad, &view, scope);
    let (mut mas_data, max_mutable) = match initialized {
        MasInitialization::Initialized(MasDataInit::Initialized {
            mas_data,
            max_mutable,
        }) => (mas_data, max_mutable),
        // `if (indices.size() == 0) return;` leaves the caller's `mas_data` empty and its
        // `max_mutable` at the `0` it was initialised with, which no range is exceeded by.
        MasInitialization::Initialized(MasDataInit::NoIndices) => (Vec::new(), MutableAddr(0)),
        refusal => return TransferSplit::NotInitialized(refusal),
    };

    // `if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable, ad.getElementWidth()))
    //  return;` — ⚠️ `elem` where the reference passes the bare width: see [`AddrRange::elements`].
    // ⭐ AND THE L3 GATE BELONGS HERE, NOT EARLIER: it is `getMaxMutableRange`'s own `DT_CHECK`, which
    // the reference reaches only after `initialize` has had its say.
    let Some(half) = L3Half::of(candidate.comp) else {
        return TransferSplit::CandidateIsNotOnAnL3Half(candidate.comp);
    };
    let overflow = has_mutable_addr_overflow::<A>(&mas_data, half, max_mutable, elem, correction);
    if overflow != MutableAddrOverflow::Overflow {
        return TransferSplit::AddressFits(overflow);
    }

    // "Vector load and store patterns, despite being two separate operations, will be later merged
    // into one. Collect mem views from load and store patterns, if any."
    let mut all_mem_views = vec![op.view];
    let mut store_mem_view: Option<Val> = None;
    if let UseChain::ConsumerWard(chain) = get_use_chain(op, scope) {
        for used_by in chain {
            if let DfirOp::Agen(agen::Op::VectorStore { view, .. }) = used_by {
                store_mem_view = Some(*view);
                all_mem_views.push(*view);
            }
        }
    }

    let sized = setup_for_partitioning::<A>(
        &mut mas_data,
        &all_mem_views,
        half,
        &view,
        max_mutable,
        elem,
        num_conditionals,
        scope,
    );
    let SetupForPartitioning::Sized(Partitioning::Split(plan)) = &sized else {
        return TransferSplit::NotSetUpForPartitioning(sized);
    };
    let sizes = plan.sizes.clone();

    let mut adjustments: Vec<EvenImmutableAdjustment> = Vec::new();
    let mut hoisted: Vec<DfirOp> = Vec::new();
    let partitioned = {
        let mut create_ops = |vals: &mut Values, map: &AffineMap, start_addr_mod: i64| {
            // `adjustForEvenImmutableAddr(evaluator, mem_view_op.getStartAddress(),
            //  new_subscripts_map, ad, start_addr_mod, ad.getElementWidth());` — ⭐ BOTH
            // OUT-PARAMETERS ARE THE LEAF'S OWN LOCALS in [`fill_partitions`], reconstructed for the
            // next leaf, so copying them here is the same storage the reference writes through.
            let mut map = map.clone();
            let mut start_addr_mod = start_addr_mod;
            adjustments.push(adjust_for_even_immutable_addr::<A>(
                view.start,
                &mut map,
                &ad,
                &mut start_addr_mod,
                elem,
            ));

            // `createNewMemViewWithMod(..)`, then `setInsertionPointAfter(new_mem_view_op)`.
            let new_view = create_new_mem_view_with_mod(vals, &view, start_addr_mod);
            hoisted.push(new_view.start_address);
            let mut created = vec![new_view.mem_view];

            // `op.cloneWithNewAccessInfo(partition_builder, new_mem_view_op, new_subscripts_map,
            //  ad.getIndices())`, `setInsertionPointAfter`, then `op.cloneUseChainToNewOp(..)` —
            // which is entry 128's body exactly, so it is called rather than written twice.
            let new_mem_op = create_new_mem_op(
                vals,
                op,
                new_view.result,
                view.ty.clone(),
                indices_from_map(&map, &ad.base.indices),
                scope,
            );
            let mut chain = new_mem_op.ops;

            // "Every memory operand should have it's own unique mem view. Clone the mem views for
            // the other memory operands into the partition." — `setInsertionPoint(new_mem_op)` puts
            // them BEFORE the new load, which is why they are spliced ahead of the chain here.
            if let Some(store_view) = store_mem_view
                && let Some(store_view_op) = defining_op(store_view, scope)
            {
                for cloned in &mut chain {
                    if let DfirOp::Agen(agen::Op::VectorStore { view, .. }) = cloned {
                        let new_store_view =
                            dialects::clone_with_fresh_results(store_view_op, vals);
                        if let Some(bound) = dialects::results(&new_store_view).first().copied() {
                            *view = bound;
                            created.push(new_store_view);
                        }
                    }
                }
            }

            created.extend(chain);
            created
        };
        // `createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(), createOps);` —
        // ⚠️ `op` is the insertion anchor and has no other use.
        create_partitions(vals, &mas_data, &sizes, &subscripts_map, &mut create_ops)
    };
    let Partitions::Created { ops, partitions } = partitioned else {
        return TransferSplit::NotPartitioned(partitioned);
    };

    // `op.eraseOpAndUseChain();` — and the `LLVM_DEBUG` line below it is a diagnostic.
    TransferSplit::Split {
        ops,
        hoisted,
        erased: erase_vector_load_and_use_chain(op, scope),
        partitions,
        adjustments,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 307/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e307_transformVectorStore
///
/// **307/384** `MutableAddrSplittingPass::transformVectorStore` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375` (74L): [`transform_vector_load`]'s twin,
/// and everything down to the lambda is that function with `agen::VectorStoreOp` in place of the load.
///
/// ⛔⛔ INSIDE THE LAMBDA THE ORDER IS THE OTHER WAY ROUND, AND THE PRODUCER IS RE-POINTED BEFORE IT
/// IS CLONED. `setInsertionPoint(new_mem_op)` (`:428`) seats the builder BEFORE the new store, the
/// loop walks the **new** store's chain — which still reaches the ORIGINAL load, because
/// `cloneWithNewAccessInfo` carries `getValueToStore()` over — and assigns each load the partition's
/// own view clone (`:429-435`); only then is the chain cloned (`:436`), so the clone inherits that
/// assignment and the new store is re-pointed at it last. The load's version emits its chain AFTER
/// the new op and never touches the original.
#[must_use]
pub fn transform_vector_store<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    elem: DataType,
    correction: EarOverflowCorrection,
    num_conditionals: &mut Conditionals,
    scope: &'s [DfirOp],
) -> TransferSplit<'s> {
    // `auto op = dyn_cast<agen::VectorStoreOp>(candidate.op_); DT_CHECK(op);`
    let (Some(op), DfirOp::Agen(agen_op)) = (VectorStoreOp::of(candidate.op), candidate.op) else {
        return TransferSplit::NotAVectorStore;
    };

    // "Collect the relevant access details." — as entry 306, and a vector store has one operand too.
    let mut ad = AccessDetailsAffine::new(agen_op, candidate.comp);
    let details = ad.construct_details(candidate.mem_index, scope);
    if details != ConstructedDetails::Complete {
        return TransferSplit::DetailsNotConstructed(details);
    }
    let Some(subscripts_map) = ad.subscripts_map.clone() else {
        return TransferSplit::SubscriptsMapIsNotSet;
    };

    // `cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp())`.
    let view = match SplitCandidateView::resolve(op.view, scope) {
        SplitCandidateView::ConstantStart(view) => view,
        SplitCandidateView::NonConstantStart => return TransferSplit::MemViewStartIsNotConstant,
        SplitCandidateView::NotAMemoryView => return TransferSplit::MemRefIsNotAMemoryView,
    };

    // `int64_t max_mutable = 0; initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);`
    let initialized = initialize::<A>(&ad, &view, scope);
    let (mut mas_data, max_mutable) = match initialized {
        MasInitialization::Initialized(MasDataInit::Initialized {
            mas_data,
            max_mutable,
        }) => (mas_data, max_mutable),
        MasInitialization::Initialized(MasDataInit::NoIndices) => (Vec::new(), MutableAddr(0)),
        refusal => return TransferSplit::NotInitialized(refusal),
    };

    // `if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable, ad.getElementWidth()))
    //  return;`, and `getMaxMutableRange`'s own `DT_CHECK` with it — see entry 306 on its position.
    let Some(half) = L3Half::of(candidate.comp) else {
        return TransferSplit::CandidateIsNotOnAnL3Half(candidate.comp);
    };
    let overflow = has_mutable_addr_overflow::<A>(&mas_data, half, max_mutable, elem, correction);
    if overflow != MutableAddrOverflow::Overflow {
        return TransferSplit::AddressFits(overflow);
    }

    // "Vector load and store patterns … will be later merged into one. Collect mem views from load
    // and store patterns, if any." — ⭐ THE STORE'S CHAIN RUNS PRODUCER-WARD, so what it finds is the
    // LOAD that feeds it where entry 306 finds the store that drains it.
    let chain = vector_store_use_chain(op, scope);
    let mut all_mem_views = vec![op.view];
    let mut load_mem_view: Option<Val> = None;
    if let UseChain::ProducerWard(links) = &chain {
        for used in links {
            if let Some(load) = VectorLoadOp::of(used) {
                load_mem_view = Some(load.view);
                all_mem_views.push(load.view);
            }
        }
    }

    let sized = setup_for_partitioning::<A>(
        &mut mas_data,
        &all_mem_views,
        half,
        &view,
        max_mutable,
        elem,
        num_conditionals,
        scope,
    );
    let SetupForPartitioning::Sized(Partitioning::Split(plan)) = &sized else {
        return TransferSplit::NotSetUpForPartitioning(sized);
    };
    let sizes = plan.sizes.clone();

    let mut adjustments: Vec<EvenImmutableAdjustment> = Vec::new();
    let mut hoisted: Vec<DfirOp> = Vec::new();
    let partitioned = {
        let mut create_ops = |vals: &mut Values, map: &AffineMap, start_addr_mod: i64| {
            // `adjustForEvenImmutableAddr(..)` on this leaf's own map and modifier, as entry 306.
            let mut map = map.clone();
            let mut start_addr_mod = start_addr_mod;
            adjustments.push(adjust_for_even_immutable_addr::<A>(
                view.start,
                &mut map,
                &ad,
                &mut start_addr_mod,
                elem,
            ));

            // `createNewMemViewWithMod(..)`, then `setInsertionPointAfter(new_mem_view_op)`.
            let new_view = create_new_mem_view_with_mod(vals, &view, start_addr_mod);
            hoisted.push(new_view.start_address);
            let mut created = vec![new_view.mem_view];

            // `op.cloneWithNewAccessInfo(partition_builder, new_mem_view_op, new_subscripts_map,
            //  ad.getIndices())` — the store, still reading the ORIGINAL producer's value.
            let mut new_mem_op = clone_store_with_new_access_info(
                op,
                new_view.result,
                indices_from_map(&map, &ad.base.indices),
            );

            // "Every memory operand should have it's own unique mem view. Clone the mem views for
            // the other memory operands into the partition." — `setInsertionPoint(new_mem_op)` puts
            // them BEFORE the new store, and the loop is over the NEW store's chain, which is the
            // original's producers.
            let mut view_clones: Vec<DfirOp> = Vec::new();
            let mut partition_load_view: Option<Val> = None;
            if let UseChain::ProducerWard(links) = &chain {
                for used in links {
                    if VectorLoadOp::of(used).is_some()
                        && let Some(load_view_op) =
                            load_mem_view.and_then(|mem_view| defining_op(mem_view, scope))
                    {
                        let cloned = dialects::clone_with_fresh_results(load_view_op, vals);
                        partition_load_view = dialects::results(&cloned).first().copied();
                        view_clones.push(cloned);
                    }
                }
            }

            // `op.cloneUseChainToNewOp(partition_builder, new_mem_op);` — ⭐ AND IT INHERITS THE
            // ASSIGNMENT ABOVE, which the reference made on the original load itself (`:433`). The
            // original is erased at the end, so re-pointing the clone is the same surviving program.
            let mut cloned_chain = clone_use_chain_to_new_store(vals, &chain, &mut new_mem_op);
            if let Some(partition_view) = partition_load_view {
                for cloned in &mut cloned_chain {
                    if let DfirOp::Agen(agen::Op::VectorLoad { view, .. }) = cloned {
                        *view = partition_view;
                    }
                }
            }

            created.extend(view_clones);
            created.extend(cloned_chain);
            created.push(new_mem_op);
            created
        };
        // `createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(), createOps);`
        create_partitions(vals, &mas_data, &sizes, &subscripts_map, &mut create_ops)
    };
    let Partitions::Created { ops, partitions } = partitioned else {
        return TransferSplit::NotPartitioned(partitioned);
    };

    // `op.eraseOpAndUseChain();` — FORWARDS over a producer-ward chain, which is still consumer first.
    TransferSplit::Split {
        ops,
        hoisted,
        erased: erase_vector_store_and_use_chain(op, scope),
        partitions,
        adjustments,
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// ══════════════════════════════════════════════════════════════════════════════════════════════
// 321/384 and 322/384 — the two composite transfers, and what they share
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT THE TWO COMPOSITE SPLITS AGREE ON — everything from the access container down to the
/// overflow gate (`:456-479` and `:551-573`, the same steps in the same order).
enum CompositeOverflow<'a> {
    /// The mutable address overflows, and here is what the partition plan needs.
    Overflows {
        /// The split side's `dataflow.get_logical_memory_view`.
        view: ConstStartMemView<'a>,
        /// The iterators, including [`synthesize_time_info`]'s implicit time dimensions.
        mas_data: Vec<MasData>,
        /// `max_mutable` after both contributions.
        max_mutable: MutableAddr,
        /// Which L3 half the candidate runs on.
        half: L3Half,
    },
    /// The reference stopped, or aborted, before the gate.
    Stopped(TransferSplit<'a>),
}

/// `transformComp{,Ind}LoadAndStore`'S SHARED PROLOGUE — `MutableAddrSplitting.cpp:456-479`.
///
/// ⛔ `synthesizeTimeInfo` IS THE STEP ENTRIES 306/307 DO NOT HAVE (`:476`): a composite walks its
/// own time dimensions in hardware, and every one of them adds to `max_mutable` — without it a
/// transfer that overflows only through its time nest is never split.
/// ⛔ AND `constructTimeStepsInfo` RUNS BEFORE THE `DT_CHECK` OVER BOTH ANSWERS (`:459-465`), so it
/// runs even when `constructDetails` already failed; the first refusal is what gets reported.
fn composite_overflow<'s, A: Arch>(
    access_details: &mut AccessContainer<AccessDetailsAffineComposite<'s>>,
    candidate: &MasCandidate<'s>,
    agen_op: &'s agen::Op,
    elem: DataType,
    correction: EarOverflowCorrection,
    scope: &'s [DfirOp],
) -> CompositeOverflow<'s> {
    // `access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_)`.
    let Some(slot) = access_details.vacancy(candidate.mem_index) else {
        return CompositeOverflow::Stopped(TransferSplit::AccessSlotUnavailable(
            candidate.mem_index,
        ));
    };
    let ad = slot.emplace_insert(AccessDetailsAffineComposite::new(agen_op, candidate.comp));
    // `bool result = ad.constructDetails(candidate.mem_index_).succeeded();`
    let details = ad.construct_details(candidate.mem_index, scope);
    // "Construct the time steps info without coalescing and without burst/IL calcs." — `result &=`.
    let steps = construct_time_steps_info(access_details, false, false, scope);
    // `DT_CHECK(result);`
    if details != ConstructedDetails::Complete {
        return CompositeOverflow::Stopped(TransferSplit::DetailsNotConstructed(details));
    }
    if !steps.constructed() {
        return CompositeOverflow::Stopped(TransferSplit::TimeStepsNotConstructed(steps));
    }

    let Some(ad) = access_details.get(candidate.mem_index) else {
        return CompositeOverflow::Stopped(TransferSplit::AccessSlotUnavailable(
            candidate.mem_index,
        ));
    };
    // `dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(ad.getMemRef().getDefiningOp())` — the
    // view of the operand under consideration, which is what `initialize` splits against.
    let Some(mem_ref) = ad.affine.base.mem_ref else {
        return CompositeOverflow::Stopped(TransferSplit::MemRefIsNotAMemoryView);
    };
    let view = match SplitCandidateView::resolve(mem_ref, scope) {
        SplitCandidateView::ConstantStart(view) => view,
        SplitCandidateView::NonConstantStart => {
            return CompositeOverflow::Stopped(TransferSplit::MemViewStartIsNotConstant);
        }
        SplitCandidateView::NotAMemoryView => {
            return CompositeOverflow::Stopped(TransferSplit::MemRefIsNotAMemoryView);
        }
    };

    // `initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);`
    let initialized = initialize::<A>(&ad.affine, &view, scope);
    let (mut mas_data, mut max_mutable) = match initialized {
        MasInitialization::Initialized(MasDataInit::Initialized {
            mas_data,
            max_mutable,
        }) => (mas_data, max_mutable),
        MasInitialization::Initialized(MasDataInit::NoIndices) => (Vec::new(), MutableAddr(0)),
        refusal => return CompositeOverflow::Stopped(TransferSplit::NotInitialized(refusal)),
    };

    // `synthesizeTimeInfo(mas_data, ad, max_mutable);`
    synthesize_time_info(&mut mas_data, ad, &mut max_mutable);

    // `if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable, ad.getElementWidth()))
    //  return;`, with `getMaxMutableRange`'s own `DT_CHECK` — see entry 306 on its position.
    let Some(half) = L3Half::of(candidate.comp) else {
        return CompositeOverflow::Stopped(TransferSplit::CandidateIsNotOnAnL3Half(candidate.comp));
    };
    let overflow = has_mutable_addr_overflow::<A>(&mas_data, half, max_mutable, elem, correction);
    if overflow != MutableAddrOverflow::Overflow {
        return CompositeOverflow::Stopped(TransferSplit::AddressFits(overflow));
    }

    CompositeOverflow::Overflows {
        view,
        mas_data,
        max_mutable,
        half,
    }
}

/// THE PARTITIONS GO IN THE INNERMOST TIME LOOP — `OpBuilder cond_builder(op)` (`:1005`) seats the
/// conditional tree where `op` now is, and `createExplicitTimeLoops` has just moved `op` to the
/// bottom of the nest (`:1326-1327`). So the tree replaces the transfer inside that body.
fn splice_partitions_into_nest(nest: &mut DfirOp, partitions: Vec<DfirOp>) {
    let DfirOp::Affine(affine::Op::For { body, .. }) = nest else {
        return;
    };
    // [`nest_explicit_time_loops`] gives every loop but the innermost a body of exactly one loop.
    if matches!(body.as_slice(), [DfirOp::Affine(affine::Op::For { .. })]) {
        if let Some(inner) = body.first_mut() {
            splice_partitions_into_nest(inner, partitions);
        }
        return;
    }
    *body = partitions;
}

/// ONE CLONE OF A VIEW, OR THE VALUE ITSELF WHERE NO OP IN SCOPE BINDS IT.
///
/// *"Every memory operand should have it's own unique mem view"* (`:504-505`) —
/// `partition_builder.clone(*v.getDefiningOp())` followed by `assign`ing the clone onto the operand.
/// ⭐ THE REFERENCE ASSIGNS ONTO THE ORIGINAL, so partition 2 clones partition 1's clone; a clone
/// carries its operands over unchanged, so every partition gets the same view either way.
fn clone_mem_view_for_partition(
    vals: &mut Values,
    mem_view: Val,
    scope: &[DfirOp],
    into: &mut Vec<DfirOp>,
) -> Val {
    let Some(view_op) = defining_op(mem_view, scope) else {
        return mem_view;
    };
    let cloned = dialects::clone_with_fresh_results(view_op, vals);
    let result = dialects::results(&cloned).first().copied();
    into.push(cloned);
    result.unwrap_or(mem_view)
}

/// `agen::CompositeLoadAndStoreOp::cloneWithNewAccessInfo` — `Agen.cpp:478-514`.
///
/// ⛔ `dir` IS **ALWAYS** `PseudoRandom` (`:491-493`), whatever the original carried, and an absent
/// `dbgName` becomes the EMPTY STRING (`:485`) — `dbgName = ""` is what the vendor's key prints.
/// ⛔ AND THE BODY IS RE-BOUND: the new op's `load_iv` is fresh and every use inside the cloned
/// region is remapped onto it (`:507-511`), which an `IRMapping` clone plus one seeded pair is.
pub(super) fn clone_composite_with_new_access_info(
    vals: &mut Values,
    transfer: &agen::CompositeTransfer,
    src: Val,
    src_indices: Vec<Index>,
    dst: Val,
    dst_indices: Vec<Index>,
    time_set: IntegerSet,
) -> DfirOp {
    let load_iv = vals.mint();
    let mut mapping = ValueMapping::new();
    mapping.map(transfer.load_iv, load_iv);
    let body = vals.clone_ops(&transfer.body, &mut mapping);
    DfirOp::Agen(agen::Op::CompositeLoadAndStore(Box::new(
        agen::CompositeTransfer {
            src,
            src_indices,
            src_ty: transfer.src_ty.clone(),
            dst,
            dst_indices,
            dst_ty: transfer.dst_ty.clone(),
            load_iv,
            load_iv_ty: transfer.load_iv_ty,
            load_set: transfer.load_set.clone(),
            load_order: transfer.load_order.clone(),
            store_set: transfer.store_set.clone(),
            store_order: transfer.store_order.clone(),
            // `getTimeSymbols()` goes straight through (`Agen.cpp:487`): the new op resolves its
            // time set against the same values.
            time_symbols: transfer.time_symbols.clone(),
            time_set,
            time_order: transfer.time_order.clone(),
            load_time_addr_map: transfer.load_time_addr_map.clone(),
            store_time_addr_map: transfer.store_time_addr_map.clone(),
            dir: Some(agen::RoutingDirection::PseudoRandom),
            multicast_info: transfer.multicast_info,
            dbg_name: Some(transfer.dbg_name.clone().unwrap_or_default()),
            body,
        },
    )))
}

/// `agen::CompositeIndirectLoadAndStoreOp::cloneWithNewAccessInfo` — `Agen.cpp:1369-1435`.
///
/// ⛔ THE INDIRECT SIDES' INDICES COME FROM `this`, NOT FROM THE ARGUMENTS (`:1379-1389`): the
/// caller hands over four views and four maps, and `num_ind_src_indices`/`num_ind_dst_indices` are
/// recounted off the ORIGINAL op — so an indirect side handed a view the original did not have gets
/// that view with NO indices. `dbgName`, the fresh `load_iv` and the region are entry 321's clone's.
pub(super) fn clone_composite_indirect_with_new_access_info(
    vals: &mut Values,
    transfer: &agen::CompositeIndirectTransfer,
    indirect_src: Option<agen::IndirectAccess>,
    direct_src: Val,
    direct_src_indices: Vec<Index>,
    indirect_dst: Option<agen::IndirectAccess>,
    direct_dst: Val,
    direct_dst_indices: Vec<Index>,
    time_set: IntegerSet,
) -> DfirOp {
    let load_iv = vals.mint();
    let mut mapping = ValueMapping::new();
    mapping.map(transfer.load_iv, load_iv);
    let body = vals.clone_ops(&transfer.body, &mut mapping);
    DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(Box::new(
        agen::CompositeIndirectTransfer {
            indirect_src,
            direct_src,
            direct_src_indices,
            direct_src_ty: transfer.direct_src_ty.clone(),
            indirect_dst,
            direct_dst,
            direct_dst_indices,
            direct_dst_ty: transfer.direct_dst_ty.clone(),
            load_iv,
            load_iv_ty: transfer.load_iv_ty,
            load_set: transfer.load_set.clone(),
            load_order: transfer.load_order.clone(),
            store_set: transfer.store_set.clone(),
            store_order: transfer.store_order.clone(),
            // `getTimeSymbols()` goes straight through (`Agen.cpp:487`): the new op resolves its
            // time set against the same values.
            time_symbols: transfer.time_symbols.clone(),
            time_set,
            time_order: transfer.time_order.clone(),
            load_indirect_time_addr_map: transfer.load_indirect_time_addr_map.clone(),
            load_direct_time_addr_map: transfer.load_direct_time_addr_map.clone(),
            store_indirect_time_addr_map: transfer.store_indirect_time_addr_map.clone(),
            store_direct_time_addr_map: transfer.store_direct_time_addr_map.clone(),
            multicast_info: transfer.multicast_info,
            dbg_name: Some(transfer.dbg_name.clone().unwrap_or_default()),
            body,
        },
    )))
}

/// Replaces: e321_transformCompLoadAndStore
///
/// **321/384** `MutableAddrSplittingPass::transformCompLoadAndStore` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451` (92L): split one overflowing
/// `agen.composite_load_and_store`, making its innermost split time dimension an explicit loop first.
///
/// ⛔⛔ `createPartitions` IS HANDED `ad.getSubscriptsMap()` (`:533`) WHILE `createExplicitTimeLoops`
/// REWROTE THE **LOCAL** (`:491`) — so every partition's map descends from the ORIGINAL two-dimensional
/// subscripts while `indices` has grown the new loop's iterator. Ported as written; see
/// [`TimeLoopNest::access`] for why the vendor's own key cannot show it.
/// ⛔ AND THE UNSPLIT SIDE IS CLONED PER PARTITION, THEN READ BACK OFF `op` (`:506-513`) — the clone,
/// not the original, is what the new transfer points at.
#[must_use]
pub fn transform_comp_load_and_store<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    elem: DataType,
    correction: EarOverflowCorrection,
    num_conditionals: &mut Conditionals,
    scope: &'s [DfirOp],
) -> TransferSplit<'s> {
    // `auto op = dyn_cast<agen::CompositeLoadAndStoreOp>(candidate.op_); DT_CHECK(op);`
    let DfirOp::Agen(agen_op) = candidate.op else {
        return TransferSplit::NotACompositeLoadAndStore;
    };
    let agen::Op::CompositeLoadAndStore(transfer) = agen_op else {
        return TransferSplit::NotACompositeLoadAndStore;
    };

    let mut access_details: AccessContainer<AccessDetailsAffineComposite<'s>> =
        AccessContainer::default();
    let (view, mut mas_data, max_mutable, half) = match composite_overflow::<A>(
        &mut access_details,
        candidate,
        agen_op,
        elem,
        correction,
        scope,
    ) {
        CompositeOverflow::Overflows {
            view,
            mas_data,
            max_mutable,
            half,
        } => (view, mas_data, max_mutable, half),
        CompositeOverflow::Stopped(stopped) => return stopped,
    };
    let Some(ad) = access_details.get(candidate.mem_index) else {
        return TransferSplit::AccessSlotUnavailable(candidate.mem_index);
    };

    // `SmallVector<Value> all_mem_views = {op.getSrcMemRef(), op.getDstMemRef()};`
    let all_mem_views = vec![transfer.src, transfer.dst];

    // `setupForPartitioning(evaluator, mas_data, partition_sizes, all_mem_views, candidate.comp_,
    //                       mem_view_op, max_mutable, ad.getElementWidth());`
    let sized = setup_for_partitioning::<A>(
        &mut mas_data,
        &all_mem_views,
        half,
        &view,
        max_mutable,
        elem,
        num_conditionals,
        scope,
    );
    let SetupForPartitioning::Sized(Partitioning::Split(plan)) = &sized else {
        return TransferSplit::NotSetUpForPartitioning(sized);
    };
    let sizes = plan.sizes.clone();

    // `auto subscripts_map = ad.getSubscriptsMap(); auto indices = ad.getIndices();
    //  auto time_set = ad.getTimeSet();`
    let Some(subscripts_map) = ad.affine.subscripts_map.clone() else {
        return TransferSplit::SubscriptsMapIsNotSet;
    };
    let Some(time_set) = ad.time_set.clone() else {
        return TransferSplit::TimeSetIsNotSet;
    };
    let original = SubscriptsAndTime {
        subscripts_map: subscripts_map.clone(),
        indices: ad.affine.base.indices.clone(),
        time_set,
    };

    // `createExplicitTimeLoops(mas_data, partition_sizes, op, ad, subscripts_map, indices,
    //  time_set);` — the transfer goes in by value because it ends up inside the innermost loop.
    let loops = create_explicit_time_loops(
        vals,
        &mut mas_data,
        &sizes,
        (*candidate.op).clone(),
        ad,
        &original,
    );
    let (mut nest, access) = match loops {
        ExplicitTimeLoops::Created(created) => (Some(created.nest), created.access),
        // No split dimension is a time dimension: nothing was created and nothing was rewritten.
        ExplicitTimeLoops::NotSplitOnATimeDimension(_) => (None, original),
        refusal => return TransferSplit::TimeLoopsNotCreated(Box::new(refusal)),
    };

    let mut adjustments: Vec<EvenImmutableAdjustment> = Vec::new();
    let mut hoisted: Vec<DfirOp> = Vec::new();
    let partitioned = {
        let mut create_ops = |vals: &mut Values, map: &AffineMap, start_addr_mod: i64| {
            // `adjustForEvenImmutableAddr(..)` on this leaf's own map and modifier, as entry 306.
            let mut map = map.clone();
            let mut start_addr_mod = start_addr_mod;
            adjustments.push(adjust_for_even_immutable_addr::<A>(
                view.start,
                &mut map,
                &ad.affine,
                &mut start_addr_mod,
                elem,
            ));

            // `createNewMemViewWithMod(..)`, then `setInsertionPointAfter(new_mem_view_op)`.
            let new_view = create_new_mem_view_with_mod(vals, &view, start_addr_mod);
            hoisted.push(new_view.start_address);
            let mut created = vec![new_view.mem_view];
            let split_indices = indices_from_map(&map, &access.indices);

            if candidate.mem_index == MemoryOperandIndex::DirSrc {
                let dst = clone_mem_view_for_partition(vals, transfer.dst, scope, &mut created);
                created.push(clone_composite_with_new_access_info(
                    vals,
                    transfer,
                    new_view.result,
                    split_indices,
                    dst,
                    transfer.dst_indices.clone(),
                    access.time_set.clone(),
                ));
            } else {
                let src = clone_mem_view_for_partition(vals, transfer.src, scope, &mut created);
                created.push(clone_composite_with_new_access_info(
                    vals,
                    transfer,
                    src,
                    transfer.src_indices.clone(),
                    new_view.result,
                    split_indices,
                    access.time_set.clone(),
                ));
            }
            created
        };
        // `createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(), createOps);`
        create_partitions(vals, &mas_data, &sizes, &subscripts_map, &mut create_ops)
    };
    let Partitions::Created { ops, partitions } = partitioned else {
        return TransferSplit::NotPartitioned(partitioned);
    };

    // `op->erase();` — the transfer alone, NOT its use chain: a composite carries its own body.
    TransferSplit::Split {
        ops: match nest.take() {
            Some(mut nest) => {
                splice_partitions_into_nest(&mut nest, ops);
                vec![nest]
            }
            None => ops,
        },
        hoisted,
        erased: vec![candidate.op],
        partitions,
        adjustments,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 322/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e322_transformCompIndLoadAndStore
///
/// **322/384** `MutableAddrSplittingPass::transformCompIndLoadAndStore` —
/// `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546` (124L): entry 321 for an
/// `agen.composite_indirect_load_and_store`, whose ADDRESS views are cloned per partition too.
///
/// ⛔ THE INDIRECT MUST BE ON THE OTHER SIDE (`:577-581`): *"The immutable address must be zero for
/// the side of the transfer involving the indirect."*
/// ⛔⛔ AND THE `kDirDst` BRANCH PASSES `op.getIndirectSrcMemref()` INTO THE INDIRECT **DESTINATION**
/// SLOT (`:651`) with `getEmptyAffineMap()` for its map (`:655`) — so a gather whose direct
/// destination overflows is rebuilt claiming an indirect destination that is really its address view,
/// with no indices. Ported as written; the vendor's key only exercises the `kDirSrc` branch.
#[must_use]
pub fn transform_comp_ind_load_and_store<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    elem: DataType,
    correction: EarOverflowCorrection,
    num_conditionals: &mut Conditionals,
    scope: &'s [DfirOp],
) -> TransferSplit<'s> {
    // `auto op = dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_); DT_CHECK(op);`
    let DfirOp::Agen(agen_op) = candidate.op else {
        return TransferSplit::NotACompositeIndirectLoadAndStore;
    };
    let agen::Op::CompositeIndirectLoadAndStore(transfer) = agen_op else {
        return TransferSplit::NotACompositeIndirectLoadAndStore;
    };

    let mut access_details: AccessContainer<AccessDetailsAffineComposite<'s>> =
        AccessContainer::default();
    let (view, mut mas_data, max_mutable, half) = match composite_overflow::<A>(
        &mut access_details,
        candidate,
        agen_op,
        elem,
        correction,
        scope,
    ) {
        CompositeOverflow::Overflows {
            view,
            mas_data,
            max_mutable,
            half,
        } => (view, mas_data, max_mutable, half),
        CompositeOverflow::Stopped(stopped) => return stopped,
    };
    let Some(ad) = access_details.get(candidate.mem_index) else {
        return TransferSplit::AccessSlotUnavailable(candidate.mem_index);
    };

    // `DT_CHECK(candidate.mem_index_ == kDirSrc ? !op.hasIndirectSrc() : !op.hasIndirectDst());`
    let on_dir_src = candidate.mem_index == MemoryOperandIndex::DirSrc;
    let indirect_on_split_side = if on_dir_src {
        transfer.indirect_src.is_some()
    } else {
        transfer.indirect_dst.is_some()
    };
    if indirect_on_split_side {
        return TransferSplit::IndirectIsOnTheSplitSide(candidate.mem_index);
    }

    // `all_mem_views = {op.getDirectSrcMemref(), op.getDirectDstMemref()}`, plus each indirect side
    // that is present — an address view is split against the same ranges the data views are.
    let mut all_mem_views = vec![transfer.direct_src, transfer.direct_dst];
    if let Some(indirect) = &transfer.indirect_src {
        all_mem_views.push(indirect.view);
    }
    if let Some(indirect) = &transfer.indirect_dst {
        all_mem_views.push(indirect.view);
    }

    let sized = setup_for_partitioning::<A>(
        &mut mas_data,
        &all_mem_views,
        half,
        &view,
        max_mutable,
        elem,
        num_conditionals,
        scope,
    );
    let SetupForPartitioning::Sized(Partitioning::Split(plan)) = &sized else {
        return TransferSplit::NotSetUpForPartitioning(sized);
    };
    let sizes = plan.sizes.clone();

    let Some(subscripts_map) = ad.affine.subscripts_map.clone() else {
        return TransferSplit::SubscriptsMapIsNotSet;
    };
    let Some(time_set) = ad.time_set.clone() else {
        return TransferSplit::TimeSetIsNotSet;
    };
    let original = SubscriptsAndTime {
        subscripts_map: subscripts_map.clone(),
        indices: ad.affine.base.indices.clone(),
        time_set,
    };

    let loops = create_explicit_time_loops(
        vals,
        &mut mas_data,
        &sizes,
        (*candidate.op).clone(),
        ad,
        &original,
    );
    let (mut nest, access) = match loops {
        ExplicitTimeLoops::Created(created) => (Some(created.nest), created.access),
        ExplicitTimeLoops::NotSplitOnATimeDimension(_) => (None, original),
        refusal => return TransferSplit::TimeLoopsNotCreated(Box::new(refusal)),
    };

    let mut adjustments: Vec<EvenImmutableAdjustment> = Vec::new();
    let mut hoisted: Vec<DfirOp> = Vec::new();
    let partitioned = {
        let mut create_ops = |vals: &mut Values, map: &AffineMap, start_addr_mod: i64| {
            let mut map = map.clone();
            let mut start_addr_mod = start_addr_mod;
            adjustments.push(adjust_for_even_immutable_addr::<A>(
                view.start,
                &mut map,
                &ad.affine,
                &mut start_addr_mod,
                elem,
            ));

            let new_view = create_new_mem_view_with_mod(vals, &view, start_addr_mod);
            hoisted.push(new_view.start_address);
            let mut created = vec![new_view.mem_view];
            let split_indices = indices_from_map(&map, &access.indices);

            if on_dir_src {
                // The direct destination, then the indirect one where the transfer scatters.
                let direct_dst =
                    clone_mem_view_for_partition(vals, transfer.direct_dst, scope, &mut created);
                let indirect_dst =
                    transfer
                        .indirect_dst
                        .as_ref()
                        .map(|indirect| agen::IndirectAccess {
                            view: clone_mem_view_for_partition(
                                vals,
                                indirect.view,
                                scope,
                                &mut created,
                            ),
                            indices: indirect.indices.clone(),
                            ty: indirect.ty.clone(),
                        });
                created.push(clone_composite_indirect_with_new_access_info(
                    vals,
                    transfer,
                    // `op.getIndirectSrcMemref()` with `getEmptyAffineMap()` — null, by the
                    // `DT_CHECK` above.
                    None,
                    new_view.result,
                    split_indices,
                    indirect_dst,
                    direct_dst,
                    transfer.direct_dst_indices.clone(),
                    access.time_set.clone(),
                ));
            } else {
                let direct_src =
                    clone_mem_view_for_partition(vals, transfer.direct_src, scope, &mut created);
                let indirect_src =
                    transfer
                        .indirect_src
                        .as_ref()
                        .map(|indirect| agen::IndirectAccess {
                            view: clone_mem_view_for_partition(
                                vals,
                                indirect.view,
                                scope,
                                &mut created,
                            ),
                            indices: indirect.indices.clone(),
                            ty: indirect.ty.clone(),
                        });
                // ⛔ THE INDIRECT **SOURCE** VIEW IN THE INDIRECT **DESTINATION** SLOT (`:651`),
                // with an empty map and — `num_ind_dst_indices` being recounted off the original —
                // no indices. See this function's banner.
                let indirect_dst = indirect_src.as_ref().map(|indirect| agen::IndirectAccess {
                    view: indirect.view,
                    indices: Vec::new(),
                    ty: indirect.ty.clone(),
                });
                created.push(clone_composite_indirect_with_new_access_info(
                    vals,
                    transfer,
                    indirect_src,
                    direct_src,
                    transfer.direct_src_indices.clone(),
                    indirect_dst,
                    new_view.result,
                    split_indices,
                    access.time_set.clone(),
                ));
            }
            created
        };
        create_partitions(vals, &mas_data, &sizes, &subscripts_map, &mut create_ops)
    };
    let Partitions::Created { ops, partitions } = partitioned else {
        return TransferSplit::NotPartitioned(partitioned);
    };

    // `op->erase();`
    TransferSplit::Split {
        ops: match nest.take() {
            Some(mut nest) => {
                splice_partitions_into_nest(&mut nest, ops);
                vec![nest]
            }
            None => ops,
        },
        hoisted,
        erased: vec![candidate.op],
        partitions,
        adjustments,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 351/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A [`DataType`] AS WIDE AS ONE ELEMENT OF THIS TYPE — the `getElementWidth()`-to-[`DataType`]
/// bridge the four transforms' `elem` argument leaves to their caller.
///
/// ⛔ CHOSEN BY WIDTH, NOT BY IDENTITY. [`AddrRange::elements`] reads `elem.bits()` and nothing else,
/// so two formats of one width are interchangeable here. [`None`] is an element type NO format is as
/// wide as: `i1`, whose [`DataType::Bool`] is eight bits, and the PT's `i24`.
fn transfer_format(elem: ElemType) -> Option<DataType> {
    match elem {
        // ⛔ `Senint24` IS THE 16-BIT ONE — the format [`ElemType::of`] maps to `Int(16)` off the PT.
        ElemType::Int(16) => Some(DataType::Senint24),
        ElemType::Int(8) | ElemType::MxInt(8) => Some(DataType::Senint8),
        ElemType::Int(4) | ElemType::MxInt(4) => Some(DataType::Senint4),
        ElemType::Int(32) => Some(DataType::Senuint32),
        ElemType::F16 => Some(DataType::Sen169Fp16),
        ElemType::Bf16 => Some(DataType::Bfloat16),
        ElemType::F32 => Some(DataType::IeeeFp32),
        ElemType::F8E4M3Fn => Some(DataType::Sen143Fp8),
        ElemType::F8E8M0Fnu | ElemType::MxFloat(8) => Some(DataType::Sen080Fp8),
        // ⭐ NO FORMAT NAMES E5M2, and `Sen053_FP8` is the one the reference itself stands in for a
        // missing exponent layout with; both are eight bits, which is all this is read for.
        ElemType::F8E5M2 => Some(DataType::Sen053Fp8),
        ElemType::F4E2M1Fn | ElemType::MxFloat(4) => Some(DataType::Sen121Fp4),
        ElemType::Int(_) | ElemType::MxFloat(_) | ElemType::MxInt(_) => None,
    }
}

/// `isCandidateMemView` (`MutableAddrSplitting.cpp:229-237`) — AN HBM VIEW WHOSE START ADDRESS IS
/// NOT A SYMBOL.
///
/// ⭐ THE SECOND TEST IS WHY THE PASS AND `MutableStartAddrShifting` DO NOT FIGHT: a start address
/// bound by `symbol.create_symbol` or `symbol.query_map` is one the schedule fixes later, and there
/// is no constant to add a partition offset to ([`is_eligible_for_splitting`]).
pub(super) fn is_candidate_mem_view(
    from: Val,
    start: Val,
    body: &[DfirOp],
    preamble: &[DfirOp],
) -> bool {
    // `dcc::getUnitType(mem_view_op.getFromUnit().getDefiningOp()) != SenComponents::HBM`.
    // ⚠️ THE `dataflow.get_unit` OPS SIT AT FUNCTION SCOPE while the view sits in a unit's body
    // (`tests/sentient_corpus/group_0__g0_0_mul.dfir.mlir:3-12`), so the def walk spans both — which
    // is one search for `getDefiningOp()`, whose reach is the whole module.
    let from_unit = defining_op(from, preamble).or_else(|| defining_op(from, body));
    if !matches!(
        from_unit,
        Some(DfirOp::Dataflow(dataflow::Op::GetUnit {
            unit: DfirUnit::Hbm,
            ..
        }))
    ) {
        return false;
    }
    // `!isa_and_nonnull<symbol::CreateSymbolOp, symbol::SymbolQueryMapOp>(getStartAddress()
    //  .getDefiningOp())` — ⛔ `_and_nonnull`: a start address no op in scope binds is NOT one of the
    // two, so a region argument stays a candidate and fails later, at the eligibility check.
    !matches!(
        defining_op(start, body).or_else(|| defining_op(start, preamble)),
        Some(DfirOp::Symbol(
            symbol::Op::CreateSymbol { .. } | symbol::Op::QueryMap { .. }
        ))
    )
}

/// `WalkOrder::PreOrder` over one unit's body, keeping the `dataflow.get_logical_memory_view` ops.
///
/// ⛔ THROUGH EVERY REGION. A view bound inside an `scf.for` is one the reference's `unit.walk`
/// reaches, and [`dialects::regions`] is the one table that knows which ops have one.
pub(super) fn mem_views_in_pre_order<'a>(body: &'a [DfirOp], found: &mut Vec<&'a DfirOp>) {
    for op in body {
        if matches!(
            op,
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { .. })
        ) {
            found.push(op);
        }
        for region in dialects::regions(op) {
            mem_views_in_pre_order(region, found);
        }
    }
}

/// WHICH OF THE FOUR TRANSFORMS RAN ON ONE CANDIDATE, AND WHAT IT ANSWERED — `:284-294`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MasDispatch<'a> {
    /// `if (isa<agen::VectorLoadOp>(candidate.op_)) transformVectorLoad(candidate);`
    VectorLoad(TransferSplit<'a>),
    /// `else if (isa<agen::VectorStoreOp>(..)) transformVectorStore(candidate);`
    VectorStore(TransferSplit<'a>),
    /// `else if (isa<agen::CompositeLoadAndStoreOp>(..)) transformCompLoadAndStore(candidate);`
    CompLoadAndStore(TransferSplit<'a>),
    /// `else if (isa<agen::CompositeIndirectLoadAndStoreOp>(..))
    /// transformCompIndLoadAndStore(candidate);`
    CompIndLoadAndStore(TransferSplit<'a>),
    /// `else llvm_unreachable("Unexpected candidate operation.")` — the one user of an HBM view is
    /// none of the four transfers.
    UnexpectedCandidateOperation,
    /// No [`DataType`] is as wide as the transfer's elements — see [`transfer_format`].
    ElementFormatIsUnnamed(ElemType),
}

/// ONE ENTRY OF `candidates`, AND WHAT THE DISPATCH LOOP DID WITH IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MasSplit<'a> {
    /// `candidates.emplace_back(mem_op, unit, comp, mem_index)` (`:278`).
    pub candidate: MasCandidate<'a>,
    /// The transform's own answer.
    pub dispatch: MasDispatch<'a>,
}

/// WHAT THE PASS DID TO A PROGRAM — or the `DT_CHECK` in the collecting walk that stopped it.
///
/// ⛔ THE THREE REFUSALS END THE WHOLE PASS, WHICH IS WHY THEY ARE NOT PER-CANDIDATE. All three are
/// `DT_CHECK`s inside the module walk (`:255`, `:272`, `:276`), and the walk finishes before the
/// first transform runs — so an abort there means NOTHING was transformed, in any unit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum MutableAddrSplitting<'a> {
    /// One entry per candidate, in the order the walk collected them.
    Split(Vec<MasSplit<'a>>),
    /// `DT_CHECK(mem_view->hasOneUse())` — this HBM view is read by zero users or by more than one.
    MemViewIsNotSingleUsed(Val),
    /// `DT_CHECK_MSG(mem_index != kMax, "Invalid HBM memory operand.")` — an indirect transfer whose
    /// HBM view is neither of its two DIRECT memory operands.
    InvalidHbmMemoryOperand(Val),
    /// `DT_CHECK_MSG(res.second, "Data transfers should only have one HBM memory operand.")` — a
    /// second HBM view on a transfer already collected through the first.
    SecondHbmMemoryOperand(Val),
}

/// Replaces: e351_runOnOperation
///
/// **351/384** `MutableAddrSplittingPass::runOnOperation` — `MutableAddrSplitting.cpp:226` (70L):
/// collect every HBM view in an L3 unit, then run one transform per memory operation found through
/// one.
///
/// ⛔⛔ `num_conditionals_ = 0` PER UNIT (`:246`) CANNOT DO WHAT ITS COMMENT SAYS. The resets are in
/// the COLLECTING walk and the counter's only writer is `calculatePartitionSizes`, reached from the
/// dispatch loop AFTER that walk — so every unit's conditionals go on one total starting at zero.
pub fn run_on_operation<'p, A: Arch>(
    program: &'p Program<A>,
    vals: &mut Values,
    correction: EarOverflowCorrection,
) -> MutableAddrSplitting<'p> {
    // ⛔ `if (DisableThisPass) return;` (`:227`) IS DROPPED, as at entry 305:
    // `dcc-mutable-addr-splitting-disable` (`:51-54`) is a `dcc-opt` command-line flag, and which
    // passes run is a call in this crate.
    //
    // `std::vector<MASCandidate> candidates;` — ⭐ EACH PAIRED WITH ITS UNIT'S BODY, which is
    // `candidate.unit_` (the transforms resolve their operands against it), and with the element
    // format the transform's own `ad.getElementWidth()` will be.
    let mut candidates: Vec<(MasCandidate<'p>, &'p [DfirOp], ElemType)> = Vec::new();
    // `std::unordered_set<Operation *> analyzed_candidates;` — ⚠️ BY IDENTITY, so a program with two
    // structurally equal transfers is two entries. That is what `Operation *` keys on.
    let mut analyzed: Vec<&'p DfirOp> = Vec::new();

    // `module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit) { .. })` — the units are
    // a field of the program, so the walk is an iteration (entries 178 and 288 take the same shape).
    for unit in program.units.iter() {
        // `auto comp = dcc::getUnitType(unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());`
        // then `if (!is_any_of(comp, L3LU, L3SU)) return WalkResult::advance();` — [`L3Half::of`] is
        // that membership test, and [`dfir::Units`] carries the kind so there is nothing to resolve.
        let comp = unit.on.kind();
        if L3Half::of(comp).is_none() {
            continue;
        }
        let body = unit.body.as_slice();
        let mut views: Vec<&'p DfirOp> = Vec::new();
        mem_views_in_pre_order(body, &mut views);
        for view in views {
            let DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                ty: view_ty,
                ..
            }) = view
            else {
                continue;
            };
            // `if (!mem_view || !isCandidateMemView(mem_view)) return WalkResult::advance();`
            if !is_candidate_mem_view(*from, *start, body, &program.preamble) {
                continue;
            }
            // `DT_CHECK(mem_view->hasOneUse());` then
            // `auto mem_op = *mem_view->getUsers().begin();` — [`uses`] gives one entry per USE, so a
            // view read twice by one transfer is not single-used either.
            let users = uses(*result, body);
            let [mem_op] = users.as_slice() else {
                return MutableAddrSplitting::MemViewIsNotSingleUsed(*result);
            };
            let mem_op = *mem_op;

            // `agen::MemoryOperandIndex mem_index = agen::MemoryOperandIndex::kMax;`
            let mem_index = match mem_op {
                // ⛔ A COMPOSITE'S SOURCE IS COMPARED, AND EVERYTHING ELSE IS THE DESTINATION:
                // `comp_las.getSrcMemRef() == mem_view.getResult() ? kDirSrc : kDirDst`.
                DfirOp::Agen(agen::Op::CompositeLoadAndStore(transfer)) => {
                    Some(if transfer.src == *result {
                        MemoryOperandIndex::DirSrc
                    } else {
                        MemoryOperandIndex::DirDst
                    })
                }
                // ⛔⛔ TWO `if`s AND NO `else` (`:265-268`) — an indirect transfer whose HBM view is
                // one of its INDIRECT operands leaves `mem_index` at `kMax` and hits the check below.
                DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(transfer)) => {
                    if transfer.direct_src == *result {
                        Some(MemoryOperandIndex::DirSrc)
                    } else if transfer.direct_dst == *result {
                        Some(MemoryOperandIndex::DirDst)
                    } else {
                        None
                    }
                }
                // `} else { mem_index = agen::MemoryOperandIndex::kDirSrc; }` — a vector load reads
                // its one memref as the source, and so does a vector STORE's, which is the reference's
                // own answer for it.
                _ => Some(MemoryOperandIndex::DirSrc),
            };
            // `DT_CHECK_MSG(mem_index != kMax, "Invalid HBM memory operand.");`
            let Some(mem_index) = mem_index else {
                return MutableAddrSplitting::InvalidHbmMemoryOperand(*result);
            };
            // `auto res = analyzed_candidates.insert(mem_op);`
            // `DT_CHECK_MSG(res.second, "Data transfers should only have one HBM memory operand.");`
            if analyzed.iter().any(|seen| core::ptr::eq(*seen, mem_op)) {
                return MutableAddrSplitting::SecondHbmMemoryOperand(*result);
            }
            analyzed.push(mem_op);
            // `ad.getElementWidth()`, which `AccessDetailsAffine::initialize` takes from the VECTOR
            // type of a vector transfer and from the memory operand's VIEW type for a composite — and
            // this view IS that operand, so its own `memref` is the composite's answer.
            let elem = match mem_op {
                DfirOp::Agen(
                    agen::Op::VectorLoad { ty, .. } | agen::Op::VectorStore { ty, .. },
                ) => ty.elem,
                _ => view_ty.elem,
            };
            candidates.push((
                MasCandidate {
                    op: mem_op,
                    comp,
                    mem_index,
                },
                body,
                elem,
            ));
        }
    }

    // `for (auto &candidate : candidates)` — ⭐ A SECOND LOOP, and that is what makes the per-unit
    // reset above inert.
    let mut num_conditionals = Conditionals::default();
    let mut split: Vec<MasSplit<'p>> = Vec::new();
    for (candidate, scope, elem) in candidates {
        let dispatch = match transfer_format(elem) {
            None => MasDispatch::ElementFormatIsUnnamed(elem),
            Some(elem) => match candidate.op {
                DfirOp::Agen(agen::Op::VectorLoad { .. }) => {
                    MasDispatch::VectorLoad(transform_vector_load::<A>(
                        vals,
                        &candidate,
                        elem,
                        correction,
                        &mut num_conditionals,
                        scope,
                    ))
                }
                DfirOp::Agen(agen::Op::VectorStore { .. }) => {
                    MasDispatch::VectorStore(transform_vector_store::<A>(
                        vals,
                        &candidate,
                        elem,
                        correction,
                        &mut num_conditionals,
                        scope,
                    ))
                }
                DfirOp::Agen(agen::Op::CompositeLoadAndStore(_)) => {
                    MasDispatch::CompLoadAndStore(transform_comp_load_and_store::<A>(
                        vals,
                        &candidate,
                        elem,
                        correction,
                        &mut num_conditionals,
                        scope,
                    ))
                }
                DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(_)) => {
                    MasDispatch::CompIndLoadAndStore(transform_comp_ind_load_and_store::<A>(
                        vals,
                        &candidate,
                        elem,
                        correction,
                        &mut num_conditionals,
                        scope,
                    ))
                }
                _ => MasDispatch::UnexpectedCandidateOperation,
            },
        };
        split.push(MasSplit {
            candidate,
            dispatch,
        });
    }
    MutableAddrSplitting::Split(split)
}
