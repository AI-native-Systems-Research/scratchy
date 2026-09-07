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

use crate::arch::{Arch, Bounded, Elements, Sticks};
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, arith, dataflow, defining_op};
use crate::islands::dataflow_ir::ty::{AffineMap, MemRef};
use crate::units::DfirUnit;


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
            | DfirUnit::CrossPtnLink => None,
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
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::dataflow_ir::ty::{ElemType, ScalarTy};
    use crate::units::Row;

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
        assert_eq!(mutable.elements(DataType::Sen169Fp16), Elements(134_217_728));
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
}
