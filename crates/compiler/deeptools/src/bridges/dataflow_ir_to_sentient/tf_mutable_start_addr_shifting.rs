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

//! `MutableStartAddrShifting.cpp` — 13 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e116_getMaxImmutableRange` | 116/384 | 8 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355` |
//! | `e191_calculateFullShift` | 191/384 | 25 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462` |
//! | `e192_calculateDimWeights` | 192/384 | 26 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560` |
//! | `e254_offsetShifts` | 254/384 | 22 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590` |
//! | `e255_applyShifts` | 255/384 | 27 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616` |
//! | `e291_calculatePartialShift` | 291/384 | 66 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490` |
//! | `e308_calculateShifts` | 308/384 | 61 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396` |
//! | `e323_shiftMutableAddr` | 323/384 | 19 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366` |
//! | `e352_transformVectorLoad` | 352/384 | 25 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201` |
//! | `e353_transformVectorStore` | 353/384 | 24 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229` |
//! | `e354_transformCompLoadAndStore` | 354/384 | 39 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256` |
//! | `e355_transformCompIndLoadAndStore` | 355/384 | 54 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298` |
//! | `e372_runOnOperation` | 372/384 | 69 | `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130` |

use super::agen_access_details::{
    AccessDetailsAffine, AccessDetailsAffineComposite, ConstructedDetails, LayoutCoeff,
    MemoryOperandIndex,
};
use super::tf_mutable_addr_splitting::{
    AddrRange, ConstStartMemView, L3Half, MasCandidate, SplitCandidateView,
    clone_composite_indirect_with_new_access_info, clone_composite_with_new_access_info,
    is_candidate_mem_view, mem_views_in_pre_order,
};
use super::tf_transform_paged_mem_view_impl::{
    VectorLoadOp, VectorStoreOp, clone_load_with_new_access_info, clone_store_with_new_access_info,
    indices_from_map,
};
use crate::arch::{Arch, Elements, IsaGen};
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, agen, arith, dataflow, uses};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap};
use crate::islands::dataflow_ir::{Program, Values};
use crate::units::DfirUnit;
use core::cmp::Reverse;
use core::num::{NonZeroU32, NonZeroU64};

/// `-dcc-mutable-start-addr-shifting-max-immutable-size`, `cl::init(-1)` —
/// `MutableStartAddrShifting.cpp:47-52`:
///
/// ```cpp
/// static llvm::cl::opt<int64_t> MaxImmutableSize(
///     "dcc-mutable-start-addr-shifting-max-immutable-size",
///     llvm::cl::desc(
///         "Set a maximum value Mutable Start Address Shifting should use for "
///         "external memory immutable addresses. Measured in bits."),
///     llvm::cl::init(-1));
/// ```
///
/// ⛔⛔ ITS OWN FLAG, NOT THE SPLITTING PASS'S. `-dcc-mutable-addr-splitting-max-immutable-size`
/// ([`super::tf_mutable_addr_splitting::MAX_IMMUTABLE_SIZE`]) is a DIFFERENT `cl::opt` in a different
/// translation unit, and that is the entire difference between [`max_immutable_range`] and
/// [`super::tf_mutable_addr_splitting::max_immutable_range`] — the two bodies are otherwise identical,
/// down to the register. Sharing one constant between them would make an override meant for one pass
/// silently move the other pass's addresses.
///
/// ⭐ `None` IS `< 0`, and it is a const because nothing in this crate parses `dcc-opt`'s command
/// line — see [`super::tf_mutable_addr_splitting::MAX_MUTABLE_SIZE`].
pub const MAX_IMMUTABLE_SIZE: Option<AddrRange> = None;

/// Replaces: e116_getMaxImmutableRange
///
/// **116/384** `MutableStartAddrShiftingPass::getMaxImmutableRange` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355` (8L).
///
/// ```cpp
/// int64_t MutableStartAddrShiftingPass::getMaxImmutableRange(
///     SenComponents comp) const {
///   DT_CHECK(is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU));
///   auto &sys_def = dcc_ext_ctx_.dsc_global_->sysDef;
///   return MaxImmutableSize < 0
///              ? pow(2,
///                    sys_def.regInfoPerUnit.at(comp).at(RegType::EBR).bitSize) *
///                    sys_def.bytesPerStick * 8
///              : MaxImmutableSize;
/// }
/// ```
///
/// # ⛔⛔ A SECOND FUNCTION FOR A SECOND FLAG, NOT A DUPLICATE
///
/// Character for character this is `MutableAddrSplittingPass::getMaxImmutableRange`
/// (`MutableAddrSplitting.cpp:683`, entry 112) over the same `EBR` — except that the `MaxImmutableSize`
/// it reads is THIS pass's `cl::opt` ([`MAX_IMMUTABLE_SIZE`]). Two passes, two overrides, one
/// register: which is why the arithmetic lives in [`AddrRange::of_register`] and is not written twice,
/// while the entry point is.
///
/// # ⭐ WHY THE PRODUCT IS `2^bitSize * bytesPerStick * 8`
///
/// The EBR holds a transfer's immutable base as a count of GRANULES: a `bitSize`-wide unsigned
/// register names `2^bitSize` of them, each granule is one stick, a stick is `bytesPerStick` bytes,
/// and a byte is 8 bits. So the product is the addressable span in bits — the unit the flag documents
/// and the unit `ad.getElementWidth()` divides. `bytesPerStick` is 128 on both arches
/// (`sysdef.cpp:206`), so the span is `2^30 * 1024` bits on RCUDD1A and `2^32 * 1024` on SEN1P5 —
/// 128 GiB and 512 GiB of external memory.
///
/// # ⛔ THE PORT DOES NOT USE `A::EBR_GRANULARITY`, AND THE REFERENCE DOES NOT EITHER
///
/// SEN1P5 addresses the EBR in TWO-stick granules (`ebrGranurality = 2`, `sysdef.cpp:236`), so the
/// span this reports is arguably half of what that register can reach there. The reference computes
/// `bytesPerStick` flat, so this port does too: the shift budget it feeds (`immutable_space` at
/// `:433-437`) is a bound, and reporting the smaller bound shifts less, not wrongly. A change here is
/// a change to the reference.
///
/// # ⚠️ THE COMPONENT IS TAKEN AND NOT READ
///
/// `regInfoPerUnit.at(comp).at(RegType::EBR).bitSize` is a lookup whose two rows are identical on both
/// arches (`sysdef.cpp:313-360`, and see [`Arch::L3_EBR_BITS`]). The parameter stays because the
/// [`L3Half`] the caller must produce IS the `DT_CHECK`: dropping it would delete the precondition,
/// not simplify it.
///
/// # Arguments
///
/// * `_comp` — which L3 half is asking. See [`L3Half`].
#[must_use]
pub fn max_immutable_range<A: Arch>(_comp: L3Half) -> AddrRange {
    match MAX_IMMUTABLE_SIZE {
        // `: MaxImmutableSize` — taken as given, already in bits.
        Some(given) => given,
        // `pow(2, ..bitSize) * sys_def.bytesPerStick * 8`.
        None => AddrRange::of_register::<A>(A::L3_EBR_BITS),
    }
}

/// HOW MANY ELEMENTS FILL ONE STICK — `dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth()`,
/// the unit every shift this pass computes has to be a whole number of.
///
/// The pass asks for it FOUR times over — `MutableStartAddrShifting.cpp:373` (the precondition
/// `shiftMutableAddr` hands `hasValidL3ImmutableAddr`), `:405` (`calculateShifts`), `:485` (this
/// function) and `:553` (`calculatePartialShift`) — always from the same two facts: a stick is
/// [`Arch::BYTES_PER_STICK`] bytes, a byte is 8 bits, and an element is `ad.getElementWidth()` bits.
/// [`super::agen_helper`] computes the same quotient inline for a load type's full-stick test
/// (`agen_helper.rs:930`); here it is a type because the shift arithmetic's one invariant is stated in
/// terms of it.
///
/// ⛔ `NonZeroU64`, SO THE `%` CANNOT BE A DIVISION BY ZERO. `AccessDetailsBase`'s element width
/// starts at `Bits(0)` — the unset state, C++'s `element_width_` before `setElementWidth` runs — and
/// dividing by it is the reference's own undefined behaviour, not an answer worth reproducing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElementsPerStick(NonZeroU64);

impl ElementsPerStick {
    /// THE QUOTIENT FOR ONE ELEMENT WIDTH, or `None` when there is no whole element in a stick.
    ///
    /// ⚠️ `None` IS UNREACHABLE ON BOTH ARCHES AND IS STILL NOT AN `expect`. A stick is 1024 bits
    /// (`bytesPerStick = 128`, `sysdef.cpp:206`) and the widest format this crate has is 32 bits
    /// (`IeeeFp32`/`Senuint32`, `formats.rs:38`), so the quotient is at least 32 — but it is the TYPE
    /// that has to say so, and an element wider than a stick is a fact about a future format rather
    /// than about this function.
    #[must_use]
    pub fn of<A: Arch>(element_width: NonZeroU32) -> Option<ElementsPerStick> {
        NonZeroU64::new(A::BYTES_PER_STICK.get() * 8 / u64::from(element_width.get()))
            .map(ElementsPerStick)
    }

    /// The quotient itself.
    #[must_use]
    pub const fn elements(self) -> Elements {
        Elements(self.0.get())
    }

    /// `DT_CHECK(total_shift % num_elems_in_stick == 0)` (`:486`) — AS THE TOTAL'S TYPE.
    ///
    /// ⛔ THE CHECK IS NOT DROPPED AND IT IS NOT AN `assert!`. This crate never runtime-refuses, so
    /// the question "does this shift land on a stick boundary" is answered where the total is BUILT and
    /// travels with it; a caller that wants the reference's behaviour matches on
    /// [`TotalShift::PartialStick`] and can say what it is going to do about it.
    ///
    /// ⭐ `unsigned_abs`, BECAUSE A SHIFT CAN BE NEGATIVE. `offsetShifts(shifts, ad, -num_elems_in_stick)`
    /// (`:427`, `:457`) subtracts a whole stick, and Rust's `%` keeps the sign of the dividend — which
    /// `== 0` does not care about, but a reader does.
    #[must_use]
    pub const fn total(self, elements: i64) -> TotalShift {
        if elements.unsigned_abs().is_multiple_of(self.0.get()) {
            TotalShift::WholeSticks(elements)
        } else {
            TotalShift::PartialStick(elements)
        }
    }
}

/// ONE SUBSCRIPT'S CONSTANT OFFSET — how much of that subscript can move out of the mutable start
/// address and into the immutable one.
///
/// ⛔ IN THE DIMENSION'S OWN INDEX UNITS, NOT IN ELEMENTS. `applyShifts` subtracts it from the
/// subscripts map result itself — `new_exprs.emplace_back(res - shift)` (`:628`) — so it is an index,
/// and multiplying it by that dimension's [`LayoutCoeff`] is what turns it into the count of elements
/// the start address moves by. Typing both as [`Elements`] would let the two be added.
///
/// ⭐ THE SIGN IS THE DIRECTION, and the reference documents it on `calculateShifts`
/// (`:390-395`): *"A positive value indicates the shift should be from the mutable start address to
/// the immutable address. A negative value indicates the shift should be from the immutable address to
/// the mutable address."*
///
/// ⚠️ `int64_t`, WHERE THE REFERENCE NARROWS TO `int` AND BACK. `int constant_offset = ..coeffs.back()`
/// (`:478`) truncates a 64-bit constant column to 32 bits and then pushes it into a
/// `SmallVectorImpl<int64_t>`; an offset above `2^31` wraps there and does not here. No subscript in
/// the corpus comes near it, and reproducing the truncation would mean reproducing a defect rather
/// than a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shift(pub i64);

/// WHICH RESULT OF THE SUBSCRIPTS MAP a shift belongs to — `DimWeight::dim_` (`:84`), and the position
/// `shifts` is indexed by.
///
/// ⛔⛔ THIS IS A POSITION IN **TRANSFER ORDER**, AND THE REFERENCE'S CONSUMERS TREAT IT AS A POSITION
/// IN THE ORIGINAL MAP. Both [`calculate_full_shift`] and [`calculate_dim_weights`] walk
/// `transfer_order.compose(subscripts_map)`, so index `i` is the subscript the transfer visits `i`th;
/// but `applyShifts` zips the same list against `subscripts_map.getResults()` in ORIGINAL order
/// (`:627-628`), and both functions pair it with `layout_coeffs[i]`, which is the ORIGINAL dimension's
/// stride. The three agree exactly when the transfer order is the identity, which is what
/// `load_order`/`store_order` is for every access this crate emits (see [`AffineMap::identity`]).
///
/// ⛔ AND THE REFERENCE STATES THE ASSUMPTION AS A FACT: *"The shifts were calculated with respect to
/// transfer order already, so the shifts can be directly applied to the results"* (`:621-622`) — which
/// is true of the LIST's length and false of its order.
///
/// ⚠️ AND THE REFERENCE KNEW: `calculatePartialShift` builds a `dim_order` out of the transfer order's
/// positions (`:511-517`) — the translation that would fix it — and then **never reads it**. Checked
/// against the whole translation unit: `dim_order` is DECLARED at `:511`, WRITTEN at `:516` and read
/// nowhere; those are the only two lines of the file that mention it. A
/// newtype cannot fix that alone, but it can stop the two spaces being spelled the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubscriptResult(pub u32);

impl SubscriptResult {
    /// The position as a slice index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// HOW MANY ELEMENTS A SET OF SHIFTS MOVES THE IMMUTABLE START ADDRESS BY — `total_shift`, with the
/// pass's one invariant attached.
///
/// ⛔ IN ELEMENTS, because it is `Σ shift[i] * layout_coeffs[i]`: the layout coefficients linearise a
/// dimension's index into the view's element space (`Dataflow.td:250`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TotalShift {
    /// A whole number of sticks — *"Ensure shifts are completed in terms of sticks. The mutable
    /// address start should already be in terms of sticks."* (`:482-483`).
    WholeSticks(i64),
    /// NOT a whole number of sticks: the input the reference's `DT_CHECK` (`:486`) rejects.
    ///
    /// ⛔ A VARIANT AND NOT A PANIC. It is reachable — a subscript whose constant offset times its
    /// stride is not stick-aligned — and `calculatePartialShift` handles exactly that case by shifting
    /// the remainder back (`:554-556`), so the state is one the pass has a policy for.
    PartialStick(i64),
}

impl TotalShift {
    /// The count itself, whichever variant it is.
    #[must_use]
    pub const fn elements(self) -> i64 {
        match self {
            TotalShift::WholeSticks(elements) | TotalShift::PartialStick(elements) => elements,
        }
    }
}

/// WHAT `calculateFullShift` HANDS BACK — the out-parameter and the return value, together.
///
/// ⭐ ONE VALUE, BECAUSE THE TWO ARE ONE ANSWER. The reference fills a
/// `SmallVectorImpl<int64_t> &shifts` and returns the total; `DT_CHECK(shifts.empty())` on entry
/// (`:464`) is the whole of what an out-parameter needs saying about it, and a returned `Vec` cannot
/// arrive non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullShift {
    /// One per result of the ordered subscripts map — see [`SubscriptResult`].
    pub shifts: Vec<Shift>,
    /// What those shifts add up to, in elements.
    pub total: TotalShift,
}

/// WHAT THE SHIFT ARITHMETIC READS OUT OF AN `AccessDetailsAffine`, and the states it cannot be read
/// through.
///
/// ⭐ THE MECHANISM FOR REACHING OPERANDS IS ALLOWED TO CHANGE; THE ARITHMETIC IS NOT. Entries 191,
/// 192 and 291 each open with four getter calls on the same `ad` — `getLayoutCoeffs`,
/// `getTransferOrder`, `getSubscriptsMap`, `getElementWidth` — and the C++ can call them because a
/// null `AffineMap` and a zero element width are values it will happily dereference and divide by.
/// Resolving both at ONE named seam is what keeps every function below free of a state it has no
/// answer for.
///
/// ⛔ `None` IS "THESE ACCESS DETAILS WERE NEVER CONSTRUCTED", not a refusal to shift.
/// `constructAccessDetails` sets the subscripts map and the element width before any of this runs;
/// `AccessDetailsAffine::new` leaves them `None` and `Bits(0)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftInputs<'a> {
    /// `ad.getLayoutCoeffs()` — one stride per dimension of the memory view, plus a trailing constant
    /// term (see [`LayoutCoeff`]).
    pub layout_coeffs: &'a [LayoutCoeff],
    /// `ad.getTransferOrder()` — the access's `load_order`/`store_order`.
    pub transfer_order: &'a AffineMap,
    /// `ad.getSubscriptsMap()` — one result per dimension of the view, in the view's own order.
    pub subscripts_map: &'a AffineMap,
    /// `dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth()`, resolved once.
    pub elements_per_stick: ElementsPerStick,
    /// `ad.getElementWidth()` itself — the divisor `getMaxImmutableRange(comp)` crosses to become a
    /// count of elements ([`calculate_shifts`], `:436`). [`ElementsPerStick`] is a quotient BY it and
    /// cannot give it back.
    pub element_width: NonZeroU32,
    /// `ad.getIndicesCoeffDict()[nullptr]` — the whole composed constant offset of the access, which
    /// is what `:412` tests against zero and `:439` against the immutable budget.
    pub const_offset: i64,
}

impl<'a> ShiftInputs<'a> {
    /// THE FOUR FACTS, READ OFF ONE `AccessDetailsAffine`.
    #[must_use]
    pub fn of<A: Arch>(ad: &'a AccessDetailsAffine<'a>) -> Option<ShiftInputs<'a>> {
        let element_width = NonZeroU32::new(ad.base.element_width.0)?;
        Some(ShiftInputs {
            layout_coeffs: &ad.base.layout_coeffs,
            transfer_order: &ad.base.transfer_order,
            subscripts_map: ad.subscripts_map.as_ref()?,
            elements_per_stick: ElementsPerStick::of::<A>(element_width)?,
            element_width,
            const_offset: ad.indices_coeff_dict.constant,
        })
    }
}

/// Replaces: e191_calculateFullShift
///
/// **191/384** `MutableStartAddrShiftingPass::calculateFullShift` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462` (25L).
///
/// ```cpp
/// int64_t MutableStartAddrShiftingPass::calculateFullShift(
///     SmallVectorImpl<int64_t> &shifts, agen::AccessDetailsAffine &ad) const {
///   DT_CHECK(shifts.empty());
///   auto layout_coeffs = ad.getLayoutCoeffs();
///
///   auto ordered_subscripts_map =
///       ad.getTransferOrder().compose(ad.getSubscriptsMap());
///   int num_dims = ordered_subscripts_map.getNumDims();
///   int64_t total_shift = 0;
///   for (int i = 0, num_res = ordered_subscripts_map.getNumResults(); i < num_res;
///        ++i) {
///     AffineExpr expr = ordered_subscripts_map.getResult(i);
///     SmallVector<int64_t> coeffs;
///     affine::FlatAffineValueConstraints constraints;
///     auto flat_result =
///         getFlattenedAffineExpr(expr, num_dims, 0, &coeffs, &constraints);
///     int constant_offset = coeffs.size() != num_dims ? coeffs.back() : 0;
///     shifts.push_back(constant_offset);
///     total_shift += constant_offset * layout_coeffs[i];
///   }
///   // Ensure shifts are completed in terms of sticks. The mutable address start
///   // should already be in terms of sticks.
///   int num_elems_in_stick =
///       dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();
///   DT_CHECK(total_shift % num_elems_in_stick == 0);
///   return total_shift;
/// }
/// ```
///
/// # ⭐⭐ THE WHOLE CONSTANT OFFSET, MOVED OUT OF THE SUBSCRIPTS
///
/// This is the arm `calculateShifts` takes when the immutable address space can absorb everything —
/// `if (immutable_space >= const_offset) total_shift = calculateFullShift(shifts, ad);` (`:439-440`).
/// Each subscript of the access is `<something that varies> + <a constant>`; the constant is an offset
/// the transfer will pay for on every step, and it can instead be added ONCE into the transfer's
/// immutable base. So: flatten each subscript, take its constant column, and that is the shift for
/// that dimension. `calculatePartialShift` (entry 291) is the other arm, and it is the one that has to
/// choose which dimensions to spend a limited budget on.
///
/// # ⛔ THE `getFlattenedAffineExpr` CALL IS THE FUNCTION, AND ITS TERNARY IS NOT A BOUNDS CHECK
///
/// `coeffs.size() != num_dims ? coeffs.back() : 0` looks like it guards an index. It does not: the
/// flattened row is `num_dims + num_symbols + num_locals + 1` wide, so it is longer than `num_dims`
/// for every expression that flattens at all, and the `: 0` arm is dead. What the guard really shields
/// is FAILURE — `flat_result` is bound and never read (`:476-477`), and a non-affine subscript leaves
/// `coeffs` empty, where `coeffs.back()` is undefined behaviour. [`AffineExpr::flatten`](crate::islands::dataflow_ir::ty::AffineExpr::flatten) names the
/// constant column instead, and answers the two non-affine shapes with a `todo!` rather than with
/// whatever was on the stack.
///
/// # ⛔ AND THE CONSTANT COLUMN IS NOT "WHATEVER LITERAL APPEARS IN THE SUBSCRIPT"
///
/// `(d0 + 5) floordiv 8` has a 5 in it and a constant column of ZERO: the 5 lives inside the quotient,
/// and shifting 5 out of it would move the start address eight times too far. `(d0 * 8) mod 4` is
/// nothing at all. That is the whole reason this goes through MLIR's flattener rather than pattern-
/// matching an `Add` against a literal — see [`AffineExpr::flatten`](crate::islands::dataflow_ir::ty::AffineExpr::flatten).
///
/// # ⚠️ THREE INDEX SPACES THE REFERENCE SPELLS THE SAME WAY
///
/// `shifts[i]`, `layout_coeffs[i]` and `ordered_subscripts_map.getResult(i)` are indexed by the same
/// `i` here, and they do not all mean the same thing once the transfer order is not the identity. See
/// [`SubscriptResult`], which is where that is written down.
///
/// # ⭐ `zip` WHERE THE REFERENCE INDEXES, AND IT IS THE SAME PAIRING
///
/// `layout_coeffs` is one stride per view dimension PLUS a trailing constant term — `layout_coeffs.size()
/// == operands.size() + 1` (`LoweringXRF.cpp:50`) — so it is longer than the result list, and
/// `zip` drops exactly that trailing term, which `layout_coeffs[i]` never reaches either. Where it is
/// SHORTER, the reference reads out of bounds and this stops early; entry 192's
/// `DT_CHECK(layout_coeffs.size() > num_dims)` (`:570`) is the reference's own half-measure against
/// that, and it compares against the ITERATOR count rather than the result count. Entry 255 asks the
/// right question — `DT_CHECK(layout_coeffs.size() >= shifts.size())` (`:636`) — but only after this
/// function has already read the list.
///
/// # Arguments
///
/// * `inputs` — the access's layout coefficients, transfer order, subscripts map and stick size. See
///   [`ShiftInputs`].
#[must_use]
pub fn calculate_full_shift(inputs: &ShiftInputs<'_>) -> FullShift {
    // `auto ordered_subscripts_map = ad.getTransferOrder().compose(ad.getSubscriptsMap());`
    let ordered_subscripts_map = inputs.transfer_order.compose(inputs.subscripts_map);
    // `int num_dims = ordered_subscripts_map.getNumDims();` — the space the subscripts are flattened
    // over, which is the composed map's, i.e. the SUBSCRIPTS map's dimension count.
    let num_dims = ordered_subscripts_map.dims;

    let mut shifts = Vec::new();
    let mut total_shift = 0;
    for (expr, layout_coeff) in ordered_subscripts_map
        .results
        .iter()
        .zip(inputs.layout_coeffs)
    {
        // `getFlattenedAffineExpr(expr, num_dims, 0, &coeffs, &constraints)`, then `coeffs.back()`.
        // The `0` is the symbol count: a subscripts map is built by `AffineMap::get(num_dims, 0, ..)`.
        let constant_offset = expr.flatten(num_dims, 0).constant;
        shifts.push(Shift(constant_offset));
        total_shift += constant_offset * layout_coeff.0;
    }

    FullShift {
        shifts,
        // `DT_CHECK(total_shift % num_elems_in_stick == 0); return total_shift;`
        total: inputs.elements_per_stick.total(total_shift),
    }
}

/// HOW MUCH IMMUTABLE ADDRESS ONE DIMENSION'S CONSTANT OFFSET BUYS — `layout_coeffs[i] *
/// constant_offset` (`:582`), the key `calculateDimWeights` ranks by.
///
/// ⛔⛔ THE PRODUCT, THOUGH THE REFERENCE'S OWN COMMENT SAYS QUOTIENT. *"The weight would be
/// <layout coefficient> / <constant>"* (`:500`) describes a division and `:582` writes a
/// multiplication. The product is the one that means something: shifting all of dimension `i`'s
/// constant offset out moves the immutable start address by exactly `offset * stride` elements, so
/// ranking by it spends a limited budget on the dimension that empties the most of the mutable
/// address. A quotient would rank a 4096-stride dimension carrying an offset of 1 ABOVE a
/// 256-stride one carrying 128, and buy 4096 elements where 32768 were available. The comment is
/// stale; the code is the specification.
///
/// ⭐ IN ELEMENTS, like [`TotalShift`] and unlike [`Shift`] — it is a stride times an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Weight(pub i64);

/// ONE SUBSCRIPT'S CONSTANT OFFSET AND WHAT IT IS WORTH — `struct DimWeight` (`:81-88`):
///
/// ```cpp
/// /// @brief This data structure contains information about the constant offset
/// /// for a given subscripts map result represented by dim.
/// struct DimWeight {
///   DimWeight(int dim, int64_t offset, int64_t weight)
///       : dim_(dim), offset_(offset), weight_(weight) {}
///   int dim_;
///   // The constant offset for this dim, not considering layout map.
///   int64_t offset_;
///   int64_t weight_;
/// };
/// ```
///
/// ⭐ THE THREE FIELDS ARE THREE UNITS, AND EACH IS ITS OWN NEWTYPE HERE: a POSITION
/// ([`SubscriptResult`]), an INDEX ([`Shift`]) and an ELEMENT COUNT ([`Weight`]). The reference
/// spells the last two `int64_t` and the header has to say in a comment which is which — *"The
/// constant offset for this dim, not considering layout map"* (`:85`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimWeight {
    /// `dim_` (`:84`) — which result of the ORDERED subscripts map. See [`SubscriptResult`].
    pub dim: SubscriptResult,
    /// `offset_` (`:86`) — that result's constant offset, in the dimension's own index units.
    pub offset: Shift,
    /// `weight_` (`:87`) — `offset * layout_coeffs[dim]`, in elements.
    pub weight: Weight,
}

/// Replaces: e192_calculateDimWeights
///
/// **192/384** `MutableStartAddrShiftingPass::calculateDimWeights` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560` (26L).
///
/// ```cpp
/// void MutableStartAddrShiftingPass::calculateDimWeights(
///     SmallVectorImpl<DimWeight> &dim_weights,
///     agen::AccessDetailsAffine &ad) const {
///   DT_CHECK(dim_weights.empty());
///   auto layout_coeffs = ad.getLayoutCoeffs();
///
///   auto ordered_subscripts_map =
///       ad.getTransferOrder().compose(ad.getSubscriptsMap());
///   SmallVector<int64_t> ordered_constant_offsets;
///   int num_dims = ordered_subscripts_map.getNumDims();
///   DT_CHECK(layout_coeffs.size() > num_dims);
///   for (int i = 0, num_res = ordered_subscripts_map.getNumResults(); i < num_res;
///        ++i) {
///     AffineExpr expr = ordered_subscripts_map.getResult(i);
///     SmallVector<int64_t> coeffs;
///     affine::FlatAffineValueConstraints constraints;
///     auto flat_result =
///         getFlattenedAffineExpr(expr, num_dims, 0, &coeffs, &constraints);
///     DT_CHECK(!coeffs.empty());
///     int constant_offset = coeffs.size() != num_dims ? coeffs.back() : 0;
///     ordered_constant_offsets.push_back(constant_offset);
///     dim_weights.emplace_back(i, constant_offset,
///                              layout_coeffs[i] * constant_offset);
///   }
///
///   llvm::sort(dim_weights, [](DimWeight &a, DimWeight &b) -> bool {
///     return a.weight_ > b.weight_;
///   });
/// }
/// ```
///
/// # ⭐⭐ THE ORDER A LIMITED BUDGET IS SPENT IN
///
/// This is [`calculate_full_shift`]'s loop again — the same `compose`, the same flatten, the same
/// constant column — with one extra product and a sort. It exists because the other arm of
/// `calculateShifts` cannot shift everything: `calculatePartialShift` walks these weights highest
/// first and takes as much of each dimension's offset as the remaining immutable space affords
/// (`:527-547`), which is steps 1-3 of its own plan (`:498-503`).
///
/// ⭐ AND THE VENDOR'S FIXTURE PROVES THE ORDER, ARITHMETICALLY. `mutable_start_addr_shift_partial.mlir`
/// runs the same access as the full-shift fixture — offsets `(64, 16, 128)` against strides
/// `(1, 64, 256)` — under `-dcc-mutable-start-addr-shifting-max-immutable-size=240000`, i.e. a budget
/// of `240000 / 16 - 2048 = 12952` elements. The weights are `(64, 1024, 32768)`, so the walk starts
/// at dimension 2: `12952 / 256 = 50` of its 128 goes, leaving 152 and a subscript of `+ 78`; then
/// dimension 1 takes `152 / 64 = 2` of its 16, leaving 24 and `+ 14`; then dimension 0 takes all 24,
/// which the stick remainder immediately gives back. That is exactly the `CHECK`:
/// `[64, %arg1 * 3 + 14, %arg2 * 2 + 78]` with a start address of `2048 + 12928 = 14976`
/// (`mutable_start_addr_shift_partial.mlir:23`, `:29`). Any other order produces other numbers.
///
/// # ⛔ THE DEAD LIST
///
/// `ordered_constant_offsets` is declared at `:568` and pushed at `:580` and those are the only two
/// lines of the translation unit that mention it — the same shape as `calculatePartialShift`'s
/// `dim_order` (see [`SubscriptResult`]). The offsets it collects are already in `dim_weights`, so
/// the port keeps the field and drops the vector.
///
/// # ⛔ THE SORT IS UNSTABLE IN THE REFERENCE AND STABLE HERE, AND THAT IS A DECISION
///
/// `llvm::sort` is `std::sort` with introsort's pivoting, so two dimensions of EQUAL weight come out
/// in an unspecified order — and `calculatePartialShift` is order-dependent, because each dimension
/// it visits consumes budget the next one then does not have (`:544`). The reference therefore has an
/// input class for which its own output is unspecified: any access whose `offset * stride` products
/// collide, which `(offset 2, stride 64)` and `(offset 64, stride 2)` do. This port sorts stably, so
/// ties keep TRANSFER order, and the answer is a function of the input.
///
/// # ⛔ TWO `DT_CHECK`S THAT ARE NOT THE ONES THIS LOOP NEEDS
///
/// `DT_CHECK(layout_coeffs.size() > num_dims)` (`:570`) bounds the list by the map's DIMENSION count
/// while the loop indexes it by RESULT number — on the fixture that is `4 > 2` for three results,
/// so it passes without covering the access it is guarding. `DT_CHECK(!coeffs.empty())` (`:578`) is
/// the real one, and it is the guard [`calculate_full_shift`] omits from the identical loop one
/// function earlier. Neither is reproduced: `zip` ends the walk where the strides do, and
/// [`AffineExpr::flatten`](crate::islands::dataflow_ir::ty::AffineExpr::flatten) has no empty row to
/// return.
///
/// # Arguments
///
/// * `inputs` — the access's layout coefficients, transfer order and subscripts map. See
///   [`ShiftInputs`]. The element width goes unread here; the invariant it serves belongs to the
///   total, and this function computes no total.
#[must_use]
pub fn calculate_dim_weights(inputs: &ShiftInputs<'_>) -> Vec<DimWeight> {
    // The same two lines as entry 191, deliberately: `ad.getTransferOrder().compose(..)` and the
    // dimension count of what comes out of it (`:566-569`).
    let ordered_subscripts_map = inputs.transfer_order.compose(inputs.subscripts_map);
    let num_dims = ordered_subscripts_map.dims;

    let mut dim_weights: Vec<DimWeight> = (0u32..)
        .zip(
            ordered_subscripts_map
                .results
                .iter()
                .zip(inputs.layout_coeffs),
        )
        .map(|(i, (expr, layout_coeff))| {
            // `getFlattenedAffineExpr(expr, num_dims, 0, ..)`, then `coeffs.back()` (`:576-579`).
            let constant_offset = expr.flatten(num_dims, 0).constant;
            // `dim_weights.emplace_back(i, constant_offset, layout_coeffs[i] * constant_offset);`
            DimWeight {
                dim: SubscriptResult(i),
                offset: Shift(constant_offset),
                weight: Weight(layout_coeff.0 * constant_offset),
            }
        })
        .collect();

    // `llvm::sort(dim_weights, [](DimWeight &a, DimWeight &b) { return a.weight_ > b.weight_; });`
    // — descending by weight. `Reverse` over a stable sort, for the reason above.
    dim_weights.sort_by_key(|dim_weight| Reverse(dim_weight.weight));
    dim_weights
}

/// Replaces: e254_offsetShifts
///
/// **254/384** `MutableStartAddrShiftingPass::offsetShifts` — adds `offset` to the shift of the first
/// subscript the transfer visits along `d0`, and to no other.
///
/// ⚠️ THE SECOND `DT_CHECK` IS VACUOUS (`:610`): `offset < 0 ? extents[res] >= offset : true` compares
/// a count against a NEGATIVE, so it holds for every input, and the reference's own TODO above it
/// says the check it wanted — that the innermost extent has room for the elements a negative shift
/// adds — is unwritten.
pub fn offset_shifts(shifts: &mut [Shift], transfer_order: &AffineMap, offset: Shift) {
    // `if (offset == 0) return;`
    if offset == Shift(0) {
        return;
    }
    // `for (; res < num_res; ++res) if (transfer_order.getResult(res).isFunctionOfDim(0)) break;`
    // — a transfer order with no `d0` anywhere is the reference's abort, and there is no shift to
    // place; see [`SubscriptResult`] for why the position indexes `shifts` at all.
    if let Some((slot, _)) = shifts
        .iter_mut()
        .zip(&transfer_order.results)
        .find(|(_, expr)| expr.is_function_of_dim(0))
    {
        // `shifts[res] += offset;`
        slot.0 += offset.0;
    }
}

/// WHAT `applyShifts` LEAVES BEHIND — its returned map, plus the start address
/// `updateMemViewStartAddress` assigned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftedMemView {
    /// The subscripts map with every shift taken out of it — `res - shift` per result.
    pub subscripts_map: AffineMap,
    /// `arith.constant <view.start + total.elements()> : index`.
    pub start_address: DfirOp,
    /// The value the view's `start_address` operand is assigned.
    pub start: Val,
    /// `Σ shifts[i] * layout_coeffs[i]` — what the shifts moved the immutable address by.
    pub total: TotalShift,
}

/// Replaces: e255_applyShifts
///
/// **255/384** `MutableStartAddrShiftingPass::applyShifts` — subtracts the shifts from the subscripts
/// map and adds what they are worth in elements to the view's start address.
///
/// ⛔ A [`ConstStartMemView`] PINS ARM ONE of `updateMemViewStartAddress`, tabled on
/// [`super::tf_mutable_addr_splitting::create_new_mem_view_with_mod`] — but UNLIKE entry 115 the
/// other arms are reachable here: `hasValidL3ImmutableAddr` (`Dialect/Agen/Utils.cpp:140`), which
/// entry 323 asserts before calling this, admits a `subi` toggle and an `scf.if` tree too.
#[must_use]
pub fn apply_shifts(
    vals: &mut Values,
    shifts: &[Shift],
    inputs: &ShiftInputs<'_>,
    view: &ConstStartMemView<'_>,
) -> ShiftedMemView {
    // `for (auto [shift, res] : zip(shifts, subscripts_map.getResults())) new_exprs.emplace_back(res -
    // shift);` — `AffineExpr::operator-` SIMPLIFIES, so a result whose whole constant term was taken
    // prints as `d0 * 3` and not as `d0 * 3 + 0`. The zip is `DT_CHECK(shifts.size() ==
    // subscripts_map.getNumResults())`.
    let results = inputs
        .subscripts_map
        .results
        .iter()
        .zip(shifts)
        .map(|(res, shift)| res.clone().added(AffineExpr::Const(-shift.0)))
        .collect();

    // `total_shift += shifts[i] * layout_coeffs[i]`, RECOMPUTED and not entry 191's total, because
    // `calculatePartialShift` has since rewritten `shifts` (`:544-557`). The zip is
    // `DT_CHECK(layout_coeffs.size() >= shifts.size())`, and this arm asks nothing about sticks — see
    // [`TotalShift::PartialStick`].
    let total = inputs.elements_per_stick.total(
        shifts
            .iter()
            .zip(inputs.layout_coeffs)
            .map(|(shift, layout_coeff)| shift.0 * layout_coeff.0)
            .sum(),
    );

    // `DT_CHECK(mem_view_op->hasOneUse())` — the pass refused a multi-use L3 view before it got here
    // (`:156-158`). Then `updateMemViewStartAddress`, arm one: one `arith.constant` for `start +
    // modifier`, assigned to the view's start address.
    let start = vals.mint();
    ShiftedMemView {
        // `AffineMap::get(num_dims, 0, new_exprs, ctx)` — the ORIGINAL map's dimension count.
        subscripts_map: AffineMap {
            dims: inputs.subscripts_map.dims,
            syms: 0,
            results,
        },
        start_address: DfirOp::Arith(arith::Op::Constant {
            result: start,
            value: view.start + total.elements(),
        }),
        start,
        total,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 291/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// HOW MANY ELEMENTS OF IMMUTABLE ADDRESS SPACE ARE STILL FREE — `(max_immutable_range /
/// ad.getElementWidth()) - max_immutable` (`:436-437`).
///
/// ⛔ SIGNED, BECAUSE IT CAN BE NEGATIVE: the view's own maximum immutable address may already exceed
/// the range the EBR can name, and the reference then divides a negative budget by every stride
/// (`:534`) and takes the `constant_shift <= 0` arm on all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImmutableSpace(pub i64);

/// WHAT `calculatePartialShift` HANDS BACK — the out-parameter and the return value together, as
/// [`FullShift`] does. The total is stick-aligned by construction here: the remainder is subtracted
/// off and handed back to the mutable side through [`offset_shifts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialShift {
    /// One per result of the ORIGINAL subscripts map, zero where nothing moved — see
    /// [`SubscriptResult`].
    pub shifts: Vec<Shift>,
    /// What those shifts add up to, in elements, less the part-stick remainder.
    pub total: TotalShift,
}

/// Replaces: e291_calculatePartialShift
///
/// **291/384** `MutableStartAddrShiftingPass::calculatePartialShift` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490` (66L): spend a limited immutable
/// budget on the heaviest dimensions first, then give back what does not fill a whole stick.
///
/// ⛔ A NEGATIVE `offset_` PRODUCES A NEGATIVE SHIFT AND HANDS BUDGET BACK, because
/// `if (constant_shift > dim_weight.offset_) constant_shift = dim_weight.offset_` clamps DOWNWARD
/// only (`:541-542`) — which is also the one way `DT_CHECK(total_shift >= 0)` (`:551`) can fail, and
/// [`TotalShift`] is signed already.
/// ⚠️ `dim_order` (`:511-517`) IS WRITTEN AND NEVER READ; see [`SubscriptResult`].
#[must_use]
pub fn calculate_partial_shift(
    inputs: &ShiftInputs<'_>,
    immutable_space: ImmutableSpace,
) -> PartialShift {
    // `calculateDimWeights(dim_weights, ad);` — heaviest first, which is steps 1-3 of the plan the
    // reference states at `:497-503`.
    let dim_weights = calculate_dim_weights(inputs);

    // *"Initialize all shifts to zero as the shifts may not be analyzed in order."* (`:522-524`).
    let mut shifts = vec![Shift(0); inputs.subscripts_map.results.len()];

    let mut curr_immutable_space = immutable_space;
    for dim_weight in &dim_weights {
        // `DT_CHECK(layout_coeffs.size() > dim_weight.dim_);` — and a stride of zero is the
        // reference's own division by zero; skipping leaves this dim's shift 0 and its budget whole.
        let Some(&layout_coeff) = inputs.layout_coeffs.get(dim_weight.dim.index()) else {
            continue;
        };
        if layout_coeff.0 == 0 {
            continue;
        }

        // `int64_t constant_shift = curr_immutable_space / layout_coeffs[dim_weight.dim_];`
        // *"If there isn't enough space to fit even 1*<layout coeff>, move the the next dim."*
        let mut constant_shift = curr_immutable_space.0 / layout_coeff.0;
        if constant_shift <= 0 {
            continue;
        }
        // `if (constant_shift > dim_weight.offset_) constant_shift = dim_weight.offset_;`
        constant_shift = constant_shift.min(dim_weight.offset.0);

        // `curr_immutable_space -= constant_shift * layout_coeffs[dim_weight.dim_];`
        curr_immutable_space.0 -= constant_shift * layout_coeff.0;
        // `DT_CHECK(shifts.size() > dim_weight.dim_); shifts[dim_weight.dim_] = constant_shift;`
        if let Some(slot) = shifts.get_mut(dim_weight.dim.index()) {
            *slot = Shift(constant_shift);
        }
    }

    // `int64_t total_shift = immutable_space - curr_immutable_space;`
    let total_shift = immutable_space.0 - curr_immutable_space.0;
    // `int64_t remainder = total_shift % num_elems_in_stick; total_shift -= remainder;`
    let stick = i64::try_from(inputs.elements_per_stick.elements().0).unwrap_or(i64::MAX);
    let remainder = total_shift % stick;
    // `if (remainder != 0) offsetShifts(shifts, ad, -remainder);` — the guard is [`offset_shifts`]'s
    // own first line.
    offset_shifts(&mut shifts, inputs.transfer_order, Shift(-remainder));

    PartialShift {
        shifts,
        total: inputs.elements_per_stick.total(total_shift - remainder),
    }
}

/// `isL3ImmutableAddrEven(evaluator, mem_view_op.getStartAddress(), num_elems_in_stick)` FOR A
/// CONSTANT START — `evaluateDivideByConst(ev, n).isDivisibleBy(2)` over the one value that
/// evaluation yields (`Dialect/Agen/Utils.cpp:218-288`), which is
/// [`super::tf_mutable_addr_splitting::calculate_partition_sizes`]'s own reading of it.
fn is_l3_immutable_addr_even(view: &ConstStartMemView<'_>, stick: ElementsPerStick) -> bool {
    (view.start / stick.elements().0.cast_signed()) % 2 == 0
}

/// Replaces: e308_calculateShifts
///
/// **308/384** `MutableStartAddrShifting.cpp:396` (61L) — how much of each subscript's constant
/// offset moves out into the immutable start address, one entry per subscripts-map result.
///
/// ⛔ THE `const_offset == 0` ARM STILL SHIFTS ON SEN1P5: an odd immutable address is realigned by
/// [`offset_shifts`] of MINUS one stick, so the shifts come back NEGATIVE. ⚠️ And both
/// `isL3ImmutableAddrAllOdd` aborts are unreachable on a constant start: it is even or it is all-odd.
#[must_use]
pub fn calculate_shifts<A: Arch>(
    inputs: &ShiftInputs<'_>,
    half: L3Half,
    view: &ConstStartMemView<'_>,
) -> Vec<Shift> {
    // `int64_t num_elems_in_stick = dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();`
    let stick = inputs.elements_per_stick;
    let one_stick = Shift(-stick.elements().0.cast_signed());

    // *"If the constant offset is 0, there is no mutable address to shift."*
    if inputs.const_offset == 0 {
        // `for (int i = 0, e = subscripts_map.getNumResults(); i < e; ++i) shifts.push_back(0);`
        let mut shifts = vec![Shift(0); inputs.subscripts_map.results.len()];
        // *"If arch is sen1p5 up, immutable addresses need to contain even values only."*
        // `if (dcc_ext_ctx_.getArch() < IsaCoreGen::SEN1P5_ISA) return;`
        if A::GEN < IsaGen::Sen1p5 {
            return shifts;
        }
        // `if (isL3ImmutableAddrEven(evaluator, getStartAddress(), num_elems_in_stick)) return;`
        if is_l3_immutable_addr_even(view, stick) {
            return shifts;
        }
        // `DT_CHECK_MSG(isL3ImmutableAddrAllOdd(...), "All immutable addrs must be odd to execute
        // even shift.");` — see this function's doc for why a constant start cannot fail it.
        //
        // *"If the immutable address is odd, it needs to be shifted by one stick."*
        // `offsetShifts(shifts, ad, -num_elems_in_stick);`
        offset_shifts(&mut shifts, inputs.transfer_order, one_stick);
        return shifts;
    }

    // *"If the constant offset is not 0, how much mutable address we can shift needs to be
    // calculated based on immutable address space."*
    //
    // `auto max_immutable = getMaxImmutableAddress(evaluator, mem_view_op);` — the constant itself,
    // for the reason [`super::tf_mutable_addr_splitting::calculate_partition_sizes`] records.
    // `int64_t immutable_space = (getMaxImmutableRange(ad.getComp()) / ad.getElementWidth()) -
    // max_immutable;`
    let immutable_space = ImmutableSpace(
        0i64.saturating_add_unsigned(
            max_immutable_range::<A>(half).bits() / u64::from(inputs.element_width.get()),
        )
        .saturating_sub(view.start),
    );

    // `if (immutable_space >= const_offset) total_shift = calculateFullShift(shifts, ad); else
    // total_shift = calculatePartialShift(shifts, ad, immutable_space);`
    let (mut shifts, total) = if immutable_space.0 >= inputs.const_offset {
        let full = calculate_full_shift(inputs);
        (full.shifts, full.total)
    } else {
        let partial = calculate_partial_shift(inputs, immutable_space);
        (partial.shifts, partial.total)
    };

    // `if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA)` — [`IsaGen`] is ordered.
    if A::GEN >= IsaGen::Sen1p5 {
        // *"The immutable addr and the shift amount must either both be even or both be odd to keep
        // the immutable addr an even number of sticks."*
        let is_even = is_l3_immutable_addr_even(view, stick);
        // `bool is_even_shift = (total_shift / num_elems_in_stick) % 2 == 0;` — the TOTAL over the
        // stick, not the total itself, and [`TotalShift::elements`] is the reference's `int64_t`.
        let is_even_shift = (total.elements() / stick.elements().0.cast_signed()) % 2 == 0;
        if is_even != is_even_shift {
            // The `is_even ? true : isL3ImmutableAddrAllOdd(...)` abort sits here and cannot fire.
            // `offsetShifts(shifts, ad, -num_elems_in_stick);`
            offset_shifts(&mut shifts, inputs.transfer_order, one_stick);
        }
    }

    shifts
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 323/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e323_shiftMutableAddr
///
/// **323/384** `MutableStartAddrShifting.cpp:366` (19L) — the pass's one entry point per access:
/// move what fits of the constant offset into the immutable start address, or report no shift.
///
/// ⛔ `AffineMap::get(ctx)` IS THE "NO SHIFT" ANSWER, not an empty rewrite — a zero-result map is
/// falsy at both call sites (`:265`, `:307`), so `None` is it. ⛔ And the
/// `DT_CHECK(hasValidL3ImmutableAddr(…, req_even_toggle = arch >= SEN1P5))` is the
/// [`ConstStartMemView`] TYPE: a view whose start is not a constant cannot reach this function.
#[must_use]
pub fn shift_mutable_addr<A: Arch>(
    vals: &mut Values,
    inputs: &ShiftInputs<'_>,
    half: L3Half,
    view: &ConstStartMemView<'_>,
) -> Option<ShiftedMemView> {
    // `SmallVector<int64_t> shifts; calculateShifts(evaluator, shifts, mem_view_op, ad);`
    let shifts = calculate_shifts::<A>(inputs, half, view);

    // *"If all the shifts are 0, there is nothing to shift. Return and report no shift."*
    // `if (std::all_of(shifts.begin(), shifts.end(), [](int64_t s) { return s == 0; })) return
    // AffineMap::get(mem_view_op->getContext());`
    if shifts.iter().all(|shift| shift.0 == 0) {
        return None;
    }

    // `return applyShifts(evaluator, shifts, unit, op, mem_view_op, ad);`
    Some(apply_shifts(vals, &shifts, inputs, view))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 352/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ONE SHIFTED TRANSFER LEAVES BEHIND — the three ops the pass emits, and the one it erases.
///
/// ⛔ `start_address` IS HOISTED OUT OF THE UNIT: `updateMemViewStartAddress`'s const builder sits at
/// the `dataflow.program_unit`, which is why the vendor's `%c33856` prints at function scope while
/// the view it feeds stays inside two `affine.for`s
/// (`Transform/MutableStartAddrShifting/mutable_start_addr_shift_full.mlir:22`, `:26`).
/// ⛔ AND `mem_view` KEEPS THE ORIGINAL'S RESULT [`Val`]: the reference ASSIGNS the new start into the
/// existing view op rather than cloning it, so nothing that reads that view is re-pointed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftedTransfer<'s> {
    /// `arith.constant <start + total>`, at unit scope.
    pub start_address: DfirOp,
    /// The view rebuilt on it, under its own result — see this type's banner.
    pub mem_view: DfirOp,
    /// The transfer rebuilt on the shifted subscripts, where the erased one was.
    pub mem_op: DfirOp,
    /// `if (!op->use_empty()) op->replaceAllUsesWith(new_mem_op);` — (old result, new result). Only
    /// entry 352 can have one: a store and both composites bind nothing.
    pub replaced: Option<(Val, Val)>,
    /// `op->erase();` — the transfer alone. Its use chain stays and is re-pointed.
    pub erased: &'s DfirOp,
    /// What the shift moved the immutable address by, in elements.
    pub total: TotalShift,
}

/// WHAT THE FOUR TRANSFORMS ANSWER — one shifted transfer, or what the reference stopped on.
///
/// ⛔ NOT AN ERROR TYPE. [`Self::NoShiftRequired`] is the reference's own early return for an access
/// with nothing to move (`:216`), [`Self::IndirectIsOnTheShiftedSide`] is entry 355's (`:305-310`),
/// and the four class variants are its opening `DT_CHECK`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutableStartAddrShift<'s> {
    /// The shift, applied.
    Shifted(ShiftedTransfer<'s>),
    /// *"If the new subscripts map is empty that means there was no shifting required."*
    NoShiftRequired,
    /// `DT_CHECK(dyn_cast<agen::VectorLoadOp>(candidate.op_))`.
    NotAVectorLoad,
    /// `DT_CHECK(dyn_cast<agen::VectorStoreOp>(candidate.op_))`.
    NotAVectorStore,
    /// `DT_CHECK(dyn_cast<agen::CompositeLoadAndStoreOp>(candidate.op_))`.
    NotACompositeLoadAndStore,
    /// `DT_CHECK(dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_))`.
    NotACompositeIndirectLoadAndStore,
    /// *"The memory operands on the side of the indirect must have a 0 immutable address so no
    /// shifting can be done."*
    IndirectIsOnTheShiftedSide(MemoryOperandIndex),
    /// `DT_CHECK(succeeded(ad.constructDetails(candidate.mem_index_)))`.
    DetailsNotConstructed(ConstructedDetails),
    /// The access details are complete but carry no subscripts map or element width — see
    /// [`ShiftInputs::of`].
    ShiftInputsNotReadable,
    /// `cast<dataflow::GetLogicalMemoryViewOp>` (entries 352/353) or the
    /// `DT_CHECK(dyn_cast_or_null<..>)` on `ad.getMemRef()` (entries 354/355).
    MemRefIsNotAMemoryView,
    /// The view's start address is not an `arith.constant` — a seam of [`ConstStartMemView`], not of
    /// these four functions: the vendor's query-map, toggle and conditional start addresses all land
    /// here, and `shiftMutableAddr`'s own `hasValidL3ImmutableAddr` admits them.
    MemViewStartIsNotConstant,
    /// `getMaxImmutableRange`'s `DT_CHECK(is_any_of(comp, L3LU, L3SU))`, reached through
    /// [`calculate_shifts`]. ⚠️ ASKED EARLIER THAN THE REFERENCE ASKS IT — `calculateShifts` only
    /// reaches that range on a NON-zero constant offset (`:433`) — and unreachable either way:
    /// `runOnOperation`'s own component gate (`:149-150`) admits none but those two.
    CandidateIsNotOnAnL3Half(DfirUnit),
}

/// `updateMemViewStartAddress` ARM ONE, **IN PLACE** — the view on its new start address, keeping its
/// own result for the reason [`ShiftedTransfer`] gives.
/// [`super::tf_mutable_addr_splitting::create_new_mem_view_with_mod`] is the same arm for a CLONE and
/// tables the other three; ⛔ unlike there, they are reachable from here — `hasValidL3ImmutableAddr`
/// (`Dialect/Agen/Utils.cpp:140`) admits a `subi` toggle and an `scf.if` tree, and this island's
/// [`ConstStartMemView`] cannot spell either.
fn mem_view_with_start(result: Val, view: &ConstStartMemView<'_>, start: Val) -> DfirOp {
    DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
        result,
        from: view.from,
        start,
        layout: view.layout.clone(),
        ty: view.ty.clone(),
    })
}

/// WHAT THE FOUR TRANSFORMS SHARE BETWEEN THEIR OPENING CAST AND THEIR CLONE — the view, the L3 half,
/// the shift inputs and [`shift_mutable_addr`], in the reference's order (`:210-217`, `:238-245`,
/// `:270-281`, `:317-328`, line for line the same).
enum ShiftOfOneAccess<'s> {
    /// The split side's view, and the shift applied to it.
    Shifted(ConstStartMemView<'s>, ShiftedMemView),
    /// The reference returned, or aborted, before the clone.
    Stopped(MutableStartAddrShift<'s>),
}

/// The four transforms' shared middle — see [`ShiftOfOneAccess`].
fn shift_one_access<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    ad: &AccessDetailsAffine<'_>,
    mem_ref: Val,
    scope: &'s [DfirOp],
) -> ShiftOfOneAccess<'s> {
    let view = match SplitCandidateView::resolve(mem_ref, scope) {
        SplitCandidateView::ConstantStart(view) => view,
        SplitCandidateView::NonConstantStart => {
            return ShiftOfOneAccess::Stopped(MutableStartAddrShift::MemViewStartIsNotConstant);
        }
        SplitCandidateView::NotAMemoryView => {
            return ShiftOfOneAccess::Stopped(MutableStartAddrShift::MemRefIsNotAMemoryView);
        }
    };
    // `getMaxImmutableRange(ad.getComp())`'s own `DT_CHECK`, reached from `calculateShifts`.
    let Some(half) = L3Half::of(candidate.comp) else {
        return ShiftOfOneAccess::Stopped(MutableStartAddrShift::CandidateIsNotOnAnL3Half(
            candidate.comp,
        ));
    };
    let Some(inputs) = ShiftInputs::of::<A>(ad) else {
        return ShiftOfOneAccess::Stopped(MutableStartAddrShift::ShiftInputsNotReadable);
    };
    // `shiftMutableAddr(candidate.unit_, op, mem_view_op, ad)`, then *"if the new subscripts map is
    // empty that means there was no shifting required."*
    match shift_mutable_addr::<A>(vals, &inputs, half, &view) {
        Some(shifted) => ShiftOfOneAccess::Shifted(view, shifted),
        None => ShiftOfOneAccess::Stopped(MutableStartAddrShift::NoShiftRequired),
    }
}

/// Replaces: e352_transformVectorLoad
///
/// **352/384** `MutableStartAddrShiftingPass::transformVectorLoad` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201` (28L): move what fits of one
/// `agen.vector_load`'s constant offset out of its subscripts and into its view's start address.
///
/// ⛔ THE CONSUMERS ARE RE-POINTED, NOT CLONED (`:222`): `replaceAllUsesWith` where entry 306's split
/// needs `cloneUseChainToNewOp`, because one load still becomes exactly one load here.
#[must_use]
pub fn transform_vector_load<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    scope: &'s [DfirOp],
) -> MutableStartAddrShift<'s> {
    // `auto op = dyn_cast<agen::VectorLoadOp>(candidate.op_); DT_CHECK(op);`
    let (Some(op), DfirOp::Agen(agen_op)) = (VectorLoadOp::of(candidate.op), candidate.op) else {
        return MutableStartAddrShift::NotAVectorLoad;
    };

    // "Collect the relevant access details." — `access_details.emplace_insert(mem_index, op, comp)`
    // then `DT_CHECK(succeeded(ad.constructDetails(mem_index)))`. ⚠️ THE CONTAINER IS MECHANISM, as
    // entry 306 records: it memoises one record per memory operand and a vector load has one.
    let mut ad = AccessDetailsAffine::new(agen_op, candidate.comp);
    let details = ad.construct_details(candidate.mem_index, scope);
    if details != ConstructedDetails::Complete {
        return MutableStartAddrShift::DetailsNotConstructed(details);
    }

    // `cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp())`, then the shift.
    let (view, shifted) = match shift_one_access::<A>(vals, candidate, &ad, op.view, scope) {
        ShiftOfOneAccess::Shifted(view, shifted) => (view, shifted),
        ShiftOfOneAccess::Stopped(stopped) => return stopped,
    };

    // `op.cloneWithNewAccessInfo(builder, mem_view_op, new_subscripts_map, ad.getIndices())` — the
    // SAME view value, which is why `mem_view` below keeps its result.
    let result = vals.mint();
    let mem_op = clone_load_with_new_access_info(
        op,
        result,
        op.view,
        view.ty.clone(),
        indices_from_map(&shifted.subscripts_map, &ad.base.indices),
    );

    MutableStartAddrShift::Shifted(ShiftedTransfer {
        start_address: shifted.start_address,
        mem_view: mem_view_with_start(op.view, &view, shifted.start),
        mem_op,
        // `if (!op->use_empty()) op->replaceAllUsesWith(new_mem_op);`
        replaced: (!uses(op.result, scope).is_empty()).then_some((op.result, result)),
        // `op->erase();`
        erased: candidate.op,
        total: shifted.total,
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 353/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e353_transformVectorStore
///
/// **353/384** `MutableStartAddrShiftingPass::transformVectorStore` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229` (27L): entry 352 for an
/// `agen.vector_store`, which is entry 352 minus the re-pointing — a store binds nothing.
///
/// ⚠️ ITS `mem_index_` IS `kDirSrc`, NOT `kDirDst`: the collection's `else` arm hands every
/// non-composite candidate the SOURCE index (`:172`), and the access details read the store's one
/// memref either way.
#[must_use]
pub fn transform_vector_store<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    scope: &'s [DfirOp],
) -> MutableStartAddrShift<'s> {
    // `auto op = dyn_cast<agen::VectorStoreOp>(candidate.op_); DT_CHECK(op);`
    let (Some(op), DfirOp::Agen(agen_op)) = (VectorStoreOp::of(candidate.op), candidate.op) else {
        return MutableStartAddrShift::NotAVectorStore;
    };

    // "Collect the relevant access details." — entry 352's, container and all.
    let mut ad = AccessDetailsAffine::new(agen_op, candidate.comp);
    let details = ad.construct_details(candidate.mem_index, scope);
    if details != ConstructedDetails::Complete {
        return MutableStartAddrShift::DetailsNotConstructed(details);
    }

    // `cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp())`, then the shift.
    let (view, shifted) = match shift_one_access::<A>(vals, candidate, &ad, op.view, scope) {
        ShiftOfOneAccess::Shifted(view, shifted) => (view, shifted),
        ShiftOfOneAccess::Stopped(stopped) => return stopped,
    };

    MutableStartAddrShift::Shifted(ShiftedTransfer {
        start_address: shifted.start_address,
        mem_view: mem_view_with_start(op.view, &view, shifted.start),
        // `(void)op.cloneWithNewAccessInfo(builder, mem_view_op, new_subscripts_map,
        //  ad.getIndices());` — ⭐ THE `(void)` DISCARDS THE HANDLE, NOT THE OP: the builder has
        // already inserted it where the store was.
        mem_op: clone_store_with_new_access_info(
            op,
            op.view,
            indices_from_map(&shifted.subscripts_map, &ad.base.indices),
        ),
        // A store has no result, so the reference has no `replaceAllUsesWith` to make.
        replaced: None,
        // `op->erase();`
        erased: candidate.op,
        total: shifted.total,
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 354/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e354_transformCompLoadAndStore
///
/// **354/384** `MutableStartAddrShiftingPass::transformCompLoadAndStore` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256` (41L): entries 352/353 for one side
/// of an `agen.composite_load_and_store`, the OTHER side's view, map and operands passed through.
///
/// ⛔ THE VIEW COMES OFF `ad.getMemRef()`, NOT OFF THE OP (`:268-270`): a composite has two memrefs
/// and only the record built for `candidate.mem_index_` knows which one is the HBM side. Its
/// `dyn_cast_or_null` is [`MutableStartAddrShift::MemRefIsNotAMemoryView`].
#[must_use]
pub fn transform_comp_load_and_store<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    scope: &'s [DfirOp],
) -> MutableStartAddrShift<'s> {
    // `auto op = dyn_cast<agen::CompositeLoadAndStoreOp>(candidate.op_); DT_CHECK(op);`
    let DfirOp::Agen(agen_op) = candidate.op else {
        return MutableStartAddrShift::NotACompositeLoadAndStore;
    };
    let agen::Op::CompositeLoadAndStore(transfer) = agen_op else {
        return MutableStartAddrShift::NotACompositeLoadAndStore;
    };

    // "Collect the relevant access details." — the COMPOSITE record, whose `initialize` picks the
    // operand `mem_index` names.
    let mut ad = AccessDetailsAffineComposite::new(agen_op, candidate.comp);
    let details = ad.construct_details(candidate.mem_index, scope);
    if details != ConstructedDetails::Complete {
        return MutableStartAddrShift::DetailsNotConstructed(details);
    }

    // `dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(ad.getMemRef().getDefiningOp())`.
    let Some(mem_ref) = ad.affine.base.mem_ref else {
        return MutableStartAddrShift::MemRefIsNotAMemoryView;
    };
    let (view, shifted) = match shift_one_access::<A>(vals, candidate, &ad.affine, mem_ref, scope) {
        ShiftOfOneAccess::Shifted(view, shifted) => (view, shifted),
        ShiftOfOneAccess::Stopped(stopped) => return stopped,
    };

    // `op.cloneWithNewAccessInfo(builder, getSrcMemRef(), getDstMemRef(), <src map>, <dst map>,
    //  <src indices>, <dst indices>, getTimeSet().getValue())` — the shifted side takes
    // `new_subscripts_map` with `ad.getIndices()`, the other its own map and its own operands.
    let shifted_side = indices_from_map(&shifted.subscripts_map, &ad.affine.base.indices);
    let (src_indices, dst_indices) = if candidate.mem_index == MemoryOperandIndex::DirSrc {
        (shifted_side, transfer.dst_indices.clone())
    } else {
        (transfer.src_indices.clone(), shifted_side)
    };

    MutableStartAddrShift::Shifted(ShiftedTransfer {
        start_address: shifted.start_address,
        mem_view: mem_view_with_start(mem_ref, &view, shifted.start),
        mem_op: clone_composite_with_new_access_info(
            vals,
            transfer,
            transfer.src,
            src_indices,
            transfer.dst,
            dst_indices,
            transfer.time_set.clone(),
        ),
        // A composite has no result.
        replaced: None,
        // `op->erase();`
        erased: candidate.op,
        total: shifted.total,
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 355/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e355_transformCompIndLoadAndStore
///
/// **355/384** `MutableStartAddrShiftingPass::transformCompIndLoadAndStore` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298` (55L): entry 354 for an
/// `agen.composite_indirect_load_and_store`, refusing the side that carries the indirect.
///
/// ⛔ THE `getEmptyAffineMap()` THE SHIFTED SIDE'S INDIRECT MAP IS GIVEN (`:337`, `:349`) DESCRIBES
/// NOTHING: the two early returns above guarantee that side has no indirect memref, so both arms hand
/// both indirect accesses through unchanged. ⛔ AND THERE IS NO SLOT SWAP — entry 322's
/// `clone_composite_indirect_with_new_access_info` moves a split indirect source into the destination
/// slot; this one keeps every operand where the original had it.
#[must_use]
pub fn transform_comp_ind_load_and_store<'s, A: Arch>(
    vals: &mut Values,
    candidate: &MasCandidate<'s>,
    scope: &'s [DfirOp],
) -> MutableStartAddrShift<'s> {
    // `auto op = dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_); DT_CHECK(op);`
    let DfirOp::Agen(agen_op) = candidate.op else {
        return MutableStartAddrShift::NotACompositeIndirectLoadAndStore;
    };
    let agen::Op::CompositeIndirectLoadAndStore(transfer) = agen_op else {
        return MutableStartAddrShift::NotACompositeIndirectLoadAndStore;
    };

    // *"The memory operands on the side of the indirect must have a 0 immutable address so no
    // shifting can be done."* — `if (mem_index == kDirSrc && op.hasIndirectSrc()) return;` and its
    // destination twin.
    let indirect_on_this_side = match candidate.mem_index {
        MemoryOperandIndex::DirSrc => transfer.indirect_src.is_some(),
        MemoryOperandIndex::DirDst => transfer.indirect_dst.is_some(),
        MemoryOperandIndex::IndSrc | MemoryOperandIndex::IndDst => false,
    };
    if indirect_on_this_side {
        return MutableStartAddrShift::IndirectIsOnTheShiftedSide(candidate.mem_index);
    }

    // "Collect the relevant access details.", then `ad.getMemRef()` — entry 354's, unchanged.
    let mut ad = AccessDetailsAffineComposite::new(agen_op, candidate.comp);
    let details = ad.construct_details(candidate.mem_index, scope);
    if details != ConstructedDetails::Complete {
        return MutableStartAddrShift::DetailsNotConstructed(details);
    }
    let Some(mem_ref) = ad.affine.base.mem_ref else {
        return MutableStartAddrShift::MemRefIsNotAMemoryView;
    };
    let (view, shifted) = match shift_one_access::<A>(vals, candidate, &ad.affine, mem_ref, scope) {
        ShiftOfOneAccess::Shifted(view, shifted) => (view, shifted),
        ShiftOfOneAccess::Stopped(stopped) => return stopped,
    };

    // The clone's twelve arguments, of which only the shifted side's map and operands are new.
    let shifted_side = indices_from_map(&shifted.subscripts_map, &ad.affine.base.indices);
    let (direct_src_indices, direct_dst_indices) =
        if candidate.mem_index == MemoryOperandIndex::DirSrc {
            (shifted_side, transfer.direct_dst_indices.clone())
        } else {
            (transfer.direct_src_indices.clone(), shifted_side)
        };

    MutableStartAddrShift::Shifted(ShiftedTransfer {
        start_address: shifted.start_address,
        mem_view: mem_view_with_start(mem_ref, &view, shifted.start),
        mem_op: clone_composite_indirect_with_new_access_info(
            vals,
            transfer,
            transfer.indirect_src.clone(),
            transfer.direct_src,
            direct_src_indices,
            transfer.indirect_dst.clone(),
            transfer.direct_dst,
            direct_dst_indices,
            transfer.time_set.clone(),
        ),
        // A composite has no result.
        replaced: None,
        // `op->erase();`
        erased: candidate.op,
        total: shifted.total,
    })
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::formats::Bits;
    use crate::generated::DataType;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::{Index, Val, affine, agen, symbol};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, Constraint, ElemType, IntegerSet, MemRef, Vector,
    };
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::{Core, Corelet, DfirUnit, Residency};

    /// 🎯 116/384 — THE DERIVED RANGE IS `2^bitSize * bytesPerStick * 8` ON EACH ARCH.
    ///
    /// `{8, 30, 32, UNSIGNED, true}` for L3LU and L3SU under `coreArch <= RCUDD1A_ISA`, and
    /// `{16, 32, 32, UNSIGNED, true}` above it (`sys-arch-spec/sysdef.cpp:313-360`), with
    /// `bytesPerStick = 128` on both (`:206`).
    #[test]
    fn the_derived_range_is_the_ebr_span_in_bits() {
        for comp in [L3Half::Load, L3Half::Store] {
            assert_eq!(max_immutable_range::<Dd2>(comp).bits(), (1 << 30) * 128 * 8);
            assert_eq!(
                max_immutable_range::<Sen1p5>(comp).bits(),
                (1u64 << 32) * 128 * 8
            );
        }
    }

    /// 🎯 116/384 — AND IT IS THE SPLITTING PASS'S OWN ANSWER, BECAUSE NEITHER FLAG IS SET.
    ///
    /// ⛔ WHICH IS THE ONLY CONFIGURATION IN WHICH THEY AGREE. The two functions read two different
    /// `cl::opt`s ([`MAX_IMMUTABLE_SIZE`] and
    /// [`super::tf_mutable_addr_splitting::MAX_IMMUTABLE_SIZE`]); with both at `cl::init(-1)` they
    /// derive the same span from the same register, and setting one moves one pass only.
    #[test]
    fn the_two_passes_agree_while_neither_override_is_set() {
        assert_eq!(MAX_IMMUTABLE_SIZE, None);
        assert_eq!(
            max_immutable_range::<Dd2>(L3Half::Load),
            super::super::tf_mutable_addr_splitting::max_immutable_range::<Dd2>(L3Half::Load)
        );
        assert_eq!(
            max_immutable_range::<Sen1p5>(L3Half::Store),
            super::super::tf_mutable_addr_splitting::max_immutable_range::<Sen1p5>(L3Half::Store)
        );
    }

    /// 🎯 116/384 — AND THE `double` THE REFERENCE COMPUTES IT IN AGREES, EXACTLY.
    ///
    /// `pow(2, bitSize) * 128 * 8` truncated back to `int64_t` — the arm the port replaced with a
    /// shift.
    #[test]
    fn the_shift_agrees_with_the_reference_pow() {
        for bits in [30u32, 32] {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "reproducing the reference's own double arithmetic, to compare against it"
            )]
            let as_double = (f64::from(2.0_f32).powi(bits as i32) * 128.0 * 8.0) as u64;
            assert_eq!(as_double, (1u64 << bits) * 128 * 8);
        }
    }

    /// 🎯 116/384 — THE `DT_CHECK` ADMITS THE TWO L3 HALVES AND NOTHING ELSE.
    #[test]
    fn only_the_l3_halves_have_an_immutable_range() {
        assert_eq!(L3Half::of(DfirUnit::L3lu), Some(L3Half::Load));
        assert_eq!(L3Half::of(DfirUnit::L3su), Some(L3Half::Store));
        for unit in [
            DfirUnit::Lx,
            DfirUnit::L0,
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::L0lu,
            DfirUnit::Hbm,
            DfirUnit::Sfp,
        ] {
            assert_eq!(L3Half::of(unit), None);
        }
    }

    /// 🎯 116/384 — AND THE RANGE CONVERTS TO ELEMENTS BY THE ELEMENT WIDTH, AS `:437` DIVIDES IT.
    #[test]
    fn the_range_in_elements_is_the_reference_division() {
        let range = max_immutable_range::<Dd2>(L3Half::Load);
        assert_eq!(
            range.elements(DataType::Sen169Fp16).0,
            (1 << 30) * 128 * 8 / 16
        );
        // Twice as many of half the width — the whole point of the division.
        assert_eq!(
            range.elements(DataType::Senint8).0,
            range.elements(DataType::Sen169Fp16).0 * 2
        );
    }

    /// 🎯 191/384 — THE VENDOR'S OWN CASE, END TO END: `[%c64, %arg1 * 3 + %c16, %arg2 * 2 + %c128]`
    /// BECOMES `[0, %arg1 * 3, %arg2 * 2]` AND THE IMMUTABLE START ADDRESS BECOMES **33856**.
    ///
    /// `dcc/test/Transform/MutableStartAddrShifting/mutable_start_addr_shift_full.mlir` — the pass's
    /// own FileCheck fixture, run as `dcc-opt --dcc-mutable-start-addr-shifting`. Its input
    /// (`:149-151`) is an `agen.vector_load` off a view whose `layout_map` is
    /// `(d0, d1, d2) -> (d2 * 256 + d1 * 64 + d0)` over `memref<?x64x4xf16>`, with the identity
    /// `load_order`; its `CHECK-SENT-IR` (`:22-28`) is a `get_logical_memory_view` on
    /// `arith.constant 33856` with every constant gone from the subscripts.
    ///
    /// ⭐ 33856 IS THIS FUNCTION'S RETURN VALUE, AND THE FIXTURE IS WHERE IT IS CHECKABLE:
    /// `64 * 1 + 16 * 64 + 128 * 256 = 64 + 1024 + 32768`. The three shifts are what entry 255
    /// (`applyShifts`) subtracts from the map's results to leave `(0, d0 * 3, d1 * 2)`.
    #[test]
    fn the_vendors_full_shift_is_thirty_three_thousand_eight_hundred_and_fifty_six() {
        let subscripts = vendor_subscripts_map();
        let full = calculate_full_shift(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        assert_eq!(
            full.shifts,
            vec![Shift(64), Shift(16), Shift(128)],
            "one constant offset per subscript, in the transfer's order"
        );
        assert_eq!(
            full.total,
            TotalShift::WholeSticks(64 + 16 * 64 + 128 * 256)
        );
        assert_eq!(full.total.elements(), 33856, "the fixture's arith.constant");
    }

    /// 🎯 191/384 — AND THE FIXTURE'S FOUR START ADDRESSES ARE THAT ONE TOTAL, ADDED ON.
    ///
    /// The same access hangs off four different mutable bases and every `CHECK` is `base + 33856`:
    /// `%c0` → `33856` (`:22`), a `query_map`'s `1024`/`2048` → `34880`/`35904` (`:47-48`), the
    /// toggling `iter_arg`'s `2048` → `35904` with its `subi` operand `2048 + 2 * 33856` → `69760`
    /// (`:76-77`), and an `scf.if`'s two arms → `34880`/`35904` (`:110-111`).
    ///
    /// ⭐ WHICH IS WHY THE TOTAL IS A COUNT OF ELEMENTS AND NOT AN ADDRESS. `shiftMutableAddr`
    /// (entry 323) is what adds it to whichever value the view's base turns out to be, and this
    /// function never sees that value.
    #[test]
    fn the_fixtures_four_start_addresses_are_the_same_total_added_on() {
        let subscripts = vendor_subscripts_map();
        let total = calculate_full_shift(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ))
        .total
        .elements();

        for (mutable_base, checked) in [(0, 33856), (1024, 34880), (2048, 35904)] {
            assert_eq!(mutable_base + total, checked);
        }
        // The toggle's `subi` operand pays for the shift on both halves of the swap (`:76`).
        assert_eq!(2048 + 2 * total, 69760);
    }

    /// 🎯 191/384 — 33856 IS A WHOLE NUMBER OF STICKS **AT `f16`**, AND NOT AT `i8`.
    ///
    /// ⛔ THE ELEMENT WIDTH IS THE INVARIANT'S ONLY MOVING PART. `num_elems_in_stick =
    /// bytesPerStick * 8 / element_width` is 64 at 16 bits and 128 at 8 bits (`:484-485`), and
    /// `33856 = 529 * 64` while `33856 = 264.5 * 128`. So the reference's
    /// `DT_CHECK(total_shift % num_elems_in_stick == 0)` (`:486`) PASSES on the fixture's `f16` view
    /// and would FIRE on the same subscripts over an `i8` one — the state
    /// [`TotalShift::PartialStick`] exists for, and the state entry 291 has a policy for.
    #[test]
    fn the_same_total_is_whole_sticks_at_sixteen_bits_and_partial_at_eight() {
        let order = AffineMap::identity(3);
        let subscripts = vendor_subscripts_map();

        assert_eq!(
            calculate_full_shift(&shift_inputs(&VENDOR_COEFFS, &order, &subscripts, 16)).total,
            TotalShift::WholeSticks(33856),
            "529 sticks of 64 elements"
        );
        assert_eq!(
            ElementsPerStick::of::<Dd2>(NonZeroU32::new(16).unwrap())
                .unwrap()
                .elements(),
            Elements(64)
        );

        assert_eq!(
            calculate_full_shift(&shift_inputs(&VENDOR_COEFFS, &order, &subscripts, 8)).total,
            TotalShift::PartialStick(33856),
            "264 sticks and half of another"
        );
        assert_eq!(
            ElementsPerStick::of::<Dd2>(NonZeroU32::new(8).unwrap())
                .unwrap()
                .elements(),
            Elements(128)
        );
    }

    /// 🎯 191/384 — AND A NEGATIVE TOTAL IS STILL MEASURED IN WHOLE STICKS.
    ///
    /// ⛔ `%` KEEPS THE SIGN OF ITS DIVIDEND IN BOTH LANGUAGES, so `-64 % 64 == 0` needs no
    /// `unsigned_abs` to pass — but the shifts really do go negative (`offsetShifts(shifts, ad,
    /// -num_elems_in_stick)`, `:427` and `:457`, subtracts a whole stick from a chosen dimension),
    /// and a reader should not have to work out which way Rust rounds to know that.
    #[test]
    fn a_negative_total_is_whole_sticks_too() {
        let per_stick = ElementsPerStick::of::<Dd2>(NonZeroU32::new(16).unwrap()).unwrap();
        assert_eq!(per_stick.total(-64), TotalShift::WholeSticks(-64));
        assert_eq!(per_stick.total(-33856), TotalShift::WholeSticks(-33856));
        assert_eq!(per_stick.total(-1), TotalShift::PartialStick(-1));
    }

    /// 🎯 191/384 — A NON-IDENTITY TRANSFER ORDER PAIRS EACH SHIFT WITH THE WRONG DIMENSION'S
    /// STRIDE, AND THIS IS THE REFERENCE'S ANSWER.
    ///
    /// ⛔⛔ NOT A PORTING DEFECT — THE REFERENCE'S OWN INDEX-SPACE CONFUSION, EXECUTED. The loop
    /// walks `transfer_order.compose(subscripts_map)` and multiplies result `i` by
    /// `layout_coeffs[i]` (`:473-480`), so reversing the order reverses which subscript each STRIDE
    /// is charged for: `128 * 1 + 16 * 64 + 64 * 256 = 17536`, not 33856, for an access that reads
    /// exactly the same elements. See [`SubscriptResult`] for the three spaces involved and for
    /// `calculatePartialShift`'s written-and-never-read `dim_order` (`:511-517`).
    ///
    /// ⭐ AND IT IS UNREACHABLE IN PRACTICE, WHICH IS WHY IT SURVIVED: every `load_order` and
    /// `store_order` in the vendor's own fixture is `(d0, d1, d2) -> (d0, d1, d2)`, and so is every
    /// one this crate emits ([`AffineMap::identity`]).
    #[test]
    fn a_reversed_transfer_order_charges_each_offset_to_another_dimensions_stride() {
        let reversed = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![AffineExpr::dim(2), AffineExpr::dim(1), AffineExpr::dim(0)],
        };
        let subscripts = vendor_subscripts_map();
        let full = calculate_full_shift(&shift_inputs(&VENDOR_COEFFS, &reversed, &subscripts, 16));

        assert_eq!(
            full.shifts,
            vec![Shift(128), Shift(16), Shift(64)],
            "the subscripts, visited in the transfer's order"
        );
        assert_eq!(
            full.total,
            TotalShift::WholeSticks(128 + 16 * 64 + 64 * 256),
            "17536 — the same elements, a different address"
        );
    }

    /// 🎯 191/384 — A SUBSCRIPT WITH NOTHING CONSTANT IN IT SHIFTS NOTHING, AND STILL GETS AN ENTRY.
    ///
    /// ⭐ ONE SHIFT PER RESULT, ALWAYS. `shifts.push_back(constant_offset)` runs on every iteration
    /// (`:479`), so the list entry 255 zips against the map's results cannot go out of step —
    /// `applyShifts` subtracts a zero and leaves that subscript alone (`:628`).
    #[test]
    fn a_subscript_with_no_constant_offset_shifts_nothing() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(3),
                AffineExpr::dim(1).times(2),
            ],
        };
        let full = calculate_full_shift(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        assert_eq!(full.shifts, vec![Shift(0), Shift(0), Shift(0)]);
        assert_eq!(full.total, TotalShift::WholeSticks(0));
    }

    /// 🎯 191/384 — AND THE CONSTANT COLUMN IS NOT WHATEVER LITERAL THE SUBSCRIPT MENTIONS.
    ///
    /// ⛔⛔ THE FLATTENER IS THE FUNCTION. `(d0 + 5) floordiv 8` has a 5 in it and shifts ZERO: the
    /// 5 is inside the quotient, and moving 5 elements into the start address would move it eight
    /// times too far. `(d0 * 8) mod 4` is identically zero and shifts zero. `d0 * 3 + 16` shifts 16.
    /// Pattern-matching an `Add` against a literal gets the first two wrong;
    /// [`AffineExpr::flatten`](crate::islands::dataflow_ir::ty::AffineExpr::flatten) — MLIR's
    /// `getFlattenedAffineExpr` (`:476-477`) — gets all three right.
    #[test]
    fn the_constant_column_is_not_the_literal_the_subscript_mentions() {
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).plus(AffineExpr::Const(5)).floordiv(8),
                AffineExpr::dim(0).times(8).modulo(4),
                AffineExpr::dim(0).times(3).plus(AffineExpr::Const(16)),
            ],
        };
        let full = calculate_full_shift(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        assert_eq!(
            full.shifts,
            vec![Shift(0), Shift(0), Shift(16)],
            "the 5 stays inside the floordiv and the mod has no constant at all"
        );
        assert_eq!(
            full.total,
            TotalShift::WholeSticks(16 * 256),
            "only the third subscript moves, and it moves by its own stride: 64 sticks"
        );
    }

    /// 🎯 191/384 — THE LAYOUT COEFFICIENTS' TRAILING CONSTANT TERM IS NOT READ.
    ///
    /// ⭐ `layout_coeffs.size() == operands.size() + 1` (`VectorChainToSentientPT/LoweringXRF.cpp:50`)
    /// — one stride per view dimension PLUS the linearised map's own constant. `layout_coeffs[i]`
    /// never reaches it and neither does the port's `zip`, so a view whose `layout_map` ends in
    /// `+ 4096` shifts by exactly as much as one that does not.
    #[test]
    fn the_trailing_constant_term_of_the_layout_coefficients_is_not_read() {
        let order = AffineMap::identity(3);
        let subscripts = vendor_subscripts_map();
        let with_offset = [
            LayoutCoeff(1),
            LayoutCoeff(64),
            LayoutCoeff(256),
            LayoutCoeff(4096),
        ];
        assert_eq!(
            calculate_full_shift(&shift_inputs(&with_offset, &order, &subscripts, 16)),
            calculate_full_shift(&shift_inputs(&VENDOR_COEFFS, &order, &subscripts, 16))
        );
    }

    /// 🎯 191/384 — WHERE THE REFERENCE READS PAST THE END OF THE COEFFICIENTS, THE PORT STOPS.
    ///
    /// ⛔ `layout_coeffs[i]` ON A LIST SHORTER THAN THE RESULT COUNT IS UNDEFINED BEHAVIOUR, and
    /// `calculateFullShift` reads it before anything has checked it. The file has two checks and
    /// neither covers this call: entry 192's `DT_CHECK(layout_coeffs.size() > num_dims)` (`:570`)
    /// compares against the ITERATOR count rather than the result count, and entry 255's
    /// `DT_CHECK(layout_coeffs.size() >= shifts.size())` (`:636`) runs one function LATER, on the
    /// list this one has already walked. `zip` ends the loop instead, which is the same answer for
    /// every well-formed access and a defined one otherwise.
    #[test]
    fn a_short_coefficient_list_ends_the_walk_instead_of_running_off_it() {
        let subscripts = vendor_subscripts_map();
        let full = calculate_full_shift(&shift_inputs(
            &[LayoutCoeff(1), LayoutCoeff(64)],
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));
        assert_eq!(full.shifts, vec![Shift(64), Shift(16)]);
        assert_eq!(full.total, TotalShift::WholeSticks(64 + 16 * 64));
    }

    /// 🎯 191/384 — THE FOUR GETTERS COME OFF AN `AccessDetailsAffine` ONLY ONCE IT HAS BEEN
    /// CONSTRUCTED.
    ///
    /// ⛔ THE UNSET STATES ARE THE C++'S OWN, AND THEY ARE THE UNDEFINED BEHAVIOUR THE SEAM RETIRES.
    /// A freshly constructed `AccessDetailsAffine` has a null `subscripts_map_` — which
    /// `.compose()` dereferences — and `element_width_ == 0`, which `bytesPerStick * 8 /
    /// element_width` divides by. `constructAccessDetails` sets both before this pass runs
    /// (`AccessDetails.cpp:303-307`), so [`ShiftInputs::of`] is where "was it constructed" is asked
    /// once rather than in every function below.
    #[test]
    fn the_shift_inputs_are_absent_until_the_access_details_are_constructed() {
        let op = fixture_load();
        let mut ad = AccessDetailsAffine::new(&op, DfirUnit::L3lu);
        assert_eq!(ShiftInputs::of::<Dd2>(&ad), None, "no subscripts map yet");

        ad.set_subscripts_map(vendor_subscripts_map());
        assert_eq!(
            ShiftInputs::of::<Dd2>(&ad),
            None,
            "element_width_ is still the constructor's 0"
        );

        ad.base.set_element_width(Bits(16));
        ad.base.set_layout_coeffs(&VENDOR_COEFFS);
        ad.base.set_transfer_order(AffineMap::identity(3));
        let inputs = ShiftInputs::of::<Dd2>(&ad).expect("both facts are set now");
        assert_eq!(inputs.elements_per_stick.elements(), Elements(64));
        assert_eq!(
            calculate_full_shift(&inputs).total,
            TotalShift::WholeSticks(33856),
            "the fixture's own answer, read off the access details"
        );
    }

    /// THE FIXTURE'S LAYOUT COEFFICIENTS — `(d0, d1, d2) -> (d2 * 256 + d1 * 64 + d0)` linearised,
    /// one stride per dimension in dimension order plus the map's constant term (0 here).
    const VENDOR_COEFFS: [LayoutCoeff; 4] = [
        LayoutCoeff(1),
        LayoutCoeff(64),
        LayoutCoeff(256),
        LayoutCoeff(0),
    ];

    /// `(d0, d1) -> (64, d0 * 3 + 16, d1 * 2 + 128)` — the fixture's `[%c64, %arg1 * 3 + %c16,
    /// %arg2 * 2 + %c128]` as `constructIndices` leaves it: the constant operands folded into the
    /// map, the two loop iterators as its dimensions (`mutable_start_addr_shift_full.mlir:151`).
    fn vendor_subscripts_map() -> AffineMap {
        AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(64),
                AffineExpr::dim(0).times(3).plus(AffineExpr::Const(16)),
                AffineExpr::dim(1).times(2).plus(AffineExpr::Const(128)),
            ],
        }
    }

    /// The fixture's access, with every input the shift arithmetic reads open to the caller.
    fn shift_inputs<'a>(
        layout_coeffs: &'a [LayoutCoeff],
        transfer_order: &'a AffineMap,
        subscripts_map: &'a AffineMap,
        element_width: u32,
    ) -> ShiftInputs<'a> {
        ShiftInputs {
            layout_coeffs,
            transfer_order,
            subscripts_map,
            elements_per_stick: ElementsPerStick::of::<Dd2>(
                NonZeroU32::new(element_width).expect("a width the fixture states"),
            )
            .expect("an element narrower than a stick"),
            element_width: NonZeroU32::new(element_width).expect("a width the fixture states"),
            const_offset: 0,
        }
    }

    /// The fixture's `agen.vector_load %src_mem_view[%c64, %arg1 * 3 + %c16, %arg2 * 2 + %c128] :
    /// memref<?x64x4xf16>, vector<64xf16>` (`mutable_start_addr_shift_full.mlir:151`) — the op an
    /// `AccessDetailsAffine` hangs off. Nothing in entry 191 reads it; `AccessDetailsAffine::new`
    /// takes one because the reference's constructor does.
    fn fixture_load() -> agen::Op {
        agen::Op::VectorLoad {
            dbg_name: None,
            result: Val(2),
            view: Val(1),
            indices: vec![Index::Const(64), Index::Val(Val(3)), Index::Val(Val(4))],
            view_ty: MemRef {
                shape: vec![8, 64, 4],
                elem: ElemType::F16,
            },
            ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            multicast_info: None,
        }
    }

    /// 🎯 192/384 — THE VENDOR'S ACCESS IS RANKED **DIMENSION 2, DIMENSION 1, DIMENSION 0**.
    ///
    /// Offsets `(64, 16, 128)` against strides `(1, 64, 256)` weigh `(64, 1024, 32768)`, so the
    /// order is the reverse of the transfer's: the outermost stride carries the most address per
    /// index step and its offset is the largest as well.
    ///
    /// ⭐ ONE ENTRY PER RESULT, and each keeps the POSITION it had before the sort — that is what
    /// `dim_` is for, and what `calculatePartialShift` indexes `shifts` and `layout_coeffs` by
    /// (`:533-546`).
    #[test]
    fn the_vendors_access_is_ranked_outermost_stride_first() {
        let subscripts = vendor_subscripts_map();
        let weights = calculate_dim_weights(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        assert_eq!(
            weights,
            vec![
                DimWeight {
                    dim: SubscriptResult(2),
                    offset: Shift(128),
                    weight: Weight(128 * 256),
                },
                DimWeight {
                    dim: SubscriptResult(1),
                    offset: Shift(16),
                    weight: Weight(16 * 64),
                },
                DimWeight {
                    dim: SubscriptResult(0),
                    offset: Shift(64),
                    weight: Weight(64),
                },
            ]
        );
    }

    /// 🎯 192/384 — AND THAT ORDER IS WHAT THE PARTIAL FIXTURE'S `+ 14`, `+ 78` AND `14976` ARE MADE
    /// OF.
    ///
    /// `mutable_start_addr_shift_partial.mlir` runs the same access under
    /// `-dcc-mutable-start-addr-shifting-max-immutable-size=240000`, so the budget is
    /// `240000 / 16 - 2048 = 12952` elements. Walking THESE weights highest first and spending
    /// `curr_space / layout_coeffs[dim]` of each offset (`:534-546`) reproduces the `CHECK` exactly:
    /// `[64, %arg1 * 3 + 14, %arg2 * 2 + 78]` and `arith.constant 14976` (`:29`, `:23`).
    ///
    /// ⛔ THE WALK IS ENTRY 291'S AND IS NOT PORTED HERE — it is written out in the test because it
    /// is the only place the ORDER is externally checkable. Reverse the two leading entries and the
    /// first dimension takes `12952 / 64 = 202` capped at its 16, and every number below changes.
    #[test]
    fn the_ranking_is_what_reproduces_the_partial_fixtures_own_numbers() {
        let subscripts = vendor_subscripts_map();
        let weights = calculate_dim_weights(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        // `int64_t curr_immutable_space = immutable_space;` (`:526`) — 240000 bits at 16 bits an
        // element, less the view's own `%c2048` start address.
        let mut curr_space = 240_000 / 16 - 2048;
        let mut shifts = vec![0; weights.len()];
        for dim_weight in &weights {
            let stride = VENDOR_COEFFS[dim_weight.dim.index()].0;
            // `:534-536` THEN `:541-542`, IN THAT ORDER: the skip tests the quotient and the clamp to
            // the dimension's own offset comes after, which is what makes a NEGATIVE offset a
            // negative shift rather than a skipped dimension.
            let mut constant_shift = curr_space / stride;
            if constant_shift <= 0 {
                continue;
            }
            if constant_shift > dim_weight.offset.0 {
                constant_shift = dim_weight.offset.0;
            }
            curr_space -= constant_shift * stride;
            shifts[dim_weight.dim.index()] = constant_shift;
        }
        assert_eq!(shifts, vec![24, 2, 50], "50 of 128, then 2 of 16, then 24");

        // `total_shift -= remainder; if (remainder != 0) offsetShifts(shifts, ad, -remainder);`
        // (`:554-556`) — the remainder goes back to the innermost dimension, which is dimension 0
        // for an identity transfer order (`:596-598`).
        let total_shift = 240_000 / 16 - 2048 - curr_space;
        let remainder = total_shift % 64;
        shifts[0] -= remainder;

        assert_eq!(
            shifts,
            vec![0, 2, 50],
            "dimension 0's 24 is given straight back, so its subscript stays at 64"
        );
        assert_eq!(
            [64 - shifts[0], 16 - shifts[1], 128 - shifts[2]],
            [64, 14, 78],
            "the fixture's own subscripts (`:29`)"
        );
        assert_eq!(
            2048 + total_shift - remainder,
            14976,
            "the fixture's own start address (`:23`)"
        );
    }

    /// 🎯 192/384 — THE WEIGHT IS THE PRODUCT, AND THE COMMENT'S QUOTIENT WOULD RANK THE OTHER WAY.
    ///
    /// ⛔⛔ THE TWO DISAGREE ON A REAL ACCESS. A dimension of stride 256 carrying an offset of 128 is
    /// worth 32768 elements of immutable address; one of stride 4096 carrying an offset of 1 is worth
    /// 4096. `layout_coeffs[i] * constant_offset` (`:582`) ranks the first higher; the
    /// `<layout coefficient> / <constant>` of the comment (`:500`) would rank the second higher —
    /// `4096 / 1` against `256 / 128` — and spend the budget on the dimension with almost nothing to
    /// give.
    #[test]
    fn the_weight_is_the_product_and_not_the_comments_quotient() {
        let coeffs = [LayoutCoeff(4096), LayoutCoeff(256), LayoutCoeff(0)];
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).plus(AffineExpr::Const(1)),
                AffineExpr::dim(0).plus(AffineExpr::Const(128)),
            ],
        };
        let weights = calculate_dim_weights(&shift_inputs(
            &coeffs,
            &AffineMap::identity(2),
            &subscripts,
            16,
        ));

        assert_eq!(
            weights.iter().map(|w| w.dim).collect::<Vec<_>>(),
            vec![SubscriptResult(1), SubscriptResult(0)],
            "32768 outranks 4096; a quotient would have said 4096/1 outranks 256/128"
        );
        assert_eq!(weights[0].weight, Weight(32768));
        assert_eq!(weights[1].weight, Weight(4096));
    }

    /// 🎯 192/384 — A SUBSCRIPT WITH NO CONSTANT OFFSET WEIGHS NOTHING AND SORTS LAST, AND IS STILL
    /// LISTED.
    ///
    /// ⭐ WHICH IS WHY `calculatePartialShift` CAN SKIP RATHER THAN STOP: `if (constant_shift <= 0)
    /// continue;` (`:536`) steps over a dimension there is nothing to take from and keeps going —
    /// *"Even if there is not enough space for the current dim, there may be space for another
    /// smaller dim"* (`:531-532`).
    #[test]
    fn a_dimension_with_nothing_to_give_weighs_nothing_and_sorts_last() {
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0),
                AffineExpr::dim(0).plus(AffineExpr::Const(8)),
                AffineExpr::dim(0).times(2),
            ],
        };
        let weights = calculate_dim_weights(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        assert_eq!(weights.len(), 3, "one per result, including the empty ones");
        assert_eq!(weights[0].dim, SubscriptResult(1));
        assert_eq!(weights[0].weight, Weight(8 * 64));
        assert_eq!(
            [weights[1].weight, weights[2].weight],
            [Weight(0), Weight(0)]
        );
    }

    /// 🎯 192/384 — A NEGATIVE OFFSET WEIGHS NEGATIVELY AND SORTS BELOW THE EMPTY DIMENSIONS.
    ///
    /// ⛔ `a.weight_ > b.weight_` IS A SIGNED COMPARISON, and a subscript may carry `d0 - 5`: its
    /// weight is `-5 * stride`, so it lands last, and `curr_immutable_space / layout_coeffs[dim]`
    /// then exceeds its offset and `constant_shift` becomes NEGATIVE (`:541-542`) — a shift back into
    /// the mutable address, which is the direction the reference documents at `:394-395`.
    #[test]
    fn a_negative_offset_weighs_negatively_and_sorts_last() {
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).plus(AffineExpr::Const(-5)),
                AffineExpr::dim(0),
                AffineExpr::dim(0).plus(AffineExpr::Const(1)),
            ],
        };
        let weights = calculate_dim_weights(&shift_inputs(
            &VENDOR_COEFFS,
            &AffineMap::identity(3),
            &subscripts,
            16,
        ));

        assert_eq!(
            weights.iter().map(|w| w.weight).collect::<Vec<_>>(),
            vec![Weight(256), Weight(0), Weight(-5)]
        );
        assert_eq!(weights[2].dim, SubscriptResult(0));
        assert_eq!(weights[2].offset, Shift(-5));
    }

    /// 🎯 192/384 — TIED WEIGHTS KEEP TRANSFER ORDER HERE, WHERE `llvm::sort` LEAVES THEM
    /// UNSPECIFIED.
    ///
    /// ⛔⛔ THE TIE IS REACHABLE AND IT CHANGES THE OUTPUT. `(offset 2, stride 64)` and
    /// `(offset 64, stride 2)` both weigh 128, and `calculatePartialShift` spends budget in the
    /// order it is handed (`:544`), so which of the two subscripts loses its constant depends on
    /// `std::sort`'s pivoting. This port's sort is stable, so the answer is a function of the input;
    /// the reference's is not.
    #[test]
    fn tied_weights_keep_the_transfer_order() {
        let coeffs = [LayoutCoeff(64), LayoutCoeff(2), LayoutCoeff(0)];
        let subscripts = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).plus(AffineExpr::Const(2)),
                AffineExpr::dim(0).plus(AffineExpr::Const(64)),
            ],
        };
        let weights = calculate_dim_weights(&shift_inputs(
            &coeffs,
            &AffineMap::identity(2),
            &subscripts,
            16,
        ));

        assert_eq!(
            [weights[0].weight, weights[1].weight],
            [Weight(128), Weight(128)]
        );
        assert_eq!(
            weights.iter().map(|w| w.dim).collect::<Vec<_>>(),
            vec![SubscriptResult(0), SubscriptResult(1)],
            "stable: the first result of the ORDERED map stays first"
        );
    }

    /// 🎯 191/384 + 192/384 — THE TWO LOOPS READ THE SAME CONSTANT OFFSETS, WHICH IS THE POINT OF
    /// PORTING THEM ONCE.
    ///
    /// ⭐ SORTED BACK BY `dim_`, ENTRY 192'S OFFSETS **ARE** ENTRY 191'S `shifts`. The bodies are the
    /// same five lines of flatten (`:473-478` against `:573-579`, which inserts one extra
    /// `DT_CHECK(!coeffs.empty())`), and only what they do with the result differs: 191 accumulates
    /// `offset * stride` into one total, 192 keeps each product as a ranking key.
    #[test]
    fn the_offsets_entry_192_ranks_are_the_shifts_entry_191_returns() {
        let subscripts = vendor_subscripts_map();
        let order = AffineMap::identity(3);
        let inputs = shift_inputs(&VENDOR_COEFFS, &order, &subscripts, 16);

        let mut weights = calculate_dim_weights(&inputs);
        weights.sort_by_key(|dim_weight| dim_weight.dim);
        assert_eq!(
            weights.iter().map(|w| w.offset).collect::<Vec<_>>(),
            calculate_full_shift(&inputs).shifts
        );

        // And the total is the sum of the weights, which is what makes 33856 the same number twice.
        assert_eq!(
            weights.iter().map(|w| w.weight.0).sum::<i64>(),
            calculate_full_shift(&inputs).total.elements()
        );
    }
    /// 🎯 254/384 — THE SLOT IS THE FIRST RESULT THE TRANSFER READS ALONG `d0`, NOT RESULT 0.
    ///
    /// `offsetShifts(shifts, ad, -num_elems_in_stick)` (`:427`) takes one stick of f16 back out of the
    /// innermost dimension; under the fixture's identity order that is subscript 0, and under a
    /// reversed order the same call lands on subscript 2.
    #[test]
    fn the_offset_lands_on_the_transfers_innermost_result() {
        let mut shifts = vec![Shift(64), Shift(16), Shift(128)];
        offset_shifts(&mut shifts, &AffineMap::identity(3), Shift(-64));
        assert_eq!(shifts, vec![Shift(0), Shift(16), Shift(128)]);

        let reversed = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![AffineExpr::dim(2), AffineExpr::dim(1), AffineExpr::dim(0)],
        };
        let mut shifts = vec![Shift(64), Shift(16), Shift(128)];
        offset_shifts(&mut shifts, &reversed, Shift(-64));
        assert_eq!(
            shifts,
            vec![Shift(64), Shift(16), Shift(64)],
            "the transfer reads d0 third"
        );
    }

    /// 🎯 255/384 — THE VENDOR'S FULL SHIFT: `arith.constant 33856` AND `[0, %arg1 * 3, %arg2 * 2]`.
    ///
    /// `@full_shift_zero_const_start` (`mutable_start_addr_shift_full.mlir:14-34`): every constant
    /// leaves the subscripts and the view's start address carries all 33856 elements of them.
    #[test]
    fn the_vendors_shift_empties_the_subscripts_into_the_start_address() {
        let subscripts = vendor_subscripts_map();
        let layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(256)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        let ty = MemRef {
            shape: vec![8, 64, 4],
            elem: ElemType::F16,
        };
        let mut vals = Values::default();
        let shifted = apply_shifts(
            &mut vals,
            &[Shift(64), Shift(16), Shift(128)],
            &shift_inputs(&VENDOR_COEFFS, &AffineMap::identity(3), &subscripts, 16),
            &ConstStartMemView {
                from: Val(0),
                start: 0,
                layout: &layout,
                ty: &ty,
            },
        );

        assert_eq!(
            shifted.subscripts_map,
            AffineMap {
                dims: 2,
                syms: 0,
                results: vec![
                    AffineExpr::Const(0),
                    AffineExpr::dim(0).times(3),
                    AffineExpr::dim(1).times(2),
                ],
            },
            "d0 * 3 + 16 - 16 simplifies to d0 * 3"
        );
        assert_eq!(shifted.total, TotalShift::WholeSticks(33856));
        assert_eq!(
            shifted.start_address,
            DfirOp::Arith(arith::Op::Constant {
                result: shifted.start,
                value: 33856,
            })
        );
    }
    /// 🎯 291/384 — THE VENDOR'S PARTIAL CASE: A BUDGET OF **12952** BUYS `[0, 2, 50]` AND MOVES THE
    /// IMMUTABLE START TO **14976**.
    ///
    /// `mutable_start_addr_shift_partial.mlir:23`, `:29` — the same access as the full-shift fixture
    /// under `-dcc-mutable-start-addr-shifting-max-immutable-size=240000`, so the budget is
    /// `240000 / 16 - 2048`. ⭐ AND DIMENSION 0's 24 ELEMENTS ARE THE PART-STICK REMAINDER,
    /// handed straight back: `12952 % 64 = 24`, and `2048 + 12928 = 14976`.
    #[test]
    fn the_vendors_partial_shift_spends_twelve_thousand_nine_hundred_and_twenty_eight() {
        let subscripts = vendor_subscripts_map();
        let layout = AffineMap::identity(3);
        let inputs = shift_inputs(&VENDOR_COEFFS, &layout, &subscripts, 16);
        let partial = calculate_partial_shift(&inputs, ImmutableSpace(240_000 / 16 - 2048));

        assert_eq!(
            partial.shifts,
            vec![Shift(0), Shift(2), Shift(50)],
            "50 of d2's 128 and 2 of d1's 16, with d0's 24 given back as the remainder"
        );
        assert_eq!(partial.total, TotalShift::WholeSticks(12928));
        assert_eq!(
            apply_shifts(
                &mut Values::default(),
                &partial.shifts,
                &inputs,
                &ConstStartMemView {
                    from: Val(0),
                    start: 2048,
                    layout: &AffineMap {
                        dims: 3,
                        syms: 0,
                        results: vec![
                            AffineExpr::dim(2)
                                .times(256)
                                .plus(AffineExpr::dim(1).times(64))
                                .plus(AffineExpr::dim(0)),
                        ],
                    },
                    ty: &MemRef {
                        shape: vec![8, 64, 4],
                        elem: ElemType::F16,
                    },
                },
            )
            .subscripts_map,
            AffineMap {
                dims: 2,
                syms: 0,
                results: vec![
                    AffineExpr::Const(64),
                    AffineExpr::dim(0).times(3).plus(AffineExpr::Const(14)),
                    AffineExpr::dim(1).times(2).plus(AffineExpr::Const(78)),
                ],
            },
            "the CHECK's own subscripts: `[64, %arg1 * 3 + 14, %arg2 * 2 + 78]`"
        );

        // ⛔ AND A BUDGET SMALLER THAN EVERY STRIDE BUYS NOTHING, which is where a negative one lands
        // too (`:534-539`).
        let nothing = calculate_partial_shift(&inputs, ImmutableSpace(-4096));
        assert_eq!(nothing.shifts, vec![Shift(0); 3]);
        assert_eq!(nothing.total, TotalShift::WholeSticks(0));
    }

    /// 🎯 308/384 — THE VENDOR'S `@full_shift_zero_const_start` MOVES `(64, 16, 128)` OUT AND ITS
    /// VIEW START BECOMES `arith.constant 33856`
    /// (`dcc/test/Transform/MutableStartAddrShifting/mutable_start_addr_shift_full.mlir:14-34`),
    /// which is also the `const_offset` this access's coefficient dictionary carries.
    ///
    /// ⛔ AND THE `const_offset == 0` ARM IS NOT A NO-OP ON SEN1P5: an immutable address of an ODD
    /// number of sticks takes a WHOLE STICK BACK, negative, on the first subscript the transfer order
    /// visits along `d0` — while the same view below SEN1P5 shifts nothing at all.
    #[test]
    fn the_vendors_full_shift_moves_every_constant_offset_out() {
        let subscripts = vendor_subscripts_map();
        let order = AffineMap::identity(3);
        let layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(256)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        let ty = MemRef {
            shape: vec![8, 64, 4],
            elem: ElemType::F16,
        };
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };

        // `immutable_space` is the whole EBR span in fp16 elements against a start of 0, so the
        // `>= const_offset` arm is `calculateFullShift` and every offset moves.
        let full = ShiftInputs {
            const_offset: 33856,
            ..shift_inputs(&VENDOR_COEFFS, &order, &subscripts, 16)
        };
        assert_eq!(
            calculate_shifts::<Dd2>(&full, L3Half::Load, &view),
            vec![Shift(64), Shift(16), Shift(128)],
            "`[%c64, %arg1 * 3 + %c16, %arg2 * 2 + %c128]` becomes `[0, %arg1 * 3, %arg2 * 2]`"
        );

        // 65 sticks of fp16 — odd, so SEN1P5 realigns it by minus one stick and nothing else.
        let odd = ConstStartMemView {
            start: 64 * 65,
            ..view
        };
        let zero = shift_inputs(&VENDOR_COEFFS, &order, &subscripts, 16);
        assert_eq!(zero.const_offset, 0);
        assert_eq!(
            calculate_shifts::<Sen1p5>(&zero, L3Half::Load, &odd),
            vec![Shift(-64), Shift(0), Shift(0)]
        );
        assert!(Dd2::GEN < IsaGen::Sen1p5);
        assert_eq!(
            calculate_shifts::<Dd2>(&zero, L3Half::Load, &odd),
            vec![Shift(0); 3],
            "below SEN1P5 the parity of the immutable address is not a constraint"
        );
    }

    /// 🎯 323/384 — THE VENDOR'S FULL CASE END TO END, AND THE "NOTHING TO SHIFT" ANSWER BESIDE IT.
    ///
    /// `@full_shift_zero_const_start` (`mutable_start_addr_shift_full.mlir:14-34`) reaches
    /// [`apply_shifts`] through this function; with the offset already 0 every shift is 0 and the
    /// reference returns its empty map instead.
    #[test]
    fn the_vendors_access_shifts_and_a_zero_offset_reports_no_shift() {
        let subscripts = vendor_subscripts_map();
        let layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(256)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        let ty = MemRef {
            shape: vec![8, 64, 4],
            elem: ElemType::F16,
        };
        let view = ConstStartMemView {
            from: Val(0),
            start: 0,
            layout: &layout,
            ty: &ty,
        };
        let layout = AffineMap::identity(3);
        let mut inputs = shift_inputs(&VENDOR_COEFFS, &layout, &subscripts, 16);
        inputs.const_offset = 64 + 16 * 64 + 128 * 256;

        let mut vals = Values::default();
        let shifted = shift_mutable_addr::<Dd2>(&mut vals, &inputs, L3Half::Load, &view)
            .expect("a shift the fixture reports");
        assert_eq!(shifted.total, TotalShift::WholeSticks(33856));
        assert_eq!(
            shifted.subscripts_map.results,
            vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(3),
                AffineExpr::dim(1).times(2),
            ]
        );

        inputs.const_offset = 0;
        assert_eq!(
            shift_mutable_addr::<Dd2>(&mut vals, &inputs, L3Half::Load, &view),
            None,
            "every shift 0 is `AffineMap::get(ctx)` — the pass reports no shift"
        );
    }
    /// 🎯 352/384 — ⭐⭐ IBM'S `full_shift_zero_const_start` KEY END TO END: ALL 33856 ELEMENTS OF
    /// CONSTANT OFFSET LEAVE THE SUBSCRIPTS AND ARRIVE IN THE VIEW'S START ADDRESS.
    ///
    /// `mutable_start_addr_shift_full.mlir:137-158` in, `:22-27` out. ⛔ THE ACCESS IS THE VENDOR'S
    /// TRANSPOSED, as entry 306's test is: this island derives the lane run on the LAST axis where the
    /// key's `d2 * 256 + d1 * 64 + d0` puts it on `d0`, and its `?` extent is one vector wide here.
    #[test]
    fn the_full_shift_keys_whole_constant_offset_moves_into_the_start_address() {
        let mut vals = Values::default();
        let hbm = vals.mint();
        let lx = vals.mint();
        let start = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let src_view = vals.mint();
        let dst_view = vals.mint();
        let data = vals.mint();

        let layout = AffineMap::linear(&[256, 64, 1]);
        let ty = MemRef {
            shape: vec![4, 64, 64],
            elem: ElemType::F16,
        };
        let vec_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let view = |result: Val, from: Val| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                layout: layout.clone(),
                ty: ty.clone(),
            })
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 0,
            }),
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
            view(src_view, hbm),
            view(dst_view, lx),
            // `%data = agen.vector_load %src_mem_view[%c64, %arg1 * 3 + %c16, %arg2 * 2 + %c128]`.
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: data,
                view: src_view,
                indices: vec![
                    Index::Strided(vec![(arg2, 2)], 128),
                    Index::Strided(vec![(arg1, 3)], 16),
                    Index::Const(64),
                ],
                view_ty: ty.clone(),
                ty: vec_ty,
                multicast_info: None,
            }),
            // `agen.vector_store %data, %dst_mem_view[0, 0, 0]` — the one use that gets re-pointed.
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
            op: &scope[4],
            comp: DfirUnit::L3lu,
            mem_index: MemoryOperandIndex::DirSrc,
        };
        let shift = transform_vector_load::<Dd2>(&mut vals, &candidate, &scope);
        let MutableStartAddrShift::Shifted(shifted) = shift else {
            panic!("64 * 1 + 16 * 64 + 128 * 256 fits the EBR whole: {shift:?}")
        };
        assert_eq!(shifted.total, TotalShift::WholeSticks(33856));
        // `%[[VAL_7]] = arith.constant 33856 : index`, with the view rebuilt on it under its OWN
        // result — `%[[VAL_11]]` still feeds the load below.
        let DfirOp::Arith(arith::Op::Constant {
            result: new_start,
            value,
        }) = shifted.start_address
        else {
            panic!("arm one of `updateMemViewStartAddress` is one `arith.constant`")
        };
        assert_eq!(value, 33856);
        assert_eq!(
            shifted.mem_view,
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: src_view,
                from: hbm,
                start: new_start,
                layout,
                ty: ty.clone(),
            })
        );
        // `agen.vector_load %[[VAL_11]][0, %[[VAL_9]] * 3, %[[VAL_10]] * 2] {dbgName = "", ..}`.
        let DfirOp::Agen(agen::Op::VectorLoad {
            result,
            view: loaded,
            indices,
            dbg_name,
            ..
        }) = &shifted.mem_op
        else {
            panic!("the clone is a vector load")
        };
        assert_eq!(*loaded, src_view);
        assert_eq!(dbg_name.as_deref(), Some(""), "`Agen.cpp:166`");
        assert_eq!(
            *indices,
            [
                Index::Strided(vec![(arg2, 2)], 0),
                Index::Strided(vec![(arg1, 3)], 0),
                Index::Const(0),
            ]
        );
        assert_eq!(
            shifted.replaced,
            Some((data, *result)),
            "the store reads the load, so `use_empty()` is false"
        );
        assert_eq!(shifted.erased, &scope[4]);
    }
    /// 🎯 353/384 — THE SAME 33856 ELEMENTS, OUT OF A STORE'S SUBSCRIPTS, WITH NOTHING RE-POINTED.
    ///
    /// ⛔ THE VENDOR'S OWN STORE CASE IS NOT EXPRESSIBLE HERE: `@partial_shift_zero_query`
    /// (`mutable_start_addr_shift_partial.mlir:157-180`) starts its HBM view at a
    /// `uniform.query_map`, which is [`MutableStartAddrShift::MemViewStartIsNotConstant`] in this
    /// island. So this is its access on a constant start, and the total is entry 352's.
    #[test]
    fn a_stores_offsets_shift_the_same_way_and_re_point_nothing() {
        let mut vals = Values::default();
        let hbm = vals.mint();
        let lx = vals.mint();
        let start = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let src_view = vals.mint();
        let dst_view = vals.mint();
        let data = vals.mint();

        let layout = AffineMap::linear(&[256, 64, 1]);
        let ty = MemRef {
            shape: vec![4, 64, 64],
            elem: ElemType::F16,
        };
        let vec_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let view = |result: Val, from: Val| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                layout: layout.clone(),
                ty: ty.clone(),
            })
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 0,
            }),
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
            view(src_view, lx),
            view(dst_view, hbm),
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: data,
                view: src_view,
                indices: vec![Index::Const(0); 3],
                view_ty: ty.clone(),
                ty: vec_ty,
                multicast_info: None,
            }),
            // `agen.vector_store %data, %dst_mem_view[%c64, %arg1 * 3 + %c16, %arg2 * 2 + %c128]`.
            DfirOp::Agen(agen::Op::VectorStore {
                dbg_name: None,
                value: data,
                view: dst_view,
                indices: vec![
                    Index::Strided(vec![(arg2, 2)], 128),
                    Index::Strided(vec![(arg1, 3)], 16),
                    Index::Const(64),
                ],
                view_ty: ty.clone(),
                ty: vec_ty,
            }),
        ];

        let candidate = MasCandidate {
            op: &scope[5],
            // The collection's `else` arm, and it is the source index — see this entry's banner.
            comp: DfirUnit::L3su,
            mem_index: MemoryOperandIndex::DirSrc,
        };
        let shift = transform_vector_store::<Dd2>(&mut vals, &candidate, &scope);
        let MutableStartAddrShift::Shifted(shifted) = shift else {
            panic!("the store's offsets are entry 352's: {shift:?}")
        };
        assert_eq!(shifted.total, TotalShift::WholeSticks(33856));
        assert_eq!(shifted.replaced, None, "a store binds nothing to re-point");
        assert_eq!(shifted.erased, &scope[5]);
        let DfirOp::Arith(arith::Op::Constant { value, .. }) = shifted.start_address else {
            panic!("arm one of `updateMemViewStartAddress` is one `arith.constant`")
        };
        assert_eq!(value, 33856);
        let DfirOp::Agen(agen::Op::VectorStore {
            value: stored,
            view: written,
            indices,
            dbg_name,
            ..
        }) = &shifted.mem_op
        else {
            panic!("the clone is a vector store")
        };
        assert_eq!(
            (*stored, *written),
            (data, dst_view),
            "`getValueToStore()` and the view, both carried over"
        );
        assert_eq!(dbg_name.as_deref(), Some(""), "`Agen.cpp:244`");
        assert_eq!(
            *indices,
            [
                Index::Strided(vec![(arg2, 2)], 0),
                Index::Strided(vec![(arg1, 3)], 0),
                Index::Const(0),
            ]
        );
    }
    /// THE VENDOR'S COMPOSITE FROM `mutable_start_addr_shift_partial.mlir:181-215`
    /// (`@partial_shift_toggle_start`), on a CONSTANT start: its `arith.subi` toggle is
    /// [`MutableStartAddrShift::MemViewStartIsNotConstant`] here, so the shift is entry 352's full one
    /// rather than the vendor's partial `[64, .. * 3 + 14, .. * 2 + 78]`.
    ///
    /// ⚠️ THE ACCESS IS TRANSPOSED into the island's row-major convention, as entry 306's fixtures are.
    fn toggle_start_transfer(src: Val, dst: Val, arg1: Val, arg3: Val, load_iv: Val) -> agen::Op {
        let ty = MemRef {
            shape: vec![4, 64, 64],
            elem: ElemType::F16,
        };
        // `load_set`/`store_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 == 0,
        // d2 == 0)>`, transposed: the 64 lanes are the INNERMOST subscript here.
        let transfer_set = IntegerSet {
            dims: 3,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(1),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(2),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(2).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
            ],
        };
        // `time_set = affine_set<(d0, d1, d2)[s0] : (d<i> >= 0, -d<i> + s0 - 1 >= 0)>`.
        let spanning = |dim: u32| {
            [
                Constraint {
                    expr: AffineExpr::dim(dim),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(dim)
                        .times(-1)
                        .plus(AffineExpr::sym(0))
                        .plus(AffineExpr::Const(-1)),
                    is_equality: false,
                },
            ]
        };
        agen::Op::CompositeLoadAndStore(Box::new(agen::CompositeTransfer {
            // `src:%src_mem_view[0, 0, 0]`.
            src,
            src_indices: vec![Index::Const(0); 3],
            src_ty: ty.clone(),
            // `dst:%dst_mem_view[%c64, %arg1 * 3 + %c16, %arg3 * 2 + 128]`.
            dst,
            dst_indices: vec![
                Index::Strided(vec![(arg3, 2)], 128),
                Index::Strided(vec![(arg1, 3)], 16),
                Index::Const(64),
            ],
            dst_ty: ty,
            load_iv,
            load_iv_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            load_set: transfer_set.clone(),
            load_order: AffineMap::identity(3),
            store_set: transfer_set,
            store_order: AffineMap::identity(3),
            time_symbols: vec![arg1],
            time_set: IntegerSet {
                dims: 3,
                symbols: 1,
                constraints: (0..3).flat_map(spanning).collect(),
            },
            time_order: AffineMap::identity(3),
            load_time_addr_map: AffineMap::identity(3),
            store_time_addr_map: AffineMap::identity(3),
            dir: None,
            multicast_info: None,
            dbg_name: None,
            body: vec![DfirOp::Agen(agen::Op::Yield)],
        }))
    }

    /// 🎯 354/384 — ONE SIDE SHIFTS AND THE OTHER IS COPIED THROUGH, on the vendor's own composite.
    #[test]
    fn a_composites_shifted_side_is_the_one_the_memory_index_names() {
        let mut vals = Values::default();
        let hbm = vals.mint();
        let lx = vals.mint();
        let start = vals.mint();
        let arg1 = vals.mint();
        let arg3 = vals.mint();
        let src_view = vals.mint();
        let dst_view = vals.mint();
        let load_iv = vals.mint();

        let layout = AffineMap::linear(&[256, 64, 1]);
        let ty = MemRef {
            shape: vec![4, 64, 64],
            elem: ElemType::F16,
        };
        let view = |result: Val, from: Val| {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                layout: layout.clone(),
                ty: ty.clone(),
            })
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 0,
            }),
            DfirOp::Affine(affine::Op::For {
                iv: arg1,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: vec![DfirOp::Affine(affine::Op::For {
                    iv: arg3,
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Const(8),
                    carried: Vec::new(),
                    body: Vec::new(),
                    dbg_name: None,
                })],
                dbg_name: None,
            }),
            view(src_view, lx),
            view(dst_view, hbm),
            DfirOp::Agen(toggle_start_transfer(
                src_view, dst_view, arg1, arg3, load_iv,
            )),
        ];
        let candidate = MasCandidate {
            op: &scope[4],
            comp: DfirUnit::L3su,
            // The HBM operand is the DESTINATION — `runOnOperation:161-166`.
            mem_index: MemoryOperandIndex::DirDst,
        };
        let shift = transform_comp_load_and_store::<Dd2>(&mut vals, &candidate, &scope);
        let MutableStartAddrShift::Shifted(shifted) = shift else {
            panic!("the destination's whole constant offset fits the immutable range: {shift:?}")
        };
        assert_eq!(shifted.total, TotalShift::WholeSticks(33856));
        assert_eq!(shifted.replaced, None, "a composite binds nothing");
        assert_eq!(shifted.erased, &scope[4]);
        let DfirOp::Agen(agen::Op::CompositeLoadAndStore(rebuilt)) = &shifted.mem_op else {
            panic!("the clone is a composite load and store")
        };
        let DfirOp::Agen(agen::Op::CompositeLoadAndStore(original)) = &scope[4] else {
            unreachable!("the fixture is one")
        };
        assert_eq!(
            (rebuilt.src, rebuilt.dst),
            (src_view, dst_view),
            "`getSrcMemRef()` and `getDstMemRef()`, both unchanged"
        );
        assert_eq!(
            rebuilt.src_indices, original.src_indices,
            "the load side keeps its own map and operands"
        );
        assert_eq!(
            rebuilt.dst_indices,
            [
                Index::Strided(vec![(arg3, 2)], 0),
                Index::Strided(vec![(arg1, 3)], 0),
                Index::Const(0),
            ]
        );
        assert_eq!(rebuilt.time_set, original.time_set, "`getTimeSet()`");
    }
    /// 🎯 355/384 — THE GATHER'S DIRECT SOURCE SHIFTS; ITS INDIRECT DESTINATION AND THAT SIDE'S
    /// SUBSCRIPT DO NOT, AND ASKING FOR THE INDIRECT'S OWN SIDE IS REFUSED.
    ///
    /// The vendor's `@full_shift_conditional_start`
    /// (`mutable_start_addr_shift_partial.mlir:218-259`) on a CONSTANT start of 2048 — its `scf.if`
    /// start is [`MutableStartAddrShift::MemViewStartIsNotConstant`] here. ⛔ AND THE SHIFT IS THE
    /// WHOLE 33856 WHERE ITS CHECK SHOWS `[64, .. * 3 + 14, .. * 2 + 78]`: that run passes
    /// `--dcc-mutable-start-addr-shifting-max-immutable-size=240000`, i.e. 15000 `f16` elements, and
    /// [`MAX_IMMUTABLE_SIZE`] is [`None`] in this crate, so the whole offset fits.
    #[test]
    fn only_the_side_without_the_indirect_can_shift() {
        let mut vals = Values::default();
        let hbm = vals.mint();
        let lx = vals.mint();
        let ibr = vals.mint();
        let zero = vals.mint();
        let start = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let src_view = vals.mint();
        let ibr_view = vals.mint();
        let dst_view = vals.mint();
        let load_iv = vals.mint();

        let layout = AffineMap::linear(&[256, 64, 1]);
        let ty = MemRef {
            shape: vec![4, 64, 64],
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
        let ibr_ty = MemRef {
            shape: vec![32],
            elem: ElemType::Int(32),
        };
        // `load_set`/`store_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 == 0,
        // d2 == 0)>`, transposed with the access.
        let transfer_set = IntegerSet {
            dims: 3,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(1),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(2),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(2).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
            ],
        };
        let spanning = |dim: u32| {
            [
                Constraint {
                    expr: AffineExpr::dim(dim),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(dim)
                        .times(-1)
                        .plus(AffineExpr::sym(0))
                        .plus(AffineExpr::Const(-1)),
                    is_equality: false,
                },
            ]
        };
        let transfer =
            agen::Op::CompositeIndirectLoadAndStore(Box::new(agen::CompositeIndirectTransfer {
                indirect_src: None,
                // `direct_src:%src_mem_view[%c64, %arg1 * 3 + %c16, %arg2 * 2 + 128]`.
                direct_src: src_view,
                direct_src_indices: vec![
                    Index::Strided(vec![(arg2, 2)], 128),
                    Index::Strided(vec![(arg1, 3)], 16),
                    Index::Const(64),
                ],
                direct_src_ty: ty.clone(),
                // `indirect_dst:%ibr_mem_view[%arg2]`.
                indirect_dst: Some(agen::IndirectAccess {
                    view: ibr_view,
                    indices: vec![Index::Val(arg2)],
                    ty: ibr_ty.clone(),
                }),
                // `direct_dst:%dst_mem_view[0, 0, %arg1]`.
                direct_dst: dst_view,
                direct_dst_indices: vec![Index::Val(arg1), Index::Const(0), Index::Const(0)],
                direct_dst_ty: ty.clone(),
                load_iv,
                load_iv_ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
                load_set: transfer_set.clone(),
                load_order: AffineMap::identity(3),
                store_set: transfer_set,
                store_order: AffineMap::identity(3),
                time_symbols: vec![arg1],
                time_set: IntegerSet {
                    dims: 3,
                    symbols: 1,
                    constraints: (0..3).flat_map(spanning).collect(),
                },
                time_order: AffineMap::identity(3),
                load_indirect_time_addr_map: None,
                load_direct_time_addr_map: AffineMap::identity(3),
                // `store_indirect_time_addr_map = affine_map<(d0, d1, d2) -> (0)>`.
                store_indirect_time_addr_map: Some(AffineMap {
                    dims: 3,
                    syms: 0,
                    results: vec![AffineExpr::Const(0)],
                }),
                store_direct_time_addr_map: AffineMap::identity(3),
                multicast_info: None,
                dbg_name: None,
                body: vec![DfirOp::Agen(agen::Op::Yield)],
            }));
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: zero,
                value: 0,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: start,
                value: 2048,
            }),
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
            view(src_view, hbm, start),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: ibr_view,
                from: ibr,
                start: zero,
                layout: AffineMap::identity(1),
                ty: ibr_ty,
            }),
            view(dst_view, lx, zero),
            DfirOp::Agen(transfer),
        ];
        let candidate = |mem_index| MasCandidate {
            op: &scope[6],
            comp: DfirUnit::L3lu,
            mem_index,
        };

        // ⛔ THE DESTINATION IS THE SCATTER'S OWN SIDE — `runOnOperation` never seats it, and the
        // early return is what would stop it if it did.
        assert_eq!(
            transform_comp_ind_load_and_store::<Dd2>(
                &mut vals,
                &candidate(MemoryOperandIndex::DirDst),
                &scope
            ),
            MutableStartAddrShift::IndirectIsOnTheShiftedSide(MemoryOperandIndex::DirDst)
        );

        let shift = transform_comp_ind_load_and_store::<Dd2>(
            &mut vals,
            &candidate(MemoryOperandIndex::DirSrc),
            &scope,
        );
        let MutableStartAddrShift::Shifted(shifted) = shift else {
            panic!("the direct source carries no indirect: {shift:?}")
        };
        assert_eq!(shifted.total, TotalShift::WholeSticks(33856));
        assert_eq!(shifted.replaced, None);
        assert_eq!(shifted.erased, &scope[6]);
        let DfirOp::Arith(arith::Op::Constant { value, .. }) = shifted.start_address else {
            panic!("the new start address is one `arith.constant`")
        };
        assert_eq!(
            value,
            2048 + 33856,
            "the start it had plus what moved into it"
        );
        let DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(rebuilt)) = &shifted.mem_op else {
            panic!("the clone is a composite indirect load and store")
        };
        let DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(original)) = &scope[6] else {
            unreachable!("the fixture is one")
        };
        assert_eq!(
            rebuilt.direct_src_indices,
            [
                Index::Strided(vec![(arg2, 2)], 0),
                Index::Strided(vec![(arg1, 3)], 0),
                Index::Const(0),
            ]
        );
        assert_eq!(
            (
                rebuilt.indirect_src.clone(),
                rebuilt.indirect_dst.clone(),
                rebuilt.direct_dst_indices.clone()
            ),
            (
                None,
                original.indirect_dst.clone(),
                original.direct_dst_indices.clone()
            ),
            "the scatter's side is copied through, subscript and all"
        );
    }

    /// 🎯 372/384 — THE WALK COLLECTS ONE CANDIDATE OUT OF FOUR VIEWS AND TWO UNITS.
    ///
    /// The LX view is not HBM, the symbol-started HBM view is excluded by `isCandidateMemView`, and
    /// the `lxlu` unit fails the `is_any_of(comp, L3LU, L3SU)` gate — so only the L3LU HBM load is a
    /// candidate, reached as that view's single user, and it dispatches to `transformVectorLoad`.
    #[test]
    fn the_walk_collects_the_one_hbm_view_in_an_l3_unit() {
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
        let (hbm, lx, l3lu, lxlu) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
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
        let start = vals.mint();
        let start_op = DfirOp::Arith(arith::Op::Constant {
            result: start,
            value: 0,
        });
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

        let shifting = run_on_operation::<Dd2>(&program, &mut vals);
        let MutableStartAddrShifting::Shifted(shifted) = shifting else {
            panic!("every candidate has one user and one direct memory operand: {shifting:?}")
        };
        assert_eq!(shifted.len(), 1);
        let MsasShift {
            candidate,
            dispatch,
        } = &shifted[0];
        assert_eq!(candidate.comp, DfirUnit::L3lu);
        assert_eq!(candidate.mem_index, MemoryOperandIndex::DirSrc);
        assert!(
            matches!(candidate.op, DfirOp::Agen(agen::Op::VectorLoad { view, .. }) if *view == hbm_view),
            "the load of the HBM view, reached as its one user"
        );
        assert!(
            matches!(dispatch, MsasDispatch::VectorLoad(_)),
            "{dispatch:?}"
        );
    }
}

// ═══════════════════════════════════════ 372/384 ═══════════════════════════════════════

/// WHICH OF THE FOUR TRANSFORMS RAN ON ONE CANDIDATE, AND WHAT IT ANSWERED — `:186-197`.
///
/// ⛔ `llvm_unreachable("Unexpected candidate operation.")` IS [`Self::UnexpectedCandidateOperation`]
/// and it is reachable: the collecting walk's `else` arm admits ANY user of an HBM view (`:170`), so a
/// `dataflow.send` reading one lands here.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum MsasDispatch<'s> {
    /// `isa<agen::VectorLoadOp>` → `transformVectorLoad`.
    VectorLoad(MutableStartAddrShift<'s>),
    /// `isa<agen::VectorStoreOp>` → `transformVectorStore`.
    VectorStore(MutableStartAddrShift<'s>),
    /// `isa<agen::CompositeLoadAndStoreOp>` → `transformCompLoadAndStore`.
    CompLoadAndStore(MutableStartAddrShift<'s>),
    /// `isa<agen::CompositeIndirectLoadAndStoreOp>` → `transformCompIndLoadAndStore`.
    CompIndLoadAndStore(MutableStartAddrShift<'s>),
    /// `llvm_unreachable("Unexpected candidate operation.")`.
    UnexpectedCandidateOperation,
}

/// ONE COLLECTED CANDIDATE AND WHAT THE DISPATCH LOOP DID TO IT.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct MsasShift<'s> {
    /// `candidates.emplace_back(mem_op, unit, comp, mem_index)` (`:181`).
    pub candidate: MasCandidate<'s>,
    /// The transform's own answer.
    pub dispatch: MsasDispatch<'s>,
}

/// WHAT THE PASS DID TO A PROGRAM — or the `DT_CHECK` in the collecting walk that stopped it.
///
/// ⛔ THE THREE REFUSALS END THE WHOLE PASS. All three are `DT_CHECK`s inside the module walk
/// (`:157`, `:175`, `:179`) and the walk finishes before the first transform runs, so an abort there
/// means NOTHING was shifted, in any unit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum MutableStartAddrShifting<'s> {
    /// One entry per candidate, in the order the walk collected them.
    Shifted(Vec<MsasShift<'s>>),
    /// `DT_CHECK_MSG(mem_view->hasOneUse(), "Expecting L3 memory views to be used in one memory
    /// operand.")`.
    MemViewIsNotSingleUsed(Val),
    /// `DT_CHECK_MSG(mem_index != kMax, "Invalid HBM memory operand.")`.
    InvalidHbmMemoryOperand(Val),
    /// `DT_CHECK_MSG(res.second, "Data transfers should only have one HBM memory operand.")`.
    SecondHbmMemoryOperand(Val),
}

/// Replaces: e372_runOnOperation
///
/// **372/384** `MutableStartAddrShiftingPass::runOnOperation` —
/// `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130` (69L): collect every HBM view in an
/// L3 unit, then run one transform per memory operation reached through one.
///
/// ⭐ THE SAME WALK AS [`super::tf_mutable_addr_splitting::run_on_operation`], line for line — the
/// only differences are that pass's inert per-unit `num_conditionals_ = 0` and its `DT_CHECK` wording,
/// so `isCandidateMemView` and the pre-order view walk are reused rather than copied.
pub fn run_on_operation<'p, A: Arch>(
    program: &'p Program<A>,
    vals: &mut Values,
) -> MutableStartAddrShifting<'p> {
    // ⛔ `if (DisableThisPass) return;` (`:131`) IS DROPPED: `dcc-mutable-start-addr-shifting-disable`
    // is a `dcc-opt` command-line flag, and which passes run is a call in this crate.
    let mut candidates: Vec<(MasCandidate<'p>, &'p [DfirOp])> = Vec::new();
    // `std::unordered_set<Operation *> analyzed_candidates;` — ⚠️ BY IDENTITY, so two structurally
    // equal transfers are two entries.
    let mut analyzed: Vec<&'p DfirOp> = Vec::new();

    // `module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit) { .. })`.
    for unit in program.units.iter() {
        // `if (!is_any_of(comp, L3LU, L3SU)) return WalkResult::advance();`
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
                ..
            }) = view
            else {
                continue;
            };
            // `if (!mem_view || !isCandidateMemView(mem_view)) return WalkResult::advance();`
            if !is_candidate_mem_view(*from, *start, body, &program.preamble) {
                continue;
            }
            // `DT_CHECK_MSG(mem_view->hasOneUse(), ..)` then `*mem_view->getUsers().begin()` —
            // [`uses`] gives one entry per USE, so a view read twice by one transfer is not
            // single-used either.
            let users = uses(*result, body);
            let [mem_op] = users.as_slice() else {
                return MutableStartAddrShifting::MemViewIsNotSingleUsed(*result);
            };
            let mem_op = *mem_op;

            // `agen::MemoryOperandIndex mem_index = agen::MemoryOperandIndex::kMax;`
            let mem_index = match mem_op {
                // `comp_las.getSrcMemRef() == mem_view.getResult() ? kDirSrc : kDirDst`.
                DfirOp::Agen(agen::Op::CompositeLoadAndStore(transfer)) => {
                    Some(if transfer.src == *result {
                        MemoryOperandIndex::DirSrc
                    } else {
                        MemoryOperandIndex::DirDst
                    })
                }
                // ⛔ TWO `if`s AND NO `else` (`:167-172`) — an indirect transfer whose HBM view is one
                // of its INDIRECT operands leaves `mem_index` at `kMax` and hits the check below.
                DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(transfer)) => {
                    if transfer.direct_src == *result {
                        Some(MemoryOperandIndex::DirSrc)
                    } else if transfer.direct_dst == *result {
                        Some(MemoryOperandIndex::DirDst)
                    } else {
                        None
                    }
                }
                // `} else { mem_index = kDirSrc; }` — a vector load reads its one memref as the
                // source, and so does a vector STORE's, which is the reference's own answer for it.
                _ => Some(MemoryOperandIndex::DirSrc),
            };
            let Some(mem_index) = mem_index else {
                return MutableStartAddrShifting::InvalidHbmMemoryOperand(*result);
            };
            // `auto res = analyzed_candidates.insert(mem_op); DT_CHECK_MSG(res.second, ..);`
            if analyzed.iter().any(|seen| core::ptr::eq(*seen, mem_op)) {
                return MutableStartAddrShifting::SecondHbmMemoryOperand(*result);
            }
            analyzed.push(mem_op);
            candidates.push((
                MasCandidate {
                    op: mem_op,
                    comp,
                    mem_index,
                },
                body,
            ));
        }
    }

    // `for (auto &candidate : candidates)` — a SECOND loop, after the whole module was walked.
    let mut shifted: Vec<MsasShift<'p>> = Vec::new();
    for (candidate, scope) in candidates {
        let dispatch = match candidate.op {
            DfirOp::Agen(agen::Op::VectorLoad { .. }) => {
                MsasDispatch::VectorLoad(transform_vector_load::<A>(vals, &candidate, scope))
            }
            DfirOp::Agen(agen::Op::VectorStore { .. }) => {
                MsasDispatch::VectorStore(transform_vector_store::<A>(vals, &candidate, scope))
            }
            DfirOp::Agen(agen::Op::CompositeLoadAndStore(_)) => MsasDispatch::CompLoadAndStore(
                transform_comp_load_and_store::<A>(vals, &candidate, scope),
            ),
            DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(_)) => {
                MsasDispatch::CompIndLoadAndStore(transform_comp_ind_load_and_store::<A>(
                    vals, &candidate, scope,
                ))
            }
            _ => MsasDispatch::UnexpectedCandidateOperation,
        };
        shifted.push(MsasShift {
            candidate,
            dispatch,
        });
    }
    MutableStartAddrShifting::Shifted(shifted)
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
