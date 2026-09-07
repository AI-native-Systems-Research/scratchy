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

use super::agen_access_details::{AccessDetailsAffine, LayoutCoeff};
use super::tf_mutable_addr_splitting::{AddrRange, ConstStartMemView, L3Half};
use crate::arch::{Arch, Elements};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, arith};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap};
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
}

impl<'a> ShiftInputs<'a> {
    /// THE FOUR FACTS, READ OFF ONE `AccessDetailsAffine`.
    #[must_use]
    pub fn of<A: Arch>(ad: &'a AccessDetailsAffine<'a>) -> Option<ShiftInputs<'a>> {
        Some(ShiftInputs {
            layout_coeffs: &ad.base.layout_coeffs,
            transfer_order: &ad.base.transfer_order,
            subscripts_map: ad.subscripts_map.as_ref()?,
            elements_per_stick: ElementsPerStick::of::<A>(NonZeroU32::new(
                ad.base.element_width.0,
            )?)?,
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
    // (`:172`). Then `updateMemViewStartAddress`, arm one: one `arith.constant` for `start +
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::formats::Bits;
    use crate::generated::DataType;
    use crate::islands::dataflow_ir::dialects::{Index, Val, agen};
    use crate::islands::dataflow_ir::ty::{AffineExpr, ElemType, MemRef, Vector};
    use crate::units::DfirUnit;

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
        }
    }

    /// The fixture's `agen.vector_load %src_mem_view[%c64, %arg1 * 3 + %c16, %arg2 * 2 + %c128] :
    /// memref<?x64x4xf16>, vector<64xf16>` (`mutable_start_addr_shift_full.mlir:151`) — the op an
    /// `AccessDetailsAffine` hangs off. Nothing in entry 191 reads it; `AccessDetailsAffine::new`
    /// takes one because the reference's constructor does.
    fn fixture_load() -> agen::Op {
        agen::Op::VectorLoad {
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
            let constant_shift = (curr_space / stride).min(dim_weight.offset.0);
            if constant_shift <= 0 {
                continue;
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
    /// `@full_shift_zero_const_start` (`mutable_start_addr_shift_full.mlir:14-33`): every constant
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
}
