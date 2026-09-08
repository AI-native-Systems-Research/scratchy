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

//! `AccessDetails.cpp` — 40 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e002_setCoalescedBoundValues` | 002/384 | 5 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672` |
//! | `e003_setIndices` | 003/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77` |
//! | `e004_setMemViewStartAddr` | 004/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81` |
//! | `e005_setMemoryIndex` | 005/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93` |
//! | `e006_setLayoutCoeffs` | 006/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96` |
//! | `e007_setMemViewLayoutMap` | 007/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99` |
//! | `e008_setShuffleMode` | 008/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104` |
//! | `e009_setRotationPosition` | 009/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107` |
//! | `e010_setExpectedTotalElements` | 010/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110` |
//! | `e011_setExtents` | 011/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113` |
//! | `e012_setTotalElements` | 012/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116` |
//! | `e013_setElementWidth` | 013/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119` |
//! | `e014_setTransferSet` | 014/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122` |
//! | `e015_setTransferOrder` | 015/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125` |
//! | `e016_AccessDetailsBase` | 016/384 | 0 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218` |
//! | `e017_setSubscriptsMap` | 017/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228` |
//! | `e018_setIndicesCoeffDict` | 018/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231` |
//! | `e019_AccessDetailsAffine` | 019/384 | 0 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259` |
//! | `e020_setTimeAddrMap` | 020/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286` |
//! | `e021_setTimeSymbols` | 021/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289` |
//! | `e022_setTimeBounds` | 022/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292` |
//! | `e023_setTimeOffsets` | 023/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295` |
//! | `e024_setInterleaveGroupIndex` | 024/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299` |
//! | `e025_setStrides` | 025/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355` |
//! | `e026_has` | 026/384 | 2 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397` |
//! | `e143_constructExtentAndTotalElements` | 143/384 | 64 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32` |
//! | `e144_constructLdOrStType` | 144/384 | 5 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267` |
//! | `e145_initializeMemViewInfo` | 145/384 | 16 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274` |
//! | `e146_constructIndices` | 146/384 | 50 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354` |
//! | `e147_constructIteratorCoefficients` | 147/384 | 10 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406` |
//! | `e148_computeBurstAndGroup` | 148/384 | 36 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796` |
//! | `e149_insert` | 149/384 | 7 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388` |
//! | `e150_get` | 150/384 | 4 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401` |
//! | `e206_constructChunkAndShuffleInfo` | 206/384 | 167 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98` |
//! | `e207_initialize` | 207/384 | 57 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295` |
//! | `e208_emplace_insert` | 208/384 | 8 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378` |
//! | `e209_getFirst` | 209/384 | 7 | `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411` |
//! | `e265_constructDetails` | 265/384 | 18 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418` |
//! | `e266_coalesceTimeDimensions` | 266/384 | 112 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681` |
//! | `e297_constructTimeStepsInfo` | 297/384 | 42 | `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626` |
//!
//! Original files homed here: `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp`, `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp`

use crate::arch::Elements;
use crate::formats::Bits;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, agen, arith, dataflow, defining_op, region_owner, scf, uses,
    vectorchain,
};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, FlatConstraints, IntegerSet};
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

use super::vc_vector_operands::access_map;

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// THE VOCABULARY `AccessDetails` IS WRITTEN IN.
//
// ⭐ These are the types the reference declares alongside the class in the same header, so they are
// homed with it. Each one exists because a field of `AccessDetailsBase` or a parameter of one of its
// members is an `int64_t`/`std::string`/`enum` in the C++ that this crate may not leave raw.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH ADDRESS OPERAND OF A MEMORY ACCESS ONE `AccessDetails` DESCRIBES — `MemoryOperandIndex`
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:38`).
///
/// ⛔⛔ FOUR OPERANDS, NOT FIVE. The reference's fifth enumerator `kMax` is not an operand at all —
/// it is simultaneously the COUNT (`AccessContainer` sizes `index_mapping_` with `(int)kMax`,
/// `:375`) and the "not set yet" value of `memory_index_` (`:151`), which the reference then has to
/// defend at run time: `DT_CHECK_MSG(getMemoryIndex() != MemoryOperandIndex::kMax, ...)` appears
/// three times (`AccessDetails.cpp:296`, `:443`, `:859`). Here "not set" is `None` in the field's own
/// `Option`, so all three of those checks become the absence of a value rather than an assertion.
///
/// ⭐ WHICH ONES A GIVEN OP USES (`:31-35`): every memory op but the composite stores uses `kDirSrc`;
/// `composite_load_and_store` uses `kDirSrc` for the load and `kDirDst` for the store;
/// `composite_indirect_load_and_store` uses either or both of `kIndSrc` and `kIndDst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryOperandIndex {
    /// `kDirSrc` — the direct source operand.
    DirSrc,
    /// `kIndSrc` — the indirect source operand.
    IndSrc,
    /// `kDirDst` — the direct destination operand.
    DirDst,
    /// `kIndDst` — the indirect destination operand.
    IndDst,
}
impl MemoryOperandIndex {
    /// HOW MANY THERE ARE — the reference's `kMax`, which is a COUNT and not an operand.
    ///
    /// ⛔ AN ASSOCIATED CONSTANT, NOT A FIFTH VARIANT. `AccessContainer` sizes `index_mapping_` with
    /// `(int)kMax` (`AccessDetails.hpp:375`), so in C++ `has(MemoryOperandIndex::kMax)` compiles and
    /// reads one past the end of a four-element vector. Spelling the count here makes that call
    /// unwritable.
    pub const COUNT: usize = 4;

    /// All four, in the reference's enumerator order.
    pub const ALL: [MemoryOperandIndex; MemoryOperandIndex::COUNT] = [
        MemoryOperandIndex::DirSrc,
        MemoryOperandIndex::IndSrc,
        MemoryOperandIndex::DirDst,
        MemoryOperandIndex::IndDst,
    ];

    /// ITS SLOT IN AN [`AccessContainer`]'S INDEX MAP — the reference's `(int)moi`, and so its
    /// enumerator value.
    #[must_use]
    pub const fn slot(self) -> usize {
        match self {
            MemoryOperandIndex::DirSrc => 0,
            MemoryOperandIndex::IndSrc => 1,
            MemoryOperandIndex::DirDst => 2,
            MemoryOperandIndex::IndDst => 3,
        }
    }
}

/// ONE COEFFICIENT OF A MEMORY VIEW'S LAYOUT MAP — a STRIDE, in elements.
///
/// ⛔ A NEWTYPE, AND SIGNED. `agen::utils::getMapCoefficients` flattens the view's `layout_map` into
/// one coefficient per dimension plus a trailing CONSTANT term —
/// `Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:50` asserts
/// `layout_coeffs.size() == operands.size() + 1` — and the reference carries them as `int64_t`,
/// flattened by MLIR's own `getFlattenedAffineExpr` (`dialect_utils/Agen/Utils.cpp:65-71`).
/// Typing them as [`crate::arch::Elements`] would claim they are a COUNT of elements rather than a
/// stride in elements, and its `u64` would assert a non-negativity `getMapCoefficients` never
/// promises. Later units divide neighbours to recover extents (`AccessDetails.cpp:69`, `:151`,
/// `:176`), so the ratio between two coefficients carries as much as either value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayoutCoeff(pub i64);

/// ONE TIME DIMENSION'S POSITION among a composite transfer's ordered time dimensions.
///
/// ⛔ A NEWTYPE, BECAUSE THE `int` IT REPLACES HELD TWO DIFFERENT KINDS OF THING. `burst_index_` and
/// `interleave_group_index_` index `time_bounds_` and `time_offsets_` — `time_bounds[burst_index]`,
/// `time_offsets[burst_index]` (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:819-821`) — and
/// both spell "no dimension chosen" as `-1` (`AccessDetails.hpp:324,329`). That is why
/// `computeBurstAndGroup` has to ask `if (burst_index < 0)` before it may index with the value
/// (`AccessDetails.cpp:813`). `Option<TimeDim>` cannot be indexed with until the question is asked,
/// so the guard is the match and no arm can index with the sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeDim(pub u32);

impl TimeDim {
    /// The position as a slice index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// ONE **TRANSFER SET** DIMENSION'S POSITION — ⛔ NOT A [`TimeDim`].
///
/// `constructExtentAndTotalElements` walks the dimensions of the composed `load_set`/`store_set`
/// (`AccessDetails.cpp:55`), which are the dimensions of ONE transfer's data, while a [`TimeDim`]
/// indexes the sequence of transfers. The two are counted separately, indexed into different vectors
/// — `extents_` and `layout_coeffs_` here, `time_bounds_` and `time_offsets_` there — and a composite
/// op has both at once. Sharing one newtype would let an extent's position be passed where a time
/// step's is meant, which is the E0308 this exists for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransferDim(pub u32);

impl TransferDim {
    /// The position as a slice index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// ONE TIME DIMENSION'S BOUND — ⛔ THE THREE STATES ONE `int64_t` WAS CARRYING AT ONCE.
///
/// ```text
/// enum SpecialTimeBoundValues {
///   kInvalid = -1,
///   kCoalesced = -2,  // time dimension that should be skipped
/// };
/// ```
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:252-255`), and the comment above it says
/// exactly why the overload existed: the flags are there "to distinguish coalesced bounds
/// (equivalent in value to 1) from non-coalesceable bounds with value 1" (`:249-251`). A reader must
/// therefore test the SIGN before it may use the number —
/// `if (curr_bound == kCoalesced) continue; else if (curr_bound < 0) return;`
/// (`AccessDetails.cpp:803-808`).
///
/// ⭐ AS AN ENUM THE TEST IS THE MATCH. Nothing can multiply a flag into a trip product the way
/// `coalesced_bound *= time_bounds[time_index_outer]` (`AccessDetails.cpp:777`) would if the -2
/// reached it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeBound {
    /// This dimension takes this many steps.
    ///
    /// `calculateTimeBounds` pushes `1` for a pinned dimension, the constant extent of a ranged one,
    /// or a constant symbol's value (`dialect_utils/Agen/Utils.cpp:216-256`). ⛔ `Steps(0)` IS
    /// REACHABLE AND IS ITS OWN CASE: `computeBurstAndGroup` gives `curr_bound == 0` an explicit
    /// empty branch that neither terminates the search nor claims a field
    /// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:809`).
    Steps(u64),
    /// `kCoalesced = -2` — this dimension was merged into an inner one and is skipped.
    ///
    /// Written only by `setCoalescedBoundValues` over the strictly-interior dimensions of a merged
    /// run (`AccessDetails.cpp:676-678`), and skipped by `continue` on the way out
    /// (`AccessDetails.cpp:803-805`).
    Coalesced,
    /// `kInvalid = -1` — a bound that is not a compile-time constant.
    ///
    /// The reference documents it on the producer — "Non-constant dimensions get -1 as bound value"
    /// (`AccessDetails.hpp:273-274`) — and the consumer defends against it by terminating the burst
    /// search: "if forOp bound is variable, terminate search" (`AccessDetails.cpp:806-808`).
    ///
    /// ⛔ ON THIS PATH IT IS UNREACHABLE, AND THE VARIANT STAYS ANYWAY. dcc's own
    /// `calculateTimeBounds` returns `failure()` for a non-constant dimension rather than pushing -1
    /// (`dialect_utils/Agen/Utils.cpp:216-256`), so `constructTimeStepsInfo` aborts the whole
    /// lowering instead (`AccessDetails.cpp:643-646`). Dropping the variant would delete the only
    /// distinction the consumer's own guard is written against.
    Variable,
}

/// THE ADDRESS OFFSETS ALONG A COMPOSITE TRANSFER'S TIME DIMENSIONS — ⛔ THE TRAILING CONSTANT IS
/// NOT A DIMENSION.
///
/// `calculateTimeOffsets` flattens `mem_view_layout_map.compose(time_addr_map)`, drops the flattened
/// expression's last term from the per-dimension part, reorders what is left by `time_order`, and
/// only then appends that last term back:
/// ```text
/// for (int i = 0, e = tmp_time_offsets.size() - 1; i < e; ++i)
///   tmp_non_const_time_offsets.push_back(tmp_time_offsets[i]);
/// time_offsets = time_order.compose(tmp_non_const_time_offsets);
/// time_offsets.push_back(tmp_time_offsets.back());
/// ```
/// (`dialect_utils/Agen/Utils.cpp:112-116`). So the vector is n offsets plus one constant, and
/// `getFlattenedAffineExpr`'s trailing term is a CONSTANT TERM, not the (n+1)-th dimension.
///
/// ⭐⭐ WHICH IS WHY THE REFERENCE HAS TO CHECK THE LENGTHS BY HAND, TWICE:
/// `DT_CHECK(getTimeBounds().size() == getTimeOffsets().size() - 1)`
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:684-685`, again at `:696-701`) and
/// `DT_CHECK_MSG(time_bounds.size() + 1 == time_offsets.size(), "expected same number of dimensions
/// for time_addr map and time_set")` (`:742-744`). Naming the constant makes `per_dim` index in
/// lockstep with `time_bounds` by construction, and those three run-time checks have nothing left to
/// check.
///
/// ⛔ `Default` IS THE CONSTRUCTED-BUT-UNSET STATE and it is not an invention: `time_offsets.back()`
/// is a `getFlattenedAffineExpr` coefficient vector's LAST entry, which that function always emits
/// and which is 0 for an expression with no constant term. `calculateTimeOffsets` takes it
/// unconditionally (`dialect_utils/Agen/Utils.cpp:116`) — it has no fallback of its own; the
/// `const_value ? … : 0` ternary belongs to `constructIteratorCoeffDict`
/// (`dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Agen/Utils.cpp:73-76`),
/// which is [`IndicesCoeffDict`]'s producer, not this one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimeOffsets {
    /// One offset per time dimension, ordered by `time_order` — outermost first.
    pub per_dim: Vec<i64>,
    /// The flattened constant term, `time_offsets.back()`.
    pub constant: i64,
}

/// THE LOOP-ITERATOR COEFFICIENTS OF AN ACCESS — ⛔ WITH THE `nullptr` KEY GIVEN A NAME.
///
/// `constructIteratorCoeffDict` builds a `DenseMap<Value, int64_t>` of one coefficient per index and
/// then stores the flattened constant term under a NULL `Value`:
/// ```text
/// // indices_coeff_dict is a map from loop iterators to coefficients associated
/// // with them. nullptr refers to constant offset
/// // for, e.g., 2xi + 3xj + 10
/// // coefficient with i is 2, j is 3, and nullptr is 10.
/// indices_coeff_dict[nullptr] = const_value;
/// ```
/// (`dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Agen/Utils.cpp:69-82`).
/// [`Val`] has no null, and inventing one would put the flag back.
///
/// ⭐⭐ EVERY CONSUMER READS THAT KEY BY HAND AND ONE OF THEM COUNTS IT:
/// `auto const_offset = iter_coeff_dict[nullptr]` (`dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:402`),
/// `max_mutable = iter_coeff_dict[nullptr]` (`dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:724`)
/// guarded by `DT_CHECK(iter_coeff_dict.size() == indices.size() + 1)` (`:717`) — the `+ 1` IS this
/// field — and the iterating consumer has to test for it, `if (inner_record.first && ...)`
/// (`dcc/src/Conversion/AgenToSentient/Helper.cpp:577`). As a named field the check is structural and
/// the test disappears.
///
/// ⛔ A VEC, NOT A MAP, BECAUSE EVERY CONSUMER REACHES A COEFFICIENT THROUGH `indices_`, POSITION AND
/// ALL. ⚠️ Not because the C++ map carries an order — a `DenseMap` cannot, whatever order it was
/// filled in, and the *"Sort the indices from outermost to innnermost"* comment (`Utils.cpp:68`)
/// describes the `indices` vector the loop reads, not the map it writes. The ordered walk is
/// `initMASData`'s, over `ad.getIndices()`, keying the dict per index and recording the POSITION `i`
/// in the `MASData` it builds (`MutableAddrSplitting.cpp:724-737`); `Helper.cpp:559-572` and `:922`
/// key it the same way. So `per_index[i]` pairs `indices[i]` with its coefficient and one field
/// carries what the reference needs a vector plus a map to express.
///
/// ⚠️ THE ONE THING THE MAP DID THAT A VEC DOES NOT is deduplicate a repeated index. The reference's
/// own `DT_CHECK(iter_coeff_dict.size() == indices.size() + 1)` (`MutableAddrSplitting.cpp:717`) is
/// the evidence that it never has to: a duplicate would collapse two entries into one and fail that
/// length relation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndicesCoeffDict {
    /// One `(index, coefficient)` pair per access index, outermost loop first.
    pub per_index: Vec<(Val, i64)>,
    /// The constant offset — C++'s `indices_coeff_dict[nullptr]`.
    pub constant: i64,
}

/// Replaces: e002_setCoalescedBoundValues
///
/// `AccessDetailsAffineComposite::setCoalescedBoundValues` —
/// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672` (5L), entry 002/384.
///
/// ```cpp
/// void AccessDetailsAffineComposite::setCoalescedBoundValues(
///     SmallVectorImpl<int64_t>& time_bounds, int outer_dim, int inner_dim,
///     int64_t coalesced_bound) {
///   time_bounds[inner_dim] = coalesced_bound;
///   for (int i = outer_dim + 1; i < inner_dim; i++) {
///     time_bounds[i] = SpecialTimeBoundValues::kCoalesced;
///   }
/// }
/// ```
///
/// ⚠️⚠️ THE EXTRACT DROPPED THE LOOP'S ONLY STATEMENT. `crustify-bridge2/source/bridge2.cpp:35-41`
/// ends with `for (int i = outer_dim + 1; i < inner_dim; i++) {` and two bare braces — an empty loop,
/// which is HALF of this function and the half that marks the merged dimensions. Ported from the
/// authority at the cited line.
///
/// # ⛔⛔ THE WRITE SET IS `inner` PLUS THE DIMENSIONS STRICTLY BETWEEN, AND `outer` IS UNTOUCHED
///
/// The merged run's whole product lands on the INNERMOST dimension of the run and every dimension
/// above it inside the run becomes [`TimeBound::Coalesced`], i.e. a trip count of 1
/// (`Helper.cpp:1813-1819`). `outer_dim` itself is the dimension the run stopped BELOW — the loop
/// starts at `outer_dim + 1` — so it keeps its own bound. Off by one at either end and the product of
/// the trip counts changes, which is a transfer of the wrong length.
///
/// ⭐ `outer_dim == -1` IS A DELIBERATE SENTINEL, NOT AN ERROR. Its caller's scan runs
/// `for (int time_index_outer = num_of_dim - 2; time_index_outer >= -1; --time_index_outer)` with the
/// comment *"time_index_outer goes down to -1 to catch the outermost time dim"* (`:752-755`), so `-1`
/// means *the run reaches the top of the nest* and `outer_dim + 1` is dimension 0. That is [`None`]
/// here: an `Option<TimeDim>` cannot be confused with dimension zero, where an `i32` could.
///
/// ⭐ ONE DELIBERATE DIVERGENCE, AND THE MERGE IS ALL-OR-NOTHING BECAUSE OF IT. The reference
/// subscripts `time_bounds[inner_dim]` directly, so an `inner_dim` past the end writes past the
/// vector — and its marking loop then writes `kCoalesced` over every dimension there is, because
/// `i < inner_dim` is true for all of them. Both halves of that are dropped here: with no
/// `inner_dim` to receive the product, nothing is marked either, so the nest's trip count is
/// preserved instead of collapsing to 1. For every in-range argument the written slots are exactly
/// the reference's, and the only caller (`coalesceTimeDimensions:763`) is always in range.
pub fn set_coalesced_bound_values(
    time_bounds: &mut [TimeBound],
    outer_dim: Option<TimeDim>,
    inner_dim: TimeDim,
    coalesced_bound: TimeBound,
) {
    // ⛔ `outer_dim + 1`, WITH `-1` MAPPING TO 0. `None` is the reference's `-1` sentinel, so the run
    // starts at the outermost dimension; `Some(d)` starts one inside `d`, leaving `d` alone.
    let first_merged = outer_dim.map_or(0, |outer| outer.index() + 1);
    // ⛔ NO DIMENSION TO CARRY THE PRODUCT MEANS NO MERGE AT ALL — see the divergence note above.
    if inner_dim.index() >= time_bounds.len() {
        return;
    }
    for (dim, bound) in time_bounds.iter_mut().enumerate() {
        if dim == inner_dim.index() {
            // ⛔ THE WHOLE RUN'S PRODUCT, on the innermost dimension of the run. The caller has
            // already multiplied it up (`coalesceTimeDimensions:777`).
            *bound = coalesced_bound;
        } else if dim >= first_merged && dim < inner_dim.index() {
            *bound = TimeBound::Coalesced;
        }
    }
}

/// THE MEMORY VIEW AN ACCESS ADDRESSES THROUGH — the two facts `initializeMemViewInfo` narrows
/// `mem_ref_` down to, plus the unit they belong to.
///
/// # ⛔⛔ ONE STRUCT, BECAUSE THE REFERENCE'S TWO BRANCHES DIFFER ONLY IN SPELLING
///
/// ```cpp
/// if (auto mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(mem_ref)) {
///   setMemViewLayoutMap(mem_view.getLayoutMap());
///   setMemViewStartAddr(mem_view.getStartAddress());
///   setMemory(mem_view.getFromUnit());
/// } else if (auto paged_mem_view = dyn_cast<dataflow::GetPagedLogicalMemoryViewOp>(mem_ref)) {
///   setMemViewLayoutMap(paged_mem_view.getLayoutMap());
///   setMemViewStartAddr(paged_mem_view.getStartAddr());
///   setMemory(paged_mem_view.getUnit());
/// } else {
///   llvm_unreachable("unsupported mem_ref type");
/// }
/// ```
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274-289`). The two arms read the SAME three
/// facts through differently-named accessors — `getStartAddress`/`getStartAddr`,
/// `getFromUnit`/`getUnit` — and write them to the same three members in the same order. Nothing
/// downstream of this function can tell which arm ran, so an enum here would carry a tag no reader
/// has, and `TransformPagedMemViewImpl.cpp:553` confirms the paged form's own type is the plain
/// form's.
///
/// # ⛔ THE WITNESS IS WHAT RETIRES `llvm_unreachable("unsupported mem_ref type")`
///
/// The reference resolves `mem_ref_` and aborts the compiler when it is neither view op. Here
/// [`Self::of`] is the only way to obtain one, so
/// [`initialize_mem_view_info`](AccessDetailsBase::initialize_mem_view_info) cannot be called on an
/// unresolved `mem_ref` at all and the abort has nothing left to guard — the same witness-constructor
/// shape as [`EnclosingLoop::of`](super::vc_helper::EnclosingLoop::of).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemViewSource<'a> {
    /// `getLayoutMap()` — how the view's indices map onto its unit's linear region.
    pub layout: &'a AffineMap,
    /// `getStartAddress()` / `getStartAddr()` — the view's base, in elements.
    pub start_address: Val,
    /// `getFromUnit()` / `getUnit()` — the memory unit the view is cut from.
    pub memory: Val,
}

impl<'a> MemViewSource<'a> {
    /// THE VIEW A `mem_ref` RESOLVES TO — `mem_ref.getDefiningOp()` and the two `dyn_cast`s
    /// (`AccessDetails.cpp:275-286`).
    ///
    /// ⛔ [`None`] IS BOTH OF THE REFERENCE'S UNWRITABLE STATES AT ONCE: a `mem_ref` that is a region
    /// argument, so `getDefiningOp()` is null and the first `dyn_cast` reads through it, and one
    /// defined by any other op, which is the `llvm_unreachable`. Neither is a state a caller may pass
    /// on, and neither can be turned into a view.
    #[must_use]
    pub fn of(mem_ref: Val, scope: &'a [DfirOp]) -> Option<MemViewSource<'a>> {
        match defining_op(mem_ref, scope)? {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                from,
                start,
                layout,
                ..
            }) => Some(MemViewSource {
                layout,
                start_address: *start,
                memory: *from,
            }),
            DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(view)) => {
                Some(MemViewSource {
                    layout: &view.layout,
                    start_address: view.start_addr,
                    memory: view.unit,
                })
            }
            _ => None,
        }
    }
}

/// WHAT `constructExtentAndTotalElements` FOUND — ⛔ THE `LogicalResult` WITH ITS FIVE
/// `emitError`S KEPT APART.
///
/// The reference returns `success()` or one of five distinct `op->emitError(...)`s
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32-95`), and its caller
/// `constructDetails` propagates the failure without reading the message (`:418-437`). A `bool`
/// would erase which refusal fired, and a `Result` is what this crate does not have — so the
/// outcomes are variants, carrying the numbers the messages describe in prose.
///
/// ⭐ THE PRECEDENT IS `agen_helper.rs:834-864`: a `LogicalResult` + `emitOpError` function becomes a
/// `#[must_use]` enum with one variant per outcome.
///
/// ⛔ A REFUSAL LEAVES THE OBJECT PARTLY WRITTEN, AND THAT IS THE REFERENCE'S BEHAVIOUR.
/// `setLayoutCoeffs` runs before the loop (`:46`), and `setTotalElements`/`setExtents` run only after
/// it completes (`:87-88`) — so an [`Self::ExtentOverLimit`] leaves the coefficients installed and the
/// extents from the previous call in place, while an [`Self::ElementCountMismatch`] leaves all three
/// written. Nothing rolls back, because the caller abandons the lowering.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferExtents {
    /// `success()` — every dimension is a constant range or pinned, and the element count agrees with
    /// the return type.
    Rectangular,

    /// *"Extent requsted along a dimension is more than its limit"* (`:71-74`, the reference's own
    /// spelling).
    ///
    /// ⭐ THE LIMIT IS THE RATIO BETWEEN NEIGHBOURING LAYOUT STRIDES — how many elements fit along
    /// this dimension before the address walks into the next one — except at the fastest-moving end,
    /// where `width + 1` makes the test vacuous (`:64-70`).
    ExtentOverLimit {
        /// Which transfer-set dimension.
        dim: TransferDim,
        /// The extent asked for.
        width: i64,
        /// The limit the layout allows.
        limit: i64,
    },

    /// *"Extent along a dimension is negative"* (`:81`) — reached for a width of **zero** as well,
    /// because the reference tests `width > 0` (`:62`).
    ExtentNotPositive {
        /// Which transfer-set dimension.
        dim: TransferDim,
        /// The non-positive width `isDimAConstantRange` reported.
        width: i64,
    },

    /// *"Not in a hyper-rectangular form"* (`:84`) — the dimension is neither pinned by an equality
    /// nor bounded by a constant range on both sides, so it has no extent to read.
    NotHyperRectangular {
        /// Which transfer-set dimension.
        dim: TransferDim,
    },

    /// *"Number of elements in return type not matching with load_set/store_set elements"* (`:91-93`).
    ElementCountMismatch {
        /// `getExpectedTotalElements()` — what the op's return type says it moves.
        expected: Elements,
        /// The product of the extents just computed.
        found: Elements,
    },

    /// ⛔ NOT A REFERENCE OUTCOME: `mem_view_layout_map_` IS STILL THE NULL MAP.
    ///
    /// The reference hands `getMemViewLayoutMap()` straight to `getMapCoefficients`, which calls
    /// `map.getResult(0)` on it (`dialect_utils/Agen/Utils.cpp:65-71`) — on a default-constructed
    /// `AffineMap` that is a null dereference, so there is no behaviour to preserve and nothing is
    /// knowable about the extents. Naming the state is the only total reading; it is unreachable on
    /// the reference's own call order, because `constructDetails` runs `initialize()` — and with it
    /// [`AccessDetailsBase::initialize_mem_view_info`] — before this function (`:418-437`).
    MemoryViewUnresolved,
}

/// THE OUTCOME OF [`AccessDetailsBase::construct_chunk_and_shuffle_info`] (`AccessDetails.cpp:98`).
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkAndShuffleInfo {
    /// `success()` (`:265`) — chunk size, chunk stride and any shuffle or rotation are set.
    Chunked,

    /// `failure()` with NO message (`:126-130`): the coefficients are not one longer than the extents,
    /// or the innermost stride is not 1.
    ///
    /// ⛔ IT SUBSUMES `DT_CHECK(layout_coeffs.size() >= extents.size())` (`:110`) and it is where an
    /// EMPTY extent list lands — the reference indexes `[extents.size() - 1]` on it (`:111`, `:128`).
    LayoutDoesNotMatchExtents,

    /// *"Extent in load/store set is larger than from the layout"* — the SAME message from both loops
    /// (`:154-156`, `:177-179`), so the dimension names which.
    ExtentLargerThanLayout {
        /// Row-major dimension index, innermost first.
        dim: usize,
    },

    /// *"Chunk starting offset from load/store set are not supported in lowering"* (`:194-197`).
    ChunkStartingOffsetUnsupported,

    /// *"Variable chunk strides cannot be lowered"* (`:200-203`) — a second, outer chunk stride.
    VariableChunkStride {
        /// Row-major dimension index carrying it.
        dim: usize,
    },

    /// *"Abort due to invalid select map"* (`:239`), after one
    /// *"Non-supported select map for VectorLoadOp!"* (`:234`) per offending expression.
    InvalidSelectMap,

    /// `DT_CHECK_MSG(getRotationPosition() <= getTotalElements(), …)` (`:250-253`).
    RotationPositionOverTotalElements,

    /// `DT_CHECK_MSG(rotate_op.getRightShift(), …)` (`:256-257`).
    LeftRotationUnsupported,

    /// A negative rotation constant — the reference's `getRotationPosition() >= 0` (`:254-255`), which
    /// only an `int` field could ever fail; ⛔ A COUNT CANNOT HOLD IT, so it is refused on the way in.
    RotationPositionNegative,

    /// `DT_ERROR("index position to rotation op has to be a constant")` (`:259-260`).
    RotationPositionNotConstant,
}

impl ChunkAndShuffleInfo {
    /// `success()` only for [`Self::Chunked`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, ChunkAndShuffleInfo::Chunked)
    }

    /// The diagnostic the C++ emits, verbatim (`AccessDetails.cpp:126-260`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            ChunkAndShuffleInfo::Chunked | ChunkAndShuffleInfo::LayoutDoesNotMatchExtents => None,
            ChunkAndShuffleInfo::ExtentLargerThanLayout { .. } => {
                Some("Extent in load/store set is larger than from the layout")
            }
            ChunkAndShuffleInfo::ChunkStartingOffsetUnsupported => {
                Some("Chunk starting offset from load/store set are not supported in lowering")
            }
            ChunkAndShuffleInfo::VariableChunkStride { .. } => {
                Some("Variable chunk strides cannot be lowered")
            }
            ChunkAndShuffleInfo::InvalidSelectMap => Some("Abort due to invalid select map"),
            ChunkAndShuffleInfo::RotationPositionOverTotalElements => {
                Some("Rotation position has to be less than or equal to total elements")
            }
            ChunkAndShuffleInfo::LeftRotationUnsupported => Some("only right rotation supported"),
            ChunkAndShuffleInfo::RotationPositionNegative => {
                Some("right rotation amount has to be non-negative")
            }
            ChunkAndShuffleInfo::RotationPositionNotConstant => {
                Some("index position to rotation op has to be a constant")
            }
        }
    }
}

/// WHAT ONE MEMORY OPERAND'S LOWERING KNOWS ABOUT ITS ACCESS — `AccessDetailsBase`
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:43-212`).
///
/// One of these describes ONE address operand of ONE memory operation: a
/// `composite_load_and_store` fills two of them (`kDirSrc` for the load, `kDirDst` for the store,
/// `hpp:32-33`), which is why the object is per-operand and not per-op.
///
/// # ⛔ THE C++ HIERARCHY IS COMPOSITION HERE, NOT INHERITANCE
///
/// `AccessDetailsBase` → `AccessDetailsAffine` → `AccessDetailsAffineComposite`, and
/// `AccessDetailsBase` → `AccessDetailsSymbolic` (`hpp:214-245`, `:247`, `:345`). The derived classes add
/// FIELDS and override `initialize`/`constructDetails`; they never re-`private` a base member. So a
/// derived type here OWNS a base and reaches its members through it, which is what
/// [`AccessDetailsAffine`] does.
///
/// # ⛔ THE FIELDS ARE `pub` BECAUSE THAT IS WHERE THE GETTERS WENT
///
/// The winnow excludes 47 one-line C++ accessors on the stated ground that *"`unsigned
/// getElementWidth() { return element_width_; }` is a struct field in Rust"*
/// (`docs/bridge2-porting-order.md`), and `getElementWidth`, `getTotalElements`, `getExtents`,
/// `getRotationPosition`, `getTransferOrder`, `getExpectedTotalElements` and `getTransferSet` are
/// all on that list. Their readers are in other modules — `MutableStartAddrShifting.cpp:373` divides
/// the stick by `ad.getElementWidth()`, `:610` indexes `ad.getExtents()[res]` — so the field has to
/// be reachable from there for the exclusion to hold. The SETTERS are separate scheduled units
/// (entries 003-025) because they are what the lowering calls to fill this object.
///
/// # ⛔ THIS STRUCT IS FILLED WAVE BY WAVE
///
/// Only the members whose setters have been ported are declared. Entries 003-008 brought
/// `memory_index_`, `layout_coeffs_`, `mem_view_start_addr_`, `mem_view_layout_map_`,
/// `shuffle_mode_` and `indices_`, and entries 009-015 brought `rotation_position_`,
/// `expected_total_elements_`, `extents_`, `total_elements_`, `element_width_`, `transfer_set_` and
/// `transfer_order_`; entries 143-148 brought `memory_`, `chunk_size_` and `ld_or_st_size_`, and
/// entry 206 brought `chunk_stride_`, and `mem_ref_` still arrives with the entry that writes it. The
/// order below is the C++'s own declaration order
/// (`hpp:150-210`), so each wave inserts rather than reorders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessDetailsBase<'a> {
    /// `memory_index_` — *"The memory operand this object represents"* (`:150`). Setter e005,
    /// **protected**.
    ///
    /// ⛔ `None` IS THE REFERENCE'S `kMax` DEFAULT, the one it guards with three
    /// `DT_CHECK_MSG(... != kMax)`s. Once `constructDetails` sets it (`:421`, `:837`, `:901`) it is
    /// never returned to unset.
    pub memory_index: Option<MemoryOperandIndex>,

    /// `op_` (`hpp:154`) — *"Operation represented by this object"*.
    ///
    /// ⛔ NARROWED FROM `mlir::Operation*` TO THE AGEN DIALECT, and the source supports it: every
    /// `initialize()` in the family casts to an agen access and errors otherwise — `VectorLoadOp`,
    /// `VectorStoreOp`, `IndirectVectorLoadOp`, `IndirectVectorStoreOp` (`AccessDetails.cpp:295-352`),
    /// `CompositeLoadOp`/`CompositeStoreOp` (`:442-`), `SymbolicVectorLoadOp`/`SymbolicVectorStoreOp`
    /// (`:858-888`) — and every construction site passes one: `emplace_insert(kDirSrc, src_op, comp)`
    /// from `constructAffineDetailsAndAddrs` (`Helper.cpp:2794`) and from the transform passes'
    /// candidates (`MutableStartAddrShifting.cpp:203-209` casts to `agen::VectorLoadOp` first).
    ///
    /// ⛔ A BORROW, NOT A COPY. The C++ holds a non-owning `Operation*` into the module being
    /// lowered; the lifetime says the same thing and stops this object outliving the program it
    /// describes.
    pub op: &'a agen::Op,

    /// `comp_` (`hpp:157`) — *"Component containing the operation represented by this object"*.
    ///
    /// [`DfirUnit`] is this crate's `SenComponents` subset (`sys-arch-spec/arch_enums.h:13`); it is
    /// what the addressing rules branch on (`is_any_of(getComp(), L0LU, L0SU)` in
    /// `constructLdOrStType`, `AccessDetails.cpp:267-272`).
    pub comp: DfirUnit,

    /// `layout_coeffs_` — *"Coefficients of layout associated with memory view"* (`:159`). Setter
    /// e006, **protected**: only `constructExtentAndTotalElements` writes it, from
    /// `getMapCoefficients` on the view's layout map (`:45-46`).
    pub layout_coeffs: Vec<LayoutCoeff>,

    /// `memory_` — *"Memory unit"* (`hpp:162-163`). Setter `setMemory` (`hpp:84`), **public** and
    /// excluded from the port as *"a struct field in Rust"*; the only writer is
    /// [`Self::initialize_mem_view_info`], which takes it from the view's own unit operand.
    ///
    /// ⭐ THE UNIT THE VIEW IS CUT FROM, NOT THE VIEW. `get_logical_memory_view %unit, %start` names
    /// its memory as an operand, and it is that operand this field holds — `getFromUnit()` on the
    /// plain view, `getUnit()` on the paged one (`AccessDetails.cpp:279`, `:285`).
    ///
    /// ⭐ `None` IS THE C++ NULL `Value`, for [`Self::mem_view_start_addr`]'s reason: the member is
    /// default-constructed and only `initializeMemViewInfo` gives it one.
    pub memory: Option<Val>,

    /// `mem_ref_` — the view operand the access addresses through (`hpp:166`). Getter `getMemRef`
    /// (`hpp:59`), setter `setMemRef` (`hpp:80`), both **public** and both excluded from the port as
    /// field accessors.
    ///
    /// ⭐ `None` IS THE C++ NULL `Value`, as for [`Self::mem_view_start_addr`]: the member
    /// default-constructs and only an `initialize()` gives it one.
    ///
    /// ⛔ IT IS THE VIEW, NOT THE MEMORY. [`Self::memory`] holds the view's own unit operand; this
    /// holds the `get_logical_memory_view` result the access subscripts — which is why
    /// `e155_updateSymbolicAccessDetails` remaps BOTH (`Helper.cpp:1037-1046`).
    pub mem_ref: Option<Val>,

    /// `mem_view_start_addr_` — *"Associated memory view start address"* (`:168`). Setter e004,
    /// **public**: `generateAffineAddressManipulationStmts` rewrites it to a mutable address base
    /// (`Helper.cpp:901`), and `updateSymbolicAccessDetails` remaps it onto a clone (`:1043`).
    ///
    /// ⭐ `None` IS THE C++ NULL `Value`. The member is default-constructed and only
    /// `initializeMemViewInfo` (`AccessDetails.cpp:277-283`) gives it one, so "no view resolved yet"
    /// is a real state and not a zero address.
    pub mem_view_start_addr: Option<Val>,

    /// `mem_view_layout_map_` — *"Associated memory view layout"* (`:171`). Setter e007,
    /// **protected**: only `initializeMemViewInfo` writes it (`:277`, `:282`).
    ///
    /// ⭐ `None` IS THE C++ NULL `AffineMap`, for the same reason as `mem_view_start_addr`.
    pub mem_view_layout_map: Option<AffineMap>,

    /// `chunk_size_` — *"Chunk size - a group of chunks make a data set to transfer"*
    /// (`hpp:174-175`), in elements. Setter `setChunkSize` (`hpp:102`), **protected** and excluded
    /// from the port as a field.
    ///
    /// ⛔ TWO ENTRIES SHARE IT: `constructLdOrStType` copies it into `ld_or_st_size_` for an L0
    /// component (`AccessDetails.cpp:267-272`), and its writer is
    /// [`AccessDetailsBase::construct_chunk_and_shuffle_info`] (`:97-265`).
    ///
    /// ⭐ `int chunk_size_ = 0` (`hpp:175`), and zero is the "not computed yet" value rather than an
    /// empty chunk: `constructChunkAndShuffleInfo` assigns it unconditionally before anything reads
    /// it.
    pub chunk_size: Elements,

    /// `chunk_stride_` — *"Difference between consecutive chunks"* (`hpp:177-178`), in elements.
    /// Setter `setChunkStride` (`hpp:103`), **protected** and excluded from the port as a field.
    ///
    /// ⛔ THE STRIDE IS BETWEEN CHUNK **STARTS**, NOT THE GAP BETWEEN THEM. It is the layout's
    /// running extent product taken at the first non-unit extent ABOVE the chunk dimension
    /// (`AccessDetails.cpp:169-192`) — the distance one chunk index moves the address.
    ///
    /// ⭐ `int chunk_stride_ = 0` (`hpp:178`), and ZERO IS A MEANINGFUL ANSWER HERE, unlike
    /// [`Self::chunk_size`]: *no* dimension above the chunk has a non-unit extent, i.e. the transfer
    /// is one chunk. The legality test reads exactly that — `chunk_stride != 0` beside a chunk
    /// dimension whose layout coefficient is 1 (`:194-197`).
    pub chunk_stride: Elements,

    /// `shuffle_mode_` — *"Shuffle mode on the data"* (`:180`). Setter e008, **protected**.
    ///
    /// ⛔ AN ENUM, NOT A STRING. See [`AccessDetailsBase::set_shuffle_mode`].
    pub shuffle_mode: sen::ShuffleMode,

    /// `rotation_position_` (`hpp:184`) — *"Rotation position on the data"*. Written by
    /// [`Self::set_rotation_position`].
    pub rotation_position: Elements,

    /// `expected_total_elements_` (`hpp:187`) — *"Expected number of elements involved in the
    /// transfer from the return type"*. Written by [`Self::set_expected_total_elements`].
    pub expected_total_elements: Elements,

    /// `extents_` (`hpp:190`) — *"Extents or sizes of each dimension in the layout map"*. Written by
    /// [`Self::set_extents`].
    pub extents: Vec<Elements>,

    /// `total_elements_` (`hpp:193`) — *"Total elements involved in a transfer"*. Written by
    /// [`Self::set_total_elements`].
    pub total_elements: Elements,

    /// `element_width_` (`hpp:196`) — *"Width (in terms of bits) of an element inside the
    /// transfer"*. Written by [`Self::set_element_width`].
    pub element_width: Bits,

    /// `indices_` — *"Indices used in the access subscripts"* (`:198`, declared `:199`). Setter e003,
    /// **public**: `updateSymbolicAccessDetails` remaps every index onto its clone and writes that
    /// back from another translation unit (`Helper.cpp:1027`).
    pub indices: Vec<Val>,

    /// `transfer_set_` (`hpp:202`) — *"Load or store set within a transfer"*. Written by
    /// [`Self::set_transfer_set`].
    pub transfer_set: IntegerSet,

    /// `transfer_order_` (`hpp:205`) — *"Load order or store order within a transfer"*. Written by
    /// [`Self::set_transfer_order`].
    pub transfer_order: AffineMap,

    /// `ld_or_st_size_` — *"Load or store size (different from total_element) and this is specific to
    /// Sentient aspects"* (`hpp:207-211`), in elements. Written by
    /// [`Self::construct_ld_or_st_type`].
    ///
    /// ⭐ THE COMMENT SAYS THE WHOLE RULE: *"In case of L0, ld size refers to chunk size. In other
    /// cases, ld size refers to total_elements"* (`hpp:209-210`) — which is entry 144, and the one
    /// place the two candidates are distinguished.
    ///
    /// ⛔ AND IT IS NOT `total_elements_`, WHICH IS WHY IT EXISTS. `computeBurstAndGroup` compares a
    /// time offset against THIS field to decide whether an interleave group may be claimed
    /// (`AccessDetails.cpp:820`); reading `total_elements_` there would use the transfer's whole
    /// element count where the hardware's per-instruction count is meant, and on L0 those differ by
    /// the number of chunks.
    pub ld_or_st_size: Elements,
}

impl<'a> AccessDetailsBase<'a> {
    /// THE BASE CONSTRUCTOR — `explicit AccessDetailsBase(mlir::Operation* op, SenComponents comp)
    /// : op_(op), comp_(comp) {}` (`AccessDetails.hpp:46-47`).
    ///
    /// ⭐ NOT A SCHEDULED UNIT, DELIBERATELY: the winnow's extractor caught this constructor as the
    /// data member `comp_` (`hpp:47`) and listed it among the 34 excluded *"C++ data MEMBER, not a
    /// function"* entries, so it carries no anchor. It is written here because entry 016 —
    /// [`AccessDetailsAffine::new`] — delegates to it, exactly as the C++ constructor does.
    ///
    /// ⛔ EVERY OTHER MEMBER TAKES ITS DECLARED DEFAULT, and those defaults are load-bearing rather
    /// than filler: `rotation_position_ = 0` (`hpp:184`) is what "no `vectorchain.rotate` consumer"
    /// means, because `constructChunkAndShuffleInfo` only assigns it when it finds one
    /// (`AccessDetails.cpp:243-256`).
    ///
    /// ⛔ AND TWO OF THEM ARE NOT ZERO. `shuffle_mode_ = "noshuffle"` (`hpp:181`) is this class's
    /// starting mode, not an absent one, and `memory_index_ = kMax` (`hpp:151`) is the reference's
    /// "no operand chosen yet" — [`None`] here, which is what retires its three
    /// `DT_CHECK_MSG(getMemoryIndex() != kMax)`s (`AccessDetails.cpp:296`, `:443`, `:859`).
    ///
    /// ⛔ THE TWO MLIR ATTRIBUTES DEFAULT-CONSTRUCT TO **NULL** IN THE C++ (`hpp:202`, `:205`), and a
    /// null `AffineMap`/`IntegerSetAttr` has no Rust counterpart that is not an `Option` — which
    /// would put an unwrap on every read. They start EMPTY instead: the [`IntegerSet`] with no
    /// dimensions and no constraints — what `IntegerSet::from_sizes(&[])` builds, and what it
    /// documents as MLIR's `getEmptySet(0, 0)` — and an [`AffineMap`] with no results, which
    /// produces no address. Nothing reads either before
    /// `initialize()` sets both — it is the first thing every one of the four
    /// `initialize()` overrides does (`AccessDetails.cpp:303-304`, `:450-451`, `:866-867`), and
    /// `constructDetails` calls `initialize()` before `constructExtentAndTotalElements`
    /// (`:418-437`).
    #[must_use]
    pub fn new(op: &'a agen::Op, comp: DfirUnit) -> Self {
        AccessDetailsBase {
            // `MemoryOperandIndex memory_index_ = MemoryOperandIndex::kMax;` (`hpp:151`) — unset.
            memory_index: None,
            op,
            comp,
            // `SmallVector<int64_t> layout_coeffs_;` (`hpp:160`).
            layout_coeffs: Vec::new(),
            // `Value memory_;` (`hpp:163`) — a null `Value` until a view resolves.
            memory: None,
            // `Value mem_ref_;` (`hpp:166`) — null for the same reason.
            mem_ref: None,
            // `Value mem_view_start_addr_;` (`hpp:169`) — a null `Value` until a view resolves.
            mem_view_start_addr: None,
            // `AffineMap mem_view_layout_map_;` (`hpp:172`) — null for the same reason.
            mem_view_layout_map: None,
            // `int chunk_size_ = 0;` (`hpp:175`).
            chunk_size: Elements(0),
            // `int chunk_stride_ = 0;` (`hpp:178`).
            chunk_stride: Elements(0),
            // `std::string shuffle_mode_ = "noshuffle";` (`hpp:181`) — NOT the zero value.
            shuffle_mode: sen::ShuffleMode::NoShuffle,
            // `SmallVector<Value> indices_;` (`hpp:199`).
            indices: Vec::new(),
            // `int rotation_position_ = 0;` (`hpp:184`).
            rotation_position: Elements(0),
            // `int expected_total_elements_{};` (`hpp:187`) — value-initialised, so zero.
            expected_total_elements: Elements(0),
            // `SmallVector<int> extents_;` (`hpp:190`).
            extents: Vec::new(),
            // `int total_elements_ = 0;` (`hpp:193`).
            total_elements: Elements(0),
            // `unsigned element_width_ = 0;` (`hpp:196`).
            element_width: Bits(0),
            // `IntegerSetAttr transfer_set_;` (`hpp:202`) — see the null-attribute note above.
            transfer_set: IntegerSet {
                dims: 0,
                symbols: 0,
                constraints: Vec::new(),
            },
            // `AffineMap transfer_order_;` (`hpp:205`).
            transfer_order: AffineMap {
                dims: 0,
                syms: 0,
                results: Vec::new(),
            },
            // `int ld_or_st_size_ = 0;` (`hpp:211`).
            ld_or_st_size: Elements(0),
        }
    }

    /// Replaces: e003_setIndices
    ///
    /// `AccessDetailsBase::setIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77`
    /// (2L), entry 003/384. **public** in the reference.
    ///
    /// ```cpp
    /// void setIndices(const SmallVectorImpl<Value>& indices) {
    ///   indices_.assign(indices.begin(), indices.end());
    /// }
    /// ```
    ///
    /// ⛔⛔ `assign`, NOT `append` — THE WHOLE LIST IS REPLACED, and that is load-bearing rather than
    /// incidental. Of its five callers three are the `initialize()` overrides that fill the list from
    /// the op (`AccessDetails.cpp:347`, `:619`, `:890`), and TWO WRITE OVER AN OBJECT THAT ALREADY
    /// HOLDS ONE: `AccessDetailsAffine::constructIndices` reads it back with `getIndices()` (`:358`),
    /// keeps only the loop iterators, folds the constant subscripts into the subscripts map instead,
    /// and writes the SHORTER list over it (`:380`); `updateSymbolicAccessDetails` remaps each index
    /// onto its clone and writes that (`Helper.cpp:1027`). Appending would leave the stale subscripts
    /// in front of the live ones, which is an access at the wrong offset.
    pub fn set_indices(&mut self, indices: &[Val]) {
        self.indices = indices.to_vec();
    }

    /// Replaces: e004_setMemViewStartAddr
    ///
    /// `AccessDetailsBase::setMemViewStartAddr` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81` (2L), entry 004/384. **public** in
    /// the reference.
    ///
    /// ```cpp
    /// void setMemViewStartAddr(Value mem_view_start_addr) {
    ///   mem_view_start_addr_ = mem_view_start_addr;
    /// }
    /// ```
    ///
    /// ⛔ IT TAKES A VALUE, NOT AN OPTION. The C++ parameter is a `Value` that could be null, but no
    /// caller passes one: the two inside this file pass a view's own start address
    /// (`AccessDetails.cpp:278`, `:283`), `Helper.cpp:901` passes a mutable address base, and
    /// `:1043` passes the clone of the address already there. So the field's `None` means *never
    /// set*, and this setter cannot return it to that state.
    pub fn set_mem_view_start_addr(&mut self, mem_view_start_addr: Val) {
        self.mem_view_start_addr = Some(mem_view_start_addr);
    }

    /// Replaces: e005_setMemoryIndex
    ///
    /// `AccessDetailsBase::setMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93`
    /// (2L), entry 005/384. **protected** in the reference — only the `constructDetails` overrides
    /// call it (`AccessDetails.cpp:421`, `:837`, `:901`), each as the first statement of the
    /// construction it names.
    ///
    /// ```cpp
    /// void setMemoryIndex(MemoryOperandIndex memory_index) {
    ///   memory_index_ = memory_index;
    /// }
    /// ```
    ///
    /// ⛔ THE ARGUMENT IS ONE OF THE FOUR REAL OPERANDS, so this can only move the field from unset to
    /// set. `kMax` is not in [`MemoryOperandIndex`] and therefore cannot be passed — which is exactly
    /// what the reference's three `DT_CHECK_MSG(getMemoryIndex() != kMax)` assertions were checking.
    pub fn set_memory_index(&mut self, memory_index: MemoryOperandIndex) {
        self.memory_index = Some(memory_index);
    }

    /// Replaces: e006_setLayoutCoeffs
    ///
    /// `AccessDetailsBase::setLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96`
    /// (2L), entry 006/384. **protected** in the reference — only
    /// `constructExtentAndTotalElements` writes it (`AccessDetails.cpp:46`), straight from the
    /// `getMapCoefficients` call one line above it (`:45`).
    ///
    /// ```cpp
    /// void setLayoutCoeffs(const SmallVectorImpl<int64_t>& layout_coeffs) {
    ///   layout_coeffs_.assign(layout_coeffs.begin(), layout_coeffs.end());
    /// }
    /// ```
    ///
    /// ⛔ `assign` AGAIN — a replacement, not an append. The one caller passes a fresh local
    /// `SmallVector<int64_t, 8>` it has just filled, so appending would double the list on any object
    /// whose extents were constructed twice.
    pub fn set_layout_coeffs(&mut self, layout_coeffs: &[LayoutCoeff]) {
        self.layout_coeffs = layout_coeffs.to_vec();
    }

    /// Replaces: e007_setMemViewLayoutMap
    ///
    /// `AccessDetailsBase::setMemViewLayoutMap` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99` (2L), entry 007/384. **protected** in
    /// the reference — only `initializeMemViewInfo` writes it, from a `get_logical_memory_view`'s
    /// `getLayoutMap()` (`AccessDetails.cpp:277`) or a paged view's (`:282`).
    ///
    /// ```cpp
    /// void setMemViewLayoutMap(AffineMap mem_view_layout_map) {
    ///   mem_view_layout_map_ = mem_view_layout_map;
    /// }
    /// ```
    ///
    /// ⛔ THE MAP IS THE VIEW'S, IN ELEMENTS — the same `layout_map` the island documents on
    /// `get_logical_memory_view` (*"Addresses in Dataflow IR are in \*element\* granularity, not
    /// bytes."*, `dataflow-scheduler-dialects/…/Dialect/Dataflow/Dataflow.td:250`). It is taken by
    /// value here as it is there; `AffineMap` is an interned handle
    /// in MLIR and a small owned value in this crate, and neither is shared mutably.
    pub fn set_mem_view_layout_map(&mut self, mem_view_layout_map: AffineMap) {
        self.mem_view_layout_map = Some(mem_view_layout_map);
    }

    /// Replaces: e008_setShuffleMode
    ///
    /// `AccessDetailsBase::setShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104`
    /// (2L), entry 008/384. **protected** in the reference.
    ///
    /// ```cpp
    /// void setShuffleMode(std::string shuffle_mode) {
    ///   shuffle_mode_ = shuffle_mode;
    /// }
    /// ```
    ///
    /// # ⛔⛔ THE `std::string` IS A CLOSED SET AND BECOMES ONE
    ///
    /// The reference's field is a `std::string` initialised to `"noshuffle"` (`:181`) and there is
    /// **exactly one** call to this setter in the whole of `dcc/src`: `setShuffleMode("splat")` at
    /// `AccessDetails.cpp:231`. So the reachable value set is two spellings, both of them
    /// enumerators of `SentientShuffleMode` (`dcc/src/Dialect/Sentient/SentientTypes.td:511-533`),
    /// which this crate
    /// already carries as [`sen::ShuffleMode`]. Taking the enum is what makes an unspellable mode
    /// unrepresentable instead of a string that fails somewhere downstream.
    ///
    /// ⭐ THE WIDER ENUM IS STILL THE RIGHT PARAMETER TYPE. `setldtype` (`Helper.cpp:1647`) assigns
    /// `splat2b` and `zpad16b` for LX loads (`:1669`, `:1683`) on the sentient op directly rather than
    /// through this field, so the mode vocabulary is genuinely the attribute's and narrowing this
    /// parameter to the two spellings seen here would split one closed set into two.
    pub fn set_shuffle_mode(&mut self, shuffle_mode: sen::ShuffleMode) {
        self.shuffle_mode = shuffle_mode;
    }

    /// Replaces: e009_setRotationPosition
    ///
    /// **009/384** `setRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107`
    /// (2L).
    ///
    /// ```cpp
    /// void setRotationPosition(int rotation_position) {
    ///   rotation_position_ = rotation_position;
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING. This is a `protected` state setter, not a lowering: the op that carries
    /// the value is emitted by `constructLoadAndSendStmt`, which reads the field back at
    /// `Helper.cpp:1926` alongside `total_elements`, `element_width`, `chunk_size`, `chunk_stride`
    /// and `shuffle_mode`.
    ///
    /// ⛔ THE ONLY WRITER IS A `vectorchain.rotate` CONSUMER, and the position is the constant that
    /// op's `position` operand is defined by — `setRotationPosition(const_op.value())`
    /// (`AccessDetails.cpp:243-248`), where a non-constant position is a hard `DT_ERROR`.
    ///
    /// ⛔ IN ELEMENTS, WHICH IS WHY IT IS COMPARABLE TO `total_elements`. The C++ guards
    /// *"Rotation position has to be less than or equal to total elements"* (`:249-252`) — the two
    /// quantities are the same unit, and [`Elements`] is what makes that comparison type-correct.
    ///
    /// ⛔ AND ITS SECOND GUARD IS DISCHARGED BY THE TYPE. `DT_CHECK_MSG(getRotationPosition() >= 0,
    /// "right rotation amount has to be non-negative")` (`:253-254`) cannot fail here: [`Elements`]
    /// wraps a `u64`, so a negative rotation is unrepresentable rather than checked. The `<=
    /// total_elements` bound belongs to the writer (entry 206, `constructChunkAndShuffleInfo`),
    /// which is where both operands are in hand.
    pub fn set_rotation_position(&mut self, rotation_position: Elements) {
        self.rotation_position = rotation_position;
    }

    /// Replaces: e010_setExpectedTotalElements
    ///
    /// **010/384** `setExpectedTotalElements` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110` (2L).
    ///
    /// ```cpp
    /// void setExpectedTotalElements(int expected_total_elements) {
    ///   expected_total_elements_ = expected_total_elements;
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING — a `protected` state setter.
    ///
    /// ⛔⛔ THIS IS NOT `total_elements`, AND CONFLATING THEM DEFEATS THE ONE CROSS-CHECK THIS CLASS
    /// HAS. `expected_total_elements_` comes from the operation's TYPE —
    /// `getNumElements(load_op.getResult().getType())` (`AccessDetails.cpp:308-309`), the stored
    /// value's type for a store (`:318-319`), the load induction variable's type for a composite (`:459-460`) —
    /// while `total_elements_` is derived from the load_set/store_set geometry. `constructExtentAndTotalElements`
    /// then refuses the transfer when they disagree: *"Number of elements in return type not matching
    /// with load_set/store_set elements"* (`:91-94`).
    ///
    /// ⭐ IT IS ALSO THE LD/ST SIZE OUTSIDE THE L0: `constructLdOrStType` sets `ld_or_st_size_` to
    /// the chunk size on `L0LU`/`L0SU` and to THIS on every other component (`:267-272`).
    pub fn set_expected_total_elements(&mut self, expected_total_elements: Elements) {
        self.expected_total_elements = expected_total_elements;
    }

    /// Replaces: e011_setExtents
    ///
    /// **011/384** `setExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113` (2L).
    ///
    /// ```cpp
    /// void setExtents(const SmallVectorImpl<int>& extents) {
    ///   extents_.assign(extents.begin(), extents.end());
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING — a `protected` state setter.
    ///
    /// ⛔ `assign`, NOT `append` — it REPLACES the whole vector, so a second call does not leave a
    /// stale tail behind. That is the entire content of this unit, and it matters because the caller
    /// builds a fresh local `SmallVector<int> extents` per call
    /// (`AccessDetails.cpp:55`) and calls `setExtents` after the loop (`:89`).
    ///
    /// ⛔ ONE EXTENT PER DIMENSION OF THE TRANSFER SET, IN ITS DIMENSION ORDER, and each is at least
    /// one: a dimension pinned to zero contributes `1` (`:58-60`) and a constant range contributes
    /// its `width`, with a non-positive width refused as *"Extent along a dimension is negative"*
    /// (`:82`). [`Elements`] wraps a `u64`, so the negative case is unrepresentable here; the refusal
    /// stays where the width is computed (entry 143).
    ///
    /// ⛔ AND THE READERS INDEX IT BY DIMENSION, so the ORDER is part of the contract:
    /// `ad.getExtents()[res]` in `MutableStartAddrShifting.cpp:610` and
    /// `MutableAddrSplitting.cpp:1241`, and `constructChunkAndShuffleInfo` reverse-copies it against
    /// the layout coefficients (`AccessDetails.cpp:109-120`).
    ///
    /// ⛔ IT DOES NOT TOUCH `total_elements`. The C++ setter assigns one member; the product is
    /// accumulated by the caller and stored through [`Self::set_total_elements`] separately
    /// (`:88-89`).
    pub fn set_extents(&mut self, extents: &[Elements]) {
        self.extents.clear();
        self.extents.extend_from_slice(extents);
    }

    /// Replaces: e012_setTotalElements
    ///
    /// **012/384** `setTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116`
    /// (2L).
    ///
    /// ```cpp
    /// void setTotalElements(int total_elements) {
    ///   total_elements_ = total_elements;
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING — a `protected` state setter, but the value it stores reaches the wire:
    /// `constructLoadAndSendStmt` reads it at `Helper.cpp:1921` and it becomes the `total_elements`
    /// of the emitted sentient transfer.
    ///
    /// ⛔ THE PRODUCT OF THE EXTENTS, accumulated by `constructExtentAndTotalElements` with the
    /// idiom `total_elements = (total_elements == 0) ? width : total_elements * width`
    /// (`AccessDetails.cpp:60-64`) — so a zero start means "one", not "nothing".
    ///
    /// ⛔⛔ AND THE VALUE ON THE WIRE CAN STOP BEING THIS ONE. `constructLoadAndSendStmt` reads the
    /// field into a LOCAL (`Helper.cpp:1921`) and passes that local BY REFERENCE to `setldtype`
    /// (entry 130, `:1975`, signature `int& total_elements` at `:1649`), which overwrites it with the
    /// full stick — `sysDef.bytesPerStick * 8 / element_width` (`:1664`) — whenever a
    /// `vectorchain.shuffle` picks a non-default ldtype. The mutated LOCAL, not the field, is what
    /// reaches the emission at `:1988`; `getTotalElements()` still answers the extent product
    /// afterwards. A port that wrote back through the object would change what every later reader
    /// sees.
    pub fn set_total_elements(&mut self, total_elements: Elements) {
        self.total_elements = total_elements;
    }

    /// Replaces: e013_setElementWidth
    ///
    /// **013/384** `setElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119`
    /// (2L).
    ///
    /// ```cpp
    /// void setElementWidth(unsigned element_width) {
    ///   element_width_ = element_width;
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING — a `protected` state setter.
    ///
    /// ⛔⛔ IN BITS, AND THE UNIT IS THE WHOLE POINT. Every writer passes
    /// `dataflow::utils::getElementTypeBitWidth(..)` of the transferred value's type
    /// (`AccessDetails.cpp:306-307`, `:316-317`, `:453-454`, `:868-869`) — the ELEMENT's width, not
    /// the vector's — and the readers divide a stick by it: `dcc_ext_ctx_.getBytesPerStick() * 8 /
    /// ad.getElementWidth()` is elements-per-stick (`MutableStartAddrShifting.cpp:373`, `:405`,
    /// `:485`, `:553`). A byte width there would be eight times too small while still looking
    /// plausible. [`Bits`] is the crate's newtype for exactly this quantity.
    pub fn set_element_width(&mut self, element_width: Bits) {
        self.element_width = element_width;
    }

    /// Replaces: e014_setTransferSet
    ///
    /// **014/384** `setTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122`
    /// (2L).
    ///
    /// ```cpp
    /// void setTransferSet(IntegerSetAttr transfer_set) {
    ///   transfer_set_ = transfer_set;
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING — a `protected` state setter.
    ///
    /// ⛔ ONE FIELD, TWO ROLES, AND THE OP DECIDES WHICH: it is the `load_set` of a load and the
    /// `store_set` of a store (`AccessDetails.cpp:303`, `:313`, `:450`, `:465`, `:866`, `:877`).
    /// *"Load or store set within a transfer"* (`hpp:201`) — the class holds one operand, so one
    /// field serves both.
    ///
    /// ⛔ IT IS THE GEOMETRY `total_elements` AND `extents` ARE DERIVED FROM:
    /// `constructExtentAndTotalElements` builds `FlatAffineValueConstraints` from it and composes the
    /// transfer order onto it (`:33-37`). An [`IntegerSet`] here, rather than MLIR's
    /// `IntegerSetAttr`, because this crate never wraps a type in an attribute to carry it.
    pub fn set_transfer_set(&mut self, transfer_set: IntegerSet) {
        self.transfer_set = transfer_set;
    }

    /// Replaces: e015_setTransferOrder
    ///
    /// **015/384** `setTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125`
    /// (2L).
    ///
    /// ```cpp
    /// void setTransferOrder(AffineMap transfer_order) {
    ///   transfer_order_ = transfer_order;
    /// }
    /// ```
    ///
    /// ⭐ IT EMITS NOTHING — a `protected` state setter.
    ///
    /// ⛔ THE `load_order`/`store_order` OF THE SAME OPERAND, alongside [`Self::set_transfer_set`]
    /// at every writer (`AccessDetails.cpp:304`, `:314`, `:451`, `:466`, `:867`, `:878`) —
    /// *"Load order or store order within a transfer"* (`hpp:204`).
    ///
    /// ⛔ ITS DIMENSION COUNT IS READ, NOT JUST ITS RESULTS.
    /// `constructExtentAndTotalElements` composes it onto the transfer set and then projects out
    /// `transfer_order.getNumDims()` variables starting at `getNumDims()` (`:39-40`), and
    /// `MutableStartAddrShifting.cpp:468` composes it with the subscripts map. So an
    /// [`AffineMap`] carrying both `dims` and `results` is the whole value, not a convenience.
    pub fn set_transfer_order(&mut self, transfer_order: AffineMap) {
        self.transfer_order = transfer_order;
    }

    /// Replaces: e143_constructExtentAndTotalElements
    ///
    /// **143/384** `AccessDetailsBase::constructExtentAndTotalElements` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32` (64L).
    ///
    /// # ⭐⭐ THE EXTENT OF EVERY DIMENSION, READ OFF THE TRANSFER SET'S OWN MATRIX
    ///
    /// A `load_set`/`store_set` is an [`IntegerSet`] over the transfer's data dimensions and a
    /// `load_order`/`store_order` is the [`AffineMap`] that says which order those dimensions are
    /// walked in. This function re-expresses the set in the ORDER'S input space and then reads one
    /// extent per dimension out of the result:
    ///
    /// ```cpp
    /// affine::FlatAffineValueConstraints transfer_set_csts(getTransferSet().getValue());
    /// auto transfer_order = getTransferOrder();
    /// if (transfer_set_csts.composeMatchingMap(transfer_order).failed())
    ///   return LogicalResult::failure();
    /// transfer_set_csts.projectOut(transfer_order.getNumDims(), transfer_order.getNumDims());
    /// transfer_set_csts.removeRedundantConstraints();
    /// ```
    /// (`:33-41`) — [`FlatConstraints::compose_matching_map`], [`FlatConstraints::project_out`] and
    /// [`FlatConstraints::remove_redundant_constraints`], which this batch added to the island for
    /// this function.
    ///
    /// ⛔⛔ THE `removeRedundantConstraints()` IS NOT TIDYING. `projectOut` substitutes each tied
    /// dimension away using the equality that ties it, and `isDimValueZero` answers TRUE for a
    /// dimension as soon as it finds ANY equality naming no other variable — an all-zero row included.
    /// A single leftover `0 == 0` would therefore report every dimension as pinned, every extent as
    /// 1, and a transfer of one element. See [`FlatConstraints::is_dim_value_zero`].
    ///
    /// ⛔ `composeMatchingMap().failed()` HAS NO VARIANT HERE. MLIR's only failure path is
    /// `flattenAlignedMapAndMergeLocals` refusing a map it cannot flatten — a semi-affine one, with a
    /// multiplication or division by a non-constant. A transfer order is a permutation of dimensions,
    /// and the island's flattener names that case with its own `todo!` rather than an outcome
    /// (`islands/dataflow_ir/ty.rs`, `flattened`). The dimension-count mismatch MLIR *asserts* on is
    /// covered too: a short result list leaves surplus dimensions untied, `project_out` eliminates
    /// them by Fourier–Motzkin instead of by substitution, and what comes out is a system
    /// [`TransferExtents::NotHyperRectangular`] already refuses.
    ///
    /// # ⛔ ROW MAJOR IS DECIDED BY COMPARING THE OUTERMOST STRIDE WITH THE INNERMOST
    ///
    /// ```cpp
    /// auto layout_size = layout_coeffs.size();
    /// auto last_index = layout_size != transfer_set_csts.getNumDimVars() ? layout_size - 2
    ///                                                                   : layout_size - 1;
    /// bool is_row_major = (layout_coeffs[0] > layout_coeffs[last_index]);
    /// ```
    /// (`:47-51`). `getMapCoefficients` flattens the layout map into one stride per dimension PLUS a
    /// trailing constant term (see [`LayoutCoeff`]), so `layout_size` is one MORE than the number of
    /// dimensions whenever the constant column is there — which is what the `!=` test detects, and why
    /// it then steps back TWO to reach the last real stride. `[0] > [last]` says the first dimension
    /// moves the address furthest, which is row major.
    ///
    /// ⛔ THE COMPARISON IS THROUGH [`Option`] AND NOT AN INDEX. `layout_coeffs[0]` on an empty
    /// coefficient list is the reference's own out-of-bounds read; `first() > get(last_index)`
    /// answers `false` — column major — for the layout with no strides at all, whose orientation is
    /// unobservable because no dimension moves the address. It is unreachable in any case:
    /// [`Self::initialize_mem_view_info`] only ever installs a real view's layout map.
    ///
    /// # ⛔ THE ACCUMULATOR STARTS AT THE FIELD, NOT AT ZERO
    ///
    /// `auto total_elements = getTotalElements();` (`:54`) reads `total_elements_` back, and every
    /// step is written `(total_elements == 0) ? width : total_elements * width` — so a zero field
    /// SEEDS the product rather than annihilating it, and a second call on the same object multiplies
    /// into the count the first one left. Transcribed as written.
    ///
    /// ⭐ THE PINNED CASE MULTIPLIES BY ONE, WHICH THE REFERENCE SPELLS OUT: `total_elements =
    /// (total_elements == 0) ? 1 : total_elements * 1;` (`:59`). Only the seeding half is observable.
    ///
    /// ⛔ THE LIMIT'S DIVISION IS `checked_div`, BECAUSE A ZERO STRIDE IS DIVIDING BY ZERO. A
    /// dimension the address does not read has stride 0 — `(d0, d1) -> (d1)` flattens to `[0, 1, 0]` —
    /// and `layout_coeffs[i - 1] / layout_coeffs[i]` on it faults in the reference. With no ratio
    /// there is no limit to compare against, so the [`TransferExtents::ExtentOverLimit`] test is the
    /// one thing skipped; every extent is still pushed and still multiplied in.
    pub fn construct_extent_and_total_elements(&mut self) -> TransferExtents {
        let transfer_order = self.transfer_order.clone();
        let transfer_set_csts = FlatConstraints::from_integer_set(&self.transfer_set)
            .compose_matching_map(&transfer_order)
            .project_out(transfer_order.dims, transfer_order.dims)
            .remove_redundant_constraints();

        // TODO: currently, we do support only row major and column major in lowering. (the
        // reference's own note, `:43`)
        let Some(layout_map) = self.mem_view_layout_map.clone() else {
            return TransferExtents::MemoryViewUnresolved;
        };
        let layout_coeffs: Vec<LayoutCoeff> = layout_map
            .coefficients(0)
            .into_iter()
            .map(LayoutCoeff)
            .collect();
        self.set_layout_coeffs(&layout_coeffs);
        let stride = |position: usize| layout_coeffs.get(position).map(|coeff| coeff.0);
        let layout_size = layout_coeffs.len();
        let dim_vars = transfer_set_csts.num_dim_vars();
        let last_index = if layout_size == dim_vars as usize {
            layout_size.saturating_sub(1)
        } else {
            layout_size.saturating_sub(2)
        };
        let is_row_major = layout_coeffs.first() > layout_coeffs.get(last_index);

        let mut total_elements = self.total_elements;
        let mut extents: Vec<Elements> = Vec::new();
        for position in 0..dim_vars {
            let dim = TransferDim(position);
            if transfer_set_csts.is_dim_value_zero(position) {
                extents.push(Elements(1));
                if total_elements == Elements(0) {
                    total_elements = Elements(1);
                }
            } else if let Some(width) = transfer_set_csts.is_dim_a_constant_range(position) {
                if width <= 0 {
                    return TransferExtents::ExtentNotPositive { dim, width };
                }
                extents.push(Elements(width.unsigned_abs()));
                total_elements = if total_elements == Elements(0) {
                    Elements(width.unsigned_abs())
                } else {
                    Elements(total_elements.0 * width.unsigned_abs())
                };

                let index = dim.index();
                let limit = if is_row_major {
                    if index == 0 {
                        Some(width + 1)
                    } else {
                        ratio(stride(index - 1), stride(index))
                    }
                } else if index == last_index {
                    Some(width + 1)
                } else {
                    ratio(stride(index + 1), stride(index))
                };
                if let Some(limit) = limit
                    && width > limit
                {
                    return TransferExtents::ExtentOverLimit { dim, width, limit };
                }
            } else {
                return TransferExtents::NotHyperRectangular { dim };
            }
        }
        self.set_total_elements(total_elements);
        self.set_extents(&extents);

        if self.expected_total_elements != total_elements {
            return TransferExtents::ElementCountMismatch {
                expected: self.expected_total_elements,
                found: total_elements,
            };
        }
        TransferExtents::Rectangular
    }

    /// Replaces: e206_constructChunkAndShuffleInfo
    ///
    /// **206/384** `AccessDetailsBase::constructChunkAndShuffleInfo` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98` (167L). The contiguous run the
    /// hardware issues per instruction, the distance between two such runs, and — for a load whose
    /// result feeds a `vectorchain` reshuffle — the shuffle mode and rotation amount.
    ///
    /// ⛔ IT WORKS ON A ROW-MAJOR **COPY**: a column-major layout is reversed all but its trailing
    /// constant term, and the extents with it (`:111-123`); the members keep their own order.
    /// ⛔ THE TWO `dim_idx` CURSORS START AT `-1` AND THE REFERENCE THEN INDEXES WITH ONE (`:191`) —
    /// [`None`] here, and the legality test it guards is skipped, which is observably the same because
    /// `chunk_stride` is 0 in exactly that case and the `&&` fails.
    /// ⛔ `dim == 0`'S MULTIPLIER IS `INT32_MAX`, VERBATIM (`:147-149`): the outermost dimension has no
    /// stride above it, so nothing can match and the loop always breaks there. Its `*=` overflows an
    /// `int` in the reference and is unobservable, because no later loop reads the product.
    /// ⛔ A ZERO STRIDE IS A DIVISION BY ZERO THERE; with no ratio there is no multiplier for the
    /// extent to fit under, so [`ChunkAndShuffleInfo::ExtentLargerThanLayout`] refuses it.
    /// ⛔ ONLY `agen.vector_load` REACHES THE SHUFFLE HALF: the reference's `isa<>` (`:205-206`) lists
    /// the two vector loads and the two composite LOADS, and of those four the island has one —
    /// `CompositeLoadAndStore` is deliberately NOT in that list.
    pub fn construct_chunk_and_shuffle_info(&mut self, scope: &[DfirOp]) -> ChunkAndShuffleInfo {
        // `layout_coeffs[extents.size() - 1] > layout_coeffs[0]` (`:111`) — the innermost stride
        // moving furthest is column major. Through `Option` because both indices are the reference's
        // own out-of-bounds reads on an empty list.
        let innermost = self
            .extents
            .len()
            .checked_sub(1)
            .and_then(|last| self.layout_coeffs.get(last));
        let is_column_major = match (innermost, self.layout_coeffs.first()) {
            (Some(inner), Some(outer)) => inner > outer,
            _ => false,
        };
        let (tmp_layout_coeffs, tmp_extents): (Vec<LayoutCoeff>, Vec<Elements>) = if is_column_major
        {
            // `:112-122` — the strides reversed with the constant term put back on the end.
            let mut coeffs: Vec<LayoutCoeff> = Vec::new();
            if let Some((constant, strides)) = self.layout_coeffs.split_last() {
                coeffs.extend(strides.iter().rev().copied());
                coeffs.push(*constant);
            }
            (coeffs, self.extents.iter().rev().copied().collect())
        } else {
            (self.layout_coeffs.clone(), self.extents.clone())
        };

        // `:127-130` — the layout carries one extra dimension of constant, and the innermost stride of
        // a row-major layout is 1.
        let innermost_stride = tmp_extents
            .len()
            .checked_sub(1)
            .and_then(|last| tmp_layout_coeffs.get(last));
        if tmp_layout_coeffs.len() != tmp_extents.len() + 1
            || innermost_stride != Some(&LayoutCoeff(1))
        {
            return ChunkAndShuffleInfo::LayoutDoesNotMatchExtents;
        }

        // `int multiplier = dim == 0 ? INT32_MAX : coeffs[dim - 1] / coeffs[dim];` (`:147-149`, `:173-175`)
        let multiplier = |dim: usize| {
            if dim == 0 {
                Some(i64::from(i32::MAX))
            } else {
                ratio(
                    tmp_layout_coeffs.get(dim - 1).map(|coeff| coeff.0),
                    tmp_layout_coeffs.get(dim).map(|coeff| coeff.0),
                )
            }
        };
        let extent_at = |dim: usize| tmp_extents.get(dim).copied().unwrap_or(Elements(0));

        // `:138-165` — the contiguous run, innermost dimension outward.
        let mut chunk_dim_idx: Option<usize> = None;
        let mut layout_extent_multiplier = Elements(1);
        let mut chunk_size = Elements(0);
        for dim in (0..tmp_extents.len()).rev() {
            let extent = extent_at(dim);
            let Some(multiplier) = multiplier(dim).filter(|found| *found >= 0) else {
                return ChunkAndShuffleInfo::ExtentLargerThanLayout { dim };
            };
            let multiplier = Elements(multiplier.unsigned_abs());
            if multiplier < extent {
                return ChunkAndShuffleInfo::ExtentLargerThanLayout { dim };
            }
            if multiplier != extent {
                chunk_dim_idx = Some(dim);
                chunk_size = Elements(layout_extent_multiplier.0.saturating_mul(extent.0));
                layout_extent_multiplier =
                    Elements(layout_extent_multiplier.0.saturating_mul(multiplier.0));
                break;
            }
            layout_extent_multiplier =
                Elements(layout_extent_multiplier.0.saturating_mul(multiplier.0));
        }
        self.chunk_size = chunk_size;

        // `:169-188` — the chunk stride is the running extent product at the first non-unit extent
        // ABOVE the chunk dimension.
        let mut chunk_stride = Elements(0);
        let mut chunk_stride_dim_idx: Option<usize> = None;
        for dim in (0..chunk_dim_idx.unwrap_or(0)).rev() {
            let extent = extent_at(dim);
            let Some(multiplier) = multiplier(dim).filter(|found| *found >= 0) else {
                return ChunkAndShuffleInfo::ExtentLargerThanLayout { dim };
            };
            let multiplier = Elements(multiplier.unsigned_abs());
            if multiplier < extent {
                return ChunkAndShuffleInfo::ExtentLargerThanLayout { dim };
            }
            if extent != Elements(1) {
                chunk_stride = layout_extent_multiplier;
                chunk_stride_dim_idx = Some(dim);
                break;
            }
            layout_extent_multiplier =
                Elements(layout_extent_multiplier.0.saturating_mul(multiplier.0));
        }
        self.chunk_stride = chunk_stride;

        // `:191-197` — a chunk that sits at the innermost stride and yet is one of several.
        if let Some(chunk_dim) = chunk_dim_idx
            && tmp_layout_coeffs.get(chunk_dim) == Some(&LayoutCoeff(1))
            && chunk_stride != Elements(0)
        {
            return ChunkAndShuffleInfo::ChunkStartingOffsetUnsupported;
        }

        // `:200-203` — a second, outer chunk stride.
        for dim in (0..chunk_stride_dim_idx.unwrap_or(0)).rev() {
            if extent_at(dim) != Elements(1) {
                return ChunkAndShuffleInfo::VariableChunkStride { dim };
            }
        }

        // `:207-218` — the users of the loaded vector.
        let users = match self.op {
            agen::Op::VectorLoad { result, .. } => uses(*result, scope),
            agen::Op::VectorStore { .. }
            | agen::Op::CompositeLoadAndStore(_)
            | agen::Op::Yield
            // ⭐ A SAMV BINDS A VECTOR BUT LOADS NOTHING, so it has no loaded value to have users of.
            | agen::Op::SetTransferMaskState { .. } => Vec::new(),
        };
        for user in users {
            match user {
                // `:220-241` — the selection map decides the shuffle mode.
                DfirOp::VectorChain(vectorchain::Op::Select { selection_map, .. }) => {
                    let mut splat = false;
                    let mut valid_select_map = true;
                    selection_map.walk_exprs(&mut |expr| match expr {
                        AffineExpr::Mod(..) => splat = true,
                        // The constant arm's `DT_CHECK(input_size == modulo)` is commented out
                        // (`:232`), and a bare `d<n>` is the map's own identity.
                        AffineExpr::Const(_) | AffineExpr::Dim(_) => {}
                        _ => valid_select_map = false,
                    });
                    if splat {
                        self.set_shuffle_mode(sen::ShuffleMode::Splat);
                    }
                    if !valid_select_map {
                        return ChunkAndShuffleInfo::InvalidSelectMap;
                    }
                }
                // `:242-261` — the rotation amount has to be a constant, forward, and in range.
                DfirOp::VectorChain(vectorchain::Op::Rotate {
                    position,
                    right_shift,
                    ..
                }) => {
                    let Some(DfirOp::Arith(arith::Op::Constant { value, .. })) =
                        defining_op(*position, scope)
                    else {
                        return ChunkAndShuffleInfo::RotationPositionNotConstant;
                    };
                    if *value < 0 {
                        return ChunkAndShuffleInfo::RotationPositionNegative;
                    }
                    self.set_rotation_position(Elements(value.unsigned_abs()));
                    if self.rotation_position > self.total_elements {
                        return ChunkAndShuffleInfo::RotationPositionOverTotalElements;
                    }
                    if !right_shift {
                        return ChunkAndShuffleInfo::LeftRotationUnsupported;
                    }
                }
                _ => {}
            }
        }
        ChunkAndShuffleInfo::Chunked
    }

    /// Replaces: e144_constructLdOrStType
    ///
    /// **144/384** `AccessDetailsBase::constructLdOrStType` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267` (5L).
    ///
    /// ```cpp
    /// void AccessDetailsBase::constructLdOrStType() {
    ///   if (is_any_of(getComp(), L0LU, L0SU))
    ///     setLdOrStSize(getChunkSize());
    ///   else
    ///     setLdOrStSize(getExpectedTotalElements());
    /// }
    /// ```
    ///
    /// ⭐⭐ THE WHOLE FUNCTION IS THE `hpp:209-210` COMMENT MADE EXECUTABLE: *"In case of L0, ld size
    /// refers to chunk size. In other cases, ld size refers to total_elements."* An L0 load/store
    /// unit issues ONE CHUNK per instruction, so the size its `ldtype`/`sttype` field carries is the
    /// chunk; every other component issues the whole transfer.
    ///
    /// ⛔ AND IT IS `expected_total_elements_`, NOT `total_elements_`, DESPITE THAT COMMENT. The
    /// comment says "total_elements"; the code reads the *expected* count — the one
    /// [`Self::set_expected_total_elements`] takes from the op's RETURN TYPE (`hpp:186-187`) rather
    /// than the one [`Self::construct_extent_and_total_elements`] derives from the transfer set. They
    /// are equal on every accepted access, because that function refuses with
    /// [`TransferExtents::ElementCountMismatch`] when they differ (`:90-93`) — which is exactly why
    /// reading either is safe here and why the code's choice is the one to transcribe.
    ///
    /// ⛔ IT RETURNS `void`, ALONE AMONG THE `construct*` MEMBERS (`hpp:147`). There is no outcome to
    /// carry: both branches assign, neither can fail.
    ///
    /// ⭐ `is_any_of(getComp(), L0LU, L0SU)` IS A [`matches!`], and the closed set is [`DfirUnit`].
    /// L0's load unit and store unit are the two components with a chunked instruction; `L3LU`/`L3SU`
    /// — the pair [`AccessDetailsAffineComposite::compute_burst_and_group`] singles out — are not
    /// among them.
    pub fn construct_ld_or_st_type(&mut self) {
        self.ld_or_st_size = if matches!(self.comp, DfirUnit::L0lu | DfirUnit::L0su) {
            self.chunk_size
        } else {
            self.expected_total_elements
        };
    }

    /// Replaces: e145_initializeMemViewInfo
    ///
    /// **145/384** `AccessDetailsBase::initializeMemViewInfo` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274` (16L).
    ///
    /// ⭐⭐ THREE FACTS OFF THE VIEW, IN THE REFERENCE'S ORDER — layout map, then start address, then
    /// unit. See [`MemViewSource`] for the two `dyn_cast` arms this collapses and for why
    /// `llvm_unreachable("unsupported mem_ref type")` (`:287`) has nothing left to guard.
    ///
    /// ⛔ THE `LogicalResult` IS NOT AN OUTCOME. Every path through the reference reaches
    /// `return LogicalResult::success()` (`:288`) — the third arm aborts the process instead of
    /// returning failure — so there is exactly one outcome and it is `()`.
    ///
    /// ⛔ IT TAKES THE VIEW RATHER THAN READING `mem_ref_`, and that is where the abort went. The
    /// reference opens with `auto mem_ref = getMemRef().getDefiningOp();` (`:275`); here the caller
    /// resolves it with [`MemViewSource::of`] and can only proceed with a view that exists. The
    /// `mem_ref_` field itself belongs to `initialize()`, which is the entry that writes it.
    ///
    /// ⛔ THE ORDER OF THE THREE WRITES IS KEPT even though they touch different members, because a
    /// later entry may come to read one of them mid-flight; reordering would be a change nothing in
    /// the reference justifies.
    pub fn initialize_mem_view_info(&mut self, view: MemViewSource<'_>) {
        self.set_mem_view_layout_map(view.layout.clone());
        self.set_mem_view_start_addr(view.start_address);
        self.memory = Some(view.memory);
    }
}

/// THE RATIO BETWEEN TWO NEIGHBOURING LAYOUT STRIDES, or [`None`] where the reference divides by
/// zero or reads past its coefficient list.
///
/// ⭐ EXTRACTED SO THE FOUR CALLS IN
/// [`AccessDetailsBase::construct_extent_and_total_elements`] read as the reference's two ternaries
/// do. `layout_coeffs[i - 1] / layout_coeffs[i]` is an `int64_t` division there
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:66-70`), truncating toward zero, which is
/// what [`i64::checked_div`] does.
fn ratio(numerator: Option<i64>, denominator: Option<i64>) -> Option<i64> {
    numerator?.checked_div(denominator?)
}

/// AN ACCESS WHOSE SUBSCRIPTS ARE AFFINE — `AccessDetailsAffine` (`AccessDetails.hpp:214-245`).
///
/// The `agen.vector_load`/`agen.vector_store` case, and the indirect pair: subscripts come from an
/// affine map over loop induction variables rather than from symbolic strides
/// (`AccessDetails.cpp:295-351`).
///
/// ⛔ COMPOSITION, NOT INHERITANCE — see [`AccessDetailsBase`]. `subscripts_map_` and
/// `indices_coeff_dict_` (`hpp:241-244`) arrive with their setters, entries 017 and 018.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessDetailsAffine<'a> {
    /// The `AccessDetailsBase` this derives from.
    pub base: AccessDetailsBase<'a>,
    /// `subscripts_map_` — the affine map holding the subscripts (`AccessDetails.hpp:241`).
    ///
    /// ⛔ `None` IS C++'S DEFAULT-CONSTRUCTED NULL `AffineMap`, NOT AN EMPTY ONE. A null `AffineMap`
    /// is a pointer that `getNumResults()` cannot be called on; `affine_map<() -> ()>` is a legal map
    /// of no results that it can. Substituting the empty map for the unset state would be a stand-in
    /// op, and `MutableStartAddrShifting.cpp:411` loops over `subscripts_map.getNumResults()`.
    pub subscripts_map: Option<AffineMap>,
    /// `indices_coeff_dict_` — the loop iterators involved with their coefficients
    /// (`AccessDetails.hpp:244`).
    pub indices_coeff_dict: IndicesCoeffDict,
}

/// THE OUTCOME OF [`AccessDetailsAffine::initialize`] — initialized, or WHY the op said nothing.
///
/// ⛔ NONE OF THE THREE IS A DIAGNOSTIC. The reference's `initialize()` returns `void` and reaches
/// `DT_ERROR("unsupported operation")` (`AccessDetails.cpp:346`) — an ABORT — on an op outside its
/// chain, and reads `getMemoryIndex()` with the memory index already set by `insert`'s caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum AffineInitialize {
    /// Every field is set and the memory view is resolved (`:349`).
    Initialized,
    /// `memory_index_` was never set, so the record does not know which operand it is.
    MemoryIndexUnset,
    /// `DT_ERROR("unsupported operation")` (`:346`) — a composite, a yield, or one of the two
    /// `indirect_` twins the island has no op for.
    UnsupportedOperation,
    /// `mem_ref_` does not trace back to a `dataflow.get_*_memory_view` in this scope.
    MemoryViewUnresolved,
}

impl AffineInitialize {
    /// Initialized only for [`Self::Initialized`].
    #[must_use]
    pub const fn admissible(self) -> bool {
        matches!(self, AffineInitialize::Initialized)
    }

    /// The reference's abort message, verbatim (`AccessDetails.cpp:346`).
    #[must_use]
    pub const fn diagnostic(self) -> Option<&'static str> {
        match self {
            AffineInitialize::Initialized
            | AffineInitialize::MemoryIndexUnset
            | AffineInitialize::MemoryViewUnresolved => None,
            AffineInitialize::UnsupportedOperation => Some("unsupported operation"),
        }
    }
}

impl<'a> AccessDetailsAffine<'a> {
    /// Replaces: e016_AccessDetailsBase
    ///
    /// **016/384** `AccessDetailsBase` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218`
    /// (0L).
    ///
    /// ```cpp
    /// explicit AccessDetailsAffine(mlir::Operation* op, SenComponents comp)
    ///     : AccessDetailsBase(op, comp) {}
    /// ```
    ///
    /// ⛔ THE UNIT IS THE **`AccessDetailsAffine`** CONSTRUCTOR, DESPITE ITS NAME. The banner names
    /// the entry after the token the extract found at the cited line, and `hpp:218` is the
    /// mem-initializer line `: AccessDetailsBase(op, comp) {}` — so `e016_AccessDetailsBase` is the
    /// DERIVED constructor delegating to the base, and the base's own constructor (`hpp:46-47`) is
    /// the excluded entry that the extractor recorded as the member `comp_`. Entry 019
    /// (`e019_AccessDetailsAffine`, `hpp:259`) is the same pattern one level down: the
    /// `AccessDetailsAffineComposite` constructor delegating to THIS one.
    ///
    /// ⭐ ITS WHOLE BODY IS THE DELEGATION: it forwards both arguments and leaves every other
    /// member — the base's and its own — at its declared default. `explicit` has no Rust
    /// counterpart; a named constructor is never an implicit conversion.
    #[must_use]
    pub fn new(op: &'a agen::Op, comp: DfirUnit) -> Self {
        AccessDetailsAffine {
            base: AccessDetailsBase::new(op, comp),
            subscripts_map: None,
            indices_coeff_dict: IndicesCoeffDict::default(),
        }
    }

    /// Replaces: e017_setSubscriptsMap
    ///
    /// **`setSubscriptsMap`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228-230` (3L).
    ///
    /// ```text
    /// void setSubscriptsMap(AffineMap subscripts_map) {
    ///   subscripts_map_ = subscripts_map;
    /// }
    /// ```
    ///
    /// ⭐ AN `AffineMap` IS ALREADY A HANDLE IN C++ — a uniqued, immutable, pointer-sized value — so
    /// passing it by value there and moving our own owned [`AffineMap`] here are the same operation.
    ///
    /// ⛔ A SET, NEVER A CLEAR, SO IT TAKES AN [`AffineMap`] AND NOT AN `Option`. Every callsite
    /// hands it a map the op itself carries — `setSubscriptsMap(load_op.getAffineMap())`
    /// (`AccessDetails.cpp:305`), `composite_load_and_store_op.getSrcAffineMapAttr().getValue()`
    /// (`:528`) or its `getDst` twin (`:539`) — and the one remaining callsite re-installs the map it
    /// just read after folding constant operands out of it (`:398-401`). None of them can be null,
    /// and the unset state belongs to the constructor alone.
    pub fn set_subscripts_map(&mut self, subscripts_map: AffineMap) {
        self.subscripts_map = Some(subscripts_map);
    }

    /// Replaces: e018_setIndicesCoeffDict
    ///
    /// **`setIndicesCoeffDict`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231-233` (3L).
    ///
    /// ```text
    /// void setIndicesCoeffDict(DenseMap<Value, int64_t>& indices_coeff_dict) {
    ///   indices_coeff_dict_ = indices_coeff_dict;
    /// }
    /// ```
    ///
    /// ⭐ C++ TAKES A MUTABLE REFERENCE AND THEN COPY-ASSIGNS FROM IT — the reference is non-const
    /// only because `DenseMap`'s `operator[]` is, and the caller's map is dead after the call
    /// (`AccessDetails.cpp:411-413`, which hands over the dictionary
    /// `constructIteratorCoeffDict` just returned and then returns itself). Taking
    /// [`IndicesCoeffDict`] by value is that copy without the copy.
    ///
    /// ⛔ WHOLESALE REPLACEMENT, NOT A MERGE. `operator=` on a `DenseMap` drops every existing entry,
    /// so a second call cannot leave a coefficient from the first behind.
    pub fn set_indices_coeff_dict(&mut self, indices_coeff_dict: IndicesCoeffDict) {
        self.indices_coeff_dict = indices_coeff_dict;
    }

    /// Replaces: e207_initialize
    ///
    /// **207/384** `AccessDetailsAffine::initialize` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295` (57L). Everything one affine load or
    /// store says about itself, read off the op and into the record, then the memory view.
    ///
    /// ⛔ FOUR ARMS, TWO ISLAND OPS. The reference's `dyn_cast` chain (`:301-344`) takes
    /// `agen.vector_load`/`agen.vector_store` and their `indirect_` twins; the indirect pair differs
    /// only in reading `getDirectMemref()`/`getDirectAffineMapAttr()` and has no island op yet, so it
    /// lands on [`AffineInitialize::UnsupportedOperation`] with the composite and the yield.
    /// ⛔ THE WIDTH AND THE COUNT COME FROM THE **VECTOR** OPERAND, not from the memref: the loaded
    /// result on a load, the stored value on a store (`:305-310`, `:316-321`).
    /// ⛔ AND `getMapIndices()`/`getMapOperands()` ARE THE MAP'S OPERANDS, so [`access_map`] hands back
    /// both the subscripts map and the index list in one pass — one `d<i>` per DISTINCT operand.
    pub fn initialize(&mut self, scope: &[DfirOp]) -> AffineInitialize {
        if self.base.memory_index.is_none() {
            return AffineInitialize::MemoryIndexUnset;
        }

        let (view, view_ty, indices, vector_ty) = match self.base.op {
            agen::Op::VectorLoad {
                view,
                view_ty,
                indices,
                ty,
                ..
            }
            | agen::Op::VectorStore {
                view,
                view_ty,
                indices,
                ty,
                ..
            } => (view, view_ty, indices, ty),
            agen::Op::CompositeLoadAndStore(_)
            | agen::Op::Yield
            // ⭐ A SAMV CARRIES NO VIEW AND NO SUBSCRIPTS.
            | agen::Op::SetTransferMaskState { .. } => {
                return AffineInitialize::UnsupportedOperation;
            }
        };
        let (subscripts_map, map_indices) = access_map(indices);
        self.base.mem_ref = Some(*view);
        self.base
            .set_transfer_set(agen::access_set(view_ty, vector_ty.len));
        self.base
            .set_transfer_order(agen::access_order(view_ty.shape.len()));
        self.set_subscripts_map(subscripts_map);
        self.base.set_element_width(Bits(vector_ty.elem.bits()));
        self.base
            .set_expected_total_elements(Elements(vector_ty.len));
        self.base.set_indices(&map_indices);

        // `initializeMemViewInfo()` (`:349`) reads `mem_ref_` back; entry 145 takes the resolved view.
        let Some(source) = MemViewSource::of(*view, scope) else {
            return AffineInitialize::MemoryViewUnresolved;
        };
        self.base.initialize_mem_view_info(source);
        AffineInitialize::Initialized
    }

    /// Replaces: e146_constructIndices
    ///
    /// **146/384** `AccessDetailsAffine::constructIndices` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354` (50L).
    ///
    /// # ⭐⭐ IT SEPARATES THE SUBSCRIPTS THAT VARY FROM THE ONES THAT DO NOT
    ///
    /// `indices_` arrives from `initialize()` holding every operand of the access's affine map. This
    /// function keeps only the loop iterators among them and FOLDS each constant operand into the
    /// subscripts map itself, so that afterwards `indices_` and the map's dimensions are the same
    /// list in the same order — which is the invariant
    /// [`Self::construct_iterator_coefficients`] then indexes by position and
    /// `MutableAddrSplitting.cpp:717` checks by length.
    ///
    /// ```cpp
    /// for (int dim = 0; dim < indices.size(); dim++) {
    ///   auto& index = indices[dim];
    ///   if (mlir::isa<BlockArgument>(index)) {
    ///     auto* loop_op = mlir::cast<BlockArgument>(index).getOwner()->getParentOp();
    ///     if (isa<affine::AffineForOp, scf::ForOp>(loop_op))
    ///       new_indices.push_back(index);
    ///     else
    ///       return op->emitError("The loop iterators involved in the agen memory "
    ///                            "operation subscripts have to be affine loops");
    ///   } else if (auto const_op = index.getDefiningOp<mlir::arith::ConstantOp>()) {
    ///     auto val = mlir::cast<IntegerAttr>(const_op.getValue()).getInt();
    ///     dim_to_cst_val_map[dim] = val;
    ///   } else {
    ///     return op->emitError("All the map operands need to be either loop"
    ///                          "iterators or constant values");
    ///   }
    /// }
    /// setIndices(new_indices);
    /// ```
    /// (`:360-380`) — the classification is [`SubscriptOperand`].
    ///
    /// ⛔⛔ BOTH REFUSALS RETURN **BEFORE** `setIndices`, so a refused access keeps the index list it
    /// arrived with, constants and all. That is load-bearing: `constructDetails` propagates the
    /// failure and the op is never lowered, but nothing clears the object, and a half-filtered list
    /// would be an access at the wrong offset if anything read it. Writing the list only on the
    /// success path is the reference's own ordering, not a tidy-up.
    ///
    /// # ⛔ THE FOLD RENUMBERS THE SURVIVORS AND DECLARES THE NEW ARITY ITSELF
    ///
    /// ```cpp
    /// if (!dim_to_cst_val_map.empty()) {
    ///   int ndims = 0;
    ///   for (int dim = 0; dim < indices.size(); dim++) {
    ///     auto record = dim_to_cst_val_map.find(dim);
    ///     if (record == dim_to_cst_val_map.end())
    ///       operand_exprs.push_back(getAffineDimExpr(ndims++, op->getContext()));
    ///     else
    ///       operand_exprs.push_back(getAffineConstantExpr(record->second, op->getContext()));
    ///   }
    ///   subscripts_map = subscripts_map.replaceDimsAndSymbols(operand_exprs, symbol_exprs, ndims, 0);
    /// }
    /// setSubscriptsMap(subscripts_map);
    /// ```
    /// (`:384-403`). ⛔ `ndims` COUNTS ONLY THE SURVIVORS and is passed as the result map's dimension
    /// count, so the folded dimensions' columns disappear rather than staying as unused ones — see
    /// [`AffineMap::replace_dims_and_symbols`], whose doc records why declaring the arity matters to
    /// every matrix built from the map afterwards.
    ///
    /// ⛔ `symbol_exprs` IS EMPTY AND `0` IS PASSED FOR THE SYMBOL COUNT, so a subscripts map with
    /// symbols would lose them. The reference relies on affine subscripts having none.
    ///
    /// ⭐ THE `dim_to_cst_val_map` IS A `std::map<int, int>` KEYED BY POSITION, which is what makes
    /// the second loop able to walk the ORIGINAL positions while `ndims` counts the new ones. A
    /// `Vec<Option<i64>>` indexed by position is the same structure with the lookup made positional;
    /// ⚠️ the reference's value type is `int` while `getInt()` returns `int64_t`, so a subscript
    /// constant wider than 32 bits truncates there. Not copied: an index constant that large is not a
    /// contract worth reproducing, and the folded value is carried as [`i64`] here.
    ///
    /// ⛔ THE UNCONDITIONAL `setSubscriptsMap` AT THE END IS A NO-OP WHEN NOTHING WAS FOLDED — it
    /// writes back the map it just read. With the map still unset it writes null over null, which is
    /// why leaving [`None`] alone is the same operation.
    pub fn construct_indices(&mut self, scope: &[DfirOp]) -> ConstructedIndices {
        // Step-1: Construct indices (`:356`)
        let indices = self.base.indices.clone();
        let mut folded: Vec<Option<i64>> = vec![None; indices.len()];
        let mut new_indices: Vec<Val> = Vec::new();
        for (position, index) in indices.iter().copied().enumerate() {
            match SubscriptOperand::of(index, scope) {
                SubscriptOperand::LoopIterator => new_indices.push(index),
                SubscriptOperand::ForeignRegionArgument => {
                    return ConstructedIndices::NotAnAffineLoopIterator { index };
                }
                SubscriptOperand::Constant(value) => folded[position] = Some(value),
                SubscriptOperand::Computed => {
                    return ConstructedIndices::NotAnIteratorOrConstant { index };
                }
            }
        }
        self.base.set_indices(&new_indices);

        // substitute constant indices into the subscripts map itself (`:383`)
        let mut subscripts_map = self.subscripts_map.clone();
        if folded.iter().any(Option::is_some) {
            subscripts_map = subscripts_map.as_ref().map(|map| {
                let mut ndims = 0;
                let operand_exprs: Vec<AffineExpr> = folded
                    .iter()
                    .map(|value| match value {
                        Some(value) => AffineExpr::Const(*value),
                        None => {
                            let expr = AffineExpr::dim(ndims);
                            ndims += 1;
                            expr
                        }
                    })
                    .collect();
                map.replace_dims_and_symbols(&operand_exprs, &[], ndims, 0)
            });
        }
        if let Some(subscripts_map) = subscripts_map {
            self.set_subscripts_map(subscripts_map);
        }

        ConstructedIndices::Affine
    }

    /// Replaces: e147_constructIteratorCoefficients
    ///
    /// **147/384** `AccessDetailsAffine::constructIteratorCoefficients` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406` (10L).
    ///
    /// ```cpp
    /// auto indices_coeff_dict = agen::utils::constructIteratorCoeffDict(
    ///     subscripts_map, transfer_order, mem_view_layout_map, indices);
    /// setIndicesCoeffDict(indices_coeff_dict);
    /// return success();
    /// ```
    ///
    /// ⭐ FOUR READS, ONE CALL AND ONE WRITE — the whole body. The composition it delegates to is
    /// [`construct_iterator_coeff_dict`], which is where the substance is.
    ///
    /// ⛔ `success()` UNCONDITIONALLY (`:414`), so there is no outcome: `constructIteratorCoeffDict`
    /// returns a dictionary, not a `LogicalResult`.
    ///
    /// ⛔ WITH EITHER MAP STILL UNSET THERE IS NOTHING TO COMPOSE. The reference would compose
    /// through a null `AffineMap` and dereference it; here the dictionary is the default —
    /// no per-index entries and a zero constant, which is what `indices_coeff_dict[nullptr] = 0`
    /// alone would leave. Unreachable on the reference's call order: `constructDetails` runs
    /// `initialize()` and [`Self::construct_indices`] first, and both maps are set by then
    /// (`:418-437`).
    pub fn construct_iterator_coefficients(&mut self) {
        let indices_coeff_dict = match (&self.subscripts_map, &self.base.mem_view_layout_map) {
            (Some(subscripts_map), Some(mem_view_layout_map)) => construct_iterator_coeff_dict(
                subscripts_map,
                &self.base.transfer_order,
                mem_view_layout_map,
                &self.base.indices,
            ),
            _ => IndicesCoeffDict::default(),
        };
        self.set_indices_coeff_dict(indices_coeff_dict);
    }
}

/// WHAT ONE SUBSCRIPT OF AN AFFINE ACCESS IS — the four cases `constructIndices` distinguishes
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:360-378`).
///
/// ⭐⭐ FOUR, NOT THREE. The reference's two nested tests produce four outcomes, and it treats all
/// four differently: a block argument of a loop is KEPT, a block argument of anything else is the
/// first refusal, a constant is FOLDED, and everything else is the second refusal. Collapsing the two
/// block-argument cases would lose the distinction the first `emitError` exists to report.
///
/// ⛔ THE CLASSIFICATION IS COMPLETE AND MUTUALLY EXCLUSIVE, because a value is either an op's result
/// or a region's argument — see [`region_owner`], which is the island half this batch added for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptOperand {
    /// A region argument of an `affine.for` or an `scf.for` — `isa<affine::AffineForOp, scf::ForOp>`
    /// on the argument's parent op (`:361-362`). Kept in `indices_`.
    ///
    /// ⛔⛔ A LOOP'S **CARRIED** ARGUMENT ANSWERS THIS TOO, and the reference means it to: the test is
    /// on the owning OP's type, not on whether the value is that loop's induction variable. An
    /// `iter_args` argument of an `affine.for` is therefore accepted as a subscript. Narrowing it to
    /// the induction variable would refuse accesses the reference lowers.
    LoopIterator,
    /// A region argument of any other op — the first refusal, *"The loop iterators involved in the
    /// agen memory operation subscripts have to be affine loops"* (`:363-365`).
    ///
    /// ⭐ `scf.parallel`'s INDUCTION VARIABLES LAND HERE, not in [`Self::LoopIterator`]: the
    /// reference names `scf::ForOp` and not `scf::ParallelOp`. So does an
    /// `agen.composite_load_and_store`'s `load_iv`.
    ForeignRegionArgument,
    /// Defined by an `arith.constant` — folded into the subscripts map and dropped from `indices_`
    /// (`:366-368`).
    Constant(i64),
    /// Defined by any other op — the second refusal, *"All the map operands need to be either loop
    /// iterators or constant values"* (`:370-372`).
    ///
    /// ⭐ AND BY NO OP AT ALL. A `Val` that neither an op in `scope` defines nor a region binds is a
    /// value from outside the scope handed in; the reference's `getDefiningOp<ConstantOp>()` returns
    /// null for it and takes the same branch.
    Computed,
}

impl SubscriptOperand {
    /// WHICH OF THE FOUR A SUBSCRIPT IS — the two nested `isa`/`dyn_cast` tests, in the reference's
    /// order.
    ///
    /// ⛔ THE BLOCK-ARGUMENT TEST COMES FIRST, and it must: `getDefiningOp()` on a block argument is
    /// null, so asking about the constant first would classify an induction variable as
    /// [`Self::Computed`] only if the null test were forgotten — and asking in the reference's order
    /// makes the question unaskable.
    #[must_use]
    pub fn of(index: Val, scope: &[DfirOp]) -> SubscriptOperand {
        if let Some(owner) = region_owner(index, scope) {
            return match owner {
                DfirOp::Affine(affine::Op::For { .. }) | DfirOp::Scf(scf::Op::For { .. }) => {
                    SubscriptOperand::LoopIterator
                }
                _ => SubscriptOperand::ForeignRegionArgument,
            };
        }
        match defining_op(index, scope) {
            Some(DfirOp::Arith(arith::Op::Constant { value, .. })) => {
                SubscriptOperand::Constant(*value)
            }
            _ => SubscriptOperand::Computed,
        }
    }
}

/// WHAT `constructIndices` FOUND — its `success()` and its two `emitError`s.
///
/// ⛔ THE TWO REFUSALS STAY APART because the reference reports them differently, and because they
/// mean different things to whoever reads the diagnostic: the first says the access is inside a loop
/// form this lowering does not handle, the second says a subscript is computed rather than iterated.
/// See [`SubscriptOperand`].
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstructedIndices {
    /// `success()` (`:404`) — every subscript was a loop iterator or a constant, `indices_` now holds
    /// the iterators alone and the subscripts map holds the constants.
    Affine,
    /// *"The loop iterators involved in the agen memory operation subscripts have to be affine
    /// loops"* (`:363-365`).
    NotAnAffineLoopIterator {
        /// The subscript that is a region argument of something other than an `affine.for`/`scf.for`.
        index: Val,
    },
    /// *"All the map operands need to be either loop iterators or constant values"* (`:370-372`).
    NotAnIteratorOrConstant {
        /// The subscript that is neither a region argument nor an `arith.constant`.
        index: Val,
    },
}

/// THE COEFFICIENT OF EVERY LOOP ITERATOR IN AN ACCESS'S ADDRESS — `constructIteratorCoeffDict`
/// (`dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Agen/Utils.cpp:60-85`).
///
/// # ⭐⭐ TWO COMPOSITIONS, THEN ONE FLATTENED ROW
///
/// ```cpp
/// auto composed_load_order_map = transfer_order.compose(subscripts_map);
/// auto composed_layout_map = mem_view_layout_map.compose(composed_load_order_map);
/// affine::fullyComposeAffineMapAndOperands(&composed_layout_map, &indices);
/// auto composed_layout_coeffs = constructDimCoefficients(composed_layout_map);
/// for (std::size_t i = 0; i < indices.size(); ++i)
///   indices_coeff_dict[indices[i]] = composed_layout_coeffs[i];
/// int const_value = composed_layout_coeffs.size() != composed_layout_map.getNumDims()
///                       ? composed_layout_coeffs.back() : 0;
/// indices_coeff_dict[nullptr] = const_value;
/// ```
///
/// The subscripts map turns loop iterators into tensor coordinates, the transfer order permutes those
/// coordinates, and the layout map turns them into a linear element address — so the flattened row of
/// the whole composition is, per iterator, HOW MANY ELEMENTS ONE STEP OF THAT LOOP MOVES THE ADDRESS.
/// That is what `MutableStartAddrShifting.cpp:402` and `MutableAddrSplitting.cpp:724` read it for.
///
/// ⛔ THE COMPOSITION ORDER IS INSIDE-OUT AND THE `compose` DIRECTION MATTERS.
/// `a.compose(b)` substitutes `b`'s results into `a`'s dimensions, so the map applied FIRST is the
/// argument: iterators → subscripts → order → layout. See [`AffineMap::compose`].
///
/// ⛔ THE `const_value` TERNARY ALWAYS TAKES `back()`. A flattened row is one entry per dimension,
/// per symbol and per local, PLUS the constant — so its length is at least `getNumDims() + 1` and
/// the `!=` can only be true. The `: 0` arm is dead in the reference and there is nothing to
/// transcribe for it; `back()` on the row is the constant term, and it is 0 when the address has no
/// constant offset, which is the same answer that arm would have given.
///
/// ⛔ `compressUnusedSymbols` AND `simplifyAffineMap` (`Utils.cpp:38-39`) ARE DROPPED AS
/// CANONICALISATIONS. They exist so MLIR's flattener sees a minimal map; the island's flattener
/// derives the coefficient row from the expression tree directly, and an unused symbol contributes a
/// zero column either way. ⚠️ They are NOT dropped silently: had they changed `getNumDims()`, they
/// would have changed the ternary — and the ternary is dead regardless, as above.
///
/// ⛔ `fullyComposeAffineMapAndOperands` IS THE OPERAND-REACHING MECHANISM, which is the one thing a
/// port may leave behind. It folds `affine.apply` producers among `indices` into the map. This
/// island has no `affine.apply`: a subscript is a [`Val`] or an inline strided sum
/// (`islands/dataflow_ir/dialects/mod.rs`, `Index::Strided`, which records that IBM's own files
/// contain zero of them and that emitting one made `dbo-opt` refuse). With nothing to fold, the call
/// is the identity.
///
/// ⛔ A COEFFICIENT PAST THE END OF THE ROW IS 0, NOT A READ PAST IT. The reference indexes
/// `composed_layout_coeffs[i]` for every index, relying on the composed map having one dimension per
/// index — which [`AccessDetailsAffine::construct_indices`] is what establishes. A dimension the
/// address does not read has coefficient 0, and that is the only meaning available for a column that
/// is not there.
fn construct_iterator_coeff_dict(
    subscripts_map: &AffineMap,
    transfer_order: &AffineMap,
    mem_view_layout_map: &AffineMap,
    indices: &[Val],
) -> IndicesCoeffDict {
    let composed_load_order_map = transfer_order.compose(subscripts_map);
    let composed_layout_map = mem_view_layout_map.compose(&composed_load_order_map);
    let composed_layout_coeffs = composed_layout_map.coefficients(0);

    // Sort the indices from outermost to innnermost (`Utils.cpp:78`) — the walk is over `indices`,
    // which is already in that order; see [`IndicesCoeffDict`] on why this is a vector.
    IndicesCoeffDict {
        per_index: indices
            .iter()
            .enumerate()
            .map(|(position, index)| {
                (
                    *index,
                    composed_layout_coeffs.get(position).copied().unwrap_or(0),
                )
            })
            .collect(),
        constant: composed_layout_coeffs.last().copied().unwrap_or(0),
    }
}

/// AN AFFINE ACCESS THAT ALSO WALKS TIME — `AccessDetailsAffineComposite`
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:247-343`).
///
/// ⭐⭐ TIME IS WHAT A COMPOSITE TRANSFER ADDS. "An AGEN composite transfer moves at most one
/// hardware vector per time step, so a transfer wider than that has to walk the remaining elements
/// over AGEN time dimensions" — the seven members below are that walk, and
/// [`agen::Op::CompositeLoadAndStore`](crate::islands::dataflow_ir::dialects::agen::Op::CompositeLoadAndStore)
/// is the op they are read off.
///
/// ⭐ THE `pub` FIELDS ARE THE PUBLIC GETTERS OF `:262-269`, and the three setters
/// `docs/bridge2-porting-order.md` excludes — `setTimeOrder` (`:284`), `setTimeSet` (`:285`) and
/// `setBurstIndex` (`:298`) — are the field itself in Rust.
///
/// ⛔⛔ A FIELD IS A BORROW AND `getTimeBounds()` WAS A COPY. Every getter here returns a
/// `SmallVector` BY VALUE, and two consumers rely on that: `coalesceTimeDimensions` mutates
/// `auto time_bounds = ...getTimeBounds()` and only then stores it back (`AccessDetails.cpp:690-691,
/// 720-739, 787`), and `computeBurstAndGroup` reads its own copy while calling setters on `self`
/// (`:797-798`). Whoever ports those must `.clone()` where C++ copied; mutating the field in place
/// would let a half-rebuilt vector be read through `getFirst()` by the other operand's pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessDetailsAffineComposite<'a> {
    /// The base class (`AccessDetails.hpp:247`).
    pub affine: AccessDetailsAffine<'a>,
    /// `time_order_` — the order the time dimensions are walked in (`AccessDetails.hpp:304`).
    ///
    /// ⛔ `None` is the default-constructed NULL map, for the same reason as
    /// [`AccessDetailsAffine::subscripts_map`]. Every bound and offset is reordered THROUGH it —
    /// `time_offsets = time_order.compose(...)` (`dialect_utils/Agen/Utils.cpp:115`),
    /// `time_bounds = time_order.compose(time_bounds)` (`:255`) — which is why the two vectors below
    /// are documented as already ordered (`AccessDetails.cpp:689`).
    pub time_order: Option<AffineMap>,
    /// `time_set_` — the time iteration domain (`AccessDetails.hpp:307`).
    pub time_set: Option<IntegerSet>,
    /// `time_addr_map_` — the address map over the time dimensions (`AccessDetails.hpp:310`).
    pub time_addr_map: Option<AffineMap>,
    /// `time_symbols_` — the SSA values the time set's bounds are written against
    /// (`AccessDetails.hpp:313`).
    pub time_symbols: Vec<Val>,
    /// `time_offsets_` — the address offset along each time dimension (`AccessDetails.hpp:316`),
    /// plus the flattened constant. See [`TimeOffsets`].
    pub time_offsets: TimeOffsets,
    /// `time_bounds_` — the bound of each time dimension (`AccessDetails.hpp:319`). See
    /// [`TimeBound`].
    pub time_bounds: Vec<TimeBound>,
    /// `burst_index_` — the time dimension claimed as the burst (`AccessDetails.hpp:321-324`).
    ///
    /// ⛔ `None` IS THE `-1` THE FIELD DEFAULTS TO, and `computeBurstAndGroup` reads it as a question
    /// before an index: `if (burst_index < 0) setBurstIndex(i);` (`AccessDetails.cpp:812-815`).
    /// `setBurstIndex` (`AccessDetails.hpp:298`) is excluded from the port as the field itself.
    pub burst_index: Option<TimeDim>,
    /// `interleave_group_index_` — the time dimension claimed as the interleave group
    /// (`AccessDetails.hpp:326-329`).
    pub interleave_group_index: Option<TimeDim>,
}

impl<'a> AccessDetailsAffineComposite<'a> {
    /// Replaces: e019_AccessDetailsAffine
    ///
    /// **`AccessDetailsAffineComposite::AccessDetailsAffineComposite`** —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:257-259` (3L).
    ///
    /// ```text
    /// explicit AccessDetailsAffineComposite(mlir::Operation* op, SenComponents comp)
    ///     : AccessDetailsAffine(op, comp) {}
    /// ```
    ///
    /// ⛔ THE ENTRY'S NAME IS THE BASE INITIALIZER, NOT THE CONSTRUCTOR. The extractor named unit 019
    /// `AccessDetailsAffine` after the `AccessDetailsAffine(op, comp)` token on line 259, but the
    /// definition at that citation is `AccessDetailsAffineComposite`'s constructor — `:217-218` is
    /// the `AccessDetailsAffine` one, and it is unit 016. The port follows the line citation.
    ///
    /// ⭐ AN EMPTY BODY IS THE WHOLE PORT, AND WHAT IT LEAVES UNSET IS THE POINT: seven of the nine
    /// members default-initialize (`:304-329`) and every one of them is filled later by
    /// `initialize()` and `constructTimeStepsInfo`. The two negative sentinels among them become
    /// [`None`] here, so a freshly constructed object cannot be indexed with as though a burst had
    /// already been chosen.
    ///
    /// ⭐ THE BASE CHAIN IS A CALL, NOT A LITERAL, exactly as `: AccessDetailsAffine(op, comp)` is.
    /// That constructor is unit 016 and carries its own anchor; this one delegates to it.
    #[must_use]
    pub fn new(op: &'a agen::Op, comp: DfirUnit) -> AccessDetailsAffineComposite<'a> {
        AccessDetailsAffineComposite {
            affine: AccessDetailsAffine::new(op, comp),
            time_order: None,
            time_set: None,
            time_addr_map: None,
            time_symbols: Vec::new(),
            time_offsets: TimeOffsets::default(),
            time_bounds: Vec::new(),
            burst_index: None,
            interleave_group_index: None,
        }
    }

    /// Replaces: e020_setTimeAddrMap
    ///
    /// **`setTimeAddrMap`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286-288` (3L).
    ///
    /// ```text
    /// void setTimeAddrMap(AffineMap time_addr_map) {
    ///   time_addr_map_ = time_addr_map;
    /// }
    /// ```
    ///
    /// ⭐⭐ THIS FIELD IS PER OPERAND WHILE ITS NEIGHBOURS ARE PER OP, and `initialize` draws the
    /// line: `setTimeSet`, `setTimeOrder` and `setTimeSymbols` run once for the whole
    /// `composite_load_and_store` (`AccessDetails.cpp:519-521`), but the branch below them picks
    /// `getLoadTimeAddrMap()` for `kDirSrc` (`:532`) and `getStoreTimeAddrMap()` for the dst (`:543`).
    /// Those are the two maps
    /// [`CompositeTransfer`](crate::islands::dataflow_ir::dialects::agen::CompositeTransfer) carries,
    /// which is why one transfer needs one access detail per operand.
    ///
    /// ⭐ `calculateTimeOffsets` composes it under the memory view's layout to get the offsets
    /// (`dialect_utils/Agen/Utils.cpp:103`).
    pub fn set_time_addr_map(&mut self, time_addr_map: AffineMap) {
        self.time_addr_map = Some(time_addr_map);
    }

    /// Replaces: e021_setTimeSymbols
    ///
    /// **`setTimeSymbols`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289-291` (3L).
    ///
    /// ```text
    /// void setTimeSymbols(const Operation::operand_range time_symbols) {
    ///   time_symbols_.assign(time_symbols.begin(), time_symbols.end());
    /// }
    /// ```
    ///
    /// ⭐ `Operation::operand_range` IS A BORROWED VIEW OVER THE OP'S OWN OPERANDS and `assign`
    /// copies out of it, which is exactly `&[Val]` plus `to_vec`. `assign` also CLEARS first, so a
    /// second call replaces rather than appends.
    ///
    /// ⭐ ONE CALL PER OP, NOT PER OPERAND: it sits beside `setTimeSet` and `setTimeOrder` above the
    /// `kDirSrc` branch (`AccessDetails.cpp:519-521`), unlike
    /// [`set_time_addr_map`](Self::set_time_addr_map).
    ///
    /// ⛔⛔ OUR ISLAND CANNOT YET PRODUCE A NON-EMPTY RANGE, AND THAT IS A FACT ABOUT THE ISLAND, NOT
    /// A REASON TO SKIP THE SETTER. `agen.composite_load_and_store` prints its time symbols as a
    /// literal empty `time_symbols()` in this crate's emitter, and all eighteen
    /// `tests/sentient_corpus/*.dfir.mlir` files agree. The reason is structural: the only consumer
    /// is `calculateTimeBounds`, which consults a symbol solely for a dimension whose bound is
    /// symbolic (`dialect_utils/Agen/Utils.cpp:216-256`), and
    /// [`IntegerSet`](crate::islands::dataflow_ir::ty::IntegerSet) here has a dimension count and no
    /// symbol count at all — so no set we can build has a symbolic bound to resolve. The field is
    /// ported faithfully and the island is deliberately NOT widened, because widening it would add a
    /// symbol operand that nothing in this crate can populate. Reported to the orchestrator.
    pub fn set_time_symbols(&mut self, time_symbols: &[Val]) {
        self.time_symbols = time_symbols.to_vec();
    }

    /// Replaces: e022_setTimeBounds
    ///
    /// **`setTimeBounds`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292-294` (3L).
    ///
    /// ```text
    /// void setTimeBounds(const SmallVectorImpl<int64_t>& time_bounds) {
    ///   time_bounds_.assign(time_bounds.begin(), time_bounds.end());
    /// }
    /// ```
    ///
    /// ⛔ `assign` REPLACES, AND COALESCING DEPENDS ON IT. `coalesceTimeDimensions` rebuilds the
    /// vector from scratch — clearing it, refilling it inner-to-outer, then re-inserting the
    /// dimensions above the cut — and stores the result over BOTH mandatory operands
    /// (`AccessDetails.cpp:720-739,785-792`). An appending setter would double the time dimensions
    /// of every coalesced transfer.
    ///
    /// ⭐ THE SLICE IS `&[TimeBound]` AND NOT `&[i64]`, so the two sentinels of
    /// `SpecialTimeBoundValues` cannot arrive here as ordinary counts. See [`TimeBound`].
    pub fn set_time_bounds(&mut self, time_bounds: &[TimeBound]) {
        self.time_bounds = time_bounds.to_vec();
    }

    /// Replaces: e023_setTimeOffsets
    ///
    /// **`setTimeOffsets`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295-297` (3L).
    ///
    /// ```text
    /// void setTimeOffsets(const SmallVectorImpl<int64_t>& time_offsets) {
    ///   time_offsets_.assign(time_offsets.begin(), time_offsets.end());
    /// }
    /// ```
    ///
    /// ⛔ THE PARAMETER IS NOT A FLAT VECTOR OF OFFSETS — its last element is a constant term and
    /// never a dimension, which is why the reference has to write
    /// `time_bounds.size() == time_offsets.size() - 1` three times to keep the two in step. Taking
    /// [`TimeOffsets`] makes `per_dim` index in lockstep with `time_bounds` by construction; see
    /// that type for the derivation.
    ///
    /// ⭐ BY VALUE, BECAUSE THE CALLER'S VECTOR IS DEAD AFTER THE CALL. `constructTimeStepsInfo`
    /// reads the current value out, lets `calculateTimeOffsets` fill it, and hands it straight back
    /// (`AccessDetails.cpp:651-658`) — a move, not a shared borrow.
    pub fn set_time_offsets(&mut self, time_offsets: TimeOffsets) {
        self.time_offsets = time_offsets;
    }

    /// Replaces: e024_setInterleaveGroupIndex
    ///
    /// **`setInterleaveGroupIndex`** — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299-301`
    /// (3L).
    ///
    /// ```text
    /// void setInterleaveGroupIndex(int interleave_group_index) {
    ///   interleave_group_index_ = interleave_group_index;
    /// }
    /// ```
    ///
    /// ⛔⛔ IT TAKES A [`TimeDim`] AND NOT AN `int`, BECAUSE ITS ONE CALLSITE HANDS IT THE BURST'S OWN
    /// INDEX. `computeBurstAndGroup` promotes the dimension already holding the burst into the group
    /// field and moves the burst OUTWARD onto the dimension it is currently visiting:
    /// ```text
    /// if ((time_bounds[burst_index] == 2 || time_bounds[burst_index] == 4) &&
    ///     time_offsets[i] == getLdOrStSize() && time_offsets[burst_index] != 0) {
    ///   setInterleaveGroupIndex(burst_index);
    ///   setBurstIndex(i);
    /// }
    /// ```
    /// (`AccessDetails.cpp:819-823`).
    ///
    /// ⚠️ OUTWARD, AND THE DIRECTION IS THE SCAN'S. The loop runs
    /// `for (int i = time_bounds.size() - 1; i >= 0; --i)` (`:798`) over a vector whose index 0 is
    /// the OUTERMOST time dimension — `constructTimeLoops` builds loop `idx` inside loop `idx - 1`
    /// (`Helper.cpp:1808-1857`) and stops at `loop_num = burst_index`, so the burst dimension and
    /// everything inside it are absorbed by the hardware fields instead of becoming loops
    /// (`:1807`, `:1859-1860`). The burst is therefore claimed on the INNERMOST valid dimension
    /// first, and the promotion hands that inner dimension to the group while the burst moves out
    /// to `i`. `burst_index` is `>= 0` there — the `< 0` branch above it took
    /// the other path (`:813`) — so the argument is always a real dimension and the `-1` this field
    /// starts at is never passed in. Nothing clears the field, which is why there is no
    /// `Option`-taking form.
    pub fn set_interleave_group_index(&mut self, interleave_group_index: TimeDim) {
        self.interleave_group_index = Some(interleave_group_index);
    }

    /// Replaces: e148_computeBurstAndGroup
    ///
    /// **148/384** `AccessDetailsAffineComposite::computeBurstAndGroup` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796` (36L).
    ///
    /// # ⭐⭐ IT CLAIMS TIME DIMENSIONS FOR THE TWO HARDWARE FIELDS IN THE LDST INSTRUCTION
    ///
    /// *"This function tries to assign time dims to burst_size and group interleaving fields in LDST
    /// instr"* (`hpp:335-336`). Every time dimension a field absorbs is a dimension
    /// `constructTimeLoops` does NOT have to emit as a loop — it stops at `loop_num = burst_index`
    /// (`Helper.cpp:1807`, `:1859-1860`) — so this is where a nest becomes an instruction.
    ///
    /// ```cpp
    /// for (int i = time_bounds.size() - 1; i >= 0; --i) {
    ///   auto curr_bound = time_bounds[i];
    ///   if (curr_bound == kCoalesced) {
    ///     continue;
    ///   } else if (curr_bound < 0) {
    ///     return;
    ///   } else if (curr_bound == 0) {
    ///   } else {
    ///     auto burst_index = getBurstIndex();
    ///     if (burst_index < 0) {
    ///       setBurstIndex(i);
    ///     } else if (!is_any_of(getComp(), L3LU, L3SU)) {
    ///       if ((time_bounds[burst_index] == 2 || time_bounds[burst_index] == 4) &&
    ///           time_offsets[i] == getLdOrStSize() && time_offsets[burst_index] != 0) {
    ///         setInterleaveGroupIndex(burst_index);
    ///         setBurstIndex(i);
    ///       }
    ///       return;
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// # ⛔⛔ INNERMOST FIRST, AND THE PROMOTION MOVES THE BURST **OUTWARD**
    ///
    /// The scan runs from `size() - 1` down to 0 over a vector whose index 0 is the OUTERMOST time
    /// dimension, so the first valid dimension it meets is the innermost one and that is what takes
    /// the burst. When a second valid dimension appears further out, the dimension already holding the
    /// burst is handed to the interleave group and the burst moves out to the current one — see
    /// [`Self::set_interleave_group_index`], whose doc traces the direction through
    /// `constructTimeLoops`.
    ///
    /// # ⛔ FOUR ARMS, AND THE THREE THAT ARE NOT THE SEARCH ALL DO SOMETHING DIFFERENT
    ///
    /// A [`TimeBound::Coalesced`] dimension is SKIPPED and the scan goes on; a
    /// [`TimeBound::Variable`] one TERMINATES the scan, because a bound the compiler cannot see
    /// cannot be folded into an instruction field and neither can anything outside it; and
    /// `Steps(0)` — the reference's empty `else if (curr_bound == 0) {}` — neither claims a field nor
    /// stops the search. Those are three distinct behaviours behind one `int64_t`'s sign, which is
    /// what [`TimeBound`] exists to keep apart.
    ///
    /// # ⛔⛔ THE `return` AFTER THE GROUP BRANCH IS UNCONDITIONAL, AND THE L3 ARM FALLS THROUGH
    ///
    /// Once a burst is held, a non-L3 component gets exactly ONE attempt at the group and then the
    /// function returns whether or not it took it (`:824`) — so a third valid dimension is never
    /// examined. An L3 component takes neither branch: `!is_any_of(getComp(), L3LU, L3SU)` is false,
    /// nothing happens, and the loop keeps scanning outward without ever claiming a group. L3's LDST
    /// has no interleave-group field to fill.
    ///
    /// ⛔ THE GROUP IS ONLY TAKEN FROM A BURST OF **2 OR 4** whose own offset is non-zero, and only
    /// when the current dimension's offset is exactly one load/store's worth of elements
    /// ([`AccessDetailsBase::ld_or_st_size`], which is why entry 144 has to have run first). Those
    /// three conditions are what "interleaved" means in the hardware: two or four consecutive
    /// transfers, each displaced from the last, repeating one whole instruction's stride outward.
    ///
    /// ⛔ THE BOUNDS AND OFFSETS ARE READ FROM COPIES, as `auto time_bounds = getTimeBounds()` is
    /// (`:797-798`): the setters called inside the loop write `self`, and the reference's copies are
    /// what keep the comparison `time_bounds[burst_index]` reading the vector as it was on entry.
    pub fn compute_burst_and_group(&mut self) {
        let time_bounds = self.time_bounds.clone();
        let time_offsets = self.time_offsets.clone();
        let offset = |dim: TimeDim| time_offsets.per_dim.get(dim.index()).copied().unwrap_or(0);
        let count = u32::try_from(time_bounds.len()).unwrap_or(u32::MAX);

        for position in (0..count).rev() {
            let dim = TimeDim(position);
            let Some(curr_bound) = time_bounds.get(dim.index()).copied() else {
                continue;
            };
            match curr_bound {
                // ignore coalesced time dim
                TimeBound::Coalesced => continue,
                // if forOp bound is variable, terminate search
                TimeBound::Variable => return,
                TimeBound::Steps(0) => {}
                // if current bound is valid value, find a field to fit
                TimeBound::Steps(_) => match self.burst_index {
                    // always fill in burst field first before group.
                    None => self.burst_index = Some(dim),
                    Some(burst_index) => {
                        // ⛔ L3 CLAIMS NO GROUP AND KEEPS SCANNING — the arm's condition is false, so
                        // neither the promotion nor the `return` below is reached.
                        if matches!(self.affine.base.comp, DfirUnit::L3lu | DfirUnit::L3su) {
                            continue;
                        }
                        // if burst field has already been used, check if group field can be used.
                        let burst_is_interleavable =
                            matches!(time_bounds[burst_index.index()], TimeBound::Steps(2 | 4));
                        let one_whole_transfer = u64::try_from(offset(dim))
                            .is_ok_and(|offset| Elements(offset) == self.affine.base.ld_or_st_size);
                        if burst_is_interleavable && one_whole_transfer && offset(burst_index) != 0
                        {
                            self.set_interleave_group_index(burst_index);
                            self.burst_index = Some(dim);
                        }
                        return;
                    }
                },
            }
        }
    }
}

/// THE ACCESS DETAILS OF A **SYMBOLIC** ACCESS — one whose strides are SSA VALUES rather than
/// literals (`AccessDetails.hpp:340-367`).
///
/// ⭐ THAT IS THE WHOLE DIFFERENCE FROM THE AFFINE FORM. An affine access states its coefficients as
/// integers the pass can fold; a symbolic one indexes through values the program computes — a
/// `symbol.create_symbol`, a toggling JCR base — so the strides can only be carried as the values
/// themselves and the arithmetic has to be emitted.
///
/// Only the field [`e025_setStrides`](Self::set_strides) owns and the base class it derives from are
/// present; the rest arrives with `e265_constructDetails`, which is another entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessDetailsSymbolic<'a> {
    /// The `AccessDetailsBase` this derives from (`AccessDetails.hpp:340`).
    ///
    /// ⛔ `e155_updateSymbolicAccessDetails` REWRITES FIVE OF ITS FIELDS — the op, the indices, the
    /// mem_ref, the view start address and the memory (`Helper.cpp:1017-1046`) — so the base is not
    /// optional decoration on this class; without it that entry has nothing to remap.
    pub base: AccessDetailsBase<'a>,
    /// `strides_` — *"strides used in the indices accesses"* (`AccessDetails.hpp:366`).
    strides: Vec<Val>,
}

impl<'a> AccessDetailsSymbolic<'a> {
    /// `explicit AccessDetailsSymbolic(mlir::Operation* op, SenComponents comp)
    /// : AccessDetailsBase(op, comp) {}` (`AccessDetails.hpp:342-343`).
    ///
    /// ⭐ NOT A SCHEDULED UNIT — the winnow excluded it with the other delegating constructors; it is
    /// written here for the reason [`AccessDetailsBase::new`] is, because a caller needs a way to
    /// build one and `Default` cannot name the operation the object describes.
    #[must_use]
    pub fn new(op: &'a agen::Op, comp: DfirUnit) -> Self {
        AccessDetailsSymbolic {
            base: AccessDetailsBase::new(op, comp),
            // `SmallVector<Value> strides_;` (`hpp:366`).
            strides: Vec::new(),
        }
    }
    /// The strides, in the order the access lists its indices — `getStrides`
    /// (`AccessDetails.hpp:352`).
    #[must_use]
    pub fn strides(&self) -> &[Val] {
        &self.strides
    }

    /// Replaces: e025_setStrides
    ///
    /// `setStrides` (`AccessDetails.hpp:355-357`):
    ///
    /// ```c++
    /// void setStrides(const SmallVectorImpl<Value>& strides) {
    ///   strides_.assign(strides.begin(), strides.end());
    /// }
    /// ```
    ///
    /// ⛔ `assign`, NOT `append` — THE LIST IS REPLACED. `e155_updateSymbolicAccessDetails`
    /// (`Helper.cpp:1013-1047`) calls this after it has recomputed the strides for a NEW enclosing
    /// loop nest; appending would leave the previous nest's strides in front of them, and an access
    /// with twice as many strides as indices addresses whatever the extra ones happen to reach.
    pub fn set_strides(&mut self, strides: &[Val]) {
        self.strides.clear();
        self.strides.extend_from_slice(strides);
    }
}

/// A MAP FROM EACH [`MemoryOperandIndex`] TO THE ACCESS DETAIL, ADDRESS OR VALUE IT OWNS —
/// `AccessContainer<T>` (`AccessDetails.hpp:369-430`).
///
/// ⭐ TWO STRUCTURES IN ONE, AND BOTH ARE READ. The C++ derives from `std::vector<T>`, so callers
/// index it POSITIONALLY (`access_details[0]`, `Helper.cpp:3181-3203`) in insertion order, while
/// `index_mapping_` answers *which operand does slot `i` belong to*. Dropping either half would
/// break a caller: the vector order is what "the first mandatory operand" means, and the mapping is
/// what makes `has`/`get` answerable at all.
///
/// ⛔ NO `-1` SENTINEL. The C++ fills `index_mapping_` with `-1` and asks `!= -1`; a slot that is
/// EMPTY is [`None`] here, so "unfilled" and "filled with entry 0" are different values rather than
/// two readings of one `int`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessContainer<T> {
    /// The entries, in insertion order — the `std::vector<T>` half.
    entries: Vec<T>,
    /// Which entry each memory operand owns, or [`None`] where it owns none — `index_mapping_`.
    slots: [Option<usize>; MemoryOperandIndex::COUNT],
}

impl<T> Default for AccessContainer<T> {
    /// An empty container: no entries, and every operand unfilled — the C++'s
    /// `index_mapping_((int)kMax, -1)`.
    fn default() -> Self {
        AccessContainer {
            entries: Vec::new(),
            slots: [None; MemoryOperandIndex::COUNT],
        }
    }
}

impl<T> AccessContainer<T> {
    /// The entries in insertion order — what the C++'s `std::vector<T>` base class exposes, and what
    /// `access_details[0]` reads (`Helper.cpp:3203`).
    #[must_use]
    pub fn entries(&self) -> &[T] {
        &self.entries
    }

    /// The entries in insertion order, MUTABLY — `for (auto& ad : access_details)` over the C++'s
    /// `std::vector<T>` base class (`Helper.cpp:1015`).
    ///
    /// ⛔ THE SLOT MAP IS UNTOUCHED BY DESIGN. A caller may rewrite what an entry SAYS —
    /// `e155_updateSymbolicAccessDetails` remaps every value in it onto a clone — but not which
    /// operand owns it, so `slots` stays private and the invariant it carries cannot be broken from
    /// outside.
    pub fn entries_mut(&mut self) -> &mut [T] {
        &mut self.entries
    }

    /// Replaces: e026_has
    ///
    /// `has` (`AccessDetails.hpp:397-399`):
    ///
    /// ```c++
    /// bool has(MemoryOperandIndex moi) const {
    ///   return index_mapping_[(int)moi] != -1;
    /// }
    /// ```
    ///
    /// ⭐ THE ONE QUESTION EVERY CALLER OF `get` MUST ASK FIRST. `get` is `DT_CHECK`ed on it
    /// (*"no entry exists for the requested memory operand index"*, `:401-405`), and the lowerings
    /// branch on it to decide what an op even is: `e374_lowerSymbolicVectorLoadOp` and
    /// `e327_gatherSymbolicLoadStoreDetails` use it to tell an indirect access from a direct one.
    ///
    /// ⛔ FILLED WITH ENTRY **0** IS FILLED. The C++ compares against `-1` rather than testing for
    /// zero, and the first insertion always maps to slot 0 — so a `has` written as "nonzero" would
    /// answer `false` for the very first operand inserted, which is `kDirSrc` on almost every op.
    #[must_use]
    pub fn has(&self, moi: MemoryOperandIndex) -> bool {
        self.slots[moi.slot()].is_some()
    }

    /// AN OPERAND'S SLOT, IF IT IS STILL EMPTY — the `!has(moi)` half of `insert`'s first
    /// `DT_CHECK_MSG` (`AccessDetails.hpp:389-390`), turned into the only way to reach the insertion.
    ///
    /// ⛔⛔ THIS IS WHAT REPLACES *"trying to insert into a slot that is already filled"*. The check
    /// is not moved, it is inverted: a [`VacantSlot`] can only be handed out for an empty slot, it
    /// borrows the container exclusively so no second one can exist beside it, and
    /// [`VacantSlot::fill`] consumes it — so filling the same operand twice is not a call the compiler
    /// accepts. See [`VacantSlot`].
    #[must_use]
    pub fn vacancy(&mut self, moi: MemoryOperandIndex) -> Option<VacantSlot<'_, T>> {
        if self.has(moi) {
            return None;
        }
        Some(VacantSlot {
            container: self,
            moi,
        })
    }

    /// Replaces: e150_get
    ///
    /// **150/384** `AccessContainer::get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`
    /// (4L).
    ///
    /// ```cpp
    /// const T& get(MemoryOperandIndex moi) const {
    ///   DT_CHECK_MSG(has(moi),
    ///                "no entry exists for the requested memory operand index");
    ///   return this->at(index_mapping_[(int)moi]);
    /// }
    /// ```
    ///
    /// ⛔⛔ THE `DT_CHECK_MSG` IS THE RETURN TYPE. *"no entry exists for the requested memory operand
    /// index"* is precisely [`None`], and the reference's own callers already ask the question first —
    /// `if (has(kDirSrc)) return get(kDirSrc);` is how `getFirst` is written (`:411-418`), and
    /// `e374_lowerSymbolicVectorLoadOp` and `e327_gatherSymbolicLoadStoreDetails` branch on `has` to
    /// tell an indirect access from a direct one. An [`Option`] makes the pair one lookup instead of
    /// two, and there is no path left on which the check can be forgotten.
    ///
    /// ⛔ TWO INDIRECTIONS, AND BOTH ARE CHECKED BY THE SAME `?`. `index_mapping_[(int)moi]` is the
    /// slot and `at(...)` is the entry; the reference's `at` throws where a slot names an entry that
    /// is not there, which nothing can arrange because [`VacantSlot::fill`] writes the slot and the
    /// entry together.
    ///
    /// ⚠️ THE NON-CONST OVERLOAD (`:406-408`) IS NOT THIS ENTRY. It is a `const_cast` twin of this one
    /// and no unit of its own; in Rust the shared and exclusive lookups are separate functions, so it
    /// arrives with the first entry that needs to mutate an entry in place —
    /// `coalesceTimeDimensions`, which writes `access_details.get(kDirDst).setTimeBounds(...)`
    /// (`AccessDetails.cpp:787`).
    #[must_use]
    pub fn get(&self, moi: MemoryOperandIndex) -> Option<&T> {
        self.entries.get(self.slots[moi.slot()]?)
    }

    /// Replaces: e209_getFirst
    ///
    /// **209/384** `AccessContainer::getFirst` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411` (7L). The SOURCE operand if the
    /// container has one, else the DESTINATION operand.
    ///
    /// ⛔ ONLY THE **DIRECT** PAIR IS CONSULTED (`:412-415`): a container holding `kIndSrc` alone —
    /// which `composite_indirect_load_and_store` may build (`:34-35`) — reaches the
    /// `llvm_unreachable("expected at least one of kDirSrc or kDirDst operands")` (`:417`), and that
    /// is [`None`] here.
    #[must_use]
    pub fn get_first(&self) -> Option<&T> {
        self.get(MemoryOperandIndex::DirSrc)
            .or_else(|| self.get(MemoryOperandIndex::DirDst))
    }
}

/// PROOF THAT ONE MEMORY OPERAND'S SLOT IS EMPTY, AND THE RIGHT TO FILL IT ONCE.
///
/// ⭐⭐ THE TWO `DT_CHECK_MSG`S OF `insert` BECOME THIS TYPE. `AccessDetails.hpp:388-395` guards its
/// two statements with *"trying to insert into a slot that is already filled"* and *"no more than kMax
/// entries are allowed"*; both are questions about the container that a caller could get wrong at run
/// time. Obtained only from [`AccessContainer::vacancy`] and consumed by [`Self::fill`], this handle
/// answers the first before the call exists and makes the second unreachable — with one slot per
/// operand and a slot required, at most [`MemoryOperandIndex::COUNT`] entries can ever be pushed,
/// which is the `kMax` the reference is counting against.
#[derive(Debug)]
pub struct VacantSlot<'a, T> {
    /// The container the entry goes into, borrowed exclusively so no second vacancy can be live.
    container: &'a mut AccessContainer<T>,
    /// Which operand the new entry belongs to.
    moi: MemoryOperandIndex,
}

impl<'a, T> VacantSlot<'a, T> {
    /// Replaces: e149_insert
    ///
    /// **149/384** `AccessContainer::insert` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388` (7L).
    ///
    /// ```cpp
    /// void insert(MemoryOperandIndex moi, T access) {
    ///   DT_CHECK_MSG(!has(moi),
    ///                "trying to insert into a slot that is already filled");
    ///   DT_CHECK_MSG(this->size() <= MemoryOperandIndex::kMax,
    ///                "no more than kMax entries are allowed");
    ///   index_mapping_[moi] = this->size();
    ///   this->push_back(access);
    /// }
    /// ```
    ///
    /// ⭐⭐ THE BODY IS TWO STATEMENTS AND THEIR ORDER IS THE WHOLE OF IT: the slot records
    /// `size()` — the position the entry is ABOUT to take — and only then is the entry pushed. Reading
    /// `size()` after the push would map every operand one past its own entry.
    ///
    /// ⛔ IT RETURNS `void`. `emplace_insert` (`:377-386`) is the twin that ends `return get(moi)`,
    /// and it is entry 208; nothing here hands back the entry, so nothing here needs the exclusive
    /// lookup that `get` has no unit for.
    ///
    /// ⛔ `T` BY VALUE, AS THE REFERENCE TAKES IT — `T access` is a copy in C++ and a move here, and
    /// `push_back` transfers it into the vector either way.
    ///
    /// ⛔ CONSUMING `self` IS WHAT MAKES ONE VACANCY ONE INSERTION. See [`VacantSlot`] for the pair of
    /// `DT_CHECK_MSG`s this retires.
    pub fn fill(self, access: T) {
        self.container.slots[self.moi.slot()] = Some(self.container.entries.len());
        self.container.entries.push(access);
    }

    /// Replaces: e208_emplace_insert
    ///
    /// **208/384** `AccessContainer::emplace_insert` —
    /// `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378` (8L). [`Self::fill`] plus the
    /// reference's closing `return get(moi);` (`:385`) — the entry, ready to be written into.
    ///
    /// ⛔ THE VARIADIC HALF DOES NOT SURVIVE. `template <class... Args>` forwarding into
    /// `emplace_back` is C++'s way to build the entry in place; the caller builds it here and moves
    /// it, which is what a `T` by value already is.
    ///
    /// ⛔ AND THE RETURN IS **EXCLUSIVE**, unlike [`AccessContainer::get`]: `gatherAffineLoadStoreDetails`
    /// keeps writing through it (`Helper.cpp:566`), so this is the one place the non-const lookup is
    /// needed and the [`VacantSlot`] hands out its own borrow rather than a second one.
    pub fn emplace_insert(self, access: T) -> &'a mut T {
        let at = self.container.entries.len();
        self.container.slots[self.moi.slot()] = Some(at);
        self.container.entries.push(access);
        &mut self.container.entries[at]
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AccessContainer, AccessDetailsAffine, AccessDetailsAffineComposite, AccessDetailsBase,
        AccessDetailsSymbolic, AffineInitialize, ChunkAndShuffleInfo, ConstructedIndices,
        IndicesCoeffDict, LayoutCoeff, MemViewSource, MemoryOperandIndex, SubscriptOperand,
        TimeBound, TimeDim, TimeOffsets, TransferDim, TransferExtents, sen,
        set_coalesced_bound_values,
    };
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::dialects::agen;
    use crate::islands::dataflow_ir::dialects::{
        Index, Op as DfirOp, Val, affine, arith, dataflow, scf, vectorchain,
    };
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef, ScalarTy, Vector,
    };
    use crate::units::DfirUnit;

    /// THE TYPE THE FIXTURE'S LOAD PRODUCES — `vector<64xf16>`.
    ///
    /// ⭐ NAMED SO NO TEST HAS TO DESTRUCTURE THE OP TO REACH IT. Matching one variant out of an
    /// `Op` needs a `_ =>` arm, and this island's own note on `compute_attrs`
    /// (`src/islands/sentient/dialects/sentient.rs:3136-3140`) records why that arm — an
    /// `unreachable!` asserting the caller passed the right variant — is the class of guard the
    /// crate does not use. The pieces are in hand here, so nothing needs asserting.
    const LOADED: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };

    /// A `agen.vector_load %view[0, 0] : memref<8x64xf16>, vector<64xf16>` — the op an
    /// `AccessDetailsAffine` is built from (`AccessDetails.cpp:299`).
    fn vector_load() -> agen::Op {
        agen::Op::VectorLoad {
            result: Val(2),
            view: Val(1),
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty: MemRef {
                shape: vec![8, 64],
                elem: ElemType::F16,
            },
            ty: LOADED,
        }
    }

    /// 009 — `rotation_position_ = 0` UNTIL A `vectorchain.rotate` CONSUMER SAYS OTHERWISE.
    ///
    /// The default is the load-bearing half: `constructChunkAndShuffleInfo` assigns this field only
    /// inside its `isa<vectorchain::RotateOp>(user)` arm (`AccessDetails.cpp:243-248`), so every
    /// transfer without a rotate consumer must read zero here.
    #[test]
    fn rotation_position_defaults_to_zero_and_the_setter_replaces_it() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.rotation_position, Elements(0));

        ad.set_rotation_position(Elements(16));
        assert_eq!(ad.rotation_position, Elements(16));

        // ⛔ AND IT IS COMPARABLE TO `total_elements` WITHOUT A CONVERSION, which is what the C++
        // guard at `:249-252` needs: both are counts of elements.
        ad.set_total_elements(Elements(64));
        assert!(ad.rotation_position <= ad.total_elements);
    }

    /// 010 — THE RETURN TYPE'S COUNT, WHICH IS NOT THE SET-DERIVED COUNT.
    ///
    /// `initialize()` passes `getNumElements(load_op.getResult().getType())`
    /// (`AccessDetails.cpp:308-309`): for `vector<64xf16>` that is 64, and it is stored INDEPENDENTLY
    /// of `total_elements` so that `constructExtentAndTotalElements` can refuse a disagreement
    /// (`:91-94`).
    #[test]
    fn expected_total_elements_is_the_return_types_count_and_is_not_total_elements() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.expected_total_elements, Elements(0));

        ad.set_expected_total_elements(Elements(LOADED.len));

        assert_eq!(ad.expected_total_elements, Elements(64));
        // ⛔ THE OTHER COUNT IS UNTOUCHED — the two fields are the cross-check.
        assert_eq!(ad.total_elements, Elements(0));
    }

    /// 011 — `assign` REPLACES; A SHORTER SECOND CALL LEAVES NO TAIL.
    ///
    /// And it does not recompute `total_elements`: the C++ setter assigns one member, and the caller
    /// stores the product separately (`AccessDetails.cpp:88-89`).
    #[test]
    fn set_extents_replaces_the_whole_vector_and_leaves_total_elements_alone() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.extents, Vec::new());

        // `layouts: [1][4][2][3], extent: [4][1][1]` — the worked example at `AccessDetails.cpp:135`.
        ad.set_extents(&[Elements(4), Elements(1), Elements(1)]);
        assert_eq!(ad.extents, vec![Elements(4), Elements(1), Elements(1)]);
        assert_eq!(ad.total_elements, Elements(0));

        // A SHORTER vector: `assign` drops the third extent rather than keeping it.
        ad.set_extents(&[Elements(8), Elements(2)]);
        assert_eq!(ad.extents, vec![Elements(8), Elements(2)]);
    }

    /// 012 — THE PRODUCT OF THE EXTENTS, AS THE CALLER ACCUMULATES IT.
    ///
    /// The loop's idiom is `total = (total == 0) ? width : total * width` (`:60-64`), so extents
    /// `[4, 1, 1]` give 4 and not 0.
    #[test]
    fn set_total_elements_stores_the_extent_product() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.total_elements, Elements(0));

        let extents = [Elements(4), Elements(1), Elements(1)];
        ad.set_extents(&extents);
        let product = extents
            .iter()
            .fold(0u64, |acc, e| if acc == 0 { e.0 } else { acc * e.0 });
        ad.set_total_elements(Elements(product));

        assert_eq!(ad.total_elements, Elements(4));
    }

    /// 013 — BITS OF THE **ELEMENT**, NOT BYTES AND NOT THE VECTOR.
    ///
    /// `initialize()` stores `getElementTypeBitWidth(load_op.getResult().getType())` (`:306-307`).
    /// For `vector<64xf16>` that is 16 — the width of one `f16` — while the vector is 64 elements
    /// and 128 bytes wide. The reader turns it into elements-per-stick by dividing a stick's bits by
    /// it (`MutableStartAddrShifting.cpp:373`).
    #[test]
    fn element_width_is_the_element_types_bit_width() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.element_width, Bits(0));

        ad.set_element_width(Bits(LOADED.elem.bits()));

        assert_eq!(ad.element_width, Bits(16));
        // 128 bytes per stick, 8 bits each, 16 bits per element -> 64 elements per stick.
        assert_eq!(128 * 8 / ad.element_width.0, 64);
    }

    /// 014 — THE LOAD SET GOES IN WHOLE, CONSTRAINTS AND DIMENSION COUNT BOTH.
    ///
    /// It starts as the empty set — the Rust stand-in for the C++'s null `IntegerSetAttr` — and
    /// `constructExtentAndTotalElements` reads the stored value's constraints to derive the extents
    /// (`AccessDetails.cpp:33-37`).
    #[test]
    fn set_transfer_set_stores_the_load_set_whole() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.transfer_set.dims, 0);
        assert!(ad.transfer_set.constraints.is_empty());

        // `affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>` — one pinned dimension and one
        // 64-element span, as `IntegerSet::from_sizes` builds them.
        let load_set = IntegerSet::from_sizes(&[1, 64]);
        ad.set_transfer_set(load_set.clone());

        assert_eq!(ad.transfer_set, load_set);
        assert_eq!(ad.transfer_set.dims, 2);
        assert_eq!(
            ad.transfer_set.constraints,
            vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: true,
                },
                Constraint {
                    expr: AffineExpr::dim(1),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(1).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
            ]
        );
    }

    /// 015 — THE ORDER'S DIMENSION COUNT SURVIVES, BECAUSE THE READER PROJECTS BY IT.
    ///
    /// `constructExtentAndTotalElements` projects out `transfer_order.getNumDims()` variables
    /// starting at `getNumDims()` (`AccessDetails.cpp:39-40`), so a map stored without its arity
    /// would project the wrong range.
    #[test]
    fn set_transfer_order_stores_the_maps_arity_as_well_as_its_results() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.transfer_order.dims, 0);
        assert!(ad.transfer_order.results.is_empty());

        // `affine_map<(d0, d1) -> (d0, d1)>` — the identity load order the scheduler writes.
        let load_order = AffineMap::identity(2);
        ad.set_transfer_order(load_order.clone());

        assert_eq!(ad.transfer_order, load_order);
        assert_eq!(ad.transfer_order.dims, 2);
        assert_eq!(
            ad.transfer_order.results,
            vec![AffineExpr::dim(0), AffineExpr::dim(1)]
        );
    }

    /// 016 — THE DERIVED CONSTRUCTOR FORWARDS BOTH ARGUMENTS AND DEFAULTS EVERYTHING ELSE.
    #[test]
    fn the_affine_constructor_delegates_op_and_comp_and_defaults_the_rest() {
        let op = vector_load();
        let ad = AccessDetailsAffine::new(&op, DfirUnit::L0lu);

        assert_eq!(ad.base.op, &op);
        assert_eq!(ad.base.comp, DfirUnit::L0lu);

        // ⛔ EVERY OTHER MEMBER AT ITS DECLARED DEFAULT — identical to what the base constructor
        // leaves, because the delegation body is empty (`hpp:217-218`).
        assert_eq!(ad.base, AccessDetailsBase::new(&op, DfirUnit::L0lu));
        assert_eq!(ad.base.rotation_position, Elements(0));
        assert_eq!(ad.base.expected_total_elements, Elements(0));
        assert!(ad.base.extents.is_empty());
        assert_eq!(ad.base.total_elements, Elements(0));
        assert_eq!(ad.base.element_width, Bits(0));
        assert_eq!(ad.base.transfer_set.dims, 0);
        assert!(ad.base.transfer_set.constraints.is_empty());
        assert_eq!(ad.base.transfer_order.dims, 0);
        assert!(ad.base.transfer_order.results.is_empty());
    }

    /// A REAL `agen.composite_load_and_store`, because that is what an access detail describes.
    ///
    /// `constructTimeStepsInfo` checks the op it was handed is one of the six composite forms
    /// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:632-635`), so a stand-in op would make
    /// every test below a test of something that cannot reach these setters. It is an HBM-to-LX
    /// transfer — the form `tests/sentient_corpus/group_0__g0_7_matmul.dfir.mlir:25-30` prints, src
    /// on the `hbm` unit and dst on `C0-lx` — with every set, order and time-addr map built through
    /// the same [`plan`](crate::bridges::subtile_to_dataflow_ir::transfer::plan) the emitter uses
    /// rather than transcribed. ⚠️ ITS EXTENTS ARE THIS TEST'S OWN, not that file's: the corpus op
    /// moves `memref<1x2048xf16>`, and nothing below depends on the shapes agreeing with it.
    fn composite_load_and_store() -> agen::Op {
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

    /// The `AccessDetailsAffine` half of a freshly constructed composite. ⛔ `AccessDetailsAffine`'s
    /// own constructor is unit 016 and is not this batch's, so the base is reached through 019.
    fn fresh_affine(op: &agen::Op) -> AccessDetailsAffine<'_> {
        AccessDetailsAffineComposite::new(op, DfirUnit::Lxlu).affine
    }

    /// e017 — `subscripts_map_` starts as C++'s NULL `AffineMap` and the setter installs a real one.
    #[test]
    fn set_subscripts_map_installs_the_ops_own_map() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);
        assert_eq!(affine.subscripts_map, None, "the constructor sets no map");

        let map = AffineMap::identity(3);
        affine.set_subscripts_map(map.clone());
        assert_eq!(affine.subscripts_map, Some(map));
    }

    /// e018 — the `nullptr` key is a named field, so `size() == indices.size() + 1` is structural.
    ///
    /// `MutableAddrSplitting.cpp:717` checks that length relation by hand and `:724` reads the key as
    /// `iter_coeff_dict[nullptr]`; here the constant is not in `per_index` at all.
    #[test]
    fn the_coeff_dict_holds_the_constant_beside_the_indices() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);
        assert_eq!(affine.indices_coeff_dict, IndicesCoeffDict::default());

        // `2*i + 3*j + 10` — the reference's own worked example (Agen/Utils.cpp:80-81).
        affine.set_indices_coeff_dict(IndicesCoeffDict {
            per_index: vec![(Val(1), 2), (Val(4), 3)],
            constant: 10,
        });

        assert_eq!(affine.indices_coeff_dict.per_index.len(), 2);
        assert_eq!(affine.indices_coeff_dict.constant, 10);
        assert_eq!(affine.indices_coeff_dict.per_index[0], (Val(1), 2));
    }

    /// e018 — `operator=` on a `DenseMap` drops every existing entry, so the second call wins whole.
    #[test]
    fn set_indices_coeff_dict_replaces_rather_than_merges() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);

        affine.set_indices_coeff_dict(IndicesCoeffDict {
            per_index: vec![(Val(1), 2), (Val(4), 3)],
            constant: 10,
        });
        affine.set_indices_coeff_dict(IndicesCoeffDict {
            per_index: vec![(Val(7), 1)],
            constant: 0,
        });

        assert_eq!(affine.indices_coeff_dict.per_index, vec![(Val(7), 1)]);
        assert_eq!(affine.indices_coeff_dict.constant, 0);
    }

    /// e019 — the constructor binds `op_` and `comp_` and leaves every other member unset.
    ///
    /// ⛔ THE TWO SENTINELS ARE THE POINT. `burst_index_ = -1` and `interleave_group_index_ = -1`
    /// (`AccessDetails.hpp:324,329`) are `None` here, so nothing can index `time_bounds` with a
    /// freshly constructed object's burst the way `time_bounds[burst_index]` would.
    #[test]
    fn the_composite_constructor_binds_only_the_op_and_the_component() {
        let op = composite_load_and_store();
        let details = AccessDetailsAffineComposite::new(&op, DfirUnit::L3lu);

        assert_eq!(*details.affine.base.op, op);
        assert_eq!(details.affine.base.comp, DfirUnit::L3lu);
        assert_eq!(details.affine.subscripts_map, None);
        assert_eq!(
            details.affine.indices_coeff_dict,
            IndicesCoeffDict::default()
        );
        assert_eq!(details.time_order, None);
        assert_eq!(details.time_set, None);
        assert_eq!(details.time_addr_map, None);
        assert_eq!(details.time_symbols, Vec::new());
        assert_eq!(details.time_offsets, TimeOffsets::default());
        assert_eq!(details.time_bounds, Vec::new());
        assert_eq!(details.burst_index, None);
        assert_eq!(details.interleave_group_index, None);
    }

    /// e020 — the map stored is the one belonging to THIS operand.
    ///
    /// `initialize` picks `getLoadTimeAddrMap()` for `kDirSrc` and `getStoreTimeAddrMap()` for the
    /// dst (`AccessDetails.cpp:532,543`), so two access details over one op hold two different maps.
    #[test]
    fn set_time_addr_map_is_per_operand() {
        let op = composite_load_and_store();
        let mut load = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxlu);
        let mut store = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxsu);

        let load_map = AffineMap::linear(&[128, 1024]);
        let store_map = AffineMap::linear(&[1, 64]);
        load.set_time_addr_map(load_map.clone());
        store.set_time_addr_map(store_map.clone());

        assert_eq!(load.time_addr_map, Some(load_map));
        assert_eq!(store.time_addr_map, Some(store_map));
    }

    /// e021 — `assign` clears first, so a second call replaces the symbols rather than appending.
    #[test]
    fn set_time_symbols_replaces_rather_than_appends() {
        let op = composite_load_and_store();
        let mut details = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxlu);

        details.set_time_symbols(&[Val(3), Val(5)]);
        assert_eq!(details.time_symbols, vec![Val(3), Val(5)]);

        details.set_time_symbols(&[Val(8)]);
        assert_eq!(details.time_symbols, vec![Val(8)]);

        // ⭐ AND THE EMPTY RANGE IS THE ONE OUR OWN ISLAND PRODUCES — every
        // `tests/sentient_corpus/*.dfir.mlir` prints `time_symbols()`.
        details.set_time_symbols(&[]);
        assert_eq!(details.time_symbols, Vec::new());
    }

    /// e022 — all three states of a bound survive the setter as distinct values.
    ///
    /// The vector below is what `setCoalescedBoundValues` leaves behind: the merged trip count on
    /// the innermost dimension of the run and `kCoalesced` on the interior ones
    /// (`AccessDetails.cpp:675-678`).
    #[test]
    fn set_time_bounds_carries_all_three_states() {
        let op = composite_load_and_store();
        let mut details = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxlu);

        details.set_time_bounds(&[
            TimeBound::Steps(1),
            TimeBound::Coalesced,
            TimeBound::Steps(12),
            TimeBound::Variable,
            TimeBound::Steps(0),
        ]);

        assert_eq!(details.time_bounds.len(), 5);
        // ⛔ A COALESCED DIMENSION IS NOT A BOUND OF 1, which is the whole reason the enum exists
        // (`AccessDetails.hpp:249-251`).
        assert_ne!(details.time_bounds[1], TimeBound::Steps(1));
        assert_eq!(details.time_bounds[2], TimeBound::Steps(12));
        assert_ne!(details.time_bounds[3], details.time_bounds[4]);

        // `assign` replaces.
        details.set_time_bounds(&[TimeBound::Steps(2)]);
        assert_eq!(details.time_bounds, vec![TimeBound::Steps(2)]);
    }

    /// e023 — the trailing constant is not a dimension, so `per_dim` indexes with `time_bounds`.
    ///
    /// ⭐⭐ THIS IS THE THREE `DT_CHECK`s DISSOLVED. `time_bounds.size() + 1 == time_offsets.size()`
    /// (`AccessDetails.cpp:684-685`, `:696-701`, `:742-744`) holds by construction once the constant
    /// has its own field, and `computeBurstAndGroup` indexes both vectors with one `i`
    /// (`time_bounds[i]` at `:800`, `time_offsets[i]` at `:820`).
    #[test]
    fn set_time_offsets_keeps_the_constant_out_of_the_dimensions() {
        let op = composite_load_and_store();
        let mut details = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxlu);

        details.set_time_bounds(&[TimeBound::Steps(4), TimeBound::Steps(2)]);
        details.set_time_offsets(TimeOffsets {
            per_dim: vec![1024, 128],
            constant: 64,
        });

        assert_eq!(
            details.time_offsets.per_dim.len(),
            details.time_bounds.len()
        );
        assert_eq!(details.time_offsets.constant, 64);
        assert_eq!(details.time_offsets.per_dim[1], 128);
    }

    /// e024 — the group takes the burst's OWN dimension and the burst moves OUTWARD.
    ///
    /// ⛔ THE DIRECTION IS THE SCAN'S, AND THE SCAN RUNS INWARD-OUT.
    /// `for (int i = time_bounds.size() - 1; i >= 0; --i)` (`AccessDetails.cpp:798`) walks a vector
    /// whose index 0 is the OUTERMOST time dimension (`constructTimeLoops` nests loop `idx` inside
    /// loop `idx - 1`, `Helper.cpp:1808-1857`), so the burst is claimed on the innermost valid
    /// dimension first. This replays the promotion from there: the inner dimension holding the
    /// burst has bound 2 and a nonzero offset, the outer dimension's offset is one load's worth, so
    /// the inner one becomes the interleave group and the burst moves out to it
    /// (`AccessDetails.cpp:812-823`).
    #[test]
    fn the_group_takes_the_old_burst_and_the_burst_moves_outward() {
        let op = composite_load_and_store();
        let mut details = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxlu);

        // Dimension 0 is the outermost; its offset is the 64-element load size the promotion looks
        // for, and the inner dimension carries the bound of 2 that gates it.
        details.set_time_bounds(&[TimeBound::Steps(8), TimeBound::Steps(2)]);
        details.set_time_offsets(TimeOffsets {
            per_dim: vec![64, 4096],
            constant: 0,
        });

        // `i = 1`, the innermost: `if (burst_index < 0) setBurstIndex(i)` — the field itself, not a
        // ported setter.
        details.burst_index = Some(TimeDim(1));
        assert_eq!(details.interleave_group_index, None);
        // ⛔ ASSERTED BEFORE THE `if let` BELOW, so the promotion cannot be skipped silently.
        assert_eq!(details.burst_index, Some(TimeDim(1)));

        // `i = 0`: all three conjuncts hold, so the promotion fires. `getBurstIndex()` is read
        // before it is overwritten, which is why the group gets the OLD value.
        if let Some(old_burst) = details.burst_index {
            details.set_interleave_group_index(old_burst);
            details.burst_index = Some(TimeDim(0));

            assert_eq!(details.interleave_group_index, Some(TimeDim(1)));
            assert_eq!(details.burst_index, Some(TimeDim(0)));
            // ⛔ AND THE INDEX IS USABLE AS A POSITION, which is what `time_bounds[burst_index]` and
            // `time_offsets[group_index]` need (`Helper.cpp:1859-1863`).
            assert_eq!(
                details.time_bounds[old_burst.index()],
                TimeBound::Steps(2),
                "the promoted dimension is the one whose bound gated the promotion"
            );
            assert_eq!(
                details.time_offsets.per_dim[old_burst.index()],
                4096,
                "and `stride_step` comes from the group's offset, not the burst's"
            );
        }
    }

    /// THE TIME NEST'S TRIP COUNT FOR ONE DIMENSION — ⭐ DERIVED FROM A DIFFERENT FILE THAN THE CODE
    /// UNDER TEST, so the expectations below are not this module's own arithmetic played back.
    ///
    /// `constructTimeLoops` reads the bounds this way and no other
    /// (`dcc/src/Conversion/AgenToSentient/Helper.cpp:1815-1820`):
    ///
    /// ```cpp
    /// size_t loop_bound =
    ///     (time_bounds[idx] == AccessDetailsAffineComposite::
    ///                              SpecialTimeBoundValues::kCoalesced
    ///          ? 1
    ///          : time_bounds[idx]);
    /// DT_CHECK(loop_bound > 0);
    /// ```
    fn trip_count(bound: TimeBound) -> u64 {
        match bound {
            TimeBound::Coalesced => 1,
            TimeBound::Steps(n) => n,
            // ⛔ `kInvalid` (`TimeBound::Variable`) REACHES THE `DT_CHECK(loop_bound > 0)` AND FAILS IT — a variable bound
            // never gets as far as a time loop. Zero marks it here so a product that includes one
            // collapses instead of quietly passing.
            TimeBound::Variable => 0,
        }
    }

    fn trips(bounds: &[TimeBound]) -> u64 {
        bounds.iter().copied().map(trip_count).product()
    }

    /// 🎯 002/384 — THE MERGED RUN'S PRODUCT LANDS ON ITS INNERMOST DIMENSION AND THE ONES ABOVE IT
    /// GO TO `kCoalesced`.
    ///
    /// The scenario is `coalesceTimeDimensions`': a four-deep nest `2 × 3 × 4 × 5` whose inner three
    /// dimensions turned out contiguous, so the scan stopped with `outer_dim = 0` and had already
    /// multiplied `3 * 4 * 5 = 60` into `coalesced_bound` (`AccessDetails.cpp:777`).
    #[test]
    fn the_merged_run_collapses_onto_its_innermost_dimension() {
        let mut bounds = [
            TimeBound::Steps(2),
            TimeBound::Steps(3),
            TimeBound::Steps(4),
            TimeBound::Steps(5),
        ];
        set_coalesced_bound_values(
            &mut bounds,
            Some(TimeDim(0)),
            TimeDim(3),
            TimeBound::Steps(60),
        );
        assert_eq!(
            bounds,
            [
                TimeBound::Steps(2),
                TimeBound::Coalesced,
                TimeBound::Coalesced,
                TimeBound::Steps(60),
            ],
            "dimension 0 is the one the run stopped BELOW and keeps its own bound"
        );
    }

    /// 🎯 002/384 — AND THE NEST STILL RUNS THE SAME NUMBER OF TIMES.
    ///
    /// ⛔ THIS IS THE INVARIANT THE OFF-BY-ONES BREAK. Both totals are written out as literals and the
    /// per-dimension rule comes from [`trip_count`] above, i.e. from `Helper.cpp`, not from
    /// [`set_coalesced_bound_values`]: `2 × 3 × 4 × 5 = 120` before, and `2 × 1 × 1 × 60 = 120`
    /// after. Starting the run at `outer_dim` instead of `outer_dim + 1` would give 60, and stopping
    /// at `inner_dim - 1` would give 600.
    #[test]
    fn the_trip_count_product_survives_the_merge() {
        let mut bounds = [
            TimeBound::Steps(2),
            TimeBound::Steps(3),
            TimeBound::Steps(4),
            TimeBound::Steps(5),
        ];
        assert_eq!(trips(&bounds), 120, "2 * 3 * 4 * 5");
        set_coalesced_bound_values(
            &mut bounds,
            Some(TimeDim(0)),
            TimeDim(3),
            TimeBound::Steps(60),
        );
        assert_eq!(trips(&bounds), 120, "2 * 1 * 1 * 60");
    }

    /// 🎯 002/384 — A RUN THAT REACHES THE TOP OF THE NEST IS `outer_dim == -1`, WHICH IS [`None`].
    ///
    /// `coalesceTimeDimensions`' scan runs down to `-1` deliberately — *"time_index_outer goes down to
    /// -1 to catch the outermost time dim"* (`AccessDetails.cpp:753`) — and then dimension 0 IS part
    /// of the run. `2 × 3 × 4 × 5 = 120` all on the innermost.
    #[test]
    fn an_absent_outer_dimension_merges_from_dimension_zero() {
        let mut bounds = [
            TimeBound::Steps(2),
            TimeBound::Steps(3),
            TimeBound::Steps(4),
            TimeBound::Steps(5),
        ];
        set_coalesced_bound_values(&mut bounds, None, TimeDim(3), TimeBound::Steps(120));
        assert_eq!(
            bounds,
            [
                TimeBound::Coalesced,
                TimeBound::Coalesced,
                TimeBound::Coalesced,
                TimeBound::Steps(120),
            ]
        );
        assert_eq!(trips(&bounds), 120, "one loop of 120, three of 1");
    }

    /// 🎯 002/384 — DIMENSIONS OUTSIDE THE RUN ARE NOT TOUCHED AT EITHER END.
    ///
    /// Five deep, run over 2..=3 only: dimensions 0 and 1 sit above it, dimension 4 below it.
    #[test]
    fn the_dimensions_outside_the_run_keep_their_bounds() {
        let mut bounds = [
            TimeBound::Steps(2),
            TimeBound::Steps(3),
            TimeBound::Steps(4),
            TimeBound::Steps(5),
            TimeBound::Steps(6),
        ];
        set_coalesced_bound_values(
            &mut bounds,
            Some(TimeDim(1)),
            TimeDim(3),
            TimeBound::Steps(20),
        );
        assert_eq!(
            bounds,
            [
                TimeBound::Steps(2),
                TimeBound::Steps(3),
                TimeBound::Coalesced,
                TimeBound::Steps(20),
                TimeBound::Steps(6),
            ],
            "dimension 4 is INSIDE the merged one and is not part of the run"
        );
        assert_eq!(
            trips(&bounds),
            720,
            "2 * 3 * 1 * 20 * 6 == 2 * 3 * 4 * 5 * 6"
        );
    }

    /// 🎯 002/384 — WITH NOTHING BETWEEN THEM ONLY THE INNER BOUND IS WRITTEN.
    ///
    /// ⭐ THE REFERENCE NEVER ASKS FOR THIS, and that is the point of testing it: the call is guarded
    /// by `if ((time_index_inner - time_index_outer) > 1)` (`AccessDetails.cpp:759`), so adjacent
    /// dimensions are not a merge. The function still has to be well behaved on it — the empty range
    /// marks nothing, and the bound written is the caller's own single bound.
    #[test]
    fn adjacent_dimensions_mark_nothing_as_coalesced() {
        let mut bounds = [TimeBound::Steps(4), TimeBound::Steps(7)];
        set_coalesced_bound_values(
            &mut bounds,
            Some(TimeDim(0)),
            TimeDim(1),
            TimeBound::Steps(7),
        );
        assert_eq!(bounds, [TimeBound::Steps(4), TimeBound::Steps(7)]);
    }

    /// 🎯 002/384 — A COALESCED DIMENSION IS NOT A DIMENSION OF ONE STEP, AND THE TYPE KEEPS THEM
    /// APART.
    ///
    /// ⛔⛔ THIS IS WHY [`TimeBound`] IS AN ENUM. The reference's comment says it outright: *"eg to
    /// distinguish coalesced bounds (equivalent in value to 1) from non-coalesceable bounds with
    /// value 1"* (`AccessDetails.hpp:249-251`). Both run their loop once, and only one of them is
    /// skipped by `computeBurstAndGroup`'s `continue` (`AccessDetails.cpp:803-805`).
    #[test]
    fn a_coalesced_bound_is_distinguishable_from_a_single_step() {
        assert_ne!(TimeBound::Coalesced, TimeBound::Steps(1));
        assert_eq!(
            trip_count(TimeBound::Coalesced),
            trip_count(TimeBound::Steps(1))
        );
    }

    /// 🎯 002/384 — AN OUT-OF-RANGE INNER DIMENSION WRITES NOTHING RATHER THAN OUT OF BOUNDS.
    ///
    /// ⭐ THE ONE DELIBERATE DIVERGENCE from `time_bounds[inner_dim] = coalesced_bound;`. No caller
    /// reaches it; if one ever did, the reference would store past the vector AND mark every
    /// dimension it does have as coalesced, which loses the whole nest's trip count. Dropping the
    /// marking with the store is what keeps the product below intact.
    #[test]
    fn an_inner_dimension_past_the_end_leaves_the_nest_alone() {
        let mut bounds = [TimeBound::Steps(2), TimeBound::Steps(3)];
        set_coalesced_bound_values(&mut bounds, None, TimeDim(9), TimeBound::Steps(6));
        assert_eq!(bounds, [TimeBound::Steps(2), TimeBound::Steps(3)]);
        assert_eq!(
            trips(&bounds),
            6,
            "2 * 3, not the 1 a partial merge would leave"
        );
    }

    /// 🎯 002-008 — THE SIX MEMBERS THESE SETTERS OWN START AT THE REFERENCE'S OWN INITIALISERS.
    ///
    /// ⛔ `shuffle_mode_` STARTS AT `"noshuffle"`, NOT EMPTY (`AccessDetails.hpp:181`), and
    /// `memory_index_` starts at `kMax` (`:151`) — the two that are not zero values. The other four
    /// are default-constructed. [`the_affine_constructor_delegates_op_and_comp_and_defaults_the_rest`]
    /// covers the members entries 009-015 own.
    #[test]
    fn the_setter_owned_members_start_at_the_references_initialisers() {
        let op = vector_load();
        let ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.shuffle_mode, sen::ShuffleMode::NoShuffle);
        assert_eq!(ad.shuffle_mode.spelling(), "noshuffle");
        assert!(ad.indices.is_empty());
        assert!(ad.layout_coeffs.is_empty());
        assert_eq!(ad.mem_view_start_addr, None);
        assert_eq!(ad.mem_view_layout_map, None);
        assert_eq!(
            ad.memory_index, None,
            "the reference's kMax — set by constructDetails, not before"
        );
    }

    /// 🎯 003/384 — `setIndices` REPLACES THE LIST.
    ///
    /// ⛔⛔ THE SECOND CALL IS REAL. `AccessDetailsAffine::initialize` fills the list from the op
    /// (`AccessDetails.cpp:347`); `constructIndices` then reads it back (`:358`) and writes a SHORTER
    /// one over it, because the constant subscripts move into the map itself (`:380`). The
    /// reference's `assign` throws the first list away — appending would leave four subscripts on a
    /// two-dimensional access.
    #[test]
    fn setting_the_indices_twice_keeps_only_the_second_list() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_indices(&[Val(3), Val(4)]);
        assert_eq!(ad.indices, vec![Val(3), Val(4)]);
        ad.set_indices(&[Val(11)]);
        assert_eq!(ad.indices, vec![Val(11)], "assign, not append");
    }

    /// 🎯 006/384 — AND SO DOES `setLayoutCoeffs`, WHICH IS THE SAME `assign`.
    ///
    /// ⭐ THE COEFFICIENTS ARE SIGNED AND ORDERED, one per view dimension plus the trailing constant
    /// term (`VectorChainToSentientPT/LoweringXRF.cpp:50`).
    #[test]
    fn setting_the_layout_coefficients_twice_keeps_only_the_second_list() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_layout_coeffs(&[LayoutCoeff(128), LayoutCoeff(1), LayoutCoeff(0)]);
        ad.set_layout_coeffs(&[LayoutCoeff(512), LayoutCoeff(-4), LayoutCoeff(64)]);
        assert_eq!(
            ad.layout_coeffs,
            vec![LayoutCoeff(512), LayoutCoeff(-4), LayoutCoeff(64)],
            "a stride may be negative, and the last entry is the constant term"
        );
    }

    /// 🎯 004/384 + 007/384 — THE VIEW'S START ADDRESS AND ITS LAYOUT MAP BOTH GO FROM ABSENT TO SET.
    ///
    /// `initializeMemViewInfo` writes the pair together, from a `get_logical_memory_view`
    /// (`AccessDetails.cpp:277-278`) or from a paged view (`:282-283`), and `Helper.cpp:901` later
    /// rewrites the address alone once the base becomes a mutable value.
    #[test]
    fn the_view_start_address_and_layout_map_go_from_absent_to_set() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        // `affine_map<(d0, d1) -> (d0 * 128 + d1)>` — a row-major 2-D view, 128 elements to a row.
        let layout = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(0).times(128).plus(AffineExpr::dim(1))],
        };
        ad.set_mem_view_start_addr(Val(21));
        ad.set_mem_view_layout_map(layout.clone());
        assert_eq!(ad.mem_view_start_addr, Some(Val(21)));
        assert_eq!(ad.mem_view_layout_map, Some(layout));

        // ⛔ AND THE ADDRESS IS REWRITABLE — `Helper.cpp:1043` does exactly this, remapping the
        // address onto its clone when the op is copied into the time loops.
        ad.set_mem_view_start_addr(Val(40));
        assert_eq!(ad.mem_view_start_addr, Some(Val(40)));
    }

    /// 🎯 005/384 — THE MEMORY INDEX GOES FROM ABSENT TO ONE OF THE FOUR OPERANDS, AND STAYS SET.
    ///
    /// ⛔ THE THREE `DT_CHECK_MSG(getMemoryIndex() != kMax, ...)`s (`AccessDetails.cpp:296`, `:443`,
    /// `:859`) ARE THIS TEST'S FIRST LINE. `kMax` is not a variant, so the only way to observe "not
    /// set" is `None`, and no argument to the setter can restore it.
    #[test]
    fn the_memory_index_starts_absent_and_names_one_operand() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.memory_index, None);
        ad.set_memory_index(MemoryOperandIndex::DirSrc);
        assert_eq!(ad.memory_index, Some(MemoryOperandIndex::DirSrc));
        // A composite load-and-store's store half — `constructDetails` sets `kDirDst` for it.
        ad.set_memory_index(MemoryOperandIndex::DirDst);
        assert_eq!(ad.memory_index, Some(MemoryOperandIndex::DirDst));
    }

    /// 🎯 008/384 — THE ONLY SHUFFLE MODE THE REFERENCE EVER SETS IS `splat`.
    ///
    /// ⛔⛔ THE `std::string` WAS A CLOSED SET ALL ALONG. `setShuffleMode("splat")`
    /// (`AccessDetails.cpp:231`) is the sole call in `dcc/src`, against a `"noshuffle"` default —
    /// and both spellings are enumerators of `SentientShuffleModeAttr`. The spellings are asserted
    /// because they are what reaches the emitted attribute.
    #[test]
    fn the_shuffle_mode_moves_from_noshuffle_to_splat() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        assert_eq!(ad.shuffle_mode.spelling(), "noshuffle");
        ad.set_shuffle_mode(sen::ShuffleMode::Splat);
        assert_eq!(ad.shuffle_mode, sen::ShuffleMode::Splat);
        assert_eq!(ad.shuffle_mode.spelling(), "splat");
    }

    /// ⭐ `assign` REPLACES. Two sets in a row leave only the second — the difference between
    /// `assign` and `append`, and the reason `e155_updateSymbolicAccessDetails` can call this
    /// repeatedly as it re-derives the strides for a new loop nest.
    #[test]
    fn set_strides_replaces_the_previous_list() {
        let op = vector_load();
        let mut details = AccessDetailsSymbolic::new(&op, DfirUnit::Lxlu);
        assert_eq!(details.strides(), &[]);

        details.set_strides(&[Val(3), Val(4), Val(5)]);
        assert_eq!(details.strides(), &[Val(3), Val(4), Val(5)]);

        details.set_strides(&[Val(9)]);
        assert_eq!(
            details.strides(),
            &[Val(9)],
            "assign() replaces the strides; appending would leave the previous nest's in front"
        );
    }

    /// An empty set of strides is a set of strides — `assign` from an empty range empties the list.
    #[test]
    fn set_strides_accepts_none() {
        let op = vector_load();
        let mut details = AccessDetailsSymbolic::new(&op, DfirUnit::Lxlu);
        details.set_strides(&[Val(1)]);
        details.set_strides(&[]);
        assert_eq!(details.strides(), &[]);
    }

    /// ⭐ A FRESH CONTAINER HAS NOTHING — the C++'s `index_mapping_((int)kMax, -1)`.
    #[test]
    fn a_fresh_container_has_no_operand() {
        let container = AccessContainer::<Val>::default();
        for moi in MemoryOperandIndex::ALL {
            assert!(!container.has(moi), "{moi:?} was never inserted");
        }
    }

    /// ⛔⛔ ENTRY **ZERO** IS AN ENTRY. `index_mapping_[moi] = 0` is the FIRST insertion, which is
    /// `kDirSrc` on almost every memory op; a `has` that tested for a nonzero slot would answer
    /// `false` for it and every `get` behind it would refuse.
    #[test]
    fn the_first_entry_ever_inserted_reads_as_present() {
        let container = AccessContainer {
            entries: vec![Val(7)],
            slots: [Some(0), None, None, None],
        };
        assert!(container.has(MemoryOperandIndex::DirSrc));
        assert_eq!(container.entries(), &[Val(7)]);
    }

    /// ⭐ ONE FILLED SLOT IS ONE FILLED SLOT. A container holding the two INDIRECT operands of a
    /// `composite_indirect_load_and_store` answers for exactly those two.
    #[test]
    fn has_answers_per_operand() {
        let container = AccessContainer {
            entries: vec![Val(10), Val(11)],
            slots: [None, Some(0), None, Some(1)],
        };
        assert!(container.has(MemoryOperandIndex::IndSrc));
        assert!(container.has(MemoryOperandIndex::IndDst));
        assert!(!container.has(MemoryOperandIndex::DirSrc));
        assert!(!container.has(MemoryOperandIndex::DirDst));
    }

    /// The slots are the C++ enumerator values, which is what `(int)moi` indexes with.
    #[test]
    fn the_slots_are_the_cpp_enumerator_values() {
        assert_eq!(MemoryOperandIndex::DirSrc.slot(), 0);
        assert_eq!(MemoryOperandIndex::IndSrc.slot(), 1);
        assert_eq!(MemoryOperandIndex::DirDst.slot(), 2);
        assert_eq!(MemoryOperandIndex::IndDst.slot(), 3);
        assert_eq!(MemoryOperandIndex::ALL.len(), MemoryOperandIndex::COUNT);
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  143/384 — `constructExtentAndTotalElements`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// THE VIEW THE FIXTURE LOADS FROM — `memref<8x64xf16>`, row major.
    ///
    /// `affine_map<(d0, d1) -> (d0 * 64 + d1)>`, which is what a `get_logical_memory_view` over that
    /// memref carries: 64 elements to a row, one to a column.
    fn row_major_8x64() -> AffineMap {
        AffineMap::linear(&[64, 1])
    }

    /// A SET WITH NO UPPER BOUND ON ITS ONE DIMENSION — `affine_set<(d0) : (d0 >= 0)>`.
    ///
    /// ⭐ `IntegerSet::from_sizes` CANNOT BUILD THIS, and that is the point: every set this campaign
    /// mints is a bound PAIR per dimension, so the reference's *"Not in a hyper-rectangular form"*
    /// refusal is only reachable from a set written by hand.
    fn unbounded_above() -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0),
                is_equality: false,
            }],
        }
    }

    /// 🎯 143/384 — ONE PINNED DIMENSION AND ONE 64-ELEMENT SPAN GIVE EXTENTS `[1, 64]` AND 64
    /// ELEMENTS.
    ///
    /// ⭐ THIS IS THE FIXTURE'S OWN TRANSFER. `agen.vector_load %view[0, 0]` off `memref<8x64xf16>`
    /// into `vector<64xf16>`: the load set pins the row and spans the 64 columns, the load order is
    /// the identity, and the expected count comes from the return type
    /// (`AccessDetails.cpp:308-309`). All three agree, so the walk ends in
    /// [`TransferExtents::Rectangular`] with `layout_coeffs_` installed as a side effect (`:45`).
    #[test]
    fn a_row_of_sixty_four_elements_reads_back_as_one_pinned_dimension_and_a_span() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(IntegerSet::from_sizes(&[1, 64]));
        ad.set_transfer_order(AffineMap::identity(2));
        ad.set_mem_view_layout_map(row_major_8x64());
        ad.set_expected_total_elements(Elements(LOADED.len));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::Rectangular
        );

        assert_eq!(ad.extents, vec![Elements(1), Elements(64)]);
        assert_eq!(ad.total_elements, Elements(64));
        // ⛔ THE COEFFICIENTS ARE A SIDE EFFECT OF THIS FUNCTION, not of the layout setter — the
        // strides plus the trailing constant term.
        assert_eq!(
            ad.layout_coeffs,
            vec![LayoutCoeff(64), LayoutCoeff(1), LayoutCoeff(0)]
        );
    }

    /// 🎯 143/384 — `total_elements` MULTIPLIES INTO WHATEVER IS ALREADY THERE.
    ///
    /// ⛔⛔ THE ACCUMULATOR IS READ, NOT RESET. `auto total_elements = getTotalElements();`
    /// (`AccessDetails.cpp:54`) starts from the member, and `constructChunkAndShuffleInfo` runs this
    /// function on the SAME object after having written it (`:220`), so a second walk multiplies
    /// rather than replaces. Seeding it with 2 turns the 64 above into 128 — which is also why the
    /// element-count cross-check below has to be against a count the return type states.
    #[test]
    fn a_nonzero_total_is_multiplied_into_rather_than_replaced() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(IntegerSet::from_sizes(&[1, 64]));
        ad.set_transfer_order(AffineMap::identity(2));
        ad.set_mem_view_layout_map(row_major_8x64());
        ad.set_total_elements(Elements(2));
        ad.set_expected_total_elements(Elements(128));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::Rectangular
        );
        assert_eq!(ad.total_elements, Elements(128), "2 * 1 * 64");
    }

    /// 🎯 143/384 — A TRANSPOSING LOAD ORDER TRANSPOSES THE EXTENTS.
    ///
    /// ⛔⛔ THE ORDER IS NOT DECORATION. `composeMatchingMap(transfer_order)` re-expresses the set in
    /// the ORDER's input space (`:35-40`), so the same rectangle walked `(d1, d0)` yields the extents
    /// the other way round — 64 then 1 rather than 1 then 64. The element count is invariant, which
    /// is exactly why it cannot catch a transposition and the extents must be checked directly.
    #[test]
    fn a_transposing_load_order_transposes_the_extents() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(IntegerSet::from_sizes(&[1, 64]));
        // `affine_map<(d0, d1) -> (d1, d0)>`.
        ad.set_transfer_order(AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(1), AffineExpr::dim(0)],
        });
        // ⭐ AND THE LAYOUT HAS TO FOLLOW. `affine_map<(d0, d1) -> (d0 + d1 * 64)>` is column major
        // (`layout_coeffs[0] < layout_coeffs[last_index]`), so the limit on the leading extent is
        // `layout_coeffs[1] / layout_coeffs[0]` = 64 (`:69-70`) rather than the row-major ratio — the
        // transposed walk is legal only against the transposed layout.
        ad.set_mem_view_layout_map(AffineMap::linear(&[1, 64]));
        ad.set_expected_total_elements(Elements(64));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::Rectangular
        );
        assert_eq!(ad.extents, vec![Elements(64), Elements(1)]);
        assert_eq!(ad.total_elements, Elements(64));
    }

    /// 🎯 143/384 — AN EXTENT WIDER THAN THE NEXT STRIDE ALLOWS IS REFUSED, AND THE LIMIT IS THE
    /// STRIDE RATIO.
    ///
    /// *"Extent requsted along a dimension is more than its limit"* (`:71-74`). Row major, so
    /// dimension 1's limit is `layout_coeffs[0] / layout_coeffs[1]` = 64/1 = 64 (`:66-67`); a set
    /// spanning 128 columns of a 64-column row would read into the next row.
    #[test]
    fn an_extent_wider_than_the_stride_ratio_is_refused_with_that_ratio() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(IntegerSet::from_sizes(&[1, 128]));
        ad.set_transfer_order(AffineMap::identity(2));
        ad.set_mem_view_layout_map(row_major_8x64());
        ad.set_expected_total_elements(Elements(128));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::ExtentOverLimit {
                dim: TransferDim(1),
                width: 128,
                limit: 64,
            }
        );
        // ⛔ AND NOTHING IS ROLLED BACK — the refusal returns before `setExtents` (`:87`), so the
        // extents stay as they were while the coefficients from `:45` are installed.
        assert!(ad.extents.is_empty());
        assert_eq!(
            ad.layout_coeffs,
            vec![LayoutCoeff(64), LayoutCoeff(1), LayoutCoeff(0)]
        );
    }

    /// 🎯 143/384 — THE OUTERMOST DIMENSION OF A ROW-MAJOR LAYOUT HAS NO STRIDE ABOVE IT, SO ITS
    /// LIMIT IS `width + 1`.
    ///
    /// ⭐ `size = (i == 0) ? (width + 1) : ..` (`:66`) — a limit deliberately one MORE than the width
    /// being tested, which is the reference saying "unbounded" without writing a special case. All
    /// eight rows of the view are therefore loadable at once, and the check never fires on `i == 0`.
    #[test]
    fn the_leading_row_major_dimension_is_never_over_its_limit() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        // All 8 rows and all 64 columns — 512 elements.
        ad.set_transfer_set(IntegerSet::from_sizes(&[8, 64]));
        ad.set_transfer_order(AffineMap::identity(2));
        ad.set_mem_view_layout_map(row_major_8x64());
        ad.set_expected_total_elements(Elements(512));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::Rectangular
        );
        assert_eq!(ad.extents, vec![Elements(8), Elements(64)]);
        assert_eq!(ad.total_elements, Elements(512));
    }

    /// 🎯 143/384 — A COUNT THAT DISAGREES WITH THE RETURN TYPE IS REFUSED **AFTER** EVERYTHING IS
    /// WRITTEN.
    ///
    /// *"Number of elements in return type not matching with load_set/store_set elements"*
    /// (`:89-94`). ⛔ THE ORDER MATTERS: `setTotalElements` and `setExtents` run at `:86-87`, before
    /// the comparison, so the members hold the SET's answer and the refusal only reports the
    /// disagreement.
    #[test]
    fn a_set_that_disagrees_with_the_return_type_still_writes_the_sets_own_count() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(IntegerSet::from_sizes(&[1, 64]));
        ad.set_transfer_order(AffineMap::identity(2));
        ad.set_mem_view_layout_map(row_major_8x64());
        // A `vector<32xf16>` return type over a 64-element load set.
        ad.set_expected_total_elements(Elements(32));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::ElementCountMismatch {
                expected: Elements(32),
                found: Elements(64),
            }
        );
        assert_eq!(ad.total_elements, Elements(64), "the set's count, written");
        assert_eq!(ad.extents, vec![Elements(1), Elements(64)]);
    }

    /// 🎯 143/384 — A DIMENSION WITH ONLY A LOWER BOUND IS NOT A HYPER-RECTANGLE.
    ///
    /// *"Not in a hyper-rectangular form"* (`:84`), the `else` of both predicates: `isDimValueZero`
    /// finds no equality and `isDimAConstantRange` finds a min without a max
    /// (`dialect_utils/Agen/Utils.cpp:163-169`).
    #[test]
    fn a_dimension_with_no_upper_bound_is_not_hyper_rectangular() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(unbounded_above());
        ad.set_transfer_order(AffineMap::identity(1));
        ad.set_mem_view_layout_map(AffineMap::linear(&[1]));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::NotHyperRectangular {
                dim: TransferDim(0)
            }
        );
    }

    /// 🎯 143/384 — AN UPPER BOUND BELOW THE LOWER ONE GIVES A NEGATIVE WIDTH, WHICH IS ITS OWN
    /// REFUSAL.
    ///
    /// *"Extent along a dimension is negative"* (`:80-81`). `5 <= d0 <= 2` is a constant range by
    /// `isDimAConstantRange`'s reckoning — it finds both bounds — and its `max - min + 1` is `-2`.
    /// ⛔ THE REFERENCE'S GUARD IS `width > 0`, so a width of exactly zero takes this arm too.
    #[test]
    fn an_upper_bound_below_the_lower_one_is_a_negative_extent() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        // `affine_set<(d0) : (d0 - 5 >= 0, -d0 + 2 >= 0)>`.
        ad.set_transfer_set(IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0).plus(AffineExpr::Const(-5)),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(2)),
                    is_equality: false,
                },
            ],
        });
        ad.set_transfer_order(AffineMap::identity(1));
        ad.set_mem_view_layout_map(AffineMap::linear(&[1]));

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::ExtentNotPositive {
                dim: TransferDim(0),
                width: -2,
            }
        );
    }

    /// 🎯 143/384 — WITH NO LAYOUT MAP THERE IS NOTHING TO TAKE COEFFICIENTS FROM.
    ///
    /// ⭐ THE ONE OUTCOME WITH NO `emitError` BEHIND IT. `getMemViewLayoutMap()` is a null `AffineMap`
    /// until `initializeMemViewInfo` writes it, and `getMapCoefficients` would dereference it
    /// (`dialect_utils/Agen/Utils.cpp:65-71`); the reference's own call order makes that unreachable
    /// (`:461-464` runs entry 145 first), and [`TransferExtents::MemoryViewUnresolved`] is what a
    /// caller that got the order wrong is told instead of a crash.
    #[test]
    fn a_transfer_with_no_memory_view_yet_resolves_nothing() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_transfer_set(IntegerSet::from_sizes(&[1, 64]));
        ad.set_transfer_order(AffineMap::identity(2));
        assert_eq!(ad.mem_view_layout_map, None);

        assert_eq!(
            ad.construct_extent_and_total_elements(),
            TransferExtents::MemoryViewUnresolved
        );
        assert!(ad.layout_coeffs.is_empty(), "nothing was installed");
        assert_eq!(ad.total_elements, Elements(0));
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  144/384 — `constructLdOrStType`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// 🎯 144/384 — L0 TAKES THE **CHUNK** SIZE AND EVERY OTHER COMPONENT TAKES THE WHOLE TRANSFER.
    ///
    /// ⛔⛔ THE FIELD READ IS `expected_total_elements_`, NOT `total_elements_`, even though the
    /// header calls it "total_elements" (`AccessDetails.hpp:208-210`):
    ///
    /// ```cpp
    /// void AccessDetailsBase::constructLdOrStType() {
    ///   setLdOrStSize(is_any_of(getComp(), L0LU, L0SU) ? getChunkSize()
    ///                                                  : getExpectedTotalElements());
    /// }
    /// ```
    ///
    /// The two counts are seeded from different places and only agree once
    /// `constructExtentAndTotalElements` has confirmed they do — so on a `ld_or_st_size` derived from
    /// a transfer that has not been checked yet, it is the RETURN TYPE's count that reaches the
    /// instruction.
    #[test]
    fn the_ld_or_st_size_is_the_chunk_on_l0_and_the_whole_transfer_elsewhere() {
        let op = vector_load();
        for comp in [DfirUnit::L0lu, DfirUnit::L0su] {
            let mut ad = AccessDetailsBase::new(&op, comp);
            ad.chunk_size = Elements(8);
            ad.set_expected_total_elements(Elements(64));
            ad.construct_ld_or_st_type();
            assert_eq!(ad.ld_or_st_size, Elements(8), "{comp:?} takes the chunk");
        }
        for comp in [
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::L3lu,
            DfirUnit::L3su,
        ] {
            let mut ad = AccessDetailsBase::new(&op, comp);
            ad.chunk_size = Elements(8);
            ad.set_expected_total_elements(Elements(64));
            ad.construct_ld_or_st_type();
            assert_eq!(
                ad.ld_or_st_size,
                Elements(64),
                "{comp:?} takes the whole transfer"
            );
        }
    }

    /// 🎯 144/384 — AND IT READS THE **EXPECTED** COUNT, WHICH THE SET-DERIVED ONE CANNOT STAND IN
    /// FOR.
    ///
    /// ⛔ THE TWO ARE SEPARATE MEMBERS AND THIS IS THE TEST THAT SAYS WHICH ONE. With
    /// `expected_total_elements_ = 64` and `total_elements_ = 512`, reading the wrong field gives an
    /// LDST eight times too long — and `computeBurstAndGroup` compares a time offset against exactly
    /// this value (`:820`), so the interleave group would be claimed on the wrong dimension too.
    #[test]
    fn the_ld_or_st_size_is_not_the_set_derived_count() {
        let op = vector_load();
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.set_expected_total_elements(Elements(64));
        ad.set_total_elements(Elements(512));
        ad.construct_ld_or_st_type();
        assert_eq!(ad.ld_or_st_size, Elements(64));
        assert_ne!(ad.ld_or_st_size, ad.total_elements);
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  145/384 — `initializeMemViewInfo`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// A `dataflow.get_logical_memory_view %unit, %start {layout_map}` binding [`Val`] 1.
    fn logical_view() -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: Val(1),
            from: Val(0),
            start: Val(21),
            layout: row_major_8x64(),
            ty: MemRef {
                shape: vec![8, 64],
                elem: ElemType::F16,
            },
        })
    }

    /// The paged form of the same view — one page covering the whole memref.
    fn paged_view() -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
            dataflow::PagedMemView {
                result: Val(1),
                unit: Val(0),
                start_addr: Val(21),
                pages: vec![dataflow::Page {
                    idx_set: dataflow::PageRect {
                        spans: vec![
                            dataflow::PageSpan { lo: 0, hi: 7 },
                            dataflow::PageSpan { lo: 0, hi: 63 },
                        ],
                    },
                    start_addr: Val(22),
                }],
                layout: row_major_8x64(),
                ty: MemRef {
                    shape: vec![8, 64],
                    elem: ElemType::F16,
                },
            },
        )))
    }

    /// 🎯 145/384 — THE THREE MEMBERS ARE WRITTEN TOGETHER FROM THE OP THAT DEFINES THE `mem_ref`.
    ///
    /// ```cpp
    /// if (auto mem_view = mem_ref.getDefiningOp<dataflow::GetLogicalMemoryViewOp>()) {
    ///   setMemViewLayoutMap(mem_view.getLayoutMap());
    ///   setMemViewStartAddr(mem_view.getStartAddress());
    ///   setMemory(mem_view.getFromUnit());
    /// } else if (auto paged = ..GetPagedLogicalMemoryViewOp>()) { .. } else {
    ///   return op->emitError("Expecting a memory view producing operation");
    /// }
    /// ```
    ///
    /// ⛔⛔ THE `emitError` IS [`MemViewSource::of`]'S [`None`], WHICH IS WHY THIS FUNCTION CANNOT
    /// REFUSE. There is no argument to it that names an op that is not a view, so the third branch has
    /// nothing left to report — the same witness-constructor shape as `EnclosingLoop::of`.
    #[test]
    fn the_view_writes_the_layout_the_start_and_the_unit_together() {
        let op = vector_load();
        let scope = vec![logical_view()];
        let source = MemViewSource::of(Val(1), &scope).expect("a get_logical_memory_view");

        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.initialize_mem_view_info(source);

        assert_eq!(ad.mem_view_layout_map, Some(row_major_8x64()));
        assert_eq!(ad.mem_view_start_addr, Some(Val(21)));
        assert_eq!(ad.memory, Some(Val(0)), "the unit, not the view");
    }

    /// 🎯 145/384 — A PAGED VIEW ANSWERS WITH ITS OWN THREE, AND WITH `%start_addr` RATHER THAN A
    /// PAGE'S.
    ///
    /// ⛔ `getStartAddr()` AND NOT A PAGE'S. The reference reads the paged op's own three getters
    /// (`:282-284`); a page's `start_addr` is RELATIVE to this one, so taking it would place every
    /// access at the wrong base. [`Val`] 22 below is the page's and must not appear.
    #[test]
    fn a_paged_view_answers_with_the_views_own_start_and_not_a_pages() {
        let op = vector_load();
        let scope = vec![paged_view()];
        let source = MemViewSource::of(Val(1), &scope).expect("a paged view");

        let mut ad = AccessDetailsBase::new(&op, DfirUnit::Lxlu);
        ad.initialize_mem_view_info(source);

        assert_eq!(ad.mem_view_layout_map, Some(row_major_8x64()));
        assert_eq!(ad.mem_view_start_addr, Some(Val(21)));
        assert_ne!(ad.mem_view_start_addr, Some(Val(22)));
        assert_eq!(ad.memory, Some(Val(0)));
    }

    /// 🎯 145/384 — AND AN OP THAT IS NOT A VIEW IS *"Expecting a memory view producing operation"*.
    #[test]
    fn an_operand_that_is_not_a_view_has_no_source() {
        let scope = vec![DfirOp::Arith(arith::Op::Constant {
            result: Val(1),
            value: 0,
        })];
        assert!(MemViewSource::of(Val(1), &scope).is_none());
        // And a value nothing in scope defines at all.
        assert!(MemViewSource::of(Val(99), &scope).is_none());
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  146/384 — `constructIndices`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// A scope holding an `affine.for %arg0 = 0 to 8` whose induction variable is [`Val`] 30, an
    /// `scf.for` binding [`Val`] 31, an `arith.constant 0` binding [`Val`] 40, and an `arith.addi`
    /// binding [`Val`] 50.
    fn subscript_scope() -> Vec<DfirOp> {
        vec![
            DfirOp::Affine(affine::Op::For {
                iv: Val(30),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(8),
                carried: vec![affine::Carried {
                    init: Val(41),
                    arg: Val(32),
                    result: Val(33),
                }],
                body: vec![],
                dbg_name: None,
            }),
            DfirOp::Scf(scf::Op::For {
                iv: Val(31),
                lo: Val(42),
                hi: Val(43),
                step: Val(44),
                carried: vec![],
                body: vec![],
                dbg_name: None,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(40),
                value: 0,
            }),
            DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: Val(50),
                lhs: Val(30),
                rhs: Val(40),
                ty: ScalarTy::Index,
            })),
        ]
    }

    /// 🎯 146/384 — A CONSTANT SUBSCRIPT LEAVES `indices_` AND MOVES INTO THE MAP.
    ///
    /// ```cpp
    /// for (auto index : llvm::enumerate(indices)) { .. }
    /// setIndices(new_indices);
    /// if (fold_operands) {
    ///   auto operand_exprs = ..;                      // dim(n) or constant, per operand
    ///   subscripts_map = subscripts_map.replaceDimsAndSymbols(operand_exprs, {}, ndims, 0);
    ///   setSubscriptsMap(subscripts_map);
    /// }
    /// ```
    ///
    /// ⛔⛔ THE RENUMBERING IS THE WHOLE OF IT. `%view[%i, 0]` keeps one index and its map must become
    /// one-dimensional — `(d0, d1) -> (d0, d1)` folds to `(d0) -> (d0, 0)`. Leaving the arity at two
    /// would make every later `compose` read a dimension the access no longer has.
    #[test]
    fn a_constant_subscript_folds_into_the_map_and_renumbers_the_rest() {
        let op = composite_load_and_store();
        let scope = subscript_scope();
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30), Val(40)]);
        affine.set_subscripts_map(AffineMap::identity(2));

        assert_eq!(affine.construct_indices(&scope), ConstructedIndices::Affine);

        assert_eq!(affine.base.indices, vec![Val(30)], "the constant is gone");
        assert_eq!(
            affine.subscripts_map,
            Some(AffineMap {
                dims: 1,
                syms: 0,
                results: vec![AffineExpr::dim(0), AffineExpr::Const(0)],
            })
        );
    }

    /// 🎯 146/384 — WITH NO CONSTANT AMONG THEM THE MAP IS LEFT EXACTLY AS IT WAS.
    ///
    /// ⭐ `fold_operands` GATES THE REWRITE (`:376-383`), and an `scf.for`'s induction variable
    /// passes the affine-loop test alongside an `affine.for`'s: `isa<affine::AffineForOp, scf::ForOp>`
    /// (`:361-362`).
    #[test]
    fn two_loop_iterators_leave_the_map_untouched() {
        let op = composite_load_and_store();
        let scope = subscript_scope();
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30), Val(31)]);
        affine.set_subscripts_map(AffineMap::identity(2));

        assert_eq!(affine.construct_indices(&scope), ConstructedIndices::Affine);
        assert_eq!(affine.base.indices, vec![Val(30), Val(31)]);
        assert_eq!(affine.subscripts_map, Some(AffineMap::identity(2)));
    }

    /// 🎯 146/384 — A LOOP'S **CARRIED** ARGUMENT PASSES THE TEST TOO.
    ///
    /// ⛔⛔ AND THE REFERENCE MEANS IT TO. The check is on the owning op's TYPE — *"if the operand is a
    /// block argument of an affine loop"* — not on whether the value is that loop's induction
    /// variable, so `iter_args(%arg1 = %c0)`'s `%arg1` is kept in `indices_` as readily as `%arg0`.
    /// [`Val`] 32 is the carried argument of the fixture's `affine.for`.
    #[test]
    fn a_carried_argument_counts_as_a_loop_iterator() {
        let op = composite_load_and_store();
        let scope = subscript_scope();
        assert_eq!(
            SubscriptOperand::of(Val(32), &scope),
            SubscriptOperand::LoopIterator
        );

        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(32)]);
        affine.set_subscripts_map(AffineMap::identity(1));
        assert_eq!(affine.construct_indices(&scope), ConstructedIndices::Affine);
        assert_eq!(affine.base.indices, vec![Val(32)]);
    }

    /// 🎯 146/384 — A COMPOSITE'S `load_iv` IS A REGION ARGUMENT OF SOMETHING THAT IS NOT A LOOP.
    ///
    /// *"The loop iterators involved in the agen memory operation subscripts have to be affine
    /// loops"* (`:363-365`). ⛔ AND THE REFUSAL RETURNS BEFORE `setIndices` (`:374`), so the original
    /// list survives intact — a refused access is not left with a truncated subscript list.
    #[test]
    fn a_region_argument_of_a_non_loop_is_not_an_affine_iterator() {
        let op = composite_load_and_store();
        let mut scope = subscript_scope();
        scope.push(DfirOp::Agen(op.clone()));

        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30), Val(9)]);
        affine.set_subscripts_map(AffineMap::identity(2));

        assert_eq!(
            affine.construct_indices(&scope),
            ConstructedIndices::NotAnAffineLoopIterator { index: Val(9) }
        );
        assert_eq!(
            affine.base.indices,
            vec![Val(30), Val(9)],
            "the refusal returns before setIndices"
        );
        assert_eq!(affine.subscripts_map, Some(AffineMap::identity(2)));
    }

    /// 🎯 146/384 — A COMPUTED SUBSCRIPT IS NEITHER AN ITERATOR NOR A CONSTANT.
    ///
    /// *"All the map operands need to be either loop iterators or constant values"* (`:370-372`).
    /// [`Val`] 50 is `arith.addi %arg0, %c0` — affine in form, but not a value this lowering can
    /// attach a coefficient to.
    #[test]
    fn a_computed_subscript_is_refused_and_the_indices_survive() {
        let op = composite_load_and_store();
        let scope = subscript_scope();
        assert_eq!(
            SubscriptOperand::of(Val(50), &scope),
            SubscriptOperand::Computed
        );

        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(50)]);
        affine.set_subscripts_map(AffineMap::identity(1));

        assert_eq!(
            affine.construct_indices(&scope),
            ConstructedIndices::NotAnIteratorOrConstant { index: Val(50) }
        );
        assert_eq!(affine.base.indices, vec![Val(50)]);
    }

    /// 🎯 146/384 — AN ALL-CONSTANT ACCESS LEAVES NO INDICES AND A NULLARY MAP.
    ///
    /// ⭐ THIS IS THE FIXTURE'S STORE HALF: `dst_indices` is five `arith.constant 0`s, which is what
    /// `tests/sentient_corpus`' composite transfers write. The map that comes out takes no dimensions
    /// at all, and its results are the constants themselves.
    #[test]
    fn an_all_constant_access_leaves_a_nullary_map() {
        let op = composite_load_and_store();
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(40),
                value: 0,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(45),
                value: 3,
            }),
        ];
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(40), Val(45)]);
        affine.set_subscripts_map(AffineMap::identity(2));

        assert_eq!(affine.construct_indices(&scope), ConstructedIndices::Affine);
        assert!(affine.base.indices.is_empty());
        assert_eq!(
            affine.subscripts_map,
            Some(AffineMap {
                dims: 0,
                syms: 0,
                results: vec![AffineExpr::Const(0), AffineExpr::Const(3)],
            })
        );
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  147/384 — `constructIteratorCoefficients`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// 🎯 147/384 — EACH ITERATOR'S COEFFICIENT IS ITS STRIDE THROUGH THE **COMPOSED** MAP.
    ///
    /// ```cpp
    /// AffineMap composed_load_order_map = getTransferOrder().compose(getSubscriptsMap());
    /// AffineMap composed_layout_map = getMemViewLayoutMap().compose(composed_load_order_map);
    /// SmallVector<int64_t> composed_layout_coeffs = constructDimCoefficients(composed_layout_map);
    /// ..
    /// indices_coeff_dict[index] = composed_layout_coeffs[i];
    /// indices_coeff_dict[nullptr] = composed_layout_coeffs.back();
    /// ```
    ///
    /// ⛔⛔ TWO COMPOSITIONS, NOT ONE, AND THE ORDER IS OUTERMOST-FIRST. The subscripts map goes
    /// through the transfer ORDER before it meets the LAYOUT, so a transposing order changes every
    /// coefficient. Here both are the identity over `(i, j)` and the layout is row-major
    /// `d0 * 64 + d1`, so `%i` strides 64 elements and `%j` strides 1 — which is the reference's own
    /// worked example, `2*i + 3*j + 10` in shape (`dialect_utils/Agen/Utils.cpp:73-81`).
    #[test]
    fn each_iterator_gets_its_stride_through_the_composed_layout() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30), Val(31)]);
        affine.base.set_transfer_order(AffineMap::identity(2));
        affine.base.set_mem_view_layout_map(row_major_8x64());
        affine.set_subscripts_map(AffineMap::identity(2));

        affine.construct_iterator_coefficients();

        assert_eq!(
            affine.indices_coeff_dict,
            IndicesCoeffDict {
                per_index: vec![(Val(30), 64), (Val(31), 1)],
                constant: 0,
            }
        );
    }

    /// 🎯 147/384 — A TRANSPOSING TRANSFER ORDER SWAPS THE COEFFICIENTS.
    ///
    /// ⛔ THE PROOF THAT THE ORDER IS COMPOSED IN AND NOT IGNORED. Same indices, same layout, order
    /// `(d0, d1) -> (d1, d0)`: `%i` now strides 1 and `%j` strides 64. An implementation that read the
    /// layout coefficients directly would give the same answer as the test above and pass it.
    #[test]
    fn a_transposing_transfer_order_swaps_the_iterator_coefficients() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30), Val(31)]);
        affine.base.set_transfer_order(AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(1), AffineExpr::dim(0)],
        });
        affine.base.set_mem_view_layout_map(row_major_8x64());
        affine.set_subscripts_map(AffineMap::identity(2));

        affine.construct_iterator_coefficients();

        assert_eq!(
            affine.indices_coeff_dict,
            IndicesCoeffDict {
                per_index: vec![(Val(30), 1), (Val(31), 64)],
                constant: 0,
            }
        );
    }

    /// 🎯 147/384 — THE CONSTANT TERM IS THE COMPOSED MAP'S, AND IT IS NOT AN ITERATOR'S.
    ///
    /// ⭐⭐ `composed_layout_coeffs.back()` IS ALWAYS THE CONSTANT. The reference writes
    /// `(composed_layout_coeffs.size() == indices.size() + 1) ? back() : 0` (`:419-421`), and the
    /// ternary can only ever take the first arm: a flattened row is `numDims + numSymbols + 1` wide
    /// and the dimensions ARE the surviving indices. A subscripts map with an offset — `(d0) -> (d0 + 3)`
    /// over a row-major layout — puts `3 * 64 = 192` there.
    #[test]
    fn the_constant_term_is_the_composed_offset_and_not_an_index() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30)]);
        affine.base.set_transfer_order(AffineMap::identity(2));
        affine.base.set_mem_view_layout_map(row_major_8x64());
        // `(d0) -> (d0 + 3, 0)` — a row offset of three on a one-iterator access.
        affine.set_subscripts_map(AffineMap {
            dims: 1,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).plus(AffineExpr::Const(3)),
                AffineExpr::Const(0),
            ],
        });

        affine.construct_iterator_coefficients();

        assert_eq!(
            affine.indices_coeff_dict,
            IndicesCoeffDict {
                per_index: vec![(Val(30), 64)],
                constant: 192,
            }
        );
    }

    /// 🎯 147/384 — WITH EITHER MAP STILL ABSENT THE DICTIONARY IS EMPTY RATHER THAN WRONG.
    ///
    /// ⭐ THE `nullptr` KEY IS THE REFERENCE'S ONLY ENTRY ON THIS PATH TOO. `subscripts_map_` is null
    /// until `constructIndices` runs and `mem_view_layout_map_` until entry 145 does; MLIR's
    /// `compose` on a null map is a null dereference, so this is the ordering the reference's callers
    /// keep (`:466-474`) rather than a case it handles.
    #[test]
    fn an_unresolved_map_leaves_the_dictionary_at_its_default() {
        let op = composite_load_and_store();
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30)]);
        affine.base.set_transfer_order(AffineMap::identity(1));

        // Layout present, subscripts absent.
        affine.base.set_mem_view_layout_map(row_major_8x64());
        affine.construct_iterator_coefficients();
        assert_eq!(affine.indices_coeff_dict, IndicesCoeffDict::default());

        // Subscripts present, layout absent.
        let mut affine = fresh_affine(&op);
        affine.base.set_indices(&[Val(30)]);
        affine.base.set_transfer_order(AffineMap::identity(1));
        affine.set_subscripts_map(AffineMap::identity(1));
        affine.construct_iterator_coefficients();
        assert_eq!(affine.indices_coeff_dict, IndicesCoeffDict::default());
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  148/384 — `computeBurstAndGroup`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// A composite whose bounds, offsets, component and load size are all set at once.
    fn burst_case<'a>(
        op: &'a agen::Op,
        comp: DfirUnit,
        bounds: &[TimeBound],
        offsets: &[i64],
        ld_or_st_size: Elements,
    ) -> AccessDetailsAffineComposite<'a> {
        let mut details = AccessDetailsAffineComposite::new(op, comp);
        details.set_time_bounds(bounds);
        details.set_time_offsets(TimeOffsets {
            per_dim: offsets.to_vec(),
            constant: 0,
        });
        details.affine.base.ld_or_st_size = ld_or_st_size;
        details
    }

    /// 🎯 148/384 — THE BURST LANDS ON THE **INNERMOST** VALID DIMENSION.
    ///
    /// ⛔⛔ THE SCAN'S DIRECTION IS THE RESULT. `for (int i = time_bounds.size() - 1; i >= 0; --i)`
    /// (`:798`) over a vector whose index 0 is the OUTERMOST dimension
    /// (`Helper.cpp:1808-1857` nests loop `idx` inside `idx - 1`), so dimension 1 is examined first
    /// and takes the field. Dimension 0's `Steps(0)` is the reference's empty
    /// `else if (curr_bound == 0) {}` arm — it neither claims a field nor stops the search.
    #[test]
    fn the_burst_is_claimed_on_the_innermost_valid_dimension() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[TimeBound::Steps(0), TimeBound::Steps(8)],
            &[0, 512],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(1)));
        assert_eq!(details.interleave_group_index, None);
    }

    /// 🎯 148/384 — WITH A SECOND VALID DIMENSION THE GROUP TAKES THE OLD BURST AND THE BURST MOVES
    /// OUTWARD.
    ///
    /// ⭐ ALL THREE CONJUNCTS HOLD (`:815-817`): the held burst's bound is 2, the current dimension's
    /// offset is exactly one load's worth of elements, and the burst's own offset is nonzero. The
    /// group therefore gets the value `getBurstIndex()` had BEFORE the reassignment, which is what
    /// makes `stride_step` read the inner dimension's offset (`Helper.cpp:1859-1863`).
    #[test]
    fn a_second_valid_dimension_promotes_the_burst_and_seats_the_group() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[TimeBound::Steps(8), TimeBound::Steps(2)],
            &[64, 4096],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(0)));
        assert_eq!(details.interleave_group_index, Some(TimeDim(1)));
    }

    /// 🎯 148/384 — A BURST BOUND THAT IS NOT 2 OR 4 TAKES NO GROUP, AND THE SCAN STOPS ANYWAY.
    ///
    /// ⛔⛔ THE `return` AT `:824` IS OUTSIDE THE `if`. Once a burst is held, a non-L3 component gets
    /// exactly ONE attempt at the group and then returns whether or not it took it — so dimension 0
    /// below is never examined even though it would promote. A `return` moved inside the `if` would
    /// give `burst = 0`, `group = 1` here.
    #[test]
    fn a_burst_bound_outside_two_or_four_ends_the_scan_without_a_group() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[
                TimeBound::Steps(2),
                TimeBound::Steps(4),
                TimeBound::Steps(8),
            ],
            &[64, 64, 4096],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(2)), "the innermost");
        assert_eq!(details.interleave_group_index, None);
    }

    /// 🎯 148/384 — AN OFFSET THAT IS NOT ONE WHOLE TRANSFER TAKES NO GROUP EITHER.
    ///
    /// `time_offsets[i] == getLdOrStSize()` (`:816`) — the outer dimension must step by exactly one
    /// load's worth of elements for the two inner transfers to be interleaved rather than scattered.
    /// 128 against a 64-element load fails it, and the `return` still fires.
    #[test]
    fn an_offset_that_is_not_one_transfer_wide_takes_no_group() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[TimeBound::Steps(8), TimeBound::Steps(2)],
            &[128, 4096],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(1)));
        assert_eq!(details.interleave_group_index, None);
    }

    /// 🎯 148/384 — AND NEITHER DOES A BURST WHOSE OWN OFFSET IS ZERO.
    ///
    /// `time_offsets[burst_index] != 0` (`:817`): two transfers at the same address are not two
    /// interleaved transfers.
    #[test]
    fn a_burst_with_no_offset_of_its_own_takes_no_group() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[TimeBound::Steps(8), TimeBound::Steps(2)],
            &[64, 0],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(1)));
        assert_eq!(details.interleave_group_index, None);
    }

    /// 🎯 148/384 — A COALESCED DIMENSION IS SKIPPED AND THE SCAN CARRIES ON PAST IT.
    ///
    /// ⛔ `continue`, NOT `return` (`:803-805`). A merged-away dimension neither takes a field nor
    /// stops the search, so the promotion below still fires from dimension 0 across a coalesced
    /// dimension 1 — and it is dimension 2, not dimension 1, that becomes the group.
    #[test]
    fn a_coalesced_dimension_is_skipped_and_the_scan_continues_past_it() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[
                TimeBound::Steps(8),
                TimeBound::Coalesced,
                TimeBound::Steps(2),
            ],
            &[64, 0, 4096],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(0)));
        assert_eq!(details.interleave_group_index, Some(TimeDim(2)));
    }

    /// 🎯 148/384 — A VARIABLE BOUND TERMINATES THE SCAN.
    ///
    /// ⛔⛔ `else if (curr_bound < 0) return;` (`:806-807`) — and `kCoalesced` is `-2`, so the ORDER of
    /// the two arms is what keeps a merged dimension from ending the search. Dimension 0 here would
    /// promote on every conjunct; the variable dimension 1 above it means it is never reached.
    #[test]
    fn a_variable_bound_terminates_the_scan_before_the_promotion() {
        let op = composite_load_and_store();
        let mut details = burst_case(
            &op,
            DfirUnit::Lxlu,
            &[
                TimeBound::Steps(8),
                TimeBound::Variable,
                TimeBound::Steps(2),
            ],
            &[64, 0, 4096],
            Elements(64),
        );

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, Some(TimeDim(2)));
        assert_eq!(details.interleave_group_index, None);
    }

    /// 🎯 148/384 — L3 CLAIMS NO GROUP AND KEEPS SCANNING OUTWARD.
    ///
    /// ⛔⛔ THE L3 ARM FALLS THROUGH — it takes neither the promotion nor the `return`, because
    /// `!is_any_of(getComp(), L3LU, L3SU)` is false and the `if` has no `else` (`:813-825`). L3's LDST
    /// has no interleave-group field to fill. The identical input on an LX component promotes
    /// ([`a_second_valid_dimension_promotes_the_burst_and_seats_the_group`]); here the burst stays on
    /// the innermost dimension and the loop runs to the top of the nest.
    #[test]
    fn an_l3_component_claims_no_group_and_scans_to_the_top() {
        let op = composite_load_and_store();
        for comp in [DfirUnit::L3lu, DfirUnit::L3su] {
            let mut details = burst_case(
                &op,
                comp,
                &[TimeBound::Steps(8), TimeBound::Steps(2)],
                &[64, 4096],
                Elements(64),
            );

            details.compute_burst_and_group();

            assert_eq!(details.burst_index, Some(TimeDim(1)), "{comp:?}");
            assert_eq!(details.interleave_group_index, None, "{comp:?}");
        }
    }

    /// 🎯 148/384 — A NEST WITH NO TIME DIMENSIONS LEAVES BOTH FIELDS UNSET.
    ///
    /// ⭐ WHICH IS THE `-1`/`-1` THE EMITTER READS AS "no burst, no group": `size() - 1` on an empty
    /// vector is where the reference's `int i` signedness earns its keep, and the [`None`]s here are
    /// the same answer without it.
    #[test]
    fn an_empty_time_nest_claims_neither_field() {
        let op = composite_load_and_store();
        let mut details = burst_case(&op, DfirUnit::Lxlu, &[], &[], Elements(64));

        details.compute_burst_and_group();

        assert_eq!(details.burst_index, None);
        assert_eq!(details.interleave_group_index, None);
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════════
    //  149/384 and 150/384 — `AccessContainer::insert` and `::get`
    // ══════════════════════════════════════════════════════════════════════════════════════════════

    /// 🎯 149/384 + 150/384 — A FILLED SLOT ANSWERS `has` AND HANDS BACK THE ENTRY.
    #[test]
    fn filling_a_vacancy_makes_the_operand_present_and_readable() {
        let mut container = AccessContainer::<Val>::default();
        container
            .vacancy(MemoryOperandIndex::DirSrc)
            .expect("a fresh container has every slot empty")
            .fill(Val(7));

        assert!(container.has(MemoryOperandIndex::DirSrc));
        assert_eq!(container.get(MemoryOperandIndex::DirSrc), Some(&Val(7)));
        assert_eq!(container.entries(), &[Val(7)]);
    }

    /// 🎯 149/384 — *"trying to insert into a slot that is already filled"* IS A [`None`] BEFORE THE
    /// CALL EXISTS.
    ///
    /// ⛔⛔ THE CHECK IS INVERTED, NOT MOVED (`AccessDetails.hpp:389-390`). A second
    /// [`AccessContainer::vacancy`] on a filled operand answers [`None`], so there is nothing to
    /// `.fill()` — and because the vacancy borrows the container exclusively and `fill` consumes it,
    /// two live vacancies on one container do not compile either.
    #[test]
    fn a_filled_slot_yields_no_second_vacancy() {
        let mut container = AccessContainer::<Val>::default();
        container
            .vacancy(MemoryOperandIndex::DirSrc)
            .expect("empty")
            .fill(Val(7));

        assert!(container.vacancy(MemoryOperandIndex::DirSrc).is_none());
        assert_eq!(container.entries(), &[Val(7)], "and nothing was appended");
    }

    /// 🎯 149/384 — THE SLOT RECORDS THE POSITION THE ENTRY IS **ABOUT TO** TAKE.
    ///
    /// ⛔⛔ `index_mapping_[moi] = this->size();` COMES BEFORE `push_back` (`:393-394`). Filling
    /// `kDirDst` first and `kDirSrc` second — which is not the order the enum declares them in — must
    /// map dst to entry 0 and src to entry 1. Reading `size()` after the push would map every operand
    /// one past its own entry, and `getFirst` would hand a store's details to a load.
    #[test]
    fn the_slot_is_the_position_before_the_push_not_after() {
        let mut container = AccessContainer::<Val>::default();
        container
            .vacancy(MemoryOperandIndex::DirDst)
            .expect("empty")
            .fill(Val(11));
        container
            .vacancy(MemoryOperandIndex::DirSrc)
            .expect("empty")
            .fill(Val(22));

        assert_eq!(container.entries(), &[Val(11), Val(22)], "insertion order");
        assert_eq!(container.slots, [Some(1), None, Some(0), None]);
        assert_eq!(container.get(MemoryOperandIndex::DirDst), Some(&Val(11)));
        assert_eq!(container.get(MemoryOperandIndex::DirSrc), Some(&Val(22)));
    }

    /// 🎯 150/384 — *"no entry exists for the requested memory operand index"* IS [`None`].
    ///
    /// ⭐ AND THE REFERENCE'S OWN CALLERS ALREADY ASK THE QUESTION FIRST — `getFirst` is written
    /// `if (has(kDirSrc)) return get(kDirSrc);` (`:411-418`), so the [`Option`] makes one lookup of
    /// what was two.
    #[test]
    fn an_empty_slot_has_no_entry_to_get() {
        let mut container = AccessContainer::<Val>::default();
        container
            .vacancy(MemoryOperandIndex::IndSrc)
            .expect("empty")
            .fill(Val(3));

        assert_eq!(container.get(MemoryOperandIndex::IndSrc), Some(&Val(3)));
        for moi in [
            MemoryOperandIndex::DirSrc,
            MemoryOperandIndex::DirDst,
            MemoryOperandIndex::IndDst,
        ] {
            assert_eq!(container.get(moi), None, "{moi:?} was never filled");
        }
    }

    /// 🎯 149/384 — *"no more than kMax entries are allowed"* IS STRUCTURALLY UNREACHABLE.
    ///
    /// ⛔⛔ ONE SLOT PER OPERAND AND A SLOT REQUIRED. Filling all four leaves exactly
    /// [`MemoryOperandIndex::COUNT`] entries, and a fifth `fill` has no operand left to name — so the
    /// second `DT_CHECK_MSG` (`:391-392`) is counting against a bound the type already imposes.
    #[test]
    fn every_operand_can_be_filled_once_and_that_is_the_whole_bound() {
        let mut container = AccessContainer::<Val>::default();
        for (i, moi) in MemoryOperandIndex::ALL.into_iter().enumerate() {
            let value = Val(u32::try_from(i).expect("four operands fit a u32"));
            container.vacancy(moi).expect("still empty").fill(value);
        }

        assert_eq!(container.entries().len(), MemoryOperandIndex::COUNT);
        for (i, moi) in MemoryOperandIndex::ALL.into_iter().enumerate() {
            assert!(container.vacancy(moi).is_none(), "{moi:?} is filled");
            let value = Val(u32::try_from(i).expect("four operands fit a u32"));
            assert_eq!(container.get(moi), Some(&value));
        }
    }

    /// 🎯 206/384 — TWO ROWS, ONE CHUNK EACH, 512 ELEMENTS APART, AND A ROTATION ON THE WAY OUT.
    ///
    /// ⭐ A `memref<4x8x64xf16>` walked as `[2][1][64]`: the innermost extent fills its layout ratio
    /// (64 == 64) so the chunk grows to a whole row, the middle dimension stops it (8 != 1) and the
    /// outer non-unit extent seats the stride at the running product 512. The chunk dimension's own
    /// stride is 64 and not 1, which is what keeps
    /// [`ChunkAndShuffleInfo::ChunkStartingOffsetUnsupported`] off it (`:191-197`).
    #[test]
    fn the_chunk_is_the_contiguous_row_and_the_stride_is_the_plane_below_it() {
        let op = agen::Op::VectorLoad {
            result: Val(2),
            view: Val(1),
            indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
            view_ty: MemRef {
                shape: vec![4, 8, 64],
                elem: ElemType::F16,
            },
            ty: LOADED,
        };
        // `vectorchain.rotate %loaded, %c8 : vector<64xf16>` — a right rotation by a constant.
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(3),
                value: 8,
            }),
            DfirOp::VectorChain(vectorchain::Op::Rotate {
                result: Val(4),
                input: Val(2),
                position: Val(3),
                right_shift: true,
                input_ty: LOADED,
                ty: LOADED,
            }),
        ];
        let mut ad = AccessDetailsBase::new(&op, DfirUnit::L0lu);
        // `affine_map<(d0, d1, d2) -> (d0 * 512 + d1 * 64 + d2)>`, flattened.
        ad.set_layout_coeffs(&[
            LayoutCoeff(512),
            LayoutCoeff(64),
            LayoutCoeff(1),
            LayoutCoeff(0),
        ]);
        ad.set_extents(&[Elements(2), Elements(1), Elements(64)]);
        ad.set_total_elements(Elements(128));

        assert_eq!(
            ad.construct_chunk_and_shuffle_info(&scope),
            ChunkAndShuffleInfo::Chunked
        );
        assert_eq!(ad.chunk_size, Elements(64));
        assert_eq!(ad.chunk_stride, Elements(512));
        // The rotation is read off the constant, and no `vectorchain.select` means no splat.
        assert_eq!(ad.rotation_position, Elements(8));
        assert_eq!(ad.shuffle_mode, sen::ShuffleMode::NoShuffle);
    }

    /// 🎯 207/384 — THE LOAD FILLS EIGHT MEMBERS, AND THE LAST TWO COME FROM THE VIEW.
    ///
    /// ⛔ THE WIDTH AND THE COUNT ARE THE **VECTOR**'S, not the memref's: `vector<64xf16>` gives 16
    /// bits and 64 elements where `memref<8x64xf16>` would give 512.
    #[test]
    fn the_load_names_its_own_memref_width_and_count_and_then_resolves_the_view() {
        let op = vector_load();
        let scope = vec![logical_view()];
        let mut ad = AccessDetailsAffine::new(&op, DfirUnit::Lxlu);
        ad.base.set_memory_index(MemoryOperandIndex::DirSrc);

        assert_eq!(ad.initialize(&scope), AffineInitialize::Initialized);
        assert_eq!(ad.base.mem_ref, Some(Val(1)));
        assert_eq!(ad.base.element_width, Bits(16));
        assert_eq!(ad.base.expected_total_elements, Elements(64));
        assert_eq!(ad.base.transfer_order, AffineMap::identity(2));
        let view_ty = MemRef {
            shape: vec![8, 64],
            elem: ElemType::F16,
        };
        assert_eq!(ad.base.transfer_set, agen::access_set(&view_ty, 64));
        // Both subscripts are literals, so the map takes no operand and the index list stays empty.
        assert_eq!(ad.subscripts_map, Some(AffineMap::constants(0, &[0, 0])));
        assert_eq!(ad.base.indices, Vec::new());
        assert_eq!(ad.base.mem_view_layout_map, Some(row_major_8x64()));
        assert_eq!(ad.base.mem_view_start_addr, Some(Val(21)));
    }

    /// 🎯 208/384 — THE RETURNED ENTRY IS THE ONE IN THE CONTAINER, AND IT IS WRITABLE.
    ///
    /// ⭐ `return get(moi)` (`:385`) IS WHAT THE CALLER USES: `gatherAffineLoadStoreDetails` inserts a
    /// record and then keeps writing into it, so a copy would drop every later write.
    #[test]
    fn the_inserted_entry_is_handed_back_for_writing_at_the_slot_it_took() {
        let mut container = AccessContainer::<Val>::default();
        container
            .vacancy(MemoryOperandIndex::DirDst)
            .expect("empty")
            .fill(Val(7));

        let entry = container
            .vacancy(MemoryOperandIndex::DirSrc)
            .expect("still empty")
            .emplace_insert(Val(0));
        *entry = Val(9);

        assert_eq!(container.get(MemoryOperandIndex::DirSrc), Some(&Val(9)));
        assert_eq!(container.get(MemoryOperandIndex::DirDst), Some(&Val(7)));
        assert_eq!(container.entries(), [Val(7), Val(9)]);
    }

    /// 🎯 209/384 — THE SOURCE FIRST, THEN THE DESTINATION, AND NOTHING ELSE COUNTS.
    ///
    /// ⛔ AN INDIRECT-ONLY CONTAINER IS THE `llvm_unreachable` (`:417`), which is [`None`] here.
    #[test]
    fn the_first_operand_is_the_direct_source_or_else_the_direct_destination() {
        let mut only_dst = AccessContainer::<Val>::default();
        only_dst
            .vacancy(MemoryOperandIndex::DirDst)
            .expect("empty")
            .fill(Val(5));
        assert_eq!(only_dst.get_first(), Some(&Val(5)));

        let mut both = only_dst;
        both.vacancy(MemoryOperandIndex::DirSrc)
            .expect("still empty")
            .fill(Val(4));
        assert_eq!(both.get_first(), Some(&Val(4)));

        let mut only_indirect = AccessContainer::<Val>::default();
        only_indirect
            .vacancy(MemoryOperandIndex::IndSrc)
            .expect("empty")
            .fill(Val(6));
        assert_eq!(only_indirect.get_first(), None);
    }
}
