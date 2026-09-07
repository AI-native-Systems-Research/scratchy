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
use crate::islands::dataflow_ir::dialects::{Val, agen};
use crate::islands::dataflow_ir::ty::{AffineMap, IntegerSet};
use crate::islands::sentient::dialects::sentient as sen;
use crate::units::DfirUnit;

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
/// `time_offsets[burst_index]` (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:820-822`) — and
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
/// ⛔ `Default` IS THE CONSTRUCTED-BUT-UNSET STATE and it is not an invention: an affine expression
/// with no constant term has constant term 0, which is what `calculateTimeOffsets`'s own
/// `const_value` fallback says (`dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Agen/Utils.cpp:73-76`).
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
/// ⛔ A VEC, NOT A MAP, BECAUSE THE ORDER IS THE LOOP NESTING. The producer inserts in `indices_`
/// order under its own comment "Sort the indices from outermost to innnermost" (`Utils.cpp:68-71`);
/// a `BTreeMap<Val, i64>` would reorder that by SSA number and a `HashMap` by hash, and
/// `MutableAddrSplitting`'s weights are accumulated per nesting level (`:725-738`).
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
/// `shuffle_mode_` and `indices_`; `memory_`, `mem_ref_`, `chunk_size_`, `chunk_stride_` and
/// `ld_or_st_size_` still arrive with the `construct*` entries that write them. The order below is the C++'s own declaration order
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
            // `Value mem_view_start_addr_;` (`hpp:169`) — a null `Value` until a view resolves.
            mem_view_start_addr: None,
            // `AffineMap mem_view_layout_map_;` (`hpp:172`) — null for the same reason.
            mem_view_layout_map: None,
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
                constraints: Vec::new(),
            },
            // `AffineMap transfer_order_;` (`hpp:205`).
            transfer_order: AffineMap {
                dims: 0,
                syms: 0,
                results: Vec::new(),
            },
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
    /// field and moves the burst inward:
    /// ```text
    /// if ((time_bounds[burst_index] == 2 || time_bounds[burst_index] == 4) &&
    ///     time_offsets[i] == getLdOrStSize() && time_offsets[burst_index] != 0) {
    ///   setInterleaveGroupIndex(burst_index);
    ///   setBurstIndex(i);
    /// }
    /// ```
    /// (`AccessDetails.cpp:820-824`). `burst_index` is `>= 0` there — the `< 0` branch above it took
    /// the other path (`:813`) — so the argument is always a real dimension and the `-1` this field
    /// starts at is never passed in. Nothing clears the field, which is why there is no
    /// `Option`-taking form.
    pub fn set_interleave_group_index(&mut self, interleave_group_index: TimeDim) {
        self.interleave_group_index = Some(interleave_group_index);
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
/// Only the field [`e025_setStrides`](Self::set_strides) owns is present; the rest of this class
/// arrives with `e155_updateSymbolicAccessDetails`, `e265_constructDetails` and the base class's own
/// fields, which are other entries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccessDetailsSymbolic {
    /// `strides_` — *"strides used in the indices accesses"* (`AccessDetails.hpp:366`).
    strides: Vec<Val>,
}

impl AccessDetailsSymbolic {
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
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AccessContainer, AccessDetailsAffine, AccessDetailsAffineComposite, AccessDetailsBase,
        AccessDetailsSymbolic, IndicesCoeffDict, LayoutCoeff, MemoryOperandIndex, TimeBound,
        TimeDim, TimeOffsets, sen, set_coalesced_bound_values,
    };
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::dialects::agen;
    use crate::islands::dataflow_ir::dialects::{Index, Op as DfirOp, Val};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef, Vector,
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
    /// every test below a test of something that cannot reach these setters. This is the HBM-to-LX
    /// transfer of `/tmp/ktir_ref/export/debug/dfir.mlir:78-84`, built through the same
    /// [`plan`](crate::bridges::subtile_to_dataflow_ir::transfer::plan) the emitter uses.
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
    /// has its own field, and `computeBurstAndGroup` indexes both vectors with one `i` (`:820-822`).
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

    /// e024 — the group takes the burst's OWN dimension and the burst moves inward.
    ///
    /// This replays `computeBurstAndGroup`'s promotion verbatim: with a burst already claimed on the
    /// outer dimension and its bound 2, the inner dimension whose offset equals the load size takes
    /// the burst and the outer one becomes the interleave group (`AccessDetails.cpp:812-824`).
    #[test]
    fn the_group_takes_the_old_burst_and_the_burst_moves_inward() {
        let op = composite_load_and_store();
        let mut details = AccessDetailsAffineComposite::new(&op, DfirUnit::Lxlu);

        details.set_time_bounds(&[TimeBound::Steps(2), TimeBound::Steps(8)]);
        details.set_time_offsets(TimeOffsets {
            per_dim: vec![4096, 64],
            constant: 0,
        });

        // `if (burst_index < 0) setBurstIndex(i)` — the field itself, not a ported setter.
        details.burst_index = Some(TimeDim(0));
        assert_eq!(details.interleave_group_index, None);

        let old_burst = details.burst_index.expect("the burst was claimed above");
        details.set_interleave_group_index(old_burst);
        details.burst_index = Some(TimeDim(1));

        assert_eq!(details.interleave_group_index, Some(TimeDim(0)));
        assert_eq!(details.burst_index, Some(TimeDim(1)));
        // ⛔ AND THE INDEX IS USABLE AS A POSITION, which is what `time_bounds[burst_index]` needs.
        assert_eq!(
            details.time_bounds[old_burst.index()],
            TimeBound::Steps(2),
            "the promoted dimension is the one whose bound gated the promotion"
        );
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
        let mut details = AccessDetailsSymbolic::default();
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
        let mut details = AccessDetailsSymbolic::default();
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
}
