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
use crate::islands::dataflow_ir::dialects::agen;
use crate::islands::dataflow_ir::ty::{AffineMap, IntegerSet};
use crate::units::DfirUnit;

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
/// Only the members whose setters have been ported are declared. `memory_index_`, `layout_coeffs_`,
/// `memory_`, `mem_ref_`, `mem_view_start_addr_`, `mem_view_layout_map_`, `chunk_size_`,
/// `chunk_stride_`, `shuffle_mode_`, `indices_` and `ld_or_st_size_` arrive with entries 003-008 and
/// the `construct*` entries that write them. The order below is the C++'s own declaration order
/// (`hpp:150-210`), so each wave inserts rather than reorders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessDetailsBase<'a> {
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
            op,
            comp,
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
                results: Vec::new(),
            },
        }
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
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{AccessDetailsAffine, AccessDetailsBase};
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::dialects::agen;
    use crate::islands::dataflow_ir::dialects::{Index, Val};
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
}
