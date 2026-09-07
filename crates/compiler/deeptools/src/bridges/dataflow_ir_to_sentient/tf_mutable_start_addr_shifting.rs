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

use super::tf_mutable_addr_splitting::{AddrRange, L3Half};
use crate::arch::Arch;

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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Sen1p5};
    use crate::generated::DataType;
    use crate::units::DfirUnit;

    /// 🎯 116/384 — THE DERIVED RANGE IS `2^bitSize * bytesPerStick * 8` ON EACH ARCH.
    ///
    /// `{8, 30, 32, UNSIGNED, true}` for L3LU and L3SU under `coreArch <= RCUDD1A_ISA`, and
    /// `{16, 32, 32, UNSIGNED, true}` above it (`sys-arch-spec/sysdef.cpp:313-360`), with
    /// `bytesPerStick = 128` on both (`:206`).
    #[test]
    fn the_derived_range_is_the_ebr_span_in_bits() {
        for comp in [L3Half::Load, L3Half::Store] {
            assert_eq!(
                max_immutable_range::<Dd2>(comp).bits(),
                (1 << 30) * 128 * 8
            );
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
                clippy::cast_precision_loss,
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
}
