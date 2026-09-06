// SPDX-License-Identifier: Apache-2.0
//! THE TWO ADDRESSES A TRANSFER CARRIES, ITS INCREMENT, AND THE STORE-SIDE ARRANGEMENT.
//!
//! Ported from `AgenToSentientLoweringPass`: `constructImmutableAddress` (`Helper.cpp:1217`),
//! `setImmutableAddrAndIncrements` (`:1581`), `setsttype` (`:1731`), `processInterleaveOp` (`:305`),
//! `lowerVectorLoadHelper`'s arity rule (`:2899`) and
//! `cleanupTriviallyRedundantSetSendDestination` (`:4084`).
//!
//! # 🛑 THE L3 IS A DIFFERENT MACHINE HERE, AND IT IS DIFFERENT THREE TIMES
//!
//! ⛔⛔ EVERY ADDRESS RULE IN THIS FILE FORKS ON `is_any_of(comp, L3LU, L3SU)`, and the two arms do not
//! merely differ in value — they read **different quantities**:
//!
//! | | L3 | everything else |
//! |---|---|---|
//! | immutable address | the mem-view start address, unchanged | the **updated** start address |
//! | when bursting, what is set | the **increment**, as `total_elements * burst_size` | the **immutable address**, as `stride_size` |
//!
//! ⭐ THE SECOND ROW IS THE ONE TO NOTICE. On the L3 a burst writes the *increment*; off it, the same
//! branch writes the *immutable address* — a different field of the same op, from a different quantity
//! (`:1596-1606`). Reading that as "the burst sets the increment" and applying it uniformly puts a
//! stride where a start address belongs.

use crate::arch::{Arch, Elements};
use sys_arch_spec::fields::Gen;
use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::sentient::dialects::sentient::ShuffleMode;
use crate::units::DfirUnit;

/// WHETHER A UNIT TAKES THE L3'S ADDRESS RULES.
#[must_use]
pub const fn is_l3(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::L3lu | DfirUnit::L3su)
}

/// WHICH START ADDRESS BECOMES THE IMMUTABLE ONE — `constructImmutableAddress` (`:1226-1232`).
///
/// ⛔⛔ THE L3 TAKES THE VIEW'S OWN START ADDRESS; EVERYTHING ELSE TAKES THE **UPDATED** ONE. Those are
/// two different values — `MutableStartAddrShifting` (D10) is the pass that produces the updated one, so
/// off the L3 the immutable address is post-shift and on the L3 it is pre-shift. Using one for the
/// other addresses the wrong buffer of a double-buffered pair.
#[must_use]
pub const fn immutable_address(unit: DfirUnit, view_start: Val, updated_start: Val) -> Val {
    if is_l3(unit) { view_start } else { updated_start }
}

/// WHAT A BURSTING TRANSFER SETS — `setImmutableAddrAndIncrements` (`:1594-1610`).
///
/// ⛔⛔ **DIFFERENT FIELDS ON DIFFERENT UNITS.** See this module's table: the L3 arm writes the
/// increment as `total_elements * burst_size`; the other arm writes the immutable address as
/// `stride_size`. One value each, but not the same one, which is why this is an enum rather than a
/// number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurstSetting {
    /// The L3 arm: the **increment** takes `total_elements * burst_size`.
    Increment(Elements),
    /// Everything else: the **immutable address** takes `stride_size`.
    ImmutableAddress(Elements),
}

impl BurstSetting {
    /// What this unit's burst sets.
    ///
    /// ⭐ THE L3 ARM HAS AN UNPROVEN ASSUMPTION IN THE REFERENCE, left as a comment:
    /// *"TODO: assert that stride_size x precision = stick size"* (`:1597`). So the increment it
    /// computes is only right when the stride happens to be a stick — carried here as a note rather
    /// than as an assertion this bridge is not allowed to make.
    #[must_use]
    pub const fn of(unit: DfirUnit, total_elements: Elements, burst: u64, stride: Elements) -> BurstSetting {
        if is_l3(unit) {
            BurstSetting::Increment(Elements(total_elements.0 * burst))
        } else {
            BurstSetting::ImmutableAddress(stride)
        }
    }
}

/// THE STORE-SIDE ARRANGEMENT — `setsttype` (`:1731-1781`).
///
/// # 🛑 THE STORE SIDE IS NOT THE LOAD SIDE, AND THREE THINGS DIFFER
///
/// ⛔⛔ COMPARE [`super::ldtype`], WHICH PORTS `setldtype`. Same file, fifty lines apart, and:
///
/// | | load (`setldtype`) | store (`setsttype`) |
/// |---|---|---|
/// | modes | `splat2b`, `zpad16b`, `splat16b` | **`masked2b`, `masked16b`** |
/// | 2-byte repetition | **64** | **1** |
/// | 16-byte repetition | 1 (pad) or 8 (splat) | **1** |
/// | implicit route | yes — picks a mode from the byte count | **none** |
///
/// ⭐ SO A STORE HAS NO IMPLICIT ROUTE AT ALL: without a shuffle the mode stays default whatever the
/// width, where a load would have chosen one. And both store modes want `repetition == 1`, so the
/// repetition count does not distinguish them — the splat width does.
///
/// ⛔ MISREADING EITHER TABLE FOR THE OTHER PRODUCES A MODE THE PARSER KNOWS AND THE HARDWARE DOES NOT
/// EXPECT, which is worse than a rejected attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreArrangement {
    /// `isSplatFromFirstElem(shuffle, 2)` with `repetition == 1` — mode 1.
    Masked2B,
    /// `isSplatFromFirstElem(shuffle, 16)` with `repetition == 1` — mode 2.
    Masked16B,
}

impl StoreArrangement {
    /// The mode this arrangement selects.
    #[must_use]
    pub const fn mode(self) -> ShuffleMode {
        match self {
            StoreArrangement::Masked2B => ShuffleMode::Masked2B,
            StoreArrangement::Masked16B => ShuffleMode::Masked16B,
        }
    }

    /// The repetition count both arrangements require — ⛔ ONE, unlike the load side's 64 and 8.
    pub const REPETITION: u32 = 1;

    /// `total_elements` rewritten to the full stick, as the load side also does.
    #[must_use]
    pub const fn total_elements<A: Arch>(element_bits: u32) -> Elements {
        Elements((A::BYTES_PER_STICK.get() * 8) / element_bits as u64)
    }
}

/// A MEMORY INTERLEAVE'S GRANULARITY — `processInterleaveOp` (`:305-385`).
///
/// ⛔ **L3 ONLY**, and the reference both fails the pass and emits *"only supported in L3 units"*
/// (`:313-317`).
///
/// ⛔ AND THE GRANULARITY DEFAULTS TO THE UNIT'S MAXIMUM BURST, then must be non-zero and no greater
/// than it — *"illegal granularity setting"* (`:322-330`). Zero is called out separately from
/// out-of-range because an absent attribute and an attribute of zero are different inputs that would
/// otherwise both look like "unset".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Granularity(u64);

impl Granularity {
    /// The granularity, defaulting to the L3's maximum burst when the attribute is absent.
    ///
    /// ⛔ `None` FOR ZERO OR OUT-OF-RANGE — the reference's two error arms, as an absence of a legal
    /// value rather than a refusal.
    #[must_use]
    pub const fn checked(stated: Option<u64>, max_burst: u64) -> Option<Granularity> {
        match stated {
            None => Some(Granularity(max_burst)),
            Some(0) => None,
            Some(g) if g > max_burst => None,
            Some(g) => Some(Granularity(g)),
        }
    }

    /// The value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// HOW MANY ACCESSES A LOWERED LOAD NEEDS — `lowerVectorLoadHelper` (`:2908-2925`).
///
/// ⛔ ONE WITHOUT A STORE, TWO WITH ONE, AND THE REFERENCE ASSERTS BOTH: *"single access info needed"*
/// against *"double access info needed"*. A load feeding a store is one `load_and_store` statement
/// describing both ends, so it needs a source access and a destination access — which is the same
/// two-endedness `sentient.load_and_store`'s four addresses carry.
#[must_use]
pub const fn accesses_needed(has_store: bool) -> usize {
    if has_store { 2 } else { 1 }
}

/// WHETHER A GENERATION EMITS `set_send_dst` AT ALL —
/// `cleanupTriviallyRedundantSetSendDestination` (`:4087`).
///
/// ⛔⛔ **NOT BELOW RCUDD1A.** The reference returns immediately when
/// `dccExtContext().getArch() < RCUDD1A_ISA`, under the comment *"We do not generate set_send_dst
/// operations at arch levels that don't support it."* So this is an arch gate on an OP, not on an
/// attribute: on an MPW generation the statement does not exist, and emitting one produces a program
/// for a machine that cannot decode it.
///
/// ⭐⭐ AND FOR US IT IS **VACUOUSLY TRUE**, WHICH IS WORTH SAYING RATHER THAN HIDING.
/// [`crate::arch::IsaGen`] has exactly two members — `Rcudd1a` and `Sen1p5` — so every arch this crate
/// can build for is at or above the gate. The comparison is kept, against the vendor's full ordered
/// [`Gen`] rather than our two-member enum, because the gate is a fact about the ISA that outlives our
/// current arch scope: a future variant is a new ordinal and `>=` keeps working, whereas an
/// `IsaGen`-shaped test would silently have nothing to compare.
///
/// ⛔ IT IS ORDERED, NOT AN EQUALITY. Every generation from RCUDD1A onward has the op.
#[must_use]
pub fn generation_has_set_send_dst(generation: Gen) -> bool {
    generation >= Gen::Rcudd1a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Dd2;

    /// 🎯 THE L3 AND EVERYTHING ELSE TAKE DIFFERENT START ADDRESSES.
    #[test]
    fn the_l3_keeps_the_view_start_address() {
        let (view, updated) = (Val(10), Val(20));
        assert_eq!(immutable_address(DfirUnit::L3lu, view, updated), view);
        assert_eq!(immutable_address(DfirUnit::L3su, view, updated), view);
        // ⛔ OFF THE L3 IT IS THE POST-SHIFT VALUE.
        assert_eq!(immutable_address(DfirUnit::Lxlu, view, updated), updated);
        assert_eq!(immutable_address(DfirUnit::L0su, view, updated), updated);
    }

    /// 🎯 A BURST SETS A DIFFERENT FIELD ON THE L3 THAN OFF IT.
    ///
    /// ⭐ THE POINT OF THE ENUM. Same call, different field written, from a different quantity.
    #[test]
    fn a_burst_sets_the_increment_on_l3_and_the_address_elsewhere() {
        let (total, burst, stride) = (Elements(64), 4, Elements(7));
        assert_eq!(
            BurstSetting::of(DfirUnit::L3lu, total, burst, stride),
            BurstSetting::Increment(Elements(256)),
            "total_elements * burst_size"
        );
        assert_eq!(
            BurstSetting::of(DfirUnit::Lxsu, total, burst, stride),
            BurstSetting::ImmutableAddress(Elements(7)),
            "stride_size, and into a different field"
        );
    }

    /// 🎯 THE STORE MODES ARE `masked*`, AND BOTH WANT ONE REPETITION.
    ///
    /// ⛔ THE LOAD SIDE'S 2-BYTE CASE WANTS 64. If these ever read the same, one of the two tables has
    /// been copied over the other.
    #[test]
    fn the_store_side_differs_from_the_load_side() {
        assert_eq!(StoreArrangement::Masked2B.mode(), ShuffleMode::Masked2B);
        assert_eq!(StoreArrangement::Masked16B.mode(), ShuffleMode::Masked16B);
        assert_eq!(StoreArrangement::REPETITION, 1);
        // ⭐ AND THE LOAD SIDE'S MODES ARE NOT THESE.
        assert_ne!(
            StoreArrangement::Masked2B.mode(),
            super::super::ldtype::Explicit::Splat2BAcross64.mode()
        );
    }

    /// 🎯 A STORE ALSO REWRITES `total_elements` TO THE FULL STICK.
    #[test]
    fn a_store_mode_carries_the_full_stick_count() {
        assert_eq!(StoreArrangement::total_elements::<Dd2>(16), Elements(64));
        assert_eq!(StoreArrangement::total_elements::<Dd2>(8), Elements(128));
    }

    /// 🎯 GRANULARITY DEFAULTS TO THE MAX BURST, AND ZERO IS SEPARATELY ILLEGAL.
    #[test]
    fn granularity_defaults_and_bounds() {
        assert_eq!(Granularity::checked(None, 8).map(Granularity::get), Some(8));
        assert_eq!(Granularity::checked(Some(4), 8).map(Granularity::get), Some(4));
        assert_eq!(Granularity::checked(Some(8), 8).map(Granularity::get), Some(8));
        // ⛔ ZERO IS NOT "UNSET" — the reference names it as its own illegal case.
        assert!(Granularity::checked(Some(0), 8).is_none());
        assert!(Granularity::checked(Some(9), 8).is_none());
    }

    /// 🎯 A LOAD FEEDING A STORE NEEDS TWO ACCESSES.
    #[test]
    fn the_access_arity() {
        assert_eq!(accesses_needed(false), 1);
        assert_eq!(accesses_needed(true), 2);
    }

    /// 🎯 `set_send_dst` DOES NOT EXIST BELOW RCUDD1A, and the test is ordered.
    ///
    /// ⭐ THE MPW CASES ARE THE ONLY REASON THIS FUNCTION HAS A FALSE ARM. Our own `IsaGen` cannot
    /// express them, which is exactly why the comparison is against the vendor's `Gen`.
    #[test]
    fn set_send_dst_is_gated_on_the_generation() {
        assert!(generation_has_set_send_dst(Gen::Rcudd1a));
        assert!(generation_has_set_send_dst(Gen::Sen1p5), "and every later variant");
        assert!(!generation_has_set_send_dst(Gen::Mpw2));
        assert!(!generation_has_set_send_dst(Gen::Mpw3));
        assert!(!generation_has_set_send_dst(Gen::Mpw4));
    }
}
