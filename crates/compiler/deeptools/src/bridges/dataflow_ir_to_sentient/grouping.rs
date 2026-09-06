// SPDX-License-Identifier: Apache-2.0
//! WHICH PROGRAM UNITS COLLAPSE, AND HOW ONE TRANSFER'S DESTINATIONS SPLIT.
//!
//! Ported from `ProgramUnitsReductionPass::matchUnits`
//! (`Transform/Dataflow/ProgramUnitsReduction.cpp:69-145`), `areCoreletsDifferent` and
//! `getUnitNameFromAListOfGetUnitOp` (`Conversion/DataflowToSentient/DataflowToSentient.cpp:119-137`),
//! `separateBasedOnDestinationUnits` (`:761-778`), and `CanonicalizeTogglePass::runOnOperation`
//! (`Transform/Dataflow/CanonicalizeToggle.cpp:49-90`).

use crate::islands::dataflow_ir::dialects::Val;
use crate::units::{Core, Corelet, DfirUnit};

/// WHETHER TWO PROGRAM UNITS CAN REDUCE INTO ONE — `matchUnits`' first gate.
///
/// ⛔ **SAME UNIT TYPE IS NECESSARY AND CHECKED FIRST**: `if (base_get_unit.getType() !=
/// curr_get_unit.getType()) return failure()` (`:83-84`). Everything after it — the
/// `OperationEquivalence` analysis with its core/corelet-aware comparator — only runs for units of one
/// type, so it is refining a match rather than finding one.
///
/// ⭐ AND THE ANALYSIS IS THE REST OF THE RULE, not plumbing: the comparator exists so that
/// *"If base_unit uses its default core_id in statement S"* the candidate must use **its own** default
/// there too. So two units are equivalent when their programs agree **modulo a consistent substitution
/// of core and corelet**, not when they are textually identical. That is what makes one program unit
/// serve many cores, and it is why `ProgramUnitsReduction` is D1 — it undoes the per-component minting
/// that would otherwise produce a unit per (op, component).
#[must_use]
pub const fn may_reduce_together(base: DfirUnit, candidate: DfirUnit) -> bool {
    // ⛔ TYPE EQUALITY ONLY. The consistent-substitution half is the caller's, over statements.
    matches!(
        (base, candidate),
        (DfirUnit::Sfp, DfirUnit::Sfp)
            | (DfirUnit::Pe, DfirUnit::Pe)
            | (DfirUnit::Lxlu, DfirUnit::Lxlu)
            | (DfirUnit::Lxsu, DfirUnit::Lxsu)
            | (DfirUnit::Lx, DfirUnit::Lx)
            | (DfirUnit::Hbm, DfirUnit::Hbm)
            | (DfirUnit::L0lu, DfirUnit::L0lu)
            | (DfirUnit::L0su, DfirUnit::L0su)
            | (DfirUnit::L3lu, DfirUnit::L3lu)
            | (DfirUnit::L3su, DfirUnit::L3su)
            | (DfirUnit::SfpState, DfirUnit::SfpState)
            | (DfirUnit::PeState, DfirUnit::PeState)
            | (DfirUnit::SfpRing, DfirUnit::SfpRing)
    ) || matches!((base, candidate), (DfirUnit::PtRow(a), DfirUnit::PtRow(b)) if a.get() == b.get())
}

/// WHICH (core, corelet) A UNIT SITS AT — what `matchUnits` reads to build its substitution.
///
/// ⛔ `corelet` IS OPTIONAL AND THAT IS NOT A CONVENIENCE. The reference reads `getCoreletId` and
/// compares against `-1` throughout (`UnitFiltering.cpp:114`, `cleanup:277`), because an L3 unit is
/// shared across a core's corelets and has none. `Option<Corelet>` is that `-1` with no sentinel to
/// leak into an attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// Which core.
    pub core: Core,
    /// Which corelet, where the unit has one.
    pub corelet: Option<Corelet>,
}

/// WHETHER A UNIT'S CORELET PUTS IT ON THE OTHER SIDE FROM THE SET IT IS BEING JOINED TO —
/// `areCoreletsDifferent` (`DataflowToSentient.cpp:132-137`).
///
/// ⛔ IT IS NOT `a != b`. The reference asks whether *this* unit is corelet 0 **and** any corelet-1
/// unit exists, or the mirror — so it is a question about a unit against two accumulated sets, not
/// about two units. A pairwise inequality would answer differently whenever one side is empty.
#[must_use]
pub const fn corelet_differs(unit_is_corelet_zero: bool, saw_corelet_zero: bool, saw_corelet_one: bool) -> bool {
    (unit_is_corelet_zero && saw_corelet_one) || (!unit_is_corelet_zero && saw_corelet_zero)
}

/// HOW ONE TRANSFER'S DESTINATIONS PARTITION — `separateBasedOnDestinationUnits`
/// (`DataflowToSentient.cpp:761-778`).
///
/// ⛔⛔ FOUR WAYS, AND THE TEST FOR L3 IS A **STRING PREFIX**: `dst_unit.getType().str().substr(0, 2)
/// != "l3"` means LX (`:771`). Everything that is not spelled `l3…` is treated as LX and then split by
/// corelet; only the L3 escapes the corelet question, because it has no corelet.
///
/// ⭐ THE PARTITION EXISTS BECAUSE ONE `dataflow` TRANSFER BECOMES SEVERAL SENTIENT STATEMENTS — one
/// per destination class — so getting the classes wrong does not produce a wrong attribute, it produces
/// the wrong number of statements.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DestinationSplit {
    /// LX destinations on corelet 0.
    pub lx_corelet0: Vec<(Val, Val)>,
    /// LX destinations on corelet 1.
    pub lx_corelet1: Vec<(Val, Val)>,
    /// L3 destinations — ⭐ NOT SPLIT BY CORELET, because an L3 unit has none.
    pub l3: Vec<(Val, Val)>,
    /// Destinations reached through a group rather than a bare unit.
    pub group: Vec<(Val, Val)>,
}

impl DestinationSplit {
    /// Place one (source, destination) pair.
    ///
    /// ⛔ THE ORDER OF THE TESTS IS THE REFERENCE'S: group first (a group is not a `get_unit` at all),
    /// then L3 by prefix, then LX by corelet.
    pub fn place(&mut self, source: Val, destination: Val, unit: Option<DfirUnit>, corelet_zero: bool) {
        match unit {
            // ⭐ NOT A BARE UNIT — reached through a group.
            None => self.group.push((source, destination)),
            Some(DfirUnit::L3lu | DfirUnit::L3su) => self.l3.push((source, destination)),
            Some(_) if corelet_zero => self.lx_corelet0.push((source, destination)),
            Some(_) => self.lx_corelet1.push((source, destination)),
        }
    }

    /// How many sentient statements this transfer becomes — one per non-empty class.
    #[must_use]
    pub fn statement_count(&self) -> usize {
        usize::from(!self.lx_corelet0.is_empty())
            + usize::from(!self.lx_corelet1.is_empty())
            + usize::from(!self.l3.is_empty())
            + usize::from(!self.group.is_empty())
    }
}

/// WHETHER TOGGLE CANONICALISATION APPLIES TO A UNIT — `CanonicalizeTogglePass`
/// (`CanonicalizeToggle.cpp:59-64`).
///
/// ⛔ **L3 HALVES ONLY.** The pass collects candidates with
/// `is_any_of(comp, L3LU, L3SU)` and walks nothing else. A toggle on an LX or L0 view is not
/// canonicalised, so applying this transformation more widely changes programs the reference leaves
/// alone.
#[must_use]
pub const fn toggle_canonicalises(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::L3lu | DfirUnit::L3su)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 🎯 ONLY LIKE UNITS REDUCE, and the PT rows are distinguished by row.
    #[test]
    fn reduction_requires_the_same_unit_type() {
        assert!(may_reduce_together(DfirUnit::Lxlu, DfirUnit::Lxlu));
        assert!(!may_reduce_together(DfirUnit::Lxlu, DfirUnit::Lxsu));
        assert!(!may_reduce_together(DfirUnit::L3lu, DfirUnit::L0lu));
    }

    /// 🎯 `corelet_differs` IS ABOUT A UNIT AGAINST TWO SETS, not two units.
    ///
    /// ⛔ WITH NEITHER SET POPULATED IT IS FALSE for both answers — which a pairwise `!=` could not
    /// express, because there is no second unit to compare against.
    #[test]
    fn corelet_difference_is_against_the_accumulated_sets() {
        assert!(corelet_differs(true, false, true), "corelet 0 and a corelet-1 unit exists");
        assert!(corelet_differs(false, true, false), "corelet 1 and a corelet-0 unit exists");
        assert!(!corelet_differs(true, true, false), "corelet 0 and only corelet-0 units");
        assert!(!corelet_differs(true, false, false), "nothing seen yet");
    }

    /// 🎯 THE FOUR-WAY SPLIT, AND THE L3 ESCAPING THE CORELET QUESTION.
    #[test]
    fn destinations_partition_four_ways() {
        let mut split = DestinationSplit::default();
        split.place(Val(1), Val(2), Some(DfirUnit::Lxlu), true);
        split.place(Val(3), Val(4), Some(DfirUnit::Lxlu), false);
        // ⭐ THE CORELET FLAG IS IGNORED FOR AN L3 — it has none.
        split.place(Val(5), Val(6), Some(DfirUnit::L3su), true);
        split.place(Val(7), Val(8), None, true);
        assert_eq!(split.lx_corelet0.len(), 1);
        assert_eq!(split.lx_corelet1.len(), 1);
        assert_eq!(split.l3.len(), 1);
        assert_eq!(split.group.len(), 1);
        assert_eq!(split.statement_count(), 4, "one statement per non-empty class");
    }

    /// 🎯 ONE CLASS MEANS ONE STATEMENT — the count is what the split is for.
    #[test]
    fn an_unsplit_transfer_is_one_statement() {
        let mut split = DestinationSplit::default();
        split.place(Val(1), Val(2), Some(DfirUnit::L3lu), false);
        split.place(Val(3), Val(4), Some(DfirUnit::L3su), false);
        assert_eq!(split.statement_count(), 1, "both L3, so one class");
    }

    /// 🎯 TOGGLE CANONICALISATION IS L3-ONLY.
    #[test]
    fn only_the_l3_halves_canonicalise_toggles() {
        assert!(toggle_canonicalises(DfirUnit::L3lu));
        assert!(toggle_canonicalises(DfirUnit::L3su));
        for unit in [DfirUnit::Lxlu, DfirUnit::Lxsu, DfirUnit::L0lu, DfirUnit::Pe] {
            assert!(!toggle_canonicalises(unit));
        }
    }
}
