// SPDX-License-Identifier: Apache-2.0
//! UNIT CLASSIFICATION AND NAMING, as the lowering asks it.
//!
//! Ported from `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp`
//! (`isSenComponentL0LU` `:96`, `isSenComponentL0SU` `:100`, `ExtendUnitNameToCorelet` `:104`,
//! `isTargetL3` `:1720`) and `AgenToSentientLoweringPass::generateSetSendDestinationStmts`
//! (`Helper.cpp:2731-2779`).

use crate::islands::dataflow_ir::dialects::Val;
use crate::units::{Corelet, DfirUnit};

/// WHETHER A SET-SEND-DESTINATION STATEMENT IS EMITTED AT ALL.
///
/// ⛔⛔ **LXLU ONLY, AND THE REFERENCE RETURNS IMMEDIATELY OTHERWISE.**
/// `generateSetSendDestinationStmts` opens with `if (comp != LXLU) return success()`
/// (`Helper.cpp:2735-2738`). So `sentient.set_send_dst` is not a statement any unit may carry — it
/// belongs to the LX load unit, and emitting one elsewhere is not a different arrangement but a
/// program the reference would never produce.
///
/// ⛔ NOT `Lxsu`. The store unit is excluded even though it is the LX's other half, which is easy to
/// get wrong when a rule is remembered as "the LX does this".
#[must_use]
pub const fn sets_send_destination(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::Lxlu)
}

/// WHETHER A UNIT IS THE L0's LOAD OR STORE HALF — `isSenComponentL0LU` / `isSenComponentL0SU`.
///
/// ⭐ THE REFERENCE ASKS THIS OF THE **GENERIC** COMPONENT, not the specific one:
/// `senCompToGenericComp.at(comp) == L0LU`. A specific component carries its corelet (`l0lu0`,
/// `l0lu1`); the generic one does not, so the question "is this an L0 load unit" is answered after that
/// distinction is dropped. Our [`DfirUnit`] is already the generic vocabulary, so the lookup is the
/// match below.
#[must_use]
pub const fn is_l0_load(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::L0lu)
}

/// The store half — see [`is_l0_load`].
#[must_use]
pub const fn is_l0_store(unit: DfirUnit) -> bool {
    matches!(unit, DfirUnit::L0su)
}

/// WHETHER A QUERY'S TARGET IS AN L3 UNIT — `isTargetL3` (`DataflowToSentient.cpp:1720-1726`).
///
/// ⛔⛔ **IT ANSWERS `true` WHEN IT DOES NOT KNOW.** The reference reads the unit type out of a uniform
/// mapping, tests `substr(0, 2) == "l3"`, and when the mapping yields nothing at all it
/// `return true` — the unknown case is treated as L3, not as not-L3. That default is the whole reason
/// this six-line function is worth porting rather than inlining: a reimplementation that returned
/// `false` on absence would take the other branch everywhere the mapping is silent.
///
/// ⭐ FOR US THE UNIT IS KNOWN, so the unknown arm has no input — but it is named here so that if a
/// path ever does reach this with no unit, the answer is the reference's and not a fresh guess.
#[must_use]
pub const fn is_target_l3(unit: Option<DfirUnit>) -> bool {
    match unit {
        Some(DfirUnit::L3lu | DfirUnit::L3su) => true,
        // ⛔ THE REFERENCE'S DEFAULT: unknown means L3.
        None => true,
        Some(_) => false,
    }
}

/// A UNIT'S SENTIENT NAME, WITH ITS CORELET — `ExtendUnitNameToCorelet`
/// (`DataflowToSentient.cpp:104-114`).
///
/// ⛔⛔ THE SUFFIX IS THE CORELET INDEX, AND A UNIT WITHOUT ONE IS AN ERROR IN THE REFERENCE:
/// *"Unknown corelet information for sentient"* when the `corelet` attribute is absent
/// (`:107-110`). So a sentient unit name is not complete without it — which matches the other end of
/// the ladder, where `Dpc::convertIr2Senprog` recovers the corelet by taking the last character of the
/// unit name (`dpc.cpp:632-637`).
///
/// ⛔ AND THE REFERENCE ONLY EVER WRITES `0` OR `1`: it tests the attribute against zero and appends
/// `"0"`, else `"1"` (`:111-114`). It does not print the index — a corelet 2 would be named `1`. That
/// is a two-corelet assumption baked into the naming, and [`Corelet`] is bounded by the arch's own
/// count so a third corelet would be a build error here rather than a silently mislabelled unit.
#[must_use]
pub fn name_with_corelet(unit: DfirUnit, corelet: Corelet) -> String {
    format!("{}{}", unit.spelling(), corelet.get())
}

/// WHICH UNITS ONE `set_send_dst` NAMES.
///
/// ⛔ THE REFERENCE COLLECTS THEM FROM TWO SHAPES: a plain `dataflow.get_unit`, giving one component;
/// or a `uniform.QueryMapOp`, whose `getAllQueriedValues` yields several (`Helper.cpp:2749-2760`). The
/// second is how one statement names a uniformized set of units across cores or folds.
///
/// ⛔ NON-EMPTY BY CONSTRUCTION, for the reason [`crate::islands::dataflow_ir::Units`] is: a
/// destination list with nothing in it is a statement that says nothing, and `getUnits()[0]` on the
/// far side is an unguarded index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Destinations {
    head: Val,
    rest: Vec<Val>,
}

impl Destinations {
    /// One destination — the `get_unit` shape.
    #[must_use]
    pub fn one(unit: Val) -> Destinations {
        Destinations {
            head: unit,
            rest: Vec::new(),
        }
    }

    /// Several — the queried-map shape, whose first value is what makes the statement exist.
    #[must_use]
    pub fn several(head: Val, rest: Vec<Val>) -> Destinations {
        Destinations { head, rest }
    }

    /// Every destination, head first.
    #[must_use]
    pub fn vals(&self) -> Vec<Val> {
        core::iter::once(self.head)
            .chain(self.rest.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 🎯 ONLY THE LX **LOAD** UNIT SETS A SEND DESTINATION.
    #[test]
    fn only_the_lx_load_unit_sets_send_destinations() {
        assert!(sets_send_destination(DfirUnit::Lxlu));
        // ⛔ THE OTHER LX HALF DOES NOT — the easy mistake.
        assert!(!sets_send_destination(DfirUnit::Lxsu));
        for unit in [DfirUnit::L3lu, DfirUnit::L0lu, DfirUnit::Pe, DfirUnit::Sfp] {
            assert!(!sets_send_destination(unit));
        }
    }

    /// 🎯 `isTargetL3` SAYS YES WHEN IT DOES NOT KNOW.
    ///
    /// ⛔ THE REFERENCE'S DEFAULT, AND THE REASON THE FUNCTION EXISTS. A reimplementation defaulting to
    /// `false` inverts every silent case.
    #[test]
    fn an_unknown_target_counts_as_l3() {
        assert!(is_target_l3(None), "the reference returns true on absence");
        assert!(is_target_l3(Some(DfirUnit::L3lu)));
        assert!(is_target_l3(Some(DfirUnit::L3su)));
        assert!(!is_target_l3(Some(DfirUnit::Lxlu)));
    }

    /// 🎯 THE L0 HALVES ARE DISTINGUISHED, not merged.
    #[test]
    fn the_l0_halves_are_separate() {
        assert!(is_l0_load(DfirUnit::L0lu));
        assert!(!is_l0_load(DfirUnit::L0su));
        assert!(is_l0_store(DfirUnit::L0su));
        assert!(!is_l0_store(DfirUnit::L0lu));
    }

    /// 🎯 A DESTINATION LIST CANNOT BE EMPTY.
    #[test]
    fn destinations_are_non_empty() {
        assert_eq!(Destinations::one(Val(3)).vals(), vec![Val(3)]);
        assert_eq!(
            Destinations::several(Val(1), vec![Val(2), Val(3)]).vals(),
            vec![Val(1), Val(2), Val(3)]
        );
    }
}
