// SPDX-License-Identifier: Apache-2.0
//! WHAT THE PASS ACCUMULATES WHILE IT LOWERS — the per-unit register graph and the register sets a
//! unit must initialise.
//!
//! Authority: `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.hpp` and
//! `sys-arch-spec/progir/progir.h`, tree `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! ⛔⛔ THE REFERENCE KEYS EVERY ONE OF THESE MAPS BY A FORMATTED STRING. `getUnitName` builds
//! `"<type>core<N>corelet<M>"` (`dcc/src/Utils/Utils.cpp:264-270`) and `ProgIrGraphMap` is an
//! `unordered_map<std::string, ProgIrRegGraph>` over it. A key built by concatenation is a key two
//! spellings of the same unit can disagree on, so [`UnitKey`] carries the three fields instead.

use crate::islands::progir::UnitRegState;
use crate::islands::progir::ty::RegType;
use crate::islands::sentient::dialects::sentient::RegIndex;
use crate::units::{Core, Corelet, DfirUnit};
use std::collections::{BTreeMap, BTreeSet};

/// WHICH UNIT — the typed form of `getUnitName`'s string (`Utils.cpp:264-270`).
///
/// ⛔ THE PT ROW IS PART OF THE IDENTITY, not a detail the component collapses.
/// `sys_arch_spec::regfile::Component` answers `Pt` for all eight rows, so keying by it would merge
/// eight units' register state into one; the reference's string carries `ptrow3` and so does this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnitKey {
    /// The `type=` the unit was bound with.
    pub unit: DfirUnit,
    /// Which core — `getCoreId` (`DccExtContext.cpp:78-87`).
    pub core: Core,
    /// Which corelet of it — ⛔ `None` FOR AN L3 UNIT, which is shared across a core's corelets and
    /// whose `getCoreletId` answers -1 (`DccExtContext.cpp:116-124`).
    pub corelet: Option<Corelet>,
}

/// WHAT EVERY UNIT'S REGISTERS START AS — `ProgIrGraphMap`
/// (`SentientToProgIR.hpp`, `unordered_map<std::string, ProgIrRegGraph>`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegGraphs {
    /// One entry per unit that has any register initialiser, in insertion order.
    pub per_unit: Vec<(UnitKey, UnitRegState)>,
}

impl RegGraphs {
    /// `reg_graph[unit].addRegInit(...)` — the two operations the reference spells as one subscript.
    ///
    /// ⛔⛔ `addRegInit` ASSIGNS, IT DOES NOT APPEND: `regInfo[regType][regNum] = regContent`
    /// (`progir.h:486-501`), so initialising the same register twice keeps the LAST write and the
    /// first is lost without trace. A `push` here would keep both and print two initialisers.
    ///
    /// ⭐ AND THE SUBSCRIPT DEFAULT-CONSTRUCTS: a unit with no graph yet gets an empty one rather
    /// than being skipped, which is why an unnamed unit still reaches the output.
    pub fn add_reg_init(&mut self, unit: UnitKey, init: crate::islands::progir::RegInit) {
        let state = match self.per_unit.iter_mut().find(|(at, _)| *at == unit) {
            Some(entry) => &mut entry.1,
            None => {
                self.per_unit.push((unit, UnitRegState::new()));
                &mut self.per_unit.last_mut().expect("just pushed").1
            }
        };
        match state
            .iter_mut()
            .find(|held| held.file == init.file && held.index == init.index)
        {
            Some(held) => *held = init,
            None => state.push(init),
        }
    }

    /// One unit's register state, or `None` where the reference's subscript would create an empty one.
    #[must_use]
    pub fn get(&self, unit: UnitKey) -> Option<&UnitRegState> {
        self.per_unit
            .iter()
            .find(|(at, _)| *at == unit)
            .map(|(_, state)| state)
    }
}

/// WHICH REGISTERS OF WHICH FILES ONE UNIT MUST INITIALISE — `UtilizedRegisters`
/// (`progir.h`, `std::map<RegType, std::set<unsigned>>`).
pub type UtilizedRegisters = BTreeMap<RegType, BTreeSet<RegIndex>>;

/// THE SAME, FOR EVERY UNIT — `regs_to_init_`, keyed by the reference's `CoreAndComponent`.
///
/// ⭐ THE REFERENCE'S KEY IS `pair<int, SenComponents>` WITH THE CORELET ENCODED INTO THE COMPONENT
/// (`PE_CL1`, `PT_ROW3_CL0`); [`UnitKey`] carries the same three facts as three fields.
pub type RegsToInit = BTreeMap<UnitKey, UtilizedRegisters>;
