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
use crate::units::{Core, Corelet, DfirUnit, Residency};
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::regfile::Component;

/// WHICH UNIT — the typed form of `getUnitName`'s string (`Utils.cpp:264-270`).
///
/// ⛔ THE PT ROW IS PART OF THE IDENTITY, not a detail the component collapses.
/// `sys_arch_spec::regfile::Component` answers `Pt` for all eight rows, so keying by it would merge
/// eight units' register state into one; the reference's string carries `ptrow3` and so does this.
///
/// ⛔⛔ THIS IS THE BRIDGE'S ONLY UNIT KEY. A second one keyed by
/// [`crate::units::Residency`] used to live in `lower::control`, which meant the unit a uniform
/// block routed to a region and the unit an instruction's operand map answered for were different
/// types, and the two halves of the port could not be joined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnitKey {
    /// The `type=` the unit was bound with.
    pub unit: DfirUnit,
    /// Which core — `getCoreId` (`DccExtContext.cpp:78-87`).
    pub core: Core,
    /// Which corelet of it — ⛔ `None` IS A MISSING `corelet` ATTRIBUTE, which is what
    /// `getCoreletId`'s -1 means (`DccExtContext.cpp:115-127`), and it is a MEMORY unit's case,
    /// NOT an L3's: a non-parallel component is materialised with `corelet = 0`
    /// (`UnitMaterializer.cpp:72-73`), so `getUnitName` spells the L3 halves `…corelet0`.
    pub corelet: Option<Corelet>,
}

impl UnitKey {
    /// THE THREE FACTS `getUnitName` STRINGIFIES, from the `dataflow.get_unit` it reads them off —
    /// `type`, `getCoreId`, `getCoreletId`.
    ///
    /// ⛔ `None` WHERE `getCoreId` ANSWERS -1, which is a unit with no `core` attribute: the
    /// reference keys it under the literal name `…core-1corelet-1` and no compute unit is one.
    #[must_use]
    pub fn of(unit: DfirUnit, residency: Residency) -> Option<UnitKey> {
        let (core, corelet) = match residency {
            Residency::Global => return None,
            Residency::Scratchpad { core } => (core, None),
            Residency::CoreWide { core } => (core, Some(Corelet::checked(0)?)),
            Residency::Corelet { core, corelet } => (core, Some(corelet)),
        };
        Some(UnitKey {
            unit,
            core,
            corelet,
        })
    }

    /// `getSenComponentForProgramStateInfo(get_unit, corelet)` (`DccExtContext.cpp:210-238`), reduced
    /// to what [`crate::islands::progir::Program`] keys by.
    ///
    /// ⛔ `None` IS THAT SWITCH'S `DT_ERROR` ARM — *"Unrecognized unit name in lowering to progir"*
    /// for anything but the nine compute units, so a memory, a state, a ring or a link has no key.
    /// ⛔⛔ AND THE CORELET AND THE PT ROW ARE DROPPED HERE, because the island keys by the
    /// nine-variant generic component while the reference keys `PE_CL1` / `PT_ROW3_CL0`: two corelets'
    /// register sets MERGE, which over-initialises rather than mis-initialises. The qualified
    /// vocabulary is already vendored as [`sys_arch_spec::arch_enums::SenComponent`] — re-keying the
    /// island onto it is the fix, and it is island work, not this seam's.
    #[must_use]
    pub const fn program_key(&self) -> Option<(Core, Component)> {
        let comp = match self.unit {
            DfirUnit::Sfp => Component::Sfp,
            DfirUnit::Pe => Component::Pe,
            DfirUnit::PtRow(_) => Component::Pt,
            DfirUnit::L0lu => Component::L0lu,
            DfirUnit::L0su => Component::L0su,
            DfirUnit::Lxlu => Component::Lxlu,
            DfirUnit::Lxsu => Component::Lxsu,
            DfirUnit::L3lu => Component::L3lu,
            DfirUnit::L3su => Component::L3su,
            DfirUnit::Lx
            | DfirUnit::Hbm
            | DfirUnit::L0
            | DfirUnit::Constant
            | DfirUnit::SfpState
            | DfirUnit::PeState
            | DfirUnit::SfpRing
            | DfirUnit::LxVirtualIbr
            | DfirUnit::CrossPtnLink => return None,
        };
        Some((self.core, comp))
    }
}

/// WHAT EVERY UNIT'S REGISTERS START AS — `ProgIrGraphMap`
/// (`SentientToProgIR.hpp:37`, `unordered_map<std::string, ProgIrRegGraph>`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegGraphs {
    /// One entry per unit that has any register initialiser, in insertion order.
    pub per_unit: Vec<(UnitKey, UnitRegState)>,
}

impl RegGraphs {
    /// `reg_graph[unit].addRegInit(...)` — the two operations the reference spells as one subscript.
    ///
    /// ⛔⛔ `addRegInit` ASSIGNS, IT DOES NOT APPEND: `regInfo[regType][regNum] = regContent`
    /// (`progir.h:491-496`), so initialising the same register twice keeps the LAST write and the
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
/// (`SentientToProgIR.hpp:53`, `std::map<RegType, std::set<unsigned>>`).
pub type UtilizedRegisters = BTreeMap<RegType, BTreeSet<RegIndex>>;

/// THE SAME, FOR EVERY UNIT — `regs_to_init_` (`SentientToProgIR.hpp:122`), keyed by the
/// reference's `CoreAndComponent` (`:63`).
///
/// ⭐ THE REFERENCE'S KEY IS `pair<int, SenComponents>` WITH THE CORELET ENCODED INTO THE COMPONENT
/// (`PE_CL1`, `PT_ROW3_CL0`); [`UnitKey`] carries the same three facts as three fields.
pub type RegsToInit = BTreeMap<UnitKey, UtilizedRegisters>;

/// WHICH OPERATION A BRANCH LABEL IS ATTACHED TO — the `Operation*` key of the reference's
/// `std::map<Operation*, std::string> labels`, threaded through every jump lowering.
///
/// ⛔ A CALLER'S NUMBER, NOT A POINTER: the lowering only ever asks whether two ops are the same one,
/// and a pointer is not a value this crate can compare. Same discipline as [`UniformLabel`].
///
/// [`UniformLabel`]: crate::bridges::sentient_to_progir::uniform::instr::UniformLabel
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpSite(pub u32);

/// WHICH OPERATIONS CARRY A LABEL SOMETHING JUMPS TO — `std::map<Operation*, std::string>&`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Labels {
    /// One entry per labelled op, in insertion order.
    pub per_op: Vec<(OpSite, String)>,
}

impl Labels {
    /// The label this op carries, if it has one.
    #[must_use]
    pub fn get(&self, at: OpSite) -> Option<&str> {
        self.per_op
            .iter()
            .find(|(site, _)| *site == at)
            .map(|(_, label)| label.as_str())
    }

    /// `if (labels.count(at) == 0) labels[at] = label;` — ⛔ THE FIRST CLAIM WINS, and the answer is
    /// the label the op ends up carrying, which is NOT always the one offered.
    pub fn claim(&mut self, at: OpSite, label: String) -> &str {
        match self.per_op.iter().position(|(site, _)| *site == at) {
            Some(held) => &self.per_op[held].1,
            None => {
                self.per_op.push((at, label));
                &self.per_op.last().expect("just pushed").1
            }
        }
    }
}

/// THE NUMBER THE NEXT MINTED LABEL TAKES — `int& labels_counter`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct LabelCounter(pub u32);

impl LabelCounter {
    /// `labels_counter++` — the number to use, the counter left pointing past it.
    pub fn bump(&mut self) -> u32 {
        let now = self.0;
        self.0 += 1;
        now
    }
}

/// HOW MANY COPY INSTRUCTIONS ONE LOWERING EMITTED — `CodeQualityStats::num_copy_ops_`.
///
/// ⭐ A COUNT COMING BACK, NOT AN `optional<CodeQualityStats>&` GOING IN: the reference threads the
/// absence of statistics through every lowering, where a caller collecting none can just drop this.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CopyOps(pub u32);
