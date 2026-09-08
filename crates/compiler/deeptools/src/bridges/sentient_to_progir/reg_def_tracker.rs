//! WHICH REGISTERS ARE DEFINED, AND WHETHER ANYTHING READS ONE THAT IS NOT.
//!
//! ⛔⛔ `checkRegDefs` IS A CHECK, AND THIS CRATE NEVER RUNTIME REFUSES. Port it as a function
//! that RETURNS THE OFFENDERS, the way `progir::Program::overflowing` does — never as an
//! `assert!`, a `Result` or a `signalPassFailure`.
//! ⭐ `~UniformRegionContext` IS RAII: real work runs on scope exit (it merges the region's reg
//! defs). It is unit `e070_dtor_UniformRegionContext`, not a destructor to drop.
//!
//! 5 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e012_setRegDef` | 0 | 7 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:53` |
//! | `e013_addRegDefsForUnit` | 0 | 15 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:61` |
//! | `e014_checkRegDefs` | 0 | 63 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:106` |
//! | `e069_recordOpRegDefs` | 1 | 30 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:22` |
//! | `e070_dtor_UniformRegionContext` | 1 | 28 | `dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:77` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::Arch;
use crate::islands::progir::ty::RegType;
use crate::islands::progir::{Program, RegBits, RegDefs};
use crate::islands::sentient::dialects::sentient::{Reg, RegIndex, RegType as Locale};
use crate::model::Model;
use crate::units::Core;
use crate::workload::Workload;
use sys_arch_spec::regfile::Component;

/// WHETHER THE REFERENCE COMPARES REG DEFS AT ALL — `RegDefTracker::regDefChecking()`, a
/// `static constexpr bool` returning **false** (`RegDefTracker.hpp:25-32`).
///
/// ⛔ IT IS THE CALLER'S GATE, NOT AN EARLY RETURN INSIDE [`check_reg_defs`]. The reference opens that
/// function with `if (!regDefChecking()) return;` over a compile-time false; keeping the test inside
/// the ported body would make the whole comparison dead code and untestable, so the gate is hoisted
/// to the one place that decides whether to run it.
pub const REG_DEF_CHECKING: bool = false;

/// WHICH REGISTERS OF WHICH FILES ONE UNIT TOUCHES — `RegDefTracker::RegDefContext::RegSet`,
/// `std::array<std::bitset<kMaxCompRegs>, RegType::MAX_VALUE + 1>` (`RegDefTracker.hpp:66-67`).
///
/// ⛔⛔ FIFTEEN FILES WIDE, NOT THE REFERENCE'S FOURTEEN, so a scale-file def is representable rather
/// than one past the end of the array — see
/// [`sys_arch_spec::arch_enums::MAX_VALUE_IS_NOT_THE_MAXIMUM`]. Which files the reference's own loops
/// REACH is reproduced where it loops, by `RegType::is_reached_by_a_max_value_loop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegSet {
    files: [RegBits; RegType::ALL.len()],
}

impl Default for RegSet {
    fn default() -> Self {
        Self::empty()
    }
}

impl RegSet {
    /// Nothing defined — what a `RegDefContext` starts as.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            files: [RegBits::empty(); RegType::ALL.len()],
        }
    }

    /// What is set in one file.
    #[must_use]
    pub const fn get(&self, file: RegType) -> RegBits {
        self.files[file as usize]
    }

    /// `regs()[file].set(index)`.
    pub fn set(&mut self, file: RegType, index: RegIndex) {
        self.files[file as usize] = self.files[file as usize].with(index);
    }
}

/// WHICH FILE A SENTIENT REGISTER LOCALE NAMES — `dcc::getRegType` (`DccExtContext.cpp:240-265`).
///
/// ⛔ `None` FOR A LOCALE THAT IS NOT A FILE. The reference `DT_ERROR`s on `imm`, `jcr`, `lccr`, the
/// two xrf pointers, `unknown` and `unrelated`; [`set_reg_def`] filters five of those before calling,
/// so only `unknown`/`unrelated` differ — a def by a value that has no register yet.
#[must_use]
pub fn reg_file(locale: Locale) -> Option<RegType> {
    match locale {
        Locale::Lrf => Some(RegType::Lrf),
        Locale::Lar => Some(RegType::Lar),
        Locale::Lbr => Some(RegType::Lbr),
        Locale::Ear => Some(RegType::Ear),
        Locale::Ebr => Some(RegType::Ebr),
        Locale::Gtr => Some(RegType::Gtr),
        Locale::Mvr => Some(RegType::Mvr),
        Locale::Imm
        | Locale::Jcr
        | Locale::Lccr
        | Locale::XrfRdPtr
        | Locale::XrfWrPtr
        | Locale::Unknown
        | Locale::Unrelated => None,
    }
}

/// Replaces: e012_setRegDef
///
/// Record that the op being visited defines one register of the tracked unit.
/// ⛔ AN UNASSIGNED REGISTER IS `None`, which is the reference's `regNum < 0` — the `.td`'s `-1`.
/// ⛔ AND A LOCALE THAT NAMES NO FILE IS DROPPED, NOT RECORDED: `imm`, `jcr`, `lccr` and the two xrf
/// pointers are the reference's own exclusion list, and [`reg_file`] answers `None` for the rest.
pub fn set_reg_def(regs: &mut RegSet, locale: Locale, index: Option<RegIndex>) {
    let (Some(index), Some(file)) = (index, reg_file(locale)) else {
        return;
    };
    regs.set(file, index);
}

/// Replaces: e013_addRegDefsForUnit
///
/// Merge one unit's tracked defs into its core's program state, keyed by `(component, file)`.
/// ⭐ AN EMPTY MAP IS PUT THERE FIRST BECAUSE `Some(empty)` MEANS "TRACKED" — see
/// [`Program::reg_defs`].
/// ⛔ ONLY THE FILES A `MAX_VALUE`-BOUNDED LOOP REACHES ARE KEYED, because that is the width of the
/// reference's array: the scale file is never merged, however set it is.
pub fn add_reg_defs_for_unit<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    comp: Component,
    regs: &RegSet,
) {
    let defs = program.reg_defs.get_or_insert_with(RegDefs::new);
    for file in RegType::ALL {
        if !file.is_reached_by_a_max_value_loop() {
            continue;
        }
        let bits = regs.get(file);
        // ⭐ `|=` THROUGH `operator[]`, so a file with nothing set still gets its key.
        match defs.iter_mut().find(|(key, _)| *key == (comp, file)) {
            Some((_, defined)) => *defined = defined.union(bits),
            None => defs.push(((comp, file), bits)),
        }
    }
}

/// ONE REGISTER A UNIT IS SAID TO DEFINE THAT NOTHING REFERENCES — what `checkRegDefs` prints as a
/// *"Reg ref discrepancy"* (`RegDefTracker.cpp:148`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegRefDiscrepancy {
    /// Which unit.
    pub component: Component,
    /// Which file.
    pub file: RegType,
    /// Which register of it.
    pub index: RegIndex,
}

/// Replaces: e014_checkRegDefs
///
/// Every register the defs claim for `comp` that neither its reg inits nor its instructions reach —
/// ⛔ THE OFFENDERS, never a refusal, and never a bool.
/// ⛔ THE INSTRUCTION WALK IS `RegVisitor::visitInstrRegRefs` (`sys-arch-spec/progir/regvisitor.cpp:15`),
/// an ISA table outside this campaign's 130, so what it saw arrives as `referenced`.
/// ⛔ AND ONLY SIX COMPONENTS ARE COMPARED AT ALL: the PT and both L0 halves are outside the
/// reference's list, so a claim about one of them is not an offender here.
#[must_use]
pub fn check_reg_defs<A: Arch, M: Model, W: Workload>(
    program: &Program<A, M, W>,
    comp: Component,
    referenced: &RegSet,
) -> Vec<RegRefDiscrepancy> {
    // ⛔ NOT SET IS NOT "NO OFFENDERS": the reference `DT_CHECK`s that the data is there rather than
    // comparing against nothing, which would report every def as a discrepancy.
    let Some(defs) = program.reg_defs.as_ref() else {
        return Vec::new();
    };
    if !matches!(
        comp,
        Component::L3lu
            | Component::L3su
            | Component::Lxlu
            | Component::Lxsu
            | Component::Sfp
            | Component::Pe
    ) {
        return Vec::new();
    }
    // The evaluated set: the register inits, then the instruction references.
    let mut evaluated = *referenced;
    if let Some((_, state)) = program.reg_state.iter().find(|(unit, _)| *unit == comp) {
        for init in state {
            evaluated.set(init.file, init.index);
        }
    }
    let mut discrepancies = Vec::new();
    for ((unit, file), defined) in defs {
        if *unit != comp {
            continue;
        }
        for index in defined.iter() {
            if !evaluated.get(*file).holds(index) {
                // ⭐ THE REFERENCE ALSO DUMPS THE INITS AND THE WHOLE PROGRAM to `std::cerr` here;
                // that is the caller's to print, and it needs this list to know there is anything
                // to print.
                discrepancies.push(RegRefDiscrepancy {
                    component: comp,
                    file: *file,
                    index,
                });
            }
        }
    }
    discrepancies
}

/// WHICH REGISTERS ONE OP SAYS IT DEFINES — the singular `regLocale`/`regIndex` pair, or the
/// `regLocales`/`regIndices` arrays (`:30,41`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpRegDefs {
    /// `regIndex` is present — ⛔ AND IT WINS AND RETURNS: an op carrying both attribute shapes has
    /// its arrays ignored (`:37`).
    One(Reg),
    /// `regIndices`, zipped with `regLocales`.
    Many(Vec<Reg>),
    /// Neither attribute — nothing is recorded, which the reference reaches by falling off the end.
    None,
}

/// Replaces: e069_recordOpRegDefs
///
/// Record every register the op being visited defines.
/// ⛔ AN INDEX WITHOUT A LOCALE IS UNREPRESENTABLE: [`Reg`] carries both, so `"op expected to have
/// regLocale attr"` and its array twin (`:32,43`) have nothing to refuse — and see [`set_reg_def`]
/// for the locale that names no file.
/// ⛔ `enabled()` AND `reg_def_ctx_` ARE THE CALLER'S GATE, like [`REG_DEF_CHECKING`].
pub fn record_op_reg_defs(regs: &mut RegSet, defs: &OpRegDefs) {
    match defs {
        OpRegDefs::One(reg) => set_reg_def(regs, reg.locale, reg.index),
        OpRegDefs::Many(pairs) => {
            for reg in pairs {
                set_reg_def(regs, reg.locale, reg.index);
            }
        }
        OpRegDefs::None => {}
    }
}

/// ONE ENTRY OF A UNIFORM REGION'S UNIT LIST — a `get_unit`, or a `create_group` of them (`:81-88`).
///
/// ⭐ THE CORELET IS ALREADY FOLDED INTO THE COMPONENT: `getSenComponentForProgramStateInfo`
/// (`Utils/DccExtContext.cpp:210-238`) keys the program state by `(type, corelet)`, so a unit arrives
/// here as the pair `addRegDefsForUnit` writes under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionUnit {
    /// One `dataflow.get_unit`.
    One(Core, Component),
    /// A `dataflow.create_group`, whose ids are each a `get_unit`.
    ///
    /// ⛔ A MEMBER THAT IS NOT ONE IS A NULL THERE (`:86`) — `subunit.getDefiningOp<GetUnitOp>()` is
    /// handed straight to `addRegDefsForUnit`, which this cannot express.
    Group(Vec<(Core, Component)>),
}

/// Replaces: e070_dtor_UniformRegionContext
///
/// Leaving a uniform region merges what it defined into every one of its units' programs.
/// ⛔ THE TWO OP KINDS RUN THE SAME WALK (`:79,91`), so the op is not a parameter and the trailing
/// `DT_ERROR("Unexpected operation for uniform region")` has no spelling.
/// ⛔ AND `enable_ctx_` IS THE CALLER'S GATE (`:78`), the third of them in this file.
pub fn close_uniform_region<A: Arch, M: Model, W: Workload>(
    region_units: &[RegionUnit],
    regs: &RegSet,
    progstateinfo: &mut Vec<(Core, Program<A, M, W>)>,
) {
    for entry in region_units {
        match entry {
            RegionUnit::One(core, comp) => merge_unit_reg_defs(progstateinfo, *core, *comp, regs),
            RegionUnit::Group(members) => {
                for (core, comp) in members {
                    merge_unit_reg_defs(progstateinfo, *core, *comp, regs);
                }
            }
        }
    }
}

/// `addRegDefsForUnit(get_unit)` — ⭐ `progStateInfo()[core]` DEFAULT-CONSTRUCTS (`:64`), so a core
/// nothing has emitted for still gets its defs.
pub fn merge_unit_reg_defs<A: Arch, M: Model, W: Workload>(
    progstateinfo: &mut Vec<(Core, Program<A, M, W>)>,
    core: Core,
    comp: Component,
    regs: &RegSet,
) {
    if !progstateinfo.iter().any(|(id, _)| *id == core) {
        progstateinfo.push((core, Program::default()));
    }
    for (_, program) in progstateinfo.iter_mut().filter(|(id, _)| *id == core) {
        add_reg_defs_for_unit(program, comp, regs);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::islands::progir::{RegInit, ty::Operand, ty::OperandValue};

    struct M;
    impl Model for M {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    struct W;
    impl Workload for W {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 128;
    }

    #[test]
    fn the_five_excluded_locales_and_the_unassigned_index_define_nothing() {
        let mut regs = RegSet::empty();
        set_reg_def(&mut regs, Locale::Lar, Some(RegIndex::at::<3>()));
        assert!(regs.get(RegType::Lar).holds(RegIndex::at::<3>()));
        // The exclusion list, and the `-1`.
        for locale in [
            Locale::Imm,
            Locale::Jcr,
            Locale::Lccr,
            Locale::XrfRdPtr,
            Locale::XrfWrPtr,
        ] {
            set_reg_def(&mut regs, locale, Some(RegIndex::at::<7>()));
        }
        set_reg_def(&mut regs, Locale::Lrf, None);
        assert_eq!(
            regs.get(RegType::Lar),
            RegBits::empty().with(RegIndex::at::<3>()),
            "only the one lar def was recorded"
        );
        assert!(regs.get(RegType::Lrf).is_empty());
        assert!(regs.get(RegType::Jcr).is_empty());
    }

    #[test]
    fn merging_keys_every_file_a_max_value_loop_reaches_and_not_the_scale_file() {
        let mut program = Program::<Target, M, W>::default();
        let mut regs = RegSet::empty();
        regs.set(RegType::Ear, RegIndex::at::<1>());
        regs.set(RegType::Scale, RegIndex::at::<2>());
        add_reg_defs_for_unit(&mut program, Component::L3lu, &regs);
        let defs = program.reg_defs.as_ref().expect("emplaced");
        assert_eq!(defs.len(), RegType::MAX_VALUE as usize + 1);
        assert!(!defs.iter().any(|((_, file), _)| *file == RegType::Scale));
        assert_eq!(
            defs.iter()
                .find(|(key, _)| *key == (Component::L3lu, RegType::Ear))
                .map(|(_, bits)| *bits),
            Some(RegBits::empty().with(RegIndex::at::<1>()))
        );
        // A second merge unions rather than replaces.
        let mut more = RegSet::empty();
        more.set(RegType::Ear, RegIndex::at::<5>());
        add_reg_defs_for_unit(&mut program, Component::L3lu, &more);
        assert_eq!(
            program.reg_defs.as_ref().and_then(|defs| defs
                .iter()
                .find(|(key, _)| *key == (Component::L3lu, RegType::Ear))
                .map(|(_, bits)| *bits)),
            Some(
                RegBits::empty()
                    .with(RegIndex::at::<1>())
                    .with(RegIndex::at::<5>())
            )
        );
    }

    #[test]
    fn a_def_no_init_and_no_instruction_reaches_is_an_offender() {
        let mut program = Program::<Target, M, W>::default();
        let mut regs = RegSet::empty();
        regs.set(RegType::Ear, RegIndex::at::<1>());
        regs.set(RegType::Ear, RegIndex::at::<2>());
        regs.set(RegType::Lar, RegIndex::at::<4>());
        add_reg_defs_for_unit(&mut program, Component::L3lu, &regs);
        // `ear1` is initialised, `lar4` is referenced by an instruction, `ear2` is neither.
        program.reg_state.push((
            Component::L3lu,
            vec![RegInit {
                file: RegType::Ear,
                index: RegIndex::at::<1>(),
                value: Operand::every(OperandValue::Int(0)),
            }],
        ));
        let mut referenced = RegSet::empty();
        referenced.set(RegType::Lar, RegIndex::at::<4>());
        assert_eq!(
            check_reg_defs(&program, Component::L3lu, &referenced),
            vec![RegRefDiscrepancy {
                component: Component::L3lu,
                file: RegType::Ear,
                index: RegIndex::at::<2>(),
            }]
        );
        // ⛔ The PT is outside the reference's list, and unset defs are not compared at all.
        assert!(check_reg_defs(&program, Component::Pt, &referenced).is_empty());
        assert!(
            check_reg_defs(
                &Program::<Target, M, W>::default(),
                Component::L3lu,
                &referenced
            )
            .is_empty()
        );
    }

    /// e069: the singular attribute wins over the arrays, and an op with neither records nothing.
    #[test]
    fn the_singular_reg_def_attribute_wins_over_the_arrays() {
        let mut regs = RegSet::empty();
        record_op_reg_defs(
            &mut regs,
            &OpRegDefs::One(Reg {
                locale: Locale::Lar,
                index: Some(RegIndex::at::<3>()),
            }),
        );
        assert_eq!(
            regs.get(RegType::Lar),
            RegBits::empty().with(RegIndex::at::<3>())
        );
        record_op_reg_defs(
            &mut regs,
            &OpRegDefs::Many(vec![
                Reg {
                    locale: Locale::Lrf,
                    index: Some(RegIndex::at::<1>()),
                },
                Reg {
                    locale: Locale::Ebr,
                    index: None,
                },
            ]),
        );
        assert_eq!(
            regs.get(RegType::Lrf),
            RegBits::empty().with(RegIndex::at::<1>())
        );
        assert!(regs.get(RegType::Ebr).is_empty(), "an unassigned index");
        let before = regs;
        record_op_reg_defs(&mut regs, &OpRegDefs::None);
        assert_eq!(regs, before);
    }

    /// e070: every unit of the region list — a bare unit and every member of a group — gets the
    /// region's defs, on a core the pass has emitted nothing for yet.
    #[test]
    fn leaving_a_region_merges_its_defs_into_every_units_program() {
        let mut regs = RegSet::empty();
        regs.set(RegType::Lar, RegIndex::at::<2>());
        let core = |id: u32| Core::checked(id).expect("every arch has 32 cores");
        let mut progstateinfo: Vec<(Core, Program<Target, M, W>)> = Vec::new();
        close_uniform_region(
            &[
                RegionUnit::One(core(0), Component::Lxlu),
                RegionUnit::Group(vec![(core(1), Component::Sfp), (core(1), Component::Pe)]),
            ],
            &regs,
            &mut progstateinfo,
        );
        assert_eq!(progstateinfo.len(), 2, "two cores, default-constructed");
        for (core, comp) in [
            (0, Component::Lxlu),
            (1, Component::Sfp),
            (1, Component::Pe),
        ] {
            let defs = progstateinfo
                .iter()
                .find(|(id, _)| id.get() == core)
                .and_then(|(_, program)| program.reg_defs.as_ref())
                .expect("the core was emplaced");
            assert_eq!(
                defs.iter()
                    .find(|(key, _)| *key == (comp, RegType::Lar))
                    .map(|(_, bits)| *bits),
                Some(RegBits::empty().with(RegIndex::at::<2>()))
            );
        }
    }
}
