//! THE UNITS A PROGRAM DECLARES, AND WHAT EACH ONE IS NEXT TO.
//!
//! A transcription of `DSC2ToDataflowIR::buildNeighborUnits` and `createGetUnitOp`
//! (`DSC2ToDataflowIRUtils.hpp:64,161`). Every unit a program's body names has to be bound by a
//! `dataflow.get_unit` first, and which units those are is a property of the one being programmed —
//! the PT reads from the west and passes its partial sum south, so its program names both.
//!
//! ⭐⭐ THE ARCH IS WHAT MAKES THE NEIGHBOURS DIFFERENT, AND THE BOUND IS THE CONST GENERIC. Row 3's
//! south is row 4 on RCUDD1A and the PE on SEN1P5 — not a special case, but the same rule
//! (`the LAST row's south is the PE`) read at two different row counts
//! (`DSC2ToDataflowIRUtils.hpp:213-222`). [`PtRow`] carries that count, so the rule is written once.

use crate::arch::{Arch, Bounded, IsaGen, Target};
use crate::generated::Unit;

/// ⭐⭐ THIS BUILD'S PT ROW TYPE — and the reason the arch is a cargo feature rather than a value.
///
/// `Target` is a concrete type once a feature is selected, so `Target::PT_ROWS` is a literal and may
/// stand as a const-generic argument. That is what turns "the last row's south is the PE" into
/// arithmetic the compiler performs: on `arch-rcudd1a` this is `PtRow<8>` and on `arch-sen1p5` it is
/// `PtRow<4>`, and the two are different types.
///
/// ⛔ AN ASSOCIATED CONST OF A GENERIC `A: Arch` CANNOT DO THIS. `PtRow<{ A::PT_ROWS }>` needs
/// `generic_const_exprs`; the first version of this file wrote `PtRow::<{ 8 }>` to get around it,
/// which is the DD2 row count hard-coded into the SEN1P5 build.
pub type Row = PtRow<{ Target::PT_ROWS }>;

/// This build's core index type.
pub type Core = CoreId<{ Target::CORES }>;

/// This build's corelet index type.
pub type Corelet = CoreletId<{ Target::CORELETS_PER_CORE }>;

/// WHICH CORE — an index that cannot exceed the arch's core count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoreId<const CORES: u32>(Bounded<CORES>);

impl<const CORES: u32> CoreId<CORES> {
    /// A core, or `None` if this arch has no such core.
    #[must_use]
    pub const fn checked(index: u32) -> Option<CoreId<CORES>> {
        match Bounded::checked(index) {
            Some(bounded) => Some(CoreId(bounded)),
            None => None,
        }
    }

    /// The index, for the one place it becomes an attribute.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// WHICH CORELET OF A CORE.
///
/// ⛔ AN L3 UNIT HAS NONE, and that is a fact rather than a missing value: `createGetUnitOp` writes
/// the `corelet` attribute only when the id is not -1 (`DSC2ToDataflowIRUtils.hpp:79-80`), because
/// the L3 is shared across a core's corelets. So a corelet is `Option<CoreletId>` at the use site,
/// and there is no -1 to leak into an attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoreletId<const CORELETS: u32>(Bounded<CORELETS>);

impl<const CORELETS: u32> CoreletId<CORELETS> {
    /// A corelet, or `None` if this arch has no such corelet.
    #[must_use]
    pub const fn checked(index: u32) -> Option<CoreletId<CORELETS>> {
        match Bounded::checked(index) {
            Some(bounded) => Some(CoreletId(bounded)),
            None => None,
        }
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// WHERE A UNIT LIVES — and therefore how many of it exist, what it is called, and which of the
/// `core`/`corelet` attributes it carries.
///
/// ⛔⛔ FOUR CASES, NOT A CORE PLUS AN OPTIONAL CORELET. This was `core: u32, corelet: Option<u32>`,
/// which can spell three of the four and **cannot spell the first at all** — a global memory has
/// neither attribute, and `core` was not optional. That is not a cosmetic gap: it is why every
/// operand ended up addressed into an LX that nothing fills. A unit vocabulary that cannot name the
/// HBM cannot put a weight there.
///
/// ⛔ AND THE FOURTH IS NOT THE THIRD. `C0-lx` carries `core` and NO `corelet`, while `C0-l3lu`
/// carries `core` AND `corelet = 0` (`/tmp/ktir_ref/export/debug/dfir.mlir:45-63`, and
/// `UnitMaterializer.cpp:62-80` against `:142-152`). The distinction is load-bearing downstream:
/// `ExtendUnitNameToCorelet` treats a MISSING `corelet` as a hard error and turns `corelet = 0` into
/// the name suffix `"0"` — so `type = "lxlu"` with `corelet = 1` is what becomes
/// `SentientLoadConsumer::lxlu1` (`DataflowToSentient.cpp:104-117`). Emitting `corelet = 0` for a
/// scratchpad would take a different branch in the consumer.
///
/// ⭐ THE CLASSIFICATION IS POSITIONAL IN THE MEMORY TREE, NOT A NAME TEST: global is a ROOT node,
/// per-core scratchpad is a node at DEPTH 1 (`MemoryTree.cpp:492-544`). For Spyre DD2 the root is
/// `%dram = memory #HBM` (16 GiB) and depth 1 is `%lx = memory #LX` (2 MiB)
/// (`sys-arch-spec/KTDFArchGraphDevice/spyre_dd2_basic.mlir:5-13,69-76`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// A ROOT of the memory tree: ONE for the whole device, named by its tag alone, carrying
    /// NEITHER attribute (`UnitMaterializer.cpp:136-141`, keyed `{space, -1}`).
    Global,
    /// A memory at depth 1: one per core, `C{core}-{tag}`, `core` and no `corelet`
    /// (`UnitMaterializer.cpp:142-152`).
    Scratchpad {
        /// Which core's scratchpad.
        core: Core,
    },
    /// A compute unit declared in the `group { kind = "core" }` and so shared across that core's
    /// corelets — `C{core}-{tag}` with `core` AND `corelet = 0`
    /// (`UnitMaterializer.cpp:62-80`). The L3 halves are the case.
    CoreWide {
        /// Which core.
        core: Core,
    },
    /// A compute unit declared in the `group { kind = "corelet" }`: one per corelet per core,
    /// `C{core}-{tag}-CL{corelet}` (`UnitMaterializer.cpp:82-115`).
    Corelet {
        /// Which core.
        core: Core,
        /// Which of that core's corelets.
        corelet: Corelet,
    },
}

impl Residency {
    /// WHICH CORE, or `None` for a unit the whole device shares.
    #[must_use]
    pub const fn core(self) -> Option<Core> {
        match self {
            Self::Global => None,
            Self::Scratchpad { core } | Self::CoreWide { core } | Self::Corelet { core, .. } => {
                Some(core)
            }
        }
    }
}

/// HOW MANY FOLDS a `dataflow.get_unit` produces results for.
///
/// ⛔ IT IS THE OP'S RESULT COUNT, NOT DECORATION. `get_unit` is `Variadic<Index>` with "each return
/// value corresponding to an instance of program time steps" (`Dataflow.td:56-58`), and
/// `createGetUnitOp` builds exactly `num_folds_` of them and writes the count as an attribute
/// (`DSC2ToDataflowIRUtils.hpp:70-77`). A unit bound with the wrong count has the wrong number of
/// SSA results, which is a parse failure rather than a silent one — but only if the count is carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NumFolds(pub u32);

impl NumFolds {
    /// The unfolded case — one result, one program time step.
    pub const ONE: NumFolds = NumFolds(1);
}

/// WHICH ROW OF THE PT — an index the arch's row count bounds.
///
/// ⭐⭐ THE CONST GENERIC IS LOAD-BEARING, NOT DECORATION. [`PtRow::south`] answers "the PE" for the
/// last row and "the next row" otherwise, and *which row is last* is `ROWS - 1`. Writing that with a
/// runtime row count would mean every caller could pass a different one; writing it per arch would
/// mean two copies of one rule. Here it is one rule read at two row counts, and a `PtRow<8>` cannot
/// be handed to something expecting a `PtRow<4>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtRow<const ROWS: u32>(Bounded<ROWS>);

/// WHAT SITS NORTH OR SOUTH OF A PT ROW.
///
/// ⛔ THREE VARIANTS BECAUSE THERE ARE THREE CASES. The chain runs SFP -> row 0 -> .. -> last row ->
/// PE (`DSC2ToDataflowIRUtils.hpp:165-167,213-222`, and `:399-404` from the PE's side), so the ends
/// are not rows and an `Option<PtRow>` would lose which end it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjacent<const ROWS: u32> {
    /// Another row of the same PT.
    Row(PtRow<ROWS>),
    /// The SFP, which is north of row 0.
    Sfp,
    /// The PE, which is south of the last row.
    Pe,
}

impl<const ROWS: u32> PtRow<ROWS> {
    /// A row, or `None` if this arch's PT has no such row.
    #[must_use]
    pub const fn checked(index: u32) -> Option<PtRow<ROWS>> {
        match Bounded::checked(index) {
            Some(bounded) => Some(PtRow(bounded)),
            None => None,
        }
    }

    /// The row index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// WHAT IS NORTH OF THIS ROW — the previous row, or the SFP at row 0.
    ///
    /// `component_to_handler_[SFP] = component_to_handler_[PTNORTH]` for `PTROW0`
    /// (`DSC2ToDataflowIRUtils.hpp:166-167`); every other row's north is the row above
    /// (`:185-186` and the arms below it).
    #[must_use]
    pub const fn north(self) -> Adjacent<ROWS> {
        match self.get() {
            0 => Adjacent::Sfp,
            row => match PtRow::checked(row - 1) {
                Some(above) => Adjacent::Row(above),
                // Unreachable: `row` is in bounds and non-zero, so `row - 1` is in bounds.
                None => Adjacent::Sfp,
            },
        }
    }

    /// WHAT IS SOUTH OF THIS ROW — the next row, or the PE at the last one.
    ///
    /// ⭐ ONE RULE, TWO ROW COUNTS. On RCUDD1A row 3's south is row 4 and row 7's is the PE; on
    /// SEN1P5 row 3 IS the last row and its south is the PE
    /// (`DSC2ToDataflowIRUtils.hpp:213-222`). The C++ writes that as an `if` on the arch inside
    /// row 3's arm; here it falls out of `ROWS`.
    #[must_use]
    pub const fn south(self) -> Adjacent<ROWS> {
        match PtRow::checked(self.get() + 1) {
            Some(below) => Adjacent::Row(below),
            None => Adjacent::Pe,
        }
    }
}

// ⛔ THERE WAS A `PtRow::unit() -> Option<Unit>` HERE AND IT WAS A TRAP. It mapped a row index back
// to a template [`Unit`] variant, which only exist for rows 0, 3 and 7 — the ones a `.ddl` names
// individually — so it answered `None` for row 1, a row every arch has. Anything that used it to
// decide "is this a real row" would have been wrong for five rows out of eight. A row's spelling is
// [`DfirUnit::spelling`]'s job, which is total over the arch's rows because DataflowIR names every
// one of them.

/// EVERY REGISTER FILE AND MEMORY A UNIT OWNS, which its program binds with `get_local_unit`.
///
/// ⛔ `l0scale` ONLY FROM SEN1P5. Each PT row's arm ends with
/// `if (coreArch >= SEN1P5_ISA) component_to_handler_[L0_SCALE] = createGetLocalUnitOp(..)`
/// (`DSC2ToDataflowIRUtils.hpp:180-183`), which matches the arch having no scale region before then
/// ([`Arch::L0_SCALE_CAPACITY`] is zero on RCUDD1A).
#[must_use]
pub fn local_units(of: Unit) -> Vec<crate::islands::dataflow_ir::op::LocalUnit> {
    use crate::islands::dataflow_ir::op::LocalUnit;
    use crate::islands::dataflow_ir::ty::GenericComp;

    let mut files = match of.generic() {
        GenericComp::Pt => vec![LocalUnit::PtLrf, LocalUnit::PtXrf],
        GenericComp::Pe => vec![LocalUnit::PeLrf],
        GenericComp::Sfp => vec![LocalUnit::SfpLrf],
        // A load or store unit owns no register file of its own; it moves other units' data.
        GenericComp::Lx | GenericComp::L0 | GenericComp::Constant => Vec::new(),
    };
    if matches!(of.generic(), GenericComp::Pt) && matches!(Target::GEN, IsaGen::Sen1p5) {
        files.push(LocalUnit::L0Scale);
    }
    files
}

/// A UNIT AS **DATAFLOWIR** NAMES IT — a strict superset of the templates' `unit=` vocabulary.
///
/// ⛔⛔ THE TWO SETS ARE NOT THE SAME, AND ASSUMING THEY WERE SILENTLY TRUNCATED THE NEIGHBOUR LIST.
/// [`Unit`] is censused from `unit=` across the `.ddl` files, which is 16 spellings. But
/// `buildNeighborUnits` binds units the templates never write that way: the LXLU's arm binds `LX`,
/// `L3LU` and `L3SU` (`DSC2ToDataflowIRUtils.hpp:369-386`), and a template refers to those through a
/// `data_connect="l3_lx_kernel"` or a `memory="lx"` instead of a `unit=`. The first version of
/// [`neighbours`] returned `Vec<Unit>` and so dropped all three — a program whose body could not send
/// to the L3 at all.
///
/// ⭐ SO THIS IS THE `SenComponents` SUBSET DATAFLOWIR BINDS, and [`Unit`] converts into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DfirUnit {
    /// `sfp`.
    Sfp,
    /// `pe`.
    Pe,
    /// One row of the PT.
    PtRow(Row),
    /// `lxlu` / `lxsu` — the LX load and store units.
    Lxlu,
    /// `lxsu`.
    Lxsu,
    /// `lx` — the LX memory itself, which a view is taken over.
    Lx,
    /// `hbm` — the device's global memory, which a view is taken over.
    ///
    /// ⛔⛔ THIS WAS MISSING AND ITS ABSENCE WAS THE WHOLE DEFECT. The subset below was censused
    /// from what `buildNeighborUnits` binds, and the HBM is never a *neighbour* — nothing sends to
    /// it directly; the L3 halves move data across the `%dram <-> %l3lu` / `%l3su <-> %dram`
    /// datapaths (`spyre_dd2_basic.mlir:82-85`). It belongs here for the same reason [`Self::Lx`]
    /// does: it is a memory a view is TAKEN OVER. With no variant for it the emitter could not put
    /// a weight anywhere but the LX, so every operand got an LX address nothing ever filled.
    ///
    /// `SenComponents::HBM` spells `"hbm"` (`sys-arch-spec/arch_enums.cpp:14`), which is what IBM's
    /// own DataflowIR carries: `{name = "hbm", type = "hbm"}` with no `core` and no `corelet`
    /// (`/tmp/ktir_ref/export/debug/dfir.mlir:61`).
    Hbm,
    /// `l0lu`.
    L0lu,
    /// `l0su`.
    L0su,
    /// `l0` — the L0 memory itself.
    L0,
    /// `l3lu`.
    L3lu,
    /// `l3su`.
    L3su,
    /// `constant` — a constant bitstream source.
    Constant,
    /// `sfpstate`.
    SfpState,
    /// `pestate`.
    PeState,
    /// `sfpring`.
    SfpRing,
}

impl DfirUnit {
    /// IS THIS A PT ROW? The PT is the only unit that reads an operand off a wire rather than out
    /// of a register file, so several rules turn on it.
    #[must_use]
    pub const fn is_pt_row(&self) -> bool {
        matches!(self, Self::PtRow(_))
    }

    /// The `type=` this unit is bound with — `senComponentsToString`
    /// (`sys-arch-spec/arch_enums.cpp:11-120`).
    ///
    /// ⛔ LOWERCASE, ALWAYS. `unitTypeTag` lowercases every tag because "DFIR code generation
    /// requires lowercase" (`UnitMaterializer.cpp:34-51`). The `type = "L1LU"` in `Dataflow.td`'s
    /// illustrative example is not what any emitter writes.
    ///
    /// ⛔ AND THIS IS THE `type`, NOT THE `name`. The `name` carries the residency prefix
    /// (`C0-sfp-CL1`); this is the bare tag the consumer matches on. See [`super::islands`]'
    /// `Op::GetUnit` for why only one of the two is load-bearing.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Sfp => "sfp",
            Self::Pe => "pe",
            Self::PtRow(row) => match row.get() {
                0 => "ptrow0",
                1 => "ptrow1",
                2 => "ptrow2",
                3 => "ptrow3",
                4 => "ptrow4",
                5 => "ptrow5",
                6 => "ptrow6",
                _ => "ptrow7",
            },
            Self::Lxlu => "lxlu",
            Self::Lxsu => "lxsu",
            Self::Lx => "lx",
            Self::Hbm => "hbm",
            Self::L0lu => "l0lu",
            Self::L0su => "l0su",
            Self::L0 => "l0",
            Self::L3lu => "l3lu",
            Self::L3su => "l3su",
            Self::Constant => "constant",
            Self::SfpState => "sfpstate",
            Self::PeState => "pestate",
            Self::SfpRing => "sfpring",
        }
    }
}

/// THE UNITS A PROGRAM ON `of` MUST BIND BEFORE ITS BODY CAN NAME THEM.
///
/// A transcription of `buildNeighborUnits`' arms (`DSC2ToDataflowIRUtils.hpp:161-410`). The PT rows
/// are covered by [`PtRow::north`] and [`PtRow::south`] instead, since their arms are one rule.
///
/// ⛔ THE LIST IS WHAT THE C++ BINDS, NOT WHAT SEEMS REASONABLE. The LXLU binds eight units
/// including both L3 halves and PT row 0 (`:369-386`); the SFP binds seven (`:354-370`). A unit
/// missing from a program's preamble is one its body cannot send to.
#[must_use]
pub fn neighbours(of: DfirUnit) -> Vec<DfirUnit> {
    match of {
        // `:369-386` — the LX load unit is the hub: both L3 halves, both compute units, its store
        // twin, the LX itself, PT row 0 and the L0 store unit. EIGHT, and all eight are here.
        DfirUnit::Lxlu => vec![
            DfirUnit::Lxsu,
            DfirUnit::Sfp,
            DfirUnit::Lx,
            DfirUnit::Pe,
            DfirUnit::L3lu,
            DfirUnit::L3su,
            row_or(0, DfirUnit::Pe),
            DfirUnit::L0su,
        ],
        // `:387-398`.
        DfirUnit::Lxsu => vec![
            DfirUnit::Lxlu,
            DfirUnit::Sfp,
            DfirUnit::Lx,
            DfirUnit::Pe,
            DfirUnit::L3lu,
            DfirUnit::L3su,
        ],
        // `:354-370` — the SFP.
        DfirUnit::Sfp => vec![
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::L0su,
            row_or(0, DfirUnit::Pe),
            DfirUnit::Pe,
            DfirUnit::Constant,
            DfirUnit::SfpState,
        ],
        // `:399-410` — the PE, whose north is the LAST PT row: `ptrow7` on RCUDD1A and `ptrow3` on
        // SEN1P5, which the C++ writes as an `if` on the arch and this reads off `Row`'s own bound.
        DfirUnit::Pe => vec![
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::Sfp,
            DfirUnit::Constant,
            DfirUnit::PeState,
            row_or(Target::PT_ROWS - 1, DfirUnit::Sfp),
        ],
        // `:291-296` — the L0 store unit.
        DfirUnit::L0su => vec![DfirUnit::Sfp, DfirUnit::L0lu, DfirUnit::L0, DfirUnit::Lxlu],
        // `:285-290` — the L0 load unit feeds PT row 0.
        DfirUnit::L0lu => vec![DfirUnit::L0su, row_or(0, DfirUnit::Sfp), DfirUnit::L0],
        // A PT row's neighbours are its north and south, plus the L0 load unit to its west
        // (`PTWEST = L0LU`, `:171-172`), and for row 0 the LX load unit as well (`:168-169`).
        DfirUnit::PtRow(row) => {
            let mut units = vec![DfirUnit::L0lu, adjacent(row.north()), adjacent(row.south())];
            if row.get() == 0 {
                units.push(DfirUnit::Lxlu);
            }
            units
        }
        // Memories and sources, not units that run a program of their own.
        DfirUnit::Hbm
        | DfirUnit::Lx
        | DfirUnit::L0
        | DfirUnit::L3lu
        | DfirUnit::L3su
        | DfirUnit::Constant
        | DfirUnit::SfpState
        | DfirUnit::PeState
        | DfirUnit::SfpRing => Vec::new(),
    }
}

/// WHERE A UNIT LIVES ON THIS MACHINE — the residency it must be bound with.
///
/// ⭐ READ, NOT ASSIGNED BY CATEGORY. Three sources, and each covers a different part of the
/// vocabulary:
///
/// * the device description, for everything from the DDR down to the SFP. `%dram` is the ROOT of
///   the memory tree and `%lx` its only depth-1 child, so the first is global and the second is a
///   per-core scratchpad (`spyre_dd2_basic.mlir:70,75`). `%l3lu`/`%l3su` are declared in the
///   `group { kind = "core" }` and `%sfp`/`%lxlu`/`%lxsu` in the `group { kind = "corelet" }`
///   (`:78-79` against `:90-98`), which is exactly the core-wide/per-corelet split.
/// * IBM's own emitted DataflowIR, which agrees: `C0-l3lu` carries `corelet = 0` while `C0-lx`
///   carries none (`/tmp/ktir_ref/export/debug/dfir.mlir:45,62`).
/// * the vendored `dcc/test/PT/xrfbmm_int8_fwd.mlir` for everything BELOW the SFP, which the device
///   description does not yet cover — it binds the L0, the L0 movers and every PT row with
///   `core = 0, corelet = 0`, so they are per-corelet.
///
/// ⛔ THE L0 IS PER-CORELET, NOT A SCRATCHPAD. It looks like the LX — a memory a view is taken over
/// — but the golden binds it with a corelet, and a memory bound without one takes a different
/// branch in `ExtendUnitNameToCorelet`.
#[must_use]
pub fn residency_of(unit: DfirUnit, core: Core, corelet: Corelet) -> Residency {
    match unit {
        // The root of the memory tree: one for the device, neither attribute.
        DfirUnit::Hbm => Residency::Global,
        // Depth one: one per core, `core` and no `corelet`.
        DfirUnit::Lx => Residency::Scratchpad { core },
        // Declared in the core group, so shared across its corelets: `corelet = 0`.
        DfirUnit::L3lu | DfirUnit::L3su => Residency::CoreWide { core },
        // Everything else is declared per corelet.
        DfirUnit::Sfp
        | DfirUnit::Pe
        | DfirUnit::PtRow(_)
        | DfirUnit::Lxlu
        | DfirUnit::Lxsu
        | DfirUnit::L0
        | DfirUnit::L0lu
        | DfirUnit::L0su
        | DfirUnit::Constant
        | DfirUnit::SfpState
        | DfirUnit::PeState
        | DfirUnit::SfpRing => Residency::Corelet { core, corelet },
    }
}

/// WHAT A TEMPLATE'S `unit=` IS, AS THE UNITS DATAFLOWIR BINDS.
///
/// ⛔⛔ ONE `unit=` CAN BE SEVERAL. `ptrow1-7` is a span of SEVEN instruction streams, not a unit:
/// `generatePTInitPacket` runs per row (`dip.cpp:2124-2142`), so collapsing the span to one unit
/// emits a program for one row where the template asked for seven and leaves the other six empty.
/// [`rows_of`] is what expands it, and it is bounded by this arch's own row count.
///
/// ⛔ `Ptnorth` AND `Ptsouth` ARE DIRECTIONS, NOT UNITS. They name the wire a PT row reads from or
/// passes to, which is [`PtRow::north`]/[`PtRow::south`]' job and depends on WHICH row is asking —
/// so they resolve to nothing here rather than to a wrong row.
#[must_use]
pub fn dfir_units(of: Unit) -> Vec<DfirUnit> {
    match of {
        Unit::Sfp => vec![DfirUnit::Sfp],
        Unit::Pe => vec![DfirUnit::Pe],
        Unit::Lxlu => vec![DfirUnit::Lxlu],
        Unit::Lxsu => vec![DfirUnit::Lxsu],
        Unit::L0lu => vec![DfirUnit::L0lu],
        Unit::L0su => vec![DfirUnit::L0su],
        Unit::Constant => vec![DfirUnit::Constant],
        Unit::Sfpring => vec![DfirUnit::SfpRing],
        // The spans, and the singletons that are still written as one.
        Unit::Pt | Unit::Ptrow0 | Unit::Ptrow3 | Unit::Ptrow7 | Unit::Ptrow1To3 | Unit::Ptrow1To7 => {
            rows_of(of).into_iter().map(DfirUnit::PtRow).collect()
        }
        // A direction, resolved by the row that asks. See above.
        Unit::Ptnorth | Unit::Ptsouth => Vec::new(),
    }
}

/// A PT row by index, or `fallback` where this arch has no such row.
fn row_or(index: u32, fallback: DfirUnit) -> DfirUnit {
    Row::checked(index).map_or(fallback, DfirUnit::PtRow)
}

/// What an [`Adjacent`] is, as a unit.
pub fn adjacent(next: Adjacent<{ Target::PT_ROWS }>) -> DfirUnit {
    match next {
        Adjacent::Row(row) => DfirUnit::PtRow(row),
        Adjacent::Sfp => DfirUnit::Sfp,
        Adjacent::Pe => DfirUnit::Pe,
    }
}

/// WHAT THE ROW SPANS EXPAND TO on this arch.
///
/// ⛔ `ptrow1-7` IS SEVEN INSTRUCTION STREAMS. `generatePTInitPacket` runs per row —
/// `"pt_row" + p` (`dip.cpp:2124-2142`) — so a span left unexpanded is a program emitted for one row
/// where the template asked for seven, and the other six are silently empty.
///
/// ⭐ AND THE UPPER END IS THE ARCH'S. `ptrow1-7` on SEN1P5, which has four rows, is rows 1..=3.
#[must_use]
pub fn rows_of(span: Unit) -> Vec<Row> {
    // ⛔ EVERY ROW GOES THROUGH `Row::checked`, INCLUDING THE SINGLETONS. `ptrow7` names a row
    // SEN1P5 does not have, and returning it unchecked would emit a program for an instruction
    // stream the chip has none of.
    let last = Target::PT_ROWS - 1;
    let rows: Vec<u32> = match span {
        Unit::Ptrow0 => vec![0],
        Unit::Ptrow3 => vec![3],
        Unit::Ptrow7 => vec![7],
        Unit::Ptrow1To3 => (1..=3.min(last)).collect(),
        Unit::Ptrow1To7 => (1..=7.min(last)).collect(),
        // The whole PT: every row.
        Unit::Pt => (0..Target::PT_ROWS).collect(),
        Unit::Ptnorth
        | Unit::Ptsouth
        | Unit::Sfp
        | Unit::Pe
        | Unit::Lxlu
        | Unit::Lxsu
        | Unit::L0lu
        | Unit::L0su
        | Unit::Sfpring
        | Unit::Constant => Vec::new(),
    };
    rows.into_iter().filter_map(Row::checked).collect()
}

/// EVERY INVARIANT OF THE UNIT MODEL, AS CONSTS THE COMPILER EVALUATES FOR THIS BUILD'S ARCH.
const _: () = {
    // The chain is closed at both ends: row 0's north is the SFP, and the last row's south is the PE.
    assert!(matches!(
        Row::checked(0).expect("every arch has a PT row 0").north(),
        Adjacent::Sfp
    ));
    assert!(matches!(
        Row::checked(Target::PT_ROWS - 1)
            .expect("PT_ROWS - 1 is a row")
            .south(),
        Adjacent::Pe
    ));
    // And it is a chain, not a ring: no interior row's south is the PE.
    assert!(matches!(
        Row::checked(0).expect("every arch has a PT row 0").south(),
        Adjacent::Row(_)
    ));
    // ⛔ THE ROW COUNT IS THE ARCH'S, NOT A LITERAL. `Row::checked(PT_ROWS)` must be `None` — that is
    // the whole content of the bound, and the check that would have caught `PtRow::<{ 8 }>` on a
    // four-row build.
    assert!(Row::checked(Target::PT_ROWS).is_none());
};
