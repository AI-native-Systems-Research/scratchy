// SPDX-License-Identifier: Apache-2.0
//! WHAT EVERY REGISTER HOLDS BEFORE THE FIRST INSTRUCTION — `regState_.getSimpleRegInit()` (`dip.cpp:3514-3515`),
//! the map the init packet's LRF flits are built from.
//!
//! ⭐⭐ THIS IS A REGION, NOT A PREAMBLE OF INSTRUCTIONS, and that is the reason it is its own island-3 output
//! rather than more ops on the tape. `ConstructAssignInstr` emits NOTHING for `imm→lrf` and `imm→lar/ear` when
//! `program_header` is set — `return std::nullopt` (`ConstructProgIRHelper.cpp:149`, `:219`) — and
//! `LowerSentientHelper.cpp:617-640` calls `addToRegInit` in its place. So a stated initial value is packet bytes,
//! and only a value that CHANGES during the program is an instruction.
//!
//! ⛔ THE SCALE IS PER (UNIT, SPACE) AND IT IS NOT 1. Every immediate that carries an address is
//! `value * element_size / 8 / GetAddressScale(comp, locale)` (`ConstructProgIRHelper.cpp:3122-3127`,
//! `LowerSentientHelper.cpp:416`; ⛔ `GetAddressScale` itself is DEFINED in
//! `LowerSentientHelper.cpp:502`, NOT in `ConstructProgIRHelper.cpp` — the lines this cited there,
//! `:154-157` and `:229-232`, are the JCR arm and the `lar`/`ear` locale branch of a different function), and the table is [`AddressScale`]. An L0SU address is scaled by the PT ROW COUNT and an L3
//! address by 128, so writing an element offset into either names somewhere else entirely.

use crate::alloc::{Bytes, Placement, Sticks};
use crate::bridges::sentient_to_progir::Assignment;
use crate::expansion_tape::GroupPos;
use crate::isa::regfile::{Component, RegType};
use crate::islands::superdsc::Group;
use crate::islands::progir::{PhysicalOperand, RegIndex};
use crate::template::{Attrs, Operands};

/// WHICH CORE A REGISTER VALUE BELONGS TO — the index `target_core_mask` addresses.
///
/// ⭐ AN IDENTITY, NOT AN OFFSET: it keys the init values and picks the mask a block is addressed to, and nothing
/// arithmetic is done with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Core(u32);

impl Core {
    pub const fn of(index: u32) -> Self {
        Self(index)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// WHICH ADDRESS SPACE A REGISTER POINTS INTO — the second key of `addressGranularityScalePerUnit`
/// (`sysdef.cpp:504-525`).
///
/// ⛔ THE SPACE IS NOT THE UNIT'S OWN MEMORY. An L3 unit's LAR points into LX and its EAR into HBM, at DIFFERENT
/// scales (`LowerSentientHelper.cpp:508-514` picks by register locale), so "which unit" cannot answer "which
/// scale".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Space {
    /// HBM — an L3 unit's external address.
    Hbm,
    /// The LX scratchpad.
    Lx,
    /// A core's L0.
    L0,
    /// The L0 scale region.
    L0Scale,
    /// The SFP's local register file — `memory="sfplrf"`, 184 of the templates' allocations.
    SfpLrf,
    /// The SFP's state file — `memory="sfpstate"`.
    SfpState,
    /// The PE's local register file — `memory="pelrf"`.
    PeLrf,
    /// The PT's accumulator file — `memory="ptarf"`.
    PtArf,
    /// The PT's weight file — `memory="ptxrf"`.
    PtXrf,
    /// An L3 unit's indirect-base register file.
    Ibr,
}

impl Space {
    /// ⭐ WHETHER THIS SPACE IS A COMPUTE UNIT'S REGISTER FILE — the five the templates spell, asked as one
    /// question wherever they share an answer.
    pub const fn is_register_file(self) -> bool {
        match self {
            Self::SfpLrf | Self::SfpState | Self::PeLrf | Self::PtArf | Self::PtXrf => true,
            Self::Hbm | Self::Lx | Self::L0 | Self::L0Scale | Self::Ibr => false,
        }
    }
}

/// WHAT A START ADDRESS IS MULTIPLIED BY TO BECOME AN ELEMENT OFFSET — `getAddressGranularityMultiplyFactor`
/// (`SNDSCLowering.cpp:156-172`).
///
/// ```cpp
/// loc.unit_ = senCompToGenericComp.at(unit);  loc.storage_ = storage;
/// auto factor = (double)sysDef.addressGranularityScalePerUnit.at(loc);
/// factor = factor * 8;
/// unsigned bitwidth = getIntOrFloatBitWidth(precision);
/// if (bitwidth == 24) factor = factor / 16.0; else factor = factor / bitwidth;
/// ```
///
/// ⛔⛔ THE 24-BIT CASE DIVIDES BY **16**, NOT 24. A `SENINT24` element is carried in a 16-bit-granular slot, so the
/// scale is the SLOT's width and not the datum's — dividing by 24 would give an offset ⅔ of the right one on every
/// int24 transfer.
///
/// ⛔ A `f64` BEHIND A NEWTYPE, because the reference's is a `double` and the product is later TRUNCATED
/// (`int(getSingleDataStrict(…) * factor)`, `SNControlFlowLowering.cpp:621`). Keeping it integral would round
/// differently for any non-integral factor, and the difference is a wrong base address with nothing to catch it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularityFactor(f64);

impl GranularityFactor {
    /// ⭐ `A` IS READ, NOT DECORATION: `address_scale::<A>` answers `numPTRows` for the two L0 units, so this
    /// function's value differs per arch and the type parameter is what carries that.
    pub fn of<A: crate::consts::Arch>(
        unit: Component,
        space: Space,
        element: crate::islands::dataflow_ir::vectorchain::Element,
    ) -> Self {
        let scale = f64::from(address_scale::<A>(unit, space));
        let bits = element.bits();
        Self(scale * 8.0 / if bits == 24 { 16.0 } else { bits as f64 })
    }

    /// `int(start_addr * factor)` — a START address becomes an ELEMENT offset, truncating toward zero.
    ///
    /// ⛔ THE TWO TYPES ARE THE POINT: this is the only crossing between them, so applying the factor twice or not
    /// at all is a type error rather than a plausible number. Truncation, not rounding — the reference's cast.
    pub fn scale(
        self,
        address: crate::bridges::a_superdsc_to_dataflow_ir::passes::stages::StartAddress,
    ) -> crate::bridges::a_superdsc_to_dataflow_ir::passes::stages::ElementOffset {
        crate::bridges::a_superdsc_to_dataflow_ir::passes::stages::ElementOffset::of(
            (address.get() as f64 * self.0) as i64,
        )
    }
}

/// HOW MANY BYTES ONE UNIT OF A REGISTER'S ADDRESS IS — `addressGranularityScalePerUnit` (`sysdef.cpp:504-525`).
///
/// ⛔ THE L0SU's SCALE IS THE PT ROW COUNT, WHICH IS AN ARCH FACT: 8 up to RCUDD1A and 4 above
/// (`sysdef.cpp:199`, `:514`). And it differs from the L0LU's, which is 1 — the same memory, two units, two
/// scales, because a store fans across the rows.
///
/// ⛔ A PAIRING THE TABLE DOES NOT LIST IS A REFUSAL, not a 1. `GetAddressScale` returns 1 only for units that
/// have no entry at all — the PT, SFP and PE reached through their `else` (`LowerSentientHelper.cpp:516`) — while
/// an L3 register with an unexpected locale is `llvm_unreachable` (`:514`). Defaulting to 1 for an unlisted L0/LX
/// pairing would write an address 8x too large and pass every check that only tries the listed ones.
pub const fn address_scale<A: crate::consts::Arch>(unit: Component, space: Space) -> u32 {
    // The refusal is one string because a `const fn`'s `panic!` cannot format the pair; which pair it was is in
    // the caller's own line.
    const UNLISTED: &str = "`addressGranularityScalePerUnit` has no entry for this unit and this space \
                            (`sysdef.cpp:504-525`), so there is no scale to divide an address by. \
                            `GetAddressScale` answers 1 only for a unit with NO entries at all \
                            (`LowerSentientHelper.cpp:502-517`), and defaulting to 1 for a listed unit writes an \
                            address the scale would have divided";
    match unit {
        // `{L3LU, HBM} = 128`, `{L3LU, LX} = 128`, `{L3LU, L3LUIBR} = 4`, and the L3SU's the same.
        Component::L3lu | Component::L3su => match space {
            Space::Hbm | Space::Lx => 128,
            Space::Ibr => 4,
            Space::L0 | Space::L0Scale | Space::SfpLrf | Space::SfpState | Space::PeLrf | Space::PtArf | Space::PtXrf => panic!("{}", UNLISTED),
        },
        // `{LXLU, LX} = 1` — an LX address is in BYTES, which is why its register is 24 bits wide for a 2 MiB part.
        Component::Lxlu | Component::Lxsu => match space {
            Space::Lx => 1,
            Space::Hbm | Space::L0 | Space::L0Scale | Space::Ibr | Space::SfpLrf | Space::SfpState | Space::PeLrf | Space::PtArf | Space::PtXrf => {
                panic!("{}", UNLISTED)
            }
        },
        // ⛔⛔ THE TABLE'S 1 IS NOT THE L0LU's SCALE. `addressGranularityScalePerUnit[{L0LU, L0}] = 1`
        // (`sysdef.cpp:542`), and then the reader MULTIPLIES IT BY THE ROWS:
        //
        //     auto addrScale = sysDef.addressGranularityScalePerUnit.at({genericUnit, loc.storage_});
        //     if (genericUnit == L0LU) addrScale *= sysDef.numPTRows;      (`dsc2.cpp:2953-2957`)
        //
        // So an L0LU address is in units of `numPTRows`, exactly like the L0SU's — the table entry and the SCALE
        // are two different numbers for this one unit, and reading the table alone answers 1 for a quantity the
        // machine counts in eights.
        Component::L0lu => match space {
            Space::L0 | Space::L0Scale => <A::PtRows as crate::consts::PtRows>::VALUE as u32,
            Space::Hbm | Space::Lx | Space::Ibr | Space::SfpLrf | Space::SfpState | Space::PeLrf | Space::PtArf | Space::PtXrf => panic!("{}", UNLISTED),
        },
        // `{L0SU, L0} = numPTRows` straight from the table (`sysdef.cpp:543`), with no further multiply — which is
        // why the two units' entries differ while their SCALES do not.
        //
        // ⭐ AND `L0_SCALE` FOLLOWS THE SAME PATTERN, which is why both spaces map alike here:
        // `[{L0LU, L0_SCALE}] = 1` and `[{L0SU, L0_SCALE}] = numPTRows` (`sysdef.cpp:544-545`), with the L0LU
        // multiply at `dsc2.cpp:2954-2955` applying whatever the storage is.
        Component::L0su => match space {
            Space::L0 | Space::L0Scale => <A::PtRows as crate::consts::PtRows>::VALUE as u32,
            Space::Hbm | Space::Lx | Space::Ibr | Space::SfpLrf | Space::SfpState | Space::PeLrf | Space::PtArf | Space::PtXrf => panic!("{}", UNLISTED),
        },
        Component::Sfp | Component::Pe | Component::Pt => match space {
            // ⭐ ONE ENTRY FOR ALL FIVE FILES: `addressGranularityScalePerUnit` reaches the PT, SFP and
            // PE through `GetAddressScale`'s `else` (`LowerSentientHelper.cpp:516`) and a register IS a stick,
            // so a file's address is in units of 128 bytes whichever file it is.
            Space::SfpLrf | Space::SfpState | Space::PeLrf | Space::PtArf | Space::PtXrf => 128,
            Space::Hbm | Space::Lx | Space::L0 | Space::L0Scale | Space::Ibr => {
                panic!("{}", UNLISTED)
            }
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WHAT THE SCALES PIN
// ─────────────────────────────────────────────────────────────────────────────

/// ⛔⛔ THE TWO L0 UNITS AGREE, AND THE OPPOSITE USED TO BE PINNED HERE. This assertion read
/// `address_scale(L0lu, L0) != address_scale(L0su, L0)`, justified as "a store fans across the PT rows and a load
/// does not" — a sentence about the table's entries (`{L0LU, L0} = 1` vs `{L0SU, L0} = numPTRows`) and not about
/// the scales the machine uses. `dsc2.cpp:2953-2957` multiplies the L0LU's entry by `numPTRows` before using it,
/// so both units address L0 in units of `numPTRows`.
///
/// ⛔ AN ASSERTION THAT AGREES WITH THE CODE PROVES NOTHING, so this pins the wrong answer as ABSENT: neither
/// scale is 1, which is what reading the table alone produces.
const _: () =
    assert!(address_scale::<crate::arch::this::ThisArch>(Component::L0lu, Space::L0) == address_scale::<crate::arch::this::ThisArch>(Component::L0su, Space::L0));
const _: () = assert!(address_scale::<crate::arch::this::ThisArch>(Component::L0lu, Space::L0) != 1);
const _: () =
    assert!(address_scale::<crate::arch::this::ThisArch>(Component::L0su, Space::L0) == crate::isa::unit::NUM_PT_ROWS as u32);

/// ⛔ AN LX ADDRESS IS IN BYTES AND AN L3 ADDRESS IS IN STICKS, so the same distance is two numbers 128 apart.
const _: () = assert!(address_scale::<crate::arch::this::ThisArch>(Component::Lxlu, Space::Lx) == 1);
const _: () = assert!(
    address_scale::<crate::arch::this::ThisArch>(Component::L3lu, Space::Lx) == crate::isa::memory::BYTES_PER_STICK as u32
);

// ─────────────────────────────────────────────────────────────────────────────
// THE ADDRESS REGISTERS — what each one holds before the first instruction.
// ─────────────────────────────────────────────────────────────────────────────

/// WHICH BACKING STORE A DATASPACE LIVES IN.
///
/// ⛔ THE TWO ARE DIFFERENT MEMORIES AND A SPAN IS ONLY MEANINGFUL BESIDE ONE. `Move` states an `hbm` end and
/// an `lx` end and their stick numbers are counted from different bases, so a span without its store is a
/// number that indexes either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Store {
    /// The device's HBM — one image shared by every core.
    Hbm,
    /// One core's LX scratchpad.
    Lx,
}

/// ⛔⛔⛔ WHO PUT THE BYTES IN ONE END OF A MOVEMENT — a fact about the PAIR (end, direction), and
/// meaningless from either alone.
///
/// A [`Move`] states an HBM end AND an LX end, and BOTH are recorded as dataspaces
/// (`bridges/1.rs`'s `(Store::Hbm, &moved.hbm)` beside `(Store::Lx, &moved.lx)`). So a dataspace is
/// ONE END of a transfer, never "the place this op writes" — and [`Direction`] is the direction of the
/// MOVEMENT, not a property of the end: an L3 load brings HBM INTO the LX, an L3 store carries what
/// the op produced back OUT (`RegisterTypeAssignment.cpp:877-879`).
///
/// ⛔ READING ONE FIELD IS HOW THIS GOES WRONG, TWICE MEASURED. `direction == Store` was read as "the
/// program writes here" and made the read-before-write check assert the very thing it tests; then
/// `store == Hbm && direction == Store` was read as "this op stores to HBM" and refused 6108 correct
/// reductions whose template (`summeanmaxexx2.ddl`) names no `hbm` at all. Both are the same error:
/// asking a question of half a fact.
///
/// ⭐ SO THERE IS ONE PREDICATE AND IT TAKES THE WHOLE PAIR. Four cases, all named, no `_` arm — a
/// fifth combination cannot exist because neither enum can grow without breaking this match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fills {
    /// The HOST put these bytes there before the program ran: reading them first is correct.
    Host,
    /// The PROGRAM must write these bytes: reading them before it does is a producer/consumer fault.
    Program,
}

/// ONE DATASPACE THE MODEL MUST LAY DOWN — where it is, and what it is MADE OF.
///
/// ⛔⛔⛔ THIS IS WHAT A SINGLE `fill` CANNOT SAY. `run_the_machine` painted every store with one format taken
/// from the op's FIRST INPUT tensor (`islands::superdsc::SdscOp::precision`), so a quantiser's fp16 input and its
/// fp8 output were both written in whichever of the two came first — and the datapath then refused an operand
/// for a difference the HARNESS introduced rather than the program.
///
/// ⭐ EVERY FIELD COMES FROM THE MOVEMENT THAT ADDRESSES IT: `Move::touches` for the span and `Move::format`
/// for the format, so the memory seeded is the memory the program walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dataspace {
    pub store: Store,
    pub span: crate::islands::progir::Touched,
    pub format: crate::islands::superdsc::Format,
    /// WHETHER THE MOVEMENT THAT LAYS THIS DOWN BRINGS THE REGION IN OR CARRIES IT OUT.
    ///
    /// ⛔ WITHOUT IT AN ALIASING REPORT IS UNREADABLE. "op 45 paints LX line 0 fp8" is two completely different
    /// findings — its INPUT is fp8, so the conversion emitted for it is the wrong one; or its OUTPUT is fp8 and
    /// has been painted over the fp16 input it has not read yet — and they need opposite fixes.
    pub direction: crate::islands::progir::Direction,
    /// ⛔⛔⛔ HOW MANY STICKS THIS END'S WALK ACTUALLY VISITS — which is NOT [`Self::span`]'s extent.
    ///
    /// [`crate::islands::progir::Touched`] is a first..last ENVELOPE, so a STRIDED walk has an extent far
    /// larger than the set it touches. MEASURED: a store with a 21-stick envelope visits 5 sticks at
    /// stride 5, and comparing a moved COUNT against that envelope reads as "16 sticks missing" when
    /// nothing is missing at all. A count and an envelope are different facts; this is the count.
    pub sticks: u64,
    /// ⛔⛔⛔ WHOSE TENSOR THIS IS — and WITHOUT IT `fills` CANNOT ANSWER ITS OWN QUESTION.
    ///
    /// "Who put the bytes here" is not decided by the store and the direction alone: an HBM LOAD of a
    /// KERNEL reads what the host DMA'd, and an HBM LOAD of an ACTIVATION reads what an earlier
    /// program was supposed to have written. Those are the two cases the senulator exists to tell
    /// apart, and folding them together is what made it blind to every read of unwritten memory.
    pub role: crate::islands::superdsc::DsType,
}

impl Dataspace {
    /// WHO PUT THE BYTES IN **THIS END** — see [`Fills`]. The only way to ask.
    ///
    /// ⛔ ALL FOUR CASES NAMED. An L3 LOAD moves HBM -> LX, so its HBM end is the SOURCE the host
    /// filled and its LX end is the DESTINATION the transfer writes. An L3 STORE moves LX -> HBM, so
    /// its LX end is the source the compute produced and its HBM end is the destination
    /// (`RegisterTypeAssignment.cpp:877-879`). Neither field decides this alone.
    /// ⛔⛔⛔ AND THE ROLE IS THE THIRD TERM. An HBM LOAD used to answer [`Fills::Host`] for EVERY
    /// tensor, so the senulator SEEDED every HBM address a program read — a load from an address
    /// nothing ever wrote came back well-formed, the op computed, drained and RETURNed, and all four
    /// counters stayed at zero. That is the senulator declining the one job it exists for: tracking
    /// addressing and memory. The card has no seeder and reads whatever is actually there.
    ///
    /// ⭐ THE MACHINE ALREADY HAS THE MECHANISM. [`Fills::Program`] becomes a
    /// `Filled::ByProgram` span, and `Machine::synthesize_seed` keeps the span — so the address
    /// resolves and a store still lands — while withholding the FORMAT, "the one bit that claims a
    /// value was written". Only this predicate ever pointed at it.
    #[must_use]
    pub fn fills(&self) -> Fills {
        use crate::islands::superdsc::DsType;
        use crate::islands::progir::Direction;
        match (self.store, self.direction, self.role) {
            // The host DMA'd it before the model ran: the weights.
            (Store::Hbm, Direction::Load, DsType::Kernel) => Fills::Host,
            // ⛔⛔⛔ AN ACTIVATION READ OUT OF HBM IS SOMEBODY'S OUTPUT. Answering `Host` here is
            // what made the senulator blind to a read of unwritten memory: the seeder filled every
            // HBM address a program read, so a load from an address NOTHING EVER WROTE came back
            // well-formed, the op drained, and all four counters stayed at zero. The card has no
            // seeder. `Fills::Program` makes `synthesize_seed` withhold the FORMAT — "the one bit
            // that claims a value was written" — and the read is refused instead.
            //
            // ⭐ THIS ONLY BECAME SAFE ONCE `Machine::fill_authority` EXISTED. The seed lookup used
            // to take the LAST covering span, so an op's `ByProgram` OUTPUT span overlapping its
            // `Kernel` INPUT span poisoned a legitimate weight read — MEASURED, 1,565 of 173,701
            // reads had covering spans that disagree, and the first fault this arm produced was
            // exactly one of them (`HbmStickAddr(402653653)`, a Kernel by the dataflow census).
            // ⛔⛔⛔ `Fills::Program` IS THE HONEST ANSWER AND IT CANNOT BE ARMED YET — it has FALSE
            // POSITIVES BY CONSTRUCTION. A machine is built PER OP, so op N's activation input was
            // written by op N-1 on a DIFFERENT machine: an honest answer calls EVERY legitimate
            // cross-op edge unwritten. MEASURED — it refuses `HbmStickAddr(402653653)`, a region this
            // op reads as an `Output` and which another op genuinely produces.
            //
            // ⭐ IT BECOMES PRECISE THE MOMENT A GROUP'S OPS RUN IN ORDER ON ONE MACHINE, and not
            // before. Until then `Host` is the only answer that does not fire on correct programs.
            (Store::Hbm, Direction::Load, DsType::Input | DsType::Output) => Fills::Host,
            // The load's destination — nothing is there until the transfer runs.
            (Store::Lx, Direction::Load, DsType::Kernel | DsType::Input | DsType::Output) => {
                Fills::Program
            }
            // The store's source: the compute wrote this scratchpad line.
            (Store::Lx, Direction::Store, DsType::Kernel | DsType::Input | DsType::Output) => {
                Fills::Program
            }
            // The store's destination out in HBM.
            (Store::Hbm, Direction::Store, DsType::Kernel | DsType::Input | DsType::Output) => {
                Fills::Program
            }
        }
    }
}

/// WHAT ONE GROUP INITIALISES BEFORE ITS FIRST INSTRUCTION.
///
/// ⭐ BUILT BY BRIDGE 2's PASS, PACKED BY BRIDGE 3, the same split as the instruction tape: bridge 2 says WHICH
/// register holds the value because it is the pass that hands registers out, and bridge 3 says which bytes of which
/// flit that register's value lands in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterInits {
    /// EVERY DATASPACE THIS OP'S MOVEMENTS ADDRESS, per (op, CORE) — what the model has to lay memory down in.
    ///
    /// ⛔ PER CORE FOR THE SAME REASON `addresses` IS: the placement is a per-core fold, so one core's spans
    /// are not another's. See [`Dataspace`].
    dataspaces: std::collections::BTreeMap<(GroupPos, Core), Vec<Dataspace>>,
    /// WHICH CORES EACH OP RUNS ON — `numCoresUsed_`, carried here because the tape does not carry it.
    ///
    /// ⛔⛔⛔ PER OP, AND A GROUP'S OPS DISAGREE. `progstateinfo_` is a map keyed by core id and every entry is
    /// built by walking the program units that name that core (`SentientToProgIR.cpp:665-673`), so an op
    /// contributes NOTHING to a core it does not use — `convertIr2Senprog` then refuses a core that is not in the
    /// map at all (`dpc.cpp:627-628`). Bridge 3 walks one tape for every core of the group and had no way to ask
    /// which of them an op belongs to.
    op_cores: std::collections::BTreeMap<GroupPos, crate::islands::superdsc::Cores>,
    /// An address register's initial value, per (CORE, unit, register), in the unit's own address units.
    ///
    /// ⛔⛔ THE CORE IS PART OF THE KEY BECAUSE THE VALUE DIFFERS BY CORE. `startAddressCoreCorelet_` is a per-core
    /// FOLD and dxp's init packet is per core (`initcores`, `dip.cpp:1326-1340`) — measured: tensor `t0` at
    /// `0xc00000000` on one core and `0xc00000080` on another, one stick apart, which is the work split. One value
    /// for all cores puts core 0's slice in every core's program.
    addresses: std::collections::BTreeMap<(GroupPos, Core, Component, RegIndex), Bytes>,
    /// AN L3 ADDRESS REGISTER'S INITIAL VALUE, per (CORE, unit, FILE, register).
    ///
    /// ⛔⛔ THE FILE IS IN THE KEY BECAUSE THE L3 HAS FOUR OF THEM AND NO LRF. LAR and LBR address the LX side,
    /// EAR and EBR the HBM side (`ConstructProgIRHelper.cpp:2972-2996`), with different depths — so `register 0`
    /// names four different registers and the map above cannot tell them apart.
    /// ⛔⛔⛔ **STICKS, NOT BYTES.** An L3 address register counts 128-byte sticks, and typing this map as `Bytes`
    /// is exactly what let `crate::alloc::Bytes::of(value.get())` in `bridges::l3` pour a `Sticks` count into a
    /// `Bytes` newtype with no multiply — silently, for the whole staging path. Every consumer then read a stick
    /// count as a byte count and was wrong by 128: the LX definedness law compared a staged base of 65 against a
    /// read at byte 8320, which is the SAME address (65 x 128), and reported a defect that was not there.
    ///
    /// ⭐ THE TYPE IS THE GUARD. Naming the unit here makes the mislabel a COMPILE ERROR at every consumer rather
    /// than a factor of 128 nobody can see — which is the crate's own rule: no `From` between quantities that must
    /// not interchange.
    l3: std::collections::BTreeMap<(GroupPos, Core, Component, RegType, RegIndex), Sticks>,
}

impl RegisterInits {

    /// THE L3's ADDRESS REGISTERS ON ONE UNIT — the file, the register, and the value it starts at.
    ///
    /// ⛔ A SECOND MAP BECAUSE THE KEY IS A DIFFERENT ONE. `addresses` is keyed `(core, unit, register)`, which
    /// names nothing on an L3 unit: its addresses live in four FILES of different depths (`EBR holds 8 and LAR 16
    /// on RCUDD1A`), so LAR 0 and EAR 0 are two registers and one key cannot hold both.
    pub fn l3_addresses_of(
        &self,
        at: GroupPos,
        core: Core,
        unit: Component,
    ) -> Vec<(RegType, RegIndex, Sticks)> {
        // A range for the same reason as [`Self::addresses_of`] — the key is prefix-ordered, so the entries for
        // one (program, core, unit) are contiguous.
        self.l3
            .range(
                (at, core, unit, RegType::FIRST, RegIndex::FIRST)
                    ..=(at, core, unit, RegType::LAST, RegIndex::LAST),
            )
            .map(|((_, _, _, file, reg), value)| (*file, *reg, *value))
            .collect()
    }

    /// The address registers one unit holds, in register order.
    pub fn addresses_of(
        &self,
        at: GroupPos,
        core: Core,
        unit: Component,
    ) -> Vec<(RegIndex, Bytes)> {
        // ⛔⛔ A RANGE, NOT A FILTER OVER THE WHOLE MAP. This scanned every entry per query, and bridge 3 asks it
        // once per (program, stream, CORE) — so on a 512-op group with 32 cores it re-walked the entire register
        // map thousands of times. Measured in the build-script trace before this: `i3::to_senprog` cost 897 s in
        // total with a worst single group of 288 s, against a whole-emit budget of seconds.
        //
        // ⭐ THE KEY IS ALREADY PREFIX-ORDERED — `(GroupPos, Core, Component, RegIndex)` — so the matching entries
        // are one CONTIGUOUS span and a `range` reaches them in `O(log n + k)`.
        self.addresses
            .range((at, core, unit, RegIndex::FIRST)..=(at, core, unit, RegIndex::LAST))
            .map(|((_, _, _, reg), value)| (*reg, *value))
            .collect()
    }

    /// HOW MANY DISTINCT VALUES EACH LRF INDEX TAKES **ACROSS THESE CORES** — `LXL0LRFInitCompression`'s
    /// `uniqueValsForIdx` (`dip.cpp:1325-1345`).
    ///
    /// ⛔⛔ A CROSS-CORE FACT, AND THAT IS THE WHOLE POINT OF IT. The count is what the LX/L0 header's index list
    /// is SORTED BY (`arrangelrfidx`, `dip.cpp:1304-1312`), ascending — an index holding the same value on every
    /// core sorts before one that varies, so the flits carrying the agreeing registers can be shared between
    /// cores. A per-core census makes every count one, which is exactly the wrong answer: measured against
    /// `dip_standalone`'s own `init.txt`, the LxLu header reads `0x2310` — index order `[0,1,3,2]` — where the
    /// ascending list gives `0x3210`.
    ///
    /// ⛔ AND THE CORES ARE `initcores`, the cores that HAVE a program (`dip.cpp:1326`), not every core the machine
    /// has — so the caller passes the set it is emitting for.
    pub fn lrf_unique_counts(
        &self,
        at: GroupPos,
        unit: Component,
        cores: &[Core],
    ) -> std::collections::BTreeMap<RegIndex, usize> {
        let mut seen: std::collections::BTreeMap<RegIndex, std::collections::BTreeSet<Bytes>> =
            std::collections::BTreeMap::new();
        for core in cores {
            for (reg, value) in self.addresses_of(at, *core, unit) {
                seen.entry(reg).or_default().insert(value);
            }
        }
        seen.into_iter()
            .map(|(reg, values)| (reg, values.len()))
            .collect()
    }

    /// EVERY CORE THIS GROUP INITIALISES A REGISTER ON, ascending — the cores bridge 3 must emit a block for.
    pub fn cores(&self) -> Vec<Core> {
        let mut all: Vec<Core> = self.addresses.keys().map(|(_, core, _, _)| *core).collect();
        // ⛔ THE L3'S CORES COUNT TOO. A staging transfer's EAR is the one value that really differs per core, so a
        // group whose only per-core values are the L3's would otherwise report no cores at all.
        all.extend(self.l3.keys().map(|(_, core, _, _, _)| *core));
        all.sort();
        all.dedup();
        all
    }

    // Record one address, refusing a second value for a register already spoken for.
    //
    // ⛔ THE CONFLICT IS THE POINT: two memory views in one address register means the allocation handed the same
    // register to two live values, and the second load would read the first view's base.

    /// RECORD ONE L3 ADDRESS REGISTER'S STATED VALUE — what the declared staging path hands over.
    ///
    /// ⭐ THE VALUE IS ALREADY IN THE UNIT'S ADDRESS UNITS. An L3 register counts in 128-byte sticks on both
    /// sides (`addressGranularityScalePerUnit`, `sysdef.cpp:504-510`) and the LRF flit carries that number, so
    /// nothing here scales it again — which is why this takes the value rather than a byte address plus a space.
    /// RECORD WHAT A MOVEMENT ADDRESSES, so the model can lay that memory down in the format it IS.
    ///
    /// ⭐ CALLED WHERE THE MOVEMENT IS DERIVED, beside `set_l3_value`, because both are the same fact about
    /// the same `Move` at the same (op, core): one says which registers point at the dataspace and this says
    /// what the dataspace is made of. Recording them apart is how they would come to disagree.
    pub fn lays_down(&mut self, at: GroupPos, core: Core, space: Dataspace) {
        let held = self.dataspaces.entry((at, core)).or_default();
        // ⛔ ONE ENTRY PER DISTINCT DATASPACE. A group states the same movement once per chunk level and per
        // buffering pass; seeding it twice writes the same format twice, but the LIST is read by a caller
        // that reports what it laid down, and a duplicate there reads as two tensors where there is one.
        if !held.contains(&space) {
            held.push(space);
        }
    }

    /// EVERY DATASPACE THE **GROUP** LAYS DOWN ON ONE CORE, in op order.
    ///
    /// ⛔⛔⛔ PER GROUP AND NOT PER OP, BECAUSE MEMORY IS NOT PER OP. The model is built fresh for each
    /// (op, core) — a senprog is per op (`dxp.cpp:836-846`) — but the LX and HBM it reads were staged by
    /// EARLIER ops of the same group. MEASURED, granite-3.1-2b fp8: the 695 ops that refuse an operand's
    /// format record ZERO movements of their own; every stick they read was laid down by somebody else, so a
    /// per-op seed leaves exactly those ops with nothing.
    ///
    /// ⭐ IN OP ORDER, AND LAST WRITER WINS. `BTreeMap` orders by `(GroupPos, Core)`, so filtering to one
    /// core yields the ops in the order they run — which is the order the dataspaces are actually written in,
    /// and therefore the order that decides what a region holds when two of them overlap.
    ///
    /// ⛔⛔⛔ UP TO AND INCLUDING `at`, AND THE PARAMETER IS WHY THIS FUNCTION CANNOT BE ASKED THE WRONG
    /// QUESTION. "What has this core's memory been made of" has no answer on its own: the LX is a SCRATCHPAD
    /// every op of the group reuses, so a region holds whatever the most recent op to write it put there, and
    /// that depends on WHEN you ask. Laying the whole group down and then running op N gave op N the formats
    /// of ops that have not run yet.
    ///
    /// ⛔ MEASURED, granite-3.1-2b fp8: a `GCVT` with `imm = 4` — "fp16 pair to SEN143_FP8", a QUANTISER, so
    /// fp16 in and fp8 out (`ConstructProgIRHelper.cpp:1918-1934`, and `:2018` puts the selector in `imm`) —
    /// read `Sen143Fp8` on both of its sources. Its fp16 input had been painted over by the fp8 OUTPUT
    /// dataspace of a later op sharing the same LX lines. 647 refusals a bake, every one of them this.
    /// ⭐ AND EACH ONE NAMES THE OP THAT LAID IT. Two spans overlapping in different formats is an ALIASING
    /// question, and a list that does not say WHO laid each cannot answer it: "an earlier op left fp8 here" and
    /// "this op's own output is painted over its input" are the same picture without the op.
    pub fn laid_down(&self, core: Core, at: GroupPos) -> Vec<(GroupPos, Dataspace)> {
        self.dataspaces
            .iter()
            .filter(|((op, held), _)| *held == core && *op <= at)
            .flat_map(|((op, _), spaces)| spaces.iter().map(move |space| (*op, *space)))
            .collect()
    }

    pub fn set_l3_value(
        &mut self,
        at: GroupPos,
        core: Core,
        unit: Component,
        file: RegType,
        reg: RegIndex,
        value: Sticks,
    ) {
        self.set_l3(at, core, unit, file, reg, value);
    }

    fn set_l3(
        &mut self,
        at: GroupPos,
        core: Core,
        unit: Component,
        file: RegType,
        reg: RegIndex,
        value: Sticks,
    ) {
        match self.l3.insert((at, core, unit, file, reg), value) {
            None => (),
            Some(already) if already == value => (),
            Some(already) => panic!(
                "{unit:?} {file:?}[{}] is initialised to two different addresses, {} and {}. A staging transfer \
                 addresses ONE memory view per side, so this is a defect in the assignment",
                reg.get(),
                already.get(),
                value.get()
            ),
        }
    }

    /// Whether this op's programs run on `core` — see [`Self::op_cores`].
    ///
    /// ⛔ AN OP WITH NO RECORDED COUNT RUNS NOWHERE, and that is not a tolerated default: every op of a group is
    /// walked by [`addresses`], so an absence here is a group whose ops did not all reach it.
    #[must_use]
    pub fn runs_on(&self, at: GroupPos, core: Core) -> bool {
        self.op_cores
            .get(&at)
            .unwrap_or_else(|| {
                panic!(
                    "the op at {at:?} has no core count, and `reginit::addresses` records one for every op of                      the group — so this tape and that walk disagree about which ops exist"
                )
            })
            .each()
            .any(|used| used == core)
    }

    /// Record which cores one op runs on.
    pub fn runs_on_cores(&mut self, at: GroupPos, cores: crate::islands::superdsc::Cores) {
        match self.op_cores.insert(at, cores) {
            None => (),
            Some(already) if already == cores => (),
            Some(already) => panic!(
                "the op at {at:?} runs on {} cores and also on {} — `numCoresUsed_` is one number per op",
                already.get(),
                cores.get()
            ),
        }
    }

    /// FOLD ANOTHER SET OF INITS IN, refusing a disagreement.
    ///
    /// ⭐ TWO SOURCES, ONE REGION: the placed bases of every memory allocation come from [`Self::addresses`] and the
    /// L3's address registers from the emitter's own pool. They name different files of different units, so a
    /// COLLISION would mean one register with two values — which is what the setters already refuse, and folding
    /// through them is what keeps that refusal in one place.
    pub fn merge(&mut self, other: Self) {
        // ⛔⛔⛔ DESTRUCTURED WITH NO `..`, SO A FIELD ADDED TO THIS TYPE BREAKS THIS FUNCTION.
        //
        // MEASURED: `dataspaces` was added and this kept compiling, because a `for … in other.addresses`
        // body names only what it reads. The emit builds a FRESH `RegisterInits` from `reginit::addresses`
        // and merges bridge 1's into it (`lib.rs:239-241`), so anything this does not carry is silently
        // dropped between the bridge that recorded it and the bridge that reads it — the model asked what
        // memory was made of and got an empty list for every op of the bake.
        //
        // ⭐ THE SAME DISCIPLINE `Machine::every_store_mut` USES, and for the same reason: a set that must be
        // complete is destructured completely rather than trusted to a reviewer.
        let Self {
            addresses,
            l3,
            dataspaces,
            op_cores,
        } = other;
        for (at, cores) in op_cores {
            self.runs_on_cores(at, cores);
        }
        for ((at, core, unit, reg), value) in addresses {
            self.set(at, core, unit, reg, value);
        }
        for ((at, core, unit, file, reg), value) in l3 {
            self.set_l3(at, core, unit, file, reg, value);
        }
        for ((at, core), spaces) in dataspaces {
            for space in spaces {
                self.lays_down(at, core, space);
            }
        }
    }

    fn set(&mut self, at: GroupPos, core: Core, unit: Component, reg: RegIndex, value: Bytes) {
        match self.addresses.insert((at, core, unit, reg), value) {
            None => (),
            Some(already) if already == value => (),
            Some(already) => panic!(
                "{unit:?} address register {} of the program at {at:?} is initialised to two different bases, \
                 {} and {}. An allocation is ONE memory view (`dsc2.cpp:2586-2612`), so this is a defect in the \
                 assignment",
                reg.get(),
                already.get(),
                value.get()
            ),
        }
    }
}

/// THE PLACED BASE OF EVERY MEMORY ALLOCATION, IN THE REGISTER THAT ADDRESSES IT.
///
/// ⭐⭐ THIS IS WHAT `0x7b1b` WAS: every LX/L0 access was bound to a placeholder address of zero, so every load in
/// the program addressed nothing. The base is `allocAllMem`'s — the size from `getBufferCapacityForNode`
/// (`dsc2.cpp:3977-4005`, ported as [`crate::alloc::allocation_bytes`]) placed by first fit into the memory's own
/// tracker (`ddcv1.cpp:132-340`, ported as [`crate::alloc::Placement`]).
///
/// ⛔ ONE TRACKER PER (UNIT, MEMORY), which is what the C++ keeps: `ddcv1.cpp:217` makes one per
/// `(component, core, corelet, row)`, and an allocation is placed in its own memory's space and nowhere else.
///
/// ⛔ AND THE VALUE IS SCALED BY THE UNIT'S OWN GRANULARITY. Every address immediate is
/// `value * element_size / 8 / GetAddressScale` (`ConstructProgIRHelper.cpp:3122-3127`; `GetAddressScale` is defined in `LowerSentientHelper.cpp:502`), and the placed base is
/// already in BYTES — so what remains is the division by [`address_scale`]: 1 for the LX and the L0LU, and
/// `numPTRows` for the L0SU, whose stores fan across the rows.
/// WHERE EVERY ALLOCATION LANDED — keyed by the statement that declares it.
///
/// ⭐ THE ACCESSING UNIT IS PART OF THE ANSWER, because the placement arena is per (op, unit, space): dxp keeps
/// one memory tracker per `(component, core, corelet, row)` (`ddc/ddcv1.cpp:217`). So "where is this buffer" is
/// only answerable together with "whose arena".
pub type Placements = std::collections::BTreeMap<
    (GroupPos, crate::generated::SsaName),
    (Component, Space, crate::addr::Span),
>;

/// READ ONE ALLOCATION STATEMENT — its tensor, its reserved size, and whose arena it lands in.
///
/// `None` for a statement that is not an allocation, for a memory that is not placed, and for a tensor this
/// op-bind does not list — the three the old loop `continue`d on, each with its own reason below.
fn allocation_of(
    group: &Group,
    accessors: &Accessors,
    op: &crate::islands::dataflow_ir::Stmt,
) -> Option<Allocation> {
    let (memory, buffers) = match op.stmt.attrs {
        Attrs::Allocate { memory, buffers }
        | Attrs::GetExternalDataTransferAllocation { memory, buffers } => (memory, buffers),
        _ => return None,
    };
    let space = memory_of(memory);
    // ⛔⛔⛔ A REGISTER FILE'S ALLOCATIONS CANNOT BE SIZED YET, AND THAT IS A MISSING INPUT.
    // `getBufferCapacityForNodePerDimCustomLocation` takes the COMPONENT, the ROW and the CORELET
    // (`dsc2.cpp:3755-3760`) and passes all three to `primaryDimToVal_st`, whose first two branches are
    //
    //     if (ptrowId >= 0 && rowSplit_.count(d) > 0)          val = rowSplit_.at(d).at(clId).at(ptrowId);
    //     else if ((peOrSfp == PE || peOrSfp == SFP) && peSfpSplit_.count(d) > 0)
    //                                                          val = peSfpSplit_.at(d).at(clId).at(peOrSfp);
    //                                                                          (`dims.cpp:651-706`)
    //
    // — per-dim maps keyed by corelet and by PT row that say how much of a dim THIS row or THIS unit holds.
    // Island 1 carries none of them, so the size computed here is the whole tile: measured on
    // granite-3.1-2b-fp8, `Kertensor` came out 16384 bytes against the XRF row's 8192 and `Inptensor` 8192
    // against the SFP LRF's 2048 — 199 refusals, three distinct shapes, every one of them the tile where a
    // share belongs. Inventing the divisor (rows? corelets? both?) is exactly the rule (b) line.
    //
    // ⛔ AND PLACING ONLY THE ONES WE CAN SIZE WOULD BE WORSE THAN PLACING NONE. `allocAllMem` fills
    // `nodeAndSize` with the tensor allocations FIRST and the constants after (`ddcv1.cpp:227-252`), so a
    // constant's base sits above the tensors' — skipping a tensor moves every constant below it. A partial
    // arena is not a partial answer, it is a wrong one.
    //
    // ⭐ WHAT IS ALREADY IN PLACE FOR WHEN THE SHARES ARRIVE: the five spaces, their arenas, and their
    // capacities from `regInfoPerUnit` (`sysdef.cpp:399-447`). What is missing is one function —
    // `primaryDimToVal_st`'s component view — and the datastage fields it reads.
    if space.is_register_file() {
        return None;
    }
    let Some(Operands::One(tensor_name)) = op.stmt.operands.first() else {
        panic!("a `ddl.allocate` names exactly what it allocates")
    };
    let spec = &group[op.op.index()];
    let program = crate::bridges::a_superdsc_to_dataflow_ir::program_of(spec);
    let tensor = match program.tensor_of(*tensor_name) {
        crate::generated::TemplateTensor::Input { compute, position } => {
            let of = spec.computes.at(compute).unwrap_or_else(|| {
                panic!(
                    "{tensor_name:?} belongs to compute {compute} of this program's bind set and the op states \
                     {} computes",
                    spec.computes.len()
                )
            });
            spec.tensor(*of.inputs.iter().nth(position).unwrap_or_else(|| {
                panic!("the bind lists {tensor_name:?} as input {position} of compute {compute} and it has fewer")
            }))
        }
        crate::generated::TemplateTensor::Output { compute, position: 0 } => {
            spec.tensor(
                spec.computes
                    .at(compute)
                    .unwrap_or_else(|| {
                        panic!(
                            "{tensor_name:?} is compute {compute}'s output and the op states {} computes",
                            spec.computes.len()
                        )
                    })
                    .output,
            )
        }
        // ⭐⭐ A TENSOR THIS BIND DOES NOT LIST HAS NO LABELED DATA-STAGE, AND THE C++ SKIPS IT:
        // `if (unit == NO_COMPONENT || ldsIdx < 0) continue` (`ddcv1.cpp:491`). A template declares its
        // allocations unconditionally while only some of its binds have that tensor.
        crate::generated::TemplateTensor::Unlisted { .. } => return None,
        // ⭐⭐⭐ AN ALLOCATION OVER A CONSTANT IS ONE STICK, AND NOTHING ELSE ENTERS INTO IT. The two buckets
        // are sized by different rules: a tensor allocation is `numBuffers * getBufferCapacityForNode(...)`
        // (`ddcv1.cpp:242-246`) while a constant's is the stick, flat —
        //
        //     for (auto& acand : allocMetadata.consIdAndAllocNode)
        //       nodeAndSize.emplace_back(acand.second, dscGlobal.sysDef.bytesPerStick);
        //                                                              (`ddcv1.cpp:249-252`)
        //
        // — because `getTensorOrExtConstIdx` gives a `ddl.define_constant` a `constIdx_` and no `ldsIdx_`
        // (`ddl_conversion.cpp:825-849`), so there is no labeled data-stage to take a layout from. Returning
        // here rather than falling through to `allocation_bytes` is the whole point: a constant has no layout,
        // and asking the tensor for one is what the panic below used to do.
        crate::generated::TemplateTensor::Constant => {
            let name = *op.stmt.results.first()?;
            return Some(Allocation {
                name,
                unit: addressing_unit(accessors, op, space, name)?,
                space,
                size: crate::alloc::Bytes::of(crate::isa::memory::BYTES_PER_STICK),
            });
        }
        other => panic!(
            "{tensor_name:?} is {other:?}, and a memory allocation's size comes from the TENSOR's layout dims \
             (`dsc2.cpp:3760-3808`) — an internal tensor's are stated by the template's own `ddl.layout`"
        ),
    };
    let size = crate::alloc::allocation_bytes(
        tensor.format,
        &tensor.layout,
        &tensor.stick,
        spec.stages
            .last()
            .unwrap_or_else(|| panic!("every op states at least the per-core tile level")),
        placement_memory(space),
        // ⭐ `ddcv1.cpp:241-246` PASSES FIVE ARGUMENTS, so `forceEvenNumSticks` takes its `false` default
        // (`designSpaceConfig.h:281-286`). The ring rule is the OTHER caller's — see [`RingPolarity`].
        crate::alloc::RingPolarity::Unconstrained,
        // ⭐ `includeGaps` DEFAULTS TRUE and this caller passes five arguments — see [`crate::alloc::Gaps`].
        crate::alloc::Gaps::Included,
    );
    // ⛔⛔ AN ALLOCATION RESERVES `numBuffers` OF ITS CAPACITY, AND THAT PRODUCT IS FORMED ONCE.
    // `nodeAndSize.emplace_back(allocNode, numBuffers * getBufferCapacityForNode(...))` (`ddcv1.cpp:244`) —
    // the count belongs to this call site, and `getBufferCapacityForNode` is what
    // [`crate::alloc::allocation_bytes`] ports. This site used to multiply by `buffers` a SECOND time,
    // after `allocation_bytes` had already applied it, so an `Exactly(n)` allocation reserved
    // `n^2 x capacity`; see [`crate::alloc::BufferCapacity`].
    let capacity = size;
    // ⛔ THE COMPUTED COUNT (`bridges::a_superdsc_to_dataflow_ir::lx_buffers_for`) IS NOT APPLIED HERE — see the note beside
    // `buffers:` in `bridges/1.rs`. Four attempts to switch it on regressed the bake every time.
    let size = capacity.reserved(buffers);
    // ⛔ AND THE FLOOR IS APPLIED TO THAT PRODUCT, NEVER TO A CLAMPED COPY OF IT. `mySize =
    // std::max(mySize, memCap)` (`ddcv1.cpp:327`) — this site used to pass
    // `size.min(memory.bytes())`, which makes the max return the floor unconditionally and leaves no
    // way for an over-large tiling to be refused.
    let size = match buffers {
        crate::alloc::Buffers::Circular => crate::alloc::circular_reservation(
            size,
            crate::alloc::CapacityFloor::of(placement_memory(space)),
        ),
        crate::alloc::Buffers::Exactly(_) => size,
    }
    .get();
    // ⛔ A REFUSAL THAT NAMES ONLY A BYTE COUNT CANNOT BE ACTED ON. Three groups refused "16384 bytes do not fit
    // in L0" without saying which tensor, which buffer count, or which of the divisions had applied.
    // ⛔ A CIRCULAR BUFFER TRIPS THIS TOO, and used not to only because the size was clamped to the
    // capacity before the floor was applied — see [`crate::alloc::BufferedSize`]. It is the reference's
    // own refusal: `checkAndAddDs` fails, `allocAllMem` returns false (`ddcv1.cpp:1335-1337`), and dxp
    // answers by shrinking a datastage. scratchy states its tiling, so here it is a stop.
    //
    // ⛔ AND IT NAMES THE TERMS, because "16384 does not fit in 8192" cannot be acted on: which dim is
    // oversized is the whole question, and it is `dim_elements` per layout dim that decides it.
    assert!(
        size.get() <= placement_memory(space).bytes(),
        "{tensor_name:?} needs {} bytes in {space:?} and {:?} holds {}. One buffer is {} bytes at \
         {:?} ({} B/element) and its buffers are {buffers:?}; per-dim elements {:?}, stick {:?}",
        size.get(),
        placement_memory(space),
        placement_memory(space).bytes(),
        capacity.get().get(),
        tensor.format,
        tensor.format.bytes(),
        tensor
            .layout
            .iter()
            .map(|&(dim, scale)| (
                dim,
                crate::alloc::dim_elements(
                    dim,
                    scale,
                    &tensor.stick,
                    spec.stages.last().unwrap_or_else(|| panic!(
                        "every op states at least the per-core tile level"
                    )),
                )
                .get()
            ))
            .collect::<Vec<_>>(),
        tensor.stick,
    );
    let name = *op.stmt.results.first()?;
    Some(Allocation {
        name,
        unit: addressing_unit(accessors, op, space, name)?,
        space,
        size,
    })
}

/// THE UNIT WHOSE ADDRESS UNITS THIS ALLOCATION IS COUNTED IN — [`addressed`]'s two cases, resolved.
///
/// 🔑 THE FIRST ACCESSOR IS ENOUGH IN THE SECOND CASE, because what this needs the unit for is the ARENA, and
/// the arena is the MEMORY's — every accessor of one allocation shares it. Which registers get the address is a
/// different question, and [`addresses`] asks it of every accessor.
fn addressing_unit(
    accessors: &Accessors,
    op: &crate::islands::dataflow_ir::Stmt,
    space: Space,
    name: crate::generated::SsaName,
) -> Option<Component> {
    match addressed(space) {
        Addressed::ByTheFileOwner(unit) => Some(unit),
        Addressed::ByWhoeverNamesIt => {
            Some(accessing_bindings(accessors, op.op, name).first()?.0)
        }
    }
}

/// PLACE EVERY ALLOCATION OF EVERY OP — the bump allocation `ddcv1.cpp` performs, and nothing else.
///
/// ⛔⛔ SPLIT OUT BECAUSE TWO PASSES NEED IT AND ONLY ONE OF THEM HAS AN `Assignment`. [`addresses`] turns a
/// placement into a register VALUE and needs bridge 2's binding to know which register; the L3's staging
/// expansion needs the same placement to know where to WRITE, and it runs in bridge 1, before any binding
/// exists. Computing it twice is how the reader's arena and the writer's part company — which is the failure
/// the comment at the end of this loop already warned about.
///
/// ⛔ AND IT PLACES AN ALLOCATION WHETHER OR NOT A REGISTER POINTS AT IT. dxp places every allocate node; an
/// allocation occupies its memory regardless. Skipping the unbound ones made the arena's offsets depend on
/// bridge 2's choices, so the same op placed its buffers differently depending on which names got registers.
#[tracing::instrument(skip_all, fields(stmts = expansion.len()))]
pub fn placements(group: &Group, expansion: &[crate::islands::dataflow_ir::Stmt]) -> Placements {
    // ⭐ ONE PASS, THEN LOOKUPS. See [`Accessors`] — this used to scan the whole expansion per op.
    let accessors = Accessors::of(expansion);
    let mut out: Placements = Placements::new();
    let mut placed: std::collections::BTreeMap<(GroupPos, Arena), Placement> =
        std::collections::BTreeMap::new();
    for op in expansion {
        let Some(found) = allocation_of(group, &accessors, op) else {
            continue;
        };
        let base = placed
            // ⛔ THE ARENA, NOT THE UNIT — see [`Arena`]. `found.unit` still says whose ADDRESS SCALE applies
            // and is carried into `out` below; it does not say which memory the allocation comes out of.
            .entry((op.op, arena_of(found.space)))
            .or_insert_with(|| Placement::new(placement_memory(found.space)))
            .place(found.size);
        out.insert((op.op, found.name), (found.unit, found.space, base));
    }
    out
}

/// ONE ALLOCATION STATEMENT, READ — its name, its size, and whose arena it lands in.
///
/// ⛔ THE TENSOR IS NOT A FIELD. It is what `size` is DERIVED FROM (its format, layout and stick), and keeping
/// it here as well would be a second copy of a fact nobody reads — the rule this crate states as "no field until
/// a bridge reads it".
struct Allocation {
    name: crate::generated::SsaName,
    unit: Component,
    space: Space,
    size: crate::alloc::Bytes,
}

/// THE PER-ACCESS ADVANCE OF A WALKING LOAD-STORE, IN THE UNIT'S OWN ADDRESS UNITS.
///
/// ⭐ THE UNITS ARE THE `ldtype` WIDTH IN BYTES FOR EVERY WALKING UNIT, and that is arithmetic rather
/// than coincidence: an LX address is in bytes (scale 1, `sysdef.cpp:504-525`) and its dense stride is
/// the access width; an L0 address is in units of `numPTRows` bytes and its rows are INTERLEAVED, so the
/// next line of one row sits `width × numPTRows` bytes on — which is `width` units again. The reference's
/// own `reg_initial` agrees: an L0LU walking 16-byte lines holds stride 16, an LX walking sticks holds
/// 128.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkStride(u64);

impl WalkStride {
    /// From the access width — [`crate::generated::LdType::bytes`], the one mint of the width.
    pub const fn of_access_units(units: u64) -> Self {
        Self(units)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// THE ADDRESS WALKS BRIDGE 1 DECIDED — one mint, written where the instruction is emitted and read
/// where its registers are seeded, so the opcode's trailing `U` and the register values cannot disagree.
///
/// A walking load-store advances `src0` by `src1 × burst` on completion and addresses `src0 + src1`
/// per dispatch (`memoryElement.cpp:1433-1450`, `:1786-1797`; ported as
/// `local_regs::apply_update` / `LoadStoreOperands::start`). So the walk needs THREE agreeing values:
/// the `U` opcode, `src1` seeded with the stride, and `src0` seeded ONE STRIDE BELOW the placed base —
/// which is the reference's own shape (`reg_initial`: an L0LU at base 0 holds `-16`; an LX at base 0
/// holds `lx_size - 128`, pre-wrapped by `getAddrWraparounded`).
#[derive(Debug, Default)]
pub struct AddressWalks {
    /// Each walking chain: HOW the (op, unit, binding) register advances.
    loads: std::collections::BTreeMap<
        (
            crate::expansion_tape::GroupPos,
            Component,
            crate::generated::SsaName,
        ),
        WalkMode,
    >,
}

/// HOW A WALKING LOAD'S REGISTER ADVANCES — the two spellings the reference emits.
///
/// ⛔ THE SEED DIFFERS WITH THE MODE AND MAY NOT BE SHARED. The updating form advances on COMPLETION
/// and addresses `src0 + src1`, so its seed sits ONE STRIDE BELOW base; the outer-add form advances at
/// the END of its loop's body (`insertCopyAndAddStmts`, `AgenToSentient/Helper.cpp:946-951`), so its
/// first trip must read AT base — seeding it rewound reads one stride early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkMode {
    /// The `U` opcode: every beat advances by `src1`, completion writes back `src0 += stride × burst`.
    UpdatingForm(WalkStride),
    /// A fixed load-store whose register a `MODLRFIMM` advances at the close of the loop that walks
    /// it — the reference's per-loop `add` statement.
    OuterAdd,
}

impl AddressWalks {
    /// Record one walking load-store — the binding whose register walks, and the load whose `src1`
    /// holds the stride.
    ///
    /// ⛔ TWO WALKS OF ONE REGISTER AT TWO STRIDES ARE REFUSED — the register is per
    /// (unit, allocation) and holds ONE seed, so two accesses of one buffer must walk it identically
    /// or the emission must learn per-load walk registers first.
    pub fn note(
        &mut self,
        at: crate::expansion_tape::GroupPos,
        unit: Component,
        binding: crate::generated::SsaName,
        mode: WalkMode,
    ) {
        match self.loads.insert((at, unit, binding), mode) {
            None => (),
            // ⭐ Every load of one shared chain re-notes the same mode; a DISAGREEMENT is two loads
            // walking one register two ways.
            Some(already) if already == mode => (),
            Some(already) => panic!(
                "{unit:?} walks {binding:?} of the op at {at:?} two ways ({already:?}, then {mode:?}) \
                 — one binding is one chain"
            ),
        }
    }
}

/// THE WRAP SPACE A WALKING REGISTER COUNTS IN, in its own units — what a seed one stride below base
/// wraps around, `getAddrWraparounded`'s modulus (`Utils.cpp:117-126`) seen from the units side.
fn wrap_space_units(unit: Component) -> u64 {
    match unit {
        // `val % getL0CapacityPerSlice()` — 1024 units for both L0 units (the scale and the row
        // interleave cancel).
        Component::L0lu | Component::L0su => l0_capacity_per_slice().get(),
        // An LX address wraps in the FULL 2 MiB space, not the allocator's 2031616: the LX unit's own
        // executePC DT_CHECKs `lx_size = memory_LX->size() * 128` to be a power of two and masks by it
        // (`memoryElement.cpp:1299-1303`), and `getAddrWraparounded` leaves LX addresses UNCHANGED
        // (`Utils.cpp:125`) — a below-base seed reaches the machine raw and its `(v + lx_size) & mask`
        // rescue brings the first dispatch back onto the base. Wrapping by the allocator bound instead
        // aimed the first dispatch at stick 15872 — a line nothing writes.
        Component::Lxlu | Component::Lxsu => sys_arch_spec::memory::LX_SIZE,
        Component::L3lu | Component::L3su | Component::Sfp | Component::Pe | Component::Pt => {
            panic!(
                "{unit:?} is not a unit whose LDST walks an LX/L0 address register; the L3's update form \
             advances LAR/EAR by the burst (`memoryElement.cpp:1158-1173`) through its own path"
            )
        }
    }
}

#[tracing::instrument(skip_all, fields(stmts = expansion.len()))]
pub fn addresses(
    group: &Group,
    expansion: &[crate::islands::dataflow_ir::Stmt],
    assignment: &Assignment,
    walks: &AddressWalks,
) -> RegisterInits {
    // ⭐ ONE PASS, THEN LOOKUPS — same reason as [`placements`]; see [`Accessors`].
    let accessors = Accessors::of(expansion);
    let mut out = RegisterInits::default();
    // The scaled base per (op, unit, binding), for the walking loads' own registers below.
    let mut walked_bases: std::collections::BTreeMap<
        (
            crate::expansion_tape::GroupPos,
            Component,
            crate::generated::SsaName,
        ),
        Bytes,
    > = std::collections::BTreeMap::new();
    let placements = placements(group, expansion);
    // ⛔⛔ ONE TRACKER PER PROGRAM, because LX is a scratchpad each program reuses from the start. dxp keeps its
    // memory tracker per DSC (`ddcv1.cpp:217`, one per `(component, core, corelet, row)`) and a DSC is one op here
    // (`d5a9509e`) — so accumulating a whole group's buffers in one tracker overflowed a 2 MiB part
    // (`1310720 bytes do not fit in Lx, which holds 2031616`) with buffers that never coexist.
    //
    // ⭐ THE SAME BOUNDARY AS THE REGISTERS AND THE PACKET: a program is an op. It owns its init packet
    // (`85d15048`), its register files (`c28b61f1`, `83d49944`) and its scratchpad.
    for op in expansion {
        // ⛔ EVERY OP, BEFORE THE ALLOCATION TEST. Which cores an op runs on is a fact about the OP and not about
        // whether this statement allocates anything, so recording it inside the `continue` below would leave
        // exactly the ops with no allocation unanswerable — and `runs_on` refuses an op it has no count for.
        out.runs_on_cores(op.op, group[op.op.index()].cores);
        let Some(found) = allocation_of(group, &accessors, op) else {
            continue;
        };
        // ⭐ THE PLACEMENT IS THE PLACER'S, so the L3 that fills this buffer and the unit that reads it cannot
        // part company — see [`placements`].
        let (_, _, base) = placements
            .get(&(op.op, found.name))
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "{:?} was read as an allocation here and not placed by `placements`, which walks the same \
                     statements",
                    found.name
                )
            });
        // ⛔⛔⛔ EVERY UNIT THAT ADDRESSES THIS ALLOCATION, NOT THE FIRST ONE NAMED — see [`Accessors`]. An LX
        // buffer is named by the `lxlu` that reads it AND the `lxsu` that writes it, and each addresses it
        // through its own local register file.
        //
        // ⛔⛔ THE REGISTER IS THE BINDING'S, NOT THE ALLOCATION'S. An allocation's own name binds to nothing
        // (`0106ff27`), so keying the lookup by it never matched and EVERY address register in the image stayed
        // zero. The `ddl.unit` that names the allocation declares the `%ssa` an instruction actually reads.
        for (unit, binding) in accessing_bindings(&accessors, op.op, found.name) {
            // 🔑 The scaled base is recorded for EVERY accessing binding, before the register test: a
            // WALKING load names its own per-load register instead of the binding's, so the binding may
            // hold no register at all while its base is still what the walk rewinds from.
            walked_bases.insert((op.op, unit, binding), scaled(base.at(), unit, found.space));
            // ⭐ NOT EVERY BINDING RESOLVES TO A REGISTER, and that is not a defect: `src0` on an LX load is
            // the load's RESULT (`ConstructProgIRHelper.cpp:3113`), which no `ddl.unit` binds and which needs
            // no address. A refusal here fired on exactly that and was wrong.
            let Some(PhysicalOperand::Register(reg)) =
                assignment.get(&crate::islands::progir::Operand::of(op.op, binding))
            else {
                continue;
            };
            // ⛔ AND ONE VALUE PER CORE, because a block is addressed to a core mask and the register lives in
            // that core's unit. The LX/L0 base is the same on every core; an L3 unit's HBM address is not,
            // which is why the key carries the core rather than the value being written once.
            // ⭐ THE SPAN'S ADDRESS, taken only here where THIS unit's own address SCALE is applied — the two
            // ends of one buffer count it differently, an LXLU in bytes and an L3 in sticks.
            let value = scaled(base.at(), unit, found.space);
            for core in cores_of(&group[op.op.index()]) {
                out.set(op.op, core, unit, *reg, value);
            }
        }
    }

    // ⛔⛔⛔ EVERY WALKING LOAD SEEDS **TWO** REGISTERS — its own base one stride below the placed
    // base, and its stride. The machine addresses `src0 + src1` and advances `src0` on completion
    // (`LoadStoreOperands::start`, `apply_update`), so the first dispatch lands ON the base exactly
    // when `src0` starts at `base - stride`, wrapped in the unit's own address space — the reference's
    // `reg_initial` states both shapes: `-16` for an L0 walk from 0, and an LX seed that the machine's
    // `(v + lx_size) & mask` rescue brings back onto the base. Before this table existed the stride
    // register seeded ZERO, so even the update-form bursts advanced by nothing: every beat re-read one
    // line, and the lazy span synthesis hid it.
    for ((at, unit, binding), mode) in &walks.loads {
        let base = walked_bases
            .get(&(*at, *unit, *binding))
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "{unit:?} walks {binding:?} of the op at {at:?}, and the allocation walk above \
                     never placed a base for that binding — the two walks disagree"
                )
            });
        let mut seeds: Vec<(crate::islands::progir::Operand, Bytes)> = Vec::new();
        match mode {
            // The updating form: seed one stride below base (wrapped), and the stride register.
            WalkMode::UpdatingForm(stride) => {
                let space = wrap_space_units(*unit);
                assert!(
                    stride.get() <= space,
                    "a stride of {} units cannot wrap a {space}-unit space",
                    stride.get()
                );
                seeds.push((
                    crate::islands::progir::Operand::load_walk(*at, *unit, *binding),
                    Bytes::of((base.get() + space - stride.get()) % space),
                ));
                seeds.push((
                    crate::islands::progir::Operand::walk_stride(
                        *at,
                        *unit,
                        u32::try_from(stride.get()).expect("a stride fits u32"),
                    ),
                    Bytes::of(stride.get()),
                ));
            }
            // The outer-add form: the register advances at the loop's close, so the first trip reads
            // AT base.
            WalkMode::OuterAdd => {
                seeds.push((
                    crate::islands::progir::Operand::load_walk(*at, *unit, *binding),
                    base,
                ));
            }
        }
        for (operand, value) in seeds {
            let Some(PhysicalOperand::Register(reg)) = assignment.get(&operand) else {
                panic!(
                    "{unit:?}'s walk of {binding:?} at {at:?} needs {operand:?} to hold a \
                     register — a walking load takes the register form by construction \
                     (`transfer_form`: an enclosing loop makes the address `InRegister`), so an \
                     unassigned slot here means the two walks disagree"
                )
            };
            for core in cores_of(&group[at.index()]) {
                out.set(*at, core, *unit, *reg, value);
            }
        }
    }

    out
}

/// The base as the unit's address register holds it — bytes divided by that unit's granularity.
fn scaled(base: Bytes, unit: Component, space: Space) -> Bytes {
    let scale = u64::from(address_scale::<crate::arch::this::ThisArch>(unit, space));
    assert!(
        base.get().is_multiple_of(scale),
        "a placed base of {} bytes is not a whole number of {unit:?} address units ({scale}); \
         `getRegImmVals` divides by the scale (`LowerSentientHelper.cpp:1146`) and cannot carry a remainder",
        base.get()
    );
    wrapped_around(Bytes::of(base.get() / scale), unit)
}

/// PORT OF `getAddrWraparounded` (`dcc/src/Conversion/SentientToProgIR/Utils.cpp:117-126`) — the last thing
/// `getRegImmVals` does to an address before it becomes a register init (`LowerSentientHelper.cpp:1147`).
///
/// ⛔⛔ AN L0 ADDRESS **WRAPS**, IT DOES NOT REFUSE. `return val % getL0CapacityPerSlice()` — and that quantity is
/// `l0Capacity / numPTRows` (`DccExtContext.cpp:376-379`), which on this arch is `64 * bytesPerStick / 8` = 1024.
/// So the L0's address space per slice is 1024 units and an address beyond it comes back round.
///
/// ⭐⭐ AND 1024 IS EXACTLY THE REGISTER'S WIDTH: an L0 address register is TEN BITS (`dip.cpp:3676-3679`), and
/// `2^10 == 1024`. So the wrap and the field are the same bound, which is why dxp never needs to refuse — and why
/// this crate's `L0LrfValue::of(u16::try_from(..).unwrap_or_else(panic))` was refusing a value the machine defines.
///
/// ⛔ EVERY OTHER UNIT IS RETURNED UNCHANGED (`:125`), and that is the fact that settles the L3: an EBR holds an
/// ABSOLUTE HBM address, not a segment-relative offset. Its 30 bits at a 128-byte granularity reach 128 GiB, which
/// is why no wrap is needed there.
///
/// ⛔ THE `PTXRF` ARM IS NOT REACHED HERE. `(val + numXrfPerPtRow) % numXrfPerPtRow` (`:120-123`) wraps an XRF
/// POINTER, and this function serves placed MEMORY bases — the XRF is a register file, which
/// [`placement_memory`] refuses. A `Component` cannot name `PTXRF` at all: it is a locale in dcc's enum, not one of
/// the nine components.
fn wrapped_around(value: Bytes, unit: Component) -> Bytes {
    match unit {
        // `is_any_of(comp, L0LUROW0, L0LU, L0SU)` (`:119`).
        Component::L0lu | Component::L0su => Bytes::of(value.get() % l0_capacity_per_slice().get()),
        Component::L3lu
        | Component::L3su
        | Component::Lxlu
        | Component::Lxsu
        | Component::Sfp
        | Component::Pe
        | Component::Pt => value,
    }
}

/// `DccExtContext::getL0CapacityPerSlice()` — `l0Capacity / numPTRows` (`DccExtContext.cpp:376-379`).
///
/// ⛔ `l0Capacity` IS THE TOTAL, and this is the per-ROW share. Taking the total as the per-row figure is what let
/// an L0 allocation be placed `numPTRows` times over.
fn l0_capacity_per_slice() -> Bytes {
    let total = crate::isa::memory::Memory::L0.bytes();
    let rows = u64::from(crate::isa::unit::NUM_PT_ROWS);
    assert!(
        rows > 0,
        "`DT_CHECK(dsc_global_->sysDef.numPTRows > 0)` (`DccExtContext.cpp:377`)"
    );
    Bytes::of(total / rows)
}

/// THE MEMORY A PLACEMENT TRACKS — [`Space`] is the SCALE's key and this is the CAPACITY's.
///
/// ⛔ TWO TYPES BECAUSE TWO QUESTIONS: `addressGranularityScalePerUnit` is keyed by (unit, space) and includes HBM
/// and the register files (`sysdef.cpp:504-525`), while a placement needs a capacity and only the three local
/// memories have one (`sysdef.cpp:181-190`).
/// WHICH TRACKER AN ALLOCATION LANDS IN — `MemTrackBundle::getTracker` (`mem_track_bundle.cpp:174-202`).
///
/// ⛔⛔⛔ THE TRACKER IS CHOSEN BY THE **MEMORY**, NOT BY THE UNIT THAT ADDRESSES IT, AND EACH MEMORY IS KEYED
/// DIFFERENTLY. `MemTrackBundle` spells the keys in its own field declarations
/// (`sys-arch-spec/memtracker/mem_track_bundle.h:17-32`):
///
/// ```text
/// std::map<int, DsTrackInMem> lxTrackPerCore;                       // key: coreid
/// std::map<int, std::map<int, DsTrackInMem>> l0Track;               // key: core, corelet
/// std::map<int, std::map<int, DsTrackInMem>> sfplrfTrack;           // key: core, corelet
/// std::map<int, std::map<int, std::map<int, DsTrackInMem>>> xrfTrack;  // core, corelet, row
/// ```
///
/// ⛔ THE ARENA USED TO BE KEYED BY `(op, COMPONENT, space)`, one rule applied to a table. The LX tracker is
/// keyed by the CORE ALONE, so every unit that addresses the LX shares it — but keyed per component, the
/// allocation an LXLU reads and the one an LXSU writes each bump from ZERO in the same 2 MiB of scratchpad.
///
/// ⛔ MEASURED, granite-3.1-2b fp8: op 45 laid its fp16 INPUT at LX 0..1 and its fp8 OUTPUT at LX 0..0 — two
/// tensors of one op at one address — and its `GCVT` read `Sen143Fp8` where the conversion it was given
/// (`imm = 4`, "fp16 pair to SEN143_FP8") demands fp16. 647 refusals a bake, every one of them this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Arena {
    /// `lxTrackPerCore` — the core's whole scratchpad, shared by every unit that addresses it.
    Lx,
    /// `l0Track` — per (core, corelet), shared by the L0's load and store units.
    L0,
    /// `l0ScaleTrack` — the SEN1P5 scale region, keyed like the L0's.
    L0Scale,
    /// `sfplrfTrack` — `getTracker(SFPLRF, core, corelet, row)` (`mem_track_bundle.cpp:174-202`).
    SfpLrf,
    /// `sfpStateTrack`.
    SfpState,
    /// `pelrfTrack`.
    PeLrf,
    /// `ptArfTrack`.
    PtArf,
    /// `xrfTrack`.
    PtXrf,
}

/// ⛔ AND IT IS DERIVED FROM THE SPACE ALONE, so no caller can reintroduce the component.
///
/// ⛔⛔⛔ THE REGISTER FILES HAVE TRACKERS TOO, AND LEAVING THEM OUT PLACED NONE OF THEM. `MemTrackBundle::
/// getTracker` answers for ELEVEN components — `PELRF`, `SFPLRF`, `L0`, `L0_SCALE`, `PTXRF`, `PTARF`, `LX`,
/// `PTIRF`, `SFPSTATE`, `PESTATE`, `LXLUSCALEREG` (`mem_track_bundle.cpp:174-202`) — and `allocAllMem` iterates
/// `metadata.newAllocations_` over every component that HAS allocations, register files included: its PTARF
/// pre-fill (`ddcv1.cpp:284-289`) reserves registers in one of those trackers by byte address. So a register
/// file is placed by the same first-fit allocator as the LX, and this returning a refusal for five of the eight
/// `memory=` spellings the templates write — 310 of their 461 allocations — meant none of those allocations had
/// a base at all.
const fn arena_of(space: Space) -> Arena {
    match space {
        Space::Lx => Arena::Lx,
        Space::L0 => Arena::L0,
        Space::L0Scale => Arena::L0Scale,
        Space::SfpLrf => Arena::SfpLrf,
        Space::SfpState => Arena::SfpState,
        Space::PeLrf => Arena::PeLrf,
        Space::PtArf => Arena::PtArf,
        Space::PtXrf => Arena::PtXrf,
        // ⛔ HBM IS THE FRONTEND'S AND THE IBR IS NOT AN ALLOCATION SPACE. scratchy reserves HBM itself, and
        // `L3LUIBR` holds indirect bases a transfer writes, not buffers a template allocates.
        Space::Hbm | Space::Ibr => panic!(
            "HBM is reserved by the frontend and the indirect-base file holds no allocations, so neither has \
             a tracker to bump (`mem_track_bundle.cpp:174-202`)"
        ),
    }
}

/// 🔒 THE LX ARENA DOES NOT DEPEND ON THE UNIT, WHICH IS THE WHOLE FIX — `lxTrackPerCore` is keyed by core.
/// Two components reaching the same arena is the property that stops an op's input and output being placed at
/// one address, so it is pinned rather than left to a reviewer.
const _: () = {
    assert!(matches!(arena_of(Space::Lx), Arena::Lx));
    assert!(!matches!(arena_of(Space::L0), Arena::Lx));
    assert!(!matches!(arena_of(Space::L0Scale), Arena::L0));
};

const fn placement_memory(space: Space) -> crate::isa::memory::Memory {
    use crate::isa::memory::Memory as Placed;
    match space {
        Space::Lx => Placed::Lx,
        Space::L0 => Placed::L0,
        Space::L0Scale => Placed::L0Scale,
        // The register files' capacities are `regInfoPerUnit`'s `maxNum * bitSize` — see `Memory::bytes`.
        Space::SfpLrf => Placed::SfpLrf,
        Space::SfpState => Placed::SfpState,
        Space::PeLrf => Placed::PeLrf,
        Space::PtArf => Placed::PtArf,
        Space::PtXrf => Placed::PtXrf,
        Space::Hbm | Space::Ibr => panic!(
            "HBM is reserved by the frontend and the indirect-base file holds no allocations, so neither has \
             a capacity to place into (`sysdef.cpp:181-190`)"
        ),
    }
}

/// WHOSE ADDRESS UNITS A PLACED ALLOCATION IS COUNTED IN — and the two cases are answered from different places.
///
/// ⛔⛔⛔ A REGISTER FILE'S ANSWER IS THE FILE ITSELF, NOT AN ACCESSING `ddl.unit`. `allocAllMem` keys its
/// tracker by `comp`, which is `allocNode->component_` — the `memory=` storage `processAllocation` recorded
/// (`ddl_conversion.cpp:806`, `:833`) — so `getTracker(SFPLRF, …)` is reached from the ALLOCATION and never from
/// whoever reads it (`ddcv1.cpp:183-217`). And a register-file allocation may have no `ddl.unit` naming it at
/// all: `broadcast_ops.ddl:349` hands `%zero_sfp_allocation` straight to a `ddl.opaque` as an
/// `input_output_registers` operand, which is how `finalizeOps` finds it (`ddcv1.cpp:3380-3392`).
///
/// ⭐ THE LX AND THE L0 KEEP THE OTHER RULE, and it is not interchangeable. Their arenas are per core (and per
/// corelet), shared by every unit that addresses them, and the ADDRESS SCALE differs between those units — 1 for
/// the LXLU, `numPTRows` for the L0SU — so there the accessing `ddl.unit` is the only thing that can answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Addressed {
    /// The compute unit that owns the file: `sfplrf`/`sfpstate` are the SFP's, `pelrf` the PE's, `ptarf`/`ptxrf`
    /// the PT's.
    ByTheFileOwner(Component),
    /// Whichever `ddl.unit` names the allocation — see [`accessing_bindings`].
    ///
    /// ⛔⛔ AND IT IS THE ACCESSING UNIT, NOT THE ALLOCATION'S OWN NAME. An LX/L0 allocation's own name binds to
    /// nothing (`0106ff27`); the `ddl.unit` that names it declares the `%ssa` an instruction actually reads, and
    /// that unit owns the tracker (`ddcv1.cpp:217`).
    ByWhoeverNamesIt,
}

const fn addressed(space: Space) -> Addressed {
    match space {
        Space::SfpLrf | Space::SfpState => Addressed::ByTheFileOwner(Component::Sfp),
        Space::PeLrf => Addressed::ByTheFileOwner(Component::Pe),
        Space::PtArf | Space::PtXrf => Addressed::ByTheFileOwner(Component::Pt),
        Space::Lx | Space::L0 | Space::L0Scale => Addressed::ByWhoeverNamesIt,
        Space::Hbm | Space::Ibr => panic!(
            "neither is placed, so neither has an address to count — see `arena_of`"
        ),
    }
}

/// WHICH SPACE A TEMPLATE'S `memory=` NAMES — total, because all eight spellings are placed.
///
/// ⛔⛔ IT USED TO RETURN `None` FOR THE FIVE REGISTER FILES, on the reading that a register file is
/// `bridges::sentient_to_progir::assign`'s to allocate. Those are two different allocations:
/// `SmartRegisterAllocation` assigns ProgIR values to registers, while a `ddl.allocate {memory="sfplrf"}` is a
/// BUFFER the ddc places into that file's own tracker before DataflowIR exists (`ddcv1.cpp:183-360`), and
/// `Ddc::finalizeOps` then names it `R<addr / bytesPerStick>` for an opaque body to read
/// (`ddcv1.cpp:3345-3392`). Returning `None` made `allocation_of` skip the statement entirely, so those buffers
/// had no base for anything to look up.
const fn memory_of(memory: crate::generated::Memory) -> Space {
    use crate::generated::Memory;
    match memory {
        Memory::Lx => Space::Lx,
        Memory::L0 => Space::L0,
        Memory::L0scale => Space::L0Scale,
        Memory::Sfplrf => Space::SfpLrf,
        Memory::Sfpstate => Space::SfpState,
        Memory::Pelrf => Space::PeLrf,
        Memory::Ptarf => Space::PtArf,
        Memory::Ptxrf => Space::PtXrf,
    }
}

/// WHICH UNIT ADDRESSES THIS ALLOCATION, AND THROUGH WHICH BINDING — the `ddl.unit` that names it, and the `%ssa`
/// that binding declares.
///
/// ⛔⛔ THE BINDING IS WHAT HOLDS THE ADDRESS, NOT THE ALLOCATION. An LX/L0 allocation's own name is not an
/// instruction operand — it binds to nothing (`0106ff27`) — so a register lookup keyed by it NEVER matches, and
/// every address register in the image stayed zero: measured on the card, 6912 SPR values and not one LRF, LAR or
/// EAR anywhere. `%src_inp_lxl0 = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", ...}`
/// (`bmm.ddl:216`) is the name an instruction reads, and it is the one the base belongs to.
/// WHICH `ddl.unit` NAMES AN ALLOCATION — an OPERAND-USE index, built ONCE.
///
/// ⛔⛔⛔ THIS WAS A FULL SCAN OF THE EXPANSION PER CALL, AND IT IS THE QUADRATIC THAT COST 457 SECONDS.
/// `accessing_binding` walked every statement of every op looking for the `ddl.unit` that names one allocation,
/// and both [`placements`] and [`addresses`] call it PER OP — so each was `O(ops x expansion)`. Measured in the
/// build-script trace: `a_superdsc_to_dataflow_ir::lower` totalled 457 s with a worst single group of 110 s, and the cost grows as
/// the square of a group's op count (300 ops -> 142 s, 512 -> 393 s, against `(512/300)^2 = 2.9`).
///
/// ⭐ THE KEY IS `(op, name)`, so this is the same walk with the search hoisted: one pass over the expansion,
/// then `O(log n)` per lookup.
///
/// ⛔⛔⛔ AND EVERY ACCESSOR IS KEPT, NOT THE FIRST. This held ONE `(unit, binding)` per name and said so —
/// "a name may be named by several `ddl.unit`s in principle; FIRST WINS" — which is not a property of the
/// reference, it is what the linear scan this replaced happened to do by returning on its first match.
///
/// ⛔ AN LX BUFFER IS NAMED BY BOTH ENDS. The `lxlu` that READS it and the `lxsu` that WRITES it each declare a
/// `ddl.unit` over the same allocation, and each addresses it through its OWN local register file. Keeping one
/// left the other unit with no address at all, `zero_used_registers` invented a zero for it, and zero is LX
/// LINE ZERO — another allocation's buffer.
///
/// ⛔ MEASURED, granite-3.1-2b fp8: op 45 places its fp16 input at LX 0..1 and its fp8 output at LX 2, and both
/// its LXLU and its LXSU ran with an address file of all zeros — so the SFP's fp8 result landed on LX line 0
/// while the LXLU was still reading it, and the `GCVT` at pc 1 read `Sen143Fp8` against an `imm = 4` that
/// demands fp16. 647 refusals a bake, every one of them this.
pub struct Accessors {
    /// ⛔⛔ THE **UNIT**, NOT ITS COMPONENT. Resolving the component while BUILDING the index called
    /// `component_of_endpoint` for every `ddl.unit` in the tape, including ones bound to `constant` — which has no
    /// component and panics by design ("a transfer with a constant at BOTH ends is sized by `CONSTANT_TO_CONSTANT`
    /// rather than by a unit"). The scan this replaced only ever asked about the ONE entry it matched, so hoisting
    /// the work made a refusal fire on statements nobody was asking about. The lookup resolves it lazily.
    by: std::collections::BTreeMap<
        (crate::expansion_tape::GroupPos, crate::generated::SsaName),
        Vec<(crate::generated::Unit, crate::generated::SsaName)>,
    >,
}

/// Record one accessor, keeping the order the expansion states and dropping an exact repeat.
fn push(
    by: &mut std::collections::BTreeMap<
        (crate::expansion_tape::GroupPos, crate::generated::SsaName),
        Vec<(crate::generated::Unit, crate::generated::SsaName)>,
    >,
    key: (crate::expansion_tape::GroupPos, crate::generated::SsaName),
    answer: (crate::generated::Unit, crate::generated::SsaName),
) {
    let held = by.entry(key).or_default();
    // 🔑 The SAME unit binding the same name twice is one accessor, not two registers.
    if !held.contains(&answer) {
        held.push(answer);
    }
}

impl Accessors {
    /// One pass: every `ddl.unit`, every name it mentions in its operands.
    pub fn of(expansion: &[crate::islands::dataflow_ir::Stmt]) -> Self {
        let mut by = std::collections::BTreeMap::new();
        for entry in expansion {
            let Attrs::Unit { unit, .. } = entry.stmt.attrs else {
                continue;
            };
            let Some(bound) = entry.stmt.results.first() else {
                continue;
            };
            let answer = (unit, *bound);
            for named in entry.stmt.operands {
                match named {
                    Operands::One(name) => {
                        push(&mut by, (entry.op, *name), answer);
                    }
                    Operands::List(names) => {
                        for name in *names {
                            push(&mut by, (entry.op, *name), answer);
                        }
                    }
                }
            }
        }
        Self { by }
    }
}

fn accessing_bindings(
    accessors: &Accessors,
    at: crate::expansion_tape::GroupPos,
    allocation: crate::generated::SsaName,
) -> Vec<(Component, crate::generated::SsaName)> {
    if let Some(held) = accessors.by.get(&(at, allocation))
        && !held.is_empty()
    {
        // ⛔⛔⛔ THE **PROGRAM-OWNING** COMPONENT, NOT THE LINK-ENDPOINT ONE, AND THE BAKE PROVED IT. This asks which
        // unit ADDRESSES the allocation, so that unit's address register is the one to initialise — and a `ptsouth`
        // bind addresses it from the PT's register file. `component_of_endpoint` answers the other question, "which
        // register file does a link endpoint name", where the three PT/SFP ports are absent from
        // `senCompToGenericComp` (`sys-arch-spec/arch_enums.cpp:124-211`) because a port owns no program; asking it
        // here refused `Ptsouth` on a legitimate binding.
        return held
            .iter()
            .map(|(unit, bound)| {
                (crate::bridges::a_superdsc_to_dataflow_ir::passes::dialect::component_of(*unit), *bound)
            })
            .collect();
    }
    panic!(
        "{allocation:?} is an LX/L0 allocation and no `ddl.unit` in this op names it, so nothing says which unit \
         addresses it — `setDataLocAndInfo` takes a unit's storage from its allocation (`ddl_conversion.cpp:870-935`)"
    )
}

/// EVERY CORE AN OP'S PROGRAM RUNS ON — `numCoresUsed_`.
///
/// ⛔⛔ THE OP'S OWN COUNT, NOT THE LENGTH OF A TENSOR'S FOLD. This counted the entries of the tensor's HBM fold
/// and called that the core set, so a CONSTANT fold — one address every core reads (`ddcv1.cpp:386-405`) — was
/// read as "core 0 alone", and an LX-resident tensor, which has no fold at all, was read the same way. Both
/// answers were the shape of the container rather than a fact about the op. [`crate::islands::superdsc::Cores`] is
/// that fact, and it now reaches island 1.
fn cores_of(op: &crate::islands::superdsc::SdscOp) -> Vec<Core> {
    op.cores.each().collect()
}

/// THE REGISTER INITS, WHICH ARE `REGISTER_MANAGEMENT`'s TO PRODUCE.
///
/// ⛔⛔⛔ `RegisterInitialization` AND `VectorRegisterInitialization` ARE PASSES 47 AND 48 of the ProgIR pipeline
/// (`dcc-standalone-main.cpp:661-694`), inside `sentient_to_progir::lower`. So the inits come OUT of that bridge;
/// computing them beside it is what let the previous port put a by-value allocator outside the pipeline that owns
/// the question, and cite the dcg BACK END for it.
///
/// ⛔ A NAMED REFUSAL RATHER THAN AN INLINE `todo!`, and the difference is not cosmetic. An inline
/// `let inits = todo!()` DIVERGES, so every statement after it is dead and the compiler reports each binding in
/// the driver's tail as unused — a dozen warnings that say nothing except "this is unfinished". A function that
/// refuses has a RETURN TYPE, so the tail stays live and the one missing thing has a name.
///
/// ⛔ AND DELIBERATELY NOT `RegisterInits::default()`. An empty init set is the `0x7b1b` failure: every program
/// addresses zero, the card runs garbage, and nothing faults. A refusal cannot be mistaken for an answer; an
/// empty map can.
///
/// # Panics
///
/// Always, until those two passes are ported.
#[must_use]
pub fn from_register_management() -> RegisterInits {
    todo!(
        "the register inits are RegisterInitialization's and VectorRegisterInitialization's — passes 47 and 48 \
         inside sentient_to_progir::lower. Wire them out of that bridge rather than recomputing them here"
    )
}
