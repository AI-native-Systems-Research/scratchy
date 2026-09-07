//! THE MACHINE, AS CONSTANTS — every size the emitter is allowed to know, carried by a trait rather
//! than read out of a config at run time.
//!
//! Ported from `sys-arch-spec/sysdef.h`'s `SenSystemDef` and the defaults its constructor sets
//! (`sys-arch-spec/sysdef.cpp:192-249`). Each constant below cites the line it came from, because a
//! size nobody can trace back to the model is a size someone guessed.
//!
//! ⭐ A TRAIT AND NOT A STRUCT, so the values reach the emitter as `A::CORES` — a constant the
//! compiler folds — rather than as a field someone could have written to. The generation is a TYPE:
//! `emit::<Dd2, _>` and `emit::<Sen1p5, _>` are different programs, and there is no value either of
//! them could carry that would turn one into the other.

use core::num::NonZeroU64;

/// WHICH INSTRUCTION-SET GENERATION — `IsaCoreGen` (`sys-arch-spec/sysdef.cpp:201,207,224`).
///
/// ⛔ ORDERED, AND THE ORDER IS LOAD-BEARING. The C++ branches on `coreArch <=
/// IsaCoreGen::RCUDD1A_ISA` in nine places, so this is a comparison and not just a label.
///
/// ⛔ TWO OF THE MODEL'S GENERATIONS, DELIBERATELY. `IsaCoreGen` also names MPW4, an older chip this
/// crate does not model; a generation no build can select cannot arrive here, and carrying it would
/// mean a match arm nothing can reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IsaGen {
    /// `RCUDD1A_ISA` — IBM's own `DEFAULT_ISA`. 8 PT rows, a 64-stick L0, no L0 scale region.
    Rcudd1a,
    /// `SEN1P5_ISA`. 4 PT rows, a 128-stick L0, a 16-stick L0 scale region, and cores tethered in
    /// pairs (`sysdef.cpp:201-205`).
    Sen1p5,
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The quantities. THREE UNITS OF LENGTH, AND THEY ARE NOT INTERCHANGEABLE.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// A count of BYTES.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

/// A count of 128-byte STICKS — how the model measures HBM, the LX program region and the XRF scale
/// start (`sysdef.h:104,105,237,238`).
///
/// ⛔ A SEPARATE TYPE FROM [`Bytes`] BECAUSE THE MODEL MIXES THEM IN ONE STRUCT. `hbmCapacity` is in
/// sticks and `lxCapacity` is in bytes, three fields apart (`sysdef.h:99-105`), and both are `uint64_t`
/// there. The conversion needs the stick size, which is [`Arch::BYTES_PER_STICK`], so it is a method on
/// the arch and not a `From`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sticks(pub u64);

/// A count of ELEMENTS.
///
/// ⛔⛔ THIS IS THE UNIT DATAFLOWIR ADDRESSES IN, AND IT IS NOT BYTES. "Addresses in Dataflow IR are
/// in *element* granularity, not bytes" — `Dataflow.td:250`, on `get_logical_memory_view`. scratchy's
/// own addresses are bytes, so every one of them crosses a conversion on the way into a view; making
/// that conversion a TYPE change is what stops it being applied twice, or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Elements(pub u64);

/// AN INDEX THAT CANNOT BE OUT OF BOUNDS — the bound is the type, not a check someone remembers.
///
/// ⭐ THE CONSTRUCTOR IS THE PROOF. [`Bounded::at`] takes the index as a const generic and asserts it
/// against `N` at monomorphisation, so an out-of-range core id is a COMPILE error. [`Bounded::checked`]
/// is for an index that is genuinely computed, and it hands back `None` rather than a wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bounded<const N: u32>(u32);

impl<const N: u32> Bounded<N> {
    /// A CONSTANT INDEX, CHECKED WHERE IT IS WRITTEN.
    #[must_use]
    pub const fn at<const I: u32>() -> Self {
        const { assert!(N > 0, "a bound of zero admits no index at all") }
        // ⛔⛔ `const { }`, NOT A BARE `assert!`. Both operands are const generics, so the comparison
        // is const-evaluable either way — but a bare `assert!` inside a `const fn` is only evaluated
        // at compile time when the CALL is in a const context. At a runtime call site it compiled to
        // a runtime panic, which is the one thing this constructor exists to rule out. Wrapping it
        // forces the evaluation regardless of where it is called from, so an out-of-range index is a
        // build error in every context.
        const { assert!(I < N, "index is out of bounds for this arch") }
        Bounded(I)
    }

    /// A COMPUTED INDEX, or `None` where it does not fit.
    #[must_use]
    pub const fn checked(index: u32) -> Option<Self> {
        if index < N {
            Some(Bounded(index))
        } else {
            None
        }
    }

    /// The index itself, for the one place it becomes text.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// The arch.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// THE TARGET, AS A TYPE. Every constant the emitter may read about the machine.
///
/// ⛔ NO DEFAULTS. A generation that differs from another in a value states its own; an associated
/// const with a default is a value a new arch can forget to set and still compile.
pub trait Arch {
    /// Which generation this is (`sysdef.cpp:195`).
    const GEN: IsaGen;

    /// `numCores` (`sysdef.h:95`). The C++ takes it from the `SENCORES` environment variable or 32
    /// (`dbo/src/Pipeline/Pipeline.cpp:76-80`); here it is stated by the type, which is the same fact
    /// without the environment.
    const CORES: u32;

    /// `numCoreletsPerCore` (`sysdef.h:96`), 2.
    const CORELETS_PER_CORE: u32;

    /// `tetheredCoreUnitSize` (`sysdef.cpp:200-205`) — 1 before SEN1P5, 2 from it, where cores are
    /// tethered in pairs and the core count must be a multiple of it.
    const TETHERED_CORE_UNIT: u32;

    /// `bytesPerStick` (`sysdef.cpp:206`), 128 on every generation.
    const BYTES_PER_STICK: NonZeroU64;

    /// `numSlicesPerStick` (`sysdef.cpp:229`), 8.
    const SLICES_PER_STICK: u32;

    /// `num_sfp_pe_slices_` (`sysdef.cpp:223`), 8.
    const SFP_PE_SLICES: u32;

    /// `numPTRows` (`sysdef.cpp:224`) — 8 on RCUDD1A, 4 on SEN1P5.
    ///
    /// ⛔⛔ THE PT IS NOT ONE INSTRUCTION STREAM BUT THIS MANY. `generatePTInitPacket` and
    /// `finalizeInitFlit` are called per row — `"pt_row" + p` (`dip.cpp:2124-2142`) — so naming the PT
    /// without its row loses which stream an instruction belongs to.
    const PT_ROWS: u32;

    /// `numPTCols` (`sysdef.cpp:225`), 8.
    const PT_COLS: u32;

    /// `numSimdPerPT` (`sysdef.cpp:227`), 8.
    const SIMD_PER_PT: u32;

    /// `l0Capacity` (`sysdef.cpp:207-208`) — 64 sticks on RCUDD1A, 128 from SEN1P5.
    const L0_CAPACITY: Sticks;

    /// `l0ScaleCapacity` (`sysdef.cpp:209-210`) — none on RCUDD1A, 16 sticks from SEN1P5.
    ///
    /// ⭐ ZERO IS A REAL VALUE HERE, NOT AN ABSENCE: RCUDD1A has no scale region, and `Sticks(0)` says
    /// exactly that. An `Option` would make every reader decide what `None` meant.
    const L0_SCALE_CAPACITY: Sticks;

    /// `lxCapacityFull` — the whole LX, program and debug regions included (`sysdef.h:101`).
    const LX_CAPACITY_FULL: Bytes;

    /// `lxCapacity` — what a program may allocate from: the full capacity less the 64 KiB the program
    /// and debug data reserve (`sysdef.cpp:211`).
    const LX_CAPACITY: Bytes;

    /// `xrfCapacity` (`sysdef.cpp:214`) — 64 KiB per PT array, i.e. per corelet.
    const XRF_CAPACITY: Bytes;

    /// `xrfScaleStart` (`sysdef.cpp:238`) — the register, in sticks, where weight scales begin.
    const XRF_SCALE_START: Sticks;

    /// `l3BurstSize` (`sysdef.cpp:215`), 32.
    const L3_BURST: u32;
    /// `lxBurstSize` (`sysdef.cpp:216`), 64.
    const LX_BURST: u32;
    /// `l0BurstSize` (`sysdef.cpp:217`), 64.
    const L0_BURST: u32;

    /// `maxNestedLoops` (`sysdef.cpp:218`), 16 — the deepest `affine.for` nest a unit's program may
    /// carry, and therefore the bound the emitter's loop depth is checked against.
    const MAX_NESTED_LOOPS: u32;

    /// `numLCCRRegisters` (`sysdef.cpp:228`), 16.
    const LCCR_REGISTERS: u32;

    /// `hbmNumSegments` (`sysdef.cpp:221`) — 8 on RCUDD1A, 32 from SEN1P5.
    const HBM_SEGMENTS: u32;

    /// `ebrGranurality` (`sysdef.cpp:236`) — the EBR/IBR granularity in sticks; 1 on RCUDD1A, 2 from
    /// SEN1P5. (The model's own spelling of the name is kept so a reader can find it.)
    const EBR_GRANULARITY: Sticks;

    /// `reservedProgLxAddr` (`sysdef.cpp:237`), stick `0x3f00` — where the program region starts.
    const RESERVED_PROG_LX_ADDR: Sticks;

    /// HOW WIDE AN L3 **EAR** IS — `regInfoPerUnit[L3LU][RegType::EAR].bitSize`, 21 on every arch
    /// (`sysdef.cpp:313-314`, and `:336-337` for `L3SU`).
    ///
    /// ⭐ THE EXTERNAL **ADDRESS** REGISTER, which is the MUTABLE half of an external address: what a
    /// transfer's address may reach without splitting is `2^bitSize` sticks of it
    /// (`MutableAddrSplitting.cpp:673-681`).
    ///
    /// ⛔ BOUNDED SO THE RANGE CANNOT OVERFLOW. The range the splitting pass computes from this is
    /// `2^bitSize * bytesPerStick * 8`, returned as an `int64_t`; a register wider than 52 bits would
    /// overflow the reference's own return type with a 128-byte stick. The bound is that type, not a
    /// guess, and [`Bounded::at`] checks it where the arch writes the number.
    const L3_EAR_BITS: Bounded<53>;

    /// HOW WIDE AN L3 **EBR** IS — `regInfoPerUnit[L3LU][RegType::EBR].bitSize`.
    ///
    /// ⛔⛔ THIS ONE IS ARCH-DEPENDENT WHERE THE EAR IS NOT: 30 bits on `coreArch <= RCUDD1A_ISA`
    /// (`sysdef.cpp:321-322`) and 32 from SEN1P5 (`sysdef.cpp:331-332`) — the same split the store
    /// half declares at `:344-345` and `:354-355`. The IMMUTABLE range a program may occupy is four
    /// times larger on SEN1P5 for that reason alone, so reading one arch's number on the other
    /// under- or over-states the space the splitting pass believes it has
    /// (`MutableAddrSplitting.cpp:683-692`).
    ///
    /// ⭐ AND THE SAME BRANCH DELETES THE `LBR`, which is the note the reference leaves beside it:
    /// *"NO LBR for sentient1.5"*.
    const L3_EBR_BITS: Bounded<53>;

    /// A COUNT OF STICKS IN BYTES. On the arch and not a `From`, because the factor is
    /// [`Arch::BYTES_PER_STICK`] and a free conversion would be one that had to assume it.
    #[must_use]
    fn sticks_to_bytes(sticks: Sticks) -> Bytes {
        Bytes(sticks.0 * Self::BYTES_PER_STICK.get())
    }

    /// EVERY (core, corelet) PAIR THIS ARCH HAS, in the order a program declares its units.
    #[must_use]
    fn corelets() -> impl Iterator<Item = (u32, u32)> {
        (0..Self::CORES).flat_map(|core| (0..Self::CORELETS_PER_CORE).map(move |cl| (core, cl)))
    }
}

/// THE DD2 GENERATION — `RCUDD1A_ISA`, IBM's own `DEFAULT_ISA` (`isa.hpp:30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dd2;

impl Arch for Dd2 {
    const GEN: IsaGen = IsaGen::Rcudd1a;
    const CORES: u32 = 32;
    const CORELETS_PER_CORE: u32 = 2;
    const TETHERED_CORE_UNIT: u32 = 1;
    const BYTES_PER_STICK: NonZeroU64 = NonZeroU64::new(128).expect("128 is not zero");
    const SLICES_PER_STICK: u32 = 8;
    const SFP_PE_SLICES: u32 = 8;
    const PT_ROWS: u32 = 8;
    const PT_COLS: u32 = 8;
    const SIMD_PER_PT: u32 = 8;
    const L0_CAPACITY: Sticks = Sticks(64);
    const L0_SCALE_CAPACITY: Sticks = Sticks(0);
    const LX_CAPACITY_FULL: Bytes = Bytes(2 * 1024 * 1024);
    const LX_CAPACITY: Bytes = Bytes(2 * 1024 * 1024 - 64 * 1024);
    const XRF_CAPACITY: Bytes = Bytes(64 * 1024);
    const XRF_SCALE_START: Sticks = Sticks(64);
    const L3_BURST: u32 = 32;
    const LX_BURST: u32 = 64;
    const L0_BURST: u32 = 64;
    const MAX_NESTED_LOOPS: u32 = 16;
    const LCCR_REGISTERS: u32 = 16;
    const HBM_SEGMENTS: u32 = 8;
    const EBR_GRANULARITY: Sticks = Sticks(1);
    const RESERVED_PROG_LX_ADDR: Sticks = Sticks(0x3f00);
    const L3_EAR_BITS: Bounded<53> = Bounded::at::<21>();
    const L3_EBR_BITS: Bounded<53> = Bounded::at::<30>();
}

/// THE SEN1P5 GENERATION — `SEN1P5_ISA`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sen1p5;

impl Arch for Sen1p5 {
    const GEN: IsaGen = IsaGen::Sen1p5;
    const CORES: u32 = 32;
    const CORELETS_PER_CORE: u32 = 2;
    const TETHERED_CORE_UNIT: u32 = 2;
    const BYTES_PER_STICK: NonZeroU64 = NonZeroU64::new(128).expect("128 is not zero");
    const SLICES_PER_STICK: u32 = 8;
    const SFP_PE_SLICES: u32 = 8;
    const PT_ROWS: u32 = 4;
    const PT_COLS: u32 = 8;
    const SIMD_PER_PT: u32 = 8;
    const L0_CAPACITY: Sticks = Sticks(128);
    const L0_SCALE_CAPACITY: Sticks = Sticks(16);
    const LX_CAPACITY_FULL: Bytes = Bytes(2 * 1024 * 1024);
    const LX_CAPACITY: Bytes = Bytes(2 * 1024 * 1024 - 64 * 1024);
    const XRF_CAPACITY: Bytes = Bytes(64 * 1024);
    const XRF_SCALE_START: Sticks = Sticks(64);
    const L3_BURST: u32 = 32;
    const LX_BURST: u32 = 64;
    const L0_BURST: u32 = 64;
    const MAX_NESTED_LOOPS: u32 = 16;
    const LCCR_REGISTERS: u32 = 16;
    const HBM_SEGMENTS: u32 = 32;
    const EBR_GRANULARITY: Sticks = Sticks(2);
    const RESERVED_PROG_LX_ADDR: Sticks = Sticks(0x3f00);
    const L3_EAR_BITS: Bounded<53> = Bounded::at::<21>();
    const L3_EBR_BITS: Bounded<53> = Bounded::at::<32>();
}

/// 🛑 EXACTLY ONE ARCH FEATURE. Both is a compiler built for one machine against another's tables;
/// neither leaves [`Target`] undefined, and a build that got that far would fail with a confusing
/// "cannot find type" instead of the reason.
#[cfg(all(feature = "arch-rcudd1a", feature = "arch-sen1p5"))]
compile_error!(
    "deeptools: `arch-rcudd1a` and `arch-sen1p5` are both enabled. The target arch is a build \
     constant — the PT row count, the L0 capacity and which .ddl template serves an op-func all \
     follow from it — so exactly one must be selected."
);
#[cfg(not(any(feature = "arch-rcudd1a", feature = "arch-sen1p5")))]
compile_error!(
    "deeptools: no arch feature is enabled. Select `arch-rcudd1a` (IBM's own default) or \
     `arch-sen1p5`; a build with neither has no machine to emit for."
);

/// ⭐⭐ THE ARCH THIS BUILD IS FOR, AS A TYPE. Everything downstream is generic over [`Arch`] and
/// instantiated at this one, so `Target::PT_ROWS` is a literal the compiler folds — not a field, not
/// an environment variable, and not a value any code path can vary.
#[cfg(feature = "arch-rcudd1a")]
pub type Target = Dd2;

/// The arch this build is for. See the `arch-rcudd1a` variant above.
#[cfg(all(feature = "arch-sen1p5", not(feature = "arch-rcudd1a")))]
pub type Target = Sen1p5;

/// WHAT THE MODEL STATES ABOUT ITSELF, AS CONST ASSERTIONS THE COMPILER EVALUATES.
///
/// Each of these is a relation `sysdef.cpp` establishes and every later stage assumes. A property
/// that needs a test harness to observe is being observed one compilation too late.
const _: () = {
    // `DT_CHECK_MSG(numCores % tetheredCoreUnitSize == 0, "In Sen1p5, num cores can only be multiples
    // of 2")` — `sysdef.cpp:203-204`, checked at run time there and at compile time here.
    assert!(Sen1p5::CORES % Sen1p5::TETHERED_CORE_UNIT == 0);
    // ⛔ AND NOT THE SAME FOR DD2, WHICH WOULD BE `% 1`. Its cores are not tethered
    // (`TETHERED_CORE_UNIT` is 1), so the constraint is vacuous there — clippy's `modulo_one` says
    // so, and it is right: an assertion that cannot fail states nothing. What DD2 owes instead is
    // that it is untethered at all, which is checkable.
    assert!(Dd2::TETHERED_CORE_UNIT == 1);

    // `lxCapacity = lxCap - 64 * 1024` (`sysdef.cpp:211`): the allocatable LX is the full one less the
    // program and debug region, and it is the SAME 64 KiB on both generations.
    assert!(Dd2::LX_CAPACITY_FULL.0 - Dd2::LX_CAPACITY.0 == 64 * 1024);
    assert!(Sen1p5::LX_CAPACITY_FULL.0 - Sen1p5::LX_CAPACITY.0 == 64 * 1024);

    // The generations differ in exactly the places `sysdef.cpp` branches on `coreArch`. Stated as
    // inequalities so that making the two impls identical — the easy way to "fix" a build — breaks it.
    assert!(Dd2::PT_ROWS != Sen1p5::PT_ROWS);
    assert!(Dd2::L0_CAPACITY.0 != Sen1p5::L0_CAPACITY.0);
    assert!(Dd2::HBM_SEGMENTS != Sen1p5::HBM_SEGMENTS);
    assert!(Dd2::L0_SCALE_CAPACITY.0 == 0 && Sen1p5::L0_SCALE_CAPACITY.0 > 0);
};
