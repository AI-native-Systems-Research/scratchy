// SPDX-License-Identifier: Apache-2.0
//! HOW MUCH EACH MEMORY HOLDS — `SenSystemDef`'s capacities (`sys-arch-spec/sysdef.cpp:181-190`).
//!
//! An allocation needs a PLACE, and a place needs a bound. These are the bounds, ported from the one block that
//! states them, so nothing downstream guesses how much L0 or LX there is.
//!
//! ⛔ AND EVERY ONE OF THEM IS ARCH-KEYED, which is why they are `const` under a cargo feature rather than runtime
//! values: `coreArch <= RCUDD1A_ISA ? … : …` appears on three of the five, and SEN1P5 doubles the L0 while giving it
//! a scale region RCUDD1A does not have at all.

/// `bytesPerStick = 128` (`sysdef.cpp:181`).
///
/// ⭐ THE SAME 128 THE SHUFFLE ALGEBRA CALLS A SLICE, in different units: `deeptools`'s `shuffle::BITS_PER_SLICE` is 128
/// BITS of lane selection within a stick, while this is 128 BYTES of storage. Both are stated as 128 by their own
/// source and they are not the same quantity, so neither is derived from the other.
pub const BYTES_PER_STICK: u64 = 128;

/// WHICH MEMORY AN ALLOCATION LIVES IN — the ones `allocAllMem` walks (`ddcv1.cpp:183-200`).
///
/// ⛔ `L0_SCALE` IS ITS OWN MEMORY AND NOT PART OF L0. `sysdef.cpp:184-185` gives it a separate capacity, ZERO on
/// RCUDD1A, so on that arch an allocation placed there has nowhere to go — a refusal rather than a share of L0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Memory {
    /// Per-core L0.
    L0,
    /// The L0 scale region — SEN1P5 only.
    L0Scale,
    /// The LX scratchpad.
    Lx,
    /// The SFP's local register file — `regInfoPerUnit[SFP][LRF]` (`sysdef.cpp:399-408`).
    SfpLrf,
    /// The SFP's state file — `regInfoPerUnit[SFP][STATE]` (`sysdef.cpp:401`, `:406`).
    SfpState,
    /// The PE's local register file — `regInfoPerUnit[PE][LRF]` (`sysdef.cpp:415-424`).
    PeLrf,
    /// The PT's accumulator file — `regInfoPerUnit[PT][ARF]` (`sysdef.cpp:433-434`).
    PtArf,
    /// ONE PT ROW's weight file — `regInfoPerUnit[PT][XRF]` (`sysdef.cpp:437-442`).
    ///
    /// ⛔ PER ROW, NOT THE WHOLE ARRAY. `getTracker(PTXRF, core, corelet, row)` keys by row
    /// (`mem_track_bundle.cpp:180`), so this is a row's share and [`XRF_BYTES`] is the array's total. The pin
    /// below states the product.
    PtXrf,
}

/// HOW MANY REGISTERS ONE FILE HOLDS — `regInfoPerUnit[unit][type].maxNum` (`sysdef.cpp:399-447`).
///
/// ⭐ SPLIT OUT FROM [`Memory::bytes`] BECAUSE THE REGISTER COUNT IS WHAT AN INDEX IS BOUND BY, and a byte
/// capacity divided back by 128 at a consumer would be a second derivation of it. `allocAllMem` places into the
/// byte space (`ddcv1.cpp:316-360`) while `R<n>` names the register, so both are wanted and only one is stated.
const fn registers(memory: Memory) -> u64 {
    match memory {
        // `{16, 128, 128}` up to RCUDD1A and `{32, 128, 128}` above (`sysdef.cpp:399-408`).
        Memory::SfpLrf | Memory::PeLrf => match cfg!(feature = "arch-sen1p5") {
            true => 32,
            false => 16,
        },
        // `{1, 128, -1}` up to RCUDD1A and `{4, 128, -1}` above (`sysdef.cpp:401`, `:406`).
        Memory::SfpState => match cfg!(feature = "arch-sen1p5") {
            true => 4,
            false => 1,
        },
        // ⛔ `ARF` AND NOT `LRF`, WHICH IS AN ARCH CHOICE THIS BUILD HAS ALREADY MADE. `regInfoPerUnit[PT]` is
        // keyed `LRF` below RCUDD1A and `ARF` at or above it (`sysdef.cpp:430-435`), and the tracker picks the
        // same way (`mem_track_bundle.cpp:150-153`). Both arches this crate builds are at or above, so the file
        // is the ARF and its count is 4 on each.
        Memory::PtArf => 4,
        // `{64, 128, -1}` up to RCUDD1A and `{128, 128, -1}` above (`sysdef.cpp:437-442`).
        Memory::PtXrf => match cfg!(feature = "arch-sen1p5") {
            true => 128,
            false => 64,
        },
        // ⛔ NOT A REGISTER FILE, and answering 0 would let `bytes` compute a capacity for one. The three local
        // memories state their capacity directly.
        Memory::L0 | Memory::L0Scale | Memory::Lx => panic!(
            "only a register file has a register count; the local memories' capacities are              `sysdef.cpp:181-190`"
        ),
    }
}

impl Memory {
    /// How many BYTES this memory holds.
    ///
    /// ⛔ THE LX FIGURE IS NOT ITS SIZE. `lxCapacity = lxCap - 64 * 1024` (`sysdef.cpp:186`), commented "for
    /// program+debug data" — so 64 KiB of the part is reserved, and an allocator using the full size would place its
    /// last allocations on top of the program. `lxCap` itself defaults to 2 MiB (`sysdef.cpp:557`).
    ///
    /// ⛔ AND THE L0 DOUBLES ON SEN1P5: `64 * bytesPerStick` up to RCUDD1A and `128 * bytesPerStick` above it
    /// (`sysdef.cpp:182-183`).
    pub const fn bytes(self) -> u64 {
        match self {
            Self::L0 => match cfg!(feature = "arch-sen1p5") {
                true => 128 * BYTES_PER_STICK,
                false => 64 * BYTES_PER_STICK,
            },
            // ⛔ ZERO ON RCUDD1A, a real state and not a missing number: the arch has no scale region.
            Self::L0Scale => match cfg!(feature = "arch-sen1p5") {
                true => 16 * BYTES_PER_STICK,
                false => 0,
            },
            Self::Lx => LX_SIZE - LX_RESERVED,
            // ⭐ A REGISTER FILE'S CAPACITY IS `maxNum * bitSize`, AND `bitSize` IS 128 BYTES — the stick.
            // `initMemTrack(name, reg.maxNum * reg.bitSize, numSteps, reg.bitSize)`
            // (`mem_track_bundle.cpp:156-158`) sizes the tracker, and `allocAllMem` addresses it in BYTES:
            // its PTARF pre-fill reserves `4 * bytesPerStick` (`ddcv1.cpp:284-289`). So one register is one
            // stick, which is why `getStartAddress` divides a placed address by `bytesPerStick` to name it
            // (`ddcv1.cpp:3359-3366`).
            Self::SfpLrf | Self::SfpState | Self::PeLrf | Self::PtArf | Self::PtXrf => {
                registers(self) * BYTES_PER_STICK
            }
        }
    }

    /// HOW MANY REGISTERS THIS FILE HOLDS — see [`registers`], which every index is bound by.
    pub const fn registers(self) -> u64 {
        registers(self)
    }

    /// Whether this memory exists on the arch being built.
    ///
    /// ⭐ A PREDICATE RATHER THAN A ZERO TO COMPARE AGAINST. "This memory holds nothing" and "this memory is not on
    /// this part" are the same number and different facts, and an allocator checking only the capacity would report
    /// an overflow where the answer is that the region does not exist.
    pub const fn exists(self) -> bool {
        match self {
            Self::L0 | Self::Lx => true,
            Self::L0Scale => cfg!(feature = "arch-sen1p5"),
            // Every register file exists on both arches this crate builds; only their widths differ.
            Self::SfpLrf | Self::SfpState | Self::PeLrf | Self::PtArf | Self::PtXrf => true,
        }
    }
}

/// `lxSize`'s default — 2 MiB (`sysdef.cpp:557`).
///
/// ⛔ THE FULL PART, NOT THE PLACEMENT CAPACITY. `Memory::Lx.bytes()` is `lxCapacity = lxCap - 64 * 1024`
/// (`sysdef.cpp:186`), which holds back program and debug space; the LBR partitions divide `lxCapacityFull`
/// (`dcgbeCodegen.cpp:77-78`), so the two are different numbers and a caller must say which it means.
pub const LX_SIZE: u64 = 2 * 1024 * 1024;

/// What `lxCapacity` holds back "for program+debug data" (`sysdef.cpp:186`).
const LX_RESERVED: u64 = 64 * 1024;

/// `xrfCapacity = 64 * 1024` (`sysdef.cpp:188`).
///
/// ⭐ THE COMMENT STATES THE TWO SHAPES THAT REACH IT: "64*8*128 for <=dd2, 128*4*128 for >= sen1p5". Both are 65536,
/// so the capacity is arch-independent while its geometry is not.
pub const XRF_BYTES: u64 = 64 * 1024;

/// `l3BurstSize = 32` (`sysdef.cpp:189`).
pub const L3_BURST_SIZE: u64 = 32;

/// HOW WIDE ONE HBM VIRTUAL SEGMENT IS, IN STICKS — `hbmVirtualSegmentSize = 1 << 27` (`sysdef.cpp:194-195`).
///
/// ⭐ A SEGMENT INDEX TIMES THIS IS A STICK ADDRESS, which is how a device pointer is formed: `dxpSegId *
/// hbmVirtualSegmentSize + stAddr / bytesPerStick`, then `* 128` for bytes (`dxp.cpp:504-506`, `:606-607`).
///
/// ⛔ IT IS QUOTED IN STICKS AND THE ADDRESSES BUILT FROM IT ARE IN BYTES. `sysdef.h:104` says so in a comment —
/// `uint64_t hbmVirtualSegmentSize;  // stick (128 B)` — and `createTrackers` multiplies by `bytesPerStick` to get
/// the tracker's capacity (`dxp.cpp:512-513`). The product is 16 GiB, the segment stride scratchy states from the
/// other side (torch-spyre `constants.py:55`).
pub const HBM_SEGMENT_STICKS: u64 = 1 << 27;

/// `hbmNumSegments` — 8 up to RCUDD1A, 32 above (`sysdef.cpp:196`).
pub const HBM_SEGMENTS: u64 = match cfg!(feature = "arch-sen1p5") {
    true => 32,
    false => 8,
};

// ─────────────────────────────────────────────────────────────────────────────
// WHAT THESE CAPACITIES PIN
// ─────────────────────────────────────────────────────────────────────────────

/// ⛔ THE LX RESERVATION IS REAL AND IT IS NOT ROUNDING. Using `lxSize` would hand out 64 KiB the program occupies.
const _: () = assert!(Memory::Lx.bytes() == 2 * 1024 * 1024 - 64 * 1024);
const _: () = assert!(Memory::Lx.bytes() < LX_SIZE);

/// ⛔ AND THE L0's SIZE IS THE ARCH'S. Both arms are stated in one expression, so a build for either gets a number
/// the other would refuse.
const _: () = assert!(
    Memory::L0.bytes()
        == if cfg!(feature = "arch-sen1p5") {
            16384
        } else {
            8192
        }
);

/// ⛔ AND THE SCALE REGION IS ABSENT ON RCUDD1A rather than empty: `exists` says so, and its capacity agrees.
const _: () = assert!(Memory::L0Scale.exists() == cfg!(feature = "arch-sen1p5"));
const _: () = assert!(Memory::L0Scale.exists() || Memory::L0Scale.bytes() == 0);

/// ⛔ THE PROGRAM SEGMENT MUST EXIST ON THE ARCH BEING BUILT. `dxpSegId` is 7 (`dxp.h:114-117`), so a part with
/// fewer segments than that has nowhere to put a program image.
const _: () = assert!(HBM_SEGMENTS > 7);

/// The stick is the unit every capacity above is quoted in, so it divides all three.
const _: () = assert!(Memory::L0.bytes().is_multiple_of(BYTES_PER_STICK));
const _: () = assert!(Memory::Lx.bytes().is_multiple_of(BYTES_PER_STICK));

/// ⛔ EVERY REGISTER FILE IS A WHOLE NUMBER OF STICKS, because a register IS a stick — the property
/// `getStartAddress`'s division relies on (`ddcv1.cpp:3359-3366`).
const _: () = {
    let files = [
        Memory::SfpLrf,
        Memory::SfpState,
        Memory::PeLrf,
        Memory::PtArf,
        Memory::PtXrf,
    ];
    let mut i = 0;
    while i < files.len() {
        assert!(files[i].bytes().is_multiple_of(BYTES_PER_STICK));
        assert!(files[i].bytes() / BYTES_PER_STICK == files[i].registers());
        i += 1;
    }
};
