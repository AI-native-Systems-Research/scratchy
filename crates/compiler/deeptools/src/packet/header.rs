// SPDX-License-Identifier: Apache-2.0
//! THE PER-UNIT HEADER SLICE — which unit and cores a block targets, and how many flits follow it.
//!
//! `Dip::finalizeHeaderFlit` (`dip.cpp:2136-2191`). Every field here is either STATED by the emitter or FIXED by
//! the path this port targets; nothing is computed, defaulted or recovered.
//!
//! # Which of dxp's two disagreeing layouts is authoritative
//!
//! `HeaderInitSlice::convertToSixteenByteStream` (`initpacket.cpp:214-239`) is, because it states the bit ranges
//! outright rather than through a bitfield's declaration order:
//!
//! ```cpp
//! putU32OneHotInBitRange(uvalue, 96, 127, targetCores);
//! putU32BitRange(uvalue, 85, 95, targetUnitBits);
//! putU32BitRange(uvalue, 78, 84, numIbuffFlits);
//! putU32BitRange(uvalue, 73, 77, numLrfFilts);
//! putU32BitRange(uvalue, 64, 69, numIbuffCorrectionFlits);
//! ```
//!
//! i.e. correction 64-69, bit 70 reserved, LCCR 71-72, LRF 73-77, IBUFF 78-84, units 85-95, cores 96-127. The
//! GENERATED shifts agree: `dip.cpp:4519-4524` packs `pktIbuffCorrection_flit | pktLCCR_flit << 7 |
//! pktLRF_flit << 9 | pktIbuff_flit << 14 | target_unit_mask << 21` into word 2, with `target_core_mask` in
//! word 3.
//!
//! The header slice's field layout (`initpacket.cpp:223-237`) — `lrf_idx:64, reserved:1, pktLCCR_flit:2,
//! pktLRF_flit:5, pktIbuffCorrection_flit:6, pktIbuff_flit:7, target_unit_mask:11, target_core_mask:32` — which
//! puts correction at 72-77 and LRF at 67-71, disagreeing with the generated shifts over bits 64-77 while
//! agreeing above them. So the struct is NOT the spec. The previous port justified the choice as "verified
//! against the golden": true, but a golden shows what one bundle contained, not which of two layouts the
//! compiler follows. The quoted function is that answer.
//!
//! # ⛔ WHAT OF dxp's GENERALITY IS ABSENT, AND WHY IT CANNOT ARISE
//!
//! - **No fallible `header(..) -> Result`.** The previous port returned a `HeaderRefusal::FieldOverflow` from
//!   every field conversion. Each count now CARRIES its field width as a const generic, so an over-wide value is
//!   not a value: there is no overflow left to report and no `Result` to propagate.
//! - **No `numCoreletsUsed` door.** dxp counts corelets out of a descriptor (`ddcv1.cpp:207-209`) and so can
//!   produce corelet 2. The emitter here STATES [`Corelets`], and there is no way to name a third corelet, so the
//!   count and its refusal are both gone rather than checked.
//! - **No correction-flit parameter.** Fixed at zero by `dip.cpp:2667-2668` — see [`Header::CORRECTION_FLITS`].
//! - **No unit-from-mask decoding.** `dip.cpp:844-870` reconstructs a component from a mask because a reader is
//!   handed bytes. We are the writer and the unit is stated, so only the encoding direction exists here.

use crate::islands::init_packet::Slice;

/// A value that FITS A BITFIELD OF `BITS` BITS.
///
/// ⛔⛔ THIS EXISTS BECAUSE OVERFLOWING A HEADER FIELD CORRUPTS ITS NEIGHBOUR. [`Header::slice`] packs several
/// counts into one word by SHIFTING, and each has a narrow width (`initpacket.cpp:223-237` writes each by bit range:
/// `pktLRF_flit:5`, `pktIbuffCorrection_flit:6`, `pktIbuff_flit:7`, `target_unit_mask:11`). A value wider than
/// its field does not saturate or wrap in place — its high bits land in the FIELD ABOVE, so an over-long IBUFF
/// count would corrupt `target_unit_mask` and deliver the block to the wrong units.
///
/// ⭐ A MACHINE TYPE IS THE WRONG BOUND HERE. `u8` admits 0..=255 where a 7-bit field admits 0..=127, and
/// `u8::try_from` succeeding says only that the CARRIER fits. The domain is the field WIDTH, so the width is a
/// const generic and the door a raw integer comes through is exactly two functions wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Bits<const BITS: u32>(u32);

impl<const BITS: u32> Bits<BITS> {
    /// The largest value this field can hold.
    pub const MAX: u32 = if BITS >= 32 {
        u32::MAX
    } else {
        (1u32 << BITS) - 1
    };

    /// `None` when the value does not fit — see the type doc for why that is not pedantry.
    ///
    /// A zero-width field holds nothing, and a width past 32 is not a field of a 32-bit word: both are refused
    /// rather than shifted by a distance the carrier does not have. This form exists so the invariants below can
    /// pin the REFUSAL; emission uses [`Self::exactly`].
    pub const fn new(value: u32) -> Option<Self> {
        if BITS == 0 || BITS > 32 || value > Self::MAX {
            None
        } else {
            Some(Self(value))
        }
    }

    /// The value, ASSERTED to fit — the form for a value this crate's caller STATES.
    ///
    /// ⭐ NOT AN `expect`, AND NOT A RUNTIME REFUSAL. This crate runs inside scratchy's `#[forward]` expansion,
    /// so the `panic!` IS a compile error: a stated constant that does not fit its field stops the build at the
    /// site that stated it, where the reader can see both the value and the width.
    pub const fn exactly(value: u32) -> Self {
        match Self::new(value) {
            Some(bits) => bits,
            None => panic!(
                "a stated value does not fit its header field; its high bits would land in the field ABOVE it \
                 (initpacket.cpp:214-239)"
            ),
        }
    }

    /// The value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A unit's id bit in the HEADER's `targetUnitBits`.
///
/// `dip.h:38-48`, verbatim:
///
/// ```cpp
/// #define L3LUID     0x200
/// #define L3SUID     0x400
/// #define CORELET1ID 0x100
/// #define CORELET0ID 0x80
/// #define LXSUID     0x40
/// #define LXLUID     0x20
/// #define L0SUID     0x10
/// #define L0LUID     0x8
/// #define SFPID      0x4
/// #define PEID       0x2
/// #define PTID       0x1
/// ```
///
/// A newtype, not a `u16`: it is OR-ed with a corelet bit and written into an 11-bit field, and a unit id that
/// shares a type with the COMBINED mask is one that can be written where the mask belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnitId(u16);

impl UnitId {
    /// The id's bits, for OR-ing into a [`TargetMask`] and for checking against `dip.h`'s defines.
    pub const fn bits(self) -> u16 {
        self.0
    }
}

/// A unit that LIVES ON A CORELET, and therefore MUST name one in the header.
///
/// ⭐ THE "MUST" IS IN THE TYPE, NOT IN A CHECK. For `slice_id < 6` the reader accepts only
/// `CORELET0|CORELET1|unitID`, `CORELET0|unitID` or `CORELET1|unitID`, and otherwise raises "Header with an
/// invalid unit_mask" (`dip.cpp:864`) — a bare compute id decodes to NO UNIT AT ALL. The previous port said that
/// with a `TargetMask::new(unit, Option<Corelets>)` returning `None`, and an earlier version of that function
/// ALLOWED the corelet-less case: wrong in the one direction nothing could catch. Here the corelet is a field of
/// the variant, so the illegal mask cannot be written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreletUnit {
    /// The systolic array, `PTID = 0x1`.
    Pt,
    /// `PEID = 0x2`.
    Pe,
    /// `SFPID = 0x4`.
    Sfp,
    /// `L0LUID = 0x8`.
    L0Lu,
    /// `L0SUID = 0x10`.
    L0Su,
    /// `LXLUID = 0x20`.
    LxLu,
    /// `LXSUID = 0x40`.
    LxSu,
}

impl CoreletUnit {
    /// Every corelet-resident unit `dip.h:38-48` defines an id for — enumerated, so a sweep over the world
    /// cannot silently skip one.
    pub const ALL: [CoreletUnit; 7] = [
        CoreletUnit::Pt,
        CoreletUnit::Pe,
        CoreletUnit::Sfp,
        CoreletUnit::L0Lu,
        CoreletUnit::L0Su,
        CoreletUnit::LxLu,
        CoreletUnit::LxSu,
    ];

    /// This unit's id bit (`dip.h:38-48`). Total: every unit in the set has one, which is why the set is exactly
    /// these seven.
    pub const fn id(self) -> UnitId {
        match self {
            CoreletUnit::Pt => UnitId(0x1),
            CoreletUnit::Pe => UnitId(0x2),
            CoreletUnit::Sfp => UnitId(0x4),
            CoreletUnit::L0Lu => UnitId(0x8),
            CoreletUnit::L0Su => UnitId(0x10),
            CoreletUnit::LxLu => UnitId(0x20),
            CoreletUnit::LxSu => UnitId(0x40),
        }
    }
}

/// An L3 unit, which carries NO CORELET.
///
/// ⭐ SEPARATE FROM [`CoreletUnit`] BECAUSE THE HEADER RULE IS THE OPPOSITE ONE, and it is not a convention:
/// `dpc.cpp:1072-1075` makes specifying a corelet for L3 AN ERROR, and from the reader's side `slice_id >= 6`
/// requires `mask == unitID` exactly (`dip.cpp:866-868`). Two units with an inverted obligation are two types;
/// one enum plus a check is how the previous port had to say it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3Unit {
    /// `L3LUID = 0x200`.
    L3Lu,
    /// `L3SUID = 0x400`.
    L3Su,
}

impl L3Unit {
    /// Both L3 units, enumerated.
    pub const ALL: [L3Unit; 2] = [L3Unit::L3Lu, L3Unit::L3Su];

    /// This unit's id bit (`dip.h:38-48`).
    pub const fn id(self) -> UnitId {
        match self {
            L3Unit::L3Lu => UnitId(0x200),
            L3Unit::L3Su => UnitId(0x400),
        }
    }
}

/// WHICH CORELET — one of the two.
///
/// `CORELET0ID = 0x80`, `CORELET1ID = 0x100` (`dip.h:40-41`).
///
/// ⭐ AN ENUM, SO THERE IS NO CORELET 2. dxp reaches one by COUNTING: `allocAllMem` builds its corelet list with
/// `for (int i = 1; i < currDsc->numCoreletsUsed_DSC2_; i++) corelets.push_back(i);` (`ddcv1.cpp:207-209`), so a
/// `numCoreletsUsed` of 3 yields corelet 2 — a corelet no core has and the header has no bit for. That count is a
/// runtime field of a descriptor dxp is handed; the emitter here states the corelet set outright, so neither the
/// count nor a refusal of it exists in this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Corelet {
    /// `CORELET0ID = 0x80`.
    Zero,
    /// `CORELET1ID = 0x100`.
    One,
}

impl Corelet {
    /// This corelet's bit (`dip.h:40-41`).
    pub const fn bits(self) -> u16 {
        match self {
            Corelet::Zero => 0x80,
            Corelet::One => 0x100,
        }
    }
}

/// WHICH CORELETS one block targets.
///
/// ⭐ A BLOCK CAN TARGET BOTH, AND THAT IS NOT AN EDGE CASE — IT IS THE NORMAL CASE. The two corelets SHARE a
/// slice column (`dip.cpp:851-853`), so when their inits are identical the block is sent ONCE with
/// `CORELET0ID | CORELET1ID | unitID`, and the reader accepts exactly that mask (`dip.cpp:854-863`). PT sets both
/// unconditionally — `unit_mask = CORELET0ID | CORELET1ID | PTID` (`dip.cpp:4611`, commented "assume 2 corelets
/// and identical init packets") — and only `finalizeHeaderFlit` narrows it if the corelets turn out to differ.
///
/// This exists because an `Option<Corelet>` could not say "both", so the block builder would have had to OR the
/// second bit in by hand — which is precisely the composition [`TargetMask`] is sealed to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corelets {
    /// One corelet only — the block's inits differ from the other's, or only one is used.
    Just(Corelet),
    /// Both, with identical inits, sent once (`dip.cpp:854-858`).
    Both,
}

impl Corelets {
    /// Every set a block can target, enumerated.
    pub const ALL: [Corelets; 3] = [
        Corelets::Just(Corelet::Zero),
        Corelets::Just(Corelet::One),
        Corelets::Both,
    ];

    /// The corelets this set contains, in index order.
    ///
    /// What `allocAllMem`'s corelet loop walks (`ddcv1.cpp:205-210`). Returning the MEMBERS means a caller never
    /// counts up to a bound, which is where corelet 2 came from.
    pub const fn members(self) -> &'static [Corelet] {
        match self {
            Corelets::Just(Corelet::Zero) => &[Corelet::Zero],
            Corelets::Just(Corelet::One) => &[Corelet::One],
            Corelets::Both => &[Corelet::Zero, Corelet::One],
        }
    }

    /// The OR of this set's corelet bits.
    pub const fn bits(self) -> u16 {
        match self {
            Corelets::Just(c) => c.bits(),
            Corelets::Both => Corelet::Zero.bits() | Corelet::One.bits(),
        }
    }
}

/// WHAT ONE HEADER TARGETS — a unit, and the corelets it must or must not name.
///
/// ⭐ THE TWO HALVES OF THE READER'S RULE (`dip.cpp:844-870`, which reconstructs a component from a mask and
/// errors on anything else) ARE THE TWO VARIANTS:
///
/// - **L3 must carry NO corelet.** `slice_id >= 6` requires `mask == unitID` exactly (`dip.cpp:866-868`),
///   matching `dpc.cpp:1072-1075` from the writer's side.
/// - **Every other unit MUST carry one**, or the reader raises "Header with an invalid unit_mask"
///   (`dip.cpp:864`).
///
/// As variants, [`Self::mask`] is TOTAL: no illegal combination is left to refuse, so no `Option` stands in for
/// the fact of which unit this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// A corelet-resident unit, with the corelets it names.
    OnCorelets {
        /// Which unit.
        unit: CoreletUnit,
        /// Which corelets — one or both.
        corelets: Corelets,
    },
    /// An L3 unit, whose mask is the bare id.
    L3(L3Unit),
}

impl Target {
    /// EVERY TARGET A HEADER CAN NAME — 7 corelet units × 3 corelet sets, plus the 2 L3 units.
    ///
    /// ⭐ HERE SO THE INVARIANTS CAN SWEEP THE WHOLE WORLD RATHER THAN A SAMPLE.
    /// [`_EVERY_TARGET_BIT_IS_REACHED`] checks the OR over this table against the union of `dip.h:38-48`'s
    /// eleven defines, so a unit dropped from the table and a bit invented anywhere both break the build.
    pub const ALL: [Target; 23] = {
        let mut all = [Target::L3(L3Unit::L3Lu); 23];
        let mut at = 0;
        let mut unit = 0;
        while unit < CoreletUnit::ALL.len() {
            let mut corelets = 0;
            while corelets < Corelets::ALL.len() {
                all[at] = Target::OnCorelets {
                    unit: CoreletUnit::ALL[unit],
                    corelets: Corelets::ALL[corelets],
                };
                at += 1;
                corelets += 1;
            }
            unit += 1;
        }
        let mut l3 = 0;
        while l3 < L3Unit::ALL.len() {
            all[at] = Target::L3(L3Unit::ALL[l3]);
            at += 1;
            l3 += 1;
        }
        assert!(at == all.len(), "every target must be listed exactly once");
        all
    };

    /// This target's `targetUnitBits`. Total, because the variants already excluded the two illegal shapes.
    pub const fn mask(self) -> TargetMask {
        match self {
            Target::OnCorelets { unit, corelets } => {
                TargetMask(Bits::exactly((unit.id().bits() | corelets.bits()) as u32))
            }
            // `dip.cpp:866-868` / `dpc.cpp:1072-1075`: the bare id, with no corelet bit to OR in.
            Target::L3(unit) => TargetMask(Bits::exactly(unit.id().bits() as u32)),
        }
    }
}

/// The HEADER's `target_unit_mask` — a unit id, OR-ed with a corelet bit where the unit lives on one.
///
/// Sealed: [`Target::mask`] is the only producer. The previous attempt's equivalent was a bare `u16` any site
/// could compose, and `0x81` (`CORELET0ID|PTID`) is what a CORRECT PT mask looks like — indistinguishable from a
/// wrong one at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetMask(Bits<{ TargetMask::FIELD_BITS }>);

impl TargetMask {
    /// The width of `target_unit_mask` — ELEVEN BITS (`initpacket.cpp:224`, bits 85-95;
    /// `target_unit_mask:11`).
    pub const FIELD_BITS: u32 = W2_UNIT_MASK_BITS;

    /// The mask as the field value it will be shifted into — the carrier it was BUILT in, so there is no
    /// conversion here and no second bound to keep in step.
    pub const fn field(self) -> Bits<{ Self::FIELD_BITS }> {
        self.0
    }

    /// The raw bits, for comparison against `dip.h`'s defines.
    pub const fn bits(self) -> u16 {
        self.0.get() as u16
    }
}

/// WHICH CORE — an index, never a mask and never a count.
///
/// `core_mask = 1 << coreNum` (`dip.cpp:4594`), so the two are one shift apart and BOTH are `u32` in the C++. A
/// block built for core 3 and masked for core 8 is a block the wrong core runs; the shift is the only place the
/// two meet, and it happens inside [`CoreMask::of_cores`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoreIdx(Bits<{ CoreIdx::INDEX_BITS }>);

impl CoreIdx {
    /// How many bits a core index has: `target_core_mask` is 32 bits (`initpacket.cpp:214`, bits 96-127), so it
    /// addresses 32 cores and an index needs exactly five bits.
    ///
    /// ⭐ A CORE THE MASK CANNOT ADDRESS IS NOT EXPRESSIBLE AS ONE. A block targeting fewer cores than intended
    /// leaves a core running nothing — silent, and visible only as a wrong answer from that core.
    pub const INDEX_BITS: u32 = 5;

    /// The core with this index. Total: five bits is exactly the addressable set.
    pub const fn of_index(index: Bits<{ Self::INDEX_BITS }>) -> Self {
        Self(index)
    }

    /// The index.
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// WHICH CORES a block targets — `1 << core`, OR-ed.
///
/// `putU32OneHotInBitRange(uvalue, 96, 127, targetCores)` (`initpacket.cpp:214`). A newtype because it is a
/// BITSET over core ids while the field beside it is a unit bitmask: two 32-bit masks over different domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreMask(Bits<{ CoreMask::FIELD_BITS }>);

impl CoreMask {
    /// The width of `target_core_mask`.
    ///
    /// ⭐ THIRTY-TWO BITS — THE FULL CARRIER. Unlike [`TargetMask::FIELD_BITS`] (11 of a `u16`), this field is
    /// exactly as wide as the value that fills it, and it is the LAST field in the layout so there is nothing
    /// above it to corrupt. The two fields are therefore safe for DIFFERENT reasons, and collapsing the two
    /// stories would either drop a real guarantee or add a phantom one: `TargetMask` fits because of what
    /// [`Target`] can produce, this fits by WIDTH for every `u32` there is.
    pub const FIELD_BITS: u32 = 32;

    /// THE MASK FOR THE ONE CORE A REGULAR BLOCK INITIALISES.
    ///
    /// ⛔⛔⛔ A REGULAR BLOCK TARGETS EXACTLY ONE CORE, AND A MULTI-CORE MASK IS A DIFFERENT ENCODING.
    /// `Dip::isPatchInit` tests `if (targetCore & (targetCore - 1)) return true;` under the comment *"targeting
    /// more than one core"* (`patchinit.cpp:1268-1269`) — not-a-power-of-two — so a mask naming two cores makes
    /// the whole image a PATCH-INIT image BY DEFINITION. IBM's reader then enters `reversePatchInit`, finds none
    /// of the patch headers such an image must carry, and dies in `updateLxAddress`'s
    /// `DT_CHECK(check.size() == 1)` (`patchinit.cpp:31`), which requires every core a block names to reach the
    /// SAME accumulated LX address.
    ///
    /// ⭐ SO THE SHARING THAT USED TO HAPPEN HERE WAS NOT A COMPACT SPELLING — it was an invalid one. dxp's much
    /// smaller image comes from EMITTING patch-init (`constructPatchInit`, `patchinit.cpp:200-941`), an encoding
    /// with its own headers and LX bookkeeping, and until that is ported a block is one core's.
    pub const fn of_one(core: CoreIdx) -> CoreMask {
        CoreMask(Bits::exactly(1u32 << core.get()))
    }

    /// The mask for a set of cores — the cores a PROGRAM runs on, never a block header's target.
    ///
    /// ⛔ NOT FOR A BLOCK HEADER: see [`CoreMask::of_one`]. This answers "which cores have this program",
    /// which is `initcores` and is legitimately many.
    pub const fn of_cores(cores: &[CoreIdx]) -> CoreMask {
        let mut mask = 0u32;
        let mut at = 0;
        while at < cores.len() {
            mask |= 1u32 << cores[at].get();
            at += 1;
        }
        CoreMask(Bits::exactly(mask))
    }

    /// The mask as the word-3 value.
    pub const fn field(self) -> Bits<{ Self::FIELD_BITS }> {
        self.0
    }

    /// THE CORES THIS MASK NAMES, ascending — `initcores`, the set a program is initialised for
    /// (`dip.cpp:134-136`).
    ///
    /// ⭐ A READER OF THE MASK, NOT A SECOND LIST. dxp takes the set from `progstateinfo`'s keys — the cores that
    /// have a PROGRAM — and the mask a block is addressed to is built from that same set, so one of the two is
    /// derived and this is the derivation.
    pub fn cores(self) -> Vec<CoreIdx> {
        (0..Self::FIELD_BITS)
            .filter(|core| self.0.get() & (1u32 << core) != 0)
            .map(|core| CoreIdx::of_index(Bits::exactly(core)))
            .collect()
    }

    /// The raw bits.
    pub const fn bits(self) -> u32 {
        self.0.get()
    }
}

/// ONE NIBBLE OF `lrf_idx` — WHAT IT NAMES DEPENDS ON THE UNIT.
///
/// ⛔ THIS IS NOT ALWAYS A REGISTER NUMBER, and the doc here said it was. For a COMPUTE unit (SFP/PE) a slice is
/// one 128-bit register and the nibble is that register's number; for an L3 unit a slice is an init PACKET
/// covering several registers of one file (`dip.cpp:2672-2858`, ten packets), and the nibble is the packet's
/// number. One four-bit encoding, two meanings — so the type is named for the ENCODING it is, and the two
/// meanings are stated here rather than one of them being asserted in the name.
///
/// ⭐ FOUR BITS PER ENTRY, so nothing above 15 can be named in the header at all — which is independent
/// corroboration, from the PACKET FORMAT rather than from `reg_info_per_unit`, of the compute LRF depth of 16
/// (`6d86e23c`). Two unrelated sources agreeing on that number is worth more than either alone, and on SEN1P5
/// (depth 32) they DISAGREE: registers 16..31 have no nibble, which is a real question for whoever targets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LrfRegIdx(Bits<{ LrfRegIdx::BITS }>);

impl LrfRegIdx {
    /// Bits per entry in `lrf_idx`: `dip.cpp:403-406` decodes `idx = lrf_idx & 0xF; lrf_idx >>= 4`.
    pub const BITS: u32 = 4;

    /// The register with this number. Total: four bits is the whole nameable set.
    pub const fn of_register(register: Bits<{ Self::BITS }>) -> Self {
        Self(register)
    }

    /// The register number.
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// WHICH REGISTERS this block initialises — the header's words 0 and 1.
///
/// ⭐⭐ `lrf_idx` IS A PACKED LIST OF 4-BIT REGISTER NUMBERS, NOT A BITMAP. `dip.cpp:403-406` decodes it as
/// `idx = lrf_idx & 0xF; lrf_idx >>= 4` — four bits per entry, LOW NIBBLE FIRST, `numIdx` entries. So the
/// register list `{0, 2, 4, 6}` is `0x6420`, which is exactly what the captured L3SU bundle carries in word 0.
///
/// ⛔⛔ AND ON SEN1P5 IT IS ONE-HOT AFTER ALL — see [`Self::of_registers_one_hot`]. `dip.cpp:4237-4251` chooses by
/// arch: a one-hot mask over 32 bits on SEN1P5, the nibble list above. This file previously asserted that "one-hot
/// is not this field's encoding" as a universal fact, which is right for RCUDD1A and exactly wrong for the other
/// arch — and it matches the register depths, since the SFP's LRF holds 16 there and 32 here.
///
/// ⛔ WHAT THIS WORD DOES NOT CARRY IS THE ENTRY COUNT. The reader loops `numIdx` times (`dip.cpp:403-406`) and
/// this port does NOT invent where that comes from: the registers are handed in as a list by whoever builds the
/// LRF rows, so the count is that list's length and cannot disagree with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LrfIndexList(u64);

impl LrfIndexList {
    /// How many WORDS of the slice the list occupies: words 0 and 1 (`initpacket.cpp:234-237` writes the indices into bits 0..31).
    pub const WORDS: usize = 2;

    /// How many entries those words hold: 64 bits at four bits each.
    pub const ENTRIES: usize = (u64::BITS / LrfRegIdx::BITS) as usize;

    /// The list this block's LRF rows initialise, packed LOW NIBBLE FIRST (`dip.cpp:403-406`).
    ///
    /// Taken WITH the LRF rows from one call site, so the header's `lrf_idx` cannot describe a different set of
    /// registers than the block's rows initialise.
    ///
    /// ⭐ THE LENGTH IS A `panic!`, NOT A TRUNCATION. dxp would be clamping a runtime vector; a list longer than
    /// the two words is a caller's mistake at expansion time, and shifting its tail into nothing would lose
    /// register initialisations invisibly.
    pub const fn of_registers(registers: &[LrfRegIdx]) -> Self {
        if registers.len() > Self::ENTRIES {
            panic!(
                "lrf_idx holds 16 four-bit register numbers (dip.cpp:403-406); a longer list would shift \
                 entries out of words 0 and 1 entirely"
            );
        }
        let mut bits = 0u64;
        let mut at = 0;
        while at < registers.len() {
            bits |= (registers[at].get() as u64) << (at as u32 * LrfRegIdx::BITS);
            at += 1;
        }
        Self(bits)
    }

    /// THE SAME FIELD ON SEN1P5, WHERE IT IS A ONE-HOT BITMASK AND NOT A LIST.
    ///
    /// ⛔⛔ TWO ENCODINGS OF ONE FIELD, CHOSEN BY ARCH. `dip.cpp:4237-4251` is explicit:
    ///
    /// ```text
    /// if (coreArch == SEN1P5_ISA) {              // "multi one-hot encoding of LSB 32 bits"
    ///     if (elem > 31) DT_ERROR(...);
    ///     unitHeader.lrf_idx |= 1ull << (elem & 0x1F);
    /// } else {
    ///     unitHeader.lrf_idx |= (elem & 0xF) << bitpos;  bitpos += 4;
    /// }
    /// ```
    ///
    /// ⭐ AND THE TWO AGREE WITH THE REGISTER DEPTHS, which is what makes this more than a spelling difference: the
    /// SFP's LRF holds 16 on RCUDD1A and 32 on SEN1P5 (`sysdef.cpp`, ported in `isa::regfile`). Sixteen registers
    /// fit a 4-bit number; thirty-two do not, so the nibble list could not name half of SEN1P5's file and the
    /// one-hot mask is exactly 32 bits wide.
    ///
    /// ⛔ SO A REGISTER ABOVE 31 IS AN ERROR RATHER THAN A WRAP — the C++ raises `DT_ERROR`, and `& 0x1F` would
    /// otherwise alias register 32 onto register 0.
    pub const fn of_registers_one_hot(registers: &[OneHotRegIdx]) -> Self {
        let mut bits = 0u64;
        let mut at = 0;
        while at < registers.len() {
            bits |= 1u64 << registers[at].get();
            at += 1;
        }
        Self(bits)
    }

    /// The packed 64 bits — the low half is word 0.
    pub const fn bits(self) -> u64 {
        self.0
    }
}

/// A REGISTER NAMED IN SEN1P5'S ONE-HOT `lrf_idx` — `0..=31` (`dip.cpp:4245-4248`).
///
/// ⛔ A DIFFERENT TYPE FROM [`LrfRegIdx`] BECAUSE IT IS A DIFFERENT FIELD ENCODING, not a wider version of the
/// same one. A `LrfRegIdx` is one entry of a packed list and cannot exceed 15; this is a bit POSITION in a mask and
/// reaches 31. Sharing a type would let a nibble-packed list be built from indices the nibbles cannot hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OneHotRegIdx(u8);

impl OneHotRegIdx {
    /// The highest register the mask can name — `if (elem > 31) DT_ERROR` (`dip.cpp:4247`).
    pub const MAX: u8 = 31;

    /// ⛔ NOT A MASK. `& 0x1F` would alias register 32 onto register 0 and initialise the wrong one; the C++ errors
    /// instead, and `const` makes it a build error here.
    pub const fn of_register(register: u8) -> Self {
        assert!(
            register <= Self::MAX,
            "SEN1P5's `lrf_idx` is a one-hot mask over 32 registers; a higher index has no bit and masking it \
             would alias onto another register (dip.cpp:4245-4248)"
        );
        Self(register)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// HOW MANY LRF FLITS FOLLOW — a five-bit field, distinct from the IBUFF count beside it.
///
/// ⛔⛔ THE PREVIOUS PORT'S DOC SAID "5-bit field" AND ITS CONSTRUCTOR TOOK ANY `u8`. `new` was total, so
/// 32..=255 were expressible for a field holding 0..=31 — and [`Header::slice`] packs by SHIFTING, so those high
/// bits land in `pktIbuff_flit` above and the block declares instruction flits it does not carry. A stated
/// constraint the constructor does not enforce is the same defect as no constraint, and this one was written down
/// in the type's own doc, which is what made it invisible. Here the bound is the const generic of the value
/// handed in, so 32 is not a `LrfFlitCount` at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct LrfFlitCount(Bits<{ LrfFlitCount::FIELD_BITS }>);

impl LrfFlitCount {
    /// The width of `pktLRF_flit` — FIVE BITS, bits 73-77 (`initpacket.cpp:226`).
    pub const FIELD_BITS: u32 = W2_PKT_LRF_BITS;

    /// The count. Total: a `Bits<5>` is already inside the field.
    pub const fn of_flits(flits: Bits<{ Self::FIELD_BITS }>) -> Self {
        Self(flits)
    }

    /// The count as the field value it will be shifted into.
    pub const fn field(self) -> Bits<{ Self::FIELD_BITS }> {
        self.0
    }
}

/// HOW MANY IBUFF (INSTRUCTION) FLITS FOLLOW — a seven-bit field.
///
/// `numIbuffFlits`, bits 78-84 (`initpacket.cpp:225`). A type of its own and not a second [`LrfFlitCount`]: the
/// two are adjacent counts in the same word, which is exactly the pair whose swap at a positional call site the
/// previous port found it could not detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct InstrFlitCount(Bits<{ InstrFlitCount::FIELD_BITS }>);

impl InstrFlitCount {
    /// The width of `pktIbuff_flit` — SEVEN BITS.
    pub const FIELD_BITS: u32 = W2_INSTR_FLITS_BITS;

    /// The count. Total: a `Bits<7>` is already inside the field.
    pub const fn of_flits(flits: Bits<{ Self::FIELD_BITS }>) -> Self {
        Self(flits)
    }

    /// The count as the field value it will be shifted into.
    pub const fn field(self) -> Bits<{ Self::FIELD_BITS }> {
        self.0
    }
}

/// WHERE EACH FIELD OF HEADER WORD 2 SITS — declared once, and used BY the packing.
///
/// ⭐ THE SHIFTS ARE `dip.cpp:4519-4524`'s — THE CODE THAT PACKS, NOT `dip.h`'s DECLARATION.
/// `initpacket.cpp:226` writes `numIbuffCorrectionFlits` into bits 64..69 — six bits where the distance to the next shift
/// leaves NINE bits (the 6, plus bit 70 reserved, plus `pktLCCR_flit:2`). The two disagree, and the module doc
/// resolves it: the struct is a LEGACY layout and the generating function is the answer.
///
/// The citation lives HERE rather than as a pointer to prose elsewhere in the file — "the doc above records it"
/// is a reference, not a source. Each width is the distance to the next field's shift, which is the room the
/// packing gives.
///
/// Order is low bit to high: `(shift, width)`.
const HEADER_W2_FIELDS: [(u32, u32); 4] = [
    // `pktIbuffCorrection_flit` — NINE bits of room by the shifts, of which the low 6 are the count. Fixed at 0
    // by this port (`dip.cpp:2667-2668`: LCCR flits are "Not used in code generation"), so the reserved bit and
    // the LCCR field sharing this room are zero too.
    (0, 9),
    // `pktLRF_flit`.
    (9, 5),
    // `pktIbuff_flit`.
    (14, 7),
    // `target_unit_mask` — the top field of the word.
    (21, 11),
];

/// WHICH FIELD of header word 2 — a closed set of four, NAMED.
///
/// ⛔⛔ THIS REPLACED POSITIONAL INDICES. The first version of the layout offered `w2_shift(0)`, `w2_shift(1)`,
/// `w2_shift(2)`, `w2_shift(3)`, so the packing shifted by MAGIC NUMBERS into a table and swapping two calls
/// would have compiled and packed each field into the other's bits. The word is the last thing before the device
/// reads it; there is no later check.
///
/// ⭐ AND THE TABLE'S ENTRIES ARE `(u32, u32)`, so a shift and a width are positionally swappable THERE too —
/// `(9, 5)` and `(5, 9)` are both well-typed. Naming the field and reading through [`Self::shift`] /
/// [`Self::width`] means a call site never touches either position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderW2Field {
    /// `pktIbuffCorrection_flit`, with the reserved bit and `pktLCCR_flit` sharing its room.
    Correction,
    /// `pktLRF_flit`.
    PktLrf,
    /// `pktIbuff_flit`.
    InstrFlits,
    /// `target_unit_mask`.
    UnitMask,
}

impl HeaderW2Field {
    /// Position in [`HEADER_W2_FIELDS`], low bit to high — the ONE place the order is written.
    const fn index(self) -> usize {
        match self {
            HeaderW2Field::Correction => 0,
            HeaderW2Field::PktLrf => 1,
            HeaderW2Field::InstrFlits => 2,
            HeaderW2Field::UnitMask => 3,
        }
    }

    /// How far up the word this field starts.
    pub const fn shift(self) -> u32 {
        HEADER_W2_FIELDS[self.index()].0
    }

    /// How many bits it has, which is the distance to the next field's shift.
    pub const fn width(self) -> u32 {
        HEADER_W2_FIELDS[self.index()].1
    }
}

/// `pktIbuffCorrection_flit`'s room, from [`HEADER_W2_FIELDS`].
///
/// ⭐ THE POINT OF DERIVING THESE: the count types are `Bits<W2_*_BITS>` and [`Header::slice`] shifts by
/// `HeaderW2Field::*.shift()`, both out of the SAME declaration, so a field's width and its position cannot
/// drift apart. The third copy of the layout — packing literals, then a test's own table — never exists.
pub const W2_CORRECTION_BITS: u32 = HeaderW2Field::Correction.width();
/// `pktLRF_flit`'s width, from [`HEADER_W2_FIELDS`].
pub const W2_PKT_LRF_BITS: u32 = HeaderW2Field::PktLrf.width();
/// `pktIbuff_flit`'s width, from [`HEADER_W2_FIELDS`].
pub const W2_INSTR_FLITS_BITS: u32 = HeaderW2Field::InstrFlits.width();
/// `target_unit_mask`'s width, from [`HEADER_W2_FIELDS`].
pub const W2_UNIT_MASK_BITS: u32 = HeaderW2Field::UnitMask.width();

/// WHICH SLICE BIT WORD 2 BEGINS AT — 64, because words 0 and 1 are `lrf_idx`.
///
/// `initpacket.cpp:214-239` states the layout in SLICE bits (64-127) while the packing shifts within a WORD, and
/// this is the only place the two coordinate systems are related. Writing `73 - 64` at each use site would put
/// four copies of that relation in the file.
pub const W2_BASE_BIT: u32 = 64;

/// A UNIT'S HEADER — the first row of its init block (`Dip::finalizeHeaderFlit`, `dip.cpp:2136-2191`).
///
/// It names the unit and cores the block targets, and declares how many LRF and IBUFF flits follow.
///
/// ⛔ THE FIELDS ARE PUBLIC, AND THAT IS SAFE ONLY BECAUSE EVERY ONE OF THEM IS A CHECKED TYPE. The previous
/// port had to seal an equivalent struct `pub(crate)` because its members were raw integers, and a raw
/// `unit_mask: u16` can express `0x81` (`CORELET0ID|PTID`) — which is what a CORRECT PT mask looks like and is
/// therefore indistinguishable from a wrong one. A [`Target`] cannot be a wrong mask, so there is nothing left
/// for a seal to protect.
///
/// ⛔ THERE IS NO SPR FLIT COUNT, and none is invented: `initpacket.cpp:214-239` states five bit ranges and an
/// SPR count is not among them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Which registers the block's LRF rows initialise (words 0/1).
    pub registers: LrfIndexList,
    /// How many LRF flits follow.
    pub lrf_flits: LrfFlitCount,
    /// How many IBUFF (instruction) flits follow.
    pub instr_flits: InstrFlitCount,
    /// Which unit — and, unless it is L3, which corelets.
    pub target: Target,
    /// WHICH ONE CORE — see [`CoreMask::of_one`] for why this cannot be a mask.
    pub core: CoreIdx,
}

impl Header {
    /// The IBUFF-CORRECTION FLIT COUNT, FIXED AT ZERO — which is why it is not a field.
    ///
    /// `dip.cpp:2890` states it outright — *"LCCR Flits: Not used in code generation 10/21 -- so assuming no
    /// LCCR"* — and repeats it at `:3528`, `:3959` and `:4626`, with `:4269` adding *"s/w does not generate them
    /// yet"*. Four independent sites saying the same thing is what makes this a documented property of the path
    /// rather than an assumption, and a field for it would be a question asked at emission that the source text
    /// already answers.
    ///
    /// ⛔ THE LINE NUMBER USED TO BE `2667-2668`, WHICH IN THE CURRENT MIRROR IS `updateL3RegFoldInfo` AND SAYS
    /// NOTHING ABOUT LCCR. The FACT survived the check; the citation did not. `dxp.cpp`'s citations drift the
    /// same way — re-locate by SYMBOL or by the quoted text, never by trusting a line.
    pub const CORRECTION_FLITS: Bits<{ W2_CORRECTION_BITS }> = Bits::exactly(0);

    /// The 16-byte HEADER slice: `w0/w1 = lrf_idx`, `w2` the four packed fields, `w3 = target_core_mask`.
    ///
    /// ⭐ SHIFTED BY THE DECLARATION, NOT BY LITERALS — see [`HEADER_W2_FIELDS`]. A changed shift moves the
    /// packing AND the invariants below together, instead of leaving an invariant validating a copy.
    ///
    /// ⭐ NO VALUE HERE CAN OVERFLOW ITS FIELD. Each is a `Bits<N>` whose `N` is the same width the table gives
    /// the shift for, so the packing cannot EXPRESS a collision — the guarantee moved from "every producer
    /// refuses" to "this expression cannot go wrong".
    pub const fn slice(self) -> Slice {
        let w2 = (Self::CORRECTION_FLITS.get() << HeaderW2Field::Correction.shift())
            | (self.lrf_flits.field().get() << HeaderW2Field::PktLrf.shift())
            | (self.instr_flits.field().get() << HeaderW2Field::InstrFlits.shift())
            | (self.target.mask().field().get() << HeaderW2Field::UnitMask.shift());
        let registers = self.registers.bits();
        Slice([
            registers as u32,
            (registers >> 32) as u32,
            w2,
            CoreMask::of_one(self.core).field().get(),
        ])
    }
}

/// THE FOUR FIELDS OF WORD 2 TILE IT EXACTLY, AND EACH BEGINS WHERE THE LAST ENDS.
///
/// ⭐ THE TILING GUARANTEE AS A `cargo build` ERROR rather than a test. Every input is a compile-time constant,
/// so a widened field or a moved shift stops the crate compiling HERE, naming the sum it got.
const _W2_FIELDS_TILE_THE_WORD: () = {
    let sum = HeaderW2Field::Correction.width()
        + HeaderW2Field::PktLrf.width()
        + HeaderW2Field::InstrFlits.width()
        + HeaderW2Field::UnitMask.width();
    assert!(
        sum == 32,
        "header word 2's four fields must fill exactly 32 bits"
    );
    assert!(
        HeaderW2Field::Correction.shift() == 0,
        "the low field starts at bit 0"
    );
    assert!(
        HeaderW2Field::PktLrf.shift()
            == HeaderW2Field::Correction.shift() + HeaderW2Field::Correction.width()
    );
    assert!(
        HeaderW2Field::InstrFlits.shift()
            == HeaderW2Field::PktLrf.shift() + HeaderW2Field::PktLrf.width()
    );
    assert!(
        HeaderW2Field::UnitMask.shift()
            == HeaderW2Field::InstrFlits.shift() + HeaderW2Field::InstrFlits.width()
    );
};

/// THE SHIFTS ARE `initpacket.cpp:214-239`'s BIT RANGES, RESTATED.
///
/// A range `[lo, hi]` of `initpacket.cpp`'s slice bits is shift `lo - `[`W2_BASE_BIT`] and width `hi - lo + 1`.
/// Pinned against the RANGES the C++ writes rather than against the shifts the packing uses, so the layout's two
/// independent statements must agree.
const _W2_SHIFTS_ARE_THE_C_PLUS_PLUS_BIT_RANGES: () = {
    // Words 0 and 1 are `lrf_idx` in full, which is what puts word 2 at slice bit 64.
    assert!(W2_BASE_BIT == LrfIndexList::WORDS as u32 * u32::BITS);
    // `putU32BitRange(uvalue, 73, 77, numLrfFilts)` — 5 bits at 73.
    assert!(HeaderW2Field::PktLrf.shift() == 73 - W2_BASE_BIT);
    assert!(HeaderW2Field::PktLrf.width() == 77 - 73 + 1);
    // `putU32BitRange(uvalue, 78, 84, numIbuffFlits)` — 7 bits at 78.
    assert!(HeaderW2Field::InstrFlits.shift() == 78 - W2_BASE_BIT);
    assert!(HeaderW2Field::InstrFlits.width() == 84 - 78 + 1);
    // `putU32BitRange(uvalue, 85, 95, targetUnitBits)` — 11 bits at 85, the top of the word.
    assert!(HeaderW2Field::UnitMask.shift() == 85 - W2_BASE_BIT);
    assert!(HeaderW2Field::UnitMask.width() == 95 - 85 + 1);
    // `putU32OneHotInBitRange(uvalue, 96, 127, targetCores)` — word 3 begins where word 2 ends, so the core mask
    // is a whole word and not a field of this one.
    assert!(96 - W2_BASE_BIT == u32::BITS, "word 2 ends at slice bit 95");
    // `putU32BitRange(uvalue, 64, 69, numIbuffCorrectionFlits)` — the COUNT is 6 bits at 64, and the room above
    // it (bit 70 reserved, LCCR 71-72) is part of the same tile only because this port writes zero across all
    // of it.
    assert!(HeaderW2Field::Correction.shift() == 64 - W2_BASE_BIT);
    assert!(
        HeaderW2Field::Correction.width() == 9,
        "6 count bits + reserved + 2 LCCR bits"
    );
    assert!(
        Header::CORRECTION_FLITS.get() == 0,
        "dip.cpp:2890 and three more"
    );
};

/// THE ID TABLE IS `dip.h:38-48`'s, ENTRY BY ENTRY.
///
/// ⭐ AND IT IS CONFIRMED AGAINST A SECOND, INDEPENDENT SOURCE: `HeaderInitSlice::bitposToComp`
/// (`sys-arch-spec/initpacket/initpacket.cpp:1120-1142`) maps the very bits back to units — PT/PE/SFP/L0LU/
/// L0SU/LXLU/LXSU at bits 0..6, `1 << 7` and `1 << 8` for the two corelets, `1 << 9` L3LU and `1 << 10` L3SU.
/// Two files stating one table is what makes these numbers a FACT rather than a transcription, and it is also
/// why `TargetMask` is eleven bits: bit 10 is the highest any target reaches.
const _UNIT_IDS_ARE_THE_C_PLUS_PLUS_DEFINES: () = {
    assert!(CoreletUnit::Pt.id().bits() == 0x1);
    assert!(CoreletUnit::Pe.id().bits() == 0x2);
    assert!(CoreletUnit::Sfp.id().bits() == 0x4);
    assert!(CoreletUnit::L0Lu.id().bits() == 0x8);
    assert!(CoreletUnit::L0Su.id().bits() == 0x10);
    assert!(CoreletUnit::LxLu.id().bits() == 0x20);
    assert!(CoreletUnit::LxSu.id().bits() == 0x40);
    assert!(L3Unit::L3Lu.id().bits() == 0x200);
    assert!(L3Unit::L3Su.id().bits() == 0x400);
    assert!(Corelet::Zero.bits() == 0x80);
    assert!(Corelet::One.bits() == 0x100);
    assert!(Corelets::Both.bits() == 0x180, "CORELET0ID | CORELET1ID");
    assert!(Corelets::Just(Corelet::One).members().len() == 1);
    assert!(Corelets::Both.members().len() == 2);
};

/// EVERY BIT `dip.h` DEFINES IS REACHED, AND NO OTHER BIT EVER APPEARS.
///
/// ⭐ THE "WRONG ANSWER IS ABSENT" HALF: the OR over every constructible target is exactly the union of the
/// eleven defines (`0x67F` of unit ids, `0x180` of corelets, together `0x7FF`). A unit dropped from
/// [`Target::ALL`] makes the OR too small; a bit invented anywhere makes it too large. Both break the build.
const _EVERY_TARGET_BIT_IS_REACHED: () = {
    let mut reached = 0u16;
    let mut at = 0;
    while at < Target::ALL.len() {
        let bits = Target::ALL[at].mask().bits();
        reached |= bits;
        // And no mask may exceed the field, whatever it is made of.
        assert!(bits as u32 <= Bits::<{ TargetMask::FIELD_BITS }>::MAX);
        at += 1;
    }
    assert!(
        reached == 0x7FF,
        "every dip.h:38-48 define, and nothing else"
    );
};

/// THE MASK SHAPES THE READER ACCEPTS, PINNED — INCLUDING THE ONES THAT MUST NOT APPEAR.
const _TARGET_MASKS_ARE_THE_READERS_SHAPES: () = {
    // `dip.cpp:4611`: PT's default is `CORELET0ID | CORELET1ID | PTID`.
    assert!(
        Target::OnCorelets {
            unit: CoreletUnit::Pt,
            corelets: Corelets::Both
        }
        .mask()
        .bits()
            == 0x181
    );
    // One corelet only: `CORELET0ID|PTID`, the mask a real PT block carries.
    assert!(
        Target::OnCorelets {
            unit: CoreletUnit::Pt,
            corelets: Corelets::Just(Corelet::Zero)
        }
        .mask()
        .bits()
            == 0x81
    );
    assert!(
        Target::OnCorelets {
            unit: CoreletUnit::Sfp,
            corelets: Corelets::Just(Corelet::Zero)
        }
        .mask()
        .bits()
            == 0x84,
        "SFPID | CORELET0ID"
    );
    assert!(
        Target::OnCorelets {
            unit: CoreletUnit::Sfp,
            corelets: Corelets::Both
        }
        .mask()
        .bits()
            == 0x184
    );
    // `dip.cpp:866-868` / `dpc.cpp:1072-1075`: L3 is the BARE id, with no corelet bit anywhere in it.
    assert!(Target::L3(L3Unit::L3Lu).mask().bits() == 0x200);
    assert!(Target::L3(L3Unit::L3Su).mask().bits() == 0x400);
    let mut at = 0;
    while at < L3Unit::ALL.len() {
        assert!(
            Target::L3(L3Unit::ALL[at]).mask().bits() & Corelets::Both.bits() == 0,
            "a corelet bit on L3 is an ERROR (dpc.cpp:1072-1075), not a no-op"
        );
        at += 1;
    }
    // And the converse over the whole world: every corelet-resident target carries a corelet bit AND its own id,
    // because a bare compute id decodes to no unit at all (`dip.cpp:864`).
    let mut at = 0;
    while at < Target::ALL.len() {
        match Target::ALL[at] {
            Target::OnCorelets { unit, corelets } => {
                let bits = Target::ALL[at].mask().bits();
                assert!(bits & Corelets::Both.bits() != 0);
                assert!(bits & unit.id().bits() != 0);
                assert!(bits == unit.id().bits() | corelets.bits());
            }
            Target::L3(_) => (),
        }
        at += 1;
    }
};

/// THE COUNTS CANNOT HOLD A VALUE THEIR FIELD CANNOT, AT EITHER END.
///
/// ⛔ THE WIDEST VALUE IS THE ONE THAT MATTERS. 31 is the widest LRF count and 127 the widest IBUFF count; 32
/// and 128 are the first values that would set the bit ABOVE the field, and neither is expressible.
const _FLIT_COUNTS_ARE_BOUNDED_BY_THEIR_FIELDS: () = {
    assert!(LrfFlitCount::FIELD_BITS == 5);
    assert!(Bits::<{ LrfFlitCount::FIELD_BITS }>::MAX == 31);
    assert!(Bits::<{ LrfFlitCount::FIELD_BITS }>::new(32).is_none());
    assert!(LrfFlitCount::of_flits(Bits::exactly(31)).field().get() == 31);

    assert!(InstrFlitCount::FIELD_BITS == 7);
    assert!(Bits::<{ InstrFlitCount::FIELD_BITS }>::MAX == 127);
    assert!(Bits::<{ InstrFlitCount::FIELD_BITS }>::new(128).is_none());
    assert!(InstrFlitCount::of_flits(Bits::exactly(127)).field().get() == 127);

    // The unit mask is NARROWER than the `u16` its ids are written as, which is why its bound is a property of
    // what `Target` can produce; 2048 is the first value the field cannot hold.
    assert!(Bits::<{ TargetMask::FIELD_BITS }>::MAX == 2047);
    assert!(Bits::<{ TargetMask::FIELD_BITS }>::new(2048).is_none());
    // The core mask is THE FULL CARRIER: every `u32` fits, by width alone.
    assert!(Bits::<{ CoreMask::FIELD_BITS }>::MAX == u32::MAX);
    assert!(Bits::<{ CoreMask::FIELD_BITS }>::new(u32::MAX).is_some());
    // A zero-width field holds nothing, and a width past the word is not a field of it.
    assert!(Bits::<0>::new(0).is_none());
    assert!(Bits::<33>::new(0).is_none());
};

/// A CORE IS AN INDEX AND A MASK IS `1 << index`, AND THE 33RD CORE IS NOT AN INDEX.
const _CORE_MASKS_ARE_ONE_SHIFTED_LEFT: () = {
    assert!(
        CoreMask::of_cores(&[
            CoreIdx::of_index(Bits::exactly(0)),
            CoreIdx::of_index(Bits::exactly(1)),
            CoreIdx::of_index(Bits::exactly(2)),
        ])
        .bits()
            == 0b111
    );
    // ⛔ THE TOP CORE IS THE ONE A NARROW CARRIER LOSES: `1 << 31` is the sign bit, not an overflow.
    assert!(CoreMask::of_cores(&[CoreIdx::of_index(Bits::exactly(31))]).bits() == 1 << 31);
    assert!(
        CoreMask::of_cores(&[]).bits() == 0,
        "no cores is not all cores"
    );
    // `dip.cpp:4594`'s `core_mask = 1 << coreNum`, for every core the field addresses.
    let mut at = 0;
    while at < 32 {
        assert!(CoreMask::of_cores(&[CoreIdx::of_index(Bits::exactly(at))]).bits() == 1u32 << at);
        at += 1;
    }
    // A core the mask cannot address does not exist as a `CoreIdx`.
    assert!(Bits::<{ CoreIdx::INDEX_BITS }>::new(32).is_none());
};

/// `lrf_idx` PACKS FOUR-BIT REGISTER NUMBERS, LOW NIBBLE FIRST — AND THE ONE-HOT READING IS ABSENT.
///
/// `dip.cpp:403-406`. The captured L3SU bundle's word 0 is `0x6420`, the register list `{0, 2, 4, 6}`.
const _LRF_INDEX_PACKS_NIBBLES_LOW_FIRST: () = {
    let observed = LrfIndexList::of_registers(&[
        LrfRegIdx::of_register(Bits::exactly(0)),
        LrfRegIdx::of_register(Bits::exactly(2)),
        LrfRegIdx::of_register(Bits::exactly(4)),
        LrfRegIdx::of_register(Bits::exactly(6)),
    ]);
    assert!(observed.bits() == 0x6420, "low nibble is entry 0");
    // ⛔ THE ONE-HOT READING OF {0, 3} IS `0b1001`, AND THIS FIELD IS NOT THAT. The nibble packing gives `0x30`;
    // the previous port's test asserted the one-hot value and thereby blessed a helper that built one-hot bits.
    let zero_and_three = LrfIndexList::of_registers(&[
        LrfRegIdx::of_register(Bits::exactly(0)),
        LrfRegIdx::of_register(Bits::exactly(3)),
    ]);
    assert!(zero_and_three.bits() == 0x30);
    // ⛔ THE TWO ENCODINGS DISAGREE ON THE SAME REGISTER SET, which is the whole reason the arch has to pick one:
    // {0, 3} is `0x30` packed and `0b1001` one-hot. Pinning both, so neither can be mistaken for universal.
    assert!(
        zero_and_three.bits() != 0b1001,
        "the PACKED reading of registers 0 and 3 is 0x30, not the one-hot 0b1001"
    );
    let one_hot_zero_and_three = LrfIndexList::of_registers_one_hot(&[
        OneHotRegIdx::of_register(0),
        OneHotRegIdx::of_register(3),
    ]);
    assert!(
        one_hot_zero_and_three.bits() == 0b1001,
        "SEN1P5's reading of registers 0 and 3 IS one-hot (dip.cpp:4245-4248)"
    );
    // ⭐ AND ONE-HOT REACHES REGISTER 31, which the nibble list cannot name at all — the packet format agreeing
    // with the SFP LRF holding 32 on that arch and 16 on this one.
    assert!(
        LrfIndexList::of_registers_one_hot(&[OneHotRegIdx::of_register(31)]).bits() == 1u64 << 31
    );
    // A register above 15 cannot be named at all — the packet format's own witness for an LRF depth of 16.
    assert!(Bits::<{ LrfRegIdx::BITS }>::new(16).is_none());
    // The list fills both words, and the WIDEST one must not lose its last entry off the top.
    assert!(LrfIndexList::ENTRIES == 16);
    let full = [LrfRegIdx::of_register(Bits::exactly(15)); LrfIndexList::ENTRIES];
    assert!(LrfIndexList::of_registers(&full).bits() == u64::MAX);
    let mut top_only = [LrfRegIdx::of_register(Bits::exactly(0)); LrfIndexList::ENTRIES];
    top_only[LrfIndexList::ENTRIES - 1] = LrfRegIdx::of_register(Bits::exactly(15));
    assert!(LrfIndexList::of_registers(&top_only).bits() == 0xF000_0000_0000_0000);
};

/// THE WIDEST VALUE OF EVERY FIELD, PACKED TOGETHER, LEAVES ITS NEIGHBOURS ALONE.
///
/// ⛔⛔ THIS IS THE INVARIANT A FIXTURE CANNOT SUPPLY. Every captured bundle is small — a handful of LRF flits, a
/// two-digit IBUFF count — so a too-wide field or a shift off by one agrees with all of them. Here every count is
/// at its MAXIMUM simultaneously, the whole word is pinned to one literal, and each field is then read back out:
/// an overflow into a neighbour changes a read-back, and a moved shift changes the literal.
const _MAXIMAL_FIELDS_DO_NOT_COLLIDE: () = {
    let header = Header {
        registers: LrfIndexList::of_registers(&[]),
        lrf_flits: LrfFlitCount::of_flits(Bits::exactly(31)),
        instr_flits: InstrFlitCount::of_flits(Bits::exactly(127)),
        // The largest mask any target produces: L3SU's bare `0x400`, whose bit lands at the very top of word 2.
        target: Target::L3(L3Unit::L3Su),
        core: CoreIdx::of_index(Bits::exactly(31)),
    };
    let words = header.slice().0;
    assert!(words[2] == 0x801F_FE00, "31 << 9 | 127 << 14 | 0x400 << 21");
    // Read each field back out of the word it shares.
    assert!((words[2] >> HeaderW2Field::PktLrf.shift()) & 0x1F == 31);
    assert!((words[2] >> HeaderW2Field::InstrFlits.shift()) & 0x7F == 127);
    assert!((words[2] >> HeaderW2Field::UnitMask.shift()) == 0x400);
    // ⭐ AND THE CORRECTION ROOM IS STILL ZERO: nothing below `pktLRF_flit` was touched by the counts above it,
    // which is what a too-wide `pktLRF_flit` or a misplaced shift would break.
    assert!(
        words[2] & 0x1FF == 0,
        "the correction room stays zero (dip.cpp:2667-2668)"
    );
    // ⛔⛔⛔ WORD 3 NAMES EXACTLY ONE CORE, AND THAT IS THE RULE `isPatchInit` TESTS.
    // This used to read `(1 << 31) | 1` — cores 0 and 31 — pinning the multi-core mask that made the whole
    // image a patch-init image with no patch headers. A single bit is what a REGULAR block's word 3 is.
    assert!(words[3] == 1u32 << 31, "core 31 alone, in word 3");
    // 🔒 THE not-a-power-of-two TEST IS IBM'S OWN (`patchinit.cpp:1268`): if this can ever fail, a block is
    // addressing more than one core and `reversePatchInit` is what reads the image.
    assert!(
        words[3] & words[3].wrapping_sub(1) == 0,
        "a block targets ONE core"
    );
    assert!(words[0] == 0 && words[1] == 0, "no registers named");

    // With the IBUFF count at zero the LRF field stands alone, so a bit of it anywhere else would show.
    let lrf_only = Header {
        registers: LrfIndexList::of_registers(&[]),
        lrf_flits: LrfFlitCount::of_flits(Bits::exactly(31)),
        instr_flits: InstrFlitCount::of_flits(Bits::exactly(0)),
        target: Target::L3(L3Unit::L3Su),
        core: CoreIdx::of_index(Bits::exactly(0)),
    };
    assert!(lrf_only.slice().0[2] == 0x3E00 | (0x400 << 21));
};

/// THE REGISTER LIST SPANS WORDS 0 AND 1, AND THE HALVES DO NOT SWAP.
///
/// ⛔ A 64-BIT VALUE SPLIT ACROSS TWO WORDS IS TWO CHANCES TO GET THE ORDER WRONG, and a bundle whose registers
/// all fit the low word cannot tell the two orders apart. This one names a register in the TOP nibble, so the
/// halves are distinguishable.
const _LRF_INDEX_LOW_HALF_IS_WORD_ZERO: () = {
    let mut registers = [LrfRegIdx::of_register(Bits::exactly(0)); LrfIndexList::ENTRIES];
    registers[0] = LrfRegIdx::of_register(Bits::exactly(1));
    registers[LrfIndexList::ENTRIES - 1] = LrfRegIdx::of_register(Bits::exactly(8));
    let header = Header {
        registers: LrfIndexList::of_registers(&registers),
        lrf_flits: LrfFlitCount::of_flits(Bits::exactly(0)),
        instr_flits: InstrFlitCount::of_flits(Bits::exactly(0)),
        target: Target::OnCorelets {
            unit: CoreletUnit::Sfp,
            corelets: Corelets::Both,
        },
        core: CoreIdx::of_index(Bits::exactly(0)),
    };
    let words = header.slice().0;
    assert!(
        words[0] == 0x0000_0001,
        "entry 0 is the LOW nibble of word 0"
    );
    assert!(
        words[1] == 0x8000_0000,
        "entry 15 is the HIGH nibble of word 1"
    );
    assert!(
        words[2] == 0x184 << HeaderW2Field::UnitMask.shift(),
        "SFPID | both corelets"
    );
    assert!(words[3] == 1);
};
