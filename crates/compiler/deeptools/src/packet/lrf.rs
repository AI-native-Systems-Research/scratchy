// SPDX-License-Identifier: Apache-2.0
//! THE LRF SLICES — a unit's REGISTER INITIALISATIONS, packed into 16-byte slices.
//!
//! # ⛔ THREE UNITS, THREE LAYOUTS, AND THEY ARE NOT ONE LAYOUT WITH A PARAMETER
//!
//! The units DISAGREE about what an LRF init slice is, and each disagreement is a different bit layout in
//! `dip.cpp` rather than a different argument to one packer:
//!
//! - an **L3** unit (L3LU/L3SU) packs **eight 16-bit** register values per slice ([`L3Lrf`], `L3LRFFlitSlice`,
//!   `dip.cpp:2672-2858`), its 32-bit EBR and 21-bit EAR values SPLIT across two of those lanes;
//! - the **LX/L0** units pack **four 24-bit** LRF values at a **3-byte stride**, then **two 16-bit** MVR values
//!   from byte 12 ([`LxLrfFlit`], `LXL0LRFFlitSlice`, `dip.cpp:3342-3394`, RCUDD1A);
//! - a **compute** unit (SFP/PE) packs **one 128-bit** constant per slice — the whole 16 bytes are ONE register
//!   ([`ComputeLrfValue`], `dip.cpp:3746-3779`).
//!
//! ⭐ SO THE CALLER PICKS THE TYPE, AND NOTHING HERE BRANCHES ON A UNIT TAG. A packer handed a unit would have to
//! answer "which of the three layouts" at run time from a value the call site already knows — that is dxp's
//! generality, and dxp needs it because it is handed one descriptor at a time. The three layouts differ in LANE
//! WIDTH, and a lane width is not a parameter of a bit layout: it *is* the bit layout. The `const` assertions at
//! the foot of this file pin the three against each other on the same bits, so no future "unify them" can pass.
//!
//! What every unit DOES share is how the HEADER names the slices it carries, and that encoding is
//! [`crate::packet::header`]'s — [`LrfRegIdx`], [`LrfIndexList`], [`LrfFlitCount`]. It is not restated here.
//!
//! ⚠️ ONE NAMING TENSION WORTH A REVIEWER's EYE: `header`'s type is called `LrfRegIdx` and documented as a
//! REGISTER number, which is what a compute unit's nibble is. An L3 unit's nibble is its INIT PACKET number
//! (`dip.cpp:2672-2858`; ten packets, each covering several registers of one file). One encoding, two meanings —
//! the header's own file would be the place to widen that name, and this file does not reach into it.
//!
//! # What this file does NOT decide
//!
//! ⛔ WHICH VALUE GOES IN WHICH REGISTER IS BRIDGE 3'S, NOT THIS FILE'S. The pre-set LBR partition values
//! (`dcgbeCodegen.cpp:148-159`: `lbrPartSize * i` for `i` in `0..4`, with `:159`'s `DT_CHECK` requiring a zero
//! among them) and a tensor base's conversion to a stick address are CONTENT the layer above chooses; this file
//! says where a chosen value's bits land. Keeping them apart is what stops values being read out of a captured
//! bundle: a bundle may CONFIRM a layout, it may never supply one.
//!
//! ⭐ THE LX/L0 GRID IS `LXL0LRFInitCompression` — [`lx_lrf_region`], [`l0_lrf_region`]. The initialised indices
//! in ASCENDING order, padded with UNUSED ones so the count fills whole flits, and the header nibble of slice `k`
//! is the `k`th index (`dip.cpp:1314-1400`, `:3548-3736`, `:3809-3820`).

use crate::isa::regfile::{Component as RegUnit, Presence, RegType as RegFile, info_of};
use crate::islands::progir::{FileDepth, RegIndex};
use crate::islands::init_packet::Slice;
use crate::packet::header::{Bits, LrfFlitCount, LrfIndexList, LrfRegIdx};

/// The depth of one register file of one unit, from the vendored table, as a CONSTANT rather than an `Option` a
/// caller must handle.
///
/// ⛔ THE `None` IS AN ARCH FACT AND SO IS THE REFUSAL. `regfile::depth` returns `None` when the unit does not
/// have the file at all — SEN1P5 drops `Lbr` from the L3 units entirely and widens `Ebr` to 16 — and that is not
/// a case this file can carry, because the packet grid below is RCUDD1A's. A `panic!` in a `const fn` is a build
/// error, which is the correct answer: the SEN1P5 grid must come from `dip.cpp`, not from a port of the RCUDD1A
/// one with a hole patched over.
const fn file_depth(unit: RegUnit, file: RegFile) -> FileDepth {
    match info_of(unit, file) {
        Presence::Present(info) => info.depth,
        Presence::Absent => panic!(
            "this arch's unit does not have this register file, so the RCUDD1A LRF packet grid \
             (dip.cpp:2672-2858) does not describe it. SEN1P5 drops the L3 Lbr file and widens Ebr to 16; its \
             grid must be read out of dip.cpp rather than derived from this one"
        ),
    }
}

/// ONE EMITTED LRF SLICE: the bits, and the header nibbles that name what is in them.
///
/// ⭐ ONE STRUCT RATHER THAN TWO PARALLEL LISTS. The header's `lrf_idx` describes the slices in ORDER, so a
/// separate list of names is a second place the region's length is decided — and the device applies slice `k` to
/// whatever nibbles `k` says. Paired, they cannot drift.
///
/// ⛔⛔ AND THE NAMES ARE A LIST BECAUSE ONE SLICE CAN CARRY SEVERAL. `dip.cpp:3813-3818` writes one nibble per
/// LRF INDEX while `pktLRF_flit` counts FLITS (`:3829`), and an LX flit holds four registers — so the two counts
/// are equal only for the L3 and compute layouts, where a slice really is one name. A single `name` field made the
/// LX grid unexpressible, and expressing it as one nibble per slice would leave the device applying slice 1's
/// bytes to register 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedLrfSlice {
    /// The header nibbles this slice contributes, in order: an L3 unit's init PACKET number, a compute unit's
    /// REGISTER number, or the four (LX) / eight (L0) register numbers the slice covers.
    pub names: Vec<LrfRegIdx>,
    /// The sixteen bytes.
    pub slice: Slice,
}

/// A UNIT's LRF REGION — the slices it carries, in emission order, each with its header nibble.
///
/// ⭐ THE HEADER's TWO FIELDS ARE METHODS, NOT STORED. `pkt_lrf` is the region's length and `lrf_idx` is its
/// names; the device walks the region by the first and applies it by the second, so a stored copy of either
/// would not corrupt one slice — it would misplace every region after it in the block. This mirrors
/// [`crate::packet::ibuff::Ibuff`], whose flit count is derived for the same reason.
///
/// A `Vec`, and that is not a compromise: this crate runs inside scratchy's proc-macro expansion, so the
/// allocation happens at COMPILE time. What is forbidden is searching, not allocating.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LrfRegion(Vec<NamedLrfSlice>);

impl LrfRegion {
    /// Append a slice the header names once — the L3 and compute layouts.
    fn push(&mut self, name: LrfRegIdx, slice: Slice) {
        self.push_covering(vec![name], slice);
    }

    /// Append a slice that carries SEVERAL header nibbles — the LX/L0 layouts.
    ///
    /// ⛔ THE BOUND IS ON THE NIBBLES, NOT THE SLICES. `lrf_idx` holds sixteen four-bit entries
    /// (`LrfIndexList::ENTRIES`), and four LX flits already use all of them; a seventeenth NAME would be described
    /// by a list that silently lost its tail. Bounding slices instead would accept four LX flits and a fifth.
    fn push_covering(&mut self, names: Vec<LrfRegIdx>, slice: Slice) {
        assert!(
            !names.is_empty(),
            "a slice the header cannot name is a slice the device cannot apply"
        );
        let already: usize = self.0.iter().map(|named| named.names.len()).sum();
        assert!(
            already + names.len() <= LrfIndexList::ENTRIES,
            "a unit's LRF region is named by sixteen four-bit nibbles of a 64-bit lrf_idx (dip.cpp:403-406, \
             :3813-3818), and this slice's names do not fit"
        );
        self.0.push(NamedLrfSlice { names, slice });
    }

    /// The region's slices with their names, in emission order.
    pub fn named(&self) -> &[NamedLrfSlice] {
        &self.0
    }

    /// Just the bits, in emission order — what bridge 4 lays into flits.
    pub fn slices(&self) -> Vec<Slice> {
        self.0.iter().map(|named| named.slice).collect()
    }

    /// The header's `lrf_idx` for this region — every slice's nibbles, in emission order.
    pub fn index_list(&self) -> LrfIndexList {
        let names: Vec<LrfRegIdx> = self
            .0
            .iter()
            .flat_map(|named| named.names.iter().copied())
            .collect();
        LrfIndexList::of_registers(&names)
    }

    /// The header's `pkt_lrf` for this region — a count of FLITS, not of names.
    ///
    /// Total: the nibble bound in `push_covering` caps the region at sixteen names and so at sixteen slices, while
    /// the field holds 31.
    pub fn flit_count(&self) -> LrfFlitCount {
        LrfFlitCount::of_flits(Bits::exactly(self.0.len() as u32))
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// LAYOUT 1 — the L3 units: EIGHT 16-BIT LANES per slice.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// ONE 16-BIT LANE of an L3 LRF flit slice.
///
/// ⛔ A LANE, NOT A REGISTER VALUE. An LBR/GTR/LAR value fills one lane, while an EBR value fills TWO (low then
/// high 16) and an EAR value fills two (low 16 then high 5). So "the value" and "the lane" are different
/// quantities, and the split belongs to the layout — which is why [`EbrValue::lanes`] and [`EarValue::lanes`] are
/// the only way to obtain a pair.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L3Lane(u16);

impl L3Lane {
    /// The lane's sixteen bits. Total: a lane is exactly as wide as its carrier.
    pub const fn of(bits: u16) -> Self {
        Self(bits)
    }

    /// The bits.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// WHICH LANE of the eight — `0..8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LaneIdx(u8);

impl LaneIdx {
    /// Lanes per L3 LRF flit slice: eight 16-bit values in 16 bytes.
    pub const COUNT: u8 = 8;

    /// The lane at `index`.
    pub const fn of(index: u8) -> Self {
        assert!(
            index < Self::COUNT,
            "an L3 LRF flit slice has eight 16-bit lanes; there is no ninth"
        );
        Self(index)
    }

    /// The index.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// A 14-BIT L3 ADDRESS REGISTER VALUE — what LBR, GTR and LAR hold.
///
/// ⛔ ONE TYPE FOR THREE FILES, ON PURPOSE. LBR, GTR and LAR are three *files*; the quantity each holds is the
/// same one — a 14-bit address in the unit's own space, packed identically into one lane. Three newtypes whose
/// conversions were all `Self(other.0)` would be one type wearing three names, and WHICH FILE a value belongs to
/// is said by [`L3AddrInit`]'s variant rather than by the value.
///
/// ⛔ AND THE REFERENCE MASKED. Its port wrote `v & 0x3FFF` at all three sites, so a 15-bit value silently became
/// a different, valid address 16384 away. Refusing at the door leaves the mask nothing to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr14(u16);

impl Addr14 {
    /// The field's width.
    pub const BITS: u32 = 14;
    /// The largest value the field holds.
    pub const MAX: u16 = (1 << Self::BITS) - 1;

    /// ⛔ NOT A MASK AND NOT A CLAMP. Either names a different address, which the load unit would then use.
    pub const fn of(value: u16) -> Self {
        assert!(
            value <= Self::MAX,
            "an L3 LBR/GTR/LAR field is 14 bits. The reference masked with 0x3FFF, which turns an out-of-range \
             address into a valid one pointing somewhere else"
        );
        Self(value)
    }

    /// The value.
    pub const fn get(self) -> u16 {
        self.0
    }

    /// The single lane this value occupies.
    pub const fn lane(self) -> L3Lane {
        L3Lane::of(self.0)
    }
}

/// AN EBR VALUE — a 32-bit stick address, split low-16 then high-16 across TWO lanes (`dip.cpp:2672-2858`,
/// packets 4 and 5).
///
/// A different quantity from [`Addr14`] and from [`EarValue`]: 32 bits where they are 14 and 21. The
/// stick-versus-byte distinction is bridge 3's (`islands::progir::ByteAddr`), and the width is what separates it
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EbrValue(u32);

impl EbrValue {
    /// The width of the PACKET FIELD: the whole of both lanes.
    pub const BITS: u32 = 32;

    /// The width of the REGISTER, which is NOT the packet's.
    ///
    /// ⛔⛔ ONE QUANTITY WITH TWO WIDTHS, AND THE WIDER ONE IS THE ONE THAT DOES NOT HOLD. `regInfoPerUnit` gives
    /// the L3's EBR `bitSize = 30` and `bitSizeInitPacket = 32` (`crate::isa::regfile::info_of`), so two lanes
    /// carry 32 bits into a register that keeps 30. A value between them is packed without complaint and then
    /// TRUNCATED by the hardware — the address is accepted, written, and wrong by 2^30 sticks.
    ///
    /// ⭐ TAKEN FROM THE REGISTER TABLE RATHER THAN WRITTEN AS 30, so the two cannot drift.
    pub const REGISTER_BITS: u32 = match crate::isa::regfile::info_of(
        crate::isa::regfile::Component::L3lu,
        crate::isa::regfile::RegType::Ebr,
    ) {
        crate::isa::regfile::Presence::Present(info) => info.bits.get() as u32,
        crate::isa::regfile::Presence::Absent => {
            panic!("an L3 unit has an EBR file (`regInfoPerUnit`)")
        }
    };

    /// The largest address the REGISTER holds.
    ///
    /// ⛔ THE SHIFT HAS TO BE GUARDED BECAUSE THE WIDTH IS ARCH-CONDITIONAL. SEN1P5 gives the EBR the full 32
    /// bits, and `1u32 << 32` overflows — which `#[deny(arithmetic_overflow)]` caught at compile time on that
    /// feature and would otherwise have been a wrong bound rather than an error.
    pub const MAX: u32 = match Self::REGISTER_BITS >= u32::BITS {
        true => u32::MAX,
        false => (1u32 << Self::REGISTER_BITS) - 1,
    };

    /// ⛔ BOUNDED BY THE REGISTER, NOT BY THE LANES. [`EarValue::of`] directly below already asserts its own
    /// 21 bits; this one accepted every `u32` because its bound was read off the packet, which is the wider of the
    /// two numbers and so the one that never refuses.
    ///
    /// ⛔ COMPARED IN `u64` BECAUSE THE BOUND IS SOMETIMES THE WHOLE TYPE. On SEN1P5 `MAX` IS `u32::MAX`, so
    /// `sticks <= Self::MAX` is a comparison against the type's own extreme — vacuously true there, which clippy
    /// denies as `absurd_extreme_comparisons` and which reads as a bound that cannot fire. Widening states the
    /// same rule in a form that is expressible on both arches rather than silencing the lint. `as u64` and not
    /// `u64::from` because `From` is not a const trait and this is a `const fn`; the cast is a widening one, so
    /// it cannot change a value.
    pub const fn of(sticks: u32) -> Self {
        assert!(
            sticks as u64 <= Self::MAX as u64,
            "an L3 EBR holds 30 bits and the init packet carries 32 — this address needs more than the REGISTER \
             has, so it would be packed intact and then truncated on load, naming memory 2^30 sticks away"
        );
        // ⛔⛔⛔ AND IT MUST NOT SET THE SIGN BIT OF THE `int32_t` THE MODEL KEEPS IT IN.
        //
        // On RCUDD1A the 30-bit bound above already settles this, so this arm is unreachable there. It is here
        // for the WIDER arch: SEN1P5 gives the EBR the full 32 bits, and `reg_file_EBR` is a signed `int` with
        // exactly ONE call site casting it back (`memoryElement.cpp:710-715`, whose own comment says so). A
        // value at or above `2^31` is therefore a correct offset at that site and a NEGATIVE number at every
        // other reader — about four billion granules from where it belongs, past `isWritableAddr`, which checks
        // the access MODE and not the range.
        //
        // ⛔ NOT A REFUSAL OF THE ARCH — a refusal of the VALUE. Every address below `2^31` sticks is emitted
        // exactly as before on both arches; only one that would be misread is stopped, and it is stopped at the
        // point it is minted rather than found later in a bus fence.
        assert!(
            !Self::CAN_REACH_THE_SIGN_BIT || sticks < 1u32 << 31,
            "this EBR sets the sign bit of the `int32_t` the model stores it in, and only one of its readers \
             casts it back (`memoryElement.cpp:710-715`) — every other reader would see a negative offset of \
             about four billion granules, which `isWritableAddr` does not range-check"
        );
        Self(sticks)
    }

    /// The value.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Its two lanes: low 16 first, then high 16 (`regVal[2i]` / `regVal[2i+1]`, `dip.cpp:2672-2858`).
    pub const fn lanes(self) -> (L3Lane, L3Lane) {
        (
            L3Lane::of((self.0 & 0xFFFF) as u16),
            L3Lane::of((self.0 >> 16) as u16),
        )
    }

    /// WHETHER AN EMITTED `EBR` CAN SET THE SIGN BIT OF THE `int32_t` THE MODEL STORES IT IN.
    ///
    /// ⛔⛔ THE `EBR` IS UNSIGNED IN THE TABLE AND SIGNED IN THE FILE, and the C++ says so itself: *"A
    /// static_cast to uint32_t is required when accessing reg_file_EBR. EBR contains unsigned 32 bit addr, but
    /// reg_file_EBR is defined using signed int to be consistent with other registers"*
    /// (`memoryElement.cpp:710-715`). ONE call site casts. So a value at or above `2^31` is a large positive
    /// offset at that site and a NEGATIVE number at every other reader of the same file — and multiplied by the
    /// granularity it is roughly four billion granules away from where it should be, with nothing between it and
    /// the bus: `isWritableAddr` checks the access MODE, not the range.
    ///
    /// ⭐ ON THIS ARCH IT CANNOT HAPPEN, AND THAT IS A CONSEQUENCE OF A BOUND ALREADY HERE. The RCUDD1A register
    /// is 30 bits (`regInfoPerUnit`), so [`Self::of`] already refuses anything above `2^30 - 1` — which is below
    /// `2^31`, so the sign bit is unreachable. The source is unsigned too ([`crate::alloc::Sticks`] is a `u64`),
    /// so there is no signed quantity anywhere on the path.
    ///
    /// ⛔ ON SEN1P5 IT IS REACHABLE: that arch gives the EBR the full 32 bits, so `of` admits values with the top
    /// bit set. This constant is arch-conditional rather than a flat `false` BECAUSE that difference is real, and
    /// stating it as a constant is what stops the RCUDD1A proof from being quietly assumed on the other arch.
    pub const CAN_REACH_THE_SIGN_BIT: bool = Self::MAX >= 1u32 << 31;
}

/// ⛔ ON RCUDD1A THE 30-BIT REGISTER IS WHAT KEEPS AN EMITTED `EBR` POSITIVE — asserted, not observed once.
///
/// This is a PROPERTY OF THIS ARCH, and it is stated here so that the next one does not inherit the conclusion
/// along with the code. If the table ever widened this file on RCUDD1A, [`EbrValue::of`] would start admitting
/// values that read as negative to every uncast reader of `reg_file_EBR`, and nothing else in this crate would
/// notice.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    !EbrValue::CAN_REACH_THE_SIGN_BIT,
    "an RCUDD1A EBR is 30 bits, so an emitted value cannot set the sign bit of the `int32_t` the model keeps it \
     in — see `EbrValue::CAN_REACH_THE_SIGN_BIT`"
);

/// AN EAR VALUE — an END ADDRESS, **21 bits**: low 16 in one lane, high **5** in the next (`dip.cpp:2672-2858`,
/// packets 6-9, `(v >> 16) & 0x1F`).
///
/// ⛔ TWENTY-ONE BITS, NOT THIRTY-TWO, AND THAT IS WHY IT IS NOT AN [`EbrValue`]. Both occupy two lanes and both
/// are addresses; the HIGH lane's width differs, so a value that is a legal EBR is a corrupt EAR. The reference
/// masked the high lane with `0x1F` and would have written an end address 2 MiB short without saying so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EarValue(u32);

impl EarValue {
    /// Bits in the high lane (`& 0x1F`).
    pub const HIGH_BITS: u32 = 5;
    /// The field's total width.
    pub const BITS: u32 = 16 + Self::HIGH_BITS;
    /// The largest end address expressible.
    pub const MAX: u32 = (1 << Self::BITS) - 1;

    /// ⛔ NOT A MASK. `(v >> 16) & 0x1F` on a wider value drops the top bits and yields an end address BELOW its
    /// start.
    pub const fn of(value: u32) -> Self {
        assert!(
            value <= Self::MAX,
            "an L3 EAR is 21 bits — 16 in the low lane and 5 in the high one. The reference masked the high lane \
             with 0x1F, which silently truncates an end address to one below the start"
        );
        Self(value)
    }

    /// The value.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Its two lanes: low 16, then the high 5.
    pub const fn lanes(self) -> (L3Lane, L3Lane) {
        (
            L3Lane::of((self.0 & 0xFFFF) as u16),
            L3Lane::of((self.0 >> 16) as u16),
        )
    }
}

/// THE EIGHT LANES OF ONE L3 LRF FLIT SLICE, and WHETHER ANY WAS WRITTEN.
///
/// ⛔⛔ WRITTEN-NESS IS NOT DERIVABLE FROM THE VALUE, AND THAT IS THE BUG THIS SHAPE PREVENTS. The pre-set LBR
/// values are `lbrPartSize * i` (`dcgbeCodegen.cpp:148-159`), so **LBR 0 is legitimately ZERO** — and
/// `dcgbeCodegen.cpp:159`'s `DT_CHECK(valueToLBRIdx.count(0))` says a zero LBR is *required*. A packet emitted
/// only when some lane is non-zero would DROP that packet, and dropping it does not lose one register: every
/// later packet's nibble shifts down a position, so the device applies each remaining slice to the WRONG PACKET
/// NUMBER. The reference tracked this as a separate `present` flag; here it is in the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L3FlitLanes {
    lanes: [L3Lane; LaneIdx::COUNT as usize],
    any_written: bool,
}

impl L3FlitLanes {
    /// A slice with no register set — which is NOT the same as a slice of zeros.
    pub const fn none() -> Self {
        Self {
            lanes: [L3Lane::of(0); LaneIdx::COUNT as usize],
            any_written: false,
        }
    }

    /// Write one lane. Marks the slice written even when the value is zero.
    pub const fn set(mut self, at: LaneIdx, value: L3Lane) -> Self {
        self.lanes[at.get() as usize] = value;
        self.any_written = true;
        self
    }

    /// Whether this packet carries any set register — the reference's `present`.
    pub const fn any_written(self) -> bool {
        self.any_written
    }

    /// Pack the lanes: `word[k] = (lane[2k+1] << 16) | lane[2k]` (`L3LRFFlitSlice`, `dip.cpp:2672-2858`).
    pub const fn pack(self) -> Slice {
        let mut words = [0u32; 4];
        let mut word = 0;
        while word < 4 {
            words[word] =
                ((self.lanes[2 * word + 1].get() as u32) << 16) | self.lanes[2 * word].get() as u32;
            word += 1;
        }
        Slice(words)
    }
}

/// WHICH OF AN L3 UNIT's ADDRESS FILES a register belongs to — the five the LRF packet grid covers.
///
/// ⭐ A CLOSED SET AND A COMPLETE MATCH, so a sixth file added to [`RegFile`] cannot silently be left out of the
/// grid. `RegFile` also spells `Lrf`, `Spr`, `Jcr` and the PT's files; an L3 unit has no `Lrf` and none of the
/// PT's, `Spr` is a different slice KIND (`packet::spr`), and `Jcr` is `initializable: false` in the vendored
/// table — so those absences are facts rather than omissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum L3AddrFile {
    /// Line-base register.
    Lbr,
    /// Gather register.
    Gtr,
    /// Line-address register.
    Lar,
    /// Element-base register — where a tensor's segment base lands.
    Ebr,
    /// End-address register.
    Ear,
}

impl L3AddrFile {
    /// The vendored table's name for this file.
    pub const fn reg_file(self) -> RegFile {
        match self {
            Self::Lbr => RegFile::Lbr,
            Self::Gtr => RegFile::Gtr,
            Self::Lar => RegFile::Lar,
            Self::Ebr => RegFile::Ebr,
            Self::Ear => RegFile::Ear,
        }
    }

    /// How many registers an L3 unit has in this file — `regfile::depth`, the crate's one source.
    ///
    /// ⛔ NOT PARAMETERISED BY L3LU-VERSUS-L3SU, AND THE COMPILER CHECKS THAT. The vendored table gives the two
    /// units the SAME rows (`regfile.rs` matches `L3lu | L3su` together), so a unit tag threaded through this
    /// file would be a field nothing reads — dxp carries one because it holds a table for every chip. The
    /// `file_depth(L3lu, ..) == file_depth(L3su, ..)` assertions at the foot of this file are what LICENSE the
    /// collapse: if the table ever separates them, the build fails there rather than quietly answering L3LU's
    /// depth for an L3SU register.
    pub const fn depth(self) -> FileDepth {
        file_depth(RegUnit::L3lu, self.reg_file())
    }

    /// The one door to a register OF THIS FILE: [`RegIndex::of_file`], bounded by this file's own depth.
    ///
    /// ⛔ THE FILE's DEPTH, NEVER A CARRIER's WIDTH. `u8::try_from` fails at 256 while the EBR file holds 8 — the
    /// nuked crate bounded EBR indices that way and handed out registers the unit does not have (`6d86e23c`).
    /// `RegIndex::of_file` is the crate's only register constructor precisely so a depth must be named.
    pub const fn reg(self, index: u8) -> RegIndex {
        RegIndex::of_file(index, self.depth())
    }
}

/// ONE REGISTER INITIALISATION for an L3 unit — the value, and by its variant the FILE.
///
/// ⭐ THE VARIANT IS THE FILE, so a value cannot be filed under the wrong one: an [`EarValue`] is not expressible
/// as an `Ebr`, which matters because the two differ only in the width of their high lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3AddrInit {
    /// `LBR[reg]`.
    Lbr(Addr14),
    /// `GTR[reg]`.
    Gtr(Addr14),
    /// `LAR[reg]`.
    Lar(Addr14),
    /// `EBR[reg]`.
    Ebr(EbrValue),
    /// `EAR[reg]`.
    Ear(EarValue),
}

impl L3AddrInit {
    /// Which file this initialisation writes.
    pub const fn file(self) -> L3AddrFile {
        match self {
            Self::Lbr(_) => L3AddrFile::Lbr,
            Self::Gtr(_) => L3AddrFile::Gtr,
            Self::Lar(_) => L3AddrFile::Lar,
            Self::Ebr(_) => L3AddrFile::Ebr,
            Self::Ear(_) => L3AddrFile::Ear,
        }
    }
}

/// AN L3 LRF INIT PACKET — the ten packet numbers of the RCUDD1A grid (`dip.cpp:2672-2858`).
///
/// ⭐ TEN NAMED PACKETS, NOT `0..10`. The reference matched `0 | 1`, `2 | 3`, `4 | 5` and then a `_` arm for
/// 6-9 — and a `_` arm is what makes an eleventh packet number readable as an EAR packet. Enumerated, the grid's
/// shape is in the type, and each packet's base register is arithmetic on the variant instead of
/// `if pkt == 2 { 0 } else { 8 }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum L3LrfPacket {
    /// Packet 0: `LBR[0..4]` in lanes 0-3, `GTR[0..4]` in lanes 4-7.
    LbrGtrLow,
    /// Packet 1: `LBR[4..8]` in lanes 0-3, `GTR[4..8]` in lanes 4-7.
    LbrGtrHigh,
    /// Packet 2: `LAR[0..8]`, one per lane.
    LarLow,
    /// Packet 3: `LAR[8..16]`, one per lane.
    LarHigh,
    /// Packet 4: `EBR[0..4]`, low/high 16 in lanes `2i`/`2i+1`.
    EbrLow,
    /// Packet 5: `EBR[4..8]`, low/high 16 in lanes `2i`/`2i+1`.
    EbrHigh,
    /// Packet 6: `EAR[0..4]`, low 16 / high 5 in lanes `2i`/`2i+1`.
    Ear0,
    /// Packet 7: `EAR[4..8]`.
    Ear1,
    /// Packet 8: `EAR[8..12]`.
    Ear2,
    /// Packet 9: `EAR[12..16]`.
    Ear3,
}

impl L3LrfPacket {
    /// Every packet, in emission order — the order the reference's `for pkt in 0..10` walked.
    pub const ALL: [Self; 10] = [
        Self::LbrGtrLow,
        Self::LbrGtrHigh,
        Self::LarLow,
        Self::LarHigh,
        Self::EbrLow,
        Self::EbrHigh,
        Self::Ear0,
        Self::Ear1,
        Self::Ear2,
        Self::Ear3,
    ];

    /// How many LBR — and how many GTR — a `LbrGtr*` packet covers: four of each fill the eight lanes.
    pub const LBR_PER_PACKET: u8 = LaneIdx::COUNT / 2;
    /// How many LAR a LAR packet covers: one per lane.
    pub const LAR_PER_PACKET: u8 = LaneIdx::COUNT;
    /// How many EBR or EAR such a packet covers: each of those values takes two lanes.
    pub const WIDE_PER_PACKET: u8 = LaneIdx::COUNT / 2;

    /// The packet number, which is what the header's nibble holds for an L3 unit.
    pub const fn number(self) -> LrfRegIdx {
        LrfRegIdx::of_register(Bits::exactly(match self {
            Self::LbrGtrLow => 0,
            Self::LbrGtrHigh => 1,
            Self::LarLow => 2,
            Self::LarHigh => 3,
            Self::EbrLow => 4,
            Self::EbrHigh => 5,
            Self::Ear0 => 6,
            Self::Ear1 => 7,
            Self::Ear2 => 8,
            Self::Ear3 => 9,
        }))
    }

    /// The first register index this packet covers, in each file it carries.
    pub const fn base(self) -> u8 {
        match self {
            Self::LbrGtrLow => 0,
            Self::LbrGtrHigh => Self::LBR_PER_PACKET,
            Self::LarLow => 0,
            Self::LarHigh => Self::LAR_PER_PACKET,
            Self::EbrLow => 0,
            Self::EbrHigh => Self::WIDE_PER_PACKET,
            Self::Ear0 => 0,
            Self::Ear1 => Self::WIDE_PER_PACKET,
            Self::Ear2 => 2 * Self::WIDE_PER_PACKET,
            Self::Ear3 => 3 * Self::WIDE_PER_PACKET,
        }
    }
}

/// AN L3 UNIT's ADDRESS-REGISTER INITIALISATIONS — L3LU or L3SU, whose files the vendored table gives
/// identically.
///
/// Sparse by register: only the packets carrying a set register are emitted, and each emitted packet's NUMBER
/// becomes the next header nibble (`dip.cpp:2672-2858`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct L3Lrf {
    lbr: std::collections::BTreeMap<RegIndex, Addr14>,
    gtr: std::collections::BTreeMap<RegIndex, Addr14>,
    lar: std::collections::BTreeMap<RegIndex, Addr14>,
    ebr: std::collections::BTreeMap<RegIndex, EbrValue>,
    ear: std::collections::BTreeMap<RegIndex, EarValue>,
}

impl L3Lrf {
    /// Initialise one register.
    ///
    /// ⛔ A SECOND VALUE FOR ONE REGISTER IS A CONTRADICTION, NOT AN OVERWRITE. The reference's `insert` kept the
    /// last one silently; two callers each certain of a different tensor base is a defect upstream, and the
    /// register holds one value, so this refuses rather than choosing.
    ///
    /// `reg` must lie inside the file the VALUE names. It is minted by [`L3AddrFile::reg`], which bounds it by
    /// that file's depth, and this re-checks the pairing because a `RegIndex` does not carry which file produced
    /// it — EBR holds 8 registers and LAR 16, so `LAR[12]` is a `RegIndex` that is not an EBR register.
    pub fn set(&mut self, reg: RegIndex, init: L3AddrInit) {
        assert!(
            reg.get() < init.file().depth().get(),
            "this register lies outside the file this value belongs to. The L3 files have different depths — EBR \
             holds 8 and LAR 16 on RCUDD1A — so a register minted for one file is not automatically inside \
             another"
        );
        match init {
            L3AddrInit::Lbr(value) => Self::insert_once(&mut self.lbr, reg, value),
            L3AddrInit::Gtr(value) => Self::insert_once(&mut self.gtr, reg, value),
            L3AddrInit::Lar(value) => Self::insert_once(&mut self.lar, reg, value),
            L3AddrInit::Ebr(value) => Self::insert_once(&mut self.ebr, reg, value),
            L3AddrInit::Ear(value) => Self::insert_once(&mut self.ear, reg, value),
        }
    }

    /// One register, one value.
    fn insert_once<V>(file: &mut std::collections::BTreeMap<RegIndex, V>, reg: RegIndex, value: V) {
        assert!(
            file.insert(reg, value).is_none(),
            "this register was already initialised. A register holds one value, so a second is a contradiction \
             upstream rather than a later value to prefer"
        );
    }

    /// The lanes packet `pkt` carries.
    fn lanes_of(&self, pkt: L3LrfPacket) -> L3FlitLanes {
        let mut lanes = L3FlitLanes::none();
        let base = pkt.base();
        match pkt {
            // Packets 0-1: four LBR in the low lanes, four GTR in the high ones.
            L3LrfPacket::LbrGtrLow | L3LrfPacket::LbrGtrHigh => {
                for e in 0..L3LrfPacket::LBR_PER_PACKET {
                    if let Some(value) = self.lbr.get(&L3AddrFile::Lbr.reg(base + e)) {
                        lanes = lanes.set(LaneIdx::of(e), value.lane());
                    }
                    if let Some(value) = self.gtr.get(&L3AddrFile::Gtr.reg(base + e)) {
                        lanes =
                            lanes.set(LaneIdx::of(L3LrfPacket::LBR_PER_PACKET + e), value.lane());
                    }
                }
            }
            // Packets 2-3: eight LAR, one per lane.
            L3LrfPacket::LarLow | L3LrfPacket::LarHigh => {
                for e in 0..L3LrfPacket::LAR_PER_PACKET {
                    if let Some(value) = self.lar.get(&L3AddrFile::Lar.reg(base + e)) {
                        lanes = lanes.set(LaneIdx::of(e), value.lane());
                    }
                }
            }
            // Packets 4-5: four EBR, each across two lanes.
            L3LrfPacket::EbrLow | L3LrfPacket::EbrHigh => {
                for i in 0..L3LrfPacket::WIDE_PER_PACKET {
                    if let Some(value) = self.ebr.get(&L3AddrFile::Ebr.reg(base + i)) {
                        let (low, high) = value.lanes();
                        lanes = lanes
                            .set(LaneIdx::of(2 * i), low)
                            .set(LaneIdx::of(2 * i + 1), high);
                    }
                }
            }
            // Packets 6-9: four EAR, each low 16 / high 5.
            L3LrfPacket::Ear0 | L3LrfPacket::Ear1 | L3LrfPacket::Ear2 | L3LrfPacket::Ear3 => {
                for i in 0..L3LrfPacket::WIDE_PER_PACKET {
                    if let Some(value) = self.ear.get(&L3AddrFile::Ear.reg(base + i)) {
                        let (low, high) = value.lanes();
                        lanes = lanes
                            .set(LaneIdx::of(2 * i), low)
                            .set(LaneIdx::of(2 * i + 1), high);
                    }
                }
            }
        }
        lanes
    }

    /// The unit's LRF region: every packet that carries a set register, named by its packet number.
    pub fn build(&self) -> LrfRegion {
        let mut region = LrfRegion::default();
        for pkt in L3LrfPacket::ALL {
            let lanes = self.lanes_of(pkt);
            if lanes.any_written() {
                region.push(pkt.number(), lanes.pack());
            }
        }
        region
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// LAYOUT 2 — the LX/L0 units: FOUR 24-BIT VALUES AT A 3-BYTE STRIDE, then TWO 16-BIT MVR VALUES.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// AN LX/L0 LRF VALUE — **24 bits**, because the layout gives it **three bytes** (`LXL0LRFFlitSlice`,
/// `dip.cpp:3342-3394`, RCUDD1A).
///
/// ⛔ TWENTY-FOUR, NOT THIRTY-TWO, AND THE REFERENCE's CARRIER SAID OTHERWISE. It took `[u32; 4]` and wrote only
/// `val & 0xff`, `val >> 8` and `val >> 16` — so bits 24-31 were dropped without a word and the value written
/// was the intended one modulo 16 MiB. A 24-bit newtype makes the unwritable bits unrepresentable.
///
/// ⛔ AND A 3-BYTE STRIDE IS NOT A 4-BYTE ONE. Value `i` starts at byte `3i`, so values 1 and 2 STRADDLE the
/// slice's `u32` word boundaries. That is why this layout cannot be [`L3FlitLanes`] with wider lanes, and why
/// nothing here shares a packer with the L3 units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct LxLrfValue(u32);

impl LxLrfValue {
    /// The field's width in bits.
    pub const BITS: u32 = 24;
    /// The field's width in bytes — the stride between consecutive values.
    pub const BYTES: usize = 3;
    /// The largest value three bytes hold.
    pub const MAX: u32 = (1 << Self::BITS) - 1;

    /// ⛔ NOT A TRUNCATION. The reference dropped bits 24-31 silently.
    pub const fn of(value: u32) -> Self {
        assert!(
            value <= Self::MAX,
            "an LX/L0 LRF value occupies THREE BYTES of the flit slice (dip.cpp:3342-3394), so bits 24-31 have \
             nowhere to go. The reference took a u32 and wrote three bytes of it"
        );
        Self(value)
    }

    /// A slot this flit does not initialise, written as zero — what the reference's `[u32; 4]` argument held for
    /// every unsupplied entry.
    ///
    /// ⚠️ WHETHER WRITING ZERO IS CORRECT IS THE GRID's QUESTION, NOT THIS TYPE's. The L3 layout emits a packet
    /// only when a register in it was set, and nothing in the reference states the equivalent rule for LX — see
    /// this module's header. So this says "these three bytes are zero", never "this register is unset".
    pub const fn zero() -> Self {
        Self(0)
    }

    /// The value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// AN MVR VALUE — the LXLU's move register, **16 bits**, packed from byte 12 of the slice
/// (`dip.cpp:3342-3394`).
///
/// A distinct quantity from [`LxLrfValue`]: different file, different width, and it sits in a tail of the slice
/// the LRF values cannot reach. The reference passed `([u32; 4], [u16; 2])`, where the two arrays were kept apart
/// only by their element type — separate newtypes forbid the positional swap outright.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MvrValue(u16);

impl MvrValue {
    /// The field's width in bytes.
    pub const BYTES: usize = 2;

    /// Total: every 16-bit value fits its two bytes.
    pub const fn of(value: u16) -> Self {
        Self(value)
    }

    /// The value.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// ONE LX/L0 LRF FLIT SLICE: four 24-bit LRF values, then two 16-bit MVR values (`LXL0LRFFlitSlice`,
/// `dip.cpp:3342-3394`, RCUDD1A).
///
/// ⛔ NO `build`, AND THAT ABSENCE IS DELIBERATE — see this module's header. Which four LRF registers and which
/// two MVR registers a given LX init packet covers is not stated by the reference, and a guessed grid produces a
/// file that assembles and initialises the wrong registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LxLrfFlit {
    lrf: [LxLrfValue; Self::LRF_PER_FLIT],
    mvr: [MvrValue; Self::MVR_PER_FLIT],
}

impl LxLrfFlit {
    /// How many LRF values one flit slice carries.
    pub const LRF_PER_FLIT: usize = 4;
    /// How many MVR values follow them.
    pub const MVR_PER_FLIT: usize = 2;
    /// The byte the MVR values begin at — ARITHMETIC, not the reference's literal 12. Writing both a product and
    /// its factors is how the two come to disagree.
    pub const MVR_AT: usize = Self::LRF_PER_FLIT * LxLrfValue::BYTES;

    /// The flit's contents.
    pub const fn new(
        lrf: [LxLrfValue; Self::LRF_PER_FLIT],
        mvr: [MvrValue; Self::MVR_PER_FLIT],
    ) -> Self {
        Self { lrf, mvr }
    }

    /// Pack the 16 bytes, little-endian into the four slice words.
    pub const fn pack(self) -> Slice {
        let mut bytes = [0u8; Slice::BYTES];
        let mut i = 0;
        while i < Self::LRF_PER_FLIT {
            let value = self.lrf[i].get();
            bytes[i * LxLrfValue::BYTES] = (value & 0xff) as u8;
            bytes[i * LxLrfValue::BYTES + 1] = ((value >> 8) & 0xff) as u8;
            // Bits 24-31 do not exist — `LxLrfValue` refuses them — so these three bytes are the whole value.
            bytes[i * LxLrfValue::BYTES + 2] = ((value >> 16) & 0xff) as u8;
            i += 1;
        }
        let mut e = 0;
        while e < Self::MVR_PER_FLIT {
            let value = self.mvr[e].get();
            bytes[Self::MVR_AT + e * MvrValue::BYTES] = (value & 0xff) as u8;
            bytes[Self::MVR_AT + e * MvrValue::BYTES + 1] = ((value >> 8) & 0xff) as u8;
            e += 1;
        }
        Slice([
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        ])
    }
}

/// AN L0 LRF VALUE — **two bytes** of the slice, masked to **10 bits** on RCUDD1A and 12 above
/// (`dip.cpp:3660-3682`).
///
/// ⛔ A DIFFERENT WIDTH FROM THE LX ONE, ON THE SAME FILE NAME. Eight values share the sixteen bytes, so the
/// stride is two rather than three, and the C++ masks with `0x3ff` up to RCUDD1A and `0xfff` above
/// (`:3676-3679`) — narrower than the two bytes it writes. An L0 address is therefore not an LX address with a
/// smaller number in it: `addressGranularityScalePerUnit[{L0SU, L0}] = numPTRows` (`sysdef.cpp:514`) scales it by
/// the PT row count, which is what makes ten bits enough.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct L0LrfValue(u16);

impl L0LrfValue {
    /// The field's width in bytes — the stride between consecutive values.
    pub const BYTES: usize = 2;

    /// What the C++ masks an L0 LRF value with — `0x3ff` up to RCUDD1A, `0xfff` above (`dip.cpp:3676-3679`).
    pub const MASK: u16 = match cfg!(feature = "arch-sen1p5") {
        true => 0xfff,
        false => 0x3ff,
    };

    /// ⛔ NOT A TRUNCATION: the C++ ANDs, so a wider value silently loses its top bits and addresses the wrong
    /// place in L0.
    pub const fn of(value: u16) -> Self {
        assert!(
            value <= Self::MASK,
            "an L0 LRF value is masked to this arch's width (dip.cpp:3676-3679); a wider one addresses \
             somewhere else in L0"
        );
        Self(value)
    }

    /// A slot this flit does not initialise.
    pub const fn zero() -> Self {
        Self(0)
    }

    /// The value.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// ONE L0 LRF FLIT SLICE: **eight** values, two bytes each (`dip.cpp:3658-3682`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L0LrfFlit {
    lrf: [L0LrfValue; Self::LRF_PER_FLIT],
}

impl L0LrfFlit {
    /// How many LRF values one L0 flit slice carries — `numLRFperFlitSlice` for an L0 unit (`dip.cpp:3536-3546`).
    pub const LRF_PER_FLIT: usize = 8;

    /// The flit's contents.
    pub const fn new(lrf: [L0LrfValue; Self::LRF_PER_FLIT]) -> Self {
        Self { lrf }
    }

    /// Pack the 16 bytes, little-endian into the four slice words.
    ///
    /// ⭐ NO MVR TAIL. The MVR values are written only on the LX branch (`dip.cpp:3752-3766`); an L0 flit is eight
    /// LRF values and nothing else, which is why this is its own type rather than [`LxLrfFlit`] with a parameter.
    pub const fn pack(self) -> Slice {
        let mut bytes = [0u8; Slice::BYTES];
        let mut i = 0;
        while i < Self::LRF_PER_FLIT {
            let value = self.lrf[i].get();
            bytes[i * L0LrfValue::BYTES] = (value & 0xff) as u8;
            bytes[i * L0LrfValue::BYTES + 1] = ((value >> 8) & 0xff) as u8;
            i += 1;
        }
        Slice([
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        ])
    }
}

/// HOW MANY LRF INDICES THE LX/L0 INIT PACKET COVERS — `MAXL0LXL3LRF` (`dip.h:49`).
///
/// ⭐ SIXTEEN, WHICH IS WHY THE HEADER CAN NAME THEM ALL. The nibble list gives each slice four bits
/// (`dip.cpp:3813-3818`: `elem & 0xF`), so a seventeenth index would alias onto the first — and the census walks
/// exactly `0..MAXL0LXL3LRF`, so it cannot arise.
pub const LX_L0_LRF_INDICES: usize = 16;

/// WHICH LRF INDICES AN LX/L0 INIT PACKET NAMES, IN ORDER — `LXL0LRFInitCompression` (`dip.cpp:1314-1400`).
///
/// The initialised indices ascending, then UNUSED indices ascending until the count fills whole flits and covers
/// the MVR values the flits have to carry.
///
/// ⛔⛔ AND THEN SORTED BY THE CROSS-CORE UNIQUE-VALUE COUNT, WHICH IS NOT A NO-OP. `:1325` walks `idx` from 0
/// upward and pushes each initialised one, and the final `std::sort` is by `std::get<1>` — the number of DISTINCT
/// VALUES that index holds ACROSS `initcores` (`arrangelrfidx`, `:1304-1312`), ascending, so an index that agrees
/// on every core sorts before one that varies and the flits carrying the agreeing registers can be shared.
///
/// ⛔ THIS COMMENT USED TO CLAIM THE SORT COULD NOT REORDER ANYTHING, on the premise that "this crate emits one
/// program for one core mask, so every entry's count is one". The premise is false — a staging transfer's address
/// IS the per-core HBM fold — and `dip_standalone`'s own `init.txt` says so: the LxLu header reads `0x2310`, index
/// order `[0,1,3,2]`, where the ascending list gives `0x3210`. The counts come from
/// [`crate::reginit::RegisterInits::lrf_unique_counts`].
///
/// ⭐ THE TIE-BREAK IS THE INDEX, and that is a reading of `std::sort` rather than a choice: it is UNSTABLE, but
/// libstdc++ sorts a range this short with insertion sort, which preserves the census's ascending order. A padding
/// entry counts as ONE — `make_tuple(idxToUse, 1, 0)` (`:1390`) — so it sorts among the agreeing indices.
///
/// ⛔ AND THE PADDING IS NOT SLACK. `:1352-1394` requires the count to be a multiple of the flit width, because
/// the packing loop reads exactly `numLRFperFlitSlice` entries per flit and `DT_CHECK`s the multiple (`:3553`).
/// The padding indices are UNUSED ones at value zero, so they initialise registers no instruction reads.
///
/// HOW MANY LRF ENTRIES ONE FLIT SLICE HOLDS — `numLRFperFlitSlice`, a property of the FLIT LAYOUT.
///
/// ⛔⛔ A NEWTYPE BECAUSE THE NUMBER BESIDE IT IS ALSO A `usize` AND MEANS SOMETHING ELSE ENTIRELY. The two
/// were adjacent bare parameters of [`lx_l0_index_list`]: this one divides the index list into flits, and
/// [`MvrIndices`] raises a FLOOR under how many flits there must be. Swapping them changes both the length of
/// the emitted region and the floor it is checked against, and every value involved is small and plausible —
/// `LRF_PER_FLIT` is 8 and an MVR count of 8 is ordinary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LrfPerFlit(usize);

impl LrfPerFlit {
    pub const fn of(count: usize) -> Self {
        Self(count)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// THE MVR INDEX COUNT — one past the LARGEST MVR index, which is what the floor below is computed from.
///
/// ⛔ NOT A NUMBER OF FLITS AND NOT A NUMBER OF ENTRIES PER FLIT — see [`LrfPerFlit`]. The L0 has no MVR at
/// all and passes zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvrIndices(usize);

impl MvrIndices {
    pub const fn of(count: usize) -> Self {
        Self(count)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// ⛔ THE MVR FLOOR IS THE C++'s AND IT IS A FLOOR, NOT THE ANSWER. `reqdMvrFlits = ceil(maxMvrIdx / 2.0)`
/// (`:1379`) is one short whenever the largest MVR index is EVEN — index 2 lives in flit 1 and so needs two flits
/// — but it only ever raises the LRF count, and the MVR values are written into whatever flits exist
/// (`:3752-3766`). Ported as written: computing a tighter bound would emit a flit dxp does not.
fn lx_l0_index_list(
    used: &[RegIndex],
    per_flit: LrfPerFlit,
    mvr_indices: MvrIndices,
    unique: &std::collections::BTreeMap<RegIndex, usize>,
) -> Vec<RegIndex> {
    let (per_flit, mvr_indices) = (per_flit.get(), mvr_indices.get());
    let depth = LX_L0_LRF_INDICES;
    let mut ordered: Vec<RegIndex> = used.to_vec();
    ordered.sort();
    ordered.dedup();
    assert!(
        ordered.iter().all(|reg| usize::from(reg.get()) < depth),
        "an LX/L0 LRF index is one of {depth} (`dip.h:49`); the header names it in a nibble"
    );
    // `ceil(maxMvrIdx / 2.0)` flits, from the LARGEST index rather than the count — `dip.cpp:1370-1380`.
    let mvr_flits = match mvr_indices {
        0 => 0,
        largest_plus_one => (largest_plus_one - 1).div_ceil(2),
    };
    let floor = per_flit * mvr_flits;
    let whole = ordered.len().next_multiple_of(per_flit);
    let target = whole.max(floor.next_multiple_of(per_flit));
    let spare: Vec<RegIndex> = (0..depth)
        .map(|index| RegIndex::of_file(index as u8, FileDepth::of_packet_bound(depth as u8)))
        .filter(|index| !ordered.contains(index))
        .collect();
    let mut unused = spare.into_iter();
    while ordered.len() < target {
        ordered.push(unused.next().unwrap_or_else(|| {
            panic!(
                "the LX/L0 init packet needs {target} of {depth} LRF indices to fill whole flits and there are \
                 not that many; `DT_CHECK(unusedLRFindices.size() >= numIdxNeeded)` (`dip.cpp:1387`)"
            )
        }));
    }
    // `std::sort(..., arrangelrfidx)` (`:1397-1398`) — by the cross-core unique-value count ALONE.
    //
    // ⛔⛔ AND TIES KEEP THE PUSH ORDER, WHICH IS NOT ASCENDING. The list is the initialised indices ascending and
    // THEN the padding ones appended (`:1386-1394`), so a padding index sorts AFTER an initialised index of the
    // same count even when its number is smaller. Measured: our LxLu names `{0, 1, 3}` and pads with `2`, so the
    // list is `[0, 1, 3, 2]` and the header reads `0x2310` — which is what `dip_standalone` writes. Adding the
    // index as a tie-break re-sorted the padding INTO the run and gave `0x3210`.
    //
    // ⛔ THE PADDING INDEX'S KEY IS THE C++'s OWN LITERAL, NOT A NUMBER CHOSEN TO MATCH THE DIFF.
    // `auto mytuple = std::make_tuple(idxToUse, 1, 0);` (`dip.cpp:1394`) is how an unused index is pushed, and
    // `arrangelrfidx` sorts on `std::get<1>` — that middle element. Our `unique` map holds only the indices a
    // core initialises, so a padding one is absent and takes the same value. Written as a bare `1` it read as a
    // number fitted to the measurement above, which is the one thing this function cannot afford.
    //
    // 🛑 AND THE TIE ORDER IS THE C++'s INCIDENTAL BEHAVIOUR RATHER THAN ITS GUARANTEE. `arrangelrfidx` is
    // `val1 < val2` with no tie-break (`dip.cpp:1308-1316`) and `std::sort` is UNSTABLE, so equal counts are
    // formally unordered; a range this short — an LRF file is 16 deep — is insertion-sorted in practice and so
    // keeps the push order, which is what the measured `0x2310` agrees with. `sort_by_key` is stable, so this
    // port is DETERMINISTIC where the reference is only predictable.
    //
    // ⭐⭐ AND THE ASSUMPTION IS NOW MEASURED AT SCALE, not inferred from one header. `dip_standalone -p` run
    // over this crate's own senprogs produces a BYTE-IDENTICAL image for 1141 of 1198 ops (the rest differ only
    // in the sentinel bit and the fp8/fp16 SPR, both artefacts of how dip is invoked). Every LRF index list in
    // those 1141 agreed with this ordering, ties included — so the reading of `std::sort`'s incidental
    // behaviour is confirmed across a whole model rather than one measurement.
    ordered.sort_by_key(|index| unique.get(index).copied().unwrap_or(PADDING_INDEX_SORTS_AS));
    ordered
}

/// The sort key an UNUSED LRF index is pushed with — `std::make_tuple(idxToUse, 1, 0)` (`dip.cpp:1394`).
///
/// 🔒 THE SAME NUMBER A ONE-VALUE INDEX HAS, which is what makes padding TIE with those rather than sort ahead
/// of them. A zero would put every padding index FIRST and rewrite the header's index list.
const PADDING_INDEX_SORTS_AS: usize = 1;

/// WHICH LRF INDICES THE PACKET NAMES — the UNION across every core, ascending.
///
/// ⛔⛔ NOT THIS CORE'S OWN SET. `LXL0LRFInitCompression` pushes an index if ANY core initialises it
/// (`dip.cpp:1325-1345`: the inner loop is `for (auto& coreId : initcores)`), so every core's header names the SAME
/// indices in the SAME order and a core that does not write one leaves that slot at zero — which is exactly what
/// the packing loop does, `idxList.push_back(-1)` for "no valid idx in this position" (`:3581`). Taking this core's
/// set instead gives two cores two different headers for one program.
fn named_across_cores(unique: &std::collections::BTreeMap<RegIndex, usize>) -> Vec<RegIndex> {
    unique.keys().copied().collect()
}

/// AN LX UNIT's LRF REGION — four 24-bit values and two MVR values per flit.
///
/// `lrf` and `mvr` are the registers this unit initialises; every other index in the packet is padding at zero.
///
/// ⭐ THE MVR VALUES ARE POSITIONAL, NOT NAMED. Flit `k` carries MVR indices `2k` and `2k+1`
/// (`dip.cpp:3757-3766`: `mvrOffset = numLRFFlitsProcessed * 2`), so an MVR index decides WHICH FLIT holds it and
/// the header nibble says nothing about it. That is why they are passed as a dense list rather than a map.
pub fn lx_lrf_region(
    lrf: &[(RegIndex, LxLrfValue)],
    mvr: &[MvrValue],
    unique: &std::collections::BTreeMap<RegIndex, usize>,
) -> LrfRegion {
    let used = named_across_cores(unique);
    for pair in lrf.windows(2) {
        assert!(
            pair[0].0 != pair[1].0 || pair[0].1 == pair[1].1,
            "two different values name one LX LRF register; a register holds one value, so this is a defect in \
             the allocation rather than an order to choose between"
        );
    }
    let indices = lx_l0_index_list(
        &used,
        LrfPerFlit::of(LxLrfFlit::LRF_PER_FLIT),
        MvrIndices::of(mvr.len()),
        unique,
    );
    let mut region = LrfRegion::default();
    for (flit, covered) in indices.chunks(LxLrfFlit::LRF_PER_FLIT).enumerate() {
        let mut values = [LxLrfValue::zero(); LxLrfFlit::LRF_PER_FLIT];
        for (at, index) in covered.iter().enumerate() {
            if let Some((_, value)) = lrf.iter().find(|(reg, _)| reg == index) {
                values[at] = *value;
            }
        }
        let mut moves = [MvrValue::default(); LxLrfFlit::MVR_PER_FLIT];
        for (at, slot) in moves.iter_mut().enumerate() {
            if let Some(value) = mvr.get(flit * LxLrfFlit::MVR_PER_FLIT + at) {
                *slot = *value;
            }
        }
        push_named(&mut region, covered, LxLrfFlit::new(values, moves).pack());
    }
    region
}

/// AN L0 UNIT's LRF REGION — eight 2-byte values per flit, and no MVR tail.
pub fn l0_lrf_region(
    lrf: &[(RegIndex, L0LrfValue)],
    unique: &std::collections::BTreeMap<RegIndex, usize>,
) -> LrfRegion {
    let used = named_across_cores(unique);
    let indices = lx_l0_index_list(
        &used,
        LrfPerFlit::of(L0LrfFlit::LRF_PER_FLIT),
        MvrIndices::of(0),
        unique,
    );
    let mut region = LrfRegion::default();
    for covered in indices.chunks(L0LrfFlit::LRF_PER_FLIT) {
        let mut values = [L0LrfValue::zero(); L0LrfFlit::LRF_PER_FLIT];
        for (at, index) in covered.iter().enumerate() {
            if let Some((_, value)) = lrf.iter().find(|(reg, _)| reg == index) {
                values[at] = *value;
            }
        }
        push_named(&mut region, covered, L0LrfFlit::new(values).pack());
    }
    region
}

/// Push one LX/L0 flit under the nibbles of every index it covers.
fn push_named(region: &mut LrfRegion, covered: &[RegIndex], slice: Slice) {
    let names: Vec<LrfRegIdx> = covered
        .iter()
        .map(|index| LrfRegIdx::of_register(Bits::exactly(index.get() as u32)))
        .collect();
    region.push_covering(names, slice);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// LAYOUT 3 — the compute units (SFP/PE): ONE 128-BIT CONSTANT per slice.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// AN FP16 CONSTANT's BIT PATTERN — an opaque nonlinear op's `c<n>`.
///
/// ⛔ A BIT PATTERN, NEVER A FLOAT. The value reaches this crate already encoded (`bitWidth == 16`,
/// `dip.cpp:3746-3779`); re-deriving it from an `f32` here would round a number the frontend already rounded,
/// and the two would differ in the last bit for exactly the constants that matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fp16Const(u16);

impl Fp16Const {
    /// The sixteen bits.
    pub const fn of(bits: u16) -> Self {
        Self(bits)
    }

    /// The bits.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A COMPUTE UNIT's LRF REGISTER VALUE — **128 bits**, one whole [`Slice`] (`dip.cpp:3746-3779`,
/// `slice_id < 6`, `bitWidth == 16`).
///
/// ⛔ ONE REGISTER, NOT EIGHT LANES, WHICH IS THE WHOLE DIFFERENCE FROM [`L3FlitLanes`]. An L3 slice holds eight
/// INDEPENDENT 16-bit registers; a compute slice holds ONE register that happens to be filled with eight copies
/// of a 16-bit constant. Under the L3 shape a caller could write eight different constants into what is a single
/// register, and the device would read the whole 128 bits as one value.
///
/// ⭐ AND THE fp16 HALF-WORD SWAP IS THE IDENTITY HERE, SO IT IS NOT PORTED. `dip.cpp:3746-3779` writes
/// `regVal[j] = (x >> 16) | (x << 16)` per `uint32`; for a splat both halves are equal, so no input can observe
/// it — porting it would be code no value distinguishes, and the assertion below pins that. A non-splat 128-bit
/// constant, which nothing in the reference builds, would need the swap's meaning from `dip.cpp` first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeLrfValue([u32; 4]);

impl ComputeLrfValue {
    /// The fp16 constant broadcast across all eight 16-bit lanes of the 128-bit register.
    pub const fn splat_fp16(constant: Fp16Const) -> Self {
        let word = ((constant.get() as u32) << 16) | constant.get() as u32;
        Self([word; 4])
    }

    /// The slice this register value occupies — the whole 16 bytes.
    pub const fn slice(self) -> Slice {
        Slice(self.0)
    }
}

/// A REGISTER OF A COMPUTE UNIT's LRF — the file whose depth the nuked crate replaced with the ISA field's width.
///
/// ⛔ THE REGISTER IS SUPPLIED, NOT DERIVED FROM POSITION. The reference had a second function
/// (`sfp_lrf_const_flits`) that put constant `k` into register `R{k+1}` — a rule read off ONE captured bundle
/// (`efc27d2f977c0497`, whose header carries `0x654321` for six constants), not out of `dip.cpp`. That bundle
/// shows what one program contained; it cannot say whether the numbering is the layout's or that program's
/// allocation. So the `R{k+1}` rule is NOT ported and the allocator names the register.
pub const fn compute_lrf_reg(index: u8) -> RegIndex {
    RegIndex::of_file(index, file_depth(RegUnit::Sfp, RegFile::Lrf))
}

/// A COMPUTE UNIT's LRF CONSTANT PRELOAD — the register file an opaque nonlinear op (mish/exp/gelu) needs loaded
/// before it runs (`dip.cpp:3746-3779`).
///
/// One slice per constant, emitted in ASCENDING REGISTER ORDER: the decode walks a `std::set<int>` of register
/// numbers (`dip.cpp:508-556`), and each emitted register number is the next header nibble.
///
/// ⭐ SORTING IS NOT SEARCHING. The order is the device's, stated by the C++; what this crate forbids is
/// recovering a value nobody stated, and the register numbers are given.
///
/// ⛔ AND A REGISTER ABOVE 15 CANNOT BE NAMED AT ALL. The nibble is four bits, so on SEN1P5 — whose compute LRF
/// holds 32 — registers 16..31 have no encoding in the header; `Bits::<4>::exactly` refuses rather than naming
/// register 3 when 19 was meant.
pub fn compute_lrf_slices(constants: &[(RegIndex, Fp16Const)]) -> LrfRegion {
    let mut ordered: Vec<(RegIndex, Fp16Const)> = constants.to_vec();
    ordered.sort_by_key(|(reg, _)| *reg);
    // ⛔ TWO CONSTANTS FOR ONE REGISTER IS A CONTRADICTION, AND THE SORT MAKES IT INVISIBLE: the pair would emit
    // two slices with the SAME nibble, so one silently wins on the device and the op runs with a constant
    // missing.
    for pair in ordered.windows(2) {
        assert!(
            pair[0].0 != pair[1].0,
            "two constants name the same compute LRF register. A register holds one value, so this is a defect \
             in the allocation rather than an order to choose between"
        );
    }
    let mut region = LrfRegion::default();
    for (reg, constant) in ordered {
        region.push(
            LrfRegIdx::of_register(Bits::exactly(reg.get() as u32)),
            ComputeLrfValue::splat_fp16(constant).slice(),
        );
    }
    region
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
// THE INVARIANTS, AS BUILD ERRORS.
//
// Each pins the RIGHT answer and the WRONG one, at the TOP of the field's width — a packing that only works for
// small values passes every check that only tries small values, which is how the header's flit-count/flag-bit
// collision survived every captured fixture.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

// ── L3: eight 16-bit lanes ───────────────────────────────────────────────────────────────────────────────────

/// ⛔ AN L3 LRF SLICE HOLDS EIGHT 16-BIT VALUES, NOT FOUR 32-BIT ONES.
///
/// A lane at the TOP of its width says which reading is packed: lane 2 is the LOW half of word 1. Under the
/// four-32-bit reading, value 2 would BE word 2 — so the wrong reading is pinned as absent, and it is the reading
/// the LX layout actually uses, which is what makes the confusion available in the first place.
const _: () = assert!(
    L3FlitLanes::none()
        .set(LaneIdx::of(2), L3Lane::of(0xFFFF))
        .pack()
        .0[1]
        == 0x0000_FFFF
);
const _: () = assert!(
    L3FlitLanes::none()
        .set(LaneIdx::of(2), L3Lane::of(0xFFFF))
        .pack()
        .0[2]
        == 0
);
/// An ODD lane is the HIGH half of its word, and the even lane beside it stays untouched.
const _: () = assert!(
    L3FlitLanes::none()
        .set(LaneIdx::of(7), L3Lane::of(0xFFFF))
        .pack()
        .0[3]
        == 0xFFFF_0000
);
/// All eight lanes at the top of their width fill the slice EXACTLY — no lane overlaps another, and none is
/// unreachable.
const _: () = {
    let mut lanes = L3FlitLanes::none();
    let mut lane = 0;
    while lane < LaneIdx::COUNT {
        lanes = lanes.set(LaneIdx::of(lane), L3Lane::of(0xFFFF));
        lane += 1;
    }
    let packed = lanes.pack();
    assert!(
        packed.0[0] == u32::MAX
            && packed.0[1] == u32::MAX
            && packed.0[2] == u32::MAX
            && packed.0[3] == u32::MAX
    );
};

/// ⛔⛔ A WRITTEN ZERO IS NOT AN UNWRITTEN LANE, AND THE GRID DEPENDS ON IT. LBR 0's pre-set value IS zero
/// (`dcgbeCodegen.cpp:148-159`, and `:159`'s `DT_CHECK` requires a zero among them), so a packet whose only set
/// register is LBR 0 must still be emitted: dropping it shifts every later packet's nibble down a position and
/// the device applies each remaining slice to the wrong packet number.
const _: () = assert!(
    L3FlitLanes::none()
        .set(LaneIdx::of(0), L3Lane::of(0))
        .any_written()
);
const _: () = assert!(!L3FlitLanes::none().any_written());
/// And presence is not smuggled into the bits: an all-zero written slice is still all zero.
const _: () = assert!(
    L3FlitLanes::none()
        .set(LaneIdx::of(0), L3Lane::of(0))
        .pack()
        .0[0]
        == 0
);

/// The 14-bit fields: the top value survives its lane, so the reference's `0x3FFF` mask has nothing to do.
const _: () = assert!(Addr14::MAX == 0x3FFF);
const _: () = assert!(Addr14::of(Addr14::MAX).lane().get() == 0x3FFF);

/// ⭐ EBR IS 32 BITS AND EAR IS 21, WHICH IS WHY THEY ARE TWO TYPES: the same `u32` splits differently, because
/// EBR's high lane is 16 bits wide and EAR's is 5.
const _: () = assert!(EbrValue::BITS == 32);
const _: () = assert!(EarValue::BITS == 21);
const _: () = assert!(EarValue::MAX == 0x001F_FFFF);

/// ⛔⛔ AND EBR'S TWO WIDTHS ARE NOT NECESSARILY THE SAME NUMBER, WHICH IS WHAT THIS PINS. The init packet always
/// carries 32 bits in two lanes; the REGISTER keeps 30 up to RCUDD1A and 32 from SEN1P5 (`bitSize` against
/// `bitSizeInitPacket` in `regInfoPerUnit`). So on this arch the pair is NOT redundant and the wider one is the
/// one that does not hold: a value between them is packed intact and truncated on load.
///
/// ⛔ AND THE PACKET IS NEVER NARROWER THAN THE REGISTER, which is the direction that would lose bits in the
/// packing rather than in the load. That holds on both arches and is the assertion worth keeping either way.
const _: () = assert!(EbrValue::BITS >= EbrValue::REGISTER_BITS);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(EbrValue::REGISTER_BITS == 30 && EbrValue::MAX == 0x3FFF_FFFF);
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(EbrValue::REGISTER_BITS == 32 && EbrValue::MAX == u32::MAX);

/// The widest EBR the REGISTER admits, split across its lanes.
///
/// ⛔ THIS USED TO WITNESS THE SPLIT WITH `u32::MAX` ON EVERY ARCH, which is not a constructible EBR up to
/// RCUDD1A: it asserted that both lanes fill, which is true of the LANES and false of the register they load
/// into. The widest legal value is the only honest witness, and asking for it is what caught the missing bound.
#[cfg(feature = "arch-rcudd1a")]
const _: () = {
    let (low, high) = EbrValue::of(EbrValue::MAX).lanes();
    assert!(low.get() == 0xFFFF && high.get() == 0x3FFF);
};
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = {
    let (low, high) = EbrValue::of(EbrValue::MAX).lanes();
    assert!(low.get() == 0xFFFF && high.get() == 0xFFFF);
};
/// The widest EAR fills the low lane and exactly five bits of the high one — so a value that is a legal EBR is
/// not a legal EAR, and the wrong type cannot be handed over.
const _: () = {
    let (low, high) = EarValue::of(EarValue::MAX).lanes();
    assert!(low.get() == 0xFFFF && high.get() == 0x1F);
};
/// A wide value lands low lane first.
const _: () = {
    let (low, high) = EbrValue::of(0x1234_5678).lanes();
    assert!(low.get() == 0x5678 && high.get() == 0x1234);
};

/// ⭐ THE PACKET GRID COVERS EACH FILE EXACTLY ONCE — no register unreachable, none covered twice.
///
/// Two packets of four LBR reach all 8; two of eight LAR reach all 16; two of four EBR reach all 8; four of four
/// EAR reach all 16. If a depth moves, the grid stops covering its file and the build stops here — which is
/// precisely the SEN1P5 case, where EBR grows to 16 and eight of its registers would have no packet.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(2 * L3LrfPacket::LBR_PER_PACKET == L3AddrFile::Lbr.depth().get());
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(2 * L3LrfPacket::LBR_PER_PACKET == L3AddrFile::Gtr.depth().get());
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(2 * L3LrfPacket::LAR_PER_PACKET == L3AddrFile::Lar.depth().get());
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(2 * L3LrfPacket::WIDE_PER_PACKET == L3AddrFile::Ebr.depth().get());
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(4 * L3LrfPacket::WIDE_PER_PACKET == L3AddrFile::Ear.depth().get());

/// The last packet of each file ENDS at the file's depth — a base plus a count that never leaves the file.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    L3LrfPacket::LbrGtrHigh.base() + L3LrfPacket::LBR_PER_PACKET == L3AddrFile::Lbr.depth().get()
);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    L3LrfPacket::Ear3.base() + L3LrfPacket::WIDE_PER_PACKET == L3AddrFile::Ear.depth().get()
);

/// ⛔ AND THIS IS WHAT LICENSES DROPPING THE L3LU/L3SU DISTINCTION. The vendored table gives the two units the
/// same depth in every file the grid touches, so a unit tag threaded through [`L3Lrf`] would be a field nothing
/// reads. If the table ever separates them, this fails and the tag has to come back — the collapse is CHECKED,
/// not assumed.
const _: () = assert!(
    file_depth(RegUnit::L3lu, RegFile::Ebr).get() == file_depth(RegUnit::L3su, RegFile::Ebr).get()
);
const _: () = assert!(
    file_depth(RegUnit::L3lu, RegFile::Ear).get() == file_depth(RegUnit::L3su, RegFile::Ear).get()
);
const _: () = assert!(
    file_depth(RegUnit::L3lu, RegFile::Lar).get() == file_depth(RegUnit::L3su, RegFile::Lar).get()
);
const _: () = assert!(
    file_depth(RegUnit::L3lu, RegFile::Gtr).get() == file_depth(RegUnit::L3su, RegFile::Gtr).get()
);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    file_depth(RegUnit::L3lu, RegFile::Lbr).get() == file_depth(RegUnit::L3su, RegFile::Lbr).get()
);

/// The ten packet numbers are 0..10, in order — the position in [`L3LrfPacket::ALL`] IS the packet number, so a
/// variant inserted in the wrong place is a build error rather than a slice applied to the wrong packet.
const _: () = {
    let mut i = 0;
    while i < L3LrfPacket::ALL.len() {
        assert!(L3LrfPacket::ALL[i].number().get() as usize == i);
        i += 1;
    }
};
/// ⭐ TEN PACKETS FIT THE HEADER's SIXTEEN NIBBLES, which is why [`L3Lrf::build`] needs no cap of its own: an L3
/// unit can emit every packet it has and still be describable.
const _: () = assert!(L3LrfPacket::ALL.len() <= LrfIndexList::ENTRIES);

// ── LX/L0: four 24-bit values at a 3-byte stride, then two 16-bit MVR ────────────────────────────────────────

/// The tail begins where the four 3-byte values end — computed, not the reference's literal 12.
const _: () = assert!(LxLrfFlit::MVR_AT == 12);
/// And the flit is exactly full: four times three bytes plus two times two is sixteen. A layout with slack would
/// mean a field this port does not know about.
const _: () = assert!(
    LxLrfFlit::LRF_PER_FLIT * LxLrfValue::BYTES + LxLrfFlit::MVR_PER_FLIT * MvrValue::BYTES
        == Slice::BYTES
);
const _: () = assert!(LxLrfValue::BITS == 24);
const _: () = assert!(LxLrfValue::MAX == 0x00FF_FFFF);

/// ⛔⛔ THE STRIDE IS THREE BYTES, NOT FOUR, AND THIS IS THE ASSERTION THAT SAYS SO. Value 1 at the top of its
/// width STRADDLES words 0 and 1 — its low byte is word 0's top byte, its other two are word 1's bottom. Under a
/// 4-byte stride it would sit entirely inside word 1, so that reading is pinned as absent.
const _: () = {
    let packed = LxLrfFlit::new(
        [
            LxLrfValue::zero(),
            LxLrfValue::of(LxLrfValue::MAX),
            LxLrfValue::zero(),
            LxLrfValue::zero(),
        ],
        [MvrValue::of(0), MvrValue::of(0)],
    )
    .pack();
    assert!(packed.0[0] == 0xFF00_0000);
    assert!(packed.0[1] == 0x0000_FFFF);
    assert!(packed.0[2] == 0 && packed.0[3] == 0);
};

/// The first value occupies the low three bytes and leaves the fourth alone — that byte is its NEIGHBOUR's, not
/// padding.
const _: () = {
    let packed = LxLrfFlit::new(
        [
            LxLrfValue::of(LxLrfValue::MAX),
            LxLrfValue::zero(),
            LxLrfValue::zero(),
            LxLrfValue::zero(),
        ],
        [MvrValue::of(0), MvrValue::of(0)],
    )
    .pack();
    assert!(packed.0[0] == 0x00FF_FFFF);
    assert!(packed.0[1] == 0);
};

/// ⭐ THE FOURTH VALUE STOPS BEFORE THE MVR TAIL. Value 3 at the top of its width fills bytes 9-11 — word 2's
/// upper three — and word 3, the MVR pair's, stays zero. An off-by-one stride would spill into it and initialise
/// a move register nobody named.
const _: () = {
    let packed = LxLrfFlit::new(
        [
            LxLrfValue::zero(),
            LxLrfValue::zero(),
            LxLrfValue::zero(),
            LxLrfValue::of(LxLrfValue::MAX),
        ],
        [MvrValue::of(0), MvrValue::of(0)],
    )
    .pack();
    assert!(packed.0[2] == 0xFFFF_FF00);
    assert!(packed.0[3] == 0);
};

/// The two MVR values fill word 3, the second ABOVE the first, and reach nothing below it.
const _: () = {
    let packed = LxLrfFlit::new(
        [LxLrfValue::zero(); LxLrfFlit::LRF_PER_FLIT],
        [MvrValue::of(0), MvrValue::of(0xFFFF)],
    )
    .pack();
    assert!(packed.0[3] == 0xFFFF_0000);
    assert!(packed.0[0] == 0 && packed.0[1] == 0 && packed.0[2] == 0);
};
const _: () = {
    let packed = LxLrfFlit::new(
        [LxLrfValue::zero(); LxLrfFlit::LRF_PER_FLIT],
        [MvrValue::of(0xABCD), MvrValue::of(0)],
    )
    .pack();
    assert!(packed.0[3] == 0x0000_ABCD);
};

/// ⛔ THE LX LAYOUT IS NOT THE L3 LAYOUT, side by side on the same bits. Four 24-bit values with every bit set do
/// NOT fill the slice, because only 12 of the 16 bytes are theirs; eight 16-bit L3 lanes with every bit set DO. A
/// single "LRF flit" packer parameterised by a lane width could not produce both.
const _: () = {
    let packed = LxLrfFlit::new(
        [LxLrfValue::of(LxLrfValue::MAX); LxLrfFlit::LRF_PER_FLIT],
        [MvrValue::of(0), MvrValue::of(0)],
    )
    .pack();
    assert!(packed.0[0] == u32::MAX && packed.0[1] == u32::MAX && packed.0[2] == u32::MAX);
    assert!(packed.0[3] == 0, "the MVR tail belongs to the MVR file");
};

// ── Compute (SFP/PE): one 128-bit constant per slice ─────────────────────────────────────────────────────────

/// ⭐ A COMPUTE LRF REGISTER IS THE WHOLE SLICE. The splat writes all four words, so the register is 128 bits and
/// not 16 — a reader that packed one constant per 16-bit lane would produce the same FIRST word and three zeros,
/// which is the mistake this pins.
const _: () = {
    let slice = ComputeLrfValue::splat_fp16(Fp16Const::of(0xFFFF)).slice();
    assert!(
        slice.0[0] == u32::MAX
            && slice.0[1] == u32::MAX
            && slice.0[2] == u32::MAX
            && slice.0[3] == u32::MAX
    );
};
/// The broadcast form is `(v << 16) | v`, the same word four times. `0x46dc` is the first constant of the silu
/// bundle `efc27d2f977c0497`, which CONFIRMS the layout — it does not supply it.
const _: () = {
    let slice = ComputeLrfValue::splat_fp16(Fp16Const::of(0x46dc)).slice();
    assert!(slice.0[0] == 0x46dc_46dc);
    assert!(slice.0[1] == slice.0[0] && slice.0[2] == slice.0[0] && slice.0[3] == slice.0[0]);
};
/// ⭐ AND THE HALF-WORD SWAP IS UNOBSERVABLE, which is why it is not ported: for a splat,
/// `(x >> 16) | (x << 16)` returns `x`. Only an asymmetric 128-bit constant could distinguish them, and nothing
/// here can build one.
const _: () = {
    let word = ComputeLrfValue::splat_fp16(Fp16Const::of(0x46dc)).slice().0[0];
    assert!(word.rotate_left(16) == word);
};

/// ⛔ THE HEADER's NIBBLE NAMES EVERY COMPUTE LRF REGISTER ON THIS ARCH, AND WOULD NOT ON THE NEXT. RCUDD1A's
/// SFP/PE LRF holds 16, exactly the nibble's range — independent corroboration of that depth from the packet
/// format rather than from `reg_info_per_unit`. SEN1P5 holds 32, and registers 16..31 have no nibble at all.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    file_depth(RegUnit::Sfp, RegFile::Lrf).get() as u32 == Bits::<{ LrfRegIdx::BITS }>::MAX + 1
);
/// The SFP and the PE have the same LRF depth, which is what lets one function serve both compute units — the
/// only other unit distinction this file collapses that the vendored table could have separated.
const _: () = assert!(
    file_depth(RegUnit::Sfp, RegFile::Lrf).get() == file_depth(RegUnit::Pe, RegFile::Lrf).get()
);
/// The file's first and last registers are nameable, and one past the last is not a register at all —
/// [`RegIndex::of_file`] refuses it, which is the bound the nuked crate replaced with the ISA field's width.
const _: () = assert!(compute_lrf_reg(0).get() == 0);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    compute_lrf_reg(file_depth(RegUnit::Sfp, RegFile::Lrf).get() - 1).get()
        == file_depth(RegUnit::Sfp, RegFile::Lrf).get() - 1
);

/// ⛔⛔ AN L0 VALUE'S STRIDE IS TWO BYTES, AND THIS PINS THE OTHER SIX READINGS OUT. Value 1 at the top of its
/// width sits in word 0's HIGH half; under the LX layout's 3-byte stride it would straddle into word 1, and under a
/// 4-byte stride it would be word 1's low half. Both are pinned absent.
const _: () = {
    let packed = L0LrfFlit::new([
        L0LrfValue::zero(),
        L0LrfValue::of(L0LrfValue::MASK),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
    ])
    .pack();
    assert!(packed.0[0] == (L0LrfValue::MASK as u32) << 16);
    assert!(packed.0[1] == 0 && packed.0[2] == 0 && packed.0[3] == 0);
};

/// The LAST of the eight reaches word 3's high half — so all sixteen bytes are covered and none is padding.
const _: () = {
    let packed = L0LrfFlit::new([
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::zero(),
        L0LrfValue::of(L0LrfValue::MASK),
    ])
    .pack();
    assert!(packed.0[3] == (L0LrfValue::MASK as u32) << 16);
    assert!(packed.0[0] == 0 && packed.0[1] == 0 && packed.0[2] == 0);
};

/// ⛔ AND THE L0 WIDTH IS AN ARCH FACT — `0x3ff` up to RCUDD1A and `0xfff` above (`dip.cpp:3676-3679`). Both arms
/// are written here, so a build for either gets a mask the other would refuse.
const _: () = assert!(
    L0LrfValue::MASK
        == if cfg!(feature = "arch-sen1p5") {
            0xfff
        } else {
            0x3ff
        }
);

/// ⭐ THE TWO LAYOUTS FILL THE SAME SIXTEEN BYTES WITH DIFFERENT COUNTS, which is the whole reason they are two
/// types: 4 x 3 + 2 x 2 for LX, 8 x 2 for L0.
const _: () = assert!(
    LxLrfFlit::LRF_PER_FLIT * LxLrfValue::BYTES + LxLrfFlit::MVR_PER_FLIT * MvrValue::BYTES
        == Slice::BYTES
);
const _: () = assert!(L0LrfFlit::LRF_PER_FLIT * L0LrfValue::BYTES == Slice::BYTES);

/// ⛔ AND SIXTEEN INDICES IS EXACTLY WHAT THE HEADER CAN NAME — four LX flits' worth, or two L0 flits'.
const _: () = assert!(LX_L0_LRF_INDICES == LrfIndexList::ENTRIES);
const _: () = assert!(LX_L0_LRF_INDICES.is_multiple_of(LxLrfFlit::LRF_PER_FLIT));
const _: () = assert!(LX_L0_LRF_INDICES.is_multiple_of(L0LrfFlit::LRF_PER_FLIT));
