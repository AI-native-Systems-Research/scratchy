// SPDX-License-Identifier: Apache-2.0
//! THE SPR SLICE — a unit's EIGHT 16-bit special-purpose configuration words, packed two per 32-bit word.
//!
//! ✅ VERIFIED AGAINST THE POD, 2026-08-13. Both halves of this module's layout were re-read in
//! `/project_src/deeptools/dip/dip.cpp`: `:4460-4470` is the eight-slot loop with `& 0x0000FFFF`, and `:361-380`
//! is the word packing — `SPR[elem] = word & 0xFFFF`, `SPR[elem+1] = word >> 16`, for `elem` in 0, 2, 4, 6 against
//! word index `j` — which is exactly `w[k] = (SPR[2k+1] << 16) | SPR[2k]`. Eight slots also agrees with the SPR's
//! depth of 8 in `isa::regfile` (`sysdef.cpp`), so the packet format and the register table corroborate.
//!
//! # What this stage does, from `dip.cpp:4462-4470`
//!
//! ```cpp
//! std::map<RegType, std::map<unsigned int, OperandAttr>> regInits =
//!     psInfo.regState_.at(senComp).getSimpleRegInit();
//! for (int elem = 0; elem < 8; elem++) {
//!   sprFlit.SPR[elem] = (regInits.at(RegType::SPR).find(elem) == regInits.at(RegType::SPR).end())
//!       ? 0
//!       : (regInits.at(RegType::SPR).at(elem).asInt() & 0x0000FFFF);
//! }
//! ```
//!
//! Structurally an SPR is a REGISTER INITIALISATION: eight slots, each 16 bits, **zero when the program
//! declares none**, read out of the same `getSimpleRegInit` map every other register init comes from. This
//! stage READS those values; it does not derive them.
//!
//! ⛔ WHAT PRODUCES AN SPR VALUE IS NOT STATED, AND IS NOT GUESSED HERE. Two earlier readings of this row were
//! wrong: it was first guessed as `dataStageParam_`/`N_` (the schedule), and then sourced to "machine config
//! (machine-check / stall / tether) + op precision" citing `dip.cpp:1175-1338` — **that citation does not
//! hold**, those lines are `padInstructionsOfSenComp` calls, instruction padding, nothing to do with SPR. A
//! third guess would be the third wrong one, so this module models only what `dip.cpp:4462-4470` states: the
//! READ, the width, the zero default, and the packing. Whoever lands SPR production supplies the words through
//! [`SprInits::with`]; nothing in this file may invent one.
//!
//! ⛔ EIGHT SLOTS, NOT THIRTY-TWO. `for (int elem = 0; elem < 8; elem++)` (`dip.cpp:4464`). A register INDEX
//! has 32 values ([`RegIndex::COUNT`], `SentientTypes.td:96-159`); an SPR SLOT has 8. Handing out indices from
//! the wider set is a defect this crate has already shipped once (`6d86e23c`, registers 16..31 of a 16-deep
//! file), so [`SprSlot`] is its own type with its own count, and the two counts are pinned unequal below.
//!
//! ⛔ EVERY PT ROW CARRIES AN SPR FLIT. An earlier version of this comment claimed only row 0 did.
//! `generatePTInitPacket` builds `sprFlit` for each row it is called with and sizes the row's block as
//! `maxFlits + 2` — "+1 spr Flit, +1 header flit" (`dip.cpp:4668`). Each `pt_row<p>` is its own program with
//! its own `regState_` entry, so each has its own eight words to carry.
//!
//! # ⭐ WHAT ONE SPR ACTUALLY MEANS — PT SLOT 1 IS A PRECISION, AND IT IS NOT PORTED
//!
//! An SPR is a 16-bit CONFIGURATION word, **not an address**. An earlier `physical_op::Addressed` doc claimed
//! SPRs were tied to allocations; corrected in `5c7715e9` with the citations below, because a justification is
//! what the next reader reasons from.
//!
//! For `PT` + slot **1** the word is structured (`progir.cpp:884-895`, inside `progir.cpp:884-930`):
//!
//! - `(spr1 & 0b111) == 0b111` gates it — the precision is only meaningful when the low 3 MASK bits are set.
//! - `(spr1 >> 8) & 0b11` selects the data format: `0b00` `SEN169_FP16`, `0b01` `SEN143_FP8`, `0b11` `SENINT2`,
//!   and `0b10` re-reads THREE bits — `((spr1 >> 8) & 0b111) == 0b110` is `SENINT8`, else `SENINT4`. **Bit 10
//!   distinguishes INT8 from INT4**, so the field is 3 bits wide in that arm, not 2.
//!
//! ⛔ DELIBERATELY NOT GIVEN A TYPE, for three reasons worth stating so it is not "tidied up" later:
//!
//! 1. **there is no illegal encoding to refuse.** The switch is over 2 bits with all 4 arms assigned, so the
//!    C++'s `default: "Illegal precision bits in SPR1"` is dead code — and a decoder here would need either
//!    that dead arm or a `_`, both forbidden. A fallible decoder would be a refusal that can never fire.
//! 2. **`gates::Dtype` is a deliberate SUBSET** of `DataFormats` — only what scratchy's `Df` emits
//!    (`gates.rs:10-31`) — and has no `SENINT4`/`SENINT2`. Reusing it needs unreachable variants; a second enum
//!    gives FP16/FP8/INT8 two homes. Neither is right until something needs it.
//! 3. **nothing emits a PT SPR 1.** Both `add_spr` callers in the previous attempt were tests, so building the
//!    encoder would model a quantity this port does not produce.
//!
//! ⭐ THE RAW INTEGERS LIVE ONLY AT THE DOOR. [`SprSlot::of`], [`SprValue::of`] and [`SprValue::of_reg_init`]
//! take a machine integer because that is where the C++'s untyped `unsigned int` / `OperandAttr::asInt()`
//! arrives, and each refuses or narrows on the spot. Every other signature in this file is typed, so a bare
//! number cannot travel past the constructor and be mistaken for the other quantity.
//!
//! ✅ THE ONE REAL INVARIANT, for whoever lands folded SPRs: `prec_map` requires **one precision per unit** —
//! every fold value of SPR1 must decode to the SAME precision or dxp rejects the program (*"Precision SPR1
//! conflicts with previous definition"*). That is a CROSS-VALUE constraint, unlike the per-value bit check, and
//! it is the thing to make unconstructable — one precision per unit, projected into every fold value — at the
//! point PT SPR 1 emission exists.

use crate::islands::progir::RegIndex;
use crate::islands::init_packet::Slice;

/// WHICH OF THE EIGHT SPR SLOTS a word initialises.
///
/// `for (int elem = 0; elem < 8; elem++)` (`dip.cpp:4464`). A newtype so a slot index cannot be confused with
/// the 16-bit word that goes in it — the C++ holds the index as `unsigned int` and the value as an
/// `OperandAttr` integer, and both become plain numbers the moment they are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SprSlot(u8);

impl SprSlot {
    /// The eight slots a flit carries (`dip.cpp:4464`).
    pub const COUNT: u8 = 8;

    /// Whether the hardware HAS this slot — a fact about the machine, asked before naming one.
    ///
    /// ⭐ A PREDICATE, NOT A FALLIBLE CONSTRUCTOR. [`Self::of`] takes a constant and refuses at build time, so
    /// there is no `Option<SprSlot>` anywhere to unwrap; this exists so the absence of a ninth slot can be
    /// PINNED as a `const` assertion instead of only being tested for.
    pub const fn exists(slot: u8) -> bool {
        slot < Self::COUNT
    }

    /// The slot numbered `slot`.
    ///
    /// ⛔ `panic!` IS THE REFUSAL, because every call site in this crate passes a literal: a ninth SPR is a
    /// build error, not a runtime check, and not an `Option` a caller talks itself past. The number to bound
    /// against is 8 — the SLOT count — and never 32, which is how wide a register INDEX field is.
    pub const fn of(slot: u8) -> Self {
        assert!(
            Self::exists(slot),
            "an SPR slot must be one of the eight the flit carries (dip.cpp:4464); 32 is the width of a \
             REGISTER INDEX field, and bounding a slot by it hands out slots the hardware has not got"
        );
        Self(slot)
    }

    /// The slot's number, for indexing the eight-wide array it addresses.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// ONE SPR'S 16-BIT CONFIGURATION WORD.
///
/// `& 0x0000FFFF` (`dip.cpp:4469`) — the field is 16 bits, and the mask is in the C++ because the
/// `OperandAttr` behind it is wider. Carrying a `u16` makes the narrowing a property of the TYPE instead of a
/// masking step repeated at every write.
///
/// ⛔ NOT AN ADDRESS. See this module's header: for `PT` slot 1 the bits are a mask plus a precision selector
/// (`progir.cpp:884-895`), so nothing here may be treated as a pointer or an allocation offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SprValue(u16);

impl SprValue {
    /// The field's width in bits (`& 0x0000FFFF`, `dip.cpp:4469`).
    pub const BITS: u32 = u16::BITS;

    /// What an undeclared slot holds — the C++'s own `find() == end() ? 0` (`dip.cpp:4466-4468`).
    pub const ZERO: Self = Self(0);

    /// The widest word the field holds.
    pub const MAX: Self = Self(u16::MAX);

    /// The word `word`.
    ///
    /// Total, and deliberately so: the parameter is the field's own width, so "a value too wide for an SPR" is
    /// not a case to check — it cannot be spelled.
    pub const fn of(word: u16) -> Self {
        Self(word)
    }

    /// Whether a `getSimpleRegInit` entry FITS the field, the C++ having read it as a wider integer
    /// (`OperandAttr::asInt()`, `dip.cpp:4469`).
    ///
    /// Exists so [`Self::of_reg_init`]'s refusal can be pinned as a `const` assertion: the words the C++ would
    /// silently mangle are named below rather than merely absent.
    ///
    /// Written as a ROUND TRIP through the field rather than a range check, because that is the question being
    /// asked — does narrowing lose anything the entry stated — and it answers it for a negative entry too,
    /// which `& 0x0000FFFF` turns into a large positive word.
    pub const fn fits(attr: i64) -> bool {
        attr == (attr as u16) as i64
    }

    /// The word a register-init entry declares (`regInits.at(RegType::SPR).at(elem).asInt()`, `dip.cpp:4469`).
    ///
    /// ⚠️ THE C++ MASKS (`& 0x0000FFFF`) WHERE THIS REFUSES, and the difference is deliberate rather than an
    /// oversight: a truncated SPR is a machine-state register set to a value nobody asked for, and since every
    /// argument here is a constant the refusal costs a build error instead of a wrong device configuration. If
    /// a real program is ever found to declare a wider value legitimately, THIS is the one place to change.
    pub const fn of_reg_init(attr: i64) -> Self {
        assert!(
            Self::fits(attr),
            "a getSimpleRegInit SPR entry does not fit the 16-bit field; the C++ masks it with 0x0000FFFF \
             (dip.cpp:4469) and would write a register value nobody asked for"
        );
        Self(attr as u16)
    }

    /// The word, for packing.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// THE EIGHT SPR WORDS A BLOCK INITIALISES, zero where the program declares none.
///
/// Zero-when-absent is the C++'s own behaviour (`dip.cpp:4466-4468`), not a convenience: a slot the program
/// does not declare is WRITTEN as 0, and all eight are carried regardless. That is why this holds an array of
/// eight and not a list of declared slots — a short list would make "slot 5 was never mentioned" and "slot 5
/// is zero" two different states, and the device only has the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SprInits {
    /// Indexed by [`SprSlot::get`]. `[SprValue; _]` rather than `[u16; _]` so no raw word can be dropped in.
    values: [SprValue; SprSlot::COUNT as usize],
}

impl SprInits {
    /// A block that declares no SPR at all: eight zeros, which is exactly what the C++ writes for it.
    pub const UNDECLARED: Self = Self {
        values: [SprValue::ZERO; SprSlot::COUNT as usize],
    };

    /// Declare one slot's word.
    ///
    /// Takes and returns `Self` so a block's inits are built in one `const` expression at the call site, which
    /// keeps the whole slice a compile-time constant.
    pub const fn with(mut self, slot: SprSlot, value: SprValue) -> Self {
        self.values[slot.get() as usize] = value;
        self
    }

    /// One slot's word — [`SprValue::ZERO`] if the program declared none (`dip.cpp:4466-4468`).
    pub const fn get(&self, slot: SprSlot) -> SprValue {
        self.values[slot.get() as usize]
    }

    /// The slice these eight words pack into.
    pub const fn slice(&self) -> SprSlice {
        SprSlice::from_values(self.values)
    }
}

/// THE SPR INIT SLICE — eight 16-bit words packed two per 32-bit word:
/// `w[k] = (SPR[2k+1] << 16) | SPR[2k]`.
///
/// ⚠️ TWO CITATIONS FOR THE SAME FORMULA, both carried because they differ and neither may be dropped: the
/// previous attempt's SPR bridge cites `dip.cpp:4528-4529`, and its slice type cites `dip.cpp:3634-3645`. The
/// packing they state is identical, so the layout is not in doubt; which line names it is, and collapsing them
/// to one invented number would destroy the evidence that two places were read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SprSlice {
    /// The eight words, index = slot number.
    spr: [SprValue; SprSlot::COUNT as usize],
}

impl SprSlice {
    /// How many slots share one 32-bit word — 16 bits each, so two.
    pub const SLOTS_PER_WORD: usize = 32 / SprValue::BITS as usize;

    /// How many 32-bit words the eight slots occupy — DERIVED, never written as a literal 4, so the day the
    /// slot count or the field width changes the slice does not silently keep its old shape.
    pub const WORDS: usize = SprSlot::COUNT as usize / Self::SLOTS_PER_WORD;

    /// The ONLY way to build one, private to this module so [`SprInits::slice`] is its sole caller.
    ///
    /// ⛔ SEALED. `SprInits` keeps its array private and is written through typed [`SprSlot`]/[`SprValue`]; a
    /// public constructor taking eight raw words would let a caller skip all of that and put arbitrary bits
    /// where the device reads its register initialisations.
    const fn from_values(spr: [SprValue; SprSlot::COUNT as usize]) -> Self {
        Self { spr }
    }

    /// The eight words, index = slot number. Read-only; reading bypasses no validation.
    pub const fn values(&self) -> [SprValue; SprSlot::COUNT as usize] {
        self.spr
    }

    /// Pack the eight words into the unit's 16-byte slice: `w[k] = (SPR[2k+1] << 16) | SPR[2k]`.
    ///
    /// ⛔ THE HALVES ARE NOT INTERCHANGEABLE — a transposed pair writes each of two configuration words into
    /// the other's register, and the device accepts that silently. The `const` assertions below pin the low
    /// half to the EVEN slot with eight distinct words, so a swap breaks the build rather than the machine.
    ///
    /// A loop over the derived [`Self::WORDS`] rather than four hand-written words: four literal expressions
    /// are four chances to write `2 * k` once as `2 * k + 1`.
    pub const fn slice(&self) -> Slice {
        let mut words = [0u32; Self::WORDS];
        let mut pair = 0;
        while pair < Self::WORDS {
            let low = self.spr[Self::SLOTS_PER_WORD * pair].get();
            let high = self.spr[Self::SLOTS_PER_WORD * pair + 1].get();
            words[pair] = ((high as u32) << SprValue::BITS) | (low as u32);
            pair += 1;
        }
        Slice(words)
    }
}

// ───────────── THE INVARIANTS, `const` SO A WRONG LAYOUT DOES NOT BUILD ─────────────
//
// ⛔ NO TESTS HERE ON PURPOSE. A bit layout is a compile-time fact, and a test only reports it after something
// has already built. Each block below also pins the WRONG answer as ABSENT, because an assertion that merely
// agrees with the code above it proves nothing.

/// ⭐ EIGHT SLOTS, NOT THIRTY-TWO — the count is the loop bound of `dip.cpp:4464`, and it is NOT the width of a
/// register-index field. Pinning them UNEQUAL makes the confusion a build error the day either one moves.
const _: () = assert!(SprSlot::COUNT == 8);
const _: () = assert!(RegIndex::COUNT == 32);
const _: () = assert!(SprSlot::COUNT < RegIndex::COUNT);

/// ⭐ SLOT 7 EXISTS AND SLOT 8 DOES NOT. `SprSlot::of(8)` does not compile — verified — so the absence is
/// pinned through the predicate that constructor is bounded by, and register 31 is pinned absent with it.
const _: () = assert!(SprSlot::exists(0));
const _: () = assert!(SprSlot::exists(SprSlot::COUNT - 1));
const _: () = assert!(!SprSlot::exists(SprSlot::COUNT));
const _: () = assert!(!SprSlot::exists(RegIndex::COUNT - 1));
const _: () = assert!(SprSlot::of(SprSlot::COUNT - 1).get() == 7);

/// ⭐ THE FIELD IS 16 BITS. The widest word round-trips, and the values the C++ would MASK are refused: a
/// wider `OperandAttr` cannot become an `SprValue`, and neither can a negative one.
const _: () = assert!(SprValue::BITS == 16);
const _: () = assert!(SprValue::MAX.get() == 0xFFFF);
const _: () = assert!(SprValue::fits(0xFFFF));
const _: () = assert!(!SprValue::fits(0x1_0000), "the C++ would mask this to 0");
const _: () = assert!(!SprValue::fits(-1), "and this to 0xFFFF");
const _: () = assert!(SprValue::of_reg_init(0xFFFF).get() == SprValue::MAX.get());
/// And [`SprValue::of`] cannot even be OFFERED a wider word — its parameter is the field's own width, so
/// truncation is unrepresentable rather than checked.
const _: () = assert!(SprValue::of(u16::MAX).get() == SprValue::MAX.get());

/// ⭐ AN UNDECLARED SLOT IS ZERO, all eight of them, and the slice it packs into is all zeros
/// (`dip.cpp:4466-4468`).
const _: () = {
    let mut slot = 0;
    while slot < SprSlot::COUNT {
        assert!(SprInits::UNDECLARED.get(SprSlot::of(slot)).get() == 0);
        slot += 1;
    }
    let words = SprInits::UNDECLARED.slice().slice().0;
    let mut word = 0;
    while word < SprSlice::WORDS {
        assert!(words[word] == 0);
        word += 1;
    }
};

/// ⭐ A DECLARED SLOT IS THE ONLY ONE SET — `with` writes one register, not a neighbourhood.
const _: () = {
    let inits = SprInits::UNDECLARED.with(SprSlot::of(3), SprValue::of(0x1234));
    let mut slot = 0;
    while slot < SprSlot::COUNT {
        let expected = if slot == 3 { 0x1234 } else { 0 };
        assert!(inits.get(SprSlot::of(slot)).get() == expected);
        slot += 1;
    }
    // Slot 3 is the HIGH half of word 1, which is where a lost `+ 1` or a stray `/ 2` would put it elsewhere.
    let words = inits.slice().slice().0;
    assert!(words[0] == 0);
    assert!(words[1] == 0x1234_0000);
    assert!(words[2] == 0);
    assert!(words[3] == 0);
};

/// ⭐ TWO SLOTS SHARE A WORD, LOW SLOT IN THE LOW HALF: `(SPR[1] << 16) | SPR[0]`
/// (`dip.cpp:4528-4529`, also cited as `dip.cpp:3634-3645`).
const _: () = {
    let inits = SprInits::UNDECLARED
        .with(SprSlot::of(0), SprValue::of(0xAAAA))
        .with(SprSlot::of(1), SprValue::of(0xBBBB));
    let words = inits.slice().slice().0;
    assert!(words[0] == 0xBBBB_AAAA);
    // ⛔ THE TRANSPOSITION IS PINNED ABSENT. Both halves are populated, so a swap still yields a valid slice
    // and only this line refuses it.
    assert!(words[0] != 0xAAAA_BBBB);
};

/// ⭐ EVERY SLOT LANDS IN ITS OWN HALF-WORD, pinned with eight DISTINCT words so no permutation of the eight —
/// not a swapped pair, not a reversed word order — satisfies this. Slot `k` sits in word `k / 2`, in the low
/// half when `k` is even and the high half when it is odd.
const _: () = {
    let mut inits = SprInits::UNDECLARED;
    let mut slot = 0;
    while slot < SprSlot::COUNT {
        // 0x1111 … 0x8888: distinct, and non-zero so a dropped slot shows up as 0.
        inits = inits.with(SprSlot::of(slot), SprValue::of(0x1111 * (slot as u16 + 1)));
        slot += 1;
    }
    let words = inits.slice().slice().0;
    let mut slot = 0;
    while slot < SprSlot::COUNT {
        let expected = 0x1111u32 * (slot as u32 + 1);
        let shift = SprValue::BITS * (slot as u32 % SprSlice::SLOTS_PER_WORD as u32);
        let half = (words[slot as usize / SprSlice::SLOTS_PER_WORD] >> shift) & 0xFFFF;
        assert!(half == expected);
        slot += 1;
    }
};

/// ⭐ THE SLICE HOLDS EXACTLY THE EIGHT SLOTS — four 32-bit words, two slots each. Writing a product and its
/// factors separately is how the two come to disagree, so the relation is asserted rather than restated.
const _: () = assert!(SprSlice::SLOTS_PER_WORD == 2);
const _: () = assert!(SprSlice::WORDS == 4);
const _: () = assert!(SprSlice::WORDS * SprSlice::SLOTS_PER_WORD == SprSlot::COUNT as usize);
const _: () = assert!(SprValue::BITS as usize * SprSlice::SLOTS_PER_WORD == 32);
/// ⭐ AND THE SLICE IS EXACTLY THOSE WORDS, no fifth word and no unused half: a 16-byte slice at four bytes
/// per word is where the eight slots stop.
const _: () = assert!(Slice::BYTES == SprSlice::WORDS * 4);

/// EVERY UNIT'S SPR WORDS — `fillSPRInfoPsinfo` (`dip.cpp:1402-1564`).
///
/// ⛔⛔ AN ALL-ZERO SPR FLIT CONFIGURES NOTHING, WHICH IS NOT THE SAME AS A DEFAULT. The low byte of each word is
/// the MASK and the high byte the SETTING — "First 8bits are the setting and the last 8 bits are the mask"
/// (`dip.cpp:1409-1411`) — so `0x01FF` writes setting `0x01` under mask `0xFF` while `0x0000` writes nothing under
/// mask `0x00` and leaves the unit as it was. The C++ writes SPR0 for EVERY unit without exception, including its
/// `else { addRegInit(0, 0x01FF | machineCheckExceptionNoComputeOrL3, SPR) }` (`:1560-1563`).
///
/// ⭐ AND EVERY VALUE HERE IS AT THIS ARCH'S DEFAULTS, which is what makes it a constant rather than a search:
/// * `sparseOp` is false, so `ptSparse` is 0 (`:1403-1406`);
/// * `l0Tethered` comes from `DIP_SET_L0TETHERING` and is unset, so `l0TetheredVal` is 0 (`:1432`);
/// * `ptCrossCl` is `(false)` in the source itself — "to be determined" (`:1433`);
/// * the PT/PE/SFP stall flags are off, so no `0x8000` throttle bit;
/// * `maxpendingNbrSPR` is 0, so an L3 unit takes SPR1 `0x00FF` (`:1536-1543`).
///
/// ⛔ THE PRECISION PICKS THE PT'S PAIR, and fp8 is a DIFFERENT SPR0 as well as SPR1: `0x81FF` against `0x01FF`
/// (`:1467-1478`, `:1497-1501`).
///
/// ⛔⛔ MACHINE-CHECK IS ON FOR EVERY UNIT BUT THE SFP, and this table used to take the DISABLED form for all of
/// them. `unitToMachineCheckEnabledMutableMap` states the defaults literally (`dip.h:501-508`):
///
/// ```text
/// {"pt", true}, {"pe", true}, {"sfp", false}, {"l3_lu_su", true}, {"no_compute_or_l3", true}
/// ```
///
/// and each `true` ORs `MACHINE_CHECK_ENABLED = 0x0404` into that unit's **SPR0** (`dip.cpp:1411-1429`). So the
/// PT, PE, L3 and every "no compute or L3" unit — the LX and L0 — carry it, and only the SFP does not.
///
/// ⭐ THE ENV OVERRIDE IS NOT PORTED AND MUST NOT BE. `MACHINE_CHECK_ENABLED_UNIT` re-parses the map at dxp's run
/// time (`dip.h:512-543`); scratchy STATES the arch's defaults, so the map is a const table here.
///
/// ⛔ AND IT GOES ONLY IN SPR0. Every arm below that sets SPR1 leaves it untouched — the C++ ORs the machine-check
/// value into the `addRegInit(0, ...)` call and never into `addRegInit(1, ...)`.
/// WHETHER THIS UNIT ABORTS EXECUTION ON A MACHINE CHECK — `unitToMachineCheckEnabledMap()` (`dip.h:499-546`),
/// as the bits SPR0 carries for it.
///
/// ⛔ THE FIVE KEYS ARE NOT FIVE UNITS. `"no_compute_or_l3"` is the C++'s `else` bucket — everything that is
/// neither a compute unit nor an L3, i.e. the LX and L0 units (`dip.cpp:1560-1563`) — so this match spreads one
/// key across four components rather than inventing an entry for each.
const fn machine_check(unit: crate::isa::regfile::Component) -> SprValue {
    use crate::isa::regfile::Component;
    // `static const auto MACHINE_CHECK_ENABLED = 0x0404;` (`dip.cpp:1411`).
    const ENABLED: u16 = 0x0404;
    // `MACHINE_CHECK_DISABLED = 0x0` (`dip.cpp:1412`) — a real value, not an absence, which is why the SFP takes
    // it explicitly rather than skipping the OR.
    const DISABLED: u16 = 0x0;
    match unit {
        // `{"pt", true}`, `{"pe", true}` (`dip.h:502-503`).
        Component::Pt | Component::Pe => SprValue::of(ENABLED),
        // `{"sfp", false}` — the ONE unit whose default is off (`dip.h:504`).
        Component::Sfp => SprValue::of(DISABLED),
        // `{"l3_lu_su", true}` (`dip.h:505`).
        Component::L3lu | Component::L3su => SprValue::of(ENABLED),
        // `{"no_compute_or_l3", true}` (`dip.h:506`).
        Component::Lxlu | Component::Lxsu | Component::L0lu | Component::L0su => {
            SprValue::of(ENABLED)
        }
    }
}

/// ⛔⛔⛔ THE FOUR TERMS `fillSPRInfoPsinfo` ORs IN THAT THIS CRATE DOES **NOT** CARRY — each one checked to
/// the source, not assumed away. If any becomes reachable, the values below are wrong.
///
/// * `ptCrossCl` — `int ptCrossCl = (false) ? 0x2000 : 0x0000;` (`dip.cpp:1436`), hardcoded FALSE by IBM with
///   the comment *"to be determined"*. It can never be set on any path, this arch or another.
/// * `PTstall` / `PEstall` / `SFPstall` — the `0x8000` throttle on the PT's SPR1 and the stall arms on the
///   PE's and SFP's. They come from the `DIP_UNITSTALL` ENV VAR, parsed into an options list
///   (`dip.h:321-345`), over a default of `std::array<bool, 8> PTstall = {{false}}` (`dip.h:233`). A debug
///   toggle, absent on every path this crate compiles.
/// * `ptSparse` — `sparseOp && coreArch >= RCUDD1A ? 0b10 : 0`, then `+= ptSparse << 8` to build value and
///   mask (`dip.cpp:1407-1410`), ORed into the PT's SPR0. `isSparseOp_` defaults false
///   (`ProgramArtifacts.h:38`) and is only ever set by `computeIsSparseOp` (`dip_prep.cpp:108`) — and NO
///   sparse op-func is in this crate's set: `precisionOfComputeOp`'s lists name `BATCHMATMUL_FP8_FWD_SPARSEKG3`
///   and its siblings, and none of them is an `OpFunc` the vendored `.ddl` templates define.
///   ⛔ IF A SPARSE OP-FUNC EVER ENTERS THAT SET, SPR0 GAINS `0x202` AND EVERY VALUE BELOW IS WRONG. The
///   no-`_` arm in [`crate::precision_of_compute_op`] is what forces that moment to be noticed.
/// * `l0Tethered` — already named on the L0 arm below.
///
/// ⭐ THIS IS WHY THE VALUES BELOW ARE FLAT LITERALS AND NOT EXPRESSIONS. They are the C++'s values with every
/// conditional term resolved to zero, and the resolution is cited rather than inherited.
pub const fn of_unit(unit: crate::isa::regfile::Component, precision: Precision) -> SprInits {
    use crate::isa::regfile::Component;
    let base = SprInits::UNDECLARED;
    let mce = machine_check(unit).get();
    match unit {
        // `:1467-1500` — the PT's SPR1 carries the QFP8 bias in bits <5:2>, and fp8 raises SPR0's bit 7.
        Component::Pt => match precision {
            Precision::Fp8 => base
                .with(SprSlot::of(0), SprValue::of(0x81FF | mce))
                .with(SprSlot::of(1), SprValue::of(0x1DFF)),
            // ⛔ MXFP8 IS THE SAME SPR0 AND A DIFFERENT SPR1 — `if (mxfp8) addRegInit(1, 0x5dFF | throttle)`
            // against the plain fp8 `0x1dFF` (`dip.cpp:1475-1482`), under its own comment *"Additionally,
            // SPR 1 bit 6 = 1 for MXFP8"*. It is a SECOND field in the C++ (`isMxfp8Op_`, set beside
            // `precisionOfOp_` at `dip_standalone.cpp:53-54`), and only the PT reads it: the PE and SFP arms
            // branch on `PEstall`/`SFPstall`/`ptCrossCl` and never on this.
            Precision::Mxfp8 => base
                .with(SprSlot::of(0), SprValue::of(0x81FF | mce))
                .with(SprSlot::of(1), SprValue::of(0x5DFF)),
            Precision::Fp16 | Precision::Fp32 => base
                .with(SprSlot::of(0), SprValue::of(0x01FF | mce))
                .with(SprSlot::of(1), SprValue::of(0x00FF)),
        },
        // `:1503-1515` — "SPR 1 bits <5:2> hold the bias for QFP8. Bias being set to 8".
        Component::Pe => base
            .with(SprSlot::of(0), SprValue::of(0x01FF | mce))
            .with(SprSlot::of(1), SprValue::of(0x1CFF)),
        // `:1516-1534` — the SFP additionally masks parity errors on incoming streaming data, in SPR4.
        Component::Sfp => base
            .with(SprSlot::of(0), SprValue::of(0x01FF | mce))
            .with(SprSlot::of(1), SprValue::of(0x1CFF))
            .with(SprSlot::of(4), SprValue::of(0x0800)),
        // `:1536-1550` — no hang detection, and SPR1 `0x00FF` because `maxpendingNbrSPR` is 0.
        Component::L3lu | Component::L3su => base
            .with(SprSlot::of(0), SprValue::of(0x01FF | mce))
            .with(SprSlot::of(1), SprValue::of(0x00FF)),
        // ⛔⛔ THE L0 HAS ITS OWN BRANCH AND IT NEVER READS THE MACHINE-CHECK MAP. `dip.cpp:1552-1554` is
        // `addRegInit(0, 0x01FF | l0TetheredVal, SPR)` — the TETHER, not the exception word — and
        // `l0TetheredVal = l0Tethered ? 0x2000 : 0x0000` (`:1431`). `isL0Tethered()` is an env override OR
        // `progArtifacts->l0TetheredMode_` (`:5097-5102`), which defaults false and is a SCHEDULER output
        // scratchy's frontend drops entirely (`lower_subtile_tape_to_superdsc.rs:1237`) — so it is `0x0000` here
        // and SPR0 is `0x01FF` flat.
        //
        // ⭐ MEASURED AGAINST `dip_standalone`: the L0SU and L0LU slices read `0x000001ff` where ORing the
        // machine-check word gave `0x000005ff`. It was the only real difference left in that group.
        //
        // `:1555-1558` — SPR1 is written only for fp4 or mxfp8, neither of which this build emits.
        Component::L0lu | Component::L0su => base.with(SprSlot::of(0), SprValue::of(0x01FF)),
        // ⭐ THE LX UNITS ARE THE C++'s OWN `else`: every unit that is neither a compute unit nor an L3 nor an L0
        // takes SPR0 alone — `addRegInit(0, 0x01FF | machineCheckExceptionNoComputeOrL3, SPR)` (`:1560-1563`).
        Component::Lxlu | Component::Lxsu => base.with(SprSlot::of(0), SprValue::of(0x01FF | mce)),
    }
}

/// ⛔⛔⛔ THE L0's SPR0 IS `0x01FF` AND NOTHING IS ORed INTO IT. `dip.cpp:1552-1554` is
/// `addRegInit(0, 0x01FF | l0TetheredVal, SPR)` — the TETHER, and `l0TetheredMode_` is a scheduler output
/// scratchy drops, so the term is `0x0000`. ORing the machine-check word here instead produced `0x05FF`, which
/// `dip_standalone` caught as the only real difference in a whole group.
///
/// ⭐ THE ASSERTION IS OVER THE FUNCTION, NOT A COMMENT ABOUT IT: whatever `of_unit` computes for an L0 unit must be
/// this value, so re-introducing an `| mce` on that arm cannot compile.
const _: () = {
    let mut which = 0;
    while which < crate::isa::regfile::Component::ALL.len() {
        let comp = crate::isa::regfile::Component::ALL[which];
        if matches!(
            comp,
            crate::isa::regfile::Component::L0lu | crate::isa::regfile::Component::L0su
        ) {
            let mut slot = 0;
            while slot < SprSlot::COUNT {
                let held = of_unit(comp, Precision::Fp16).get(SprSlot::of(slot)).get();
                assert!(
                    (slot == 0 && held == 0x01FF) || (slot != 0 && held == 0),
                    "an L0 unit's SPR0 is `0x01FF | l0TetheredVal` and the tether is 0 here (`dip.cpp:1552-1554`, \
                     `:1431`); every other slot is written only for fp4 or mxfp8 (`:1555-1558`)"
                );
                slot += 1;
            }
        }
        which += 1;
    }
};

/// WHICH PRECISION THE PT'S SPR PAIR FOLLOWS — `precisionOfOp` (`dip.cpp:1467`).
///
/// ⛔ THREE, NOT THE FIVE ISLAND 1 CARRIES: `fillSPRInfoPsinfo` branches on `SEN143_FP8`, `SENINT4`, `SENINT8`,
/// `SEN121_FP4` and an `else`, and the integer arms and fp4 are precisions this crate does not emit — so they are
/// absent here rather than aliased onto the fp16 default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp8,
    Fp16,
    Fp32,
    /// `BATCHMATMUL_MXFP8_FWD` — fp8 whose PT SPR1 is `0x5dFF` rather than `0x1dFF`
    /// (`dip.cpp:1475-1482`). Its own variant because `isMxfp8ComputeOp` is its own predicate
    /// (`ProgramArtifacts.cpp:46-48`) and folding it into [`Precision::Fp8`] would emit the wrong bias.
    Mxfp8,
}

// ─────────────────────────────────────────────────────────────────────────────
// WHAT THE SPR TABLE PINS
// ─────────────────────────────────────────────────────────────────────────────

/// ⛔ NO UNIT TAKES AN ALL-ZERO SPR0, because a zero mask changes nothing and the C++ writes SPR0 for every unit.
const _: () = {
    use crate::isa::regfile::Component;
    let mut at = 0;
    while at < Component::ALL.len() {
        let inits = of_unit(Component::ALL[at], Precision::Fp16);
        assert!(inits.get(SprSlot::of(0)).get() != 0);
        at += 1;
    }
};

/// ⭐ AND THE MASK IS THE LOW BYTE: every value this table states has `0xFF` there, so every bit of the setting is
/// actually applied (`dip.cpp:1409-1411`).
const _: () = {
    use crate::isa::regfile::Component;
    let mut at = 0;
    while at < Component::ALL.len() {
        let inits = of_unit(Component::ALL[at], Precision::Fp8);
        assert!(inits.get(SprSlot::of(0)).get() & 0xFF == 0xFF);
        at += 1;
    }
};

/// ⛔ THE PT's fp8 SPR0 IS NOT ITS fp16 ONE — `0x81FF` against `0x01FF` — so a build that took one for the other
/// would configure the systolic array for the wrong precision.
const _: () = assert!(
    of_unit(crate::isa::regfile::Component::Pt, Precision::Fp8)
        .get(SprSlot::of(0))
        .get()
        != of_unit(crate::isa::regfile::Component::Pt, Precision::Fp16)
            .get(SprSlot::of(0))
            .get()
);
