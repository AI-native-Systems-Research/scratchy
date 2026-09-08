// SPDX-License-Identifier: Apache-2.0
//! THE IBUFF SLICES — a unit's instruction words, RETURN-terminated, padded, and packed 16 bytes at a time.
//!
//! Bridge 3 decides WHERE things go. Here that is: how wide one instruction is, how many of them fit in a
//! 16-byte slice, how the stream is padded to a whole number of slices, and which word does the padding. Every
//! one of those is a hardware fact READ from `dip.cpp` rather than chosen, and each carries its citation.
//!
//! # THREE CASES, NOT ONE
//!
//! * **32-bit units** (L3 / LX / L0 / PT) — four instructions to a slice, padded to a multiple of FOUR
//!   (`padInstructionsOfSenComp`, `dip.cpp:292-320`'s `else` arm, `dip.cpp:406-448`).
//! * **64-bit units** (PE / SFP) — each instruction emits two 32-bit words LOW-THEN-HIGH (`dip.cpp:1685`), so a
//!   slice holds TWO of them and the stream is padded to a multiple of TWO (same function, `dip.cpp:292-320`).
//! * **one PT ROW** — `constructIBuffFlitsForPTRow` (`dip.h:414`, `dip.cpp:4078-4095`) does NOT pad. It
//!   `DT_CHECK`s that the count is already a multiple of four (`dip.cpp:4083`), because an earlier stage
//!   (`padInstructionsOfSenComp`, the two functions above) already did the padding.
//!
//! ⛔ THE TWO WIDTHS ARE TWO QUANTITIES AND THEY ARE KEPT APART BY TYPE. [`Word32`] and [`Word64`] have no
//! conversion between them, so a 64-bit instruction cannot reach the 32-bit packer at all — the reference port
//! had to refuse that at run time (`IbuffRefusal::WordTooWide`) because it carried both widths in one
//! `InstrWord`. The low 32 bits of a PE instruction are a DIFFERENT instruction that encodes perfectly cleanly,
//! which is exactly the corruption a type can make unrepresentable.
//!
//! ⛔ AND THE PAD MULTIPLE IS PER WIDTH, NOT GLOBAL. The reference records that this code once used the
//! multiple-of-four rule for PE/SFP as well: that over-pads by a WHOLE FLIT whenever the count mod 4 is 1 or 2,
//! and leaves the header's IBUFF flit count describing a stream that is not there.
//!
//! # ⭐ WHAT OF dxp's GENERALITY IS GONE, AND WHY IT CANNOT ARISE
//!
//! dxp is handed one descriptor at a time, so it asks at run time what we already know. Each of these is
//! deliberately absent, not overlooked:
//!
//! * **The ISA lookup.** The reference read the pad from `Isa::opcodeToValue` per unit (`isa.hpp:264`) and had a
//!   `NoReturnOpcode` refusal for a unit whose table lacked `RETURN`. The values are per-unit CONSTANTS here
//!   ([`ReturnPad32`], [`ReturnPad64`]), so a missing entry is not a state this crate can be in — and the one
//!   unit whose value the reference never reported is a `panic!`, not a default.
//! * **`coreArch` at run time.** dxp reads `dscGlobal->sysDef.coreArch` to choose `numPad`; the arch is a cargo
//!   FEATURE here, so the comparison is made once by cargo and [`PE_SFP_PAD_FLOOR`] is a `const`.
//! * **The width refusal.** `IbuffRefusal::WordTooWide` existed because the reference carried both widths in one
//!   `InstrWord` and had to check at run time. [`Word32`] and [`Word64`] are different types with no conversion,
//!   so the call that would truncate does not compile.
//! * **`Result` and `Option` throughout.** `Result<_, IbuffRefusal>`, a `String` error for the PT row's
//!   `DT_CHECK`, and `for_field() -> Option<Bits<7>>` are all gone: there is no run time to recover in, so every
//!   refusal is a `panic!` — which is a compile error — and every bound is a `const` assertion below.
//! * **The flit count as a returned value.** The reference returned `(Vec<Slice>, u8)`; the count is DERIVED from
//!   the slices ([`Ibuff::flit_count`]) and from the instruction count ([`InstrWidth::flit_count`]), which are
//!   pinned equal below. Two places to state one length is how a header comes to describe a stream that is not
//!   there.
//!
//! ⛔ WHAT IS *NOT* COLLAPSED, and must not be: the per-unit arms. Which unit a program belongs to varies per op
//! — it is what island 3 states — so `of_unit`'s split is a real distinction, not dxp asking a question we have
//! already answered.
//!
//! ⭐ AND THE PADDING ARITHMETIC IS ALREADY CONST-EVALUABLE: [`InstrWidth::padded_instr_count`] and
//! [`InstrWidth::flit_count`] are `const fn` and the assertions at the foot of this file are the compiler running
//! them. What stays a loop is only the COPYING, because the program arrives as a slice of a `Tape` — a `Vec`,
//! whose length is not a literal at any call site in this crate (see the crate's note that the "shapes are const
//! generics" rule was removed for exactly this reason). Allocating at expansion time is not the forbidden thing;
//! searching is, and nothing here searches.

use crate::isa::regfile::Component as RegUnit;
use crate::islands::init_packet::Slice;

/// How wide one instruction is in a unit's IBUFF.
///
/// `dip.cpp:611-613` splits on the unit, and `dip.cpp:651`'s `if (slice_id == 4 || slice_id == 5)` is commented
/// "PE and SFP have 64-bit instructions" — those are the PE and SFP slice columns.
///
/// ⛔ A `bool is_64` WOULD SHARE A TYPE WITH EVERY OTHER FLAG in the packet builder, so it could be passed to the
/// wrong parameter and read as the wrong fact. Two named states cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrWidth {
    /// Every unit but PE and SFP: L3, LX, L0 and the PT.
    Bits32,
    /// PE and SFP alone.
    Bits64,
}

/// How many BITS one instruction of a given width occupies.
///
/// Its own type because it is the factor that ties [`InstrsPerSlice`] to [`Slice::BYTES`]: 4 × 32 and 2 × 64 are
/// both 128 bits, and a pair of hand-written counts without the widths they belong to is how the two disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstrBits(u32);

impl InstrBits {
    /// The number of bits itself. Minted only by [`InstrWidth::instr_bits`] — a width the C++ does not state is
    /// not a width.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// HOW MANY INSTRUCTIONS ONE 16-BYTE SLICE HOLDS — four of the narrow ones, two of the wide ones.
///
/// ⭐ ONE TYPE, NOT TWO, AND THE WIDTH SUPPLIES THE VALUE. The two numbers are the same quantity measured for two
/// widths, so a second newtype whose body was `Self(other.0)` would only add a way to launder one into the other.
/// What stops the 32-bit rule reaching a PE stream is that the value comes from
/// [`InstrWidth::instrs_per_slice`], and the width comes from [`InstrWidth::of_unit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstrsPerSlice(usize);

impl InstrsPerSlice {
    /// The count itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// HOW MANY INSTRUCTIONS a unit's program holds.
///
/// ⛔ NOT A BYTE COUNT, NOT A SLICE INDEX, AND NOT A WORD COUNT. For a 32-bit unit an instruction is one word and
/// the two counts coincide numerically, which is precisely why they need different types: the same reading
/// applied to a PE program is out by a factor of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstrCount(usize);

impl InstrCount {
    /// A count of instructions.
    pub const fn of(instructions: usize) -> Self {
        Self(instructions)
    }

    /// The count itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// HOW MANY 32-BIT WORDS are being packed — the count [`Slice`] geometry is measured in.
///
/// A 64-bit instruction contributes TWO of these ([`Word64::halves`]), so this is never interchangeable with
/// [`InstrCount`]: the whole reason the wide path exists is that the two differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Word32Count(usize);

impl Word32Count {
    /// A count of 32-bit words.
    pub const fn of(words: usize) -> Self {
        Self(words)
    }

    /// The count itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// HOW MANY IBUFF FLITS a unit's block declares.
///
/// One IBUFF slice occupies one flit in the unit's slice COLUMN, so the flit count and the slice count are one
/// number — the header carries it as `pktIbuff_flit`. It is a COUNT OF FLITS: not a count of instructions, not a
/// slice index, not a byte size. The reference's builders handed it back as a bare `u8` beside the slices; it
/// stops being bare here, at the first place it could be confused with any of those.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IbuffFlitCount(usize);

impl IbuffFlitCount {
    /// The width of the header's IBUFF flit count — SEVEN BITS, `initpacket.cpp:224`'s
    /// `putU32BitRange(uvalue, 78, 84, numIbuffFlits)`, read back the same way at `:494`.
    pub const FIELD_BITS: u32 = 7;

    /// The largest count the header field can declare.
    ///
    /// ⛔ NOT `u8::MAX`. The count is packed into a 7-bit field, and a wider value's high bits land in
    /// `target_unit_mask` above it — the block would go to the WRONG UNITS, not merely declare the wrong length.
    pub const MAX: usize = (1 << Self::FIELD_BITS) - 1;

    /// Whether a count can be declared at all.
    ///
    /// A predicate rather than an `Option`-returning constructor: the const assertions pin BOTH answers from this
    /// one reading, so the refusal below cannot drift from what the field actually holds.
    pub const fn fits_field(flits: usize) -> bool {
        flits <= Self::MAX
    }

    /// The count of IBUFF flits, refusing one the header cannot declare.
    ///
    /// ⛔ A `panic!` AND NOT A `Result`, because there is no run time here: this crate expands inside scratchy's
    /// proc macro, so refusing IS a compile error and a program too long for the field never gets built. The
    /// consequence of not refusing is not a wrong length — see [`Self::MAX`].
    pub const fn of(flits: usize) -> Self {
        assert!(
            Self::fits_field(flits),
            "an IBUFF longer than the header's 7-bit flit count can declare (`initpacket.cpp:224` writes it into bits \
             78..84): the high \
             bits land in target_unit_mask, so the block would be delivered to the wrong units"
        );
        Self(flits)
    }

    /// The count itself.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// ONE 32-BIT INSTRUCTION WORD, already encoded — an L3 / LX / L0 / PT instruction.
///
/// ⛔ NOT A WORD OF A WIDER INSTRUCTION. [`Word64::halves`] mints these from a 64-bit instruction on purpose and
/// says so in its name; there is no conversion the other way, so a PE instruction cannot arrive here truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word32(u32);

impl Word32 {
    /// An already-encoded 32-bit instruction word.
    pub const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// The bits. This is the sanctioned place a word becomes an integer again — a flit IS integers.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ONE 64-BIT INSTRUCTION WORD, already encoded — a PE or SFP instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word64(u64);

impl Word64 {
    /// An already-encoded 64-bit instruction word.
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    /// The bits.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The two 32-bit words this instruction occupies, **LOW THEN HIGH** (`dip.cpp:1685`).
    ///
    /// ⛔ THE ORDER IS THE WHOLE CONTENT OF THIS FUNCTION. High-then-low produces a file of exactly the right
    /// length in which every PE instruction is a different instruction, so nothing about the shape of the output
    /// can reveal the mistake — only this citation and the assertion at the foot of the file.
    pub const fn halves(self) -> [Word32; 2] {
        [
            Word32::new(self.0 as u32),
            Word32::new((self.0 >> 32) as u32),
        ]
    }
}

/// THE 32-BIT RETURN WORD that pads a narrow unit's IBUFF (`inst.instn_ = OpCodeT::RETURN`, `dip.cpp:292-320`).
///
/// ⛔ THE PAD IS THE UNIT'S OWN. SFP/PE/LX/L0 use `0x3f` and L3 uses `0x37`; the attempt before the reference
/// wrote ONE of them down, which pads an L3 block with the compute units' terminator and a PE block with L3's —
/// an instruction the unit does not have. So this enumerates the units, and a unit whose value the reference does
/// not state is a refusal rather than a guess.
///
/// ⭐ THESE ARE CONSTANTS BECAUSE dxp'S RUNTIME LOOKUP IS OUR CONSTANT. The reference read them from
/// `Isa::opcodeToValue` (declared at `isa.hpp:264`, POPULATED IN `isa.cpp`, which this crate does not vendor) and
/// reported the value it found per unit. When that table reaches `crate::generated`, this function becomes a read
/// of it; until then the enumeration is the only place either value appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReturnPad32(Word32);

impl ReturnPad32 {
    /// This unit's RETURN word, from the PORTED opcode table.
    ///
    /// ⛔⛔ THIS USED TO HOLD ITS OWN COPY OF THE VALUES — `0x3f` for the LX/L0 units, `0x37` for the L3 — and
    /// REFUSED for the PT, "the PT's RETURN opcode VALUE is not stated by the reference port". It is stated:
    /// [`crate::isa::values`] is the ported `Isa::opcodeToValue`, and it gives `RETURN` as 55 on the L3 and 63 on
    /// the PT, with `assert!(OpUnit::Ptop.value_of(InstOpCode::RETURN).get() == 63)` pinning the latter. So this was
    /// ONE QUANTITY IN TWO PLACES, with a hole in the copy where the original has a value — and `0x3f` is 63, the
    /// number the refusal warned against guessing.
    ///
    /// ⭐ READING THE TABLE MAKES THE PAD FOLLOW THE ISA, so a unit whose opcode value changes cannot end up with a
    /// stale pad, and no unit can be missing one.
    pub const fn of_unit(unit: RegUnit) -> Self {
        match unit {
            RegUnit::Sfp | RegUnit::Pe => panic!(
                "the SFP and PE hold 64-bit instructions (dip.cpp:655-657), so their RETURN pad is a \
                 ReturnPad64 — a 32-bit pad here would leave every second word of the tail unwritten"
            ),
            RegUnit::Lxlu
            | RegUnit::Lxsu
            | RegUnit::L0lu
            | RegUnit::L0su
            | RegUnit::L3lu
            | RegUnit::L3su
            | RegUnit::Pt => Self(Word32::new(
                crate::isa::values::OpUnit::of_component(unit)
                    .value_of(crate::isa::InstOpCode::RETURN)
                    .get() as u32,
            )),
        }
    }

    /// The word itself.
    pub const fn word(self) -> Word32 {
        self.0
    }
}

/// THE 64-BIT RETURN WORD that pads a PE or SFP IBUFF (`inst.instn_ = OpCodeT::RETURN`, `dip.cpp:292-320`).
///
/// Separate from [`ReturnPad32`] for the same reason [`Word64`] is separate from [`Word32`]: the wide path pads in
/// whole 64-bit instructions, and a 32-bit pad would fill only half of each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReturnPad64(Word64);

impl ReturnPad64 {
    /// This unit's RETURN word, or a refusal naming the units that have no wide instructions.
    pub const fn of_unit(unit: RegUnit) -> Self {
        match unit {
            RegUnit::Sfp | RegUnit::Pe => Self(Word64::new(0x3f)),
            RegUnit::L3lu
            | RegUnit::L3su
            | RegUnit::Lxlu
            | RegUnit::Lxsu
            | RegUnit::L0lu
            | RegUnit::L0su
            | RegUnit::Pt => panic!(
                "only the SFP and PE hold 64-bit instructions (dip.cpp:655-657); every other unit's \
                 RETURN pad is a ReturnPad32, and packing a narrow unit two-per-slice would halve its IBUFF"
            ),
        }
    }

    /// The word itself.
    pub const fn word(self) -> Word64 {
        self.0
    }
}

/// `numPad` — THE FLOOR a PE/SFP program is padded up to (`dip.cpp:292-320`):
/// `int numPad = (dscGlobal->sysDef.coreArch > IsaCoreGen::RCUDD1A_ISA) ? 4 : 2;`
///
/// ⭐ THE ARCH IS A CARGO FEATURE HERE, so the comparison the C++ makes at run time is made by cargo. On RCUDD1A
/// the floor is inert — one instruction plus its terminator is already two — and it BITES on SEN1P5, where a
/// one-instruction program must reach four. The reference declined to port this arm, which leaves SEN1P5 silently
/// padding to two; both values are stated in the C++ it quotes verbatim, so both are ported.
#[cfg(feature = "arch-rcudd1a")]
const PE_SFP_PAD_FLOOR: InstrCount = InstrCount::of(2);
/// `numPad` on every arch beyond RCUDD1A — citation on the RCUDD1A definition.
#[cfg(not(feature = "arch-rcudd1a"))]
const PE_SFP_PAD_FLOOR: InstrCount = InstrCount::of(4);

impl InstrWidth {
    /// The width this unit's instructions occupy — `dip.cpp:655-657`, whose own comment is
    /// *"PE and SFP have 64-bit instructions"*, and `patchinit.cpp:538-543`, which reaches the same answer as
    /// `numInstrPerIBuffEntry`: 2 per sixteen-byte entry for slices 4 and 5, 4 for the rest.
    ///
    /// ⭐ EVERY UNIT SPELLED OUT, so a tenth component is a build error here rather than reading as narrow. And
    /// the mistake this shape prevents is the plausible one: the PT is a COMPUTE unit like the PE and the SFP,
    /// and it is nonetheless 32-bit.
    ///
    /// ⛔⛔ WHICH IS WHY THIS KEYS ON THE **UNIT** AND NOT ON THE SLICE. The C++ decides by column —
    /// `slice_id == 4 || slice_id == 5` — and therefore needs a second term to stay correct:
    /// `if ((bPTinit == false) && (slice_id == 4 || slice_id == 5))`. It needs it because a PT ROW LIVES IN THE
    /// COLUMN ITS NUMBER NAMES (`Placement::Pt(row) => row.index()`), so columns 4 and 5 hold PT rows 4 and 5
    /// during a PT init — and their instructions are 32-bit. Asking the unit instead makes that case
    /// unrepresentable rather than guarded.
    pub const fn of_unit(unit: RegUnit) -> Self {
        match unit {
            RegUnit::Sfp | RegUnit::Pe => Self::Bits64,
            RegUnit::L3lu
            | RegUnit::L3su
            | RegUnit::Lxlu
            | RegUnit::Lxsu
            | RegUnit::L0lu
            | RegUnit::L0su
            | RegUnit::Pt => Self::Bits32,
        }
    }

    /// How many bits one instruction of this width occupies. The name is the value; it exists so the count below
    /// can be tied to [`Slice::BYTES`] instead of asserted against a literal.
    pub const fn instr_bits(self) -> InstrBits {
        match self {
            Self::Bits32 => InstrBits(32),
            Self::Bits64 => InstrBits(64),
        }
    }

    /// How many instructions of this width one 16-byte slice holds — FOUR narrow, TWO wide.
    ///
    /// This is also the multiple the stream is padded to (`dip.cpp:292-320`): 32-bit units pad to a multiple of
    /// four and PE/SFP to a multiple of two, which is to say both pad to WHOLE SLICES. One reading, so the pad
    /// rule and the packing geometry cannot disagree.
    pub const fn instrs_per_slice(self) -> InstrsPerSlice {
        match self {
            Self::Bits32 => InstrsPerSlice(4),
            Self::Bits64 => InstrsPerSlice(2),
        }
    }

    /// The instruction count after the RETURN terminator and the pad (`padInstructionsOfSenComp`,
    /// `dip.cpp:292-320`, `:406-448`).
    ///
    /// An empty program stays empty: a unit with no instructions declares no IBUFF flits, so there is nothing to
    /// terminate.
    ///
    /// ⛔ AND THAT BRANCH IS NOT DEFENSIVE. A unit's program is a FILTER over the group's tape, so a group with no
    /// PE op yields an empty PE program — the case is reachable by construction, not a guard against a value that
    /// cannot occur. It is also the one place dxp's floor arm would disagree: `instrCount = 0 < numPad` gives it a
    /// target of two RETURNs, which the reference does not do, because a unit with no program gets no IBUFF region
    /// for those RETURNs to live in.
    ///
    /// ⛔ IT ROUNDS UP, NEVER DOWN. Truncating a partial slice would drop real instructions while producing a file
    /// of a perfectly legal length — and a check that only tries exact multiples of four cannot tell the two
    /// apart, which is why the assertions below try 1, 3, 5 and 7.
    ///
    /// Two places this port differs from the C++ deliberately, both recorded by the reference:
    ///
    /// * dxp computes its target from the ORIGINAL count and fills the gap with RETURNs; it does not append a
    ///   terminator first. Our streams carry no trailing RETURN of their own, so the terminator is added here and
    ///   the round-up applies to `n + 1`. For a stream whose generator already ended in RETURN the two agree.
    /// * dxp compares the ORIGINAL count against the floor; here the comparison happens after the round-up, which
    ///   is the same answer for every count because the floor is even.
    pub const fn padded_instr_count(self, program: InstrCount) -> InstrCount {
        // ⛔⛔ THERE IS NO UNCONDITIONAL TERMINATOR. `padInstructionsOfSenComp` (`dip.cpp:292-333`) pads ONLY when
        // the count is not already aligned: its 32-bit arm is `if (instrCount % 4 != 0)` and its PE/SFP arm is
        // `targetCount = (instrCount % 2 == 0) ? instrCount : instrCount + 1`. Adding a RETURN first would make
        // every ALREADY-ALIGNED program one whole slice longer — a 4-instruction L0 program becomes 8, and the
        // extra RETURNs are instructions the unit executes.
        //
        // ⭐ THE PAD INSTRUCTION IS `RETURN` WITH NO FIELDS (`:312-316`, `:326-331`), which is why a pad is
        // harmless where an extra one is not: it ends the program, so a program padded one slice too far runs a
        // RETURN and then three more.
        match self {
            // The C++'s `else` arm: round up to a multiple of four, and nothing when it already is.
            Self::Bits32 => InstrCount::of(program.get().next_multiple_of(4)),
            // `numPad` first, then even — and note the C++ pads an EMPTY program up to `numPad` rather than
            // leaving it empty, because the floor is compared against the count itself.
            Self::Bits64 => {
                if program.get() < PE_SFP_PAD_FLOOR.get() {
                    PE_SFP_PAD_FLOOR
                } else {
                    InstrCount::of(program.get().next_multiple_of(2))
                }
            }
        }
    }

    /// How many IBUFF flits a program of this many instructions occupies, padding included.
    ///
    /// Derived from [`Self::padded_instr_count`] rather than counted after the fact, so the header's declared
    /// length and the stream's real length are ONE number. The reference returned the count beside the slices,
    /// which is a second place the length is decided.
    pub const fn flit_count(self, program: InstrCount) -> IbuffFlitCount {
        IbuffFlitCount::of(self.padded_instr_count(program).get() / self.instrs_per_slice().get())
    }
}

/// How many 32-bit words one 16-byte slice holds.
///
/// ⭐ DERIVED, NOT WRITTEN. A 32-bit instruction is exactly one word, so the narrow width's
/// instructions-per-slice IS the words-per-slice — which is why the wide path expands to [`Word32`] halves first
/// and then reuses the very same packer instead of having a geometry of its own.
pub const WORDS_PER_SLICE: usize = InstrWidth::Bits32.instrs_per_slice().get();

/// Whether a word count is a whole number of slices.
///
/// The one reading behind both the packer's refusal and [`ibuff_flits_for_pt_row`]'s `DT_CHECK`
/// (`dip.cpp:4083`), so the two cannot come to disagree about what "already padded" means.
pub const fn is_whole_slices(words: Word32Count) -> bool {
    words.get().is_multiple_of(WORDS_PER_SLICE)
}

/// A unit's IBUFF: the 16-byte slices its instruction words occupy.
///
/// ⛔ THE FLIT COUNT IS NOT A FIELD. It is what the header declares and what the reader uses to walk the block,
/// so storing it beside the slices would be a second place the length is decided — and a disagreement there does
/// not corrupt one block, it makes the header describe a stream that is not there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ibuff(Vec<Slice>);

impl Ibuff {
    /// The slices, in the order they are laid into the unit's slice column.
    pub fn slices(&self) -> &[Slice] {
        &self.0
    }

    /// The flit count the header must declare — DERIVED from the slices, one flit per slice.
    pub fn flit_count(&self) -> IbuffFlitCount {
        IbuffFlitCount::of(self.0.len())
    }
}

/// Pack 32-bit words four to a slice.
///
/// A `Vec`, and that is not a compromise: this crate runs inside scratchy's proc macro, so the allocation happens
/// at compile time. What "everything is a constant" forbids is SEARCHING, and nothing here searches.
///
/// ⛔ A PARTIAL SLICE IS A REFUSAL, NOT A ZERO-FILL. Zero-filling the tail writes a full slice the unit decodes as
/// an instruction it never issued, and the file's length would be exactly right.
fn pack_words_into_slices(words: &[Word32]) -> Vec<Slice> {
    assert!(
        is_whole_slices(Word32Count::of(words.len())),
        "an IBUFF word count that is not a whole number of 16-byte slices reached the packer; the pad is \
         InstrWidth::padded_instr_count's job (dip.cpp:292-320) and zero-filling the tail here would emit an \
         instruction the unit never issued"
    );
    words
        .chunks(WORDS_PER_SLICE)
        .map(|slice| {
            Slice([
                slice[0].get(),
                slice[1].get(),
                slice[2].get(),
                slice[3].get(),
            ])
        })
        .collect()
}

/// Pack a 32-bit unit's (L3 / LX / L0 / PT) instruction stream into IBUFF slices, RETURN-terminated and padded to
/// a multiple of four (`padInstructionsOfSenComp`, `dip.cpp:406-448`, and `:292-320`'s `else` arm).
///
/// The pad word is the unit's own, and [`ReturnPad32`] is the only thing that can supply it — which is also what
/// stops a PE or SFP program being packed four-to-a-slice, since no `ReturnPad32` exists for those units.
pub fn ibuff_slices_32(program: &[Word32], pad: ReturnPad32) -> Ibuff {
    if program.is_empty() {
        // No instructions, no IBUFF: there is nothing to terminate.
        return Ibuff(Vec::new());
    }
    let padded = InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(program.len()))
        .get();
    let mut words: Vec<Word32> = Vec::with_capacity(padded);
    words.extend_from_slice(program);
    while words.len() < padded {
        words.push(pad.word());
    }
    Ibuff(pack_words_into_slices(&words))
}

/// Pack a 64-bit unit's (PE / SFP) instruction stream into IBUFF slices, RETURN-terminated and padded to a
/// multiple of TWO (`padInstructionsOfSenComp`, `dip.cpp:292-320`), each instruction emitting two 32-bit words
/// low-then-high (`dip.cpp:1685`).
///
/// ⛔ THE PADDING HAPPENS IN INSTRUCTIONS, BEFORE THE SPLIT. Padding after the split would let a stream end on the
/// low half of an instruction whose high half is a pad word — half of a RETURN and half of something else.
pub fn ibuff_slices_64(program: &[Word64], pad: ReturnPad64) -> Ibuff {
    if program.is_empty() {
        return Ibuff(Vec::new());
    }
    let padded = InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(program.len()))
        .get();
    let mut instrs: Vec<Word64> = Vec::with_capacity(padded);
    instrs.extend_from_slice(program);
    while instrs.len() < padded {
        instrs.push(pad.word());
    }
    let mut words: Vec<Word32> = Vec::with_capacity(instrs.len() * 2);
    for instr in instrs {
        let [low, high] = instr.halves();
        words.push(low);
        words.push(high);
    }
    Ibuff(pack_words_into_slices(&words))
}

/// Pack ONE PT ROW's already-padded instruction words into IBUFF slices (`Dip::constructIBuffFlitsForPTRow`,
/// `dip.h:414`, `dip.cpp:4078-4095`).
///
/// ⛔ THIS ONE DOES NOT PAD, and that is the whole difference from [`ibuff_slices_32`]. The real function
/// `DT_CHECK`s `instrCount % 4 == 0` (`dip.cpp:4083`) over the WHOLE `senCompProgram_[senComp]` vector, which
/// `padInstructionsOfSenComp` has already terminated and padded. Padding again here would append RETURNs after the
/// row's terminator.
///
/// The caller in the C++ is `Dip::constructPTInitSnt1p5` (`dip.cpp:4163+`), which builds one such list per PT row
/// before diffing rows 1..3 against row 0 in `constructIBuffCorrectionForPT` (`dip.cpp:4097-4161`) — that diff is
/// not ported here, and neither is the number of rows, which the reference does not state for this arch.
pub fn ibuff_flits_for_pt_row(program: &[Word32]) -> Ibuff {
    assert!(
        is_whole_slices(Word32Count::of(program.len())),
        "constructIBuffFlitsForPTRow received a PT-row instruction count that is not a multiple of four \
         (DT_CHECK, dip.cpp:4083); this function does not pad — padInstructionsOfSenComp already did"
    );
    Ibuff(pack_words_into_slices(program))
}

// ─── THE INVARIANTS, AS COMPILE ERRORS ──────────────────────────────────────────────────────────────────────
//
// ⛔ AND EACH ONE PINS THE WRONG ANSWER AS ABSENT. `assert!(x == right)` passes for code that computes `right` by
// accident; `assert!(x != wrong)` names the mistake that was actually made.

/// PE and SFP are the 64-bit units (`dip.cpp:611-613`, `:651`).
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::Sfp),
    InstrWidth::Bits64
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::Pe),
    InstrWidth::Bits64
));

/// ⛔ AND THE PT IS NOT, though it is a compute unit like the other two — the plausible reading is the wrong one.
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::Pt),
    InstrWidth::Bits32
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::L3lu),
    InstrWidth::Bits32
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::L3su),
    InstrWidth::Bits32
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::Lxlu),
    InstrWidth::Bits32
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::Lxsu),
    InstrWidth::Bits32
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::L0lu),
    InstrWidth::Bits32
));
const _: () = assert!(matches!(
    InstrWidth::of_unit(RegUnit::L0su),
    InstrWidth::Bits32
));

/// FOUR narrow instructions to a slice, TWO wide ones.
const _: () = assert!(InstrWidth::Bits32.instrs_per_slice().get() == 4);
const _: () = assert!(InstrWidth::Bits64.instrs_per_slice().get() == 2);

/// ⛔ AND THE TWO COUNTS ARE NOT INTERCHANGEABLE — four wide instructions are two slices, not one.
const _: () = assert!(
    InstrWidth::Bits32.instrs_per_slice().get() != InstrWidth::Bits64.instrs_per_slice().get()
);
const _: () = assert!(
    InstrWidth::Bits32.instrs_per_slice().get() == 2 * InstrWidth::Bits64.instrs_per_slice().get()
);

/// ⭐ THE CONSERVATION LAW BEHIND BOTH: instructions-per-slice times instruction width IS the slice, for each
/// width. That is what a pair of hand-written literals cannot state — 4 with 64 bits, or 2 with 32, would overrun
/// the slice or fill half of it, and both are ruled out here rather than in one direction only.
const _: () = assert!(
    InstrWidth::Bits32.instrs_per_slice().get() * InstrWidth::Bits32.instr_bits().get() as usize
        == Slice::BYTES * 8
);
const _: () = assert!(
    InstrWidth::Bits64.instrs_per_slice().get() * InstrWidth::Bits64.instr_bits().get() as usize
        == Slice::BYTES * 8
);
const _: () = assert!(
    InstrWidth::Bits32.instrs_per_slice().get() * InstrWidth::Bits64.instr_bits().get() as usize
        != Slice::BYTES * 8,
    "four 64-bit instructions are TWO slices; if this passes, the two counts have been swapped"
);
const _: () = assert!(
    InstrWidth::Bits64.instrs_per_slice().get() * InstrWidth::Bits32.instr_bits().get() as usize
        != Slice::BYTES * 8,
    "two 32-bit instructions fill HALF a slice"
);

/// A word count is a whole number of slices only at multiples of four.
///
/// ⛔ BOTH ANSWERS PINNED. A layout that only works for exact multiples passes any check that only tries exact
/// multiples, so 1, 3, 5 and 7 are here on purpose.
const _: () = assert!(is_whole_slices(Word32Count::of(0)));
const _: () = assert!(is_whole_slices(Word32Count::of(4)));
const _: () = assert!(is_whole_slices(Word32Count::of(8)));
const _: () = assert!(!is_whole_slices(Word32Count::of(1)));
const _: () = assert!(!is_whole_slices(Word32Count::of(3)));
const _: () = assert!(!is_whole_slices(Word32Count::of(5)));
const _: () = assert!(!is_whole_slices(Word32Count::of(7)));

/// ⛔⛔ AN ALREADY-ALIGNED PROGRAM IS NOT PADDED AT ALL. `padInstructionsOfSenComp` (`dip.cpp:292-333`) pads only
/// when `instrCount % 4 != 0` (32-bit) or when the count is odd (PE/SFP). Terminating first and then rounding up
/// would grow every aligned program by a whole slice — four L0 instructions to eight — and those extra RETURNs are
/// instructions the unit executes.
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(4))
        .get()
        == 4
);
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(8))
        .get()
        == 8
);
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(4))
        .get()
        == 4
);
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(6))
        .get()
        == 6
);

/// The narrow rule: round UP to a multiple of four, with RETURNs (`dip.cpp:326-331`).
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(1))
        .get()
        == 4
);
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(3))
        .get()
        == 4
);
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(7))
        .get()
        == 8
);
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(9))
        .get()
        == 12
);

/// ⛔ PADDED, NEVER TRUNCATED. A count below the next multiple grows; truncation would keep the file legal and
/// lose instructions.
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(3))
        .get()
        > 3
);
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(5))
        .get()
        > 5
);

/// The wide rule: the `numPad` floor first, then round up to a multiple of two (`dip.cpp:301-317`).
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(3))
        .get()
        == 4
);
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(5))
        .get()
        == 6
);
/// ⭐ AND AN EMPTY PE PROGRAM IS PADDED TO THE FLOOR, not left empty: the C++ compares `instrCount < numPad`
/// against the count itself, so zero takes the floor arm.
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(0))
        .get()
        == PE_SFP_PAD_FLOOR.get()
);
/// A narrow one is left empty, because zero already is a multiple of four.
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(0))
        .get()
        == 0
);
const _: () = assert!(InstrWidth::Bits32.flit_count(InstrCount::of(0)).get() == 0);

/// ⛔⛔ THE MISTAKE THIS FILE EXISTS TO PREVENT: the narrow rule applied to a PE stream. At five instructions it
/// over-pads by two — a WHOLE FLIT — and the header then declares a flit the stream does not contain. Counts
/// congruent to 3 mod 4 agree by coincidence, which is why a discriminating one is pinned.
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(5))
        .get()
        != InstrWidth::Bits64
            .padded_instr_count(InstrCount::of(5))
            .get()
);
const _: () = assert!(
    InstrWidth::Bits32.flit_count(InstrCount::of(5)).get()
        != InstrWidth::Bits64.flit_count(InstrCount::of(5)).get()
);

/// ⛔ AND ON RCUDD1A THE TWO FLIT COUNTS AGREE AT ONE INSTRUCTION — one slice either way. That coincidence is why
/// the count above is discriminated at five, where the narrow rule declares two flits and the wide rule three.
///
/// ⭐ ON SEN1P5 THEY DO NOT, and the reason is `numPad`: the floor of four makes a one-instruction PE program two
/// slices where a narrow one is still one. So this pair is the only assertion in the file whose ANSWER depends on
/// the arch, and it is stated for both rather than left to whichever cargo happened to select.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    InstrWidth::Bits32.flit_count(InstrCount::of(1)).get()
        == InstrWidth::Bits64.flit_count(InstrCount::of(1)).get()
);
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(InstrWidth::Bits32.flit_count(InstrCount::of(1)).get() == 1);
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(InstrWidth::Bits64.flit_count(InstrCount::of(1)).get() == 2);

/// `numPad`, the PE/SFP floor (`dip.cpp:292-320`): a one-instruction program reaches it. RCUDD1A's 2 is inert
/// because the terminator already gets there; SEN1P5's 4 is not, and pinning it per arch is what keeps the
/// unported branch from coming back.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(PE_SFP_PAD_FLOOR.get() == 2);
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(1))
        .get()
        == 2
);
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(PE_SFP_PAD_FLOOR.get() == 4);
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(1))
        .get()
        == 4
);

/// The flit count is the padded stream divided by the slice's capacity — measured in INSTRUCTIONS, which for the
/// narrow units is also words and for the wide ones is HALF the words.
const _: () = assert!(InstrWidth::Bits32.flit_count(InstrCount::of(7)).get() == 2);
/// ⭐ AND EIGHT NARROW INSTRUCTIONS ARE TWO SLICES, NOT THREE. This assertion said three, which is the terminator
/// fabrication measured in flits: it declared a slice the stream does not contain, and the header is what the
/// runtime reads to find the next region.
const _: () = assert!(InstrWidth::Bits32.flit_count(InstrCount::of(8)).get() == 2);
const _: () = assert!(InstrWidth::Bits64.flit_count(InstrCount::of(5)).get() == 3);

/// ⭐ AND THE FLIT COUNT IS A COUNT OF SLICES IN BOTH CASES: the padded stream, expanded to 32-bit words, is
/// exactly the slices times four. For the wide width the factor of two is the whole difference, so it appears
/// here explicitly rather than being folded into a second words-per-slice constant.
const _: () = assert!(
    InstrWidth::Bits64
        .padded_instr_count(InstrCount::of(5))
        .get()
        * 2
        == InstrWidth::Bits64.flit_count(InstrCount::of(5)).get() * WORDS_PER_SLICE
);
const _: () = assert!(
    InstrWidth::Bits32
        .padded_instr_count(InstrCount::of(7))
        .get()
        == InstrWidth::Bits32.flit_count(InstrCount::of(7)).get() * WORDS_PER_SLICE
);

/// The header's IBUFF flit count is SEVEN bits — `initpacket.cpp:224`, bits 78..84.
const _: () = assert!(IbuffFlitCount::FIELD_BITS == 7);
const _: () = assert!(IbuffFlitCount::MAX == 127);

/// ⛔ NOT A BYTE'S WORTH. `u8::MAX` here lets the count's high bit into `target_unit_mask` above the field, and
/// the block is then delivered to the wrong units — a failure that does not look like a length error at all.
const _: () = assert!(IbuffFlitCount::MAX != 255);
const _: () = assert!(IbuffFlitCount::fits_field(0));
const _: () = assert!(IbuffFlitCount::fits_field(127));
const _: () = assert!(!IbuffFlitCount::fits_field(128));
const _: () = assert!(!IbuffFlitCount::fits_field(255));

/// LOW WORD FIRST (`dip.cpp:1685`). The reversed order emits a file of identical length in which every PE
/// instruction is a different instruction, so this assertion is the only thing that can catch it.
const _: () = assert!(Word64::new(0x1122_3344_5566_7788).halves()[0].get() == 0x5566_7788);
const _: () = assert!(Word64::new(0x1122_3344_5566_7788).halves()[1].get() == 0x1122_3344);
const _: () = assert!(Word64::new(0x1122_3344_5566_7788).halves()[0].get() != 0x1122_3344);

/// The RETURN pad is the unit's own: `0x3f` for the compute and LX/L0 units, `0x37` for L3.
///
/// ⛔ ASSERTING THEY DIFFER IS THE CHECK THAT MATTERS. One hardcoded pad — the mistake the attempt before the
/// reference made — would make these equal.
const _: () = assert!(ReturnPad32::of_unit(RegUnit::L0lu).word().get() == 0x3f);
const _: () = assert!(ReturnPad32::of_unit(RegUnit::L0su).word().get() == 0x3f);
const _: () = assert!(ReturnPad32::of_unit(RegUnit::Lxlu).word().get() == 0x3f);
const _: () = assert!(ReturnPad32::of_unit(RegUnit::Lxsu).word().get() == 0x3f);
const _: () = assert!(ReturnPad32::of_unit(RegUnit::L3lu).word().get() == 0x37);
const _: () = assert!(ReturnPad32::of_unit(RegUnit::L3su).word().get() == 0x37);
const _: () = assert!(
    ReturnPad32::of_unit(RegUnit::L3lu).word().get()
        != ReturnPad32::of_unit(RegUnit::L0lu).word().get()
);
const _: () = assert!(ReturnPad64::of_unit(RegUnit::Sfp).word().get() == 0x3f);
const _: () = assert!(ReturnPad64::of_unit(RegUnit::Pe).word().get() == 0x3f);

/// ⛔ AND THE WIDE PAD IS NOT L3's, which is what padding a PE block with L3's terminator would give — an
/// instruction the PE does not have.
const _: () = assert!(ReturnPad64::of_unit(RegUnit::Sfp).word().get() != 0x37);

impl Ibuff {
    /// A unit's IBUFF from its encoded instruction words.
    ///
    /// The words are padded to a whole slice with `RETURN`, which is what `padInstructionsOfSenComp`
    /// (`dip.cpp:292-333`) appends: a 32-bit unit pads to a multiple of four, a 64-bit one to `numPad` and then
    /// to an even count. `pad` is the encoded RETURN for the unit, since its opcode value differs per component.
    pub fn of_words(width: InstrWidth, words: &[Word32], pad: Word32) -> Self {
        let per_slice = width.instrs_per_slice().get();
        let mut padded: Vec<Word32> = words.to_vec();
        // A 64-bit instruction is two words, so the word count is what must reach a whole slice either way.
        while !is_whole_slices(Word32Count::of(padded.len())) || padded.len() < per_slice {
            padded.push(pad);
        }
        Self(pack_words_into_slices(&padded))
    }
}
