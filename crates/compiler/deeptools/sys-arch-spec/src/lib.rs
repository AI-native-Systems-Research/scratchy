// SPDX-License-Identifier: Apache-2.0
//! THE ARCHITECTURE SPEC — `/project_src/deeptools/sys-arch-spec/`.
//!
//! What the machine IS, as opposed to what a compiler does with it or what a model does to imitate it: the ISA
//! (`isa/isa.cpp`, `isa/isa.hpp`, `isa/isaSystemc.h`), the register-file and memory sizes (`sysdef.cpp`), and
//! the program IR's register view (`progir/regvisitor.cpp`).
//!
//! ⛔ A LIBRARY BECAUSE BOTH SIDES INCLUDE IT. In the C++ the compiler and the senulator each `#include` these
//! headers; in Rust that has to be a crate, or each side keeps its own copy and the copies drift. They did:
//! `Isa::InstOperand` was 89 enumerators in the compiler and 89 in the model, and that they were in the SAME
//! ORDER was provable only by a hand-written assertion — two enums that pass their own length checks and put
//! every operand of every instruction in the wrong slot.
//!
//! ⭐ THE OPCODE ENUM IS THE AUTHORITY FOR WHAT IS AN OPCODE. `deeptools`'s `build.rs` emits references like
//! `InstOpCode::FMA` for the mnemonics the `.ddl`/`.smc` templates state, so a template naming something that
//! is not an opcode fails to compile.

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// EXACTLY ONE TARGET ISA, OR THE BUILD FAILS.
//
// 🛑 BOTH DEGENERATE FEATURE SETS COMPILED SILENTLY, AND EACH PRODUCED A SPECIFIC MACHINE'S TABLES.
//
// ⛔⛔ THE FIELD TABLES ARE GUARDED AS `#[cfg(feature = "arch-rcudd1a")]` / `#[cfg(not(…))]` — nineteen
// pairs — so `arch-sen1p5` is not read anywhere in `fields.rs`. SEN1P5 is spelled "not RCUDD1A". Two
// things followed, both measured:
//
// * `--no-default-features` with NO arch named compiled, and silently selected the **SEN1P5** tables.
//   Absence of a choice was a choice.
// * BOTH features on compiled, and silently selected **RCUDD1A** — so `--features arch-sen1p5` was a
//   flag that changed nothing.
//
// ⛔ THE SECOND IS THE HAZARD `deeptools`' OWN MANIFEST WARNS ABOUT: *"cargo UNIONS features across a
// graph — so one dependency edge that takes the default pins the whole build to the RCUDD1A field
// tables"*. Every edge there forwards rather than defaults, precisely to avoid it — but a missed
// forward failed SILENTLY, which is compiling a compiler for one machine against the tables of
// another, the thing that manifest says must not happen.
//
// ⭐ SO IT IS A BUILD ERROR, NOT A DIAGNOSTIC ANYONE HAS TO NOTICE. The same rule this project applies
// to every other invariant: a wrong configuration must not be expressible, and a check that only fires
// at run time is not a check.
//
// 🛑 ADDING A THIRD VARIANT — read this first.
//
// ⛔⛔ THE NINETEEN `not(feature = "arch-rcudd1a")` SITES IN `fields.rs` ARE ONLY CORRECT WHILE THERE
// ARE TWO ARCHES. With a third, "not RCUDD1A" stops meaning SEN1P5 and the new arch silently inherits
// SEN1P5's field table — the wrong answer as the DEFAULT. A new variant therefore means either
// converting those complements to positive `cfg`s, or the real fix: giving a field row the `min_arch`
// column an opcode row already carries (`DefOpcode::min_arch`), so one table serves every arch and a
// variant is an ordinal rather than a copy.
//
// ⭐ THE `any(…)` BELOW IS WHAT FORCES THAT READ: a new feature enabled alone matches none of the known
// arches, so the build stops here rather than in the tables.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

#[cfg(not(any(feature = "arch-rcudd1a", feature = "arch-sen1p5")))]
compile_error!(
    "no target ISA selected: enable exactly one of `arch-rcudd1a` or `arch-sen1p5`. \
     Selecting none used to compile and silently choose the SEN1P5 field tables, because they are \
     guarded by `not(feature = \"arch-rcudd1a\")`. If you are adding a new Spyre variant, read the \
     note above this check in lib.rs first: the nineteen complement sites in fields.rs are only \
     correct while there are two arches."
);

#[cfg(all(feature = "arch-rcudd1a", feature = "arch-sen1p5"))]
compile_error!(
    "two target ISAs selected: `arch-rcudd1a` and `arch-sen1p5` are mutually exclusive, and cargo \
     unions features across the dependency graph. This used to compile and silently choose RCUDD1A, \
     making `--features arch-sen1p5` a flag that changed nothing. Every dependency edge must pass \
     `default-features = false` and forward the choice rather than making it."
);

/// The architecture's enumerations — `arch_enums.h`: the register files, the LX segments, the `{unit, storage}`
/// pair every location is named by, the 176-name `OpFunc` vocabulary, and `SenComponents` itself.
pub mod arch_enums;
/// THE INSTRUCTION LAYOUT TABLES — which fields each instruction type has and where their bits are. A port of
/// `Isa::initIsa` (`isa.cpp:161-1221`): 440 opcode definitions, 1075 field definitions.
pub mod fields;
/// How much each memory holds — the capacities an allocation is placed within (`sysdef.cpp`).
pub mod memory;
/// `Isa::InstOperand` — every field an instruction word can have, in the C++ enum's DECLARATION ORDER, which is
/// the order a senprog's terms are written in. Names as variants means no strings.
pub mod operand;
/// The program IR's own bounds — `progir/progir.h`'s `ProgramAndStateInfo`.
pub mod progir;

/// WHICH CORE — `arch_enums.h`'s core identity.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct CoreId(pub u8);

/// A STICK IS 128 BYTES — `STICK_BYTESIZE`.
///
/// ⛔ THE DECLARING HEADER IS `util/sendefs/dataType.h`, WHICH IS A FOURTH DIRECTORY. `util/` is its own library
/// in the C++ and will want its own crate; this constant is here rather than in one created for three values,
/// and it is a fact about the machine either way. Move it when `util/` earns a crate.
pub const STICK_BYTES: u32 = 128;

/// A stick is 1024 bits.
pub const STICK_BITS: u32 = STICK_BYTES * 8;

/// THE SFP HAS EIGHT SLICES.
pub const SFP_SLICES: usize = 8;

/// HOW MANY PT ROWS A CORELET HAS.
///
/// ⛔ NOT THE SAME FACT AS "how many rows can be ADDRESSED". `dip.cpp:2043` loops
/// `p < dscGlobal->sysDef.numPTRows` — a system-definition field — while the init packet sizes its per-row
/// arrays at eight because a flit has eight slices. The two numbers agree, and they agree for different
/// reasons; the compiler asserts its addressing bound against this rather than merging the two.
pub const PT_ROWS: usize = 8;
/// Which registers an instruction REFERS TO — `progir/regvisitor.cpp`'s visitor, as a query over the fields.
pub mod reg_refs;
/// What each unit's register files hold — `regInfoPerUnit` in `sysdef.cpp`.
pub mod regfile;
/// What each unit family encodes for an opcode — `isaSystemc.h`'s six opcode enums.
pub mod values;

/// ONE MACHINE OPCODE — `Isa::InstOpCode` (`isa.hpp:145-245`), all 91 of them.
///
/// Island 2's opcode set.
///
/// ⛔ THE `LD_FIRST = LD` STYLE ALIASES ARE NOT VARIANTS. Six of them (`LD_FIRST`/`LD_LAST`, `LDST_FIRST`/`LDST_LAST`,
/// `ST_FIRST`/`ST_LAST`) are a second spelling of an opcode already here, which the C++ uses to bracket ranges — see
/// [`OpClass`] for why those brackets are not the load/store predicates. The trailing `NumOpCodes` is an array bound.
/// ⭐ THE SPELLINGS ARE THE C++'s, EXACTLY — `COPY_LFSR`, `LDZimm16`, `LD_FIRST`'s neighbours. Renaming them to
/// Rust's convention would make every citation a translation step, and `build.rs` emits references by the same
/// spellings the templates state.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InstOpCode {
    ADDEARIMM,
    ADDLARIMM,
    COPY_LFSR,
    EARIMM,
    EARREGCOPY,
    EE,
    FCMP,
    FCVT,
    FEST,
    FMA,
    FMA8,
    FMA4,
    FMINMAX,
    FMUL,
    FNMS,
    GCVT,
    GTRIMM,
    ICVT,
    IMA4,
    IMA8,
    XMA4,
    IME,
    IMMCOPY,
    INCRMASK,
    JADD,
    JCMP,
    JCMPI,
    JCRSWAP,
    JIMMCOPY,
    JSUB,
    LARIMM,
    LARREGCOPY,
    LD,
    LDCVTI,
    LDCVTIU,
    LDG,
    LDGM,
    LDGMU,
    LDGU,
    LDM,
    LDMU,
    LDU,
    LDIM,
    LDIMU,
    LDIGM,
    LDIGMU,
    LDZ,
    LDZimm16,
    LDZU,
    LDST,
    LDSTI,
    LDSTIU,
    LDSTU,
    LOAD_LFSR,
    LOGICAL,
    LRFCOPY,
    LRFREGCOPY,
    MERGE,
    MODEARREG,
    MODLARREG,
    MODLRFIMM,
    MODLRFREG,
    MVLOOPCNT,
    NOP,
    PACK,
    PERMUTE,
    REDUCE,
    RETURN,
    SAMV,
    SELECT,
    SETDEST,
    SETDSTMASK,
    SETMASK,
    SHR,
    SJCMP,
    SPLAT,
    SPMV,
    ST,
    STG,
    STGU,
    STM,
    STMU,
    STU,
    STIM,
    STIMU,
    STZ,
    SUBLARIMM,
    SUBLRFIMM,
    SYNC,
    TILEADV,
    XRFACCESS,
}

/// WHETHER AN OPCODE MOVES DATA IN, OUT, OR NEITHER.
///
/// ⛔⛔ THE C++ SETS ARE NOT THE C++ RANGES. `isa.hpp` brackets `LD_FIRST = LD` … `LD_LAST = LDZU`, which spans
/// `LDCVTI` and `LDCVTIU` — but `is_load_inst` (`isa.cpp:1513-1521`) enumerates a set that OMITS both and INCLUDES
/// the four `LDST*` opcodes, which sit outside that bracket. The predicate is what the compiler asks; a range check
/// over `LD_FIRST..=LD_LAST` calls the two converting loads loads.
///
/// ⭐ ONE TOTAL MATCH, NOT TWO PREDICATES. Asking `is_load` and `is_store` separately admits an opcode that answers
/// yes to both, which the machine has no such thing as. Classifying once makes that unrepresentable, and the match is
/// total, so a new opcode must be classified before the crate builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// `is_load_inst` (`isa.cpp:1513-1521`) — 19 opcodes, the `LDST*` family included.
    Load,
    /// `is_store_inst` (`isa.cpp:1523-1528`) — 9 opcodes.
    Store,
    /// Neither: a compute, a jump, a sync, a register move.
    Neither,
}

/// `is_load_inst`'s SET, in the order `isa.cpp:1513-1521` writes it.
///
/// ⭐⭐ NOT A SECOND CLASSIFICATION — A CHECK ON THE FIRST. [`InstOpCode::class`] is a total match written out
/// opcode by opcode, and this is the C++'s `std::set` literal written out member by member; the assertion below
/// says they agree. Two independent transcriptions of one set, tied together, is what catches a misread — and a
/// misread here is invisible, because a `Load` that should be `Neither` classifies fine and simply lies.
pub const LOAD_INSTRUCTIONS: [InstOpCode; 19] = [
    InstOpCode::LD,
    InstOpCode::LDG,
    InstOpCode::LDGM,
    InstOpCode::LDGMU,
    InstOpCode::LDGU,
    InstOpCode::LDM,
    InstOpCode::LDMU,
    InstOpCode::LDU,
    InstOpCode::LDIM,
    InstOpCode::LDIMU,
    InstOpCode::LDIGM,
    InstOpCode::LDIGMU,
    InstOpCode::LDZ,
    InstOpCode::LDZimm16,
    InstOpCode::LDZU,
    InstOpCode::LDST,
    InstOpCode::LDSTI,
    InstOpCode::LDSTIU,
    InstOpCode::LDSTU,
];

/// `is_store_inst`'s set, in the order the source writes it.
pub const STORE_INSTRUCTIONS: [InstOpCode; 9] = [
    InstOpCode::ST,
    InstOpCode::STG,
    InstOpCode::STGU,
    InstOpCode::STM,
    InstOpCode::STMU,
    InstOpCode::STU,
    InstOpCode::STIM,
    InstOpCode::STIMU,
    InstOpCode::STZ,
];

/// ⛔ THE SET AND THE CLASSIFICATION AGREE, IN BOTH DIRECTIONS.
///
/// Membership is checked one way and the census the other, so neither a member classified `Neither` nor an
/// opcode classified `Load` that the set omits can survive a build.
const _: () = {
    let mut at = 0;
    while at < LOAD_INSTRUCTIONS.len() {
        assert!(
            LOAD_INSTRUCTIONS[at].is_load(),
            "an opcode in `is_load_inst`'s set does not classify as a load"
        );
        at += 1;
    }
    at = 0;
    while at < STORE_INSTRUCTIONS.len() {
        assert!(
            STORE_INSTRUCTIONS[at].is_store(),
            "an opcode in `is_store_inst`'s set does not classify as a store"
        );
        at += 1;
    }
    let (mut loads, mut stores, mut i) = (0, 0, 0);
    while i < InstOpCode::ALL.len() {
        match InstOpCode::ALL[i].class() {
            OpClass::Load => loads += 1,
            OpClass::Store => stores += 1,
            OpClass::Neither => {}
        }
        i += 1;
    }
    assert!(
        loads == LOAD_INSTRUCTIONS.len(),
        "an opcode classifies as a load that `is_load_inst`'s set does not contain"
    );
    assert!(
        stores == STORE_INSTRUCTIONS.len(),
        "an opcode classifies as a store that `is_store_inst`'s set does not contain"
    );
};

impl InstOpCode {
    /// HOW MANY REAL OPCODES THERE ARE — `NumOpCodes`, which the aliases do not inflate.
    pub const COUNT: u8 = Self::ALL.len() as u8;

    /// The load family's brackets, `LD_FIRST ..= LD_LAST` (`isa.hpp`).
    pub const LOAD_FAMILY: (Self, Self) = (Self::LD, Self::LDZU);
    /// The load-store family's brackets, `LDST_FIRST ..= LDST_LAST`.
    pub const LOAD_STORE_FAMILY: (Self, Self) = (Self::LDST, Self::LDSTU);
    /// The store family's brackets, `ST_FIRST ..= ST_LAST`.
    pub const STORE_FAMILY: (Self, Self) = (Self::ST, Self::STZ);

    /// WHETHER THIS OPCODE FALLS IN A FAMILY'S RANGE.
    ///
    /// ⛔⛔ THIS IS THE RANGE TEST, NOT THE CLASSIFICATION, AND THEY DISAGREE. `isa.hpp`'s markers bracket
    /// `LD_FIRST = LD` … `LD_LAST = LDZU`, a span that includes `LDCVTI`/`LDCVTIU` and excludes the `LDST*`
    /// family — while `is_load_inst` enumerates a SET that does the opposite on both counts. [`Self::class`] is
    /// what the compiler asks; this is only what the header brackets, and the two are kept apart deliberately.
    #[must_use]
    pub fn in_family(self, family: (Self, Self)) -> bool {
        self >= family.0 && self <= family.1
    }

    /// `Isa::is_load_inst` — the model's own load classification.
    #[must_use]
    pub const fn is_load(self) -> bool {
        matches!(self.class(), OpClass::Load)
    }

    /// `Isa::is_store_inst`.
    #[must_use]
    pub const fn is_store(self) -> bool {
        matches!(self.class(), OpClass::Store)
    }

    /// THE MNEMONIC — `Isa::to_string` (`isa.cpp:1360-1456`, `instOpCodeToStr`).
    ///
    /// ⭐⭐ A RENAME, AND THAT IS MEASURED RATHER THAN ASSUMED: all **91** entries of `instOpCodeToStr` map an
    /// enumerator to a string identical to its own name — zero mismatches. So the variant IS the mnemonic.
    ///
    /// ⛔ AND IT IS NOT `{:?}`. This text goes into a senprog line that `dip_standalone` parses, so the spelling
    /// is an interface and not a debug convenience: a `Debug` impl is free to change and this is not.
    ///
    /// ⛔ `JCMPI` IS SPELLED HERE AS ITSELF. The WRITER prints it as `JCMP` because the field names depend on the
    /// `isimm` bit and it is not a separate instruction (`dpc.cpp:665-668`) — that substitution belongs to the
    /// writer, not to the opcode's name.
    /// EVERY OPCODE, in `Isa::InstOpCode`'s declaration order (`isa.hpp:147+`).
    ///
    /// ⭐ THE ORDER IS THE ENUM'S, so a `const` assertion can check the count without a second list to drift from.
    pub const ALL: [Self; 91] = [
        Self::ADDEARIMM,
        Self::ADDLARIMM,
        Self::COPY_LFSR,
        Self::EARIMM,
        Self::EARREGCOPY,
        Self::EE,
        Self::FCMP,
        Self::FCVT,
        Self::FEST,
        Self::FMA,
        Self::FMA8,
        Self::FMA4,
        Self::FMINMAX,
        Self::FMUL,
        Self::FNMS,
        Self::GCVT,
        Self::GTRIMM,
        Self::ICVT,
        Self::IMA4,
        Self::IMA8,
        Self::XMA4,
        Self::IME,
        Self::IMMCOPY,
        Self::INCRMASK,
        Self::JADD,
        Self::JCMP,
        Self::JCMPI,
        Self::JCRSWAP,
        Self::JIMMCOPY,
        Self::JSUB,
        Self::LARIMM,
        Self::LARREGCOPY,
        Self::LD,
        Self::LDCVTI,
        Self::LDCVTIU,
        Self::LDG,
        Self::LDGM,
        Self::LDGMU,
        Self::LDGU,
        Self::LDM,
        Self::LDMU,
        Self::LDU,
        Self::LDIM,
        Self::LDIMU,
        Self::LDIGM,
        Self::LDIGMU,
        Self::LDZ,
        Self::LDZimm16,
        Self::LDZU,
        Self::LDST,
        Self::LDSTI,
        Self::LDSTIU,
        Self::LDSTU,
        Self::LOAD_LFSR,
        Self::LOGICAL,
        Self::LRFCOPY,
        Self::LRFREGCOPY,
        Self::MERGE,
        Self::MODEARREG,
        Self::MODLARREG,
        Self::MODLRFIMM,
        Self::MODLRFREG,
        Self::MVLOOPCNT,
        Self::NOP,
        Self::PACK,
        Self::PERMUTE,
        Self::REDUCE,
        Self::RETURN,
        Self::SAMV,
        Self::SELECT,
        Self::SETDEST,
        Self::SETDSTMASK,
        Self::SETMASK,
        Self::SHR,
        Self::SJCMP,
        Self::SPLAT,
        Self::SPMV,
        Self::ST,
        Self::STG,
        Self::STGU,
        Self::STM,
        Self::STMU,
        Self::STU,
        Self::STIM,
        Self::STIMU,
        Self::STZ,
        Self::SUBLARIMM,
        Self::SUBLRFIMM,
        Self::SYNC,
        Self::TILEADV,
        Self::XRFACCESS,
    ];

    /// THE OPCODE A SPELLING NAMES — `Isa::to_instopcode` (`isa.cpp:155`), read backwards.
    ///
    /// ⛔ `None` IS "NO SUCH OPCODE", NOT A DEFAULT. The C++ has a `DT_CHECK` per opcode that the round trip
    /// holds; a spelling outside the set is a name this ISA does not have, and the caller says what to do about
    /// it rather than receiving a plausible neighbour.
    #[must_use]
    pub fn of_spelling(spelling: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.spelling() == spelling)
    }

    pub const fn spelling(self) -> &'static str {
        match self {
            Self::ADDEARIMM => "ADDEARIMM",
            Self::ADDLARIMM => "ADDLARIMM",
            Self::COPY_LFSR => "COPY_LFSR",
            Self::EARIMM => "EARIMM",
            Self::EARREGCOPY => "EARREGCOPY",
            Self::EE => "EE",
            Self::FCMP => "FCMP",
            Self::FCVT => "FCVT",
            Self::FEST => "FEST",
            Self::FMA => "FMA",
            Self::FMA8 => "FMA8",
            Self::FMA4 => "FMA4",
            Self::FMINMAX => "FMINMAX",
            Self::FMUL => "FMUL",
            Self::FNMS => "FNMS",
            Self::GCVT => "GCVT",
            Self::GTRIMM => "GTRIMM",
            Self::ICVT => "ICVT",
            Self::IMA4 => "IMA4",
            Self::IMA8 => "IMA8",
            Self::XMA4 => "XMA4",
            Self::IME => "IME",
            Self::IMMCOPY => "IMMCOPY",
            Self::INCRMASK => "INCRMASK",
            Self::JADD => "JADD",
            Self::JCMP => "JCMP",
            Self::JCMPI => "JCMPI",
            Self::JCRSWAP => "JCRSWAP",
            Self::JIMMCOPY => "JIMMCOPY",
            Self::JSUB => "JSUB",
            Self::LARIMM => "LARIMM",
            Self::LARREGCOPY => "LARREGCOPY",
            Self::LD => "LD",
            Self::LDCVTI => "LDCVTI",
            Self::LDCVTIU => "LDCVTIU",
            Self::LDG => "LDG",
            Self::LDGM => "LDGM",
            Self::LDGMU => "LDGMU",
            Self::LDGU => "LDGU",
            Self::LDM => "LDM",
            Self::LDMU => "LDMU",
            Self::LDU => "LDU",
            Self::LDIM => "LDIM",
            Self::LDIMU => "LDIMU",
            Self::LDIGM => "LDIGM",
            Self::LDIGMU => "LDIGMU",
            Self::LDZ => "LDZ",
            Self::LDZimm16 => "LDZimm16",
            Self::LDZU => "LDZU",
            Self::LDST => "LDST",
            Self::LDSTI => "LDSTI",
            Self::LDSTIU => "LDSTIU",
            Self::LDSTU => "LDSTU",
            Self::LOAD_LFSR => "LOAD_LFSR",
            Self::LOGICAL => "LOGICAL",
            Self::LRFCOPY => "LRFCOPY",
            Self::LRFREGCOPY => "LRFREGCOPY",
            Self::MERGE => "MERGE",
            Self::MODEARREG => "MODEARREG",
            Self::MODLARREG => "MODLARREG",
            Self::MODLRFIMM => "MODLRFIMM",
            Self::MODLRFREG => "MODLRFREG",
            Self::MVLOOPCNT => "MVLOOPCNT",
            Self::NOP => "NOP",
            Self::PACK => "PACK",
            Self::PERMUTE => "PERMUTE",
            Self::REDUCE => "REDUCE",
            Self::RETURN => "RETURN",
            Self::SAMV => "SAMV",
            Self::SELECT => "SELECT",
            Self::SETDEST => "SETDEST",
            Self::SETDSTMASK => "SETDSTMASK",
            Self::SETMASK => "SETMASK",
            Self::SHR => "SHR",
            Self::SJCMP => "SJCMP",
            Self::SPLAT => "SPLAT",
            Self::SPMV => "SPMV",
            Self::ST => "ST",
            Self::STG => "STG",
            Self::STGU => "STGU",
            Self::STM => "STM",
            Self::STMU => "STMU",
            Self::STU => "STU",
            Self::STIM => "STIM",
            Self::STIMU => "STIMU",
            Self::STZ => "STZ",
            Self::SUBLARIMM => "SUBLARIMM",
            Self::SUBLRFIMM => "SUBLRFIMM",
            Self::SYNC => "SYNC",
            Self::TILEADV => "TILEADV",
            Self::XRFACCESS => "XRFACCESS",
        }
    }
}

impl InstOpCode {
    /// Which class this opcode belongs to.
    pub const fn class(self) -> OpClass {
        match self {
            Self::ADDEARIMM => OpClass::Neither,
            Self::ADDLARIMM => OpClass::Neither,
            Self::COPY_LFSR => OpClass::Neither,
            Self::EARIMM => OpClass::Neither,
            Self::EARREGCOPY => OpClass::Neither,
            Self::EE => OpClass::Neither,
            Self::FCMP => OpClass::Neither,
            Self::FCVT => OpClass::Neither,
            Self::FEST => OpClass::Neither,
            Self::FMA => OpClass::Neither,
            Self::FMA8 => OpClass::Neither,
            Self::FMA4 => OpClass::Neither,
            Self::FMINMAX => OpClass::Neither,
            Self::FMUL => OpClass::Neither,
            Self::FNMS => OpClass::Neither,
            Self::GCVT => OpClass::Neither,
            Self::GTRIMM => OpClass::Neither,
            Self::ICVT => OpClass::Neither,
            Self::IMA4 => OpClass::Neither,
            Self::IMA8 => OpClass::Neither,
            Self::XMA4 => OpClass::Neither,
            Self::IME => OpClass::Neither,
            Self::IMMCOPY => OpClass::Neither,
            Self::INCRMASK => OpClass::Neither,
            Self::JADD => OpClass::Neither,
            Self::JCMP => OpClass::Neither,
            Self::JCMPI => OpClass::Neither,
            Self::JCRSWAP => OpClass::Neither,
            Self::JIMMCOPY => OpClass::Neither,
            Self::JSUB => OpClass::Neither,
            Self::LARIMM => OpClass::Neither,
            Self::LARREGCOPY => OpClass::Neither,
            Self::LD => OpClass::Load,
            Self::LDCVTI => OpClass::Neither,
            Self::LDCVTIU => OpClass::Neither,
            Self::LDG => OpClass::Load,
            Self::LDGM => OpClass::Load,
            Self::LDGMU => OpClass::Load,
            Self::LDGU => OpClass::Load,
            Self::LDM => OpClass::Load,
            Self::LDMU => OpClass::Load,
            Self::LDU => OpClass::Load,
            Self::LDIM => OpClass::Load,
            Self::LDIMU => OpClass::Load,
            Self::LDIGM => OpClass::Load,
            Self::LDIGMU => OpClass::Load,
            Self::LDZ => OpClass::Load,
            Self::LDZimm16 => OpClass::Load,
            Self::LDZU => OpClass::Load,
            Self::LDST => OpClass::Load,
            Self::LDSTI => OpClass::Load,
            Self::LDSTIU => OpClass::Load,
            Self::LDSTU => OpClass::Load,
            Self::LOAD_LFSR => OpClass::Neither,
            Self::LOGICAL => OpClass::Neither,
            Self::LRFCOPY => OpClass::Neither,
            Self::LRFREGCOPY => OpClass::Neither,
            Self::MERGE => OpClass::Neither,
            Self::MODEARREG => OpClass::Neither,
            Self::MODLARREG => OpClass::Neither,
            Self::MODLRFIMM => OpClass::Neither,
            Self::MODLRFREG => OpClass::Neither,
            Self::MVLOOPCNT => OpClass::Neither,
            Self::NOP => OpClass::Neither,
            Self::PACK => OpClass::Neither,
            Self::PERMUTE => OpClass::Neither,
            Self::REDUCE => OpClass::Neither,
            Self::RETURN => OpClass::Neither,
            Self::SAMV => OpClass::Neither,
            Self::SELECT => OpClass::Neither,
            Self::SETDEST => OpClass::Neither,
            Self::SETDSTMASK => OpClass::Neither,
            Self::SETMASK => OpClass::Neither,
            Self::SHR => OpClass::Neither,
            Self::SJCMP => OpClass::Neither,
            Self::SPLAT => OpClass::Neither,
            Self::SPMV => OpClass::Neither,
            Self::ST => OpClass::Store,
            Self::STG => OpClass::Store,
            Self::STGU => OpClass::Store,
            Self::STM => OpClass::Store,
            Self::STMU => OpClass::Store,
            Self::STU => OpClass::Store,
            Self::STIM => OpClass::Store,
            Self::STIMU => OpClass::Store,
            Self::STZ => OpClass::Store,
            Self::SUBLARIMM => OpClass::Neither,
            Self::SUBLRFIMM => OpClass::Neither,
            Self::SYNC => OpClass::Neither,
            Self::TILEADV => OpClass::Neither,
            Self::XRFACCESS => OpClass::Neither,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE CLASSIFICATION IS THE C++'S SETS, NOT ITS RANGES
// ─────────────────────────────────────────────────────────────────────────────

/// The two opcodes inside `LD_FIRST..=LD_LAST` that `is_load_inst` leaves out.
const _: () = assert!(matches!(InstOpCode::LDCVTI.class(), OpClass::Neither));
const _: () = assert!(matches!(InstOpCode::LDCVTIU.class(), OpClass::Neither));
/// Their neighbours in the bracket ARE loads, so the two above are an omission and not a gap in the port.
const _: () = assert!(matches!(InstOpCode::LD.class(), OpClass::Load));
const _: () = assert!(matches!(InstOpCode::LDG.class(), OpClass::Load));
const _: () = assert!(matches!(InstOpCode::LDZU.class(), OpClass::Load));
/// The `LDST*` family is in the load set while sitting outside the LD bracket.
const _: () = assert!(matches!(InstOpCode::LDST.class(), OpClass::Load));
const _: () = assert!(matches!(InstOpCode::LDSTI.class(), OpClass::Load));
const _: () = assert!(matches!(InstOpCode::LDSTIU.class(), OpClass::Load));
const _: () = assert!(matches!(InstOpCode::LDSTU.class(), OpClass::Load));
/// The store set, at both ends.
const _: () = assert!(matches!(InstOpCode::ST.class(), OpClass::Store));
const _: () = assert!(matches!(InstOpCode::STZ.class(), OpClass::Store));
/// `STIM`/`STIMU` are stores; the similarly-spelled `LDIM`/`LDIMU` are loads.
const _: () = assert!(matches!(InstOpCode::STIM.class(), OpClass::Store));
const _: () = assert!(matches!(InstOpCode::LDIM.class(), OpClass::Load));
/// A compute, a sync and a jump are neither.
const _: () = assert!(matches!(InstOpCode::FMA.class(), OpClass::Neither));
const _: () = assert!(matches!(InstOpCode::SYNC.class(), OpClass::Neither));
const _: () = assert!(matches!(InstOpCode::JCMP.class(), OpClass::Neither));

/// THE FIELD TABLE'S COMPONENT, from `regfile`'s.
///
/// ⛔⛔ TWO ENUMS FOR ONE THING, AND THE REASON IS THE SHARED FILE. `src/isa/fields.rs` is read by `build.rs` too,
/// where `crate::regfile` is not in scope — so the table carries its own `Comp` and this conversion lives
/// here, on the `src`-only side. One-to-one, so it cannot lose a unit.
pub const fn table_component(comp: regfile::Component) -> fields::Comp {
    match comp {
        regfile::Component::Pt => fields::Comp::Pt,
        regfile::Component::Pe => fields::Comp::Pe,
        regfile::Component::Sfp => fields::Comp::Sfp,
        regfile::Component::L0lu => fields::Comp::L0lu,
        regfile::Component::L0su => fields::Comp::L0su,
        regfile::Component::Lxlu => fields::Comp::Lxlu,
        regfile::Component::Lxsu => fields::Comp::Lxsu,
        regfile::Component::L3lu => fields::Comp::L3lu,
        regfile::Component::L3su => fields::Comp::L3su,
    }
}

/// WHICH BIT A FIELD OF THIS INSTRUCTION STARTS AT — `checkIfFieldExists` then
/// `typeToFieldBitShift.at(instrType).at(fieldPos)` (`dpc.cpp:676-713`), as one lookup.
///
/// ⛔⛔ THE TYPE IS SCOPED TO THE COMPONENT, so both are arguments. Type 20 on the PT and type 20 on the PE are
/// different field sets — `typeToFieldName` is a member of a per-component `Isa` — so a table keyed by type alone
/// would merge them.
///
/// ⛔ `None` MEANS THIS INSTRUCTION HAS NO POSITION FOR THAT FIELD, which the C++ treats as an error
/// (`DT_ERROR_FMT("Illegal instruction/operand combination for this architecture")`). Returned rather than
/// panicked so the caller can name what it was writing.
///
/// ⭐ THE TEST IS `answers_to`, NOT `==`: `MVLOOPCNT`'s bit 10 is named `"imm/pc_target"`, so an exact comparison
/// for `imm` finds nothing there — the bug that once made every loop's trip count encode as zero.
/// EVERY FIELD ONE INSTRUCTION TYPE DECLARES, on this component — the key set of
/// `typeToFieldBitShift[type]` (`isa.cpp:167`).
///
/// ⛔ THIS IS THE SET AN INSTRUCTION ZEROES AT CONSTRUCTION. `Instruction::Instruction` clears each of them
/// before any value is read, so a field of the type reads 0 when unstated while a field the type does NOT have
/// keeps the absent sentinel. The two are different answers and only this list separates them.
///
/// ⭐ THE `/` PAIR CONTRIBUTES ITS FIRST NAME, which is the one a reader binds — `defineField` splits
/// `"imm/pc_target"` and `getFieldNameForOffset` returns the pair with `imm` first.
pub fn fields_of(
    comp: regfile::Component,
    opcode: InstOpCode,
) -> impl Iterator<Item = operand::Operand> {
    let table = table_component(comp);
    // ⛔ AN OPCODE THIS COMPONENT DOES NOT DEFINE IS A REFUSAL, NOT AN EMPTY SET. `defineOpcode` is per
    // component, so "no type for this opcode here" means the caller put the instruction on the wrong unit — and
    // an empty iterator lets a caller conclude "this instruction declares no such field", which is a DIFFERENT
    // fact. That silence is what let a `Register` reach a PT `src0` past the guard that exists to stop it.
    let ty = table
        .opcodes()
        .iter()
        .find(|def| def.op == opcode.spelling())
        .map(|def| def.ty)
        .unwrap_or_else(|| {
            panic!(
                "the {comp:?} has no {opcode:?} instruction: `defineOpcode` does not define it on this \
                 component, so it declares no fields either"
            )
        });
    table
        .fields()
        .iter()
        .filter(move |def| def.ty == ty)
        .map(|def| match def.name {
            fields::FieldName::One(name) => name,
            fields::FieldName::Either(first, _) => first,
        })
}

pub fn bit_of(
    comp: regfile::Component,
    opcode: InstOpCode,
    field: operand::Operand,
) -> Option<fields::WordBit> {
    let comp = table_component(comp);
    let ty = comp
        .opcodes()
        .iter()
        .find(|def| def.op == opcode.spelling())?
        .ty;
    comp.fields()
        .iter()
        .find(|def| def.ty == ty && def.name.answers_to(field))
        .map(|def| def.bit)
}
