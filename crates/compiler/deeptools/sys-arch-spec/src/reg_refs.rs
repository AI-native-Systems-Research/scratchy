//! WHICH REGISTER FILE EACH OF AN INSTRUCTION'S SLOTS NAMES — the port of `RegVisitor`
//! (`sys-arch-spec/progir/regvisitor.cpp`).
//!
//! ⭐ THE TABLE IS PER (UNIT CLASS, OPCODE), which is the only key the C++ has: `visitInstrRegRefs` dispatches on
//! the unit and then switches on `instr.instn_` (`regvisitor.cpp:14-28`). The same slot names different files on
//! different units — `src0` is a LAR on an L3 unit and an LRF on an LX one — so neither half of the key is
//! optional.
//!
//! ⛔ THE C++ ASKS WHETHER THE SLOT IS PRESENT AND SO MUST THE CALLER. Its transfer arm guards every visit with
//! `operands.find(OperandT::srcN) != operands.end()` (`regvisitor.cpp:111-122`), because "not all LD/ST variants
//! have the same operands, however, when they are present they always have the same register reference semantics".
//! This table therefore states EVERY slot a transfer could name, and the caller intersects it with the fields the
//! instruction actually holds — `deeptools`'s `generated::PhysicalOp::fields`.
//!
//! ⛔ `regRefToRegNum` IS NOT PORTED (`regvisitor.cpp:275-295`). It parses `R<n>` and `GTR<n>` out of an assembly
//! string, because a C++ `OperandAttr` may hold either an int or descriptive text. Our operands are already
//! `deeptools`'s `islands::i3::RegIndex` — the number, minted by the pass that hands registers out — so there is no
//! text to parse and no parser belongs in `src/`.

use crate::InstOpCode;
use crate::operand::Operand;
use crate::regfile::{Component, RegType};

/// EVERY SLOT AN L3 TRANSFER COULD NAME, in the order `visitL3InstrRegRefs` visits them
/// (`regvisitor.cpp:111-122`).
///
/// ⭐ `src2` IS AN EAR AND THE C++ NOTES IT IS "not set in sentient 1.5" (`regvisitor.cpp:117`) — so on that arch
/// the slot is simply absent from the instruction and the caller's intersection drops it, with no arch test here.
const L3_TRANSFER_REFS: &[(Operand, RegType)] = &[
    (Operand::Src0, RegType::Lar),
    (Operand::Src1, RegType::Lbr),
    (Operand::Src2, RegType::Ear),
    (Operand::Src3, RegType::Ebr),
    (Operand::Group, RegType::Gtr),
];

/// WHICH OF THE FOUR ARMS `visitInstrRegRefs` DISPATCHES TO (`regvisitor.cpp:14-28`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitClass {
    /// `SenComponents::L3`, `L3LU`, `L3SU`.
    L3,
    /// `SenComponents::LX`, `LXLU`, `LXSU`.
    Lx,
    Pe,
    Sfp,
    /// `SenComponents::L0LU`, `L0SU`.
    ///
    /// ⛔⛔⛔ THIS ARM HAS NO `RegVisitor` COUNTERPART AND IS NOT PORTED FROM ONE. `visitInstrRegRefs` refuses an
    /// L0 outright (`regvisitor.cpp:14-28`), so there is nothing there to transcribe — but dcgbe still owes an L0
    /// LRF register the second arm's explicit zero (`dcgbeCodegen.cpp:2654-2659`), because the L0's LRF is
    /// `Initializable::Yes`. The C++ that DOES state which of an L0 instruction's slots name an LRF is the unit's
    /// own execution, `L0::executePC` (`senulator/memoryElement.cpp:2193-2969`), and this arm is the port of the
    /// `reg_file_LRF[…]` subscripts in it. ⭐ The L0 class starts at `:2069`, so those subscripts are the L0's own
    /// and not the LX's — the two units declare separate `reg_file_LRF` members (`:1253`, `:2072`).
    L0,
}

/// WHAT READS ONE COMPONENT'S INSTRUCTIONS — an arm of `visitInstrRegRefs`, or the stated reason there is none.
///
/// ⛔⛔⛔ "NO ARM" IS TWO DIFFERENT FACTS AND THEY MUST NOT SHARE A PATTERN. The PT has no arm and needs none; the
/// L0 has no arm and DOES need one. Spelling both as a bare empty set is what let the L0's missing second-arm
/// zeros read as "this unit references nothing" for the whole bake — a missing fact silently becoming a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reads {
    /// `visitInstrRegRefs` dispatches this component to `class`.
    Arm(UnitClass),
    /// NO ARM AND NONE NEEDED. Every file this component has that a reference could name is `Initializable::No`,
    /// so a walk of its instructions could not ask for an initial value even if one existed. ⛔ NOT PROSE —
    /// [`PT_NEEDS_NO_ARM`] proves both halves, so a component whose files change stops compiling.
    NothingReferenceableIsInitialisable,
}

impl UnitClass {
    /// EVERY ARM `visitInstrRegRefs` HAS (`regvisitor.cpp:14-28`), so a fifth cannot appear without the pins below
    /// walking it.
    pub const ALL: [Self; 5] = [Self::L3, Self::Lx, Self::Pe, Self::Sfp, Self::L0];

    /// The arm this component's instructions are read by, or the stated reason it has none.
    ///
    /// ⛔⛔ THERE IS NO PT ARM AND NO L0 ARM, and that is the C++'s own dispatcher: `visitInstrRegRefs` tests L3,
    /// LX, SFP and PE and answers anything else with `DT_ERROR("Progstitching requested on unsupported unit")`
    /// (`regvisitor.cpp:14-28`).
    ///
    /// ⛔ THIS IS THE ONLY ANSWER TO THAT QUESTION. `bridges/3.rs::referenced_registers` used to carry a second
    /// component-to-class `match` of its own whose PT/L0 arm was a bare `return` of the empty set, so this
    /// refusal was never reached and the L0 gap was invisible.
    pub const fn of_component(comp: Component) -> Reads {
        match comp {
            Component::L3lu | Component::L3su => Reads::Arm(Self::L3),
            Component::Lxlu | Component::Lxsu => Reads::Arm(Self::Lx),
            Component::Pe => Reads::Arm(Self::Pe),
            Component::Sfp => Reads::Arm(Self::Sfp),
            Component::Pt => Reads::NothingReferenceableIsInitialisable,
            Component::L0lu | Component::L0su => Reads::Arm(Self::L0),
        }
    }
}

/// ⛔⛔⛔ THE PT'S EMPTY ANSWER IS PROVED HERE, NOT ASSERTED IN A COMMENT — and the comment it replaces was WRONG.
/// It read "its XRF, ARF and IRF are all `Initializable::No`", which enumerates three of the PT's five files and
/// omits the one that is `Initializable::Yes`: the `SPR` (`regfile.rs`, `(Pt, Spr) => … Initializable::Yes`).
///
/// The true reason has two halves:
///  1. every file the PT has EXCEPT `Spr` is absent or `Initializable::No` — CHECKED BELOW; and
///  2. no arm of this table names an `Spr` on any unit, so a reference walk cannot yield the PT's one
///     initialisable file. (The `SPR` is written by the SPR path instead, never by this one.)
///
/// ⛔ HALF 2 IS NOT CHECKED HERE AND THE REASON IS THE TABLE'S OWN SHAPE, not an omission: [`reg_refs`] REFUSES a
/// (class, opcode) pair that does not belong together — a faithful port of `Unhandled L3 instruction while
/// analyzing register references` (`regvisitor.cpp:123-125`) — so a walk over `UnitClass::ALL × InstOpCode::ALL`
/// panics on the invalid pairs rather than reporting the files. Bounding it needs each arm's valid opcode set as
/// a list, which would be a second copy of the four `match`es and free to drift from them. ⭐ WHAT PROTECTS IT
/// INSTEAD: an `Spr` row added to any arm makes that arm's slice name a file the PT has and the PT's early return
/// hides, so half 1's carve-out is the thing to revisit when one appears.
///
/// ⛔ AND THE L0 IS PINNED APART FROM IT: the L0's `Lrf` is `Initializable::Yes` AND `Lrf` IS named by an arm, so
/// the L0 fails half 1 and half 2 both. That is why it has [`UnitClass::L0`] and the PT does not, and why moving
/// `L0lu`/`L0su` into the PT's `NothingReferenceableIsInitialisable` arm has to stop compiling — the assertion
/// below is what makes it stop. The two units' emptiness is NOT the same fact and must not share a pattern.
const PT_NEEDS_NO_ARM: () = {
    let mut i = 0;
    while i < RegType::ALL.len() {
        let file = RegType::ALL[i];
        if !matches!(file, RegType::Spr)
            && let crate::regfile::Presence::Present(info) =
                crate::regfile::info_of(Component::Pt, file)
        {
            assert!(
                matches!(info.initializable, crate::regfile::Initializable::No),
                "a PT file other than the SPR became initialisable, so `Reads::NothingReferenceableIsInitialisable` \
                 no longer describes the PT — it needs an arm, or a narrower reason"
            );
        }
        i += 1;
    }

    // ── THE L0 IS NOT THE PT. Both halves fail for it, and this states that so the two cannot be merged.
    let crate::regfile::Presence::Present(l0lu) =
        crate::regfile::info_of(Component::L0lu, RegType::Lrf)
    else {
        panic!(
            "the L0LU lost its LRF, so the gap this table refuses no longer has the shape it refuses"
        )
    };
    let crate::regfile::Presence::Present(l0su) =
        crate::regfile::info_of(Component::L0su, RegType::Lrf)
    else {
        panic!(
            "the L0SU lost its LRF, so the gap this table refuses no longer has the shape it refuses"
        )
    };
    assert!(matches!(
        l0lu.initializable,
        crate::regfile::Initializable::Yes
    ));
    assert!(matches!(
        l0su.initializable,
        crate::regfile::Initializable::Yes
    ));
};

const _: () = PT_NEEDS_NO_ARM;

/// THE SLOTS OF `op` THAT NAME A REGISTER ON THIS CLASS OF UNIT, each with the file it names.
///
/// ⛔ A SLOT LISTED HERE IS NOT NECESSARILY PRESENT — see the module comment. Intersect with the instruction's own
/// fields.
pub const fn reg_refs(class: UnitClass, op: InstOpCode) -> &'static [(Operand, RegType)] {
    match class {
        UnitClass::L3 => match op {
            InstOpCode::ADDEARIMM | InstOpCode::EARIMM => &[(Operand::Src0, RegType::Ear)],

            InstOpCode::ADDLARIMM | InstOpCode::LARIMM | InstOpCode::SUBLARIMM => {
                &[(Operand::Src0, RegType::Lar)]
            }

            InstOpCode::EARREGCOPY => {
                &[(Operand::Src0, RegType::Ear), (Operand::Src1, RegType::Ear)]
            }

            InstOpCode::LARREGCOPY => {
                &[(Operand::Src0, RegType::Lar), (Operand::Src1, RegType::Lar)]
            }

            InstOpCode::GTRIMM => &[(Operand::Src0, RegType::Gtr)],

            InstOpCode::MODEARREG => &[(Operand::Src1, RegType::Ear)],

            InstOpCode::MODLARREG => &[(Operand::Src1, RegType::Lar)],
            // `// Visitation of JCRs not currently needed.` — the C++ says so in every arm that
            // lists a jump, so a JCR reference is a deliberate omission and not a missing case.
            InstOpCode::JADD
            | InstOpCode::JCMP
            | InstOpCode::JCMPI
            | InstOpCode::JIMMCOPY
            | InstOpCode::JSUB => &[],

            InstOpCode::LDZimm16
            | InstOpCode::MVLOOPCNT
            | InstOpCode::NOP
            | InstOpCode::RETURN
            | InstOpCode::SJCMP
            | InstOpCode::SYNC => &[],

            InstOpCode::LD
            | InstOpCode::LDG
            | InstOpCode::LDGM
            | InstOpCode::LDGMU
            | InstOpCode::LDGU
            | InstOpCode::LDIGM
            | InstOpCode::LDIGMU
            | InstOpCode::LDIM
            | InstOpCode::LDIMU
            | InstOpCode::LDM
            | InstOpCode::LDMU
            | InstOpCode::LDU
            | InstOpCode::LDZ
            | InstOpCode::LDZU
            | InstOpCode::ST
            | InstOpCode::STG
            | InstOpCode::STGU
            | InstOpCode::STIM
            | InstOpCode::STIMU
            | InstOpCode::STZ
            | InstOpCode::STM
            | InstOpCode::STMU
            | InstOpCode::STU => L3_TRANSFER_REFS,
            InstOpCode::COPY_LFSR
            | InstOpCode::EE
            | InstOpCode::FCMP
            | InstOpCode::FCVT
            | InstOpCode::FEST
            | InstOpCode::FMA
            | InstOpCode::FMA4
            | InstOpCode::FMA8
            | InstOpCode::FMINMAX
            | InstOpCode::FMUL
            | InstOpCode::FNMS
            | InstOpCode::GCVT
            | InstOpCode::ICVT
            | InstOpCode::IMA4
            | InstOpCode::IMA8
            | InstOpCode::IME
            | InstOpCode::IMMCOPY
            | InstOpCode::INCRMASK
            | InstOpCode::JCRSWAP
            | InstOpCode::LDCVTI
            | InstOpCode::LDCVTIU
            | InstOpCode::LDST
            | InstOpCode::LDSTI
            | InstOpCode::LDSTIU
            | InstOpCode::LDSTU
            | InstOpCode::LOAD_LFSR
            | InstOpCode::LOGICAL
            | InstOpCode::LRFCOPY
            | InstOpCode::LRFREGCOPY
            | InstOpCode::MERGE
            | InstOpCode::MODLRFIMM
            | InstOpCode::MODLRFREG
            | InstOpCode::PACK
            | InstOpCode::PERMUTE
            | InstOpCode::REDUCE
            | InstOpCode::SAMV
            | InstOpCode::SELECT
            | InstOpCode::SETDEST
            | InstOpCode::SETDSTMASK
            | InstOpCode::SETMASK
            | InstOpCode::SHR
            | InstOpCode::SPLAT
            | InstOpCode::SPMV
            | InstOpCode::SUBLRFIMM
            | InstOpCode::TILEADV
            | InstOpCode::XMA4
            | InstOpCode::XRFACCESS => panic!(
                "an instruction this unit's `RegVisitor` arm does not list ran on an L3 unit \
                 (`regvisitor.cpp:123-125`: `Unhandled L3 instruction while analyzing register references`)"
            ),
        },
        // ── THE L0, PORTED FROM `L0::executePC` (`memoryElement.cpp:2193-2969`) — see [`UnitClass::L0`].
        UnitClass::L0 => match op {
            // `:2929-2947` — each takes `src0` as its LRF and its immediate from `OperandT::imm`.
            //
            // ⭐ THE IMMEDIATE IS `imm` HERE AND `lrfimm` ON THE LX (`:1348`), which is the one documented
            // difference between the two units' LRF opcodes — and it is a difference in a field that is NOT a
            // register, so it changes nothing about which slots this table names.
            //
            // ⛔ `LRFCOPY` IS ABSENT ON PURPOSE. The LX handles it (`:1978`) and the L0's chain does not, so an
            // `LRFCOPY` reaching an L0 falls to `errorAtLine("ImplementationError", "Instruction not yet
            // added.")` (`:2965-2967`) — it is refused below rather than listed here.
            InstOpCode::IMMCOPY | InstOpCode::MODLRFIMM | InstOpCode::SUBLRFIMM => {
                &[(Operand::Src0, RegType::Lrf)]
            }

            // `:2949-2963` — `reg_file_LRF[src0] = reg_file_LRF[src1]` and `… + reg_file_LRF[src1]`.
            InstOpCode::LRFREGCOPY | InstOpCode::MODLRFREG => {
                &[(Operand::Src0, RegType::Lrf), (Operand::Src1, RegType::Lrf)]
            }

            // `isLXLdSt` (`:25-28`) — the `LDST` range plus the two converting loads. `src0` is the address
            // register, read at `:2340` and written back at `:2424`/`:2589`/`:2823`/`:2917`; `src1` is an LRF
            // only when the variant is not immediate-source (`:2302-2317`, `isLXLdStImmSrc` at `:29-32`), and
            // the caller's intersection with the instruction's own fields drops it when it is not there.
            //
            // ⛔ `Src2` IS DELIBERATELY NOT LISTED, and this is the one place the L0 reads an LRF that the LX
            // never does (`:2352-2355`). It is reached only on the coalesce path — `isImmSrc && !isLoadUnit() &&
            // !isSen1p5() && getOperand(OperandT::coalesce)` (`:2346-2350`) — so the slot is a register
            // reference only when `coalesce` is SET. Listing it unconditionally would claim an LRF reference for
            // every immediate-source store, and the second arm would then emit a zero for a register dcgbe's
            // allocator never handed out. A slot whose register-ness depends on ANOTHER operand's value is not
            // what this table encodes, which is why it is stated here instead.
            InstOpCode::LDST
            | InstOpCode::LDSTI
            | InstOpCode::LDSTIU
            | InstOpCode::LDSTU
            | InstOpCode::LDCVTI
            | InstOpCode::LDCVTIU => {
                &[(Operand::Src0, RegType::Lrf), (Operand::Src1, RegType::Lrf)]
            }

            // `:2252` `MVLOOPCNT`, `:2267` `SYNC`, and the `RETURN`/`NOP` the chain leaves to
            // `handleCommonInstructions` — none of them subscripts `reg_file_LRF`.
            InstOpCode::MVLOOPCNT | InstOpCode::NOP | InstOpCode::RETURN | InstOpCode::SYNC => &[],

            InstOpCode::ADDEARIMM
            | InstOpCode::ADDLARIMM
            | InstOpCode::COPY_LFSR
            | InstOpCode::EARIMM
            | InstOpCode::EARREGCOPY
            | InstOpCode::EE
            | InstOpCode::FCMP
            | InstOpCode::FCVT
            | InstOpCode::FEST
            | InstOpCode::FMA
            | InstOpCode::FMA4
            | InstOpCode::FMA8
            | InstOpCode::FMINMAX
            | InstOpCode::FMUL
            | InstOpCode::FNMS
            | InstOpCode::GCVT
            | InstOpCode::GTRIMM
            | InstOpCode::ICVT
            | InstOpCode::IMA4
            | InstOpCode::IMA8
            | InstOpCode::IME
            | InstOpCode::INCRMASK
            | InstOpCode::JADD
            | InstOpCode::JCMP
            | InstOpCode::JCMPI
            | InstOpCode::JCRSWAP
            | InstOpCode::JIMMCOPY
            | InstOpCode::JSUB
            | InstOpCode::LARIMM
            | InstOpCode::LARREGCOPY
            | InstOpCode::LD
            | InstOpCode::LDG
            | InstOpCode::LDGM
            | InstOpCode::LDGMU
            | InstOpCode::LDGU
            | InstOpCode::LDIGM
            | InstOpCode::LDIGMU
            | InstOpCode::LDIM
            | InstOpCode::LDIMU
            | InstOpCode::LDM
            | InstOpCode::LDMU
            | InstOpCode::LDU
            | InstOpCode::LDZ
            | InstOpCode::LDZU
            | InstOpCode::LDZimm16
            | InstOpCode::LOAD_LFSR
            | InstOpCode::LOGICAL
            | InstOpCode::LRFCOPY
            | InstOpCode::MERGE
            | InstOpCode::MODEARREG
            | InstOpCode::MODLARREG
            | InstOpCode::PACK
            | InstOpCode::PERMUTE
            | InstOpCode::REDUCE
            | InstOpCode::SAMV
            | InstOpCode::SELECT
            | InstOpCode::SETDEST
            | InstOpCode::SETDSTMASK
            | InstOpCode::SETMASK
            | InstOpCode::SHR
            | InstOpCode::SJCMP
            | InstOpCode::SPLAT
            | InstOpCode::SPMV
            | InstOpCode::ST
            | InstOpCode::STG
            | InstOpCode::STGU
            | InstOpCode::STIM
            | InstOpCode::STIMU
            | InstOpCode::STM
            | InstOpCode::STMU
            | InstOpCode::STU
            | InstOpCode::STZ
            | InstOpCode::SUBLARIMM
            | InstOpCode::TILEADV
            | InstOpCode::XMA4
            | InstOpCode::XRFACCESS => panic!(
                "an instruction the L0's own execution does not handle ran on an L0 unit \
                 (`memoryElement.cpp:2965-2967`: `Instruction not yet added.`)"
            ),
        },
        UnitClass::Lx => match op {
            InstOpCode::IMMCOPY
            | InstOpCode::LRFCOPY
            | InstOpCode::MODLRFIMM
            | InstOpCode::SUBLRFIMM => &[(Operand::Src0, RegType::Lrf)],

            InstOpCode::LRFREGCOPY | InstOpCode::MODLRFREG => {
                &[(Operand::Src0, RegType::Lrf), (Operand::Src1, RegType::Lrf)]
            }

            InstOpCode::LDST
            | InstOpCode::LDSTI
            | InstOpCode::LDSTIU
            | InstOpCode::LDSTU
            | InstOpCode::LDCVTI
            | InstOpCode::LDCVTIU => {
                &[(Operand::Src0, RegType::Lrf), (Operand::Src1, RegType::Lrf)]
            }
            // `// Visitation of JCRs not currently needed.` — the C++ says so in every arm that
            // lists a jump, so a JCR reference is a deliberate omission and not a missing case.
            InstOpCode::JADD | InstOpCode::JCMP | InstOpCode::JIMMCOPY | InstOpCode::JSUB => &[],

            InstOpCode::SAMV => &[(Operand::Mvridx, RegType::Mvr)],

            InstOpCode::MVLOOPCNT
            | InstOpCode::NOP
            | InstOpCode::RETURN
            | InstOpCode::SETDSTMASK
            | InstOpCode::SJCMP
            | InstOpCode::SPMV
            | InstOpCode::SYNC => &[],
            InstOpCode::ADDEARIMM
            | InstOpCode::ADDLARIMM
            | InstOpCode::COPY_LFSR
            | InstOpCode::EARIMM
            | InstOpCode::EARREGCOPY
            | InstOpCode::EE
            | InstOpCode::FCMP
            | InstOpCode::FCVT
            | InstOpCode::FEST
            | InstOpCode::FMA
            | InstOpCode::FMA4
            | InstOpCode::FMA8
            | InstOpCode::FMINMAX
            | InstOpCode::FMUL
            | InstOpCode::FNMS
            | InstOpCode::GCVT
            | InstOpCode::GTRIMM
            | InstOpCode::ICVT
            | InstOpCode::IMA4
            | InstOpCode::IMA8
            | InstOpCode::IME
            | InstOpCode::INCRMASK
            | InstOpCode::JCMPI
            | InstOpCode::JCRSWAP
            | InstOpCode::LARIMM
            | InstOpCode::LARREGCOPY
            | InstOpCode::LD
            | InstOpCode::LDG
            | InstOpCode::LDGM
            | InstOpCode::LDGMU
            | InstOpCode::LDGU
            | InstOpCode::LDIGM
            | InstOpCode::LDIGMU
            | InstOpCode::LDIM
            | InstOpCode::LDIMU
            | InstOpCode::LDM
            | InstOpCode::LDMU
            | InstOpCode::LDU
            | InstOpCode::LDZ
            | InstOpCode::LDZU
            | InstOpCode::LDZimm16
            | InstOpCode::LOAD_LFSR
            | InstOpCode::LOGICAL
            | InstOpCode::MERGE
            | InstOpCode::MODEARREG
            | InstOpCode::MODLARREG
            | InstOpCode::PACK
            | InstOpCode::PERMUTE
            | InstOpCode::REDUCE
            | InstOpCode::SELECT
            | InstOpCode::SETDEST
            | InstOpCode::SETMASK
            | InstOpCode::SHR
            | InstOpCode::SPLAT
            | InstOpCode::ST
            | InstOpCode::STG
            | InstOpCode::STGU
            | InstOpCode::STIM
            | InstOpCode::STIMU
            | InstOpCode::STM
            | InstOpCode::STMU
            | InstOpCode::STU
            | InstOpCode::STZ
            | InstOpCode::SUBLARIMM
            | InstOpCode::TILEADV
            | InstOpCode::XMA4
            | InstOpCode::XRFACCESS => panic!(
                "an instruction this unit's `RegVisitor` arm does not list ran on an LX unit \
                 (`regvisitor.cpp:181-183`: `Unhandled LX instruction while analyzing register references`)"
            ),
        },
        UnitClass::Pe => match op {
            InstOpCode::EE
            | InstOpCode::FCMP
            | InstOpCode::FEST
            | InstOpCode::FMA
            | InstOpCode::FMINMAX
            | InstOpCode::FMUL
            | InstOpCode::FNMS
            | InstOpCode::GCVT
            | InstOpCode::FCVT
            | InstOpCode::ICVT
            | InstOpCode::IME => &[],

            InstOpCode::IMMCOPY => &[(Operand::Src0, RegType::Lrf)],
            // `// Visitation of JCRs not currently needed.` — the C++ says so in every arm that
            // lists a jump, so a JCR reference is a deliberate omission and not a missing case.
            InstOpCode::JADD | InstOpCode::JCMP | InstOpCode::JIMMCOPY | InstOpCode::JSUB => &[],

            InstOpCode::LOGICAL
            | InstOpCode::MERGE
            | InstOpCode::MVLOOPCNT
            | InstOpCode::NOP
            | InstOpCode::PACK
            | InstOpCode::PERMUTE
            | InstOpCode::REDUCE
            | InstOpCode::RETURN
            | InstOpCode::SELECT
            | InstOpCode::SHR
            | InstOpCode::SJCMP
            | InstOpCode::SPLAT => &[],
            InstOpCode::ADDEARIMM
            | InstOpCode::ADDLARIMM
            | InstOpCode::COPY_LFSR
            | InstOpCode::EARIMM
            | InstOpCode::EARREGCOPY
            | InstOpCode::FMA4
            | InstOpCode::FMA8
            | InstOpCode::GTRIMM
            | InstOpCode::IMA4
            | InstOpCode::IMA8
            | InstOpCode::INCRMASK
            | InstOpCode::JCMPI
            | InstOpCode::JCRSWAP
            | InstOpCode::LARIMM
            | InstOpCode::LARREGCOPY
            | InstOpCode::LD
            | InstOpCode::LDCVTI
            | InstOpCode::LDCVTIU
            | InstOpCode::LDG
            | InstOpCode::LDGM
            | InstOpCode::LDGMU
            | InstOpCode::LDGU
            | InstOpCode::LDIGM
            | InstOpCode::LDIGMU
            | InstOpCode::LDIM
            | InstOpCode::LDIMU
            | InstOpCode::LDM
            | InstOpCode::LDMU
            | InstOpCode::LDST
            | InstOpCode::LDSTI
            | InstOpCode::LDSTIU
            | InstOpCode::LDSTU
            | InstOpCode::LDU
            | InstOpCode::LDZ
            | InstOpCode::LDZU
            | InstOpCode::LDZimm16
            | InstOpCode::LOAD_LFSR
            | InstOpCode::LRFCOPY
            | InstOpCode::LRFREGCOPY
            | InstOpCode::MODEARREG
            | InstOpCode::MODLARREG
            | InstOpCode::MODLRFIMM
            | InstOpCode::MODLRFREG
            | InstOpCode::SAMV
            | InstOpCode::SETDEST
            | InstOpCode::SETDSTMASK
            | InstOpCode::SETMASK
            | InstOpCode::SPMV
            | InstOpCode::ST
            | InstOpCode::STG
            | InstOpCode::STGU
            | InstOpCode::STIM
            | InstOpCode::STIMU
            | InstOpCode::STM
            | InstOpCode::STMU
            | InstOpCode::STU
            | InstOpCode::STZ
            | InstOpCode::SUBLARIMM
            | InstOpCode::SUBLRFIMM
            | InstOpCode::SYNC
            | InstOpCode::TILEADV
            | InstOpCode::XMA4
            | InstOpCode::XRFACCESS => panic!(
                "an instruction this unit's `RegVisitor` arm does not list ran on the PE — `LOAD_LFSR` and \
                 `COPY_LFSR` are named there and fall into the error deliberately (`regvisitor.cpp:242-245`)"
            ),
        },
        UnitClass::Sfp => match op {
            InstOpCode::EE
            | InstOpCode::FCMP
            | InstOpCode::FEST
            | InstOpCode::FMA
            | InstOpCode::FMINMAX
            | InstOpCode::FMUL
            | InstOpCode::FNMS
            | InstOpCode::GCVT
            | InstOpCode::FCVT
            | InstOpCode::ICVT
            | InstOpCode::IME => &[],

            InstOpCode::IMMCOPY => &[(Operand::Src0, RegType::Lrf)],
            // `// Visitation of JCRs not currently needed.` — the C++ says so in every arm that
            // lists a jump, so a JCR reference is a deliberate omission and not a missing case.
            InstOpCode::JADD | InstOpCode::JCMP | InstOpCode::JIMMCOPY | InstOpCode::JSUB => &[],

            InstOpCode::LOGICAL
            | InstOpCode::MERGE
            | InstOpCode::MVLOOPCNT
            | InstOpCode::NOP
            | InstOpCode::PACK
            | InstOpCode::PERMUTE
            | InstOpCode::REDUCE
            | InstOpCode::RETURN
            | InstOpCode::SELECT
            | InstOpCode::SHR
            | InstOpCode::SJCMP
            | InstOpCode::SPLAT
            | InstOpCode::SETDEST => &[],
            InstOpCode::ADDEARIMM
            | InstOpCode::ADDLARIMM
            | InstOpCode::COPY_LFSR
            | InstOpCode::EARIMM
            | InstOpCode::EARREGCOPY
            | InstOpCode::FMA4
            | InstOpCode::FMA8
            | InstOpCode::GTRIMM
            | InstOpCode::IMA4
            | InstOpCode::IMA8
            | InstOpCode::INCRMASK
            | InstOpCode::JCMPI
            | InstOpCode::JCRSWAP
            | InstOpCode::LARIMM
            | InstOpCode::LARREGCOPY
            | InstOpCode::LD
            | InstOpCode::LDCVTI
            | InstOpCode::LDCVTIU
            | InstOpCode::LDG
            | InstOpCode::LDGM
            | InstOpCode::LDGMU
            | InstOpCode::LDGU
            | InstOpCode::LDIGM
            | InstOpCode::LDIGMU
            | InstOpCode::LDIM
            | InstOpCode::LDIMU
            | InstOpCode::LDM
            | InstOpCode::LDMU
            | InstOpCode::LDST
            | InstOpCode::LDSTI
            | InstOpCode::LDSTIU
            | InstOpCode::LDSTU
            | InstOpCode::LDU
            | InstOpCode::LDZ
            | InstOpCode::LDZU
            | InstOpCode::LDZimm16
            | InstOpCode::LOAD_LFSR
            | InstOpCode::LRFCOPY
            | InstOpCode::LRFREGCOPY
            | InstOpCode::MODEARREG
            | InstOpCode::MODLARREG
            | InstOpCode::MODLRFIMM
            | InstOpCode::MODLRFREG
            | InstOpCode::SAMV
            | InstOpCode::SETDSTMASK
            | InstOpCode::SETMASK
            | InstOpCode::SPMV
            | InstOpCode::ST
            | InstOpCode::STG
            | InstOpCode::STGU
            | InstOpCode::STIM
            | InstOpCode::STIMU
            | InstOpCode::STM
            | InstOpCode::STMU
            | InstOpCode::STU
            | InstOpCode::STZ
            | InstOpCode::SUBLARIMM
            | InstOpCode::SUBLRFIMM
            | InstOpCode::SYNC
            | InstOpCode::TILEADV
            | InstOpCode::XMA4
            | InstOpCode::XRFACCESS => panic!(
                "an instruction this unit's `RegVisitor` arm does not list ran on the SFP — `LOAD_LFSR` and \
                 `COPY_LFSR` are named there and fall into the error deliberately (`regvisitor.cpp:290-293`)"
            ),
        },
    }
}
