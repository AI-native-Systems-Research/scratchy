// SPDX-License-Identifier: Apache-2.0
//! THE OPCODE VALUES — what each unit family actually encodes for an opcode.
//!
//! Ported from `/project_src/deeptools/sys-arch-spec/isa/isaSystemc.h:37-320`: `ptISAOpcodesT`,
//! `L0InstructionName`, `LXInstructionName`, `PEInstructionName`, `SFPInstructionName`, `L3InstructionName`.
//! `defineOpcode(unit, op, …)` (`isa.cpp:150-159`) reads `unit##_##op` out of exactly these enums.
//!
//! 173 (unit, opcode) pairs are defined; every other pair is refused.

use super::InstOpCode;

/// A UNIT FAMILY — which of `isaSystemc.h`'s six opcode enums applies.
///
/// ⛔ A FAMILY IS NOT A COMPONENT. There are nine components (PT, PE, SFP, L0LU, L0SU, LXLU, LXSU, L3LU, L3SU) and
/// six value enums: `L0LU` and `L0SU` both encode from `L0InstructionName`, and so on down. The direction is what
/// separates `L3_LD` from `L3_ST`, which share the value `0x20`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpUnit {
    /// `ptISAOpcodesT` — the PT's 64-slot opcode space.
    Ptop,
    /// `PEInstructionName`.
    Pe,
    /// `SFPInstructionName`.
    Sfp,
    /// `L0InstructionName`, for both L0LU and L0SU.
    L0,
    /// `LXInstructionName`, for both LXLU and LXSU.
    Lx,
    /// `L3InstructionName`, for both L3LU and L3SU.
    L3,
}

/// THE VALUE ONE UNIT ENCODES FOR ONE OPCODE — the low bits of the instruction word.
///
/// ⛔ NOT AN [`InstOpCode`], AND NOT AN INSTRUCTION TYPE. The opcode NAMES the operation, the value is what the
/// unit's decoder reads, and the type says which fields the word has. `FMA` is `PTOP_FMA = 3` on the PT and
/// `PE_FMA = 0x04` on the PE, so the value is meaningless without the unit it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpcodeValue(u8);

impl OpcodeValue {
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl OpUnit {
    /// WHICH FAMILY A COMPONENT'S OPCODES COME FROM — the L3 pair share one table, as do the LX pair and the L0
    /// pair, while the PT, PE and SFP each have their own (`isaSystemc.h`).
    ///
    /// ⛔ ONE MAPPING, NOT ONE PER CALLER. `bridges::i3`'s IBUFF pad and `dcgbe::opcode`'s validity check both need
    /// it, and two copies of a nine-arm match are two places for the L0/LX split to drift.
    pub const fn of_component(comp: crate::regfile::Component) -> Self {
        use crate::regfile::Component;
        match comp {
            Component::Pt => Self::Ptop,
            Component::Pe => Self::Pe,
            Component::Sfp => Self::Sfp,
            Component::L0lu | Component::L0su => Self::L0,
            Component::Lxlu | Component::Lxsu => Self::Lx,
            Component::L3lu | Component::L3su => Self::L3,
        }
    }

    /// The value this unit family encodes for `op`.
    ///
    /// ⛔ A PAIR THE HARDWARE DOES NOT HAVE IS A `panic!`, WHICH IS A BUILD ERROR HERE. `SFP` has no `LDST`; asking
    /// for one is a fault in whichever bridge chose the unit, not a value to invent.
    pub const fn value_of(self, op: InstOpCode) -> OpcodeValue {
        match self {
            OpUnit::Ptop => match op {
                InstOpCode::FMA => OpcodeValue(3),
                InstOpCode::FMA8 => OpcodeValue(5),
                InstOpCode::FMA4 => OpcodeValue(10),
                InstOpCode::IMA4 => OpcodeValue(8),
                InstOpCode::IMA8 => OpcodeValue(4),
                InstOpCode::XMA4 => OpcodeValue(9),
                InstOpCode::INCRMASK => OpcodeValue(34),
                InstOpCode::JADD => OpcodeValue(49),
                InstOpCode::JCMP => OpcodeValue(52),
                InstOpCode::JCRSWAP => OpcodeValue(39),
                InstOpCode::JIMMCOPY => OpcodeValue(51),
                InstOpCode::JSUB => OpcodeValue(50),
                InstOpCode::MVLOOPCNT => OpcodeValue(33),
                InstOpCode::NOP => OpcodeValue(32),
                InstOpCode::RETURN => OpcodeValue(63),
                InstOpCode::SETMASK => OpcodeValue(35),
                InstOpCode::SJCMP => OpcodeValue(53),
                InstOpCode::SYNC => OpcodeValue(38),
                InstOpCode::XRFACCESS => OpcodeValue(40),
                InstOpCode::ADDEARIMM
                | InstOpCode::ADDLARIMM
                | InstOpCode::COPY_LFSR
                | InstOpCode::EARIMM
                | InstOpCode::EARREGCOPY
                | InstOpCode::EE
                | InstOpCode::FCMP
                | InstOpCode::FCVT
                | InstOpCode::FEST
                | InstOpCode::FMINMAX
                | InstOpCode::FMUL
                | InstOpCode::FNMS
                | InstOpCode::GCVT
                | InstOpCode::GTRIMM
                | InstOpCode::ICVT
                | InstOpCode::IME
                | InstOpCode::IMMCOPY
                | InstOpCode::JCMPI
                | InstOpCode::LARIMM
                | InstOpCode::LARREGCOPY
                | InstOpCode::LD
                | InstOpCode::LDCVTI
                | InstOpCode::LDCVTIU
                | InstOpCode::LDG
                | InstOpCode::LDGM
                | InstOpCode::LDGMU
                | InstOpCode::LDGU
                | InstOpCode::LDM
                | InstOpCode::LDMU
                | InstOpCode::LDU
                | InstOpCode::LDIM
                | InstOpCode::LDIMU
                | InstOpCode::LDIGM
                | InstOpCode::LDIGMU
                | InstOpCode::LDZ
                | InstOpCode::LDZimm16
                | InstOpCode::LDZU
                | InstOpCode::LDST
                | InstOpCode::LDSTI
                | InstOpCode::LDSTIU
                | InstOpCode::LDSTU
                | InstOpCode::LOAD_LFSR
                | InstOpCode::LOGICAL
                | InstOpCode::LRFCOPY
                | InstOpCode::LRFREGCOPY
                | InstOpCode::MERGE
                | InstOpCode::MODEARREG
                | InstOpCode::MODLARREG
                | InstOpCode::MODLRFIMM
                | InstOpCode::MODLRFREG
                | InstOpCode::PACK
                | InstOpCode::PERMUTE
                | InstOpCode::REDUCE
                | InstOpCode::SAMV
                | InstOpCode::SELECT
                | InstOpCode::SETDEST
                | InstOpCode::SETDSTMASK
                | InstOpCode::SHR
                | InstOpCode::SPLAT
                | InstOpCode::SPMV
                | InstOpCode::ST
                | InstOpCode::STG
                | InstOpCode::STGU
                | InstOpCode::STM
                | InstOpCode::STMU
                | InstOpCode::STU
                | InstOpCode::STIM
                | InstOpCode::STIMU
                | InstOpCode::STZ
                | InstOpCode::SUBLARIMM
                | InstOpCode::SUBLRFIMM
                | InstOpCode::TILEADV => panic!(
                    "the Ptop encodes no such instruction; `isaSystemc.h`'s PTOP enum does not name it"
                ),
            },
            OpUnit::Pe => match op {
                InstOpCode::COPY_LFSR => OpcodeValue(26),
                InstOpCode::EE => OpcodeValue(13),
                InstOpCode::FCMP => OpcodeValue(10),
                InstOpCode::FCVT => OpcodeValue(17),
                InstOpCode::FEST => OpcodeValue(20),
                InstOpCode::FMA => OpcodeValue(4),
                InstOpCode::FMINMAX => OpcodeValue(11),
                InstOpCode::FMUL => OpcodeValue(0),
                InstOpCode::FNMS => OpcodeValue(6),
                InstOpCode::GCVT => OpcodeValue(22),
                InstOpCode::ICVT => OpcodeValue(8),
                InstOpCode::IME => OpcodeValue(9),
                InstOpCode::IMMCOPY => OpcodeValue(37),
                InstOpCode::JADD => OpcodeValue(49),
                InstOpCode::JCMP => OpcodeValue(52),
                InstOpCode::JCRSWAP => OpcodeValue(39),
                InstOpCode::JIMMCOPY => OpcodeValue(51),
                InstOpCode::JSUB => OpcodeValue(50),
                InstOpCode::LOAD_LFSR => OpcodeValue(25),
                InstOpCode::LOGICAL => OpcodeValue(16),
                InstOpCode::MERGE => OpcodeValue(18),
                InstOpCode::MVLOOPCNT => OpcodeValue(33),
                InstOpCode::NOP => OpcodeValue(48),
                InstOpCode::PACK => OpcodeValue(23),
                InstOpCode::PERMUTE => OpcodeValue(24),
                InstOpCode::REDUCE => OpcodeValue(42),
                InstOpCode::RETURN => OpcodeValue(63),
                InstOpCode::SELECT => OpcodeValue(14),
                InstOpCode::SHR => OpcodeValue(15),
                InstOpCode::SJCMP => OpcodeValue(53),
                InstOpCode::SPLAT => OpcodeValue(21),
                InstOpCode::SYNC => OpcodeValue(38),
                InstOpCode::ADDEARIMM
                | InstOpCode::ADDLARIMM
                | InstOpCode::EARIMM
                | InstOpCode::EARREGCOPY
                | InstOpCode::FMA8
                | InstOpCode::FMA4
                | InstOpCode::GTRIMM
                | InstOpCode::IMA4
                | InstOpCode::IMA8
                | InstOpCode::XMA4
                | InstOpCode::INCRMASK
                | InstOpCode::JCMPI
                | InstOpCode::LARIMM
                | InstOpCode::LARREGCOPY
                | InstOpCode::LD
                | InstOpCode::LDCVTI
                | InstOpCode::LDCVTIU
                | InstOpCode::LDG
                | InstOpCode::LDGM
                | InstOpCode::LDGMU
                | InstOpCode::LDGU
                | InstOpCode::LDM
                | InstOpCode::LDMU
                | InstOpCode::LDU
                | InstOpCode::LDIM
                | InstOpCode::LDIMU
                | InstOpCode::LDIGM
                | InstOpCode::LDIGMU
                | InstOpCode::LDZ
                | InstOpCode::LDZimm16
                | InstOpCode::LDZU
                | InstOpCode::LDST
                | InstOpCode::LDSTI
                | InstOpCode::LDSTIU
                | InstOpCode::LDSTU
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
                | InstOpCode::STM
                | InstOpCode::STMU
                | InstOpCode::STU
                | InstOpCode::STIM
                | InstOpCode::STIMU
                | InstOpCode::STZ
                | InstOpCode::SUBLARIMM
                | InstOpCode::SUBLRFIMM
                | InstOpCode::TILEADV
                | InstOpCode::XRFACCESS => panic!(
                    "the Pe encodes no such instruction; `isaSystemc.h`'s PE enum does not name it"
                ),
            },
            OpUnit::Sfp => match op {
                InstOpCode::COPY_LFSR => OpcodeValue(26),
                InstOpCode::EE => OpcodeValue(13),
                InstOpCode::FCMP => OpcodeValue(10),
                InstOpCode::FCVT => OpcodeValue(17),
                InstOpCode::FEST => OpcodeValue(20),
                InstOpCode::FMA => OpcodeValue(4),
                InstOpCode::FMINMAX => OpcodeValue(11),
                InstOpCode::FMUL => OpcodeValue(0),
                InstOpCode::FNMS => OpcodeValue(6),
                InstOpCode::GCVT => OpcodeValue(22),
                InstOpCode::ICVT => OpcodeValue(8),
                InstOpCode::IME => OpcodeValue(9),
                InstOpCode::IMMCOPY => OpcodeValue(37),
                InstOpCode::JADD => OpcodeValue(49),
                InstOpCode::JCMP => OpcodeValue(52),
                InstOpCode::JCRSWAP => OpcodeValue(39),
                InstOpCode::JIMMCOPY => OpcodeValue(51),
                InstOpCode::JSUB => OpcodeValue(50),
                InstOpCode::LOAD_LFSR => OpcodeValue(25),
                InstOpCode::LOGICAL => OpcodeValue(16),
                InstOpCode::MERGE => OpcodeValue(18),
                InstOpCode::MVLOOPCNT => OpcodeValue(33),
                InstOpCode::NOP => OpcodeValue(48),
                InstOpCode::PACK => OpcodeValue(23),
                InstOpCode::PERMUTE => OpcodeValue(24),
                InstOpCode::REDUCE => OpcodeValue(42),
                InstOpCode::RETURN => OpcodeValue(63),
                InstOpCode::SELECT => OpcodeValue(14),
                InstOpCode::SETDEST => OpcodeValue(54),
                InstOpCode::SHR => OpcodeValue(15),
                InstOpCode::SJCMP => OpcodeValue(53),
                InstOpCode::SPLAT => OpcodeValue(21),
                InstOpCode::SYNC => OpcodeValue(38),
                InstOpCode::ADDEARIMM
                | InstOpCode::ADDLARIMM
                | InstOpCode::EARIMM
                | InstOpCode::EARREGCOPY
                | InstOpCode::FMA8
                | InstOpCode::FMA4
                | InstOpCode::GTRIMM
                | InstOpCode::IMA4
                | InstOpCode::IMA8
                | InstOpCode::XMA4
                | InstOpCode::INCRMASK
                | InstOpCode::JCMPI
                | InstOpCode::LARIMM
                | InstOpCode::LARREGCOPY
                | InstOpCode::LD
                | InstOpCode::LDCVTI
                | InstOpCode::LDCVTIU
                | InstOpCode::LDG
                | InstOpCode::LDGM
                | InstOpCode::LDGMU
                | InstOpCode::LDGU
                | InstOpCode::LDM
                | InstOpCode::LDMU
                | InstOpCode::LDU
                | InstOpCode::LDIM
                | InstOpCode::LDIMU
                | InstOpCode::LDIGM
                | InstOpCode::LDIGMU
                | InstOpCode::LDZ
                | InstOpCode::LDZimm16
                | InstOpCode::LDZU
                | InstOpCode::LDST
                | InstOpCode::LDSTI
                | InstOpCode::LDSTIU
                | InstOpCode::LDSTU
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
                | InstOpCode::STM
                | InstOpCode::STMU
                | InstOpCode::STU
                | InstOpCode::STIM
                | InstOpCode::STIMU
                | InstOpCode::STZ
                | InstOpCode::SUBLARIMM
                | InstOpCode::SUBLRFIMM
                | InstOpCode::TILEADV
                | InstOpCode::XRFACCESS => panic!(
                    "the Sfp encodes no such instruction; `isaSystemc.h`'s SFP enum does not name it"
                ),
            },
            OpUnit::L0 => match op {
                InstOpCode::IMMCOPY => OpcodeValue(12),
                InstOpCode::JADD => OpcodeValue(36),
                InstOpCode::JCMP => OpcodeValue(34),
                InstOpCode::JCRSWAP => OpcodeValue(35),
                InstOpCode::JIMMCOPY => OpcodeValue(39),
                InstOpCode::JSUB => OpcodeValue(37),
                InstOpCode::LDST => OpcodeValue(49),
                InstOpCode::LDSTI => OpcodeValue(57),
                InstOpCode::LDSTIU => OpcodeValue(25),
                InstOpCode::LDSTU => OpcodeValue(17),
                InstOpCode::LRFREGCOPY => OpcodeValue(4),
                InstOpCode::MODLRFIMM => OpcodeValue(8),
                InstOpCode::MODLRFREG => OpcodeValue(0),
                InstOpCode::MVLOOPCNT => OpcodeValue(33),
                InstOpCode::NOP => OpcodeValue(48),
                InstOpCode::RETURN => OpcodeValue(63),
                InstOpCode::SJCMP => OpcodeValue(50),
                InstOpCode::SYNC => OpcodeValue(38),
                InstOpCode::TILEADV => OpcodeValue(40),
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
                | InstOpCode::FMA8
                | InstOpCode::FMA4
                | InstOpCode::FMINMAX
                | InstOpCode::FMUL
                | InstOpCode::FNMS
                | InstOpCode::GCVT
                | InstOpCode::GTRIMM
                | InstOpCode::ICVT
                | InstOpCode::IMA4
                | InstOpCode::IMA8
                | InstOpCode::XMA4
                | InstOpCode::IME
                | InstOpCode::INCRMASK
                | InstOpCode::JCMPI
                | InstOpCode::LARIMM
                | InstOpCode::LARREGCOPY
                | InstOpCode::LD
                | InstOpCode::LDCVTI
                | InstOpCode::LDCVTIU
                | InstOpCode::LDG
                | InstOpCode::LDGM
                | InstOpCode::LDGMU
                | InstOpCode::LDGU
                | InstOpCode::LDM
                | InstOpCode::LDMU
                | InstOpCode::LDU
                | InstOpCode::LDIM
                | InstOpCode::LDIMU
                | InstOpCode::LDIGM
                | InstOpCode::LDIGMU
                | InstOpCode::LDZ
                | InstOpCode::LDZimm16
                | InstOpCode::LDZU
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
                | InstOpCode::SPLAT
                | InstOpCode::SPMV
                | InstOpCode::ST
                | InstOpCode::STG
                | InstOpCode::STGU
                | InstOpCode::STM
                | InstOpCode::STMU
                | InstOpCode::STU
                | InstOpCode::STIM
                | InstOpCode::STIMU
                | InstOpCode::STZ
                | InstOpCode::SUBLARIMM
                | InstOpCode::SUBLRFIMM
                | InstOpCode::XRFACCESS => panic!(
                    "the L0 encodes no such instruction; `isaSystemc.h`'s L0 enum does not name it"
                ),
            },
            OpUnit::Lx => match op {
                InstOpCode::IMMCOPY => OpcodeValue(12),
                InstOpCode::JADD => OpcodeValue(36),
                InstOpCode::JCMP => OpcodeValue(34),
                InstOpCode::JCRSWAP => OpcodeValue(35),
                InstOpCode::JIMMCOPY => OpcodeValue(39),
                InstOpCode::JSUB => OpcodeValue(37),
                InstOpCode::LDCVTI => OpcodeValue(59),
                InstOpCode::LDCVTIU => OpcodeValue(27),
                InstOpCode::LDST => OpcodeValue(49),
                InstOpCode::LDSTI => OpcodeValue(57),
                InstOpCode::LDSTIU => OpcodeValue(25),
                InstOpCode::LDSTU => OpcodeValue(17),
                InstOpCode::LRFCOPY => OpcodeValue(5),
                InstOpCode::LRFREGCOPY => OpcodeValue(4),
                InstOpCode::MODLRFIMM => OpcodeValue(8),
                InstOpCode::MODLRFREG => OpcodeValue(0),
                InstOpCode::MVLOOPCNT => OpcodeValue(33),
                InstOpCode::NOP => OpcodeValue(48),
                InstOpCode::RETURN => OpcodeValue(63),
                InstOpCode::SAMV => OpcodeValue(42),
                InstOpCode::SETDSTMASK => OpcodeValue(43),
                InstOpCode::SJCMP => OpcodeValue(50),
                InstOpCode::SPMV => OpcodeValue(41),
                InstOpCode::SUBLRFIMM => OpcodeValue(13),
                InstOpCode::SYNC => OpcodeValue(38),
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
                | InstOpCode::FMA8
                | InstOpCode::FMA4
                | InstOpCode::FMINMAX
                | InstOpCode::FMUL
                | InstOpCode::FNMS
                | InstOpCode::GCVT
                | InstOpCode::GTRIMM
                | InstOpCode::ICVT
                | InstOpCode::IMA4
                | InstOpCode::IMA8
                | InstOpCode::XMA4
                | InstOpCode::IME
                | InstOpCode::INCRMASK
                | InstOpCode::JCMPI
                | InstOpCode::LARIMM
                | InstOpCode::LARREGCOPY
                | InstOpCode::LD
                | InstOpCode::LDG
                | InstOpCode::LDGM
                | InstOpCode::LDGMU
                | InstOpCode::LDGU
                | InstOpCode::LDM
                | InstOpCode::LDMU
                | InstOpCode::LDU
                | InstOpCode::LDIM
                | InstOpCode::LDIMU
                | InstOpCode::LDIGM
                | InstOpCode::LDIGMU
                | InstOpCode::LDZ
                | InstOpCode::LDZimm16
                | InstOpCode::LDZU
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
                | InstOpCode::STM
                | InstOpCode::STMU
                | InstOpCode::STU
                | InstOpCode::STIM
                | InstOpCode::STIMU
                | InstOpCode::STZ
                | InstOpCode::SUBLARIMM
                | InstOpCode::TILEADV
                | InstOpCode::XRFACCESS => panic!(
                    "the Lx encodes no such instruction; `isaSystemc.h`'s LX enum does not name it"
                ),
            },
            OpUnit::L3 => match op {
                InstOpCode::ADDEARIMM => OpcodeValue(5),
                InstOpCode::ADDLARIMM => OpcodeValue(1),
                InstOpCode::EARIMM => OpcodeValue(4),
                InstOpCode::EARREGCOPY => OpcodeValue(20),
                InstOpCode::GTRIMM => OpcodeValue(8),
                InstOpCode::JADD => OpcodeValue(51),
                InstOpCode::JCMP => OpcodeValue(52),
                InstOpCode::JCMPI => OpcodeValue(52),
                InstOpCode::JCRSWAP => OpcodeValue(50),
                InstOpCode::JIMMCOPY => OpcodeValue(49),
                InstOpCode::JSUB => OpcodeValue(53),
                InstOpCode::LARIMM => OpcodeValue(0),
                InstOpCode::LARREGCOPY => OpcodeValue(16),
                InstOpCode::LD => OpcodeValue(32),
                InstOpCode::LDG => OpcodeValue(40),
                InstOpCode::LDGM => OpcodeValue(41),
                InstOpCode::LDGMU => OpcodeValue(43),
                InstOpCode::LDGU => OpcodeValue(42),
                InstOpCode::LDM => OpcodeValue(33),
                InstOpCode::LDMU => OpcodeValue(35),
                InstOpCode::LDU => OpcodeValue(34),
                InstOpCode::LDIM => OpcodeValue(45),
                InstOpCode::LDIMU => OpcodeValue(47),
                InstOpCode::LDIGM => OpcodeValue(44),
                InstOpCode::LDIGMU => OpcodeValue(46),
                InstOpCode::LDZ => OpcodeValue(36),
                InstOpCode::LDZimm16 => OpcodeValue(36),
                InstOpCode::LDZU => OpcodeValue(38),
                InstOpCode::MODEARREG => OpcodeValue(21),
                InstOpCode::MODLARREG => OpcodeValue(17),
                InstOpCode::MVLOOPCNT => OpcodeValue(48),
                InstOpCode::NOP => OpcodeValue(54),
                InstOpCode::RETURN => OpcodeValue(55),
                InstOpCode::SJCMP => OpcodeValue(60),
                InstOpCode::ST => OpcodeValue(32),
                InstOpCode::STG => OpcodeValue(40),
                InstOpCode::STGU => OpcodeValue(42),
                InstOpCode::STM => OpcodeValue(33),
                InstOpCode::STMU => OpcodeValue(35),
                InstOpCode::STU => OpcodeValue(34),
                InstOpCode::STIM => OpcodeValue(45),
                InstOpCode::STIMU => OpcodeValue(47),
                InstOpCode::STZ => OpcodeValue(36),
                InstOpCode::SUBLARIMM => OpcodeValue(2),
                InstOpCode::SYNC => OpcodeValue(12),
                InstOpCode::COPY_LFSR
                | InstOpCode::EE
                | InstOpCode::FCMP
                | InstOpCode::FCVT
                | InstOpCode::FEST
                | InstOpCode::FMA
                | InstOpCode::FMA8
                | InstOpCode::FMA4
                | InstOpCode::FMINMAX
                | InstOpCode::FMUL
                | InstOpCode::FNMS
                | InstOpCode::GCVT
                | InstOpCode::ICVT
                | InstOpCode::IMA4
                | InstOpCode::IMA8
                | InstOpCode::XMA4
                | InstOpCode::IME
                | InstOpCode::IMMCOPY
                | InstOpCode::INCRMASK
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
                | InstOpCode::XRFACCESS => panic!(
                    "the L3 encodes no such instruction; `isaSystemc.h`'s L3 enum does not name it"
                ),
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE VALUES THE PORT MUST NOT SMOOTH OVER
// ─────────────────────────────────────────────────────────────────────────────

/// One opcode, two units, two values: `PTOP_FMA = 3` and `PE_FMA = 0x04`.
const _: () = assert!(OpUnit::Ptop.value_of(InstOpCode::FMA).get() == 3);
const _: () = assert!(OpUnit::Pe.value_of(InstOpCode::FMA).get() == 0x04);
const _: () = assert!(
    OpUnit::Ptop.value_of(InstOpCode::FMA).get() != OpUnit::Pe.value_of(InstOpCode::FMA).get()
);

/// ⛔ `L3_LD` AND `L3_ST` ARE BOTH `0x20`. The value does not say which it is; the component's DIRECTION does, and
/// L3LU/L3SU share one value enum. A port that keyed a load/store decision on the encoded value would read a store
/// as a load.
const _: () = assert!(OpUnit::L3.value_of(InstOpCode::LD).get() == 0x20);
const _: () = assert!(OpUnit::L3.value_of(InstOpCode::ST).get() == 0x20);

/// ⛔ `JCMPI` IS A PSEUDO-INSTRUCTION: `#define L3_JCMPI L3_JCMP` (`isa.hpp:38`). It is in `InstOpCode` and in no
/// value enum, so it encodes as `JCMP` — and only on the L3.
const _: () = assert!(
    OpUnit::L3.value_of(InstOpCode::JCMPI).get() == OpUnit::L3.value_of(InstOpCode::JCMP).get()
);
const _: () = assert!(OpUnit::L3.value_of(InstOpCode::JCMP).get() == 0x34);

/// ⛔ `L3_LDZimm16 = L3_LDZ` IS AN ALIAS TOO, and the two are one value.
const _: () = assert!(
    OpUnit::L3.value_of(InstOpCode::LDZimm16).get() == OpUnit::L3.value_of(InstOpCode::LDZ).get()
);

/// The SYNC value is `0x26` on both L0 and LX, and `0x0c` on the L3 — equal on two families and not the third, so a
/// port that took one for granted would be right twice.
const _: () = assert!(OpUnit::L0.value_of(InstOpCode::SYNC).get() == 0x26);
const _: () = assert!(OpUnit::Lx.value_of(InstOpCode::SYNC).get() == 0x26);
const _: () = assert!(OpUnit::L3.value_of(InstOpCode::SYNC).get() == 0x0c);

/// The PT's space is 6 bits wide (64 slots), and `PTOP_RETURN = 63` is its top.
const _: () = assert!(OpUnit::Ptop.value_of(InstOpCode::RETURN).get() == 63);
/// Every value fits six bits, on every family — the opcode field's width.
const _: () = assert!(OpUnit::L3.value_of(InstOpCode::LDIMU).get() < 64);
const _: () = assert!(OpUnit::Lx.value_of(InstOpCode::LDCVTI).get() == 0x3b);
const _: () = assert!(OpUnit::Sfp.value_of(InstOpCode::FMA).get() < 64);
