// SPDX-License-Identifier: Apache-2.0
//! THE INSTRUCTION LAYOUT TABLES — which fields each instruction type has, and where their bits are.
//!
//! Ported from `/project_src/deeptools/sys-arch-spec/isa/isa.cpp:161-1221` (`Isa::initIsa`): every
//! `defineOpcode(unit, op, instType, minArch)` (macro at `isa.cpp:150-159`) and every `defineField(type, name,
//! bitPos, encodeList, immLen, sign)` (lambda at `isa.cpp:163-221`).
//!
//! 440 opcode definitions and 1075 field definitions — 546 fields on RCUDD1A, 529 on SEN1P5.
//!
//! ⭐ `initIsa` RUNS ONCE PER COMPONENT AND PER ARCH, so its `if (myComponent == …)` / `if (coreArch …)` branches
//! are not conditions this crate evaluates: they SELECT the table. Both selections are constants for us — the
//! component is which unit an op names, the arch is a cargo feature — so what the C++ decides at construction is
//! resolved here into one table per (arch, component).
//!
//! ⛔ AN INSTRUCTION TYPE IS SCOPED TO ITS COMPONENT. Type 20 on the PT and type 20 on the PE are different field
//! sets, because `typeToFieldName` is a member of a per-component `Isa`. A table keyed by type alone would merge
//! them.
//!
//! ⭐⭐ THIS FILE USED TO LIVE IN `ddl/`, OUTSIDE `src/`, BECAUSE IT HELD THE FIELD NAMES AS TEXT — and this crate
//! bars strings from islands. Only `build.rs` can read a file outside `src/`, so that one placement handed the
//! whole ISA to `build.rs` and the encoder followed its data there. The names are now
//! [`crate::operand::Operand`] variants, so there are no strings, and the table belongs where the rest of the
//! ISA does.

use super::operand::Operand;

/// A core generation — `IsaCoreGen` (`isa.hpp:25-32`), as `defineOpcode`'s `minArch` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gen {
    Mpw2,
    Mpw3,
    Mpw4,
    Rcudd1a,
    Sen1p5,
}

/// WHICH FIELD LAYOUT AN INSTRUCTION HAS — `defineOpcode`/`defineField`'s `typeName` (`isa.cpp:163-167`), the key
/// of `typeToFieldShift`, `typeToFieldName`, `typeToFieldEncoding` and `typeToImmInfo`.
///
/// ⛔⛔ A TYPE IS NOT AN OPCODE VALUE, NOT A BIT AND NOT A FIELD'S VALUE. The value is what the decoder reads
/// (`isa::values`); the type says which fields the word HAS, and `SFP/SHR` is type 31 up to RCUDD1A and type 30
/// above it at one unchanged value — so the same instruction takes a different layout per arch.
///
/// ⛔⛔⛔ AN ENUM, BECAUSE THE SET IS CLOSED. The types are exactly the first arguments the vendored `defineField`
/// and `defineOpcode` calls pass — THIRTY-FIVE of them across both arches — so a thirty-sixth means a new call in
/// `isa.cpp`, which must break this build rather than appear as a number that matches no field table. A newtype
/// over `u16` admits a type of 999 — representable, constructible, and silently matching nothing.
///
/// ⭐ EACH VARIANT IS NAMED BY ITS OWN NUMBER, so nothing is invented: the C++ names no set of these, and giving
/// them descriptive names would assert a grouping the ISA does not state. `T20` is type 20 and says no more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InstrType {
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    T20,
    T21,
    T22,
    T24,
    T30,
    T31,
    T32,
    T40,
    T41,
    T43,
    T50,
    T60,
    T111,
    T118,
    T1000,
    T1010,
    T1020,
    T1030,
    T1040,
    T1100,
    T1110,
    T1112,
    T1120,
    T2210,
    T2300,
    T2310,
}

impl InstrType {
    /// Every type the vendored tables name, ascending.
    pub const ALL: [Self; 35] = [
        Self::T10,
        Self::T11,
        Self::T12,
        Self::T13,
        Self::T14,
        Self::T15,
        Self::T16,
        Self::T17,
        Self::T18,
        Self::T20,
        Self::T21,
        Self::T22,
        Self::T24,
        Self::T30,
        Self::T31,
        Self::T32,
        Self::T40,
        Self::T41,
        Self::T43,
        Self::T50,
        Self::T60,
        Self::T111,
        Self::T118,
        Self::T1000,
        Self::T1010,
        Self::T1020,
        Self::T1030,
        Self::T1040,
        Self::T1100,
        Self::T1110,
        Self::T1112,
        Self::T1120,
        Self::T2210,
        Self::T2300,
        Self::T2310,
    ];

    /// The number `defineField` was given.
    pub const fn get(self) -> u16 {
        match self {
            Self::T10 => 10,
            Self::T11 => 11,
            Self::T12 => 12,
            Self::T13 => 13,
            Self::T14 => 14,
            Self::T15 => 15,
            Self::T16 => 16,
            Self::T17 => 17,
            Self::T18 => 18,
            Self::T20 => 20,
            Self::T21 => 21,
            Self::T22 => 22,
            Self::T24 => 24,
            Self::T30 => 30,
            Self::T31 => 31,
            Self::T32 => 32,
            Self::T40 => 40,
            Self::T41 => 41,
            Self::T43 => 43,
            Self::T50 => 50,
            Self::T60 => 60,
            Self::T111 => 111,
            Self::T118 => 118,
            Self::T1000 => 1000,
            Self::T1010 => 1010,
            Self::T1020 => 1020,
            Self::T1030 => 1030,
            Self::T1040 => 1040,
            Self::T1100 => 1100,
            Self::T1110 => 1110,
            Self::T1112 => 1112,
            Self::T1120 => 1120,
            Self::T2210 => 2210,
            Self::T2300 => 2300,
            Self::T2310 => 2310,
        }
    }
}

// ⛔ THE 35 IS ASSERTED AGAINST THE ARRAY so a variant added without an `ALL` entry — or the reverse — is a
// build error rather than a type that silently never appears in a walk over the set.
const _: () = assert!(
    InstrType::ALL.len() == 35,
    "every instruction type the tables name is in ALL"
);

/// ⭐ A TYPE PRINTS AS ITS NUMBER, because that is what it IS — `defineField`'s first argument. Every message
/// naming one is quoting the ISA back, so `T20` would be this crate's spelling of a number the C++ writes as 20.
impl core::fmt::Display for InstrType {
    fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(out, "{}", self.get())
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// THE ENCODE LISTS, NAMED ONCE.
//
// ⛔⛔ THESE 92 VALUES WERE WRITTEN OUT 877 TIMES. `&[Enc::Lit("be", 1)]` appeared 185 times,
// `&[Enc::Lit("yes", 1), Enc::Lit("no", 0)]` 130 times, the register family 119 times. A field's
// encoding is a property of the FIELD KIND, not of the row, so restating it per row turned ~785 rows
// of real information into ~9,300 lines, and gave every copy its own chance to be wrong.
//
// ⭐ ORDERED BY USE, so the list doubles as a census of which encodings the ISA actually leans on.
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Used 185×.
const ENC_BE: &[Enc] = &[Enc::Lit("be", 1)];
/// Used 130×.
const ENC_YES_NO: &[Enc] = &[Enc::Lit("yes", 1), Enc::Lit("no", 0)];
/// Used 119×.
const ENC_REGS: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::ZERO)];
/// Used 54×.
const ENC_JCRS: &[Enc] = &[Enc::Family(Family::Jcrs, Count::NumJcrs, RegBase::ZERO)];
/// Used 40×.
const ENC_LCCRS_JCRS: &[Enc] = &[Enc::Family(Family::Lccrs, Count::NumLccrs, RegBase::ZERO), Enc::Family(Family::Jcrs, Count::NumJcrs, RegBase::ZERO)];
/// Used 22×.
const ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER: &[Enc] = &[Enc::Lit("always", 0), Enc::Lit("lt", 1), Enc::Lit("eq", 2), Enc::Lit("le", 3), Enc::Lit("gt", 4), Enc::Lit("ne", 5), Enc::Lit("ge", 6), Enc::Lit("never", 7)];
/// Used 21×.
const ENC_NO_SRC0_SRC2_RESULT: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("src0", 1), Enc::Lit("src2", 2), Enc::Lit("result", 3)];
/// Used 18×.
const ENC_USEJCR_USELCCR: &[Enc] = &[Enc::Lit("usejcr", 1), Enc::Lit("uselccr", 0)];
/// Used 16×.
const ENC_NO_YES: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("yes", 1)];
/// Used 16×.
const ENC_MIX1: &[Enc] = &[Enc::Family( Family::BurstLen, Count::Fixed(32), RegBase::ZERO, )];
/// Used 13×.
const ENC_NO_RESULT: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("result", 1)];
/// Used 13×.
const ENC_FP16_FP32: &[Enc] = &[Enc::Lit("fp16", 0), Enc::Lit("fp32", 1)];
/// Used 12×.
const ENC_GTRS: &[Enc] = &[Enc::Family(Family::Gtrs, Count::NumRegs, RegBase::ZERO)];
/// Used 11×.
const ENC_X1_X2_X4_X8: &[Enc] = &[Enc::Lit("x1", 0), Enc::Lit("x2", 1), Enc::Lit("x4", 2), Enc::Lit("x8", 3)];
/// Used 8×.
const ENC_BURST: &[Enc] = &[Enc::Lit("burst", 1)];
/// Used 8×.
const ENC_SELF_SFP_LXSU_PE: &[Enc] = &[Enc::Lit("self", 0), Enc::Lit("sfp", 1), Enc::Lit("lxsu", 2), Enc::Lit("pe", 3)];
/// Used 8×.
const ENC_MVRS: &[Enc] = &[Enc::Family(Family::Mvrs, Count::NumMvrs, RegBase::ZERO)];
/// Used 7×.
const ENC_MIX2: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("no", 15)];
/// Used 6×.
const ENC_16B_8B_4B_2B: &[Enc] = &[Enc::Lit("16b", 0), Enc::Lit("8b", 1), Enc::Lit("4b", 2), Enc::Lit("2b", 3)];
/// Used 6×.
const ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR: &[Enc] = &[Enc::Lit("folda", 0), Enc::Lit("foldb", 1), Enc::Lit("2fold2instr", 2), Enc::Lit("2fold1instr", 3)];
/// Used 6×.
const ENC_NO_CCW_CW_BOTH: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("ccw", 1), Enc::Lit("cw", 2), Enc::Lit("both", 3)];
/// Used 5×.
const ENC_MIX3: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("no", 0)];
/// Used 4×.
const ENC_NO_SET_INCR: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("set", 1), Enc::Lit("incr", 2)];
/// Used 4×.
const ENC_MIX6: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN)];
/// Used 4×.
const ENC_X1_X2_X3_X4: &[Enc] = &[Enc::Lit("x1", 1), Enc::Lit("x2", 2), Enc::Lit("x3", 3), Enc::Lit("x4", 4)];
/// Used 4×.
const ENC_SPLAT: &[Enc] = &[Enc::Lit("splat", 1)];
/// Used 4×.
const ENC_SEND_RECV_SENDRECV: &[Enc] = &[Enc::Lit("send", 1), Enc::Lit("recv", 2), Enc::Lit("sendrecv", 3)];
/// Used 4×.
const ENC_SFP_L0SU_PT: &[Enc] = &[Enc::Lit("sfp", 0), Enc::Lit("l0su", 1), Enc::Lit("pt", 2)];
/// Used 4×.
const ENC_NO_16B_32B_48B_64B_80B_96B_112B: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("16b", 1), Enc::Lit("32b", 2), Enc::Lit("48b", 3), Enc::Lit("64b", 4), Enc::Lit("80b", 5), Enc::Lit("96b", 6), Enc::Lit("112b", 7)];
/// Used 4×.
const ENC_128B_2BSPLAT_16BZPAD_16BSPLAT: &[Enc] = &[Enc::Lit("128b", 0), Enc::Lit("2bsplat", 1), Enc::Lit("16bzpad", 2), Enc::Lit("16bsplat", 3)];
/// Used 4×.
const ENC_128B_2BSPLAT_4BSPLAT_16BSPLAT: &[Enc] = &[Enc::Lit("128b", 0), Enc::Lit("2bsplat", 1), Enc::Lit("4bsplat", 2), Enc::Lit("16bsplat", 3)];
/// Used 4×.
const ENC_NO_X1_X2_X4: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("x1", 1), Enc::Lit("x2", 2), Enc::Lit("x4", 3)];
/// Used 4×.
const ENC_64B_32B_16B_8B_4B_2B_1B: &[Enc] = &[Enc::Lit("64b", 6), Enc::Lit("32b", 5), Enc::Lit("16b", 4), Enc::Lit("8b", 3), Enc::Lit("4b", 2), Enc::Lit("2b", 1), Enc::Lit("1b", 0)];
/// Used 4×.
const ENC_1B_2B_4B_8B_16B_32B_64B_128B: &[Enc] = &[Enc::Lit("1b", 0), Enc::Lit("2b", 1), Enc::Lit("4b", 2), Enc::Lit("8b", 3), Enc::Lit("16b", 4), Enc::Lit("32b", 5), Enc::Lit("64b", 6), Enc::Lit("128b", 7)];
/// Used 4×.
const ENC_NO_2B_4B_6B: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("2b", 1), Enc::Lit("4b", 2), Enc::Lit("6b", 3)];
/// Used 4×.
const ENC_128B_16B_2B: &[Enc] = &[Enc::Lit("128b", 0), Enc::Lit("16b", 1), Enc::Lit("2b", 2)];
/// Used 4×.
const ENC_ZERO_SFP_PE_LXLU: &[Enc] = &[Enc::Lit("zero", 0), Enc::Lit("sfp", 1), Enc::Lit("pe", 2), Enc::Lit("lxlu", 3)];
/// Used 4×.
const ENC_MIX4: &[Enc] = &[Enc::Family(Family::Jcrs, Count::NumRegs, RegBase::ZERO)];
/// Used 4×.
const ENC_MIX7: &[Enc] = &[Enc::Lit("sendlxlu0", 17), Enc::Lit("sendlxlu1", 65), Enc::Lit("sendlxsu0", 33), Enc::Lit("sendlxsu1", 129), Enc::Lit("sendlxluboth", 81), Enc::Lit("sendlxsuboth", 161), Enc::Lit("sendl3lu", 5), Enc::Lit("sendl3su", 9), Enc::Lit("recvlxlu0", 18), Enc::Lit("recvlxlu1", 66), Enc::Lit("recvlxsu0", 34), Enc::Lit("recvlxsu1", 130), Enc::Lit("recvlxluboth", 82), Enc::Lit("recvlxsuboth", 162), Enc::Lit("recvl3lu", 6), Enc::Lit("recvl3su", 10)];
/// Used 4×.
const ENC_READ_WRITE: &[Enc] = &[Enc::Lit("read", 1), Enc::Lit("write", 0)];
/// Used 4×.
const ENC_ZERO2LX_IMM2LX_ZR2LX: &[Enc] = &[Enc::Lit("zero2lx", 0), Enc::Lit("imm2lx", 1), Enc::Lit("zr2lx", 3)];
/// Used 4×.
const ENC_IMM2ZR: &[Enc] = &[Enc::Lit("imm2zr", 2)];
/// Used 4×.
const ENC_WRITE_NONE: &[Enc] = &[Enc::Lit("write", 1), Enc::Lit("none", 0)];
/// Used 4×.
const ENC_MIX5: &[Enc] = &[Enc::Family(Family::Regs, Count::Fixed(16), RegBase::ZERO)];
/// Used 3×.
const ENC_NO_INT8_INT4_DLFP16_BF16: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("int8", 1), Enc::Lit("int4", 2), Enc::Lit("dlfp16", 4), Enc::Lit("bf16", 5)];
/// Used 2×.
const ENC_MIX10: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::ZERO), Enc::Lit("n-link", 13), Enc::Lit("w-link", 14), Enc::Lit("0.0", 15)];
/// Used 2×.
const ENC_MIX9: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::ZERO), Enc::Lit("irf0", 12), Enc::Lit("IRF0", 12), Enc::Lit("irf1", 13), Enc::Lit("IRF1", 13), Enc::Lit("xrf", 14), Enc::Lit("no", 15)];
/// Used 2×.
const ENC_MIX11: &[Enc] = &[Enc::Lit("<1,4,3>", 0), Enc::Lit("<1,5,2>", 1), Enc::Family(Family::Jcrs, Count::NumJcrs, RegBase::ZERO)];
/// Used 2×.
const ENC_143_152: &[Enc] = &[Enc::Lit("<1,4,3>", 0), Enc::Lit("<1,5,2>", 1)];
/// Used 2×.
const ENC_FP16: &[Enc] = &[Enc::Lit("fp16", 0)];
/// Used 2×.
const ENC_NO_INT8_INT4: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("int8", 1), Enc::Lit("int4", 2)];
/// Used 2×.
const ENC_MIX8: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("nbrslice", 4), Enc::Lit("pe", 5), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9), Enc::Lit("icvtconst", 15)];
/// Used 2×.
const ENC_MIX12: &[Enc] = &[Enc::Lit("sendl3lu", 5), Enc::Lit("sendl3su", 9), Enc::Lit("recvl3lu", 6), Enc::Lit("recvlxsu", 34), Enc::Lit("recvlxsuboth", 162), Enc::Lit("sendrecvall", 167)];
/// Used 2×.
const ENC_MIX13: &[Enc] = &[Enc::Lit("sendl3lu", 5), Enc::Lit("sendl3su", 9), Enc::Lit("recvl3su", 10), Enc::Lit("sendlxlu", 17), Enc::Lit("sendlxluboth", 81), Enc::Lit("sendrecvall", 91)];
/// Used 2×.
const ENC_NO: &[Enc] = &[Enc::Lit("no", 0)];
/// Used 2×.
const ENC_LX_SFP: &[Enc] = &[Enc::Lit("lx", 1), Enc::Lit("sfp", 2)];
/// Used 2×.
const ENC_L0_LX_PE_PT: &[Enc] = &[Enc::Lit("l0", 0), Enc::Lit("lx", 1), Enc::Lit("pe", 2), Enc::Lit("pt", 3)];
/// Used 2×.
const ENC_64B_32B_16B: &[Enc] = &[Enc::Lit("64b", 0), Enc::Lit("32b", 1), Enc::Lit("16b", 2)];
/// Used 2×.
const ENC_SELF_NEIGHBOR_BOTH: &[Enc] = &[Enc::Lit("self", 1), Enc::Lit("neighbor", 2), Enc::Lit("both", 3)];
/// Used 2×.
const ENC_SCALE_DATA: &[Enc] = &[Enc::Lit("scale", 1), Enc::Lit("data", 0)];
/// Used 1×.
const ENC_REUSE_N_LINK_XRF_0_0: &[Enc] = &[Enc::Lit("reuse", 12), Enc::Lit("n-link", 13), Enc::Lit("xrf", 14), Enc::Lit("0.0", 15)];
/// Used 1×.
const ENC_N_LINK_W_LINK_1_0: &[Enc] = &[Enc::Lit("n-link", 1), Enc::Lit("w-link", 2), Enc::Lit("1.0", 3)];
/// Used 1×.
const ENC_NO_SRC1_SRC2: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("src1", 1), Enc::Lit("src2", 2)];
/// Used 1×.
const ENC_MIX21: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("sfp", 1), Enc::Lit("pt", 2), Enc::Lit("nbrslice", 3), Enc::Lit("lxlu", 4), Enc::Lit("ptint16", 5), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9)];
/// Used 1×.
const ENC_MIX22: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("sfp", 1), Enc::Lit("pt", 2), Enc::Lit("nbrslice", 3), Enc::Lit("lxlu", 4), Enc::Lit("reuse", 9), Enc::Lit("0.0", 10), Enc::Lit("1.0", 14)];
/// Used 1×.
const ENC_MIX19: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("sfp", 1), Enc::Lit("pt", 2), Enc::Lit("nbrslice", 3), Enc::Lit("lxlu", 4), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9), Enc::Lit("0.0", 10), Enc::Lit("muldiv2", 11), Enc::Lit("2.0", 12), Enc::Lit("3.0", 13)];
/// Used 1×.
const ENC_MIX20: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("sfp", 1), Enc::Lit("pt", 2), Enc::Lit("nbrslice", 3), Enc::Lit("lxlu", 4), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9), Enc::Lit("icvtconst", 15)];
/// Used 1×.
const ENC_MIX18: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("sfp", 1), Enc::Lit("pt", 2), Enc::Lit("nbrslice", 3), Enc::Lit("lxlu", 4), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9), Enc::Lit("0.0", 10), Enc::Lit("2.0", 12), Enc::Lit("3.0", 13)];
/// Used 1×.
const ENC_MIX14: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("nbrslice", 4), Enc::Lit("pe", 5), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9)];
/// Used 1×.
const ENC_MIX17: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("nbrslice", 4), Enc::Lit("pe", 5), Enc::Lit("reuse", 9), Enc::Lit("0.0", 10), Enc::Lit("1.0", 14)];
/// Used 1×.
const ENC_MIX16: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("nbrslice", 4), Enc::Lit("pe", 5), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9), Enc::Lit("0.0", 10), Enc::Lit("muldiv2", 11), Enc::Lit("2.0", 12), Enc::Lit("3.0", 13)];
/// Used 1×.
const ENC_MIX15: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::SIXTEEN), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("nbrslice", 4), Enc::Lit("pe", 5), Enc::Lit("nfwd", 8), Enc::Lit("reuse", 9), Enc::Lit("0.0", 10), Enc::Lit("2.0", 12), Enc::Lit("3.0", 13)];
/// Used 1×.
const ENC_0_0: &[Enc] = &[Enc::Lit("0.0", 10)];
/// Used 1×.
const ENC_RESULT: &[Enc] = &[Enc::Lit("result", 1)];
/// Used 1×.
const ENC_4B_8B_16B_32B_64B: &[Enc] = &[Enc::Lit("4b", 1), Enc::Lit("8b", 2), Enc::Lit("16b", 3), Enc::Lit("32b", 4), Enc::Lit("64b", 5)];
/// Used 1×.
const ENC_2B_4B: &[Enc] = &[Enc::Lit("2b", 1), Enc::Lit("4b", 2)];
/// Used 1×.
const ENC_XRF_0_0: &[Enc] = &[Enc::Lit("xrf", 14), Enc::Lit("0.0", 15)];
/// Used 1×.
const ENC_W_LINK_1_0: &[Enc] = &[Enc::Lit("w-link", 2), Enc::Lit("1.0", 3)];
/// Used 1×.
const ENC_MIX34: &[Enc] = &[Enc::Lit("no", 0), Enc::Lit("result", 3)];
/// Used 1×.
const ENC_MIX33: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("no", 15)];
/// Used 1×.
const ENC_MIX31: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("sfp", 8), Enc::Lit("sfpfp16tofp32l", 9), Enc::Lit("sfpfp16tofp32h", 10), Enc::Lit("sfpfp16tofp32fold", 11), Enc::Lit("ptint24tofp32fold", 12), Enc::Lit("ptfp24tofp32fold", 13), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX29: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("sfp", 8), Enc::Lit("sfpfp16tofp32l", 9), Enc::Lit("sfpfp16tofp32h", 10), Enc::Lit("sfpfp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX30: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("sfp", 8), Enc::Lit("sfpfp16tofp32l", 9), Enc::Lit("sfpfp16tofp32h", 10), Enc::Lit("sfpfp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("muldiv2", 17), Enc::Lit("1.0", 18), Enc::Lit("2.0", 19), Enc::Lit("3.0", 20), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX32: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("sfp", 8), Enc::Lit("sfpfp16tofp32l", 9), Enc::Lit("sfpfp16tofp32h", 10), Enc::Lit("sfpfp16tofp32fold", 11), Enc::Lit("ptint24tofp32fold", 12), Enc::Lit("ptfp24tofp32fold", 13), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25), Enc::Lit("icvtconst", 26)];
/// Used 1×.
const ENC_MIX28: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("sfp", 8), Enc::Lit("sfpfp16tofp32l", 9), Enc::Lit("sfpfp16tofp32h", 10), Enc::Lit("sfpfp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("2.0", 19), Enc::Lit("3.0", 20), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX24: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("pe", 8), Enc::Lit("pefp16tofp32l", 9), Enc::Lit("pefp16tofp32h", 10), Enc::Lit("pefp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX26: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("pe", 8), Enc::Lit("pefp16tofp32l", 9), Enc::Lit("pefp16tofp32h", 10), Enc::Lit("pefp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX27: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("pe", 8), Enc::Lit("pefp16tofp32l", 9), Enc::Lit("pefp16tofp32h", 10), Enc::Lit("pefp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("muldiv2", 17), Enc::Lit("1.0", 18), Enc::Lit("2.0", 19), Enc::Lit("3.0", 20), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_MIX25: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("pe", 8), Enc::Lit("pefp16tofp32l", 9), Enc::Lit("pefp16tofp32h", 10), Enc::Lit("pefp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25), Enc::Lit("icvtconst", 26)];
/// Used 1×.
const ENC_MIX23: &[Enc] = &[Enc::Family(Family::Regs, Count::NumRegs, RegBase::THIRTY_TWO), Enc::Lit("datafifo", 0), Enc::Lit("lxlu", 1), Enc::Lit("lxlufp16tofp32l", 2), Enc::Lit("lxlufp16tofp32h", 3), Enc::Lit("lxlufp16tofp32fold", 4), Enc::Lit("pe", 8), Enc::Lit("pefp16tofp32l", 9), Enc::Lit("pefp16tofp32h", 10), Enc::Lit("pefp16tofp32fold", 11), Enc::Lit("nbrslice", 15), Enc::Lit("0.0", 16), Enc::Lit("1.0", 18), Enc::Lit("2.0", 19), Enc::Lit("3.0", 20), Enc::Lit("nfwd", 24), Enc::Lit("reuse", 25)];
/// Used 1×.
const ENC_8B_16B_32B_64B_96B_128B_192B_256B: &[Enc] = &[Enc::Lit("8b", 0), Enc::Lit("16b", 1), Enc::Lit("32b", 2), Enc::Lit("64b", 3), Enc::Lit("96b", 4), Enc::Lit("128b", 5), Enc::Lit("192b", 6), Enc::Lit("256b", 7)];
/// Used 1×.
const ENC_8B_16B_32B: &[Enc] = &[Enc::Lit("8b", 1), Enc::Lit("16b", 2), Enc::Lit("32b", 3)];


/// EVERY NAMED ENCODING, so the set is walkable and none is unreachable.
///
/// ⛔⛔ THIS IS NOT BOOKKEEPING. The tables are still split per arch, so an encoding used only by
/// SEN1P5 is dead in an RCUDD1A build and the compiler says so — 27 of these on one arch, 22 on the
/// other. The alternatives were a `#[cfg]` on each (49 attributes stating what the tables already
/// state) or an `#[allow]`, which this crate does not permit. Naming them all here says the true
/// thing instead: an encoding belongs to the ISA, not to whichever arch happens to reference it.
///
/// ⭐ AND IT IS THE SAME PATTERN AS [`Comp::ALL`] AND `InstrType::ALL`, for the same reason those
/// exist — so a walk over the set cannot silently miss a member.
///
/// ⚠️ THE REAL FIX IS UPSTREAM OF THIS: give a field row its `min_arch`, as an opcode row already has,
/// and the two per-arch tables become one. Then nothing is conditionally dead and this array is a
/// census rather than a keep-alive.
pub const ALL_ENCODINGS: [&[Enc]; 92] = [
    ENC_BE,
    ENC_YES_NO,
    ENC_REGS,
    ENC_JCRS,
    ENC_LCCRS_JCRS,
    ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER,
    ENC_NO_SRC0_SRC2_RESULT,
    ENC_USEJCR_USELCCR,
    ENC_NO_YES,
    ENC_MIX1,
    ENC_NO_RESULT,
    ENC_FP16_FP32,
    ENC_GTRS,
    ENC_X1_X2_X4_X8,
    ENC_BURST,
    ENC_SELF_SFP_LXSU_PE,
    ENC_MVRS,
    ENC_MIX2,
    ENC_16B_8B_4B_2B,
    ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR,
    ENC_NO_CCW_CW_BOTH,
    ENC_MIX3,
    ENC_NO_SET_INCR,
    ENC_MIX6,
    ENC_X1_X2_X3_X4,
    ENC_SPLAT,
    ENC_SEND_RECV_SENDRECV,
    ENC_SFP_L0SU_PT,
    ENC_NO_16B_32B_48B_64B_80B_96B_112B,
    ENC_128B_2BSPLAT_16BZPAD_16BSPLAT,
    ENC_128B_2BSPLAT_4BSPLAT_16BSPLAT,
    ENC_NO_X1_X2_X4,
    ENC_64B_32B_16B_8B_4B_2B_1B,
    ENC_1B_2B_4B_8B_16B_32B_64B_128B,
    ENC_NO_2B_4B_6B,
    ENC_128B_16B_2B,
    ENC_ZERO_SFP_PE_LXLU,
    ENC_MIX4,
    ENC_MIX7,
    ENC_READ_WRITE,
    ENC_ZERO2LX_IMM2LX_ZR2LX,
    ENC_IMM2ZR,
    ENC_WRITE_NONE,
    ENC_MIX5,
    ENC_NO_INT8_INT4_DLFP16_BF16,
    ENC_MIX10,
    ENC_MIX9,
    ENC_MIX11,
    ENC_143_152,
    ENC_FP16,
    ENC_NO_INT8_INT4,
    ENC_MIX8,
    ENC_MIX12,
    ENC_MIX13,
    ENC_NO,
    ENC_LX_SFP,
    ENC_L0_LX_PE_PT,
    ENC_64B_32B_16B,
    ENC_SELF_NEIGHBOR_BOTH,
    ENC_SCALE_DATA,
    ENC_REUSE_N_LINK_XRF_0_0,
    ENC_N_LINK_W_LINK_1_0,
    ENC_NO_SRC1_SRC2,
    ENC_MIX21,
    ENC_MIX22,
    ENC_MIX19,
    ENC_MIX20,
    ENC_MIX18,
    ENC_MIX14,
    ENC_MIX17,
    ENC_MIX16,
    ENC_MIX15,
    ENC_0_0,
    ENC_RESULT,
    ENC_4B_8B_16B_32B_64B,
    ENC_2B_4B,
    ENC_XRF_0_0,
    ENC_W_LINK_1_0,
    ENC_MIX34,
    ENC_MIX33,
    ENC_MIX31,
    ENC_MIX29,
    ENC_MIX30,
    ENC_MIX32,
    ENC_MIX28,
    ENC_MIX24,
    ENC_MIX26,
    ENC_MIX27,
    ENC_MIX25,
    ENC_MIX23,
    ENC_8B_16B_32B_64B_96B_128B_192B_256B,
    ENC_8B_16B_32B,
];

/// ONE `defineField`, AS ONE LINE.
///
/// ⛔⛔ THE TABLE STATED 1,076 FIELDS AS SIX-LINE STRUCT LITERALS, AND ONLY 390 OF THEM WERE DISTINCT.
/// `defineField(type, name, bitPos, encodeList, immLen, sign)` is one call in `isa.cpp:163-221`; a row
/// spread over six lines cannot be read beside it, so the one cheap check on this table — comparing it
/// to the C++ by eye — was unavailable.
///
/// ⭐ FIVE ARMS, WHICH IS EXACTLY THE FIVE SHAPES THE TABLE USES: an operand with an encoding (861
/// rows), an immediate (95), a bare operand (85), a two-named position with an immediate (18), and an
/// immediate with an encoding (16). A sixth shape would not compile rather than being absorbed.
///
/// ⛔ `(a|b)` IS THE TWO-NAMED POSITION, NOT AN ALTERNATION. `defineField` gives `MVLOOPCNT`'s bit 10
/// the spelling `imm/pc_target`, and an exact-match lookup for `imm` missed it and encoded every loop
/// trip count as zero. The pair is one name with two spellings.
///
/// ⭐ EXPANDS TO EXACTLY THE LITERAL IT REPLACED — `tests/the_tables_are_unchanged.rs` asserts the
/// whole table byte for byte against a snapshot taken before these macros existed.
macro_rules! f {
    ($ty:ident, $name:ident, $bit:literal) => {
        DefField { ty: InstrType::$ty, name: FieldName::One(Operand::$name),
                   bit: WordBit::of($bit), imm: ImmSpec::NotAnImm, encode: &[] }
    };
    ($ty:ident, $name:ident, $bit:literal, $enc:ident) => {
        DefField { ty: InstrType::$ty, name: FieldName::One(Operand::$name),
                   bit: WordBit::of($bit), imm: ImmSpec::NotAnImm, encode: $enc }
    };
    ($ty:ident, $name:ident, $bit:literal, imm $bits:literal $sign:ident) => {
        DefField { ty: InstrType::$ty, name: FieldName::One(Operand::$name),
                   bit: WordBit::of($bit),
                   imm: ImmSpec::Imm { bits: ImmWidth::of($bits), sign: Sign::$sign },
                   encode: &[] }
    };
    ($ty:ident, $name:ident, $bit:literal, imm $bits:literal $sign:ident, $enc:ident) => {
        DefField { ty: InstrType::$ty, name: FieldName::One(Operand::$name),
                   bit: WordBit::of($bit),
                   imm: ImmSpec::Imm { bits: ImmWidth::of($bits), sign: Sign::$sign },
                   encode: $enc }
    };
    ($ty:ident, ($a:ident | $b:ident), $bit:literal, imm $bits:literal $sign:ident) => {
        DefField { ty: InstrType::$ty, name: FieldName::Either(Operand::$a, Operand::$b),
                   bit: WordBit::of($bit),
                   imm: ImmSpec::Imm { bits: ImmWidth::of($bits), sign: Sign::$sign },
                   encode: &[] }
    };
}

/// ONE `defineOpcode`, AS ONE LINE.
///
/// ⛔⛔ THE TABLE STATED 440 OPCODES AS FIVE-LINE STRUCT LITERALS — 2,200 lines for 440 facts of three
/// parts each. `defineOpcode(unit, op, instType, minArch)` is one line in `isa.cpp:150-159`, and a row
/// that cannot be read beside its C++ counterpart cannot be checked against it by eye, which is the
/// only cheap verification this table has.
///
/// ⭐ EXPANDS TO EXACTLY THE LITERAL IT REPLACED, which `tests/the_tables_are_unchanged.rs` asserts
/// byte for byte against a snapshot taken before this macro existed.
macro_rules! op {
    ($op:literal, $ty:ident, $gen:ident) => {
        DefOpcode {
            op: $op,
            ty: InstrType::$ty,
            min_arch: Gen::$gen,
        }
    };
}

/// One `defineOpcode`: which instruction TYPE an opcode is, and the earliest generation that has it.
#[derive(Debug, Clone, Copy)]
pub struct DefOpcode {
    /// The `Isa::InstOpCode` spelling.
    pub op: &'static str,
    pub ty: InstrType,
    /// `minArch`: `defineOpcode` defines the opcode only when `coreArch >= minArch`.
    pub min_arch: Gen,
}

/// HOW A FIELD'S VALUE IS READ, when it is a number rather than one of a named set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Signed,
    Unsigned,
    /// `MODULO_UNSIGNED` — a third reading, not a spelling of unsigned.
    ModuloUnsigned,
}

/// WHETHER A FIELD CARRIES AN IMMEDIATE, AND HOW WIDE IT IS.
///
/// ⛔ THE WIDTH IS ONLY PRESENT FOR AN IMMEDIATE, AND THE C++ ENFORCES BOTH DIRECTIONS: `defineField` asserts a
/// width is given exactly when the operand needs one (`immInfoNeeded`, `isa.cpp:190-205`). So this is two cases and
/// not an optional number — a field with a width that is not an immediate is a fault, not a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmSpec {
    NotAnImm,
    Imm { bits: ImmWidth, sign: Sign },
}

/// HOW MANY BITS AN IMMEDIATE FIELD HOLDS — `defineField`'s width argument (`isa.cpp:190-205`).
///
/// ⛔ A WIDTH IS NOT A POSITION AND NOT A VALUE. This one is also the MODULUS `imm_bits` masks by, so a position
/// reaching this slot would mask the value to the wrong number of bits rather than fail.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImmWidth(u8);

impl ImmWidth {
    /// ⛔ A ZERO-WIDTH IMMEDIATE HAS NO VALUE IT COULD CARRY, and `imm_bits`'s signed arm would shift by `-1`.
    /// `defineField` asserts a width is given exactly when the operand needs one, so zero is not a case it has.
    pub const fn of(bits: u8) -> Self {
        assert!(
            bits > 0,
            "an immediate field of zero bits carries no value, and `defineField` gives a width exactly when the \
             operand needs one (`immInfoNeeded`, `isa.cpp:190-205`)"
        );
        Self(bits)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// A REGISTER FAMILY a field can name, expanded by `tFieldEncodingMapFillRegs` (`isa.cpp:200-215`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `Regs` — the unit's own numbered registers, offset by `regOffset`.
    Regs,
    /// `JCRs` — jump condition registers.
    Jcrs,
    /// `LCCRs` — loop counters.
    Lccrs,
    /// `GTRs` — L3 group tag registers.
    Gtrs,
    /// `MVRs` — move registers.
    Mvrs,
    /// `BurstLen` — `e1`…`eN`, where `eN` encodes as 0 rather than N (`isa.cpp:211-216`).
    BurstLen,
}

/// HOW MANY MEMBERS A FAMILY HAS, or which named count supplies it.
///
/// ⛔ THESE ARE `regInfoPerUnit` LOOKUPS IN THE C++, so they are per-component facts and not constants of the ISA.
/// `numRegs` is the LRF's depth on the PE and the ARF's on the PT; `crate::regfile` is what answers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Count {
    /// A literal count written in the table.
    Fixed(u32),
    /// `numRegs` — the depth of the file this component's computes WRITE.
    ///
    /// ⛔ WHICH FILE THAT IS DEPENDS ON THE COMPONENT AND THE ARCH. `isa.cpp:233-237` picks
    /// `ptRegType = coreArch < RCUDD1A ? RegType::LRF : RegType::ARF` for the PT and reads `maxNum` from THAT — so
    /// on this target the PT's `numRegs` is its ARF's 4 (`sysdef.cpp:430-435`), not an LRF's 12. Every other component
    /// reads its LRF.
    NumRegs,
    /// `numJCRs`.
    NumJcrs,
    /// `numLCCRs` — 16, a constant in `initIsa` rather than a table lookup (`isa.cpp:228`).
    NumLccrs,
    /// `numMVRs`.
    NumMvrs,
    /// `allsyncTagL3LU` — 255 less the peers an L3LU cannot sync with (`isa.cpp:903`).
    AllSyncTagL3lu,
    /// `allsyncTagL3SU`.
    AllSyncTagL3su,
}

/// WHERE A REGISTER FAMILY'S FIRST MEMBER ENCODES — `regOffset` (`isa.cpp:162`, applied at `:125`).
///
/// ⛔⛔⛔ A FAMILY HAS A BASE AS WELL AS A COUNT, AND DROPPING THE BASE ALIASES `R0` ONTO A PORT.
/// `tFieldEncodingMapFillRegs` writes `field = i + regOffset` (`isa.cpp:125`), so `R<i>` on a unit whose
/// `regOffset` is 16 encodes as `i + 16` — never as `i`. The senulator reads the same rule from the other end:
/// `isReg(s)` is `s >= num_regs && s < num_regs * 2` and `reg = src - num_regs`
/// (`senulator/computeElement.cpp:2987`, `:3033`).
///
/// ⛔ SO A BASE OF ZERO ON AN SFP SOURCE PUTS `R0` AT 0, WHICH THAT FIELD SPELLS `datafifo` (`isa.cpp:592-594`) —
/// the cross-core psum FIFO. The unit then waits on every other core's SFP and never runs.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegBase(u32);

impl RegBase {
    /// `regOffset` is initialised to 0 (`isa.cpp:162`) and only the PE and SFP branches ever assign it, so every
    /// other component's families start here — as do `JCRs`, `LCCRs`, `GTRs` and `MVRs` on EVERY component,
    /// which `initIsa` fills with a literal `0` rather than with `regOffset` (`isa.cpp:201-210`).
    pub const ZERO: Self = Self(0);
    /// The PE's and SFP's own registers up to RCUDD1A — `regOffset = 16` (`isa.cpp:372`, `:566`).
    pub const SIXTEEN: Self = Self(16);
    /// The same on SEN1P5, where the file is twice as deep — `regOffset = 32` (`isa.cpp:446`, `:673`).
    pub const THIRTY_TWO: Self = Self(32);

    /// A base the generator read off a field's own `Regs` entry.
    ///
    /// ⛔ NOT A FREE CHOICE. `build.rs` mints these from `defineField`'s table and nothing else does, so a base
    /// here is one `isa.cpp` states for that field rather than a number a call site picked.
    pub const fn of(base: u32) -> Self {
        Self(base)
    }

    /// WHAT `R<index>` ENCODES AS — `i + regOffset` (`isa.cpp:125`).
    pub const fn encoding_of(self, index: u32) -> u32 {
        self.0 + index
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

// ⛔ THE BASE AND THE FILE DEPTH ARE THE SAME NUMBER ON A COMPUTE UNIT, AND THAT IS WHAT MAKES `isReg` WORK.
// `regOffset` is 16 where the LRF's `maxNum` is 16 (`sysdef.cpp:404`) and 32 where it is 32 (`:409`), which is
// exactly the window `s >= num_regs && s < num_regs * 2` tests. A base that disagreed with the depth would put
// half the file outside the range the senulator decodes as registers at all.
const _: () = assert!(
    RegBase::SIXTEEN.encoding_of(0) == 16 && RegBase::THIRTY_TWO.encoding_of(0) == 32,
    "a compute unit's R0 encodes at its file's depth, never at 0 — 0 in a source slot spells `datafifo`"
);

/// THE ENCODED VALUE OF ONE FIELD — what goes into the field, before it is shifted into position.
///
/// ⛔ NOT A `u64`, AND NOT AN INDEX, A COUNT OR A BIT POSITION. A field's VALUE, the POSITION it is shifted to
/// and the INDEX of a register that may be in it are three quantities that must not be assignable to each
/// other's slots. The value is masked to its field by whatever produced it; this says only that it IS one.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldValue(u64);

impl FieldValue {
    pub const fn of(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// WHERE A FIELD SITS IN THE ENCODED WORD — `defineField`'s `bitPos` (`isa.cpp:167`).
///
/// ⛔ A POSITION IS NOT A WIDTH AND NOT A VALUE — the bit a field starts at, how many bits it holds and the
/// number that goes into it are three quantities, and only [`ImmWidth`] is ever a width.
///
/// ⛔⛔ AND A POSITION DOES NOT DECIDE WHETHER THE WORD CARRIES THE FIELD — the field's NAME does. See
/// [`crate::operand::Operand::is_virtual`]: `defineField` reads
/// `if (isFieldVirtual(name)) DT_CHECK(bitPos >= 100)` (`isa.cpp:179-181`), so `>= 100` is a CONSEQUENCE asserted
/// of a virtual field, not the definition of one. Deriving virtuality from the position would invert the C++.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WordBit(u16);

impl WordBit {
    /// The position at or above which `defineField` asserts a VIRTUAL field's `bitPos` lies (`isa.cpp:180`).
    ///
    /// ⛔ NOT "above 31" AND NOT "above 63". The SFP's real fields already reach bit 44, so a width-based reading
    /// would call half of them virtual; the C++ tests this exact number.
    pub const VIRTUAL_FLOOR: u16 = 100;

    pub const fn of(bit: u16) -> Self {
        Self(bit)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// ONE ENCODABLE VALUE OF A FIELD.
#[derive(Debug, Clone, Copy)]
pub enum Enc {
    /// A named literal: `{"sendrecv", 3}`.
    Lit(&'static str, i64),
    /// A named literal whose value is one of the counts above rather than a number.
    LitSym(&'static str, Count),
    /// A whole register family, expanded into one literal per register — `count` members starting at `base`.
    ///
    /// ⛔ THE BASE IS PART OF THE FAMILY. `tFieldEncodingMapFillRegs` takes `regOffset` as an argument on EVERY
    /// call (`isa.cpp:200-210`); only the `Regs` call site is ever passed a non-zero one, and stating that zero
    /// rather than implying it is what keeps the two facts from being confused.
    Family(Family, Count, RegBase),
}

/// ONE `defineField`.
///
/// ⛔ A NAME MAY BE TWO OPERANDS SEPARATED BY `/`, and that is not a spelling choice. `defineField` splits on it
/// (`isa.cpp:167-176`) because ONE field position holds either operand — `"imm/pc_target"` is an immediate or a
/// branch target in the same bits — so the two share a position and only one of them is present in any instruction.
/// WHICH `Isa::InstOperand`(s) one field position answers to.
///
/// ⭐ TWO VARIANTS BECAUSE THE C++ HAS TWO CASES, not because a pair is convenient: `defineField`'s name argument
/// is one spelling or two joined by `/` (`isa.cpp:163-221`), and only `imm/pc_target` is ever the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldName {
    One(Operand),
    /// One position, two spellings — `MVLOOPCNT`'s `imm/pc_target`.
    Either(Operand, Operand),
}

impl FieldName {
    /// THE SPELLING(S) THIS POSITION ANSWERS TO — one, or the two a `/` form joins.
    ///
    /// ⛔ A SLICE, NOT A SINGLE NAME, and that is the whole reason this type exists. `defineField` gives
    /// `MVLOOPCNT`'s bit 10 the name `"imm/pc_target"`, so an exact comparison for `imm` finds nothing there —
    /// which is how every loop's TRIP COUNT once encoded as zero. Anything asking "is this field X" must ask this.
    pub const fn spellings(self) -> &'static [&'static str] {
        match self {
            Self::One(name) => name.spelling_slice(),
            // ⭐ THE ONLY PAIR IN THE ISA is `imm/pc_target` (18 of 1075 definitions), so the pair's slice is that
            // one case rather than a general two-element builder — a `const fn` cannot allocate.
            Self::Either(_, _) => &["imm", "pc_target"],
        }
    }

    /// WHETHER THE ENCODED WORD CARRIES THIS POSITION — `isFieldVirtual` over a `/` pair (`isa.hpp:319-323`),
    /// which answers yes if EITHER half is virtual.
    pub const fn is_virtual(self) -> bool {
        match self {
            Self::One(name) => name.is_virtual(),
            Self::Either(first, second) => first.is_virtual() || second.is_virtual(),
        }
    }

    /// WHETHER THIS POSITION ANSWERS TO `wanted` — the test `defineField`'s `/` form exists for.
    pub const fn answers_to(self, wanted: Operand) -> bool {
        match self {
            Self::One(name) => name as u8 == wanted as u8,
            Self::Either(first, second) => {
                first as u8 == wanted as u8 || second as u8 == wanted as u8
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DefField {
    pub ty: InstrType,
    /// WHICH FIELD THIS IS — one [`Operand`], or the TWO a single position answers to.
    ///
    /// ⛔⛔ A POSITION MAY HAVE TWO NAMES AND AN EXACT COMPARISON MISSES ONE. `defineField` gives
    /// `MVLOOPCNT`'s bit 10 the spelling `"imm/pc_target"` — 18 of the 1075 definitions — so a lookup for `imm`
    /// that compared the whole spelling found nothing there, and every loop's TRIP COUNT encoded as zero. The pair
    /// is a variant so the second name cannot be dropped by a comparison.
    pub name: FieldName,
    /// The bit the field starts at.
    ///
    /// ⛔ A POSITION AT OR ABOVE 100 IS A VIRTUAL FIELD, which `defineField` asserts (`isa.cpp:177-179`): it is not
    /// in the encoded word at all, and the SFP's real fields already reach bit 44, so the word is wider than 32 bits
    /// and a "> 31 means virtual" reading would be wrong.
    pub bit: WordBit,
    pub imm: ImmSpec,
    pub encode: &'static [Enc],
}

#[cfg(feature = "arch-rcudd1a")]
const OPCODES_PT: &[DefOpcode] = &[
    op!("FMA", T20, Mpw2),
    op!("FMA4", T20, Sen1p5),
    op!("FMA8", T20, Mpw2),
    op!("IMA8", T20, Mpw2),
    op!("IMA4", T20, Mpw2),
    op!("JADD", T12, Mpw3),
    op!("JCMP", T13, Mpw3),
    op!("JCRSWAP", T14, Mpw4),
    op!("JIMMCOPY", T12, Mpw3),
    op!("JSUB", T12, Mpw3),
    op!("MVLOOPCNT", T10, Mpw2),
    op!("NOP", T11, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T13, Mpw4),
    op!("XRFACCESS", T15, Mpw4),
    op!("INCRMASK", T16, Mpw3),
    op!("SETMASK", T17, Mpw3),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_PT: &[DefField] = &[
    f!(T10, (Imm|PcTarget), 6, imm 16 Unsigned),
    f!(T10, Src0, 22, ENC_JCRS),
    f!(T10, DynLoop, 26, ENC_YES_NO),
    f!(T10, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Imm, 6, imm 16 Signed),
    f!(T12, Src0, 22, ENC_LCCRS_JCRS),
    f!(T12, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T12, JcrTarget, 27, ENC_JCRS),
    f!(T12, Be, 31, ENC_BE),
    f!(T13, Src1, 6, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T13, PcTarget, 14),
    f!(T13, Src0, 22, ENC_LCCRS_JCRS),
    f!(T13, Isimm, 26, ENC_YES_NO),
    f!(T13, Mode, 27, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T13, Usejcr, 31, ENC_YES_NO),
    f!(T14, Src1, 6, ENC_JCRS),
    f!(T14, Src0, 22, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T15, RdptrImm, 6, imm 6 ModuloUnsigned),
    f!(T15, RdptrUpd, 12, ENC_NO_SET_INCR),
    f!(T15, WrptrImm, 14, imm 6 ModuloUnsigned),
    f!(T15, WrptrUpd, 20, ENC_NO_SET_INCR),
    f!(T15, Be, 31, ENC_BE),
    f!(T16, Be, 31, ENC_BE),
    f!(T17, Imm, 6, imm 3 Unsigned),
    f!(T17, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REUSE_N_LINK_XRF_0_0),
    f!(T20, Src1, 10, ENC_MIX10),
    f!(T20, Src2, 14, ENC_N_LINK_W_LINK_1_0),
    f!(T20, Tgte, 16, ENC_NO_SRC1_SRC2),
    f!(T20, Tgts, 18, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtrf, 20, ENC_MIX9),
    f!(T20, Unrlfldsrc0, 24, ENC_YES_NO),
    f!(T20, Unrlfldsrc1, 25, ENC_YES_NO),
    f!(T20, Unrlfldtgt, 26, ENC_YES_NO),
    f!(T20, Unroll, 27, ENC_X1_X2_X4_X8),
    f!(T20, Fma8ctrlsrc0, 29, ENC_MIX11),
    f!(T20, Fma8ctrlsrc2, 30, ENC_143_152),
    f!(T20, Be, 31, ENC_BE),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_PE: &[DefOpcode] = &[
    op!("EE", T30, Mpw2),
    op!("FCMP", T30, Mpw2),
    op!("FEST", T30, Mpw2),
    op!("FMA", T20, Mpw2),
    op!("FMINMAX", T30, Mpw2),
    op!("FMUL", T20, Mpw2),
    op!("FNMS", T20, Mpw2),
    op!("GCVT", T30, Mpw2),
    op!("FCVT", T30, Sen1p5),
    op!("ICVT", T30, Mpw2),
    op!("IME", T30, Mpw2),
    op!("IMMCOPY", T13, Mpw2),
    op!("JADD", T41, Mpw2),
    op!("JCMP", T43, Mpw2),
    op!("JIMMCOPY", T41, Mpw2),
    op!("JSUB", T41, Mpw2),
    op!("LOGICAL", T30, Mpw2),
    op!("MERGE", T30, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("PACK", T30, Mpw2),
    op!("PERMUTE", T30, Sen1p5),
    op!("REDUCE", T118, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SELECT", T30, Mpw2),
    op!("SHR", T30, Mpw2),
    op!("SJCMP", T43, Mpw4),
    op!("SPLAT", T30, Mpw2),
    op!("LOAD_LFSR", T30, Sen1p5),
    op!("COPY_LFSR", T30, Sen1p5),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_PE: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 11, imm 16 Unsigned),
    f!(T11, DynLoop, 28, ENC_YES_NO),
    f!(T11, Be, 35, ENC_BE),
    f!(T12, Be, 35, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T118, RsmSrc0, 11, ENC_MIX6),
    f!(T118, RsmSrc1, 16),
    f!(T118, RsmUnroll, 21, ENC_X1_X2_X3_X4),
    f!(T118, RsmTgtrf, 24, ENC_MIX6),
    f!(T118, Be, 35, ENC_BE),
    f!(T13, Imm, 11, imm 16 Unsigned),
    f!(T13, Replica, 29),
    f!(T13, Tgtrf, 30, ENC_MIX2),
    f!(T13, Be, 35, ENC_BE),
    f!(T13, Mask, 36),
    f!(T20, Src0, 6, ENC_MIX21),
    f!(T20, Src1, 11, ENC_MIX22),
    f!(T20, Src2, 16, ENC_MIX19),
    f!(T20, Tgtpe, 21, ENC_NO_RESULT),
    f!(T20, Tgtlx, 22, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtsfp, 24, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtrf, 30, ENC_MIX2),
    f!(T20, Be, 35, ENC_BE),
    f!(T20, Mask, 36),
    f!(T20, Mode, 44, ENC_FP16),
    f!(T20, Unroll, 45, ENC_X1_X2_X4_X8),
    f!(T20, Unrlfldsrc0, 47, ENC_YES_NO),
    f!(T20, Unrlfldsrc1, 48, ENC_YES_NO),
    f!(T20, Unrlfldsrc2, 49, ENC_YES_NO),
    f!(T20, Unrlfldtgt, 50, ENC_YES_NO),
    f!(T20, Fpuop, 51, ENC_NO_INT8_INT4),
    f!(T20, Reluop, 53, ENC_NO_YES),
    f!(T30, Src0, 6, ENC_MIX20),
    f!(T30, Imm, 11, imm 5 Unsigned),
    f!(T30, Src2, 16, ENC_MIX18),
    f!(T30, Tgtpe, 21, ENC_NO_RESULT),
    f!(T30, Tgtlx, 22, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtsfp, 24, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtrf, 30, ENC_MIX2),
    f!(T30, Be, 35, ENC_BE),
    f!(T30, Mask, 36),
    f!(T30, Mode, 44, ENC_FP16),
    f!(T30, Unroll, 45, ENC_X1_X2_X4_X8),
    f!(T30, Unrlfldsrc0, 47, ENC_YES_NO),
    f!(T30, Unrlfldsrc1, 48, ENC_YES_NO),
    f!(T30, Unrlfldsrc2, 49, ENC_YES_NO),
    f!(T30, Unrlfldtgt, 50, ENC_YES_NO),
    f!(T30, Reluop, 53, ENC_NO_YES),
    f!(T41, Src0, 6, ENC_LCCRS_JCRS),
    f!(T41, Imm, 11, imm 16 Signed),
    f!(T41, JcrSelect, 28, ENC_USEJCR_USELCCR),
    f!(T41, Be, 35, ENC_BE),
    f!(T41, JcrTarget, 36, ENC_JCRS),
    f!(T43, Src0, 6, ENC_LCCRS_JCRS),
    f!(T43, Src1, 11, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T43, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T43, Isimm, 28, ENC_YES_NO),
    f!(T43, Usejcr, 35, ENC_YES_NO),
    f!(T43, PcTarget, 36),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_SFP: &[DefOpcode] = &[
    op!("EE", T30, Mpw2),
    op!("FCMP", T30, Mpw2),
    op!("FEST", T30, Mpw2),
    op!("FMA", T20, Mpw2),
    op!("FMINMAX", T30, Mpw2),
    op!("FMUL", T20, Mpw2),
    op!("FNMS", T20, Mpw2),
    op!("GCVT", T30, Mpw2),
    op!("FCVT", T30, Mpw2),
    op!("ICVT", T30, Mpw2),
    op!("IME", T30, Mpw2),
    op!("IMMCOPY", T13, Mpw2),
    op!("JADD", T41, Mpw2),
    op!("JCMP", T43, Mpw2),
    op!("JIMMCOPY", T41, Mpw2),
    op!("JSUB", T41, Mpw2),
    op!("LOGICAL", T30, Mpw2),
    op!("MERGE", T30, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("PACK", T30, Mpw2),
    op!("PERMUTE", T30, Mpw2),
    op!("REDUCE", T118, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SELECT", T30, Mpw2),
    op!("SHR", T31, Mpw2),
    op!("SJCMP", T43, Mpw4),
    op!("SPLAT", T30, Mpw2),
    op!("SETDEST", T14, Rcudd1a),
    op!("LOAD_LFSR", T30, Mpw4),
    op!("COPY_LFSR", T30, Mpw4),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_SFP: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 11, imm 16 Unsigned),
    f!(T11, DynLoop, 28, ENC_YES_NO),
    f!(T11, Be, 35, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T12, Be, 35, ENC_BE),
    f!(T13, Imm, 11, imm 16 Unsigned),
    f!(T13, Replica, 29),
    f!(T13, Tgtrf, 30, ENC_MIX2),
    f!(T13, Be, 35, ENC_BE),
    f!(T13, Mask, 36),
    f!(T13, Mode, 44, ENC_FP16_FP32),
    f!(T14, Imm, 11, imm 32 Unsigned),
    f!(T118, RsmSrc0, 11, ENC_MIX6),
    f!(T118, RsmSrc1, 16),
    f!(T118, RsmUnroll, 21, ENC_X1_X2_X3_X4),
    f!(T118, RsmTgtrf, 24, ENC_MIX6),
    f!(T118, Be, 35, ENC_BE),
    f!(T118, Mode, 44, ENC_FP16_FP32),
    f!(T20, Src0, 6, ENC_MIX14),
    f!(T20, Src1, 11, ENC_MIX17),
    f!(T20, Src2, 16, ENC_MIX16),
    f!(T20, Tgtsfp, 21, ENC_NO_RESULT),
    f!(T20, Tgtpe, 22, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtlx, 24, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtl0, 26, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtpt, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtrf, 30, ENC_MIX2),
    f!(T20, Be, 35, ENC_BE),
    f!(T20, Mask, 36),
    f!(T20, Mode, 44, ENC_FP16_FP32),
    f!(T20, Unroll, 45, ENC_X1_X2_X4_X8),
    f!(T20, Unrlfldsrc0, 47, ENC_YES_NO),
    f!(T20, Unrlfldsrc1, 48, ENC_YES_NO),
    f!(T20, Unrlfldsrc2, 49, ENC_YES_NO),
    f!(T20, Unrlfldtgt, 50, ENC_YES_NO),
    f!(T20, Fpuop, 51, ENC_NO_INT8_INT4),
    f!(T20, Reluop, 53, ENC_NO_YES),
    f!(T20, Tgtdatafifo, 55, ENC_NO_RESULT),
    f!(T30, Src0, 6, ENC_MIX8),
    f!(T30, Imm, 11, imm 5 Unsigned),
    f!(T30, Src2, 16, ENC_MIX15),
    f!(T30, Tgtsfp, 21, ENC_NO_RESULT),
    f!(T30, Tgtpe, 22, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtlx, 24, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtl0, 26, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtpt, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtrf, 30, ENC_MIX2),
    f!(T30, Be, 35, ENC_BE),
    f!(T30, Mask, 36),
    f!(T30, Mode, 44, ENC_FP16_FP32),
    f!(T30, Unroll, 45, ENC_X1_X2_X4_X8),
    f!(T30, Unrlfldsrc0, 47, ENC_YES_NO),
    f!(T30, Unrlfldsrc1, 48, ENC_YES_NO),
    f!(T30, Unrlfldsrc2, 49, ENC_YES_NO),
    f!(T30, Unrlfldtgt, 50, ENC_YES_NO),
    f!(T30, Reluop, 53, ENC_NO_YES),
    f!(T30, Tgtdatafifo, 55, ENC_NO_RESULT),
    f!(T31, Src0, 6, ENC_MIX8),
    f!(T31, Imm, 11, imm 6 Signed),
    f!(T31, Src2, 16, ENC_0_0),
    f!(T31, Tgtsfp, 21, ENC_RESULT),
    f!(T31, Tgtpe, 22, ENC_NO_SRC0_SRC2_RESULT),
    f!(T31, Tgtlx, 24, ENC_NO_SRC0_SRC2_RESULT),
    f!(T31, Tgtl0, 26, ENC_NO_SRC0_SRC2_RESULT),
    f!(T31, Tgtpt, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T31, Tgtrf, 30, ENC_MIX2),
    f!(T31, Be, 35, ENC_BE),
    f!(T31, Mask, 36),
    f!(T31, Mode, 44, ENC_FP16_FP32),
    f!(T31, Unroll, 45, ENC_X1_X2_X4_X8),
    f!(T31, Unrlfldsrc0, 47, ENC_YES_NO),
    f!(T31, Unrlfldsrc1, 48, ENC_YES_NO),
    f!(T31, Unrlfldsrc2, 49, ENC_YES_NO),
    f!(T31, Unrlfldtgt, 50, ENC_YES_NO),
    f!(T31, Reluop, 53, ENC_NO_YES),
    f!(T31, Tgtdatafifo, 55, ENC_NO_RESULT),
    f!(T41, Src0, 6, ENC_LCCRS_JCRS),
    f!(T41, Imm, 11, imm 16 Signed),
    f!(T41, JcrSelect, 28, ENC_USEJCR_USELCCR),
    f!(T41, Be, 35, ENC_BE),
    f!(T41, JcrTarget, 36, ENC_JCRS),
    f!(T43, Src0, 6, ENC_LCCRS_JCRS),
    f!(T43, Src1, 11, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T43, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T43, Isimm, 28, ENC_YES_NO),
    f!(T43, Usejcr, 35, ENC_YES_NO),
    f!(T43, PcTarget, 36),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_L0LU: &[DefOpcode] = &[
    op!("IMMCOPY", T16, Mpw2),
    op!("LDST", T20, Mpw2),
    op!("LDSTI", T11, Mpw2),
    op!("LDSTIU", T11, Mpw2),
    op!("LDSTU", T20, Mpw2),
    op!("LRFREGCOPY", T21, Mpw2),
    op!("MODLRFREG", T21, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T30, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("MODLRFIMM", T16, Mpw2),
    op!("MVLOOPCNT", T10, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T30, Mpw4),
    op!("SYNC", T13, Mpw2),
    op!("TILEADV", T12, Sen1p5),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_L0LU: &[DefField] = &[
    f!(T10, Src0, 6, ENC_JCRS),
    f!(T10, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T10, DynLoop, 26, ENC_YES_NO),
    f!(T10, Be, 31, ENC_BE),
    f!(T11, Src0, 6, ENC_REGS),
    f!(T11, Imm, 10, imm 10 Signed),
    f!(T11, Ldtype, 26, ENC_16B_8B_4B_2B),
    f!(T11, Splat, 28, ENC_SPLAT),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_SEND_RECV_SENDRECV),
    f!(T13, Implicit, 7, ENC_NO_YES),
    f!(T13, Tilesize, 22, imm 7 Unsigned),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T16, Src0, 6, ENC_REGS),
    f!(T16, Imm, 10, imm 10 ModuloUnsigned),
    f!(T16, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Src1, 10, ENC_REGS),
    f!(T20, Group, 14),
    f!(T20, Burst, 16, ENC_BURST),
    f!(T20, Burstsize, 17, imm 6 Unsigned),
    f!(T20, Chunkstride, 23, ENC_4B_8B_16B_32B_64B),
    f!(T20, Ldtype, 26, ENC_16B_8B_4B_2B),
    f!(T20, Chunksize, 29, ENC_2B_4B),
    f!(T20, Splat, 28, ENC_SPLAT),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Src1, 10, ENC_REGS),
    f!(T21, Be, 31, ENC_BE),
    f!(T30, Src0, 6, ENC_LCCRS_JCRS),
    f!(T30, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T30, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T30, PcTarget, 22),
    f!(T30, Isimm, 30, ENC_YES_NO),
    f!(T30, Usejcr, 31, ENC_YES_NO),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_L0SU: &[DefOpcode] = &[
    op!("IMMCOPY", T16, Mpw2),
    op!("LDST", T20, Mpw2),
    op!("LDSTI", T11, Mpw2),
    op!("LDSTIU", T11, Mpw2),
    op!("LDSTU", T20, Mpw2),
    op!("LRFREGCOPY", T21, Mpw2),
    op!("MODLRFREG", T21, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T30, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("MODLRFIMM", T16, Mpw2),
    op!("MVLOOPCNT", T10, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T30, Mpw4),
    op!("SYNC", T13, Mpw2),
    op!("TILEADV", T12, Sen1p5),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_L0SU: &[DefField] = &[
    f!(T10, Src0, 6, ENC_JCRS),
    f!(T10, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T10, DynLoop, 26, ENC_YES_NO),
    f!(T10, Be, 31, ENC_BE),
    f!(T11, Src0, 6, ENC_REGS),
    f!(T11, Imm, 10, imm 10 Signed),
    f!(T11, Permute, 21),
    f!(T11, Subwordlen, 22),
    f!(T11, Stride, 24),
    f!(T11, Coalesce, 26),
    f!(T11, Src2, 27, ENC_REGS),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_SEND_RECV_SENDRECV),
    f!(T13, Implicit, 7, ENC_NO_YES),
    f!(T13, Tilesize, 22, imm 7 Unsigned),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T16, Src0, 6, ENC_REGS),
    f!(T16, Imm, 10, imm 10 ModuloUnsigned),
    f!(T16, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Src1, 10, ENC_REGS),
    f!(T20, Group, 14),
    f!(T20, Burst, 16, ENC_BURST),
    f!(T20, Burstsize, 17, imm 6 Unsigned),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Src1, 10, ENC_REGS),
    f!(T21, Be, 31, ENC_BE),
    f!(T30, Src0, 6, ENC_LCCRS_JCRS),
    f!(T30, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T30, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T30, PcTarget, 22),
    f!(T30, Isimm, 30, ENC_YES_NO),
    f!(T30, Usejcr, 31, ENC_YES_NO),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_LXLU: &[DefOpcode] = &[
    op!("IMMCOPY", T111, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T40, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("LDST", T32, Mpw2),
    op!("LDSTI", T20, Mpw2),
    op!("LDSTIU", T20, Mpw2),
    op!("LDSTU", T32, Mpw2),
    op!("LDCVTI", T21, Sen1p5),
    op!("LDCVTIU", T21, Sen1p5),
    op!("SAMV", T50, Mpw3),
    op!("SETDSTMASK", T15, Rcudd1a),
    op!("SPMV", T60, Rcudd1a),
    op!("LRFREGCOPY", T31, Mpw2),
    op!("MODLRFIMM", T111, Mpw2),
    op!("MODLRFREG", T31, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T40, Mpw4),
    op!("SUBLRFIMM", T111, Mpw2),
    op!("SYNC", T13, Mpw2),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_LXLU: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T11, DynLoop, 26, ENC_YES_NO),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_MIX12),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T15, Mode, 19, ENC_SFP_L0SU_PT),
    f!(T15, Be, 31, ENC_BE),
    f!(T111, Src0, 6, ENC_REGS),
    f!(T111, Lrfimm, 10, imm 21 ModuloUnsigned),
    f!(T111, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Imm, 10, imm 14 Signed),
    f!(T20, Rottype, 24, ENC_NO_16B_32B_48B_64B_80B_96B_112B),
    f!(T20, Ldtype, 27, ENC_128B_2BSPLAT_16BZPAD_16BSPLAT),
    f!(T20, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Imm, 10, imm 14 Signed),
    f!(T21, Elemidx, 24),
    f!(T21, Scaleidx, 26),
    f!(T21, Ldtype, 27, ENC_128B_2BSPLAT_4BSPLAT_16BSPLAT),
    f!(T21, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T21, Be, 31, ENC_BE),
    f!(T31, Src0, 6, ENC_REGS),
    f!(T31, Src1, 10, ENC_REGS),
    f!(T31, Be, 31, ENC_BE),
    f!(T32, Src0, 6, ENC_REGS),
    f!(T32, Src1, 10, ENC_REGS),
    f!(T32, Group, 14, ENC_NO_X1_X2_X4),
    f!(T32, Burst, 16, ENC_BURST),
    f!(T32, Burstsize, 17, imm 6 Unsigned),
    f!(T32, Rottype, 24, ENC_NO_16B_32B_48B_64B_80B_96B_112B),
    f!(T32, Ldtype, 27, ENC_128B_2BSPLAT_16BZPAD_16BSPLAT),
    f!(T32, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T32, Be, 31, ENC_BE),
    f!(T40, Src0, 6, ENC_LCCRS_JCRS),
    f!(T40, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T40, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T40, PcTarget, 22),
    f!(T40, Isimm, 30, ENC_YES_NO),
    f!(T40, Usejcr, 31, ENC_YES_NO),
    f!(T50, Maskall, 10, ENC_YES_NO),
    f!(T50, Sliceidxsl, 11),
    f!(T50, Numvalidentry, 14),
    f!(T50, Mvridx, 21, ENC_MVRS),
    f!(T50, Precision, 24, ENC_16B_8B_4B_2B),
    f!(T50, Xslinner, 26, ENC_YES_NO),
    f!(T50, Wsllen, 27, ENC_64B_32B_16B_8B_4B_2B_1B),
    f!(T50, Be, 31, ENC_BE),
    f!(T60, Mvridx, 6, ENC_MVRS),
    f!(T60, Subslicesize, 9, ENC_1B_2B_4B_8B_16B_32B_64B_128B),
    f!(T60, Startbit, 12),
    f!(T60, Endbit, 20),
    f!(T60, Byteshift, 28, ENC_NO_2B_4B_6B),
    f!(T60, Be, 31, ENC_BE),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_LXSU: &[DefOpcode] = &[
    op!("IMMCOPY", T111, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T40, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("LDST", T32, Mpw2),
    op!("LDSTI", T20, Mpw2),
    op!("LDSTIU", T20, Mpw2),
    op!("LDSTU", T32, Mpw2),
    op!("LRFCOPY", T20, Mpw2),
    op!("LRFREGCOPY", T31, Mpw2),
    op!("MODLRFIMM", T111, Mpw2),
    op!("MODLRFREG", T31, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T40, Mpw4),
    op!("SUBLRFIMM", T111, Mpw2),
    op!("SYNC", T13, Mpw2),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_LXSU: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T11, DynLoop, 26, ENC_YES_NO),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_MIX13),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T15, Mode, 19, ENC_SFP_L0SU_PT),
    f!(T15, Be, 31, ENC_BE),
    f!(T111, Src0, 6, ENC_REGS),
    f!(T111, Lrfimm, 10, imm 21 ModuloUnsigned),
    f!(T111, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Imm, 10, imm 14 Signed),
    f!(T20, Hwsel, 24, ENC_NO),
    f!(T20, Sttype, 27, ENC_128B_16B_2B),
    f!(T20, Producertag, 29, ENC_ZERO_SFP_PE_LXLU),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Imm, 10, imm 14 Signed),
    f!(T21, Elemidx, 24),
    f!(T21, Scaleidx, 26),
    f!(T21, Ldtype, 27, ENC_128B_2BSPLAT_4BSPLAT_16BSPLAT),
    f!(T21, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T21, Be, 31, ENC_BE),
    f!(T31, Src0, 6, ENC_REGS),
    f!(T31, Src1, 10, ENC_REGS),
    f!(T31, Be, 31, ENC_BE),
    f!(T32, Src0, 6, ENC_REGS),
    f!(T32, Src1, 10, ENC_REGS),
    f!(T32, Group, 14, ENC_NO_X1_X2_X4),
    f!(T32, Burst, 16, ENC_BURST),
    f!(T32, Burstsize, 17, imm 6 Unsigned),
    f!(T32, Sttype, 27, ENC_128B_16B_2B),
    f!(T32, Producertag, 29, ENC_ZERO_SFP_PE_LXLU),
    f!(T32, Be, 31, ENC_BE),
    f!(T40, Src0, 6, ENC_LCCRS_JCRS),
    f!(T40, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T40, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T40, PcTarget, 22),
    f!(T40, Isimm, 30, ENC_YES_NO),
    f!(T40, Usejcr, 31, ENC_YES_NO),
    f!(T50, Maskall, 10, ENC_YES_NO),
    f!(T50, Sliceidxsl, 11),
    f!(T50, Numvalidentry, 14),
    f!(T50, Mvridx, 21, ENC_MVRS),
    f!(T50, Precision, 24, ENC_16B_8B_4B_2B),
    f!(T50, Xslinner, 26, ENC_YES_NO),
    f!(T50, Wsllen, 27, ENC_64B_32B_16B_8B_4B_2B_1B),
    f!(T50, Be, 31, ENC_BE),
    f!(T60, Mvridx, 6, ENC_MVRS),
    f!(T60, Subslicesize, 9, ENC_1B_2B_4B_8B_16B_32B_64B_128B),
    f!(T60, Startbit, 12),
    f!(T60, Endbit, 20),
    f!(T60, Byteshift, 28, ENC_NO_2B_4B_6B),
    f!(T60, Be, 31, ENC_BE),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_L3LU: &[DefOpcode] = &[
    op!("ADDEARIMM", T1110, Mpw2),
    op!("ADDLARIMM", T1100, Mpw2),
    op!("EARIMM", T1112, Mpw2),
    op!("EARREGCOPY", T12, Mpw2),
    op!("GTRIMM", T1120, Mpw2),
    op!("JADD", T1020, Mpw2),
    op!("JCMP", T1030, Mpw2),
    op!("JCMPI", T1040, Mpw2),
    op!("JIMMCOPY", T1020, Mpw2),
    op!("JSUB", T1020, Mpw2),
    op!("LARIMM", T1100, Mpw2),
    op!("LARREGCOPY", T12, Mpw2),
    op!("MODLARREG", T12, Mpw2),
    op!("MODEARREG", T12, Mpw2),
    op!("MVLOOPCNT", T1010, Mpw2),
    op!("NOP", T1000, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T1030, Mpw4),
    op!("SUBLARIMM", T1100, Mpw2),
    op!("SYNC", T13, Mpw2),
    op!("LD", T21, Mpw2),
    op!("LDG", T21, Mpw2),
    op!("LDGM", T22, Mpw2),
    op!("LDGMU", T22, Mpw2),
    op!("LDGU", T21, Mpw2),
    op!("LDIGM", T22, Mpw4),
    op!("LDIGMU", T22, Mpw4),
    op!("LDIM", T22, Mpw4),
    op!("LDIMU", T22, Mpw4),
    op!("LDM", T22, Mpw2),
    op!("LDMU", T22, Mpw2),
    op!("LDU", T21, Mpw2),
    op!("LDZ", T2300, Mpw2),
    op!("LDZimm16", T2310, Mpw2),
    op!("LDZU", T2300, Mpw2),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_L3LU: &[DefField] = &[
    f!(T1000, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T1010, Src0, 6, ENC_MIX4),
    f!(T1010, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T1010, DynLoop, 26, ENC_YES_NO),
    f!(T1010, Be, 31, ENC_BE),
    f!(T1020, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1020, Imm, 10, imm 16 Signed),
    f!(T1020, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T1020, JcrTarget, 27, ENC_JCRS),
    f!(T1020, Be, 31, ENC_BE),
    f!(T1030, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1030, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T1030, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1030, PcTarget, 22),
    f!(T1030, Isimm, 30, ENC_YES_NO),
    f!(T1030, Usejcr, 31, ENC_YES_NO),
    f!(T1040, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1040, CmpImm, 10, imm 8 Unsigned),
    f!(T1040, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1040, PcTarget, 22),
    f!(T1040, Isimm, 30, ENC_YES_NO),
    f!(T1040, Usejcr, 31, ENC_YES_NO),
    f!(T1100, Src0, 6, ENC_REGS),
    f!(T1100, Imm, 10, imm 14 ModuloUnsigned),
    f!(T1100, Be, 31, ENC_BE),
    f!(T1110, Src0, 6, ENC_REGS),
    f!(T1110, Imm, 10, imm 21 ModuloUnsigned),
    f!(T1110, Be, 31, ENC_BE),
    f!(T1112, Src0, 6, ENC_REGS),
    f!(T1112, Imm, 10, imm 21 Unsigned),
    f!(T1112, Be, 31, ENC_BE),
    f!(T1120, Src0, 6, ENC_GTRS),
    f!(T1120, Imm, 10, imm 14 Unsigned),
    f!(T1120, Be, 31, ENC_BE),
    f!(T12, Src0, 6, ENC_REGS),
    f!(T12, Src1, 10, ENC_REGS),
    f!(T12, Be, 31, ENC_BE),
    f!(T13, Soft, 8, ENC_NO_YES),
    f!(T13, Synctag, 10, ENC_MIX7),
    f!(T13, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Src1, 10, ENC_REGS),
    f!(T21, Node, 14),
    f!(T21, Burst, 22, ENC_MIX1),
    f!(T21, Group, 27, ENC_GTRS),
    f!(T21, Be, 31, ENC_BE),
    f!(T22, Src0, 6, ENC_REGS),
    f!(T22, Src1, 10, ENC_REGS),
    f!(T22, Src2, 14, ENC_REGS),
    f!(T22, Src3, 18, ENC_REGS),
    f!(T22, Burst, 22, ENC_MIX1),
    f!(T22, Group, 27, ENC_GTRS),
    f!(T22, Readibr, 30, ENC_READ_WRITE),
    f!(T22, Be, 31, ENC_BE),
    f!(T2210, Src0, 6, ENC_REGS),
    f!(T2210, Src1, 10, ENC_REGS),
    f!(T2210, Src2, 14, ENC_REGS),
    f!(T2210, Src3, 18, ENC_REGS),
    f!(T2210, Burst, 22, ENC_MIX1),
    f!(T2210, Be, 31, ENC_BE),
    f!(T2300, Src0, 6, ENC_REGS),
    f!(T2300, Src1, 10, ENC_REGS),
    f!(T2300, Imm, 14, imm 8 Unsigned),
    f!(T2300, Burst, 22, ENC_MIX1),
    f!(T2300, Mode, 27, ENC_ZERO2LX_IMM2LX_ZR2LX),
    f!(T2300, Be, 31, ENC_BE),
    f!(T2300, DatatypeVirtual, 100),
    f!(T2310, Imm, 6, imm 16 Unsigned),
    f!(T2310, Mode, 27, ENC_IMM2ZR),
    f!(T2310, Be, 31, ENC_BE),
    f!(T24, Src0, 6, ENC_REGS),
    f!(T24, Src1, 10, ENC_REGS),
    f!(T24, Ibr, 30, ENC_WRITE_NONE),
    f!(T24, Be, 31, ENC_BE),
];
#[cfg(feature = "arch-rcudd1a")]
const OPCODES_L3SU: &[DefOpcode] = &[
    op!("ADDEARIMM", T1110, Mpw2),
    op!("ADDLARIMM", T1100, Mpw2),
    op!("EARIMM", T1112, Mpw2),
    op!("EARREGCOPY", T12, Mpw2),
    op!("GTRIMM", T1120, Mpw2),
    op!("JADD", T1020, Mpw2),
    op!("JCMP", T1030, Mpw2),
    op!("JCMPI", T1040, Mpw2),
    op!("JIMMCOPY", T1020, Mpw2),
    op!("JSUB", T1020, Mpw2),
    op!("LARIMM", T1100, Mpw2),
    op!("LARREGCOPY", T12, Mpw2),
    op!("MODLARREG", T12, Mpw2),
    op!("MODEARREG", T12, Mpw2),
    op!("MVLOOPCNT", T1010, Mpw2),
    op!("NOP", T1000, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T1030, Mpw4),
    op!("SUBLARIMM", T1100, Mpw2),
    op!("SYNC", T13, Mpw2),
    op!("ST", T21, Mpw2),
    op!("STG", T21, Mpw2),
    op!("STGU", T21, Mpw2),
    op!("STIM", T2210, Mpw4),
    op!("STIMU", T2210, Mpw4),
    op!("STZ", T24, Mpw4),
    op!("STM", T22, Mpw2),
    op!("STMU", T22, Mpw2),
    op!("STU", T21, Mpw2),
];
#[cfg(feature = "arch-rcudd1a")]
const FIELDS_L3SU: &[DefField] = &[
    f!(T1000, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T1010, Src0, 6, ENC_MIX4),
    f!(T1010, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T1010, DynLoop, 26, ENC_YES_NO),
    f!(T1010, Be, 31, ENC_BE),
    f!(T1020, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1020, Imm, 10, imm 16 Signed),
    f!(T1020, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T1020, JcrTarget, 27, ENC_JCRS),
    f!(T1020, Be, 31, ENC_BE),
    f!(T1030, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1030, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T1030, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1030, PcTarget, 22),
    f!(T1030, Isimm, 30, ENC_YES_NO),
    f!(T1030, Usejcr, 31, ENC_YES_NO),
    f!(T1040, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1040, CmpImm, 10, imm 8 Unsigned),
    f!(T1040, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1040, PcTarget, 22),
    f!(T1040, Isimm, 30, ENC_YES_NO),
    f!(T1040, Usejcr, 31, ENC_YES_NO),
    f!(T1100, Src0, 6, ENC_REGS),
    f!(T1100, Imm, 10, imm 14 ModuloUnsigned),
    f!(T1100, Be, 31, ENC_BE),
    f!(T1110, Src0, 6, ENC_REGS),
    f!(T1110, Imm, 10, imm 21 ModuloUnsigned),
    f!(T1110, Be, 31, ENC_BE),
    f!(T1112, Src0, 6, ENC_REGS),
    f!(T1112, Imm, 10, imm 21 Unsigned),
    f!(T1112, Be, 31, ENC_BE),
    f!(T1120, Src0, 6, ENC_GTRS),
    f!(T1120, Imm, 10, imm 14 Unsigned),
    f!(T1120, Be, 31, ENC_BE),
    f!(T12, Src0, 6, ENC_REGS),
    f!(T12, Src1, 10, ENC_REGS),
    f!(T12, Be, 31, ENC_BE),
    f!(T13, Synctag, 10, ENC_MIX7),
    f!(T13, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Src1, 10, ENC_REGS),
    f!(T21, Node, 14),
    f!(T21, Burst, 22, ENC_MIX1),
    f!(T21, Group, 27, ENC_GTRS),
    f!(T21, Be, 31, ENC_BE),
    f!(T22, Src0, 6, ENC_REGS),
    f!(T22, Src1, 10, ENC_REGS),
    f!(T22, Src2, 14, ENC_REGS),
    f!(T22, Src3, 18, ENC_REGS),
    f!(T22, Burst, 22, ENC_MIX1),
    f!(T22, Group, 27, ENC_GTRS),
    f!(T22, Readibr, 30, ENC_READ_WRITE),
    f!(T22, Be, 31, ENC_BE),
    f!(T2210, Src0, 6, ENC_REGS),
    f!(T2210, Src1, 10, ENC_REGS),
    f!(T2210, Src2, 14, ENC_REGS),
    f!(T2210, Src3, 18, ENC_REGS),
    f!(T2210, Burst, 22, ENC_MIX1),
    f!(T2210, Be, 31, ENC_BE),
    f!(T2300, Src0, 6, ENC_REGS),
    f!(T2300, Src1, 10, ENC_REGS),
    f!(T2300, Imm, 14, imm 8 Unsigned),
    f!(T2300, Burst, 22, ENC_MIX1),
    f!(T2300, Mode, 27, ENC_ZERO2LX_IMM2LX_ZR2LX),
    f!(T2300, Be, 31, ENC_BE),
    f!(T2300, DatatypeVirtual, 100),
    f!(T2310, Imm, 6, imm 16 Unsigned),
    f!(T2310, Mode, 27, ENC_IMM2ZR),
    f!(T2310, Be, 31, ENC_BE),
    f!(T24, Src0, 6, ENC_REGS),
    f!(T24, Src1, 10, ENC_REGS),
    f!(T24, Ibr, 30, ENC_WRITE_NONE),
    f!(T24, Be, 31, ENC_BE),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_PT: &[DefOpcode] = &[
    op!("FMA", T20, Mpw2),
    op!("FMA4", T20, Sen1p5),
    op!("FMA8", T20, Mpw2),
    op!("IMA8", T20, Mpw2),
    op!("IMA4", T20, Mpw2),
    op!("JADD", T12, Mpw3),
    op!("JCMP", T13, Mpw3),
    op!("JCRSWAP", T14, Mpw4),
    op!("JIMMCOPY", T12, Mpw3),
    op!("JSUB", T12, Mpw3),
    op!("MVLOOPCNT", T10, Mpw2),
    op!("NOP", T11, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T13, Mpw4),
    op!("XRFACCESS", T15, Mpw4),
    op!("INCRMASK", T16, Mpw3),
    op!("SETMASK", T17, Mpw3),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_PT: &[DefField] = &[
    f!(T10, (Imm|PcTarget), 6, imm 16 Unsigned),
    f!(T10, Src0, 22, ENC_JCRS),
    f!(T10, DynLoop, 26, ENC_YES_NO),
    f!(T10, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Imm, 6, imm 16 Signed),
    f!(T12, Src0, 22, ENC_LCCRS_JCRS),
    f!(T12, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T12, JcrTarget, 27, ENC_JCRS),
    f!(T12, Be, 31, ENC_BE),
    f!(T13, Src1, 6, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T13, PcTarget, 14),
    f!(T13, Src0, 22, ENC_LCCRS_JCRS),
    f!(T13, Isimm, 26, ENC_YES_NO),
    f!(T13, Mode, 27, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T13, Usejcr, 31, ENC_YES_NO),
    f!(T14, Src1, 6, ENC_JCRS),
    f!(T14, Src0, 22, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T15, RdptrImm, 6, imm 7 ModuloUnsigned),
    f!(T15, RdptrUpd, 13, ENC_NO_SET_INCR),
    f!(T15, WrptrImm, 15, imm 7 ModuloUnsigned),
    f!(T15, WrptrUpd, 22, ENC_NO_SET_INCR),
    f!(T15, Be, 31, ENC_BE),
    f!(T16, Be, 31, ENC_BE),
    f!(T17, Imm, 6, imm 3 Unsigned),
    f!(T17, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_XRF_0_0),
    f!(T20, Src1, 10, ENC_MIX10),
    f!(T20, Src2, 14, ENC_W_LINK_1_0),
    f!(T20, IfifoConv, 16, ENC_NO_YES),
    f!(T20, Tgts, 18, ENC_MIX34),
    f!(T20, Tgtrf, 20, ENC_MIX9),
    f!(T20, Unrlfldsrc0, 24, ENC_YES_NO),
    f!(T20, Unrlfldsrc1, 25, ENC_YES_NO),
    f!(T20, Unrlfldtgt, 26, ENC_YES_NO),
    f!(T20, Unroll, 27, ENC_X1_X2_X4_X8),
    f!(T20, Fma8ctrlsrc0, 29, ENC_MIX11),
    f!(T20, Fma8ctrlsrc2, 30, ENC_143_152),
    f!(T20, Be, 31, ENC_BE),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_PE: &[DefOpcode] = &[
    op!("EE", T30, Mpw2),
    op!("FCMP", T30, Mpw2),
    op!("FEST", T30, Mpw2),
    op!("FMA", T20, Mpw2),
    op!("FMINMAX", T30, Mpw2),
    op!("FMUL", T20, Mpw2),
    op!("FNMS", T20, Mpw2),
    op!("GCVT", T30, Mpw2),
    op!("FCVT", T30, Sen1p5),
    op!("ICVT", T30, Mpw2),
    op!("IME", T30, Mpw2),
    op!("IMMCOPY", T13, Mpw2),
    op!("JADD", T41, Mpw2),
    op!("JCMP", T43, Mpw2),
    op!("JIMMCOPY", T41, Mpw2),
    op!("JSUB", T41, Mpw2),
    op!("LOGICAL", T30, Mpw2),
    op!("MERGE", T30, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("PACK", T30, Mpw2),
    op!("PERMUTE", T30, Sen1p5),
    op!("REDUCE", T118, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SELECT", T30, Mpw2),
    op!("SHR", T30, Mpw2),
    op!("SJCMP", T43, Mpw4),
    op!("SPLAT", T30, Mpw2),
    op!("LOAD_LFSR", T30, Sen1p5),
    op!("COPY_LFSR", T30, Sen1p5),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_PE: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 12, imm 16 Unsigned),
    f!(T11, DynLoop, 29, ENC_YES_NO),
    f!(T11, Be, 35, ENC_BE),
    f!(T18, Subroutine, 21, ENC_YES_NO),
    f!(T12, Be, 35, ENC_BE),
    f!(T13, Imm, 6, imm 16 Unsigned),
    f!(T13, Replica, 31),
    f!(T13, Tgtrf, 36, ENC_MIX33),
    f!(T13, Be, 35, ENC_BE),
    f!(T13, Mask, 42),
    f!(T13, Mode, 34, ENC_FP16_FP32),
    f!(T118, RsmSrc0, 6, ENC_MIX5),
    f!(T118, RsmSrc1, 11),
    f!(T118, RsmUnroll, 16, ENC_X1_X2_X3_X4),
    f!(T118, RsmTgtrf, 19, ENC_MIX5),
    f!(T118, RsmStateReg, 24),
    f!(T118, Be, 35, ENC_BE),
    f!(T118, Mode, 44, ENC_FP16_FP32),
    f!(T118, Foldctrl, 62, ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR),
    f!(T20, Src0, 6, ENC_MIX31),
    f!(T20, Src1, 12, ENC_MIX29),
    f!(T20, Src2, 18, ENC_MIX30),
    f!(T20, Tgtpe, 27, ENC_NO_RESULT),
    f!(T20, Fwdencoding, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtencoding, 30, ENC_LX_SFP),
    f!(T20, Mode, 34, ENC_FP16_FP32),
    f!(T20, Be, 35, ENC_BE),
    f!(T20, Tgtrf, 36, ENC_MIX3),
    f!(T20, Mask, 42),
    f!(T20, Unroll, 50, ENC_X1_X2_X4_X8),
    f!(T20, Unrlfldsrc0, 52, ENC_YES_NO),
    f!(T20, Unrlfldsrc1, 53, ENC_YES_NO),
    f!(T20, Unrlfldsrc2, 54, ENC_YES_NO),
    f!(T20, Unrlfldtgt, 55, ENC_YES_NO),
    f!(T20, Fpuop, 56, ENC_NO_INT8_INT4_DLFP16_BF16),
    f!(T20, Reluop, 59, ENC_NO_YES),
    f!(T20, Foldctrl, 62, ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR),
    f!(T30, Src0, 6, ENC_MIX32),
    f!(T30, Imm, 12, imm 6 Unsigned),
    f!(T30, Src2, 18, ENC_MIX28),
    f!(T30, Tgtpe, 27, ENC_NO_RESULT),
    f!(T30, Fwdencoding, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtencoding, 30, ENC_LX_SFP),
    f!(T30, Mode, 34, ENC_FP16_FP32),
    f!(T30, Be, 35, ENC_BE),
    f!(T30, Tgtrf, 36, ENC_MIX3),
    f!(T30, Mask, 42),
    f!(T30, Unroll, 50, ENC_X1_X2_X4_X8),
    f!(T30, Unrlfldsrc0, 52, ENC_YES_NO),
    f!(T30, Unrlfldsrc1, 53, ENC_YES_NO),
    f!(T30, Unrlfldsrc2, 54, ENC_YES_NO),
    f!(T30, Unrlfldtgt, 55, ENC_YES_NO),
    f!(T30, Fpuop, 56, ENC_NO_INT8_INT4_DLFP16_BF16),
    f!(T30, Reluop, 59, ENC_NO_YES),
    f!(T30, Foldctrl, 62, ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR),
    f!(T41, Src0, 6, ENC_LCCRS_JCRS),
    f!(T41, Imm, 12, imm 16 Signed),
    f!(T41, JcrSelect, 34, ENC_USEJCR_USELCCR),
    f!(T41, Be, 35, ENC_BE),
    f!(T41, JcrTarget, 36, ENC_JCRS),
    f!(T43, Src0, 6, ENC_LCCRS_JCRS),
    f!(T43, Src1, 12, ENC_JCRS),
    f!(T43, Mode, 20, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T43, Isimm, 28, ENC_YES_NO),
    f!(T43, Usejcr, 35, ENC_YES_NO),
    f!(T43, PcTarget, 36),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_SFP: &[DefOpcode] = &[
    op!("EE", T30, Mpw2),
    op!("FCMP", T30, Mpw2),
    op!("FEST", T30, Mpw2),
    op!("FMA", T20, Mpw2),
    op!("FMINMAX", T30, Mpw2),
    op!("FMUL", T20, Mpw2),
    op!("FNMS", T20, Mpw2),
    op!("GCVT", T30, Mpw2),
    op!("FCVT", T30, Mpw2),
    op!("ICVT", T30, Mpw2),
    op!("IME", T30, Mpw2),
    op!("IMMCOPY", T13, Mpw2),
    op!("JADD", T41, Mpw2),
    op!("JCMP", T43, Mpw2),
    op!("JIMMCOPY", T41, Mpw2),
    op!("JSUB", T41, Mpw2),
    op!("LOGICAL", T30, Mpw2),
    op!("MERGE", T30, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("PACK", T30, Mpw2),
    op!("PERMUTE", T30, Mpw2),
    op!("REDUCE", T118, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SELECT", T30, Mpw2),
    op!("SHR", T30, Mpw2),
    op!("SJCMP", T43, Mpw4),
    op!("SPLAT", T30, Mpw2),
    op!("SETDEST", T14, Rcudd1a),
    op!("LOAD_LFSR", T30, Mpw4),
    op!("COPY_LFSR", T30, Mpw4),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_SFP: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 12, imm 16 Unsigned),
    f!(T11, DynLoop, 29, ENC_YES_NO),
    f!(T11, Be, 35, ENC_BE),
    f!(T18, Subroutine, 21, ENC_YES_NO),
    f!(T12, Be, 35, ENC_BE),
    f!(T13, Imm, 6, imm 16 Unsigned),
    f!(T13, Replica, 31),
    f!(T13, Tgtrf, 36, ENC_MIX3),
    f!(T13, Be, 35, ENC_BE),
    f!(T13, Mask, 42),
    f!(T13, Mode, 34, ENC_FP16_FP32),
    f!(T14, Imm, 6, imm 32 Unsigned),
    f!(T118, RsmSrc0, 6, ENC_MIX5),
    f!(T118, RsmSrc1, 11),
    f!(T118, RsmUnroll, 16, ENC_X1_X2_X3_X4),
    f!(T118, RsmTgtrf, 19, ENC_MIX5),
    f!(T118, RsmStateReg, 24),
    f!(T118, Be, 35, ENC_BE),
    f!(T118, Mode, 44, ENC_FP16_FP32),
    f!(T118, Foldctrl, 62, ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR),
    f!(T20, Src0, 6, ENC_MIX24),
    f!(T20, Src1, 12, ENC_MIX26),
    f!(T20, Src2, 18, ENC_MIX27),
    f!(T20, Tgtsfp, 27, ENC_NO_RESULT),
    f!(T20, Fwdencoding, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T20, Tgtencoding, 30, ENC_L0_LX_PE_PT),
    f!(T20, Mode, 34, ENC_FP16_FP32),
    f!(T20, Be, 35, ENC_BE),
    f!(T20, Tgtrf, 36, ENC_MIX3),
    f!(T20, Mask, 42),
    f!(T20, Unroll, 50, ENC_X1_X2_X4_X8),
    f!(T20, Unrlfldsrc0, 52, ENC_YES_NO),
    f!(T20, Unrlfldsrc1, 53, ENC_YES_NO),
    f!(T20, Unrlfldsrc2, 54, ENC_YES_NO),
    f!(T20, Unrlfldtgt, 55, ENC_YES_NO),
    f!(T20, Fpuop, 56, ENC_NO_INT8_INT4_DLFP16_BF16),
    f!(T20, Reluop, 59, ENC_NO_YES),
    f!(T20, Tgtdatafifo, 61, ENC_NO_RESULT),
    f!(T20, Foldctrl, 62, ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR),
    f!(T30, Src0, 6, ENC_MIX25),
    f!(T30, Imm, 12, imm 6 Unsigned),
    f!(T30, Src2, 18, ENC_MIX23),
    f!(T30, Tgtsfp, 27, ENC_NO_RESULT),
    f!(T30, Fwdencoding, 28, ENC_NO_SRC0_SRC2_RESULT),
    f!(T30, Tgtencoding, 30, ENC_L0_LX_PE_PT),
    f!(T30, Mode, 34, ENC_FP16_FP32),
    f!(T30, Be, 35, ENC_BE),
    f!(T30, Tgtrf, 36, ENC_MIX3),
    f!(T30, Mask, 42),
    f!(T30, Unroll, 50, ENC_X1_X2_X4_X8),
    f!(T30, Unrlfldsrc0, 52, ENC_YES_NO),
    f!(T30, Unrlfldsrc1, 53, ENC_YES_NO),
    f!(T30, Unrlfldsrc2, 54, ENC_YES_NO),
    f!(T30, Unrlfldtgt, 55, ENC_YES_NO),
    f!(T30, Reluop, 59, ENC_NO_YES),
    f!(T30, Tgtdatafifo, 61, ENC_NO_RESULT),
    f!(T30, Foldctrl, 62, ENC_FOLDA_FOLDB_2FOLD2INSTR_2FOLD1INSTR),
    f!(T41, Src0, 6, ENC_LCCRS_JCRS),
    f!(T41, Imm, 12, imm 16 Signed),
    f!(T41, JcrSelect, 34, ENC_USEJCR_USELCCR),
    f!(T41, Be, 35, ENC_BE),
    f!(T41, JcrTarget, 36, ENC_JCRS),
    f!(T43, Src0, 6, ENC_LCCRS_JCRS),
    f!(T43, Src1, 12, ENC_JCRS),
    f!(T43, Mode, 20, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T43, Isimm, 28, ENC_YES_NO),
    f!(T43, Usejcr, 35, ENC_YES_NO),
    f!(T43, PcTarget, 36),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_L0LU: &[DefOpcode] = &[
    op!("IMMCOPY", T16, Mpw2),
    op!("LDST", T20, Mpw2),
    op!("LDSTI", T11, Mpw2),
    op!("LDSTIU", T11, Mpw2),
    op!("LDSTU", T20, Mpw2),
    op!("LRFREGCOPY", T21, Mpw2),
    op!("MODLRFREG", T21, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T30, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("MODLRFIMM", T16, Mpw2),
    op!("MVLOOPCNT", T10, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T30, Mpw4),
    op!("SYNC", T13, Mpw2),
    op!("TILEADV", T12, Sen1p5),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_L0LU: &[DefField] = &[
    f!(T10, Src0, 6, ENC_JCRS),
    f!(T10, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T10, DynLoop, 26, ENC_YES_NO),
    f!(T10, Be, 31, ENC_BE),
    f!(T11, Src0, 6, ENC_REGS),
    f!(T11, Imm, 10, imm 12 Signed),
    f!(T11, Ldtype, 26, ENC_64B_32B_16B),
    f!(T11, Splat, 28, ENC_SPLAT),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_SEND_RECV_SENDRECV),
    f!(T13, Implicit, 7, ENC_NO_YES),
    f!(T13, Tilesize, 22, imm 7 Unsigned),
    f!(T13, Syncdest, 12, ENC_SELF_NEIGHBOR_BOTH),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T16, Src0, 6, ENC_REGS),
    f!(T16, Imm, 10, imm 12 ModuloUnsigned),
    f!(T16, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Src1, 10, ENC_REGS),
    f!(T20, Group, 14),
    f!(T20, Burst, 16, ENC_BURST),
    f!(T20, Burstsize, 17, imm 6 Unsigned),
    f!(T20, Chunkstride, 23, ENC_8B_16B_32B_64B_96B_128B_192B_256B),
    f!(T20, Ldtype, 26, ENC_64B_32B_16B),
    f!(T20, Chunksize, 29, ENC_8B_16B_32B),
    f!(T20, Splat, 28, ENC_SPLAT),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Src1, 10, ENC_REGS),
    f!(T21, Be, 31, ENC_BE),
    f!(T30, Src0, 6, ENC_LCCRS_JCRS),
    f!(T30, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T30, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T30, PcTarget, 22),
    f!(T30, Isimm, 30, ENC_YES_NO),
    f!(T30, Usejcr, 31, ENC_YES_NO),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_L0SU: &[DefOpcode] = &[
    op!("IMMCOPY", T16, Mpw2),
    op!("LDST", T20, Mpw2),
    op!("LDSTI", T11, Mpw2),
    op!("LDSTIU", T11, Mpw2),
    op!("LDSTU", T20, Mpw2),
    op!("LRFREGCOPY", T21, Mpw2),
    op!("MODLRFREG", T21, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T30, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("MODLRFIMM", T16, Mpw2),
    op!("MVLOOPCNT", T10, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T30, Mpw4),
    op!("SYNC", T13, Mpw2),
    op!("TILEADV", T12, Sen1p5),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_L0SU: &[DefField] = &[
    f!(T10, Src0, 6, ENC_JCRS),
    f!(T10, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T10, DynLoop, 26, ENC_YES_NO),
    f!(T10, Be, 31, ENC_BE),
    f!(T11, Src0, 6, ENC_REGS),
    f!(T11, Imm, 10, imm 12 Signed),
    f!(T11, ScaleArray, 23, ENC_SCALE_DATA),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_SEND_RECV_SENDRECV),
    f!(T13, Implicit, 7, ENC_NO_YES),
    f!(T13, Tilesize, 22, imm 7 Unsigned),
    f!(T13, Syncdest, 12, ENC_SELF_NEIGHBOR_BOTH),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T16, Src0, 6, ENC_REGS),
    f!(T16, Imm, 10, imm 12 ModuloUnsigned),
    f!(T16, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Src1, 10, ENC_REGS),
    f!(T20, Group, 14),
    f!(T20, Burst, 16, ENC_BURST),
    f!(T20, Burstsize, 17, imm 6 Unsigned),
    f!(T20, ScaleArray, 23, ENC_SCALE_DATA),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Src1, 10, ENC_REGS),
    f!(T21, Be, 31, ENC_BE),
    f!(T30, Src0, 6, ENC_LCCRS_JCRS),
    f!(T30, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T30, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T30, PcTarget, 22),
    f!(T30, Isimm, 30, ENC_YES_NO),
    f!(T30, Usejcr, 31, ENC_YES_NO),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_LXLU: &[DefOpcode] = &[
    op!("IMMCOPY", T111, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T40, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("LDST", T32, Mpw2),
    op!("LDSTI", T20, Mpw2),
    op!("LDSTIU", T20, Mpw2),
    op!("LDSTU", T32, Mpw2),
    op!("LDCVTI", T21, Sen1p5),
    op!("LDCVTIU", T21, Sen1p5),
    op!("SAMV", T50, Mpw3),
    op!("SETDSTMASK", T15, Rcudd1a),
    op!("SPMV", T60, Rcudd1a),
    op!("LRFREGCOPY", T31, Mpw2),
    op!("MODLRFIMM", T111, Mpw2),
    op!("MODLRFREG", T31, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T40, Mpw4),
    op!("SUBLRFIMM", T111, Mpw2),
    op!("SYNC", T13, Mpw2),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_LXLU: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T11, DynLoop, 26, ENC_YES_NO),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_MIX12),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T15, Mode, 19, ENC_SFP_L0SU_PT),
    f!(T15, Be, 31, ENC_BE),
    f!(T111, Src0, 6, ENC_REGS),
    f!(T111, Lrfimm, 10, imm 21 ModuloUnsigned),
    f!(T111, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Imm, 10, imm 14 Signed),
    f!(T20, Rottype, 24, ENC_NO_16B_32B_48B_64B_80B_96B_112B),
    f!(T20, Ldtype, 27, ENC_128B_2BSPLAT_16BZPAD_16BSPLAT),
    f!(T20, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Imm, 10, imm 14 Signed),
    f!(T21, Elemidx, 24),
    f!(T21, Scaleidx, 26),
    f!(T21, Ldtype, 27, ENC_128B_2BSPLAT_4BSPLAT_16BSPLAT),
    f!(T21, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T21, Be, 31, ENC_BE),
    f!(T31, Src0, 6, ENC_REGS),
    f!(T31, Src1, 10, ENC_REGS),
    f!(T31, Be, 31, ENC_BE),
    f!(T32, Src0, 6, ENC_REGS),
    f!(T32, Src1, 10, ENC_REGS),
    f!(T32, Group, 14, ENC_NO_X1_X2_X4),
    f!(T32, Burst, 16, ENC_BURST),
    f!(T32, Burstsize, 17, imm 6 Unsigned),
    f!(T32, Rottype, 24, ENC_NO_16B_32B_48B_64B_80B_96B_112B),
    f!(T32, Ldtype, 27, ENC_128B_2BSPLAT_16BZPAD_16BSPLAT),
    f!(T32, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T32, Be, 31, ENC_BE),
    f!(T40, Src0, 6, ENC_LCCRS_JCRS),
    f!(T40, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T40, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T40, PcTarget, 22),
    f!(T40, Isimm, 30, ENC_YES_NO),
    f!(T40, Usejcr, 31, ENC_YES_NO),
    f!(T50, Maskall, 10, ENC_YES_NO),
    f!(T50, Sliceidxsl, 11),
    f!(T50, Numvalidentry, 14),
    f!(T50, Mvridx, 21, ENC_MVRS),
    f!(T50, Precision, 24, ENC_16B_8B_4B_2B),
    f!(T50, Xslinner, 26, ENC_YES_NO),
    f!(T50, Wsllen, 27, ENC_64B_32B_16B_8B_4B_2B_1B),
    f!(T50, Be, 31, ENC_BE),
    f!(T60, Mvridx, 6, ENC_MVRS),
    f!(T60, Subslicesize, 9, ENC_1B_2B_4B_8B_16B_32B_64B_128B),
    f!(T60, Startbit, 12),
    f!(T60, Endbit, 20),
    f!(T60, Byteshift, 28, ENC_NO_2B_4B_6B),
    f!(T60, Be, 31, ENC_BE),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_LXSU: &[DefOpcode] = &[
    op!("IMMCOPY", T111, Mpw2),
    op!("JADD", T14, Mpw2),
    op!("JCMP", T40, Mpw2),
    op!("JIMMCOPY", T14, Mpw2),
    op!("JSUB", T14, Mpw2),
    op!("LDST", T32, Mpw2),
    op!("LDSTI", T20, Mpw2),
    op!("LDSTIU", T20, Mpw2),
    op!("LDSTU", T32, Mpw2),
    op!("LRFCOPY", T20, Mpw2),
    op!("LRFREGCOPY", T31, Mpw2),
    op!("MODLRFIMM", T111, Mpw2),
    op!("MODLRFREG", T31, Mpw2),
    op!("MVLOOPCNT", T11, Mpw2),
    op!("NOP", T12, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T40, Mpw4),
    op!("SUBLRFIMM", T111, Mpw2),
    op!("SYNC", T13, Mpw2),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_LXSU: &[DefField] = &[
    f!(T11, Src0, 6, ENC_JCRS),
    f!(T11, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T11, DynLoop, 26, ENC_YES_NO),
    f!(T11, Be, 31, ENC_BE),
    f!(T12, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T13, Synctag, 10, ENC_MIX13),
    f!(T13, Be, 31, ENC_BE),
    f!(T14, Src0, 6, ENC_LCCRS_JCRS),
    f!(T14, Imm, 10, imm 16 Signed),
    f!(T14, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T14, JcrTarget, 27, ENC_JCRS),
    f!(T14, Be, 31, ENC_BE),
    f!(T15, Mode, 19, ENC_SFP_L0SU_PT),
    f!(T15, Be, 31, ENC_BE),
    f!(T111, Src0, 6, ENC_REGS),
    f!(T111, Lrfimm, 10, imm 21 ModuloUnsigned),
    f!(T111, Be, 31, ENC_BE),
    f!(T20, Src0, 6, ENC_REGS),
    f!(T20, Imm, 10, imm 14 Signed),
    f!(T20, Hwsel, 24, ENC_NO),
    f!(T20, Sttype, 27, ENC_128B_16B_2B),
    f!(T20, Producertag, 29, ENC_ZERO_SFP_PE_LXLU),
    f!(T20, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Imm, 10, imm 14 Signed),
    f!(T21, Elemidx, 24),
    f!(T21, Scaleidx, 26),
    f!(T21, Ldtype, 27, ENC_128B_2BSPLAT_4BSPLAT_16BSPLAT),
    f!(T21, Consumertag, 29, ENC_SELF_SFP_LXSU_PE),
    f!(T21, Be, 31, ENC_BE),
    f!(T31, Src0, 6, ENC_REGS),
    f!(T31, Src1, 10, ENC_REGS),
    f!(T31, Be, 31, ENC_BE),
    f!(T32, Src0, 6, ENC_REGS),
    f!(T32, Src1, 10, ENC_REGS),
    f!(T32, Group, 14, ENC_NO_X1_X2_X4),
    f!(T32, Burst, 16, ENC_BURST),
    f!(T32, Burstsize, 17, imm 6 Unsigned),
    f!(T32, Sttype, 27, ENC_128B_16B_2B),
    f!(T32, Producertag, 29, ENC_ZERO_SFP_PE_LXLU),
    f!(T32, Be, 31, ENC_BE),
    f!(T40, Src0, 6, ENC_LCCRS_JCRS),
    f!(T40, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T40, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T40, PcTarget, 22),
    f!(T40, Isimm, 30, ENC_YES_NO),
    f!(T40, Usejcr, 31, ENC_YES_NO),
    f!(T50, Maskall, 10, ENC_YES_NO),
    f!(T50, Sliceidxsl, 11),
    f!(T50, Numvalidentry, 14),
    f!(T50, Mvridx, 21, ENC_MVRS),
    f!(T50, Precision, 24, ENC_16B_8B_4B_2B),
    f!(T50, Xslinner, 26, ENC_YES_NO),
    f!(T50, Wsllen, 27, ENC_64B_32B_16B_8B_4B_2B_1B),
    f!(T50, Be, 31, ENC_BE),
    f!(T60, Mvridx, 6, ENC_MVRS),
    f!(T60, Subslicesize, 9, ENC_1B_2B_4B_8B_16B_32B_64B_128B),
    f!(T60, Startbit, 12),
    f!(T60, Endbit, 20),
    f!(T60, Byteshift, 28, ENC_NO_2B_4B_6B),
    f!(T60, Be, 31, ENC_BE),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_L3LU: &[DefOpcode] = &[
    op!("ADDEARIMM", T1110, Mpw2),
    op!("ADDLARIMM", T1100, Mpw2),
    op!("EARIMM", T1112, Mpw2),
    op!("EARREGCOPY", T12, Mpw2),
    op!("GTRIMM", T1120, Mpw2),
    op!("JADD", T1020, Mpw2),
    op!("JCMP", T1030, Mpw2),
    op!("JCMPI", T1040, Mpw2),
    op!("JIMMCOPY", T1020, Mpw2),
    op!("JSUB", T1020, Mpw2),
    op!("LARIMM", T1100, Mpw2),
    op!("LARREGCOPY", T12, Mpw2),
    op!("MODLARREG", T12, Mpw2),
    op!("MODEARREG", T12, Mpw2),
    op!("MVLOOPCNT", T1010, Mpw2),
    op!("NOP", T1000, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T1030, Mpw4),
    op!("SUBLARIMM", T1100, Mpw2),
    op!("SYNC", T13, Mpw2),
    op!("LD", T21, Mpw2),
    op!("LDG", T21, Mpw2),
    op!("LDGM", T22, Mpw2),
    op!("LDGMU", T22, Mpw2),
    op!("LDGU", T21, Mpw2),
    op!("LDIGM", T22, Mpw4),
    op!("LDIGMU", T22, Mpw4),
    op!("LDIM", T22, Mpw4),
    op!("LDIMU", T22, Mpw4),
    op!("LDM", T22, Mpw2),
    op!("LDMU", T22, Mpw2),
    op!("LDU", T21, Mpw2),
    op!("LDZ", T2300, Mpw2),
    op!("LDZimm16", T2310, Mpw2),
    op!("LDZU", T2300, Mpw2),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_L3LU: &[DefField] = &[
    f!(T1000, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T1010, Src0, 6, ENC_MIX4),
    f!(T1010, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T1010, DynLoop, 26, ENC_YES_NO),
    f!(T1010, Be, 31, ENC_BE),
    f!(T1020, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1020, Imm, 10, imm 16 Signed),
    f!(T1020, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T1020, JcrTarget, 27, ENC_JCRS),
    f!(T1020, Be, 31, ENC_BE),
    f!(T1030, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1030, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T1030, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1030, PcTarget, 22),
    f!(T1030, Isimm, 30, ENC_YES_NO),
    f!(T1030, Usejcr, 31, ENC_YES_NO),
    f!(T1040, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1040, CmpImm, 10, imm 8 Unsigned),
    f!(T1040, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1040, PcTarget, 22),
    f!(T1040, Isimm, 30, ENC_YES_NO),
    f!(T1040, Usejcr, 31, ENC_YES_NO),
    f!(T1100, Src0, 6, ENC_REGS),
    f!(T1100, Imm, 10, imm 15 ModuloUnsigned),
    f!(T1100, Be, 31, ENC_BE),
    f!(T1110, Src0, 6, ENC_REGS),
    f!(T1110, Imm, 10, imm 21 ModuloUnsigned),
    f!(T1110, Be, 31, ENC_BE),
    f!(T1112, Src0, 6, ENC_REGS),
    f!(T1112, Imm, 10, imm 21 Unsigned),
    f!(T1112, Be, 31, ENC_BE),
    f!(T1120, Src0, 6, ENC_GTRS),
    f!(T1120, Imm, 10, imm 14 Unsigned),
    f!(T1120, Be, 31, ENC_BE),
    f!(T12, Src0, 6, ENC_REGS),
    f!(T12, Src1, 10, ENC_REGS),
    f!(T12, Be, 31, ENC_BE),
    f!(T13, Soft, 8, ENC_NO_YES),
    f!(T13, Synctag, 10, ENC_MIX7),
    f!(T13, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Drm, 10, ENC_NO_CCW_CW_BOTH),
    f!(T21, Node, 14),
    f!(T21, Burst, 22, ENC_MIX1),
    f!(T21, Group, 27, ENC_GTRS),
    f!(T21, Be, 31, ENC_BE),
    f!(T22, Src0, 6, ENC_REGS),
    f!(T22, Drm, 10, ENC_NO_CCW_CW_BOTH),
    f!(T22, Src2, 14, ENC_REGS),
    f!(T22, Src3, 18, ENC_REGS),
    f!(T22, Burst, 22, ENC_MIX1),
    f!(T22, Group, 27, ENC_GTRS),
    f!(T22, Readibr, 30, ENC_READ_WRITE),
    f!(T22, Be, 31, ENC_BE),
    f!(T2210, Src0, 6, ENC_REGS),
    f!(T2210, Drm, 10, ENC_NO_CCW_CW_BOTH),
    f!(T2210, Src2, 14, ENC_REGS),
    f!(T2210, Src3, 18, ENC_REGS),
    f!(T2210, Burst, 22, ENC_MIX1),
    f!(T2210, Be, 31, ENC_BE),
    f!(T2300, Src0, 6, ENC_REGS),
    f!(T2300, Imm, 14, imm 8 Unsigned),
    f!(T2300, Burst, 22, ENC_MIX1),
    f!(T2300, Mode, 27, ENC_ZERO2LX_IMM2LX_ZR2LX),
    f!(T2300, Be, 31, ENC_BE),
    f!(T2300, DatatypeVirtual, 100),
    f!(T2310, Imm, 6, imm 16 Unsigned),
    f!(T2310, Mode, 27, ENC_IMM2ZR),
    f!(T2310, Be, 31, ENC_BE),
    f!(T24, Src0, 6, ENC_REGS),
    f!(T24, Ibr, 30, ENC_WRITE_NONE),
    f!(T24, Be, 31, ENC_BE),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const OPCODES_L3SU: &[DefOpcode] = &[
    op!("ADDEARIMM", T1110, Mpw2),
    op!("ADDLARIMM", T1100, Mpw2),
    op!("EARIMM", T1112, Mpw2),
    op!("EARREGCOPY", T12, Mpw2),
    op!("GTRIMM", T1120, Mpw2),
    op!("JADD", T1020, Mpw2),
    op!("JCMP", T1030, Mpw2),
    op!("JCMPI", T1040, Mpw2),
    op!("JIMMCOPY", T1020, Mpw2),
    op!("JSUB", T1020, Mpw2),
    op!("LARIMM", T1100, Mpw2),
    op!("LARREGCOPY", T12, Mpw2),
    op!("MODLARREG", T12, Mpw2),
    op!("MODEARREG", T12, Mpw2),
    op!("MVLOOPCNT", T1010, Mpw2),
    op!("NOP", T1000, Mpw2),
    op!("RETURN", T18, Mpw2),
    op!("SJCMP", T1030, Mpw4),
    op!("SUBLARIMM", T1100, Mpw2),
    op!("SYNC", T13, Mpw2),
    op!("ST", T21, Mpw2),
    op!("STG", T21, Mpw2),
    op!("STGU", T21, Mpw2),
    op!("STIM", T2210, Mpw4),
    op!("STIMU", T2210, Mpw4),
    op!("STZ", T24, Mpw4),
    op!("STM", T22, Mpw2),
    op!("STMU", T22, Mpw2),
    op!("STU", T21, Mpw2),
];
#[cfg(not(feature = "arch-rcudd1a"))]
const FIELDS_L3SU: &[DefField] = &[
    f!(T1000, Be, 31, ENC_BE),
    f!(T18, Subroutine, 26, ENC_YES_NO),
    f!(T1010, Src0, 6, ENC_MIX4),
    f!(T1010, (Imm|PcTarget), 10, imm 16 Unsigned),
    f!(T1010, DynLoop, 26, ENC_YES_NO),
    f!(T1010, Be, 31, ENC_BE),
    f!(T1020, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1020, Imm, 10, imm 16 Signed),
    f!(T1020, JcrSelect, 26, ENC_USEJCR_USELCCR),
    f!(T1020, JcrTarget, 27, ENC_JCRS),
    f!(T1020, Be, 31, ENC_BE),
    f!(T1030, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1030, Src1, 10, imm 8 ModuloUnsigned, ENC_JCRS),
    f!(T1030, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1030, PcTarget, 22),
    f!(T1030, Isimm, 30, ENC_YES_NO),
    f!(T1030, Usejcr, 31, ENC_YES_NO),
    f!(T1040, Src0, 6, ENC_LCCRS_JCRS),
    f!(T1040, CmpImm, 10, imm 8 Unsigned),
    f!(T1040, Mode, 19, ENC_ALWAYS_LT_EQ_LE_GT_NE_GE_NEVER),
    f!(T1040, PcTarget, 22),
    f!(T1040, Isimm, 30, ENC_YES_NO),
    f!(T1040, Usejcr, 31, ENC_YES_NO),
    f!(T1100, Src0, 6, ENC_REGS),
    f!(T1100, Imm, 10, imm 15 ModuloUnsigned),
    f!(T1100, Be, 31, ENC_BE),
    f!(T1110, Src0, 6, ENC_REGS),
    f!(T1110, Imm, 10, imm 21 ModuloUnsigned),
    f!(T1110, Be, 31, ENC_BE),
    f!(T1112, Src0, 6, ENC_REGS),
    f!(T1112, Imm, 10, imm 21 Unsigned),
    f!(T1112, Be, 31, ENC_BE),
    f!(T1120, Src0, 6, ENC_GTRS),
    f!(T1120, Imm, 10, imm 14 Unsigned),
    f!(T1120, Be, 31, ENC_BE),
    f!(T12, Src0, 6, ENC_REGS),
    f!(T12, Src1, 10, ENC_REGS),
    f!(T12, Be, 31, ENC_BE),
    f!(T13, Synctag, 10, ENC_MIX7),
    f!(T13, Be, 31, ENC_BE),
    f!(T21, Src0, 6, ENC_REGS),
    f!(T21, Drm, 10, ENC_NO_CCW_CW_BOTH),
    f!(T21, Node, 14),
    f!(T21, Burst, 22, ENC_MIX1),
    f!(T21, Group, 27, ENC_GTRS),
    f!(T21, Be, 31, ENC_BE),
    f!(T22, Src0, 6, ENC_REGS),
    f!(T22, Drm, 10, ENC_NO_CCW_CW_BOTH),
    f!(T22, Src2, 14, ENC_REGS),
    f!(T22, Src3, 18, ENC_REGS),
    f!(T22, Burst, 22, ENC_MIX1),
    f!(T22, Group, 27, ENC_GTRS),
    f!(T22, Readibr, 30, ENC_READ_WRITE),
    f!(T22, Be, 31, ENC_BE),
    f!(T2210, Src0, 6, ENC_REGS),
    f!(T2210, Drm, 10, ENC_NO_CCW_CW_BOTH),
    f!(T2210, Src2, 14, ENC_REGS),
    f!(T2210, Src3, 18, ENC_REGS),
    f!(T2210, Burst, 22, ENC_MIX1),
    f!(T2210, Be, 31, ENC_BE),
    f!(T2300, Src0, 6, ENC_REGS),
    f!(T2300, Imm, 14, imm 8 Unsigned),
    f!(T2300, Burst, 22, ENC_MIX1),
    f!(T2300, Mode, 27, ENC_ZERO2LX_IMM2LX_ZR2LX),
    f!(T2300, Be, 31, ENC_BE),
    f!(T2300, DatatypeVirtual, 100),
    f!(T2310, Imm, 6, imm 16 Unsigned),
    f!(T2310, Mode, 27, ENC_IMM2ZR),
    f!(T2310, Be, 31, ENC_BE),
    f!(T24, Src0, 6, ENC_REGS),
    f!(T24, Ibr, 30, ENC_WRITE_NONE),
    f!(T24, Be, 31, ENC_BE),
];

/// A COMPONENT, as `initIsa` branches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Comp {
    Pt,
    Pe,
    Sfp,
    L0lu,
    L0su,
    Lxlu,
    Lxsu,
    L3lu,
    L3su,
}

/// WHICH `coreArch` THE TABLES IN THIS BUILD WERE SELECTED FOR — `SenSystemDef::coreArch`, as the
/// cargo feature fixes it.
///
/// ⛔ ONLY TWO ARE PORTED, AND A THIRD IS NOT A NEAR MISS. `initIsa`'s branches turn on
/// `coreArch <= MPW3`, `< / == / >= RCUDD1A` and `>= SEN1P5` (`isa.cpp:235-1212`). MPW4 and RCUDD1A
/// therefore agree wherever the test is `<= RCUDD1A`, `<= MPW3` or `>= SEN1P5`, and DISAGREE wherever
/// it names the RCUDD1A boundary directly — `grep -c` counts **11** such sites. A program written for
/// one arch and assembled against another's table gets silently different field POSITIONS at those
/// sites, which is a wrong program that loads.
///
/// 🛑 The count is of BRANCH SITES, not of fields: one branch may define several. So 11 is the number
/// of places the two tables can part company, and an upper bound on nothing.
///
/// 🔑 So this constant exists to be COMPARED AGAINST, not just recorded: a caller holding a program
/// whose arch is known can refuse to assemble it rather than produce a plausible answer.
pub const TABLE_ARCH: Gen = {
    #[cfg(feature = "arch-rcudd1a")]
    {
        Gen::Rcudd1a
    }
    #[cfg(not(feature = "arch-rcudd1a"))]
    {
        Gen::Sen1p5
    }
};

impl Comp {
    /// Every opcode this component defines, for the arch being built.
    pub const fn opcodes(self) -> &'static [DefOpcode] {
        match self {
            Self::Pt => OPCODES_PT,
            Self::Pe => OPCODES_PE,
            Self::Sfp => OPCODES_SFP,
            Self::L0lu => OPCODES_L0LU,
            Self::L0su => OPCODES_L0SU,
            Self::Lxlu => OPCODES_LXLU,
            Self::Lxsu => OPCODES_LXSU,
            Self::L3lu => OPCODES_L3LU,
            Self::L3su => OPCODES_L3SU,
        }
    }

    /// Every field this component's instruction types have, for the arch being built.
    pub const fn fields(self) -> &'static [DefField] {
        match self {
            Self::Pt => FIELDS_PT,
            Self::Pe => FIELDS_PE,
            Self::Sfp => FIELDS_SFP,
            Self::L0lu => FIELDS_L0LU,
            Self::L0su => FIELDS_L0SU,
            Self::Lxlu => FIELDS_LXLU,
            Self::Lxsu => FIELDS_LXSU,
            Self::L3lu => FIELDS_L3LU,
            Self::L3su => FIELDS_L3SU,
        }
    }

    /// All nine, so a walk cannot miss one.
    pub const ALL: [Self; 9] = [
        Self::Pt,
        Self::Pe,
        Self::Sfp,
        Self::L0lu,
        Self::L0su,
        Self::Lxlu,
        Self::Lxsu,
        Self::L3lu,
        Self::L3su,
    ];
}
