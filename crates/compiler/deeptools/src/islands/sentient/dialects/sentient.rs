//! `SentientOps.td` + `SentientTypes.td` — THE MACHINE'S OWN VOCABULARY: PORTS, REGISTERS,
//! FORWARDING, PRECISION PER OPERAND, AND UNROLL.
//!
//! The dialect declares **twenty-nine** operations
//! (`dcc/src/Dialect/Sentient/SentientOps.td`) over **sixteen** enums
//! (`dcc/src/Dialect/Sentient/SentientTypes.td`). Both files are on the pod at
//! `/project_src/deeptools`, which is the authority; nothing here is vendored.
//!
//! ⭐⭐ THIS IS THE RUNG WHERE THE DATAPATH BECOMES EXPLICIT, and the DataflowIR island says so from
//! the other side: *"No register, no port, no result forwarding, no unroll factor, no precision per
//! operand. Those are `sentient.*`"*
//! ([`crate::islands::dataflow_ir::dialects`]). Every one of those five now has a type here.
//!
//! # 🛑 THE WIRE'S TWO ENDS ARE THE RUNG BELOW'S, AND THAT IS DELIBERATE
//!
//! ⛔⛔ A LOCKDOWN THAT DOES NOT SURVIVE ITS LOWERING IS WORSE THAN NONE. The DataflowIR island mints
//! a [`crate::islands::dataflow_ir::link::Link`] once and hands out its two ends once, so a
//! `dataflow.send`'s destination and the matching `dataflow.receive`'s source are the same wire *by
//! construction*. The first version of this file then took `consumer: Val` and `producer: Val` —
//! two bare handles — so the guarantee evaporated at exactly the rung where an instruction finally
//! names a unit, and `Link<Lxlu, Sfp>` became two unrelated integers.
//!
//! ⭐ SO THESE OPS TAKE [`SendEnd`] AND [`RecvEnd`] THEMSELVES. Same types, same wire, one lowering
//! further down: a send here cannot be handed a receive's end, and neither can be conjured from a
//! unit handle, because only `Link::ends` mints them and it consumes the link to do it.
//!
//! ⛔ WHAT IS STILL NOT HERE: a register INDEX inside a file, an instruction encoding, a program
//! counter. `regIndex` appears on these ops as a *hint* an allocator fills in
//! (`DefaultValuedAttr<I32Attr, "-1">` — minus one means unassigned), and the passes that decide it
//! are D64-D75, `registerManagementPasses`. A real index and a real opcode are ProgIR
//! ([`crate::islands::progir`]).

use std::fmt::Write as _;

use crate::arch::{Bounded, Elements};
use crate::formats::Bits;
use crate::generated::{OpaqueFunc, ParamKey, ParamValue, RegName};
use crate::islands::dataflow_ir::dialects::dataflow::RegAddr;
use crate::islands::dataflow_ir::link::{RecvEnd, SendEnd};
use crate::islands::dataflow_ir::ty::{ScalarTy, Vector};
use crate::islands::sentient::dialects::Val;
use crate::islands::sentient::print;

/// ONE ELEMENT PRECISION — `SentientPrecisionAttr`, spelled `precision`.
///
/// ⛔ THE DISCRIMINANTS ARE THE WIRE VALUES AND **SEVEN IS ABSENT**. `int32` is 6 and `mxfp4` is 8
/// (`SentientTypes.td:55-56`); a contiguous `#[repr]` would shift every float format down by one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    /// `int1`.
    Int1,
    /// `int2`.
    Int2,
    /// `int4`.
    Int4,
    /// `int8`.
    Int8,
    /// `int16`.
    Int16,
    /// `int24` — ⛔ SIXTEEN BITS WIDE, not twenty-four; see [`crate::formats`].
    Int24,
    /// `int32`.
    Int32,
    /// `mxfp4`.
    Mxfp4,
    /// `mxfp8`.
    Mxfp8,
    /// `mxint4`.
    Mxint4,
    /// `fp4`.
    Fp4,
    /// `fp8`.
    Fp8,
    /// `fp16` — the default every compute operand carries when nothing sets one.
    Fp16,
    /// `bf16`.
    Bf16,
    /// `ieee_fp16`.
    IeeeFp16,
    /// `fp24`.
    Fp24,
    /// `fp32`.
    Fp32,
    /// `int64`.
    Int64,
    /// `none`.
    None,
}

impl Precision {
    /// The spelling the attribute carries (`SentientTypes.td:49-67`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Int1 => "int1",
            Self::Int2 => "int2",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int24 => "int24",
            Self::Int32 => "int32",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp8 => "mxfp8",
            Self::Mxint4 => "mxint4",
            Self::Fp4 => "fp4",
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::IeeeFp16 => "ieee_fp16",
            Self::Fp24 => "fp24",
            Self::Fp32 => "fp32",
            Self::Int64 => "int64",
            Self::None => "none",
        }
    }

    /// The wire value (`SentientTypes.td:49-67`). ⛔ SEVEN IS SKIPPED — see the type's own note.
    #[must_use]
    pub const fn encoding(self) -> u32 {
        match self {
            Self::Int1 => 0,
            Self::Int2 => 1,
            Self::Int4 => 2,
            Self::Int8 => 3,
            Self::Int16 => 4,
            Self::Int24 => 5,
            Self::Int32 => 6,
            Self::Mxfp4 => 8,
            Self::Mxfp8 => 9,
            Self::Mxint4 => 10,
            Self::Fp4 => 11,
            Self::Fp8 => 12,
            Self::Fp16 => 13,
            Self::Bf16 => 14,
            Self::IeeeFp16 => 15,
            Self::Fp24 => 16,
            Self::Fp32 => 17,
            Self::Int64 => 18,
            Self::None => 19,
        }
    }
}

/// WHERE A COMPUTE OPERAND COMES FROM, OR WHERE A RESULT GOES — `SentientComputePortAttr`.
///
/// Sixty-seven cases (`SentientTypes.td:96-232`): the register files, the four neighbour links, the
/// pseudo-constants, the units a result may be forwarded to, and the internal state registers.
///
/// ⛔⛔ THE LRF RANGE IS **SPLIT**, AND THE GAP IS OCCUPIED. `lrf0..lrf15` are 12..27 and
/// `lrf16..lrf31` are **48..63** (`SentientTypes.td:108-160`) — `latch` is 28, exactly where a
/// naive `12 + n` puts `lrf16`. [`Self::encoding`] is the only place that arithmetic is done.
///
/// ⛔ AND [`Self::LRF_COUNT`] IS WHAT AN INSTRUCTION FIELD MAY NAME, NOT WHAT A UNIT HAS. The
/// SFP/PE LRF holds sixteen on this target and the state file one; treating thirty-two as a file
/// depth hands out registers a unit does not have. What a unit has is that unit's own register-file
/// depth, which is a ProgIR question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Port {
    /// `none`.
    None,
    /// `zero` — the pseudo-unit supplying a zero operand.
    Zero,
    /// `one` — the pseudo-unit supplying a one operand.
    One,
    /// `two`.
    Two,
    /// `three`.
    Three,
    /// `west`.
    West,
    /// `north`.
    North,
    /// `east`.
    East,
    /// `south`.
    South,
    /// `xrf` — the PT's transposed register file.
    Xrf,
    /// `irf0`.
    Irf0,
    /// `irf1`.
    Irf1,
    /// `lrf<n>` — ⛔ THE INDEX IS BOUNDED BY THE TYPE; see [`LrfIndex`].
    Lrf(LrfIndex),
    /// `latch`.
    Latch,
    /// `opA` — ⭐ CAMEL-CASED IN THE IR, unlike every other case.
    OpA,
    /// `opB`.
    OpB,
    /// `opC`.
    OpC,
    /// `result` — the forwarding target meaning "this op's own result".
    Result,
    /// `pt`.
    Pt,
    /// `pe`.
    Pe,
    /// `sfp`.
    Sfp,
    /// `sfpring`.
    SfpRing,
    /// `lx`.
    Lx,
    /// `l0`.
    L0,
    /// `nfwd0`.
    Nfwd0,
    /// `nfwd2`.
    Nfwd2,
    /// `nbrslice`.
    NbrSlice,
    /// `istate<n>` — the internal state registers a compare may forward to. ⛔ Bounded; see
    /// [`IStateIndex`].
    IState(IStateIndex),
    /// `crossptnlink`.
    CrossPtNorthLink,
}

/// WHICH `lrf<n>` — ⛔ ONE VARIANT PER CASE THE `.td` DECLARES, and no way to write another.
///
/// ⛔⛔ THIS WAS A CHECKED CONSTRUCTOR OVER `Bounded<32>`, AND A CHECKED CONSTRUCTOR IS STILL A
/// RUNTIME REFUSAL — `LrfIndex::checked(200)` handed back `None` at run time and left every caller
/// with an `Option` to mishandle. Before that it was a bare `u8` behind an `assert!`. Both are gone:
/// the set is thirty-two cases, so it is thirty-two variants.
///
/// ⭐ AND THIS IS WHAT THE VENDOR DOES. `SentientTypes.td:108-160` writes thirty-two separate
/// `def SentientLRF<n> : I32EnumAttrCase<"lrf<n>", …>` lines. Enumerating them is transcription, not
/// verbosity.
///
/// ⛔ THE WIRE VALUES ARE **SPLIT** AND THE GAP IS OCCUPIED: 0..15 encode as `12 + n`, 16..31 as
/// `48 + (n - 16)`, and `latch` is 28 — exactly where a naive `12 + n` puts `lrf16`.
/// [`Port::encoding`] is the only place that arithmetic happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LrfIndex {
    /// `lrf0`.
    L0,
    /// `lrf1`.
    L1,
    /// `lrf2`.
    L2,
    /// `lrf3`.
    L3,
    /// `lrf4`.
    L4,
    /// `lrf5`.
    L5,
    /// `lrf6`.
    L6,
    /// `lrf7`.
    L7,
    /// `lrf8`.
    L8,
    /// `lrf9`.
    L9,
    /// `lrf10`.
    L10,
    /// `lrf11`.
    L11,
    /// `lrf12`.
    L12,
    /// `lrf13`.
    L13,
    /// `lrf14`.
    L14,
    /// `lrf15`.
    L15,
    /// `lrf16`.
    L16,
    /// `lrf17`.
    L17,
    /// `lrf18`.
    L18,
    /// `lrf19`.
    L19,
    /// `lrf20`.
    L20,
    /// `lrf21`.
    L21,
    /// `lrf22`.
    L22,
    /// `lrf23`.
    L23,
    /// `lrf24`.
    L24,
    /// `lrf25`.
    L25,
    /// `lrf26`.
    L26,
    /// `lrf27`.
    L27,
    /// `lrf28`.
    L28,
    /// `lrf29`.
    L29,
    /// `lrf30`.
    L30,
    /// `lrf31`.
    L31,
}

impl LrfIndex {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::L0 => 0,
            Self::L1 => 1,
            Self::L2 => 2,
            Self::L3 => 3,
            Self::L4 => 4,
            Self::L5 => 5,
            Self::L6 => 6,
            Self::L7 => 7,
            Self::L8 => 8,
            Self::L9 => 9,
            Self::L10 => 10,
            Self::L11 => 11,
            Self::L12 => 12,
            Self::L13 => 13,
            Self::L14 => 14,
            Self::L15 => 15,
            Self::L16 => 16,
            Self::L17 => 17,
            Self::L18 => 18,
            Self::L19 => 19,
            Self::L20 => 20,
            Self::L21 => 21,
            Self::L22 => 22,
            Self::L23 => 23,
            Self::L24 => 24,
            Self::L25 => 25,
            Self::L26 => 26,
            Self::L27 => 27,
            Self::L28 => 28,
            Self::L29 => 29,
            Self::L30 => 30,
            Self::L31 => 31,
        }
    }
}

/// WHICH `istate<n>` — ⛔ FOUR EXIST (`SentientTypes.td:143-146`), so four variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IStateIndex {
    /// `istate0`.
    S0,
    /// `istate1`.
    S1,
    /// `istate2`.
    S2,
    /// `istate3`.
    S3,
}

impl IStateIndex {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::S0 => 0,
            Self::S1 => 1,
            Self::S2 => 2,
            Self::S3 => 3,
        }
    }
}

impl Port {
    /// The spelling the attribute carries.
    ///
    /// ⛔ RETURNS `String`, NOT `&'static str`, because `Lrf` and `IState` are parameterised. Every
    /// other case is a literal.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            Self::None => "none".to_owned(),
            Self::Zero => "zero".to_owned(),
            Self::One => "one".to_owned(),
            Self::Two => "two".to_owned(),
            Self::Three => "three".to_owned(),
            Self::West => "west".to_owned(),
            Self::North => "north".to_owned(),
            Self::East => "east".to_owned(),
            Self::South => "south".to_owned(),
            Self::Xrf => "xrf".to_owned(),
            Self::Irf0 => "irf0".to_owned(),
            Self::Irf1 => "irf1".to_owned(),
            Self::Lrf(n) => format!("lrf{}", n.get()),
            Self::Latch => "latch".to_owned(),
            Self::OpA => "opA".to_owned(),
            Self::OpB => "opB".to_owned(),
            Self::OpC => "opC".to_owned(),
            Self::Result => "result".to_owned(),
            Self::Pt => "pt".to_owned(),
            Self::Pe => "pe".to_owned(),
            Self::Sfp => "sfp".to_owned(),
            Self::SfpRing => "sfpring".to_owned(),
            Self::Lx => "lx".to_owned(),
            Self::L0 => "l0".to_owned(),
            Self::Nfwd0 => "nfwd0".to_owned(),
            Self::Nfwd2 => "nfwd2".to_owned(),
            Self::NbrSlice => "nbrslice".to_owned(),
            Self::IState(n) => format!("istate{}", n.get()),
            Self::CrossPtNorthLink => "crossptnlink".to_owned(),
        }
    }

    /// THE WIRE VALUE.
    ///
    /// ⛔⛔ THE LRF SPLIT LIVES HERE AND NOWHERE ELSE: `lrf0..lrf15` are `12 + n`, `lrf16..lrf31`
    /// are `48 + (n - 16)`. `latch` is 28.
    ///
    /// ⛔ AND `pe` IS 35, NOT 34. `pt` is 33 and thirty-four is unassigned
    /// (`SentientTypes.td:131-132`) — a contiguous run would put `pe` on a value the enum does not
    /// define.
    ///
    /// ⭐ AND IT IS TOTAL — no assertion and no panic, because [`LrfIndex`] and [`IStateIndex`]
    /// admit no index this could not encode.
    #[must_use]
    pub const fn encoding(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Zero => 1,
            Self::One => 2,
            Self::Two => 3,
            Self::Three => 4,
            Self::West => 5,
            Self::North => 6,
            Self::East => 7,
            Self::South => 8,
            Self::Xrf => 9,
            Self::Irf0 => 10,
            Self::Irf1 => 11,
            Self::Lrf(n) => {
                // ⛔ THE SPLIT LIVES HERE AND NOWHERE ELSE. `latch` is 28, which is where a naive
                // `12 + n` puts `lrf16`.
                if n.get() < 16 {
                    12 + n.get() as u32
                } else {
                    48 + (n.get() as u32 - 16)
                }
            }
            Self::Latch => 28,
            Self::OpA => 29,
            Self::OpB => 30,
            Self::OpC => 31,
            Self::Result => 32,
            Self::Pt => 33,
            Self::Pe => 35,
            Self::Sfp => 36,
            Self::SfpRing => 37,
            Self::Lx => 38,
            Self::L0 => 39,
            Self::Nfwd0 => 40,
            Self::Nfwd2 => 41,
            Self::NbrSlice => 42,
            Self::IState(n) => 43 + n.get() as u32,
            Self::CrossPtNorthLink => 47,
        }
    }
}

/// WHICH REGISTER A SCALAR VALUE LIVES IN — `SentientRegTypeAttr`, spelled `reg_locale`.
///
/// ⛔ `unknown` IS THE DEFAULT AND MEANS UNASSIGNED, not "no register". `RegisterTypeAssignment`
/// (D66) is the pass that replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegType {
    /// `unknown` — not yet assigned.
    Unknown,
    /// `imm` — an immediate field rather than a register. The default a `scalar_constant` carries.
    Imm,
    /// `jcr` — the jump-condition register.
    Jcr,
    /// `lccr` — the loop-count/condition register a `for` bound lands in.
    Lccr,
    /// `lrf` — the local register file.
    Lrf,
    /// `xrfrdptr` — the XRF read pointer.
    XrfRdPtr,
    /// `xrfwrptr` — the XRF write pointer.
    XrfWrPtr,
    /// `lar`.
    Lar,
    /// `lbr` — ⭐ HOLDS AN INDEX, NOT AN ADDRESS, on the store side.
    Lbr,
    /// `ear`.
    Ear,
    /// `ebr`.
    Ebr,
    /// `gtr`.
    Gtr,
    /// `mvr` — the move/loop-count register `MVLOOPCNT` writes.
    Mvr,
    /// `unrelated`.
    Unrelated,
}

impl RegType {
    /// The spelling (`SentientTypes.td:254-267`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Imm => "imm",
            Self::Jcr => "jcr",
            Self::Lccr => "lccr",
            Self::Lrf => "lrf",
            Self::XrfRdPtr => "xrfrdptr",
            Self::XrfWrPtr => "xrfwrptr",
            Self::Lar => "lar",
            Self::Lbr => "lbr",
            Self::Ear => "ear",
            Self::Ebr => "ebr",
            Self::Gtr => "gtr",
            Self::Mvr => "mvr",
            Self::Unrelated => "unrelated",
        }
    }
}

/// FMA OR FNMS — `SentientFMAmodeAttr`, spelled `mode` (`SentientTypes.td:238-250`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FmaMode {
    /// `fused_mul_add` — the default.
    FusedMulAdd,
    /// `fused_neg_mul_sub`.
    FusedNegMulSub,
}

impl FmaMode {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::FusedMulAdd => "fused_mul_add",
            Self::FusedNegMulSub => "fused_neg_mul_sub",
        }
    }
}

/// HOW AN XRF POINTER MOVES — `SentientXRFAddmodeAttr` (`SentientTypes.td:293-307`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum XrfMode {
    /// `add` — advance by the increment.
    Add,
    /// `copy` — set outright.
    Copy,
}

impl XrfMode {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Copy => "copy",
        }
    }
}

/// WHICH OPERANDS A PE+SFP PAIR FOLDS — `SentientFoldModeAttr` (`SentientTypes.td:311-330`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FoldMode {
    /// `none`.
    None,
    /// `fold_A`.
    FoldA,
    /// `fold_B`.
    FoldB,
    /// `fold_AB_A`.
    FoldAbA,
    /// `fold_AB_B`.
    FoldAbB,
    /// `fold_AB_Both`.
    FoldAbBoth,
}

impl FoldMode {
    /// The spelling — ⭐ CAPITALISED AS THE `.td` WRITES IT (`fold_AB_Both`, not `fold_ab_both`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FoldA => "fold_A",
            Self::FoldB => "fold_B",
            Self::FoldAbA => "fold_AB_A",
            Self::FoldAbB => "fold_AB_B",
            Self::FoldAbBoth => "fold_AB_Both",
        }
    }
}

/// A `gcvt_imm<n>` A **BINARY** MAY TAKE — ⛔ FOUR CASES, so four variants.
///
/// ⛔⛔ NO CONSTRUCTOR, BECAUSE A CHECKED CONSTRUCTOR IS STILL A RUNTIME REFUSAL. This was first a
/// bare `u8` with the legal set in a `const` array beside it (a comment), then a
/// `checked(u8) -> Option` (a refusal). `gcvt_imm3` is now unwritable rather than rejected.
///
/// ⛔ AND **DISJOINT** FROM [`UnaryGcvt`]'s — same spelling, no shared value, which is why they are two
/// types and not one wider set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinaryGcvt {
    /// `gcvt_imm0`.
    Imm0,
    /// `gcvt_imm4`.
    Imm4,
    /// `gcvt_imm24`.
    Imm24,
    /// `gcvt_imm28`.
    Imm28,
}

impl BinaryGcvt {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Imm0 => 0,
            Self::Imm4 => 4,
            Self::Imm24 => 24,
            Self::Imm28 => 28,
        }
    }
}

/// AN `fcvt_imm<n>` A **BINARY** MAY TAKE — ⛔ four cases (`SentientTypes.td:349-352`), disjoint from
/// [`UnaryFcvt`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinaryFcvt {
    /// `fcvt_imm2`.
    Imm2,
    /// `fcvt_imm3`.
    Imm3,
    /// `fcvt_imm4`.
    Imm4,
    /// `fcvt_imm7`.
    Imm7,
}

impl BinaryFcvt {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Imm2 => 2,
            Self::Imm3 => 3,
            Self::Imm4 => 4,
            Self::Imm7 => 7,
        }
    }
}

/// WHICH `pack<n>` — ⛔⛔ **TEN AND ELEVEN DO NOT EXIST**.
///
/// The enum runs 0..9 and then 12..27 (`SentientTypes.td:368-388`), and a gap in the middle of a range
/// is exactly the shape a reader completes by hand without noticing. Twenty-six cases, twenty-six
/// variants, and no way to name the two that are absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackIndex {
    /// `pack0`.
    P0,
    /// `pack1`.
    P1,
    /// `pack2`.
    P2,
    /// `pack3`.
    P3,
    /// `pack4`.
    P4,
    /// `pack5`.
    P5,
    /// `pack6`.
    P6,
    /// `pack7`.
    P7,
    /// `pack8`.
    P8,
    /// `pack9`.
    P9,
    /// `pack12`.
    P12,
    /// `pack13`.
    P13,
    /// `pack14`.
    P14,
    /// `pack15`.
    P15,
    /// `pack16`.
    P16,
    /// `pack17`.
    P17,
    /// `pack18`.
    P18,
    /// `pack19`.
    P19,
    /// `pack20`.
    P20,
    /// `pack21`.
    P21,
    /// `pack22`.
    P22,
    /// `pack23`.
    P23,
    /// `pack24`.
    P24,
    /// `pack25`.
    P25,
    /// `pack26`.
    P26,
    /// `pack27`.
    P27,
}

impl PackIndex {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::P0 => 0,
            Self::P1 => 1,
            Self::P2 => 2,
            Self::P3 => 3,
            Self::P4 => 4,
            Self::P5 => 5,
            Self::P6 => 6,
            Self::P7 => 7,
            Self::P8 => 8,
            Self::P9 => 9,
            Self::P12 => 12,
            Self::P13 => 13,
            Self::P14 => 14,
            Self::P15 => 15,
            Self::P16 => 16,
            Self::P17 => 17,
            Self::P18 => 18,
            Self::P19 => 19,
            Self::P20 => 20,
            Self::P21 => 21,
            Self::P22 => 22,
            Self::P23 => 23,
            Self::P24 => 24,
            Self::P25 => 25,
            Self::P26 => 26,
            Self::P27 => 27,
        }
    }
}

/// HOW WIDE A `merge` MERGES — ⛔ FOUR WIDTHS, not any integer (`SentientTypes.td:353-360`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MergeWidth {
    /// `merge8l` / `merge8h`.
    W8,
    /// `merge16l` / `merge16h`.
    W16,
    /// `merge32l` / `merge32h`.
    W32,
    /// `merge64l` / `merge64h`.
    W64,
}

impl MergeWidth {
    /// The width as it is spelled in the mnemonic.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::W8 => 8,
            Self::W16 => 16,
            Self::W32 => 32,
            Self::W64 => 64,
        }
    }
}

/// A `gcvt_imm<n>` A **UNARY** MAY TAKE — ⛔ seven cases (`SentientTypes.td:630-636`).
///
/// ⛔⛔ DISJOINT FROM [`BinaryGcvt`]'s: binary takes 0/4/24/28, unary takes 1/2/5/6/8/16/17. Two types
/// sharing a spelling and no value, so neither can be handed the other's case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnaryGcvt {
    /// `gcvt_imm1`.
    Imm1,
    /// `gcvt_imm2`.
    Imm2,
    /// `gcvt_imm5`.
    Imm5,
    /// `gcvt_imm6`.
    Imm6,
    /// `gcvt_imm8`.
    Imm8,
    /// `gcvt_imm16`.
    Imm16,
    /// `gcvt_imm17`.
    Imm17,
}

impl UnaryGcvt {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Imm1 => 1,
            Self::Imm2 => 2,
            Self::Imm5 => 5,
            Self::Imm6 => 6,
            Self::Imm8 => 8,
            Self::Imm16 => 16,
            Self::Imm17 => 17,
        }
    }
}

/// AN `fcvt_imm<n>` A **UNARY** MAY TAKE — ⛔ four cases (`SentientTypes.td:637-640`), disjoint from
/// [`BinaryFcvt`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnaryFcvt {
    /// `fcvt_imm0`.
    Imm0,
    /// `fcvt_imm1`.
    Imm1,
    /// `fcvt_imm5`.
    Imm5,
    /// `fcvt_imm6`.
    Imm6,
}

impl UnaryFcvt {
    /// The number this case spells.
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::Imm0 => 0,
            Self::Imm1 => 1,
            Self::Imm5 => 5,
            Self::Imm6 => 6,
        }
    }
}

/// WHAT A `vector_binary` COMPUTES — `SentientBinaryOperatorAttr`, spelled `binaryOp`.
///
/// Fifty-eight cases (`SentientTypes.td:333-458`) in four families whose wire values are NOT one
/// run: the arithmetic and convert operators at 0..19, the merges and packs at 101..134, and the
/// float compares at 135..138.
///
/// ⛔ `and` AND `or` ARE SPELLED `and0` AND `or0` — the `.td` says why in as many words:
/// *"and and or are reserved keyword in c++"*. Emitting `and` produces an attribute the parser does
/// not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinaryOp {
    /// `and0` — ⛔ not `and`.
    And,
    /// `or0` — ⛔ not `or`.
    Or,
    /// `xnor`.
    Xnor,
    /// `and_not`.
    AndNot,
    /// `min`.
    Min,
    /// `max`.
    Max,
    /// `abs_min`.
    AbsMin,
    /// `abs_max`.
    AbsMax,
    /// `add`.
    Add,
    /// `mul`.
    Mul,
    /// `sub`.
    Sub,
    /// `mul_div2`.
    MulDiv2,
    /// `gcvt_imm<n>` — a general convert with an immediate.
    GcvtImm(BinaryGcvt),
    /// `fcvt_imm<n>` — a float convert with an immediate.
    FcvtImm(BinaryFcvt),
    /// `merge<w><half>`.
    Merge {
        /// The element width.
        width: MergeWidth,
        /// Whether it is the high half rather than the low.
        high: bool,
    },
    /// `pack<n>` — ⛔ 10 AND 11 DO NOT EXIST; see [`PackIndex`].
    Pack(PackIndex),
    /// `fcmp_neq`.
    CompareNeq,
    /// `fcmp_eq`.
    CompareEq,
    /// `fcmp_lt`.
    CompareLt,
    /// `fcmp_le`.
    CompareLe,
}

impl BinaryOp {
    /// The spelling the attribute carries.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            Self::And => "and0".to_owned(),
            Self::Or => "or0".to_owned(),
            Self::Xnor => "xnor".to_owned(),
            Self::AndNot => "and_not".to_owned(),
            Self::Min => "min".to_owned(),
            Self::Max => "max".to_owned(),
            Self::AbsMin => "abs_min".to_owned(),
            Self::AbsMax => "abs_max".to_owned(),
            Self::Add => "add".to_owned(),
            Self::Mul => "mul".to_owned(),
            Self::Sub => "sub".to_owned(),
            Self::MulDiv2 => "mul_div2".to_owned(),
            Self::GcvtImm(n) => format!("gcvt_imm{}", n.get()),
            Self::FcvtImm(n) => format!("fcvt_imm{}", n.get()),
            Self::Merge { width, high } => {
                format!("merge{}{}", width.bits(), if high { "h" } else { "l" })
            }
            Self::Pack(n) => format!("pack{}", n.get()),
            Self::CompareNeq => "fcmp_neq".to_owned(),
            Self::CompareEq => "fcmp_eq".to_owned(),
            Self::CompareLt => "fcmp_lt".to_owned(),
            Self::CompareLe => "fcmp_le".to_owned(),
        }
    }
}

/// THE SEVEN OPERATORS THAT MAY FORWARD A LOGICAL RESULT — and there is no eighth.
///
/// ⛔⛔ THE VERIFIER REFUSES ANY OTHER WITH A BARE `failure()`. `BinaryOp::verify`
/// (`SentientOps.cpp:2251-2262`) checks `logical_result_forwarding` against exactly
/// `fcmp_eq | fcmp_neq | fcmp_le | fcmp_lt | min | max | abs_max` and returns failure with **no
/// message**, so a wrong pairing surfaces as an unexplained refusal rather than a diagnostic.
///
/// ⛔⛔ AND A `debug_assert!` IN THE PRINTER WAS THE WRONG GUARD, WHICH IS WHAT THIS TYPE REPLACES.
/// It fired in a debug build and **vanished in release** — so the one configuration that reaches the
/// backend was the one with no check at all, and its punishment is a silent refusal. The pairing is
/// now the value: only [`Binary::Forwarding`] carries a forward, and only these seven can be named in
/// it.
///
/// ⛔ `abs_min` IS ABSENT WHILE `abs_max` IS PRESENT. The asymmetry is the reference's; a reader
/// completing the pair by hand writes a program the verifier rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ForwardingOp {
    /// `fcmp_eq`.
    CompareEq,
    /// `fcmp_neq`.
    CompareNeq,
    /// `fcmp_le`.
    CompareLe,
    /// `fcmp_lt`.
    CompareLt,
    /// `min`.
    Min,
    /// `max`.
    Max,
    /// `abs_max` — ⛔ AND NOT `abs_min`; see the type's note.
    AbsMax,
}

impl ForwardingOp {
    /// The same operator as a [`BinaryOp`], which is what the attribute spells.
    #[must_use]
    pub const fn as_binary(self) -> BinaryOp {
        match self {
            Self::CompareEq => BinaryOp::CompareEq,
            Self::CompareNeq => BinaryOp::CompareNeq,
            Self::CompareLe => BinaryOp::CompareLe,
            Self::CompareLt => BinaryOp::CompareLt,
            Self::Min => BinaryOp::Min,
            Self::Max => BinaryOp::Max,
            Self::AbsMax => BinaryOp::AbsMax,
        }
    }
}

/// WHAT A `vector_binary` COMPUTES, AND WHETHER IT FORWARDS A LOGICAL RESULT.
///
/// ⭐⭐ ONE VALUE, BECAUSE THE TWO FACTS ARE NOT INDEPENDENT. Holding the operator and an
/// `Option<Port>` side by side let the pairing be wrong; holding them together means an illegal one
/// cannot be written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    /// Any operator, forwarding no logical result — the common case.
    Plain(BinaryOp),
    /// One of the seven, forwarding its logical result to a port.
    Forwarding {
        /// Which of the seven.
        op: ForwardingOp,
        /// `$LogicalResultForwarding`.
        to: Port,
        /// `$unrollIncrLogicalResult`.
        unroll_incr: bool,
    },
}

impl Binary {
    /// Which operator this is, whichever arm it took.
    #[must_use]
    pub const fn op(self) -> BinaryOp {
        match self {
            Self::Plain(op) => op,
            Self::Forwarding { op, .. } => op.as_binary(),
        }
    }
}

/// WHAT A `vector_ternary` COMPUTES — one case (`SentientTypes.td:461-470`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TernaryOp {
    /// `select`.
    Select,
}

impl TernaryOp {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Select => "select",
        }
    }
}

/// WHAT A `vector_unary` COMPUTES — `SentientUnaryAttr`, spelled `unary_op`.
///
/// Twenty-seven cases (`SentientTypes.td:614-674`): the transcendental estimates, the reductions,
/// and the converts.
///
/// ⭐ THE ESTIMATES COME IN SLOPE/OFFSET PAIRS for sigmoid and tanh, and in A/B halves for exp —
/// one instruction each, not one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnaryOp {
    /// `exp_a`.
    ExpA,
    /// `exp_b`.
    ExpB,
    /// `rec` — reciprocal.
    Rec,
    /// `ln`.
    Ln,
    /// `rsqrt`.
    Rsqrt,
    /// `sigm_slope`.
    SigmSlope,
    /// `sigm_offset`.
    SigmOffset,
    /// `tanh_slope`.
    TanhSlope,
    /// `tanh_offset`.
    TanhOffset,
    /// `floor`.
    Floor,
    /// `reduction_add`.
    ReductionAdd,
    /// `reduction_min`.
    ReductionMin,
    /// `reduction_max`.
    ReductionMax,
    /// `reduction_abs_min`.
    ReductionAbsMin,
    /// `reduction_abs_max`.
    ReductionAbsMax,
    /// `fast_exp`.
    FastExp,
    /// `gcvt_imm<n>` — ⛔ A DIFFERENT SET FROM THE BINARY ONE'S; see [`UnaryGcvt`].
    GcvtImm(UnaryGcvt),
    /// `fcvt_imm<n>` — ⛔ disjoint from the binary set again; see [`UnaryFcvt`].
    FcvtImm(UnaryFcvt),
}

impl UnaryOp {
    /// The spelling the attribute carries.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            Self::ExpA => "exp_a".to_owned(),
            Self::ExpB => "exp_b".to_owned(),
            Self::Rec => "rec".to_owned(),
            Self::Ln => "ln".to_owned(),
            Self::Rsqrt => "rsqrt".to_owned(),
            Self::SigmSlope => "sigm_slope".to_owned(),
            Self::SigmOffset => "sigm_offset".to_owned(),
            Self::TanhSlope => "tanh_slope".to_owned(),
            Self::TanhOffset => "tanh_offset".to_owned(),
            Self::Floor => "floor".to_owned(),
            Self::ReductionAdd => "reduction_add".to_owned(),
            Self::ReductionMin => "reduction_min".to_owned(),
            Self::ReductionMax => "reduction_max".to_owned(),
            Self::ReductionAbsMin => "reduction_abs_min".to_owned(),
            Self::ReductionAbsMax => "reduction_abs_max".to_owned(),
            Self::FastExp => "fast_exp".to_owned(),
            Self::GcvtImm(n) => format!("gcvt_imm{}", n.get()),
            Self::FcvtImm(n) => format!("fcvt_imm{}", n.get()),
        }
    }
}

/// WHICH COMPARISON A `sentient.if` BRANCHES ON — `CmpIPredicateAttr` (`SentientTypes.td:474-489`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CmpPredicate {
    /// `eq`.
    Eq,
    /// `ne`.
    Ne,
    /// `slt` — signed less-than.
    Slt,
    /// `sle`.
    Sle,
    /// `sgt`.
    Sgt,
    /// `sge`.
    Sge,
}

impl CmpPredicate {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Slt => "slt",
            Self::Sle => "sle",
            Self::Sgt => "sgt",
            Self::Sge => "sge",
        }
    }
}

/// HOW FAR A COMPUTE IS UNROLLED — `SentientUnrollFactorAttr` (`SentientTypes.td:491-509`).
///
/// ⛔ THE SPELLING IS `x<n>` AND THE SET IS NOT EVERY POWER OF TWO: 1, 2, 3, 4, 8. Three exists
/// and the `.td` comments say it is *"used only for reduction"*, while eight is *"used for
/// non-reduction"*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnrollFactor {
    /// `x1` — the default.
    X1,
    /// `x2`.
    X2,
    /// `x3` — reduction only.
    X3,
    /// `x4`.
    X4,
    /// `x8` — non-reduction only.
    X8,
}

impl UnrollFactor {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::X1 => "x1",
            Self::X2 => "x2",
            Self::X3 => "x3",
            Self::X4 => "x4",
            Self::X8 => "x8",
        }
    }

    /// The factor as a number — what `getUnrollFactorVal` derives by parsing the spelling
    /// (`SentientOps.cpp:2266-2269`).
    #[must_use]
    pub const fn count(self) -> u32 {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X3 => 3,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }
}

/// HOW A LOAD OR STORE RESHAPES WHAT IT MOVES — `SentientShuffleModeAttr`
/// (`SentientTypes.td:511-537`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShuffleMode {
    /// `noshuffle` — the default.
    NoShuffle,
    /// `splat`.
    Splat,
    /// `rotate`.
    Rotate,
    /// `splat2b`.
    Splat2B,
    /// `splat4b`.
    Splat4B,
    /// `splat16b`.
    Splat16B,
    /// `zpad16b`.
    ZeroPad16B,
    /// `masked2b`.
    Masked2B,
    /// `masked16b`.
    Masked16B,
}

impl ShuffleMode {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::NoShuffle => "noshuffle",
            Self::Splat => "splat",
            Self::Rotate => "rotate",
            Self::Splat2B => "splat2b",
            Self::Splat4B => "splat4b",
            Self::Splat16B => "splat16b",
            Self::ZeroPad16B => "zpad16b",
            Self::Masked2B => "masked2b",
            Self::Masked16B => "masked16b",
        }
    }
}

/// WHICH HALF OF A SYNC THIS IS — `SentientSyncModeAttr` (`SentientTypes.td:539-554`).
///
/// ⭐⭐ `sendrecv` IS ONE INSTRUCTION, NOT TWO. `SyncSendRecvFusion` (inside `O2O3EarlyPasses`) is
/// the pass that fuses a matched pair into it, which is why a program that has not run that pass
/// carries only halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyncMode {
    /// `send`.
    Send,
    /// `recv`.
    Recv,
    /// `sendrecv` — a fused pair.
    SendRecv,
}

impl SyncMode {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Send => "send",
            Self::Recv => "recv",
            Self::SendRecv => "sendrecv",
        }
    }
}

/// A UNIT THAT MAY BE NAMED IN A SYNC — the bridge between the wire's [`UnitKind`] vocabulary and
/// this dialect's [`Consumer`] one.
///
/// ⭐⭐ TOTAL BY CONSTRUCTION, WHICH IS WHY IT IS A TRAIT AND NOT A FUNCTION. `DfirUnit` has members
/// with no `Consumer` counterpart at all (`hbm`, `lx`, the state files, the SFP ring), so a
/// `DfirUnit -> Consumer` function would have to return an `Option` — a refusal. `UnitKind` is sealed
/// to the four kinds that terminate a wire (`L3lu`, `Lxlu`, `Sfp`, `Lxsu`), and every one of those has
/// a `Consumer`, so stating the mapping per type makes it total with nothing to refuse.
pub trait SyncPeer: crate::islands::dataflow_ir::link::UnitKind {
    /// How this unit is named in a `sentient.sync`'s `units` attribute.
    const CONSUMER: Consumer;
}

impl SyncPeer for crate::islands::dataflow_ir::link::L3lu {
    const CONSUMER: Consumer = Consumer::L3lu;
}
impl SyncPeer for crate::islands::dataflow_ir::link::Lxlu {
    const CONSUMER: Consumer = Consumer::Lxlu;
}
impl SyncPeer for crate::islands::dataflow_ir::link::Sfp {
    const CONSUMER: Consumer = Consumer::Sfp;
}
impl SyncPeer for crate::islands::dataflow_ir::link::Lxsu {
    const CONSUMER: Consumer = Consumer::Lxsu;
}

/// ONE SIDE OF A SYNC RENDEZVOUS — the peer this unit signals and waits on.
///
/// # 🛑 BOTH SIDES, OR NEITHER
///
/// ⛔⛔ THE INNER `Consumer` IS PRIVATE AND ONLY [`rendezvous`] MINTS ONE, so a `sentient.sync` cannot
/// be written naming a peer that is not the other half of a real rendezvous. This is the same
/// discipline as the wire's [`SendEnd`]/[`RecvEnd`], applied to the sync — and it is needed here for the
/// same reason: `dataflow.sync_recv` is BLOCKING (*"it does not return until the matching signal has
/// been received"*, `Dataflow.td:209-212`), so a half whose peer never signals back is a unit that
/// waits forever.
///
/// ⛔ AND A SYNC OP HAS **NO OPERANDS** — its peers are an attribute (`SentientOps.td:876`) — which is
/// why this could not be locked down until the island held per-unit programs. With a flat `Vec<Op>`
/// there was no second program to put the other half in.
///
/// ⛔ THE ORDER WAS ALSO WRONG ONCE, AND IT WROTE ZEROS: both store-side syncs came out inverted
/// against dxp's on an otherwise byte-matching op. A pairing minted from one value cannot invert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncHalf(Consumer);

impl SyncHalf {
    /// The peer this side names.
    #[must_use]
    pub const fn peer(self) -> Consumer {
        self.0
    }

    /// THE PEER A **LOWERED** `dataflow.sync_send`/`sync_recv` NAMES — the destination that op
    /// already carried, not a new rendezvous.
    ///
    /// ⛔⛔ [`rendezvous`] CANNOT EXPRESS IT, AND THAT IS NOT A GAP IN THE PAIRING. It mints from
    /// [`SyncPeer`], whose four kinds are the ones that terminate a wire — so `lxlu0`, `lxlu1`,
    /// `lxsu0`, `lxsu1`, `l3lu` and `l3su` are unnameable through it, and those are exactly the
    /// consumers `lowerL3SyncOperationForAUnit` derives from the destination `get_unit`'s type and
    /// `corelet` (entries 223/224, via [`crate::bridges::dataflow_ir_to_sentient::dfs_dataflow_to_sentient::extend_unit_name_to_corelet`]).
    ///
    /// ⛔ AND "BOTH SIDES OR NEITHER" IS NOT A PROPERTY OF A LOWERED SYNC'S PEER LIST: the vendor's
    /// own key has an `l3su` unit's sync naming `l3su` among its peers
    /// (`dcc/test/L3SU/sync-op-l3su.mlir:13`). The pairing is established on the rung below, where
    /// the `sync_send` and its `sync_recv` are two ops; this is that op's lowering.
    #[must_use]
    pub const fn of_lowered_destination(dst: Consumer) -> SyncHalf {
        SyncHalf(dst)
    }
}

/// THE TWO SIDES OF ONE SYNC, ONCE.
///
/// ⛔ RETURNS BOTH OR NEITHER: `A`'s half names `B` and `B`'s half names `A`, from one call, so
/// getting them backwards is not expressible — the halves are typed by whose side they are only in
/// the sense that each carries the OTHER's consumer, which is what the op must state.
#[must_use]
pub fn rendezvous<A: SyncPeer, B: SyncPeer>() -> (SyncHalf, SyncHalf) {
    (SyncHalf(B::CONSUMER), SyncHalf(A::CONSUMER))
}

/// WHICH UNIT CONSUMES A LOAD, OR PARTICIPATES IN A SYNC — `SentientConsumerAttr`
/// (`SentientTypes.td:556-596`), whose C++ name is `SentientLoadConsumer`.
///
/// ⛔ THIS IS A **DIFFERENT** UNIT VOCABULARY FROM [`Port`]'s. Sixteen cases naming L0/LX halves
/// and their numbered instances (`lxlu0`, `lxluN`, ...), where `Port` names `lx` and `l0` whole.
/// Reading one where the other belongs is what made a send name its own unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Consumer {
    /// `sfp`.
    Sfp,
    /// `pe`.
    Pe,
    /// `l0`.
    L0,
    /// `pt`.
    Pt,
    /// `l0lu`.
    L0lu,
    /// `l0su`.
    L0su,
    /// `lxlu`.
    Lxlu,
    /// `lxluN`.
    LxluN,
    /// `lxlu0`.
    Lxlu0,
    /// `lxlu1`.
    Lxlu1,
    /// `lxsu`.
    Lxsu,
    /// `lxsuN`.
    LxsuN,
    /// `lxsu0`.
    Lxsu0,
    /// `lxsu1`.
    Lxsu1,
    /// `l3lu`.
    L3lu,
    /// `l3su`.
    L3su,
}

impl Consumer {
    /// The spelling — ⭐ `lxluN`/`lxsuN` KEEP THEIR CAPITAL N.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Sfp => "sfp",
            Self::Pe => "pe",
            Self::L0 => "l0",
            Self::Pt => "pt",
            Self::L0lu => "l0lu",
            Self::L0su => "l0su",
            Self::Lxlu => "lxlu",
            Self::LxluN => "lxluN",
            Self::Lxlu0 => "lxlu0",
            Self::Lxlu1 => "lxlu1",
            Self::Lxsu => "lxsu",
            Self::LxsuN => "lxsuN",
            Self::Lxsu0 => "lxsu0",
            Self::Lxsu1 => "lxsu1",
            Self::L3lu => "l3lu",
            Self::L3su => "l3su",
        }
    }
}

/// HOW A `splat` PADS — `SentientSplatPadAttr` (`SentientTypes.td:598-611`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplatPad {
    /// `none`.
    None,
    /// `left`.
    Left,
}

impl SplatPad {
    /// The spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Left => "left",
        }
    }
}

/// WHICH WAY A SEND IS ROUTED AROUND THE RING — `SentientRoutingDirectionAttr`, spelled `dir`
/// (`SentientTypes.td:676-693`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoutingDirection {
    /// `PseudoRandom`.
    PseudoRandom,
    /// `CounterClockwise`.
    CounterClockwise,
    /// `Clockwise`.
    Clockwise,
    /// `BothWays`.
    BothWays,
}

impl RoutingDirection {
    /// The spelling — ⭐ CAMEL-CASED, unlike every other enum in this dialect.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::PseudoRandom => "PseudoRandom",
            Self::CounterClockwise => "CounterClockwise",
            Self::Clockwise => "Clockwise",
            Self::BothWays => "BothWays",
        }
    }
}

/// WHICH REGISTER WITHIN ITS FILE — `regIndex`.
///
/// ⛔⛔ THIS WAS `Option<i32>`, WHICH ADMITTED `Some(-5)` AND `Some(9999)`. The `.td` writes the
/// unassigned case as `-1` (`DefaultValuedAttr<I32Attr, "-1">`), so a signed field is how the
/// *reference* spells absence — but carrying that spelling into Rust makes every negative expressible
/// while only one of them means anything. `Option<RegIndex>` says the same thing with nothing else
/// sayable.
///
/// ⛔ BOUNDED BY `kMaxCompRegs` = 128 (`progir.h:508-509`), the width of the
/// `std::bitset<kMaxCompRegs>` in `RegDefs` and therefore a hard cap rather than a convention.
///
/// ⚠️ THE PER-FILE DEPTH IS TIGHTER AND IS AN ARCH FACT — the SFP/PE LRF holds sixteen on this target
/// and the state file one. This is the ISA's ceiling, not permission to use 127 of a sixteen-deep
/// file; that question belongs to whatever assigns the register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegIndex(Bounded<{ sys_arch_spec::progir::MAX_REGISTERS_PER_UNIT as u32 }>);

impl RegIndex {
    /// A REGISTER INDEX, CHECKED WHERE IT IS WRITTEN.
    ///
    /// ⛔⛔ THE ONLY CONSTRUCTOR, AND IT TAKES THE INDEX AS A CONST GENERIC. `Bounded::at` asserts in
    /// a `const { }` block, so `RegIndex::at::<200>()` is a **build error** — not a `None` a caller
    /// might unwrap, and not a panic at print time. A `checked(u32) -> Option` stood here and was
    /// removed: a refusal at run time is still a run-time failure, just a politer one.
    ///
    /// ⚠️ SO AN INDEX MUST BE A LITERAL AT ITS CONSTRUCTION SITE. That is consistent with this crate —
    /// everything is a compile-time constant, and a register chosen at expansion is a literal by the
    /// time it is written down. A genuinely computed index has no constructor here on purpose: if one
    /// is ever needed, the bound belongs to the file's own depth (an arch fact, tighter than this
    /// ceiling) and that is a design decision, not something to paper over with an `Option`.
    #[must_use]
    pub const fn at<const I: u32>() -> RegIndex {
        RegIndex(Bounded::at::<I>())
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// HOW MANY ENTRIES OF A MASK ARE VALID — `samv`'s `numvalidentry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidEntries(pub u32);

/// WHICH CROSS-SLICE SLICE — `samv`'s `sliceid_xsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceId(pub u32);

/// THE WITHIN-SLICE LENGTH — `samv`'s `wsllen`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WslLen(pub u32);

/// `samv`'s `precision` — ⛔⛔ A **RAW ISA FIELD**, NOT A [`Precision`].
///
/// `SentientOps.td:991` declares it `I32Attr`, the one place in the dialect where a precision is an
/// untyped integer rather than the enum. A newtype keeps it from being handed the enum's encoding, and
/// keeps it from being swapped with [`ValidEntries`], [`SliceId`] or [`WslLen`] — four adjacent
/// integers on one op, which is exactly the transposition the crate's newtype rule exists for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RawPrecision(pub u32);

/// WHERE ONE SCALAR LIVES — `regLocale` PLUS `regIndex`.
///
/// ⭐⭐ THE `.td` DECLARES THEM AS A PAIR, on every op that has them (`scalar_add`, `scalar_sub`,
/// `scalar_copy`, `load_and_send`, `receive_and_store`, `load_compute_and_send`,
/// `receive_and_extract_scalar`), and as PARALLEL ARRAYS on the ops that have several
/// (`SentientRegTypeArrayAttr:$regLocales` beside `I32ArrayAttr:$regIndices`). Carrying them together
/// is what the declaration says they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    /// `regLocale`.
    pub locale: RegType,
    /// `regIndex` — ⛔ `None` is the `.td`'s `-1`, unassigned. An ABSENCE, not a refusal.
    pub index: Option<RegIndex>,
}

/// ONE VALUE A LOOP CARRIES — its initial value, the value it yields, and where it lives.
///
/// # 🛑 FIVE PARALLEL VECTORS WERE FIVE FACTS TRUSTED TO LINE UP
///
/// ⛔⛔ `Op::For` HELD `init_args`, `results`, `reg_locales`, `reg_indices` AND `program_header` AS
/// FIVE INDEPENDENT `Vec`s. Each is one entry per carried value, and nothing said they were the same
/// length — a loop carrying two values with three locales and one index was constructible, and the
/// printed `regLocales`/`regIndices` arrays would then disagree with the iter-operand list about how
/// many values the loop even has.
///
/// ⭐ THIS IS THE PRODUCER-CONSUMER LOCKDOWN GENERALISED. `link.rs`'s point is that *a pairing is not
/// two facts that agree*; a carried value is five facts that agree, and the fix is the same one — make
/// it a single value, so there is nothing to disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Carried {
    /// One of `$initArgs` — what enters the loop.
    pub init: Val,
    /// THE BODY ARGUMENT THIS VALUE ARRIVES AS — what the ops inside the loop actually read.
    ///
    /// ⛔⛔ WITHOUT IT THE LOOP'S OWN INTERFACE IS UNADDRESSABLE. `getRegionIterArgs()` is
    /// `getBody()->getArguments().drop_front(1)` (`SentientOps.td:100-102`), and `getXrfValue`
    /// (entry 090) answers an xrf pointer carried by a loop with exactly
    /// `cast<sentient::ForOp>(xrf_ptr).getBody()->getArgument(1 + idx)`
    /// (`Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:252-254`) — the
    /// argument, NOT the initial value and NOT the result. A `Carried` holding only `init` and
    /// `result` makes that answer inexpressible, and substituting `init` for it hands the caller a
    /// value defined OUTSIDE the loop: every iteration would then read the pointer the loop started
    /// with instead of the one the previous iteration advanced.
    ///
    /// ⭐ AND IT IS THE FIELD `affine::Carried` ALREADY HAS — see
    /// [`crate::islands::dataflow_ir::dialects::affine::Carried::arg`]. The two rungs describe the
    /// same loop interface, so they carry the same three values.
    pub arg: Val,
    /// The matching result — what leaves it.
    pub result: Val,
    /// Where it lives, from `$regLocales` and `$regIndices` at this position.
    pub reg: Reg,
    /// This position's `$programHeader` flag.
    pub program_header: bool,
}

/// ONE VALUE A `sentient.if` REGION YIELDS.
///
/// ⛔ THE SAME DEFECT AS [`Carried`], one field smaller: an `if` has results and register arrays but
/// no initial values, and those three were three parallel `Vec`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yielded {
    /// The value the region yields.
    pub result: Val,
    /// Where it lives.
    pub reg: Reg,
}

/// ONE COMPUTE OPERAND'S WHOLE DESCRIPTION — the six attributes that repeat per operand.
///
/// ⭐⭐ A STRUCT BECAUSE THE `.td` REPEATS IT VERBATIM, NOT BECAUSE IT READS TIDIER. `vector_mac`
/// declares `opA`/`opAForwarding`/`opAPrecision`/`opADataID`/`opAPortID`/`unrollIncrOpA` and then
/// the same six for B and C (`SentientOps.td:232-288`); `vector_binary` declares them for A and B,
/// `vector_unary` for A alone. Thirty attributes on one op are six facts about five operands.
///
/// ⛔ `data_id` AND `port_id` DEFAULT TO **MINUS ONE**, WHICH MEANS UNASSIGNED. `PortAssignment`
/// (D30) sets them, and its own ordering note says it must run before op re-rolling, unrolling and
/// splitting, *"all of which duplicate the data ids it reads"*. Zero is a real port; absence is not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    /// `op<X>` — where the value comes from.
    pub port: Port,
    /// `op<X>Forwarding` — every port this operand is also forwarded to. Empty is the common case.
    pub forwarding: Vec<Port>,
    /// `op<X>Precision` — defaults to [`Precision::Fp16`].
    pub precision: Precision,
    /// `op<X>DataID` — ⛔ `None` is the `.td`'s `-1`, meaning unassigned, NOT data id zero.
    pub data_id: Option<i32>,
    /// `op<X>PortID` — ⛔ `None` is `-1`, unassigned.
    pub port_id: Option<i32>,
    /// `unrollIncrOp<X>` — whether this operand advances per unrolled copy.
    pub unroll_incr: bool,
}

impl Operand {
    /// AN OPERAND READ STRAIGHT FROM A PORT, with every derived attribute left at its default.
    ///
    /// ⭐ THE DEFAULTS ARE THE `.td`'S OWN, not conveniences: fp16 precision, no forwarding, no
    /// assigned data or port id, no unroll increment.
    #[must_use]
    pub fn from(port: Port) -> Operand {
        Operand {
            port,
            forwarding: Vec::new(),
            precision: Precision::Fp16,
            data_id: None,
            port_id: None,
            unroll_incr: false,
        }
    }
}

/// WHERE A COMPUTE'S RESULT GOES — the operand bundle's counterpart, with no source port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultPorts {
    /// `ResultForwarding` — every port the result is written to.
    ///
    /// ⭐⭐ THIS IS WHERE A COMPUTE'S DESTINATION LIVES. An FMA's destination is its
    /// `ResultForwarding`, not a separate store — which is why a compute writing to a register file
    /// emits no store op of its own.
    pub forwarding: Vec<Port>,
    /// `ResultPrecision` — defaults to [`Precision::Fp16`].
    pub precision: Precision,
    /// `unrollIncrResult`.
    pub unroll_incr: bool,
}

impl Default for ResultPorts {
    fn default() -> ResultPorts {
        ResultPorts {
            forwarding: Vec::new(),
            precision: Precision::Fp16,
            unroll_incr: false,
        }
    }
}

/// WHERE A `sentient.receive_and_store` TAKES ITS DATA FROM — `producer_info.second->getResult(0)`
/// (`Helper.cpp:2089`) and the one thing that is not a unit.
///
/// ⛔⛔ A `vectorchain.constant_bitstream` PRODUCER IS REPLACED BY A `sentient.scalar_constant` AND
/// THE STORE READS **THAT** (`Helper.cpp:2069-2088`), so this operand is not always a wire end. It is
/// L3LU-only — every other component reaches `emitError("ConstantBitstreamOp producers are only
/// supported in L3")` — which is why the two cases are told apart here rather than collapsed to a
/// bare [`Val`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreSource {
    /// The wire's receive end, paired with a send by construction.
    Wire(RecvEnd),
    /// The `sentient.scalar_constant` the bitstream became.
    Constant(Val),
}

impl StoreSource {
    /// The value the `$producer` operand names.
    #[must_use]
    pub const fn val(self) -> Val {
        match self {
            StoreSource::Wire(end) => end.val(),
            StoreSource::Constant(val) => val,
        }
    }
}

/// HOW MANY ELEMENTS A TRANSFER MOVES AND HOW WIDE THEY ARE — the attributes every transfer shares.
///
/// ⭐ A STRUCT FOR THE SAME REASON AS [`Operand`]: `load_and_send`, `receive_and_store`,
/// `load_and_store` and `load_compute_and_send` each declare `total_elements` and `element_size`,
/// and the transfer family's other counts hang off them.
/// ⛔⛔ EVERY FIELD IS A NEWTYPE, AND `total_elements` AGAINST `element_size` IS EXACTLY WHY. The two
/// are adjacent, both small integers, and mean different things in different units — a COUNT against
/// a WIDTH. The first version of this struct held five bare `u32`s, where swapping the first two
/// compiles and describes a 64-byte transfer of 2 elements instead of a 2-byte transfer of 64.
/// The crate's rule says it: *transposing two extents must be E0308*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// `total_elements` — HOW MANY.
    pub total_elements: Elements,
    /// `element_size` — HOW WIDE ONE IS.
    ///
    /// ⛔⛔ **BITS**, AND IT WAS TYPED `Bytes` UNTIL THIS BATCH. Every `element_size` the reference
    /// prints is a bit width — `memref<32xi32>` gives `element_size = 32` and `memref<64xf16>` gives
    /// `element_size = 16` in one function (`dcc/test/Conversion/AgenToSentient/l3-gather.mlir:37`,
    /// `:56`) — and `constructLoadAndStoreStmt` hands it `getElementWidth()`, which
    /// [`AccessDetailsBase`](crate::bridges::dataflow_ir_to_sentient::agen_access_details::AccessDetailsBase)
    /// already types [`Bits`]. A bit width in a `Bytes` field is exactly the confusion this island's
    /// newtypes exist to make an E0308.
    pub element_size: Bits,
    /// `chunk_size` — defaults to one element.
    pub chunk_size: Elements,
    /// `chunk_stride` — defaults to one element.
    pub chunk_stride: Elements,
    /// `burst_size` — defaults to zero, meaning unbursted.
    pub burst_size: Elements,
}

impl Extent {
    /// A TRANSFER OF `total_elements` ELEMENTS EACH `element_size` WIDE, unchunked and unbursted —
    /// the `.td`'s own defaults (`SentientOps.td:504-520`).
    #[must_use]
    pub const fn of(total_elements: Elements, element_size: Bits) -> Extent {
        Extent {
            total_elements,
            element_size,
            chunk_size: Elements(1),
            chunk_stride: Elements(1),
            burst_size: Elements(0),
        }
    }
}

/// ONE `sentient.*` OPERATION — all twenty-nine the dialect declares.
///
/// ⛔ NO `_` ARM ANYWHERE THIS IS MATCHED. A thirtieth operation must be a build error, not a
/// silent fall-through — which is how three statement kinds once reached the emitter classified and
/// printed by nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    // ───────────────────────── control flow ─────────────────────────
    /// `sentient.for` — a counted loop (`SentientOps.td:47`).
    ///
    /// ⛔ THE BOUND IS AN SSA VALUE, NOT A LITERAL, and `regLocales`/`regIndices` are ARRAYS: one
    /// entry per iter arg, because each carried value needs its own register.
    For {
        /// THE INDUCTION VARIABLE — ⛔ THE REGION'S FIRST ARGUMENT, and the first thing the op prints.
        ///
        /// ⛔⛔ IT IS SYNTAX, NOT AN OPERAND. `ForOp::print` opens with
        /// `p << " " << op.getInductionVar() << " = " << op.getBound()`
        /// (`Dialect/Sentient/SentientOps.cpp:1000`), so `sentient.for %arg1 = %2` names the
        /// induction variable and then the trip count — and the reference's own expectations are
        /// exactly that: `sentient.for %[[VAL_11:.*]] = %[[VAL_10]] {..}{`
        /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:25`). This field was
        /// absent and the printer put the RESULT LIST where the induction variable goes, which made a
        /// loop carrying nothing print a leading ` = ` and a loop carrying values print its results
        /// twice.
        ///
        /// ⛔ AND `getMaskValueForPT` (entry 089) IDENTIFIES A DYNAMIC MASK BY IT. The mask parameter
        /// must be `arith.subi %bound, %iv` of the enclosing loop —
        /// `lhs != rhs_parent.getBound() || rhs != rhs_parent.getInductionVar()` is the refusal
        /// (`VectorChainToSentientPT/Helper.cpp:160`) — so a loop with no nameable induction variable
        /// cannot be the parent of any mask this pipeline accepts.
        iv: Val,
        /// `$bound` — the trip count.
        bound: Val,
        /// THE VALUES THE LOOP CARRIES — ⛔ ONE ENTRY EACH, so the iter-operand list and the register
        /// arrays cannot disagree about how many there are. See [`Carried`].
        carried: Vec<Carried>,
        /// `$dbgName`.
        dbg_name: Option<String>,
        /// The body.
        ///
        /// ⛔ THE UNION TYPE, NOT THIS DIALECT'S. A region on this rung holds ops of whatever dialects
        /// have reached it — the module is mixed all the way down — so typing it `Vec<sentient::Op>`
        /// would make a surviving `agen.composite_load_and_store` inside a `sentient.for`
        /// inexpressible, which is a shape a real granite program actually has.
        body: Vec<super::Op>,
    },

    /// `sentient.if` — a two-operand comparison and its region (`SentientOps.td:159`).
    If {
        /// `$predicate`.
        predicate: CmpPredicate,
        /// `$lhs`.
        lhs: Val,
        /// `$rhs`.
        rhs: Val,
        /// THE VALUES THE REGION YIELDS — ⛔ ONE ENTRY EACH. See [`Yielded`].
        yielded: Vec<Yielded>,
        /// `$dbgName`.
        dbg_name: Option<String>,
        /// The `then` body — ⛔ the union type; see [`Op::For`]'s body.
        then_body: Vec<super::Op>,
        /// The `else` body, where there is one. ⭐ EMPTY MEANS NO `else` REGION AT ALL, which is what
        /// `printRegion` is skipped for (`SentientOps.cpp:1310-1315`).
        else_body: Vec<super::Op>,
    },

    /// `sentient.yield` — what a region hands back (`SentientOps.td:32`).
    Yield {
        /// `$results`.
        results: Vec<Val>,
    },

    // ───────────────────────── compute ─────────────────────────
    /// `sentient.vector_mac` — the multiply-accumulate, with three operands and the XRF pointers
    /// (`SentientOps.td:232`).
    ///
    /// ⛔ THE POINTER ORDER IS WRITE THEN READ, and the `.td` says so in a comment because the
    /// operand list cannot: *"the order of pointers is xrfWritePtr and xrfReadPtr"*. Swapping them
    /// reads the block being written.
    VectorMac {
        /// `$mask` — optional on this op alone among the computes.
        mask: Option<Val>,
        /// `$pointers` — ⛔ WRITE POINTER FIRST, then read.
        xrf_write_ptr: Option<Val>,
        /// The read pointer.
        xrf_read_ptr: Option<Val>,
        /// The values it binds.
        results: Vec<Val>,
        /// Operand A.
        op_a: Operand,
        /// Operand B.
        op_b: Operand,
        /// Operand C — the accumulator.
        op_c: Operand,
        /// Where the result goes.
        result: ResultPorts,
        /// `$mode` — FMA or FNMS.
        mode: FmaMode,
        /// `$ComputePrecision` — ⛔ DISTINCT FROM EVERY OPERAND'S, and from the result's.
        compute_precision: Precision,
        /// `$fold_mode`.
        fold_mode: Option<FoldMode>,
        /// `$unrollFactor`.
        unroll_factor: UnrollFactor,
        /// `$xrfReadIncr`.
        xrf_read_incr: u32,
        /// `$xrfWriteIncr`.
        xrf_write_incr: u32,
        /// `DataTransferOnly` — a DISCARDABLE attribute rather than a `.td` field, set on the dummy
        /// MACs the dangling-op lowering emits (`VectorChainToSentientPESFP.cpp:1341`).
        ///
        /// ⛔ AND `PortAssignment` READS IT (`PortAssignment.cpp:518`, `:690`) to give such a MAC its
        /// ports without treating it as a compute, then removes the attribute (`:703`) — so a dummy
        /// MAC that lost this flag is port-assigned as if it computed something.
        data_transfer_only: bool,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.vector_binary` — two operands and an operator (`SentientOps.td:321`).
    VectorBinary {
        /// `$mask` — ⛔ REQUIRED HERE, unlike on [`Op::VectorMac`].
        mask: Val,
        /// Operand A.
        op_a: Operand,
        /// Operand B.
        op_b: Operand,
        /// `$binaryOp`, and its logical forward where it has one — ⛔ ONE VALUE; see [`Binary`].
        binary_op: Binary,
        /// Where the result goes.
        result: ResultPorts,
        /// `$ComputePrecision`.
        compute_precision: Precision,
        /// `$fold_mode`.
        fold_mode: Option<FoldMode>,
        /// `$unrollFactor`.
        unroll_factor: UnrollFactor,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.vector_unary` — one operand and an estimate, reduction or convert
    /// (`SentientOps.td:289`).
    VectorUnary {
        /// `$mask` — required.
        mask: Val,
        /// Operand A.
        op_a: Operand,
        /// `$unary_op`.
        unary_op: UnaryOp,
        /// Where the result goes.
        result: ResultPorts,
        /// `$ComputePrecision`.
        compute_precision: Precision,
        /// `$fold_mode`.
        fold_mode: Option<FoldMode>,
        /// `$unrollFactor`.
        unroll_factor: UnrollFactor,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.vector_ternary` — three operands and a select (`SentientOps.td:362`).
    VectorTernary {
        /// `$mask` — optional.
        mask: Option<Val>,
        /// Operand A.
        op_a: Operand,
        /// Operand B.
        op_b: Operand,
        /// Operand C.
        op_c: Operand,
        /// `$ternaryOp`.
        ternary_op: TernaryOp,
        /// Where the result goes.
        result: ResultPorts,
        /// `$ComputePrecision`.
        compute_precision: Precision,
        /// `$fold_mode`.
        fold_mode: Option<FoldMode>,
        /// `$unrollFactor`.
        unroll_factor: UnrollFactor,
        /// `$unrollIncrLogicalResult`.
        unroll_incr_logical_result: bool,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    // ───────────────────────── transfers ─────────────────────────
    /// `sentient.load` — a plain read (`SentientOps.td:404`). One of the ten ops whose syntax the
    /// `.td` declares outright: `$src_val attr-dict`.
    Load {
        /// `$src_val`.
        src: Val,
        /// The extents.
        extent: Extent,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
    },

    /// `sentient.load_and_send` — read from a view and put it on the wire (`SentientOps.td:504`).
    ///
    /// ⛔⛔ TWO ADDRESSES, NOT ONE. `mutable_addr` is the double-buffer's toggling half and
    /// `immutable_addr` the fixed base; the pair is what `MutableStartAddrShifting` (D10) and
    /// `MutableAddrSplitting` (D11) exist to produce. Collapsing them loses the buffer switch.
    LoadAndSend {
        /// `$mutable_addr`.
        mutable_addr: Val,
        /// `$immutable_addr`.
        immutable_addr: Val,
        /// `$increment`.
        increment: Val,
        /// `$consumer` — ⛔ THE WIRE'S SEND END, not a unit handle. See the module note.
        consumer: SendEnd,
        /// The value it binds.
        result: Val,
        /// The extents.
        extent: Extent,
        /// `$interleaved_group` — `group_size` at the two construction sites
        /// (`Helper.cpp:1991`, `:2126`), which is a count of elements like its `burst_size` twin.
        interleaved_group: Elements,
        /// `$rotate_val`.
        rotate_val: Option<u32>,
        /// `$dir`.
        dir: Option<RoutingDirection>,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
        /// Where the scalar lives.
        reg: Reg,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.receive_and_store` — take from the wire and write a view
    /// (`SentientOps.td:546`).
    ReceiveAndStore {
        /// `$mutable_addr`.
        mutable_addr: Val,
        /// `$immutable_addr`.
        immutable_addr: Val,
        /// `$increment`.
        increment: Val,
        /// `$producer` — the wire's receive end, or the constant that replaced a bitstream.
        producer: StoreSource,
        /// The value it binds.
        result: Val,
        /// `$dst`.
        dst: Option<Val>,
        /// `$drop_first`.
        drop_first: Option<Val>,
        /// `$multicast_info`.
        multicast_info: Option<Val>,
        /// The extents.
        extent: Extent,
        /// `$interleaved_group` — `group_size` at the two construction sites
        /// (`Helper.cpp:1991`, `:2126`), which is a count of elements like its `burst_size` twin.
        interleaved_group: Elements,
        /// `$coalesce`.
        coalesce: bool,
        /// `$subword_length`.
        subword_length: u32,
        /// `$stride`.
        stride: u32,
        /// `$permute`.
        permute: bool,
        /// `$shuffle_mode` — ⛔ DOUBLY OPTIONAL in the `.td`.
        shuffle_mode: Option<ShuffleMode>,
        /// Where the scalar lives.
        reg: Reg,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.load_and_store` — a transfer with both ends in one op, which is what an
    /// `agen.composite_load_and_store` becomes (`SentientOps.td:715`).
    ///
    /// ⛔ FOUR ADDRESSES: a mutable/immutable pair per end.
    LoadAndStore {
        /// `$src`.
        src: Val,
        /// `$dst`.
        dst: Val,
        /// `$src_mutable_addr`.
        src_mutable_addr: Val,
        /// `$src_immutable_addr`.
        src_immutable_addr: Val,
        /// `$src_inc`.
        src_inc: Val,
        /// `$dst_mutable_addr`.
        dst_mutable_addr: Val,
        /// `$dst_immutable_addr`.
        dst_immutable_addr: Val,
        /// `$dst_inc`.
        dst_inc: Val,
        /// `$multicast_info`.
        multicast_info: Option<Val>,
        /// The values it binds — source then destination.
        results: (Val, Val),
        /// The extents.
        extent: Extent,
        /// `$stride` — ⛔ **SIGNED**, AND IT WAS `u32` UNTIL THIS BATCH. The `.td` declares
        /// `DefaultValuedAttr<I32Attr, "1">:$stride` and the builder takes `"int":$stride`
        /// (`SentientOps.td:739`, `:756`), and `constructLoadAndStoreStmt` hands it `stride_step`,
        /// whose sign IS the direction of the adjustment
        /// ([`StrideStep`](crate::bridges::dataflow_ir_to_sentient::agen_agen_to_sentient::StrideStep)).
        stride: i32,
        /// `$rotate_val`.
        rotate_val: Option<u32>,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
        /// The SOURCE end's register — ⛔ TWO NAMED FIELDS, NOT TWO PARALLEL ARRAYS. This op has
        /// exactly two ends, so a `Vec` of locales beside a `Vec` of indices admitted a length
        /// mismatch and admitted no way to say which entry was the source's.
        src_reg: Reg,
        /// The DESTINATION end's register.
        dst_reg: Reg,
        /// `$dir`.
        dir: Option<RoutingDirection>,
        /// `"is-ibr-write"` — the DESTINATION is an L3 image-buffer-read unit.
        ///
        /// ⛔ NOT AN OPERAND AND NOT IN THE `.td` ARGUMENT LIST: a discardable attribute the
        /// conversion stamps on afterwards, named by `getIsIBRWriteAttrStrName()`
        /// (`SentientOps.td:796`) and printed hyphenated and therefore quoted
        /// (`dcc/test/Conversion/AgenToSentient/l3-gather.mlir:37`).
        is_ibr_write: bool,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.load_compute_and_send` — a load that scales on the way out
    /// (`SentientOps.td:419`).
    ///
    /// ⛔ SOURCE AND DESTINATION EXTENTS ARE SEPARATE (`src_total_elements` against
    /// `dst_total_elements`), because the compute in the middle may change the width.
    LoadComputeAndSend {
        /// `$mutable_addr`.
        mutable_addr: Val,
        /// `$immutable_addr`.
        immutable_addr: Val,
        /// `$increment`.
        increment: Val,
        /// `$element_index`.
        element_index: Val,
        /// `$scale_index`.
        scale_index: Val,
        /// `$consumer` — ⛔ the wire's send end.
        consumer: SendEnd,
        /// The value it binds.
        result: Val,
        /// `$src_total_elements` — ⛔ A COUNT.
        src_total_elements: Elements,
        /// `$dst_total_elements` — ⛔ SEPARATE FROM THE SOURCE'S, because the compute may change the
        /// width.
        dst_total_elements: Elements,
        /// `$src_element_size` — ⛔ A WIDTH IN **BITS**, not a count and not bytes; see
        /// [`Extent::element_size`]. `getElementTypeBitWidth()` is what fills it
        /// (`Helper.cpp:3735`), and `SentientOps.td`'s own example spells `src_element_size = 4`
        /// for a four-bit view.
        src_element_size: Bits,
        /// `$dst_element_size` — ⛔ a width in **bits**, from the compute result's element type.
        dst_element_size: Bits,
        /// `$dir`.
        dir: Option<RoutingDirection>,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
        /// Where the scalar lives.
        reg: Reg,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.load_and_extract_scalar` — a load whose value also lands in a scalar register
    /// (`SentientOps.td:606`). ⭐ BINDS TWO VALUES: the address and the datum.
    LoadAndExtractScalar {
        /// `$mutable_addr`.
        mutable_addr: Val,
        /// `$immutable_addr`.
        immutable_addr: Val,
        /// `$increment`.
        increment: Val,
        /// `$consumer` — ⛔ the wire's send end.
        consumer: SendEnd,
        /// `$addr` — the address it binds.
        addr_result: Val,
        /// `$data` — the datum it binds.
        data_result: Val,
        /// `$total_elements` — ⛔ A COUNT.
        total_elements: Elements,
        /// `$element_size` — ⛔ A WIDTH IN **BITS**; see [`Extent::element_size`].
        element_size: Bits,
        /// The register the ADDRESS lands in — ⛔ named, not an array position.
        addr_reg: Reg,
        /// The register the DATUM lands in.
        data_reg: Reg,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.receive_and_extract_scalar` — take one datum off the wire into a register
    /// (`SentientOps.td:675`).
    ReceiveAndExtractScalar {
        /// `$unit` — ⛔ THE WIRE'S RECEIVE END; this op drains a wire like any other receive.
        unit: RecvEnd,
        /// `$position` — which object to extract.
        position: Val,
        /// The value it binds.
        result: Val,
        /// Where the scalar lives.
        reg: Reg,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    // ───────────────────────── scalars ─────────────────────────
    /// `sentient.scalar_add` (`SentientOps.td:700`). Syntax declared in the `.td`.
    ///
    /// # ⛔⛔ THE REGISTER IS AN `Option` BECAUSE A LOWERING LEAVES IT UNSAID
    ///
    /// `regLocale` and `regIndex` are `DefaultValuedAttr`s (`SentientOps.td:703-704`), and
    /// `AddOp::create(builder, loc, type, lhs, rhs)` passes neither — so the op the lowering emits
    /// carries no register attributes at all and `attr-dict` prints nothing:
    ///
    /// ```text
    /// %12 = sentient.scalar_sub %5, %11 : index, index
    /// %29 = sentient.scalar_add %18, %1 : index, index
    /// ```
    ///
    /// (`dcc/test/Conversion/StandardToSentient/cmpi_select_different_BB.mlir:19,58`, the output of
    /// `--dcc-standard-to-sentient` alone.) Once `registerManagementPasses` (D64-D75) has decided,
    /// the same op prints them:
    ///
    /// ```text
    /// %20 = sentient.scalar_add %17, %1 {element_size = 8 : i32, regIndex = 1 : i32, regLocale = #sentient<reg_type lrf>} : index, index
    /// ```
    ///
    /// (`dcc/test/LXLU/rotate-composite.mlir:24`.) ⛔ A NON-OPTIONAL `Reg` COULD NOT SPELL THE
    /// FIRST: `RegType::Unknown` is a locale the reference also writes *explicitly*
    /// (`regLocale = #sentient<reg_type unknown>` on the same test's `load_and_send`), so "unknown"
    /// and "unsaid" are two states and collapsing them loses the one every lowering produces.
    ScalarAdd {
        /// `$inp1`.
        lhs: Val,
        /// `$inp2`.
        rhs: Val,
        /// `$out`.
        result: Val,
        /// Where the scalar lives — ⛔ `None` UNTIL AN ALLOCATOR SAYS. See [`Op::ScalarAdd`]'s note.
        reg: Option<Reg>,
        /// The type of both operands and of the result — `SameOperandsAndResultType`.
        ty: ScalarTy,
    },

    /// `sentient.scalar_sub` (`SentientOps.td:801`).
    ScalarSub {
        /// `$inp1`.
        lhs: Val,
        /// `$inp2`.
        rhs: Val,
        /// `$out`.
        result: Val,
        /// Where the scalar lives — ⛔ `None` until an allocator says.
        reg: Option<Reg>,
        /// The type of both operands and of the result.
        ty: ScalarTy,
    },

    /// `sentient.scalar_mul` (`SentientOps.td:816`).
    ///
    /// ⛔ NO `regIndex` ON THIS ONE. `scalar_add` and `scalar_sub` declare both `regLocale` and
    /// `regIndex`; `scalar_mul` declares only `regLocale` (`SentientOps.td:816-829`). The asymmetry
    /// is the reference's.
    ScalarMul {
        /// `$inp1`.
        lhs: Val,
        /// `$inp2`.
        rhs: Val,
        /// `$out`.
        result: Val,
        /// `$regLocale` — ⛔ `None` until an allocator says; see [`Op::ScalarAdd`].
        reg_locale: Option<RegType>,
        /// The type of both operands and of the result.
        ty: ScalarTy,
    },

    /// `sentient.scalar_copy` (`SentientOps.td:830`).
    ScalarCopy {
        /// `$inp`.
        input: Val,
        /// `$out`.
        result: Val,
        /// Where the scalar lives.
        reg: Reg,
        /// `$programHeader`.
        program_header: bool,
    },

    /// `sentient.scalar_constant` — an immediate (`SentientOps.td:848`).
    ///
    /// ⛔ THE VALUE IS **SIGNED** (`SI64Attr`) and the default locale is [`RegType::Imm`], not
    /// `unknown` — a constant is an instruction field until something spills it to a register.
    ScalarConstant {
        /// `$value`.
        value: i64,
        /// `$out`.
        result: Val,
        /// `$regLocale` — ⛔ CARRIED AND NEVER PRINTED. `ConstantOp` has
        /// `hasCustomAssemblyFormat` and its printer emits the value and the result type only,
        /// with no attribute dictionary unless the op is a symbol
        /// (`SentientOps.cpp:1698-1715`), which is why the reference's own output shows
        /// `sentient.scalar_constant {value = 0 : si64} : index` and never a locale. The attribute
        /// is still on the op for the passes that read it.
        reg_locale: RegType,
        /// `$out`'s type — ⭐ PRINTED, AND IT VARIES. `{value = 0 : si64} : index` sits beside
        /// `{value = 0 : si64} : i1` in one function
        /// (`dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir:6-8`).
        ty: ScalarTy,
        /// The `is_symbol` UNIT ATTRIBUTE — ⛔ NOT DECLARED IN THE `.td`, AND STILL LOAD-BEARING.
        /// `createSentientConstants` (entry 235) copies it off the `constant_bitstream` it
        /// replaces (`Splat.cpp:50-51`), `ConstructProgIRHelper.cpp:4354` reads it to decide an
        /// immediate is a symbol to be resolved later, and `ConstantOp::print` switches to the
        /// whole attribute dictionary when it is set (`SentientOps.cpp:1700-1714`) — so it changes
        /// what this op prints, not just what a later pass thinks.
        is_symbol: bool,
    },

    /// `sentient.vector_constant` — a whole vector of immediates (`SentientOps.td:866`).
    VectorConstant {
        /// `$value` — the elements.
        value: Vec<i64>,
        /// `$out`.
        result: Val,
        /// `$out`'s type — ⭐ PRINTED, AND IT IS NOT DERIVABLE FROM `value.len()` ALONE: the
        /// element type decides the length (`128 / bitwidth`, `Splat.cpp:56`), and
        /// `VectorConstantOp::print` writes it out (`SentientOps.cpp:1747`).
        ty: Vector,
    },

    // ───────────────────────── masks, sync, ports ─────────────────────────
    /// `sentient.sync` — ⛔ NO OPERANDS AT ALL, only attributes (`SentientOps.td:876`). Which units
    /// it synchronises is the `units` attribute, not an operand list.
    Sync {
        /// `$mode`.
        mode: SyncMode,
        /// `$units` — ⛔ HALVES OF RENDEZVOUS, NOT BARE CONSUMERS. See [`SyncHalf`]: a peer can only
        /// be named here if the matching half exists, so a signal with nobody waiting is unwritable.
        peers: Vec<SyncHalf>,
        /// `$soft`.
        soft: bool,
        /// `$implicit_sync_memory_boundary` — ⛔ `None` is the `.td`'s `-1`.
        implicit_sync_memory_boundary: Option<i32>,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.nop` (`SentientOps.td:895`). What `NOPInsertionForBackToBackSyncs` inserts.
    Nop {
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.set_send_dst` — ⭐ WHAT `SetSendDestinationRE` (D33) WRITES
    /// (`SentientOps.td:904`).
    ///
    /// ⛔ ITS OPERAND IS A SEND END TOO. The op exists to say where subsequent sends go, so it names
    /// a consumer and is subject to the same pairing as the sends themselves.
    SetSendDst {
        /// `$units`.
        units: SendEnd,
    },

    /// `sentient.logical_port` — binds a port name as a value (`SentientOps.td:920`).
    LogicalPort {
        /// `$portName`.
        port_name: Port,
        /// `$port` — the value it binds.
        result: Val,
    },

    /// `sentient.splat` — broadcast one value across a vector (`SentientOps.td:933`).
    Splat {
        /// `$input`.
        input: Val,
        /// `$output`.
        output: Val,
        /// `$mask`.
        mask: Val,
        /// `$pad`.
        pad: SplatPad,
        /// `$precision`.
        precision: Precision,
        /// `$programHeader`.
        program_header: bool,
        /// `$unrollFactor`.
        unroll_factor: UnrollFactor,
        /// `$unrollIncrResult`.
        unroll_incr_result: bool,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.samv` — set the active mask value (`SentientOps.td:984`).
    ///
    /// ⛔ `$precision` HERE IS A PLAIN `I32Attr`, NOT A [`Precision`] ENUM
    /// (`SentientOps.td:991`) — the one place in the dialect where a precision is an untyped
    /// integer, so it must not be given the enum's spelling.
    Samv {
        /// `$mask_value`.
        mask_value: Val,
        /// `$maskall`.
        mask_all: bool,
        /// `$numvalidentry`.
        num_valid_entry: ValidEntries,
        /// `$sliceid_xsl`.
        slice_id_xsl: SliceId,
        /// `$xslinner`.
        xsl_inner: bool,
        /// `$wsllen`.
        wsl_len: WslLen,
        /// `$precision` — ⛔ A RAW ISA FIELD; see [`RawPrecision`].
        precision: RawPrecision,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.set_mask` (`SentientOps.td:1038`).
    SetMask {
        /// `$mask_value`.
        mask_value: Val,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.incrmask` — ⛔ NO OPERANDS (`SentientOps.td:1059`).
    IncrMask {
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.opaque` — the `.smc`-body escape hatch (`SentientOps.td:963`).
    ///
    /// ⭐ THE REGISTER DICTIONARIES ARE THE WHOLE POINT: an opaque body names its registers by
    /// symbol and the dictionaries bind those symbols to real ones, split read-write from
    /// read-only.
    ///
    /// ⛔⛔ THE SAME TYPES AS THE RUNG BELOW'S `dataflow.opaque`, NOT STRINGS. This is one escape
    /// hatch appearing on two rungs, so `func_name` is [`OpaqueFunc`] and a register binds to a
    /// [`RegAddr`] — the crate's rule is *no strings from the ddl/smc parsers; every closed set is a
    /// generated enum*, and an empty `String` here once satisfied the reference's `StringAttr` check
    /// and then substituted an empty operand into the instruction.
    Opaque {
        /// `$func_name`.
        func: OpaqueFunc,
        /// `$read_write_register_dictionary` — the body's own scratch.
        read_write: Vec<(RegName, RegAddr)>,
        /// `$read_only_register_dictionary` — caller-bound.
        read_only: Vec<(RegName, RegAddr)>,
        /// `$parameter_dictionary`.
        params: Vec<(ParamKey, ParamValue)>,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },
}

/// THE ATTRIBUTES EVERY COMPUTE SHARES — the tail of each one's dictionary.
///
/// ⭐ A STRUCT SO [`compute_attrs`] TAKES THREE ARGUMENTS RATHER THAN SEVEN, which is the difference
/// between a total function and one needing an `#[allow(clippy::too_many_arguments)]` this crate does
/// not permit.
struct ComputeShared<'a> {
    /// Where the result goes.
    result: &'a ResultPorts,
    /// `$ComputePrecision` — ⛔ DISTINCT FROM EVERY OPERAND'S AND FROM THE RESULT'S.
    compute_precision: Precision,
    /// `$fold_mode`.
    fold_mode: Option<FoldMode>,
    /// `$unrollFactor`.
    unroll_factor: UnrollFactor,
    /// `$dbgName`.
    dbg_name: &'a Option<String>,
}

/// THE VALUES ONE `sentient.*` OP **READS**, AS PLACES THAT CAN BE WRITTEN — `getOpOperands()`.
///
/// # 🛑 THIS EXISTS SO THAT `replaceAllUsesWith` CAN BE PERFORMED, NOT APPROXIMATED
///
/// ⛔⛔ `replaceAndEraseDummyMacOps` (entry 093) IS TWO RAUWs AND TWO ERASES AND NOTHING ELSE
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681-688`). The dummy
/// `sentient.mac` inserted by `insertDummyMacOp` (`:312-327`) is a placeholder standing where the
/// real compute's xrf pointer result will be, and every op downstream of it — the next
/// `sentient.add` in the pointer chain, the loop's `sentient.yield`, the real mac's own
/// `pointers(..)` list — was wired to the placeholder. Without a way to WRITE an operand, the
/// placeholder survives into the emitted program and the real mac's pointers are read by nobody.
///
/// ⛔ SO IT IS TOTAL OVER THE TWENTY-NINE, WITH NO WILDCARD. A thirtieth operation must be a build
/// error here: an op left out would be an op whose uses a rewrite silently fails to update, and a
/// partial rewrite is worse than none — it leaves two values live where the reference leaves one.
///
/// ⭐ OPERANDS ONLY. What an op BINDS is [`results`]; what its region binds is [`block_args`]. The
/// three lists are disjoint, which is what makes a use-walk and a def-walk answer different
/// questions about the same value.
#[must_use]
pub fn operands_mut(op: &mut Op) -> Vec<&mut Val> {
    let mut reads: Vec<&mut Val> = Vec::new();
    match op {
        // ⛔ THE INDUCTION VARIABLE IS NOT AN OPERAND and neither is a carried value's `arg` or
        // `result`: the first two are the region's arguments and the third is what the loop binds.
        // Only the trip count and the initialisers are read, and the initialisers are read OUTSIDE
        // the loop — which is why an `iter_args` value can be rewritten without touching the body.
        Op::For {
            iv: _,
            bound,
            carried,
            dbg_name: _,
            body: _,
        } => {
            reads.push(bound);
            reads.extend(carried.iter_mut().map(|carried| &mut carried.init));
        }
        // ⭐ THE PREDICATE IS AN ATTRIBUTE, the two compared values are operands. `yielded`'s
        // `result` is what the op binds.
        Op::If { lhs, rhs, .. } => reads.extend([lhs, rhs]),
        // ⛔⛔ `sentient.yield`'s `results` FIELD IS ITS **OPERAND LIST**, and the `.td` is why the
        // field is called that (`SentientOps.td:32`, `$results` under `let arguments`). A terminator
        // binds nothing; it reads what the region hands back — and this is the very list
        // `updateYieldArgs` (entry 175) appends an xrf pointer to.
        Op::Yield { results } => reads.extend(results.iter_mut()),
        // ⛔ THE POINTERS ARE OPERANDS AND THE POINTERS ARE WHAT ENTRY 093 REWIRES. `results` is what
        // this op binds — the advanced write pointer then the advanced read pointer, in that order.
        Op::VectorMac {
            mask,
            xrf_write_ptr,
            xrf_read_ptr,
            ..
        } => {
            reads.extend(mask.iter_mut());
            reads.extend(xrf_write_ptr.iter_mut());
            reads.extend(xrf_read_ptr.iter_mut());
        }
        // ⭐ THE MASK IS THE ONLY SSA OPERAND OF THE OTHER THREE COMPUTES. Their A/B/C operands are
        // PORTS — `opA = #sentient<compute_port zero>` — which are attributes, not values; see
        // [`Operand`].
        Op::VectorBinary { mask, .. } | Op::VectorUnary { mask, .. } => reads.push(mask),
        Op::VectorTernary { mask, .. } => reads.extend(mask.iter_mut()),
        Op::Load { src, .. } => reads.push(src),
        Op::LoadAndSend {
            mutable_addr,
            immutable_addr,
            increment,
            ..
        }
        | Op::LoadAndExtractScalar {
            mutable_addr,
            immutable_addr,
            increment,
            ..
        } => reads.extend([mutable_addr, immutable_addr, increment]),
        Op::ReceiveAndStore {
            mutable_addr,
            immutable_addr,
            increment,
            dst,
            drop_first,
            multicast_info,
            ..
        } => {
            reads.extend([mutable_addr, immutable_addr, increment]);
            reads.extend(dst.iter_mut());
            reads.extend(drop_first.iter_mut());
            reads.extend(multicast_info.iter_mut());
        }
        Op::LoadAndStore {
            src,
            dst,
            src_mutable_addr,
            src_immutable_addr,
            src_inc,
            dst_mutable_addr,
            dst_immutable_addr,
            dst_inc,
            multicast_info,
            ..
        } => {
            reads.extend([
                src,
                dst,
                src_mutable_addr,
                src_immutable_addr,
                src_inc,
                dst_mutable_addr,
                dst_immutable_addr,
                dst_inc,
            ]);
            reads.extend(multicast_info.iter_mut());
        }
        Op::LoadComputeAndSend {
            mutable_addr,
            immutable_addr,
            increment,
            element_index,
            scale_index,
            ..
        } => {
            reads.extend([
                mutable_addr,
                immutable_addr,
                increment,
                element_index,
                scale_index,
            ]);
        }
        Op::ReceiveAndExtractScalar { position, .. } => reads.push(position),
        Op::ScalarAdd { lhs, rhs, .. }
        | Op::ScalarSub { lhs, rhs, .. }
        | Op::ScalarMul { lhs, rhs, .. } => reads.extend([lhs, rhs]),
        Op::ScalarCopy { input, .. } => reads.push(input),
        // ⛔ ALL THREE OF `sentient.splat`'s VALUES ARE ARGUMENTS, `output` INCLUDED
        // (`SentientOps.td:949-951` declares `input`, `output` and `mask` under `let arguments` and
        // the op has no results at all). Treating `output` as a binding would make a splat the
        // definition of a value some other op already defines.
        Op::Splat {
            input,
            output,
            mask,
            ..
        } => reads.extend([input, output, mask]),
        Op::Samv { mask_value, .. } | Op::SetMask { mask_value, .. } => reads.push(mask_value),
        // ⭐ A LITERAL, A PORT NAME AND A BARRIER READ NOTHING. `sentient.sync`'s peers are an
        // attribute (`SentientOps.td:876`), `set_send_dst`'s units likewise, and `sentient.opaque`
        // names its registers by NAME and ADDRESS rather than by value.
        Op::ScalarConstant { .. }
        | Op::VectorConstant { .. }
        | Op::Sync { .. }
        | Op::Nop { .. }
        | Op::SetSendDst { .. }
        | Op::LogicalPort { .. }
        | Op::IncrMask { .. }
        | Op::Opaque { .. } => {}
    }
    reads
}

/// THE VALUES ONE `sentient.*` OP **BINDS AS RESULTS** — what `getResult(n)` answers.
///
/// ⛔⛔ THE ORDER IS THE `.td`'S, AND TWO PORTS READ IT BY POSITION. `getXrfValue` (entry 090) takes
/// `getResult(0 + idx)` of a loop's parent op and `replaceAndEraseDummyMacOps` (entry 093) takes
/// `getResult(0)` and `getResult(1)` of a real `sentient.mac` — the WRITE pointer then the READ
/// pointer, the same order the `pointers(..)` operand list uses (`SentientOps.td:234`). A list built
/// in any other order would swap the two pointers of every fused compute.
///
/// ⛔ RESULTS ONLY: a region argument is defined by no op, which is what makes
/// `Value::getDefiningOp()` null for one. Those are [`block_args`].
#[must_use]
pub fn results(op: &Op) -> Vec<Val> {
    match op {
        // ⭐ ONE RESULT PER CARRIED VALUE, and none for a loop that carries nothing.
        Op::For { carried, .. } => carried.iter().map(|carried| carried.result).collect(),
        Op::If { yielded, .. } => yielded.iter().map(|yielded| yielded.result).collect(),
        Op::VectorMac { results, .. } => results.clone(),
        Op::LoadAndStore { results, .. } => vec![results.0, results.1],
        Op::LoadAndExtractScalar {
            addr_result,
            data_result,
            ..
        } => vec![*addr_result, *data_result],
        Op::LoadAndSend { result, .. }
        | Op::ReceiveAndStore { result, .. }
        | Op::LoadComputeAndSend { result, .. }
        | Op::ReceiveAndExtractScalar { result, .. }
        | Op::ScalarAdd { result, .. }
        | Op::ScalarSub { result, .. }
        | Op::ScalarMul { result, .. }
        | Op::ScalarCopy { result, .. }
        | Op::ScalarConstant { result, .. }
        | Op::VectorConstant { result, .. }
        | Op::LogicalPort { result, .. } => vec![*result],
        // ⛔ NONE OF THESE BINDS ANYTHING — `let results` is absent from all nine
        // (`SentientOps.td`). `sentient.yield`'s `results` field is its OPERAND list; `splat`'s
        // `output` is an argument.
        Op::Yield { .. }
        | Op::VectorBinary { .. }
        | Op::VectorUnary { .. }
        | Op::VectorTernary { .. }
        | Op::Load { .. }
        | Op::Sync { .. }
        | Op::Nop { .. }
        | Op::SetSendDst { .. }
        | Op::Splat { .. }
        | Op::Samv { .. }
        | Op::SetMask { .. }
        | Op::IncrMask { .. }
        | Op::Opaque { .. } => Vec::new(),
    }
}

/// THE VALUES ONE `sentient.*` OP'S REGION **BINDS AS ARGUMENTS** — what no op defines.
///
/// ⛔⛔ THE INDUCTION VARIABLE IS ARGUMENT **0** AND THE CARRIED VALUES FOLLOW. `getInductionVar()`
/// is `getBody()->getArgument(0)` and `getRegionIterArgs()` drops exactly that one front argument
/// (`SentientOps.td:90-102`) — which is why `getXrfValue` reads a loop-carried xrf pointer as
/// `getArgument(1 + idx)` (`LoweringXRF.cpp:252-254`). An off-by-one here hands out the trip
/// counter where a pointer was asked for.
#[must_use]
pub fn block_args(op: &Op) -> Vec<Val> {
    match op {
        Op::For { iv, carried, .. } => {
            let mut args = vec![*iv];
            args.extend(carried.iter().map(|carried| carried.arg));
            args
        }
        // ⛔ A `sentient.if` REGION TAKES NO ARGUMENTS. It has no `initArgs` — only results — so
        // there is no argument list to drop a front element from.
        Op::If { .. } => Vec::new(),
        Op::Yield { .. }
        | Op::VectorMac { .. }
        | Op::VectorBinary { .. }
        | Op::VectorUnary { .. }
        | Op::VectorTernary { .. }
        | Op::Load { .. }
        | Op::LoadAndSend { .. }
        | Op::ReceiveAndStore { .. }
        | Op::LoadAndStore { .. }
        | Op::LoadComputeAndSend { .. }
        | Op::LoadAndExtractScalar { .. }
        | Op::ReceiveAndExtractScalar { .. }
        | Op::ScalarAdd { .. }
        | Op::ScalarSub { .. }
        | Op::ScalarMul { .. }
        | Op::ScalarCopy { .. }
        | Op::ScalarConstant { .. }
        | Op::VectorConstant { .. }
        | Op::Sync { .. }
        | Op::Nop { .. }
        | Op::SetSendDst { .. }
        | Op::LogicalPort { .. }
        | Op::Splat { .. }
        | Op::Samv { .. }
        | Op::SetMask { .. }
        | Op::IncrMask { .. }
        | Op::Opaque { .. } => Vec::new(),
    }
}

/// THE REGIONS ONE `sentient.*` OP HOLDS, in the order `getRegions()` indexes them.
///
/// ⭐ ONLY TWO OPS OF THIS DIALECT HAVE ANY, and their bodies hold [`super::Op`] — the whole mixed
/// rung, not just this dialect. See [`Op::For`]'s `body`.
#[must_use]
pub fn regions(op: &Op) -> Vec<&[super::Op]> {
    match op {
        Op::For { body, .. } => vec![body.as_slice()],
        // ⭐ `then` FIRST, and an empty `else` is NO REGION rather than an empty one — the
        // distinction the printer skips a region for (`SentientOps.cpp:1310-1315`).
        Op::If {
            then_body,
            else_body,
            ..
        } => vec![then_body.as_slice(), else_body.as_slice()],
        // ⛔ NO `_` ARM: a thirtieth operation with a region must be a build error here, not an op
        // whose body every walk quietly skips.
        Op::Yield { .. }
        | Op::VectorMac { .. }
        | Op::VectorBinary { .. }
        | Op::VectorUnary { .. }
        | Op::VectorTernary { .. }
        | Op::Load { .. }
        | Op::LoadAndSend { .. }
        | Op::ReceiveAndStore { .. }
        | Op::LoadAndStore { .. }
        | Op::LoadComputeAndSend { .. }
        | Op::LoadAndExtractScalar { .. }
        | Op::ReceiveAndExtractScalar { .. }
        | Op::ScalarAdd { .. }
        | Op::ScalarSub { .. }
        | Op::ScalarMul { .. }
        | Op::ScalarCopy { .. }
        | Op::ScalarConstant { .. }
        | Op::VectorConstant { .. }
        | Op::Sync { .. }
        | Op::Nop { .. }
        | Op::SetSendDst { .. }
        | Op::LogicalPort { .. }
        | Op::Splat { .. }
        | Op::Samv { .. }
        | Op::SetMask { .. }
        | Op::IncrMask { .. }
        | Op::Opaque { .. } => Vec::new(),
    }
}

/// THE REGIONS ONE `sentient.*` OP HOLDS, AS PLACES THAT CAN BE WRITTEN — [`regions`]'s counterpart.
///
/// ⛔ A REWRITE HAS TO DESCEND. The xrf pointer chain runs THROUGH loops — the placeholder a
/// `sentient.yield` inside a `sentient.for` reads is the one entry 093 rewires — so a rewrite that
/// stopped at the top level would leave every nested use pointing at an erased op.
#[must_use]
pub fn regions_mut(op: &mut Op) -> Vec<&mut Vec<super::Op>> {
    match op {
        Op::For { body, .. } => vec![body],
        Op::If {
            then_body,
            else_body,
            ..
        } => vec![then_body, else_body],
        // ⛔ NO `_` ARM — see [`regions`].
        Op::Yield { .. }
        | Op::VectorMac { .. }
        | Op::VectorBinary { .. }
        | Op::VectorUnary { .. }
        | Op::VectorTernary { .. }
        | Op::Load { .. }
        | Op::LoadAndSend { .. }
        | Op::ReceiveAndStore { .. }
        | Op::LoadAndStore { .. }
        | Op::LoadComputeAndSend { .. }
        | Op::LoadAndExtractScalar { .. }
        | Op::ReceiveAndExtractScalar { .. }
        | Op::ScalarAdd { .. }
        | Op::ScalarSub { .. }
        | Op::ScalarMul { .. }
        | Op::ScalarCopy { .. }
        | Op::ScalarConstant { .. }
        | Op::VectorConstant { .. }
        | Op::Sync { .. }
        | Op::Nop { .. }
        | Op::SetSendDst { .. }
        | Op::LogicalPort { .. }
        | Op::Splat { .. }
        | Op::Samv { .. }
        | Op::SetMask { .. }
        | Op::IncrMask { .. }
        | Op::Opaque { .. } => Vec::new(),
    }
}

/// ONE `sentient` OP AS TEXT. The caller has already indented.
///
/// ⛔⛔ THE ATTRIBUTE DICTIONARY IS **ALPHABETICAL**, because MLIR's `printOptionalAttrDict` sorts it
/// and every one of the nineteen custom printers delegates to that (`SentientOps.cpp:2219-2229`, and
/// the same three lines for ternary and unary). Writing attributes in declaration order produces text
/// that parses but never matches a reference dump byte for byte, which is the only oracle this rung
/// has.
///
/// ⛔ AND THE FOUR COMPUTE OPS PRINT ONLY THEIR MASK — every port, precision, forwarding list and
/// unroll flag is in the dictionary, not the operand list.
///
/// ⭐ TOTAL: every one of the twenty-nine variants is written here, so there is no
/// `unimplemented!()` and no arm that can be reached without a printer.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        // ───────────────────────── control flow ─────────────────────────
        Op::Yield { results } => {
            if results.is_empty() {
                let _ = writeln!(out, "sentient.yield");
            } else {
                let _ = writeln!(out, "sentient.yield {}", print::vals(results));
            }
        }
        // `p << " " << inductionVar << " = " << bound`, then the iter-operand types, the dict, and
        // the region (`SentientOps.cpp:998-1010`).
        Op::For {
            iv,
            bound,
            carried,
            dbg_name,
            body,
        } => {
            // ⭐ ONE WALK OVER `carried` PRODUCES ALL FOUR RENDERINGS, so the iter-operand list, the
            // result list and the two register arrays are the same length by construction.
            let iter_types = if carried.is_empty() {
                String::new()
            } else {
                let tys: Vec<&str> = carried.iter().map(|_| "index").collect();
                format!(" -> ({})", tys.join(", "))
            };
            let results: Vec<Val> = carried.iter().map(|c| c.result).collect();
            let mut attrs = vec![
                attr("regLocales", &locale_array(carried.iter().map(|c| c.reg))),
                attr("regIndices", &index_array(carried.iter().map(|c| c.reg))),
            ];
            if carried.iter().any(|c| c.program_header) {
                attrs.push(attr(
                    "programHeader",
                    &bool_array(carried.iter().map(|c| c.program_header)),
                ));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let bound_val = print::val(*bound);
            // ⭐ `printInitializationList` PAIRS THE BODY ARGUMENT WITH ITS INITIAL VALUE, under the
            // prefix ` iter_args` and only when there is one (`SentientOps.cpp:984-996, 1002`):
            // `iter_args(%28 = %0, %29 = %25)`
            // (`dcc/test/L3SU/dyn_node_e2e.mlir:79` writes exactly that form). The `= ` between them
            // is why [`Carried::arg`] has to exist for this line to be writable at all.
            let inits = if carried.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> = carried
                    .iter()
                    .map(|c| format!("{} = {}", print::val(c.arg), print::val(c.init)))
                    .collect();
                format!(" iter_args({})", pairs.join(", "))
            };
            // ⛔ NO RESULT LIST WHERE THE LOOP CARRIES NOTHING. The generic printer writes the bound
            // values before the mnemonic, and a loop with no `initArgs` binds none — so
            // `sentient.for %arg1 = %2 {..}{` is the whole line, with no leading ` = `.
            let produced = if results.is_empty() {
                String::new()
            } else {
                format!("{} = ", print::vals(&results))
            };
            let _ = writeln!(
                out,
                "{produced}sentient.for {} = {bound_val}{inits}{iter_types} {} {{",
                print::val(*iv),
                dict(&attrs)
            );
            for inner in body {
                crate::islands::sentient::print::emit(out, inner, depth + 1);
            }
            indent(out, depth);
            let _ = writeln!(out, "}}");
        }
        // `p << " " << predicate << ", " << lhs << ", " << rhs << " : " << lhsType`, the result types,
        // the dict with `predicate` ELIDED, then the regions (`SentientOps.cpp:1288-1305`).
        Op::If {
            predicate,
            lhs,
            rhs,
            yielded,
            dbg_name,
            then_body,
            else_body,
        } => {
            let produced = if yielded.is_empty() {
                String::new()
            } else {
                let tys: Vec<&str> = yielded.iter().map(|_| "index").collect();
                format!(" -> ({})", tys.join(", "))
            };
            let mut attrs = vec![
                attr("regLocales", &locale_array(yielded.iter().map(|y| y.reg))),
                attr("regIndices", &index_array(yielded.iter().map(|y| y.reg))),
            ];
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            // ⛔ `predicate` IS ELIDED FROM THE DICTIONARY because it is already printed as syntax
            // (`p.printOptionalAttrDict(op->getAttrs(), {"predicate"})`, `SentientOps.cpp:1300`).
            // Emitting it twice is a parse error, not a cosmetic difference.
            let _ = writeln!(
                out,
                "sentient.if {}, {}, {} : index{produced} {} {{",
                predicate.spelling(),
                print::val(*lhs),
                print::val(*rhs),
                dict(&attrs)
            );
            for inner in then_body {
                crate::islands::sentient::print::emit(out, inner, depth + 1);
            }
            if else_body.is_empty() {
                indent(out, depth);
                let _ = writeln!(out, "}}");
            } else {
                indent(out, depth);
                let _ = writeln!(out, "}} else {{");
                for inner in else_body {
                    crate::islands::sentient::print::emit(out, inner, depth + 1);
                }
                indent(out, depth);
                let _ = writeln!(out, "}}");
            }
        }

        // ───────────────────────── compute ─────────────────────────
        Op::VectorMac {
            mask,
            xrf_write_ptr,
            xrf_read_ptr,
            results,
            op_a,
            op_b,
            op_c,
            result,
            mode,
            compute_precision,
            fold_mode,
            unroll_factor,
            xrf_read_incr,
            xrf_write_incr,
            data_transfer_only,
            dbg_name,
        } => {
            let mut specific = vec![attr("mode", &quoted(mode.spelling()))];
            if *data_transfer_only {
                specific.push(attr("DataTransferOnly", "true"));
            }
            if *xrf_read_incr != 0 {
                specific.push(attr("xrfReadIncr", &format!("{xrf_read_incr} : i32")));
            }
            if *xrf_write_incr != 0 {
                specific.push(attr("xrfWriteIncr", &format!("{xrf_write_incr} : i32")));
            }
            // ⛔ WRITE POINTER FIRST, THEN READ — the `.td` states the order in a comment because the
            // operand list cannot (`SentientOps.td:234`). Swapping them reads the block being written.
            let mut pointers: Vec<Val> = Vec::new();
            if let Some(write) = xrf_write_ptr {
                pointers.push(*write);
            }
            if let Some(read) = xrf_read_ptr {
                pointers.push(*read);
            }
            let attrs = compute_attrs(
                &[('A', op_a), ('B', op_b), ('C', op_c)],
                specific,
                ComputeShared {
                    result,
                    compute_precision: *compute_precision,
                    fold_mode: *fold_mode,
                    unroll_factor: *unroll_factor,
                    dbg_name,
                },
            );
            let bound = if results.is_empty() {
                String::new()
            } else {
                format!("{} = ", print::vals(results))
            };
            let ptrs = if pointers.is_empty() {
                String::new()
            } else {
                format!(" pointers({})", print::vals(&pointers))
            };
            let _ = writeln!(
                out,
                "{bound}sentient.vector_mac {}{ptrs} {} : index",
                masked(*mask),
                dict(&attrs)
            );
        }
        Op::VectorBinary {
            mask,
            op_a,
            op_b,
            binary_op,
            result,
            compute_precision,
            fold_mode,
            unroll_factor,
            dbg_name,
        } => {
            let mut specific = vec![attr("binaryOp", &quoted(&binary_op.op().spelling()))];
            // ⭐ NO CHECK NEEDED: only the forwarding arm carries a port, and only the seven legal
            // operators can be named in it.
            if let Binary::Forwarding {
                to, unroll_incr, ..
            } = binary_op
            {
                specific.push(attr("LogicalResultForwarding", &quoted(&to.spelling())));
                if *unroll_incr {
                    specific.push(attr("unrollIncrLogicalResult", "true"));
                }
            }
            let attrs = compute_attrs(
                &[('A', op_a), ('B', op_b)],
                specific,
                ComputeShared {
                    result,
                    compute_precision: *compute_precision,
                    fold_mode: *fold_mode,
                    unroll_factor: *unroll_factor,
                    dbg_name,
                },
            );
            let _ = writeln!(
                out,
                "sentient.vector_binary mask({}) {} : index",
                print::val(*mask),
                dict(&attrs)
            );
        }
        Op::VectorUnary {
            mask,
            op_a,
            unary_op,
            result,
            compute_precision,
            fold_mode,
            unroll_factor,
            dbg_name,
        } => {
            let attrs = compute_attrs(
                &[('A', op_a)],
                vec![attr("unary_op", &quoted(&unary_op.spelling()))],
                ComputeShared {
                    result,
                    compute_precision: *compute_precision,
                    fold_mode: *fold_mode,
                    unroll_factor: *unroll_factor,
                    dbg_name,
                },
            );
            let _ = writeln!(
                out,
                "sentient.vector_unary mask({}) {} : index",
                print::val(*mask),
                dict(&attrs)
            );
        }
        Op::VectorTernary {
            mask,
            op_a,
            op_b,
            op_c,
            ternary_op,
            result,
            compute_precision,
            fold_mode,
            unroll_factor,
            unroll_incr_logical_result,
            dbg_name,
        } => {
            let mut specific = vec![attr("ternaryOp", &quoted(ternary_op.spelling()))];
            if *unroll_incr_logical_result {
                specific.push(attr("unrollIncrLogicalResult", "true"));
            }
            let attrs = compute_attrs(
                &[('A', op_a), ('B', op_b), ('C', op_c)],
                specific,
                ComputeShared {
                    result,
                    compute_precision: *compute_precision,
                    fold_mode: *fold_mode,
                    unroll_factor: *unroll_factor,
                    dbg_name,
                },
            );
            let _ = writeln!(
                out,
                "sentient.vector_ternary {} {} : index",
                masked(*mask),
                dict(&attrs)
            );
        }

        // ───────────────────────── transfers ─────────────────────────
        Op::Load {
            src,
            extent,
            shuffle_mode,
        } => {
            let mut attrs = extent_attrs(extent);
            attrs.push(attr("shuffle_mode", &quoted(shuffle_mode.spelling())));
            attrs.sort();
            let _ = writeln!(out, "sentient.load {} {}", print::val(*src), dict(&attrs));
        }
        // `SentientOps.cpp:313-327`.
        Op::LoadAndSend {
            mutable_addr,
            immutable_addr,
            increment,
            consumer,
            result,
            extent,
            interleaved_group,
            rotate_val,
            dir,
            shuffle_mode,
            reg,
            dbg_name,
        } => {
            let mut attrs = extent_attrs(extent);
            attrs.push(attr("shuffle_mode", &quoted(shuffle_mode.spelling())));
            attrs.push(attr("reg_locale", &quoted(reg.locale.spelling())));
            if interleaved_group.0 != 0 {
                attrs.push(attr(
                    "interleaved_group",
                    &format!("{} : i32", interleaved_group.0),
                ));
            }
            if let Some(rotate) = rotate_val {
                attrs.push(attr("rotate_val", &format!("{rotate} : i32")));
            }
            if let Some(direction) = dir {
                attrs.push(attr("dir", &quoted(direction.spelling())));
            }
            if let Some(index) = reg.index {
                attrs.push(attr("reg_index", &format!("{} : i32", index.get())));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "{} = sentient.load_and_send mutable_addr({}), immutable_addr({}), increment({}), consumer({}) {} : index, index, index, index : index",
                print::val(*result),
                print::val(*mutable_addr),
                print::val(*immutable_addr),
                print::val(*increment),
                print::val(consumer.val()),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:470-492`. ⛔ THE OPTIONAL OPERANDS ARE PRINTED CONDITIONALLY AND THEIR TYPES
        // TOO, so the type list's length tracks which of them are present.
        Op::ReceiveAndStore {
            mutable_addr,
            immutable_addr,
            increment,
            producer,
            result,
            dst,
            drop_first,
            multicast_info,
            extent,
            interleaved_group,
            coalesce,
            subword_length,
            stride,
            permute,
            shuffle_mode,
            reg,
            dbg_name,
        } => {
            let mut attrs = extent_attrs(extent);
            attrs.push(attr("reg_locale", &quoted(reg.locale.spelling())));
            if let Some(mode) = shuffle_mode {
                attrs.push(attr("shuffle_mode", &quoted(mode.spelling())));
            }
            if interleaved_group.0 != 0 {
                attrs.push(attr(
                    "interleaved_group",
                    &format!("{} : i32", interleaved_group.0),
                ));
            }
            if *coalesce {
                attrs.push(attr("coalesce", "true"));
            }
            if *permute {
                attrs.push(attr("permute", "true"));
            }
            if *subword_length != 1 {
                attrs.push(attr("subword_length", &format!("{subword_length} : i32")));
            }
            if *stride != 1 {
                attrs.push(attr("stride", &format!("{stride} : i32")));
            }
            if let Some(index) = reg.index {
                attrs.push(attr("reg_index", &format!("{} : i32", index.get())));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let mut operands = format!(
                " mutable_addr({}), immutable_addr({}), increment({}), producer({})",
                print::val(*mutable_addr),
                print::val(*immutable_addr),
                print::val(*increment),
                print::val(producer.val())
            );
            let mut types = String::from("index, index, index, index");
            if let Some(value) = dst {
                operands.push_str(&format!(", dst({})", print::val(*value)));
                types.push_str(", index");
            }
            if let Some(value) = drop_first {
                operands.push_str(&format!(", drop_first({})", print::val(*value)));
                types.push_str(", index");
            }
            if let Some(value) = multicast_info {
                operands.push_str(&format!(", multicast_info({})", print::val(*value)));
                types.push_str(", index");
            }
            let _ = writeln!(
                out,
                "{} = sentient.receive_and_store{operands} {} : {types} : index",
                print::val(*result),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:775-800`.
        Op::LoadAndStore {
            src,
            dst,
            src_mutable_addr,
            src_immutable_addr,
            src_inc,
            dst_mutable_addr,
            dst_immutable_addr,
            dst_inc,
            multicast_info,
            results,
            extent,
            stride,
            rotate_val,
            shuffle_mode,
            src_reg,
            dst_reg,
            dir,
            is_ibr_write,
            dbg_name,
        } => {
            let mut attrs = extent_attrs(extent);
            attrs.push(attr("shuffle_mode", &quoted(shuffle_mode.spelling())));
            // ⭐ THE ARRAY IS BUILT FROM THE TWO NAMED ENDS, SOURCE FIRST, so its length is two by
            // construction and the order is stated rather than remembered.
            let ends = [*src_reg, *dst_reg];
            attrs.push(attr("regLocales", &locale_array(ends.into_iter())));
            attrs.push(attr("regIndices", &index_array(ends.into_iter())));
            if *stride != 1 {
                attrs.push(attr("stride", &format!("{stride} : i32")));
            }
            if let Some(rotate) = rotate_val {
                attrs.push(attr("rotate_val", &format!("{rotate} : i32")));
            }
            if let Some(direction) = dir {
                attrs.push(attr("dir", &quoted(direction.spelling())));
            }
            if *is_ibr_write {
                attrs.push(attr("\"is-ibr-write\"", "1 : i8"));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            // ⛔ MLIR SORTS BY THE ATTRIBUTE'S NAME, NOT BY ITS PRINTED FORM. `"is-ibr-write"` is
            // quoted because it is hyphenated, and a plain sort would put the quote before every
            // letter — the vendor prints it between `element_size` and `regIndices`
            // (`l3-gather.mlir:37`).
            attrs.sort_by(|a, b| a.trim_start_matches('"').cmp(b.trim_start_matches('"')));
            let mut operands = format!(
                " src({}), dst({}), src_mutable_addr({}), src_immutable_addr({}), src_inc({}), \
                 dst_mutable_addr({}), dst_immutable_addr({}), dst_inc({})",
                print::val(*src),
                print::val(*dst),
                print::val(*src_mutable_addr),
                print::val(*src_immutable_addr),
                print::val(*src_inc),
                print::val(*dst_mutable_addr),
                print::val(*dst_immutable_addr),
                print::val(*dst_inc)
            );
            let mut types = String::from("index, index, index, index, index, index, index, index");
            if let Some(value) = multicast_info {
                operands.push_str(&format!(", multicast_info({})", print::val(*value)));
                types.push_str(", index");
            }
            let (src_res, dst_res) = results;
            let _ = writeln!(
                out,
                "{}, {} = sentient.load_and_store{operands} {} : {types} : index, index",
                print::val(*src_res),
                print::val(*dst_res),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:206-222`.
        Op::LoadComputeAndSend {
            mutable_addr,
            immutable_addr,
            increment,
            element_index,
            scale_index,
            consumer,
            result,
            src_total_elements,
            dst_total_elements,
            src_element_size,
            dst_element_size,
            dir,
            shuffle_mode,
            reg,
            dbg_name,
        } => {
            // ⛔ SOURCE AND DESTINATION EXTENTS ARE SEPARATE, because the compute in the middle may
            // change the width.
            let mut attrs = vec![
                attr(
                    "src_total_elements",
                    &format!("{} : i32", src_total_elements.0),
                ),
                attr(
                    "dst_total_elements",
                    &format!("{} : i32", dst_total_elements.0),
                ),
                attr("src_element_size", &format!("{} : i32", src_element_size.0)),
                attr("dst_element_size", &format!("{} : i32", dst_element_size.0)),
                attr("shuffle_mode", &quoted(shuffle_mode.spelling())),
                attr("reg_locale", &quoted(reg.locale.spelling())),
            ];
            if let Some(direction) = dir {
                attrs.push(attr("dir", &quoted(direction.spelling())));
            }
            if let Some(index) = reg.index {
                attrs.push(attr("reg_index", &format!("{} : i32", index.get())));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "{} = sentient.load_compute_and_send mutable_addr({}), immutable_addr({}), increment({}), element_index({}), scale_index({}), consumer({}) {} : index, index, index, index, index, index : index",
                print::val(*result),
                print::val(*mutable_addr),
                print::val(*immutable_addr),
                print::val(*increment),
                print::val(*element_index),
                print::val(*scale_index),
                print::val(consumer.val()),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:580-592`. ⭐ BINDS TWO VALUES: the address and the datum.
        Op::LoadAndExtractScalar {
            mutable_addr,
            immutable_addr,
            increment,
            consumer,
            addr_result,
            data_result,
            total_elements,
            element_size,
            addr_reg,
            data_reg,
            dbg_name,
        } => {
            let mut attrs = vec![
                attr("total_elements", &format!("{} : i32", total_elements.0)),
                attr("element_size", &format!("{} : i32", element_size.0)),
                attr(
                    "regLocales",
                    &locale_array([*addr_reg, *data_reg].into_iter()),
                ),
                attr(
                    "regIndices",
                    &index_array([*addr_reg, *data_reg].into_iter()),
                ),
            ];
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "{}, {} = sentient.load_and_extract_scalar mutable_addr({}), immutable_addr({}), increment({}), consumer({}) {} : index, index, index, index : index, index",
                print::val(*addr_result),
                print::val(*data_result),
                print::val(*mutable_addr),
                print::val(*immutable_addr),
                print::val(*increment),
                print::val(consumer.val()),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:665-672`.
        Op::ReceiveAndExtractScalar {
            unit,
            position,
            result,
            reg,
            dbg_name,
        } => {
            let mut attrs = vec![attr("reg_locale", &quoted(reg.locale.spelling()))];
            if let Some(index) = reg.index {
                attrs.push(attr("reg_index", &format!("{} : i32", index.get())));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "{} = sentient.receive_and_extract_scalar unit({}), position({}) {} : index, index : index",
                print::val(*result),
                print::val(unit.val()),
                print::val(*position),
                dict(&attrs)
            );
        }

        // ───────────────────────── scalars ─────────────────────────
        Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg,
            ty,
        } => scalar_binary(out, "scalar_add", *result, *lhs, *rhs, *reg, *ty),
        Op::ScalarSub {
            lhs,
            rhs,
            result,
            reg,
            ty,
        } => scalar_binary(out, "scalar_sub", *result, *lhs, *rhs, *reg, *ty),
        Op::ScalarMul {
            lhs,
            rhs,
            result,
            reg_locale,
            ty,
        } => {
            // ⛔ NO `regIndex` ON THIS ONE — the asymmetry is the reference's
            // (`SentientOps.td:816-829`).
            let attrs: Vec<String> = reg_locale
                .map(|locale| vec![attr("regLocale", &locale_attr(locale))])
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "{} = sentient.scalar_mul {}, {}{} : {}, {}",
                print::val(*result),
                print::val(*lhs),
                print::val(*rhs),
                dict(&attrs),
                ty.spelling(),
                ty.spelling()
            );
        }
        Op::ScalarCopy {
            input,
            result,
            reg,
            program_header,
        } => {
            let mut attrs = vec![attr("reg_locale", &quoted(reg.locale.spelling()))];
            if let Some(index) = reg.index {
                attrs.push(attr("reg_index", &format!("{} : i32", index.get())));
            }
            if *program_header {
                attrs.push(attr("programHeader", "true"));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "{} = sentient.scalar_copy {} {} : index",
                print::val(*result),
                print::val(*input),
                dict(&attrs)
            );
        }
        Op::ScalarConstant {
            value,
            result,
            reg_locale,
            ty,
            is_symbol,
        } => {
            // ⛔ TWO FORMS, AND `is_symbol` CHOOSES (`SentientOps.cpp:1698-1715`). Without it: the
            // value and the result type only, so the locale does not appear even when set — and
            // ⛔ HEX FOR A FLOAT RESULT, `0x` + `raw_ostream::write_hex`'s LOWERCASE digits, which
            // is how `vector<1xf16>`'s bit pattern survives an `si64` attribute
            // (`dcc/test/Conversion/VectorChainToSentientPESFP/splat_const_bit.mlir:20`). With it:
            // the whole attribute dictionary, printed decimal because that is what an
            // `IntegerAttr` prints as.
            let attrs = if *is_symbol {
                dict(&[
                    attr("is_symbol", "true"),
                    attr(
                        "regLocale",
                        &format!("#sentient<reg_type {}>", reg_locale.spelling()),
                    ),
                    attr("value", &format!("{value} : si64")),
                ])
            } else if ty.is_float() {
                dict(&[attr("value", &format!("0x{:x} : si64", *value as u64))])
            } else {
                dict(&[attr("value", &format!("{value} : si64"))])
            };
            let _ = writeln!(
                out,
                "{} = sentient.scalar_constant{attrs} : {}",
                print::val(*result),
                ty.spelling()
            );
        }
        Op::VectorConstant { value, result, ty } => {
            // ⛔ HEX, AND NO PER-ELEMENT TYPE. `VectorConstantOp::print` writes `0x` +
            // `Twine::utohexstr` per element and the vector type once, at the end
            // (`SentientOps.cpp:1735-1748`).
            let elems: Vec<String> = value.iter().map(|v| format!("0x{:x}", *v as u64)).collect();
            let _ = writeln!(
                out,
                "{} = sentient.vector_constant{} : {}",
                print::val(*result),
                dict(&[attr("value", &format!("[{}]", elems.join(", ")))]),
                crate::islands::dataflow_ir::print::vector(*ty)
            );
        }

        // ───────────────────────── masks, sync, ports ─────────────────────────
        Op::Sync {
            mode,
            peers,
            soft,
            implicit_sync_memory_boundary,
            dbg_name,
        } => {
            let spelled: Vec<String> = peers.iter().map(|p| quoted(p.peer().spelling())).collect();
            let mut attrs = vec![
                attr("mode", &quoted(mode.spelling())),
                attr("units", &format!("[{}]", spelled.join(", "))),
            ];
            if *soft {
                attrs.push(attr("soft", "true"));
            }
            if let Some(boundary) = implicit_sync_memory_boundary {
                attrs.push(attr(
                    "implicit_sync_memory_boundary",
                    &format!("{boundary} : si32"),
                ));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(out, "sentient.sync{}", dict(&attrs));
        }
        Op::Nop { dbg_name } => {
            let _ = writeln!(out, "sentient.nop{}", dict(&[dbg(dbg_name)]));
        }
        Op::IncrMask { dbg_name } => {
            // `SentientOps.cpp:2455` — the dictionary and nothing else.
            let _ = writeln!(out, "sentient.incrmask{}", dict(&[dbg(dbg_name)]));
        }
        Op::SetSendDst { units } => {
            let _ = writeln!(out, "sentient.set_send_dst({})", print::val(units.val()));
        }
        Op::LogicalPort { port_name, result } => {
            let _ = writeln!(
                out,
                "{} = sentient.logical_port {} : index",
                print::val(*result),
                dict(&[attr("portName", &quoted(&port_name.spelling()))])
            );
        }
        // `SentientOps.cpp:2433-2440`.
        Op::SetMask {
            mask_value,
            dbg_name,
        } => {
            let _ = writeln!(
                out,
                "sentient.set_mask mask_value({}){} : index",
                print::val(*mask_value),
                dict(&[dbg(dbg_name)])
            );
        }
        // `SentientOps.cpp:2368-2375`.
        Op::Samv {
            mask_value,
            mask_all,
            num_valid_entry,
            slice_id_xsl,
            xsl_inner,
            wsl_len,
            precision,
            dbg_name,
        } => {
            let mut attrs = vec![
                attr("maskall", if *mask_all { "true" } else { "false" }),
                attr("numvalidentry", &format!("{} : i32", num_valid_entry.0)),
                attr("sliceid_xsl", &format!("{} : i32", slice_id_xsl.0)),
                attr("xslinner", if *xsl_inner { "true" } else { "false" }),
                attr("wsllen", &format!("{} : i32", wsl_len.0)),
                // ⛔ A RAW ISA FIELD, never the Precision enum's spelling.
                attr("precision", &format!("{} : i32", precision.0)),
            ];
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "sentient.samv mask_value({}) {} : index",
                print::val(*mask_value),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:2182-2196` — the input carries its own type inside the parentheses.
        Op::Splat {
            input,
            output,
            mask,
            pad,
            precision,
            program_header,
            unroll_factor,
            unroll_incr_result,
            dbg_name,
        } => {
            let mut attrs = vec![
                attr("pad", &quoted(pad.spelling())),
                attr("precision", &quoted(precision.spelling())),
                attr("unrollFactor", &quoted(unroll_factor.spelling())),
            ];
            if *program_header {
                attrs.push(attr("programHeader", "true"));
            }
            if *unroll_incr_result {
                attrs.push(attr("unrollIncrResult", "true"));
            }
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(
                out,
                "sentient.splat input({} : index) output({}) mask({}) {}",
                print::val(*input),
                print::val(*output),
                print::val(*mask),
                dict(&attrs)
            );
        }
        // `SentientOps.cpp:2205-2210` — the dictionary is the whole syntax.
        Op::Opaque {
            func,
            read_write,
            read_only,
            params,
            dbg_name,
        } => {
            let mut attrs = vec![
                // ⛔ LOWER-CASED, WHICH IS WHAT THE RUNG BELOW WROTE. The generated enum spells the
                // template's `RECIPROCAL`; `dataflow.opaque` prints `func_name = "reciprocal"` and
                // `lowerOpaqueOperation` forwards that `StringAttr` untouched, so IBM's lowered
                // output is `func_name = "reciprocal"` too
                // (`dcc/test/Conversion/DataflowToSentient/opaque.mlir:18`).
                attr("func_name", &quoted(&func.spelling().to_lowercase())),
                attr("read_write_register_dictionary", &reg_dict(read_write)),
                attr("read_only_register_dictionary", &reg_dict(read_only)),
                attr("parameter_dictionary", &param_dict(params)),
            ];
            if let Some(name) = dbg_name {
                attrs.push(attr("dbgName", &quoted(name)));
            }
            attrs.sort();
            let _ = writeln!(out, "sentient.opaque{}", dict(&attrs));
        }
    }
}

/// Two spaces per level — a region's closing brace lines up with the op that opened it.
fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

/// `mask(%v)`, or `mask()` where the op declares it optional and it is absent.
fn masked(mask: Option<Val>) -> String {
    mask.map_or_else(
        || "mask()".to_owned(),
        |m| format!("mask({})", print::val(m)),
    )
}

/// `scalar_add` and `scalar_sub`, whose printed shape the `.td` declares identically.
fn scalar_binary(
    out: &mut String,
    mnemonic: &str,
    result: Val,
    lhs: Val,
    rhs: Val,
    reg: Option<Reg>,
    ty: ScalarTy,
) {
    // ⛔ NO DICTIONARY AT ALL WHEN NO ALLOCATOR HAS SPOKEN — see [`Op::ScalarAdd`].
    let mut attrs: Vec<String> = Vec::new();
    if let Some(reg) = reg {
        attrs.push(attr("regLocale", &locale_attr(reg.locale)));
        if let Some(index) = reg.index {
            attrs.push(attr("regIndex", &format!("{} : i32", index.get())));
        }
        attrs.sort();
    }
    let _ = writeln!(
        out,
        "{} = sentient.{mnemonic} {}, {}{} : {}, {}",
        print::val(result),
        print::val(lhs),
        print::val(rhs),
        dict(&attrs),
        ty.spelling(),
        ty.spelling()
    );
}

/// ONE `SentientRegTypeAttr` AS THE REFERENCE WRITES IT — `#sentient<reg_type lrf>`.
///
/// ⚠️ THE ARRAY FORMS AND THE OTHER SINGLE-REGISTER OPS STILL SPELL IT `"lrf"`, a quoted string
/// ([`locale_array`]), which no reference output shows. Correcting them is a change to the printed
/// form of ops other units own, so it is left to those units; this is the spelling
/// `dcc/test/LXLU/rotate-composite.mlir:24` and `dcc/test/L3SU/dyn_node_e2e.mlir:75` show.
fn locale_attr(locale: RegType) -> String {
    format!("#sentient<reg_type {}>", locale.spelling())
}

/// THE ATTRIBUTE DICTIONARY OF ONE COMPUTE OP, sorted.
///
/// ⛔⛔ IT TAKES THE COMPUTE'S PIECES, NOT AN `Op`. The first version took `&Op` and ended in
/// `_ => unreachable!("compute_attrs called on an op that is not one of the four computes")` — a
/// runtime assertion that a caller passed the right variant, which is precisely the class of guard
/// this island is not allowed to use. A function that may only be called with a compute takes a
/// compute's parts, and the wrong call is then a type error.
///
/// ⛔ SORTED, NOT DECLARATION-ORDERED — see [`emit`]'s note.
fn compute_attrs(
    operands: &[(char, &Operand)],
    specific: Vec<String>,
    shared: ComputeShared<'_>,
) -> Vec<String> {
    let mut attrs = specific;
    for (tag, operand) in operands {
        attrs.push(attr(&format!("op{tag}"), &quoted(&operand.port.spelling())));
        attrs.push(attr(
            &format!("op{tag}Forwarding"),
            &port_array(&operand.forwarding),
        ));
        attrs.push(attr(
            &format!("op{tag}Precision"),
            &quoted(operand.precision.spelling()),
        ));
        if let Some(id) = operand.data_id {
            attrs.push(attr(&format!("op{tag}DataID"), &format!("{id} : si32")));
        }
        if let Some(id) = operand.port_id {
            attrs.push(attr(&format!("op{tag}PortID"), &format!("{id} : si32")));
        }
        if operand.unroll_incr {
            attrs.push(attr(&format!("unrollIncrOp{tag}"), "true"));
        }
    }
    attrs.push(attr(
        "ResultForwarding",
        &port_array(&shared.result.forwarding),
    ));
    attrs.push(attr(
        "ResultPrecision",
        &quoted(shared.result.precision.spelling()),
    ));
    if shared.result.unroll_incr {
        attrs.push(attr("unrollIncrResult", "true"));
    }
    attrs.push(attr(
        "ComputePrecision",
        &quoted(shared.compute_precision.spelling()),
    ));
    if let Some(mode) = shared.fold_mode {
        attrs.push(attr("fold_mode", &quoted(mode.spelling())));
    }
    attrs.push(attr(
        "unrollFactor",
        &quoted(shared.unroll_factor.spelling()),
    ));
    if let Some(name) = shared.dbg_name {
        attrs.push(attr("dbgName", &quoted(name)));
    }
    attrs.sort();
    attrs
}

/// The extent attributes every transfer shares.
fn extent_attrs(extent: &Extent) -> Vec<String> {
    let mut attrs = vec![
        attr(
            "total_elements",
            &format!("{} : i32", extent.total_elements.0),
        ),
        attr("element_size", &format!("{} : i32", extent.element_size.0)),
    ];
    // ⭐ ONLY WHERE IT DIFFERS FROM THE `.td`'S DEFAULT, so a printed attribute always means a
    // decision was made. The defaults are one element of chunk and stride, and no burst.
    if extent.chunk_size != Elements(1) {
        attrs.push(attr(
            "chunk_size",
            &format!("{} : i32", extent.chunk_size.0),
        ));
    }
    if extent.chunk_stride != Elements(1) {
        attrs.push(attr(
            "chunk_stride",
            &format!("{} : i32", extent.chunk_stride.0),
        ));
    }
    if extent.burst_size != Elements(0) {
        attrs.push(attr(
            "burst_size",
            &format!("{} : i32", extent.burst_size.0),
        ));
    }
    attrs
}

/// A `SentientRegTypeArrayAttr` — one locale per carried value.
fn locale_array(regs: impl Iterator<Item = Reg>) -> String {
    let spelled: Vec<String> = regs.map(|r| quoted(r.locale.spelling())).collect();
    format!("[{}]", spelled.join(", "))
}

/// An `I32ArrayAttr` of register indices — ⛔ AN ABSENT ONE PRINTS AS THE `.td`'S `-1`, which is how
/// the reference spells unassigned. That is the one place the sentinel is written, and it is written
/// from an `Option` rather than stored as an integer.
fn index_array(regs: impl Iterator<Item = Reg>) -> String {
    let rendered: Vec<String> = regs
        .map(|r| {
            r.index
                .map_or_else(|| "-1".to_owned(), |i| i.get().to_string())
        })
        .collect();
    format!("[{}]", rendered.join(", "))
}

/// A `BoolArrayAttr`.
fn bool_array(flags: impl Iterator<Item = bool>) -> String {
    let rendered: Vec<&str> = flags
        .map(|flag| if flag { "true" } else { "false" })
        .collect();
    format!("[{}]", rendered.join(", "))
}

/// An opaque's register dictionary — symbol to the address its allocation starts at.
fn reg_dict(entries: &[(RegName, RegAddr)]) -> String {
    // ⛔⛔ THE `R` IS LOAD-BEARING AND WAS MISSING. `insertReg` writes
    // `"R" + std::to_string(startAddress)` (`ddc/ddcv1.cpp:3350`) and the consumer takes it back off
    // BY POSITION: `port_str = "lrf" + port_str.erase(0, 1)`
    // (`dcc/src/Dialect/Sentient/Utils.cpp:157`). IBM's own lowered output is
    // `read_write_register_dictionary = {P0 = "R0", P1 = "R1"}`
    // (`dcc/test/Conversion/DataflowToSentient/opaque.mlir:18`); a bare `"0"` becomes `lrf`, which is
    // no port at all. `dataflow.opaque` prints it — `lowerOpaqueOperation` forwards the dictionary
    // unchanged, so `sentient.opaque` must print the same string.
    sorted_dict(
        entries,
        |name: RegName| name.spelling(),
        |addr: RegAddr| format!("R{}", addr.0),
    )
}

/// An opaque's parameter dictionary.
fn param_dict(entries: &[(ParamKey, ParamValue)]) -> String {
    sorted_dict(
        entries,
        |key: ParamKey| key.spelling(),
        |value: ParamValue| value.spelling().to_owned(),
    )
}

/// A `{key = "value", ..}` attribute dictionary, KEY-SORTED.
///
/// ⛔ SORTED BECAUSE MLIR SORTS. A `DictionaryAttr` is stored key-ordered, so a round trip through
/// the parser reorders anything else — and a printer whose output does not survive a round trip
/// cannot be checked against the vendored files. The rung below prints its dictionaries the same way
/// (`crate::islands::dataflow_ir::dialects::dataflow`).
fn sorted_dict<K: Copy, V: Copy>(
    entries: &[(K, V)],
    key: impl Fn(K) -> &'static str,
    value: impl Fn(V) -> String,
) -> String {
    let mut sorted: Vec<(&'static str, String)> =
        entries.iter().map(|(k, v)| (key(*k), value(*v))).collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{{{}}}",
        sorted
            .iter()
            .map(|(key, value)| format!("{key} = \"{value}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `name = value`.
fn attr(name: &str, value: &str) -> String {
    format!("{name} = {value}")
}

/// A string attribute's quoted form.
fn quoted(text: &str) -> String {
    format!("\"{text}\"")
}

/// `$dbgName`, which every op declares and most leave unset.
fn dbg(name: &Option<String>) -> String {
    name.as_ref()
        .map_or_else(String::new, |n| attr("dbgName", &quoted(n)))
}

/// A `SentientComputePortArrayAttr` — ⭐ EMPTY IS `[]`, not an absent attribute.
fn port_array(ports: &[Port]) -> String {
    let spelled: Vec<String> = ports.iter().map(|p| quoted(&p.spelling())).collect();
    format!("[{}]", spelled.join(", "))
}

/// ` {a = 1, b = 2}` — or nothing at all where every attribute was absent.
fn dict(attrs: &[String]) -> String {
    let live: Vec<&str> = attrs
        .iter()
        .map(String::as_str)
        .filter(|a| !a.is_empty())
        .collect();
    if live.is_empty() {
        return String::new();
    }
    format!(" {{{}}}", live.join(", "))
}
