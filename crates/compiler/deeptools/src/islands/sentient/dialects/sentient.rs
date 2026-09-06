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
//! ⛔ WHAT IS STILL NOT HERE: a register INDEX inside a file, an instruction encoding, a program
//! counter. `regIndex` appears on these ops as a *hint* an allocator fills in
//! (`DefaultValuedAttr<I32Attr, "-1">` — minus one means unassigned), and the passes that decide it
//! are D64-D75, `registerManagementPasses`. A real index and a real opcode are ProgIR
//! ([`crate::islands::progir`]).

use std::fmt::Write as _;

use crate::arch::{Bounded, Bytes, Elements};
use crate::generated::{OpaqueFunc, ParamKey, ParamValue, RegName};
use crate::islands::dataflow_ir::dialects::dataflow::RegAddr;
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

/// WHICH `lrf<n>` — ⛔ AN INDEX THE INSTRUCTION FIELD CAN ACTUALLY NAME.
///
/// ⛔⛔ THIS WAS A BARE `u8` GUARDED BY A RUNTIME `assert!` INSIDE `Port::encoding`, WHICH IS THE
/// FAILURE MODE THIS REPO REFUSES ON PRINCIPLE: `Port::Lrf(200)` was constructible, travelled through
/// every lowering, and aborted at print time — the furthest possible point from the mistake. A
/// checked constructor makes it unrepresentable instead, which is the same shape as
/// [`crate::units::CoreId::checked`].
///
/// ⛔ THIRTY-TWO IS WHAT THE ISA FIELD ENCODES, NOT A REGISTER FILE'S DEPTH. The SFP/PE LRF holds
/// sixteen on this target and the state file one, and those vary by arch where this number does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LrfIndex(Bounded<32>);

impl LrfIndex {
    /// An index, or `None` where the instruction field could not name it.
    #[must_use]
    pub const fn checked(index: u32) -> Option<LrfIndex> {
        match Bounded::checked(index) {
            Some(bounded) => Some(LrfIndex(bounded)),
            None => None,
        }
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// WHICH `istate<n>` — four exist (`SentientTypes.td:143-146`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IStateIndex(Bounded<4>);

impl IStateIndex {
    /// An index, or `None` where there is no such state register.
    #[must_use]
    pub const fn checked(index: u32) -> Option<IStateIndex> {
        match Bounded::checked(index) {
            Some(bounded) => Some(IStateIndex(bounded)),
            None => None,
        }
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
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
                if n.get() < 16 {
                    12 + n.get()
                } else {
                    48 + (n.get() - 16)
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
            Self::IState(n) => 43 + n.get(),
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

/// A `gcvt_imm<n>` A **BINARY** MAY TAKE — ⛔ 0, 4, 24, 28 AND NOTHING ELSE
/// (`SentientTypes.td:345-348`).
///
/// ⛔⛔ THIS WAS A BARE `u8` WITH THE LEGAL SET IN A `const` ARRAY BESIDE IT, which documents the
/// rule without enforcing it: `GcvtImm(3)` was constructible and printed `gcvt_imm3`, an attribute
/// the parser does not know. A closed vendor set gets a total constructor.
///
/// ⛔ AND IT IS **DISJOINT** FROM [`UnaryGcvt`]'s. Same spelling, no shared value — which is why they
/// are two types rather than one with a wider set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinaryGcvt(u8);

impl BinaryGcvt {
    /// The immediate, or `None` where the enum defines no such case.
    #[must_use]
    pub const fn checked(imm: u8) -> Option<BinaryGcvt> {
        match imm {
            0 | 4 | 24 | 28 => Some(BinaryGcvt(imm)),
            _ => None,
        }
    }

    /// The immediate.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// AN `fcvt_imm<n>` A **BINARY** MAY TAKE — ⛔ 2, 3, 4, 7 only (`SentientTypes.td:349-352`).
/// Disjoint from [`UnaryFcvt`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinaryFcvt(u8);

impl BinaryFcvt {
    /// The immediate, or `None` where the enum defines no such case.
    #[must_use]
    pub const fn checked(imm: u8) -> Option<BinaryFcvt> {
        match imm {
            2 | 3 | 4 | 7 => Some(BinaryFcvt(imm)),
            _ => None,
        }
    }

    /// The immediate.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// WHICH `pack<n>` — ⛔⛔ **TEN AND ELEVEN DO NOT EXIST**.
///
/// The enum runs 0..9 and then 12..27 (`SentientTypes.td:368-388`). A bare `u8` made `Pack(10)` and
/// `Pack(11)` constructible, and they print attributes the parser rejects — a gap in the middle of a
/// range being exactly the shape a reader completes by hand without noticing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackIndex(u8);

impl PackIndex {
    /// The index, or `None` for the two the enum skips and anything past its end.
    #[must_use]
    pub const fn checked(index: u8) -> Option<PackIndex> {
        match index {
            0..=9 | 12..=27 => Some(PackIndex(index)),
            _ => None,
        }
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
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

/// A `gcvt_imm<n>` A **UNARY** MAY TAKE — ⛔ 1, 2, 5, 6, 8, 16, 17 (`SentientTypes.td:630-636`).
/// Disjoint from [`BinaryGcvt`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnaryGcvt(u8);

impl UnaryGcvt {
    /// The immediate, or `None` where the enum defines no such case.
    #[must_use]
    pub const fn checked(imm: u8) -> Option<UnaryGcvt> {
        match imm {
            1 | 2 | 5 | 6 | 8 | 16 | 17 => Some(UnaryGcvt(imm)),
            _ => None,
        }
    }

    /// The immediate.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// AN `fcvt_imm<n>` A **UNARY** MAY TAKE — ⛔ 0, 1, 5, 6 (`SentientTypes.td:637-640`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnaryFcvt(u8);

impl UnaryFcvt {
    /// The immediate, or `None` where the enum defines no such case.
    #[must_use]
    pub const fn checked(imm: u8) -> Option<UnaryFcvt> {
        match imm {
            0 | 1 | 5 | 6 => Some(UnaryFcvt(imm)),
            _ => None,
        }
    }

    /// The immediate.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
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
pub struct RegIndex(Bounded<128>);

impl RegIndex {
    /// An index, or `None` where no register file is that deep.
    #[must_use]
    pub const fn checked(index: u32) -> Option<RegIndex> {
        match Bounded::checked(index) {
            Some(bounded) => Some(RegIndex(bounded)),
            None => None,
        }
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
    /// `element_size` — HOW WIDE ONE IS. ⛔ A width in bytes, not a count.
    pub element_size: Bytes,
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
    pub const fn of(total_elements: Elements, element_size: Bytes) -> Extent {
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
        /// `$bound` — the trip count.
        bound: Val,
        /// `$initArgs` — the values carried into the loop.
        init_args: Vec<Val>,
        /// The values the loop yields.
        results: Vec<Val>,
        /// `$regLocales` — one per carried value.
        reg_locales: Vec<RegType>,
        /// `$regIndices` — one per carried value; ⛔ `None` is the `.td`'s `-1`.
        reg_indices: Vec<Option<RegIndex>>,
        /// `$programHeader` — one flag per carried value.
        program_header: Vec<bool>,
        /// `$dbgName`.
        dbg_name: Option<String>,
        /// The body.
        body: Vec<Op>,
    },

    /// `sentient.if` — a two-operand comparison and its region (`SentientOps.td:159`).
    If {
        /// `$predicate`.
        predicate: CmpPredicate,
        /// `$lhs`.
        lhs: Val,
        /// `$rhs`.
        rhs: Val,
        /// The values the region yields.
        results: Vec<Val>,
        /// `$regLocales`.
        reg_locales: Vec<RegType>,
        /// `$regIndices`.
        reg_indices: Vec<Option<RegIndex>>,
        /// `$dbgName`.
        dbg_name: Option<String>,
        /// The `then` body.
        then_body: Vec<Op>,
        /// The `else` body, where there is one.
        else_body: Vec<Op>,
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
        /// `$consumer` — who receives it.
        consumer: Val,
        /// The value it binds.
        result: Val,
        /// The extents.
        extent: Extent,
        /// `$interleaved_group`.
        interleaved_group: u32,
        /// `$rotate_val`.
        rotate_val: Option<u32>,
        /// `$dir`.
        dir: Option<RoutingDirection>,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex` — ⛔ `None` is `-1`, unassigned.
        reg_index: Option<RegIndex>,
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
        /// `$producer` — who sent it.
        producer: Val,
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
        /// `$interleaved_group`.
        interleaved_group: u32,
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
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex`.
        reg_index: Option<RegIndex>,
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
        /// `$stride`.
        stride: u32,
        /// `$rotate_val`.
        rotate_val: Option<u32>,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
        /// `$regLocales`.
        reg_locales: Vec<RegType>,
        /// `$regIndices`.
        reg_indices: Vec<Option<RegIndex>>,
        /// `$dir`.
        dir: Option<RoutingDirection>,
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
        /// `$consumer`.
        consumer: Val,
        /// The value it binds.
        result: Val,
        /// `$src_total_elements`.
        src_total_elements: u32,
        /// `$dst_total_elements`.
        dst_total_elements: u32,
        /// `$src_element_size`.
        src_element_size: u32,
        /// `$dst_element_size`.
        dst_element_size: u32,
        /// `$dir`.
        dir: Option<RoutingDirection>,
        /// `$shuffle_mode`.
        shuffle_mode: ShuffleMode,
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex`.
        reg_index: Option<RegIndex>,
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
        /// `$consumer`.
        consumer: Val,
        /// `$addr` — the address it binds.
        addr_result: Val,
        /// `$data` — the datum it binds.
        data_result: Val,
        /// `$total_elements`.
        total_elements: u32,
        /// `$element_size`.
        element_size: u32,
        /// `$regLocales`.
        reg_locales: Vec<RegType>,
        /// `$regIndices`.
        reg_indices: Vec<Option<RegIndex>>,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    /// `sentient.receive_and_extract_scalar` — take one datum off the wire into a register
    /// (`SentientOps.td:675`).
    ReceiveAndExtractScalar {
        /// `$unit` — who to receive from.
        unit: Val,
        /// `$position` — which object to extract.
        position: Val,
        /// The value it binds.
        result: Val,
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex`.
        reg_index: Option<RegIndex>,
        /// `$dbgName`.
        dbg_name: Option<String>,
    },

    // ───────────────────────── scalars ─────────────────────────
    /// `sentient.scalar_add` (`SentientOps.td:700`). Syntax declared in the `.td`.
    ScalarAdd {
        /// `$inp1`.
        lhs: Val,
        /// `$inp2`.
        rhs: Val,
        /// `$out`.
        result: Val,
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex`.
        reg_index: Option<RegIndex>,
    },

    /// `sentient.scalar_sub` (`SentientOps.td:801`).
    ScalarSub {
        /// `$inp1`.
        lhs: Val,
        /// `$inp2`.
        rhs: Val,
        /// `$out`.
        result: Val,
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex`.
        reg_index: Option<RegIndex>,
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
        /// `$regLocale`.
        reg_locale: RegType,
    },

    /// `sentient.scalar_copy` (`SentientOps.td:830`).
    ScalarCopy {
        /// `$inp`.
        input: Val,
        /// `$out`.
        result: Val,
        /// `$regLocale`.
        reg_locale: RegType,
        /// `$regIndex`.
        reg_index: Option<RegIndex>,
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
        /// `$regLocale`.
        reg_locale: RegType,
    },

    /// `sentient.vector_constant` — a whole vector of immediates (`SentientOps.td:866`).
    VectorConstant {
        /// `$value` — the elements.
        value: Vec<i64>,
        /// `$out`.
        result: Val,
    },

    // ───────────────────────── masks, sync, ports ─────────────────────────
    /// `sentient.sync` — ⛔ NO OPERANDS AT ALL, only attributes (`SentientOps.td:876`). Which units
    /// it synchronises is the `units` attribute, not an operand list.
    Sync {
        /// `$mode`.
        mode: SyncMode,
        /// `$units`.
        units: Vec<Consumer>,
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
    SetSendDst {
        /// `$units`.
        units: Val,
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

/// ONE `sentient` OP AS TEXT. The caller has already indented.
///
/// ⛔⛔ THE ATTRIBUTE DICTIONARY IS **ALPHABETICAL**, because MLIR's `printOptionalAttrDict` sorts
/// it and every one of the nineteen custom printers delegates to that
/// (e.g. `BinaryOp::print`, `SentientOps.cpp:2219-2229`). Writing attributes in declaration order
/// produces text that parses but never matches a reference dump byte for byte, which is the only
/// oracle this rung has.
///
/// ⛔ AND THE FOUR COMPUTE OPS PRINT ONLY THEIR MASK. `sentient.vector_binary mask(%m) {…} : index`
/// — every port, precision, forwarding list and unroll flag is in the dictionary, not the operand
/// list (`SentientOps.cpp:2219-2229`, and the same three lines for ternary and unary).
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::Yield { results } => {
            if results.is_empty() {
                let _ = writeln!(out, "sentient.yield");
            } else {
                let _ = writeln!(out, "sentient.yield {}", print::vals(results));
            }
        }
        Op::Nop { dbg_name } => {
            let _ = writeln!(out, "sentient.nop{}", dict(&[dbg(dbg_name)]));
        }
        Op::IncrMask { dbg_name } => {
            let _ = writeln!(out, "sentient.incrmask{}", dict(&[dbg(dbg_name)]));
        }
        Op::SetSendDst { units } => {
            let _ = writeln!(
                out,
                "sentient.set_send_dst({})",
                print::val(*units)
            );
        }
        Op::LogicalPort { port_name, result } => {
            let _ = writeln!(
                out,
                "{} = sentient.logical_port {} : index",
                print::val(*result),
                dict(&[attr("portName", &quoted(&port_name.spelling()))])
            );
        }
        Op::ScalarConstant {
            value,
            result,
            reg_locale,
        } => {
            let _ = writeln!(
                out,
                "{} = sentient.scalar_constant {} : index",
                print::val(*result),
                dict(&[
                    attr("reg_locale", &quoted(reg_locale.spelling())),
                    attr("value", &format!("{value} : si64")),
                ])
            );
        }
        Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg_locale,
            reg_index,
        } => scalar_binary(
            out,
            "scalar_add",
            *result,
            *lhs,
            *rhs,
            ScalarReg {
                locale: *reg_locale,
                index: *reg_index,
            },
        ),
        Op::ScalarSub {
            lhs,
            rhs,
            result,
            reg_locale,
            reg_index,
        } => scalar_binary(
            out,
            "scalar_sub",
            *result,
            *lhs,
            *rhs,
            ScalarReg {
                locale: *reg_locale,
                index: *reg_index,
            },
        ),
        Op::ScalarMul {
            lhs,
            rhs,
            result,
            reg_locale,
        } => {
            // ⛔ NO `regIndex` — see [`Op::ScalarMul`].
            let _ = writeln!(
                out,
                "{} = sentient.scalar_mul {}, {} {} : index, index",
                print::val(*result),
                print::val(*lhs),
                print::val(*rhs),
                dict(&[attr("reg_locale", &quoted(reg_locale.spelling()))])
            );
        }
        Op::ScalarCopy {
            input,
            result,
            reg_locale,
            reg_index,
            program_header,
        } => {
            let mut attrs = vec![attr("reg_locale", &quoted(reg_locale.spelling()))];
            if let Some(index) = reg_index {
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
        Op::VectorConstant { value, result } => {
            let elems: Vec<String> = value.iter().map(|v| format!("{v} : si64")).collect();
            let _ = writeln!(
                out,
                "{} = sentient.vector_constant {}",
                print::val(*result),
                dict(&[attr("value", &format!("[{}]", elems.join(", ")))])
            );
        }
        Op::Sync {
            mode,
            units,
            soft,
            implicit_sync_memory_boundary,
            dbg_name,
        } => {
            let spelled: Vec<String> = units.iter().map(|u| quoted(u.spelling())).collect();
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
        Op::SetMask {
            mask_value,
            dbg_name,
        } => {
            let _ = writeln!(
                out,
                "sentient.set_mask {}{}",
                print::val(*mask_value),
                dict(&[dbg(dbg_name)])
            );
        }
        Op::Load {
            src,
            extent,
            shuffle_mode,
        } => {
            let mut attrs = extent_attrs(extent);
            attrs.push(attr("shuffle_mode", &quoted(shuffle_mode.spelling())));
            attrs.sort();
            let _ = writeln!(
                out,
                "sentient.load {} {}",
                print::val(*src),
                dict(&attrs)
            );
        }
        // ⭐ THE COMPUTE OPS ALL PRINT THE SAME SHAPE: the mask, the dictionary, the mask's type.
        Op::VectorBinary { mask, .. } => {
            let _ = writeln!(
                out,
                "sentient.vector_binary mask({}) {} : index",
                print::val(*mask),
                dict(&compute_attrs(op))
            );
        }
        Op::VectorUnary { mask, .. } => {
            let _ = writeln!(
                out,
                "sentient.vector_unary mask({}) {} : index",
                print::val(*mask),
                dict(&compute_attrs(op))
            );
        }
        Op::VectorTernary { mask, .. } => {
            let rendered = mask.map_or_else(|| "mask()".to_owned(), |m| format!("mask({})", print::val(m)));
            let _ = writeln!(
                out,
                "sentient.vector_ternary {} {} : index",
                rendered,
                dict(&compute_attrs(op))
            );
        }
        Op::VectorMac { mask, .. } => {
            let rendered = mask.map_or_else(|| "mask()".to_owned(), |m| format!("mask({})", print::val(m)));
            let _ = writeln!(
                out,
                "sentient.vector_mac {} {} : index",
                rendered,
                dict(&compute_attrs(op))
            );
        }
        // The remaining ops carry syntax this island does not yet write. See the module note.
        Op::For { .. }
        | Op::If { .. }
        | Op::LoadAndSend { .. }
        | Op::ReceiveAndStore { .. }
        | Op::LoadAndStore { .. }
        | Op::LoadComputeAndSend { .. }
        | Op::LoadAndExtractScalar { .. }
        | Op::ReceiveAndExtractScalar { .. }
        | Op::Splat { .. }
        | Op::Samv { .. }
        | Op::Opaque { .. } => {
            unimplemented!(
                "syntax not yet read from SentientOps.cpp's custom printer for this op; \
                 see islands/sentient/dialects/sentient.rs's module note"
            )
        }
    }
}

/// `regLocale` PLUS `regIndex` — ⭐ THE `.td` DECLARES THEM AS A PAIR, on every op that has them
/// (`scalar_add`, `scalar_sub`, `scalar_copy`, `load_and_send`, `receive_and_store`,
/// `load_compute_and_send`, `receive_and_extract_scalar`). Carrying them together is what the
/// reference's own declaration says they are, and it keeps this helper under the argument count that
/// would otherwise need an `#[allow]` — which this crate does not permit.
#[derive(Debug, Clone, Copy)]
struct ScalarReg {
    /// `regLocale`.
    locale: RegType,
    /// `regIndex` — ⛔ `None` is the `.td`'s `-1`, unassigned.
    index: Option<RegIndex>,
}

/// `scalar_add` and `scalar_sub`, whose printed shape the `.td` declares identically.
fn scalar_binary(out: &mut String, mnemonic: &str, result: Val, lhs: Val, rhs: Val, reg: ScalarReg) {
    let mut attrs = vec![attr("reg_locale", &quoted(reg.locale.spelling()))];
    if let Some(index) = reg.index {
        attrs.push(attr("reg_index", &format!("{} : i32", index.get())));
    }
    attrs.sort();
    let _ = writeln!(
        out,
        "{} = sentient.{mnemonic} {}, {} {} : index, index",
        print::val(result),
        print::val(lhs),
        print::val(rhs),
        dict(&attrs)
    );
}

/// THE ATTRIBUTE DICTIONARY OF ONE COMPUTE OP, sorted.
///
/// ⛔ SORTED, NOT DECLARATION-ORDERED — see [`emit`]'s note.
fn compute_attrs(op: &Op) -> Vec<String> {
    let mut attrs = Vec::new();
    let mut operand = |tag: &str, o: &Operand| {
        attrs.push(attr(&format!("op{tag}"), &quoted(&o.port.spelling())));
        attrs.push(attr(
            &format!("op{tag}Forwarding"),
            &port_array(&o.forwarding),
        ));
        attrs.push(attr(
            &format!("op{tag}Precision"),
            &quoted(o.precision.spelling()),
        ));
        if let Some(id) = o.data_id {
            attrs.push(attr(&format!("op{tag}DataID"), &format!("{id} : si32")));
        }
        if let Some(id) = o.port_id {
            attrs.push(attr(&format!("op{tag}PortID"), &format!("{id} : si32")));
        }
        if o.unroll_incr {
            attrs.push(attr(&format!("unrollIncrOp{tag}"), "true"));
        }
    };
    let (result, compute_precision, fold_mode, unroll_factor, dbg_name) = match op {
        Op::VectorMac {
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
            dbg_name,
            ..
        } => {
            operand("A", op_a);
            operand("B", op_b);
            operand("C", op_c);
            attrs.push(attr("mode", &quoted(mode.spelling())));
            if *xrf_read_incr != 0 {
                attrs.push(attr("xrfReadIncr", &format!("{xrf_read_incr} : i32")));
            }
            if *xrf_write_incr != 0 {
                attrs.push(attr("xrfWriteIncr", &format!("{xrf_write_incr} : i32")));
            }
            (result, compute_precision, fold_mode, unroll_factor, dbg_name)
        }
        Op::VectorBinary {
            op_a,
            op_b,
            binary_op,
            result,
            compute_precision,
            fold_mode,
            unroll_factor,
            dbg_name,
            ..
        } => {
            operand("A", op_a);
            operand("B", op_b);
            attrs.push(attr("binaryOp", &quoted(&binary_op.op().spelling())));
            // ⭐ NO CHECK NEEDED: only the forwarding arm carries a port, and only the seven legal
            // operators can be named in it.
            if let Binary::Forwarding { to, unroll_incr, .. } = binary_op {
                attrs.push(attr("LogicalResultForwarding", &quoted(&to.spelling())));
                if *unroll_incr {
                    attrs.push(attr("unrollIncrLogicalResult", "true"));
                }
            }
            (result, compute_precision, fold_mode, unroll_factor, dbg_name)
        }
        Op::VectorUnary {
            op_a,
            unary_op,
            result,
            compute_precision,
            fold_mode,
            unroll_factor,
            dbg_name,
            ..
        } => {
            operand("A", op_a);
            attrs.push(attr("unary_op", &quoted(&unary_op.spelling())));
            (result, compute_precision, fold_mode, unroll_factor, dbg_name)
        }
        Op::VectorTernary {
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
            ..
        } => {
            operand("A", op_a);
            operand("B", op_b);
            operand("C", op_c);
            attrs.push(attr("ternaryOp", &quoted(ternary_op.spelling())));
            if *unroll_incr_logical_result {
                attrs.push(attr("unrollIncrLogicalResult", "true"));
            }
            (result, compute_precision, fold_mode, unroll_factor, dbg_name)
        }
        _ => unreachable!("compute_attrs called on an op that is not one of the four computes"),
    };
    attrs.push(attr("ResultForwarding", &port_array(&result.forwarding)));
    attrs.push(attr(
        "ResultPrecision",
        &quoted(result.precision.spelling()),
    ));
    if result.unroll_incr {
        attrs.push(attr("unrollIncrResult", "true"));
    }
    attrs.push(attr(
        "ComputePrecision",
        &quoted(compute_precision.spelling()),
    ));
    if let Some(mode) = fold_mode {
        attrs.push(attr("fold_mode", &quoted(mode.spelling())));
    }
    attrs.push(attr("unrollFactor", &quoted(unroll_factor.spelling())));
    if let Some(name) = dbg_name {
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
        attrs.push(attr("chunk_size", &format!("{} : i32", extent.chunk_size.0)));
    }
    if extent.chunk_stride != Elements(1) {
        attrs.push(attr(
            "chunk_stride",
            &format!("{} : i32", extent.chunk_stride.0),
        ));
    }
    if extent.burst_size != Elements(0) {
        attrs.push(attr("burst_size", &format!("{} : i32", extent.burst_size.0)));
    }
    attrs
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
    name.as_ref().map_or_else(String::new, |n| attr("dbgName", &quoted(n)))
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
