// SPDX-License-Identifier: Apache-2.0
//! `Isa::InstOperand` — every field an instruction word can have.
//!
//! A port of the enum at `/project_src/deeptools/sys-arch-spec/isa/isa.hpp:48-139`, in its DECLARATION ORDER.
//!
//! # ⛔⛔ THE ORDER IS THE WIRE ORDER, WHICH IS WHY THIS IS AN ENUM AND NOT A SET OF STRINGS
//!
//! `InstrInfo::instFields_` is a `std::map<OperandT, OperandAttr>` (`progir.h:284`), so a senprog line emits its
//! terms in this enum's order — NOT in bit order. A real `SFP_LOGICAL` runs
//! `be@35 foldctrl@62 fwdencoding@28 imm@12 mask@42 src0@6 src2@18 tgtencoding@30 …`: ascending here, unordered in
//! the shift. Derive `Ord` from this declaration and a `BTreeMap` keyed on it produces the wire order for free;
//! keep the names as text and the order becomes whatever the producer happened to push.
//!
//! ⭐ AND IT IS WHAT LETS THE FIELD TABLE LIVE IN `src/`. `ddl/isa_fields.rs` sits outside `src/` for one reason —
//! it holds field names as `&'static str`, and this crate bars strings from islands. Only `build.rs` can read a
//! file outside `src/`, so that placement handed the whole ISA to `build.rs`, and the encoder followed its data
//! there. With the names as variants there are no strings, so the table has no reason to be outside `src/` and
//! `build.rs` has no reason to own the ISA.
//!
//! ⛔ `NumOperands` IS NOT PORTED. It is the C++'s count-terminator (`isa.hpp:138`), not a field: a variant for it
//! would be a value no instruction can hold.

/// One field of an instruction word — `Isa::InstOperand` (`isa.hpp:48-139`).
///
/// ⛔ THE DERIVED `Ord` IS LOAD-BEARING. It reproduces the C++ enum's order, which is the order a senprog's terms
/// are written in; reordering these variants alphabetically would silently reorder every instruction's operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operand {
    /// `be`.
    Be,
    /// `burst`.
    Burst,
    /// `burstsize`.
    Burstsize,
    /// `byteshift`.
    Byteshift,
    /// `chunksize`.
    Chunksize,
    /// `chunkstride`.
    Chunkstride,
    /// `cmp_imm`.
    CmpImm,
    /// `coalesce`.
    Coalesce,
    /// `consumertag`.
    Consumertag,
    /// `datatype_virtual`.
    DatatypeVirtual,
    /// `drm`.
    Drm,
    /// `dyn_loop`.
    DynLoop,
    /// `elemidx`.
    Elemidx,
    /// `endbit`.
    Endbit,
    /// `fma8ctrlsrc0`.
    Fma8ctrlsrc0,
    /// `fma8ctrlsrc2`.
    Fma8ctrlsrc2,
    /// `foldctrl`.
    Foldctrl,
    /// `fpuop`.
    Fpuop,
    /// `fwdencoding`.
    Fwdencoding,
    /// `group`.
    Group,
    /// `hwsel`.
    Hwsel,
    /// `ibr`.
    Ibr,
    /// `implicit`.
    Implicit,
    /// `imm`.
    Imm,
    /// `ififo_conv`.
    IfifoConv,
    /// `isimm`.
    Isimm,
    /// `jcr_select`.
    JcrSelect,
    /// `jcr_target`.
    JcrTarget,
    /// `ldtype`.
    Ldtype,
    /// `lrfimm`.
    Lrfimm,
    /// `mask`.
    Mask,
    /// `maskall`.
    Maskall,
    /// `mode`.
    Mode,
    /// `mvridx`.
    Mvridx,
    /// `node`.
    Node,
    /// `numvalidentry`.
    Numvalidentry,
    /// `pc_target`.
    PcTarget,
    /// `permute`.
    Permute,
    /// `precision`.
    Precision,
    /// `producertag`.
    Producertag,
    /// `rdptr_imm`.
    RdptrImm,
    /// `rdptr_upd`.
    RdptrUpd,
    /// `readibr`.
    Readibr,
    /// `reluop`.
    Reluop,
    /// `replica`.
    Replica,
    /// `rottype`.
    Rottype,
    /// `rsm_src0`.
    RsmSrc0,
    /// `rsm_src1`.
    RsmSrc1,
    /// `rsm_state_reg`.
    RsmStateReg,
    /// `rsm_tgtrf`.
    RsmTgtrf,
    /// `rsm_unroll`.
    RsmUnroll,
    /// `scale_array`.
    ScaleArray,
    /// `scaleidx`.
    Scaleidx,
    /// `sliceidxsl`.
    Sliceidxsl,
    /// `soft`.
    Soft,
    /// `splat`.
    Splat,
    /// `src0`.
    Src0,
    /// `src1`.
    Src1,
    /// `src2`.
    Src2,
    /// `src3`.
    Src3,
    /// `startbit`.
    Startbit,
    /// `stride`.
    Stride,
    /// `sttype`.
    Sttype,
    /// `subroutine`.
    Subroutine,
    /// `subslicesize`.
    Subslicesize,
    /// `subwordlen`.
    Subwordlen,
    /// `syncdest`.
    Syncdest,
    /// `synctag`.
    Synctag,
    /// `tgtdatafifo`.
    Tgtdatafifo,
    /// `tgte`.
    Tgte,
    /// `tgtencoding`.
    Tgtencoding,
    /// `tgtl0`.
    Tgtl0,
    /// `tgtlx`.
    Tgtlx,
    /// `tgtpe`.
    Tgtpe,
    /// `tgtpt`.
    Tgtpt,
    /// `tgtrf`.
    Tgtrf,
    /// `tgts`.
    Tgts,
    /// `tgtsfp`.
    Tgtsfp,
    /// `tilesize`.
    Tilesize,
    /// `unrlfldsrc0`.
    Unrlfldsrc0,
    /// `unrlfldsrc1`.
    Unrlfldsrc1,
    /// `unrlfldsrc2`.
    Unrlfldsrc2,
    /// `unrlfldtgt`.
    Unrlfldtgt,
    /// `unroll`.
    Unroll,
    /// `usejcr`.
    Usejcr,
    /// `wsllen`.
    Wsllen,
    /// `wrptr_imm`.
    WrptrImm,
    /// `wrptr_upd`.
    WrptrUpd,
    /// `xslinner`.
    Xslinner,
}

impl Operand {
    /// WHETHER THE ENCODED WORD CARRIES THIS FIELD AT ALL — `Isa::isFieldVirtual` (`isa.hpp:315-317`).
    ///
    /// ```text
    /// inline bool isFieldVirtual(InstOperand operandName) const {
    ///   return operandName == InstOperand::datatype_virtual;
    /// }
    /// ```
    ///
    /// ⛔⛔ EXACTLY ONE OPERAND, AND IT IS DECIDED BY NAME. `defineField` then asserts the consequence —
    /// `if (isFieldVirtual(name)) DT_CHECK(bitPos >= 100)` (`isa.cpp:179-181`) — so the position follows from
    /// virtuality and not the other way round. A field that is virtual has a `bitPos` the word cannot hold, and
    /// shifting a 64-bit word by it is undefined; the encoder must skip it on the strength of its NAME.
    ///
    /// ⭐ THE `/` PAIR ANSWERS YES IF EITHER HALF DOES (`isa.hpp:319-323`), which [`crate::fields::FieldName`]
    /// resolves before this is asked.
    pub const fn is_virtual(self) -> bool {
        matches!(self, Self::DatatypeVirtual)
    }

    /// `NumOperands` — the array width every `Instruction` allocates.
    pub const COUNT: usize = Self::ALL.len();

    /// EVERY FIELD, in `Isa::InstOperand`'s declaration order.
    ///
    /// ⭐ THE ORDER OF THIS ARRAY IS THE ENUM'S, so a `const` assertion can check the two agree without a second
    /// list to drift from.
    pub const ALL: [Self; 89] = [
        Self::Be,
        Self::Burst,
        Self::Burstsize,
        Self::Byteshift,
        Self::Chunksize,
        Self::Chunkstride,
        Self::CmpImm,
        Self::Coalesce,
        Self::Consumertag,
        Self::DatatypeVirtual,
        Self::Drm,
        Self::DynLoop,
        Self::Elemidx,
        Self::Endbit,
        Self::Fma8ctrlsrc0,
        Self::Fma8ctrlsrc2,
        Self::Foldctrl,
        Self::Fpuop,
        Self::Fwdencoding,
        Self::Group,
        Self::Hwsel,
        Self::Ibr,
        Self::Implicit,
        Self::Imm,
        Self::IfifoConv,
        Self::Isimm,
        Self::JcrSelect,
        Self::JcrTarget,
        Self::Ldtype,
        Self::Lrfimm,
        Self::Mask,
        Self::Maskall,
        Self::Mode,
        Self::Mvridx,
        Self::Node,
        Self::Numvalidentry,
        Self::PcTarget,
        Self::Permute,
        Self::Precision,
        Self::Producertag,
        Self::RdptrImm,
        Self::RdptrUpd,
        Self::Readibr,
        Self::Reluop,
        Self::Replica,
        Self::Rottype,
        Self::RsmSrc0,
        Self::RsmSrc1,
        Self::RsmStateReg,
        Self::RsmTgtrf,
        Self::RsmUnroll,
        Self::ScaleArray,
        Self::Scaleidx,
        Self::Sliceidxsl,
        Self::Soft,
        Self::Splat,
        Self::Src0,
        Self::Src1,
        Self::Src2,
        Self::Src3,
        Self::Startbit,
        Self::Stride,
        Self::Sttype,
        Self::Subroutine,
        Self::Subslicesize,
        Self::Subwordlen,
        Self::Syncdest,
        Self::Synctag,
        Self::Tgtdatafifo,
        Self::Tgte,
        Self::Tgtencoding,
        Self::Tgtl0,
        Self::Tgtlx,
        Self::Tgtpe,
        Self::Tgtpt,
        Self::Tgtrf,
        Self::Tgts,
        Self::Tgtsfp,
        Self::Tilesize,
        Self::Unrlfldsrc0,
        Self::Unrlfldsrc1,
        Self::Unrlfldsrc2,
        Self::Unrlfldtgt,
        Self::Unroll,
        Self::Usejcr,
        Self::Wsllen,
        Self::WrptrImm,
        Self::WrptrUpd,
        Self::Xslinner,
    ];

    /// THE SPELLING `Isa::to_string` GIVES — the name `isa.cpp`'s `defineField` calls this field by.
    ///
    /// ⛔ THE ONE PLACE A FIELD NAME IS TEXT, and it exists so a DIAGNOSTIC can name a field. Nothing in an island
    /// compares it, and nothing derives a field FROM a string: `strToInstOperand` is the C++'s parser and this
    /// crate has no text to parse.
    /// THE OPERAND A SPELLING NAMES — `Isa::to_instoperand` (`isa.cpp:172-176`), read backwards.
    ///
    /// ⛔ IT ANSWERS THE `/` PAIR'S HALVES SEPARATELY, which is what the C++ does: `defineField` splits
    /// `"imm/pc_target"` on the slash and converts each half on its own, so this takes ONE spelling and the
    /// splitting belongs to whoever holds the pair.
    #[must_use]
    pub fn of_spelling(spelling: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.spelling() == spelling)
    }

    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Be => "be",
            Self::Burst => "burst",
            Self::Burstsize => "burstsize",
            Self::Byteshift => "byteshift",
            Self::Chunksize => "chunksize",
            Self::Chunkstride => "chunkstride",
            Self::CmpImm => "cmp_imm",
            Self::Coalesce => "coalesce",
            Self::Consumertag => "consumertag",
            Self::DatatypeVirtual => "datatype_virtual",
            Self::Drm => "drm",
            Self::DynLoop => "dyn_loop",
            Self::Elemidx => "elemidx",
            Self::Endbit => "endbit",
            Self::Fma8ctrlsrc0 => "fma8ctrlsrc0",
            Self::Fma8ctrlsrc2 => "fma8ctrlsrc2",
            Self::Foldctrl => "foldctrl",
            Self::Fpuop => "fpuop",
            Self::Fwdencoding => "fwdencoding",
            Self::Group => "group",
            Self::Hwsel => "hwsel",
            Self::Ibr => "ibr",
            Self::Implicit => "implicit",
            Self::Imm => "imm",
            Self::IfifoConv => "ififo_conv",
            Self::Isimm => "isimm",
            Self::JcrSelect => "jcr_select",
            Self::JcrTarget => "jcr_target",
            Self::Ldtype => "ldtype",
            Self::Lrfimm => "lrfimm",
            Self::Mask => "mask",
            Self::Maskall => "maskall",
            Self::Mode => "mode",
            Self::Mvridx => "mvridx",
            Self::Node => "node",
            Self::Numvalidentry => "numvalidentry",
            Self::PcTarget => "pc_target",
            Self::Permute => "permute",
            Self::Precision => "precision",
            Self::Producertag => "producertag",
            Self::RdptrImm => "rdptr_imm",
            Self::RdptrUpd => "rdptr_upd",
            Self::Readibr => "readibr",
            Self::Reluop => "reluop",
            Self::Replica => "replica",
            Self::Rottype => "rottype",
            Self::RsmSrc0 => "rsm_src0",
            Self::RsmSrc1 => "rsm_src1",
            Self::RsmStateReg => "rsm_state_reg",
            Self::RsmTgtrf => "rsm_tgtrf",
            Self::RsmUnroll => "rsm_unroll",
            Self::ScaleArray => "scale_array",
            Self::Scaleidx => "scaleidx",
            Self::Sliceidxsl => "sliceidxsl",
            Self::Soft => "soft",
            Self::Splat => "splat",
            Self::Src0 => "src0",
            Self::Src1 => "src1",
            Self::Src2 => "src2",
            Self::Src3 => "src3",
            Self::Startbit => "startbit",
            Self::Stride => "stride",
            Self::Sttype => "sttype",
            Self::Subroutine => "subroutine",
            Self::Subslicesize => "subslicesize",
            Self::Subwordlen => "subwordlen",
            Self::Syncdest => "syncdest",
            Self::Synctag => "synctag",
            Self::Tgtdatafifo => "tgtdatafifo",
            Self::Tgte => "tgte",
            Self::Tgtencoding => "tgtencoding",
            Self::Tgtl0 => "tgtl0",
            Self::Tgtlx => "tgtlx",
            Self::Tgtpe => "tgtpe",
            Self::Tgtpt => "tgtpt",
            Self::Tgtrf => "tgtrf",
            Self::Tgts => "tgts",
            Self::Tgtsfp => "tgtsfp",
            Self::Tilesize => "tilesize",
            Self::Unrlfldsrc0 => "unrlfldsrc0",
            Self::Unrlfldsrc1 => "unrlfldsrc1",
            Self::Unrlfldsrc2 => "unrlfldsrc2",
            Self::Unrlfldtgt => "unrlfldtgt",
            Self::Unroll => "unroll",
            Self::Usejcr => "usejcr",
            Self::Wsllen => "wsllen",
            Self::WrptrImm => "wrptr_imm",
            Self::WrptrUpd => "wrptr_upd",
            Self::Xslinner => "xslinner",
        }
    }

    /// THIS FIELD'S SPELLING AS A ONE-ELEMENT SLICE, so [`crate::fields::FieldName::spellings`] can stay a
    /// `const fn` — a const context cannot build an array from a value.
    pub const fn spelling_slice(self) -> &'static [&'static str] {
        match self {
            Self::Be => &["be"],
            Self::Burst => &["burst"],
            Self::Burstsize => &["burstsize"],
            Self::Byteshift => &["byteshift"],
            Self::Chunksize => &["chunksize"],
            Self::Chunkstride => &["chunkstride"],
            Self::CmpImm => &["cmp_imm"],
            Self::Coalesce => &["coalesce"],
            Self::Consumertag => &["consumertag"],
            Self::DatatypeVirtual => &["datatype_virtual"],
            Self::Drm => &["drm"],
            Self::DynLoop => &["dyn_loop"],
            Self::Elemidx => &["elemidx"],
            Self::Endbit => &["endbit"],
            Self::Fma8ctrlsrc0 => &["fma8ctrlsrc0"],
            Self::Fma8ctrlsrc2 => &["fma8ctrlsrc2"],
            Self::Foldctrl => &["foldctrl"],
            Self::Fpuop => &["fpuop"],
            Self::Fwdencoding => &["fwdencoding"],
            Self::Group => &["group"],
            Self::Hwsel => &["hwsel"],
            Self::Ibr => &["ibr"],
            Self::Implicit => &["implicit"],
            Self::Imm => &["imm"],
            Self::IfifoConv => &["ififo_conv"],
            Self::Isimm => &["isimm"],
            Self::JcrSelect => &["jcr_select"],
            Self::JcrTarget => &["jcr_target"],
            Self::Ldtype => &["ldtype"],
            Self::Lrfimm => &["lrfimm"],
            Self::Mask => &["mask"],
            Self::Maskall => &["maskall"],
            Self::Mode => &["mode"],
            Self::Mvridx => &["mvridx"],
            Self::Node => &["node"],
            Self::Numvalidentry => &["numvalidentry"],
            Self::PcTarget => &["pc_target"],
            Self::Permute => &["permute"],
            Self::Precision => &["precision"],
            Self::Producertag => &["producertag"],
            Self::RdptrImm => &["rdptr_imm"],
            Self::RdptrUpd => &["rdptr_upd"],
            Self::Readibr => &["readibr"],
            Self::Reluop => &["reluop"],
            Self::Replica => &["replica"],
            Self::Rottype => &["rottype"],
            Self::RsmSrc0 => &["rsm_src0"],
            Self::RsmSrc1 => &["rsm_src1"],
            Self::RsmStateReg => &["rsm_state_reg"],
            Self::RsmTgtrf => &["rsm_tgtrf"],
            Self::RsmUnroll => &["rsm_unroll"],
            Self::ScaleArray => &["scale_array"],
            Self::Scaleidx => &["scaleidx"],
            Self::Sliceidxsl => &["sliceidxsl"],
            Self::Soft => &["soft"],
            Self::Splat => &["splat"],
            Self::Src0 => &["src0"],
            Self::Src1 => &["src1"],
            Self::Src2 => &["src2"],
            Self::Src3 => &["src3"],
            Self::Startbit => &["startbit"],
            Self::Stride => &["stride"],
            Self::Sttype => &["sttype"],
            Self::Subroutine => &["subroutine"],
            Self::Subslicesize => &["subslicesize"],
            Self::Subwordlen => &["subwordlen"],
            Self::Syncdest => &["syncdest"],
            Self::Synctag => &["synctag"],
            Self::Tgtdatafifo => &["tgtdatafifo"],
            Self::Tgte => &["tgte"],
            Self::Tgtencoding => &["tgtencoding"],
            Self::Tgtl0 => &["tgtl0"],
            Self::Tgtlx => &["tgtlx"],
            Self::Tgtpe => &["tgtpe"],
            Self::Tgtpt => &["tgtpt"],
            Self::Tgtrf => &["tgtrf"],
            Self::Tgts => &["tgts"],
            Self::Tgtsfp => &["tgtsfp"],
            Self::Tilesize => &["tilesize"],
            Self::Unrlfldsrc0 => &["unrlfldsrc0"],
            Self::Unrlfldsrc1 => &["unrlfldsrc1"],
            Self::Unrlfldsrc2 => &["unrlfldsrc2"],
            Self::Unrlfldtgt => &["unrlfldtgt"],
            Self::Unroll => &["unroll"],
            Self::Usejcr => &["usejcr"],
            Self::Wsllen => &["wsllen"],
            Self::WrptrImm => &["wrptr_imm"],
            Self::WrptrUpd => &["wrptr_upd"],
            Self::Xslinner => &["xslinner"],
        }
    }
}

// ⛔ THE ARRAY AND THE ENUM CANNOT DRIFT: every variant appears once, in order, and the count is the C++'s
// `NumOperands` minus the terminator itself.
const _: () = assert!(Operand::ALL.len() == 89);
const _: () = {
    let mut i = 0;
    while i < Operand::ALL.len() {
        // A variant out of place would make this comparison fail at the first swapped pair.
        assert!((Operand::ALL[i] as usize) == i);
        i += 1;
    }
};
