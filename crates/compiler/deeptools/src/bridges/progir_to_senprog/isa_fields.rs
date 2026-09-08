// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/isa/isa.cpp` — 6 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use core::marker::PhantomData;

use sys_arch_spec::fields::{
    Count, DefField, Enc, Family, FieldValue, Gen, InstrType, RegBase, TABLE_ARCH, WordBit,
};
use sys_arch_spec::operand::Operand;
use sys_arch_spec::regfile::{self, Component, Presence, RegType as RegFile};
use sys_arch_spec::{InstOpCode, table_component};

use crate::arch::{Arch, IsaGen};

/// WHICH POSITION A FIELD TAKES IN ITS INSTRUCTION TYPE'S LIST — the `int` `getFieldPosForName`
/// returns, and the index `typeToFieldBitShift.at(type)` / `typeToFieldEncoding.at(type)` are then
/// read at (`dpc.cpp:693`, `:712`).
///
/// ⛔ AN ORDINAL, NOT A BIT. What fixes it is `defineField`'s CALL ORDER for that type
/// (`isa.cpp:178`); the bit the field starts at is `WordBit` and is a different number entirely.
///
/// ⛔ NO PUBLIC CONSTRUCTOR: a position is minted by the lookup that found it, so a number nothing
/// looked up cannot be passed off as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldPos(usize);

impl FieldPos {
    /// The ordinal.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// WHETHER A UNIT HAS AN OPCODE AT THIS ARCH — what `isValidOpcode` computes, as a value.
///
/// ⛔⛔ THREE OUTCOMES, NOT A BOOL, AND THE THIRD IS THE ONE A BOOL HIDES. `defineOpcode` files an
/// opcode under `opcodeToType` when `coreArch >= minArch` and under `unsupportedOpcodes` otherwise
/// (`isa.cpp:150-159`) — so an opcode the component never names at all is in NEITHER map, which the
/// reference answers `false` to silently while printing a diagnostic for the other case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpcodeSupport {
    /// In `opcodeToType`: the instruction type whose fields this opcode's operands are declared as.
    Supported(InstrType),
    /// In `unsupportedOpcodes`: this unit gained the opcode at this generation, later than ours.
    RequiresGen(Gen),
    /// `defineOpcode` never names this opcode on this component.
    NotOnThisUnit,
}

/// ONE UNIT'S ISA — the per-component `Isa` that `isaPerUnit` maps a unit to (`dpc.cpp:645-646`).
///
/// ⛔⛔ AN INSTRUCTION TYPE IS SCOPED TO ITS COMPONENT. Type 20 on the PT and type 20 on the PE are
/// different field sets, because `typeToFieldName` is a member of a per-component `Isa` — so the
/// unit is part of every lookup below and not an argument any of them may default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Isa<A: Arch> {
    /// `myComponent`.
    unit: Component,
    /// The generation whose `defineOpcode` filter applies.
    arch: PhantomData<A>,
}

impl<A: Arch> Isa<A> {
    /// `sysDef.coreArch` (`isa.hpp:24-31`) as the table's own generation type.
    const CORE_ARCH: Gen = match A::GEN {
        IsaGen::Rcudd1a => Gen::Rcudd1a,
        IsaGen::Sen1p5 => Gen::Sen1p5,
    };

    /// ⛔⛔ THE TABLES ARE SELECTED BY A CARGO FEATURE AND THE PROGRAM'S ARCH BY A CONST GENERIC,
    /// so reading one under the other is E0080 here rather than a plausible answer — `TABLE_ARCH`
    /// records the 11 branch sites where RCUDD1A and SEN1P5 part company.
    const TABLE_IS_THIS_ARCH: () = assert!(
        Self::CORE_ARCH as u8 == TABLE_ARCH as u8,
        "the ISA field tables in this build were selected for a different generation than the \
         program's arch: one of them is wrong, and the answers would still look plausible"
    );

    /// THIS UNIT'S ISA. ⛔ The arch guard above evaluates here, so it cannot go unread.
    #[must_use]
    pub const fn of(unit: Component) -> Self {
        let () = Self::TABLE_IS_THIS_ARCH;
        Self {
            unit,
            arch: PhantomData,
        }
    }

    /// Replaces: e005_getFieldPosForName
    ///
    /// Where `field` sits in `ty`'s field list on this unit, or nothing — the reference's `-1`.
    ///
    /// ⛔ TRAP: THE TEST IS `answers_to`, NOT `==`. One position may carry TWO spellings —
    /// `MVLOOPCNT`'s bit 10 is `imm/pc_target`, 18 of the 1075 definitions — and an exact
    /// comparison for `imm` finds nothing there, which is how every loop's trip count once
    /// encoded as zero.
    #[must_use]
    pub fn field_pos_for_name(&self, ty: InstrType, field: Operand) -> Option<FieldPos> {
        table_component(self.unit)
            .fields()
            .iter()
            .filter(|def| def.ty == ty)
            .position(|def| def.name.answers_to(field))
            .map(FieldPos)
    }

    /// Replaces: e006_getOpcodeType
    ///
    /// The instruction type this opcode's fields are declared under — `opcodeToType.at(opcode)`.
    ///
    /// ⛔ TRAP: `.at` ON A MISSING KEY THROWS, and the key is missing for an opcode this
    /// generation does not have as well as for one this unit never names. Nothing here, both.
    #[must_use]
    pub fn opcode_type(&self, opcode: InstOpCode) -> Option<InstrType> {
        match self.opcode_support(opcode) {
            OpcodeSupport::Supported(ty) => Some(ty),
            OpcodeSupport::RequiresGen(_) | OpcodeSupport::NotOnThisUnit => None,
        }
    }

    /// Replaces: e007_isValidOpcode
    ///
    /// ⭐ THE DIAGNOSTIC IS THE RETURN VALUE. The reference prints which generation an unsupported
    /// opcode needs to `std::cerr` and then answers `false` (`isa.cpp:102-113`); here that
    /// generation is carried, so a caller can say it without a stream — and the `silent` argument,
    /// which exists only to suppress that print, has no counterpart.
    ///
    /// ⛔ TRAP: THE MIN-ARCH FILTER IS NOT IN THE TABLE'S `cfg`. `OPCODES_PT` on an RCUDD1A build
    /// still carries `FMA4` with `min_arch = Sen1p5`, so a row exists for an opcode this ISA has no
    /// instruction for; only this comparison separates them.
    #[must_use]
    pub fn opcode_support(&self, opcode: InstOpCode) -> OpcodeSupport {
        let Some(def) = table_component(self.unit)
            .opcodes()
            .iter()
            .find(|def| def.op == opcode.spelling())
        else {
            return OpcodeSupport::NotOnThisUnit;
        };
        if def.min_arch <= Self::CORE_ARCH {
            OpcodeSupport::Supported(def.ty)
        } else {
            OpcodeSupport::RequiresGen(def.min_arch)
        }
    }

    /// Replaces: e008_checkIfFieldExists
    ///
    /// Which position `opcode` gives `field` on this unit — `isValidOpcode` and then
    /// `getFieldPosForName(getOpcodeType(opcode), field)`.
    ///
    /// ⛔ Nothing here is `convertIr2Senprog`'s *"Illegal instruction/operand combination for this
    /// architecture"* (`dpc.cpp:678-685`), which is the only thing it does with a negative.
    #[must_use]
    pub fn field_pos(&self, opcode: InstOpCode, field: Operand) -> Option<FieldPos> {
        match self.opcode_support(opcode) {
            OpcodeSupport::Supported(ty) => self.field_pos_for_name(ty, field),
            OpcodeSupport::RequiresGen(_) | OpcodeSupport::NotOnThisUnit => None,
        }
    }

    /// WHERE AN OPERAND'S VALUE GOES — its position in `ty`'s field list and the bit it is shifted
    /// to, which `convertIr2Senprog` reads as one pair: `checkIfFieldExists` then
    /// `typeToFieldBitShift.at(instrType).at(fieldPos)` (`dpc.cpp:678`, `:712-713`).
    ///
    /// ⭐ ONE LOOKUP FOR BOTH, because the two tables are the same list under two keys — the
    /// position is an index INTO `typeToFieldBitShift[type]`, so a pair that disagreed with itself
    /// is not representable.
    #[must_use]
    pub fn field_slot(&self, ty: InstrType, field: Operand) -> Option<(FieldPos, WordBit)> {
        let pos = self.field_pos_for_name(ty, field)?;
        Some((pos, self.field_def(ty, pos)?.bit))
    }

    /// `typeToFieldEncoding.at(instrType).at(fieldPos).at(name)` — what a `DESCRIPTIVE` operand's
    /// TEXT encodes as (`dpc.cpp:691-694`), the one operand kind senprog encodes instead of printing.
    ///
    /// ⛔ TRAP: TWO WRITE RULES IN ONE MAP. `defineField` fills a named literal with
    /// `encodeMap[k] = v`, which OVERWRITES, and a register family through
    /// `tFieldEncodingMapFillRegs`'s `insert`, which does NOT (`isa.cpp:122-131`, `:196-220`) — so a
    /// literal beats a family whatever their order, and the first family to claim a name keeps it.
    #[must_use]
    pub fn field_encoding(&self, ty: InstrType, pos: FieldPos, name: &str) -> Option<FieldValue> {
        let def = self.field_def(ty, pos)?;
        let mut found = None;
        for encoding in def.encode {
            match *encoding {
                Enc::Lit(key, value) if key == name => {
                    found = Some(FieldValue::of(value.cast_unsigned()));
                }
                Enc::LitSym(key, count) if key == name => {
                    found = Some(FieldValue::of(u64::from(self.count(count))));
                }
                Enc::Family(family, count, base) if found.is_none() => {
                    found = family_encoding(family, self.count(count), base, name);
                }
                Enc::Lit(_, _) | Enc::LitSym(_, _) | Enc::Family(_, _, _) => {}
            }
        }
        found
    }

    /// The `defineField` row at one position of one type — the element BOTH `typeToFieldBitShift`
    /// and `typeToFieldEncoding` are indexed at, which is why they cannot drift apart here.
    fn field_def(&self, ty: InstrType, pos: FieldPos) -> Option<&'static DefField> {
        table_component(self.unit)
            .fields()
            .iter()
            .filter(|def| def.ty == ty)
            .nth(pos.get())
    }

    /// The named counts `initIsa` computes before its `defineField` calls.
    ///
    /// ⛔ `numRegs` IS THE PT's ARF AND EVERY OTHER UNIT's LRF from RCUDD1A up (`isa.cpp:233-237`),
    /// and `numMVRs` is always the **LXLU's**, whichever unit is asking (`isa.cpp:880-881`).
    /// ⛔ `numLCCRs` IS A LITERAL 16 in `initIsa` (`isa.cpp:228`), not a `regInfoPerUnit` row.
    fn count(&self, count: Count) -> u32 {
        match count {
            Count::Fixed(members) => members,
            Count::NumRegs => depth(
                self.unit,
                match self.unit {
                    Component::Pt if Self::CORE_ARCH >= Gen::Rcudd1a => RegFile::Arf,
                    _ => RegFile::Lrf,
                },
            ),
            Count::NumJcrs => depth(self.unit, RegFile::Jcr),
            Count::NumLccrs => 16,
            Count::NumMvrs => depth(Component::Lxlu, RegFile::Mvr),
            // `255 - 32 - 128 - 4` and `255 - 16 - 64 - 8` (`isa.cpp:1135-1136`).
            Count::AllSyncTagL3lu => 91,
            Count::AllSyncTagL3su => 167,
        }
    }
}

/// `regInfoPerUnit.at(unit).at(file).maxNum`.
///
/// ⛔ AN ABSENT FILE HAS NO MEMBERS, where the reference's `.at` throws — every file `initIsa` asks
/// a unit for is one that unit has, so the empty family is unreachable rather than handled.
fn depth(unit: Component, file: RegFile) -> u32 {
    match regfile::depth_of(unit, file) {
        Presence::Present(info) => u32::from(info.depth.get()),
        Presence::Absent => 0,
    }
}

/// `tFieldEncodingMapFillRegs` READ BACKWARDS — which member of `family` the text `name` spells, and
/// what that member encodes as (`isa.cpp:118-136`, `:196-220`).
///
/// ⛔ TRAP: DECIMAL BEATS HEX. The hex alias is only written for `i >= 10`, and with `insert`, so
/// `R11`'s decimal 11 is already in the map when `i = 17` offers `"11"` and 17 loses.
/// ⛔ TRAP: `eN` ENCODES AS 0, NOT AS N — the last burst length wraps (`isa.cpp:211-216`).
fn family_encoding(family: Family, count: u32, base: RegBase, name: &str) -> Option<FieldValue> {
    let (upper, lower) = match family {
        Family::Regs => ("R", "r"),
        Family::Jcrs => ("JCR", "jcr"),
        Family::Lccrs => ("LCCR", "lccr"),
        Family::Gtrs => ("GTR", "gtr"),
        Family::Mvrs => ("MVR", "mvr"),
        Family::BurstLen => {
            let text = name.strip_prefix('e')?;
            let length = decimal(text)?;
            if length == 0 || length > count {
                return None;
            }
            return Some(FieldValue::of(u64::from(
                if length == count { 0 } else { length },
            )));
        }
    };
    let text = name.strip_prefix(upper).or_else(|| name.strip_prefix(lower))?;
    if let Some(index) = decimal(text) {
        if index < count {
            return Some(FieldValue::of(u64::from(base.encoding_of(index))));
        }
    }
    let index = u32::from_str_radix(text, 16).ok()?;
    // ⛔ THE ROUND TRIP REJECTS A SPELLING THE MAP HAS NO KEY FOR: `sshex` writes `a`, never `0a`.
    (format!("{index:x}") == text && index >= 10 && index < count)
        .then(|| FieldValue::of(u64::from(base.encoding_of(index))))
}

/// `std::to_string(i)` read backwards — ⛔ `007` IS NOT A KEY, so the round trip is the check.
fn decimal(text: &str) -> Option<u32> {
    let value: u32 = text.parse().ok()?;
    (value.to_string() == text).then_some(value)
}

/// A MNEMONIC'S UNIT PREFIX — the six `op_code` strings `getOpCodePrefix` can return.
///
/// ⛔ `PTOP`, NOT `PT`: the PT's prefix is the one that is not its unit's own spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCodePrefix {
    /// `PTOP`.
    Ptop,
    /// `SFP`.
    Sfp,
    /// `PE`.
    Pe,
    /// `L0`.
    L0,
    /// `LX`.
    Lx,
    /// `L3`.
    L3,
}

impl OpCodePrefix {
    /// The prefix as it reaches the senprog text.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Ptop => "PTOP",
            Self::Sfp => "SFP",
            Self::Pe => "PE",
            Self::L0 => "L0",
            Self::Lx => "LX",
            Self::L3 => "L3",
        }
    }
}

/// Replaces: e012_getOpCodePrefix
///
/// ⭐ TOTAL, BECAUSE [`Component`] IS ALREADY THE GENERIC UNIT. The reference maps its argument
/// through `senCompToGenericComp` and `DT_ERROR`s on what is left — a memory, a link, a register
/// file. `Component`'s nine executing units are exactly the arms that succeed, so the error arm is
/// unreachable rather than handled.
///
/// ⛔ AND THE LOAD AND STORE HALVES SHARE A PREFIX: `L0LU` and `L0SU` both give `L0`.
#[must_use]
pub const fn op_code_prefix(unit: Component) -> OpCodePrefix {
    match unit {
        Component::Pt => OpCodePrefix::Ptop,
        Component::Sfp => OpCodePrefix::Sfp,
        Component::Pe => OpCodePrefix::Pe,
        Component::L0lu | Component::L0su => OpCodePrefix::L0,
        Component::Lxlu | Component::Lxsu => OpCodePrefix::Lx,
        Component::L3lu | Component::L3su => OpCodePrefix::L3,
    }
}

/// Replaces: e013_getOpCodeWithPrefix
///
/// The prefix, `_`, and the mnemonic — the whole opcode token a senprog line opens with
/// (`dpc.cpp:668-675`).
///
/// ⛔ THE MNEMONIC IS `instOpCodeToStr`'s (`isa.cpp:1360`), which is what [`InstOpCode::spelling`]
/// carries — `PTOP_IMA8`, not `PTOP_Ima8`.
#[must_use]
pub fn op_code_with_prefix(unit: Component, opcode: InstOpCode) -> String {
    format!("{}_{}", op_code_prefix(unit).spelling(), opcode.spelling())
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;

    /// The PT's type 10 is `defineField`'d as `imm/pc_target`, `src0`, `dyn_loop`, `be`
    /// (`isa.cpp:259-262`) — so `pc_target` is position 0 through the `/` pair, `src0` is 1, and a
    /// field of another type is nowhere.
    #[test]
    fn a_field_position_is_the_define_field_order_and_a_slash_pair_answers_to_both() {
        let pt = Isa::<Dd2>::of(Component::Pt);
        assert_eq!(
            pt.field_pos_for_name(InstrType::T10, Operand::PcTarget),
            Some(FieldPos(0))
        );
        assert_eq!(
            pt.field_pos_for_name(InstrType::T10, Operand::Imm),
            Some(FieldPos(0))
        );
        assert_eq!(
            pt.field_pos_for_name(InstrType::T10, Operand::Src0)
                .map(FieldPos::get),
            Some(1)
        );
        assert_eq!(pt.field_pos_for_name(InstrType::T10, Operand::Mode), None);
    }

    /// `defineOpcode(PTOP, FMA, 20, MPW2_ISA)` and `defineOpcode(PTOP, FMA4, 20, SEN1P5_ISA)`
    /// (`isa.cpp:239-240`): on RCUDD1A only the first is in `opcodeToType`.
    #[test]
    fn an_opcode_this_generation_lacks_has_no_type() {
        let pt = Isa::<Dd2>::of(Component::Pt);
        assert_eq!(pt.opcode_type(InstOpCode::FMA), Some(InstrType::T20));
        assert_eq!(pt.opcode_type(InstOpCode::FMA4), None);
    }

    /// The three outcomes of `isValidOpcode`, carrying which generation the middle one needs.
    #[test]
    fn opcode_support_separates_a_later_generation_from_another_unit() {
        let pt = Isa::<Dd2>::of(Component::Pt);
        assert_eq!(
            pt.opcode_support(InstOpCode::FMA),
            OpcodeSupport::Supported(InstrType::T20)
        );
        assert_eq!(
            pt.opcode_support(InstOpCode::FMA4),
            OpcodeSupport::RequiresGen(Gen::Sen1p5)
        );
        // `LD` is defined on the load units, never on the PT — neither map holds it.
        assert_eq!(
            pt.opcode_support(InstOpCode::LD),
            OpcodeSupport::NotOnThisUnit
        );
    }

    /// The PT's type 20 starts `src0`, `src1`, `src2` (`isa.cpp:304-312`), so an FMA's `src0` is
    /// position 0; `mode` belongs to type 13 and an FMA4 is not an RCUDD1A instruction at all.
    #[test]
    fn a_field_exists_only_for_an_opcode_this_unit_and_generation_has() {
        let pt = Isa::<Dd2>::of(Component::Pt);
        assert_eq!(
            pt.field_pos(InstOpCode::FMA, Operand::Src0)
                .map(FieldPos::get),
            Some(0)
        );
        assert_eq!(pt.field_pos(InstOpCode::FMA, Operand::Mode), None);
        assert_eq!(pt.field_pos(InstOpCode::FMA4, Operand::Src0), None);
    }

    /// e012: the PT's prefix is not its unit's spelling, and the load/store halves share theirs.
    #[test]
    fn the_pt_prefix_is_ptop_and_the_halves_share_theirs() {
        assert_eq!(op_code_prefix(Component::Pt), OpCodePrefix::Ptop);
        assert_eq!(
            op_code_prefix(Component::L0lu),
            op_code_prefix(Component::L0su)
        );
        assert_eq!(op_code_prefix(Component::Sfp).spelling(), "SFP");
    }

    /// e013: the whole opcode token, uppercase on both sides of the `_`.
    #[test]
    fn the_opcode_token_joins_prefix_and_mnemonic() {
        assert_eq!(
            op_code_with_prefix(Component::Pt, InstOpCode::IMA8),
            "PTOP_IMA8"
        );
    }

    /// The PT's type 10 is `imm/pc_target` at bit 6, `src0` (JCRs) at 22, `dyn_loop` at 26 and `be`
    /// at 31 (`isa.cpp:259-262`), and its JCR file is 16 deep (`regfile.rs`).
    #[test]
    fn a_slot_carries_its_bit_and_a_register_encodes_under_both_its_spellings() {
        let pt = Isa::<Dd2>::of(Component::Pt);
        assert_eq!(
            pt.field_slot(InstrType::T10, Operand::Be)
                .map(|(pos, bit)| (pos.get(), bit.get())),
            Some((3, 31))
        );
        let enc = |pos, name| {
            pt.field_encoding(InstrType::T10, FieldPos(pos), name)
                .map(FieldValue::get)
        };
        assert_eq!(enc(3, "be"), Some(1));
        assert_eq!(enc(2, "no"), Some(0));
        // ⭐ ONE REGISTER, TWO KEYS: `tFieldEncodingMapFillRegs` inserts the decimal name and, from
        // 10 up, its hex alias — both mapping to the same index (`isa.cpp:124-135`).
        assert_eq!(enc(1, "jcr10"), Some(10));
        assert_eq!(enc(1, "jcra"), Some(10));
        // ⛔ A ZERO-PADDED NAME IS NO KEY AT ALL, and neither is one past the file's depth.
        assert_eq!(enc(1, "jcr010"), None);
        assert_eq!(enc(1, "jcr16"), None);
    }
}
