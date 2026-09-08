// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/isa/isa.cpp` — 6 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use core::marker::PhantomData;

use sys_arch_spec::fields::{Gen, InstrType, TABLE_ARCH};
use sys_arch_spec::operand::Operand;
use sys_arch_spec::regfile::Component;
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
}

// crustify:todo: e012_getOpCodePrefix
//   authority: sys-arch-spec/isa/isa.cpp:1480  (22 lines)  `Isa::getOpCodePrefix`

// crustify:todo: e013_getOpCodeWithPrefix
//   authority: sys-arch-spec/isa/isa.cpp:1503  (4 lines)  `Isa::getOpCodeWithPrefix`

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
            pt.field_pos_for_name(InstrType::T10, Operand::Src0).map(FieldPos::get),
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
            pt.field_pos(InstOpCode::FMA, Operand::Src0).map(FieldPos::get),
            Some(0)
        );
        assert_eq!(pt.field_pos(InstOpCode::FMA, Operand::Mode), None);
        assert_eq!(pt.field_pos(InstOpCode::FMA4, Operand::Src0), None);
    }
}
