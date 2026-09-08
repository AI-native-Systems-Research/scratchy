//! WHAT MUST BE IN A REGISTER BEFORE ANYTHING READS IT — the reg-init accumulation and the
//! PE/SFP LRF immediate copies.
//!
//! ⛔ REGISTER-FILE ALLOCATIONS NEED `primaryDimToVal_st`'s rowSplit_/peSfpSplit_ SHARE. That
//! input was missing once and produced 199 refusals.
//!
//! 3 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e006_addToRegsToInit` | 0 | 61 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4157` |
//! | `e007_addPESFPLRFImmcopyToRegInit` | 0 | 228 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4226` |
//! | `e067_fillImmField` | 1 | 26 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4130` |

use super::int;
use crate::arch::Arch;
use crate::bridges::sentient_to_progir::lower::labels_and_regs::AddressScale;
use crate::bridges::sentient_to_progir::state::{RegGraphs, RegsToInit, UnitKey};
use crate::bridges::sentient_to_progir::uniform::instr::{
    MapMode, MappedEntry, OperandMapRefusal, UniformInstrInfo, add_entry_to_operand_map,
};
use crate::bridges::sentient_to_progir::utils::{AddrSpace, addr_wraparounded};
use crate::formats::{Bits, DataFormat};
use crate::islands::progir::OperandField;
use crate::islands::progir::RegInit;
use crate::islands::progir::ty::{FoldId, Operand, OperandValue, RegType, reg_file_of};
use crate::islands::sentient::dialects::sentient::{Reg, RegIndex, RegType as SenRegType};
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e006_addToRegsToInit
///
/// Record that every unit of this program unit must initialise each of these registers.
///
/// ⛔ `IMM` AND `JCR` ARE SKIPPED BEFORE THE FILE LOOKUP (`:4184`) — an immediate is an instruction
/// field, not a register, and the JCR is written by the jump itself.
///
/// ⛔ AND A LOCALE WITH NO FILE IS SKIPPED WHERE THE REFERENCE ABORTS (`lccr`, the XRF pointers,
/// unassigned): `.at()` throws for them, and no caller here passes one. See [`reg_file_of`].
pub fn add_to_regs_to_init(units: &[UnitKey], regs: &[Reg], regs_to_init: &mut RegsToInit) {
    for reg in regs {
        if matches!(reg.locale, SenRegType::Imm | SenRegType::Jcr) {
            continue;
        }
        // `reg_num < 0` — a result that uses no register is nothing to track (`:4188`).
        let (Some(index), Some(file)) = (reg.index, reg_file_of(reg.locale)) else {
            continue;
        };
        for unit in units {
            regs_to_init
                .entry(*unit)
                .or_default()
                .entry(file)
                .or_default()
                .insert(index);
        }
    }
}

/// A FORMAT A REGISTER INITIALISER CAN BE PACKED IN — the seven arms `scalar_const_val` accepts
/// (`:4248-4267`), its `DT_ERROR` else being every other [`DataFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegInitFormat {
    /// `SEN169_FP16` — two per word.
    Sen169Fp16,
    /// `SENINT8` — four per word.
    Senint8,
    /// `SEN143_FP8` — four per word, packed identically to `SENINT8`.
    Sen143Fp8,
    /// `SENINT4` — eight per word.
    Senint4,
    /// `SENINT2` — sixteen per word. ⛔ SCALAR ONLY; see [`VectorImm`].
    Senint2,
    /// `IEEE_FP32` — one per word, unshifted.
    IeeeFp32,
    /// `IEEE_INT32` — one per word, unshifted.
    IeeeInt32,
}

impl RegInitFormat {
    /// What `setSenDataType` records (`:4362`).
    #[must_use]
    pub const fn format(self) -> DataFormat {
        match self {
            Self::Sen169Fp16 => DataFormat::Sen169Fp16,
            Self::Senint8 => DataFormat::Senint8,
            Self::Sen143Fp8 => DataFormat::Sen143Fp8,
            Self::Senint4 => DataFormat::Senint4,
            Self::Senint2 => DataFormat::Senint2,
            Self::IeeeFp32 => DataFormat::IeeeFp32,
            Self::IeeeInt32 => DataFormat::IeeeInt32,
        }
    }

    /// ONE SCALAR ACROSS ALL 128 BITS — `scalar_const_val` (`:4244-4271`).
    ///
    /// ⭐ THE SAME WORD FOUR TIMES, always: the LRF holds 128 bits per slice and a scalar fills every
    /// one of them.
    #[must_use]
    pub const fn pack_scalar(self, input: u32) -> [u32; 4] {
        let word = match self {
            Self::Sen169Fp16 => input | input << 16,
            Self::Senint8 | Self::Sen143Fp8 => input | input << 8 | input << 16 | input << 24,
            Self::Senint4 => {
                let nibbles = input | input << 4 | input << 8 | input << 12;
                nibbles | nibbles << 16
            }
            Self::Senint2 => {
                let pairs = input | input << 2 | input << 4 | input << 6;
                let byte_pair = pairs | pairs << 8;
                byte_pair | byte_pair << 16
            }
            Self::IeeeFp32 | Self::IeeeInt32 => input,
        };
        [word, word, word, word]
    }
}

/// A VECTOR CONSTANT'S ELEMENTS AT THE WIDTH THAT FILLS 128 BITS — `vector_const_val`'s four
/// `DT_CHECK_MSG(values.size() == N)` arms (`:4282-4312`) as four types, so a miscounted vector is an
/// E0308 rather than an abort.
///
/// ⛔ NO 2-BIT VARIANT, AND THAT IS THE REFERENCE'S OWN GAP: the scalar path packs `SENINT2` and the
/// vector path `DT_ERROR`s it (`:4328-4330`), so a 64-element vector constant has no spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorImm {
    /// `IEEE_FP32` — four 32-bit values.
    Bits32([i64; 4]),
    /// `SEN169_FP16` — eight 16-bit values.
    Bits16([i64; 8]),
    /// `SENINT8`/`SEN143_FP8` — sixteen 8-bit values.
    Bits8([i64; 16]),
    /// `SENINT4` — thirty-two 4-bit values.
    Bits4([i64; 32]),
}

impl VectorImm {
    /// THE 128 BITS THESE ELEMENTS MAKE — `vector_const_val` (`:4273-4334`).
    ///
    /// ⛔⛔ THE VECTOR PATH PUTS ELEMENT 0 IN THE **HIGH** BITS AND THE SCALAR PATH IN THE LOW ONES:
    /// `v0 << 16 | v1` (`:4294`) against `input | input << 16` (`:4249`). Identical for a repeated
    /// value, reversed for a real vector — so a fp16 pair read the scalar way lands swapped.
    #[must_use]
    pub const fn pack(&self) -> [u32; 4] {
        match self {
            Self::Bits32(v) => [v[0] as u32, v[1] as u32, v[2] as u32, v[3] as u32],
            Self::Bits16(v) => [
                pack16(v[0], v[1]),
                pack16(v[2], v[3]),
                pack16(v[4], v[5]),
                pack16(v[6], v[7]),
            ],
            Self::Bits8(v) => [
                pack8(v[0], v[1], v[2], v[3]),
                pack8(v[4], v[5], v[6], v[7]),
                pack8(v[8], v[9], v[10], v[11]),
                pack8(v[12], v[13], v[14], v[15]),
            ],
            Self::Bits4(v) => [
                pack4(&[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]),
                pack4(&[v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]]),
                pack4(&[v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23]]),
                pack4(&[v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]]),
            ],
        }
    }
}

/// `fillInt32(v0, v1)` (`:4294`).
const fn pack16(v0: i64, v1: i64) -> u32 {
    (v0 as u32) << 16 | v1 as u32
}

/// `fillInt32(v0..v3)` (`:4304-4306`).
const fn pack8(v0: i64, v1: i64, v2: i64, v3: i64) -> u32 {
    (v0 as u32) << 24 | (v1 as u32) << 16 | (v2 as u32) << 8 | v3 as u32
}

/// `fillInt32(v0..v7)` (`:4315-4319`).
const fn pack4(v: &[i64; 8]) -> u32 {
    (v[0] as u32) << 28
        | (v[1] as u32) << 24
        | (v[2] as u32) << 20
        | (v[3] as u32) << 16
        | (v[4] as u32) << 12
        | (v[5] as u32) << 8
        | (v[6] as u32) << 4
        | v[7] as u32
}

/// WHAT A REGISTER INITIALISER'S CONSTANT IS — the three op kinds
/// `addPESFPLRFImmcopyToRegInit` dispatches on (`:4238-4242`), its `DT_ERROR` else being every other
/// op, which has no variant here.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstInput {
    /// `sentient::ConstantOp` — one scalar, packed across the whole register.
    Scalar {
        /// `getValue()`.
        value: i64,
        /// ⛔ `is_symbol` MAKES THE VALUE A SYMBOL ID rather than a number to pack (`:4354-4356`).
        is_symbol: bool,
    },
    /// `sentient::VectorConstantOp` — the register written out element by element.
    Vector(VectorImm),
    /// `mlir::uniform::QueryMapOp` — one constant per unit and per fold.
    Uniform(Vec<UniformConst>),
}

/// ONE UNIT'S CONSTANT UNDER A QUERY MAP — `getValuesFromKeys(units)` zipped with `unit_foldid_map_`
/// (`:4382-4386`).
#[derive(Debug, Clone, PartialEq)]
pub struct UniformConst {
    /// Which unit the key resolved to.
    pub unit: UnitKey,
    /// `unit_foldid_map_.at(unit)` — a `SdscFoldIdInput`, optional in the reference too.
    pub fold: Option<FoldId>,
    /// ⛔ `None` IS A UNIT THE MAP DOES NOT COVER, which the reference fills from another unit.
    pub value: Option<UniformValue>,
}

/// A QUERY MAP'S MAPPED CONSTANT — ⛔ NO SYMBOL ARM, because the reference refuses a symbolic
/// `sentient::ConstantOp` inside a uniform operation (`:4394-4396`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UniformValue {
    /// A `sentient::ConstantOp` the map holds.
    Scalar(i64),
    /// A `sentient::VectorConstantOp` the map holds.
    Vector(VectorImm),
}

/// THE SPLAT A REGISTER INITIALISER MAY COME FROM — the witness for the three head checks
/// (`:4229-4236`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegInitSplat;

impl RegInitSplat {
    /// `None` WHERE THE REFERENCE ABORTS: a non-constant or non-zero mask (*"No masking is supported
    /// in register initialization"*), or an unrolled target.
    #[must_use]
    pub fn of(mask: Option<i64>, unroll_incr_result: bool) -> Option<RegInitSplat> {
        match (mask, unroll_incr_result) {
            (Some(0), false) => Some(RegInitSplat),
            _ => None,
        }
    }
}

/// Replaces: e007_addPESFPLRFImmcopyToRegInit
///
/// Put a PE or SFP LRF's initial 128 bits into every unit's register graph.
///
/// ⛔ ALWAYS `RegType::LRF` AND ALWAYS `reg_val` — all three arms write the same file and index; only
/// the value differs (`:4365`, `:4375`, `:4447`).
///
/// ⛔ AN UNCOVERED UNIT TAKES ANOTHER UNIT'S WHOLE FOLD MAP — *"use a random value for this unit"*
/// (`:4413-4421`). Kept: the register must be initialised or the program reads it undefined.
pub fn add_pe_sfp_lrf_immcopy_to_reg_init(
    units: &[UnitKey],
    input: &ConstInput,
    op_precision: RegInitFormat,
    reg_val: RegIndex,
    _splat: RegInitSplat,
    reg_graph: &mut RegGraphs,
) {
    match input {
        ConstInput::Scalar { value, is_symbol } => {
            let held = if *is_symbol {
                OperandValue::VariableSymbol(*value)
            } else {
                OperandValue::Int128(op_precision.pack_scalar(*value as u32))
            };
            let val_attr = Operand::every(held).in_format(op_precision.format());
            for unit in units {
                add_lrf_init(reg_graph, *unit, reg_val, &val_attr);
            }
        }
        ConstInput::Vector(imm) => {
            let val_attr =
                Operand::every(OperandValue::Int128(imm.pack())).in_format(op_precision.format());
            for unit in units {
                add_lrf_init(reg_graph, *unit, reg_val, &val_attr);
            }
        }
        ConstInput::Uniform(consts) => {
            // `imm128bit_vals` — one per-fold map per unit, filled only where the map has a value.
            let mut per_unit: Vec<(UnitKey, Vec<(Option<FoldId>, [u32; 4])>)> = Vec::new();
            let mut empty_units: Vec<UnitKey> = Vec::new();
            for entry in consts {
                let Some(value) = entry.value else {
                    empty_units.push(entry.unit);
                    continue;
                };
                let packed = match value {
                    UniformValue::Scalar(scalar) => op_precision.pack_scalar(scalar as u32),
                    UniformValue::Vector(imm) => imm.pack(),
                };
                set_fold(&mut per_unit, entry.unit, entry.fold, packed);
            }
            // ⛔ WITH NO COVERED UNIT AT ALL there is nothing to copy, where the reference's
            // `DT_CHECK(!imm128bit_vals.empty())` aborts.
            if let Some((_, donor)) = per_unit.first().cloned() {
                for unit in empty_units {
                    match per_unit.iter_mut().find(|(at, _)| *at == unit) {
                        Some(entry) => entry.1 = donor.clone(),
                        None => per_unit.push((unit, donor.clone())),
                    }
                }
            }
            for (unit, folds) in &per_unit {
                let Some((_, front)) = folds.first() else {
                    continue;
                };
                let mut imm_operand = Operand::default();
                if folds.iter().all(|(_, value)| value == front) {
                    imm_operand.set(None, OperandValue::Int128(*front));
                } else {
                    for (fold, value) in folds {
                        imm_operand.set(*fold, OperandValue::Int128(*value));
                    }
                }
                let imm_operand = imm_operand.in_format(op_precision.format());
                add_lrf_init(reg_graph, *unit, reg_val, &imm_operand);
            }
        }
    }
}

/// `reg_graph[unit].addRegInit(reg_val, val_attr, RegType::LRF)`.
fn add_lrf_init(reg_graph: &mut RegGraphs, unit: UnitKey, index: RegIndex, value: &Operand) {
    reg_graph.add_reg_init(
        unit,
        RegInit {
            file: RegType::Lrf,
            index,
            value: value.clone(),
        },
    );
}

/// `imm128bit_vals[unit][fold] = packed` — ⭐ A PLAIN ASSIGNMENT, not [`Operand::set`]'s merge.
fn set_fold(
    per_unit: &mut Vec<(UnitKey, Vec<(Option<FoldId>, [u32; 4])>)>,
    unit: UnitKey,
    fold: Option<FoldId>,
    packed: [u32; 4],
) {
    let folds = match per_unit.iter_mut().find(|(at, _)| *at == unit) {
        Some(entry) => &mut entry.1,
        None => {
            per_unit.push((unit, Vec::new()));
            &mut per_unit.last_mut().expect("just pushed").1
        }
    };
    match folds.iter_mut().find(|(at, _)| *at == fold) {
        Some(entry) => entry.1 = packed,
        None => folds.push((fold, packed)),
    }
}

/// WHERE ONE IMMEDIATE FIELD'S VALUE COMES FROM — the two defining ops `fillImmField` dispatches on
/// (`:4131`, `:4141`). ⛔ AND THERE IS NO THIRD: any other op leaves the field UNSET, which is the
/// reference's silent fallthrough.
#[derive(Debug, Clone, PartialEq)]
pub enum ImmSource {
    /// `sentient.constant` — one value for every unit.
    Constant(i64),
    /// `uniform.query_map` — one value per unit; see [`add_entry_to_operand_map`].
    Mapped(Vec<MappedEntry>),
}

/// Replaces: e067_fillImmField
///
/// Fill one immediate field: a constant is scaled into the unit's address granularity and shared, a
/// mapping becomes the instruction's own per-unit map.
/// ⛔ `setOperandMap` REPLACES THE MAP (`UniformInstrAndBlock.hpp:133`) — every per-unit field the
/// instruction already carried is dropped, not merged.
/// ⚠️ AND THE SCALING IS FLOATING POINT HERE against integer in [`get_reg_imm_vals`] (`:4136-4137` vs
/// `LowerSentientHelper.cpp:1146`), so one immediate can round two ways.
pub fn fill_imm_field<A: Arch>(
    instr: &mut UniformInstrInfo,
    field: OperandField,
    source: &ImmSource,
    comp: Component,
    element_size: Bits,
    scale: AddressScale,
) -> Vec<OperandMapRefusal> {
    match source {
        ImmSource::Constant(value) => {
            let imm = *value as f64 * f64::from(element_size.0) / 8.0 / f64::from(scale.get());
            // The L0's MODLRF immediate is 10 bits and its addressing is cyclic (`:4138-4141`).
            let imm = addr_wraparounded::<A>(imm as i64, AddrSpace::Unit(comp));
            instr.set_common_field(field, int(imm));
            Vec::new()
        }
        ImmSource::Mapped(entries) => {
            let mode = match comp {
                Component::L0lu | Component::L0su => MapMode::L0WrapAround,
                _ => MapMode::None,
            };
            let mapped = add_entry_to_operand_map(
                entries,
                mode,
                f64::from(element_size.0) / 8.0 / f64::from(scale.get()),
            );
            instr.operand_map = mapped
                .value
                .map_or_else(Vec::new, |value| vec![(field, value)]);
            mapped.refused
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Component, ConstInput, ImmSource, OperandField, RegInitFormat, RegInitSplat, UniformConst,
        UniformValue, add_pe_sfp_lrf_immcopy_to_reg_init, add_to_regs_to_init, fill_imm_field, int,
    };
    use crate::arch::Target;
    use crate::bridges::sentient_to_progir::lower::labels_and_regs::address_scale;
    use crate::bridges::sentient_to_progir::state::{RegGraphs, RegsToInit, UnitKey};
    use crate::bridges::sentient_to_progir::uniform::instr::{
        MapMode, MappedEntry, MappedOp, UniformInstrInfo, add_entry_to_operand_map,
    };
    use crate::formats::Bits;
    use crate::islands::progir::OpCode;
    use crate::islands::progir::ty::{OperandValue, RegType};
    use crate::islands::sentient::dialects::sentient::{Reg, RegIndex, RegType as SenRegType};
    use crate::units::{Core, Corelet, DfirUnit};

    fn sfp(core: u32) -> UnitKey {
        UnitKey {
            unit: DfirUnit::Sfp,
            core: Core::checked(core).expect("every arch has cores 0 and 1"),
            corelet: Corelet::checked(0),
        }
    }

    /// `IMM` and `JCR` are dropped, an unassigned result is dropped, and what survives lands on every
    /// unit of the program unit.
    #[test]
    fn only_assigned_register_locales_are_tracked() {
        let regs = [
            Reg {
                locale: SenRegType::Lrf,
                index: Some(RegIndex::at::<3>()),
            },
            Reg {
                locale: SenRegType::Imm,
                index: Some(RegIndex::at::<0>()),
            },
            Reg {
                locale: SenRegType::Jcr,
                index: Some(RegIndex::at::<1>()),
            },
            Reg {
                locale: SenRegType::Lar,
                index: None,
            },
        ];
        let mut regs_to_init = RegsToInit::new();
        add_to_regs_to_init(&[sfp(0), sfp(1)], &regs, &mut regs_to_init);
        assert_eq!(regs_to_init.len(), 2);
        let one = &regs_to_init[&sfp(1)];
        assert_eq!(one.keys().copied().collect::<Vec<_>>(), vec![RegType::Lrf]);
        assert_eq!(
            one[&RegType::Lrf].iter().copied().collect::<Vec<_>>(),
            vec![RegIndex::at::<3>()]
        );
    }

    /// IBM'S OWN `uniform_pe_sfp_lrf_immcopy_program_header.mlir` — a query map giving core 0 the
    /// fp16 constant `0xff` and core 1 `0xfe`, whose CHECK lines read
    /// `LRF : 0 : #SEN169_FP16 : 0x00ff00ff00ff00ff00ff00ff00ff00ff` and `…00fe00fe…`.
    #[test]
    fn a_query_map_splats_each_units_own_fp16_constant() {
        let input = ConstInput::Uniform(vec![
            UniformConst {
                unit: sfp(0),
                fold: None,
                value: Some(UniformValue::Scalar(0xff)),
            },
            UniformConst {
                unit: sfp(1),
                fold: None,
                value: Some(UniformValue::Scalar(0xfe)),
            },
        ]);
        let mut reg_graph = RegGraphs::default();
        add_pe_sfp_lrf_immcopy_to_reg_init(
            &[sfp(0), sfp(1)],
            &input,
            RegInitFormat::Sen169Fp16,
            RegIndex::at::<0>(),
            RegInitSplat::of(Some(0), false).expect("an unmasked, unrolled-free splat"),
            &mut reg_graph,
        );
        for (core, word) in [(0, 0x00ff_00ffu32), (1, 0x00fe_00fe)] {
            let state = reg_graph.get(sfp(core)).expect("the unit was initialised");
            assert_eq!(state.len(), 1);
            assert_eq!(state[0].file, RegType::Lrf);
            assert_eq!(state[0].index, RegIndex::at::<0>());
            assert_eq!(
                state[0].value.value,
                crate::islands::progir::ty::PerFold::Every(OperandValue::Int128([word; 4]))
            );
            assert_eq!(
                state[0].value.format,
                Some(crate::formats::DataFormat::Sen169Fp16)
            );
        }
    }

    /// IBM'S OWN `uniformization_small_elem_size.mlir` — an `element_size = 4` LRF add of the
    /// constant 128 on an LXLU prints `LX_MODLRFIMM :: lrfimm:64 src0:0  // lrf add`. The mapped arm
    /// then REPLACES that field's map, dropping what the instruction already carried.
    #[test]
    fn a_constant_immediate_is_scaled_into_bytes_and_a_mapping_replaces_the_map() {
        let scale = address_scale::<Target>(Component::Lxlu);
        let mut instr = UniformInstrInfo::of(OpCode::MODLRFIMM);
        assert!(
            fill_imm_field::<Target>(
                &mut instr,
                OperandField::Lrfimm,
                &ImmSource::Constant(128),
                Component::Lxlu,
                Bits(4),
                scale,
            )
            .is_empty()
        );
        assert_eq!(
            instr.common_fields,
            vec![(OperandField::Lrfimm, int(64))],
            "128 * 4 bits / 8 / 1"
        );
        // A per-unit field this instruction already held, and the mapping that supersedes it.
        let lxlu = UnitKey {
            unit: DfirUnit::Lxlu,
            core: Core::checked(3).expect("core 3"),
            corelet: Corelet::checked(0),
        };
        instr.operand_map = add_entry_to_operand_map(
            &[MappedEntry {
                key: lxlu,
                fold: None,
                value: MappedOp::Constant(1),
            }],
            MapMode::None,
            1.0,
        )
        .value
        .map_or_else(Vec::new, |value| vec![(OperandField::Src0, value)]);
        assert!(
            fill_imm_field::<Target>(
                &mut instr,
                OperandField::Imm,
                &ImmSource::Mapped(vec![MappedEntry {
                    key: lxlu,
                    fold: None,
                    value: MappedOp::Constant(216),
                }]),
                Component::Lxlu,
                Bits(4),
                scale,
            )
            .is_empty()
        );
        assert_eq!(
            instr
                .operand_map
                .iter()
                .map(|(at, _)| *at)
                .collect::<Vec<_>>(),
            vec![OperandField::Imm],
            "setOperandMap replaces the whole map"
        );
        assert_eq!(
            instr.operand_map[0].1.get(lxlu),
            &crate::islands::progir::ty::Operand::every(OperandValue::Int(108))
        );
    }
}
