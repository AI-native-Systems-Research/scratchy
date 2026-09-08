//! THE PER-COMPONENT LOWERING — unit, corelet and core identity, and the component handler.
//! ⭐ ONE ProgramUnitOp SPANS THE SET: handles = cores x corelets x num_folds. The component-to-handler
//! map is CLEARED per unit; the unit-to-value map is module-wide.
//!
//! 12 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e022_retrieveGetUnitOpInSameCore` | 0 | 35 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:32` |
//! | `e023_createGetUnitOpInDifferentCore` | 0 | 18 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:73` |
//! | `e024_getMLIRTypeFromDSCDataFormat` | 0 | 53 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:97` |
//! | `e025_getAddressGranularityMultiplyFactor` | 0 | 17 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:156` |
//! | `e026_emitError` | 0 | 5 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:175` |
//! | `e027_setBuilderForDataTransfer` | 0 | 32 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:185` |
//! | `e028_getMLIRLoopFromLoopNode` | 0 | 14 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:220` |
//! | `e029_constructUniformizedAddress` | 0 | 47 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:236` |
//! | `e030_constructUniformizedFoldedAddress` | 0 | 51 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:285` |
//! | `e031_constructUniformizedFoldedDoubleBufferToggling` | 0 | 83 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:339` |
//! | `e032_constructUniformizedFoldedConstantBitStream` | 0 | 75 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:462` |
//! | `e059_constructUniformizedFoldedDestinationCore` | 1 | 36 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:425` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{Val, dataflow};
use crate::islands::dataflow_ir::ty::{ElemType, TensorCategory};
use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency};

/// A COMPONENT A HANDLE IS ASKED FOR — `component_to_handler_`'s key space, censused from what
/// reaches [`retrieve_get_unit_op_in_same_core`].
///
/// ⭐ THE THREE REGISTER FILES ARE HERE BECAUSE A TENSOR CAN BE PINNED IN ONE. Every explicit
/// caller passes a unit (`PTWEST`, `PTROW7`, `LXLU`, `PE`, `SFP`, …), but the transfer lowerings pass
/// a `storage` (`SNTransferLowering.cpp:426`, `:505`, `:601`, `:876`, `:1231`), and `pinnedComponent`
/// answers `PTXRF`, `PTARF`, `SFPLRF` or `PELRF` among the memories (`dscdefn.h:442-453`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    /// A unit in its own right — what a `dataflow.get_unit` names.
    Unit(DfirUnit),
    /// `LRFREG` — the generic register file `arch_enums.h:62-66` keeps "for DSC-level and arch-level
    /// compatibility".
    Lrfreg,
    /// `PTARF` — the PT's accumulator register file.
    PtArf,
    /// `PELRF` — the PE's register file.
    PeLrf,
    /// `SFPLRF` — the SFP's register file.
    SfpLrf,
    /// `PTXRF` — the PT's transposed register file.
    PtXrf,
}

/// THE KEY A COMPONENT IS LOOKED UP UNDER — [`Component`] with the register-file collapse applied,
/// and so three spellings shorter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    /// A unit, which is the only key whose handler can be missing.
    Unit(DfirUnit),
    /// `LRFREG`, which `PTARF`, `PELRF` and `SFPLRF` all arrive as.
    Lrfreg,
    /// `PTXRF`.
    PtXrf,
}

impl Component {
    /// THE KEY, after this lowering's own rewrite of the three named register files.
    ///
    /// ⭐ THE REWRITE IS WHY THE THREE SHARE ONE HANDLER: "Currently, PC forcing component to be
    /// LRFREG since DCC backend doesn't understand PELRF or SFPLRF yet"
    /// (`SNDSCLowering.cpp:35-37`).
    const fn key(self) -> Key {
        match self {
            Component::Unit(unit) => Key::Unit(unit),
            Component::Lrfreg | Component::PtArf | Component::PeLrf | Component::SfpLrf => {
                Key::Lrfreg
            }
            Component::PtXrf => Key::PtXrf,
        }
    }
}

/// WHAT `component_to_handler_` HOLDS FOR A KEY, as far as the reuse test reads it.
///
/// ⭐ THE TEST IS ON THE **DEFINING OP**, and the two arms are its two answers: `isa<GetUnitOp>`
/// decides whether a `corelet` attribute can be there to compare at all
/// (`SNDSCLowering.cpp:43-58`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// A `dataflow.get_unit`, and the corelet its `corelet` attribute names — [`None`] for one that
    /// carries no such attribute, which the reference's `hasAttr("corelet")` reads as an absence.
    Unit {
        /// The handle.
        handle: Val,
        /// Its `corelet` attribute.
        corelet: Option<Corelet>,
    },
    /// Anything else — every `get_local_unit` among them, which is what the register files are bound
    /// to (`DSC2ToDataflowIRUtils.hpp:176-179`).
    Local(Val),
}

/// WHAT A PROGRAM UNIT HAS ALREADY BOUND — `component_to_handler_`, CLEARED PER UNIT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handlers {
    /// The unit handles, in binding order.
    pub units: Vec<(DfirUnit, Bound)>,
    /// `LRFREG`'s handler — the ASKING unit's own register file, and a different one per unit:
    /// `PT_LRFREG` on each PT row, `PE_LRFREG` on the PE, `SFP_LRFREG` on the SFP
    /// (`DSC2ToDataflowIRUtils.hpp:176`, and twenty arms in total).
    ///
    /// ⛔ NOT AN [`Option`], AND THAT IS THE TYPE GUARD ON THE FALLBACK. Every arm binds it before
    /// any transfer is lowered, so the reference's create path is unreachable for it — which it must
    /// be, because that path names the op `senComponentsToString.at(LRFREG)` and a `get_unit` typed
    /// `"lrfreg"` is outside the sixteen-member `SentientLoadConsumer` set the next rung symbolizes
    /// `type=` against (`SentientTypes.td:556-593`). The reference agrees: its
    /// `DT_CHECK(false && "GetUnitOp must have been created already")` sits at `SNDSCLowering.cpp:64`.
    pub own_lrf: Val,
    /// `PTXRF`'s handler, bound in the same breath as `LRFREG` in every arm that has one
    /// (`DSC2ToDataflowIRUtils.hpp:178-179`). ⛔ NOT AN [`Option`], for the reason above.
    pub pt_xrf: Val,
}

impl Handlers {
    /// WHAT IS BOUND FOR A UNIT, or [`None`] for a unit this program unit never bound.
    #[must_use]
    pub fn unit(&self, unit: DfirUnit) -> Option<Bound> {
        self.units
            .iter()
            .find(|(bound, _)| *bound == unit)
            .map(|(_, held)| *held)
    }
}

/// THE ANSWER TO A HANDLE REQUEST — the binding already there, or the op that has to be created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Retrieved {
    /// `component_to_handler_`'s handle answers.
    Reused(Val),
    /// The `dataflow.get_unit` this call creates, which the caller emits and binds.
    Created(dataflow::Op),
}

/// Replaces: e022_retrieveGetUnitOpInSameCore
///
/// THE HANDLE FOR A COMPONENT OF **THIS** CORE, reusing what the program unit bound where that
/// binding names the right corelet and creating one where it does not.
///
/// ⛔⛔ THE THREE-WAY TEST IS NOT A CORELET COMPARISON WITH TWO SHORTCUTS. An L3 half answers
/// whatever the corelet, because the L3 is core-wide; a `get_unit` WITHOUT a `corelet` attribute also
/// answers, because the reference's `else` covers it; and only a `get_unit` naming a DIFFERENT
/// corelet falls through to creation (`SNDSCLowering.cpp:43-58`).
///
/// ⛔ THE CREATED OP CARRIES BOTH ATTRIBUTES AND THE ISLAND NAMES IT — the reference writes
/// `name = type = senComponentsToString.at(comp)` while [`dataflow::Op::GetUnit`] writes the
/// scheduler's `C{core}-{tag}-CL{corelet}`; that field's own note records why.
pub fn retrieve_get_unit_op_in_same_core(
    vals: &mut Values,
    handlers: &Handlers,
    comp: Component,
    core: Core,
    corelet: Corelet,
) -> Retrieved {
    match comp.key() {
        Key::Lrfreg => Retrieved::Reused(handlers.own_lrf),
        Key::PtXrf => Retrieved::Reused(handlers.pt_xrf),
        Key::Unit(unit) => match handlers.unit(unit) {
            // ⛔ THE L3 HALVES ARE CORE-WIDE: the corelet is not compared for them at all.
            Some(Bound::Unit { handle, .. }) if matches!(unit, DfirUnit::L3lu | DfirUnit::L3su) => {
                Retrieved::Reused(handle)
            }
            Some(Bound::Unit {
                handle,
                corelet: Some(named),
            }) if named == corelet => Retrieved::Reused(handle),
            // ⛔ A BINDING THAT NAMES ANOTHER CORELET IS NOT AN ANSWER — it is a second unit.
            Some(Bound::Unit {
                corelet: Some(_), ..
            })
            | None => Retrieved::Created(dataflow::Op::GetUnit {
                result: vals.mint(),
                residency: Residency::Corelet { core, corelet },
                unit,
                num_folds: None,
            }),
            Some(Bound::Unit { handle, .. }) | Some(Bound::Local(handle)) => {
                Retrieved::Reused(handle)
            }
        },
    }
}

/// Replaces: e023_createGetUnitOpInDifferentCore
///
/// A `dataflow.get_unit` FOR A UNIT OF ANOTHER CORE — one result per fold, and `core`, `corelet` and
/// `num_folds` all set (`SNDSCLowering.cpp:73-94`). ⭐ NEVER MEMOISED: the reference creates rather
/// than retrieves, and its caller reads `getResult(fold_id)` per fold (`:441-445`).
///
/// ⛔ THE REFERENCE'S RESULT COUNT IS THE MEMBER `num_folds_` WHILE ITS ATTRIBUTE IS THE PARAMETER
/// `num_folds` (`:80-81` against `:90`) — a disagreement all three callers avoid by passing
/// `num_folds_` (`SNComputeLowering.cpp:713`, `:893`, `SNDSCLowering.cpp:438`). One count here,
/// because the island prints the group's width from the attribute it carries.
pub fn create_get_unit_op_in_different_core(
    vals: &mut Values,
    unit: DfirUnit,
    core: Core,
    corelet: Corelet,
    num_folds: NumFolds,
) -> dataflow::Op {
    dataflow::Op::GetUnit {
        result: vals.mint(),
        residency: Residency::Corelet { core, corelet },
        unit,
        num_folds: Some(num_folds),
    }
}

/// Replaces: e024_getMLIRTypeFromDSCDataFormat
///
/// THE ELEMENT TYPE ONE DSC FORMAT HAS, and [`None`] where the reference's if-chain falls through to
/// `DT_ERROR("Unknown data format")` and returns a `NoneType` (`SNDSCLowering.cpp:147-149`) — which
/// among the generated formats is `BOOL` alone.
///
/// ⛔⛔ NOT [`ElemType::of`], AND THE DIFFERENCE IS 8 BITS OF ACCUMULATOR. This one takes no
/// component, so `SENINT24` is `i24` UNCONDITIONALLY (`:106-107`) where the compute-side
/// transcription makes it `i16` off the PT (`SNComputeLowering.cpp:291-297`). The two disagree in the
/// reference; a data structure's element type is THIS one.
#[must_use]
pub const fn mlir_type_from_dsc_data_format(
    format: DataType,
    category: TensorCategory,
) -> Option<ElemType> {
    match format {
        DataType::Senint4 => Some(ElemType::Int(4)),
        DataType::Senint8 => Some(ElemType::Int(8)),
        DataType::Senint24 => Some(ElemType::Int(24)),
        DataType::Senuint32 => Some(ElemType::Int(32)),
        DataType::Sen169Fp16 => Some(ElemType::F16),
        DataType::Bfloat16 => Some(ElemType::Bf16),
        DataType::IeeeFp32 => Some(ElemType::F32),
        // ⭐ `SEN053_FP8` USES E4M3 TOO, and that is the reference's own note: "temporarily using
        // E4M3 since E5M3 don't exist in MLIR" (`:139-140`).
        DataType::Sen143Fp8 | DataType::Sen053Fp8 => Some(match category {
            TensorCategory::Regular => ElemType::F8E4M3Fn,
            TensorCategory::Scaled => ElemType::MxFloat(8),
        }),
        DataType::Sen080Fp8 => Some(match category {
            TensorCategory::Regular => ElemType::F8E8M0Fnu,
            TensorCategory::Scaled => ElemType::MxFloat(8),
        }),
        DataType::Sen121Fp4 => Some(match category {
            TensorCategory::Regular => ElemType::F4E2M1Fn,
            TensorCategory::Scaled => ElemType::MxFloat(4),
        }),
        DataType::Bool => None,
    }
}

// crustify:todo: e025_getAddressGranularityMultiplyFactor
// crustify:todo: e026_emitError
// crustify:todo: e027_setBuilderForDataTransfer
// crustify:todo: e028_getMLIRLoopFromLoopNode
// crustify:todo: e029_constructUniformizedAddress
// crustify:todo: e030_constructUniformizedFoldedAddress
// crustify:todo: e031_constructUniformizedFoldedDoubleBufferToggling
// crustify:todo: e032_constructUniformizedFoldedConstantBitStream
// crustify:todo: e059_constructUniformizedFoldedDestinationCore

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// The three-way test: an L3 half ignores the corelet, a match reuses, a mismatch creates.
    #[test]
    fn a_mismatched_corelet_is_a_second_unit_and_an_l3_half_has_none() {
        let cl0 = Corelet::checked(0).expect("corelet 0");
        let cl1 = Corelet::checked(1).expect("corelet 1");
        let core = Core::checked(0).expect("core 0");
        let handlers = Handlers {
            units: vec![
                (
                    DfirUnit::L3lu,
                    Bound::Unit {
                        handle: Val(7),
                        corelet: Some(cl0),
                    },
                ),
                (
                    DfirUnit::Lxlu,
                    Bound::Unit {
                        handle: Val(8),
                        corelet: Some(cl0),
                    },
                ),
            ],
            own_lrf: Val(1),
            pt_xrf: Val(2),
        };
        let mut vals = Values::default();

        // The L3 is core-wide, so corelet 1 still reads corelet 0's handle.
        let l3 = retrieve_get_unit_op_in_same_core(
            &mut vals,
            &handlers,
            Component::Unit(DfirUnit::L3lu),
            core,
            cl1,
        );
        assert_eq!(l3, Retrieved::Reused(Val(7)));

        // The LX load unit is per corelet: corelet 0 reuses, corelet 1 gets its own op.
        let same = retrieve_get_unit_op_in_same_core(
            &mut vals,
            &handlers,
            Component::Unit(DfirUnit::Lxlu),
            core,
            cl0,
        );
        assert_eq!(same, Retrieved::Reused(Val(8)));
        let other = retrieve_get_unit_op_in_same_core(
            &mut vals,
            &handlers,
            Component::Unit(DfirUnit::Lxlu),
            core,
            cl1,
        );
        assert_eq!(
            other,
            Retrieved::Created(dataflow::Op::GetUnit {
                result: Val(0),
                residency: Residency::Corelet { core, corelet: cl1 },
                unit: DfirUnit::Lxlu,
                num_folds: None,
            })
        );

        // All four register-file spellings collapse onto the two local handlers.
        for comp in [
            Component::Lrfreg,
            Component::PtArf,
            Component::PeLrf,
            Component::SfpLrf,
        ] {
            let got = retrieve_get_unit_op_in_same_core(&mut vals, &handlers, comp, core, cl1);
            assert_eq!(got, Retrieved::Reused(Val(1)));
        }
        let xrf =
            retrieve_get_unit_op_in_same_core(&mut vals, &handlers, Component::PtXrf, core, cl1);
        assert_eq!(xrf, Retrieved::Reused(Val(2)));
    }

    /// The other core's unit carries `core`, `corelet` and the fold count.
    #[test]
    fn a_remote_unit_carries_its_core_corelet_and_fold_count() {
        let core = Core::checked(1).expect("core 1");
        let corelet = Corelet::checked(0).expect("corelet 0");
        let mut vals = Values::default();
        let op = create_get_unit_op_in_different_core(
            &mut vals,
            DfirUnit::L3lu,
            core,
            corelet,
            NumFolds(11),
        );
        assert_eq!(
            op,
            dataflow::Op::GetUnit {
                result: Val(0),
                residency: Residency::Corelet { core, corelet },
                unit: DfirUnit::L3lu,
                num_folds: Some(NumFolds(11)),
            }
        );
    }

    /// `SENINT24` is 24 bits with no component to ask, the MX pairs split on category, and `BOOL`
    /// is the chain's fall-through.
    #[test]
    fn the_accumulator_keeps_its_24_bits_and_bool_has_no_type() {
        assert_eq!(
            mlir_type_from_dsc_data_format(DataType::Senint24, TensorCategory::Regular),
            Some(ElemType::Int(24))
        );
        assert_eq!(
            mlir_type_from_dsc_data_format(DataType::Sen143Fp8, TensorCategory::Regular),
            Some(ElemType::F8E4M3Fn)
        );
        assert_eq!(
            mlir_type_from_dsc_data_format(DataType::Sen143Fp8, TensorCategory::Scaled),
            Some(ElemType::MxFloat(8))
        );
        assert_eq!(
            mlir_type_from_dsc_data_format(DataType::Sen080Fp8, TensorCategory::Regular),
            Some(ElemType::F8E8M0Fnu)
        );
        assert_eq!(
            mlir_type_from_dsc_data_format(DataType::Sen121Fp4, TensorCategory::Scaled),
            Some(ElemType::MxFloat(4))
        );
        assert_eq!(
            mlir_type_from_dsc_data_format(DataType::Bool, TensorCategory::Regular),
            None
        );
    }
}
