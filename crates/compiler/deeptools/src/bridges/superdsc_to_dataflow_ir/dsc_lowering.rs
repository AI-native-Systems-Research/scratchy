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

use super::control_flow::{PrimaryDim, mlir_loop_from_sn_loop_node};
use crate::arch::{Arch, Bytes, Target};
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::uniform::MappedTy;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, arith, dataflow, uniform, vectorchain,
};
use crate::islands::dataflow_ir::ty::{ElemType, TensorCategory, Vector};
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
    /// `LXLUSCALEREG` — the LXLU's scale-register region (`arch_enums.h:118`, spelled
    /// `"lxluscalereg"` at `arch_enums.cpp:117`), which entry 079 both retrieves as a storage and
    /// tests for (`SNTransferLowering.cpp:615`, `:628`).
    LxluScaleReg,
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
    /// `LXLUSCALEREG`, which no neighbour arm binds — so every lookup of it creates.
    LxluScaleReg,
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
            Component::LxluScaleReg => Key::LxluScaleReg,
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

impl Retrieved {
    /// THE HANDLE, WITH ANY OP THIS RETRIEVAL CREATED PUSHED FIRST — the two lines every caller of a
    /// retrieval helper writes (`SNDSCLowering.cpp:57-66`).
    #[must_use]
    pub fn bind(self, ops: &mut Vec<DfirOp>) -> Val {
        match self {
            Retrieved::Reused(handle) => handle,
            Retrieved::Created(op) => {
                let handle = match &op {
                    dataflow::Op::GetUnit { result, .. }
                    | dataflow::Op::GetLocalUnit { result, .. } => *result,
                    // ⛔ NOT REACHABLE BY CONSTRUCTION: entries 007, 022 and 044 are the only
                    // builders of this variant and each builds one of the two above.
                    other => todo!("Retrieved::Created holds {other:?}, which no retrieval builds"),
                };
                ops.push(DfirOp::Dataflow(op));
                handle
            }
        }
    }
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
/// ⛔ [`None`] IS THE REFERENCE'S `corelet_id = -1`, WHICH ITS L3 CALLER PASSES (`SNSyncLowering.cpp:36`)
/// — a core-wide residency here, because a `corelet = -1` attribute is not a corelet.
///
/// ⛔ THE CREATED OP CARRIES BOTH ATTRIBUTES AND THE ISLAND NAMES IT — the reference writes
/// `name = type = senComponentsToString.at(comp)` while [`dataflow::Op::GetUnit`] writes the
/// scheduler's `C{core}-{tag}-CL{corelet}`; that field's own note records why.
pub fn retrieve_get_unit_op_in_same_core(
    vals: &mut Values,
    handlers: &Handlers,
    comp: Component,
    core: Core,
    corelet: Option<Corelet>,
) -> Retrieved {
    match comp.key() {
        Key::Lrfreg => Retrieved::Reused(handlers.own_lrf),
        Key::PtXrf => Retrieved::Reused(handlers.pt_xrf),
        // ⛔ THE ISLAND CANNOT NAME THIS UNIT. `component_to_handler_` never binds it, so the
        // reference always CREATES here, and what it creates is a `get_unit` typed
        // `"lxluscalereg"` — a spelling [`DfirUnit`] does not carry. Entry 079 needs the KEY
        // regardless, because it is what selects its scale-register address form.
        Key::LxluScaleReg => todo!("get_unit type=\"lxluscalereg\" is not a DfirUnit"),
        Key::Unit(unit) => match handlers.unit(unit) {
            // ⛔ THE L3 HALVES ARE CORE-WIDE: the corelet is not compared for them at all.
            Some(Bound::Unit { handle, .. }) if matches!(unit, DfirUnit::L3lu | DfirUnit::L3su) => {
                Retrieved::Reused(handle)
            }
            Some(Bound::Unit {
                handle,
                corelet: Some(named),
            }) if Some(named) == corelet => Retrieved::Reused(handle),
            // ⛔ A BINDING THAT NAMES ANOTHER CORELET IS NOT AN ANSWER — it is a second unit.
            Some(Bound::Unit {
                corelet: Some(_), ..
            })
            | None => Retrieved::Created(dataflow::Op::GetUnit {
                result: vals.mint(),
                residency: match corelet {
                    Some(corelet) => Residency::Corelet { core, corelet },
                    None => Residency::CoreWide { core },
                },
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 025/110
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHERE DATA SITS, AS THE GRANULARITY TABLE KEYS IT — one variant per row of
/// `addressGranularityScalePerUnit` (`sys-arch-spec/sysdef.cpp:531-554`).
///
/// ⛔⛔ THE TABLE'S KEYS ARE THE WHOLE DOMAIN, AND THAT IS WHY THIS IS AN ENUM. The reference builds
/// `DataLocation{unit_, storage_}` from two independent `SenComponents` and reads
/// `addressGranularityScalePerUnit.at(loc)`, which **throws** for any pair the table does not carry —
/// `{PT, LX}`, `{L0LU, HBM}`, `{SFP, L0}` and every other combination of the 100-odd components. Two
/// free enums would make 10,000 spellings of which 23 are addresses; naming the rows instead makes
/// the lookup total, so the reference's throw has no counterpart here and no caller can ask for a
/// granularity the machine does not define.
///
/// ⭐ `LRFREG` AND THE UNIT'S OWN FILE ARE SEPARATE ROWS ON PURPOSE. `{SFP, LRFREG}` and
/// `{SFP, SFPLRF}` are two keys assigned in one statement (`:546-547`), as are the PE's and the PT's
/// pair — the generic register-file spelling and the specific one, both 128. Collapsing them would
/// state that the caller's two spellings are the same fact, which is the table's own claim to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataLocation {
    /// `{L3LU, HBM}` — the L3 load half reading device memory.
    L3luHbm,
    /// `{L3LU, LX}` — the L3 load half reading an LX.
    L3luLx,
    /// `{L3LU, L3LUIBR}` — the L3 load half's indirection base register.
    L3luIbr,
    /// `{L3SU, HBM}`.
    L3suHbm,
    /// `{L3SU, LX}`.
    L3suLx,
    /// `{L3SU, L3SUIBR}`.
    L3suIbr,
    /// `{LXLU, LX}`.
    LxluLx,
    /// `{LXLU, LXLUSCALEREG}` — the load unit's scale register.
    LxluScaleReg,
    /// `{LXLU, LXLUVALUE}` — the load unit's immediate value register.
    LxluValue,
    /// `{LXSU, LX}`.
    LxsuLx,
    /// `{L0LU, L0}`.
    L0luL0,
    /// `{L0SU, L0}` — ⛔ `numPTRows`, NOT 1. See [`DataLocation::granularity`].
    L0suL0,
    /// `{L0LU, L0_SCALE}`.
    L0luL0Scale,
    /// `{L0SU, L0_SCALE}` — ⛔ `numPTRows`, NOT 1.
    L0suL0Scale,
    /// `{SFP, LRFREG}`.
    SfpLrfReg,
    /// `{SFP, SFPLRF}`.
    SfpLrf,
    /// `{SFP, SFPSTATE}`.
    SfpState,
    /// `{PE, LRFREG}`.
    PeLrfReg,
    /// `{PE, PELRF}`.
    PeLrf,
    /// `{PE, PESTATE}`.
    PeState,
    /// `{PT, LRFREG}`.
    PtLrfReg,
    /// `{PT, PTARF}`.
    PtArf,
    /// `{PT, PTXRF}`.
    PtXrf,
}

impl DataLocation {
    /// HOW MANY BYTES ONE ADDRESS STEP COVERS — `addressGranularityScalePerUnit`
    /// (`sys-arch-spec/sysdef.cpp:531-554`), row for row.
    ///
    /// ⛔⛔ THE L0 STORE UNIT'S TWO ROWS ARE `numPTRows`, NOT 1. `{L0SU, L0}` and `{L0SU, L0_SCALE}`
    /// are the only rows in the table whose value is not a literal (`:542,545`): a store into the L0
    /// lands one stick per PT row, so one address step covers the whole column. Reading them as 1 —
    /// which is what the surrounding `{L0LU, ...}` rows invite — divides every L0 store address by 8
    /// on RCUDD1A and by 4 on SEN1P5.
    ///
    /// ⛔ THE ROW COUNT IS THE ARCH'S, and it is [`Target`] rather than a parameter for the reason
    /// [`crate::units::Row`] gives: the arch is a cargo feature, so this is a literal the compiler
    /// folds.
    #[must_use]
    pub const fn granularity(self) -> Bytes {
        Bytes(match self {
            Self::L3luIbr | Self::L3suIbr => 4,
            Self::LxluLx
            | Self::LxluScaleReg
            | Self::LxluValue
            | Self::LxsuLx
            | Self::L0luL0
            | Self::L0luL0Scale => 1,
            // ⛔ `numPTRows`. See above.
            Self::L0suL0 | Self::L0suL0Scale => Target::PT_ROWS as u64,
            Self::L3luHbm
            | Self::L3luLx
            | Self::L3suHbm
            | Self::L3suLx
            | Self::SfpLrfReg
            | Self::SfpLrf
            | Self::SfpState
            | Self::PeLrfReg
            | Self::PeLrf
            | Self::PeState
            | Self::PtLrfReg
            | Self::PtArf
            | Self::PtXrf => 128,
        })
    }
}

/// HOW MANY ADDRESS STEPS ONE ELEMENT OF DSC ADDRESS IS WORTH — the `double` every
/// `constructUniformized*` multiplies a DSC address by before it becomes an `index`.
///
/// ⛔⛔ NOT AN INTEGER, AND ROUNDING IT WOULD MOVE EVERY LX ADDRESS. `{LXLU, LX}` is one byte per step
/// against a 16-bit element: `1 * 8 / 16 = 0.5`, so the factor is fractional wherever the element is
/// wider than the granularity. It is a `f64` because the reference's is, and the truncation to an
/// integer happens once, at the end, in [`Factor::scale`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Factor(f64);

impl Factor {
    /// THE DSC ADDRESS THIS FACTOR SCALES TO, TRUNCATED TOWARD ZERO — the reference's
    /// `int(address * factor)`.
    ///
    /// ⚠️ THE REFERENCE NARROWS TO 32 BITS HERE AND THIS DOES NOT. `int(...)` is a 32-bit `int` at
    /// every one of the four call sites (`SNDSCLowering.cpp:265,296,349,392`), so an HBM address past
    /// 2 GiB wraps — on a 16 GiB device. Reproducing that would be reproducing a wrong address, and
    /// an `index` is 64-bit in the IR either way, so the truncation kept is the one the reference
    /// MEANS: the fractional part of the multiply.
    #[must_use]
    pub fn scale(self, address: i64) -> i64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "an f64 holds every address below 2^53 exactly, and the device has 2^34 bytes"
        )]
        #[expect(
            clippy::cast_possible_truncation,
            reason = "`int(address * factor)` — the truncation toward zero is the reference's own"
        )]
        let scaled = (address as f64 * self.0) as i64;
        scaled
    }
}

/// Replaces: e025_getAddressGranularityMultiplyFactor
///
/// **025/110** `SNDSCLowering::getAddressGranularityMultiplyFactor` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:156` (17L).
///
/// ⭐⭐ BYTES PER STEP, IN BITS, OVER BITS PER ELEMENT. `factor = granularity * 8 / bitwidth` — the
/// `* 8` converts the table's bytes to bits so the divisor can be the element width, which the
/// reference reads off the MLIR type.
///
/// ⛔⛔ THE REFERENCE'S `bitwidth == 24` ARM HAS NO COUNTERPART HERE, AND IT IS NOT A DROPPED CASE.
/// `getMLIRTypeFromDSCDataFormat` sends `SENINT24` to `builder.getIntegerType(24)` (entry 024,
/// `:110-111`), so `getIntOrFloatBitWidth` answers 24 for a format that PACKS at 16 —
/// `{DataFormats::SENINT24, 16}` (`sendefs.cpp:135`) — and `factor /= 16.0` puts the packing width
/// back. This port divides by [`DataType::bits`], which IS the packing table, so the correction is
/// already applied and `factor / bits` is right for all twelve formats. Adding the arm here would
/// divide by 16 twice.
///
/// ⭐ NO DIVISION GUARD IS NEEDED: every row of `dataFormatsToBitWidth` is 4 or more, so
/// [`DataType::bits`] is never zero and the quotient is always finite.
#[must_use]
pub fn address_granularity_multiply_factor(loc: DataLocation, precision: DataType) -> Factor {
    // `auto factor = (double)...at(loc); factor = factor * 8;`
    #[expect(
        clippy::cast_precision_loss,
        reason = "the table's rows are 1, 4, 128 and numPTRows"
    )]
    let bits_per_step = (loc.granularity().0 * 8) as f64;

    // `unsigned bitwidth = getIntOrFloatBitWidth(precision); .. factor = factor / bitwidth;`
    Factor(bits_per_step / f64::from(precision.bits().0))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 026/110
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e026_emitError
///
/// **026/110** `SNDSCLowering::emitError` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:175` (5L).
///
/// ⛔⛔ IT DOES NOT RETURN, AND THAT IS THE WHOLE FUNCTION. The MLIR `emitError` is a diagnostic the
/// caller could ignore, but the `DT_ERROR` on the next line aborts — so every caller of
/// `emitError(..)` in this lowering is a caller that stops. `-> !` is that fact in the signature:
/// a caller cannot log it and carry on, which is exactly how this bridge went blind five times
/// (`crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`).
///
/// ⭐ BOTH SENTENCES ARE CARRIED. The prefixed one names the pass and the second names the DSC node,
/// and it is the node that makes a report actionable — a translation error without it says only that
/// some node of some model failed.
pub fn emit_error(node: &str, message: &str) -> ! {
    // `module_op_->emitError("[DSC2.0 to Dataflow IR]: " + message);`
    // `DT_ERROR("Error encountered during DSC2 to DFIR translation for the node: " + dsc_->name_);`
    panic!(
        "[DSC2.0 to Dataflow IR]: {message}\n\
         Error encountered during DSC2 to DFIR translation for the node: {node}"
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 027/110
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF THE TWO MLIR LOOP OPS a DSC loop became — the two `dyn_cast`s
/// `setBuilderForDataTransfer` tries.
///
/// ⛔ THE THIRD CASE IS THE ONE THIS TYPE REMOVES. The reference tests `affine::AffineForOp` then
/// `scf::ForOp` and returns `LogicalResult::failure()` for anything else, commented *"By
/// construction, it shouldn't happen"* (`SNDSCLowering.cpp:203-204,213-214`). Naming the two forms
/// makes that comment the type: a loop is one or the other, so there is no failure to return and no
/// caller that has to decide what a failed insertion point means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopForm {
    /// `affine.for` — what `constructLoops` builds when the trip count is known.
    AffineFor,
    /// `scf.for` — the dynamic-bound form.
    ScfFor,
}

/// WHETHER A TRANSFER GOES OUTSIDE THE LOOP OR INSIDE ITS BODY — the reference's `is_outer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nesting {
    /// `is_outer == true` — beside the loop op, in its parent block.
    Outer,
    /// `is_outer == false` — inside the loop's body.
    Inner,
}

/// WHICH END — the reference's `is_before`.
///
/// ⛔ TWO ENUMS RATHER THAN TWO `bool`s BECAUSE THE CALL SITE PASSES THEM ADJACENT.
/// `setBuilderForDataTransfer(builder, loop, is_outer, is_before)` transposed is a transfer emitted
/// after the loop instead of inside it, which the compiler cannot see and the IR does not reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// `is_before == true`.
    Before,
    /// `is_before == false`.
    After,
}

/// WHERE THE BUILDER IS PARKED — the four `setInsertionPoint*` calls, as a value.
///
/// ⭐ A RETURNED POSITION RATHER THAN A MUTATED BUILDER, because there is no `OpBuilder` on this side
/// of the bridge: an op list is built in order and this says where the next one belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertionPoint {
    /// `builder.setInsertionPoint(mlir_loop)` — immediately before the loop op.
    BeforeLoop,
    /// `builder.setInsertionPointAfter(mlir_loop)`.
    AfterLoop,
    /// `builder.setInsertionPointToStart(for_op.getBody())`.
    BodyStart,
    /// `builder.setInsertionPoint(for_op.getBody()->getTerminator())` — the last position in the body
    /// that is still before the yield.
    BeforeTerminator,
}

/// Replaces: e027_setBuilderForDataTransfer
///
/// **027/110** `SNDSCLowering::setBuilderForDataTransfer` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:185` (32L).
///
/// ⭐⭐ FOUR ARMS FROM TWO FLAGS, and the loop form only decides HOW the inner two are reached.
/// `affine.for` and `scf.for` answer the same body, so the reference's two `dyn_cast` chains
/// (`:198-214`) differ only in the class they cast to — which is why both arms below give one
/// answer, and why the arm they exist to guard is [`LoopForm`]'s job rather than a return value.
///
/// ⛔ THE INNER-AFTER CASE IS BEFORE THE TERMINATOR, NOT AT THE END OF THE BODY. A body's last op is
/// its `affine.yield`/`scf.yield`; appending after it puts the transfer past the block's terminator,
/// which is a verifier failure rather than a late transfer.
#[must_use]
pub fn insertion_point_for_data_transfer(
    loop_form: LoopForm,
    nesting: Nesting,
    side: Side,
) -> InsertionPoint {
    match (nesting, side) {
        // `if (is_outer && is_before) builder.setInsertionPoint(mlir_loop);`
        (Nesting::Outer, Side::Before) => InsertionPoint::BeforeLoop,
        // `else if (is_outer && !is_before) builder.setInsertionPointAfter(mlir_loop);`
        (Nesting::Outer, Side::After) => InsertionPoint::AfterLoop,
        // `setInsertionPointToStart(affine_for.getBody())` / `(scf_for.getBody())`.
        (Nesting::Inner, Side::Before) => match loop_form {
            LoopForm::AffineFor | LoopForm::ScfFor => InsertionPoint::BodyStart,
        },
        // `setInsertionPoint(affine_for.getBody()->getTerminator())` / the `scf` one.
        (Nesting::Inner, Side::After) => match loop_form {
            LoopForm::AffineFor | LoopForm::ScfFor => InsertionPoint::BeforeTerminator,
        },
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 028/110
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e028_getMLIRLoopFromLoopNode
///
/// **028/110** `SNDSCLowering::getMLIRLoopFromLoopNode` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:220` (14L).
///
/// ⛔⛔ THE TWO LISTS RUN IN OPPOSITE ORDERS, WHICH IS WHY THE INDEX IS MIRRORED.
/// `record->second[size - i - 1]` for `dims_[i]`: the loops are pushed while `constructLoopsRecursive`
/// walks `dim_idx` from `dims_.size() - 1` DOWN to 0 (`SNControlFlowLowering.cpp:895-948`), so
/// element 0 of the record is the OUTERMOST loop — the reference reads it back as `outer_most_loop`
/// at `:893` and `:1041` — while `dims_[0]` is the INNERMOST dimension. Indexing the record with `i`
/// would return the loop of a different dimension, at the same nesting depth, silently.
///
/// ⛔ FIRST MATCH WINS. `dims_` may name a dimension twice (a split band); the reference returns on
/// the first hit, so the innermost of the two is the one a transfer is placed against.
///
/// ⭐ AN ABSENT RECORD IS THE EMPTY SLICE. `find(loop_node) == end()` and "no dim matches" are both
/// `None` in the reference (`nullptr`), so they are one answer here too — and the mirrored index is
/// taken with checked arithmetic, which turns the reference's unchecked `[size - i - 1]` on an empty
/// record into the same `None` rather than a read past the end.
///
/// ⛔⛔ AND IT IS ENTRY 018'S BODY, CHARACTER FOR CHARACTER. `getMLIRLoopFromSNLoopNode`
/// (`SNControlFlowLowering.cpp:49-58`) does the same walk over the same vocabulary against the same
/// mirrored index; the two differ only in which map the caller looked the record up in. This
/// delegates rather than restating it, so the mirror is written down once — a second copy is a second
/// place for the `- i - 1` to drift, and a silently wrong nesting depth is the failure it produces.
#[must_use]
pub fn mlir_loop_from_loop_node(
    dims: &[PrimaryDim],
    mlir_loops: &[Val],
    dim: PrimaryDim,
) -> Option<Val> {
    mlir_loop_from_sn_loop_node(dims, mlir_loops, dim).copied()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 029-032/110 — THE UNIFORMIZED-VALUE SEAM
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE `(core, corelet, fold)` HANDLE OF THE PROGRAM UNIT SET, AND THE UNIT VALUE THAT NAMES IT.
///
/// ⭐ ONE `ProgramUnitOp` SPANS THE SET: the handles are `core_ids_used_ x corelet_ids_used_ x
/// num_folds_`, and `units_involved_` holds one `dataflow.get_unit` result per handle in exactly
/// that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle {
    /// Which core — an element of `core_ids_used_`.
    pub core: Core,
    /// Which corelet — an element of `corelet_ids_used_`.
    pub corelet: Corelet,
    /// Which fold, `0..num_folds_`.
    pub fold: u32,
    /// This handle's entry of `*units_involved_`, the key the mapping is queried by.
    pub unit: Val,
}

/// THE HANDLE SET A UNIFORMIZED VALUE IS DEFINED OVER — `core_ids_used_`, `corelet_ids_used_`,
/// `num_folds_`, `units_involved_` and `program_unit_iterator_`, which the four
/// `constructUniformized*` functions read off `this`.
///
/// ⛔⛔ `DT_CHECK((this->units_involved_)->size() == values.size())` IS THIS TYPE'S CONSTRUCTOR.
/// Every one of the four functions ends with that assertion, because the mapping pairs a unit with a
/// value positionally and a short list would pair the wrong ones. Here the units are BUILT by the
/// same nested walk that the values are, so a value list of a different length cannot be written:
/// each function pushes one pair per [`Handle`] as it emits that handle's constant.
///
/// ⛔ THE WALK ORDER IS CORE-MAJOR, CORELET, THEN FOLD INNERMOST, and it is not free to change:
/// `units_involved_` is filled by the same `for (core) for (corelet) for (fold)` nesting, so the
/// pairing is the order or nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handles {
    /// One group per `(core, corelet)`, each holding that pair's `num_folds_` handles in fold order.
    groups: Vec<Vec<Handle>>,
    /// `program_unit_iterator_` — the key every `uniform.query_map` built here reads.
    iterator: Val,
}

impl Handles {
    /// THE SET, WALKED ONCE — `for (core_ids_used_) for (corelet_ids_used_) for (fold < num_folds_)`,
    /// with `unit` answering `units_involved_` for each handle in turn.
    ///
    /// ⭐ `unit` IS A FUNCTION, NOT A LIST, so the count is arithmetic rather than an assertion: the
    /// caller holds the `dataflow.get_unit` results per `(core, corelet)` and each fold is one of
    /// that op's `num_folds` results.
    pub fn new(
        cores: &[Core],
        corelets: &[Corelet],
        folds: NumFolds,
        iterator: Val,
        mut unit: impl FnMut(Core, Corelet, u32) -> Val,
    ) -> Self {
        let mut groups = Vec::with_capacity(cores.len() * corelets.len());
        for &core in cores {
            for &corelet in corelets {
                groups.push(
                    (0..folds.0)
                        .map(|fold| Handle {
                            core,
                            corelet,
                            fold,
                            unit: unit(core, corelet, fold),
                        })
                        .collect(),
                );
            }
        }
        Self { groups, iterator }
    }

    /// EVERY HANDLE, in the walk order the mapping pairs along.
    pub fn iter(&self) -> impl Iterator<Item = Handle> + '_ {
        self.groups.iter().flatten().copied()
    }

    /// THE HANDLES GROUPED BY `(core, corelet)` — what
    /// [`uniformized_address`] needs, because it emits ONE constant per corelet and names it once per
    /// fold.
    pub fn corelet_groups(&self) -> impl Iterator<Item = &[Handle]> {
        self.groups.iter().map(Vec::as_slice)
    }

    /// THE FIRST HANDLE OF THE WALK, or `None` for an empty set.
    ///
    /// ⚠️ THE REFERENCE READS ITS `first_val` FROM THE ADDRESS MAP'S FIRST KEY, not from
    /// `core_ids_used_.front()` — `addresses.begin()->second.begin()->second`
    /// (`SNDSCLowering.cpp:244`). The two agree when the map holds exactly the handles in use, which
    /// is how its builder fills it; they would differ for a map carrying an unused core below the
    /// first used one.
    #[must_use]
    pub fn first(&self) -> Option<Handle> {
        self.iter().next()
    }

    /// `program_unit_iterator_`.
    #[must_use]
    pub fn iterator(&self) -> Val {
        self.iterator
    }
}

/// `DefImmutableMappingOp::create(..)` THEN `QueryMapOp::create(..)` — the tail all four
/// `constructUniformized*` functions share, and [`super::sync::construct_units_for_uniformization`]
/// with them.
///
/// ⛔ THE QUERY'S RESULT TYPE AND THE MAPPING'S VALUE TYPE ARE THE SAME `ty` AT EVERY CALL SITE, and
/// both are `index` for the three address functions and the bitstream's vector type for entry 032
/// (`:530-537`). The mapping's own RESULT type is `index` in all four — see [`MappedTy`].
pub(super) fn query_over_handles(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    iterator: Val,
    pairs: Vec<(Val, Val)>,
    ty: MappedTy,
) -> Val {
    // `uniform::DefImmutableMappingOp::create(builder, loc, builder.getIndexType(),
    //  *(this->units_involved_), values)`
    let map = vals.mint();
    ops.push(DfirOp::Uniform(uniform::Op::DefImmutableMapping {
        result: map,
        pairs,
        values_ty: ty,
    }));

    // `uniform::QueryMapOp::create(builder, map.getLoc(), <ty>, map.getResult(),
    //  this->program_unit_iterator_)`
    let result = vals.mint();
    ops.push(DfirOp::Uniform(uniform::Op::QueryMap {
        result,
        map,
        key: iterator,
        ty,
    }));
    result
}

/// `mlir::arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), value)`.
///
/// ⭐ `pub` BECAUSE THE SCRATCHY-SIDE ADAPTER BINDS THE PLACEMENT WITH IT. A `LoadAndStore`
/// statement's address is the `AllocNode`'s own start scaled by entry 025's factor, and the scaling
/// happens in the closure the caller supplies — so the caller needs this to bind the result.
pub fn constant_index(vals: &mut Values, ops: &mut Vec<DfirOp>, value: i64) -> Val {
    let result = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::Constant { result, value }));
    result
}

/// Replaces: e029_constructUniformizedAddress
///
/// **029/110** `SNDSCLowering::constructUniformizedAddress` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:236` (47L).
///
/// ⭐⭐ AN ADDRESS THAT IS THE SAME EVERYWHERE IS A CONSTANT, NOT A MAPPING — *"check if all addresses
/// are same. if so, don't create uniform operations"* (`:239`). A `uniform.def_immutable_mapping` per
/// address would make every unit's program textually different, which is the thing uniformization
/// exists to avoid.
///
/// ⛔⛔ ONE CONSTANT PER `(core, corelet)`, NAMED ONCE PER FOLD — the inner
/// `for (fold) values.push_back(const_op.getResult())` pushes the SAME result `num_folds_` times
/// (`:261-267`). Entry 030 emits one constant PER FOLD instead, because its addresses can differ per
/// fold; doing that here would emit `num_folds_` identical `arith.constant`s and leave a mapping
/// whose values are distinct SSA names for one value.
///
/// ⚠️ AN EMPTY HANDLE SET IS THE REFERENCE'S `DT_CHECK(!this->core_ids_used_.empty())` (`:241`).
/// There is nothing to assert here: the walk emits no pairs, so the mapping prints with an empty pair
/// list and dbo-opt names it — a stop at the oracle rather than one before it.
#[must_use]
pub fn uniformized_address(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    handles: &Handles,
    address: impl Fn(Core, Corelet) -> i64,
    factor: Factor,
) -> Val {
    // `int first_val = addresses.begin()->second.begin()->second;` then the `for (core) for (corelet)`
    // that compares every used handle against it — see [`Handles::first`] on which entry that is.
    let first_val = handles
        .first()
        .map(|first| address(first.core, first.corelet));
    let all_same = first_val.is_some_and(|first| {
        handles
            .iter()
            .all(|handle| address(handle.core, handle.corelet) == first)
    });

    // `if (is_all_addresses_same) { auto start_addr = int(first_val * factor); .. }`
    if let (true, Some(first)) = (all_same, first_val) {
        return constant_index(vals, ops, factor.scale(first));
    }

    // `else { for (core) for (corelet) { one ConstantIndexOp, pushed once per fold } }`
    let mut pairs = Vec::new();
    for group in handles.corelet_groups() {
        let Some(first) = group.first() else {
            // `num_folds_` is zero, so this corelet names no handle and holds no constant.
            continue;
        };
        let start_addr = factor.scale(address(first.core, first.corelet));
        let constant = constant_index(vals, ops, start_addr);
        // ⛔ THE SAME RESULT, ONCE PER FOLD. See above.
        pairs.extend(group.iter().map(|handle| (handle.unit, constant)));
    }

    query_over_handles(vals, ops, handles.iterator(), pairs, MappedTy::Index)
}

/// Replaces: e030_constructUniformizedFoldedAddress
///
/// **030/110** `SNDSCLowering::constructUniformizedFoldedAddress` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:285` (51L).
///
/// ⛔⛔ TWO DIFFERENT READS OF ONE `FoldManager`, WHICH IS WHY THERE ARE TWO ARGUMENTS.
/// `addresses.getAllData()` enumerates the WHOLE fold space with its constant dimensions collapsed
/// (`foldInfrastructure.h:2108-2126`) and decides the all-same question; the values are then read
/// per handle with `getAllDataWithMapUnrolled({{0, core}, {1, corelet}})`, which fixes the core and
/// corelet dimensions and unrolls the mapped ones (`:2128-2146`). Deciding the question over the
/// per-handle walk instead would answer it over a SUBSET — the used cores — and emit a single
/// constant for a fold space that is not constant outside them.
///
/// ⛔ ONE CONSTANT PER FOLD, unlike entry 029: `foldedAddresses[fold_id]` when the manager holds one
/// value per fold and `foldedAddresses[0]` when it holds one for all of them
/// (`:307-313`) — the `DT_CHECK_MSG(is_any_of(size, 1, num_folds_))` at `:303` says those are the
/// only two shapes. `address` answers per handle, so both shapes reach it as the same total
/// function and the check has nothing left to assert.
///
/// ⭐ `getAllData()` IS NEVER EMPTY: a zero-dimensional fold space still pushes `getData()`
/// (`foldInfrastructure.h:2110-2114`), so the reference's `std::equal(begin + 1, end, begin)` and
/// `allAddresses.at(0)` are safe. An empty slice takes the mapping branch here rather than reading
/// past the end.
#[must_use]
pub fn uniformized_folded_address(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    handles: &Handles,
    all_addresses: &[i64],
    address: impl Fn(Core, Corelet, u32) -> i64,
    factor: Factor,
) -> Val {
    // `const bool is_all_addresses_same = std::equal(allAddresses.begin() + 1, allAddresses.end(),
    //  allAddresses.begin());` and, when it holds, `int(allAddresses.at(0) * factor)`.
    if let Some((first, rest)) = all_addresses.split_first() {
        if rest.iter().all(|address| address == first) {
            return constant_index(vals, ops, factor.scale(*first));
        }
    }

    // `else { for (core) for (corelet) for (fold) { a ConstantIndexOp per fold } }`
    let mut pairs = Vec::new();
    for handle in handles.iter() {
        let start_addr = factor.scale(address(handle.core, handle.corelet, handle.fold));
        let constant = constant_index(vals, ops, start_addr);
        pairs.push((handle.unit, constant));
    }

    query_over_handles(vals, ops, handles.iterator(), pairs, MappedTy::Index)
}

/// Replaces: e031_constructUniformizedFoldedDoubleBufferToggling
///
/// **031/110** `SNDSCLowering::constructUniformizedFoldedDoubleBufferToggling` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:339` (83L).
///
/// ⭐⭐ THE TOGGLE IS `buffer + 2 * start`, AND THE 2 IS THE DOUBLE BUFFER. A folded transfer
/// alternates between two halves, so the fold's start address is added at twice its value — the
/// offset walks a stride of two buffers per fold (`:369-371`).
///
/// ⛔⛔ `first_val` IS READ AT FOLD COORDINATES `(0, 0, 0)`, NOT AT THE FIRST USED HANDLE.
/// `start_addresses.getSingleData()` with no fixed coordinates zeroes every dimension
/// (`foldInfrastructure.h:1934-1940`), and dimensions 0 and 1 of this manager are the core and the
/// corelet — so the value the comparison is made against belongs to core 0, corelet 0, fold 0, which
/// need not be a handle this program unit uses. `start_at_origin` is that read, kept as its own
/// argument so the asymmetry is visible instead of being quietly repaired.
///
/// ⛔ THE PREDICATE COMPARES SCALED OFFSETS, NOT RAW ADDRESSES. Both sides go through
/// `int(x * factor)` before the comparison (`:349,369-371`), so two addresses that differ by less
/// than one address step are the SAME value here and take the single-constant branch. Comparing
/// before scaling would emit a mapping for a program that has one address.
///
/// ⚠️ THE WALK IS PERFORMED TWICE IN THE REFERENCE — once to decide (`:351-373`) and once to emit
/// (`:381-400`) — with the same body. It is one walk here, and the values are dropped when the
/// predicate holds; the ops are pushed only in the branch that keeps them.
#[must_use]
pub fn uniformized_folded_double_buffer_toggling(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    handles: &Handles,
    buffer_address: impl Fn(Core, Corelet) -> i64,
    start_address: impl Fn(Core, Corelet, u32) -> i64,
    start_at_origin: i64,
    factor: Factor,
) -> Val {
    // `int first_val = buffer_addresses.begin()->second.begin()->second;`
    // `first_val += 2 * start_addresses.getSingleData(); first_val = int(first_val * factor);`
    let first_val = handles
        .first()
        .map(|first| factor.scale(buffer_address(first.core, first.corelet) + 2 * start_at_origin));

    // `int buff_addr_offset = int((buffer_addresses.at(core).at(corelet) + 2 * foldedAddr) * factor);`
    let offsets: Vec<i64> = handles
        .iter()
        .map(|handle| {
            let folded = start_address(handle.core, handle.corelet, handle.fold);
            factor.scale(buffer_address(handle.core, handle.corelet) + 2 * folded)
        })
        .collect();

    // `if (buff_addr_offset != first_val) is_all_addresses_same = false;` over the whole walk.
    if let Some(first) = first_val {
        if offsets.iter().all(|offset| *offset == first) {
            return constant_index(vals, ops, first);
        }
    }

    // The second walk — one `ConstantIndexOp` per handle, paired with that handle's unit.
    let pairs = handles
        .iter()
        .zip(offsets)
        .map(|(handle, offset)| (handle.unit, constant_index(vals, ops, offset)))
        .collect();

    query_over_handles(vals, ops, handles.iterator(), pairs, MappedTy::Index)
}

/// WHAT A `vectorchain.constant_bitstream`'S ELEMENTS MEAN — the reference's `is_symbolic` flag.
///
/// ⛔ IT DECIDES THE PRINTED FORM, NOT JUST AN ATTRIBUTE. See
/// [`vectorchain::Op::ConstantBitstream::is_symbol`]: a symbolic bitstream prints its generic
/// attribute dictionary with decimal `i64` values, a literal one prints hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitstreamValues {
    /// `is_symbolic == false` — the elements are bit patterns.
    BitPatterns,
    /// `is_symbolic == true` — the elements are symbol ids a later pass resolves.
    Symbols,
}

/// Replaces: e032_constructUniformizedFoldedConstantBitStream
///
/// **032/110** `SNDSCLowering::constructUniformizedFoldedConstantBitStream` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:462` (75L).
///
/// ⛔⛔ THE GATE IS THE FOLD SPACE'S SHAPE, NOT ITS VALUES. `if (all_fold_info.size() == 1)`
/// (`:1026`) asks whether the manager holds ONE datum at all — the other three functions ask whether
/// the data are all EQUAL. A fold space of four identical bitstreams takes the mapping branch here
/// and the single-constant branch there, so reading this as "all the same" would emit one
/// `vectorchain.constant_bitstream` where the reference emits four and a query.
///
/// ⛔⛔ AND THIS IS THE FUNCTION THE ISLAND HAD TO GROW FOR. Its mapping's values are VECTORS while
/// the mapping's own result type is `index`, and its query's result is the bitstream's vector type
/// (`:530-537`) — `DefImmutableMappingOp::print` appends `, vector<..>` exactly then
/// (`Uniform.cpp:468-472`) and `QueryMapOp::print` prints it (`:591`). Emitting `: index` for a
/// vector-producing query is a parse error in the consumer, not a wrong number.
///
/// ⭐ `is_symbolic` IS SET ON EVERY BITSTREAM IN BOTH BRANCHES (`:479-481` and `:519-521`), so it is
/// a property of the value being built rather than of the branch.
#[must_use]
pub fn uniformized_folded_constant_bitstream(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    handles: &Handles,
    all_fold_info: &[Vec<i64>],
    bitstream: impl Fn(Core, Corelet, u32) -> Vec<i64>,
    bitstream_ty: Vector,
    values: BitstreamValues,
) -> Val {
    let is_symbol = matches!(values, BitstreamValues::Symbols);

    // `if (all_fold_info.size() == 1) { .. ConstantBitstreamOp::create(.., all_fold_info.front()) }`
    if let [single] = all_fold_info {
        let result = vals.mint();
        ops.push(DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
            result,
            value: single.clone(),
            ty: bitstream_ty,
            is_symbol,
        }));
        return result;
    }

    // `for (core) for (corelet) for (fold) { one ConstantBitstreamOp per fold }`
    let mut pairs = Vec::new();
    for handle in handles.iter() {
        let result = vals.mint();
        ops.push(DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
            result,
            value: bitstream(handle.core, handle.corelet, handle.fold),
            ty: bitstream_ty,
            is_symbol,
        }));
        pairs.push((handle.unit, result));
    }

    // ⛔ THE VECTOR TYPE, on both the mapping's values and the query's result. See above.
    query_over_handles(
        vals,
        ops,
        handles.iterator(),
        pairs,
        MappedTy::Vector(bitstream_ty),
    )
}

/// Replaces: e059_constructUniformizedFoldedDestinationCore
///
/// THE DESTINATION CORE'S HANDLE, PER PROGRAM-UNIT HANDLE — a `dataflow.get_unit` for `unit` in the
/// core this handle's fold data names, mapped from the asking handle and queried once
/// (`SNDSCLowering.cpp:425-460`). ⭐ NO ALL-SAME SHORTCUT: unlike entries 029-031 it always maps.
///
/// ⛔ THE ITERATOR IS `uniform_region_iterator_` WHERE THERE IS ONE, `program_unit_iterator_`
/// OTHERWISE (`:456-457`) — the only one of these tails that chooses.
///
/// ⚠️ ONE PAIR PER FOLD IN THE REFERENCE, ONE HANDLE PER FOLD HERE: it pairs
/// `unit_def_op->getResult(fold_id)` with `dst_unit_op.getResult(fold_id)`, and this island binds ONE
/// result per `get_unit` — so a corelet's folds all name the same destination handle.
#[must_use]
pub fn uniformized_folded_destination_core(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    handles: &Handles,
    unit: DfirUnit,
    num_folds: NumFolds,
    region_iterator: Option<Val>,
    dst_core: impl Fn(Core, Corelet) -> Core,
) -> Val {
    let mut pairs = Vec::new();
    for group in handles.corelet_groups() {
        let Some(first) = group.first() else {
            // `num_folds_` is zero, so this corelet names no handle and asks for no destination.
            continue;
        };
        // `createGetUnitOpInDifferentCore(builder, comp_, dst_core_id, corelet_id, num_folds_)`,
        // where `dst_core_id` is `getSingleDataStrict(addresses, {{0, core_id}, {1, corelet_id}})`.
        let dst_op = create_get_unit_op_in_different_core(
            vals,
            unit,
            dst_core(first.core, first.corelet),
            first.corelet,
            num_folds,
        );
        let dataflow::Op::GetUnit { result: dst, .. } = &dst_op else {
            // Entry 023 builds nothing else, so this corelet has no destination to name.
            continue;
        };
        let dst = *dst;
        ops.push(DfirOp::Dataflow(dst_op));
        pairs.extend(group.iter().map(|handle| (handle.unit, dst)));
    }

    query_over_handles(
        vals,
        ops,
        region_iterator.unwrap_or_else(|| handles.iterator()),
        pairs,
        MappedTy::Index,
    )
}

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
            Some(cl1),
        );
        assert_eq!(l3, Retrieved::Reused(Val(7)));

        // The LX load unit is per corelet: corelet 0 reuses, corelet 1 gets its own op.
        let same = retrieve_get_unit_op_in_same_core(
            &mut vals,
            &handlers,
            Component::Unit(DfirUnit::Lxlu),
            core,
            Some(cl0),
        );
        assert_eq!(same, Retrieved::Reused(Val(8)));
        let other = retrieve_get_unit_op_in_same_core(
            &mut vals,
            &handlers,
            Component::Unit(DfirUnit::Lxlu),
            core,
            Some(cl1),
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
            let got = retrieve_get_unit_op_in_same_core(&mut vals, &handlers, comp, core, Some(cl1));
            assert_eq!(got, Retrieved::Reused(Val(1)));
        }
        let xrf =
            retrieve_get_unit_op_in_same_core(&mut vals, &handlers, Component::PtXrf, core, Some(cl1));
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

    /// A core of this arch.
    fn core(index: u32) -> Core {
        Core::checked(index).expect("the arch has more than two cores")
    }

    /// A corelet of this arch.
    fn corelet(index: u32) -> Corelet {
        Corelet::checked(index).expect("the arch has more than one corelet")
    }

    /// `cores x {corelet 0} x folds`, with each handle's unit named `%(1000 + core * 100 + fold)` so
    /// that an expected pair list reads as which handle it belongs to.
    fn handles(cores: u32, folds: u32) -> Handles {
        let cores: Vec<Core> = (0..cores).map(core).collect();
        Handles::new(
            &cores,
            &[corelet(0)],
            NumFolds(folds),
            Val(9),
            |core, _corelet, fold| Val(1000 + core.get() * 100 + fold),
        )
    }

    /// `vector<64xf16>` — the bitstream type entry 032's query is typed by.
    const BITSTREAM: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };

    /// 🎯 025/110 — ⭐⭐ THE FACTOR IS THE TABLE'S BYTES IN BITS OVER THE ELEMENT'S BITS, AND FOR
    /// `SENINT24` THAT IS 16.
    ///
    /// ⛔⛔ CARRIED AS VALUES, ONE PER SHAPE OF ROW. A relation — "the L3 rows agree with each
    /// other" — passes on a table where all of them are wrong together, and the `SENINT24` row is
    /// precisely the one a reader supplies from the name: `128 * 8 / 24` is 42.67, and every address
    /// scaled by it lands somewhere else. The reference reaches 16 through a special case on the MLIR
    /// type's width; this reaches it through the packing table, and the assertion is that the two
    /// answers are the same one.
    ///
    /// ⛔ AND ONE ROW IS FRACTIONAL. `{LXLU, LX}` against a 16-bit element is `1 * 8 / 16 = 0.5`;
    /// rounding the factor to an integer would either halve or double every LX address.
    #[test]
    fn the_granularity_factor_is_bits_per_step_over_bits_per_element() {
        // `{L3LU, HBM}` = 128 bytes, 16-bit element: `128 * 8 / 16`.
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L3luHbm, DataType::Sen169Fp16),
            Factor(64.0)
        );
        // ⛔ THE SAME 64 FOR `SENINT24`, NOT `128 * 8 / 24`.
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L3luHbm, DataType::Senint24),
            Factor(64.0)
        );
        // An 8-bit element halves it, a 4-bit one halves it again — the divisor is the ELEMENT.
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L3luHbm, DataType::Senint8),
            Factor(128.0)
        );
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L3luHbm, DataType::Senint4),
            Factor(256.0)
        );
        // ⛔ FRACTIONAL: `{LXLU, LX}` is one byte per step.
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::LxluLx, DataType::Sen169Fp16),
            Factor(0.5)
        );
        // `{L3LU, L3LUIBR}` = 4 bytes.
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L3luIbr, DataType::Sen169Fp16),
            Factor(2.0)
        );
        // ⛔⛔ THE L0 STORE UNIT IS `numPTRows` WHERE THE LOAD UNIT IS 1 — the arch's row count, not
        // a literal, so this reads the same on both builds.
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L0luL0, DataType::Sen169Fp16),
            Factor(0.5)
        );
        assert_eq!(
            address_granularity_multiply_factor(DataLocation::L0suL0, DataType::Sen169Fp16),
            Factor(f64::from(Target::PT_ROWS) / 2.0)
        );
        assert_ne!(
            address_granularity_multiply_factor(DataLocation::L0suL0, DataType::Sen169Fp16),
            address_granularity_multiply_factor(DataLocation::L0luL0, DataType::Sen169Fp16),
            "a store into the L0 lands one stick per PT row (sysdef.cpp:542)"
        );

        // ⭐ AND THE SCALING TRUNCATES TOWARD ZERO, which is what `int(address * factor)` does.
        assert_eq!(Factor(0.5).scale(7), 3);
        assert_eq!(Factor(0.5).scale(-7), -3);
        assert_eq!(Factor(64.0).scale(3), 192);
    }

    /// 🎯 026/110 — THE REPORT NAMES THE PASS AND THE NODE, AND DOES NOT RETURN.
    ///
    /// ⛔ THE NODE NAME IS THE HALF THAT MAKES IT ACTIONABLE. `DT_ERROR("... for the node: " +
    /// dsc_->name_)` is the second sentence; without it the report says only that some node of some
    /// model failed to translate.
    #[test]
    #[should_panic(expected = "[DSC2.0 to Dataflow IR]: no handler for the component")]
    fn an_error_carries_the_pass_prefix() {
        emit_error("matmul_qkv_0", "no handler for the component");
    }

    /// 🎯 026/110 — AND THE SECOND SENTENCE CARRIES THE NODE.
    #[test]
    #[should_panic(expected = "translation for the node: matmul_qkv_0")]
    fn an_error_carries_the_node_it_failed_on() {
        emit_error("matmul_qkv_0", "no handler for the component");
    }

    /// 🎯 027/110 — ⭐⭐ FOUR POSITIONS FROM TWO FLAGS, AND THE INNER PAIR IS THE BODY.
    ///
    /// ⛔ THE TWO FLAGS ARE ADJACENT AT THE CALL SITE, so the test that matters is that the four
    /// combinations give four DIFFERENT answers — a port that transposed them would still return a
    /// legal insertion point for every input.
    ///
    /// ⛔ AND THE LOOP FORM DOES NOT CHANGE THE ANSWER. The reference's two `dyn_cast` chains differ
    /// only in the class they cast to; the third arm they guard is the one [`LoopForm`] removes.
    #[test]
    fn the_four_insertion_points_are_four_different_positions() {
        for form in [LoopForm::AffineFor, LoopForm::ScfFor] {
            assert_eq!(
                insertion_point_for_data_transfer(form, Nesting::Outer, Side::Before),
                InsertionPoint::BeforeLoop
            );
            assert_eq!(
                insertion_point_for_data_transfer(form, Nesting::Outer, Side::After),
                InsertionPoint::AfterLoop
            );
            assert_eq!(
                insertion_point_for_data_transfer(form, Nesting::Inner, Side::Before),
                InsertionPoint::BodyStart
            );
            // ⛔ BEFORE THE TERMINATOR, not after the body's last op — the last op IS the yield.
            assert_eq!(
                insertion_point_for_data_transfer(form, Nesting::Inner, Side::After),
                InsertionPoint::BeforeTerminator
            );
        }
    }

    /// 🎯 028/110 — ⭐⭐ THE RECORD IS INDEXED FROM THE OTHER END: `dims_[0]` IS THE INNERMOST
    /// DIMENSION AND `loops[0]` IS THE OUTERMOST LOOP.
    ///
    /// ⛔⛔ THE MIRROR IS THE WHOLE FUNCTION. `record->second[size - i - 1]` against `dims_[i]`
    /// (`SNDSCLowering.cpp:227-228`), and the reference reads element 0 back as `outer_most_loop`
    /// (`SNControlFlowLowering.cpp:893`) while `constructLoopsRecursive` pushes the loops walking
    /// `dims_` backwards (`:895-948`). Indexing with `i` returns a loop at the wrong nesting depth for
    /// the RIGHT dimension name — an address emitted against the wrong induction variable, which
    /// nothing downstream can detect.
    #[test]
    fn a_dimension_maps_to_the_loop_at_its_mirrored_depth() {
        // `dims_` innermost-first, so `Out` is the outermost band; `loops` outermost-first.
        let dims = [PrimaryDim::Ij, PrimaryDim::Mb, PrimaryDim::Out];
        let loops = [Val(10), Val(11), Val(12)];

        // ⛔ THE INNERMOST DIMENSION TAKES THE LAST LOOP, NOT THE FIRST.
        assert_eq!(
            mlir_loop_from_loop_node(&dims, &loops, PrimaryDim::Ij),
            Some(Val(12))
        );
        assert_eq!(
            mlir_loop_from_loop_node(&dims, &loops, PrimaryDim::Mb),
            Some(Val(11))
        );
        assert_eq!(
            mlir_loop_from_loop_node(&dims, &loops, PrimaryDim::Out),
            Some(Val(10))
        );

        // A dimension the node does not carry, and an absent record — both `nullptr`.
        assert_eq!(
            mlir_loop_from_loop_node(&dims, &loops, PrimaryDim::Kij),
            None
        );
        assert_eq!(mlir_loop_from_loop_node(&dims, &[], PrimaryDim::Ij), None);
    }

    /// 🎯 029/110 — ⭐⭐ ONE CONSTANT PER `(core, corelet)`, NAMED ONCE PER FOLD.
    ///
    /// ⛔⛔ THE REPEATED SSA NAME IS THE POINT. The inner `for (fold) values.push_back(const_op)`
    /// pushes the SAME result `num_folds_` times (`:261-267`), so a two-core two-fold set has FOUR
    /// pairs over TWO constants. Entry 030 emits four constants for the same set — and this is the
    /// function whose addresses cannot differ per fold, so a fourth `arith.constant` here would be a
    /// second name for a value that already has one.
    #[test]
    fn a_uniformized_address_names_one_constant_per_corelet_once_per_fold() {
        let handles = handles(2, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_address(
            &mut vals,
            &mut ops,
            &handles,
            |core, _corelet| 100 + i64::from(core.get()),
            Factor(2.0),
        );

        assert_eq!(result, Val(3));
        assert_eq!(
            ops,
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(0),
                    value: 200,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(1),
                    value: 202,
                }),
                DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(2),
                    // ⛔ FOUR PAIRS, TWO CONSTANTS — each named once per fold.
                    pairs: vec![
                        (Val(1000), Val(0)),
                        (Val(1001), Val(0)),
                        (Val(1100), Val(1)),
                        (Val(1101), Val(1)),
                    ],
                    values_ty: MappedTy::Index,
                }),
                DfirOp::Uniform(uniform::Op::QueryMap {
                    result: Val(3),
                    map: Val(2),
                    key: Val(9),
                    ty: MappedTy::Index,
                }),
            ]
        );
    }

    /// 🎯 029/110 — AND AN ADDRESS THAT IS THE SAME EVERYWHERE IS A CONSTANT, NOT A MAPPING.
    ///
    /// ⛔ *"check if all addresses are same. if so, don't create uniform operations"* (`:239`). A
    /// mapping per address makes every unit's program textually different, which is the thing
    /// uniformization exists to avoid — and the compared value is the SCALED one.
    #[test]
    fn a_uniform_address_emits_one_constant_and_no_mapping() {
        let handles = handles(2, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_address(&mut vals, &mut ops, &handles, |_, _| 40, Factor(0.5));

        assert_eq!(result, Val(0));
        assert_eq!(
            ops,
            vec![DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 20,
            })],
            "one `arith.constant` and neither uniform op"
        );
    }

    /// 🎯 030/110 — ⭐⭐ ONE CONSTANT PER FOLD, AND THE ALL-SAME QUESTION IS ASKED OF THE WHOLE FOLD
    /// SPACE.
    ///
    /// ⛔⛔ TWO DIFFERENT READS OF ONE MANAGER (`:288,297-302`). `getAllData()` covers the whole space;
    /// the values come from `getAllDataWithMapUnrolled` per used `(core, corelet)`. Here the used
    /// handles all hold 5 while the space also holds a 9, so the reference takes the MAPPING branch
    /// and emits two identical constants — deciding the question over the used handles instead would
    /// collapse it to one constant and specialise nothing.
    #[test]
    fn a_folded_address_decides_over_the_whole_fold_space() {
        let handles = handles(1, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_folded_address(
            &mut vals,
            &mut ops,
            &handles,
            // `getAllData()` — the whole space, which is NOT constant.
            &[5, 9],
            // Every USED handle holds 5.
            |_, _, _| 5,
            Factor(1.0),
        );

        assert_eq!(result, Val(3));
        assert_eq!(
            ops,
            vec![
                // ⛔ TWO CONSTANTS FOR ONE VALUE — one per fold, as the reference emits them.
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(0),
                    value: 5,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(1),
                    value: 5,
                }),
                DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(2),
                    pairs: vec![(Val(1000), Val(0)), (Val(1001), Val(1))],
                    values_ty: MappedTy::Index,
                }),
                DfirOp::Uniform(uniform::Op::QueryMap {
                    result: Val(3),
                    map: Val(2),
                    key: Val(9),
                    ty: MappedTy::Index,
                }),
            ]
        );
    }

    /// 🎯 030/110 — AND A CONSTANT FOLD SPACE TAKES ITS VALUE FROM `getAllData().at(0)`.
    ///
    /// ⛔ FROM THE SPACE, NOT FROM A HANDLE (`:296`). The two agree for a space that is constant,
    /// which is the branch's own precondition — so this pins WHICH read the single constant comes
    /// from, and the per-handle function is not consulted at all.
    #[test]
    fn a_constant_fold_space_emits_one_scaled_constant() {
        let handles = handles(2, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_folded_address(
            &mut vals,
            &mut ops,
            &handles,
            &[7, 7, 7, 7],
            |_, _, _| panic!("the per-handle read is not reached when the space is constant"),
            Factor(2.0),
        );

        assert_eq!(result, Val(0));
        assert_eq!(
            ops,
            vec![DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 14,
            })]
        );
    }

    /// 🎯 031/110 — ⭐⭐ THE TOGGLE IS `buffer + 2 * start`, AND `first_val` COMES FROM FOLD
    /// COORDINATES `(0, 0, 0)`.
    ///
    /// ⛔⛔ THE ASYMMETRY IS REAL AND IT DECIDES THE BRANCH. `start_addresses.getSingleData()` zeroes
    /// every dimension (`foldInfrastructure.h:1934-1940`), so `first_val` is built from the ORIGIN's
    /// start address while every comparison is made against a USED handle's. Here every used handle
    /// agrees with every other — 100 + 2*4 = 108 throughout — and the reference still takes the
    /// mapping branch, because the origin holds 0 and `first_val` is 100. Repairing that to "the first
    /// used handle" would emit a single constant for a program the reference uniformizes.
    #[test]
    fn the_double_buffer_toggle_compares_against_the_fold_origin() {
        let handles = handles(1, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_folded_double_buffer_toggling(
            &mut vals,
            &mut ops,
            &handles,
            |_, _| 100,
            |_, _, _| 4,
            // `getSingleData()` at `(0, 0, 0)` — a coordinate no used handle need occupy.
            0,
            Factor(1.0),
        );

        assert_eq!(result, Val(3));
        assert_eq!(
            ops,
            vec![
                // ⭐ `100 + 2 * 4` — the 2 is the double buffer.
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(0),
                    value: 108,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(1),
                    value: 108,
                }),
                DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(2),
                    pairs: vec![(Val(1000), Val(0)), (Val(1001), Val(1))],
                    values_ty: MappedTy::Index,
                }),
                DfirOp::Uniform(uniform::Op::QueryMap {
                    result: Val(3),
                    map: Val(2),
                    key: Val(9),
                    ty: MappedTy::Index,
                }),
            ]
        );
    }

    /// 🎯 031/110 — AND WHEN THE ORIGIN AGREES, IT IS ONE CONSTANT.
    ///
    /// ⛔ THE COMPARISON IS OF SCALED OFFSETS (`:349,369-371`), so a factor of 0.5 makes `108` and
    /// `109` the same value — two raw addresses inside one address step are one `index`, and comparing
    /// before scaling would uniformize a program that has a single address.
    #[test]
    fn a_toggle_that_agrees_with_the_origin_is_one_constant() {
        let handles = handles(1, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_folded_double_buffer_toggling(
            &mut vals,
            &mut ops,
            &handles,
            |_, _| 100,
            // fold 0 holds 4, fold 1 holds 4 — and `2 * 4` scaled by 0.5 is 4.
            |_, _, _| 4,
            4,
            Factor(0.5),
        );

        assert_eq!(result, Val(0));
        assert_eq!(
            ops,
            vec![DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                // `int((100 + 8) * 0.5)`
                value: 54,
            })]
        );
    }

    /// 🎯 032/110 — ⭐⭐ THE GATE IS `size() == 1`: A FOLD SPACE OF IDENTICAL BITSTREAMS STILL GETS A
    /// MAPPING.
    ///
    /// ⛔⛔ THE OTHER THREE FUNCTIONS ASK WHETHER THE DATA ARE EQUAL; THIS ONE ASKS WHETHER THERE IS
    /// ONE DATUM (`:1026`). Reading it as "all the same" would emit a single
    /// `vectorchain.constant_bitstream` where the reference emits one per fold and a query.
    ///
    /// ⛔ AND THE MAPPING'S VALUES ARE VECTORS while its result stays `index` — the query is typed
    /// `vector<64xf16>` (`:530-537`), which is why [`MappedTy`] exists.
    #[test]
    fn a_bitstream_fold_space_of_identical_values_still_gets_a_mapping() {
        let handles = handles(1, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_folded_constant_bitstream(
            &mut vals,
            &mut ops,
            &handles,
            // TWO entries, equal — `size() == 1` is false.
            &[vec![1, 2], vec![1, 2]],
            |_, _, _| vec![1, 2],
            BITSTREAM,
            BitstreamValues::Symbols,
        );

        assert_eq!(result, Val(3));
        assert_eq!(
            ops,
            vec![
                DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
                    result: Val(0),
                    value: vec![1, 2],
                    ty: BITSTREAM,
                    // ⭐ SET ON EVERY BITSTREAM IN BOTH BRANCHES (`:479-481`, `:519-521`).
                    is_symbol: true,
                }),
                DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
                    result: Val(1),
                    value: vec![1, 2],
                    ty: BITSTREAM,
                    is_symbol: true,
                }),
                DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(2),
                    pairs: vec![(Val(1000), Val(0)), (Val(1001), Val(1))],
                    // ⛔ THE VECTOR TYPE, on the values.
                    values_ty: MappedTy::Vector(BITSTREAM),
                }),
                DfirOp::Uniform(uniform::Op::QueryMap {
                    result: Val(3),
                    map: Val(2),
                    key: Val(9),
                    // ⛔ AND ON THE QUERY'S RESULT.
                    ty: MappedTy::Vector(BITSTREAM),
                }),
            ]
        );
    }

    /// 🎯 032/110 — AND ONE DATUM IS ONE BITSTREAM, WITH NO `is_symbol` WHEN THE VALUES ARE BITS.
    #[test]
    fn a_single_datum_bitstream_is_one_op() {
        let handles = handles(1, 2);
        let mut vals = Values::default();
        let mut ops = Vec::new();

        let result = uniformized_folded_constant_bitstream(
            &mut vals,
            &mut ops,
            &handles,
            &[vec![0xffff, 0]],
            |_, _, _| panic!("the per-handle read is not reached for a single datum"),
            BITSTREAM,
            BitstreamValues::BitPatterns,
        );

        assert_eq!(result, Val(0));
        assert_eq!(
            ops,
            vec![DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
                result: Val(0),
                value: vec![0xffff, 0],
                ty: BITSTREAM,
                is_symbol: false,
            })]
        );
    }

    /// ⭐⭐ AND THE WALK IS WHAT THE `DT_CHECK` ASSERTS: ONE HANDLE PER `(core, corelet, fold)`, IN
    /// THAT NESTING, EACH CARRYING ITS OWN UNIT.
    ///
    /// ⛔ `DT_CHECK((this->units_involved_)->size() == values.size())` ends all four functions. It has
    /// no counterpart because the units are built BY the walk the values are: this pins the order the
    /// pairing depends on — core-major, corelet, fold innermost — so a change to it fails here rather
    /// than pairing a unit with another unit's address.
    #[test]
    fn the_handle_walk_is_core_major_and_fold_innermost() {
        let handles = handles(2, 2);

        assert_eq!(
            handles.iter().collect::<Vec<Handle>>(),
            vec![
                Handle {
                    core: core(0),
                    corelet: corelet(0),
                    fold: 0,
                    unit: Val(1000),
                },
                Handle {
                    core: core(0),
                    corelet: corelet(0),
                    fold: 1,
                    unit: Val(1001),
                },
                Handle {
                    core: core(1),
                    corelet: corelet(0),
                    fold: 0,
                    unit: Val(1100),
                },
                Handle {
                    core: core(1),
                    corelet: corelet(0),
                    fold: 1,
                    unit: Val(1101),
                },
            ]
        );

        // ⭐ AND THE GROUPS ARE THE `(core, corelet)` PAIRS entry 029 emits one constant for.
        assert_eq!(handles.corelet_groups().count(), 2);
        assert!(
            handles
                .corelet_groups()
                .all(|group| group.len() == 2 && group[0].fold == 0 && group[1].fold == 1)
        );
        assert_eq!(handles.first().map(|first| first.unit), Some(Val(1000)));
        assert_eq!(handles.iterator(), Val(9));
    }

    /// 🎯 059/110 — ⛔ THE DESTINATION HANDLE IS PER `(core, corelet)` AND NAMED ONCE PER FOLD, and
    /// the query reads the REGION iterator where the caller has one.
    ///
    /// A per-fold destination op would emit `num_folds_` identical `get_unit`s for one remote unit,
    /// and querying `program_unit_iterator_` inside a uniform region asks the wrong index.
    #[test]
    fn every_fold_of_a_corelet_names_one_destination_and_the_region_iterator_wins() {
        let handles = handles(2, 2);
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();

        // The fold data sends each core's traffic to the other one.
        let result = uniformized_folded_destination_core(
            &mut vals,
            &mut ops,
            &handles,
            DfirUnit::L3lu,
            NumFolds(2),
            Some(Val(77)),
            |from, _corelet| core(1 - from.get()),
        );

        assert_eq!(
            ops,
            vec![
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(0),
                    residency: Residency::Corelet {
                        core: core(1),
                        corelet: corelet(0),
                    },
                    unit: DfirUnit::L3lu,
                    num_folds: Some(NumFolds(2)),
                }),
                DfirOp::Dataflow(dataflow::Op::GetUnit {
                    result: Val(1),
                    residency: Residency::Corelet {
                        core: core(0),
                        corelet: corelet(0),
                    },
                    unit: DfirUnit::L3lu,
                    num_folds: Some(NumFolds(2)),
                }),
                DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                    result: Val(2),
                    pairs: vec![
                        (Val(1000), Val(0)),
                        (Val(1001), Val(0)),
                        (Val(1100), Val(1)),
                        (Val(1101), Val(1)),
                    ],
                    values_ty: MappedTy::Index,
                }),
                DfirOp::Uniform(uniform::Op::QueryMap {
                    result: Val(3),
                    map: Val(2),
                    key: Val(77),
                    ty: MappedTy::Index,
                }),
            ]
        );
        assert_eq!(result, Val(3));
    }
}
