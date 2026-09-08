//! THE SHARED SHAPE AND FOLD HELPERS every lowering below stands on.
//!
//! 11 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e006_getTranslatorVersion` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:22` |
//! | `e007_createGetLocalUnitOp` | 0 | 13 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:44` |
//! | `e008_setPrecisionInUnitOp` | 0 | 12 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:143` |
//! | `e009_constructAVectorOfIndexType` | 0 | 5 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:711` |
//! | `e010_emitError` | 0 | 4 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:718` |
//! | `e044_createGetUnitOp` | 1 | 24 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:64` |
//! | `e072_createUniformizedGetUnitOp` | 2 | 42 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:95` |
//! | `e073_buildNeighborUnits` | 2 | 213 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:161` |
//! | `e083_buildUniformizedNeighborUnits` | 3 | 239 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:380` |
//! | `e084_initializeUnit` | 3 | 33 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:624` |
//! | `e093_initializeUniformizedUnit` | 4 | 47 | `dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:663` |

use super::dsc_lowering::Retrieved;
use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{Val, dataflow};
use crate::islands::dataflow_ir::ty::{GenericComp, ScalarTy};
use crate::islands::dataflow_ir::{ProgramUnit, Values};
use crate::units::{DfirUnit, NumFolds, Residency};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// WHAT ONE DSC OF A SUPERDSC IS, as far as the version choice is concerned — `computeOp_.empty()`
/// and `isDSC2()` (`DSC2ToDataflowIRUtils.hpp:26,30`) as the three states they enumerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DscKind {
    /// `dsc.computeOp_.empty()` — a DSC with no compute op at all.
    NoComputeOp,
    /// `!dsc.isDSC2()` — a DSC1.0 schedule.
    Dsc1,
    /// A DSC2.0 schedule with a compute op.
    Dsc2,
}

/// WHICH TRANSLATOR A SUPERDSC NEEDS — the `int &version` out-parameter and the `LogicalResult`
/// read as ONE answer.
///
/// ⛔⛔ THE TWO V1 ARMS ARE INDISTINGUISHABLE AT THE ONLY CALLER, AND THEY ARE STILL BOTH HERE.
/// `runTranslator` returns `failure()` both on `failed(getTranslatorVersion(..))`
/// (`DSC2ToDataflowIR.cpp:546-548`) and from its own trailing `else` for any version but 3
/// (`:558-560`) — and both arms set `version = 1`. So the `success()`/`failure()` distinction reaches
/// nothing; collapsing it would still be dropping a value the reference computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslatorVersion {
    /// `version = 3`, `success()` — every DSC is a DSC2.0 with a compute op.
    V3,
    /// `version = 1`, `success()` — a DSC1.0 was found.
    V1,
    /// `version = 1`, `failure()` — a DSC with no compute op was found first.
    V1NoComputeOp,
}

/// Replaces: e006_getTranslatorVersion
///
/// WHICH TRANSLATOR THE SUPERDSC NEEDS — `DSC2ToDataflowIRUtils.hpp:22`.
///
/// ⛔ FIRST TRIP WINS AND THE LOOP STOPS: a compute-less DSC standing AFTER a DSC1.0 answers
/// [`TranslatorVersion::V1`], not [`TranslatorVersion::V1NoComputeOp`] (`:25-33`).
///
/// ⭐ AND AN EMPTY `dscs_` IS [`TranslatorVersion::V3`] — the initial `version = 3` with nothing to
/// lower it.
#[must_use]
pub fn translator_version(dscs: &[DscKind]) -> TranslatorVersion {
    for dsc in dscs {
        match dsc {
            DscKind::NoComputeOp => return TranslatorVersion::V1NoComputeOp,
            DscKind::Dsc1 => return TranslatorVersion::V1,
            DscKind::Dsc2 => {}
        }
    }
    TranslatorVersion::V3
}

/// Replaces: e007_createGetLocalUnitOp
///
/// THE HANDLE FOR A REGISTER FILE OF A UNIT ALREADY HELD — `DSC2ToDataflowIRUtils.hpp:44`.
///
/// ⛔⛔ THE REUSE ARM IS UNREACHABLE FROM EVERY CALLER IN THE TREE, AND IT IS STILL EMITTED.
/// `component_to_handler_` never keys the three LRFREGs — `buildNeighborUnits` files their results
/// under the generic `LRFREG` (`:177-179`) — and `PTXRF`, `PTARF` and `L0_SCALE` are each asked for
/// exactly once per unit, after `initializeUnit`'s `component_to_handler_.clear()` (`:628`). So
/// `count(comp)` is 0 at every call site; deleting the else arm would be a port that could not state
/// which arm its callers take.
///
/// ⭐ THE RESULT TYPE IS `index`, NOT THE FILE'S (`:48`): it is a HANDLE to the file, the same shape a
/// `get_unit` binds, and [`dataflow::LocalUnit::spelling`] is the `name=` the reference reads out of
/// `senComponentsToString`.
pub fn create_get_local_unit_op(
    vals: &mut Values,
    held: Option<Val>,
    which: dataflow::LocalUnit,
    of: Val,
) -> Retrieved {
    match held {
        Some(handle) => Retrieved::Reused(handle),
        None => Retrieved::Created(dataflow::Op::GetLocalUnit {
            result: vals.mint(),
            of,
            which,
        }),
    }
}

/// Replaces: e008_setPrecisionInUnitOp
///
/// PUTS `precision =` ON A COMPUTE UNIT'S PROGRAM AND ON NO OTHER — `DSC2ToDataflowIRUtils.hpp:143`.
///
/// ⛔⛔ THE PARTIAL-MAP `find` AND THE `is_any_of(PT, PE, SFP)` ARE ONE TEST. The `find != end()`
/// guard (`:146-147`) covers a map that lacks `L0`, `CONSTANT` and `SFPRING`
/// (`arch_enums.cpp:124-211`), none of which is PT, PE or SFP — so [`DfirUnit::generic`], which is
/// total where that map throws, answers both halves.
///
/// ⛔ THE COMPONENT IS THE UNIT'S OWN. `unit_op.on.kind()` is what `comp` names, so taking them
/// separately would be a pair that could disagree about which unit is being written.
///
/// ⛔ AND `DT_CHECK_MSG(precision != "", "Invalid compute precision")` (`:150`) IS
/// [`dataflow::Precision`] — an empty spelling is not one of its arms.
pub fn set_precision_in_unit_op<A: Arch>(
    unit_op: &mut ProgramUnit<A>,
    precision: dataflow::Precision,
) {
    if matches!(
        unit_op.on.kind().generic(),
        GenericComp::Pt | GenericComp::Pe | GenericComp::Sfp
    ) {
        unit_op.precision = Some(precision);
    }
}

/// Replaces: e009_constructAVectorOfIndexType
///
/// APPENDS ONE `index` PER FOLD to a type list — `DSC2ToDataflowIRUtils.hpp:711`.
///
/// ⛔ CORRECTION — NEITHER CALL SITE RELIES ON THE APPEND. An earlier note here said `:72` and
/// `:679` each hand it a list that already holds the unit's own result type; they do not. Both
/// declare a FRESH `SmallVector<Type> get_unit_type;` on the line above the call (`:71-72` and
/// `:678-679`), and those are the only two callers in the tree. The `emplace_back` near the second
/// one is onto `units`, a different vector (`:687-688`). So the shape is the reference's and the
/// list is always empty on entry — a version that cleared it would be observationally identical.
///
/// ⛔ THE COUNT IS ALWAYS `num_folds_`, never a free integer — a folded `get_unit` returns one
/// address per fold, so the arity IS the fold count.
pub fn append_index_types(folds: NumFolds, types: &mut Vec<ScalarTy>) {
    for _ in 0..folds.0 {
        types.push(ScalarTy::Index);
    }
}

/// Replaces: e010_emitError
///
/// THE TRANSLATOR'S DIAGNOSTIC TEXT — `DSC2ToDataflowIRUtils.hpp:718`.
///
/// ⛔ THE PREFIX IS THE WHOLE FUNCTION. `module_op_->emitError` is MLIR's sink and this crate has
/// none: a lowering that reported instead of emitting would be the runtime refusal
/// `dfir_never_runtime_refuses.rs` freezes at zero. The text is handed back so the caller that
/// cannot proceed names it in its own `todo!`.
#[must_use]
pub fn error_diagnostic(message: &str) -> String {
    format!("[DSC2.0 to Dataflow IR]: {message}")
}

/// Replaces: e044_createGetUnitOp
///
/// THE HANDLE FOR A UNIT OF A NAMED CORE AND CORELET, created once and reused after —
/// `DSC2ToDataflowIRUtils.hpp:64`.
///
/// ⛔⛔ THE MAP IT TESTS AND THE MAP IT WRITES ARE TWO DIFFERENT MAPS. It asks
/// `component_to_handler_.count(comp)` (`:69`) and stores into
/// `unit_to_value_map_[core_id][corelet_id][comp]` (`:82`), so a second call for the SAME triple does
/// not hit the entry the first one just made. That is not an oversight to repair: it is what makes
/// [`e072`](super::utils)'s cores×corelets loop emit one op per handle (`:101-119`) instead of
/// collapsing the whole set onto one, and a cache keyed by the triple would break that function's own
/// `DT_CHECK(get_unit_ops.size() == units.size())` at `:122`. So the reuse this takes is the
/// COMPONENT'S handler and nothing finer — one [`Option`], not a lookup.
///
/// ⛔ AND THAT IS WHY THERE ARE TWO ARMS AND NOT THREE. The reference's else branch is
/// `component_to_handler_[comp].getDefiningOp<dataflow::GetUnitOp>()` (`:86`), which yields NULL for a
/// handler whose defining op is something else — a `uniform.query_map` result, say. Only
/// `initializeUnit` (`:648-649`) and `buildNeighborUnits` (`:167-175`) ever write that map for a
/// component this can be called with, and both write `get_unit` results; the uniformized path stores
/// its query result under a guard that stops this function being reached at all (`:100`). A third
/// variant would be a state no caller can produce.
///
/// ⭐⭐ THE `index` LIST **IS** THE `num_folds` FIELD. `constructAVectorOfIndexType(num_folds_, ..)`
/// (`:72`, entry 009) sizes the op's RESULT GROUP, and [`dataflow::Op::GetUnit`] prints that group off
/// `num_folds` — `%28:11 = dataflow.get_unit {.. num_folds = 11 : i32 ..} : index, index, ..`. So the
/// type vector and the attribute are one fact here, and `getResult(i)` is `%28#i`.
///
/// ⛔ THE `-1` IS AN ABSENT ATTRIBUTE, NOT A CORELET ZERO. `if (corelet_id != -1)` gates the
/// `corelet` attribute (`:80-81`) and the default argument is `-1` (`DSC2ToDataflowIR.hpp:122`); the
/// island's [`Residency`] is where that gate lives, since `Residency::Scratchpad` prints `core` with
/// no `corelet` where `Residency::CoreWide` prints `corelet = 0`. Its one `-1` caller is the L3 pair
/// (`:104`), which the island's own golden already writes as a `Scratchpad` `l3lu`.
///
/// ⛔ AND THE NAME IS NOT `initializeUnit`'S. This one writes `name = type =
/// senComponentsToString.at(comp)` (`:75-77`) where `initializeUnit` writes `type + "-CL" +
/// corelet_id` (`:632-633`) for the same op kind; the island derives both from the residency and the
/// unit, and [`dataflow::Op::GetUnit`]'s own note records why.
pub fn create_get_unit_op(
    vals: &mut Values,
    held: Option<Val>,
    unit: DfirUnit,
    residency: Residency,
    folds: NumFolds,
) -> Retrieved {
    match held {
        Some(handle) => Retrieved::Reused(handle),
        None => Retrieved::Created(dataflow::Op::GetUnit {
            result: vals.mint(),
            residency,
            unit,
            num_folds: Some(folds),
        }),
    }
}

// crustify:todo: e072_createUniformizedGetUnitOp
// crustify:todo: e073_buildNeighborUnits
// crustify:todo: e083_buildUniformizedNeighborUnits
// crustify:todo: e084_initializeUnit
// crustify:todo: e093_initializeUniformizedUnit

#[cfg(test)]
mod unit_tests {
    use super::{
        DscKind, TranslatorVersion, append_index_types, create_get_local_unit_op,
        create_get_unit_op, error_diagnostic, set_precision_in_unit_op, translator_version,
    };
    use crate::arch::Dd2;
    use crate::bridges::superdsc_to_dataflow_ir::dsc_lowering::Retrieved;
    use crate::islands::dataflow_ir::dialects::{Val, dataflow};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{ProgramUnit, Units, Values};
    use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency, Row};

    /// ⛔ IT APPENDS, THOUGH NOTHING IN THE TREE DEPENDS ON THAT: both call sites hand it a fresh
    /// empty vector, so this pins the reference's shape rather than a caller's requirement.
    #[test]
    fn one_index_per_fold_appended_after_what_is_there() {
        let mut types = vec![ScalarTy::Int(1)];
        append_index_types(NumFolds(3), &mut types);
        assert_eq!(
            types,
            vec![
                ScalarTy::Int(1),
                ScalarTy::Index,
                ScalarTy::Index,
                ScalarTy::Index
            ]
        );
    }

    #[test]
    fn the_diagnostic_carries_the_translator_prefix() {
        assert_eq!(
            error_diagnostic("Invalid compute precision"),
            "[DSC2.0 to Dataflow IR]: Invalid compute precision"
        );
    }
    /// 🎯 044/110 — ⭐ THE FOLD COUNT IS THE OP'S RESULT ARITY, and the corelet the residency names.
    #[test]
    fn an_unheld_component_creates_a_folded_get_unit() {
        let mut vals = Values::default();
        let row0 = DfirUnit::PtRow(Row::checked(0).expect("row 0"));
        let residency = Residency::Corelet {
            core: Core::checked(0).expect("core 0"),
            corelet: Corelet::checked(1).expect("corelet 1"),
        };
        assert_eq!(
            create_get_unit_op(&mut vals, None, row0, residency, NumFolds(11)),
            Retrieved::Created(dataflow::Op::GetUnit {
                result: Val(0),
                residency,
                unit: row0,
                num_folds: Some(NumFolds(11)),
            })
        );
        // ⛔ THE `-1` CALLER: the L3 halves carry `core` and no `corelet` attribute.
        let core_only = Residency::Scratchpad {
            core: Core::checked(0).expect("core 0"),
        };
        assert_eq!(
            create_get_unit_op(&mut vals, None, DfirUnit::L3lu, core_only, NumFolds::ONE),
            Retrieved::Created(dataflow::Op::GetUnit {
                result: Val(1),
                residency: core_only,
                unit: DfirUnit::L3lu,
                num_folds: Some(NumFolds::ONE),
            })
        );
    }

    /// 🎯 044/110 — ⛔ A HELD COMPONENT MINTS NOTHING: the whole point of the cache is that the
    /// second request for a component emits no second `get_unit`.
    #[test]
    fn a_held_component_reuses_its_handler_and_mints_nothing() {
        let mut vals = Values::default();
        let handle = vals.mint();
        assert_eq!(
            create_get_unit_op(
                &mut vals,
                Some(handle),
                DfirUnit::L0lu,
                Residency::Corelet {
                    core: Core::checked(0).expect("core 0"),
                    corelet: Corelet::checked(0).expect("corelet 0"),
                },
                NumFolds::ONE,
            ),
            Retrieved::Reused(handle)
        );
        assert_eq!(vals.issued(), 1, "the reuse arm creates no value");
    }
    /// 🎯 006/110 — ⛔ FIRST TRIP WINS: the compute-less DSC behind a DSC1.0 is never reached, and an
    /// empty SuperDSC is the initial `version = 3`.
    #[test]
    fn the_version_is_decided_by_the_first_dsc_that_is_not_a_dsc2() {
        assert_eq!(
            translator_version(&[DscKind::Dsc2, DscKind::Dsc2]),
            TranslatorVersion::V3
        );
        assert_eq!(translator_version(&[]), TranslatorVersion::V3);
        assert_eq!(
            translator_version(&[DscKind::Dsc2, DscKind::NoComputeOp, DscKind::Dsc1]),
            TranslatorVersion::V1NoComputeOp
        );
        // ⛔ ORDER, NOT PRESENCE: the same two DSCs the other way round answer differently.
        assert_eq!(
            translator_version(&[DscKind::Dsc1, DscKind::NoComputeOp]),
            TranslatorVersion::V1
        );
    }

    /// 🎯 007/110 — ⛔⛔ THE REUSE ARM IS UNREACHABLE FROM EVERY CALLER, so this is the only place it
    /// is ever taken; the created arm is what the translator actually emits.
    #[test]
    fn an_unheld_register_file_creates_a_get_local_unit_on_the_unit_it_belongs_to() {
        let mut vals = Values::default();
        let unit = vals.mint();
        assert_eq!(
            create_get_local_unit_op(&mut vals, None, dataflow::LocalUnit::PtXrf, unit),
            Retrieved::Created(dataflow::Op::GetLocalUnit {
                result: Val(1),
                of: unit,
                which: dataflow::LocalUnit::PtXrf,
            })
        );
        let held = vals.mint();
        assert_eq!(
            create_get_local_unit_op(&mut vals, Some(held), dataflow::LocalUnit::PtXrf, unit),
            Retrieved::Reused(held)
        );
        assert_eq!(vals.issued(), 3, "the reuse arm mints nothing");
    }

    /// 🎯 008/110 — ⛔ THE PT, PE AND SFP TAKE IT AND NOTHING ELSE DOES: a mover's program unit is
    /// left without the attribute, which is what `find != end()` plus `is_any_of(PT, PE, SFP)` says.
    #[test]
    fn only_a_compute_unit_takes_the_precision_attribute() {
        let unit_of = |kind| ProgramUnit::<Dd2> {
            on: Units::one(kind, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        for kind in [
            DfirUnit::PtRow(Row::checked(3).expect("row 3")),
            DfirUnit::Pe,
            DfirUnit::Sfp,
        ] {
            let mut unit = unit_of(kind);
            set_precision_in_unit_op(&mut unit, dataflow::Precision::Fp8);
            assert_eq!(unit.precision, Some(dataflow::Precision::Fp8), "{kind:?}");
        }
        // ⛔ AND THE THREE THE REFERENCE'S MAP THROWS ON ARE AMONG THE ONES IT SKIPS.
        for kind in [
            DfirUnit::Lxlu,
            DfirUnit::L0,
            DfirUnit::Constant,
            DfirUnit::SfpRing,
        ] {
            let mut unit = unit_of(kind);
            set_precision_in_unit_op(&mut unit, dataflow::Precision::Fp8);
            assert_eq!(unit.precision, None, "{kind:?}");
        }
    }
}
