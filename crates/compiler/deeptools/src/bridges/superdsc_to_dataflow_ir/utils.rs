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

use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::units::NumFolds;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e006_getTranslatorVersion
// crustify:todo: e007_createGetLocalUnitOp
// crustify:todo: e008_setPrecisionInUnitOp

/// Replaces: e009_constructAVectorOfIndexType
///
/// APPENDS ONE `index` PER FOLD to a type list — `DSC2ToDataflowIRUtils.hpp:711`.
///
/// ⛔ IT APPENDS, AND BOTH CALL SITES RELY ON IT: `:72` and `:679` each `emplace_back` onto a list
/// that already holds the unit's own result type, so a version that cleared it would drop that.
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

// crustify:todo: e044_createGetUnitOp
// crustify:todo: e072_createUniformizedGetUnitOp
// crustify:todo: e073_buildNeighborUnits
// crustify:todo: e083_buildUniformizedNeighborUnits
// crustify:todo: e084_initializeUnit
// crustify:todo: e093_initializeUniformizedUnit

#[cfg(test)]
mod unit_tests {
    use super::{append_index_types, error_diagnostic};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::units::NumFolds;

    /// ⛔ IT APPENDS: the unit's own result type is already in the list at `:72`.
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
}
