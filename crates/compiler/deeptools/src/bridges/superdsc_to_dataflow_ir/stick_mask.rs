//! THE STICK MASK — the SAMV set-transfer-mask-state op.
//!
//! 1 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e060_constructStickMaskOperation` | 1 | 58 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNStickMaskLowering.cpp:21` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.


use super::dsc_lowering::{constant_index, mlir_type_from_dsc_data_format};
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::agen::{MaskCounts, SliceMask};
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, agen};
use crate::islands::dataflow_ir::ty::{TensorCategory, Vector};

/// HOW MANY SLICES A STICK HAS — `int num_slices = 8` (`SNStickMaskLowering.cpp:47`), which is the
/// `num_slices` attribute and the length of the slice mask map.
const NUM_SLICES: i32 = 8;

/// 128 BYTES OF MASK — `(128 * 8) / getIntOrFloatBitWidth(element_type)`, the reference's own
/// `// 128B/size` (`SNStickMaskLowering.cpp:42-45`).
const MASK_BITS: u64 = 128 * 8;

/// THE STICK MASK'S VIEW — `stick_mask_->getView()`, as the SAMV attributes read it.
///
/// ⛔ EACH MASK IS `(first, second)` = `(unmasked, masked)`, and the reference builds the two
/// attribute arrays by column: `unmasked_offsets = {maskA_.first, maskB_.first}` against
/// `masked_offsets = {maskA_.second, maskB_.second}` (`SNStickMaskLowering.cpp:26-31`). Transposing
/// them prints a mask whose two counts have swapped roles and still verifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StickMaskView {
    /// `maskA_`.
    pub mask_a: MaskCounts,
    /// `maskB_`.
    pub mask_b: MaskCounts,
    /// `transitionSliceId_` — the one slice masked from BOTH; below it is `A`, above it is `(1)`.
    pub transition_slice: i32,
}

/// Replaces: e060_constructStickMaskOperation
///
/// THE `agen.set_transfer_mask_state` FOR A STICK MASK — an `arith.constant` holding the mask value,
/// then the SAMV op over an 8-entry slice map and the two mask elements
/// (`SNStickMaskLowering.cpp:21-78`).
///
/// ⛔ THE MASK VALUE IS ONE SCALAR, WHICH IS WHY IT IS A PARAMETER: `DT_CHECK_MSG(all_data.size() == 1,
/// "No folding over SAMV values")` and `front().size() == 1` (`:70-73`) — a fold space, and a vector,
/// are both refused.
///
/// ⚠️ [`None`] IS THE `NoneType` ELEMENT — `BOOL` alone — where the reference divides by a width the
/// type does not have. Nothing is emitted on that path.
#[must_use]
pub fn construct_stick_mask_operation(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    view: StickMaskView,
    name: &str,
    format: DataType,
    mask_value: i64,
) -> Option<Val> {
    // `getMLIRTypeFromDSCDataFormat(cst_info.dataFormat_, REGULAR_TENSOR, builder)` — the category is
    // the literal, not the constant's own.
    let elem = mlir_type_from_dsc_data_format(format, TensorCategory::Regular)?;
    let ty = Vector {
        len: MASK_BITS / u64::from(elem.bits()),
        elem,
    };

    // `for (i < num_slices) { i < transition ? "(A)" : i == transition ? "(A|B)" : "(1)" }`
    let slice_mask_map: Vec<SliceMask> = (0..NUM_SLICES)
        .map(|slice| match slice.cmp(&view.transition_slice) {
            std::cmp::Ordering::Less => SliceMask::A,
            std::cmp::Ordering::Equal => SliceMask::AOrB,
            std::cmp::Ordering::Greater => SliceMask::Full,
        })
        .collect();

    // `mask_val = ConstantIndexOp::create(builder, loc, all_data.front()[0])`
    let mask_val = constant_index(vals, ops, mask_value);

    let result = vals.mint();
    ops.push(DfirOp::Agen(agen::Op::SetTransferMaskState {
        result,
        mask_value: mask_val,
        dbg_name: Some(name.to_owned()),
        slice_mask_map,
        masks: vec![view.mask_a, view.mask_b],
        ty,
    }));
    Some(result)
}

#[cfg(test)]
mod unit_tests {
    use super::{StickMaskView, construct_stick_mask_operation};
    use crate::generated::DataType;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::Op as DfirOp;
    use crate::islands::dataflow_ir::dialects::agen::MaskCounts;
    use crate::islands::dataflow_ir::print;

    /// ⛔ THE VENDOR'S OWN ROUND-TRIP LINE, BYTE FOR BYTE — `dcc/test/Dialect/Agen/
    /// set_transfer_mask_state_rt.mlir:10`, whose slice map transitions on slice 5 and whose
    /// `vector<128xi8>` is `128 * 8 / 8` elements.
    ///
    /// The two spaces after the type colon and the space before the comma are the printer's
    /// (`Agen.cpp:2705`); `dbo-opt` parses that text and nothing else.
    #[test]
    fn the_vendors_round_trip_line_is_reproduced_from_the_view() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let result = construct_stick_mask_operation(
            &mut vals,
            &mut ops,
            StickMaskView {
                mask_a: MaskCounts {
                    unmasked: 8,
                    masked: 8,
                },
                mask_b: MaskCounts {
                    unmasked: 1,
                    masked: 1,
                },
                transition_slice: 5,
            },
            "samv_0",
            DataType::Senint8,
            3,
        );

        let mut text = String::new();
        for op in &ops {
            print::emit(&mut text, op, 0);
        }
        assert_eq!(
            text,
            concat!(
                "%0 = arith.constant 3 : index\n",
                "%1 = agen.set_transfer_mask_state mask_value(%0) { num_slices = 8 : i32, ",
                "slice_mask_map = \"(A)(A)(A)(A)(A)(A|B)(1)(1)\", ",
                "maskA = \"(unmasked = 8 : i32, masked = 8 : i32)\", ",
                "maskB = \"(unmasked = 1 : i32, masked = 1 : i32)\" } :  index , vector<128xi8>\n",
            )
        );
        assert_eq!(result, Some(crate::islands::dataflow_ir::dialects::Val(1)));

        // ⚠️ `BOOL` has no element type, so nothing is emitted and no width is divided by.
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        assert_eq!(
            construct_stick_mask_operation(
                &mut vals,
                &mut ops,
                StickMaskView {
                    mask_a: MaskCounts {
                        unmasked: 8,
                        masked: 8,
                    },
                    mask_b: MaskCounts {
                        unmasked: 1,
                        masked: 1,
                    },
                    transition_slice: 5,
                },
                "samv_0",
                DataType::Bool,
                3,
            ),
            None
        );
        assert!(ops.is_empty());
    }
}
