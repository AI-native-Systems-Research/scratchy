//! THE TRANSFER STATEMENTS — load, store and send, and how each is walked over the AGEN time axis.
//! Carries the 2B/16B store shuffles, the constant bit streams, the burst and interleave settings.
//!
//! 29 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e034_constructLogicalMemoryViewOp` | 0 | 32 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:94` |
//! | `e035_constructTimeOrder` | 0 | 12 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:206` |
//! | `e036_constructTimeAddressMap` | 0 | 29 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:224` |
//! | `e037_getImmediateParentWithMatchingDim` | 0 | 33 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:670` |
//! | `e038_areEpiloguesInTransferSizes` | 0 | 12 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1491` |
//! | `e039_construct2B16BLoadShuffle` | 0 | 86 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1778` |
//! | `e040_construct2B16BStoreShuffle` | 0 | 87 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1865` |
//! | `e063_getLabeledDsType` | 1 | 21 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:28` |
//! | `e064_getBufferingOrStreamingMode` | 1 | 39 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:50` |
//! | `e065_constructBaseAddress` | 1 | 73 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:132` |
//! | `e066_constructTimeSet` | 1 | 47 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:260` |
//! | `e067_constructLoadOrStoreSet` | 1 | 90 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:314` |
//! | `e068_constructImplicitLoopsForContiguousTransfer` | 1 | 137 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:707` |
//! | `e069_areEpiloguesInLoops` | 1 | 24 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1464` |
//! | `e070_GenerateConstantBitStreamAndShuffle` | 1 | 41 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2475` |
//! | `e077_constructElementsOfAgenDataTransfer` | 2 | 68 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:412` |
//! | `e078_constructElementsOfAgenCompositeDataTransfer` | 2 | 94 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:488` |
//! | `e079_constructElementsOfAffineDataTransferViaAgenTransfer` | 2 | 77 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:590` |
//! | `e080_constructStreamingOrDoubleBufferingLoad` | 2 | 347 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:851` |
//! | `e081_GenerateReceiveAndSendFromDataTransferNode` | 2 | 66 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1395` |
//! | `e082_ConstructSAMVOperation` | 2 | 96 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1957` |
//! | `e089_constructStreamingOrDoubleBufferingStore` | 3 | 185 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1205` |
//! | `e090_GenerateLoadAndSendFromDataTransferNode` | 3 | 274 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2058` |
//! | `e091_GenerateLoadAndStoreFromDataTransferNode` | 3 | 132 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2337` |
//! | `e092_GenerateDataTransfersForViaIfSo` | 3 | 52 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2617` |
//! | `e097_GenerateReceiveAndStoreFromDataTransferNode` | 4 | 269 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1508` |
//! | `e098_GenerateDataTranferForSrc` | 4 | 34 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2522` |
//! | `e102_GenerateDataTranferForDst` | 5 | 47 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2563` |
//! | `e104_constructDataTransfer` | 6 | 164 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2676` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e034_constructLogicalMemoryViewOp
// crustify:todo: e035_constructTimeOrder
// crustify:todo: e036_constructTimeAddressMap
// crustify:todo: e037_getImmediateParentWithMatchingDim
// crustify:todo: e038_areEpiloguesInTransferSizes
// crustify:todo: e039_construct2B16BLoadShuffle
// crustify:todo: e040_construct2B16BStoreShuffle
// crustify:todo: e066_constructTimeSet
// crustify:todo: e067_constructLoadOrStoreSet
// crustify:todo: e068_constructImplicitLoopsForContiguousTransfer
// crustify:todo: e069_areEpiloguesInLoops
// crustify:todo: e070_GenerateConstantBitStreamAndShuffle
// crustify:todo: e077_constructElementsOfAgenDataTransfer
// crustify:todo: e078_constructElementsOfAgenCompositeDataTransfer
// crustify:todo: e079_constructElementsOfAffineDataTransferViaAgenTransfer
// crustify:todo: e080_constructStreamingOrDoubleBufferingLoad
// crustify:todo: e081_GenerateReceiveAndSendFromDataTransferNode
// crustify:todo: e082_ConstructSAMVOperation
// crustify:todo: e089_constructStreamingOrDoubleBufferingStore
// crustify:todo: e090_GenerateLoadAndSendFromDataTransferNode
// crustify:todo: e091_GenerateLoadAndStoreFromDataTransferNode
// crustify:todo: e092_GenerateDataTransfersForViaIfSo
// crustify:todo: e097_GenerateReceiveAndStoreFromDataTransferNode
// crustify:todo: e098_GenerateDataTranferForSrc
// crustify:todo: e102_GenerateDataTranferForDst
// crustify:todo: e104_constructDataTransfer

use super::dsc_lowering::{Component, constant_index, mlir_type_from_dsc_data_format};
use crate::arch::Elements;
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, TensorCategory, Vector};

/// Replaces: e063_getLabeledDsType
///
/// THE `vector<NxT>` ONE TRANSFER MOVES IN A UNIT OF TIME — the product of
/// `unitTimeTransferChunkSize_`'s per-dim sizes, times the chunk count
/// (`SNTransferLowering.cpp:28-48`).
///
/// ⛔ THE CHUNK COUNT MULTIPLIES ONLY WHERE IT IS POSITIVE (`:36-37`): zero means *not chunked*, not
/// *no elements*, and multiplying by it would type every unchunked transfer `vector<0xT>`.
///
/// ⚠️ [`None`] IS THE REFERENCE'S `LogicalResult::failure()` — `isa<NoneType>(element_type)`, which
/// among the generated formats is `BOOL` alone (`:43-45`).
#[must_use]
pub fn labeled_ds_type(
    chunk_sizes: &[Elements],
    num_chunks: i64,
    format: DataType,
    category: TensorCategory,
) -> Option<Vector> {
    // `int elements = 1; for (dim_size : unitTimeTransferChunkSize_) elements *= dim_size.sizeDim_.size_;`
    let mut elements: u64 = 1;
    for chunk in chunk_sizes {
        elements = elements.saturating_mul(chunk.0);
    }
    if num_chunks > 0 {
        elements = elements.saturating_mul(num_chunks.unsigned_abs());
    }

    Some(Vector {
        len: elements,
        elem: mlir_type_from_dsc_data_format(format, category)?,
    })
}

/// WHETHER A TRANSFER'S END DOUBLE-BUFFERS, STREAMS, OR DOES NEITHER — the reference's `int &mode`
/// with its three magic values `-1`, `1` and `2` (`SNTransferLowering.cpp:51,61,64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferingMode {
    /// `-1` — "Neither buffering or streaming".
    Neither,
    /// `1` — buffering, over a fixed number of buffers.
    Buffering,
    /// `2` — streaming.
    Streaming,
}

/// HOW MANY BUFFERS AN ALLOCATION HOLDS — `allocateNode_->numBuffers_`.
///
/// ⛔ [`Buffers::Streaming`] IS THE REFERENCE'S `-1` SENTINEL (`:59`), which is the whole decision
/// this function makes; a raw count would let `-2` reach it and be read as buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffers {
    /// `numBuffers_ == -1`.
    Streaming,
    /// Any other count.
    Count(u32),
}

/// ONE END OF A TRANSFER, AS THE MODE TEST READS IT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferEnd {
    /// `src_.unit_`, or a via's `loc_.unit_`.
    pub unit: Component,
    /// The buffers of this end's labeled DS at this end's storage — [`None`] where
    /// `bufferSwitchPosition_` is null, which is an end that switches no buffer.
    pub buffers: Option<Buffers>,
}

/// Replaces: e064_getBufferingOrStreamingMode
///
/// WHETHER **THIS** COMPONENT'S END OF A TRANSFER BUFFERS OR STREAMS — the source first, then the
/// destination vias in order (`SNTransferLowering.cpp:50-88`).
///
/// ⛔⛔ AN END ON `comp` WITH NO BUFFER SWITCH FALLS THROUGH RATHER THAN ANSWERING. Neither the source
/// arm nor the via arm returns when `bufferSwitchPosition_` is null (`:53`, `:71`), so a source that
/// switches nothing lets a via decide and a via that switches nothing lets the NEXT via decide.
///
/// ⛔ THE FIRST MATCHING VIA WINS — the loop returns inside the body (`:62-66`), so a later via on the
/// same component is never consulted.
#[must_use]
pub fn buffering_or_streaming_mode(
    comp: Component,
    src: TransferEnd,
    dst_vias: &[TransferEnd],
) -> BufferingMode {
    if src.unit == comp
        && let Some(buffers) = src.buffers
    {
        return mode_of(buffers);
    }

    for via in dst_vias {
        if via.unit == comp
            && let Some(buffers) = via.buffers
        {
            return mode_of(buffers);
        }
    }

    BufferingMode::Neither
}

/// `num_buffers == -1 ? 2 : 1` — streaming or buffering (`SNTransferLowering.cpp:59-65`).
const fn mode_of(buffers: Buffers) -> BufferingMode {
    match buffers {
        Buffers::Streaming => BufferingMode::Streaming,
        Buffers::Count(_) => BufferingMode::Buffering,
    }
}

/// ONE ENTRY OF `TransferNode::UnitView::LoopInfo` — which dim of the address it strides, by how much,
/// and the induction variable it strides against.
///
/// ⛔ `iv` IS [`None`] ONLY FOR `loop_ == nullptr`, THE CONSTANT-OFFSET CASE (`:2767-2770`). The
/// reference has a third state this type has not: a non-null `loop_` whose `getMLIRLoopFromLoopNode`
/// answers neither loop kind bumps `unique_variables` and pushes NO argument (`:2762-2766`), leaving a
/// map whose arity exceeds its operand list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopStride {
    /// `sizeIdx_` — which dim of the base address this stride belongs to.
    pub size_idx: usize,
    /// `elemOffset_`, in elements.
    pub elem_offset: i64,
    /// The induction variable of `getMLIRLoopFromLoopNode(loop_, dim_)`.
    pub iv: Option<Val>,
}

/// WHICH OF THE THREE BASE-ADDRESS FORMS IS BUILT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressForm {
    /// `!bypass_strides && !is_scalereg` — one expression per dim, over the loops that stride it.
    Strided,
    /// A scale register: `ndims` zero results and `ndims` zero operands.
    ScaleReg,
    /// Strides bypassed: `getConstantMap(0)`, which is ZERO dims and one zero result.
    Bypass,
}

impl AddressForm {
    /// ⛔ SCALEREG WINS: the reference's else-branch tests `is_scalereg` FIRST (`:2798`), so a
    /// transfer that both bypasses its strides and is a scale register takes the scale-register form.
    #[must_use]
    pub const fn of(bypass_strides: bool, is_scalereg: bool) -> AddressForm {
        if is_scalereg {
            AddressForm::ScaleReg
        } else if bypass_strides {
            AddressForm::Bypass
        } else {
            AddressForm::Strided
        }
    }
}

/// A BASE ADDRESS — the map and the operands it is applied to, in the order the map's dims name them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseAddress {
    /// `base_address_map`, whose dim count is `unique_variables`.
    pub map: AffineMap,
    /// `base_address_args`, one per dim of the map.
    pub args: Vec<Val>,
}

/// `base_address_expr + mul_expr` WITH THE TWO FOLDS MLIR'S OWN BUILDER APPLIES HERE — the additive
/// identity and a constant sum. The island's [`AffineExpr::plus`] deliberately does not fold, so
/// building the reference's literal `getAffineConstantExpr(0) + ..` would print `0 + d0 * 4`.
fn add_stride(acc: Option<AffineExpr>, term: AffineExpr) -> Option<AffineExpr> {
    match (acc, term) {
        (None, term) => Some(term),
        (Some(AffineExpr::Const(lhs)), AffineExpr::Const(rhs)) => Some(
            lhs.checked_add(rhs).map_or_else(
                || AffineExpr::Const(lhs).plus(AffineExpr::Const(rhs)),
                AffineExpr::Const,
            ),
        ),
        (Some(acc), term) => Some(acc.plus(term)),
    }
}

/// Replaces: e065_constructBaseAddress
///
/// THE BASE ADDRESS OF A TRANSFER — one affine result per dim, summing each loop's induction variable
/// times its element offset (`SNTransferLowering.cpp:132-204`).
///
/// ⛔⛔ A DIM NO STRIDE MENTIONS STILL CONSUMES A DIM AND AN OPERAND: the `!modified` arm bumps
/// `unique_variables` and pushes a zero constant while its result stays the constant 0 (`:2788-2794`),
/// so the map has a dim its results never read. Dropping it renumbers every later `d<n>`.
///
/// ⛔ [`AddressForm::Bypass`] IS ZERO DIMS AND NO OPERANDS — `getConstantMap(0)` (`:2810`) — where
/// [`AddressForm::ScaleReg`] is `ndims` of each.
#[must_use]
pub fn construct_base_address(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    ndims: usize,
    loop_strides: &[LoopStride],
    form: AddressForm,
) -> BaseAddress {
    match form {
        AddressForm::Strided => {
            let mut unique_variables: u32 = 0;
            let mut results = Vec::with_capacity(ndims);
            let mut args = Vec::new();
            for dim in 0..ndims {
                let mut expr: Option<AffineExpr> = None;
                for stride in loop_strides.iter().filter(|stride| stride.size_idx == dim) {
                    let term = match stride.iv {
                        // `auto iv_expr = getAffineDimExpr(unique_variables++, context);
                        //  mul_expr = iv_expr * loop_strides[i].elemOffset_;`
                        Some(iv) => {
                            let iv_expr = AffineExpr::dim(unique_variables);
                            unique_variables = unique_variables.saturating_add(1);
                            args.push(iv);
                            match stride.elem_offset {
                                0 => AffineExpr::Const(0),
                                1 => iv_expr,
                                offset => iv_expr.times(offset),
                            }
                        }
                        // `mul_expr = getAffineConstantExpr(loop_strides[i].elemOffset_, context);`
                        None => AffineExpr::Const(stride.elem_offset),
                    };
                    expr = add_stride(expr, term);
                }

                // `if (!modified) { unique_variables++; base_address_args.push_back(<zero>); }`
                if expr.is_none() {
                    unique_variables = unique_variables.saturating_add(1);
                    args.push(constant_index(vals, ops, 0));
                }
                results.push(expr.unwrap_or(AffineExpr::Const(0)));
            }

            BaseAddress {
                map: AffineMap {
                    dims: unique_variables,
                    syms: 0,
                    results,
                },
                args,
            }
        }
        AddressForm::ScaleReg => BaseAddress {
            map: AffineMap::constants(
                u32::try_from(ndims).unwrap_or(u32::MAX),
                &vec![0; ndims],
            ),
            args: (0..ndims)
                .map(|_| constant_index(vals, ops, 0))
                .collect(),
        },
        AddressForm::Bypass => BaseAddress {
            map: AffineMap::constants(0, &[0]),
            args: Vec::new(),
        },
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AddressForm, BaseAddress, BufferingMode, Buffers, LoopStride, TransferEnd,
        buffering_or_streaming_mode, construct_base_address, labeled_ds_type,
    };
    use crate::arch::Elements;
    use crate::bridges::superdsc_to_dataflow_ir::dsc_lowering::Component;
    use crate::generated::DataType;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, arith};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, TensorCategory, Vector};
    use crate::units::DfirUnit;

    /// 🎯 063/110 — ⛔ A ZERO CHUNK COUNT DOES NOT MULTIPLY, and `BOOL` has no type at all.
    ///
    /// Multiplying by a zero count types the transfer `vector<0xT>`, which the backend accepts and
    /// which moves nothing.
    #[test]
    fn an_unchunked_transfer_keeps_the_product_of_its_chunk_sizes() {
        let sizes = [Elements(4), Elements(8)];
        assert_eq!(
            labeled_ds_type(&sizes, 0, DataType::Bfloat16, TensorCategory::Regular),
            Some(Vector {
                len: 32,
                elem: ElemType::Bf16,
            })
        );
        assert_eq!(
            labeled_ds_type(&sizes, 3, DataType::Bfloat16, TensorCategory::Regular),
            Some(Vector {
                len: 96,
                elem: ElemType::Bf16,
            })
        );
        assert_eq!(
            labeled_ds_type(&sizes, 3, DataType::Bool, TensorCategory::Regular),
            None
        );
    }

    /// 🎯 064/110 — ⛔ AN END ON THIS COMPONENT THAT SWITCHES NO BUFFER FALLS THROUGH TO THE VIAS,
    /// and the first via that does switch wins.
    ///
    /// Answering `Neither` at the source would leave a streaming destination lowered as a plain
    /// transfer, and taking the last matching via would read the wrong allocation's buffer count.
    #[test]
    fn a_source_without_a_switch_lets_the_first_switching_via_decide() {
        let comp = Component::Unit(DfirUnit::Lxlu);
        let other = Component::Unit(DfirUnit::L3lu);

        // The source IS this component but switches nothing, so the vias are walked.
        assert_eq!(
            buffering_or_streaming_mode(
                comp,
                TransferEnd {
                    unit: comp,
                    buffers: None,
                },
                &[
                    TransferEnd {
                        unit: other,
                        buffers: Some(Buffers::Count(2)),
                    },
                    TransferEnd {
                        unit: comp,
                        buffers: None,
                    },
                    TransferEnd {
                        unit: comp,
                        buffers: Some(Buffers::Streaming),
                    },
                ],
            ),
            BufferingMode::Streaming
        );

        // A source that does switch answers, and `-1` there is streaming too.
        assert_eq!(
            buffering_or_streaming_mode(
                comp,
                TransferEnd {
                    unit: comp,
                    buffers: Some(Buffers::Count(2)),
                },
                &[TransferEnd {
                    unit: comp,
                    buffers: Some(Buffers::Streaming),
                }],
            ),
            BufferingMode::Buffering
        );

        // Nothing on this component at all.
        assert_eq!(
            buffering_or_streaming_mode(
                comp,
                TransferEnd {
                    unit: other,
                    buffers: Some(Buffers::Streaming),
                },
                &[TransferEnd {
                    unit: other,
                    buffers: Some(Buffers::Streaming),
                }],
            ),
            BufferingMode::Neither
        );
    }

    /// 🎯 065/110 — ⛔ A DIM NO STRIDE MENTIONS STILL CONSUMES A DIM AND AN OPERAND.
    ///
    /// Dim 1 here is unstrided: it contributes the result `0`, a third map dim, and a zero constant in
    /// operand position 1 — so dim 2's own stride is `d2`, not `d1`. Dropping the placeholder would
    /// renumber it and read the wrong induction variable.
    #[test]
    fn an_unstrided_dim_still_takes_a_map_dim_and_a_zero_operand() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let strided = construct_base_address(
            &mut vals,
            &mut ops,
            3,
            &[
                LoopStride {
                    size_idx: 0,
                    elem_offset: 4,
                    iv: Some(Val(80)),
                },
                LoopStride {
                    size_idx: 0,
                    elem_offset: 7,
                    iv: None,
                },
                LoopStride {
                    size_idx: 2,
                    elem_offset: 1,
                    iv: Some(Val(81)),
                },
            ],
            AddressForm::of(false, false),
        );

        assert_eq!(
            strided,
            BaseAddress {
                map: AffineMap {
                    dims: 3,
                    syms: 0,
                    results: vec![
                        AffineExpr::dim(0).times(4).plus(AffineExpr::Const(7)),
                        AffineExpr::Const(0),
                        AffineExpr::dim(2),
                    ],
                },
                args: vec![Val(80), Val(0), Val(81)],
            }
        );
        assert_eq!(
            ops,
            vec![DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 0,
            })]
        );

        // ⛔ SCALEREG WINS OVER A BYPASS, and it is `ndims` of each where a bypass is zero dims.
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let scale = construct_base_address(&mut vals, &mut ops, 2, &[], AddressForm::of(true, true));
        assert_eq!(
            scale,
            BaseAddress {
                map: AffineMap::constants(2, &[0, 0]),
                args: vec![Val(0), Val(1)],
            }
        );

        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let bypass =
            construct_base_address(&mut vals, &mut ops, 2, &[], AddressForm::of(true, false));
        assert_eq!(
            bypass,
            BaseAddress {
                map: AffineMap::constants(0, &[0]),
                args: Vec::new(),
            }
        );
        assert!(ops.is_empty());
    }
}
