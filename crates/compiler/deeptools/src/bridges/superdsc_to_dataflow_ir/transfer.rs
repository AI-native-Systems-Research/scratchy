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

use super::compute::{VectorWidth, type_from_format};
use super::control_flow::PrimaryDim;
use super::dsc_lowering::{
    BitstreamValues, Component, Handles, constant_index, mlir_type_from_dsc_data_format,
    uniformized_folded_constant_bitstream,
};
use crate::arch::Elements;
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::arith::{CmpIPredicate, IntBinary};
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, arith, dataflow, defining_op, scf, vectorchain,
};
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, Constraint, ElemType, GenericComp, IntegerSet, MemRef, ScalarTy,
    TensorCategory, Vector,
};
use crate::units::{Core, Corelet};

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
        (Some(AffineExpr::Const(lhs)), AffineExpr::Const(rhs)) => {
            Some(lhs.checked_add(rhs).map_or_else(
                || AffineExpr::Const(lhs).plus(AffineExpr::Const(rhs)),
                AffineExpr::Const,
            ))
        }
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
            map: AffineMap::constants(u32::try_from(ndims).unwrap_or(u32::MAX), &vec![0; ndims]),
            args: (0..ndims).map(|_| constant_index(vals, ops, 0)).collect(),
        },
        AddressForm::Bypass => BaseAddress {
            map: AffineMap::constants(0, &[0]),
            args: Vec::new(),
        },
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 034/110 — THE VIEW A TRANSFER ADDRESSES THROUGH
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE VIEW ENTRY 034 EMITTED, PLUS THE TWO FACTS ITS CALLERS READ BACK OFF THE OP
/// (`view.getLayoutMap().getNumDims()` at `SNTransferLowering.cpp:436`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalMemoryView {
    /// The `get_logical_memory_view` result.
    pub result: Val,
    /// `view.getLayoutMap()`.
    pub layout: AffineMap,
    /// `view.getType()`.
    pub ty: MemRef,
}

/// Replaces: e034_constructLogicalMemoryViewOp
///
/// THE `dataflow.get_logical_memory_view` A TRANSFER ADDRESSES THROUGH — the view sizes as extents,
/// linearised with the FIRST dim fastest (`SNTransferLowering.cpp:94-125`).
///
/// ⛔⛔ THE STRIDE OF `d0` IS 1 AND EVERY LATER DIM IS SLOWER, because `size` multiplies AFTER dim
/// `i`'s term (`:1204` of the extract): a `(4, 8)` view is `(d0, d1) -> (d1 * 4 + d0)`, NOT the
/// `(d0 * 8 + d1)` that [`AffineMap::linear`]'s ascending stride list builds from the same extents.
///
/// ⛔ THE NEW TERM GOES ON THE **LEFT** OF THE ACCUMULATOR, so the printed sum descends and nests to
/// the right: `d2 * 32 + (d1 * 4 + d0)`.
///
/// ⛔ `bypass_viewsizes` KEEPS THE EXTENTS AND REPLACES ONLY THE LAYOUT (`:1210`) — a rank-`n` memref
/// carrying the rank-1 identity map, which is what `:436` then counts.
///
/// ⚠️ [`None`] IS AN EXTENT THAT IS NOT A COUNT: the reference's `int size` accumulator overflows
/// where this one stops, and a negative `size_` is a dynamic-extent sentinel to `MemRefType`.
#[must_use]
pub fn logical_memory_view(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    view_sizes: &[ViewSize],
    storage_unit: Val,
    start_address: Val,
    elem: ElemType,
    bypass_view_sizes: bool,
) -> Option<LogicalMemoryView> {
    let mut stride: i64 = 1;
    let mut extents = Vec::with_capacity(view_sizes.len());
    let mut layout_expr = AffineExpr::Const(0);
    for (i, view) in view_sizes.iter().enumerate() {
        let dim = AffineExpr::dim(u32::try_from(i).ok()?);
        // `getAffineBinaryOpExpr(Mul, getAffineConstantExpr(size), dim)` — `simplifyMul` moves the
        // constant to the right and folds `* 1` and `* 0`.
        let term = match stride {
            0 => AffineExpr::Const(0),
            1 => dim,
            _ => dim.times(stride),
        };
        // `getAffineBinaryOpExpr(Add, mul_expr, layout_expr)`, whose only fold here is `x + 0`.
        layout_expr = match (term, layout_expr) {
            (AffineExpr::Const(0), acc) => acc,
            (new, AffineExpr::Const(0)) => new,
            (new, acc) => new.plus(acc),
        };
        stride = stride.checked_mul(view.size)?;
        extents.push(u64::try_from(view.size).ok()?);
    }

    let layout = if bypass_view_sizes {
        // `AffineMap::getMultiDimIdentityMap(1, ..)`.
        AffineMap::identity(1)
    } else {
        AffineMap {
            dims: u32::try_from(view_sizes.len()).ok()?,
            syms: 0,
            results: vec![layout_expr],
        }
    };
    let ty = MemRef {
        shape: extents,
        elem,
    };
    let result = vals.mint();
    ops.push(DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
        result,
        from: storage_unit,
        start: start_address,
        layout: layout.clone(),
        ty: ty.clone(),
    }));
    Some(LogicalMemoryView { result, layout, ty })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 066 + 069/110 — THE ONE FACT BOTH FUNCTIONS READ OFF AN ALREADY-EMITTED LOOP
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT AN EMITTED LOOP'S UPPER BOUND IS WORTH TO A TIME SET — a literal, or nothing usable.
///
/// ⛔⛔ THE TWO LOOP KINDS ANSWER THE SAME QUESTION THROUGH TWO DIFFERENT MECHANISMS, and both
/// `constructTimeSet` (entry 066) and `areEpiloguesInLoops` (entry 069) run the identical
/// four-branch dispatch to ask it (`SNTransferLowering.cpp:284-299` and `:1468-1483`). An
/// `affine.for`'s bounds are ATTRIBUTES, so the question is `hasConstantUpperBound()`; an
/// `scf.for`'s are OPERANDS, so it is `dyn_cast<arith::ConstantIndexOp>(getOperand(1))`. One enum
/// for the answer means the dispatch is written once — see [`loop_upper_bound`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopBound {
    /// The bound is a literal: `hasConstantUpperBound()`, or the `scf` operand's defining constant.
    Constant(i64),
    /// The bound is an SSA value no `arith.constant` defines — a symbol, a `select`, an `apply`.
    ///
    /// ⭐ THIS IS WHAT AN EPILOGUE **IS** AT THIS RUNG. Entry 069's whole job is to answer "does any
    /// of these loops have a bound that is not a literal", and a non-literal bound is a loop whose
    /// last iteration may be short — the epilogue.
    Dynamic,
}

/// `%v.getDefiningOp<mlir::arith::ConstantIndexOp>()` — the literal behind an `index` value.
///
/// ⭐ A LOCAL COPY OF ONE LINE, DELIBERATELY. Bridge 2 has the same walk in
/// `dataflow_ir_to_sentient::tf_utils`, but it is `pub(super)` there and reaching across two bridges
/// for three lines would couple them; `dsc_lowering`'s same-named helper mints a constant rather than
/// reading one back.
fn defining_constant_index(val: Val, scope: &[DfirOp]) -> Option<i64> {
    match defining_op(val, scope)? {
        DfirOp::Arith(arith::Op::Constant { value, .. }) => Some(*value),
        _ => None,
    }
}

/// THE UPPER BOUND OF ONE LOOP THIS BRIDGE ALREADY EMITTED, AND WHETHER IT IS A LITERAL.
///
/// `None` is the reference's *"Unknown for-loops"* — an op that is neither an `affine.for` nor an
/// `scf.for`, which entry 069 reaches by `llvm_unreachable` (`:1481`).
///
/// # ⛔⛔ THE SAME UNHANDLED KIND IS A **SILENT** CORRUPTION IN ENTRY 066
///
/// Entry 066's dispatch has NO `else` at all (`:284-299`): a loop of a third kind pushes no
/// expression, while `eq_flags` was already sized to `2 * ndims` at `:270`. The vector of
/// expressions and the vector of flags then disagree in length, and `IntegerSet::get` is handed a
/// pair MLIR asserts on — `assert(eqFlags.size() == constraints.size())`. Where entry 069 aborts
/// loudly on the same input, entry 066 builds a malformed set. Answering `None` here makes the two
/// callers reach the SAME outcome, and makes the length agreement in [`time_set`] structural rather
/// than a coincidence of two counters.
///
/// ⛔ `getOperand(1)` IS THE UPPER BOUND, not the lower. An `scf.for`'s operand list is
/// `lb, ub, step, inits...` (`ForOp`'s ODS order), so index 1 is `$upperBound`, which is
/// [`scf::Op::For::hi`].
///
/// ⛔ AND A LOOP THE MAP DID NOT HOLD IS ALSO `None`. The reference reaches this through
/// `getMLIRLoopFromLoopNode`, which answers `nullptr` for a loop node it never recorded, and
/// `llvm::dyn_cast` on a null `Operation *` is undefined — see
/// [`super::dsc_lowering::mlir_loop_from_loop_node`], whose `Option` the caller resolves before it
/// can call this at all.
#[must_use]
pub fn loop_upper_bound(loop_op: &DfirOp, scope: &[DfirOp]) -> Option<LoopBound> {
    match loop_op {
        // `affine_for.hasConstantUpperBound()` — an attribute, so no scope walk.
        DfirOp::Affine(affine::Op::For { hi, .. }) => Some(match hi {
            affine::Bound::Const(ub) => LoopBound::Constant(*ub),
            affine::Bound::Val(_) => LoopBound::Dynamic,
        }),
        // `dyn_cast<mlir::arith::ConstantIndexOp>(scf_for->getOperand(1).getDefiningOp())`.
        DfirOp::Scf(scf::Op::For { hi, .. }) => Some(match defining_constant_index(*hi, scope) {
            Some(ub) => LoopBound::Constant(ub),
            None => LoopBound::Dynamic,
        }),
        _ => None,
    }
}

/// Replaces: e066_constructTimeSet
///
/// **066/110** `SNTransferLowering::constructTimeSet` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:260` (47L).
///
/// THE SET OF TIME STEPS A TRANSFER OCCUPIES: one dimension per unit-view loop, each spanning
/// `0 ..= ub-1` of the MLIR loop that walks it.
///
/// # ⛔⛔ INEQUALITY-ONLY, AND THAT IS NOT THE SAME SET SPELLING AS [`IntegerSet::from_sizes`]
///
/// `num_eq = 0` at `:265`, so `total_constraints == num_ineq` and the second `std::fill` at `:273`
/// fills an EMPTY range: every flag is `false`. A loop with an upper bound of one therefore comes
/// out as the PAIR `d0 >= 0, -d0 + 0 >= 0` — not as `d0 == 0`, which is what
/// `buildIntegerSetFromSizes` writes for a size of one. The two describe the same points and print
/// differently, and this is the function the reference uses for a time axis, so reusing
/// `from_sizes` here would change the emitted `affine_set<>` text on every unit-extent time step.
///
/// ⛔ THE PAIR IS PUSHED PER DIMENSION, WHICH IS WHY THE FLAG COUNT CANNOT DRIFT. The reference
/// sizes `eq_flags` from `2 * ndims` up front and then pushes expressions in a loop that may push
/// one, two, or none (see [`loop_upper_bound`]); here both constraints of a dimension are appended
/// together from one `Constant`, so `constraints.len() == 2 * dims` holds by construction.
///
/// ⭐ `num_symbols++` AT `:288` IS A DEAD STORE. It is incremented and the very next statement is
/// `return LogicalResult::failure()`, and the surviving path passes a hard-coded `0` to
/// `IntegerSet::get` (`:303`) — with the `args.push_back` that would have consumed a symbol
/// commented out on the line between. So the set this builds never takes a symbol, and
/// [`IntegerSet::symbols`] is `0` on every answer.
///
/// ⛔ A DYNAMIC BOUND IS THE REFUSAL, on both loop kinds (`:289` and `:298`). `None` here is the
/// reference's `LogicalResult::failure()`, and the caller's fallback is a different lowering, not a
/// looser set.
#[must_use]
pub fn time_set(loops: &[LoopBound]) -> Option<IntegerSet> {
    let mut constraints = Vec::with_capacity(2 * loops.len());
    for (i, bound) in loops.iter().enumerate() {
        // `if (affine_for.hasConstantUpperBound())` .. `else return failure()`.
        let LoopBound::Constant(ub) = *bound else {
            return None;
        };
        let id = AffineExpr::dim(u32::try_from(i).ok()?);
        // `exprs.push_back(id);  // id >= 0`
        constraints.push(Constraint {
            expr: id.clone(),
            is_equality: false,
        });
        // `exprs.push_back(affine_for.getConstantUpperBound() - id - 1);`, which normalises to
        // `-id + (ub - 1) >= 0` — the same spelling `from_sizes` uses for its upper half.
        constraints.push(Constraint {
            expr: id.times(-1).plus(AffineExpr::Const(ub - 1)),
            is_equality: false,
        });
    }
    Some(IntegerSet {
        // `IntegerSet::get(ndims, 0, exprs, eq_flags)` — `ndims = loops.size()`.
        dims: u32::try_from(loops.len()).ok()?,
        symbols: 0,
        constraints,
    })
}

/// Replaces: e069_areEpiloguesInLoops
///
/// **069/110** `SNTransferLowering::areEpiloguesInLoops` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1464` (24L).
///
/// DOES ANY OF THESE LOOPS HAVE A BOUND THAT IS NOT A LITERAL — i.e. does the transfer have an
/// epilogue the lowering must build a second, shorter shape for.
///
/// ⛔⛔ THE WHOLE FUNCTION IS [`loop_upper_bound`] TWENTY-FOUR LINES LONG. The reference's
/// `continue` / `return true` arms are exactly "constant" and "not constant" over the same
/// four-branch dispatch entry 066 runs, and its third arm — `llvm_unreachable("Unknown
/// for-loops")` (`:1481`) — is that function's `None`, which the caller has already had to resolve
/// to build this slice. Once the answer is a value, this is a predicate over it.
///
/// ⭐ AN EMPTY LOOP LIST IS `false`, which is the reference's fall-through at `:1486`: no loops, no
/// epilogue.
#[must_use]
pub fn epilogues_in_loops(loops: &[LoopBound]) -> bool {
    loops.contains(&LoopBound::Dynamic)
}

/// Replaces: e035_constructTimeOrder
///
/// THE TIME ORDER OF A COMPOSITE TRANSFER — the REVERSAL permutation over its loops,
/// `(d0, .., dn) -> (dn, .., d0)` (`SNTransferLowering.cpp:206-217`).
///
/// ⛔⛔ IT RETURNS `LogicalResult::failure()` UNCONDITIONALLY (`:216`) AND HAS NO CALLER IN THE TREE.
/// The map it writes to its out-parameter is therefore the whole of its behaviour; answering [`None`]
/// to mirror the failure would leave it with none at all.
///
/// ⚠️ AN EMPTY LOOP LIST IS [`None`] — `AffineMap::getPermutationMap({})` asserts on an empty
/// permutation vector (`mlir/lib/IR/AffineMap.cpp:262`), so a zero-loop transfer has no time order
/// rather than the zero-dim map. The loop list is read for its LENGTH only.
#[must_use]
pub fn time_order(composite_loops: usize) -> Option<AffineMap> {
    if composite_loops == 0 {
        return None;
    }
    let dims = u32::try_from(composite_loops).ok()?;
    Some(AffineMap {
        // `getMultiDimMapWithTargets(*max_element(permutation) + 1, ..)`.
        dims,
        syms: 0,
        results: (0..dims).rev().map(AffineExpr::dim).collect(),
    })
}

/// ONE ENTRY OF `TransferNode::UnitView::LoopInfo` AS THE TIME ADDRESS MAP READS IT
/// (`dsc/dsc2.h:500-505`).
///
/// ⛔ NOT [`LoopStride`], WHOSE `size_idx` CANNOT BE `-1` AND WHOSE `iv` IS A VALUE: here the loop's
/// dimension is POSITIONAL — `dims[i]`, the map's own `d<i>` — and `-1` is a loop that addresses
/// nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeLoop {
    /// `sizeIdx_` — [`None`] is the reference's `-1`.
    pub size_idx: Option<usize>,
    /// `elemOffset_`, in elements.
    pub elem_offset: i64,
}

/// Replaces: e036_constructTimeAddressMap
///
/// THE ADDRESS ONE TIME STEP LANDS AT — one result per OUTPUT dim, summing each composite loop's own
/// `d<i>` times its element offset (`SNTransferLowering.cpp:224-252`).
///
/// ⛔⛔ THE DIMENSION IS THE LOOP'S POSITION, NOT ITS `sizeIdx_` (`:1261` of the extract):
/// `base_address_exprs[idx] + elemOffset * dims[i]` reads `i` for the variable and `idx` only for
/// which result it lands in, so two loops addressing one dim accumulate into one result.
///
/// ⛔ A RESULT NO LOOP ADDRESSES STAYS THE CONSTANT `0` (`:1250-1252`) — the map's arity is
/// `output_dims`, never the loop count.
///
/// ⚠️ [`None`] IS THE REFERENCE'S OUT-OF-BOUNDS READ: `dims[i]` for `i >= input_dims` and
/// `base_address_exprs[idx]` for `idx >= output_dims` are both unchecked there.
#[must_use]
pub fn time_address_map(
    input_dims: u32,
    output_dims: usize,
    composite_loops: &[CompositeLoop],
) -> Option<AffineMap> {
    let mut results = vec![AffineExpr::Const(0); output_dims];
    for (i, walk) in composite_loops.iter().enumerate() {
        let Some(idx) = walk.size_idx else {
            continue;
        };
        let dim = AffineExpr::dim(u32::try_from(i).ok().filter(|at| *at < input_dims)?);
        let term = match walk.elem_offset {
            0 => AffineExpr::Const(0),
            1 => dim,
            offset => dim.times(offset),
        };
        let slot = results.get_mut(idx)?;
        // `expr + term`, with `simplifyAdd`'s constant-lhs canonicalisation: `0 + t` is `t`.
        *slot = match (slot.clone(), term) {
            (AffineExpr::Const(0), new) => new,
            (acc, AffineExpr::Const(0)) => acc,
            (acc, new) => acc.plus(new),
        };
    }
    Some(AffineMap {
        // `AffineMap::get(input_dims, 0, base_address_exprs, context)`.
        dims: input_dims,
        syms: 0,
        results,
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 067/110 — WHICH ELEMENTS OF THE VIEW ONE UNIT-TIME TRANSFER TOUCHES
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE CHUNKED DIMENSION OF A UNIT-TIME TRANSFER — `TransferNode::SizeAndIndex` (`dsc/dsc2.h:820`).
///
/// ⛔ THE `-1`s ARE ABSENCE, NOT INDICES. `srcSizeIdx_` and `dstSizeIdx_` both default to `-1`
/// (`:822`) and the reference compares them against a `dim_id` that counts from zero, so the
/// sentinel works only because it can never be a valid position. An [`Option`] says that outright,
/// and makes "this chunk is not matched on the side being asked about" a case the match has to
/// handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkDim {
    /// `sizeDim_.size_` — how many elements of this dimension one chunk covers.
    pub size: i64,
    /// `srcSizeIdx_` — which position of the SOURCE view this chunk is a chunk of.
    pub src_index: Option<u32>,
    /// `dstSizeIdx_` — the same for the DESTINATION view.
    pub dst_index: Option<u32>,
}

/// WHICH SIDE'S VIEW POSITIONS A CHUNK IS MATCHED ON — the reference's `bool is_load`.
///
/// ⭐ THE FLAG IS READ FOUR TIMES IN ONE FUNCTION (`:340`, `:344`, `:355`, `:359`) and selects the
/// same field of the same record every time, so it is one operation on the record rather than a
/// boolean the body keeps re-testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSide {
    /// `is_load == true` — positions come from `srcSizeIdx_`.
    Load,
    /// `is_load == false` — positions come from `dstSizeIdx_`.
    Store,
}

impl TransferSide {
    /// The view position this chunk occupies on this side, if any.
    #[must_use]
    pub const fn index_of(self, chunk: &ChunkDim) -> Option<u32> {
        match self {
            TransferSide::Load => chunk.src_index,
            TransferSide::Store => chunk.dst_index,
        }
    }
}

/// Replaces: e067_constructLoadOrStoreSet
///
/// **067/110** `SNTransferLowering::constructLoadOrStoreSet` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:314` (90L).
///
/// THE SET OF VIEW ELEMENTS ONE UNIT-TIME TRANSFER TOUCHES: every view position is pinned to zero
/// unless a chunk size or a chunk stride claims it, and a claimed position spans its chunk count.
///
/// # ⛔⛔ THE CONSTRAINTS ARE **GROUPED**, INEQUALITIES FIRST — NOT INTERLEAVED PER DIMENSION
///
/// The body accumulates `ineq_exprs` and `eq_exprs` separately and splices them in that order at
/// `:392-393`, then fills the flags to match (`:395-397`). So for view sizes claimed as
/// `[span, pinned, span]` the reference emits
/// `(d0 >= 0, -d0 + n >= 0, d2 >= 0, -d2 + m >= 0, d1 == 0)`, while
/// [`IntegerSet::from_sizes`] would emit the same five constraints in DIMENSION order. Same point
/// set, different printed `affine_set<>`, and this is the one the reference writes for a load or a
/// store — so `from_sizes` is not reusable here either. See [`time_set`], which is inequality-only
/// for the same reason.
///
/// # ⛔ `view_sizes` IS READ FOR ITS **LENGTH** AND NOTHING ELSE
///
/// The body walks `dim_id` over `0 .. view_sizes.size()` and never reads `view_sizes[dim_id]` — not
/// its `dim_`, not its `size_`. The extents come from the chunk records; the view supplies only the
/// RANK of the set (`:333`, `:399`). Taking a `&[Size]` here would suggest the extents matter and
/// invite a caller to fix a mismatch that cannot exist.
///
/// # ⛔ A STRIDED DIMENSION IS MEASURED IN CHUNKS, NOT IN ELEMENTS
///
/// `size = (idx_in_chunk_stride != -1) ? num_chunk_strides : ..sizeDim_.size_` (`:375-379`): a
/// dimension the transfer STRIDES over spans the number of chunks, and the stride record's own
/// `size_` is never read. Reading the stride's extent instead would give the set the element count
/// of one chunk where the reference gives it the count of chunks.
///
/// # ⛔ AND THE TWO REFUSALS BECOME TWO DIFFERENT SHAPES, BECAUSE THEY ARE DIFFERENT KINDS OF FACT
///
/// *"Currently supports chunk strides with striding over a single dimension"* (`:325-331`) is a
/// property of the ARGUMENT — so the argument is an [`Option<&ChunkDim>`] and a list of two cannot
/// be handed in. `!empty() && size() > 1` is then the same predicate as `size() > 1`, and the
/// reference's first conjunct is redundant.
///
/// *"A dimension cannot be present in both chunk size and stride"* (`:368-372`) is a property of
/// the two lists TOGETHER at one position, discoverable only by walking them, so it stays a typed
/// absence in the result.
#[must_use]
pub fn load_or_store_set(
    chunk_sizes: &[ChunkDim],
    chunk_stride: Option<&ChunkDim>,
    view_rank: u32,
    side: TransferSide,
    num_chunk_strides: i64,
) -> Option<IntegerSet> {
    let mut ineq_exprs = Vec::new();
    let mut eq_exprs = Vec::new();

    for dim_id in 0..view_rank {
        let id = AffineExpr::dim(dim_id);

        // `find index of dim_id in unit time transfer chunk size` — FIRST match wins (`:339-350`).
        let in_chunk_size = chunk_sizes
            .iter()
            .find(|chunk| side.index_of(chunk) == Some(dim_id));
        // The same walk over a stride list that cannot hold more than one record.
        let in_chunk_stride = chunk_stride.filter(|chunk| side.index_of(chunk) == Some(dim_id));

        // `if (idx_in_chunk_size != -1 && idx_in_chunk_stride != -1)` — the second refusal.
        if in_chunk_size.is_some() && in_chunk_stride.is_some() {
            return None;
        }

        let size = match (in_chunk_stride, in_chunk_size) {
            // ⛔ THE STRIDE ARM MEASURES CHUNKS. See above.
            (Some(_), _) => num_chunk_strides,
            (None, Some(chunk)) => chunk.size,
            // `} else { eq_exprs.push_back(id); }` — a position no chunk claims is pinned.
            (None, None) => {
                eq_exprs.push(Constraint {
                    expr: id,
                    is_equality: true,
                });
                continue;
            }
        };

        if size > 1 {
            // `id >= 0` and `size - id - 1 >= 0`.
            ineq_exprs.push(Constraint {
                expr: id.clone(),
                is_equality: false,
            });
            ineq_exprs.push(Constraint {
                expr: id.times(-1).plus(AffineExpr::Const(size - 1)),
                is_equality: false,
            });
        } else {
            // `id == 0` — a claimed dimension of extent one is still a point.
            eq_exprs.push(Constraint {
                expr: id,
                is_equality: true,
            });
        }
    }

    // `exprs.insert(.., ineq_exprs)` then `exprs.insert(.., eq_exprs)`.
    ineq_exprs.extend(eq_exprs);
    Some(IntegerSet {
        // `IntegerSet::get(view_sizes.size(), 0, exprs, eq_flags)`.
        dims: view_rank,
        symbols: 0,
        constraints: ineq_exprs,
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 037/110 — THE NEAREST LOOP WALKING A DIM AT THE STAGE THIS TRANSFER NEEDS
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE FOUR `dataStageDimToVal_compView_st(dim, comp_)` READINGS ONE LOOP NODE ANSWERS FOR ONE DIM —
/// its `denId_` stage's start and end slice, and its `numId_` stage's.
///
/// ⛔ [`None`] IS THE REFERENCE'S `-1`, and on the QUERYING node it means *matches anything*:
/// `is_any_of(ss_val, -1, p_den_ss_val)` (`SNTransferLowering.cpp:694`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimStageVals {
    /// `ss_` of `denId_`.
    pub den_ss: Option<i32>,
    /// `el_` of `denId_`.
    pub den_el: Option<i32>,
    /// `ss_` of `numId_`.
    pub num_ss: Option<i32>,
    /// `el_` of `numId_`.
    pub num_el: Option<i32>,
}

/// ONE NODE OF THE `getOwnerLoop()` CHAIN, as far as entry 037 reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimChainNode<'a, N> {
    /// The node itself — what the reference returns.
    pub node: &'a N,
    /// `dims_`, where MEMBERSHIP is the whole test — unlike
    /// [`super::control_flow::parent_loop`]'s slice equality.
    pub dims: &'a [PrimaryDim],
    /// The four readings, for the dim being asked about.
    pub stages: DimStageVals,
}

/// Replaces: e037_getImmediateParentWithMatchingDim
///
/// THE NEAREST NODE THAT WALKS `dim` AT THE QUERY'S OWN `denId_` SLICE AND WHOSE `numId_` STAGE DOES
/// NOT SPLIT (`SNTransferLowering.cpp:670-702`).
///
/// ⛔⛔ THE QUERY IS THE CHAIN HEAD'S OWN `denId_` READING (`:676-680`) — which is why it is not a
/// separate argument. `auto *parent = node` starts the walk AT the node, so a node that walks the dim
/// itself is its own answer and the name *parent* is not a claim about the result.
///
/// ⛔ THE `numId_` TEST IS ON THE **CANDIDATE**, NOT THE QUERY (`:695`): `p_num_ss == p_num_el` rejects
/// an ancestor whose numerator stage differs between start and end, however well its `denId_` matches.
///
/// The chain is mechanism — `getOwnerLoop()`'s null root — so `chain` IS it, NEAREST FIRST, and empty
/// for the reference's null node (`:672`).
#[must_use]
pub fn immediate_parent_with_matching_dim<'c, N>(
    chain: &'c [DimChainNode<'c, N>],
    dim: PrimaryDim,
) -> Option<&'c N> {
    let query = chain.first()?.stages;
    chain
        .iter()
        .find(|candidate| {
            candidate.dims.contains(&dim)
                && (query.den_ss.is_none() || query.den_ss == candidate.stages.den_ss)
                && (query.den_el.is_none() || query.den_el == candidate.stages.den_el)
                && candidate.stages.num_ss == candidate.stages.num_el
        })
        .map(|candidate| candidate.node)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 068/110 — THE LOOPS A CONTIGUOUS TRANSFER NEEDS THAT NO SCHEDULE LOOP SUPPLIES
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DIMENSION OF A UNIT VIEW — `ScheduleNode::Size` (`dsc/dsc2.h:486`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewSize {
    /// `dim_`.
    pub dim: PrimaryDim,
    /// `size_`.
    pub size: i64,
}

/// HOW MANY CONTIGUOUS TRANSFERS A DIMENSION COSTS IN THE STEADY STATE AND IN THE EPILOGUE.
///
/// ⛔⛔ THE TWO ARE A PAIR EVERYWHERE THIS LOWERING TOUCHES THEM, and whether they are EQUAL is what
/// decides which kind of loop gets built: `sticks_src_ss`/`sticks_src_el` and
/// `src_sticks_ss_per_dim`/`src_sticks_el_per_dim` are four members read, divided, compared and
/// decremented in lockstep (`SNTransferLowering.hpp:63-66`). Two parallel maps let one be updated
/// and the other forgotten; one record of two fields cannot be half-updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StickCounts {
    /// `..._ss` — the steady state.
    pub steady: i64,
    /// `..._el` — the epilogue.
    pub epilogue: i64,
}

/// `transfer_->replicationFactor_` (`dsc/dsc2.h:834`) — AND IT IS A DIVISOR.
///
/// ⛔ THE `DT_CHECK` THE REFERENCE DOES NOT HAVE IS THIS CONSTRUCTOR. `record_ss->second / factor`
/// at `:728-729` divides by the member directly; the field defaults to `1` but nothing in the type
/// stops a schedule from leaving it at zero, and the division is then a fault inside a lowering.
/// Refusing the zero once, here, is the only place it can be refused before it is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Replication(i64);

impl Replication {
    /// `replicationFactor_`, which must be positive to be a divisor.
    #[must_use]
    pub const fn checked(factor: i64) -> Option<Replication> {
        if factor <= 0 {
            return None;
        }
        Some(Replication(factor))
    }

    /// The factor itself.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// THE CONTIGUOUS-TRANSFER COUNTS A TRANSFER CARRIES — `src_sticks_ss_per_dim` and
/// `src_sticks_el_per_dim` zipped, plus the whole-transfer `sticks_src_ss`/`sticks_src_el`.
///
/// ⛔ THE PER-DIM RECORDS ARE **MUTATED IN PLACE** BY ENTRY 068, twice each: `OUT` is divided by the
/// replication factor (`:728-729`) and every dim that gets a loop has the view's own extent
/// subtracted from it (`:748-749`). Those writes are the function's second output and a later
/// lowering reads them back, so the argument is `&mut` rather than a copy.
///
/// ⭐ A `Vec` OF PAIRS RATHER THAN A MAP, because the reference's `std::unordered_map` has no order
/// and every access here is a keyed lookup — see [`super::compute::dictionary_order`], which exists
/// because the reference's map order is not reproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousSticks {
    per_dim: Vec<(PrimaryDim, StickCounts)>,
    whole: StickCounts,
}

impl ContiguousSticks {
    /// The four members, as one record.
    #[must_use]
    pub fn new(whole: StickCounts, per_dim: Vec<(PrimaryDim, StickCounts)>) -> ContiguousSticks {
        ContiguousSticks { per_dim, whole }
    }

    /// `src_sticks_ss_per_dim.find(dim)` and its `_el` twin, as one lookup.
    #[must_use]
    pub fn per_dim(&self, dim: PrimaryDim) -> Option<StickCounts> {
        self.per_dim
            .iter()
            .find(|(walked, _)| *walked == dim)
            .map(|(_, counts)| *counts)
    }

    /// `sticks_src_ss` and `sticks_src_el`.
    #[must_use]
    pub const fn whole(&self) -> StickCounts {
        self.whole
    }

    fn per_dim_mut(&mut self, dim: PrimaryDim) -> Option<&mut StickCounts> {
        self.per_dim
            .iter_mut()
            .find(|(walked, _)| *walked == dim)
            .map(|(_, counts)| counts)
    }
}

/// Replaces: e038_areEpiloguesInTransferSizes
///
/// DOES ANY DIM'S STEADY-STATE STICK COUNT DISAGREE WITH ITS EPILOGUE'S — the per-dim twin of
/// [`epilogues_in_loops`] (`SNTransferLowering.cpp:1491-1502`).
///
/// ⛔⛔ A DIM WHERE NEITHER COUNT EXCEEDS ONE CANNOT REPORT AN EPILOGUE: the outer test is
/// `ss_val > 1 || el_val > 1` (`:1497`), so `(1, 0)` — one steady stick and none in the epilogue — is
/// skipped before the inequality is asked. Testing `ss != el` alone would answer `true` there.
///
/// ⚠️ THE WHOLE-TRANSFER PAIR IS NOT READ, only `src_sticks_ss_per_dim` and its `_el` twin.
#[must_use]
pub fn epilogues_in_transfer_sizes(sticks: &ContiguousSticks) -> bool {
    sticks.per_dim.iter().any(|(_, counts)| {
        (counts.steady > 1 || counts.epilogue > 1) && counts.steady != counts.epilogue
    })
}

/// ONE LOOP ENTRY 068 DECIDED TO BUILD — `ScheduleNode::UnitView::LoopInfo` (`dsc/dsc2.h:500`)
/// paired with the `outer_loop_sizes` entry that was pushed beside it.
///
/// ⛔ THE TWO VECTORS ARE INDEXED BY THE SAME `i` IN THE EMISSION LOOP (`:774-776` reads
/// `outer_loop_sizes[i]` and `implicit_loops[i]` in one breath), so they are one list here. Two
/// lists that must stay the same length is the shape that lets a `push` be forgotten.
///
/// ⭐ `elemOffset_` IS NOT A FIELD BECAUSE IT IS THE CONSTANT `1`. All three sites that build a
/// `LoopInfo` in this function set `loop_info.elemOffset_ = 1` (`:737`, `:768`) and nothing here
/// varies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplicitLoop {
    /// `sizeIdx_` — the view position this loop walks. `None` is the reference's `-1`, which only the
    /// dummy loop of the empty-view branch carries (`:766`).
    pub size_index: Option<usize>,
    /// `dim_`. `None` is `PrimaryDimTypesCount`, the unset default — again only the dummy loop, whose
    /// `LoopNode` is default-constructed with no dims at all (`:765`).
    pub dim: Option<PrimaryDim>,
    /// The `outer_loop_sizes` pair: the steady-state and epilogue trip counts, BEFORE the view's
    /// extent is subtracted from the record they came from (`:741` precedes `:748`).
    pub extents: StickCounts,
}

/// Replaces: e068_constructImplicitLoopsForContiguousTransfer
///
/// **068/110** `SNTransferLowering::constructImplicitLoopsForContiguousTransfer` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:707` (137L), first half (`:713-772`).
///
/// WHICH LOOPS A CONTIGUOUS TRANSFER NEEDS BEYOND THE ONES THE SCHEDULE ALREADY WALKS — the view
/// positions past the unit-time chunk that still carry more than one contiguous transfer.
///
/// # ⛔ `ctgs_transfer_sizes` IS A DEAD PARAMETER
///
/// The reference's fifth argument (`:711`, declared at `SNTransferLowering.hpp:207`) is never read
/// and never written in the whole 137-line body. Every count comes from and goes back to the MEMBERS
/// `src_sticks_ss_per_dim` / `src_sticks_el_per_dim` / `sticks_src_ss` / `sticks_src_el` — verified
/// by grepping the body for the name, which occurs exactly once, in the signature. Carrying it
/// forward would be a parameter every caller has to invent a value for.
///
/// # ⛔ `unit_time_transfer` IS READ FOR ITS **LENGTH** ONLY
///
/// `int unit_time_dims = unit_time_transfer.size();` (`:722`) is its single use: it is where the
/// walk over the view STARTS, because the positions below it are the ones the unit-time transfer
/// itself already covers. The records are never inspected.
///
/// # ⛔ `OUT` IS DIVIDED BY THE REPLICATION FACTOR, AND ONLY `OUT`
///
/// `if (view_sizes[i].dim_ == PrimaryDimTypes::OUT)` (`:727-730`) — a replicated transfer sends the
/// same output features to several destinations, so its own count of contiguous transfers is the
/// per-destination one. The division is integer and happens BEFORE the `> 1` test, so a dim whose
/// count divides down to one gets no loop at all: the reference's own comment is *"avoid unit size
/// loops or zero loops after substituition"* (`:732`).
///
/// # ⛔ THE `DT_ERROR` AND THE `DT_CHECK_MSG` ARE TYPED ABSENCES, NOT ABORTS
///
/// *"view dims and contigous transfer dim didn't match"* (`:753`) and *"Number of contiguous
/// transfers should be same in steady state/epilouge state"* (`:759-761`) are both `DT_ERROR` /
/// `DT_CHECK_MSG`, which is `DT_CHECK_MSG(false, ..)` and raises a `DtException`
/// (`util/dt_exception.hpp:121`) from inside a lowering. Here they are `None`, which the caller
/// resolves on the same path as the function's own `LogicalResult::failure()`.
///
/// ⛔ THE EMPTY-VIEW BRANCH BUILDS AT MOST ONE LOOP, and only when `sticks_src_ss > 1` — a whole
/// transfer that fits in one contiguous burst needs no loop, and one that does not is guarded to
/// have no epilogue, so the loop it gets can only ever be the constant-bound kind.
pub fn implicit_loops_for_contiguous_transfer(
    view_sizes: &[ViewSize],
    unit_time_dims: usize,
    sticks: &mut ContiguousSticks,
    replication: Replication,
) -> Option<Vec<ImplicitLoop>> {
    let mut implicit_loops = Vec::new();

    if view_sizes.is_empty() {
        // `DT_CHECK_MSG(sticks_src_ss == sticks_src_el, ..)`.
        let whole = sticks.whole;
        if whole.steady != whole.epilogue {
            return None;
        }
        if whole.steady > 1 {
            // `create outer dummy loop` — no view position and no dim.
            implicit_loops.push(ImplicitLoop {
                size_index: None,
                dim: None,
                extents: whole,
            });
        }
        return Some(implicit_loops);
    }

    // `for (int i = unit_time_dims; i < view_sizes.size(); i++)`.
    for (i, view) in view_sizes.iter().enumerate().skip(unit_time_dims) {
        // `src_sticks_ss_per_dim.find(view_sizes[i].dim_) == end()` is the `DT_ERROR`.
        let counts = sticks.per_dim_mut(view.dim)?;

        if view.dim == PrimaryDim::Out {
            counts.steady /= replication.get();
            counts.epilogue /= replication.get();
        }

        // `if (record_ss->second > 1 || record_el->second > 1)`.
        if counts.steady > 1 || counts.epilogue > 1 {
            implicit_loops.push(ImplicitLoop {
                size_index: Some(i),
                dim: Some(view.dim),
                extents: *counts,
            });
            // `record_ss->second -= view_sizes[i].size_;` and its `_el` twin.
            counts.steady -= view.size;
            counts.epilogue -= view.size;
        }
    }

    Some(implicit_loops)
}

/// THE PARENT LOOP OF A REPEATED DIM — `SNTransferLowering.cpp:785-793`.
///
/// # ⛔⛔ THE SAME WALK AS ENTRY 018, WITH THE OPPOSITE TIE-BREAK
///
/// `dims_[j] == dim` then `record.at(ndims - j - 1)` is character for character
/// [`super::control_flow::mlir_loop_from_sn_loop_node`] — the same mirrored index, for the same
/// reason (element 0 of the record is the OUTERMOST loop while `dims_[0]` is the INNERMOST dim). But
/// this copy has **no `break`**: it keeps assigning, so the LAST matching `j` wins. Entry 018 and
/// entry 028 `return` on the first.
///
/// For a loop node that names a dimension once the two agree. For a SPLIT BAND that names it twice,
/// entry 018 answers the innermost of the two loops and this answers the OUTERMOST — and the value
/// it feeds is the induction variable a `cmpi` compares against the band's last iteration, so the
/// difference is which of two nested loops decides that a transfer is in its epilogue. Recorded as
/// a divergence between two copies of one walk, not normalised away.
///
/// ⛔ `.at()` ON BOTH LOOKUPS IS A THROW IN THE REFERENCE. `dsc_loops_to_mlir_loops_map_.at(parent)`
/// throws for a parent never recorded, and `.at(ndims - j - 1)` throws when the record holds fewer
/// loops than the node has dims. Both are `None` here, which is also where the reference's
/// `DT_CHECK("parent loop in implicit loops cannot be empty" && parent_loop)` (`:795`) lands.
#[must_use]
pub fn outermost_mlir_loop_for_dim<'l, L>(
    dims: &[PrimaryDim],
    loops: &'l [L],
    dim: PrimaryDim,
) -> Option<&'l L> {
    let mut found = None;
    for (j, walked) in dims.iter().enumerate() {
        if *walked == dim {
            // `at(ndims - j - 1)`, with the reference's unchecked subtraction made checked.
            found = loops.get(dims.len().checked_sub(j + 1)?);
        }
    }
    found
}

/// THE NEST ENTRY 068 EMITS, AND THE LOOP HANDLES ITS CALLER HAS TO RECORD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitNest {
    /// The ops at the level the reference's builder started at: the outermost loop, with everything
    /// else inside it.
    pub ops: Vec<DfirOp>,
    /// `(*dsc_loops_to_mlir_loops_map_)[implicit_loops[i].loop_].push_back(loop)` (`:781`, `:838`) —
    /// the induction variable of the loop built for `implicit_loops[i]`, in that list's own order.
    ///
    /// ⭐ AN INDUCTION VARIABLE **IS** THE ISLAND'S HANDLE ON A LOOP IN THAT MAP:
    /// [`super::dsc_lowering::mlir_loop_from_loop_node`] instantiates entry 018's generic `L` at
    /// [`Val`] for exactly this map.
    pub ivs: Vec<Val>,
}

/// Replaces: e068_constructImplicitLoopsForContiguousTransfer
///
/// **068/110** `SNTransferLowering::constructImplicitLoopsForContiguousTransfer` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:707` (137L), second half (`:774-840`).
///
/// THE LOOPS THEMSELVES, NESTED. `body` is what the reference's builder would have gone on to emit
/// once it had descended through all of them.
///
/// # ⛔⛔ THE LIST IS WALKED **BACKWARDS**, SO ENTRY `0` IS THE INNERMOST LOOP
///
/// `for (int i = outer_loop_sizes.size() - 1; i >= 0; i--)` with
/// `builder.setInsertionPointToStart(loop.getBody())` at the end of each iteration (`:774`, `:780`,
/// `:837`): the FIRST loop the derivation pushed is built LAST and therefore sits deepest. Walking
/// the list forwards would invert the whole nest — and since the outer positions are the view's
/// higher dims, that is an addressing order, not a preference.
///
/// # ⛔⛔ THE TWO ARMS ARE DIFFERENT LOOP KINDS, AND ONLY ONE OF THEM IS NAMED
///
/// `ss == el` gives a plain `affine.for 0 to ss` carrying
/// `dbgName = "ImplicitLoopForContiguousTransfer(<transfer name>)"` (`:777-779`). `ss != el` gives
/// an `scf.for` whose upper bound is an `arith.select` between the two counts, keyed on whether the
/// PARENT loop is on its last iteration — and `scf::ForOp::create` at `:836` sets no `dbgName` at
/// all. That asymmetry is the reference's; the `scf` loops it produces are exactly the input
/// `TransformLoopToLegalizeForSentientLowering` exists to remove, and that pass copies the name it
/// finds — so an unnamed one arrives at the pass unnamed.
///
/// ⛔ SO THIS BRIDGE **DOES** EMIT [`scf::Op::For`]. That variant's own note says it is an input op
/// the bridge does not build, which was true of every unit ported before this one.
///
/// # ⭐ THE VALUES ARE MINTED IN THE REFERENCE'S ORDER AND THE NEST IS ASSEMBLED AFTERWARDS
///
/// A nest held as a value has to be built innermost-first, while the reference's builder walks
/// outermost-first and mints as it goes. Doing both in one pass would renumber every SSA name in the
/// emitted text. So the pieces are minted in the reference's order (`i` descending) and the nest is
/// folded together from the innermost end in a second pass.
///
/// ⛔ THE `scf` ARM'S CONSTANTS LAND IN THE **ENCLOSING** LOOP'S BODY, before the loop they bound,
/// because that is where the builder's insertion point is when they are created (`:803`, `:824-831`).
///
/// ⛔ A LOOP WITH NO PARENT IS THE REFUSAL, on all four of the reference's stops: the
/// `DT_CHECK_MSG(parent_loops[i] != nullptr, ..)` at `:789-791`, the `DT_CHECK` at `:795`, an
/// `affine.for` parent without CONSTANT BOUNDS at `:804` — note `hasConstantBounds()`, both bounds,
/// not just the upper one entries 066 and 069 ask about — and a parent that is neither loop kind at
/// `:820`, which the reference's own comment calls unreachable by construction.
pub fn emit_implicit_loops_for_contiguous_transfer<'s>(
    vals: &mut Values,
    implicit_loops: &[ImplicitLoop],
    transfer_name: &str,
    parent_loop: impl Fn(PrimaryDim) -> Option<&'s DfirOp>,
    body: Vec<DfirOp>,
) -> Option<ImplicitNest> {
    /// One level of the nest, with everything it needs already minted.
    enum Level {
        /// The `ss == el` arm.
        Affine { extent: i64 },
        /// The `ss != el` arm: the ops that precede the loop, and its three operands.
        Scf {
            prefix: Vec<DfirOp>,
            lo: Val,
            hi: Val,
            step: Val,
        },
    }

    let mut levels: Vec<(usize, Val, Level)> = Vec::with_capacity(implicit_loops.len());

    // ── PASS ONE: mint in the reference's order, `i` from the last entry down to the first ──
    for (i, implicit) in implicit_loops.iter().enumerate().rev() {
        let StickCounts { steady, epilogue } = implicit.extents;

        if steady == epilogue {
            let iv = vals.mint();
            levels.push((i, iv, Level::Affine { extent: steady }));
            continue;
        }

        // A loop whose two counts differ needs a parent to compare against, so it needs a dim to
        // find that parent by — which the dummy loop of the empty-view branch has not got.
        let parent = parent_loop(implicit.dim?)?;
        let mut prefix = Vec::new();

        let (parent_iv, parent_last) = match parent {
            DfirOp::Affine(affine::Op::For { iv, lo, hi, .. }) => {
                // `if (affine_for.hasConstantBounds())` — BOTH bounds.
                let (affine::Bound::Const(_), affine::Bound::Const(ub)) = (lo, hi) else {
                    return None;
                };
                // `parent_loop_last_val = arith::ConstantIndexOp::create(builder, loc, ub - 1)`.
                let last = vals.mint();
                prefix.push(DfirOp::Arith(arith::Op::Constant {
                    result: last,
                    value: ub - 1,
                }));
                (*iv, last)
            }
            DfirOp::Scf(scf::Op::For { iv, hi, .. }) => {
                // `val_one`, then `SubIOp(scf_for.getUpperBound(), val_one)` — the bound is an
                // operand here, so "one less than it" has to be computed rather than written down.
                let one = vals.mint();
                prefix.push(DfirOp::Arith(arith::Op::Constant {
                    result: one,
                    value: 1,
                }));
                let last = vals.mint();
                prefix.push(DfirOp::Arith(arith::Op::SubI(IntBinary {
                    result: last,
                    lhs: *hi,
                    rhs: one,
                    ty: ScalarTy::Index,
                })));
                (*iv, last)
            }
            _ => return None,
        };

        // `cond = CmpIOp(slt, parent_loop_iv, parent_loop_last_val)` — TRUE on every iteration but
        // the parent's last, which is the one the epilogue count belongs to.
        let cond = vals.mint();
        prefix.push(DfirOp::Arith(arith::Op::Compare {
            result: cond,
            predicate: CmpIPredicate::Slt,
            lhs: parent_iv,
            rhs: parent_last,
            ty: ScalarTy::Index,
        }));

        let ss_val = vals.mint();
        prefix.push(DfirOp::Arith(arith::Op::Constant {
            result: ss_val,
            value: steady,
        }));
        let el_val = vals.mint();
        prefix.push(DfirOp::Arith(arith::Op::Constant {
            result: el_val,
            value: epilogue,
        }));
        // `ub = SelectOp(cond, ss_val, el_val)`.
        let hi = vals.mint();
        prefix.push(DfirOp::Arith(arith::Op::Select {
            result: hi,
            condition: cond,
            true_value: ss_val,
            false_value: el_val,
            ty: ScalarTy::Index,
        }));
        let lo = vals.mint();
        prefix.push(DfirOp::Arith(arith::Op::Constant {
            result: lo,
            value: 0,
        }));
        let step = vals.mint();
        prefix.push(DfirOp::Arith(arith::Op::Constant {
            result: step,
            value: 1,
        }));
        let iv = vals.mint();
        levels.push((
            i,
            iv,
            Level::Scf {
                prefix,
                lo,
                hi,
                step,
            },
        ));
    }

    // ── PASS TWO: fold the nest together from the innermost end ──
    let mut ivs = Vec::with_capacity(implicit_loops.len());
    let mut current = body;
    // `levels` was pushed with `i` descending, so reversing it walks innermost outward — which is
    // also `i` ASCENDING, so `ivs` comes out in `implicit_loops` order by construction.
    for (_, iv, level) in levels.into_iter().rev() {
        ivs.push(iv);
        let (mut ops, loop_op) = match level {
            Level::Affine { extent } => (
                Vec::new(),
                DfirOp::Affine(affine::Op::For {
                    iv,
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Const(extent),
                    carried: Vec::new(),
                    body: current,
                    dbg_name: Some(format!(
                        "ImplicitLoopForContiguousTransfer({transfer_name})"
                    )),
                }),
            ),
            Level::Scf {
                prefix,
                lo,
                hi,
                step,
            } => (
                prefix,
                DfirOp::Scf(scf::Op::For {
                    iv,
                    lo,
                    hi,
                    step,
                    carried: Vec::new(),
                    body: current,
                    dbg_name: None,
                }),
            ),
        };
        ops.push(loop_op);
        current = ops;
    }

    Some(ImplicitNest { ops: current, ivs })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 039 + 040/110 — THE 2B/16B SHUFFLES, WHOSE ARM TABLE IS THE SAME IN BOTH DIRECTIONS
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH `(element type, width, replication)` TRIPLES THE 2B/16B TABLE ADMITS — the nine arms both
/// directions list, identically (`SNTransferLowering.cpp:1786-1861` and `:1873-1948`), read against
/// the RESULT's element count for a load and against the quotient for a store.
///
/// ⛔⛔ `isF16()` IS NOT `bf16` (`:1786`): a bf16 transfer matches no arm at all and gets no shuffle,
/// which is the reference's default-constructed — null — `ShuffleOp` and [`None`] here.
fn is_2b16b_arm(elem: ElemType, width: u64, replication: i64) -> bool {
    match elem {
        ElemType::F16 | ElemType::Int(16) => {
            matches!((width, replication), (1, 64) | (16, 8) | (8, 8))
        }
        ElemType::Int(8) => matches!((width, replication), (2, 64) | (32, 8)),
        ElemType::Int(4) => matches!((width, replication), (4, 64) | (64, 8)),
        ElemType::F32 => matches!((width, replication), (4, 8) | (32, 1)),
        _ => false,
    }
}

/// Replaces: e039_construct2B16BLoadShuffle
///
/// THE SPLAT THAT WIDENS A LOAD TO ITS REPLICATED WIDTH — `indices = [0 .. elements_total)` and
/// `repetition = replicationFactor` on every one of the nine arms (`SNTransferLowering.cpp:1778-1863`).
///
/// ⛔⛔ THE INDEX LIST IS THE INPUT'S WHOLE WIDTH AND THE RESULT IS `rf` TIMES IT (`:1783-1785`),
/// which is exactly the relation `ShuffleOp::verify` enforces — `num_elements == indices.size() *
/// repetition` (`dataflow-scheduler/lib/Dialect/VectorChain/IR/VectorChain.cpp:198-211`).
///
/// ⚠️ THE `dbgName` SET FROM `transfer_->name_` HAS NO FIELD ON [`vectorchain::Op::Shuffle`] — the
/// same gap entry 070 records.
#[must_use]
pub fn construct_2b16b_load_shuffle(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    load_result: Val,
    result_ty: Vector,
    replication: Replication,
) -> Option<Val> {
    let elements_total = result_ty.len;
    if !is_2b16b_arm(result_ty.elem, elements_total, replication.get()) {
        return None;
    }
    let result = vals.mint();
    ops.push(DfirOp::VectorChain(vectorchain::Op::Shuffle {
        result,
        input: load_result,
        indices: (0..i32::try_from(elements_total).ok()?).collect(),
        repetition: u32::try_from(replication.get()).ok()?,
        input_ty: result_ty,
        // `constructVectorType(element_type, replicationFactor * elements_total)`.
        ty: Vector {
            len: elements_total.checked_mul(u64::try_from(replication.get()).ok()?)?,
            elem: result_ty.elem,
        },
    }));
    Some(result)
}

/// Replaces: e040_construct2B16BStoreShuffle
///
/// THE SHUFFLE THAT NARROWS REPLICATED DATA BACK TO ONE STORE'S WIDTH — `element_size =
/// getNumElements(result_type) / replicationFactor` indices, and `repetition = 1`
/// (`SNTransferLowering.cpp:1865-1949`).
///
/// ⛔⛔ DELIBERATE DIVERGENCE, ONE ARM OF NINE: the f32 `element_size == 4 && rf == 8` arm writes
/// `repetition = 8` (`:1937`) where the other eight write 1, and `4 != 4 * 8` is precisely what
/// `ShuffleOp::verify` rejects (`VectorChain.cpp:198-211`) — an op the reference cannot round-trip
/// through its own verifier. This emits 1, as every other arm does.
///
/// ⚠️ THE DIVISION IS THE REFERENCE'S TRUNCATING ONE, and it is the QUOTIENT — not the input width —
/// that the arm table is read against.
#[must_use]
pub fn construct_2b16b_store_shuffle(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    data: Val,
    result_ty: Vector,
    replication: Replication,
) -> Option<Val> {
    let element_size = result_ty.len / u64::try_from(replication.get()).ok()?;
    if !is_2b16b_arm(result_ty.elem, element_size, replication.get()) {
        return None;
    }
    let result = vals.mint();
    ops.push(DfirOp::VectorChain(vectorchain::Op::Shuffle {
        result,
        input: data,
        indices: (0..i32::try_from(element_size).ok()?).collect(),
        repetition: 1,
        input_ty: result_ty,
        // `constructVectorType(element_type, element_size)`.
        ty: Vector {
            len: element_size,
            elem: result_ty.elem,
        },
    }));
    Some(result)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 070/110 — A CONSTANT AS A TRANSFER'S INPUT
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE FOLD SPACE OF A `ddl.define_constant`, AND THE ONE WIDTH EVERY FOLD OF IT HAS —
/// `dsc2::ConstantInfo::data_.getAllData()`.
///
/// # ⛔⛔ THE REFERENCE TYPES THE WHOLE FOLD SPACE FROM `all_data.front().size()`
///
/// `constructTypeFromFormat(.., all_data.front().size(), ..)` (`SNTransferLowering.cpp:2485-2489`)
/// takes the width from the FIRST fold and then hands that one vector type to
/// [`super::dsc_lowering::uniformized_folded_constant_bitstream`], which stamps it on the
/// `vectorchain.constant_bitstream` it builds for EVERY fold. A fold of a different length would
/// therefore get an op whose value list and type disagree — an op MLIR's verifier rejects, built
/// from data the reference never compares. Requiring one width in the constructor is what makes
/// that unbuildable.
///
/// ⛔ AND `front()` ON AN EMPTY VECTOR IS UNDEFINED. An empty fold space is `None`, as is a
/// zero-width one — see [`constant_bitstream_and_shuffle`], where the width is a divisor.
///
/// ⭐ THE FORMAT TRAVELS WITH THE DATA for the same reason: [`ConstantData::stream_ty`] is the only
/// place the element type and the element count are put together, so the pair cannot be assembled
/// from a format belonging to one constant and a width belonging to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantData {
    folds: Vec<Vec<i64>>,
    width: Elements,
    format: DataType,
    on: GenericComp,
}

impl ConstantData {
    /// `cst_info.data_.getAllData()` with `cst_info.format_` and the lowering's `comp_`, and the one
    /// width all three are read for proved once.
    #[must_use]
    pub fn new(folds: Vec<Vec<i64>>, format: DataType, on: GenericComp) -> Option<ConstantData> {
        let width = folds.first()?.len();
        if width == 0 || folds.iter().any(|fold| fold.len() != width) {
            return None;
        }
        Some(ConstantData {
            folds,
            width: Elements(u64::try_from(width).ok()?),
            format,
            on,
        })
    }

    /// `constructTypeFromFormat(cst_info.format_, REGULAR_TENSOR, comp_, all_data.front().size(),
    /// ..)` (`SNTransferLowering.cpp:2485-2489`) — the type every fold's
    /// `vectorchain.constant_bitstream` is stamped with.
    ///
    /// ⭐ `REGULAR_TENSOR` IS FIXED AT THE CALL (`:2487`), so a scaled pair cannot reach this path.
    #[must_use]
    pub fn stream_ty(&self) -> Vector {
        type_from_format(
            self.format,
            self.on,
            TensorCategory::Regular,
            VectorWidth::Given(self.width),
        )
    }

    /// `all_data.front().size()` — the element count of every fold.
    #[must_use]
    pub const fn width(self: &ConstantData) -> Elements {
        self.width
    }

    /// The fold space itself, which entry 032 reads for its `size() == 1` gate.
    #[must_use]
    pub fn folds(&self) -> &[Vec<i64>] {
        &self.folds
    }
}

/// Replaces: e070_GenerateConstantBitStreamAndShuffle
///
/// **070/110** `SNTransferLowering::GenerateConstantBitStreamAndShuffle` —
/// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2475` (41L).
///
/// A CONSTANT WIDENED TO THE WIDTH ITS CONSUMER READS: one `vectorchain.constant_bitstream` per fold
/// of the constant, then one `vectorchain.shuffle` that repeats the pattern to fill the transfer's
/// destination type.
///
/// # ⛔⛔ `repetition` IS A QUOTIENT THE VERIFIER CHECKS, SO A NON-MULTIPLE IS A REFUSAL
///
/// `repetition = getNumElements(result_type) / getNumElements(bitstream_vector_type)` (`:2505-2506`)
/// is C++ integer division, and `ShuffleOp` states the invariant it has to satisfy in its own
/// description: *"the product of the number of `indices` and `repetitions` must match the number of
/// elements in the output"* — with `hasVerifier = 1` behind it (`VectorChain.td:441-443`, `:467`).
/// The reference truncates, so a destination width that is not a multiple of the constant's width
/// builds an op the verifier rejects, several passes downstream of the division that caused it.
/// Refusing the inexact quotient here is that failure moved to the line that can explain it.
///
/// ⭐ AND THE REFUSAL MOVES **AHEAD OF THE EMISSION**. The reference computes `repetition` after it
/// has already built the bitstream ops (`:2493-2497` precede `:2505`), so a refused call leaves them
/// behind; both checks here happen before anything is pushed, and `ops` is untouched on `None`.
///
/// ⭐ `indices` IS THE IDENTITY, `0 .. width` (`:2500-2504`), so the shuffle reorders nothing — it
/// only repeats.
///
/// ⭐ `(SNTransferLowering *)this` AT `:2492` IS MECHANISM, NOT MEANING. The function is `const` and
/// the one it calls is not, so the reference casts the constness away; nothing about the value being
/// built depends on it.
///
/// ⚠️ `ShuffleOp::create` ALSO PASSES `getStringAttr(transfer_->name_)`, which is the op's
/// `OptionalAttr<StrAttr>:$dbgName` (`VectorChain.td:462`), and [`vectorchain::Op::Shuffle`] has no
/// field for it — so the transfer's name is dropped from this one op. Recorded rather than papered
/// over: adding the field touches every `Shuffle` construction in bridge 2 and belongs to whoever
/// ports that op's `DebugNameOpInterface`, not to this entry.
pub fn constant_bitstream_and_shuffle(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    handles: &Handles,
    data: &ConstantData,
    values: BitstreamValues,
    bitstream: impl Fn(Core, Corelet, u32) -> Vec<i64>,
    result_ty: Vector,
) -> Option<Val> {
    let bitstream_ty = data.stream_ty();

    // ⛔ THE EXACT QUOTIENT, CHECKED BEFORE ANYTHING IS EMITTED. `bitstream_ty.len` is
    // `data.width()`, which `ConstantData::new` already proved positive.
    if !result_ty.len.is_multiple_of(bitstream_ty.len) {
        return None;
    }
    let repetition = u32::try_from(result_ty.len / bitstream_ty.len).ok()?;
    // `for (i < all_data.front().size()) index_attrs.push_back(getI32IntegerAttr(i))`.
    let indices = (0..data.width().0)
        .map(i32::try_from)
        .collect::<Result<Vec<i32>, _>>()
        .ok()?;

    let input = uniformized_folded_constant_bitstream(
        vals,
        ops,
        handles,
        data.folds(),
        bitstream,
        bitstream_ty,
        values,
    );

    let result = vals.mint();
    ops.push(DfirOp::VectorChain(vectorchain::Op::Shuffle {
        result,
        input,
        indices,
        repetition,
        input_ty: bitstream_ty,
        ty: result_ty,
    }));
    Some(result)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::symbol;
    use crate::islands::dataflow_ir::ty::ElemType;
    use crate::units::{DfirUnit, NumFolds};

    use super::super::control_flow::mlir_loop_from_sn_loop_node;

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
        let scale =
            construct_base_address(&mut vals, &mut ops, 2, &[], AddressForm::of(true, true));
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

    fn affine_for(iv: Val, ub: i64) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(ub),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    fn scf_for(iv: Val, hi: Val) -> DfirOp {
        DfirOp::Scf(scf::Op::For {
            iv,
            lo: Val(90),
            hi,
            step: Val(91),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })
    }

    /// ⛔ THE TIME SET IS INEQUALITY-ONLY, so a loop of extent one is a PAIR and not an `== 0` — a
    /// different `affine_set<>` text from the one [`IntegerSet::from_sizes`] writes for the same
    /// points.
    #[test]
    fn the_time_set_is_inequality_only_so_a_unit_loop_is_not_pinned() {
        let set = time_set(&[LoopBound::Constant(4), LoopBound::Constant(1)])
            .expect("both bounds are literals");
        assert_eq!(set.dims, 2);
        // `IntegerSet::get(ndims, 0, ..)`, and the `num_symbols++` above it is a dead store.
        assert_eq!(set.symbols, 0);
        assert_eq!(set.constraints.len(), 4);
        assert!(set.constraints.iter().all(|entry| !entry.is_equality));
        assert_eq!(set.constraints[2].expr, AffineExpr::dim(1));
        assert_eq!(
            set.constraints[3].expr,
            AffineExpr::dim(1).times(-1).plus(AffineExpr::Const(0))
        );
        assert_ne!(set, IntegerSet::from_sizes(&[4, 1]));

        // `else { return LogicalResult::failure(); }` on either loop kind.
        assert_eq!(
            time_set(&[LoopBound::Constant(4), LoopBound::Dynamic]),
            None
        );
        // No loops is the empty set of no dimensions, which is what `IntegerSet::get(0, 0, {}, {})`
        // builds.
        let none = time_set(&[]).expect("no loops is not a refusal");
        assert_eq!(none.dims, 0);
        assert!(none.constraints.is_empty());
    }

    /// ⛔ AN `affine.for`'s BOUND IS AN ATTRIBUTE AND AN `scf.for`'s IS AN OPERAND, so only one of
    /// the two needs the scope walked — and a third op kind is no answer at all, which is entry
    /// 069's `llvm_unreachable` and entry 066's silent length mismatch.
    #[test]
    fn an_scf_bound_is_read_through_its_defining_constant_and_a_third_kind_is_no_answer() {
        let six = Val(1);
        let sym = Val(2);
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: six,
                value: 6,
            }),
            DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: sym,
                symbol_id: 3,
                max_value: None,
            }),
        ];

        assert_eq!(
            loop_upper_bound(&scf_for(Val(10), six), &scope),
            Some(LoopBound::Constant(6))
        );
        // `symbol.create_symbol` is not an `arith::ConstantIndexOp`.
        assert_eq!(
            loop_upper_bound(&scf_for(Val(11), sym), &scope),
            Some(LoopBound::Dynamic)
        );
        // The attribute form needs no scope at all.
        assert_eq!(
            loop_upper_bound(&affine_for(Val(12), 8), &[]),
            Some(LoopBound::Constant(8))
        );
        assert_eq!(
            loop_upper_bound(
                &DfirOp::Affine(affine::Op::For {
                    iv: Val(13),
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Val(sym),
                    carried: Vec::new(),
                    body: Vec::new(),
                    dbg_name: None,
                }),
                &scope
            ),
            Some(LoopBound::Dynamic)
        );
        // ⛔ *"Unknown for-loops"*.
        assert_eq!(loop_upper_bound(&scope[0], &scope), None);
    }

    /// ⭐ ENTRY 069 IS A PREDICATE OVER THE SAME ANSWER ENTRY 066 BUILDS ITS SET FROM.
    #[test]
    fn an_epilogue_is_a_bound_no_constant_defines() {
        assert!(!epilogues_in_loops(&[]));
        assert!(!epilogues_in_loops(&[
            LoopBound::Constant(4),
            LoopBound::Constant(1)
        ]));
        assert!(epilogues_in_loops(&[
            LoopBound::Constant(4),
            LoopBound::Dynamic
        ]));
    }

    /// ⛔ THE STRIDED DIMENSION SPANS THE **CHUNK COUNT**, and every equality is spliced after every
    /// inequality rather than interleaved per dimension.
    #[test]
    fn the_strided_dim_counts_chunks_and_every_equality_comes_last() {
        let sizes = [ChunkDim {
            size: 8,
            src_index: Some(0),
            dst_index: None,
        }];
        // ⛔ ITS OWN `size_` IS NEVER READ — 999 must not appear anywhere in the answer.
        let stride = ChunkDim {
            size: 999,
            src_index: Some(2),
            dst_index: None,
        };

        let set = load_or_store_set(&sizes, Some(&stride), 3, TransferSide::Load, 4)
            .expect("no dimension is in both lists");
        assert_eq!(set.dims, 3);
        assert_eq!(set.symbols, 0);
        let flags: Vec<bool> = set
            .constraints
            .iter()
            .map(|entry| entry.is_equality)
            .collect();
        assert_eq!(flags, vec![false, false, false, false, true]);
        // d0 spans the chunk's 8 elements; d2 spans the 4 chunks; d1, claimed by neither, is pinned.
        assert_eq!(
            set.constraints[1].expr,
            AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(7))
        );
        assert_eq!(
            set.constraints[3].expr,
            AffineExpr::dim(2).times(-1).plus(AffineExpr::Const(3))
        );
        assert_eq!(set.constraints[4].expr, AffineExpr::dim(1));

        // ⛔ THE SIDE SELECTS THE FIELD: these records name only `srcSizeIdx_`, so a store matches
        // nothing and every position is pinned.
        let store = load_or_store_set(&sizes, Some(&stride), 3, TransferSide::Store, 4)
            .expect("the store side matches nothing");
        assert!(store.constraints.iter().all(|entry| entry.is_equality));
        assert_eq!(store.constraints.len(), 3);

        // ⛔ A CLAIMED DIMENSION OF EXTENT ONE IS STILL A POINT.
        let unit = [ChunkDim {
            size: 1,
            src_index: Some(0),
            dst_index: None,
        }];
        assert_eq!(
            load_or_store_set(&unit, None, 1, TransferSide::Load, 4)
                .expect("one dimension")
                .constraints,
            vec![Constraint {
                expr: AffineExpr::dim(0),
                is_equality: true,
            }]
        );

        // ⛔ *"A dimension cannot be present in both chunk size and stride"*.
        let clash = [ChunkDim {
            size: 8,
            src_index: Some(2),
            dst_index: None,
        }];
        assert_eq!(
            load_or_store_set(&clash, Some(&stride), 3, TransferSide::Load, 4),
            None
        );
    }

    /// ⛔ `OUT` IS DIVIDED BY THE REPLICATION FACTOR BEFORE IT IS ASKED FOR A LOOP, and the records
    /// the answer came from are left decremented.
    #[test]
    fn the_out_dim_is_divided_by_the_replication_factor_before_it_is_asked_for_a_loop() {
        let mut sticks = ContiguousSticks::new(
            StickCounts {
                steady: 1,
                epilogue: 1,
            },
            vec![
                (
                    PrimaryDim::Out,
                    StickCounts {
                        steady: 8,
                        epilogue: 8,
                    },
                ),
                (
                    PrimaryDim::Y,
                    StickCounts {
                        steady: 4,
                        epilogue: 2,
                    },
                ),
            ],
        );
        // Position 0 is below `unit_time_dims`, so the walk never reaches `In`.
        let view = [
            ViewSize {
                dim: PrimaryDim::In,
                size: 2,
            },
            ViewSize {
                dim: PrimaryDim::Out,
                size: 1,
            },
            ViewSize {
                dim: PrimaryDim::Y,
                size: 1,
            },
        ];
        let four = Replication::checked(4).expect("a factor of four");
        let loops = implicit_loops_for_contiguous_transfer(&view, 1, &mut sticks, four)
            .expect("both walked dims have stick records");

        assert_eq!(
            loops,
            vec![
                ImplicitLoop {
                    size_index: Some(1),
                    dim: Some(PrimaryDim::Out),
                    extents: StickCounts {
                        steady: 2,
                        epilogue: 2,
                    },
                },
                ImplicitLoop {
                    size_index: Some(2),
                    dim: Some(PrimaryDim::Y),
                    extents: StickCounts {
                        steady: 4,
                        epilogue: 2,
                    },
                },
            ]
        );
        // ⛔ THE SECOND OUTPUT: 8/4 = 2, then `-= 1`; and Y is decremented without being divided.
        assert_eq!(
            sticks.per_dim(PrimaryDim::Out),
            Some(StickCounts {
                steady: 1,
                epilogue: 1,
            })
        );
        assert_eq!(
            sticks.per_dim(PrimaryDim::Y),
            Some(StickCounts {
                steady: 3,
                epilogue: 1,
            })
        );

        // ⛔ *"avoid unit size loops or zero loops after substituition"* — 8/8 is 1, so no loop.
        let mut alone = ContiguousSticks::new(
            StickCounts {
                steady: 1,
                epilogue: 1,
            },
            vec![(
                PrimaryDim::Out,
                StickCounts {
                    steady: 8,
                    epilogue: 8,
                },
            )],
        );
        assert_eq!(
            implicit_loops_for_contiguous_transfer(
                &[ViewSize {
                    dim: PrimaryDim::Out,
                    size: 1,
                }],
                0,
                &mut alone,
                Replication::checked(8).expect("a factor of eight"),
            ),
            Some(Vec::new())
        );

        // ⛔ AND THE DIVISOR CANNOT BE ZERO.
        assert_eq!(Replication::checked(0), None);
    }

    /// ⛔ THE TWO STOPS OF THE DERIVATION, AND THE ONE LOOP AN EMPTY VIEW BUILDS.
    #[test]
    fn a_view_dim_with_no_stick_record_and_a_skewed_whole_count_are_both_refusals() {
        let one = Replication::checked(1).expect("a factor of one");

        // `DT_ERROR("view dims and contigous transfer dim didn't match")`.
        let mut no_records = ContiguousSticks::new(
            StickCounts {
                steady: 1,
                epilogue: 1,
            },
            Vec::new(),
        );
        assert_eq!(
            implicit_loops_for_contiguous_transfer(
                &[ViewSize {
                    dim: PrimaryDim::Mb,
                    size: 1,
                }],
                0,
                &mut no_records,
                one,
            ),
            None
        );

        // `DT_CHECK_MSG(sticks_src_ss == sticks_src_el, ..)` — only reachable with an empty view.
        let mut skewed = ContiguousSticks::new(
            StickCounts {
                steady: 4,
                epilogue: 2,
            },
            Vec::new(),
        );
        assert_eq!(
            implicit_loops_for_contiguous_transfer(&[], 0, &mut skewed, one),
            None
        );

        // The dummy loop: no view position, no dim, and counts that cannot differ.
        let mut whole = ContiguousSticks::new(
            StickCounts {
                steady: 4,
                epilogue: 4,
            },
            Vec::new(),
        );
        assert_eq!(
            implicit_loops_for_contiguous_transfer(&[], 0, &mut whole, one),
            Some(vec![ImplicitLoop {
                size_index: None,
                dim: None,
                extents: StickCounts {
                    steady: 4,
                    epilogue: 4,
                },
            }])
        );
        // A whole transfer that fits in one burst needs no loop.
        let mut single = ContiguousSticks::new(
            StickCounts {
                steady: 1,
                epilogue: 1,
            },
            Vec::new(),
        );
        assert_eq!(
            implicit_loops_for_contiguous_transfer(&[], 0, &mut single, one),
            Some(Vec::new())
        );
    }

    /// ⛔ THE LIST IS WALKED BACKWARDS, SO ENTRY 0 ENDS UP INNERMOST — and the `ss != el` arm's
    /// bound is an `arith.select` keyed on whether the PARENT loop is on its last iteration.
    #[test]
    fn the_nest_is_built_backwards_and_the_epilogue_bound_is_a_select_on_the_parent_iv() {
        let parent_iv = Val(100);
        let parent = affine_for(parent_iv, 4);
        let loops = vec![
            ImplicitLoop {
                size_index: Some(1),
                dim: Some(PrimaryDim::Y),
                extents: StickCounts {
                    steady: 4,
                    epilogue: 2,
                },
            },
            ImplicitLoop {
                size_index: Some(2),
                dim: Some(PrimaryDim::Out),
                extents: StickCounts {
                    steady: 3,
                    epilogue: 3,
                },
            },
        ];

        let mut vals = Values::default();
        let nest = emit_implicit_loops_for_contiguous_transfer(
            &mut vals,
            &loops,
            "load0",
            |_| Some(&parent),
            Vec::new(),
        )
        .expect("an affine.for parent with constant bounds");

        // ⛔ ENTRY 1 IS THE OUTERMOST LOOP, and it is the only one that carries a name.
        let [
            DfirOp::Affine(affine::Op::For {
                iv: outer_iv,
                lo,
                hi,
                body,
                dbg_name,
                ..
            }),
        ] = nest.ops.as_slice()
        else {
            panic!("one outermost affine.for, got {:?}", nest.ops)
        };
        assert_eq!(*lo, affine::Bound::Const(0));
        assert_eq!(*hi, affine::Bound::Const(3));
        assert_eq!(
            dbg_name.as_deref(),
            Some("ImplicitLoopForContiguousTransfer(load0)")
        );
        // ⭐ MINTED IN THE REFERENCE'S ORDER: the outermost loop's IV first, then the inner arm's
        // seven values, then its IV.
        assert_eq!(nest.ivs, vec![Val(8), Val(0)]);
        assert_eq!(*outer_iv, Val(0));

        let [
            DfirOp::Arith(arith::Op::Constant {
                result: last,
                value: 3,
            }),
            DfirOp::Arith(arith::Op::Compare {
                predicate: CmpIPredicate::Slt,
                lhs,
                rhs,
                ..
            }),
            DfirOp::Arith(arith::Op::Constant { value: 4, .. }),
            DfirOp::Arith(arith::Op::Constant { value: 2, .. }),
            DfirOp::Arith(arith::Op::Select {
                result: selected,
                condition,
                ty: ScalarTy::Index,
                ..
            }),
            DfirOp::Arith(arith::Op::Constant { value: 0, .. }),
            DfirOp::Arith(arith::Op::Constant { value: 1, .. }),
            DfirOp::Scf(scf::Op::For {
                iv: inner_iv,
                hi: scf_hi,
                dbg_name: None,
                ..
            }),
        ] = body.as_slice()
        else {
            panic!("the scf arm's seven values then its loop, got {body:?}")
        };
        // `ub - 1` compared against the parent's own induction variable.
        assert_eq!(*lhs, parent_iv);
        assert_eq!(rhs, last);
        assert_eq!(scf_hi, selected);
        assert_eq!(*inner_iv, Val(8));
        // The compare feeds the select that bounds the loop.
        let DfirOp::Arith(arith::Op::Compare { result: cond, .. }) = &body[1] else {
            panic!("a compare")
        };
        assert_eq!(condition, cond);
    }

    /// ⛔ AN `scf.for` PARENT HAS TO **COMPUTE** ONE LESS THAN ITS BOUND, and a parent that cannot
    /// answer at all is a refusal on every one of the reference's four stops.
    #[test]
    fn an_scf_parent_subtracts_and_a_parent_that_cannot_answer_is_a_refusal() {
        let ss_el = [ImplicitLoop {
            size_index: Some(0),
            dim: Some(PrimaryDim::Y),
            extents: StickCounts {
                steady: 4,
                epilogue: 2,
            },
        }];

        let scf_parent = scf_for(Val(200), Val(202));
        let mut vals = Values::default();
        let nest = emit_implicit_loops_for_contiguous_transfer(
            &mut vals,
            &ss_el,
            "s",
            |_| Some(&scf_parent),
            Vec::new(),
        )
        .expect("an scf.for parent");
        assert!(matches!(
            nest.ops.first(),
            Some(DfirOp::Arith(arith::Op::Constant { value: 1, .. }))
        ));
        assert!(matches!(
            nest.ops.get(1),
            Some(DfirOp::Arith(arith::Op::SubI(IntBinary { lhs, ty: ScalarTy::Index, .. })))
                if *lhs == Val(202)
        ));

        // `if (affine_for.hasConstantBounds())` — BOTH bounds, so a dynamic upper one refuses.
        let dynamic = DfirOp::Affine(affine::Op::For {
            iv: Val(300),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Val(Val(301)),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });
        assert!(
            emit_implicit_loops_for_contiguous_transfer(
                &mut Values::default(),
                &ss_el,
                "s",
                |_| Some(&dynamic),
                Vec::new(),
            )
            .is_none()
        );

        // A parent that is not a loop at all — the branch the reference calls unreachable.
        let not_a_loop = DfirOp::Arith(arith::Op::Constant {
            result: Val(400),
            value: 0,
        });
        assert!(
            emit_implicit_loops_for_contiguous_transfer(
                &mut Values::default(),
                &ss_el,
                "s",
                |_| Some(&not_a_loop),
                Vec::new(),
            )
            .is_none()
        );

        // `DT_CHECK_MSG(parent_loops[i] != nullptr, ..)`.
        assert!(
            emit_implicit_loops_for_contiguous_transfer(
                &mut Values::default(),
                &ss_el,
                "s",
                |_| None,
                Vec::new(),
            )
            .is_none()
        );

        // ⛔ AND THE DUMMY LOOP HAS NO DIM TO FIND A PARENT BY, so a skewed one cannot be emitted.
        let dummy = [ImplicitLoop {
            size_index: None,
            dim: None,
            extents: StickCounts {
                steady: 4,
                epilogue: 2,
            },
        }];
        assert!(
            emit_implicit_loops_for_contiguous_transfer(
                &mut Values::default(),
                &dummy,
                "s",
                |_| Some(&scf_parent),
                Vec::new(),
            )
            .is_none()
        );
    }

    /// ⛔ THE SAME MIRRORED INDEX AS ENTRY 018, WITH THE OPPOSITE TIE-BREAK — the divergence is
    /// visible only on a band that names one dimension twice.
    #[test]
    fn a_split_band_gives_this_walk_the_outermost_loop_and_entry_018_the_innermost() {
        // `dims_` is innermost-first; the record is outermost-first.
        let dims = [PrimaryDim::Y, PrimaryDim::Out, PrimaryDim::Y];
        let loops = [Val(1), Val(2), Val(3)];

        assert_eq!(
            outermost_mlir_loop_for_dim(&dims, &loops, PrimaryDim::Y),
            Some(&Val(1))
        );
        assert_eq!(
            mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Y),
            Some(&Val(3))
        );
        // They agree on a dim named once.
        assert_eq!(
            outermost_mlir_loop_for_dim(&dims, &loops, PrimaryDim::Out),
            mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Out)
        );
        // A dim the node does not name, and a record shorter than the node's dims: both `.at()`
        // throws in the reference.
        assert_eq!(
            outermost_mlir_loop_for_dim(&dims, &loops, PrimaryDim::Mb),
            None
        );
        assert_eq!(
            outermost_mlir_loop_for_dim(&dims, &loops[..1], PrimaryDim::Out),
            None
        );
    }

    /// ⛔ `repetition` IS AN EXACT QUOTIENT THE OP'S VERIFIER CHECKS, and an inexact one is refused
    /// before any op is pushed.
    #[test]
    fn the_shuffle_repetition_is_an_exact_quotient_and_the_refusal_emits_nothing() {
        let mut vals = Values::default();
        let iterator = vals.mint();
        let unit = vals.mint();
        let handles = Handles::new(
            &[Core::checked(0).expect("core 0")],
            &[Corelet::checked(0).expect("corelet 0")],
            NumFolds(1),
            iterator,
            |_, _, _| unit,
        );
        let data = ConstantData::new(
            vec![vec![0x00, 0x01]],
            DataType::Sen169Fp16,
            GenericComp::Lxlu,
        )
        .expect("one fold of two elements");
        let result_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };

        let mut ops = Vec::new();
        let got = constant_bitstream_and_shuffle(
            &mut vals,
            &mut ops,
            &handles,
            &data,
            BitstreamValues::BitPatterns,
            |_, _, _| Vec::new(),
            result_ty,
        )
        .expect("64 elements is 32 repeats of 2");

        let [
            DfirOp::VectorChain(vectorchain::Op::ConstantBitstream {
                result: stream,
                value,
                ty: stream_ty,
                is_symbol: false,
            }),
            DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result,
                input,
                indices,
                repetition,
                input_ty,
                ty: out_ty,
            }),
        ] = ops.as_slice()
        else {
            panic!("a single-fold bitstream then the shuffle, got {ops:?}")
        };
        assert_eq!(value, &vec![0x00, 0x01]);
        // The width is `all_data.front().size()`, and the element the format's own.
        assert_eq!(
            *stream_ty,
            Vector {
                len: 2,
                elem: ElemType::F16,
            }
        );
        assert_eq!(input, stream);
        assert_eq!(input_ty, stream_ty);
        // ⭐ THE IDENTITY: the shuffle repeats, it does not reorder.
        assert_eq!(indices, &vec![0, 1]);
        assert_eq!(*repetition, 32);
        assert_eq!(*out_ty, result_ty);
        assert_eq!(got, *result);

        // ⛔ 65 IS NOT A MULTIPLE OF 2, and the refusal leaves nothing behind.
        let mut refused = Vec::new();
        assert!(
            constant_bitstream_and_shuffle(
                &mut vals,
                &mut refused,
                &handles,
                &data,
                BitstreamValues::BitPatterns,
                |_, _, _| Vec::new(),
                Vector {
                    len: 65,
                    elem: ElemType::F16,
                },
            )
            .is_none()
        );
        assert!(refused.is_empty());

        // ⛔ AND A FOLD SPACE WITH NO ONE WIDTH CANNOT TYPE THE STREAM AT ALL.
        let fp16 = |folds| ConstantData::new(folds, DataType::Sen169Fp16, GenericComp::Lxlu);
        assert_eq!(fp16(Vec::new()), None);
        assert_eq!(fp16(vec![Vec::new()]), None);
        assert_eq!(fp16(vec![vec![1], vec![1, 2]]), None);
    }

    /// 🎯 034/110 — ⛔ THE FIRST DIM IS THE FASTEST, and a bypassed layout keeps the extents.
    ///
    /// A `(4, 8, 2)` view whose `d0` strided by 8 or 32 instead of 1 addresses the wrong element of
    /// every transfer that reaches this view.
    #[test]
    fn the_view_layout_strides_the_first_dim_by_one_and_nests_to_the_right() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let sizes = [
            ViewSize {
                dim: PrimaryDim::Out,
                size: 4,
            },
            ViewSize {
                dim: PrimaryDim::Y,
                size: 8,
            },
            ViewSize {
                dim: PrimaryDim::X,
                size: 2,
            },
        ];

        let got = logical_memory_view(
            &mut vals,
            &mut ops,
            &sizes,
            Val(50),
            Val(51),
            ElemType::F16,
            false,
        )
        .expect("three positive extents");

        assert_eq!(
            got.ty,
            MemRef {
                shape: vec![4, 8, 2],
                elem: ElemType::F16,
            }
        );
        assert_eq!(
            got.layout,
            AffineMap {
                dims: 3,
                syms: 0,
                results: vec![
                    AffineExpr::dim(2)
                        .times(32)
                        .plus(AffineExpr::dim(1).times(4).plus(AffineExpr::dim(0)))
                ],
            }
        );
        assert_eq!(
            ops,
            vec![DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(0),
                from: Val(50),
                start: Val(51),
                layout: got.layout.clone(),
                ty: got.ty.clone(),
            })]
        );

        let bypassed = logical_memory_view(
            &mut vals,
            &mut ops,
            &sizes,
            Val(50),
            Val(51),
            ElemType::F16,
            true,
        )
        .expect("three positive extents");
        assert_eq!(bypassed.layout, AffineMap::identity(1));
        assert_eq!(bypassed.ty, got.ty);
    }

    /// 🎯 035/110 — ⛔ THE PERMUTATION IS THE REVERSAL, and an empty nest has no time order.
    #[test]
    fn the_time_order_reverses_every_loop() {
        assert_eq!(
            time_order(3),
            Some(AffineMap {
                dims: 3,
                syms: 0,
                results: vec![AffineExpr::dim(2), AffineExpr::dim(1), AffineExpr::dim(0)],
            })
        );
        assert_eq!(time_order(0), None);
    }

    /// 🎯 036/110 — ⛔ THE VARIABLE IS THE LOOP'S POSITION AND `sizeIdx_` IS ONLY THE RESULT SLOT.
    ///
    /// Reading `d<sizeIdx_>` instead of `d<i>` would address the third loop's time step with the
    /// first loop's iterator, and every result no loop names would still have to be zero.
    #[test]
    fn the_time_address_map_indexes_by_loop_position() {
        let got = time_address_map(
            3,
            3,
            &[
                CompositeLoop {
                    size_idx: Some(1),
                    elem_offset: 4,
                },
                CompositeLoop {
                    size_idx: None,
                    elem_offset: 9,
                },
                CompositeLoop {
                    size_idx: Some(0),
                    elem_offset: 1,
                },
            ],
        )
        .expect("every loop inside the input arity");

        assert_eq!(
            got,
            AffineMap {
                dims: 3,
                syms: 0,
                results: vec![
                    AffineExpr::dim(2),
                    AffineExpr::dim(0).times(4),
                    AffineExpr::Const(0),
                ],
            }
        );
        // `dims[i]` past the input arity is the reference's out-of-bounds read.
        assert_eq!(
            time_address_map(
                1,
                3,
                &[
                    CompositeLoop {
                        size_idx: Some(0),
                        elem_offset: 1,
                    },
                    CompositeLoop {
                        size_idx: Some(1),
                        elem_offset: 1,
                    },
                ],
            ),
            None
        );
    }

    /// 🎯 037/110 — ⛔ THE QUERIED NODE IS ITS OWN ANSWER, and a split numerator stage is passed over.
    ///
    /// Starting the walk at `getOwnerLoop()` would skip the node that already walks the dim, and
    /// accepting a candidate whose `numId_` stage differs between start and end picks a loop whose
    /// trip count is not the one the transfer was sized for.
    #[test]
    fn the_node_itself_can_answer_and_a_split_numerator_stage_is_passed_over() {
        let (node, split, outer) = (1i32, 2i32, 3i32);
        let query = DimStageVals {
            den_ss: Some(7),
            den_el: Some(7),
            num_ss: Some(0),
            num_el: Some(0),
        };
        let walks = [PrimaryDim::Y];
        let walks_nothing: [PrimaryDim; 0] = [];

        assert_eq!(
            immediate_parent_with_matching_dim(
                &[
                    DimChainNode {
                        node: &node,
                        dims: &walks,
                        stages: query,
                    },
                    DimChainNode {
                        node: &outer,
                        dims: &walks,
                        stages: query,
                    },
                ],
                PrimaryDim::Y,
            ),
            Some(&node)
        );

        assert_eq!(
            immediate_parent_with_matching_dim(
                &[
                    DimChainNode {
                        node: &node,
                        dims: &walks_nothing,
                        stages: query,
                    },
                    DimChainNode {
                        node: &split,
                        dims: &walks,
                        stages: DimStageVals {
                            num_el: Some(1),
                            ..query
                        },
                    },
                    DimChainNode {
                        node: &outer,
                        dims: &walks,
                        stages: query,
                    },
                ],
                PrimaryDim::Y,
            ),
            Some(&outer)
        );
        assert_eq!(
            immediate_parent_with_matching_dim::<i32>(&[], PrimaryDim::Y),
            None
        );
    }

    /// 🎯 038/110 — ⛔ NEITHER COUNT ABOVE ONE MEANS NO EPILOGUE, however unequal the two are.
    #[test]
    fn one_steady_stick_against_an_empty_epilogue_is_not_an_epilogue() {
        let whole = StickCounts {
            steady: 1,
            epilogue: 0,
        };
        let of = |steady, epilogue| {
            ContiguousSticks::new(
                whole,
                vec![(PrimaryDim::Out, StickCounts { steady, epilogue })],
            )
        };
        assert!(!epilogues_in_transfer_sizes(&of(1, 0)));
        assert!(epilogues_in_transfer_sizes(&of(4, 3)));
        assert!(!epilogues_in_transfer_sizes(&of(4, 4)));
    }

    /// 🎯 039/110 — ⛔ THE INDICES ARE THE INPUT'S WHOLE WIDTH AND `bf16` MATCHES NO ARM.
    #[test]
    fn the_load_shuffle_repeats_the_whole_input_rf_times() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let rf = Replication::checked(8).expect("a positive factor");
        let ty = Vector {
            len: 8,
            elem: ElemType::F16,
        };

        assert_eq!(
            construct_2b16b_load_shuffle(&mut vals, &mut ops, Val(70), ty, rf),
            Some(Val(0))
        );
        assert_eq!(
            ops,
            vec![DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(0),
                input: Val(70),
                indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
                repetition: 8,
                input_ty: ty,
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            })]
        );
        assert_eq!(
            construct_2b16b_load_shuffle(
                &mut vals,
                &mut ops,
                Val(70),
                Vector {
                    len: 8,
                    elem: ElemType::Bf16,
                },
                rf,
            ),
            None
        );
    }

    /// 🎯 040/110 — ⛔ THE DELIBERATE DIVERGENCE: `repetition = 1` on the f32 arm the reference wrote
    /// `8` on, because `4 != 4 * 8` is what `ShuffleOp::verify` rejects.
    #[test]
    fn the_store_shuffle_narrows_to_the_quotient_and_repeats_once() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let rf = Replication::checked(8).expect("a positive factor");
        let input_ty = Vector {
            len: 32,
            elem: ElemType::F32,
        };

        assert_eq!(
            construct_2b16b_store_shuffle(&mut vals, &mut ops, Val(80), input_ty, rf),
            Some(Val(0))
        );
        assert_eq!(
            ops,
            vec![DfirOp::VectorChain(vectorchain::Op::Shuffle {
                result: Val(0),
                input: Val(80),
                indices: vec![0, 1, 2, 3],
                repetition: 1,
                input_ty,
                ty: Vector {
                    len: 4,
                    elem: ElemType::F32,
                },
            })]
        );
    }
}
