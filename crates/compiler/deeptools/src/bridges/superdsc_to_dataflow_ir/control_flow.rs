//! THE LOOP NEST AND THE CONDITIONALS — DSC loops, blocks and conds becoming the scf/affine nest.
//!
//! 14 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e017_getCmpIPredicate_dup` | 0 | 25 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:21` |
//! | `e018_getMLIRLoopFromSNLoopNode` | 0 | 16 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:49` |
//! | `e019_getParentLoop` | 0 | 26 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:381` |
//! | `e020_propagateBufferSwitchLoopsToRootRecursively` | 0 | 50 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:519` |
//! | `e021_resetIterArguments` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:738` |
//! | `e054_constructConditionalOperation` | 1 | 211 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:66` |
//! | `e055_constructConditionalsForSAMV` | 1 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:284` |
//! | `e056_getBufferingOrStreamingMode` | 1 | 54 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:408` |
//! | `e057_propagateBufferSwitchLoopsToRoot` | 1 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:575` |
//! | `e058_constructLoopForADim` | 1 | 101 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:754` |
//! | `e074_getBlockingOrStreamingBufferLoopLocations` | 2 | 46 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:467` |
//! | `e075_constructLoopIterArgs` | 2 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:595` |
//! | `e088_constructLoopsRecursive` | 3 | 324 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:862` |
//! | `e096_constructLoops` | 4 | 27 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:1191` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use std::num::NonZeroI32;

use super::dsc_lowering::{
    DataLocation, Factor, Handles, address_granularity_multiply_factor, constant_index,
    uniformized_folded_address,
};
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::arith::{CmpIPredicate, IntBinary, IntConst};
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith, scf};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::units::{Core, Corelet, DfirUnit};

/// A DIMENSION THE SCHEDULE IS WRITTEN OVER — `PrimaryDimTypes` (`dsc/dims.h:34-48`).
///
/// ⛔ TWELVE, NOT THIRTEEN: `PrimaryDimTypesCount` is the reference's count-and-unset sentinel, the
/// value `PrimaryDimAndKind` defaults to (`dims.h:77`). Absence is an [`Option`] here, so a dim that
/// was never set cannot be compared equal to a real one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimaryDim {
    /// `IN` — input features.
    In,
    /// `OUT` — output features.
    Out,
    /// `IJ` — the output image, rows and cols together.
    Ij,
    /// `MB` — the minibatch.
    Mb,
    /// `X` — a repeat dim that adds no reuse.
    X,
    /// `Y` — a kernel reuse dim.
    Y,
    /// `KIJ` — the kernel, rows and cols together.
    Kij,
    /// `I` — output image rows.
    I,
    /// `J` — output image cols.
    J,
    /// `KI` — kernel rows.
    Ki,
    /// `KJ` — kernel cols.
    Kj,
    /// `X1` — a second repeat dim.
    X1,
}

/// WHICH COMPARISON A SCHEDULE CONDITION MAKES — `CondOp` (`dsc/dscdefn.h:95-107`).
///
/// ⭐ ELEVEN, AND ONLY SIX OF THEM COMPARE. The other five are how a condition says *always*,
/// *never*, *flip*, *do not evaluate me* and *whatever the default is* — see
/// [`CondOp::cmp_predicate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondOp {
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `TOGGLE` — flip on every visit.
    Toggle,
    /// `ALWAYS`.
    Always,
    /// `NEVER`.
    Never,
    /// `CONST` — "used when the condition should not be evaluated" (`dscdefn.h:105`).
    Const,
    /// `DEFAULT`.
    Default,
}

impl CondOp {
    /// Replaces: e017_getCmpIPredicate_dup
    ///
    /// THE `arith.cmpi` PREDICATE A CONDITION LOWERS TO, or [`None`] for a condition that is not a
    /// comparison at all — the reference's `default: return LogicalResult::failure()`
    /// (`SNControlFlowLowering.cpp:41-42`), whose caller then emits no compare.
    ///
    /// ⛔ SIGNED, EVERY ONE: `LT` is `slt` and not `ult` (`:30-31`), which is the same six
    /// [`CmpIPredicate`] holds and the reason it holds six rather than MLIR's ten.
    #[must_use]
    pub const fn cmp_predicate(self) -> Option<CmpIPredicate> {
        match self {
            CondOp::Eq => Some(CmpIPredicate::Eq),
            CondOp::Ne => Some(CmpIPredicate::Ne),
            CondOp::Lt => Some(CmpIPredicate::Slt),
            CondOp::Le => Some(CmpIPredicate::Sle),
            CondOp::Gt => Some(CmpIPredicate::Sgt),
            CondOp::Ge => Some(CmpIPredicate::Sge),
            CondOp::Toggle | CondOp::Always | CondOp::Never | CondOp::Const | CondOp::Default => {
                None
            }
        }
    }
}

/// Replaces: e018_getMLIRLoopFromSNLoopNode
///
/// WHICH EMITTED LOOP CARRIES A NODE'S DIM, or [`None`] for a dim this node does not walk.
///
/// ⛔⛔ THE LOOPS ARE INDEXED IN REVERSE — dim `i` of `dims_` is loop `len - i - 1`
/// (`SNControlFlowLowering.cpp:56-58`) — and `len` is the EMITTED count, not the dim count, so the
/// reference's unchecked `record->second[size - i - 1]` reads out of bounds on a node whose dims
/// outnumber its loops. That is [`None`] here. The map lookup is mechanism and stays the caller's.
pub fn mlir_loop_from_sn_loop_node<'l, L>(
    dims: &[PrimaryDim],
    loops: &'l [L],
    dim: PrimaryDim,
) -> Option<&'l L> {
    let index = dims.iter().position(|walked| *walked == dim)?;
    loops.get(loops.len().checked_sub(index + 1)?)
}

/// A DIM OF ONE NODE AND THE SLICE RANGE ONE COMPONENT SEES OVER IT.
///
/// ⭐ THE TWO VALUES ARE `ss_` AND `el_` READ THROUGH THE COMPONENT — the start-slice and end-slice
/// data stages of `dsc_->dataStageParam_.at(numId_)`, each asked
/// `dataStageDimToVal_compView_st(dim, comp_)` (`SNControlFlowLowering.cpp:388-393`). Reaching them
/// is mechanism; that they are asked PER COMPONENT is not, which is why they are carried and not
/// recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimSlice {
    /// Which dim.
    pub dim: PrimaryDim,
    /// `ss_`'s value for it — the first slice.
    pub start: i32,
    /// `el_`'s value for it — the last slice.
    pub end: i32,
}

/// ONE ENCLOSING LOOP, AS [`parent_loop`] READS IT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ancestor<'a, L> {
    /// `parent->dims_`, in the order the node declares them.
    pub dims: &'a [DimSlice],
    /// The loops emitted for it, indexed exactly as [`mlir_loop_from_sn_loop_node`] indexes them.
    pub loops: &'a [L],
}

/// Replaces: e019_getParentLoop
///
/// THE NEAREST ENCLOSING LOOP WALKING `dim` AS A SINGLE SLICE, or [`None`] when no ancestor does.
///
/// ⛔ SINGLE SLICE IS THE WHOLE TEST: an ancestor qualifies only where its start and end slice for
/// `dim` are EQUAL (`SNControlFlowLowering.cpp:394`), so an ancestor that walks several slices of the
/// dim is passed over and the walk continues outward.
///
/// The parent chain is mechanism — `getPrev()`'s null root and the `getOwnerLoop()` walk — so
/// `ancestors` IS that chain, NEAREST FIRST, and empty for a node with no enclosing loop.
pub fn parent_loop<'l, L>(ancestors: &'l [Ancestor<'l, L>], dim: PrimaryDim) -> Option<&'l L> {
    for parent in ancestors {
        for (index, slice) in parent.dims.iter().enumerate() {
            if slice.dim == dim && slice.start == slice.end {
                return parent.loops.get(parent.loops.len().checked_sub(index + 1)?);
            }
        }
    }
    None
}

/// WHICH KIND OF SCHEDULE NODE — `ScheduleNode::NodeType` (`dsc/dsc2.h:445-456`), as far as the
/// buffer-switch propagation tells them apart.
///
/// ⭐ THE THREE THAT CARRY CHILDREN ARE EXACTLY `isBlockNode()`'s three (`dsc2.h:481-483`); every
/// other node type is a leaf the walk does not descend into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// `LOOP` — and the only kind that contributes switches of its own.
    Loop,
    /// `CONDITION`.
    Condition,
    /// `BLOCK`.
    Block,
    /// A transfer, compute, sync, allocate or stick-mask node — no children, no switches.
    Leaf,
}

/// A SCHEDULE NODE AS THE BUFFER-SWITCH PROPAGATION READS AND WRITES IT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchNode<T> {
    /// Which kind it is.
    pub kind: NodeKind,
    /// `getNextView(comp_, corelet_id_, core_id_)` — the children RELEVANT to this component,
    /// corelet and core. ⭐ ALREADY FILTERED: which children a component sees is mechanism, and the
    /// reference calls the view rather than the raw child list (`SNControlFlowLowering.cpp:524`).
    pub children: Vec<SwitchNode<T>>,
    /// `dsc_loops_to_buffers_switch_map_[loop]` — the transfers whose buffers THIS loop switches.
    /// Read only for a [`NodeKind::Loop`], because that map is keyed by `LoopNode*` (`:558-560`).
    pub own: Vec<T>,
    /// OUTPUT: `(*dsc_all_parent_loops_to_buffers_switch_map_)[node]` — every switching transfer at
    /// or below this node. ⭐ THE REAL RESULT: entry 057 discards the vector it passes in and keeps
    /// this map (`:575-589`).
    pub all: Vec<T>,
}

/// Replaces: e020_propagateBufferSwitchLoopsToRootRecursively
///
/// EVERY BUFFER-SWITCHING TRANSFER AT OR BELOW A NODE, WRITTEN ONTO THAT NODE and APPENDED to
/// `node_contrib` — appended, not assigned, which is how a parent accumulates its children's.
///
/// ⭐ THE THREE `dynamic_cast` ARMS ARE ONE ARM, and that is the reference's own note: "the code for
/// all the bodies are same" (`SNControlFlowLowering.cpp:526-529`). A child that is none of the three
/// contributes nothing and is not descended into.
///
/// ⛔ A LOOP'S OWN SWITCHES COME **AFTER** ITS CHILDREN'S, in both outputs (`:557-567`); the
/// reference's `LogicalResult` has no failing path to carry.
pub fn propagate_buffer_switch_loops_to_root_recursively<T: Clone>(
    node: &mut SwitchNode<T>,
    node_contrib: &mut Vec<T>,
) {
    let mut below: Vec<T> = Vec::new();
    for child in &mut node.children {
        let mut child_contrib: Vec<T> = Vec::new();
        match child.kind {
            NodeKind::Loop | NodeKind::Condition | NodeKind::Block => {
                propagate_buffer_switch_loops_to_root_recursively(child, &mut child_contrib);
            }
            NodeKind::Leaf => {}
        }
        below.append(&mut child_contrib);
    }
    if node.kind == NodeKind::Loop {
        below.extend(node.own.iter().cloned());
    }
    node_contrib.extend(below.iter().cloned());
    node.all.extend(below);
}

/// Replaces: e021_resetIterArguments
///
/// THE REGION ITER ARGUMENTS OF A LOOP — `getRegionIterArgs()` of whichever loop op it is, built
/// fresh, which is the reference's `iter_args.clear()` and then its push loop
/// (`SNControlFlowLowering.cpp:739-750`).
///
/// ⛔ [`None`] IS THE `llvm_unreachable("Unknown loop")`, moved where a caller cannot observe it:
/// the two `dyn_cast`s are the whole dispatch and an op that is neither loop has no iter arguments
/// to reset. Same shape as `tf_loop_unroll_for_shuffle_op::Loop::of`.
#[must_use]
pub fn reset_iter_arguments(loop_op: &DfirOp) -> Option<Vec<Val>> {
    match loop_op {
        DfirOp::Affine(affine::Op::For { carried, .. })
        | DfirOp::Scf(scf::Op::For { carried, .. }) => {
            Some(carried.iter().map(|value| value.arg).collect())
        }
        DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 058/110
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE FOUR STAGING VALUES ONE DIM'S TRIP COUNT IS DIVIDED OUT OF — `ss_` and `el_` of the
/// numerator and denominator data stages of one `LoopNode`, each read through the component with
/// `dataStageDimToVal_compView_st(dim, comp_, corelet_id_, padInfo)`.
///
/// ⛔ ONLY `den_ss` EVER DIVIDES (`SNControlFlowLowering.cpp:774,785,794,798`), which is why it
/// alone is a [`NonZeroI32`]; `den_el` is only ever subtracted and the numerators only divided.
///
/// ⛔ `std::ceil` IS DEAD CODE HERE: that reader returns `int` (`dsc/dims.h:277`), so every
/// expression it appears in has already truncated before `ceil` sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimStages {
    /// The numerator stage's `ss_`.
    pub num_ss: i32,
    /// The numerator stage's `el_`.
    pub num_el: i32,
    /// The denominator stage's `ss_` — the divisor.
    pub den_ss: NonZeroI32,
    /// The denominator stage's `el_`.
    pub den_el: i32,
}

/// THE ENCLOSING LOOP THE EPILOGUE COMPARE IS MADE AGAINST — [`parent_loop`]'s answer, in the two
/// shapes a last iteration can be read out of.
///
/// ⛔⛔ THE TWO VARIANTS ARE THE TYPE GUARD ON TWO CHECKS THAT CANNOT FIRE. The reference's
/// `DT_CHECK("Unexpected branch")` at `:809` and `:816` are handed a STRING LITERAL — always
/// truthy — so an `affine.for` without constant bounds, and a parent that is neither loop, both
/// fall through with a null induction variable and crash the builder later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParentLoop {
    /// An `affine.for` with constant bounds: the last iteration is the constant `upper_bound - 1`.
    AffineConst {
        /// `getInductionVar()`.
        iv: Val,
        /// `getConstantUpperBound()`.
        upper_bound: i64,
    },
    /// An `scf.for`: the last iteration is an `arith.subi` of its upper bound and one.
    Scf {
        /// `getInductionVar()`.
        iv: Val,
        /// `getUpperBound()`, which is an operand and not a literal.
        upper_bound: Val,
    },
}

/// A DIM'S LOOP, AND THE ARGUMENTS ITS BODY READS THE CARRIED VALUES THROUGH.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimLoop {
    /// The `affine.for` or `scf.for`, body EMPTY — filling it is the reference's
    /// `setInsertionPointToStart`, which is mechanism.
    pub loop_op: DfirOp,
    /// [`reset_iter_arguments`]' answer for it: the loop's own region arguments.
    pub iter_args: Vec<Val>,
}

/// `iter_args(%arg = %init) -> (index)` — one [`affine::Carried`] per value passed in.
fn carried_from(vals: &mut Values, iter_args: &[Val]) -> Vec<affine::Carried> {
    iter_args
        .iter()
        .map(|init| affine::Carried {
            init: *init,
            arg: vals.mint(),
            result: vals.mint(),
        })
        .collect()
}

/// `AffineForOp::create(builder, loc, 0, size, 1, iter_args)` then `resetIterArguments` — the tail
/// three of the reference's four arms share.
fn affine_dim_loop(vals: &mut Values, size: i64, iter_args: &[Val]) -> DimLoop {
    let iv = vals.mint();
    let carried = carried_from(vals, iter_args);
    let args = carried.iter().map(|value| value.arg).collect();
    DimLoop {
        loop_op: DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(size),
            carried,
            body: Vec::new(),
            dbg_name: None,
        }),
        iter_args: args,
    }
}

/// Replaces: e058_constructLoopForADim
///
/// ONE DIM'S LOOP — an `affine.for` over a constant trip count, or, where the steady-state and
/// epilogue counts differ, an `scf.for` whose bound is an `arith.select` taking `ss_iters` while the
/// parent is BELOW its last iteration (`slt`) and `el_iters` on it (`:804-828`).
///
/// ⛔ THE UNIFORM ARM'S EPILOGUE DIVIDES THE **NUMERATOR'S** `ss_` — `(num_ss - den_el) / den_ss + 1`
/// (`:785`) — where the split arm's own epilogue count uses `num_el - den_el` (`:794`).
///
/// ⚠️ `parent` IS UNREAD WHERE `num_ss == num_el`: that is the only arm which never looks it up.
pub fn construct_loop_for_a_dim(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    stages: DimStages,
    parent: ParentLoop,
    iter_args: &[Val],
) -> DimLoop {
    let num_ss = i64::from(stages.num_ss);
    let num_el = i64::from(stages.num_el);
    let den_ss = i64::from(stages.den_ss.get());
    let den_el = i64::from(stages.den_el);

    // `if (num_ss == num_el) { .. }` — one count for every iteration of the parent.
    if num_ss == num_el {
        let size = if den_ss == den_el {
            num_ss / den_ss
        } else {
            (num_ss - den_el) / den_ss + 1
        };
        return affine_dim_loop(vals, size, iter_args);
    }

    // `int ss_iters = num_ss / den_ss;  // SS + SS`
    let ss_iters = num_ss / den_ss;
    // `EL + EL` where the denominators differ, `EL + SS` where they agree.
    let el_iters = if den_ss == den_el {
        num_el / den_ss
    } else {
        (num_el - den_el) / den_ss + 1
    };

    // `} else { loop = AffineForOp::create(builder, loc, 0, ss_iters, 1, iter_args); }` — the two
    // counts agreed after all, so nothing is selected and the parent is not read.
    if ss_iters == el_iters {
        return affine_dim_loop(vals, ss_iters, iter_args);
    }

    let (parent_iv, parent_last) = match parent {
        ParentLoop::AffineConst { iv, upper_bound } => {
            (iv, constant_index(vals, ops, upper_bound - 1))
        }
        ParentLoop::Scf { iv, upper_bound } => {
            let one = constant_index(vals, ops, 1);
            let last = vals.mint();
            ops.push(DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                result: last,
                lhs: upper_bound,
                rhs: one,
                ty: ScalarTy::Index,
            })));
            (iv, last)
        }
    };

    // `cond = CmpIOp::create(.., slt, parent_loop_iv, parent_loop_last_val)` and the select over it.
    let cond = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::Compare {
        result: cond,
        predicate: CmpIPredicate::Slt,
        lhs: parent_iv,
        rhs: parent_last,
        ty: ScalarTy::Index,
    }));
    let ss_val = constant_index(vals, ops, ss_iters);
    let el_val = constant_index(vals, ops, el_iters);
    let upper = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::Select {
        result: upper,
        condition: cond,
        true_value: ss_val,
        false_value: el_val,
        ty: ScalarTy::Index,
    }));
    let lower = constant_index(vals, ops, 0);
    let step = constant_index(vals, ops, 1);

    let iv = vals.mint();
    let carried = carried_from(vals, iter_args);
    let args = carried.iter().map(|value| value.arg).collect();
    DimLoop {
        loop_op: DfirOp::Scf(scf::Op::For {
            iv,
            lo: lower,
            hi: upper,
            step,
            carried,
            body: Vec::new(),
            dbg_name: None,
        }),
        iter_args: args,
    }
}
/// WHAT A CONDITION COMPARES A LOOP'S ITERATOR AGAINST — `LoopCond::CondValType` (`dsc/dsc2.h:655`),
/// carrying the `condValInt_` that only means anything for the first (`:663`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondValType {
    /// `INT` — this literal.
    Int(i64),
    /// `FIRST` — the loop's lower bound.
    First,
    /// `LAST` — its upper bound MINUS ONE, which is the last value the iterator takes.
    Last,
}

/// ONE `AND` TERM OF A SCHEDULE CONDITION — `LoopCond` (`dsc/dsc2.h:654-673`).
///
/// ⭐ `loopComp_` AND `dim_` ARE A LOOKUP, and [`mlir_loop_from_sn_loop_node`] is that lookup
/// (`SNControlFlowLowering.cpp:92-94`), so what is carried here is its ANSWER: a term whose loop was
/// never emitted has no iterator to compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopCondition<'l> {
    /// `condOp_` — which comparison.
    pub op: CondOp,
    /// `condValType_` and `condValInt_` — what it compares against.
    pub against: CondValType,
    /// The `affine.for` or `scf.for` whose induction variable is compared.
    pub loop_op: &'l DfirOp,
}

/// A WHOLE SCHEDULE CONDITION — `LoopCondComposite` (`dsc/dsc2.h:675-677`): an OR of ANDs, with one
/// negation over the lot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondComposite<'l> {
    /// `twoLevelOrOfAnds_` — each inner list is one AND set, and the sets are OR-ed.
    pub or_of_ands: Vec<Vec<LoopCondition<'l>>>,
    /// `negated_`.
    pub negated: bool,
}

/// THE ITERATOR AND THE VALUE ONE TERM COMPARES IT AGAINST, with whatever ops that value needs
/// APPENDED to `into`.
///
/// ⛔ [`None`] IS BOTH REFUSALS: an `affine.for` whose bounds are not constant ("Unsupported dynamic
/// affine loop!", `SNControlFlowLowering.cpp:116-119`) and an op that is neither loop (`:134-136`).
///
/// ⭐ `FIRST` ON AN `scf.for` MINTS NOTHING — its lower bound is already a value (`:128`), where an
/// `affine.for`'s is a literal that needs an `arith.constant`; and `LAST` there is an `arith.subi`
/// off the upper bound rather than a fold, because that bound is an SSA value (`:122-126`).
fn compared_position(
    vals: &mut Values,
    into: &mut Vec<DfirOp>,
    loop_op: &DfirOp,
    against: CondValType,
) -> Option<(Val, Val)> {
    fn index_const(vals: &mut Values, into: &mut Vec<DfirOp>, value: i64) -> Val {
        let result = vals.mint();
        into.push(DfirOp::Arith(arith::Op::Constant { result, value }));
        result
    }

    match *loop_op {
        DfirOp::Affine(affine::Op::For { iv, lo, hi, .. }) => {
            let (affine::Bound::Const(lo), affine::Bound::Const(hi)) = (lo, hi) else {
                return None;
            };
            let value = match against {
                CondValType::Int(value) => value,
                CondValType::First => lo,
                CondValType::Last => hi - 1,
            };
            Some((iv, index_const(vals, into, value)))
        }
        DfirOp::Scf(scf::Op::For { iv, lo, hi, .. }) => match against {
            CondValType::Int(value) => Some((iv, index_const(vals, into, value))),
            CondValType::First => Some((iv, lo)),
            CondValType::Last => {
                let one = index_const(vals, into, 1);
                let result = vals.mint();
                into.push(DfirOp::Arith(arith::Op::SubI(IntBinary {
                    result,
                    lhs: hi,
                    rhs: one,
                    ty: ScalarTy::Index,
                })));
                Some((iv, result))
            }
        },
        DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
    }
}

/// ONE AND SET AS A CHAIN OF `scf.if %cmp -> (i1)`, AND THE `i1` ITS TOP BINDS.
///
/// ⛔⛔ THE CHAIN IS THE `AND`, AND THE INNERMOST LEVEL IS THE ONLY ONE THAT YIELDS `on_true`: every
/// level's `then` yields the level below it and every level's `else` yields `on_false`
/// (`SNControlFlowLowering.cpp:152-172`), so one failing term answers for the whole set.
///
/// ⭐ AND `on_false` IS WHERE THE `OR` LIVES — see [`conditional_operation`].
fn and_chain(
    vals: &mut Values,
    conditions: &[LoopCondition<'_>],
    name: Option<&str>,
    on_true: Val,
    on_false: Val,
) -> Option<(Vec<DfirOp>, Val)> {
    let (term, rest) = conditions.split_first()?;
    let mut ops = Vec::new();
    let (iterator, against) = compared_position(vals, &mut ops, term.loop_op, term.against)?;
    let cmp = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::Compare {
        result: cmp,
        // ⛔ A CONDITION THAT IS NOT A COMPARISON IS "Unsupported CondOp!" (`:140-144`).
        predicate: term.op.cmp_predicate()?,
        lhs: iterator,
        rhs: against,
        ty: ScalarTy::Index,
    }));
    // ⭐ THE RESULT IS MINTED BEFORE THE LEVEL BELOW IS BUILT, which is the reference's creation
    // order: this level's constant, its compare, its `scf.if`, then everything inside it.
    let result = vals.mint();
    let body = if rest.is_empty() {
        vec![DfirOp::Scf(scf::Op::Yield {
            operands: vec![on_true],
        })]
    } else {
        let (inner, forwarded) = and_chain(vals, rest, name, on_true, on_false)?;
        let mut body = inner;
        body.push(DfirOp::Scf(scf::Op::Yield {
            operands: vec![forwarded],
        }));
        body
    };
    ops.push(DfirOp::Scf(scf::Op::If {
        cond: cmp,
        results: vec![result],
        // ⛔ `builder.getI1Type()` (`:148-150`) — NOT `index`. See [`scf::Op::If::result_ty`].
        result_ty: ScalarTy::Int(1),
        body,
        else_body: vec![DfirOp::Scf(scf::Op::Yield {
            operands: vec![on_false],
        })],
        dbg_name: name.map(str::to_owned),
    }));
    Some((ops, result))
}

/// ONE AND SET AS NESTED RESULT-LESS `scf.if`s, `body` INNERMOST.
///
/// ⛔ NO `else` AND NOTHING BOUND: the reference builds these with `withElseRegion = false` and
/// keeps descending into the `then` region (`SNControlFlowLowering.cpp:255-274`), so the guarded body
/// runs only where every term held — the same `AND` [`and_chain`] spells with results.
fn nested_guards(
    vals: &mut Values,
    conditions: &[LoopCondition<'_>],
    name: &str,
    body: Vec<DfirOp>,
) -> Option<Vec<DfirOp>> {
    let (term, rest) = conditions.split_first()?;
    let mut ops = Vec::new();
    let (iterator, against) = compared_position(vals, &mut ops, term.loop_op, term.against)?;
    let cmp = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::Compare {
        result: cmp,
        predicate: term.op.cmp_predicate()?,
        lhs: iterator,
        rhs: against,
        ty: ScalarTy::Index,
    }));
    let inner = if rest.is_empty() {
        body
    } else {
        nested_guards(vals, rest, name, body)?
    };
    ops.push(DfirOp::Scf(scf::Op::If {
        cond: cmp,
        // ⛔ RESULT-LESS, so `result_ty` is never printed for this one.
        results: Vec::new(),
        result_ty: ScalarTy::Index,
        body: inner,
        else_body: Vec::new(),
        dbg_name: Some(name.to_owned()),
    }));
    Some(ops)
}

/// Replaces: e054_constructConditionalOperation
///
/// A SCHEDULE CONDITION AS `scf` OPS, with `then_body` and `else_body` already in the arms of the one
/// conditional the reference hands its caller a builder for — [`None`] where it fails.
///
/// ⛔⛔ TWO SHAPES, AND AN ELSE BRANCH ALONE PICKS THE FIRST. An OR, a negation or an else region
/// makes every AND set an `i1`-yielding [`and_chain`] under a final guard `scf.if`; a bare single AND
/// set becomes [`nested_guards`] instead (`SNControlFlowLowering.cpp:80`, `:197`). The bodies land in
/// the innermost `then` either way, which is why one function returns both.
pub fn conditional_operation(
    vals: &mut Values,
    name: &str,
    cond: &CondComposite<'_>,
    then_body: Vec<DfirOp>,
    else_body: Option<Vec<DfirOp>>,
) -> Option<Vec<DfirOp>> {
    if cond.or_of_ands.len() > 1 || cond.negated || else_body.is_some() {
        let mut ops = Vec::new();
        // The two `i1` constants, minted at the OUTER level before any chain (`:73-78`).
        let val_true = vals.mint();
        ops.push(DfirOp::Arith(arith::Op::ConstantInt {
            result: val_true,
            value: IntConst::Bool(true),
        }));
        let val_false = vals.mint();
        ops.push(DfirOp::Arith(arith::Op::ConstantInt {
            result: val_false,
            value: IntConst::Bool(false),
        }));
        // ⭐⭐ THE `OR` IS THE `else` OPERAND AND THERE IS NO `arith.ori` ANYWHERE. Set 0's levels
        // yield `val_false` when they fail and every later set's yield the PREVIOUS set's top result
        // — `cmp_list[cmp_list.size() - 2]` (`:165-171`) — so a set that fails hands the reader back
        // to the set before it. The tops are SIBLINGS at this level, not nested (`:87`, `:174`).
        let mut top = None;
        for and_set in &cond.or_of_ands {
            let (chain, result) = and_chain(
                vals,
                and_set,
                Some(name),
                val_true,
                top.unwrap_or(val_false),
            )?;
            ops.extend(chain);
            top = Some(result);
        }
        // ⛔ AN EMPTY OR IS THE REFERENCE'S `cmp_list[cmp_list.size() - 1]` ON AN EMPTY VECTOR
        // (`:176-177`), and so is an OR of empty AND sets.
        let top = top?;
        let guard = if cond.negated {
            // ⛔ THE NEGATION IS A COMPARE AGAINST `val_false`, not an `arith.xori` (`:179-183`).
            let negated = vals.mint();
            ops.push(DfirOp::Arith(arith::Op::Compare {
                result: negated,
                predicate: CmpIPredicate::Eq,
                lhs: top,
                rhs: val_false,
                // ⛔ `i1`, NOT `index` — both operands are the `i1`s this function built.
                ty: ScalarTy::Int(1),
            }));
            negated
        } else {
            top
        };
        ops.push(DfirOp::Scf(scf::Op::If {
            cond: guard,
            // ⛔ THE GUARD BINDS NOTHING — `IfOp::create(builder, loc, cond, has_else_branch)` is the
            // result-less overload (`:184-190`).
            results: Vec::new(),
            result_ty: ScalarTy::Index,
            body: then_body,
            // ⛔ AN ELSE BRANCH ASKED FOR AND LEFT EMPTY IS A BLOCK, NOT AN ABSENT REGION: the
            // reference creates the region and hands its builder back (`:194-196`), so `Some(vec![])`
            // becomes a region holding only its terminator — see [`scf::Op::If::else_body`].
            else_body: match else_body {
                None => Vec::new(),
                Some(body) if body.is_empty() => vec![DfirOp::Scf(scf::Op::Yield {
                    operands: Vec::new(),
                })],
                Some(body) => body,
            },
            dbg_name: Some(name.to_owned()),
        }));
        Some(ops)
    } else {
        // ⛔ AND THE TWO CONSTANTS ARE NOT EMITTED HERE, WHICH IS A DELIBERATE DIVERGENCE. The
        // reference mints them before it knows which shape it is building (`:73-78`) and this shape
        // reads neither, leaving two dangling `arith.constant`s in the region — the class of op
        // `dbo-opt` refuses with "Dangling non-compute op has no use".
        nested_guards(vals, cond.or_of_ands.first()?, name, then_body)
    }
}

/// THE NUMERATOR AND DENOMINATOR DATA STAGES OF A LOOP — `numId_` and `denId_`
/// (`dsc/dsc2.h:573-574`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePair {
    /// `numId_`.
    pub num: i32,
    /// `denId_`.
    pub den: i32,
}

/// ONE LOOP OF A UNIT VIEW — `ScheduleNode::UnitView::LoopInfo` (`dsc/dsc2.h:500-505`) as far as the
/// SAMV filter reads it, with its emitted loop resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamvLoop<'l> {
    /// `dim_` — [`None`] for the reference's unset `PrimaryDimTypesCount`, as in [`PrimaryDim`].
    pub dim: Option<PrimaryDim>,
    /// `loop_->numId_` and `loop_->denId_`.
    pub stage: StagePair,
    /// The loop emitted for this node's `OUT` dim.
    pub loop_op: &'l DfirOp,
}

/// Replaces: e055_constructConditionalsForSAMV
///
/// THE `AND` OF EVERY `OUT` LOOP BEING ON ITS LAST ITERATION, and the `i1` the OUTERMOST if binds —
/// the guard a SUM/MAX stick's accumulated value is written out under.
///
/// ⛔⛔ [`None`] COVERS THE REFERENCE'S NULL `if_op` AS WELL AS ITS FAILURES. With no `OUT` loop it
/// reports SUCCESS having assigned `if_op = nullptr` (`SNControlFlowLowering.cpp:299-302`), and its
/// caller dereferences `if_op->getResult(0)` on both paths (`SNTransferLowering.cpp:1957-1982`) — so
/// the null is a crash there and not a conditional-free transfer.
pub fn conditionals_for_samv(
    vals: &mut Values,
    loops: &[SamvLoop<'_>],
) -> Option<(Vec<DfirOp>, Val)> {
    // ⭐ THE FILTER IS THE `OUT` DIM MINUS ONE STAGE PAIR, which the reference excludes by hand and
    // without a word (`:292-296`).
    let out_loops: Vec<LoopCondition<'_>> = loops
        .iter()
        .filter(|walked| {
            walked.dim == Some(PrimaryDim::Out) && !(walked.stage.num == 0 && walked.stage.den == 1)
        })
        .map(|walked| LoopCondition {
            // ⭐ ALWAYS `eq` AGAINST `LAST`: "create AND condition covering for all the last
            // iterations" (`:303-304`).
            op: CondOp::Eq,
            against: CondValType::Last,
            loop_op: walked.loop_op,
        })
        .collect();
    // ⛔ THE EMPTY TEST COMES BEFORE THE TWO CONSTANTS (`:299`), so a stick with no `OUT` loop of its
    // own mints nothing at all.
    if out_loops.is_empty() {
        return None;
    }

    let mut ops = Vec::new();
    let val_true = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::ConstantInt {
        result: val_true,
        value: IntConst::Bool(true),
    }));
    let val_false = vals.mint();
    ops.push(DfirOp::Arith(arith::Op::ConstantInt {
        result: val_false,
        value: IntConst::Bool(false),
    }));
    // ⛔ ONE AND SET, NO `dbgName`, AND NO GUARD `scf.if` AROUND IT — every `else` yields `val_false`
    // because there is no earlier set to fall back to (`:365-367`), and the value the caller wants is
    // the OUTERMOST if's (`:352-354`), not an innermost one.
    let (chain, top) = and_chain(vals, &out_loops, None, val_true, val_false)?;
    ops.extend(chain);
    Some((ops, top))
}

/// HOW MANY BUFFERS AN ALLOCATION HAS — `AllocateNode::numBuffers_`, whose `-1` is not a count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumBuffers {
    /// `-1` — the allocation STREAMS instead of switching between buffers.
    Streaming,
    /// Any other value — that many buffers to switch between.
    Count(i32),
}

/// WHICH OF THE TWO A TRANSFER'S BUFFERS DO — the reference's `mode`, minus its `-1`, which is the
/// [`None`] of [`buffering_or_streaming_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchMode {
    /// `mode = 1`.
    Buffering,
    /// `mode = 2`.
    Streaming,
}

/// ONE END OF A TRANSFER, AS THE BUFFER-SWITCH TEST READS IT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchSide<'a, M> {
    /// `src_.unit_` / `via.loc_.unit_` — the unit this end sits on, which is what `comp_` is compared
    /// against (`SNControlFlowLowering.cpp:416`, `:438`).
    pub unit: DfirUnit,
    /// `{comp_, storage_}` as the granularity table keys it. ⛔ NOT DERIVABLE FROM `unit`: the table
    /// has ONE `PT` row against eight PT rows here, so the pair and the unit are two reads of one
    /// fact and neither implies the other.
    pub loc: DataLocation,
    /// `bufferSwitchPosition_`, and the `numBuffers_` of the allocation behind it — [`None`] for an
    /// end whose buffers do not switch, which is the whole test (`:417`, `:441`).
    pub switches: Option<NumBuffers>,
    /// `startAddr_` — the fold manager this end's addresses come out of.
    pub start_addresses: &'a M,
}

/// WHAT A SWITCHING END ANSWERS WITH — the reference's `mode`, `start_address_map` and `factor`, set
/// together or not at all (`SNControlFlowLowering.cpp:421-430`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferSwitch<'a, M> {
    /// Buffering or streaming.
    pub mode: SwitchMode,
    /// The fold manager the switching end's start addresses come out of.
    pub start_addresses: &'a M,
    /// What a DSC address of that end is multiplied by.
    pub factor: Factor,
}

/// Replaces: e056_getBufferingOrStreamingMode
///
/// THE FIRST END OF A TRANSFER ON THIS COMPONENT WHOSE BUFFERS SWITCH, or [`None`] for "neither
/// buffering or streaming" — the reference's `mode = -1` (`SNControlFlowLowering.cpp:409`).
///
/// ⛔⛔ THE SOURCE IS ASKED FIRST AND A SOURCE ON THIS COMPONENT THAT DOES NOT SWITCH FALLS THROUGH
/// to the destinations (`:415-435`), so the answer is the first SWITCHING end and not the first end
/// found here. `precision` is the SOURCE lds's `dataFormat_` at BOTH — the reference reads its
/// element type once, above the branch, and the destination loop reuses it (`:411-419`).
pub fn buffering_or_streaming_mode<'a, M>(
    comp: DfirUnit,
    precision: DataType,
    src: &SwitchSide<'a, M>,
    dsts: &[SwitchSide<'a, M>],
) -> Option<BufferSwitch<'a, M>> {
    // ⭐ ONE SIDE PER `dstVias_` INDEX: the reference reads `dstVias_[i]` and
    // `dstLdsAndLoopOffsets_[i]` at one index (`:437-448`), and a pairing that can slip is one a type
    // should hold instead.
    for side in std::iter::once(src).chain(dsts) {
        if side.unit != comp {
            continue;
        }
        let Some(buffers) = side.switches else {
            continue;
        };
        return Some(BufferSwitch {
            mode: match buffers {
                NumBuffers::Streaming => SwitchMode::Streaming,
                NumBuffers::Count(_) => SwitchMode::Buffering,
            },
            start_addresses: side.start_addresses,
            factor: address_granularity_multiply_factor(side.loc, precision),
        });
    }
    None
}

/// Replaces: e057_propagateBufferSwitchLoopsToRoot
///
/// EVERY SCHEDULE NODE'S SWITCHING TRANSFERS, WRITTEN ONTO IT — the driver over the schedule head's
/// own children, whose returned vectors are DISCARDED because the result is each node's
/// [`SwitchNode::all`] (`SNControlFlowLowering.cpp:575-589`).
///
/// ⛔ LOOPS ONLY AT THE TOP, unlike the three kinds the recursion descends through: the head's
/// `dynamic_cast<const dsc2::LoopNode *>` (`:578`) skips a `CONDITION` or `BLOCK` child of the head
/// outright, and with it everything below it. The head itself is never visited, which is why this
/// takes the children.
pub fn propagate_buffer_switch_loops_to_root<T: Clone>(head_children: &mut [SwitchNode<T>]) {
    for child in head_children {
        if child.kind == NodeKind::Loop {
            let mut discarded: Vec<T> = Vec::new();
            propagate_buffer_switch_loops_to_root_recursively(child, &mut discarded);
        }
    }
}
/// WHICH TRANSFER — the identity a `const dsc2::TransferNode *` is compared by (`:657`), which is its
/// position in the component's own transfer list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferId(pub usize);

/// A TRANSFER AS THE BUFFER-LOOP SCAN READS IT — what entry 056 needs, plus the two
/// `bufferSwitchPosition_` a match on this component reads.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchingTransfer<'a, L, M> {
    /// The source lds's `dataFormat_`, which entry 056 reads the granularity factor at.
    pub precision: DataType,
    /// `src_`.
    pub src: SwitchSide<'a, M>,
    /// `srcLdsAndLoopOffsets_.bufferSwitchPosition_` — the loop whose iteration switches its buffers.
    pub src_position: Option<&'a L>,
    /// `dstVias_`, in index order.
    pub dsts: Vec<SwitchSide<'a, M>>,
    /// `dstLdsAndLoopOffsets_[0].bufferSwitchPosition_` — index ZERO, for every via.
    pub dst_position: Option<&'a L>,
}

/// Replaces: e074_getBlockingOrStreamingBufferLoopLocations
///
/// **074/110** `SNControlFlowLowering.cpp:467` — `dsc_loops_to_buffers_switch_map_`'s pairs: which
/// loop switches which of this component's transfers, in the DFS order the walk produces them.
///
/// ⛔ THE SIDE TEST HERE IS `unit_ == comp_` ALONE, not entry 056's "and switches" — a source on this
/// component that does NOT switch still contributes the SOURCE's position (`:485-487`).
/// ⛔ AND EVERY MATCHING VIA READS `dstLdsAndLoopOffsets_[0]`: the `dst_idx++` sits outside its own
/// loop and is never read (`:488-497`), so via *n* takes via 0's position.
pub fn blocking_or_streaming_buffer_loop_locations<'a, L, M>(
    comp: DfirUnit,
    transfers: &'a [SwitchingTransfer<'a, L, M>],
) -> Vec<(&'a L, TransferId)> {
    let mut located = Vec::new();
    for (index, transfer) in transfers.iter().enumerate() {
        // `if (mode == 1 || mode == 2)` — entry 056's [`None`] is the reference's `mode = -1`, whose
        // `failed(..)` arm cannot be taken because the mode is an answer and not a refusal.
        if buffering_or_streaming_mode(comp, transfer.precision, &transfer.src, &transfer.dsts)
            .is_none()
        {
            continue;
        }
        // `consider_transfer` and `buffer_position`, which are set together or not at all.
        let position = if transfer.src.unit == comp {
            transfer.src_position
        } else if transfer.dsts.iter().any(|via| via.unit == comp) {
            transfer.dst_position
        } else {
            None
        };
        if let Some(position) = position {
            located.push((position, TransferId(index)));
        }
    }
    located
}

/// THE START ADDRESSES ONE SWITCHING TRANSFER CONTRIBUTES — `*start_address_map`, read the two ways
/// entry 075 reads it (`:613-627`).
pub struct StartAddresses<'a> {
    /// Every `(core, corelet, fold)`'s address, for the all-the-same test entry 030 makes.
    pub all: &'a [i64],
    /// `startAddr_` at one `(core, corelet, fold)`, which the uniform arm names once per handle.
    pub at: &'a dyn Fn(Core, Corelet, u32) -> i64,
    /// `getSingleDataStrict(*start_address_map, {{0, core_id_}, {1, corelet_id_}})` — this core and
    /// corelet's one address across folds, which is what the non-uniform arm scales.
    pub single: i64,
}

/// ONE SWITCHING TRANSFER OF A ROOT LOOP, AS ITS ITER ARG IS BUILT.
pub struct RootIterArg<'a> {
    /// `mode` — [`None`] is the reference's `llvm_unreachable("Unknown buffering mode")`.
    pub mode: Option<SwitchMode>,
    /// `factor`.
    pub factor: Factor,
    /// `*start_address_map`.
    pub addresses: StartAddresses<'a>,
}

/// WHICH ITER ARGS A LOOP TAKES — the reference's `parent_node == nullptr && dim_idx_band == 0`
/// against everything else (`:600`), which are two different answers and not two paths to one.
pub enum LoopIterArgs<'a> {
    /// The outermost loop of a band with no enclosing loop: its iter args are FRESH start addresses.
    Root {
        /// `needsUniform()` — the handle set the uniform arm maps over, or [`None`] for the constant.
        uniform: Option<&'a Handles>,
        /// `(*dsc_all_parent_loops_to_buffers_switch_map_)[node]`, with entry 056 already asked.
        transfers: Vec<RootIterArg<'a>>,
    },
    /// Every other loop: its iter args are the ENCLOSING loop's, at each transfer's index there.
    Enclosing {
        /// `(*dsc_all_parent_loops_to_buffers_switch_map_)[node]`.
        own: &'a [TransferId],
        /// `(*dsc_all_parent_loops_to_buffers_switch_map_)[parent_node_loop]`, whose INDEX of a
        /// transfer is that transfer's iter arg position.
        parent_transfers: &'a [TransferId],
        /// `(*dsc_loops_to_mlir_loops_map_)[parent_node_loop]` — the band, indexed as
        /// [`mlir_loop_from_sn_loop_node`] indexes it.
        parent_loops: &'a [DfirOp],
        /// `dim_idx_band`: `0` takes the band's LAST loop, and any other takes `dim_idx_band - 1`
        /// because the node is then its own parent (`:640-673`).
        dim_idx_band: usize,
    },
}

/// Replaces: e075_constructLoopIterArgs
///
/// **075/110** `SNControlFlowLowering.cpp:595` — a loop's `iter_args`: one start address per switching
/// transfer at the root, and the enclosing loop's own region iter args below it.
///
/// ⛔ THE ROOT ARM EMITS, THE OTHER ARM ONLY NAMES — a root loop's args are new `arith.constant`s or a
/// new uniform mapping, so calling it twice for one loop emits the addresses twice.
/// ⛔ THE `index == -1` FAILURE AND THE `llvm_unreachable` ARE BOTH SKIPS HERE: entry 020 appends a
/// child's transfers to its parent's list, and this map holds only switching transfers.
pub fn construct_loop_iter_args(
    vals: &mut Values,
    ops: &mut Vec<DfirOp>,
    args: &LoopIterArgs<'_>,
) -> Vec<Val> {
    let mut iter_args = Vec::new();
    match args {
        LoopIterArgs::Root { uniform, transfers } => {
            for transfer in transfers {
                if transfer.mode.is_none() {
                    continue;
                }
                let address = match uniform {
                    // `constructUniformizedFoldedAddress(builder, *start_address_map, factor)`.
                    Some(handles) => uniformized_folded_address(
                        vals,
                        ops,
                        handles,
                        transfer.addresses.all,
                        transfer.addresses.at,
                        transfer.factor,
                    ),
                    // `int(getSingleDataStrict(..) * factor)` — "expecting constant value across
                    // sdsc folds".
                    None => {
                        constant_index(vals, ops, transfer.factor.scale(transfer.addresses.single))
                    }
                };
                iter_args.push(address);
            }
        }
        LoopIterArgs::Enclosing {
            own,
            parent_transfers,
            parent_loops,
            dim_idx_band,
        } => {
            let parent_for = match dim_idx_band.checked_sub(1) {
                Some(previous) => parent_loops.get(previous),
                None => parent_loops.last(),
            };
            // `if (parent_node_loop)` — and `getRegionIterArgs()` of whichever loop op it is.
            let Some(carried) = parent_for.and_then(reset_iter_arguments) else {
                return iter_args;
            };
            for id in *own {
                let Some(index) = parent_transfers.iter().position(|other| other == id) else {
                    continue;
                };
                if let Some(&arg) = carried.get(index) {
                    iter_args.push(arg);
                }
            }
        }
    }
    iter_args
}

// crustify:todo: e088_constructLoopsRecursive
// crustify:todo: e096_constructLoops

#[cfg(test)]
mod unit_tests {
    use std::num::NonZeroI32;

    use super::super::dsc_lowering::{DataLocation, address_granularity_multiply_factor};
    use super::{
        Ancestor, BufferSwitch, CondComposite, CondOp, CondValType, DimSlice, DimStages,
        LoopCondition, LoopIterArgs, NodeKind, NumBuffers, ParentLoop, PrimaryDim, RootIterArg,
        SamvLoop, StagePair, StartAddresses, SwitchMode, SwitchNode, SwitchSide, SwitchingTransfer,
        TransferId, blocking_or_streaming_buffer_loop_locations, buffering_or_streaming_mode,
        conditional_operation, conditionals_for_samv, construct_loop_for_a_dim,
        construct_loop_iter_args, mlir_loop_from_sn_loop_node, parent_loop,
        propagate_buffer_switch_loops_to_root, propagate_buffer_switch_loops_to_root_recursively,
        reset_iter_arguments,
    };
    use crate::generated::DataType;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::arith::CmpIPredicate;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith, scf};
    use crate::islands::dataflow_ir::print::emit;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::units::{Core, Corelet, DfirUnit};

    /// 🎯 074/110 — ⛔⛔ A SOURCE ON THIS COMPONENT CONTRIBUTES ITS OWN POSITION EVEN WHEN A VIA IS
    /// WHAT SWITCHES: the mode comes from entry 056's first switching end, the POSITION from a plain
    /// `unit_ == comp_`, and reading the two off one end would key the map by the wrong loop.
    #[test]
    fn the_position_follows_the_unit_test_and_a_via_reads_index_zero() {
        let precision = DataType::Sen169Fp16;
        let comp = DfirUnit::Lxlu;
        let on_comp = |switches| SwitchSide {
            unit: comp,
            loc: DataLocation::LxluLx,
            switches,
            start_addresses: &"lxlu",
        };
        let elsewhere = SwitchSide {
            unit: DfirUnit::L3lu,
            loc: DataLocation::L3luHbm,
            switches: Some(NumBuffers::Count(4)),
            start_addresses: &"l3lu",
        };
        let transfers = vec![
            SwitchingTransfer {
                precision,
                src: on_comp(Some(NumBuffers::Count(2))),
                src_position: Some(&"outer"),
                dsts: vec![elsewhere],
                dst_position: Some(&"inner"),
            },
            SwitchingTransfer {
                precision,
                src: on_comp(None),
                src_position: Some(&"outer"),
                dsts: vec![on_comp(Some(NumBuffers::Count(2)))],
                dst_position: Some(&"inner"),
            },
            SwitchingTransfer {
                precision,
                src: elsewhere,
                src_position: Some(&"never read"),
                dsts: vec![elsewhere, on_comp(Some(NumBuffers::Streaming))],
                dst_position: Some(&"inner"),
            },
            // ⛔ NOTHING ON THIS COMPONENT SWITCHES — no entry at all, and the reference's
            // `emitError` arm is unreachable because `mode = -1` is an answer.
            SwitchingTransfer {
                precision,
                src: on_comp(None),
                src_position: Some(&"outer"),
                dsts: vec![elsewhere],
                dst_position: None,
            },
        ];
        assert_eq!(
            blocking_or_streaming_buffer_loop_locations(comp, &transfers),
            vec![
                (&"outer", TransferId(0)),
                (&"outer", TransferId(1)),
                (&"inner", TransferId(2)),
            ]
        );
    }

    /// 🎯 075/110 — ⭐ THE ROOT ARM EMITS ONE ADDRESS PER SWITCHING TRANSFER while every other loop
    /// only NAMES its parent's region iter args at that transfer's index in the PARENT's list.
    #[test]
    fn a_root_loop_emits_its_addresses_and_an_inner_loop_names_its_parents() {
        let mut vals = Values::default();
        let mut ops = Vec::new();
        // ⭐ FOUR BYTES PER STEP OVER A 16-BIT ELEMENT IS A FACTOR OF TWO.
        let factor =
            address_granularity_multiply_factor(DataLocation::L3luIbr, DataType::Sen169Fp16);
        let all = [1000_i64];
        let at = |_: Core, _: Corelet, _: u32| 1000;
        let arg = |mode, single| RootIterArg {
            mode,
            factor,
            addresses: StartAddresses {
                all: &all,
                at: &at,
                single,
            },
        };
        let root = LoopIterArgs::Root {
            uniform: None,
            transfers: vec![
                arg(Some(SwitchMode::Buffering), 1000),
                arg(Some(SwitchMode::Streaming), 7),
                // ⛔ THE `llvm_unreachable("Unknown buffering mode")`, as a skip.
                arg(None, 5),
            ],
        };
        assert_eq!(
            construct_loop_iter_args(&mut vals, &mut ops, &root),
            vec![Val(0), Val(1)]
        );
        assert_eq!(
            ops,
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(0),
                    value: 2000,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(1),
                    value: 14,
                }),
            ]
        );

        // ⛔ THE BAND'S LAST LOOP AT `dim_idx_band == 0`, AND `dim_idx_band - 1` ABOVE IT.
        let band = |iv, arg| {
            DfirOp::Scf(scf::Op::For {
                iv,
                lo: Val(30),
                hi: Val(31),
                step: Val(32),
                carried: vec![
                    affine::Carried {
                        init: Val(40),
                        arg,
                        result: Val(42),
                    },
                    affine::Carried {
                        init: Val(43),
                        arg: Val(44),
                        result: Val(45),
                    },
                ],
                body: Vec::new(),
                dbg_name: None,
            })
        };
        let parent_loops = [band(Val(20), Val(21)), band(Val(22), Val(23))];
        let mut ops = Vec::new();
        let inner = LoopIterArgs::Enclosing {
            // ⛔ TRANSFER 9 IS NOT IN THE PARENT'S LIST, and the reference's `index == -1` refusal
            // for it is unreachable by construction — a child's transfers are appended to its
            // parent's (entry 020), so here it is a skip.
            own: &[TransferId(7), TransferId(9), TransferId(4)],
            parent_transfers: &[TransferId(4), TransferId(7)],
            parent_loops: &parent_loops,
            dim_idx_band: 0,
        };
        assert_eq!(
            construct_loop_iter_args(&mut vals, &mut ops, &inner),
            vec![Val(44), Val(23)]
        );
        let banded = LoopIterArgs::Enclosing {
            own: &[TransferId(4)],
            parent_transfers: &[TransferId(4)],
            parent_loops: &parent_loops,
            dim_idx_band: 1,
        };
        assert_eq!(
            construct_loop_iter_args(&mut vals, &mut ops, &banded),
            vec![Val(21)]
        );
        assert!(ops.is_empty(), "this arm names values and emits nothing");
    }

    /// ⛔⛔ SIGNED, AND THE FIVE NON-COMPARISONS ARE NOT PREDICATES AT ALL.
    ///
    /// A condition lowered with `ult` where `slt` belongs compares two negative offsets the wrong
    /// way round and still parses, so the six are checked by name; and `ALWAYS`/`NEVER`/`TOGGLE`/
    /// `CONST`/`DEFAULT` reaching a predicate at all would emit an `arith.cmpi` for a condition the
    /// reference refuses to build one for (`SNControlFlowLowering.cpp:41-42`).
    #[test]
    fn six_signed_predicates_and_five_conditions_that_are_not_comparisons() {
        assert_eq!(
            [
                CondOp::Eq.cmp_predicate(),
                CondOp::Ne.cmp_predicate(),
                CondOp::Lt.cmp_predicate(),
                CondOp::Le.cmp_predicate(),
                CondOp::Gt.cmp_predicate(),
                CondOp::Ge.cmp_predicate(),
            ],
            [
                Some(CmpIPredicate::Eq),
                Some(CmpIPredicate::Ne),
                Some(CmpIPredicate::Slt),
                Some(CmpIPredicate::Sle),
                Some(CmpIPredicate::Sgt),
                Some(CmpIPredicate::Sge),
            ],
        );
        for not_a_comparison in [
            CondOp::Toggle,
            CondOp::Always,
            CondOp::Never,
            CondOp::Const,
            CondOp::Default,
        ] {
            assert_eq!(not_a_comparison.cmp_predicate(), None);
        }
    }

    /// ⛔⛔ THE FIRST DIM IS THE **LAST** LOOP, and a node walking one dim cannot show it.
    ///
    /// `dims_` is declared outermost first while the loop vector is appended to as the nest is
    /// built, so `IN`, the outermost dim, is the loop at index `len - 1`
    /// (`SNControlFlowLowering.cpp:56-58`). Read forward, every dim of a three-deep node but the
    /// middle one pairs with the wrong loop — which is a legal program addressing the wrong axis.
    #[test]
    fn a_dim_reads_the_loop_at_its_mirrored_index() {
        let dims = [PrimaryDim::In, PrimaryDim::Out, PrimaryDim::Ij];
        let loops = ["innermost", "middle", "outermost"];

        assert_eq!(
            [
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::In),
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Out),
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Ij),
                mlir_loop_from_sn_loop_node(&dims, &loops, PrimaryDim::Mb),
            ],
            [
                Some(&"outermost"),
                Some(&"middle"),
                Some(&"innermost"),
                None,
            ],
        );
        // ⛔ AND A DIM DEEPER THAN THE LOOPS EMITTED IS THE REFERENCE'S OUT-OF-BOUNDS READ, not a
        // wrong loop: `loops.size() - index - 1` underflows `size_t` there and indexes the vector
        // with it, while the outermost dim still reads loop 0 in both.
        assert_eq!(
            [
                mlir_loop_from_sn_loop_node(&dims, &loops[..1], PrimaryDim::In),
                mlir_loop_from_sn_loop_node(&dims, &loops[..1], PrimaryDim::Ij),
            ],
            [Some(&"innermost"), None],
        );
    }

    /// ⛔⛔ AN ANCESTOR WALKING SEVERAL SLICES OF THE DIM IS PASSED OVER, NOT ANSWERED WITH.
    ///
    /// The test is `ss_ == el_` for the dim on this component (`SNControlFlowLowering.cpp:394`): the
    /// nearest ancestor here declares `MB` across three slices and must be skipped in favour of the
    /// outer one that pins it, and the loop returned is that outer node's mirrored index. Answering
    /// with the nearest match would hoist a value to a loop that changes it.
    #[test]
    fn the_nearest_single_slice_ancestor_wins_and_a_multi_slice_one_is_skipped() {
        let near_dims = [DimSlice {
            dim: PrimaryDim::Mb,
            start: 0,
            end: 2,
        }];
        let far_dims = [
            DimSlice {
                dim: PrimaryDim::Mb,
                start: 4,
                end: 4,
            },
            DimSlice {
                dim: PrimaryDim::In,
                start: 0,
                end: 7,
            },
        ];
        let near_loops = ["near-mb"];
        let far_loops = ["far-in", "far-mb"];
        let ancestors = [
            Ancestor {
                dims: &near_dims,
                loops: &near_loops,
            },
            Ancestor {
                dims: &far_dims,
                loops: &far_loops,
            },
        ];

        assert_eq!(parent_loop(&ancestors, PrimaryDim::Mb), Some(&"far-mb"));
        assert_eq!(parent_loop(&ancestors, PrimaryDim::In), None, "0 != 7");
        assert_eq!(parent_loop::<u32>(&[], PrimaryDim::Mb), None, "no parent");
    }

    /// ⛔⛔ A LOOP'S OWN SWITCHES COME AFTER ITS CHILDREN'S, AND BOTH OUTPUTS SEE THE SAME ORDER.
    ///
    /// The reference pushes each child's contribution into the caller's vector AND the node's map
    /// entry, then its own (`SNControlFlowLowering.cpp:551-567`), so a node's entry reads
    /// inner-to-outer. It is also the whole point that the map is written at EVERY level: the inner
    /// loop's own entry holds only what is at or below it, and reversing the two appends would make a
    /// parent switch a buffer before the child that fills it.
    #[test]
    fn a_loops_own_switches_follow_its_childrens_at_every_level() {
        let mut root = SwitchNode {
            kind: NodeKind::Loop,
            children: vec![
                SwitchNode {
                    kind: NodeKind::Block,
                    children: vec![SwitchNode {
                        kind: NodeKind::Loop,
                        children: Vec::new(),
                        own: vec!["inner"],
                        all: Vec::new(),
                    }],
                    own: vec!["a block is never asked for its own"],
                    all: Vec::new(),
                },
                SwitchNode {
                    kind: NodeKind::Leaf,
                    children: vec![SwitchNode {
                        kind: NodeKind::Loop,
                        children: Vec::new(),
                        own: vec!["below a leaf"],
                        all: Vec::new(),
                    }],
                    own: vec!["nor is a leaf"],
                    all: Vec::new(),
                },
            ],
            own: vec!["root"],
            all: Vec::new(),
        };
        let mut node_contrib = vec!["what the caller already had"];

        propagate_buffer_switch_loops_to_root_recursively(&mut root, &mut node_contrib);

        assert_eq!(
            node_contrib,
            vec!["what the caller already had", "inner", "root"],
            "appended, and the loop's own last",
        );
        assert_eq!(root.all, vec!["inner", "root"]);
        assert_eq!(
            root.children[0].all,
            vec!["inner"],
            "the block sees only below"
        );
        assert_eq!(root.children[0].children[0].all, vec!["inner"]);
        assert_eq!(
            root.children[1].all,
            Vec::<&str>::new(),
            "a leaf is not descended into",
        );
    }

    /// ⭐ BOTH LOOP OPS ANSWER WITH THEIR REGION ARGUMENTS, AND IT IS THE `arg` OF THE THREE.
    ///
    /// `getRegionIterArgs()` is the name the BODY reads the carried value through, not the init it
    /// starts from nor the result the loop binds — handing back an init would build a body that reads
    /// a value defined outside the region it is verified in. An op that is neither loop has no iter
    /// arguments, which is where the reference's `llvm_unreachable` went.
    #[test]
    fn both_loops_reset_to_their_region_arguments() {
        let carried = vec![
            affine::Carried {
                init: Val(1),
                arg: Val(2),
                result: Val(3),
            },
            affine::Carried {
                init: Val(4),
                arg: Val(5),
                result: Val(6),
            },
        ];
        let affine_for = DfirOp::Affine(affine::Op::For {
            iv: Val(0),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(8),
            carried: carried.clone(),
            body: Vec::new(),
            dbg_name: None,
        });
        let scf_for = DfirOp::Scf(scf::Op::For {
            iv: Val(0),
            lo: Val(10),
            hi: Val(11),
            step: Val(12),
            carried,
            body: Vec::new(),
            dbg_name: None,
        });
        let not_a_loop = DfirOp::Scf(scf::Op::Yield {
            operands: vec![Val(2)],
        });

        assert_eq!(
            reset_iter_arguments(&affine_for),
            Some(vec![Val(2), Val(5)])
        );
        assert_eq!(reset_iter_arguments(&scf_for), Some(vec![Val(2), Val(5)]));
        assert_eq!(reset_iter_arguments(&not_a_loop), None);
    }

    /// ⛔ THE EPILOGUE COUNT DIVIDES THE NUMERATOR'S `ss_` IN ONE ARM AND ITS `el_` IN THE OTHER,
    /// and only the second arm selects between them.
    ///
    /// A uniform dim whose denominators differ is ONE `affine.for` sized `(num_ss - den_el)/den_ss + 1`
    /// with no compare at all; a dim whose numerators differ is an `scf.for` whose bound is a select
    /// taking the steady-state count while the parent is `slt` its last iteration.
    #[test]
    fn a_split_dim_selects_its_bound_and_a_uniform_one_is_a_constant_affine_loop() {
        let mut vals = Values::default();
        let mut ops: Vec<DfirOp> = Vec::new();
        let parent = ParentLoop::Scf {
            iv: Val(100),
            upper_bound: Val(101),
        };

        // `num_ss == num_el`, denominators differ: (9 - 3) / 2 + 1 = 4, and nothing is emitted.
        let uniform = construct_loop_for_a_dim(
            &mut vals,
            &mut ops,
            DimStages {
                num_ss: 9,
                num_el: 9,
                den_ss: NonZeroI32::new(2).expect("two"),
                den_el: 3,
            },
            parent,
            &[Val(50)],
        );
        assert!(ops.is_empty());
        assert_eq!(
            uniform.loop_op,
            DfirOp::Affine(affine::Op::For {
                iv: Val(0),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: vec![affine::Carried {
                    init: Val(50),
                    arg: Val(1),
                    result: Val(2),
                }],
                body: Vec::new(),
                dbg_name: None,
            })
        );
        assert_eq!(uniform.iter_args, vec![Val(1)]);

        // `num_ss != num_el` with equal denominators: 8/2 = 4 against 6/2 = 3, so a select.
        let split = construct_loop_for_a_dim(
            &mut vals,
            &mut ops,
            DimStages {
                num_ss: 8,
                num_el: 6,
                den_ss: NonZeroI32::new(2).expect("two"),
                den_el: 2,
            },
            parent,
            &[Val(50)],
        );
        assert_eq!(
            ops,
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(3),
                    value: 1,
                }),
                DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                    result: Val(4),
                    lhs: Val(101),
                    rhs: Val(3),
                    ty: ScalarTy::Index,
                })),
                DfirOp::Arith(arith::Op::Compare {
                    result: Val(5),
                    predicate: CmpIPredicate::Slt,
                    lhs: Val(100),
                    rhs: Val(4),
                    ty: ScalarTy::Index,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(6),
                    value: 4,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(7),
                    value: 3,
                }),
                DfirOp::Arith(arith::Op::Select {
                    result: Val(8),
                    condition: Val(5),
                    true_value: Val(6),
                    false_value: Val(7),
                    ty: ScalarTy::Index,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(9),
                    value: 0,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(10),
                    value: 1,
                }),
            ]
        );
        assert_eq!(
            split.loop_op,
            DfirOp::Scf(scf::Op::For {
                iv: Val(11),
                lo: Val(9),
                hi: Val(8),
                step: Val(10),
                carried: vec![affine::Carried {
                    init: Val(50),
                    arg: Val(12),
                    result: Val(13),
                }],
                body: Vec::new(),
                dbg_name: None,
            })
        );
        assert_eq!(split.iter_args, vec![Val(12)]);
    }

    /// ⛔⛔ THE OR IS THE `else` OPERAND, THE NEGATION IS A COMPARE, AND THE `i1` CHAIN IS TYPED `i1`.
    ///
    /// Every value here is load-bearing: set 1's `else` yielding `%1` instead of set 0's top `%4`
    /// would make the two sets an AND; the negation printed on `index` operands is "use of value
    /// expects different type"; and a chain stated `-> (index)` while yielding `arith.constant true`
    /// is the same refusal one rung down. The bare single set takes the OTHER shape entirely — nested
    /// result-less ifs, and NOT the two dangling constants the reference leaves there.
    #[test]
    fn an_or_chains_through_the_else_and_a_bare_and_set_nests_instead() {
        let by_four = DfirOp::Affine(affine::Op::For {
            iv: Val(100),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });
        let dynamic = DfirOp::Scf(scf::Op::For {
            iv: Val(200),
            lo: Val(201),
            hi: Val(202),
            step: Val(203),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });
        let marker = || {
            vec![DfirOp::Arith(arith::Op::Constant {
                result: Val(300),
                value: 7,
            })]
        };
        let text = |ops: &[DfirOp]| {
            let mut out = String::new();
            for op in ops {
                emit(&mut out, op, 0);
            }
            out
        };

        let mut vals = Values::default();
        let or_of_two = CondComposite {
            or_of_ands: vec![
                vec![
                    LoopCondition {
                        op: CondOp::Eq,
                        against: CondValType::Last,
                        loop_op: &by_four,
                    },
                    LoopCondition {
                        op: CondOp::Lt,
                        against: CondValType::First,
                        loop_op: &dynamic,
                    },
                ],
                vec![LoopCondition {
                    op: CondOp::Ne,
                    against: CondValType::Int(2),
                    loop_op: &by_four,
                }],
            ],
            negated: true,
        };
        let ops = conditional_operation(&mut vals, "cond", &or_of_two, marker(), None)
            .expect("two constant-bound loops and three comparisons");
        assert_eq!(
            text(&ops),
            concat!(
                "%0 = arith.constant true\n",
                "%1 = arith.constant false\n",
                "%2 = arith.constant 3 : index\n",
                "%3 = arith.cmpi eq, %100, %2 : index\n",
                "%4 = scf.if %3 -> (i1) {\n",
                "  %5 = arith.cmpi slt, %200, %201 : index\n",
                "  %6 = scf.if %5 -> (i1) {\n",
                "    scf.yield %0 : i1\n",
                "  } else {\n",
                "    scf.yield %1 : i1\n",
                "  } {dbgName = \"cond\"}\n",
                "  scf.yield %6 : i1\n",
                "} else {\n",
                "  scf.yield %1 : i1\n",
                "} {dbgName = \"cond\"}\n",
                "%7 = arith.constant 2 : index\n",
                "%8 = arith.cmpi ne, %100, %7 : index\n",
                "%9 = scf.if %8 -> (i1) {\n",
                "  scf.yield %0 : i1\n",
                "} else {\n",
                "  scf.yield %4 : i1\n",
                "} {dbgName = \"cond\"}\n",
                "%10 = arith.cmpi eq, %9, %1 : i1\n",
                "scf.if %10 {\n",
                "  %300 = arith.constant 7 : index\n",
                "} {dbgName = \"cond\"}\n",
            ),
        );

        // ⭐ ONE AND SET, NOT NEGATED, NO ELSE — the other shape, and the `scf.for`'s LAST is a
        // `subi` off its upper bound where the `affine.for`'s folds to a literal.
        let mut vals = Values::default();
        let bare = CondComposite {
            or_of_ands: vec![vec![
                LoopCondition {
                    op: CondOp::Eq,
                    against: CondValType::Last,
                    loop_op: &by_four,
                },
                LoopCondition {
                    op: CondOp::Ge,
                    against: CondValType::Last,
                    loop_op: &dynamic,
                },
            ]],
            negated: false,
        };
        let ops = conditional_operation(&mut vals, "guard", &bare, marker(), None)
            .expect("both loops are comparable");
        assert_eq!(
            text(&ops),
            concat!(
                "%0 = arith.constant 3 : index\n",
                "%1 = arith.cmpi eq, %100, %0 : index\n",
                "scf.if %1 {\n",
                "  %2 = arith.constant 1 : index\n",
                "  %3 = arith.subi %202, %2 : index\n",
                "  %4 = arith.cmpi sge, %200, %3 : index\n",
                "  scf.if %4 {\n",
                "    %300 = arith.constant 7 : index\n",
                "  } {dbgName = \"guard\"}\n",
                "} {dbgName = \"guard\"}\n",
            ),
            "no `arith.constant true`/`false` at all, and the body innermost",
        );

        // ⛔ AN ELSE BRANCH ASKED FOR AND LEFT EMPTY IS A BLOCK: `Some(vec![])` prints `} else {` and
        // an empty pair of braces, which is a different op from no else region at all.
        let mut vals = Values::default();
        let ops = conditional_operation(&mut vals, "e", &bare, marker(), Some(Vec::new()))
            .expect("an else branch alone takes the chained shape");
        assert!(
            text(&ops).ends_with(concat!(
                "scf.if %4 {\n",
                "  %300 = arith.constant 7 : index\n",
                "} else {\n",
                "} {dbgName = \"e\"}\n",
            )),
            "{}",
            text(&ops),
        );

        // ⛔ THE THREE REFUSALS.
        let mut vals = Values::default();
        let open_ended = DfirOp::Affine(affine::Op::For {
            iv: Val(100),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Val(Val(400)),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });
        for refused in [
            CondComposite {
                or_of_ands: vec![vec![LoopCondition {
                    op: CondOp::Eq,
                    against: CondValType::Last,
                    loop_op: &open_ended,
                }]],
                negated: false,
            },
            CondComposite {
                or_of_ands: vec![vec![LoopCondition {
                    op: CondOp::Always,
                    against: CondValType::Last,
                    loop_op: &by_four,
                }]],
                negated: false,
            },
            CondComposite {
                or_of_ands: Vec::new(),
                negated: true,
            },
        ] {
            assert_eq!(
                conditional_operation(&mut vals, "no", &refused, marker(), None),
                None,
            );
        }
    }

    /// ⛔⛔ THE `OUT` DIM MINUS THE `{0, 1}` STAGE PAIR, AND THE VALUE HANDED BACK IS THE OUTERMOST.
    ///
    /// A SAMV stick writes its accumulated value out under this guard: answering with the INNERMOST
    /// if's result would gate the write on the innermost loop alone, and letting the `{0, 1}` loop in
    /// would gate it on a stage the reference excludes by hand. With no `OUT` loop of its own the
    /// reference leaves `if_op` null for a caller that dereferences it either way, which is [`None`]
    /// here — and nothing is minted on that path.
    #[test]
    fn every_out_loop_but_the_excluded_stage_and_the_outermost_result() {
        let by_four = DfirOp::Affine(affine::Op::For {
            iv: Val(100),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });
        let dynamic = DfirOp::Scf(scf::Op::For {
            iv: Val(200),
            lo: Val(201),
            hi: Val(202),
            step: Val(203),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        });
        let loops = [
            SamvLoop {
                dim: Some(PrimaryDim::In),
                stage: StagePair { num: 1, den: 2 },
                loop_op: &by_four,
            },
            SamvLoop {
                dim: Some(PrimaryDim::Out),
                stage: StagePair { num: 0, den: 1 },
                loop_op: &by_four,
            },
            SamvLoop {
                dim: Some(PrimaryDim::Out),
                stage: StagePair { num: 1, den: 2 },
                loop_op: &by_four,
            },
            SamvLoop {
                dim: None,
                stage: StagePair { num: 1, den: 2 },
                loop_op: &by_four,
            },
            SamvLoop {
                dim: Some(PrimaryDim::Out),
                stage: StagePair { num: 2, den: 3 },
                loop_op: &dynamic,
            },
        ];

        let mut vals = Values::default();
        let (ops, top) = conditionals_for_samv(&mut vals, &loops).expect("two OUT loops remain");
        let mut text = String::new();
        for op in &ops {
            emit(&mut text, op, 0);
        }
        assert_eq!(
            text,
            concat!(
                "%0 = arith.constant true\n",
                "%1 = arith.constant false\n",
                "%2 = arith.constant 3 : index\n",
                "%3 = arith.cmpi eq, %100, %2 : index\n",
                "%4 = scf.if %3 -> (i1) {\n",
                "  %5 = arith.constant 1 : index\n",
                "  %6 = arith.subi %202, %5 : index\n",
                "  %7 = arith.cmpi eq, %200, %6 : index\n",
                "  %8 = scf.if %7 -> (i1) {\n",
                "    scf.yield %0 : i1\n",
                "  } else {\n",
                "    scf.yield %1 : i1\n",
                "  }\n",
                "  scf.yield %8 : i1\n",
                "} else {\n",
                "  scf.yield %1 : i1\n",
                "}\n",
            ),
            "no dbgName, no guard `scf.if`, and every `else` yields false",
        );
        assert_eq!(top, Val(4), "the OUTERMOST if, not %8");

        // ⛔ AND THE REFUSAL MINTS NOTHING.
        let issued = vals.issued();
        assert_eq!(conditionals_for_samv(&mut vals, &loops[..2]), None);
        assert_eq!(conditionals_for_samv(&mut vals, &[]), None);
        assert_eq!(vals.issued(), issued);
    }

    /// ⛔⛔ THE SOURCE IS ASKED FIRST, AND A SOURCE ON THIS COMPONENT THAT DOES NOT SWITCH FALLS
    /// THROUGH.
    ///
    /// Stopping at the first end found on the component would answer "neither" for a transfer whose
    /// destination switches — no buffer switch emitted, and every iteration writing the same buffer.
    /// `numBuffers_ == -1` is streaming and not a count, and the factor comes from the SWITCHING
    /// end's own granularity row.
    #[test]
    fn the_source_is_asked_first_and_a_non_switching_one_falls_through() {
        let precision = DataType::Sen169Fp16;
        let src = SwitchSide {
            unit: DfirUnit::Lxlu,
            loc: DataLocation::LxluLx,
            switches: None,
            start_addresses: &"src",
        };
        let elsewhere = SwitchSide {
            unit: DfirUnit::L3lu,
            loc: DataLocation::L3luHbm,
            switches: Some(NumBuffers::Count(4)),
            start_addresses: &"another unit's",
        };
        let on_comp = SwitchSide {
            unit: DfirUnit::Lxlu,
            loc: DataLocation::LxluScaleReg,
            switches: Some(NumBuffers::Count(2)),
            start_addresses: &"dst",
        };

        let dsts = [elsewhere, on_comp];
        assert_eq!(
            buffering_or_streaming_mode(DfirUnit::Lxlu, precision, &src, &dsts),
            Some(BufferSwitch {
                mode: SwitchMode::Buffering,
                start_addresses: &"dst",
                factor: address_granularity_multiply_factor(DataLocation::LxluScaleReg, precision),
            }),
        );
        // ⭐ A SWITCHING SOURCE WINS OVER BOTH, AND `-1` IS STREAMING.
        let streaming = SwitchSide {
            switches: Some(NumBuffers::Streaming),
            ..src
        };
        assert_eq!(
            buffering_or_streaming_mode(DfirUnit::Lxlu, precision, &streaming, &dsts),
            Some(BufferSwitch {
                mode: SwitchMode::Streaming,
                start_addresses: &"src",
                factor: address_granularity_multiply_factor(DataLocation::LxluLx, precision),
            }),
        );
        // ⛔ AND NEITHER IS THE `mode = -1` THE REFERENCE RETURNS SUCCESS WITH.
        assert_eq!(
            buffering_or_streaming_mode(DfirUnit::L0lu, precision, &src, &dsts),
            None,
        );
    }

    /// ⛔⛔ A `CONDITION` OR `BLOCK` CHILD OF THE HEAD IS SKIPPED WITH EVERYTHING BELOW IT.
    ///
    /// The head's own cast admits loops only, where the recursion descends through all three kinds —
    /// so a switching transfer under a top-level condition never reaches any node's map. Reading the
    /// recursion's three arms as the driver's would write entries the reference does not have.
    #[test]
    fn only_a_loop_child_of_the_head_is_descended_into() {
        let switching_loop = |own: &'static str| SwitchNode {
            kind: NodeKind::Loop,
            children: Vec::new(),
            own: vec![own],
            all: Vec::new(),
        };
        let mut children = vec![
            SwitchNode {
                kind: NodeKind::Loop,
                children: vec![switching_loop("under a loop")],
                own: vec!["the loop's own"],
                all: Vec::new(),
            },
            SwitchNode {
                kind: NodeKind::Condition,
                children: vec![switching_loop("under a condition")],
                own: Vec::new(),
                all: Vec::new(),
            },
        ];

        propagate_buffer_switch_loops_to_root(&mut children);

        assert_eq!(children[0].all, vec!["under a loop", "the loop's own"]);
        assert_eq!(children[0].children[0].all, vec!["under a loop"]);
        assert_eq!(
            children[1].all,
            Vec::<&str>::new(),
            "the head admits loops only",
        );
        assert_eq!(children[1].children[0].all, Vec::<&str>::new());
    }
}
