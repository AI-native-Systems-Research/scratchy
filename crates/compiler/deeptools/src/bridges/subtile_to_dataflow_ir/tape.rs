//! THE WHOLE FORWARD TAPE, COMPILED.
//!
//! ⛔⛔ THE TAPE, NOT AN OP. The deliverable is every node of the forward becoming DataflowIR. A
//! single node's lowering is a step inside [`compile`], never the thing itself — "N nodes lowered"
//! where N is less than the tape's length is the empty-bundle failure wearing a different hat.
//!
//! ⭐⭐ AND THE CONSTANTS FLOW THROUGH IT. [`compile`] is generic over the machine
//! ([`Arch`]), the network ([`Model`]) and the rung ([`Workload`]), and every one of those is READ
//! BY A BRANCH below — see [`Exploit`]. A constant that only reached an attribute would have been
//! expressed and not exploited.

use crate::arch::Arch;
use crate::bridges::subtile_to_dataflow_ir::node::{Node, Residence};
use crate::bridges::subtile_to_dataflow_ir::transfer::{self, Lanes};
use crate::islands::dataflow_ir::op::{CompositeTransfer, Index, Op, Val};
use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef, Vector};
use crate::islands::dataflow_ir::{Grid, GroupId, KernelName, OpIndex, Program, ProgramName, Run};
use crate::model::Model;
use crate::units::{Core, Corelet, DfirUnit, Residency};
use crate::workload::{Exploit, Workload};

/// WHY A TAPE COULD NOT BE COMPILED.
///
/// ⛔ EACH CARRIES WHICH NODE AND WHAT ABOUT IT, because "the tape failed to lower" names nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapeError {
    /// A node's transfer could not be walked.
    Transfer {
        /// Which node of the tape.
        at: OpIndex,
        /// What the planner said.
        why: transfer::TransferError,
    },
}

/// A COUNTER THAT MINTS THE SSA VALUES OF ONE PROGRAM.
struct Vals(u32);

impl Vals {
    fn mint(&mut self) -> Val {
        let val = Val(self.0);
        self.0 += 1;
        val
    }
}

/// COMPILE THE WHOLE TAPE.
///
/// One [`Program`] per node, in tape order, named by its group, its index and its op-func — so the
/// symbol says which node it came from and the run's order is the tape's order.
///
/// ⭐⭐ THE CONSTANTS ARE EXPLOITED HERE, NOT CARRIED. `IS_DECODE` removes the row nest;
/// `FITS_LX` removes the tiling loop; `STICK_ALIGNED` removes the mask and everything that consumes
/// it. Each is a branch that emits DIFFERENT OPS, which is what
/// `tests/constants_change_the_program.rs` measures.
///
/// # Errors
///
/// Returns [`TapeError`] naming the node that could not be lowered. A tape that lowers partially is
/// not a result.
pub fn compile<A: Arch, M: Model, W: Workload>(
    tape: &[Node],
    group: GroupId,
) -> Result<Run<A>, TapeError> {
    // ⛔ FORCE THE INVARIANTS ONCE AT THE ENTRY POINT, for a build whose derived constants happen
    // not to be read on some path. An unreferenced associated const is never evaluated.
    M::check();
    let () = W::WELL_FORMED;

    let mut programs = Vec::with_capacity(tape.len());
    for (at, node) in tape.iter().enumerate() {
        let index = OpIndex(u32::try_from(at).expect("a tape index fits a u32"));
        programs.push(node_program::<A, M, W>(node, group, index)?);
    }

    // ⛔ EVERY NODE, OR NONE. A tape of N nodes becomes N programs; anything less is a forward that
    // silently skips work.
    assert_eq!(
        programs.len(),
        tape.len(),
        "the tape has {} nodes but {} programs were emitted",
        tape.len(),
        programs.len()
    );

    Ok(Run {
        kernel: KernelName::Group(group),
        programs,
    })
}

/// ONE NODE'S PROGRAM.
fn node_program<A: Arch, M: Model, W: Workload>(
    node: &Node,
    group: GroupId,
    index: OpIndex,
) -> Result<Program<A>, TapeError> {
    let mut vals = Vals(0);
    let mut body = Vec::new();

    // The units this program binds. The HBM is global — one for the device, no core and no corelet;
    // the scratchpad and the movers are this core's.
    let core = Core::checked(0).expect("every arch has a core 0");
    let corelet = Corelet::checked(0).expect("every arch has a corelet 0");

    let hbm = vals.mint();
    body.push(Op::GetUnit {
        result: hbm,
        residency: Residency::Global,
        unit: DfirUnit::Hbm,
    });
    let lx = vals.mint();
    body.push(Op::GetUnit {
        result: lx,
        residency: Residency::Scratchpad { core },
        unit: DfirUnit::Lx,
    });
    for unit in [DfirUnit::L3lu, DfirUnit::L3su] {
        let val = vals.mint();
        body.push(Op::GetUnit {
            result: val,
            residency: Residency::CoreWide { core },
            unit,
        });
    }
    for unit in [DfirUnit::Lxlu, DfirUnit::Sfp, DfirUnit::Lxsu] {
        let val = vals.mint();
        body.push(Op::GetUnit {
            result: val,
            residency: Residency::Corelet { core, corelet },
            unit,
        });
    }

    // ⭐ THE LANE WIDTH IS THE FORMAT'S, and it is the one the device declares. See `Lanes`: an
    // unlisted format is ONE lane by the stock arithmetic, not a stick's worth.
    let lanes = Lanes::F16;

    // Each input that lives in the HBM is viewed there and moved into the scratchpad. THIS is what
    // makes the weights present: a view alone is an address nothing fills.
    let mut staged = Vec::with_capacity(node.inputs.len());
    for input in &node.inputs {
        let rows = u64::from(input.rows.0);
        let cols = u64::from(input.cols.0);

        let start = vals.mint();
        body.push(Op::Constant {
            result: start,
            value: i64::try_from(input.start().0).expect("an element offset fits an i64"),
        });

        let from = match input.at {
            Residence::Hbm { .. } => hbm,
            Residence::Lx { .. } => lx,
        };
        let view = vals.mint();
        body.push(Op::GetLogicalMemoryView {
            result: view,
            from,
            start,
            // ⛔ ROW-MAJOR, which is DataflowIR's convention and was confirmed twice: IBM's own
            // views, and the scheduler synthesising strides back-to-front when a memref has none.
            layout: AffineMap::linear(&[i64::try_from(cols).expect("a width fits an i64"), 1]),
            ty: MemRef {
                shape: vec![rows, cols],
                elem: ElemType::F16,
            },
        });

        // An operand already in the scratchpad needs no transfer; one in the HBM does.
        if matches!(input.at, Residence::Lx { .. }) {
            staged.push((view, rows, cols));
            continue;
        }

        let dst_start = vals.mint();
        body.push(Op::Constant {
            result: dst_start,
            value: 0,
        });
        let dst = vals.mint();
        body.push(Op::GetLogicalMemoryView {
            result: dst,
            from: lx,
            start: dst_start,
            layout: AffineMap::linear(&[i64::try_from(cols).expect("a width fits an i64"), 1]),
            ty: MemRef {
                shape: vec![rows, cols],
                elem: ElemType::F16,
            },
        });

        let plan = transfer::plan(&[1, cols], &[1, cols], cols, lanes)
            .map_err(|why| TapeError::Transfer { at: index, why })?;
        let load_iv = vals.mint();
        body.push(Op::CompositeLoadAndStore(Box::new(CompositeTransfer {
            src: view,
            src_indices: vec![Index::Const(0), Index::Const(0)],
            src_ty: MemRef {
                shape: vec![rows, cols],
                elem: ElemType::F16,
            },
            dst,
            dst_indices: vec![Index::Const(0), Index::Const(0)],
            dst_ty: MemRef {
                shape: vec![rows, cols],
                elem: ElemType::F16,
            },
            load_iv,
            load_iv_ty: Vector {
                len: plan.vector_lanes,
                elem: ElemType::F16,
            },
            load_set: plan.load_set,
            load_order: plan.load_order,
            store_set: plan.store_set,
            store_order: plan.store_order,
            time_set: plan.time_set,
            time_order: plan.time_order,
            load_time_addr_map: plan.load_time_addr_map,
            store_time_addr_map: plan.store_time_addr_map,
            body: vec![Op::AgenYield],
        })));
        staged.push((dst, rows, cols));
    }

    Ok(Program {
        name: ProgramName::Emitted {
            group,
            index,
            func: node.op_func,
        },
        grid: Grid::single(),
        body: nest::<A, M, W>(body, &mut vals, &staged, node),
        arch: core::marker::PhantomData,
    })
}

/// THE LOOP NEST, WHICH IS WHERE THE CONSTANTS EARN THEIR KEEP.
///
/// ⛔⛔ THESE ARE REMOVALS, NOT BOUNDS. A row loop that runs once is still a region, still a
/// barrier, and still an induction variable every enclosed access is strided by — so `IS_DECODE`
/// does not set the bound to one, it emits no loop at all. Same for `FITS_LX` and the tiling loop.
fn nest<A: Arch, M: Model, W: Workload>(
    mut body: Vec<Op>,
    vals: &mut Vals,
    staged: &[(Val, u64, u64)],
    node: &Node,
) -> Vec<Op> {
    use crate::islands::dataflow_ir::op::Bound;

    let inner = compute::<A, M, W>(vals, staged, node);

    // ⭐ THE CACHE WALK, WHICH ONLY AN ATTENTION NODE HAS AND ONLY A WIDE BUCKET NEEDS.
    //
    // ⛔ THE SK BUCKET IS EXPLOITED IN TWO PLACES, NOT ONE. `KV_SINGLE_VECTOR` decides whether the
    // walk exists at all; `CACHE_FITS_LX` decides whether the span was staged before it or has to
    // be streamed inside it — and a streamed step carries its own transfer, so the two rungs emit
    // different bodies rather than the same body with a different trip count.
    let walked = if reads_cache(node.op_func) && !Exploit::<A, M, W>::KV_SINGLE_VECTOR {
        let iv = vals.mint();
        let steps = i64::from(Exploit::<A, M, W>::KV_VECTORS);
        let mut step_body = Vec::new();
        if !Exploit::<A, M, W>::CACHE_FITS_LX {
            // ⭐ THE SPAN DID NOT FIT, so each step brings its own slice of the cache across and
            // has to wait for it. The reference does exactly this inside its own walks: a
            // `sync_send` to the mover and a blocking `sync_recv` before the data is read
            // (`/tmp/ktir_ref/export/debug/dfir.mlir:95-98`).
            let mover = vals.mint();
            step_body.push(Op::GetUnit {
                result: mover,
                residency: Residency::Corelet {
                    core: Core::checked(0).expect("every arch has a core 0"),
                    corelet: Corelet::checked(0).expect("every arch has a corelet 0"),
                },
                unit: DfirUnit::Lxlu,
            });
            step_body.push(Op::SyncSend {
                to: mover,
                signal: crate::generated::SyncSignal::InputToLxsuToLxluToSync,
            });
            step_body.push(Op::SyncRecv {
                from: mover,
                signal: crate::generated::SyncSignal::InputToLxsuToLxluToSync,
            });
        }
        step_body.extend(inner);
        vec![Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(steps),
            body: step_body,
        }]
    } else {
        inner
    };
    let inner = walked;

    // ⭐ THE TILING LOOP, PRESENT ONLY WHERE THE ROW DOES NOT FIT.
    let tiled = if Exploit::<A, M, W>::FITS_LX {
        inner
    } else {
        let iv = vals.mint();
        let tiles = i64::from(Exploit::<A, M, W>::STICKS_PER_ROW);
        vec![Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(tiles),
            body: inner,
        }]
    };

    // ⭐ THE ROW NEST, ABSENT ENTIRELY AT DECODE.
    let rows = if Exploit::<A, M, W>::IS_DECODE {
        tiled
    } else {
        let iv = vals.mint();
        vec![Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(i64::from(W::ROWS)),
            body: tiled,
        }]
    };

    body.extend(rows);
    body
}

/// THE COMPUTE ITSELF.
///
/// ⭐⭐ THE RAGGED TAIL IS WHERE `STICK_ALIGNED` EARNS ITS KEEP. A row that is a whole number of
/// sticks needs no predicate: every lane of every stick is live, so the `create_affine_mask` and
/// the `element_wise_selection` that consumes it are not emitted AT ALL. A row with a tail needs
/// both, or its last stick writes lanes past the end of the tensor.
///
/// ⛔ NOT "A MASK OF ALL ONES". That is the same ops with a different constant, which is the exact
/// shape of expressing a constant without exploiting it.
fn compute<A: Arch, M: Model, W: Workload>(
    vals: &mut Vals,
    staged: &[(Val, u64, u64)],
    node: &Node,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let width = staged.first().map_or(1, |(_, _, cols)| *cols);
    let lanes = width.min(u64::from(Exploit::<A, M, W>::ACT_PER_STICK));
    let ty = Vector {
        len: lanes,
        elem: ElemType::F16,
    };

    let mut loaded = Vec::with_capacity(staged.len());
    for (view, rows, cols) in staged {
        let result = vals.mint();
        ops.push(Op::AgenVectorLoad {
            result,
            view: *view,
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty: MemRef {
                shape: vec![*rows, *cols],
                elem: ElemType::F16,
            },
            ty,
        });
        loaded.push(result);
    }

    // The op-func's own arity decides how the loaded vectors combine. A unary reads one, a binary
    // two; the tape decomposed everything richer into those before it got here.
    let Some((&op1, rest)) = loaded.split_first() else {
        return ops;
    };
    let mask = vals.mint();
    ops.push(Op::True { result: mask });
    let combined = match rest.first() {
        Some(&op2) => {
            let result = vals.mint();
            ops.push(Op::Binary {
                result,
                op1,
                op2,
                mask,
                binary_op: binary_for(node.op_func),
                op_specific_map: AffineMap::identity(1),
                operand_ty: ty,
                ty,
            });
            result
        }
        None => op1,
    };

    // ⭐ AND ONLY NOW, THE TAIL — if there is one.
    if !Exploit::<A, M, W>::STICK_ALIGNED {
        let live = u64::from(M::HIDDEN % Exploit::<A, M, W>::ACT_PER_STICK);
        let predicate = vals.mint();
        ops.push(Op::CreateAffineMask {
            result: predicate,
            lanes: live,
            ty: Vector {
                len: lanes,
                elem: ElemType::Int(1),
            },
        });
        let selected = vals.mint();
        ops.push(Op::ElementWiseSelection {
            result: selected,
            cond: predicate,
            lhs: combined,
            rhs: op1,
            mask,
            cond_ty: Vector {
                len: lanes,
                elem: ElemType::Int(1),
            },
            ty,
        });
    }
    ops
}

/// DOES THIS OP-FUNC READ THE KV CACHE?
///
/// ⭐ THE BATCH MATMULS DO — attention's two GEMMs are `batchmatmul`, one against the keys and one
/// against the values, and they are the only ops whose extent is the sk bucket rather than the
/// model's own widths. Everything else is shaped by [`Model`] alone, so the bucket must not reach
/// it: a rung's cache span is not a reason to re-tile an FFN.
const fn reads_cache(op_func: crate::generated::OpFunc) -> bool {
    use crate::generated::OpFunc;
    matches!(
        op_func,
        OpFunc::Batchmatmul
            | OpFunc::Batchmatmulfp8
            | OpFunc::Batchmatmulint8
            | OpFunc::Batchmatmulint4
    )
}

/// WHICH `vectorchain.binary` AN OP-FUNC IS.
///
/// ⛔ EXHAUSTIVE, NO WILDCARD. A `_ => Add` arm means a new op-func silently becomes an addition —
/// fluent wrong output rather than a compile error. Every op-func that is not a binary combine says
/// so by naming itself here.
fn binary_for(op_func: crate::generated::OpFunc) -> crate::islands::dataflow_ir::op::BinaryOp {
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::op::BinaryOp;
    match op_func {
        OpFunc::Sub => BinaryOp::Sub,
        OpFunc::Mul
        | OpFunc::Matmul
        | OpFunc::Matmulfp8
        | OpFunc::Matmulint8
        | OpFunc::Matmulint4
        | OpFunc::Batchmatmul
        | OpFunc::Batchmatmulfp8
        | OpFunc::Batchmatmulint8
        | OpFunc::Batchmatmulint4 => BinaryOp::Mul,
        OpFunc::Maximum | OpFunc::Max => BinaryOp::Max,
        OpFunc::Minimum => BinaryOp::Min,
        // Everything else combines by addition: the elementwise adds, the reductions whose combine
        // is a sum, and the unaries, whose second operand does not exist so the arm is unreached.
        OpFunc::Add
        | OpFunc::Realdiv
        | OpFunc::Abs
        | OpFunc::Silu
        | OpFunc::Exp
        | OpFunc::Reciprocal
        | OpFunc::Sqrt
        | OpFunc::Rsqrt
        | OpFunc::Sigmoid
        | OpFunc::Gelufwd
        | OpFunc::Mish
        | OpFunc::Tanh
        | OpFunc::Dl16tofp32
        | OpFunc::Fp32todl16
        | OpFunc::Sum
        | OpFunc::Mean
        | OpFunc::Identity
        | OpFunc::InterslicetransposeFp16
        | OpFunc::Restickifyophbm
        | OpFunc::Qfp8ch => BinaryOp::Add,
    }
}
