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
use crate::bridges::subtile_to_dataflow_ir::schedule;
use crate::bridges::subtile_to_dataflow_ir::transfer::{self, Lanes};
use crate::islands::dataflow_ir::op::{
    CompositeTransfer, Index, LaneMask, Op, Precision, Received, Val,
};
use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef, Vector};
use crate::islands::dataflow_ir::{
    Grid, GroupId, KernelName, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Run, Units,
};
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

    // ⭐⭐ THE FIVE STRUCTURAL DECISIONS, AND ONLY THOSE, BECOME THE EMITTER'S CONST GENERICS.
    //
    // ⛔ THIS IS THE EXPRESS/EXPLOIT LINE DRAWN EXACTLY. A flag decides which OPS EXIST, so it is a
    // const generic and the compiler folds the branch away. A count — a loop bound, a lane width —
    // only APPEARS IN an op that exists either way, so it travels as a value. Making the counts
    // const generics too would put the model and the rung in the emitter's signature and
    // monomorphise it once per (model, rows, cap): 63 x 27 x 6 is ten thousand copies of the same
    // code, which is what a three-and-a-half-minute single-threaded rustc looks like.
    //
    // ⭐ THE FLAGS ARE STILL DECIDED BY THE COMPILER. They are read off `Exploit<A, M, W>`, whose
    // consts are const-evaluated per arm; the dispatch below turns them into literal const-generic
    // arguments. `emit` is instantiated at most 2^5 times no matter how many models or rungs exist.
    let counts = Counts {
        rows: W::ROWS,
        kv_vectors: Exploit::<A, M, W>::KV_VECTORS,
        sticks_per_row: Exploit::<A, M, W>::STICKS_PER_ROW,
        act_per_stick: Exploit::<A, M, W>::ACT_PER_STICK,
        ragged_lanes: M::HIDDEN % Exploit::<A, M, W>::ACT_PER_STICK,
    };
    dispatch::<A, M, W>(tape, group, counts)
}

/// THE NUMBERS THAT APPEAR IN THE OUTPUT rather than deciding its shape.
///
/// ⛔ EVERY ONE OF THESE WAS COMPUTED FROM CONSTANTS. They are values here because a bound is a
/// value; the DECISIONS that turn ops on and off are the const generics on [`emit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// The rung's row count.
    pub rows: u32,
    /// How many hardware vectors this rung's cache span covers.
    pub kv_vectors: u32,
    /// How many sticks one row occupies, rounded up.
    pub sticks_per_row: u32,
    /// How many activation elements fill one stick.
    pub act_per_stick: u32,
    /// How many lanes of the final stick are live.
    pub ragged_lanes: u32,
}

/// TURN THE FIVE DECIDED FLAGS INTO CONST-GENERIC ARGUMENTS.
///
/// ⛔ FIVE NESTED TWO-WAY BRANCHES RATHER THAN A THIRTY-TWO-ARM MATCH, because each level is one
/// line and the reader can see that every flag reaches [`emit`] as a literal. `generic_const_exprs`
/// would let `emit::<{Exploit::<A,M,W>::IS_DECODE}, ..>` be written directly; it is unstable, so
/// the branch is written out.
fn dispatch<A: Arch, M: Model, W: Workload>(
    tape: &[Node],
    group: GroupId,
    counts: Counts,
) -> Result<Run<A>, TapeError> {
    macro_rules! cache {
        ($d:literal, $f:literal, $s:literal, $k:literal) => {
            if Exploit::<A, M, W>::CACHE_FITS_LX {
                emit::<A, $d, $f, $s, $k, true>(tape, group, counts)
            } else {
                emit::<A, $d, $f, $s, $k, false>(tape, group, counts)
            }
        };
    }
    macro_rules! kv {
        ($d:literal, $f:literal, $s:literal) => {
            if Exploit::<A, M, W>::NO_CACHE_WALK {
                cache!($d, $f, $s, true)
            } else {
                cache!($d, $f, $s, false)
            }
        };
    }
    macro_rules! stick {
        ($d:literal, $f:literal) => {
            if Exploit::<A, M, W>::STICK_ALIGNED {
                kv!($d, $f, true)
            } else {
                kv!($d, $f, false)
            }
        };
    }
    macro_rules! lx {
        ($d:literal) => {
            if Exploit::<A, M, W>::FITS_LX {
                stick!($d, true)
            } else {
                stick!($d, false)
            }
        };
    }
    if Exploit::<A, M, W>::IS_DECODE {
        lx!(true)
    } else {
        lx!(false)
    }
}

/// THE EMITTER, at one combination of the five decisions.
fn emit<
    A: Arch,
    const IS_DECODE: bool,
    const FITS_LX: bool,
    const STICK_ALIGNED: bool,
    const NO_CACHE_WALK: bool,
    const CACHE_FITS_LX: bool,
>(
    tape: &[Node],
    group: GroupId,
    counts: Counts,
) -> Result<Run<A>, TapeError> {
    let mut programs = Vec::with_capacity(tape.len());
    for (at, node) in tape.iter().enumerate() {
        let index = OpIndex(u32::try_from(at).expect("a tape index fits a u32"));
        programs.push(node_program::<
            A,
            IS_DECODE,
            FITS_LX,
            STICK_ALIGNED,
            NO_CACHE_WALK,
            CACHE_FITS_LX,
        >(node, group, index, counts)?);
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
        kernel: KernelName(group),
        programs,
    })
}

/// ONE NODE'S PROGRAM.
fn node_program<
    A: Arch,
    const IS_DECODE: bool,
    const FITS_LX: bool,
    const STICK_ALIGNED: bool,
    const NO_CACHE_WALK: bool,
    const CACHE_FITS_LX: bool,
>(
    node: &Node,
    group: GroupId,
    index: OpIndex,
    counts: Counts,
) -> Result<Program<A>, TapeError> {
    let mut vals = Vals(0);
    let mut body = Vec::new();

    // ⭐⭐ THE SCHEDULE IS READ HERE. Which units take part is the TEMPLATE's to say — it is the
    // same for every op of this op-func and knows no extents — so it comes from the vendored
    // `ddl.unit` statements rather than from a list written here. A hand-written unit set is a
    // schedule invented to look plausible.
    //
    // ⛔ AND THE FORMAT IS PART OF THE QUESTION. A template serves an op-func AT A PRECISION;
    // resolving without it hands an fp16 op the fp32 kernel.
    let schedule = node.op_func.program(A::GEN, node.format);

    let core = Core::checked(0).expect("every arch has a core 0");
    let corelet = Corelet::checked(0).expect("every arch has a corelet 0");

    // The two memories a view is taken over. Neither is named by a `ddl.unit` — the template says
    // `memory="lx"` on an allocation instead — so they are bound from the residences, not the walk.
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

    // Then every unit the schedule itself names, in the order it names them, each with the
    // residency the machine gives it.
    //
    // ⭐⭐ AND THE HANDLE IS KEPT WITH ITS KIND. The walk used to bind each unit and throw the `Val`
    // away, so nothing downstream could say WHICH unit a program unit runs on — and a
    // `dataflow.program_unit` is exactly "these units run this".
    //
    // ⛔⛔ KIND, NOT ROLE. This sorted into `movers` and `computers`, which put an `lxlu` and an
    // `lxsu` in ONE list — and the kind was known here and discarded one line later. See [`Units`]:
    // the backend reads the kind off `getUnits()[0]` alone.
    let mut bound: Vec<(DfirUnit, Val)> = Vec::new();
    let mut bind = |vals: &mut Vals, body: &mut Vec<Op>, unit: DfirUnit| {
        let val = vals.mint();
        bound.push((unit, val));
        body.push(Op::GetUnit {
            result: val,
            residency: crate::units::residency_of(unit, core, corelet),
            unit,
        });
        val
    };
    for unit in schedule::units_of(schedule) {
        bind(&mut vals, &mut body, unit);
    }

    // ⛔⛔ THE L3 HALF IS THE MACHINE'S, NOT THE TEMPLATE'S, AND ITS ABSENCE IS NOT A CHOICE.
    // `getDataTransferType(src_is_fifo, dst_is_fifo)` is total over three cases
    // (`DataTransferLowering.cpp:165-172`): memref→memref lowers to
    // `agen.composite_load_and_store` (`:255, :311`), memref→FIFO to `agen.vector_load` +
    // `dataflow.send` (`:274, :424`), FIFO→memref to `dataflow.receive` + `agen.vector_store`
    // (`:293, :495`). An HBM→LX move has a memref at both ends, so it IS a composite transfer, and
    // `Helper.cpp:2177-2179` then requires an L3 half — memory-to-memory needs a unit with two
    // memory ports.
    //
    // ⭐ AND NO `ddl.unit` NAMES ONE, in any of the 32 templates. That is consistent, not a
    // contradiction: a template describes what the OP does, which is scratchpad↔wire — memref↔FIFO,
    // on `lxlu`/`lxsu`. Staging a weight out of the HBM is not part of any op's schedule. So the
    // unit that performs it is bound from the ARCH's topology, exactly as `hbm` and `lx` above are.
    let l3lu = bind(&mut vals, &mut body, DfirUnit::L3lu);

    // ⛔ WHERE THE UNIT BINDINGS END. Everything pushed from here on is work, and work belongs
    // inside a `dataflow.program_unit`.
    let units_end = body.len();

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
            staged.push(Staged {
                start: i64::try_from(input.start().0).expect("an element offset fits an i64"),
                rows,
                cols,
            });
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
        // ⛔⛔ THE PLACEMENT, NOT THE VIEW HANDLE. `dst` is bound inside the L3 unit's region and is
        // gone at its `}` — see [`Received`]. The unit that reads this operand takes its own view of
        // the same LX address, which is exactly what IBM's `lxlu` does (`dfir.mlir:112-116`) rather
        // than reuse the `l3lu`'s handle from `:73-74`.
        staged.push(Staged {
            start: 0,
            rows,
            cols,
        });
    }

    // ⛔⛔ THE UNITS ARE BOUND OUTSIDE, THE WORK INSIDE. IBM's own emitted DataflowIR puts only
    // `dataflow.get_unit` and the constants at function scope, and every view, transfer and loop
    // inside a `dataflow.program_unit` (`/tmp/ktir_ref/export/debug/dfir.mlir:44-78`) — five of
    // them in one function, each naming the units it runs on.
    //
    // ⭐⭐ ONE WIRE WIDTH, READ ONCE. The type the mover sends at and the type the compute receives
    // at are the same value, so the two ends of the wire cannot disagree.
    let wire_ty = wire(&staged, counts);

    // ⭐⭐ THREE UNITS, ONE KIND EACH. The L3 half moves HBM to LX, the LX loader reads the
    // scratchpad and sends, the SFP receives and computes.
    //
    // ⛔ A KIND THE SCHEDULE DOES NOT NAME IS AN ABSENCE, NOT A REFUSAL. `Units::of` yields `None`
    // and the program has one fewer unit — the same shape as `IS_DECODE` removing the row nest.
    // What it must never be is an EMPTY unit list: `ProgramUnitsReduction.cpp:175` indexes
    // `getUnits()[0]` unguarded and aborts the whole compiler on an LLVM assertion with no
    // diagnostic at all.
    let transfers = Units::one(DfirUnit::L3lu, l3lu);
    let loaders = Units::of(DfirUnit::Lxlu, &bound);
    let computers = Units::of(DfirUnit::Sfp, &bound);

    // `to`/`from` are the FIRST HOP, so each names the unit at the other end of the wire.
    let to = computers.as_ref().map_or(lx, Units::first);
    let from = loaders.as_ref().map_or(lx, Units::first);

    // ⭐ THE LOADER'S BODY: its OWN view of each staged operand, then the nest that reads and sends.
    // The view is retaken rather than inherited — `transfers`' handles died at its region's `}`.
    let mut loads = Vec::new();
    let views: Vec<Val> = staged
        .iter()
        .map(|operand| {
            let start = vals.mint();
            loads.push(Op::Constant {
                result: start,
                value: operand.start,
            });
            let view = vals.mint();
            loads.push(Op::GetLogicalMemoryView {
                result: view,
                from: lx,
                start,
                layout: AffineMap::linear(&[
                    i64::try_from(operand.cols).expect("a width fits an i64"),
                    1,
                ]),
                ty: MemRef {
                    shape: vec![operand.rows, operand.cols],
                    elem: ElemType::F16,
                },
            });
            view
        })
        .collect();
    loads.extend(nest::<IS_DECODE, FITS_LX, NO_CACHE_WALK, CACHE_FITS_LX>(
        &mut vals,
        node,
        counts,
        |vals| load_and_send(vals, &staged, &views, to, wire_ty),
    ));

    // ⭐ THE COMPUTE'S NEST: the same nest, one `dataflow.receive` per operand, then the compute.
    let computes =
        nest::<IS_DECODE, FITS_LX, NO_CACHE_WALK, CACHE_FITS_LX>(&mut vals, node, counts, |vals| {
            let mut ops = Vec::new();
            let operands: Vec<Received> = (0..staged.len())
                .map(|_| {
                    let result = vals.mint();
                    Received::receive(&mut ops, result, from, wire_ty)
                })
                .collect();
            ops.extend(compute::<STICK_ALIGNED>(
                vals, &operands, node, counts, wire_ty,
            ));
            ops
        });

    // ⭐ THE SPLIT IS WHERE THE UNIT WALK ENDED. Everything up to `units_end` binds units — that is
    // function scope in IBM's own output, which puts only `dataflow.get_unit` and constants outside
    // a program unit (`/tmp/ktir_ref/export/debug/dfir.mlir:44-63`). Everything after it is the
    // views the HBM operands take and the transfers that fill them, which is the L3 HALF's work.
    let mut preamble = body;
    let moves = preamble.split_off(units_end);

    Ok(Program {
        name: ProgramName {
            group,
            index,
            func: node.op_func,
        },
        grid: Grid::single(),
        preamble,
        // ⭐ THREE UNITS, AND THE PRECISION IS ON THE ONE THAT COMPUTES. R182 `Unknown parent op for
        // precision calculation` is the refusal for putting it on a unit that does not.
        units: ProgramUnits::of(
            // ⛔⛔ THE TRANSFER RUNS ON THE L3 HALF. `Helper.cpp:2177-2179` returns a bare
            // `failure()` for any other kind — see [`Units::moves_memory`].
            //
            // ⭐ AND IT IS THE HEAD, so a program always has at least one unit whatever the
            // template named. `Units::one` is total because the L3 half was bound above from the
            // arch rather than looked for in the schedule.
            ProgramUnit {
                on: transfers,
                precision: None,
                body: moves,
                arch: core::marker::PhantomData,
            },
            [
                loaders.map(|on| ProgramUnit {
                    on,
                    precision: None,
                    body: loads,
                    arch: core::marker::PhantomData,
                }),
                computers.map(|on| ProgramUnit {
                    on,
                    // ⛔⛔ THE PRECISION IS THE COMPUTE'S OPCODE, NOT THE TENSOR'S DTYPE.
                    // `stringifyComputePrecision` takes a `ComputeOpType` and maps `FMA16 -> "fp16"`,
                    // `FMA8 -> "fp8"`, `IMA4 -> "int4"` (`DSC2ToDataflowIR.hpp:54-71`) — "used to
                    // identify the MAC op code used in the units". A `DataType -> Precision` map would
                    // be answering a different question.
                    //
                    // ⭐ AND THIS EMITTER'S COMPUTES ARE fp16, because the activation stream is fp16
                    // whatever the weights are quantised to — every `Vector` it builds is `F16`. When a
                    // compute runs at another width this becomes that compute's opcode, read from the
                    // op it emits rather than from the node.
                    precision: Some(Precision::Fp16),
                    body: computes,
                    arch: core::marker::PhantomData,
                }),
            ]
            .into_iter()
            .flatten()
            .collect(),
        ),
        arch: core::marker::PhantomData,
    })
}

/// WHERE ONE OPERAND SITS IN THE SCRATCHPAD, once the transfer has put it there.
///
/// ⛔⛔ A PLACEMENT, NOT A VIEW HANDLE. This was the `Val` the L3 unit's
/// `get_logical_memory_view` bound — a name that is gone at that region's `}`
/// (`Parser.cpp:2273-2276`). Every unit that reads this operand takes its own view of the address,
/// which is what IBM's `lxlu` does at `dfir.mlir:112-116` rather than reuse `:73-74`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Staged {
    /// Its LX start, in ELEMENTS.
    start: i64,
    /// Its rows.
    rows: u64,
    /// Its columns.
    cols: u64,
}

/// THE LOOP NEST, WHICH IS WHERE THE CONSTANTS EARN THEIR KEEP.
///
/// ⛔⛔ THESE ARE REMOVALS, NOT BOUNDS. A row loop that runs once is still a region, still a
/// barrier, and still an induction variable every enclosed access is strided by — so `IS_DECODE`
/// does not set the bound to one, it emits no loop at all. Same for `FITS_LX` and the tiling loop.
///
/// ⭐⭐ THE INNERMOST BODY IS THE CALLER'S, because the SAME nest is built for TWO units. The mover
/// loads and sends inside it; the compute receives and computes inside it. Both walk one iteration
/// space, so both must lose exactly the same loops — building the nest twice from one function is
/// what makes that a fact rather than a coincidence between two copies.
fn nest<
    const IS_DECODE: bool,
    const FITS_LX: bool,
    const NO_CACHE_WALK: bool,
    const CACHE_FITS_LX: bool,
>(
    vals: &mut Vals,
    node: &Node,
    counts: Counts,
    innermost: impl FnOnce(&mut Vals) -> Vec<Op>,
) -> Vec<Op> {
    use crate::islands::dataflow_ir::op::Bound;

    let inner = innermost(vals);

    // ⭐ THE CACHE WALK, WHICH ONLY AN ATTENTION NODE HAS AND ONLY A WIDE BUCKET NEEDS.
    //
    // ⛔ THE SK BUCKET IS EXPLOITED IN TWO PLACES, NOT ONE. `NO_CACHE_WALK` decides whether the
    // walk exists at all; `CACHE_FITS_LX` decides whether the span was staged before it or has to
    // be streamed inside it — and a streamed step carries its own transfer, so the two rungs emit
    // different bodies rather than the same body with a different trip count.
    let walked = if reads_cache(node.op_func) && !NO_CACHE_WALK {
        let iv = vals.mint();
        let steps = i64::from(counts.kv_vectors);
        let mut step_body = Vec::new();
        if !CACHE_FITS_LX {
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
    let tiled = if FITS_LX {
        inner
    } else {
        let iv = vals.mint();
        let tiles = i64::from(counts.sticks_per_row);
        vec![Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(tiles),
            body: inner,
        }]
    };

    // ⭐ THE ROW NEST, ABSENT ENTIRELY AT DECODE.
    let rows = if IS_DECODE {
        tiled
    } else {
        let iv = vals.mint();
        vec![Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(i64::from(counts.rows)),
            body: tiled,
        }]
    };

    // ⛔⛔ ONE UNIT'S NEST. This used to `body.extend(rows)` — folding the loader's views and
    // transfers together with the compute into one flat sequence, which is how a program with no
    // `dataflow.program_unit` at all came to be emitted. The two run on DIFFERENT units (a compute
    // has no read port to the scratchpad — `VectorOperands.cpp:187-206`), so the caller puts them
    // in different program units and calls this once for each.
    rows
}

/// THE MOVER'S INNERMOST BODY: read each staged view, and put it on the wire.
///
/// ⭐⭐ THE LOAD BELONGS HERE, NOT IN THE COMPUTE. IBM's `lxlu` unit is exactly this —
/// `agen.vector_load` then `dataflow.send` (`/tmp/ktir_ref/export/debug/dfir.mlir:122-129`) — and
/// their `sfp` unit holds nothing but receives, the compute, and a send on (`:144-151`).
fn load_and_send(
    vals: &mut Vals,
    staged: &[Staged],
    views: &[Val],
    to: Val,
    ty: Vector,
) -> Vec<Op> {
    let mut ops = Vec::with_capacity(staged.len() * 2);
    for (operand, view) in staged.iter().zip(views) {
        let loaded = vals.mint();
        ops.push(Op::AgenVectorLoad {
            result: loaded,
            view: *view,
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty: MemRef {
                shape: vec![operand.rows, operand.cols],
                elem: ElemType::F16,
            },
            ty,
        });
        ops.push(Op::Send {
            to,
            data: loaded,
            ty,
        });
    }
    ops
}

/// THE WIDTH ONE OPERAND ARRIVES AT — the wire's, which is a stick or the row if the row is
/// narrower.
fn wire(staged: &[Staged], counts: Counts) -> Vector {
    let width = staged.first().map_or(1, |operand| operand.cols);
    Vector {
        len: width.min(u64::from(counts.act_per_stick)),
        elem: ElemType::F16,
    }
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
fn compute<const STICK_ALIGNED: bool>(
    vals: &mut Vals,
    operands: &[Received],
    node: &Node,
    counts: Counts,
    ty: Vector,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let lanes = ty.len;

    // ⛔⛔ THE OPERANDS ARRIVED ON THE WIRE. They used to be `agen.vector_load`s of the views the
    // MOVER took, which is an SSA name that does not exist here: MLIR pushes a definitions scope
    // per region (`Parser.cpp:2273-2276`), so a value a sibling `dataflow.program_unit` bound is
    // gone at its `}`. See [`Received`] for the two faces that one defect wore.
    let loaded: Vec<Val> = operands.iter().map(|r| r.val()).collect();

    // The op-func's own arity decides how the received vectors combine. A unary reads one, a binary
    // two; the tape decomposed everything richer into those before it got here.
    let Some((&op1, rest)) = loaded.split_first() else {
        return ops;
    };
    // ⛔⛔ NO MASK ON AN UNMASKED BINARY. This minted an `arith.constant true` and handed it to
    // every binary as a stand-in for "no mask" — an `i1` where the use site printed
    // `vector<64xi1>`, which is exactly what dbo-opt refused on granite-2b's `group_7`:
    // "use of value '%42' expects different type than prior uses: 'vector<64xi1>' vs 'i1'".
    //
    // ⛔ AND AN ALL-ONES MASK WOULD NOT BE THE FIX. That is the same op with a wider constant — a
    // mask that masks nothing. `$mask` is `Optional` in the dialect, so an unmasked binary omits
    // the operand entirely.
    let combined = match rest.first() {
        Some(&op2) => {
            let result = vals.mint();
            ops.push(Op::Binary {
                result,
                op1,
                op2,
                mask: None,
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
    if !STICK_ALIGNED {
        // ⭐ ONE `LaneMask`, AND EVERY MENTION OF ITS TYPE COMES FROM IT. The definition below and
        // the condition's type at the use are the same `Vector` value, so they cannot drift.
        let prefix = LaneMask::prefix_of(
            u64::from(counts.ragged_lanes),
            Vector {
                len: lanes,
                elem: ElemType::Int(1),
            },
        );
        let bound = vals.mint();
        ops.push(Op::CreateAffineMask {
            result: bound,
            mask: prefix,
        });
        let selected = vals.mint();
        ops.push(Op::ElementWiseSelection {
            result: selected,
            // ⭐ THE PREDICATE IS THE CONDITION, carrying the type it was bound at.
            cond: prefix.binds(bound),
            lhs: combined,
            rhs: op1,
            // ⛔ AND NO MASK. The condition and the mask are SEPARATE operands
            // (`VectorChain.td:401-402`); reusing the predicate as both would be one value doing
            // two jobs, and the dialect makes the mask optional precisely so it can be absent.
            mask: None,
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
