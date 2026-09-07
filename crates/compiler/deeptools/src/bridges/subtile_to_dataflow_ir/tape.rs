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
use crate::islands::dataflow_ir::dialects::agen::CompositeTransfer;
use crate::islands::dataflow_ir::dialects::dataflow::{Precision, Received};
use crate::islands::dataflow_ir::dialects::vectorchain::{Computed, LaneMask};
use crate::islands::dataflow_ir::dialects::{
    Index, Op, Val, affine, agen, arith, dataflow, vectorchain,
};
use crate::islands::dataflow_ir::link::{self, Link, Placed, RecvEnd, SendEnd};
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
    body.push(Op::Dataflow(dataflow::Op::GetUnit {
        result: hbm,
        residency: Residency::Global,
        unit: DfirUnit::Hbm,
        num_folds: None,
    }));
    let lx = vals.mint();
    body.push(Op::Dataflow(dataflow::Op::GetUnit {
        result: lx,
        residency: Residency::Scratchpad { core },
        unit: DfirUnit::Lx,
        num_folds: None,
    }));

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
        body.push(Op::Dataflow(dataflow::Op::GetUnit {
            result: val,
            residency: crate::units::residency_of(unit, core, corelet),
            unit,
            num_folds: None,
        }));
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
    //
    // ⛔ AND EACH LANDS SOMEWHERE OF ITS OWN. The watermark is where the next staged operand goes,
    // in ELEMENTS; it was the literal `0` for all of them, so every staged tensor aliased the
    // first. See [`Placed`].
    let mut staged: Vec<Placed> = Vec::with_capacity(node.inputs.len());
    let mut lx_watermark: i64 = 0;
    for input in &node.inputs {
        let rows = u64::from(input.rows.0);
        let cols = u64::from(input.cols.0);

        let start = vals.mint();
        body.push(Op::Arith(arith::Op::Constant {
            result: start,
            value: i64::try_from(input.start().0).expect("an element offset fits an i64"),
        }));

        let from = match input.at {
            Residence::Hbm { .. } => hbm,
            Residence::Lx { .. } => lx,
        };
        let view = vals.mint();
        body.push(Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
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
        }));

        // An operand already in the scratchpad needs no transfer; one in the HBM does. Either way
        // the PLACEMENT is what the reader gets — see [`Placed`].
        if matches!(input.at, Residence::Lx { .. }) {
            staged.push(Placed::written_at(
                i64::try_from(input.start().0).expect("an element offset fits an i64"),
                rows,
                cols,
            ));
            continue;
        }

        // ⛔⛔ AND THE STAGED OPERANDS DO NOT ALL LAND AT ZERO. `dst_start` was the literal `0` for
        // every operand, so in `g6_1_matmul` the activation's `memref<1x2048xf16>` and the weight's
        // `memref<2048x2048xf16>` were BOTH views of LX element 0 — the weight written over the
        // activation. dbo-opt compiles one program unit at a time and has no cross-unit alias
        // analysis, so nothing refused it; it would have been wrong numbers on hardware.
        //
        // ⭐ EACH OPERAND FOLLOWS THE LAST, IN ELEMENTS, which is the unit `start` is stated in
        // (`Dataflow.td:250`).
        let dst_start = vals.mint();
        body.push(Op::Arith(arith::Op::Constant {
            result: dst_start,
            value: lx_watermark,
        }));
        let dst = vals.mint();
        body.push(Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: dst,
            from: lx,
            start: dst_start,
            layout: AffineMap::linear(&[i64::try_from(cols).expect("a width fits an i64"), 1]),
            ty: MemRef {
                shape: vec![rows, cols],
                elem: ElemType::F16,
            },
        }));

        let plan = transfer::plan(&[1, cols], &[1, cols], cols, lanes)
            .map_err(|why| TapeError::Transfer { at: index, why })?;
        let load_iv = vals.mint();
        body.push(Op::Agen(agen::Op::CompositeLoadAndStore(Box::new(
            CompositeTransfer {
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
                body: vec![Op::Agen(agen::Op::Yield)],
            },
        ))));
        // ⛔⛔ THE PLACEMENT, NOT THE VIEW HANDLE. `dst` is bound inside the L3 unit's region and is
        // gone at its `}` — see [`Received`]. The unit that reads this operand takes its own view of
        // the same LX address, which is exactly what IBM's `lxlu` does (`dfir.mlir:112-116`) rather
        // than reuse the `l3lu`'s handle from `:73-74`.
        //
        // ⭐⭐ AND THE WRITER SAYS WHERE. `Placed::written_at` is minted HERE, by the unit that puts
        // the bytes there, and every reader spends this value rather than recomputing an address of
        // its own. See [`Placed`].
        staged.push(Placed::written_at(lx_watermark, rows, cols));
        lx_watermark += i64::try_from(rows * cols).expect("an LX extent fits an i64");
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
    let storers = Units::of(DfirUnit::Lxsu, &bound);

    // ⭐⭐ ONE WIRE, TWO ENDS, SPENT ONCE EACH. The mover's `to` and the compute's `from` used to be
    // two independent `Units::first` lookups that nothing tied together — and if a schedule named
    // neither kind, both fell back to `lx` and the program described a wire from a memory to
    // itself. See [`Link`]: the ends come out of ONE value, so they name one wire by construction.
    let load_wire: Link<link::Lxlu, link::Sfp> = Link::between(
        loaders.as_ref().map_or(lx, Units::first),
        computers.as_ref().map_or(lx, Units::first),
    );
    let (to_compute, from_loader) = load_wire.ends();

    // ⭐ AND THE SECOND WIRE: the compute's result to the unit that stores it.
    let store_wire: Link<link::Sfp, link::Lxsu> = Link::between(
        computers.as_ref().map_or(lx, Units::first),
        storers.as_ref().map_or(lx, Units::first),
    );
    let (to_storer, from_compute) = store_wire.ends();

    // ⭐ THE LOADER'S BODY: its OWN view of each staged operand, then the nest that reads and sends.
    // The view is retaken rather than inherited — `transfers`' handles died at its region's `}`.
    let mut loads = Vec::new();
    let views: Vec<Val> = staged
        .iter()
        .map(|operand| {
            let start = vals.mint();
            loads.push(Op::Arith(arith::Op::Constant {
                result: start,
                value: operand.start(),
            }));
            let view = vals.mint();
            loads.push(Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: view,
                from: lx,
                start,
                layout: AffineMap::linear(&[
                    i64::try_from(operand.cols()).expect("a width fits an i64"),
                    1,
                ]),
                ty: MemRef {
                    shape: vec![operand.rows(), operand.cols()],
                    elem: ElemType::F16,
                },
            }));
            view
        })
        .collect();
    loads.extend(nest::<IS_DECODE, FITS_LX, NO_CACHE_WALK, CACHE_FITS_LX>(
        &mut vals,
        node,
        counts,
        |vals| load_and_send(vals, &staged, &views, to_compute, wire_ty),
    ));

    // ⭐ THE COMPUTE'S NEST: the same nest, one `dataflow.receive` per operand, then the compute.
    let computes =
        nest::<IS_DECODE, FITS_LX, NO_CACHE_WALK, CACHE_FITS_LX>(&mut vals, node, counts, |vals| {
            let mut ops = Vec::new();
            // ⛔⛔⛔ THE COUNT IS THE TEMPLATE'S, NOT `staged.len()`. This minted one edge per
            // STAGED OPERAND and handed them all to a binary compute, which spent two and dropped
            // the rest — dbo-opt: "Dangling non-compute op has no use | no OperandReuse entry:
            // never an operand of a lowered compute", naming a `dataflow.receive`
            // (`VectorChainToSentientPESFP.cpp:1343`). `Program::input_arity` reads the count off
            // the vendored `ddl.operation_bind` that this node's op-func and format resolved to,
            // and the array head it selects is the only way to mint them — so the number received
            // and the number consumed are one fact with one source.
            match schedule.input_arity() {
                1 => {
                    let operands = edges::<1>(vals, &mut ops, from_loader, wire_ty);
                    ops.extend(compute_unary::<STICK_ALIGNED>(
                        vals, operands, counts, wire_ty, to_storer,
                    ));
                }
                2 => {
                    let operands = edges::<2>(vals, &mut ops, from_loader, wire_ty);
                    ops.extend(compute::<STICK_ALIGNED>(
                        vals, operands, node, counts, wire_ty, to_storer,
                    ));
                }
                // ⭐ AND A WIDER BIND IS WORK, NOT A CASE TO ABSORB. `rope.ddl:26-27` binds
                // `rope64p1` and `rope64p2` with three inputs each, chained through `%iatensor` —
                // two `vectorchain` computes, not one. Emitting the first two operands and dropping
                // the third is precisely the defect above wearing a different arity.
                wider => todo!(
                    "{:?} at {:?} binds {wider} inputs; only the 1- and 2-operand heads are \
                     written, so this node has no decomposition yet",
                    node.op_func,
                    node.format
                ),
            }
            ops
        });

    // ⭐⭐ THE STORE UNIT SPENDS THE OTHER END. A compute whose result nothing consumes is never
    // lowered at all — `BinaryOpLowering` resolves the destination in `fillOpInfo` and fails before
    // `setReuseInformation` (`VectorChainToSentientPESFP.cpp:332-338`), and the failure resurfaces
    // as "Dangling non-compute op has no use" naming a RECEIVE. This is `kReceiveAndStore`:
    // FIFO to memref, which is `dataflow.receive` + `agen.vector_store` on the LX store unit
    // (`DataTransferLowering.cpp:293, :495`).
    //
    // ⛔ AND THE VIEW IS THIS UNIT'S OWN. `node.output`'s placement is retaken here rather than
    // inherited — a view bound in a sibling region is gone at its `}` (see [`Received`]).
    let out_rows = u64::from(node.output.rows.0);
    let out_cols = u64::from(node.output.cols.0);
    let mut stores = Vec::new();
    let out_start = vals.mint();
    stores.push(Op::Arith(arith::Op::Constant {
        result: out_start,
        value: i64::try_from(node.output.start().0).expect("an element offset fits an i64"),
    }));
    let out_view = vals.mint();
    stores.push(Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
        result: out_view,
        from: lx,
        start: out_start,
        layout: AffineMap::linear(&[i64::try_from(out_cols).expect("a width fits an i64"), 1]),
        ty: MemRef {
            shape: vec![out_rows, out_cols],
            elem: ElemType::F16,
        },
    }));
    stores.extend(nest::<IS_DECODE, FITS_LX, NO_CACHE_WALK, CACHE_FITS_LX>(
        &mut vals,
        node,
        counts,
        |vals| {
            let mut ops = Vec::new();
            let arrived = vals.mint();
            let received = Received::receive(&mut ops, arrived, from_compute, wire_ty);
            // ⭐ THE STORE IS A CONSUMER TOO, and it spends its one edge. Read the width BEFORE
            // spending, because `operand` takes `self`.
            let stored_ty = received.ty();
            ops.push(Op::Agen(agen::Op::VectorStore {
                value: received.operand(),
                view: out_view,
                indices: vec![Index::Const(0), Index::Const(0)],
                view_ty: MemRef {
                    shape: vec![out_rows, out_cols],
                    elem: ElemType::F16,
                },
                ty: stored_ty,
            }));
            ops
        },
    ));

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
                storers.map(|on| ProgramUnit {
                    on,
                    precision: None,
                    body: stores,
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
    use crate::islands::dataflow_ir::dialects::affine::Bound;

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
            step_body.push(Op::Dataflow(dataflow::Op::GetUnit {
                result: mover,
                residency: Residency::Corelet {
                    core: Core::checked(0).expect("every arch has a core 0"),
                    corelet: Corelet::checked(0).expect("every arch has a corelet 0"),
                },
                unit: DfirUnit::Lxlu,
                num_folds: None,
            }));
            step_body.push(Op::Dataflow(dataflow::Op::SyncSend {
                to: mover,
                signal: crate::generated::SyncSignal::InputToLxsuToLxluToSync,
            }));
            step_body.push(Op::Dataflow(dataflow::Op::SyncRecv {
                from: mover,
                signal: crate::generated::SyncSignal::InputToLxsuToLxluToSync,
            }));
        }
        step_body.extend(inner);
        vec![Op::Affine(affine::Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(steps),
            carried: Vec::new(),
            body: step_body,
            dbg_name: None,
        })]
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
        vec![Op::Affine(affine::Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(tiles),
            carried: Vec::new(),
            body: inner,
            dbg_name: None,
        })]
    };

    // ⭐ THE ROW NEST, ABSENT ENTIRELY AT DECODE.
    let rows = if IS_DECODE {
        tiled
    } else {
        let iv = vals.mint();
        vec![Op::Affine(affine::Op::For {
            iv,
            lo: Bound::Const(0),
            hi: Bound::Const(i64::from(counts.rows)),
            carried: Vec::new(),
            body: tiled,
            dbg_name: None,
        })]
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
    staged: &[Placed],
    views: &[Val],
    to: SendEnd,
    ty: Vector,
) -> Vec<Op> {
    let mut ops = Vec::with_capacity(staged.len() * 2);
    for (operand, view) in staged.iter().zip(views) {
        let loaded = vals.mint();
        ops.push(Op::Agen(agen::Op::VectorLoad {
            result: loaded,
            view: *view,
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty: MemRef {
                shape: vec![operand.rows(), operand.cols()],
                elem: ElemType::F16,
            },
            ty,
        }));
        ops.push(Op::Dataflow(dataflow::Op::Send {
            to,
            data: loaded,
            ty,
        }));
    }
    ops
}

/// THE WIDTH ONE OPERAND ARRIVES AT — the wire's, which is a stick or the row if the row is
/// narrower.
fn wire(staged: &[Placed], counts: Counts) -> Vector {
    let width = staged.first().map_or(1, |operand| operand.cols());
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
///
/// ⛔⛔ AND `to` IS NOT OPTIONAL. A compute whose result nothing consumes is never lowered at all:
/// `BinaryOpLowering::matchAndRewrite` resolves the destination in `fillOpInfo`
/// (`VectorChainToSentientPESFP.cpp:332-335`) and fails before `setReuseInformation` (`:338`), so
/// the operands are never registered and the failure reappears far away as "Dangling non-compute
/// op has no use" naming a RECEIVE. See [`Computed`]. Taking the destination as a parameter is what
/// makes "produce a value and place it nowhere" something no call site can express.
fn compute<const STICK_ALIGNED: bool>(
    vals: &mut Vals,
    // ⛔⛔⛔ AN ARRAY, BECAUSE THAT IS WHERE RUST'S LINEARITY IS. A `&[Received]` let this unit read
    // two of six; a `Vec<Received>` by value let it DROP four, and neither move-only nor
    // `#[must_use]` caught that — measured, the same dbo-opt refusal came back byte for byte. An
    // array destructured as `let [a, b] = operands` binds EVERY element or does not compile, so
    // "minted and not consumed" stops being a program that can be written. The LENGTH is the
    // `.ddl`'s: `Program::input_arity` picks which of these heads the emission site may call.
    operands: [Received; 2],
    node: &Node,
    counts: Counts,
    ty: Vector,
    to: SendEnd,
) -> Vec<Op> {
    let mut ops = Vec::new();

    // ⛔⛔ THE OPERANDS ARRIVED ON THE WIRE. They used to be `agen.vector_load`s of the views the
    // MOVER took, which is an SSA name that does not exist here: MLIR pushes a definitions scope
    // per region (`Parser.cpp:2273-2276`), so a value a sibling `dataflow.program_unit` bound is
    // gone at its `}`. See [`Received`] for the two faces that one defect wore.
    // ⛔⛔ SPENT, NOT BORROWED. This was `operands.iter().map(|r| r.val())` — it borrowed ALL N
    // received vectors and then took two, and the rest were dropped on the floor. On granite that
    // was six received and two consumed, and dbo-opt refused the four: "Dangling non-compute op has
    // no use | no OperandReuse entry: never an operand of a lowered compute". `Received::operand`
    // takes `self`, so the borrow is not expressible and every edge this unit receives must be
    // spent into the compute that consumes it.
    let [first, second] = operands;
    let op1 = first.operand();
    let op2 = second.operand();
    let result = vals.mint();
    // ⛔⛔ NO MASK ON AN UNMASKED BINARY. This minted an `arith.constant true` and handed it to
    // every binary as a stand-in for "no mask" — an `i1` where the use site printed
    // `vector<64xi1>`, which is exactly what dbo-opt refused on granite-2b's `group_7`:
    // "use of value '%42' expects different type than prior uses: 'vector<64xi1>' vs 'i1'".
    //
    // ⛔ AND AN ALL-ONES MASK WOULD NOT BE THE FIX. That is the same op with a wider constant — a
    // mask that masks nothing. `$mask` is `Optional` in the dialect, so an unmasked binary omits
    // the operand entirely.
    ops.push(Op::VectorChain(vectorchain::Op::Binary {
        result,
        op1,
        op2,
        mask: None,
        binary_op: binary_for(node.op_func),
        op_specific_map: AffineMap::identity(1),
        operand_ty: ty,
        ty,
    }));
    ops.extend(tail::<STICK_ALIGNED>(
        vals,
        Computed::of(result, ty),
        op1,
        counts,
        ty,
        to,
    ));
    ops
}

/// MINT EXACTLY `N` REUSE EDGES — one `dataflow.receive` each, at the wire's type.
///
/// ⛔⛔ `N` IS INFERRED FROM THE HEAD IT FEEDS, which is the whole point. The call site writes
/// `edges::<2>(..)` and passes the result to a head taking `[Received; 2]`; there is no length to
/// get wrong independently, because the two are the same `N`. A `Vec` here is what let the compute
/// unit receive six and consume two.
fn edges<const N: usize>(
    vals: &mut Vals,
    into: &mut Vec<Op>,
    from: RecvEnd,
    ty: Vector,
) -> [Received; N] {
    core::array::from_fn(|_| {
        let result = vals.mint();
        Received::receive(into, result, from, ty)
    })
}

/// THE ONE-OPERAND HEAD — the same unit, for an op-func whose `ddl.operation_bind` names one input.
///
/// ⭐ A SEPARATE FUNCTION BECAUSE THE ARITY IS A SEPARATE TYPE. `[Received; 1]` and `[Received; 2]`
/// are different types, so the emission site cannot hand a unary node's single edge to the binary
/// head or vice versa, and neither head can be reached with the wrong number of edges minted.
fn compute_unary<const STICK_ALIGNED: bool>(
    vals: &mut Vals,
    operands: [Received; 1],
    counts: Counts,
    ty: Vector,
    to: SendEnd,
) -> Vec<Op> {
    let [only] = operands;
    let op1 = only.operand();
    tail::<STICK_ALIGNED>(vals, Computed::of(op1, ty), op1, counts, ty, to)
}

/// THE RAGGED TAIL AND THE SEND — shared by every arity, because neither depends on the operand
/// count.
///
/// ⭐ `fallback` IS THE VALUE THE RAGGED LANES KEEP: the first operand, which is the one the mask
/// selects against for the lanes past `counts.ragged_lanes`.
fn tail<const STICK_ALIGNED: bool>(
    vals: &mut Vals,
    computed: Computed,
    fallback: Val,
    counts: Counts,
    ty: Vector,
    to: SendEnd,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let lanes = ty.len;
    let mut produced = computed;
    let op1 = fallback;

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
        ops.push(Op::VectorChain(vectorchain::Op::CreateAffineMask {
            result: bound,
            mask: prefix,
        }));
        let selected = vals.mint();
        ops.push(Op::VectorChain(vectorchain::Op::ElementWiseSelection {
            result: selected,
            // ⭐ THE PREDICATE IS THE CONDITION, carrying the type it was bound at.
            cond: prefix.binds(bound),
            lhs: produced.val(),
            rhs: op1,
            // ⛔ AND NO MASK. The condition and the mask are SEPARATE operands
            // (`VectorChain.td:401-402`); reusing the predicate as both would be one value doing
            // two jobs, and the dialect makes the mask optional precisely so it can be absent.
            mask: None,
            ty,
        }));
        produced = Computed::of(selected, ty);
    }

    // ⛔⛔ AND IT GOES SOMEWHERE. A `Computed` that is never sent is a compute the backend never
    // lowers — see [`Computed`]. `to` is the first hop, so it names the unit that stores it.
    ops.push(Op::Dataflow(dataflow::Op::Send {
        to,
        data: produced.val(),
        ty: produced.ty(),
    }));
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
fn binary_for(
    op_func: crate::generated::OpFunc,
) -> crate::islands::dataflow_ir::dialects::vectorchain::BinaryOp {
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::vectorchain::BinaryOp;
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
