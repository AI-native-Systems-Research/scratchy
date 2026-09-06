// SPDX-License-Identifier: Apache-2.0
//! `DataflowIR -> SentientIR` — the D1-D28 span, as one lowering over the TYPED program.
//!
//! # 🛑 THERE IS NO MLIR TEXT ON THE INPUT SIDE
//!
//! ⛔⛔ THIS BRIDGE IS CALLED WHERE THE TYPED VALUE LIVES, and that is the only place it can be
//! called: `lower_subtile_tape_to_dataflow_ir.rs`'s `emit_run` holds the
//! [`dataflow_ir::Run`] that `tape::compile` just produced and prints it ONLY because `dbo-opt`
//! consumes characters. Nothing in this crate parses MLIR and nothing should — so a bridge that
//! took text would need a whole second front end, and a bridge validated against text fixtures
//! would need the input transcribed by hand.
//!
//! ⭐ THE VALIDATOR IS THE ACCEPTANCE BUILD.
//! `cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel` runs this
//! over every program the bake stages — 417 of them, six op-funcs — and a [`todo!`] here stops it
//! and names the op. There is no unit test that can say anything this cannot.
//!
//! # 🛑 WHAT THE REFERENCE'S OWN OUTPUT DEMANDS
//!
//! ⭐ MEASURED ACROSS ALL 417 PROGRAMS, the reference's SentientIR uses **seven** of the dialect's
//! twenty-nine ops: `scalar_constant` (3,094), `load_and_store` (943), `load_and_send` (943),
//! `vector_binary` (417), `receive_and_store` (417), `vector_mac` (386), `for` (45). So the output
//! surface is seven ops, and `dataflow.get_unit` (2,502) and `dataflow.program_unit` (1,668)
//! SURVIVE — the rung is mixed, not a dialect swap.
//!
//! ⛔ AND FOUR STRUCTURAL FACTS THE SMALLEST GOLDEN STATES OUTRIGHT
//! (`tests/sentient_corpus/group_0__g0_0_mul.sentient.mlir`, 30 lines):
//!
//! 1. The constants are HOISTED ABOVE THE `get_unit`s, and there are **two separate zeros**.
//! 2. The `constant` unit's `get_unit` is **DROPPED** — seven in, six out.
//! 3. Every `get_logical_memory_view` **VANISHES**; a transfer names the UNIT, not the view.
//! 4. Ops FUSE ACROSS the DataflowIR statement boundary: a load and its send become one
//!    `load_and_send`, a receive and its store one `receive_and_store`, and one
//!    `vectorchain.binary` becomes **TWO** sentient computes.

/// `AgenToSentientLoweringPass`, function by function.
pub mod agen_to_sentient;

use crate::arch::{Arch, Bytes, Elements};
use crate::islands::dataflow_ir::dialects::{self as dfir_op, Op as DfirOp, Val};
use crate::islands::dataflow_ir::{self as dfir, Values};
use crate::islands::sentient::dialects::{Op as SenOp, sentient as sen};
use crate::islands::sentient::{self, ProgramUnit, ProgramUnits};
use crate::model::Model;
use crate::units::DfirUnit;
use crate::workload::Workload;
use std::collections::HashMap;

/// LOWER A WHOLE RUN.
///
/// ⭐ THE KERNEL NAME AND EVERY PROGRAM NAME SURVIVE, because a program keeps its symbol as it is
/// lowered — see [`sentient::Program::name`].
#[must_use]
pub fn lower<A: Arch, M: Model, W: Workload>(
    run: &dfir::Run<A>,
) -> sentient::Run<A, M, W> {
    sentient::Run {
        kernel: run.kernel,
        programs: run.programs.iter().map(program).collect(),
    }
}

/// WHAT THE PREAMBLE BOUND, so the units' bodies can name it after renumbering.
///
/// ⛔⛔ THE VIEWS ARE RESOLVED TO THEIR UNITS RATHER THAN CARRIED. A `sentient.load_and_store` takes
/// `src(%unit), dst(%unit)` where the DataflowIR took two `get_logical_memory_view` results, so the
/// view's own SSA value has no counterpart at this rung — what survives is WHICH MEMORY it viewed
/// and WHERE it started. Keeping a `view -> view` map would leave the emitter reaching for a value
/// that is never bound.
#[derive(Debug, Default)]
struct Bound {
    /// Old `get_unit` value -> the renumbered one.
    units: HashMap<Val, Val>,
    /// Old view value -> the OLD unit value it viewed. Resolved through `units` at use.
    view_of: HashMap<Val, Val>,
    /// Old view value -> its start address, as the old `arith.constant` value.
    view_start: HashMap<Val, Val>,
    /// Old `arith.constant` RESULT -> the value it carried, so a use can be resolved by value.
    ///
    /// ⛔ TWO HOPS ARE NEEDED AND ONE WOULD NOT DO. A view's start address is an SSA reference to an
    /// `arith.constant`, and that constant's own value is gone from this rung — several of them fold
    /// into one `scalar_constant`. So a use resolves OLD RESULT -> VALUE -> NEW RESULT, and a map
    /// straight from old result to new result would need one entry per folded copy, which is exactly
    /// the bookkeeping the fold removes.
    carried: HashMap<Val, i64>,
    /// `arith.constant` value -> the renumbered `sentient.scalar_constant` that replaced it.
    ///
    /// ⭐⭐ KEYED BY THE **VALUE**, WHICH IS WHY THE GOLDEN HAS ONE ZERO HERE AND NOT FOUR. The input
    /// binds `arith.constant 0` once per memory view — four times in the smallest program — and the
    /// reference's pipeline runs `SCCP` and canonicalisation over them, folding identical index
    /// constants into one. The SECOND zero in the golden is not this one; see [`Consts`].
    folded: HashMap<i64, Val>,
}

/// THE CONSTANTS THIS RUNG MINTS FOR ITSELF, as distinct from the ones it inherited.
///
/// # 🛑 THE GOLDEN HAS TWO ZEROS AND THEY ARE NOT A DUPLICATE
///
/// ⛔⛔ `%0 = 0` AND `%1 = 0` BOTH APPEAR, and reading that as sloppiness loses the mechanism. The
/// first is the INHERITED one — the `arith.constant 0` each memory view carried as its start
/// address, folded across four views ([`Bound::folded`]). The second is MINTED HERE: the
/// non-L3 `immutable_addr`/`increment` pair and the compute's mask are `sentient::ConstantOp`s the
/// lowering creates (`Splat.cpp:76-77` calls its own the *"default mask value"*), and a
/// `sentient.scalar_constant` does not CSE with an `arith.constant` because they are different ops.
///
/// ⛔ SO A SINGLE "constant pool" KEYED BY VALUE WOULD EMIT ONE ZERO AND BE WRONG, in a way that
/// shifts every SSA number after it and makes the whole program uncomparable.
#[derive(Debug, Default)]
struct Consts {
    /// Value -> the renumbered `scalar_constant` this rung minted for it.
    minted: HashMap<i64, Val>,
}

/// LOWER ONE PROGRAM.
fn program<A: Arch, M: Model, W: Workload>(
    input: &dfir::Program<A>,
) -> sentient::Program<A, M, W> {
    // ⛔ A FRESH NUMBERING. The two rungs do not share SSA numbers: the views go away and three
    // constants arrive ahead of the units, so every value after the first shifts. Carrying the old
    // numbers would print a module whose values do not exist.
    let mut vals = Values::default();
    let mut bound = Bound::default();
    let mut consts = Consts::default();
    let mut preamble: Vec<SenOp> = Vec::new();

    // ── the constants come first, above the units ────────────────────────────────────────────────
    // ⭐ TWO SOURCES, IN THIS ORDER: what the input already bound, then what this rung mints.
    inherit_constants(input, &mut vals, &mut bound, &mut preamble);
    mint_constants(input, &mut vals, &mut consts, &mut preamble);

    // ── then the units, minus the ones this rung does not bind ───────────────────────────────────
    for op in &input.preamble {
        match op {
            DfirOp::Dataflow(dfir_op::dataflow::Op::GetUnit {
                result,
                residency,
                unit,
            }) => {
                // ⛔⛔ THE `constant` UNIT IS DROPPED. The input binds seven units and the golden
                // holds six: `C0-constant-CL0` is gone. It is a pseudo-unit naming where immediates
                // live, and at this rung an immediate IS a `sentient.scalar_constant` in the
                // preamble — so the handle has nothing left to denote.
                if matches!(unit, DfirUnit::Constant) {
                    continue;
                }
                let fresh = vals.mint();
                bound.units.insert(*result, fresh);
                preamble.push(SenOp::Dataflow(dfir_op::dataflow::Op::GetUnit {
                    result: fresh,
                    residency: *residency,
                    unit: *unit,
                }));
            }
            // ⛔ VIEWS ARE RECORDED, NOT EMITTED — see [`Bound`].
            DfirOp::Dataflow(dfir_op::dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                ..
            }) => {
                bound.view_of.insert(*result, *from);
                bound.view_start.insert(*result, *start);
            }
            // ⛔ ALREADY ACCOUNTED FOR by `inherit_constants`.
            DfirOp::Arith(dfir_op::arith::Op::Constant { .. }) => {}
            other => todo!("lower a preamble op this bridge has not met: {other:?}"),
        }
    }

    let mut units = input.units.iter().map(|unit| {
        // ⛔ VIEWS AND CONSTANTS BOUND INSIDE A UNIT'S BODY TOO. The smallest program puts an
        // `arith.constant` and a `get_logical_memory_view` inside three of its four units, not in
        // the program preamble, so the walk has to record them per unit as well.
        let mut local = Bound {
            units: bound.units.clone(),
            view_of: bound.view_of.clone(),
            view_start: bound.view_start.clone(),
            carried: bound.carried.clone(),
            folded: bound.folded.clone(),
        };
        body(unit, &mut local, &consts)
    });
    // ⛔ NON-EMPTY BY THE TYPE, and the input's is too — `ProgramUnits` on both rungs makes "a
    // program with no units" unconstructible, which is what `dbo-adapt-scheduler-dfir found no
    // program to compile` was.
    let head = units.next().expect("the rung below's units are non-empty");
    let rest: Vec<_> = units.collect();

    sentient::Program {
        name: input.name,
        preamble,
        units: ProgramUnits::of(head, rest),
        bound: core::marker::PhantomData,
    }
}

/// THE CONSTANTS THE INPUT ALREADY BOUND, folded by value.
fn inherit_constants<A: Arch>(
    input: &dfir::Program<A>,
    vals: &mut Values,
    bound: &mut Bound,
    preamble: &mut Vec<SenOp>,
) {
    let bodies = input.units.iter().flat_map(|unit| unit.body.iter());
    for op in input.preamble.iter().chain(bodies) {
        if let DfirOp::Arith(dfir_op::arith::Op::Constant { result, value, .. }) = op {
            // ⭐ FOLDED BY VALUE — one `scalar_constant` however many the input bound.
            bound.carried.insert(*result, *value);
            if !bound.folded.contains_key(value) {
                let fresh = vals.mint();
                bound.folded.insert(*value, fresh);
                preamble.push(SenOp::Sentient(sen::Op::ScalarConstant {
                    value: *value,
                    result: fresh,
                    reg_locale: sen::RegType::Unknown,
                }));
            }
        }
    }
}

/// THE CONSTANTS THIS RUNG MINTS — see [`Consts`] for why they are separate from the inherited ones.
fn mint_constants<A: Arch>(
    input: &dfir::Program<A>,
    vals: &mut Values,
    consts: &mut Consts,
    preamble: &mut Vec<SenOp>,
) {
    let mut want: Vec<i64> = Vec::new();
    for unit in input.units.iter() {
        for op in &unit.body {
            match op {
                // ⛔ A NON-L3 TRANSFER'S `immutable_addr` AND `increment` ARE ZERO, and the compute's
                // mask is zero. One zero serves all of them.
                DfirOp::Agen(dfir_op::agen::Op::VectorLoad { .. } | dfir_op::agen::Op::VectorStore { .. })
                | DfirOp::VectorChain(_) => want.push(0),
                // ⛔⛔ AND AN L3 TRANSFER'S INCREMENT IS `total_elements * burst`, WHICH IS NOT ZERO.
                // The golden's third constant is 2048 = 64 * 32 — the value level 2 read out of
                // `setImmutableAddrAndIncrements`' L3 arm, confirmed here on real output.
                DfirOp::Agen(dfir_op::agen::Op::CompositeLoadAndStore(transfer)) => {
                    let extent = extent_of(transfer);
                    want.push(
                        i64::try_from(extent.total_elements.0 * extent.burst_size.0.max(1))
                            .expect("an increment fits an i64"),
                    );
                }
                _ => {}
            }
        }
    }
    for value in want {
        if !consts.minted.contains_key(&value) {
            let fresh = vals.mint();
            consts.minted.insert(value, fresh);
            preamble.push(SenOp::Sentient(sen::Op::ScalarConstant {
                value,
                result: fresh,
                reg_locale: sen::RegType::Unknown,
            }));
        }
    }
}

/// ONE UNIT'S BODY, FUSED.
///
/// # 🛑 THE FUSIONS ARE WHY THIS CANNOT BE AN OP-BY-OP MAP
///
/// ⛔⛔ THE SENTIENT RUNG HAS FEWER, WIDER STATEMENTS. A DataflowIR `agen.vector_load` followed by a
/// `dataflow.send` is ONE `sentient.load_and_send`; a `dataflow.receive` followed by an
/// `agen.vector_store` is ONE `sentient.receive_and_store`. So the walk consumes a window, not an op,
/// and an op-by-op `match` would emit a load with no destination and a send with no source.
fn body<A: Arch>(
    unit: &dfir::ProgramUnit<A>,
    bound: &mut Bound,
    consts: &Consts,
) -> ProgramUnit<A> {
    let mut out: Vec<SenOp> = Vec::new();
    let mut i = 0;
    while i < unit.body.len() {
        i += statement(&unit.body[i..], unit, bound, consts, &mut out);
    }
    ProgramUnit {
        on: unit.on.clone(),
        precision: unit.precision,
        body: out,
        arch: core::marker::PhantomData,
    }
}

/// LOWER ONE SENTIENT STATEMENT FROM THE HEAD OF `rest`, RETURNING HOW MANY DATAFLOWIR OPS IT ATE.
///
/// ⛔ THE COUNT IS THE POINT. A fusion that reported one would lower its own operands again.
fn statement<A: Arch>(
    rest: &[DfirOp],
    unit: &dfir::ProgramUnit<A>,
    bound: &mut Bound,
    consts: &Consts,
    out: &mut Vec<SenOp>,
) -> usize {
    match rest {
        // ── bookkeeping the body carries but the rung does not emit ──────────────────────────────
        [DfirOp::Arith(dfir_op::arith::Op::Constant { .. }), ..] => 1,
        [DfirOp::Dataflow(dfir_op::dataflow::Op::GetLogicalMemoryView { result, from, start, .. }), ..] => {
            bound.view_of.insert(*result, *from);
            bound.view_start.insert(*result, *start);
            1
        }

        // ── a memory-to-memory transfer: ONE op in, ONE op out ──────────────────────────────────
        [DfirOp::Agen(dfir_op::agen::Op::CompositeLoadAndStore(transfer)), ..] => {
            out.push(load_and_store(transfer, bound, consts));
            1
        }

        // ── the `affine` ops ─────────────────────────────────────────────────────────────────────
        // ⛔ NO WILDCARD: a fifth `affine` op must be a build error, not an inherited default.
        [DfirOp::Affine(op), ..] => {
            match op {
                dfir_op::affine::Op::Yield { operands } => {
                    // ⭐ 001/490. The parent is the op whose body we are walking; `body` is only ever
                    // called on a `dataflow.program_unit`'s statement list, so at this call site the
                    // parent is never an `scf.parallel` — `lower_affine_yield` is still the one place
                    // the rule lives, and the loop lowering passes its own parent when it walks a body.
                    match lower_affine_yield(Parent::Other, operands) {
                        YieldRewrite::Yielded(op) => out.push(SenOp::Scf(op)),
                        YieldRewrite::Declined => {}
                    }
                }
                dfir_op::affine::Op::For { .. } => todo!("affine.for -> the loop lowering"),
                dfir_op::affine::Op::Apply { .. } => todo!("affine.apply"),
                dfir_op::affine::Op::VectorLoad { .. } => todo!("affine.vector_load"),
                dfir_op::affine::Op::VectorStore { .. } => todo!("affine.vector_store"),
            }
            1
        }

        _ => todo!(
            "lower a statement this bridge has not met, on {:?}: {:?}",
            unit.on.kind(),
            rest.first()
        ),
    }
}

/// `agen.composite_load_and_store` -> `sentient.load_and_store`.
///
/// ⛔⛔ THE VIEWS BECOME UNITS. `src` and `dst` are the MEMORIES, resolved through [`Bound::view_of`];
/// the view's own value is not bound at this rung at all.
fn load_and_store(
    transfer: &dfir_op::agen::CompositeTransfer,
    bound: &Bound,
    consts: &Consts,
) -> SenOp {
    let extent = extent_of(transfer);
    let increment = i64::try_from(extent.total_elements.0 * extent.burst_size.0.max(1))
        .expect("an increment fits an i64");
    let inc = *consts
        .minted
        .get(&increment)
        .expect("mint_constants walked this same transfer");
    // ⛔ THE VIEW'S START ADDRESS IS THE ADDRESS, and on the L3 it is the view's OWN start rather
    // than a shifted one — `constructImmutableAddress` (`Helper.cpp:1226-1232`) takes the mem-view
    // start for `L3LU`/`L3SU` and the UPDATED start everywhere else, and `load_and_store` is L3-only
    // (`Helper.cpp:2177-2179`), so this arm is always the L3's.
    let src_addr = address(transfer.src, bound);
    let dst_addr = address(transfer.dst, bound);
    SenOp::Sentient(sen::Op::LoadAndStore {
        src: unit_of(transfer.src, bound),
        dst: unit_of(transfer.dst, bound),
        src_mutable_addr: src_addr,
        src_immutable_addr: src_addr,
        src_inc: inc,
        dst_mutable_addr: dst_addr,
        dst_immutable_addr: dst_addr,
        dst_inc: inc,
        multicast_info: None,
        results: (Val(0), Val(0)),
        extent,
        stride: 0,
        rotate_val: None,
        shuffle_mode: sen::ShuffleMode::NoShuffle,
        src_reg: sen::Reg { locale: sen::RegType::Unknown, index: None },
        dst_reg: sen::Reg { locale: sen::RegType::Unknown, index: None },
        dir: None,
        dbg_name: None,
    })
}

/// WHICH UNIT A VIEW VIEWED, renumbered.
fn unit_of(view: Val, bound: &Bound) -> Val {
    let old = bound.view_of.get(&view).expect("every view names a memory");
    *bound.units.get(old).expect("every memory is a bound unit")
}

/// A VIEW'S START ADDRESS, renumbered through the folded constants.
///
/// ⛔⛔ TWO HOPS, AND THE FALLBACK THAT WAS HERE WAS A DEFECT. This read the folded pool's only
/// entry when it held exactly one and otherwise returned the OLD value unchanged — an SSA number from
/// the rung below, which at this rung either names a different value or names nothing. It printed
/// fine on the one program that has a single distinct constant and would have silently mis-addressed
/// every program that has two.
fn address(view: Val, bound: &Bound) -> Val {
    let start = bound.view_start.get(&view).expect("every view has a start");
    let value = bound
        .carried
        .get(start)
        .expect("a view's start address is an arith.constant this walk recorded");
    *bound
        .folded
        .get(value)
        .expect("every recorded constant was minted into the preamble")
}

/// THE EXTENT OF A TRANSFER, from its own affine sets and maps.
///
/// ⚠️⚠️ **PARTIAL, AND NAMED AS PARTIAL.** The reference derives these in `AccessDetailsBase`
/// (`AccessDetails.cpp:32-205`) and it is not a read of the sets: `constructExtentAndTotalElements`
/// composes the transfer ORDER into the set's constraints, projects out the composed dimensions,
/// and reads a per-dimension width; `constructChunkAndShuffleInfo` then walks the memory view's
/// LAYOUT COEFFICIENTS against those widths to find the contiguous run (`chunk_size`) and the first
/// non-unit extent outside it (`chunk_stride`).
///
/// ⛔ SO THE `chunk_stride` HERE IS THE `.td` DEFAULT AND NOT A DERIVATION, and the golden for the
/// smallest program says `chunk_stride = 0` where [`sen::Extent::of`] defaults to one. That is a
/// known difference, left visible rather than papered over: it is the next thing the build's diff
/// will name.
fn extent_of(transfer: &dfir_op::agen::CompositeTransfer) -> sen::Extent {
    let elements: u64 = rectangle(&transfer.load_set).iter().product();
    let trips: u64 = rectangle(&transfer.time_set).iter().product();
    sen::Extent {
        total_elements: Elements(elements),
        // ⚠️ THE WIDTH IS IN **BITS** in every `element_size` attribute (`SentientOps.td:443-456`
        // walks *"64 16 bit elements"* at `element_size = 16`), and this island types the field as
        // `Bytes`. The number emitted is right and the TYPE is wrong; retyping it is a separate
        // change from wiring this bridge.
        element_size: Bytes(u64::from(transfer.load_iv_ty.elem.bits())),
        chunk_size: Elements(elements),
        chunk_stride: Elements(0),
        // ⛔ THE BURST IS THE INNERMOST TIME BOUND — `computeBurstAndGroup` scans the time dims
        // inner to outer and the first valid bound becomes the burst (`AccessDetails.cpp:796-830`).
        burst_size: Elements(trips),
    }
}

/// THE PER-DIMENSION WIDTHS AN `affine_set` PINS, for the hyper-rectangular case.
///
/// ⛔ `d == 0` IS WIDTH ONE AND `-d + n >= 0` IS WIDTH `n + 1` — the two shapes
/// [`crate::islands::dataflow_ir::ty::IntegerSet::from_sizes`] writes, read back.
fn rectangle(set: &crate::islands::dataflow_ir::ty::IntegerSet) -> Vec<u64> {
    use crate::islands::dataflow_ir::ty::AffineExpr;
    let mut widths = vec![1u64; set.dims as usize];
    for c in &set.constraints {
        // ⛔ THE UPPER BOUND'S SHAPE IS `-d<k> + (n-1) >= 0`, which `from_sizes` builds as
        // `dim.times(-1).plus(Const(n-1))` — an `Add` of a `Mul` and a `Const`, not a flat sum. A
        // pattern that expected a term list would match nothing and silently leave every width at
        // one, which is a transfer of a single element.
        if let AffineExpr::Add(lhs, rhs) = &c.expr {
            if let (AffineExpr::Mul(inner, factor), AffineExpr::Const(bound)) =
                (lhs.as_ref(), rhs.as_ref())
            {
                if let (AffineExpr::Dim(dim), AffineExpr::Const(-1)) =
                    (inner.as_ref(), factor.as_ref())
                {
                    widths[*dim as usize] = u64::try_from(bound + 1).unwrap_or(1);
                }
            }
        }
    }
    widths
}

/// WHAT ENCLOSES A TERMINATOR — the one input `AffineYieldOpLowering` reads besides the op itself.
///
/// ⛔ THE KIND, NOT THE OP. The reference asks exactly one question of the parent —
/// `isa<scf::ParallelOp>(op->getParentOp())` — so carrying the whole parent would offer callers a
/// dozen other questions the rule does not ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parent {
    /// An `scf.parallel`. ⛔ THE ONE CASE THAT DECLINES.
    ScfParallel,
    /// Anything else — an `affine.for`, an `scf.if`, a `dataflow.program_unit`.
    Other,
}

/// THE RESULT OF THE REWRITE — ⛔ DECLINING IS NOT FAILING.
///
/// ⛔⛔ THE REFERENCE RETURNS `LogicalResult::failure()` AND THAT IS NOT AN ERROR. In MLIR a pattern
/// returning failure means *this pattern does not apply here*; the driver tries others and the module
/// is untouched. `AffineParallelLowering` rewrites the terminator instead
/// (`AffineToStandard.cpp:42-46`: *"Terminator is rewritten as part of the 'affine.parallel' lowering
/// pattern."*). Modelling it as an error would stop a lowering that the reference completes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum YieldRewrite {
    /// The `scf.yield` that replaces the `affine.yield`.
    Yielded(dfir_op::scf::Op),
    /// This pattern does not apply — the parallel lowering owns this terminator.
    Declined,
}

/// **001/490** `AffineYieldOpLowering::matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41` (8L).
///
/// ```cpp
/// LogicalResult matchAndRewrite(AffineYieldOp op, PatternRewriter &rewriter) const override {
///   if (isa<scf::ParallelOp>(op->getParentOp())) {
///     // Terminator is rewritten as part of the "affine.parallel" lowering pattern.
///     return failure();
///   }
///   rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
///   return success();
/// }
/// ```
///
/// ⛔ THE OPERANDS CARRY THROUGH UNCHANGED. `replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands())`
/// passes the yield's own operand list to the new op, so a loop carrying two values yields two. An
/// empty list is a terminator of a loop that carries nothing, not a missing list.
pub fn lower_affine_yield(parent: Parent, operands: &[Val]) -> YieldRewrite {
    match parent {
        Parent::ScfParallel => YieldRewrite::Declined,
        Parent::Other => YieldRewrite::Yielded(dfir_op::scf::Op::Yield {
            operands: operands.to_vec(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 🎯 001/490 — AN `affine.yield` BECOMES AN `scf.yield` CARRYING THE SAME OPERANDS.
    #[test]
    fn an_affine_yield_becomes_an_scf_yield() {
        let carried = [Val(7), Val(9)];
        assert_eq!(
            lower_affine_yield(Parent::Other, &carried),
            YieldRewrite::Yielded(dfir_op::scf::Op::Yield {
                operands: vec![Val(7), Val(9)],
            }),
            "the operand list passes through unchanged"
        );
    }

    /// 🎯 001/490 — AND A LOOP CARRYING NOTHING STILL YIELDS.
    ///
    /// ⛔ AN EMPTY OPERAND LIST IS A TERMINATOR, not an absent one. The reference has no arm for it:
    /// `getOperands()` on a bare `affine.yield` is empty and the rewrite runs anyway.
    #[test]
    fn a_yield_with_no_carried_values_still_rewrites() {
        assert_eq!(
            lower_affine_yield(Parent::Other, &[]),
            YieldRewrite::Yielded(dfir_op::scf::Op::Yield { operands: Vec::new() })
        );
    }

    /// 🎯 001/490 — UNDER AN `scf.parallel` THE PATTERN DECLINES.
    ///
    /// ⛔ AND DECLINING MUST NOT EMIT. The parallel lowering rewrites this terminator itself; a
    /// rewrite here as well would produce two `scf.yield`s for one `affine.yield`.
    #[test]
    fn under_a_parallel_parent_the_pattern_declines() {
        assert_eq!(lower_affine_yield(Parent::ScfParallel, &[Val(1)]), YieldRewrite::Declined);
    }

    /// 🎯 001/490 — AND THE TWO TERMINATORS PRINT AS THE REFERENCE WRITES THEM.
    #[test]
    fn the_terminators_print() {
        use crate::islands::dataflow_ir::print;
        let mut out = String::new();
        print::emit(&mut out, &DfirOp::Scf(dfir_op::scf::Op::Yield { operands: vec![Val(3)] }), 0);
        assert_eq!(out.trim(), "scf.yield %3");
        out.clear();
        print::emit(&mut out, &DfirOp::Affine(dfir_op::affine::Op::Yield { operands: Vec::new() }), 0);
        assert_eq!(out.trim(), "affine.yield");
    }
}
