//! THE SENTIENTIR OPS. One variant per operation an emitted program contains, under the dialect
//! that declares it.
//!
//! ⭐⭐ THE RUNG IS **MIXED**, AND THAT IS MEASURED, NOT ASSUMED. Running a real granite program
//! (`g0_0_mul`, from the bake's `group_0`) through the reference's own D1-D28 and dumping after
//! `DataflowToSentientLoweringPass` leaves:
//!
//! ```text
//! %0 = sentient.scalar_constant {value = 0 : si64} : index
//! %1 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
//! %3 = dataflow.get_logical_memory_view %1, %0 {layout_map = #map} : ...
//! agen.composite_load_and_store src:%3[0, 0] dst:%4[0, 0] ...
//! ```
//!
//! One `sentient.*` op and three that are not. So SentientIR is not "the DataflowIR ops replaced" —
//! it is the same module with the compute and transfer bodies progressively rewritten, and
//! `dataflow.get_unit`, `dataflow.get_logical_memory_view` and the `agen` composites survive well
//! past the conversion named after them.
//!
//! ⛔ WHICH IS WHY THE SHARED DIALECTS ARE **RE-EXPORTED, NOT RE-DECLARED**. An `agen.vector_load`
//! is one operation of one dialect (`Agen.td`) whichever rung holds it, and two definitions of it
//! would be two things to keep in step. [`crate::islands::dataflow_ir::dialects`] owns them.
//!
//! ⚠️ EVENTUALLY THOSE SHARED MODULES BELONG AT `islands::dialects`, above both islands, rather than
//! under the lower one. That is a move across a file the DataflowIR island owns, so it wants doing
//! deliberately and not as a side effect of adding this island.

/// Upstream `affine` — re-exported; see the module note.
pub use crate::islands::dataflow_ir::dialects::affine;
/// `Agen.td` — re-exported; the composites outlive `AgenToSentient`.
pub use crate::islands::dataflow_ir::dialects::agen;
/// Upstream `arith` — re-exported.
pub use crate::islands::dataflow_ir::dialects::arith;
/// `Dataflow.td` — re-exported; units and views survive the whole rung.
pub use crate::islands::dataflow_ir::dialects::dataflow;
/// Upstream `scf` — re-exported.
pub use crate::islands::dataflow_ir::dialects::scf;
/// `Symbol.td` — re-exported; a symbolic loop bound survives into this rung. See [`Op::Symbol`].
pub use crate::islands::dataflow_ir::dialects::symbol;
/// `Uniform.td` — re-exported; a local region survives this rung. See [`Op::Uniform`].
pub use crate::islands::dataflow_ir::dialects::uniform;
/// Upstream `vector` — re-exported; the two plain accesses the vectorchain lowerings read.
pub use crate::islands::dataflow_ir::dialects::vector;
/// `VectorChain.td` — re-exported; what `VectorChainToSentientPE_SFP`/`_PT` consume.
pub use crate::islands::dataflow_ir::dialects::vectorchain;

pub mod sentient;

/// AN SSA VALUE — ⭐ THE SAME TYPE THE LOWER RUNG USES, re-exported.
///
/// ⛔⛔ NOT A DISTINCT NEWTYPE, AND THE FIRST VERSION OF THIS FILE GOT IT WRONG. I gave this island
/// its own `Val` on the reasoning that each rung numbers its values independently. It does not: a
/// Sentient module is the DataflowIR module *progressively rewritten*, one numbering throughout, and
/// the dump in this file's own note proves it — `%0 = sentient.scalar_constant` sits beside
/// `%1 = dataflow.get_unit` in one function.
///
/// ⛔ AND A SEPARATE TYPE WAS NOT MERELY REDUNDANT BUT UNBUILDABLE: the shared dialects' `Op`s carry
/// [`crate::islands::dataflow_ir::dialects::Val`] in their own fields, so `Op::Dataflow` would have
/// held one value type while `Op::Sentient` held another, and no printer could take both.
pub use crate::islands::dataflow_ir::dialects::Val;

/// The value minter and the `IRMapping` a clone at this rung needs — see [`clone_ops`].
use crate::islands::dataflow_ir::{ValueMapping, Values};

/// AN `affine.for` WHOSE BODY HAS REACHED THIS RUNG — the loop [`Op::Affine`] cannot hold.
///
/// ⛔⛔ THE VENDOR'S OWN OUTPUT PUTS A `sentient.load_and_store` INSIDE AN `affine.for` BODY
/// (`dcc/test/Conversion/AgenToSentient/l3-gather.mlir:33-40`), and the shared `affine::Op::For`
/// carries `Vec<`[`crate::islands::dataflow_ir::dialects::Op`]`>` — the rung below, which has no
/// `Sentient` arm. So [`Op::Affine`] is a loop still entirely below this rung and this is one the
/// conversion has written a transfer statement into. The two print identically, through the one
/// rendering in [`affine::for_header`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineFor {
    /// The induction variable — a region argument, so nothing defines it.
    pub iv: Val,
    /// The lower bound; always `0` for a time loop.
    pub lo: affine::Bound,
    /// The trip count.
    pub hi: affine::Bound,
    /// The addresses the loop carries — its `iter_args`, one per access detail.
    pub carried: Vec<affine::Carried>,
    /// The body, at THIS rung.
    pub body: Vec<Op>,
    /// `dbgName`, printed after the closing brace.
    pub dbg_name: Option<String>,
}

/// ONE REGION OF A [`UniformRegions`], WITH THE UNITS IT IS MAPPED ONTO — the same record as
/// [`uniform::LocalRegion`], with a body that has reached THIS rung.
///
/// ⛔⛔ IT IS THE SAME REGION, NOT A SECOND KIND OF ONE. Every field means what
/// [`uniform::LocalRegion`]'s means and every citation there applies here; only [`Self::body`]'s
/// element type differs. See [`UniformRegions`] for why that one difference needs a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRegion {
    /// `getRegionArg(i)` (`Uniform.td:96`) — *"whichever unit of [`Self::units`] is running this
    /// region"*, and the value that NAMES this region: each region binds its own.
    pub arg: Val,
    /// The units this region runs on — `getRegionUnitList(i)` (`Uniform.cpp:184`).
    pub units: Vec<Val>,
    /// What the region runs, AT THIS RUNG, ending in a `uniform.yield`
    /// ([`Op::Uniform`]`(`[`uniform::Op::Yield`]`)`).
    pub body: Vec<Op>,
}

/// A `uniform.uniformize_regions` OR `uniform.equalize_pattern` WHOSE REGIONS HOLD OPS OF **THIS**
/// RUNG — the op [`Op::Uniform`] cannot hold.
///
/// # ⛔⛔ WITHOUT IT NOTHING CAN BE PUT INTO A LOCAL REGION, AND `SinkScalarCopy` EXISTS TO DO THAT
///
/// [`uniform::LocalRegion::body`] is `Vec<`[`crate::islands::dataflow_ir::dialects::Op`]`>` — the rung
/// BELOW, which has no arm for a `sentient.scalar_copy` at all. `SinkScalarCopyPass::sinkCopyOps`
/// clones exactly that op into a local region (`OpBuilder builder(region.getPointer()); Operation
/// *new_op = builder.clone(orig_op)`, `Transform/Sentient/SinkScalarCopy.cpp:222-223`), so with only
/// [`Op::Uniform`] the pass's whole effect is a type error rather than a port.
///
/// ⚠️ THE GAP WAS ALREADY RECORDED, by
/// [`OriginalRegion`](crate::transform::sentient::local_region_splitting_for_value_commoning::local_region::OriginalRegion),
/// which names *"a sentient-rung `uniform.uniformize_regions` beside [`AffineFor`]"* as the fix and
/// e444_analyze/e505_transform as the other units that need it. This is that op.
///
/// ⭐ AN ENUM AND NOT A `kind` FIELD, BECAUSE ONLY ONE OF THE TWO BINDS RESULTS.
/// `EqualizePatternOp` declares `$units`, `$list_sizes` and a `VariadicRegion` and no `let results` at
/// all (`Uniform.td:196-198`) — so `EqualizePattern { results: .. }` must be unwritable, exactly as it
/// is in [`uniform::Op`].
///
/// ⭐ IT PRINTS THROUGH [`uniform`]'S OWN HEADER FUNCTIONS, so the two rungs cannot disagree about a
/// character — the same arrangement [`AffineFor`] has with [`affine::for_header`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniformRegions {
    /// `uniform.uniformize_regions -> (..) { (%arg -> %0, %2){ .. } .. }` — see
    /// [`uniform::Op::UniformizeRegions`].
    UniformizeRegions {
        /// One record per region, in region order.
        regions: Vec<LocalRegion>,
        /// `Variadic<AnyType>:$results` (`Uniform.td:90`) — one per operand of every region's
        /// `uniform.yield`.
        results: Vec<Val>,
    },
    /// `uniform.equalize_pattern { (%arg -> %0, %2){ .. } .. }` — see
    /// [`uniform::Op::EqualizePattern`].
    EqualizePattern {
        /// One record per region, in region order.
        regions: Vec<LocalRegion>,
    },
}

impl UniformRegions {
    /// ITS REGIONS, whichever of the two ops this is.
    #[must_use]
    pub fn regions(&self) -> &[LocalRegion] {
        match self {
            UniformRegions::UniformizeRegions { regions, .. }
            | UniformRegions::EqualizePattern { regions } => regions,
        }
    }

    /// ITS REGIONS, FOR A REWRITE — what a sink writes into.
    pub fn regions_mut(&mut self) -> &mut Vec<LocalRegion> {
        match self {
            UniformRegions::UniformizeRegions { regions, .. }
            | UniformRegions::EqualizePattern { regions } => regions,
        }
    }
}

/// ONE SENTIENTIR OPERATION, under the dialect that declares it.
///
/// ⛔ NO `_` ARM WHERE THIS IS MATCHED. A new dialect reaching this rung must be a build error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `SentientOps.td` — ports, registers, forwarding, precision, unroll. The rung's own dialect.
    Sentient(sentient::Op),
    /// `Dataflow.td` — units and views, which survive to the end of the rung.
    Dataflow(dataflow::Op),
    /// `Agen.td` — the composites still awaiting a `sentient.load_and_store`.
    Agen(agen::Op),
    /// `VectorChain.td` — computes still awaiting a `sentient.vector_*`.
    VectorChain(vectorchain::Op),
    /// Upstream `affine` — the loop nest and the applied maps, body and all below this rung.
    Affine(affine::Op),
    /// Upstream `affine` — a loop whose body holds ops of THIS rung. See [`AffineFor`].
    AffineFor(AffineFor),
    /// Upstream `vector` — a plain access still awaiting its `sentient` form.
    Vector(vector::Op),
    /// Upstream `arith` — constants and predicates.
    Arith(arith::Op),
    /// Upstream `scf`.
    Scf(scf::Op),
    /// `Symbol.td` — a symbolic extent, still unresolved at this rung.
    ///
    /// # ⛔⛔ WITHOUT IT `getForOpBound` HAS NO SYMBOLIC ARM TO TAKE
    ///
    /// Entry 091 reads a `sentient.for`'s trip count by walking its bound operand backwards, and the
    /// walk ends at one of TWO ops:
    ///
    /// ```cpp
    /// if (auto const_op = sub_op.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    ///   return const_op.value();
    /// } else if (auto symbol_op =
    ///                sub_op.getLhs().getDefiningOp<mlir::symbol::CreateSymbolOp>()) {
    ///   … return 0;
    /// }
    /// ```
    /// (`Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:278-288`)
    ///
    /// The loop being asked about is a `sentient.for` — the reference's own note says *"At this point
    /// in the pipeline all loops have been lowered to sentient.for operations"*
    /// (`VectorChainToSentientPT/Helper.cpp:141-143`) — so the `symbol.create_symbol` it may find
    /// behind that loop's bound is an op sitting in a SENTIENT module. With this arm absent, the
    /// symbolic case is not a branch this island can be shown; it would have to be a `todo!`, and the
    /// campaign brief's rule for exactly that situation is to add the op to the island rather than
    /// declare the input unreachable.
    ///
    /// ⭐ RE-EXPORTED, NOT RE-DECLARED, like every other shared dialect here — see the module note.
    Symbol(symbol::Op),
    /// `Uniform.td` — a local region, still unflattened at this rung.
    ///
    /// ⛔⛔ NEITHER OF `SCFToSentient`'S TWO LISTS MENTIONS `uniform`
    /// (`SCFToSentient.cpp:261-266`), so a `uniform.uniformize_regions` is left exactly where it is
    /// by that partial conversion — which entry 225 discovered the hard way: it lifts an
    /// `scf.for`'s whole body onto this rung, and without this arm a loop with a local region in it
    /// could not be lifted at all.
    ///
    /// ⭐ ITS REGIONS STAY `Vec<`[`crate::islands::dataflow_ir::dialects::Op`]`>`, like every other
    /// shared dialect's — the op is the same op whichever rung holds it. ⛔ WHICH IS WHY
    /// [`Op::UniformRegions`] EXISTS: a local region holding a `sentient.*` op is not this variant.
    Uniform(uniform::Op),
    /// `Uniform.td` — one of the two region-carrying `uniform` ops whose regions hold ops of THIS
    /// rung. See [`UniformRegions`].
    UniformRegions(UniformRegions),
}

/// ONE SHARED-DIALECT OP AS ITS LOWER-RUNG SELF — `None` for this rung's own dialect.
///
/// ⛔⛔ THE SHARED DIALECTS ARE RE-EXPORTED, NOT RE-DECLARED (see this module's note), so an
/// `agen.composite_load_and_store` sitting in a Sentient module IS a
/// [`crate::islands::dataflow_ir::dialects::Op`] value — and the DataflowIR island's own total
/// `operands`/`results`/`uses` walks already describe it. Re-answering those questions here would be
/// a second description of one operation to keep in step, which is exactly the defect this island's
/// header forbids.
///
/// ⭐ PUBLIC BECAUSE ONE CALLER IS A BRIDGE, NOT THIS ISLAND: entry 367 walks a body holding
/// `sentient.for` around still-unlowered `agen`/`vector` accesses and has to ask the rung below
/// about each of them (its `XrfCensus`).
///
/// ⭐ A CLONE, BECAUSE THESE ARE QUERIES. The answer is read and dropped; nothing is written through
/// it. Where a REWRITE has to reach a shared op, the callers below say so out loud rather than
/// pretending the walk covered it.
#[must_use]
pub fn lowered(op: &Op) -> Option<crate::islands::dataflow_ir::dialects::Op> {
    use crate::islands::dataflow_ir::dialects::Op as LowerOp;
    match op {
        Op::Sentient(_) | Op::AffineFor(_) | Op::UniformRegions(_) => None,
        Op::Dataflow(op) => Some(LowerOp::Dataflow(op.clone())),
        Op::Agen(op) => Some(LowerOp::Agen(op.clone())),
        Op::VectorChain(op) => Some(LowerOp::VectorChain(op.clone())),
        Op::Affine(op) => Some(LowerOp::Affine(op.clone())),
        Op::Vector(op) => Some(LowerOp::Vector(op.clone())),
        Op::Arith(op) => Some(LowerOp::Arith(op.clone())),
        Op::Scf(op) => Some(LowerOp::Scf(op.clone())),
        Op::Symbol(op) => Some(LowerOp::Symbol(op.clone())),
        Op::Uniform(op) => Some(LowerOp::Uniform(op.clone())),
    }
}

/// ONE LOWER-RUNG OP AS A MEMBER OF THIS RUNG — the inverse of `lowered`, and TOTAL.
///
/// ⛔⛔ WHAT `inlineRegionBefore` NEEDS. Entry 225 moves an `scf.for`'s body into the
/// `sentient.for` that replaces it, and that body is a `Vec` of the rung below's ops: without a
/// total lift, lowering a loop would have to refuse one whose body holds a dialect this island's
/// union had forgotten. Every one of the lower rung's nine dialects has an arm here, so a tenth is a
/// build error rather than a refusal at that call site.
#[must_use]
pub fn raised(op: crate::islands::dataflow_ir::dialects::Op) -> Op {
    use crate::islands::dataflow_ir::dialects::Op as LowerOp;
    match op {
        LowerOp::Dataflow(op) => Op::Dataflow(op),
        LowerOp::Agen(op) => Op::Agen(op),
        LowerOp::VectorChain(op) => Op::VectorChain(op),
        LowerOp::Affine(op) => Op::Affine(op),
        LowerOp::Vector(op) => Op::Vector(op),
        LowerOp::Arith(op) => Op::Arith(op),
        LowerOp::Scf(op) => Op::Scf(op),
        LowerOp::Symbol(op) => Op::Symbol(op),
        LowerOp::Uniform(op) => Op::Uniform(op),
    }
}

/// THE VALUES AN OP OF THIS RUNG BINDS AS RESULTS — `getResult(n)`, whichever dialect declares it.
///
/// ⛔ ONE `arith.constant` IS ONE OP WHICHEVER RUNG HOLDS IT, so the shared arms delegate; see
/// [`lowered`].
#[must_use]
pub fn results(op: &Op) -> Vec<Val> {
    match op {
        Op::Sentient(op) => sentient::results(op),
        // ⭐ AN `iter_args` LOOP BINDS ONE RESULT PER CARRIED ADDRESS — the same answer
        // `affine::Op::For` gets from the rung below.
        Op::AffineFor(loop_op) => loop_op.carried.iter().map(|c| c.result).collect(),
        // ⭐ THE SAME ANSWER [`uniform::Op`] GETS FROM THE RUNG BELOW — `$results`
        // (`Uniform.td:90`) for `uniformize_regions`, and none at all for `equalize_pattern`, which
        // declares no `let results` (`:196-198`).
        Op::UniformRegions(UniformRegions::UniformizeRegions { results, .. }) => results.clone(),
        Op::UniformRegions(UniformRegions::EqualizePattern { .. }) => Vec::new(),
        // ⛔ BOUND RATHER THAN A BARE `_`: the shared dialects delegate, and a variant of THIS rung
        // that fell in here would be answered "no results" instead of being a build error.
        other @ (Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_)) => lowered(other).map_or_else(Vec::new, |op| {
            crate::islands::dataflow_ir::dialects::results(&op)
        }),
    }
}

/// THE VALUES AN OP OF THIS RUNG **READS** — `getOpOperands()`, whichever dialect declares it.
///
/// # 🛑 THE USE-WALK AND THE EQUIVALENCE TEST BOTH START HERE
///
/// ⛔⛔ [`replace_all_uses_with`] COULD WRITE AN OPERAND BUT NOTHING COULD *ASK* WHICH ONES THERE
/// WERE, and two of loop rolling's questions are exactly that. `Window::checkOpUsage` refuses to roll
/// an op whose result is read outside its window (`dcc/src/Transform/Sentient/LoopRolling.cpp:144`) —
/// a use-walk, which is this list asked of every op in the block — and
/// `collectDeltasOfOperands` walks `op_a.getOperand(i)` against `op_b.getOperand(i)` by INDEX
/// (`:400-402`), so the position in this list is the `i` a delta and an `OperandKind` are keyed by.
///
/// ⛔ THE ORDER IS [`sentient::operands_mut`]'S FOR THIS RUNG'S OWN DIALECT and the rung below's for
/// every shared one, because it is the same `i`. ⚠️ AND FOR TWO OPS IT IS NOT THE `.td`'S: this
/// island models `sentient.load_and_send`'s `$consumer` and `sentient.receive_and_store`'s
/// `$producer` as a wire end rather than a [`Val`], so both lists are one entry short from position 3
/// on. Positions 0-2 — the mutable address, the immutable address and the increment, which are the
/// fields loop rolling keys deltas by — are the `.td`'s (`SentientOps.td:507-509`, `:550-552`).
///
/// ⭐ COPIES, LIKE THE RUNG BELOW'S: this answers a question. A REWRITE goes through
/// [`replace_all_uses_with`], or through [`sentient::operands_mut`] for one slot of one op.
#[must_use]
pub fn operands(op: &Op) -> Vec<Val> {
    match op {
        // ⭐ DERIVED FROM THE MUTABLE WALK RATHER THAN RESTATED, so the twenty-nine arms cannot drift
        // apart: a clone answered and dropped is the cost of having one description of the list.
        Op::Sentient(inner) => {
            let mut copy = inner.clone();
            sentient::operands_mut(&mut copy)
                .into_iter()
                .map(|val| *val)
                .collect()
        }
        // ⭐ THE SAME LIST `affine::Op::For` GETS FROM THE RUNG BELOW — a constant bound is no
        // operand, and a carried value's initialiser is one. The two variants print identically, so
        // they must answer identically.
        Op::AffineFor(loop_op) => {
            let mut reads: Vec<Val> = Vec::new();
            for bound in [&loop_op.lo, &loop_op.hi] {
                if let affine::Bound::Val(val) = bound {
                    reads.push(*val);
                }
            }
            reads.extend(loop_op.carried.iter().map(|carried| carried.init));
            reads
        }
        // ⭐ THE UNITS, CONCATENATED IN REGION ORDER — the one flat `$units` range the reference
        // slices by the prefix sum of `$list_sizes` (`Uniform.cpp:184-192`), exactly as the rung
        // below answers for [`uniform::Op`]. ⛔ THE REGION ARGUMENTS ARE NOT HERE: they are block
        // arguments (`Uniform.td:96`), which is what [`parent_uniform_region_units`] answers for.
        Op::UniformRegions(regions) => regions
            .regions()
            .iter()
            .flat_map(|region| region.units.iter().copied())
            .collect(),
        // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
        other @ (Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_)) => lowered(other).map_or_else(Vec::new, |op| {
            crate::islands::dataflow_ir::dialects::operands(&op)
        }),
    }
}

/// WRITE ONE OPERAND SLOT — `Operation::setOperand(i, v)`.
///
/// ⛔⛔ [`replace_all_uses_with`] IS THE WRONG TOOL FOR THIS AND LOOP ROLLING PROVES IT.
/// `updateBody` re-points ONE field of ONE transfer at a loop's `iter_arg`
/// (`dcc/src/Transform/Sentient/LoopRolling.cpp:712`, `:727-728`) while every other reader of the
/// value that field held must keep reading it — a value-keyed rewrite would move all of them.
///
/// ⛔ THE INDEX IS [`operands`]'S, so the two must be read together; see that function's note on the
/// two ops whose list is not the `.td`'s.
///
/// ⭐ AN OUT-OF-RANGE SLOT WRITES NOTHING, which is the one thing MLIR's own `setOperand` cannot do:
/// it asserts. There is no operand to name, so there is nothing to correct.
pub fn set_operand(op: &mut Op, at: usize, val: Val) {
    match op {
        Op::Sentient(inner) => {
            if let Some(slot) = sentient::operands_mut(inner).into_iter().nth(at) {
                *slot = val;
            }
        }
        // ⭐ THE SAME ORDER [`operands`] ANSWERS IN — a constant bound occupies no slot.
        Op::AffineFor(loop_op) => {
            let mut slot = 0usize;
            for bound in [&mut loop_op.lo, &mut loop_op.hi] {
                if let affine::Bound::Val(bound) = bound {
                    if slot == at {
                        *bound = val;
                        return;
                    }
                    slot += 1;
                }
            }
            for carried in &mut loop_op.carried {
                if slot == at {
                    carried.init = val;
                    return;
                }
                slot += 1;
            }
        }
        // ⭐ THE SAME ORDER [`operands`] ANSWERS IN — region by region, units within a region.
        Op::UniformRegions(regions) => {
            let mut slot = 0usize;
            for region in regions.regions_mut() {
                for unit in &mut region.units {
                    if slot == at {
                        *unit = val;
                        return;
                    }
                    slot += 1;
                }
            }
        }
        // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
        other @ (Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_)) => {
            if let Some(mut lower) = lowered(other) {
                if let Some(slot) = crate::islands::dataflow_ir::dialects::operands_mut(&mut lower)
                    .into_iter()
                    .nth(at)
                {
                    *slot = val;
                }
                *other = raised(lower);
            }
        }
    }
}

/// THE REGIONS AN OP OF THIS RUNG HOLDS, AT THIS RUNG'S TYPE — `Operation::getRegions()`.
///
/// ⛔⛔ ⚠️ **OWNED, UNLIKE [`sentient::regions`]**, AND THAT IS WHAT MAKES IT TOTAL. A shared
/// dialect's region holds `Vec<`[`crate::islands::dataflow_ir::dialects::Op`]`>` — the rung below's
/// type, because the op is the same op whichever rung holds it (see [`lowered`]) — so a borrowing
/// signature could return the `sentient.for` and `affine.for` bodies and nothing else, and a walk
/// built on it would silently stop at a `uniform.uniformize_regions`. [`raised`] is what puts those
/// ops back in this rung's vocabulary, and it produces values.
///
/// ⭐ THE COST IS A CLONE PER SHARED REGION, paid only by callers that descend. Loop rolling's
/// use-walk is one: a result read from inside another op's region is read OUTSIDE the window that
/// defines it, which is what `Window::checkOpUsage` refuses to roll
/// (`dcc/src/Transform/Sentient/LoopRolling.cpp:144-155`).
#[must_use]
pub fn regions(op: &Op) -> Vec<Vec<Op>> {
    match op {
        Op::Sentient(inner) => sentient::regions(inner)
            .into_iter()
            .map(<[Op]>::to_vec)
            .collect(),
        Op::AffineFor(loop_op) => vec![loop_op.body.clone()],
        // ⭐ ALREADY THIS RUNG'S TYPE, so this is the one variant the clone costs nothing to state;
        // [`regions_ref`] is the borrowing form.
        Op::UniformRegions(regions) => regions
            .regions()
            .iter()
            .map(|region| region.body.clone())
            .collect(),
        // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
        other @ (Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_)) => lowered(other).map_or_else(Vec::new, |op| {
            crate::islands::dataflow_ir::dialects::regions(&op)
                .into_iter()
                .map(|region| region.iter().cloned().map(raised).collect())
                .collect()
        }),
    }
}

/// THE OP THAT DEFINES A VALUE AS A RESULT — `Value::getDefiningOp()`.
///
/// ⛔⛔ `getForOpBound` (entry 091) IS FOUR OF THESE IN A ROW. It walks a `sentient.for`'s bound
/// backwards — `arith.divsi` → its `arith.subi` lhs → the `arith.constant` or `symbol.create_symbol`
/// that feeds THAT — and every step is a `getDefiningOp`
/// (`Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262-294`). Without this
/// the trip count of a lowered loop is unreadable and the xrf pointer's travel distance
/// (`:498`) cannot be computed.
///
/// ⛔ `None` FOR A REGION ARGUMENT, which is the null pointer the reference gets — the answer
/// `getMaskValueForPT` distinguishes a loop iterator by (`Helper.cpp:147-151`).
#[must_use]
pub fn defining_op(val: Val, scope: &[Op]) -> Option<&Op> {
    for op in scope {
        if results(op).contains(&val) {
            return Some(op);
        }
        match op {
            Op::Sentient(inner) => {
                for region in sentient::regions(inner) {
                    if let Some(found) = defining_op(val, region) {
                        return Some(found);
                    }
                }
            }
            Op::AffineFor(loop_op) => {
                if let Some(found) = defining_op(val, &loop_op.body) {
                    return Some(found);
                }
            }
            // ⭐ A LOCAL REGION AT THIS RUNG IS DESCENDED INTO AND ITS ANSWER RETURNED, unlike the
            // shared-dialect arm below — its ops are values of THIS island's type.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    if let Some(found) = defining_op(val, &region.body) {
                        return Some(found);
                    }
                }
            }
            // ⛔ A DEFINITION INSIDE A LOWER-RUNG REGION IS PROVED ABSENT, NOT ASSUMED ABSENT. The
            // DataflowIR island's own walk descends into the region and answers for the whole
            // subtree; only if it finds the definition is there nothing this signature can return,
            // because the op it found is a value of the other island's type.
            // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
            other @ (Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_)) => {
                if let Some(op) = lowered(other)
                    && crate::islands::dataflow_ir::dialects::defining_op(
                        val,
                        core::slice::from_ref(&op),
                    )
                    .is_some()
                {
                    todo!("getDefiningOp reached a definition inside a lower-rung region: {op:?}");
                }
            }
        }
    }
    None
}

/// REWIRE EVERY USE OF ONE VALUE TO ANOTHER — `Value::replaceAllUsesWith`.
///
/// ⛔⛔ THIS IS WHAT ENTRY 093 DOES, AND IT MUST BE COMPLETE OR IT IS WORSE THAN NOTHING. A dummy
/// `sentient.mac` is a placeholder standing where a real compute's xrf pointer will be
/// (`LoweringXRF.cpp:312-327`); `replaceAndEraseDummyMacOps` moves every reader onto the real result
/// and then erases the placeholder (`:681-688`). A rewrite that missed one reader would leave that
/// reader pointing at an op that no longer exists.
///
/// ⛔ SO THE SHARED DIALECTS ARE **REWRITTEN, NOT SKIPPED**, and they were once a [`todo!`] that
/// only said so. Entry 225 is the caller that needs them: it re-points every reader of a lowered
/// loop's induction variable at `arith.subi %bound, %iv`, and the readers inside a lowered loop body
/// ARE `agen`, `vector`, `arith` and `dataflow` ops — the check would have stopped the build on
/// every real loop. The rewrite goes through the DataflowIR island's own [`raised`]/[`lowered`] pair
/// and its total per-op walk, so it cannot reach fewer ops than the check did.
pub fn replace_all_uses_with(scope: &mut [Op], of: Val, with: Val) {
    for op in scope {
        match op {
            Op::Sentient(inner) => {
                for operand in sentient::operands_mut(inner) {
                    if *operand == of {
                        *operand = with;
                    }
                }
                for region in sentient::regions_mut(inner) {
                    replace_all_uses_with(region, of, with);
                }
            }
            Op::AffineFor(loop_op) => {
                for bound in [&mut loop_op.lo, &mut loop_op.hi] {
                    if let affine::Bound::Val(v) = bound
                        && *v == of
                    {
                        *bound = affine::Bound::Val(with);
                    }
                }
                for carried in &mut loop_op.carried {
                    if carried.init == of {
                        carried.init = with;
                    }
                }
                replace_all_uses_with(&mut loop_op.body, of, with);
            }
            // ⭐ THE UNITS ARE THE USES AND THE BODIES ARE DESCENDED INTO. ⛔ NOT
            // [`LocalRegion::arg`]: a region argument is BOUND by the region, not read by the op, so
            // rewriting it would rename a definition (see [`operands`]).
            Op::UniformRegions(regions) => {
                for region in regions.regions_mut() {
                    for unit in &mut region.units {
                        if *unit == of {
                            *unit = with;
                        }
                    }
                    replace_all_uses_with(&mut region.body, of, with);
                }
            }
            // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
            other @ (Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_)) => {
                if let Some(mut lower) = lowered(other) {
                    replace_all_uses_in_lower(&mut lower, of, with);
                    *other = raised(lower);
                }
            }
        }
    }
}

/// [`replace_all_uses_with`] inside one shared-dialect op, regions included.
fn replace_all_uses_in_lower(
    op: &mut crate::islands::dataflow_ir::dialects::Op,
    of: Val,
    with: Val,
) {
    use crate::islands::dataflow_ir::dialects as lower;
    lower::replace_uses_of_with(op, of, with);
    for region in lower::regions_mut(op) {
        for inner in region.iter_mut() {
            replace_all_uses_in_lower(inner, of, with);
        }
    }
}

/// HOW MANY TIMES ONE VALUE IS **READ** IN `scope` — `Value::hasOneUse()` and `Value::use_empty()`.
///
/// ⛔⛔ ONE PER USE, NOT PER USER, BECAUSE THAT IS WHAT MLIR COUNTS. `hasOneUse()` is false for a
/// value one op reads twice, so `use_count(v, scope) == 1` is exactly `hasOneUse()` and `== 0` is
/// exactly `use_empty()`. `coalesceScalarArithSimplification` (entry 054) spends this on both of an
/// add's operands: it only rewrites when an op will be REMOVED to pay for the one it creates
/// (`LightweightSimplification.cpp:125-129`), and an over-count there declines a rewrite the
/// reference performs while an under-count performs one it declines.
///
/// ⛔ AND IT DESCENDS INTO REGIONS, like the rung below's [`crate::islands::dataflow_ir::dialects::uses`]
/// — a scalar read by an op inside a `sentient.for` body is read.
///
/// ⭐ A COUNT OFF A CLONE FOR THIS RUNG'S OWN OPS. [`sentient::operands_mut`] is the ONE operand walk
/// and it hands out PLACES so that [`replace_all_uses_with`] can be performed rather than
/// approximated; a second by-value walk would be a twenty-nine-arm list to keep in step. Cloning to
/// ask a question is what [`lowered`] already does, for that same reason.
#[must_use]
pub fn use_count(of: Val, scope: &[Op]) -> usize {
    let mut count = 0;
    for op in scope {
        match op {
            Op::Sentient(inner) => {
                let mut probe = inner.clone();
                count += sentient::operands_mut(&mut probe)
                    .into_iter()
                    .filter(|read| **read == of)
                    .count();
                for region in sentient::regions(inner) {
                    count += use_count(of, region);
                }
            }
            Op::AffineFor(loop_op) => {
                for bound in [&loop_op.lo, &loop_op.hi] {
                    if let affine::Bound::Val(v) = bound
                        && *v == of
                    {
                        count += 1;
                    }
                }
                count += loop_op
                    .carried
                    .iter()
                    .filter(|carried| carried.init == of)
                    .count();
                count += use_count(of, &loop_op.body);
            }
            // ⭐ ARM FOR ARM WITH [`replace_all_uses_with`], which is what makes the count the number
            // of slots that rewrite would touch.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    count += region.units.iter().filter(|unit| **unit == of).count();
                    count += use_count(of, &region.body);
                }
            }
            // ⭐ THE RUNG BELOW ANSWERS FOR ITS OWN OPS, REGIONS INCLUDED — see [`lowered`].
            // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
            other @ (Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_)) => {
                if let Some(lower) = lowered(other) {
                    count += crate::islands::dataflow_ir::dialects::uses(
                        of,
                        core::slice::from_ref(&lower),
                    )
                    .len();
                }
            }
        }
    }
    count
}

/// REMOVE THE OP THAT DEFINES A VALUE — `Operation::erase()`, reached through its result.
///
/// ⛔⛔ THE ERASE IS HALF OF ENTRY 093 AND IT COMES **AFTER** BOTH REWIRES. `Operation::erase()` on
/// an op that still has uses is MLIR's *"operation destroyed but still has uses"* abort — the same
/// crash the `scf.if` note in [`scf::Op::If`] records — so the order the reference writes the four
/// statements in is load-bearing, not stylistic.
///
/// ⭐ AN ABSENT DEFINITION IS A NO-OP, NOT A REFUSAL: the value is a region argument or already
/// gone, and neither is a state this function can improve on.
pub fn erase_defining_op(scope: &mut Vec<Op>, val: Val) {
    if let Some(at) = scope.iter().position(|op| results(op).contains(&val)) {
        scope.remove(at);
        return;
    }
    for op in scope.iter_mut() {
        match op {
            Op::Sentient(inner) => {
                for region in sentient::regions_mut(inner) {
                    erase_defining_op(region, val);
                }
            }
            Op::AffineFor(loop_op) => erase_defining_op(&mut loop_op.body, val),
            // ⭐ DESCENDED INTO, as in [`defining_op`] — a copy sunk into a local region is erased
            // from that region.
            Op::UniformRegions(regions) => {
                for region in regions.regions_mut() {
                    erase_defining_op(&mut region.body, val);
                }
            }
            // ⛔ PROVED ABSENT, as in [`defining_op`].
            // ⛔ BOUND RATHER THAN A BARE `_` — see [`results`].
            other @ (Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_)) => {
                if let Some(op) = lowered(other)
                    && crate::islands::dataflow_ir::dialects::defining_op(
                        val,
                        core::slice::from_ref(&op),
                    )
                    .is_some()
                {
                    todo!("erase(): the defining op sits in a lower-rung region: {op:?}");
                }
            }
        }
    }
}

/// THE `getDefiningOp` LINK, SUPPLIED RATHER THAN STORED — the enclosing regions a value may be
/// defined in, innermost first.
///
/// # ⛔⛔ ONE REGION IS NOT ENOUGH, AND THE REFERENCE'S OWN FIXTURE PROVES IT
///
/// `getForOpBound` (entry 091) walks four definitions back from a loop's bound, and in
/// `dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:216-227` they do not all live
/// in one block:
///
/// ```text
/// %c2 = arith.constant 2 : index                     <- func body
/// %c0 = arith.constant 0 : index
/// dataflow.program_unit iter_arg : %arg0 -> (%0) {precision = "fp16"} : {
///   %1 = arith.subi %c2, %c0 : index                 <- the unit's region
///   %2 = arith.divsi %1, %c1 : index
///   sentient.for %arg1 = %2 { .. }
/// ```
///
/// The `divsi` and the `subi` are in the unit's region; the constants they read are two levels up. An
/// MLIR `Value` knows its own owner, so `getDefiningOp()` needs no scope at all — this island's ops
/// are a tree with no parent pointers, which the campaign brief names as exactly the *mechanism* a
/// port may drop and supply instead. ⭐ SO THE SCOPE IS THE CALLER'S, and it is a LIST because
/// visibility in MLIR runs outwards through enclosing regions.
///
/// ⛔ INNERMOST FIRST, BECAUSE THE FIRST MATCH WINS. Searching outwards is how MLIR's own lookup
/// reads, and the order matters the moment two regions bind the same [`Val`] — which the minter makes
/// impossible ([`crate::islands::dataflow_ir::Values`]) but which a hand-built fixture can still do.
#[derive(Debug, Clone, Copy)]
pub struct Definitions<'a> {
    regions: &'a [&'a [Op]],
    program_unit: Option<(Val, &'a [Val])>,
}

impl<'a> Definitions<'a> {
    /// THE ENCLOSING REGIONS, INNERMOST FIRST.
    #[must_use]
    pub const fn from_innermost(regions: &'a [&'a [Op]]) -> Definitions<'a> {
        Definitions {
            regions,
            program_unit: None,
        }
    }

    /// THE SAME REGIONS PLUS THE ENCLOSING `dataflow.program_unit`'S OWN ARGUMENT AND UNIT LIST —
    /// `iter_arg : %arg -> (%units)` (`DataflowOps.cpp:145-155`).
    ///
    /// ⛔⛔ THE THIRD PARENT FORM OF `getListOfKeyOpsFromUniformMapping`, AND UNSTATABLE UNTIL NOW.
    /// A key may be bound by a `uniform.uniformize_regions` region, by a `uniform.equalize_pattern`
    /// region, or by the program unit itself, in which case the keys are `prog_unit_op.getUnits()`
    /// (`Dialect/Uniform/Utils.cpp:180-186`). [`crate::islands::sentient::ProgramUnit`] now carries
    /// that argument, so the arm is reachable — see [`uniform_mapping_keys`].
    ///
    /// ⭐ A BUILDER RATHER THAN A SECOND CONSTRUCTOR, so every existing caller keeps
    /// [`Self::from_innermost`] unchanged and only a walk that IS inside a program unit says so.
    #[must_use]
    pub const fn with_program_unit(self, arg: Val, units: &'a [Val]) -> Definitions<'a> {
        Definitions {
            regions: self.regions,
            program_unit: Some((arg, units)),
        }
    }

    /// `Value::getDefiningOp()` — the first enclosing region that binds it.
    ///
    /// ⭐ `None` FOR A BLOCK ARGUMENT, which is the null pointer the reference gets; see
    /// [`defining_op`].
    #[must_use]
    pub fn of(&self, val: Val) -> Option<&'a Op> {
        self.regions
            .iter()
            .find_map(|region| defining_op(val, region))
    }
}

/// WHICH `sentient.for` BINDS A VALUE AS A REGION ARGUMENT, AND AT WHICH POSITION — the
/// `cast<BlockArgument>(val).getOwner()->getParentOp()` plus `getArgNumber()` that every reader of a
/// loop's attribute arrays performs.
///
/// ⭐ POSITION 0 IS THE INDUCTION VARIABLE AND `i + 1` IS `carried[i]`, which is exactly how the two
/// readers index: `getValueRegLocale` takes `locales[0]` for the induction variable and
/// `locales[i + 1]` for region iter arg `i` (`Dialect/Sentient/SentientOps.cpp:1762-1776`), and
/// `getElementSize` indexes `element_sizes` by the raw `getArgNumber()`
/// (`Dialect/Sentient/Utils.cpp:307-315`).
///
/// ⛔ `None` FOR AN OP RESULT — the failed `dyn_cast<BlockArgument>`. And `None` rather than a wrong
/// answer for a region argument of anything else, which is the `DT_CHECK_MSG(for_op, "Expect parent
/// region of block arg to be ForOp")` both readers make (`Utils.cpp:310`) expressed as a fact about
/// the island: `sentient.for` is the only op of this dialect that binds any
/// ([`sentient::block_args`]).
#[must_use]
pub fn parent_for_arg(val: Val, scope: &[Op]) -> Option<(&Op, usize)> {
    for op in scope {
        match op {
            Op::Sentient(inner) => {
                if let Some(index) = sentient::block_args(inner)
                    .iter()
                    .position(|arg| *arg == val)
                {
                    return Some((op, index));
                }
                for region in sentient::regions(inner) {
                    if let Some(found) = parent_for_arg(val, region) {
                        return Some(found);
                    }
                }
            }
            Op::AffineFor(loop_op) => {
                if let Some(found) = parent_for_arg(val, &loop_op.body) {
                    return Some(found);
                }
            }
            // ⭐ DESCENDED INTO, UNLIKE [`Op::Uniform`] BELOW: a local region at THIS rung CAN hold a
            // `sentient.for`. ⛔ AND ITS OWN REGION ARGUMENTS ARE NOT CHECKED — they are not a
            // `sentient.for`'s, which is the `DT_CHECK_MSG(for_op, ..)` this signature encodes.
            Op::UniformRegions(regions) => {
                for region in regions.regions() {
                    if let Some(found) = parent_for_arg(val, &region.body) {
                        return Some(found);
                    }
                }
            }
            // ⛔ NO `_` ARM: the remaining dialects bind region arguments of their own — a
            // `uniform.uniformize_regions` region has one — but none of those regions can hold a
            // `sentient.for`, because the shared dialects carry the rung BELOW this one in their
            // bodies (see [`defining_op`]'s own note on the same limitation).
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// `sentient::IfOp`'s ONE CANONICALIZATION PATTERN
//
// ⛔⛔ AN ISLAND EXTENSION, AND NOT AN OPTIONAL ONE. `IfOp::getCanonicalizationPatterns` adds
// `RemoveStaticCondition` and nothing else (`dcc/src/Dialect/Sentient/SentientOps.cpp:1509-1512`),
// and campaign unit `e096_simplifyConditionals` IS a run of that pattern
// (`MultiDimLoopPeeling.cpp:529-541`). The pattern lives in `Dialect/Sentient/`, which is outside the
// campaign's `Transform/Sentient/` file scope but is NOT one of the out-of-scope `Analyses/`
// directories — so `todo!` would not be a faithful stand-in, it would make e096 and both of its
// callers (`copyOneIter`, `performLoopPeeling`) panic on every peeled loop. Region-level rewrites are
// this module's job, beside [`replace_all_uses_with`] and [`erase_defining_op`].
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT `RemoveStaticCondition` KNOWS ABOUT ONE OPERAND OF A `sentient.if`
/// (`dcc/src/Dialect/Sentient/SentientOps.cpp:1370-1427`).
///
/// ⛔ THE THREE SHAPES ARE THE PATTERN'S OWN `dyn_cast` CHAINS, not a classification of values in
/// general: a `sentient.scalar_constant`, a `sentient.for`'s induction variable, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticOperand {
    /// `dyn_cast<sentient::ConstantOp>(v.getDefiningOp())` succeeded, carrying this `$value`.
    Constant(i64),
    /// A `sentient.for`'s `$iv`, with the loop's `$bound` — `None` where that bound is not itself a
    /// `sentient.scalar_constant`, which is the reference's innermost `dyn_cast` failing (`:1382`).
    InductionVar(Option<i64>),
    /// Neither shape: `findBranchesToDelete` returns false and `both_consts` stays unset.
    Dynamic,
}

/// READ ONE `sentient.if` OPERAND AS A [`StaticOperand`].
///
/// ⛔ A REGION ARGUMENT HAS NO DEFINING OP, which is exactly how the pattern tells a loop iterator
/// from a constant — `isa<BlockArgument>` guards every `getDefiningOp()` it calls. See
/// [`defining_op`].
#[must_use]
pub fn static_operand(val: Val, scope: &[Op]) -> StaticOperand {
    if let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) = defining_op(val, scope)
    {
        return StaticOperand::Constant(*value);
    }
    if let Some(bound) = loop_bound_of_iv(val, scope) {
        return StaticOperand::InductionVar(match defining_op(bound, scope) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
            _ => None,
        });
    }
    StaticOperand::Dynamic
}

/// The `$bound` of the `sentient.for` whose induction variable is `iv` —
/// `cast<BlockArgument>(rhs).getOwner()->getParentOp()` narrowed by
/// `for_op.getInductionVar() != rhs` (`SentientOps.cpp:1377-1381`).
fn loop_bound_of_iv(iv: Val, scope: &[Op]) -> Option<Val> {
    for op in scope {
        if let Op::Sentient(sentient::Op::For {
            iv: loop_iv, bound, ..
        }) = op
            && *loop_iv == iv
        {
            return Some(*bound);
        }
        for region in regions_ref(op) {
            if let Some(bound) = loop_bound_of_iv(iv, region) {
                return Some(bound);
            }
        }
    }
    None
}

/// THE ELEMENT WIDTH AN ADDRESS IS STEPPED BY — `dcc::sentient::utils::getElementSize`
/// (`Dialect/Sentient/Utils.cpp:306`).
///
/// ⭐ `None` IS THE REFERENCE'S `-1`, its `return -1;` at `:367` and at every `hasAttr` that fails:
/// "no op in this chain says". The reference's return type is `const uint32_t` and its callers assign
/// it to an `int`, which is how `-1` survives to be compared `< 1`.
///
/// ⛔ THE REFERENCE'S `element_sizes` ARMS ARE NOT ANSWERED HERE AND ANSWER `None`, WHICH IS ITS OWN
/// `hasAttr == false` BRANCH FOR EACH — the discardable ARRAY on `sentient.for` (`:341-352`, both the
/// block-argument arm and the result arm) and on `uniform.uniformize_regions` (`:353-366`). The
/// island now carries HALF of the first: [`sentient::Carried::element_size`] is this loop's
/// `element_sizes` collapsed to one slot per carried value, so both of the reference's `for` arms
/// reduce to that slot; the BOUND's entry 0 and the whole `uniformize_regions` array are still
/// absent. Whoever ports the readers owns those arms — the writer is
/// `LiveRangeReduction::addResultToYield` (`Transform/Sentient/LiveRangeReduction.cpp:1033-1070`,
/// campaign unit `e442`), and the reference pins both the spelling and the layout:
/// `element_sizes = [-1 : i32, 8 : i32, 8 : i32]` beside a `regLocales` of the same `1 + 2n` length
/// (`dcc/test/Transform/LiveRangeReduction/uniformizeRegions.mlir:174`), the op declaring it in words
/// as "First entry is register info for `bound`, followed by entries for `initArgs`, followed by
/// entries for `results`" (`SentientOps.td:58-61`).
#[must_use]
pub fn element_size(val: Val, defs: Definitions<'_>) -> Option<crate::formats::Bits> {
    // ⭐ THE BLOCK-ARGUMENT ARM IS FIRST, as the reference's `isa<BlockArgument>` is (`:307`).
    if defs.for_arg_of(val).is_some() {
        return None;
    }
    match defs.of(val)? {
        Op::Sentient(
            sentient::Op::LoadAndSend { extent, .. }
            | sentient::Op::ReceiveAndStore { extent, .. }
            | sentient::Op::LoadAndStore { extent, .. },
        ) => Some(extent.element_size),
        // ⛔ THE **SRC** WIDTH HERE, AND THE **DST** WIDTH IN `getElementSizeOfMemOp` (`:302`). The
        // asymmetry is the reference's own and its comment claims the opposite of what it does.
        Op::Sentient(sentient::Op::LoadComputeAndSend {
            src_element_size, ..
        }) => Some(*src_element_size),
        Op::Sentient(sentient::Op::LoadAndExtractScalar { element_size, .. }) => Some(*element_size),
        // ⭐ THE DISCARDABLE `element_size` ATTRIBUTE (`:333-340`) — `None` is the reference's own
        // `hasAttr == false` on each of these two arms.
        // ⛔ NO `ScalarCopy` ARM: the copy carries the attribute (unit `e154` writes it) but
        // `getElementSize` names AddOp and SubOp only — its readers reach a copy's width through
        // `dcc::utils::getAttr(copy_op.getResult(), "element_size")` instead.
        Op::Sentient(
            sentient::Op::ScalarAdd { element_size, .. }
            | sentient::Op::ScalarSub { element_size, .. },
        ) => *element_size,
        // ⭐ THE `_` ARM IS THE REFERENCE'S OWN `return -1;` (`:367`), not a fall-through: every op
        // it does not name answers "no element size", and so do the `element_sizes` arms above.
        _ => None,
    }
}

/// WHERE A SCALAR VALUE LIVES — `sentient::getValueRegLocale`
/// (`Dialect/Sentient/SentientOps.cpp:1758`).
///
/// ⭐ [`sentient::RegType::Unknown`] IS THE REFERENCE'S `locale` LOCAL, initialised to
/// `SentientRegType::unknown` at `:1759` and returned by every arm that finds no attribute — an
/// unassigned register, which `RegisterTypeAssignment` (D66) is what replaces.
///
/// ⛔ THREE ISLAND GAPS ANSWER `Unknown`, EACH THE REFERENCE'S OWN NO-ATTRIBUTE BRANCH: `regLocales`
/// entry 0, which the op declares to be `bound`'s and which the reference reads for the INDUCTION
/// VARIABLE (`locales[argNumber]`, `SentientOps.td:58-61`) — this island's `For` has a field per
/// CARRIED value and none for the bound; `regLocale` as a discardable attribute on
/// `dataflow.get_unit` (`:1793-1800`, whose own `else` returns `unknown`); and `regLocales` on
/// `uniform.uniformize_regions` (`:1832-1842`, whose own `else` returns `unknown`).
#[must_use]
pub fn value_reg_locale(val: Val, defs: Definitions<'_>) -> sentient::RegType {
    if let Some((Op::Sentient(sentient::Op::For { carried, .. }), index)) = defs.for_arg_of(val) {
        return index
            .checked_sub(1)
            .and_then(|position| carried.get(position))
            .map_or(sentient::RegType::Unknown, |value| value.reg.locale);
    }
    match defs.of(val) {
        Some(Op::Sentient(
            sentient::Op::LoadAndSend { reg, .. }
            | sentient::Op::ReceiveAndStore { reg, .. }
            | sentient::Op::LoadComputeAndSend { reg, .. }
            | sentient::Op::ScalarCopy { reg, .. }
            | sentient::Op::ReceiveAndExtractScalar { reg, .. },
        )) => reg.locale,
        Some(Op::Sentient(sentient::Op::LoadAndStore {
            results,
            src_reg,
            dst_reg,
            ..
        })) => {
            if val == results.0 {
                src_reg.locale
            } else {
                dst_reg.locale
            }
        }
        // ⛔ NO ARM OF ITS OWN IN THE REFERENCE — this is its GENERIC TAIL (`:1846-1858`) reaching
        // `regLocales[resultIndex]`, and `load_and_extract_scalar` declares that array beside
        // `results = (outs Index:$addr, Index:$data)` (`SentientOps.td:616, 622`), so the address
        // result reads entry 0 and the data result entry 1.
        Some(Op::Sentient(sentient::Op::LoadAndExtractScalar {
            addr_result,
            addr_reg,
            data_reg,
            ..
        })) => {
            if val == *addr_result {
                addr_reg.locale
            } else {
                data_reg.locale
            }
        }
        Some(Op::Sentient(sentient::Op::ScalarConstant { reg_locale, .. })) => *reg_locale,
        Some(Op::Sentient(
            sentient::Op::ScalarAdd { reg, .. } | sentient::Op::ScalarSub { reg, .. },
        )) => reg.map_or(sentient::RegType::Unknown, |reg| reg.locale),
        // ⛔ ONE LOCALE PER CARRIED VALUE WHERE THE REFERENCE HAS TWO. Its `regLocales` is
        // `1 + 2n` long and a RESULT reads `attrs[index + 1 + numRegionIterArgs]` (`:1846-1858`)
        // while the matching region argument reads `attrs[i + 1]`; this island's one
        // `Carried::reg` is that position's whole answer, so it serves both.
        Some(Op::Sentient(sentient::Op::For { carried, .. })) => carried
            .iter()
            .find(|value| value.result == val)
            .map_or(sentient::RegType::Unknown, |value| value.reg.locale),
        // ⛔ THE `sentient.mac` ARM IS A WORKAROUND THE REFERENCE LABELS AS ONE (`:1810-1820`): the
        // op declares no register arrays, so its first result is the XRF write pointer and its
        // second the read pointer by position alone.
        Some(Op::Sentient(sentient::Op::VectorMac { results, .. })) => {
            match results.iter().position(|result| *result == val) {
                Some(0) => sentient::RegType::XrfWrPtr,
                Some(1) => sentient::RegType::XrfRdPtr,
                _ => sentient::RegType::Unknown,
            }
        }
        Some(Op::Symbol(symbol::Op::CreateSymbol { .. })) => sentient::RegType::Imm,
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => value_reg_locale(*map, defs),
        // ⛔ THE FIRST VALUE'S LOCALE, WITH NO CHECK THAT THE REST AGREE. The reference
        // `DT_CHECK(curr == locale)`s over the whole map (`:1826-1829`); this crate never refuses at
        // runtime, and a mapping whose per-core constants disagree about their register file is a
        // defect in whatever built it rather than a question this reader can answer.
        Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs
            .first()
            .map_or(sentient::RegType::Unknown, |(_, value)| {
                value_reg_locale(*value, defs)
            }),
        // ⭐ THE FALL-THROUGH IS THE REFERENCE'S OWN `return locale;` (`:1861`) — including for
        // `scalar_mul`, which the reference names no arm for either and which declares `regLocale`
        // singular rather than the `regLocales` array its generic tail reads.
        _ => sentient::RegType::Unknown,
    }
}

/// WRITE THE REGISTER INDEX A VALUE LIVES IN — `sentient::setValueRegIndex`
/// (`dcc/src/Dialect/Sentient/SentientOps.cpp:1982`), the write twin of [`value_reg_locale`].
///
/// ⛔⛔ THE REFERENCE HAS TWO SPELLINGS FOR ONE FACT AND THIS ISLAND HAS ONE. Six ops get their
/// singular `regIndex` set by name (`:2004-2035`); everything else falls into a generic tail that
/// CREATES an all-`-1` `regIndices` array on demand and writes `[resultIndex]` (`:2036-2059`) — and
/// `getValueRegIndex` (`:1884`) reads that same array back, so the two spellings round-trip. A
/// [`sentient::Reg`] is that position's whole answer, so writing it serves both.
///
/// ⛔ AN OP WITH NO [`sentient::Reg`] AT THAT POSITION IS A NO-OP, not a refusal — see
/// [`erase_defining_op`]. That is where the reference attaches an array nothing subsequently reads.
///
/// ⭐ `None` IS THE `-1` THE REFERENCE WRITES BACK for an unassigned register.
pub fn set_value_reg_index(scope: &mut [Op], val: Val, index: Option<sentient::RegIndex>) {
    for op in scope {
        if let Op::Sentient(inner) = op {
            set_reg_index_on(inner, val, index);
            for region in sentient::regions_mut(inner) {
                set_value_reg_index(region, val, index);
            }
        }
    }
}

/// [`set_value_reg_index`] against one `sentient` op — the value is this op's result, or one of the
/// region arguments it binds.
fn set_reg_index_on(op: &mut sentient::Op, val: Val, index: Option<sentient::RegIndex>) {
    match op {
        sentient::Op::For { iv, carried, .. } => {
            if *iv == val {
                todo!(
                    "setValueRegIndex on a sentient.for induction variable writes regIndices[0] \
                     (Dialect/Sentient/SentientOps.cpp:1992), and this island's `For` has a `Reg` \
                     per CARRIED value and none for the bound the induction variable counts against"
                );
            }
            // ⭐ ONE ENTRY FOR THE ARGUMENT AND THE RESULT, where the reference has `[i + 1]` and
            // `[i + numRegionIterArgs + 1]` of one `1 + 2n` array — see [`sentient::Carried`].
            for value in carried.iter_mut() {
                if value.arg == val || value.result == val {
                    value.reg.index = index;
                }
            }
        }
        sentient::Op::If { yielded, .. } => {
            for value in yielded.iter_mut() {
                if value.result == val {
                    value.reg.index = index;
                }
            }
        }
        sentient::Op::LoadAndSend { result, reg, .. }
        | sentient::Op::ReceiveAndStore { result, reg, .. }
        | sentient::Op::LoadComputeAndSend { result, reg, .. }
        | sentient::Op::ScalarCopy { result, reg, .. }
        | sentient::Op::ReceiveAndExtractScalar { result, reg, .. } => {
            if *result == val {
                reg.index = index;
            }
        }
        sentient::Op::LoadAndStore {
            results,
            src_reg,
            dst_reg,
            ..
        } => {
            if results.0 == val {
                src_reg.index = index;
            }
            if results.1 == val {
                dst_reg.index = index;
            }
        }
        sentient::Op::LoadAndExtractScalar {
            addr_result,
            data_result,
            addr_reg,
            data_reg,
            ..
        } => {
            if *addr_result == val {
                addr_reg.index = index;
            }
            if *data_result == val {
                data_reg.index = index;
            }
        }
        // ⭐ THE REFERENCE CREATES THE ATTRIBUTE WHERE NONE STOOD (`:2004-2013`), so an absent
        // `reg` becomes one whose locale is still unassigned.
        sentient::Op::ScalarAdd { result, reg, .. }
        | sentient::Op::ScalarSub { result, reg, .. } => {
            if *result == val {
                match reg {
                    Some(reg) => reg.index = index,
                    None => {
                        *reg = Some(sentient::Reg {
                            locale: sentient::RegType::Unknown,
                            index,
                        });
                    }
                }
            }
        }
        // ⭐ NO REGISTER FIELD, SO NOTHING TO WRITE: this is exactly the set of ops whose
        // `regIndices` the reference materialises and no reader of this island ever asks for.
        _ => {}
    }
}

impl<'a> Definitions<'a> {
    /// THE `sentient.for` THAT BINDS A VALUE AS A REGION ARGUMENT, and at which position — see
    /// [`parent_for_arg`].
    #[must_use]
    pub fn for_arg_of(&self, val: Val) -> Option<(&'a Op, usize)> {
        self.regions
            .iter()
            .find_map(|region| parent_for_arg(val, region))
    }
}

/// THE OP WHOSE REGION BINDS A VALUE AS AN ARGUMENT — `cast<BlockArgument>(val).getOwner()
/// ->getParentOp()`.
///
/// ⛔⛔ WIDER THAN [`parent_for_arg`] AND THAT IS THE POINT. `addToWorkListAndUpdateAssignment` blames
/// an assignment on whatever op binds the value, `sentient.for` or not
/// (`Transform/Sentient/EnhancedDeadVariableElimination.cpp:87-91`), so a caller that can only name
/// loops would attribute a `uniform.uniformize_regions` region argument to nothing.
///
/// ⛔ `None` FOR AN OP RESULT — the failed `dyn_cast<BlockArgument>`; see [`defining_op`]. And `None`
/// for a `dataflow.program_unit`'s own argument, which is no op of this island at all
/// ([`Definitions::program_unit_units_of`] answers for that one).
#[must_use]
pub fn parent_op_of_block_arg(val: Val, scope: &[Op]) -> Option<&Op> {
    for op in scope {
        match op {
            Op::Sentient(inner) => {
                if sentient::block_args(inner).contains(&val) {
                    return Some(op);
                }
                for region in sentient::regions(inner) {
                    if let Some(found) = parent_op_of_block_arg(val, region) {
                        return Some(found);
                    }
                }
            }
            Op::AffineFor(loop_op) => {
                if loop_op.iv == val || loop_op.carried.iter().any(|carried| carried.arg == val) {
                    return Some(op);
                }
                if let Some(found) = parent_op_of_block_arg(val, &loop_op.body) {
                    return Some(found);
                }
            }
            Op::UniformRegions(regions) => {
                if regions.regions().iter().any(|region| region.arg == val) {
                    return Some(op);
                }
                for region in regions.regions() {
                    if let Some(found) = parent_op_of_block_arg(val, &region.body) {
                        return Some(found);
                    }
                }
            }
            // ⛔ NO `_` ARM, and the same limitation [`parent_uniform_region_units`] records: a shared
            // dialect's regions hold ops of the rung BELOW, so an argument bound in one is a value of
            // the other island's type and cannot be answered for from here.
            Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_)
            | Op::Uniform(_) => {}
        }
    }
    None
}

impl<'a> Definitions<'a> {
    /// THE OP THAT BINDS A VALUE AS A REGION ARGUMENT — see [`parent_op_of_block_arg`].
    #[must_use]
    pub fn block_arg_owner_of(&self, val: Val) -> Option<&'a Op> {
        self.regions
            .iter()
            .find_map(|region| parent_op_of_block_arg(val, region))
    }
}

/// THE REGIONS OF ONE OP OF THIS RUNG, BORROWED — the reading half of the pair
/// [`regions_mut`] mutates through.
///
/// ⛔⛔ NOT [`regions`], AND THE DIFFERENCE IS LOAD-BEARING FOR A DECIDE-THEN-MUTATE WALK. `regions`
/// answers OWNED and therefore total, descending into a shared dialect's region by cloning it through
/// [`raised`]; nothing can be written back through that clone, so a walk that records a
/// `(op index, region index)` path and later reopens it with [`regions_mut`] must enumerate regions
/// the same way BOTH times — which is this function.
///
/// ⛔ EVERY REMAINING VARIANT'S REGIONS HOLD `dataflow_ir::dialects::Op`, PROVED BY THE TYPE. An
/// `agen.composite_load_and_store` body is a `Vec` of the rung BELOW's ops — see
/// [`agen::CompositeStoreSource`] — so it cannot contain a `sentient.*` op at all and there is
/// nothing of this rung's type to hand back. The rung below answers for its own regions.
#[must_use]
pub fn regions_ref(op: &Op) -> Vec<&[Op]> {
    match op {
        Op::Sentient(inner) => sentient::regions(inner),
        Op::AffineFor(loop_op) => vec![loop_op.body.as_slice()],
        Op::UniformRegions(regions) => regions
            .regions()
            .iter()
            .map(|region| region.body.as_slice())
            .collect(),
        // ⛔ NO `_` ARM — a twelfth dialect reaching this rung must be a build error here, not a
        // region tree every walk quietly stops at.
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => Vec::new(),
    }
}

/// THE REGIONS OF ONE OP OF THIS RUNG, MUTABLY — see [`regions_ref`].
pub fn regions_mut(op: &mut Op) -> Vec<&mut Vec<Op>> {
    match op {
        Op::Sentient(inner) => sentient::regions_mut(inner),
        Op::AffineFor(loop_op) => vec![&mut loop_op.body],
        // ⭐ WHAT A SINK WRITES INTO — see [`UniformRegions`].
        Op::UniformRegions(regions) => regions
            .regions_mut()
            .iter_mut()
            .map(|region| &mut region.body)
            .collect(),
        // ⛔ NO `_` ARM — see [`regions_ref`].
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => Vec::new(),
    }
}

/// CLONES `ops` AT **THIS** RUNG, MINTING A FRESH VALUE FOR EVERYTHING THEY DEFINE —
/// `OpBuilder::clone(Operation &, IRMapping &)`.
///
/// # ⛔⛔ THE ISLAND HAD NO CLONE AT THIS RUNG AND `copyOneIter` IS ONE
///
/// `builder.clone(*for_op)` copies a whole `sentient.for`, hoists the copy's body out and deletes the
/// copy (`dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:465-467`), which is how a peeled
/// iteration is built. [`crate::islands::dataflow_ir::Values::clone_ops`] is typed to the rung BELOW —
/// it goes through that island's [`crate::islands::dataflow_ir::dialects::parts_mut`], which has no
/// `Sentient` arm — so without this the peel would emit a second definition of every value the loop
/// body binds, and nothing would say so.
///
/// ⭐ THE ORDER IS THE RUNG BELOW'S, BECAUSE IT IS MLIR'S PRINTING ORDER: operands read through
/// `mapping`, then this op's results minted, then its regions' arguments, then the regions.
///
/// ⛔ FOUR SEPARATELY SCOPED PHASES, NOT ONE `parts_mut`: each of [`sentient::operands_mut`],
/// [`sentient::results_mut`], [`sentient::block_args_mut`] and [`sentient::regions_mut`] borrows the
/// whole op, so only one of the four lists can be live at a time and the groups cannot be collected
/// together the way the rung below collects them.
///
/// ⭐ THE SHARED DIALECTS DELEGATE through [`lowered`]/[`raised`] onto that same `parts_mut` — one
/// description of one operation, as everywhere else in this module.
#[must_use]
pub fn clone_ops(ops: &[Op], values: &mut Values, mapping: &mut ValueMapping) -> Vec<Op> {
    ops.iter().map(|op| clone_op(op, values, mapping)).collect()
}

/// One op of [`clone_ops`].
fn clone_op(op: &Op, values: &mut Values, mapping: &mut ValueMapping) -> Op {
    if let Some(lower) = lowered(op) {
        let mut cloned = values.clone_ops(&[lower], mapping);
        return raised(cloned.remove(0));
    }
    let mut copy = op.clone();
    match &mut copy {
        Op::Sentient(inner) => {
            for operand in sentient::operands_mut(inner) {
                *operand = mapping.lookup_or_default(*operand);
            }
            for result in sentient::results_mut(inner) {
                *result = minted(*result, values, mapping);
            }
            for arg in sentient::block_args_mut(inner) {
                *arg = minted(*arg, values, mapping);
            }
            for region in sentient::regions_mut(inner) {
                // ⭐ TAKEN, NOT COPIED AGAIN — `op.clone()` above already deep-copied the body.
                let source = core::mem::take(region);
                *region = clone_ops(&source, values, mapping);
            }
        }
        // ⭐ ARM FOR ARM WITH THE RUNG BELOW'S `affine::Op::For`: a constant bound is not a value, and
        // the induction variable is region argument 0.
        Op::AffineFor(loop_op) => {
            for bound in [&mut loop_op.lo, &mut loop_op.hi] {
                if let affine::Bound::Val(val) = bound {
                    *val = mapping.lookup_or_default(*val);
                }
            }
            for carried in &mut loop_op.carried {
                carried.init = mapping.lookup_or_default(carried.init);
            }
            for carried in &mut loop_op.carried {
                carried.result = minted(carried.result, values, mapping);
            }
            loop_op.iv = minted(loop_op.iv, values, mapping);
            for carried in &mut loop_op.carried {
                carried.arg = minted(carried.arg, values, mapping);
            }
            let source = core::mem::take(&mut loop_op.body);
            loop_op.body = clone_ops(&source, values, mapping);
        }
        // ⭐ ARM FOR ARM WITH THE RUNG BELOW'S `uniform::Op::UniformizeRegions`: each region reads its
        // unit list and binds its own argument, and only `uniformize_regions` binds results.
        Op::UniformRegions(regions) => {
            for region in regions.regions_mut() {
                for unit in &mut region.units {
                    *unit = mapping.lookup_or_default(*unit);
                }
            }
            if let UniformRegions::UniformizeRegions { results, .. } = regions {
                for result in results {
                    *result = minted(*result, values, mapping);
                }
            }
            for region in regions.regions_mut() {
                region.arg = minted(region.arg, values, mapping);
            }
            for region in regions.regions_mut() {
                let source = core::mem::take(&mut region.body);
                region.body = clone_ops(&source, values, mapping);
            }
        }
        // ⛔ NO `_` ARM, and unreachable: [`lowered`] answered every one of these above.
        Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Arith(_)
        | Op::Scf(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => {}
    }
    copy
}

/// One value the clone DEFINES: a fresh name, recorded so every later read of the original finds it.
fn minted(of: Val, values: &mut Values, mapping: &mut ValueMapping) -> Val {
    let fresh = values.mint();
    mapping.map(of, fresh);
    fresh
}

/// WHICH BRANCH OF A `sentient.if` SURVIVES CANONICALISATION — `RemoveStaticCondition`'s three
/// outcomes (`SentientOps.cpp:1489-1502`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticBranch {
    /// `replaceOpWithRegion(rewriter, op, op.getThenRegion())`.
    Then,
    /// `replaceOpWithRegion(rewriter, op, op.getElseRegion())`.
    Else,
    /// `rewriter.eraseOp(op)` — reached only where the else region is EMPTY, which the entry gate has
    /// already paired with "no results", so there is nothing left to rewire.
    Neither,
}

/// `findBranchesToDelete`'s two out-parameters (`SentientOps.cpp:1370-1374`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct DeadBranches {
    then_branch: bool,
    else_branch: bool,
}

/// `findBranchesToDelete` — `lhs` a constant compared against `rhs`, a loop iterator running over
/// `[1, bound]` (`SentientOps.cpp:1370-1427`).
///
/// ⛔ IT RESETS BOTH FLAGS ON ENTRY (`:1374`), which is what makes the caller's second, reversed
/// attempt safe to run over the first's leftovers.
fn branches_to_delete(
    predicate: sentient::CmpPredicate,
    lhs: StaticOperand,
    rhs: StaticOperand,
    dead: &mut DeadBranches,
) -> bool {
    *dead = DeadBranches::default();
    let (StaticOperand::Constant(l), StaticOperand::InductionVar(Some(bound))) = (lhs, rhs) else {
        return false;
    };
    match predicate {
        sentient::CmpPredicate::Eq => dead.then_branch = l > bound || l < 1,
        sentient::CmpPredicate::Ne => dead.else_branch = l > bound || l < 1,
        sentient::CmpPredicate::Sge => {
            dead.else_branch = l >= bound;
            dead.then_branch = l < 1;
        }
        sentient::CmpPredicate::Sgt => {
            dead.else_branch = l > bound;
            dead.then_branch = l <= 1;
        }
        sentient::CmpPredicate::Sle => {
            dead.else_branch = l <= 1;
            dead.then_branch = l > bound;
        }
        sentient::CmpPredicate::Slt => {
            dead.else_branch = l < 1;
            dead.then_branch = l >= bound;
        }
    }
    dead.then_branch || dead.else_branch
}

/// `RemoveStaticCondition::matchAndRewrite`'s DECISION, with no rewriting
/// (`SentientOps.cpp:1430-1506`).
///
/// ⛔⛔ `None` IS `failure()` — *"this `sentient.if` is not static"*, never an error. The pattern is
/// a fold: a condition it cannot decide leaves the op exactly as it was.
///
/// ⛔ THE DECISION IS SPLIT FROM THE REWRITE because it reads the whole region tree — the constants
/// and the enclosing loops' bounds — while the rewrite mutates one block of it. An MLIR `Value`
/// carries its own owner and needs no such split.
///
/// ⭐ THE `else return failure()` ON AN UNRECOGNISED PREDICATE (`:1464`) IS UNWRITABLE HERE:
/// [`sentient::CmpPredicate`] carries exactly the six the arms above it cover.
#[must_use]
pub fn static_if_branch(
    predicate: sentient::CmpPredicate,
    lhs: StaticOperand,
    rhs: StaticOperand,
    has_else: bool,
    has_results: bool,
) -> Option<StaticBranch> {
    // `if (op->getNumResults() > 0 && op.getElseRegion().empty()) return failure();` (`:1436`).
    if has_results && !has_else {
        return None;
    }
    let mut both_consts = false;
    let mut use_then_branch = false;
    if let (StaticOperand::Constant(l), StaticOperand::Constant(r)) = (lhs, rhs) {
        use_then_branch = match predicate {
            sentient::CmpPredicate::Eq => l == r,
            sentient::CmpPredicate::Ne => l != r,
            sentient::CmpPredicate::Sle => l <= r,
            sentient::CmpPredicate::Sge => l >= r,
            sentient::CmpPredicate::Slt => l < r,
            sentient::CmpPredicate::Sgt => l > r,
        };
        both_consts = true;
    }
    let mut dead = DeadBranches::default();
    let mut cond_out_of_loop_bounds = false;
    if !both_consts {
        cond_out_of_loop_bounds = branches_to_delete(predicate, lhs, rhs, &mut dead);
        // ⭐ THE SECOND ATTEMPT SWAPS THE OPERANDS **AND** REVERSES THE PREDICATE, so it is the same
        // condition read the other way round (`:1483-1485`).
        if !cond_out_of_loop_bounds {
            cond_out_of_loop_bounds = branches_to_delete(predicate.reversed(), rhs, lhs, &mut dead);
        }
    }
    if !both_consts && !cond_out_of_loop_bounds {
        return None;
    }
    if use_then_branch || dead.else_branch {
        return Some(StaticBranch::Then);
    }
    if has_else && (!use_then_branch || dead.then_branch) {
        return Some(StaticBranch::Else);
    }
    Some(StaticBranch::Neither)
}

/// SPLICE ONE BRANCH OF THE `sentient.if` AT `at` INTO ITS OWN BLOCK — `replaceOpWithRegion`
/// (`SentientOps.cpp:1354-1363`), and the `eraseOp` arm beside it for [`StaticBranch::Neither`].
///
/// ⛔⛔ IT RETURNS THE REWIRES RATHER THAN PERFORMING THEM. `rewriter.replaceOp(op, results)` moves
/// every reader of the `sentient.if`'s results onto the surviving region's `sentient.yield` operands,
/// and those readers are in the ENCLOSING scope — of which this holds `&mut` to one block. The caller
/// feeds each pair to [`replace_all_uses_with`] over the whole scope.
///
/// ⛔ THE TERMINATOR IS DROPPED, NOT INLINED — `rewriter.eraseOp(terminator)` (`:1362`).
pub fn replace_if_with_region(
    block: &mut Vec<Op>,
    at: usize,
    branch: StaticBranch,
) -> Vec<(Val, Val)> {
    let mut produced: Vec<Val> = Vec::new();
    let mut region: Vec<Op> = Vec::new();
    if let Some(Op::Sentient(sentient::Op::If {
        yielded,
        then_body,
        else_body,
        ..
    })) = block.get_mut(at)
    {
        produced = yielded.iter().map(|y| y.result).collect();
        region = match branch {
            StaticBranch::Then => core::mem::take(then_body),
            StaticBranch::Else => core::mem::take(else_body),
            StaticBranch::Neither => Vec::new(),
        };
    }
    let mut rewires: Vec<(Val, Val)> = Vec::new();
    if let Some(Op::Sentient(sentient::Op::Yield { results })) = region.last() {
        rewires = produced.into_iter().zip(results.iter().copied()).collect();
        region.pop();
    }
    drop(block.splice(at..=at, region));
    rewires
}

/// WHICH UNIT HANDLES A `uniform.query_map`'S KEY STANDS FOR — `collectUnitOps` over the key's own
/// region unit list (`dcc/src/Dialect/Uniform/Utils.cpp:98-115, 151-192`).
///
/// ⛔⛔ A `dataflow.create_group` KEY EXPANDS TO ITS MEMBERS. The reference pushes the group's whole
/// `getUnitIds()` rather than the group handle (`Utils.cpp:105-108`), so one entry of a region's unit
/// list can contribute several keys — and the mapping is keyed by the individual `get_unit`s.
///
/// ⛔ A KEY THAT IS NEITHER STOPS THE WALK. `op->emitError("Key has to be GetUnitOp or
/// CreateGroupOp."); break;` (`:110-113`) — the reference returns the PARTIAL list it has built and
/// carries on, which is what the `break` here reproduces.
///
/// ⭐ `collectUnitVals` (`Utils.cpp:117-133`) IS A BYTE-IDENTICAL SECOND COPY of `collectUnitOps`
/// (`:98-115`) under another name, so both spellings land here — `AddressPinningAndToggle`'s
/// `turnHBMConstantOpAddrsToQueryMapsHelper` calls it by the second (`:1608`, `:1613`).
#[must_use]
pub fn collect_unit_ops(units: &[Val], defs: Definitions<'_>) -> Vec<Val> {
    let mut keys = Vec::new();
    for value in units {
        match defs.of(*value) {
            Some(Op::Dataflow(dataflow::Op::GetUnit { .. })) => keys.push(*value),
            Some(Op::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) => {
                keys.extend(unit_ids.iter().copied());
            }
            _ => break,
        }
    }
    keys
}

/// THE UNITS OF THE REGION THAT BINDS A VALUE AS ITS ARGUMENT — the
/// `getRegionUnitList(block_arg)` both `uniform.uniformize_regions` and `uniform.equalize_pattern`
/// declare (`Uniform.td:98`, `:206`).
///
/// ⭐⭐ THE UNIT LIST AND NOT THE REGION, BECAUSE THE REGION HAS TWO TYPES AT THIS RUNG AND ONE
/// ANSWER. A local region whose body has reached this rung is a [`LocalRegion`] and one that has not
/// is a [`uniform::LocalRegion`] (see [`UniformRegions`]) — so a signature returning either could
/// only answer for half the island, while `getRegionUnitList`'s own answer, `ValueRange`, is the same
/// list in both. Every caller reads exactly that: [`uniform_mapping_values`] looks the units up in a
/// `uniform.def_immutable_mapping`.
#[must_use]
pub fn parent_uniform_region_units(val: Val, scope: &[Op]) -> Option<&[Val]> {
    for op in scope {
        match op {
            Op::Uniform(
                uniform::Op::UniformizeRegions { regions, .. }
                | uniform::Op::EqualizePattern { regions },
            ) => {
                if let Some(region) = regions.iter().find(|region| region.arg == val) {
                    return Some(&region.units);
                }
            }
            // ⭐ THE SAME QUESTION OF THE SAME OP WITH ITS REGIONS REWRITTEN — and its bodies ARE
            // descended into, unlike the arm above, because they hold ops of this island's type.
            Op::UniformRegions(regions) => {
                if let Some(region) = regions.regions().iter().find(|region| region.arg == val) {
                    return Some(&region.units);
                }
                for region in regions.regions() {
                    if let Some(found) = parent_uniform_region_units(val, &region.body) {
                        return Some(found);
                    }
                }
            }
            Op::Sentient(inner) => {
                for region in sentient::regions(inner) {
                    if let Some(found) = parent_uniform_region_units(val, region) {
                        return Some(found);
                    }
                }
            }
            Op::AffineFor(loop_op) => {
                if let Some(found) = parent_uniform_region_units(val, &loop_op.body) {
                    return Some(found);
                }
            }
            // ⛔ NO `_` ARM. A shared dialect's regions hold ops of the rung BELOW this one, so a
            // `uniform` op nested inside one is a value of the other island's type and its regions
            // are not walked from here. [`defining_op`] records the same limitation.
            Op::Uniform(
                uniform::Op::Yield { .. }
                | uniform::Op::DefImmutableMapping { .. }
                | uniform::Op::QueryMap { .. },
            )
            | Op::Dataflow(_)
            | Op::Agen(_)
            | Op::VectorChain(_)
            | Op::Affine(_)
            | Op::Vector(_)
            | Op::Arith(_)
            | Op::Scf(_)
            | Op::Symbol(_) => {}
        }
    }
    None
}

impl<'a> Definitions<'a> {
    /// THE UNITS OF THE UNIFORM REGION THAT BINDS A VALUE AS ITS ARGUMENT — see
    /// [`parent_uniform_region_units`].
    #[must_use]
    pub fn uniform_region_units_of(&self, val: Val) -> Option<&'a [Val]> {
        self.regions
            .iter()
            .find_map(|region| parent_uniform_region_units(val, region))
    }

    /// THE UNITS OF THE `dataflow.program_unit` THAT BINDS A VALUE AS ITS REGION ARGUMENT — `None`
    /// for any other value, and for a walk that did not say which unit it is inside
    /// ([`Self::with_program_unit`]).
    #[must_use]
    pub fn program_unit_units_of(&self, val: Val) -> Option<&'a [Val]> {
        self.program_unit
            .and_then(|(arg, units)| (arg == val).then_some(units))
    }
}

/// EVERY `dataflow.create_group` IN A UNIT LIST REPLACED BY ITS MEMBERS —
/// `dcc::uniform::utils::expandAllGroupsToUnits` (`dcc/src/Dialect/Uniform/Utils.cpp:1149-1161`).
///
/// ⛔ NOT [`collect_unit_ops`], WHICH IS A DIFFERENT FUNCTION OF THE SAME FILE. `collectUnitOps`
/// STOPS at a value that is neither a `get_unit` nor a group and returns the partial list
/// (`Utils.cpp:110-113`); this one passes such a value through unchanged, so its result is always as
/// long as its input or longer.
#[must_use]
pub fn expand_all_groups_to_units(units: &[Val], defs: Definitions<'_>) -> Vec<Val> {
    units
        .iter()
        .flat_map(|unit| match defs.of(*unit) {
            Some(Op::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) => unit_ids.clone(),
            _ => vec![*unit],
        })
        .collect()
}

/// THE UNITS ONE `uniform.query_map`'S KEY STANDS FOR — `dcc::uniform::utils::
/// getListOfKeyOpsFromUniformMapping` (`dcc/src/Dialect/Uniform/Utils.cpp:151-192`).
///
/// ⭐ SPLIT OUT OF [`uniform_mapping_values`], WHICH IS THE REFERENCE'S OWN SHAPE: `getListOf
/// ValueOpsFromUniformMapping` opens by calling this (`:194-197`), and entry 203 compares the two
/// lists SEPARATELY — key lists first, then value lists — so the keys have to be reachable alone.
///
/// See [`uniform_mapping_values`] for the two traps this walk carries: a `dataflow.create_group` key
/// expanding to its members, and the `dataflow.program_unit` region-argument arm being an island gap.
#[must_use]
pub fn uniform_mapping_keys(key: Val, defs: Definitions<'_>) -> Vec<Val> {
    match defs.of(key) {
        // `else if (auto unit = dyn_cast<GetUnitOp>(key.getDefiningOp())) key_vals.push_back(key);`
        // — ⭐ THE KEY ITSELF, not the unit list of anything (`:187-190`).
        Some(Op::Dataflow(dataflow::Op::GetUnit { .. })) => vec![key],
        Some(_) => Vec::new(),
        // No defining op: a block argument, so the parent op's unit list for that region — a local
        // region's (`:172-179`), or the enclosing `dataflow.program_unit`'s own (`:180-186`).
        None => match defs.uniform_region_units_of(key) {
            Some(units) => collect_unit_ops(units, defs),
            None => defs
                .program_unit_units_of(key)
                .map_or_else(Vec::new, |units| collect_unit_ops(units, defs)),
        },
    }
}

/// THE VALUES ONE `uniform.query_map` CAN ANSWER WITH — `dcc::uniform::utils::
/// getListOfValueOpsFromUniformMapping` (`dcc/src/Dialect/Uniform/Utils.cpp:194-202`).
///
/// The query map's key names one unit or a whole region's worth of them; those units are looked up in
/// the `uniform.def_immutable_mapping` behind `map`, and the values found are the answers.
///
/// ⛔ MISSES ARE DROPPED, NOT REPORTED. `getNonNullValuesFromKeys` pushes only the keys the map holds
/// (`dataflow-scheduler/.../lib/Dialect/Uniform/Uniform.cpp:548-556`), so the result is SHORTER than
/// the key list when a unit has no entry — which is why its sibling `getValuesFromKeys`, which keeps
/// the holes as `std::nullopt`, exists separately.
///
/// ⭐ THE THIRD ARM OF THE KEY WALK — a key bound as the region argument of a
/// `dataflow.program_unit`, whose keys are `prog_unit_op.getUnits()` (`Utils.cpp:180-186`) — WAS AN
/// ISLAND GAP AND IS NOW DISCHARGED: [`crate::islands::sentient::ProgramUnit::iter_arg`] carries that
/// argument and [`Definitions::with_program_unit`] hands it to this walk. A walk built with plain
/// [`Definitions::from_innermost`] still answers the empty list for it, which is the reference's own
/// `key_vals` when none of the three parent forms matches (`:154-190`).
#[must_use]
pub fn uniform_mapping_values(map: Val, key: Val, defs: Definitions<'_>) -> Vec<Val> {
    let keys = uniform_mapping_keys(key, defs);
    // `auto target_map = dyn_cast<DefImmutableMappingOp>(query_map_op.getMap().getDefiningOp());`
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(map) else {
        return Vec::new();
    };
    keys.iter()
        .filter_map(|sought| {
            pairs
                .iter()
                .find(|(mapped, _)| mapped == sought)
                .map(|(_, value)| *value)
        })
        .collect()
}

#[cfg(test)]
mod unit_tests {
    use super::{Op, Val, clone_ops, sentient};
    use crate::islands::dataflow_ir::dialects::arith;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{ValueMapping, Values};

    /// `builder.clone(*for_op, mapping)` — a `sentient.for` and everything under it get fresh names,
    /// and every read inside follows the mapping.
    ///
    /// ⛔ THE SHARED-DIALECT ARM IS THE POINT: the `arith.constant` in the body goes through
    /// [`super::lowered`]/[`super::raised`], which is the rung below's clone, so its result must be
    /// minted too — otherwise the copy would define `%2` a second time.
    #[test]
    fn clone_ops_mints_every_value_the_copy_defines() {
        let ops = vec![Op::Sentient(sentient::Op::For {
            iv: Val(1),
            bound: Val(0),
            carried: Vec::new(),
            dbg_name: None,
            body: vec![
                Op::Arith(arith::Op::Constant {
                    result: Val(2),
                    value: 3,
                }),
                Op::Sentient(sentient::Op::ScalarAdd {
                    lhs: Val(1),
                    rhs: Val(2),
                    result: Val(3),
                    reg: None,
                    ty: ScalarTy::Index,
                    element_size: None,
                }),
            ],
        })];
        let mut values = Values::default();
        for _ in 0..4 {
            let _ = values.mint();
        }
        let mut mapping = ValueMapping::new();
        let cloned = clone_ops(&ops, &mut values, &mut mapping);

        let Op::Sentient(sentient::Op::For {
            iv, bound, body, ..
        }) = &cloned[0]
        else {
            panic!("the clone is the same op")
        };
        // ⭐ THE BOUND IS AN OPERAND DEFINED OUTSIDE, so it is NOT remapped.
        assert_eq!(*bound, Val(0));
        assert_eq!(*iv, Val(4));
        assert_eq!(
            body[0],
            Op::Arith(arith::Op::Constant {
                result: Val(5),
                value: 3
            })
        );
        let Op::Sentient(sentient::Op::ScalarAdd {
            lhs, rhs, result, ..
        }) = body[1]
        else {
            panic!("the body's second op is the add")
        };
        assert_eq!((lhs, rhs, result), (Val(4), Val(5), Val(6)));
    }
}
