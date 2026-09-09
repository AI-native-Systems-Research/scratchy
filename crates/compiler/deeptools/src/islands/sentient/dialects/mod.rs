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
    /// shared dialect's — the op is the same op whichever rung holds it.
    Uniform(uniform::Op),
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
        Op::Sentient(_) | Op::AffineFor(_) => None,
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
        other => lowered(other).map_or_else(Vec::new, |op| {
            crate::islands::dataflow_ir::dialects::results(&op)
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
            // ⛔ A DEFINITION INSIDE A LOWER-RUNG REGION IS PROVED ABSENT, NOT ASSUMED ABSENT. The
            // DataflowIR island's own walk descends into the region and answers for the whole
            // subtree; only if it finds the definition is there nothing this signature can return,
            // because the op it found is a value of the other island's type.
            other => {
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
            other => {
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
            // ⛔ PROVED ABSENT, as in [`defining_op`].
            other => {
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
}

impl<'a> Definitions<'a> {
    /// THE ENCLOSING REGIONS, INNERMOST FIRST.
    #[must_use]
    pub const fn from_innermost(regions: &'a [&'a [Op]]) -> Definitions<'a> {
        Definitions { regions }
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
