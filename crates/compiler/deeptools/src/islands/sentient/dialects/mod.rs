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
    /// Upstream `affine` — the loop nest and the applied maps.
    Affine(affine::Op),
    /// Upstream `arith` — constants and predicates.
    Arith(arith::Op),
    /// Upstream `scf`.
    Scf(scf::Op),
}
