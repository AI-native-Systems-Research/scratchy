//! THE DATAFLOWIR OPS. One variant per operation the emitted programs contain, and no more.
//!
//! The set is the union of `Dataflow.td`'s own ops and the standard-dialect ops a real program uses,
//! read off IBM's `dcc/test/PT/xrfbmm_int8_fwd.mlir` — a complete int8 BMM in about sixty lines.
//!
//! ⛔ WHAT IS NOT HERE IS THE POINT. No register, no port, no result forwarding, no unroll factor,
//! no precision per operand. Those are `sentient.*`, and dcc's 76 passes derive them
//! (`dbo/docs/pass_pipeline.md` D1-D76). An op here that named a register would be one rung down the
//! ladder from where this crate stands.
//!
//! ⭐⭐ ONE MODULE PER DIALECT, AND THE DIALECT IS THE OUTER ARM. Which `.td` declares an op is a
//! fact about the op, so the type carries it rather than a prefix on a variant's name.
//! `agen.vector_load` and `affine.vector_load` are two operations of two dialects — see
//! [`agen::Op::VectorLoad`] for which of them the scheduler's own producer emits — and as sibling
//! variants of one flat enum nothing but the spelling said so.

pub mod affine;
pub mod agen;
pub mod arith;
pub mod dataflow;
pub mod scf;
pub mod vectorchain;

/// AN SSA VALUE, minted by the builder and never spelled by hand.
///
/// ⛔ A NEWTYPE OVER THE NUMBER, so a value cannot be confused with an extent, an address or a loop
/// bound — all of which are also small integers in this IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Val(pub u32);

/// ONE INDEX OF A LOAD OR STORE: an induction variable, an applied map, or a literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// An SSA value — a loop's induction variable or an `affine.apply` result.
    Val(Val),
    /// A literal, as `%lrf_memory[4, 0]` writes it.
    Const(i64),
    /// ⭐⭐ A SUM OF STRIDED INDUCTION VARIABLES, WRITTEN INLINE — `%arg9 + %arg8 * 8`.
    ///
    /// ⛔⛔ INLINE, NOT AN `affine.apply`. One view dimension is walked by EVERY enclosing loop that
    /// strides its axis, so an index is a sum, not a single variable. IBM writes those sums straight
    /// into the index list — `agen.vector_store %48, %49[0, %arg9 + %arg8 * 8, 0]`
    /// (`dcc/test/PT/bf16-pt.mlir:161`) — and `bf16-pt.mlir` contains **zero** `affine.apply`.
    /// Emitting one per index instead made `dbo-opt` refuse outright: "'affine.apply' op Expanded
    /// affine.apply operation does not resolve to a constant".
    ///
    /// ⭐ A STRIDE OF ONE PRINTS BARE. `%arg9 * 1` is the same address written longer, and the
    /// vendored files never write it.
    ///
    /// ⭐⭐ AND A CONSTANT ADDEND, WHICH IS HOW A FAN-OUT'S SLICES DIFFER. One `ddl.data_transfer`
    /// to a row-expanding `unit="pt"` becomes one send PER ROW, and the eight PT rows are a systolic
    /// accumulation chain (`bmm.ddl:256-260`: row 0 seeds with `%zero_const`, rows 1-7 add
    /// `%pt_src02_north`) each MACing from its OWN XRF — so they need eight DIFFERENT slices, not
    /// one broadcast. The i-th slice sits `i * stride/count` along the axis the innermost enclosing
    /// loop strides.
    ///
    /// ⛔ INLINE, NOT AN `affine.apply` — same reason as the terms above.
    Strided(Vec<(Val, i64)>, i64),
}

/// ONE DATAFLOWIR OPERATION, under the dialect that declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Upstream `arith` — the constants and the integer predicates.
    Arith(arith::Op),
    /// Upstream `scf` — the undecided branch.
    Scf(scf::Op),
    /// Upstream `affine` — the loop nest, the applied maps, and `dcc-opt`'s vector accesses.
    Affine(affine::Op),
    /// `Dataflow.td` — units, views, transfers between units, and the opaque bodies.
    Dataflow(dataflow::Op),
    /// `Agen.td` — the address-generator's accesses and composite transfers.
    Agen(agen::Op),
    /// `VectorChain.td` — everything the PE and the SFP compute.
    VectorChain(vectorchain::Op),
}
