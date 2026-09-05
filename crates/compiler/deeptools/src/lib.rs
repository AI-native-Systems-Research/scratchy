//! THE DEEPTOOLS IRs, AND THE BRIDGES BETWEEN THEM.
//!
//! ```text
//! SubtileIR tape ──► DataflowIR ──► SentientIR ──► ProgIR ──► SenProg ──► init_binary
//!    (scratchy)      island 1       island 2       island 3    (ported)
//!               ^bridge 1       ^bridge 2      ^bridge 3   ^bridge 4
//! ```
//!
//! # ⭐⭐ AN ISLAND IS AN IR AND NOTHING ELSE
//!
//! [`islands`]`::<ir>` holds one IR's types, its invariants and its printer. It knows nothing
//! about who produces it or who consumes it — that is what keeps it checkable against IBM's own
//! reference files on its own terms, rather than against whatever we happened to emit.
//!
//! # ⭐⭐ A BRIDGE IS WHERE TWO VOCABULARIES MEET
//!
//! [`bridges`]`::<from>_to_<to>` is exactly where a join gets fudged, so each gets its own
//! directory and its own tests rather than being spread through the island it feeds.
//!
//! # ⛔⛔ NO SCRATCHY DEPENDENCY, EVER
//!
//! Anything that speaks scratchy's `TensorRegion` or `SubtileNode` lives on the scratchy side of
//! bridge 1. An edge back into scratchy is what lets a lowering be *recovered* from its own output
//! instead of read from its input.
//!
//! # ⛔ EVERY EXTENT IS A NEWTYPE AND EVERY MACHINE FACT IS A CONSTANT
//!
//! [`arch::Arch`] and [`model::Model`] carry the machine and the network as associated constants
//! resolved at compile time, so a value like `Target::PT_ROWS` is a literal the compiler folds
//! rather than a field something reads at runtime. A bare `u32` crossing a boundary here is a
//! defect: transposing two extents has to be an E0308, not a wrong program.

/// THE MACHINE, AS CONSTANTS — one type per ISA generation, selected by feature.
pub mod arch;

/// WHERE TWO VOCABULARIES MEET — one module per bridge.
pub mod bridges;

/// AN IR AND NOTHING ELSE — one module per IR.
pub mod islands;

/// THE NETWORK, AS CONSTANTS — the shape facts a lowering is specialised on.
pub mod model;

/// Per-template facts computed at BUILD time by `build.rs` — the one place `.ddl` text exists.
///
/// The tables are emitted uniformly from the template census, so their shape follows the data
/// rather than anyone's hand.
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}

/// THE UNITS A PROGRAM DECLARES, and what each one is next to.
pub mod units;

/// THE WALKED DATAFLOW FOR ONE OP-FUNC ON THE ARCH BEING BUILT FOR.
///
/// ⭐ TOTAL, AND THAT IS THE WHOLE POINT. `OpFunc` is the sealed set scratchy can emit and every
/// one of them resolves to exactly one program on every generation this crate builds for —
/// resolved at build time by `opFuncToDdlTemplate`'s own ordered candidate list. So there is no
/// `Option` to unwrap, no string to compare, and no search.
///
/// ⛔⛔ AND THE FORMAT IS PART OF THE QUESTION, NOT DECORATION. A template serves an op-func AT A
/// PRECISION: `unary_parallel.ddl:30` binds `exp` for fp32 and `unary_pipeline.ddl:19` binds it
/// for fp16, each with its own `.smc`. Resolving without the format hands an fp16 `exp` the fp32
/// kernel, whose `SFP_IMMCOPY` of an fp32 epsilon the backend refuses outright.
#[must_use]
pub fn program_for<A: arch::Arch>(
    op_func: generated::OpFunc,
    format: generated::DataType,
) -> &'static generated::Program {
    op_func.program(A::GEN, format)
}
