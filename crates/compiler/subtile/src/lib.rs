// SPDX-License-Identifier: Apache-2.0
//! scratchy-subtile — the subtile-wavefront scheduling layer.
//!
//! This is a normal library crate (not the proc-macro), so it is
//! unit-testable on any host — the host-first correctness path runs
//! here. The proc-macro crate `scratchy-forward-compiler-macro` depends on this
//! crate and, at expansion time, walks its (internal) solved FUF and
//! calls the lowering here to build a [`subtile::SubtileGraph`], then
//! emits GPU tape players. This mirrors how `scratchy-forward-compiler-macro`
//! already runs its own FUF (fusion-synthesis) logic; the subtile IR and
//! `Fuf`/`Assignment` can't live in the
//! proc-macro crate because proc-macro crates export only macros.
//!
//! `cpu_golden` (from `scratchy-forward-compiler`) is the host *calculator* the
//! validation computes with — not the correctness *oracle* (that is
//! scratchy-target-metal non-mega at temp=0).

// First, so the pattern it generates (`expansion_ops!`) is in textual scope for every module.
#[macro_use]
pub mod ops;
pub mod fixtures;
pub mod lower;
/// The dataflow facts one persistent kernel needs to play a tape: waits, units, coherence.
/// SubtileIR → KTIR (IBM Spyre target). The macro calls it under `-Fspyre`.
/// ALSO needed by `-Fsuperdsc`: `emit_bundle` builds the source/dim manifest via
/// `lower_graph_to_ktir`, which the sdsc bundle embeds — so gate on either.
/// Off the SAME fused tape as the KTIR lowering; emits the TILE-LEVEL,
/// work-divided SuperDSC (SDSC) bundle DeepTools' `dxp_standalone --bundle`
/// compiles — the perf path that OWNS the 32-core work-division. (The old
/// `lower_subtile_tape_to_sengraph` peer was DELETED — sengraph is a dead end.)
/// The model's head geometry as types, and the two value→const doors whose arms the build script
/// generated from `crates/models/arch/*/configs/*.json`. Always compiled: the tape carries a
/// [`model_geometry::ModelAttnGeometry`] on every attention node, whatever the target. Except the
/// attention-geometry door, whose body takes the SuperDSC emitter's witness: it is `spyre`-only.
pub mod model_geometry;
pub mod subtile_ir;
// ⭐ THE SuperDSC DEVICE FACTS, FROM THE `ktir-superdsc` LEAF CRATE: the typed SuperDSC IR
// (`superdsc_opspec`/`superdsc_error`), `sdsc_abstract`, and the `addr` build-time address
// interpreter that PROVES the emitted SDSC attention computes the SubtileIR math
// (`softmax(q·kᵀ·scale+mask)·v`) over the real device addresses, so a translation bug is a
// `cargo build` panic rather than an on-card discovery. Re-exported so the Spyre lowering and
// worker name them as `scratchy_subtile::sdsc_abstract::…` and `model_geometry`'s attention door
// names `AttnGeometry` through `crate::sdsc_abstract`.
//
// ⛔ GATED ON `spyre`, WITH THE DEPENDENCY. Nothing a metal or cuda build compiles names these, so
// those builds leave the whole `ktir-superdsc`/`ktir-core` graph out.
#[cfg(feature = "spyre")]
pub use ktir_superdsc::{addr, sdsc_abstract, superdsc_error, superdsc_opspec};
// ⛔ UNGATED, AND THAT IS THE POINT. The re-roll machinery (`reroll_subtile_tape` /
// `lower_dag_to_tape` / the `OpenLoop`/`CloseLoop` tape) is THE shared artifact: the
// SuperDSC emitter (which re-rolls the layer loop so dxp compiles ONE small body
// bundle instead of the 30×-unrolled monster) and metal both lower from it. Gating it
// on a target-only feature made it INVISIBLE to a metal build, which is a large part
// of why metal forked upstream of it instead. Its non-test code depends only on
// `subtile_ir`, which is always compiled, so carrying it unconditionally pulls in no
// target-only modules.
/// The compiler -> target handoff types (`TileId`, `SlotMap`, `SourceBinding`,
/// `LoweredDecode`, `WeightKind`). Shared so a target's own compiler can name them.
pub mod handoff;
/// A target's KV codec steps, inserted around its coded KV writers and attentions.
pub mod kv_codec;
/// A target's sampled rows, inserted around the result matmul of a multi-row canonical.
pub mod sample_rows;

pub mod subtile_tape;
/// Arena colours for the tape's slots, under a target's declared alias facts.
pub mod tape_colouring;
/// Which tape steps a target computes inside another step's fused command, under its declared
/// fusion patterns.
pub mod tape_folding;
/// The unrolled tape read as steps over the source ops they compute.
pub mod tape_steps;

/// THE HOST TAPE — the program as data: the ordered host-side sequence of
/// kernel launches, transfers and CPU steps that a compiled bundle's kernels
/// are only the pieces of. Distinct from [`subtile_tape`] (the compiler's
/// rerolled op tape) and from metal's instruction tape.
///
/// UNGATED: it is plain data plus one generic loop, with no dependency on any
/// emitter — every target's player is the same function over it.
pub mod host_tape;

pub mod wave_schedule;
