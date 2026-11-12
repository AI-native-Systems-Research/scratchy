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
/// The head-dim value→const door whose arms the build script generated from
/// `crates/models/arch/*/configs/*.json`, typed on `scratchy_ir::model_facts::HeadDim`. Always
/// compiled: the value side of the geometry is the facts' own [`Attention`], threaded from the
/// bake, whatever the target (scratchy_ir::model_facts::Attention).
pub mod model_geometry;
pub mod subtile_ir;
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
/// Tensor parallelism by analysis: the per-rank forward, each source's slice law, and the
/// collectives the dataflow demands — derived from the op registry's sharding rules alone.
pub mod tensor_parallel;

/// THE HOST TAPE — the program as data: the ordered host-side sequence of
/// kernel launches, transfers and CPU steps that a compiled bundle's kernels
/// are only the pieces of. Distinct from [`subtile_tape`] (the compiler's
/// rerolled op tape) and from metal's instruction tape.
///
/// UNGATED: it is plain data plus one generic loop, with no dependency on any
/// emitter — every target's player is the same function over it.
pub mod host_tape;

pub mod wave_schedule;
