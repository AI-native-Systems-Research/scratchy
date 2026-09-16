// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Turnkey entrypoints to run a whole KTIR program through the optimized
//! (fused / resident) execution path.
//!
//! Running a whole program *optimized* — whole-program fusion + GPU offloads +
//! head-parallel attention — otherwise takes a few steps: gather the launch
//! group's functions, build a [`ProgramSpec`], then call
//! [`crate::segmented::execute_segmented`] or [`crate::resident::ResidentExecutor`].
//! (The per-node `interpreter::execute_function` path does NONE of this — it runs
//! each node in isolation at its native grid, where the GPU offloads, gated on a
//! single-core grid, never fire. That is the slow parity-oracle path.)
//!
//! These helpers collapse that to one call. They are **manifest-agnostic**: the
//! caller supplies the baked [`IRFunction`]s and a [`ProgramSpec`] built from its
//! own manifest (`ProgramSpec` / `NodeSpec` / `Binding` are plain public structs in
//! `ktir_optimizer::fusion`, re-exported as `ktir_emulator::ktir_optimizer`).
//!
//! There is NO text front door: a program is baked const IR, never parsed. The
//! `module_from_nodes` / `execute(&[&str], ..)` pair that turned per-node MLIR
//! into an `IRModule` went with the parser.
//!
//! - [`execute`] — turnkey single-shot (e.g. one prefill pass).
//! - [`Session`] — resident multi-pass serving (decode): weights uploaded ONCE,
//!   kernels chained on-device per pass with no weight re-marshal.
//!
//! Both require the `optimizer` feature (on by default).

use std::collections::HashMap;

use crate::interpreter::{Arg, Output};
use crate::ir::IRFunction;
use ktir_optimizer::fusion::ProgramSpec;

/// Turnkey single-shot: run a launch group's functions through the optimized
/// segmented path (whole-program fusion + K-loop GEMM / map-window GPU offloads +
/// native head-parallel attention).
///
/// `spec` describes the program (node order, arg↔tensor bindings, which tensors
/// are sources vs results) — build it from your manifest. `args` are the source
/// tensors (weights + inputs); `outputs` the result tensor ids to read back. For
/// repeated passes over the same weights (decode), use [`Session`].
pub fn execute(
    funcs: &[IRFunction<'static>],
    spec: &ProgramSpec,
    args: &[(u64, Arg)],
    outputs: &[u64],
) -> Result<HashMap<u64, Output>, String> {
    crate::segmented::execute_segmented(funcs, spec, args, outputs)
}

/// A resident serving session for MULTI-pass execution (e.g. autoregressive
/// decode). Weights are marshaled into a persistent GPU HBM **once** at
/// construction; each [`run`](Session::run) chains the program's kernels
/// on-device with no per-pass weight re-marshal. Between passes, overwrite only
/// the changing source tensors (the next token's input activation, the updated
/// attention mask) with [`set_sources`](Session::set_sources).
///
/// The session owns its object graph and is therefore `Send` — a serving worker
/// can store it and move it between threads (it is single-threaded internally, so
/// NOT `Sync`: don't share one by `&` across threads; run it serially):
/// ```ignore
/// let mut sess = program::Session::new(group.programs, &spec, &weights)?;
/// loop {
///     sess.set_sources(&[(0, next_input), (mask_tid, mask)])?;
///     let out = sess.run(&[result_tid])?;
/// }
/// ```
pub struct Session {
    exec: crate::resident::ResidentExecutor,
}

impl Session {
    /// Build the session over a launch group's functions. `weights` is the full
    /// initial source set (weights + mask + first input), keyed by tensor id; it is
    /// uploaded to resident HBM once here.
    pub fn new(
        funcs: &[IRFunction<'static>],
        spec: &ProgramSpec,
        weights: &[(u64, Arg)],
    ) -> Result<Self, String> {
        let mut exec = crate::resident::ResidentExecutor::new(funcs, spec)?;
        exec.set_sources(weights)?;
        Ok(Self { exec })
    }

    /// Build a session whose resident weights are SHARED across multiple programs
    /// — the prefill and decode bundles, which use the SAME weights but different
    /// shapes (M). The weight set is uploaded ONCE here; both programs run against
    /// it with no second load. `programs` are in declaration order: `run_program(0,
    /// ..)` runs the first, `run_program(1, ..)` the second, etc.
    pub fn new_multi(
        programs: Vec<(&[IRFunction<'static>], &ProgramSpec)>,
        weights: Vec<(u64, Arg)>,
    ) -> Result<Self, String> {
        let mut exec = crate::resident::ResidentExecutor::new_multi(programs)?;
        // TAKEN, not borrowed: at construction the whole checkpoint is written once, so the
        // weight buffers can be moved into HBM rather than copied into it.
        exec.set_sources_owned(weights)?;
        Ok(Self { exec })
    }

    /// Overwrite source tensors in resident HBM (the per-pass input / mask).
    /// Sources you don't pass keep their resident bytes — so weights stay put.
    pub fn set_sources(&mut self, args: &[(u64, Arg)]) -> Result<(), String> {
        self.exec.set_sources(args)
    }

    /// Run one forward pass of program 0 and read back `outputs` (empty = results).
    pub fn run(&mut self, outputs: &[u64]) -> Result<HashMap<u64, Output>, String> {
        self.exec.run(outputs)
    }

    /// Run one forward pass of program `idx` (e.g. 0 = prefill, 1 = decode) against
    /// the shared resident weights, reading back `outputs` (empty = that program's
    /// results). Switching programs re-uploads no weights.
    pub fn run_program(
        &mut self,
        idx: usize,
        outputs: &[(u64, usize)],
    ) -> Result<HashMap<u64, Output>, String> {
        self.exec.run_program(idx, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Locks in the `unsafe impl Send for ResidentExecutor` contract: a serving
    // worker needs `Session: Send`. If a future change reintroduces a borrow or a
    // non-Send field, this stops compiling instead of silently regressing.
    #[test]
    fn session_and_executor_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Session>();
        assert_send::<crate::resident::ResidentExecutor>();
    }
}
