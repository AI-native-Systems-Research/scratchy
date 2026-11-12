// SPDX-License-Identifier: Apache-2.0
//! Spyre's compile-time emission. The shared pipeline (`scratchy-forward-compiler-macro`) calls
//! [`Spyre`] through its `TargetEmitter` seam; the per-model bake — the decode/prefill ladders
//! lowered to KTIR and SuperDSC, the launch wiring, the sharded worlds — is `bake`'s, and the
//! baked programs reach the binary as const tokens (`ktir_tokens`).

mod bake;
mod ktir_tokens;

use proc_macro2::TokenStream;
use scratchy_forward_compiler_macro::target::{BakeCx, TargetEmitter};

/// The spyre target's emitter.
pub struct Spyre;

impl TargetEmitter for Spyre {
    fn bake_model(&self, cx: &BakeCx<'_>) -> TokenStream {
        let mut out = TokenStream::new();
        match cx.decode_bounds {
            Some(decode_bounds) => bake::dump_wavefront_mega(
                cx.program,
                cx.model,
                cx.facts,
                cx.fuf,
                cx.inferred,
                decode_bounds,
                &mut out,
                cx.dump_dir,
            ),
            None => eprintln!(
                "[wavefront] {}: no decode (num_tokens=1) workload point",
                cx.model.source_stem
            ),
        }
        out
    }
}
