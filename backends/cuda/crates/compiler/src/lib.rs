// SPDX-License-Identifier: Apache-2.0
//! CUDA'S COMPILE-TIME LOWERING.
//!
//! ⭐ TARGET CODEGEN, IN THE TARGET'S DIRECTORY — the counterpart of `scratchy-target-metal-compiler`.
//! [`canonical::plan_canonical`] runs the shared passes (sampled rows, tape, folds, colours) under
//! CUDA's declared tables; [`canonical::lower_canonical`] adds CUDA's opcode lowering
//! ([`steps_from_tape`]) on top. [`Cuda`] is the shared pipeline's entry (its `TargetEmitter`):
//! [`emit::emit_model`] lowers every canonical of one model and emits its `static` tapes.

/// One decode canonical, planned and lowered.
pub mod canonical;
/// One model's tapes, lowered per bucket and emitted as `static` tokens: the macro's entry.
pub mod emit;
/// CUDA's opcode lowering: one tape step to one [`scratchy_target_cuda::step::CudaStep`].
pub mod steps_from_tape;

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_forward_compiler_macro::target::{LoadCx, ModelCx, ModelEmission, TargetEmitter};
use scratchy_subtile::handoff::TargetModelInput;

/// The CUDA target's emitter: the shared weights struct, and beside it the model's lowered tapes
/// ([`emit::emit_model`]). The worker loads the weights itself through each model's
/// `CUDA_WEIGHTS`, so the canonical `load` refuses, and the facade is the shared layers crate over
/// CUDA's host-only allocator (`scratchy_target_cuda::facade`).
pub struct Cuda;

impl TargetEmitter for Cuda {
    fn name(&self) -> &'static str {
        "cuda"
    }

    /// CUDA serves FP8 models only (#978): the one preset its kernels realize.
    fn lowers_preset(&self, preset: &str) -> bool {
        preset == "fp8-dynamic-per-channel"
    }

    fn load_fn(&self, cx: &LoadCx<'_>) -> TokenStream {
        let _ = cx;
        quote! {
            /// Refuses: the CUDA worker loads a model's weights through its `CUDA_WEIGHTS` table.
            pub fn load(
                _gw: &mut crate::__gpu::weights::GpuWeights,
                _stream: crate::__gpu::LoadStream,
                _max_model_len: usize,
                _tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                ::anyhow::bail!(
                    "a CUDA model loads through its `CUDA_WEIGHTS` table, in the CUDA worker"
                )
            }
        }
    }

    fn facade(&self) -> TokenStream {
        quote! { ::scratchy_target_cuda::facade }
    }

    fn model(&self, cx: &ModelCx<'_>) -> ModelEmission {
        ModelEmission {
            weights: cx.struct_of(None),
            items: emit_handoff(&cx.handoff()),
        }
    }
}

/// [`emit::emit_model`], refusing by name a model whose facts state no rope: CUDA has no unroped
/// attention kernel.
fn emit_handoff(input: &TargetModelInput) -> TokenStream {
    if input.facts.rope_theta.is_none() {
        panic!(
            "[cuda emit] {}: the model states no rope, and CUDA has no unroped attention kernel",
            input.stem
        );
    }
    emit::emit_model(input)
}

/// [`Cuda`], keeping a copy of every model's handoff: what a test checks the emitted tapes against.
#[cfg(feature = "test-support")]
struct Recording(std::sync::Mutex<Vec<TargetModelInput>>);

#[cfg(feature = "test-support")]
impl TargetEmitter for Recording {
    fn name(&self) -> &'static str {
        Cuda.name()
    }

    fn lowers_preset(&self, preset: &str) -> bool {
        Cuda.lowers_preset(preset)
    }

    fn load_fn(&self, cx: &LoadCx<'_>) -> TokenStream {
        Cuda.load_fn(cx)
    }

    fn facade(&self) -> TokenStream {
        Cuda.facade()
    }

    fn model(&self, cx: &ModelCx<'_>) -> ModelEmission {
        let input = cx.handoff();
        (self.0.lock().unwrap_or_else(|e| e.into_inner())).push(input.clone());
        ModelEmission {
            weights: cx.struct_of(None),
            items: emit_handoff(&input),
        }
    }
}

/// The shared pipeline over `carrier` with [`Cuda`], returning every [`TargetModelInput`] it handed
/// the emitter, one per emitted model, instead of the tokens: so a test can check what CUDA emitted
/// against what it lowers from the same canonicals.
#[cfg(feature = "test-support")]
pub fn target_model_inputs(
    carrier: scratchy_forward_compiler_macro::PythonCarrier,
    models_dir: &std::path::Path,
    scope: &scratchy_forward_compiler_macro::Scope,
) -> Result<Vec<TargetModelInput>, scratchy_forward_compiler_macro::CarrierError> {
    let recording = Recording(std::sync::Mutex::new(Vec::new()));
    scratchy_forward_compiler_macro::compile_carrier(carrier, models_dir, None, scope, &recording)?;
    Ok(recording.0.into_inner().unwrap_or_else(|e| e.into_inner()))
}
