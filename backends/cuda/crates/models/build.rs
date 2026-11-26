// SPDX-License-Identifier: Apache-2.0
//! The selected models, compiled for CUDA: the scope `scratchy-models` publishes, through the
//! shared models build driver with CUDA's emitter, as spyre's are — refused unless every selected
//! arch compiles under the one preset CUDA builds.
use scratchy_models_build::{ModelsBuild, ScopeMetadata, quant_union, selected_arches};
use std::path::PathBuf;

/// The quant presets the cuda backend builds, and the only ones: CUDA serves FP8 models only
/// (#978), so a cuda build that would compile a model dense, or under a preset CUDA has no
/// kernels for, is refused below.
///
/// Every selected arch goes to CUDA's emitter, as on spyre: one CUDA does not lower fails the
/// build at its first refusal, by name. Build one FP8 checkpoint's stem with the preset:
/// `model/qwen3.8-27b-text-only` (the bare stem selects the vision arch), `model/qwen3.6-35b-a3b`
/// or `model/gemma-4-26b-a4b-it`, each with `quant/fp8-dynamic-per-channel`.
const CUDA_PRESETS: &[&str] = &["fp8-dynamic-per-channel"];

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let metadata = ScopeMetadata::from_dep_env();
    // Every arch whose `arch-<name>` feature is on.
    let selected = selected_arches(&metadata.root, &metadata.scope());
    // CUDA builds FP8 models only (#978): every selected arch compiles under exactly
    // `CUDA_PRESETS`, never dense (no preset) and never under a preset with no CUDA kernels.
    // Refused here, before any expansion.
    for arch in &selected {
        let q: Vec<String> = quant_union(std::slice::from_ref(arch), &metadata.presets)
            .into_iter()
            .collect();
        if q.is_empty() || q.iter().any(|p| !CUDA_PRESETS.contains(&p.as_str())) {
            panic!(
                "CUDA builds FP8 models only: add quant/fp8-dynamic-per-channel and no other \
                 quant preset (arch `{}` would compile with presets {q:?})",
                arch.1
            );
        }
    }
    scratchy_models_build::build(&ModelsBuild {
        metadata: &metadata,
        out_dir: &out_dir,
        emitter: &scratchy_target_cuda_compiler::Cuda,
        dump_dir: None,
        inspect: &|_| Vec::new(),
    });
}
