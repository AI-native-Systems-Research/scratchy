// SPDX-License-Identifier: Apache-2.0
//! The selected models, compiled for CUDA: the scope `scratchy-models` publishes, through the
//! shared models build driver with CUDA's emitter — less the arches CUDA does not lower, and
//! refused unless every arch it lowers compiles under the one preset CUDA builds.
use scratchy_forward_compiler_macro::config::feature_env_name;
use scratchy_models_build::{ModelsBuild, ScopeMetadata, quant_union, selected_arches};
use std::path::PathBuf;

/// Arches the cuda backend lowers: every op of their prefill canonicals has a CUDA kernel. Each
/// model of one emits its per-bucket `static` CUDA tapes (`CUDA_TAPES`).
///
/// Listed per arch, built per stem, and under `fp8-dynamic-per-channel` only ([`CUDA_PRESETS`]):
/// CUDA serves FP8 models only (#978). Not every config of a listed arch lowers (a tiny parity
/// config's GEMMs sit off the tiles), so the arch-wide `model/<arch>` panics at build time naming
/// that refusal. Build one FP8 checkpoint's stem with the preset: `model/qwen3.8-27b-text-only`
/// (the bare stem selects the vision arch), `model/qwen3.6-35b-a3b` or `model/gemma-4-26b-a4b-it`,
/// each with `quant/fp8-dynamic-per-channel`.
///
/// An arch selected but not listed here is refused by name below rather than silently dropped.
const CUDA_CAPABLE: &[&str] = &["qwen3-5", "qwen3-5-moe", "gemma4-moe"];
/// The quant presets the cuda backend builds, and the only ones: CUDA serves FP8 models only
/// (#978), so a cuda build that would compile a model dense, or under a preset CUDA has no
/// kernels for, is refused below.
const CUDA_PRESETS: &[&str] = &["fp8-dynamic-per-channel"];

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut metadata = ScopeMetadata::from_dep_env();
    // Every arch whose `arch-<name>` feature is on, whether or not cuda lowers it.
    let selected = selected_arches(&metadata.root, &metadata.scope());
    let arch_of = |mod_name: &str| mod_name.replace('_', "-");
    // Only the arches the cuda backend lowers (#978): the others leave the scope here, and are
    // refused by name below when none of the selected arches is one.
    let (targets, dropped): (Vec<_>, Vec<_>) = (selected.iter().cloned())
        .partition(|(_, mod_name, _)| CUDA_CAPABLE.contains(&arch_of(mod_name).as_str()));
    if !selected.is_empty() && targets.is_empty() {
        let selected: Vec<String> = selected.iter().map(|(_, m, _)| arch_of(m)).collect();
        panic!(
            "the cuda backend lowers none of the selected architectures \
             (CUDA_CAPABLE = {CUDA_CAPABLE:?}); selected: {selected:?}"
        );
    }
    // CUDA builds FP8 models only (#978): every selected CUDA arch compiles under exactly
    // `CUDA_PRESETS`, never dense (no preset) and never under a preset with no CUDA kernels.
    // Refused here, before any expansion.
    for arch in &targets {
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
    let off: Vec<String> = (dropped.iter())
        .map(|(_, m, _)| feature_env_name(&format!("arch-{}", arch_of(m))))
        .collect();
    metadata.features.retain(|f| !off.contains(f));
    scratchy_models_build::build(&ModelsBuild {
        metadata: &metadata,
        out_dir: &out_dir,
        emitter: &scratchy_target_cuda_compiler::Cuda,
        dump_dir: None,
        inspect: &|_| Vec::new(),
    });
}
