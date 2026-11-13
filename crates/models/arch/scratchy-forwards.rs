// SPDX-License-Identifier: Apache-2.0
//! The model crate's build script. `scratchy-models` owns the models — every arch's DSL carrier
//! (`dsl/<name>.py`), its configs (`configs/<name>/`) — and the model SCOPE: its `<stem>`,
//! `<arch>`, `all`, `mtp`/`mtp-heads` features, with the `quant/<preset>` presets from
//! scratchy-quantizations. This script:
//!
//! - publishes that scope as build metadata (`links = "scratchy-models"`), which a target's own
//!   models crate reads back to compile the same selection with its emitter
//!   (`scratchy_models_build::ScopeMetadata`);
//! - writes the shell-completion registry (`hf_registry.rs`), a fact of the scope alone.
//!
//! It compiles no model: every target's models crate does, for its target.
use scratchy_models_build::{ScopeMetadata, write_if_changed};
use std::path::{Path, PathBuf};

// Gated on `hf-completions` (on by default for `scratchy-cli` builds; opt out
// with `--no-default-features` for an air-gapped build). This module owns every
// huggingface.co request the build makes, and its only dependency, `ureq`, is
// optional — so a build without the feature cannot reach the network from here
// even by accident. The registry FILE is still written either way
// (`write_hf_registry_file` below); only its contents differ.
#[cfg(feature = "hf-completions")]
#[path = "hf_registry_build.rs"]
mod hf_registry_build;

/// Write `$OUT_DIR/hf_registry.rs`, the shell-completion candidate list
/// `src/lib.rs` includes and `compiled_hf_registry()` returns.
///
/// Called on EVERY build, with `ids` empty when no registry was resolved (see
/// `hf_registry_build`) — the include in `src/lib.rs` carries no cfg
/// of its own, so the file must always exist. Rewrites only on a real change,
/// so an unchanged registry doesn't touch the mtime and retrigger downstream
/// crates.
fn write_hf_registry_file(out_dir: &Path, ids: &[String]) {
    let lits: Vec<String> = ids.iter().map(|s| format!("{s:?}")).collect();
    let rendered = format!(
        "/// Real HF org/repo ids matching an architecture (AND, when a quant\n\
         /// preset is enabled, a quant family) this binary compiled in support\n\
         /// for, resolved once against huggingface.co at BUILD time (see\n\
         /// `hf_registry_build.rs`). `scr model names` completes from this,\n\
         /// in this order: the Hub's, most downloaded first.\n\
         ///\n\
         /// Empty when this build resolved no registry — resolving it needs the\n\
         /// network, which an air-gapped build deliberately skips. `scr` then\n\
         /// completes no model names, rather than guessing.\n\
         pub static COMPILED_HF_REGISTRY: &[&str] = &[{}];\n",
        lits.join(", ")
    );
    write_if_changed(&out_dir.join("hf_registry.rs"), &rendered);
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // Load-bearing for the registry: gated repos (`meta-llama/*`) 401 and get dropped when no
    // token is set, so the SAME model scope resolves to a different candidate list with and
    // without it. Untracked, two builders would silently disagree about what completes.
    println!("cargo:rerun-if-env-changed=HF_TOKEN");

    // The scope's stem set: cargo's delivery of THIS crate's `model/*` feature
    // flags. THE LAW: model and quant selection IS the feature flags — no env
    // var of our invention, no second channel around them. This is the one
    // dynamic read (`cfg!` cannot loop over the stem list, and a hand-restated
    // probe table would be a second copy of Cargo.toml): the `CARGO_FEATURE_*`
    // names cargo itself sets for the enabled features of the crate whose
    // build script this is. The quant presets come from the crate that owns their feature flags:
    // scratchy-quantizations' build script publishes its `enabled_presets()`, and cargo hands
    // them to this script, its direct dependent, as `DEP_SCRATCHY_QUANTIZATIONS_PRESETS`.
    let metadata = ScopeMetadata {
        root: manifest.clone(),
        features: std::env::vars()
            .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(str::to_string))
            .collect(),
        mtp: cfg!(feature = "mtp"),
        mtp_heads: cfg!(feature = "mtp-heads"),
        presets: std::env::var("DEP_SCRATCHY_QUANTIZATIONS_PRESETS")
            .expect("scratchy-quantizations publishes its presets (`links` metadata)")
            .split(',')
            .filter(|p| !p.is_empty())
            .map(str::to_string)
            .collect(),
    };
    metadata.publish();

    // Shell-completion registry. Resolving it needs the network, so it happens
    // only under `hf-completions`; WRITING it is unconditional, because
    // `src/lib.rs` includes the file with no cfg of its own and
    // `compiled_hf_registry()` is public API. Feature off => an empty registry,
    // not a missing one.
    #[cfg(feature = "hf-completions")]
    let hf_ids = {
        // `crates/models/arch` -> repo root is 3 levels up.
        let repo_root = manifest
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .expect("crates/models/arch has a repo root 3 levels up")
            .to_path_buf();
        let arches = scratchy_models_build::selected_arches(&manifest, &metadata.scope());
        let quant_union = scratchy_models_build::quant_union(&arches, &metadata.presets);
        // `mod_name` is `arch.replace('-', "_")`; every arch name in
        // `configs/` uses `-` exclusively (never `_`), so this reverses
        // cleanly back to the original hyphenated identifier
        // `resolve_arch_tag`'s candidate rewrites need to split on.
        let hf_registry_targets: Vec<(String, PathBuf)> = arches
            .iter()
            .map(|(_, mod_name, configs_dir)| (mod_name.replace('_', "-"), configs_dir.clone()))
            .collect();
        hf_registry_build::resolve_hf_registry(&hf_registry_targets, &quant_union, &repo_root)
    };
    #[cfg(not(feature = "hf-completions"))]
    let hf_ids = Vec::<String>::new();
    write_hf_registry_file(&out_dir, &hf_ids);
}
