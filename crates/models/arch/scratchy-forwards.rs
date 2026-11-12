// SPDX-License-Identifier: Apache-2.0
//! The model crate's build script. `scratchy-models` owns the models — every arch's DSL carrier
//! (`dsl/<name>.py`), its configs (`configs/<name>/`) — and the model SCOPE: its `<stem>`,
//! `<arch>`, `all`, `mtp`/`mtp-heads` features, with the `quant/<preset>` presets from
//! scratchy-quantizations. This script:
//!
//! - publishes that scope as build metadata (`links = "scratchy-models"`), which a target's own
//!   models crate reads back to compile the same selection with its emitter
//!   (`scratchy_models_build::ScopeMetadata`);
//! - under `metal`, compiles the selected models for metal itself, through the shared driver
//!   (`scratchy_models_build::build`) — `src/lib.rs` wraps each emitted arch in `pub mod <mod>`;
//! - writes the shell-completion registry (`hf_registry.rs`), a fact of the scope alone.
#[cfg(feature = "metal")]
use proc_macro2::{TokenStream, TokenTree};
#[cfg(feature = "metal")]
use scratchy_models_build::ModelsBuild;
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

/// The model modules of one arch's emit that carry a metal tape: each top-level
/// `pub mod <model> { … }` whose body names `METAL_BUCKETS` (a canonical's
/// `static`, a shim's re-export of its canonical's).
#[cfg(feature = "metal")]
fn metal_tape_models(arch: &TokenStream) -> Vec<String> {
    fn names_metal_buckets(ts: TokenStream) -> bool {
        ts.into_iter().any(|tt| match tt {
            TokenTree::Ident(i) => i == "METAL_BUCKETS",
            TokenTree::Group(g) => names_metal_buckets(g.stream()),
            _ => false,
        })
    }
    let tts: Vec<TokenTree> = arch.clone().into_iter().collect();
    (tts.windows(4))
        .filter_map(|w| match w {
            [
                TokenTree::Ident(vis),
                TokenTree::Ident(kw),
                TokenTree::Ident(model),
                TokenTree::Group(body),
            ] if vis == "pub" && kw == "mod" && names_metal_buckets(body.stream()) => {
                Some(model.to_string())
            }
            _ => None,
        })
        .collect()
}

/// Write `$OUT_DIR/metal_models.rs`: `METAL_MODELS`, every model module this
/// build bakes a metal tape for (`<arch>::<model>`, sorted) beside its
/// `METAL_BUCKETS`. `src/lib.rs` includes it under `metal`.
#[cfg(feature = "metal")]
fn write_metal_models(out_dir: &Path, models: &[String]) {
    let rows: String = (models.iter())
        .map(|m| format!("    ({m:?}, crate::{m}::METAL_BUCKETS),\n"))
        .collect();
    let rendered = format!(
        "/// Every model module this build bakes a metal tape for (`<arch>::<model>`), beside\n\
         /// its baked buckets: read off the emitted modules by the build script, sorted.\n\
         pub static METAL_MODELS: &[(\n    \
             &str,\n    \
             &[::scratchy_target_metal::interpreter::metal::MetalBucketSpec],\n\
         )] = &[\n{rows}];\n"
    );
    write_if_changed(&out_dir.join("metal_models.rs"), &rendered);
}

/// Write `$OUT_DIR/metal_bake.rs`: the crate-root `__metal_bake` module (`src/lib.rs`) the
/// emitted models' baked kernels resolve in — every kernel the arches named, baked once. Runs
/// after every arch is emitted.
#[cfg(feature = "metal")]
fn write_metal_bake(out_dir: &Path) {
    let mut rendered = String::with_capacity(1 << 20);
    scratchy_forward_compiler_macro::render_tokens(
        scratchy_forward_compiler_macro::metal_bake_module(),
        &mut rendered,
    );
    write_if_changed(&out_dir.join("metal_bake.rs"), &rendered);
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
    // build script this is. The quant presets come from
    // `scratchy_quantizations::enabled_presets()` — the `quant/*` feature flags read by `cfg!`
    // in the crate that owns them.
    let metadata = ScopeMetadata {
        root: manifest.clone(),
        features: std::env::vars()
            .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(str::to_string))
            .collect(),
        mtp: cfg!(feature = "mtp"),
        mtp_heads: cfg!(feature = "mtp-heads"),
        presets: scratchy_quantizations::enabled_presets()
            .into_iter()
            .map(str::to_string)
            .collect(),
    };
    metadata.publish();

    // The arch gates in `src/lib.rs` read `scratchy_backend`, which this script sets exactly when
    // it compiles the models itself — so an arch's `pub mod` exists iff its `arch-<name>` feature
    // is on and this build emitted it.
    println!("cargo::rustc-check-cfg=cfg(scratchy_backend)");
    #[cfg(feature = "metal")]
    {
        println!("cargo::rustc-cfg=scratchy_backend");
        let built = scratchy_models_build::build(&ModelsBuild {
            metadata: &metadata,
            out_dir: &out_dir,
            emitter: &scratchy_forward_compiler_macro::target::Metal,
            dump_dir: None,
            inspect: &metal_tape_models,
        });
        write_metal_bake(&out_dir);
        write_metal_models(&out_dir, &built.inspected);
    }
    // Without a target compiled here the crate is the scope alone: no model, no tokenizer.
    #[cfg(not(feature = "metal"))]
    {
        write_if_changed(&out_dir.join("baked_tokenizers.rs"), "");
    }

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
