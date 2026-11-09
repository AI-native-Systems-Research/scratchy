// SPDX-License-Identifier: Apache-2.0
//! THE single build-time driver for every model arch (Step 6 of the codegen
//! redesign — the 25 per-arch `#[forward]` crates collapsed into one).
//!
//! For each arch whose `arch-<name>` feature is enabled, this reads that arch's
//! DSL carrier from `dsl/<name>.py` (the `@forward`/`@vision_forward`-decorated
//! `def` — the same math the `#[forward]` attribute used to annotate, now in
//! torch-idiom Python), runs the shared pipeline
//! (`scratchy_forward_compiler_macro::compile_carrier`) over it against that arch's
//! own `configs/<name>/`, and writes the emitted modules to
//! `$OUT_DIR/<mod>.rs`, which `src/lib.rs` wraps in `pub mod <mod>` and
//! `include!`s. The pipeline lib is a build-dependency; the backend feature
//! (metal/spyre/…) is forwarded onto it so the emit matches what we then compile.
use proc_macro2::{Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use rayon::prelude::*;
use scratchy_forward_compiler_macro::{
    CarrierError, Scope, compile_carrier, parse_python_file, render_tokens,
};
use std::collections::{BTreeMap, BTreeSet};
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

/// The emitted code roots its own items at `crate::` (each arch used to be its
/// own crate, so `crate::__gpu` / `crate::<Model>` meant "this arch's root").
/// In the consolidated crate every arch lives under `pub mod <mod>`, so rewrite
/// every path-root `crate ::` → `crate :: <mod> ::`. Leaves `pub(crate)` (a
/// `crate` not followed by `::`), absolute `::foo` paths and `crate::__metal_bake`
/// — the one module every arch shares, at the crate root (`write_metal_bake`) —
/// untouched.
fn reroot_crate(ts: TokenStream, mod_name: &str) -> TokenStream {
    let mut out: Vec<TokenTree> = Vec::new();
    let mut it = ts.into_iter().peekable();
    while let Some(tt) = it.next() {
        match tt {
            TokenTree::Group(g) => {
                let ng = Group::new(g.delimiter(), reroot_crate(g.stream(), mod_name));
                out.push(TokenTree::Group(ng));
            }
            TokenTree::Ident(id) if id == "crate" => {
                out.push(TokenTree::Ident(id));
                // Followed by `::` (first colon is Joint)? Then it's a path root.
                let is_path = matches!(it.peek(),
                    Some(TokenTree::Punct(p)) if p.as_char() == ':' && p.spacing() == Spacing::Joint);
                if is_path {
                    out.push(it.next().unwrap()); // ':'
                    out.push(it.next().unwrap()); // ':'
                    let shared =
                        matches!(it.peek(), Some(TokenTree::Ident(id)) if id == "__metal_bake");
                    if !shared {
                        out.push(TokenTree::Ident(Ident::new(mod_name, Span::call_site())));
                        out.push(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                        out.push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
                    }
                }
            }
            other => out.push(other),
        }
    }
    out.into_iter().collect()
}

/// The quant presets enabled for one arch: its `quantizations.json`
/// ∩ the enabled preset features (no `quant-` prefix on scratchy-quantizations,
/// same reasoning as `<stem>` having no `model-` prefix on this crate).
/// Quant scope lives on scratchy-quantizations, not this crate (so
/// `crates/cli/scr/Cargo.toml` can alias it `quant`, distinct from this
/// crate's `model` alias — Cargo forbids aliasing the same crate twice), so
/// this reads `scratchy_quantizations::enabled_presets()` (a build-dependency call,
/// baked in at scratchy-quantizations' OWN compile time via `cfg!`) instead
/// of this crate's own `CARGO_FEATURE_QUANT_*`, which no longer exist.
fn enabled_quants(
    qpath: &Path,
    enabled_presets: &std::collections::HashSet<&'static str>,
) -> Vec<String> {
    // Only watch the file if it EXISTS. `rerun-if-changed` on a missing path
    // makes cargo rerun the build script on every build — and 6 arches (the
    // vision ones + modernbert) ship no quantizations.json. Adds/removes of the
    // file are still caught: emit_arch watches the whole `configs/<arch>/` dir.
    if qpath.is_file() {
        println!("cargo:rerun-if-changed={}", qpath.display());
    }
    let mut enabled: Vec<String> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(qpath)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(arr) = json.get("quantizations").and_then(|v| v.as_array())
    {
        for entry in arr {
            let name = match entry {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(o) if o.len() == 1 => o.keys().next().unwrap().clone(),
                _ => continue,
            };
            if enabled_presets.contains(name.as_str()) {
                enabled.push(name);
            }
        }
    }
    enabled
}

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

/// Emit one arch: parse its DSL carrier (`dsl/<arch>.py`), run the pipeline
/// against its configs (`configs/<arch>/`), write `$OUT_DIR/<mod>.rs`. Called
/// concurrently across arches — must not mutate global state (e.g. env); the
/// scope (models, heads, quant presets) is the one `&Scope`
/// value `main` stated before the fan-out, borrowed by every arch.
///
/// Returns the path (`<mod>::<model>`) of every model module the emit gives a
/// `METAL_BUCKETS` ([`metal_tape_models`]).
fn emit_arch(
    dsl_path: &Path,
    configs_dir: &Path,
    out_dir: &Path,
    mod_name: &str,
    scope: &Scope,
) -> Vec<String> {
    println!("cargo:rerun-if-changed={}", dsl_path.display());
    // Watch the dir itself (its mtime bumps on add/remove) plus each file.
    println!("cargo:rerun-if-changed={}", configs_dir.display());
    if let Ok(rd) = std::fs::read_dir(configs_dir) {
        for e in rd.flatten() {
            println!("cargo:rerun-if-changed={}", e.path().display());
        }
    }

    let dsl = std::fs::read_to_string(dsl_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", dsl_path.display()));
    let carrier =
        parse_python_file(&dsl).unwrap_or_else(|e| panic!("carrier {}: {e}", dsl_path.display()));
    // The stem arrives as a FIELD of the typed error (compile_carrier is called once per
    // arch, so the per-config stem can only enter there) — printed, never parsed.
    // A `superdsc-dump` build hands the bake THIS build script's output directory — the one place
    // that knows it — so each baked bundle's dxp input lands beside the device images; any other
    // build writes no dump.
    let dump_dir = cfg!(feature = "superdsc-dump").then_some(out_dir);
    let tokens =
        compile_carrier(carrier, configs_dir, dump_dir, scope).unwrap_or_else(|e| match e {
            CarrierError::Config { stem, source } => {
                panic!("forward pipeline ({mod_name}/{stem}): {source}")
            }
            CarrierError::Arch(source) => panic!("forward pipeline ({mod_name}): {source}"),
        });
    // Re-root the emit under `pub mod <mod>` (see reroot_crate).
    let tokens = reroot_crate(tokens, mod_name);
    let models = (metal_tape_models(&tokens).into_iter())
        .map(|model| format!("{mod_name}::{model}"))
        .collect();

    let mut rendered = String::with_capacity(1 << 20);
    render_tokens(tokens, &mut rendered);
    write_if_changed(&out_dir.join(format!("{mod_name}.rs")), &rendered);
    models
}

/// The model modules of one arch's emit that carry a metal tape: each top-level
/// `pub mod <model> { … }` whose body names `METAL_BUCKETS` (a canonical's
/// `static`, a shim's re-export of its canonical's).
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
/// build bakes a metal tape for (`emit_arch`'s returns, sorted) beside its
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
    render_tokens(
        scratchy_forward_compiler_macro::metal_bake_module(),
        &mut rendered,
    );
    write_if_changed(&out_dir.join("metal_bake.rs"), &rendered);
}

/// Write `text` to `out` only when it changed, so a build-script rerun with identical output
/// doesn't bump the file's mtime and force rustc to recompile the crate.
fn write_if_changed(out: &Path, text: &str) {
    let unchanged = std::fs::read_to_string(out).is_ok_and(|old| old == text);
    if !unchanged {
        std::fs::write(out, text).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    }
}

/// A checkpoint config's identity for matching: its architecture and shape (hidden size, layers,
/// vocab size; the text decoder's when `text_config` nests it).
fn config_shape(path: &Path) -> Option<(String, u64, u64, u64)> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let text = cfg
        .get("text_config")
        .filter(|t| t.is_object())
        .unwrap_or(&cfg);
    let field = |k: &str| {
        text.get(k)
            .or_else(|| cfg.get(k))
            .and_then(serde_json::Value::as_u64)
    };
    Some((
        cfg["architectures"][0].as_str()?.to_string(),
        field("hidden_size")?,
        field("num_hidden_layers")?,
        field("vocab_size")?,
    ))
}

/// Write `$OUT_DIR/baked_tokenizers.rs`: a `BakedTokenizer` for the tokenizer of every checkpoint
/// in the Hub cache that is a model this build compiles — its `config.json` has a selected
/// config's architecture and shape — baked from its `tokenizer.json`
/// (`scratchy_core_model::baked_tokenizer`). A tokenizer `bake` refuses is reported and loads
/// from `tokenizer.json` at runtime as before, as does any checkpoint the cache does not hold.
fn bake_tokenizers(out_dir: &Path) {
    println!("cargo:rerun-if-env-changed=HF_HUB_CACHE");
    println!("cargo:rerun-if-env-changed=HF_HOME");
    let compiled: BTreeSet<_> = scratchy_forward_compiler_macro::selected_config_paths()
        .iter()
        .filter_map(|config| config_shape(config))
        .collect();
    let root = hf_hub_downloader::cache::default_root();
    if root.is_dir() {
        // A model pulled after this build bakes on the next.
        println!("cargo:rerun-if-changed={}", root.display());
    }
    let snapshots = std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|repo| std::fs::read_dir(repo.path().join("snapshots")).ok())
        .flatten()
        .flatten()
        .map(|snapshot| snapshot.path());
    let mut bakes = BTreeMap::new();
    for snapshot in snapshots {
        let tokenizer = snapshot.join("tokenizer.json");
        if !config_shape(&snapshot.join("config.json")).is_some_and(|s| compiled.contains(&s)) {
            continue;
        }
        let Some(digest) = scratchy_core_model::baked_tokenizer::cached_digest(&tokenizer) else {
            continue;
        };
        if bakes.contains_key(&digest) {
            continue;
        }
        let json = std::fs::read_to_string(&tokenizer).expect("read a cached tokenizer.json");
        match scratchy_core_model::baked_tokenizer::bake(&json, &digest) {
            Ok(bake) => {
                bakes.insert(digest, bake);
            }
            Err(why) => println!("cargo:warning={} is not baked: {why}", tokenizer.display()),
        }
    }
    // Checkpoints whose `tokenizer.json`s differ only outside the BPE model (fine-tunes of one
    // base) bake the same tables: they share one copy, named by the first digest baking it.
    let mut tables: BTreeMap<&[u8], String> = BTreeMap::new();
    let mut code = String::new();
    for (digest, bake) in &bakes {
        let owner = tables.entry(bake.tables.as_slice()).or_insert_with(|| {
            let owner = format!("TOKENIZER_TABLES_{}", digest.to_uppercase());
            write_if_changed_bytes(&out_dir.join(format!("tokenizer-{digest}.bin")), &bake.tables);
            code += &format!(
                "static {owner}: ::scratchy_core_model::baked_tokenizer::Aligned<[u8; {len}]> = \
                 ::scratchy_core_model::baked_tokenizer::Aligned(*include_bytes!(concat!(env!(\"OUT_DIR\"), \"/tokenizer-{digest}.bin\")));\n",
                len = bake.tables.len(),
            );
            owner
        });
        write_if_changed(
            &out_dir.join(format!("tokenizer-{digest}.json")),
            &bake.pipeline,
        );
        code += &format!(
            "::scratchy_forward_compiler::inventory::submit! {{\n    \
             ::scratchy_core_model::baked_tokenizer::BakedTokenizer {{\n        \
             digest: \"{digest}\",\n        \
             tables: &{owner},\n        \
             pipeline: include_str!(concat!(env!(\"OUT_DIR\"), \"/tokenizer-{digest}.json\")),\n    \
             }}\n}}\n"
        );
    }
    write_if_changed(&out_dir.join("baked_tokenizers.rs"), &code);
}

fn write_if_changed_bytes(out: &Path, bytes: &[u8]) {
    if std::fs::read(out).ok().as_deref() != Some(bytes) {
        std::fs::write(out, bytes).expect("write a baked tokenizer");
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for name in [
        // Spyre/KTIR bundle knobs read at emit time by `dump_wavefront_mega`:
        // they bake the decode row count and prefill width into the embedded KTIR
        // bundle, so a change must re-run this build script (which regenerates the
        // bundle). Without this, cargo keeps the stale OUT_DIR codegen.
        "KTIR_M",
        "KTIR_PREFILL_LEN",
        // Load-bearing for the registry: gated repos (`meta-llama/*`) 401 and
        // get dropped when no token is set, so the SAME model scope resolves to
        // a different candidate list with and without it. Untracked, two
        // builders would silently disagree about what completes.
        "HF_TOKEN",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    // The backend law, stated once: `src/lib.rs`'s arch gates read `scratchy_backend`,
    // which this script sets exactly when a backend feature is on — so an arch compiles
    // under a backend iff its `arch-<name>` feature and that backend are on, with no
    // per-arch spelling left to drift.
    println!("cargo::rustc-check-cfg=cfg(scratchy_backend)");
    if cfg!(feature = "metal") || cfg!(feature = "spyre") {
        println!("cargo::rustc-cfg=scratchy_backend");
    }

    // Each arch is a `dsl/<arch>.py` DSL carrier paired with a
    // `configs/<arch>/` dir; emit the ones whose `arch-<name>` feature is on AND
    // a backend they support is on. Sorted for deterministic output.
    let dsl_dir = manifest.join("dsl");
    println!("cargo:rerun-if-changed={}", dsl_dir.display());
    // A head that prefills at its target's chunk width must have every rung its target has, or a
    // chunk at a rung it lacks panics at runtime: refuse the build instead, before any expansion.
    // Every head is checked, selected or not, so every arch.json the check reads is watched (dsl/
    // already is, file by file), and an incremental build agrees with a clean one.
    for e in std::fs::read_dir(manifest.join("configs"))
        .unwrap()
        .flatten()
    {
        let arch_json = e.path().join("arch.json");
        if arch_json.is_file() {
            println!("cargo:rerun-if-changed={}", arch_json.display());
        }
    }
    scratchy_forward_compiler_macro::check_head_ladders(&manifest)
        .unwrap_or_else(|e| panic!("head ladder: {e}"));
    let mut dsl_files: Vec<PathBuf> = std::fs::read_dir(&dsl_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".py"))
        })
        .collect();
    dsl_files.sort();

    // The scope's stem set: cargo's delivery of THIS crate's `model/*` feature
    // flags. THE LAW: model and quant selection IS the feature flags — no env
    // var of our invention, no second channel around them. This is the one
    // dynamic read (`cfg!` cannot loop over the stem list, and a hand-restated
    // probe table would be a second copy of Cargo.toml): the `CARGO_FEATURE_*`
    // names cargo itself sets for the enabled features of the crate whose
    // build script this is — the same fact `cfg!` states one name at a time.
    // The head flags ride the same statement as `cfg!` booleans, valid here in
    // the crate that owns them.
    let enabled_features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(str::to_string))
        .collect();
    // The build's one statement of what it selects — stated once here, before
    // the fan-out; every arch borrows the SAME &Scope, no re-derivation per
    // thread, no second read.
    let scope = Scope::from_feature_flags(
        enabled_features,
        cfg!(feature = "mtp"),
        cfg!(feature = "mtp-heads"),
    );

    // (dsl_path, mod_name, configs_dir) for each arch that passes the feature +
    // backend gates.
    let targets: Vec<(PathBuf, String, PathBuf)> = dsl_files
        .iter()
        .filter_map(|dsl_path| {
            let arch = dsl_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".py"))?
                .to_string();
            // Its `arch-<name>` feature and a backend it compiles for, as the `pub mod <name>`
            // cfg in src/lib.rs has it: never the pipeline of an arch whose mod won't compile.
            if !scope.arch_compiled(&arch) {
                return None;
            }
            let mod_name = arch.replace('-', "_");
            let configs_dir = manifest.join("configs").join(&arch);
            Some((dsl_path.clone(), mod_name, configs_dir))
        })
        .collect();

    // The global union of enabled presets, computed once and stated IN the scope —
    // config.rs intersects it with each arch's own quantizations.json, so the
    // per-arch result is identical to setting it per arch. A value, not an env
    // var: the scope is threaded through the rayon fan-out below, so nothing is
    // restated through the environment to survive the parallelism. The presets
    // come from `scratchy_quantizations::enabled_presets()` — the `quant/*`
    // feature flags read by `cfg!` in the crate that owns them, the pure form.
    let enabled_presets: std::collections::HashSet<&'static str> =
        scratchy_quantizations::enabled_presets()
            .into_iter()
            .collect();
    let quant_union: BTreeSet<String> = targets
        .iter()
        .flat_map(|(_, _, cfg)| enabled_quants(&cfg.join("quantizations.json"), &enabled_presets))
        .collect();
    let scope = scope.with_quants(quant_union.iter().cloned());

    // Emit arches in parallel. compile_carrier is itself rayon-parallel over
    // models, so this is nested (arch × model) work-stealing across all cores —
    // recovering the cross-crate parallelism the per-arch crate split used to get
    // from cargo, in one process.
    let mut metal_models: Vec<String> = targets
        .par_iter()
        .flat_map(|(dsl_path, mod_name, configs_dir)| {
            emit_arch(dsl_path, configs_dir, &out_dir, mod_name, &scope)
        })
        .collect();
    metal_models.sort();
    #[cfg(feature = "metal")]
    write_metal_bake(&out_dir);
    #[cfg(feature = "metal")]
    write_metal_models(&out_dir, &metal_models);
    bake_tokenizers(&out_dir);

    // Shell-completion registry. Resolving it needs the network, so it happens
    // only under `hf-completions`; WRITING it is unconditional, because
    // `src/lib.rs` includes the file with no cfg of its own and
    // `compiled_hf_registry()` is public API. Feature off => an empty registry,
    // not a missing one (same shape as `crates/compiler/subtile/build.rs`'s
    // `CARGO_FEATURE_SUPERDSC` gate: always emit, vary the contents).
    #[cfg(feature = "hf-completions")]
    let hf_ids = {
        // `crates/models/arch` -> repo root is 3 levels up.
        let repo_root = manifest
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .expect("crates/models/arch has a repo root 3 levels up")
            .to_path_buf();
        // `mod_name` is `arch.replace('-', "_")`; every arch name in
        // `configs/` uses `-` exclusively (never `_`), so this reverses
        // cleanly back to the original hyphenated identifier
        // `resolve_arch_tag`'s candidate rewrites need to split on.
        let hf_registry_targets: Vec<(String, PathBuf)> = targets
            .iter()
            .map(|(_, mod_name, configs_dir)| (mod_name.replace('_', "-"), configs_dir.clone()))
            .collect();
        hf_registry_build::resolve_hf_registry(&hf_registry_targets, &quant_union, &repo_root)
    };
    #[cfg(not(feature = "hf-completions"))]
    let hf_ids = Vec::<String>::new();
    write_hf_registry_file(&out_dir, &hf_ids);

    // scratchy-models has no default model features (deliberately — see
    // `default = []` in Cargo.toml) — a build naming zero `<stem>`/`<arch>`/
    // `all` features would otherwise silently link a binary with no models
    // baked in. Fail loudly instead: name at least one model feature (e.g.
    // `--features metal,granite-3.1-2b-instruct`, or `model/<stem>` from
    // scratchy-cli).
    if scratchy_forward_compiler_macro::total_models_emitted() == 0 {
        panic!(
            "no models selected — scratchy-models has no default model features. \
             Enable at least one, e.g. `--features metal,granite-3.1-2b-instruct` \
             (or `--features metal,<arch>` / `--features metal,all` for a wider \
             scope; from scratchy-cli, `--features metal,model/<stem>`)."
        );
    }
}
