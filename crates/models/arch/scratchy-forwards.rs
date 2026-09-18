// SPDX-License-Identifier: Apache-2.0
//! THE single build-time driver for every model arch (Step 6 of the codegen
//! redesign — the 25 per-arch `#[forward]` crates collapsed into one).
//!
//! For each arch whose `arch-<name>` feature is enabled, this reads that arch's
//! DSL carrier from `dsl/<name>.rs.in` (the `#[forward]`/`#[vision_forward]`
//! item, written exactly as it used to appear under the attribute macro), runs
//! the shared pipeline (`scratchy_forward_compiler_macro::compile_in_dir`) over
//! it against that arch's own `configs/<name>/`, and writes the emitted modules
//! to `$OUT_DIR/<mod>.rs`, which `src/lib.rs` wraps in `pub mod <mod>` and
//! `include!`s. The pipeline lib is a build-dependency; the backend feature
//! (metal/cuda/…) is forwarded onto it so the emit matches what we then compile.
use proc_macro2::{Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use rayon::prelude::*;
use scratchy_forward_compiler_macro::{
    CompileMode, DEFAULT_DECODER_WORKLOADS, ForwardArgs, compile_in_dir, parse_carrier,
    render_tokens,
};
use std::collections::BTreeSet;
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
/// `crate` not followed by `::`) and absolute `::foo` paths untouched.
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
                    out.push(TokenTree::Ident(Ident::new(mod_name, Span::call_site())));
                    out.push(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                    out.push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
                }
            }
            other => out.push(other),
        }
    }
    out.into_iter().collect()
}

/// Lossy FORWARD-only feature/env name mangle: non-alphanumeric → `_`, upper.
fn envify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
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

/// Build `ForwardArgs` from the carrier's `#[forward(...)]`/`#[vision_forward(...)]`
/// attribute tokens, filling the default decoder ladder when `workloads` is
/// omitted (bare `#[forward]`). Vision carriers always pass `workloads`.
fn args_from_attr(attr: &syn::Attribute) -> ForwardArgs {
    let mut args: ForwardArgs = match &attr.meta {
        syn::Meta::List(list) => syn::parse2(list.tokens.clone()).expect("parse #[forward] args"),
        _ => syn::parse_str("").expect("empty #[forward] args"),
    };
    if args.workloads.is_empty() {
        args.workloads = DEFAULT_DECODER_WORKLOADS.to_vec();
    }
    args
}

/// `Some((mode, index))` if this item carries a `#[forward]`/`#[vision_forward]`
/// attribute, identifying the carrier and its compile mode.
fn carrier_attr(attrs: &[syn::Attribute]) -> Option<(CompileMode, usize)> {
    attrs.iter().enumerate().find_map(|(i, a)| {
        match a
            .path()
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .as_deref()
        {
            Some("forward") => Some((CompileMode::DECODER, i)),
            Some("vision_forward") => Some((CompileMode::VISION, i)),
            _ => None,
        }
    })
}

fn item_attrs_mut(item: &mut syn::Item) -> Option<&mut Vec<syn::Attribute>> {
    match item {
        syn::Item::Fn(f) => Some(&mut f.attrs),
        syn::Item::Mod(m) => Some(&mut m.attrs),
        _ => None,
    }
}

/// Write `$OUT_DIR/hf_registry.rs`, the shell-completion candidate list
/// `src/lib.rs` includes and `compiled_hf_registry()` returns.
///
/// Called on EVERY build, with `ids` empty when no registry was resolved (see
/// `hf_registry_build`) — the include in `src/lib.rs` carries no cfg
/// of its own, so the file must always exist. Rewrites only on a real change,
/// so an unchanged registry doesn't touch the mtime and retrigger downstream
/// crates.
fn write_hf_registry_file(out_dir: &Path, ids: &BTreeSet<String>) {
    let lits: Vec<String> = ids.iter().map(|s| format!("{s:?}")).collect();
    let rendered = format!(
        "/// Real HF org/repo ids matching an architecture (AND, when a quant\n\
         /// preset is enabled, a quant family) this binary compiled in support\n\
         /// for, resolved once against huggingface.co at BUILD time (see\n\
         /// `hf_registry_build.rs`). `scr model names` completes from this.\n\
         ///\n\
         /// Empty when this build resolved no registry — resolving it needs the\n\
         /// network, which an air-gapped build deliberately skips. `scr` then\n\
         /// completes no model names, rather than guessing.\n\
         pub static COMPILED_HF_REGISTRY: &[&str] = &[{}];\n",
        lits.join(", ")
    );
    let out = out_dir.join("hf_registry.rs");
    let unchanged = std::fs::read_to_string(&out).is_ok_and(|old| old == rendered);
    if !unchanged {
        std::fs::write(&out, rendered).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    }
}

/// Emit one arch: parse its DSL carrier (`dsl/<arch>.rs.in`), run the pipeline
/// against its configs (`configs/<arch>/`), write `$OUT_DIR/<mod>.rs`. Called
/// concurrently across arches — must not mutate global state (e.g. env). The
/// quant scoping (SCRATCHY_QUANTS) is set once in `main` before the fan-out.
fn emit_arch(dsl_path: &Path, configs_dir: &Path, out_dir: &Path, mod_name: &str) {
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
    let file =
        syn::parse_file(&dsl).unwrap_or_else(|e| panic!("parse {}: {e}", dsl_path.display()));

    // Find the carrier item + its forward/vision_forward attribute; strip the
    // attribute (parse_carrier expects the item without it, mirroring how an
    // attribute macro receives its annotated item).
    let mut carrier_item = None;
    let mut mode = CompileMode::DECODER;
    let mut args = None;
    for mut item in file.items {
        let found = item_attrs_mut(&mut item).and_then(|attrs| carrier_attr(attrs));
        if let Some((m, i)) = found {
            let attrs = item_attrs_mut(&mut item).unwrap();
            let attr = attrs.remove(i);
            mode = m;
            args = Some(args_from_attr(&attr));
            carrier_item = Some(item);
            break;
        }
    }
    let carrier_item = carrier_item.unwrap_or_else(|| {
        panic!(
            "{}: no #[forward]/#[vision_forward] carrier",
            dsl_path.display()
        )
    });
    let args = args.unwrap();

    let carrier = parse_carrier(carrier_item).unwrap_or_else(|e| panic!("carrier: {e}"));
    let tokens = compile_in_dir(&args, &carrier, mode, configs_dir)
        .unwrap_or_else(|e| panic!("forward pipeline ({mod_name}): {e}"));
    // Re-root the emit under `pub mod <mod>` (see reroot_crate).
    let tokens = reroot_crate(tokens, mod_name);

    let mut rendered = String::with_capacity(1 << 20);
    render_tokens(tokens, &mut rendered);
    let out = out_dir.join(format!("{mod_name}.rs"));
    // Write only when the emit changed, so a build-script rerun with identical
    // output doesn't bump the file mtime and force rustc to recompile the crate.
    let unchanged = std::fs::read_to_string(&out).is_ok_and(|old| old == rendered);
    if !unchanged {
        std::fs::write(&out, rendered).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    }
}

/// Arches whose emit only compiles under cuda (no metal surface). Under a pure
/// `--features metal` build we must NOT run their pipeline (it would emit metal
/// code for ops metal doesn't implement). Kept in sync with src/lib.rs gates.
const CUDA_ONLY: &[&str] = &[
    "deepseek-v2",
    "deepseek-v3",
    "deepseek-v3-flat",
    "gemma3-mm",
];
/// Arches that additionally support the spyre (KTIR) backend.
///
/// Must agree with the arches `Cargo.toml`'s `spyre` feature enables; an arch listed there but not here
/// is dropped by the `supported` gate below before its configs are read.
const SPYRE_CAPABLE: &[&str] = &["llama", "granite"];

/// Compile every vendored chat template for an enabled arch into `$OUT_DIR`,
/// returning the module source that wraps them.
///
/// ⛔ PER MODEL, NOT PER ARCH. Unlike `weights.json`, chat templates are not
/// shared within an arch: the single `llama` arch holds smollm2 (368 B),
/// tinyllama (410 B) and llama-3.2 (3.8 KB), and their templates are unrelated.
/// So this gates on the same per-stem feature `config.rs` uses
/// (`CARGO_FEATURE_<STEM>`), and a model with no `<stem>.chat.jinja` simply
/// contributes nothing — the serving side falls back to the interpreter.
///
/// ⛔ AN UNSUPPORTED CONSTRUCT PANICS THE BUILD. It does not warn and skip. The
/// error names template, line, column and construct, so the build states exactly
/// what to implement next. Skipping would silently ship the interpreter path and
/// make the compiled path look like it worked.
fn emit_chat_templates(configs_dir: &Path, out_dir: &Path, arch: &str) -> String {
    let mut modules = String::new();
    let Ok(rd) = std::fs::read_dir(configs_dir) else {
        return modules;
    };
    let mut jinjas: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".chat.jinja"))
        })
        .collect();
    jinjas.sort();

    for path in jinjas {
        let stem = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".chat.jinja"))
            .expect("checked above")
            .to_string();

        // Same per-stem gate config.rs applies to `<stem>.json`.
        let feat = format!("CARGO_FEATURE_{}", stem.to_uppercase().replace('-', "_"));
        if std::env::var(&feat).is_err() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());

        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let rel = format!("configs/{arch}/{stem}.chat.jinja");
        let code = scratchy_chat_template_compiler::compile::compile(&src, &rel, "render")
            .unwrap_or_else(|e| {
                panic!(
                    "chat template did not compile.\n  {e}\n\
                     This is a build error on purpose: implement the named construct in \
                     crates/compiler/serving/chat-template, or remove {rel} to fall back \
                     to the interpreter for this model."
                )
            });

        let mod_name = format!("chat_{}", stem.replace(['-', '.'], "_"));
        let file = out_dir.join(format!("{mod_name}.rs"));
        std::fs::write(&file, &code).unwrap_or_else(|e| panic!("write {}: {e}", file.display()));

        modules.push_str(&format!(
            "pub mod {mod_name} {{\n\
             \x20   include!(concat!(env!(\"OUT_DIR\"), \"/{mod_name}.rs\"));\n\
             }}\n\
             ::scratchy_forward_compiler::inventory::submit! {{\n\
             \x20   ::scratchy_forward_compiler::chat_registry::ChatTemplateRegistration {{\n\
             \x20       arch: {arch:?},\n\
             \x20       stem: {stem:?},\n\
             \x20       compiled: &{mod_name}::COMPILED,\n\
             \x20   }}\n\
             }}\n"
        ));
    }
    modules
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for name in [
        "SCRATCHY_GPU",
        // Spyre/KTIR bundle knobs read at emit time by `dump_wavefront_mega`:
        // they bake the prefix cap, decode row count, and prefill width into the
        // embedded KTIR bundle, so a change must re-run this build script (which
        // regenerates the bundle). Without this, cargo keeps the stale OUT_DIR
        // codegen and e.g. `KTIR_PREFIX_LEN=4096` silently reuses the old cap.
        "KTIR_PREFIX_LEN",
        "KTIR_M",
        "KTIR_PREFILL_LEN",
        // Version + base/instruct precision filter (see config.rs) — a
        // hand-set env var, not Cargo-feature-derived, so it needs explicit
        // tracking or changing it silently reuses the stale config set.
        "SCRATCHY_BUILD_FILTER",
        // Toggling `hf-completions` changes what the registry contains, and a
        // feature flip alone doesn't otherwise invalidate this script.
        "CARGO_FEATURE_HF_COMPLETIONS",
        // Load-bearing for the registry: gated repos (`meta-llama/*`) 401 and
        // get dropped when no token is set, so the SAME model scope resolves to
        // a different candidate list with and without it. Untracked, two
        // builders would silently disagree about what completes.
        "HF_TOKEN",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    let metal = std::env::var("CARGO_FEATURE_METAL").is_ok();
    let spyre = std::env::var("CARGO_FEATURE_SPYRE").is_ok();

    // Each arch is a `dsl/<arch>.rs.in` DSL carrier paired with a
    // `configs/<arch>/` dir; emit the ones whose `arch-<name>` feature is on AND
    // a backend they support is on. Sorted for deterministic output.
    let dsl_dir = manifest.join("dsl");
    println!("cargo:rerun-if-changed={}", dsl_dir.display());
    let mut dsl_files: Vec<PathBuf> = std::fs::read_dir(&dsl_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".rs.in"))
        })
        .collect();
    dsl_files.sort();

    // (dsl_path, mod_name, configs_dir) for each arch that passes the feature +
    // backend gates.
    let targets: Vec<(PathBuf, String, PathBuf)> = dsl_files
        .iter()
        .filter_map(|dsl_path| {
            let arch = dsl_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".rs.in"))?
                .to_string();
            // Feature gate: `arch-<name>` → CARGO_FEATURE_ARCH_<ENVIFY(name)>.
            if std::env::var(format!("CARGO_FEATURE_ARCH_{}", envify(&arch))).is_err() {
                return None;
            }
            // Backend gate: mirror the `pub mod <name>` cfg in src/lib.rs, so we
            // never run the pipeline for an arch whose mod won't be compiled.
            let supported = if CUDA_ONLY.contains(&arch.as_str()) {
                cuda
            } else if SPYRE_CAPABLE.contains(&arch.as_str()) {
                cuda || metal || spyre
            } else {
                cuda || metal
            };
            if !supported {
                return None;
            }
            let mod_name = arch.replace('-', "_");
            let configs_dir = manifest.join("configs").join(&arch);
            Some((dsl_path.clone(), mod_name, configs_dir))
        })
        .collect();

    // Set SCRATCHY_QUANTS ONCE to the global union of enabled presets. config.rs
    // intersects it with each arch's own quantizations.json, so the per-arch
    // result is identical to setting it per arch — but doing it once lets us emit
    // arches in parallel below without a set_var data race.
    let enabled_presets: std::collections::HashSet<&'static str> =
        scratchy_quantizations::enabled_presets()
            .into_iter()
            .collect();
    let quant_union: BTreeSet<String> = targets
        .iter()
        .flat_map(|(_, _, cfg)| enabled_quants(&cfg.join("quantizations.json"), &enabled_presets))
        .collect();
    let quants = quant_union.iter().cloned().collect::<Vec<_>>().join(",");
    // SAFETY: set once here, before the rayon fan-out reads it; no other thread
    // mutates the environment. `set_var` is unsafe on 2024.
    unsafe {
        std::env::set_var("SCRATCHY_QUANTS", &quants);
    }

    // Emit arches in parallel. compile_in_dir is itself rayon-parallel over
    // models, so this is nested (arch × model) work-stealing across all cores —
    // recovering the cross-crate parallelism the per-arch crate split used to get
    // from cargo, in one process.
    targets
        .par_iter()
        .for_each(|(dsl_path, mod_name, configs_dir)| {
            emit_arch(dsl_path, configs_dir, &out_dir, mod_name);
        });

    // Compiled chat templates. Serial and cheap (a few KB of Jinja per model)
    // next to the forward expansion, so it does not join the rayon fan-out.
    //
    // ALWAYS WRITTEN, even when empty — `src/lib.rs` includes this file with no
    // cfg of its own, matching how `hf_registry.rs` is handled. A missing file is
    // a build break; an empty one is a binary with no compiled templates, which
    // is a valid state (the serving side falls back to the interpreter).
    let mut chat_mods = String::new();
    for (dsl_path, _mod_name, configs_dir) in &targets {
        let arch = dsl_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".rs.in"))
            .expect("arch name");
        chat_mods.push_str(&emit_chat_templates(configs_dir, &out_dir, arch));
    }
    std::fs::write(out_dir.join("chat_templates.rs"), &chat_mods).expect("write chat_templates.rs");

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
    let hf_ids = BTreeSet::<String>::new();
    write_hf_registry_file(&out_dir, &hf_ids);

    // Every arch's configs/ dir has now been walked, so every
    // SCRATCHY_BUILD_FILTER tag that could ever match has had its chance.
    // A tag that matched nothing anywhere is almost certainly a typo, not
    // an intentional "compile zero models" request — fail loudly instead
    // of silently producing a binary with no models baked in.
    let unmatched = scratchy_forward_compiler_macro::unmatched_build_filter_tags();
    if !unmatched.is_empty() {
        panic!("SCRATCHY_BUILD_FILTER tag(s) matched no config in any arch: {unmatched:?}");
    }

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
