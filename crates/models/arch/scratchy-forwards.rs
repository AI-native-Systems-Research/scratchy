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
    let quants = quant_union.into_iter().collect::<Vec<_>>().join(",");
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
