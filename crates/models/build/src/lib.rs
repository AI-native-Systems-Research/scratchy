// SPDX-License-Identifier: Apache-2.0
//! THE build-time driver for every model arch, shared by every target's models crate.
//!
//! For each arch whose `arch-<name>` feature the scope selects, this reads that arch's DSL
//! carrier from `dsl/<name>.py`, runs the shared pipeline
//! (`scratchy_forward_compiler_macro::compile_carrier`) over it against that arch's own
//! `configs/<name>/` with the calling target's emitter, and writes the emitted modules to
//! `$OUT_DIR/<mod>.rs` — plus `$OUT_DIR/modules.rs` declaring each as `pub mod <mod>`, and the
//! baked tokenizers of the selected models (`$OUT_DIR/baked_tokenizers.rs`).
//!
//! The scope (which models, heads and quant presets) is the model crate's (`scratchy-models`,
//! whose features name them): a target's models crate reads it from that crate's build metadata
//! ([`ScopeMetadata`]), so the selection is one set of features whichever target compiles it.

use proc_macro2::{Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use rayon::prelude::*;
use scratchy_forward_compiler_macro::target::TargetEmitter;
use scratchy_forward_compiler_macro::{
    CarrierError, Scope, compile_carrier, parse_python_file, render_tokens,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

/// The model scope `scratchy-models` publishes to a target's models crate (its build script's
/// `cargo::metadata`, read back from `DEP_SCRATCHY_MODELS_*`): the models' root (`dsl/` and
/// `configs/`), its enabled features as cargo names them, the head flags, and the enabled quant
/// presets.
pub struct ScopeMetadata {
    pub root: PathBuf,
    pub features: Vec<String>,
    pub mtp: bool,
    pub mtp_heads: bool,
    pub presets: Vec<String>,
}

impl ScopeMetadata {
    /// Print it as `scratchy-models`' build metadata (its `links` key makes it visible to the
    /// build scripts of the crates that depend on it).
    pub fn publish(&self) {
        println!("cargo::metadata=root={}", self.root.display());
        println!("cargo::metadata=features={}", self.features.join(","));
        println!("cargo::metadata=mtp={}", self.mtp);
        println!("cargo::metadata=mtp_heads={}", self.mtp_heads);
        println!("cargo::metadata=presets={}", self.presets.join(","));
    }

    /// Read it back in the build script of a crate depending on `scratchy-models`.
    pub fn from_dep_env() -> ScopeMetadata {
        let var = |k: &str| {
            let name = format!("DEP_SCRATCHY_MODELS_{k}");
            println!("cargo:rerun-if-env-changed={name}");
            std::env::var(&name).unwrap_or_else(|_| {
                panic!("{name} unset: this build script's crate must depend on scratchy-models")
            })
        };
        let list = |s: String| -> Vec<String> {
            s.split(',')
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        };
        ScopeMetadata {
            root: PathBuf::from(var("ROOT")),
            features: list(var("FEATURES")),
            mtp: var("MTP") == "true",
            mtp_heads: var("MTP_HEADS") == "true",
            presets: list(var("PRESETS")),
        }
    }

    /// The scope it states.
    pub fn scope(&self) -> Scope {
        Scope::from_feature_flags(self.features.iter().cloned(), self.mtp, self.mtp_heads)
    }

    /// The model stems this scope selects by name (`<stem>` features), for a crate exposing them
    /// as `cfg(model = "<stem>")`.
    pub fn stem_cfgs(&self) {
        println!("cargo::rustc-check-cfg=cfg(model, values(any()))");
        let configs = self.root.join("configs");
        let Ok(arches) = std::fs::read_dir(&configs) else {
            return;
        };
        let on: HashSet<&str> = self.features.iter().map(String::as_str).collect();
        for arch in arches.flatten() {
            for config in std::fs::read_dir(arch.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                let path = config.path();
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                if path.extension().is_some_and(|e| e == "json")
                    && on.contains(
                        scratchy_forward_compiler_macro::config::feature_env_name(stem).as_str(),
                    )
                {
                    println!("cargo::rustc-cfg=model=\"{stem}\"");
                }
            }
        }
    }
}

/// One models build: the target, the scope, where its output goes.
pub struct ModelsBuild<'a> {
    pub metadata: &'a ScopeMetadata,
    pub out_dir: &'a Path,
    pub emitter: &'a dyn TargetEmitter,
    /// Where a bake writes its device inputs when the build asked for them; `None` writes none.
    pub dump_dir: Option<&'a Path>,
    /// Read off each arch's emitted tokens (before rendering) — what a target's own build
    /// tables need of them.
    pub inspect: &'a (dyn Fn(&TokenStream) -> Vec<String> + Sync),
}

/// What a models build emitted.
pub struct Built {
    /// The emitted arch modules (`<mod>`), sorted.
    pub arches: Vec<String>,
    /// `inspect`'s findings, each as `<mod>::<finding>`, sorted.
    pub inspected: Vec<String>,
}

/// The emitted code roots its own items at `crate::` (each arch used to be its
/// own crate, so `crate::__gpu` / `crate::<Model>` meant "this arch's root").
/// In the consolidated crate every arch lives under `pub mod <mod>`, so rewrite
/// every path-root `crate ::` → `crate :: <mod> ::`. Leaves `pub(crate)` (a
/// `crate` not followed by `::`), absolute `::foo` paths and `crate::__metal_bake`
/// — the one module every arch shares, at the crate root — untouched.
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

/// The quant presets enabled for one arch: its `quantizations.json` ∩ the enabled presets.
fn enabled_quants(qpath: &Path, enabled_presets: &HashSet<&str>) -> Vec<String> {
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

/// Every selected arch: `(dsl path, mod name, configs dir)`, sorted, for the scope.
pub fn selected_arches(root: &Path, scope: &Scope) -> Vec<(PathBuf, String, PathBuf)> {
    let dsl_dir = root.join("dsl");
    println!("cargo:rerun-if-changed={}", dsl_dir.display());
    let mut dsl_files: Vec<PathBuf> = std::fs::read_dir(&dsl_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dsl_dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".py"))
        })
        .collect();
    dsl_files.sort();
    dsl_files
        .iter()
        .filter_map(|dsl_path| {
            let arch = dsl_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".py"))?
                .to_string();
            if !scope.arch_compiled(&arch) {
                return None;
            }
            let mod_name = arch.replace('-', "_");
            let configs_dir = root.join("configs").join(&arch);
            Some((dsl_path.clone(), mod_name, configs_dir))
        })
        .collect()
}

/// The union of the enabled presets across the selected arches' own `quantizations.json`s.
pub fn quant_union(arches: &[(PathBuf, String, PathBuf)], presets: &[String]) -> BTreeSet<String> {
    let presets: HashSet<&str> = presets.iter().map(String::as_str).collect();
    arches
        .iter()
        .flat_map(|(_, _, cfg)| enabled_quants(&cfg.join("quantizations.json"), &presets))
        .collect()
}

/// Emit one arch: parse its DSL carrier, run the pipeline against its configs, write
/// `$OUT_DIR/<mod>.rs`. Called concurrently across arches — must not mutate global state; the
/// scope is the one `&Scope` value stated before the fan-out, borrowed by every arch.
fn emit_arch(
    build: &ModelsBuild<'_>,
    dsl_path: &Path,
    configs_dir: &Path,
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
    let tokens = compile_carrier(carrier, configs_dir, build.dump_dir, scope, build.emitter)
        .unwrap_or_else(|e| match e {
            CarrierError::Config { stem, source } => {
                panic!("forward pipeline ({mod_name}/{stem}): {source}")
            }
            CarrierError::Arch(source) => panic!("forward pipeline ({mod_name}): {source}"),
        });
    // Re-root the emit under `pub mod <mod>` (see reroot_crate).
    let tokens = reroot_crate(tokens, mod_name);
    let found = ((build.inspect)(&tokens).into_iter())
        .map(|f| format!("{mod_name}::{f}"))
        .collect();
    let mut rendered = String::with_capacity(1 << 20);
    render_tokens(tokens, &mut rendered);
    write_if_changed(&build.out_dir.join(format!("{mod_name}.rs")), &rendered);
    found
}

/// Run the build: every selected arch through the pipeline with the build's emitter, then the
/// selected models' tokenizers. Panics when the scope selects no model — a build naming zero
/// `<stem>`/`<arch>`/`all` features would otherwise silently link a binary with no models.
pub fn build(build: &ModelsBuild<'_>) -> Built {
    let root = &build.metadata.root;
    // The target the emitted code is compiled for, as the cfg it gates its target's own items on.
    // Its values are open: any target names itself.
    println!("cargo::rustc-check-cfg=cfg(scratchy_target, values(any()))");
    println!(
        "cargo::rustc-cfg=scratchy_target=\"{}\"",
        build.emitter.name()
    );
    println!("cargo:rerun-if-env-changed=HF_TOKEN");
    // A head that prefills at its target's chunk width must have every rung its target has, or a
    // chunk at a rung it lacks panics at runtime: refuse the build instead, before any expansion.
    // Every head is checked, selected or not, so every arch.json the check reads is watched (dsl/
    // already is, file by file), and an incremental build agrees with a clean one.
    for e in std::fs::read_dir(root.join("configs")).unwrap().flatten() {
        let arch_json = e.path().join("arch.json");
        if arch_json.is_file() {
            println!("cargo:rerun-if-changed={}", arch_json.display());
        }
    }
    scratchy_forward_compiler_macro::check_head_ladders(root)
        .unwrap_or_else(|e| panic!("head ladder: {e}"));

    let scope = build.metadata.scope();
    let arches = selected_arches(root, &scope);
    // The global union of enabled presets, computed once and stated IN the scope — config.rs
    // intersects it with each arch's own quantizations.json. A value, not an env var: the scope
    // is threaded through the rayon fan-out below.
    let scope = scope.with_quants(quant_union(&arches, &build.metadata.presets));

    // Emit arches in parallel. compile_carrier is itself rayon-parallel over models, so this is
    // nested (arch × model) work-stealing across all cores.
    let mut inspected: Vec<String> = arches
        .par_iter()
        .flat_map(|(dsl_path, mod_name, configs_dir)| {
            emit_arch(build, dsl_path, configs_dir, mod_name, &scope)
        })
        .collect();
    inspected.sort();
    let mut mods: Vec<String> = arches.iter().map(|(_, m, _)| m.clone()).collect();
    mods.sort();
    let modules: String = (mods.iter())
        .map(|m| {
            format!("pub mod {m} {{\n    include!(concat!(env!(\"OUT_DIR\"), \"/{m}.rs\"));\n}}\n")
        })
        .collect();
    write_if_changed(&build.out_dir.join("modules.rs"), &modules);
    bake_tokenizers(build.out_dir);

    if scratchy_forward_compiler_macro::total_models_emitted() == 0 {
        panic!(
            "no models selected — scratchy-models has no default model features. \
             Enable at least one, e.g. `model/granite-3.1-2b-instruct` \
             (or `model/<arch>` / `model/all` for a wider scope)."
        );
    }
    Built {
        arches: mods,
        inspected,
    }
}

/// Write `text` to `out` only when it changed, so a build-script rerun with identical output
/// doesn't bump the file's mtime and force rustc to recompile the crate.
pub fn write_if_changed(out: &Path, text: &str) {
    let unchanged = std::fs::read_to_string(out).is_ok_and(|old| old == text);
    if !unchanged {
        std::fs::write(out, text).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    }
}

fn write_if_changed_bytes(out: &Path, bytes: &[u8]) {
    if std::fs::read(out).ok().as_deref() != Some(bytes) {
        std::fs::write(out, bytes).expect("write a baked tokenizer");
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
