// SPDX-License-Identifier: Apache-2.0
//! Build script for `scratchy-subtile`.
//!
//! Reads every model `config.json` in the workspace and writes the
//! geometries they declare into `$OUT_DIR/config_geometry.rs` — see
//! [`emit_config_geometry`] and `src/model_geometry.rs`.

fn main() {
    emit_config_geometry();
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE MODEL GEOMETRIES, READ OUT OF `config.json` AND WRITTEN AS LITERALS.
//
//  scratchy is a per-model compiler: the `#[forward]` proc-macro parses a model's `config.json` at
//  expansion time and the SuperDSC lowering runs INSIDE that expansion, so the head counts and the
//  head dim are bake-time constants of the bundle. The emitter wants them as CONST GENERICS (a head
//  dim parameterises the device layout, and a branch on a const is reviewable where a branch on a
//  value is not), and a const generic is monomorphised when the emitter crate itself is compiled —
//  which happens BEFORE any proc-macro expansion. So the instantiations cannot be emitted by the
//  macro: they have to exist one compilation earlier, read from the same files the macro reads.
//
//  That is what this does. It parses every model config in the workspace, applies the same implicit
//  derivations the macro's config loader applies (`head_dim` from the MLA summands or from
//  `hidden_size / num_attention_heads`, `num_key_value_heads` defaulting to `num_attention_heads`),
//  and writes the resulting geometries out as literal tokens. Dropping a new `config.json` into
//  `crates/models/arch/<arch>/configs/` re-runs this script and yields a new instantiation, with no
//  edit to any file a human wrote.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One model's attention geometry as its config declares it: query heads, kv heads, head dim.
type Geometry = (u64, u64, u64);

/// Write `$OUT_DIR/config_geometry.rs` — the literal geometries `src/model_geometry.rs` turns into
/// const-generic instantiations.
fn emit_config_geometry() {
    // Naming any `rerun-if-changed` turns off cargo's default "re-run on any package change", so
    // this script's own source has to be declared too.
    println!("cargo:rerun-if-changed=build.rs");

    let arch_root = arch_root();
    // The per-arch config directories under the arch crate's single `configs/` root, tracked so that
    // DROPPING A NEW CONFIG IN re-runs this script: cargo walks a directory dependency, so a file
    // that did not exist at the last run still invalidates it.
    let configs_root = arch_root.join("configs");
    let mut config_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&configs_root)
        .unwrap_or_else(|e| {
            panic!(
                "scratchy-subtile: cannot read {}: {e}",
                configs_root.display()
            )
        })
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    config_dirs.sort();
    for d in &config_dirs {
        println!("cargo:rerun-if-changed={}", d.display());
    }

    let mut stems: Vec<String> = Vec::new();
    let mut geometries: Vec<Geometry> = Vec::new();
    /// Every checked-in config as a WHOLE model — the seven constants a lowering specialises on:
    /// `(nqh, nkvh, hd, hidden, layers, ffn, vocab)`.
    type WholeModel = (u64, u64, u64, u64, u64, u64, u64);
    let mut models: Vec<WholeModel> = Vec::new();
    let mut head_dims: Vec<u64> = Vec::new();
    for dir in &config_dirs {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("scratchy-subtile: cannot read {}: {e}", dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        files.sort();
        for path in files {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // `weights.json` is the weight manifest, `quantizations.json` the preset list, and
            // `*.overrides.json` a drift overlay — none of the three is a model.
            if matches!(stem, "weights" | "quantizations") || stem.ends_with(".overrides") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("scratchy-subtile: cannot read {}: {e}", path.display())
            });
            // A malformed config is the config loader's error to report, with its own path and
            // span. Skipping it here means its geometry is simply absent, and the emitter's door
            // then refuses that model by name — never a wrong instantiation.
            let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let bounds = geometry_bounds(&json);
            let mut used = false;
            if let (Some(&nqh), Some(&nkvh), Some(&hd)) = (
                bounds.get("num_attention_heads"),
                bounds.get("num_key_value_heads"),
                bounds.get("head_dim"),
            ) {
                // A pair whose kv-head count does not divide the query-head count has no GQA
                // grouping, so `AttnGeometry` cannot be named at it — its `GQA` const refuses. Such
                // a config yields no arm; baking it is then a loud refusal at the door, rather than
                // a const-evaluation failure that would stop this crate compiling at all.
                if nqh > 0 && nkvh > 0 && hd > 0 && nqh % nkvh == 0 {
                    geometries.push((nqh, nkvh, hd));
                    used = true;
                }
            }
            // ⭐⭐ THE WHOLE MODEL, not just its attention geometry. A lowering specialises on the
            // FFN width and the hidden size as much as on the head counts — whether a row fits the
            // scratchpad, whether it is a whole number of sticks — so those have to arrive as
            // constants too, or the flags derived from them fold to nothing.
            //
            // ⛔ ALL SEVEN OR NONE. A model missing any one of them yields no arm at all, so the
            // door refuses it BY NAME rather than instantiating a lowering against a default that
            // was never in anyone's config.
            if let (
                Some(&nqh),
                Some(&nkvh),
                Some(&hd),
                Some(&hidden),
                Some(&layers),
                Some(&ffn),
                Some(&vocab),
            ) = (
                bounds.get("num_attention_heads"),
                bounds.get("num_key_value_heads"),
                bounds.get("head_dim"),
                bounds.get("hidden_size"),
                bounds.get("num_hidden_layers"),
                bounds.get("intermediate_size"),
                bounds.get("vocab_size"),
            ) && nqh > 0
                && nkvh > 0
                && hd > 0
                && hidden > 0
                && layers > 0
                && ffn > 0
                && vocab > 0
                && nqh % nkvh == 0
            {
                models.push((nqh, nkvh, hd, hidden, layers, ffn, vocab));
                used = true;
            }
            // Every head dim a rotary op of this model can carry. `head_dim` is the decoder's;
            // `global_head_dim` is the second head dim of a per-layer-class model (Gemma-4's global
            // layers), and `qk_rope_head_dim` is MLA's rope width.
            for key in ["head_dim", "global_head_dim", "qk_rope_head_dim"] {
                if let Some(&v) = bounds.get(key)
                    && v > 0
                {
                    head_dims.push(v);
                    used = true;
                }
            }
            if used {
                stems.push(stem.to_string());
            }
        }
    }
    models.sort_unstable();
    models.dedup();
    geometries.sort_unstable();
    geometries.dedup();
    head_dims.sort_unstable();
    head_dims.dedup();
    stems.sort();
    stems.dedup();

    let mut out = String::new();
    out.push_str("// @generated by scratchy-subtile/build.rs — DO NOT EDIT.\n");
    out.push_str("// Every literal below was parsed out of a model config.json.\n\n");
    out.push_str("/// The model stems whose `config.json` produced the geometries below.\n");
    out.push_str("pub const GEOMETRY_SOURCE_STEMS: &[&str] = &[\n");
    for s in &stems {
        out.push_str(&format!("    {s:?},\n"));
    }
    out.push_str("];\n\n");

    // The attention-geometry door has exactly one consumer, `with_config_attn_geometry`, and it is
    // `#[cfg(feature = "superdsc")]`. Emitting the macro on a metal/cuda build leaves it dead, which
    // `-D warnings` rejects as `unused_macros`, so generate it under the same condition its caller
    // compiles under. The head-dim door below is unconditional because its consumer is.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SUPERDSC");
    if std::env::var_os("CARGO_FEATURE_SUPERDSC").is_some() {
        out.push_str("macro_rules! for_each_config_attn_geometry {\n");
        out.push_str("    ($emit:ident) => {\n        $emit! {\n");
        for (nqh, nkvh, hd) in &geometries {
            out.push_str(&format!("            ({nqh}, {nkvh}, {hd}),\n"));
        }
        out.push_str("        }\n    };\n}\n\n");

        // The WHOLE-model door, same mechanism and the same gate. Its consumer specialises on the
        // FFN width and the hidden size as well as the head counts, so it takes all seven.
        out.push_str("macro_rules! for_each_config_model {\n");
        out.push_str("    ($emit:ident) => {\n        $emit! {\n");
        for (nqh, nkvh, hd, hidden, layers, ffn, vocab) in &models {
            out.push_str(&format!(
                "            ({nqh}, {nkvh}, {hd}, {hidden}, {layers}, {ffn}, {vocab}),\n"
            ));
        }
        out.push_str("        }\n    };\n}\n\n");
    }

    out.push_str("macro_rules! for_each_config_head_dim {\n");
    out.push_str("    ($emit:ident) => {\n        $emit! {\n");
    for hd in &head_dims {
        out.push_str(&format!("            {hd},\n"));
    }
    out.push_str("        }\n    };\n}\n");

    let dst = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"))
        .join("config_geometry.rs");
    std::fs::write(&dst, out)
        .unwrap_or_else(|e| panic!("scratchy-subtile: cannot write {}: {e}", dst.display()));
}

/// `crates/models/arch`, found by walking up from this crate. The emitter is compiled inside the
/// workspace that owns the configs; if that directory is gone there is no geometry to read and no
/// bundle this crate could correctly bake, so this is a hard build failure rather than an empty set
/// that would surface much later as a bake refusal.
fn arch_root() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"),
    );
    let mut dir = manifest.as_path();
    loop {
        let candidate = dir.join("crates/models/arch");
        if candidate.is_dir() {
            return candidate;
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => panic!(
                "scratchy-subtile: no `crates/models/arch` above {} — the emitter's const-generic \
                 geometries are read from the model configs there",
                manifest.display()
            ),
        }
    }
}

/// The geometry bounds a config declares, after the same implicit derivations the macro's config
/// loader performs (`config.rs::derive_implicit_bounds`). Nested `text_config` / `rope_parameters`
/// blocks are hoisted first, exactly as `normalize_hf_config` hoists them, so a multimodal
/// wrapper's decoder geometry is found where the flat harvest looks for it.
fn geometry_bounds(json: &serde_json::Value) -> std::collections::BTreeMap<String, u64> {
    let mut flat: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let mut absorb = |v: &serde_json::Value, overwrite: bool| {
        let Some(obj) = v.as_object() else { return };
        for (k, val) in obj {
            if let Some(n) = val.as_u64()
                && (overwrite || !flat.contains_key(k))
            {
                flat.insert(k.clone(), n);
            }
        }
    };
    absorb(json, true);
    for nested in ["text_config", "rope_parameters"] {
        if let Some(sub) = json.get(nested) {
            absorb(sub, false);
        }
    }
    // MLA (DeepSeek family): the per-head Q/K width is the sum of the two summands, and that rule
    // must win over the generic hidden/heads one.
    if !flat.contains_key("head_dim")
        && let (Some(&nope), Some(&rope)) =
            (flat.get("qk_nope_head_dim"), flat.get("qk_rope_head_dim"))
    {
        flat.insert("head_dim".into(), nope + rope);
    }
    if !flat.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (flat.get("hidden_size"), flat.get("num_attention_heads"))
        && heads != 0
        && hidden % heads == 0
    {
        flat.insert("head_dim".into(), hidden / heads);
    }
    if !flat.contains_key("num_key_value_heads")
        && let Some(&heads) = flat.get("num_attention_heads")
    {
        flat.insert("num_key_value_heads".into(), heads);
    }
    flat
}
