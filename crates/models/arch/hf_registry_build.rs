// SPDX-License-Identifier: Apache-2.0
//! Build-time HuggingFace Hub query: for every arch this build compiles in
//! (its scratchy arch identifier IS the Hub's own architecture-family tag
//! for the great majority of arches — `llama`, `granite`, `qwen2`, `qwen3`,
//! `mixtral`, `gemma3`, `deepseek_v3` all verified directly against
//! `huggingface.co/api/models?filter=<tag>`), AND-filtered by whichever
//! quantization family this build's enabled quant presets map to (`mlx`,
//! `gptq`, `awq`, `gguf`, …; no filter added for a dense/bf16 build), ask
//! huggingface.co which repos match, and bake the union into
//! `$OUT_DIR/hf_registry.rs` as a `&'static [&'static str]`.
//!
//! This exists so `scr model names` (shell tab-completion candidates) never
//! makes a runtime network call and never reads the local hf-hub cache — the
//! candidate list is resolved ONCE, at build time, scoped to exactly the
//! `-Fmodel/<stem>` / `-Fquant/<preset>` scope this binary was built with,
//! the same "everything is a compile-time constant" invariant every other
//! `#[forward]` artifact follows.
//!
//! A handful of compound arch identifiers (e.g. `gemma4-moe`) don't match
//! the Hub's own tag verbatim (the Hub tags the whole family `gemma4`, MoE
//! and dense alike — `gemma4-moe/gemma-4-26b-a4b-it.json` and
//! `gemma4/gemma-4-31b-it.json` even share the identical HF `architectures`
//! class, `Gemma4ForConditionalGeneration`; the Hub has no facet for this
//! split at all). So after the tag query narrows to a family, each
//! candidate's own `config.json` is fetched and checked against the SHAPE
//! (`hidden_size` + `num_hidden_layers`) of the model(s) actually compiled
//! for this arch — the same numbers that make the compiled forward tape
//! what it is. A candidate whose shape doesn't match any compiled variant
//! is dropped, so `gemma4-moe` completes only the MoE checkpoints, not
//! gemma4's dense siblings.
//!
//! The Hub query result is cached under `<target-dir>/hf-registry-cache/` — a
//! build-time artifact shared by whoever builds this checkout, not a per-user
//! runtime completion cache — with a TTL, so repeated builds don't re-hit the
//! network. huggingface.co being unreachable must never fail the build: a miss
//! falls back to the last cached result, or an empty list.
//!
//! # This module only exists under `hf-completions`
//!
//! Everything here is network I/O, so the whole module is gated (see
//! `scratchy-forwards.rs`) and its only dependency, `ureq`, is an optional
//! build-dependency. Graceful degradation is not enough on its own: verifying
//! candidates costs one `config.json` GET each, ~160 for a single small arch and
//! hundreds per arch at `all` scope, which an air-gapped build would pay as
//! timeouts before falling back to an empty list anyway. Off by default, that
//! cost is never paid and the registry is simply emitted empty.

use rayon::prelude::*;
use scratchy_forward_compiler_macro::{QuantMethod, QuantizationConfig};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CACHE_TTL: Duration = Duration::from_secs(7 * 24 * 3600);

/// Declared preset-family → Hub library/quant tag correspondence (data, not
/// logic — same status as `quantizations.json`'s own preset tables).
/// Presets not listed here fall back to their name's leading `-`-delimited
/// token, which already matches for most families (`gptq-sym` → `gptq`,
/// `awq-gemm` → `awq`).
const PRESET_TAG_OVERRIDES: &[(&str, &str)] = &[
    ("ggml", "gguf"),
    ("bnb-nf4-dq", "bitsandbytes"),
    ("ct-int4-sym", "compressed-tensors"),
    ("gptq-sym-desc_act", "gptq"),
];

fn preset_hub_tag(preset: &str) -> &str {
    for (name, tag) in PRESET_TAG_OVERRIDES {
        if *name == preset {
            return tag;
        }
    }
    preset.split('-').next().unwrap_or(preset)
}

/// One compiled model's discriminating shape — enough to tell apart
/// siblings the Hub's own tags/architectures class conflate (MoE vs dense
/// gemma4, e.g.). Not a general fingerprint (that's the runtime
/// `HfFingerprint`'s job); just the two fields universally present and
/// almost always sufficient to separate genuinely different compiled
/// variants of the same Hub-tagged family.
// (hidden_size, num_hidden_layers, num_experts). `num_experts` is `None` for
// dense arches (absent from their config — comparable across dense
// candidates as long as neither side has it) and `Some` for MoE arches,
// where it's load-bearing: the expert-routing/dispatch tape is sized to the
// compiled expert count at macro-expansion time, not runtime-flexible, so an
// expert-pruned checkpoint (e.g. the REAP technique, which ships a REAL
// checkpoint with FEWER experts than its base — verified: a
// `gemma-4-26B-A4B` REAP repo declares `num_experts: 64` against the
// compiled model's `128`) is a genuinely different, unloadable shape that
// `hidden_size`/`num_hidden_layers` alone can't catch.
type Shape = (i64, i64, Option<i64>);

/// For every `((arch_name, configs_dir), enabled quant preset set)` this
/// build compiles, resolve Hub-matching repo ids (cache-first) and verify each
/// against this arch's own compiled shape(s) and quant preset(s).
///
/// Returns the deduped/sorted union. The caller
/// (`scratchy-forwards.rs::write_hf_registry_file`) renders it — writing happens
/// on every build, resolving only under `hf-completions`, so both paths share
/// one renderer and `$OUT_DIR/hf_registry.rs` always exists.
pub fn resolve_hf_registry(
    targets: &[(String, PathBuf)],
    enabled_quant_presets: &BTreeSet<String>,
    repo_root: &Path,
) -> BTreeSet<String> {
    let quant_tags: BTreeSet<&str> = enabled_quant_presets
        .iter()
        .map(|p| preset_hub_tag(p))
        .collect();
    let allowed_methods = compiled_quant_methods(enabled_quant_presets, repo_root);

    let cache_dir = cache_dir();
    let _ = std::fs::create_dir_all(&cache_dir);
    let config_cache_dir = cache_dir.join("configs");
    let _ = std::fs::create_dir_all(&config_cache_dir);

    let mut ids: BTreeSet<String> = BTreeSet::new();
    for (arch, configs_dir) in targets {
        // One query per DISTINCT `model_type` among this arch's compiled
        // configs, not one per arch directory: the Hub tags by `model_type`, and
        // an arch dir can hold several (see `compiled_variants`).
        for (model_type, shapes) in compiled_variants(arch, configs_dir) {
            let tag = hub_tag_for(&model_type, arch, &cache_dir);

            let candidates: Vec<String> = if quant_tags.is_empty() {
                resolve(&[&tag], &cache_dir)
            } else {
                let mut v = Vec::new();
                for quant_tag in &quant_tags {
                    v.extend(resolve(&[&tag, quant_tag], &cache_dir));
                }
                v
            };

            // Verification is one HTTPS GET per candidate (up to `limit=200` of
            // them), so this is the whole cost of the module and it is pure I/O
            // wait. Fanned out on the rayon pool the arch emit already uses;
            // each candidate touches its own cache file, so there is nothing to
            // serialize. Cache hits make repeat builds trivially fast either
            // way.
            ids.par_extend(
                candidates.into_par_iter().filter(|id| {
                    candidate_matches(id, &shapes, &allowed_methods, &config_cache_dir)
                }),
            );
        }
    }
    ids
}

/// The Hub filter tag for one compiled group.
///
/// `model_type` is authoritative and free (it comes out of a local
/// `config.json`), so it is tried first and costs no extra request. It is also
/// the only thing that finds several arches at all: `configs/commandr/` declares
/// `cohere`, `configs/deepseek-v3-flat/` declares `deepseek_v3`, and
/// `configs/gemma4/gemma-4-12b-it.json` declares `gemma4_unified` — none of
/// which the directory name would ever produce.
///
/// The structural rewrites of the arch NAME are the fallback, and they are NOT
/// dead code: five checked-in configs declare no `model_type`, so
/// `compiled_variants` keys them by directory name, and `qwen2-5-vl` matches
/// zero repos while `qwen2_5_vl` matches plenty. Without this the `-`→`_`
/// rewrite that recovers them, that config's repos would silently vanish from
/// completion. The fallback costs up to three counting queries, so it only runs
/// when the free answer came back empty.
fn hub_tag_for(model_type: &str, arch: &str, cache_dir: &Path) -> String {
    if !resolve(&[model_type], cache_dir).is_empty() {
        return model_type.to_string();
    }
    resolve_arch_tag(arch, cache_dir)
}

/// `<target-dir>/hf-registry-cache` — derived from `OUT_DIR`
/// (`<target-dir>/<profile>/build/<pkg>-<hash>/out`, so four levels up), NOT by
/// climbing to the repo root and appending `target`: `CARGO_TARGET_DIR` and
/// `--target-dir` move the build directory out of the checkout entirely, and a
/// cache written to a stale guess is a cache nobody ever reads.
fn cache_dir() -> PathBuf {
    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set for build scripts"));
    out_dir
        .ancestors()
        .nth(4)
        .expect("OUT_DIR is <target>/<profile>/build/<pkg>/out")
        .join("hf-registry-cache")
}

/// Cargo's own feature-name → env-var transform (mirrors
/// `crates/compiler/macros/src/config.rs`'s `feature_env_var`).
fn feature_enabled(feature: &str) -> bool {
    std::env::var(format!(
        "CARGO_FEATURE_{}",
        feature.to_uppercase().replace('-', "_")
    ))
    .is_ok()
}

/// Whether this arch's `<stem>`-scoped config is actually part of THIS
/// build — a bare `<stem>` feature, this arch's own bare `<arch>` feature
/// (widens to every config in it), or the crate-wide `all`. Doesn't
/// reproduce the vision-arch stem/text-only collision table in
/// `config.rs` (a handful of shared-checkpoint vision stems) — those may
/// occasionally over/under-include here; every plain decoder arch (the
/// large majority) resolves exactly.
fn stem_compiled(stem: &str, arch: &str) -> bool {
    feature_enabled(stem) || feature_enabled(arch) || feature_enabled("all")
}

/// This arch's compiled configs grouped as `model_type -> {shape}` — scoped to
/// configs whose `<stem>` feature is actually part of THIS build, not every
/// config checked into the arch's directory (an arch dir holds every
/// size/variant; only some are compiled). Skips
/// `quantizations.json`/`weights.json` (not model configs).
///
/// Keyed by `model_type` because that field IS the Hub's own architecture-family
/// tag, and one scratchy arch directory can hold configs that declare different
/// ones: `configs/gemma4/` has `gemma-4-31b-it.json` (`model_type: gemma4`)
/// alongside `gemma-4-12b-it.json` (`model_type: gemma4_unified`), and the Hub
/// tags matching repos accordingly — `mlx-community/gemma-4-12B-it-4bit` carries
/// `gemma4_unified` and no `gemma4` tag at all. A single tag per arch therefore
/// cannot see every compiled variant's repos.
///
/// Grouping rather than unioning also keeps shapes from cross-matching: a
/// `gemma4_unified` candidate is checked only against `gemma4_unified` shapes,
/// never against the 31B's.
fn compiled_variants(arch: &str, configs_dir: &Path) -> BTreeMap<String, BTreeSet<Shape>> {
    let mut variants: BTreeMap<String, BTreeSet<Shape>> = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(configs_dir) else {
        return variants;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if matches!(file_name, "quantizations.json" | "weights.json") {
            continue;
        }
        let stem = file_name
            .strip_suffix(".overrides.json")
            .or_else(|| file_name.strip_suffix(".json"))
            .unwrap_or(file_name);
        if !stem_compiled(stem, arch) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(json): Result<serde_json::Value, _> = serde_json::from_str(&text) else {
            continue;
        };
        let Some(shape) = shape_of(&json) else {
            continue;
        };
        // `model_type` is the Hub's own tag vocabulary, so it is the key. Five
        // checked-in configs omit the field entirely (one each under `gemma3`,
        // `llama`, `qwen2`, `qwen2-5-vl`, `qwen2-vl`); those fall back to the
        // arch directory name, which is a GUESS at a tag rather than a declared
        // one — `qwen2-5-vl` matches no repo at all, and `hub_tag_for` is what
        // recovers it. Grouping them under the directory name is still correct
        // here: it keeps their shapes together and distinct from the shapes of
        // siblings that DO declare a type.
        let group = json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or(arch)
            .to_string();
        variants.entry(group).or_default().insert(shape);
    }
    variants
}

/// Whether a Hub candidate repo actually matches this build: its own
/// `config.json` shape (`hidden_size`/`num_hidden_layers`) must match one of
/// this arch's compiled shapes (skipped if extraction found no compiled
/// shape at all — an extraction gap, not grounds to reject everything), AND
/// its own `quantization_config` must match one of the compiled quant presets
/// (by [`QuantMethod::same_storage_format`]) — a dense build requires the
/// candidate to have NO `quantization_config` at all. Fetches (cache-first,
/// same TTL as the search queries)
/// `https://huggingface.co/<id>/raw/main/config.json` once for both checks.
/// Network/parse failure degrades to `false` (excluded) — a candidate we
/// can't verify is not one we can promise `scr chat`/`scr serve` will load.
fn candidate_matches(
    id: &str,
    compiled_shapes: &BTreeSet<Shape>,
    allowed_methods: &[QuantMethod],
    config_cache_dir: &Path,
) -> bool {
    let cache_path = config_cache_dir.join(format!("{}.json", sanitize(id)));

    let cached_text = std::fs::read_to_string(&cache_path).ok();
    let is_fresh = cached_text
        .as_deref()
        .and_then(|_| std::fs::metadata(&cache_path).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.elapsed().ok())
        .map(|age| age < CACHE_TTL)
        .unwrap_or(false);

    let text = if is_fresh {
        cached_text
    } else {
        match fetch_config(id) {
            Some(t) => {
                let _ = std::fs::write(&cache_path, &t);
                Some(t)
            }
            None => cached_text,
        }
    };

    let Some(text) = text else { return false };
    let Ok(json): Result<serde_json::Value, _> = serde_json::from_str(&text) else {
        return false;
    };

    if !compiled_shapes.is_empty() && !shape_of(&json).is_some_and(|s| compiled_shapes.contains(&s))
    {
        return false;
    }

    let candidate_method = QuantizationConfig::parse(&json)
        .ok()
        .flatten()
        .map(|c| c.method);
    match (candidate_method, allowed_methods.is_empty()) {
        // Dense build (no quant preset enabled): only a candidate with NO
        // quantization_config at all is something this build can load.
        (None, true) => true,
        (None, false) => false,
        (Some(_), true) => false,
        (Some(m), false) => allowed_methods.iter().any(|a| m.same_storage_format(a)),
    }
}

/// Loads each enabled quant preset's own declared `quantization_config`
/// (`crates/models/quantization/presets/<preset>.json` — the SAME file
/// `scratchy-quantizations` ships, parsed with the SAME parser the compiler
/// uses to interpret a real checkpoint's config) into its `QuantMethod`.
/// Empty for a dense/bf16 build (no quant preset enabled).
fn compiled_quant_methods(
    enabled_quant_presets: &BTreeSet<String>,
    repo_root: &Path,
) -> Vec<QuantMethod> {
    let presets_dir = repo_root
        .join("crates")
        .join("models")
        .join("quantization")
        .join("presets");
    enabled_quant_presets
        .iter()
        .filter_map(|preset| {
            let text = std::fs::read_to_string(presets_dir.join(format!("{preset}.json"))).ok()?;
            let json: serde_json::Value = serde_json::from_str(&text).ok()?;
            QuantizationConfig::parse(&json)
                .ok()
                .flatten()
                .map(|c| c.method)
        })
        .collect()
}

/// Fetches with `HF_TOKEN` (same env var `hf-hub-downloader` reads at
/// runtime) when set — without it, a gated repo (e.g. `meta-llama/*`, which
/// requires accepting a license) 401s and its shape can't be verified, so
/// it's excluded even though `scr chat`/`scr serve` would load it fine once
/// the user authenticates. With the builder's own token (already required
/// to download those checkpoints at all), the same gate that's already
/// been cleared for real usage is cleared here too.
fn fetch_config(id: &str) -> Option<String> {
    let url = format!("https://huggingface.co/{id}/raw/main/config.json");
    let mut req = ureq::get(&url);
    if let Some(token) = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()) {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let res = req.call().ok()?;
    let mut body = String::new();
    std::io::Read::read_to_string(&mut res.into_body().into_reader(), &mut body).ok()?;
    Some(body)
}

/// The decoder's own shape.
///
/// Every field is read from ONE scope, resolved by [`decoder_scope`]. Reading
/// them independently with a whole-tree search is what the first version did,
/// and it silently mixed towers: `gemma-4-12b-it.json` has no top-level
/// `hidden_size`, and its sibling sub-configs sort `audio_config` (hidden 640,
/// no layers) before `text_config` (hidden 3840, 48 layers) — so the "shape"
/// came out as the audio encoder's width paired with the decoder's depth.
///
/// That was self-cancelling as long as candidate configs nested identically,
/// which is why it went unnoticed. It breaks the moment one side omits a tower —
/// a text-only MLX conversion of a multimodal checkpoint — and then a loadable
/// model is rejected for a shape neither side actually declares.
fn shape_of(json: &serde_json::Value) -> Option<Shape> {
    let scope = decoder_scope(json);
    Some((
        find_number(scope, "hidden_size")?,
        find_number(scope, "num_hidden_layers")?,
        find_number(scope, "num_experts").or_else(|| find_number(scope, "num_local_experts")),
    ))
}

/// The sub-object describing the text decoder: the top level when it carries the
/// decoder's own fields, else the conventional wrapper key that does.
///
/// Multimodal/unified configs put the decoder under `text_config` and put
/// unrelated encoders in siblings (`audio_config`, `vision_config`), so the
/// wrapper must be resolved BEFORE any field lookup. Falls back to the whole
/// document, which leaves [`find_number`]'s tree walk as the last resort for a
/// layout not covered here.
fn decoder_scope(json: &serde_json::Value) -> &serde_json::Value {
    if json.get("hidden_size").and_then(|v| v.as_i64()).is_some() {
        return json;
    }
    for key in ["text_config", "language_model", "llm_config", "decoder"] {
        if let Some(inner) = json.get(key) {
            // Only accept a wrapper that really holds the decoder's width;
            // otherwise keep looking rather than locking onto an empty stub.
            if find_number(inner, "hidden_size").is_some() {
                return inner;
            }
        }
    }
    json
}

/// Depth-first search for the first `key` anywhere in the JSON tree with an
/// integer value — text configs commonly nest the decoder's own fields
/// under `text_config` (multimodal wrapper configs), so a flat top-level
/// lookup misses exactly the models that need this check most.
fn find_number(json: &serde_json::Value, key: &str) -> Option<i64> {
    match json {
        serde_json::Value::Object(map) => {
            if let Some(v) = map.get(key).and_then(|v| v.as_i64()) {
                return Some(v);
            }
            map.values().find_map(|v| find_number(v, key))
        }
        serde_json::Value::Array(arr) => arr.iter().find_map(|v| find_number(v, key)),
        _ => None,
    }
}

/// The Hub's own tag vocabulary for a compound arch identifier is
/// inconsistent (`qwen2_moe` — underscore — matches hundreds of repos,
/// `qwen2-moe` matches one; `gemma4-moe` has no separate MoE tag at all —
/// the whole family is just tagged `gemma4`). Rather than hardcode each
/// arch's quirk, try purely STRUCTURAL rewrites of the identifier itself
/// (verbatim, `-`→`_`, trailing `-<suffix>` dropped) and keep whichever
/// candidate matches the MOST repos — a lone coincidental match (e.g.
/// `gemma4-moe` matching exactly one unrelated repo) must not shadow a
/// broader, more useful tag (`gemma4`). Falls back to the arch name
/// verbatim (an honest empty result) if every rewrite matches nothing.
fn resolve_arch_tag(arch: &str, cache_dir: &Path) -> String {
    let mut candidates = vec![arch.to_string(), arch.replace('-', "_")];
    if let Some((base, _suffix)) = arch.rsplit_once('-') {
        candidates.push(base.to_string());
    }
    candidates.dedup();

    candidates
        .into_iter()
        .max_by_key(|candidate| resolve(&[candidate], cache_dir).len())
        .unwrap_or_else(|| arch.to_string())
}

/// One `filter=` tag combination's ids: fresh cache hit, or a Hub query
/// (falling back to stale cache, then to empty, on any failure).
fn resolve(tags: &[&str], cache_dir: &Path) -> Vec<String> {
    let cache_path = cache_dir.join(format!("{}.json", sanitize(&tags.join("+"))));

    let cached = std::fs::read_to_string(&cache_path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());

    if let Some(json) = &cached {
        let fresh = json
            .get("fetched_at")
            .and_then(|v| v.as_u64())
            .map(|t| now_secs().saturating_sub(t) < CACHE_TTL.as_secs())
            .unwrap_or(false);
        if fresh {
            return ids_from_cache_json(json);
        }
    }

    match query_hub(tags) {
        Some(ids) => {
            write_cache(&cache_path, &ids);
            ids
        }
        None => cached.as_ref().map(ids_from_cache_json).unwrap_or_default(),
    }
}

fn ids_from_cache_json(json: &serde_json::Value) -> Vec<String> {
    json.get("ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn write_cache(cache_path: &Path, ids: &[String]) {
    let json = serde_json::json!({
        "fetched_at": now_secs(),
        "ids": ids,
    });
    let _ = std::fs::write(cache_path, json.to_string());
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Filename-safes a tag combination for the cache path. Must be injective on
/// the inputs this module actually produces — critically, preserving `-` vs
/// `_` distinctly (they're DIFFERENT candidate tags in `resolve_arch_tag`;
/// collapsing both to one character makes two distinct queries collide on
/// one cache file, silently returning one candidate's stale result for the
/// other — exactly the bug this module used to have).
fn sanitize(tag: &str) -> String {
    tag.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '.'
            }
        })
        .collect()
}

/// `GET https://huggingface.co/api/models?filter=<tag1>&filter=<tag2>&...`,
/// returning the repo ids matching every given tag (the Hub ANDs repeated
/// `filter=` params). `None` on any network/parse failure — the caller
/// falls back to cache.
fn query_hub(tags: &[&str]) -> Option<Vec<String>> {
    let filters: String = tags
        .iter()
        .map(|t| format!("filter={}", urlencoding_minimal(t)))
        .collect::<Vec<_>>()
        .join("&");
    let url = format!(
        "https://huggingface.co/api/models?{filters}&sort=downloads&direction=-1&limit=200"
    );
    let res = ureq::get(&url).call().ok()?;
    let mut body = String::new();
    std::io::Read::read_to_string(&mut res.into_body().into_reader(), &mut body).ok()?;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    let arr = json.as_array()?;
    Some(
        arr.iter()
            .filter_map(|entry| entry.get("id").and_then(|v| v.as_str()))
            .map(str::to_string)
            .collect(),
    )
}

/// Percent-encodes only what a Hub tag can actually contain (alphanumerics,
/// `-`/`_`/`.`) — not a general-purpose URL encoder, scoped to this call site.
fn urlencoding_minimal(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c.to_string()
            } else {
                format!("%{:02X}", c as u32)
            }
        })
        .collect()
}
