// SPDX-License-Identifier: Apache-2.0
//! probe-weights — bootstrap a new model architecture's
//! `crates/models/arch/configs/<arch>/` directory by downloading
//! `config.json` for each listed HuggingFace repo and extracting a
//! per-arch `weights.json` manifest from one representative
//! checkpoint's safetensors header.
//!
//! Usage:
//! ```text
//! cargo run -p scratchy-forward-compiler --bin probe-weights --features probe -- \
//!     --arch qwen3 \
//!     Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B
//! ```
//!
//! Output for the above command (under workspace root):
//! ```text
//! crates/models/arch/configs/qwen3/qwen3-0.6b.json
//! crates/models/arch/configs/qwen3/qwen3-1.7b.json
//! crates/models/arch/configs/qwen3/qwen3-4b.json
//! crates/models/arch/configs/qwen3/qwen3-8b.json
//! crates/models/arch/configs/qwen3/weights.json
//! ```
//!
//! TODO (future): add a `--hf-family Qwen/Qwen3` auto-discovery mode
//! that hits `https://huggingface.co/api/models?search=Qwen3&author=Qwen`,
//! filters by `config.model_type`, and enumerates base (non-finetuned)
//! checkpoints so the dev doesn't have to type out every repo ID by
//! hand.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use hf_hub_downloader::Client;
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(about = "Bootstrap a scratchy-forward-compiler model architecture directory from HF")]
struct Args {
    /// Architecture name — used to locate
    /// `crates/models/arch/configs/<arch>/` under the workspace
    /// root. Use the arch's standard HF `model_type` value (e.g.
    /// `qwen3`, `gemma3`, `llama`). Underscores in the arch name
    /// (e.g. `deepseek_v2`) are converted to dashes when locating
    /// the crate directory.
    #[arg(long)]
    arch: String,

    /// HuggingFace repo IDs for each model size the arch supports.
    /// The FIRST repo is probed for weight shapes; every repo
    /// contributes its config.json. Example:
    /// `Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B`.
    #[arg(required = true)]
    repos: Vec<String>,

    /// Override for the output directory. When omitted, the prober
    /// walks up from the current working directory to find the
    /// workspace root, then writes to
    /// `<workspace>/crates/models/arch/configs/<arch>/`. The directory
    /// must already exist (see `docs/MODELS.md` for the
    /// new-arch recipe).
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Overwrite existing size configs and weights.json. Without
    /// this flag, the prober refuses to clobber a committed
    /// `weights.json` (sanity check for accidental re-runs).
    #[arg(long)]
    force: bool,
}

/// Walk up from `start` to find the workspace root — the nearest
/// ancestor whose `Cargo.toml` contains `[workspace]`.
fn find_workspace_root(start: &Path) -> Result<PathBuf> {
    let mut cur: Option<&Path> = Some(start);
    while let Some(d) = cur {
        let cargo_toml = d.join("Cargo.toml");
        if cargo_toml.exists()
            && fs::read_to_string(&cargo_toml)
                .ok()
                .is_some_and(|s| s.contains("[workspace]"))
        {
            return Ok(d.to_path_buf());
        }
        cur = d.parent();
    }
    Err(anyhow!(
        "no workspace `Cargo.toml` (with `[workspace]`) found walking up from {}",
        start.display(),
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve the configs/ directory: explicit override or inferred
    // from arch + workspace root. Every arch's configs live under
    // `crates/models/arch/configs/<arch>/` in the consolidated
    // scratchy-models crate. Arch DSL idents may use underscores
    // (e.g. `deepseek_v2`); the on-disk dir uses dashes
    // (`deepseek-v2`).
    let arch_dir = match args.out_dir.clone() {
        Some(p) => p,
        None => {
            let cwd = std::env::current_dir().context("reading current directory")?;
            let workspace = find_workspace_root(&cwd)?;
            let arch_dir_name = args.arch.replace('_', "-");
            workspace
                .join("crates")
                .join("models")
                .join("arch")
                .join("configs")
                .join(arch_dir_name)
        }
    };
    if !arch_dir.is_dir() {
        bail!(
            "configs directory does not exist: {}\n\
             Add a config JSON under crates/models/arch/configs/<arch>/ first (see docs/MODELS.md).",
            arch_dir.display(),
        );
    }

    let api = Client::from_env();

    // ── Step 1: download + write every repo's config.json ──────────
    let mut configs: Vec<(String, PathBuf, Value)> = Vec::new();
    for repo in &args.repos {
        let (size_name, out_path) = size_name_and_path(&arch_dir, repo)?;
        if out_path.exists() && !args.force {
            eprintln!(
                "skip  {} (exists; pass --force to overwrite)",
                out_path.display()
            );
            let existing: Value = serde_json::from_slice(&fs::read(&out_path)?)?;
            configs.push((size_name, out_path, existing));
            continue;
        }
        eprintln!("fetch {} → {}", repo, out_path.display());
        let config = fetch_config_json(&api, repo)?;
        let pretty = serde_json::to_string_pretty(&config)? + "\n";
        fs::write(&out_path, pretty).with_context(|| format!("writing {}", out_path.display()))?;
        configs.push((size_name, out_path, config));
    }

    if configs.is_empty() {
        bail!("no repos were processed");
    }

    // ── Step 2: sanity-check all configs agree on model_type ───────
    let expected_model_type = configs[0].2.get("model_type").and_then(|v| v.as_str());
    for (name, _, cfg) in &configs {
        let t = cfg.get("model_type").and_then(|v| v.as_str());
        if t != expected_model_type {
            eprintln!(
                "warn: {} has model_type={:?}, others have {:?}",
                name, t, expected_model_type,
            );
        }
    }

    // ── Step 3: probe EVERY size's safetensors for weight shapes.
    //           With only one size, ambiguous cases (`hidden_size`
    //           vs `num_kv_heads * head_dim` when they coincide)
    //           can't be disambiguated. Every listed size acts as
    //           a constraint: a formula must evaluate to the
    //           *actual* dim in *every* size to be valid.
    let mut per_size: Vec<SizeProbe> = Vec::new();
    for (name, _, cfg) in &configs {
        let repo = args
            .repos
            .iter()
            .find(|r| r.rsplit('/').next().unwrap().to_ascii_lowercase() == *name)
            .ok_or_else(|| anyhow!("no repo id maps back to {}", name))?;
        eprintln!("probe {} for weight shapes", repo);
        let raw = extract_safetensors_shapes(&api, repo)?;
        eprintln!("  found {} weight tensors", raw.len());
        let mut tensors = raw_shapes_by_arch_path(&raw);
        // Tied-embedding models don't store `lm_head.weight` in
        // safetensors — it aliases `embed_tokens.weight` at load
        // time. Detect this STRUCTURALLY: if the safetensors has
        // `embed_tokens` but no `lm_head`, the model is necessarily
        // tied (an untied LM without an lm_head couldn't generate
        // tokens). The DSL body still references `lm_head`, so
        // synthesize a manifest entry in compiler gemm order
        // `[hidden_size, vocab_size]` by swapping the embed's
        // `[vocab_size, hidden_size]`.
        //
        // Trusting the structural signal rather than the config's
        // `tie_word_embeddings` flag avoids breakage on forks
        // (e.g. unsloth) that re-serialize configs and drop fields.
        if !tensors.contains_key("lm_head")
            && let Some(embed_shape) = tensors.get("embed_tokens").cloned()
            && embed_shape.len() == 2
        {
            tensors.insert("lm_head".to_string(), vec![embed_shape[1], embed_shape[0]]);
            // Tied embedding implied by missing lm_head on disk. Some
            // upstream configs (Cohere's tiny-random mirror) omit
            // `tie_word_embeddings` even though the safetensors clearly
            // ties them — without this stamp, codegen's tied-load arm
            // wouldn't fire and the runtime would `not found:
            // lm_head.weight`. Inject the flag into the saved per-model
            // config so the macro reads `true` from disk.
            //
            // Locate the matching saved config and rewrite it. This is
            // the *committed* JSON in `crates/models/arch/configs/<arch>/`,
            // not just an in-memory edit — the macro reads from disk
            // at expansion time.
            let saved_path = configs
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, p, _)| p.clone())
                .unwrap_or_else(|| arch_dir.join(format!("{name}.json")));
            if let Ok(bytes) = fs::read(&saved_path)
                && let Ok(mut v) = serde_json::from_slice::<Value>(&bytes)
                && let Some(obj) = v.as_object_mut()
                && !obj.contains_key("tie_word_embeddings")
            {
                obj.insert(
                    "tie_word_embeddings".to_string(),
                    serde_json::Value::Bool(true),
                );
                let pretty = serde_json::to_string_pretty(&v).unwrap() + "\n";
                let _ = fs::write(&saved_path, pretty);
                eprintln!(
                    "  injected tie_word_embeddings=true into {} (lm_head missing on disk)",
                    saved_path.display(),
                );
            }
        }
        let _ = cfg;
        per_size.push(SizeProbe {
            repo: repo.clone(),
            bounds: bounds_from_config(cfg),
            tensors,
        });
    }

    let manifest = build_weights_manifest_multi(&per_size)?;

    // ── Step 5: write weights.json ─────────────────────────────────
    let weights_path = arch_dir.join("weights.json");
    if weights_path.exists() && !args.force {
        bail!(
            "{} already exists; pass --force to overwrite",
            weights_path.display(),
        );
    }
    let pretty = serde_json::to_string_pretty(&manifest)? + "\n";
    fs::write(&weights_path, pretty)
        .with_context(|| format!("writing {}", weights_path.display()))?;
    eprintln!("wrote {}", weights_path.display());

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

/// One per probed size: the bounds from its config and the arch-level
/// path → actual dim shape of every weight in its safetensors.
/// Multiple raw tensor names collapse to the same arch path after
/// the layer prefix strip (`model.layers.<N>.<path>` → `<path>`);
/// we assert all duplicates agree on shape when building the map.
struct SizeProbe {
    #[allow(dead_code)]
    repo: String,
    bounds: BTreeMap<String, u64>,
    tensors: BTreeMap<String, Vec<usize>>,
}

/// Build a `path → shape` map from a raw `(name, shape)` list of
/// WEIGHT tensors only (bias entries are dropped — bias is a
/// safetensors property the runtime loader handles directly). Keys
/// are layer-prefix-stripped and suffix-stripped so
/// `model.layers.3.self_attn.q_proj.weight` becomes `self_attn.q_proj`.
///
/// GEMM-layout normalisation: HF stores projection weights as
/// `[out_features, in_features]`; the scratchy-forward-compiler compiler's
/// `sig_gemm` models them as `[K, N] = [in_features, out_features]`.
/// We swap the two dims of every 2-D weight so the manifest is in
/// the compiler's order. `embed_tokens` is the exception — it's
/// consumed by `sig_embed`, whose signature treats the tensor as
/// `[vocab_size, hidden_size]` directly (the same as the
/// safetensors layout), so it's kept as-is.
fn raw_shapes_by_arch_path(raw: &[(String, Vec<usize>)]) -> BTreeMap<String, Vec<usize>> {
    let mut out: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (name, dims) in raw {
        let Some((path, kind, _)) = strip_layer_and_suffix(name) else {
            eprintln!("warn: skipping non-standard tensor name `{}`", name);
            continue;
        };
        if kind != TensorKind::Weight {
            continue; // drop biases
        }
        let dims = gemm_layout_normalise(&path, dims);
        match out.get(&path) {
            Some(existing) if existing != &dims => {
                eprintln!(
                    "warn: `{}` has inconsistent per-layer shapes: {:?} vs {:?}",
                    path, existing, dims,
                );
            }
            _ => {
                out.insert(path, dims);
            }
        }
    }
    out
}

fn gemm_layout_normalise(path: &str, dims: &[usize]) -> Vec<usize> {
    if dims.len() == 2 && path != "embed_tokens" {
        vec![dims[1], dims[0]]
    } else {
        dims.to_vec()
    }
}

/// Derive `(size_name, out_path)` for a repo id. `Qwen/Qwen3-0.6B` →
/// `("qwen3-0.6b", <arch_dir>/qwen3-0.6b.json)`.
fn size_name_and_path(arch_dir: &Path, repo: &str) -> Result<(String, PathBuf)> {
    let last = repo
        .rsplit('/')
        .next()
        .ok_or_else(|| anyhow!("empty repo id"))?;
    let size_name = last.to_ascii_lowercase();
    Ok((
        size_name.clone(),
        arch_dir.join(format!("{size_name}.json")),
    ))
}

/// Download and parse a repo's `config.json` via the HF Hub API. The
/// file is cached under `$HF_HOME` (or `~/.cache/huggingface`) by
/// `hf-hub-downloader` — subsequent invocations read from disk.
fn fetch_config_json(api: &Client, repo: &str) -> Result<Value> {
    let path = api
        .model(repo)
        .get("config.json")
        .with_context(|| format!("downloading {}/config.json", repo))?;
    let bytes = fs::read(&path)?;
    let v: Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}/config.json", repo))?;
    Ok(v)
}

/// Collect every integer-valued top-level field from a HF
/// `config.json`, plus the HF-convention derived bounds (head_dim,
/// num_key_value_heads) when they're missing.
fn bounds_from_config(cfg: &Value) -> BTreeMap<String, u64> {
    let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
    if let Some(obj) = cfg.as_object() {
        for (k, v) in obj {
            if let Some(n) = v.as_u64() {
                bounds.insert(k.clone(), n);
            }
        }
    }
    // HF defaults — head_dim = hidden_size / num_attention_heads,
    // num_key_value_heads = num_attention_heads.
    if !bounds.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (bounds.get("hidden_size"), bounds.get("num_attention_heads"))
        && heads != 0
        && hidden.is_multiple_of(heads)
    {
        bounds.insert("head_dim".into(), hidden / heads);
    }
    if !bounds.contains_key("num_key_value_heads")
        && let Some(&heads) = bounds.get("num_attention_heads")
    {
        bounds.insert("num_key_value_heads".into(), heads);
    }
    bounds
}

/// Extract `(tensor_name, shape)` for every weight tensor in a repo
/// by reading ONLY the safetensors header of each shard — never the
/// full weight data. Avoids multi-GB downloads.
///
/// Safetensors format: `[8 bytes little-endian u64 header_len]
/// [header_len bytes JSON]{"tensor_name": {"dtype", "shape",
/// "data_offsets"}, …}[tensor data ...]`. We only need the first
/// `8 + header_len` bytes; the tensor data is discarded.
///
/// Uses HTTP `Range` requests via `ureq`:
/// 1. Fetch first 4 MB of each shard (most headers fit comfortably).
/// 2. Read the first 8 bytes → `header_len`.
/// 3. If the header extends beyond what we fetched, re-request with
///    `Range: bytes=0-(8+header_len-1)`.
/// 4. Parse the header JSON for tensor metadata.
///
/// `config.json` and `model.safetensors.index.json` are small, so
/// those still flow through `Client::get` (which caches them to disk).
fn extract_safetensors_shapes(api: &Client, repo: &str) -> Result<Vec<(String, Vec<usize>)>> {
    let model = api.model(repo);

    // List the safetensors shards. Try the sharded index first; fall
    // back to a single-file layout.
    let shard_names: Vec<String> = match model.get("model.safetensors.index.json") {
        Ok(index_path) => {
            let bytes = fs::read(&index_path)?;
            let index: Value = serde_json::from_slice(&bytes)?;
            let wt_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow!("index.json missing `weight_map`"))?;
            let mut names: std::collections::BTreeSet<String> = Default::default();
            for v in wt_map.values() {
                if let Some(s) = v.as_str() {
                    names.insert(s.to_string());
                }
            }
            names.into_iter().collect()
        }
        Err(_) => vec!["model.safetensors".to_string()],
    };

    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    for shard in &shard_names {
        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, shard);
        let header = fetch_safetensors_header(&url)
            .with_context(|| format!("range-fetching header of {}", url))?;
        let index: Map<String, Value> = serde_json::from_slice(&header)
            .with_context(|| format!("parsing safetensors header JSON from {}", url))?;
        for (name, meta) in &index {
            // The safetensors header also carries a top-level
            // `"__metadata__"` entry (not a tensor). Skip it.
            if name == "__metadata__" {
                continue;
            }
            let shape_val = meta
                .get("shape")
                .ok_or_else(|| anyhow!("tensor `{}` in {} has no `shape`", name, url))?;
            let shape_arr = shape_val
                .as_array()
                .ok_or_else(|| anyhow!("tensor `{}` shape is not an array", name))?;
            let shape: Vec<usize> = shape_arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| anyhow!("shape dim is not a u64"))
                        .map(|n| n as usize)
                })
                .collect::<Result<_, _>>()?;
            out.push((name.to_string(), shape));
        }
    }
    Ok(out)
}

const INITIAL_HEADER_FETCH_BYTES: u64 = 4 * 1024 * 1024; // 4 MB

/// Fetch just the JSON header bytes of a safetensors file via HTTP
/// `Range`. Issues one request for the first 4 MB; if the header
/// extends past that, issues a second request for the exact range.
/// Returns the header JSON bytes (without the 8-byte length prefix).
fn fetch_safetensors_header(url: &str) -> Result<Vec<u8>> {
    let buf = fetch_range(url, 0, INITIAL_HEADER_FETCH_BYTES - 1)?;
    if buf.len() < 8 {
        bail!("short read: got {} bytes from {}", buf.len(), url);
    }
    let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap());
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or_else(|| anyhow!("header_len {header_len} overflows when added to 8 prefix bytes"))?;
    if (buf.len() as u64) >= header_end {
        return Ok(buf[8..(header_end as usize)].to_vec());
    }
    // Header bigger than our initial fetch — get the exact range.
    let mut full = fetch_range(url, 0, header_end - 1)?;
    if (full.len() as u64) < header_end {
        bail!(
            "short read on follow-up: wanted {header_end} bytes, got {}",
            full.len()
        );
    }
    Ok(full.split_off(8))
}

/// Issue a single HTTP GET with a byte-range header and return the
/// body. Follows redirects (HF Hub returns 302 → CDN URL). HF's CDN
/// honours ranges.
fn fetch_range(url: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .max_redirects(5)
        .timeout_global(Some(std::time::Duration::from_secs(60)))
        .build()
        .into();
    let mut req = agent
        .get(url)
        .header("Range", format!("bytes={start}-{end_inclusive}"));
    // Pass through the user's HF token if set — needed for gated
    // checkpoints like `google/gemma-2-*`. `hf_hub` reads the same
    // env var internally; we mirror it here for consistency.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let resp = req.call().map_err(|e| anyhow!("GET {url}: {e}"))?;
    let mut body = Vec::with_capacity((end_inclusive - start + 1) as usize);
    resp.into_body()
        .into_reader()
        .take(end_inclusive - start + 1)
        .read_to_end(&mut body)?;
    Ok(body)
}

/// Strip the per-layer prefix (`model.layers.<N>.`) AND the tensor
/// suffix (`.weight` or `.bias`) from a safetensors name.
///
/// Returns `Some((arch_path, kind, is_layered))` for tensors that
/// match the expected HF naming, or `None` for exotic names that
/// don't end in `.weight` / `.bias` (skipped from the manifest with
/// a warning).
///
/// The shape manifest keys on the weight-only path (e.g.
/// `self_attn.q_proj`) — matching the `classified::Program`'s weight
/// IDs, which are the dotted path without the `.weight` tail. Bias
/// entries are dropped: bias existence and shape are safetensors
/// properties read at runtime by the loader, not something the
/// manifest needs to declare.
fn strip_layer_and_suffix(name: &str) -> Option<(String, TensorKind, bool)> {
    let (body, kind) = if let Some(stem) = name.strip_suffix(".weight") {
        (stem, TensorKind::Weight)
    } else if let Some(stem) = name.strip_suffix(".bias") {
        (stem, TensorKind::Bias)
    } else {
        return None;
    };
    // Try layer-scoped strip: model.layers.<N>.<path>
    if let Some(rest) = body.strip_prefix("model.layers.")
        && let Some(dot) = rest.find('.')
    {
        let (idx, path) = rest.split_at(dot);
        if idx.parse::<u64>().is_ok() {
            return Some((path[1..].to_string(), kind, true));
        }
    }
    if let Some(rest) = body.strip_prefix("model.") {
        return Some((rest.to_string(), kind, false));
    }
    Some((body.to_string(), kind, false))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    Weight,
    Bias,
}

fn score_formula(
    formula: &str,
    probe_value: u64,
    other_size_bounds: &[BTreeMap<String, u64>],
) -> i32 {
    let mut score = 0i32;
    // Reward formulas that RESOLVE across all other sizes. A
    // formula that depends on a bound missing in another size is
    // disqualified (score -100).
    for bounds in other_size_bounds {
        match eval_formula(formula, bounds) {
            Some(_) => score += 10,
            None => return -100,
        }
    }
    // Prefer single-bound over product in ties. When `hidden_size`
    // and `heads * head_dim` are numerically equal for every size
    // we tested (as in Qwen2.5, where they coincide by design), we
    // have no data to say which is "semantically right" — the
    // single-bound form matches the HF convention for most weights
    // (norms, embeddings, lm_head). The product form only wins
    // when at least one size breaks the coincidence, which makes
    // the single-bound resolve to the wrong number and drops it
    // from the candidate set (score = -100) earlier in this fn.
    if !formula.contains('*') {
        score += 1;
    }
    // Tiny penalty for using a bound whose value equals another
    // bound's value trivially — discourage e.g. picking
    // `num_key_value_heads` when `num_attention_heads` is the
    // semantically correct choice. In practice, shorter bound
    // names sort later alphabetically in the `*` form, but here we
    // just break ties deterministically.
    score -= formula.len() as i32 / 100;
    let _ = probe_value;
    score
}

/// Enumerate every way `d` can be expressed over `bounds`: a single
/// bound name, or `a * b` for two distinct bounds. Returned strings
/// are canonical (bound-name products sorted alphabetically to match
/// `canonical_mul` in `shape.rs`).
///
/// Cross-size resolution disambiguates between multiple candidates:
/// a dim like `hidden_size == num_attention_heads * head_dim` in
/// Qwen2.5-0.5B (896 = 14 * 64) could be either formula — but only
/// one of them will also resolve consistently for Qwen2.5-7B, where
/// the two quantities are no longer equal.
fn candidate_dim_formulas(d: u64, bounds: &BTreeMap<String, u64>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (k, &v) in bounds {
        if v == d {
            out.push(k.clone());
        }
    }
    for (k1, &v1) in bounds {
        if v1 == 0 || !d.is_multiple_of(v1) {
            continue;
        }
        let q = d / v1;
        for (k2, &v2) in bounds {
            if k1 == k2 || v2 != q {
                continue;
            }
            let (a, b) = if k1 < k2 {
                (k1.clone(), k2.clone())
            } else {
                (k2.clone(), k1.clone())
            };
            let formula = format!("{a} * {b}");
            if !out.contains(&formula) {
                out.push(formula);
            }
        }
    }
    out
}

/// Evaluate a shape-formula string (`"head_dim"` or
/// `"head_dim * num_attention_heads"`) against a bounds table.
/// Returns the numeric value, or `None` if any factor is unknown.
fn eval_formula(formula: &str, bounds: &BTreeMap<String, u64>) -> Option<u64> {
    let mut acc = 1u64;
    for factor in formula.split('*').map(str::trim) {
        if let Ok(n) = factor.parse::<u64>() {
            acc = acc.checked_mul(n)?;
        } else {
            let v = bounds.get(factor)?;
            acc = acc.checked_mul(*v)?;
        }
    }
    Some(acc)
}

/// Build the `{ arch_path: [shape_formula] }` map from every probed
/// size's actual tensor shapes. For each dim, a candidate formula
/// must evaluate to the ACTUAL dim integer in EVERY size — only
/// those survive cross-validation. Among survivors, pick the best
/// via `score_formula`. If a path appears in some sizes but not
/// others (e.g. MoE vs non-MoE under one family), skip it.
fn build_weights_manifest_multi(per_size: &[SizeProbe]) -> Result<Map<String, Value>> {
    if per_size.is_empty() {
        bail!("no sizes to probe");
    }
    // Union of all paths seen in any size.
    let mut all_paths: std::collections::BTreeSet<String> = Default::default();
    for s in per_size {
        all_paths.extend(s.tensors.keys().cloned());
    }

    let mut manifest: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in &all_paths {
        // Collect (actual_dims, bounds) for every size that has this path.
        let entries: Vec<(&Vec<usize>, &BTreeMap<String, u64>)> = per_size
            .iter()
            .filter_map(|s| s.tensors.get(path).map(|d| (d, &s.bounds)))
            .collect();
        if entries.len() < per_size.len() {
            eprintln!(
                "skip  {} (present in {}/{} sizes)",
                path,
                entries.len(),
                per_size.len(),
            );
            continue;
        }
        // Every size's dims for this path must have the same rank.
        let rank = entries[0].0.len();
        if entries.iter().any(|(d, _)| d.len() != rank) {
            bail!("`{}` has inconsistent rank across sizes", path);
        }
        let mut shape_formulas: Vec<String> = Vec::with_capacity(rank);
        for dim_idx in 0..rank {
            let formula = pick_formula_for_dim(path, dim_idx, &entries)?;
            shape_formulas.push(formula);
        }
        manifest.insert(path.clone(), shape_formulas);
    }

    // Convert to serde_json::Map with stable ordering.
    let mut out = Map::new();
    for (k, v) in manifest {
        let arr: Vec<Value> = v.into_iter().map(Value::String).collect();
        out.insert(k, Value::Array(arr));
    }
    Ok(out)
}

/// Pick a single formula that evaluates to the actual dim `dim_idx`
/// of tensor `path` in EVERY size. Cross-validation: intersect
/// candidate sets per size and score the survivors.
fn pick_formula_for_dim(
    path: &str,
    dim_idx: usize,
    entries: &[(&Vec<usize>, &BTreeMap<String, u64>)],
) -> Result<String> {
    // Start with size-0's candidates, then intersect with each
    // subsequent size's candidates.
    let (dims0, bounds0) = entries[0];
    let d0 = dims0[dim_idx] as u64;
    let mut survivors: Vec<String> = candidate_dim_formulas(d0, bounds0);
    for (dims, bounds) in entries.iter().skip(1) {
        let actual = dims[dim_idx] as u64;
        survivors.retain(|f| eval_formula(f, bounds) == Some(actual));
    }
    if survivors.is_empty() {
        bail!(
            "no formula resolves to the actual dim in every size for `{}`[dim {}]; \
             per-size actuals: {:?}",
            path,
            dim_idx,
            entries.iter().map(|(d, _)| d[dim_idx]).collect::<Vec<_>>(),
        );
    }
    // Score remaining candidates. `score_formula`'s per-size
    // "validates" check is redundant now (survivors already validate
    // by construction), but its single-bound-vs-product tiebreak
    // still fires.
    let mut scored: Vec<(&String, i32)> = survivors
        .iter()
        .map(|c| (c, score_formula(c, d0, &[])))
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(scored[0].0.clone())
}
