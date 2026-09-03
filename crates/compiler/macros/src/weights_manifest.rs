// SPDX-License-Identifier: Apache-2.0
//! Per-architecture weights.json manifest.
//!
//! `crates/models/arch/configs/<arch>/weights.json` declares the shape of
//! every weight tensor the forward pass consumes, expressed in bound
//! names from the arch's config.json (e.g. `"hidden_size"`,
//! `"head_dim * num_attention_heads"`). The file is produced by the
//! `probe-weights` bin in the `scratchy-forward-compiler` crate by probing a
//! representative checkpoint's safetensors headers.
//!
//! Shape inference uses this manifest to anchor weight shapes that
//! dataflow can't pin by itself (the canonical example: Qwen3/Gemma3
//! per-head `q_norm` of shape `[head_dim]`, where a naïve op-signature
//! inference would resolve to `[heads * head_dim]` via the upstream
//! gemm). Unlike the old `weight_conventions.rs` table, the manifest
//! is per-arch data, not baked-in global knowledge.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::shape::{Dim, Shape, canonical_mul};

/// Parsed `weights.json`. Keys are dotted weight paths (e.g.
/// `"self_attn.q_proj"`), values are symbolic shapes (`Vec<Dim>`).
///
/// An optional `__packed_splits__` top-level key declares checkpoints
/// that ship fused tensors on disk (e.g. Phi-3's `self_attn.qkv_proj.weight`,
/// `mlp.gate_up_proj.weight`) instead of the per-slice logical tensors
/// the DSL references. Each entry maps the packed prefix (unindexed —
/// the codegen fans over `model.layers.{L}.…`) to an ordered list of
/// target paths; each target must itself appear in `entries`, and its
/// first-dim shape formula is the per-slice row count. At load time the
/// codegen-emitted prelude calls
/// [`GpuWeights::synthesize_packed_row_split_sizes`] per (layer, packed
/// prefix), splitting the mmap into virtual siblings before any
/// `Linear::load` runs.
#[derive(Debug, Default, Clone)]
pub struct WeightsManifest {
    pub entries: BTreeMap<String, Shape>,
    pub packed_splits: BTreeMap<String, Vec<String>>,
    /// Optional `__pad_to_mult8__` top-level key — list of weights to
    /// zero-pad on a specified axis to the next multiple of 8 at CPU
    /// load time. Each entry pairs a stem (matched against the
    /// indexed/unindexed loader convention, same as `entries`) with
    /// the dim to pad: `[{ "weight": "mlp.down_proj", "dim": 0 }, ...]`.
    /// Used by Qwen2.5-VL to dodge cuBLAS bf16 GEMM's K=3420 rejection
    /// — `vision_intermediate_size` rounded to the next mult of 8 lands
    /// every affected gemm on a supported algo (zero-fill rows/cols
    /// preserve math because `silu(0)·0 = 0` on the activation side
    /// and zero-row contractions are zero on the weight side).
    pub pad_to_mult8: Vec<PadHint>,
    /// Per-entry kind override. Absent ⇒ default
    /// [`ManifestEntryKind::LinearLayer`] (the standard PyTorch
    /// `nn.Linear` convention with `<prefix>.weight` / optional
    /// `<prefix>.bias` on disk). Present ⇒ the codegen routes
    /// the matching `LinearLayer` accessor to the alternate loader.
    /// See [`ManifestEntryKind`] for the variants.
    pub kinds: BTreeMap<String, ManifestEntryKind>,
}

/// Per-entry storage convention. The default
/// ([`ManifestEntryKind::LinearLayer`]) is the standard PyTorch
/// `nn.Linear` layout — the macro reads `<prefix>.weight` and
/// optional `<prefix>.bias` off disk. [`ManifestEntryKind::RawLinear`]
/// names an `nn.Parameter` weight (e.g. Gemma3's
/// `multi_modal_projector.mm_input_projection_weight`) that ships
/// without the `.weight` / `.bias` suffix; the codegen reads
/// `<prefix>` verbatim and the loaded layer carries no bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ManifestEntryKind {
    /// PyTorch `nn.Linear` convention: `<prefix>.weight` (+ optional
    /// `<prefix>.bias`).
    #[default]
    LinearLayer,
    /// PyTorch `nn.Parameter` convention: the on-disk key is
    /// `<prefix>` verbatim with no `.weight` suffix; no bias.
    RawLinear,
}

/// One zero-pad-to-mult-8 entry from `__pad_to_mult8__`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadHint {
    /// Weight stem (same naming convention as `entries`). For
    /// indexed weights (per-block `mlp.down_proj`) the stem is
    /// repeated under each layer index by the load-time prelude.
    pub weight: String,
    /// Axis to pad: 0 or 1. 0 = rows (pad N for an `[N, K]` weight);
    /// 1 = cols (pad K for `[N, K]`, which is what `down_proj` needs).
    pub dim: usize,
}

impl WeightsManifest {
    /// Look up a weight path (as a slice of segments from
    /// `WeightTable::path`) in the manifest, joining segments with
    /// `.` to match the file's key format.
    pub fn lookup<S: AsRef<str>>(&self, path_segments: &[S]) -> Option<&Shape> {
        let dotted: Vec<&str> = path_segments.iter().map(|s| s.as_ref()).collect();
        self.entries.get(&dotted.join("."))
    }

    /// Storage convention for a weight path. Returns the manifest's
    /// declared `kind` if present; otherwise falls back to the
    /// default ([`ManifestEntryKind::LinearLayer`]).
    pub fn kind<S: AsRef<str>>(&self, path_segments: &[S]) -> ManifestEntryKind {
        let dotted: Vec<&str> = path_segments.iter().map(|s| s.as_ref()).collect();
        self.kinds
            .get(&dotted.join("."))
            .copied()
            .unwrap_or_default()
    }

    /// Empty manifest — used when an arch directory has no
    /// `weights.json` yet (arches predating the probe workflow).
    /// Shape inference falls back to its dataflow-only behavior.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Test helper: build a manifest from a literal list of
    /// (dotted-path, shape-formula-strings). Parses each formula
    /// via the same path as [`load_file`].
    #[cfg(test)]
    pub fn from_literals(entries: &[(&str, &[&str])]) -> Self {
        let mut out = BTreeMap::new();
        for (path, dims) in entries {
            let shape: Shape = dims.iter().map(|d| parse_dim(d).unwrap()).collect();
            out.insert((*path).to_string(), shape);
        }
        Self {
            entries: out,
            packed_splits: BTreeMap::new(),
            pad_to_mult8: Vec::new(),
            kinds: BTreeMap::new(),
        }
    }

    /// The convention set the original `weight_conventions.rs` table
    /// encoded — the "universal HF dense-attention + SwiGLU decoder"
    /// shapes every Llama/Qwen2/Gemma2/Granite test body exercises.
    /// Exposed for test call sites; real macro runs load a per-arch
    /// `weights.json` off disk instead.
    #[cfg(test)]
    pub fn llama_test_conventions() -> Self {
        Self::from_literals(&[
            ("embed_tokens", &["vocab_size", "hidden_size"]),
            ("lm_head", &["hidden_size", "vocab_size"]),
            (
                "self_attn.q_proj",
                &["hidden_size", "head_dim * num_attention_heads"],
            ),
            (
                "self_attn.k_proj",
                &["hidden_size", "head_dim * num_key_value_heads"],
            ),
            (
                "self_attn.v_proj",
                &["hidden_size", "head_dim * num_key_value_heads"],
            ),
            (
                "self_attn.o_proj",
                &["head_dim * num_attention_heads", "hidden_size"],
            ),
            ("input_layernorm", &["hidden_size"]),
            ("post_attention_layernorm", &["hidden_size"]),
            ("norm", &["hidden_size"]),
            ("mlp.gate_proj", &["hidden_size", "intermediate_size"]),
            ("mlp.up_proj", &["hidden_size", "intermediate_size"]),
            ("mlp.down_proj", &["intermediate_size", "hidden_size"]),
        ])
    }
}

/// Errors produced when loading a `weights.json`.
#[derive(Debug)]
pub enum ManifestError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// A shape value in the JSON wasn't an array of strings.
    BadShape {
        path: PathBuf,
        weight: String,
        reason: String,
    },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "parsing {}: {source}", path.display()),
            Self::BadShape {
                path,
                weight,
                reason,
            } => write!(
                f,
                "{}: bad shape for `{}`: {reason}",
                path.display(),
                weight
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Load the `weights.json` next to the arch's config files. If the
/// file doesn't exist, returns an empty manifest — caller decides
/// whether that's OK or an error.
pub fn load_or_empty(arch_dir: &Path) -> Result<WeightsManifest, ManifestError> {
    let path = arch_dir.join("weights.json");
    if !path.exists() {
        return Ok(WeightsManifest::empty());
    }
    load_file(&path)
}

/// Load and parse an explicit path. Strict: file must exist and
/// parse cleanly.
pub fn load_file(path: &Path) -> Result<WeightsManifest, ManifestError> {
    let bytes = fs::read(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let raw: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| ManifestError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let obj = raw.as_object().ok_or_else(|| ManifestError::BadShape {
        path: path.to_path_buf(),
        weight: "<top-level>".into(),
        reason: "expected a JSON object".into(),
    })?;
    let mut entries: BTreeMap<String, Shape> = BTreeMap::new();
    let mut packed_splits: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut pad_to_mult8: Vec<PadHint> = Vec::new();
    let mut kinds: BTreeMap<String, ManifestEntryKind> = BTreeMap::new();
    for (k, v) in obj {
        if k == "__pad_to_mult8__" {
            let arr = v.as_array().ok_or_else(|| ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: k.clone(),
                reason: "`__pad_to_mult8__` must be an array of `{weight, dim}` objects".into(),
            })?;
            for (i, item) in arr.iter().enumerate() {
                let obj = item.as_object().ok_or_else(|| ManifestError::BadShape {
                    path: path.to_path_buf(),
                    weight: k.clone(),
                    reason: format!("entry {i} is not an object"),
                })?;
                let weight = obj
                    .get("weight")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: k.clone(),
                        reason: format!("entry {i} missing `weight: \"<stem>\"`"),
                    })?
                    .to_string();
                let dim = obj.get("dim").and_then(|v| v.as_u64()).ok_or_else(|| {
                    ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: k.clone(),
                        reason: format!("entry {i} (`{weight}`) missing integer `dim` field"),
                    }
                })? as usize;
                if dim > 1 {
                    return Err(ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: k.clone(),
                        reason: format!("entry {i} (`{weight}`): dim must be 0 or 1, got {dim}"),
                    });
                }
                pad_to_mult8.push(PadHint { weight, dim });
            }
            continue;
        }
        if k == "__packed_splits__" {
            let map = v.as_object().ok_or_else(|| ManifestError::BadShape {
                path: path.to_path_buf(),
                weight: k.clone(),
                reason: "`__packed_splits__` must be an object".into(),
            })?;
            for (packed, targets_v) in map {
                let arr = targets_v
                    .as_array()
                    .ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: packed.clone(),
                        reason: "packed-split value must be an array of target paths".into(),
                    })?;
                let mut targets = Vec::with_capacity(arr.len());
                for (i, item) in arr.iter().enumerate() {
                    let s = item.as_str().ok_or_else(|| ManifestError::BadShape {
                        path: path.to_path_buf(),
                        weight: packed.clone(),
                        reason: format!("target {i} is not a string"),
                    })?;
                    targets.push(s.to_string());
                }
                packed_splits.insert(packed.clone(), targets);
            }
            continue;
        }
        let (shape, kind) = parse_entry(v).map_err(|reason| ManifestError::BadShape {
            path: path.to_path_buf(),
            weight: k.clone(),
            reason,
        })?;
        entries.insert(k.clone(), shape);
        if let Some(kind) = kind {
            kinds.insert(k.clone(), kind);
        }
    }
    Ok(WeightsManifest {
        entries,
        packed_splits,
        pad_to_mult8,
        kinds,
    })
}

/// Parse one weight entry value. Two accepted shapes:
///   - Array of dim strings (`["hidden_size", "num_heads * head_dim"]`)
///     — the legacy default; entry kind defaults to `LinearLayer`.
///   - Object `{ "shape": [...], "kind": "raw_linear" }` — opt-in
///     storage convention override. Today the only non-default
///     kind is `raw_linear` (Gemma3 MM projector's
///     `mm_input_projection_weight` is an `nn.Parameter`, not a
///     `nn.Linear`).
fn parse_entry(v: &serde_json::Value) -> Result<(Shape, Option<ManifestEntryKind>), String> {
    if v.is_array() {
        return Ok((parse_shape(v)?, None));
    }
    let obj = v.as_object().ok_or_else(|| {
        "expected an array of shape strings or `{shape, kind}` object".to_string()
    })?;
    let shape_v = obj
        .get("shape")
        .ok_or_else(|| "object form requires a `shape` field".to_string())?;
    let shape = parse_shape(shape_v)?;
    let kind = match obj.get("kind").and_then(|k| k.as_str()) {
        Some("raw_linear") => Some(ManifestEntryKind::RawLinear),
        Some(other) => return Err(format!("unknown manifest entry kind `{other}`")),
        None => None,
    };
    Ok((shape, kind))
}

/// Parse a JSON shape value — must be an array of strings, each
/// string either a bound name (`"hidden_size"`) or a `*`-separated
/// product (`"head_dim * num_attention_heads"`).
fn parse_shape(v: &serde_json::Value) -> Result<Shape, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "expected an array of shape strings".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or_else(|| format!("dim {i} is not a string"))?;
        out.push(parse_dim(s)?);
    }
    Ok(out)
}

fn parse_dim(s: &str) -> Result<Dim, String> {
    let factors: Vec<Dim> = s
        .split('*')
        .map(str::trim)
        .map(|f| {
            if f.is_empty() {
                return Err("empty factor in product".to_string());
            }
            if let Ok(n) = f.parse::<u64>() {
                Ok(Dim::Lit(n))
            } else {
                Ok(Dim::Bound(f.to_string()))
            }
        })
        .collect::<Result<_, _>>()?;
    match factors.len() {
        0 => Err("empty shape string".to_string()),
        1 => Ok(factors.into_iter().next().unwrap()),
        _ => Ok(canonical_mul(factors)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Dim;

    #[test]
    fn parses_single_bound() {
        assert_eq!(
            parse_dim("hidden_size").unwrap(),
            Dim::Bound("hidden_size".into())
        );
    }

    #[test]
    fn parses_product_of_two_bounds_canonically() {
        // Product form is canonicalised — bound names sorted
        // alphabetically so `head_dim * num_attention_heads` produces
        // the same `Dim` as `num_attention_heads * head_dim`.
        let a = parse_dim("head_dim * num_attention_heads").unwrap();
        let b = parse_dim("num_attention_heads * head_dim").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn parses_integer_literal_factor() {
        let d = parse_dim("2 * head_dim").unwrap();
        match d {
            Dim::Mul(fs) => {
                assert!(fs.iter().any(|f| matches!(f, Dim::Lit(2))));
                assert!(
                    fs.iter()
                        .any(|f| matches!(f, Dim::Bound(b) if b == "head_dim"))
                );
            }
            _ => panic!("expected Mul"),
        }
    }

    #[test]
    fn lookup_joins_segments_with_dots() {
        let mut m = WeightsManifest::empty();
        m.entries.insert(
            "self_attn.q_proj".into(),
            vec![Dim::Bound("hidden_size".into())],
        );
        let got = m.lookup(&["self_attn", "q_proj"]);
        assert!(got.is_some());
    }

    #[test]
    fn missing_file_returns_empty_manifest() {
        let tmp = std::env::temp_dir().join("scratchy_weights_manifest_test_nonexistent");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let m = load_or_empty(&tmp).unwrap();
        assert!(m.entries.is_empty());
        assert!(m.pad_to_mult8.is_empty());
    }

    #[test]
    fn parses_pad_to_mult8_top_level_array() {
        // G.6.5 contract: `__pad_to_mult8__` is a top-level array of
        // `{ "weight": "...", "dim": 0|1 }` objects. Order is
        // preserved, dim is parsed as usize, and other top-level keys
        // (regular weight stems) coexist cleanly.
        let tmp = std::env::temp_dir().join("scratchy_pad_mult8_manifest_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{
              "__pad_to_mult8__": [
                {"weight": "mlp.gate_proj", "dim": 0},
                {"weight": "mlp.up_proj", "dim": 0},
                {"weight": "mlp.down_proj", "dim": 1}
              ],
              "norm1": ["vision_embed_dim"]
            }"#,
        )
        .unwrap();
        let m = load_file(&path).expect("load_file");
        assert_eq!(m.pad_to_mult8.len(), 3);
        assert_eq!(
            m.pad_to_mult8[0],
            PadHint {
                weight: "mlp.gate_proj".into(),
                dim: 0
            }
        );
        assert_eq!(
            m.pad_to_mult8[2],
            PadHint {
                weight: "mlp.down_proj".into(),
                dim: 1
            }
        );
        // Coexists with regular entries.
        assert!(m.entries.contains_key("norm1"));
    }

    #[test]
    fn parses_raw_linear_object_form() {
        // G.7(a) contract: a manifest value of the form
        // `{ "shape": [...], "kind": "raw_linear" }` parses cleanly,
        // populates `entries` with the shape AND `kinds` with the
        // override. Array-form entries leave `kinds` absent (default
        // is `LinearLayer`, exposed via `WeightsManifest::kind`).
        let tmp = std::env::temp_dir().join("scratchy_raw_linear_manifest_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{
              "self_attn.q_proj": ["hidden_size", "hidden_size"],
              "mm.mm_input_projection_weight": {
                "shape": ["vision_embed_dim", "d_model"],
                "kind": "raw_linear"
              }
            }"#,
        )
        .unwrap();
        let m = load_file(&path).expect("load_file");
        // Both entries land in `entries`.
        assert!(m.entries.contains_key("self_attn.q_proj"));
        assert!(m.entries.contains_key("mm.mm_input_projection_weight"));
        // Kind override only on the object-form entry.
        assert_eq!(
            m.kind(&["mm", "mm_input_projection_weight"]),
            ManifestEntryKind::RawLinear
        );
        assert_eq!(
            m.kind(&["self_attn", "q_proj"]),
            ManifestEntryKind::LinearLayer
        );
        // Absent path → default kind.
        assert_eq!(
            m.kind(&["does", "not", "exist"]),
            ManifestEntryKind::LinearLayer
        );
    }

    #[test]
    fn rejects_unknown_manifest_entry_kind() {
        let tmp = std::env::temp_dir().join("scratchy_bad_kind_manifest_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{ "x": { "shape": ["d"], "kind": "totally_made_up" } }"#,
        )
        .unwrap();
        let err = load_file(&path).expect_err("unknown kind must reject");
        let msg = format!("{err}");
        assert!(msg.contains("unknown manifest entry kind"), "got: {msg}");
    }

    #[test]
    fn pad_to_mult8_rejects_dim_out_of_range() {
        let tmp = std::env::temp_dir().join("scratchy_pad_mult8_bad_dim_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("weights.json");
        std::fs::write(
            &path,
            r#"{ "__pad_to_mult8__": [{"weight": "x", "dim": 7}] }"#,
        )
        .unwrap();
        let err = load_file(&path).expect_err("dim=7 is out of range");
        let msg = format!("{err}");
        assert!(msg.contains("dim must be 0 or 1"), "got: {msg}");
    }
}
