// SPDX-License-Identifier: Apache-2.0
//! Per-arch declarations, in memory: everything about an arch that is
//! not derivable from a verbatim HF `config.json` plus the DSL body.
//!
//! This module holds the RESOLVED form ([`DeclaredArchSpec`]) plus the
//! two evaluators that turn it into bounds — [`eval_expr`] for the
//! `expr` arithmetic and [`json_path`] for dotted config lookups. It
//! does no parsing of its own: the declarations are read from
//! `configs/<arch>/arch.json` by [`crate::config::load_arch_json`].
//!
//! THE LAW (same one `scratchy_vision::MmMetadata` documents): no arch
//! names appear in scratchy-forward-compiler, scratchy-forward-compiler-macro, or
//! scratchy-serving-api — only in the per-arch data that owns the declaration.
//!
//! These facts used to be written as `const` items inside a `mod` DSL
//! carrier, and scraped back out of the token stream here. That surface
//! is gone, because it was a config file costumed as Rust: the type
//! ascriptions (`Layout`, `Fingerprint`, `ScaleDtype`, …) never
//! resolved to anything, so the macro string-matched const NAMES and
//! pulled literals out of the expressions. Declarations about
//! checkpoints now live WITH the checkpoints, as JSON — the same
//! channel (and the same key names) the per-checkpoint
//! `<stem>.overrides.json` drift files already used, so there is one
//! vocabulary instead of two.
//!
//! Precedence, widest to narrowest: verbatim HF `config.json`
//! (per size) → `arch.json` (per arch) → `<stem>.overrides.json`
//! (per checkpoint), each winning field-by-field over the one before.

use crate::config::{VisionDModelFingerprint, VisionPatchEmbedFlatten, VisionSafetensorsLayout};

// ── Resolved declaration (internal) ─────────────────────────────────

/// The arch's declarations, token-parsed into plain data. One per
/// `#[forward]` / `#[vision_forward]` invocation; per-checkpoint JSON
/// drift merges on top in `config::resolve_arch_spec`.
/// One `published_as` entry (arch.json): a compiled variant of the `head_targets` config its
/// target drafts with by default, and the Hub repo it loads from (Gemma 4 26B-A4B's assistant at
/// `mlx-affine-b4-g32`: `guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit`). The config
/// compiles beside its selected target under `mtp-heads` (which `metal` turns on), and a checkpoint
/// of that target's shape served with no `--speculative-model` loads the repo of the variant the
/// build compiled. At most one config a target; a config no entry names compiles only under
/// `spec/mtp`, and a variant none names drafts only when `--speculative-model` names a checkpoint.
#[derive(Clone, Debug)]
pub struct PublishedAs {
    /// The `head_targets` config.
    pub config: String,
    /// Its compiled variant: the config's own stem (dense), or `<config>-<preset>`.
    pub variant: String,
    pub repo: String,
}

#[derive(Clone, Debug, Default)]
pub struct DeclaredArchSpec {
    pub safetensors: Option<VisionSafetensorsLayout>,
    pub fingerprint: Option<VisionDModelFingerprint>,
    pub patch_embed_flatten: Option<VisionPatchEmbedFlatten>,
    pub pos_embed_key: Option<String>,
    /// `"bilinear"` (default) / `"bicubic"`.
    pub pos_emb_interp: Option<String>,
    /// `"neox_hw"` (default) / `"interleaved_xy"`.
    pub rope_style: Option<String>,
    pub vision_norm_eps: Option<f64>,
    /// RMSNorm-GAIN tensor dtype the metal kernels' `_s_<dtype>_`
    /// symbol arm reads: `"f16"` (default — the mlx-community
    /// f16-gain repack convention) / `"bf16"` (checkpoints shipping
    /// bf16 gains: Qwen3-family, Gemma4, full-bf16 originals).
    /// Mis-declaring reads gain bytes in the wrong float layout —
    /// e.g. bf16 0x3E87 (0.264) as f16 1.63 — and garbles every
    /// norm. Per-checkpoint repacks override via the JSON
    /// `scale_dtype` drift key.
    pub scale_dtype: Option<String>,
    /// `decoder_safetensors_prefix`: a namespace WRAPPING the standard
    /// `model.<...>` decoder tree (Gemma3-MM / mlx-vlm `language_model`
    /// → `language_model.model.<...>`); a value that itself starts with
    /// `model` names the root it replaces (`model.language_model.<...>`);
    /// the empty value keys the tree at the file's top level (an MLX
    /// multi-token-prediction drafter's bare `fc.*`, `layers.0.*`).
    pub decoder_prefix: Option<String>,
    /// `decoder_safetensors_root`: the decoder tree's root under ANOTHER
    /// name, replacing `model` outright (Nemotron-H `backbone.<...>`).
    /// Exclusive with `decoder_prefix` — enforced where both tiers merge.
    pub decoder_root: Option<String>,
    /// The identity the arch registers under, replacing its config's `architectures`: an arch
    /// that is one part of another model's checkpoint (a multi-token-prediction head ships inside
    /// its target's, whose config names the target) declares its own. Empty: the config's.
    pub hf_architectures: Vec<String>,
    /// How a head published apart from its target names its repo: the target's repo id with this
    /// inserted before its last `-`-delimited token (MLX: `org/X-4bit` → `org/X-MTP-4bit`).
    pub drafter_repo_infix: Option<String>,
    /// A head published as its own checkpoint, with a config of its own (Gemma 4's assistant):
    /// each config stem's target config, `<arch>/<stem>`. It compiles when that target is selected
    /// under `spec/mtp` (under `mtp-heads` too, for the config `published_as` names), the target's
    /// verify steps carry its drafts, and it drafts for the target's architectures. Empty for a
    /// head that ships inside its target's checkpoint (`checkpoint_of`).
    pub head_targets: Vec<(String, String)>,
    /// The Hub repos the `head_targets` config its target drafts with by default is published
    /// as, one a compiled variant of it (a drafter checkpoint loads only into the variant of its
    /// own precision).
    pub published_as: Vec<PublishedAs>,
    /// The arches whose checkpoints carry this head (a multi-token-prediction head). It has no
    /// configs of its own: under `spec/mtp` it compiles on every selected config of these arches
    /// that declares its layers (the source of its `num_hidden_layers` param, above zero).
    pub checkpoint_of: Vec<String>,
    /// Where its tensors sit inside such a checkpoint (`language_model.mtp`): a target checkpoint
    /// holding tensors under it carries its head, which then loads from the target's own store.
    pub checkpoint_prefix: Option<String>,
    /// The head's attention reads its target's KV (`num_kv_shared_layers`: every layer of it): it
    /// writes none and has no KV pool of its own; each layer reads the target layer its config's
    /// overrides name for its class (`target_kv_sliding_layer`, `target_kv_full_layer`).
    pub reads_target_kv: bool,
    /// The weights a head borrows from its target (`embed_tokens`, `lm_head`); `None`: both.
    pub borrows: Option<Vec<String>>,
    /// Weights at the checkpoint's top level beside the decoder root, as `lm_head` is.
    pub top_level_weights: Vec<String>,
    pub tie_default: Option<bool>,
    pub bound_defaults: Vec<(String, u64)>,
    /// Float-valued analogue of `bound_defaults` for arch-constant
    /// SCALARS the checkpoint omits — e.g. Gemma-4's softmax
    /// `attention_multiplier = 1.0` (mlx hardcodes `scale = 1.0` in
    /// the attention module; no config field carries it). Applied to
    /// the `scalars` map; an explicit config value always wins.
    pub scalar_defaults: Vec<(String, f64)>,
    pub config_aliases: Vec<(String, String)>,
    pub weight_leaf_renames: Vec<(String, String)>,
    /// The `struct Params` schema, evaluated per config.json into the
    /// bound set. Empty for bare-`fn` arches (standard flat harvest
    /// only).
    pub params: Vec<ParamField>,
}

/// One `struct Params` field: the bound it defines + where its value
/// comes from.
#[derive(Clone, Debug)]
pub struct ParamField {
    pub name: String,
    pub source: ParamSource,
    /// Overwrite the harvested bound of the same name instead of filling it only when absent:
    /// an MTP head's config carries its target's `num_hidden_layers` beside its own
    /// `mtp_num_hidden_layers`, and the head's model has the latter.
    pub replaces: bool,
}

#[derive(Clone, Debug)]
pub enum ParamSource {
    /// `#[from = "dotted.path"]` into the verbatim config.json
    /// (numeric segments index arrays), with an optional
    /// `default = <int>` for configs that omit the key.
    From { path: String, default: Option<u64> },
    /// `#[expr = "a * b / c"]` over previously-declared fields and
    /// integer literals (left-assoc, `*`/`/` bind tighter, parens).
    Expr(String),
    /// `#[value = <int>]` literal.
    Value(u64),
}

// ── Schema evaluation ───────────────────────────────────────────────

/// Dotted-path lookup into a JSON value: segments descend objects;
/// all-digit segments index arrays (`"merge_kernel_size.0"`).
pub fn json_path<'a>(root: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cur = root;
    for seg in path.split('.') {
        cur = if seg.bytes().all(|b| b.is_ascii_digit()) && cur.is_array() {
            cur.get(seg.parse::<usize>().ok()?)?
        } else {
            cur.get(seg)?
        };
    }
    Some(cur)
}

impl DeclaredArchSpec {
    /// The decoder tree's safetensors root — the ONE resolution of
    /// `decoder_safetensors_prefix` / `decoder_safetensors_root` every
    /// key builder reads (`<root>.layers.<L>.<leaf>`, `<root>.<leaf>`;
    /// `lm_head` always sits at the file's top level).
    pub fn decoder_root(&self) -> String {
        match (self.decoder_root.as_deref(), self.decoder_prefix.as_deref()) {
            (Some(_), Some(_)) => {
                unreachable!("decoder_safetensors_root and _prefix are exclusive (config parse)")
            }
            (Some(root), None) => root.trim_end_matches('.').to_string(),
            (None, Some(prefix)) => {
                let prefix = prefix.trim_end_matches('.');
                if prefix.is_empty() || prefix.starts_with("model") {
                    prefix.to_string()
                } else {
                    format!("{prefix}.model")
                }
            }
            (None, None) => "model".to_string(),
        }
    }

    /// Evaluate the `Params` schema against one verbatim config.json,
    /// inserting each field into `bounds` in declaration order. A
    /// bound already present (the flat `.overrides.json` drift
    /// surface) wins over the schema's value, and is visible to later
    /// `#[expr]` fields.
    pub fn eval_params(
        &self,
        json: &serde_json::Value,
        bounds: &mut std::collections::BTreeMap<String, u64>,
    ) -> Result<(), String> {
        for f in &self.params {
            if bounds.contains_key(&f.name) && !f.replaces {
                continue;
            }
            let v = match &f.source {
                ParamSource::From { path, default } => json_path(json, path)
                    .and_then(|v| v.as_u64())
                    .or(*default)
                    .ok_or_else(|| format!("params field `{}`: config has no `{path}`", f.name))?,
                ParamSource::Expr(e) => eval_expr(e, bounds)
                    .map_err(|err| format!("params field `{}`: {err}", f.name))?,
                ParamSource::Value(v) => *v,
            };
            bounds.insert(f.name.clone(), v);
        }
        Ok(())
    }
}

/// Minimal integer expression evaluator for `#[expr]`: identifiers
/// (earlier fields), integer literals, `+ - * /`, parentheses.
fn eval_expr(src: &str, env: &std::collections::BTreeMap<String, u64>) -> Result<u64, String> {
    struct P<'a> {
        toks: Vec<&'a str>,
        pos: usize,
    }
    fn tokenize(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut start = None::<usize>;
        for (i, c) in s.char_indices() {
            if c.is_alphanumeric() || c == '_' {
                if start.is_none() {
                    start = Some(i);
                }
            } else {
                if let Some(st) = start.take() {
                    out.push(&s[st..i]);
                }
                if !c.is_whitespace() {
                    out.push(&s[i..i + c.len_utf8()]);
                }
            }
        }
        if let Some(st) = start {
            out.push(&s[st..]);
        }
        out
    }
    impl<'a> P<'a> {
        fn peek(&self) -> Option<&'a str> {
            self.toks.get(self.pos).copied()
        }
        fn next(&mut self) -> Option<&'a str> {
            let t = self.peek();
            self.pos += 1;
            t
        }
    }
    fn atom(p: &mut P, env: &std::collections::BTreeMap<String, u64>) -> Result<u64, String> {
        match p.next() {
            Some("(") => {
                let v = sum(p, env)?;
                if p.next() != Some(")") {
                    return Err("expected `)`".to_string());
                }
                Ok(v)
            }
            // `sqrt(x)` — exact integer square root (errors when the
            // argument isn't a perfect square; pooling kernels etc.
            // are exact by construction).
            Some("sqrt") => {
                if p.next() != Some("(") {
                    return Err("sqrt: expected `(`".to_string());
                }
                let v = sum(p, env)?;
                if p.next() != Some(")") {
                    return Err("sqrt: expected `)`".to_string());
                }
                let r = (v as f64).sqrt().round() as u64;
                if r * r != v {
                    return Err(format!("sqrt({v}) is not an integer"));
                }
                Ok(r)
            }
            Some(t) if t.bytes().all(|b| b.is_ascii_digit()) => {
                t.parse().map_err(|e| format!("bad int `{t}`: {e}"))
            }
            Some(t) => env
                .get(t)
                .copied()
                .ok_or_else(|| format!("unknown field `{t}` (declare it earlier in Params)")),
            None => Err("unexpected end of expression".to_string()),
        }
    }
    fn prod(p: &mut P, env: &std::collections::BTreeMap<String, u64>) -> Result<u64, String> {
        let mut v = atom(p, env)?;
        while matches!(p.peek(), Some("*") | Some("/")) {
            let op = p.next().unwrap();
            let r = atom(p, env)?;
            v = match op {
                "*" => v * r,
                // `/` is FLOOR division (HF-config arithmetic like
                // `(x + 7) / 8 * 8` rounding needs it; exact cases
                // are unaffected).
                _ => {
                    if r == 0 {
                        return Err(format!("{v} / 0"));
                    }
                    v / r
                }
            };
        }
        Ok(v)
    }
    fn sum(p: &mut P, env: &std::collections::BTreeMap<String, u64>) -> Result<u64, String> {
        let mut v = prod(p, env)?;
        while matches!(p.peek(), Some("+") | Some("-")) {
            let op = p.next().unwrap();
            let r = prod(p, env)?;
            v = match op {
                "+" => v + r,
                _ => v
                    .checked_sub(r)
                    .ok_or_else(|| format!("{v} - {r} underflows"))?,
            };
        }
        Ok(v)
    }
    let mut p = P {
        toks: tokenize(src),
        pos: 0,
    };
    let v = sum(&mut p, env)?;
    if p.peek().is_some() {
        return Err(format!("trailing tokens after expression in `{src}`"));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ONE decoder-root resolution every safetensors key builder reads:
    /// the HF default, a wrapper around `model`, a `model.*` prefix naming
    /// the root it replaces, the file's top level, and a renamed root
    /// (Nemotron-H `backbone`).
    #[test]
    fn decoder_root_resolves_every_declaration() {
        let root = |prefix: Option<&str>, root: Option<&str>| {
            DeclaredArchSpec {
                decoder_prefix: prefix.map(str::to_string),
                decoder_root: root.map(str::to_string),
                ..Default::default()
            }
            .decoder_root()
        };
        assert_eq!(root(None, None), "model");
        assert_eq!(root(Some("language_model"), None), "language_model.model");
        assert_eq!(root(Some("language_model."), None), "language_model.model");
        assert_eq!(
            root(Some("model.language_model"), None),
            "model.language_model"
        );
        assert_eq!(root(Some(""), None), "");
        assert_eq!(root(None, Some("backbone")), "backbone");
        assert_eq!(root(None, Some("backbone.")), "backbone");
    }

    #[test]
    fn expr_eval_precedence_and_division() {
        let mut env = std::collections::BTreeMap::new();
        env.insert("a".to_string(), 1152u64);
        env.insert("b".to_string(), 16u64);
        assert_eq!(eval_expr("a / b", &env).unwrap(), 72);
        assert_eq!(eval_expr("a / b / 2", &env).unwrap(), 36);
        assert_eq!(eval_expr("3 * (a + b)", &env).unwrap(), 3504);
        assert_eq!(eval_expr("a * 2 * 2", &env).unwrap(), 4608);
        assert_eq!(eval_expr("a / 5", &env).unwrap(), 230); // floor division
        assert_eq!(eval_expr("sqrt(b)", &env).unwrap(), 4);
        assert!(eval_expr("sqrt(a)", &env).is_err()); // not a perfect square
        assert!(eval_expr("c + 1", &env).is_err()); // unknown
    }

    #[test]
    fn json_path_descends_objects_and_arrays() {
        let j: serde_json::Value = serde_json::json!({
            "vision_config": { "merge_kernel_size": [2, 2], "hidden_size": 1152 },
            "text_config": { "hidden_size": 2048 }
        });
        assert_eq!(
            json_path(&j, "vision_config.merge_kernel_size.0").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            json_path(&j, "text_config.hidden_size").and_then(|v| v.as_u64()),
            Some(2048)
        );
        assert!(json_path(&j, "vision_config.nope").is_none());
    }
}
