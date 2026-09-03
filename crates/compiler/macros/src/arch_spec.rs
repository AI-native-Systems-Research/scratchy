// SPDX-License-Identifier: Apache-2.0
//! Per-arch declaration surface — the typed Rust the model crate
//! writes and this macro READS (as tokens, at expansion) to learn
//! everything about an arch that is not derivable from a verbatim HF
//! `config.json` plus the DSL body.
//!
//! THE LAW (same one `scratchy_vision::MmMetadata` documents): no arch
//! names appear in scratchy-forward-compiler, scratchy-forward-compiler-macro, or
//! scratchy-serving-api — only in the per-arch crate that owns the declaration.
//!
//! An arch with declarations uses the `mod` carrier form:
//!
//! ```ignore
//! #[vision_forward(workloads = [256, 1024], processor = crate::PROCESSOR)]
//! mod locateanything {
//!     /// THE params schema: field name = the bound name the DSL and
//!     /// weights.json reference. One field per bound — the schema is
//!     /// the complete, typed answer to "what are this arch's params".
//!     struct Params {
//!         #[from = "vision_config.hidden_size"]         vision_embed_dim: u64,
//!         #[from = "vision_config.num_hidden_layers"]   vision_depth: u64,
//!         #[from = "vision_config.temporal_patch_size", default = 1]
//!         vision_temporal_patch_size: u64,
//!         #[value = 2]                                  vision_spatial_merge_size: u64,
//!         #[expr = "vision_embed_dim / vision_num_heads"] vision_head_dim: u64,
//!     }
//!
//!     const NORM_EPS: f64 = 1e-5;
//!     const ROPE_STYLE: RopeStyle = RopeStyle::InterleavedXy;
//!     const POS_EMB_INTERP: PosEmbInterp = PosEmbInterp::Bicubic;
//!     const POS_EMBED_KEY: &str = "vision_tower.patch_embed.pos_emb.weight";
//!     const SAFETENSORS: Layout = Layout {
//!         root: "vision_tower", blocks: "blocks",
//!         subtrees: &[("mm", "multi_modal_projector")],
//!     };
//!     const FINGERPRINT: Fingerprint =
//!         Fingerprint { key: "multi_modal_projector.linear_2.weight", dim: 0 };
//!     const PATCH_EMBED_FLATTEN: Flatten = Flatten {
//!         key: "vision_tower.patch_embed.proj.weight", channels_last: true,
//!     };
//!     const WEIGHT_LEAF_RENAMES: &[(&str, &str)] =
//!         &[("mm.proj_in", "mm.linear_1"), ("mm.proj_out", "mm.linear_2")];
//!
//!     fn forward() { /* DSL */ }
//! }
//! ```
//!
//! Decoder-side consts: `DECODER_PREFIX: &str`, `TIE_DEFAULT: bool`,
//! `BOUND_DEFAULTS: &[(&str, u64)]` (inserted when the config omits
//! them — zero-centered norms, sliding cadence, router renorm, …),
//! `CONFIG_ALIASES: &[(&str, &str)]` (read standard key `.0` from
//! alt key `.1` when absent — alias, never rename).
//!
//! The `mod` items are CONSUMED by the macro (the DSL body is not
//! Rust; neither are these — the type ascriptions are documentation,
//! the macro keys on the const NAMES and parses literal values).
//! Plain arches keep the bare `fn` carrier unchanged. Per-size values
//! live in each verbatim config.json (mapped by `#[from]`);
//! per-checkpoint drift keeps the `.overrides.json` surface, which
//! wins field-by-field over the declaration.

use syn::Token;
use syn::parse::ParseStream;
use syn::spanned::Spanned;

use crate::config::{VisionDModelFingerprint, VisionPatchEmbedFlatten, VisionSafetensorsLayout};

// ── Resolved declaration (internal) ─────────────────────────────────

/// The arch's declarations, token-parsed into plain data. One per
/// `#[forward]` / `#[vision_forward]` invocation; per-checkpoint JSON
/// drift merges on top in `config::resolve_arch_spec`.
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
    pub decoder_prefix: Option<String>,
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
            if bounds.contains_key(&f.name) {
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

// ── Token parsing: the `mod` carrier's items ────────────────────────

/// String literal value of a const expr, tolerating `&str` refs.
fn lit_str(expr: &syn::Expr) -> syn::Result<String> {
    if let syn::Expr::Lit(l) = expr
        && let syn::Lit::Str(s) = &l.lit
    {
        return Ok(s.value());
    }
    Err(syn::Error::new(expr.span(), "expected a string literal"))
}

fn lit_u64(expr: &syn::Expr) -> syn::Result<u64> {
    if let syn::Expr::Lit(l) = expr
        && let syn::Lit::Int(i) = &l.lit
    {
        return i.base10_parse();
    }
    Err(syn::Error::new(expr.span(), "expected an integer literal"))
}

/// `&[("a", "b"), …]` — slice of 2-string tuples.
fn lit_str_pairs(expr: &syn::Expr) -> syn::Result<Vec<(String, String)>> {
    let arr = slice_elems(expr)?;
    let mut out = Vec::new();
    for el in arr {
        if let syn::Expr::Tuple(t) = el
            && t.elems.len() == 2
        {
            out.push((lit_str(&t.elems[0])?, lit_str(&t.elems[1])?));
        } else {
            return Err(syn::Error::new(el.span(), "expected (\"a\", \"b\") tuple"));
        }
    }
    out.sort();
    Ok(out)
}

/// `&[("a", 1), …]` — slice of (string, int) tuples.
fn lit_str_f64_pairs(expr: &syn::Expr) -> syn::Result<Vec<(String, f64)>> {
    let arr = slice_elems(expr)?;
    let mut out = Vec::new();
    for el in arr {
        if let syn::Expr::Tuple(t) = el
            && t.elems.len() == 2
        {
            out.push((lit_str(&t.elems[0])?, lit_f64(&t.elems[1])?));
        } else {
            return Err(syn::Error::new(
                el.span(),
                "expected (\"name\", <float>) tuple",
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Parse a float OR integer literal as `f64` (`1.0` and `1` both
/// accepted — config scalars are written either way).
fn lit_f64(expr: &syn::Expr) -> syn::Result<f64> {
    if let syn::Expr::Lit(l) = expr {
        match &l.lit {
            syn::Lit::Float(f) => return f.base10_parse::<f64>(),
            syn::Lit::Int(i) => return Ok(i.base10_parse::<u64>()? as f64),
            _ => {}
        }
    }
    Err(syn::Error::new(expr.span(), "expected a float literal"))
}

fn lit_str_u64_pairs(expr: &syn::Expr) -> syn::Result<Vec<(String, u64)>> {
    let arr = slice_elems(expr)?;
    let mut out = Vec::new();
    for el in arr {
        if let syn::Expr::Tuple(t) = el
            && t.elems.len() == 2
        {
            out.push((lit_str(&t.elems[0])?, lit_u64(&t.elems[1])?));
        } else {
            return Err(syn::Error::new(
                el.span(),
                "expected (\"name\", <int>) tuple",
            ));
        }
    }
    Ok(out)
}

fn slice_elems(expr: &syn::Expr) -> syn::Result<Vec<&syn::Expr>> {
    let inner = match expr {
        syn::Expr::Reference(r) => &*r.expr,
        e => e,
    };
    if let syn::Expr::Array(a) = inner {
        return Ok(a.elems.iter().collect());
    }
    Err(syn::Error::new(
        expr.span(),
        "expected `&[…]` slice literal",
    ))
}

/// Struct-literal field access: `Layout { root: "…", … }`.
fn struct_fields(expr: &syn::Expr) -> syn::Result<Vec<(String, &syn::Expr)>> {
    if let syn::Expr::Struct(s) = expr {
        let mut out = Vec::new();
        for f in &s.fields {
            let syn::Member::Named(name) = &f.member else {
                return Err(syn::Error::new(f.span(), "expected named field"));
            };
            out.push((name.to_string(), &f.expr));
        }
        return Ok(out);
    }
    Err(syn::Error::new(expr.span(), "expected a struct literal"))
}

/// Trailing path segment of `RopeStyle::InterleavedXy`-style exprs.
fn variant_ident(expr: &syn::Expr) -> syn::Result<String> {
    if let syn::Expr::Path(p) = expr
        && let Some(seg) = p.path.segments.last()
    {
        return Ok(seg.ident.to_string());
    }
    Err(syn::Error::new(
        expr.span(),
        "expected a path like `Enum::Variant`",
    ))
}

impl DeclaredArchSpec {
    /// Consume one `const NAME: _ = <literal>;` item from the carrier
    /// mod. Unknown names error (typos must not silently no-op).
    pub fn parse_const(&mut self, item: &syn::ItemConst) -> syn::Result<()> {
        let name = item.ident.to_string();
        let expr = &*item.expr;
        match name.as_str() {
            "NORM_EPS" => {
                if let syn::Expr::Lit(l) = expr
                    && let syn::Lit::Float(f) = &l.lit
                {
                    self.vision_norm_eps = Some(f.base10_parse()?);
                } else {
                    return Err(syn::Error::new(expr.span(), "NORM_EPS: float literal"));
                }
            }
            "ROPE_STYLE" => {
                self.rope_style = Some(match variant_ident(expr)?.as_str() {
                    "NeoxHw" => "neox_hw".to_string(),
                    "InterleavedXy" => "interleaved_xy".to_string(),
                    other => {
                        return Err(syn::Error::new(
                            expr.span(),
                            format!("ROPE_STYLE: unknown variant `{other}` (NeoxHw|InterleavedXy)"),
                        ));
                    }
                });
            }
            "POS_EMB_INTERP" => {
                self.pos_emb_interp = Some(match variant_ident(expr)?.as_str() {
                    "Bilinear" => "bilinear".to_string(),
                    "Bicubic" => "bicubic".to_string(),
                    other => {
                        return Err(syn::Error::new(
                            expr.span(),
                            format!("POS_EMB_INTERP: unknown variant `{other}` (Bilinear|Bicubic)"),
                        ));
                    }
                });
            }
            "POS_EMBED_KEY" => self.pos_embed_key = Some(lit_str(expr)?),
            "SCALE_DTYPE" => {
                self.scale_dtype = Some(match variant_ident(expr)?.as_str() {
                    "F16" => "f16".to_string(),
                    "Bf16" => "bf16".to_string(),
                    other => {
                        return Err(syn::Error::new(
                            expr.span(),
                            format!("SCALE_DTYPE: unknown variant `{other}` (F16|Bf16)"),
                        ));
                    }
                });
            }
            "SAFETENSORS" => {
                let mut root = None;
                let mut blocks = None;
                let mut subtrees = std::collections::BTreeMap::new();
                for (k, v) in struct_fields(expr)? {
                    match k.as_str() {
                        "root" => root = Some(lit_str(v)?),
                        "blocks" => blocks = Some(lit_str(v)?),
                        "subtrees" => {
                            for (a, b) in lit_str_pairs(v)? {
                                subtrees.insert(a, b);
                            }
                        }
                        other => {
                            return Err(syn::Error::new(
                                expr.span(),
                                format!("SAFETENSORS: unknown field `{other}`"),
                            ));
                        }
                    }
                }
                self.safetensors = Some(VisionSafetensorsLayout {
                    default_root: root.ok_or_else(|| {
                        syn::Error::new(expr.span(), "SAFETENSORS: missing `root`")
                    })?,
                    layered_subpath: blocks.ok_or_else(|| {
                        syn::Error::new(expr.span(), "SAFETENSORS: missing `blocks`")
                    })?,
                    subtrees,
                });
            }
            "FINGERPRINT" => {
                let mut key = None;
                let mut dim = 0usize;
                for (k, v) in struct_fields(expr)? {
                    match k.as_str() {
                        "key" => key = Some(lit_str(v)?),
                        "dim" => dim = lit_u64(v)? as usize,
                        other => {
                            return Err(syn::Error::new(
                                expr.span(),
                                format!("FINGERPRINT: unknown field `{other}`"),
                            ));
                        }
                    }
                }
                self.fingerprint = Some(VisionDModelFingerprint {
                    key: key.ok_or_else(|| {
                        syn::Error::new(expr.span(), "FINGERPRINT: missing `key`")
                    })?,
                    dim,
                });
            }
            "PATCH_EMBED_FLATTEN" => {
                let mut key = None;
                let mut leading_dim = 0usize;
                let mut channels_last = false;
                for (k, v) in struct_fields(expr)? {
                    match k.as_str() {
                        "key" => key = Some(lit_str(v)?),
                        "leading_dim" => leading_dim = lit_u64(v)? as usize,
                        "channels_last" => {
                            if let syn::Expr::Lit(l) = v
                                && let syn::Lit::Bool(b) = &l.lit
                            {
                                channels_last = b.value();
                            } else {
                                return Err(syn::Error::new(v.span(), "expected bool literal"));
                            }
                        }
                        other => {
                            return Err(syn::Error::new(
                                expr.span(),
                                format!("PATCH_EMBED_FLATTEN: unknown field `{other}`"),
                            ));
                        }
                    }
                }
                self.patch_embed_flatten = Some(VisionPatchEmbedFlatten {
                    key: key.ok_or_else(|| {
                        syn::Error::new(expr.span(), "PATCH_EMBED_FLATTEN: missing `key`")
                    })?,
                    leading_dim,
                    channels_last,
                });
            }
            "WEIGHT_LEAF_RENAMES" => self.weight_leaf_renames = lit_str_pairs(expr)?,
            "DECODER_PREFIX" => self.decoder_prefix = Some(lit_str(expr)?),
            "TIE_DEFAULT" => {
                if let syn::Expr::Lit(l) = expr
                    && let syn::Lit::Bool(b) = &l.lit
                {
                    self.tie_default = Some(b.value());
                } else {
                    return Err(syn::Error::new(expr.span(), "TIE_DEFAULT: bool literal"));
                }
            }
            "BOUND_DEFAULTS" => self.bound_defaults = lit_str_u64_pairs(expr)?,
            "SCALAR_DEFAULTS" => self.scalar_defaults = lit_str_f64_pairs(expr)?,
            "CONFIG_ALIASES" => self.config_aliases = lit_str_pairs(expr)?,
            other => {
                return Err(syn::Error::new(
                    item.ident.span(),
                    format!(
                        "unknown arch declaration const `{other}` — known: NORM_EPS, \
                         ROPE_STYLE, POS_EMB_INTERP, POS_EMBED_KEY, SCALE_DTYPE, SAFETENSORS, \
                         FINGERPRINT, PATCH_EMBED_FLATTEN, WEIGHT_LEAF_RENAMES, \
                         DECODER_PREFIX, TIE_DEFAULT, BOUND_DEFAULTS, SCALAR_DEFAULTS, \
                         CONFIG_ALIASES",
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Consume the `struct Params { … }` schema from the carrier mod.
    pub fn parse_params_struct(&mut self, item: &syn::ItemStruct) -> syn::Result<()> {
        let syn::Fields::Named(fields) = &item.fields else {
            return Err(syn::Error::new(
                item.ident.span(),
                "Params must use named fields",
            ));
        };
        for f in &fields.named {
            let name = f
                .ident
                .as_ref()
                .ok_or_else(|| syn::Error::new(f.span(), "Params field needs a name"))?
                .to_string();
            let mut from: Option<String> = None;
            let mut default: Option<u64> = None;
            let mut expr: Option<String> = None;
            let mut value: Option<u64> = None;
            for attr in &f.attrs {
                let ident = attr
                    .path()
                    .get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_default();
                match ident.as_str() {
                    // `#[from = "path"]` or `#[from = "path", default = N]`
                    "from" => {
                        let meta: FromMeta = attr.parse_args_with(FromMeta::parse).or_else(
                            |_| -> syn::Result<FromMeta> {
                                // name-value form: #[from = "path"]
                                if let syn::Meta::NameValue(nv) = &attr.meta {
                                    Ok(FromMeta {
                                        path: lit_str(&nv.value)?,
                                        default: None,
                                    })
                                } else {
                                    Err(syn::Error::new(
                                        attr.span(),
                                        "expected #[from = \"dotted.path\"]",
                                    ))
                                }
                            },
                        )?;
                        from = Some(meta.path);
                        if meta.default.is_some() {
                            default = meta.default;
                        }
                    }
                    "default" => {
                        if let syn::Meta::NameValue(nv) = &attr.meta {
                            default = Some(lit_u64(&nv.value)?);
                        }
                    }
                    "expr" => {
                        if let syn::Meta::NameValue(nv) = &attr.meta {
                            expr = Some(lit_str(&nv.value)?);
                        }
                    }
                    "value" => {
                        if let syn::Meta::NameValue(nv) = &attr.meta {
                            value = Some(lit_u64(&nv.value)?);
                        }
                    }
                    "doc" => {}
                    other => {
                        return Err(syn::Error::new(
                            attr.span(),
                            format!("Params field `{name}`: unknown attribute `{other}`"),
                        ));
                    }
                }
            }
            let source = match (from, expr, value) {
                (Some(path), None, None) => ParamSource::From { path, default },
                (None, Some(e), None) => ParamSource::Expr(e),
                (None, None, Some(v)) => ParamSource::Value(v),
                _ => {
                    return Err(syn::Error::new(
                        f.span(),
                        format!(
                            "Params field `{name}` needs exactly one of \
                             #[from = …], #[expr = …], #[value = …]",
                        ),
                    ));
                }
            };
            self.params.push(ParamField { name, source });
        }
        Ok(())
    }
}

/// `#[from("path", default = N)]` paren form.
struct FromMeta {
    path: String,
    default: Option<u64>,
}

impl FromMeta {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: syn::LitStr = input.parse()?;
        let mut default = None;
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let key: syn::Ident = input.parse()?;
            if key != "default" {
                return Err(syn::Error::new(key.span(), "expected `default`"));
            }
            input.parse::<Token![=]>()?;
            let v: syn::LitInt = input.parse()?;
            default = Some(v.base10_parse()?);
        }
        Ok(Self {
            path: path.value(),
            default,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
