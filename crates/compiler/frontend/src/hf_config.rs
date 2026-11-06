// SPDX-License-Identifier: Apache-2.0
//! How a verbatim HF `config.json` and its arch's `arch.json` become bounds: the config's
//! normalization, the bounds it states, an arch's `params` expression grammar, and a flag's value.
//! One rule each, read by the config loader and by the macros crate's build script, which compiles
//! this file by path (the typed-Rust emit pass resolves per-carrier facts from the configs before
//! the crate compiles), so the two cannot drift.

use std::collections::BTreeMap;

/// Dotted-path lookup into a JSON value: segments descend objects; numeric segments index arrays.
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

/// A config as the bounds read it: `text_config` and `rope_parameters` hoisted (an explicit top
/// level wins), then the arch's key aliases: where the standard key `.0` is absent, its value is
/// read from the alternate spelling `.1`, which may be a dotted path into the hoisted config.
/// Alias, don't rename: the config stays verbatim and an explicit standard-name field still wins.
/// `vision_config` stays nested: the vision glue owns that subtree. The `configs/` files are
/// verbatim HF copies; this normalization is the compiler's job, never an edit to the json.
pub fn normalize_hf_config(
    json: &serde_json::Value,
    aliases: &[(String, String)],
) -> serde_json::Value {
    let mut out = json.clone();
    let Some(obj) = out.as_object_mut() else {
        return out;
    };
    if let Some(text) = obj.get("text_config").cloned()
        && let Some(text_obj) = text.as_object()
    {
        for (k, v) in text_obj {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    if let Some(rope) = obj.get("rope_parameters").cloned()
        && let Some(rope_obj) = rope.as_object()
    {
        for (k, v) in rope_obj {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    // Resolve every alt against a snapshot taken AFTER the hoists, so paths may reference hoisted
    // subtrees.
    if !aliases.is_empty() {
        let snapshot = serde_json::Value::Object(obj.clone());
        for (std_key, alt_key) in aliases {
            if obj.contains_key(std_key) {
                continue;
            }
            if let Some(v) = json_path(&snapshot, alt_key) {
                obj.insert(std_key.clone(), v.clone());
            }
        }
    }
    out
}

/// The bounds a normalized config states at its top level: integers as they are, booleans as 0 / 1
/// (HF stores model config booleans like `norm_topk_prob` here; the bounds map is the only integer
/// table downstream consumers read).
pub fn extract_bounds(json: &serde_json::Value) -> BTreeMap<String, u64> {
    json.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| {
                    v.as_u64()
                        .map(|n| (k.clone(), n))
                        .or_else(|| v.as_bool().map(|b| (k.clone(), b as u64)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A flag's value for one config, as the config loader reads it: the normalized config's (under
/// the arch's `aliases`), else the arch's declared default, else 0.
pub fn flag_value(
    json: &serde_json::Value,
    aliases: &[(String, String)],
    bound_defaults: &[(String, u64)],
    flag: &str,
) -> u64 {
    (extract_bounds(&normalize_hf_config(json, aliases))
        .get(flag)
        .copied())
    .or_else(|| {
        (bound_defaults.iter())
            .find(|(k, _)| k == flag)
            .map(|(_, v)| *v)
    })
    .unwrap_or(0)
}

/// An arch's `params` `expr`: identifiers (earlier fields), integer literals, `+ - * /` (left
/// associative, `*` and `/` binding tighter; `/` is floor division), parentheses, and `sqrt(x)`
/// (an exact integer square root).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamExpr {
    Bound(String),
    Lit(u64),
    Add(Box<ParamExpr>, Box<ParamExpr>),
    Sub(Box<ParamExpr>, Box<ParamExpr>),
    Mul(Box<ParamExpr>, Box<ParamExpr>),
    Div(Box<ParamExpr>, Box<ParamExpr>),
    /// `a // b`: the floor quotient, a remainder dropped (`/` is exact).
    FloorDiv(Box<ParamExpr>, Box<ParamExpr>),
    Sqrt(Box<ParamExpr>),
}

impl ParamExpr {
    /// The expression `src`.
    pub fn parse(src: &str) -> Result<Self, String> {
        let toks = tokenize(src);
        let mut pos = 0;
        let e = sum(&toks, &mut pos)?;
        if pos < toks.len() {
            return Err(format!("trailing tokens after expression in `{src}`"));
        }
        Ok(e)
    }

    /// Its value under `env`.
    pub fn eval(&self, env: &BTreeMap<String, u64>) -> Result<u64, String> {
        Ok(match self {
            Self::Bound(t) => *(env.get(t))
                .ok_or_else(|| format!("unknown field `{t}` (declare it earlier in Params)"))?,
            Self::Lit(n) => *n,
            Self::Add(a, b) => a.eval(env)? + b.eval(env)?,
            Self::Sub(a, b) => {
                let (v, r) = (a.eval(env)?, b.eval(env)?);
                v.checked_sub(r)
                    .ok_or_else(|| format!("{v} - {r} underflows"))?
            }
            Self::Mul(a, b) => a.eval(env)? * b.eval(env)?,
            Self::Div(a, b) | Self::FloorDiv(a, b) => {
                let (v, r) = (a.eval(env)?, b.eval(env)?);
                if r == 0 {
                    return Err(format!("{v} / 0"));
                }
                v / r
            }
            Self::Sqrt(a) => {
                let v = a.eval(env)?;
                let r = (v as f64).sqrt().round() as u64;
                if r * r != v {
                    return Err(format!("sqrt({v}) is not an integer"));
                }
                r
            }
        })
    }
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

fn atom(toks: &[&str], pos: &mut usize) -> Result<ParamExpr, String> {
    let tok = toks.get(*pos).copied();
    *pos += 1;
    match tok {
        Some("(") => {
            let v = sum(toks, pos)?;
            if toks.get(*pos).copied() != Some(")") {
                return Err("expected `)`".to_string());
            }
            *pos += 1;
            Ok(v)
        }
        Some("sqrt") => {
            if toks.get(*pos).copied() != Some("(") {
                return Err("sqrt: expected `(`".to_string());
            }
            *pos += 1;
            let v = sum(toks, pos)?;
            if toks.get(*pos).copied() != Some(")") {
                return Err("sqrt: expected `)`".to_string());
            }
            *pos += 1;
            Ok(ParamExpr::Sqrt(Box::new(v)))
        }
        Some(t) if t.bytes().all(|b| b.is_ascii_digit()) => t
            .parse()
            .map(ParamExpr::Lit)
            .map_err(|e| format!("bad int `{t}`: {e}")),
        Some(t) => Ok(ParamExpr::Bound(t.to_string())),
        None => Err("unexpected end of expression".to_string()),
    }
}

fn prod(toks: &[&str], pos: &mut usize) -> Result<ParamExpr, String> {
    let mut v = atom(toks, pos)?;
    while let Some(op @ ("*" | "/")) = toks.get(*pos).copied() {
        *pos += 1;
        let floor = op == "/" && toks.get(*pos).copied() == Some("/");
        *pos += usize::from(floor);
        let r = atom(toks, pos)?;
        v = match (op, floor) {
            ("*", _) => ParamExpr::Mul(Box::new(v), Box::new(r)),
            (_, true) => ParamExpr::FloorDiv(Box::new(v), Box::new(r)),
            _ => ParamExpr::Div(Box::new(v), Box::new(r)),
        };
    }
    Ok(v)
}

fn sum(toks: &[&str], pos: &mut usize) -> Result<ParamExpr, String> {
    let mut v = prod(toks, pos)?;
    while let Some(op @ ("+" | "-")) = toks.get(*pos).copied() {
        *pos += 1;
        let r = prod(toks, pos)?;
        v = match op {
            "+" => ParamExpr::Add(Box::new(v), Box::new(r)),
            _ => ParamExpr::Sub(Box::new(v), Box::new(r)),
        };
    }
    Ok(v)
}

// ── arch.json ───────────────────────────────────────────────────────

/// One `params` entry of an arch.json: the bound it defines + where its value comes from.
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
    /// `from = "dotted.path"` into the verbatim config.json (numeric segments index arrays), with
    /// an optional `default = <int>` for configs that omit the key.
    From { path: String, default: Option<u64> },
    /// `expr = "a * b / c"` over previously-declared fields and integer literals ([`ParamExpr`]).
    Expr(String),
    /// `value = <int>` literal.
    Value(u64),
    /// `len_of = "dotted.path"`: the length of the config's array at the path (a DFlash2 drafter's
    /// `dflash_config.target_layer_ids`: how many of its target's layers it reads).
    LenOf(String),
}

/// An arch.json object of string keys to non-negative integers (`bound_defaults`), sorted.
pub fn object_u64_pairs(json: &serde_json::Value, key: &str) -> Result<Vec<(String, u64)>, String> {
    let Some(obj) = json.get(key) else {
        return Ok(Vec::new());
    };
    let obj = obj
        .as_object()
        .ok_or_else(|| format!("`{key}` must be an object"))?;
    let mut out: Vec<(String, u64)> = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let n = v
            .as_u64()
            .ok_or_else(|| format!("`{key}.{k}` must be a non-negative integer"))?;
        out.push((k.clone(), n));
    }
    out.sort();
    Ok(out)
}

/// An arch.json object of string keys to strings (`config_aliases`), sorted.
pub fn object_str_pairs(
    json: &serde_json::Value,
    key: &str,
) -> Result<Vec<(String, String)>, String> {
    let Some(obj) = json.get(key) else {
        return Ok(Vec::new());
    };
    let obj = obj
        .as_object()
        .ok_or_else(|| format!("`{key}` must be an object"))?;
    let mut out: Vec<(String, String)> = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let s = v
            .as_str()
            .ok_or_else(|| format!("`{key}.{k}` must be a string"))?;
        out.push((k.clone(), s.to_string()));
    }
    out.sort();
    Ok(out)
}

/// An arch.json array of strings (`checkpoint_of`), `None` when absent.
pub fn string_list(json: &serde_json::Value, key: &str) -> Result<Option<Vec<String>>, String> {
    json.get(key)
        .map(|v| {
            (v.as_array())
                .and_then(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect())
                .ok_or_else(|| format!("`{key}` must be an array of strings"))
        })
        .transpose()
}

/// An arch.json's `params` table, in order (an `expr` reads the fields above it).
pub fn parse_params(json: &serde_json::Value) -> Result<Vec<ParamField>, String> {
    let Some(v) = json.get("params") else {
        return Ok(Vec::new());
    };
    let arr = v.as_array().ok_or_else(|| {
        "`params` must be an array (order matters: `expr` reads earlier fields)".to_string()
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, f) in arr.iter().enumerate() {
        let obj = f
            .as_object()
            .ok_or_else(|| format!("`params[{i}]` must be an object"))?;
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("`params[{i}]` has no string `name`"))?
            .to_string();
        let replaces = match obj.get("replaces") {
            None => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| format!("`params[{i}].replaces` must be a bool"))?,
        };
        let len_of = obj.get("len_of");
        let exactly_one = || {
            format!(
                "`params[{i}]` ({name}) needs EXACTLY one of `from` / `expr` / `value` / `len_of`"
            )
        };
        let source = match (obj.get("from"), obj.get("expr"), obj.get("value")) {
            (None, None, None) if len_of.is_some() => ParamSource::LenOf(
                len_of
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("`params[{i}].len_of` must be a string"))?
                    .to_string(),
            ),
            _ if len_of.is_some() => return Err(exactly_one()),
            (Some(p), None, None) => {
                let path = p
                    .as_str()
                    .ok_or_else(|| format!("`params[{i}].from` must be a string"))?
                    .to_string();
                let default = match obj.get("default") {
                    None => None,
                    Some(d) => Some(
                        d.as_u64()
                            .ok_or_else(|| format!("`params[{i}].default` must be an int"))?,
                    ),
                };
                ParamSource::From { path, default }
            }
            (None, Some(e), None) => ParamSource::Expr(
                e.as_str()
                    .ok_or_else(|| format!("`params[{i}].expr` must be a string"))?
                    .to_string(),
            ),
            (None, None, Some(v)) => ParamSource::Value(
                v.as_u64()
                    .ok_or_else(|| format!("`params[{i}].value` must be an int"))?,
            ),
            _ => return Err(exactly_one()),
        };
        out.push(ParamField {
            name,
            source,
            replaces,
        });
    }
    Ok(out)
}

/// Whether a checkpoint config carries an in-checkpoint head (an arch with `checkpoint_of`): the
/// head replaces `num_hidden_layers` from a config path, and the config, normalized under the
/// head's `aliases`, states a positive count there.
pub fn carries_head(
    params: &[ParamField],
    aliases: &[(String, String)],
    json: &serde_json::Value,
) -> bool {
    let json = normalize_hf_config(json, aliases);
    params.iter().any(|f| {
        let ParamSource::From { path, .. } = &f.source else {
            return false;
        };
        let layers = json_path(&json, path).and_then(|v| v.as_u64());
        f.name == "num_hidden_layers" && f.replaces && layers.is_some_and(|n| n > 0)
    })
}

/// The arch.json facts the bounds read: its key aliases, its declared bound defaults, the archs
/// whose checkpoints it ships in, and its `params` table.
#[derive(Clone, Debug, Default)]
pub struct ArchFacts {
    pub config_aliases: Vec<(String, String)>,
    pub bound_defaults: Vec<(String, u64)>,
    pub checkpoint_of: Vec<String>,
    pub params: Vec<ParamField>,
}

/// An arch.json's [`ArchFacts`], by the readers the config loader uses for the same fields.
pub fn arch_facts(json: &serde_json::Value) -> Result<ArchFacts, String> {
    Ok(ArchFacts {
        config_aliases: object_str_pairs(json, "config_aliases")?,
        bound_defaults: object_u64_pairs(json, "bound_defaults")?,
        checkpoint_of: string_list(json, "checkpoint_of")?.unwrap_or_default(),
        params: parse_params(json)?,
    })
}
