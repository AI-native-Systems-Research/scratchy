// SPDX-License-Identifier: Apache-2.0
//! Compile a HuggingFace Jinja chat template into Rust source that renders it.
//!
//! ⛔ THIS IS A LEAF CRATE. It must never depend on any `scratchy-*` crate. The
//! whole point is that it is publishable on its own: "compile a chat template
//! to Rust" is useful to anyone building a Rust LLM server. Two consequences
//! that are easy to violate by accident:
//!   - it does not know about `inventory`, models, arches, or serving; and
//!   - it does not emit a registration. It emits a plain `fn render` — wrapping
//!     that for a registry is the caller's job.
//!
//! Renaming this crate must therefore be a one-line change. If a rename would
//! require touching anything in here, leaf-ness has already been lost.
//!
//! ## Two halves
//!
//! * default features — the **runtime**: the value model and the operator
//!   semantics the generated code calls into. This is the module below.
//! * `compile` — the **build-time** half: parse a template, walk minijinja's
//!   AST, emit Rust source. See [`compile`].
//!
//! ## Why most of this file is operator semantics
//!
//! The obvious guess is that compiling a template is mostly about text
//! substitution. It is not. The hard part is that Jinja has its own rules for
//! what values *mean*, and Rust's differ. `{{ true }}` renders `True`, not
//! `true`. `2.0` renders `2.0`, not `2`. A missing variable renders as the empty
//! string instead of failing. `'x' in y` answers sensibly whether `y` is a list,
//! a map, a string, or absent.
//!
//! Getting any of these wrong is not a cosmetic bug. From
//! `crates/serving/api/src/chat_template.rs`: for TinyLlama a stray newline
//! moves `<|user|>`'s leading `<` from token 529 to 29966, and "the model has
//! effectively never seen that input, so generation produces nonsense
//! fragments." A one-character difference silently degrades output. So every
//! function here exists to match the interpreter exactly, and the tests assert
//! byte-identity against the real interpreter rather than against expectations.

#[cfg(feature = "compile")]
pub mod compile;

use std::borrow::Cow;
use std::fmt::Write as _;

use serde_json::Value as Json;

/// Re-exported so GENERATED CODE has exactly one crate to name.
///
/// ⛔ LOAD-BEARING FOR CONSUMERS. The emitted file refers to JSON types through
/// this re-export, never through `::serde_json`. Otherwise every crate that
/// `include!`s generated code would need its own `serde_json` dependency — a
/// hidden requirement that shows up as a confusing "cannot find `serde_json` in
/// the crate root" error in a file the consumer never wrote. Generated code must
/// only name paths it can guarantee.
pub use serde_json;

/// A Jinja value.
///
/// `Undefined` is a distinct state from JSON `null`: Jinja renders a missing
/// variable as the empty string and treats it as falsy, but `null` is a value
/// you can compare against. Collapsing them breaks `is defined`.
#[derive(Clone, Debug, Default)]
pub enum Val<'a> {
    #[default]
    Undefined,
    Ref(&'a Json),
    Owned(Json),
    /// A borrowed string. Not redundant with `Ref(Json::String)`: the emitted
    /// code concatenates with `+`, and this lets the common case avoid
    /// allocating a `Json::String` for every intermediate.
    Str(Cow<'a, str>),
}

impl<'a> Val<'a> {
    pub fn json(&self) -> Option<&Json> {
        match self {
            Val::Ref(j) => Some(j),
            Val::Owned(j) => Some(j),
            Val::Undefined | Val::Str(_) => None,
        }
    }

    pub fn is_undefined(&self) -> bool {
        matches!(self, Val::Undefined)
    }

    /// The `str` view, when this value is textually a string.
    ///
    /// ⛔ NOT a general stringification — `as_str` on a number returns `None`,
    /// matching Jinja, where `'a' in 5` is an error rather than a substring
    /// test on `"5"`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            Val::Ref(Json::String(s)) => Some(s),
            Val::Owned(Json::String(s)) => Some(s),
            _ => None,
        }
    }
}

/// One compiled template: its identity, the exact bytes it was compiled from,
/// and the generated renderer.
///
/// ⛔ THE `source` FIELD IS THE DRIFT GATE, AND IT IS THE WHOLE SOURCE, NOT A HASH.
/// A vendored template can silently diverge from the one the served model
/// actually ships — a model updates its `tokenizer_config.json`, or the operator
/// passes `--chat-template`. Using the compiled renderer against a different
/// template would produce a confidently wrong prompt.
///
/// A hash was the obvious choice and is the wrong one here. Templates are
/// 368 B–4 KB, so embedding the source costs nothing measurable, and it buys
/// three things a digest cannot: an exact comparison with no collision question,
/// no dependency on a hash crate (keeping this crate's default half dependency-
/// light), and no reliance on a digest computed by the BUILD host matching one
/// computed on the TARGET. It also lets the caller log *what* differs.
pub struct CompiledTemplate {
    /// Where the template was vendored from. For diagnostics only.
    pub name: &'static str,
    /// The exact bytes this renderer was generated from.
    pub source: &'static str,
    /// The generated renderer.
    pub render: fn(&Json) -> Result<String, TemplateError>,
}

impl CompiledTemplate {
    /// Whether this compiled renderer is valid for `resolved`.
    ///
    /// Byte equality, deliberately: a differing trailing newline or line ending
    /// changes the rendered prompt and therefore the tokenization, so "close
    /// enough" is not a useful answer. A mismatch means fall back to the
    /// interpreter — never render anyway.
    pub fn matches(&self, resolved: &str) -> bool {
        self.source == resolved
    }
}

/// Errors the generated code can raise.
///
/// The generated `render` returns `Result`, not `String`, because Jinja can
/// fail: `raise_exception` is a template-callable, and filters reject values of
/// the wrong shape. A compiled template must fail exactly where the interpreter
/// fails, so "errors where the oracle errors" is part of the test bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// `{{ raise_exception("...") }}`
    Raised(String),
    /// A filter or operator got a value it cannot accept.
    BadValue(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::Raised(m) => write!(f, "{m}"),
            TemplateError::BadValue(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for TemplateError {}

// ---------------------------------------------------------------------------
// Display parity
// ---------------------------------------------------------------------------

/// Append `v` to `out` the way Jinja's `{{ ... }}` would.
///
/// ⛔ EVERY ARM HERE IS A PARITY DECISION, NOT A STYLE CHOICE:
///   - `true` renders `True` and `false` renders `False` — Jinja is
///     Python-flavoured, and Rust's `{}` would emit `true` and shift the
///     tokenization;
///   - `undefined` renders as nothing at all, and does NOT error;
///   - `null` renders `None`, which is NOT the same as undefined's empty string;
///   - floats keep a `.0`: `2.0` renders `2.0`, not `2`. Rust's `{}` on `f64`
///     already does this, but the tests assert it so a later "simplification"
///     cannot pass silently;
///   - lists and maps get Python-`repr`-flavoured output — see `repr_into`.
pub fn render_into(out: &mut String, v: &Val<'_>) {
    match v {
        Val::Undefined => {}
        Val::Str(s) => out.push_str(s),
        Val::Ref(j) => render_json_into(out, j),
        Val::Owned(j) => render_json_into(out, j),
    }
}

fn render_json_into(out: &mut String, j: &Json) {
    match j {
        // ⛔ `None`, NOT the empty string. Verified against the interpreter.
        // This is distinct from `undefined`, which really is empty — collapsing
        // the two changes every prompt that prints a null field.
        Json::Null => out.push_str("None"),
        Json::Bool(true) => out.push_str("True"),
        Json::Bool(false) => out.push_str("False"),
        // A TOP-LEVEL string prints bare. Inside a container it is quoted —
        // see `repr_into`. Two modes, deliberately not one function.
        Json::String(s) => out.push_str(s),
        Json::Number(n) => {
            let _ = write!(out, "{n}");
        }
        Json::Array(_) | Json::Object(_) => repr_into(out, j),
    }
}

/// Container rendering — Python-`repr`-flavoured, which is what the interpreter
/// does for a bare `{{ some_list }}`.
///
/// ⛔ NOT `serde_json::to_string`. Three differences, all verified against the
/// interpreter and all silent if you get them wrong:
///   - separators are `", "` and `": "`, not `,` and `:` — `[1, 2, 3]`, not
///     `[1,2,3]`;
///   - scalars use Python names inside containers too: `True`, `False`, `None`;
///   - strings ARE quoted here, unlike at top level.
///
/// Map key order is `serde_json`'s (a `BTreeMap`, so sorted), which matches the
/// interpreter because the serving runtime hands it the same `serde_json` values.
fn repr_into(out: &mut String, j: &Json) {
    match j {
        Json::Null => out.push_str("None"),
        Json::Bool(true) => out.push_str("True"),
        Json::Bool(false) => out.push_str("False"),
        Json::Number(n) => {
            let _ = write!(out, "{n}");
        }
        // Quoted + escaped. Delegated to serde_json so escaping of `"`, `\\` and
        // control characters is not re-implemented here.
        Json::String(_) => {
            let _ = write!(out, "{j}");
        }
        Json::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr_into(out, item);
            }
            out.push(']');
        }
        Json::Object(o) => {
            out.push('{');
            for (i, (k, v)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}", Json::String(k.clone()));
                out.push_str(": ");
                repr_into(out, v);
            }
            out.push('}');
        }
    }
}

// ---------------------------------------------------------------------------
// Truthiness
// ---------------------------------------------------------------------------

/// Jinja truthiness. Empty string, empty list, empty map, `0`, `false`, `null`
/// and `undefined` are all falsy — matching Python, not Rust.
pub fn truthy(v: &Val<'_>) -> bool {
    match v {
        Val::Undefined => false,
        Val::Str(s) => !s.is_empty(),
        Val::Ref(j) => json_truthy(j),
        Val::Owned(j) => json_truthy(j),
    }
}

fn json_truthy(j: &Json) -> bool {
    match j {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::String(s) => !s.is_empty(),
        Json::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// `v[key]` where `key` is a string — map lookup.
///
/// A miss yields `Undefined` rather than an error, which is what lets templates
/// write `{% if message['tool_calls'] %}` against messages that have no such
/// key.
/// ⛔ SIGNATURE IS LOAD-BEARING. Returns `Val<'a>`, tied to the ROOT value's
/// lifetime rather than to the borrow of `v`, so generated code can chain
/// `messages[0]['role']` without binding a temporary at every step. `Ref`
/// forwards the reference; `Owned` has to clone the child, since we cannot hand
/// out a reference into a value we only borrowed.
pub fn get_key<'a>(v: &Val<'a>, key: &str) -> Val<'a> {
    match v {
        Val::Ref(Json::Object(o)) => o.get(key).map(Val::Ref).unwrap_or(Val::Undefined),
        Val::Owned(Json::Object(o)) => o
            .get(key)
            .cloned()
            .map(Val::Owned)
            .unwrap_or(Val::Undefined),
        _ => Val::Undefined,
    }
}

/// `v[i]` where `i` is an integer — sequence index. Negative indices count from
/// the end, as in Python. Out of range yields `Undefined`.
pub fn get_index<'a>(v: &Val<'a>, i: i64) -> Val<'a> {
    let len = match v.json() {
        Some(Json::Array(a)) => a.len(),
        _ => return Val::Undefined,
    };
    let idx = if i < 0 { len as i64 + i } else { i };
    if idx < 0 || idx as usize >= len {
        return Val::Undefined;
    }
    let idx = idx as usize;
    match v {
        Val::Ref(Json::Array(a)) => Val::Ref(&a[idx]),
        Val::Owned(Json::Array(a)) => Val::Owned(a[idx].clone()),
        _ => Val::Undefined,
    }
}

/// `v[k]` where `k` is itself a value — the general form the emitter uses when
/// the subscript is not a literal. Dispatches on the key's kind so
/// `messages[i]` and `message['role']` both go through one emitted shape.
pub fn get_item<'a>(v: &Val<'a>, k: &Val<'_>) -> Val<'a> {
    if let Some(key) = k.as_str() {
        return get_key(v, key);
    }
    match k.json() {
        Some(Json::Number(n)) => match n.as_i64() {
            Some(i) => get_index(v, i),
            None => Val::Undefined,
        },
        _ => Val::Undefined,
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// `a + b`.
///
/// Dispatches on kind the way the interpreter does: strings concatenate,
/// numbers add, sequences concatenate. Anything else is an error rather than a
/// silent stringification — a template relying on coercion we do not implement
/// should fail loudly at the point of divergence, not produce a subtly wrong
/// prompt.
pub fn add<'a>(a: &Val<'_>, b: &Val<'_>) -> Result<Val<'a>, TemplateError> {
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        let mut s = String::with_capacity(x.len() + y.len());
        s.push_str(x);
        s.push_str(y);
        return Ok(Val::Str(Cow::Owned(s)));
    }
    match (a.json(), b.json()) {
        (Some(Json::Number(x)), Some(Json::Number(y))) => match (x.as_i64(), y.as_i64()) {
            (Some(i), Some(j)) => Ok(Val::Owned(Json::from(i + j))),
            _ => {
                let (x, y) = (x.as_f64().unwrap_or(0.0), y.as_f64().unwrap_or(0.0));
                Ok(Val::Owned(serde_json::json!(x + y)))
            }
        },
        (Some(Json::Array(x)), Some(Json::Array(y))) => {
            let mut v = x.clone();
            v.extend(y.iter().cloned());
            Ok(Val::Owned(Json::Array(v)))
        }
        _ => Err(TemplateError::BadValue(
            "unsupported operand types for +".to_string(),
        )),
    }
}

/// Structural equality, with the string/`Json::String` split bridged so
/// `Val::Str("a") == Val::Ref(Json::String("a"))`.
pub fn eq(a: &Val<'_>, b: &Val<'_>) -> bool {
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        return x == y;
    }
    match (a, b) {
        (Val::Undefined, Val::Undefined) => true,
        (Val::Undefined, _) | (_, Val::Undefined) => false,
        _ => a.json() == b.json(),
    }
}

pub fn ne(a: &Val<'_>, b: &Val<'_>) -> bool {
    !eq(a, b)
}

/// `needle in haystack`.
///
/// ⛔ THREE-WAY AND NEVER AN ERROR. Verified against the interpreter: over a
/// list it is membership, over a map it is key membership, over a string it is
/// substring, and over `undefined` it is `false`. granite's
/// `'citations' in controls` is documented by IBM in both list and map form, so
/// this cannot be resolved from the template text — the dispatch has to happen
/// at runtime. This function is the reason values stay dynamic.
pub fn contains(needle: &Val<'_>, haystack: &Val<'_>) -> bool {
    if let Some(hs) = haystack.as_str() {
        return needle.as_str().map(|n| hs.contains(n)).unwrap_or(false);
    }
    match haystack.json() {
        Some(Json::Array(a)) => a.iter().any(|item| eq(needle, &Val::Ref(item))),
        Some(Json::Object(o)) => needle.as_str().map(|k| o.contains_key(k)).unwrap_or(false),
        _ => false,
    }
}

/// A value as a sequence, for `{% for %}`.
///
/// Returns an owned `Vec` rather than an iterator so the emitted loop needs no
/// lifetime gymnastics and knows its length up front — `loop.last` needs the
/// length before the first iteration.
///
/// ⛔ FIVE BEHAVIOURS, ALL VERIFIED AGAINST THE INTERPRETER, NONE GUESSABLE:
///   - array: its items;
///   - string: its CHARACTERS — `"abc"` gives three iterations, not one;
///   - map: its KEYS, in `serde_json` order (sorted);
///   - null and undefined: empty, and NOT an error;
///   - number and bool: an ERROR ("invalid operation").
///
/// An earlier version of this returned empty for all non-arrays. That silently
/// turned `{% for %}` over a string or a map into a no-op, and silently accepted
/// two cases the interpreter rejects.
pub fn to_seq<'a>(v: &Val<'a>) -> Result<Vec<Val<'a>>, TemplateError> {
    fn chars_of(s: &str) -> Vec<Val<'static>> {
        s.chars()
            .map(|c| Val::Owned(Json::String(c.to_string())))
            .collect()
    }
    fn keys_of(o: &serde_json::Map<String, Json>) -> Vec<Val<'static>> {
        o.keys()
            .map(|k| Val::Owned(Json::String(k.clone())))
            .collect()
    }
    match v {
        Val::Undefined => Ok(Vec::new()),
        Val::Str(s) => Ok(chars_of(s)),
        Val::Ref(j) => match j {
            Json::Null => Ok(Vec::new()),
            Json::Array(a) => Ok(a.iter().map(Val::Ref).collect()),
            Json::String(s) => Ok(chars_of(s)),
            Json::Object(o) => Ok(keys_of(o)),
            Json::Number(_) | Json::Bool(_) => Err(not_iterable(j)),
        },
        Val::Owned(j) => match j {
            Json::Null => Ok(Vec::new()),
            Json::Array(a) => Ok(a.iter().cloned().map(Val::Owned).collect()),
            Json::String(s) => Ok(chars_of(s)),
            Json::Object(o) => Ok(keys_of(o)),
            Json::Number(_) | Json::Bool(_) => Err(not_iterable(j)),
        },
    }
}

fn not_iterable(j: &Json) -> TemplateError {
    let kind = match j {
        Json::Number(_) => "number",
        Json::Bool(_) => "boolean",
        _ => "value",
    };
    TemplateError::BadValue(format!("cannot iterate over {kind}"))
}

/// `raise_exception(msg)`.
pub fn raise<T>(msg: &Val<'_>) -> Result<T, TemplateError> {
    let mut s = String::new();
    render_into(&mut s, msg);
    Err(TemplateError::Raised(s))
}
