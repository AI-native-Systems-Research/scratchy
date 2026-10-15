// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HuggingFace's `tojson` chat-template filter.
//!
//! transformers replaces Jinja's `tojson` — which escapes `<`, `>`, `&` and
//! `'` for HTML — with its own
//! `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`,
//! a plain `json.dumps` with those arguments. Chat templates are written
//! against that filter, so the prompt bytes a model was trained on come from
//! it. This is a port of CPython's `json.encoder` over minijinja values.

use std::borrow::Cow;

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Error, ErrorKind};

#[cfg(test)]
mod hf_renders;

/// HuggingFace's parameters after the value, in positional order.
const PARAMS: [&str; 4] = ["ensure_ascii", "indent", "separators", "sort_keys"];

/// `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`.
pub(super) fn tojson(value: &Value, args: &[Value], kwargs: Kwargs) -> Result<Value, Error> {
    let dumps = Dumps::bind(args, &kwargs)?;
    let mut out = String::new();
    dumps.encode(&mut out, value, 0)?;
    Ok(Value::from(out))
}

/// `json.dumps` options, bound from the filter's arguments as Python binds
/// them.
struct Dumps {
    ensure_ascii: bool,
    /// One level of indentation; `None` keeps the output on one line.
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
    sort_keys: bool,
}

impl Dumps {
    fn bind(args: &[Value], kwargs: &Kwargs) -> Result<Self, Error> {
        if args.len() > PARAMS.len() {
            return Err(Error::new(
                ErrorKind::TooManyArguments,
                format!(
                    "tojson() takes from 1 to {} positional arguments but {} were given",
                    PARAMS.len() + 1,
                    args.len() + 1
                ),
            ));
        }
        let mut bound: [Option<Value>; PARAMS.len()] = Default::default();
        for (i, name) in PARAMS.into_iter().enumerate() {
            let keyword: Option<Value> = kwargs.get(name)?;
            bound[i] = match (args.get(i), keyword) {
                (Some(_), Some(_)) => {
                    return Err(type_error(format!(
                        "tojson() got multiple values for argument '{name}'"
                    )));
                }
                (positional, keyword) => positional.cloned().or(keyword),
            };
        }
        kwargs.assert_all_used()?;

        let [ensure_ascii, indent, separators, sort_keys] = bound;
        let indent = indent_unit(indent)?;
        let (item_separator, key_separator) = separator_pair(separators, indent.is_some())?;
        Ok(Self {
            ensure_ascii: ensure_ascii.is_some_and(|v| v.is_true()),
            indent,
            item_separator,
            key_separator,
            sort_keys: sort_keys.is_some_and(|v| v.is_true()),
        })
    }

    fn encode(&self, out: &mut String, value: &Value, depth: usize) -> Result<(), Error> {
        if let Some(s) = value.as_str() {
            self.push_string(out, s);
            return Ok(());
        }
        match value.kind() {
            ValueKind::None | ValueKind::Bool | ValueKind::Number => push_scalar(out, value),
            ValueKind::Seq => {
                let items = value.try_iter()?.collect::<Vec<_>>();
                self.push_container(out, ['[', ']'], depth, &items, |out, item, depth| {
                    self.encode(out, item, depth)
                })
            }
            ValueKind::Map => {
                let entries = self.entries(value)?;
                self.push_container(
                    out,
                    ['{', '}'],
                    depth,
                    &entries,
                    |out, (key, item), depth| {
                        self.push_string(out, key);
                        out.push_str(&self.key_separator);
                        self.encode(out, item, depth)
                    },
                )
            }
            ValueKind::Invalid => Err(type_error(value.to_string())),
            kind => Err(type_error(format!(
                "Object of type {kind} is not JSON serializable"
            ))),
        }
    }

    /// `_iterencode_list` / `_iterencode_dict`: an empty container stays
    /// `[]` / `{}`; otherwise items after the first follow the item
    /// separator, and with `indent` each item starts a new line one level
    /// deeper and the closing bracket returns to the container's level.
    fn push_container<T>(
        &self,
        out: &mut String,
        [open, close]: [char; 2],
        depth: usize,
        items: &[T],
        mut push_item: impl FnMut(&mut String, &T, usize) -> Result<(), Error>,
    ) -> Result<(), Error> {
        out.push(open);
        if !items.is_empty() {
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&self.item_separator);
                }
                self.push_newline(out, depth + 1);
                push_item(out, item, depth + 1)?;
            }
            self.push_newline(out, depth);
        }
        out.push(close);
        Ok(())
    }

    fn push_newline(&self, out: &mut String, depth: usize) {
        if let Some(unit) = &self.indent {
            out.push('\n');
            for _ in 0..depth {
                out.push_str(unit);
            }
        }
    }

    /// A map's `(key, value)` pairs in output order, keys already converted
    /// to the strings Python writes for them.
    fn entries(&self, map: &Value) -> Result<Vec<(String, Value)>, Error> {
        let mut entries = Vec::new();
        for key in map.try_iter()? {
            let item = map.get_item(&key)?;
            entries.push((key, item));
        }
        if self.sort_keys {
            sort_keys(&mut entries)?;
        }
        entries
            .into_iter()
            .map(|(key, item)| Ok((key_string(&key)?, item)))
            .collect()
    }

    /// `encode_basestring` / `encode_basestring_ascii`: `"`, `\` and C0
    /// controls are always escaped (`\n` `\r` `\t` `\b` `\f` short, the rest
    /// as `\u00xx`); `ensure_ascii` also escapes everything outside printable
    /// ASCII, as UTF-16 code units.
    fn push_string(&self, out: &mut String, s: &str) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if c < ' ' || (self.ensure_ascii && !(' '..='~').contains(&c)) => {
                    for unit in c.encode_utf16(&mut [0; 2]) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// Python's indentation unit: a string is used as-is; an integer `n`, bools
/// included, is `' ' * n`, which is empty for `n <= 0` but still breaks lines.
fn indent_unit(indent: Option<Value>) -> Result<Option<String>, Error> {
    let Some(indent) = indent.filter(|v| !v.is_none()) else {
        return Ok(None);
    };
    if let Some(unit) = indent.as_str() {
        return Ok(Some(unit.to_owned()));
    }
    let width = match indent.kind() {
        ValueKind::Bool => usize::from(indent.is_true()),
        // `' ' * n` is empty for every negative `n`.
        ValueKind::Number if indent.is_integer() => {
            usize::try_from(i64::try_from(indent)?).unwrap_or(0)
        }
        kind => {
            return Err(type_error(format!(
                "can't multiply sequence by non-int of type '{kind}'"
            )));
        }
    };
    Ok(Some(" ".repeat(width)))
}

/// The `(item, key)` separators: as given, else Python's `(', ', ': ')`,
/// whose item separator drops its space once `indent` breaks lines.
fn separator_pair(separators: Option<Value>, indented: bool) -> Result<(String, String), Error> {
    let Some(pair) = separators.filter(|v| !v.is_none()) else {
        let item = if indented { "," } else { ", " };
        return Ok((item.to_owned(), ": ".to_owned()));
    };
    let parts = pair.try_iter()?.collect::<Vec<_>>();
    let [item, key] = parts.as_slice() else {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("separators must unpack to 2 values, got {}", parts.len()),
        ));
    };
    let separator = |part: &Value| {
        part.as_str()
            .map(str::to_owned)
            .ok_or_else(|| type_error(format!("separator must be str, not {}", part.kind())))
    };
    Ok((separator(item)?, separator(key)?))
}

/// `sorted(dct.items())`: Python orders string keys by code point and
/// numeric keys (bools as `0`/`1`) by value, and refuses to compare one kind
/// with another.
fn sort_keys(entries: &mut [(Value, Value)]) -> Result<(), Error> {
    let numeric = |key: &Value| matches!(key.kind(), ValueKind::Number | ValueKind::Bool);
    if entries.iter().all(|(key, _)| key.as_str().is_some()) {
        entries.sort_by(|(a, _), (b, _)| a.as_str().cmp(&b.as_str()));
    } else if entries.iter().all(|(key, _)| numeric(key)) {
        entries.sort_by_key(|(key, _)| match key.kind() {
            ValueKind::Bool => Value::from(u8::from(key.is_true())),
            _ => key.clone(),
        });
    } else if entries.len() > 1 {
        return Err(type_error(
            "'<' not supported between keys of different types",
        ));
    }
    Ok(())
}

/// The string Python writes for an object key: strings as-is; floats, ints,
/// bools and `None` as their JSON text.
fn key_string(key: &Value) -> Result<String, Error> {
    if let Some(s) = key.as_str() {
        return Ok(s.to_owned());
    }
    let mut out = String::new();
    match key.kind() {
        ValueKind::None | ValueKind::Bool | ValueKind::Number => push_scalar(&mut out, key)?,
        kind => {
            return Err(type_error(format!(
                "keys must be str, int, float, bool or None, not {kind}"
            )));
        }
    }
    Ok(out)
}

/// `null`, `true` / `false`, an integer's digits, or `float.__repr__`.
fn push_scalar(out: &mut String, value: &Value) -> Result<(), Error> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        _ if value.is_integer() => out.push_str(&value.to_string()),
        _ => push_float(out, f64::try_from(value.clone())?),
    }
    Ok(())
}

/// `float.__repr__` as `json.dumps` writes it (`allow_nan=True`): the
/// shortest digits that round-trip, positional while the decimal exponent is
/// in `-4..16`, else `d.ddde±XX`.
fn push_float(out: &mut String, f: f64) {
    if f.is_nan() {
        return out.push_str("NaN");
    }
    if f.is_infinite() {
        return out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
    }
    // Rust's `{:e}` is the shortest round-trip rendering too, e.g. `-1.25e-7`.
    let scientific = format!("{f:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` always writes an exponent");
    let exponent: i32 = exponent
        .parse()
        .expect("`{:e}` always writes an integer exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(magnitude) => ("-", magnitude),
        None => ("", mantissa),
    };
    let digits = mantissa.replace('.', "");
    out.push_str(sign);
    if (-4..16).contains(&exponent) {
        // How many digits stand before the decimal point.
        let point = exponent + 1;
        match usize::try_from(point) {
            Ok(point) if point > 0 && point < digits.len() => {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
            Ok(point) if point > 0 => {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', point - digits.len()));
                out.push_str(".0");
            }
            _ => {
                out.push_str("0.");
                out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
                out.push_str(&digits);
            }
        }
    } else {
        let (lead, rest) = digits.split_at(1);
        out.push_str(lead);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        out.push_str(&format!("e{exponent_sign}{:02}", exponent.unsigned_abs()));
    }
}

fn type_error(detail: impl Into<Cow<'static, str>>) -> Error {
    Error::new(ErrorKind::InvalidOperation, detail)
}

#[cfg(test)]
mod tests {
    use super::hf_renders::{HF_TOJSON_ERRORS, HF_TOJSON_RENDERS};
    use crate::chat_template::ChatTemplate;

    fn render(template: &str, tools: &str) -> Result<String, crate::error::ServeError> {
        let tools: serde_json::Value = serde_json::from_str(tools).unwrap();
        ChatTemplate::new(template.to_owned())?.apply(&[], false, Some(&tools))
    }

    #[test]
    fn test_tojson_renders_what_huggingface_renders() {
        for &(template, tools, expected) in HF_TOJSON_RENDERS {
            assert_eq!(
                render(template, tools).unwrap(),
                expected,
                "{template} over {tools}"
            );
        }
    }

    #[test]
    fn test_tojson_refuses_what_huggingface_refuses() {
        for &(template, tools) in HF_TOJSON_ERRORS {
            assert!(render(template, tools).is_err(), "{template} over {tools}");
        }
    }
}
