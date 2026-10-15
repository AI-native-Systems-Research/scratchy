// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HuggingFace's `{% generation %}` block.
//!
//! transformers' chat-template environment carries an `AssistantTracker`
//! extension whose `{% generation %}…{% endgeneration %}` parses to a Jinja
//! `CallBlock`: the body renders as a caller macro, in its own scope, and the
//! tag outputs `caller()` unchanged — the span it records only feeds
//! `return_assistant_tokens_mask`, which serving never asks for. minijinja has
//! call blocks but no custom statements, so before the template is compiled
//! each tag is respelled as the call block it parses to, `{% call
//! __hf_generation() %}` / `{% endcall %}` with its whitespace markers kept,
//! and [`render_caller`] is registered under [`CALLEE`].

use std::borrow::Cow;
use std::ops::Range;

use minijinja::value::{Kwargs, Value};
use minijinja::{Error, State};

/// The function each respelled `{% generation %}` calls.
pub(super) const CALLEE: &str = "__hf_generation";

/// `AssistantTracker._generation_support`: the block's output is its body's.
pub(super) fn render_caller(state: &mut State, kwargs: Kwargs) -> Result<Value, Error> {
    let caller: Value = kwargs.get("caller")?;
    kwargs.assert_all_used()?;
    caller.call(state, &[])
}

/// `template` with every bare `{% generation %}` / `{% endgeneration %}` tag
/// respelled as a call block; borrowed when it has none. Tags inside string
/// literals, comments and `{% raw %}` blocks are text, not statements, and
/// stay as they are; a `generation` tag with arguments is left for the
/// compiler to reject, as Jinja does.
pub(super) fn respell(template: &str) -> Cow<'_, str> {
    let mut out = String::new();
    let mut copied = 0;
    let mut at = 0;
    while let Some(offset) = template[at..].find('{') {
        let start = at + offset;
        at = match template.as_bytes().get(start + 1) {
            Some(b'#') => template[start..]
                .find("#}")
                .map_or(template.len(), |end| start + end + 2),
            Some(b'{') => past_close(template, start + 2, "}}"),
            Some(b'%') => {
                let end = past_close(template, start + 2, "%}");
                let tag = block_tag(template, start..end);
                match (&template[tag.keyword.clone()], tag.bare) {
                    ("raw", true) => past_endraw(template, end),
                    (keyword @ ("generation" | "endgeneration"), true) => {
                        out.push_str(&template[copied..tag.keyword.start]);
                        if keyword == "generation" {
                            out.push_str(&format!("call {CALLEE}()"));
                        } else {
                            out.push_str("endcall");
                        }
                        copied = tag.keyword.end;
                        end
                    }
                    _ => end,
                }
            }
            _ => start + 1,
        };
    }
    if copied == 0 {
        return Cow::Borrowed(template);
    }
    out.push_str(&template[copied..]);
    Cow::Owned(out)
}

/// A block tag `{%[-+] keyword … [-+]%}`: where its keyword is, and whether
/// the keyword is all the tag holds.
struct BlockTag {
    keyword: Range<usize>,
    bare: bool,
}

fn block_tag(template: &str, tag: Range<usize>) -> BlockTag {
    let body = &template[tag.start + 2..tag.end];
    let body_start = tag.start + 2;
    let opened = body.strip_prefix(['-', '+']).unwrap_or(body);
    let keyword_start = body_start + (body.len() - opened.trim_start().len());
    let keyword_len = template[keyword_start..tag.end]
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(tag.end - keyword_start);
    let keyword = keyword_start..keyword_start + keyword_len;
    let bare = template[keyword.end..tag.end]
        .strip_suffix("%}")
        .map(|rest| rest.strip_suffix(['-', '+']).unwrap_or(rest))
        .is_some_and(|rest| rest.trim().is_empty());
    BlockTag { keyword, bare }
}

/// The index just past `close` at or after `from`, stepping over string
/// literals; the template's end when it never closes (the compiler reports
/// that).
fn past_close(template: &str, from: usize, close: &str) -> usize {
    let bytes = template.as_bytes();
    let mut quote = None;
    let mut at = from;
    while at < bytes.len() {
        match (quote, bytes[at]) {
            (Some(_), b'\\') => at += 1,
            (Some(open), byte) if byte == open => quote = None,
            (Some(_), _) => {}
            (None, byte @ (b'"' | b'\'')) => quote = Some(byte),
            (None, _) if template[at..].starts_with(close) => return at + close.len(),
            (None, _) => {}
        }
        at += 1;
    }
    bytes.len()
}

/// The index just past the `{% endraw %}` closing a raw block whose opening
/// tag ends at `from`.
fn past_endraw(template: &str, from: usize) -> usize {
    let mut at = from;
    while let Some(offset) = template[at..].find("{%") {
        let start = at + offset;
        let end = past_close(template, start + 2, "%}");
        let tag = block_tag(template, start..end);
        if tag.bare && &template[tag.keyword] == "endraw" {
            return end;
        }
        at = start + 2;
    }
    template.len()
}

#[cfg(test)]
mod tests {
    use super::respell;

    #[test]
    fn test_respell_turns_bare_tags_into_the_call_block_and_keeps_markers() {
        assert_eq!(
            respell("{%- generation -%}x{%+endgeneration%}"),
            "{%- call __hf_generation() -%}x{%+endcall%}"
        );
    }

    #[test]
    fn test_respell_leaves_text_that_only_looks_like_a_tag() {
        for template in [
            r#"{{ "{% generation %}" }}"#,
            "{# {% generation %} #}",
            "{% raw %}{% generation %}{% endraw %}",
            "{% set s = '%}{% generation %}' %}",
            "{% generation foo %}",
            "{% generations %}",
        ] {
            assert_eq!(respell(template), template);
        }
    }
}
