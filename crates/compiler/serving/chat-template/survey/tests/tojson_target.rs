// SPDX-License-Identifier: Apache-2.0
//! The byte-exact target the DSL's tool serializers must hit.
//!
//! ⛔ THIS IS A CHARACTERISATION TEST, AND THAT IS THE POINT. It asserts what
//! Jinja's `tojson` does, pinned against the real interpreter, because SPEC.md §6.1
//! requires our serializers to reproduce it exactly and `serde_json::to_string`
//! does NOT. Reaching for serde_json looks obviously correct and is wrong for any
//! tool description containing markup or an apostrophe — entirely ordinary text.
//!
//! Two jobs:
//!   1. pin the target before the serializers exist, so the expectations are
//!      measured rather than remembered;
//!   2. fail loudly if a minijinja bump changes `tojson`, which would silently
//!      move the goalposts for every serializer at once.
//!
//! Hermetic: no network and no fixtures, unlike the survey binary beside it — so it
//! runs in CI on every commit. It lives here because this is where minijinja and
//! serde_json already sit side by side; it moves next to the serializers once the
//! compiler crate exists, and the expectations travel unchanged.

use minijinja::{Environment, Value, context};

fn tojson(filter: &str, v: Value) -> String {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_template_owned("t", format!("{{{{ v | {filter} }}}}"))
        .unwrap();
    env.get_template("t")
        .unwrap()
        .render(context! { v })
        .unwrap()
}

/// SPEC.md §6.1: exactly these four characters diverge from a standard JSON writer.
/// Both directions are asserted — the ones that DO differ and the ones that do NOT —
/// because a serializer that over-escapes is just as wrong as one that under-escapes.
#[test]
fn tojson_escapes_exactly_four_characters_more_than_serde_json() {
    // Derived from the codepoint rather than spelled out: this states the RULE
    // (a four-hex-digit \u escape) instead of four magic literals.
    fn jinja_escaped(ch: char) -> String {
        format!("\"\\u{:04x}\"", ch as u32)
    }

    for ch in ['<', '>', '&', '\''] {
        let s = ch.to_string();
        let got = tojson("tojson", Value::from(s.clone()));
        assert_eq!(got, jinja_escaped(ch), "tojson of {ch:?} changed");
        assert_ne!(
            got,
            serde_json::to_string(&s).unwrap(),
            "{ch:?} is supposed to DIVERGE from serde_json; if this fires, \
             serde_json changed and §6.1's table needs revisiting"
        );
    }

    // Everything else must agree, so a serializer cannot just escape aggressively.
    for ch in ['"', '/', '\\', '\n', '\t', '\u{7f}', '\u{2028}', 'é', '🎉'] {
        let s = ch.to_string();
        assert_eq!(
            tojson("tojson", Value::from(s.clone())),
            serde_json::to_string(&s).unwrap(),
            "tojson and serde_json are supposed to AGREE on {ch:?}"
        );
    }
}

/// SPEC.md §6.1: `indent=4` is not `serde_json::to_string_pretty` — 4 spaces vs 2.
/// One surveyed template needs the pretty form on one line and the compact form on
/// another, so a single model requires both modes to be right.
#[test]
fn tojson_indent_4_is_not_serde_json_pretty() {
    let v = serde_json::json!({"a": {"b": 1}});
    let pretty = tojson("tojson(indent=4)", Value::from_serialize(&v));

    assert!(
        pretty.contains("\n    \"a\""),
        "expected 4-space indent, got:\n{pretty}"
    );
    assert_ne!(
        pretty,
        serde_json::to_string_pretty(&v).unwrap(),
        "indent=4 is supposed to differ from serde_json's 2-space pretty printer"
    );

    // And the compact form must stay compact — no spaces after separators.
    let compact = tojson("tojson", Value::from_serialize(&v));
    assert_eq!(compact, r#"{"a":{"b":1}}"#);
}

/// SPEC.md §6.1: `tojson` sorts object keys regardless of insertion order.
///
/// ⛔ THE HAZARD THIS GUARDS IS NOT MINIJINJA, IT IS A CARGO FEATURE. `serde_json`
/// agrees with this today only because `preserve_order` is off. Cargo features unify
/// across the whole dependency graph, so one future transitive dependency enabling it
/// would flip serde_json to insertion order and silently change tool bytes for every
/// model. The serializers must therefore sort keys THEMSELVES rather than inherit
/// ordering from whatever map type happens to be in use — and this test states the
/// required ordering independently of serde_json, so it keeps passing either way.
#[test]
fn tojson_sorts_object_keys() {
    // Insertion order z, a, m.
    let v = Value::from_iter([("z", 1), ("a", 2), ("m", 3)]);
    assert_eq!(tojson("tojson", v), r#"{"a":2,"m":3,"z":1}"#);

    // Same via a parsed serde_json::Value, which is how a request body arrives.
    let parsed: serde_json::Value = serde_json::from_str(r#"{"z":1,"a":2,"m":3}"#).unwrap();
    assert_eq!(
        tojson("tojson", Value::from_serialize(&parsed)),
        r#"{"a":2,"m":3,"z":1}"#
    );
}

/// The two facts together, on a realistic tool declaration: markup and an
/// apostrophe in a description, pretty-printed. This is the shape granite emits,
/// and the string a serializer will be diffed against.
#[test]
fn realistic_tool_declaration_is_the_full_trap() {
    let tool = serde_json::json!({
        "name": "render",
        "description": "Wrap text in <b>bold</b> & emit <br> for breaks, don't forget 'quotes'",
    });
    let got = tojson("tojson(indent=4)", Value::from_serialize(&tool));

    assert_eq!(
        got,
        "{\n    \"description\": \"Wrap text in \\u003cb\\u003ebold\\u003c/b\\u003e \
         \\u0026 emit \\u003cbr\\u003e for breaks, don\\u0027t forget \
         \\u0027quotes\\u0027\",\n    \"name\": \"render\"\n}"
    );

    // The naive implementation, for contrast: same data, different bytes, and the
    // difference is invisible unless you look for it.
    assert_ne!(got, serde_json::to_string_pretty(&tool).unwrap());
}
