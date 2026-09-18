// SPDX-License-Identifier: Apache-2.0
//! Differential tests: every runtime helper is checked against the REAL
//! interpreter, not against what we think Jinja does.
//!
//! ⛔ WHY DIFFERENTIAL AND NOT EXPECTED-VALUE. A hand-written expectation
//! encodes our belief about Jinja. When the belief is wrong the test passes and
//! the prompt is wrong — which is the exact failure mode this work exists to
//! prevent (a one-character difference silently degrades generation; see the
//! token 529 → 29966 note in crates/serving/api/src/chat_template.rs). Rendering
//! the same value through minijinja and comparing bytes cannot encode a wrong
//! belief.
//!
//! The oracle's configuration MUST match the serving runtime's `build_env`
//! (`trim_blocks` + `lstrip_blocks`), or these tests certify a different
//! language than the one production renders.

use minijinja::Environment;
use scratchy_chat_template_compiler::{
    Val, add, contains, eq, get_index, get_key, render_into, to_seq, truthy,
};
use serde_json::{Value as Json, json};

/// An interpreter configured exactly like `crates/serving/api/src/chat_template.rs::build_env`.
fn oracle() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env
}

/// What the interpreter prints for `{{ expr }}` with `v` bound to `x`.
fn oracle_render(expr: &str, v: &Json) -> String {
    let mut env = oracle();
    let src = format!("{{{{ {expr} }}}}");
    env.add_template_owned("t", src).unwrap();
    env.get_template("t")
        .unwrap()
        .render(minijinja::context! { v => v.clone() })
        .unwrap()
}

/// Every value kind, rendered by us and by the interpreter.
#[test]
fn display_parity() {
    let cases = [
        json!(true),
        json!(false),
        json!(0),
        json!(3),
        json!(-7),
        json!(2.0),
        json!(2.5),
        json!(""),
        json!("hi"),
        json!("héllo 🌍"),
        json!("with \"quotes\" and <tags> & 'ticks'"),
        json!(null),
        // containers — the cases that caught two real bugs in the first pass:
        // `null` renders `None` (not empty) and separators are ", " (not ",").
        json!([]),
        json!({}),
        json!([1, 2, 3]),
        json!(["a", "b"]),
        json!([1, "a", true, null]),
        json!([[1, 2], [3]]),
        json!({"b": 1, "a": 2}),
        json!({"k": "v"}),
        json!([{"role": "user"}]),
        json!(["it's"]),
        json!(["a\"b"]),
        json!({"k": ["nested", 1, false]}),
    ];
    for c in &cases {
        let mut ours = String::new();
        render_into(&mut ours, &Val::Ref(c));
        let theirs = oracle_render("v", c);
        assert_eq!(ours, theirs, "display parity for {c:?}");
    }
}

/// The single most likely silent break: Rust prints `true`, Jinja prints `True`.
#[test]
fn booleans_render_python_style() {
    let mut s = String::new();
    render_into(&mut s, &Val::Ref(&json!(true)));
    assert_eq!(
        s, "True",
        "if this is `true`, every boolean in every prompt is wrong"
    );
    assert_eq!(s, oracle_render("v", &json!(true)));
}

/// An absent variable renders as nothing and does not fail.
#[test]
fn undefined_renders_empty_and_does_not_error() {
    let mut ours = String::new();
    render_into(&mut ours, &Val::Undefined);
    assert_eq!(ours, "");

    let mut env = oracle();
    env.add_template("t", "[{{ nope }}]").unwrap();
    let theirs = env.get_template("t").unwrap().render(()).unwrap();
    assert_eq!(format!("[{ours}]"), theirs);
}

#[test]
fn truthiness_parity() {
    let cases = [
        json!(true),
        json!(false),
        json!(0),
        json!(1),
        json!(-1),
        json!(0.0),
        json!(""),
        json!("x"),
        json!([]),
        json!([1]),
        json!({}),
        json!({"a": 1}),
        json!(null),
    ];
    let mut env = oracle();
    env.add_template("t", "{% if v %}T{% else %}F{% endif %}")
        .unwrap();
    for c in &cases {
        let theirs = env
            .get_template("t")
            .unwrap()
            .render(minijinja::context! { v => c.clone() })
            .unwrap();
        let ours = if truthy(&Val::Ref(c)) { "T" } else { "F" };
        assert_eq!(ours, theirs, "truthiness for {c:?}");
    }
    // undefined is falsy
    assert!(!truthy(&Val::Undefined));
    let t = env.get_template("t").unwrap().render(()).unwrap();
    assert_eq!("F", t);
}

/// `in` is three-way and never errors — the construct that forces dynamic values.
#[test]
fn contains_dispatch_parity() {
    let cases: [(Json, &str); 5] = [
        (json!(["citations", "other"]), "citations"),
        (json!({"citations": true}), "citations"),
        (json!("citations"), "cit"),
        (json!(["nope"]), "citations"),
        (json!({"nope": 1}), "citations"),
    ];
    let mut env = oracle();
    env.add_template("t", "{% if needle in v %}T{% else %}F{% endif %}")
        .unwrap();
    for (hay, needle) in &cases {
        let theirs = env
            .get_template("t")
            .unwrap()
            .render(minijinja::context! { v => hay.clone(), needle => needle })
            .unwrap();
        let n = json!(needle);
        let ours = if contains(&Val::Ref(&n), &Val::Ref(hay)) {
            "T"
        } else {
            "F"
        };
        assert_eq!(ours, theirs, "`{needle} in {hay:?}`");
    }
    // over undefined => false, not an error
    let n = json!("x");
    assert!(!contains(&Val::Ref(&n), &Val::Undefined));
    let theirs = env
        .get_template("t")
        .unwrap()
        .render(minijinja::context! { needle => "x" })
        .unwrap();
    assert_eq!("F", theirs);
}

#[test]
fn add_parity_strings_numbers_sequences() {
    let mut env = oracle();
    env.add_template("t", "{{ a + b }}").unwrap();
    let cases: [(Json, Json); 4] = [
        (json!("x"), json!("y")),
        (json!(1), json!(2)),
        (json!(1.5), json!(2.0)),
        (json!([1, 2]), json!([3])),
    ];
    for (a, b) in &cases {
        let theirs = env
            .get_template("t")
            .unwrap()
            .render(minijinja::context! { a => a.clone(), b => b.clone() })
            .unwrap();
        let got = add(&Val::Ref(a), &Val::Ref(b)).expect("add should succeed");
        let mut ours = String::new();
        render_into(&mut ours, &got);
        assert_eq!(ours, theirs, "`{a:?} + {b:?}`");
    }
}

/// Adding incompatible kinds must fail — where the interpreter fails.
#[test]
fn add_errors_where_oracle_errors() {
    let mut env = oracle();
    env.add_template("t", "{{ a + b }}").unwrap();
    let theirs = env
        .get_template("t")
        .unwrap()
        .render(minijinja::context! { a => "x", b => 1 });
    let ours = add(&Val::Ref(&json!("x")), &Val::Ref(&json!(1)));
    assert_eq!(
        theirs.is_err(),
        ours.is_err(),
        "string + number: oracle err={:?}, ours err={:?}",
        theirs.is_err(),
        ours.is_err()
    );
}

#[test]
fn indexing_parity() {
    let msgs = json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]);
    let v = Val::Ref(&msgs);

    // messages[0]['role']
    let first = get_index(&v, 0);
    assert_eq!(get_key(&first, "role").as_str(), Some("user"));

    // negative index, Python-style
    let last = get_index(&v, -1);
    assert_eq!(get_key(&last, "role").as_str(), Some("assistant"));

    // out of range and missing key are Undefined, not errors
    assert!(get_index(&v, 99).is_undefined());
    assert!(get_key(&first, "tool_calls").is_undefined());

    // and the interpreter agrees that a missing key renders empty
    let mut env = oracle();
    env.add_template("t", "[{{ messages[0]['tool_calls'] }}]")
        .unwrap();
    let theirs = env
        .get_template("t")
        .unwrap()
        .render(minijinja::context! { messages => msgs.clone() })
        .unwrap();
    assert_eq!("[]", theirs);
}

#[test]
fn eq_bridges_str_and_json_string() {
    let j = json!("system");
    assert!(eq(&Val::Str("system".into()), &Val::Ref(&j)));
    assert!(!eq(&Val::Str("user".into()), &Val::Ref(&j)));
    // undefined equals only undefined
    assert!(eq(&Val::Undefined, &Val::Undefined));
    assert!(!eq(&Val::Undefined, &Val::Ref(&j)));
}

/// Iteration semantics: array/string/map all iterate (differently), null and
/// undefined are empty, numbers and bools are errors. All differential.
#[test]
fn to_seq_parity() {
    let mut env = oracle();
    env.add_template("t", "{% for x in v %}<{{ x }}>{% endfor %}")
        .unwrap();

    // Cases the interpreter renders successfully.
    let ok_cases = [
        json!([1, "a"]),
        json!("abc"),            // => three iterations, one per char
        json!({"b": 1, "a": 2}), // => keys, sorted
        json!([]),
        json!({}),
        json!(null), // => empty, no error
    ];
    for v in &ok_cases {
        let theirs = env
            .get_template("t")
            .unwrap()
            .render(minijinja::context! { v => v.clone() })
            .unwrap_or_else(|e| panic!("oracle failed on {v:?}: {e}"));
        let items = to_seq(&Val::Ref(v)).unwrap_or_else(|e| panic!("ours failed on {v:?}: {e}"));
        let mut ours = String::new();
        for it in &items {
            ours.push('<');
            render_into(&mut ours, it);
            ours.push('>');
        }
        assert_eq!(ours, theirs, "iterating {v:?}");
    }

    // undefined: empty loop, no error, on both sides.
    assert!(to_seq(&Val::Undefined).unwrap().is_empty());
    assert_eq!("", env.get_template("t").unwrap().render(()).unwrap());

    // Cases the interpreter REJECTS — we must reject them too.
    for v in [json!(5), json!(true)] {
        let theirs = env
            .get_template("t")
            .unwrap()
            .render(minijinja::context! { v => v.clone() });
        let ours = to_seq(&Val::Ref(&v));
        assert!(theirs.is_err(), "oracle should reject iterating {v:?}");
        assert!(ours.is_err(), "we must also reject iterating {v:?}");
    }
}
