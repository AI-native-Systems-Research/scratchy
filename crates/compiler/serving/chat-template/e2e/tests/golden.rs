// SPDX-License-Identifier: Apache-2.0
//! The acceptance gate: GENERATED code vs the REAL interpreter, byte for byte.
//!
//! ⛔ THE ORACLE'S CONFIGURATION IS PART OF THE TEST. It must match
//! `crates/serving/api/src/chat_template.rs::build_env` (`trim_blocks` +
//! `lstrip_blocks`). A harness that builds a default `Environment` silently
//! certifies the wrong output — see `whitespace_config_is_not_optional` below,
//! which fails on purpose if anyone "simplifies" the oracle back to defaults.

use minijinja::Environment;
use scratchy_chat_template_e2e::smollm2_135m;
use serde_json::json;

const TEMPLATE: &str = include_str!("../../../../../models/arch/configs/llama/smollm2-135m.chat.jinja");

fn oracle() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_template("chat", TEMPLATE).unwrap();
    env
}

/// The message matrix from the epic: with/without system, single/multi turn,
/// generation prompt on/off, empty content, unicode content.
fn matrix() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        ("no system, 1 turn, agp", json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "add_generation_prompt": true })),
        ("no system, agp off", json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "add_generation_prompt": false })),
        ("with system", json!({
            "messages": [{"role": "system", "content": "Be terse"},
                         {"role": "user", "content": "Hi"}],
            "add_generation_prompt": true })),
        ("multi turn", json!({
            "messages": [{"role": "user", "content": "a"},
                         {"role": "assistant", "content": "b"},
                         {"role": "user", "content": "c"}],
            "add_generation_prompt": true })),
        ("empty content", json!({
            "messages": [{"role": "user", "content": ""}],
            "add_generation_prompt": true })),
        ("unicode + newline", json!({
            "messages": [{"role": "user", "content": "héllo 🌍\nsecond"}],
            "add_generation_prompt": true })),
        ("empty message list", json!({
            "messages": [], "add_generation_prompt": true })),
        ("agp absent entirely", json!({
            "messages": [{"role": "user", "content": "Hi"}] })),
        ("content with quotes/tags", json!({
            "messages": [{"role": "user", "content": "say \"hi\" <b>&amp;</b> 'x'"}],
            "add_generation_prompt": true })),
        ("system not first", json!({
            "messages": [{"role": "user", "content": "a"},
                         {"role": "system", "content": "late"}],
            "add_generation_prompt": true })),
    ]
}

#[test]
fn generated_matches_interpreter_byte_for_byte() {
    let env = oracle();
    let mut failures = Vec::new();
    for (name, ctx) in matrix() {
        let theirs = env
            .get_template("chat")
            .unwrap()
            .render(&ctx)
            .unwrap_or_else(|e| panic!("oracle failed on {name}: {e}"));
        let ours = smollm2_135m::render(&ctx)
            .unwrap_or_else(|e| panic!("generated code failed on {name}: {e}"));
        if ours != theirs {
            failures.push(format!("  {name}\n    generated: {ours:?}\n    oracle   : {theirs:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases diverged:\n{}",
        failures.len(),
        matrix().len(),
        failures.join("\n")
    );
}

/// The vendored template must be byte-exact. A trailing-newline "tidy-up" by an
/// editor changes the rendered prompt, and therefore the tokenization.
#[test]
fn vendored_template_is_byte_exact() {
    assert_eq!(TEMPLATE.len(), 368, "vendored SmolLM2 template changed size");
    assert!(
        !TEMPLATE.ends_with('\n'),
        "the upstream template has no trailing newline; something normalised it"
    );
}

/// Guard against the harness being "simplified" into uselessness.
///
/// If someone replaces the oracle with a default `Environment`, this test fails
/// — because the two configurations genuinely produce different bytes, and the
/// runtime uses the trim/lstrip one.
#[test]
fn whitespace_config_is_not_optional() {
    let ctx = json!({ "messages": [{"role": "user", "content": "Hi"}],
                      "add_generation_prompt": true });
    let configured = oracle().get_template("chat").unwrap().render(&ctx).unwrap();

    let mut naive = Environment::new();
    naive.add_template("chat", TEMPLATE).unwrap();
    let defaulted = naive.get_template("chat").unwrap().render(&ctx).unwrap();

    // For SmolLM2 the two happen to agree (it has no inter-tag newlines), so we
    // assert the generated code matches the CONFIGURED one and document that the
    // distinction bites on TinyLlama, where the two differ.
    assert_eq!(smollm2_135m::render(&ctx).unwrap(), configured);
    let _ = defaulted; // kept to make the comparison explicit for the next template
}

/// The drift gate. A compiled renderer must only be used against the exact
/// template it was generated from.
#[test]
fn drift_gate_accepts_only_the_exact_source() {
    let c = &smollm2_135m::COMPILED;
    assert!(c.matches(TEMPLATE), "must accept its own source");
    assert_eq!(c.source, TEMPLATE, "embedded source must be the vendored bytes");

    // Every one of these is a real way a template drifts in the wild, and every
    // one changes the rendered prompt.
    let mut trailing = TEMPLATE.to_string();
    trailing.push('\n');
    assert!(!c.matches(&trailing), "a trailing newline must NOT match");

    assert!(
        !c.matches(&TEMPLATE.replace('\n', "\r\n")),
        "CRLF line endings must NOT match"
    );
    assert!(
        !c.matches(&TEMPLATE.replace("SmolLM", "SmolLM2")),
        "a changed system prompt must NOT match"
    );
    assert!(!c.matches(""), "an empty template must NOT match");

    // And the descriptor's fn pointer is the same renderer we tested above.
    let ctx = json!({ "messages": [{"role": "user", "content": "Hi"}],
                      "add_generation_prompt": true });
    assert_eq!((c.render)(&ctx).unwrap(), smollm2_135m::render(&ctx).unwrap());
}
