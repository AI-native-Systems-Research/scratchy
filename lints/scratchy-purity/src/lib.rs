//! The scratchy purity rules (`.claude/skills/scratchy-purity-audit/SKILL.md`) as rustc lints.
//! One module per rule. Every lint is `Forbid`, and no site opts out.
#![feature(rustc_private)]
#![warn(unused_extern_crates)]

extern crate rustc_ast;
extern crate rustc_errors;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;

mod comments;
mod exceptions;
mod names;
mod p11;
mod p7;

dylint_linting::dylint_library!();

#[unsafe(no_mangle)]
pub fn register_lints(sess: &rustc_session::Session, lint_store: &mut rustc_lint::LintStore) {
    dylint_linting::init_config(sess);
    lint_store.register_lints(p7::LINTS);
    lint_store.register_lints(p11::LINTS);
    lint_store.register_pre_expansion_lint_pass(Box::new(|| Box::new(p7::P7Track)));
    lint_store.register_pre_expansion_lint_pass(Box::new(|| Box::new(p11::P11Track)));
    lint_store.register_early_lint_pass(Box::new(|| Box::new(p7::P7)));
    lint_store.register_early_lint_pass(Box::new(|| Box::new(p11::P11)));
}

/// The ui cases, then the exempt door, built under the crate name its `exceptions.toml` entry
/// names. One test, since the two runs share the test driver.
#[test]
fn ui() {
    dylint_testing::ui_test(env!("CARGO_PKG_NAME"), "ui");
    dylint_testing::ui::Test::src_base(env!("CARGO_PKG_NAME"), "ui-door")
        .rustc_flags(["--crate-name", "scratchy_forward_compiler"])
        .run();
}
