//! The scratchy purity rules (`.claude/skills/scratchy-purity-audit/SKILL.md`) as rustc lints.
//! One module per rule. Every lint is `Deny`, and no site opts out.
#![feature(rustc_private)]
#![warn(unused_extern_crates)]

extern crate rustc_ast;
extern crate rustc_lexer;
extern crate rustc_lint;
extern crate rustc_session;
extern crate rustc_span;

mod comments;
mod p11;
mod p5;

dylint_linting::dylint_library!();

#[unsafe(no_mangle)]
pub fn register_lints(sess: &rustc_session::Session, lint_store: &mut rustc_lint::LintStore) {
    dylint_linting::init_config(sess);
    lint_store.register_lints(p11::LINTS);
    lint_store.register_lints(p5::LINTS);
    lint_store.register_pre_expansion_lint_pass(Box::new(|| Box::new(p11::P11Track)));
    lint_store.register_early_lint_pass(Box::new(|| Box::new(p11::P11)));
    lint_store.register_early_lint_pass(Box::new(|| Box::<p5::P5>::default()));
}

#[test]
fn ui() {
    dylint_testing::ui_test(env!("CARGO_PKG_NAME"), "ui");
}
