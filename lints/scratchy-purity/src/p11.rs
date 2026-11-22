//! P11, lock and comment hygiene.

use clippy_utils::diagnostics::span_lint_and_help;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};

declare_lint! {
    /// P11: no escape-hatch constructors. A checked type has one checking constructor, and it is
    /// the only way in, so a `from_untyped` or `conflated_with_*` beside it is a second way in.
    pub P11_ESCAPE_HATCH_CONSTRUCTOR,
    Deny,
    "an escape-hatch constructor beside a checked type's checking constructor"
}

declare_lint_pass!(P11 => [P11_ESCAPE_HATCH_CONSTRUCTOR]);

pub static LINTS: &[&rustc_lint::Lint] = &[P11_ESCAPE_HATCH_CONSTRUCTOR];

fn escape_hatch(name: &str) -> bool {
    name == "from_untyped" || name.starts_with("conflated_with_")
}

impl EarlyLintPass for P11 {
    fn check_fn(
        &mut self,
        cx: &EarlyContext<'_>,
        kind: rustc_ast::visit::FnKind<'_>,
        _: rustc_span::Span,
        _: rustc_ast::NodeId,
    ) {
        let rustc_ast::visit::FnKind::Fn(_, _, f) = kind else {
            return;
        };
        let ident = f.ident;
        if !escape_hatch(ident.as_str()) || ident.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        span_lint_and_help(
            cx,
            P11_ESCAPE_HATCH_CONSTRUCTOR,
            ident.span,
            format!(
                "P11: `{ident}` is an escape-hatch constructor; a checked type has one checking constructor, the only way in"
            ),
            None,
            "delete it, and mint the value through the type's checking constructor",
        );
    }
}
