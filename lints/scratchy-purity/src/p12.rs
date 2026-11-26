//! P12, proper tests.

use clippy_utils::diagnostics::span_lint_and_help;
use rustc_lint::EarlyContext;
use rustc_session::declare_lint;

declare_lint! {
    /// P12: a type proof is a `compile_fail` doctest that names its error code. Only nightly
    /// rustdoc checks the code, and a doctest that names none passes for any error, so it
    /// proves nothing.
    pub P12_COMPILE_FAIL_WITHOUT_CODE,
    Forbid,
    "a compile_fail doctest that names no error code"
}

pub static LINTS: &[&rustc_lint::Lint] = &[P12_COMPILE_FAIL_WITHOUT_CODE];

/// Whether one line of doc text opens a `compile_fail` code block that names no `E` code.
/// The line may carry a block doc comment's `*` margin. Rustdoc splits a fence's attributes on
/// commas and whitespace, so they are read the same way here.
pub fn bare_compile_fail(line: &str) -> bool {
    let line = line.trim_start();
    let line = line.strip_prefix('*').unwrap_or(line).trim_start();
    let Some(fence) = line.chars().next().filter(|c| *c == '`' || *c == '~') else {
        return false;
    };
    let info = line.trim_start_matches(fence);
    if line.len() - info.len() < 3 {
        return false;
    }
    let attrs: Vec<&str> = info.split(|c: char| c == ',' || c.is_whitespace()).filter(|a| !a.is_empty()).collect();
    let code = |a: &&str| a.strip_prefix('E').is_some_and(|n| n.len() == 4 && n.bytes().all(|b| b.is_ascii_digit()));
    attrs.contains(&"compile_fail") && !attrs.iter().any(code)
}

/// Checks one doc comment line, or one `#[doc = ".."]` string, at `span`.
pub fn check_doc(cx: &EarlyContext<'_>, span: rustc_span::Span, text: &str) {
    for _ in text.lines().filter(|line| bare_compile_fail(line)) {
        span_lint_and_help(
            cx,
            P12_COMPILE_FAIL_WITHOUT_CODE,
            span,
            "P12: a compile_fail doctest without an error code passes for any error; name the code it proves, e.g. compile_fail,E0277",
            None,
            "add the error code the doctest is meant to fail with",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_compile_fail_fence_must_name_an_error_code() {
        for bare in [
            "```compile_fail",
            " ```compile_fail",
            "``` compile_fail ",
            "````compile_fail",
            "~~~compile_fail",
            "```compile_fail,edition2021",
            "```rust, compile_fail",
            " * ```compile_fail",
            "```compile_fail,E277",
            "```compile_fail,e0277",
        ] {
            assert!(bare_compile_fail(bare), "{bare:?}");
        }
        for named in [
            "```compile_fail,E0277",
            "```compile_fail,E0599,edition2021",
            "``` compile_fail , E0277 ",
            "```compile_fail E0277",
            "```rust,compile_fail,E0308",
            " * ```compile_fail,E0277",
        ] {
            assert!(!bare_compile_fail(named), "{named:?}");
        }
        for other in ["```", "```rust", "```should_panic", "compile_fail", "see ```compile_fail``` below", "``compile_fail"] {
            assert!(!bare_compile_fail(other), "{other:?}");
        }
    }
}
