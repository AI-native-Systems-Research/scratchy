//! The lint-level attributes P11 refuses, found in one file's tokens.

use rustc_lexer::TokenKind;
use std::ops::Range;

/// The attribute names that lower a lint's level, each one an opt-out of a check.
/// `deny`, `warn` and `forbid` are not here: they tighten a level, and CI runs with warnings
/// denied.
pub const OPT_OUTS: &[&str] = &["allow", "expect"];

/// One token of code (not whitespace, not a comment): its kind and its byte range in the source.
struct Tok {
    kind: TokenKind,
    at: Range<usize>,
}

fn code_tokens(src: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut offset = 0usize;
    for token in rustc_lexer::tokenize(src, rustc_lexer::FrontmatterAllowed::No) {
        let at = offset..offset + token.len as usize;
        offset = at.end;
        if !matches!(
            token.kind,
            TokenKind::Whitespace | TokenKind::LineComment { .. } | TokenKind::BlockComment { .. }
        ) {
            toks.push(Tok {
                kind: token.kind,
                at,
            });
        }
    }
    toks
}

fn opens(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace
    )
}

fn closes(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace
    )
}

/// The index of the delimiter that closes the one at `open`.
fn closing(toks: &[Tok], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, t) in toks.iter().enumerate().skip(open) {
        if opens(t.kind) {
            depth += 1;
        } else if closes(t.kind) {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// An identifier's name, without a raw identifier's `r#`.
fn ident<'s>(src: &'s str, t: &Tok) -> Option<&'s str> {
    match t.kind {
        TokenKind::Ident => Some(&src[t.at.clone()]),
        TokenKind::RawIdent => Some(&src[t.at.start + 2..t.at.end]),
        _ => None,
    }
}

/// Whether one attribute's content (the tokens between its brackets) sets a lint level an
/// opt-out lowers: `allow(..)` or `expect(..)`, or a `cfg_attr` (nested too) whose attributes
/// after its predicate hold one.
fn opts_out(src: &str, meta: &[Tok]) -> bool {
    let Some((first, rest)) = meta.split_first() else {
        return false;
    };
    let Some(name) = ident(src, first) else {
        return false;
    };
    if rest.first().is_some_and(|t| t.kind == TokenKind::Colon) {
        // A path (`rustfmt::skip`), not a built-in lint attribute.
        return false;
    }
    if OPT_OUTS.contains(&name) {
        return true;
    }
    if name != "cfg_attr" || !rest.first().is_some_and(|t| t.kind == TokenKind::OpenParen) {
        return false;
    }
    let Some(close) = closing(rest, 0) else {
        return false;
    };
    let mut args = Vec::new();
    let (mut start, mut depth) = (1usize, 0usize);
    for (i, t) in rest.iter().enumerate().take(close).skip(1) {
        if opens(t.kind) {
            depth += 1;
        } else if closes(t.kind) {
            depth -= 1;
        } else if depth == 0 && t.kind == TokenKind::Comma {
            args.push(&rest[start..i]);
            start = i + 1;
        }
    }
    args.push(&rest[start..close]);
    // The first argument is the predicate; the rest are the attributes it gates.
    args.into_iter().skip(1).any(|a| opts_out(src, a))
}

/// The byte range of every attribute in `src`, outer (`#[..]`) or inner (`#![..]`), that sets
/// `allow` or `expect` for any lint, directly or through `cfg_attr`. Read from tokens, so the
/// word in a string, a comment or an identifier is not one.
pub fn lint_allows(src: &str) -> Vec<Range<usize>> {
    let toks = code_tokens(src);
    let mut found = Vec::new();
    for (i, pound) in toks.iter().enumerate() {
        if pound.kind != TokenKind::Pound {
            continue;
        }
        let bang = toks.get(i + 1).is_some_and(|t| t.kind == TokenKind::Bang);
        let open = i + 1 + usize::from(bang);
        if toks
            .get(open)
            .is_none_or(|t| t.kind != TokenKind::OpenBracket)
        {
            continue;
        }
        if let Some(close) = closing(&toks, open)
            && opts_out(src, &toks[open + 1..close])
        {
            found.push(pound.at.start..toks[close].at.end);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(src: &str) -> Vec<&str> {
        lint_allows(src).into_iter().map(|r| &src[r]).collect()
    }

    #[test]
    fn allow_and_expect_are_found_outer_and_inner() {
        assert_eq!(
            texts("#[allow(dead_code)] fn a() {}"),
            ["#[allow(dead_code)]"]
        );
        assert_eq!(texts("#![allow(unused)]"), ["#![allow(unused)]"]);
        assert_eq!(
            texts("#[expect(clippy::too_many_arguments)] fn a() {}"),
            ["#[expect(clippy::too_many_arguments)]"]
        );
        assert_eq!(
            texts("#![expect(rustdoc::broken_intra_doc_links)]"),
            ["#![expect(rustdoc::broken_intra_doc_links)]"]
        );
        assert_eq!(
            texts("#[r#allow(dead_code)] fn a() {}"),
            ["#[r#allow(dead_code)]"]
        );
    }

    #[test]
    fn spacing_and_comments_inside_the_attribute_do_not_hide_it() {
        assert_eq!(texts("#[ allow ( x ) ]"), ["#[ allow ( x ) ]"]);
        assert_eq!(texts("# ! [\n  allow(x)\n]"), ["# ! [\n  allow(x)\n]"]);
        assert_eq!(texts("#[/* c */ allow(x)]"), ["#[/* c */ allow(x)]"]);
    }

    #[test]
    fn cfg_attr_is_read_through_nested_too() {
        assert_eq!(
            texts("#[cfg_attr(test, allow(dead_code))]"),
            ["#[cfg_attr(test, allow(dead_code))]"]
        );
        assert_eq!(
            texts("#![cfg_attr(any(a, b), inline, expect(unused))]"),
            ["#![cfg_attr(any(a, b), inline, expect(unused))]"]
        );
        assert_eq!(
            texts("#[cfg_attr(a, cfg_attr(b, allow(x)))]"),
            ["#[cfg_attr(a, cfg_attr(b, allow(x)))]"]
        );
        // The predicate is not an attribute: a cfg named allow gates nothing that opts out.
        assert_eq!(texts("#[cfg_attr(allow, inline)]"), Vec::<&str>::new());
        assert_eq!(
            texts("#[cfg_attr(feature = \"x\", deny(unused))]"),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn the_word_elsewhere_is_not_an_attribute() {
        for src in [
            r####"const S: &str = r#"#[allow(dead_code)]"#;"####,
            "const S: &str = \"#[allow(x)]\";",
            "/// Never write `#[allow]`; fix the code.\nfn a() {}",
            "//! See #![allow(x)] in the skill.",
            "/* #[expect(x)] */",
            "struct S { allow: bool, allow_list: Vec<u8> }",
            "fn f(s: S) -> bool { s.allow }",
            "#[allow_list(x)] fn a() {}",
            "#[rustfmt::skip] fn a() {}",
            "#[doc = \"#[allow(x)]\"] fn a() {}",
        ] {
            assert_eq!(texts(src), Vec::<&str>::new(), "{src}");
        }
    }

    #[test]
    fn tightening_levels_are_not_opt_outs() {
        for src in [
            "#[deny(unused)] fn a() {}",
            "#![warn(missing_docs)]",
            "#![forbid(unsafe_code)]",
        ] {
            assert_eq!(texts(src), Vec::<&str>::new(), "{src}");
        }
    }

    #[test]
    fn a_macro_body_that_stamps_an_allow_is_found() {
        assert_eq!(
            texts("macro_rules! m { () => { #[allow(dead_code)] fn f() {} }; }"),
            ["#[allow(dead_code)]"]
        );
        assert_eq!(
            texts("macro_rules! m { ($($a:meta)*) => { $(#[$a])* fn f() {} }; }"),
            Vec::<&str>::new()
        );
    }
}
