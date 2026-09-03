// SPDX-License-Identifier: Apache-2.0
//! Render an emitted token stream to the text rustc will read.
//!
//! The other half of the build-script contract (`compile_in_dir` produces
//! the tokens, this writes them). Nothing here is for a human: the emit is
//! machine-read and 78 MB for a single llama stem. What rustc DOES care
//! about is LINE STRUCTURE — `TokenStream::to_string()` yields one 77 MB
//! line and rustc's span/line-column bookkeeping collapses on it.
//!
//! Measured on `-Fmetal --release` with the default scope (5 small llama
//! configs, llama-3.2-3b among the emitted variants), build script vs
//! `scratchy-models` rustc:
//!
//! | render         | emit  | rustc  | total | file   |
//! |----------------|-------|--------|-------|--------|
//! | `prettyplease` | 4.86s | 10.02s | 14.9s | 140 MB |
//! | `to_string()`  | 2.91s | 17.41s | 20.3s |  77 MB |
//! | `render_tokens`| 3.21s |  9.91s | 13.1s |  78 MB |
//!
//! So NEWLINES, not formatting, are what buy the rustc time — and they
//! cost one linear walk instead of `syn::parse_file` + `prettyplease`
//! re-printing the whole file from an AST.

use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};

/// Render `ts` to Rust source text, appending to `out`.
///
/// Breaks the line after `;` and `,` and around brace/bracket/paren
/// groups. The `,` break is the load-bearing one: the bulk of the emit is
/// `&[Struct { field: val, .. }, ..]` tape literals, which stay one line
/// without it. Indentation is deliberately omitted — it is bytes rustc
/// must lex for no gain.
///
/// The output is TOKEN-IDENTICAL to `TokenStream::to_string()` — this is a
/// whitespace choice and nothing more (pinned by
/// `render_is_token_identical_to_to_string`).
pub fn render_tokens(ts: TokenStream, out: &mut String) {
    for tt in ts {
        match tt {
            TokenTree::Group(g) => {
                let (open, close) = match g.delimiter() {
                    Delimiter::Brace => ("{", "}"),
                    Delimiter::Parenthesis => ("(", ")"),
                    Delimiter::Bracket => ("[", "]"),
                    // A `None`-delimited group is invisible: it carries
                    // grouping for the compiler, not text.
                    Delimiter::None => ("", ""),
                };
                out.push_str(open);
                if !open.is_empty() {
                    out.push('\n');
                }
                render_tokens(g.stream(), out);
                if !close.is_empty() {
                    out.push('\n');
                }
                out.push_str(close);
                // A closing BRACE must end the line. `clippy::possible_missing_else`
                // is whitespace-sensitive — it fires on `} if` sharing a line, and
                // the CI gate is `-D warnings` over the emitted crate. Other
                // delimiters take a plain separator.
                out.push(if g.delimiter() == Delimiter::Brace {
                    '\n'
                } else {
                    ' '
                });
            }
            TokenTree::Punct(p) => {
                out.push(p.as_char());
                if matches!(p.as_char(), ';' | ',') {
                    out.push('\n');
                } else if p.spacing() == Spacing::Alone {
                    // Joint spacing means the next punct is part of the
                    // same operator (`::`, `->`, `..`) — no separator.
                    out.push(' ');
                }
            }
            // Idents and literals carry their own text; the trailing
            // space keeps `a b` from lexing as `ab`.
            other => {
                out.push_str(&other.to_string());
                out.push(' ');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collapse whitespace runs, then drop whitespace adjacent to
    /// punctuation. What survives are single spaces separating word
    /// tokens — i.e. a token-boundary-preserving normal form, which two
    /// renderings of the SAME stream must agree on.
    fn token_form(s: &str) -> String {
        let collapsed: String = {
            let mut out = String::with_capacity(s.len());
            let mut in_ws = false;
            for c in s.chars() {
                if c.is_whitespace() {
                    in_ws = true;
                } else {
                    if in_ws && !out.is_empty() {
                        out.push(' ');
                    }
                    in_ws = false;
                    out.push(c);
                }
            }
            out
        };
        let chars: Vec<char> = collapsed.chars().collect();
        let mut out = String::with_capacity(chars.len());
        for (i, &c) in chars.iter().enumerate() {
            if c != ' ' {
                out.push(c);
                continue;
            }
            let prev = i.checked_sub(1).map(|j| chars[j]);
            let next = chars.get(i + 1).copied();
            let wordish =
                |o: Option<char>| o.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '"');
            // Keep the space only where removing it would fuse two tokens.
            if wordish(prev) && wordish(next) {
                out.push(' ');
            }
        }
        out
    }

    /// THE LAW: `render_tokens` differs from `TokenStream::to_string()`
    /// in whitespace only. If this ever fails, the renderer is changing
    /// the program, not its layout.
    #[test]
    fn render_is_token_identical_to_to_string() {
        let src = r#"
            #[cfg(feature = "metal")]
            pub static TAPE: &[Cmd] = &[
                Cmd { kernel: KernelId::Gemm, m: 512u32, dims: Some(D { n: 1, k: -2 }) },
                Cmd { kernel: KernelId::Add, m: 1u32, dims: None },
            ];
            impl Foo for Bar {
                fn go(&self) -> ::core::option::Option<u32> {
                    let x = self.a..self.b;
                    Some(x.len() as u32)
                }
            }
        "#;
        let ts: TokenStream = src.parse().expect("fixture parses");

        let mut rendered = String::new();
        render_tokens(ts.clone(), &mut rendered);

        assert_eq!(
            token_form(&rendered),
            token_form(&ts.to_string()),
            "render_tokens changed the token stream, not just its whitespace"
        );
        // …and it must actually have produced lines, or it is buying nothing.
        assert!(
            rendered.lines().count() > 20,
            "render_tokens produced {} lines — the line structure IS the point",
            rendered.lines().count()
        );
    }

    /// Re-parsing the rendered text must give back the same stream: the
    /// whitespace choice cannot make the output un-lexable.
    #[test]
    fn rendered_text_reparses_to_the_same_stream() {
        let ts: TokenStream = r#"
            const A: i32 = -1;
            let p = a::b::<T>(x, y)?.z;
            #[doc = "hi"] struct S { f: &'static [u8] }
        "#
        .parse()
        .expect("fixture parses");

        let mut rendered = String::new();
        render_tokens(ts.clone(), &mut rendered);
        let round: TokenStream = rendered.parse().expect("rendered text re-lexes");

        assert_eq!(round.to_string(), ts.to_string());
    }
}
