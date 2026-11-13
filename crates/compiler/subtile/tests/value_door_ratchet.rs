// SPDX-License-Identifier: Apache-2.0
//! THE #1094 RATCHET: the shape laws still checked on values, and the doors through which a value
//! still becomes a type, counted from the source.
//!
//! #1094's goal is every shape constraint a bound rustc's type checker rejects. Two things still
//! stand in for it, and this test counts both in the library code of the compiler, tensor, layer
//! and target crates (every `.rs` outside `tests/`, `benches/` and `examples/`; items under
//! `#[cfg(test)]` and `#[test]` functions are not library code):
//!
//! * **Law asserts**: an `assert!`, `assert_eq!`, `assert_ne!` or `panic!` in the body of a
//!   constructor (`fn new`), whether an impl's or one a `macro_rules!` body stamps out. Each is a
//!   law checked on numbers where a bound could state it.
//! * **Value→type doors**, of two forms:
//!   - a value constructor (`fn new`) in an impl that also has a typed constructor (`fn of` with
//!     generic parameters): the typed one states the laws as bounds, and the value one lets a
//!     number in past them;
//!   - a `match` arm whose pattern is a literal or a `macro_rules!` metavariable and whose body
//!     passes that same token as a generic argument (`64 => f::<64>()`, `$w => g::<$w>()`): the
//!     arm turns a runtime value into a const.
//!
//! Each count is pinned below. A PR may lower a count, and must then lower its pin; none may raise
//! one. On a failure the message lists every site the scanner found, grouped by file.

use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The law asserts left: `assert!`/`assert_eq!`/`assert_ne!`/`panic!` in a `fn new`.
const LAW_ASSERTS: usize = 31;
/// The value→type doors left: value constructors beside a typed one, and value→const match arms.
const VALUE_DOORS: usize = 10;

/// The crates scanned, from the repository root: the root workspace's compiler, tensor, layer and
/// target crates, and the spyre backend's own workspace (its crates left `crates/targets` with it).
const ROOTS: &[&str] = &[
    "crates/compiler",
    "crates/tensors",
    "crates/layers",
    "crates/targets",
    "backends/spyre/crates",
];

/// The directories whose files are not library code.
const SKIPPED_DIRS: &[&str] = &["tests", "benches", "examples", "target"];

/// One site the scanner counts: the file it is in and what it is.
#[derive(Debug)]
struct Site {
    file: String,
    what: String,
}

#[derive(Default)]
struct Sites {
    asserts: Vec<Site>,
    doors: Vec<Site>,
}

#[test]
fn law_asserts_and_value_doors_only_go_down() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut files = Vec::new();
    for root in ROOTS {
        let dir = repo.join(root);
        assert!(
            dir.is_dir(),
            "{root}: no such directory under {}",
            repo.display()
        );
        rust_files(&dir, &mut files);
    }
    files.sort();

    let mut sites = Sites::default();
    for path in &files {
        let rel = path
            .strip_prefix(&repo)
            .map_or_else(|_| path.display().to_string(), |p| p.display().to_string());
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let file = syn::parse_file(&text).unwrap_or_else(|e| panic!("{rel} does not parse: {e}"));
        scan_items(&file.items, &rel, &mut sites);
    }

    let mut failures = Vec::new();
    check("law asserts", LAW_ASSERTS, &sites.asserts, &mut failures);
    check("value→type doors", VALUE_DOORS, &sites.doors, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Compare one category's count with its pin.
fn check(name: &str, pinned: usize, sites: &[Site], failures: &mut Vec<String>) {
    let found = sites.len();
    if found == pinned {
        return;
    }
    let verdict = if found > pinned {
        format!(
            "{name}: {found} found, {pinned} pinned. A PR may not add one; the new site is among these:"
        )
    } else {
        format!(
            "{name}: {found} found, {pinned} pinned. Lower the pin to {found}, so no later PR can add one back:"
        )
    };
    let mut by_file: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for s in sites {
        by_file.entry(&s.file).or_default().push(&s.what);
    }
    let mut msg = verdict;
    for (file, whats) in by_file {
        msg.push_str(&format!("\n  {file} ({})", whats.len()));
        for w in whats {
            msg.push_str(&format!("\n    {w}"));
        }
    }
    failures.push(msg);
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if !SKIPPED_DIRS.contains(&name.as_ref()) && !name.starts_with('.') {
                rust_files(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

/// Whether an item's attributes take it out of library code: `#[test]`, `#[cfg(test)]`, or a
/// `cfg(all(…))` that requires `test`.
fn is_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if a.path().is_ident("test") {
            return true;
        }
        if !a.path().is_ident("cfg") {
            return false;
        }
        let Ok(list) = a.meta.require_list() else {
            return false;
        };
        let mut tokens = list.tokens.clone().into_iter();
        match tokens.next() {
            Some(TokenTree::Ident(i)) if i == "test" => tokens.next().is_none(),
            Some(TokenTree::Ident(i)) if i == "all" => match tokens.next() {
                Some(TokenTree::Group(g)) => g
                    .stream()
                    .into_iter()
                    .any(|t| matches!(&t, TokenTree::Ident(i) if i == "test")),
                _ => false,
            },
            _ => false,
        }
    })
}

fn scan_items(items: &[syn::Item], file: &str, sites: &mut Sites) {
    for item in items {
        match item {
            syn::Item::Mod(m) if !is_test(&m.attrs) => {
                if let Some((_, inner)) = &m.content {
                    scan_items(inner, file, sites);
                }
            }
            syn::Item::Impl(i) if !is_test(&i.attrs) => scan_impl(i, file, sites),
            syn::Item::Fn(f) if !is_test(&f.attrs) => {
                let tokens = quote::quote!(#f);
                scan_tokens(tokens, file, &f.sig.ident.to_string(), sites);
            }
            syn::Item::Trait(t) if !is_test(&t.attrs) => {
                let tokens = quote::quote!(#t);
                scan_tokens(tokens, file, &t.ident.to_string(), sites);
            }
            syn::Item::Macro(m) if !is_test(&m.attrs) => {
                let name = m
                    .ident
                    .as_ref()
                    .map_or_else(|| "macro".to_string(), |i| format!("{i}!"));
                scan_tokens(m.mac.tokens.clone(), file, &name, sites);
            }
            syn::Item::Const(c) if !is_test(&c.attrs) => {
                let tokens = quote::quote!(#c);
                scan_tokens(tokens, file, &c.ident.to_string(), sites);
            }
            syn::Item::Static(s) if !is_test(&s.attrs) => {
                let tokens = quote::quote!(#s);
                scan_tokens(tokens, file, &s.ident.to_string(), sites);
            }
            _ => {}
        }
    }
}

/// The last path segment of an impl's self type, for naming its sites.
fn self_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => p
            .path
            .segments
            .last()
            .map_or_else(|| "_".to_string(), |s| s.ident.to_string()),
        _ => "_".to_string(),
    }
}

fn scan_impl(imp: &syn::ItemImpl, file: &str, sites: &mut Sites) {
    let ty = self_name(&imp.self_ty);
    let fns: Vec<&syn::ImplItemFn> = imp
        .items
        .iter()
        .filter_map(|it| match it {
            syn::ImplItem::Fn(f) if !is_test(&f.attrs) => Some(f),
            _ => None,
        })
        .collect();
    let typed_of = fns
        .iter()
        .any(|f| f.sig.ident == "of" && !f.sig.generics.params.is_empty());
    for f in &fns {
        let name = format!("{ty}::{}", f.sig.ident);
        if f.sig.ident == "new" && typed_of {
            sites.doors.push(Site {
                file: file.to_string(),
                what: format!("{name}: a value constructor beside the typed `{ty}::of`"),
            });
        }
        let tokens = quote::quote!(#f);
        scan_tokens(tokens, file, &name, sites);
    }
    for it in &imp.items {
        if let syn::ImplItem::Macro(m) = it {
            scan_tokens(m.mac.tokens.clone(), file, &ty, sites);
        }
    }
}

/// Scan an item's tokens for both kinds of site: the asserts of every `fn new` body in them, and
/// every value→const match arm.
fn scan_tokens(tokens: TokenStream, file: &str, owner: &str, sites: &mut Sites) {
    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    news(&tts, file, owner, sites);
    arms(&tts, file, owner, sites);
}

/// Every `fn new … { body }` at any depth of `tts`, with the asserts its body holds.
fn news(tts: &[TokenTree], file: &str, owner: &str, sites: &mut Sites) {
    let mut i = 0;
    while i < tts.len() {
        if is_ident(&tts[i], "fn") && tts.get(i + 1).is_some_and(|t| is_ident(t, "new")) {
            let body = tts[i + 2..].iter().find_map(|t| match t {
                TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => Some(g),
                _ => None,
            });
            if let Some(body) = body {
                let name = if owner.ends_with("::new") {
                    owner.to_string()
                } else {
                    format!("{owner}::new")
                };
                let mut found = Vec::new();
                asserts(&body.stream().into_iter().collect::<Vec<_>>(), &mut found);
                for what in found {
                    sites.asserts.push(Site {
                        file: file.to_string(),
                        what: format!("{name}: {what}"),
                    });
                }
                // The body is consumed; skip past it.
                if let Some(pos) = tts[i + 2..].iter().position(
                    |t| matches!(t, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace),
                ) {
                    i += 2 + pos + 1;
                    continue;
                }
            }
        }
        if let TokenTree::Group(g) = &tts[i] {
            news(
                &g.stream().into_iter().collect::<Vec<_>>(),
                file,
                owner,
                sites,
            );
        }
        i += 1;
    }
}

/// The asserts in `tts` at any depth, each named by its macro and its message.
fn asserts(tts: &[TokenTree], out: &mut Vec<String>) {
    for (i, t) in tts.iter().enumerate() {
        if let TokenTree::Ident(id) = t {
            let name = id.to_string();
            if matches!(
                name.as_str(),
                "assert" | "assert_eq" | "assert_ne" | "panic"
            ) && tts.get(i + 1).is_some_and(|t| is_punct(t, '!'))
            {
                let message = tts.get(i + 2).and_then(|t| match t {
                    TokenTree::Group(g) => first_string(g.stream()),
                    _ => None,
                });
                out.push(format!("{name}!({})", message.unwrap_or_default()));
            }
        }
        if let TokenTree::Group(g) = t {
            asserts(&g.stream().into_iter().collect::<Vec<_>>(), out);
        }
    }
}

/// The first string literal at any depth of `tokens`, on one line: an assert's message, or the
/// literal part of a `concat!` one.
fn first_string(tokens: TokenStream) -> Option<String> {
    tokens.into_iter().find_map(|t| match t {
        TokenTree::Literal(l) if l.to_string().starts_with('"') => Some(
            l.to_string()
                .replace("\\\n", " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        ),
        TokenTree::Group(g) => first_string(g.stream()),
        _ => None,
    })
}

/// Every value→const match arm at any depth of `tts`: `PAT => BODY` with `PAT` one literal or one
/// `$metavariable`, and `BODY` passing that token in a turbofish's generic arguments.
fn arms(tts: &[TokenTree], file: &str, owner: &str, sites: &mut Sites) {
    for i in 0..tts.len() {
        if !(is_joint(&tts[i], '=') && tts.get(i + 1).is_some_and(|t| is_punct(t, '>'))) {
            continue;
        }
        // The pattern: one literal, or `$` and an identifier, right after an arm boundary.
        let (pattern, start) = match (
            i.checked_sub(2).map(|j| &tts[j]),
            i.checked_sub(1).map(|j| &tts[j]),
        ) {
            (Some(d), Some(TokenTree::Ident(id))) if is_punct(d, '$') => (format!("${id}"), i - 2),
            (_, Some(TokenTree::Literal(l))) => (l.to_string(), i - 1),
            _ => continue,
        };
        let at_boundary = start == 0
            || is_punct(&tts[start - 1], ',')
            || matches!(&tts[start - 1], TokenTree::Group(_));
        if !at_boundary {
            continue;
        }
        // The body: up to the next `,` at this depth.
        let body: Vec<TokenTree> = tts[i + 2..]
            .iter()
            .take_while(|t| !is_punct(t, ','))
            .cloned()
            .collect();
        let mut flat = Vec::new();
        flatten(&body, &mut flat);
        if turbofish_names(&flat, &pattern) {
            sites.doors.push(Site {
                file: file.to_string(),
                what: format!("{owner}: the match arm `{pattern} => …::<{pattern}>` mints a const from a value"),
            });
        }
    }
    for t in tts {
        if let TokenTree::Group(g) = t {
            arms(
                &g.stream().into_iter().collect::<Vec<_>>(),
                file,
                owner,
                sites,
            );
        }
    }
}

/// The leaf tokens of `tts` in order, as text, with every group's contents inlined.
fn flatten(tts: &[TokenTree], out: &mut Vec<String>) {
    for t in tts {
        match t {
            TokenTree::Group(g) => {
                let (open, close) = match g.delimiter() {
                    Delimiter::Parenthesis => ("(", ")"),
                    Delimiter::Brace => ("{", "}"),
                    Delimiter::Bracket => ("[", "]"),
                    Delimiter::None => ("", ""),
                };
                out.push(open.to_string());
                flatten(&g.stream().into_iter().collect::<Vec<_>>(), out);
                out.push(close.to_string());
            }
            TokenTree::Punct(p) => out.push(p.as_char().to_string()),
            other => out.push(other.to_string()),
        }
    }
}

/// Whether some turbofish `::<…>` in the flat tokens names `pattern` (a literal, or `$name`) among
/// its generic arguments.
fn turbofish_names(flat: &[String], pattern: &str) -> bool {
    let wanted: Vec<&str> = match pattern.strip_prefix('$') {
        Some(name) => vec!["$", name],
        None => vec![pattern],
    };
    let mut i = 0;
    while i + 2 < flat.len() {
        if flat[i] == ":" && flat[i + 1] == ":" && flat[i + 2] == "<" {
            let mut depth = 1;
            let mut j = i + 3;
            let start = j;
            while j < flat.len() && depth > 0 {
                match flat[j].as_str() {
                    "<" => depth += 1,
                    ">" => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let args = &flat[start..j.saturating_sub(1)];
            if args.windows(wanted.len()).any(|w| w == wanted.as_slice()) {
                return true;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    false
}

fn is_ident(t: &TokenTree, s: &str) -> bool {
    matches!(t, TokenTree::Ident(i) if i == s)
}

fn is_punct(t: &TokenTree, c: char) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == c)
}

fn is_joint(t: &TokenTree, c: char) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == c && p.spacing() == Spacing::Joint)
}
