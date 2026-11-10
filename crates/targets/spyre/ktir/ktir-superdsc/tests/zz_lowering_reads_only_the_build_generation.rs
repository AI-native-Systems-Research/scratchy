// SPDX-License-Identifier: Apache-2.0
//! ⛔ LOWERING CODE READS `Baked`, NEVER A GENERATION BY NAME.
//!
//! A lowering that named `Gen1` would bake gen1's facts into a gen2 build. The generation a binary
//! bakes for is `ktir_core::ddl_binds::Baked` (#1158), chosen once; `Gen1` may appear in
//! `ktir-core/src/ddl_binds.rs` (its impl and the `Baked` alias), in `src/device_gen.rs`'s
//! re-export, and in tests — nowhere else. This walks every `.rs` file of the lowering's `src/`
//! trees (ktir-core, ktir-superdsc and the spyre target crate) and fails on any other non-comment
//! occurrence, skipping `#[cfg(test)]` modules.

use std::path::Path;

fn rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).expect("readable src dir") {
        let p = e.expect("dir entry").path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The byte ranges of `#[cfg(test)] mod … { … }` blocks.
fn test_spans(s: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut from = 0;
    while let Some(i) = s[from..].find("#[cfg(test)]") {
        let at = from + i;
        let rest = &s[at..];
        let Some(brace) = rest.find('{') else { break };
        let head = &rest[..brace];
        if head.contains("mod ") && !head.contains(';') {
            let mut depth = 0usize;
            let mut end = s.len();
            for (j, c) in s[at + brace..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = at + brace + j;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            spans.push((at, end));
            from = end;
        } else {
            from = at + "#[cfg(test)]".len();
        }
    }
    spans
}

fn is_word(s: &str, at: usize, len: usize) -> bool {
    let b = s.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    (at == 0 || !ident(b[at - 1])) && (at + len >= b.len() || !ident(b[at + len]))
}

#[test]
fn no_lowering_code_names_a_generation() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let roots = [
        here.join("src"),
        here.join("../ktir-core/src"),
        here.join("../../src"),
    ];
    let mut files = Vec::new();
    for r in &roots {
        let before = files.len();
        rs_files(r, &mut files);
        assert!(
            files.len() - before > 5,
            "the walk found {} files under {r:?}",
            files.len() - before
        );
    }
    let src = here.join("../..");
    let mut hits = Vec::new();
    for f in &files {
        if f.ends_with("ddl_binds.rs") || f.ends_with("ktir-superdsc/src/device_gen.rs") {
            continue;
        }
        let s = std::fs::read_to_string(f).expect("utf-8 source");
        let spans = test_spans(&s);
        let mut line_start = 0;
        for line in s.split_inclusive('\n') {
            let code = line.split("//").next().unwrap_or("");
            let mut from = 0;
            while let Some(i) = code[from..].find("Gen1") {
                let at = line_start + from + i;
                if is_word(&s, at, 4) && !spans.iter().any(|&(a, b)| a <= at && at <= b) {
                    hits.push(format!(
                        "{}: {}",
                        f.strip_prefix(&src).unwrap_or(f).display(),
                        line.trim()
                    ));
                }
                from += i + 4;
            }
            line_start += line.len();
        }
    }
    assert!(
        hits.is_empty(),
        "lowering code names a generation — read `Baked` instead:\n{}",
        hits.join("\n")
    );
}

/// The guard is not vacuous: it sees a name in code, and skips comments and test modules.
#[test]
fn the_guard_sees_code_and_skips_comments_and_tests() {
    let s =
        "fn a() { x::<Gen1>(); }\n// Gen1 in a comment\n#[cfg(test)]\nmod t { fn b() { Gen1; } }\n";
    let spans = test_spans(s);
    assert_eq!(spans.len(), 1);
    let code_hit = s.find("Gen1").unwrap();
    assert!(is_word(s, code_hit, 4) && !spans.iter().any(|&(a, b)| a <= code_hit && code_hit <= b));
    let test_hit = s.rfind("Gen1").unwrap();
    assert!(spans.iter().any(|&(a, b)| a <= test_hit && test_hit <= b));
    assert!(
        !is_word("Gen10", 0, 4),
        "a longer identifier is not the name"
    );
}
