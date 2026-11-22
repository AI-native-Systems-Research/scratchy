//! The comment checks P11 runs over every source file a crate compiles.

use std::path::{Component, Path, PathBuf};

/// The phrases that narrate a comment's history instead of describing the code.
const TOMBSTONES: &[&str] = &[
    "used to be",
    "used to live",
    "now lives",
    "moved from",
    "moved here",
    // A code path after it: "moved to the worker thread" is an ownership move, not a tombstone.
    "moved to `",
    "superseded",
    "was removed",
];

/// Whether a file is the package's own source: under its manifest directory, and not under a
/// `target` directory there (generated code, `OUT_DIR` included, is not the package's comments).
pub fn scanned(path: &Path, manifest_dir: &Path) -> bool {
    path.strip_prefix(manifest_dir)
        .is_ok_and(|rest| !rest.components().any(|c| c == Component::Normal("target".as_ref())))
}

fn word_at(text: &str, start: usize, len: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[start + len..].chars().next();
    !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
}

/// The tombstone phrase a comment's text contains, matched case-insensitively on word boundaries.
pub fn tombstone(comment: &str) -> Option<&'static str> {
    let lower = comment.to_lowercase();
    TOMBSTONES.iter().copied().find(|phrase| {
        lower
            .match_indices(phrase)
            .any(|(i, _)| word_at(&lower, i, phrase.len()))
    })
}

/// The byte offset of a `/tmp/` or `~/tmp/` path a comment cites: one that starts the text, or
/// follows whitespace, a backtick, a quote, `(` or `[`.
pub fn tmp_path(comment: &str) -> Option<usize> {
    comment.match_indices("/tmp/").find_map(|(i, _)| {
        let start = if comment[..i].ends_with('~') { i - 1 } else { i };
        let opens = comment[..start]
            .chars()
            .next_back()
            .is_none_or(|c| c.is_whitespace() || "`'\"([".contains(c));
        opens.then_some(start)
    })
}

/// The target that checks a package's comments, relative to its manifest directory: the lib
/// (`[lib] path`, else `src/lib.rs`), else `src/main.rs`, else the first `[[bin]]`'s path. One
/// target per package reads every file, so each comment is checked once, compiled or not.
pub fn owner_target(manifest: &str, exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let table: toml::Table = manifest.parse().ok()?;
    let path_of = |t: &toml::Value| t.get("path").and_then(toml::Value::as_str).map(PathBuf::from);
    if let Some(lib) = table.get("lib").and_then(path_of) {
        return Some(lib);
    }
    ["src/lib.rs", "src/main.rs"]
        .iter()
        .map(PathBuf::from)
        .find(|p| exists(p))
        .or_else(|| table.get("bin")?.as_array()?.first().and_then(path_of))
}

/// Whether a crate root is outside a package's standard target directories: a single file
/// compiled on its own, as a lint's ui test is. Such a crate checks only the files it compiles.
pub fn loose(root: &Path) -> bool {
    !["src", "tests", "benches", "examples"].iter().any(|d| root.starts_with(d)) && root != Path::new("build.rs")
}

/// Every `.rs` file of the package at `dir`, skipping `target`, hidden directories, and nested
/// packages (a directory below `dir` with its own `[package]` manifest).
pub fn package_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                let nested = std::fs::read_to_string(path.join("Cargo.toml")).is_ok_and(|m| m.contains("[package]"));
                if name != "target" && !name.starts_with('.') && !nested {
                    stack.push(path);
                }
            } else if name.ends_with(".rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// The comments of a TOML document: each one's 1-based line and its text after the `#`.
pub fn toml_comments(text: &str) -> Vec<(usize, &str)> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    // The string open at `i`: its closing delimiter, and whether `\\` escapes inside it.
    let mut open: Option<(&str, bool)> = None;
    let (mut i, mut line) = (0, 1);
    while i < bytes.len() {
        // Bytes, not str: `i` steps through multi-byte characters, and every delimiter is ASCII.
        let rest = &bytes[i..];
        match open {
            Some((close, escapes)) => {
                if escapes && bytes[i] == b'\\' {
                    i += 1;
                } else if rest.starts_with(close.as_bytes()) {
                    i += close.len() - 1;
                    open = None;
                }
            }
            None => {
                if let Some(delim) = ["\"\"\"", "'''", "\"", "'"].into_iter().find(|d| rest.starts_with(d.as_bytes())) {
                    open = Some((delim, delim.starts_with('"')));
                    i += delim.len() - 1;
                } else if bytes[i] == b'#' {
                    let end = rest.iter().position(|&b| b == b'\n').map_or(text.len(), |n| i + n);
                    found.push((line, &text[i + 1..end]));
                    i = end;
                    continue;
                }
            }
        }
        if bytes.get(i) == Some(&b'\n') {
            line += 1;
        }
        i += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn every_tombstone_phrase_is_found_on_word_boundaries() {
        for phrase in TOMBSTONES {
            assert_eq!(tombstone(&format!("this {phrase} there")), Some(*phrase));
        }
        assert_eq!(tombstone("the pool is created once and moved to the worker thread"), None);
        assert_eq!(tombstone("Moved From the old module"), Some("moved from"));
        assert_eq!(tombstone("removed"), None);
        assert_eq!(tombstone("it moves to the next row"), None);
        assert_eq!(tombstone("no longer than 4"), None);
        assert_eq!(tombstone("if that directory is gone there is nothing to read"), None);
        assert_eq!(tombstone("unmoved from here"), None);
    }

    #[test]
    fn tmp_paths_are_found_only_as_paths() {
        assert_eq!(tmp_path("see `/tmp/x.md`"), Some(5));
        assert_eq!(tmp_path(" /tmp/a"), Some(1));
        assert_eq!(tmp_path("(~/tmp/b)"), Some(1));
        assert_eq!(tmp_path("/tmp/c"), Some(0));
        assert_eq!(tmp_path("gate/up/tmp/out"), None);
        assert_eq!(tmp_path("{out}_tmp"), None);
        assert_eq!(tmp_path("tmpfs"), None);
    }

    #[test]
    fn the_owner_is_the_lib_then_main_then_the_first_bin() {
        let all = |_: &Path| true;
        let none = |_: &Path| false;
        let main_only = |p: &Path| p == Path::new("src/main.rs");
        assert_eq!(owner_target("[package]\nname = \"a\"", all), Some(PathBuf::from("src/lib.rs")));
        assert_eq!(owner_target("[lib]\npath = \"lib/a.rs\"", none), Some(PathBuf::from("lib/a.rs")));
        assert_eq!(owner_target("[package]\nname = \"a\"", main_only), Some(PathBuf::from("src/main.rs")));
        assert_eq!(owner_target("[[bin]]\nname = \"b\"\npath = \"bin/b.rs\"", none), Some(PathBuf::from("bin/b.rs")));
        assert_eq!(owner_target("[package]\nname = \"a\"", none), None);
    }

    #[test]
    fn a_loose_root_is_outside_the_target_directories() {
        assert!(loose(Path::new("ui/p11_comments.rs")));
        for root in ["src/lib.rs", "src/bin/x.rs", "tests/t.rs", "benches/b.rs", "examples/e.rs", "build.rs"] {
            assert!(!loose(Path::new(root)), "{root}");
        }
    }

    #[test]
    fn a_package_walk_skips_target_hidden_and_nested_packages() {
        let dir = std::env::temp_dir().join(format!("scratchy-purity-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for f in ["src/lib.rs", "tests/t.rs", "build.rs", "target/gen.rs", ".git/x.rs", "inner/src/lib.rs", "notes.md"] {
            let p = dir.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "").unwrap();
        }
        std::fs::write(dir.join("inner/Cargo.toml"), "[package]\nname = \"inner\"").unwrap();
        let found: Vec<_> = package_files(&dir).iter().map(|p| p.strip_prefix(&dir).unwrap().to_path_buf()).collect();
        assert_eq!(found, ["build.rs", "src/lib.rs", "tests/t.rs"].map(PathBuf::from));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn toml_comments_skip_hashes_inside_strings() {
        assert_eq!(toml_comments("a = \"x#y\" # moved from b"), vec![(1, " moved from b")]);
        assert_eq!(toml_comments("s = '''\n# not\n'''"), vec![]);
        assert_eq!(toml_comments("t = \"\"\"\n# not \\\" either\n\"\"\""), vec![]);
        assert_eq!(toml_comments("u = 'a#b'\n# /tmp/x"), vec![(2, " /tmp/x")]);
        assert_eq!(toml_comments("v = \"q\\\"#\" # real"), vec![(1, " real")]);
        assert_eq!(toml_comments("# a — b\nw = \"—#\" # c — d"), vec![(1, " a — b"), (2, " c — d")]);
    }

    #[test]
    fn only_the_packages_own_sources_are_scanned() {
        let m = Path::new("/r/crates/a");
        assert!(scanned(Path::new("/r/crates/a/src/a.rs"), m));
        assert!(!scanned(Path::new("/r/crates/a/target/debug/build/x/out/gen.rs"), m));
        assert!(!scanned(Path::new("/tmp/out_dir/gen.rs"), m));
        assert!(!scanned(Path::new("/other/a.rs"), m));
    }
}
