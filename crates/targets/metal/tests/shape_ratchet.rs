// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE SHAPE RATCHET (#1094): how many shape constraints are still values rustc does not check,
//! counted in the source, each kind pinned. A change that types one lowers its count and lowers
//! the pin with it; a change that adds one fails here. Counts are of code, not comments or tests:
//! each file is read up to its first `#[cfg(test)]`, its `//` lines skipped.

use std::path::{Path, PathBuf};

/// One kind of door: what it is, the files it is counted in (relative to the workspace root, a
/// directory meaning every `.rs` under it), the texts an occurrence is, the occurrences that are
/// its definition rather than a use, and its pinned count.
struct Door {
    what: &'static str,
    under: &'static [&'static str],
    patterns: &'static [&'static str],
    definitions: &'static [&'static str],
    pinned: usize,
}

const DOORS: &[Door] = &[
    Door {
        what: "a law between config facts asserted at construction (Phase F: type-level dims)",
        under: &["crates/compiler/ir/src/model_facts.rs"],
        patterns: &["assert!("],
        definitions: &[],
        pinned: 11,
    },
    Door {
        what: "a head dim crossing the value→const door (Phase G: spyre's typed dims)",
        under: &["crates"],
        patterns: &["with_config_head_dim("],
        definitions: &["pub fn with_config_head_dim("],
        pinned: 1,
    },
    Door {
        what: "a kernel parameter minted from a value (`AffineBits::new`, …)",
        under: &["crates"],
        patterns: &[
            "AffineBits::new(",
            "AffineGroupSize::new(",
            "NaxGroupSize::new(",
            "Int4Bits::new(",
            "AffineBits::of(",
            "AffineGroupSize::of(",
            "NaxGroupSize::of(",
            "Int4Bits::of(",
            "Gs::new(",
            "Bits::new(",
        ],
        definitions: &[],
        pinned: 10,
    },
    Door {
        what: "an arena slot bound by its index, not a colour hosting its extent (E.3's operands)",
        under: &["crates/targets/metal/src"],
        patterns: &["Binding::ArenaSlot {"],
        definitions: &[],
        pinned: 116,
    },
    Door {
        what: "a step's width read off the graph as a number (E.3's operands)",
        under: &["crates/targets/metal/compiler/src/steps_from_tape.rs"],
        patterns: &["self.graph.shape("],
        definitions: &[],
        pinned: 4,
    },
];

/// The workspace root: this crate is `crates/targets/metal`.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn files(at: &Path, out: &mut Vec<PathBuf>) {
    if at.is_file() {
        out.push(at.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(at) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() && !matches!(name, "tests" | "target" | "benches" | "examples") {
            files(&p, out);
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// `pattern`'s occurrences in `line` that start a word (`Bits::new(` is not `AffineBits::new(`).
fn occurrences(line: &str, pattern: &str) -> usize {
    (line.match_indices(pattern))
        .filter(|(at, _)| {
            let before = line[..*at].chars().next_back();
            !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
        .count()
}

/// `door`'s occurrences in code, by file.
fn count(door: &Door) -> Vec<(PathBuf, usize)> {
    let mut paths = Vec::new();
    for under in door.under {
        files(&root().join(under), &mut paths);
    }
    let mut counts = Vec::new();
    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let code = text.split("#[cfg(test)]").next().unwrap_or("");
        let n: usize = (code.lines())
            .filter(|l| !l.trim_start().starts_with("//"))
            .filter(|l| !door.definitions.iter().any(|d| l.contains(d)))
            .map(|l| {
                door.patterns
                    .iter()
                    .map(|p| occurrences(l, p))
                    .sum::<usize>()
            })
            .sum();
        if n > 0 {
            counts.push((path, n));
        }
    }
    counts
}

#[test]
fn every_door_is_at_its_pin() {
    let mut off = Vec::new();
    for door in DOORS {
        let counts = count(door);
        let total: usize = counts.iter().map(|(_, n)| n).sum();
        if total != door.pinned {
            let by_file = (counts.iter())
                .map(|(p, n)| format!("    {n} {}", p.strip_prefix(root()).unwrap_or(p).display()))
                .collect::<Vec<_>>()
                .join("\n");
            off.push(format!(
                "{}: {total}, pinned at {}{}\n{by_file}",
                door.what,
                door.pinned,
                match total < door.pinned {
                    true => " (lower the pin: the count went down)",
                    false => " (a door was added)",
                }
            ));
        }
    }
    assert!(
        off.is_empty(),
        "the shape ratchet moved:\n{}",
        off.join("\n")
    );
}
