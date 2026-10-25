//! Architectural guard: target-specific (`cuda`/`metal`) `cfg` gates are a code
//! smell that this workspace forbids outside of dedicated test code.
//!
//! Principle, not a file list: a backend is selected by which target crate the
//! binary links (unconditional `dep:` + a feature on the *binary*), and dispatch
//! happens through neutral trait/registry seams — NOT by sprinkling
//! `#[cfg(feature = "cuda")]` / `#[cfg(any(cuda, metal))]` through neutral code
//! or even inside the target crates (which are unconditionally their target).
//!
//! These tests walk whole directory zones and fail listing every violation, so
//! they keep working as files move. Renaming a file or moving code between
//! crates cannot make a violation invisible.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("locate workspace root")
}

/// `needle` occurs in `hay` bounded by non-`[A-Za-z0-9_]` on both sides.
fn has_word(hay: &str, needle: &str) -> bool {
    let (h, n) = (hay.as_bytes(), needle.as_bytes());
    let word = |c: u8| c == b'_' || c.is_ascii_alphanumeric();
    let mut i = 0;
    while i + n.len() <= h.len() {
        if &h[i..i + n.len()] == n {
            let before = i > 0 && word(h[i - 1]);
            let after = i + n.len() < h.len() && word(h[i + n.len()]);
            if !before && !after {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// A source line that gates on a `cuda`/`metal` cfg (covers `feature = "cuda"`,
/// bare `cfg(cuda)`, and `any(...)`/`all(...)`/`not(...)` wrappers). A `#[cfg(...)]`
/// or `#![cfg(...)]` attribute that rustfmt wrapped over several lines reaches this
/// joined into one line ([`gates_in`]), so its target is seen beside its `cfg(`.
fn line_gates_on_target(line: &str) -> bool {
    let mut rest = line;
    while let Some(i) = rest.find("cfg(") {
        let seg = &rest[i + 4..];
        if has_word(seg, "cuda") || has_word(seg, "metal") {
            return true;
        }
        rest = &rest[i + 4..];
    }
    false
}

/// A per-target crate root: a directory named `cuda`/`metal` that is itself a
/// crate (`Cargo.toml` present). These ARE a backend and are skipped by the
/// agnostic scans. A mere module dir like `compiler/macros/src/metal/` has no
/// `Cargo.toml`, so it is NOT skipped — agnostic code there must be cleaned.
fn is_per_target_crate(dir: &Path) -> bool {
    matches!(
        dir.file_name().and_then(|n| n.to_str()),
        Some("cuda") | Some("metal")
    ) && dir.join("Cargo.toml").exists()
}

fn is_test_path(p: &Path) -> bool {
    p.components().any(|c| c.as_os_str() == "tests")
        || p.file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| f.ends_with("_tests.rs"))
}

/// Every `cuda`/`metal` cfg gate in one file's `text`, as `(line, content)` with
/// 1-based lines. A line starting a `#[cfg(` or `#![cfg(` attribute is read with the
/// lines after it up to the `]` that closes the attribute's `#[`, so a gate rustfmt
/// wrapped is one gate, reported at its first line with its whitespace collapsed; the
/// lines it spans are not read again. Every other line is read alone. Brackets in
/// string literals inside a `cfg` are not told apart from the attribute's own (none
/// exist in the zones).
fn gates_in(text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        let end = if trimmed.starts_with("#[cfg(") || trimmed.starts_with("#![cfg(") {
            attribute_end(&lines, i)
        } else {
            i
        };
        let content = if end == i {
            line.trim().to_string()
        } else {
            let joined: Vec<&str> = (lines[i..=end].iter())
                .flat_map(|l| l.split_whitespace())
                .collect();
            joined.join(" ")
        };
        if line_gates_on_target(&content) {
            out.push((i + 1, content));
        }
        i = end + 1;
    }
    out
}

/// The line on which the attribute starting on line `start` closes: where the `]`
/// matching its `#[` falls. One that never closes is read as its own line alone, as
/// any line was before, so it cannot swallow the gates after it.
fn attribute_end(lines: &[&str], start: usize) -> usize {
    let mut depth = 0i32;
    for (n, line) in lines.iter().enumerate().skip(start) {
        let from = if n == start {
            line.find('#').unwrap_or(0)
        } else {
            0
        };
        for c in line[from..].chars() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return n;
                    }
                }
                _ => {}
            }
        }
    }
    start
}

/// All `cuda`/`metal` cfg-gate violations under `rel` (a path relative to the
/// workspace root), as `"relpath:line: content"`, skipping `target/` build
/// output and any test code. Each file is read by [`gates_in`], attributes whole.
fn target_cfg_violations(rel: &str) -> Vec<String> {
    let root = workspace_root();
    let base = root.join(rel);
    let mut out = Vec::new();
    let mut stack = vec![base];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name == "target" || name == "tests" || is_per_target_crate(&path) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|x| x.to_str()) != Some("rs") || is_test_path(&path) {
                continue;
            }
            let Ok(txt) = fs::read_to_string(&path) else {
                continue;
            };
            let relp = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .display()
                .to_string();
            for (n, content) in gates_in(&txt) {
                out.push(format!("{relp}:{n}: {content}"));
            }
        }
    }
    out.sort();
    out
}

fn assert_zone_clean(zone: &str) {
    let v = target_cfg_violations(zone);
    assert!(
        v.is_empty(),
        "\n`crates/{zone}` has {} cuda/metal cfg gate(s) — neutral/target code must \
         not gate on a backend cfg (dispatch through a trait/registry seam; target \
         crates are unconditionally their target):\n{}\n",
        v.len(),
        v.join("\n")
    );
}

// ---- Principle: no neutral zone gates on a target cfg -------------------------

#[test]
fn no_target_cfg_in_compiler() {
    assert_zone_clean("crates/compiler");
}

#[test]
fn no_target_cfg_in_serving() {
    assert_zone_clean("crates/serving");
}

#[test]
fn no_target_cfg_in_models() {
    assert_zone_clean("crates/models");
}

#[test]
fn no_target_cfg_in_core() {
    assert_zone_clean("crates/core");
}

#[test]
fn no_target_cfg_in_cli() {
    assert_zone_clean("crates/cli");
}

#[test]
fn no_target_cfg_in_layers() {
    assert_zone_clean("crates/layers");
}

#[test]
fn no_target_cfg_in_tensors() {
    assert_zone_clean("crates/tensors");
}

// ---- The guard reads an attribute whole, however rustfmt wrapped it ----------

/// The lines [`gates_in`] reports a gate at.
fn gate_lines(text: &str) -> Vec<usize> {
    gates_in(text).into_iter().map(|(n, _)| n).collect()
}

/// #885's gemma4-moe module gate as rustfmt wrapped it (101 characters on one line):
/// its `cfg(` is on line 1 and its targets on line 3, which a line-at-a-time read missed.
#[test]
fn a_wrapped_gate_is_one_gate_at_its_first_line() {
    let wrapped = "#[cfg(all(\n    feature = \"arch-gemma4-moe\",\n    any(feature = \"metal\", \
                   feature = \"spyre\", feature = \"cuda\")\n))]\npub mod gemma4_moe;";
    assert_eq!(gate_lines(wrapped), [1]);
    let (_, content) = &gates_in(wrapped)[0];
    assert_eq!(
        content,
        "#[cfg(all( feature = \"arch-gemma4-moe\", any(feature = \"metal\", feature = \"spyre\", \
         feature = \"cuda\") ))]"
    );
}

#[test]
fn the_same_gate_on_one_line_is_one_gate() {
    let one = "#[cfg(all(feature = \"arch-gemma4-moe\", any(feature = \"metal\", feature = \
               \"spyre\", feature = \"cuda\")))]\npub mod gemma4_moe;";
    assert_eq!(gate_lines(one), [1]);
}

/// #885's workaround (`f8a6ec4a`): two one-line attributes, only the second naming a target.
#[test]
fn stacked_attributes_count_the_one_naming_a_target() {
    let stacked = "#[cfg(feature = \"arch-gemma4-moe\")]\n#[cfg(any(feature = \"metal\", \
                   feature = \"spyre\", feature = \"cuda\"))]\npub mod gemma4_moe;";
    assert_eq!(gate_lines(stacked), [2]);
}

#[test]
fn a_wrapped_gate_naming_no_target_is_none() {
    let wrapped = "#[cfg(all(feature = \"x\",\n    feature = \"y\"))]\nfn f() {}";
    assert_eq!(gate_lines(wrapped), Vec::<usize>::new());
}

/// An attribute ends at its own `]`: a target named in the item after it is not the gate's.
#[test]
fn a_target_in_the_gated_item_is_not_the_gate() {
    let after = "#[cfg(feature = \"x\")]\nfn metal() {}";
    assert_eq!(gate_lines(after), Vec::<usize>::new());
}

/// An attribute whose brackets never close is its own line alone: the gates after it
/// still count where they are.
#[test]
fn an_unclosed_attribute_does_not_hide_the_gates_after_it() {
    let unclosed = "#[cfg(all(feature = \"x\",\nfn f() {}\n#[cfg(feature = \"metal\")]\nfn g() {}";
    assert_eq!(gate_lines(unclosed), [3]);
    let then_wrapped = "#[cfg(all(feature = \"x\",\nfn f() {}\n#[cfg(any(\n    feature = \"metal\",\n    \
                        feature = \"spyre\"\n))]\nfn g() {}";
    assert_eq!(gate_lines(then_wrapped), [3]);
}

#[test]
fn a_wrapped_inner_attribute_is_one_gate() {
    let inner = "#![cfg(any(\n    feature = \"metal\",\n    feature = \"spyre\"\n))]\nuse x;";
    assert_eq!(gate_lines(inner), [1]);
}

// NOTE: `crates/targets/cuda/{builder,cost-sweep}`, `crates/targets/metal/cost-sweep`,
// `crates/targets/{cuda,metal}`, and `crates/serving/cuda` are split per-target
// — they ARE a specific backend and may name it freely. They are deliberately
// NOT in the agnostic zone list above. (A per-target crate must still pull its
// deps unconditionally rather than cfg-gate them, but that is a Cargo concern,
// not a source-cfg one.)

// ---- Principle: neutral code never imports a target-named crate ---------------

fn imports_target_crate(rel: &str, krate: &str) -> Vec<String> {
    let root = workspace_root();
    let mut out = Vec::new();
    let mut stack = vec![root.join(rel)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name != "target" && name != "tests" && !is_per_target_crate(&path) {
                    stack.push(path);
                }
                continue;
            }
            if path.extension().and_then(|x| x.to_str()) != Some("rs") || is_test_path(&path) {
                continue;
            }
            let Ok(txt) = fs::read_to_string(&path) else {
                continue;
            };
            let relp = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .display()
                .to_string();
            for (n, line) in txt.lines().enumerate() {
                if line.contains(krate) {
                    out.push(format!("{relp}:{}: {}", n + 1, line.trim()));
                }
            }
        }
    }
    out.sort();
    out
}

const AGNOSTIC_ZONES: [&str; 6] = [
    "crates/serving",
    "crates/compiler",
    "crates/models",
    "crates/cli",
    "crates/layers",
    "crates/core",
];

/// Hardware-agnostic code must not name a backend at all — not the cuda-named
/// crate (its neutral facade belongs in a neutral crate), not the cuda serving
/// crate, and not backend-specific concepts like CUDA graphs. Dispatch goes
/// through neutral seams.
fn agnostic_symbol_violations(symbol: &str) -> Vec<String> {
    let mut v = Vec::new();
    for zone in AGNOSTIC_ZONES {
        v.extend(imports_target_crate(zone, symbol));
    }
    v.sort();
    v
}

#[test]
fn agnostic_zones_do_not_name_cuda_crate() {
    let v = agnostic_symbol_violations("scratchy_target_cuda");
    assert!(
        v.is_empty(),
        "\nhardware-agnostic code names `scratchy_target_cuda` ({} site(s)) — hoist the neutral facade into a neutral crate:\n{}\n",
        v.len(),
        v.join("\n")
    );
}

#[test]
fn agnostic_zones_do_not_name_cuda_serving_crate() {
    let v = agnostic_symbol_violations("scratchy_serving_cuda");
    assert!(
        v.is_empty(),
        "\nhardware-agnostic code names `scratchy_serving_cuda` ({} site(s)) — route through the backend seam:\n{}\n",
        v.len(),
        v.join("\n")
    );
}

#[test]
fn agnostic_zones_do_not_reference_cuda_graphs() {
    let v = agnostic_symbol_violations("CudaGraph");
    assert!(
        v.is_empty(),
        "\nhardware-agnostic code references CUDA-graph concepts ({} site(s)) — serving/compiler must not know about CUDA graphs:\n{}\n",
        v.len(),
        v.join("\n")
    );
}

// ---- Principle: the cuda crate does not know about metal ----------------------

#[test]
fn cuda_crate_does_not_reexport_metal() {
    let v = imports_target_crate("crates/targets/cuda", "scratchy_target_metal");
    assert!(
        v.is_empty(),
        "\n`scratchy-target-cuda` references `scratchy_target_metal` ({} site(s)) — a target \
         crate must not multiplex another backend:\n{}\n",
        v.len(),
        v.join("\n")
    );
}
