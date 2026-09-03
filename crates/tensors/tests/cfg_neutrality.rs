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
/// bare `cfg(cuda)`, and `any(...)`/`all(...)`/`not(...)` wrappers).
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

/// All `cuda`/`metal` cfg-gate violations under `rel` (a path relative to the
/// workspace root), as `"relpath:line: content"`, skipping `target/` build
/// output and any test code.
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
            for (n, line) in txt.lines().enumerate() {
                if line_gates_on_target(line) {
                    out.push(format!("{relp}:{}: {}", n + 1, line.trim()));
                }
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
