//! Architectural guard: target-specific `cfg` gates (any target under `crates/targets/` or
//! `backends/`, or the removed `cuda`) are a code smell this workspace forbids outside of dedicated
//! test code.
//!
//! Principle, not a file list: a backend is selected by which target crate the
//! binary links (unconditional `dep:` + a feature on the *binary*), and dispatch
//! happens through neutral trait/registry seams — NOT by sprinkling
//! `#[cfg(feature = "<target>")]` / `#[cfg(any(<target>, cuda))]` through neutral code
//! or even inside the target crates (which are unconditionally their target).
//!
//! The same holds for the manifests: no neutral crate has a feature named for a
//! target or depends on one, and a crate named for a target lives under
//! `crates/targets/<target>/`. CI runs this file on every push.
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

/// Where the targets live, from the repository root: the root workspace's target crates, and the
/// backends that are workspaces of their own (spyre's left `crates/targets` with its workspace).
const TARGET_HOMES: &[&str] = &["crates/targets", "backends"];

/// The targets' names: each directory under a [`TARGET_HOMES`] entry, read off the tree so a new
/// target is checked without a word here, and the removed CUDA backend's, so it does not creep
/// back. A home that is gone fails here, rather than its targets silently leaving the check.
fn target_names() -> Vec<String> {
    let mut names = vec!["cuda".to_string()];
    for home in TARGET_HOMES {
        let dir = workspace_root().join(home);
        let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        names.extend(
            (entries.flatten())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().to_str().map(str::to_string)),
        );
    }
    names
}

/// A source line that gates on a target's cfg (covers `feature = "<target>"`,
/// bare `cfg(<target>)`, and `any(...)`/`all(...)`/`not(...)` wrappers).
fn line_gates_on_target(line: &str, targets: &[String]) -> bool {
    let mut rest = line;
    while let Some(i) = rest.find("cfg(") {
        let seg = &rest[i + 4..];
        if targets.iter().any(|t| has_word(seg, t)) {
            return true;
        }
        rest = &rest[i + 4..];
    }
    false
}

/// A per-target crate root: a directory named for a target that is itself a
/// crate (`Cargo.toml` present). These ARE a backend and are skipped by the
/// agnostic scans. A mere module dir named for one has no `Cargo.toml`, so it
/// is NOT skipped — agnostic code there must be cleaned.
fn is_per_target_crate(dir: &Path, targets: &[String]) -> bool {
    (dir.file_name().and_then(|n| n.to_str())).is_some_and(|n| targets.iter().any(|t| t == n))
        && dir.join("Cargo.toml").exists()
}

fn is_test_path(p: &Path) -> bool {
    p.components().any(|c| c.as_os_str() == "tests")
        || p.file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| f.ends_with("_tests.rs"))
}

/// All target cfg-gate violations under `rel` (a path relative to the
/// workspace root), as `"relpath:line: content"`, skipping `target/` build
/// output and any test code.
fn target_cfg_violations(rel: &str) -> Vec<String> {
    let root = workspace_root();
    let targets = target_names();
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
                if name == "target" || name == "tests" || is_per_target_crate(&path, &targets) {
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
                if line_gates_on_target(line, &targets) {
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
        "\n`{zone}` has {} target cfg gate(s) — neutral/target code must \
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

// NOTE: the crates under `crates/targets/` are split per-target — they ARE a
// specific backend and may name it freely. They are deliberately
// NOT in the agnostic zone list above. (A per-target crate must still pull its
// deps unconditionally rather than cfg-gate them, but that is a Cargo concern,
// not a source-cfg one.)

// ---- Principle: neutral code never imports a target-named crate ---------------

fn imports_target_crate(rel: &str, krate: &str) -> Vec<String> {
    let root = workspace_root();
    let targets = target_names();
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
                if name != "target" && name != "tests" && !is_per_target_crate(&path, &targets) {
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

/// Hardware-agnostic code must not name a backend at all — including the
/// removed CUDA backend's crates and concepts (CUDA graphs), so they do not
/// creep back in. Dispatch goes
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

// ---- Principle: no neutral crate's manifest names a target --------------------
//
// The cfg scans above read source. These read the manifests, through `cargo metadata`, so a
// target cannot come back as a feature flag or a dependency of a neutral crate either. A neutral
// crate is every member of the root workspace outside the target homes ([`TARGET_HOMES`]).

/// One root-workspace member, as `cargo metadata` reports it.
struct Package {
    name: String,
    /// Its manifest, relative to the workspace root.
    manifest: String,
    /// Each feature with what it enables.
    features: Vec<(String, Vec<String>)>,
    /// Each dependency: its name and, for a platform-specific one, the platform's cfg.
    deps: Vec<(String, Option<String>)>,
}

fn workspace_packages() -> Vec<Package> {
    let root = workspace_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = std::process::Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()
        .expect("run cargo metadata");
    assert!(
        out.status.success(),
        "cargo metadata failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout).expect("parse metadata");
    let text = |v: &serde_json::Value| v.as_str().unwrap_or_default().to_string();
    let packages = meta["packages"]
        .as_array()
        .expect("metadata lists packages");
    let packages: Vec<Package> = (packages.iter())
        .map(|p| Package {
            name: text(&p["name"]),
            manifest: Path::new(p["manifest_path"].as_str().unwrap_or_default())
                .strip_prefix(&root)
                .map(|m| m.display().to_string())
                .unwrap_or_else(|_| text(&p["manifest_path"])),
            features: (p["features"].as_object().into_iter().flatten())
                .map(|(f, on)| {
                    (
                        f.clone(),
                        on.as_array().into_iter().flatten().map(text).collect(),
                    )
                })
                .collect(),
            deps: (p["dependencies"].as_array().into_iter().flatten())
                .map(|d| (text(&d["name"]), d["target"].as_str().map(str::to_string)))
                .collect(),
        })
        .collect();
    assert!(!packages.is_empty(), "cargo metadata lists no packages");
    packages
}

/// A crate outside every target home ([`TARGET_HOMES`]).
fn is_neutral(p: &Package) -> bool {
    !(TARGET_HOMES.iter()).any(|home| p.manifest.starts_with(&format!("{home}/")))
}

/// `name` is named for a target. A dependency only a target may take that is not named for one
/// (metal's `objc2` bridge) is CI's to refuse: a neutral crate taking it fails to build on the
/// targets it does not support.
fn names_target(name: &str, targets: &[String]) -> bool {
    targets.iter().any(|t| has_word(name, t))
}

/// Every feature of a neutral crate that is named for a target or enables one, as
/// `"manifest: feature ..."`.
fn neutral_feature_violations(packages: &[Package], targets: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for p in packages.iter().filter(|p| is_neutral(p)) {
        for (feature, enables) in &p.features {
            if names_target(feature, targets) {
                out.push(format!("{}: feature `{feature}`", p.manifest));
            }
            for e in enables.iter().filter(|e| names_target(e, targets)) {
                out.push(format!("{}: feature `{feature}` enables `{e}`", p.manifest));
            }
        }
    }
    out.sort();
    out
}

/// Every dependency of a neutral crate that is a target's, or that only some platforms get, as
/// `"manifest: dependency ..."`.
fn neutral_dependency_violations(packages: &[Package], targets: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for p in packages.iter().filter(|p| is_neutral(p)) {
        for (dep, platform) in &p.deps {
            if names_target(dep, targets) {
                out.push(format!("{}: dependency `{dep}`", p.manifest));
            }
            if let Some(platform) = platform {
                out.push(format!(
                    "{}: dependency `{dep}` only for `{platform}`",
                    p.manifest
                ));
            }
        }
    }
    out.sort();
    out
}

/// Every crate named for a target that is not under that target's directory in a target home
/// (`<home>/<target>/`, for each of [`TARGET_HOMES`]).
fn misplaced_target_crates(packages: &[Package], targets: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for p in packages {
        for t in targets.iter().filter(|t| has_word(&p.name, t)) {
            let homes: Vec<String> = TARGET_HOMES.iter().map(|h| format!("{h}/{t}/")).collect();
            if !homes.iter().any(|h| p.manifest.starts_with(h)) {
                out.push(format!(
                    "{} is `{}`, not under {}",
                    p.name,
                    p.manifest,
                    homes.join(" or ")
                ));
            }
        }
    }
    out.sort();
    out
}

fn assert_none(what: &str, v: Vec<String>) {
    assert!(
        v.is_empty(),
        "\n{what} ({} found):\n{}\n",
        v.len(),
        v.join("\n")
    );
}

#[test]
fn neutral_crates_have_no_target_feature() {
    assert_none(
        "a neutral crate (core, serving, compiler, ...) has a feature named for a target or \
         enabling one; a build selects its target by the binary it links, not by a flag",
        neutral_feature_violations(&workspace_packages(), &target_names()),
    );
}

#[test]
fn neutral_crates_depend_on_no_target() {
    assert_none(
        "a neutral crate (core, serving, compiler, ...) depends on a target's crate, or has a \
         dependency only some platforms get; it must build the same on every target",
        neutral_dependency_violations(&workspace_packages(), &target_names()),
    );
}

#[test]
fn target_crates_live_under_their_target() {
    assert_none(
        "a crate named for a target is outside its target home's <target>/ directory",
        misplaced_target_crates(&workspace_packages(), &target_names()),
    );
}

/// The scans above report what they should: a planted violation of each kind is found, and a
/// target's own crate is left alone, in either target home.
#[test]
fn manifest_scans_catch_planted_violations() {
    let targets = vec!["cuda".to_string(), "metal".to_string(), "spyre".to_string()];
    let pkg = |name: &str, manifest: &str| Package {
        name: name.to_string(),
        manifest: manifest.to_string(),
        features: Vec::new(),
        deps: Vec::new(),
    };
    let mut engine = pkg(
        "scratchy-serving-engine",
        "crates/serving/engine/Cargo.toml",
    );
    engine.features = vec![
        ("metal".to_string(), Vec::new()),
        ("fast".to_string(), vec!["dep:objc2-metal".to_string()]),
        ("mlx-affine".to_string(), Vec::new()),
    ];
    engine.deps = vec![
        ("scratchy-target-spyre".to_string(), None),
        (
            "libc".to_string(),
            Some("cfg(target_os = \"macos\")".to_string()),
        ),
        ("serde".to_string(), None),
    ];
    let mut metal = pkg("scratchy-target-metal", "crates/targets/metal/Cargo.toml");
    metal.features = vec![("metal4".to_string(), Vec::new())];
    metal.deps = vec![(
        "objc2".to_string(),
        Some("cfg(target_os = \"macos\")".to_string()),
    )];
    let stray = pkg("scratchy-cuda-kernels", "crates/kernels/Cargo.toml");
    // A target crate in the other home: its target's own, so neither neutral nor misplaced.
    let mut backend = pkg(
        "scratchy-spyre-runtime",
        "backends/spyre/crates/runtime/Cargo.toml",
    );
    backend.features = vec![("spyre-hw".to_string(), Vec::new())];
    backend.deps = vec![("scratchy-target-spyre".to_string(), None)];
    let packages = [engine, metal, stray, backend];

    assert_eq!(
        neutral_feature_violations(&packages, &targets),
        [
            "crates/serving/engine/Cargo.toml: feature `fast` enables `dep:objc2-metal`",
            "crates/serving/engine/Cargo.toml: feature `metal`",
        ]
    );
    assert_eq!(
        neutral_dependency_violations(&packages, &targets),
        [
            "crates/serving/engine/Cargo.toml: dependency `libc` only for `cfg(target_os = \"macos\")`",
            "crates/serving/engine/Cargo.toml: dependency `scratchy-target-spyre`",
        ]
    );
    assert_eq!(
        misplaced_target_crates(&packages, &targets),
        [
            "scratchy-cuda-kernels is `crates/kernels/Cargo.toml`, not under crates/targets/cuda/ \
             or backends/cuda/"
        ]
    );
}
