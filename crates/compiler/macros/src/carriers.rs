// SPDX-License-Identifier: Apache-2.0
//! The carriers a build compiles: ONE enumeration, shared by both build stages.
//!
//! Stage 1 (`scratchy-model-facts`' build script) and stage 2 (scratchy-models'
//! `scratchy-forwards.rs`) run the same front end over the same arches, so both read their
//! selection from here. Each stage's build script runs with the same feature names (stage 1's are
//! forwarded one-for-one from scratchy-models), so the `CARGO_FEATURE_*` reads below select the
//! same arches, models and quant presets in both.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

/// One arch a build compiles.
pub struct CarrierTarget {
    /// `dsl/<arch>.py`.
    pub dsl: PathBuf,
    /// The arch's module name (`-` → `_`).
    pub mod_name: String,
    /// `configs/<arch>/`.
    pub configs_dir: PathBuf,
}

impl CarrierTarget {
    /// Parse the arch's DSL carrier, telling cargo to rerun the build script when the DSL or any of
    /// the arch's configs changes.
    pub fn carrier(&self) -> crate::PythonCarrier {
        println!("cargo:rerun-if-changed={}", self.dsl.display());
        // Watch the dir itself (its mtime bumps on add/remove) plus each file.
        println!("cargo:rerun-if-changed={}", self.configs_dir.display());
        if let Ok(rd) = std::fs::read_dir(&self.configs_dir) {
            for e in rd.flatten() {
                println!("cargo:rerun-if-changed={}", e.path().display());
            }
        }
        let dsl = std::fs::read_to_string(&self.dsl)
            .unwrap_or_else(|e| panic!("read {}: {e}", self.dsl.display()));
        crate::parse_python_file(&dsl)
            .unwrap_or_else(|e| panic!("carrier {}: {e}", self.dsl.display()))
    }
}

/// The arches this build compiles (its `arch-<name>` feature and a backend it compiles for), with
/// the quant scope set: `SCRATCHY_QUANTS` becomes the union of every compiled arch's enabled
/// presets, which `config.rs` intersects with each arch's own `quantizations.json`.
///
/// `arch_root` is scratchy-models' crate dir (its `dsl/` and `configs/`); `enabled_presets` is
/// `scratchy_quantizations::enabled_presets()`, read by the caller (a build-dependency of its own).
///
/// Call once, before any parallel fan-out: it sets an environment variable.
pub fn scope_compiled_carriers(
    arch_root: &Path,
    enabled_presets: &HashSet<&'static str>,
) -> CompiledCarriers {
    let dsl_dir = arch_root.join("dsl");
    println!("cargo:rerun-if-changed={}", dsl_dir.display());
    let mut dsl_files: Vec<PathBuf> = std::fs::read_dir(&dsl_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dsl_dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".py"))
        })
        .collect();
    dsl_files.sort();
    let targets: Vec<CarrierTarget> = dsl_files
        .iter()
        .filter_map(|dsl| {
            let arch = dsl
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".py"))?
                .to_string();
            // Its `arch-<name>` feature and a backend it compiles for, as the `pub mod <name>`
            // cfg in scratchy-models' src/lib.rs has it: never an arch whose mod won't compile.
            if !crate::arch_compiled(&arch) {
                return None;
            }
            Some(CarrierTarget {
                dsl: dsl.clone(),
                mod_name: arch.replace('-', "_"),
                configs_dir: arch_root.join("configs").join(&arch),
            })
        })
        .collect();
    let quant_union: BTreeSet<String> = targets
        .iter()
        .flat_map(|t| enabled_quants(&t.configs_dir.join("quantizations.json"), enabled_presets))
        .collect();
    let quants = quant_union.iter().cloned().collect::<Vec<_>>().join(",");
    // SAFETY: the caller sets this once, before any thread that reads the environment starts.
    unsafe {
        std::env::set_var("SCRATCHY_QUANTS", &quants);
    }
    CompiledCarriers {
        targets,
        quants: quant_union,
    }
}

/// What [`scope_compiled_carriers`] scoped.
pub struct CompiledCarriers {
    /// The arches this build compiles, sorted by DSL file name.
    pub targets: Vec<CarrierTarget>,
    /// The quant presets in scope: the union `SCRATCHY_QUANTS` holds.
    pub quants: BTreeSet<String>,
}

/// The quant presets enabled for one arch: its `quantizations.json` ∩ the enabled preset features
/// (no `quant-` prefix on scratchy-quantizations, as `<stem>` has no `model-` prefix on
/// scratchy-models). Quant scope lives on scratchy-quantizations, not scratchy-models (so
/// `crates/cli/scr/Cargo.toml` can alias it `quant`, distinct from the `model` alias — Cargo
/// forbids aliasing the same crate twice), so the presets come from
/// `scratchy_quantizations::enabled_presets()`, baked in at that crate's own compile time via
/// `cfg!`.
fn enabled_quants(qpath: &Path, enabled_presets: &HashSet<&'static str>) -> Vec<String> {
    // Only watch the file if it EXISTS. `rerun-if-changed` on a missing path makes cargo rerun the
    // build script on every build — and 6 arches (the vision ones + modernbert) ship no
    // quantizations.json. Adds/removes of the file are still caught: the configs dir is watched.
    if qpath.is_file() {
        println!("cargo:rerun-if-changed={}", qpath.display());
    }
    let mut enabled: Vec<String> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(qpath)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(arr) = json.get("quantizations").and_then(|v| v.as_array())
    {
        for entry in arr {
            let name = match entry {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(o) if o.len() == 1 => o.keys().next().unwrap().clone(),
                _ => continue,
            };
            if enabled_presets.contains(name.as_str()) {
                enabled.push(name);
            }
        }
    }
    enabled
}
