// SPDX-License-Identifier: Apache-2.0
//! Every arch whose carrier runs a recurrent mixer (`gated_delta_net` or `mamba2`) turns on this
//! crate's `gdn`, and no other arch does. Cargo unifies features, so a wide scope (`all`) always
//! has `gdn` and only a narrow one shows a missing edge: this checks every `arch-<name>` feature
//! against its carrier's text instead, with no model built. The features are cargo's own reading
//! of `Cargo.toml` (`cargo metadata`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

/// The calls a carrier makes to run a recurrent mixer: the ops whose emitted state config names
/// the `gdn` re-export.
const RECURRENT_MIXER_CALLS: [&str; 2] = ["gated_delta_net(", "mamba2("];

/// This crate's `[features]` table, as cargo reads it.
fn features(manifest_dir: &Path) -> BTreeMap<String, Vec<String>> {
    let out = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .arg("--manifest-path")
        .arg(manifest_dir.join("Cargo.toml"))
        .output()
        .expect("run cargo metadata");
    assert!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout).expect("metadata JSON");
    let package = (meta["packages"].as_array().expect("packages"))
        .iter()
        .find(|p| p["name"] == env!("CARGO_PKG_NAME"))
        .expect("this package in its own workspace's metadata");
    let table = package["features"].as_object().expect("features table");
    (table.iter())
        .map(|(name, entries)| {
            let entries = (entries.as_array().expect("feature entries").iter())
                .map(|e| e.as_str().expect("a feature entry is a string").to_string())
                .collect();
            (name.clone(), entries)
        })
        .collect()
}

/// Every feature `feature` turns on, itself included (its `dep:` and `pkg/feature` entries are
/// not features of this crate).
fn closure(table: &BTreeMap<String, Vec<String>>, feature: &str) -> BTreeSet<String> {
    let mut on = BTreeSet::new();
    let mut todo = vec![feature.to_string()];
    while let Some(f) = todo.pop() {
        if let Some(entries) = table.get(&f)
            && on.insert(f)
        {
            todo.extend(entries.iter().cloned());
        }
    }
    on
}

#[test]
fn an_arch_turns_on_gdn_exactly_when_it_runs_a_recurrent_mixer() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let table = features(manifest_dir);
    assert_eq!(
        table.get("gdn").map(Vec::as_slice),
        Some(&["dep:scratchy-state-decoder-gdn".to_string()][..]),
        "`gdn` pulls the recurrent state crate and nothing else"
    );

    let dsl_dir = manifest_dir.join("dsl");
    let mut recurrent: BTreeSet<String> = BTreeSet::new();
    let mut carriers = 0;
    for entry in std::fs::read_dir(&dsl_dir).expect("dsl/").flatten() {
        let path = entry.path();
        let Some(arch) =
            (path.file_name().and_then(|n| n.to_str())).and_then(|n| n.strip_suffix(".py"))
        else {
            continue;
        };
        carriers += 1;
        let text = std::fs::read_to_string(&path).expect("read a carrier");
        let feature = format!("arch-{arch}");
        assert!(
            table.contains_key(&feature),
            "carrier dsl/{arch}.py has no `{feature}` feature"
        );
        if RECURRENT_MIXER_CALLS.iter().any(|call| text.contains(call)) {
            recurrent.insert(feature);
        }
    }
    assert!(carriers > 0, "no carriers under {}", dsl_dir.display());
    assert!(
        !recurrent.is_empty(),
        "no carrier runs a recurrent mixer: the scan reads the wrong calls"
    );

    for feature in table.keys().filter(|f| f.starts_with("arch-")) {
        let on = closure(&table, feature);
        let runs: Vec<&String> = on.intersection(&recurrent).collect();
        assert_eq!(
            on.contains("gdn"),
            !runs.is_empty(),
            "`{feature}` turns on {runs:?}, whose carriers run a recurrent mixer, so it must turn \
             on `gdn` exactly when that list is non-empty"
        );
    }
}
