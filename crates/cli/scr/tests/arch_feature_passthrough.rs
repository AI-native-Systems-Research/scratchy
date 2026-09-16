// SPDX-License-Identifier: Apache-2.0
//! Guards against reintroducing the hand-maintained duplication this file
//! used to check (a passthrough feature in this crate's `Cargo.toml` for
//! every scratchy-models model-scope feature: `arch-<name>`, `<stem>`,
//! `<arch>` (bare arch name = every config in that arch), and `all` — quant
//! presets live on the separate scratchy-quantizations crate, reached via
//! its own `quant` alias, not checked by this test). That duplication is
//! gone: the `scratchy-models` dependency edge is locally renamed to `model`
//! (`model = { package = "scratchy-models", ... }`), and Cargo's `pkg/feature` CLI syntax resolves
//! against that local name and activates the (optional) dependency itself —
//! so `-p scratchy-cli --features model/granite-3.1-2b-instruct` already
//! reaches scratchy-models' own features directly, with nothing to forward by
//! hand. This test fails loudly if that duplication creeps back.
//!
//! Since model-scope feature names on scratchy-models no longer carry a
//! distinguishing text prefix (bare `<stem>`/`<arch>`/`all`), the only
//! reliable check is a direct diff against scratchy-models' own declared
//! feature keys — a naming-convention heuristic can't tell a re-duplicated
//! `granite` apart from a legitimately-scoped new feature by string shape
//! alone.

use std::collections::BTreeSet;
use std::path::Path;

/// Extract top-level `name = [...]` (or `name = []`) feature keys from a
/// `[features]` table. Deliberately dumb line scanning, not a TOML parser:
/// every entry this test cares about is a single line of that form, which is
/// how both files are written. Continuation lines of a multi-line array
/// (`"foo",`) have no top-level `=` and are naturally skipped.
fn feature_keys(cargo_toml: &str) -> BTreeSet<String> {
    let features_start = cargo_toml.find("[features]").expect("no [features] table");
    let after = &cargo_toml[features_start + "[features]".len()..];
    let features_end = after.find("\n[").unwrap_or(after.len());
    let body = &after[..features_end];

    body.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, _) = line.split_once('=')?;
            Some(key.trim().trim_matches('"').to_string())
        })
        .collect()
}

/// Features that legitimately exist on BOTH crates under the same name —
/// not passthrough duplication. `default` is a standard Cargo.toml key every
/// `[features]` table has, unrelated across crates. The backend names
/// (`cuda`/`metal`/...) are real, independent declarations on each crate —
/// `scratchy-cli`'s versions do more than forward (they also gate
/// `scratchy-serving-api`/-worker deps), not hand-forwarded scope features.
///
/// The admission rule, so this list cannot become a dumping ground: a name
/// belongs here ONLY if it is not a model scope — i.e. it is not a checked-in
/// `configs/<arch>/<stem>.json` stem, an `<arch>` directory name, `arch-<name>`,
/// or `all`. Backend tiers (`spyre-hw`) and capability features
/// (`hf-completions`) pass that rule; a model or arch name never does, which is
/// the duplication this test exists to catch.
const SHARED_NON_SCOPE_FEATURES: &[&str] = &[
    "default",
    "cuda",
    "metal",
    "nccl",
    "spyre",
    "spyre-hw",
    "hf-completions",
];

#[test]
fn scratchy_models_is_aliased_model_with_no_hand_forwarded_scope_features() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR unset");
    let models_toml = Path::new(&manifest_dir).join("../../models/arch/Cargo.toml");
    let cli_toml = Path::new(&manifest_dir).join("Cargo.toml");

    let models_src = std::fs::read_to_string(&models_toml)
        .unwrap_or_else(|e| panic!("reading {}: {e}", models_toml.display()));
    let cli_src = std::fs::read_to_string(&cli_toml)
        .unwrap_or_else(|e| panic!("reading {}: {e}", cli_toml.display()));

    assert!(
        cli_src.contains("model = { package = \"scratchy-models\""),
        "crates/cli/scr/Cargo.toml must declare the scratchy-models dependency \
         as `model = {{ package = \"scratchy-models\", ... }}` — that local rename \
         is what makes `--features model/<scratchy-models-feature>` work directly \
         from this crate's build line without a hand-forwarded passthrough feature."
    );

    let models_keys = feature_keys(&models_src);
    let cli_keys = feature_keys(&cli_src);

    let reintroduced: Vec<&String> = models_keys
        .intersection(&cli_keys)
        .filter(|k| !SHARED_NON_SCOPE_FEATURES.contains(&k.as_str()))
        .collect();

    assert!(
        reintroduced.is_empty(),
        "crates/cli/scr/Cargo.toml re-declares scratchy-models feature(s) \
         {reintroduced:?} — these duplicate what `model/<feature>` (the `model` \
         dependency alias) already reaches directly. Remove them instead of \
         hand-forwarding."
    );
}
