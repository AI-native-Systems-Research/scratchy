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

use std::collections::BTreeMap;
use std::path::Path;

/// A `[features]` table: each feature's name and entries. A reader for this one table's shape,
/// not a TOML parser — `name = ["a", "b"]`, an array on one line or across several, a `#` comment
/// anywhere outside a quoted entry — which is every form both files use.
fn features(cargo_toml: &str) -> BTreeMap<String, Vec<String>> {
    let start = cargo_toml.find("[features]").expect("no [features] table");
    let after = &cargo_toml[start + "[features]".len()..];
    let body = &after[..after.find("\n[").unwrap_or(after.len())];

    let mut table = BTreeMap::new();
    // A feature whose array has not closed yet: its name and the array text so far.
    let mut open: Option<(String, String)> = None;
    for line in body.lines().map(uncommented) {
        let (name, array) = match open.take() {
            Some((name, array)) => (name, array + line),
            None => match line.split_once('=') {
                Some((name, array)) => (name.trim().trim_matches('"').to_string(), array.into()),
                None => continue,
            },
        };
        if array.contains(']') {
            table.insert(name, entries(&array));
        } else {
            open = Some((name, array));
        }
    }
    table
}

/// `line` up to a `#` outside quotes.
fn uncommented(line: &str) -> &str {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..i],
            _ => {}
        }
    }
    line
}

/// The quoted entries of an array's text, `[` to `]`.
fn entries(array: &str) -> Vec<String> {
    let inner = array.split_once('[').map_or("", |(_, rest)| rest);
    let inner = inner.split_once(']').map_or(inner, |(inside, _)| inside);
    inner
        .split(',')
        .map(|e| e.trim().trim_matches('"').to_string())
        .filter(|e| !e.is_empty())
        .collect()
}

/// Features that legitimately exist on BOTH crates under the same name —
/// not passthrough duplication. `default` is a standard Cargo.toml key every
/// `[features]` table has, unrelated across crates. The backend names
/// (`metal`/`spyre`/...) are real, independent declarations on each crate —
/// `scratchy-cli`'s versions do more than forward (they also gate
/// `scratchy-serving-api`/-worker deps), not hand-forwarded scope features.
///
/// The admission rule, so this list cannot become a dumping ground: a name
/// belongs here ONLY if this crate's declaration does more than forward to
/// scratchy-models. The backends and their tiers (`spyre-hw`, `spyre-emu`) gate
/// the serving crates. A feature that only forwards (`x = ["model?/x"]`) is
/// never admitted, whatever it names — a model scope or a capability alike:
/// name the dependency feature where it is used instead (`metal` names
/// `model?/turboquant`, `default` names `model?/hf-completions`), and
/// [`no_feature_of_this_crate_only_forwards_to_scratchy_models`] fails on one.
const SHARED_NON_SCOPE_FEATURES: &[&str] = &["default", "metal", "spyre", "spyre-hw", "spyre-emu"];

/// No feature of this crate only forwards to scratchy-models, whether or not
/// the two crates share its name: `model/<feature>` already reaches every
/// scratchy-models feature, so a forwarder is a second spelling of one switch.
/// Read through [`features`], so a forwarder written across several lines, or
/// with a trailing comment, fails too.
#[test]
fn no_feature_of_this_crate_only_forwards_to_scratchy_models() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR unset");
    let cli_toml = Path::new(&manifest_dir).join("Cargo.toml");
    let cli_src = std::fs::read_to_string(&cli_toml)
        .unwrap_or_else(|e| panic!("reading {}: {e}", cli_toml.display()));

    let forwarders: Vec<String> = features(&cli_src)
        .into_iter()
        .filter(|(_, entries)| {
            !entries.is_empty()
                && entries
                    .iter()
                    .all(|e| e.starts_with("model/") || e.starts_with("model?/"))
        })
        .map(|(name, _)| name)
        .collect();

    assert!(
        forwarders.is_empty(),
        "crates/cli/scr/Cargo.toml declares feature(s) {forwarders:?} that only \
         forward to scratchy-models — `model/<feature>` already reaches them. \
         Name the dependency feature where it is used instead."
    );
}

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

    let models_features = features(&models_src);
    let reintroduced: Vec<String> = features(&cli_src)
        .into_keys()
        .filter(|k| models_features.contains_key(k))
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
