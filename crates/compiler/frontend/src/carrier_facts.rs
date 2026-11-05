// SPDX-License-Identifier: Apache-2.0
//! What a carrier's configs and arch.json say about the derived dims its weights name: the facts
//! the typed-Rust emit pass reads ([`typed_emit::CarrierFacts`]), for the build script and for a
//! test that types a program of its own.

use crate::{bundles, classified::Program, dim, hf_config, typed_emit};

/// What the carrier `stem`'s configs and arch.json say about its derived dims, by the config rules
/// the config loader uses (`hf_config`): each gated rule's flag over every config of the carrier (a
/// head's: each config of the archs it ships in that carries it), its arch.json `expr` params, and
/// the weight bundles `program` passes whole to an op, derived from every config alike.
pub fn carrier_facts(
    arch_root: &std::path::Path,
    stem: &str,
    vision: bool,
    program: &Program,
    read_files: &mut Vec<std::path::PathBuf>,
) -> typed_emit::CarrierFacts {
    let configs_dir = arch_root.join("configs").join(stem);
    let arch_json = configs_dir.join("arch.json");
    read_files.push(arch_json.clone());
    let read = |path: &std::path::Path| -> serde_json::Value {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("carrier facts: read {}: {e}", path.display()));
        serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("carrier facts: parse {}: {e}", path.display()))
    };
    let facts = match arch_json.exists() {
        true => hf_config::arch_facts(&read(&arch_json))
            .unwrap_or_else(|e| panic!("carrier facts: {}: {e}", arch_json.display())),
        false => hf_config::ArchFacts::default(),
    };
    let mut model_configs = |dir: &std::path::Path| -> Vec<std::path::PathBuf> {
        read_files.push(dir.to_path_buf());
        let mut paths: Vec<_> = (std::fs::read_dir(dir).into_iter().flatten().flatten())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.ends_with(".json")
                    && !name.ends_with(".overrides.json")
                    && !matches!(name, "weights.json" | "quantizations.json" | "arch.json")
            })
            .collect();
        paths.sort();
        paths
    };
    let mut configs: Vec<(std::path::PathBuf, serde_json::Value)> = model_configs(&configs_dir)
        .into_iter()
        .map(|p| {
            let j = read(&p);
            (p, j)
        })
        .collect();
    if configs.is_empty() {
        for target in &facts.checkpoint_of {
            for p in model_configs(&arch_root.join("configs").join(target)) {
                let json = read(&p);
                if hf_config::carries_head(&facts.params, &facts.config_aliases, &json) {
                    configs.push((p, json));
                }
            }
        }
    }
    read_files.extend(configs.iter().map(|(p, _)| p.clone()));
    // The config loader reads a gated rule's flag on the decoder path only (a vision tower's
    // bounds are its own `vision_*`): a vision carrier resolves none, and naming one is refused.
    let mut gated = std::collections::BTreeMap::new();
    for flag in (dim::DERIVED_DIMS.iter())
        .filter_map(|d| d.rule.gate_flag())
        .filter(|_| !vision)
    {
        let mut sides: std::collections::BTreeMap<u64, Vec<String>> = Default::default();
        for (p, json) in &configs {
            let v = hf_config::flag_value(json, &facts.config_aliases, &facts.bound_defaults, flag);
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            sides.entry(v).or_default().push(name);
        }
        let resolved = match sides.len() {
            0 | 1 => Ok(sides.keys().next().copied().unwrap_or(0)),
            _ => Err(format!(
                "the configs disagree on `{flag}`: {}",
                (sides.iter())
                    .map(|(v, names)| format!("{v} in {}", names.join(", ")))
                    .collect::<Vec<_>>()
                    .join("; ")
            )),
        };
        gated.insert(flag.to_string(), resolved);
    }
    let exprs = (facts.params.iter())
        .filter_map(|f| match &f.source {
            hf_config::ParamSource::Expr(e) => Some((f.name.clone(), e)),
            _ => None,
        })
        .map(|(name, e)| {
            let parsed = hf_config::ParamExpr::parse(e).unwrap_or_else(|err| {
                panic!(
                    "carrier facts: {}: param `{name}`: {err}",
                    arch_json.display()
                )
            });
            (name, parsed)
        })
        .collect();
    // The bundles every config derives, as the config loader reads each (normalized under the
    // arch's aliases): one typed carrier holds them, so the configs must agree.
    let mut derived: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let mut agreed = None;
    for (p, json) in &configs {
        let json = hf_config::normalize_hf_config(json, &facts.config_aliases);
        let bounds = hf_config::extract_bounds(&json);
        let activation = bundles::ExpertActivation::of(&json);
        let config = bundles::BundleConfig {
            bounds: &bounds,
            activation: &activation,
        };
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let found = bundles::derive(program, &config);
        derived.entry(format!("{found:?}")).or_default().push(name);
        agreed.get_or_insert(found);
    }
    let bundles = match (derived.len(), agreed) {
        (0, _) | (_, None) => Ok(Default::default()),
        (1, Some(found)) => found,
        _ => Err(format!(
            "its configs derive different weight bundles: {}",
            (derived.iter())
                .map(|(b, names)| format!("{} derive {b}", names.join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        )),
    };
    typed_emit::CarrierFacts {
        gated,
        exprs,
        bundles,
    }
}
