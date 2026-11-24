//! The model, arch and stem names P7 knows, read from the workspace itself: the models crate's
//! Cargo features, the identities the checked-in model configs declare, and the DSL arch files.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The models crate's manifest, relative to the workspace root.
pub const MODELS_MANIFEST: &str = "crates/models/arch/Cargo.toml";
/// The checked-in model configs, relative to the workspace root.
pub const CONFIGS_DIR: &str = "crates/models/arch/configs";
/// The DSL arch files, relative to the workspace root.
pub const DSL_DIR: &str = "crates/models/arch/dsl";

/// The scratchy workspace root above `dir`: the nearest ancestor holding the models crate.
pub fn workspace_root(dir: &Path) -> Option<PathBuf> {
    dir.ancestors()
        .find(|d| d.join(MODELS_MANIFEST).is_file())
        .map(Path::to_path_buf)
}

/// The names the models crate's features declare: every `arch-<name>` gate (the gate and the
/// bare `<name>`), and every model stem, a feature whose one dependency is an arch gate.
pub fn from_features(manifest: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Ok(table) = manifest.parse::<toml::Table>() else {
        return names;
    };
    let Some(features) = table.get("features").and_then(toml::Value::as_table) else {
        return names;
    };
    for (key, deps) in features {
        let deps: Vec<&str> = deps
            .as_array()
            .map(|d| d.iter().filter_map(toml::Value::as_str).collect())
            .unwrap_or_default();
        if let Some(arch) = key.strip_prefix("arch-") {
            names.insert(key.clone());
            names.insert(arch.to_owned());
        } else if let [only] = deps.as_slice()
            && only.starts_with("arch-")
        {
            names.insert(key.clone());
        }
    }
    names
}

/// The identities one model config declares, at any depth (a vision tower's own config
/// included): each `architectures` entry and each `model_type`.
pub fn from_config(json: &str) -> BTreeSet<String> {
    fn walk(value: &serde_json::Value, names: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, v) in map {
                    match (key.as_str(), v) {
                        ("architectures", serde_json::Value::Array(arches)) => names
                            .extend(arches.iter().filter_map(|a| a.as_str()).map(str::to_owned)),
                        ("model_type", serde_json::Value::String(t)) => {
                            names.insert(t.clone());
                        }
                        _ => walk(v, names),
                    }
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| walk(v, names)),
            _ => {}
        }
    }
    let mut names = BTreeSet::new();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
        walk(&value, &mut names);
    }
    names
}

/// The family an arch name belongs to: its leading run of letters (`gemma` for `gemma4-moe`),
/// when that run is a word of its own, at least three letters long.
pub fn family(arch: &str) -> Option<&str> {
    let end = arch
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(arch.len());
    (end >= 3).then(|| &arch[..end])
}

/// The files the names come from, under `root`: the models manifest, every config JSON, and
/// every DSL arch file.
pub fn sources(root: &Path) -> Vec<PathBuf> {
    let mut files = vec![root.join(MODELS_MANIFEST)];
    let mut stack = vec![root.join(CONFIGS_DIR)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for path in entries.flatten().map(|e| e.path()) {
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "json") {
                files.push(path);
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(root.join(DSL_DIR)) {
        files.extend(
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "py")),
        );
    }
    files.sort();
    files
}

/// Every name the workspace at `root` declares: the features' names, the configs' identities,
/// the DSL arch file stems, and the family of each arch.
pub fn load(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut arches = BTreeSet::new();
    for path in sources(root) {
        let text = || std::fs::read_to_string(&path).unwrap_or_default();
        match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => {
                let found = from_features(&text());
                arches.extend(
                    found
                        .iter()
                        .filter_map(|n| n.strip_prefix("arch-"))
                        .map(str::to_owned),
                );
                names.extend(found);
            }
            Some("json") => names.extend(from_config(&text())),
            Some("py") => {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    arches.insert(stem.to_owned());
                }
            }
            _ => {}
        }
    }
    for arch in &arches {
        names.extend(family(arch).map(str::to_owned));
    }
    names.extend(arches);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_give_arch_gates_their_names_and_stems() {
        let manifest = r#"
[features]
default = []
arch-qwen3 = []
arch-qwen2-vl = ["arch-qwen2", "vision"]
vision = []
all-arches = ["arch-qwen3", "arch-qwen2-vl"]
"qwen3-0.6b" = ["arch-qwen3"]
qwen3 = ["qwen3-0.6b"]
all = ["qwen3-0.6b"]
mtp = ["mtp-heads"]
hf-completions = ["dep:ureq"]
"#;
        let names: Vec<_> = from_features(manifest).into_iter().collect();
        assert_eq!(
            names,
            [
                "arch-qwen2-vl",
                "arch-qwen3",
                "qwen2-vl",
                "qwen3",
                "qwen3-0.6b"
            ]
        );
    }

    #[test]
    fn configs_give_their_architectures_and_model_types_at_any_depth() {
        let json = r#"{
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": { "model_type": "gemma3_text", "hidden_size": 8 },
            "vision_config": { "model_type": "siglip_vision_model" },
            "rope_scaling": { "rope_type": "llama3" }
        }"#;
        let names: Vec<_> = from_config(json).into_iter().collect();
        assert_eq!(
            names,
            [
                "Gemma3ForConditionalGeneration",
                "gemma3",
                "gemma3_text",
                "siglip_vision_model"
            ]
        );
        assert!(from_config("not json").is_empty());
        assert!(from_config(r#"{"scale_dtype": "bf16"}"#).is_empty());
    }

    #[test]
    fn a_family_is_the_leading_word_of_an_arch_name() {
        assert_eq!(family("gemma4-moe"), Some("gemma"));
        assert_eq!(family("qwen3-5"), Some("qwen"));
        assert_eq!(family("llama"), Some("llama"));
        assert_eq!(family("gpt-oss"), Some("gpt"));
        assert_eq!(family("x1"), None);
    }

    #[test]
    fn the_workspace_names_cover_features_configs_dsl_and_families() {
        let dir =
            std::env::temp_dir().join(format!("scratchy-purity-names-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let write = |rel: &str, text: &str| {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        write(
            MODELS_MANIFEST,
            "[features]\narch-llama = []\n\"smollm2-135m\" = [\"arch-llama\"]\n",
        );
        write(
            &format!("{CONFIGS_DIR}/llama/smollm2-135m.json"),
            r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}"#,
        );
        write(&format!("{CONFIGS_DIR}/llama/weights.json"), "{}");
        write(&format!("{DSL_DIR}/gemma4-moe.py"), "");
        write(&format!("{DSL_DIR}/CLAUDE.md"), "");
        assert_eq!(workspace_root(&dir.join("crates/a/src")), Some(dir.clone()));
        let names: Vec<_> = load(&dir).into_iter().collect();
        assert_eq!(
            names,
            [
                "LlamaForCausalLM",
                "arch-llama",
                "gemma",
                "gemma4-moe",
                "llama",
                "smollm2-135m"
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
