// SPDX-License-Identifier: Apache-2.0
//! LoRA (Low-Rank Adaptation) adapter config parsing.
//!
//! Supports parsing PEFT-format adapter_config.json.
//! Actual weight loading is backend-specific (today: `scratchy-serving-cuda`
//! only; the metal path through `MetalWorker` doesn't yet wire LoRA).

use std::path::Path;

use serde::Deserialize;

use crate::error::{ModelError, ModelResult};

// ---------------------------------------------------------------------------
// LoraAdapterConfig — parsed adapter_config.json
// ---------------------------------------------------------------------------

/// Parsed PEFT adapter_config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct LoraAdapterConfig {
    /// LoRA rank (dimension of the low-rank matrices).
    pub r: usize,
    /// Scaling factor: effective scaling = lora_alpha / r.
    pub lora_alpha: f64,
    /// Which modules to apply LoRA to (e.g. ["q_proj", "k_proj", "v_proj", "o_proj"]).
    pub target_modules: Vec<String>,
    /// Whether to use rsLoRA scaling (alpha / sqrt(r) instead of alpha / r).
    #[serde(default)]
    pub use_rslora: bool,
}

impl LoraAdapterConfig {
    /// Parse from an adapter_config.json file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_json::from_str(&data)
            .map_err(|e| ModelError::Other(format!("failed to parse adapter_config.json: {e}")))?;
        Ok(config)
    }

    /// Compute the LoRA scaling factor.
    pub fn scaling(&self) -> f64 {
        if self.use_rslora {
            self.lora_alpha / (self.r as f64).sqrt()
        } else {
            self.lora_alpha / self.r as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_config_parse() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("adapter_config.json");
        std::fs::write(
            &config_path,
            r#"{
                "r": 16,
                "lora_alpha": 32.0,
                "target_modules": ["q_proj", "v_proj"],
                "use_rslora": false
            }"#,
        )
        .unwrap();

        let config = LoraAdapterConfig::from_file(&config_path).unwrap();
        assert_eq!(config.r, 16);
        assert!((config.lora_alpha - 32.0).abs() < 1e-6);
        assert_eq!(config.target_modules, vec!["q_proj", "v_proj"]);
        assert!(!config.use_rslora);
        assert!((config.scaling() - 2.0).abs() < 1e-6); // 32 / 16 = 2
    }

    #[test]
    fn test_adapter_config_rslora() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("adapter_config.json");
        std::fs::write(
            &config_path,
            r#"{
                "r": 16,
                "lora_alpha": 32.0,
                "target_modules": ["q_proj"],
                "use_rslora": true
            }"#,
        )
        .unwrap();

        let config = LoraAdapterConfig::from_file(&config_path).unwrap();
        assert!(config.use_rslora);
        // 32 / sqrt(16) = 32 / 4 = 8
        assert!((config.scaling() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_name_parsing() {
        // Verify that PEFT tensor names are parsed correctly.
        let name = "base_model.model.model.layers.5.self_attn.o_proj.lora_A.weight";
        let stripped = name.strip_suffix(".lora_A.weight").unwrap();
        let clean = stripped.strip_prefix("base_model.model.").unwrap();
        assert_eq!(clean, "model.layers.5.self_attn.o_proj");
    }
}
