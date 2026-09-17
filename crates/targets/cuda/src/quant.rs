// SPDX-License-Identifier: Apache-2.0
//! Quantization config detection and Marlin weight transformation utilities.
//!
//! Supports AWQ and GPTQ models. Both are repacked to Marlin format at load
//! time using GPU repack kernels, then use the fused Marlin INT4×FP16→FP16
//! GEMM for all quantized linear layers.

use std::path::Path;

use anyhow::{Result, bail};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// QuantConfig
// ---------------------------------------------------------------------------

/// Quantization method detected from model config.
#[derive(Debug, Clone)]
pub enum QuantConfig {
    /// Dense (unquantized) model.
    None,
    /// AWQ quantization → Marlin format.
    Awq(AwqConfig),
    /// GPTQ quantization → Marlin format.
    Gptq(GptqConfig),
    /// BitsAndBytes 4-bit (NF4/FP4) quantization.
    Bnb4bit(Bnb4bitConfig),
    /// FP8 (E4M3) weight quantization — per-tensor or per-block scales.
    Fp8(Fp8Config),
}

/// FP8 weight quantization config.
///
/// Matches Python vLLM's `Fp8Config` / `Fp8LinearMethod`. Supports both
/// serialized FP8 checkpoints (weights stored as float8_e4m3fn with
/// pre-computed scales) and online quantization (BF16 weights quantized
/// to FP8 at load time).
#[derive(Debug, Clone)]
pub struct Fp8Config {
    /// Dynamic (per-token at runtime) vs Static (pre-calibrated input_scale).
    pub activation_scheme: Fp8ActivationScheme,
    /// Per-block quantization block size, e.g. `[128, 128]` for DeepSeek-V3.
    /// `None` means per-tensor scales.
    pub weight_block_size: Option<[usize; 2]>,
    /// If true, checkpoint already has FP8 weights + weight_scale tensors.
    /// If false, weights are BF16/F16 and will be quantized online at load time.
    pub is_checkpoint_fp8_serialized: bool,
}

/// FP8 activation quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8ActivationScheme {
    /// Per-token dynamic quantization at runtime (default for SM89+).
    Dynamic,
    /// Static quantization using pre-calibrated `input_scale`.
    Static,
}

/// BitsAndBytes 4-bit quantization config.
#[derive(Debug, Clone)]
pub struct Bnb4bitConfig {
    pub blocksize: usize,
    pub quant_type: BnbQuantType,
}

/// BNB 4-bit quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnbQuantType {
    NF4,
    FP4,
}

/// NF4 lookup table — 16 quantiles of the standard normal distribution, rescaled to [-1, 1].
/// From bitsandbytes source; these values are fixed forever.
#[allow(clippy::excessive_precision)]
pub const NF4_CODE: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// FP4 lookup table — E2M1 values used by bitsandbytes FP4 quantization.
pub const FP4_CODE: [f32; 16] = [
    0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0,
];

#[derive(Debug, Clone)]
pub struct AwqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub zero_point: bool, // always true for AWQ
}

#[derive(Debug, Clone)]
pub struct GptqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub desc_act: bool,
    pub sym: bool,
}

impl QuantConfig {
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn group_size(&self) -> usize {
        match self {
            Self::None | Self::Bnb4bit(_) | Self::Fp8(_) => 0,
            Self::Awq(c) => c.group_size,
            Self::Gptq(c) => c.group_size,
        }
    }

    pub fn bits(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Awq(c) => c.bits,
            Self::Gptq(c) => c.bits,
            Self::Bnb4bit(_) => 4,
            Self::Fp8(_) => 8,
        }
    }

    /// Marlin b_type_id: 0 = GPTQ (uint4b8), 1 = AWQ (uint4).
    pub fn b_type_id(&self) -> i32 {
        match self {
            Self::Awq(_) => 1,
            Self::Gptq(_) => 0,
            Self::None | Self::Bnb4bit(_) | Self::Fp8(_) => -1,
        }
    }

    /// Whether this quant format has zero points.
    pub fn has_zp(&self) -> bool {
        match self {
            Self::Awq(_) => true,
            Self::Gptq(c) => !c.sym,
            Self::None | Self::Bnb4bit(_) | Self::Fp8(_) => false,
        }
    }

    /// Whether this quant format uses activation ordering (desc_act).
    pub fn has_act_order(&self) -> bool {
        match self {
            Self::Gptq(c) => c.desc_act,
            _ => false,
        }
    }

    /// Whether this is a BNB 4-bit quantized model.
    pub fn is_bnb4bit(&self) -> bool {
        matches!(self, Self::Bnb4bit(_))
    }

    /// Whether this is an FP8 quantized model.
    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8(_))
    }

    /// Whether this is an AWQ quantized model.
    pub fn is_awq(&self) -> bool {
        matches!(self, Self::Awq(_))
    }

    /// Whether this is a GPTQ quantized model.
    pub fn is_gptq(&self) -> bool {
        matches!(self, Self::Gptq(_))
    }
}

// ---------------------------------------------------------------------------
// Detection from model directory
// ---------------------------------------------------------------------------

/// Raw JSON structure for `quantize_config.json` (shared by AWQ and GPTQ).
#[derive(Debug, Deserialize)]
struct RawQuantConfig {
    #[serde(alias = "w_bit")]
    bits: usize,
    #[serde(alias = "q_group_size")]
    group_size: usize,
    #[serde(default)]
    quant_method: Option<String>,
    #[serde(default)]
    desc_act: bool,
    #[serde(default = "default_true")]
    sym: bool,
}

fn default_true() -> bool {
    true
}

/// Detect quantization config from a model directory.
///
/// Checks `quantize_config.json` first, then falls back to
/// `config.json` → `quantization_config`.
pub fn detect_quant_config(model_dir: impl AsRef<Path>) -> Result<QuantConfig> {
    let dir = model_dir.as_ref();

    // Try quantize_config.json first.
    let qc_path = dir.join("quantize_config.json");
    if qc_path.exists() {
        let data = std::fs::read_to_string(&qc_path)?;
        let raw: RawQuantConfig = serde_json::from_str(&data)?;
        return parse_raw_config(raw);
    }

    // Try quant_config.json (used by some AWQ models).
    let qc_path2 = dir.join("quant_config.json");
    if qc_path2.exists() {
        let data = std::fs::read_to_string(&qc_path2)?;
        let raw: RawQuantConfig = serde_json::from_str(&data)?;
        return parse_raw_config(raw);
    }

    // Fallback: config.json → quantization_config.
    let config_path = dir.join("config.json");
    if config_path.exists() {
        let data = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&data)?;
        if let Some(qc) = config.get("quantization_config") {
            let method = qc
                .get("quant_method")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Check for FP8.
            if method == "fp8" {
                return parse_fp8_config(qc);
            }

            // Check for compressed-tensors (llmcompressor/vllm-quantizer format).
            // When weights are 8-bit float with symmetric quantization, this is FP8.
            // Matches Python vLLM's CompressedTensorsConfig → FP8 dispatch.
            if method == "compressed-tensors" {
                return parse_compressed_tensors_config(qc);
            }

            // No quant_method or explicitly null → not quantized.
            if method.is_empty() {
                return Ok(QuantConfig::None);
            }

            // Check for BitsAndBytes.
            if method == "bitsandbytes" {
                let load_4bit = qc
                    .get("load_in_4bit")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if load_4bit {
                    let blocksize = 64; // BNB default
                    let qt = qc
                        .get("bnb_4bit_quant_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("nf4");
                    let quant_type = match qt {
                        "fp4" => BnbQuantType::FP4,
                        _ => BnbQuantType::NF4,
                    };
                    return Ok(QuantConfig::Bnb4bit(Bnb4bitConfig {
                        blocksize,
                        quant_type,
                    }));
                }
                bail!("bitsandbytes 8-bit (load_in_8bit) not supported");
            }
            let raw: RawQuantConfig = serde_json::from_value(qc.clone())?;
            return parse_raw_config(raw);
        }
    }

    Ok(QuantConfig::None)
}

/// Parse FP8 quantization config from `config.json → quantization_config`.
///
/// Matches Python vLLM's `Fp8Config.from_config()`:
/// - `activation_scheme`: "dynamic" (default) or "static"
/// - `weight_block_size`: null or [block_n, block_k] (e.g. [128, 128] for DeepSeek-V3)
/// - `is_checkpoint_fp8_serialized`: bool (default true for fp8 quant_method)
fn parse_fp8_config(qc: &serde_json::Value) -> Result<QuantConfig> {
    let activation_scheme = match qc
        .get("activation_scheme")
        .and_then(|v| v.as_str())
        .unwrap_or("dynamic")
    {
        "static" => Fp8ActivationScheme::Static,
        _ => Fp8ActivationScheme::Dynamic,
    };

    let weight_block_size = qc
        .get("weight_block_size")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() == 2 {
                let a = arr[0].as_u64()? as usize;
                let b = arr[1].as_u64()? as usize;
                Some([a, b])
            } else {
                None
            }
        });

    // Default: if quant_method is "fp8", checkpoint usually has FP8 weights.
    // Some checkpoints set this explicitly; others rely on weight dtype detection.
    let is_checkpoint_fp8_serialized = qc
        .get("is_checkpoint_fp8_serialized")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    tracing::info!(
        "Detected FP8 quantization: activation_scheme={activation_scheme:?}, \
         weight_block_size={weight_block_size:?}, serialized={is_checkpoint_fp8_serialized}"
    );

    Ok(QuantConfig::Fp8(Fp8Config {
        activation_scheme,
        weight_block_size,
        is_checkpoint_fp8_serialized,
    }))
}

/// Parse `compressed-tensors` quantization config.
///
/// Matches Python vLLM's `CompressedTensorsConfig` → FP8 dispatch.
/// The config has `config_groups` with weight/input_activations specs.
/// When weights are 8-bit float, this maps to our FP8 path.
fn parse_compressed_tensors_config(qc: &serde_json::Value) -> Result<QuantConfig> {
    // Find the first config group to determine weight/activation quantization.
    let config_groups = qc
        .get("config_groups")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("compressed-tensors: missing config_groups"))?;

    let group = config_groups
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("compressed-tensors: empty config_groups"))?;

    let weights = group
        .get("weights")
        .ok_or_else(|| anyhow::anyhow!("compressed-tensors: missing weights in config group"))?;

    let weight_type = weights.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let weight_bits = weights
        .get("num_bits")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // INT4 quantization (WNA16 — weight-only INT4, FP16 activations).
    // compressed-tensors pack-quantized format is equivalent to GPTQ packing.
    // Symmetric quantization → no zero points → b_type_id=0 (uint4b8).
    if weight_type == "int" && weight_bits == 4 {
        let group_size = weights
            .get("group_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let symmetric = weights
            .get("symmetric")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let desc_act = false; // compressed-tensors doesn't use activation ordering

        tracing::info!(
            "Detected compressed-tensors INT4: group_size={group_size}, symmetric={symmetric}"
        );

        return Ok(QuantConfig::Gptq(GptqConfig {
            bits: 4,
            group_size,
            desc_act,
            sym: symmetric,
        }));
    }

    if weight_type != "float" || weight_bits != 8 {
        bail!("compressed-tensors: unsupported weight type={weight_type}, bits={weight_bits}");
    }

    // Determine activation scheme from input_activations.
    let activation_scheme = if let Some(input_act) = group.get("input_activations") {
        let dynamic = input_act
            .get("dynamic")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if dynamic {
            Fp8ActivationScheme::Dynamic
        } else {
            Fp8ActivationScheme::Static
        }
    } else {
        // No input activations quantized — use dynamic (weight-only FP8).
        Fp8ActivationScheme::Dynamic
    };

    // Check for block quantization via weight.block_structure.
    let weight_block_size = weights
        .get("block_structure")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() == 2 {
                let a = arr[0].as_u64()? as usize;
                let b = arr[1].as_u64()? as usize;
                Some([a, b])
            } else {
                None
            }
        });

    tracing::info!(
        "Detected compressed-tensors FP8: activation_scheme={activation_scheme:?}, \
         weight_block_size={weight_block_size:?}"
    );

    Ok(QuantConfig::Fp8(Fp8Config {
        activation_scheme,
        weight_block_size,
        is_checkpoint_fp8_serialized: true,
    }))
}

fn parse_raw_config(raw: RawQuantConfig) -> Result<QuantConfig> {
    if raw.bits != 4 {
        bail!("only 4-bit quantization supported, got {} bits", raw.bits);
    }
    if !matches!(raw.group_size, 32 | 64 | 128) && raw.group_size != usize::MAX {
        // group_size -1 in JSON is parsed as very large; allow common values
        if raw.group_size != 0 {
            tracing::warn!("unusual group_size: {}", raw.group_size);
        }
    }

    // AutoGPTQ's native `quantize_config.json` often omits the
    // `quant_method` field (it's implicitly GPTQ — that's the file's
    // format). AWQ's `quant_config.json` always sets
    // `quant_method: "awq"`. So an absent/empty method defaults to
    // GPTQ for parity with Python vLLM's detection.
    let method = raw.quant_method.as_deref().unwrap_or("gptq");
    match method {
        "awq" => Ok(QuantConfig::Awq(AwqConfig {
            bits: raw.bits,
            group_size: raw.group_size,
            zero_point: true,
        })),
        "gptq" => Ok(QuantConfig::Gptq(GptqConfig {
            bits: raw.bits,
            group_size: raw.group_size,
            desc_act: raw.desc_act,
            sym: raw.sym,
        })),
        _ => bail!("unknown quant_method: {:?}", raw.quant_method),
    }
}

// ---------------------------------------------------------------------------
// Marlin scale/zero-point permutation (CPU, at load time)
// ---------------------------------------------------------------------------
//
// Moved to `crate::layers_quant` so scratchy-forward-compiler-emitted
// code can reach them too. Re-exported here under their historical
// paths (`crate::quant::marlin_permute_scales`, etc.) so hand-written
// call sites remain unchanged.

pub use crate::layers_quant::{
    awq_to_marlin_zero_points, marlin_permute_scales, pack_cols_4bit, scale_perm,
    scale_perm_single, unpack_cols_4bit,
};

// ---------------------------------------------------------------------------
// Marlin tile constants (must match marlin.cuh)
// ---------------------------------------------------------------------------

/// Marlin tile size along K dimension.
pub const MARLIN_TILE_K: usize = 16;
/// Marlin tile size along N dimension (= tile_k * 4).
pub const MARLIN_TILE_N: usize = 64;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_awq_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 128, "quant_method": "awq", "zero_point": true}"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(matches!(config, QuantConfig::Awq(_)));
        assert_eq!(config.bits(), 4);
        assert_eq!(config.group_size(), 128);
        assert!(config.has_zp());
        assert_eq!(config.b_type_id(), 1);
    }

    #[test]
    fn test_detect_gptq_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 128, "quant_method": "gptq", "desc_act": false, "sym": true}"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(matches!(config, QuantConfig::Gptq(_)));
        assert!(!config.has_zp());
        assert!(!config.has_act_order());
        assert_eq!(config.b_type_id(), 0);
    }

    #[test]
    fn test_detect_from_config_json_fallback() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"quantization_config": {"bits": 4, "group_size": 128, "quant_method": "awq"}}"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(matches!(config, QuantConfig::Awq(_)));
    }

    #[test]
    fn test_detect_no_quant() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"hidden_size": 4096}"#).unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(matches!(config, QuantConfig::None));
    }

    #[test]
    fn test_detect_quant_method_null() {
        // Some HF models have quantization_config with quant_method: null
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"quantization_config": {"quant_method": null}}"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(matches!(config, QuantConfig::None));
    }

    #[test]
    fn test_scale_perm_values() {
        let p = scale_perm();
        assert_eq!(p.len(), 64);
        // First row: 0, 8, 16, 24, 32, 40, 48, 56
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 8);
        assert_eq!(p[7], 56);
        // Second row starts at i=1: 1, 9, 17, ...
        assert_eq!(p[8], 1);
        assert_eq!(p[9], 9);
    }

    #[test]
    fn test_scale_perm_single_values() {
        let p = scale_perm_single();
        assert_eq!(p.len(), 32);
        // i=0: 0, 1, 8, 9, 16, 17, 24, 25
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 1);
        assert_eq!(p[2], 8);
        assert_eq!(p[3], 9);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let rows = 2;
        let cols = 16;
        let original: Vec<u8> = (0..32).map(|i| (i % 16) as u8).collect();
        let packed = pack_cols_4bit(&original, rows, cols);
        let unpacked = unpack_cols_4bit(&packed, rows, cols);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn test_marlin_permute_scales_group() {
        // 2 groups, size_n=64, group_size=128
        let mut scales: Vec<u16> = (0..128).collect();
        let original = scales.clone();
        marlin_permute_scales(&mut scales, 256, 64, 128);
        // Should be permuted (not identity)
        assert_ne!(scales, original);
        // Same values, different order
        let mut sorted_orig = original.clone();
        sorted_orig.sort();
        let mut sorted_perm = scales.clone();
        sorted_perm.sort();
        assert_eq!(sorted_orig, sorted_perm);
    }

    #[test]
    fn test_awq_to_marlin_zero_points_roundtrip() {
        // 1 group, 16 output features
        let num_groups = 1;
        let size_n = 16;
        // Create packed qzeros: [1, 2] u32 (16 values / 8 per u32 = 2)
        let packed = vec![0x76543210u32, 0xFEDCBA98u32];
        let result = awq_to_marlin_zero_points(&packed, num_groups, size_n);
        // Result should be same number of u32 elements
        assert_eq!(result.len(), 2);
        // Unpack both and check same set of values (permuted)
        let orig_vals = unpack_cols_4bit(&packed, num_groups, size_n);
        let result_vals = unpack_cols_4bit(&result, num_groups, size_n);
        let mut orig_sorted = orig_vals.clone();
        orig_sorted.sort();
        let mut result_sorted = result_vals.clone();
        result_sorted.sort();
        assert_eq!(orig_sorted, result_sorted);
    }

    #[test]
    fn test_pack_cols_known_values() {
        // Pack [0, 1, 2, ..., 7] into a single u32 for one row
        let rows = 1;
        let cols = 8;
        let unpacked: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = pack_cols_4bit(&unpacked, rows, cols);
        assert_eq!(packed.len(), 1);
        // Verify roundtrip
        let roundtrip = unpack_cols_4bit(&packed, rows, cols);
        assert_eq!(roundtrip, unpacked);
    }

    #[test]
    fn test_marlin_permute_scales_single() {
        // Per-channel: group_size >= size_k uses scale_perm_single
        let mut scales: Vec<u16> = (0..32).collect();
        let original = scales.clone();
        marlin_permute_scales(&mut scales, 32, 32, 128); // group_size >= size_k
        assert_ne!(scales, original);
    }

    #[test]
    fn test_detect_gptq_desc_act() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 128, "quant_method": "gptq", "desc_act": true, "sym": true}"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(config.has_act_order());
    }

    #[test]
    fn test_detect_gptq_asymmetric() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 128, "quant_method": "gptq", "sym": false}"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(config.has_zp()); // asymmetric = has zero points
    }

    #[test]
    fn test_reject_8bit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 8, "group_size": 128, "quant_method": "gptq"}"#,
        )
        .unwrap();
        let result = detect_quant_config(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_quant_config_methods() {
        let awq = QuantConfig::Awq(AwqConfig {
            bits: 4,
            group_size: 128,
            zero_point: true,
        });
        assert!(awq.is_quantized());
        assert_eq!(awq.bits(), 4);
        assert_eq!(awq.group_size(), 128);
        assert!(awq.has_zp());
        assert!(!awq.has_act_order());
        assert_eq!(awq.b_type_id(), 1);

        let none = QuantConfig::None;
        assert!(!none.is_quantized());
        assert_eq!(none.bits(), 0);
    }

    #[test]
    fn test_fp8_config_methods() {
        let fp8 = QuantConfig::Fp8(Fp8Config {
            activation_scheme: Fp8ActivationScheme::Dynamic,
            weight_block_size: None,
            is_checkpoint_fp8_serialized: true,
        });
        assert!(fp8.is_quantized());
        assert!(fp8.is_fp8());
        assert!(!fp8.is_bnb4bit());
        assert_eq!(fp8.bits(), 8);
        assert_eq!(fp8.group_size(), 0);
        assert!(!fp8.has_zp());
        assert!(!fp8.has_act_order());
        assert_eq!(fp8.b_type_id(), -1);
    }

    #[test]
    fn test_fp8_config_block_quant() {
        let fp8_block = QuantConfig::Fp8(Fp8Config {
            activation_scheme: Fp8ActivationScheme::Dynamic,
            weight_block_size: Some([128, 128]),
            is_checkpoint_fp8_serialized: true,
        });
        assert!(fp8_block.is_fp8());
        assert_eq!(fp8_block.bits(), 8);

        let fp8_static = QuantConfig::Fp8(Fp8Config {
            activation_scheme: Fp8ActivationScheme::Static,
            weight_block_size: None,
            is_checkpoint_fp8_serialized: false,
        });
        assert!(fp8_static.is_fp8());
    }

    // CUDA kernel tests — require GPU

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use crate::alloc::CachingAllocator;
        use crate::driver;
        use crate::dtype::DType;
        use crate::tensor::GpuTensor;

        fn init_cuda() -> cudarc::driver::sys::CUstream {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
                driver::stream_create().expect("stream")
            }
        }

        /// Upload a u32 slice to GPU.
        unsafe fn upload_u32(data: &[u32], stream: cudarc::driver::sys::CUstream) -> GpuTensor {
            let nbytes = data.len() * 4;
            let ptr = driver::mem_alloc(nbytes).expect("alloc");
            driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, nbytes, stream)
                .expect("h2d");
            GpuTensor::new(ptr, &[data.len()], DType::U32)
        }

        /// Download u32 data from GPU.
        unsafe fn download_u32(
            ptr: *mut u8,
            count: usize,
            stream: cudarc::driver::sys::CUstream,
        ) -> Vec<u32> {
            let nbytes = count * 4;
            let host = driver::mem_alloc_host(nbytes).expect("host alloc");
            driver::memcpy_dtoh_async(host.as_ptr(), ptr, nbytes, stream).expect("d2h");
            driver::stream_synchronize(stream).expect("sync");
            let result = std::slice::from_raw_parts(host.as_ptr() as *const u32, count).to_vec();
            driver::mem_free_host(host).expect("free host");
            result
        }

        #[test]
        fn test_cuda_awq_repack_4bit() {
            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            // Marlin tiles: tile_k=16, tile_n=64. Dimensions must be multiples.
            let size_k = 64;
            let size_n = 64;
            let pack_factor = 8; // 32/4
            let num_packed = size_k * (size_n / pack_factor);

            // Create simple packed weights
            let host_data: Vec<u32> = (0..num_packed as u32).collect();

            unsafe {
                let gpu_input = upload_u32(&host_data, stream);

                let repacked =
                    crate::kernels::awq_repack(gpu_input, size_k, size_n, 0, &mut alloc, stream);

                driver::stream_synchronize(stream).expect("sync");

                // Verify output has same number of elements
                let out_count = size_k * size_n / 8;
                let out_data = download_u32(repacked.raw_ptr(), out_count, stream);
                assert_eq!(out_data.len(), out_count);

                // Output should differ from input (different tiling layout)
                assert_ne!(out_data, host_data, "repack should change the layout");

                driver::stream_destroy(stream).expect("destroy");
            }
        }

        #[test]
        fn test_cuda_gptq_repack_4bit() {
            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let size_k = 64;
            let size_n = 64;
            let pack_factor = 8;
            let num_packed = (size_k / pack_factor) * size_n;

            let host_data: Vec<u32> = (0..num_packed as u32).collect();

            unsafe {
                let gpu_input = upload_u32(&host_data, stream);

                let repacked = crate::kernels::gptq_repack(
                    gpu_input, None, size_k, size_n, 0, &mut alloc, stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                let out_count = size_k * size_n / 8;
                let out_data = download_u32(repacked.raw_ptr(), out_count, stream);
                assert_eq!(out_data.len(), out_count);
                assert_ne!(out_data, host_data, "repack should change the layout");

                driver::stream_destroy(stream).expect("destroy");
            }
        }

        #[test]
        fn test_cuda_marlin_gemm_f16_smoke() {
            // Smoke test: realistic dimensions matching common quantized models
            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let size_m = 1;
            let size_k = 256;
            let size_n = 256;
            let group_size = 128;
            let num_groups = size_k / group_size;

            unsafe {
                // Activation: [1, 16] f16 — all ones
                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_m * size_k];
                let a_nbytes = a_data.len() * 2;
                let a_ptr = driver::mem_alloc(a_nbytes).expect("alloc a");
                driver::memcpy_htod_async(a_ptr, a_data.as_ptr() as *const u8, a_nbytes, stream)
                    .expect("h2d a");
                let a = GpuTensor::new(a_ptr, &[size_m, size_k], DType::F16);

                // For a proper smoke test we need properly repacked weights.
                // Create zero weights (trivial case — output should be zero).
                let qw_count = size_k * size_n / 8;
                let qw_data = vec![0u32; qw_count];
                let qw_ptr = driver::mem_alloc(qw_count * 4).expect("alloc qw");
                driver::memcpy_htod_async(
                    qw_ptr,
                    qw_data.as_ptr() as *const u8,
                    qw_count * 4,
                    stream,
                )
                .expect("h2d qw");
                let qw = GpuTensor::new(qw_ptr, &[qw_count], DType::U32);

                // Scales: [1, 16] f16 — all ones
                let s_data: Vec<u16> =
                    vec![half::f16::from_f32(1.0).to_bits(); num_groups * size_n];
                let s_nbytes = s_data.len() * 2;
                let s_ptr = driver::mem_alloc(s_nbytes).expect("alloc s");
                driver::memcpy_htod_async(s_ptr, s_data.as_ptr() as *const u8, s_nbytes, stream)
                    .expect("h2d s");
                let scales = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                // Workspace
                let ws_count = 128; // num_sms, generous
                let ws_ptr = driver::mem_alloc(ws_count * 4).expect("alloc ws");
                driver::memset_d8(ws_ptr, 0, ws_count * 4, stream).expect("memset ws");
                let workspace = GpuTensor::new(ws_ptr, &[ws_count], DType::I32);

                let out = crate::kernels::marlin_gemm(
                    a, qw, scales, None, // no zeros
                    None, // no g_idx
                    None, // no perm
                    None, // no bias
                    workspace, size_m, size_n, size_k, num_groups, group_size,
                    false, // no act_order
                    false, // no zp
                    0,     // GPTQ type
                    0,     // device_id
                    &mut alloc, stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                // Read output
                let out_nbytes = size_m * size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host alloc");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let result =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_m * size_n);
                // Zero weights → output should be zero (or close)
                for &bits in result {
                    let val = half::f16::from_bits(bits).to_f32();
                    assert!(!val.is_nan(), "marlin_gemm output contains NaN");
                }

                driver::mem_free_host(host).expect("free host");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Test full GPTQ pipeline: create known weights → repack → permute scales → GEMM → verify.
        ///
        /// All INT4 values = 9 (kU4B8: 9-8 = +1), scales = 1.0, input = all-ones.
        /// Expected: output[i] = K * 1 * 1 = K for each output element.
        #[test]
        fn test_cuda_gptq_repack_gemm_correctness() {
            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let size_k: usize = 256;
            let size_n: usize = 256;
            let group_size: usize = 128;
            let num_groups = size_k / group_size;
            let size_m: usize = 1;

            unsafe {
                // 1. Create GPTQ-format qweight: [K/8, N] u32, all nibbles = 9
                //    kU4B8 dequant: val = (nibble - 8) * scale = (9-8)*1.0 = 1.0
                let pack_factor = 8;
                let gptq_rows = size_k / pack_factor;
                let gptq_count = gptq_rows * size_n;
                let gptq_data = vec![0x99999999u32; gptq_count];

                let gptq_ptr = driver::mem_alloc(gptq_count * 4).expect("alloc gptq");
                driver::memcpy_htod_async(
                    gptq_ptr,
                    gptq_data.as_ptr() as *const u8,
                    gptq_count * 4,
                    stream,
                )
                .expect("h2d gptq");
                let gptq_gpu = GpuTensor::new(gptq_ptr, &[gptq_rows, size_n], DType::U32);

                // 2. Repack GPTQ → Marlin tiled layout
                let repacked = crate::kernels::gptq_repack(
                    gptq_gpu, None, size_k, size_n, 0, &mut alloc, stream,
                );

                // 3. Create scales [num_groups, N] = all ones, then permute
                let mut scales_u16: Vec<u16> =
                    vec![half::f16::from_f32(1.0).to_bits(); num_groups * size_n];
                super::super::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

                let s_nbytes = scales_u16.len() * 2;
                let s_ptr = driver::mem_alloc(s_nbytes).expect("alloc scales");
                driver::memcpy_htod_async(
                    s_ptr,
                    scales_u16.as_ptr() as *const u8,
                    s_nbytes,
                    stream,
                )
                .expect("h2d scales");
                let scales = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                // 4. Activation: [1, K] f16, all ones
                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_m * size_k];
                let a_nbytes = a_data.len() * 2;
                let a_ptr = driver::mem_alloc(a_nbytes).expect("alloc a");
                driver::memcpy_htod_async(a_ptr, a_data.as_ptr() as *const u8, a_nbytes, stream)
                    .expect("h2d a");
                let a = GpuTensor::new(a_ptr, &[size_m, size_k], DType::F16);

                // 5. Workspace
                let ws_count = 128;
                let ws_ptr = driver::mem_alloc(ws_count * 4).expect("alloc ws");
                driver::memset_d8(ws_ptr, 0, ws_count * 4, stream).expect("memset ws");
                let workspace = GpuTensor::new(ws_ptr, &[ws_count], DType::I32);

                // 6. Run Marlin GEMM
                let out = crate::kernels::marlin_gemm(
                    a,
                    repacked.as_gpu_tensor(),
                    scales,
                    None, // no zeros
                    None, // no g_idx
                    None, // no perm
                    None, // no bias
                    workspace,
                    size_m,
                    size_n,
                    size_k,
                    num_groups,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                // 7. Read output and verify
                let out_nbytes = size_m * size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host alloc");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let result =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_m * size_n);
                let expected = size_k as f32; // each output = sum of K * 1.0 * 1.0 = K

                let mut max_err: f32 = 0.0;
                for (i, &bits) in result.iter().enumerate() {
                    let val = half::f16::from_bits(bits).to_f32();
                    let err = (val - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    if i < 8 {
                        eprintln!("out[{i}] = {val} (expected {expected}, err={err})");
                    }
                }
                eprintln!("max error across {size_n} outputs: {max_err}");

                // Allow some FP16 rounding: K=256 with f16 accumulation
                assert!(
                    max_err < 2.0,
                    "max_err={max_err} too large, expected ~{expected} for all outputs"
                );

                driver::mem_free_host(host).expect("free host");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Test GPTQ pipeline with non-uniform scales to verify scale permutation.
        ///
        /// All INT4 values = 9 (kU4B8: +1), but scales vary per group.
        /// Group 0 scales = 2.0, Group 1 scales = 3.0.
        /// With input = all-ones [1, K], K=256, group_size=128:
        ///   output[j] = 128 * 2.0 + 128 * 3.0 = 256 + 384 = 640
        #[test]
        fn test_cuda_gptq_nonuniform_scales() {
            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let size_k: usize = 256;
            let size_n: usize = 256;
            let group_size: usize = 128;
            let num_groups = size_k / group_size; // 2
            let size_m: usize = 1;

            unsafe {
                // GPTQ qweight: all nibbles = 9
                let pack_factor = 8;
                let gptq_rows = size_k / pack_factor;
                let gptq_count = gptq_rows * size_n;
                let gptq_data = vec![0x99999999u32; gptq_count];
                let gptq_ptr = driver::mem_alloc(gptq_count * 4).expect("alloc");
                driver::memcpy_htod_async(
                    gptq_ptr,
                    gptq_data.as_ptr() as *const u8,
                    gptq_count * 4,
                    stream,
                )
                .expect("h2d");
                let gptq_gpu = GpuTensor::new(gptq_ptr, &[gptq_rows, size_n], DType::U32);

                let repacked = crate::kernels::gptq_repack(
                    gptq_gpu, None, size_k, size_n, 0, &mut alloc, stream,
                );

                // Scales: [2, 256] — group 0 = 2.0, group 1 = 3.0
                let two = half::f16::from_f32(2.0).to_bits();
                let three = half::f16::from_f32(3.0).to_bits();
                let mut scales_u16: Vec<u16> = Vec::with_capacity(num_groups * size_n);
                for g in 0..num_groups {
                    let val = if g == 0 { two } else { three };
                    scales_u16.extend(std::iter::repeat_n(val, size_n));
                }
                super::super::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

                let s_nbytes = scales_u16.len() * 2;
                let s_ptr = driver::mem_alloc(s_nbytes).expect("alloc");
                driver::memcpy_htod_async(
                    s_ptr,
                    scales_u16.as_ptr() as *const u8,
                    s_nbytes,
                    stream,
                )
                .expect("h2d");
                let scales = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                // Input: all ones
                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_m * size_k];
                let a_ptr = driver::mem_alloc(a_data.len() * 2).expect("alloc");
                driver::memcpy_htod_async(
                    a_ptr,
                    a_data.as_ptr() as *const u8,
                    a_data.len() * 2,
                    stream,
                )
                .expect("h2d");
                let a = GpuTensor::new(a_ptr, &[size_m, size_k], DType::F16);

                let ws_ptr = driver::mem_alloc(128 * 4).expect("alloc");
                driver::memset_d8(ws_ptr, 0, 128 * 4, stream).expect("memset");
                let workspace = GpuTensor::new(ws_ptr, &[128], DType::I32);

                let out = crate::kernels::marlin_gemm(
                    a,
                    repacked.as_gpu_tensor(),
                    scales,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    size_m,
                    size_n,
                    size_k,
                    num_groups,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                let out_nbytes = size_m * size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let result =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_m * size_n);
                let expected = 128.0 * 2.0 + 128.0 * 3.0; // = 640.0

                let mut max_err: f32 = 0.0;
                for (i, &bits) in result.iter().enumerate() {
                    let val = half::f16::from_bits(bits).to_f32();
                    let err = (val - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    if i < 8 {
                        eprintln!("out[{i}] = {val} (expected {expected}, err={err})");
                    }
                }
                eprintln!("max error across {size_n} outputs: {max_err}");
                assert!(max_err < 4.0, "max_err={max_err}, expected ~{expected}");

                driver::mem_free_host(host).expect("free");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Load real GPTQ model weights, run Marlin GEMM, compare against CPU dequantized reference.
        #[test]
        fn test_cuda_gptq_real_model_correctness() {
            use std::path::Path;

            let model_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GPTQ-Int4/snapshots/c34a4a91629f09f73a285f32dbd26106b033c654";
            let st_path = format!("{model_dir}/model.safetensors");
            if !Path::new(&st_path).exists() {
                eprintln!("Skipping: model not found at {st_path}");
                return;
            }

            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            // Load tensors from safetensors
            let file = std::fs::File::open(&st_path).unwrap();
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + header_size]).unwrap();
            let data_start = 8 + header_size;

            let prefix = "model.layers.0.self_attn.q_proj";

            // Load qweight [K/8, N] I32
            let qw_info = &header[format!("{prefix}.qweight")];
            let qw_shape: Vec<usize> = qw_info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let qw_offsets: Vec<usize> = qw_info["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let qw_bytes = &mmap[data_start + qw_offsets[0]..data_start + qw_offsets[1]];
            let qw_i32: Vec<i32> = qw_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let size_k = qw_shape[0] * 8;
            let size_n = qw_shape[1];
            let group_size = 128usize;
            let num_groups = size_k / group_size;
            eprintln!("qweight: shape={qw_shape:?}, size_k={size_k}, size_n={size_n}");

            // Load scales [num_groups, N] F16
            let sc_info = &header[format!("{prefix}.scales")];
            let sc_offsets: Vec<usize> = sc_info["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sc_bytes = &mmap[data_start + sc_offsets[0]..data_start + sc_offsets[1]];
            let scales_f32: Vec<f32> = sc_bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();

            // CPU reference: dequantize and compute y = x @ W^T
            // x = all-ones [1, K]
            let mut cpu_output = vec![0.0f32; size_n];
            for n in 0..size_n {
                let mut sum = 0.0f32;
                for k in 0..size_k {
                    let packed_row = k / 8;
                    let bit_pos = k % 8;
                    let packed_val = qw_i32[packed_row * size_n + n] as u32;
                    let nibble = ((packed_val >> (bit_pos * 4)) & 0xF) as i32;
                    let dequant = (nibble - 8) as f32; // kU4B8

                    let group = k / group_size;
                    let scale = scales_f32[group * size_n + n];
                    sum += dequant * scale;
                }
                cpu_output[n] = sum;
            }
            eprintln!("CPU reference first8: {:?}", &cpu_output[..8]);

            // GPU: repack + permute scales + Marlin GEMM
            unsafe {
                // Upload qweight
                let qw_nbytes = qw_bytes.len();
                let qw_ptr = driver::mem_alloc(qw_nbytes).expect("alloc");
                driver::memcpy_htod_async(qw_ptr, qw_bytes.as_ptr(), qw_nbytes, stream)
                    .expect("h2d");
                let qw_gpu = GpuTensor::new(qw_ptr, &qw_shape, DType::I32);

                // Repack
                let repacked = crate::kernels::gptq_repack(
                    qw_gpu, None, size_k, size_n, 0, &mut alloc, stream,
                );

                // Permute scales on CPU, then upload
                let mut scales_u16: Vec<u16> = sc_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                super::super::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);
                let s_bytes: Vec<u8> = scales_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
                let s_ptr = driver::mem_alloc(s_bytes.len()).expect("alloc");
                driver::memcpy_htod_async(s_ptr, s_bytes.as_ptr(), s_bytes.len(), stream)
                    .expect("h2d");
                let scales_gpu = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                // Input: all ones [1, K]
                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_k];
                let a_ptr = driver::mem_alloc(a_data.len() * 2).expect("alloc");
                driver::memcpy_htod_async(
                    a_ptr,
                    a_data.as_ptr() as *const u8,
                    a_data.len() * 2,
                    stream,
                )
                .expect("h2d");
                let a = GpuTensor::new(a_ptr, &[1, size_k], DType::F16);

                let ws_ptr = driver::mem_alloc(256 * 4).expect("alloc");
                driver::memset_d8(ws_ptr, 0, 256 * 4, stream).expect("memset");
                let workspace = GpuTensor::new(ws_ptr, &[256], DType::I32);

                let out = crate::kernels::marlin_gemm(
                    a,
                    repacked.as_gpu_tensor(),
                    scales_gpu,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    1,
                    size_n,
                    size_k,
                    num_groups,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                let out_nbytes = size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let gpu_output: Vec<f32> =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_n)
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect();

                eprintln!("GPU output first8:   {:?}", &gpu_output[..8]);

                // Compare
                let mut max_err: f32 = 0.0;
                let mut max_err_idx = 0;
                for i in 0..size_n {
                    let err = (gpu_output[i] - cpu_output[i]).abs();
                    if err > max_err {
                        max_err = err;
                        max_err_idx = i;
                    }
                }
                eprintln!("max error: {max_err} at index {max_err_idx}");
                eprintln!(
                    "  CPU[{max_err_idx}]={}, GPU[{max_err_idx}]={}",
                    cpu_output[max_err_idx], gpu_output[max_err_idx]
                );

                // F16 has limited precision, allow reasonable error
                assert!(
                    max_err < 5.0,
                    "max_err={max_err} too large — GEMM mismatch at idx {max_err_idx}"
                );

                driver::mem_free_host(host).expect("free");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Same test as test_cuda_gptq_real_model_correctness but for gate_proj (K=896, N=4864).
        #[test]
        fn test_cuda_gptq_gate_proj_correctness() {
            use std::path::Path;

            let model_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GPTQ-Int4/snapshots/c34a4a91629f09f73a285f32dbd26106b033c654";
            let st_path = format!("{model_dir}/model.safetensors");
            if !Path::new(&st_path).exists() {
                eprintln!("Skipping: model not found at {st_path}");
                return;
            }

            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let file = std::fs::File::open(&st_path).unwrap();
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + header_size]).unwrap();
            let data_start = 8 + header_size;

            let prefix = "model.layers.0.mlp.gate_proj";

            let qw_info = &header[format!("{prefix}.qweight")];
            let qw_shape: Vec<usize> = qw_info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let qw_offsets: Vec<usize> = qw_info["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let qw_bytes = &mmap[data_start + qw_offsets[0]..data_start + qw_offsets[1]];
            let qw_i32: Vec<i32> = qw_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let size_k = qw_shape[0] * 8;
            let size_n = qw_shape[1];
            let group_size = 128usize;
            let num_groups = size_k / group_size;
            eprintln!(
                "gate_proj: shape={qw_shape:?}, size_k={size_k}, size_n={size_n}, num_groups={num_groups}"
            );

            let sc_info = &header[format!("{prefix}.scales")];
            let sc_offsets: Vec<usize> = sc_info["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sc_bytes = &mmap[data_start + sc_offsets[0]..data_start + sc_offsets[1]];
            let scales_f32: Vec<f32> = sc_bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();

            // CPU reference dequantization
            let mut cpu_output = vec![0.0f32; size_n];
            for n in 0..size_n {
                let mut sum = 0.0f32;
                for k in 0..size_k {
                    let packed_row = k / 8;
                    let bit_pos = k % 8;
                    let packed_val = qw_i32[packed_row * size_n + n] as u32;
                    let nibble = ((packed_val >> (bit_pos * 4)) & 0xF) as i32;
                    let dequant = (nibble - 8) as f32;
                    let group = k / group_size;
                    let scale = scales_f32[group * size_n + n];
                    sum += dequant * scale;
                }
                cpu_output[n] = sum;
            }
            eprintln!("CPU reference first8: {:?}", &cpu_output[..8]);

            unsafe {
                let qw_nbytes = qw_bytes.len();
                let qw_ptr = driver::mem_alloc(qw_nbytes).expect("alloc");
                driver::memcpy_htod_async(qw_ptr, qw_bytes.as_ptr(), qw_nbytes, stream)
                    .expect("h2d");
                let qw_gpu = GpuTensor::new(qw_ptr, &qw_shape, DType::I32);

                let repacked = crate::kernels::gptq_repack(
                    qw_gpu, None, size_k, size_n, 0, &mut alloc, stream,
                );

                let mut scales_u16: Vec<u16> = sc_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                super::super::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);
                let s_bytes: Vec<u8> = scales_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
                let s_ptr = driver::mem_alloc(s_bytes.len()).expect("alloc");
                driver::memcpy_htod_async(s_ptr, s_bytes.as_ptr(), s_bytes.len(), stream)
                    .expect("h2d");
                let scales_gpu = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_k];
                let a_ptr = driver::mem_alloc(a_data.len() * 2).expect("alloc");
                driver::memcpy_htod_async(
                    a_ptr,
                    a_data.as_ptr() as *const u8,
                    a_data.len() * 2,
                    stream,
                )
                .expect("h2d");
                let a = GpuTensor::new(a_ptr, &[1, size_k], DType::F16);

                let workspace = crate::weights::alloc_marlin_workspace(142, stream).expect("ws");

                let out = crate::kernels::marlin_gemm(
                    a,
                    repacked.as_gpu_tensor(),
                    scales_gpu,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    1,
                    size_n,
                    size_k,
                    num_groups,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                let out_nbytes = size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let gpu_output: Vec<f32> =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_n)
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect();

                eprintln!("GPU output first8:   {:?}", &gpu_output[..8]);

                let mut max_err: f32 = 0.0;
                let mut max_err_idx = 0;
                for i in 0..size_n {
                    let err = (gpu_output[i] - cpu_output[i]).abs();
                    if err > max_err {
                        max_err = err;
                        max_err_idx = i;
                    }
                }
                eprintln!("max error: {max_err} at index {max_err_idx}");
                assert!(max_err < 5.0, "max_err={max_err} too large");

                driver::mem_free_host(host).expect("free");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Same test for down_proj (K=4864, N=896).
        #[test]
        fn test_cuda_gptq_down_proj_correctness() {
            use std::path::Path;

            let model_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GPTQ-Int4/snapshots/c34a4a91629f09f73a285f32dbd26106b033c654";
            let st_path = format!("{model_dir}/model.safetensors");
            if !Path::new(&st_path).exists() {
                eprintln!("Skipping: model not found at {st_path}");
                return;
            }

            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let file = std::fs::File::open(&st_path).unwrap();
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + header_size]).unwrap();
            let data_start = 8 + header_size;

            let prefix = "model.layers.0.mlp.down_proj";

            let qw_info = &header[format!("{prefix}.qweight")];
            let qw_shape: Vec<usize> = qw_info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let qw_offsets: Vec<usize> = qw_info["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let qw_bytes = &mmap[data_start + qw_offsets[0]..data_start + qw_offsets[1]];
            let qw_i32: Vec<i32> = qw_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let size_k = qw_shape[0] * 8;
            let size_n = qw_shape[1];
            let group_size = 128usize;
            let num_groups = size_k / group_size;
            eprintln!(
                "down_proj: shape={qw_shape:?}, size_k={size_k}, size_n={size_n}, num_groups={num_groups}"
            );

            let sc_info = &header[format!("{prefix}.scales")];
            let sc_offsets: Vec<usize> = sc_info["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sc_bytes = &mmap[data_start + sc_offsets[0]..data_start + sc_offsets[1]];
            let scales_f32: Vec<f32> = sc_bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();

            let mut cpu_output = vec![0.0f32; size_n];
            for n in 0..size_n {
                let mut sum = 0.0f32;
                for k in 0..size_k {
                    let packed_row = k / 8;
                    let bit_pos = k % 8;
                    let packed_val = qw_i32[packed_row * size_n + n] as u32;
                    let nibble = ((packed_val >> (bit_pos * 4)) & 0xF) as i32;
                    let dequant = (nibble - 8) as f32;
                    let group = k / group_size;
                    let scale = scales_f32[group * size_n + n];
                    sum += dequant * scale;
                }
                cpu_output[n] = sum;
            }
            eprintln!("CPU reference first8: {:?}", &cpu_output[..8]);

            unsafe {
                let qw_nbytes = qw_bytes.len();
                let qw_ptr = driver::mem_alloc(qw_nbytes).expect("alloc");
                driver::memcpy_htod_async(qw_ptr, qw_bytes.as_ptr(), qw_nbytes, stream)
                    .expect("h2d");
                let qw_gpu = GpuTensor::new(qw_ptr, &qw_shape, DType::I32);

                let repacked = crate::kernels::gptq_repack(
                    qw_gpu, None, size_k, size_n, 0, &mut alloc, stream,
                );

                let mut scales_u16: Vec<u16> = sc_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                super::super::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);
                let s_bytes: Vec<u8> = scales_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
                let s_ptr = driver::mem_alloc(s_bytes.len()).expect("alloc");
                driver::memcpy_htod_async(s_ptr, s_bytes.as_ptr(), s_bytes.len(), stream)
                    .expect("h2d");
                let scales_gpu = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_k];
                let a_ptr = driver::mem_alloc(a_data.len() * 2).expect("alloc");
                driver::memcpy_htod_async(
                    a_ptr,
                    a_data.as_ptr() as *const u8,
                    a_data.len() * 2,
                    stream,
                )
                .expect("h2d");
                let a = GpuTensor::new(a_ptr, &[1, size_k], DType::F16);

                let workspace = crate::weights::alloc_marlin_workspace(142, stream).expect("ws");

                let out = crate::kernels::marlin_gemm(
                    a,
                    repacked.as_gpu_tensor(),
                    scales_gpu,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    1,
                    size_n,
                    size_k,
                    num_groups,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                let out_nbytes = size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let gpu_output: Vec<f32> =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_n)
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect();

                eprintln!("GPU output first8:   {:?}", &gpu_output[..8]);

                let mut max_err: f32 = 0.0;
                let mut max_err_idx = 0;
                for i in 0..size_n {
                    let err = (gpu_output[i] - cpu_output[i]).abs();
                    if err > max_err {
                        max_err = err;
                        max_err_idx = i;
                    }
                }
                eprintln!("max error: {max_err} at index {max_err_idx}");
                assert!(max_err < 5.0, "max_err={max_err} too large");

                driver::mem_free_host(host).expect("free");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Full MLP integration: gate(x) + up(x) → concat → silu_and_mul → down(result)
        /// Tests the composed pipeline, not just individual GEMMs.
        #[test]
        fn test_cuda_gptq_mlp_integration() {
            use std::path::Path;

            let model_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GPTQ-Int4/snapshots/c34a4a91629f09f73a285f32dbd26106b033c654";
            let st_path = format!("{model_dir}/model.safetensors");
            if !Path::new(&st_path).exists() {
                eprintln!("Skipping: model not found at {st_path}");
                return;
            }

            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let file = std::fs::File::open(&st_path).unwrap();
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + header_size]).unwrap();
            let data_start = 8 + header_size;

            let group_size = 128usize;
            let hidden_size = 896usize;
            let intermediate_size = 4864usize;

            // Helper: load GPTQ linear layer (qweight + scales) → repacked + permuted
            let load_linear =
                |prefix: &str,
                 alloc: &mut CachingAllocator|
                 -> (crate::alloc::OwnedTensor, GpuTensor, usize, usize, usize) {
                    let qw_info = &header[format!("{prefix}.qweight")];
                    let qw_shape: Vec<usize> = qw_info["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let qw_offsets: Vec<usize> = qw_info["data_offsets"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let qw_bytes = &mmap[data_start + qw_offsets[0]..data_start + qw_offsets[1]];
                    let size_k = qw_shape[0] * 8;
                    let size_n = qw_shape[1];
                    let num_groups = size_k / group_size;

                    unsafe {
                        let qw_ptr = driver::mem_alloc(qw_bytes.len()).expect("alloc");
                        driver::memcpy_htod_async(
                            qw_ptr,
                            qw_bytes.as_ptr(),
                            qw_bytes.len(),
                            stream,
                        )
                        .expect("h2d");
                        let qw_gpu = GpuTensor::new(qw_ptr, &qw_shape, DType::I32);
                        let repacked = crate::kernels::gptq_repack(
                            qw_gpu, None, size_k, size_n, 0, alloc, stream,
                        );

                        let sc_info = &header[format!("{prefix}.scales")];
                        let sc_offsets: Vec<usize> = sc_info["data_offsets"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect();
                        let sc_bytes =
                            &mmap[data_start + sc_offsets[0]..data_start + sc_offsets[1]];
                        let mut scales_u16: Vec<u16> = sc_bytes
                            .chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect();
                        super::super::marlin_permute_scales(
                            &mut scales_u16,
                            size_k,
                            size_n,
                            group_size,
                        );
                        let s_bytes: Vec<u8> =
                            scales_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let s_ptr = driver::mem_alloc(s_bytes.len()).expect("alloc");
                        driver::memcpy_htod_async(s_ptr, s_bytes.as_ptr(), s_bytes.len(), stream)
                            .expect("h2d");
                        let scales_gpu = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                        driver::mem_free(qw_ptr).expect("free original qweight");

                        (repacked, scales_gpu, size_k, size_n, num_groups)
                    }
                };

            let prefix = "model.layers.0.mlp";
            let (gate_w, gate_s, gate_k, gate_n, gate_ng) =
                load_linear(&format!("{prefix}.gate_proj"), &mut alloc);
            let (up_w, up_s, up_k, up_n, up_ng) =
                load_linear(&format!("{prefix}.up_proj"), &mut alloc);
            let (down_w, down_s, down_k, down_n, down_ng) =
                load_linear(&format!("{prefix}.down_proj"), &mut alloc);

            eprintln!(
                "gate: K={gate_k} N={gate_n}, up: K={up_k} N={up_n}, down: K={down_k} N={down_n}"
            );
            assert_eq!(gate_k, hidden_size);
            assert_eq!(gate_n, intermediate_size);
            assert_eq!(up_k, hidden_size);
            assert_eq!(up_n, intermediate_size);
            assert_eq!(down_k, intermediate_size);
            assert_eq!(down_n, hidden_size);

            unsafe {
                let workspace = crate::weights::alloc_marlin_workspace(142, stream).expect("ws");

                // Create input: small values mimicking normed hidden state
                let input_f32: Vec<f32> = (0..hidden_size)
                    .map(|i| (i as f32 * 0.01).sin() * 0.5)
                    .collect();
                let input_f16: Vec<u16> = input_f32
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect();
                let a_ptr = driver::mem_alloc(input_f16.len() * 2).expect("alloc");
                driver::memcpy_htod_async(
                    a_ptr,
                    input_f16.as_ptr() as *const u8,
                    input_f16.len() * 2,
                    stream,
                )
                .expect("h2d");
                let a = GpuTensor::new(a_ptr, &[1, hidden_size], DType::F16);

                // Step 1: gate(x) → [1, 4864]
                let gate_out = crate::kernels::marlin_gemm(
                    a,
                    gate_w.as_gpu_tensor(),
                    gate_s,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    1,
                    gate_n,
                    gate_k,
                    gate_ng,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                // Step 2: up(x) → [1, 4864]
                let up_out = crate::kernels::marlin_gemm(
                    a,
                    up_w.as_gpu_tensor(),
                    up_s,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    1,
                    up_n,
                    up_k,
                    up_ng,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                // Step 3: concat → [1, 9728]
                let gate_up = crate::kernels::concat_dim1(
                    gate_out.as_gpu_tensor(),
                    up_out.as_gpu_tensor(),
                    &mut alloc,
                    stream,
                );

                // Step 4: silu_and_mul → [1, 4864]
                let activated = crate::kernels::silu_and_mul_fused(
                    gate_up.as_gpu_tensor(),
                    intermediate_size,
                    &mut alloc,
                    stream,
                );

                // Step 5: down(activated) → [1, 896]
                let final_out = crate::kernels::marlin_gemm(
                    activated.as_gpu_tensor(),
                    down_w.as_gpu_tensor(),
                    down_s,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    1,
                    down_n,
                    down_k,
                    down_ng,
                    group_size,
                    false,
                    false,
                    0,
                    0,
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                // Read back intermediate and final results
                let read_gpu = |t: &crate::alloc::OwnedTensor, n: usize| -> Vec<f32> {
                    let nbytes = n * 2;
                    let host = driver::mem_alloc_host(nbytes).expect("host");
                    driver::memcpy_dtoh_async(host.as_ptr(), t.raw_ptr(), nbytes, stream)
                        .expect("d2h");
                    driver::stream_synchronize(stream).expect("sync");
                    let vals: Vec<f32> = std::slice::from_raw_parts(host.as_ptr() as *const u16, n)
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect();
                    driver::mem_free_host(host).expect("free");
                    vals
                };

                let gate_vals = read_gpu(&gate_out, gate_n.min(8));
                let up_vals = read_gpu(&up_out, up_n.min(8));
                let activated_vals = read_gpu(&activated, intermediate_size.min(8));
                let final_vals = read_gpu(&final_out, hidden_size.min(8));

                eprintln!("gate_out first8: {gate_vals:?}");
                eprintln!("up_out first8:   {up_vals:?}");
                eprintln!("activated first8: {activated_vals:?}");
                eprintln!("final_out first8: {final_vals:?}");

                // Check that output has reasonable magnitude (not near-zero)
                let max_abs = final_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("final_out max_abs: {max_abs}");
                assert!(max_abs > 0.01, "MLP output is near-zero: max_abs={max_abs}");

                // Also check gate_out and activated aren't near-zero
                let gate_max = gate_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let act_max = activated_vals
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max);
                eprintln!("gate_max={gate_max}, activated_max={act_max}");

                driver::mem_free(a_ptr).expect("free");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        /// Test GPTQ desc_act (activation ordering) pipeline:
        /// create known weights + non-trivial g_idx → argsort → repack with perm → GEMM → verify.
        ///
        /// With desc_act, the kernel permutes activation columns via sort_indices before GEMM.
        /// All INT4 values = 9 (kU4B8: 9-8 = +1), scales = 1.0, input = all-ones.
        /// The result should be identical to the non-desc_act case (output[j] = K),
        /// because the permutation is undone by the act_order pipeline.
        #[test]
        fn test_cuda_gptq_desc_act_repack_gemm() {
            let stream = init_cuda();
            let mut alloc = CachingAllocator::new();

            let size_k: usize = 256;
            let size_n: usize = 256;
            let group_size: usize = 128;
            let num_groups = size_k / group_size;
            let size_m: usize = 1;

            unsafe {
                // 1. Create GPTQ-format qweight: [K/8, N] u32, all nibbles = 9
                let pack_factor = 8;
                let gptq_rows = size_k / pack_factor;
                let gptq_count = gptq_rows * size_n;
                let gptq_data = vec![0x99999999u32; gptq_count];

                let gptq_ptr = driver::mem_alloc(gptq_count * 4).expect("alloc gptq");
                driver::memcpy_htod_async(
                    gptq_ptr,
                    gptq_data.as_ptr() as *const u8,
                    gptq_count * 4,
                    stream,
                )
                .expect("h2d gptq");
                let gptq_gpu = GpuTensor::new(gptq_ptr, &[gptq_rows, size_n], DType::U32);

                // 2. Create g_idx: reverse order (group 1 first, then group 0)
                //    This simulates desc_act where channels are reordered by activation magnitude.
                let mut g_idx: Vec<i32> = Vec::with_capacity(size_k);
                for i in 0..size_k {
                    // Reverse: first half → group 1, second half → group 0
                    if i < group_size {
                        g_idx.push(1);
                    } else {
                        g_idx.push(0);
                    }
                }

                // 3. Argsort g_idx (stable ascending by group ID)
                let mut sort_indices: Vec<i32> = (0..size_k as i32).collect();
                sort_indices.sort_by_key(|&i| g_idx[i as usize]);

                let sorted_g_idx: Vec<i32> =
                    sort_indices.iter().map(|&i| g_idx[i as usize]).collect();

                // Verify argsort: sorted_g_idx should be [0,0,...,1,1,...]
                assert_eq!(sorted_g_idx[0], 0);
                assert_eq!(sorted_g_idx[group_size - 1], 0);
                assert_eq!(sorted_g_idx[group_size], 1);
                assert_eq!(sorted_g_idx[size_k - 1], 1);

                // Upload sorted_g_idx to GPU
                let g_idx_bytes: Vec<u8> =
                    sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let g_idx_ptr = driver::mem_alloc(g_idx_bytes.len()).expect("alloc g_idx");
                driver::memcpy_htod_async(
                    g_idx_ptr,
                    g_idx_bytes.as_ptr(),
                    g_idx_bytes.len(),
                    stream,
                )
                .expect("h2d g_idx");
                let g_idx_gpu = GpuTensor::new(g_idx_ptr, &[size_k], DType::I32);

                // Upload sort_indices to GPU
                let si_bytes: Vec<u8> =
                    sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let si_ptr = driver::mem_alloc(si_bytes.len()).expect("alloc si");
                driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_bytes.len(), stream)
                    .expect("h2d si");
                let sort_indices_gpu = GpuTensor::new(si_ptr, &[size_k], DType::I32);

                // 4. Repack GPTQ → Marlin tiled layout WITH perm (sort_indices)
                let repacked = crate::kernels::gptq_repack(
                    gptq_gpu,
                    Some(sort_indices_gpu),
                    size_k,
                    size_n,
                    0,
                    &mut alloc,
                    stream,
                );

                // 5. Create scales [num_groups, N] = all ones, then permute
                let mut scales_u16: Vec<u16> =
                    vec![half::f16::from_f32(1.0).to_bits(); num_groups * size_n];
                super::super::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

                let s_nbytes = scales_u16.len() * 2;
                let s_ptr = driver::mem_alloc(s_nbytes).expect("alloc scales");
                driver::memcpy_htod_async(
                    s_ptr,
                    scales_u16.as_ptr() as *const u8,
                    s_nbytes,
                    stream,
                )
                .expect("h2d scales");
                let scales = GpuTensor::new(s_ptr, &[num_groups, size_n], DType::F16);

                // 6. Activation: [1, K] f16, all ones
                let a_data: Vec<u16> = vec![half::f16::from_f32(1.0).to_bits(); size_m * size_k];
                let a_nbytes = a_data.len() * 2;
                let a_ptr = driver::mem_alloc(a_nbytes).expect("alloc a");
                driver::memcpy_htod_async(a_ptr, a_data.as_ptr() as *const u8, a_nbytes, stream)
                    .expect("h2d a");
                let a = GpuTensor::new(a_ptr, &[size_m, size_k], DType::F16);

                // 7. Workspace
                let ws_count = 128;
                let ws_ptr = driver::mem_alloc(ws_count * 4).expect("alloc ws");
                driver::memset_d8(ws_ptr, 0, ws_count * 4, stream).expect("memset ws");
                let workspace = GpuTensor::new(ws_ptr, &[ws_count], DType::I32);

                // 8. Run Marlin GEMM with act_order=true
                let out = crate::kernels::marlin_gemm(
                    a,
                    repacked.as_gpu_tensor(),
                    scales,
                    None,                   // no zeros (GPTQ uint4b8)
                    Some(g_idx_gpu),        // sorted g_idx
                    Some(sort_indices_gpu), // perm
                    None,                   // no bias
                    workspace,
                    size_m,
                    size_n,
                    size_k,
                    num_groups,
                    group_size,
                    true,  // has_act_order
                    false, // no zp
                    0,     // GPTQ type
                    0,     // device_id
                    &mut alloc,
                    stream,
                );

                driver::stream_synchronize(stream).expect("sync");

                // 9. Read output and verify
                let out_nbytes = size_m * size_n * 2;
                let host = driver::mem_alloc_host(out_nbytes).expect("host alloc");
                driver::memcpy_dtoh_async(host.as_ptr(), out.raw_ptr(), out_nbytes, stream)
                    .expect("d2h");
                driver::stream_synchronize(stream).expect("sync");

                let result =
                    std::slice::from_raw_parts(host.as_ptr() as *const u16, size_m * size_n);
                let expected = size_k as f32; // each output = sum of K * 1.0 * 1.0 = K

                let mut max_err: f32 = 0.0;
                for (i, &bits) in result.iter().enumerate() {
                    let val = half::f16::from_bits(bits).to_f32();
                    let err = (val - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    if i < 8 {
                        eprintln!("desc_act out[{i}] = {val} (expected {expected}, err={err})");
                    }
                }
                eprintln!("desc_act max error across {size_n} outputs: {max_err}");

                // Same tolerance as non-desc_act test
                assert!(
                    max_err < 2.0,
                    "desc_act max_err={max_err} too large, expected ~{expected} for all outputs"
                );

                driver::mem_free_host(host).expect("free host");
                driver::mem_free(a_ptr).expect("free a");
                driver::mem_free(gptq_ptr).expect("free gptq");
                driver::mem_free(g_idx_ptr).expect("free g_idx");
                driver::mem_free(si_ptr).expect("free si");
                driver::stream_destroy(stream).expect("destroy");
            }
        }

        #[test]
        fn test_cuda_marlin_workspace_alloc() {
            let stream = init_cuda();
            let workspace =
                crate::weights::alloc_marlin_workspace(128, stream).expect("workspace alloc");
            assert_eq!(workspace.dim(0), 1024 * 1024); // max(2*128, 1M)
            assert_eq!(workspace.dtype(), DType::I32);
            unsafe { driver::stream_destroy(stream).expect("destroy") };
        }
    }

    #[test]
    fn test_detect_bnb4bit_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": true,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "bfloat16"
                }
            }"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        assert!(matches!(config, QuantConfig::Bnb4bit(_)));
        assert!(config.is_bnb4bit());
        assert_eq!(config.bits(), 4);
        if let QuantConfig::Bnb4bit(ref c) = config {
            assert_eq!(c.blocksize, 64);
            assert_eq!(c.quant_type, BnbQuantType::NF4);
        }
    }

    #[test]
    fn test_detect_bnb4bit_fp4() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": true,
                    "bnb_4bit_quant_type": "fp4"
                }
            }"#,
        )
        .unwrap();
        let config = detect_quant_config(dir.path()).unwrap();
        if let QuantConfig::Bnb4bit(ref c) = config {
            assert_eq!(c.quant_type, BnbQuantType::FP4);
        } else {
            panic!("expected Bnb4bit config");
        }
    }

    #[test]
    fn test_nf4_code_table() {
        assert_eq!(NF4_CODE.len(), 16);
        assert_eq!(NF4_CODE[0], -1.0);
        assert_eq!(NF4_CODE[7], 0.0);
        assert_eq!(NF4_CODE[15], 1.0);
        // Table should be monotonically increasing.
        for i in 1..16 {
            assert!(
                NF4_CODE[i] > NF4_CODE[i - 1],
                "NF4 table not monotonic at {i}"
            );
        }
    }
}
