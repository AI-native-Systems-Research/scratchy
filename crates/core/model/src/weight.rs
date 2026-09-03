// SPDX-License-Identifier: Apache-2.0
//! SafeTensors index and HuggingFace model config.
//!
//! Provides:
//! - `SafeTensorsIndex`: parse `model.safetensors.index.json` for shard lookups
//! - `HfModelConfig`: parse HuggingFace `config.json` for architecture info

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ModelResult;

// ---------------------------------------------------------------------------
// Sharded weight index (model.safetensors.index.json)
// ---------------------------------------------------------------------------

/// The index file that maps tensor names to shard files.
///
/// Format: `model.safetensors.index.json`
/// ```json
/// {
///   "metadata": { "total_size": 12345 },
///   "weight_map": {
///     "model.layers.0.weight": "model-00001-of-00002.safetensors",
///     ...
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsIndex {
    /// Metadata (usually just total_size).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Maps tensor name → shard filename.
    pub weight_map: HashMap<String, String>,
}

impl SafeTensorsIndex {
    /// Load from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let index: Self = serde_json::from_str(&data)?;
        Ok(index)
    }

    /// Get the set of unique shard filenames.
    pub fn shard_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();
        files
    }

    /// Look up which shard file contains a given tensor.
    pub fn get_shard(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }

    /// Total on-disk size of the model in bytes, from the index's
    /// `metadata.total_size`. `None` if the field is absent or non-numeric.
    pub fn total_size(&self) -> Option<u64> {
        self.metadata.get("total_size").and_then(|v| v.as_u64())
    }

    /// Preflight free-space check before downloading shards into `model_dir`.
    ///
    /// An interrupted download from a full disk is the canonical cause of a
    /// partially-populated cache; resuming into a still-full disk just fails
    /// again mid-stream (and burns the download retry budget on a non-transient
    /// error). This turns that into an upfront, actionable error.
    ///
    /// Estimates remaining bytes as `total_size - (bytes of shards already on
    /// disk)` — shards are written via atomic rename, so an on-disk shard is
    /// complete and its `len()` is exact. Requires ~1% headroom on top.
    ///
    /// Returns `Ok(())` (skips the check) when `total_size` is absent or the
    /// filesystem's free space can't be queried — never a false alarm.
    pub fn ensure_disk_space(&self, model_dir: &Path) -> Result<(), DiskSpaceError> {
        let Some(total) = self.total_size() else {
            return Ok(());
        };
        let on_disk: u64 = self
            .shard_files()
            .iter()
            .filter_map(|s| std::fs::metadata(model_dir.join(s)).ok())
            .map(|m| m.len())
            .sum();
        let remaining = total.saturating_sub(on_disk);
        let needed = remaining + remaining / 100;

        let Some(available) = available_space(model_dir) else {
            return Ok(());
        };
        if available < needed {
            return Err(DiskSpaceError {
                needed,
                available,
                dir: model_dir.display().to_string(),
            });
        }
        Ok(())
    }
}

/// Free bytes available for a large download into `dir`, or `None` if the
/// filesystem can't be queried.
///
/// On macOS/APFS the `statvfs` figure (what `fs2::available_space` returns)
/// omits *purgeable* space: reclaimable caches and local snapshots the OS frees
/// on demand as a large file is written. Finder and System Settings instead
/// report the "important usage" capacity, which counts that reclaimable space,
/// so the two can differ by tens of GB. Trusting the `statvfs` figure made this
/// preflight reject downloads that would actually succeed. We query the
/// important-usage capacity on macOS and fall back to `statvfs` (accurate on
/// Linux and when the macOS query fails).
fn available_space(dir: &Path) -> Option<u64> {
    #[cfg(target_os = "macos")]
    if let Some(bytes) = macos::available_for_important_usage(dir) {
        return Some(bytes);
    }
    fs2::available_space(dir).ok()
}

/// CoreFoundation binding for the macOS "available for important usage"
/// volume capacity — the purgeable-aware figure Finder and System Settings show.
#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::c_void;
    use std::os::raw::c_long;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    type CFTypeRef = *const c_void;
    type CFAllocatorRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFURLRef = *const c_void;
    type CFIndex = c_long;
    type Boolean = u8;

    // kCFNumberSInt64Type
    const CF_NUMBER_SINT64: c_long = 4;

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        #[link_name = "kCFURLVolumeAvailableCapacityForImportantUsageKey"]
        static VOLUME_AVAILABLE_IMPORTANT_USAGE_KEY: CFStringRef;

        fn CFURLCreateFromFileSystemRepresentation(
            allocator: CFAllocatorRef,
            buffer: *const u8,
            buf_len: CFIndex,
            is_directory: Boolean,
        ) -> CFURLRef;
        fn CFURLCopyResourcePropertyForKey(
            url: CFURLRef,
            key: CFStringRef,
            value_ptr: *mut CFTypeRef,
            error: *mut CFTypeRef,
        ) -> Boolean;
        fn CFNumberGetValue(number: CFTypeRef, the_type: c_long, value_ptr: *mut c_void)
        -> Boolean;
        fn CFRelease(cf: CFTypeRef);
    }

    /// Bytes available for "important usage" at `dir`, counting purgeable space,
    /// or `None` on any failure (missing path, non-CFNumber result, negative).
    pub fn available_for_important_usage(dir: &Path) -> Option<u64> {
        let bytes = dir.as_os_str().as_bytes();
        // SAFETY: `bytes` outlives the call; every CF object we create is
        // released exactly once; out-params are only read after a true return.
        unsafe {
            let url = CFURLCreateFromFileSystemRepresentation(
                std::ptr::null(),
                bytes.as_ptr(),
                bytes.len() as CFIndex,
                1, // isDirectory
            );
            if url.is_null() {
                return None;
            }
            let mut value: CFTypeRef = std::ptr::null();
            let mut error: CFTypeRef = std::ptr::null();
            let ok = CFURLCopyResourcePropertyForKey(
                url,
                VOLUME_AVAILABLE_IMPORTANT_USAGE_KEY,
                &mut value,
                &mut error,
            );
            let result = if ok != 0 && !value.is_null() {
                let mut out: i64 = 0;
                let got = CFNumberGetValue(
                    value,
                    CF_NUMBER_SINT64,
                    (&mut out as *mut i64).cast::<c_void>(),
                );
                CFRelease(value);
                (got != 0 && out >= 0).then_some(out as u64)
            } else {
                if !error.is_null() {
                    CFRelease(error);
                }
                None
            };
            CFRelease(url);
            result
        }
    }
}

/// Not enough free disk space to download the remaining model shards.
#[derive(Debug, Clone)]
pub struct DiskSpaceError {
    /// Estimated bytes still needed (remaining shards + ~1% headroom).
    pub needed: u64,
    /// Bytes currently free on the cache filesystem.
    pub available: u64,
    /// The cache directory checked.
    pub dir: String,
}

impl std::fmt::Display for DiskSpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "insufficient disk space: need ~{:.1} GB more but only {:.1} GB free at {}. \
             Free up space or point HF_HOME at a larger volume.",
            self.needed as f64 / 1e9,
            self.available as f64 / 1e9,
            self.dir,
        )
    }
}

impl std::error::Error for DiskSpaceError {}

// ---------------------------------------------------------------------------
// HuggingFace config.json parser
// ---------------------------------------------------------------------------

/// Minimal HuggingFace model configuration parsed from `config.json`.
///
/// Only includes fields commonly needed by vLLM for model loading.
/// The full config can be accessed via the raw JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HfModelConfig {
    /// Model architecture identifiers (e.g., ["LlamaForCausalLM"]).
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub architectures: Vec<String>,

    /// Model type (e.g., "llama", "mistral", "qwen2").
    #[serde(default)]
    pub model_type: Option<String>,

    /// Hidden size / model dimension.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of hidden layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Intermediate size (FFN dimension).
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// Maximum sequence length.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RMS norm epsilon.
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,

    /// Layer norm epsilon.
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,

    /// RoPE theta.
    #[serde(default)]
    pub rope_theta: Option<f64>,

    /// Torch dtype string (e.g., "float16", "bfloat16").
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    /// Head dimension (if explicitly specified).
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Raw JSON for accessing any field not in this struct.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl HfModelConfig {
    /// Load from a `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> ModelResult<Self> {
        let data = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }

    /// Load from a model directory (reads `config.json` inside it).
    pub fn from_dir(dir: impl AsRef<Path>) -> ModelResult<Self> {
        let path = dir.as_ref().join("config.json");
        Self::from_file(path)
    }

    /// Load from a model directory (reads `config.json`).
    ///
    /// For `.gguf` paths, callers must use `scratchy_quantizations::gguf::gguf_model_config`
    /// directly — GGUF format support lives in scratchy, not scratchy-core-model.
    pub fn from_path(path: impl AsRef<Path>) -> ModelResult<Self> {
        Self::from_dir(path)
    }

    /// Effective head dimension.
    ///
    /// For MLA models (DeepSeek V2/V3) this returns `qk_nope_head_dim + qk_rope_head_dim`
    /// so that the KV cache is allocated with the correct dimension.
    pub fn head_dim(&self) -> Option<usize> {
        if let (Some(nope), Some(rope)) = (
            self.extra.get("qk_nope_head_dim").and_then(|v| v.as_u64()),
            self.extra.get("qk_rope_head_dim").and_then(|v| v.as_u64()),
        ) {
            return Some((nope + rope) as usize);
        }
        self.head_dim
            .or_else(|| match (self.hidden_size, self.num_attention_heads) {
                (Some(h), Some(n)) if n > 0 => Some(h / n),
                _ => None,
            })
            // Composite configs (gemma4, kimi, etc.) keep the attention dims
            // only under nested `text_config`. Without this fallback the KV
            // cache sizer reads 0 and trips `compute_num_blocks`' 1024-block
            // fallback -> a 15 GB KV pool that OOMs the command buffer.
            .or_else(|| self.text_config_usize("head_dim"))
            .or_else(|| {
                match (
                    self.text_config_usize("hidden_size"),
                    self.text_config_usize("num_attention_heads"),
                ) {
                    (Some(h), Some(n)) if n > 0 => Some(h / n),
                    _ => None,
                }
            })
    }

    /// Effective number of KV heads (defaults to num_attention_heads for MHA).
    pub fn num_kv_heads(&self) -> Option<usize> {
        self.num_key_value_heads
            .or(self.num_attention_heads)
            .or_else(|| self.text_config_usize("num_key_value_heads"))
            .or_else(|| self.text_config_usize("num_attention_heads"))
    }

    /// Effective maximum context length (`max_position_embeddings`).
    ///
    /// Composite configs (gemma4, kimi, etc.) keep `max_position_embeddings`
    /// only under nested `text_config` — gemma-4-26b has 262144 there and
    /// nothing at top level. Without this fallback the engine's
    /// `max_model_len` silently defaults to 4096, so a 32k prompt is rejected
    /// as over-length. Mirrors [`Self::head_dim`] / [`Self::num_kv_heads`].
    pub fn max_position_embeddings(&self) -> Option<usize> {
        self.max_position_embeddings
            .or_else(|| self.text_config_usize("max_position_embeddings"))
    }

    /// Sliding-window size for SWA models (gemma2/3/4), falling through to
    /// nested `text_config` for composite configs (gemma-4-26b keeps
    /// `sliding_window: 1024` only there). `None` for full-attention models.
    pub fn sliding_window(&self) -> Option<usize> {
        self.extra
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| self.text_config_usize("sliding_window"))
    }

    /// Per-layer attention types, e.g.
    /// `["sliding_attention", ..., "full_attention", ...]`. Falls through to
    /// nested `text_config` (gemma4 keeps it there). `None` if absent — this
    /// is the same array the compiler derives its sliding-window pattern from,
    /// so the host's view of which layers are sliding matches the kernel's.
    pub fn layer_types(&self) -> Option<Vec<String>> {
        let val = self.extra.get("layer_types").or_else(|| {
            self.extra
                .get("text_config")
                .and_then(|t| t.get("layer_types"))
        })?;
        val.as_array().map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect()
        })
    }

    /// KV head count for the GLOBAL (full-attention) layer class on hybrid
    /// arches (gemma4: `num_global_key_value_heads = 2`). Falls back to the base
    /// [`Self::num_kv_heads`] for uniform-geometry models.
    pub fn num_global_kv_heads(&self) -> Option<usize> {
        self.extra
            .get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| self.text_config_usize("num_global_key_value_heads"))
            .or_else(|| self.num_kv_heads())
    }

    /// Head dim for the GLOBAL (full-attention) layer class on hybrid arches
    /// (gemma4: `global_head_dim = 512`). Falls back to the base
    /// [`Self::head_dim`] for uniform-geometry models.
    pub fn global_head_dim(&self) -> Option<usize> {
        self.extra
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| self.text_config_usize("global_head_dim"))
            .or_else(|| self.head_dim())
    }

    /// Per-layer KV geometry for the hybrid SWA KV layout:
    /// `(is_sliding, num_kv_heads, head_size, sliding_window)` in layer order.
    /// `None` unless [`Self::is_swa_hybrid`]. Sliding layers use the base dims;
    /// full layers use the GLOBAL dims (gemma4: sliding 8×256 / full 2×512).
    #[allow(clippy::type_complexity)]
    pub fn hybrid_layer_geometry(&self) -> Option<Vec<(bool, usize, usize, Option<usize>)>> {
        if !self.is_swa_hybrid() {
            return None;
        }
        let types = self.layer_types()?;
        let base_kv = self.num_kv_heads()?;
        let base_hd = self.head_dim()?;
        let global_kv = self.num_global_kv_heads()?;
        let global_hd = self.global_head_dim()?;
        let window = self.sliding_window();
        Some(
            types
                .iter()
                .map(|t| {
                    if t == "sliding_attention" {
                        (true, base_kv, base_hd, window)
                    } else {
                        (false, global_kv, global_hd, None)
                    }
                })
                .collect(),
        )
    }

    /// Whether the model mixes full and sliding-window attention layers
    /// (gemma4: 5 `full_attention` + 25 `sliding_attention`) — the case the
    /// hybrid SWA KV allocator targets. Requires both a `sliding_window` and a
    /// `layer_types` array that contains both kinds; models that are uniformly
    /// full (Llama) or uniformly sliding stay single-group.
    pub fn is_swa_hybrid(&self) -> bool {
        let Some(types) = self.layer_types() else {
            return false;
        };
        let has_full = types.iter().any(|t| t == "full_attention");
        let has_sliding = types.iter().any(|t| t == "sliding_attention");
        has_full && has_sliding && self.sliding_window().is_some()
    }

    /// Read an unsigned-int field from the nested `text_config` (composite
    /// HF configs like gemma4 keep the real text-backbone dims there).
    fn text_config_usize(&self, key: &str) -> Option<usize> {
        self.extra
            .get("text_config")
            .and_then(|t| t.get(key))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Effective norm epsilon.
    pub fn norm_eps(&self) -> f64 {
        self.rms_norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5)
    }

    /// For composite models (e.g. vision-language models like Kimi K2.5),
    /// extract the text sub-config.
    ///
    /// Returns `Some((text_config, weight_prefix_to_strip))` if this is a
    /// composite model whose text backbone is a supported architecture.
    /// Returns `None` for non-composite models.
    pub fn resolve_text_config(&self) -> Option<(Self, &'static str)> {
        match self.model_type.as_deref() {
            Some("kimi_k25") => {
                let text_config_val = self.extra.get("text_config")?;
                let mut cfg: Self = serde_json::from_value(text_config_val.clone()).ok()?;
                if cfg.architectures.is_empty() {
                    cfg.architectures = self.architectures.clone();
                }
                Some((cfg, "language_model."))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deserialize a value that may be `null` as the type's `Default`.
fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_index_parse() {
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");

        let index_json = r#"{
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.embed.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(&index_path, index_json).unwrap();

        let index = SafeTensorsIndex::from_file(&index_path).unwrap();
        assert_eq!(index.weight_map.len(), 4);

        let shards = index.shard_files();
        assert_eq!(shards.len(), 2);

        assert_eq!(
            index.get_shard("model.embed.weight"),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(
            index.get_shard("model.layers.1.weight"),
            Some("model-00002-of-00002.safetensors")
        );
        assert_eq!(index.get_shard("nonexistent"), None);
    }

    #[test]
    fn test_hf_model_config_parse() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "torch_dtype": "float16",
            "tie_word_embeddings": false
        }"#;

        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(config.model_type, Some("llama".to_string()));
        assert_eq!(config.hidden_size, Some(4096));
        assert_eq!(config.num_attention_heads, Some(32));
        assert_eq!(config.num_key_value_heads, Some(8));
        assert_eq!(config.num_hidden_layers, Some(32));
        assert_eq!(config.intermediate_size, Some(11008));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.head_dim(), Some(128));
        assert_eq!(config.num_kv_heads(), Some(8));
        assert!((config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_attention_dims_resolve_from_nested_text_config() {
        // Composite configs (gemma4 and friends) keep the attention dims
        // only under `text_config`, with NOTHING at top level. The KV-cache
        // sizer (`compute_num_blocks`) MUST get real `head_dim`/`num_kv_heads`
        // here — otherwise `bytes_per_block` is 0, it falls back to 1024
        // blocks, and the over-sized KV pool OOMs the GPU (gemma-4-31B).
        let config_json = r#"{
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": 5376,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "head_dim": 256,
                "num_hidden_layers": 60
            }
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.head_dim, None, "top-level head_dim absent");
        assert_eq!(config.num_key_value_heads, None);
        // Resolved from the nested text_config:
        assert_eq!(config.head_dim(), Some(256));
        assert_eq!(config.num_kv_heads(), Some(16));
    }

    #[test]
    fn test_hf_model_config_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_json = r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_attention_heads": 32
        }"#;
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();

        let config = HfModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, Some("mistral".to_string()));
    }

    #[test]
    fn test_hf_model_config_defaults() {
        let config: HfModelConfig = serde_json::from_str("{}").unwrap();
        assert!(config.architectures.is_empty());
        assert_eq!(config.model_type, None);
        assert_eq!(config.hidden_size, None);
        assert_eq!(config.head_dim(), None);
        assert_eq!(config.num_kv_heads(), None);
        assert!((config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_hf_model_config_null_architectures() {
        let config: HfModelConfig =
            serde_json::from_str(r#"{"architectures": null, "hidden_size": 2560}"#).unwrap();
        assert!(config.architectures.is_empty());
        assert_eq!(config.hidden_size, Some(2560));
    }

    #[test]
    fn test_resolve_text_config_kimi_k25() {
        let config_json = r#"{
            "architectures": ["KimiK25ForCausalLM"],
            "model_type": "kimi_k25",
            "text_config": {
                "model_type": "deepseek_v2",
                "hidden_size": 7168,
                "num_attention_heads": 128,
                "num_hidden_layers": 61,
                "vocab_size": 129280,
                "rms_norm_eps": 1e-6
            },
            "vision_config": {
                "image_size": 384
            }
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        let result = config.resolve_text_config();
        assert!(result.is_some());

        let (text_cfg, prefix) = result.unwrap();
        assert_eq!(prefix, "language_model.");
        assert_eq!(text_cfg.model_type, Some("deepseek_v2".to_string()));
        assert_eq!(text_cfg.hidden_size, Some(7168));
        assert_eq!(text_cfg.num_hidden_layers, Some(61));
        assert_eq!(text_cfg.architectures, vec!["KimiK25ForCausalLM"]);
    }

    #[test]
    fn test_resolve_text_config_non_composite() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert!(config.resolve_text_config().is_none());
    }

    #[test]
    fn test_hf_model_config_extra_fields() {
        let config_json = r#"{
            "model_type": "qwen2",
            "sliding_window": 4096,
            "use_cache": true
        }"#;
        let config: HfModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.model_type, Some("qwen2".to_string()));
        assert!(config.extra.contains_key("sliding_window"));
        assert_eq!(config.extra["sliding_window"], 4096);
    }

    #[test]
    fn test_available_space_queries_real_fs() {
        let dir = tempfile::tempdir().unwrap();
        let got = available_space(dir.path()).expect("a real dir has queryable free space");

        // On macOS/APFS the purgeable-aware "important usage" figure is always
        // >= the statvfs figure `fs2` reports (it counts reclaimable space on
        // top); on Linux the two are identical. It must never *under*-report,
        // which is exactly the bug that failed downloads that would succeed.
        let statvfs = fs2::available_space(dir.path()).unwrap();
        assert!(
            got >= statvfs,
            "available_space {got} under-reported vs statvfs {statvfs}"
        );
    }
}
