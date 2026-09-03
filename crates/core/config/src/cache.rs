// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! KV cache configuration types, ported from `vllm/config/cache.py` and
//! `vllm/v1/kv_cache_interface.py`.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CacheConfig  (from vllm/config/cache.py)
// ---------------------------------------------------------------------------

/// Data type used for KV cache storage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheDType {
    #[default]
    Auto,
    #[serde(rename = "bfloat16")]
    BFloat16,
    Fp8,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4m3,
    #[serde(rename = "fp8_e5m2")]
    Fp8E5m2,
    #[serde(rename = "fp8_inc")]
    Fp8Inc,
    #[serde(rename = "fp8_ds_mla")]
    Fp8DsMla,
}

/// Mamba cache mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MambaCacheMode {
    /// No Mamba state caching (prefix caching disabled).
    #[default]
    None,
    /// Cache the Mamba state at every `i * block_size` position.
    All,
    /// Only cache when the token is at position `i * block_size` AND it is the
    /// last token of a scheduler step.
    Align,
}

/// Hash algorithm used for prefix caching.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefixCachingHashAlgo {
    #[default]
    Sha256,
    #[serde(rename = "sha256_cbor")]
    Sha256Cbor,
    Xxhash,
    #[serde(rename = "xxhash_cbor")]
    XxhashCbor,
}

/// Configuration for the KV cache.
///
/// Ported from `vllm.config.cache.CacheConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Size of a contiguous cache block in number of tokens.
    /// On CUDA devices, only block sizes up to 32 are supported.
    /// If `None`, will be set by the platform at runtime.
    #[serde(default)]
    pub block_size: Option<usize>,

    /// Fraction of GPU memory to use for the model executor (0, 1].
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// CPU swap space per GPU in GiB.
    #[serde(default = "default_swap_space")]
    pub swap_space: f64,

    /// Data type for KV cache storage.
    #[serde(default)]
    pub cache_dtype: CacheDType,

    /// Whether the model is attention-free (no KV cache needed).
    #[serde(default)]
    pub is_attention_free: bool,

    /// Override for the number of GPU blocks (for testing).
    #[serde(default)]
    pub num_gpu_blocks_override: Option<usize>,

    /// Sliding window size (set from ModelConfig).
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Whether prefix caching is enabled.
    #[serde(default = "bool_true")]
    pub enable_prefix_caching: bool,

    /// Hash algorithm for prefix caching.
    #[serde(default)]
    pub prefix_caching_hash_algo: PrefixCachingHashAlgo,

    /// CPU offload space per GPU in GiB. Deprecated -- use OffloadConfig.
    #[serde(default)]
    pub cpu_offload_gb: f64,

    /// Whether to dynamically calculate k/v scales for fp8 KV cache.
    #[serde(default)]
    pub calculate_kv_scales: bool,

    /// CPU KV cache space in bytes (CPU backend only).
    #[serde(default)]
    pub cpu_kvcache_space_bytes: Option<usize>,

    /// Optional override for Mamba page size.
    #[serde(default)]
    pub mamba_page_size_padded: Option<usize>,

    /// Size of a contiguous Mamba cache block (must be multiple of 8).
    #[serde(default)]
    pub mamba_block_size: Option<usize>,

    /// Data type for the Mamba cache (conv + ssm state).
    #[serde(default = "default_mamba_dtype")]
    pub mamba_cache_dtype: Cow<'static, str>,

    /// Data type for the Mamba SSM state only.
    #[serde(default = "default_mamba_dtype")]
    pub mamba_ssm_cache_dtype: Cow<'static, str>,

    /// Cache strategy for Mamba layers.
    #[serde(default)]
    pub mamba_cache_mode: MambaCacheMode,

    // -- Post-profiling fields (set at runtime) --
    /// Number of GPU blocks allocated after profiling.
    #[serde(default)]
    pub num_gpu_blocks: Option<usize>,

    /// Number of CPU blocks allocated after profiling.
    #[serde(default)]
    pub num_cpu_blocks: Option<usize>,

    /// Enable fast prefill optimization for KV sharing setups.
    #[serde(default)]
    pub kv_sharing_fast_prefill: bool,

    /// Size of KV cache per GPU in bytes. `None` means auto-detect from
    /// `gpu_memory_utilization`.
    #[serde(default)]
    pub kv_cache_memory_bytes: Option<usize>,

    /// KV offloading buffer size in GiB. `None` means no offloading.
    #[serde(default)]
    pub kv_offloading_size: Option<f64>,

    /// Backend for KV cache offloading.
    #[serde(default = "default_kv_offloading_backend")]
    pub kv_offloading_backend: Cow<'static, str>,
}

fn default_gpu_memory_utilization() -> f64 {
    0.9
}

fn default_swap_space() -> f64 {
    4.0
}

fn bool_true() -> bool {
    true
}

fn default_mamba_dtype() -> Cow<'static, str> {
    Cow::Borrowed("auto")
}

fn default_kv_offloading_backend() -> Cow<'static, str> {
    Cow::Borrowed("native")
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: None,
            gpu_memory_utilization: 0.9,
            swap_space: 4.0,
            cache_dtype: CacheDType::default(),
            is_attention_free: false,
            num_gpu_blocks_override: None,
            sliding_window: None,
            enable_prefix_caching: true,
            prefix_caching_hash_algo: PrefixCachingHashAlgo::default(),
            cpu_offload_gb: 0.0,
            calculate_kv_scales: false,
            cpu_kvcache_space_bytes: None,
            mamba_page_size_padded: None,
            mamba_block_size: None,
            mamba_cache_dtype: Cow::Borrowed("auto"),
            mamba_ssm_cache_dtype: Cow::Borrowed("auto"),
            mamba_cache_mode: MambaCacheMode::default(),
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            kv_sharing_fast_prefill: false,
            kv_cache_memory_bytes: None,
            kv_offloading_size: None,
            kv_offloading_backend: Cow::Borrowed("native"),
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache interface types  (from vllm/v1/kv_cache_interface.py)
// ---------------------------------------------------------------------------

/// The type of a KV cache spec, describing the attention pattern for a group
/// of layers.
///
/// This is a simplified Rust enum that captures the variants from the Python
/// class hierarchy rooted at `KVCacheSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KVCacheSpecType {
    /// Standard full (causal) attention.
    FullAttention {
        num_kv_heads: usize,
        head_size: usize,
        /// Head size for values (may differ from key head size in GQA-V).
        #[serde(default)]
        head_size_v: Option<usize>,
        /// Optional sliding window applied on top of full attention.
        #[serde(default)]
        sliding_window: Option<usize>,
        /// Optional attention chunk size.
        #[serde(default)]
        attention_chunk_size: Option<usize>,
    },
    /// Multi-Latent Attention (DeepSeek-style).
    MLA {
        num_kv_heads: usize,
        head_size: usize,
        #[serde(default)]
        cache_dtype_str: Option<String>,
    },
    /// Sliding window attention with a fixed window.
    SlidingWindow {
        num_kv_heads: usize,
        head_size: usize,
        sliding_window: usize,
    },
    /// Chunked local attention with a fixed chunk size.
    ChunkedLocalAttention {
        num_kv_heads: usize,
        head_size: usize,
        attention_chunk_size: usize,
    },
    /// Cross-attention for encoder-decoder models.
    CrossAttention {
        num_kv_heads: usize,
        head_size: usize,
    },
    /// Encoder-only attention (no KV cache needed at inference).
    EncoderOnlyAttention {
        num_kv_heads: usize,
        head_size: usize,
    },
    /// Mamba (state-space model) cache.
    Mamba {
        /// Shapes of the Mamba state tensors.
        shapes: Vec<Vec<usize>>,
        #[serde(default = "default_mamba_type")]
        mamba_type: Cow<'static, str>,
        #[serde(default)]
        mamba_cache_mode: MambaCacheMode,
    },
}

fn default_mamba_type() -> Cow<'static, str> {
    Cow::Borrowed("mamba2")
}

/// A single KV cache spec for a group of layers.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheSpec` and its subclasses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheSpec {
    /// Number of tokens in a single cache block.
    pub block_size: usize,

    /// The specific attention type and associated parameters.
    #[serde(flatten)]
    pub spec_type: KVCacheSpecType,
}

/// A group of model layers that share the same KV cache block table.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheGroupSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheGroupSpec {
    /// The names of model layers in this group.
    pub layer_names: Vec<String>,

    /// The KV cache spec shared by all layers in this group.
    pub kv_cache_spec: KVCacheSpec,
}

/// Describes how a single KV cache tensor should be initialised by workers.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheTensor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheTensor {
    /// Size of the KV cache tensor in bytes.
    pub size: usize,

    /// Layer names that share this tensor.
    pub shared_by: Vec<String>,
}

/// The complete KV cache configuration of a model.
///
/// Ported from `vllm.v1.kv_cache_interface.KVCacheConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// The number of KV cache blocks.
    pub num_blocks: usize,

    /// How the model runner should initialise the KV cache tensors.
    pub kv_cache_tensors: Vec<KVCacheTensor>,

    /// The KV cache groups of the model.  For models with only one type of
    /// attention there is a single group containing all layers.  Hybrid models
    /// will have multiple groups.
    pub kv_cache_groups: Vec<KVCacheGroupSpec>,
}

/// Per-layer attention geometry the hybrid grouper needs. One per model layer,
/// in layer order. The minimal subset of `AttentionSpec` for grouping.
#[derive(Debug, Clone, Copy)]
pub struct LayerKvGeometry {
    /// `true` for a sliding-window (local) layer, `false` for full/global.
    pub is_sliding: bool,
    /// KV heads for this layer's attention.
    pub num_kv_heads: usize,
    /// Head dim for this layer's attention (keys).
    pub head_size: usize,
    /// Head dim for values; `None` == `head_size` (the symmetric case). vLLM's
    /// `FullAttentionSpec.real_page_size_bytes` uses `(head_size + head_size_v)`,
    /// so an asymmetric K/V model (GQA-V) would diverge if we assumed `2*head_size`.
    pub head_size_v: Option<usize>,
    /// Sliding window (tokens); `Some` iff `is_sliding`.
    pub sliding_window: Option<usize>,
}

/// The physical hybrid KV layout — vLLM's `get_kv_cache_groups` result, plus
/// the per-layer placement the metal pool/binding needs.
///
/// This is the keystone of "do what vLLM does": layers are split into groups of
/// equal size, the smaller-page group's `block_size` is scaled up so all groups
/// share ONE page size, and the worker allocates `group_size` PHYSICAL TENSORS
/// (each `shared_by` one layer from every group). A block ID indexes the same
/// physical offset in whichever tensor its layer maps to; the sliding groups
/// free out-of-window blocks back to the one shared pool. See
/// `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`.
#[derive(Debug, Clone)]
pub struct HybridKvLayout {
    /// vLLM-shaped config: shared `num_blocks`, `group_size` tensors, groups.
    pub config: KVCacheConfig,
    /// Number of layers per group == number of physical tensors == max layers
    /// in any group.
    pub group_size: usize,
    /// Unified per-block bytes (K+V) every tensor uses.
    pub page_size_bytes: usize,
    /// Per-layer (layer order) tensor position `0..group_size` — which of the
    /// `group_size` shared tensors this layer's KV lives in.
    pub layer_to_tensor: Vec<usize>,
    /// Per-layer (layer order) group id `0..num_groups`.
    pub layer_to_group: Vec<usize>,
    /// Per-layer (layer order) effective `block_size` (sliding layers keep the
    /// base block_size; full layers may be scaled up by page unification).
    pub layer_block_size: Vec<usize>,
}

impl HybridKvLayout {
    /// Number of KV-cache groups (per-group block tables): group 0 = full,
    /// the rest sliding.
    pub fn num_groups(&self) -> usize {
        self.config.kv_cache_groups.len()
    }

    /// Per-group `(is_sliding, window_tokens, block_size)` in group order
    /// (group 0 = full) — the form the scheduler's hybrid allocator
    /// (`SimpleBlockTracker::enable_hybrid`) and `EngineCoreConfig::hybrid_kv`
    /// consume.
    pub fn engine_groups(&self) -> Vec<(bool, usize, usize)> {
        self.config
            .kv_cache_groups
            .iter()
            .map(|g| {
                let block_size = g.kv_cache_spec.block_size;
                match &g.kv_cache_spec.spec_type {
                    KVCacheSpecType::SlidingWindow { sliding_window, .. } => {
                        (true, *sliding_window, block_size)
                    }
                    KVCacheSpecType::FullAttention {
                        sliding_window: Some(w),
                        ..
                    } => (true, *w, block_size),
                    _ => (false, 0, block_size),
                }
            })
            .collect()
    }

    /// `layer_to_group` as `u32` (for the metal pool / runtime bindings).
    pub fn layer_to_group_u32(&self) -> Vec<u32> {
        self.layer_to_group.iter().map(|&g| g as u32).collect()
    }

    /// The full (group-0) block size — the page-unified `GLOBAL_BLOCK_SIZE`
    /// the worker must use to encode the full group's slot_mapping.
    pub fn full_block_size(&self) -> usize {
        self.config
            .kv_cache_groups
            .first()
            .map(|g| g.kv_cache_spec.block_size)
            .unwrap_or(0)
    }
}

/// `ceil(a / b)`.
const fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Compute vLLM's hybrid KV-cache layout for a model with mixed full + sliding
/// attention layers (gemma4). Faithful to
/// `vllm/v1/core/kv_cache_utils.py`: page-size unification + uniform-page-size
/// grouping + `num_blocks = available // page_size // group_size`.
///
/// `layers` is one geometry per model layer, in layer order. `base_block_size`
/// is the model's block_size (16). `available_bytes` is the KV memory budget;
/// `elem_bytes` the cache dtype size (2 for bf16). Returns `None` if the model
/// is NOT a full+sliding hybrid (caller should fall back to the uniform pool).
pub fn compute_hybrid_kv_layout(
    layers: &[LayerKvGeometry],
    base_block_size: usize,
    available_bytes: usize,
    elem_bytes: usize,
) -> Option<HybridKvLayout> {
    if layers.is_empty() {
        return None;
    }
    let has_full = layers.iter().any(|l| !l.is_sliding);
    let has_sliding = layers.iter().any(|l| l.is_sliding);
    if !(has_full && has_sliding) {
        // Uniform (all full or all sliding) — not a hybrid; uniform pool path.
        return None;
    }

    // Per-block bytes (K+V) for a layer at a given block_size. vLLM:
    // `block_size * num_kv_heads * (head_size + head_size_v) * dtype` — equals
    // `2 * ... * head_size` in the symmetric K/V case.
    let page_of = |l: &LayerKvGeometry, bs: usize| {
        let hv = l.head_size_v.unwrap_or(l.head_size);
        bs * l.num_kv_heads * (l.head_size + hv) * elem_bytes
    };
    let base_page = |l: &LayerKvGeometry| page_of(l, base_block_size);

    // 1. Page-size unification: scale each layer's block_size up so every
    //    layer's page == max_page (vLLM `unify_kv_cache_spec_page_size`).
    let max_page = layers.iter().map(base_page).max().unwrap_or(0);
    if max_page == 0 {
        return None;
    }
    let mut layer_block_size = Vec::with_capacity(layers.len());
    for l in layers {
        let p = base_page(l);
        if max_page % p != 0 {
            // Non-divisible page sizes can't share a tensor — bail to uniform.
            return None;
        }
        let ratio = max_page / p;
        layer_block_size.push(base_block_size * ratio);
    }
    let page_size_bytes = max_page;

    // 2. Bucket layers by (is_sliding, block_size) — vLLM groups by spec
    //    identity. After unification full and sliding differ in block_size, so
    //    they remain distinct buckets.
    let mut full_layers: Vec<usize> = Vec::new();
    let mut sliding_layers: Vec<usize> = Vec::new();
    for (i, l) in layers.iter().enumerate() {
        if l.is_sliding {
            sliding_layers.push(i);
        } else {
            full_layers.push(i);
        }
    }

    // 3. group_size: min bucket size, or max if the buckets are within 1.25x
    //    (vLLM's "avoid excessive padding" rule).
    let min_n = full_layers.len().min(sliding_layers.len());
    let max_n = full_layers.len().max(sliding_layers.len());
    let group_size = if (max_n as f64) < (min_n as f64) * 1.25 {
        max_n
    } else {
        min_n
    };
    if group_size == 0 {
        return None;
    }

    // 4. Split each bucket into cdiv(len, group_size) interleaved groups
    //    (vLLM `layers[i::num_groups]`). Returns the group's layer indices.
    let make_groups = |bucket: &[usize]| -> Vec<Vec<usize>> {
        let num_groups = cdiv(bucket.len(), group_size);
        (0..num_groups)
            .map(|g| bucket.iter().skip(g).step_by(num_groups).copied().collect())
            .collect()
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    groups.extend(make_groups(&full_layers));
    groups.extend(make_groups(&sliding_layers));

    // 5. Per-layer tensor position (i within its group) + group id.
    let mut layer_to_tensor = vec![0usize; layers.len()];
    let mut layer_to_group = vec![0usize; layers.len()];
    for (gid, g) in groups.iter().enumerate() {
        for (pos, &layer) in g.iter().enumerate() {
            layer_to_tensor[layer] = pos;
            layer_to_group[layer] = gid;
        }
    }

    // 6. num_blocks = available // page_size // group_size (vLLM `get_num_blocks`).
    let num_blocks = available_bytes / page_size_bytes / group_size;

    // 7. KVCacheTensors: one per position i, shared_by the i-th layer of every
    //    group (the layers that share physical tensor i).
    let mut kv_cache_tensors = Vec::with_capacity(group_size);
    for i in 0..group_size {
        let shared_by: Vec<String> = groups
            .iter()
            .filter_map(|g| g.get(i))
            .map(|&layer| format!("layer.{layer}"))
            .collect();
        kv_cache_tensors.push(KVCacheTensor {
            size: page_size_bytes * num_blocks,
            shared_by,
        });
    }

    // 8. KVCacheGroupSpecs.
    let kv_cache_groups: Vec<KVCacheGroupSpec> = groups
        .iter()
        .map(|g| {
            let layer0 = g[0];
            let l = &layers[layer0];
            let spec_type = if l.is_sliding {
                KVCacheSpecType::SlidingWindow {
                    num_kv_heads: l.num_kv_heads,
                    head_size: l.head_size,
                    sliding_window: l.sliding_window.unwrap_or(0),
                }
            } else {
                KVCacheSpecType::FullAttention {
                    num_kv_heads: l.num_kv_heads,
                    head_size: l.head_size,
                    head_size_v: None,
                    sliding_window: None,
                    attention_chunk_size: None,
                }
            };
            KVCacheGroupSpec {
                layer_names: g.iter().map(|&layer| format!("layer.{layer}")).collect(),
                kv_cache_spec: KVCacheSpec {
                    block_size: layer_block_size[layer0],
                    spec_type,
                },
            }
        })
        .collect();

    Some(HybridKvLayout {
        config: KVCacheConfig {
            num_blocks,
            kv_cache_tensors,
            kv_cache_groups,
        },
        group_size,
        page_size_bytes,
        layer_to_tensor,
        layer_to_group,
        layer_block_size,
    })
}

#[cfg(test)]
mod hybrid_layout_tests {
    use super::*;

    /// gemma-4-26b: 30 layers, full at idx 5,11,17,23,29 (head_dim 512 / 2 kv);
    /// the other 25 sliding (head_dim 256 / 8 kv, window 1024).
    fn gemma4_layers() -> Vec<LayerKvGeometry> {
        (0..30)
            .map(|i| {
                let full = i % 6 == 5;
                LayerKvGeometry {
                    is_sliding: !full,
                    num_kv_heads: if full { 2 } else { 8 },
                    head_size: if full { 512 } else { 256 },
                    head_size_v: None,
                    sliding_window: if full { None } else { Some(1024) },
                }
            })
            .collect()
    }

    #[test]
    fn test_gemma4_grouping_matches_vllm() {
        let layers = gemma4_layers();
        // 5 GiB budget, bf16.
        let layout = compute_hybrid_kv_layout(&layers, 16, 5 << 30, 2).expect("hybrid");
        // full page = 2*16*2*512*2 = 65536; sliding = 2*16*8*256*2 = 131072.
        // Unified page = 131072; full block_size scaled x2 -> 32, sliding 16.
        assert_eq!(layout.page_size_bytes, 131072);
        for (i, &bs) in layout.layer_block_size.iter().enumerate() {
            if i % 6 == 5 {
                assert_eq!(bs, 32, "full layer {i} block_size scaled x2");
            } else {
                assert_eq!(bs, 16, "sliding layer {i} block_size unchanged");
            }
        }
        // 5 full + 25 sliding -> group_size = min(5,25) = 5; 1 full + 5 sliding
        // groups = 6 groups of 5 -> 5 shared tensors.
        assert_eq!(layout.group_size, 5);
        assert_eq!(layout.config.kv_cache_groups.len(), 6);
        assert_eq!(layout.config.kv_cache_tensors.len(), 5);
        // num_blocks = 5GiB // 131072 // 5 = 8192 -> supports >> 32k.
        assert_eq!(layout.config.num_blocks, 8192);
        // Each shared tensor is shared by one layer from each of the 6 groups.
        for t in &layout.config.kv_cache_tensors {
            assert_eq!(t.shared_by.len(), 6, "tensor shared by 1 layer per group");
            assert_eq!(t.size, 131072 * 8192);
        }
        // Every layer maps to a tensor position < group_size and a valid group.
        for layer in 0..30 {
            assert!(layout.layer_to_tensor[layer] < 5);
            assert!(layout.layer_to_group[layer] < 6);
        }
        // Layers sharing a tensor must be exactly the per-position layers.
        let mut by_tensor: Vec<Vec<usize>> = vec![Vec::new(); 5];
        for layer in 0..30 {
            by_tensor[layout.layer_to_tensor[layer]].push(layer);
        }
        for t in &by_tensor {
            assert_eq!(t.len(), 6, "each physical tensor holds 6 layers");
        }
    }

    #[test]
    fn test_uniform_model_returns_none() {
        // All full attention -> not a hybrid -> None (uniform pool path).
        let layers: Vec<LayerKvGeometry> = (0..8)
            .map(|_| LayerKvGeometry {
                is_sliding: false,
                num_kv_heads: 8,
                head_size: 128,
                head_size_v: None,
                sliding_window: None,
            })
            .collect();
        assert!(compute_hybrid_kv_layout(&layers, 16, 5 << 30, 2).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cache_config() {
        let cfg = CacheConfig::default();
        assert!(cfg.enable_prefix_caching);
        assert_eq!(cfg.gpu_memory_utilization, 0.9);
        assert_eq!(cfg.mamba_cache_mode, MambaCacheMode::None);
        assert!(cfg.num_gpu_blocks.is_none());
    }

    #[test]
    fn test_cache_config_roundtrip() {
        let cfg = CacheConfig {
            block_size: Some(16),
            num_gpu_blocks: Some(1024),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: CacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.block_size, cfg2.block_size);
        assert_eq!(cfg.num_gpu_blocks, cfg2.num_gpu_blocks);
    }

    #[test]
    fn test_kv_cache_spec_full_attention_roundtrip() {
        let spec = KVCacheSpec {
            block_size: 16,
            spec_type: KVCacheSpecType::FullAttention {
                num_kv_heads: 8,
                head_size: 128,
                head_size_v: None,
                sliding_window: None,
                attention_chunk_size: None,
            },
        };
        let json = serde_json::to_string(&spec).unwrap();
        let spec2: KVCacheSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec.block_size, spec2.block_size);
    }

    #[test]
    fn test_kv_cache_config_roundtrip() {
        let kv_cfg = KVCacheConfig {
            num_blocks: 512,
            kv_cache_tensors: vec![KVCacheTensor {
                size: 1024 * 1024,
                shared_by: vec!["layer0".to_string(), "layer1".to_string()],
            }],
            kv_cache_groups: vec![KVCacheGroupSpec {
                layer_names: vec!["layer0".to_string(), "layer1".to_string()],
                kv_cache_spec: KVCacheSpec {
                    block_size: 16,
                    spec_type: KVCacheSpecType::FullAttention {
                        num_kv_heads: 8,
                        head_size: 128,
                        head_size_v: Some(128),
                        sliding_window: None,
                        attention_chunk_size: None,
                    },
                },
            }],
        };
        let json = serde_json::to_string(&kv_cfg).unwrap();
        let kv_cfg2: KVCacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(kv_cfg.num_blocks, kv_cfg2.num_blocks);
        assert_eq!(kv_cfg.kv_cache_groups.len(), kv_cfg2.kv_cache_groups.len());
    }
}
