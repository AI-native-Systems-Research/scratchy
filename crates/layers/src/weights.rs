// SPDX-License-Identifier: Apache-2.0
//! Safetensors weight loading — pipelined from CPU to GPU.
//!
//! Weights are memory-mapped on CPU. On first access, the OS pages data in from
//! disk. We beat Python vLLM's default (serial mmap + synchronous H2D) with:
//!
//! 1. **madvise(WILLNEED)** on every shard at mmap time — OS starts prefetching
//!    all pages from disk immediately, overlapping I/O across shards.
//! 2. **Parallel shard loading** — multi-shard models parse headers concurrently.
//! 3. **Background pre-cast pipeline** — a thread pool pre-faults mmap pages and
//!    casts float tensors (F32→BF16/F16) into pinned host buffers ahead of
//!    `take()` calls. The main thread just enqueues H2D DMAs from ready buffers,
//!    overlapping CPU work with PCIe transfers.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};

use scratchy_tensors::{DType, DeviceAllocator, GpuTensor, PrecastEntry, PrecastPipeline};

// ---------------------------------------------------------------------------
// DType conversion
// ---------------------------------------------------------------------------

/// Map safetensors dtype string to our DType.
fn safetensors_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::I32 => Ok(DType::I32),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::F8_E4M3 => Ok(DType::Fp8E4m3),
        other => bail!("unsupported safetensors dtype: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// CPU dtype conversion helpers (for LoRA merging)
// ---------------------------------------------------------------------------

/// Read raw bytes in `dtype` into a pre-allocated f32 slice.
fn read_to_f32(data: &[u8], dtype: DType, out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, out.len()) };
            out.copy_from_slice(src);
        }
        DType::F16 => {
            let src =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, out.len()) };
            for (s, d) in src.iter().zip(out.iter_mut()) {
                *d = s.to_f32();
            }
        }
        DType::BF16 => {
            let src = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, out.len())
            };
            for (s, d) in src.iter().zip(out.iter_mut()) {
                *d = s.to_f32();
            }
        }
        _ => panic!("read_to_f32: unsupported dtype {dtype}"),
    }
}

/// Write f32 values back to bytes in the given dtype.
fn write_from_f32(data: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => {
            let mut out = vec![0u8; data.len() * 4];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, data.len()) };
            dst.copy_from_slice(data);
            out
        }
        DType::F16 => {
            let mut out = vec![0u8; data.len() * 2];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, data.len()) };
            for (s, d) in data.iter().zip(dst.iter_mut()) {
                *d = half::f16::from_f32(*s).to_bits();
            }
            out
        }
        DType::BF16 => {
            let mut out = vec![0u8; data.len() * 2];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, data.len()) };
            for (s, d) in data.iter().zip(dst.iter_mut()) {
                *d = half::bf16::from_f32(*s).to_bits();
            }
            out
        }
        _ => panic!("write_from_f32: unsupported dtype {dtype}"),
    }
}

// ---------------------------------------------------------------------------
// CpuTensorRef — a reference to tensor data in a mmap'd safetensors file
// ---------------------------------------------------------------------------

/// A CPU-side reference to tensor data — either mmap'd (read-only) or owned
/// (e.g. after LoRA merging).
struct CpuTensorRef {
    /// The mmap that backs this tensor (None for owned data).
    mmap: Option<Arc<memmap2::Mmap>>,
    /// Byte offset within the mmap where tensor data starts.
    data_offset: usize,
    /// Size of tensor data in bytes.
    size_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    /// Owned data buffer (used for merged weights). When set, `data()` returns
    /// this instead of the mmap slice.
    owned: Option<Arc<Vec<u8>>>,
}

impl CpuTensorRef {
    fn data(&self) -> &[u8] {
        if let Some(ref buf) = self.owned {
            buf.as_slice()
        } else {
            let mmap = self
                .mmap
                .as_ref()
                .expect("CpuTensorRef: no mmap or owned data");
            &mmap[self.data_offset..self.data_offset + self.size_bytes]
        }
    }
}

// ---------------------------------------------------------------------------
// GpuWeights
// ---------------------------------------------------------------------------

/// Parse a single shard file into a map of tensor references.
///
/// Mmaps the file, issues madvise(WILLNEED) + madvise(SEQUENTIAL) to trigger
/// OS readahead, and parses the safetensors header. Returns tensor references
/// pointing into the mmap — no data is copied.
///
/// This is a free function (not `&mut self`) so it can be called from parallel
/// threads during multi-shard loading.
fn load_shard_into_map(path: &Path) -> Result<(HashMap<String, CpuTensorRef>, Arc<memmap2::Mmap>)> {
    let _t_total = std::time::Instant::now();
    let _t_open = std::time::Instant::now();
    let file = std::fs::File::open(path)?;
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);
    let _dt_open = _t_open.elapsed();

    // OS readahead prefault (MADV_WILLNEED) is backend-specific and now runs
    // through `DeviceAllocator::prefault_mmap` at the call sites that hold the
    // allocator (cuda hints here; metal prefaults per-shard in register_mmap).

    // Parse safetensors header to find tensor offsets.
    let _t_des = std::time::Instant::now();
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow::anyhow!("{}: {}", path.display(), e))?;
    let _dt_des = _t_des.elapsed();
    let _t_iter = std::time::Instant::now();

    let mut tensors = HashMap::new();
    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("{}: {}", name, e))?;
        let dtype = safetensors_dtype(view.dtype())?;
        let data = view.data();
        let size_bytes = data.len();
        let shape: Vec<usize> = view.shape().to_vec();

        let data_offset = data.as_ptr() as usize - mmap.as_ptr() as usize;

        tensors.insert(
            name.to_string(),
            CpuTensorRef {
                mmap: Some(Arc::clone(&mmap)),
                data_offset,
                size_bytes,
                shape,
                dtype,
                owned: None,
            },
        );
    }

    // VL-wrapper prefix aliasing. MLX-community repacks store the text
    // decoder under `language_model.model.<key>`, while the official VL
    // checkpoints use `model.language_model.<key>` — the same logical
    // tensor in a swapped on-disk ordering. A compiled scratchy variant
    // bakes ONE ordering (from its `decoder_safetensors_prefix`), so
    // without this a checkpoint in the other ordering fails to load
    // (`weight not found: model.language_model.embed_tokens.weight`).
    // Insert an alias under the swapped ordering so a single variant
    // loads either layout. Cheap (clones an mmap-backed ref: Arc + a
    // small shape Vec) and collision-free (no checkpoint ships both
    // orderings of the same tensor; we skip if the target already exists).
    let alias_pairs: Vec<(String, CpuTensorRef)> = tensors
        .iter()
        .filter_map(|(name, t)| {
            let alt = name
                .strip_prefix("language_model.model.")
                .map(|rest| format!("model.language_model.{rest}"))
                .or_else(|| {
                    name.strip_prefix("model.language_model.")
                        .map(|rest| format!("language_model.model.{rest}"))
                })
                // VL-tower prefix aliasing. Qwen3.5-VL ships under
                // `model.visual.*` on the official HF repo (`Qwen/Qwen3.5-9B`),
                // but the qwen3-5-vl scratchy variant config bakes
                // `vision_tower.*` (matching the mlx-community 4bit repack
                // the metal green_gate test loads from). A compiled variant
                // bakes ONE ordering, so without this alias `try_load_mm`'s
                // fingerprint key `vision_tower.merger.linear_fc2.weight` is
                // absent on the HF repo and `try_load_mm` returns Ok(None).
                // Same shape as the `language_model` swap above; cheap
                // (Arc-clones an mmap-backed ref) and collision-free.
                .or_else(|| {
                    name.strip_prefix("model.visual.")
                        .map(|rest| format!("vision_tower.{rest}"))
                })
                .or_else(|| {
                    name.strip_prefix("vision_tower.")
                        .map(|rest| format!("model.visual.{rest}"))
                })
                // lm_head prefix aliasing. Compiled variants always bake the
                // root `lm_head.*` key (codegen's safetensors_prefix keeps
                // lm_head outside the decoder prefix), and official VL repos
                // store it there — but mlx-community repacks nest it under
                // `language_model.lm_head.*` (e.g. Qwen3.5-9B-4bit /
                // Qwen3.5-35B-A3B-4bit). Alias it back to the root.
                .or_else(|| {
                    name.strip_prefix("language_model.lm_head.")
                        .map(|rest| format!("lm_head.{rest}"))
                })?;
            if tensors.contains_key(&alt) {
                return None;
            }
            Some((
                alt,
                CpuTensorRef {
                    mmap: t.mmap.clone(),
                    data_offset: t.data_offset,
                    size_bytes: t.size_bytes,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                    owned: t.owned.clone(),
                },
            ))
        })
        .collect();
    for (alt, t) in alias_pairs {
        tensors.insert(alt, t);
    }

    tracing::info!(
        "Parsed shard {}: {} tensors in {:?} (open+mmap {:?}, deserialize {:?}, iterate {:?})",
        path.display(),
        tensors.len(),
        _t_total.elapsed(),
        _dt_open,
        _dt_des,
        _t_iter.elapsed(),
    );

    Ok((tensors, mmap))
}

/// Locate a dim-0 shard inside a pre-staged entry's device bytes.
/// Returns `(src_dev_ptr, shard_shape, shard_bytes)`, or `None` when
/// the shard isn't a contiguous range of the entry (dim != 0, scalar
/// shape, or a non-divisible leading dim) and the caller must use
/// the gather slow path.
fn shard_of_entry(
    entry: &PrecastEntry,
    shape: &[usize],
    dim: usize,
    rank: usize,
    world_size: usize,
) -> Option<(*const u8, Vec<usize>, usize)> {
    if dim != 0 || shape.is_empty() || world_size == 0 || !shape[0].is_multiple_of(world_size) {
        return None;
    }
    let elem = entry.dtype.size_bytes();
    let row_elems: usize = shape[1..].iter().product();
    let shard_size = shape[0] / world_size;
    let shard_bytes = shard_size * row_elems * elem;
    // Sanity: the staged bytes must cover exactly world_size shards.
    if shard_bytes * world_size != entry.size_bytes || rank >= world_size {
        return None;
    }
    let mut shard_shape = shape.to_vec();
    shard_shape[0] = shard_size;
    let src = unsafe { entry.gpu.ptr().add(rank * shard_bytes) as *const u8 };
    Some((src, shard_shape, shard_bytes))
}

// `PrecastEntry` (the staged device buffer) is the neutral
// `scratchy_tensors::PrecastEntry` — its `gpu: RawGpuMem` is backend-neutral, so
// the pipeline can hand entries back to a `GpuWeights` that names no cuda type.

/// Where the bytes for a device upload come from, returned by
/// [`GpuWeights::take_upload_src`] / [`GpuWeights::take_shard_upload_src`] so a
/// backend extension trait can issue the D2D/H2D copy without touching
/// `GpuWeights`' private loader state (mmap map, cast scratch, precast).
pub enum UploadSrc {
    /// A whole tensor already device-resident (precast staged). The neutral
    /// [`PrecastEntry`] owns the device buffer until the caller drops it.
    Device(PrecastEntry),
    /// A contiguous device sub-range of a staged entry (a dim-0 TP shard).
    /// `src` points into `_keep`'s buffer — hold `_keep` until the copy ends.
    DeviceSlice {
        src: *const u8,
        bytes: usize,
        _keep: PrecastEntry,
    },
    /// Host bytes to H2D. `from_scratch` = the bytes live in the shared cast
    /// scratch (sync the stream before the next take to avoid an overwrite).
    Host {
        ptr: *const u8,
        bytes: usize,
        dtype: DType,
        shape: Vec<usize>,
        from_scratch: bool,
    },
}

/// Model weights loaded from CPU (mmap) to GPU with pipelined pre-casting.
///
/// Weights are memory-mapped on CPU with madvise(WILLNEED) to trigger OS
/// readahead. When `start_precast()` is called, a background thread pool
/// pre-faults mmap pages and casts float tensors into pinned host buffers.
/// `take()` checks for pre-cast data first — if ready, it just enqueues an
/// async H2D DMA without blocking on page faults or CPU casting.
///
/// GPU memory allocated by `take()` is NOT freed on drop — ownership transfers
/// to the caller (model layers).
pub struct GpuWeights<A: DeviceAllocator> {
    /// Per-tensor CPU references, keyed by tensor name.
    tensors: HashMap<String, CpuTensorRef>,
    /// Target dtype for floating-point weights. When set, F32 weights are cast
    /// to this dtype on CPU before H2D copy.
    target_dtype: Option<DType>,
    /// Dtype the RMSNorm gain MUST be stored as so it matches the kernel's
    /// `T_scale` pointer (metal binds the gain through `device const T_scale*`
    /// and picks `T_scale` from the compile-time `W::SCALE_DTYPE`, not the
    /// on-disk dtype). Set once per model from `W::SCALE_DTYPE`;
    /// [`take_as_dtype`](Self::take_as_dtype) honors it, converting only when
    /// the on-disk gain dtype differs (so the mlx zero-copy path is preserved).
    /// `None` leaves the gain at its on-disk dtype (cuda/spyre, which cast the
    /// gain themselves, never set it).
    rmsnorm_scale_dtype: Option<DType>,
    /// Reusable host scratch for the synchronous-cast slow path
    /// (fallback when the precast pipeline hasn't processed a
    /// tensor yet). Pageable; CUDA's `memcpy_htod_async` from
    /// pageable memory blocks the CPU but correctness is fine, and
    /// the slow path is rare (precast handles the hot path).
    /// Grows as needed, never shrinks.
    cast_scratch: Vec<u8>,
    /// Directory the safetensors shards were loaded from. Feeds the
    /// QD-sidecar key (the filesystem device backing the weights);
    /// `None` for GGUF/`empty()`-constructed instances.
    source_dir: Option<std::path::PathBuf>,
    /// Pre-stage pipeline, behind the neutral [`PrecastPipeline`] trait so this
    /// struct names no cuda type. CUDA-only optimization (the impl uses pinned
    /// host memory + driver DMAs); `None` under non-CUDA backends, which never
    /// call `start_precast`.
    precast: Option<Arc<dyn PrecastPipeline>>,
    /// Join handles for the background pre-stage workers (empty unless cuda's
    /// `start_precast` spawned them).
    precast_handles: Vec<std::thread::JoinHandle<()>>,
    /// Backend allocator: device memory + H2D primitive. The
    /// concrete type is `CudaAllocator` under cuda or
    /// `MetalAllocator` under metal — see [`BackendAllocator`]
    /// (`crate::BackendAllocator`).
    allocator: A,
    /// Keep mmaps alive for the lifetime of GpuWeights.
    ///
    /// `take()` and `take_into()` use `memcpy_htod_async` which reads from
    /// mmap'd memory asynchronously. The `CpuTensorRef` holding the `Arc<Mmap>`
    /// is dropped at the end of those functions. If that was the last reference,
    /// the mmap would be unmapped while the async DMA is still in flight,
    /// causing silent data corruption on GPU. This field retains all mmaps
    /// until the GpuWeights struct is dropped (after model loading completes).
    _mmaps: Vec<Arc<memmap2::Mmap>>,

    /// Quantized linear weights loaded from a GGUF file. Empty when
    /// the backing store is safetensors. Keyed by HF-style tensor
    /// name (e.g. `model.layers.0.self_attn.q_proj.weight`). The
    /// underlying GPU bytes are deliberately leaked for the model's
    /// lifetime; `take_quantized_linear` is non-destructive (returns
    /// a `GgmlStorage` view, leaves the entry in place) so multiple
    /// accessors can share the same source weight — see the docstring
    /// on that method for the workload-fanout case.
    quantized: HashMap<String, scratchy_quantizations::GgmlStorage>,

    /// Already-on-GPU dense weights from a GGUF file (norms,
    /// embeddings, lm_head — the GGUF loader dequantizes these at
    /// load time). Mirrors `quantized` — both are populated only on
    /// the GGUF path. Safetensors-backed `GpuWeights` populates
    /// `tensors` instead and uploads on `take()`.
    gguf_dense: HashMap<String, GpuTensor>,
}

// Safety: GPU device pointers accessible from any host thread.
unsafe impl<A: DeviceAllocator> Send for GpuWeights<A> {}
unsafe impl<A: DeviceAllocator> Sync for GpuWeights<A> {}

/// Neutral [`WeightSource`](scratchy_tensors::WeightSource) view — forwards to
/// the inherent methods so per-format quant loader types can `load()` against a
/// generic `W: WeightSource` without naming `GpuWeights` (lets them live in a
/// backend crate below the one that owns `GpuWeights`).
impl<A: DeviceAllocator> scratchy_tensors::WeightSource for GpuWeights<A> {
    fn take(&mut self, name: &str) -> Result<GpuTensor> {
        GpuWeights::take(self, name)
    }
    fn take_keep_dtype(&mut self, name: &str) -> Result<GpuTensor> {
        GpuWeights::take_keep_dtype(self, name)
    }
    fn take_cpu(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, DType)> {
        GpuWeights::take_cpu(self, name)
    }
    fn take_to_cpu_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        GpuWeights::take_to_cpu_f32(self, name)
    }
    fn contains(&self, name: &str) -> bool {
        GpuWeights::contains(self, name)
    }
    fn alloc_packed_from_host(
        &mut self,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<GpuTensor> {
        GpuWeights::alloc_packed_from_host(self, data, shape, dtype)
    }
}

impl<A: DeviceAllocator> GpuWeights<A> {
    /// Construct an empty `GpuWeights` — used by the GGUF loader in
    /// `scratchy-target-cuda`, which then populates `quantized` and
    /// `gguf_dense` directly via `quantized_map_mut` /
    /// `gguf_dense_map_mut`. Safetensors callers should use
    /// `from_dir` / `from_index` / `from_single_file` instead.
    pub fn empty(allocator: A) -> Self {
        Self {
            tensors: HashMap::new(),
            target_dtype: None,
            rmsnorm_scale_dtype: None,
            cast_scratch: Vec::new(),
            source_dir: None,
            precast: None,
            precast_handles: Vec::new(),
            allocator,
            _mmaps: Vec::new(),
            quantized: HashMap::new(),
            gguf_dense: HashMap::new(),
        }
    }

    /// Load all weights from a model directory (CPU-only — no GPU allocation).
    ///
    /// Handles both single-file (`model.safetensors`) and sharded
    /// (`model.safetensors.index.json`) models.
    pub fn from_dir(dir: impl AsRef<Path>, allocator: A) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");

        let mut gw = if index_path.exists() {
            Self::from_index(&index_path, allocator)
        } else if single_path.exists() {
            Self::from_single_file(&single_path, allocator)
        } else {
            anyhow::bail!("No safetensors files found in {}", dir.display());
        }?;
        gw.source_dir = Some(dir.to_path_buf());
        Ok(gw)
    }
}

impl<A: DeviceAllocator> GpuWeights<A> {
    /// Load from a single safetensors file (CPU-only).
    pub fn from_single_file(path: impl AsRef<Path>, allocator: A) -> Result<Self> {
        let path = path.as_ref();
        let mut gw = Self {
            tensors: HashMap::new(),
            target_dtype: None,
            rmsnorm_scale_dtype: None,
            cast_scratch: Vec::new(),
            source_dir: None,
            precast: None,
            precast_handles: Vec::new(),
            allocator,
            _mmaps: Vec::new(),
            quantized: HashMap::new(),
            gguf_dense: HashMap::new(),
        };
        gw.load_shard(path)?;
        Ok(gw)
    }

    /// Load from a sharded model (index.json) (CPU-only).
    ///
    /// Multiple shards are loaded in parallel — each thread mmaps a shard,
    /// issues madvise(WILLNEED) to start readahead, and parses the header.
    /// This overlaps disk I/O across shards.
    pub fn from_index(index_path: impl AsRef<Path>, allocator: A) -> Result<Self> {
        let index_path = index_path.as_ref();
        let dir = index_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("index file has no parent dir"))?;

        let index_data = std::fs::read_to_string(index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("missing weight_map in index"))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let total = shard_files.len();

        if total <= 1 {
            // Single shard — no need for threading.
            let mut gw = Self {
                tensors: HashMap::new(),
                target_dtype: None,
                rmsnorm_scale_dtype: None,
                cast_scratch: Vec::new(),
                source_dir: None,
                precast: None,
                precast_handles: Vec::new(),
                allocator,
                _mmaps: Vec::new(),
                quantized: HashMap::new(),
                gguf_dense: HashMap::new(),
            };
            if let Some(name) = shard_files.first() {
                gw.load_shard(&dir.join(name))?;
            }
            return Ok(gw);
        }

        // Multiple shards — load in parallel. Each thread mmaps a shard,
        // triggers madvise(WILLNEED), and parses the header. This overlaps
        // disk I/O and CPU-side parsing across shards.
        tracing::info!("Loading {total} shards in parallel");

        #[allow(clippy::type_complexity)]
        let shard_results: Vec<
            Result<(HashMap<String, CpuTensorRef>, Arc<memmap2::Mmap>)>,
        > = std::thread::scope(|scope| {
            let handles: Vec<_> = shard_files
                .iter()
                .map(|shard_name| {
                    let shard_path = dir.join(shard_name);
                    scope.spawn(move || load_shard_into_map(&shard_path))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut tensors = HashMap::new();
        let mut mmaps = Vec::with_capacity(shard_results.len());
        for (shard_name, result) in shard_files.iter().zip(shard_results) {
            let (shard_tensors, mmap) = result?;
            tensors.extend(shard_tensors);
            // Register every shard's mmap with the allocator so
            // metal `take()` can alias the safetensors pages directly
            // (zero-copy weight load). `register_mmap` dispatches
            // background `pread`s to populate a 16-aligned per-tensor
            // layout; per-tensor join happens at `take()` time inside
            // `alloc_and_copy_host`. Cuda's `register_mmap` is a no-op default;
            // its `prefault_mmap` issues the readahead hint instead.
            allocator.register_mmap(&dir.join(shard_name), Arc::clone(&mmap))?;
            allocator.prefault_mmap(&mmap);
            mmaps.push(mmap);
        }

        Ok(Self {
            tensors,
            target_dtype: None,
            rmsnorm_scale_dtype: None,
            cast_scratch: Vec::new(),
            source_dir: None,
            precast: None,
            precast_handles: Vec::new(),
            allocator,
            _mmaps: mmaps,
            quantized: HashMap::new(),
            gguf_dense: HashMap::new(),
        })
    }

    /// Parse a shard file, madvise(WILLNEED), and store tensor references.
    fn load_shard(&mut self, path: &Path) -> Result<()> {
        let (shard_tensors, mmap) = load_shard_into_map(path)?;
        self.tensors.extend(shard_tensors);
        // Register the mmap with the allocator so metal `take()` can alias the
        // safetensors pages directly (cuda register_mmap is a no-op); cuda's
        // prefault_mmap issues the readahead hint.
        self.allocator.register_mmap(path, Arc::clone(&mmap))?;
        self.allocator.prefault_mmap(&mmap);
        self._mmaps.push(mmap);
        Ok(())
    }

    /// Ensure the pinned cast buffer has at least `needed` bytes.
    /// Grows by freeing + reallocating (pinned memory can't realloc).
    /// If target_dtype is set and the weight needs casting, cast on CPU into
    /// the reusable host scratch buffer. Returns (data_ptr, size_bytes,
    /// effective_dtype).
    ///
    /// Only floating-point weights (F32, BF16, F16) are cast. Integer dtypes
    /// (I32, U32, I64) are left untouched — they're used for indices/metadata.
    fn maybe_cast_cpu(&mut self, cpu_ref: &CpuTensorRef) -> (*const u8, usize, DType) {
        let target = match self.target_dtype {
            Some(t) => t,
            None => return (cpu_ref.data().as_ptr(), cpu_ref.size_bytes, cpu_ref.dtype),
        };

        // Only cast floating-point types.
        let is_float = matches!(cpu_ref.dtype, DType::F32 | DType::F16 | DType::BF16);
        if !is_float || cpu_ref.dtype == target {
            return (cpu_ref.data().as_ptr(), cpu_ref.size_bytes, cpu_ref.dtype);
        }

        let numel = cpu_ref.size_bytes / cpu_ref.dtype.size_bytes();
        let cast_size = numel * target.size_bytes();
        // Vec::resize is grow-only-cheap when capacity already
        // satisfies; the slow path is rare, so an occasional realloc
        // is fine.
        if self.cast_scratch.len() < cast_size {
            self.cast_scratch.resize(cast_size, 0);
        }

        let src = cpu_ref.data();
        let dst = self.cast_scratch.as_mut_ptr();

        // Dispatch cast. The common case is F32 → BF16/F16. Each
        // float-pair path uses (1) the `half` crate's SIMD-accelerated
        // `convert_from_f32_slice` / `convert_to_f32_slice` where
        // available (NEON on aarch64, F16C on x86_64 with the right
        // target features), and (2) rayon to parallelize across cores.
        // The threshold below avoids rayon overhead on tiny tensors.
        use half::slice::HalfFloatSliceExt;
        use rayon::prelude::*;
        const PAR_THRESHOLD: usize = 256 * 1024; // ~256K elements before splitting.

        match (cpu_ref.dtype, target) {
            (DType::F32, DType::BF16) => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
                let dst_bf16 =
                    unsafe { std::slice::from_raw_parts_mut(dst as *mut half::bf16, numel) };
                if numel >= PAR_THRESHOLD {
                    let chunk = numel.div_ceil(rayon::current_num_threads().max(1));
                    src_f32
                        .par_chunks(chunk)
                        .zip(dst_bf16.par_chunks_mut(chunk))
                        .for_each(|(s, d)| d.convert_from_f32_slice(s));
                } else {
                    dst_bf16.convert_from_f32_slice(src_f32);
                }
            }
            (DType::F32, DType::F16) => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
                let dst_f16 =
                    unsafe { std::slice::from_raw_parts_mut(dst as *mut half::f16, numel) };
                if numel >= PAR_THRESHOLD {
                    let chunk = numel.div_ceil(rayon::current_num_threads().max(1));
                    src_f32
                        .par_chunks(chunk)
                        .zip(dst_f16.par_chunks_mut(chunk))
                        .for_each(|(s, d)| d.convert_from_f32_slice(s));
                } else {
                    dst_f16.convert_from_f32_slice(src_f32);
                }
            }
            (DType::F16, DType::BF16) => {
                let src_f16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const half::f16, numel) };
                let dst_bf16 =
                    unsafe { std::slice::from_raw_parts_mut(dst as *mut half::bf16, numel) };
                // Two-step: f16 → f32 (SIMD via convert_to_f32_slice) → bf16.
                src_f16
                    .par_chunks(PAR_THRESHOLD.max(1))
                    .zip(dst_bf16.par_chunks_mut(PAR_THRESHOLD.max(1)))
                    .for_each(|(s, d)| {
                        let mut tmp = vec![0.0_f32; s.len()];
                        s.convert_to_f32_slice(&mut tmp);
                        d.convert_from_f32_slice(&tmp);
                    });
            }
            (DType::BF16, DType::F16) => {
                let src_bf16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const half::bf16, numel) };
                let dst_f16 =
                    unsafe { std::slice::from_raw_parts_mut(dst as *mut half::f16, numel) };
                src_bf16
                    .par_chunks(PAR_THRESHOLD.max(1))
                    .zip(dst_f16.par_chunks_mut(PAR_THRESHOLD.max(1)))
                    .for_each(|(s, d)| {
                        let mut tmp = vec![0.0_f32; s.len()];
                        s.convert_to_f32_slice(&mut tmp);
                        d.convert_from_f32_slice(&tmp);
                    });
            }
            (DType::BF16, DType::F32) => {
                let src_bf16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const half::bf16, numel) };
                let dst_f32 = unsafe { std::slice::from_raw_parts_mut(dst as *mut f32, numel) };
                if numel >= PAR_THRESHOLD {
                    let chunk = numel.div_ceil(rayon::current_num_threads().max(1));
                    src_bf16
                        .par_chunks(chunk)
                        .zip(dst_f32.par_chunks_mut(chunk))
                        .for_each(|(s, d)| s.convert_to_f32_slice(d));
                } else {
                    src_bf16.convert_to_f32_slice(dst_f32);
                }
            }
            (DType::F16, DType::F32) => {
                let src_f16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const half::f16, numel) };
                let dst_f32 = unsafe { std::slice::from_raw_parts_mut(dst as *mut f32, numel) };
                if numel >= PAR_THRESHOLD {
                    let chunk = numel.div_ceil(rayon::current_num_threads().max(1));
                    src_f16
                        .par_chunks(chunk)
                        .zip(dst_f32.par_chunks_mut(chunk))
                        .for_each(|(s, d)| s.convert_to_f32_slice(d));
                } else {
                    src_f16.convert_to_f32_slice(dst_f32);
                }
            }
            _ => unreachable!("unhandled cast: {:?} → {:?}", cpu_ref.dtype, target),
        }

        tracing::debug!(
            "Cast weight: {:?} → {:?} ({} elements)",
            cpu_ref.dtype,
            target,
            numel,
        );

        (dst as *const u8, cast_size, target)
    }

    /// Set the target dtype for floating-point weight casting.
    ///
    /// When set, floating-point weights (F32, F16, BF16) are cast to the target
    /// dtype on CPU before H2D copy. Integer weights are never cast.
    /// This matches Python vLLM where model parameters are initialized with
    /// `torch_dtype` and PyTorch auto-casts during weight loading.
    pub fn set_target_dtype(&mut self, dtype: DType) {
        self.target_dtype = Some(dtype);
    }

    /// Read back the configured target dtype (the same value
    /// [`Self::take_into`] casts floating-point weights to). `None` when no
    /// target is configured — in that case the caller should treat the
    /// on-disk dtype as authoritative. Used by stacked-tensor loaders
    /// (fused MoE expert stacks) that pre-allocate a single buffer and
    /// need to size it against the post-cast element width.
    pub fn target_dtype(&self) -> Option<DType> {
        self.target_dtype
    }

    /// Pin the dtype RMSNorm gains must be stored as to match the metal
    /// `_s_<scale>_` kernel's `T_scale` pointer. Set once per model from
    /// `W::SCALE_DTYPE`. See [`Self::take_as_dtype`] and the
    /// `rmsnorm_scale_dtype` field doc.
    pub fn set_rmsnorm_scale_dtype(&mut self, dtype: scratchy_tensors::ScaleDtype) {
        self.rmsnorm_scale_dtype = Some(dtype.as_dtype());
    }

    /// The pinned RMSNorm gain dtype, if any. Metal's RMSNorm loader uses it
    /// to decide whether the on-disk gain needs converting; `None` (cuda /
    /// spyre, GGUF) means keep the on-disk dtype.
    pub fn rmsnorm_scale_dtype(&self) -> Option<DType> {
        self.rmsnorm_scale_dtype
    }

    // ---- Accessors for backend extension traits ------------------------------
    // GpuWeights itself is backend-neutral; the cuda/metal upload + precast
    // methods live in extension traits in `targets/{cuda,metal}` and reach the
    // private loader state through these.

    /// The backend allocator (device memory + H2D primitive).
    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Mutable backend allocator — for the cuda alloc-tracking passthroughs
    /// (`record_alloc` / `take_gpu_allocs` / …).
    pub fn allocator_mut(&mut self) -> &mut A {
        &mut self.allocator
    }

    /// Model source directory (the per-disk QD-sidecar key for the precast ramp).
    pub fn source_dir(&self) -> Option<&std::path::Path> {
        self.source_dir.as_deref()
    }

    /// Whether a precast pipeline has been installed.
    pub fn precast_started(&self) -> bool {
        self.precast.is_some()
    }

    /// Install the backend-built precast pipeline.
    pub fn set_precast(&mut self, pipeline: Arc<dyn PrecastPipeline>) {
        self.precast = Some(pipeline);
    }

    /// Register a precast worker thread handle (joined on drop).
    pub fn push_precast_handle(&mut self, handle: std::thread::JoinHandle<()>) {
        self.precast_handles.push(handle);
    }

    /// Per-tensor `(mmap, data_offset, size_bytes, dtype)` for every tensor
    /// backed by an mmap — the precast pipeline's work-list source.
    pub fn mmap_tensor_refs(&self) -> Vec<(Arc<memmap2::Mmap>, usize, usize, DType)> {
        self.tensors
            .values()
            .filter_map(|r| {
                r.mmap
                    .as_ref()
                    .map(|m| (m.clone(), r.data_offset, r.size_bytes, r.dtype))
            })
            .collect()
    }

    /// Remove `name` and resolve where its (cast) bytes live for a device
    /// upload — a staged device buffer (precast) or host bytes. The backend
    /// extension issues the actual D2D/H2D copy.
    pub fn take_upload_src(&mut self, name: &str) -> Result<UploadSrc> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        if let Some(entry) = self.take_precast(&cpu_ref) {
            return Ok(UploadSrc::Device(entry));
        }
        let (ptr, bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);
        let from_scratch = std::ptr::eq(ptr, self.cast_scratch.as_ptr());
        Ok(UploadSrc::Host {
            ptr,
            bytes,
            dtype,
            shape: cpu_ref.shape,
            from_scratch,
        })
    }

    /// Like [`Self::take_upload_src`] but for a TP shard (`dim`/`rank`/`world_size`).
    /// A staged entry yields a `DeviceSlice` for a dim-0 contiguous shard;
    /// otherwise the host slow path (strided gather for dim-1) is used.
    pub fn take_shard_upload_src(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<UploadSrc> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        if let Some(entry) = self.take_precast(&cpu_ref)
            && let Some((src, _shard_shape, bytes)) =
                shard_of_entry(&entry, &cpu_ref.shape, dim, rank, world_size)
        {
            return Ok(UploadSrc::DeviceSlice {
                src,
                bytes,
                _keep: entry,
            });
        }
        let (data, shard_shape, dtype) = self.shard_cpu_data(&cpu_ref, dim, rank, world_size);
        let bytes = shard_shape.iter().product::<usize>() * dtype.size_bytes();
        let from_scratch = std::ptr::eq(data, self.cast_scratch.as_ptr());
        Ok(UploadSrc::Host {
            ptr: data,
            bytes,
            dtype,
            shape: shard_shape,
            from_scratch,
        })
    }

    /// Remove a tensor by name and copy it to GPU. Returns a GPU tensor.
    ///
    /// If the pre-cast pipeline has already processed this tensor, the H2D
    /// DMA uses the pre-cast pinned buffer (fast path — no page faults or
    /// CPU casting on the hot path). Otherwise falls back to synchronous
    /// cast from the mmap.
    ///
    /// GGUF backing: if the tensor is in `gguf_dense` (already on GPU,
    /// dequantized at load), this short-circuits and returns it
    /// directly — no upload, no precast.
    pub fn take(&mut self, name: &str) -> Result<GpuTensor> {
        // GGUF fast-path: norms / embeddings / lm_head are
        // pre-uploaded and dequantized by `load_gguf_into_weights`.
        if let Some(t) = self.gguf_dense.remove(name) {
            return Ok(t);
        }

        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        // Fast path: the pre-stage pipeline already uploaded this
        // tensor (blocking per-chunk H2D — no sync needed). Transfer
        // the device buffer into the allocator's lifetime tracker and
        // wrap it. Cuda-only — Metal has no precast.
        if let Some(entry) = self.take_precast(&cpu_ref) {
            let ptr = entry.gpu.ptr();
            let dtype = entry.dtype;
            self.allocator.adopt_raw(entry.gpu);
            return Ok(unsafe { GpuTensor::new(ptr, &cpu_ref.shape, dtype) });
        }

        // Slow path: synchronous pre-fault + cast + DMA.
        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);
        let gpu_ptr = unsafe { self.allocator.alloc_and_copy_host(data, size_bytes)? };
        Ok(unsafe { GpuTensor::new(gpu_ptr, &cpu_ref.shape, dtype) })
    }

    /// Like [`take`] but skips the `target_dtype` cast — bytes go to
    /// the device exactly as they were stored on disk.
    ///
    /// Used by `AffineQuantLinear::load` / `AffineQuantEmbedding::load`
    /// (Metal int4 path) for `*.scales` / `*.biases`: those ship F16
    /// on disk and the scratchy-target-metal int4 kernels read them as `F16`
    /// `T_scale` pointers, casting to the activation dtype in-register.
    /// Routing through `take()` would F16→BF16 truncate the scales at
    /// load on the bf16 stack — that's the P10b regression site.
    ///
    /// Routes through [`alloc_and_copy_host_aligned`] with the
    /// dtype's scalar size as `min_align`. On Metal this unlocks the
    /// mmap-zero-copy fast path for F16/BF16 scales/biases/RMSNorm
    /// gains in `mlx-community` 4bit safetensors (whose data section
    /// lands at file-offset `mod 16 = 2`, permanently failing the
    /// strict 16-byte `MetalAllocator::MIN_BIND_ALIGN` gate). On
    /// CUDA the call is forwarded to `alloc_and_copy_host` — the
    /// alignment hint is a no-op there.
    ///
    /// [`take`]: Self::take
    /// [`alloc_and_copy_host_aligned`]: crate::device_allocator::DeviceAllocator::alloc_and_copy_host_aligned
    pub fn take_keep_dtype(&mut self, name: &str) -> Result<GpuTensor> {
        if let Some(t) = self.gguf_dense.remove(name) {
            return Ok(t);
        }
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        // Consume the pre-stage slot so it's not leaked. When the entry
        // holds a byte-identical staged copy (no cast — entry dtype ==
        // on-disk dtype), hand its device buffer over directly
        // (cuMemAlloc alignment satisfies any scalar `min_align`); a
        // cast entry's bytes are ignored — dropping it frees the
        // staging — and the on-disk view is uploaded instead. (Metal
        // has no precast.)
        if let Some(entry) = self.take_precast(&cpu_ref)
            && entry.dtype == cpu_ref.dtype
        {
            let ptr = entry.gpu.ptr();
            self.allocator.adopt_raw(entry.gpu);
            return Ok(unsafe { GpuTensor::new(ptr, &cpu_ref.shape, cpu_ref.dtype) });
        }
        let gpu_ptr = unsafe {
            self.allocator.alloc_and_copy_host_aligned(
                cpu_ref.data().as_ptr(),
                cpu_ref.size_bytes,
                cpu_ref.dtype.size_bytes(),
            )?
        };
        Ok(unsafe { GpuTensor::new(gpu_ptr, &cpu_ref.shape, cpu_ref.dtype) })
    }

    /// Like [`take_keep_dtype`](Self::take_keep_dtype), but guarantees the
    /// uploaded buffer is in `target` dtype. When the on-disk dtype already
    /// equals `target` this is byte-for-byte identical to `take_keep_dtype`
    /// (the mlx zero-copy / aligned fast path — NO CPU cast); only a genuine
    /// mismatch (e.g. a standard HF `bfloat16` checkpoint feeding a kernel
    /// whose `T_scale` arm is `half`) triggers a one-shot CPU cast.
    ///
    /// Metal's RMSNorm gain load uses this: the `_s_<scale>_` kernel binds the
    /// gain through a `device const T_scale*` chosen at compile time from
    /// `W::SCALE_DTYPE`, so the gain buffer's dtype must match `T_scale`
    /// regardless of what the checkpoint shipped. cuda/spyre don't need it —
    /// they cast the gain at load unconditionally.
    pub fn take_as_dtype(&mut self, name: &str, target: DType) -> Result<GpuTensor> {
        // Inspect the on-disk dtype without consuming the entry. GGUF dense
        // tensors are already device-resident in compute dtype — keep_dtype
        // handles them and the cast (if any) would have happened at GGUF load.
        let on_disk = self.tensors.get(name).map(|r| r.dtype);
        match on_disk {
            // No CPU cast needed — identical to the keep-dtype fast path.
            Some(dt) if dt == target => self.take_keep_dtype(name),
            // Non-float or absent (gguf_dense) — fall back to keep-dtype.
            Some(dt) if !matches!(dt, DType::F32 | DType::F16 | DType::BF16) => {
                self.take_keep_dtype(name)
            }
            None => self.take_keep_dtype(name),
            // Float dtype that doesn't match the kernel's T_scale: cast once.
            Some(_) => {
                let shape = self
                    .tensors
                    .get(name)
                    .map(|r| r.shape.clone())
                    .expect("on_disk was Some");
                let f32_data = self.take_to_cpu_f32(name)?;
                let numel = f32_data.len();
                let bytes: Vec<u8> = match target {
                    DType::F16 => {
                        use half::slice::HalfFloatSliceExt;
                        let mut dst = vec![half::f16::ZERO; numel];
                        dst.convert_from_f32_slice(&f32_data);
                        unsafe {
                            std::slice::from_raw_parts(dst.as_ptr() as *const u8, numel * 2)
                                .to_vec()
                        }
                    }
                    DType::BF16 => {
                        use half::slice::HalfFloatSliceExt;
                        let mut dst = vec![half::bf16::ZERO; numel];
                        dst.convert_from_f32_slice(&f32_data);
                        unsafe {
                            std::slice::from_raw_parts(dst.as_ptr() as *const u8, numel * 2)
                                .to_vec()
                        }
                    }
                    DType::F32 => unsafe {
                        std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, numel * 4)
                            .to_vec()
                    },
                    other => anyhow::bail!("take_as_dtype: unsupported target dtype {other:?}"),
                };
                self.alloc_packed_from_host(&bytes, &shape, target)
            }
        }
    }

    /// Same as [`take`] but creates the returned `GpuTensor` with a
    /// caller-provided shape instead of the on-disk shape. The two
    /// shapes must agree on total element count. Used to flatten
    /// `>MAX_DIMS`-dim tensors at load time — e.g. Qwen2-VL's
    /// `visual.patch_embed.proj.weight` which lands as a 5D Conv3d
    /// weight on disk but the runtime treats it as a 2D GEMM kernel
    /// (stride==kernel collapses the conv).
    ///
    /// [`take`]: Self::take
    pub fn take_with_shape(&mut self, name: &str, shape: &[usize]) -> Result<GpuTensor> {
        if let Some(t) = self.gguf_dense.remove(name) {
            // GGUF tensors don't usually overflow MAX_DIMS, but if a
            // future arch's GGUF spec lands a 5D tensor, the same
            // reshape applies. Same-size invariant.
            let new_numel: usize = shape.iter().product();
            anyhow::ensure!(
                t.numel() == new_numel,
                "take_with_shape: {name} numel mismatch (gguf {} vs new {new_numel})",
                t.numel(),
            );
            return Ok(unsafe { GpuTensor::new(t.raw_ptr(), shape, t.dtype()) });
        }

        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let on_disk_numel: usize = cpu_ref.shape.iter().product();
        let new_numel: usize = shape.iter().product();
        anyhow::ensure!(
            on_disk_numel == new_numel,
            "take_with_shape: {name} on-disk shape {:?} (numel {on_disk_numel}) != new shape {:?} (numel {new_numel})",
            cpu_ref.shape,
            shape,
        );

        if let Some(entry) = self.take_precast(&cpu_ref) {
            let ptr = entry.gpu.ptr();
            let dtype = entry.dtype;
            self.allocator.adopt_raw(entry.gpu);
            return Ok(unsafe { GpuTensor::new(ptr, shape, dtype) });
        }

        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);
        let gpu_ptr = unsafe { self.allocator.alloc_and_copy_host(data, size_bytes)? };
        Ok(unsafe { GpuTensor::new(gpu_ptr, shape, dtype) })
    }

    /// Allocate a single GPU buffer and copy `data` into it via the
    /// active [`DeviceAllocator`]. Returns a fresh `GpuTensor` of the
    /// given `shape` and `dtype` over that buffer.
    ///
    /// Backend-neutral: under cuda this does `mem_alloc` + sync H2D;
    /// under metal it allocates a `StorageModeShared` arena slice and
    /// memcpys into it. Used by stream-free fused loaders (e.g.
    /// `LinearLayer::load_dense_concat_packed`) that pre-concatenate
    /// CPU bytes and need a single packed device buffer.
    pub fn alloc_packed_from_host(
        &mut self,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<GpuTensor> {
        let elem = dtype.size_bytes();
        let expected = shape.iter().product::<usize>() * elem;
        anyhow::ensure!(
            data.len() == expected,
            "alloc_packed_from_host: bytes {} != shape {:?} × {}",
            data.len(),
            shape,
            elem,
        );
        let gpu_ptr = unsafe {
            self.allocator
                .alloc_and_copy_host(data.as_ptr(), data.len())?
        };
        Ok(unsafe { GpuTensor::new(gpu_ptr, shape, dtype) })
    }

    /// Take a tensor and upload it to GPU as **F32**, upcasting F16/BF16
    /// on the CPU first. Use for kernel bindings that are typed
    /// `const device float*` but whose on-disk dtype varies across
    /// checkpoint conventions — e.g. the Gated-DeltaNet gated-RMSNorm
    /// `linear_attn.norm.weight`, which the official Qwen3.5 checkpoint
    /// ships as F32 but the `mlx-community/*-MLX-4bit` repack ships as
    /// BF16. Without the upcast the kernel reads 2-byte BF16 as 4-byte
    /// F32 → garbage → NaN. Upcast is lossless, so the F32-on-disk path
    /// is byte-identical to before.
    pub fn take_as_f32(&mut self, name: &str) -> Result<GpuTensor> {
        let shape = self
            .tensors
            .get(name)
            .map(|r| r.shape.clone())
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let f32_data = self.take_to_cpu_f32(name)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        self.alloc_packed_from_host(bytes, &shape, DType::F32)
    }

    /// Try to take a pre-staged entry for the given tensor's bytes.
    /// CUDA-only.
    ///
    /// Keyed by data identity, so aliased names resolve to the same
    /// entry. If a worker is mid-copy on this key, waits for it (the
    /// wait is bounded by one tensor's copy). If no worker has
    /// started it, marks the key consumed — so no worker wastes a
    /// pinned buffer on bytes the caller is about to upload via the
    /// slow path — and returns `None`.
    fn take_precast(&self, cpu_ref: &CpuTensorRef) -> Option<PrecastEntry> {
        let pipeline = self.precast.as_ref()?;
        let mmap = cpu_ref.mmap.as_ref()?;
        pipeline.take(
            mmap.as_ptr() as usize,
            cpu_ref.data_offset,
            cpu_ref.size_bytes,
        )
    }

    /// Take a tensor and return its data as a CPU `Vec<f32>`.
    ///
    /// Useful for small per-head parameters (A_log, dt_bias, norm weights)
    /// that need to be kept on CPU or uploaded to GPU as f32.
    /// Take a tensor's RAW typed bytes (no f32 conversion) + its dtype + shape —
    /// the zero-conversion ingest path for backends whose device dtype matches
    /// the on-disk dtype (Spyre f16), or that narrow typed bytes themselves
    /// (bf16→f16 on ingest). Destructive, like [`take_to_cpu_f32`]; copies the
    /// mmap'd bytes once into an owned `Vec<u8>`.
    pub fn take_to_cpu_bytes(&mut self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let bytes = cpu_ref.data()[..cpu_ref.size_bytes].to_vec();
        Ok((bytes, cpu_ref.dtype, cpu_ref.shape))
    }

    pub fn take_to_cpu_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let data = cpu_ref.data();
        let num_elems: usize = cpu_ref.shape.iter().product();
        let mut result = Vec::with_capacity(num_elems);

        match cpu_ref.dtype {
            DType::F32 => {
                let src =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, num_elems) };
                result.extend_from_slice(src);
            }
            // Unaligned byte reads: mmap'd safetensors tensors are NOT
            // guaranteed 2-byte aligned (a tensor can start at an odd
            // offset after an odd-sized neighbor), so reinterpreting the
            // raw pointer as `*const f16/bf16` trips the debug-build
            // alignment precondition → abort. `from_le_bytes` over a
            // stack-copied `[u8; 2]` is alignment-free.
            DType::F16 => {
                let (pairs, _) = data[..num_elems * 2].as_chunks::<2>();
                for &pair in pairs {
                    result.push(half::f16::from_le_bytes(pair).to_f32());
                }
            }
            DType::BF16 => {
                let (pairs, _) = data[..num_elems * 2].as_chunks::<2>();
                for &pair in pairs {
                    result.push(half::bf16::from_le_bytes(pair).to_f32());
                }
            }
            other => anyhow::bail!("take_to_cpu_f32: unsupported dtype {other}"),
        }

        Ok(result)
    }
}

impl<A: DeviceAllocator> GpuWeights<A> {
    /// Get the shape and effective dtype of a tensor without loading it to GPU.
    ///
    /// If `target_dtype` is set and the tensor is a floating-point type, the
    /// returned dtype reflects the cast target (matching what `take`/`take_into`
    /// will produce). This ensures callers compute correct byte sizes for
    /// pre-allocated buffers.
    pub fn tensor_info(&self, name: &str) -> Option<(&[usize], DType)> {
        self.tensors.get(name).map(|r| {
            let effective_dtype = match self.target_dtype {
                Some(target)
                    if matches!(r.dtype, DType::F32 | DType::F16 | DType::BF16)
                        && r.dtype != target =>
                {
                    target
                }
                _ => r.dtype,
            };
            (r.shape.as_slice(), effective_dtype)
        })
    }

    /// Rewrite a safetensors entry's shape in place. Element count
    /// must match the on-disk size. Used to flatten conv-style
    /// tensors (e.g. Qwen2-VL `visual.patch_embed.proj.weight`
    /// `[E, C, T, P, P]` 5D, or SigLIP's `[E, C, P, P]` 4D) to the
    /// 2D form a downstream `take`-style loader expects, when that
    /// loader doesn't have a shape-override entry point. No data is
    /// moved; only the shape metadata changes.
    pub fn reshape_in_place(&mut self, name: &str, new_shape: &[usize]) -> Result<()> {
        let cpu_ref = self
            .tensors
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("reshape_in_place: weight not found: {name}"))?;
        let old_numel: usize = cpu_ref.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();
        anyhow::ensure!(
            old_numel == new_numel,
            "reshape_in_place: {name} on-disk shape {:?} (numel {old_numel}) != new shape {:?} (numel {new_numel})",
            cpu_ref.shape,
            new_shape,
        );
        cpu_ref.shape = new_shape.to_vec();
        Ok(())
    }

    /// Physically transpose a 2D weight in place at CPU side, before
    /// any `take`/`take_with_shape` runs. The `CpuTensorRef` switches
    /// from mmap-backed to owned-backed: an owned `Vec<u8>` of the same
    /// size is allocated, the source bytes are copied with axes swapped,
    /// and the shape metadata is `[d0, d1]` → `[d1, d0]`. Element dtype
    /// must be 2-byte (bf16 / fp16) or 4-byte (fp32).
    ///
    /// Used by `LinearLayer::load_raw` for `nn.Parameter` weights that
    /// ship matmul-natural `[K, N]` (HF `Gemma3MultiModalProjector::
    /// mm_input_projection_weight`, used as `act @ param`) — scratchy's
    /// gemm expects `[N, K]` (PyTorch nn.Linear convention), so the
    /// transpose lands the on-disk param into the runtime convention.
    pub fn transpose_2d_in_place(&mut self, name: &str) -> Result<()> {
        let cpu_ref = self
            .tensors
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("transpose_2d_in_place: weight not found: {name}"))?;
        anyhow::ensure!(
            cpu_ref.shape.len() == 2,
            "transpose_2d_in_place: {name} must be 2D, got {:?}",
            cpu_ref.shape,
        );
        let d0 = cpu_ref.shape[0];
        let d1 = cpu_ref.shape[1];
        let elem = cpu_ref.dtype.size_bytes();
        let total_bytes = d0 * d1 * elem;
        let src = cpu_ref.data();
        anyhow::ensure!(
            src.len() == total_bytes,
            "transpose_2d_in_place: {name} byte length {} doesn't match shape {:?} × elem {}",
            src.len(),
            cpu_ref.shape,
            elem,
        );

        let mut buf = vec![0u8; total_bytes];
        for i in 0..d0 {
            for j in 0..d1 {
                let src_off = (i * d1 + j) * elem;
                let dst_off = (j * d0 + i) * elem;
                buf[dst_off..dst_off + elem].copy_from_slice(&src[src_off..src_off + elem]);
            }
        }

        cpu_ref.mmap = None;
        cpu_ref.data_offset = 0;
        cpu_ref.size_bytes = total_bytes;
        cpu_ref.shape = vec![d1, d0];
        cpu_ref.owned = Some(Arc::new(buf));
        Ok(())
    }

    /// Physically permute a **channels-last** conv weight to the
    /// flattened 2D form the patch-embed gemm needs, moving the
    /// trailing in-channels axis to the front of the kernel dims.
    ///
    /// MLX ships Conv3d/Conv2d weights channels-LAST, e.g. Qwen3.5-VL's
    /// `vision_tower.patch_embed.proj.weight` is `[out, kt, kh, kw, in]`.
    /// The Conv's forward `moveaxis(in→last)` on the input means the
    /// equivalent gemm pairs each patch (packed channels-FIRST as
    /// `[in, kt, kh, kw]` by the processor / mlx `reshape(-1,C,T,P,P)`)
    /// with the weight flattened in the SAME `[in, kt, kh, kw]` order.
    /// A plain `reshape_in_place` keeps the on-disk `[kt,kh,kw,in]`
    /// order, mispairing every element — the dot product is only
    /// order-invariant for uniform patches, so it silently corrupts the
    /// high-variance (image-content) patches and leaves the flat
    /// background ones correct. This moves `in` to the front and
    /// flattens to `[out, in*kt*kh*kw]`. `leading_dim` is the output
    /// axis (0); all axes after it are the conv dims, the LAST of which
    /// is in-channels.
    pub fn flatten_conv_weight_channels_last(
        &mut self,
        name: &str,
        leading_dim: usize,
    ) -> Result<()> {
        self.flatten_conv_weight(name, leading_dim, /* in_chans_hint */ None)
    }

    /// Same as [`flatten_conv_weight_channels_last`] but with a shape-
    /// discriminator. When `in_chans_hint` is `Some(c)`, sniffs whether the
    /// on-disk weight is channels-LAST `[out, kt, kh, kw, c]` (mlx-converted
    /// checkpoints) or already channels-FIRST `[out, c, kt, kh, kw]` (HF
    /// native PyTorch Conv3d weight, e.g. `Qwen/Qwen3.5-9B`'s
    /// `model.visual.patch_embed.proj.weight` `[1152, 3, 2, 16, 16]`).
    /// Channels-FIRST inputs are just flattened in place; channels-LAST go
    /// through the existing permute. `None` for the hint preserves the
    /// always-permute legacy behavior.
    pub fn flatten_conv_weight(
        &mut self,
        name: &str,
        leading_dim: usize,
        in_chans_hint: Option<usize>,
    ) -> Result<()> {
        let cpu_ref = self
            .tensors
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("flatten_conv_weight: not found: {name}"))?;
        let shape = cpu_ref.shape.clone();
        anyhow::ensure!(
            leading_dim + 2 < shape.len(),
            "flatten_conv_weight: {name} shape {shape:?} needs >=2 conv dims after \
             leading_dim {leading_dim}",
        );
        let lead: usize = shape[..=leading_dim].iter().product(); // = out
        let conv = &shape[leading_dim + 1..];
        let elem = cpu_ref.dtype.size_bytes();

        // Channels-FIRST detection: if the caller supplied `in_chans_hint`
        // and the FIRST conv dim equals it (and the LAST does not), the
        // weight is already in PyTorch Conv3d native order — just flatten,
        // no permute needed. Equivalent old code path was a partial
        // identity permute that mispaired indices.
        let channels_first = match in_chans_hint {
            Some(c) => conv.first() == Some(&c) && conv.last() != Some(&c),
            None => false,
        };

        let rest: usize = conv.iter().product(); // in*kt*kh*kw
        let total_bytes = lead * rest * elem;
        let src = cpu_ref.data();
        anyhow::ensure!(
            src.len() == total_bytes,
            "flatten_conv_weight: {name} byte len {} != shape {shape:?} × elem {elem}",
            src.len(),
        );

        if channels_first {
            // Already row-major [out, in, kt, kh, kw]; just reshape to
            // [out, in*kt*kh*kw]. No data copy.
            cpu_ref.shape = vec![lead, rest];
            return Ok(());
        }

        // Channels-LAST permute: src [out, kt, kh, kw, in] -> dst [out, in, kt, kh, kw].
        let in_ch = *conv.last().unwrap();
        let kdims = &conv[..conv.len() - 1];
        let kprod: usize = kdims.iter().product(); // kt*kh*kw
        // src row-major [out, kdims..., in]: src_flat = (out*kprod + kidx)*in_ch + ic
        // dst row-major [out, in, kdims...]:  dst_flat =  out*rest + ic*kprod + kidx
        let mut buf = vec![0u8; total_bytes];
        for out in 0..lead {
            for kidx in 0..kprod {
                for ic in 0..in_ch {
                    let s = ((out * kprod + kidx) * in_ch + ic) * elem;
                    let d = (out * rest + ic * kprod + kidx) * elem;
                    buf[d..d + elem].copy_from_slice(&src[s..s + elem]);
                }
            }
        }
        cpu_ref.mmap = None;
        cpu_ref.data_offset = 0;
        cpu_ref.size_bytes = total_bytes;
        cpu_ref.shape = vec![lead, rest];
        cpu_ref.owned = Some(Arc::new(buf));
        Ok(())
    }

    /// Zero-pad a 2D weight's `dim` axis (0 or 1) to the next multiple
    /// of `mult` at CPU side, before any `take`/`take_with_shape` runs.
    /// The CpuTensorRef switches from mmap-backed to owned-backed: an
    /// owned `Vec<u8>` of the padded size is allocated, zero-init'd,
    /// and the original on-disk bytes are memcpy'd into the leading
    /// slice. Subsequent loads see the padded shape directly. No-op
    /// when the axis is already a multiple of `mult`.
    ///
    /// Used by the vision encoder to work around cuBLAS bf16 GEMM
    /// failing on K=3420 (Qwen2.5-VL-3B `vision_intermediate_size`):
    /// padding K (or N on the producing weights) to the next mult of
    /// 8 lands every gemm on an algo cuBLAS supports, with zero-fill
    /// columns/rows preserving math (`silu(0)·0 = 0` on the activation
    /// side, weight-side zero rows contract to zero in the next gemm).
    pub fn pad_axis_to_mult8(&mut self, name: &str, dim: usize, mult: usize) -> Result<()> {
        let cpu_ref = self
            .tensors
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("pad_axis_to_mult8: weight not found: {name}"))?;
        anyhow::ensure!(
            cpu_ref.shape.len() == 2,
            "pad_axis_to_mult8: {name} must be 2D, got {:?}",
            cpu_ref.shape,
        );
        anyhow::ensure!(dim < 2, "pad_axis_to_mult8: dim must be 0 or 1, got {dim}");
        let old_shape = cpu_ref.shape.clone();
        let pad_to = old_shape[dim].next_multiple_of(mult);
        if pad_to == old_shape[dim] {
            return Ok(());
        }
        let elem = cpu_ref.dtype.size_bytes();
        let mut new_shape = old_shape.clone();
        new_shape[dim] = pad_to;
        let new_numel: usize = new_shape.iter().product();
        let new_size_bytes = new_numel * elem;

        let src = cpu_ref.data();
        let mut buf = vec![0u8; new_size_bytes];
        match dim {
            0 => {
                // Padding rows: copy old `[d0_old, d1] * elem` bytes
                // to leading slice; trailing rows stay zero.
                let copy_bytes = old_shape[0] * old_shape[1] * elem;
                buf[..copy_bytes].copy_from_slice(&src[..copy_bytes]);
            }
            1 => {
                // Padding cols: per-row copy of `d1_old * elem` bytes
                // into the first `d1_old * elem` of each padded row.
                let src_row = old_shape[1] * elem;
                let dst_row = new_shape[1] * elem;
                let rows = old_shape[0];
                for r in 0..rows {
                    let s = &src[r * src_row..(r + 1) * src_row];
                    buf[r * dst_row..r * dst_row + src_row].copy_from_slice(s);
                }
            }
            _ => unreachable!(),
        }

        cpu_ref.mmap = None;
        cpu_ref.data_offset = 0;
        cpu_ref.size_bytes = new_size_bytes;
        cpu_ref.shape = new_shape;
        cpu_ref.owned = Some(Arc::new(buf));
        Ok(())
    }

    /// Tensor shape lookup that checks all three backing maps —
    /// safetensors `tensors`, `gguf_dense`, and `quantized`. Used by
    /// the per-variant fingerprint sniff which needs to verify
    /// embedding / first-layer shapes regardless of backing store.
    /// Returns `Vec<usize>` to avoid borrow lifetime tangles across
    /// heterogeneous backings (CpuTensorRef carries usize, GpuTensor
    /// carries u32, GgmlStorage stores nrows/ncols).
    pub fn tensor_shape_any(&self, name: &str) -> Option<Vec<usize>> {
        if let Some(r) = self.tensors.get(name) {
            return Some(r.shape.clone());
        }
        if let Some(t) = self.gguf_dense.get(name) {
            return Some(t.shape().iter().map(|&d| d as usize).collect());
        }
        if let Some(s) = self.quantized.get(name) {
            // 2D row-major weight; nrows = out, ncols = in. Match
            // the safetensors layout convention.
            return Some(vec![s.nrows, s.ncols]);
        }
        None
    }

    /// Read a CPU-side weight and convert to `f32`, regardless of its
    /// on-disk dtype (bf16 / f16 / f32). For host-side preprocessing that
    /// can't run on the GPU — e.g. Qwen3.5-VL's `fast_pos_embed_interpolate`
    /// over the learned `pos_embed.weight`. Returns `None` if the tensor is
    /// absent or quantized (not a plain dense float).
    pub fn tensor_to_f32(&self, name: &str) -> Option<Vec<f32>> {
        let r = self.tensors.get(name)?;
        let bytes = r.data();
        let v = match r.dtype {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| half::bf16::from_bits(u16::from_le_bytes(c)).to_f32())
                .collect(),
            DType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| half::f16::from_bits(u16::from_le_bytes(c)).to_f32())
                .collect(),
            _ => return None,
        };
        Some(v)
    }

    /// Get a tensor by name (copies to GPU). For read-only access.
    ///
    /// WARNING: The returned GPU tensor is leaked — caller must arrange cleanup.
    /// Prefer `take()` which is more explicit about ownership transfer.
    pub fn get(&mut self, name: &str) -> Option<GpuTensor> {
        // Remove temporarily to satisfy borrow checker, then re-insert.
        let cpu_ref = self.tensors.remove(name)?;
        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);
        let gpu_ptr = unsafe { self.allocator.alloc_and_copy_host(data, size_bytes).ok()? };

        let shape = cpu_ref.shape.clone();
        self.tensors.insert(name.to_string(), cpu_ref);

        Some(unsafe { GpuTensor::new(gpu_ptr, &shape, dtype) })
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
            || self.quantized.contains_key(name)
            || self.gguf_dense.contains_key(name)
    }

    // ----------------------------------------------------------------------
    // GGUF backing-store accessors
    // ----------------------------------------------------------------------
    //
    // Populated by `from_gguf` (in `scratchy-target-cuda`, since it depends on
    // GGUF-specific kernels for dequantizing norms / embeddings). The
    // safetensors construction paths leave these maps empty.

    /// Get a quantized linear weight by HF tensor name. Returns
    /// `None` when the backing store is safetensors or the tensor
    /// is absent.
    ///
    /// **Non-destructive** — the entry stays in the map so multiple
    /// consumers can each obtain a `GgmlStorage` view of the same
    /// GPU buffer. Required because the same source weight (e.g.
    /// `model.layers.5.self_attn.k_proj.weight`) may be referenced
    /// by both a singleton accessor (prefill workload's
    /// `GgmlGemmImpl`) and a fused-QKV accessor (decode workload's
    /// `GgmlFusedQkvRopeCacheImpl`); the codegen emits both load
    /// paths and both must succeed. `GgmlStorage` is `Copy` and
    /// holds no ownership — the underlying GPU bytes are leaked
    /// for the model's lifetime, matching the existing pattern.
    ///
    /// The returned storage is valid as long as the `GpuWeights`
    /// (or its successor after `take_gpu_allocs`) is alive.
    pub fn take_quantized_linear(
        &mut self,
        name: &str,
    ) -> Option<scratchy_quantizations::GgmlStorage> {
        self.quantized.get(name).copied()
    }

    /// Take a dense (already dequantized) GGUF weight — norms,
    /// embeddings, lm_head. Returns `None` for the safetensors path.
    pub fn take_gguf_dense(&mut self, name: &str) -> Option<GpuTensor> {
        self.gguf_dense.remove(name)
    }

    /// True when this `GpuWeights` was populated from a GGUF file
    /// (i.e. has at least one entry in `quantized` or `gguf_dense`).
    /// Used by codegen to decide between the safetensors and GGUF
    /// load helpers.
    pub fn is_gguf(&self) -> bool {
        !self.quantized.is_empty() || !self.gguf_dense.is_empty()
    }

    /// True iff the named tensor is present as a GGUF-quantized
    /// linear (in `quantized`). Lets concat-loaders probe before
    /// committing to the GGUF byte-pack path vs the dense fallback.
    pub fn contains_quantized_linear(&self, name: &str) -> bool {
        self.quantized.contains_key(name)
    }

    /// True iff the named tensor is present in the GGUF-dense map
    /// (norms / embeddings / lm_head — pre-uploaded and dequantized).
    pub fn gguf_dense_contains(&self, name: &str) -> bool {
        self.gguf_dense.contains_key(name)
    }

    /// True iff the named tensor is present in the safetensors-backed
    /// `tensors` map (CPU-mmap, uploaded on `take`).
    pub fn safetensor_contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Iterator over GGUF-quantized linear tensor names. Diagnostic
    /// helper for `SCRATCHY_GGUF_TRACE`-style debug output.
    pub fn quantized_linear_names(&self) -> impl Iterator<Item = &String> {
        self.quantized.keys()
    }

    /// Direct access to the quantized-linear map for the GGUF
    /// loader's population step. Not part of the public API for
    /// model code — use `take_quantized_linear` from the model side.
    #[doc(hidden)]
    pub fn quantized_map_mut(
        &mut self,
    ) -> &mut HashMap<String, scratchy_quantizations::GgmlStorage> {
        &mut self.quantized
    }

    /// Direct access to the GGUF-dense map for the loader's
    /// population step. Same caveats as `quantized_map_mut`.
    #[doc(hidden)]
    pub fn gguf_dense_map_mut(&mut self) -> &mut HashMap<String, GpuTensor> {
        &mut self.gguf_dense
    }

    /// Synthesize N virtual per-slice entries from a packed safetensors
    /// source by taking an even row-wise split.
    ///
    /// `packed_prefix` is the full dotted path to the packed tensor (its
    /// `.weight` is at `{packed_prefix}.weight`; `.bias` is optional and
    /// split the same way). `split_targets` lists the sibling suffixes
    /// under the shared parent; each becomes a new entry at
    /// `{grandparent}.{target}.weight` whose `CpuTensorRef` points at the
    /// matching row range within the original mmap (no data is copied).
    /// The packed entry itself is removed on success.
    ///
    /// Returns `Ok(true)` when the split happened, `Ok(false)` when the
    /// packed tensor does not exist (caller should propagate the
    /// downstream `weight not found` error the normal way).
    ///
    /// Even-split convenience wrapper around
    /// `synthesize_packed_row_split_sizes`. Only valid when `total_rows`
    /// is divisible by `split_targets.len()` — e.g. MHA `qkv_proj`
    /// (`num_attention_heads == num_key_value_heads`) and every
    /// `gate_up_proj`. GQA checkpoints must go through the sized
    /// variant with explicit per-slice row counts.
    pub fn synthesize_packed_row_split(
        &mut self,
        packed_prefix: &str,
        split_targets: &[&str],
    ) -> Result<bool> {
        let packed_weight_name = format!("{packed_prefix}.weight");
        if !self.tensors.contains_key(&packed_weight_name) {
            return Ok(false);
        }
        let n = split_targets.len();
        if n == 0 {
            anyhow::bail!("synthesize_packed_row_split: empty split_targets");
        }
        let total_rows = self.tensors[&packed_weight_name].shape[0];
        if !total_rows.is_multiple_of(n) {
            anyhow::bail!(
                "packed source `{packed_weight_name}` rows ({total_rows}) not divisible by {n} slices; \
                 GQA-packed qkv needs `synthesize_packed_row_split_sizes` with explicit per-slice row counts",
            );
        }
        let rows_per_slice = total_rows / n;
        let sized: Vec<(&str, usize)> =
            split_targets.iter().map(|t| (*t, rows_per_slice)).collect();
        self.synthesize_packed_row_split_sizes(packed_prefix, &sized)
    }

    /// Tensor-parallel-aware split. Carves per-rank views out of a
    /// GGUF packed parent (e.g. Phi-3's `self_attn.qkv_proj` fused
    /// across q/k/v heads) so downstream `_sharded` loaders see slices
    /// that are already the right per-rank shape.
    ///
    /// At tp_world_size == 1, identical to
    /// [`Self::synthesize_packed_row_split_sizes`].
    ///
    /// At tp_world_size > 1 with a **quantized** parent (`ShardDim0`
    /// replicated across ranks by the GGUF loader because the fused
    /// parent's name isn't in the shard-kind rule table), carves each
    /// child slice at offset `slice_start + tp_rank * (slice_rows / tp)`
    /// with `slice_rows / tp` rows. Each child ends up per-rank sized
    /// and its `.weight` entry goes into `self.quantized` for the
    /// subsequent `_sharded` load helpers to consume as-is (they see
    /// the correct per-rank shape and skip further sharding).
    ///
    /// Safetensors parents at tp > 1 fall through to the unsharded
    /// carve: the CPU slices keep their full sizes and the downstream
    /// `Linear::load_sharded(dim=0, …)` does the per-rank `take_shard`.
    ///
    /// `split_targets` takes **full** per-slice row counts (what the
    /// unsharded manifest declares). Per-rank division is done inside
    /// this helper against `tp_world_size`; each `full_rows` must be
    /// divisible by `tp_world_size`, else bail.
    pub fn synthesize_packed_row_split_sizes_tp(
        &mut self,
        packed_prefix: &str,
        split_targets: &[(&str, usize)],
        tp_rank: usize,
        tp_world_size: usize,
    ) -> Result<bool> {
        if tp_world_size <= 1 {
            return self.synthesize_packed_row_split_sizes(packed_prefix, split_targets);
        }
        if tp_rank >= tp_world_size {
            anyhow::bail!(
                "synthesize_packed_row_split_sizes_tp: tp_rank ({tp_rank}) >= \
                 tp_world_size ({tp_world_size})"
            );
        }
        if split_targets.is_empty() {
            anyhow::bail!("synthesize_packed_row_split_sizes_tp: empty split_targets");
        }
        let grandparent = packed_prefix
            .rsplit_once('.')
            .map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("packed prefix has no parent: {packed_prefix}"))?;
        let packed_weight_name = format!("{packed_prefix}.weight");

        // Quantized-parent path: Phi-3 family at tp > 1 ships the
        // fused `attn_qkv` / `ffn_up` parent as a block-quantized
        // GgmlStorage, replicated across ranks by the GGUF loader
        // (the fused-parent name isn't in `gguf_shard_kind_for_hf_name`).
        // Carve per-rank views directly here so `_sharded` helpers
        // below see children of the right shape.
        if self.quantized.contains_key(&packed_weight_name) {
            let packed = self
                .quantized
                .remove(&packed_weight_name)
                .expect("contains");
            let total_rows = packed.nrows;
            let hidden = packed.ncols;
            let full_sum: usize = split_targets.iter().map(|(_, r)| *r).sum();
            if full_sum != total_rows {
                anyhow::bail!(
                    "synthesize_packed_row_split_sizes_tp: `{packed_weight_name}` rows \
                     ({total_rows}) != sum of full split sizes ({full_sum}) across {:?}",
                    split_targets
                        .iter()
                        .map(|(s, r)| format!("{s}={r}"))
                        .collect::<Vec<_>>(),
                );
            }
            for (t, r) in split_targets {
                if !r.is_multiple_of(tp_world_size) {
                    anyhow::bail!(
                        "synthesize_packed_row_split_sizes_tp: slice `{t}` rows ({r}) \
                         not divisible by tp_world_size ({tp_world_size}) in \
                         `{packed_weight_name}`"
                    );
                }
            }
            let block_elems = packed.dtype.block_size();
            let type_size = packed.dtype.type_size();
            if !hidden.is_multiple_of(block_elems) {
                anyhow::bail!(
                    "synthesize_packed_row_split_sizes_tp: `{packed_weight_name}` ncols \
                     ({hidden}) not divisible by block_size ({block_elems}, dtype {:?})",
                    packed.dtype,
                );
            }
            let row_bytes = (hidden / block_elems) * type_size;
            let mut full_row_offset = 0usize;
            for (target, full_rows) in split_targets {
                let per_rank_rows = full_rows / tp_world_size;
                let rank_row_offset = full_row_offset + tp_rank * per_rank_rows;
                let slice_bytes = per_rank_rows * row_bytes;
                // Row-aligned byte offset inside the replicated parent
                // buffer. Each row is a whole number of GGML blocks
                // (validated above via `hidden % block_elems == 0`) so
                // the pointer arithmetic never straddles a block boundary.
                let child_ptr = unsafe { packed.ptr.add(rank_row_offset * row_bytes) };
                let child = scratchy_quantizations::GgmlStorage {
                    ptr: child_ptr,
                    len: slice_bytes,
                    dtype: packed.dtype,
                    nrows: per_rank_rows,
                    ncols: hidden,
                };
                let vname = format!("{grandparent}.{target}.weight");
                self.quantized.insert(vname, child);
                full_row_offset += full_rows;
            }
            return Ok(true);
        }

        // Safetensors parent: fall through to the unsharded carve.
        // Downstream `Linear::load_sharded(dim=0, …)` applies per-rank
        // `take_shard` to each full-size slice, yielding the same
        // per-rank shape the quantized path produces above.
        self.synthesize_packed_row_split_sizes(packed_prefix, split_targets)
    }

    /// Sized-split sibling of `synthesize_packed_row_split`. Takes
    /// per-slice row counts so GQA-packed qkv (q and kv slices differ
    /// in output dim) and any other non-even split can be materialized.
    ///
    /// `split_targets` is `&[(suffix, rows)]`; the sum of `rows` must
    /// equal the packed tensor's first dim. Bias, if present on the
    /// packed parent, is split the same way.
    ///
    /// Used by scratchy-forward-compiler's manifest-driven `__packed_splits__`
    /// prelude (codegen emits one call per layer × packed-prefix before
    /// any `Weights::load` field read). No-op when the packed tensor
    /// isn't present — returns `Ok(false)` so non-packed checkpoints
    /// (Llama, Mistral, …) fall through unharmed.
    ///
    /// Two backing maps are checked: dense `tensors` (safetensors mmap
    /// refs) and quantized `quantized` (GGUF `GgmlStorage` on GPU).
    /// GGUF checkpoints (Phi-3, etc.) ship the fused parent in the
    /// quantized map; the row-split there carves the existing GPU
    /// buffer into N children that share the same allocation — no
    /// memory doubling, no extra H2D, no dequant.
    pub fn synthesize_packed_row_split_sizes(
        &mut self,
        packed_prefix: &str,
        split_targets: &[(&str, usize)],
    ) -> Result<bool> {
        let packed_weight_name = format!("{packed_prefix}.weight");
        if split_targets.is_empty() {
            anyhow::bail!("synthesize_packed_row_split_sizes: empty split_targets");
        }
        let grandparent = packed_prefix
            .rsplit_once('.')
            .map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("packed prefix has no parent: {packed_prefix}"))?;

        // GGUF quantized path — Phi-3 family ships fused qkv_proj /
        // gate_up_proj as block-quantized GgmlStorage. Each child is a
        // view into the parent's GPU buffer at a row-aligned byte
        // offset; the parent allocation is leaked-by-design (model
        // weights live for the model's lifetime, same as before this
        // split) so single-buffer ownership is preserved without any
        // refcount machinery.
        if self.quantized.contains_key(&packed_weight_name) {
            let packed = self
                .quantized
                .remove(&packed_weight_name)
                .expect("contains");
            let total_rows = packed.nrows;
            let hidden = packed.ncols;
            let sum_rows: usize = split_targets.iter().map(|(_, r)| *r).sum();
            if sum_rows != total_rows {
                anyhow::bail!(
                    "packed quantized source `{packed_weight_name}` rows ({total_rows}) != \
                     sum of split sizes ({sum_rows}) across {:?}",
                    split_targets
                        .iter()
                        .map(|(s, r)| format!("{s}={r}"))
                        .collect::<Vec<_>>(),
                );
            }
            let block_elems = packed.dtype.block_size();
            let type_size = packed.dtype.type_size();
            // Each row contributes hidden / block_elems blocks; rows are
            // stored contiguously, so a child at row offset `r` starts
            // at byte offset `r * (hidden / block_elems) * type_size`.
            // Row-alignment check: every quantized format we ship has
            // hidden % block_elems == 0 in practice (k-quants block
            // size 256, hidden ≥ 256 always; legacy quants block size
            // 32). If a future format violates this, error rather than
            // silently produce torn blocks.
            if !hidden.is_multiple_of(block_elems) {
                anyhow::bail!(
                    "quantized row-split unsafe: `{packed_weight_name}` has ncols ({hidden}) \
                     not divisible by `{}` block size ({block_elems})",
                    packed.dtype,
                );
            }
            let row_bytes = (hidden / block_elems) * type_size;
            let mut row_offset = 0usize;
            for (target, rows) in split_targets {
                let slice_bytes = *rows * row_bytes;
                let child_ptr = unsafe { packed.ptr.add(row_offset * row_bytes) };
                let child = scratchy_quantizations::GgmlStorage {
                    ptr: child_ptr,
                    len: slice_bytes,
                    dtype: packed.dtype,
                    nrows: *rows,
                    ncols: hidden,
                };
                let vname = format!("{grandparent}.{target}.weight");
                self.quantized.insert(vname, child);
                row_offset += rows;
            }
            return Ok(true);
        }

        if !self.tensors.contains_key(&packed_weight_name) {
            return Ok(false);
        }

        let packed_weight = self.tensors.remove(&packed_weight_name).expect("contains");
        if packed_weight.shape.len() != 2 {
            anyhow::bail!(
                "packed source `{packed_weight_name}` has rank {}, expected 2",
                packed_weight.shape.len(),
            );
        }
        let total_rows = packed_weight.shape[0];
        let hidden = packed_weight.shape[1];
        let sum_rows: usize = split_targets.iter().map(|(_, r)| *r).sum();
        if sum_rows != total_rows {
            anyhow::bail!(
                "packed source `{packed_weight_name}` rows ({total_rows}) != sum of split sizes ({sum_rows}) \
                 across {:?}",
                split_targets
                    .iter()
                    .map(|(s, r)| format!("{s}={r}"))
                    .collect::<Vec<_>>(),
            );
        }
        let elem = packed_weight.dtype.size_bytes();

        let mut row_offset = 0usize;
        for (target, rows) in split_targets {
            let slice_bytes = rows * hidden * elem;
            let vname = format!("{grandparent}.{target}.weight");
            let entry = CpuTensorRef {
                mmap: packed_weight.mmap.clone(),
                data_offset: packed_weight.data_offset + row_offset * hidden * elem,
                size_bytes: slice_bytes,
                shape: vec![*rows, hidden],
                dtype: packed_weight.dtype,
                owned: packed_weight.owned.clone(),
            };
            self.tensors.insert(vname, entry);
            row_offset += rows;
        }

        // MLX-affine quantized siblings: a fused `qkv_proj` /
        // `gate_up_proj` ships per-group `.scales` and `.biases`
        // alongside the packed `.weight`. They are `[total_rows, cols]`
        // (cols = K / group_size) and carve along the same output-row
        // boundaries as the weight — just with their own column count.
        for sibling in ["scales", "biases"] {
            let packed_name = format!("{packed_prefix}.{sibling}");
            let Some(packed) = self.tensors.remove(&packed_name) else {
                continue;
            };
            if packed.shape.len() != 2 || packed.shape[0] != total_rows {
                anyhow::bail!(
                    "packed affine `{packed_name}` shape {:?} inconsistent with weight rows {total_rows}",
                    packed.shape,
                );
            }
            let cols = packed.shape[1];
            let selem = packed.dtype.size_bytes();
            let mut row_offset = 0usize;
            for (target, rows) in split_targets {
                let vname = format!("{grandparent}.{target}.{sibling}");
                let entry = CpuTensorRef {
                    mmap: packed.mmap.clone(),
                    data_offset: packed.data_offset + row_offset * cols * selem,
                    size_bytes: *rows * cols * selem,
                    shape: vec![*rows, cols],
                    dtype: packed.dtype,
                    owned: packed.owned.clone(),
                };
                self.tensors.insert(vname, entry);
                row_offset += rows;
            }
        }

        let packed_bias_name = format!("{packed_prefix}.bias");
        if let Some(packed_bias) = self.tensors.remove(&packed_bias_name) {
            if packed_bias.shape.len() != 1 || packed_bias.shape[0] != total_rows {
                anyhow::bail!(
                    "packed bias `{packed_bias_name}` shape {:?} inconsistent with weight rows {total_rows}",
                    packed_bias.shape,
                );
            }
            let belem = packed_bias.dtype.size_bytes();
            let mut row_offset = 0usize;
            for (target, rows) in split_targets {
                let bslice_bytes = rows * belem;
                let vname = format!("{grandparent}.{target}.bias");
                let entry = CpuTensorRef {
                    mmap: packed_bias.mmap.clone(),
                    data_offset: packed_bias.data_offset + row_offset * belem,
                    size_bytes: bslice_bytes,
                    shape: vec![*rows],
                    dtype: packed_bias.dtype,
                    owned: packed_bias.owned.clone(),
                };
                self.tensors.insert(vname, entry);
                row_offset += rows;
            }
        }

        Ok(true)
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterator over all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Strip a prefix from all tensor names (e.g. "model.").
    pub fn strip_prefix(&mut self, prefix: &str) {
        let stripped: HashMap<String, CpuTensorRef> = self
            .tensors
            .drain()
            .filter_map(|(name, tensor)| {
                name.strip_prefix(prefix)
                    .map(|rest| (rest.to_string(), tensor))
            })
            .collect();
        self.tensors = stripped;
    }

    // -----------------------------------------------------------------------
    // LoRA weight merging (CPU-side, before H2D copy)
    // -----------------------------------------------------------------------

    /// Merge a LoRA adapter's A/B weight pairs into base weights on CPU.
    ///
    /// For each LoRA target module, computes `W_merged = W + scaling * B @ A`
    /// in f32 intermediate precision and replaces the mmap'd `CpuTensorRef`
    /// with an owned buffer containing the merged result.
    ///
    /// Must be called BEFORE `take()`/`take_into()` so that model construction
    /// picks up already-merged weights (including fused QKV / gate_up).
    ///
    /// Returns the number of weight tensors merged.
    pub fn merge_lora(&mut self, adapter_dir: &Path) -> Result<usize> {
        use scratchy_core_model::lora::LoraAdapterConfig;

        // 1. Parse adapter config.
        let config_path = adapter_dir.join("adapter_config.json");
        let config = LoraAdapterConfig::from_file(&config_path)
            .map_err(|e| anyhow::anyhow!("LoRA config: {e}"))?;
        let scaling = config.scaling();

        // 2. Load adapter weights (CPU mmap).
        let st_path = adapter_dir.join("adapter_model.safetensors");
        if !st_path.exists() {
            bail!(
                "adapter_model.safetensors not found in {}",
                adapter_dir.display()
            );
        }
        let file = std::fs::File::open(&st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("LoRA safetensors: {e}"))?;

        // 3. Group A/B pairs by layer prefix.
        //    PEFT names: base_model.model.{prefix}.lora_A.weight
        #[allow(clippy::type_complexity)]
        let mut pairs: HashMap<
            String,
            (Option<&[u8]>, Vec<usize>, Option<&[u8]>, Vec<usize>),
        > = HashMap::new();

        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let (prefix, is_a) = if let Some(p) = name.strip_suffix(".lora_A.weight") {
                (p, true)
            } else if let Some(p) = name.strip_suffix(".lora_B.weight") {
                (p, false)
            } else {
                continue;
            };
            let clean = prefix.strip_prefix("base_model.model.").unwrap_or(prefix);
            let entry = pairs
                .entry(clean.to_string())
                .or_insert((None, vec![], None, vec![]));
            if is_a {
                entry.0 = Some(view.data());
                entry.1 = view.shape().to_vec();
            } else {
                entry.2 = Some(view.data());
                entry.3 = view.shape().to_vec();
            }
        }

        // 4. For each pair, merge into the base weight.
        let mut merged_count = 0usize;
        for (prefix, (a_data, a_shape, b_data, b_shape)) in &pairs {
            let a_data = match a_data {
                Some(d) => d,
                None => {
                    tracing::warn!("LoRA: missing lora_A for {prefix}, skipping");
                    continue;
                }
            };
            let b_data = match b_data {
                Some(d) => d,
                None => {
                    tracing::warn!("LoRA: missing lora_B for {prefix}, skipping");
                    continue;
                }
            };

            // Find matching base weight. The prefix should match a key in self.tensors
            // (after strip_prefix("model.") has been applied, or not).
            let base_name = if self.tensors.contains_key(&format!("{prefix}.weight")) {
                format!("{prefix}.weight")
            } else {
                tracing::debug!("LoRA: no base weight for {prefix}, skipping");
                continue;
            };

            let base = &self.tensors[&base_name];
            if base.shape.len() != 2 {
                tracing::warn!("LoRA: base weight {base_name} is not 2D, skipping");
                continue;
            }

            // A: [rank, in], B: [out, rank], W: [out, in]
            let rank = a_shape[0];
            let in_feat = a_shape[1];
            let out_feat = b_shape[0];

            if base.shape != [out_feat, in_feat] {
                tracing::warn!(
                    "LoRA: shape mismatch for {base_name}: base {:?} vs LoRA out={out_feat} in={in_feat}",
                    base.shape
                );
                continue;
            }

            // Read base weight to f32.
            let numel = out_feat * in_feat;
            let mut w_f32 = vec![0.0f32; numel];
            read_to_f32(base.data(), base.dtype, &mut w_f32);

            // Read A to f32 [rank, in_feat].
            let a_numel = rank * in_feat;
            let mut a_f32 = vec![0.0f32; a_numel];
            // LoRA weights are typically F32 in PEFT safetensors.
            read_to_f32(a_data, DType::F32, &mut a_f32);

            // Read B to f32 [out_feat, rank].
            let b_numel = out_feat * rank;
            let mut b_f32 = vec![0.0f32; b_numel];
            read_to_f32(b_data, DType::F32, &mut b_f32);

            // Compute delta = B @ A → [out_feat, in_feat], then W += scaling * delta.
            let scaling_f32 = scaling as f32;
            for i in 0..out_feat {
                for j in 0..in_feat {
                    let mut dot = 0.0f32;
                    for k in 0..rank {
                        dot += b_f32[i * rank + k] * a_f32[k * in_feat + j];
                    }
                    w_f32[i * in_feat + j] += scaling_f32 * dot;
                }
            }

            // Write merged weight back in base dtype.
            let merged_bytes = write_from_f32(&w_f32, base.dtype);
            let size_bytes = merged_bytes.len();
            let owned = Arc::new(merged_bytes);

            // Replace CpuTensorRef with one backed by owned data.
            self.tensors.insert(
                base_name,
                CpuTensorRef {
                    mmap: None,
                    data_offset: 0,
                    size_bytes,
                    shape: vec![out_feat, in_feat],
                    dtype: base.dtype,
                    owned: Some(owned),
                },
            );

            merged_count += 1;
            tracing::debug!("LoRA: merged {prefix} → [{out_feat}, {in_feat}]");
        }

        tracing::info!(
            "LoRA: merged {} weight tensors (rank={}, alpha={}, scaling={:.4})",
            merged_count,
            config.r,
            config.lora_alpha,
            scaling,
        );

        Ok(merged_count)
    }

    // -----------------------------------------------------------------------
    // Tensor-parallel sharding (CPU-side slice → GPU)
    // -----------------------------------------------------------------------

    /// Remove a tensor by name, slice it along `dim` for tensor parallelism,
    /// and copy only the shard to GPU. Returns a GPU tensor of the shard.
    ///
    /// For dim=0 sharding (column parallel): contiguous slice of rows.
    /// For dim=1 sharding (row parallel): strided extraction of columns,
    /// copied row-by-row into a contiguous pinned buffer before H2D.
    pub fn take_shard(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<GpuTensor> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        // Pre-staged fast path: a dim-0 shard is a contiguous row range
        // of the staged (possibly cast) device-resident full tensor.
        // world==1 hands the whole staged buffer over zero-copy; world>1
        // D2D-copies the slice out and frees the staging. A dim-1 shard
        // needs a strided gather on host bytes — the staged entry is
        // unusable there and just dropped (tp>1 row-parallel only).
        if let Some(entry) = self.take_precast(&cpu_ref)
            && let Some((src, shard_shape, shard_bytes)) =
                shard_of_entry(&entry, &cpu_ref.shape, dim, rank, world_size)
        {
            let dtype = entry.dtype;
            if shard_bytes == entry.size_bytes {
                // world == 1: the staged buffer IS the tensor.
                let ptr = entry.gpu.ptr();
                self.allocator.adopt_raw(entry.gpu);
                return Ok(unsafe { GpuTensor::new(ptr, &shard_shape, dtype) });
            }
            // world > 1: D2D the strided slice out of the staged buffer into a
            // fresh tracked allocation (syncs before we drop the staging src).
            let dev = unsafe { self.allocator.alloc_and_copy_device(src, shard_bytes)? };
            drop(entry);
            return Ok(unsafe { GpuTensor::new(dev, &shard_shape, dtype) });
        }

        let (data, shard_shape, dtype) = self.shard_cpu_data(&cpu_ref, dim, rank, world_size);

        let size_bytes = shard_shape.iter().product::<usize>() * dtype.size_bytes();
        let gpu_ptr = unsafe { self.allocator.alloc_and_copy_host(data, size_bytes)? };

        Ok(unsafe { GpuTensor::new(gpu_ptr, &shard_shape, dtype) })
    }

    /// Internal: extract a shard from CPU tensor data. Returns (ptr, shard_shape, dtype).
    ///
    /// For dim=0: returns a pointer into the original data (contiguous slice).
    /// For dim=1: copies strided columns into the pinned cast buffer, returns pointer to that.
    fn shard_cpu_data(
        &mut self,
        cpu_ref: &CpuTensorRef,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> (*const u8, Vec<usize>, DType) {
        assert!(!cpu_ref.shape.is_empty(), "cannot shard scalar");
        assert!(dim < cpu_ref.shape.len(), "dim out of range");
        let full_size = cpu_ref.shape[dim];
        assert!(
            full_size.is_multiple_of(world_size),
            "dim {dim} size {full_size} not divisible by world_size {world_size}"
        );
        let shard_size = full_size / world_size;

        // Apply dtype casting first if needed.
        let (src_data, _src_bytes, dtype) = self.maybe_cast_cpu(cpu_ref);

        let elem_size = dtype.size_bytes();
        let mut shard_shape = cpu_ref.shape.clone();
        shard_shape[dim] = shard_size;

        if dim == 0 {
            // Contiguous slice: rows [rank*shard_size .. (rank+1)*shard_size].
            // Each row has product(shape[1:]) elements.
            let row_elems: usize = cpu_ref.shape[1..].iter().product();
            let row_bytes = row_elems * elem_size;
            let offset = rank * shard_size * row_bytes;
            let data = unsafe { src_data.add(offset) };
            (data, shard_shape, dtype)
        } else if dim == 1 && cpu_ref.shape.len() == 2 {
            // Strided column extraction for 2D tensor [rows, cols].
            // Extract columns [rank*shard_size .. (rank+1)*shard_size] from each row.
            let rows = cpu_ref.shape[0];
            let cols = cpu_ref.shape[1];
            let col_start = rank * shard_size;
            let shard_row_bytes = shard_size * elem_size;
            let needed = rows * shard_row_bytes;
            if self.cast_scratch.len() < needed {
                self.cast_scratch.resize(needed, 0);
            }
            let dst = self.cast_scratch.as_mut_ptr();
            for r in 0..rows {
                let src_offset = (r * cols + col_start) * elem_size;
                let dst_offset = r * shard_row_bytes;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_data.add(src_offset),
                        dst.add(dst_offset),
                        shard_row_bytes,
                    );
                }
            }
            (dst as *const u8, shard_shape, dtype)
        } else {
            panic!(
                "take_shard: unsupported dim={dim} for {}D tensor",
                cpu_ref.shape.len()
            );
        }
    }

    /// Take a tensor's raw CPU bytes without uploading to GPU.
    /// Returns (data_bytes, shape, dtype).
    ///
    /// Routes through [`Self::maybe_cast_cpu`] so the bytes match
    /// `target_dtype` when one is set — same cast surface cuda's
    /// `take_into` uses, so callers don't have to track the dtype
    /// drift between disk and target separately. Under metal this is
    /// the only on-the-way-in cast path; under cuda it's available to
    /// pre-allocated-host loaders that don't run on a stream.
    pub fn take_cpu(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, DType)> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let (ptr, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);
        // SAFETY: `ptr` points into either `cpu_ref.data()` (if no cast
        // happened) or `self.cast_scratch` (filled by `maybe_cast_cpu`).
        // We immediately copy out before either source is reused.
        let data = unsafe { std::slice::from_raw_parts(ptr, size_bytes) }.to_vec();
        Ok((data, cpu_ref.shape, dtype))
    }
}

impl<A: DeviceAllocator> Drop for GpuWeights<A> {
    fn drop(&mut self) {
        // Signal pre-stage workers to stop, then wait for them. After the
        // join the workers' `Arc<dyn PrecastPipeline>` clones are gone, so when
        // `self.precast` (this struct's field) drops it is the last ref and the
        // pipeline's own `Drop` frees any unconsumed pre-staged device buffers
        // (`RawGpuMem`s in its ready-map) on this thread, which holds the
        // load-time CUDA context. No-op when no pipeline was started (metal).
        if let Some(pipeline) = &self.precast {
            pipeline.shutdown();
        }
        for handle in self.precast_handles.drain(..) {
            handle.join().ok();
        }
        // `cast_scratch: Vec<u8>` drops itself.
        // `allocator` drops itself (CUDA: frees GPU mem; Metal: frees MTLBuffers).
        // GPU memory allocated by take()/take_into() is owned by model layers
        //   on the cuda path (transferred via take_gpu_allocs); on metal it's
        //   held by the allocator's MetalBuffer arenas.
        // CPU mmaps are dropped automatically when Arc<Mmap> refcounts reach zero.
    }
}
