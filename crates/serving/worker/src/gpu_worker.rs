// SPDX-License-Identifier: Apache-2.0
//! `CudaWorker` / `MetalWorker`: `Worker` implementations backed by the GPU runtime.
//!
//! Cuda build: scratchy-serving-cuda + scratchy-forward-compiler (full feature set: CUDA graphs,
//! NCCL, FP8, etc.). Metal build: scratchy-forward-compiler MetalWorkerPool + the
//! scratchy-target-metal crate. The struct is shared with cfg-mutex'd fields.
//!
//! Gated behind the `cuda` or `metal` feature flag.

// Force the linker to keep `scratchy_models` — its `#[forward]`
// modules register with `inventory::submit!` for auto-discovery by
// `scratchy_forward_compiler::try_load`. Without this, the linker gc's the
// crate (no direct symbol references after the Phase B collapse)
// and the inventory comes up empty. Mirrored under metal now that
// the per-canonical metal `forward` body + `inventory::submit!`
// registration are emitted under `cfg(any(cuda, metal))` (Step 3.E).
#[cfg(any(feature = "cuda", feature = "metal"))]
extern crate scratchy_models as _;

// Force the linker to keep `scratchy_builder_cuda` so its build.rs runs (it
// compiles the CUDA kernel `.a` files that `scratchy-target-cuda/build.rs`
// emits `rustc-link-lib` directives for). No code is imported from it; the
// reference exists solely to anchor the crate in this binary's dep graph.
// cuda-only — the builder is not a dependency under metal.
#[cfg(feature = "cuda")]
extern crate scratchy_builder_cuda as _;

#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::path::{Path, PathBuf};

#[cfg(feature = "metal")]
use scratchy_core_common::SamplingParams;
// Read only by the metal embeddings path below (it walks a
// StorageModeShared arena slot via `buf.contents()`), so the import follows
// that gate rather than being carried on every backend.
#[cfg(feature = "metal")]
use scratchy_core_common::engine_io::EmbeddingData;
#[cfg(feature = "metal")]
use scratchy_core_model::weight::HfModelConfig;
#[cfg(feature = "metal")]
use scratchy_serving_engine::executor::ModelRunnerOutput;
#[cfg(feature = "metal")]
use scratchy_serving_scheduler::scheduler::output::SchedulerOutput;
#[cfg(feature = "metal")]
use tracing::info;

// Backend-neutral types lifted in Step 1 of the cfg-mutex extension:
// `KvCachePool` is `cfg(any(cuda, metal))` in `scratchy-target-cuda::kv_cache`,
// and `DType` is unconditional in `scratchy-target-cuda::dtype`.
#[cfg(feature = "metal")]
use scratchy_forward_compiler::gdn_slot_allocator::GdnSlotAllocator;
#[cfg(feature = "metal")]
use scratchy_target_metal::dtype::DType as GpuDType;
#[cfg(feature = "metal")]
use scratchy_target_metal::gdn_state::GdnStatePool;
#[cfg(feature = "metal")]
use scratchy_target_metal::kv_cache::KvCachePool;

// The CUDA `Worker` lives in `scratchy-target-cuda`; re-exported here for
// back-compat so callers naming `gpu_worker::{CudaWorker, CudaWorkerFactory}`
// keep resolving.
#[cfg(feature = "cuda")]
pub use scratchy_target_cuda::cuda_worker::{CudaWorker, CudaWorkerFactory};

#[cfg(feature = "metal")]
use scratchy_target_metal::OwnedTensor;

// Backend-neutral types used by metal lifecycle bodies. Under cfg(metal),
// `GpuDevice` resolves to the Apple-silicon arm carrying `device + queue +
// allocator`; `OwnedTensor` / `TensorView` / `GpuTensor` / `GpuWeights` are
// the ones lifted to `cfg(any(cuda, metal))` in Step 1; `MetalAllocator`
// is the metal-side `BackendAllocator`; `MetalMem::from_buffer` wraps
// `metal::Buffer` for the metal arm of the KV/GDN pools.
#[cfg(feature = "metal")]
use ::objc2_metal::{MTLBuffer as _, MTLDevice as _};
#[cfg(feature = "metal")]
use scratchy_target_metal::MetalWeightsExt;
#[cfg(feature = "metal")]
use scratchy_target_metal::weights::GpuWeights;
#[cfg(feature = "metal")]
use scratchy_target_metal::{GpuDevice, GpuTensor, MetalAllocator, MetalMem, TensorView};

#[cfg(feature = "metal")]
use crate::error::{ExecutorError, ExecutorResult};
#[cfg(feature = "metal")]
use crate::input_batch::InputBatch;
#[cfg(all(feature = "metal", feature = "vision"))]
use crate::input_batch::PreparedInputs;
#[cfg(feature = "metal")]
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// `WorkerCreateConfig` lives in the neutral `worker_factory` module so the
// dispatch path compiles without a backend feature; re-exported here for the
// gated worker code that constructs it.
pub use crate::worker_factory::WorkerCreateConfig;

/// Per-image MRoPE metadata in seq space, used by
/// `build_mrope_positions_2d` to walk the entire seq for
/// each MM-bearing req (including the cached-prefix portion) so the
/// cursor lands at the same position the encoder used to encode KV.
///
/// Tuple is `(seq_offset, length, grid_t, grid_h_merged, grid_w_merged)`
/// where `seq_offset` is relative to the req's full sequence start (NOT
/// batch start) — the same coordinate system as `PlaceholderRange.offset`.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub type SeqMmInfo = (u32, u32, u32, u32, u32);

// ---------------------------------------------------------------------------
// Model enum (dispatches to LLaMA / Qwen2 / Gemma2)
// ---------------------------------------------------------------------------

/// Host-gathered grammar allow-masks for one decode step: a dense vocab
/// bitset per constrained request (`allow_bits`, row-major,
/// `words_per_row` u32 per row, bit set = token allowed) plus the logits
/// batch row each bitset masks (`rows`). Built in `execute_model` where
/// the req_id -> batch-row mapping is known, then uploaded + dispatched
/// before argmax in `forward_argmax_blocking`.
#[cfg(all(feature = "metal", feature = "guided-decoding"))]
struct GrammarMaskHost {
    rows: Vec<u32>,
    allow_bits: Vec<u32>,
    vocab: u32,
    words_per_row: u32,
}

/// GPU-resident form of [`GrammarMaskHost`]: the uploaded bitset / row
/// buffers + a small `[vocab, words_per_row]` consts buffer + the MTL4
/// argument table, captured by the forward followup closure to dispatch
/// the grammar mask just before argmax. `kernels_addr` is a raw pointer
/// to the worker's `GrammarMaskKernels` (same `'static` trick the argmax
/// path uses for `argmax_kernels`); valid for the synchronous duration of
/// the forward call.
#[cfg(all(feature = "metal", feature = "guided-decoding"))]
struct GrammarMaskGpu {
    allow_bits: scratchy_target_metal::grammar_mask::Buffer,
    rows: scratchy_target_metal::grammar_mask::Buffer,
    gconsts: scratchy_target_metal::grammar_mask::Buffer,
    arg_table: ::objc2::rc::Retained<
        ::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ArgumentTable>,
    >,
    num_rows: u32,
    kernels_addr: usize,
}

/// A worker backed by the GPU runtime (metal) for
/// zero-allocation inference.
#[cfg(feature = "metal")]
pub struct MetalWorker {
    // ---------------------------------------------------------------
    // Backend-neutral state (cfg-mutexed only by transitive types).
    // ---------------------------------------------------------------
    config: WorkerCreateConfig,
    kv_cache: Option<KvCachePool>,
    /// TurboQuant KV compression runtime — `Some` only when
    /// `kv_cache_dtype == "turboquant"`. After each forward writes KV, the
    /// worker calls `compress_layer` per layer over the new slots: quantizes
    /// them into the per-layer packed store (~4.6x) and writes the dequant back
    /// into the fp16 pool, so attention reads TurboQuant'd KV. OFF by default.
    turboquant: Option<scratchy_target_metal::turboquant::TurboQuantRuntime>,
    /// TurboQuant eviction-layer window map: logical KV block -> bounded fp16
    /// window slot. `Some` alongside `turboquant`. Opt-in via SCRATCHY_TQ_WINDOW
    /// — when set, the forward remaps block_table/slot_mapping logical->window,
    /// dequant-fills evicted misses from the packed store, and the packed store
    /// is the canonical full-capacity cache (the 4.6x). Off => the in-place path.
    tq_window: Option<scratchy_target_metal::turboquant_window::WindowMap>,
    /// Gated-DeltaNet recurrent-state pool for hybrid arches (Qwen3.5 /
    /// Qwen3-Next). `Some(_)` only when the loaded model's
    /// `gdn_runtime_config()` is `Some` (built in `initialize_cache`).
    /// The non-paged sibling of `kv_cache`: the metal forward binds its
    /// per-linear-layer conv/ssm buffers into the runtime through
    /// `ForwardCtx::gdn_state`.
    gdn_state: Option<GdnStatePool<scratchy_target_metal::PoolMem>>,
    /// Per-request GDN state-slot allocator (one slot per concurrently
    /// resident sequence, recycled on finish — the "degeneration after
    /// N requests" guard). `Some` iff the model is hybrid; drives the
    /// per-step `gdn_state_indices` / `gdn_is_fresh` the forward reads.
    gdn_slot_allocator: Option<GdnSlotAllocator>,
    /// Per-step GDN scratch: the `(state_indices, is_fresh)` vectors
    /// built in `execute_model` (one entry per batched sequence in
    /// cu_seqlens order) and consumed by the metal
    /// `forward_argmax_blocking` to populate `ForwardCtx::{gdn_state_indices,
    /// gdn_is_fresh}`. `None` between steps / for non-hybrid arches.
    gdn_pending: Option<(Vec<i32>, Vec<u32>)>,
    model_dir: Option<PathBuf>,
    hf_config: Option<HfModelConfig>,
    /// How `/v1/embeddings` turns this step's hidden states into one
    /// vector. Resolved once at load from `config.pooling_strategy`
    /// (`"auto"` consults the model's `1_Pooling/config.json`); read
    /// only on the `config.is_pooling` path.
    pooling_strategy: scratchy_core_model::embedding::PoolingStrategy,
    /// Draft model's resolved snapshot dir, populated when
    /// `config.draft_model_path` is set.
    #[allow(dead_code)]
    draft_model_dir: Option<PathBuf>,
    /// Draft model's `config.json`, populated alongside `draft_model_dir`.
    #[allow(dead_code)]
    draft_hf_config: Option<HfModelConfig>,
    /// Second KV cache pool, sized for the draft model. Lives on the
    /// same `MetalAllocator` / residency set as the target's pool —
    /// one allocator covers all weights + both pools.
    #[allow(dead_code)]
    draft_kv_cache: Option<KvCachePool>,
    model_dtype: GpuDType,
    resolved_architecture: Option<String>,
    is_shutdown: bool,
    token_buffers: HashMap<String, Vec<u32>>,
    /// Per-request prompt length (for discard_request_mask on intermediate prefill chunks).
    prompt_lengths: HashMap<String, usize>,
    /// Per-request block annotations for span-aware RoPE.
    annotation_buffers: HashMap<String, scratchy_core_common::BlockAnnotations>,
    /// Spans Phase 2: per-request logical block indices that are reused cache
    /// hits (shared via §a) → skip rewriting their KV (slot_mapping u32::MAX);
    /// the cached block already holds valid position-independent K.
    reused_block_buffers: HashMap<String, std::collections::HashSet<usize>>,
    /// Per-request multimodal data (images), only populated for requests
    /// scheduled at first as image-bearing.
    mm_data_buffers: HashMap<String, scratchy_core_common::MultimodalData>,
    sampling_params_map: HashMap<String, SamplingParams>,
    input_batch: InputBatch,
    preloaded_tokenizer: Option<tokenizers::Tokenizer>,
    /// Per-request seeded RNGs for deterministic sampling.
    seeded_rngs: HashMap<String, rand::rngs::StdRng>,
    /// Optional progress callback for startup initialization.
    #[allow(clippy::type_complexity)]
    progress_callback: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,

    // ---------------------------------------------------------------
    // Metal-only state (Phase F: cfg-mutex extension).
    // ---------------------------------------------------------------
    /// Shared `MetalDevice` handle. Set in `init_device`; carries
    /// `recommended_max_working_set_size` / `current_allocated_size`
    /// for `determine_available_memory`'s pre-load profile path.
    metal_device: Option<std::sync::Arc<scratchy_target_metal::device::MetalDevice>>,
    /// Device + command queue + caching allocator. Built lazily in
    /// `load_model` from `metal_device.device.clone()` once we know we
    /// will be loading weights through scratchy. Holds the `GpuWeights`
    /// allocator (via `Arc<MetalAllocator>` shared with the per-arch
    /// `Weights::load` upload path) plus the per-step CommandQueue.
    gpu_device: Option<GpuDevice>,
    /// Largest prefill bucket this device can afford, chosen at load time by
    /// `select_prefill_bucket` from the compiled ladder + the memory budget.
    /// Used to clamp the scheduler's `max_num_batched_tokens` so a single
    /// forward never exceeds the largest resident (pruned) bucket. `None` until
    /// `determine_available_memory` runs / on arches without a cost table.
    metal_prefill_bucket_max_m: Option<u32>,
    /// Loaded scratchy-forward-compiler weights — `Box<dyn ScratchyWeights>`
    /// dispatched through `try_load`. The trait `forward` body
    /// collapses to the per-canonical metal `forward` fn under
    /// cfg(metal); the worker calls it through the trait vtable.
    model: Option<Box<dyn scratchy_forward_compiler::ScratchyWeights>>,
    /// Vision tower for multimodal arches (Qwen3.5-VL). Loaded by
    /// `load_model` via `try_load_mm` alongside the text weights; `None`
    /// for text-only models / checkpoints without `visual.*` tensors.
    /// `execute_model` calls `mm.vision_forward` when the batch carries
    /// `mm_data` and parks the result in `mm_pending`.
    #[cfg(feature = "vision")]
    mm: Option<Box<dyn scratchy_target_metal::MultimodalForward>>,
    /// Vision-encoder output staged this step in `execute_model` for the
    /// upcoming TARGET forward: `(mm_embeds [n_img_tokens, hidden],
    /// embed_patches)`. Consumed (`.take()`) in `forward_argmax_blocking`
    /// into `ForwardCtx::{mm_embeds, embed_patches}` to drive the
    /// `Instruction::Embed` splice. Mirrors the `gdn_pending` hand-off.
    mm_pending: Option<(OwnedTensor, Vec<scratchy_target_metal::EmbedPatch>)>,
    /// Draft model for speculative decoding (loaded onto the same
    /// device + allocator + command queue as the target). Idle until
    /// the proposer is wired.
    draft_model: Option<Box<dyn scratchy_forward_compiler::ScratchyWeights>>,
    /// Compiled greedy-sampling pipeline. Cached once at load_model
    /// to avoid recompiling the MSL kernel each step.
    argmax_kernels: Option<scratchy_target_metal::argmax::ArgmaxKernels>,
    /// Compiled on-GPU token sampler (temperature / top-k / top-p / min-p +
    /// penalties). Cached once at load_model alongside `argmax_kernels`; used
    /// only for non-greedy requests (greedy decode stays on the argmax path).
    sampler_kernels: Option<scratchy_target_metal::sampling::SamplerKernels>,
    /// Terminal logits slot + its runtime column count, captured by the
    /// `forward_argmax_blocking` followup on the last TARGET forward. The
    /// sampler reads sampling requests' logits rows out of this buffer (on the
    /// GPU) after the forward completes; `None` until the first forward. The
    /// arena slot is a persistent buffer reused across forwards, so holding a
    /// retained clone is cheap and stays valid until the next forward overwrites
    /// it (which is after we sample this step).
    sampler_logits: Option<(scratchy_target_metal::mtl4_dispatch::Buffer, u32)>,
    /// This step's non-greedy sampler work, prepared (buffers + arg tables,
    /// pinned resident) BEFORE the target forward so the sampler can be encoded
    /// onto the forward's OWN command buffer (in the `forward_argmax_blocking`
    /// followup, after argmax) — one commit, one host wait, instead of a second
    /// `Mtl4DispatchBatch`. `Some` only between `execute_model`'s pre-forward
    /// setup and the followup that consumes it; `None` for greedy-only steps.
    pending_sampler: Option<scratchy_target_metal::sampling::PendingSampler>,
    /// Tokens the fused sampler produced this step, aligned to the `sample_jobs`
    /// order `execute_model` built. Written by `forward_argmax_blocking` after
    /// the (single) host wait, read + cleared by `execute_model`.
    fused_sampled: Option<Vec<u32>>,
    /// Per-request grammar FSM state for constrained / guided decoding
    /// (`guided_grammar` / `response_format`). Keyed by req_id; created
    /// the first time a request with a grammar is scheduled and dropped
    /// when the request finishes. Empty (zero overhead) otherwise.
    #[cfg(feature = "guided-decoding")]
    grammar_states: HashMap<String, scratchy_core_model::grammar::GrammarGuide>,
    /// Shared llguidance parser factory, lazily built from the model's
    /// `tokenizer.json` the first time a grammar request appears.
    #[cfg(feature = "guided-decoding")]
    grammar_factory: Option<std::sync::Arc<scratchy_core_model::grammar::LlgParserFactory>>,
    /// Compiled grammar-mask kernel (forces disallowed logits to -inf
    /// before argmax). Built alongside `argmax_kernels` at load_model.
    #[cfg(feature = "guided-decoding")]
    grammar_mask_kernels: Option<scratchy_target_metal::grammar_mask::GrammarMaskKernels>,
    /// This step's gathered grammar allow-masks (host-side), built in
    /// `execute_model` and consumed (`.take()`) in
    /// `forward_argmax_blocking` to dispatch the mask before argmax.
    #[cfg(feature = "guided-decoding")]
    grammar_pending: Option<GrammarMaskHost>,
    /// Persistent residency-pinned GPU buffers for the grammar mask, reused
    /// across decode steps (memcpy per step). They MUST be pinned in the
    /// residency set: a freshly-allocated, gpuAddress-bound bitset is evicted
    /// under KV memory pressure (large-vocab models like Qwen at long context),
    /// so the kernel would read garbage and the mask would silently fail. Grown
    /// (realloc + re-pin + commit) only when a batch's bitset/row count exceeds
    /// the current capacity; steady-state steps just memcpy into them.
    #[cfg(feature = "guided-decoding")]
    grammar_buf_allow: Option<scratchy_target_metal::grammar_mask::Buffer>,
    #[cfg(feature = "guided-decoding")]
    grammar_buf_rows: Option<scratchy_target_metal::grammar_mask::Buffer>,
    #[cfg(feature = "guided-decoding")]
    grammar_buf_gconsts: Option<scratchy_target_metal::grammar_mask::Buffer>,
    /// Capacities (in u32 words) of `grammar_buf_allow` / `grammar_buf_rows`.
    #[cfg(feature = "guided-decoding")]
    grammar_cap_allow: usize,
    #[cfg(feature = "guided-decoding")]
    grammar_cap_rows: usize,
    /// Phase 6 chain-advance kernel. One small kernel that bumps
    /// per-req `runtime.positions` / `slot_mapping` / `seqused_k` in
    /// place between K-step chain iters. Cached at load_model so the
    /// pipeline is built once.
    chain_advance_kernel: Option<scratchy_target_metal::chain_advance::ChainAdvanceKernel>,
    /// Second `MTLCommandQueue` on the same device, dedicated to the
    /// draft chain. Metal device-level parallelism: dispatches on
    /// distinct queues run concurrently on Apple Silicon when they
    /// don't contend for the same residency set / arena slots.
    ///
    /// Used by Phase 8: lockstep prefill submitted here in parallel
    /// with target verify on the main queue. Lockstep's draft-KV
    /// writes at target positions don't overlap with target's
    /// target-KV writes (separate KV pools), so they're safe to run
    /// concurrently. Allocated when the draft model loads.
    draft_queue: Option<
        ::objc2::rc::Retained<::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLCommandQueue>>,
    >,
    /// One `StorageModePrivate` MTLBuffer per (layer, K/V) for the target
    /// KV pool — `layer * 2 + kv_idx` indexing. The pool's logical chunks
    /// are byte offsets within these buffers; Apple's pager handles lazy
    /// physical commit on first GPU access. Always populated on the metal
    /// path (the alternative — one MTLBuffer per chunk — exposed too many
    /// distinct VA ranges to the GPU's UAT, costing ~5% per decode
    /// forward via the kernel's `chunk_table[chunk]` device load).
    target_kv_single_buffers: Vec<scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer>,
    /// Page-unified block size of the FULL KV-cache group (group 0) for hybrid
    /// SWA arches (gemma4: 32); equals `config.block_size` on uniform models.
    /// The worker encodes the full group's slot_mapping with this so it matches
    /// the kernel's baked `GLOBAL_BLOCK_SIZE`. Set in `init_cache`.
    kv_full_block_size: usize,
    /// `true` iff this worker runs a hybrid-SWA KV layout (gemma4): the shared
    /// pool spans multiple chunks and the fp16 sliding tensors must be grown to
    /// cover the live block set. Uniform pools (incl. uniform TurboQuant) are
    /// single-chunk and never grow. The right discriminator for both the grow
    /// gate and `max_chunks` (`kv_full_block_size` is nonzero on uniform too, so
    /// it can't distinguish them). Set in `init_cache`.
    kv_is_hybrid: bool,
    /// Same as `target_kv_single_buffers` for the draft pool. Required so
    /// the `AttentionViaCache` pipeline's `chunk_table[0]` always points
    /// at the layer base — both target's and draft's pipelines see the
    /// same kernel function constants (set process-wide by lowering).
    draft_kv_single_buffers: Vec<scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer>,
    /// When the target KV batch first went idle (no active requests), or `None`
    /// while a batch is active. The reactive shrink is deferred until the GPU
    /// has been idle for [`KV_SHRINK_IDLE_AFTER`]: dropping grown chunks frees
    /// their StorageModePrivate pages, which the next request would re-fault on
    /// first GPU touch — a multi-second stall on a long prefill. Deferring keeps
    /// back-to-back requests fast while still releasing memory when genuinely
    /// idle.
    #[cfg(feature = "metal")]
    kv_idle_since: Option<std::time::Instant>,
}

/// How long the target KV batch must stay idle before the reactive shrink
/// releases grown chunks back to the OS. Long enough that interactive,
/// back-to-back requests never pay the re-fault cost.
#[cfg(feature = "metal")]
const KV_SHRINK_IDLE_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

// Safety: MetalWorker contains raw GPU pointers (via GpuDevice, model weights,
// KV cache) and raw pointer fields in OwnedTensor. All GPU resources are
// allocated on a single device and accessed exclusively from the worker
// thread. Send is required because the worker is created on the main thread
// and moved to its dedicated worker thread via the executor's spawn.
#[cfg(feature = "metal")]
unsafe impl Send for MetalWorker {}

// `resolve_model_path` is backend-agnostic HF-hub plumbing; it lives in the
// neutral `worker_factory` module and is re-exported here for the gated
// worker load paths.
pub use crate::worker_factory::resolve_model_path;

/// Backend-neutral MRoPE position builders. Shared by the cuda and metal
/// worker paths — `build_mrope_positions_2d` is pure index math over
/// `PreparedInputs` + per-req patch grids, and `build_per_req_mm_seq_info_metal`
/// is the metal twin of the cuda `build_per_req_mm_seq_info` (it takes the
/// `MultimodalForward` handle directly rather than destructuring a
/// `CudaModel`, which is cuda-only).
#[cfg(feature = "metal")]
impl MetalWorker {
    /// Build `[3, n_tokens]` u32 MRoPE positions for an image-bearing
    /// batch. Mirrors Python vLLM's
    /// `Qwen2VLForConditionalGeneration.get_input_positions_tensor`:
    /// text tokens get `(cursor, cursor, cursor)` with `cursor`
    /// incrementing by 1; image tokens scan the post-spatial-merge
    /// grid in `(t, h, w)` row-major and emit `(st+t, st+h, st+w)`
    /// where `st` is the cursor at image entry; cursor advances by
    /// `max(grid_t, grid_h_merged, grid_w_merged)` after each image.
    ///
    /// Walks **every** seq position from 0 to `tokens_before + q_len`,
    /// advancing the cursor through preceding image patches even when
    /// they lie entirely in the cached prefix. Positions are written
    /// only for tokens in the current q range — but the cursor that
    /// gets written is the same one the encoder used to encode the
    /// cached KV, so `cached==prompt-1` (Bug 3) decodes correctly.
    /// Reqs with no patches get the linear cursor for every row —
    /// numerically identical to text-only 1D rope through the kernel's
    /// broadcast path, but in the 2D shape so a mixed batch (text req +
    /// image req) can share one positions tensor.
    #[cfg(feature = "vision")]
    fn build_mrope_positions_2d(
        prepared: &PreparedInputs,
        per_req_mm: &[Vec<SeqMmInfo>],
    ) -> Vec<u32> {
        let n_tokens = prepared.flat_positions.len();
        let mut t_row = vec![0u32; n_tokens];
        let mut h_row = vec![0u32; n_tokens];
        let mut w_row = vec![0u32; n_tokens];
        let meta = &prepared.attn_meta;

        for (req_idx, req_patches) in per_req_mm.iter().enumerate() {
            let req_start_batch = meta.query_start_loc[req_idx];
            let q_len = meta.q_lens[req_idx] as u32;
            let num_computed = meta.tokens_before[req_idx] as u32;
            let q_seq_end = num_computed + q_len;

            let mut cursor = 0u32;
            let mut seq_pos = 0u32;
            let mut patch_iter = req_patches.iter().peekable();

            while seq_pos < q_seq_end {
                if let Some(&&(s_off, length, gt, mh, mw)) = patch_iter.peek()
                    && s_off == seq_pos
                {
                    let st = cursor;
                    let stride_hw = mh * mw;
                    for img_idx in 0..length {
                        let pos_seq = seq_pos + img_idx;
                        if pos_seq >= num_computed && pos_seq < q_seq_end {
                            let local_q = (pos_seq - num_computed) as usize;
                            let flat_idx = req_start_batch + local_q;
                            let t = img_idx / stride_hw;
                            let rem = img_idx % stride_hw;
                            let h = rem / mw;
                            let w = rem % mw;
                            t_row[flat_idx] = st + t;
                            h_row[flat_idx] = st + h;
                            w_row[flat_idx] = st + w;
                        }
                    }
                    cursor = st + gt.max(mh).max(mw);
                    seq_pos += length;
                    patch_iter.next();
                    continue;
                }
                if seq_pos >= num_computed {
                    let local_q = (seq_pos - num_computed) as usize;
                    let flat_idx = req_start_batch + local_q;
                    t_row[flat_idx] = cursor;
                    h_row[flat_idx] = cursor;
                    w_row[flat_idx] = cursor;
                }
                cursor += 1;
                seq_pos += 1;
            }
        }

        let mut out = Vec::with_capacity(3 * n_tokens);
        out.extend_from_slice(&t_row);
        out.extend_from_slice(&h_row);
        out.extend_from_slice(&w_row);
        out
    }

    /// Metal twin of [`Self::build_per_req_mm_seq_info`]: builds per-req
    /// seq-space MM info from the `MultimodalForward` handle directly
    /// (the metal worker holds `self.mm: Option<Box<dyn MultimodalForward>>`,
    /// not a cuda-only `CudaModel`). Same tuple/semantics: empty inner
    /// `Vec` for reqs without mm_data, all-empty outer `Vec` (caller
    /// falls back to 1D positions) for a text-only batch.
    #[cfg(all(feature = "metal", feature = "vision"))]
    fn build_per_req_mm_seq_info_metal(
        mm: &dyn scratchy_target_metal::MultimodalForward,
        mm_data_buffers: &HashMap<String, scratchy_core_common::MultimodalData>,
        prepared: &PreparedInputs,
    ) -> Vec<Vec<SeqMmInfo>> {
        let mut per_req: Vec<Vec<SeqMmInfo>> = vec![Vec::new(); prepared.req_inputs.len()];
        let mut any = false;
        for (i, req) in prepared.req_inputs.iter().enumerate() {
            let Some(mm_data) = mm_data_buffers.get(&req.req_id) else {
                continue;
            };
            if mm_data.images.is_empty() {
                continue;
            }
            let pixel_inputs: Vec<scratchy_target_metal::PixelInput<'_>> = mm_data
                .images
                .iter()
                .map(|img| scratchy_target_metal::PixelInput {
                    pixels: &img.pixels,
                    height: img.height as u32,
                    width: img.width as u32,
                })
                .collect();
            let grids = mm.embed_patch_grids(&pixel_inputs);
            for (ph, &(gt, mh, mw)) in mm_data.image_placeholders.iter().zip(grids.iter()) {
                per_req[i].push((ph.offset as u32, ph.length as u32, gt, mh, mw));
            }
            per_req[i].sort_by_key(|t| t.0);
            any = true;
        }
        if !any {
            return Vec::new();
        }
        per_req
    }
}

/// Per-block KV bytes for one model: layers × 2 (K+V) × heads × head_dim × block_size × 2 bytes.
/// bf16 and f16 are both 2 bytes/elt under metal; int4 KV is unsupported.
#[cfg(feature = "metal")]
fn kv_per_block_bytes(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    block_size: usize,
) -> usize {
    let elt_bytes: usize = 2;
    // Hybrid-attention-geometry arches (Gemma4) size each layer by its
    // own kv_heads*head_dim; uniform arches use the single product.
    // Mis-summing here mis-sizes the scheduler's KV budget.
    if let Some(per_layer) = model.per_layer_kv_token_elems() {
        return per_layer
            .iter()
            .map(|e| e.saturating_mul(2).saturating_mul(block_size))
            .sum::<usize>()
            .saturating_mul(elt_bytes);
    }
    (model.num_hidden_layers() as usize)
        .saturating_mul(2)
        .saturating_mul(model.num_key_value_heads() as usize)
        .saturating_mul(model.head_dim() as usize)
        .saturating_mul(block_size)
        .saturating_mul(elt_bytes)
}

// `compute_available_kv_bytes`, `PrefillBucketSelection`, and
// `select_prefill_bucket` are backend-neutral; they live in
// `scratchy-serving-engine` so the CUDA worker (in `scratchy-serving-cuda`)
// can reach them without a dependency cycle. Re-exported here so callers
// naming `gpu_worker::{...}` keep resolving.
pub use scratchy_serving_engine::gpu_budget::{
    PrefillBucketSelection, compute_available_kv_bytes, select_prefill_bucket,
};

// ---------------------------------------------------------------------------
// Metal arm (Phase F: cfg-mutex extension; Step 2 = instantiate-only).
// ---------------------------------------------------------------------------
//
// The metal `new()` constructor + `Worker` impl live in their own
// cfg-gated blocks rather than as `#[cfg]` arms inside the cuda
// constructor / cuda Worker impl. The cuda body is ~3500 lines that
// call cuda-only helpers (CudaModel dispatch, CudaGraphRunner, NCCL
// PP, FP8 scales, LogitsProcessorPipeline, GdnStatePool, …); none
// of those compile under metal, so wrapping each cuda body with an
// `#[cfg(feature = "cuda")] { … }` arm would still drag the
// dead-under-metal code through type-checking. Per the Phase F
// rules, the MetalWorker struct stays one struct with cfg-mutex'd
// fields; the trait impl uses cfg-mutex'd blocks for the same reason
// the macro emission uses cfg-mutex'd `Weights` blocks.
//
// Step 2 wires `init_device` / `shutdown` / `rank` etc. and stubs the
// heavy lifecycle methods (`load_model`, `initialize_cache`,
// `execute_model`, `compile_or_warm_up_model`,
// `determine_available_memory`) so scratchy-serving-api can instantiate a
// MetalWorker under metal. Step 3 fills those bodies in via the
// per-arch `metal_pool()` factory + `MetalWorkerPool::forward`.

// `gdn_slot_key` is backend-neutral; it lives in `scratchy-serving-engine`
// (cycle-free for both the CUDA and Metal workers) and is re-exported here.
pub use scratchy_serving_engine::gpu_budget::gdn_slot_key;

#[cfg(feature = "metal")]
impl MetalWorker {
    pub fn new(config: WorkerCreateConfig) -> Self {
        Self {
            config,
            kv_cache: None,
            turboquant: None,
            tq_window: None,
            gdn_state: None,
            gdn_slot_allocator: None,
            gdn_pending: None,
            model_dir: None,
            hf_config: None,
            pooling_strategy: scratchy_core_model::embedding::PoolingStrategy::Last,
            // Metal forward currently produces f16 outputs; cuda's "auto"
            // resolves to bf16 from the checkpoint, but scratchy-target-metal's
            // shaders are specialized on f16 and the worker only needs
            // this dtype to size KV cache slots, which Step 3 will revise.
            model_dtype: GpuDType::F16,
            resolved_architecture: None,
            is_shutdown: false,
            token_buffers: HashMap::new(),
            prompt_lengths: HashMap::new(),
            annotation_buffers: HashMap::new(),
            reused_block_buffers: HashMap::new(),
            mm_data_buffers: HashMap::new(),
            sampling_params_map: HashMap::new(),
            input_batch: InputBatch::new(),
            preloaded_tokenizer: None,
            seeded_rngs: HashMap::new(),
            progress_callback: None,
            metal_device: None,
            gpu_device: None,
            #[cfg(feature = "metal")]
            metal_prefill_bucket_max_m: None,
            model: None,
            #[cfg(feature = "vision")]
            mm: None,
            mm_pending: None,
            draft_model: None,
            draft_model_dir: None,
            draft_hf_config: None,
            draft_kv_cache: None,
            argmax_kernels: None,
            sampler_kernels: None,
            sampler_logits: None,
            pending_sampler: None,
            fused_sampled: None,
            #[cfg(feature = "guided-decoding")]
            grammar_states: HashMap::new(),
            #[cfg(feature = "guided-decoding")]
            grammar_factory: None,
            #[cfg(feature = "guided-decoding")]
            grammar_mask_kernels: None,
            #[cfg(feature = "guided-decoding")]
            grammar_pending: None,
            #[cfg(feature = "guided-decoding")]
            grammar_buf_allow: None,
            #[cfg(feature = "guided-decoding")]
            grammar_buf_rows: None,
            #[cfg(feature = "guided-decoding")]
            grammar_buf_gconsts: None,
            #[cfg(feature = "guided-decoding")]
            grammar_cap_allow: 0,
            #[cfg(feature = "guided-decoding")]
            grammar_cap_rows: 0,
            chain_advance_kernel: None,
            draft_queue: None,
            target_kv_single_buffers: Vec::new(),
            kv_full_block_size: 0,
            kv_is_hybrid: false,
            draft_kv_single_buffers: Vec::new(),
            #[cfg(feature = "metal")]
            kv_idle_since: None,
        }
    }

    /// Set the progress callback for startup loading.
    pub fn set_progress_callback(&mut self, cb: std::sync::Arc<dyn Fn(&str) + Send + Sync>) {
        self.progress_callback = Some(cb);
    }

    /// Expose the HF config after load_model. Mirrors the cuda-side
    /// accessor — scratchy-serving-api reads this to drive `compute_num_blocks`
    /// and the cache-init path.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    /// Expose the model directory after load_model.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Lazily build the shared llguidance parser factory from the model's
    /// `tokenizer.json`. Called the first time a request with a grammar is
    /// scheduled, so models that never use constrained decoding pay
    /// nothing. Idempotent: a no-op once the factory exists. On failure
    /// (missing/invalid tokenizer.json) the factory stays `None` and
    /// grammar requests fall back to unconstrained decoding with a logged
    /// warning rather than failing the whole batch.
    #[cfg(feature = "guided-decoding")]
    fn ensure_grammar_factory(&mut self) {
        if self.grammar_factory.is_some() {
            return;
        }
        let Some(model_dir) = self.model_dir.as_ref() else {
            tracing::warn!("guided-decoding: model_dir unset; cannot build grammar parser factory");
            return;
        };
        let tokenizer_path = model_dir.join("tokenizer.json");
        let bytes = match std::fs::read(&tokenizer_path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "guided-decoding: read {} failed: {e}; grammar requests will be unconstrained",
                    tokenizer_path.display()
                );
                return;
            }
        };
        match scratchy_core_model::grammar::build_parser_factory(&bytes) {
            Ok(factory) => {
                info!("guided-decoding: built grammar parser factory from tokenizer.json");
                self.grammar_factory = Some(factory);
            }
            Err(e) => {
                tracing::warn!("guided-decoding: build_parser_factory failed: {e}");
            }
        }
    }

    /// Runtime per-sequence block-table capacity (the KV pool's
    /// `max_blocks_per_seq`): the number of paged blocks one sequence can span,
    /// derived from the resolved `max_model_len` rather than the compile-time
    /// `MAX_BLOCKS_PER_SEQ` (default 128 ≈ 2k tokens, which silently truncated
    /// long context). `min`-capped at `num_blocks` (the pool capacity actually
    /// allocated) and floored at 1. The same value must drive the kernel's
    /// `MaxBlocksPerSeq` function constant, the host block-table row stride, and
    /// the rope-once scratch — all three read the pool's stored field so they
    /// agree. `max_model_len` resolves the same way as `load_model`:
    /// CLI override → HF `max_position_embeddings` → 4096.
    fn kv_block_cap(&self, num_blocks: usize) -> usize {
        let max_model_len = self
            .config
            .max_model_len
            .or_else(|| {
                self.hf_config
                    .as_ref()
                    .and_then(|c| c.max_position_embeddings())
            })
            .unwrap_or(4096);
        max_model_len
            .div_ceil(self.config.block_size.max(1))
            .min(num_blocks)
            .max(1)
    }

    /// Bytes per element for the model's KV cache dtype. Metal forces
    /// f16 for both compute and cache, so this is a constant; we keep
    /// the same accessor name as the cuda side so scratchy-serving-api treats
    /// the worker uniformly.
    pub fn resolved_dtype_elem_bytes(&self) -> usize {
        self.model_dtype.size_bytes()
    }
}

// Inherent impl block for metal-only helpers that aren't on the
// `Worker` trait — kept separate from the trait impl below so non-
// trait methods don't get implicitly added to `Worker`.
#[cfg(feature = "metal")]
impl MetalWorker {
    /// Spans (rope-on-read): is it SAFE to OR bit 31 into the block_table
    /// / slot_mapping for this run? Only when the loaded model's kernels
    /// mask bit 31 (`W::ROPE_ON_READ`, surfaced via `ScratchyWeights`). A
    /// non-rope-on-read model reads block_table raw, so a stray bit 31
    /// would corrupt the physical block id.
    fn rope_on_read_active(&self) -> bool {
        self.model.as_ref().is_some_and(|m| m.rope_on_read())
    }

    /// Run the vision tower for every MM-bearing request at its first
    /// prefill step and return `(mm_embeds [n_img_tokens, hidden],
    /// embed_patches)`. Metal twin of the cuda `run_mm_vision_forward`
    /// (which destructures a `CudaModel`); here the `mm` handle is passed
    /// in directly from `self.mm`. `None` when no batched request carries
    /// pixel data this step (text-only / decode steps).
    ///
    /// # Safety
    /// `mm.vision_forward` issues GPU work on `device`; callers must hold
    /// the device for the duration. `pixels`/`placeholders` borrow the
    /// per-request `mm_data` which outlives this call.
    #[cfg(feature = "vision")]
    unsafe fn run_mm_vision_forward_metal(
        mm: &dyn scratchy_target_metal::MultimodalForward,
        mm_data_buffers: &HashMap<String, scratchy_core_common::MultimodalData>,
        prepared: &PreparedInputs,
        device: &mut GpuDevice,
    ) -> Option<(OwnedTensor, Vec<scratchy_target_metal::EmbedPatch>)> {
        let meta = &prepared.attn_meta;
        let mut pixel_inputs: Vec<scratchy_target_metal::PixelInput<'_>> = Vec::new();
        let mut placeholders: Vec<scratchy_target_metal::EmbedPatch> = Vec::new();
        for (i, req) in prepared.req_inputs.iter().enumerate() {
            let Some(mm_data) = mm_data_buffers.get(&req.req_id) else {
                continue;
            };
            // Only run on the first prefill chunk; later steps consume the
            // already-spliced KV cache.
            if meta.tokens_before[i] != 0 {
                continue;
            }
            let batch_offset = meta.query_start_loc[i] as u32;
            for (img, ph) in mm_data.images.iter().zip(mm_data.image_placeholders.iter()) {
                pixel_inputs.push(scratchy_target_metal::PixelInput {
                    pixels: &img.pixels,
                    height: img.height as u32,
                    width: img.width as u32,
                });
                placeholders.push(scratchy_target_metal::EmbedPatch {
                    token_offset: batch_offset + ph.offset as u32,
                    length: ph.length as u32,
                    // Filled by `vision_forward` from per-image grid_thw.
                    grid_t: 0,
                    grid_h_merged: 0,
                    grid_w_merged: 0,
                });
            }
        }
        if pixel_inputs.is_empty() {
            return None;
        }
        let (out, patches) = unsafe {
            mm.vision_forward(
                &pixel_inputs,
                &placeholders,
                scratchy_target_metal::ForwardDeviceHandle::new(device),
            )
        };
        Some((out, patches))
    }

    /// Load the speculative draft model onto the same Metal device +
    /// allocator + command queue as the target. Called from
    /// `load_model` after the target is fully loaded.
    ///
    /// Skips the metal device setup (reuses `self.gpu_device`), the
    /// argmax kernel compile (per-device, not per-model — `self.argmax_*`
    /// stays target-aligned for batch sizing), and the worker-pool /
    /// SpecializedPipelineCache for the draft is created lazily inside
    /// `scratchy_forward_compiler::try_load`. Two ScratchyWeights instances on the
    /// same device → two pipeline caches; that's fine because cache
    /// keys include the library name and each model's synthesized
    /// libraries are derived from its own per-arch macro expansion.
    fn load_draft_model_metal(&mut self) -> ExecutorResult<()> {
        let path = self.config.draft_model_path.as_deref().ok_or_else(|| {
            ExecutorError::WorkerInit(
                "load_draft_model_metal called without draft_model_path".into(),
            )
        })?;

        let t_resolve = std::time::Instant::now();
        let draft_dir = resolve_model_path(path, self.config.hf_token.as_deref(), None, None)?;
        info!(
            "ScratchyWorker(metal): resolved draft model dir in {:?} ({})",
            t_resolve.elapsed(),
            draft_dir.display()
        );

        if draft_dir.is_file() && draft_dir.extension().is_some_and(|e| e == "gguf") {
            return Err(ExecutorError::WorkerInit(
                "draft GGUF not supported on metal (safetensors only)".into(),
            ));
        }

        let draft_hf_config = HfModelConfig::from_path(&draft_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("draft config parse failed: {e}")))?;
        let draft_arch = draft_hf_config
            .architectures
            .first()
            .cloned()
            .unwrap_or_default();

        // Reuse the target's gpu_device — its allocator owns the
        // residency set the target weights live in, and the draft must
        // share that set or attention reads will race the Apple pager.
        // `MetalAllocator::clone` shares the underlying arena via
        // Arc<Mutex<…>>, so a weight allocated through the draft clone
        // is reachable via the target's clone too.
        let gpu_device = self.gpu_device.as_ref().ok_or_else(|| {
            ExecutorError::WorkerInit(
                "gpu_device not initialized — draft load must run after target load".into(),
            )
        })?;
        let allocator = (*gpu_device.allocator).clone();

        let t_from_dir = std::time::Instant::now();
        let mut draft_weights = GpuWeights::from_dir(&draft_dir, allocator)
            .map_err(|e| ExecutorError::WorkerInit(format!("draft weight load failed: {e}")))?;
        info!(
            "ScratchyWorker(metal): draft GpuWeights::from_dir in {:?} ({} tensors)",
            t_from_dir.elapsed(),
            draft_weights.len()
        );
        draft_weights.set_target_dtype(GpuDType::BF16);

        let hf_fp = scratchy_forward_compiler::HfFingerprint {
            rope_scaling_type: draft_hf_config
                .extra
                .get("rope_scaling")
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str()),
            rope_scaling_hash: draft_hf_config
                .extra
                .get("rope_scaling")
                .map(scratchy_forward_compiler::hash_json_value),
            rope_theta: draft_hf_config.rope_theta,
        };
        let max_model_len = self
            .config
            .max_model_len
            .or(draft_hf_config.max_position_embeddings())
            .unwrap_or(4096);

        let t_try_load = std::time::Instant::now();
        let draft_model = scratchy_forward_compiler::try_load(
            &mut draft_weights,
            (),
            draft_arch.as_str(),
            1,
            0,
            max_model_len,
            hf_fp,
        )
        .map_err(|e| ExecutorError::WorkerInit(format!("draft try_load: {e}")))?
        .ok_or_else(|| ExecutorError::ArchNotSupported(draft_arch.clone()))?;

        info!(
            "ScratchyWorker(metal): draft try_load in {:?} ({} via scratchy-forward-compiler, {})",
            t_try_load.elapsed(),
            draft_arch,
            draft_model.arch_name()
        );

        self.draft_model = Some(draft_model);
        self.draft_model_dir = Some(draft_dir);
        self.draft_hf_config = Some(draft_hf_config);

        // Phase 8 foundation: dedicated MTLCommandQueue for the draft
        // chain. Two queues on the same device run concurrently on
        // Apple Silicon; this enables lockstep prefill to overlap
        // with target verify (separate KV pools → no contention).
        if self.draft_queue.is_none() {
            let device = self
                .gpu_device
                .as_ref()
                .expect("gpu_device must be initialized before draft model loads");
            self.draft_queue = Some(
                device
                    .device
                    .newCommandQueue()
                    .expect("newCommandQueue for draft_queue returned nil"),
            );
            info!("ScratchyWorker(metal): allocated dedicated draft MTLCommandQueue");
        }
        Ok(())
    }

    /// Allocate the draft model's KV pool. Mirrors `initialize_cache`'s
    /// target-pool path: StorageModePrivate buffers pinned into the
    /// allocator's shared residency set so attention reads don't race
    /// the Apple pager.
    /// Reactive KV (2b): grow the chunked target pool so every block id
    /// `<= max_block` is backed by a resident chunk before a forward
    /// derefs the chunk-address table. Allocates missing chunks as
    /// StorageModePrivate (residency-inserted) and commits the residency
    /// set once if anything grew. No-op for non-chunked pools.
    fn grow_metal_kv_to_cover(&mut self, max_block: usize) {
        // TurboQuant on a UNIFORM model never reads the fp16 KV pool (every layer
        // uses the packed store + the one-layer fp16 scratch, both provisioned by
        // the RuntimeFactory). Skipping growth keeps the pool at its 1-chunk seed
        // — THE memory win: per-layer fp16 KV is replaced by the ~4.7x smaller
        // packed store + a single reused scratch.
        //
        // 🛑 BUT a HYBRID SWA model (gemma4) only TurboQuant-compresses its GLOBAL
        // layers; its FP16 SLIDING layers DO read the fp16 pool. Skipping growth
        // there leaves chunks past the 1-chunk seed un-committed, so once a sliding
        // group's blocks cross BLOCKS_PER_CHUNK (128) the sliding K-write/read hits
        // a stale chunk-address-table entry → KV corruption (garbage past ~384
        // tokens for gemma4-12b, where total blocks = 8 + 5·ceil(tok/16) ≥ 128).
        // Uniform TurboQuant keeps a single-chunk pool that's never read here, so
        // skip growth there; hybrid SWA must grow its fp16 sliding tensors. Gate
        // on `kv_is_hybrid` (the same discriminator `max_chunks` uses) — NOT
        // `kv_full_block_size`, which is nonzero on uniform too.
        if self.config.kv_cache_dtype == "turboquant" && !self.kv_is_hybrid {
            return;
        }
        let Some(device) = self.gpu_device.as_ref() else {
            return;
        };
        // Hold the residency set so we can make the chunk(s) that `grow_to_cover`
        // is about to allocate GPU-resident. A freshly-allocated MTLBuffer chunk
        // is NOT resident until it is committed to the residency set, and the GPU
        // reads it through the bindless chunk-address table on the next dispatch.
        let residency = device.allocator.residency().clone();
        // `grow_to_cover` allocates K+V for each PHYSICAL TENSOR (group-shared:
        // gemma4 has `num_tensors` < `num_layers`), so the slot cycle is
        // `2 * num_tensors` — matching `target_kv_single_buffers.len()` and the
        // init-time `alloc_chunk` closure. Using `num_layers` here would index
        // past the buffer vec on the hybrid pool.
        let n_slots = self
            .kv_cache
            .as_ref()
            .map(|kv| kv.num_tensors * 2)
            .unwrap_or(0);
        if n_slots == 0 {
            return;
        }
        let single_buf_ref = &mut self.target_kv_single_buffers;
        let Some(kv) = self.kv_cache.as_mut() else {
            return;
        };
        let alloc_chunk_counter = std::cell::Cell::new(0usize);
        // Returns Ok(n) where n = number of NEW chunks allocated to reach
        // `max_block` (Ok(0) = the pool already covered it); Err if a chunk
        // allocation or commit failed.
        let grew_result = kv.grow_to_cover(
            max_block,
            |bytes| {
                let c = alloc_chunk_counter.get();
                alloc_chunk_counter.set(c + 1);
                let slot = c % n_slots;
                let layer = &mut single_buf_ref[slot];
                let chunk_idx = layer.committed_chunks();
                layer
                    .commit_through(chunk_idx + 1)
                    .map_err(|e| anyhow::anyhow!("KV grow slot={slot} chunk={chunk_idx}: {e}"))?;
                Ok(MetalMem::from_buffer_with_offset(
                    layer.buffer_clone(),
                    chunk_idx * layer.chunk_bytes(),
                    bytes,
                ))
            },
            |m| m.gpu_address(),
        );
        // If we allocated new chunk(s), make their pages GPU-resident before the
        // next dispatch reads them. WHY THIS MATTERS: on Apple Silicon a GPU read
        // of un-wired (non-resident) device memory does not raise a recoverable
        // fault — it silently hangs the command buffer, and the watchdog cannot
        // reclaim it (requires a reboot). That is the bug this commit fixes:
        // previously the grow result was discarded and the new chunk was never
        // committed, so any prompt that crossed a chunk boundary (~128 blocks)
        // could wedge the GPU.
        match &grew_result {
            Ok(0) => {}                  // nothing grew — already covered, skip the commit
            Ok(_) => residency.commit(), // wire the newly-allocated chunk pages
            Err(e) => tracing::error!("KV grow_to_cover failed (max_block={max_block}): {e}"),
        }
    }

    /// Reactive KV (2c): when the batch is fully idle (no live blocks), drop
    /// the per-layer commit-counter back to 1 chunk. The `MetalMem` chunk
    /// handles are clones of the layer buffer; dropping them does NOT free
    /// memory. Apple's pager evicts physical pages for untouched offsets
    /// under memory pressure. Only call when `input_batch.num_active() == 0`
    /// so no in-flight forward references a dropped chunk index.
    fn shrink_metal_kv_idle(&mut self) {
        let freed = match self.kv_cache.as_mut() {
            Some(kv) => kv.shrink_to_chunks(1),
            None => return,
        };
        if freed.is_empty() {
            return;
        }
        for layer in self.target_kv_single_buffers.iter_mut() {
            let _ = layer.shrink_to(1);
        }
        let n = freed.len();
        drop(freed);
        info!(
            "ScratchyWorker(metal): reactive KV shrink — {} chunk handles dropped on {} layers",
            n,
            self.target_kv_single_buffers.len()
        );
    }

    fn initialize_draft_cache_metal(&mut self, num_gpu_blocks: usize) -> ExecutorResult<()> {
        let model = self.draft_model.as_ref().ok_or_else(|| {
            ExecutorError::WorkerInit(
                "initialize_draft_cache_metal called before draft_model load".into(),
            )
        })?;
        // The lockstep proposer needs a
        // 1:1 mirror — every target block has a sibling draft block at the
        // same index. `determine_available_memory` already carved off
        // `draft / (target + draft)` of the KV budget for us, so the
        // engine's `num_gpu_blocks` is the post-split target count and
        // the draft can mirror it exactly.
        let draft_blocks = num_gpu_blocks;
        // Runtime per-sequence block-table capacity for the draft pool (mirrors
        // the target: same max_model_len, capped at the draft block count).
        let draft_block_cap = self.kv_block_cap(draft_blocks);
        let device = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("gpu_device not initialized".into()))?;

        let cache_dtype = match model.metal_dtype() {
            scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => GpuDType::BF16,
            scratchy_target_metal::interpreter::metal::MetalDtype::F16 => GpuDType::F16,
            scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                return Err(ExecutorError::WorkerInit(
                    "draft KV cache cannot be int4-quantized".into(),
                ));
            }
        };

        let mtl_device = device.device.clone();
        let residency = device.allocator.residency().clone();

        // The `attention_via_cache_v2_*` kernel's BPC=0 fast path has an
        // unresolved interaction with the draft K-step chain on 3B-class+
        // models (out-of-vocab token IDs from the draft proposer). The
        // chunked-addressing path produces coherent output. Force it now
        // — single-buffer backing still applies; only the kernel's per-
        // block addressing differs.
        scratchy_target_metal::interpreter::metal::lowering::force_chunked_attention_addressing();
        info!(
            "ScratchyWorker(metal): draft model loaded — engaging chunked attention \
             addressing (BPC=0 fast path off for spec-decode safety)"
        );

        let t_pool = std::time::Instant::now();
        let blocks_per_chunk = scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize;
        let num_layers_draft = model.num_hidden_layers() as usize;
        let num_chunks_total_draft = draft_blocks.div_ceil(blocks_per_chunk);
        let per_block_elems_draft = (model.num_key_value_heads() as usize)
            * self.config.block_size
            * (model.head_dim() as usize);
        let chunk_bytes_logical_draft =
            blocks_per_chunk * per_block_elems_draft * cache_dtype.size_bytes();

        let mut draft_single_buf_layers: Vec<
            scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer,
        > = Vec::with_capacity(num_layers_draft * 2);
        for _ in 0..(num_layers_draft * 2) {
            let layer = scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer::new(
                &mtl_device,
                chunk_bytes_logical_draft,
                num_chunks_total_draft,
            )
            .map_err(|e| ExecutorError::WorkerInit(format!("draft SingleBufferKvLayer: {e}")))?;
            residency.insert(layer.buffer());
            draft_single_buf_layers.push(layer);
        }
        info!(
            "ScratchyWorker(metal): draft KV layer buffers — {} buffers",
            draft_single_buf_layers.len()
        );

        let pool = unsafe {
            // Layer-major iteration: outer for layer, inner for chunk.
            // Draft uses `initial_chunks = num_chunks_total` (eager: the
            // spec-decode draft forward path has no growth hook). The slot
            // formula `(c / (2 * chunks_per_layer)) * 2 + c % 2` matches
            // the call sequence L0_K_c0, L0_V_c0, L0_K_c1, L0_V_c1, ...
            let alloc_chunk_counter = std::cell::Cell::new(0usize);
            let draft_single_ref = &mut draft_single_buf_layers;
            let chunks_per_layer = num_chunks_total_draft;
            KvCachePool::new_metal_chunked(
                num_layers_draft,
                draft_blocks,
                self.config.block_size,
                model.num_key_value_heads() as usize,
                model.head_dim() as usize,
                draft_block_cap,
                // Draft pool stays uniform (matching the uniform
                // SingleBufferKvLayer sizing above): spec-decode is not
                // enabled for hybrid-attention-geometry arches (Gemma4).
                None,
                // One physical tensor per layer (no group sharing on draft).
                None,
                cache_dtype,
                blocks_per_chunk,
                usize::MAX,
                |bytes| {
                    let c = alloc_chunk_counter.get();
                    alloc_chunk_counter.set(c + 1);
                    let kv = c % 2;
                    let layer_idx = c / (2 * chunks_per_layer);
                    let slot = layer_idx * 2 + kv;
                    let layer = &mut draft_single_ref[slot];
                    let chunk_idx = layer.committed_chunks();
                    layer.commit_through(chunk_idx + 1).map_err(|e| {
                        anyhow::anyhow!("draft KV commit slot={slot} chunk={chunk_idx}: {e}")
                    })?;
                    Ok(MetalMem::from_buffer_with_offset(
                        layer.buffer_clone(),
                        chunk_idx * layer.chunk_bytes(),
                        bytes,
                    ))
                },
                |bytes| {
                    let buffer = mtl_device
                        .newBufferWithLength_options(
                            bytes,
                            ::objc2_metal::MTLResourceOptions::StorageModeShared,
                        )
                        .expect("newBufferWithLength_options returned nil");
                    residency.insert(&buffer);
                    Ok(MetalMem::from_buffer(buffer))
                },
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("draft KvCachePool: {e}")))?;
        self.draft_kv_single_buffers = draft_single_buf_layers;
        pool.fill_chunk_tables(|m| m.gpu_address());
        info!(
            "ScratchyWorker(metal): draft KV pool ready in {:?} ({} layers × {} blocks × {} tokens, \
             {} blocks/chunk, 1:1 mirror of target's {} blocks)",
            t_pool.elapsed(),
            model.num_hidden_layers(),
            draft_blocks,
            self.config.block_size,
            blocks_per_chunk,
            num_gpu_blocks,
        );
        self.draft_kv_cache = Some(pool);
        Ok(())
    }

    /// Allocate a `StorageModeShared` u32 buffer and memcpy `data` into it.
    /// Shared between the verify path (via the `SpecDecodeBackend` trait
    /// impl) and the K-step draft chain inlined in `execute_model`.
    fn alloc_shared_u32_buf(
        device: &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLDevice>,
        data: &[u32],
    ) -> ::objc2::rc::Retained<::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>> {
        let bytes = (data.len().max(1)) * 4;
        let buf = device
            .newBufferWithLength_options(
                bytes,
                ::objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithLength_options returned nil");
        if !data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buf.contents().as_ptr() as *mut u32,
                    data.len(),
                );
            }
        }
        buf
    }

    /// Run the on-GPU token sampler for a set of non-greedy decode rows and
    /// return one sampled token per job (in job order).
    ///
    /// `jobs[j] = (result_index, logits_row)`: `logits_row` is the request's
    /// sample position in the logits buffer captured this step by
    /// `forward_argmax_blocking` (`sample_indices[i]`); `result_index` is the
    /// caller's slot in `sampled_token_ids` (used only by the caller). `req_ids`
    /// is the full per-batch request-id list (indexed by `result_index`).
    ///
    /// Pipeline (mirrors cuda's f32 sampling path): gather each row's logits
    /// (f16/bf16) into a compact f32 scratch, apply repetition / frequency /
    /// presence penalties (only when a row needs them), then top-k/top-p/min-p
    /// sample. All three stages ride ONE MTL4 command buffer with dispatch→
    /// dispatch barriers between the dependent kernels; one host wait at commit.
    fn prepare_gpu_sampler(
        &mut self,
        jobs: &[(usize, u32)],
        req_ids: &[String],
    ) -> ExecutorResult<scratchy_target_metal::sampling::PendingSampler> {
        // `vocab` is the lm_head logits width (== METAL_VOCAB_SIZE == config
        // vocab_size). Take it + the compute dtype from the model so the sampler
        // is preparable BEFORE this step's forward; the logits BUFFER itself is
        // bound later, in the forward's followup (it's the forward's own output).
        let (is_bf16, vocab) = {
            let model = self.model.as_deref().ok_or_else(|| {
                ExecutorError::WorkerExecution("gpu sampler: model not loaded".into())
            })?;
            let is_bf16 = match model.metal_dtype() {
                scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => true,
                scratchy_target_metal::interpreter::metal::MetalDtype::F16 => false,
                scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                    return Err(ExecutorError::WorkerExecution(
                        "gpu sampler: int4 logits dtype unsupported".into(),
                    ));
                }
            };
            (is_bf16, model.vocab_size() as u32)
        };
        let device = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("gpu sampler: no gpu_device".into()))?
            .device
            .clone();

        let residency = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("gpu sampler: no gpu_device".into()))?
            .allocator
            .residency()
            .clone();

        // Assemble this step's sampler inputs (metal's `GpuSampleParams` layout)
        // and hand them to the metal sampler. The per-request seed inside comes
        // from the shared `scratchy_core_common::fnv_seed`, so it matches cuda;
        // the format-assembly + all MTL4 detail live in `scratchy_target_metal`.
        let params = scratchy_target_metal::sampling::gather_gpu_sample_params(
            jobs,
            req_ids,
            &self.sampling_params_map,
            &mut self.seeded_rngs,
            &self.prompt_lengths,
            &self.token_buffers,
            vocab,
        );
        Ok(scratchy_target_metal::sampling::PendingSampler::prepare(
            &device,
            &residency,
            &params,
            jobs.len() as u32,
            vocab,
            is_bf16,
        ))
    }
}

/// Phase 6 K-step chain dispatch — the actual work of
/// `MetalWorker::forward_chain_k` extracted into a free fn so the
/// Phase 9 speculative path (worker's lockstep thread, parallel to
/// target verify) can call it too without `&self` access. Caller
/// supplies the model + KV cache + GpuDevice + kernel refs.
///
/// Allocates K argmax_dual_write argument tables + per-iter argmax
/// output buffers + one stable chain_advance arg table + a packed
/// 16-byte consts buffer, then drives K forwards on one MTL4 CB via
/// `model.metal_chain_with_encoder`. After CB completion, host-reads
/// the K argmax buffers and returns iter-major `[k][num_reqs]`
/// argmax IDs.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn metal_chain_dispatch(
    model_ref: &dyn scratchy_forward_compiler::ScratchyWeights,
    kv_cache_ref: &KvCachePool,
    device_mut: &mut GpuDevice,
    argmax_kernels: &scratchy_target_metal::argmax::ArgmaxKernels,
    chain_kernel: &scratchy_target_metal::chain_advance::ChainAdvanceKernel,
    req: &::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest<'_>,
    block_size: usize,
    k: usize,
) -> Result<Vec<Vec<u32>>, String> {
    use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};

    if k == 0 {
        return Ok(Vec::new());
    }
    let num_reqs = req.cu_seqlens_q.len().saturating_sub(1);
    if num_reqs == 0 || num_reqs != req.num_tokens {
        return Err(format!(
            "metal_chain_dispatch: K-step decode requires num_tokens == num_reqs \
             (got num_tokens={}, num_reqs={})",
            req.num_tokens, num_reqs
        ));
    }

    let mtl_device = device_mut.device.clone();

    // ── 1. Upload iter-0 inputs ─────────────────────────────────
    let buf_input_ids = MetalWorker::alloc_shared_u32_buf(&mtl_device, req.input_ids);
    let buf_positions = MetalWorker::alloc_shared_u32_buf(&mtl_device, req.positions);
    let buf_slot_mapping = MetalWorker::alloc_shared_u32_buf(&mtl_device, req.slot_mapping);
    let buf_cu_seqlens = MetalWorker::alloc_shared_u32_buf(&mtl_device, req.cu_seqlens_q);
    let buf_seqused_k = MetalWorker::alloc_shared_u32_buf(&mtl_device, req.seqused_k);
    let buf_block_table = MetalWorker::alloc_shared_u32_buf(&mtl_device, req.block_table);

    let dtype_u32 = scratchy_target_metal::dtype::DType::U32;
    let view_input_ids = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_input_ids.contents().as_ptr() as *mut u8,
            &[req.num_tokens.max(1)],
            dtype_u32,
        ))
    };
    let view_positions = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_positions.contents().as_ptr() as *mut u8,
            &[req.num_tokens.max(1)],
            dtype_u32,
        ))
    };
    let view_slot_mapping = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_slot_mapping.contents().as_ptr() as *mut u8,
            &[req.num_tokens.max(1)],
            dtype_u32,
        ))
    };
    let view_cu_seqlens = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_cu_seqlens.contents().as_ptr() as *mut u8,
            &[req.cu_seqlens_q.len().max(1)],
            dtype_u32,
        ))
    };
    let view_seqused_k = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_seqused_k.contents().as_ptr() as *mut u8,
            &[req.seqused_k.len().max(1)],
            dtype_u32,
        ))
    };
    let view_block_table = unsafe {
        TensorView::from_raw(GpuTensor::new(
            buf_block_table.contents().as_ptr() as *mut u8,
            &[num_reqs.max(1), req.block_table_stride.max(1)],
            dtype_u32,
        ))
    };

    // ── 2. Allocate K argmax output buffers (host-visible) ──────
    let argmax_bytes = (req.num_tokens.max(1)) * 4;
    let argmax_bufs: Vec<
        ::objc2::rc::Retained<::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>>,
    > = (0..k)
        .map(|_| {
            mtl_device
                .newBufferWithLength_options(
                    argmax_bytes,
                    ::objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .expect("argmax_buf alloc returned nil")
        })
        .collect();

    // ── 3. Pack constants ───────────────────────────────────────
    let consts_buf = mtl_device
        .newBufferWithLength_options(16, ::objc2_metal::MTLResourceOptions::StorageModeShared)
        .expect("consts_buf alloc returned nil");
    let vocab_u32 = model_ref.vocab_size() as u32;
    unsafe {
        let p = consts_buf.contents().as_ptr() as *mut u32;
        *p.add(0) = req.num_tokens as u32;
        *p.add(1) = vocab_u32;
        *p.add(2) = block_size as u32;
        *p.add(3) = req.block_table_stride as u32;
    }

    // ── 4. Build K argmax_dual_write argument tables ────────────
    let argmax_arg_tables: Vec<_> = (0..k)
        .map(|_| {
            use ::objc2_metal::MTL4ArgumentTableDescriptor;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(5);
            mtl_device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("argmax dual_write arg_table alloc returned nil")
        })
        .collect();

    // ── 5. Build chain_advance argument table ───────────────────
    let chain_arg_table = {
        use ::objc2_metal::MTL4ArgumentTableDescriptor;
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(7);
        mtl_device
            .newArgumentTableWithDescriptor_error(&desc)
            .expect("chain_advance arg_table alloc returned nil")
    };

    // ── 6. ForwardCtx ───────────────────────────────────────────
    // Draft K-step chain: every forward step decodes 1 token per seq
    // (num_tokens == num_reqs), so `last_token_indices = [0, 1, …,
    // num_reqs-1]` would just be an identity gather. Skip it — the
    // gather/scatter early-out when src == dst makes the slice a
    // no-op anyway, and a non-zero indices buffer would cost only
    // memcpy. None here keeps the chain dispatch lean.
    // Draft K-step chain is single-group (no SWA ring) — null sliding views.
    let ctx = scratchy_target_metal::ForwardCtx {
        input_ids: view_input_ids,
        positions: view_positions,
        slot_mapping: view_slot_mapping,
        cu_seqlens_q: view_cu_seqlens,
        seqused_k: view_seqused_k,
        span_ids: None,
        block_table: view_block_table,
        // This path (non-paged / draft warmup) has no sliding groups.
        sliding_slot_mappings: Vec::new(),
        sliding_block_tables: Vec::new(),
        max_seqlen_q: req.max_seqlen_q,
        max_seqlen_k: req.max_seqlen_k,
        kv_cache: kv_cache_ref,
        // Text-only forward — no vision splice / encoder inputs.
        mm_embeds: None,
        embed_patches: &[],
        vision_rope_cos: None,
        vision_rope_sin: None,
        vision_rope_freqs: None,
        pixels: None,
        pos_embeds: None,
        vision_cu_seqlens_full: None,
        vision_cu_seqlens_window: None,
        vision_max_seqlen_full: None,
        vision_max_seqlen_window: None,
        vision_window_index: None,
        vision_reverse_indices: None,
        vision_position_ids: None,
        gdn_state: None,
        gdn_state_indices: None,
        gdn_is_fresh: None,
        has_spec_tokens: false,
        // metal_chain_dispatch (free fn): tq is provisioned at worker spawn via
        // the main forward_argmax_blocking path; the shared worker runtime
        // already carries it, so this flag is inert here.
        kv_turboquant: false,
        last_token_indices: None,
    };

    // SAFETY: kernel refs survive the synchronous call below.
    let argmax_kernels_addr: usize =
        argmax_kernels as *const scratchy_target_metal::argmax::ArgmaxKernels as usize;
    let chain_kernel_addr: usize =
        chain_kernel as *const scratchy_target_metal::chain_advance::ChainAdvanceKernel as usize;
    let dtype = model_ref.metal_dtype();

    let argmax_bufs_for_closure = argmax_bufs.clone();
    let consts_buf_for_closure = consts_buf.clone();
    let argmax_arg_tables_for_closure = argmax_arg_tables.clone();
    let chain_arg_table_for_closure = chain_arg_table.clone();

    let num_tokens_u32 = req.num_tokens as u32;
    let num_reqs_u32 = num_reqs as u32;
    let k_usize = k;

    let body: scratchy_forward_compiler::MetalChainBody<'_> = Box::new(
        move |handle, runtime, enc| -> Result<(), String> {
            // `MetalChainBody` hands the per-forward bindings through the
            // neutral `MetalRuntimeHandle` (the compiler's chain-body type
            // names no metal-target struct); recover the concrete metal
            // `RuntimeBindings` the closure reads buffer addresses off.
            let runtime: &scratchy_target_metal::interpreter::metal::RuntimeBindings =
                unsafe { runtime.as_ref() };
            let argmax_kernels_ref: &scratchy_target_metal::argmax::ArgmaxKernels = unsafe {
                &*(argmax_kernels_addr as *const scratchy_target_metal::argmax::ArgmaxKernels)
            };
            let chain_kernel_ref: &scratchy_target_metal::chain_advance::ChainAdvanceKernel = unsafe {
                &*(chain_kernel_addr
                    as *const scratchy_target_metal::chain_advance::ChainAdvanceKernel)
            };

            let logits_buf = handle.logits_buf();
            let vocab = handle.vocab();
            let logits_addr = logits_buf.gpuAddress();
            let consts_addr = consts_buf_for_closure.gpuAddress();
            let next_in_addr = runtime.input_ids.gpuAddress();

            unsafe {
                chain_arg_table_for_closure.setAddress_atIndex(runtime.positions.gpuAddress(), 0);
                // Spec-9 chain is single-group → KV-cache group 0.
                chain_arg_table_for_closure
                    .setAddress_atIndex(runtime.slot_mappings[0].gpuAddress(), 1);
                chain_arg_table_for_closure.setAddress_atIndex(runtime.seq_used_k.gpuAddress(), 2);
                chain_arg_table_for_closure
                    .setAddress_atIndex(runtime.block_tables[0].gpuAddress(), 3);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr + 8, 4);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr + 12, 5);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr, 6);
            }

            for iter in 0..k_usize {
                handle.run_forward_step(enc, num_tokens_u32, num_reqs_u32, false)?;

                let table = &argmax_arg_tables_for_closure[iter];
                let out_addr = argmax_bufs_for_closure[iter].gpuAddress();
                unsafe {
                    table.setAddress_atIndex(logits_addr, 0);
                    table.setAddress_atIndex(out_addr, 1);
                    table.setAddress_atIndex(consts_addr, 2);
                    table.setAddress_atIndex(consts_addr + 4, 3);
                    table.setAddress_atIndex(next_in_addr, 4);
                }
                let _ = vocab;
                match dtype {
                    scratchy_target_metal::interpreter::metal::MetalDtype::F16 => {
                        scratchy_target_metal::argmax::encode_argmax_f16_dual_write_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            table,
                            num_reqs_u32,
                        )
                        .map_err(|e| format!("encode argmax_f16_dual_write: {e:?}"))?;
                    }
                    scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => {
                        scratchy_target_metal::argmax::encode_argmax_bf16_dual_write_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            table,
                            num_reqs_u32,
                        )
                        .map_err(|e| format!("encode argmax_bf16_dual_write: {e:?}"))?;
                    }
                    scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                        return Err("argmax: int4 dtype has no direct kernel".into());
                    }
                }

                if iter + 1 < k_usize {
                    scratchy_target_metal::chain_advance::encode_chain_advance_into_mtl4(
                        chain_kernel_ref,
                        enc,
                        &chain_arg_table_for_closure,
                        num_reqs_u32,
                    )
                    .map_err(|e| format!("encode chain_advance: {e:?}"))?;
                }
            }
            Ok(())
        },
    );

    unsafe {
        model_ref.metal_chain_with_encoder(
            scratchy_target_metal::ForwardCtxHandle::new(&ctx),
            scratchy_target_metal::ForwardDeviceHandle::new(device_mut),
            req.num_tokens as u64,
            body,
        )?
    };

    let mut out: Vec<Vec<u32>> = Vec::with_capacity(k);
    for buf in argmax_bufs.iter() {
        let slice: &[u32] = unsafe {
            std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, num_reqs.max(1))
        };
        out.push(slice[..num_reqs.max(1)].to_vec());
    }

    Ok(out)
}

#[cfg(feature = "metal")]
impl ::scratchy_serving_engine::spec_decode::SpecDecodeBackend for MetalWorker {
    fn forward_argmax_blocking(
        &mut self,
        model: ::scratchy_serving_engine::spec_decode::ModelHandle,
        kv_pool: ::scratchy_serving_engine::spec_decode::KvPoolHandle,
        req: &::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest<'_>,
    ) -> Result<Vec<u32>, ::scratchy_serving_engine::spec_decode::BackendError> {
        use ::scratchy_serving_engine::spec_decode::{BackendError, KvPoolHandle, ModelHandle};

        // Resolve handles. 0 → target, 1 → draft; everything else is
        // an unknown handle.
        let model_ref: &dyn scratchy_forward_compiler::ScratchyWeights = match model {
            ModelHandle::TARGET => self
                .model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("target model not loaded".into()))?,
            ModelHandle(1) => self
                .draft_model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("draft model not loaded".into()))?,
            _ => return Err(BackendError::UnknownHandle("ModelHandle")),
        };
        let kv_cache_ref: &KvCachePool =
            match kv_pool {
                KvPoolHandle::TARGET => self.kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("target kv_cache not initialized".into())
                })?,
                KvPoolHandle(1) => self.draft_kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("draft kv_cache not initialized".into())
                })?,
                _ => return Err(BackendError::UnknownHandle("KvPoolHandle")),
            };

        // GDN recurrent state for hybrid arches — TARGET model only (the
        // speculative draft is a separate, non-GDN model). The pool's
        // per-linear-layer conv/ssm buffers reach the metal runtime
        // through `ForwardCtx::gdn_state`; the per-sequence slot ids /
        // fresh flags were built this step in `execute_model` and parked
        // in `self.gdn_pending` (consumed here). `None` for non-hybrid
        // arches / the draft → unchanged behavior.
        let gdn_state_ref = if matches!(model, ModelHandle::TARGET) {
            self.gdn_state.as_ref()
        } else {
            None
        };
        let gdn_pending = if gdn_state_ref.is_some() {
            self.gdn_pending.take()
        } else {
            None
        };

        // Vision-encoder output staged this step in `execute_model`.
        // TARGET model only (the speculative draft is text-only). Threads
        // into `ForwardCtx::{mm_embeds, embed_patches}` below to drive the
        // baked `Instruction::Embed` splice. `.take()` so it fires exactly
        // once (the upcoming target prefill), not on draft-chain calls.
        let mm_pending = if matches!(model, ModelHandle::TARGET) {
            self.mm_pending.take()
        } else {
            None
        };

        let device_buf = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?;
        let mtl_device = device_buf.device.clone();
        // Owned handle to the shared residency set, cloned early (before the
        // later `self.gpu_device.as_mut()` borrow) so the grammar-mask buffers
        // can be pinned without extending `device_buf`'s borrow.
        #[cfg(feature = "guided-decoding")]
        let grammar_residency = device_buf.allocator.residency().clone();

        // ── TurboQuant eviction-layer remap (opt-in: SCRATCHY_TQ_WINDOW) ──────
        // The packed store is the canonical full-capacity cache; attention reads
        // a BOUNDED fp16 window. Remap the logical block_table/slot_mapping to
        // window slots, and dequant-fill evicted misses from packed before
        // attention. A logical block written at offset 0 this forward is "fresh"
        // (reuse-safe: a reallocated logical block is always written from 0), so
        // it's assigned a window slot with NO dequant. Off => the in-place path
        // below is byte-identical. Uniform geometry (gemma4 hybrid deferred).
        let window_on = matches!(kv_pool, KvPoolHandle::TARGET)
            && self.turboquant.is_some()
            && self.tq_window.is_some()
            && std::env::var_os("SCRATCHY_TQ_WINDOW").is_some();
        let bs = self.config.block_size.max(1);
        let num_reqs_rm = req.cu_seqlens_q.len().saturating_sub(1);
        let (eff_block_table, eff_slot_mapping, window_misses): (
            Vec<u32>,
            Vec<u32>,
            Vec<(u32, u32)>,
        ) = if window_on {
            let max_blocks = req
                .block_table
                .len()
                .checked_div(num_reqs_rm)
                .unwrap_or(req.block_table.len());
            let bs32 = bs as u32;
            let wm = self.tq_window.as_mut().unwrap();
            let mut fresh = std::collections::HashSet::new();
            for &s in req.slot_mapping {
                if s % bs32 == 0 {
                    fresh.insert(s / bs32);
                }
            }
            let mut rbt = req.block_table.to_vec();
            let mut misses: Vec<(u32, u32)> = Vec::new();
            for sq in 0..num_reqs_rm {
                let used = req.seqused_k.get(sq).copied().unwrap_or(0) as usize;
                let valid = used.div_ceil(bs).min(max_blocks);
                for i in 0..valid {
                    let logical = req.block_table[sq * max_blocks + i];
                    let wslot = if fresh.contains(&logical) {
                        wm.assign_fresh(logical) as u32
                    } else {
                        let r = wm.resolve(logical);
                        if r.miss {
                            misses.push((logical, r.slot as u32));
                        }
                        r.slot as u32
                    };
                    rbt[sq * max_blocks + i] = wslot;
                }
            }
            let mut rsm = req.slot_mapping.to_vec();
            for (t, &ls) in req.slot_mapping.iter().enumerate() {
                let lb = ls / bs32;
                let off = ls % bs32;
                let wb = wm.slot_of(lb).map(|x| x as u32).unwrap_or(lb);
                rsm[t] = wb * bs32 + off;
            }
            (rbt, rsm, misses)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        if window_on
            && !window_misses.is_empty()
            && let Some(tq) = self.turboquant.as_ref()
        {
            let bs32 = bs as u32;
            let mut src = Vec::with_capacity(window_misses.len() * bs);
            let mut dst = Vec::with_capacity(window_misses.len() * bs);
            for &(lblock, wblock) in &window_misses {
                for tok in 0..bs32 {
                    src.push(lblock * bs32 + tok);
                    dst.push(wblock * bs32 + tok);
                }
            }
            let src_buf = Self::alloc_shared_u32_buf(&mtl_device, &src);
            let dst_buf = Self::alloc_shared_u32_buf(&mtl_device, &dst);
            let n = src.len() as u32;
            for layer in 0..tq.num_layers() {
                let kb: Vec<&_> = kv_cache_ref
                    .k_chunk_bufs(layer)
                    .iter()
                    .map(|m| m.buffer())
                    .collect();
                let vb: Vec<&_> = kv_cache_ref
                    .v_chunk_bufs(layer)
                    .iter()
                    .map(|m| m.buffer())
                    .collect();
                tq.dequant_into_window(
                    &mtl_device,
                    layer,
                    kv_cache_ref.k_chunk_table_mem(layer).buffer(),
                    &kb,
                    kv_cache_ref.v_chunk_table_mem(layer).buffer(),
                    &vb,
                    &src_buf,
                    &dst_buf,
                    n,
                )
                .map_err(|e| BackendError::Backend(format!("tq dequant_into_window: {e:?}")))?;
            }
        }

        let argmax_kernels = self
            .argmax_kernels
            .as_ref()
            .ok_or_else(|| BackendError::Backend("argmax_kernels not built".into()))?;

        // ── 1. Upload host slices to fresh shared-storage MTLBuffers ─────
        let buf_input_ids = Self::alloc_shared_u32_buf(&mtl_device, req.input_ids);
        let buf_positions = Self::alloc_shared_u32_buf(&mtl_device, req.positions);
        // window path binds the REMAPPED (window) slot_mapping/block_table for
        // the forward; the logical slot_mapping is kept for the packed-store
        // index in the post-forward compress.
        let buf_slot_mapping = Self::alloc_shared_u32_buf(
            &mtl_device,
            if window_on {
                &eff_slot_mapping
            } else {
                req.slot_mapping
            },
        );
        let buf_logical_slot_mapping = Self::alloc_shared_u32_buf(&mtl_device, req.slot_mapping);
        let buf_cu_seqlens = Self::alloc_shared_u32_buf(&mtl_device, req.cu_seqlens_q);
        let buf_seqused_k = Self::alloc_shared_u32_buf(&mtl_device, req.seqused_k);
        let buf_block_table = Self::alloc_shared_u32_buf(
            &mtl_device,
            if window_on {
                &eff_block_table
            } else {
                req.block_table
            },
        );
        // Stage the sample-row indices for the lm_head slice. The
        // closure inside macro-generated `forward` reads these via
        // `ctx.last_token_indices.as_raw()` and converts to a `&[u32]`
        // that drives the index-driven gather/scatter kernels.
        let buf_last_token_indices = req
            .last_token_indices
            .map(|s| Self::alloc_shared_u32_buf(&mtl_device, s));
        // ── 2. Wrap MTLBuffers in TensorViews. Dtype is purely
        //       descriptive; the macro-emitted forward only reads
        //       `as_raw().raw_ptr()` and `numel()`.
        let dtype_u32 = scratchy_target_metal::dtype::DType::U32;
        let num_reqs = req.cu_seqlens_q.len().saturating_sub(1);
        let view_input_ids = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_input_ids.contents().as_ptr() as *mut u8,
                &[req.num_tokens.max(1)],
                dtype_u32,
            ))
        };
        // Sized from the actual slice length, not `num_tokens`: MRoPE
        // batches pass `[3, n]` positions (len = 3·num_tokens), and the
        // macro forward reads this numel to detect the band-split layout.
        // For 1D positions `req.positions.len() == num_tokens`, so this is
        // unchanged for every non-MRoPE forward.
        let view_positions = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_positions.contents().as_ptr() as *mut u8,
                &[req.positions.len().max(1)],
                dtype_u32,
            ))
        };
        let view_slot_mapping = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_slot_mapping.contents().as_ptr() as *mut u8,
                &[req.num_tokens.max(1)],
                dtype_u32,
            ))
        };
        let view_cu_seqlens = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_cu_seqlens.contents().as_ptr() as *mut u8,
                &[req.cu_seqlens_q.len().max(1)],
                dtype_u32,
            ))
        };
        let view_seqused_k = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_seqused_k.contents().as_ptr() as *mut u8,
                &[req.seqused_k.len().max(1)],
                dtype_u32,
            ))
        };
        let view_block_table = unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_block_table.contents().as_ptr() as *mut u8,
                &[num_reqs.max(1), req.block_table_stride.max(1)],
                dtype_u32,
            ))
        };
        // last_token_indices is optional — chunked-prefill intermediate
        // chunks pass None, every other workload provides it.
        let view_last_token_indices = req.last_token_indices.map(|s| unsafe {
            TensorView::from_raw(GpuTensor::new(
                buf_last_token_indices.as_ref().unwrap().contents().as_ptr() as *mut u8,
                &[s.len().max(1)],
                dtype_u32,
            ))
        });

        // ── 3. Build ForwardCtx + run forward ────────────────────────────
        // Disjoint borrows: `model_ref`/`kv_cache_ref`/`argmax_kernels`
        // are immutable references into `self`; `device_mut` is a
        // mutable borrow of `self.gpu_device`. NLL accepts the split.
        //
        // Phase 8 routing: when the call is for the draft model
        // (ModelHandle(1)) and a dedicated `draft_queue` exists, we
        // build a shadow `GpuDevice` wrapping the same `MTLDevice` +
        // allocator but the draft queue, so the draft model's pool
        // attaches its residency set to that queue (instead of the
        // main queue). This lets target verify and draft work execute
        // concurrently on Apple Silicon. The shadow device is local
        // to this call; the worker's `gpu_device.queue` is untouched.
        let use_draft_queue = matches!(model, ModelHandle(1)) && self.draft_queue.is_some();
        let mut shadow_draft_device: Option<GpuDevice> = if use_draft_queue {
            let main_dev = self
                .gpu_device
                .as_ref()
                .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?;
            let draft_q = self.draft_queue.as_ref().unwrap().clone();
            Some(GpuDevice {
                device: main_dev.device.clone(),
                queue: draft_q,
                allocator: main_dev.allocator.clone(),
                metal_bucket_max_m: main_dev.metal_bucket_max_m,
            })
        } else {
            None
        };
        let device_mut: &mut GpuDevice = if let Some(ref mut shadow) = shadow_draft_device {
            shadow
        } else {
            self.gpu_device
                .as_mut()
                .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?
        };

        // Upload the per-step GDN slot ids / fresh flags into shared
        // (host-readable) buffers. The macro-emitted metal forward reads
        // them back off `ctx.gdn_state_indices` / `gdn_is_fresh` as host
        // slices for `ForwardInputs`, then `write_runtime_inputs` copies
        // them into the runtime buffers the kernels bind. i32 and u32
        // share the 4-byte layout `alloc_shared_u32_buf` writes. The
        // `buf_*` locals must outlive the forward call below — the views
        // hold raw pointers into their `contents()`.
        let gdn_pending_views = gdn_pending.as_ref().filter(|_| gdn_state_ref.is_some());
        let buf_gdn_indices = gdn_pending_views.map(|(idx, _)| {
            let idx_u32: &[u32] =
                unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u32, idx.len()) };
            MetalWorker::alloc_shared_u32_buf(&mtl_device, idx_u32)
        });
        let buf_gdn_is_fresh = gdn_pending_views
            .map(|(_, fresh)| MetalWorker::alloc_shared_u32_buf(&mtl_device, fresh));
        let view_gdn_indices =
            buf_gdn_indices
                .as_ref()
                .zip(gdn_pending_views)
                .map(|(b, (idx, _))| unsafe {
                    TensorView::from_raw(GpuTensor::new(
                        b.contents().as_ptr() as *mut u8,
                        &[idx.len().max(1)],
                        scratchy_target_metal::dtype::DType::I32,
                    ))
                });
        let view_gdn_is_fresh =
            buf_gdn_is_fresh
                .as_ref()
                .zip(gdn_pending_views)
                .map(|(b, (_, fresh))| unsafe {
                    TensorView::from_raw(GpuTensor::new(
                        b.contents().as_ptr() as *mut u8,
                        &[fresh.len().max(1)],
                        scratchy_target_metal::dtype::DType::U32,
                    ))
                });

        // Sliding KV-cache group (gemma4 SWA): wrap the ring's slot_mapping +
        // block table. Empty request slices (non-SWA models / draft) → null
        // views → the macro reads `None` → the sliding runtime buffers stay
        // unbound. The `buf_*` locals must outlive the forward (the views hold
        // raw pointers into their `contents()`).
        // Per sliding GROUP (vLLM group-shared): one Shared buffer + view each.
        let bufs_sliding_sm: Vec<_> = req
            .sliding_slot_mappings
            .iter()
            .map(|s| MetalWorker::alloc_shared_u32_buf(&mtl_device, s))
            .collect();
        let bufs_sliding_bt: Vec<_> = req
            .sliding_block_tables
            .iter()
            .map(|s| MetalWorker::alloc_shared_u32_buf(&mtl_device, s))
            .collect();
        let view_sliding_slot_mappings: Vec<TensorView> = bufs_sliding_sm
            .iter()
            .map(|b| unsafe {
                TensorView::from_raw(GpuTensor::new(
                    b.contents().as_ptr() as *mut u8,
                    &[req.num_tokens.max(1)],
                    dtype_u32,
                ))
            })
            .collect();
        let view_sliding_block_tables: Vec<TensorView> = bufs_sliding_bt
            .iter()
            .map(|b| unsafe {
                TensorView::from_raw(GpuTensor::new(
                    b.contents().as_ptr() as *mut u8,
                    &[num_reqs.max(1), req.block_table_stride.max(1)],
                    dtype_u32,
                ))
            })
            .collect();

        let ctx = scratchy_target_metal::ForwardCtx {
            input_ids: view_input_ids,
            positions: view_positions,
            slot_mapping: view_slot_mapping,
            cu_seqlens_q: view_cu_seqlens,
            seqused_k: view_seqused_k,
            span_ids: req.span_ids.map(|s| s.to_vec()),
            block_table: view_block_table,
            sliding_slot_mappings: view_sliding_slot_mappings,
            sliding_block_tables: view_sliding_block_tables,
            max_seqlen_q: req.max_seqlen_q,
            max_seqlen_k: req.max_seqlen_k,
            kv_cache: kv_cache_ref,
            // Vision splice inputs (text-only forwards leave both empty:
            // `embed_patches.is_empty()` degenerates the splice arm to a
            // plain embedding gather). The encoder ran in `execute_model`.
            mm_embeds: mm_pending.as_ref().map(|(t, _)| t.view()),
            embed_patches: mm_pending
                .as_ref()
                .map(|(_, p)| p.as_slice())
                .unwrap_or(&[]),
            vision_rope_cos: None,
            vision_rope_sin: None,
            vision_rope_freqs: None,
            pixels: None,
            pos_embeds: None,
            vision_cu_seqlens_full: None,
            vision_cu_seqlens_window: None,
            vision_max_seqlen_full: None,
            vision_max_seqlen_window: None,
            vision_window_index: None,
            vision_reverse_indices: None,
            vision_position_ids: None,
            gdn_state: gdn_state_ref,
            gdn_state_indices: view_gdn_indices,
            gdn_is_fresh: view_gdn_is_fresh,
            has_spec_tokens: req.has_spec_tokens,
            kv_turboquant: self.config.kv_cache_dtype == "turboquant",
            last_token_indices: view_last_token_indices,
        };

        // ── 4. Fused argmax via forward_with_metal_followup. ─────────────
        // Phase 6a: argmax encodes onto the SAME MTL4 compute encoder
        // as the forward — forward + argmax share one CB, one commit,
        // one host wait. Pre-6a took the unfused path (separate dispatch
        // + commit + wait + readback) for 5.2a simplicity; this re-folds
        // it. Per call: 2 commit+waits → 1.
        let argmax_bytes = (req.num_tokens.max(1)) * 4;
        let argmax_out = mtl_device
            .newBufferWithLength_options(
                argmax_bytes,
                ::objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .expect("argmax_out alloc returned nil");
        let consts_buf = mtl_device
            .newBufferWithLength_options(8, ::objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("argmax consts alloc returned nil");
        let arg_table = {
            use ::objc2_metal::MTL4ArgumentTableDescriptor;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(4);
            mtl_device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("argmax arg_table alloc returned nil")
        };
        let dtype = model_ref.metal_dtype();
        let argmax_out_for_closure = argmax_out.clone();
        let consts_for_closure = consts_buf.clone();
        let arg_table_for_closure = arg_table.clone();
        // SAFETY: `argmax_kernels` is borrowed from `&self.argmax_kernels`;
        // the closure runs synchronously inside `model_ref.forward_*` while
        // that borrow is live. Cast to a raw pointer to keep the closure
        // 'static + Send-safe per `MetalForwardFollowup`'s bound.
        let argmax_kernels_ptr: *const scratchy_target_metal::argmax::ArgmaxKernels =
            argmax_kernels;
        let argmax_kernels_addr = argmax_kernels_ptr as usize;
        // Stage this step's grammar mask (constrained / guided decoding) into
        // the PERSISTENT residency-pinned buffers (grown + re-pinned only when a
        // batch exceeds capacity). Pinning is mandatory: a per-step freshly
        // allocated bitset gets evicted under KV pressure and the kernel reads
        // garbage (see the `grammar_buf_*` field docs). The closure binds these
        // by gpuAddress and runs the mask on the forward encoder before argmax.
        #[cfg(feature = "guided-decoding")]
        let grammar_pending = self.grammar_pending.take();
        #[cfg(feature = "guided-decoding")]
        let grammar_mask_ctx: Option<GrammarMaskGpu> = if let (Some(h), Some(kernels)) =
            (grammar_pending, self.grammar_mask_kernels.as_ref())
        {
            use ::objc2_metal::{MTLBuffer, MTLDevice};
            let kernels_addr =
                kernels as *const scratchy_target_metal::grammar_mask::GrammarMaskKernels as usize;
            let opts = ::objc2_metal::MTLResourceOptions::StorageModeShared;
            let mut grew = false;
            let need_allow = h.allow_bits.len().max(1);
            if self.grammar_buf_allow.is_none() || self.grammar_cap_allow < need_allow {
                let buf = mtl_device
                    .newBufferWithLength_options(need_allow * 4, opts)
                    .expect("grammar allow_bits buf");
                grammar_residency.insert(&buf);
                self.grammar_buf_allow = Some(buf);
                self.grammar_cap_allow = need_allow;
                grew = true;
            }
            let need_rows = h.rows.len().max(1);
            if self.grammar_buf_rows.is_none() || self.grammar_cap_rows < need_rows {
                let buf = mtl_device
                    .newBufferWithLength_options(need_rows * 4, opts)
                    .expect("grammar rows buf");
                grammar_residency.insert(&buf);
                self.grammar_buf_rows = Some(buf);
                self.grammar_cap_rows = need_rows;
                grew = true;
            }
            if self.grammar_buf_gconsts.is_none() {
                let buf = mtl_device
                    .newBufferWithLength_options(2 * 4, opts)
                    .expect("grammar gconsts buf");
                grammar_residency.insert(&buf);
                self.grammar_buf_gconsts = Some(buf);
                grew = true;
            }
            // Commit the residency set only when a buffer was (re)allocated;
            // steady-state steps reuse the already-pinned buffers.
            if grew {
                grammar_residency.commit();
            }
            let allow = self.grammar_buf_allow.as_ref().unwrap();
            let rows = self.grammar_buf_rows.as_ref().unwrap();
            let gconsts = self.grammar_buf_gconsts.as_ref().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    h.allow_bits.as_ptr(),
                    allow.contents().as_ptr() as *mut u32,
                    h.allow_bits.len(),
                );
                std::ptr::copy_nonoverlapping(
                    h.rows.as_ptr(),
                    rows.contents().as_ptr() as *mut u32,
                    h.rows.len(),
                );
                let gc = [h.vocab, h.words_per_row];
                std::ptr::copy_nonoverlapping(
                    gc.as_ptr(),
                    gconsts.contents().as_ptr() as *mut u32,
                    gc.len(),
                );
            }
            let arg_table = {
                use ::objc2_metal::MTL4ArgumentTableDescriptor;
                let desc = MTL4ArgumentTableDescriptor::new();
                desc.setMaxBufferBindCount(5);
                mtl_device
                    .newArgumentTableWithDescriptor_error(&desc)
                    .expect("grammar_mask arg_table alloc returned nil")
            };
            Some(GrammarMaskGpu {
                allow_bits: allow.clone(),
                rows: rows.clone(),
                gconsts: gconsts.clone(),
                arg_table,
                num_rows: h.rows.len() as u32,
                kernels_addr,
            })
        } else {
            None
        };
        // Capture the terminal logits slot + its runtime column count for the
        // on-GPU sampler. The followup receives the arena's lm_head-output
        // MTLBuffer (the same one argmax reads); the sampler needs it (resident,
        // by gpuAddress) to gather sampling requests' logits rows AFTER this
        // blocking forward completes. `Rc<RefCell>` because the followup is a
        // synchronous, single-threaded `FnOnce` invoked during the forward.
        let logits_capture: std::rc::Rc<
            std::cell::RefCell<Option<(scratchy_target_metal::mtl4_dispatch::Buffer, u32)>>,
        > = std::rc::Rc::new(std::cell::RefCell::new(None));
        let logits_capture_cl = logits_capture.clone();
        // ── Fused on-GPU sampler (peak path) ─────────────────────────────
        // If `execute_model` prepared this step's non-greedy rows, encode the
        // sampler (cast → [penalties] → sample) onto THIS forward's command
        // buffer — in the followup, AFTER argmax — so forward + argmax + sampling
        // are ONE commit + ONE host wait, not a second `Mtl4DispatchBatch`. The
        // `PendingSampler` moves into the closure; its output buffer is cloned
        // out first so it can be read back after the (single) host wait below.
        let pending_sampler = self.pending_sampler.take();
        let sampler_readback = pending_sampler.as_ref().map(|p| p.output());
        #[cfg(feature = "sampler-telemetry")]
        let sampler_telem = pending_sampler.as_ref().and_then(|p| p.telemetry_output());
        let sampler_kernels_addr = self
            .sampler_kernels
            .as_ref()
            .map(|k| k as *const scratchy_target_metal::sampling::SamplerKernels as usize);
        let followup: scratchy_forward_compiler::MetalForwardFollowup<'_> = Box::new(
            move |enc, logits_buf, total_n_actual, vocab_actual| -> Result<(), String> {
                // Retain a clone of the logits buffer for the post-forward
                // sampler (see `logits_capture` above).
                let logits_retained: scratchy_target_metal::mtl4_dispatch::Buffer = unsafe {
                    ::objc2::rc::Retained::retain(
                        logits_buf as *const _
                            as *mut ::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>,
                    )
                }
                .expect("retain logits buffer");
                *logits_capture_cl.borrow_mut() = Some((logits_retained, vocab_actual));
                // Update consts buffer in-place (StorageModeShared).
                let consts_ptr = consts_for_closure.contents().as_ptr() as *mut u32;
                unsafe {
                    *consts_ptr = total_n_actual;
                    *consts_ptr.add(1) = vocab_actual;
                }
                use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};
                let logits_addr = logits_buf.gpuAddress();
                let out_addr = argmax_out_for_closure.gpuAddress();
                let consts_addr = consts_for_closure.gpuAddress();
                unsafe {
                    arg_table_for_closure.setAddress_atIndex(logits_addr, 0);
                    arg_table_for_closure.setAddress_atIndex(out_addr, 1);
                    arg_table_for_closure.setAddress_atIndex(consts_addr, 2);
                    arg_table_for_closure.setAddress_atIndex(consts_addr + 4, 3);
                }
                // ── Grammar mask (constrained / guided decoding) ─────────
                // Runs on THIS encoder before argmax: forces every token
                // disallowed by a request's grammar FSM to -inf so the greedy
                // argmax can only pick a grammar-valid token. The mask emits
                // its own pre-barrier (forward-write -> mask-read); argmax's
                // pre-barrier then orders mask-write -> argmax-read.
                #[cfg(feature = "guided-decoding")]
                if let Some(ref gm) = grammar_mask_ctx {
                    // Mask the FULL runtime logits width. The shader treats any
                    // slot beyond the bitset (`words_per_row*32`, i.e. the
                    // lm_head's vocab padding past the tokenizer vocab) as
                    // disallowed, so padded tokens can never be sampled and the
                    // bits[] read stays in bounds — no host-side clamp needed.
                    // gconsts[1] (words_per_row) was set on the host.
                    let gconsts_ptr = gm.gconsts.contents().as_ptr() as *mut u32;
                    unsafe {
                        *gconsts_ptr = vocab_actual;
                    }
                    let gconsts_addr = gm.gconsts.gpuAddress();
                    unsafe {
                        gm.arg_table.setAddress_atIndex(logits_addr, 0);
                        gm.arg_table
                            .setAddress_atIndex(gm.allow_bits.gpuAddress(), 1);
                        gm.arg_table.setAddress_atIndex(gm.rows.gpuAddress(), 2);
                        gm.arg_table.setAddress_atIndex(gconsts_addr, 3);
                        gm.arg_table.setAddress_atIndex(gconsts_addr + 4, 4);
                    }
                    let gm_kernels: &scratchy_target_metal::grammar_mask::GrammarMaskKernels = unsafe {
                        &*(gm.kernels_addr
                            as *const scratchy_target_metal::grammar_mask::GrammarMaskKernels)
                    };
                    match dtype {
                        scratchy_target_metal::interpreter::metal::MetalDtype::F16 => {
                            scratchy_target_metal::grammar_mask::encode_grammar_mask_f16_into_mtl4(
                                gm_kernels,
                                enc,
                                &gm.arg_table,
                                gm.num_rows,
                            )
                            .map_err(|e| format!("encode_grammar_mask_f16: {e:?}"))?;
                        }
                        scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => {
                            scratchy_target_metal::grammar_mask::encode_grammar_mask_bf16_into_mtl4(
                                gm_kernels,
                                enc,
                                &gm.arg_table,
                                gm.num_rows,
                            )
                            .map_err(|e| format!("encode_grammar_mask_bf16: {e:?}"))?;
                        }
                        scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                            return Err("grammar_mask: int4 dtype has no direct kernel".into());
                        }
                    }
                }
                let argmax_kernels_ref: &scratchy_target_metal::argmax::ArgmaxKernels = unsafe {
                    &*(argmax_kernels_addr as *const scratchy_target_metal::argmax::ArgmaxKernels)
                };
                match dtype {
                    scratchy_target_metal::interpreter::metal::MetalDtype::F16 => {
                        scratchy_target_metal::argmax::encode_argmax_f16_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            &arg_table_for_closure,
                            total_n_actual,
                        )
                        .map_err(|e| format!("encode_argmax_f16: {e:?}"))?;
                    }
                    scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => {
                        scratchy_target_metal::argmax::encode_argmax_bf16_into_mtl4(
                            argmax_kernels_ref,
                            enc,
                            &arg_table_for_closure,
                            total_n_actual,
                        )
                        .map_err(|e| format!("encode_argmax_bf16: {e:?}"))?;
                    }
                    scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                        return Err("argmax: int4 dtype has no direct kernel".into());
                    }
                }
                // Fused sampler: encode cast → [penalties] → sample onto THIS
                // encoder, after argmax, reading the (grammar-masked) logits.
                if let Some(ref ps) = pending_sampler
                    && let Some(addr) = sampler_kernels_addr
                {
                    let kernels_ref = unsafe {
                        &*(addr as *const scratchy_target_metal::sampling::SamplerKernels)
                    };
                    ps.encode_into(enc, logits_addr, kernels_ref);
                }
                Ok(())
            },
        );
        let logits = unsafe {
            model_ref.forward_with_metal_followup(
                scratchy_target_metal::ForwardCtxHandle::new(&ctx),
                scratchy_target_metal::ForwardDeviceHandle::new(device_mut),
                req.num_tokens as u64,
                Some(followup),
            )
        };
        let _ = logits; // argmax_out is what we read

        // ── TurboQuant: compress this step's new KV slots in place ───────
        // The forward above is blocking, so the new KV is written. For each
        // layer, quantize the new slots into the packed store (~4.6x) and
        // dequant back into the fp16 pool, so the NEXT step's attention reads
        // TurboQuant'd KV. Target pool only; OFF unless kv_cache_dtype=turboquant.
        if matches!(kv_pool, KvPoolHandle::TARGET)
            && let Some(tq) = self.turboquant.as_ref()
        {
            let n_slots = req.num_tokens as u32;
            // Batch ALL layers' K+V compress into ONE MTL4 command buffer
            // (one commit, NO host wait) — the per-layer commit+wait was a
            // ~2x decode hit (16 layers × K/V = 32 host syncs/step). The
            // kernel reaches the pool via the chunk-table gpuAddress, so the
            // chunk DATA buffers ride in as extra-resident (not bound) inside
            // encode_compress_layer. slots = WINDOW slot (fp16 read),
            // logical_slots = packed index; identical for the in-place path.
            scratchy_target_metal::turboquant::run_compress_batch(&mtl_device, |batch| {
                for layer in 0..tq.num_layers() {
                    let kb: Vec<&_> = kv_cache_ref
                        .k_chunk_bufs(layer)
                        .iter()
                        .map(|m| m.buffer())
                        .collect();
                    let vb: Vec<&_> = kv_cache_ref
                        .v_chunk_bufs(layer)
                        .iter()
                        .map(|m| m.buffer())
                        .collect();
                    tq.encode_compress_layer(
                        batch,
                        layer,
                        kv_cache_ref.k_chunk_table_mem(layer).buffer(),
                        &kb,
                        kv_cache_ref.v_chunk_table_mem(layer).buffer(),
                        &vb,
                        &buf_slot_mapping,
                        &buf_logical_slot_mapping,
                        n_slots,
                    );
                }
            })
            .map_err(|e| BackendError::Backend(format!("turboquant compress: {e:?}")))?;
        }

        // ── 5. Read host-visible argmax buffer + return. ─────────────────
        let argmax_slice: &[u32] = unsafe {
            std::slice::from_raw_parts(
                argmax_out.contents().as_ptr() as *const u32,
                req.num_tokens.max(1),
            )
        };
        let out = argmax_slice[..req.num_tokens.max(1)].to_vec();
        // Hand the captured logits slot to `execute_model`'s sample block. Only
        // the TARGET forward's logits are used for sampling; a draft/verify
        // forward's capture is harmless (never read for a sampling request).
        // Fused sampler: it rode this forward's CB, so its output is ready after
        // the single host wait above. Read it back for `execute_model` (aligned
        // to the `sample_jobs` order it built).
        if let Some((out_buf, njobs)) = sampler_readback {
            self.fused_sampled = Some(scratchy_target_metal::mtl4_dispatch::read_slice::<u32>(
                &out_buf,
                njobs as usize,
            ));
        }
        // Live "soul" telemetry: read the tiny spill buffers (row 0 only — the
        // max_num_seqs=1 live assumption) and publish the distribution for this
        // token. Rides the same host wait; only present when telemetry was on at
        // prepare time.
        #[cfg(feature = "sampler-telemetry")]
        if let Some((probs_buf, idx_buf, stats_buf, njobs, k)) = sampler_telem {
            use scratchy_core_common::sampler_telemetry::{
                SamplerCandidate, SamplerRecord, SamplerTelemetry,
            };
            let k = k as usize;
            let n = njobs as usize;
            let stats = scratchy_target_metal::mtl4_dispatch::read_slice::<f32>(&stats_buf, n * 2);
            let probs = scratchy_target_metal::mtl4_dispatch::read_slice::<f32>(&probs_buf, n * k);
            let idxs = scratchy_target_metal::mtl4_dispatch::read_slice::<u32>(&idx_buf, n * k);
            let mut top_k = Vec::with_capacity(k);
            for i in 0..k.min(probs.len()).min(idxs.len()) {
                let prob = probs[i];
                if prob <= 0.0 {
                    break; // padding past the real candidate count
                }
                top_k.push(SamplerCandidate {
                    token_id: idxs[i],
                    prob,
                });
            }
            let sampled_token_id = self
                .fused_sampled
                .as_ref()
                .and_then(|v| v.first().copied())
                .unwrap_or_else(|| top_k.first().map(|c| c.token_id).unwrap_or(0));
            SamplerTelemetry::global().publish(SamplerRecord {
                sampled_token_id,
                greedy: false,
                max_prob: stats.first().copied().unwrap_or(f32::NAN),
                entropy_nats: stats.get(1).copied().unwrap_or(f32::NAN),
                top_k,
            });
        }
        self.sampler_logits = logits_capture.borrow_mut().take();
        Ok(out)
    }

    /// Phase 6: K-step draft chain in ONE MTL4 command buffer. Forward,
    /// argmax_dual_write, and chain_advance (advances positions /
    /// slot_mapping / seqused_k in-place on the GPU between iters)
    /// for K iters share one CB, one commit, and one host wait. Per
    /// iter we spend ~kernel-time only — no host roundtrip, no per-
    /// iter allocator reset.
    ///
    /// Thin wrapper that resolves handles → refs and delegates to the
    /// free [`metal_chain_dispatch`] helper. The free fn is shared
    /// with the Phase 9 speculative path (worker's lockstep thread),
    /// which can't borrow `&self` while the main thread is mid-
    /// `forward_argmax_blocking` for target verify.
    fn forward_chain_k(
        &mut self,
        model: ::scratchy_serving_engine::spec_decode::ModelHandle,
        kv_pool: ::scratchy_serving_engine::spec_decode::KvPoolHandle,
        req: &::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest<'_>,
        block_size: usize,
        k: usize,
    ) -> Result<Vec<Vec<u32>>, ::scratchy_serving_engine::spec_decode::BackendError> {
        use ::scratchy_serving_engine::spec_decode::{BackendError, KvPoolHandle, ModelHandle};

        if k == 0 {
            return Ok(Vec::new());
        }

        // Resolve target model + KV pool (handle 1 = draft, 0 = target).
        let model_ref: &dyn scratchy_forward_compiler::ScratchyWeights = match model {
            ModelHandle::TARGET => self
                .model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("target model not loaded".into()))?,
            ModelHandle(1) => self
                .draft_model
                .as_deref()
                .ok_or_else(|| BackendError::Backend("draft model not loaded".into()))?,
            _ => return Err(BackendError::UnknownHandle("ModelHandle")),
        };
        let kv_cache_ref: &KvCachePool =
            match kv_pool {
                KvPoolHandle::TARGET => self.kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("target kv_cache not initialized".into())
                })?,
                KvPoolHandle(1) => self.draft_kv_cache.as_ref().ok_or_else(|| {
                    BackendError::Backend("draft kv_cache not initialized".into())
                })?,
                _ => return Err(BackendError::UnknownHandle("KvPoolHandle")),
            };

        let argmax_kernels = self
            .argmax_kernels
            .as_ref()
            .ok_or_else(|| BackendError::Backend("argmax_kernels not built".into()))?;
        let chain_kernel = self
            .chain_advance_kernel
            .as_ref()
            .ok_or_else(|| BackendError::Backend("chain_advance_kernel not built".into()))?;

        // Phase 8 routing: when the call is for the draft model AND a
        // dedicated `draft_queue` exists, route through a shadow
        // GpuDevice on the draft queue so target verify and draft
        // chain stay disjoint.
        let use_draft_queue = matches!(model, ModelHandle(1)) && self.draft_queue.is_some();
        let mut shadow_draft_device: Option<GpuDevice> = if use_draft_queue {
            let main_dev = self.gpu_device.as_ref().expect("checked above");
            let draft_q = self.draft_queue.as_ref().unwrap().clone();
            Some(GpuDevice {
                device: main_dev.device.clone(),
                queue: draft_q,
                allocator: main_dev.allocator.clone(),
                metal_bucket_max_m: main_dev.metal_bucket_max_m,
            })
        } else {
            None
        };
        let device_mut: &mut GpuDevice = if let Some(ref mut shadow) = shadow_draft_device {
            shadow
        } else {
            self.gpu_device
                .as_mut()
                .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?
        };

        metal_chain_dispatch(
            model_ref,
            kv_cache_ref,
            device_mut,
            argmax_kernels,
            chain_kernel,
            req,
            block_size,
            k,
        )
        .map_err(BackendError::Backend)
    }

    // (legacy body folded into the free fn `metal_chain_dispatch` below)

    fn load_secondary_model(
        &mut self,
        _path: &::std::path::Path,
        _dtype: Option<&str>,
    ) -> Result<
        ::scratchy_serving_engine::spec_decode::ModelHandle,
        ::scratchy_serving_engine::spec_decode::BackendError,
    > {
        // Thin wrapper: delegate to the existing helper which pulls
        // path + dtype from `self.config`. Phase 5.4 will tighten this
        // when `DraftModelProposer` takes ownership of the lifecycle and
        // pushes the path/dtype through the trait surface directly.
        self.load_draft_model_metal().map_err(|e| {
            ::scratchy_serving_engine::spec_decode::BackendError::Backend(e.to_string())
        })?;
        Ok(::scratchy_serving_engine::spec_decode::ModelHandle(1))
    }

    fn allocate_kv_pool(
        &mut self,
        model: ::scratchy_serving_engine::spec_decode::ModelHandle,
        num_blocks: usize,
    ) -> Result<
        ::scratchy_serving_engine::spec_decode::KvPoolHandle,
        ::scratchy_serving_engine::spec_decode::BackendError,
    > {
        if model != ::scratchy_serving_engine::spec_decode::ModelHandle(1) {
            return Err(
                ::scratchy_serving_engine::spec_decode::BackendError::UnknownHandle(
                    "ModelHandle (only the draft handle 1 supports allocate_kv_pool today)",
                ),
            );
        }
        self.initialize_draft_cache_metal(num_blocks).map_err(|e| {
            ::scratchy_serving_engine::spec_decode::BackendError::Backend(e.to_string())
        })?;
        Ok(::scratchy_serving_engine::spec_decode::KvPoolHandle(1))
    }

    fn kv_per_block_bytes(
        &self,
        model: ::scratchy_serving_engine::spec_decode::ModelHandle,
    ) -> Result<usize, ::scratchy_serving_engine::spec_decode::BackendError> {
        let model_ref: &dyn scratchy_forward_compiler::ScratchyWeights = match model {
            ::scratchy_serving_engine::spec_decode::ModelHandle::TARGET => {
                self.model.as_deref().ok_or_else(|| {
                    ::scratchy_serving_engine::spec_decode::BackendError::Backend(
                        "target model not loaded".into(),
                    )
                })?
            }
            ::scratchy_serving_engine::spec_decode::ModelHandle(1) => {
                self.draft_model.as_deref().ok_or_else(|| {
                    ::scratchy_serving_engine::spec_decode::BackendError::Backend(
                        "draft model not loaded".into(),
                    )
                })?
            }
            _ => {
                return Err(
                    ::scratchy_serving_engine::spec_decode::BackendError::UnknownHandle(
                        "ModelHandle",
                    ),
                );
            }
        };
        Ok(kv_per_block_bytes(model_ref, self.config.block_size))
    }
}

/// Guarantee the one-time aligned-sidecar build survives process exit
/// even when `shutdown()` is never called — the in-process CLI path
/// tears down by dropping the worker, not via an explicit executor
/// shutdown, so a short `scr chat -q …` would otherwise exit mid-write
/// and the cache would never persist (every launch re-pays the cold
/// realign-copy). `drop` runs before the worker's device/buffer fields,
/// so the writer's source stays valid; the drain is idempotent with the
/// one in `shutdown()`. (regression: c54955a3)
#[cfg(feature = "metal")]
impl Drop for MetalWorker {
    fn drop(&mut self) {
        scratchy_target_metal::metal_allocator::join_sidecar_writers();
    }
}

/// Releases the wired GPU residency if the guarded scope unwinds due to a panic.
///
/// A worker-thread panic (e.g. a GPU command-buffer error the generated forward
/// `.expect()`s — `ExecutionFailed(MTLCommandBufferStatus(5))`) unwinds under
/// `panic = "unwind"` WITHOUT raising a signal, so it bypasses ec877a03's
/// graceful teardown (SIGTERM → `worker.shutdown()`). The process is then left
/// holding `requestResidency`-pinned GPU pages (a zombie server) that orphan on
/// the eventual `kill -9`. Dropping this guard while the worker frame unwinds
/// runs the SAME idempotent `MetalResidencySet::shutdown()` the graceful path
/// uses, so the wired memory is released deterministically instead. On a normal
/// return `thread::panicking()` is false, so it's a no-op.
#[cfg(feature = "metal")]
struct ResidencyPanicGuard {
    sets: Vec<scratchy_target_metal::residency::MetalResidencySet>,
}

#[cfg(feature = "metal")]
impl Drop for ResidencyPanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            for set in &self.sets {
                set.shutdown();
            }
        }
    }
}

#[cfg(feature = "metal")]
/// Minimum fp16 KV footprint (bytes per token, all layers, K+V) for
/// TurboQuant to be worth enabling by default.
///
/// TurboQuant trades fidelity for KV CAPACITY. Below this, the capacity
/// is not the constraint and the trade is a bad one. Sized to sit
/// between the models measured on metal:
///
///     qwen2.5-0.5b   24 x 2 kv x 64  =  12 KiB/token   -> fp16 (garbage under TQ)
///     llama-3.2-1b   16 x 8 kv x 64  =  32 KiB/token   -> TurboQuant
///     granite-4.1-3b 40 x 8 kv x 64  =  80 KiB/token   -> TurboQuant
///     gemma-3-4b     34 x 4 kv x 256 = 544 KiB/token   -> TurboQuant
///
/// A model between 12 and 32 KiB/token is untested either way; the
/// threshold is set at 24 KiB so the two measured points stay on the
/// sides they were measured on, and is a POLICY knob, not a law.
const TQ_MIN_KV_BYTES_PER_TOKEN: usize = 24 * 1024;

#[cfg(feature = "metal")]
impl Worker for MetalWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        let device = scratchy_target_metal::detect_device()
            .ok_or_else(|| ExecutorError::WorkerInit("no Metal device available".to_string()))?;
        self.metal_device = Some(std::sync::Arc::new(device));
        info!("ScratchyWorker(metal): Metal device initialized");
        Ok(())
    }

    fn prefill_bucket_max_m(&self) -> Option<u32> {
        // Set by `determine_available_memory` from the target-reactive bucket
        // selection; the engine clamps `max_num_batched_tokens` to it.
        //
        // TurboQuant: cap the prefill chunk at 2048. A chunked-prefill
        // CONTINUATION chunk in the 4096 bucket has a residual collapse past
        // ~8k tokens (3rd+ continuation, finite K/V — a downstream compute NaN);
        // the 2048 bucket is clean (steel, verified to 31k). Capping here makes
        // long-context TurboQuant correct BY DEFAULT (no --max-num-batched-tokens
        // flag). `tq_on` mirrors `initialize_cache`'s auto-enable (power-of-2
        // head_dim <= 256) so it holds before that runs too.
        // TODO: root-cause the bucket-4096 continuation NaN to restore the 4096
        // chunk's prefill throughput.
        let tq_on = self.config.kv_cache_dtype == "turboquant"
            || (self.config.kv_cache_dtype == "auto"
                && self.model.as_ref().is_some_and(|m| {
                    let hd = m.head_dim() as usize;
                    hd.is_power_of_two() && hd <= 256
                }));
        if tq_on {
            return self.metal_prefill_bucket_max_m.map(|v| v.min(2048));
        }
        self.metal_prefill_bucket_max_m
    }

    fn kv_max_addressable_tokens(&self) -> Option<usize> {
        // Hybrid SWA arches (gemma4): the sliding KV groups use the base
        // (smallest) block size, and every group's block table is baked at the
        // `MAX_BLOCKS_PER_SEQ` row stride, so the kernel can address at most
        // `MAX_BLOCKS_PER_SEQ × block_size` tokens (gemma4: 2048×16 = 32768).
        // Beyond that the sliding block table overflows its row. Only report a
        // limit for hybrid arches (uniform models keep their existing behavior).
        let model = self.model.as_ref()?;
        model.per_layer_kv_token_elems()?;
        Some(model.max_blocks_per_seq() * self.config.block_size)
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        let _t_total = std::time::Instant::now();
        let t_resolve = std::time::Instant::now();
        let model_dir = resolve_model_path(
            &self.config.model_path,
            self.config.hf_token.as_deref(),
            self.config.gguf_file.as_deref(),
            None,
        )?;
        info!(
            "ScratchyWorker(metal): resolved model dir in {:?} ({})",
            t_resolve.elapsed(),
            model_dir.display()
        );

        // Metal currently supports safetensors only — GGUF requires the
        // cuda-side dequant kernels. Surface that early with a clear
        // message rather than failing deep in `GpuWeights::from_dir`.
        if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
            return Err(ExecutorError::WorkerInit(
                "GGUF source not supported under metal (safetensors only)".into(),
            ));
        }

        let t_cfg = std::time::Instant::now();
        let hf_config = HfModelConfig::from_path(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("config parse failed: {e}")))?;
        let arch = hf_config.architectures.first().cloned().unwrap_or_default();
        info!(
            "ScratchyWorker(metal): parsed hf_config in {:?} (arch = {arch})",
            t_cfg.elapsed()
        );

        // Build the metal `GpuDevice` (device + queue + allocator). The
        // allocator goes inside `GpuWeights` (consumed by `from_dir`) and
        // a parallel `Arc<MetalAllocator>` rides on `GpuDevice` for
        // dummy-tensor allocation during profiling. Both wrap the same
        // underlying `metal::Device` (cheap ObjC retain).
        let metal_dev = self
            .metal_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("metal device not initialized".into()))?;
        let device_arc = std::sync::Arc::new(metal_dev.device.clone());
        // Single allocator shared between `GpuWeights` (loader-side
        // bump arena) and `GpuDevice.allocator` (worker-side
        // `buffer_for` reverse-lookup at ICB-record time). `MetalAllocator::clone`
        // shares the underlying arena vec via `Arc<Mutex<…>>`, so a
        // weight allocated through the GpuWeights clone is reachable
        // via the GpuDevice clone — same `MTLBuffer`s, same offsets.
        let allocator = MetalAllocator::new((*device_arc).clone());
        let gpu_device = GpuDevice::new(device_arc.clone(), std::sync::Arc::new(allocator.clone()));
        let t_from_dir = std::time::Instant::now();
        let mut weights = GpuWeights::from_dir(&model_dir, allocator)
            .map_err(|e| ExecutorError::WorkerInit(format!("weight load failed: {e}")))?;
        info!(
            "ScratchyWorker(metal): GpuWeights::from_dir in {:?} ({} tensors)",
            t_from_dir.elapsed(),
            weights.len()
        );

        // Metal backend default dtype is bf16 — matches the on-disk
        // `torch_dtype: bfloat16` of every modern HF Llama / Qwen /
        // Phi / Mistral checkpoint and the cuda backend's native
        // dtype. Apple Silicon M3+ has hardware bf16 MMA; the metal
        // shaders ship `_bf16_specialized` siblings for every
        // on-path kernel and `MetalDtype::Bf16` is the default on
        // `CanonicalParams::METAL_DTYPE`. F16-on-disk weights get
        // cast up to bf16 here (lossy in the mantissa but matches
        // the kernel's expected binding type).
        weights.set_target_dtype(GpuDType::BF16);

        // Build HF fingerprint — same disambiguation surface cuda uses.
        // `max_position_embeddings` is intentionally omitted (see
        // `HfFingerprint`'s doc-comment).
        let hf_fp = scratchy_forward_compiler::HfFingerprint {
            rope_scaling_type: hf_config
                .extra
                .get("rope_scaling")
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str()),
            rope_scaling_hash: hf_config
                .extra
                .get("rope_scaling")
                .map(scratchy_forward_compiler::hash_json_value),
            rope_theta: hf_config.rope_theta,
        };

        let max_model_len = self
            .config
            .max_model_len
            .or(hf_config.max_position_embeddings())
            .unwrap_or(4096);

        // Under cfg(metal), `CUstream = ()`; the per-canonical
        // `Weights::load` body ignores the stream parameter (uploads go
        // through the allocator inside `GpuWeights`).
        let t_try_load = std::time::Instant::now();
        let model = scratchy_forward_compiler::try_load(
            &mut weights,
            (),
            arch.as_str(),
            1, // tp_world_size — metal is tp=1 only
            0, // tp_rank
            max_model_len,
            hf_fp,
        )
        .map_err(|e| ExecutorError::WorkerInit(format!("scratchy-forward-compiler load: {e}")))?
        .ok_or_else(|| ExecutorError::ArchNotSupported(arch.clone()))?;

        info!(
            "ScratchyWorker(metal): try_load in {:?} ({} via scratchy-forward-compiler, {})",
            t_try_load.elapsed(),
            arch,
            model.arch_name()
        );

        // Probe for a sibling `MultimodalForward` (vision tower) — same
        // arch filter as the text `try_load` above. Returns `Ok(None)`
        // for text-only arches and MM arches whose checkpoint has no
        // `visual.*` tensors. Consumed at forward time when the batch
        // carries `mm_data` (mirrors the cuda load path).
        #[cfg(feature = "vision")]
        let mm = scratchy_forward_compiler::try_load_mm(
            &mut weights,
            (),
            arch.as_str(),
            1, // tp_world_size — metal is tp=1 only
            0, // tp_rank
            max_model_len,
            hf_fp,
        )
        .map_err(|e| {
            ExecutorError::WorkerInit(format!("scratchy-forward-compiler MM load: {e}"))
        })?;
        #[cfg(feature = "vision")]
        if mm.is_some() {
            info!(
                "ScratchyWorker(metal): loaded {} vision encoder via scratchy-forward-compiler",
                arch
            );
        }

        {
            use std::sync::atomic::Ordering;
            let s = weights.metal_allocator().load_stats();
            let zc = s.zero_copy_calls.load(Ordering::Relaxed);
            let zb = s.zero_copy_bytes.load(Ordering::Relaxed);
            let zcr = s.zero_copy_relaxed_calls.load(Ordering::Relaxed);
            let zbr = s.zero_copy_relaxed_bytes.load(Ordering::Relaxed);
            let mc = s.memcpy_calls.load(Ordering::Relaxed);
            let mb = s.memcpy_bytes.load(Ordering::Relaxed);
            let small = s.memcpy_small_calls.load(Ordering::Relaxed);
            let med = s.memcpy_med_calls.load(Ordering::Relaxed);
            let large = s.memcpy_large_calls.load(Ordering::Relaxed);
            let unaligned = s.memcpy_unaligned.load(Ordering::Relaxed);
            let outside = s.memcpy_outside_mmap.load(Ordering::Relaxed);
            info!(
                "ScratchyWorker(metal): load routing — zero-copy {zc} calls / {:.1} MiB \
                 (of which {} calls / {:.1} MiB took the dtype-relaxed gate) | \
                 memcpy {mc} calls / {:.1} MiB ({} small <1MiB, {} med 1-16MiB, {} large ≥16MiB; \
                 fallback reason: {} unaligned, {} outside-mmap)",
                zb as f64 / (1 << 20) as f64,
                zcr,
                zbr as f64 / (1 << 20) as f64,
                mb as f64 / (1 << 20) as f64,
                small,
                med,
                large,
                unaligned,
                outside,
            );
            let hist: Vec<u64> = s
                .alignment_hist
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
            info!(
                "ScratchyWorker(metal): tensor-offset alignment histogram \
                 (#trailing-zeros → count): {:?}",
                hist
            );
        }

        // Compile + cache the greedy-sampling pipeline once. Argmax fires
        // outside the per-bucket ICB, so it owns its own pipeline cache
        // here on the worker rather than living in the per-canonical
        // `MetalWorkerPool`.
        let t_argmax = std::time::Instant::now();
        let argmax = scratchy_target_metal::argmax::ArgmaxKernels::new(&gpu_device.device)
            .map_err(|e| ExecutorError::WorkerInit(format!("argmax kernel compile: {e:?}")))?;
        info!(
            "ScratchyWorker(metal): ArgmaxKernels::new in {:?}",
            t_argmax.elapsed()
        );

        // On-GPU token sampler (temperature / top-k / top-p / min-p +
        // repetition/frequency/presence penalties). Built once here so the
        // first non-greedy decode step doesn't pay MSL-compile latency. Only
        // engaged when a request is non-greedy; greedy decode keeps the fused
        // argmax fast-path.
        let t_sampler = std::time::Instant::now();
        let sampler = scratchy_target_metal::sampling::SamplerKernels::new(&gpu_device.device)
            .map_err(|e| ExecutorError::WorkerInit(format!("sampler kernel compile: {e:?}")))?;
        info!(
            "ScratchyWorker(metal): SamplerKernels::new in {:?}",
            t_sampler.elapsed()
        );

        // Phase 6 chain-advance pipeline. Tiny kernel — one-time build
        // alongside argmax so the K-step chain driver never falls into
        // pipeline-compile latency on the first call.
        let t_chain = std::time::Instant::now();
        let chain_advance =
            scratchy_target_metal::chain_advance::ChainAdvanceKernel::new(&gpu_device.device)
                .map_err(|e| {
                    ExecutorError::WorkerInit(format!("chain_advance kernel compile: {e:?}"))
                })?;
        info!(
            "ScratchyWorker(metal): ChainAdvanceKernel::new in {:?}",
            t_chain.elapsed()
        );

        // Grammar-mask pipeline for constrained / guided decoding. Built
        // once here (negligible cost, no llguidance dependency) so the
        // first masked decode step never pays pipeline-compile latency.
        // The parser factory (llguidance, from tokenizer.json) is built
        // lazily on the first grammar request — see `ensure_grammar_factory`.
        #[cfg(feature = "guided-decoding")]
        {
            let t_gm = std::time::Instant::now();
            let grammar_mask =
                scratchy_target_metal::grammar_mask::GrammarMaskKernels::new(&gpu_device.device)
                    .map_err(|e| {
                        ExecutorError::WorkerInit(format!("grammar_mask kernel compile: {e:?}"))
                    })?;
            info!(
                "ScratchyWorker(metal): GrammarMaskKernels::new in {:?}",
                t_gm.elapsed()
            );
            self.grammar_mask_kernels = Some(grammar_mask);
        }

        self.gpu_device = Some(gpu_device);
        self.model = Some(model);
        #[cfg(feature = "vision")]
        {
            self.mm = mm;
        }
        self.argmax_kernels = Some(argmax);
        self.sampler_kernels = Some(sampler);
        self.chain_advance_kernel = Some(chain_advance);
        self.pooling_strategy = scratchy_core_model::embedding::resolve_pooling_strategy(
            &self.config.pooling_strategy,
            &model_dir,
        );
        self.model_dir = Some(model_dir);
        self.hf_config = Some(hf_config);
        self.resolved_architecture = Some(arch);

        // Load the speculative
        // draft model onto the same gpu_device (shared allocator +
        // residency set + command queue). KV pool for the draft is
        // allocated separately in `initialize_cache`.
        if self.config.draft_model_path.is_some() {
            self.load_draft_model_metal()?;
        }

        Ok(())
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        // TurboQuant ON BY DEFAULT (metal): the default `kv_cache_dtype` is
        // "auto". Resolve it to "turboquant" for arches TurboQuant supports
        // (power-of-2 head_dim <= 256; gemma4's 512 global group is handled by
        // the factory's GLOBAL gate). Unsupported arches (phi3 hd 96,
        // deepseek 56, ...) keep fp16. An explicit `--kv-cache-dtype` is honored
        // verbatim (only the literal "auto" default is upgraded here).
        if self.config.kv_cache_dtype == "auto" {
            let geom = self.model.as_ref().map(|m| {
                (
                    m.head_dim() as usize,
                    m.num_key_value_heads() as usize,
                    m.num_hidden_layers() as usize,
                )
            });
            if let Some((hd, nkv, layers)) = geom {
                // KV bytes per token, fp16, K and V across every layer —
                // the quantity TurboQuant exists to shrink.
                let kv_bytes_per_token = layers * nkv * hd * 2 /* K+V */ * 2 /* fp16 */;
                if !(hd.is_power_of_two() && hd <= 256) {
                    tracing::info!(
                        "ScratchyWorker(metal): head_dim {hd} unsupported by TurboQuant (needs power-of-2 <= 256) — using fp16 KV"
                    );
                } else if kv_bytes_per_token < TQ_MIN_KV_BYTES_PER_TOKEN {
                    // TurboQuant buys KV CAPACITY. A model whose whole KV
                    // row is a few KB does not need the capacity and should
                    // not pay the fidelity risk: `qwen2.5-0.5b` (24 layers
                    // x 2 kv x 64 = 12 KiB/token) decodes garbage under
                    // TurboQuant at the maximum 4 bits while fp16 KV is
                    // clean, and every component — flat kernels, paged
                    // kernels, codebook (outliers cost nothing), tape order,
                    // dispatch grids — measures correct in isolation. The
                    // cause is unidentified. Compressing 12 KiB/token was never
                    // worth it, so the honest default is fp16 and the open
                    // question is recorded rather than papered over.
                    tracing::info!(
                        "ScratchyWorker(metal): KV is {} KiB/token ({} layers x {nkv} kv x {hd}) — \
                         below the {} KiB TurboQuant threshold, using fp16 KV \
                         (compression buys little here; pass --kv-cache-dtype turboquant to force)",
                        kv_bytes_per_token / 1024,
                        layers,
                        TQ_MIN_KV_BYTES_PER_TOKEN / 1024
                    );
                } else {
                    self.config.kv_cache_dtype = "turboquant".to_string();
                    tracing::info!(
                        "ScratchyWorker(metal): TurboQuant KV ON by default (head_dim {hd}, \
                         {} KiB/token); pass --kv-cache-dtype fp16 to disable",
                        kv_bytes_per_token / 1024
                    );
                }
            }
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("model not loaded".into()))?;
        let device = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("gpu_device not initialized".into()))?;

        if self.config.kv_cache_dtype == "fp8_e4m3" || self.config.kv_cache_dtype == "fp8" {
            return Err(ExecutorError::WorkerInit(
                "FP8 KV cache not supported on metal — use F16".into(),
            ));
        }

        // Metal KV cache is f16; the per-canonical `forward` reads block
        // pointers via `KvCachePool::k_layer_mem` / `v_layer_mem` (added
        // in 3.E forward half) and binds them into the per-bucket
        // `RuntimeBindings.kv_cache_k/v` Buffer slots.
        let mtl_device = device.device.clone();
        // Mirror the model's resolved dtype so the KvCachePool's
        // tensor labels match what the kernel reads. bf16 weights →
        // bf16 cache; f16 weights → f16 cache. Byte size is the same
        // (2 bytes/elt) so this is purely a metadata fix; the
        // attention kernel binds the right `_bf16_specialized` /
        // `_f16_specialized` variant via `model.metal_dtype()`.
        let cache_dtype = match model.metal_dtype() {
            scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => GpuDType::BF16,
            scratchy_target_metal::interpreter::metal::MetalDtype::F16 => GpuDType::F16,
            scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                return Err(ExecutorError::WorkerInit(
                    "KV cache cannot be int4-quantized — must be bf16/f16".into(),
                ));
            }
        };

        // TurboQuant window (opt-in SCRATCHY_TQ_WINDOW): allocate the fp16 pool
        // at the BOUNDED window size, not the full logical capacity — this is
        // the resident-memory reduction. The packed store (built below) keeps
        // the full `num_gpu_blocks` logical capacity (4.6x smaller bytes); the
        // worker remaps logical->window every forward. capacity MUST be >= the
        // max single-forward working set (else resolve_table self-evicts).
        // Also require the supported geometry (power-of-2 head_dim <= 256) so an
        // unsupported model with SCRATCHY_TQ_WINDOW set doesn't shrink the pool
        // without the (skipped) window remap — see the tq_supported guard below.
        let tq_window_on = self.config.kv_cache_dtype == "turboquant"
            && std::env::var_os("SCRATCHY_TQ_WINDOW").is_some()
            && (model.head_dim() as usize).is_power_of_two()
            && (model.head_dim() as usize) <= 256;
        let tq_window_blocks = std::env::var("SCRATCHY_TQ_WINDOW_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(num_gpu_blocks)
            .clamp(1, num_gpu_blocks);
        let pool_blocks = if tq_window_on {
            tq_window_blocks
        } else {
            num_gpu_blocks
        };
        if tq_window_on {
            info!(
                "ScratchyWorker(metal): TurboQuant window — fp16 pool sized {pool_blocks} blocks (logical capacity {num_gpu_blocks}, packed-backed)"
            );
        }
        // Runtime per-sequence block-table capacity for this pool. Replaces the
        // compile-time `W::MAX_BLOCKS_PER_SEQ` (default 128 ≈ 2k tokens) so long
        // context isn't silently truncated. Stored on the pool; the host
        // block-table stride (execute_model), the kernel `MaxBlocksPerSeq`
        // function constant, and the rope-once scratch all read it back so the
        // three agree. Capped at `pool_blocks` (the blocks actually allocated).
        let pool_block_cap = self.kv_block_cap(pool_blocks);
        info!(
            "ScratchyWorker(metal): KV block-table capacity (max_blocks_per_seq) = {pool_block_cap} \
             (max_model_len-derived, pool {pool_blocks} blocks × {} tokens/block)",
            self.config.block_size,
        );

        // Hand newly-created KV cache buffers to the SHARED residency
        // set the allocator already owns (the same set covers the
        // weight arenas, every per-worker arena, and now KV). This is
        // the layout MLX uses — one `MTLResidencySet` per device queue,
        // not one per buffer category. Two residency sets per queue
        // (the previous layout) produced non-deterministic decode
        // output on Llama-3.2 even though every kernel passed its
        // golden in isolation.
        //
        // KvCachePool buffers are StorageModePrivate (~7.8 GiB total
        // for Llama-3.2-3B at 28 layers × 2 K/V × ~140 MiB), which
        // is exactly the working-set Apple's lazy paging tracker
        // drops out of residency under pressure. Without pinning,
        // attention reads race against the pager.
        let residency = device.allocator.residency().clone();

        let t_pool = std::time::Instant::now();
        // Reactive (chunked) KV pool with single-buffer-per-layer backing:
        // one `StorageModePrivate` MTLBuffer per `(layer, K/V)` sized to
        // span all possible chunks; "chunks" are byte offsets within it.
        // Apple's pager handles lazy physical commit on first GPU access,
        // so the floor RSS stays tied to the touched-chunk count rather
        // than the VA reservation. See `single_buffer_kv.rs` for the
        // rationale and `attention.metal`'s `ATTN_BLOCKS_PER_CHUNK==0`
        // branch for the kernel-side direct-addressing fast path that
        // matches this layout.
        let blocks_per_chunk = scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize;
        let num_layers_for_pool = model.num_hidden_layers() as usize;
        let num_chunks_total = num_gpu_blocks.div_ceil(blocks_per_chunk);
        let per_block_elems = (model.num_key_value_heads() as usize)
            * self.config.block_size
            * (model.head_dim() as usize);
        // Hybrid-attention-geometry arches (Gemma4): per-layer
        // kv_heads*head_dim from the macro-emitted IR walk; each layer's
        // buffers and pool chunks size to their own class.
        let per_layer_block_elems: Option<Vec<usize>> = model
            .per_layer_kv_token_elems()
            .map(|v| v.iter().map(|e| e * self.config.block_size).collect());
        let elem_bytes = cache_dtype.size_bytes();
        let chunk_bytes_logical = blocks_per_chunk * per_block_elems * elem_bytes;

        // vLLM group-shared hybrid KV layout (gemma4): infer the per-layer
        // sliding mask from `per_layer_kv_token_elems` (the bigger-page class
        // is the sliding class) and compute the grouping. This is `Some` only
        // for the page-differentiated case — the SAME trigger the scheduler's
        // `hybrid_kv` uses in `init`, so the two never disagree on group count.
        // `None` → one physical tensor per layer (the uniform pool).
        let hybrid_layout: Option<scratchy_core_config::HybridKvLayout> =
            model.per_layer_kv_token_elems().and_then(|elems| {
                let max_e = *elems.iter().max()?;
                let geom: Vec<scratchy_core_config::LayerKvGeometry> = elems
                    .iter()
                    .map(|&e| scratchy_core_config::LayerKvGeometry {
                        is_sliding: e == max_e,
                        // Page proxy: head_size 1 keeps the per-class page RATIO
                        // (only the ratio drives grouping + block_size scaling).
                        num_kv_heads: e,
                        head_size: 1,
                        head_size_v: None,
                        sliding_window: if e == max_e { Some(1) } else { None },
                    })
                    .collect();
                // `available` is irrelevant here — the pool sizes from the
                // passed `num_gpu_blocks`; we only read the grouping fields.
                scratchy_core_config::compute_hybrid_kv_layout(
                    &geom,
                    self.config.block_size,
                    usize::MAX / 2,
                    elem_bytes,
                )
            });
        // One physical tensor per group POSITION (gemma4: group_size=5) on the
        // hybrid path; one per layer otherwise.
        let num_tensors_for_pool = hybrid_layout
            .as_ref()
            .map(|l| l.group_size)
            .unwrap_or(num_layers_for_pool);
        // The full group's page-unified block size (Gemma4: 32) — the worker
        // encodes the full slot_mapping with this. `config.block_size` (uniform).
        self.kv_full_block_size = hybrid_layout
            .as_ref()
            .map(|l| l.full_block_size())
            .unwrap_or(self.config.block_size);
        self.kv_is_hybrid = hybrid_layout.is_some();

        let mut single_buf_layers: Vec<
            scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer,
        > = Vec::with_capacity(num_tensors_for_pool * 2);
        for slot in 0..(num_tensors_for_pool * 2) {
            // Slot s ↔ tensor s/2 (K at s%2==0, V at s%2==1) — must match the
            // pool's alloc order T0K, T0V, T1K, … (slot = c % n_slots). All
            // tensors are page-unified (same bytes) on the hybrid path, so
            // every slot uses `chunk_bytes_logical`; the uniform path with a
            // per-layer override (none today) keeps the per-layer sizing.
            let slot_chunk_bytes = if hybrid_layout.is_some() {
                chunk_bytes_logical
            } else {
                per_layer_block_elems
                    .as_ref()
                    .map(|v| blocks_per_chunk * v[slot / 2] * elem_bytes)
                    .unwrap_or(chunk_bytes_logical)
            };
            // UNIFORM TurboQuant: the fp16 pool is never read (tape ops use the
            // packed store + scratch) and never grown, so reserve only ONE chunk
            // instead of the full num_chunks_total. That drops the per-layer fp16
            // VA reservation (~13 GB) to a seed — the packed store (~4.7x smaller)
            // is the canonical cache.
            // 🛑 HYBRID SWA (gemma4): only the GLOBAL layers are TurboQuant'd; the
            // FP16 SLIDING layers DO read this (shared) pool and grow it past the
            // seed once their blocks cross a chunk boundary (BLOCKS_PER_CHUNK=128,
            // ~384 tokens for gemma4-12b). A 1-chunk seed makes grow_to_cover fail
            // ("commit_through target=2 > max_chunks=1") → the sliding KV in chunk
            // 1+ reads a stale chunk-address-table entry → garbage. So the hybrid
            // path needs the full capacity; VA is lazily paged (Apple pager), so
            // the untouched global tensors cost no physical memory — the SWA win
            // (global packed + sliding fp16-windowed) is preserved.
            let max_chunks =
                if self.config.kv_cache_dtype == "turboquant" && hybrid_layout.is_none() {
                    1
                } else {
                    num_chunks_total
                };
            let layer = scratchy_target_metal::single_buffer_kv::SingleBufferKvLayer::new(
                &mtl_device,
                slot_chunk_bytes,
                max_chunks,
            )
            .map_err(|e| ExecutorError::WorkerInit(format!("SingleBufferKvLayer: {e}")))?;
            residency.insert(layer.buffer());
            single_buf_layers.push(layer);
        }
        let mb_per_layer = (chunk_bytes_logical * num_chunks_total) as f64 / (1024.0 * 1024.0);
        info!(
            "ScratchyWorker(metal): KV layer buffers — {} buffers \
             ({} chunks × {} bytes each, {:.0} MiB VA per layer, lazy via Apple pager)",
            single_buf_layers.len(),
            num_chunks_total,
            chunk_bytes_logical,
            mb_per_layer,
        );

        let pool = unsafe {
            // Closure-shared call counter — selects which layer buffer this
            // `alloc_chunk` call belongs to via `counter % (2*num_layers)`.
            // Works for both init (`initial_chunks=1` → exactly `2*num_layers`
            // calls, one per slot) and grow (each grow step issues
            // `2*num_layers` calls in slot order, repeated per added chunk).
            let alloc_chunk_counter = std::cell::Cell::new(0usize);
            let single_buf_ref = &mut single_buf_layers;
            let n_slots = num_tensors_for_pool * 2;
            KvCachePool::new_metal_chunked(
                num_layers_for_pool,
                pool_blocks,
                self.config.block_size,
                model.num_key_value_heads() as usize,
                model.head_dim() as usize,
                pool_block_cap,
                // Hybrid: all tensors page-unified → uniform per_block_elems
                // (None falls back to num_kv_heads·block_size·head_dim, which
                // equals the unified size for gemma4). Uniform path unchanged.
                if hybrid_layout.is_some() {
                    None
                } else {
                    per_layer_block_elems.clone()
                },
                // vLLM group-shared mapping: layer → physical tensor (position).
                hybrid_layout.as_ref().map(|l| l.layer_to_tensor.clone()),
                cache_dtype,
                blocks_per_chunk,
                // Reactive: 1 chunk up front; the rest grow on demand via
                // `grow_metal_kv_to_cover` (called from execute_model each
                // step before the forward).
                1,
                |bytes| {
                    let c = alloc_chunk_counter.get();
                    alloc_chunk_counter.set(c + 1);
                    let slot = c % n_slots;
                    let layer = &mut single_buf_ref[slot];
                    let chunk_idx = layer.committed_chunks();
                    layer.commit_through(chunk_idx + 1).map_err(|e| {
                        anyhow::anyhow!("KV chunk commit slot={slot} chunk={chunk_idx}: {e}")
                    })?;
                    Ok(MetalMem::from_buffer_with_offset(
                        layer.buffer_clone(),
                        chunk_idx * layer.chunk_bytes(),
                        bytes,
                    ))
                },
                // Chunk-address table: `StorageModeShared` so the CPU can
                // write the chunk gpuAddresses (`fill_chunk_tables` below).
                // Tiny — 8 bytes/chunk — so a regular allocation is fine.
                |bytes| {
                    let buffer = mtl_device
                        .newBufferWithLength_options(
                            bytes,
                            ::objc2_metal::MTLResourceOptions::StorageModeShared,
                        )
                        .expect("newBufferWithLength_options returned nil");
                    residency.insert(&buffer);
                    Ok(MetalMem::from_buffer(buffer))
                },
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("KvCachePool: {e}")))?;
        // Attach the vLLM KV-group layout (per-layer → group) so the runtime
        // bindings resolve each layer to its group's block_table / slot_mapping.
        let mut pool = pool;
        if let Some(layout) = hybrid_layout.as_ref() {
            pool.set_kv_group_layout(layout.num_groups(), layout.layer_to_group_u32());
            info!(
                "ScratchyWorker(metal): hybrid KV — {} layers / {} physical tensors / {} groups \
                 (full block_size {})",
                num_layers_for_pool,
                num_tensors_for_pool,
                layout.num_groups(),
                self.kv_full_block_size,
            );
        }
        self.target_kv_single_buffers = single_buf_layers;
        // Commit the residency set NOW (before reading gpuAddresses):
        // the chunk buffers are referenced only by raw address from the
        // tables, so they must be resident, and we read each chunk's
        // gpuAddress *after* commit to be sure the VA is finalized.
        // (The pool's lazy commit at first forward re-commits + attaches
        // — idempotent.)
        residency.commit();
        // Populate each per-layer chunk-address table with its chunks'
        // GPU virtual addresses. `m.gpu_address()` handles BOTH the
        // dense path (returns buffer.gpuAddress()) AND the sparse path
        // (returns sparse_base + metal_offset).
        pool.fill_chunk_tables(|m| m.gpu_address());
        info!(
            "ScratchyWorker(metal): init_cache phases — KvCachePool::new_metal_chunked({} blocks/chunk) {:?} (residency.commit deferred to first forward)",
            blocks_per_chunk,
            t_pool.elapsed(),
        );
        // Attach the shared residency set to the device's queue so
        // every cmdbuf sees both arenas + KV-cache as wired. The
        // pool's lazy attach in `forward()` will commit the queued
        // KV inserts and attach the set on the first forward — moves
        // ~125ms of residency.commit() out of the init path. Attach
        // is idempotent per (queue, set) so the lazy path is safe
        // even if some other call has already attached.

        info!(
            "ScratchyWorker(metal): KV cache initialized: {} layers × {} blocks × {} tokens",
            model.num_hidden_layers(),
            num_gpu_blocks,
            self.config.block_size,
        );
        self.kv_cache = Some(pool);

        // TurboQuant KV compression — gated on `kv_cache_dtype == "turboquant"`,
        // OFF by default (the fp16 pool above is unchanged). Build the runtime
        // (kernels + codebook + per-layer packed store); the post-forward hook
        // then compresses each step's new KV slots into the ~4.6x packed store
        // and dequants them back into the fp16 pool so attention reads
        // TurboQuant'd KV. (Uniform geometry; gemma4 per-layer hybrid is a
        // follow-up — head_dim must be a power of two.)
        // TurboQuant supports only power-of-2 head_dim <= 256 (the Walsh-Hadamard
        // rotation needs a power-of-2 length; the kernel threadgroup caps at 256
        // threads). Models like phi3 (hd 96), deepseek_v3 (hd 56), kimi_k2 (hd
        // 112) fall OUTSIDE that — for those we must NOT build the runtime (it
        // would panic in PolarQuantizer); fall back to plain fp16 + warn. This is
        // what makes turboquant safe to request/default on any model.
        let tq_hd = model.head_dim() as usize;
        let tq_supported = tq_hd.is_power_of_two() && tq_hd <= 256;
        if self.config.kv_cache_dtype == "turboquant" && !tq_supported {
            tracing::warn!(
                "ScratchyWorker(metal): TurboQuant unsupported for head_dim {tq_hd} (needs power-of-2 <= 256) — falling back to fp16 KV cache"
            );
        }
        if self.config.kv_cache_dtype == "turboquant" && tq_supported {
            // The per-layer TurboQuant tape ops (factory-provisioned via
            // ForwardCtx::kv_turboquant) now own compress + dequant. The old
            // in-place TurboQuantRuntime + window map are superseded;
            // `self.turboquant` stays None so every old hook below skips.
            info!(
                "ScratchyWorker(metal): TurboQuant — per-layer tape ops active (factory-provisioned)"
            );
        }

        // Gated-DeltaNet (Qwen3.5 / Qwen3-Next) recurrent-state pool —
        // the non-paged sibling of the KV cache. Only hybrid arches
        // report a `gdn_runtime_config` (macro-emitted from the unrolled
        // IR's per-layer GDN dispatch); every other arch leaves
        // `self.gdn_state` / `self.gdn_slot_allocator` `None`. One
        // recurrent-state slot per concurrently-resident sequence
        // (`max_num_seqs`), matching the scheduler's `max_num_running_reqs`
        // so the slot allocator never exhausts.
        if let Some(gdn_cfg) = model.gdn_runtime_config() {
            let num_slots = self.config.max_num_seqs.max(1);
            let num_layers = model.num_hidden_layers() as usize;
            let t_gdn = std::time::Instant::now();
            let gdn_pool = unsafe {
                scratchy_target_metal::gdn_state::GdnStatePool::new(
                    num_layers,
                    &gdn_cfg.linear_layers,
                    num_slots,
                    gdn_cfg.conv_dim as usize,
                    gdn_cfg.conv_kernel as usize,
                    gdn_cfg.num_v_heads as usize,
                    gdn_cfg.head_v_dim as usize,
                    gdn_cfg.head_k_dim as usize,
                    |bytes| {
                        // f32 conv/ssm state. MUST be StorageModeShared:
                        // `MetalMem::from_buffer` takes `contents()` and
                        // every per-layer conv/ssm pointer derives from
                        // that CPU base. The old StorageModePrivate alloc
                        // "worked" only because pre-26.5.1 drivers handed
                        // out a CPU-mappable pointer for Private UMA
                        // memory anyway; macOS 26.5.1 stopped doing that
                        // for LARGE allocations (dedicated unmapped VM),
                        // so the Qwen3.5-MoE-35B pool (~500 MB) silently
                        // built wild per-layer addresses → garbage GDN
                        // state → degenerate logits ("!!!!"), while small
                        // pools (0.8B/9B, heap-suballocated and still
                        // mapped) kept working. Metal validation layer
                        // names it: `validateCPUWriteable` assert in
                        // `MetalMem::from_buffer`. Shared is identical
                        // bandwidth on UMA. Pinned into the same shared
                        // residency set as the KV pool so the lazy pager
                        // can't drop it mid-attention.
                        let buffer = mtl_device
                            .newBufferWithLength_options(
                                bytes,
                                ::objc2_metal::MTLResourceOptions::StorageModeShared,
                            )
                            .expect("GDN state buffer alloc returned nil");
                        residency.insert(&buffer);
                        Ok(MetalMem::from_buffer(buffer))
                    },
                )
            }
            .map_err(|e| ExecutorError::WorkerInit(format!("GdnStatePool: {e}")))?;
            info!(
                "ScratchyWorker(metal): GDN state pool — {} linear / {} layers × {} slots in {:?}",
                gdn_cfg.num_linear_layers(),
                num_layers,
                num_slots,
                t_gdn.elapsed(),
            );
            self.gdn_state = Some(gdn_pool);
            self.gdn_slot_allocator = Some(GdnSlotAllocator::new(num_slots));
        }

        // Allocate the draft model's
        // KV pool alongside the target's, sized to fit inside the same
        // shared `recommendedMaxWorkingSetSize` budget.
        if self.draft_model.is_some() {
            self.initialize_draft_cache_metal(num_gpu_blocks)?;
        }

        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        // Apple unified memory: `recommendedMaxWorkingSetSize` is Apple's
        // own recommended budget for resident MTLBuffers (typically ~75%
        // of physical RAM on M-series, accounting for OS reservations).
        // `currentAllocatedSize` covers everything Metal has allocated
        // for this process so far — model weights live there
        // post-`load_model`.
        //
        // Peak activation: the metal pool's per-worker arena is sized
        // for the elementwise-max across every bucket spec
        // (`for_buckets` does the max). We expose that per-canonical
        // sum via `ScratchyWeights::metal_arena_peak_bytes()` (emitted
        // by the per-canonical macro from the colored slot map at
        // expansion time). For metal we currently run at most
        // `max_workers = 1`, so peak == arena_peak. Runtime bindings
        // (input_ids/positions/etc.) and the per-step staging buffers
        // we allocate inside `execute_model` are <10 MiB and dwarfed
        // by the arena; we add a 64 MiB pad as a conservative bound.
        let metal_device = self
            .metal_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("metal device not initialized".into()))?;
        // `recommendedMaxWorkingSetSize` (~75% of physical RAM) OVERSTATES
        // what a single command buffer can actually keep resident: on a
        // 32 GiB M-series it reports 25 GiB, but the GPU faults
        // `kIOGPUCommandBufferCallbackErrorOutOfMemory` when the prefill CB
        // references more than ~`maxBufferLength` (18.72 GiB) of wired
        // memory — the real wireable ceiling. (mlx-lm runs gemma-3-31b-4bit
        // here peaking at 17.55 GiB and succeeds; scratchy budgeting to
        // 22.5 GiB = 25·0.9 OOMs.) Cap the budget at `maxBufferLength` so
        // arena + KV are sized from the headroom under the WIREABLE limit,
        // not the inflated working-set figure. We apply `gpu_memory_utilization`
        // only to the working-set term and then floor at `maxBufferLength`:
        // applying util to maxBufferLength too would push the budget below
        // the weight footprint on this box and falsely trip the OOM guard.
        let working_set = metal_device.device.recommendedMaxWorkingSetSize() as usize;
        // Budget = recommended working set × `gpu_memory_utilization` — the
        // same utilization-governed sizing every other backend uses.
        //
        // We used to additionally floor this at `maxBufferLength − 1.2 GiB`
        // (~17.5 GiB on a 32 GiB M-series) on the theory that one command
        // buffer cannot keep more than ~`maxBufferLength` of memory wired. That
        // was over-conservative: a single command buffer can reference a wired
        // set spread across MANY buffers far past `maxBufferLength` (a probe
        // densely streamed 24 GiB in one CB on an M5 with no OOM) — the binding
        // limit is TOTAL device residency, not the single-buffer length. The
        // floor was starving KV: gemma-4-26b got only 5,712 tokens (1.3 GiB)
        // when `working_set·util` leaves room for ~22k. Dropping it takes
        // gemma-4-26b to 21,984 tokens with no OOM on a 32 GiB M5.
        // `gpu_memory_utilization` (default 0.9) is the headroom knob — lower it
        // for a model that trips `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
        let total = (working_set as f64 * self.config.gpu_memory_utilization) as usize;
        // Reserve room for the Gated-DeltaNet recurrent-state pool. It is
        // built in `initialize_cache` (which runs AFTER this), so it is
        // NOT yet in `currentAllocatedSize`; fold it into the non-KV
        // overhead here so the engine doesn't hand back KV blocks that
        // leave no room for it. Persistent f32 state, one slot per
        // resident seq — sized identically to the pool built later. Zero
        // for non-hybrid arches.
        let gdn_reserve = self
            .model
            .as_ref()
            .and_then(|m| m.gdn_runtime_config())
            .map(|cfg| {
                scratchy_target_metal::gdn_state::GdnStatePool::<scratchy_target_metal::PoolMem>::reserve_bytes(
                    cfg.num_linear_layers(),
                    self.config.max_num_seqs.max(1),
                    cfg.conv_dim as usize,
                    cfg.conv_kernel as usize,
                    cfg.num_v_heads as usize,
                    cfg.head_v_dim as usize,
                    cfg.head_k_dim as usize,
                )
            })
            .unwrap_or(0);
        let weights_and_overhead = metal_device
            .device
            .currentAllocatedSize()
            .saturating_add(gdn_reserve);
        // ── Target-reactive prefill-bucket selection ─────────────────────
        // The forward macro now compiles a full bucket ladder for every arch
        // (no per-model `workloads` cap). Pick the largest bucket THIS device
        // can afford: the activation arena may use up to ARENA_FRACTION of the
        // headroom left after weights + GDN reserve + pads, and KV gets the
        // rest. The chosen cap is stashed on the GpuDevice the forward path
        // uses, so the lazy `MetalWorkerPool::for_buckets` prunes the ladder to
        // match — the same binary runs 512 on a 32 GiB box and 4096 on a
        // 192 GiB one, with nothing Apple-specific about the policy. An empty
        // cost table (older arches that never emitted it) falls back to the
        // prior "reserve the full-ladder arena peak" behavior.
        const ARENA_FRACTION: f64 = 0.6;
        let bucket_costs: &'static [(u32, u64)] = self
            .model
            .as_ref()
            .map(|m| m.metal_bucket_arena_costs())
            .unwrap_or(&[]);
        let selection = if bucket_costs.is_empty() {
            None
        } else {
            // `pad` mirrors the non-arena, non-KV terms the KV formula
            // subtracts (64 MiB runtime/staging + 150 MiB redundancy) so the
            // headroom the selector splits equals what is actually left.
            let pad = (64 + 150) * 1024 * 1024u64;
            // `total` is ALREADY the wireable budget (working_set·util capped
            // at maxBufferLength), so use it directly — do NOT re-apply
            // `utilization` here or it would shrink the budget twice.
            let budget = total as u64;
            let fixed = (weights_and_overhead as u64).saturating_add(pad);
            let sel = select_prefill_bucket(budget, fixed, bucket_costs, ARENA_FRACTION);
            tracing::info!(
                "ScratchyWorker(metal): target-reactive prefill bucket = {} \
                 (arena {:.1} MiB, KV ~{:.1} MiB; candidate ladder {:?} pruned \
                 to fit {:.2} GiB budget)",
                sel.max_bucket_m,
                sel.arena_bytes as f64 / 1_048_576.0,
                sel.kv_bytes as f64 / 1_048_576.0,
                bucket_costs.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
                budget as f64 / 1_073_741_824.0,
            );
            // Stash the cap on the metal `gpu_device` (the GpuDevice every
            // metal forward path runs through — `self.device` is the
            // cuda-only field). Whichever forward triggers the one-shot lazy
            // pool init reads it and prunes the ladder.
            if let Some(dev) = self.gpu_device.as_mut() {
                dev.metal_bucket_max_m = Some(sel.max_bucket_m);
            }
            self.metal_prefill_bucket_max_m = Some(sel.max_bucket_m);
            Some(sel)
        };
        let arena_peak = match selection {
            Some(sel) => sel.arena_bytes as usize,
            None => self
                .model
                .as_ref()
                .map(|m| m.metal_arena_peak_bytes() as usize)
                .unwrap_or(512 * 1024 * 1024),
        };
        // When a draft model is loaded we also need an activation arena for
        // it. Decode-only forwards on a 1B model peak well below the target,
        // but the target's prefill (M >> 1) arena is the worst case across
        // the pair — use it for both as a safe upper bound.
        let arena_peak_pair = if self.draft_model.is_some() {
            arena_peak.saturating_mul(2)
        } else {
            arena_peak
        };
        let peak_activation_estimate = arena_peak_pair.saturating_add(64 * 1024 * 1024);
        // `total` already folds in `gpu_memory_utilization` and is capped at
        // the wireable `maxBufferLength`, so pass util=1.0 here — applying it
        // again would shrink the KV budget a second time below the headroom
        // the bucket selector just split.
        let available =
            compute_available_kv_bytes(total, weights_and_overhead, peak_activation_estimate, 1.0);
        // Per-pair split: when a draft model is loaded, every target KV
        // block has a 1:1 mirror in the draft pool, so the engine should
        // think it has only `target / (target + draft)` of the budget.
        // After the engine divides by `target_per_block_bytes` to land on
        // num_gpu_blocks, the same count of draft blocks fits inside the
        // remaining `draft / (target + draft)` slice.
        let (available_reported, draft_reservation) = if let Some(draft) = self.draft_model.as_ref()
        {
            let target = self.model.as_ref().expect("model loaded before draft");
            let bs = self.config.block_size;
            let t_pb = kv_per_block_bytes(target.as_ref(), bs);
            let d_pb = kv_per_block_bytes(draft.as_ref(), bs);
            let denom = t_pb.saturating_add(d_pb).max(1);
            // available * t_pb / (t_pb + d_pb), in u128 to dodge overflow.
            let scaled = (available as u128 * t_pb as u128 / denom as u128) as usize;
            (scaled, available.saturating_sub(scaled))
        } else {
            (available, 0)
        };
        info!(
            "ScratchyWorker(metal): total={:.1} GiB, weights+overhead={:.1} GiB, \
             arena_peak={:.1} MiB (pair={:.1} MiB), kv_budget={:.1} GiB \
             (target_share={:.1} GiB, draft_reserve={:.1} GiB)",
            total as f64 / 1_073_741_824.0,
            weights_and_overhead as f64 / 1_073_741_824.0,
            arena_peak as f64 / 1_048_576.0,
            arena_peak_pair as f64 / 1_048_576.0,
            available as f64 / 1_073_741_824.0,
            available_reported as f64 / 1_073_741_824.0,
            draft_reservation as f64 / 1_073_741_824.0,
        );
        // GUARD: if what's ALREADY allocated
        // exceeds the device budget, no KV clamp can save this process —
        // command buffers will OOM at execution and (before the commit-
        // feedback guard) produced silent all-zero forwards. Qwen3.5-MoE
        // -35B hit this on macOS 26.5.1: weights+overhead=34.9 GiB >
        // total=25 GiB and the engine limped into garbage. Refuse loudly
        // with the allocation breakdown so the excess is attributable.
        if weights_and_overhead.saturating_add(peak_activation_estimate) > total {
            let (regions, arena_cap, arena_used) = self
                .gpu_device
                .as_ref()
                .map(|g| g.allocator.allocation_breakdown())
                .unwrap_or((0, 0, 0));
            // Split the arena figure by pool. Summed, it reads as "the
            // loader copied this much" — on mixtral it said 23.63 GiB
            // while the loader's routing counters said 10 MiB of
            // memcpy, because most of it is the ACTIVATION arena.
            let (scratch_cap, scratch_used, weight_cap, weight_used) = self
                .gpu_device
                .as_ref()
                .map(|g| g.allocator.arena_breakdown_by_pool())
                .unwrap_or((0, 0, 0, 0));
            // When the GDN recurrent-state reservation is what tips the
            // budget (it scales linearly with --max-num-seqs; 61 MiB/slot
            // on Qwen3.5-35B), compute the slot count that WOULD fit and
            // put the exact flag in the error. Qwen3.5-MoE-35B on a
            // 32 GiB box: 256 default slots = 15.7 GiB reserve = the
            // entire macOS 26.5.1 "!!!!" incident; 8 slots = 0.5 GiB and
            // the model runs with 2.5 GiB of KV.
            let flag_hint = if gdn_reserve > 0 {
                let per_slot = gdn_reserve / self.config.max_num_seqs.max(1);
                let base = weights_and_overhead
                    .saturating_sub(gdn_reserve)
                    .saturating_add(peak_activation_estimate);
                // Leave at least 1 GiB for KV after the pool.
                let headroom = total.saturating_sub(base).saturating_sub(1 << 30);
                let affordable = (headroom / per_slot.max(1)).max(1);
                if affordable < self.config.max_num_seqs {
                    format!(
                        " The GDN state pool ({:.2} GiB) is sized by \
                         --max-num-seqs={}; pass --max-num-seqs {} (or fewer) \
                         to fit this model on this device.",
                        gdn_reserve as f64 / 1_073_741_824.0,
                        self.config.max_num_seqs,
                        affordable.min(64),
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            return Err(ExecutorError::WorkerInit(format!(
                "model does not fit: allocated weights+overhead \
                 ({:.1} GiB) + activation estimate ({:.1} MiB) exceed the \
                 device working-set budget ({:.1} GiB). Breakdown: mmap \
                 shard buffers {:.2} GiB, arenas {:.2} GiB capacity \
                 ({:.2} GiB used) = scratch {:.2}/{:.2} GiB + weight-memcpy \
                 {:.2}/{:.2} GiB (used/cap), gdn_reserve {:.2} GiB.{flag_hint}",
                weights_and_overhead as f64 / 1_073_741_824.0,
                peak_activation_estimate as f64 / 1_048_576.0,
                total as f64 / 1_073_741_824.0,
                regions as f64 / 1_073_741_824.0,
                arena_cap as f64 / 1_073_741_824.0,
                arena_used as f64 / 1_073_741_824.0,
                scratch_used as f64 / 1_073_741_824.0,
                scratch_cap as f64 / 1_073_741_824.0,
                weight_used as f64 / 1_073_741_824.0,
                weight_cap as f64 / 1_073_741_824.0,
                gdn_reserve as f64 / 1_073_741_824.0,
            )));
        }
        Ok(available_reported)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        // Release wired GPU residency if the forward panics: a worker-thread
        // panic raises no signal under `panic = "unwind"`, so it bypasses the
        // graceful teardown and would otherwise leave the process pinning the
        // wired KV/weight/scratch pages (see `ResidencyPanicGuard`). Capture the
        // same sets `shutdown()` releases; no-op on normal return.
        let _residency_panic_guard = ResidencyPanicGuard {
            sets: self
                .gpu_device
                .as_ref()
                .map(|dev| {
                    vec![
                        dev.allocator.residency().clone(),
                        dev.allocator.weights_residency().clone(),
                    ]
                })
                .unwrap_or_default(),
        };

        // ── 1. Lifecycle: drop finished requests ──────────────────
        for req_id in &scheduler_output.finished_req_ids {
            self.token_buffers.remove(req_id);
            self.prompt_lengths.remove(req_id);
            self.annotation_buffers.remove(req_id);
            self.reused_block_buffers.remove(req_id);
            self.mm_data_buffers.remove(req_id);
            self.sampling_params_map.remove(req_id);
            self.seeded_rngs.remove(req_id);
            #[cfg(feature = "guided-decoding")]
            self.grammar_states.remove(req_id);
            // Recycle the request's GDN recurrent-state slot (hybrid
            // arches only). The slot's state is now stale; the next
            // request to claim it is flagged fresh so it zero-inits
            // rather than continuing from a finished sequence's state.
            if let Some(alloc) = self.gdn_slot_allocator.as_mut() {
                alloc.release(gdn_slot_key(req_id));
            }
        }
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // ── 2. Lifecycle: add newly scheduled requests ────────────
        for new_req in &scheduler_output.scheduled_new_reqs {
            let num_tokens = scheduler_output
                .num_scheduled_tokens
                .get(&new_req.req_id)
                .copied()
                .unwrap_or(0);
            if num_tokens == 0 {
                continue;
            }
            let prompt_ids = new_req.prompt_token_ids.as_deref().unwrap_or(&[]);
            let start = new_req.num_computed_tokens as usize;
            let end = (start + num_tokens).min(prompt_ids.len());
            let tokens_to_use = &prompt_ids[start..end];

            self.token_buffers
                .insert(new_req.req_id.clone(), prompt_ids.to_vec());
            self.prompt_lengths
                .insert(new_req.req_id.clone(), prompt_ids.len());
            if let Some(ref params) = new_req.sampling_params {
                self.sampling_params_map
                    .insert(new_req.req_id.clone(), params.clone());
                // Seeded sampling: a request with an explicit RNG seed gets a
                // per-request StdRng so its `uniform_random` draws are
                // reproducible across steps (unseeded requests derive the
                // uniform deterministically from a req_id+step hash in the
                // sample block — matches cuda's `gen_seed`).
                if let Some(seed) = params.seed {
                    use rand::SeedableRng;
                    self.seeded_rngs.insert(
                        new_req.req_id.clone(),
                        rand::rngs::StdRng::seed_from_u64(seed),
                    );
                }
                // Constrained / guided decoding: build this request's grammar
                // FSM up front so the very first sampled token is masked. The
                // parser factory is built lazily on first use. On any failure
                // the request is left unconstrained (logged) rather than failed.
                #[cfg(feature = "guided-decoding")]
                if let Some(ref grammar) = params.guided_grammar {
                    self.ensure_grammar_factory();
                    if let Some(ref factory) = self.grammar_factory {
                        match scratchy_core_model::grammar::GrammarGuide::from_guided_grammar(
                            grammar, factory,
                        ) {
                            Ok(guide) => {
                                self.grammar_states.insert(new_req.req_id.clone(), guide);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "guided-decoding: build grammar for req {}: {e}; \
                                     decoding unconstrained",
                                    new_req.req_id
                                );
                            }
                        }
                    }
                }
            }
            // Stash multimodal pixel data (first scheduling only) so the
            // vision tower runs on this request's first prefill step.
            if let Some(ref mm) = new_req.mm_data {
                self.mm_data_buffers
                    .insert(new_req.req_id.clone(), mm.clone());
            }
            // Spans (rope-on-read): stash the per-block relocatable
            // annotations so execute_model can OR bit 31 into the
            // block_table / slot_mapping for span blocks. Only when the
            // loaded model's kernels mask bit 31 (W::ROPE_ON_READ) — else
            // a stray bit 31 would corrupt the physical block id.
            if self.rope_on_read_active() {
                if let Some(ref ann) = new_req.block_annotations {
                    // Spans rope-on-read + TurboQuant compose: dequant runs first
                    // (codes → K_unrotated), THEN rope-on-read rotates to the reuse
                    // position. Rotating the dequantized K (vs quantizing the
                    // rotated K) only adds quant noise of the same magnitude
                    // TurboQuant already accepts — and on a REUSE there is no
                    // rotate-on-write twin being recomputed, so there is nothing to
                    // "diverge" from; the reused block IS the source of truth.
                    // (Coherence validated end-to-end.) The earlier structural
                    // refusal here was over-conservative.
                    self.annotation_buffers
                        .insert(new_req.req_id.clone(), ann.clone());
                }
                // Phase 2: reused-hit blocks whose KV the worker should NOT
                // rewrite (the shared cached block already holds valid K).
                if let Some(ref reused) = new_req.reused_block_idxs {
                    self.reused_block_buffers
                        .insert(new_req.req_id.clone(), reused.iter().copied().collect());
                }
            }
            // Group 0 = full (global layers); groups 1.. = the sliding groups
            // (gemma4 SWA), present only for hybrid models (vLLM group-shared).
            let block_ids = new_req.block_ids.first().cloned().unwrap_or_default();
            let sliding_groups: Vec<Vec<usize>> =
                new_req.block_ids.iter().skip(1).cloned().collect();
            self.input_batch.add_request_hybrid(
                new_req.req_id.clone(),
                tokens_to_use,
                block_ids,
                sliding_groups,
                new_req.num_computed_tokens,
            );
        }

        // ── 3. Lifecycle: refresh cached requests' block tables ───
        for (i, req_id) in scheduler_output
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .enumerate()
        {
            if let Some(Some(new_blocks)) =
                scheduler_output.scheduled_cached_reqs.new_block_ids.get(i)
                && let Some(group0) = new_blocks.first()
            {
                let sliding: Vec<Vec<usize>> = new_blocks.iter().skip(1).cloned().collect();
                self.input_batch
                    .update_blocks_hybrid(req_id, group0.clone(), sliding);
            }
        }
        let cached = &scheduler_output.scheduled_cached_reqs;
        if !cached.new_token_ids.is_empty() {
            for (i, req_id) in cached.req_ids.iter().enumerate() {
                if let Some(tokens) = cached.new_token_ids.get(i)
                    && let Some(&last_token) = tokens.last()
                {
                    self.input_batch.set_last_token(req_id, last_token);
                }
            }
        }

        // ── 3b. Re-arm chunked-prefill continuations as prefill ───
        // A cached (running) request that still has un-prefilled prompt
        // tokens must run THIS step as a PREFILL chunk, not a decode. The
        // previous chunk's commit_step cleared is_prefill, so without this
        // re-arm prepare_inputs would emit a single decode token (q_len=1)
        // and DROP the rest of the prompt — including the user's question —
        // producing garbage / a prompt echo on every prompt longer than
        // max_num_batched_tokens (the chunked-prefill threshold). Unlike the
        // CUDA worker (execute_model_inner), the metal path previously had no
        // re-arm at all, so all chunked prefill was silently broken.
        //
        // The request is still mid-prefill iff `num_computed < prompt_len`
        // (matches Python vLLM, where a request is "in prefill" while
        // num_computed_tokens < num_prompt_tokens). This INCLUDES the final
        // chunk (num_computed + num_scheduled == prompt_len). The block table
        // (step 3 above) and tokens_in_pool (prior commit_step) are already
        // up to date, so set_prefill_continuation — flip is_prefill + set this
        // chunk's tokens + pos_offset — is sufficient.
        for (i, req_id) in cached.req_ids.iter().enumerate() {
            if cached.resumed_req_ids.contains(req_id) {
                continue;
            }
            let prompt_len = self.prompt_lengths.get(req_id).copied().unwrap_or(0);
            let num_computed = cached.num_computed_tokens.get(i).copied().unwrap_or(0) as usize;
            if num_computed >= prompt_len {
                continue; // prefill complete → normal decode, leave as-is
            }
            let num_scheduled = scheduler_output
                .num_scheduled_tokens
                .get(req_id)
                .copied()
                .unwrap_or(0);
            if num_scheduled == 0 {
                continue;
            }
            let tokens = match self.token_buffers.get(req_id) {
                Some(buf) => {
                    let start = num_computed;
                    let end = (start + num_scheduled).min(buf.len());
                    buf[start..end].to_vec()
                }
                None => continue,
            };
            self.input_batch
                .set_prefill_continuation(req_id, tokens, num_computed as u32);
        }

        if self.input_batch.num_active() == 0 {
            // Reactive KV (2c): batch fully idle → no live blocks. Defer the
            // shrink until the GPU has been idle for `KV_SHRINK_IDLE_AFTER`.
            // Shrinking on every idle (i.e. after every request) drops the
            // grown chunks' StorageModePrivate pages; the next request then
            // re-faults them on first GPU touch — a multi-second stall on a
            // long prefill. Deferring keeps back-to-back requests fast while
            // still releasing memory once genuinely idle.
            #[cfg(feature = "metal")]
            {
                let now = std::time::Instant::now();
                let idle_since = *self.kv_idle_since.get_or_insert(now);
                if now.duration_since(idle_since) >= KV_SHRINK_IDLE_AFTER {
                    self.shrink_metal_kv_idle();
                }
            }
            return Ok(ModelRunnerOutput::empty());
        }

        // A batch is active again — cancel any pending idle-shrink timer.
        #[cfg(feature = "metal")]
        {
            self.kv_idle_since = None;
        }

        // ── 4. Prepare flat batch inputs ─────────────────────────
        // Phase 4.6: forward the scheduler's spec drafts into
        // `prepare_inputs` so verify batches get the
        // `[last_token, draft_0..K-1]` flat shape per req. For non-spec
        // batches `scheduled_spec_decode_tokens` is empty → existing
        // q_len=1 path is unchanged.
        let mut prepared = self
            .input_batch
            .prepare_inputs(&scheduler_output.scheduled_spec_decode_tokens);
        let attn = &prepared.attn_meta;
        let num_tokens = attn.total_tokens;
        let num_reqs = attn.num_reqs;
        let block_size = self.config.block_size;
        // The FULL KV-cache group (group 0) uses the page-unified block size
        // (gemma4: 32) so its slot_mapping matches the kernel's baked
        // GLOBAL_BLOCK_SIZE; equals `block_size` on uniform models. The sliding
        // groups below keep the base `block_size`.
        let full_block_size = if self.kv_full_block_size > 0 {
            self.kv_full_block_size
        } else {
            block_size
        };

        // Reactive KV (2b): grow the chunked pool to back every block
        // this step touches (max over all per-seq block tables) before
        // the forward dereferences the chunk-address table bindlessly.
        // On metal-chunked pools this allocates chunks on demand; no-op
        // on cuda / single-buffer. Compute the max into a local so no
        // borrow of `attn` is held across the `&mut self` grow call.
        #[cfg(feature = "metal")]
        {
            // Cover BOTH groups' block IDs — the sliding ring draws from the
            // same shared pool, so a sliding ID can sit in a chunk the full
            // group never reached; an ungrown chunk has a stale address-table
            // entry and would fault on the sliding-layer write.
            let max_block = attn
                .block_ids
                .iter()
                .chain(attn.sliding_groups.iter().flatten())
                .flat_map(|b| b.iter())
                .copied()
                .max()
                .unwrap_or(0);
            self.grow_metal_kv_to_cover(max_block);
        }

        // slot_mapping[t] = block_ids[abs_pos / bs] * bs + (abs_pos % bs)
        // u32 under metal — the macro-emitted forward reads this as
        // `&[u32]`. Rope kernel checks `slot_mapping[i] != u32::MAX`.
        let mut slot_mapping_u32: Vec<u32> = Vec::with_capacity(num_tokens);
        // Spans (rope-on-read): only set bit 31 when the loaded model's
        // kernels mask it (W::ROPE_ON_READ). False for non-spans models →
        // `ann` stays None → byte-identical slot_mapping.
        let span_ror = self.rope_on_read_active();
        for i in 0..num_reqs {
            let tokens_before = attn.tokens_before[i];
            let q_len = attn.q_lens[i];
            let block_ids = &attn.block_ids[i];
            let ann = if span_ror {
                self.annotation_buffers.get(&attn.req_ids[i])
            } else {
                None
            };
            // Phase 2: reused cache-hit blocks whose KV must NOT be rewritten.
            let reused = if span_ror {
                self.reused_block_buffers.get(&attn.req_ids[i])
            } else {
                None
            };
            for t in 0..q_len {
                let abs_pos = tokens_before + t;
                let block_idx = abs_pos / full_block_size;
                let offset = abs_pos % full_block_size;
                if block_idx >= block_ids.len() {
                    slot_mapping_u32.push(u32::MAX);
                    continue;
                }
                // Phase 2: a reused cache-hit block already holds valid
                // (position-independent) K from the request that computed it —
                // skip rewriting its KV (u32::MAX = the kernels' write-skip
                // sentinel). Attention still READS it via block_table (bit 31).
                if reused.is_some_and(|r| r.contains(&block_idx)) {
                    slot_mapping_u32.push(u32::MAX);
                    continue;
                }
                let mut slot = (block_ids[block_idx] * full_block_size + offset) as u32;
                // ⚠️ SPANS BIT-31 CONTRACT (authoritative definition).
                // Bit 31 of a slot_mapping / block_table entry = "this block is
                // stored UNROTATED" (rope_append skips K-rotation; attention
                // re-ropes on read). EVERY Metal kernel that consumes these
                // tables AS AN INDEX must strip bit 31 (`& 0x7FFFFFFFu`, like
                // attention's ATTN_BT_MASK) BEFORE dividing/indexing — an
                // unmasked read indexes ~2^31 elements OOB and silently corrupts
                // the KV cache, but ONLY when spans are active, so it slips
                // normal testing (this exact bug hit the TurboQuant + fused-QKV
                // V kernels). Consumers: attention.metal, rope.metal,
                // fused_qkv_rope_cache, fused_affine_qkv_rope_cache,
                // turboquant.metal. Enforced by
                // crates/targets/metal/tests/kv_index_bit31_mask_test.rs.
                if ann.is_some_and(|a| {
                    a.get(&block_idx)
                        .is_some_and(scratchy_core_common::BlockKind::is_relocatable)
                }) {
                    slot |= 0x8000_0000;
                }
                slot_mapping_u32.push(slot);
            }
        }

        // block_table padded to [num_reqs, max_blocks_per_seq] u32.
        //
        // Row stride MUST match the scratchy-target-metal kernel's
        // `ATTN_PAGED_MAX_BLOCKS_PER_SEQ` function constant (slot 5,
        // baked from `W::MAX_BLOCKS_PER_SEQ` — default 128 in
        // `instr.rs::CanonicalParams`). The kernel reads
        // `block_table + seq_idx * MAX_BLOCKS_PER_SEQ`; if the host
        // writes with a smaller `runtime_max_blocks` stride, every
        // `seq_idx > 0` reads from the wrong row offset and pulls
        // garbage K/V. Symptom: concurrent / batched-decode
        // requests other than seq_idx=0 produce incoherent output;
        // single-seq runs are unaffected because only row 0 is read.
        //
        // The stride is the RUNTIME per-sequence block capacity the KV pool was
        // built with (`KvCachePool::max_blocks_per_seq` =
        // `min(ceil(max_model_len/block_size), num_blocks).max(1)`), NOT the
        // compile-time `W::MAX_BLOCKS_PER_SEQ` (default 128 ≈ 2k tokens, which
        // silently truncated long context). Reading the pool field directly
        // GUARANTEES this equals the kernel function constant: the macro bakes
        // slot 5 from `ctx.kv_cache.max_blocks_per_seq`, which is this same pool,
        // so host stride == kernel stride by construction. Fall back to the
        // formula only if the pool isn't up yet (never, at execute time).
        let kernel_block_table_stride: usize = self
            .kv_cache
            .as_ref()
            .map(|p| p.max_blocks_per_seq)
            .unwrap_or_else(|| self.kv_block_cap(usize::MAX));
        let runtime_max_blocks = attn.block_ids.iter().map(|b| b.len()).max().unwrap_or(0);
        // `.max(runtime_max_blocks)` is a defensive floor only: the scheduler
        // caps admission at max_model_len so a sequence never holds more than
        // `kernel_block_table_stride` blocks — but if it ever did, growing the
        // stride keeps the host write self-consistent (the kernel would still
        // read its baked stride; this never under-runs the host buffer).
        let max_blocks_eff = kernel_block_table_stride.max(runtime_max_blocks).max(1);
        // ⚠️ MEASURED 2026-08-11, SO THE COMMENT ABOVE IS NOT THE WHOLE STORY: forcing this stride to the
        // kernel's own baked value (128) and to 2048 both left the 12-request repro at 1/12 — the same
        // score as the pool value (8192). The host/kernel stride disagreement is REAL (slot 5 is baked from
        // `W::MAX_BLOCKS_PER_SEQ` at pipelines.rs:489, not from the pool as the comment above claims) but it
        // is NOT what corrupts a prefill that shares a step with a decode. Do not re-chase it from the
        // comment alone.
        let mut block_table_u32: Vec<u32> = vec![0u32; num_reqs * max_blocks_eff];
        if runtime_max_blocks > 0 {
            for (i, blocks) in attn.block_ids.iter().enumerate() {
                // Spans: relocatable logical blocks get bit 31 so attention
                // re-ropes them on read (the kernel masks it for addressing).
                let ann = if span_ror {
                    self.annotation_buffers.get(&attn.req_ids[i])
                } else {
                    None
                };
                for (j, &bid) in blocks.iter().enumerate() {
                    let mut entry = bid as u32;
                    if ann.is_some_and(|a| {
                        a.get(&j)
                            .is_some_and(scratchy_core_common::BlockKind::is_relocatable)
                    }) {
                        entry |= 0x8000_0000;
                    }
                    block_table_u32[i * max_blocks_eff + j] = entry;
                }
            }
        }

        // Per-block span labels for block-diagonal span attention: label each
        // block by its Relocatable span (the span's first block + 1; 0 = the
        // shared prefix / query). Zero-padded to the block_table stride so the
        // kernel indexes it identically to block_table. Only populated when the
        // request carries spans; otherwise the kernel's span bound is inert
        // (full causal, byte-identical).
        let span_ids_u32: Option<Vec<u32>> = if span_ror && num_reqs == 1 {
            self.annotation_buffers
                .get(&attn.req_ids[0])
                .and_then(|ann| {
                    // Per-TOKEN labels: the kernel indexes span_ids by RAW
                    // token position, so a span may begin/end mid-block — no
                    // alignment. Length = the GLOBAL (group-0) token capacity
                    // (num_global_blocks * full_block_size). `block_ids[0]` is
                    // the global block table, whose blocks are GLOBAL_BLOCK_SIZE
                    // tokens — 32 on gemma-4, 16 on uniform. Using the base
                    // `block_size` (16) HALVED the label vector on gemma, so
                    // upper-half query positions read the zero tail → span
                    // isolation silently lost. The kernel reads only up to
                    // kv_len, so the padded tail stays untouched.
                    let num_tokens = attn.block_ids[0].len() * full_block_size;
                    let labels = scratchy_core_common::request::span_ids_per_token(ann, num_tokens);
                    labels.iter().any(|&l| l != 0).then_some(labels)
                })
        } else {
            None
        };

        // Sliding KV-cache GROUPS (gemma4 SWA): per sliding group, build the
        // per-token slot_mapping + per-logical-block block table from
        // `attn.sliding_groups[s]`, encoded with the SLIDING block size
        // (`block_size`; the full group above uses the page-unified size).
        // Same `max_blocks_eff` stride. Empty on non-SWA models — a sliding
        // layer then resolves to group 0 (the full table) via `layer_to_group`,
        // exactly the pre-SWA behavior (no mirror buffer needed).
        let mut sliding_slot_mappings_u32: Vec<Vec<u32>> =
            Vec::with_capacity(attn.sliding_groups.len());
        let mut sliding_block_tables_u32: Vec<Vec<u32>> =
            Vec::with_capacity(attn.sliding_groups.len());
        for sliding in &attn.sliding_groups {
            let mut sm: Vec<u32> = Vec::with_capacity(num_tokens);
            // `i` indexes tokens_before / q_lens / sliding in lockstep.
            #[allow(clippy::needless_range_loop)]
            for i in 0..num_reqs {
                let tokens_before = attn.tokens_before[i];
                let q_len = attn.q_lens[i];
                let block_ids = &sliding[i];
                for t in 0..q_len {
                    let abs_pos = tokens_before + t;
                    let block_idx = abs_pos / block_size;
                    let offset = abs_pos % block_size;
                    if block_idx < block_ids.len() {
                        sm.push((block_ids[block_idx] * block_size + offset) as u32);
                    } else {
                        sm.push(u32::MAX);
                    }
                }
            }
            let mut bt: Vec<u32> = vec![0u32; num_reqs * max_blocks_eff];
            for (i, blocks) in sliding.iter().enumerate() {
                // Clamp to the kernel's baked MAX_BLOCKS_PER_SEQ row stride: a
                // request whose sliding block count exceeds it is past the
                // addressable context (gemma4: 2048×16 = 32768 tokens). Clamp
                // rather than panic; the scheduler caps admission at
                // max_model_len so this only guards the boundary.
                for (j, &bid) in blocks.iter().take(max_blocks_eff).enumerate() {
                    bt[i * max_blocks_eff + j] = bid as u32;
                }
            }
            sliding_slot_mappings_u32.push(sm);
            sliding_block_tables_u32.push(bt);
        }
        // Borrowed views for the per-group ForwardArgmaxRequest fields.
        let sliding_slot_mapping_refs: Vec<&[u32]> = sliding_slot_mappings_u32
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let sliding_block_table_refs: Vec<&[u32]> = sliding_block_tables_u32
            .iter()
            .map(|v| v.as_slice())
            .collect();

        let cu_seqlens_u32: Vec<u32> = attn.query_start_loc.iter().map(|&v| v as u32).collect();
        let seqused_k_u32: Vec<u32> = attn.seq_lens.iter().map(|&v| v as u32).collect();

        let max_seqlen_q = attn.q_lens.iter().copied().max().unwrap_or(0);
        let max_seqlen_k = attn.seq_lens.iter().copied().max().unwrap_or(0);
        let sample_indices: Vec<u32> = attn.sample_indices();
        let req_ids_in_order: Vec<String> = attn.req_ids.clone();
        let q_lens: Vec<usize> = attn.q_lens.clone();

        // ── Constrained / guided decoding: gather this step's grammar
        // allow-masks ───────────────────────────────────────────────────
        // For every request with an active grammar FSM that will sample a
        // real token this step, build a dense vocab bitset of its allowed
        // tokens and record the logits row to mask (`sample_indices[i]`).
        // The mask is dispatched before argmax in `forward_argmax_blocking`
        // (it `.take()`s `grammar_pending`), and the FSM is advanced with the
        // sampled token after readback. Skipped (no overhead) when no request
        // is constrained. Speculative-verify rows are excluded: grammar needs
        // a host `advance()` between every token, which the K-row verify can't
        // provide, so a grammar request must not also carry spec drafts.
        #[cfg(feature = "guided-decoding")]
        {
            self.grammar_pending = None;
            if !self.grammar_states.is_empty() {
                let vocab = self.model.as_deref().map(|m| m.vocab_size()).unwrap_or(0) as u32;
                if vocab > 0 {
                    let wpr = scratchy_target_metal::grammar_mask::words_per_row(vocab);
                    let mut rows: Vec<u32> = Vec::new();
                    let mut allow_bits: Vec<u32> = Vec::new();
                    for (i, req_id) in req_ids_in_order.iter().enumerate() {
                        // Still-prefilling chunk → no token sampled this step.
                        let prompt_len = self.prompt_lengths.get(req_id).copied().unwrap_or(0);
                        let seq_len_after = attn.seq_lens.get(i).copied().unwrap_or(usize::MAX);
                        if seq_len_after < prompt_len {
                            continue;
                        }
                        // Speculative verify batch → unsupported with grammar.
                        if !prepared.req_inputs[i].spec_token_ids.is_empty() {
                            continue;
                        }
                        let Some(guide) = self.grammar_states.get_mut(req_id) else {
                            continue;
                        };
                        // `None` => grammar terminated; leave the row unmasked.
                        let Some(allowed) = guide.allowed_tokens() else {
                            continue;
                        };
                        rows.push(sample_indices[i]);
                        scratchy_target_metal::grammar_mask::push_allow_bitset_row(
                            &mut allow_bits,
                            &allowed,
                            vocab,
                        );
                    }
                    if !rows.is_empty() {
                        self.grammar_pending = Some(GrammarMaskHost {
                            rows,
                            allow_bits,
                            vocab,
                            words_per_row: wpr,
                        });
                    }
                }
            }
        }

        // GDN per-step state-slot indices (hybrid arches only). One i32
        // slot id + u32 fresh flag per batched sequence, in the SAME
        // order as `cu_seqlens_q` / `req_ids_in_order`. The metal
        // `forward_argmax_blocking` uploads these into
        // `ForwardCtx::{gdn_state_indices, gdn_is_fresh}`. `slot_for`
        // returns `is_fresh=true` on a request's FIRST forward (including
        // a recycled slot's new owner) so the GDN conv1d/scan kernels
        // zero-init the slot's conv/ssm state instead of continuing from
        // a finished sequence's stale data (the degeneration guard).
        self.gdn_pending = if let Some(alloc) = self.gdn_slot_allocator.as_mut() {
            let mut indices = Vec::with_capacity(req_ids_in_order.len());
            let mut fresh = Vec::with_capacity(req_ids_in_order.len());
            for req_id in &req_ids_in_order {
                match alloc.slot_for(gdn_slot_key(req_id)) {
                    Some((slot, is_fresh)) => {
                        indices.push(slot as i32);
                        fresh.push(u32::from(is_fresh));
                    }
                    None => {
                        return Err(ExecutorError::WorkerExecution(format!(
                            "GDN state-slot pool exhausted (capacity {}): scheduler \
                             admitted more concurrent sequences than max_num_seqs",
                            alloc.capacity(),
                        )));
                    }
                }
            }
            Some((indices, fresh))
        } else {
            None
        };

        // ── 5b. Vision encoder for any MM-bearing req at its first
        // prefill step. Output [n_img_tokens, hidden] + per-image patch
        // placeholders are parked in `self.mm_pending`; the upcoming
        // TARGET `forward_argmax_blocking` consumes them into
        // `ForwardCtx::{mm_embeds, embed_patches}` so the baked
        // `Instruction::Embed` splices visual tokens into the residual.
        // Borrows are disjoint fields of `self` (NLL): `mm` (shared),
        // `mm_data_buffers` (shared), `gpu_device` (mut).
        #[cfg(feature = "vision")]
        {
            self.mm_pending = match (self.mm.as_ref(), self.gpu_device.as_mut()) {
                (Some(mm), Some(dev)) => unsafe {
                    Self::run_mm_vision_forward_metal(
                        mm.as_ref(),
                        &self.mm_data_buffers,
                        &prepared,
                        dev,
                    )
                },
                _ => None,
            };
        }
        #[cfg(not(feature = "vision"))]
        {
            self.mm_pending = None;
        }

        // MRoPE (Qwen3.5-VL): when the loaded multimodal arch consumes 3D
        // positions (`mm_metadata().mrope_positions`) AND this batch
        // carries image tokens, build the `[3, n]` (T,H,W) band-split
        // positions. Built BEFORE the `flat_positions` take (the builder
        // reads `prepared.flat_positions.len()`). The metal text decoder
        // consumes these through the per-token cos/sin override
        // (`build_mrope_cos_sin_override` in the macro forward) with
        // identity positions, so image tokens aggregate over the 2D grid
        // instead of collapsing onto a 1D ramp. Text-only / decode steps
        // (no image tokens → `per_req_mm` empty) keep the 1D positions.
        #[cfg(feature = "vision")]
        let mrope_positions_2d: Option<Vec<u32>> = {
            let mrope = self
                .mm
                .as_ref()
                .map(|m| m.mm_metadata().mrope_positions)
                .unwrap_or(false);
            if mrope {
                if let Some(mm) = self.mm.as_ref() {
                    let per_req_mm = Self::build_per_req_mm_seq_info_metal(
                        mm.as_ref(),
                        &self.mm_data_buffers,
                        &prepared,
                    );
                    if per_req_mm.is_empty() {
                        None
                    } else {
                        Some(Self::build_mrope_positions_2d(&prepared, &per_req_mm))
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };
        #[cfg(not(feature = "vision"))]
        let mrope_positions_2d: Option<Vec<u32>> = None;
        let input_ids_u32 = std::mem::take(&mut prepared.flat_token_ids);
        // `[3, n]` MRoPE override (image batches) or the 1D per-token
        // positions (everything else). `forward_argmax_blocking` sizes the
        // `positions` TensorView from this slice's len (3n vs n), which the
        // macro forward disambiguates to drive the cos/sin override.
        let positions_u32 = match mrope_positions_2d {
            Some(p) => p,
            None => std::mem::take(&mut prepared.flat_positions),
        };

        // ── 6. Forward + per-row argmax via SpecDecodeBackend trait ─
        //
        // Phase 5.2a routes the verify forward through
        // `forward_argmax_blocking`. This loses the fused-CB argmax
        // optimization (~0.5 ms / 5% per step) — phase 6 re-introduces
        // it via a dedicated trait primitive. The K-step draft chain
        // below stays inline until phase 5.4 moves it host-side into
        // `DraftModelProposer::propose`.
        // Spec verify? Any req carrying spec_token_ids flips the gate
        // so the lm_head slice trio (OnlyIfSingleSeqNoSpec) skips and
        // the full GEMM (OnlyIfMultiSeqOrSpec) writes every row of
        // logits — required for rejection sampling on > 1 sample
        // positions per seq.
        let has_spec_tokens = prepared
            .req_inputs
            .iter()
            .any(|r| !r.spec_token_ids.is_empty());

        // Phase 8: when a draft model is loaded AND we're in lockstep+K
        // mode (extended-batch disabled), kick off the lockstep prefill
        // on the dedicated `draft_queue` in a scoped thread, in
        // parallel with target verify on the main queue. Target writes
        // target-KV, lockstep writes draft-KV — separate KV pools, no
        // contention. Saves ~15-20 ms / step on the draft chain
        // critical path.
        let phase8_parallel_lockstep = self.draft_model.is_some()
            && self.draft_kv_cache.is_some()
            && self.draft_queue.is_some();
        let mut async_lockstep_done = false;
        // Phase-9 results from the lockstep thread (populated when
        // enabled + eligible; empty otherwise).
        let mut spec9_seeds_final: Vec<u32> = Vec::new();
        let mut spec9_drafts_final: Vec<Vec<u32>> = Vec::new();
        // ── Fused on-GPU sampler prep (peak path) ────────────────────────
        // Compute this step's non-greedy sampling rows BEFORE the forward —
        // every input (sample position, still-prefilling gate, spec check,
        // greedy check) is known pre-forward — and prepare + pin the sampler so
        // it can be encoded onto the forward's OWN command buffer (in
        // `forward_argmax_blocking`'s followup, after argmax) instead of a second
        // command buffer + host wait. Greedy-only / spec / still-prefilling steps
        // prepare nothing → the argmax fast path stands unchanged.
        self.fused_sampled = None;
        self.pending_sampler = None;
        let mut fused_sample_jobs: Vec<(usize, u32)> = Vec::new();
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let req_slice = &prepared.req_inputs[i];
            if !req_slice.spec_token_ids.is_empty() {
                continue; // spec-verify rows use greedy rejection, not the sampler
            }
            let prompt_len = self.prompt_lengths.get(req_id).copied().unwrap_or(0);
            let seq_len_after = attn.seq_lens.get(i).copied().unwrap_or(usize::MAX);
            if seq_len_after < prompt_len {
                continue; // still-prefilling chunk emits no token
            }
            let greedy = self
                .sampling_params_map
                .get(req_id)
                .is_none_or(SamplingParams::is_greedy);
            if !greedy {
                fused_sample_jobs.push((i, sample_indices[i]));
            }
        }
        if !fused_sample_jobs.is_empty()
            && self.sampler_kernels.is_some()
            && self.model.as_deref().map(|m| m.metal_dtype())
                != Some(scratchy_target_metal::interpreter::metal::MetalDtype::Int4)
        {
            match self.prepare_gpu_sampler(&fused_sample_jobs, &req_ids_in_order) {
                Ok(ps) => self.pending_sampler = Some(ps),
                Err(e) => tracing::warn!("fused sampler prep failed ({e}); using argmax fallback"),
            }
        }
        let argmax_vec: Vec<u32> = {
            use ::scratchy_serving_engine::spec_decode::{
                ForwardArgmaxRequest, KvPoolHandle, ModelHandle, SpecDecodeBackend,
            };
            let req = ForwardArgmaxRequest {
                input_ids: &input_ids_u32,
                positions: &positions_u32,
                slot_mapping: &slot_mapping_u32,
                cu_seqlens_q: &cu_seqlens_u32,
                seqused_k: &seqused_k_u32,
                span_ids: span_ids_u32.as_deref(),
                block_table: &block_table_u32,
                sliding_slot_mappings: &sliding_slot_mapping_refs,
                sliding_block_tables: &sliding_block_table_refs,
                block_table_stride: max_blocks_eff,
                max_seqlen_q,
                max_seqlen_k,
                num_tokens,
                has_spec_tokens,
                last_token_indices: if sample_indices.is_empty() {
                    None
                } else {
                    Some(&sample_indices)
                },
            };
            // Phase 9: the worker-side speculative K-step chain (runs
            // in the lockstep thread, overlapped with target verify).
            // Off until baseline bench validates the projected ~13%
            // TPOT win.
            let phase9_speculative_chain = false;

            // Spec-9 eligibility, single-req case only for now.
            // Derive K from q_lens[0] (= drafts + 1 bonus when the
            // req was spec-decoded last step; = 1 otherwise). Skip
            // first-step / multi-req / chunked-prefill cases — they
            // fall through to the regular chain in the proposer.
            let spec9_single_req_eligible: bool = phase9_speculative_chain
                && num_reqs == 1
                && prepared
                    .req_inputs
                    .first()
                    .is_some_and(|r| !r.spec_token_ids.is_empty())
                && q_lens.first().copied().unwrap_or(0) >= 2;
            let spec9_k: usize = if spec9_single_req_eligible {
                q_lens[0] - 1
            } else {
                0
            };
            let spec9_tokens_before: usize = if spec9_single_req_eligible {
                attn.tokens_before[0]
            } else {
                0
            };
            let spec9_block_ids: Vec<u32> = if spec9_single_req_eligible {
                attn.block_ids[0].iter().map(|&b| b as u32).collect()
            } else {
                Vec::new()
            };
            let spec9_block_size: usize = self.config.block_size;

            if phase8_parallel_lockstep {
                // Snapshot of inputs the lockstep thread needs.
                // Raw-pointer borrow of draft_model / draft_kv_cache is
                // sound here: both are accessed ONLY by the lockstep
                // thread (target verify uses self.model + self.kv_cache,
                // disjoint fields) and the thread can't outlive the
                // scope (scoped threads are joined before scope exit).
                // Decompose the fat pointer to `dyn ScratchyWeights`
                // into two `usize` so the closure stays `Send`.
                // `*const dyn T` is `[data, vtable]` on the supported
                // targets; transmute to extract both.
                let dm_pair: [usize; 2] = {
                    let fat: *const dyn scratchy_forward_compiler::ScratchyWeights =
                        self.draft_model.as_deref().expect("checked above");
                    unsafe {
                        std::mem::transmute::<
                            *const dyn scratchy_forward_compiler::ScratchyWeights,
                            [usize; 2],
                        >(fat)
                    }
                };
                let dkv_addr: usize = (self.draft_kv_cache.as_ref().expect("checked above")
                    as *const KvCachePool) as usize;
                // Spec-9: raw pointers for argmax + chain_advance
                // kernels so the speculative chain can call
                // `metal_chain_dispatch` from inside the thread
                // (no `&self` access while target verify is in flight).
                let spec9_argmax_addr: usize = self.argmax_kernels.as_ref().map_or(0, |k| {
                    k as *const scratchy_target_metal::argmax::ArgmaxKernels as usize
                });
                let spec9_chain_addr: usize = self.chain_advance_kernel.as_ref().map_or(0, |k| {
                    k as *const scratchy_target_metal::chain_advance::ChainAdvanceKernel as usize
                });
                let main_dev = self.gpu_device.as_ref().expect("init");
                let mtl_device_clone = main_dev.device.clone();
                let allocator_clone = main_dev.allocator.clone();
                let draft_queue_clone = self.draft_queue.as_ref().expect("checked above").clone();
                // Snapshot host slices the lockstep thread reads.
                let lock_input_ids = &input_ids_u32;
                let lock_positions = &positions_u32;
                let lock_slot_mapping = &slot_mapping_u32;
                let lock_cu_seqlens = &cu_seqlens_u32;
                let lock_seqused_k = &seqused_k_u32;
                let lock_block_table = &block_table_u32;
                let lock_block_table_stride = max_blocks_eff;
                let lock_max_q = max_seqlen_q;
                let lock_max_k = max_seqlen_k;
                let lock_num_tokens = num_tokens;
                let lock_num_reqs = num_reqs;

                // Stage-1 spec9 hypothesis probe: capture draft's
                // last-row-per-req argmax inside the scoped lockstep
                // thread so we can compare to target's actual bonus
                // after the join. Pure measurement — no behavioral
                // change yet.
                let lock_cu_for_thread = lock_cu_seqlens.to_vec();
                // Spec-9 captures (moved into thread closure).
                let spec9_eligible_thread = spec9_single_req_eligible;
                let spec9_k_thread = spec9_k;
                let spec9_tokens_before_thread = spec9_tokens_before;
                let spec9_block_ids_thread = spec9_block_ids.clone();
                let spec9_block_size_thread = spec9_block_size;
                let spec9_block_table_stride_thread = lock_block_table_stride;
                let (v, spec9_seeds_out, spec9_drafts_out): (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) =
                    std::thread::scope(|s| {
                        let lockstep_handle =
                        s.spawn(move || -> (Vec<u32>, Vec<Vec<u32>>) {
                        // Build shadow GpuDevice on draft_queue.
                        let mut shadow_device = GpuDevice {
                            device: mtl_device_clone,
                            queue: draft_queue_clone,
                            allocator: allocator_clone,
                            // Draft chain shadow device: draft is a small
                            // non-GDN model; keep all its buckets (no prune).
                            metal_bucket_max_m: None,
                        };
                        // Upload host slices into fresh shared
                        // MTLBuffers (thread-local, dropped at thread
                        // exit).
                        let dev = &shadow_device.device;
                        let b_in = Self::alloc_shared_u32_buf(dev, lock_input_ids);
                        let b_pos = Self::alloc_shared_u32_buf(dev, lock_positions);
                        let b_slot = Self::alloc_shared_u32_buf(dev, lock_slot_mapping);
                        let b_cu = Self::alloc_shared_u32_buf(dev, lock_cu_seqlens);
                        let b_su = Self::alloc_shared_u32_buf(dev, lock_seqused_k);
                        let b_bt = Self::alloc_shared_u32_buf(dev, lock_block_table);

                        let dt = scratchy_target_metal::dtype::DType::U32;
                        let v_in = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_in.contents().as_ptr() as *mut u8,
                            &[lock_num_tokens.max(1)], dt)) };
                        let v_pos = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_pos.contents().as_ptr() as *mut u8,
                            &[lock_num_tokens.max(1)], dt)) };
                        let v_slot = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_slot.contents().as_ptr() as *mut u8,
                            &[lock_num_tokens.max(1)], dt)) };
                        let v_cu = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_cu.contents().as_ptr() as *mut u8,
                            &[lock_cu_seqlens.len().max(1)], dt)) };
                        let v_su = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_su.contents().as_ptr() as *mut u8,
                            &[lock_seqused_k.len().max(1)], dt)) };
                        let v_bt = unsafe { TensorView::from_raw(GpuTensor::new(
                            b_bt.contents().as_ptr() as *mut u8,
                            &[lock_num_reqs.max(1), lock_block_table_stride.max(1)], dt)) };

                        // Reassemble fat pointer from (data, vtable)
                        // pair, then immutable-borrow.
                        let dm_fat_reassembled: *const dyn scratchy_forward_compiler::ScratchyWeights =
                            unsafe {
                                std::mem::transmute::<
                                    [usize; 2],
                                    *const dyn scratchy_forward_compiler::ScratchyWeights,
                                >(dm_pair)
                            };
                        let dm = unsafe { &*dm_fat_reassembled };
                        let dkv: &KvCachePool =
                            unsafe { &*(dkv_addr as *const KvCachePool) };
                        let ctx = scratchy_target_metal::ForwardCtx {
                            input_ids: v_in,
                            positions: v_pos,
                            slot_mapping: v_slot,
                            cu_seqlens_q: v_cu,
                            seqused_k: v_su,
                            span_ids: None,
                            block_table: v_bt,
                            // Spec-9 chain is single-group (no sliding groups).
                            sliding_slot_mappings: Vec::new(),
                            sliding_block_tables: Vec::new(),
                            max_seqlen_q: lock_max_q,
                            max_seqlen_k: lock_max_k,
                            kv_cache: dkv,
                            // Text-only forward — no vision splice / encoder inputs.
                            mm_embeds: None,
                            embed_patches: &[],
                            vision_rope_cos: None,
                            vision_rope_sin: None,
                            vision_rope_freqs: None,
                            pixels: None,
                            pos_embeds: None,
                            vision_cu_seqlens_full: None,
                            vision_cu_seqlens_window: None,
                            vision_max_seqlen_full: None,
                            vision_max_seqlen_window: None,
                            vision_window_index: None,
                            vision_reverse_indices: None,
                            vision_position_ids: None,
                            gdn_state: None,
                            gdn_state_indices: None,
                            gdn_is_fresh: None,
                            has_spec_tokens: false,
        kv_turboquant: false, // secondary path; tq provisioned via the main forward
                            last_token_indices: None,
                        };
                        let logits = unsafe {
                            dm.forward(
                                scratchy_target_metal::ForwardCtxHandle::new(&ctx),
                                scratchy_target_metal::ForwardDeviceHandle::new(&mut shadow_device),
                                lock_num_tokens as u64,
                            )
                        };

                        // Phase-9: argmax of the last verify position
                        // per req IS draft's prediction of what target
                        // will sample as the step's bonus token.
                        // Computed when the speculative chain is
                        // enabled. Bf16 only (matches `metal_dtype()`
                        // for our test models).
                        let need_spec_seed = spec9_eligible_thread;
                        let spec_seeds: Vec<u32> = if need_spec_seed {
                            let logits_ptr =
                                logits.as_gpu_tensor().raw_ptr() as *const u8;
                            let vocab = dm.vocab_size() as usize;
                            let mut out = Vec::with_capacity(lock_num_reqs);
                            for i in 0..lock_num_reqs {
                                let last_row =
                                    lock_cu_for_thread[i + 1] as usize - 1;
                                let row_off = last_row * vocab * 2;
                                let row = unsafe {
                                    std::slice::from_raw_parts(
                                        logits_ptr.add(row_off) as *const u16,
                                        vocab,
                                    )
                                };
                                let mut best_idx: u32 = 0;
                                let mut best_val: f32 = f32::NEG_INFINITY;
                                for (j, &bits) in row.iter().enumerate() {
                                    let v = f32::from_bits((bits as u32) << 16);
                                    if v > best_val {
                                        best_val = v;
                                        best_idx = j as u32;
                                    }
                                }
                                out.push(best_idx);
                            }
                            out
                        } else {
                            Vec::new()
                        };

                        // Phase-9 speculative chain (single-req case):
                        // dispatch the K-step chain on draft_queue using
                        // spec_seeds[0] as the iter-0 input_ids. Runs
                        // serially after lockstep on draft_queue, but
                        // in parallel with target verify on main_queue.
                        // If target's sampled bonus matches spec_seed
                        // AND target accepted all K drafts, the proposer
                        // returns these drafts directly and the chain
                        // is hidden behind target verify. Otherwise,
                        // proposer falls back to running the chain
                        // with the corrected seed (re-overwrites
                        // draft KV at the now-correct positions —
                        // see `project-spec-decode-phase6-handoff`).
                        let spec_drafts: Vec<Vec<u32>> = if spec9_eligible_thread
                            && !spec_seeds.is_empty()
                        {
                            let argmax_kernels_ref: &scratchy_target_metal::argmax::ArgmaxKernels = unsafe {
                                &*(spec9_argmax_addr
                                    as *const scratchy_target_metal::argmax::ArgmaxKernels)
                            };
                            let chain_kernel_ref: &scratchy_target_metal::chain_advance::ChainAdvanceKernel = unsafe {
                                &*(spec9_chain_addr
                                    as *const scratchy_target_metal::chain_advance::ChainAdvanceKernel)
                            };
                            // Single-req inputs assuming "all K drafts
                            // accepted + bonus": position after bonus =
                            // tokens_before + (q_lens-1) + 1 + 1 ...
                            // Actually simpler — the K-step chain's
                            // iter-0 position is `tokens_before + q_lens`
                            // (= the slot after the bonus that target
                            // will sample). For was_spec_decode this
                            // ASSUMES all K drafts accepted (the
                            // "all-hit" speculative case the proposer
                            // validates).
                            let chain_pos = spec9_tokens_before_thread
                                + spec9_k_thread + 1;
                            let block_idx = chain_pos / spec9_block_size_thread;
                            let offset = chain_pos % spec9_block_size_thread;
                            let slot = if block_idx
                                < spec9_block_ids_thread.len()
                            {
                                spec9_block_ids_thread[block_idx]
                                    * spec9_block_size_thread as u32
                                    + offset as u32
                            } else {
                                u32::MAX
                            };
                            let spec_input_ids: Vec<u32> = vec![spec_seeds[0]];
                            let spec_positions: Vec<u32> = vec![chain_pos as u32];
                            let spec_slot: Vec<u32> = vec![slot];
                            let spec_cu: Vec<u32> = vec![0, 1];
                            let spec_su: Vec<u32> = vec![(chain_pos + 1) as u32];
                            let spec_req =
                                ::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest {
                                    input_ids: &spec_input_ids,
                                    positions: &spec_positions,
                                    slot_mapping: &spec_slot,
                                    cu_seqlens_q: &spec_cu,
                                    seqused_k: &spec_su,
                                    span_ids: None,
                                    block_table: lock_block_table,
                                    // Spec-9 chain is single-group (no SWA groups).
                                    sliding_slot_mappings: &[],
                                    sliding_block_tables: &[],
                                    block_table_stride:
                                        spec9_block_table_stride_thread,
                                    max_seqlen_q: 1,
                                    max_seqlen_k: chain_pos + 1,
                                    num_tokens: 1,
                                    has_spec_tokens: false,
                                    last_token_indices: None,
                                };
                            metal_chain_dispatch(
                                dm,
                                dkv,
                                &mut shadow_device,
                                argmax_kernels_ref,
                                chain_kernel_ref,
                                &spec_req,
                                spec9_block_size_thread,
                                spec9_k_thread,
                            )
                            .unwrap_or_else(|_e| Vec::new())
                        } else {
                            Vec::new()
                        };

                        (spec_seeds, spec_drafts)
                    });

                        let r = self
                            .forward_argmax_blocking(
                                ModelHandle::TARGET,
                                KvPoolHandle::TARGET,
                                &req,
                            )
                            .map_err(|e| {
                                ExecutorError::WorkerExecution(format!("spec verify forward: {e}"))
                            });
                        // join propagates any panic the lockstep thread
                        // raised; treat as worker execution failure.
                        let (spec_seeds, spec_drafts) =
                            lockstep_handle.join().expect("lockstep thread panicked");
                        let target_argmax = r.unwrap_or_else(|_| Vec::new());
                        if !spec_seeds.is_empty() && !target_argmax.is_empty() {
                            // Probe: compare draft's predicted bonus
                            // vs target's actual bonus (per req).
                            // cu_seqlens_q[i+1]-1 is the last verify row.
                            let cu = &lock_cu_seqlens;
                            for i in 0..lock_num_reqs {
                                let target_bonus_idx = cu[i + 1] as usize - 1;
                                if target_bonus_idx < target_argmax.len() {
                                    let target_bonus = target_argmax[target_bonus_idx];
                                    let spec_seed = spec_seeds[i];
                                    eprintln!(
                                        "[spec9-probe] req={} spec_seed={} \
                                     target_bonus={} match={}",
                                        i,
                                        spec_seed,
                                        target_bonus,
                                        spec_seed == target_bonus,
                                    );
                                }
                            }
                        }
                        (target_argmax, spec_seeds, spec_drafts)
                    });
                async_lockstep_done = true;
                spec9_seeds_final = spec9_seeds_out;
                spec9_drafts_final = spec9_drafts_out;
                v
            } else {
                self.forward_argmax_blocking(ModelHandle::TARGET, KvPoolHandle::TARGET, &req)
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("spec verify forward: {e}"))
                    })?
            }
        };
        let total_n = argmax_vec.len() as u32;
        let argmax_slice: &[u32] = &argmax_vec;

        // ── 6.5. Pooling (embedding) requests ───────────────────
        //
        // An encoder layout has no lm_head: the terminal arena slot the
        // forward just wrote IS `[num_tokens, hidden_size]` hidden
        // states (that's what `METAL_VOCAB_SIZE` carries on encoder
        // arches, and what `sampler_logits` captured). Slice out each
        // request's rows and pool them host-side — the vectors are
        // `hidden_size` wide, so the D2H is a few KB and the arithmetic
        // is not worth a kernel.
        //
        // The argmax the forward ran on the way here is meaningless for
        // these rows; the engine discards `sampled_token_ids` for a
        // pooling request and finishes it after this single pass.
        let pooler_output = if self.config.is_pooling {
            let (buf, width) = self.sampler_logits.as_ref().ok_or_else(|| {
                ExecutorError::WorkerExecution(
                    "pooling: forward captured no terminal buffer".into(),
                )
            })?;
            let hidden = *width as usize;
            let is_bf16 = matches!(
                self.model
                    .as_deref()
                    .ok_or_else(|| ExecutorError::WorkerExecution(
                        "pooling: model not loaded".into()
                    ))?
                    .metal_dtype(),
                scratchy_target_metal::interpreter::metal::MetalDtype::Bf16
            );
            // SAFETY: arena slots are StorageModeShared and the forward
            // is blocking — the GPU is done writing these `num_tokens *
            // hidden` elements before `contents()` is read.
            let raw: &[u16] = unsafe {
                std::slice::from_raw_parts(
                    buf.contents().as_ptr() as *const u16,
                    num_tokens * hidden,
                )
            };
            let mut map: std::collections::HashMap<String, EmbeddingData> =
                std::collections::HashMap::with_capacity(num_reqs);
            for (i, req_id) in req_ids_in_order.iter().enumerate() {
                let req_slice = &prepared.req_inputs[i];
                let rows = req_slice.token_count;
                if rows == 0 {
                    continue;
                }
                let start = req_slice.token_start * hidden;
                let f32_rows: Vec<f32> = raw[start..start + rows * hidden]
                    .iter()
                    .map(|&b| {
                        if is_bf16 {
                            half::bf16::from_bits(b).to_f32()
                        } else {
                            half::f16::from_bits(b).to_f32()
                        }
                    })
                    .collect();
                let mut pooled = scratchy_core_model::embedding::pool_rows(
                    &f32_rows,
                    rows,
                    hidden,
                    self.pooling_strategy,
                );
                let data = if self.pooling_strategy
                    == scratchy_core_model::embedding::PoolingStrategy::AllTokens
                {
                    EmbeddingData::Multi(pooled)
                } else {
                    EmbeddingData::Single(pooled.remove(0))
                };
                map.insert(req_id.clone(), data);
            }
            Some(map)
        } else {
            None
        };

        // ── 6.6. Per-request rejection sampling (phase 4.6) ──
        //
        // Build `sampled_token_ids` + `was_spec_decode` BEFORE the
        // draft phase so the K-step chain can seed from accepted
        // (= argmax under greedy) tokens instead of raw target argmax.
        // For non-spec batches (`ReqSlice.spec_token_ids` empty), grab
        // the single argmax at the req's last sample position. For
        // verify batches (K drafts), run `greedy_rejection_sample`
        // over the K+1 contiguous argmax rows.
        let mut sampled_token_ids: Vec<Vec<u32>> = Vec::with_capacity(num_reqs);
        let mut was_spec_decode: Vec<bool> = Vec::with_capacity(num_reqs);
        let mut req_id_to_index: std::collections::HashMap<String, usize> =
            std::collections::HashMap::with_capacity(num_reqs);
        // Non-greedy rows were computed + prepared (`fused_sample_jobs` /
        // `self.pending_sampler`) BEFORE the forward and sampled ON the forward
        // CB; their tokens land in `self.fused_sampled`, applied after this loop.
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let req_slice = &prepared.req_inputs[i];
            req_id_to_index.insert(req_id.clone(), i);
            // Chunked-prefill intermediate chunk: the prefill does NOT
            // complete this step (the request still has prompt tokens past
            // this chunk, i.e. `seq_len < prompt_len`). Such a chunk writes
            // its K/V into the cache but must NOT emit a sampled token —
            // only the FINAL chunk (where seq_len == prompt_len) or a decode
            // step produces output. This matches Python vLLM, which excludes
            // still-prefilling requests from the logits/sample set.
            //
            // Without this gate every non-final chunk emits one spurious
            // token: the greedy argmax at the chunk's last prompt position
            // (e.g. the model's continuation of the boilerplate). The engine
            // (`update_from_output`) appends whatever the worker returns, so
            // that token is prepended before the real first generated token
            // — the "doubled first token" / prompt-echo symptom of chunked
            // prefill. `commit_step` still advances `tokens_in_pool` by the
            // chunk's `q_len` (it keys on `input_token_count`, not the token
            // slice), so an empty emit leaves prefill bookkeeping correct and
            // the next step's re-arm continues the prefill.
            let prompt_len = self.prompt_lengths.get(req_id).copied().unwrap_or(0);
            let seq_len_after = attn.seq_lens.get(i).copied().unwrap_or(usize::MAX);
            if seq_len_after < prompt_len {
                sampled_token_ids.push(Vec::new());
                was_spec_decode.push(false);
                continue;
            }
            if req_slice.spec_token_ids.is_empty() {
                let row = sample_indices[i] as usize;
                debug_assert!(row < total_n as usize);
                // Push the GPU argmax as the token; for a non-greedy request it
                // is the fallback if the fused sampler was skipped. Non-greedy
                // rows get overwritten from `self.fused_sampled` after this loop.
                sampled_token_ids.push(vec![argmax_slice[row]]);
                was_spec_decode.push(false);
            } else {
                let start = req_slice.token_start;
                let end = start + req_slice.token_count;
                debug_assert!(end <= total_n as usize);
                let target_ids = &argmax_slice[start..end];
                let rejection = ::scratchy_serving_engine::spec_decode::greedy_rejection_sample(
                    target_ids,
                    &req_slice.spec_token_ids,
                );
                sampled_token_ids.push(rejection.accepted_tokens);
                was_spec_decode.push(true);
            }
        }

        // ── 6.62. Apply the fused on-GPU sampler's tokens ──
        // Non-greedy rows were sampled ON the forward's command buffer (prepared
        // pre-forward as `fused_sample_jobs` / `self.pending_sampler`, encoded in
        // the forward followup after argmax). Overwrite their argmax fallback
        // with the sampled token here — BEFORE the grammar advance below, so a
        // grammar-constrained sampling request advances its FSM on the sampled
        // (grammar-masked) token. Greedy-only batches produced nothing → argmax
        // fast path unchanged.
        if let Some(sampled) = self.fused_sampled.take() {
            for (&(slot, _), &tok) in fused_sample_jobs.iter().zip(sampled.iter()) {
                sampled_token_ids[slot] = vec![tok];
            }
        }
        // The captured logits slot is consumed; drop the retained clone so it
        // can't be mistaken for a later step's logits.
        self.sampler_logits = None;

        // ── 6.65. Constrained / guided decoding: advance grammar FSMs ──
        // Feed each grammar request its just-sampled (masked) token so the
        // next step's allow-set reflects it. Only the masked path (non-spec,
        // single emitted token) is advanced; speculative-verify rows are not
        // grammar-masked and are left untouched. `advance` returning false
        // means the sampled token wasn't in the grammar's allow-set — a
        // mask/FSM desync that should never happen, surfaced as a warning.
        #[cfg(feature = "guided-decoding")]
        if !self.grammar_states.is_empty() {
            for (i, req_id) in req_ids_in_order.iter().enumerate() {
                if was_spec_decode[i] || sampled_token_ids[i].len() != 1 {
                    continue;
                }
                if let Some(guide) = self.grammar_states.get_mut(req_id) {
                    let tok = sampled_token_ids[i][0];
                    if !guide.advance(tok) {
                        tracing::warn!(
                            "guided-decoding: req {req_id} sampled token {tok} rejected by \
                             grammar (mask/advance desync); decoding may diverge"
                        );
                    }
                }
            }
        }

        // ── 6.7. Draft seed bundle ──────────────────────────────
        //
        // Pre-5.4 the worker ran the lockstep prefill + K autoregressive
        // draft decode chain inline here against `self.draft_model` +
        // `self.draft_kv_cache`. As of phase 5.4 the chain lives in
        // `scratchy_serving_engine::spec_decode::DraftModelProposer::propose_for_step`
        // (host-side) and reaches the same GPU paths via the
        // `SpecDecodeBackend` trait. The worker just pre-packages the
        // owned-data the proposer needs into `draft_seed_inputs`; the
        // engine pulls `executor.spec_decode_backend()` and drives the
        // chain in `finalize_step`.
        let draft_seed_inputs: Option<::scratchy_serving_engine::spec_decode::DraftSeedInputs> =
            if self.draft_model.is_some() && self.draft_kv_cache.is_some() {
                Some(::scratchy_serving_engine::spec_decode::DraftSeedInputs {
                    input_ids: input_ids_u32,
                    positions: positions_u32,
                    slot_mapping: slot_mapping_u32,
                    cu_seqlens_q: cu_seqlens_u32,
                    seqused_k: seqused_k_u32,
                    block_table: block_table_u32,
                    block_table_stride: max_blocks_eff,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_tokens,
                    req_ids: req_ids_in_order.clone(),
                    block_ids: attn
                        .block_ids
                        .iter()
                        .map(|v| v.iter().map(|&b| b as u32).collect())
                        .collect(),
                    tokens_before: attn.tokens_before.clone(),
                    q_lens: q_lens.clone(),
                    was_spec_decode: was_spec_decode.clone(),
                    block_size,
                    // Phase 8: when target verify ran in parallel
                    // with lockstep prefill on draft_queue (above),
                    // the worker has already waited for both — the
                    // proposer skips its in-proposer lockstep call.
                    async_lockstep_done,
                    speculative_seeds: spec9_seeds_final.clone(),
                    speculative_chain_drafts: spec9_drafts_final.clone(),
                })
            } else {
                None
            };

        // ── 7. Commit per-request state ─────────────────────────
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let q_len = q_lens[i];
            self.input_batch
                .commit_step(req_id, &sampled_token_ids[i], q_len, was_spec_decode[i]);
            // Append this step's emitted tokens to the per-request token
            // history (`token_buffers` = prompt ++ generated). The repetition /
            // frequency / presence penalties in the on-GPU sampler read this
            // history; appending AFTER sampling means a step's penalty is based
            // on tokens generated strictly before it (standard vLLM behavior).
            // Prefill-continuation reads (`token_buffers[..prompt_len]`) are
            // unaffected — the appended tokens live past `prompt_len`.
            if !sampled_token_ids[i].is_empty()
                && let Some(buf) = self.token_buffers.get_mut(req_id)
            {
                buf.extend_from_slice(&sampled_token_ids[i]);
            }
        }

        Ok(ModelRunnerOutput {
            req_ids: req_ids_in_order,
            req_id_to_index,
            sampled_token_ids,
            logprobs: None,
            prompt_logprobs_dict: std::collections::HashMap::new(),
            // K-step draft chain runs host-side now; the proposer fills
            // `set_spec_token_ids` directly. `draft_token_ids` stays
            // `None` from the worker.
            draft_token_ids: None,
            draft_seed_inputs,
            pooler_output,
            // ⛔ EMPTY, AND THE EMPTINESS IS THE STATEMENT. `kv_extent` reports a KV span that has run
            // ahead of a request's token count, which happens only where a batched write appends every
            // row of a batch at ONE slot (the spyre pool). Here each request is written at its own
            // `slot_mapping` position, so slot == token, the whole computed prefix is cacheable, and the
            // scheduler's own arithmetic is already right — reporting would give it a second source for a
            // number it can derive correctly.
            //
            // ⛔⛔ THIS FILE DID NOT COMPILE FOR FOUR COMMITS BECAUSE THE FIELD WAS ADDED WITHOUT IT.
            // `--features metal` was broken on this branch from cc9d77c1 while every check I ran was
            // either `--features superdsc` or a host test in another crate. A backend that is not built
            // is not a backend that is fine: build EVERY feature a shared struct is used by.
            kv_extent: std::collections::HashMap::new(),
            // Token-addressed KV: the write slot IS the token count, so a new request's own allocation
            // already covers the page it writes. Nothing to report.
            kv_pool_reach: None,
            d2h_resolver: None,
        })
    }

    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        // Metal pipelines are JIT-compiled lazily via the MetalWorkerPool's
        // function-constant cache; no eager warmup needed for Step 2.
        //
        // Release the aligned-sidecar cache writers: load + KV init are
        // done, so the one-time background build no longer contends
        // with the realign-copy's page-ins (the contention turned an
        // 8-9.5 s miss launch into 18.6 s).
        scratchy_target_metal::metal_allocator::signal_weights_load_complete();
        Ok(())
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
        // Drain any in-flight aligned-sidecar writer FIRST, while the
        // source buffers it reads are still alive on `self`. The writer
        // is detached, so on a short `scr chat -q` run the process would
        // otherwise exit mid-write — the cache never persists and every
        // launch re-pays the cold realign-copy. Blocks only on the
        // one-time first-run build; warm runs find nothing and return
        // instantly. (regression: c54955a3)
        scratchy_target_metal::metal_allocator::join_sidecar_writers();
        // Deterministically release the GPU residency set (system-wired
        // memory) BEFORE dropping the device. `requestResidency` wires every
        // weight / KV / scratch allocation; on a hard exit the kernel's
        // reclaim is intermittent and orphans that wired GPU memory. Ending
        // residency on the graceful path (SIGTERM/Ctrl-C → this teardown)
        // makes reclaim deterministic. (kill -9 can't reach here.)
        if let Some(dev) = self.gpu_device.as_ref() {
            dev.allocator.residency().shutdown();
            // Distinct un-wired weights set (default) holds no residency
            // request, but still drop its allocations deterministically
            // before the device goes away. No-op when it aliases the wired
            // set above (WeightResidency::Wired; idempotent via `ended`).
            dev.allocator.weights_residency().shutdown();
        }
        // Drop order matters: KV cache references device buffers; model
        // holds Weights backed by `GpuWeights` whose allocator arenas
        // back every weight tensor. Drop tensors before device.
        self.kv_cache = None; // drops the file-backed store(s) → Drop msyncs their codes
        self.model = None;
        self.argmax_kernels = None;
        #[cfg(feature = "guided-decoding")]
        {
            self.grammar_mask_kernels = None;
            self.grammar_pending = None;
            self.grammar_buf_allow = None;
            self.grammar_buf_rows = None;
            self.grammar_buf_gconsts = None;
            self.grammar_cap_allow = 0;
            self.grammar_cap_rows = 0;
            self.grammar_states.clear();
            self.grammar_factory = None;
        }
        self.gpu_device = None;
        self.metal_device = None;
    }

    fn rank(&self) -> usize {
        self.config.tp_rank
    }

    fn local_rank(&self) -> usize {
        self.config.tp_rank
    }

    fn is_driver_worker(&self) -> bool {
        self.config.tp_rank == 0
    }

    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        self.preloaded_tokenizer.take()
    }

    fn architecture(&self) -> Option<String> {
        self.resolved_architecture.clone()
    }

    fn spec_decode_backend(
        &mut self,
    ) -> Option<&mut dyn scratchy_serving_engine::spec_decode::SpecDecodeBackend> {
        Some(self as &mut dyn scratchy_serving_engine::spec_decode::SpecDecodeBackend)
    }
}

// ---------------------------------------------------------------------------
// WorkerFactory registrations — backend-neutral dispatch (see worker_factory.rs)
// ---------------------------------------------------------------------------

/// Metal (Apple Silicon) backend factory.
///
/// scratchy-target-metal is the only Apple-silicon backend. The historical MLX
/// fallback (`vllm-mlx`) was deleted because it silently masked
/// scratchy-target-metal gaps — a checkpoint whose arch × quantization wasn't
/// registered with scratchy-target-metal would silently route to MLX and produce
/// garbage from MLX-side bugs the user attributed to scratchy-target-metal. Now the
/// fallback is a hard error pointing at exactly what to add.
#[cfg(feature = "metal")]
pub struct MetalWorkerFactory;

#[cfg(feature = "metal")]
impl crate::worker_factory::WorkerFactory for MetalWorkerFactory {
    fn matches(&self, device: &str) -> bool {
        matches!(device, "auto" | "metal")
    }

    fn create(
        &self,
        worker_config: WorkerCreateConfig,
        _progress: Option<crate::worker_factory::ProgressCallback>,
    ) -> anyhow::Result<crate::worker_factory::WorkerCreationResult> {
        use anyhow::Context as _;

        use crate::error::ExecutorError;

        let mut worker = MetalWorker::new(worker_config);
        worker
            .init_device()
            .context("failed to initialize Metal device for MetalWorker")?;
        match worker.load_model() {
            Ok(()) => {
                info!("Using scratchy-target-metal backend (Apple Silicon GPU)");
                let hf_config = worker
                    .hf_config()
                    .context("model config not available after scratchy-target-metal load")?
                    .clone();
                let model_dir = worker.model_dir().map(|p| p.to_path_buf());
                let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
                Ok((Box::new(worker), hf_config, model_dir, dtype_elem_bytes))
            }
            Err(ExecutorError::ArchNotSupported(arch)) => {
                Err(anyhow::anyhow!("Unsupported arch `{arch}`"))
            }
            Err(e) => Err(e).context("failed to load scratchy-target-metal model"),
        }
    }

    fn recommended_working_set_size(&self) -> Option<u64> {
        crate::metal_info::recommended_max_working_set_size()
    }
}

#[cfg(feature = "metal")]
inventory::submit!(&MetalWorkerFactory as &dyn crate::worker_factory::WorkerFactory);
