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
#[cfg(feature = "metal")]
use scratchy_core_common::lend::unlent_bytes;
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
#[cfg(feature = "metal")]
use scratchy_target_metal::interpreter::metal::{Deferral, DeviceInput, ForwardError, InFlight};
#[cfg(feature = "metal")]
use std::sync::Arc;

// Backend-neutral types used by metal lifecycle bodies. Under cfg(metal),
// `GpuDevice` resolves to the Apple-silicon arm carrying `device + queue +
// allocator`; `OwnedTensor` / `TensorView` / `GpuTensor` / `GpuWeights` are
// the ones lifted to `cfg(any(cuda, metal))` in Step 1; `MetalAllocator`
// is the metal-side `BackendAllocator`; `MetalMem` wraps `metal::Buffer`
// for the metal arm of the KV/GDN pools.
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
}

/// GPU-resident form of [`GrammarMaskHost`]: the uploaded bitset / row
/// buffers + the MTL4
/// argument table, captured by the forward followup closure to dispatch
/// the grammar mask just before argmax. `kernels_addr` is a raw pointer
/// to the worker's `GrammarMaskKernels` (same `'static` trick the argmax
/// path uses for `argmax_kernels`); valid for the synchronous duration of
/// the forward call.
#[cfg(all(feature = "metal", feature = "guided-decoding"))]
struct GrammarMaskGpu {
    allow_bits: scratchy_target_metal::grammar_mask::Buffer,
    rows: scratchy_target_metal::grammar_mask::Buffer,
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
    /// ⭐⭐⭐ THE RESOLVED `max_num_seqs` — the width every worker-side
    /// consumer actually builds at (GDN state pool slots, sampler-arena
    /// rows, spec-decode chain bounds). Set at the END of `load_model` by
    /// [`MetalWorker::resolve_max_num_seqs`]: the caller's explicit ask, or
    /// (unset) the device tier clamped to what this device's memory affords
    /// where the model carries per-sequence GDN state. Reported to the
    /// engine via [`Worker::max_num_seqs_override`] so the scheduler admits
    /// exactly this many — one process, one number. `None` until
    /// `load_model` resolves it.
    resolved_max_num_seqs: Option<usize>,
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
    /// The argmax baked for the draft model's logits (its forwards and chain).
    draft_argmax_kernels: Option<scratchy_target_metal::argmax::ArgmaxKernels>,
    /// Compiled on-GPU token sampler (temperature / top-k / top-p / min-p +
    /// penalties). Cached once at load_model alongside `argmax_kernels`; used
    /// only for non-greedy requests (greedy decode stays on the argmax path).
    sampler_kernels: Option<scratchy_target_metal::sampling::SamplerKernels>,
    /// The sampler's persistent GPU arena (buffers + argument tables), built
    /// once at load_model — vocab, max rows and history bounds are
    /// compile-time facts of the loaded config, so per-step allocation was
    /// pure waste. Buffers are pinned in the allocator's residency set, so
    /// every forward command buffer sees them resident.
    sampler_arena: Option<std::sync::Arc<scratchy_target_metal::sampling::SamplerArena>>,
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
    /// The greedy argmax's outputs, one per step: a step takes one nothing else
    /// holds — no step in flight, no output the engine has not resolved — and
    /// grows it only when the step has more rows.
    argmax_slots: Vec<Arc<ArgmaxSlot>>,
    /// This step's [`Deferral`], set by `execute_model` for
    /// `forward_argmax_blocking` to commit the target forward under; `None`
    /// for a step the host waits for.
    deferral: Option<Deferral>,
    /// The deferred forward `forward_argmax_blocking` committed, back for
    /// `execute_model`.
    committed: Option<(InFlight, Arc<ArgmaxSlot>)>,
    /// Steps committed without waiting for them, oldest first.
    in_flight: std::collections::VecDeque<InFlightStep>,
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
    /// The grammar mask's allow-bitsets and row map, reused across
    /// decode steps (memcpy per step) and grown only when a batch needs more.
    #[cfg(feature = "guided-decoding")]
    grammar_buf_allow: Option<scratchy_target_metal::residency::Pinned>,
    #[cfg(feature = "guided-decoding")]
    grammar_buf_rows: Option<scratchy_target_metal::residency::Pinned>,
    /// Phase 6 chain-advance kernel. One small kernel that bumps
    /// per-req `runtime.positions` / `slot_mapping` / `seqused_k` in
    /// place between K-step chain iters. Cached at load_model so the
    /// pipeline is built once.
    chain_advance_kernel: Option<scratchy_target_metal::chain_advance::ChainAdvanceKernel>,
    /// The chain advance baked for the draft model's KV block size.
    draft_chain_advance_kernel: Option<scratchy_target_metal::chain_advance::ChainAdvanceKernel>,
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
    /// The target's weight-store refs, snapshot before the target's load consumed them, when a
    /// draft is configured: a multi-token-prediction head is lent the target's token embedding
    /// and lm_head from them (same mmap, so nothing is uploaded twice).
    target_tensor_refs: Option<scratchy_target_metal::weights::TensorRefs>,
    /// One placement-sparse MTLBuffer per (layer, K/V) for the target KV pool
    /// — `layer * 2 + kv_idx` indexing. The pool's logical chunks are byte
    /// offsets within these buffers, whose pages are mapped in only as the
    /// pool grows (`sparse_kv`). Always populated on the metal path (the
    /// alternative — one MTLBuffer per chunk — exposed too many distinct VA
    /// ranges to the GPU's UAT, costing ~5% per decode forward via the
    /// kernel's `chunk_table[chunk]` device load).
    target_kv_layers: Vec<scratchy_target_metal::sparse_kv::SparseKvLayer>,
    /// Maps both KV pools' chunks as they grow; waited out before a forward
    /// reads them. Set in `init_cache`.
    kv_mapper: Option<std::sync::Arc<scratchy_target_metal::sparse_kv::KvMapper>>,
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
    /// Same as `target_kv_layers` for the draft pool. Required so
    /// the `AttentionViaCache` pipeline's `chunk_table[0]` always points
    /// at the layer base — both target's and draft's pipelines see the
    /// same kernel function constants (set process-wide by lowering).
    draft_kv_layers: Vec<scratchy_target_metal::sparse_kv::SparseKvLayer>,
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

/// One step's per-token KV write targets, in the kernels' `slot_mapping`
/// encoding: the slot, with [`UNROTATED_BLOCK_BIT`] for a span block, or
/// `u32::MAX` to write nothing. A slot has at most one writer per step. Two
/// sequences can be handed the same block for tokens they both compute (a full
/// prefix-cache hit backs off one block but keeps the hit block), and a second
/// write races with the first (TurboQuant's prefill staging re-rotates the row
/// in place), so a later writer skips and reads the first writer's keys like
/// any cached row.
///
/// [`UNROTATED_BLOCK_BIT`]: scratchy_target_metal::UNROTATED_BLOCK_BIT
#[cfg(feature = "metal")]
struct StepSlotMapping {
    slots: Vec<u32>,
    written: std::collections::HashSet<u32>,
}

#[cfg(feature = "metal")]
impl StepSlotMapping {
    const SKIP: u32 = u32::MAX;

    fn with_capacity(tokens: usize) -> Self {
        Self {
            slots: Vec::with_capacity(tokens),
            written: std::collections::HashSet::with_capacity(tokens),
        }
    }

    /// The next token writes its K/V to `slot`, K unrotated if `unrotated`.
    fn write(&mut self, slot: usize, unrotated: bool) {
        let bit = scratchy_target_metal::UNROTATED_BLOCK_BIT;
        let slot = u32::try_from(slot)
            .ok()
            .filter(|s| s & bit == 0)
            .expect("KV slot below the unrotated-block bit");
        if !self.written.insert(slot) {
            return self.skip();
        }
        self.slots.push(if unrotated { slot | bit } else { slot });
    }

    /// The next token writes no K/V.
    fn skip(&mut self) {
        self.slots.push(Self::SKIP);
    }

    fn into_slots(self) -> Vec<u32> {
        self.slots
    }
}

/// Bytes per element of a metal KV pool's dense rows: bf16 and f16 are both 2;
/// int4 KV is unsupported.
#[cfg(feature = "metal")]
const METAL_KV_ELEM_BYTES: usize = 2;

/// Per-block KV bytes for one model: layers × 2 (K+V) × heads × head_dim × block_size
/// × [`METAL_KV_ELEM_BYTES`].
#[cfg(feature = "metal")]
fn kv_per_block_bytes(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    block_size: usize,
) -> usize {
    let elt_bytes = METAL_KV_ELEM_BYTES;
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
    block_size.saturating_mul(pool_bytes_per_token(
        model,
        scratchy_forward_compiler::KvCodec::Dense,
        elt_bytes,
    ))
}

/// `kv_bytes_per_token` at `model`'s pool geometry: every layer that holds KV
/// ([`kv_layer_tensors`]), its KV heads and head_dim, `dense_elem_bytes` wide
/// when `codec` is dense.
#[cfg(feature = "metal")]
fn pool_bytes_per_token(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    codec: scratchy_forward_compiler::KvCodec,
    dense_elem_bytes: usize,
) -> usize {
    let layers = kv_layer_tensors(model).map_or(model.num_hidden_layers() as usize, |m| {
        m.iter().copied().max().map_or(0, |t| t + 1)
    });
    scratchy_target_metal::turboquant::kv_bytes_per_token(
        codec,
        layers,
        model.num_key_value_heads() as usize,
        model.head_dim() as usize,
        dense_elem_bytes,
    )
}

/// A Gated-DeltaNet hybrid's KV pool: one physical tensor per attention layer,
/// in layer order; a linear-attention layer holds no KV, and maps to tensor 0,
/// which its forward never binds. `None` for a model whose every layer holds KV.
#[cfg(feature = "metal")]
fn kv_layer_tensors(model: &dyn scratchy_forward_compiler::ScratchyWeights) -> Option<Vec<usize>> {
    let linear = model.gdn_runtime_config()?.linear_layers;
    let mut next = 0;
    let map = (linear.iter())
        .map(|&linear| match linear {
            true => 0,
            false => {
                next += 1;
                next - 1
            }
        })
        .collect();
    Some(map)
}

/// vLLM group-shared hybrid KV layout (gemma4): infer the per-layer sliding
/// mask from `per_layer_kv_token_elems` (the bigger-page class is the sliding
/// class) and compute the grouping. This is `Some` only for the
/// page-differentiated case — the SAME trigger the scheduler's `hybrid_kv` uses
/// in `init`, so the two never disagree on group count. `None` → one physical
/// tensor per layer (the uniform pool).
#[cfg(feature = "metal")]
fn hybrid_kv_layout(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    block_size: usize,
    elem_bytes: usize,
) -> Option<scratchy_core_config::HybridKvLayout> {
    let elems = model.per_layer_kv_token_elems()?;
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
    // `available` is irrelevant here — the pool sizes from the passed
    // `num_gpu_blocks`; callers only read the grouping fields.
    scratchy_core_config::compute_hybrid_kv_layout(&geom, block_size, usize::MAX / 2, elem_bytes)
}

/// A uniform model built with TurboQuant: its KV lives in packed codes and the
/// fp16 pool is a one-chunk seed (`initialize_cache`). A hybrid model's
/// TurboQuant global layers keep their codes BESIDE its fp16 pool, which the
/// sliding layers read ([`target_block_bytes`] charges both).
#[cfg(feature = "metal")]
fn uniform_turboquant(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    block_size: usize,
) -> bool {
    model.kv_codec().is_turboquant()
        && hybrid_kv_layout(model, block_size, METAL_KV_ELEM_BYTES).is_none()
}

/// Bytes one block of the TARGET pool costs: a hybrid model's page in each of
/// its group-shared tensors plus, built with TurboQuant, the global layers'
/// codes, norms and scratch; a uniform TurboQuant model's codes, norms and
/// scratch; every other model's dense rows.
#[cfg(feature = "metal")]
fn target_block_bytes(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    block_size: usize,
) -> usize {
    if let Some(layout) = hybrid_kv_layout(model, block_size, METAL_KV_ELEM_BYTES) {
        let shared_pages = layout.page_size_bytes * layout.group_size;
        let codec = model.kv_codec();
        if !codec.is_turboquant() {
            return shared_pages;
        }
        // What `build_tq_provision` allocates per block: the GLOBAL group's
        // (group 0) layers at the global geometry, `full_block_size` tokens each.
        let tq = scratchy_target_metal::interpreter::metal::MetalRungs::of(model.metal_rungs())
            .expect("a metal model's baked rungs are MetalRungs")
            .tq;
        let num_global = layout.layer_to_group.iter().filter(|&&g| g == 0).count();
        return shared_pages
            + layout.full_block_size()
                * scratchy_target_metal::turboquant::kv_bytes_per_token(
                    codec,
                    num_global,
                    tq.kv_heads.get() as usize,
                    tq.head_dim.get() as usize,
                    METAL_KV_ELEM_BYTES,
                );
    }
    if uniform_turboquant(model, block_size) {
        block_size.saturating_mul(pool_bytes_per_token(
            model,
            model.kv_codec(),
            METAL_KV_ELEM_BYTES,
        ))
    } else {
        kv_per_block_bytes(model, block_size)
    }
}

/// Bytes one block of the DRAFT pool costs: it is allocated dense in full
/// (`initialize_draft_cache_metal`), and a TurboQuant draft's codec buffers
/// come on top of it.
#[cfg(feature = "metal")]
fn draft_block_bytes(
    model: &dyn scratchy_forward_compiler::ScratchyWeights,
    block_size: usize,
) -> usize {
    let codec = match model.kv_codec() {
        scratchy_forward_compiler::KvCodec::Dense => 0,
        codec => block_size.saturating_mul(pool_bytes_per_token(model, codec, METAL_KV_ELEM_BYTES)),
    };
    kv_per_block_bytes(model, block_size).saturating_add(codec)
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

// `gdn_slot_key` and the shared unset-`--max-num-seqs` resolver are
// backend-neutral; they live in `scratchy-serving-engine` (cycle-free for
// both the CUDA and Metal workers) and are re-exported here.
#[cfg(feature = "metal")]
use scratchy_serving_engine::gpu_budget::MaxNumSeqsFacts;
pub use scratchy_serving_engine::gpu_budget::{gdn_slot_key, resolve_default_max_num_seqs};

#[cfg(feature = "metal")]
impl MetalWorker {
    pub fn new(config: WorkerCreateConfig) -> Self {
        Self {
            config,
            kv_cache: None,
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
            resolved_max_num_seqs: None,
            model: None,
            #[cfg(feature = "vision")]
            mm: None,
            mm_pending: None,
            draft_model: None,
            draft_model_dir: None,
            draft_hf_config: None,
            draft_kv_cache: None,
            argmax_kernels: None,
            draft_argmax_kernels: None,
            sampler_kernels: None,
            sampler_arena: None,
            sampler_logits: None,
            pending_sampler: None,
            argmax_slots: Vec::new(),
            deferral: None,
            committed: None,
            in_flight: std::collections::VecDeque::new(),
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
            chain_advance_kernel: None,
            draft_chain_advance_kernel: None,
            draft_queue: None,
            target_tensor_refs: None,
            target_kv_layers: Vec::new(),
            kv_mapper: None,
            kv_full_block_size: 0,
            kv_is_hybrid: false,
            draft_kv_layers: Vec::new(),
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

    /// ⭐⭐⭐ THE UNSET `--max-num-seqs` ANSWER — called at the END of `load_model`,
    /// after weights are resident and `model.gdn_runtime_config()` is answerable, and BEFORE every
    /// consumer of the width (the sampler arena below, the GDN state pool + slot allocator in
    /// `initialize_cache`, the spec-decode chain bounds).
    ///
    /// * `Some(n)` (the caller asked): `n`, verbatim. An ask the device cannot afford is the OOM
    ///   guard's to refuse loudly at `determine_available_memory` — never silently replaced here
    ///   (capping-is-not-validating).
    /// * `None` (nobody asked): the shared backend-neutral resolver
    ///   ([`resolve_default_max_num_seqs`]) — the base 128 (metal reports no
    ///   device facts, so no tier answer; any tier bump is a separate,
    ///   measured decision) clamped to what this device's budget affords,
    ///   counting EVERY term the OOM guard counts: allocated weights, the
    ///   peak-activation estimate (bucket arena + rung scratch + 64 MiB),
    ///   the 150 MiB redundancy pad, the GDN per-slot bytes, AND the sampler
    ///   arena's per-row bytes — the width sizes the very arena allocated
    ///   right after this resolves. The clamp leaves the same
    ///   [`scratchy_serving_engine::gpu_budget::KV_FLOOR_BYTES`] the guard's
    ///   flag hint reserves, so a default never trips the guard it is sized
    ///   against (the Qwen3.5-MoE-35B "!!!!" incident: on main the base 128
    ///   × 61 MiB = 7.9 GiB of f32 recurrent state REFUSED loudly on a
    ///   squeezed box; the default now answers an affordable width and runs).
    ///
    /// The resolved value lands in [`Self::resolved_max_num_seqs`] and is reported through
    /// [`Worker::max_num_seqs_override`] so the scheduler admits exactly the width the pools were
    /// built at. (Unlike `WorkerCreateConfig::max_num_seqs`, `self.config` keeps the REQUEST: the
    /// OOM guard's "pass --max-num-seqs N" hint must speak about what the CALLER asked, not the
    /// default this method substituted.)
    fn resolve_max_num_seqs(
        &mut self,
        model: &dyn scratchy_forward_compiler::ScratchyWeights,
        gpu_device: &scratchy_target_metal::GpuDevice,
    ) -> ExecutorResult<()> {
        if self.resolved_max_num_seqs.is_some() {
            return Ok(());
        }
        let resolved = match self.config.max_num_seqs {
            Some(asked) => asked,
            None => {
                // Facts for the shared resolver — everything the guard's
                // arithmetic names, gathered here so the resolver stays
                // backend-neutral arithmetic (no near-copy can grow in any
                // target crate).
                let checkpoint_rows = self.gdn_checkpoint_rows()?;
                let gdn_per_slot = model.gdn_runtime_config().map(|cfg| {
                    GdnStatePool::<scratchy_target_metal::PoolMem>::reserve_bytes(
                        cfg.num_linear_layers(),
                        1,
                        checkpoint_rows,
                        cfg.state_dims(),
                    )
                });
                let (budget, allocated) = self
                    .metal_device
                    .as_ref()
                    .map(|dev| {
                        (
                            (dev.device.recommendedMaxWorkingSetSize() as f64
                                * self.config.gpu_memory_utilization)
                                as usize,
                            dev.device.currentAllocatedSize(),
                        )
                    })
                    // No device = no budget to clamp against: the base
                    // constant survives as-is (the OOM guard still refuses
                    // an unaffordable pool at determine_available_memory).
                    .map(|(b, a)| (Some(b), a))
                    .unwrap_or((None, 0));
                // The ONE peak-activation estimate this worker has (see
                // `metal_peak_activation_estimate`) — already draft-doubled
                // and rung-summed; the resolver and the OOM guard share it.
                // The target is still a local here, not `self.model` (see
                // the helper's parameter doc).
                let peak = self.metal_peak_activation_estimate(
                    Some(model),
                    Some(gpu_device),
                    None,
                    self.kv_block_cap(usize::MAX),
                )?;
                // The sampler arena allocated right AFTER this resolves is
                // sized by the width itself: n·(vocab + 2·max_hist)·4 plus
                // the sliced buffers' terms — count its per-row cost so the
                // default accounts for what it triggers.
                let max_model_len = self
                    .config
                    .max_model_len
                    .or_else(|| {
                        self.hf_config
                            .as_ref()
                            .and_then(|c| c.max_position_embeddings())
                    })
                    .unwrap_or(4096);
                let sampler_row = scratchy_target_metal::sampling::SamplerArena::bytes_per_row(
                    u32::try_from(model.vocab_size()).unwrap_or(u32::MAX),
                    u32::try_from(max_model_len.max(1)).unwrap_or(u32::MAX),
                );
                let facts = MaxNumSeqsFacts {
                    // Metal reports no device facts — the base constant, the
                    // SAME default as before the resolver existed.
                    device_total_bytes: None,
                    device_name: None,
                    device_budget_bytes: budget,
                    allocated_bytes: allocated,
                    peak_activation_bytes: peak,
                    gdn_per_slot_bytes: gdn_per_slot,
                    // A verify step samples each sequence's every row: its token and its drafts.
                    sampler_bytes_per_row: sampler_row * (1 + self.config.num_speculative_tokens),
                };
                resolve_default_max_num_seqs(&facts, false)
            }
        };
        info!(
            "ScratchyWorker(metal): max_num_seqs = {resolved} ({}; pass \
             --max-num-seqs to override)",
            if self.config.max_num_seqs.is_some() {
                "requested"
            } else {
                "default"
            }
        );
        self.resolved_max_num_seqs = Some(resolved);
        Ok(())
    }

    /// The peak-activation estimate BOTH the unset-`--max-num-seqs` resolver
    /// (at the end of `load_model`) and the OOM guard
    /// (`determine_available_memory`) budget against — ONE record of the
    /// fact: the prefill-bucket arena the budget can afford, every arena that
    /// runs at it priced in ([`Self::metal_bucket_costs`]; the full-ladder
    /// peak when the model emits no cost table, doubled when a draft model
    /// rides along), plus the KV cap rungs' scratch at `block_cap` blocks a
    /// sequence ([`Self::metal_rung_scratch`]), and the 64 MiB
    /// runtime/staging pad.
    fn metal_peak_activation_estimate(
        &self,
        // `load_model` runs this BEFORE `self.model`/`self.gpu_device` take
        // the loaded target (width consumers sit between the two), so the
        // caller hands the just-loaded pair in — `determine_available_memory`
        // (after field assignment) passes `None` and the fields serve.
        model: Option<&dyn scratchy_forward_compiler::ScratchyWeights>,
        gpu_device: Option<&scratchy_target_metal::GpuDevice>,
        // The budget the internal bucket selection prunes against. `None` =
        // the raw `working_set × utilization` figure — the load-time
        // `--max-num-seqs` resolver's view, which wants the conservative
        // pre-floor estimate. `determine_available_memory` passes its
        // KV-floored total so the arena returned here matches the bucket
        // its ladder will actually run (and the KV split subtracts exactly
        // that arena, not a stale collapsed one).
        budget: Option<usize>,
        block_cap: usize,
    ) -> ExecutorResult<usize> {
        let model = model
            .or(self.model.as_deref())
            .ok_or_else(|| ExecutorError::WorkerInit("peak estimate: model not loaded".into()))?;
        let dev = gpu_device.or(self.gpu_device.as_ref()).ok_or_else(|| {
            ExecutorError::WorkerInit("peak estimate: gpu_device not initialized".into())
        })?;
        let total = budget.unwrap_or_else(|| {
            self.metal_device
                .as_ref()
                .map(|dev| {
                    (dev.device.recommendedMaxWorkingSetSize() as f64
                        * self.config.gpu_memory_utilization) as usize
                })
                .unwrap_or(0)
        });
        let weights_and_overhead = self
            .metal_device
            .as_ref()
            .map(|dev| dev.device.currentAllocatedSize())
            .unwrap_or(0);
        let bucket_costs = self.metal_bucket_costs(model, dev)?;
        let arena_peak = if bucket_costs.is_empty() {
            model.metal_arena_peak_bytes() as usize
        } else {
            // The same 0.6 arena-fraction split `determine_available_memory`
            // runs; the pad mirrors the non-arena terms the KV formula
            // subtracts so the headroom here equals what is actually left.
            const ARENA_FRACTION: f64 = 0.6;
            let pad = (64 + 150) * 1024 * 1024usize;
            let fixed = weights_and_overhead.saturating_add(pad);
            let sel =
                select_prefill_bucket(total as u64, fixed as u64, &bucket_costs, ARENA_FRACTION);
            sel.arena_bytes as usize
        };
        // A selected bucket's price already counts the draft's arena. Without cost tables, the
        // target's full-ladder peak stands for the draft's too, a safe upper bound.
        let arena_peak_pair = match (bucket_costs.is_empty(), self.draft_model.is_some()) {
            (true, true) => arena_peak.saturating_mul(2),
            _ => arena_peak,
        };
        let rung_scratch = self.metal_rung_scratch(model, dev, block_cap)?;
        Ok((arena_peak_pair.saturating_add(64 * 1024 * 1024))
            .saturating_add(usize::try_from(rung_scratch).unwrap_or(usize::MAX)))
    }

    /// Each prefill bucket's arena price: the target's, and every arena that runs at it. A draft
    /// forward (an MTP head's pass 1, a draft model's lockstep prefill) runs the target's rows, so
    /// it needs the same bucket. A draft model's arena is its own, at the smallest of its buckets
    /// that holds them; a head places its arena in the target's (`draft_device`), so it adds what
    /// that does not hold. Empty when the target emits no cost table.
    fn metal_bucket_costs(
        &self,
        target: &dyn scratchy_forward_compiler::ScratchyWeights,
        dev: &scratchy_target_metal::GpuDevice,
    ) -> ExecutorResult<Vec<(u32, u64)>> {
        let draft = self.draft_model.as_deref();
        let draft_costs = draft.map_or(&[][..], |m| m.metal_bucket_arena_costs());
        let head = draft.filter(|m| m.reads_target_hidden());
        (target.metal_bucket_arena_costs().iter())
            .map(|&(bucket_m, bytes)| {
                let draft = match head {
                    Some(head) => {
                        let layout = |m: &dyn scratchy_forward_compiler::ScratchyWeights| {
                            let rung = Self::metal_pick(m, dev, Some(bucket_m), 1)?;
                            ExecutorResult::Ok(rung.arena_layout())
                        };
                        unlent_bytes(&layout(head)?, &layout(target)?)
                    }
                    None => (draft_costs.iter())
                        .find(|&&(m, _)| m >= bucket_m)
                        .or(draft_costs.last())
                        .map_or(0, |&(_, bytes)| bytes),
                };
                Ok((bucket_m, bytes + draft))
            })
            .collect()
    }

    /// The scratch of the KV cap rung each pool runs on at `block_cap` blocks a sequence: the
    /// target's, and a draft's — a head's only what the target's scratch does not hold, as its
    /// arena.
    fn metal_rung_scratch(
        &self,
        target: &dyn scratchy_forward_compiler::ScratchyWeights,
        dev: &scratchy_target_metal::GpuDevice,
        block_cap: usize,
    ) -> ExecutorResult<u64> {
        let pick = |m: &dyn scratchy_forward_compiler::ScratchyWeights| {
            Self::metal_pick(m, dev, dev.metal_bucket_max_m, block_cap)
        };
        let target_rung = pick(target)?;
        let draft = self.draft_model.as_deref();
        let draft_bytes = match draft {
            Some(head) if head.reads_target_hidden() => {
                unlent_bytes(&pick(head)?.scratch_sizes(), &target_rung.scratch_sizes())
            }
            Some(draft) => pick(draft)?.scratch_bytes(),
            None => 0,
        };
        Ok(target_rung.scratch_bytes() + draft_bytes)
    }

    /// The resolved width, floored at 1. Every consumer of the width calls
    /// this — never `self.config.max_num_seqs`, which keeps the REQUEST so
    /// error messages can speak about what the caller asked.
    fn max_num_seqs_resolved(&self) -> usize {
        self.resolved_max_num_seqs.unwrap_or(1).max(1)
    }

    /// The KV cap rung `model`'s pool runs on for a capacity of `block_cap` blocks per sequence,
    /// on this worker's device and workload (`MetalRungs::pick`): `(cap, scratch bytes)`. The cap
    /// is the KV pool's block-table width; the scratch counts in the memory budget.
    fn metal_rung(
        &self,
        model: &dyn scratchy_forward_compiler::ScratchyWeights,
        block_cap: usize,
        dev: &scratchy_target_metal::GpuDevice,
    ) -> ExecutorResult<(usize, u64)> {
        let rung = Self::metal_pick(model, dev, dev.metal_bucket_max_m, block_cap)?;
        Ok((rung.cap.get() as usize, rung.scratch_bytes()))
    }

    /// [`Self::metal_rung`]'s pick on `dev`, its buckets those up to `max_m`.
    fn metal_pick(
        model: &dyn scratchy_forward_compiler::ScratchyWeights,
        dev: &scratchy_target_metal::GpuDevice,
        max_m: Option<u32>,
        block_cap: usize,
    ) -> ExecutorResult<scratchy_target_metal::interpreter::metal::PickedRung<'static>> {
        let rungs = scratchy_target_metal::interpreter::metal::MetalRungs::of(model.metal_rungs());
        let pick = |r: &scratchy_target_metal::interpreter::metal::MetalRungs| {
            r.pick(&dev.device, max_m, block_cap, dev.kv_addressing)
        };
        rungs
            .and_then(pick)
            .map_err(|e| ExecutorError::WorkerInit(format!("KV cap rung: {e}")))
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

    /// Whether the loaded draft is a multi-token-prediction head: it reads the target's hidden
    /// states, so it runs after the target, never beside it.
    fn draft_is_head(&self) -> bool {
        (self.draft_model.as_ref()).is_some_and(|m| m.reads_target_hidden())
    }

    /// The device a draft forward runs on: the target's device and allocator, on the draft queue
    /// when there is one (a draft model runs beside the target). A head runs after the target, on
    /// its queue, each forward waited on, so it places its pool's buffers in the target's.
    fn draft_device(
        &self,
    ) -> Result<GpuDevice, ::scratchy_serving_engine::spec_decode::BackendError> {
        use ::scratchy_serving_engine::spec_decode::BackendError;
        let main = (self.gpu_device.as_ref())
            .ok_or_else(|| BackendError::Backend("gpu_device not initialized".into()))?;
        let lent = match self.model.as_deref() {
            Some(target) if self.draft_is_head() => target.metal_lend_activation(),
            _ => None,
        };
        let lent =
            lent.map(|l| l.downcast::<scratchy_target_metal::interpreter::metal::LentActivation>());
        let lent = (lent.transpose())
            .map_err(|_| BackendError::Backend("target lent no LentActivation".into()))?;
        Ok(GpuDevice {
            device: main.device.clone(),
            queue: (self.draft_queue.clone()).unwrap_or_else(|| main.queue.clone()),
            allocator: main.allocator.clone(),
            metal_bucket_max_m: main.metal_bucket_max_m,
            kv_addressing: main.kv_addressing,
            lent: lent.map(|l| *l).unwrap_or_default(),
        })
    }

    /// Checkpoints each GDN state slot keeps: one per draft of a verify step, so a rejected draft
    /// can be dropped. Sizes both the pool and its memory reservation.
    fn gdn_checkpoint_rows(
        &self,
    ) -> ExecutorResult<scratchy_target_metal::gdn_state::CheckpointRows> {
        let drafts = self.config.num_speculative_tokens;
        u8::try_from(drafts)
            .map(scratchy_target_metal::gdn_state::CheckpointRows)
            .map_err(|_| {
                ExecutorError::WorkerInit(format!(
                    "GdnStatePool: {drafts} speculative tokens exceed a slot's u8 checkpoints"
                ))
            })
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
        let draft_arch = draft_hf_config.arch_hint().unwrap_or_default().to_string();

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
        // A multi-token-prediction head carries no embedding or lm_head: lend it the target's,
        // from the target store's refs (same mmap → no second upload).
        if let Some(head) = scratchy_forward_compiler::draft_head(&draft_arch) {
            let (Some(target), Some(refs)) =
                (self.model.as_deref(), self.target_tensor_refs.as_ref())
            else {
                return Err(ExecutorError::WorkerInit(
                    "MTP head: the target must load first (its weight refs lend the head)".into(),
                ));
            };
            for w in scratchy_forward_compiler::LentWeight::ALL {
                let from = target.lent_weight_prefix(w).ok_or_else(|| {
                    ExecutorError::WorkerInit(format!("MTP head: the target has no {w:?} to lend"))
                })?;
                let lent = draft_weights
                    .lend(refs, from, head.prefix(w), head.rows(w))
                    .map_err(|e| {
                        ExecutorError::WorkerInit(format!("MTP head: lending {w:?}: {e}"))
                    })?;
                info!(
                    "ScratchyWorker(metal): MTP head lent the target's {w:?} ({lent} tensors, {from} → {})",
                    head.prefix(w)
                );
            }
        }

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
        .map_err(|e| ExecutorError::WorkerInit(format!("draft try_load: {e}")))?;
        let draft_model = match draft_model {
            scratchy_forward_compiler::ArchLoad::Loaded(m) => m,
            scratchy_forward_compiler::ArchLoad::NoVariantMatched => {
                return Err(ExecutorError::NoVariantMatched(draft_arch.clone()));
            }
            scratchy_forward_compiler::ArchLoad::ArchNotCompiled => {
                return Err(ExecutorError::ArchNotSupported(draft_arch.clone()));
            }
        };

        info!(
            "ScratchyWorker(metal): draft try_load in {:?} ({} via scratchy-forward-compiler, {})",
            t_try_load.elapsed(),
            draft_arch,
            draft_model.arch_name()
        );

        let draft_off_tape =
            scratchy_target_metal::off_tape::OffTapeKernels::of(draft_model.metal_off_tape())
                .map_err(|e| ExecutorError::WorkerInit(format!("draft off-tape kernels: {e}")))?;
        self.draft_argmax_kernels = Some(
            scratchy_target_metal::argmax::ArgmaxKernels::new(&gpu_device.device, draft_off_tape)
                .map_err(|e| ExecutorError::WorkerInit(format!("draft argmax compile: {e:?}")))?,
        );
        self.draft_chain_advance_kernel = Some(
            scratchy_target_metal::chain_advance::ChainAdvanceKernel::new(
                &gpu_device.device,
                draft_off_tape,
            )
            .map_err(|e| {
                ExecutorError::WorkerInit(format!("draft chain_advance compile: {e:?}"))
            })?,
        );
        self.draft_model = Some(draft_model);
        self.draft_model_dir = Some(draft_dir);
        self.draft_hf_config = Some(draft_hf_config);
        // Spec-decode runs the chunked-addressing tape rung: the direct (BPC=0) path has an
        // unresolved interaction with the draft K-step chain on 3B-class+ models (out-of-vocab
        // draft tokens). Every pool picks its rung by this device field.
        if let Some(dev) = self.gpu_device.as_mut() {
            dev.kv_addressing = scratchy_target_metal::tape::lowered::KvAddressing::Chunked;
        }

        // Phase 8 foundation: dedicated MTLCommandQueue for the draft
        // chain. Two queues on the same device run concurrently on
        // Apple Silicon; this enables lockstep prefill to overlap
        // with target verify (separate KV pools → no contention). A
        // multi-token-prediction head reads the target's output, so
        // nothing of it can overlap the target: it runs on the main queue.
        if self.draft_queue.is_none() && !self.draft_is_head() {
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

    /// Whether this step can be committed without waiting for it: its rows
    /// pick their tokens on the GPU from the logits alone, so nothing on the
    /// host needs its tokens before the next step is queued. Penalties (the
    /// token history), grammar, speculative decoding, pooling, multimodal
    /// input, a draft model and live sampler telemetry all do.
    fn deferrable(&self, sched: &SchedulerOutput) -> bool {
        #[cfg(feature = "guided-decoding")]
        let constrained = |req_id: &String| self.grammar_states.contains_key(req_id);
        #[cfg(not(feature = "guided-decoding"))]
        let constrained = |_: &String| false;
        #[cfg(feature = "sampler-telemetry")]
        let telemetry =
            scratchy_core_common::sampler_telemetry::SamplerTelemetry::global().is_enabled();
        #[cfg(not(feature = "sampler-telemetry"))]
        let telemetry = false;
        !telemetry
            && sched.total_num_scheduled_tokens > 0
            && self.input_batch.num_active() > 0
            && sched.scheduled_spec_decode_tokens.is_empty()
            && !self.config.is_pooling
            && self.draft_model.is_none()
            && sched.num_scheduled_tokens.keys().all(|req_id| {
                !self
                    .sampling_params_map
                    .get(req_id)
                    .is_some_and(SamplingParams::reads_history)
                    && !self.mm_data_buffers.contains_key(req_id)
                    && !constrained(req_id)
            })
    }

    /// Wait for the oldest step still on the device and append the tokens it
    /// sampled to its requests' histories.
    fn resolve_oldest(&mut self) -> ExecutorResult<()> {
        let step = self.in_flight.pop_front().expect("a step in flight");
        let tokens = sampled(
            &step.done,
            &step.argmax,
            step.rows.iter().map(|&(_, row)| row),
        )
        .map_err(|e| ExecutorError::WorkerExecution(format!("deferred forward: {e}")))?;
        for ((req_id, _), token) in step.rows.iter().zip(tokens) {
            self.input_batch.resolve(req_id, token);
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
    fn grow_metal_kv_to_cover(&mut self, max_block: usize) -> ExecutorResult<()> {
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
        if self
            .model
            .as_ref()
            .is_some_and(|m| m.kv_codec().is_turboquant())
            && !self.kv_is_hybrid
        {
            return Ok(());
        }
        let Some(device) = self.gpu_device.as_ref() else {
            return Ok(());
        };
        // Hold the residency set so we can make the chunk(s) that `grow_to_cover`
        // is about to allocate GPU-resident. A freshly-allocated MTLBuffer chunk
        // is NOT resident until it is committed to the residency set, and the GPU
        // reads it through the bindless chunk-address table on the next dispatch.
        let residency = device.allocator.residency().clone();
        // `grow_to_cover` allocates K+V for each PHYSICAL TENSOR (group-shared:
        // gemma4 has `num_tensors` < `num_layers`), so the slot cycle is
        // `2 * num_tensors` — matching `target_kv_layers.len()` and the
        // init-time `alloc_chunk` closure. Using `num_layers` here would index
        // past the buffer vec on the hybrid pool.
        let n_slots = self
            .kv_cache
            .as_ref()
            .map(|kv| kv.num_tensors * 2)
            .unwrap_or(0);
        if n_slots == 0 {
            return Ok(());
        }
        let kv_layers_ref = &mut self.target_kv_layers;
        let Some(kv) = self.kv_cache.as_mut() else {
            return Ok(());
        };
        let alloc_chunk_counter = std::cell::Cell::new(0usize);
        // `grow_to_cover` allocates chunk by chunk, every tensor's K then V in slot
        // order, from the chunk after the last one the pool holds whole.
        let bpc = scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize;
        let first = kv.allocated_blocks().div_ceil(bpc);
        let t_grow = std::time::Instant::now();
        // Returns Ok(n) where n = number of NEW chunks allocated (Ok(0) = the pool
        // already covered it); Err if a chunk's allocation failed, with the pool
        // holding exactly the chunks it held whole before it.
        let grew_result = kv.grow_to_cover(
            kv.growth_target(max_block),
            |bytes| {
                let c = alloc_chunk_counter.get();
                alloc_chunk_counter.set(c + 1);
                let (slot, chunk) = (c % n_slots, first + c / n_slots);
                kv_layers_ref[slot]
                    .commit_chunk(chunk, bytes)
                    .map_err(|e| anyhow::anyhow!("KV grow slot={slot}: {e}"))
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
        // A grow that failed fails the step: the forward would read the chunk it
        // could not build. Chunks it did map stay mapped for the retry.
        let grew = grew_result.map_err(|e| {
            ExecutorError::WorkerExecution(format!("KV pool grow to block {max_block}: {e}"))
        })?;
        if grew > 0 {
            // The new chunks' pages are mapped on the mapper's queue: in place before
            // the forward reads them, then wired.
            if let Some(mapper) = self.kv_mapper.as_ref() {
                mapper.wait();
            }
            residency.commit();
            info!(
                "ScratchyWorker(metal): KV pool grew {grew} chunks to {} blocks in {:.1} ms",
                kv.allocated_blocks(),
                t_grow.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }

    /// Reactive KV (2c): when the batch is fully idle (no live blocks), drop
    /// the pool back to 1 chunk. The layers keep the dropped chunks mapped
    /// (`SparseKvLayer::commit_chunk` hands the same ranges back on
    /// regrowth), so memory stays at its high-water mark. Only call when
    /// `input_batch.num_active() == 0` so no in-flight forward references a
    /// dropped chunk index.
    fn shrink_metal_kv_idle(&mut self) {
        let freed = match self.kv_cache.as_mut() {
            Some(kv) => kv.shrink_to_chunks(1),
            None => return,
        };
        if freed.is_empty() {
            return;
        }
        let n = freed.len();
        drop(freed);
        info!(
            "ScratchyWorker(metal): reactive KV shrink — {} chunk handles dropped on {} layers",
            n,
            self.target_kv_layers.len()
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
        // The draft pool's block-table width: the draft model's KV cap rung for the target's
        // max_model_len, capped at the draft block count.
        let draft_dev = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerInit("gpu_device not initialized".into()))?;
        let (draft_block_cap, _) =
            self.metal_rung(model.as_ref(), self.kv_block_cap(draft_blocks), draft_dev)?;

        let cache_dtype = match model.metal_dtype() {
            scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 => GpuDType::BF16,
            scratchy_target_metal::interpreter::metal::MetalDtype::F16 => GpuDType::F16,
            scratchy_target_metal::interpreter::metal::MetalDtype::Int4 => {
                return Err(ExecutorError::WorkerInit(
                    "draft KV cache cannot be int4-quantized".into(),
                ));
            }
        };

        let mtl_device = draft_dev.device.clone();
        let residency = draft_dev.allocator.residency().clone();

        let t_pool = std::time::Instant::now();
        let blocks_per_chunk = scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize;
        let num_layers_draft = model.num_hidden_layers() as usize;
        let num_chunks_total_draft = draft_blocks.div_ceil(blocks_per_chunk);
        let per_block_elems_draft = (model.num_key_value_heads() as usize)
            * self.config.block_size
            * (model.head_dim() as usize);
        let chunk_bytes_logical_draft =
            blocks_per_chunk * per_block_elems_draft * cache_dtype.size_bytes();

        let kv_mapper = self.kv_mapper.clone().ok_or_else(|| {
            ExecutorError::WorkerInit("draft KV pool before the target's KvMapper".into())
        })?;
        let mut draft_layers: Vec<scratchy_target_metal::sparse_kv::SparseKvLayer> =
            Vec::with_capacity(num_layers_draft * 2);
        for _ in 0..(num_layers_draft * 2) {
            let layer = scratchy_target_metal::sparse_kv::SparseKvLayer::new(
                &mtl_device,
                &residency,
                &kv_mapper,
                chunk_bytes_logical_draft,
                num_chunks_total_draft,
            )
            .map_err(|e| ExecutorError::WorkerInit(format!("draft SparseKvLayer: {e}")))?;
            draft_layers.push(layer);
        }
        info!(
            "ScratchyWorker(metal): draft KV layer buffers — {} buffers",
            draft_layers.len()
        );

        let pool = unsafe {
            // Layer-major iteration: outer for layer, inner for chunk.
            // Draft uses `initial_chunks = num_chunks_total` (eager: the
            // spec-decode draft forward path has no growth hook). The slot
            // formula `(c / (2 * chunks_per_layer)) * 2 + c % 2` matches
            // the call sequence L0_K_c0, L0_V_c0, L0_K_c1, L0_V_c1, ...
            let alloc_chunk_counter = std::cell::Cell::new(0usize);
            let draft_layers_ref = &mut draft_layers;
            let chunks_per_layer = num_chunks_total_draft;
            KvCachePool::new_metal_chunked(
                num_layers_draft,
                draft_blocks,
                self.config.block_size,
                model.num_key_value_heads() as usize,
                model.head_dim() as usize,
                draft_block_cap,
                // Draft pool stays uniform (matching the uniform
                // SparseKvLayer sizing above): spec-decode is not
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
                    draft_layers_ref[slot]
                        .commit_chunk((c / 2) % chunks_per_layer, bytes)
                        .map_err(|e| anyhow::anyhow!("draft KV commit slot={slot}: {e}"))
                },
                |bytes| Ok(MetalMem::new_pinned(&mtl_device, &residency, bytes)),
            )
        }
        .map_err(|e| ExecutorError::WorkerInit(format!("draft KvCachePool: {e}")))?;
        self.draft_kv_layers = draft_layers;
        kv_mapper.wait();
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
        // vocab_size). Take it from the model so the sampler params assemble
        // BEFORE this step's forward; the logits BUFFER itself is bound later,
        // in the forward's followup (it's the forward's own output).
        let vocab = {
            let model = self.model.as_deref().ok_or_else(|| {
                ExecutorError::WorkerExecution("gpu sampler: model not loaded".into())
            })?;
            model.vocab_size() as u32
        };
        let arena = self
            .sampler_arena
            .as_ref()
            .ok_or_else(|| ExecutorError::WorkerExecution("gpu sampler: arena not built".into()))?;

        // Assemble this step's sampler inputs (metal's `GpuSampleParams` layout)
        // and hand them to the metal sampler. The per-request seed inside comes
        // from the shared `scratchy_core_common::fnv_seed`, so it matches cuda;
        // the format-assembly + all MTL4 detail live in `scratchy_target_metal`.
        let params = scratchy_target_metal::sampling::gather_gpu_sample_params(
            jobs,
            req_ids,
            &self.sampling_params_map,
            &mut self.seeded_rngs,
            |id| self.input_batch.history(id),
            |id| self.input_batch.num_generated(id),
            vocab,
        );
        let record = self.deferral.as_mut().map(|d| &mut d.host_writes);
        Ok(arena.prepare_step(&params, jobs.len() as u32, record))
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
fn metal_chain_dispatch(
    model_ref: &dyn scratchy_forward_compiler::ScratchyWeights,
    kv_cache_ref: &KvCachePool,
    device_mut: &mut GpuDevice,
    argmax_kernels: &scratchy_target_metal::argmax::ArgmaxKernels,
    chain_kernel: &scratchy_target_metal::chain_advance::ChainAdvanceKernel,
    req: &::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest<'_>,
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
    // Pinned for the duration of this call, like the constants below.
    let residency = device_mut.allocator.residency().clone();
    let pin_zeroed = |bytes| {
        residency.pin(scratchy_target_metal::mtl4_dispatch::shared_zeroed(
            &mtl_device,
            bytes,
        ))
    };
    let argmax_bytes = (req.num_tokens.max(1)) * 4;
    let pinned_argmax: Vec<_> = (0..k).map(|_| pin_zeroed(argmax_bytes)).collect();
    let argmax_bufs: Vec<scratchy_target_metal::mtl4_dispatch::Buffer> =
        pinned_argmax.iter().map(|b| (**b).clone()).collect();

    // ── 3. Pack constants ───────────────────────────────────────
    let pinned_consts = pin_zeroed(8);
    residency.commit();
    let consts_buf = (*pinned_consts).clone();
    unsafe {
        let p = consts_buf.contents().as_ptr() as *mut u32;
        *p.add(0) = req.num_tokens as u32;
        *p.add(1) = req.block_table_stride as u32;
    }

    // ── 4. Build K argmax_dual_write argument tables ────────────
    let argmax_arg_tables: Vec<_> = (0..k)
        .map(|_| {
            use ::objc2_metal::MTL4ArgumentTableDescriptor;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(4);
            mtl_device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("argmax dual_write arg_table alloc returned nil")
        })
        .collect();

    // ── 5. Build chain_advance argument table ───────────────────
    let chain_arg_table = {
        use ::objc2_metal::MTL4ArgumentTableDescriptor;
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(6);
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
        target_hidden: None,
        hidden_out: None,
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
        last_token_indices: None,
        deferred: None,
    };

    // SAFETY: kernel refs survive the synchronous call below.
    let argmax_kernels_addr: usize =
        argmax_kernels as *const scratchy_target_metal::argmax::ArgmaxKernels as usize;
    let chain_kernel_addr: usize =
        chain_kernel as *const scratchy_target_metal::chain_advance::ChainAdvanceKernel as usize;

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
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr + 4, 4);
                chain_arg_table_for_closure.setAddress_atIndex(consts_addr, 5);
            }

            for iter in 0..k_usize {
                handle.run_forward_step(enc, num_tokens_u32, num_reqs_u32, false)?;

                let table = &argmax_arg_tables_for_closure[iter];
                let out_addr = argmax_bufs_for_closure[iter].gpuAddress();
                unsafe {
                    table.setAddress_atIndex(logits_addr, 0);
                    table.setAddress_atIndex(out_addr, 1);
                    table.setAddress_atIndex(consts_addr, 2);
                    table.setAddress_atIndex(next_in_addr, 3);
                }
                scratchy_target_metal::argmax::encode_argmax_dual_write_into_mtl4(
                    argmax_kernels_ref,
                    enc,
                    table,
                    num_reqs_u32,
                )
                .map_err(|e| format!("encode argmax_dual_write: {e:?}"))?;

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
        self.forward_argmax_hidden_blocking(model, kv_pool, req, &[])
            .map(|(argmax, _)| argmax)
    }

    /// The forward behind [`Self::forward_argmax_blocking`]; `hidden_rows` empty copies no
    /// hidden rows out.
    fn forward_argmax_hidden_blocking(
        &mut self,
        model: ::scratchy_serving_engine::spec_decode::ModelHandle,
        kv_pool: ::scratchy_serving_engine::spec_decode::KvPoolHandle,
        req: &::scratchy_serving_engine::spec_decode::ForwardArgmaxRequest<'_>,
        hidden_rows: &[u32],
    ) -> Result<(Vec<u32>, Vec<u8>), ::scratchy_serving_engine::spec_decode::BackendError> {
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
        // later `self.gpu_device.as_mut()` borrow) so the argmax and grammar-mask
        // buffers can be pinned without extending `device_buf`'s borrow.
        let residency = device_buf.allocator.residency().clone();

        let argmax_kernels = match model {
            ModelHandle::TARGET => &self.argmax_kernels,
            _ => &self.draft_argmax_kernels,
        }
        .as_ref()
        .ok_or_else(|| BackendError::Backend("argmax_kernels not built".into()))?;

        // ── 1. Upload host slices to fresh shared-storage MTLBuffers ─────
        let buf_input_ids = Self::alloc_shared_u32_buf(&mtl_device, req.input_ids);
        let buf_positions = Self::alloc_shared_u32_buf(&mtl_device, req.positions);
        let buf_slot_mapping = Self::alloc_shared_u32_buf(&mtl_device, req.slot_mapping);
        let buf_cu_seqlens = Self::alloc_shared_u32_buf(&mtl_device, req.cu_seqlens_q);
        let buf_seqused_k = Self::alloc_shared_u32_buf(&mtl_device, req.seqused_k);
        let buf_block_table = Self::alloc_shared_u32_buf(&mtl_device, req.block_table);
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
        // Phase 8 routing: a draft forward (ModelHandle(1)) runs on a
        // shadow `GpuDevice` (`draft_device`): the same `MTLDevice` +
        // allocator, on the draft queue when one exists — so a draft
        // model's pool attaches its residency set to that queue and its
        // work runs concurrently with the target's — and, for an MTP
        // head, the target's buffers to place its pool's in. The shadow
        // device is local to this call; the worker's `gpu_device` is
        // untouched.
        let mut shadow_draft_device: Option<GpuDevice> = match model {
            ModelHandle(1) => Some(self.draft_device()?),
            _ => None,
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

        // A step `execute_model` defers: the target forward is committed
        // without waiting for it.
        let mut deferral = if matches!(model, ModelHandle::TARGET) {
            self.deferral.take()
        } else {
            None
        };
        // An MTP head's input hidden rows (host bytes; the forward copies them into its runtime
        // buffer) and the rows of this forward's own final hidden states the caller wants back.
        let view_target_hidden = req.target_hidden.map(|bytes| unsafe {
            TensorView::from_raw(GpuTensor::new(
                bytes.as_ptr() as *mut u8,
                &[bytes.len()],
                scratchy_target_metal::dtype::DType::U8,
            ))
        });
        let hidden_capture = std::cell::RefCell::new(Vec::new());
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
            target_hidden: view_target_hidden,
            hidden_out: (!hidden_rows.is_empty()).then_some(scratchy_target_metal::HiddenRowsOut {
                rows: hidden_rows,
                out: &hidden_capture,
            }),
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
            last_token_indices: view_last_token_indices,
            deferred: deferral.as_ref(),
        };

        // ── 4. Fused argmax via forward_with_metal_followup. ─────────────
        // Phase 6a: argmax encodes onto the SAME MTL4 compute encoder
        // as the forward — forward + argmax share one CB, one commit,
        // one host wait. Pre-6a took the unfused path (separate dispatch
        // + commit + wait + readback) for 5.2a simplicity; this re-folds
        // it. Per call: 2 commit+waits → 1.
        let argmax = free_argmax_slot(
            &mut self.argmax_slots,
            &mtl_device,
            &residency,
            req.num_tokens,
        );
        let argmax_out = (*argmax.out).clone();
        let consts_buf = (*argmax.consts).clone();
        let arg_table = {
            use ::objc2_metal::MTL4ArgumentTableDescriptor;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(3);
            mtl_device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("argmax arg_table alloc returned nil")
        };
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
        // its pinned buffers. The closure binds these by gpuAddress and runs
        // the mask on the forward encoder before argmax.
        #[cfg(feature = "guided-decoding")]
        let grammar_pending = self.grammar_pending.take();
        #[cfg(feature = "guided-decoding")]
        let grammar_mask_ctx: Option<GrammarMaskGpu> = if let (Some(h), Some(kernels)) =
            (grammar_pending, self.grammar_mask_kernels.as_ref())
        {
            use ::objc2_metal::{MTLBuffer, MTLDevice};
            let kernels_addr =
                kernels as *const scratchy_target_metal::grammar_mask::GrammarMaskKernels as usize;
            let grew = reserve_pinned(
                &mut self.grammar_buf_allow,
                &mtl_device,
                &residency,
                h.allow_bits.len().max(1) * 4,
            ) | reserve_pinned(
                &mut self.grammar_buf_rows,
                &mtl_device,
                &residency,
                h.rows.len().max(1) * 4,
            );
            if grew {
                residency.commit();
            }
            let allow = self.grammar_buf_allow.as_ref().unwrap();
            let rows = self.grammar_buf_rows.as_ref().unwrap();
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
            }
            let arg_table = {
                use ::objc2_metal::MTL4ArgumentTableDescriptor;
                let desc = MTL4ArgumentTableDescriptor::new();
                desc.setMaxBufferBindCount(3);
                mtl_device
                    .newArgumentTableWithDescriptor_error(&desc)
                    .expect("grammar_mask arg_table alloc returned nil")
            };
            Some(GrammarMaskGpu {
                allow_bits: (**allow).clone(),
                rows: (**rows).clone(),
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
        // closure only borrows the `PendingSampler`, whose buffers stay pinned
        // while it lives, so it outlives the forward and its host wait below.
        let pending_sampler = self.pending_sampler.take();
        let sampler = pending_sampler.as_ref();
        #[cfg(feature = "sampler-telemetry")]
        let sampler_telem = sampler.and_then(|p| p.telemetry_output());
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
                }
                use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};
                let logits_addr = logits_buf.gpuAddress();
                let out_addr = argmax_out_for_closure.gpuAddress();
                let consts_addr = consts_for_closure.gpuAddress();
                unsafe {
                    arg_table_for_closure.setAddress_atIndex(logits_addr, 0);
                    arg_table_for_closure.setAddress_atIndex(out_addr, 1);
                    arg_table_for_closure.setAddress_atIndex(consts_addr, 2);
                }
                // ── Grammar mask (constrained / guided decoding) ─────────
                // Runs on THIS encoder before argmax: forces every token
                // disallowed by a request's grammar FSM to -inf so the greedy
                // argmax can only pick a grammar-valid token. The mask emits
                // its own pre-barrier (forward-write -> mask-read); argmax's
                // pre-barrier then orders mask-write -> argmax-read.
                #[cfg(feature = "guided-decoding")]
                if let Some(ref gm) = grammar_mask_ctx {
                    // Mask the FULL logits width. The shader treats any slot
                    // beyond the bitset (the lm_head's vocab padding past the
                    // tokenizer vocab) as disallowed, so padded tokens can never
                    // be sampled and the bits[] read stays in bounds.
                    unsafe {
                        gm.arg_table.setAddress_atIndex(logits_addr, 0);
                        gm.arg_table
                            .setAddress_atIndex(gm.allow_bits.gpuAddress(), 1);
                        gm.arg_table.setAddress_atIndex(gm.rows.gpuAddress(), 2);
                    }
                    let gm_kernels: &scratchy_target_metal::grammar_mask::GrammarMaskKernels = unsafe {
                        &*(gm.kernels_addr
                            as *const scratchy_target_metal::grammar_mask::GrammarMaskKernels)
                    };
                    scratchy_target_metal::grammar_mask::encode_grammar_mask_into_mtl4(
                        gm_kernels,
                        enc,
                        &gm.arg_table,
                        gm.num_rows,
                    )
                    .map_err(|e| format!("encode_grammar_mask: {e:?}"))?;
                }
                let argmax_kernels_ref: &scratchy_target_metal::argmax::ArgmaxKernels = unsafe {
                    &*(argmax_kernels_addr as *const scratchy_target_metal::argmax::ArgmaxKernels)
                };
                scratchy_target_metal::argmax::encode_argmax_into_mtl4(
                    argmax_kernels_ref,
                    enc,
                    &arg_table_for_closure,
                    total_n_actual,
                )
                .map_err(|e| format!("encode_argmax: {e:?}"))?;
                // Fused sampler: encode cast → [penalties] → sample onto THIS
                // encoder, after argmax, reading the (grammar-masked) logits;
                // each sampled token overwrites its row's argmax.
                if let Some(ps) = sampler
                    && let Some(addr) = sampler_kernels_addr
                {
                    let kernels_ref = unsafe {
                        &*(addr as *const scratchy_target_metal::sampling::SamplerKernels)
                    };
                    ps.encode_into(enc, logits_addr, &argmax_out_for_closure, kernels_ref);
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
        if let Some(deferral) = deferral.as_mut() {
            let in_flight = deferral.take_in_flight().ok_or_else(|| {
                BackendError::Backend("a deferred forward returned uncommitted".into())
            })?;
            self.committed = Some((in_flight, argmax));
            return Ok((Vec::new(), Vec::new()));
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
            let sampled_token_id = sampler
                .and_then(|p| p.rows().first())
                .map(|&row| out[row as usize])
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
        Ok((out, hidden_capture.into_inner()))
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
        _block_size: usize,
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

        let argmax_kernels = match model {
            ModelHandle::TARGET => &self.argmax_kernels,
            _ => &self.draft_argmax_kernels,
        }
        .as_ref()
        .ok_or_else(|| BackendError::Backend("argmax_kernels not built".into()))?;
        let chain_kernel = match model {
            ModelHandle::TARGET => &self.chain_advance_kernel,
            _ => &self.draft_chain_advance_kernel,
        }
        .as_ref()
        .ok_or_else(|| BackendError::Backend("chain_advance_kernel not built".into()))?;

        // Phase 8 routing: a draft chain runs on the draft's shadow
        // device (`draft_device`), so target verify and draft chain stay
        // disjoint when there is a draft queue.
        let mut shadow_draft_device: Option<GpuDevice> = match model {
            ModelHandle(1) => Some(self.draft_device()?),
            _ => None,
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

/// One step's greedy argmax: its output, a token per row, and the row count it reads.
#[cfg(feature = "metal")]
struct ArgmaxSlot {
    out: scratchy_target_metal::residency::Pinned,
    consts: scratchy_target_metal::residency::Pinned,
}

// SAFETY: as for `MetalWorker`, the markers objc2's buffers lack. The host reads `out` only once the
// step that writes it is done (`InFlight::wait`), and writes `consts` only while no step reads it.
#[cfg(feature = "metal")]
unsafe impl Send for ArgmaxSlot {}
#[cfg(feature = "metal")]
unsafe impl Sync for ArgmaxSlot {}

/// One of `slots` nothing else holds, with room for `rows` tokens — grown, or added, if none has.
#[cfg(feature = "metal")]
fn free_argmax_slot(
    slots: &mut Vec<Arc<ArgmaxSlot>>,
    device: &scratchy_target_metal::mtl4_dispatch::Device,
    residency: &scratchy_target_metal::residency::MetalResidencySet,
    rows: usize,
) -> Arc<ArgmaxSlot> {
    let bytes = rows.max(1) * size_of::<u32>();
    let free = slots.iter().position(|s| Arc::strong_count(s) == 1);
    if let Some(i) = free.filter(|&i| slots[i].out.length() >= bytes) {
        return Arc::clone(&slots[i]);
    }
    let pin = |bytes| {
        residency.pin(scratchy_target_metal::mtl4_dispatch::shared_zeroed(
            device, bytes,
        ))
    };
    let slot = Arc::new(ArgmaxSlot {
        out: pin(bytes),
        consts: pin(size_of::<u32>()),
    });
    match free {
        Some(i) => slots[i] = Arc::clone(&slot),
        None => slots.push(Arc::clone(&slot)),
    }
    residency.commit();
    slot
}

/// A step committed without waiting for it: the tokens its rows sampled are on the device until it is
/// resolved.
#[cfg(feature = "metal")]
struct InFlightStep {
    done: Arc<InFlight>,
    argmax: Arc<ArgmaxSlot>,
    /// `(request, argmax row)` of each row that samples a token.
    rows: Vec<(String, usize)>,
}

/// The tokens at `rows` of a step's argmax, once its command buffer is done.
#[cfg(feature = "metal")]
fn sampled(
    done: &InFlight,
    argmax: &ArgmaxSlot,
    rows: impl Iterator<Item = usize>,
) -> Result<Vec<u32>, ForwardError> {
    done.wait()?;
    let out: Vec<u32> = scratchy_target_metal::mtl4_dispatch::read_slice(
        &argmax.out,
        argmax.out.length() / size_of::<u32>(),
    );
    Ok(rows.map(|row| out[row]).collect())
}

/// Grow `slot` to a pinned, zeroed buffer of at least `bytes`, replacing (and
/// so unpinning) a smaller one. Returns whether it allocated, i.e. whether
/// `residency` needs a commit before the next command buffer.
#[cfg(all(feature = "metal", feature = "guided-decoding"))]
fn reserve_pinned(
    slot: &mut Option<scratchy_target_metal::residency::Pinned>,
    device: &scratchy_target_metal::mtl4_dispatch::Device,
    residency: &scratchy_target_metal::residency::MetalResidencySet,
    bytes: usize,
) -> bool {
    use ::objc2_metal::MTLBuffer;
    if slot.as_ref().is_some_and(|p| p.length() >= bytes) {
        return false;
    }
    let buffer = scratchy_target_metal::mtl4_dispatch::shared_zeroed(device, bytes);
    *slot = Some(residency.pin(buffer));
    true
}

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
        self.metal_prefill_bucket_max_m
    }

    fn max_num_seqs_override(&self) -> Option<usize> {
        // The width this worker resolved (caller's ask, or the memory-
        // affordable default) and built every pool at. Reported so the
        // scheduler admits exactly this many — one process, one number —
        // including when the flag was UNSET and the worker answered it
        // itself. `None` before `load_model`.
        self.resolved_max_num_seqs
    }

    fn align_max_num_seqs(&mut self, width: usize) {
        // DOWNWARD only, and only when the flag was unset: ranks each clamp
        // on their own `currentAllocatedSize`, so the engine hands back the
        // narrowest width and every rank builds its pools/arenas at it. An
        // explicit ask is identical on every rank and never re-clamped.
        if self.config.max_num_seqs.is_none()
            && let Some(r) = self.resolved_max_num_seqs.as_mut()
            && *r > width
        {
            info!(
                "ScratchyWorker(metal): aligning max_num_seqs {} → {width} (narrowest rank)",
                *r
            );
            *r = width;
        }
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

    fn kv_block_bytes(&self, block_size: usize) -> Option<usize> {
        // A uniform TurboQuant model's blocks are packed codes, and a hybrid
        // model's are group-shared pages plus its TurboQuant global store — not
        // the dense rows the engine would derive. The same cost the draft split
        // in `determine_available_memory` divides by.
        // A Gated-DeltaNet hybrid's dense pool holds its attention layers alone, which the
        // engine's every-layer row would overcount.
        let model = self.model.as_deref()?;
        (uniform_turboquant(model, block_size)
            || hybrid_kv_layout(model, block_size, METAL_KV_ELEM_BYTES).is_some()
            || kv_layer_tensors(model).is_some())
        .then(|| target_block_bytes(model, block_size))
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
        // A configured draft may be a multi-token-prediction head, lent the target's embedding and
        // lm_head: snapshot the refs before the target's load consumes them.
        self.target_tensor_refs = self
            .config
            .draft_model_path
            .is_some()
            .then(|| weights.tensor_refs());
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
        .map_err(|e| ExecutorError::WorkerInit(format!("scratchy-forward-compiler load: {e}")))?;
        // "No compiled variant matches this checkpoint" is a different problem
        // from "this backend has no such arch", and pointed at a different fix
        // (build scope vs a missing backend) — see `ArchLoad`. #202 was the
        // second reported as the first.
        let model = match model {
            scratchy_forward_compiler::ArchLoad::Loaded(m) => m,
            scratchy_forward_compiler::ArchLoad::NoVariantMatched => {
                return Err(ExecutorError::NoVariantMatched(arch.clone()));
            }
            scratchy_forward_compiler::ArchLoad::ArchNotCompiled => {
                return Err(ExecutorError::ArchNotSupported(arch.clone()));
            }
        };

        info!(
            "ScratchyWorker(metal): try_load in {:?} ({} via scratchy-forward-compiler, {})",
            t_try_load.elapsed(),
            arch,
            model.arch_name()
        );

        // Resolve `max_num_seqs` NOW — after the model's GDN config is
        // answerable and weights are resident, before every consumer of the
        // width (sampler arena below, GDN state pool in `initialize_cache`,
        // spec-decode chain bounds).
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

        // Resolve `max_num_seqs` NOW — AFTER the vision-tower probe above
        // (its weights are then resident and counted in
        // `currentAllocatedSize`, so a VL model cannot eat into the KV
        // floor the resolver reserves) and BEFORE every consumer of the
        // width (sampler arena below, GDN state pool in
        // `initialize_cache`, spec-decode chain bounds). The target and the
        // `gpu_device` it needs for rung picking are still locals here
        // (fields assigned at the end of load_model) — pass them in.
        self.resolve_max_num_seqs(model.as_ref(), &gpu_device)?;

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
        let off_tape = scratchy_target_metal::off_tape::OffTapeKernels::of(model.metal_off_tape())
            .map_err(|e| ExecutorError::WorkerInit(format!("off-tape kernels: {e}")))?;
        let argmax =
            scratchy_target_metal::argmax::ArgmaxKernels::new(&gpu_device.device, off_tape)
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
        let sampler = scratchy_target_metal::sampling::SamplerKernels::new(
            &gpu_device.device,
            off_tape.vocab,
            off_tape.sampler(),
        )
        .map_err(|e| ExecutorError::WorkerInit(format!("sampler kernel compile: {e:?}")))?;
        info!(
            "ScratchyWorker(metal): SamplerKernels::new in {:?}",
            t_sampler.elapsed()
        );

        // The sampler's persistent arena: every buffer + argument table,
        // sized by the compile-time facts of this config (vocab, resolved
        // max_num_seqs rows, max_model_len history bound) and pinned into the
        // allocator's residency set so every forward command buffer sees it
        // resident. Per-step sampler work is then a pure host memcpy
        // (`prepare_step`).
        let t_arena = std::time::Instant::now();
        let sampler_arena = scratchy_target_metal::sampling::SamplerArena::new(
            &gpu_device.device,
            gpu_device.allocator.residency(),
            // A verify step samples each sequence's every row: its token and its drafts.
            (self.max_num_seqs_resolved() * (1 + self.config.num_speculative_tokens)) as u32,
            &sampler,
            max_model_len.max(1) as u32,
        );
        info!(
            "ScratchyWorker(metal): sampler arena in {:?}",
            t_arena.elapsed()
        );

        // Phase 6 chain-advance pipeline. Tiny kernel — one-time build
        // alongside argmax so the K-step chain driver never falls into
        // pipeline-compile latency on the first call.
        let t_chain = std::time::Instant::now();
        let chain_advance = scratchy_target_metal::chain_advance::ChainAdvanceKernel::new(
            &gpu_device.device,
            off_tape,
        )
        .map_err(|e| ExecutorError::WorkerInit(format!("chain_advance kernel compile: {e:?}")))?;
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
            let grammar_mask = scratchy_target_metal::grammar_mask::GrammarMaskKernels::new(
                &gpu_device.device,
                &off_tape.grammar_mask,
            )
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
        self.sampler_arena = Some(sampler_arena);
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
        // The KV codec was fixed when the model was built (`KV_CODEC`, from the
        // `turboquant` feature and the model's geometry). A `--kv-cache-dtype`
        // naming one asserts it rather than choosing it.
        let kv_codec = model.kv_codec();
        match (self.config.kv_cache_dtype.as_str(), kv_codec) {
            ("auto", _)
            | ("fp16", scratchy_forward_compiler::KvCodec::Dense)
            | ("turboquant", scratchy_forward_compiler::KvCodec::TurboQuant(_)) => {}
            (asked, built) => {
                return Err(ExecutorError::WorkerInit(format!(
                    "--kv-cache-dtype {asked}: this binary stores the model's KV cache as \
                     {built}, fixed when it was built by the `turboquant` feature and the \
                     model's geometry"
                )));
            }
        }
        info!("ScratchyWorker(metal): KV cache codec: {kv_codec} (built in)");

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

        // Per-sequence block-table width for this pool: the model's KV cap rung for the
        // max_model_len-derived capacity, capped at `num_gpu_blocks` (the blocks actually
        // allocated). Stored on the pool; the host block-table stride (execute_model) reads it
        // back, and the pool's tapes are baked for it.
        let (pool_block_cap, _) =
            self.metal_rung(model.as_ref(), self.kv_block_cap(num_gpu_blocks), device)?;
        info!(
            "ScratchyWorker(metal): KV block-table capacity (max_blocks_per_seq) = {pool_block_cap} \
             (max_model_len-derived, pool {num_gpu_blocks} blocks × {} tokens/block)",
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
        // Reactive (chunked) KV pool: one placement-sparse MTLBuffer per
        // `(tensor, K/V)` spanning the pool's capacity, whose pages are mapped in
        // only as the pool grows (`sparse_kv.rs`): physical memory follows the
        // blocks in use, and the readers still address blocks directly off one
        // base (`attention.metal`'s `ATTN_BLOCKS_PER_CHUNK==0` fast path).
        let kv_mapper = scratchy_target_metal::sparse_kv::KvMapper::new(&mtl_device)
            .map_err(|e| ExecutorError::WorkerInit(format!("KvMapper: {e}")))?;
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

        let hybrid_layout = hybrid_kv_layout(model.as_ref(), self.config.block_size, elem_bytes);
        // One physical tensor per group POSITION (gemma4: group_size=5) on the
        // hybrid path; one per layer otherwise.
        // A Gated-DeltaNet hybrid's attention layers alone hold KV: one tensor each.
        let layer_tensors = (hybrid_layout.as_ref().map(|l| l.layer_to_tensor.clone()))
            .or_else(|| kv_layer_tensors(model.as_ref()));
        let num_tensors_for_pool = hybrid_layout
            .as_ref()
            .map(|l| l.group_size)
            .or_else(|| {
                let tensors = layer_tensors.as_ref()?;
                Some(tensors.iter().copied().max().map_or(0, |t| t + 1))
            })
            .unwrap_or(num_layers_for_pool);
        // The full group's page-unified block size (Gemma4: 32) — the worker
        // encodes the full slot_mapping with this. `config.block_size` (uniform).
        self.kv_full_block_size = hybrid_layout
            .as_ref()
            .map(|l| l.full_block_size())
            .unwrap_or(self.config.block_size);
        self.kv_is_hybrid = hybrid_layout.is_some();

        let mut kv_layers: Vec<scratchy_target_metal::sparse_kv::SparseKvLayer> =
            Vec::with_capacity(num_tensors_for_pool * 2);
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
            // (all chunks committed) → the sliding KV in chunk 1+ reads a stale
            // chunk-address-table entry → garbage. So the hybrid path needs the
            // full capacity; a chunk's pages are only mapped when the pool grows
            // into it, so the SWA win (global packed + sliding fp16-windowed) holds.
            let max_chunks = if kv_codec.is_turboquant() && hybrid_layout.is_none() {
                1
            } else {
                num_chunks_total
            };
            let layer = scratchy_target_metal::sparse_kv::SparseKvLayer::new(
                &mtl_device,
                &residency,
                &kv_mapper,
                slot_chunk_bytes,
                max_chunks,
            )
            .map_err(|e| ExecutorError::WorkerInit(format!("SparseKvLayer: {e}")))?;
            kv_layers.push(layer);
        }
        info!(
            "ScratchyWorker(metal): KV layers — {} tensors × K/V, up to {} chunks of {} bytes \
             each, mapped as the pool grows",
            kv_layers.len() / 2,
            num_chunks_total,
            chunk_bytes_logical,
        );

        let pool = unsafe {
            // Closure-shared call counter — selects which layer buffer this
            // `alloc_chunk` call belongs to via `counter % (2*num_layers)`.
            // Works for both init (`initial_chunks=1` → exactly `2*num_layers`
            // calls, one per slot) and grow (each grow step issues
            // `2*num_layers` calls in slot order, repeated per added chunk).
            let alloc_chunk_counter = std::cell::Cell::new(0usize);
            let kv_layers_ref = &mut kv_layers;
            let n_slots = num_tensors_for_pool * 2;
            KvCachePool::new_metal_chunked(
                num_layers_for_pool,
                num_gpu_blocks,
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
                // vLLM group-shared mapping, or a GDN hybrid's attention layers:
                // layer → physical tensor (position).
                layer_tensors,
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
                    kv_layers_ref[slot]
                        .commit_chunk(c / n_slots, bytes)
                        .map_err(|e| anyhow::anyhow!("KV chunk commit slot={slot}: {e}"))
                },
                // Chunk-address table: `StorageModeShared` so the CPU can
                // write the chunk gpuAddresses (`fill_chunk_tables` below).
                // Tiny — 8 bytes/chunk — so a regular allocation is fine.
                |bytes| Ok(MetalMem::new_pinned(&mtl_device, &residency, bytes)),
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
        self.target_kv_layers = kv_layers;
        // The first chunk's pages are mapped on the mapper's queue; in place before
        // anything reads them.
        kv_mapper.wait();
        self.kv_mapper = Some(kv_mapper);
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

        // Gated-DeltaNet (Qwen3.5 / Qwen3-Next) recurrent-state pool —
        // the non-paged sibling of the KV cache. Only hybrid arches
        // report a `gdn_runtime_config` (macro-emitted from the unrolled
        // IR's per-layer GDN dispatch); every other arch leaves
        // `self.gdn_state` / `self.gdn_slot_allocator` `None`. One
        // recurrent-state slot per concurrently-resident sequence
        // (`max_num_seqs`), matching the scheduler's `max_num_running_reqs`
        // so the slot allocator never exhausts.
        if let Some(gdn_cfg) = model.gdn_runtime_config() {
            let num_slots = self.max_num_seqs_resolved();
            let num_layers = model.num_hidden_layers() as usize;
            let checkpoint_rows = self.gdn_checkpoint_rows()?;
            let t_gdn = std::time::Instant::now();
            let gdn_pool = unsafe {
                scratchy_target_metal::gdn_state::GdnStatePool::new(
                    num_layers,
                    &gdn_cfg.linear_layers,
                    num_slots,
                    checkpoint_rows,
                    gdn_cfg.state_dims(),
                    // f32 conv/ssm state. Every per-layer conv/ssm pointer
                    // derives from the CPU base of this StorageModeShared
                    // buffer (a Private one gave the Qwen3.5-MoE-35B pool wild
                    // addresses on macOS 26.5.1). Pinned in the same set as
                    // the KV pool so the lazy pager can't drop it mid-attention.
                    |bytes| Ok(MetalMem::new_pinned(&mtl_device, &residency, bytes)),
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
            self.gdn_slot_allocator = Some(GdnSlotAllocator::new(num_slots, checkpoint_rows));
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
        let utilization_budget = (working_set as f64 * self.config.gpu_memory_utilization) as usize;
        let mut total = utilization_budget;
        // Reserve room for the Gated-DeltaNet recurrent-state pool. It is
        // built in `initialize_cache` (which runs AFTER this), so it is
        // NOT yet in `currentAllocatedSize`; fold it into the non-KV
        // overhead here so the engine doesn't hand back KV blocks that
        // leave no room for it. Persistent f32 state, one slot per
        // resident seq (and its checkpoints) — sized identically to the pool
        // built later. Zero for non-hybrid arches.
        let checkpoint_rows = self.gdn_checkpoint_rows()?;
        let gdn_reserve = self
            .model
            .as_ref()
            .and_then(|m| m.gdn_runtime_config())
            .map(|cfg| {
                scratchy_target_metal::gdn_state::GdnStatePool::<scratchy_target_metal::PoolMem>::reserve_bytes(
                    cfg.num_linear_layers(),
                    self.max_num_seqs_resolved(),
                    checkpoint_rows,
                    cfg.state_dims(),
                )
            })
            .unwrap_or(0);
        let weights_and_overhead = metal_device
            .device
            .currentAllocatedSize()
            .saturating_add(gdn_reserve);
        // ── The KV floor, as a fixed point with the ladder ────────────────
        // A model whose resident set fits Apple's recommended working set
        // but not `util × working_set` (a 48 GiB checkpoint on a 64 GiB box
        // at the 0.9 default) is KV-starved, not an OOM risk: the
        // utilization default is a HEADROOM REQUEST, not a ceiling on
        // weights. Floor the budget at resident + the shared KV floor —
        // the engine then clamps context to what's left instead of
        // refusing a model the device can hold — and cap it at the working
        // set itself, so a model too big for Apple's own recommendation
        // still trips the guard below unchanged.
        //
        // The floor must run BEFORE the ladder selection under it, not
        // after: the selector prunes the bucket ladder against the budget
        // it is handed, so flooring late left a model that only fits past
        // `util × working_set` collapsing its prefill ladder to the m=1
        // fallback (m=1 prefill steps cost ~15× a decode step — TTFT
        // linear in prompt length on a device with GiB of headroom).
        // But the floor needs the peak-activation estimate, and the
        // estimate internally runs the same selector — resolve the two to
        // a fixed point: each pass floors the budget with the estimate the
        // current budget selects, then re-estimates under the floored
        // budget. Both are monotone in the budget and the result is capped
        // at the working set, so the sequence rises to its fixed point in
        // a few passes — and the FIRST pass breaks out unchanged whenever
        // the floor does not bind (the normal case: `util × working_set`
        // already covers resident + KV floor), costing exactly the one
        // estimate this function always ran.
        // The fixed point prices the KV cap rung a sequence can reach at max_model_len — the
        // largest, so the floor it settles on covers every rung below.
        let full_cap = self.kv_block_cap(usize::MAX);
        let mut peak_activation_estimate =
            self.metal_peak_activation_estimate(None, None, Some(total), full_cap)?;
        for _ in 0..8 {
            let floored = utilization_budget
                .max(
                    weights_and_overhead
                        .saturating_add(peak_activation_estimate)
                        .saturating_add(scratchy_serving_engine::gpu_budget::KV_FLOOR_BYTES),
                )
                .min(working_set);
            if floored <= total {
                break;
            }
            total = floored;
            peak_activation_estimate =
                self.metal_peak_activation_estimate(None, None, Some(total), full_cap)?;
        }
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
        let bucket_costs = match (self.model.as_deref(), self.gpu_device.as_ref()) {
            (Some(target), Some(dev)) => self.metal_bucket_costs(target, dev)?,
            _ => Vec::new(),
        };
        let selection = if bucket_costs.is_empty() {
            None
        } else {
            // `pad` mirrors the non-arena, non-KV terms the KV formula
            // subtracts (64 MiB runtime/staging + 150 MiB redundancy) so the
            // headroom the selector splits equals what is actually left.
            let pad = (64 + 150) * 1024 * 1024u64;
            // `total` is ALREADY the final budget (working_set·util,
            // KV-floored by the fixed point above), so use it directly —
            // do NOT re-apply `utilization` here or it would shrink the
            // budget twice.
            let budget = total as u64;
            let fixed = (weights_and_overhead as u64).saturating_add(pad);
            let sel = select_prefill_bucket(budget, fixed, &bucket_costs, ARENA_FRACTION);
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
        // The selected bucket's price already counts the draft's arena. Without cost tables, take
        // the target's full-ladder peak for the draft too, as a safe upper bound.
        let arena_peak_pair = match (selection.is_some(), self.draft_model.is_some()) {
            (false, true) => arena_peak.saturating_mul(2),
            _ => arena_peak,
        };
        // A uniform TurboQuant pool still reserves one fp16 chunk per layer
        // (`initialize_cache`), whatever its block count.
        let tq_seed = self
            .model
            .as_deref()
            .filter(|m| uniform_turboquant(*m, self.config.block_size))
            .map_or(0, |m| {
                scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize
                    * kv_per_block_bytes(m, self.config.block_size)
            });
        // `total` already folds in `gpu_memory_utilization` (and the KV floor's fixed point
        // above), so pass util=1.0 here — applying it again would shrink the KV budget a second
        // time below the headroom the bucket selector just split.
        let kv_left = |peak: usize| {
            compute_available_kv_bytes(total, weights_and_overhead, peak, 1.0)
                .saturating_sub(tq_seed)
        };
        // ONE record of the peak-activation estimate (`metal_peak_activation_estimate`), at the
        // most blocks a sequence can reach: max_model_len, and no more blocks than the pool holds —
        // those the KV budget buys beside every pool's scratch at its smallest rung (a larger
        // rung's scratch only shrinks it), a block of each pool apiece. A hybrid-geometry pool,
        // which the engine sizes by its own layout, takes the max_model_len bound.
        let bs = self.config.block_size;
        let reach = match self.model.as_deref() {
            Some(target) if target.per_layer_kv_token_elems().is_none() => {
                let draft = self.draft_model.as_deref();
                let pair =
                    target_block_bytes(target, bs) + draft.map_or(0, |d| draft_block_bytes(d, bs));
                kv_left(self.metal_peak_activation_estimate(None, None, Some(total), 1)?)
                    / pair.max(1)
            }
            _ => usize::MAX,
        };
        let reach_cap = self.kv_block_cap(reach);
        let peak_activation_estimate =
            self.metal_peak_activation_estimate(None, None, Some(total), reach_cap)?;
        let rung_scratch = match (self.model.as_deref(), self.gpu_device.as_ref()) {
            (Some(target), Some(dev)) => self.metal_rung_scratch(target, dev, reach_cap)?,
            _ => 0,
        };
        let available = kv_left(peak_activation_estimate);
        // Per-pair split: when a draft model is loaded, every target KV
        // block has a 1:1 mirror in the draft pool, so the engine should
        // think it has only `target / (target + draft)` of the budget.
        // After the engine divides by the target's per-block bytes
        // (`kv_block_bytes`, or its dense row) to land on num_gpu_blocks,
        // the same count of draft blocks fits inside the remaining
        // `draft / (target + draft)` slice.
        let (available_reported, draft_reservation) = if let Some(draft) = self.draft_model.as_ref()
        {
            let target = self.model.as_ref().expect("model loaded before draft");
            let bs = self.config.block_size;
            let t_pb = target_block_bytes(target.as_ref(), bs);
            let d_pb = draft_block_bytes(draft.as_ref(), bs);
            let denom = t_pb.saturating_add(d_pb).max(1);
            // available * t_pb / (t_pb + d_pb), in u128 to dodge overflow.
            let scaled = (available as u128 * t_pb as u128 / denom as u128) as usize;
            (scaled, available.saturating_sub(scaled))
        } else {
            (available, 0)
        };
        info!(
            "ScratchyWorker(metal): total={:.1} GiB, weights+overhead={:.1} GiB, \
             arena_peak={:.1} MiB (pair={:.1} MiB), rung_scratch={:.1} MiB, tq_seed={:.1} MiB, kv_budget={:.1} GiB \
             (target_share={:.1} GiB, draft_reserve={:.1} GiB)",
            total as f64 / 1_073_741_824.0,
            weights_and_overhead as f64 / 1_073_741_824.0,
            arena_peak as f64 / 1_048_576.0,
            arena_peak_pair as f64 / 1_048_576.0,
            rung_scratch as f64 / 1_048_576.0,
            tq_seed as f64 / 1_048_576.0,
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
                // The pool is sized by the RESOLVED width (the caller's ask,
                // or the memory-affordable default an unset flag resolved
                // to). An unset flag was clamped at load by the SAME
                // resolver arithmetic this hint uses (shared
                // `KV_FLOOR_BYTES`, shared terms); if a trip here disagrees
                // anyway, speak about "the default", not a flag nobody
                // passed. (self.config keeps the REQUEST, so it knows.)
                let slots = self.max_num_seqs_resolved();
                let per_slot = gdn_reserve / slots;
                let base = weights_and_overhead
                    .saturating_sub(gdn_reserve)
                    .saturating_add(peak_activation_estimate);
                // The SAME KV floor the load-time resolver subtracts — one
                // home (`gpu_budget::KV_FLOOR_BYTES`), never re-derived.
                let headroom = total
                    .saturating_sub(base)
                    .saturating_sub(scratchy_serving_engine::gpu_budget::KV_FLOOR_BYTES);
                let affordable = (headroom / per_slot.max(1)).max(1);
                if affordable < slots {
                    let sized_by = match self.config.max_num_seqs {
                        Some(_) => format!("--max-num-seqs={slots}"),
                        None => format!("the default max_num_seqs={slots}"),
                    };
                    format!(
                        " The GDN state pool ({:.2} GiB) is sized by \
                         {sized_by}; pass --max-num-seqs {} (or fewer) \
                         to fit this model on this device.",
                        gdn_reserve as f64 / 1_073_741_824.0,
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
        // A preempted request resumes by recomputing from the scheduler's cursor, so its recurrent
        // state must restart from zero like a new request's.
        if let (Some(alloc), Some(preempted)) = (
            self.gdn_slot_allocator.as_mut(),
            &scheduler_output.preempted_req_ids,
        ) {
            for req_id in preempted {
                alloc.release(gdn_slot_key(req_id));
            }
        }
        self.input_batch.update_states(scheduler_output);

        // ── 2. Lifecycle: the per-request state beside the batch ──
        for new_req in &scheduler_output.scheduled_new_reqs {
            let num_tokens = scheduler_output
                .num_scheduled_tokens
                .get(&new_req.req_id)
                .copied()
                .unwrap_or(0);
            if num_tokens == 0 {
                continue;
            }
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
        }

        // ── 3. Steps still on the device ──────────────────────────
        // Opened before anything below can fail: a step that fails leaves its requests open, so a
        // step the engine queued behind it fails here instead of building on tokens never produced.
        self.input_batch
            .open_step(scheduler_output)
            .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;
        // A step that reads the newest one's tokens on the device leaves that
        // one in flight; any other step — and an idle batch, whose KV may
        // shrink — needs every token on the host first.
        let deferrable = self.deferrable(scheduler_output);
        while self.in_flight.len() > usize::from(deferrable) {
            self.resolve_oldest()?;
        }

        #[cfg(feature = "metal")]
        if self.input_batch.num_active() == 0 {
            // Reactive KV (2c): batch fully idle → no live blocks. Defer the
            // shrink until the GPU has been idle for `KV_SHRINK_IDLE_AFTER`.
            // Shrinking on every idle (i.e. after every request) drops the
            // grown chunks' StorageModePrivate pages; the next request then
            // re-faults them on first GPU touch — a multi-second stall on a
            // long prefill. Deferring keeps back-to-back requests fast while
            // still releasing memory once genuinely idle.
            let now = std::time::Instant::now();
            let idle_since = *self.kv_idle_since.get_or_insert(now);
            if now.duration_since(idle_since) >= KV_SHRINK_IDLE_AFTER {
                self.shrink_metal_kv_idle();
            }
        } else {
            // A batch is active again — cancel any pending idle-shrink timer.
            self.kv_idle_since = None;
        }
        // Nothing scheduled — an idle batch, or every live request skipped, preempted or finished
        // this step: nothing runs.
        if scheduler_output.total_num_scheduled_tokens == 0 {
            return Ok(ModelRunnerOutput::empty());
        }

        // ── 4. Prepare flat batch inputs ─────────────────────────
        // Built from the schedule: exactly the scheduled requests, each its
        // scheduled chunk (a verify row is `[last_token, draft_0..K-1]`), so
        // `num_tokens` IS `total_num_scheduled_tokens` — within the budget the
        // engine sized to the largest resident bucket.
        let mut prepared = self.input_batch.prepare_inputs(scheduler_output);
        // A deferred step reads each token still on the device from the step
        // that samples it, the newest one in flight.
        self.deferral = deferrable.then(|| {
            Deferral::new(
                prepared
                    .pending
                    .iter()
                    .map(|input| {
                        let step = self
                            .in_flight
                            .back()
                            .expect("an in-flight token has a step");
                        let (_, row) = step
                            .rows
                            .iter()
                            .find(|(req_id, _)| *req_id == input.req_id)
                            .expect("an in-flight token is its request's row in the newest step");
                        DeviceInput {
                            src: (*step.argmax.out).clone(),
                            offset: row * size_of::<u32>(),
                            flat_index: input.flat_index,
                        }
                    })
                    .collect(),
            )
        });
        assert!(
            deferrable || prepared.pending.is_empty(),
            "a step the host waits for has every token on the host"
        );
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
            self.grow_metal_kv_to_cover(max_block)?;
        }

        // slot_mapping[t] = block_ids[abs_pos / bs] * bs + (abs_pos % bs)
        // u32 under metal — the macro-emitted forward reads this as
        // `&[u32]`. Rope kernel checks `slot_mapping[i] != u32::MAX`.
        let mut slot_mapping = StepSlotMapping::with_capacity(num_tokens);
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
                    slot_mapping.skip();
                    continue;
                }
                // Phase 2: a reused cache-hit block already holds valid
                // (position-independent) K from the request that computed it —
                // skip rewriting its KV (u32::MAX = the kernels' write-skip
                // sentinel). Attention still READS it via block_table (bit 31).
                if reused.is_some_and(|r| r.contains(&block_idx)) {
                    slot_mapping.skip();
                    continue;
                }
                let slot = block_ids[block_idx] * full_block_size + offset;
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
                let unrotated = ann.is_some_and(|a| {
                    a.get(&block_idx)
                        .is_some_and(scratchy_core_common::BlockKind::is_relocatable)
                });
                slot_mapping.write(slot, unrotated);
            }
        }
        let slot_mapping_u32 = slot_mapping.into_slots();

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
                        entry |= scratchy_target_metal::UNROTATED_BLOCK_BIT;
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
            let mut sm = StepSlotMapping::with_capacity(num_tokens);
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
                        sm.write(block_ids[block_idx] * block_size + offset, false);
                    } else {
                        sm.skip();
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
            sliding_slot_mappings_u32.push(sm.into_slots());
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
                    let mut rows: Vec<u32> = Vec::new();
                    let mut allow_bits: Vec<u32> = Vec::new();
                    for (i, req_id) in req_ids_in_order.iter().enumerate() {
                        // Still-prefilling chunk → no token sampled this step.
                        if !prepared.req_inputs[i].emits_token {
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
                        self.grammar_pending = Some(GrammarMaskHost { rows, allow_bits });
                    }
                }
            }
        }

        // GDN per-step state entries + step codes (hybrid arches only), one per batched
        // sequence in the SAME order as `cu_seqlens_q` / `req_ids_in_order`. The metal
        // `forward_argmax_blocking` uploads these into `ForwardCtx::{gdn_state_indices,
        // gdn_is_fresh}`. A request's FIRST forward (including a recycled slot's new owner)
        // starts from zero instead of a finished sequence's stale data (the degeneration
        // guard); a verify step checkpoints its drafts.
        self.gdn_pending = self
            .gdn_slot_allocator
            .as_mut()
            .map(|alloc| {
                alloc.step(
                    req_ids_in_order
                        .iter()
                        .zip(&prepared.req_inputs)
                        .map(|(req_id, input)| (gdn_slot_key(req_id), input.spec_token_ids.len())),
                )
            })
            .transpose()
            .map_err(|e| ExecutorError::WorkerExecution(e.to_string()))?;

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
        // A multi-token-prediction head drafts from this step's hidden rows, so it runs after the
        // target (in the proposer) — it has no lockstep prefill to overlap.
        let draft_is_head = self.draft_is_head();
        let target_hidden_rows: Vec<u32> = match draft_is_head {
            true => (0..num_tokens as u32).collect(),
            false => Vec::new(),
        };
        let mut target_hidden: Vec<u8> = Vec::new();
        let phase8_parallel_lockstep = self.draft_model.is_some()
            && !draft_is_head
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
        self.pending_sampler = None;
        let mut fused_sample_jobs: Vec<(usize, u32)> = Vec::new();
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let req_slice = &prepared.req_inputs[i];
            if !req_slice.emits_token {
                continue; // still-prefilling chunk emits no token
            }
            let greedy = self
                .sampling_params_map
                .get(req_id)
                .is_none_or(SamplingParams::is_greedy);
            if greedy {
                continue;
            }
            // A sampled request's verify step samples every row from the target, and the
            // rejection below keeps its drafts while the sampled tokens equal them: rejection
            // sampling of one-token (greedy) drafts, whose tokens are the target's distribution.
            match req_slice.spec_token_ids.is_empty() {
                true => fused_sample_jobs.push((i, sample_indices[i])),
                false => {
                    let rows = req_slice.token_start..req_slice.token_start + req_slice.token_count;
                    fused_sample_jobs.extend(rows.map(|row| (i, row as u32)));
                }
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
                target_hidden: None,
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
                let spec9_argmax_addr: usize = self.draft_argmax_kernels.as_ref().map_or(0, |k| {
                    k as *const scratchy_target_metal::argmax::ArgmaxKernels as usize
                });
                let spec9_chain_addr: usize =
                    self.draft_chain_advance_kernel.as_ref().map_or(0, |k| {
                        k as *const scratchy_target_metal::chain_advance::ChainAdvanceKernel
                            as usize
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
                            // The draft chain runs under spec-decode: chunked addressing.
                            kv_addressing:
                                scratchy_target_metal::tape::lowered::KvAddressing::Chunked,
                            // It runs beside the target: nothing of the target's is lent.
                            lent: Default::default(),
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
                            target_hidden: None,
                            hidden_out: None,
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
                            last_token_indices: None,
                            deferred: None,
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
                            let mut spec_slot = StepSlotMapping::with_capacity(1);
                            match spec9_block_ids_thread.get(block_idx) {
                                Some(&block) => spec_slot.write(
                                    block as usize * spec9_block_size_thread + offset,
                                    false,
                                ),
                                None => spec_slot.skip(),
                            }
                            let spec_input_ids: Vec<u32> = vec![spec_seeds[0]];
                            let spec_positions: Vec<u32> = vec![chain_pos as u32];
                            let spec_slot: Vec<u32> = spec_slot.into_slots();
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
                                    target_hidden: None,
                                };
                            metal_chain_dispatch(
                                dm,
                                dkv,
                                &mut shadow_device,
                                argmax_kernels_ref,
                                chain_kernel_ref,
                                &spec_req,
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
                let (argmax, hidden) = self
                    .forward_argmax_hidden_blocking(
                        ModelHandle::TARGET,
                        KvPoolHandle::TARGET,
                        &req,
                        &target_hidden_rows,
                    )
                    .map_err(|e| {
                        ExecutorError::WorkerExecution(format!("spec verify forward: {e}"))
                    })?;
                target_hidden = hidden;
                argmax
            }
        };
        // A deferred step: its tokens stay on the device until the engine, and
        // the next step but one, wait for them.
        if let Some((done, argmax)) = self.committed.take() {
            self.sampler_logits = None;
            let mut rows = Vec::with_capacity(num_reqs);
            for (i, req_id) in req_ids_in_order.iter().enumerate() {
                if prepared.req_inputs[i].emits_token {
                    self.input_batch.commit_in_flight(req_id, q_lens[i]);
                    rows.push((req_id.clone(), sample_indices[i] as usize));
                } else {
                    self.input_batch.commit_step(req_id, &[], q_lens[i], false);
                }
            }
            let step = InFlightStep {
                done: Arc::new(done),
                argmax,
                rows,
            };
            let (done, argmax) = (Arc::clone(&step.done), Arc::clone(&step.argmax));
            let rows: Vec<usize> = step.rows.iter().map(|&(_, row)| row).collect();
            let output = ModelRunnerOutput::deferred(
                step.rows.iter().map(|(req_id, _)| req_id.clone()).collect(),
                Box::new(move || {
                    sampled(&done, &argmax, rows.into_iter()).map_err(|e| e.to_string())
                }),
            );
            self.in_flight.push_back(step);
            return Ok(output);
        }
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
        // CB, which copied their tokens over the argmax at their rows.
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let req_slice = &prepared.req_inputs[i];
            req_id_to_index.insert(req_id.clone(), i);
            // Intermediate chunk: the chunk stops short of the end of the
            // request's history (a prefill chunk, or a resume re-prefilling
            // what it had generated — `emits_token` false). Such a chunk
            // writes its K/V into the cache but must NOT emit a sampled token
            // — only a chunk that reaches the end of the history produces
            // output. This matches Python vLLM, which excludes still-
            // prefilling requests from the logits/sample set.
            //
            // Without this gate every non-final chunk emits one spurious
            // token: the greedy argmax at the chunk's last prompt position
            // (e.g. the model's continuation of the boilerplate). The engine
            // (`update_from_output`) appends whatever the worker returns, so
            // that token is prepended before the real first generated token
            // — the "doubled first token" / prompt-echo symptom of chunked
            // prefill. `commit_step` still advances `tokens_in_pool` by the
            // chunk's `q_len` (it keys on `input_token_count`, not the token
            // slice), so an empty emit leaves prefill bookkeeping correct.
            if !req_slice.emits_token {
                sampled_token_ids.push(Vec::new());
                was_spec_decode.push(false);
                continue;
            }
            if req_slice.spec_token_ids.is_empty() {
                let row = sample_indices[i] as usize;
                debug_assert!(row < total_n as usize);
                // The row's token: its sampled token for a non-greedy request,
                // the GPU argmax otherwise (or if the fused sampler was skipped).
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
                if let Some(alloc) = self.gdn_slot_allocator.as_mut() {
                    alloc.verified(
                        gdn_slot_key(req_id),
                        rejection.num_accepted_drafts,
                        req_slice.spec_token_ids.len(),
                    );
                }
                sampled_token_ids.push(rejection.accepted_tokens);
                was_spec_decode.push(true);
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
                    target_hidden,
                })
            } else {
                None
            };

        // ── 7. Commit per-request state ─────────────────────────
        // Appends this step's emitted tokens to the history the next step's
        // chunk and the on-GPU sampler's penalties read; appending AFTER
        // sampling means a step's penalty is based on tokens generated
        // strictly before it (standard vLLM behavior).
        for (i, req_id) in req_ids_in_order.iter().enumerate() {
            let q_len = q_lens[i];
            self.input_batch
                .commit_step(req_id, &sampled_token_ids[i], q_len, was_spec_decode[i]);
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
        // A step committed without waiting for it finishes before what it reads
        // is released.
        for step in self.in_flight.drain(..) {
            if let Err(e) = step.done.wait() {
                tracing::warn!("deferred forward failed at shutdown: {e}");
            }
        }
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
        self.argmax_slots.clear();
        #[cfg(feature = "guided-decoding")]
        {
            self.grammar_mask_kernels = None;
            self.grammar_pending = None;
            self.grammar_buf_allow = None;
            self.grammar_buf_rows = None;
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

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::StepSlotMapping;

    /// A slot two sequences write in one step keeps its first writer; the later
    /// write becomes the skip sentinel. A span (unrotated) write of the same
    /// slot counts as the same slot.
    #[test]
    fn a_slot_written_twice_in_a_step_keeps_its_first_writer() {
        let mut step = StepSlotMapping::with_capacity(9);
        for (slot, unrotated) in [(64, false), (65, false)] {
            step.write(slot, unrotated);
        }
        step.skip();
        for (slot, unrotated) in [(64, false), (65, false), (66, true)] {
            step.write(slot, unrotated);
        }
        step.skip();
        for (slot, unrotated) in [(66, false), (67, false)] {
            step.write(slot, unrotated);
        }
        let (skip, span) = (u32::MAX, scratchy_target_metal::UNROTATED_BLOCK_BIT);
        assert_eq!(
            step.into_slots(),
            [64, 65, skip, skip, skip, 66 | span, skip, skip, 67]
        );
    }
}
