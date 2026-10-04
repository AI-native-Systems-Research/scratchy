// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! On-GPU token sampler dispatcher — the Metal port of cuda's
//! `sampling_kernels.cu` sampler (see `shaders/sampling.metal`), sliced across
//! the GPU's cores.
//!
//! The first revision of this port (like its cuda source) ran one 256-thread
//! threadgroup per request row, which at chat batch sizes serialized every
//! full-vocab pass through ONE core: 4.2 ms per sampled token at a 262k vocab
//! (see `tests/sampling_bench.rs`). The kernels are now a pipeline of sliced
//! passes (cast, penalties, softmax, byte-histogram descent, compaction) plus
//! tiny one-threadgroup-per-row decision kernels, all sharing ONE argument
//! table so the whole pipeline rides the forward's command buffer with Device
//! barriers between the dependent dispatches, with [`encode_into`] encoding the
//! pipeline onto the forward's own encoder. Each stage is baked per model with
//! its logits width compiled in ([`crate::off_tape`]).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer as _, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::mtl4_dispatch::{Buffer, shared_zeroed};
use crate::off_tape::OffTapePipeline;
use crate::residency::{MetalResidencySet, Pinned};
use crate::specialized_pipeline_cache::PipelineKey;
use crate::stream::MetalStreamError;
use crate::tape::ids::LogitsWidth;
use crate::tape::kernel_constants::SamplerConstants;
use crate::tape::lowered::BakedKernel;

pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Threads per threadgroup — MUST equal `SAMPLING_BLOCK_SIZE` in
/// `shaders/sampling.metal` (the kernels stride the vocab axis by exactly this
/// and size their `warp_buf` for `SAMPLING_BLOCK_SIZE / 32` warps).
pub const SAMPLER_TG_SIZE: usize = 256;

/// Simdgroup width the block reductions assume — MUST equal `WARP_SIZE` in
/// `shaders/sampling.metal`. The reductions derive `simdgroup = tid / 32`,
/// `lane = tid % 32`, shuffle across 32 lanes, and size `warp_buf` for
/// `SAMPLER_TG_SIZE / 32` slots. Every shipping Apple GPU is 32-wide, but the
/// width is a device property (not a compile-time constant), so
/// [`SamplerKernels::new`] hard-fails the load on any device that disagrees
/// rather than let the reductions silently mis-index and sample wrong tokens.
pub const SAMPLER_WARP_SIZE: usize = 32;

/// The dtype of the logits rows the cast kernel gathers: the model's compute
/// dtype in production, or [`CastDtype::F32`] to feed host f32 data through
/// the same pipeline (the parity harness).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastDtype {
    F16,
    Bf16,
    F32,
}

/// A stage of the sampler pipeline (see `shaders/sampling.metal` for the
/// dispatch order and binding table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerStage {
    Cast,
    Penalties,
    SoftmaxReduce,
    StatsPick,
    SoftmaxMaterialize,
    Histogram,
    ThresholdPick,
    CountCompact,
    QuotaPick,
    CompactTied,
    Finalize,
}

impl SamplerStage {
    pub const COUNT: usize = 11;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Cast,
        Self::Penalties,
        Self::SoftmaxReduce,
        Self::StatsPick,
        Self::SoftmaxMaterialize,
        Self::Histogram,
        Self::ThresholdPick,
        Self::CountCompact,
        Self::QuotaPick,
        Self::CompactTied,
        Self::Finalize,
    ];

    fn function(self, cast: CastDtype) -> &'static str {
        match (self, cast) {
            (Self::Cast, CastDtype::F16) => "cast_rows_f16_to_f32",
            (Self::Cast, CastDtype::Bf16) => "cast_rows_bf16_to_f32",
            (Self::Cast, CastDtype::F32) => "cast_rows_f32_to_f32",
            (Self::Penalties, _) => "apply_penalties",
            (Self::SoftmaxReduce, _) => "sample_softmax_reduce",
            (Self::StatsPick, _) => "sample_stats_pick",
            (Self::SoftmaxMaterialize, _) => "sample_softmax_materialize",
            (Self::Histogram, _) => "sample_histogram_pass",
            (Self::ThresholdPick, _) => "sample_threshold_pick",
            (Self::CountCompact, _) => "sample_count_compact",
            (Self::QuotaPick, _) => "sample_quota_pick",
            (Self::CompactTied, _) => "sample_compact_tied",
            (Self::Finalize, _) => "sample_finalize",
        }
    }

    /// The key this stage's kernel is baked under for logits `vocab` wide,
    /// cast from `cast`; the stages that spill telemetry carry `telemetry`.
    pub fn key(self, vocab: LogitsWidth, cast: CastDtype, telemetry: bool) -> PipelineKey {
        let spills = matches!(self, Self::SoftmaxMaterialize | Self::Finalize);
        let telemetry = spills.then_some(telemetry);
        let constants = SamplerConstants { vocab, telemetry };
        PipelineKey::new("sampling", self.function(cast), constants.into())
    }
}

/// The sampler pipeline's kernels, built once per worker at model load (like
/// `ArgmaxKernels`).
pub struct SamplerKernels {
    vocab: LogitsWidth,
    slicing: SliceTarget,
    /// One per [`SamplerStage`], in its order.
    stages: Vec<OffTapePipeline>,
}

impl SamplerKernels {
    /// `stages`: a model's [`OffTapeKernels::sampler`](crate::off_tape::OffTapeKernels::sampler),
    /// baked for logits `vocab` wide.
    pub fn new(
        device: &Device,
        vocab: LogitsWidth,
        stages: &[BakedKernel; SamplerStage::COUNT],
    ) -> Result<Self, MetalStreamError> {
        let slicing = SliceTarget::of(device)?;
        let stages = stages
            .iter()
            .map(|k| OffTapePipeline::new(device, k))
            .collect::<Result<Vec<_>, _>>()?;
        let kernels = Self {
            vocab,
            slicing,
            stages,
        };

        // Fence the one runtime assumption the shaders cannot check themselves:
        // the block reductions require a 32-lane simdgroup (see
        // `SAMPLER_WARP_SIZE`). This is true on every shipping Apple GPU, but a
        // device could in principle report a different execution width, which
        // would make `warp_buf` indexing / the shuffle reductions wrong. Refuse
        // to load loudly instead of silently sampling wrong tokens.
        let width = kernels
            .stage(SamplerStage::SoftmaxReduce)
            .threadExecutionWidth();
        if width != SAMPLER_WARP_SIZE {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "the sampler's block reductions require a {SAMPLER_WARP_SIZE}-lane simdgroup, \
                 but this device reports execution width {width}; they would mis-index. \
                 Refusing to load."
            )));
        }
        Ok(kernels)
    }

    fn stage(&self, stage: SamplerStage) -> &ComputePipelineState {
        &self.stages[stage as usize]
    }
}

fn tg(n: u32) -> MTLSize {
    MTLSize {
        width: n as usize,
        height: 1,
        depth: 1,
    }
}

/// Encode one sampler pipeline stage onto an EXISTING MTL4 compute encoder —
/// the forward's own encoder — so the sampler rides the forward's command
/// buffer (one commit, one host wait) instead of a second
/// [`Mtl4DispatchBatch`]. Mirrors `argmax::encode_argmax_*_into_mtl4`.
///
/// Emits a `Device`-visibility barrier first: every stage reads what a prior
/// same-encoder dispatch wrote (cast reads the forward/grammar-mask logits;
/// every later stage reads the previous stage's scratch) and MTL4 compute
/// encoders do NOT auto-serialize same-encoder dispatches. Then
/// set-pipeline / set-arg-table / dispatch.
pub fn encode_sampler_stage_into_mtl4(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    pipeline: &ComputePipelineState,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    threadgroups: u32,
) {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    encoder.setComputePipelineState(pipeline);
    encoder.setArgumentTable(Some(arg_table));
    encoder
        .dispatchThreadgroups_threadsPerThreadgroup(tg(threadgroups), tg(SAMPLER_TG_SIZE as u32));
}

/// Assemble one step's non-greedy sampler inputs into the [`GpuSampleParams`]
/// layout the metal sampler ([`PendingSampler::prepare`]) consumes — sampling
/// params + per-request uniforms + (optional) penalty token histories for the
/// rows in `jobs` = `[(result_index, logits_row)]`. cuda packs its own
/// contiguous H2D layout in `cuda_worker`, so this FORMAT is metal-only (which is
/// why this lives in the metal crate).
///
/// The per-request SEED is the one thing that must NOT differ between backends,
/// so it comes from the shared [`fnv_seed`] / [`seed_to_uniform`]: a seeded
/// request draws from its `StdRng`, an unseeded one hashes
/// `(req_id, generated-token-count)` — `generated`, which counts the tokens
/// still on the device too. `cuda_worker` derives its seed the same way, so an
/// unseeded request gets the identical uniform on either backend.
///
/// [`GpuSampleParams`]: scratchy_core_common::GpuSampleParams
/// [`fnv_seed`]: scratchy_core_common::fnv_seed
/// [`seed_to_uniform`]: scratchy_core_common::seed_to_uniform
pub fn gather_gpu_sample_params<'h>(
    jobs: &[(usize, u32)],
    req_ids: &[String],
    sampling_params_map: &std::collections::HashMap<String, scratchy_core_common::SamplingParams>,
    seeded_rngs: &mut std::collections::HashMap<String, rand::rngs::StdRng>,
    history: impl Fn(&str) -> (&'h [u32], &'h [u32]),
    generated: impl Fn(&str) -> usize,
    vocab: u32,
) -> scratchy_core_common::GpuSampleParams {
    use rand::Rng;

    let njobs = jobs.len();
    let mut params = scratchy_core_common::GpuSampleParams {
        row_indices: Vec::with_capacity(njobs),
        temperatures: Vec::with_capacity(njobs),
        top_ks: Vec::with_capacity(njobs),
        top_ps: Vec::with_capacity(njobs),
        min_ps: Vec::with_capacity(njobs),
        uniforms: Vec::with_capacity(njobs),
        rep_penalties: Vec::with_capacity(njobs),
        freq_penalties: Vec::with_capacity(njobs),
        pres_penalties: Vec::with_capacity(njobs),
        ..Default::default()
    };

    for &(i, row) in jobs {
        let req_id = &req_ids[i];
        let (t, k, tp, mp, rep, freq, pres) = sampling_params_map.get(req_id).map_or(
            (1.0f32, 0i32, 1.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32),
            |p| {
                (
                    p.temperature.max(1e-7) as f32,
                    p.top_k,
                    p.top_p as f32,
                    p.min_p as f32,
                    p.repetition_penalty as f32,
                    p.frequency_penalty as f32,
                    p.presence_penalty as f32,
                )
            },
        );
        params.row_indices.push(row);
        params.temperatures.push(t);
        params.top_ks.push(k);
        params.top_ps.push(tp);
        params.min_ps.push(mp);
        params.rep_penalties.push(rep);
        params.freq_penalties.push(freq);
        params.pres_penalties.push(pres);
        if (rep - 1.0).abs() > f32::EPSILON || freq != 0.0 || pres != 0.0 {
            params.any_penalty = true;
        }
        let seed = if let Some(rng) = seeded_rngs.get_mut(req_id) {
            rng.random::<u32>()
        } else {
            // Generated-token count = the request's decode position; advances
            // each step so the seed varies. Shared with cuda_worker.
            let position = generated(req_id) as u32;
            scratchy_core_common::fnv_seed(req_id, position)
        };
        params
            .uniforms
            .push(scratchy_core_common::seed_to_uniform(seed));
    }

    // Penalty token histories (only when a row uses penalties): the request's
    // prompt and generated tokens; both arrays are row-major `njobs * max_*`
    // padded with `vocab`.
    if params.any_penalty {
        let mut outs: Vec<Vec<i32>> = Vec::with_capacity(njobs);
        let mut prompts: Vec<Vec<i32>> = Vec::with_capacity(njobs);
        for &(i, _) in jobs {
            let req_id = &req_ids[i];
            let (prompt, generated) = history(req_id);
            prompts.push(prompt.iter().map(|&t| t as i32).collect());
            outs.push(generated.iter().map(|&t| t as i32).collect());
        }
        let max_out = outs.iter().map(Vec::len).max().unwrap_or(0);
        let max_prompt = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let pad = vocab as i32;
        let mut flat_out: Vec<i32> = vec![pad; njobs * max_out];
        let mut flat_prompt: Vec<i32> = vec![pad; njobs * max_prompt];
        for (r, v) in outs.iter().enumerate() {
            flat_out[r * max_out..r * max_out + v.len()].copy_from_slice(v);
        }
        for (r, v) in prompts.iter().enumerate() {
            flat_prompt[r * max_prompt..r * max_prompt + v.len()].copy_from_slice(v);
        }
        params.output_token_ids = flat_out;
        params.prompt_token_ids = flat_prompt;
        params.max_output_len = max_out as u32;
        params.max_prompt_len = max_prompt as u32;
    }

    params
}

/// Number of top candidates the sampler spills for the live "soul" panel when
/// [`SamplerTelemetry`](scratchy_core_common::sampler_telemetry::SamplerTelemetry)
/// is enabled.
#[cfg(feature = "sampler-telemetry")]
const SAMPLER_TELEM_K: u32 = 8;

/// `row_state` word count per row — MUST equal `ROW_STATE_LEN` in
/// `shaders/sampling.metal`.
const ROW_STATE_LEN: usize = 16;

/// The sampler's persistent GPU state: every buffer and argument table the
/// pipeline needs, allocated and bound ONCE (at model load, like
/// `RuntimeBindings` — vocab, max rows and history bounds are compile-time
/// facts of the loaded config, so per-step allocation was pure waste).
/// Buffers live in the worker's own (already-committed) residency set, so
/// every forward command buffer sees them resident with no per-step pinning.
///
/// Per step, [`prepare_step`](Self::prepare_step) only writes this step's
/// params/row-state words into the shared buffers (host memcpy — the buffers
/// are `StorageModeShared`) and returns a lightweight [`PendingSampler`]
/// handle; the descent state itself is reset by `sample_stats_pick` each
/// step, so no zeroing pass is needed between steps.
pub struct SamplerArena {
    max_rows: u32,
    vocab: u32,
    slicing: SliceTarget,
    /// The largest `nrows * nslices` any step can reach (see
    /// [`SliceTarget::sliced_max`]); sliced buffers are indexed `[row * nslices +
    /// slice]` with the CURRENT step's nslices, so they must cover this.
    sliced_max: usize,
    /// Persistent pins: dropped only when the arena drops (worker teardown).
    _pins: Vec<Pinned>,
    scratch_f32: Buffer, // f32 logits → prob bits, [max_rows, vocab]
    out_buf: Buffer,     // sampled token ids, [max_rows]
    row_idx_buf: Buffer, // [max_rows]
    out_ids_buf: Buffer, // penalties histories, [max_rows, max_hist]
    prompt_ids_buf: Buffer,
    reps_buf: Buffer, // [max_rows] f32 each
    freqs_buf: Buffer,
    press_buf: Buffer,
    row_state_buf: Buffer, // [max_rows, ROW_STATE_LEN]
    partials_buf: Buffer,  // [sliced_max, 3]
    hist_buf: Buffer,      // [sliced_max, 256]
    counts_buf: Buffer,    // [sliced_max, 4]
    staging_buf: Buffer,   // [sliced_max, 2 * MAX_CANDIDATES]
    consts_buf: Buffer,    // (nslices, nrows, max_out, max_prompt)
    max_hist: u32,
    // Sampler-telemetry spill (only compiled under `sampler-telemetry`).
    #[cfg(feature = "sampler-telemetry")]
    topk_probs_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    topk_indices_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    stats_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    telem_consts_buf: Buffer,
    // Per-stage argument tables: MTL4 binds buffer attributes by signature
    // position; each kernel's buffers sit at contiguous 0..k-1, so each stage
    // has its own table. Built + fully bound once here; the cast's slot 1
    // (the forward's logits) is rebound per forward by
    // [`PendingSampler::encode_into`].
    cast_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    penalties_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    softmax_reduce_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    stats_pick_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    softmax_materialize_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    histogram_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    threshold_pick_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    count_compact_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    quota_pick_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    compact_tied_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    finalize_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
}

// SAFETY: the Retained Metal handles are created + only touched on the worker
// thread (built at load, then encoded in the same thread's forward followup);
// never actually shared across threads — the `Arc` exists only so the
// per-step `PendingSampler` handle can hold a refcount into the (Send)
// followup. Mirrors `MetalArena`'s Send+Sync pair in `metal_allocator.rs`.
unsafe impl Send for SamplerArena {}
unsafe impl Sync for SamplerArena {}

/// One step's non-greedy sampler work: a lightweight handle into a
/// [`SamplerArena`] (whose buffers `prepare_step` already filled with this
/// step's params), encoded onto the forward's OWN command buffer
/// ([`encode_into`](Self::encode_into)) — one commit, one host wait, no
/// second command buffer. Read the sampled tokens after the wait via
/// [`output`](Self::output). Holds one `Arc` refcount on the arena, so it
/// moves freely into the forward followup.
pub struct PendingSampler {
    arena: std::sync::Arc<SamplerArena>,
    njobs: u32,
    nslices: u32,
    /// Each job's logits row.
    rows: Vec<u32>,
}

/// Threadgroups the sliced passes aim to occupy: two per GPU core, at least 8.
/// The device's core count picks it; a device without one is an error.
#[derive(Clone, Copy, Debug)]
struct SliceTarget(u32);

impl SliceTarget {
    fn of(device: &Device) -> Result<Self, MetalStreamError> {
        crate::device::gpu_cores(device)
            .map(|c| Self((c.get() * 2).max(8)))
            .ok_or(MetalStreamError::UnknownGpuCores)
    }

    /// Vocab slices a step of `nrows` rows cuts each row into — per-step data,
    /// as the row count is. Capped at 32 (the per-slice staging is
    /// `2 * MAX_CANDIDATES` u32 per row, and the one-threadgroup-per-row
    /// kernels' serial walks scale with slice count). Small batches slice hard;
    /// a full batch of rows already fills the GPU.
    fn nslices(self, nrows: u32) -> u32 {
        (self.0 / nrows.max(1)).clamp(1, 32)
    }

    /// The largest `nrows * nslices(nrows)` over `1..=max_rows` — the extent
    /// the sliced buffers (partials/hist/counts/staging) must cover: at most
    /// the target while the slice count is interior, `32 * nrows` while
    /// clamped high, and `nrows` once rows alone fill the machine.
    fn sliced_max(self, max_rows: u32) -> usize {
        self.0.max(max_rows) as usize
    }
}

impl SamplerArena {
    /// Allocate + bind everything the sampler pipeline needs, once. Sizes are
    /// compile-time facts of the loaded config: the logits width `kernels`
    /// were baked for, `max_rows` from the worker's `max_num_seqs`, `max_hist`
    /// from the request-length bounds. All buffers are `StorageModeShared`,
    /// pinned into the worker's persistent residency set (the one the pool
    /// commits once), so every forward command buffer sees them resident.
    /// Returned behind an `Arc`: the per-step [`PendingSampler`] handle holds a
    /// refcount so it can move freely into the forward followup.
    pub fn new(
        device: &Device,
        residency: &MetalResidencySet,
        max_rows: u32,
        kernels: &SamplerKernels,
        max_hist: u32,
    ) -> std::sync::Arc<Self> {
        let max_rows = max_rows.max(1);
        let max_hist = max_hist.max(1);
        let (vocab, slicing) = (kernels.vocab.get(), kernels.slicing);
        let sliced_max = slicing.sliced_max(max_rows);
        let n = max_rows as usize;
        let h = max_hist as usize;

        let mut pins: Vec<Pinned> = Vec::new();
        let mut mk = |bytes: usize, what: &str| -> Buffer {
            let buf = device
                .newBufferWithLength_options(
                    bytes.max(1),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "sampler arena: newBufferWithLength returned nil ({what}, {bytes} bytes)"
                    )
                });
            pins.push(residency.pin(buf.clone()));
            unsafe {
                std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes.max(1));
            }
            buf
        };
        let scratch_f32 = mk(n * vocab as usize * 4, "scratch");
        let out_buf = mk(n * 4, "out");
        let row_idx_buf = mk(n * 4, "row_idx");
        let out_ids_buf = mk(n * h * 4, "out_ids");
        let prompt_ids_buf = mk(n * h * 4, "prompt_ids");
        let reps_buf = mk(n * 4, "reps");
        let freqs_buf = mk(n * 4, "freqs");
        let press_buf = mk(n * 4, "press");
        let row_state_buf = mk(n * ROW_STATE_LEN * 4, "row_state");
        let partials_buf = mk(sliced_max * 3 * 4, "partials");
        let hist_buf = mk(sliced_max * 256 * 4, "hist");
        let counts_buf = mk(sliced_max * 4 * 4, "counts");
        let staging_buf = mk(sliced_max * 2 * 1024 * 4, "staging");
        let consts_buf = mk(4 * 4, "consts");
        #[cfg(feature = "sampler-telemetry")]
        let telem_k: u32 = SAMPLER_TELEM_K;
        #[cfg(feature = "sampler-telemetry")]
        let (topk_probs_buf, topk_indices_buf, stats_buf, telem_consts_buf) = {
            let k = telem_k as usize;
            (
                mk(n * k * 4, "topk_probs"),
                mk(n * k * 4, "topk_indices"),
                mk(n * 2 * 4, "stats"),
                mk(2 * 4, "telem_consts"),
            )
        };

        // Per-stage argument tables, gap-filled with a zero buffer so an
        // unused slot can never hold a stale address, then fully bound ONCE:
        // every stage's buffers are arena-persistent, so the bindings never
        // change. Only the cast's slot 1 (the forward's logits) is rebound
        // per forward by `PendingSampler::encode_into`.
        let mk_table =
            |count: usize| -> Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>> {
                use objc2_metal::MTL4ArgumentTable as _;
                let desc = objc2_metal::MTL4ArgumentTableDescriptor::new();
                desc.setMaxBufferBindCount(count);
                let table = device
                    .newArgumentTableWithDescriptor_error(&desc)
                    .expect("sampler arg table alloc");
                let zero = shared_zeroed(device, 16);
                let zero_addr = zero.gpuAddress();
                for i in 0..count {
                    unsafe { table.setAddress_atIndex(zero_addr, i) };
                }
                table
            };
        // Buffer-index layouts per kernel (as in the shaders' signatures):
        //   cast: 0=scratch, 1=logits(forward), 2=row_idx, 3=consts.
        //   penalties: 0=scratch, 1=out_ids, 2=prompt_ids, 3=rep, 4=freq,
        //   5=pres, 6=consts.
        //   softmax_reduce: 0=scratch, 1=partials, 2=row_state, 3=consts.
        //   stats_pick: 0=partials, 1=row_state, 2=consts.
        //   softmax_materialize: 0=scratch, 1=partials, 2=row_state, 3=consts.
        //   histogram: 0=prob_bits(scratch), 1=hist, 2=row_state, 3=consts.
        //   threshold_pick: 0=hist, 1=row_state, 2=consts.
        //   count_compact: 0=prob_bits, 1=counts, 2=staging, 3=row_state,
        //   4=consts.
        //   quota_pick: 0=counts, 1=row_state, 2=consts.
        //   compact_tied: 0=prob_bits, 1=counts, 2=staging, 3=row_state,
        //   4=consts.
        //   finalize: 0=staging, 1=counts, 2=row_state, 3=prob_bits, 4=output,
        //   5=partials, 6..9=telemetry spill (only under sampler-telemetry),
        //   10=consts.
        let cast_at = mk_table(4);
        let penalties_at = mk_table(7);
        let softmax_reduce_at = mk_table(4);
        let stats_pick_at = mk_table(3);
        let softmax_materialize_at = mk_table(4);
        let histogram_at = mk_table(4);
        let threshold_pick_at = mk_table(3);
        let count_compact_at = mk_table(5);
        let quota_pick_at = mk_table(3);
        let compact_tied_at = mk_table(5);
        let finalize_at = mk_table(11);

        let arena = std::sync::Arc::new(Self {
            max_rows,
            vocab,
            slicing,
            sliced_max,
            _pins: pins,
            scratch_f32,
            out_buf,
            row_idx_buf,
            out_ids_buf,
            prompt_ids_buf,
            reps_buf,
            freqs_buf,
            press_buf,
            row_state_buf,
            partials_buf,
            hist_buf,
            counts_buf,
            staging_buf,
            consts_buf,
            max_hist,
            #[cfg(feature = "sampler-telemetry")]
            topk_probs_buf,
            #[cfg(feature = "sampler-telemetry")]
            topk_indices_buf,
            #[cfg(feature = "sampler-telemetry")]
            stats_buf,
            #[cfg(feature = "sampler-telemetry")]
            telem_consts_buf,
            cast_at,
            penalties_at,
            softmax_reduce_at,
            stats_pick_at,
            softmax_materialize_at,
            histogram_at,
            threshold_pick_at,
            count_compact_at,
            quota_pick_at,
            compact_tied_at,
            finalize_at,
        });
        arena.bind_stage_tables(device);
        arena
    }

    /// Bind every stage's buffers into its (already gap-filled) argument
    /// table, reading the owning fields. Called once at construction; the
    /// bindings never change because the buffers are arena-persistent. Only
    /// the cast's slot 1 (the forward's logits) is rebound per forward by
    /// [`PendingSampler::encode_into`].
    fn bind_stage_tables(&self, device: &Device) {
        use objc2_metal::{MTL4ArgumentTable as _, MTLBuffer as _};
        let scratch = self.scratch_f32.gpuAddress();
        let partials = self.partials_buf.gpuAddress();
        let row_state = self.row_state_buf.gpuAddress();
        let hist = self.hist_buf.gpuAddress();
        let counts = self.counts_buf.gpuAddress();
        let staging = self.staging_buf.gpuAddress();
        let consts = self.consts_buf.gpuAddress();
        let zero_addr = shared_zeroed(device, 16).gpuAddress();
        unsafe {
            self.cast_at.setAddress_atIndex(scratch, 0);
            self.cast_at.setAddress_atIndex(zero_addr, 1);
            self.cast_at
                .setAddress_atIndex(self.row_idx_buf.gpuAddress(), 2);
            self.cast_at.setAddress_atIndex(consts, 3);
            self.penalties_at.setAddress_atIndex(scratch, 0);
            self.penalties_at
                .setAddress_atIndex(self.out_ids_buf.gpuAddress(), 1);
            self.penalties_at
                .setAddress_atIndex(self.prompt_ids_buf.gpuAddress(), 2);
            self.penalties_at
                .setAddress_atIndex(self.reps_buf.gpuAddress(), 3);
            self.penalties_at
                .setAddress_atIndex(self.freqs_buf.gpuAddress(), 4);
            self.penalties_at
                .setAddress_atIndex(self.press_buf.gpuAddress(), 5);
            self.penalties_at.setAddress_atIndex(consts, 6);
            self.softmax_reduce_at.setAddress_atIndex(scratch, 0);
            self.softmax_reduce_at.setAddress_atIndex(partials, 1);
            self.softmax_reduce_at.setAddress_atIndex(row_state, 2);
            self.softmax_reduce_at.setAddress_atIndex(consts, 3);
            self.stats_pick_at.setAddress_atIndex(partials, 0);
            self.stats_pick_at.setAddress_atIndex(row_state, 1);
            self.stats_pick_at.setAddress_atIndex(consts, 2);
            self.softmax_materialize_at.setAddress_atIndex(scratch, 0);
            self.softmax_materialize_at.setAddress_atIndex(partials, 1);
            self.softmax_materialize_at.setAddress_atIndex(row_state, 2);
            self.softmax_materialize_at.setAddress_atIndex(consts, 3);
            self.histogram_at.setAddress_atIndex(scratch, 0);
            self.histogram_at.setAddress_atIndex(hist, 1);
            self.histogram_at.setAddress_atIndex(row_state, 2);
            self.histogram_at.setAddress_atIndex(consts, 3);
            self.threshold_pick_at.setAddress_atIndex(hist, 0);
            self.threshold_pick_at.setAddress_atIndex(row_state, 1);
            self.threshold_pick_at.setAddress_atIndex(consts, 2);
            self.count_compact_at.setAddress_atIndex(scratch, 0);
            self.count_compact_at.setAddress_atIndex(counts, 1);
            self.count_compact_at.setAddress_atIndex(staging, 2);
            self.count_compact_at.setAddress_atIndex(row_state, 3);
            self.count_compact_at.setAddress_atIndex(consts, 4);
            self.quota_pick_at.setAddress_atIndex(counts, 0);
            self.quota_pick_at.setAddress_atIndex(row_state, 1);
            self.quota_pick_at.setAddress_atIndex(consts, 2);
            self.compact_tied_at.setAddress_atIndex(scratch, 0);
            self.compact_tied_at.setAddress_atIndex(counts, 1);
            self.compact_tied_at.setAddress_atIndex(staging, 2);
            self.compact_tied_at.setAddress_atIndex(row_state, 3);
            self.compact_tied_at.setAddress_atIndex(consts, 4);
            self.finalize_at.setAddress_atIndex(staging, 0);
            self.finalize_at.setAddress_atIndex(counts, 1);
            self.finalize_at.setAddress_atIndex(row_state, 2);
            self.finalize_at.setAddress_atIndex(scratch, 3);
            self.finalize_at
                .setAddress_atIndex(self.out_buf.gpuAddress(), 4);
            self.finalize_at.setAddress_atIndex(partials, 5);
            #[cfg(feature = "sampler-telemetry")]
            {
                self.finalize_at
                    .setAddress_atIndex(self.topk_probs_buf.gpuAddress(), 6);
                self.finalize_at
                    .setAddress_atIndex(self.topk_indices_buf.gpuAddress(), 7);
                self.finalize_at
                    .setAddress_atIndex(self.stats_buf.gpuAddress(), 8);
                self.finalize_at
                    .setAddress_atIndex(self.telem_consts_buf.gpuAddress(), 9);
            }
            self.finalize_at.setAddress_atIndex(consts, 10);
        }
    }

    /// Fill the arena's shared buffers with THIS step's sampler inputs and
    /// return the lightweight handle the forward followup encodes. Pure host
    /// memcpy — no Metal calls, no allocation. The penalties stage always
    /// runs (rows without penalties carry neutral coefficients + all-padding
    /// histories, which the kernel's `count > 0` test makes a no-op), so
    /// `any_penalty` only gates whether real histories are written.
    pub fn prepare_step(
        self: &std::sync::Arc<Self>,
        params: &scratchy_core_common::GpuSampleParams,
        njobs: u32,
        mut record: Option<&mut Vec<(Buffer, Vec<u8>)>>,
    ) -> PendingSampler {
        assert!(
            njobs <= self.max_rows,
            "sampler step rows ({njobs}) exceed arena max_rows ({})",
            self.max_rows
        );
        let n = njobs as usize;
        let nslices = self.slicing.nslices(njobs);
        assert!(
            (n * nslices as usize) <= self.sliced_max,
            "sampler step sliced extent ({n} * {nslices}) exceeds arena ({})",
            self.sliced_max
        );

        // Shared-storage contents are plain host memory: fill via memcpy — or,
        // while an earlier command buffer may still be sampling from them,
        // `record` the writes for the device to make at the head of this step's.
        let mut write = |buf: &Buffer, bytes: &[u8]| match record.as_deref_mut() {
            Some(record) => record.push((buf.clone(), bytes.to_vec())),
            None => {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u8, bytes.len())
                };
                dst.copy_from_slice(bytes);
            }
        };
        let u32s = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        write(&self.row_idx_buf, &u32s(&params.row_indices));
        write(&self.reps_buf, &f32s(&params.rep_penalties));
        write(&self.freqs_buf, &f32s(&params.freq_penalties));
        write(&self.press_buf, &f32s(&params.pres_penalties));

        // row_state: host words (temperature, top_k, top_p, min_p, uniform,
        // cap). GPU words (max, sum, threshold, round, counts) are (re)written
        // by the kernels every step — `sample_stats_pick` resets the descent
        // state — so stale words never leak into a step.
        let mut row_state: Vec<u32> = vec![0; n * ROW_STATE_LEN];
        for r in 0..n {
            let s = &mut row_state[r * ROW_STATE_LEN..(r + 1) * ROW_STATE_LEN];
            s[0] = params.temperatures[r].to_bits();
            s[3] = params.top_ks[r].max(0) as u32;
            s[4] = params.top_ps[r].to_bits();
            s[5] = params.min_ps[r].to_bits();
            s[6] = params.uniforms[r].to_bits();
            // cap = effective k; top_k == 0 → MAX_CANDIDATES (the shader's
            // `effective_k` fallback, computed host-side so the descent's
            // `pick` kernel never needs vocab).
            let k = if params.top_ks[r] > 0 {
                (params.top_ks[r] as u32).min(self.vocab)
            } else {
                1024u32.min(self.vocab)
            };
            s[8] = k;
        }
        write(&self.row_state_buf, &u32s(&row_state));

        // Penalties histories: row-major [njobs, max_*] padded with `vocab`
        // (never a real index). The arena's max_hist bound is a worker-config
        // fact; a longer history is a config violation, not data.
        let max_out = params.max_output_len.max(1);
        let max_prompt = params.max_prompt_len.max(1);
        assert!(
            max_out <= self.max_hist && max_prompt <= self.max_hist,
            "sampler history ({max_out}/{max_prompt}) exceeds arena max_hist ({})",
            self.max_hist
        );
        let pad = self.vocab as i32;
        let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut flat_out = vec![pad; n * max_out as usize];
        let mut flat_prompt = vec![pad; n * max_prompt as usize];
        if params.any_penalty {
            // Re-key the gatherer's own [njobs, its_max_*] layout into this
            // step's strides (equal in practice; kept general).
            for r in 0..n {
                let src = params
                    .output_token_ids
                    .chunks_exact(params.max_output_len.max(1) as usize)
                    .nth(r)
                    .map(|c| c.to_vec())
                    .unwrap_or_default();
                flat_out[r * max_out as usize..r * max_out as usize + src.len()]
                    .copy_from_slice(&src);
                let src = params
                    .prompt_token_ids
                    .chunks_exact(params.max_prompt_len.max(1) as usize)
                    .nth(r)
                    .map(|c| c.to_vec())
                    .unwrap_or_default();
                flat_prompt[r * max_prompt as usize..r * max_prompt as usize + src.len()]
                    .copy_from_slice(&src);
            }
        }
        write(&self.out_ids_buf, &i32s(&flat_out));
        write(&self.prompt_ids_buf, &i32s(&flat_prompt));

        write(
            &self.consts_buf,
            &u32s(&[nslices, njobs, max_out, max_prompt]),
        );
        #[cfg(feature = "sampler-telemetry")]
        {
            let telem_on = u32::from(
                scratchy_core_common::sampler_telemetry::SamplerTelemetry::global().is_enabled(),
            );
            write(&self.telem_consts_buf, &u32s(&[telem_on, SAMPLER_TELEM_K]));
        }

        PendingSampler {
            arena: self.clone(),
            njobs,
            nslices,
            rows: params.row_indices.clone(),
        }
    }
}

impl PendingSampler {
    /// Encode the whole sample pipeline onto the forward's OWN MTL4 compute
    /// encoder (from the argmax followup): cast (and penalties) → softmax
    /// reduce → stats pick → materialize → 4 × (histogram, threshold pick) →
    /// count/compact → quota pick → compact tied → finalize. Every stage
    /// rides its own (arena-persistent) argument table; the forward's logits
    /// buffer is bound into the cast's table (slot 1) first.
    pub fn encode_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        logits_addr: u64,
        kernels: &SamplerKernels,
    ) {
        use objc2_metal::MTL4ArgumentTable;
        let rows = self.njobs;
        let sliced = rows * self.nslices;
        unsafe {
            self.arena.cast_at.setAddress_atIndex(logits_addr, 1);
        }

        let stage = |s: SamplerStage,
                     table: &Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
                     tgs: u32| {
            encode_sampler_stage_into_mtl4(enc, kernels.stage(s), table, tgs)
        };
        let a = &self.arena;

        stage(SamplerStage::Cast, &a.cast_at, sliced);
        // Penalties always run: rows without penalties carry neutral
        // coefficients + all-padding histories, which the kernel's `count
        // > 0` test makes a no-op.
        stage(SamplerStage::Penalties, &a.penalties_at, sliced);

        // Softmax: per-slice partials, merged to row stats, materialized.
        stage(SamplerStage::SoftmaxReduce, &a.softmax_reduce_at, sliced);
        stage(SamplerStage::StatsPick, &a.stats_pick_at, rows);
        stage(
            SamplerStage::SoftmaxMaterialize,
            &a.softmax_materialize_at,
            sliced,
        );

        // Byte-histogram descent: four rounds.
        for _ in 0..4 {
            stage(SamplerStage::Histogram, &a.histogram_at, sliced);
            stage(SamplerStage::ThresholdPick, &a.threshold_pick_at, rows);
        }

        // Compaction: strict candidates + counts, tie quotas, tied candidates.
        stage(SamplerStage::CountCompact, &a.count_compact_at, sliced);
        stage(SamplerStage::QuotaPick, &a.quota_pick_at, rows);
        stage(SamplerStage::CompactTied, &a.compact_tied_at, sliced);

        // Sort + top-p + draw.
        stage(SamplerStage::Finalize, &a.finalize_at, rows);
    }

    /// Copy each job's sampled token over `tokens[its logits row]` — the step's
    /// per-row argmax output — once the sampler is done: the next step reads the
    /// token from there, and the arena's own output is the next step's to write.
    pub fn copy_tokens_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        tokens: &Buffer,
    ) {
        use objc2_metal::{
            MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLStages,
        };
        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Blit,
            MTL4VisibilityOptions::Device,
        );
        let at = |i: usize| i * size_of::<u32>();
        for (job, &row) in self.rows.iter().enumerate() {
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &self.arena.out_buf,
                    at(job),
                    tokens,
                    at(row as usize),
                    size_of::<u32>(),
                );
            }
        }
    }

    /// Each job's logits row.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// Telemetry spill buffers `(topk_probs, topk_indices, stats, njobs, k)`
    /// for reading back after the forward's host wait. `stats` holds
    /// `[max_prob, entropy_nats]` per row; `topk_*` hold `k` entries per row,
    /// descending by prob (prob 0.0 = padding past the candidate count).
    #[cfg(feature = "sampler-telemetry")]
    pub fn telemetry_output(&self) -> Option<(Buffer, Buffer, Buffer, u32, u32)> {
        Some((
            self.arena.topk_probs_buf.clone(),
            self.arena.topk_indices_buf.clone(),
            self.arena.stats_buf.clone(),
            self.njobs,
            SAMPLER_TELEM_K,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtl4_dispatch::{Mtl4DispatchBatch, read_slice, shared_slice};

    /// The sampler's kernels baked for logits `vocab` wide of `dtype`, as a
    /// model's are.
    fn baked(device: &Device, vocab: usize, dtype: CastDtype) -> SamplerKernels {
        let vocab = LogitsWidth(vocab as u32);
        let telemetry = cfg!(feature = "sampler-telemetry");
        let keys = SamplerStage::ALL.map(|s| s.key(vocab, dtype, telemetry));
        let stages: [BakedKernel; SamplerStage::COUNT] = crate::aot::baked_kernels(&keys)
            .try_into()
            .unwrap_or_else(|_| panic!("one baked kernel per stage"));
        SamplerKernels::new(device, vocab, &stages).expect("sampler kernels")
    }

    /// Run the full pipeline on `logits` (one row) and return the sampled
    /// token — the parity harness's metal side. Returns `None` when no metal
    /// device / MTL4 queue exists.
    fn run_metal_sample(
        device: &Device,
        logits: &[f32],
        temp: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        uniform: f32,
    ) -> Option<u32> {
        let kernels = baked(device, logits.len(), CastDtype::F32);
        let params = scratchy_core_common::GpuSampleParams {
            row_indices: vec![0],
            temperatures: vec![temp],
            top_ks: vec![top_k],
            top_ps: vec![top_p],
            min_ps: vec![min_p],
            uniforms: vec![uniform],
            rep_penalties: vec![1.0],
            freq_penalties: vec![0.0],
            pres_penalties: vec![0.0],
            ..Default::default()
        };
        run_pipeline(device, &kernels, &params, logits, 1).map(|(t, _)| t)
    }

    /// Prepare + run the pipeline on one command buffer; read back the sampled
    /// token (row 0) and the commit-wait time. `logits` is the TOTAL logits
    /// buffer, of the dtype `kernels` cast from; `row_indices` (in `params`)
    /// picks the row(s).
    fn run_pipeline<T: Copy>(
        device: &Device,
        kernels: &SamplerKernels,
        params: &scratchy_core_common::GpuSampleParams,
        logits: &[T],
        njobs: u32,
    ) -> Option<(u32, std::time::Duration)> {
        let logits_buf = shared_slice(device, logits);
        let batch = Mtl4DispatchBatch::begin(device)?;
        // The sampler's buffers AND the logits row must be resident for THIS
        // command buffer: build the arena against the batch's own set and pin
        // the logits into it too (the batch's commit attaches exactly that
        // set; the arena's pins keep its buffers in it for as long as both
        // live — the logits pin lives to the end of this scope).
        // The step's token buffer, one slot per logits row: the sampled tokens land at their rows.
        let rows = params
            .row_indices
            .iter()
            .max()
            .map_or(1, |&r| r as usize + 1);
        let tokens = shared_slice(device, &vec![u32::MAX; rows]);
        let (pending, pins) = {
            let res = batch.residency();
            let max_hist = params.max_output_len.max(params.max_prompt_len).max(1);
            let arena = SamplerArena::new(device, res, njobs, kernels, max_hist);
            let pending = arena.prepare_step(params, njobs, None);
            (
                pending,
                [res.pin(logits_buf.clone()), res.pin(tokens.clone())],
            )
        };
        use objc2_metal::MTLBuffer as _;
        let logits_addr = logits_buf.gpuAddress();
        let enc = batch.encoder();
        pending.encode_into(enc, logits_addr, kernels);
        pending.copy_tokens_into(enc, &tokens);
        let t0 = std::time::Instant::now();
        batch.commit(true);
        let wait = t0.elapsed();
        drop(pins);
        let row = params.row_indices[0] as usize;
        Some((read_slice::<u32>(&tokens, rows)[row], wait))
    }

    /// Dispatch `sample` on a synthetic f32 logits row with a near-zero
    /// temperature + top_k=1: the softmax collapses onto the argmax, the
    /// descent keeps exactly one candidate, and the categorical draw (any
    /// uniform) must return the argmax index. Guarded to skip when no Metal
    /// device / MTL4 queue is available (CI Linux, headless).
    #[test]
    fn sample_top_k1_returns_argmax() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();

        // Row of 4096 logits; index 1234 is the clear maximum.
        let vocab: u32 = 4096;
        let argmax_idx: usize = 1234;
        let mut logits = vec![0.1f32; vocab as usize];
        logits[argmax_idx] = 9.0;
        logits[7] = 3.0;
        logits[42] = 2.0;

        let got = run_metal_sample(&device, &logits, 0.01, 1, 1.0, 0.0, 0.73);
        let Some(got) = got else {
            eprintln!("skipping: no MTL4 queue");
            return;
        };
        assert_eq!(
            got as usize, argmax_idx,
            "top_k=1 near-zero-temp sample must return the argmax index"
        );
    }

    // =======================================================================
    // Parity harness: pure-Rust CPU golden vs the REAL metal sampler pipeline.
    //
    // The golden mirrors cuda `sample_top_k_top_p_core` (the spec) EXACTLY:
    //   softmax(logit/T) -> radix-select top-k threshold -> min-p ->
    //   two-phase compaction -> sort desc -> top-p cutoff -> categorical draw
    //   with the SAME host uniform (cumsum >= uniform*total).
    //
    // Because the GPU sum_exp reduction and Metal's `exp` differ from the
    // host by a few ULP, we only demand an EXACT token match when the draw is
    // provably robust to tiny perturbations (distinct candidate probs + wide
    // top-p / draw / radix margins). Otherwise we require the returned token to
    // lie in the valid post-cutoff support set. Greedy/argmax cases are always
    // robust, hence exact. Assertions are NOT weakened to pass a broken kernel:
    // an out-of-support token, or a robust case that mismatches, fails.
    // =======================================================================

    const MAX_CANDIDATES: usize = 1024;

    /// Deterministic SplitMix64 -> reproducible pseudo-random cases.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        /// f32 in [0, 1).
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        /// f32 in [lo, hi).
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.unit() * (hi - lo)
        }
        fn usize(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u64() as usize) % (hi - lo)
        }
    }

    struct Golden {
        token: u32,
        support: Vec<u32>,
        robust: bool,
    }

    /// Softmax over `logits/temp` in f32, matching the kernel's arithmetic
    /// (per-element `exp(v - max)`, then multiply by `1/sum`). Returns the
    /// probabilities and `inv_sum_exp` (== the max probability).
    fn softmax_probs(logits: &[f32], temp: f32) -> (Vec<f32>, f32) {
        let inv_temp = 1.0f32 / temp;
        let mut max_logit = f32::NEG_INFINITY;
        for &l in logits {
            let v = l * inv_temp;
            if v > max_logit {
                max_logit = v;
            }
        }
        let mut sum = 0.0f32;
        let mut exps = vec![0.0f32; logits.len()];
        for (i, &l) in logits.iter().enumerate() {
            let e = (l * inv_temp - max_logit).exp();
            exps[i] = e;
            sum += e;
        }
        let inv_sum = 1.0f32 / sum;
        let probs: Vec<f32> = exps.iter().map(|&e| e * inv_sum).collect();
        (probs, inv_sum)
    }

    /// Pure-Rust replica of `sample_top_k_top_p_core`.
    #[allow(clippy::too_many_arguments)]
    fn golden_sample(
        logits: &[f32],
        temp: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        uniform: f32,
    ) -> Golden {
        let vsize = logits.len();
        let (probs, inv_sum) = softmax_probs(logits, temp);
        let prob_bits: Vec<u32> = probs.iter().map(|p| p.to_bits()).collect();

        let effective_k = if top_k > 0 {
            (top_k as usize).min(vsize)
        } else {
            MAX_CANDIDATES.min(vsize)
        };

        // Phase 3: radix-select the top-K probability threshold (bit-for-bit).
        let mut threshold_bits: u32 = 0;
        for bit in (0..32).rev() {
            let candidate = threshold_bits | (1u32 << bit);
            let count = prob_bits.iter().filter(|&&b| b >= candidate).count();
            if count >= effective_k {
                threshold_bits = candidate;
            }
        }
        let mut threshold_prob = f32::from_bits(threshold_bits);

        // Phase 3b: min-p.
        if min_p > 0.0 {
            threshold_prob = threshold_prob.max(min_p * inv_sum);
        }
        let threshold_bits_u = threshold_prob.to_bits();
        let cap = effective_k.min(MAX_CANDIDATES);

        // Phase 4: two-phase compaction (strict-above, then tied-at-threshold).
        let strict_count_full = prob_bits.iter().filter(|&&b| b > threshold_bits_u).count();
        let tied_count = prob_bits.iter().filter(|&&b| b == threshold_bits_u).count();

        let mut cand: Vec<(f32, u32)> = Vec::new();
        for i in 0..vsize {
            if prob_bits[i] > threshold_bits_u {
                if cand.len() >= cap {
                    break;
                }
                cand.push((probs[i], i as u32));
            }
        }
        if cand.len() < cap {
            for i in 0..vsize {
                if prob_bits[i] == threshold_bits_u {
                    if cand.len() >= cap {
                        break;
                    }
                    cand.push((probs[i], i as u32));
                }
            }
        }
        let num_candidates = cand.len();
        assert!(num_candidates > 0, "golden: empty candidate set");

        // Phase 5: sort descending by probability.
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Phase 6: top-p cutoff (inclusive: first index where cumsum > top_p).
        let mut cumsum = 0.0f32;
        let mut cutoff = num_candidates;
        let mut topp_margin = f32::INFINITY;
        for (i, c) in cand.iter().enumerate() {
            let before = cumsum;
            cumsum += c.0;
            if cumsum > top_p {
                cutoff = i + 1;
                topp_margin = (cumsum - top_p).min(top_p - before);
                break;
            }
        }

        let mut total = 0.0f32;
        for c in cand.iter().take(cutoff) {
            total += c.0;
        }

        // Categorical draw with the fixed host uniform.
        let target = uniform * total;
        let mut cumsum2 = 0.0f32;
        let mut sampled = cand[cutoff - 1].1;
        let mut draw_margin = f32::INFINITY;
        for c in cand.iter().take(cutoff) {
            let before = cumsum2;
            cumsum2 += c.0;
            if cumsum2 >= target {
                sampled = c.1;
                draw_margin = (target - before).min(cumsum2 - target);
                break;
            }
        }

        let support: Vec<u32> = cand[..cutoff].iter().map(|c| c.1).collect();

        // Robustness: exact match is only demanded when the outcome cannot flip
        // under a few-ULP perturbation of sum_exp / exp.
        //  * distinct candidate probs -> no sort-order ambiguity in the cutoff
        //    region, and the draw index is well-defined;
        //  * radix boundary not over-subscribed by ties;
        //  * wide top-p and draw margins.
        let slots_for_tied = cap.saturating_sub(strict_count_full);
        let radix_ok = strict_count_full <= cap && tied_count <= slots_for_tied;
        let mut distinct_in_cutoff = true;
        for i in 0..cutoff {
            for j in (i + 1)..cutoff {
                if cand[i].0.to_bits() == cand[j].0.to_bits() {
                    distinct_in_cutoff = false;
                }
            }
        }
        const MARGIN: f32 = 1e-4;
        let topp_ok = topp_margin > MARGIN;
        let draw_ok = draw_margin > MARGIN * total.max(1e-6);
        let robust = radix_ok && distinct_in_cutoff && topp_ok && draw_ok;

        Golden {
            token: sampled,
            support,
            robust,
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Mode {
        Greedy,
        TempOnly,
        TopK,
        TopP,
        MinP,
        Combined,
        UniformTies,
        Peaked,
    }

    #[test]
    fn sample_parity_vs_cpu_golden() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();
        // Probe for an MTL4 queue once so we skip cleanly on headless hosts.
        if Mtl4DispatchBatch::begin(&device).is_none() {
            eprintln!("skipping: no MTL4 queue");
            return;
        }

        let mut rng = Rng(0x00C0_FFEE_1234_5678);
        let modes = [
            Mode::Greedy,
            Mode::TempOnly,
            Mode::TopK,
            Mode::TopP,
            Mode::MinP,
            Mode::Combined,
            Mode::UniformTies,
            Mode::Peaked,
        ];

        let mut run = 0usize;
        let mut exact = 0usize;
        let mut support_only = 0usize;
        let mut divergences: Vec<String> = Vec::new();

        for case in 0..56usize {
            let mode = modes[case % modes.len()];
            let vocab = rng.usize(64, 4097);

            // Build logits per mode.
            let mut logits: Vec<f32> = (0..vocab).map(|_| rng.range(-8.0, 8.0)).collect();
            match mode {
                Mode::UniformTies => {
                    // All identical -> maximally ambiguous ties.
                    let v = rng.range(-2.0, 2.0);
                    for l in logits.iter_mut() {
                        *l = v;
                    }
                }
                Mode::Peaked => {
                    // One dominant logit -> near-degenerate distribution.
                    let idx = rng.usize(0, vocab);
                    logits[idx] = 40.0;
                }
                _ => {}
            }

            // Params per mode.
            let (temp, top_k, top_p, min_p) = match mode {
                Mode::Greedy => (0.01f32, 1i32, 1.0f32, 0.0f32),
                Mode::TempOnly => (rng.range(0.5, 1.6), 0, 1.0, 0.0),
                Mode::TopK => {
                    let ks = [1, 2, 5, 20, 50, 200];
                    (rng.range(0.6, 1.4), ks[rng.usize(0, ks.len())], 1.0, 0.0)
                }
                Mode::TopP => {
                    let ps = [0.5f32, 0.8, 0.9, 0.95];
                    (rng.range(0.6, 1.4), 0, ps[rng.usize(0, ps.len())], 0.0)
                }
                Mode::MinP => {
                    let ms = [0.01f32, 0.05, 0.1];
                    (rng.range(0.6, 1.4), 0, 1.0, ms[rng.usize(0, ms.len())])
                }
                Mode::Combined => {
                    let ks = [10, 50, 100];
                    (rng.range(0.6, 1.4), ks[rng.usize(0, ks.len())], 0.9, 0.02)
                }
                Mode::UniformTies => (1.0, 0, 0.9, 0.0),
                Mode::Peaked => (rng.range(0.5, 1.2), 0, 1.0, 0.0),
            };

            let uniform = rng.unit();

            let g = golden_sample(&logits, temp, top_k, top_p, min_p, uniform);
            let Some(got) = run_metal_sample(&device, &logits, temp, top_k, top_p, min_p, uniform)
            else {
                eprintln!("skipping: no MTL4 queue mid-run");
                return;
            };
            run += 1;

            let in_support = g.support.contains(&got);
            let params = format!(
                "mode={mode:?} vocab={vocab} temp={temp:.3} top_k={top_k} top_p={top_p} min_p={min_p} uniform={uniform:.4}"
            );

            if got == g.token {
                exact += 1;
            } else if in_support {
                // Non-exact but valid draw. Acceptable ONLY when the golden
                // itself flagged the case as non-robust (genuine float/tie
                // ambiguity). A robust mismatch is a real kernel bug.
                if g.robust {
                    divergences.push(format!(
                        "ROBUST-MISMATCH {params}: expected {} got {} (in support)",
                        g.token, got
                    ));
                } else {
                    support_only += 1;
                }
            } else {
                // Out of support => definitively wrong.
                divergences.push(format!(
                    "OUT-OF-SUPPORT {params}: expected {} got {} (support_len={})",
                    g.token,
                    got,
                    g.support.len()
                ));
            }
        }

        eprintln!(
            "[sample_parity] cases_run={run} exact={exact} support_only={support_only} divergences={}",
            divergences.len()
        );
        for d in &divergences {
            eprintln!("  DIVERGENCE: {d}");
        }
        assert!(
            divergences.is_empty(),
            "{} sampler divergences (see stderr)",
            divergences.len()
        );
        // Sanity: the exact-match path must actually exercise (not everything
        // collapsed to membership-only), else the parity test proves nothing.
        assert!(exact >= run / 2, "too few exact matches: {exact}/{run}");
    }

    // =======================================================================
    // apply_penalties parity: rep/freq/pres must reshape logits exactly, and
    // shift the argmax off a penalized token. Runs the cast + penalties stages
    // of the sliced pipeline and reads the scratch back.
    // =======================================================================

    fn golden_penalties(
        logits: &[f32],
        out_ids: &[i32],
        prompt_ids: &[i32],
        rep: f32,
        freq: f32,
        pres: f32,
        vocab: usize,
    ) -> Vec<f32> {
        let mut row = logits.to_vec();
        for (v, r) in row.iter_mut().enumerate().take(vocab) {
            let vi = v as i32;
            let mut count = 0i32;
            for &t in out_ids {
                if t == vi {
                    count += 1;
                }
            }
            for &t in prompt_ids {
                if t == vi {
                    count += 1;
                }
            }
            if count > 0 {
                let mut logit = *r;
                if logit > 0.0 {
                    logit /= rep;
                } else {
                    logit *= rep;
                }
                logit -= freq * count as f32 + pres;
                *r = logit;
            }
        }
        row
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = f32::NEG_INFINITY;
        let mut bi = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > best {
                best = x;
                bi = i;
            }
        }
        bi
    }

    /// Cast + penalties stages only, on `njobs` rows; returns the scratch
    /// (post-penalty f32 logits) for row 0 and row `njobs-1`.
    fn run_penalties(
        device: &Device,
        params: &scratchy_core_common::GpuSampleParams,
        logits: &[f32],
        njobs: u32,
        vocab: u32,
    ) -> Option<Vec<f32>> {
        use objc2_metal::MTLBuffer as _;
        let kernels = baked(device, vocab as usize, CastDtype::F32);
        let logits_buf = shared_slice(device, logits);
        let batch = Mtl4DispatchBatch::begin(device)?;
        let (pending, arena, _logits_pin) = {
            let res = batch.residency();
            let max_hist = params.max_output_len.max(params.max_prompt_len).max(1);
            let arena = SamplerArena::new(device, res, njobs, &kernels, max_hist);
            let pending = arena.prepare_step(params, njobs, None);
            (pending, arena, res.pin(logits_buf.clone()))
        };
        let enc = batch.encoder();
        // The cast + penalties stages are the pipeline's first two; the rest
        // would consume/rewrite the scratch, so stop after penalties. Bind the
        // logits into the cast's table first (the arena's tables are already
        // fully bound otherwise).
        use objc2_metal::MTL4ArgumentTable as _;
        unsafe {
            arena.cast_at.setAddress_atIndex(logits_buf.gpuAddress(), 1);
        }
        let sliced = njobs * pending.nslices;
        let cast = kernels.stage(SamplerStage::Cast);
        encode_sampler_stage_into_mtl4(enc, cast, &arena.cast_at, sliced);
        let penalties = kernels.stage(SamplerStage::Penalties);
        encode_sampler_stage_into_mtl4(enc, penalties, &arena.penalties_at, sliced);
        batch.commit(true);
        let scratch = &arena.scratch_f32;
        Some(read_slice::<f32>(scratch, njobs as usize * vocab as usize))
    }

    #[test]
    fn penalties_parity_vs_cpu_golden() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();

        let mut rng = Rng(0x0BAD_C0DE_9999);
        let mut shifted = 0usize;

        for _case in 0..8usize {
            let vocab = rng.usize(256, 1024);
            let mut logits: Vec<f32> = (0..vocab).map(|_| rng.range(-5.0, 5.0)).collect();

            // Force the pre-penalty argmax onto a token we will penalize, so the
            // penalty demonstrably moves the argmax.
            let hot = rng.usize(0, vocab);
            logits[hot] = 9.0;
            let hot2 = (hot + 7) % vocab;
            logits[hot2] = 6.0;

            let max_out = 6u32;
            let max_prompt = 6u32;
            // Pad with `vocab` (never a real index).
            let mut out_ids = vec![vocab as i32; max_out as usize];
            let mut prompt_ids = vec![vocab as i32; max_prompt as usize];
            out_ids[0] = hot as i32;
            out_ids[1] = hot as i32; // count 2 for `hot`
            out_ids[2] = hot2 as i32;
            prompt_ids[0] = hot as i32; // count 3 total for `hot`
            prompt_ids[1] = hot2 as i32; // count 2 total for `hot2`

            let rep = 1.3f32;
            let freq = 0.7f32;
            let pres = 0.4f32;

            let golden = golden_penalties(&logits, &out_ids, &prompt_ids, rep, freq, pres, vocab);

            let params = scratchy_core_common::GpuSampleParams {
                row_indices: vec![0],
                rep_penalties: vec![rep],
                freq_penalties: vec![freq],
                pres_penalties: vec![pres],
                output_token_ids: out_ids.clone(),
                prompt_token_ids: prompt_ids.clone(),
                max_output_len: max_out,
                max_prompt_len: max_prompt,
                any_penalty: true,
                temperatures: vec![1.0],
                top_ks: vec![0],
                top_ps: vec![1.0],
                min_ps: vec![0.0],
                uniforms: vec![0.5],
            };
            let Some(got) = run_penalties(&device, &params, &logits, 1, vocab as u32) else {
                eprintln!("skipping: no MTL4 queue");
                return;
            };

            // Bit-exact row parity (identical scalar arithmetic).
            for i in 0..vocab {
                assert!(
                    (got[i] - golden[i]).abs() <= 1e-4 * (1.0 + golden[i].abs()),
                    "penalty mismatch at token {i}: got {} golden {}",
                    got[i],
                    golden[i]
                );
            }

            // The penalty must move the argmax off the original hot token.
            let old_argmax = argmax(&logits);
            let new_argmax_golden = argmax(&golden);
            let new_argmax_got = argmax(&got);
            assert_eq!(
                new_argmax_got, new_argmax_golden,
                "post-penalty argmax must match golden"
            );
            if new_argmax_got != old_argmax {
                shifted += 1;
            }
        }

        eprintln!("[penalties_parity] argmax shifted in {shifted}/8 cases");
        assert!(
            shifted >= 1,
            "penalties never shifted the argmax; test is not exercising the effect"
        );
    }

    /// Mixed batch (2 rows) sharing the batch-wide padded history buffers: row 0
    /// has real history and non-trivial penalties; row 1 has an all-padding
    /// history AND neutral penalties (rep=1, freq=pres=0). The shared
    /// `max_output_len` / `max_prompt_len` padding must NOT perturb row 1 — its
    /// logits must return bit-for-bit unchanged. Guards the "one long row
    /// inflates the loop for every row" padding hazard the perf review flagged.
    #[test]
    fn penalties_mixed_batch_padding() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();

        let vocab: usize = 512;
        let max_out = 8u32;
        let max_prompt = 8u32;

        let mut rng = Rng(0xFEED_FACE_0001);
        let row0: Vec<f32> = (0..vocab).map(|_| rng.range(-5.0, 5.0)).collect();
        let row1: Vec<f32> = (0..vocab).map(|_| rng.range(-5.0, 5.0)).collect();
        let mut logits = Vec::with_capacity(2 * vocab);
        logits.extend_from_slice(&row0);
        logits.extend_from_slice(&row1);

        // Row 0 has real (non-padding) history; row 1 stays all-padding (==vocab).
        let mut out_ids = vec![vocab as i32; 2 * max_out as usize];
        let mut prompt_ids = vec![vocab as i32; 2 * max_prompt as usize];
        out_ids[0] = 3;
        out_ids[1] = 3;
        prompt_ids[0] = 17;

        // Row 0 penalized; row 1 neutral (rep=1 exact, freq=pres=0) => no-op.
        let reps = [1.3f32, 1.0f32];
        let freqs = [0.7f32, 0.0f32];
        let press = [0.4f32, 0.0f32];

        let golden0 = golden_penalties(
            &row0,
            &out_ids[0..max_out as usize],
            &prompt_ids[0..max_prompt as usize],
            reps[0],
            freqs[0],
            press[0],
            vocab,
        );

        let params = scratchy_core_common::GpuSampleParams {
            row_indices: vec![0, 1],
            rep_penalties: reps.to_vec(),
            freq_penalties: freqs.to_vec(),
            pres_penalties: press.to_vec(),
            output_token_ids: out_ids.clone(),
            prompt_token_ids: prompt_ids.clone(),
            max_output_len: max_out,
            max_prompt_len: max_prompt,
            any_penalty: true,
            temperatures: vec![1.0, 1.0],
            top_ks: vec![0, 0],
            top_ps: vec![1.0, 1.0],
            min_ps: vec![0.0, 0.0],
            uniforms: vec![0.5, 0.5],
        };

        let Some(got) = run_penalties(&device, &params, &logits, 2, vocab as u32) else {
            eprintln!("skipping: no MTL4 queue");
            return;
        };

        // Row 1 (the neutral / all-padding row) must be bit-for-bit unchanged.
        for i in 0..vocab {
            assert_eq!(
                got[vocab + i].to_bits(),
                row1[i].to_bits(),
                "no-penalty row perturbed at token {i}: got {} want {}",
                got[vocab + i],
                row1[i]
            );
        }
        // Row 0 must match its golden.
        for i in 0..vocab {
            assert!(
                (got[i] - golden0[i]).abs() <= 1e-4 * (1.0 + golden0[i].abs()),
                "row0 penalty mismatch at {i}: got {} golden {}",
                got[i],
                golden0[i]
            );
        }
    }

    /// Radix-select + softmax at production-scale vocabularies (up to qwen3.5's
    /// 248320) — exercises the byte-histogram descent and the full-vocab
    /// softmax passes far past the 4096 used elsewhere. A dominant peak makes
    /// the draw deterministic, so every case is an exact parity match.
    #[test]
    fn sample_parity_large_vocab() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();
        if Mtl4DispatchBatch::begin(&device).is_none() {
            eprintln!("skipping: no MTL4 queue");
            return;
        }

        let mut rng = Rng(0x5EED_2483_2000);
        let vocabs = [131072usize, 248320];
        let params: [(f32, i32, f32, f32); 4] = [
            (1.0, 0, 1.0, 0.0),  // temp only, full vocab
            (0.9, 50, 1.0, 0.0), // top-k
            (1.1, 0, 0.9, 0.0),  // top-p
            (0.8, 0, 1.0, 0.05), // min-p
        ];
        let mut run = 0usize;
        let mut exact = 0usize;
        let mut divergences: Vec<String> = Vec::new();

        for &vocab in &vocabs {
            let mut logits: Vec<f32> = (0..vocab).map(|_| rng.range(-8.0, 8.0)).collect();
            // A clear dominant logit -> softmax ~ one-hot -> deterministic draw.
            logits[rng.usize(0, vocab)] = 30.0;
            for &(temp, top_k, top_p, min_p) in &params {
                let uniform = rng.unit();
                let g = golden_sample(&logits, temp, top_k, top_p, min_p, uniform);
                let Some(got) =
                    run_metal_sample(&device, &logits, temp, top_k, top_p, min_p, uniform)
                else {
                    eprintln!("skipping: no MTL4 queue mid-run");
                    return;
                };
                run += 1;
                if got == g.token {
                    exact += 1;
                } else if g.support.contains(&got) {
                    if g.robust {
                        divergences.push(format!(
                            "ROBUST-MISMATCH vocab={vocab} temp={temp} k={top_k} p={top_p} mp={min_p}: exp {} got {}",
                            g.token, got
                        ));
                    }
                } else {
                    divergences.push(format!(
                        "OUT-OF-SUPPORT vocab={vocab} temp={temp} k={top_k} p={top_p} mp={min_p}: exp {} got {}",
                        g.token, got
                    ));
                }
            }
        }
        eprintln!(
            "[large_vocab] run={run} exact={exact} divergences={}",
            divergences.len()
        );
        for d in &divergences {
            eprintln!("  DIVERGENCE: {d}");
        }
        assert!(
            divergences.is_empty(),
            "{} large-vocab divergences (see stderr)",
            divergences.len()
        );
        assert!(
            exact >= run / 2,
            "too few exact at large vocab: {exact}/{run}"
        );
    }

    /// Times the full pipeline on gemma-4-26b-shaped bf16 logits (vocab
    /// 262144) — the exact sizes `scr chat`'s decode pays — at the default
    /// sampling settings, top_k=0, and a 4-job batch. Median commit-wait of
    /// `run_pipeline` — an upper bound on the GPU time (includes the host
    /// event wait).
    ///
    /// Run:
    ///   cargo test --release -p scratchy-target-metal --lib time_sampler_stages -- --nocapture
    #[test]
    fn time_sampler_stages() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();

        let vocab: u32 = 262_144;
        let mut rng = Rng(0x1234_5678);
        // Mostly-negative noise plus a few strong tokens, so the descent's
        // histogram rounds count a real distribution.
        let row: Vec<half::bf16> = (0..vocab)
            .map(|i| match i {
                1000 => 20.0,
                _ if i % 4096 == 0 => 10.0,
                _ => rng.range(-15.0, 5.0),
            })
            .map(half::bf16::from_f32)
            .collect();
        let kernels = baked(&device, vocab as usize, CastDtype::Bf16);

        for (label, njobs, top_k, top_p) in [
            ("top_k=64, top_p=0.95, 1 row", 1u32, 64, 0.95),
            ("top_k=0 → 1024 cap, 1 row", 1, 0, 1.0),
            ("top_k=64, top_p=0.95, 4 rows", 4, 64, 0.95),
        ] {
            let n = njobs as usize;
            let params = scratchy_core_common::GpuSampleParams {
                row_indices: (0..njobs).collect(),
                temperatures: vec![1.0; n],
                top_ks: vec![top_k; n],
                top_ps: vec![top_p; n],
                min_ps: vec![0.0; n],
                uniforms: vec![0.5; n],
                ..Default::default()
            };
            let logits = row.repeat(n);
            let (warm, reps) = (3, 10);
            let mut us = Vec::with_capacity(reps);
            for r in 0..warm + reps {
                let Some((_, wait)) = run_pipeline(&device, &kernels, &params, &logits, njobs)
                else {
                    eprintln!("skipping: no MTL4 queue");
                    return;
                };
                if r >= warm {
                    us.push(wait.as_secs_f64() * 1e6);
                }
            }
            us.sort_by(f64::total_cmp);
            println!("sample ({label}, bf16): {:8.1} µs", us[reps / 2]);
        }
    }
}
