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
//! passes (gather + penalties + softmax, the radix descent, compaction) and a
//! one-threadgroup-per-row finalize; each pass makes the previous pass's
//! cross-slice decision itself, so the pipeline is six dispatches riding the
//! forward's command buffer with Device barriers between them, with
//! [`encode_into`] encoding it onto the forward's own encoder. Each stage is
//! baked per model with its logits width compiled in ([`crate::off_tape`]).

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
type ArgumentTable = Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>;

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
    SoftmaxReduce,
    SoftmaxMaterialize,
    DescentRound1,
    DescentRound2,
    CountCompact,
    Finalize,
}

impl SamplerStage {
    pub const COUNT: usize = 6;
    pub const ALL: [Self; Self::COUNT] = [
        Self::SoftmaxReduce,
        Self::SoftmaxMaterialize,
        Self::DescentRound1,
        Self::DescentRound2,
        Self::CountCompact,
        Self::Finalize,
    ];

    fn function(self, cast: CastDtype) -> &'static str {
        match (self, cast) {
            (Self::SoftmaxReduce, CastDtype::F16) => "sample_softmax_reduce_f16",
            (Self::SoftmaxReduce, CastDtype::Bf16) => "sample_softmax_reduce_bf16",
            (Self::SoftmaxReduce, CastDtype::F32) => "sample_softmax_reduce_f32",
            (Self::SoftmaxMaterialize, _) => "sample_softmax_materialize",
            (Self::DescentRound1, _) => "sample_descent_round_1",
            (Self::DescentRound2, _) => "sample_descent_round_2",
            (Self::CountCompact, _) => "sample_count_compact",
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

    for (n, &(i, row)) in jobs.iter().enumerate() {
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
            // Generated-token count + the request's earlier rows (a verify step's) = the row's
            // decode position; advances each step so the seed varies. Shared with cuda_worker.
            let ahead = jobs[..n].iter().filter(|&&(j, _)| j == i).count();
            let position = (generated(req_id) + ahead) as u32;
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

/// The descent's histogram words per row — MUST equal
/// `DESCENT_ROUNDS * DESCENT_BUCKETS` in `shaders/sampling.metal`.
const DESCENT_HIST_LEN: usize = 3 * 2048;

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
/// handle; the kernels write every GPU word of a step before any reads it,
/// so no zeroing pass is needed between steps.
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
    row_idx_buf: Buffer, // [max_rows]
    out_ids_buf: Buffer, // penalties histories, [max_rows, max_hist]
    prompt_ids_buf: Buffer,
    reps_buf: Buffer, // [max_rows] f32 each
    freqs_buf: Buffer,
    press_buf: Buffer,
    row_state_buf: Buffer, // [max_rows, ROW_STATE_LEN]
    partials_buf: Buffer,  // [sliced_max, 3]
    hist_buf: Buffer,      // [max_rows, DESCENT_HIST_LEN]
    counts_buf: Buffer,    // [sliced_max, 4]
    staging_buf: Buffer,   // [sliced_max, 4 * MAX_CANDIDATES]
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
    // has its own table. Built + fully bound once here; the forward's logits
    // (the reduce's slot 1) and tokens (the finalize's slot 5) are bound per
    // forward by [`PendingSampler::encode_into`].
    softmax_reduce_at: ArgumentTable,
    softmax_materialize_at: ArgumentTable,
    descent_at: ArgumentTable,
    count_compact_at: ArgumentTable,
    finalize_at: ArgumentTable,
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
/// second command buffer; each sampled token lands over the step's argmax of
/// its row. Holds one `Arc` refcount on the arena, so it moves freely into
/// the forward followup.
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
    /// as the row count is. Capped at one simdgroup's lanes (the finalize gives
    /// each slice a lane; the per-slice staging is `4 * MAX_CANDIDATES` u32 per
    /// row, and its serial gather scales with slice count). Small batches slice
    /// hard; a full batch of rows already fills the GPU.
    fn nslices(self, nrows: u32) -> u32 {
        (self.0 / nrows.max(1)).clamp(1, SAMPLER_WARP_SIZE as u32)
    }

    /// The largest `nrows * nslices(nrows)` over `1..=max_rows` — the extent
    /// the sliced buffers (partials/counts/staging) must cover: at most
    /// the target while the slice count is interior, `32 * nrows` while
    /// clamped high, and `nrows` once rows alone fill the machine.
    fn sliced_max(self, max_rows: u32) -> usize {
        self.0.max(max_rows) as usize
    }
}

impl SamplerArena {
    /// The arena's per-row byte cost — the term of `new`'s allocation that is
    /// LINEAR in `max_rows`. The worker counts this BEFORE resolving an unset
    /// `--max-num-seqs`, because the width it resolves sizes the very arena
    /// allocated right after (`n·(vocab + 2·max_hist)·4` in the per-row
    /// buffers, plus the slice-count terms that track `sliced_max(nrows)`).
    /// Not an exact total (the fixed ~100-byte tables and the
    /// sampler-telemetry buffers are excluded) — the rounding residue a
    /// memory-affordable default needs to account for, nothing more.
    pub fn bytes_per_row(vocab: u32, max_hist: u32) -> usize {
        let n = 1usize; // per-row terms only
        let h = max_hist.max(1) as usize;
        // `mk(n * vocab * 4, "scratch")` + row_idx/reps/freqs/press (4·n)
        // + out_ids + prompt_ids (2·n·h) + row_state (16·n) + hist.
        n * (vocab as usize * 4 + 4 * 3 + 2 * h * 4 + 16 * 4 + DESCENT_HIST_LEN * 4)
            // The sliced buffers scale with `sliced_max(nrows)` — the slice
            // target while rows are interior, `nrows` once they alone fill
            // the machine. Their per-row slope at small `n` is the slice
            // target; count it so a width near the slice boundary cannot
            // out-run the estimate. (partials 3 + counts 4 words + staging
            // 4 KiB, all × 4 B per slice.)
            + (3 * 4 + 4 * 4 + 4 * 1024 * 4)
    }

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
        let row_idx_buf = mk(n * 4, "row_idx");
        let out_ids_buf = mk(n * h * 4, "out_ids");
        let prompt_ids_buf = mk(n * h * 4, "prompt_ids");
        let reps_buf = mk(n * 4, "reps");
        let freqs_buf = mk(n * 4, "freqs");
        let press_buf = mk(n * 4, "press");
        let row_state_buf = mk(n * ROW_STATE_LEN * 4, "row_state");
        let partials_buf = mk(sliced_max * 3 * 4, "partials");
        let hist_buf = mk(n * DESCENT_HIST_LEN * 4, "hist");
        let counts_buf = mk(sliced_max * 4 * 4, "counts");
        let staging_buf = mk(sliced_max * 4 * 1024 * 4, "staging");
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

        // Per-stage argument tables, each bound in full by `bind_stage_tables`.
        let mk_table = |count: usize| -> ArgumentTable {
            let desc = objc2_metal::MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(count);
            device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("sampler arg table alloc")
        };
        let softmax_reduce_at = mk_table(12);
        let softmax_materialize_at = mk_table(5);
        let descent_at = mk_table(4);
        let count_compact_at = mk_table(6);
        let finalize_at = mk_table(if cfg!(feature = "sampler-telemetry") {
            12
        } else {
            7
        });

        let arena = std::sync::Arc::new(Self {
            max_rows,
            vocab,
            slicing,
            sliced_max,
            _pins: pins,
            scratch_f32,
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
            softmax_reduce_at,
            softmax_materialize_at,
            descent_at,
            count_compact_at,
            finalize_at,
        });
        arena.bind_stage_tables(device);
        arena
    }

    /// Bind every stage's buffers into its argument table, in its kernel's
    /// signature order, reading the owning fields. Called once at
    /// construction; the bindings never change because the buffers are
    /// arena-persistent. The forward's logits and tokens hold a zero buffer
    /// until [`PendingSampler::encode_into`] binds them.
    fn bind_stage_tables(&self, device: &Device) {
        use objc2_metal::{MTL4ArgumentTable as _, MTLBuffer as _};
        let a = |b: &Buffer| b.gpuAddress();
        let (scratch, row_idx, row_state) = (
            a(&self.scratch_f32),
            a(&self.row_idx_buf),
            a(&self.row_state_buf),
        );
        let (partials, hist, consts) = (
            a(&self.partials_buf),
            a(&self.hist_buf),
            a(&self.consts_buf),
        );
        let (counts, staging) = (a(&self.counts_buf), a(&self.staging_buf));
        let forward = shared_zeroed(device, 16).gpuAddress();
        let reduce = [
            scratch,
            forward,
            row_idx,
            a(&self.out_ids_buf),
            a(&self.prompt_ids_buf),
            a(&self.reps_buf),
            a(&self.freqs_buf),
            a(&self.press_buf),
            row_state,
            partials,
            consts,
            hist,
        ];
        let finalize = [
            staging, counts, row_state, scratch, row_idx, forward, consts,
        ];
        #[cfg(feature = "sampler-telemetry")]
        let finalize = [
            &finalize[..],
            &[
                partials,
                a(&self.topk_probs_buf),
                a(&self.topk_indices_buf),
                a(&self.stats_buf),
                a(&self.telem_consts_buf),
            ],
        ]
        .concat();
        let tables: [(&ArgumentTable, &[u64]); 5] = [
            (&self.softmax_reduce_at, &reduce),
            (
                &self.softmax_materialize_at,
                &[scratch, partials, row_state, hist, consts],
            ),
            (&self.descent_at, &[scratch, hist, row_state, consts]),
            (
                &self.count_compact_at,
                &[scratch, hist, counts, staging, row_state, consts],
            ),
            (&self.finalize_at, &finalize[..]),
        ];
        for (table, addrs) in tables {
            for (i, &addr) in addrs.iter().enumerate() {
                unsafe { table.setAddress_atIndex(addr, i) };
            }
        }
    }

    /// Fill the arena's shared buffers with THIS step's sampler inputs and
    /// return the lightweight handle the forward followup encodes. Pure host
    /// memcpy — no Metal calls, no allocation. `any_penalty` gates the
    /// penalties' coefficients and histories; a step without them reads empty
    /// histories.
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

        // row_state: host words (temperature, top_k, top_p, min_p, uniform,
        // cap). The kernels write the GPU words (max, sum, the descent's
        // thresholds and survivors) every step before any reads them, so
        // stale words never leak into a step.
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

        // Penalties: only a step with a penalized row writes them — the
        // coefficients and the histories, row-major [njobs, max_*] padded with
        // `vocab` (never a real index). Every other step reads empty histories,
        // which leave each logit as it is. The arena's max_hist bound is a
        // worker-config fact; a longer history is a config violation, not data.
        let (max_out, max_prompt) = if params.any_penalty {
            let max_out = params.max_output_len.max(1);
            let max_prompt = params.max_prompt_len.max(1);
            assert!(
                max_out <= self.max_hist && max_prompt <= self.max_hist,
                "sampler history ({max_out}/{max_prompt}) exceeds arena max_hist ({})",
                self.max_hist
            );
            write(&self.reps_buf, &f32s(&params.rep_penalties));
            write(&self.freqs_buf, &f32s(&params.freq_penalties));
            write(&self.press_buf, &f32s(&params.pres_penalties));
            let pad = self.vocab as i32;
            let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
            let mut flat_out = vec![pad; n * max_out as usize];
            let mut flat_prompt = vec![pad; n * max_prompt as usize];
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
            write(&self.out_ids_buf, &i32s(&flat_out));
            write(&self.prompt_ids_buf, &i32s(&flat_prompt));
            (max_out, max_prompt)
        } else {
            (0, 0)
        };
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
    /// encoder (from the argmax followup): softmax reduce (the penalized f32
    /// row and its slice partials) → materialize (row stats, probabilities,
    /// descent round 0) → descent rounds 1 and 2 (each picking the round
    /// before) → count/compact (round 2's pick, strict and tied candidates) → finalize
    /// (tie quotas, sort, top-p, draw). Every stage rides its own
    /// (arena-persistent) argument table; the forward's logits and `tokens` —
    /// the step's argmax output, one per logits row, which each job's sampled
    /// token overwrites at its row — are bound first.
    pub fn encode_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        logits_addr: u64,
        tokens: &Buffer,
        kernels: &SamplerKernels,
    ) {
        use objc2_metal::MTL4ArgumentTable;
        let a = &self.arena;
        unsafe {
            a.softmax_reduce_at.setAddress_atIndex(logits_addr, 1);
            a.finalize_at.setAddress_atIndex(tokens.gpuAddress(), 5);
        }
        let (rows, sliced) = (self.njobs, self.njobs * self.nslices);
        let stages = [
            (SamplerStage::SoftmaxReduce, &a.softmax_reduce_at, sliced),
            (
                SamplerStage::SoftmaxMaterialize,
                &a.softmax_materialize_at,
                sliced,
            ),
            (SamplerStage::DescentRound1, &a.descent_at, sliced),
            (SamplerStage::DescentRound2, &a.descent_at, sliced),
            (SamplerStage::CountCompact, &a.count_compact_at, sliced),
            (SamplerStage::Finalize, &a.finalize_at, rows),
        ];
        for (stage, table, threadgroups) in stages {
            encode_sampler_stage_into_mtl4(enc, kernels.stage(stage), table, threadgroups);
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

    /// A verify step samples a request's every row: each row draws at its own position (the
    /// first at the request's generated count, as a decode step's row does), so the rows' draws
    /// are independent — one uniform on every row would correlate the drafts' acceptances.
    #[test]
    fn a_requests_rows_draw_at_their_own_positions() {
        let req_ids = vec!["a".to_string(), "b".to_string()];
        let params: std::collections::HashMap<_, _> = (req_ids.iter())
            .map(|r| (r.clone(), scratchy_core_common::SamplingParams::default()))
            .collect();
        let empty: &[u32] = &[];
        let gather = |jobs: &[(usize, u32)]| {
            let mut rngs = std::collections::HashMap::new();
            gather_gpu_sample_params(
                jobs,
                &req_ids,
                &params,
                &mut rngs,
                |_| (empty, empty),
                |_| 7,
                32,
            )
            .uniforms
        };
        let at = |req: &str, position: u32| {
            scratchy_core_common::seed_to_uniform(scratchy_core_common::fnv_seed(req, position))
        };
        // "a" verifies two drafts (rows 0..3), "b" decodes (row 3).
        let verify = gather(&[(0, 0), (0, 1), (0, 2), (1, 3)]);
        assert_eq!(verify, [at("a", 7), at("a", 8), at("a", 9), at("b", 7)]);
        // A decode step's row draws as before.
        assert_eq!(gather(&[(0, 0)]), [at("a", 7)]);
    }

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
        pending.encode_into(enc, logits_addr, &tokens, kernels);
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
    // Penalties parity: rep/freq/pres must reshape logits exactly, and shift
    // the argmax off a penalized token. Runs the pipeline's reduce stage (which
    // gathers and penalizes the row) and reads the scratch back.
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

    /// The reduce stage only, on `njobs` rows; returns the scratch it writes
    /// (the post-penalty f32 logits) for every row.
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
        // The reduce stage writes the penalized f32 logits to the scratch; the
        // rest would rewrite it, so stop after it. Bind the logits into its
        // table first (the arena's tables are already fully bound otherwise).
        use objc2_metal::MTL4ArgumentTable as _;
        unsafe {
            (arena.softmax_reduce_at).setAddress_atIndex(logits_buf.gpuAddress(), 1);
        }
        let sliced = njobs * pending.nslices;
        let reduce = kernels.stage(SamplerStage::SoftmaxReduce);
        encode_sampler_stage_into_mtl4(enc, reduce, &arena.softmax_reduce_at, sliced);
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

    // =======================================================================
    // Bit-exact sampling: the probabilities, threshold and draw the pipeline
    // forms, against cuda's `sample_top_k_top_p_core` replayed on the CPU bit
    // for bit, at the production vocab sizes and default sampling — ties and
    // near-ties at the top-k and top-p cuts included. Downstream of the
    // probabilities everything is integer selection plus the finalize's
    // sequential f32 sums, which the CPU replays exactly from the pipeline's
    // own probability row. Of the softmax the CPU replays the slice maxima, the
    // row max and, on rows whose max sits in every slice (each partial's
    // rescale is then exp(0)), the merge's reduction order; equal logits must
    // give equal probabilities. A draw that lands on tied candidates may name
    // any of them (candidates are gathered in atomic arrival order): there
    // the drawn probability must match and the token must be one of them.
    // =======================================================================

    /// What the pipeline formed for one job, read back after the run.
    struct Formed {
        nslices: usize,
        /// Per slice: (max(logit / T), sum exp(logit / T - slice max)).
        partials: Vec<[f32; 2]>,
        max: f32,
        sum: f32,
        threshold: u32,
        probs: Vec<u32>,
        token: u32,
    }

    /// One pipeline on `logits` (job j samples row j of it) with `kernels`' slicing.
    fn run_formed<T: Copy>(
        device: &Device,
        kernels: &SamplerKernels,
        params: &scratchy_core_common::GpuSampleParams,
        logits: &[T],
    ) -> Vec<Formed> {
        use objc2_metal::MTLBuffer as _;
        let n = params.row_indices.len();
        let vocab = kernels.vocab.get() as usize;
        let logits_buf = shared_slice(device, logits);
        let tokens = shared_slice(device, &vec![u32::MAX; n]);
        let batch = Mtl4DispatchBatch::begin(device).expect("an MTL4 queue");
        let res = batch.residency();
        let max_hist = params.max_output_len.max(params.max_prompt_len).max(1);
        let arena = SamplerArena::new(device, res, n as u32, kernels, max_hist);
        let pending = arena.prepare_step(params, n as u32, None);
        let pins = [res.pin(logits_buf.clone()), res.pin(tokens.clone())];
        pending.encode_into(batch.encoder(), logits_buf.gpuAddress(), &tokens, kernels);
        batch.commit(true);
        drop(pins);
        let ns = pending.nslices as usize;
        let partials = read_slice::<f32>(&arena.partials_buf, n * ns * 3);
        let state = read_slice::<u32>(&arena.row_state_buf, n * ROW_STATE_LEN);
        let probs = read_slice::<u32>(&arena.scratch_f32, n * vocab);
        let tokens = read_slice::<u32>(&tokens, n);
        (0..n)
            .map(|j| {
                let st = &state[j * ROW_STATE_LEN..];
                Formed {
                    nslices: ns,
                    partials: (partials[j * ns * 3..(j + 1) * ns * 3].chunks(3))
                        .map(|p| [p[0], p[1]])
                        .collect(),
                    max: f32::from_bits(st[1]),
                    sum: f32::from_bits(st[2]),
                    threshold: st[7],
                    probs: probs[j * vocab..(j + 1) * vocab].to_vec(),
                    token: tokens[j],
                }
            })
            .collect()
    }

    /// The sampling params of one reference case.
    #[derive(Clone, Copy)]
    struct Draw {
        top_k: i32,
        top_p: f32,
        min_p: f32,
        /// Repetition / frequency / presence penalties on the row's top tokens.
        penalty: bool,
    }

    /// cuda's selection and draw on the pipeline's probability row: the threshold
    /// (the cap-th largest bits, then the min-p floor over the row's max
    /// probability), the candidates (every strict one, then each slice's share
    /// of the tied ones, slices in order), the top-p cutoff and the draw with
    /// the finalize's sequential f32 sums. Returns the threshold, the drawn
    /// candidate's probability and the tokens that may carry it.
    fn reference_draw(f: &Formed, draw: Draw, uniform: f32) -> (u32, u32, Vec<u32>) {
        let probs = &f.probs;
        let vocab = probs.len();
        let cap = if draw.top_k > 0 {
            draw.top_k as usize
        } else {
            MAX_CANDIDATES
        }
        .min(vocab);
        assert!(
            cap <= MAX_CANDIDATES,
            "the reference keeps every strict candidate"
        );
        let mut sorted = probs.clone();
        let kth = *sorted.select_nth_unstable_by(cap - 1, |a, b| b.cmp(a)).1;
        let mut threshold = f32::from_bits(kth);
        if draw.min_p > 0.0 {
            let max_prob = f32::from_bits(*probs.iter().max().expect("a row"));
            threshold = threshold.max(draw.min_p * max_prob);
        }
        let threshold = threshold.to_bits();
        let mut cand: Vec<u32> = probs.iter().copied().filter(|&b| b > threshold).collect();
        let mut avail =
            (probs.iter().filter(|&&b| b == threshold).count()).min(cap - cand.len().min(cap));
        let mut tied = Vec::new();
        for s in 0..f.nslices {
            let lo = s * vocab / f.nslices;
            let hi = (s + 1) * vocab / f.nslices;
            let here: Vec<u32> = (lo..hi)
                .filter(|&i| probs[i] == threshold)
                .map(|i| i as u32)
                .collect();
            let take = here.len().min(avail);
            avail -= take;
            cand.extend(std::iter::repeat_n(threshold, take));
            if take > 0 {
                tied.extend(here);
            }
        }
        cand.sort_unstable_by(|a, b| b.cmp(a));
        let p = |b: u32| f32::from_bits(b);
        let mut cumsum = 0.0f32;
        let mut cutoff = cand.len();
        for (i, &b) in cand.iter().enumerate() {
            cumsum += p(b);
            if cumsum > draw.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let total = cand[..cutoff].iter().fold(0.0f32, |t, &b| t + p(b));
        let target = uniform * total;
        let mut at = cutoff - 1;
        let mut cumsum = 0.0f32;
        for (i, &b) in cand[..cutoff].iter().enumerate() {
            cumsum += p(b);
            if cumsum >= target {
                at = i;
                break;
            }
        }
        let drawn = cand[at];
        let may = if drawn == threshold {
            tied
        } else {
            (0..vocab as u32)
                .filter(|&i| probs[i as usize] == drawn)
                .collect()
        };
        (threshold, drawn, may)
    }

    /// The f32 sum of the merge's simdgroup butterfly over `v` (at most 32), zero-padded.
    fn butterfly_sum(v: &[f32]) -> f32 {
        let mut lanes = [0.0f32; 32];
        lanes[..v.len()].copy_from_slice(v);
        for offset in [16, 8, 4, 2, 1] {
            let prev = lanes;
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = prev[i] + prev[i ^ offset];
            }
        }
        lanes[0]
    }

    /// Check one job against the reference; `logits` its f32 row (`exact_logits`:
    /// unpenalized at T = 1, so the slice maxima and the ties are the logits').
    fn check_formed(
        f: &Formed,
        logits: &[f32],
        exact_logits: bool,
        draw: Draw,
        uniform: f32,
    ) -> Result<(), String> {
        let vocab = logits.len();
        let maxima: Vec<f32> = f.partials.iter().map(|p| p[0]).collect();
        if exact_logits {
            for (s, &m) in maxima.iter().enumerate() {
                let lo = s * vocab / f.nslices;
                let hi = (s + 1) * vocab / f.nslices;
                let want = logits[lo..hi]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if m.to_bits() != want.to_bits() {
                    return Err(format!("slice {s} max {m} != {want}"));
                }
            }
        }
        let max = maxima.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if f.max.to_bits() != max.to_bits() {
            return Err(format!("row max {} != {max}", f.max));
        }
        if maxima.iter().all(|&m| m == max) {
            let sums: Vec<f32> = f.partials.iter().map(|p| p[1]).collect();
            let want = butterfly_sum(&sums);
            if f.sum.to_bits() != want.to_bits() {
                return Err(format!("row sum {} != merged {want}", f.sum));
            }
        }
        let mut by_logit = std::collections::HashMap::new();
        for (l, &b) in logits.iter().zip(&f.probs).filter(|_| exact_logits) {
            if *by_logit.entry(l.to_bits()).or_insert(b) != b {
                return Err(format!("equal logits {l} gave unequal probabilities"));
            }
        }
        let (threshold, drawn, may) = reference_draw(f, draw, uniform);
        if f.threshold != threshold {
            return Err(format!("threshold {:08x} != {threshold:08x}", f.threshold));
        }
        let got = f.probs.get(f.token as usize).copied();
        if got != Some(drawn) || !may.contains(&f.token) {
            return Err(format!(
                "drew {} (prob {got:08x?}), the reference draws prob {drawn:08x} from {} token(s) {:?}",
                f.token,
                may.len(),
                &may[..may.len().min(4)]
            ));
        }
        Ok(())
    }

    /// Uniforms on the draw's cumsum boundaries of the first candidates (and one
    /// ulp either side), from a run's own probabilities.
    fn boundary_uniforms(f: &Formed, draw: Draw) -> Vec<f32> {
        let (threshold, _, _) = reference_draw(f, draw, 0.5);
        let mut cand: Vec<f32> = (f.probs.iter().copied())
            .filter(|&b| b >= threshold)
            .map(f32::from_bits)
            .collect();
        cand.sort_unstable_by(|a, b| b.total_cmp(a));
        let total: f32 = cand.iter().take(64).fold(0.0, |t, &p| t + p);
        let mut cumsum = 0.0f32;
        let mut out = Vec::new();
        for &p in cand.iter().take(6) {
            cumsum += p;
            let u = cumsum / total;
            for b in [u.to_bits() - 1, u.to_bits(), u.to_bits() + 1] {
                if (0.0..1.0).contains(&f32::from_bits(b)) {
                    out.push(f32::from_bits(b));
                }
            }
        }
        out
    }

    /// A logits row of `kind`, each value representable in `bf16` when `quant`.
    fn reference_row(kind: usize, vocab: usize, k: usize, seed: u64, quant: bool) -> Vec<f32> {
        let mut rng = Rng(seed ^ ((kind as u64) << 40) ^ vocab as u64 ^ ((k as u64) << 20));
        let q = |x: f32| {
            if quant {
                half::bf16::from_f32(x).to_f32()
            } else {
                x
            }
        };
        let up = |x: f32| {
            if quant {
                half::bf16::from_bits(half::bf16::from_f32(x).to_bits() + 1).to_f32()
            } else {
                f32::from_bits(x.to_bits() + 1)
            }
        };
        let mut v: Vec<f32> = (0..vocab).map(|_| q(rng.range(-15.0, 0.0))).collect();
        let mut at = |m: usize| -> Vec<usize> { (0..m).map(|_| rng.usize(0, vocab)).collect() };
        match kind {
            // A strong token over noise with a tail of mid tokens.
            0 => {
                for (j, i) in at(64).into_iter().enumerate() {
                    v[i] = q(if j == 0 { 20.0 } else { 6.0 + j as f32 * 0.07 });
                }
            }
            // k - 3 distinct tops, then 8 tied at the cut (more tied than slots).
            1 => {
                for (j, i) in at(k - 3).into_iter().enumerate() {
                    v[i] = q(10.0 + j as f32 * 0.25);
                }
                for i in at(8) {
                    v[i] = q(9.0);
                }
            }
            // As 1, the 8 at the cut alternating 9 and one ulp above it.
            2 => {
                for (j, i) in at(k - 3).into_iter().enumerate() {
                    v[i] = q(10.0 + j as f32 * 0.25);
                }
                for (j, i) in at(8).into_iter().enumerate() {
                    v[i] = if j % 2 == 0 { q(9.0) } else { up(q(9.0)) };
                }
            }
            // Tied groups straddling the top-p cut.
            3 => {
                for i in at(5) {
                    v[i] = q(12.0);
                }
                for i in at(40) {
                    v[i] = q(10.0);
                }
            }
            // 200 tied at the top: no strict candidate.
            4 => {
                for i in at(200) {
                    v[i] = q(8.0);
                }
            }
            // A dense cluster one ulp apart at the top.
            5 => {
                for (j, i) in at(60).into_iter().enumerate() {
                    v[i] = [q(11.0), up(q(11.0)), up(up(q(11.0)))][j % 3];
                }
            }
            // The max in every slice (the merge rescales by exp(0)), a tail below.
            _ => {
                for (s, i) in at(64).into_iter().enumerate() {
                    v[s * (vocab / 64) + i % (vocab / 64)] = q(12.0);
                }
                for (j, i) in at(40).into_iter().enumerate() {
                    v[i] = q(11.0 - j as f32 * 0.125);
                }
            }
        }
        v
    }

    #[test]
    fn sampling_is_bit_exact_at_the_production_vocab_sizes() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();
        if Mtl4DispatchBatch::begin(&device).is_none() {
            eprintln!("skipping: no MTL4 queue");
            return;
        }
        let mut rng = Rng(0x0B17_E8AC_7000_0008);
        let mut failures = Vec::new();
        let (mut draws, mut tied) = (0usize, 0usize);
        let mut check = |f: &Formed, logits: &[f32], exact: bool, d: Draw, u: f32, tag: &str| {
            draws += 1;
            tied += usize::from(reference_draw(f, d, u).2.len() > 1);
            if let Err(e) = check_formed(f, logits, exact, d, u) {
                failures.push(format!("{tag} u={u}: {e}"));
            }
        };
        let base = |top_k| Draw {
            top_k,
            top_p: 0.95,
            min_p: 0.0,
            penalty: false,
        };
        for (vocab, dtype) in [
            (262_144usize, CastDtype::Bf16),
            (248_320, CastDtype::Bf16),
            (248_320, CastDtype::F32),
        ] {
            let kernels = baked(&device, vocab, dtype);
            let mut wide = baked(&device, vocab, dtype);
            wide.slicing = SliceTarget(64);
            let quant = dtype == CastDtype::Bf16;
            for (k, kind, seed) in (0..7).flat_map(|kind| [(64, kind, 0u64), (20, kind, 1)]) {
                let logits = reference_row(kind, vocab, k, seed, quant);
                let tag = format!("vocab {vocab} {dtype:?} top_k {k} kind {kind}");
                let draw = base(k as i32);
                let mut cases = vec![draw];
                if kind < 2 {
                    cases.push(Draw {
                        min_p: 0.05,
                        ..draw
                    });
                    cases.push(Draw { top_k: 0, ..draw });
                    cases.push(Draw {
                        penalty: true,
                        ..draw
                    });
                }
                // The penalized tokens: the row's top three, twice in the output.
                let mut top: Vec<usize> = (0..vocab).collect();
                top.select_nth_unstable_by(3, |&a, &b| logits[b].total_cmp(&logits[a]));
                let history: Vec<i32> = top[..3].iter().flat_map(|&t| [t as i32; 2]).collect();
                for d in cases {
                    let exact = !d.penalty;
                    let params = |us: &[f32]| {
                        let n = us.len();
                        let pen = |on: f32, off: f32| vec![if d.penalty { on } else { off }; n];
                        let hist = if d.penalty {
                            history.repeat(n)
                        } else {
                            Vec::new()
                        };
                        scratchy_core_common::GpuSampleParams {
                            row_indices: (0..n as u32).collect(),
                            temperatures: vec![1.0; n],
                            top_ks: vec![d.top_k; n],
                            top_ps: vec![d.top_p; n],
                            min_ps: vec![d.min_p; n],
                            uniforms: us.to_vec(),
                            rep_penalties: pen(1.3, 1.0),
                            freq_penalties: pen(0.4, 0.0),
                            pres_penalties: pen(0.2, 0.0),
                            max_output_len: if d.penalty { history.len() as u32 } else { 0 },
                            output_token_ids: hist,
                            any_penalty: d.penalty,
                            ..Default::default()
                        }
                    };
                    let run = |kernels: &SamplerKernels, us: &[f32]| -> Vec<Formed> {
                        let rows: Vec<f32> = logits.repeat(us.len());
                        match dtype {
                            CastDtype::F32 => run_formed(&device, kernels, &params(us), &rows),
                            _ => {
                                let bf: Vec<half::bf16> =
                                    rows.iter().map(|&x| half::bf16::from_f32(x)).collect();
                                run_formed(&device, kernels, &params(us), &bf)
                            }
                        }
                    };
                    // One row on the device's slices and on 32; then the cumsum
                    // boundaries as a batch (one slice a row) and four rows.
                    let u = rng.unit();
                    let one = run(&kernels, &[u]).remove(0);
                    check(&one, &logits, exact, d, u, &tag);
                    let u = rng.unit();
                    check(&run(&wide, &[u]).remove(0), &logits, exact, d, u, &tag);
                    let us = boundary_uniforms(&one, d);
                    for (f, &u) in run(&kernels, &us).iter().zip(&us) {
                        check(f, &logits, exact, d, u, &tag);
                    }
                    let us: Vec<f32> = (0..4).map(|_| rng.unit()).collect();
                    for (f, &u) in run(&kernels, &us).iter().zip(&us) {
                        check(f, &logits, exact, d, u, &tag);
                    }
                }
            }
        }
        eprintln!(
            "[bit_exact] {draws} draws, {tied} on tied candidates, {} failures",
            failures.len()
        );
        for f in failures.iter().take(12) {
            eprintln!("  {f}");
        }
        assert!(
            failures.is_empty(),
            "{} draws differ from the reference",
            failures.len()
        );
        assert!(
            tied > draws / 10 && tied < draws,
            "ties exercised: {tied} of {draws}"
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
