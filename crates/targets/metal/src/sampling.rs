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
//! barriers between the dependent dispatches — the same MTL4-only lifecycle
//! as before (`embedded_metallib!` + `build_pipeline`), with
//! [`encode_into`] encoding the pipeline onto the forward's own encoder.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer as _, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};

use crate::mtl4_dispatch::{Buffer, shared_slice, shared_zeroed};
use crate::residency::{MetalResidencySet, Pinned};
use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

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

/// The sampler pipeline's kernels (see `shaders/sampling.metal` for the
/// dispatch order and binding table). Cached once per device at model load
/// (like `ArgmaxKernels`) to avoid recompiling the MSL each step.
pub struct SamplerKernels {
    pub cast_f16: ComputePipelineState,
    pub cast_bf16: ComputePipelineState,
    pub cast_f32: ComputePipelineState,
    pub penalties: ComputePipelineState,
    pub softmax_reduce: ComputePipelineState,
    pub stats_pick: ComputePipelineState,
    pub softmax_materialize: ComputePipelineState,
    pub histogram: ComputePipelineState,
    pub threshold_pick: ComputePipelineState,
    pub count_compact: ComputePipelineState,
    pub quota_pick: ComputePipelineState,
    pub compact_tied: ComputePipelineState,
    pub finalize: ComputePipelineState,
    _library: Library,
}

impl SamplerKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("sampling"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load `sampling.metallib`: {e:?}"
                ))
            })?;
        let cast_f16 = build_pipeline(device, &library, "cast_rows_f16_to_f32")?;
        let cast_bf16 = build_pipeline(device, &library, "cast_rows_bf16_to_f32")?;
        let cast_f32 = build_pipeline(device, &library, "cast_rows_f32_to_f32")?;
        let penalties = build_pipeline(device, &library, "apply_penalties")?;
        let softmax_reduce = build_pipeline(device, &library, "sample_softmax_reduce")?;
        let stats_pick = build_pipeline(device, &library, "sample_stats_pick")?;
        let softmax_materialize = build_pipeline(device, &library, "sample_softmax_materialize")?;
        let histogram = build_pipeline(device, &library, "sample_histogram_pass")?;
        let threshold_pick = build_pipeline(device, &library, "sample_threshold_pick")?;
        let count_compact = build_pipeline(device, &library, "sample_count_compact")?;
        let quota_pick = build_pipeline(device, &library, "sample_quota_pick")?;
        let compact_tied = build_pipeline(device, &library, "sample_compact_tied")?;
        let finalize = build_pipeline(device, &library, "sample_finalize")?;

        // Fence the one runtime assumption the shaders cannot check themselves:
        // the block reductions require a 32-lane simdgroup (see
        // `SAMPLER_WARP_SIZE`). This is true on every shipping Apple GPU, but a
        // device could in principle report a different execution width, which
        // would make `warp_buf` indexing / the shuffle reductions wrong. Refuse
        // to load loudly instead of silently sampling wrong tokens.
        let width = softmax_reduce.threadExecutionWidth();
        if width != SAMPLER_WARP_SIZE {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "the sampler's block reductions require a {SAMPLER_WARP_SIZE}-lane simdgroup, \
                 but this device reports execution width {width}; they would mis-index. \
                 Refusing to load."
            )));
        }

        Ok(Self {
            cast_f16,
            cast_bf16,
            cast_f32,
            penalties,
            softmax_reduce,
            stats_pick,
            softmax_materialize,
            histogram,
            threshold_pick,
            count_compact,
            quota_pick,
            compact_tied,
            finalize,
            _library: library,
        })
    }
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
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
/// `(req_id, generated-token-count)`. `cuda_worker` derives its seed the same
/// way, so an unseeded request gets the identical uniform on either backend.
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
            let position = history(req_id).1.len() as u32;
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

/// One step's non-greedy sampler work: GPU buffers + the ONE argument table
/// every pipeline stage shares, prepared BEFORE the forward so it can be
/// encoded onto the forward's OWN command buffer
/// ([`encode_into`](Self::encode_into), from the argmax followup) — one
/// commit, one host wait, no second command buffer. Read the sampled tokens
/// after the wait via [`output`](Self::output).
pub struct PendingSampler {
    njobs: u32,
    nslices: u32,
    cast_dtype: CastDtype,
    // Every buffer is bound by gpuAddress in `encode_into`, so each is pinned
    // for as long as this lives: keep it until the forward's host wait.
    scratch_f32: Pinned,    // slot 0: f32 logits → prob bits, [nrows, vocab]
    out_buf: Pinned,        // slot 13: sampled token ids
    row_idx_buf: Pinned,    // slot 2
    out_ids_buf: Pinned,    // slot 3 (penalties; dummy when none)
    prompt_ids_buf: Pinned, // slot 4
    reps_buf: Pinned,       // slot 5
    freqs_buf: Pinned,      // slot 6
    press_buf: Pinned,      // slot 7
    row_state_buf: Pinned,  // slot 8
    partials_buf: Pinned,   // slot 9
    hist_buf: Pinned,       // slot 10
    counts_buf: Pinned,     // slot 11
    staging_buf: Pinned,    // slot 12
    consts_buf: Pinned,     // slot 14: (vocab, nslices, nrows, max_out, max_prompt)
    // Sampler-telemetry spill (only compiled under `sampler-telemetry`): real
    // buffers when `telem_on`, else a reused dummy.
    #[cfg(feature = "sampler-telemetry")]
    topk_probs_buf: Pinned,
    #[cfg(feature = "sampler-telemetry")]
    topk_indices_buf: Pinned,
    #[cfg(feature = "sampler-telemetry")]
    stats_buf: Pinned,
    #[cfg(feature = "sampler-telemetry")]
    telem_consts_buf: Pinned,
    #[cfg(feature = "sampler-telemetry")]
    telem_on: bool,
    #[cfg(feature = "sampler-telemetry")]
    telem_k: u32,
    // Per-stage argument tables: MTL4 binds buffer attributes by signature
    // position (the old sampler's kernels used contiguous 0..N attributes and
    // one table; these kernels keep that convention per stage, so each stage
    // has its own table). The cast's table is the one whose slot 1
    // (the forward's logits) [`encode_into`](Self::encode_into) rebinds.
    cast_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    penalties_at: Option<Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>>,
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
// thread (prepared, then encoded in the same thread's forward followup); never
// actually sent across threads. Mirrors how the argmax/grammar followup closures
// move Retained Metal objects into the (Send) followup.
unsafe impl Send for PendingSampler {}

/// Vocab slices the pipeline cuts a row into: enough threadgroups to occupy
/// the GPU's cores (two per core), without staging memory exploding at large
/// batch — capped at 32 (the per-slice staging is `2 * MAX_CANDIDATES` u32
/// per row, and the one-threadgroup-per-row kernels' serial walks scale with
/// slice count).
fn nslices_for(device: &Device, nrows: u32) -> u32 {
    let cores = crate::device::gpu_cores(device).map(|c| c.0).unwrap_or(8);
    let target = (cores * 2).max(8);
    // Small batches slice hard; a full batch of rows already fills the GPU.
    (target / nrows.max(1)).clamp(1, 32)
}

impl PendingSampler {
    /// Upload the neutral [`GpuSampleParams`](scratchy_core_common::GpuSampleParams)
    /// into GPU buffers pinned in `residency` (committed) + build the shared
    /// argument table. Address binding is deferred to
    /// [`encode_into`](Self::encode_into) (which also binds the forward's own
    /// logits).
    pub fn prepare(
        device: &Device,
        residency: &MetalResidencySet,
        params: &scratchy_core_common::GpuSampleParams,
        njobs: u32,
        vocab: u32,
        cast_dtype: CastDtype,
    ) -> Self {
        let pin = |buffer| residency.pin(buffer);
        let n = njobs as usize;
        let nslices = nslices_for(device, njobs);

        let row_idx_buf = pin(shared_slice(device, &params.row_indices));
        let (out_ids_buf, prompt_ids_buf) = if params.any_penalty {
            (
                pin(shared_slice(device, &params.output_token_ids)),
                pin(shared_slice(device, &params.prompt_token_ids)),
            )
        } else {
            (pin(shared_zeroed(device, 4)), pin(shared_zeroed(device, 4)))
        };
        let reps_buf = pin(shared_slice(device, &params.rep_penalties));
        let freqs_buf = pin(shared_slice(device, &params.freq_penalties));
        let press_buf = pin(shared_slice(device, &params.pres_penalties));

        // row_state: the pipeline's per-row decision block. Host words:
        // temperature, top_k, top_p, min_p, uniform, cap. GPU words (max, sum,
        // threshold, round, counts) start zeroed and are (re)written by the
        // kernels each dispatch.
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
                (params.top_ks[r] as u32).min(vocab)
            } else {
                1024u32.min(vocab)
            };
            s[8] = k;
        }
        let row_state_buf = pin(shared_slice(device, &row_state));

        // Element counts × 4: these are u32 arrays, and `shared_zeroed`
        // takes a byte length.
        let partials_buf = pin(shared_zeroed(device, n * nslices as usize * 3 * 4));
        let hist_buf = pin(shared_zeroed(device, n * nslices as usize * 256 * 4));
        let counts_buf = pin(shared_zeroed(device, n * nslices as usize * 4 * 4));
        let staging_buf = pin(shared_zeroed(device, n * nslices as usize * 2 * 1024 * 4));
        let scratch_f32 = pin(shared_zeroed(device, n * vocab as usize * 4));
        let out_buf = pin(shared_zeroed(device, n * 4));
        let consts_buf = pin(shared_slice(
            device,
            &[
                vocab,
                nslices,
                njobs,
                params.max_output_len,
                params.max_prompt_len,
            ],
        ));

        // Sampler telemetry: spill the sorted top-K + confidence/entropy only
        // when a consumer is watching (decided once here, honored at readback so
        // a mid-step toggle can't desync). When not watching, one reused dummy
        // backs the (never-read) spill buffers + a zeroed consts the shader reads.
        #[cfg(feature = "sampler-telemetry")]
        let telem_on =
            scratchy_core_common::sampler_telemetry::SamplerTelemetry::global().is_enabled();
        #[cfg(feature = "sampler-telemetry")]
        let telem_k: u32 = if telem_on { SAMPLER_TELEM_K } else { 0 };
        #[cfg(feature = "sampler-telemetry")]
        let (topk_probs_buf, topk_indices_buf, stats_buf, telem_consts_buf) = if telem_on {
            let k = telem_k as usize;
            (
                pin(shared_zeroed(device, n * k * 4)),
                pin(shared_zeroed(device, n * k * 4)),
                pin(shared_zeroed(device, n * 2 * 4)),
                pin(shared_slice(device, &[1u32, telem_k])),
            )
        } else {
            let dummy = shared_zeroed(device, 4);
            (
                pin(dummy.clone()),
                pin(dummy.clone()),
                pin(dummy),
                pin(shared_slice(device, &[0u32, 0u32])),
            )
        };
        residency.commit();

        // Per-stage argument tables: one per kernel, sized to its contiguous
        // signature positions (0..k-1) and gap-filled with a zero buffer so an
        // unused slot can never hold a stale address. Every real binding is
        // deferred to `encode_into` (which also binds the forward's logits),
        // matching main's deferred-address pattern.
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
        // Buffer-index layouts per kernel (bound in `encode_into`):
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
        let penalties_at = if params.any_penalty {
            Some(mk_table(7))
        } else {
            None
        };
        let softmax_reduce_at = mk_table(4);
        let stats_pick_at = mk_table(3);
        let softmax_materialize_at = mk_table(4);
        let histogram_at = mk_table(4);
        let threshold_pick_at = mk_table(3);
        let count_compact_at = mk_table(5);
        let quota_pick_at = mk_table(3);
        let compact_tied_at = mk_table(5);
        let finalize_at = mk_table(11);

        Self {
            njobs,
            nslices,
            cast_dtype,
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
            #[cfg(feature = "sampler-telemetry")]
            topk_probs_buf,
            #[cfg(feature = "sampler-telemetry")]
            topk_indices_buf,
            #[cfg(feature = "sampler-telemetry")]
            stats_buf,
            #[cfg(feature = "sampler-telemetry")]
            telem_consts_buf,
            #[cfg(feature = "sampler-telemetry")]
            telem_on,
            #[cfg(feature = "sampler-telemetry")]
            telem_k,
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
        }
    }

    /// Bind every stage's buffers into its argument table (the tables were
    /// gap-filled with a zero buffer at prepare time; only the forward's own
    /// logits address is new per encode). Idempotent — rebinding the same
    /// addresses is a no-op.
    fn bind_stage_tables(&self, logits_addr: u64) {
        use objc2_metal::{MTL4ArgumentTable, MTLBuffer};
        let scratch = self.scratch_f32.gpuAddress();
        let partials = self.partials_buf.gpuAddress();
        let row_state = self.row_state_buf.gpuAddress();
        let hist = self.hist_buf.gpuAddress();
        let counts = self.counts_buf.gpuAddress();
        let staging = self.staging_buf.gpuAddress();
        let consts = self.consts_buf.gpuAddress();
        unsafe {
            self.cast_at.setAddress_atIndex(scratch, 0);
            self.cast_at.setAddress_atIndex(logits_addr, 1);
            self.cast_at
                .setAddress_atIndex(self.row_idx_buf.gpuAddress(), 2);
            self.cast_at.setAddress_atIndex(consts, 3);
            if let Some(ref pen) = self.penalties_at {
                pen.setAddress_atIndex(scratch, 0);
                pen.setAddress_atIndex(self.out_ids_buf.gpuAddress(), 1);
                pen.setAddress_atIndex(self.prompt_ids_buf.gpuAddress(), 2);
                pen.setAddress_atIndex(self.reps_buf.gpuAddress(), 3);
                pen.setAddress_atIndex(self.freqs_buf.gpuAddress(), 4);
                pen.setAddress_atIndex(self.press_buf.gpuAddress(), 5);
                pen.setAddress_atIndex(consts, 6);
            }
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

    /// Encode the whole sample pipeline onto the forward's OWN MTL4 compute
    /// encoder (from the argmax followup): cast (and penalties, when any) →
    /// softmax reduce → stats pick → materialize → 4 × (histogram, threshold
    /// pick) → count/compact → quota pick → compact tied → finalize. Every
    /// stage rides its own argument table; the forward's logits buffer is
    /// bound into the cast's table (slot 1) first.
    pub fn encode_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        logits_addr: u64,
        kernels: &SamplerKernels,
    ) {
        let rows = self.njobs;
        let sliced = rows * self.nslices;
        self.bind_stage_tables(logits_addr);

        let stage =
            |pso: &ComputePipelineState,
             table: &Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
             tgs: u32| { encode_sampler_stage_into_mtl4(enc, pso, table, tgs) };

        // Cast (and penalties) — only the cast's dtype differs.
        let cast = match self.cast_dtype {
            CastDtype::Bf16 => &kernels.cast_bf16,
            CastDtype::F16 => &kernels.cast_f16,
            CastDtype::F32 => &kernels.cast_f32,
        };
        stage(cast, &self.cast_at, sliced);
        if let Some(ref pen) = self.penalties_at {
            stage(&kernels.penalties, pen, sliced);
        }

        // Softmax: per-slice partials, merged to row stats, materialized.
        stage(&kernels.softmax_reduce, &self.softmax_reduce_at, sliced);
        stage(&kernels.stats_pick, &self.stats_pick_at, rows);
        stage(
            &kernels.softmax_materialize,
            &self.softmax_materialize_at,
            sliced,
        );

        // Byte-histogram descent: four rounds.
        for _ in 0..4 {
            stage(&kernels.histogram, &self.histogram_at, sliced);
            stage(&kernels.threshold_pick, &self.threshold_pick_at, rows);
        }

        // Compaction: strict candidates + counts, tie quotas, tied candidates.
        stage(&kernels.count_compact, &self.count_compact_at, sliced);
        stage(&kernels.quota_pick, &self.quota_pick_at, rows);
        stage(&kernels.compact_tied, &self.compact_tied_at, sliced);

        // Sort + top-p + draw.
        stage(&kernels.finalize, &self.finalize_at, rows);
    }

    /// The sampled-token output buffer + row count, for reading back after the
    /// forward's single host wait (the sampler rode the forward CB).
    pub fn output(&self) -> (Buffer, u32) {
        (self.out_buf.clone(), self.njobs)
    }

    /// Telemetry spill buffers `(topk_probs, topk_indices, stats, njobs, k)` for
    /// reading back after the forward's host wait — `None` when telemetry was
    /// off at prepare time (so the readback matches what the kernel actually
    /// wrote). `stats` holds `[max_prob, entropy_nats]` per row; `topk_*` hold
    /// `k` entries per row, descending by prob (prob 0.0 = padding past the
    /// candidate count).
    #[cfg(feature = "sampler-telemetry")]
    pub fn telemetry_output(&self) -> Option<(Buffer, Buffer, Buffer, u32, u32)> {
        if !self.telem_on {
            return None;
        }
        Some((
            self.topk_probs_buf.clone(),
            self.topk_indices_buf.clone(),
            self.stats_buf.clone(),
            self.njobs,
            self.telem_k,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtl4_dispatch::{Mtl4DispatchBatch, read_slice, shared_slice};

    /// Run the full pipeline on `logits` (one row) and return the sampled
    /// token — the parity harness's metal side. Returns `None` when no metal
    /// device / MTL4 queue exists.
    #[allow(clippy::too_many_arguments)]
    fn run_metal_sample(
        device: &Device,
        kernels: &SamplerKernels,
        logits: &[f32],
        temp: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        uniform: f32,
    ) -> Option<u32> {
        let vocab = logits.len() as u32;
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
        run_pipeline(device, kernels, &params, logits, CastDtype::F32, 1, vocab).map(|(t, _)| t)
    }

    /// Prepare + run the pipeline on one command buffer; read back the sampled
    /// token (row 0) and the commit-wait time. `logits` is the TOTAL logits
    /// buffer, of `dtype`; `row_indices` (in `params`) picks the row(s).
    fn run_pipeline<T: Copy>(
        device: &Device,
        kernels: &SamplerKernels,
        params: &scratchy_core_common::GpuSampleParams,
        logits: &[T],
        dtype: CastDtype,
        njobs: u32,
        vocab: u32,
    ) -> Option<(u32, std::time::Duration)> {
        let logits_buf = shared_slice(device, logits);
        let batch = Mtl4DispatchBatch::begin(device)?;
        // The sampler's buffers AND the logits row must be resident for THIS
        // command buffer: prepare against the batch's own set and pin the
        // logits into it too (the batch's commit attaches exactly that set,
        // and `pending`'s pins keep the buffers in it until after the host
        // wait; the logits pin lives to the end of this scope).
        let (pending, logits_pin) = {
            let res = batch.residency();
            let pending = PendingSampler::prepare(device, res, params, njobs, vocab, dtype);
            (pending, res.pin(logits_buf.clone()))
        };
        use objc2_metal::MTLBuffer as _;
        let logits_addr = logits_buf.gpuAddress();
        let enc = batch.encoder();
        pending.encode_into(enc, logits_addr, kernels);
        let t0 = std::time::Instant::now();
        batch.commit(true);
        let wait = t0.elapsed();
        drop(logits_pin);
        let (out, n) = pending.output();
        assert_eq!(n, njobs);
        Some((read_slice::<u32>(&out, 1)[0], wait))
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
        let kernels = match SamplerKernels::new(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: sampler kernels build failed: {e:?}");
                return;
            }
        };

        // Row of 4096 logits; index 1234 is the clear maximum.
        let vocab: u32 = 4096;
        let argmax_idx: usize = 1234;
        let mut logits = vec![0.1f32; vocab as usize];
        logits[argmax_idx] = 9.0;
        logits[7] = 3.0;
        logits[42] = 2.0;

        let got = run_metal_sample(&device, &kernels, &logits, 0.01, 1, 1.0, 0.0, 0.73);
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
        let kernels = match SamplerKernels::new(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: sampler kernels build failed: {e:?}");
                return;
            }
        };
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
            let Some(got) = run_metal_sample(
                &device, &kernels, &logits, temp, top_k, top_p, min_p, uniform,
            ) else {
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
        kernels: &SamplerKernels,
        params: &scratchy_core_common::GpuSampleParams,
        logits: &[f32],
        njobs: u32,
        vocab: u32,
    ) -> Option<Vec<f32>> {
        use objc2_metal::MTLBuffer as _;
        let logits_buf = shared_slice(device, logits);
        let batch = Mtl4DispatchBatch::begin(device)?;
        let (pending, _logits_pin) = {
            let res = batch.residency();
            let pending =
                PendingSampler::prepare(device, res, params, njobs, vocab, CastDtype::F32);
            (pending, res.pin(logits_buf.clone()))
        };
        let enc = batch.encoder();
        // The cast + penalties stages are the pipeline's first two; the rest
        // would consume/rewrite the scratch, so stop after penalties.
        let sliced = njobs * pending.nslices;
        pending.bind_stage_tables(logits_buf.gpuAddress());
        let cast = match pending.cast_dtype {
            CastDtype::Bf16 => &kernels.cast_bf16,
            CastDtype::F16 => &kernels.cast_f16,
            CastDtype::F32 => &kernels.cast_f32,
        };
        encode_sampler_stage_into_mtl4(enc, cast, &pending.cast_at, sliced);
        if let Some(ref pen) = pending.penalties_at {
            encode_sampler_stage_into_mtl4(enc, &kernels.penalties, pen, sliced);
        }
        batch.commit(true);
        use std::ops::Deref;
        let scratch = pending.scratch_f32.deref();
        Some(read_slice::<f32>(scratch, njobs as usize * vocab as usize))
    }

    #[test]
    fn penalties_parity_vs_cpu_golden() {
        let Some(device) = crate::device::detect_device() else {
            eprintln!("skipping: no metal device");
            return;
        };
        let device = device.device.clone();
        let kernels = match SamplerKernels::new(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: sampler kernels build failed: {e:?}");
                return;
            }
        };

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
            let Some(got) = run_penalties(&device, &kernels, &params, &logits, 1, vocab as u32)
            else {
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
        let kernels = match SamplerKernels::new(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: sampler kernels build failed: {e:?}");
                return;
            }
        };

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

        let Some(got) = run_penalties(&device, &kernels, &params, &logits, 2, vocab as u32) else {
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
        let kernels = match SamplerKernels::new(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: sampler kernels build failed: {e:?}");
                return;
            }
        };
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
                let Some(got) = run_metal_sample(
                    &device, &kernels, &logits, temp, top_k, top_p, min_p, uniform,
                ) else {
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
        let kernels = match SamplerKernels::new(&device) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping: sampler kernels build failed: {e:?}");
                return;
            }
        };

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
                let Some((_, wait)) = run_pipeline(
                    &device,
                    &kernels,
                    &params,
                    &logits,
                    CastDtype::Bf16,
                    njobs,
                    vocab,
                ) else {
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
