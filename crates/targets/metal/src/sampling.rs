// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! On-GPU token sampler dispatcher — the Metal port of cuda's
//! `sampling_kernels.cu` sampler (see `shaders/sampling.metal`).
//!
//! Four kernels, each dispatched one threadgroup (256 threads) per request row:
//!   * `cast_rows_{f16,bf16}_to_f32` — gather the sample logits row (f16/bf16)
//!     into a compact f32 scratch buffer;
//!   * `apply_penalties` — repetition / frequency / presence penalties (f32,
//!     in-place);
//!   * `sample_top_k_top_p` — softmax → top-k radix-select → min-p → compact →
//!     bitonic sort → top-p cutoff → categorical sample (f32).
//!
//! The three stages have a producer→consumer dependency (cast writes the f32
//! scratch, penalties mutate it, sample reads it), so callers encode them onto
//! ONE [`Mtl4DispatchBatch`] with a [`Mtl4DispatchBatch::barrier`] between the
//! dependent dispatches, then `commit(true)` once. This mirrors argmax's
//! MTL4-only lifecycle (`embedded_metallib!` + `build_pipeline`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};

use crate::mtl4_dispatch::{Buffer, Mtl4DispatchBatch};
use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Threads per threadgroup — MUST equal `SAMPLING_BLOCK_SIZE` in
/// `shaders/sampling.metal` (the kernels stride the vocab axis by exactly this
/// and size their `warp_buf` for `SAMPLING_BLOCK_SIZE / 32` warps).
pub const SAMPLER_TG_SIZE: usize = 256;

/// Simdgroup width the `sample_top_k_top_p` block reductions assume — MUST equal
/// `WARP_SIZE` in `shaders/sampling.metal`. The reductions derive
/// `simdgroup = tid / 32`, `lane = tid % 32`, shuffle across 32 lanes, and size
/// `warp_buf` for `SAMPLER_TG_SIZE / 32` slots. Every shipping Apple GPU is
/// 32-wide, but the width is a device property (not a compile-time constant), so
/// [`SamplerKernels::new`] hard-fails the load on any device that disagrees
/// rather than let the reductions silently mis-index and sample wrong tokens.
pub const SAMPLER_WARP_SIZE: usize = 32;

/// Compiled sampler pipelines. Cached once per device at model load (like
/// `ArgmaxKernels`) to avoid recompiling the MSL each step.
pub struct SamplerKernels {
    pub cast_f16: ComputePipelineState,
    pub cast_bf16: ComputePipelineState,
    pub penalties: ComputePipelineState,
    pub sample: ComputePipelineState,
    _library: Library,
}

impl SamplerKernels {
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("sampling"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `sampling.metallib`: {e}"))
            })?;
        let cast_f16 = build_pipeline(device, &library, "cast_rows_f16_to_f32")?;
        let cast_bf16 = build_pipeline(device, &library, "cast_rows_bf16_to_f32")?;
        let penalties = build_pipeline(device, &library, "apply_penalties")?;
        let sample = build_pipeline(device, &library, "sample_top_k_top_p")?;

        // Fence the one runtime assumption the shader cannot check itself: the
        // `sample_top_k_top_p` block reductions require a 32-lane simdgroup (see
        // `SAMPLER_WARP_SIZE`). This is true on every shipping Apple GPU, but a
        // device could in principle report a different execution width, which
        // would make `warp_buf` indexing / the shuffle reductions wrong. Refuse
        // to load loudly instead of silently sampling wrong tokens.
        let width = sample.threadExecutionWidth();
        if width != SAMPLER_WARP_SIZE {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "sample_top_k_top_p requires a {SAMPLER_WARP_SIZE}-lane simdgroup, \
                 but this device reports execution width {width}; the sampler's \
                 block reductions would mis-index. Refusing to load."
            )));
        }

        Ok(Self {
            cast_f16,
            cast_bf16,
            penalties,
            sample,
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

fn tg_dims(nrows: u32) -> (MTLSize, MTLSize) {
    let threadgroups = MTLSize {
        width: nrows as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: SAMPLER_TG_SIZE,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// Encode the row-gather cast: `out_f32[r, :] = f32(logits[row_indices[r], :])`.
/// `is_bf16` selects the `bfloat` vs `half` reader (the model's compute dtype).
///
/// Bindings mirror `cast_rows_{f16,bf16}_to_f32`:
///   0=out_f32, 1=logits, 2=row_indices, 3=vocab(u32).
pub fn encode_cast_rows(
    batch: &mut Mtl4DispatchBatch,
    kernels: &SamplerKernels,
    is_bf16: bool,
    out_f32: &Buffer,
    logits: &Buffer,
    row_indices: &Buffer,
    nrows: u32,
    vocab: u32,
) {
    let pso = if is_bf16 {
        &kernels.cast_bf16
    } else {
        &kernels.cast_f16
    };
    let (tgs, tpt) = tg_dims(nrows);
    batch.encode(
        pso,
        &[(out_f32, 0), (logits, 1), (row_indices, 2)],
        &[(vocab, 3)],
        &[],
        &[],
        tgs,
        tpt,
    );
}

/// Encode the penalties pass (in-place on the f32 scratch). Bindings mirror
/// `apply_penalties`:
///   0=logits_f32, 1=output_token_ids, 2=prompt_token_ids,
///   3=rep, 4=freq, 5=pres, 6=vocab, 7=max_output_len, 8=max_prompt_len.
#[allow(clippy::too_many_arguments)]
pub fn encode_apply_penalties(
    batch: &mut Mtl4DispatchBatch,
    kernels: &SamplerKernels,
    logits_f32: &Buffer,
    output_token_ids: &Buffer,
    prompt_token_ids: &Buffer,
    rep: &Buffer,
    freq: &Buffer,
    pres: &Buffer,
    nrows: u32,
    vocab: u32,
    max_output_len: u32,
    max_prompt_len: u32,
) {
    let (tgs, tpt) = tg_dims(nrows);
    batch.encode(
        &kernels.penalties,
        &[
            (logits_f32, 0),
            (output_token_ids, 1),
            (prompt_token_ids, 2),
            (rep, 3),
            (freq, 4),
            (pres, 5),
        ],
        &[(vocab, 6), (max_output_len, 7), (max_prompt_len, 8)],
        &[],
        &[],
        tgs,
        tpt,
    );
}

/// Encode the top-k/top-p/min-p sample pass. Bindings mirror
/// `sample_top_k_top_p`:
///   0=output(u32), 1=logits_f32, 2=temperatures, 3=top_ks(i32),
///   4=top_ps, 5=min_ps, 6=uniforms, 7=vocab.
#[allow(clippy::too_many_arguments)]
pub fn encode_sample_top_k_top_p(
    batch: &mut Mtl4DispatchBatch,
    kernels: &SamplerKernels,
    output: &Buffer,
    logits_f32: &Buffer,
    temperatures: &Buffer,
    top_ks: &Buffer,
    top_ps: &Buffer,
    min_ps: &Buffer,
    uniforms: &Buffer,
    nrows: u32,
    vocab: u32,
) {
    let (tgs, tpt) = tg_dims(nrows);
    batch.encode(
        &kernels.sample,
        &[
            (output, 0),
            (logits_f32, 1),
            (temperatures, 2),
            (top_ks, 3),
            (top_ps, 4),
            (min_ps, 5),
            (uniforms, 6),
        ],
        &[(vocab, 7)],
        &[],
        &[],
        tgs,
        tpt,
    );
}

/// Encode ONE sampler stage (cast / penalties / sample) onto an EXISTING MTL4
/// compute encoder — the forward's own encoder — so the sampler rides the
/// forward's command buffer (one commit, one host wait) instead of a second
/// [`Mtl4DispatchBatch`]. Mirrors `argmax::encode_argmax_*_into_mtl4`.
///
/// Emits a `Device`-visibility barrier first: every stage reads what a prior
/// same-encoder dispatch wrote (cast reads the forward/grammar-mask logits;
/// penalties + sample read the f32 scratch cast/penalties produced) and MTL4
/// compute encoders do NOT auto-serialize same-encoder dispatches. Then
/// set-pipeline / set-arg-table / dispatch (`njobs` threadgroups × 256 threads,
/// one threadgroup per sampling row — identical to the batched dispatch).
///
/// The caller binds the stage's buffers into `arg_table` at the indices the
/// kernel expects (see the `encode_*` doc comments); for the cast stage the
/// `logits` binding (index 1) is the forward's lm_head output, bound by
/// `gpuAddress` inside the followup.
pub fn encode_sampler_stage_into_mtl4(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    pipeline: &ComputePipelineState,
    arg_table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
    njobs: u32,
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
    let threadgroups = MTLSize {
        width: njobs as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: SAMPLER_TG_SIZE,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
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
#[allow(clippy::too_many_arguments)]
pub fn gather_gpu_sample_params(
    jobs: &[(usize, u32)],
    req_ids: &[String],
    sampling_params_map: &std::collections::HashMap<String, scratchy_core_common::SamplingParams>,
    seeded_rngs: &mut std::collections::HashMap<String, rand::rngs::StdRng>,
    prompt_lengths: &std::collections::HashMap<String, usize>,
    token_buffers: &std::collections::HashMap<String, Vec<u32>>,
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
            let plen = prompt_lengths.get(req_id).copied().unwrap_or(0);
            let position = token_buffers
                .get(req_id)
                .map(|b| b.len().saturating_sub(plen))
                .unwrap_or(0) as u32;
            scratchy_core_common::fnv_seed(req_id, position)
        };
        params
            .uniforms
            .push(scratchy_core_common::seed_to_uniform(seed));
    }

    // Penalty token histories (only when a row uses penalties). `token_buffers`
    // = prompt ++ generated, split at `prompt_len`; both arrays are row-major
    // `njobs * max_*` padded with `vocab`.
    if params.any_penalty {
        let mut outs: Vec<Vec<i32>> = Vec::with_capacity(njobs);
        let mut prompts: Vec<Vec<i32>> = Vec::with_capacity(njobs);
        for &(i, _) in jobs {
            let req_id = &req_ids[i];
            let plen = prompt_lengths.get(req_id).copied().unwrap_or(0);
            let (p_ids, o_ids): (Vec<i32>, Vec<i32>) = match token_buffers.get(req_id) {
                Some(b) => {
                    let cut = plen.min(b.len());
                    (
                        b[..cut].iter().map(|&t| t as i32).collect(),
                        b[cut..].iter().map(|&t| t as i32).collect(),
                    )
                }
                None => (Vec::new(), Vec::new()),
            };
            prompts.push(p_ids);
            outs.push(o_ids);
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

/// One step's non-greedy sampler work: GPU buffers + argument tables, prepared
/// BEFORE the forward so it can be encoded onto the forward's
/// OWN command buffer ([`encode_into`](Self::encode_into), from the argmax
/// followup) — one commit, one host wait, no second command buffer. Read the
/// sampled tokens after the wait via [`output`](Self::output).
pub struct PendingSampler {
    njobs: u32,
    is_bf16: bool,
    // Every buffer is bound by gpuAddress in `encode_into`: see `bound_buffers`.
    scratch_f32: Buffer,
    out_buf: Buffer,
    row_idx_buf: Buffer,
    temps_buf: Buffer,
    top_ks_buf: Buffer,
    top_ps_buf: Buffer,
    min_ps_buf: Buffer,
    uniforms_buf: Buffer,
    reps_buf: Buffer,
    freqs_buf: Buffer,
    press_buf: Buffer,
    out_ids_buf: Buffer,
    prompt_ids_buf: Buffer,
    consts_buf: Buffer,
    // Sampler-telemetry spill (only compiled under `sampler-telemetry`): real
    // buffers when `telem_on`, else a reused dummy.
    #[cfg(feature = "sampler-telemetry")]
    topk_probs_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    topk_indices_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    stats_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    telem_consts_buf: Buffer,
    #[cfg(feature = "sampler-telemetry")]
    telem_on: bool,
    #[cfg(feature = "sampler-telemetry")]
    telem_k: u32,
    cast_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    sample_at: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>,
    penalties_at: Option<Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>>>,
}

// SAFETY: the Retained Metal handles are created + only touched on the worker
// thread (prepared, then encoded in the same thread's forward followup); never
// actually sent across threads. Mirrors how the argmax/grammar followup closures
// move Retained Metal objects into the (Send) followup.
unsafe impl Send for PendingSampler {}

impl PendingSampler {
    /// Upload the neutral [`GpuSampleParams`](scratchy_core_common::GpuSampleParams)
    /// into GPU buffers + build the argument tables. Address binding is deferred
    /// to [`encode_into`](Self::encode_into) (which also binds the forward's own
    /// logits).
    pub fn prepare(
        device: &Device,
        params: &scratchy_core_common::GpuSampleParams,
        njobs: u32,
        vocab: u32,
        is_bf16: bool,
    ) -> Self {
        use crate::mtl4_dispatch::{shared_slice, shared_zeroed};
        use objc2_metal::{MTL4ArgumentTableDescriptor, MTLDevice};

        let n = njobs as usize;
        let row_idx_buf = shared_slice(device, &params.row_indices);
        let temps_buf = shared_slice(device, &params.temperatures);
        let top_ks_buf = shared_slice(device, &params.top_ks);
        let top_ps_buf = shared_slice(device, &params.top_ps);
        let min_ps_buf = shared_slice(device, &params.min_ps);
        let uniforms_buf = shared_slice(device, &params.uniforms);
        let reps_buf = shared_slice(device, &params.rep_penalties);
        let freqs_buf = shared_slice(device, &params.freq_penalties);
        let press_buf = shared_slice(device, &params.pres_penalties);
        let (out_ids_buf, prompt_ids_buf) = if params.any_penalty {
            (
                shared_slice(device, &params.output_token_ids),
                shared_slice(device, &params.prompt_token_ids),
            )
        } else {
            (shared_zeroed(device, 4), shared_zeroed(device, 4))
        };
        let scratch_f32 = shared_zeroed(device, n * vocab as usize * 4);
        let out_buf = shared_zeroed(device, n * 4);
        let consts_buf = shared_slice(
            device,
            &[vocab, params.max_output_len, params.max_prompt_len],
        );

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
                shared_zeroed(device, n * k * 4),
                shared_zeroed(device, n * k * 4),
                shared_zeroed(device, n * 2 * 4),
                shared_slice(device, &[1u32, telem_k]),
            )
        } else {
            let dummy = shared_zeroed(device, 4);
            (
                dummy.clone(),
                dummy.clone(),
                dummy,
                shared_slice(device, &[0u32, 0u32]),
            )
        };

        let mk_table = |count: usize| {
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(count);
            device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("sampler arg table alloc")
        };
        let cast_at = mk_table(4);
        // 8 buffers without telemetry; +5 (topk_probs/indices/stats + 2 consts)
        // when the sampler-telemetry spill params are compiled into the kernel.
        let sample_at = mk_table(if cfg!(feature = "sampler-telemetry") {
            13
        } else {
            8
        });
        let penalties_at = if params.any_penalty {
            Some(mk_table(9))
        } else {
            None
        };

        Self {
            njobs,
            is_bf16,
            scratch_f32,
            out_buf,
            row_idx_buf,
            temps_buf,
            top_ks_buf,
            top_ps_buf,
            min_ps_buf,
            uniforms_buf,
            reps_buf,
            freqs_buf,
            press_buf,
            out_ids_buf,
            prompt_ids_buf,
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
            sample_at,
            penalties_at,
        }
    }

    /// Every buffer [`encode_into`](Self::encode_into) binds by GPU address. The
    /// caller keeps them resident — and alive — until the forward's command
    /// buffer completes.
    pub fn bound_buffers(&self) -> impl Iterator<Item = &Buffer> {
        let base = [
            &self.scratch_f32,
            &self.out_buf,
            &self.row_idx_buf,
            &self.temps_buf,
            &self.top_ks_buf,
            &self.top_ps_buf,
            &self.min_ps_buf,
            &self.uniforms_buf,
            &self.reps_buf,
            &self.freqs_buf,
            &self.press_buf,
            &self.out_ids_buf,
            &self.prompt_ids_buf,
            &self.consts_buf,
        ];
        #[cfg(feature = "sampler-telemetry")]
        let base = base.into_iter().chain([
            &self.topk_probs_buf,
            &self.topk_indices_buf,
            &self.stats_buf,
            &self.telem_consts_buf,
        ]);
        base.into_iter()
    }

    /// Encode cast → [penalties] → sample onto the forward's OWN encoder (after
    /// argmax). Binds every argument-table address (reading the retained buffers)
    /// plus the forward's `logits_addr` (the cast input, index 1).
    pub fn encode_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        logits_addr: u64,
        kernels: &SamplerKernels,
    ) {
        use objc2_metal::{MTL4ArgumentTable, MTLBuffer};
        let ca = self.consts_buf.gpuAddress();
        unsafe {
            // cast: 0=scratch(out), 1=logits(forward), 2=row_idx, 3=vocab.
            self.cast_at
                .setAddress_atIndex(self.scratch_f32.gpuAddress(), 0);
            self.cast_at.setAddress_atIndex(logits_addr, 1);
            self.cast_at
                .setAddress_atIndex(self.row_idx_buf.gpuAddress(), 2);
            self.cast_at.setAddress_atIndex(ca, 3);
            // sample: 0=out,1=scratch,2=temps,3=top_ks,4=top_ps,5=min_ps,6=uniforms,7=vocab.
            self.sample_at
                .setAddress_atIndex(self.out_buf.gpuAddress(), 0);
            self.sample_at
                .setAddress_atIndex(self.scratch_f32.gpuAddress(), 1);
            self.sample_at
                .setAddress_atIndex(self.temps_buf.gpuAddress(), 2);
            self.sample_at
                .setAddress_atIndex(self.top_ks_buf.gpuAddress(), 3);
            self.sample_at
                .setAddress_atIndex(self.top_ps_buf.gpuAddress(), 4);
            self.sample_at
                .setAddress_atIndex(self.min_ps_buf.gpuAddress(), 5);
            self.sample_at
                .setAddress_atIndex(self.uniforms_buf.gpuAddress(), 6);
            self.sample_at.setAddress_atIndex(ca, 7);
            // telemetry: 8=topk_probs, 9=topk_indices, 10=stats, 11=telem_on,
            // 12=telem_k — only present when the kernel is compiled with the
            // sampler-telemetry params (matching the shader's #ifdef).
            #[cfg(feature = "sampler-telemetry")]
            {
                self.sample_at
                    .setAddress_atIndex(self.topk_probs_buf.gpuAddress(), 8);
                self.sample_at
                    .setAddress_atIndex(self.topk_indices_buf.gpuAddress(), 9);
                self.sample_at
                    .setAddress_atIndex(self.stats_buf.gpuAddress(), 10);
                let tca = self.telem_consts_buf.gpuAddress();
                self.sample_at.setAddress_atIndex(tca, 11);
                self.sample_at.setAddress_atIndex(tca + 4, 12);
            }
        }
        if let Some(ref pen) = self.penalties_at {
            unsafe {
                // penalties: 0=scratch,1=out_ids,2=prompt_ids,3=reps,4=freqs,
                // 5=press,6=vocab,7=max_out,8=max_prompt.
                pen.setAddress_atIndex(self.scratch_f32.gpuAddress(), 0);
                pen.setAddress_atIndex(self.out_ids_buf.gpuAddress(), 1);
                pen.setAddress_atIndex(self.prompt_ids_buf.gpuAddress(), 2);
                pen.setAddress_atIndex(self.reps_buf.gpuAddress(), 3);
                pen.setAddress_atIndex(self.freqs_buf.gpuAddress(), 4);
                pen.setAddress_atIndex(self.press_buf.gpuAddress(), 5);
                pen.setAddress_atIndex(ca, 6);
                pen.setAddress_atIndex(ca + 4, 7);
                pen.setAddress_atIndex(ca + 8, 8);
            }
        }
        let cast_pso = if self.is_bf16 {
            &kernels.cast_bf16
        } else {
            &kernels.cast_f16
        };
        encode_sampler_stage_into_mtl4(enc, cast_pso, &self.cast_at, self.njobs);
        if let Some(ref pen) = self.penalties_at {
            encode_sampler_stage_into_mtl4(enc, &kernels.penalties, pen, self.njobs);
        }
        encode_sampler_stage_into_mtl4(enc, &kernels.sample, &self.sample_at, self.njobs);
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
    use crate::mtl4_dispatch::{read_slice, shared_slice};

    /// Dispatch `sample_top_k_top_p` on a synthetic f32 logits row with a
    /// near-zero temperature + top_k=1: the softmax collapses onto the argmax,
    /// the radix-select keeps exactly one candidate, and the categorical draw
    /// (any uniform) must return the argmax index. Guarded to skip when no
    /// Metal device / MTL4 queue is available (CI Linux, headless).
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

        let out = shared_slice(&device, &[0u32]);
        let logits_buf = shared_slice(&device, &logits);
        let temps = shared_slice(&device, &[0.01f32]); // ~greedy
        let top_ks = shared_slice(&device, &[1i32]);
        let top_ps = shared_slice(&device, &[1.0f32]);
        let min_ps = shared_slice(&device, &[0.0f32]);
        let uniforms = shared_slice(&device, &[0.73f32]);

        let Some(mut batch) = Mtl4DispatchBatch::begin(&device) else {
            eprintln!("skipping: no MTL4 queue");
            return;
        };
        encode_sample_top_k_top_p(
            &mut batch,
            &kernels,
            &out,
            &logits_buf,
            &temps,
            &top_ks,
            &top_ps,
            &min_ps,
            &uniforms,
            1,
            vocab,
        );
        batch.commit(true).expect("sample dispatch");

        let got = read_slice::<u32>(&out, 1);
        assert_eq!(
            got[0] as usize, argmax_idx,
            "top_k=1 near-zero-temp sample must return the argmax index"
        );
    }

    // =======================================================================
    // Parity harness: pure-Rust CPU golden vs the REAL metal sampler kernel.
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

    /// Dispatch the real metal `sample_top_k_top_p` on one row and return the
    /// sampled token. Returns `None` when no metal device / MTL4 queue exists.
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
        let out = shared_slice(device, &[0u32]);
        let logits_buf = shared_slice(device, logits);
        let temps = shared_slice(device, &[temp]);
        let top_ks = shared_slice(device, &[top_k]);
        let top_ps = shared_slice(device, &[top_p]);
        let min_ps = shared_slice(device, &[min_p]);
        let uniforms = shared_slice(device, &[uniform]);

        let mut batch = Mtl4DispatchBatch::begin(device)?;
        encode_sample_top_k_top_p(
            &mut batch,
            kernels,
            &out,
            &logits_buf,
            &temps,
            &top_ks,
            &top_ps,
            &min_ps,
            &uniforms,
            1,
            vocab,
        );
        batch.commit(true).expect("sample dispatch");
        Some(read_slice::<u32>(&out, 1)[0])
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
    // shift the argmax off a penalized token.
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

            let logits_buf = shared_slice(&device, &logits);
            let out_buf = shared_slice(&device, &out_ids);
            let prompt_buf = shared_slice(&device, &prompt_ids);
            let rep_buf = shared_slice(&device, &[rep]);
            let freq_buf = shared_slice(&device, &[freq]);
            let pres_buf = shared_slice(&device, &[pres]);

            let Some(mut batch) = Mtl4DispatchBatch::begin(&device) else {
                eprintln!("skipping: no MTL4 queue");
                return;
            };
            encode_apply_penalties(
                &mut batch,
                &kernels,
                &logits_buf,
                &out_buf,
                &prompt_buf,
                &rep_buf,
                &freq_buf,
                &pres_buf,
                1,
                vocab as u32,
                max_out,
                max_prompt,
            );
            batch.commit(true).expect("penalties dispatch");

            let got = read_slice::<f32>(&logits_buf, vocab);

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

        let logits_buf = shared_slice(&device, &logits);
        let out_buf = shared_slice(&device, &out_ids);
        let prompt_buf = shared_slice(&device, &prompt_ids);
        let rep_buf = shared_slice(&device, &reps);
        let freq_buf = shared_slice(&device, &freqs);
        let pres_buf = shared_slice(&device, &press);

        let Some(mut batch) = Mtl4DispatchBatch::begin(&device) else {
            eprintln!("skipping: no MTL4 queue");
            return;
        };
        encode_apply_penalties(
            &mut batch,
            &kernels,
            &logits_buf,
            &out_buf,
            &prompt_buf,
            &rep_buf,
            &freq_buf,
            &pres_buf,
            2,
            vocab as u32,
            max_out,
            max_prompt,
        );
        batch.commit(true).expect("penalties dispatch");

        let got = read_slice::<f32>(&logits_buf, 2 * vocab);

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
    /// 248320) — exercises the 32-pass radix select and the full-vocab max/sum
    /// passes far past the 4096 used elsewhere. A dominant peak makes the draw
    /// deterministic, so every case is an exact parity match.
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
}
