// SPDX-License-Identifier: Apache-2.0
//! `MetalWorkerPool::forward()` types — Phase 5.E.
//!
//! `forward()` itself lives on `MetalWorkerPool` (see [`super::pool`])
//! because it owns checkout/checkin around the encoder lifecycle. The
//! types in this module — [`ForwardInputs`] and [`ForwardError`] — are
//! the user-facing surface of that method, kept separate so `pool.rs`
//! stays focused on the growable/capped checkout machinery.
//!
//! The per-forward path on the
//! Metal side is the hot path: bucket pick → checkout → bind inputs →
//! the MTL4 compute encoder → checkin. Bindings hit the GPU via the
//! buffer pointers baked into the dispatch at recording time (Option B);
//! the engine refills the runtime buffers' `contents()` each call.
//! [`ForwardInputs`] is the "data the engine wants written into those
//! buffers" — the forward path validates each present slice against
//! the worker's per-buffer capacity and copies the bytes through.

use crate::interpreter::metal::__re::MTLCommandBufferStatus;

use super::worker::WorkerError;
use crate::tape::ids::MaxBlocksPerSeq;

/// One forward step's runtime inputs.
///
/// The bucket selected by [`super::pool::MetalWorkerPool::pick_bucket`]
/// determines which slices the active tape's commands consume. Buckets
/// that don't reference a particular runtime buffer (e.g. a decode
/// bucket has no `cu_seqlens_q`) accept `None` for that field — the
/// forward path only copies fields the caller provides. Validation
/// against the worker's runtime buffer sizes happens before any GPU
/// work is submitted, so a malformed input never partially executes.
///
/// `num_tokens` is the *real* token count for this step. It selects
/// the bucket (smallest `bucket_m >= num_tokens`) and bounds how many
/// elements of each per-token array (`input_ids`, `positions`,
/// `slot_mapping`) are meaningful. The shader still processes
/// `bucket_m` work units — anything past `num_tokens` is padding the
/// worker fills with whatever was last in the runtime buffer; the
/// model owns interpreting the result for `[0, num_tokens)` only.
pub struct ForwardInputs<'a> {
    /// Number of real tokens this step processes.
    pub num_tokens: u32,
    /// `[num_tokens]` u32 — token ids to embed.
    pub input_ids: &'a [u32],
    /// `[num_tokens]` u32 — RoPE position per token.
    pub positions: &'a [u32],
    /// Per-KV-cache-GROUP `[num_tokens]` u32 slot_mappings (vLLM hybrid
    /// layout). One entry per group in group order (group 0 = full); empty
    /// for buckets that don't append KV. Uniform models pass a single-element
    /// vec (byte-identical to the pre-hybrid `slot_mapping`); gemma4 SWA passes
    /// `1 + N` (full encoded with `GLOBAL_BLOCK_SIZE`, sliding with
    /// `BLOCK_SIZE`). The pool writes `slot_mappings[g]` into
    /// `RuntimeBindings::slot_mappings[g]`.
    pub slot_mappings: Vec<&'a [u32]>,
    /// `[batch+1]` u32 — prefill-only cumulative sequence boundaries.
    /// `None` for decode buckets.
    pub cu_seqlens_q: Option<&'a [u32]>,
    /// `[batch]` u32 — current K-axis used length per sequence.
    /// Required for decode buckets (read by `AttentionViaCache`);
    /// `None` for first-token-only prefill.
    pub seq_used_k: Option<&'a [u32]>,
    /// `[num_logical_blocks]` u32 — per-block SPAN LABEL for block-diagonal
    /// span attention (`0` = shared/query, `k+1` = k-th Relocatable span).
    /// `None` ⇒ no spans ⇒ the kernel's span buffer stays zeroed (mask inert).
    pub span_ids: Option<&'a [u32]>,
    /// Per-KV-cache-GROUP `[batch, max_blocks]` u32 block tables (group
    /// order; group 0 = full). Empty when no paged read. Same single-vs-N
    /// contract as [`Self::slot_mappings`].
    pub block_tables: Vec<&'a [u32]>,
    /// `true` when this forward is a spec-decode verify batch (one
    /// or more reqs carries `spec_token_ids`). Currently informational —
    /// the lm_head slice trio reads `last_token_indices` for its row
    /// list whether or not spec-decode is active, so this flag has no
    /// effect on slice dispatch today. Retained for callers that pass
    /// it; may be wired to additional gates in the future.
    pub has_spec_tokens: bool,
    /// `[num_sample_rows]` u32 — per-sample-row source index into the
    /// `[num_tokens, hidden]` lm_head input. Mirrors Python vLLM's
    /// `logits_indices = query_start_loc[1:] - 1` and the CUDA path's
    /// `ForwardCtx.last_token_indices`. When `Some`, the lm_head slice
    /// trio gathers these rows, runs the GEMM at `M = len`, scatters
    /// back. `None` falls back to the full `M = bucket_m` lm_head
    /// GEMM (used by chunked-prefill intermediate chunks that produce
    /// no sampled tokens).
    pub last_token_indices: Option<&'a [u32]>,
    /// `[num_seqs]` i32 — GDN state-pool slot id per batched sequence
    /// (cu_seqlens order). `None` for non-hybrid arches; required for any
    /// bucket that runs `Instruction::GatedDeltaNet`.
    pub gdn_state_indices: Option<&'a [i32]>,
    /// `[num_seqs]` u32 — 1 when the sequence is on its first (fresh)
    /// forward (GDN kernels treat its state as zero). `None` for
    /// non-hybrid arches.
    pub gdn_is_fresh: Option<&'a [u32]>,
    /// Vision 2D-RoPE angle table (`freqs`, f32) as raw bytes. `None`
    /// for non-vision arches; required for any bucket that runs
    /// `Instruction::VisionRope`. Carried as bytes so the worker copies
    /// it verbatim into `RuntimeBindings::vision_rope_freqs` regardless
    /// of element dtype.
    pub vision_rope_freqs: Option<&'a [u8]>,
    /// Qwen2.5-VL windowed attention: per-image cu_seqlens for the
    /// FULL-attention layers (i32 bytes, `cu_seqlens_kind = 1`).
    /// `None` for non-windowed towers.
    pub vision_cu_seqlens_full: Option<&'a [u8]>,
    /// Qwen2.5-VL windowed attention: per-window cu_seqlens for the
    /// window-attention layers (i32 bytes, `cu_seqlens_kind = 2`).
    pub vision_cu_seqlens_window: Option<&'a [u8]>,
    /// Qwen2.5-VL: merged-row window permutation (u32 bytes) read by
    /// `Instruction::EmbeddingGather(kind = 0)`.
    pub vision_window_index: Option<&'a [u8]>,
    /// Qwen2.5-VL: inverse permutation (u32 bytes) read by
    /// `Instruction::EmbeddingGather(kind = 1)`.
    pub vision_reverse_indices: Option<&'a [u8]>,
    /// SigLIP learned positional-embedding indices (u32 bytes) read by
    /// `Instruction::PosEmbed` — `[0..vision_num_positions]` per image.
    /// `None` for towers without a `pos_embed(...)` gather.
    pub vision_position_ids: Option<&'a [u8]>,
    /// Vision patch pixel rows (`[num_tokens, vision_in_features]`,
    /// model dtype) as raw bytes. `None` for non-vision arches;
    /// required for any bucket that runs `LoadRows(Pixels)`.
    pub pixels: Option<&'a [u8]>,
    /// Qwen3.5-VL host-interpolated learned positional embedding
    /// (`[num_tokens, vision_embed_dim]`, model dtype) as raw bytes.
    /// `None` for non-vision arches and towers without a learned
    /// positional embedding; required for any bucket that runs
    /// `LoadRows(PosEmbeds)`. (Field name matches the macro's
    /// emitted `ForwardInputs { .., pos_embeds }` shorthand, which reads
    /// `ctx.pos_embeds`; the worker copies it into
    /// `RuntimeBindings::vision_pos_embeds`.)
    pub pos_embeds: Option<&'a [u8]>,
    /// A target model's final hidden states (`[num_tokens, hidden_size]`, model dtype) as raw
    /// bytes. `None` for every forward but an MTP head's; required for any bucket that runs
    /// `LoadRows(TargetHidden)`.
    pub target_hidden: Option<&'a [u8]>,
    /// Projected vision embeddings (`[total_mm, hidden]`, model dtype) as
    /// raw bytes, for the multimodal splice. `None` for text-only
    /// forwards / non-MM arches.
    pub mm_embeds: Option<&'a [u8]>,
    /// Per-`mm_embeds`-row destination text-embedding row (`u32::MAX` =
    /// skip), `[num_tokens]`. Built by the worker from `embed_patches`;
    /// `None` for arches without the splice.
    pub mm_dst_rows: Option<&'a [u32]>,
    /// MRoPE per-token cos/sin override table (`[num_tokens, ROT_DIM]`,
    /// model dtype) as raw bytes. `Some` only on MRoPE text decoders
    /// (`W::MROPE_SECTION.is_some()`), where the macro forward builds it
    /// from the `[3, num_tokens]` positions each forward and the worker
    /// copies it into `RuntimeBindings::mrope_cos_sin` (the rope kernel
    /// reads it in place of the static cos/sin cache, with identity
    /// positions). `None` for 1D-rope arches.
    pub mrope_cos_sin: Option<&'a [u8]>,
    /// `Some`: the host does not wait for this forward ([`Deferral`]).
    pub deferred: Option<&'a Deferral>,
}

/// A forward the host commits without waiting for it, so the next one is queued while it runs. Its
/// runtime-input writes are made on the device at the head of its command buffer — the forward
/// before it may still be reading those buffers — and it starts once that forward is done.
#[derive(Default)]
pub struct Deferral {
    /// Input tokens an earlier forward wrote on the device, copied into `input_ids` before this one
    /// reads them.
    pub device_inputs: Vec<DeviceInput>,
    /// Bytes for buffers outside the runtime inputs that an earlier command buffer may still be
    /// reading (the sampler's per-step parameters), written on the device like the runtime inputs.
    pub host_writes: Vec<(super::__re::Buffer, Vec<u8>)>,
    in_flight: std::sync::OnceLock<InFlight>,
}

impl Deferral {
    /// A deferred forward that reads `device_inputs`.
    pub fn new(device_inputs: Vec<DeviceInput>) -> Self {
        Self {
            device_inputs,
            host_writes: Vec::new(),
            in_flight: std::sync::OnceLock::new(),
        }
    }

    /// Record the command buffer this forward was committed in.
    pub(super) fn committed(&self, in_flight: InFlight) {
        assert!(
            self.in_flight.set(in_flight).is_ok(),
            "a deferred forward is committed once"
        );
    }

    /// The command buffer this forward was committed in; `None` if it never was.
    pub fn take_in_flight(&mut self) -> Option<InFlight> {
        self.in_flight.take()
    }
}

/// One input token an earlier forward wrote on the device: the `u32` `offset` bytes into `src`,
/// for `input_ids[flat_index]`.
pub struct DeviceInput {
    pub src: super::__re::Buffer,
    pub offset: usize,
    pub flat_index: usize,
}

/// A committed command buffer the host has not waited for.
pub struct InFlight {
    pub(super) event: super::__re::SharedEvent,
    pub(super) value: u64,
    /// Its GPU execution error, recorded by its own commit feedback handler.
    pub(super) error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// The commit options carrying that handler, kept until the command buffer is done.
    pub(super) _feedback: objc2::rc::Retained<objc2_metal::MTL4CommitOptions>,
}

// SAFETY: like the pool's (`MetalWorkerPool`), the markers objc2's objects lack: a shared event is
// made to be waited on from any thread, the error slot is a mutex, and the commit options are only
// held, never used, once committed.
unsafe impl Send for InFlight {}
unsafe impl Sync for InFlight {}

impl InFlight {
    /// Block until the GPU is done with it. `Err`: the execution error its commit feedback reported
    /// — a failed command buffer otherwise yields all-zero outputs silently.
    pub fn wait(&self) -> Result<(), ForwardError> {
        crate::mtl4_dispatch::wait_drained(&self.event, self.value);
        if let Some(msg) = &*self.error.lock().expect("commit error mutex") {
            eprintln!("[scratchy-target-metal] GPU commit error surfaced: {msg}");
            return Err(ForwardError::ExecutionFailed(MTLCommandBufferStatus::Error));
        }
        Ok(())
    }
}

/// Build the per-token MRoPE cos/sin override table for the text decoder
/// (option (b): per-token table + identity positions).
///
/// Returns `[n_tokens, rot_dim]` rows in `dtype` (the rope kernel's
/// element type), laid out `[cos(0..half) | sin(half..rot_dim)]` per row —
/// byte-identical to a single row of the static
/// `scratchy_target_cuda::rotary::RotaryCache` cos/sin cache, so binding it
/// in place of that cache (with `positions[t] = t`) is a no-op for the
/// kernel math. The angle math mirrors
/// `RotaryCache::new_partial_from_gpuweights` exactly
/// (`inv_freq[i] = θ^(-2i/rot_dim)`, `angle = pos·inv_freq[i]`, cos/sin in
/// `f32`, then cast), so the 1D (text) case matches the legacy path to the
/// bit.
///
/// `positions` is either `[n_tokens]` (1D — every band uses the same
/// position, e.g. pure-text / decode) or `[3·n_tokens]` (MRoPE: rows
/// T,H,W). `mrope_section` is the `[T, H, W]` rotary-pair split (sums to
/// `rot_dim/2`); pair `i` rotates under T if `i < T`, H if `i < T+H`, else
/// W. No-scaling only (Qwen3.5-VL is partial-rotary with no rope_scaling).
pub fn build_mrope_cos_sin_override(
    positions: &[u32],
    n_tokens: usize,
    rot_dim: usize,
    rope_theta: f64,
    mrope_section: [u32; 3],
    dtype: super::lowered::MetalDtype,
) -> Vec<u8> {
    use super::lowered::MetalDtype;
    let half = rot_dim / 2;
    // `[3, n]` (band-split) vs `[n]` (broadcast all bands to one position).
    let is_3d = positions.len() == 3 * n_tokens;
    // inv_freq[i] = 1 / θ^(2i/rot_dim) — identical to the RotaryCache builder.
    let inv_freqs: Vec<f64> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / rot_dim as f64))
        .collect();
    let sec0 = mrope_section[0] as usize;
    let sec01 = (mrope_section[0] + mrope_section[1]) as usize;
    let elem_bytes: usize = match dtype {
        MetalDtype::F16 | MetalDtype::Bf16 => 2,
        MetalDtype::Int4 => panic!(
            "build_mrope_cos_sin_override: int4 cos/sin cache is not a thing — \
             the rope kernel reads f16/bf16"
        ),
    };
    let mut out = vec![0u8; n_tokens * rot_dim * elem_bytes];
    let write = |out: &mut [u8], idx: usize, v: f32| {
        let bytes: [u8; 2] = match dtype {
            MetalDtype::F16 => half::f16::from_f32(v).to_bits().to_ne_bytes(),
            MetalDtype::Bf16 => half::bf16::from_f32(v).to_bits().to_ne_bytes(),
            MetalDtype::Int4 => unreachable!(),
        };
        out[idx * 2..idx * 2 + 2].copy_from_slice(&bytes);
    };
    for t in 0..n_tokens {
        let (pos_t, pos_h, pos_w) = if is_3d {
            (
                positions[t],
                positions[n_tokens + t],
                positions[2 * n_tokens + t],
            )
        } else {
            let p = positions.get(t).copied().unwrap_or(0);
            (p, p, p)
        };
        let row = t * rot_dim;
        for (i, &inv_f) in inv_freqs.iter().enumerate() {
            let pos = if i < sec0 {
                pos_t
            } else if i < sec01 {
                pos_h
            } else {
                pos_w
            };
            let angle = pos as f64 * inv_f;
            write(&mut out, row + i, angle.cos() as f32);
            write(&mut out, row + half + i, angle.sin() as f32);
        }
    }
    out
}

/// Errors produced by [`super::pool::MetalWorkerPool::forward`] before
/// or during a forward step.
#[derive(Debug)]
pub enum ForwardError {
    /// `num_tokens == 0`. Selecting a bucket for a no-op step makes
    /// no sense — the engine must guard before calling forward.
    ZeroTokens,
    /// `num_tokens` exceeded every bucket's `bucket_m`. The engine
    /// either needs to chunk the step or the model loader needs to
    /// add a wider bucket.
    NoBucketFits { num_tokens: u32, max_bucket: u32 },
    /// One of the input slices was bigger than its runtime buffer.
    /// The runtime buffer was sized at `RuntimeFactory` invocation —
    /// either the factory under-sized it for this bucket or the
    /// caller is staging more elements than the bucket admits.
    BufferTooSmall {
        kind: &'static str,
        bytes_needed: usize,
        bytes_available: usize,
    },
    /// Worker encoding / lookup failed (pipeline lookup, GEMM encode,
    /// arena slot out of range, …).
    Worker(WorkerError),
    /// `commit` + `wait_until_completed` finished with a non-Completed
    /// status. This is the GPU-side failure mode — Metal exposes only
    /// the enum, not the underlying NSError.
    ExecutionFailed(MTLCommandBufferStatus),
    /// A caller-supplied followup hook (e.g. argmax encode + wait
    /// chained on the forward CB's shared event) failed.
    Followup(String),
    /// A step's block table of `len` entries is not `num_seqs` rows of the KV cap rung's width,
    /// the stride the pool's tapes read it at.
    BlockTableWidth {
        len: usize,
        num_seqs: usize,
        rung: MaxBlocksPerSeq,
    },
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTokens => write!(f, "MetalWorkerPool::forward: num_tokens == 0"),
            Self::NoBucketFits {
                num_tokens,
                max_bucket,
            } => write!(
                f,
                "MetalWorkerPool::forward: no bucket fits num_tokens={num_tokens} \
                 (max bucket_m = {max_bucket})"
            ),
            Self::BufferTooSmall {
                kind,
                bytes_needed,
                bytes_available,
            } => write!(
                f,
                "MetalWorkerPool::forward: runtime buffer `{kind}` too small \
                 (needs {bytes_needed} bytes, have {bytes_available})"
            ),
            Self::Worker(e) => write!(f, "MetalWorkerPool::forward: {e}"),
            Self::ExecutionFailed(status) => write!(
                f,
                "MetalWorkerPool::forward: command buffer status = {status:?} (expected Completed)"
            ),
            Self::Followup(msg) => {
                write!(f, "MetalWorkerPool::forward: followup hook failed: {msg}")
            }
            Self::BlockTableWidth {
                len,
                num_seqs,
                rung,
            } => write!(
                f,
                "MetalWorkerPool::forward: a block table of {len} entries is not {num_seqs} \
                 rows of the KV cap rung's {} blocks",
                rung.get()
            ),
        }
    }
}

impl std::error::Error for ForwardError {}

impl From<WorkerError> for ForwardError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}
