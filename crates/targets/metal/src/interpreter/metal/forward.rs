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
use scratchy_core_common::sampled_rows::{EmptySequence, SampledRows};

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
    /// `[num_sample_rows]` u32 — the rows the forward samples, ascending: each sequence's last,
    /// every row of a sequence verifying drafts. Python vLLM's `logits_indices`. The forward
    /// leaves their logits at the front of its logits buffer, row `i` sampled row `i`'s. A caller
    /// that names none samples each sequence's last ([`Self::last_rows`]).
    pub sampled_rows: &'a [u32],
    /// `[num_seqs * 6]` i32 — a [`GdnSeqSlots`](crate::gdn_state::GdnSeqSlots) record per
    /// batched sequence (cu_seqlens order), flattened as `GdnSeqSlots::words`. `None` for
    /// non-hybrid arches; required for any bucket that runs `Instruction::GatedDeltaNet`.
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
    /// A DFlash2 drafter's per-forward inputs; `None` for every other model's forward.
    pub draft: Option<DraftInputs<'a>>,
    /// Bytes earlier forwards wrote on the device, for this one's runtime inputs.
    pub device_inputs: &'a [DeviceInput],
    /// `Some`: the host does not wait for this forward ([`Deferral`]).
    pub deferred: Option<&'a Deferral>,
    /// What the caller reads of this forward.
    pub reads: ForwardReads,
}

impl ForwardInputs<'_> {
    /// How many rows the forward samples ([`Self::sampled_rows`]).
    pub fn sampled_row_count(&self) -> crate::tape::ids::SampledRowCount {
        crate::tape::ids::SampledRowCount(self.sampled_rows.len() as u32)
    }

    /// The rows a forward samples when its caller names none: each sequence `cu_seqlens_q` bounds'
    /// last row, as the serving engine's steps sample them ([`SampledRows::of`]) — every row of
    /// `num_tokens`, a step that names no sequences being one token each. Refused for a sequence
    /// of no rows, which has none to sample.
    pub fn last_rows(
        cu_seqlens_q: Option<&[u32]>,
        num_tokens: u32,
    ) -> Result<SampledRows, EmptySequence> {
        match cu_seqlens_q {
            Some(cu) => SampledRows::of(cu, |_| false),
            None => SampledRows::of(&(0..=num_tokens).collect::<Vec<u32>>(), |_| false),
        }
    }
}

/// A DFlash2 drafter's per-forward inputs (`crate::dflash2`): its lanes are the forward's
/// sequences, in order.
#[derive(Clone, Copy, Debug)]
pub struct DraftInputs<'a> {
    /// `[num_seqs]` u32 — each lane's ring slot.
    pub ring_slots: &'a [u32],
    /// `[num_seqs]` — each lane's policy for the cycle (a block draft's selection reads it).
    pub lane_params: &'a [crate::dflash2::DraftLaneParams],
    /// The forward is a cycle's context commit over its lanes' verify blocks: the rows each
    /// retained are a device input (`DeviceInputInto::DraftRetained`), its target's acceptance's.
    pub cycle: bool,
}

/// What a forward's caller reads of it: which of its tape's commands it plays
/// ([`crate::tape::lowered::PlayLive`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForwardReads {
    /// Its outputs — logits, final hidden states — and its cache writes.
    #[default]
    Outputs,
    /// Its cache writes alone — KV, recurrent state: its tape plays only the commands they need.
    CacheWrites,
    /// Its result alone: its tape plays only the commands the result needs. A cache it writes
    /// besides is written only where those commands write it; one only other commands write is
    /// left as the forward found it (a DFlash2 drafter's block draft reads its rings and leaves
    /// their commit to the play of its cache writes after its target's verify).
    Result,
}

/// A forward the host commits without waiting for it, so the next one is queued while it runs. Its
/// runtime-input writes are made on the device at the head of its command buffer — the forward
/// before it may still be reading those buffers — and it starts once that forward is done.
#[derive(Default)]
pub struct Deferral {
    /// Bytes for buffers outside the runtime inputs that an earlier command buffer may still be
    /// reading (the sampler's per-step parameters), written on the device like the runtime inputs.
    pub host_writes: Vec<(super::__re::Buffer, Vec<u8>)>,
    /// Inputs that depend on how many drafts the step before it kept, picked on the device after
    /// the writes above ([`crate::select_rows`]), by `select`, the model's kernel for it.
    pub selections: Vec<crate::select_rows::Selection>,
    pub select: Option<super::__re::ComputePipelineState>,
    in_flight: std::sync::OnceLock<InFlight>,
}

impl Deferral {
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

/// Bytes an earlier forward wrote on the device — in an earlier command buffer, or earlier in this
/// one ([`super::pool::MetalWorkerPool::forward_onto`]) — for one of this forward's runtime inputs:
/// `len` bytes `offset` into `src`, to `at` bytes into `into`, over what the host wrote there.
pub struct DeviceInput {
    pub src: super::__re::Buffer,
    pub offset: usize,
    pub len: usize,
    pub into: DeviceInputInto,
    pub at: usize,
}

impl DeviceInput {
    /// `input_ids[flat_index]`: the `u32` token `offset` bytes into `src`.
    pub fn token(src: super::__re::Buffer, offset: usize, flat_index: usize) -> Self {
        let len = size_of::<u32>();
        let (into, at) = (DeviceInputInto::InputIds, flat_index * len);
        Self {
            src,
            offset,
            len,
            into,
            at,
        }
    }

    /// `len` bytes `offset` into `src`, to the head of `into`.
    pub fn head(
        src: super::__re::Buffer,
        offset: usize,
        len: usize,
        into: DeviceInputInto,
    ) -> Self {
        let at = 0;
        Self {
            src,
            offset,
            len,
            into,
            at,
        }
    }

    /// `rows` rows of `target_hidden` from row `row`: `src`'s from row `src_row`, `row_bytes` each.
    pub fn hidden_rows(
        src: super::__re::Buffer,
        src_row: usize,
        row: usize,
        rows: usize,
        row_bytes: usize,
    ) -> Self {
        Self {
            src,
            offset: src_row * row_bytes,
            len: rows * row_bytes,
            into: DeviceInputInto::TargetHidden,
            at: row * row_bytes,
        }
    }
}

/// The runtime input a [`DeviceInput`] lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceInputInto {
    /// `input_ids`: tokens an earlier forward picked, over the host's placeholders.
    InputIds,
    /// `target_hidden`: a target's final hidden states, for its MTP head. Given no host bytes, the
    /// forward's target hidden rows are these.
    TargetHidden,
    /// `positions`, over the host's placeholders.
    Positions,
    /// The full KV-cache group's `slot_mapping`, over the host's placeholders.
    SlotMapping,
    /// Sliding KV-cache group `n`'s `slot_mapping` (gemma4 SWA: the runtime's group `1 + n`), over
    /// the host's placeholders.
    SlidingSlotMapping(u8),
    /// `seq_used_k`, over the host's placeholders.
    SeqUsedK,
    /// An MRoPE model's rotary cos/sin rows (`mrope_cos_sin`), over the host's placeholders.
    RopeRows,
    /// Each sequence's Gated-DeltaNet step code (`gdn_is_fresh`), over the host's.
    GdnSteps,
    /// A drafter cycle commit's retained counts (`DraftRetained`): its target's acceptance's.
    DraftRetained,
}

/// What one command buffer's commit feedback said, once it has said it. The handler sets it once
/// and no wait consumes it, so every wait on the command buffer, and only on it, sees the same.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delivered {
    Ok,
    Err(String),
}

/// One command buffer's commit feedback, which its handler delivers on a thread of Metal's own,
/// not ordered against the shared event the queue signals after the command buffer. A wait that
/// read it the moment that event fired could find it not yet delivered, and take a failed command
/// buffer for a finished one.
#[derive(Default)]
pub struct CommitFeedback {
    delivered: std::sync::Mutex<Option<Delivered>>,
    arrived: std::sync::Condvar,
}

impl CommitFeedback {
    /// Delivered: the handler's word, once.
    pub fn deliver(&self, d: Delivered) {
        *self.delivered.lock().expect("commit feedback mutex") = Some(d);
        self.arrived.notify_all();
    }

    /// What it said, waiting for the handler to say it.
    fn wait(&self) -> Delivered {
        let started = std::time::Instant::now();
        let mut d = self.delivered.lock().expect("commit feedback mutex");
        loop {
            if let Some(d) = d.as_ref() {
                return d.clone();
            }
            let (next, timeout) = (self.arrived)
                .wait_timeout(d, std::time::Duration::from_secs(60))
                .expect("commit feedback mutex");
            d = next;
            if timeout.timed_out() && d.is_none() {
                eprintln!(
                    "[scratchy-target-metal] a command buffer's commit feedback is {:?} late; \
                     waiting for it",
                    started.elapsed(),
                );
            }
        }
    }
}

/// A committed command buffer the host has not waited for.
pub struct InFlight {
    pub(super) event: super::__re::SharedEvent,
    pub(super) value: u64,
    /// Its GPU execution error, or none, delivered by its own commit feedback handler.
    pub(super) error: std::sync::Arc<CommitFeedback>,
    /// The commit options carrying that handler, kept until the command buffer is done.
    pub(super) _feedback: objc2::rc::Retained<objc2_metal::MTL4CommitOptions>,
    /// Every registered kernel-time heap's cursor at this command buffer's
    /// commit (`SCRATCHY_METAL_KERNEL_TIME`): its stamps up to here belong to
    /// command buffers this one's wait drains, so the wait prints them. Empty
    /// without the env var.
    pub(super) kernel_time: Vec<(std::sync::Arc<super::mtl4::KernelTimeHeap>, u64)>,
}

// SAFETY: like the pool's (`MetalWorkerPool`), the markers objc2's objects lack: a shared event is
// made to be waited on from any thread, the error slot is a mutex, and the commit options are only
// held, never used, once committed.
unsafe impl Send for InFlight {}
unsafe impl Sync for InFlight {}

/// Command buffers waited so far, to label the kernel-time tape of each
/// (`SCRATCHY_METAL_KERNEL_TIME`): the trace's per-forward lines interleave
/// with these by order.
static KERNEL_TIME_CB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl InFlight {
    /// Block until the GPU is done with it and its commit feedback has said how it went. `Err`: the
    /// execution error that feedback reported — a failed command buffer otherwise yields all-zero
    /// outputs silently.
    pub fn wait(&self) -> Result<(), ForwardError> {
        crate::mtl4_dispatch::wait_drained(&self.event, self.value);
        if let Delivered::Err(msg) = self.error.wait() {
            eprintln!("[scratchy-target-metal] GPU commit error surfaced: {msg}");
            return Err(ForwardError::ExecutionFailed(MTLCommandBufferStatus::Error));
        }
        if !self.kernel_time.is_empty() {
            let cb = KERNEL_TIME_CB.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for (heap, end) in &self.kernel_time {
                heap.print_new_since(*end, &format!("cb {cb}"));
            }
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
/// kernel math. The angle math is
/// `RotaryCache::new_partial_from_gpuweights`'s
/// (`inv_freq[i] = θ^(-2i/rot_dim)`, `angle = pos·inv_freq[i]`, cos/sin in
/// `f32`, then cast; that builder's angle addition lands on the same bf16),
/// so the 1D (text) case matches the static table to the bit.
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
    /// A step names group-0 block `block`, past the `mapped` its TurboQuant stores are mapped
    /// through: its caller grows them first (`SparseRegistry::reach`), or the forward would read
    /// un-wired memory.
    KvNotMapped { block: u32, mapped: usize },
    /// A step samples `sampled` rows, whose logits its tape writes past its bucket's logits buffer
    /// (sized for the rows a step samples at most, `GpuDevice::max_decode_rows`).
    TooManySampledRows {
        sampled: crate::tape::ids::SampledRowCount,
        overflow: crate::tape::lowered::LogitsOverflow,
    },
    /// A forward that reads more than its cache writes, of `num_tokens` rows, whose bucket
    /// (`bucket_m`) plays its cache writes alone ([`super::MetalBucketSpec::plays_result`]): its
    /// arena holds none of what the rest of its tape writes.
    CacheWritesOnly {
        num_tokens: u32,
        bucket_m: u32,
        reads: ForwardReads,
    },
    /// A step's sequence of no rows, which has no last row to sample.
    EmptySequence(EmptySequence),
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
            Self::KvNotMapped { block, mapped } => write!(
                f,
                "MetalWorkerPool::forward: block {block} is past the {mapped} blocks the \
                 TurboQuant KV stores are mapped through (`SparseRegistry::reach` them first)"
            ),
            Self::TooManySampledRows { sampled, overflow } => write!(
                f,
                "MetalWorkerPool::forward: the step samples {} rows, whose logits fill {} rows, \
                 past the {} its bucket's logits buffer holds",
                sampled.get(),
                overflow.writes,
                overflow.holds
            ),
            Self::CacheWritesOnly {
                num_tokens,
                bucket_m,
                reads,
            } => write!(
                f,
                "MetalWorkerPool::forward: a forward of {num_tokens} rows reads {reads:?}, but \
                 its bucket of {bucket_m} plays its cache writes alone"
            ),
            Self::EmptySequence(e) => write!(f, "MetalWorkerPool::forward: {e}"),
        }
    }
}

impl std::error::Error for ForwardError {}

impl From<WorkerError> for ForwardError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}

#[cfg(test)]
mod tests {
    use super::{CommitFeedback, Delivered};

    /// A wait that starts before the commit feedback arrives, as it can once the shared event has
    /// fired, waits for it, and every wait after sees the same: a failed command buffer is never
    /// taken for a finished one.
    #[test]
    fn a_wait_before_the_feedback_arrives_gets_its_answer() {
        let feedback = std::sync::Arc::new(CommitFeedback::default());
        let handler = std::thread::spawn({
            let feedback = std::sync::Arc::clone(&feedback);
            move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                feedback.deliver(Delivered::Err("out of memory".into()));
            }
        });
        let failed = Delivered::Err("out of memory".into());
        assert_eq!(feedback.wait(), failed, "the first wait");
        assert_eq!(feedback.wait(), failed, "a later wait");
        handler.join().expect("the handler thread");
    }
}
