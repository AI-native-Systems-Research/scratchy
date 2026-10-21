// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Backend primitive for spec decode.
//!
//! Spec decode (ngram, draft-model, EAGLE) needs exactly one GPU primitive
//! the rest of the architecture shouldn't know about: **forward + per-row
//! argmax** against a chosen model + KV pool. Everything else — verify
//! batch layout, rejection sampling, K-step chain orchestration, draft
//! KV mirroring, sync force, lookahead reservation — is structural and
//! lives in `scratchy_serving_engine::spec_decode` regardless of backend.
//!
//! The trait splits forward submission from completion:
//!
//!   * [`SpecDecodeBackend::submit_forward_argmax`] enqueues a forward +
//!     per-row-argmax pass against the given model + KV pool and returns
//!     an opaque handle.
//!   * [`SpecDecodeBackend::await_forward_argmax`] blocks the calling
//!     thread until the handle completes and returns the argmax row IDs.
//!
//! The split exists for phase 8 of the spec-decode refactor (overlap target
//! verify with draft chain via parallel command buffers / streams).
//! Callers that don't need overlap call [`SpecDecodeBackend::forward_argmax_blocking`]
//! — the default impl is `submit → await` back-to-back.
//!
//! Lifecycle methods ([`SpecDecodeBackend::load_secondary_model`],
//! [`SpecDecodeBackend::allocate_kv_pool`]) let the draft-model proposer
//! manage its second model + KV pool through the backend without leaking
//! `cudaMalloc` / `MTLBuffer` details.

use std::error::Error;
use std::fmt;
use std::path::Path;

/// Opaque token identifying a model registered with the backend.
///
/// `0` is reserved for the primary (target) model loaded via the executor's
/// existing `load_model` lifecycle. Secondary models registered via
/// [`SpecDecodeBackend::load_secondary_model`] get successive handles
/// starting at `1`. Backends are free to assign arbitrary values past `0`;
/// callers must treat the value as opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelHandle(pub u32);

impl ModelHandle {
    /// Handle of the target model (always registered after `load_model`).
    pub const TARGET: Self = Self(0);
}

/// Opaque token identifying a KV cache pool registered with the backend.
///
/// `0` is the target pool created by `initialize_cache`. Secondary pools
/// (draft) registered via [`SpecDecodeBackend::allocate_kv_pool`] get
/// successive handles starting at `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvPoolHandle(pub u32);

impl KvPoolHandle {
    /// Handle of the target KV pool (always allocated by `initialize_cache`).
    pub const TARGET: Self = Self(0);
}

/// Opaque token returned by [`SpecDecodeBackend::submit_forward_argmax`]
/// and consumed by [`SpecDecodeBackend::await_forward_argmax`].
///
/// Backends encode whatever they need (CB index, stream id, slot in an
/// internal vec, etc.) into the `u64` payload. Callers must round-trip the
/// handle from submit to await without inspecting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardHandle(pub u64);

/// Errors a [`SpecDecodeBackend`] method can return.
#[derive(Debug)]
pub enum BackendError {
    /// The method isn't yet implemented for this backend. Used by phase
    /// 5.1's thin wrappers where a backend hasn't built out a method
    /// (e.g. CUDA `load_secondary_model` before the cross-backend port).
    NotImplemented(&'static str),
    /// A handle (`ModelHandle`, `KvPoolHandle`, `ForwardHandle`) doesn't
    /// match anything the backend registered.
    UnknownHandle(&'static str),
    /// Pass-through for backend-specific failures (kernel dispatch error,
    /// allocation failure, file IO, etc.). Stringly-typed so the trait
    /// stays object-safe and dep-free.
    Backend(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::NotImplemented(what) => write!(f, "not implemented: {what}"),
            BackendError::UnknownHandle(kind) => write!(f, "unknown {kind}"),
            BackendError::Backend(msg) => write!(f, "backend: {msg}"),
        }
    }
}

impl Error for BackendError {}

/// Backend-agnostic forward-pass inputs for a spec-decode forward+argmax.
///
/// All fields are host-owned slices. The backend is responsible for
/// uploading them to device buffers in whatever format its kernels expect
/// (Metal `MTLBuffer` shared-storage, CUDA pinned-staged H2D, etc.). The
/// engine never sees `ForwardCtx` or any other backend-specific input
/// bundle.
///
/// Field semantics mirror Python vLLM's flash-attention varlen inputs:
///
///   * `input_ids[i]`  — token at the i-th position in the flat batch.
///   * `positions[i]`  — absolute position of `input_ids[i]` in its sequence.
///   * `slot_mapping[i]` — `block_id * block_size + offset_within_block`
///     where the K/V for `input_ids[i]` is written into the KV pool.
///   * `cu_seqlens_q`  — exclusive prefix sum over per-seq `q_len`s,
///     length `num_reqs + 1`.
///   * `seqused_k[i]`  — total KV length the attention kernel reads for
///     seq `i` (= existing KV + new tokens this step).
///   * `block_table`   — flat `[num_reqs, block_table_stride]` u32 of block
///     IDs in the chosen KV pool.
pub struct ForwardArgmaxRequest<'a> {
    pub input_ids: &'a [u32],
    pub positions: &'a [u32],
    pub slot_mapping: &'a [u32],
    pub cu_seqlens_q: &'a [u32],
    pub seqused_k: &'a [u32],
    pub block_table: &'a [u32],
    /// Per-block SPAN LABEL for block-diagonal span attention (`0` =
    /// shared/query, `k+1` = k-th Relocatable span). `None` ⇒ no spans.
    pub span_ids: Option<&'a [u32]>,
    /// Sliding KV-cache GROUPS (gemma4 SWA): per sliding group, the
    /// `slot_mapping` + `block_table` for that group's local layers (vLLM
    /// group-shared layout has N sliding groups; group 0 = full uses
    /// [`Self::slot_mapping`] / [`Self::block_table`]). Same
    /// `block_table_stride`. Empty on non-SWA models. Index `s` = KV-cache
    /// group `s + 1`.
    pub sliding_slot_mappings: &'a [&'a [u32]],
    pub sliding_block_tables: &'a [&'a [u32]],
    pub block_table_stride: usize,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub num_tokens: usize,
    /// The rows the forward samples ([`SampledRows::rows`]): Python vLLM's `logits_indices`. The
    /// forward's logits, and the argmax the backend returns, are theirs, in order. `None`: each
    /// sequence's last row.
    pub sampled_rows: Option<&'a [u32]>,
    /// `[num_tokens, hidden]` row-major, model dtype: the final (post-norm) hidden state each row
    /// of a multi-token-prediction head reads — its target's, or its own from the pass before.
    /// `None` for every other model.
    pub target_hidden: Option<&'a [u8]>,
}

/// The rows a step samples: each sequence's last, or — a sequence verifying drafts — every one of
/// its rows, each of which can be its newest. A sequence's sampled rows are its last rows, so
/// sampled row `j` of sequence `i` is step row `cu_seqlens_q[i + 1] - (count_i - j)`. The logits
/// the forward leaves, and every row read off them (argmax, sampler, grammar mask), are these
/// rows', in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledRows {
    /// Each sampled row's step row, ascending (Python vLLM's `logits_indices`).
    rows: Vec<u32>,
    /// Cumulative sampled rows per sequence, `num_seqs + 1` long, strictly ascending: sequence
    /// `i`'s are `cu[i]..cu[i + 1]`, at least one.
    cu: Vec<u32>,
}

/// A step's sequence of no rows: it has no last row to sample, so no step holds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptySequence {
    pub seq: usize,
}

impl std::fmt::Display for EmptySequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sequence {} of the step holds no rows to sample",
            self.seq
        )
    }
}

impl std::error::Error for EmptySequence {}

impl SampledRows {
    /// The sampled rows of the step whose sequences' rows `cu_seqlens_q` bounds: every row of a
    /// sequence `every_row(i)` holds, the last of any other. Refused for a sequence of no rows.
    pub fn of(
        cu_seqlens_q: &[u32],
        every_row: impl Fn(usize) -> bool,
    ) -> Result<Self, EmptySequence> {
        let mut rows = Vec::new();
        let mut cu = vec![0];
        for (i, seq) in cu_seqlens_q.windows(2).enumerate() {
            let (start, end) = (seq[0], seq[1]);
            if end <= start {
                return Err(EmptySequence { seq: i });
            }
            let first = match every_row(i) {
                true => start,
                false => end - 1,
            };
            rows.extend(first..end);
            cu.push(rows.len() as u32);
        }
        Ok(Self { rows, cu })
    }

    /// Each sampled row's step row, ascending.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// How many rows the step samples.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Sequence `i`'s sampled rows, as indices into [`Self::rows`] — rows of the logits.
    pub fn of_seq(&self, i: usize) -> std::ops::Range<usize> {
        self.cu[i] as usize..self.cu[i + 1] as usize
    }

    /// Sequence `i`'s last sampled row, as an index into [`Self::rows`]: its last row's logits.
    pub fn last_of(&self, i: usize) -> usize {
        self.cu[i + 1] as usize - 1
    }

    /// The index into [`Self::rows`] of step row `row` of sequence `i`, one of its sampled rows
    /// (`None` past them).
    pub fn index(&self, i: usize, row: u32) -> Option<usize> {
        self.of_seq(i).find(|&j| self.rows[j] == row)
    }
}

/// Primitive the backend exposes for spec decode.
///
/// Backends MUST implement [`forward_argmax_blocking`]. The async pair
/// ([`submit_forward_argmax`] + [`await_forward_argmax`]) defaults to
/// `NotImplemented` and is overridden in phase 8 of the refactor where
/// target verify overlaps with the draft chain on parallel command
/// buffers / streams.
///
/// See module docs for the design rationale.
pub trait SpecDecodeBackend {
    /// Run a forward pass against `model` + `kv_pool` with the inputs in
    /// `req`, plus a per-row argmax over its logits — its sampled rows'
    /// (`req.sampled_rows`). Blocks the calling thread until completion.
    /// Returns an argmax ID per sampled row, in order.
    fn forward_argmax_blocking(
        &mut self,
        model: ModelHandle,
        kv_pool: KvPoolHandle,
        req: &ForwardArgmaxRequest<'_>,
    ) -> Result<Vec<u32>, BackendError>;

    /// [`Self::forward_argmax_blocking`], also returning the forward's final (post-norm) hidden
    /// states at `hidden_rows`, row-major in the model dtype: what a multi-token-prediction head
    /// reads at its next depth. Default impl returns `NotImplemented`.
    fn forward_argmax_hidden_blocking(
        &mut self,
        _model: ModelHandle,
        _kv_pool: KvPoolHandle,
        _req: &ForwardArgmaxRequest<'_>,
        _hidden_rows: &[u32],
    ) -> Result<(Vec<u32>, Vec<u8>), BackendError> {
        Err(BackendError::NotImplemented(
            "forward_argmax_hidden_blocking",
        ))
    }

    /// Submit the same forward + argmax non-blocking. Default impl returns
    /// `NotImplemented`. Phase 8 overrides this on each backend for
    /// target||draft overlap.
    fn submit_forward_argmax(
        &mut self,
        _model: ModelHandle,
        _kv_pool: KvPoolHandle,
        _req: &ForwardArgmaxRequest<'_>,
    ) -> Result<ForwardHandle, BackendError> {
        Err(BackendError::NotImplemented("submit_forward_argmax"))
    }

    /// Block on a previously-submitted forward+argmax. Default impl
    /// returns `NotImplemented` — only meaningful if the backend
    /// overrides [`submit_forward_argmax`].
    fn await_forward_argmax(&mut self, _handle: ForwardHandle) -> Result<Vec<u32>, BackendError> {
        Err(BackendError::NotImplemented("await_forward_argmax"))
    }

    /// Load a second model on the same device. Used by `DraftModelProposer`
    /// at engine init. Returns the handle the proposer will pass to every
    /// later `forward_argmax` call.
    fn load_secondary_model(
        &mut self,
        path: &Path,
        dtype: Option<&str>,
    ) -> Result<ModelHandle, BackendError>;

    /// Allocate a KV pool sized for `num_blocks` blocks against the given
    /// model's per-layer / per-head / head-dim layout. Block size matches
    /// the target pool's. Used by `DraftModelProposer` after
    /// `load_secondary_model`.
    fn allocate_kv_pool(
        &mut self,
        model: ModelHandle,
        num_blocks: usize,
    ) -> Result<KvPoolHandle, BackendError>;

    /// Per-block KV bytes for the given model handle. Used by the engine's
    /// per-pair memory budget split so it scales without knowing the
    /// backend's KV element width.
    fn kv_per_block_bytes(&self, model: ModelHandle) -> Result<usize, BackendError>;

    /// Phase 6: K-step draft chain in ONE GPU submission. Replaces the
    /// proposer's K-iter `forward_argmax_blocking` loop with a single
    /// call. Backends that override this fuse forward, per-row argmax,
    /// and per-req position/slot/seqused_k advance into one command
    /// buffer with a single host wait. Returns `[k][num_reqs]` argmax
    /// IDs (iter-major).
    ///
    /// Default impl loops `forward_argmax_blocking` so CUDA / future
    /// backends remain correct without per-backend chain wiring — but
    /// each iter still pays one commit + one host wait. The Metal
    /// override (Phase 6) collapses both costs to one.
    ///
    /// `req_iter0` contains the iter-0 inputs (input_ids = seed bonus
    /// tokens, positions/slot_mapping/seqused_k computed for iter 0).
    /// For chain-aware backends, iter-1..k-1 inputs are advanced
    /// on-device from the iter-0 state via the backend's chain kernel;
    /// the trait-default fallback advances them on the host and
    /// re-submits each iter (see the existing loop in
    /// `propose_lockstep_k`).
    fn forward_chain_k(
        &mut self,
        _model: ModelHandle,
        _kv_pool: KvPoolHandle,
        _req_iter0: &ForwardArgmaxRequest<'_>,
        _block_size: usize,
        _k: usize,
    ) -> Result<Vec<Vec<u32>>, BackendError> {
        Err(BackendError::NotImplemented("forward_chain_k"))
    }
}

#[cfg(test)]
mod tests {
    use super::{EmptySequence, SampledRows};

    /// A step of a decode (row 0), a verify of three rows (1..=3) and a prompt chunk (4..=9): the
    /// decode and the chunk sample their last rows, the verify its every one — ascending, each
    /// sequence's at its own offsets.
    #[test]
    fn a_step_samples_each_sequence_s_last_rows() {
        let sampled = SampledRows::of(&[0, 1, 4, 10], |i| i == 1).expect("no empty sequence");
        assert_eq!(sampled.rows(), [0, 1, 2, 3, 9]);
        assert_eq!(sampled.len(), 5);
        let seqs: Vec<_> = (0..3).map(|i| sampled.of_seq(i)).collect();
        assert_eq!(seqs, [0..1, 1..4, 4..5]);
        let last: Vec<usize> = (0..3).map(|i| sampled.last_of(i)).collect();
        assert_eq!(last, [0, 3, 4]);
        // A row's index among the sampled rows; a row its sequence does not sample has none.
        assert_eq!(sampled.index(1, 2), Some(2));
        assert_eq!(sampled.index(2, 9), Some(4));
        assert_eq!(sampled.index(2, 8), None);
        assert_eq!(sampled.index(0, 1), None);
    }

    /// A sequence of no rows has no last row to sample, whether or not it verifies drafts: the
    /// step is refused, naming it.
    #[test]
    fn a_sequence_of_no_rows_is_refused() {
        for every_row in [false, true] {
            let refused = SampledRows::of(&[0, 1, 1, 4], |_| every_row);
            assert_eq!(refused, Err(EmptySequence { seq: 1 }));
        }
    }
}
