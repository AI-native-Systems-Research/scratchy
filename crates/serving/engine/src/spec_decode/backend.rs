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
    /// `true` when this is a spec-decode verify batch (at least one
    /// req carries `spec_token_ids`). Backend forwards this into the
    /// lm_head slice's runtime gate so the slice (which writes only
    /// the last logits row) is skipped and the full GEMM fires
    /// instead — rejection sampling needs every row populated.
    /// `false` for non-spec prefill/decode and for draft-model
    /// lockstep prefill / K-step chain forwards.
    pub has_spec_tokens: bool,
    /// `[num_sample_rows]` u32 — per-sample-row source index into the
    /// `[num_tokens, hidden]` activation produced by the forward.
    /// Mirrors Python vLLM's `logits_indices = query_start_loc[1:] - 1`
    /// and the CUDA path's `ForwardCtx.last_token_indices`. Backends
    /// thread it into their lm_head fast path so the GEMM runs at
    /// `M = num_sample_rows` instead of `M = num_tokens`. `None`
    /// signals "no sampled tokens this step" (chunked-prefill
    /// intermediate chunks) — backends fall back to the full GEMM.
    pub last_token_indices: Option<&'a [u32]>,
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
    /// `req`, plus a per-row argmax over the resulting `[req.num_tokens,
    /// vocab]` logits. Blocks the calling thread until completion.
    /// Returns per-row argmax IDs (length == `req.num_tokens`).
    fn forward_argmax_blocking(
        &mut self,
        model: ModelHandle,
        kv_pool: KvPoolHandle,
        req: &ForwardArgmaxRequest<'_>,
    ) -> Result<Vec<u32>, BackendError>;

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
