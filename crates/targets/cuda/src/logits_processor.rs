// SPDX-License-Identifier: Apache-2.0
//! LogitsProcessor framework for CudaWorker — mirrors Python vLLM's
//! `vllm/v1/sample/logits_processor/` architecture.
//!
//! Each processor maintains persistent GPU state that is rebuilt only when
//! the batch composition changes (via `update_state`), not every step.
//! This matches Python's design and avoids per-step H2D repacking overhead.

use std::collections::HashMap;

use crate::DType as GpuDType;
use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::driver;
use crate::tensor::GpuTensor;

use scratchy_core_common::sampling::SamplingParams;

// ---------------------------------------------------------------------------
// BatchUpdate — notification of batch composition changes
// ---------------------------------------------------------------------------

/// Describes changes to the persistent batch between steps.
/// Mirrors Python's `BatchUpdate` from `gpu_input_batch.py`.
pub struct BatchUpdate {
    /// Current batch size after applying changes.
    pub batch_size: usize,
    /// (batch_index, req_id) pairs for newly added requests.
    pub added: Vec<(usize, String)>,
    /// Batch indices of removed requests.
    pub removed: Vec<usize>,
}

// ---------------------------------------------------------------------------
// LogitsProcessor trait
// ---------------------------------------------------------------------------

/// GPU logit processor — mirrors Python's `LogitsProcessor` ABC.
///
/// Lifecycle:
/// 1. `update_state()` called once per step with batch changes (CPU-side bookkeeping + async H2D).
/// 2. `apply()` called to modify logits on GPU.
/// 3. `is_active()` gates whether `apply()` is called at all.
pub trait LogitsProcessor: Send {
    /// Update internal state when the batch changes or tokens grow.
    /// Called once per step. `batch_update` is `Some` if batch composition changed.
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    );

    /// Modify logits `[batch_size, vocab_size]` f32 in-place on GPU.
    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice);

    /// True if this processor doesn't affect argmax (applied after temperature).
    fn is_argmax_invariant(&self) -> bool;

    /// True if this processor has work to do this step. Skip `apply` if false.
    fn is_active(&self) -> bool;
}

// ---------------------------------------------------------------------------
// LogitsProcessorPipeline
// ---------------------------------------------------------------------------

/// Container for all logits processors, split by argmax invariance.
/// Mirrors Python's `LogitsProcessors` container.
pub struct LogitsProcessorPipeline {
    /// Processors that can change argmax (grammar, min_tokens, logit_bias, penalties).
    non_argmax_invariant: Vec<Box<dyn LogitsProcessor>>,
    /// Processors that don't change argmax (none currently — min_p is in sampling kernel).
    argmax_invariant: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorPipeline {
    /// Create a new pipeline with the given processors.
    /// Processors are automatically sorted into argmax-invariant vs non-argmax-invariant.
    pub fn new(processors: Vec<Box<dyn LogitsProcessor>>) -> Self {
        let mut non_argmax = Vec::new();
        let mut argmax = Vec::new();
        for p in processors {
            if p.is_argmax_invariant() {
                argmax.push(p);
            } else {
                non_argmax.push(p);
            }
        }
        Self {
            non_argmax_invariant: non_argmax,
            argmax_invariant: argmax,
        }
    }

    /// Update all processors with batch changes.
    pub fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        for p in &mut self.non_argmax_invariant {
            p.update_state(
                batch_update,
                sampling_params_map,
                token_buffers,
                batch_req_ids,
                device,
            );
        }
        for p in &mut self.argmax_invariant {
            p.update_state(
                batch_update,
                sampling_params_map,
                token_buffers,
                batch_req_ids,
                device,
            );
        }
    }

    /// Apply non-argmax-invariant processors (before sampling).
    pub fn apply_pre_sampling(&self, logits: GpuTensor, device: &mut GpuDevice) {
        for p in &self.non_argmax_invariant {
            if p.is_active() {
                p.apply(logits, device);
            }
        }
    }

    /// Whether any active processor needs work (gates fast-path skip).
    pub fn any_active(&self) -> bool {
        self.non_argmax_invariant.iter().any(|p| p.is_active())
            || self.argmax_invariant.iter().any(|p| p.is_active())
    }
}

// ---------------------------------------------------------------------------
// LogitBiasProcessor
// ---------------------------------------------------------------------------

/// Persistent logit bias processor. Rebuilds GPU tensors only when batch changes.
/// Mirrors Python's `LogitBiasLogitsProcessor`.
#[derive(Default)]
pub struct LogitBiasProcessor {
    active: bool,
    /// CSR-packed bias data, persistent across steps when batch is stable.
    gpu_bias_ids: Option<OwnedTensor>,
    gpu_bias_vals: Option<OwnedTensor>,
    gpu_bias_offsets: Option<OwnedTensor>,
}

impl LogitBiasProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LogitsProcessor for LogitBiasProcessor {
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        _token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        // Only rebuild when batch changes.
        if batch_update.is_none() && self.gpu_bias_ids.is_some() {
            return;
        }

        let mut bias_ids_flat: Vec<i32> = Vec::new();
        let mut bias_vals_flat: Vec<f32> = Vec::new();
        let mut bias_offsets: Vec<i32> = Vec::new();

        for req_id in batch_req_ids {
            bias_offsets.push(bias_ids_flat.len() as i32);
            if let Some(params) = sampling_params_map.get(req_id)
                && let Some(ref bias_map) = params.logit_bias
            {
                for (&tok_id, &val) in bias_map {
                    bias_ids_flat.push(tok_id as i32);
                    bias_vals_flat.push(val);
                }
            }
        }
        bias_offsets.push(bias_ids_flat.len() as i32);

        if bias_ids_flat.is_empty() {
            self.active = false;
            self.gpu_bias_ids = None;
            self.gpu_bias_vals = None;
            self.gpu_bias_offsets = None;
            return;
        }

        self.active = true;
        self.gpu_bias_ids = Some(h2d_i32_owned(&bias_ids_flat, device));
        self.gpu_bias_vals = Some(h2d_f32_owned(&bias_vals_flat, device));
        self.gpu_bias_offsets = Some(h2d_i32_owned(&bias_offsets, device));
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(ids), Some(vals), Some(offsets)) = (
            &self.gpu_bias_ids,
            &self.gpu_bias_vals,
            &self.gpu_bias_offsets,
        ) {
            unsafe {
                crate::kernels::apply_logit_bias(
                    logits,
                    ids.as_gpu_tensor(),
                    vals.as_gpu_tensor(),
                    offsets.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// PenaltiesProcessor
// ---------------------------------------------------------------------------

/// Persistent penalties processor. Penalty value tensors rebuilt on batch change;
/// output_token_ids rebuilt every step (tokens grow each step).
pub struct PenaltiesProcessor {
    active: bool,
    /// Penalty value tensors — persistent when batch is stable.
    gpu_rep_pens: Option<OwnedTensor>,
    gpu_freq_pens: Option<OwnedTensor>,
    gpu_pres_pens: Option<OwnedTensor>,
    /// Output token IDs — rebuilt every step since tokens grow.
    gpu_out_ids: Option<OwnedTensor>,
    /// Dummy prompt IDs (we don't penalize prompt tokens separately).
    gpu_prompt_ids: Option<OwnedTensor>,
    max_output_len: usize,
    num_reqs: usize,
    vocab_size: usize,
}

impl PenaltiesProcessor {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            active: false,
            gpu_rep_pens: None,
            gpu_freq_pens: None,
            gpu_pres_pens: None,
            gpu_out_ids: None,
            gpu_prompt_ids: None,
            max_output_len: 0,
            num_reqs: 0,
            vocab_size,
        }
    }
}

impl LogitsProcessor for PenaltiesProcessor {
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        let num_reqs = batch_req_ids.len();
        self.num_reqs = num_reqs;

        // Check if any request has penalties.
        let any = batch_req_ids.iter().any(|rid| {
            sampling_params_map.get(rid).is_some_and(|p| {
                p.frequency_penalty != 0.0
                    || p.presence_penalty != 0.0
                    || p.repetition_penalty != 1.0
            })
        });

        if !any {
            self.active = false;
            return;
        }
        self.active = true;

        // Rebuild penalty value tensors only on batch change.
        if batch_update.is_some() || self.gpu_rep_pens.is_none() {
            let mut rep_pens = vec![1.0f32; num_reqs];
            let mut freq_pens = vec![0.0f32; num_reqs];
            let mut pres_pens = vec![0.0f32; num_reqs];

            for (i, req_id) in batch_req_ids.iter().enumerate() {
                if let Some(params) = sampling_params_map.get(req_id) {
                    rep_pens[i] = params.repetition_penalty as f32;
                    freq_pens[i] = params.frequency_penalty as f32;
                    pres_pens[i] = params.presence_penalty as f32;
                }
            }

            self.gpu_rep_pens = Some(h2d_f32_owned(&rep_pens, device));
            self.gpu_freq_pens = Some(h2d_f32_owned(&freq_pens, device));
            self.gpu_pres_pens = Some(h2d_f32_owned(&pres_pens, device));
        }

        // Rebuild output_token_ids every step (tokens grow).
        let mut max_output_len = 0usize;
        for req_id in batch_req_ids {
            if let Some(buf) = token_buffers.get(req_id) {
                let params = sampling_params_map.get(req_id);
                if params.is_some_and(|p| {
                    p.frequency_penalty != 0.0
                        || p.presence_penalty != 0.0
                        || p.repetition_penalty != 1.0
                }) {
                    max_output_len = max_output_len.max(buf.len());
                }
            }
        }

        if max_output_len == 0 {
            self.active = false;
            return;
        }

        self.max_output_len = max_output_len;
        let pad_val = self.vocab_size as i32;
        let mut output_ids_flat = vec![pad_val; num_reqs * max_output_len];
        for (i, req_id) in batch_req_ids.iter().enumerate() {
            if let Some(buf) = token_buffers.get(req_id) {
                let row = &mut output_ids_flat[i * max_output_len..];
                for (j, &tok) in buf.iter().enumerate() {
                    if j < max_output_len {
                        row[j] = tok as i32;
                    }
                }
            }
        }

        self.gpu_out_ids = Some(h2d_i32_owned(&output_ids_flat, device));

        // Dummy prompt IDs.
        if batch_update.is_some() || self.gpu_prompt_ids.is_none() {
            let dummy_prompt = vec![pad_val; num_reqs];
            self.gpu_prompt_ids = Some(h2d_i32_owned(&dummy_prompt, device));
        }
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(out_ids), Some(prompt_ids), Some(rep), Some(freq), Some(pres)) = (
            &self.gpu_out_ids,
            &self.gpu_prompt_ids,
            &self.gpu_rep_pens,
            &self.gpu_freq_pens,
            &self.gpu_pres_pens,
        ) {
            let gpu_out_ids_2d = unsafe {
                GpuTensor::new(
                    out_ids.as_gpu_tensor().raw_ptr(),
                    &[self.num_reqs, self.max_output_len],
                    GpuDType::U32,
                )
            };
            let gpu_prompt_ids_2d = unsafe {
                GpuTensor::new(
                    prompt_ids.as_gpu_tensor().raw_ptr(),
                    &[self.num_reqs, 1],
                    GpuDType::U32,
                )
            };
            unsafe {
                crate::kernels::apply_penalties(
                    logits,
                    gpu_out_ids_2d,
                    gpu_prompt_ids_2d,
                    rep.as_gpu_tensor(),
                    freq.as_gpu_tensor(),
                    pres.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// GrammarMaskProcessor
// ---------------------------------------------------------------------------

/// Grammar mask processor. Always rebuilds since FSM state changes every token.
/// Lives in scratchy-serving-cuda (no dependency on scratchy-core-models) — caller passes allowed
/// token lists directly via `update_from_allowed_tokens`.
#[derive(Default)]
pub struct GrammarMaskProcessor {
    active: bool,
    gpu_allowed: Option<OwnedTensor>,
    gpu_offsets: Option<OwnedTensor>,
    gpu_req_indices: Option<OwnedTensor>,
    /// Backup logits needed for restoring allowed token values.
    needs_backup: bool,
}

impl GrammarMaskProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update grammar state from pre-collected allowed token lists.
    /// Each entry is `(batch_index, allowed_token_ids)`.
    pub fn update_from_allowed_tokens(
        &mut self,
        grammar_reqs: &[(usize, &[u32])],
        device: &mut GpuDevice,
    ) {
        if grammar_reqs.is_empty() {
            self.active = false;
            self.gpu_allowed = None;
            self.gpu_offsets = None;
            self.gpu_req_indices = None;
            self.needs_backup = false;
            return;
        }

        let mut allowed_ids_flat: Vec<i32> = Vec::new();
        let mut allowed_offsets: Vec<i32> = vec![0];
        let mut req_indices: Vec<i32> = Vec::new();

        for &(req_idx, allowed) in grammar_reqs {
            req_indices.push(req_idx as i32);
            for &tok in allowed {
                allowed_ids_flat.push(tok as i32);
            }
            allowed_offsets.push(allowed_ids_flat.len() as i32);
        }

        self.active = true;
        self.needs_backup = true;
        self.gpu_allowed = Some(h2d_i32_owned(&allowed_ids_flat, device));
        self.gpu_offsets = Some(h2d_i32_owned(&allowed_offsets, device));
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
    }

    /// Whether the grammar processor needs a backup of logits before masking.
    pub fn needs_backup(&self) -> bool {
        self.needs_backup
    }

    /// Whether the grammar processor has work to do this step.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Apply grammar mask using a backup tensor for restoring allowed tokens.
    pub fn apply_with_backup(&self, logits: GpuTensor, backup: GpuTensor, device: &mut GpuDevice) {
        if let (Some(allowed), Some(offsets), Some(indices)) =
            (&self.gpu_allowed, &self.gpu_offsets, &self.gpu_req_indices)
        {
            unsafe {
                crate::kernels::apply_grammar_mask(
                    logits,
                    backup,
                    allowed.as_gpu_tensor(),
                    offsets.as_gpu_tensor(),
                    indices.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MinTokensProcessor
// ---------------------------------------------------------------------------

/// Suppresses stop/EOS tokens for requests that haven't generated min_tokens yet.
/// Mirrors Python's `MinTokensLogitsProcessor`.
#[derive(Default)]
pub struct MinTokensProcessor {
    active: bool,
    gpu_req_indices: Option<OwnedTensor>,
    gpu_token_ids: Option<OwnedTensor>,
}

impl MinTokensProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LogitsProcessor for MinTokensProcessor {
    fn update_state(
        &mut self,
        _batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        // Rebuild every step since token counts change.
        let mut req_indices: Vec<i32> = Vec::new();
        let mut token_ids: Vec<i32> = Vec::new();

        for (req_idx, req_id) in batch_req_ids.iter().enumerate() {
            if let Some(params) = sampling_params_map.get(req_id) {
                if params.min_tokens == 0 {
                    continue;
                }
                let generated = token_buffers.get(req_id).map_or(0, |buf| {
                    // token_buffers includes prompt tokens; generated count is
                    // total - prompt length. But we store output tokens only
                    // starting from decode, so buf.len() tracks total tokens
                    // including prompt. We need the output token count.
                    // Actually, token_buffers in CudaWorker stores prompt_ids
                    // initially then appends generated tokens. The "generated"
                    // count is tracked by the scheduler. For min_tokens, we
                    // check if the number of generated tokens (output tokens)
                    // is below min_tokens.
                    //
                    // For now, use a conservative approach: check buf length
                    // against a threshold. The caller should pass the correct
                    // generated token count.
                    buf.len()
                });
                // If we've generated fewer tokens than min_tokens, suppress stop tokens.
                // Note: generated includes prompt tokens in token_buffers. We need
                // the number of *output* tokens. The input_batch tracks this.
                // For the initial implementation, we'll use a field in BatchUpdate
                // or rely on the caller to filter. For now, we check against the
                // params and suppress EOS + stop_token_ids.
                if (generated as u32) < params.min_tokens {
                    // Suppress all stop_token_ids.
                    for &stop_id in &params.stop_token_ids {
                        req_indices.push(req_idx as i32);
                        token_ids.push(stop_id as i32);
                    }
                }
            }
        }

        if req_indices.is_empty() {
            self.active = false;
            self.gpu_req_indices = None;
            self.gpu_token_ids = None;
            return;
        }

        self.active = true;
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
        self.gpu_token_ids = Some(h2d_i32_owned(&token_ids, device));
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(indices), Some(ids)) = (&self.gpu_req_indices, &self.gpu_token_ids) {
            unsafe {
                crate::kernels::apply_min_tokens(
                    logits,
                    indices.as_gpu_tensor(),
                    ids.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false // Can change argmax by suppressing EOS
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// AllowedTokenIdsProcessor
// ---------------------------------------------------------------------------

/// Static allow-list processor. When a request has `allowed_token_ids`, only
/// those tokens may be sampled — all others are set to `-inf`.
/// Reuses the grammar mask kernel (CSR allow-list → set disallowed to `-inf`).
/// State is static per request — rebuilt only on batch change, not every step.
pub struct AllowedTokenIdsProcessor {
    active: bool,
    needs_backup: bool,
    gpu_allowed: Option<OwnedTensor>,
    gpu_offsets: Option<OwnedTensor>,
    gpu_req_indices: Option<OwnedTensor>,
}

impl Default for AllowedTokenIdsProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl AllowedTokenIdsProcessor {
    pub fn new() -> Self {
        Self {
            active: false,
            needs_backup: false,
            gpu_allowed: None,
            gpu_offsets: None,
            gpu_req_indices: None,
        }
    }

    /// Whether this processor needs a backup of logits before masking.
    pub fn needs_backup(&self) -> bool {
        self.needs_backup
    }

    /// Apply using a backup tensor (same pattern as grammar mask).
    pub fn apply_with_backup(&self, logits: GpuTensor, backup: GpuTensor, device: &mut GpuDevice) {
        if let (Some(allowed), Some(offsets), Some(indices)) =
            (&self.gpu_allowed, &self.gpu_offsets, &self.gpu_req_indices)
        {
            unsafe {
                crate::kernels::apply_grammar_mask(
                    logits,
                    backup,
                    allowed.as_gpu_tensor(),
                    offsets.as_gpu_tensor(),
                    indices.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }
}

impl LogitsProcessor for AllowedTokenIdsProcessor {
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        _token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        // Only rebuild when batch changes (static per request).
        if batch_update.is_none() && self.gpu_allowed.is_some() {
            return;
        }

        let mut allowed_ids_flat: Vec<i32> = Vec::new();
        let mut allowed_offsets: Vec<i32> = vec![0];
        let mut req_indices: Vec<i32> = Vec::new();

        for (req_idx, req_id) in batch_req_ids.iter().enumerate() {
            if let Some(params) = sampling_params_map.get(req_id)
                && let Some(ref allowed) = params.allowed_token_ids
            {
                req_indices.push(req_idx as i32);
                for &tok in allowed {
                    allowed_ids_flat.push(tok as i32);
                }
                allowed_offsets.push(allowed_ids_flat.len() as i32);
            }
        }

        if req_indices.is_empty() {
            self.active = false;
            self.needs_backup = false;
            self.gpu_allowed = None;
            self.gpu_offsets = None;
            self.gpu_req_indices = None;
            return;
        }

        self.active = true;
        self.needs_backup = true;
        self.gpu_allowed = Some(h2d_i32_owned(&allowed_ids_flat, device));
        self.gpu_offsets = Some(h2d_i32_owned(&allowed_offsets, device));
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
    }

    fn apply(&self, _logits: GpuTensor, _device: &mut GpuDevice) {
        // apply_with_backup is used instead (needs backup logits).
        // This is a no-op; CudaWorker calls apply_with_backup directly.
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// BadWordsProcessor
// ---------------------------------------------------------------------------

/// Suppresses tokens that would complete a "bad word" sequence.
/// CPU-side suffix matching against output tokens, then GPU scatter to `-inf`.
/// Mirrors Python's `BadWordsLogitsProcessor`.
#[derive(Default)]
pub struct BadWordsProcessor {
    active: bool,
    gpu_req_indices: Option<OwnedTensor>,
    gpu_token_ids: Option<OwnedTensor>,
}

impl BadWordsProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Check if `output_tokens` ends with the prefix of `bad_word` (all but the last token).
/// If so, return the completing token (the last token of the bad word).
fn bad_word_suffix_match(output_tokens: &[u32], bad_word: &[u32]) -> Option<u32> {
    if bad_word.is_empty() {
        return None;
    }
    // Single-token bad word: always suppress it.
    if bad_word.len() == 1 {
        return Some(bad_word[0]);
    }
    // Multi-token: check if output ends with bad_word[..len-1].
    let prefix = &bad_word[..bad_word.len() - 1];
    if output_tokens.len() >= prefix.len()
        && output_tokens[output_tokens.len() - prefix.len()..] == *prefix
    {
        Some(bad_word[bad_word.len() - 1])
    } else {
        None
    }
}

impl LogitsProcessor for BadWordsProcessor {
    fn update_state(
        &mut self,
        _batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        // Rebuild every step since output tokens grow.
        let mut req_indices: Vec<i32> = Vec::new();
        let mut token_ids: Vec<i32> = Vec::new();

        for (req_idx, req_id) in batch_req_ids.iter().enumerate() {
            if let Some(params) = sampling_params_map.get(req_id)
                && let Some(ref bad_words) = params.bad_words_token_ids
            {
                let output_tokens = token_buffers
                    .get(req_id)
                    .map(|b| b.as_slice())
                    .unwrap_or(&[]);
                for bad_word in bad_words {
                    if let Some(suppress_token) = bad_word_suffix_match(output_tokens, bad_word) {
                        req_indices.push(req_idx as i32);
                        token_ids.push(suppress_token as i32);
                    }
                }
            }
        }

        if req_indices.is_empty() {
            self.active = false;
            self.gpu_req_indices = None;
            self.gpu_token_ids = None;
            return;
        }

        self.active = true;
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
        self.gpu_token_ids = Some(h2d_i32_owned(&token_ids, device));
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(indices), Some(ids)) = (&self.gpu_req_indices, &self.gpu_token_ids) {
            unsafe {
                crate::kernels::apply_min_tokens(
                    logits,
                    indices.as_gpu_tensor(),
                    ids.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// SealPadProcessor
// ---------------------------------------------------------------------------

/// 🦭 Forces pad tokens after EOS for sealed requests until block-aligned.
///
/// When a sealed request emits an EOS token, the engine keeps it running.
/// This processor detects that EOS has been generated (by scanning
/// `token_buffers`) and forces the pad token on all subsequent steps until
/// `total_tokens % block_size == 0`. At that point the engine stops the
/// request normally and the now-full final block is cacheable.
///
/// Architecture: same as `AllowedTokenIdsProcessor` — standalone (not in the
/// pipeline), called via `apply_with_backup` from the worker, using the
/// grammar-mask kernel with `allowed = [pad_token]`.
pub struct SealPadProcessor {
    /// Model's EOS token IDs (from config.json).
    eos_token_ids: Vec<u32>,
    /// Token to force during seal-padding (e.g. tokenizer whitespace token).
    pad_token: u32,
    /// KV cache block size in tokens.
    block_size: usize,

    // --- per-step GPU state ---
    active: bool,
    needs_backup: bool,
    gpu_allowed: Option<OwnedTensor>,
    gpu_offsets: Option<OwnedTensor>,
    gpu_req_indices: Option<OwnedTensor>,

    /// Tracks which sealed requests are in padding mode (EOS/stop hit or
    /// max_tokens reached). Entries removed when request leaves the batch.
    seen_eos: HashMap<String, ()>,

    /// Prompt length per sealed request (set on first observation, used to
    /// compute output token count for max_tokens detection).
    prompt_lens: HashMap<String, usize>,
}

impl SealPadProcessor {
    pub fn new(eos_token_ids: Vec<u32>, pad_token: u32, block_size: usize) -> Self {
        Self {
            eos_token_ids,
            pad_token,
            block_size,
            active: false,
            needs_backup: false,
            gpu_allowed: None,
            gpu_offsets: None,
            gpu_req_indices: None,
            seen_eos: HashMap::new(),
            prompt_lens: HashMap::new(),
        }
    }

    /// Whether this processor needs a backup of logits before masking.
    pub fn needs_backup(&self) -> bool {
        self.needs_backup
    }

    /// Whether the processor has work to do this step.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Apply using a backup tensor (same kernel as grammar mask / allowed_token_ids).
    pub fn apply_with_backup(&self, logits: GpuTensor, backup: GpuTensor, device: &mut GpuDevice) {
        if let (Some(allowed), Some(offsets), Some(indices)) =
            (&self.gpu_allowed, &self.gpu_offsets, &self.gpu_req_indices)
        {
            unsafe {
                crate::kernels::apply_grammar_mask(
                    logits,
                    backup,
                    allowed.as_gpu_tensor(),
                    offsets.as_gpu_tensor(),
                    indices.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    /// CPU-side scan: update `seen_eos` tracking and return batch indices of
    /// requests that need padding (sealed, EOS seen, not yet block-aligned).
    ///
    /// Separated from `update_state` so the detection logic is testable
    /// without a GPU device.
    fn compute_padding_requests(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
    ) -> Vec<usize> {
        // Clean up tracking for requests that left the batch.
        if batch_update.is_some() {
            let current: std::collections::HashSet<&str> =
                batch_req_ids.iter().map(|s| s.as_str()).collect();
            self.seen_eos.retain(|k, _| current.contains(k.as_str()));
            self.prompt_lens.retain(|k, _| current.contains(k.as_str()));
        }

        // Scan sealed requests for stop conditions (EOS or max_tokens).
        for req_id in batch_req_ids {
            if self.seen_eos.contains_key(req_id) {
                continue;
            }
            let params = match sampling_params_map.get(req_id) {
                Some(p) if p.seal => p,
                _ => continue,
            };
            let buf = match token_buffers.get(req_id) {
                Some(b) => b,
                None => continue,
            };

            // Track prompt length on first observation.
            let prompt_len = *self.prompt_lens.entry(req_id.clone()).or_insert(buf.len());

            // Check EOS/stop tokens in OUTPUT only (not prompt).
            // Prompt tokens often contain EOS-like tokens from the chat
            // template (e.g. <|eot_id|>, <|im_end|>). Scanning the full
            // buffer would trigger padding from step 1, preventing the
            // model from ever generating a real EOS.
            let has_eos = buf[prompt_len..].iter().any(|&tok| {
                self.eos_token_ids.contains(&tok) || params.stop_token_ids.contains(&tok)
            });

            // Check max_tokens (output count = total - prompt).
            let output_count = buf.len().saturating_sub(prompt_len);
            let hit_max = params
                .max_tokens
                .is_some_and(|max| output_count >= max as usize);

            if has_eos || hit_max {
                self.seen_eos.insert(req_id.clone(), ());
            }
        }

        // Collect batch indices of requests needing padding.
        let mut padding_indices = Vec::new();
        for (req_idx, req_id) in batch_req_ids.iter().enumerate() {
            if !self.seen_eos.contains_key(req_id) {
                continue;
            }
            let buf = match token_buffers.get(req_id) {
                Some(b) => b,
                None => continue,
            };
            if buf.len() % self.block_size != 0 {
                padding_indices.push(req_idx);
            }
        }
        padding_indices
    }
}

impl LogitsProcessor for SealPadProcessor {
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        let padding_indices = self.compute_padding_requests(
            batch_update,
            sampling_params_map,
            token_buffers,
            batch_req_ids,
        );

        if padding_indices.is_empty() {
            self.active = false;
            self.needs_backup = false;
            self.gpu_allowed = None;
            self.gpu_offsets = None;
            self.gpu_req_indices = None;
            return;
        }

        // Build GPU state: CSR allow-list with pad_token for each request.
        let mut allowed_ids_flat: Vec<i32> = Vec::new();
        let mut allowed_offsets: Vec<i32> = vec![0];
        let mut req_indices: Vec<i32> = Vec::new();
        for req_idx in padding_indices {
            req_indices.push(req_idx as i32);
            allowed_ids_flat.push(self.pad_token as i32);
            allowed_offsets.push(allowed_ids_flat.len() as i32);
        }

        self.active = true;
        self.needs_backup = true;
        self.gpu_allowed = Some(h2d_i32_owned(&allowed_ids_flat, device));
        self.gpu_offsets = Some(h2d_i32_owned(&allowed_offsets, device));
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
    }

    fn apply(&self, _logits: GpuTensor, _device: &mut GpuDevice) {
        // apply_with_backup is used instead (needs backup logits).
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// H2D helpers (allocate via caching allocator + async copy)
// ---------------------------------------------------------------------------

fn h2d_i32_owned(data: &[i32], device: &mut GpuDevice) -> OwnedTensor {
    let owned = device.caching.alloc_tensor(&[data.len()], GpuDType::U32);
    unsafe {
        let _ = driver::memcpy_htod_async(
            owned.as_gpu_tensor().raw_ptr(),
            data.as_ptr() as *const u8,
            data.len() * 4,
            device.compute_stream,
        );
    }
    owned
}

fn h2d_f32_owned(data: &[f32], device: &mut GpuDevice) -> OwnedTensor {
    let owned = device.caching.alloc_tensor(&[data.len()], GpuDType::F32);
    unsafe {
        let _ = driver::memcpy_htod_async(
            owned.as_gpu_tensor().raw_ptr(),
            data.as_ptr() as *const u8,
            data.len() * 4,
            device.compute_stream,
        );
    }
    owned
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_sorting() {
        let pipeline = LogitsProcessorPipeline::new(vec![
            Box::new(LogitBiasProcessor::new()),
            Box::new(MinTokensProcessor::new()),
        ]);
        // Both are non-argmax-invariant.
        assert_eq!(pipeline.non_argmax_invariant.len(), 2);
        assert_eq!(pipeline.argmax_invariant.len(), 0);
    }

    #[test]
    fn test_pipeline_inactive_by_default() {
        let pipeline = LogitsProcessorPipeline::new(vec![
            Box::new(LogitBiasProcessor::new()),
            Box::new(PenaltiesProcessor::new(32000)),
            Box::new(MinTokensProcessor::new()),
        ]);
        assert!(!pipeline.any_active());
    }

    #[test]
    fn test_bad_word_suffix_match_single_token() {
        // Single-token bad word: always suppress.
        assert_eq!(bad_word_suffix_match(&[], &[42]), Some(42));
        assert_eq!(bad_word_suffix_match(&[1, 2, 3], &[42]), Some(42));
    }

    #[test]
    fn test_bad_word_suffix_match_multi_token() {
        // "bad word" = [10, 20, 30]. Prefix = [10, 20].
        // Output ends with [10, 20] → suppress 30.
        assert_eq!(bad_word_suffix_match(&[5, 10, 20], &[10, 20, 30]), Some(30));
        // Output does NOT end with [10, 20] → no suppression.
        assert_eq!(bad_word_suffix_match(&[5, 10, 21], &[10, 20, 30]), None);
        // Output too short for prefix.
        assert_eq!(bad_word_suffix_match(&[20], &[10, 20, 30]), None);
    }

    #[test]
    fn test_bad_word_suffix_match_empty() {
        assert_eq!(bad_word_suffix_match(&[1, 2], &[]), None);
    }

    #[test]
    fn test_bad_word_suffix_match_exact_prefix() {
        // Output exactly equals the prefix.
        assert_eq!(bad_word_suffix_match(&[10, 20], &[10, 20, 30]), Some(30));
    }

    #[test]
    fn test_allowed_token_ids_processor_default_inactive() {
        let p = AllowedTokenIdsProcessor::new();
        assert!(!p.is_active());
        assert!(!p.needs_backup());
    }

    #[test]
    fn test_bad_words_processor_default_inactive() {
        let p = BadWordsProcessor::new();
        assert!(!p.is_active());
    }

    #[test]
    fn test_seeded_rng_deterministic() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let vals1: Vec<f32> = (0..10).map(|_| rng1.random::<f32>()).collect();
        let vals2: Vec<f32> = (0..10).map(|_| rng2.random::<f32>()).collect();
        assert_eq!(vals1, vals2, "same seed must produce same values");
    }

    #[test]
    fn test_seeded_rng_different_seeds_differ() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(99);
        let vals1: Vec<f32> = (0..10).map(|_| rng1.random::<f32>()).collect();
        let vals2: Vec<f32> = (0..10).map(|_| rng2.random::<f32>()).collect();
        assert_ne!(
            vals1, vals2,
            "different seeds must produce different values"
        );
    }

    #[test]
    fn test_seeded_rng_advances_state() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // Verify that successive calls produce different values (RNG advances).
        let mut rng = StdRng::seed_from_u64(42);
        let v1: f32 = rng.random();
        let v2: f32 = rng.random();
        assert_ne!(v1, v2, "successive calls should produce different values");
    }

    // -----------------------------------------------------------------------
    // SealPadProcessor tests (CPU-side logic via compute_padding_requests)
    // -----------------------------------------------------------------------

    fn sealed_params() -> SamplingParams {
        SamplingParams {
            seal: true,
            max_tokens: Some(100),
            ..Default::default()
        }
    }

    fn normal_params() -> SamplingParams {
        SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        }
    }

    #[test]
    fn test_seal_pad_processor_default_inactive() {
        let p = SealPadProcessor::new(vec![2], 0, 4);
        assert!(!p.is_active());
        assert!(!p.needs_backup());
    }

    #[test]
    fn test_seal_pad_no_sealed_requests() {
        // Non-sealed requests should never trigger padding.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), normal_params())].into_iter().collect();
        // Buffer has EOS token 2, but request is not sealed.
        let bufs: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 20, 2])].into_iter().collect();
        let ids = vec!["r1".into()];

        let padding = p.compute_padding_requests(None, &params, &bufs, &ids);
        assert!(padding.is_empty());
        assert!(p.seen_eos.is_empty());
    }

    #[test]
    fn test_seal_pad_detects_eos() {
        // Sealed request with EOS in output → seen_eos tracked, padding needed.
        // Two-step flow: first call establishes prompt_len, second detects EOS.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 1: prompt only (establishes prompt_len = 2).
        let bufs1: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs1, &ids);
        assert!(padding.is_empty(), "no output yet");

        // Step 2: prompt + EOS output — 3 tokens, not block-aligned (block_size=4).
        let bufs2: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 20, 2])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert_eq!(padding, vec![0], "r1 at batch index 0 should need padding");
        assert!(p.seen_eos.contains_key("r1"));
    }

    #[test]
    fn test_seal_pad_block_aligned_no_padding() {
        // Sealed request with EOS, but already block-aligned → no padding needed.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 1: prompt (2 tokens).
        let bufs1: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20])].into_iter().collect();
        p.compute_padding_requests(None, &params, &bufs1, &ids);

        // Step 2: prompt + EOS + pad = 4 tokens = block-aligned.
        let bufs2: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 20, 2, 0])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert!(padding.is_empty(), "block-aligned should not need padding");
        assert!(p.seen_eos.contains_key("r1"), "EOS should still be tracked");
    }

    #[test]
    fn test_seal_pad_no_eos_yet() {
        // Sealed request but EOS not yet generated → no padding.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let bufs: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 20, 30])].into_iter().collect();
        let ids = vec!["r1".into()];

        let padding = p.compute_padding_requests(None, &params, &bufs, &ids);
        assert!(padding.is_empty());
        assert!(!p.seen_eos.contains_key("r1"));
    }

    #[test]
    fn test_seal_pad_stop_token_triggers_padding() {
        // SealPadProcessor should also detect stop_token_ids (not just EOS).
        let mut p = SealPadProcessor::new(vec![2], 0, 4); // EOS=2
        let mut sp = sealed_params();
        sp.stop_token_ids = vec![99]; // custom stop token
        let params: HashMap<String, SamplingParams> = [("r1".into(), sp)].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 1: prompt only.
        let bufs1: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20])].into_iter().collect();
        p.compute_padding_requests(None, &params, &bufs1, &ids);

        // Step 2: prompt + stop token 99 (not EOS 2).
        let bufs2: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 20, 99])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert_eq!(padding, vec![0]);
    }

    #[test]
    fn test_seal_pad_mixed_batch() {
        // Batch with sealed+EOS, sealed+no-EOS, and non-sealed requests.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> = [
            ("sealed_eos".into(), sealed_params()),
            ("sealed_no_eos".into(), sealed_params()),
            ("normal".into(), normal_params()),
        ]
        .into_iter()
        .collect();
        let ids = vec!["sealed_eos".into(), "sealed_no_eos".into(), "normal".into()];

        // Step 1: all prompts (establish prompt_len for each).
        let bufs1: HashMap<String, Vec<u32>> = [
            ("sealed_eos".into(), vec![10, 20]),
            ("sealed_no_eos".into(), vec![10, 20]),
            ("normal".into(), vec![10, 20]),
        ]
        .into_iter()
        .collect();
        p.compute_padding_requests(None, &params, &bufs1, &ids);

        // Step 2: sealed_eos gets EOS output, others get normal output.
        let bufs2: HashMap<String, Vec<u32>> = [
            ("sealed_eos".into(), vec![10, 20, 2]), // EOS output, not aligned
            ("sealed_no_eos".into(), vec![10, 20, 30]), // no EOS output
            ("normal".into(), vec![10, 20, 2]),     // EOS output but not sealed
        ]
        .into_iter()
        .collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        // Only sealed_eos (batch index 0) needs padding.
        assert_eq!(padding, vec![0]);
    }

    #[test]
    fn test_seal_pad_progressive_padding() {
        // Simulate multiple steps of padding until block-aligned.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 0: prompt (4 tokens, block-aligned). Establishes prompt_len=4.
        let bufs0: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 20, 30, 40])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs0, &ids);
        assert!(padding.is_empty(), "prompt only, no output");

        // Step 1: 5 tokens, EOS at position 4 → needs 3 pads to reach 8.
        let bufs1: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20, 30, 40, 2])]
            .into_iter()
            .collect();
        let padding = p.compute_padding_requests(None, &params, &bufs1, &ids);
        assert_eq!(padding, vec![0]);

        // Step 2: 6 tokens (one pad added) → still needs 2 more.
        let bufs2: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20, 30, 40, 2, 0])]
            .into_iter()
            .collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert_eq!(padding, vec![0]);

        // Step 3: 7 tokens → still needs 1 more.
        let bufs3: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20, 30, 40, 2, 0, 0])]
            .into_iter()
            .collect();
        let padding = p.compute_padding_requests(None, &params, &bufs3, &ids);
        assert_eq!(padding, vec![0]);

        // Step 4: 8 tokens → block-aligned, no more padding.
        let bufs4: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 20, 30, 40, 2, 0, 0, 0])]
            .into_iter()
            .collect();
        let padding = p.compute_padding_requests(None, &params, &bufs4, &ids);
        assert!(padding.is_empty());
    }

    #[test]
    fn test_seal_pad_seen_eos_cleanup_on_batch_change() {
        // When a request leaves the batch, its seen_eos entry should be cleaned up.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 1: prompt only.
        let bufs1: HashMap<String, Vec<u32>> = [("r1".into(), vec![10])].into_iter().collect();
        p.compute_padding_requests(None, &params, &bufs1, &ids);

        // Step 2: r1 sees EOS in output.
        let bufs2: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 2])].into_iter().collect();
        p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert!(p.seen_eos.contains_key("r1"));

        // Step 3: r1 leaves the batch (batch_update signals change).
        let update = BatchUpdate {
            batch_size: 0,
            added: vec![],
            removed: vec![0],
        };
        let empty_ids: Vec<String> = vec![];
        p.compute_padding_requests(Some(&update), &params, &bufs2, &empty_ids);
        assert!(!p.seen_eos.contains_key("r1"), "should be cleaned up");
    }

    #[test]
    fn test_seal_pad_seen_eos_persists_without_batch_change() {
        // Without a batch_update, seen_eos should persist across steps.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 1: prompt only.
        let bufs0: HashMap<String, Vec<u32>> = [("r1".into(), vec![10])].into_iter().collect();
        p.compute_padding_requests(None, &params, &bufs0, &ids);

        // Step 2: EOS detected in output.
        let bufs1: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 2])].into_iter().collect();
        p.compute_padding_requests(None, &params, &bufs1, &ids);
        assert!(p.seen_eos.contains_key("r1"));

        // Step 3: No batch_update, buffer grows but no new EOS scan needed.
        let bufs2: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 2, 0])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert_eq!(padding, vec![0], "should still need padding");
        assert!(p.seen_eos.contains_key("r1"), "should persist");
    }

    #[test]
    fn test_seal_pad_eos_in_prompt_ignored() {
        // EOS tokens in the PROMPT should NOT trigger padding.
        // This is the core bug fix: chat templates include EOS-like tokens
        // (e.g. <|eot_id|>, <|im_end|>) between turns, and scanning the
        // entire buffer would wrongly activate padding from step 1.
        let mut p = SealPadProcessor::new(vec![2], 0, 4);
        let params: HashMap<String, SamplingParams> =
            [("r1".into(), sealed_params())].into_iter().collect();
        let ids = vec!["r1".into()];

        // Step 1: prompt contains EOS token (id=2) from chat template.
        // prompt_len is established as 3.
        let bufs1: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 2, 20])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs1, &ids);
        assert!(padding.is_empty(), "EOS in prompt must not trigger padding");
        assert!(!p.seen_eos.contains_key("r1"), "must not mark seen_eos");

        // Step 2: model generates a normal (non-EOS) token — still no padding.
        let bufs2: HashMap<String, Vec<u32>> =
            [("r1".into(), vec![10, 2, 20, 50])].into_iter().collect();
        let padding = p.compute_padding_requests(None, &params, &bufs2, &ids);
        assert!(padding.is_empty(), "no EOS in output");

        // Step 3: model generates actual EOS — NOW padding should trigger.
        let bufs3: HashMap<String, Vec<u32>> = [("r1".into(), vec![10, 2, 20, 50, 2])]
            .into_iter()
            .collect();
        let padding = p.compute_padding_requests(None, &params, &bufs3, &ids);
        assert_eq!(padding, vec![0], "real EOS in output triggers padding");
        assert!(p.seen_eos.contains_key("r1"));
    }
}
