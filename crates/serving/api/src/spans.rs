// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Span-query execution and optimization — the pure engine-side logic.
//!
//! Tokenizes a SPNL span query with the model tokenizer, produces the block
//! annotations for relocatable KV caching, and executes it in-process
//! (`execute_spnl_query_sync`, used by `LLM::chat_stream`). This module carries
//! NO serving/HTTP code — the axum handlers for `POST /v1/query/execute` and
//! the Anthropic spans path live in [`crate::query`] (behind the `serve`
//! feature) and call into the pure helpers here.

use std::sync::Arc;

use spnl_core::ir::{Generate, Message, Query as SpnlQuery};
use spnl_core::optimizer::llo::llir::{NonGenerateInput, SingleGenerate};

use crate::chat_template::TemplateMessage;
use crate::error::ServeResult;
use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Span configuration
// ---------------------------------------------------------------------------

pub(crate) struct SpanConfig {
    pub(crate) pad_token: Option<u32>,
    pub(crate) block_size: usize,
}

impl SpanConfig {
    #[allow(dead_code)]
    pub(crate) fn with_pad_token(block_size: usize, pad_token: u32) -> Self {
        Self {
            pad_token: Some(pad_token),
            block_size,
        }
    }

    /// Resolve pad token from `VLLM_V1_SPANS_PAD_TOKEN` env var, falling back
    /// to the tokenizer's encoding of `" "` (whitespace), then 0.
    ///
    /// Set `VLLM_V1_SPANS_PAD_TOKEN=-1` to disable padding entirely.
    pub(crate) fn from_tokenizer(
        block_size: usize,
        _tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Self {
        // Default: no padding. Padding between Plus children injects tokens
        // that corrupt text content and destroy model accuracy (verified by
        // NIAH bench). Set VLLM_V1_SPANS_PAD_TOKEN to a token ID to enable
        // padding (e.g. for synthetic benchmarks like `bench spans` that use
        // pre-aligned token sequences).
        let pad_token =
            std::env::var("VLLM_V1_SPANS_PAD_TOKEN")
                .ok()
                .and_then(|v| match v.parse::<i64>() {
                    Ok(-1) => None,
                    Ok(id) if id >= 0 => Some(id as u32),
                    _ => None,
                });

        tracing::info!("[SPANS] pad_token={pad_token:?}, block_size={block_size}");
        Self {
            pad_token,
            block_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenization state and helpers
// ---------------------------------------------------------------------------

use scratchy_core_common::BlockKind;
use std::collections::BTreeMap;

/// Accumulated state during recursive tokenization.
struct TokenizeState {
    tokens: Vec<u32>,
    annotations: BTreeMap<usize, BlockKind>,
    /// Whether we're currently inside a relocatable context.
    in_relocatable: bool,
    /// When true, the next message must NOT be consolidated with the previous
    /// one, even if they share the same role. Set at structural boundaries
    /// (e.g. Cross left→right) to preserve message turn structure.
    break_consolidation: bool,
    /// Messages rendered so far for incremental tokenization.
    /// Each `tokenize_message` call appends to this and re-renders the full
    /// conversation, taking only the delta tokens. This avoids spurious BOS
    /// tokens between messages in a Seq.
    rendered_messages: Vec<TemplateMessage>,
    /// Number of token IDs produced by the last full render of
    /// `rendered_messages` — the delta starts after this offset.
    rendered_token_count: usize,
}

impl TokenizeState {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            annotations: BTreeMap::new(),
            in_relocatable: false,
            break_consolidation: false,
            rendered_messages: Vec::new(),
            rendered_token_count: 0,
        }
    }

    /// Align `tokens` to a block boundary so each span's blocks are
    /// content-addressable (shareable). With a pad_token, pad forward; with
    /// `pad_token=None` (the default — pad tokens corrupt NIAH accuracy),
    /// TRUNCATE the partial-block tail (the only no-corruption path to
    /// alignment, matching vllm-rs). Without this, a relocatable fragment
    /// ending mid-block leaves its annotation at an unaligned offset → the
    /// block never hashes to a reusable id → zero span reuse.
    fn align_to_block(&mut self, cfg: &SpanConfig) {
        let remainder = self.tokens.len() % cfg.block_size;
        if remainder == 0 {
            return;
        }
        if let Some(pad_token) = cfg.pad_token {
            let pad_count = cfg.block_size - remainder;
            self.tokens
                .extend(std::iter::repeat_n(pad_token, pad_count));
        } else {
            self.tokens.truncate(self.tokens.len() - remainder);
        }
    }

    /// Align to block boundary and record an annotation for the next block.
    fn annotate_next_block(&mut self, relocatable: bool, cfg: &SpanConfig) {
        // NO cropping/padding. A span's partial last block is simply part of the
        // span — the label carries the span's first TOKEN (the current offset),
        // so the kernel's block-diagonal lower bound is exact and the span needs
        // no alignment. (The old default truncated the tail to a block multiple,
        // dropping tool content.)
        let block_index = self.tokens.len() / cfg.block_size;
        let kind = if relocatable {
            BlockKind::Relocatable {
                first_token: self.tokens.len() as u32,
            }
        } else {
            BlockKind::Prefixed {
                first_token: self.tokens.len() as u32,
            }
        };
        self.annotations.insert(block_index, kind);
    }
}

/// Tokenize a single message using the chat template and tokenizer.
///
/// Uses incremental rendering: the message is appended to the conversation
/// accumulated in `state.rendered_messages`, the full conversation is
/// re-rendered, and only the delta tokens (new tokens beyond the previous
/// render) are added to `state.tokens`. This avoids spurious BOS tokens
/// when multiple messages appear in a Seq.
fn tokenize_message(
    msg: &Message,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    let role = msg.role().to_string();
    let content = msg.content().to_string();

    // Consolidate consecutive messages with the same role — avoids inserting
    // redundant role header tokens between chunks of the same type (e.g.
    // multiple user messages from Plus children). Consolidation is suppressed
    // at structural boundaries (break_consolidation flag).
    let should_consolidate = !state.break_consolidation
        && state
            .rendered_messages
            .last()
            .is_some_and(|last| last.role == role);
    state.break_consolidation = false;

    if should_consolidate {
        let last = state.rendered_messages.last_mut().unwrap();
        last.content.push('\n');
        last.content.push_str(&content);
    } else {
        state
            .rendered_messages
            .push(TemplateMessage { role, content });
    }
    let rendered = template.apply_simple(&state.rendered_messages, false)?;
    let all_ids = tokenizer.encode(&rendered, false)?;

    // When consolidating, the previous rendering's template suffix (e.g.,
    // <|eot_id|> in Llama 3, <|im_end|>\n in ChatML) is still in
    // state.tokens but has shifted position in the new rendering. Fix by
    // detecting and removing the stale suffix before computing the delta.
    if should_consolidate && state.rendered_token_count > 0 {
        let old_count = state.rendered_token_count;
        let state_len = state.tokens.len();
        // Check up to 32 trailing tokens for stale template suffix.
        // Real tokenizers: 1-3 tokens (e.g., <|eot_id|> = 1, <|im_end|>\n = 2).
        // Byte-level test tokenizer: up to ~12 tokens.
        let max_check = old_count.min(state_len).min(32);
        let mut stale_count = 0;
        for k in 1..=max_check {
            let si = state_len - k;
            let ai = old_count - k;
            if ai < all_ids.len() && state.tokens[si] != all_ids[ai] {
                stale_count = k;
            } else {
                break;
            }
        }
        if stale_count > 0 {
            state.tokens.truncate(state_len - stale_count);
            state.rendered_token_count -= stale_count;
        }
    }

    // Take only the delta — tokens added by this message.
    let new_ids = &all_ids[state.rendered_token_count..];
    state.rendered_token_count = all_ids.len();

    match msg {
        Message::Assistant(_) => {
            // For assistant messages, crop to block boundary (drop suffix tokens).
            // This matches spnl's extend_crop behavior.
            let end = new_ids.len() + state.tokens.len();
            let nearest_block_boundary = end / cfg.block_size * cfg.block_size;
            let amount_to_crop =
                std::cmp::min(new_ids.len(), end.saturating_sub(nearest_block_boundary));
            let extra_end = new_ids.len() - amount_to_crop;
            state.tokens.extend_from_slice(&new_ids[..extra_end]);
        }
        _ => {
            state.tokens.extend_from_slice(new_ids);
        }
    }
    Ok(())
}

/// Recursively tokenize a NonGenerateInput tree.
fn tokenize_input(
    input: &NonGenerateInput,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    match input {
        NonGenerateInput::Seq(v) | NonGenerateInput::Par(v) => {
            for child in v {
                tokenize_input(child, tokenizer, template, cfg, state)?;
            }
        }

        NonGenerateInput::Cross(v) => {
            let (left, right) = v.split_at(v.len().saturating_sub(1));
            for child in left {
                tokenize_input(child, tokenizer, template, cfg, state)?;
            }
            if !right.is_empty() {
                state.in_relocatable = false;
                state.break_consolidation = true;
                state.annotate_next_block(false, cfg);
                for child in right {
                    tokenize_input(child, tokenizer, template, cfg, state)?;
                }
            }
        }

        NonGenerateInput::Plus(v) => {
            let prev_in_relocatable = state.in_relocatable;
            for child in v {
                state.annotate_next_block(true, cfg);
                state.in_relocatable = true;
                tokenize_input(child, tokenizer, template, cfg, state)?;
            }
            state.in_relocatable = prev_in_relocatable;
            if !prev_in_relocatable {
                // Content after this Plus is ordinary label-0 text. Without a
                // Prefixed boundary here it silently extends the last child's
                // relocatable span — mid-Cross fresh regions (e.g. the live
                // prompt between context spans) were impossible before this.
                state.annotate_next_block(false, cfg);
            }
        }

        NonGenerateInput::Message(msg) => {
            tokenize_message(msg, tokenizer, template, cfg, state)?;
        }
    }
    Ok(())
}

/// Add the final assistant generation prompt.
///
/// Uses the accumulated conversation in `state.rendered_messages` to extract
/// the generation prompt in context. If no messages have been accumulated
/// (e.g. after an annotation boundary reset), falls back to a dummy user
/// message to avoid template crashes on empty message lists.
fn add_generation_prompt(
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    state: &mut TokenizeState,
) -> ServeResult<()> {
    // If we're inside a relocatable context, add another relocatable boundary
    // for the generation prompt so it gets its own block.
    if state.in_relocatable {
        state.annotate_next_block(true, cfg);
    }

    // Use the actual accumulated messages for correct gen prompt extraction.
    // If empty (e.g. after annotation boundary reset), use a dummy.
    let messages = if state.rendered_messages.is_empty() {
        vec![TemplateMessage {
            role: "user".to_string(),
            content: "x".to_string(),
        }]
    } else {
        state.rendered_messages.clone()
    };

    // Render the conversation with add_generation_prompt=true and take the
    // token delta from the current rendered_token_count. This ensures the
    // gen prompt tokens are computed identically to how tokenize_message
    // computes message deltas (full-string encoding, not isolated substring),
    // avoiding BPE boundary mismatches.
    let with = template.apply_simple(&messages, true)?;
    let all_ids = tokenizer.encode(&with, false)?;
    let new_ids = &all_ids[state.rendered_token_count..];
    if !new_ids.is_empty() {
        state.tokens.extend_from_slice(new_ids);
        state.rendered_token_count = all_ids.len();
    }
    Ok(())
}

/// Result of tokenizing a span query: tokens + block annotations.
pub(crate) struct SpanTokenized {
    pub(crate) tokens: Vec<u32>,
    pub(crate) annotations: Option<BTreeMap<usize, BlockKind>>,
}

/// Tokenize a full SingleGenerate into a token sequence with annotations.
pub(crate) fn tokenize_span_query(
    spec: &SingleGenerate,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<SpanTokenized> {
    let mut state = TokenizeState::new();
    tokenize_input(&spec.input, tokenizer, template, cfg, &mut state)?;
    add_generation_prompt(tokenizer, template, cfg, &mut state)?;
    let annotations = if state.annotations.is_empty() {
        None
    } else {
        Some(state.annotations)
    };
    Ok(SpanTokenized {
        tokens: state.tokens,
        annotations,
    })
}

/// Tokenize a map input (user message with relocatable prefix, padded).
pub(crate) fn tokenize_map_input(
    text: &str,
    tokenizer: &Tokenizer,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<SpanTokenized> {
    let mut state = TokenizeState::new();
    state.annotate_next_block(true, cfg);
    tokenize_message(
        &Message::User(text.to_string()),
        tokenizer,
        template,
        cfg,
        &mut state,
    )?;
    state.align_to_block(cfg);
    let annotations = if state.annotations.is_empty() {
        None
    } else {
        Some(state.annotations)
    };
    Ok(SpanTokenized {
        tokens: state.tokens,
        annotations,
    })
}

// ---------------------------------------------------------------------------
// SSE streaming (mirrors server.rs stream_completion_response)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Return a short name for a query variant (for error messages).
pub(crate) fn query_variant_name(query: &SpnlQuery) -> &'static str {
    match query {
        SpnlQuery::Generate(_) => "Generate",
        SpnlQuery::Seq(_) => "Seq",
        SpnlQuery::Par(_) => "Par",
        SpnlQuery::Cross(_) => "Cross",
        SpnlQuery::Plus(_) => "Plus",
        SpnlQuery::Monad(_) => "Monad",
        SpnlQuery::Bulk(_) => "Bulk",
        SpnlQuery::Message(_) => "Message",
        SpnlQuery::Zip(_) => "Zip",
        #[cfg(feature = "rag")]
        SpnlQuery::Augment(_) => "Augment",
    }
}

/// Recursively rewrite `Augment` nodes into `Plus(Message(...))` fragments
/// by retrieving from the pre-built LEANN index.
#[cfg(feature = "rag")]
pub(crate) fn optimize_augments<'a>(
    query: &'a SpnlQuery,
    options: &'a crate::augment::AugmentOptions,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<SpnlQuery>> + Send + 'a>> {
    Box::pin(async move {
        match query {
            SpnlQuery::Augment(a) => {
                let fragments =
                    crate::augment::retrieve(&a.embedding_model, &a.body, &a.doc, options).await?;
                let fragment_nodes: Vec<SpnlQuery> = fragments
                    .into_iter()
                    .map(|s| SpnlQuery::Message(Message::User(s)))
                    .collect();
                Ok(SpnlQuery::Plus(fragment_nodes))
            }
            SpnlQuery::Generate(g) => {
                let optimized_input = Box::new(optimize_augments(&g.input, options).await?);
                Ok(SpnlQuery::Generate(Generate {
                    metadata: g.metadata.clone(),
                    input: optimized_input,
                }))
            }
            SpnlQuery::Seq(v) => {
                let mut out = Vec::with_capacity(v.len());
                for child in v {
                    out.push(optimize_augments(child, options).await?);
                }
                Ok(SpnlQuery::Seq(out))
            }
            SpnlQuery::Plus(v) => {
                let mut out = Vec::with_capacity(v.len());
                for child in v {
                    out.push(optimize_augments(child, options).await?);
                }
                Ok(SpnlQuery::Plus(out))
            }
            SpnlQuery::Cross(v) => {
                let mut out = Vec::with_capacity(v.len());
                for child in v {
                    out.push(optimize_augments(child, options).await?);
                }
                Ok(SpnlQuery::Cross(out))
            }
            other => Ok(other.clone()),
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_template::ChatTemplate;

    #[test]
    fn align_to_block_truncates_without_pad_pads_with() {
        let bs = 16usize;
        // pad_token=None (default) → truncate the partial-block tail.
        let mut s = TokenizeState::new();
        s.tokens = (0..(bs as u32 + 5)).collect(); // 21 tokens
        s.align_to_block(&SpanConfig {
            pad_token: None,
            block_size: bs,
        });
        assert_eq!(s.tokens.len(), bs, "no-pad: truncate to block boundary");
        assert_eq!(s.tokens.len() % bs, 0);

        // pad_token=Some → pad forward to the next boundary.
        let mut s2 = TokenizeState::new();
        s2.tokens = (0..(bs as u32 + 5)).collect();
        s2.align_to_block(&SpanConfig {
            pad_token: Some(0),
            block_size: bs,
        });
        assert_eq!(s2.tokens.len(), 2 * bs, "pad: extend to next boundary");
        assert_eq!(s2.tokens.len() % bs, 0);

        // Already aligned → no-op in both modes.
        let mut s3 = TokenizeState::new();
        s3.tokens = (0..(2 * bs as u32)).collect();
        s3.align_to_block(&SpanConfig {
            pad_token: None,
            block_size: bs,
        });
        assert_eq!(s3.tokens.len(), 2 * bs, "aligned: no-op");
    }

    /// Regression test for the stale template-suffix bug.
    ///
    /// When messages are consolidated (same role, appended with \n), the
    /// template's end-of-turn token (e.g., Llama 3's `<|eot_id|>` = 128009)
    /// from the PREVIOUS rendering remains in `state.tokens` even though it
    /// has shifted in the new rendering. Without the fix, these stale tokens
    /// corrupt the token sequence and destroy model accuracy.
    ///
    /// This test uses synthetic token arrays to exercise the fix logic
    /// directly, simulating what happens with a real tokenizer.
    #[test]
    fn test_stale_suffix_removal_on_consolidation() {
        // Simulate Llama 3 tokenization:
        // Step 1: [{system: "hi"}, {user: "A"}]
        //   Tokens: [BOS, SYS_HEAD, .., EOT, USR_HEAD, .., A, DOT, EOT]
        //   Simplified: [10, 20, 30, 999, 40, 50, 60, 70, 999]
        //   where 999 = <|eot_id|>
        let ids_step1: Vec<u32> = vec![10, 20, 30, 999, 40, 50, 60, 70, 999];

        // Step 2: [{system: "hi"}, {user: "A\nB"}] (consolidated)
        //   Tokens: [BOS, SYS_HEAD, .., EOT, USR_HEAD, .., A, DOT, NL, B, EOT]
        //   Simplified: [10, 20, 30, 999, 40, 50, 60, 70, 80, 90, 999]
        //   The content "A" is extended to "A\nB", so after "A" and "DOT" we
        //   get NL=80, B=90, then EOT=999 at the end.
        let ids_step2: Vec<u32> = vec![10, 20, 30, 999, 40, 50, 60, 70, 80, 90, 999];

        // Key observation: ids_step1 and ids_step2 share prefix [10..70].
        // ids_step1[8] = 999 (EOT), ids_step2[8] = 80 (NL) — MISMATCH.

        // Without fix: state.tokens = ids_step1, delta = ids_step2[9..] = [90, 999]
        let old_count = ids_step1.len(); // 9
        let delta_buggy = &ids_step2[old_count..]; // [90, 999]
        let mut buggy = ids_step1.clone();
        buggy.extend_from_slice(delta_buggy);
        // buggy = [10, 20, 30, 999, 40, 50, 60, 70, 999, 90, 999]
        // The stale 999 (EOT) at position 8 corrupts the sequence!

        assert_ne!(
            buggy, ids_step2,
            "Without fix, stale EOT (999) must cause a mismatch",
        );
        assert!(
            buggy.windows(3).any(|w| w == [999, 90, 999]),
            "Buggy sequence should have stale EOT between content: {:?}",
            buggy,
        );

        // With fix: detect stale suffix by scanning backwards.
        let mut fixed = ids_step1.clone();
        let state_len = fixed.len();
        let max_check = old_count.min(state_len).min(32);
        let mut stale_count = 0;
        for k in 1..=max_check {
            let si = state_len - k;
            let ai = old_count - k;
            if ai < ids_step2.len() && fixed[si] != ids_step2[ai] {
                stale_count = k;
            } else {
                break;
            }
        }
        assert_eq!(stale_count, 1, "Should detect 1 stale suffix token (EOT)");
        fixed.truncate(state_len - stale_count);
        let adjusted_count = old_count - stale_count;
        let delta_fixed = &ids_step2[adjusted_count..];
        fixed.extend_from_slice(delta_fixed);

        assert_eq!(
            fixed, ids_step2,
            "After fix, tokens must match one-shot rendering.\n\
             Fixed:    {:?}\n\
             Expected: {:?}",
            fixed, ids_step2,
        );
    }

    /// Verify the fix handles multi-token template suffixes (e.g., ChatML's
    /// `<|im_end|>\n` which is 2 tokens with a proper tokenizer).
    #[test]
    fn test_stale_suffix_removal_multi_token() {
        // Simulate ChatML: suffix = [IM_END=500, NL=10]
        let ids_step1: Vec<u32> = vec![1, 2, 3, 100, 200, 300, 500, 10];
        let ids_step2: Vec<u32> = vec![1, 2, 3, 100, 200, 300, 50, 400, 500, 10];
        // ids_step1[6] = 500 (IM_END), ids_step2[6] = 50 (content continuation)
        // ids_step1[7] = 10 (NL),      ids_step2[7] = 400 (more content)
        // Stale suffix = 2 tokens [500, 10]

        let old_count = ids_step1.len();
        let mut fixed = ids_step1.clone();
        let state_len = fixed.len();
        let max_check = old_count.min(state_len).min(32);
        let mut stale_count = 0;
        for k in 1..=max_check {
            let si = state_len - k;
            let ai = old_count - k;
            if ai < ids_step2.len() && fixed[si] != ids_step2[ai] {
                stale_count = k;
            } else {
                break;
            }
        }
        assert_eq!(stale_count, 2, "Should detect 2 stale suffix tokens");
        fixed.truncate(state_len - stale_count);
        let adjusted_count = old_count - stale_count;
        fixed.extend_from_slice(&ids_step2[adjusted_count..]);

        assert_eq!(fixed, ids_step2);
    }

    /// Verify the fix is a no-op when no consolidation occurs (different roles).
    #[test]
    fn test_no_stale_suffix_without_consolidation() {
        // When consecutive messages have different roles, there's no
        // consolidation and no stale suffix. The fix should be a no-op.
        let ids_step1: Vec<u32> = vec![10, 20, 30, 999]; // system msg
        let ids_step2: Vec<u32> = vec![10, 20, 30, 999, 40, 50, 60, 999]; // + user msg

        let old_count = ids_step1.len();
        // No consolidation → no suffix removal needed.
        // Delta = ids_step2[old_count..] = [40, 50, 60, 999]
        let mut result = ids_step1.clone();
        result.extend_from_slice(&ids_step2[old_count..]);
        assert_eq!(
            result, ids_step2,
            "No consolidation should produce correct tokens"
        );
    }
}

// ---------------------------------------------------------------------------
// Nested query execution
// ---------------------------------------------------------------------------

/// Returns true if the query contains any nested `Generate` nodes.
pub(crate) fn has_nested_generates(query: &SpnlQuery) -> bool {
    match query {
        SpnlQuery::Generate(_) => true,
        SpnlQuery::Seq(v) | SpnlQuery::Par(v) | SpnlQuery::Cross(v) | SpnlQuery::Plus(v) => {
            v.iter().any(has_nested_generates)
        }
        SpnlQuery::Monad(inner) => has_nested_generates(inner),
        _ => false,
    }
}

/// Collect all `Generate` nodes from a query tree in DFS order.
pub(crate) fn collect_generates<'a>(query: &'a SpnlQuery, out: &mut Vec<&'a Generate>) {
    match query {
        SpnlQuery::Generate(g) => out.push(g),
        SpnlQuery::Seq(v) | SpnlQuery::Par(v) | SpnlQuery::Cross(v) | SpnlQuery::Plus(v) => {
            for child in v {
                collect_generates(child, out);
            }
        }
        SpnlQuery::Monad(inner) => collect_generates(inner, out),
        _ => {}
    }
}

/// Strip `Generate` nodes from a `SpnlQuery` tree, replacing each with an
/// empty `Seq`. Used to extract the non-generate message content of an outer
/// generate's input for tokenization.
pub(crate) fn strip_generates(query: &SpnlQuery) -> NonGenerateInput {
    match query {
        SpnlQuery::Generate(_) => NonGenerateInput::Seq(vec![]),
        SpnlQuery::Message(msg) => NonGenerateInput::Message(msg.clone()),
        SpnlQuery::Seq(v) => NonGenerateInput::Seq(v.iter().map(strip_generates).collect()),
        SpnlQuery::Par(v) => NonGenerateInput::Par(v.iter().map(strip_generates).collect()),
        SpnlQuery::Plus(v) => NonGenerateInput::Plus(v.iter().map(strip_generates).collect()),
        SpnlQuery::Cross(v) => NonGenerateInput::Cross(v.iter().map(strip_generates).collect()),
        SpnlQuery::Monad(inner) => strip_generates(inner),
        _ => NonGenerateInput::Seq(vec![]),
    }
}

/// Returns true if a `NonGenerateInput` tree has any leaf `Message` nodes.
pub(crate) fn non_generate_input_has_messages(input: &NonGenerateInput) -> bool {
    match input {
        NonGenerateInput::Message(_) => true,
        NonGenerateInput::Seq(v)
        | NonGenerateInput::Par(v)
        | NonGenerateInput::Plus(v)
        | NonGenerateInput::Cross(v) => v.iter().any(non_generate_input_has_messages),
    }
}

/// Convert a `SpnlQuery::Generate` (whose `input: Box<SpnlQuery>` may contain
/// nested `Generate` nodes) into a `SingleGenerate` by stripping inner
/// generates from the input tree.  The resulting `SingleGenerate` covers only
/// the non-generate message content of the outer input.
pub(crate) fn outer_generate_to_single(g: &Generate) -> SingleGenerate {
    SingleGenerate {
        metadata: g.metadata.clone(),
        input: strip_generates(&g.input),
    }
}

/// Synchronous nested query execution — used by `LLM::execute_query`.
///
/// Parses `spnl_json` as a full `SpnlQuery` and executes it, supporting nested
/// generates.  The `generate` closure wraps the caller's synchronous generate
/// path (e.g. `LLM::generate_impl`); it receives `(prompts, sampling_params,
/// seal, volatile)` and returns `Vec<RequestOutput>`.
///
/// Returns the outputs of the final (outermost) generate step.
/// Wrap a timed `generate` call into a single-step `QueryOutput`.
fn timed_generate(
    label: &str,
    prompts: &[crate::llm::Prompt],
    sp: Option<scratchy_core_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<scratchy_core_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    let t0 = std::time::Instant::now();
    let outputs = generate(prompts, sp, seal, volatile)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let steps = outputs
        .into_iter()
        .enumerate()
        .map(|(i, output)| {
            let lbl = if i == 0 {
                label.to_string()
            } else {
                format!("{label}[{i}]")
            };
            crate::llm::GenerateStep {
                label: lbl,
                output,
                elapsed_ms,
            }
        })
        .collect();
    Ok(crate::llm::QueryOutput { steps })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_spnl_query_sync(
    spnl_json: &str,
    params: Option<scratchy_core_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    #[cfg(feature = "rag")] aug_options: &crate::augment::AugmentOptions,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    mut generate: impl FnMut(
        &[crate::llm::Prompt],
        Option<scratchy_core_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    use spnl_core::optimizer::llo::llir::{Bulk, Repeat, SingleGenerateQuery};

    // Try full SpnlQuery first (supports nested generates).
    if let Ok(query) = serde_json::from_str::<SpnlQuery>(spnl_json) {
        #[cfg(feature = "rag")]
        let query = {
            // Use existing tokio runtime if available, otherwise create a temporary one.
            // The runtime must be multi-threaded because augment uses spawn_blocking.
            let rt = tokio::runtime::Handle::try_current().unwrap_or_else(|_| {
                // Leak a runtime so it is never dropped inside a blocking context.
                // This only happens when LLM::execute_query is called outside a
                // tokio context (e.g. tests, CLI). The leak is bounded: at most
                // one runtime per process.
                let rt = Box::leak(Box::new(
                    tokio::runtime::Runtime::new().expect("failed to create tokio runtime"),
                ));
                rt.handle().clone()
            });
            rt.block_on(crate::augment::index(&query, aug_options))
                .map_err(|e| anyhow::anyhow!("RAG indexing failed: {e}"))?;
            rt.block_on(optimize_augments(&query, aug_options))?
        };
        return dispatch_spnl_query_sync(
            &query,
            params,
            seal,
            volatile,
            tokenizer,
            template,
            cfg,
            block_size,
            &mut generate,
        );
    }

    // Fallback: parse as SingleGenerateQuery for backward compatibility.
    let query: SingleGenerateQuery =
        serde_json::from_str(spnl_json).map_err(|e| anyhow::anyhow!("invalid SPNL query: {e}"))?;

    match query {
        SingleGenerateQuery::SingleGenerate(spec) => {
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, &mut generate)
        }
        SingleGenerateQuery::Bulk(Bulk::Repeat(Repeat { n, generate: spec })) => {
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            sp.n = n as u32;
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, &mut generate)
        }
        SingleGenerateQuery::Bulk(Bulk::Map(map)) => {
            let mut prompts = Vec::with_capacity(map.inputs.len());
            for input_text in &map.inputs {
                let span_tok = tokenize_map_input(input_text, tokenizer, template, cfg)
                    .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
                prompts.push(span_tok_to_prompt(span_tok));
            }
            let mut sp = merge_spnl_params_sync(&map.metadata, params);
            if let Some(first) = prompts.first() {
                let len = match first {
                    crate::llm::Prompt::TokenIds(ids) => ids.len(),
                    crate::llm::Prompt::TokenIdsWithAnnotations(ids, _) => ids.len(),
                    crate::llm::Prompt::Text(_) => 0,
                };
                resolve_max_tokens_sync(&mut sp, len, block_size);
            }
            timed_generate("outer", &prompts, Some(sp), seal, volatile, &mut generate)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_spnl_query_sync(
    query: &SpnlQuery,
    params: Option<scratchy_core_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<scratchy_core_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    match query {
        SpnlQuery::Generate(g) if has_nested_generates(&g.input) => execute_nested_generate_sync(
            g, params, seal, volatile, tokenizer, template, cfg, block_size, generate,
        ),
        SpnlQuery::Generate(g) => {
            let spec = outer_generate_to_single(g);
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, generate)
        }
        SpnlQuery::Seq(children) => {
            let mut last = crate::llm::QueryOutput { steps: Vec::new() };
            for child in children {
                last = dispatch_spnl_query_sync(
                    child, None, seal, volatile, tokenizer, template, cfg, block_size, generate,
                )?;
            }
            Ok(last)
        }
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Repeat(r)) => {
            let spec = outer_generate_to_single(&r.generate);
            let span_tok = tokenize_span_query(&spec, tokenizer, template, cfg)
                .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
            let mut sp = merge_spnl_params_sync(&spec.metadata, params);
            sp.n = r.n as u32;
            resolve_max_tokens_sync(&mut sp, span_tok.tokens.len(), block_size);
            let prompt = span_tok_to_prompt(span_tok);
            timed_generate("outer", &[prompt], Some(sp), seal, volatile, generate)
        }
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Map(map)) => {
            let mut prompts = Vec::with_capacity(map.inputs.len());
            for input_text in &map.inputs {
                let span_tok = tokenize_map_input(input_text, tokenizer, template, cfg)
                    .map_err(|e| anyhow::anyhow!("span tokenization failed: {e}"))?;
                prompts.push(span_tok_to_prompt(span_tok));
            }
            let mut sp = merge_spnl_params_sync(&map.metadata, params);
            if let Some(first) = prompts.first() {
                let len = match first {
                    crate::llm::Prompt::TokenIds(ids) => ids.len(),
                    crate::llm::Prompt::TokenIdsWithAnnotations(ids, _) => ids.len(),
                    crate::llm::Prompt::Text(_) => 0,
                };
                resolve_max_tokens_sync(&mut sp, len, block_size);
            }
            timed_generate("outer", &prompts, Some(sp), seal, volatile, generate)
        }
        other => Err(anyhow::anyhow!(
            "unsupported top-level query variant: {}",
            query_variant_name(other)
        )),
    }
}

/// Sync nested generate: execute inner generates, build outer prompt, execute outer.
/// Uses raw token IDs from `RequestOutput` directly — no text round-tripping needed.
#[allow(clippy::too_many_arguments)]
fn execute_nested_generate_sync(
    outer_g: &Generate,
    params: Option<scratchy_core_common::SamplingParams>,
    seal: bool,
    volatile: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
    generate: &mut impl FnMut(
        &[crate::llm::Prompt],
        Option<scratchy_core_common::SamplingParams>,
        bool,
        bool,
    ) -> anyhow::Result<Vec<crate::llm::RequestOutput>>,
) -> anyhow::Result<crate::llm::QueryOutput> {
    let mut inner_gens: Vec<&Generate> = Vec::new();
    collect_generates(&outer_g.input, &mut inner_gens);

    let mut outer_tokens: Vec<u32> = Vec::new();
    let mut outer_annotations: BTreeMap<usize, BlockKind> = BTreeMap::new();
    let mut steps: Vec<crate::llm::GenerateStep> = Vec::new();

    // Execute each inner generate with seal=true, volatile=true.
    for (i, inner_g) in inner_gens.iter().enumerate() {
        let inner_input = strip_generates(&inner_g.input);
        let inner_spec = SingleGenerate {
            metadata: inner_g.metadata.clone(),
            input: inner_input,
        };
        let inner_tok = tokenize_span_query(&inner_spec, tokenizer, template, cfg)
            .map_err(|e| anyhow::anyhow!("inner span tokenization failed: {e}"))?;
        let inner_prompt_tokens = inner_tok.tokens.clone();

        let mut sp = merge_spnl_params_sync(&inner_g.metadata, None);
        resolve_max_tokens_sync(&mut sp, inner_tok.tokens.len(), block_size);
        let prompt = span_tok_to_prompt(inner_tok);

        let t0 = std::time::Instant::now();
        let results = generate(&[prompt], Some(sp), true, true)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Build inner block: prompt_token_ids + output_token_ids (includes EOS + pads).
        let block_idx = outer_tokens.len() / block_size;
        outer_annotations.insert(
            block_idx,
            BlockKind::Relocatable {
                first_token: (block_idx * block_size) as u32,
            },
        );
        outer_tokens.extend_from_slice(&inner_prompt_tokens);
        if let Some(result) = results.first() {
            if let Some(output) = result.outputs.first() {
                outer_tokens.extend_from_slice(&output.token_ids);
            }
            steps.push(crate::llm::GenerateStep {
                label: format!("inner[{i}]"),
                output: result.clone(),
                elapsed_ms,
            });
        }
    }

    // Tokenize non-generate messages from outer input as Prefixed.
    let outer_spec = outer_generate_to_single(outer_g);
    if non_generate_input_has_messages(&outer_spec.input) {
        let msg_start_block = outer_tokens.len() / block_size;
        outer_annotations.insert(
            msg_start_block,
            BlockKind::Prefixed {
                first_token: (msg_start_block * block_size) as u32,
            },
        );
        let msg_tok = tokenize_span_query(&outer_spec, tokenizer, template, cfg)
            .map_err(|e| anyhow::anyhow!("outer span tokenization failed: {e}"))?;
        if let Some(ann) = msg_tok.annotations {
            for (k, v) in ann {
                outer_annotations.insert(msg_start_block + k, v);
            }
        }
        outer_tokens.extend_from_slice(&msg_tok.tokens);
    }

    let mut sp = merge_spnl_params_sync(&outer_g.metadata, params);
    resolve_max_tokens_sync(&mut sp, outer_tokens.len(), block_size);
    let outer_prompt = crate::llm::Prompt::TokenIdsWithAnnotations(outer_tokens, outer_annotations);

    let t0 = std::time::Instant::now();
    let results = generate(&[outer_prompt], Some(sp), seal, volatile)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if let Some(result) = results.into_iter().next() {
        steps.push(crate::llm::GenerateStep {
            label: "outer".to_string(),
            output: result,
            elapsed_ms,
        });
    }
    Ok(crate::llm::QueryOutput { steps })
}

/// Merge SPNL metadata with optional caller params (caller takes precedence).
///
/// When `caller` is `None` the metadata values are used directly, bypassing
/// `SamplingParams::default()` so that defaults like `max_tokens: Some(16)`
/// don't silently override metadata-specified values.
fn merge_spnl_params_sync(
    metadata: &spnl_core::ir::GenerateMetadata,
    caller: Option<scratchy_core_common::SamplingParams>,
) -> scratchy_core_common::SamplingParams {
    match caller {
        Some(mut sp) => {
            // Caller params take precedence; fill in only what caller left unset.
            if sp.max_tokens.is_none() {
                sp.max_tokens = metadata.max_tokens.filter(|&t| t > 0).map(|t| t as u32);
            }
            if sp.temperature == 0.0
                && let Some(t) = metadata.temperature
            {
                sp.temperature = t as f64;
            }
            sp
        }
        None => {
            // No caller — build from metadata, then apply defaults for the rest.
            scratchy_core_common::SamplingParams {
                max_tokens: metadata.max_tokens.filter(|&t| t > 0).map(|t| t as u32),
                temperature: metadata.temperature.map(|t| t as f64).unwrap_or(0.0),
                ..scratchy_core_common::SamplingParams::default()
            }
        }
    }
}

/// Resolve max_tokens against model context length (no-op placeholder; mirrors LLM::resolve_max_tokens).
fn resolve_max_tokens_sync(
    sp: &mut scratchy_core_common::SamplingParams,
    _prompt_len: usize,
    _block_size: usize,
) {
    if sp.max_tokens.is_none() {
        sp.max_tokens = Some(2048);
    }
}

/// Convert `SpanTokenized` to a `Prompt`.
fn span_tok_to_prompt(span_tok: SpanTokenized) -> crate::llm::Prompt {
    match span_tok.annotations {
        Some(ann) if !ann.is_empty() => {
            crate::llm::Prompt::TokenIdsWithAnnotations(span_tok.tokens, ann)
        }
        _ => crate::llm::Prompt::TokenIds(span_tok.tokens),
    }
}
