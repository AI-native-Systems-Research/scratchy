// SPDX-License-Identifier: Apache-2.0
//! PP-aware model forward output type.

use crate::alloc::OwnedTensor;

/// Output of a model forward pass. For single-GPU or the last PP stage,
/// this is `Logits`. For non-last PP stages, it's `Intermediate` containing
/// the hidden states and residual to pass to the next stage.
pub enum ForwardOutput {
    /// Final logits `[num_reqs, vocab_size]` — only from the last PP stage.
    /// Wrapped in `OwnedTensor` so GPU memory is freed on drop (RAII).
    Logits(OwnedTensor),
    /// Intermediate hidden states + residual to send to next PP stage.
    /// Both are `[num_tokens, hidden_size]` in the model's compute dtype.
    Intermediate {
        hidden_states: OwnedTensor,
        residual: OwnedTensor,
    },
}
