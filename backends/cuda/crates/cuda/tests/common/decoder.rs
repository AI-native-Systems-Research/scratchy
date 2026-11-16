// SPDX-License-Identifier: Apache-2.0
//! A decoder's end (FP8 S8 F2): qwen3.5-0.8b's full-attention layer ([`super::attn_layer`]), then
//! the final norm and a bf16 `lm_head` onto a [`VOCAB`]-wide vocabulary, and Gemma's final
//! soft-cap when asked. `tape_player` plans it with every row read and with the sampled rows read,
//! and compares row `q_len − 1` of the first's logits with row 0 of the second's.
//!
//! ```text
//!  0..16 the attention layer (`attn_layer_input`), 16 its residual sum `y`
//! 17 f  = RmsNorm(y, g_final)
//! 18 lg = f · W_lm   [M, VOCAB]
//! 19 c  = TanhSoftCap(lg)       (with the soft-cap only)
//! ```
//!
//! Sources: the layer's 0..13, then 14 `g_final` `[1, H]` (the zero-centred gain) and 15 `W_lm`,
//! declared `[H, VOCAB]` (its bytes on the device `[VOCAB, H]`).

use scratchy_subtile::fixtures::rebind;
use scratchy_subtile::handoff::LoweredDecode;
use scratchy_subtile::lower::{GemmWeight, InputRef, OpDesc};
use scratchy_subtile::subtile_ir::{GainConvention, SourceShape, SubOp};

use super::attn_layer::{EPS, H, attn_layer, attn_layer_input};
use super::chains::with_hidden;

/// The vocabulary: a multiple of the bf16 GEMM's 256 columns.
pub const VOCAB: u32 = 2048;
/// Gemma 4's `final_logit_softcapping`.
pub const CAP: f32 = 30.0;

/// The decoder's end at `m` rows (the module's table), its soft-cap when `soft_cap`, bound as the
/// attention layer is: `x` the embedded hidden, the rotary pairs and the prefix caches the layer's.
pub fn decoder(m: u32, soft_cap: bool) -> LoweredDecode {
    use InputRef::{Ext, Op};
    let mut input = attn_layer_input(m);
    let y = input.result;
    let (g_final, w_lm) = (input.sources.len(), input.sources.len() + 1);
    input.sources.push(SourceShape { rows: 1, cols: H });
    input.sources.push(SourceShape {
        rows: H,
        cols: VOCAB,
    });
    let op = |op, inputs| OpDesc { op, m, inputs };
    let norm = SubOp::RmsNorm {
        eps: EPS,
        gain: GainConvention::OnePlusScale,
    };
    let matmul = SubOp::MatmulTile {
        n: VOCAB,
        weight: GemmWeight::Dense,
    };
    let f = input.ops.len();
    input.ops.push(op(norm, vec![Op(y), Ext(g_final)]));
    input.ops.push(op(matmul, vec![Op(f), Ext(w_lm)]));
    input.result = f + 1;
    if soft_cap {
        input
            .ops
            .push(op(SubOp::TanhSoftCap { cap: CAP }, vec![Op(f + 1)]));
        input.result = f + 2;
    }
    let mut l = with_hidden(input);
    let layer = attn_layer(m);
    let bound = layer.bindings().len();
    rebind(&mut l, |b| b[..bound].clone_from_slice(layer.bindings()));
    l
}
