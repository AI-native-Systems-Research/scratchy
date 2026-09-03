// SPDX-License-Identifier: Apache-2.0
//
// Golden reference code — straightforward per-index loops win on
// readability. Suppress the usual clippy "iterate by element"
// pedantry so the math stays legible.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
//! CPU golden model implementations for each scratchy op.
//!
//! Pure-Rust reference implementations of the same operations the
//! GPU kernels perform. Used for correctness testing and future
//! per-layer golden-diff harnesses: run an op on GPU, run the same
//! op here on CPU, compare outputs.
//!
//! All functions operate on flat `&[f32]` slices (convert bf16→f32
//! before calling). Shape conventions match the scratchy globals
//! layout.

/// RMS normalization: output = (x / rms(x)) * weight
///
/// - `input`: [hidden_dim] — one row of activations
/// - `weight`: [hidden_dim] — per-element scale
/// - `output`: [hidden_dim] — written by this function
/// - `eps`: RMS norm epsilon (typically 1e-5)
pub fn rmsnorm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());

    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// Matrix multiply: output = input @ weight^T
///
/// - `input`: [m, k] row-major
/// - `weight`: [n, k] row-major (transposed in multiply)
/// - `output`: [m, n] row-major
///
/// Serial — this is reference code for correctness checks, not a
/// performance path. Parallelise only if a golden-harness regime
/// starts showing measurable wall-clock pain.
pub fn gemm(input: &[f32], weight: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(input.len(), m * k);
    assert_eq!(weight.len(), n * k);
    assert_eq!(output.len(), m * n);

    for i in 0..m {
        let in_row = &input[i * k..(i + 1) * k];
        for j in 0..n {
            let w_row = &weight[j * k..(j + 1) * k];
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += in_row[l] * w_row[l];
            }
            output[i * n + j] = sum;
        }
    }
}

/// Matrix multiply with residual add: output = (input @ weight^T) + residual
///
/// - `input`: [m, k]
/// - `weight`: [n, k]
/// - `residual`: [m, n]
/// - `output`: [m, n] — written as matmul result + residual
pub fn gemm_add(
    input: &[f32],
    weight: &[f32],
    residual: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    gemm(input, weight, output, m, k, n);
    for i in 0..m * n {
        output[i] += residual[i];
    }
}

/// SiLU activation: output = x * sigmoid(x)
pub fn silu(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    for i in 0..input.len() {
        let x = input[i];
        output[i] = x / (1.0 + (-x).exp());
    }
}

/// Elementwise multiply: output = a * b
pub fn mul(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

/// Token-id → embedding table lookup.
///
/// - `input_ids`: `[num_tokens]` u32 — vocab indices to gather.
/// - `embed_weight`: `[vocab_size, hidden_size]` row-major.
/// - `output`: `[num_tokens, hidden_size]` row-major; row `i` is
///   `embed_weight[input_ids[i]]`.
///
/// Mirrors `Instruction::Embed` (CUDA) and the Metal `KernelId::Embed`
/// dispatch.
pub fn embed(input_ids: &[u32], embed_weight: &[f32], output: &mut [f32], hidden_size: usize) {
    assert_eq!(output.len(), input_ids.len() * hidden_size);
    assert_eq!(embed_weight.len() % hidden_size, 0);
    let vocab = embed_weight.len() / hidden_size;
    for (i, &tok) in input_ids.iter().enumerate() {
        let row = tok as usize;
        assert!(row < vocab, "token id {row} out of range vocab={vocab}");
        let src = &embed_weight[row * hidden_size..(row + 1) * hidden_size];
        output[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(src);
    }
}

/// Elementwise add: `output = a + b`.
///
/// Mirrors `Instruction::Add(delta_slot, residual_slot)` — the CUDA
/// path adds in place into the residual buffer; this golden writes the
/// sum to a fresh `output` so call sites can supply a copy of the
/// residual when an in-place model isn't convenient.
pub fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] + b[i];
    }
}

/// Elementwise scalar multiply: `output = input * scale`.
///
/// Mirrors `Instruction::ScalarMul(in_slot, out_slot, scale)`. The
/// CUDA path is in-place (`scale_inplace`); this golden writes a
/// separate output for cleanliness.
pub fn scalar_mul(input: &[f32], output: &mut [f32], scale: f32) {
    assert_eq!(input.len(), output.len());
    for i in 0..input.len() {
        output[i] = input[i] * scale;
    }
}

/// Fused add + RMSNorm with the same semantics as the Metal
/// `fused_add_rmsnorm_<T_act>_s_<T_scale>_specialized` kernel and
/// CUDA's `fused_add_rms_norm_inplace`:
///
/// - Pass 1: `residual += delta` (in place).
/// - Pass 2: `delta = rmsnorm(residual_after_add, weight, eps)`.
///
/// Both buffers are `[M, hidden_size]` row-major; `weight` is
/// `[hidden_size]`. After the call:
/// - `residual[i]` holds the post-add value (consumed downstream as
///   the next layer's residual);
/// - `delta[i]` holds the normalized result (consumed downstream as
///   the next sublayer's input).
///
/// Mirrors `Instruction::FusedAddRmsNorm` (CUDA) and Metal
/// `KernelId::FusedAddRmsNorm`. See `shaders/fused_add_rmsnorm.metal`
/// for the device-side reference; this CPU path is bit-identical at
/// f32 (the kernel is f16, so end-to-end goldens compare with a
/// tolerance — see the per-bucket diff harness in 5.G.2).
pub fn fused_add_rmsnorm(
    residual: &mut [f32],
    delta: &mut [f32],
    weight: &[f32],
    eps: f32,
    hidden_size: usize,
) {
    assert_eq!(residual.len(), delta.len());
    assert_eq!(weight.len(), hidden_size);
    assert_eq!(residual.len() % hidden_size, 0);

    let m = residual.len() / hidden_size;
    for row in 0..m {
        let base = row * hidden_size;
        // Pass 1: residual += delta in place; accumulate sum-of-squares
        // off the post-add residual.
        let mut sum_sq = 0.0_f32;
        for i in 0..hidden_size {
            let s = residual[base + i] + delta[base + i];
            residual[base + i] = s;
            sum_sq += s * s;
        }
        let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        // Pass 2: write rmsnorm(residual_after_add, weight) into delta.
        for i in 0..hidden_size {
            delta[base + i] = residual[base + i] * inv_rms * weight[i];
        }
    }
}

/// Fused gate/up SwiGLU MLP: `output = silu(gate) * up`.
///
/// Both inputs are `[M, intermediate_size]`; the output has the same
/// shape. Mirrors `Instruction::FusedGateUpSiluMul` (CUDA) and Metal
/// `KernelId::FusedGateUpSiluMul`.
///
/// The Metal lowering folds the two gate/up Gemms (and the SiLU + Mul)
/// into one kernel call; this CPU golden takes the post-Gemm activations
/// as inputs and only models the SiLU + elementwise multiply.
pub fn fused_gate_up_silu_mul(gate: &[f32], up: &[f32], output: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), output.len());
    for i in 0..gate.len() {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        output[i] = silu_g * up[i];
    }
}

/// Rotary position embedding (RoPE).
///
/// Applies rotation to query/key vectors using cos/sin tables.
/// - `qk`: [seq_len, dim] — query or key vectors (modified in-place via output)
/// - `cos_table`: [max_pos, head_dim/2] or [max_pos, head_dim]
/// - `sin_table`: same shape as cos_table
/// - `positions`: [seq_len] — position indices
/// - `head_dim`: dimension of each attention head
/// - `output`: [seq_len, dim] — rotated vectors
pub fn rope(
    qk: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    positions: &[i32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    let half_dim = head_dim / 2;
    let dim = num_heads * head_dim;
    assert_eq!(qk.len(), seq_len * dim);
    assert_eq!(output.len(), seq_len * dim);

    output.copy_from_slice(qk);

    for s in 0..seq_len {
        let pos = positions[s] as usize;
        for h in 0..num_heads {
            let base = s * dim + h * head_dim;
            for d in 0..half_dim {
                let cos_val = cos_table[pos * head_dim + d];
                let sin_val = sin_table[pos * head_dim + d];
                let x0 = qk[base + d];
                let x1 = qk[base + half_dim + d];
                output[base + d] = x0 * cos_val - x1 * sin_val;
                output[base + half_dim + d] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// RoPE + paged KV-cache append.
///
/// Mirrors `Instruction::RopeAppend` (CUDA) and Metal
/// `KernelId::RopeAppend`. Two effects per token:
/// 1. Rotate Q (and K) using `cos_table[positions[t]]` /
///    `sin_table[positions[t]]`. NeoX-style pairing: element `d`
///    pairs with `d + half_dim`.
/// 2. Write the rotated K and unrotated V into the paged KV cache
///    at `slot_mapping[t]`. Slot id is global: `block_id =
///    slot / block_size`, `block_offset = slot % block_size`.
///
/// Layout conventions (must match the Metal shader):
/// - `q_in`, `q_out`: `[num_tokens, num_q_heads * head_dim]` row-major.
/// - `k_in`, `v_in`: `[num_tokens, num_kv_heads * head_dim]` row-major.
/// - `cos_table`, `sin_table`: `[max_pos, head_dim]` row-major; only
///   the first `head_dim/2` columns of each row are read (matches the
///   existing `cpu_golden::rope` convention).
/// - `kv_cache_k`, `kv_cache_v`: `[num_blocks, num_kv_heads,
///   block_size, head_dim]` row-major. The shader's KV cache is the
///   same shape.
/// - `slot_mapping`: `[num_tokens]` u32 — global cache slot id per
///   token; the caller picks free slots before the call.
///
/// V is written un-rotated; only Q and K are rotated. This matches
/// `Instruction::RopeAppend` semantics in `instr.rs` and the
/// (currently unwritten) `rope_append_f16_specialized` shader.
pub fn rope_append(
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    positions: &[u32],
    slot_mapping: &[u32],
    cos_table: &[f32],
    sin_table: &[f32],
    q_out: &mut [f32],
    kv_cache_k: &mut [f32],
    kv_cache_v: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) {
    let num_tokens = positions.len();
    assert_eq!(slot_mapping.len(), num_tokens);
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(q_in.len(), num_tokens * q_dim);
    assert_eq!(q_out.len(), num_tokens * q_dim);
    assert_eq!(k_in.len(), num_tokens * kv_dim);
    assert_eq!(v_in.len(), num_tokens * kv_dim);
    assert_eq!(cos_table.len(), sin_table.len());
    assert_eq!(cos_table.len() % head_dim, 0);
    // Stride math centralized — see [`crate::paged_kv_layout`].
    let layout = crate::paged_kv_layout::PagedKvLayout::from_buffer_elems(
        kv_cache_k.len(),
        num_kv_heads as u32,
        block_size as u32,
        head_dim as u32,
    );
    assert_eq!(kv_cache_v.len(), kv_cache_k.len());
    let half_dim = head_dim / 2;

    for t in 0..num_tokens {
        let pos = positions[t] as usize;
        let cos_row = &cos_table[pos * head_dim..pos * head_dim + head_dim];
        let sin_row = &sin_table[pos * head_dim..pos * head_dim + head_dim];

        // Rotate Q in place into q_out.
        for h in 0..num_q_heads {
            let base = t * q_dim + h * head_dim;
            for d in 0..half_dim {
                let x0 = q_in[base + d];
                let x1 = q_in[base + half_dim + d];
                let c = cos_row[d];
                let s = sin_row[d];
                q_out[base + d] = x0 * c - x1 * s;
                q_out[base + half_dim + d] = x1 * c + x0 * s;
            }
        }

        // Rotate K and write rotated K + unrotated V to the cache slot.
        let slot = slot_mapping[t] as usize;
        for h in 0..num_kv_heads {
            let kv_base = t * kv_dim + h * head_dim;
            let cache_base = layout.elem_offset_for_global_slot(slot as u32, h as u32);
            // K: rotate.
            for d in 0..half_dim {
                let x0 = k_in[kv_base + d];
                let x1 = k_in[kv_base + half_dim + d];
                let c = cos_row[d];
                let s = sin_row[d];
                kv_cache_k[cache_base + d] = x0 * c - x1 * s;
                kv_cache_k[cache_base + half_dim + d] = x1 * c + x0 * s;
            }
            // V: copy un-rotated.
            kv_cache_v[cache_base..cache_base + head_dim]
                .copy_from_slice(&v_in[kv_base..kv_base + head_dim]);
        }
    }
}

/// Paged-cache decode attention.
///
/// Mirrors `Instruction::AttentionViaCache` (CUDA) and Metal
/// `attention_via_cache_v2_f16_specialized` (`shaders/attention.metal`).
/// One Q token per sequence (`bucket_m == batch` for decode); reads K
/// and V from the paged cache via the per-sequence `block_table` and
/// `seq_used_k` length.
///
/// Layout conventions (must match the Metal shader):
/// - `q`: `[batch, num_q_heads * head_dim]` row-major.
/// - `kv_cache_k`, `kv_cache_v`: `[num_blocks, num_kv_heads,
///   block_size, head_dim]` row-major.
/// - `block_table`: `[batch, max_blocks_per_seq]` row-major u32 —
///   logical-to-physical block map per sequence.
/// - `seq_used_k`: `[batch]` u32 — kv-axis used length per sequence.
/// - `output`: `[batch, num_q_heads * head_dim]` row-major.
///
/// Matches the metal shader's `inv_sum = 1 / (sum_exp + 1e-6)`
/// epsilon — the CPU ref drifts from `cpu_golden::attention_decode`
/// (which uses exact 1/sum_exp) only on this term, ensuring the
/// per-bucket diff harness in 5.G.3 sees no spurious mismatch from
/// numerical-guard differences.
pub fn attention_via_cache(
    q: &[f32],
    kv_cache_k: &[f32],
    kv_cache_v: &[f32],
    block_table: &[u32],
    seq_used_k: &[u32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    attn_scale: f32,
) {
    let batch = seq_used_k.len();
    let q_dim = num_q_heads * head_dim;
    assert_eq!(q.len(), batch * q_dim);
    assert_eq!(output.len(), batch * q_dim);
    assert_eq!(block_table.len(), batch * max_blocks_per_seq);
    // Stride math centralized — see [`crate::paged_kv_layout`].
    let layout = crate::paged_kv_layout::PagedKvLayout::from_buffer_elems(
        kv_cache_k.len(),
        num_kv_heads as u32,
        block_size as u32,
        head_dim as u32,
    );
    assert_eq!(kv_cache_v.len(), kv_cache_k.len());
    let group_ratio = num_q_heads / num_kv_heads;

    for seq in 0..batch {
        let kv_len = seq_used_k[seq] as usize;
        let row_blocks = &block_table[seq * max_blocks_per_seq..(seq + 1) * max_blocks_per_seq];

        for h in 0..num_q_heads {
            let kv_h = h / group_ratio;
            let q_off = seq * q_dim + h * head_dim;

            let mut scores = vec![0.0_f32; kv_len];
            for t in 0..kv_len {
                let logical_block = t / block_size;
                let block_offset = t % block_size;
                let physical_block = row_blocks[logical_block] as usize;
                let k_base =
                    layout.elem_offset(physical_block as u32, kv_h as u32, block_offset as u32);
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * kv_cache_k[k_base + d];
                }
                scores[t] = dot * attn_scale;
            }

            let max_logit = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_logit).exp();
                sum_exp += *s;
            }
            // +1e-6 epsilon matches the metal shader's inv_sum guard.
            let inv_sum = 1.0 / (sum_exp + 1e-6);

            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for t in 0..kv_len {
                    let logical_block = t / block_size;
                    let block_offset = t % block_size;
                    let physical_block = row_blocks[logical_block] as usize;
                    let v_base =
                        layout.elem_offset(physical_block as u32, kv_h as u32, block_offset as u32);
                    acc += scores[t] * inv_sum * kv_cache_v[v_base + d];
                }
                output[q_off + d] = acc;
            }
        }
    }
}

/// Scaled dot-product attention (decode, single query token).
///
/// - `q`: [1, num_heads * head_dim]
/// - `k_cache`: [seq_len, num_kv_heads, head_dim]
/// - `v_cache`: [seq_len, num_kv_heads, head_dim]
/// - `output`: [1, num_heads * head_dim]
/// - GQA: num_heads / num_kv_heads queries share one KV head
pub fn attention_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    output: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let q_offset = h * head_dim;

        // Compute attention scores
        let mut scores = vec![0.0_f32; seq_len];
        for s in 0..seq_len {
            let k_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            scores[s] = dot * attn_scale;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        for s in &mut scores {
            *s /= sum_exp;
        }

        // Weighted sum of values
        for d in 0..head_dim {
            let mut val = 0.0_f32;
            for s in 0..seq_len {
                let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim + d;
                val += scores[s] * v_cache[v_offset];
            }
            output[q_offset + d] = val;
        }
    }
}

/// Scaled dot-product attention (prefill, variable-length sequences).
///
/// - `q`: [total_tokens, num_heads * head_dim]
/// - `k`: [total_tokens, num_kv_heads * head_dim]
/// - `v`: [total_tokens, num_kv_heads * head_dim]
/// - `output`: [total_tokens, num_heads * head_dim]
/// - `seq_starts`: [num_seqs + 1] — cumulative token offsets
/// - Causal masking applied
pub fn attention_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_starts: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_scale: f32,
) {
    let gqa_ratio = num_heads / num_kv_heads;
    let num_seqs = seq_starts.len() - 1;

    for seq in 0..num_seqs {
        let start = seq_starts[seq];
        let end = seq_starts[seq + 1];
        let seq_len = end - start;

        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;

            for qi in 0..seq_len {
                let q_offset = (start + qi) * num_heads * head_dim + h * head_dim;

                // Compute scores (causal: only attend to positions <= qi)
                let attend_len = qi + 1;
                let mut scores = vec![0.0_f32; attend_len];
                for ki in 0..attend_len {
                    let k_offset = (start + ki) * num_kv_heads * head_dim + kv_h * head_dim;
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }
                    scores[ki] = dot * attn_scale;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                for s in &mut scores {
                    *s /= sum_exp;
                }

                // Weighted sum
                for d in 0..head_dim {
                    let mut val = 0.0_f32;
                    for ki in 0..attend_len {
                        let v_offset = (start + ki) * num_kv_heads * head_dim + kv_h * head_dim + d;
                        val += scores[ki] * v[v_offset];
                    }
                    output[q_offset + d] = val;
                }
            }
        }
    }
}

/// Paged-cache prefill attention (variable-length sequences with prior
/// cached prefix).
///
/// Mirrors `attention_prefill_sdpa_v2_paged_{f16,bf16}_specialized`
/// (`shaders/attention.metal`). Same per-Q causal SDPA as
/// `attention_prefill` but K/V come from the paged cache via
/// `block_table` (mirroring `attention_via_cache`'s decode access),
/// and the K-axis covers the FULL `seq_used_k[seq]` rather than only
/// the new tokens.
///
/// Layout conventions (must match the Metal shader):
/// - `q`: `[total_q, num_q_heads, head_dim]` row-major.
/// - `kv_cache_k`, `kv_cache_v`: `[num_blocks, num_kv_heads,
///   block_size, head_dim]` row-major.
/// - `block_table`: `[batch, max_blocks_per_seq]` row-major u32.
/// - `seq_used_k`: `[batch]` u32 — TOTAL cached K per sequence
///   (prefix + just-appended new tokens). Caller is responsible for
///   running `RopeAppend` first so the new K is in cache before this
///   attention call.
/// - `cu_seqlens_q`: `[batch+1]` u32 — cumulative new-token boundaries.
/// - `output`: `[total_q, num_q_heads, head_dim]` row-major.
///
/// The per-Q absolute K position is
/// `q_abs_pos = (seq_used_k[seq] - new_q_for_seq) + q_pos_in_new`,
/// where `new_q_for_seq = cu_seqlens_q[seq+1] - cu_seqlens_q[seq]` and
/// `q_pos_in_new = global_q - cu_seqlens_q[seq]`. K positions
/// `> q_abs_pos` are masked out (causal).
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_paged(
    q: &[f32],
    kv_cache_k: &[f32],
    kv_cache_v: &[f32],
    output: &mut [f32],
    cu_seqlens_q: &[u32],
    seq_used_k: &[u32],
    block_table: &[u32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    attn_scale: f32,
) {
    let batch = seq_used_k.len();
    assert!(cu_seqlens_q.len() > batch);
    assert_eq!(block_table.len(), batch * max_blocks_per_seq);
    // Stride math centralized — see [`crate::paged_kv_layout`].
    let layout = crate::paged_kv_layout::PagedKvLayout::from_buffer_elems(
        kv_cache_k.len(),
        num_kv_heads as u32,
        block_size as u32,
        head_dim as u32,
    );
    assert_eq!(kv_cache_v.len(), kv_cache_k.len());
    let group_ratio = num_q_heads / num_kv_heads;

    for seq in 0..batch {
        let seq_start = cu_seqlens_q[seq] as usize;
        let seq_end = cu_seqlens_q[seq + 1] as usize;
        let new_q_for_seq = seq_end - seq_start;
        if new_q_for_seq == 0 {
            continue;
        }
        let kv_len = seq_used_k[seq] as usize;
        let prefix_len = kv_len - new_q_for_seq;
        let row_blocks = &block_table[seq * max_blocks_per_seq..(seq + 1) * max_blocks_per_seq];

        for q_pos_in_new in 0..new_q_for_seq {
            let global_q = seq_start + q_pos_in_new;
            let q_abs_pos = prefix_len + q_pos_in_new;
            let attend_len = q_abs_pos + 1;

            for h in 0..num_q_heads {
                let kv_h = h / group_ratio;
                let q_off = global_q * num_q_heads * head_dim + h * head_dim;

                let mut scores = vec![0.0_f32; attend_len];
                for t in 0..attend_len {
                    let logical_block = t / block_size;
                    let block_offset = t % block_size;
                    let physical_block = row_blocks[logical_block] as usize;
                    let k_base =
                        layout.elem_offset(physical_block as u32, kv_h as u32, block_offset as u32);
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * kv_cache_k[k_base + d];
                    }
                    scores[t] = dot * attn_scale;
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                for s in &mut scores {
                    *s /= sum_exp;
                }

                for d in 0..head_dim {
                    let mut val = 0.0_f32;
                    for t in 0..attend_len {
                        let logical_block = t / block_size;
                        let block_offset = t % block_size;
                        let physical_block = row_blocks[logical_block] as usize;
                        let v_base = layout.elem_offset(
                            physical_block as u32,
                            kv_h as u32,
                            block_offset as u32,
                        );
                        val += scores[t] * kv_cache_v[v_base + d];
                    }
                    output[q_off + d] = val;
                }
            }
        }
    }
}

/// MLX-affine int4 dequantization — half-precision reference.
///
/// Faithful CPU port of the Metal `affine_dequantize` kernel
/// (`mlx/backend/metal/kernels/quantized.h:2536`) for the bits=4 case.
/// Layout matches the on-disk safetensors for `mlx-community/*-4bit`:
///
/// - `packed`: `[n_bytes]` packed nibbles. Two output elements per byte
///   (low nibble → `out[oindex]`, high nibble → `out[oindex + 1]`).
///   Equivalent to the `[N, K/8]` U32 weight tensor reinterpreted as
///   `[N * K / 2]` bytes (`pack_factor = 32 / bits = 8` u32-packing, or
///   `8 / bits = 2` bytes-packing). `n_bytes = N * K / 2`.
/// - `scales` / `biases`: `[N * K / group_size]` per-group scale + affine
///   offset, each stored as half-precision. MLX terminology: "biases"
///   means the per-group affine offset, NOT a linear-layer bias.
/// - `output`: `[N * K]` half-precision. Caller sizes per `packed.len()
///   * 2 == output.len()`.
///
/// The kernel emits a hardware fp16/bf16 FMA for `scale * d + bias` — one
/// rounding at the end. We mirror by promoting to f32 for the multiply +
/// add (exact at these magnitudes: scale fits in f32 mantissa, `d ∈ [0,
/// 15]` is exact, the sum fits trivially) and rounding to the target
/// dtype once. Doing `(scale * f16(d)) + bias` with the `half` crate's
/// per-op rounding drifts ~4 ULPs vs the kernel — bit-exact parity
/// requires modelling FMA single-rounding here.
pub fn affine_dequantize_b4_f16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    output: &mut [half::f16],
    group_size: usize,
) {
    let pack_factor: usize = 2;
    assert_eq!(output.len(), packed.len() * pack_factor);
    assert_eq!(
        output.len() % group_size,
        0,
        "output length {} not divisible by group_size {group_size}",
        output.len(),
    );
    assert_eq!(scales.len(), output.len() / group_size);
    assert_eq!(biases.len(), output.len() / group_size);

    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * pack_factor;
        let gindex = oindex / group_size;
        let scale = scales[gindex].to_f32();
        let bias = biases[gindex].to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        output[oindex] = half::f16::from_f32(scale * lo + bias);
        output[oindex + 1] = half::f16::from_f32(scale * hi + bias);
    }
}

/// BFloat16 sibling of [`affine_dequantize_b4_f16`].
pub fn affine_dequantize_b4_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    output: &mut [half::bf16],
    group_size: usize,
) {
    let pack_factor: usize = 2;
    assert_eq!(output.len(), packed.len() * pack_factor);
    assert_eq!(
        output.len() % group_size,
        0,
        "output length {} not divisible by group_size {group_size}",
        output.len(),
    );
    assert_eq!(scales.len(), output.len() / group_size);
    assert_eq!(biases.len(), output.len() / group_size);

    for (offset, &byte) in packed.iter().enumerate() {
        let oindex = offset * pack_factor;
        let gindex = oindex / group_size;
        let scale = scales[gindex].to_f32();
        let bias = biases[gindex].to_f32();
        let lo = (byte & 0x0f) as f32;
        let hi = ((byte >> 4) & 0x0f) as f32;
        output[oindex] = half::bf16::from_f32(scale * lo + bias);
        output[oindex + 1] = half::bf16::from_f32(scale * hi + bias);
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) — Qwen3.5 / Qwen3-Next linear-attention references.
//
// Decomposed to mirror the four surviving CUDA kernels (`gdn_conv1d_*`,
// `gdn_gating`, `gdn_recurrent_fwd`, `gdn_rms_norm_gated`) so each can be
// golden-checked independently against these CPU refs. The recurrence math is
// transcribed from the reference Triton kernel
// `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
// (`fused_recurrent_gated_delta_rule_fwd_kernel`); the surrounding conv1d +
// gating + gated-norm from `vllm/model_executor/models/qwen3_next.py`.
// ---------------------------------------------------------------------------

/// Causal depthwise conv1d over the token axis, optionally followed by SiLU.
///
/// * `x`:      `[num_tokens, conv_dim]` row-major (the in_proj_qkv output).
/// * `weight`: `[conv_dim, kernel]` (HF stores `[conv_dim, 1, kernel]`; drop
///   the singleton dim). Causal:
///   `out[t,c] = act(Σ_j w[c,j]·x[t-(kernel-1)+j, c])`, left-padded with zeros
///   for a fresh sequence (the runtime seeds the pad from the per-sequence
///   conv_state instead).
pub fn gdn_causal_conv1d(
    x: &[f32],
    weight: &[f32],
    output: &mut [f32],
    conv_dim: usize,
    kernel: usize,
    num_tokens: usize,
    apply_silu: bool,
) {
    assert_eq!(x.len(), num_tokens * conv_dim);
    assert_eq!(weight.len(), conv_dim * kernel);
    assert_eq!(output.len(), num_tokens * conv_dim);
    for t in 0..num_tokens {
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for j in 0..kernel {
                let ti = t as isize - (kernel as isize - 1) + j as isize;
                if ti >= 0 {
                    acc += weight[c * kernel + j] * x[ti as usize * conv_dim + c];
                }
            }
            output[t * conv_dim + c] = if apply_silu {
                acc / (1.0 + (-acc).exp())
            } else {
                acc
            };
        }
    }
}

/// GDN input-dependent gating.
///
/// `g[t,h]    = -exp(A_log[h]) · softplus(a[t,h] + dt_bias[h])`  (log-decay, ≤0)
/// `beta[t,h] = sigmoid(b[t,h])`
///
/// `a`,`b`: `[num_tokens, num_heads]`; `a_log`,`dt_bias`: `[num_heads]`.
/// softplus uses the numerically-stable threshold (20.0) from the kernel.
pub fn gdn_gating(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    g_out: &mut [f32],
    beta_out: &mut [f32],
    num_heads: usize,
    num_tokens: usize,
) {
    assert_eq!(a.len(), num_tokens * num_heads);
    assert_eq!(b.len(), num_tokens * num_heads);
    assert_eq!(a_log.len(), num_heads);
    assert_eq!(dt_bias.len(), num_heads);
    let softplus = |x: f32| if x <= 20.0 { x.exp().ln_1p() } else { x };
    for t in 0..num_tokens {
        for h in 0..num_heads {
            let idx = t * num_heads + h;
            g_out[idx] = -(a_log[h].exp()) * softplus(a[idx] + dt_bias[h]);
            beta_out[idx] = 1.0 / (1.0 + (-b[idx]).exp());
        }
    }
}

/// Recurrent gated delta-rule scan (single sequence, zero initial state).
///
/// Per value-head `h`, the state `S` is `[head_v, head_k]`; its key/query head
/// is `h / (num_v_heads / num_k_heads)` (grouped value attention, HV ≥ H).
/// Per token, transcribed from the reference kernel:
/// ```text
///   q,k ← L2-normalize over head_k (eps 1e-6);  q ← q · scale
///   S   ← S · exp(g)                         (decay applied first)
///   u   ← beta · (v − S·k);   S ← S + u ⊗ k
///   o   ← S · q
/// ```
/// `q`,`k`: `[T, num_k_heads·head_k]`; `v`,`o`: `[T, num_v_heads·head_v]`;
/// `g`,`beta`: `[T, num_v_heads]`. `scale` = 1/sqrt(head_k).
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    o: &mut [f32],
    num_k_heads: usize,
    num_v_heads: usize,
    head_k: usize,
    head_v: usize,
    num_tokens: usize,
    scale: f32,
) {
    let key_dim = num_k_heads * head_k;
    let value_dim = num_v_heads * head_v;
    assert_eq!(q.len(), num_tokens * key_dim);
    assert_eq!(k.len(), num_tokens * key_dim);
    assert_eq!(v.len(), num_tokens * value_dim);
    assert_eq!(g.len(), num_tokens * num_v_heads);
    assert_eq!(beta.len(), num_tokens * num_v_heads);
    assert_eq!(o.len(), num_tokens * value_dim);
    assert_eq!(
        num_v_heads % num_k_heads,
        0,
        "GVA requires num_v_heads % num_k_heads == 0"
    );
    let groups = num_v_heads / num_k_heads;
    // state[h] flattened head_v × head_k.
    let mut state = vec![0.0f32; num_v_heads * head_v * head_k];
    let l2 = |s: &[f32]| -> f32 { (s.iter().map(|&x| x * x).sum::<f32>() + 1e-6).sqrt() };
    for t in 0..num_tokens {
        for h in 0..num_v_heads {
            let ki = h / groups;
            // L2-normalized q (scaled) and k for this token's key head.
            let qsrc = &q[t * key_dim + ki * head_k..][..head_k];
            let ksrc = &k[t * key_dim + ki * head_k..][..head_k];
            let qinv = scale / l2(qsrc);
            let kinv = 1.0 / l2(ksrc);
            let qn: Vec<f32> = qsrc.iter().map(|&x| x * qinv).collect();
            let kn: Vec<f32> = ksrc.iter().map(|&x| x * kinv).collect();
            let sh = &mut state[h * head_v * head_k..][..head_v * head_k];
            let decay = g[t * num_v_heads + h].exp();
            let gt = beta[t * num_v_heads + h];
            for s in sh.iter_mut() {
                *s *= decay;
            }
            let vsrc = &v[t * value_dim + h * head_v..][..head_v];
            for vd in 0..head_v {
                // u = beta · (v − S·k)
                let mut sk = 0.0f32;
                for kd in 0..head_k {
                    sk += sh[vd * head_k + kd] * kn[kd];
                }
                let u = gt * (vsrc[vd] - sk);
                // S += u ⊗ k
                for kd in 0..head_k {
                    sh[vd * head_k + kd] += u * kn[kd];
                }
                // o = S · q
                let mut ov = 0.0f32;
                for kd in 0..head_k {
                    ov += sh[vd * head_k + kd] * qn[kd];
                }
                o[t * value_dim + h * head_v + vd] = ov;
            }
        }
    }
}

/// Gated RMSNorm (norm_before_gate): `out = rmsnorm_over_d(x) · weight · silu(z)`.
///
/// Normalizes each row of `x` over its last `d` elements, scales by `weight`,
/// then multiplies by `silu(z)` (= z·sigmoid(z)). `x`,`z`,`out`:
/// `[total_rows, d]`; `weight`: `[d]`. NOTE: the gate is **SiLU**, not plain
/// sigmoid — matches Python `RMSNormGated`.
pub fn gdn_rms_norm_gated(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    out: &mut [f32],
    d: usize,
    total_rows: usize,
    eps: f32,
) {
    assert_eq!(x.len(), total_rows * d);
    assert_eq!(z.len(), total_rows * d);
    assert_eq!(weight.len(), d);
    assert_eq!(out.len(), total_rows * d);
    for r in 0..total_rows {
        let row = &x[r * d..][..d];
        let var = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            let zi = z[r * d + i];
            let silu_z = zi / (1.0 + (-zi).exp());
            out[r * d + i] = row[i] * inv * weight[i] * silu_z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full GDN pipeline (conv1d→split→gating→recurrent→gated-norm) pinned to a
    /// numpy oracle implementing the exact vLLM recurrence. Deterministic inputs
    /// `fill(i) = sin(i·0.1)·0.5`; tiny config (nk=2, nv=4, hk=hv=4, K=4, T=6).
    /// The golden checksums + core[0] were produced by the reference script and
    /// validate the math end-to-end on CPU (Mac), independent of any GPU kernel.
    #[test]
    #[allow(clippy::approx_constant)] // -0.693_147_2 is a golden test oracle, not LN_2
    fn test_gdn_pipeline_golden() {
        let (nk, nv, hk, hv, kk, t) = (2usize, 4usize, 4usize, 4usize, 4usize, 6usize);
        let key_dim = nk * hk;
        let value_dim = nv * hv;
        let conv_dim = 2 * key_dim + value_dim;
        let eps = 1e-6f32;
        let scale = (hk as f32).powf(-0.5);
        let fill =
            |n: usize| -> Vec<f32> { (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect() };

        let qkv = fill(t * conv_dim);
        let z = fill(t * value_dim);
        let a = fill(t * nv);
        let b = fill(t * nv);
        let conv_w = fill(conv_dim * kk);
        let a_log = fill(nv);
        let dt_bias = fill(nv);
        let norm_w = fill(hv);

        // 1. causal conv1d + SiLU
        let mut conv = vec![0f32; t * conv_dim];
        gdn_causal_conv1d(&qkv, &conv_w, &mut conv, conv_dim, kk, t, true);
        let conv_abs: f32 = conv.iter().map(|v| v.abs()).sum();
        assert!((conv_abs - 4.5901105).abs() < 1e-3, "conv_abs {conv_abs}");

        // 2. split conv output -> q,k,v
        let mut q = vec![0f32; t * key_dim];
        let mut k = vec![0f32; t * key_dim];
        let mut v = vec![0f32; t * value_dim];
        for ti in 0..t {
            let row = &conv[ti * conv_dim..][..conv_dim];
            q[ti * key_dim..][..key_dim].copy_from_slice(&row[0..key_dim]);
            k[ti * key_dim..][..key_dim].copy_from_slice(&row[key_dim..2 * key_dim]);
            v[ti * value_dim..][..value_dim].copy_from_slice(&row[2 * key_dim..]);
        }

        // 3. gating
        let mut g = vec![0f32; t * nv];
        let mut beta = vec![0f32; t * nv];
        gdn_gating(&a, &b, &a_log, &dt_bias, &mut g, &mut beta, nv, t);
        let g_sum: f32 = g.iter().sum();
        let beta_sum: f32 = beta.iter().sum();
        assert!((g_sum - (-24.235_298)).abs() < 1e-3, "g_sum {g_sum}");
        assert!((beta_sum - 14.095_717).abs() < 1e-3, "beta_sum {beta_sum}");
        assert!((g[0] - (-0.693_147_2)).abs() < 1e-5, "g[0] {}", g[0]);

        // 4. recurrent gated delta-rule scan
        let mut o = vec![0f32; t * value_dim];
        gdn_recurrent(&q, &k, &v, &g, &beta, &mut o, nk, nv, hk, hv, t, scale);
        let o_abs: f32 = o.iter().map(|x| x.abs()).sum();
        assert!((o_abs - 0.6908325).abs() < 1e-3, "o_abs {o_abs}");

        // 5. gated RMSNorm
        let mut core = vec![0f32; t * value_dim];
        gdn_rms_norm_gated(&o, &z, &norm_w, &mut core, hv, t * nv, eps);
        let core_sum: f32 = core.iter().sum();
        let core_abs: f32 = core.iter().map(|x| x.abs()).sum();
        assert!((core_sum - 0.0383676).abs() < 1e-3, "core_sum {core_sum}");
        assert!((core_abs - 0.8466035).abs() < 1e-3, "core_abs {core_abs}");

        // spot-check core[0] (norm_w[0]=sin(0)=0 zeros indices 0,4,8,12)
        let exp0 = [
            0.0f32,
            -0.00117024,
            -0.00612671,
            -0.01440156,
            0.0,
            -0.00728116,
            -0.0075505,
            0.00272334,
            0.0,
            0.01247722,
            0.02883547,
            0.0398975,
            0.0,
            0.01326165,
            0.00690384,
            -0.0015621,
        ];
        for i in 0..value_dim {
            assert!(
                (core[i] - exp0[i]).abs() < 1e-4,
                "core[0][{i}] = {} vs golden {}",
                core[i],
                exp0[i]
            );
        }
    }

    #[test]
    fn test_rmsnorm_identity() {
        // With weight=1 and uniform input, output = input / rms(input)
        let input = vec![1.0_f32; 128];
        let weight = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 128];
        rmsnorm(&input, &weight, &mut output, 1e-5);
        // rms(1,1,...,1) = 1, so output ≈ 1.0
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_scaling() {
        let input = vec![2.0_f32; 64];
        let weight = vec![0.5_f32; 64];
        let mut output = vec![0.0_f32; 64];
        rmsnorm(&input, &weight, &mut output, 1e-5);
        // rms([2,...,2]) = 2, so normalized = 1.0, * 0.5 = 0.5
        for &v in &output {
            assert!((v - 0.5).abs() < 1e-3, "expected ~0.5, got {v}");
        }
    }

    #[test]
    fn test_gemm_simple() {
        // [1,2] x [2,2]^T = [1*1+2*0, 1*0+2*1] = [1, 2]
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let mut output = vec![0.0; 2];
        gemm(&input, &weight, &mut output, 1, 2, 2);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];
        silu(&input, &mut output);
        assert!((output[0] - 0.0).abs() < 1e-5); // silu(0) = 0
        assert!((output[1] - 0.7310586).abs() < 1e-4); // silu(1) ≈ 0.731
        assert!((output[2] - (-0.2689414)).abs() < 1e-4); // silu(-1) ≈ -0.269
    }

    #[test]
    fn test_embed_gather() {
        // 4-token vocab, hidden=3. Gather rows 2, 0, 3 in that order.
        #[rustfmt::skip]
        let weight = vec![
            0.1, 0.2, 0.3,   // row 0
            1.1, 1.2, 1.3,   // row 1
            2.1, 2.2, 2.3,   // row 2
            3.1, 3.2, 3.3,   // row 3
        ];
        let ids = vec![2u32, 0, 3];
        let mut out = vec![0.0_f32; 3 * 3];
        embed(&ids, &weight, &mut out, 3);
        assert_eq!(&out[0..3], &[2.1, 2.2, 2.3]);
        assert_eq!(&out[3..6], &[0.1, 0.2, 0.3]);
        assert_eq!(&out[6..9], &[3.1, 3.2, 3.3]);
    }

    #[test]
    fn test_add_elementwise() {
        let a = vec![1.0, -2.0, 3.5];
        let b = vec![0.5, 2.0, -1.5];
        let mut out = vec![0.0; 3];
        add(&a, &b, &mut out);
        assert_eq!(out, vec![1.5, 0.0, 2.0]);
    }

    #[test]
    fn test_scalar_mul_basic() {
        let input = vec![1.0, -2.0, 0.5];
        let mut out = vec![0.0; 3];
        scalar_mul(&input, &mut out, 4.0);
        assert_eq!(out, vec![4.0, -8.0, 2.0]);
    }

    #[test]
    fn test_fused_add_rmsnorm_two_rows() {
        // M=2 rows, hidden=4. residual starts zero; delta=[2,2,2,2]
        // gives post-add residual = [2,2,2,2] with rms=2 → normed = 1
        // before scaling by weight.
        let mut residual = vec![0.0_f32; 2 * 4];
        let mut delta = vec![2.0_f32; 2 * 4];
        let weight = vec![1.0, 0.5, 0.5, 1.0];
        fused_add_rmsnorm(&mut residual, &mut delta, &weight, 1e-5, 4);
        // Post-add residual should hold the sum.
        for v in &residual {
            assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
        }
        // delta should hold weight (since normed≈1).
        for row in 0..2 {
            for i in 0..4 {
                let want = weight[i];
                let got = delta[row * 4 + i];
                assert!(
                    (got - want).abs() < 1e-3,
                    "row {row} idx {i}: want {want}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_fused_gate_up_silu_mul_signs() {
        // silu(0)=0 → product 0; silu(1)≈0.731 → 0.731 * 2 = 1.462;
        // silu(-1)≈-0.269 → -0.269 * -1 = 0.269.
        let gate = vec![0.0, 1.0, -1.0];
        let up = vec![5.0, 2.0, -1.0];
        let mut out = vec![0.0; 3];
        fused_gate_up_silu_mul(&gate, &up, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 1.4621172).abs() < 1e-4);
        assert!((out[2] - 0.26894143).abs() < 1e-4);
    }

    #[test]
    fn test_rope_append_writes_paged_cache() {
        // 1 token, num_q_heads = num_kv_heads = 1, head_dim = 4, block_size = 2.
        // cos_table = identity-like (cos=1, sin=0) at pos=0 → no rotation.
        // V is copied unrotated; K is rotated (which equals input since sin=0).
        let head_dim = 4;
        let half = head_dim / 2;
        let block_size = 2;
        let num_blocks = 3;
        let num_q_heads = 1;
        let num_kv_heads = 1;

        let q_in = vec![1.0_f32, 2.0, 3.0, 4.0];
        let k_in = vec![5.0_f32, 6.0, 7.0, 8.0];
        let v_in = vec![9.0_f32, 10.0, 11.0, 12.0];
        let positions = vec![0u32];
        // Pick slot 3 → block_id = 1, block_offset = 1 (block 1, second token).
        let slot_mapping = vec![3u32];
        // cos[0,d] = 1 for d in [0,half); sin[0,d] = 0.
        let mut cos_table = vec![0.0_f32; head_dim];
        for d in 0..half {
            cos_table[d] = 1.0;
        }
        let sin_table = vec![0.0_f32; head_dim];

        let mut q_out = vec![0.0_f32; q_in.len()];
        let layout = crate::paged_kv_layout::PagedKvLayout {
            num_blocks: num_blocks as u32,
            num_kv_heads: num_kv_heads as u32,
            block_size: block_size as u32,
            head_dim: head_dim as u32,
        };
        let mut kv_cache_k = vec![0.0_f32; layout.buffer_elems()];
        let mut kv_cache_v = vec![0.0_f32; layout.buffer_elems()];

        rope_append(
            &q_in,
            &k_in,
            &v_in,
            &positions,
            &slot_mapping,
            &cos_table,
            &sin_table,
            &mut q_out,
            &mut kv_cache_k,
            &mut kv_cache_v,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
        );

        // No rotation → q_out equals q_in.
        assert_eq!(q_out, q_in);
        // Slot 3 = block_id 1, block_offset 1. Cache stride per block:
        // num_kv_heads * block_size * head_dim = 1 * 2 * 4 = 8.
        // Per-block layout: head 0, offset 0 [0..4], offset 1 [4..8].
        // So slot 3 occupies kv_cache[1 * 8 + 0 * 8 + 1 * 4 + d] = idx 12..16.
        assert_eq!(&kv_cache_k[12..16], &k_in[..]);
        assert_eq!(&kv_cache_v[12..16], &v_in[..]);
        // Other slots untouched.
        for (i, &v) in kv_cache_k.iter().enumerate() {
            if !(12..16).contains(&i) {
                assert_eq!(v, 0.0, "kv_cache_k[{i}] should be untouched");
            }
        }
    }

    #[test]
    fn test_rope_append_actually_rotates() {
        // 1 token, head_dim = 2, half = 1. cos = 0, sin = 1 → 90° rotation:
        // (x0, x1) → (x0*cos - x1*sin, x1*cos + x0*sin) = (-x1, x0).
        let head_dim = 2;
        let q_in = vec![3.0_f32, 4.0]; // (x0, x1) = (3, 4)
        let k_in = vec![5.0_f32, 6.0];
        let v_in = vec![7.0_f32, 8.0];
        let positions = vec![0u32];
        let slot_mapping = vec![0u32];
        // cos[0, 0] = 0; sin[0, 0] = 1.
        let cos_table = vec![0.0_f32, 0.0]; // [max_pos=1, head_dim=2]
        let sin_table = vec![1.0_f32, 0.0];

        let mut q_out = vec![0.0; 2];
        let mut kv_cache_k = vec![0.0; 2]; // 1 block, 1 kv_head, block_size=1, head_dim=2
        let mut kv_cache_v = vec![0.0; 2];

        rope_append(
            &q_in,
            &k_in,
            &v_in,
            &positions,
            &slot_mapping,
            &cos_table,
            &sin_table,
            &mut q_out,
            &mut kv_cache_k,
            &mut kv_cache_v,
            1,
            1,
            head_dim,
            1,
        );
        // q_out = (3*0 - 4*1, 4*0 + 3*1) = (-4, 3)
        assert_eq!(q_out, vec![-4.0, 3.0]);
        // K rotated similarly: (5, 6) → (-6, 5).
        assert_eq!(kv_cache_k, vec![-6.0, 5.0]);
        // V un-rotated.
        assert_eq!(kv_cache_v, vec![7.0, 8.0]);
    }

    #[test]
    fn test_attention_via_cache_matches_decode_ref() {
        // 1 sequence, single head, head_dim=4, block_size=2,
        // max_blocks_per_seq=2. Build a contiguous KV cache (block_table
        // = [0, 1]) and verify attention_via_cache matches
        // attention_decode for the same K/V data.
        let head_dim = 4;
        let block_size = 2;
        let max_blocks = 2;
        let num_q_heads = 1;
        let num_kv_heads = 1;
        let kv_len = 3; // tokens 0,1 in block 0; token 2 in block 1.

        // q
        let q = vec![1.0_f32, 0.0, 0.0, 0.0];
        // contiguous K/V: [kv_len, head_dim]
        let k_contig = vec![
            1.0, 0.0, 0.0, 0.0, // t=0
            0.0, 1.0, 0.0, 0.0, // t=1
            1.0, 1.0, 0.0, 0.0, // t=2
        ];
        let v_contig = vec![
            1.0, 2.0, 3.0, 4.0, // t=0
            5.0, 6.0, 7.0, 8.0, // t=1
            9.0, 10.0, 11.0, 12.0, // t=2
        ];

        // Pack into paged cache: 2 blocks × 1 kv_head × block_size × head_dim
        let layout = crate::paged_kv_layout::PagedKvLayout {
            num_blocks: 2,
            num_kv_heads: num_kv_heads as u32,
            block_size: block_size as u32,
            head_dim: head_dim as u32,
        };
        let mut kv_cache_k = vec![0.0_f32; layout.buffer_elems()];
        let mut kv_cache_v = vec![0.0_f32; layout.buffer_elems()];
        for t in 0..kv_len {
            let logical = t / block_size;
            let off = t % block_size;
            // kv_head = 0 in this single-head test fixture.
            let dst = layout.elem_offset(logical as u32, 0, off as u32);
            kv_cache_k[dst..dst + head_dim]
                .copy_from_slice(&k_contig[t * head_dim..(t + 1) * head_dim]);
            kv_cache_v[dst..dst + head_dim]
                .copy_from_slice(&v_contig[t * head_dim..(t + 1) * head_dim]);
        }
        let block_table = vec![0u32, 1];
        let seq_used_k = vec![kv_len as u32];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut paged_out = vec![0.0_f32; head_dim];
        attention_via_cache(
            &q,
            &kv_cache_k,
            &kv_cache_v,
            &block_table,
            &seq_used_k,
            &mut paged_out,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks,
            scale,
        );

        // Compare against contiguous attention_decode (which has no
        // +1e-6 guard) — at well-conditioned softmax sums the
        // difference is negligible (sum_exp >> 1e-6 here).
        let mut ref_out = vec![0.0_f32; head_dim];
        attention_decode(
            &q,
            &k_contig,
            &v_contig,
            &mut ref_out,
            kv_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
        );
        for (i, (&p, &r)) in paged_out.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (p - r).abs() < 1e-4,
                "paged vs contig mismatch at d={i}: paged={p} contig={r}"
            );
        }
    }

    #[test]
    fn test_attention_via_cache_zero_kv_len() {
        // seq_used_k = 0 → no KV tokens. Output is all-zeros after
        // the weighted-sum loop runs zero iterations.
        let mut output = vec![123.0_f32; 4];
        let q = vec![1.0_f32; 4];
        let kv_cache_k = vec![0.0_f32; 8]; // 1 block × 1 head × bs=2 × hd=4
        let kv_cache_v = vec![0.0_f32; 8];
        attention_via_cache(
            &q,
            &kv_cache_k,
            &kv_cache_v,
            &[0u32, 0],
            &[0u32],
            &mut output,
            1,
            1,
            4,
            2,
            2,
            0.5,
        );
        for v in &output {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_attention_decode_single_head() {
        let head_dim = 4;
        let seq_len = 2;
        let num_heads = 1;
        let num_kv_heads = 1;

        let q = vec![1.0, 0.0, 0.0, 0.0]; // query
        let k_cache = vec![
            1.0, 0.0, 0.0, 0.0, // k[0]
            0.0, 1.0, 0.0, 0.0, // k[1]
        ];
        let v_cache = vec![
            1.0, 2.0, 3.0, 4.0, // v[0]
            5.0, 6.0, 7.0, 8.0, // v[1]
        ];
        let mut output = vec![0.0; 4];
        let scale = 1.0 / (head_dim as f32).sqrt();

        attention_decode(
            &q,
            &k_cache,
            &v_cache,
            &mut output,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        );

        // q·k[0] = 1*scale, q·k[1] = 0 → softmax ≈ [0.622, 0.378]
        // output ≈ 0.622 * v[0] + 0.378 * v[1]
        assert!(output[0] > 0.5 && output[0] < 4.0); // sanity check
    }
}
