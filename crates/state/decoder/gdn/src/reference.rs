// SPDX-License-Identifier: Apache-2.0
//! Gated-DeltaNet's pure-Rust f32 reference, one function per kernel, backend-free.
//!
//! Each function mirrors one GPU kernel so the kernels can be golden-checked independently. The
//! Metal target's `cpu_golden` re-exports them.

// Qwen3.5 / Qwen3-Next linear-attention references.
//
// Decomposed to mirror the four surviving CUDA kernels (`gdn_conv1d_*`,
// `gdn_gating`, `gdn_recurrent_fwd`, `gdn_rms_norm_gated`) so each can be
// golden-checked independently against these CPU refs. The recurrence math is
// transcribed from the reference Triton kernel
// `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
// (`fused_recurrent_gated_delta_rule_fwd_kernel`); the surrounding conv1d +
// gating + gated-norm from `vllm/model_executor/models/qwen3_next.py`.

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

/// The inputs of [`gdn_gating`]: `a`,`b` are `[num_tokens, num_heads]`; `a_log`,`dt_bias` are
/// `[num_heads]`.
#[derive(Clone, Copy, Debug)]
pub struct GatingInputs<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub num_heads: usize,
    pub num_tokens: usize,
}

/// GDN input-dependent gating.
///
/// `g[t,h]    = -exp(A_log[h]) · softplus(a[t,h] + dt_bias[h])`  (log-decay, ≤0)
/// `beta[t,h] = sigmoid(b[t,h])`
///
/// `a`,`b`: `[num_tokens, num_heads]`; `a_log`,`dt_bias`: `[num_heads]`.
/// softplus uses the numerically-stable threshold (20.0) from the kernel.
pub fn gdn_gating(inputs: &GatingInputs<'_>, g_out: &mut [f32], beta_out: &mut [f32]) {
    let GatingInputs {
        a,
        b,
        a_log,
        dt_bias,
        num_heads,
        num_tokens,
    } = *inputs;
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

/// The inputs of [`gdn_recurrent`], with the shapes its doc gives.
#[derive(Clone, Copy, Debug)]
pub struct RecurrentInputs<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub g: &'a [f32],
    pub beta: &'a [f32],
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub num_tokens: usize,
    pub scale: f32,
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
pub fn gdn_recurrent(inputs: &RecurrentInputs<'_>, o: &mut [f32]) {
    let RecurrentInputs {
        q,
        k,
        v,
        g,
        beta,
        num_k_heads,
        num_v_heads,
        head_k,
        head_v,
        num_tokens,
        scale,
    } = *inputs;
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
        let gating = GatingInputs {
            a: &a,
            b: &b,
            a_log: &a_log,
            dt_bias: &dt_bias,
            num_heads: nv,
            num_tokens: t,
        };
        gdn_gating(&gating, &mut g, &mut beta);
        let g_sum: f32 = g.iter().sum();
        let beta_sum: f32 = beta.iter().sum();
        assert!((g_sum - (-24.235_298)).abs() < 1e-3, "g_sum {g_sum}");
        assert!((beta_sum - 14.095_717).abs() < 1e-3, "beta_sum {beta_sum}");
        assert!(
            (g[0] + std::f32::consts::LN_2).abs() < 1e-5,
            "g[0] {}",
            g[0]
        );

        // 4. recurrent gated delta-rule scan
        let mut o = vec![0f32; t * value_dim];
        let scan = RecurrentInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            num_k_heads: nk,
            num_v_heads: nv,
            head_k: hk,
            head_v: hv,
            num_tokens: t,
            scale,
        };
        gdn_recurrent(&scan, &mut o);
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
}
