// SPDX-License-Identifier: Apache-2.0
//! Golden tests for the Gated-DeltaNet Metal kernels, validated against
//! reference math that mirrors `scratchy_forward_compiler::cpu_golden::gdn_*` (the
//! canonical oracle, itself pinned to transformers + mlx-lm). The reference
//! is inlined here because `cpu_golden` lives in `scratchy-forward-compiler`, which
//! `scratchy-target-metal` cannot depend on (wrong direction), and the
//! `scratchy-forward-compiler` lib-test target is independently pre-broken.
//!
//! Each kernel is dispatched standalone via `SpecializedPipelineCache` +
//! `get_or_build` (the same path production uses through
//! `pipeline_for_command`), using the f32 instantiation for exact parity.

#![cfg(target_os = "macos")]

mod common;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// Deterministic fill matching `cpu_golden`'s `fill(i) = sin(i*0.1)*0.5`.
fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect()
}

fn buf_f32(device: &Device, data: &[f32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

fn buf_zero_f32(device: &Device, n: usize) -> Buffer {
    let bytes = (n * 4).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes) };
    buf
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, n) }.to_vec()
}

fn buf_i32(device: &Device, data: &[i32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

fn buf_u32(device: &Device, data: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

/// CPU reference mirroring `cpu_golden::gdn_recurrent` (single sequence, zero
/// initial state). q/k: [T, nk*hk]; v/o: [T, nv*hv]; g/beta: [T, nv].
#[allow(clippy::too_many_arguments)]
fn recurrent_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    t: usize,
    scale: f32,
) -> Vec<f32> {
    let key_dim = nk * hk;
    let value_dim = nv * hv;
    let groups = nv / nk;
    let mut state = vec![0.0f32; nv * hv * hk];
    let mut o = vec![0f32; t * value_dim];
    let l2 = |s: &[f32]| -> f32 { (s.iter().map(|&x| x * x).sum::<f32>() + 1e-6).sqrt() };
    for ti in 0..t {
        for h in 0..nv {
            let ki = h / groups;
            let qsrc = &q[ti * key_dim + ki * hk..][..hk];
            let ksrc = &k[ti * key_dim + ki * hk..][..hk];
            let qinv = scale / l2(qsrc);
            let kinv = 1.0 / l2(ksrc);
            let qn: Vec<f32> = qsrc.iter().map(|&x| x * qinv).collect();
            let kn: Vec<f32> = ksrc.iter().map(|&x| x * kinv).collect();
            let sh = &mut state[h * hv * hk..][..hv * hk];
            let decay = g[ti * nv + h].exp();
            let gt = beta[ti * nv + h];
            for s in sh.iter_mut() {
                *s *= decay;
            }
            let vsrc = &v[ti * value_dim + h * hv..][..hv];
            for vd in 0..hv {
                let mut sk = 0.0f32;
                for kd in 0..hk {
                    sk += sh[vd * hk + kd] * kn[kd];
                }
                let u = gt * (vsrc[vd] - sk);
                for kd in 0..hk {
                    sh[vd * hk + kd] += u * kn[kd];
                }
                let mut ov = 0.0f32;
                for kd in 0..hk {
                    ov += sh[vd * hk + kd] * qn[kd];
                }
                o[ti * value_dim + h * hv + vd] = ov;
            }
        }
    }
    o
}

/// Slice q/k/v out of a `conv_out`-layout buffer ([q:key_dim|k:key_dim|v:value_dim]).
fn split_conv(
    conv_out: &[f32],
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    t: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let key_dim = nk * hk;
    let value_dim = nv * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let mut q = vec![0f32; t * key_dim];
    let mut k = vec![0f32; t * key_dim];
    let mut v = vec![0f32; t * value_dim];
    for ti in 0..t {
        let row = &conv_out[ti * conv_dim..][..conv_dim];
        q[ti * key_dim..][..key_dim].copy_from_slice(&row[0..key_dim]);
        k[ti * key_dim..][..key_dim].copy_from_slice(&row[key_dim..2 * key_dim]);
        v[ti * value_dim..][..value_dim].copy_from_slice(&row[2 * key_dim..]);
    }
    (q, k, v)
}

/// Dispatch `gdn_scan_varlen_f32` for one varlen batch; returns `o` [T, value_dim].
/// `state_buf` is mutated in place (shared across calls for continuity tests).
#[allow(clippy::too_many_arguments)]
fn dispatch_scan(
    device: &Device,
    cache: &SpecializedPipelineCache,
    conv_out: &[f32],
    g: &[f32],
    beta: &[f32],
    state_buf: &Buffer,
    cu: &[i32],
    si: &[i32],
    fresh: &[u32],
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    num_tokens: usize,
) -> Option<Vec<f32>> {
    let value_dim = nv * hv;
    let scale = (hk as f32).powf(-0.5);
    let key = PipelineKey::new(
        "gdn_scan_varlen",
        "gdn_scan_varlen_f32",
        vec![
            ConstantValue::uint(0, nk as u32),
            ConstantValue::uint(1, nv as u32),
            ConstantValue::uint(2, hk as u32),
            ConstantValue::uint(3, hv as u32),
            ConstantValue::float(4, scale),
        ],
    );
    let pipeline = baked_build(cache, &key).expect("gdn_scan_varlen pipeline");

    let o_buf = buf_zero_f32(device, num_tokens * value_dim);
    let conv_buf = buf_f32(device, conv_out);
    let g_buf = buf_f32(device, g);
    let beta_buf = buf_f32(device, beta);
    let cu_buf = buf_i32(device, cu);
    let si_buf = buf_i32(device, si);
    let fresh_buf = buf_u32(device, fresh);

    let num_seqs = cu.len() - 1;
    let tgx = hv.min(256);
    if !common::dispatch_threadgroups(
        device,
        &pipeline,
        &[
            &o_buf, &conv_buf, &g_buf, &beta_buf, state_buf, &cu_buf, &si_buf, &fresh_buf,
        ],
        MTLSize {
            width: hv.div_ceil(tgx),
            height: nv,
            depth: num_seqs,
        },
        MTLSize {
            width: tgx,
            height: 1,
            depth: 1,
        },
    ) {
        return None;
    }
    Some(read_f32(&o_buf, num_tokens * value_dim))
}

/// CPU reference mirroring `cpu_golden::gdn_causal_conv1d` (single fresh seq,
/// zero left-pad, +SiLU). `weight` is `[conv_dim, kernel]`.
fn conv1d_ref(
    x: &[f32],
    weight: &[f32],
    conv_dim: usize,
    kernel: usize,
    num_tokens: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; num_tokens * conv_dim];
    for t in 0..num_tokens {
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for j in 0..kernel {
                let ti = t as isize - (kernel as isize - 1) + j as isize;
                if ti >= 0 {
                    acc += weight[c * kernel + j] * x[ti as usize * conv_dim + c];
                }
            }
            out[t * conv_dim + c] = acc / (1.0 + (-acc).exp());
        }
    }
    out
}

/// `gdn_gating_f32` vs reference: g = -exp(A_log[h])*softplus(a+dt_bias[h]),
/// beta = sigmoid(b). Tiny cpu_golden config nv=4, T=6.
#[test]
fn gdn_gating_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (nv, t) = (4usize, 6usize);
    let n = t * nv;
    let a = fill(n);
    let b = fill(n);
    let a_log = fill(nv);
    let dt_bias = fill(nv);

    // Reference (mirrors cpu_golden::gdn_gating).
    let softplus = |x: f32| if x <= 20.0 { x.exp().ln_1p() } else { x };
    let mut g_ref = vec![0f32; n];
    let mut beta_ref = vec![0f32; n];
    for ti in 0..t {
        for h in 0..nv {
            let idx = ti * nv + h;
            g_ref[idx] = -(a_log[h].exp()) * softplus(a[idx] + dt_bias[h]);
            beta_ref[idx] = 1.0 / (1.0 + (-b[idx]).exp());
        }
    }

    let key = PipelineKey::new(
        "gdn_gating",
        "gdn_gating_f32",
        vec![
            ConstantValue::uint(0, n as u32),
            ConstantValue::uint(1, nv as u32),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_gating pipeline");

    let a_buf = buf_f32(&device, &a);
    let b_buf = buf_f32(&device, &b);
    let alog_buf = buf_f32(&device, &a_log);
    let dt_buf = buf_f32(&device, &dt_bias);
    let g_buf = buf_zero_f32(&device, n);
    let beta_buf = buf_zero_f32(&device, n);

    let groups = n.div_ceil(256).max(1);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&g_buf, &beta_buf, &a_buf, &b_buf, &alog_buf, &dt_buf],
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let g_metal = read_f32(&g_buf, n);
    let beta_metal = read_f32(&beta_buf, n);
    for i in 0..n {
        assert!(
            (g_metal[i] - g_ref[i]).abs() < 1e-4,
            "g[{i}] metal={} ref={}",
            g_metal[i],
            g_ref[i]
        );
        assert!(
            (beta_metal[i] - beta_ref[i]).abs() < 1e-4,
            "beta[{i}] metal={} ref={}",
            beta_metal[i],
            beta_ref[i]
        );
    }
}

/// `gdn_rms_norm_gated_f32` vs reference: per value-head rmsnorm of the scan
/// output `x`, times `weight`, times `SiLU(z)` (NOT plain sigmoid). Mirrors
/// `cpu_golden::gdn_rms_norm_gated`. Tiny config d=hv=4, total_rows=t*nv=24.
#[test]
fn gdn_rms_norm_gated_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (d, total_rows) = (4usize, 24usize);
    let eps = 1e-6f32;
    let x = fill(total_rows * d);
    let z = fill(total_rows * d);
    let weight = fill(d);

    // Reference (mirrors cpu_golden::gdn_rms_norm_gated).
    let mut out_ref = vec![0f32; total_rows * d];
    for r in 0..total_rows {
        let row = &x[r * d..][..d];
        let var = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            let zi = z[r * d + i];
            let silu_z = zi / (1.0 + (-zi).exp());
            out_ref[r * d + i] = row[i] * inv * weight[i] * silu_z;
        }
    }

    let key = PipelineKey::new(
        "gdn_rms_norm_gated",
        "gdn_rms_norm_gated_f32",
        vec![
            ConstantValue::uint(0, d as u32),
            ConstantValue::uint(1, total_rows as u32),
            ConstantValue::float(2, eps),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_rms_norm_gated pipeline");

    let out_buf = buf_zero_f32(&device, total_rows * d);
    let x_buf = buf_f32(&device, &x);
    let z_buf = buf_f32(&device, &z);
    let w_buf = buf_f32(&device, &weight);

    // One threadgroup per row; 256 threads (> any head_v_dim) do the
    // threadgroup reduction (idle lanes contribute 0).
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &x_buf, &z_buf, &w_buf],
        MTLSize {
            width: total_rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let out_metal = read_f32(&out_buf, total_rows * d);
    for i in 0..total_rows * d {
        assert!(
            (out_metal[i] - out_ref[i]).abs() < 1e-4,
            "out[{i}] metal={} ref={}",
            out_metal[i],
            out_ref[i]
        );
    }
}

/// `gdn_conv1d_varlen_f32`, single fresh sequence (is_fresh=1, zero left-pad):
/// must match `cpu_golden::gdn_causal_conv1d`. Config conv_dim=32, kernel=4.
#[test]
fn gdn_conv1d_varlen_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (conv_dim, kernel, t) = (32usize, 4usize, 6usize);
    let state_len = kernel - 1;
    let x = fill(t * conv_dim);
    let w = fill(conv_dim * kernel);
    let out_ref = conv1d_ref(&x, &w, conv_dim, kernel, t);

    let key = PipelineKey::new(
        "gdn_conv1d_varlen",
        "gdn_conv1d_varlen_f32",
        vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");

    let out_buf = buf_zero_f32(&device, t * conv_dim);
    let x_buf = buf_f32(&device, &x);
    let w_buf = buf_f32(&device, &w);
    let state_buf = buf_zero_f32(&device, conv_dim * state_len); // 1 slot
    let cu = buf_i32(&device, &[0, t as i32]);
    let si = buf_i32(&device, &[0]);
    let fresh = buf_u32(&device, &[1]);

    let tg_y = conv_dim.min(256);
    let groups_y = conv_dim.div_ceil(tg_y);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &x_buf, &w_buf, &state_buf, &cu, &si, &fresh],
        MTLSize {
            width: 1,
            height: groups_y,
            depth: 1,
        }, // x = num_seqs
        MTLSize {
            width: 1,
            height: tg_y,
            depth: 1,
        },
    ) {
        return;
    }

    let out_metal = read_f32(&out_buf, t * conv_dim);
    for i in 0..t * conv_dim {
        assert!(
            (out_metal[i] - out_ref[i]).abs() < 1e-4,
            "out[{i}] metal={} ref={}",
            out_metal[i],
            out_ref[i]
        );
    }
}

/// Multi-step continuity for the conv_state ring: forward(6) then forward(1)
/// as a *continuation* (is_fresh=0, sharing the same persistent state buffer)
/// must equal token 6 of forward(7)-fresh. This exercises the ring carry-over
/// that `cpu_golden::gdn_causal_conv1d` does not cover.
#[test]
fn gdn_conv1d_varlen_continuity() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (conv_dim, kernel) = (32usize, 4usize);
    let state_len = kernel - 1;
    let t_full = 7usize;
    let x_full = fill(t_full * conv_dim);
    let w = fill(conv_dim * kernel);
    let out_full = conv1d_ref(&x_full, &w, conv_dim, kernel, t_full);

    let key = PipelineKey::new(
        "gdn_conv1d_varlen",
        "gdn_conv1d_varlen_f32",
        vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");

    let w_buf = buf_f32(&device, &w);
    let state_buf = buf_zero_f32(&device, conv_dim * state_len); // shared across runs
    let tg_y = conv_dim.min(256);
    let groups_y = conv_dim.div_ceil(tg_y);

    let run = |x: &[f32], num_tokens: usize, is_fresh: u32| -> Option<Vec<f32>> {
        let out_buf = buf_zero_f32(&device, num_tokens * conv_dim);
        let x_buf = buf_f32(&device, x);
        let cu = buf_i32(&device, &[0, num_tokens as i32]);
        let si = buf_i32(&device, &[0]);
        let fresh = buf_u32(&device, &[is_fresh]);
        if !common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[&out_buf, &x_buf, &w_buf, &state_buf, &cu, &si, &fresh],
            MTLSize {
                width: 1,
                height: groups_y,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: tg_y,
                depth: 1,
            },
        ) {
            return None;
        }
        Some(read_f32(&out_buf, num_tokens * conv_dim))
    };

    // Run A: first 6 tokens, fresh → seeds state_buf.
    let Some(_a) = run(&x_full[..6 * conv_dim], 6, 1) else {
        return;
    };
    // Run B: 7th token as continuation → reads the carried state.
    let Some(b) = run(&x_full[6 * conv_dim..], 1, 0) else {
        return;
    };

    let ref_last = &out_full[6 * conv_dim..];
    for c in 0..conv_dim {
        assert!(
            (b[c] - ref_last[c]).abs() < 1e-4,
            "continuity out[{c}] metal={} ref={}",
            b[c],
            ref_last[c]
        );
    }
}

/// `gdn_scan_varlen_f32` vs `cpu_golden::gdn_recurrent` on the tiny config
/// (nk=2, nv=4, hk=hv=4) — which exercises GVA (groups=2, key head = h/2).
#[test]
fn gdn_scan_varlen_matches_reference() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (nk, nv, hk, hv, t) = (2usize, 4usize, 4usize, 4usize, 6usize);
    let key_dim = nk * hk;
    let value_dim = nv * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);

    let conv_out = fill(t * conv_dim);
    let g = fill(t * nv);
    let beta = fill(t * nv);
    let (q, k, v) = split_conv(&conv_out, nk, nv, hk, hv, t);
    let o_ref = recurrent_ref(&q, &k, &v, &g, &beta, nk, nv, hk, hv, t, scale);

    let state_buf = buf_zero_f32(&device, nv * hv * hk); // 1 slot
    let Some(o) = dispatch_scan(
        &device,
        &cache,
        &conv_out,
        &g,
        &beta,
        &state_buf,
        &[0, t as i32],
        &[0],
        &[1],
        nk,
        nv,
        hk,
        hv,
        t,
    ) else {
        return;
    };

    for i in 0..t * value_dim {
        assert!(
            (o[i] - o_ref[i]).abs() < 1e-4,
            "o[{i}] metal={} ref={}",
            o[i],
            o_ref[i]
        );
    }
}

/// `gdn_scan_varlen_f32` at production head dims (hk=hv=128) — exercises the
/// `b_h[128]` register state row at full width.
#[test]
fn gdn_scan_varlen_production_head_dim() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (nk, nv, hk, hv, t) = (2usize, 2usize, 128usize, 128usize, 3usize);
    let key_dim = nk * hk;
    let value_dim = nv * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);

    let conv_out = fill(t * conv_dim);
    let g = fill(t * nv);
    let beta = fill(t * nv);
    let (q, k, v) = split_conv(&conv_out, nk, nv, hk, hv, t);
    let o_ref = recurrent_ref(&q, &k, &v, &g, &beta, nk, nv, hk, hv, t, scale);

    let state_buf = buf_zero_f32(&device, nv * hv * hk);
    let Some(o) = dispatch_scan(
        &device,
        &cache,
        &conv_out,
        &g,
        &beta,
        &state_buf,
        &[0, t as i32],
        &[0],
        &[1],
        nk,
        nv,
        hk,
        hv,
        t,
    ) else {
        return;
    };

    for i in 0..t * value_dim {
        assert!(
            (o[i] - o_ref[i]).abs() < 1e-3,
            "o[{i}] metal={} ref={}",
            o[i],
            o_ref[i]
        );
    }
}

/// Multi-step continuity for the ssm_state: forward(5) then forward(1) as a
/// continuation (is_fresh=0, shared state buffer) == token 5 of forward(6)-fresh.
#[test]
fn gdn_scan_varlen_continuity() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (nk, nv, hk, hv) = (2usize, 4usize, 4usize, 4usize);
    let key_dim = nk * hk;
    let value_dim = nv * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);
    let t_full = 6usize;

    let conv_full = fill(t_full * conv_dim);
    let g_full = fill(t_full * nv);
    let beta_full = fill(t_full * nv);
    let (q, k, v) = split_conv(&conv_full, nk, nv, hk, hv, t_full);
    let o_full = recurrent_ref(
        &q, &k, &v, &g_full, &beta_full, nk, nv, hk, hv, t_full, scale,
    );

    let state_buf = buf_zero_f32(&device, nv * hv * hk);

    // Run A: first 5 tokens, fresh → seeds ssm_state.
    let Some(_a) = dispatch_scan(
        &device,
        &cache,
        &conv_full[..5 * conv_dim],
        &g_full[..5 * nv],
        &beta_full[..5 * nv],
        &state_buf,
        &[0, 5],
        &[0],
        &[1],
        nk,
        nv,
        hk,
        hv,
        5,
    ) else {
        return;
    };
    // Run B: 6th token continuation → reads carried ssm_state.
    let Some(b) = dispatch_scan(
        &device,
        &cache,
        &conv_full[5 * conv_dim..],
        &g_full[5 * nv..],
        &beta_full[5 * nv..],
        &state_buf,
        &[0, 1],
        &[0],
        &[0],
        nk,
        nv,
        hk,
        hv,
        1,
    ) else {
        return;
    };

    let ref_last = &o_full[5 * value_dim..];
    for c in 0..value_dim {
        assert!(
            (b[c] - ref_last[c]).abs() < 1e-4,
            "scan continuity o[{c}] metal={} ref={}",
            b[c],
            ref_last[c]
        );
    }
}

/// `gdn_scan_simd_f32` — the decode scan, gating folded in, mlx-lm's simdgroup-per-value-dim
/// mapping — against the two kernels it replaces chained over the same inputs and state:
/// `gdn_gating_f32` then `gdn_scan_varlen_f32`. Production head dims, two sequences — a continued
/// state and a fresh one — of one token and of three. Its dots over head_k reduce lane-parallel, not
/// serially: the outputs and the state left behind agree to f32 rounding of those sums.
#[test]
fn gdn_scan_simd_is_the_gating_then_scan() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv) = (2usize, 4usize, 128usize, 128usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    for (cu, fresh) in [
        (vec![0, 1, 2], vec![0u32, 1]),
        (vec![0, 3, 4], vec![1u32, 0]),
        // Prefill-shaped: long sequences, mixed continued/fresh — the regime
        // the lowering routes to `gdn_scan_simd` at every bucket since the
        // bucket_m cap came off. The equivalence this test pins must hold at
        // prefill lengths, not only decode steps.
        (vec![0, 517, 521], vec![1u32, 0]),
    ] {
        let t = *cu.last().expect("a sequence") as usize;
        let seqs = cu.len() - 1;
        let n = t * nv;
        let conv_out = fill(t * conv_dim);
        let (a, b, a_log, dt_bias) = (fill(n), shifted(n, 3), fill(nv), shifted(nv, 5));
        let state0 = shifted(seqs * nv * hv * hk, 11);
        let si: Vec<i32> = (0..seqs as i32).collect();
        let build = |lib: &'static str, name: &'static str, c: Vec<ConstantValue>| {
            baked_build(&cache, &PipelineKey::new(lib, name, c)).expect(name)
        };
        let scan_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
            ]
        };
        let size = |width, height, depth| MTLSize {
            width,
            height,
            depth,
        };
        let (cu_buf, si_buf) = (buf_i32(&device, &cu), buf_i32(&device, &si));
        let (fresh_buf, conv_buf) = (buf_u32(&device, &fresh), buf_f32(&device, &conv_out));
        let (a_buf, b_buf) = (buf_f32(&device, &a), buf_f32(&device, &b));
        let (alog_buf, dt_buf) = (buf_f32(&device, &a_log), buf_f32(&device, &dt_bias));

        // The chain.
        let gating_consts = vec![
            ConstantValue::uint(0, n as u32),
            ConstantValue::uint(1, nv as u32),
        ];
        let gating = build("gdn_gating", "gdn_gating_f32", gating_consts);
        let (g_buf, beta_buf) = (buf_zero_f32(&device, n), buf_zero_f32(&device, n));
        let bufs = [&g_buf, &beta_buf, &a_buf, &b_buf, &alog_buf, &dt_buf];
        let grid = size(n.div_ceil(256), 1, 1);
        if !common::dispatch_threadgroups(&device, &gating, &bufs, grid, size(256, 1, 1)) {
            return;
        }
        let scan = build("gdn_scan_varlen", "gdn_scan_varlen_f32", scan_consts());
        let chain_o = buf_zero_f32(&device, t * value_dim);
        let chain_state = buf_f32(&device, &state0);
        let bufs = [
            &chain_o,
            &conv_buf,
            &g_buf,
            &beta_buf,
            &chain_state,
            &cu_buf,
            &si_buf,
            &fresh_buf,
        ];
        let grid = size(1, nv, seqs);
        if !common::dispatch_threadgroups(&device, &scan, &bufs, grid, size(hv, 1, 1)) {
            return;
        }

        // The simdgroup scan.
        let simd = build("gdn_scan_varlen", "gdn_scan_simd_f32", scan_consts());
        let simd_o = buf_zero_f32(&device, t * value_dim);
        let simd_state = buf_f32(&device, &state0);
        let bufs = [
            &simd_o,
            &conv_buf,
            &a_buf,
            &b_buf,
            &simd_state,
            &cu_buf,
            &si_buf,
            &fresh_buf,
            &alog_buf,
            &dt_buf,
        ];
        let grid = size(1, nv * hv / 4, seqs);
        if !common::dispatch_threadgroups(&device, &simd, &bufs, grid, size(32, 4, 1)) {
            return;
        }
        let close = |what: &str, got: Vec<f32>, want: Vec<f32>| {
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * scale,
                    "{what}[{i}]: {g} vs {w}, cu {cu:?}"
                );
            }
        };
        let rows = t * value_dim;
        close("o", read_f32(&simd_o, rows), read_f32(&chain_o, rows));
        let s = state0.len();
        close("state", read_f32(&simd_state, s), read_f32(&chain_state, s));
        // The state moved: a scan that left it would pass the comparison above only by accident.
        assert_ne!(read_f32(&simd_state, s), state0, "cu {cu:?}");
    }
}

/// `gdn_scan_pipelined_f32` — the prefill scan (gdn_scan_pipelined.metal, omlx Kernel P's
/// 8-lanes-per-row staging under this crate's contracts) — against the gating→scan chain over
/// the same inputs and state: outputs and state left behind must agree to f32 rounding (the
/// butterfly's summation order is simd_sum's). Mixed continued/fresh varlen batches including a
/// decode-shaped 1-token continuation; block-tail lengths exercise the partial block.
#[test]
fn gdn_scan_pipelined_is_the_gating_then_scan() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    // Same head dims and grouping as production (hk=hv=128, 3 v-heads per
    // k-head); fewer heads keeps the reference tractable.
    let (nk, nv, hk, hv) = (2usize, 6usize, 128usize, 128usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    for (cu, fresh) in [
        // Prefill lengths around TB=12 boundaries: 25 = 2*12 + 1 (tail of 1).
        (vec![0, 25], vec![1u32]),
        (vec![0, 24, 37], vec![0u32, 1]),
        (vec![0, 13, 26], vec![1u32, 0]),
        // Decode-shaped: one token on a carried state.
        (vec![0, 1], vec![0u32]),
    ] {
        let t = *cu.last().expect("a sequence") as usize;
        let seqs = cu.len() - 1;
        let n = t * nv;
        let conv_out = fill(t * conv_dim);
        let (a, b, a_log, dt_bias) = (fill(n), shifted(n, 3), fill(nv), shifted(nv, 5));
        let state0 = shifted(seqs * nv * hv * hk, 11);
        let si: Vec<i32> = (0..seqs as i32).collect();
        let build = |lib: &'static str, name: &'static str, c: Vec<ConstantValue>| {
            baked_build(&cache, &PipelineKey::new(lib, name, c)).expect(name)
        };
        let scan_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
            ]
        };
        let pipe_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
                ConstantValue::uint(5, 12), // TB
            ]
        };
        let size = |width, height, depth| MTLSize {
            width,
            height,
            depth,
        };
        let (cu_buf, si_buf) = (buf_i32(&device, &cu), buf_i32(&device, &si));
        let (fresh_buf, conv_buf) = (buf_u32(&device, &fresh), buf_f32(&device, &conv_out));
        let (a_buf, b_buf) = (buf_f32(&device, &a), buf_f32(&device, &b));
        let (alog_buf, dt_buf) = (buf_f32(&device, &a_log), buf_f32(&device, &dt_bias));

        // The chain: gating then the per-thread varlen scan (the canonical reference).
        let gating = build(
            "gdn_gating",
            "gdn_gating_f32",
            vec![
                ConstantValue::uint(0, n as u32),
                ConstantValue::uint(1, nv as u32),
            ],
        );
        let (g_buf, beta_buf) = (buf_zero_f32(&device, n), buf_zero_f32(&device, n));
        let bufs = [&g_buf, &beta_buf, &a_buf, &b_buf, &alog_buf, &dt_buf];
        let grid = size(n.div_ceil(256), 1, 1);
        if !common::dispatch_threadgroups(&device, &gating, &bufs, grid, size(256, 1, 1)) {
            return;
        }
        let scan = build("gdn_scan_varlen", "gdn_scan_varlen_f32", scan_consts());
        let chain_o = buf_zero_f32(&device, t * value_dim);
        let chain_state = buf_f32(&device, &state0);
        let bufs = [
            &chain_o,
            &conv_buf,
            &g_buf,
            &beta_buf,
            &chain_state,
            &cu_buf,
            &si_buf,
            &fresh_buf,
        ];
        let grid = size(1, nv, seqs);
        if !common::dispatch_threadgroups(&device, &scan, &bufs, grid, size(hv, 1, 1)) {
            return;
        }

        // The pipelined scan.
        let pipe = build(
            "gdn_scan_pipelined",
            "gdn_scan_pipelined_f32",
            pipe_consts(),
        );
        let pipe_o = buf_zero_f32(&device, t * value_dim);
        let pipe_state = buf_f32(&device, &state0);
        let bufs = [
            &pipe_o,
            &conv_buf,
            &a_buf,
            &b_buf,
            &pipe_state,
            &cu_buf,
            &si_buf,
            &fresh_buf,
            &alog_buf,
            &dt_buf,
        ];
        let grid = size(value_dim / 16, 1, seqs);
        if !common::dispatch_threadgroups(&device, &pipe, &bufs, grid, size(128, 1, 1)) {
            return;
        }

        let close = |what: &str, got: Vec<f32>, want: Vec<f32>| {
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * scale,
                    "{what}[{i}]: {g} vs {w}, cu {cu:?}"
                );
            }
        };
        let rows = t * value_dim;
        close("o", read_f32(&pipe_o, rows), read_f32(&chain_o, rows));
        let s = state0.len();
        close("state", read_f32(&pipe_state, s), read_f32(&chain_state, s));
        assert_ne!(read_f32(&pipe_state, s), state0, "cu {cu:?}");
    }
}

/// `gdn_decode_f32` — a decode token's conv, gating, scan and gated norm in one command — against
/// the three commands it replaces chained over the same inputs and state: `gdn_conv1d_varlen_f32`,
/// `gdn_scan_simd_f32`, `gdn_rms_norm_gated_f32`. Production head dims, one, two and three value
/// heads a key head; two sequences of a token each, a continued one and a fresh one, in permuted
/// state slots. The conv state, a copy of inputs, agrees exactly; the output and the scan state are
/// the chain's statements in its order, which the shader compiler contracts and reassociates per
/// kernel under fast math, so they agree to f32 rounding.
#[test]
fn gdn_decode_is_the_conv_scan_norm() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (hk, hv, kernel, eps) = (128usize, 128usize, 4usize, 1e-6f32);
    let scale = (hk as f32).powf(-0.5);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let size = |width, height, depth| MTLSize {
        width,
        height,
        depth,
    };
    for ((nk, nv), fresh) in [
        ((2usize, 4usize), [0u32, 1]),
        ((2, 6), [1, 0]),
        ((2, 2), [0, 0]),
    ] {
        let (key_dim, value_dim) = (nk * hk, nv * hv);
        let conv_dim = 2 * key_dim + value_dim;
        let (t, slots) = (2usize, 2usize);
        let qkv = shifted(t * conv_dim, 7);
        let z = shifted(t * value_dim, 13);
        let (a, b) = (fill(t * nv), shifted(t * nv, 3));
        let (a_log, dt_bias, norm_w) = (fill(nv), shifted(nv, 5), shifted(hv, 17));
        let conv_w = shifted(conv_dim * kernel, 19);
        let conv0 = shifted(slots * conv_dim * (kernel - 1), 23);
        let ssm0 = shifted(slots * nv * hv * hk, 11);
        let build = |lib: &'static str, name: &'static str, c: Vec<ConstantValue>| {
            baked_build(&cache, &PipelineKey::new(lib, name, c)).expect(name)
        };
        let scan_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
            ]
        };
        let (cu, si) = (buf_i32(&device, &[0, 1, 2]), buf_i32(&device, &[1, 0]));
        let fresh = buf_u32(&device, &fresh);
        let (qkv_buf, z_buf, w_buf) = (
            buf_f32(&device, &qkv),
            buf_f32(&device, &z),
            buf_f32(&device, &conv_w),
        );
        let (a_buf, b_buf) = (buf_f32(&device, &a), buf_f32(&device, &b));
        let (alog_buf, dt_buf, norm_buf) = (
            buf_f32(&device, &a_log),
            buf_f32(&device, &dt_bias),
            buf_f32(&device, &norm_w),
        );

        // The chain.
        let conv_consts = vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
        ];
        let conv = build("gdn_conv1d_varlen", "gdn_conv1d_varlen_f32", conv_consts);
        let conv_out = buf_zero_f32(&device, t * conv_dim);
        let (chain_conv, chain_ssm) = (buf_f32(&device, &conv0), buf_f32(&device, &ssm0));
        let bufs = [&conv_out, &qkv_buf, &w_buf, &chain_conv, &cu, &si, &fresh];
        let grid = size(t, conv_dim.div_ceil(256), 1);
        if !common::dispatch_threadgroups(&device, &conv, &bufs, grid, size(1, 256, 1)) {
            return;
        }
        let scan = build("gdn_scan_varlen", "gdn_scan_simd_f32", scan_consts());
        let o = buf_zero_f32(&device, t * value_dim);
        let bufs = [
            &o, &conv_out, &a_buf, &b_buf, &chain_ssm, &cu, &si, &fresh, &alog_buf, &dt_buf,
        ];
        let grid = size(1, nv * hv / 4, t);
        if !common::dispatch_threadgroups(&device, &scan, &bufs, grid, size(32, 4, 1)) {
            return;
        }
        let norm_consts = vec![
            ConstantValue::uint(0, hv as u32),
            ConstantValue::uint(1, (t * nv) as u32),
            ConstantValue::float(2, eps),
        ];
        let norm = build("gdn_rms_norm_gated", "gdn_rms_norm_gated_f32", norm_consts);
        let chain_out = buf_zero_f32(&device, t * value_dim);
        let bufs = [&chain_out, &o, &z_buf, &norm_buf];
        let grid = size(t * nv, 1, 1);
        if !common::dispatch_threadgroups(&device, &norm, &bufs, grid, size(256, 1, 1)) {
            return;
        }

        // The one command.
        let mut consts = scan_consts();
        consts.extend([
            ConstantValue::uint(5, kernel as u32),
            ConstantValue::float(6, eps),
        ]);
        let decode = build("gdn_decode", "gdn_decode_f32", consts);
        let out = buf_zero_f32(&device, t * value_dim);
        let (conv_state, ssm) = (buf_f32(&device, &conv0), buf_f32(&device, &ssm0));
        let bufs = [
            &out,
            &qkv_buf,
            &z_buf,
            &a_buf,
            &b_buf,
            &w_buf,
            &conv_state,
            &ssm,
            &cu,
            &si,
            &fresh,
            &alog_buf,
            &dt_buf,
            &norm_buf,
        ];
        let grid = size(1, nk, t);
        if !common::dispatch_threadgroups(&device, &decode, &bufs, grid, size(1024, 1, 1)) {
            return;
        }

        let geometry = format!("nk {nk} nv {nv}");
        let close = |what: &str, got: &Buffer, want: &Buffer, n: usize| {
            let (got, want) = (read_f32(got, n), read_f32(want, n));
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * scale,
                    "{what}[{i}]: {g} vs {w}, {geometry}"
                );
            }
        };
        let n = conv0.len();
        assert_eq!(
            read_f32(&conv_state, n),
            read_f32(&chain_conv, n),
            "{geometry}"
        );
        close("ssm state", &ssm, &chain_ssm, ssm0.len());
        close("out", &out, &chain_out, t * value_dim);
        // The step moved both states and wrote every output: agreeing on untouched buffers
        // would pass the comparisons above by accident.
        assert_ne!(read_f32(&ssm, ssm0.len()), ssm0, "{geometry}");
        assert_ne!(read_f32(&conv_state, conv0.len()), conv0, "{geometry}");
        assert!(
            read_f32(&out, t * value_dim).iter().all(|v| *v != 0.0),
            "{geometry}"
        );
    }
}
