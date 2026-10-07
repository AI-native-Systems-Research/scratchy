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
use scratchy_target_metal::gdn_state::{CheckpointRows, GdnStart, GdnStep, RecordArea};
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
    drafts: u8,
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
            ConstantValue::uint(5, u32::from(drafts)),
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

/// `gdn_rms_norm_gated_f32` at the lowering's threadgroup — the smallest power of two covering
/// head_v — sums in the 256-thread order bit for bit (the larger tree's extra levels add zeros),
/// at head_v 128 and 16; a quarter-width threadgroup (four squares a thread, summed in turn) does
/// not.
#[test]
fn gdn_rms_norm_gated_sums_alike_at_every_covering_threadgroup() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    for (d, threads, narrower) in [(128usize, [128usize, 512], 32usize), (16, [16, 32], 4)] {
        let rows = 512usize;
        let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
        let (x, z, weight) = (shifted(rows * d, 7), shifted(rows * d, 3), shifted(d, 5));
        let key = PipelineKey::new(
            "gdn_rms_norm_gated",
            "gdn_rms_norm_gated_f32",
            vec![
                ConstantValue::uint(0, d as u32),
                ConstantValue::uint(1, rows as u32),
                ConstantValue::float(2, 1e-6),
            ],
        );
        let pipeline = baked_build(&cache, &key).expect("gdn_rms_norm_gated pipeline");
        let (x_buf, z_buf, w_buf) = (
            buf_f32(&device, &x),
            buf_f32(&device, &z),
            buf_f32(&device, &weight),
        );
        let run = |width: usize| {
            let out = buf_zero_f32(&device, rows * d);
            let tg = MTLSize {
                width,
                height: 1,
                depth: 1,
            };
            let grid = MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            };
            let bufs = [&out, &x_buf, &z_buf, &w_buf];
            common::dispatch_threadgroups(&device, &pipeline, &bufs, grid, tg).then(|| {
                read_f32(&out, rows * d)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            })
        };
        let Some(want) = run(256) else {
            return;
        };
        for width in threads {
            assert_eq!(
                run(width),
                Some(want.clone()),
                "head_v {d}, {width} threads"
            );
        }
        assert_ne!(run(narrower), Some(want), "head_v {d}, {narrower} threads");
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
            ConstantValue::uint(2, 0),
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
            ConstantValue::uint(2, 0),
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
        0,
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
        0,
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
        0,
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
        0,
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
                ConstantValue::uint(5, 0),
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
/// 8-lanes-per-row staging under this crate's contracts, two rows a thread, 32 a threadgroup)
/// — against the gating→scan chain over
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
                ConstantValue::uint(5, 0),
            ]
        };
        let pipe_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
                ConstantValue::uint(5, 0),
                ConstantValue::uint(6, 12), // TB
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
        let grid = size(value_dim / 32, 1, seqs);
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

/// `gdn_scan_pipelined_f32` in a drafting model's pool: slots of a state entry and two record
/// areas. A plain prefill (fresh, or from the slot) into slot 1 of two, and two sequences into
/// both slots, leave what `gdn_scan_simd` of the same drafts leaves — the entries to f32
/// rounding, slot 0's when untouched and every record area as they were.
#[test]
fn gdn_scan_pipelined_keeps_a_drafting_models_slots() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv) = (2usize, 6usize, 128usize, 128usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);
    let rows = CheckpointRows(2);
    let entry_len = nv * hv * hk;
    let slot_len = entry_len + 2 * (usize::from(rows.0) + 1) * (conv_dim + 2 * nv);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (a_log, dt_bias) = (fill(nv), shifted(nv, 5));
    let (alog_buf, dt_buf) = (buf_f32(&device, &a_log), buf_f32(&device, &dt_bias));
    let pool0 = shifted(2 * slot_len, 11);
    let scan_consts = vec![
        ConstantValue::uint(0, nk as u32),
        ConstantValue::uint(1, nv as u32),
        ConstantValue::uint(2, hk as u32),
        ConstantValue::uint(3, hv as u32),
        ConstantValue::float(4, scale),
        ConstantValue::uint(5, u32::from(rows.0)),
    ];
    let pipe_consts = [scan_consts.clone(), vec![ConstantValue::uint(6, 12)]].concat();
    let simd = baked_build(
        &cache,
        &PipelineKey::new("gdn_scan_varlen", "gdn_scan_simd_f32", scan_consts),
    )
    .expect("gdn_scan_simd_f32");
    let pipe = baked_build(
        &cache,
        &PipelineKey::new("gdn_scan_pipelined", "gdn_scan_pipelined_f32", pipe_consts),
    )
    .expect("gdn_scan_pipelined_f32");
    let plain = |start| {
        GdnStep {
            start,
            checkpoint_rows: CheckpointRows::NONE,
            records: RecordArea::Second,
        }
        .encode()
    };
    let size = |width, height, depth| MTLSize {
        width,
        height,
        depth,
    };
    for (cu, slots, starts) in [
        (vec![0, 25], vec![1], vec![GdnStart::Fresh]),
        (vec![0, 25], vec![1], vec![GdnStart::Slot]),
        (
            vec![0, 13, 26],
            vec![1, 0],
            vec![GdnStart::Fresh, GdnStart::Slot],
        ),
    ] {
        let t = *cu.last().expect("a sequence") as usize;
        let seqs = cu.len() - 1;
        let (conv, a, b) = (fill(t * conv_dim), shifted(t * nv, 3), shifted(t * nv, 7));
        let codes: Vec<u32> = starts.iter().map(|&s| plain(s)).collect();
        let (conv_buf, a_buf, b_buf) = (
            buf_f32(&device, &conv),
            buf_f32(&device, &a),
            buf_f32(&device, &b),
        );
        let (cu_buf, slot_buf, code_buf) = (
            buf_i32(&device, &cu),
            buf_i32(&device, &slots),
            buf_u32(&device, &codes),
        );
        let run = |kernel, grid, threads| {
            let (o, state) = (
                buf_zero_f32(&device, t * value_dim),
                buf_f32(&device, &pool0),
            );
            let bufs = [
                &o, &conv_buf, &a_buf, &b_buf, &state, &cu_buf, &slot_buf, &code_buf, &alog_buf,
                &dt_buf,
            ];
            common::dispatch_threadgroups(&device, kernel, &bufs, grid, threads)
                .then(|| (read_f32(&o, t * value_dim), read_f32(&state, pool0.len())))
        };
        let (Some((simd_o, simd_pool)), Some((pipe_o, pipe_pool))) = (
            run(&simd, size(1, value_dim / 4, seqs), size(32, 4, 1)),
            run(&pipe, size(value_dim / 32, 1, seqs), size(128, 1, 1)),
        ) else {
            return;
        };
        let what = format!("cu {cu:?}, slots {slots:?}, starts {starts:?}");
        let close = |part: &str, got: &[f32], want: &[f32]| {
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * scale,
                    "{what}: {part}[{i}]: {g} vs {w}"
                );
            }
        };
        close("o", &pipe_o, &simd_o);
        for slot in 0..2 {
            let entry = slot * slot_len..slot * slot_len + entry_len;
            let records = entry.end..(slot + 1) * slot_len;
            close(
                &format!("slot {slot} entry"),
                &pipe_pool[entry.clone()],
                &simd_pool[entry.clone()],
            );
            assert_eq!(
                pipe_pool[entry.clone()] != pool0[entry],
                slots.contains(&(slot as i32)),
                "{what}: slot {slot}'s entry moved only if a sequence runs in it"
            );
            assert_eq!(
                pipe_pool[records.clone()],
                pool0[records],
                "{what}: slot {slot}'s record areas"
            );
        }
    }
}

/// `gdn_decode_f32` — a decode token's conv, gating, scan and gated norm in one command — against
/// the three commands it replaces chained over the same inputs and state: `gdn_conv1d_varlen_f32`,
/// `gdn_scan_simd_f32`, `gdn_rms_norm_gated_f32`. Production head dims, one, two and three value
/// heads a key head; two sequences of a token each, a continued one and a fresh one, in permuted
/// state slots. The conv state, a copy of inputs, agrees exactly; the output and the scan state are
/// the chain's statements in its order, which the shader compiler contracts and reassociates per
/// kernel under fast math, so they agree to f32 rounding. A drafting model's pools too (2 drafts:
/// conv checkpoints and ssm record areas in every slot), on plain steps whose record area is the
/// second (code bit 16): the chain of the same drafts leaves what the one command leaves, the
/// checkpoints and record areas as they were.
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
    let geometries = [
        ((2usize, 4usize), [GdnStart::Slot, GdnStart::Fresh]),
        ((2, 6), [GdnStart::Fresh, GdnStart::Slot]),
        ((2, 2), [GdnStart::Slot, GdnStart::Slot]),
    ];
    for (((nk, nv), starts), drafts) in geometries
        .into_iter()
        .flat_map(|g| [0u8, 2].map(move |d| (g, d)))
    {
        let (key_dim, value_dim) = (nk * hk, nv * hv);
        let conv_dim = 2 * key_dim + value_dim;
        let (t, slots) = (2usize, 2usize);
        let rows = CheckpointRows(drafts);
        let conv_slot = conv_dim * (kernel - 1) * rows.conv_entries_per_slot();
        let ssm_slot = nv * hv * hk
            + match drafts {
                0 => 0,
                k => 2 * (usize::from(k) + 1) * (conv_dim + 2 * nv),
            };
        let records = match drafts {
            0 => RecordArea::First,
            _ => RecordArea::Second,
        };
        let fresh = starts.map(|start| {
            GdnStep {
                start,
                checkpoint_rows: CheckpointRows::NONE,
                records,
            }
            .encode()
        });
        let qkv = shifted(t * conv_dim, 7);
        let z = shifted(t * value_dim, 13);
        let (a, b) = (fill(t * nv), shifted(t * nv, 3));
        let (a_log, dt_bias, norm_w) = (fill(nv), shifted(nv, 5), shifted(hv, 17));
        let conv_w = shifted(conv_dim * kernel, 19);
        let conv0 = shifted(slots * conv_slot, 23);
        let ssm0 = shifted(slots * ssm_slot, 11);
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
                ConstantValue::uint(5, u32::from(drafts)),
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
            ConstantValue::uint(2, u32::from(drafts)),
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
            ConstantValue::uint(6, kernel as u32),
            ConstantValue::float(7, eps),
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

        let geometry = format!("nk {nk} nv {nv} drafts {drafts}");
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

/// Compares a kernel's rows against the reference's.
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-4, "{what}[{i}] metal={g} ref={w}");
    }
}

/// Inputs for a verify step's rejected drafts: unlike anything `fill` produces.
fn rejected(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37).cos() * 0.5).collect()
}

/// A verify step's checkpoints, conv ring: 3 tokens, then a verify step of the next token, a kept
/// draft and two rejected drafts, then one token resuming from the kept row's checkpoint — equals
/// the sequence that never saw the rejected drafts. In slot 1 of a 2-slot pool keeping 3
/// checkpoints, so the entries are offset and slot 0's must stay untouched.
#[test]
fn gdn_conv1d_varlen_resumes_from_checkpoint() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (conv_dim, kernel) = (32usize, 4usize);
    let entry_len = conv_dim * (kernel - 1);
    let rows = CheckpointRows(3);
    let slot_entries = rows.conv_entries_per_slot();
    let t_full = 6usize;
    let x_full = fill(t_full * conv_dim);
    let w = fill(conv_dim * kernel);
    let out_full = conv1d_ref(&x_full, &w, conv_dim, kernel, t_full);
    let x_verify = [
        &x_full[3 * conv_dim..5 * conv_dim],
        &rejected(2 * conv_dim)[..],
    ]
    .concat();

    let key = PipelineKey::new(
        "gdn_conv1d_varlen",
        "gdn_conv1d_varlen_f32",
        vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
            ConstantValue::uint(2, u32::from(rows.0)),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");
    let w_buf = buf_f32(&device, &w);
    let state_buf = buf_zero_f32(&device, 2 * slot_entries * entry_len);
    let run = |x: &[f32], start: GdnStart, checkpoint_rows: CheckpointRows| -> Option<Vec<f32>> {
        let num_tokens = x.len() / conv_dim;
        let out_buf = buf_zero_f32(&device, x.len());
        let x_buf = buf_f32(&device, x);
        let cu = buf_i32(&device, &[0, num_tokens as i32]);
        let si = buf_i32(&device, &[1]);
        let step = buf_u32(
            &device,
            &[GdnStep {
                start,
                checkpoint_rows,
                records: RecordArea::First,
            }
            .encode()],
        );
        common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[&out_buf, &x_buf, &w_buf, &state_buf, &cu, &si, &step],
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: conv_dim,
                depth: 1,
            },
        )
        .then(|| read_f32(&out_buf, x.len()))
    };

    let Some(_) = run(
        &x_full[..3 * conv_dim],
        GdnStart::Fresh,
        CheckpointRows::NONE,
    ) else {
        return;
    };
    let Some(verify) = run(&x_verify, GdnStart::Slot, rows) else {
        return;
    };
    assert_close(
        &verify[..2 * conv_dim],
        &out_full[3 * conv_dim..5 * conv_dim],
        "verify kept rows",
    );
    let Some(next) = run(
        &x_full[5 * conv_dim..],
        GdnStart::Checkpoint(1),
        CheckpointRows::NONE,
    ) else {
        return;
    };
    assert_close(&next, &out_full[5 * conv_dim..], "resumed row");
    let slot0 = read_f32(&state_buf, slot_entries * entry_len);
    assert!(
        slot0.iter().all(|&v| v == 0.0),
        "slot 0's entries were written"
    );
}

/// A verify step whose drafts are all kept: the slot holds the state it started from, so the next
/// step replays all its rows (`Checkpoint(drafts)`) — bit for bit the rows run without drafts.
#[test]
fn gdn_scan_varlen_replays_a_fully_kept_verify_step() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv) = (2usize, 4usize, 16usize, 16usize);
    let conv_dim = 2 * nk * hk + nv * hv;
    let entry_len = nv * hv * hk;
    let rows = CheckpointRows(3);
    let slot_len = entry_len + 2 * (usize::from(rows.0) + 1) * (conv_dim + 2 * nv);
    // Rows 0..3 the prompt, 3..7 the verify step (its token and three kept drafts), 7 the next.
    let t_full = 8usize;
    let (conv, g, beta) = (
        fill(t_full * conv_dim),
        fill(t_full * nv),
        fill(t_full * nv),
    );
    let at = |x: &[f32], w: usize, r: std::ops::Range<usize>| x[r.start * w..r.end * w].to_vec();
    let entry = [1];
    let code = |start, checkpoint_rows, records| {
        [GdnStep {
            start,
            checkpoint_rows,
            records,
        }
        .encode()]
    };
    let run = |state: &Buffer, r: std::ops::Range<usize>, step: [u32; 1]| {
        let n = r.len();
        dispatch_scan(
            &device,
            &cache,
            &at(&conv, conv_dim, r.clone()),
            &at(&g, nv, r.clone()),
            &at(&beta, nv, r),
            state,
            &[0, n as i32],
            &entry,
            &step,
            nk,
            nv,
            hk,
            hv,
            n,
            rows.0,
        )
    };
    let none = CheckpointRows::NONE;
    let (r0, r1) = (RecordArea::First, RecordArea::Second);
    let state = buf_zero_f32(&device, 2 * slot_len);
    let (Some(_), Some(_), Some(o_next)) = (
        run(&state, 0..3, code(GdnStart::Fresh, none, r0)),
        run(&state, 3..7, code(GdnStart::Slot, rows, r0)),
        run(&state, 7..8, code(GdnStart::Checkpoint(3), none, r1)),
    ) else {
        return;
    };
    let plain = buf_zero_f32(&device, 2 * slot_len);
    let (Some(_), Some(_), Some(o_plain)) = (
        run(&plain, 0..3, code(GdnStart::Fresh, none, r0)),
        run(&plain, 3..7, code(GdnStart::Slot, none, r0)),
        run(&plain, 7..8, code(GdnStart::Slot, none, r0)),
    ) else {
        return;
    };
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&o_next),
        bits(&o_plain),
        "the replayed rows' state differs"
    );
    // Not the replay: the slot as it stands is the verify step's start state.
    let Some(o_slot) = run(&state, 7..8, code(GdnStart::Slot, none, r1)) else {
        return;
    };
    assert_ne!(
        bits(&o_slot),
        bits(&o_plain),
        "the verify step wrote its last row's state"
    );
}

/// [`gdn_conv1d_varlen_resumes_from_checkpoint`] for the recurrent (ssm) state, whose verify
/// step records its draft rows and whose next step replays the kept ones from its base entry
/// (`gdn_state` module docs): the resumed row is bit for bit the one a step that never saw the
/// rejected drafts computes. Each step's code carries the slot's entry and record area as the
/// slot allocator tracks them ([`GdnStep::after`]).
#[test]
fn gdn_scan_varlen_resumes_from_checkpoint() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (nk, nv, hk, hv) = (2usize, 4usize, 16usize, 16usize);
    let key_dim = nk * hk;
    let value_dim = nv * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (hk as f32).powf(-0.5);
    let entry_len = nv * hv * hk;
    let rows = CheckpointRows(3);
    let slot_len = entry_len + 2 * (usize::from(rows.0) + 1) * (conv_dim + 2 * nv);
    let t_full = 6usize;

    let conv_full = fill(t_full * conv_dim);
    let g_full = fill(t_full * nv);
    let beta_full = fill(t_full * nv);
    let (q, k, v) = split_conv(&conv_full, nk, nv, hk, hv, t_full);
    let o_full = recurrent_ref(
        &q, &k, &v, &g_full, &beta_full, nk, nv, hk, hv, t_full, scale,
    );
    // Rows `3..5` of the kept sequence, then two rejected drafts.
    let verify = |full: &[f32], width: usize| {
        [&full[3 * width..5 * width], &rejected(2 * width)[..]].concat()
    };

    // Slot 1 of a two-slot pool; its steps' codes chained as the allocator chains them.
    let entry = [1];
    let pool = || buf_zero_f32(&device, 2 * slot_len);
    let slot = std::cell::Cell::new(RecordArea::First);
    let step = |start, checkpoint_rows| {
        let records = slot.get();
        let step = GdnStep {
            start,
            checkpoint_rows,
            records,
        };
        slot.set(step.after());
        [step.encode()]
    };
    let run = |state: &Buffer, conv: &[f32], g: &[f32], beta: &[f32], step: [u32; 1]| {
        let num_tokens = g.len() / nv;
        dispatch_scan(
            &device,
            &cache,
            conv,
            g,
            beta,
            state,
            &[0, num_tokens as i32],
            &entry,
            &step,
            nk,
            nv,
            hk,
            hv,
            num_tokens,
            rows.0,
        )
    };

    let state = pool();
    let rows_of = |full: &[f32], width: usize, r: std::ops::Range<usize>| {
        full[r.start * width..r.end * width].to_vec()
    };
    let prompt = |state: &Buffer, fresh| {
        run(
            state,
            &rows_of(&conv_full, conv_dim, 0..3),
            &rows_of(&g_full, nv, 0..3),
            &rows_of(&beta_full, nv, 0..3),
            fresh,
        )
    };
    let Some(_) = prompt(&state, step(GdnStart::Fresh, CheckpointRows::NONE)) else {
        return;
    };
    let Some(o_verify) = run(
        &state,
        &verify(&conv_full, conv_dim),
        &verify(&g_full, nv),
        &verify(&beta_full, nv),
        step(GdnStart::Slot, rows),
    ) else {
        return;
    };
    assert_close(
        &o_verify[..2 * value_dim],
        &o_full[3 * value_dim..5 * value_dim],
        "verify kept rows",
    );
    let next = |state: &Buffer, code| {
        run(
            state,
            &conv_full[5 * conv_dim..],
            &g_full[5 * nv..],
            &beta_full[5 * nv..],
            code,
        )
    };
    let Some(o_next) = next(&state, step(GdnStart::Checkpoint(1), CheckpointRows::NONE)) else {
        return;
    };
    assert_close(&o_next, &o_full[5 * value_dim..], "resumed row");
    let slot0 = read_f32(&state, slot_len);
    assert!(
        slot0.iter().all(|&v| v == 0.0),
        "slot 0's entries were written"
    );

    // The same rows without drafts: prompt, the kept rows, the next row.
    let plain_code = |start| {
        [GdnStep {
            start,
            checkpoint_rows: CheckpointRows::NONE,
            records: RecordArea::First,
        }
        .encode()]
    };
    let plain = pool();
    let kept = |state: &Buffer| {
        run(
            state,
            &rows_of(&conv_full, conv_dim, 3..5),
            &rows_of(&g_full, nv, 3..5),
            &rows_of(&beta_full, nv, 3..5),
            plain_code(GdnStart::Slot),
        )
    };
    let (Some(_), Some(_), Some(o_plain)) = (
        prompt(&plain, plain_code(GdnStart::Fresh)),
        kept(&plain),
        next(&plain, plain_code(GdnStart::Slot)),
    ) else {
        return;
    };
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&o_next),
        bits(&o_plain),
        "the replayed rows' state differs from the rows run without drafts"
    );
}

/// [`gdn_scan_varlen_resumes_from_checkpoint`] for `gdn_scan_simd`, whose gating is its own: its
/// verify step's records replay bit for bit as its rows run without drafts. A slot's steps switch
/// scans (a bucket of at most 8 rows runs this one, a bigger one `gdn_scan_varlen`), so each scan
/// replays the other's records: a verify step on either, the next step on the other, agree with
/// the rows run without drafts to the two scans' rounding.
#[test]
fn gdn_scan_simd_resumes_from_checkpoint() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Scan {
        Simd,
        Serial,
    }
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (nk, nv, hk, hv) = (2usize, 4usize, 32usize, 8usize);
    let conv_dim = 2 * nk * hk + nv * hv;
    let value_dim = nv * hv;
    let entry_len = nv * hv * hk;
    let scale = (hk as f32).powf(-0.5);
    let rows = CheckpointRows(3);
    let slot_len = entry_len + 2 * (usize::from(rows.0) + 1) * (conv_dim + 2 * nv);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (a_log, dt_bias) = (fill(nv), shifted(nv, 5));
    // Rows 0..3 the prompt, 3..5 the kept ones, 5 the next; a verify step's rejected drafts follow
    // its kept rows.
    let (conv, a, b) = (fill(6 * conv_dim), shifted(6 * nv, 3), shifted(6 * nv, 7));
    let at = |x: &[f32], width: usize, r: std::ops::Range<usize>| {
        x[r.start * width..r.end * width].to_vec()
    };
    let verify = |x: &[f32], width: usize| [at(x, width, 3..5), rejected(2 * width)].concat();

    let simd = baked_build(
        &cache,
        &PipelineKey::new(
            "gdn_scan_varlen",
            "gdn_scan_simd_f32",
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
                ConstantValue::uint(5, u32::from(rows.0)),
            ],
        ),
    )
    .expect("gdn_scan_simd_f32");
    let (alog_buf, dt_buf) = (buf_f32(&device, &a_log), buf_f32(&device, &dt_bias));
    // Slot 1 of a two-slot pool.
    let entry = [1];
    let entry_buf = buf_i32(&device, &entry);
    // `gdn_gating`'s g and beta, for the serial scan.
    let gating = |a: &[f32], b: &[f32]| {
        let g: Vec<f32> = (a.iter().enumerate())
            .map(|(i, &a)| {
                let x = a + dt_bias[i % nv];
                let sp = if x <= 20.0 { (1.0 + x.exp()).ln() } else { x };
                -a_log[i % nv].exp() * sp
            })
            .collect();
        let beta: Vec<f32> = b.iter().map(|&b| 1.0 / (1.0 + (-b).exp())).collect();
        (g, beta)
    };
    let run = |scan: Scan, state: &Buffer, conv: &[f32], a: &[f32], b: &[f32], code: [u32; 1]| {
        let t = a.len() / nv;
        match scan {
            Scan::Serial => {
                let (g, beta) = gating(a, b);
                let cu = [0, t as i32];
                dispatch_scan(
                    &device, &cache, conv, &g, &beta, state, &cu, &entry, &code, nk, nv, hk, hv, t,
                    rows.0,
                )
            }
            Scan::Simd => {
                let o = buf_zero_f32(&device, t * value_dim);
                let (conv_buf, a_buf, b_buf) = (
                    buf_f32(&device, conv),
                    buf_f32(&device, a),
                    buf_f32(&device, b),
                );
                let (cu_buf, code_buf) =
                    (buf_i32(&device, &[0, t as i32]), buf_u32(&device, &code));
                let bufs = [
                    &o, &conv_buf, &a_buf, &b_buf, state, &cu_buf, &entry_buf, &code_buf,
                    &alog_buf, &dt_buf,
                ];
                let grid = MTLSize {
                    width: 1,
                    height: value_dim / 4,
                    depth: 1,
                };
                let threads = MTLSize {
                    width: 32,
                    height: 4,
                    depth: 1,
                };
                common::dispatch_threadgroups(&device, &simd, &bufs, grid, threads)
                    .then(|| read_f32(&o, t * value_dim))
            }
        }
    };
    let pool = || buf_zero_f32(&device, 2 * slot_len);
    let prompt = |scan, state: &Buffer, code| {
        run(
            scan,
            state,
            &at(&conv, conv_dim, 0..3),
            &at(&a, nv, 0..3),
            &at(&b, nv, 0..3),
            code,
        )
    };
    let next = |scan, state: &Buffer, code| {
        run(
            scan,
            state,
            &at(&conv, conv_dim, 5..6),
            &at(&a, nv, 5..6),
            &at(&b, nv, 5..6),
            code,
        )
    };
    let plain_code = |start| {
        [GdnStep {
            start,
            checkpoint_rows: CheckpointRows::NONE,
            records: RecordArea::First,
        }
        .encode()]
    };

    // The simd scan's rows without drafts: prompt, the kept rows, the next row.
    let plain = pool();
    let (Some(_), Some(o_kept), Some(o_plain)) = (
        prompt(Scan::Simd, &plain, plain_code(GdnStart::Fresh)),
        run(
            Scan::Simd,
            &plain,
            &at(&conv, conv_dim, 3..5),
            &at(&a, nv, 3..5),
            &at(&b, nv, 3..5),
            plain_code(GdnStart::Slot),
        ),
        next(Scan::Simd, &plain, plain_code(GdnStart::Slot)),
    ) else {
        return;
    };
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let close = |what: &str, got: &[f32], want: &[f32]| {
        let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * scale, "{what}[{i}]: {g} vs {w}");
        }
    };

    // Prompt and a verify step on `verifier`, then the next row on `resumer` from `start`, the
    // slot's steps' codes chained as the allocator chains them.
    let resume = |verifier, resumer, start| {
        let slot = std::cell::Cell::new(RecordArea::First);
        let step = |start, checkpoint_rows| {
            let records = slot.get();
            let step = GdnStep {
                start,
                checkpoint_rows,
                records,
            };
            slot.set(step.after());
            [step.encode()]
        };
        let state = pool();
        prompt(
            verifier,
            &state,
            step(GdnStart::Fresh, CheckpointRows::NONE),
        )?;
        let o_verify = run(
            verifier,
            &state,
            &verify(&conv, conv_dim),
            &verify(&a, nv),
            &verify(&b, nv),
            step(GdnStart::Slot, rows),
        )?;
        let o_next = next(resumer, &state, step(start, CheckpointRows::NONE))?;
        Some((o_verify[..2 * value_dim].to_vec(), o_next))
    };

    // Not the replay: resuming from the slot as it stands — the state the verify step started
    // from, without its kept rows — is not the row run without drafts.
    let Some((_, o_tip)) = resume(Scan::Simd, Scan::Simd, GdnStart::Slot) else {
        return;
    };
    assert_ne!(
        bits(&o_tip),
        bits(&o_plain),
        "the rejected drafts left the state unchanged"
    );

    for (verifier, resumer) in [
        (Scan::Simd, Scan::Simd),
        (Scan::Simd, Scan::Serial),
        (Scan::Serial, Scan::Simd),
    ] {
        let what = format!("verify {verifier:?}, resume {resumer:?}");
        let Some((kept, o_next)) = resume(verifier, resumer, GdnStart::Checkpoint(1)) else {
            return;
        };
        match (verifier, resumer) {
            (Scan::Simd, Scan::Simd) => {
                assert_eq!(bits(&kept), bits(&o_kept), "{what}: kept rows");
                assert_eq!(
                    bits(&o_next),
                    bits(&o_plain),
                    "{what}: the replayed rows' state differs from the rows run without drafts"
                );
            }
            _ => {
                close(&format!("{what}: kept rows"), &kept, &o_kept);
                close(&format!("{what}: resumed row"), &o_next, &o_plain);
            }
        }
    }
}
