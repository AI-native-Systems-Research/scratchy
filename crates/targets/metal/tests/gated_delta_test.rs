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

use common::Dtype;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::{bake, baked_build};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::gdn_state::{
    CheckpointRows, DraftLane, DraftLanes, GDN_SAVES, GdnSave, GdnSeqSlots, GdnSlot, GdnStart,
    GdnStateDims, GdnStep, RecordArea,
};
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

/// The ssm state's element dtype the GDN kernels read and write (`GDN_SSM_STATE_DTYPE`): uploaded
/// from and read back to f32 through f16 — the store's rounding is the state's per-step
/// quantization.
fn buf_f16(device: &Device, data: &[f32]) -> Buffer {
    let half: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let bytes = std::mem::size_of_val(half.as_slice()).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            half.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(half.as_slice()),
        );
    }
    buf
}

fn buf_zero_f16(device: &Device, n: usize) -> Buffer {
    let bytes = (n * 2).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes) };
    buf
}

fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const half::f16, n) }
        .iter()
        .map(|h| h.to_f32())
        .collect()
}

/// A state fixture the kernels' f16 store reproduces bit for bit.
fn q16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect()
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

/// `state_indices` as the kernels read it: a [`GdnSeqSlots`] record a sequence.
fn buf_seqs(device: &Device, seqs: &[GdnSeqSlots]) -> Buffer {
    let words: Vec<i32> = seqs.iter().flat_map(|s| s.words()).collect();
    buf_i32(device, &words)
}

/// `state_indices` of plain steps on `slots`, a sequence each.
fn buf_slots(device: &Device, slots: &[i32]) -> Buffer {
    let seqs: Vec<GdnSeqSlots> = slots
        .iter()
        .map(|&s| GdnSeqSlots::plain(GdnSlot::new(s as usize)))
        .collect();
    buf_seqs(device, &seqs)
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
/// `slot_after_rows` names the rows whose end is a step boundary the state crosses through the
/// pool's f16 slot — a store, then the next step's load — so the reference rounds its state
/// through f16 there, as the state the kernel carries on.
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
    slot_after_rows: &[usize],
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
        if slot_after_rows.contains(&ti) {
            for s in state.iter_mut() {
                *s = half::f16::from_f32(*s).to_f32();
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
/// `state_buf` is mutated in place (shared across calls for continuity tests); so is a drafting
/// model's `(drafts, lanes)` buffer of draft lanes (buffer 9).
#[allow(clippy::too_many_arguments)]
fn dispatch_scan(
    device: &Device,
    cache: &SpecializedPipelineCache,
    conv_out: &[f32],
    g: &[f32],
    beta: &[f32],
    state_buf: &Buffer,
    cu: &[i32],
    si: &[GdnSeqSlots],
    fresh: &[u32],
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
    num_tokens: usize,
    drafts: Option<(u8, &Buffer)>,
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
            ConstantValue::uint(8, drafts.map_or(0, |(k, _)| u32::from(k))),
        ],
    );
    let pipeline = baked_build(cache, &key).expect("gdn_scan_varlen pipeline");

    let o_buf = buf_zero_f32(device, num_tokens * value_dim);
    let conv_buf = buf_f32(device, conv_out);
    let g_buf = buf_f32(device, g);
    let beta_buf = buf_f32(device, beta);
    let cu_buf = buf_i32(device, cu);
    let si_buf = buf_seqs(device, si);
    let fresh_buf = buf_u32(device, fresh);
    // Mamba-2's `D` (buffer 8), unread by the delta rule.
    let unread = buf_zero_f32(device, 1);

    let num_seqs = cu.len() - 1;
    let tgx = hv.min(256);
    let mut bufs = vec![
        &o_buf, &conv_buf, &g_buf, &beta_buf, state_buf, &cu_buf, &si_buf, &fresh_buf,
    ];
    if let Some((_, lanes)) = drafts {
        bufs.extend([&unread, lanes]);
    }
    if !common::dispatch_threadgroups(
        device,
        &pipeline,
        &bufs,
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

/// `gdn_rms_norm_gated_simd` — the simdgroup form the lowering picks for a power-of-two
/// width in [32, 256] — is BIT-IDENTICAL to the threadgroup form at that width's covering
/// threadgroup: the lane layout (a simdgroup a row, `width / 32` strided elements a lane)
/// reproduces the covering threadgroup's reduction tree exactly, its top levels as register
/// adds and its bottom five as simd shuffles. Both rows-a-threadgroup layouts the lowering
/// chooses (four when a token's rows divide by four, else one), and the Mamba-2
/// gate-before-norm form with its strided `z` and group-spanning weight. Every dtype the
/// lowering lowers the form for — the store loop keeps the threadgroup form's runtime-strided
/// shape, a codegen workaround that can behave differently per instantiation — plus f32,
/// whose output bits a ULP of sum drift shows in directly. The negative control is the
/// threadgroup form at a QUARTER-width threadgroup (four squares a thread, summed in turn —
/// a different association), which differs at f32: the bit equality can actually fail, where
/// a 16-bit output rounds a ULP of sum drift away.
#[test]
fn gdn_rms_norm_gated_simd_is_bit_identical_to_the_threadgroup_form() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    // (width, groups a token, rows-a-threadgroup layouts to try): the GDN form (one value head
    // a row) at every dispatchable width, then the Mamba-2 form (gate before norm, `z` a
    // strided column slice, weight spanning the token's rows) at the parity-tiny lowering's
    // two-groups-a-token shape and at four.
    let cases: &[(u32, u32, &[u32])] = &[
        (32, 1, &[4, 1]),
        (64, 1, &[4, 1]),
        (128, 1, &[4, 1]),
        (256, 1, &[4, 1]),
        (64, 2, &[1]),
        (128, 4, &[4]),
    ];
    let rows = 512usize;
    let eps = 1e-6f32;
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    for dt in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
        let (threadgroup_sym, simd_sym) = (
            dt.symbol("gdn_rms_norm_gated"),
            dt.symbol("gdn_rms_norm_gated_simd"),
        );
        for &(d, groups, rpts) in cases {
            let mamba = groups > 1;
            // The Mamba-2 `z`: `rows / groups` in-projection rows, each `stride` wide, the norm's
            // columns a group slice of one — padding the stride catches a stride bug.
            let stride = groups * d + 8;
            let x = shifted(rows * d as usize, 7);
            let z = if mamba {
                shifted((rows / groups as usize) * stride as usize, 3)
            } else {
                shifted(rows * d as usize, 3)
            };
            let weight = shifted(groups as usize * d as usize, 5);
            let base = vec![
                ConstantValue::uint(0, d),
                ConstantValue::uint(1, rows as u32),
                ConstantValue::float(2, eps),
            ];
            let mamba_consts = || {
                base.clone()
                    .into_iter()
                    .chain([
                        ConstantValue::uint(3, stride),
                        ConstantValue::uint(4, groups),
                        ConstantValue::boolean(5, true),
                    ])
                    .collect::<Vec<_>>()
            };
            let (x_buf, z_buf, w_buf) = (
                buf_f32(&device, &x),
                // `z` is read at `T`'s precision: both forms see this dtype's rounding of it.
                common::shared_bytes(&device, &dt.bytes(&z)),
                buf_f32(&device, &weight),
            );
            let run =
                |function: &'static str, constants: Vec<ConstantValue>, grid: usize, tg: usize| {
                    let key = PipelineKey::new("gdn_rms_norm_gated", function, constants);
                    let pipeline = baked_build(&cache, &key).expect("norm pipeline");
                    // f32-sized whatever the dtype: 16-bit outputs just use half of it.
                    let out = buf_zero_f32(&device, rows * d as usize);
                    let bufs = [&out, &x_buf, &z_buf, &w_buf];
                    common::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &bufs,
                        MTLSize {
                            width: grid,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: tg,
                            height: 1,
                            depth: 1,
                        },
                    )
                    .then(|| dt.bits(&out, rows * d as usize))
                };
            // The threadgroup form at the covering threadgroup is the oracle.
            let consts = if mamba { mamba_consts() } else { base.clone() };
            let Some(want) = run(
                threadgroup_sym,
                consts.clone(),
                rows,
                d.next_power_of_two() as usize,
            ) else {
                return;
            };
            // The control: a quarter-width threadgroup sums four squares a thread in turn —
            // a different association — and must NOT match (at f32, whose bits show it).
            if dt == Dtype::F32 {
                assert_ne!(
                    run(threadgroup_sym, consts, rows, d as usize / 4),
                    Some(want.clone()),
                    "the control itself: width {d}, groups {groups} — the quarter-width \
                     threadgroup sums like the covering one after all"
                );
            }
            for &rpt in rpts {
                let mut simd_consts = if mamba { mamba_consts() } else { base.clone() };
                simd_consts.push(ConstantValue::uint(6, rpt));
                assert_eq!(
                    run(
                        simd_sym,
                        simd_consts,
                        rows / rpt as usize,
                        32 * rpt as usize
                    ),
                    Some(want.clone()),
                    "width {d}, groups {groups}, {rpt} rows a threadgroup"
                );
                eprintln!(
                    "gdn norm bit-identity: {} width {d}, groups {groups}, {rpt} rows a tg: ok",
                    dt.tag()
                );
            }
        }
    }
}

/// The simdgroup form's pipeline key at width `d` (512 rows, eps 1e-6, 4 rows a
/// threadgroup) — the key both bake tests below share, so the refusal is the width
/// and nothing else.
fn gdn_rms_norm_gated_simd_key(d: u32) -> PipelineKey {
    PipelineKey::new(
        "gdn_rms_norm_gated",
        "gdn_rms_norm_gated_simd_f32",
        vec![
            ConstantValue::uint(0, d),
            ConstantValue::uint(1, 512),
            ConstantValue::float(2, 1e-6),
            ConstantValue::uint(6, 4),
        ],
    )
}

/// The simdgroup form bakes at an eligible width (64 — a power of two in [32, 256],
/// rows a threadgroup set): the refusal test's own key with only the width changed,
/// so a refusal there is the width's `static_assert` and not a shader that fails to
/// compile. `bake` only runs the offline compilers — no device — so this runs wherever
/// the shaders build, CI's macOS runners included.
#[test]
fn gdn_rms_norm_gated_simd_bakes_an_eligible_width() {
    bake(&[gdn_rms_norm_gated_simd_key(64)]);
}

/// The simdgroup form's layout contract is a BAKE REFUSAL, not a convention: a key with a
/// width the lane layout can't reduce (here 48 — not a power of two) fails the bake's
/// `xcrun metal` on the kernel's `static_assert`, instead of silently summing the wrong
/// elements. The eligible-width bake above proves the same key compiles, so only the
/// width refuses here. `bake` only runs the offline compilers — no device — so this runs
/// wherever the shaders build, CI's macOS runners included.
#[test]
#[should_panic(expected = "bake: `xcrun metal` failed")]
fn gdn_rms_norm_gated_simd_refuses_an_ineligible_width() {
    bake(&[gdn_rms_norm_gated_simd_key(48)]);
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
            ConstantValue::uint(5, 0),
            // A 3-token block — the lock's minimum (the seed window is 3 entries): two blocks
            // for the 6 rows, the second seeding its window from x and block 0 alone writing
            // the new ring.
            ConstantValue::uint(6, 3),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");

    let out_buf = buf_zero_f32(&device, t * conv_dim);
    let x_buf = buf_f32(&device, &x);
    let w_buf = buf_f32(&device, &w);
    let state_buf = buf_zero_f32(&device, conv_dim * state_len); // 1 slot
    let cu = buf_i32(&device, &[0, t as i32]);
    let si = buf_slots(&device, &[0]);
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
            depth: t.div_ceil(3),
        }, // x = num_seqs, z = token blocks
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
            ConstantValue::uint(5, 0),
            // A 3-token block — the lock's minimum (the seed window is 3 entries): the
            // 6-token run walks two blocks, the 1-token continuation one.
            ConstantValue::uint(6, 3),
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
        let si = buf_slots(&device, &[0]);
        let fresh = buf_u32(&device, &[is_fresh]);
        if !common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[&out_buf, &x_buf, &w_buf, &state_buf, &cu, &si, &fresh],
            MTLSize {
                width: 1,
                height: groups_y,
                depth: num_tokens.div_ceil(3),
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
    let o_ref = recurrent_ref(&q, &k, &v, &g, &beta, nk, nv, hk, hv, t, scale, &[]);

    let state_buf = buf_zero_f16(&device, nv * hv * hk); // 1 slot
    let Some(o) = dispatch_scan(
        &device,
        &cache,
        &conv_out,
        &g,
        &beta,
        &state_buf,
        &[0, t as i32],
        &[GdnSeqSlots::plain(GdnSlot::new(0))],
        &[1],
        nk,
        nv,
        hk,
        hv,
        t,
        None,
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
    let o_ref = recurrent_ref(&q, &k, &v, &g, &beta, nk, nv, hk, hv, t, scale, &[]);

    let state_buf = buf_zero_f16(&device, nv * hv * hk);
    let Some(o) = dispatch_scan(
        &device,
        &cache,
        &conv_out,
        &g,
        &beta,
        &state_buf,
        &[0, t as i32],
        &[GdnSeqSlots::plain(GdnSlot::new(0))],
        &[1],
        nk,
        nv,
        hk,
        hv,
        t,
        None,
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
        &q,
        &k,
        &v,
        &g_full,
        &beta_full,
        nk,
        nv,
        hk,
        hv,
        t_full,
        scale,
        &[4],
    );

    let state_buf = buf_zero_f16(&device, nv * hv * hk);

    // Run A: first 5 tokens, fresh → seeds ssm_state.
    let Some(_a) = dispatch_scan(
        &device,
        &cache,
        &conv_full[..5 * conv_dim],
        &g_full[..5 * nv],
        &beta_full[..5 * nv],
        &state_buf,
        &[0, 5],
        &[GdnSeqSlots::plain(GdnSlot::new(0))],
        &[1],
        nk,
        nv,
        hk,
        hv,
        5,
        None,
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
        &[GdnSeqSlots::plain(GdnSlot::new(0))],
        &[0],
        nk,
        nv,
        hk,
        hv,
        1,
        None,
    ) else {
        return;
    };

    let ref_last = &o_full[5 * value_dim..];
    // Run A stores its last row's state to the f16 slot and run B loads it back, so the two
    // carry a state that agrees to the entry's rounding — a half ULP where the kernel's and the
    // reference's registers straddle one.
    for c in 0..value_dim {
        assert!(
            (b[c] - ref_last[c]).abs() < 1e-3,
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
/// serially: the outputs agree to f32 rounding of those sums, the state left behind to its f16
/// store's rounding.
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
                ConstantValue::uint(8, 0),
            ]
        };
        let size = |width, height, depth| MTLSize {
            width,
            height,
            depth,
        };
        let (cu_buf, si_buf) = (buf_i32(&device, &cu), buf_slots(&device, &si));
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
        let chain_state = buf_f16(&device, &state0);
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
        let simd_state = buf_f16(&device, &state0);
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
        let close = |what: &str, got: Vec<f32>, want: Vec<f32>, tol: f32| {
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= tol * scale,
                    "{what}[{i}]: {g} vs {w}, cu {cu:?}"
                );
            }
        };
        let rows = t * value_dim;
        close("o", read_f32(&simd_o, rows), read_f32(&chain_o, rows), 1e-5);
        let s = state0.len();
        // The state entries are stored f16: the two scans' f32 registers agree to f32 rounding,
        // and the store rounds each to the same half — except where the two straddle a rounding
        // boundary, a half ULP apart.
        close(
            "state",
            read_f16(&simd_state, s),
            read_f16(&chain_state, s),
            1e-3,
        );
        // The state moved: a scan that left it would pass the comparison above only by accident.
        assert_ne!(read_f16(&simd_state, s), q16(&state0), "cu {cu:?}");
    }
}

/// `gdn_scan_pipelined_f32` — the prefill scan (gdn_scan_pipelined.metal, omlx Kernel P's
/// 8-lanes-per-row staging under this crate's contracts, two rows a thread, 32 a threadgroup)
/// — against the gating→scan chain over
/// the same inputs and state: outputs to f32 rounding (the butterfly's summation order is
/// simd_sum's), the state left behind to its f16 store's rounding. Mixed continued/fresh varlen
/// batches including a decode-shaped 1-token continuation; block-tail lengths exercise the
/// partial block.
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
                ConstantValue::uint(8, 0),
            ]
        };
        let pipe_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
                ConstantValue::uint(10, 12), // TB
            ]
        };
        let size = |width, height, depth| MTLSize {
            width,
            height,
            depth,
        };
        let (cu_buf, si_buf) = (buf_i32(&device, &cu), buf_slots(&device, &si));
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
        let chain_state = buf_f16(&device, &state0);
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
        let pipe_state = buf_f16(&device, &state0);
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

        let close = |what: &str, got: Vec<f32>, want: Vec<f32>, tol: f32| {
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= tol * scale,
                    "{what}[{i}]: {g} vs {w}, cu {cu:?}"
                );
            }
        };
        let rows = t * value_dim;
        close("o", read_f32(&pipe_o, rows), read_f32(&chain_o, rows), 1e-5);
        let s = state0.len();
        // The state entries are stored f16: the two scans' f32 registers agree to f32 rounding,
        // and the store rounds each to the same half — except where the two straddle a rounding
        // boundary, a half ULP apart.
        close(
            "state",
            read_f16(&pipe_state, s),
            read_f16(&chain_state, s),
            1e-3,
        );
        assert_ne!(read_f16(&pipe_state, s), q16(&state0), "cu {cu:?}");
    }
}

/// `gdn_scan_pipelined_f32` in a drafting model's pool: slots of a state entry, its draft lanes
/// apart. A plain prefill (fresh, or from the slot) into slot 1 of two, and two sequences into
/// both slots, leave what `gdn_scan_simd` of the same drafts leaves — the entries to f32
/// rounding, slot 0's when untouched — and the simd scan leaves every lane as it was.
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
    let lane_len = 2 * rows.rows() * (conv_dim + 2 * nv);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (a_log, dt_bias) = (fill(nv), shifted(nv, 5));
    let (alog_buf, dt_buf) = (buf_f32(&device, &a_log), buf_f32(&device, &dt_bias));
    let pool0 = q16(&shifted(2 * entry_len, 11));
    let lanes0 = shifted(2 * lane_len, 13);
    let scan_consts = vec![
        ConstantValue::uint(0, nk as u32),
        ConstantValue::uint(1, nv as u32),
        ConstantValue::uint(2, hk as u32),
        ConstantValue::uint(3, hv as u32),
        ConstantValue::float(4, scale),
        ConstantValue::uint(8, u32::from(rows.0)),
    ];
    // A plain step's scan takes the geometry, not the drafts (the lowering's twin).
    let pipe_consts = [scan_consts[..5].to_vec(), vec![ConstantValue::uint(10, 12)]].concat();
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
            lane: None,
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
            buf_slots(&device, &slots),
            buf_u32(&device, &codes),
        );
        // The simd scan binds the lanes (11, past the unread `D` at 10); the pipelined one none.
        let run = |kernel, grid, threads, binds_lanes: bool| {
            let (o, state) = (
                buf_zero_f32(&device, t * value_dim),
                buf_f16(&device, &pool0),
            );
            let (unread, lanes) = (buf_zero_f32(&device, 1), buf_f32(&device, &lanes0));
            let mut bufs = vec![
                &o, &conv_buf, &a_buf, &b_buf, &state, &cu_buf, &slot_buf, &code_buf, &alog_buf,
                &dt_buf,
            ];
            if binds_lanes {
                bufs.extend([&unread, &lanes]);
            }
            common::dispatch_threadgroups(&device, kernel, &bufs, grid, threads).then(|| {
                assert_eq!(read_f32(&lanes, lanes0.len()), lanes0, "the lanes moved");
                (read_f32(&o, t * value_dim), read_f16(&state, pool0.len()))
            })
        };
        let (Some((simd_o, simd_pool)), Some((pipe_o, pipe_pool))) = (
            run(&simd, size(1, value_dim / 4, seqs), size(32, 4, 1), true),
            run(&pipe, size(value_dim / 32, 1, seqs), size(128, 1, 1), false),
        ) else {
            return;
        };
        let what = format!("cu {cu:?}, slots {slots:?}, starts {starts:?}");
        let close = |part: &str, got: &[f32], want: &[f32], tol: f32| {
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= tol * scale,
                    "{what}: {part}[{i}]: {g} vs {w}"
                );
            }
        };
        close("o", &pipe_o, &simd_o, 1e-5);
        for slot in 0..2 {
            let entry = slot * entry_len..(slot + 1) * entry_len;
            // The entries are stored f16: the two scans' f32 registers agree to f32 rounding,
            // and the store rounds each to the same half — except where the two straddle a
            // rounding boundary, a half ULP apart.
            close(
                &format!("slot {slot} entry"),
                &pipe_pool[entry.clone()],
                &simd_pool[entry.clone()],
                1e-3,
            );
            assert_eq!(
                pipe_pool[entry.clone()] != pool0[entry],
                slots.contains(&(slot as i32)),
                "{what}: slot {slot}'s entry moved only if a sequence runs in it"
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
/// kernel under fast math, so they agree to f32 rounding. A drafting model's too (2 drafts: its
/// conv checkpoints and ssm records in draft lanes apart from the slots), on plain steps whose
/// record area is the second (code bit 16): the chain of the same drafts leaves what the one
/// command leaves, and the lanes as they were.
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
        let (conv_slot, ssm_slot) = (conv_dim * (kernel - 1), nv * hv * hk);
        // Two lanes of the drafts' conv checkpoints and ssm records.
        let conv_lanes0 = shifted(2 * rows.rows() * conv_slot, 29);
        let ssm_lanes0 = shifted(2 * 2 * rows.rows() * (conv_dim + 2 * nv), 31);
        let records = match drafts {
            0 => RecordArea::First,
            _ => RecordArea::Second,
        };
        let fresh = starts.map(|start| {
            GdnStep {
                start,
                checkpoint_rows: CheckpointRows::NONE,
                records,
                lane: None,
            }
            .encode()
        });
        let qkv = shifted(t * conv_dim, 7);
        let z = shifted(t * value_dim, 13);
        let (a, b) = (fill(t * nv), shifted(t * nv, 3));
        let (a_log, dt_bias, norm_w) = (fill(nv), shifted(nv, 5), shifted(hv, 17));
        let conv_w = shifted(conv_dim * kernel, 19);
        let conv0 = shifted(slots * conv_slot, 23);
        let ssm0 = q16(&shifted(slots * ssm_slot, 11));
        let build = |lib: &'static str, name: &'static str, c: Vec<ConstantValue>| {
            baked_build(&cache, &PipelineKey::new(lib, name, c)).expect(name)
        };
        // The geometry; the scan of a model that drafts takes its drafts too (slot 8), the one
        // command, which runs only plain steps, does not.
        let scan_consts = || {
            vec![
                ConstantValue::uint(0, nk as u32),
                ConstantValue::uint(1, nv as u32),
                ConstantValue::uint(2, hk as u32),
                ConstantValue::uint(3, hv as u32),
                ConstantValue::float(4, scale),
            ]
        };
        let (unread, conv_lanes, ssm_lanes) = (
            buf_zero_f32(&device, 1),
            buf_f32(&device, &conv_lanes0),
            buf_f32(&device, &ssm_lanes0),
        );
        let (cu, si) = (buf_i32(&device, &[0, 1, 2]), buf_slots(&device, &[1, 0]));
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
            ConstantValue::uint(5, u32::from(drafts)),
            // The lock's minimum (the seed window is 3 entries); one-row sequences walk one
            // block either way.
            ConstantValue::uint(6, 3),
        ];
        let conv = build("gdn_conv1d_varlen", "gdn_conv1d_varlen_f32", conv_consts);
        let conv_out = buf_zero_f32(&device, t * conv_dim);
        let (chain_conv, chain_ssm) = (buf_f32(&device, &conv0), buf_f16(&device, &ssm0));
        let mut bufs = vec![&conv_out, &qkv_buf, &w_buf, &chain_conv, &cu, &si, &fresh];
        if drafts > 0 {
            bufs.extend([&unread, &conv_lanes]);
        }
        let grid = size(t, conv_dim.div_ceil(256), 1);
        if !common::dispatch_threadgroups(&device, &conv, &bufs, grid, size(1, 256, 1)) {
            return;
        }
        let drafted = [
            vec![ConstantValue::uint(8, u32::from(drafts))],
            scan_consts(),
        ]
        .concat();
        let scan = build("gdn_scan_varlen", "gdn_scan_simd_f32", drafted);
        let o = buf_zero_f32(&device, t * value_dim);
        let mut bufs = vec![
            &o, &conv_out, &a_buf, &b_buf, &chain_ssm, &cu, &si, &fresh, &alog_buf, &dt_buf,
        ];
        if drafts > 0 {
            bufs.extend([&unread, &ssm_lanes]);
        }
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
        let (conv_state, ssm) = (buf_f32(&device, &conv0), buf_f16(&device, &ssm0));
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
        let close = |what: &str,
                     read: fn(&Buffer, usize) -> Vec<f32>,
                     got: &Buffer,
                     want: &Buffer,
                     n: usize,
                     tol: f32| {
            let (got, want) = (read(got, n), read(want, n));
            let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= tol * scale,
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
        // The ssm entries are stored f16: the two kernels' f32 registers agree to f32 rounding,
        // and the store rounds each to the same half — except where the two straddle a rounding
        // boundary, a half ULP apart.
        close("ssm state", read_f16, &ssm, &chain_ssm, ssm0.len(), 1e-3);
        close("out", read_f32, &out, &chain_out, t * value_dim, 1e-5);
        assert_eq!(
            read_f32(&conv_lanes, conv_lanes0.len()),
            conv_lanes0,
            "{geometry}"
        );
        assert_eq!(
            read_f32(&ssm_lanes, ssm_lanes0.len()),
            ssm_lanes0,
            "{geometry}"
        );
        // The step moved both states and wrote every output: agreeing on untouched buffers
        // would pass the comparisons above by accident.
        assert_ne!(read_f16(&ssm, ssm0.len()), ssm0, "{geometry}");
        assert_ne!(read_f32(&conv_state, conv0.len()), conv0, "{geometry}");
        assert!(
            read_f32(&out, t * value_dim).iter().all(|v| *v != 0.0),
            "{geometry}"
        );
    }
}

/// Compares a kernel's rows against the reference's, to `tol`.
fn assert_close(got: &[f32], want: &[f32], what: &str, tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < tol, "{what}[{i}] metal={g} ref={w}");
    }
}

/// A step's code, its sequence's draft `lane` named unless the step is plain ([`GdnStep::lane`]).
fn code(
    start: GdnStart,
    checkpoint_rows: CheckpointRows,
    records: RecordArea,
    lane: Option<DraftLane>,
) -> u32 {
    let mut step = GdnStep {
        start,
        checkpoint_rows,
        records,
        lane: None,
    };
    step.lane = lane.filter(|_| !step.is_plain());
    step.encode()
}

/// Inputs for a verify step's rejected drafts: unlike anything `fill` produces.
fn rejected(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37).cos() * 0.5).collect()
}

/// A verify step's checkpoints, conv ring: 7 tokens, then a verify step of the next token, a kept
/// draft and two rejected drafts, then one token resuming from the kept row's checkpoint — equals
/// the sequence that never saw the rejected drafts. In slot 1 of a 2-slot pool, its checkpoints in
/// lane 1 of 2 keeping 3 drafts, so the entries are offset and slot 0's and lane 0's must stay
/// untouched.
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
    let lane = DraftLanes::new(rows, 2).and_then(|l| l.lanes().nth(1));
    let lane_entries = rows.rows();
    let t_full = 10usize;
    let x_full = fill(t_full * conv_dim);
    let w = fill(conv_dim * kernel);
    let out_full = conv1d_ref(&x_full, &w, conv_dim, kernel, t_full);
    let x_verify = [
        &x_full[7 * conv_dim..9 * conv_dim],
        &rejected(2 * conv_dim)[..],
    ]
    .concat();

    let key = PipelineKey::new(
        "gdn_conv1d_varlen",
        "gdn_conv1d_varlen_f32",
        vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
            ConstantValue::uint(5, u32::from(rows.0)),
            // A 4-token block — the lock's minimum for 3 drafts (a verify step's 4 rows sit in
            // block 0 alone): the 7-token run walks two blocks, the second seeding its window
            // from x, and block 0 alone reads the seed and writes the new ring.
            ConstantValue::uint(6, 4),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");
    let w_buf = buf_f32(&device, &w);
    let state_buf = buf_zero_f32(&device, 2 * entry_len);
    let lanes_buf = buf_zero_f32(&device, 2 * lane_entries * entry_len);
    let unread_bias = buf_zero_f32(&device, 1);
    let run = |x: &[f32], start: GdnStart, checkpoint_rows: CheckpointRows| -> Option<Vec<f32>> {
        let num_tokens = x.len() / conv_dim;
        let out_buf = buf_zero_f32(&device, x.len());
        let x_buf = buf_f32(&device, x);
        let cu = buf_i32(&device, &[0, num_tokens as i32]);
        let si = buf_slots(&device, &[1]);
        let step = buf_u32(
            &device,
            &[code(start, checkpoint_rows, RecordArea::First, lane)],
        );
        common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[
                &out_buf,
                &x_buf,
                &w_buf,
                &state_buf,
                &cu,
                &si,
                &step,
                &unread_bias,
                &lanes_buf,
            ],
            MTLSize {
                width: 1,
                height: 1,
                depth: num_tokens.div_ceil(4),
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
        &x_full[..7 * conv_dim],
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
        &out_full[7 * conv_dim..9 * conv_dim],
        "verify kept rows",
        1e-4,
    );
    let Some(next) = run(
        &x_full[9 * conv_dim..],
        GdnStart::Checkpoint(1),
        CheckpointRows::NONE,
    ) else {
        return;
    };
    assert_close(&next, &out_full[9 * conv_dim..], "resumed row", 1e-4);
    let slot0 = read_f32(&state_buf, entry_len);
    assert!(
        slot0.iter().all(|&v| v == 0.0),
        "slot 0's entry was written"
    );
    let lane0 = read_f32(&lanes_buf, lane_entries * entry_len);
    assert!(
        lane0.iter().all(|&v| v == 0.0),
        "lane 0's checkpoints were written"
    );
}

/// A one-draft verify step of TWO rows (shorter than the kernel's seed window) replayed from a
/// checkpoint with its draft kept, then a slot step reading the ring the replay wrote: the ring's
/// seed entries must be the checkpoint's, not the lane row this step's own checkpoints overwrote
/// — the replay's seed IS that lane row, so a ring write that re-reads it from memory picks up
/// this step's inputs instead. The multi-sequence MTP serving shape (one draft per sequence).
#[test]
fn gdn_conv1d_varlen_ring_survives_a_short_replayed_verify() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");

    let (conv_dim, kernel) = (32usize, 4usize);
    let entry_len = conv_dim * (kernel - 1);
    let rows = CheckpointRows(1);
    let lane = DraftLanes::new(rows, 2).and_then(|l| l.lanes().nth(1));
    let lane_entries = rows.rows();
    // Rows 0..4 the prompt, 4 the verify step's token (its draft rejected), 5..6 the replayed
    // verify (token and draft, both kept), 7 the slot step that reads the ring.
    let t_full = 8usize;
    let x_full = fill(t_full * conv_dim);
    let w = fill(conv_dim * kernel);
    let out_full = conv1d_ref(&x_full, &w, conv_dim, kernel, t_full);
    let x_rejected = [&x_full[4 * conv_dim..5 * conv_dim], &rejected(conv_dim)[..]].concat();

    let key = PipelineKey::new(
        "gdn_conv1d_varlen",
        "gdn_conv1d_varlen_f32",
        vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
            ConstantValue::uint(5, u32::from(rows.0)),
            // The lock's minimum (the seed window is 3 entries): the 4-token prompt walks two
            // blocks, the 2-row verify steps one — the replay's seed branch of the ring write.
            ConstantValue::uint(6, 3),
        ],
    );
    let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");
    let w_buf = buf_f32(&device, &w);
    let state_buf = buf_zero_f32(&device, 2 * entry_len);
    let lanes_buf = buf_zero_f32(&device, 2 * lane_entries * entry_len);
    let unread_bias = buf_zero_f32(&device, 1);
    let run = |x: &[f32], start: GdnStart, checkpoint_rows: CheckpointRows| -> Option<Vec<f32>> {
        let num_tokens = x.len() / conv_dim;
        let out_buf = buf_zero_f32(&device, x.len());
        let x_buf = buf_f32(&device, x);
        let cu = buf_i32(&device, &[0, num_tokens as i32]);
        let si = buf_slots(&device, &[1]);
        let step = buf_u32(
            &device,
            &[code(start, checkpoint_rows, RecordArea::First, lane)],
        );
        common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[
                &out_buf,
                &x_buf,
                &w_buf,
                &state_buf,
                &cu,
                &si,
                &step,
                &unread_bias,
                &lanes_buf,
            ],
            MTLSize {
                width: 1,
                height: 1,
                depth: num_tokens.div_ceil(3),
            },
            MTLSize {
                width: 1,
                height: conv_dim,
                depth: 1,
            },
        )
        .then(|| read_f32(&out_buf, x.len()))
    };

    // The prompt, four rows fresh (two blocks).
    let Some(_) = run(
        &x_full[..4 * conv_dim],
        GdnStart::Fresh,
        CheckpointRows::NONE,
    ) else {
        return;
    };
    // A verify step of two rows whose draft is rejected: its token is kept, checkpointed in the
    // lane at row 0 — the row the next step replays from.
    let Some(verify) = run(&x_rejected, GdnStart::Slot, rows) else {
        return;
    };
    assert_close(
        &verify[..conv_dim],
        &out_full[4 * conv_dim..5 * conv_dim],
        "verify kept row",
        1e-4,
    );
    // The replayed verify, two rows (the seed window is 3), resumed from the kept row's
    // checkpoint with its own draft kept — the bug's regime: the ring write's seed entries come
    // from the lane row this step's checkpoints overwrite.
    let Some(replay) = run(
        &x_full[5 * conv_dim..7 * conv_dim],
        GdnStart::Checkpoint(0),
        rows,
    ) else {
        return;
    };
    assert_close(
        &replay,
        &out_full[5 * conv_dim..7 * conv_dim],
        "replayed verify rows",
        1e-4,
    );
    // The slot step reads the ring the replay wrote: its row is the reference's seventh, whose
    // window reaches the replay's seed — a ring that picked up this step's inputs instead is
    // wrong here.
    let Some(next) = run(
        &x_full[7 * conv_dim..],
        GdnStart::Slot,
        CheckpointRows::NONE,
    ) else {
        return;
    };
    assert_close(
        &next,
        &out_full[7 * conv_dim..],
        "slot row after replay",
        1e-4,
    );
    let slot0 = read_f32(&state_buf, entry_len);
    assert!(
        slot0.iter().all(|&v| v == 0.0),
        "slot 0's entry was written"
    );
    let lane0 = read_f32(&lanes_buf, lane_entries * entry_len);
    assert!(
        lane0.iter().all(|&v| v == 0.0),
        "lane 0's checkpoints were written"
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
    let lane_len = 2 * rows.rows() * (conv_dim + 2 * nv);
    let lane = DraftLanes::new(rows, 2).and_then(|l| l.lanes().nth(1));
    // Rows 0..3 the prompt, 3..7 the verify step (its token and three kept drafts), 7 the next.
    let t_full = 8usize;
    let (conv, g, beta) = (
        fill(t_full * conv_dim),
        fill(t_full * nv),
        fill(t_full * nv),
    );
    let at = |x: &[f32], w: usize, r: std::ops::Range<usize>| x[r.start * w..r.end * w].to_vec();
    let entry = [GdnSeqSlots::plain(GdnSlot::new(1))];
    let code = |start, checkpoint_rows, records| [code(start, checkpoint_rows, records, lane)];
    let run = |(state, lanes): (&Buffer, &Buffer), r: std::ops::Range<usize>, step: [u32; 1]| {
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
            Some((rows.0, lanes)),
        )
    };
    let none = CheckpointRows::NONE;
    let (r0, r1) = (RecordArea::First, RecordArea::Second);
    let pool = || {
        (
            buf_zero_f16(&device, 2 * entry_len),
            buf_zero_f32(&device, 2 * lane_len),
        )
    };
    let (state, lanes) = pool();
    let drafting = (&state, &lanes);
    let (Some(_), Some(_), Some(o_next)) = (
        run(drafting, 0..3, code(GdnStart::Fresh, none, r0)),
        run(drafting, 3..7, code(GdnStart::Slot, rows, r0)),
        run(drafting, 7..8, code(GdnStart::Checkpoint(3), none, r1)),
    ) else {
        return;
    };
    let (plain, plain_lanes) = pool();
    let plain = (&plain, &plain_lanes);
    let (Some(_), Some(_), Some(o_plain)) = (
        run(plain, 0..3, code(GdnStart::Fresh, none, r0)),
        run(plain, 3..7, code(GdnStart::Slot, none, r0)),
        run(plain, 7..8, code(GdnStart::Slot, none, r0)),
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
    let Some(o_slot) = run(drafting, 7..8, code(GdnStart::Slot, none, r1)) else {
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
/// rejected drafts computes. Each step's code carries the slot's draft lane and record area as the
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
    let lane_len = 2 * rows.rows() * (conv_dim + 2 * nv);
    let lane = DraftLanes::new(rows, 2).and_then(|l| l.lanes().nth(1));
    let t_full = 6usize;

    let conv_full = fill(t_full * conv_dim);
    let g_full = fill(t_full * nv);
    let beta_full = fill(t_full * nv);
    let (q, k, v) = split_conv(&conv_full, nk, nv, hk, hv, t_full);
    // The oracle crosses the slot's f16 entry where the steps do: the prompt's store (row 2's
    // end — the verify step's base), and the kept rows' state as the plain run leaves it there
    // (row 4's end — the replay's boundary).
    let o_full = |slot_after_rows: &[usize]| {
        recurrent_ref(
            &q,
            &k,
            &v,
            &g_full,
            &beta_full,
            nk,
            nv,
            hk,
            hv,
            t_full,
            scale,
            slot_after_rows,
        )
    };
    // Rows `3..5` of the kept sequence, then two rejected drafts.
    let verify = |full: &[f32], width: usize| {
        [&full[3 * width..5 * width], &rejected(2 * width)[..]].concat()
    };

    // Slot 1 of a two-slot pool, lane 1 of two; its steps' codes chained as the allocator chains
    // them.
    let entry = [GdnSeqSlots::plain(GdnSlot::new(1))];
    let pool = || {
        (
            buf_zero_f16(&device, 2 * entry_len),
            buf_zero_f32(&device, 2 * lane_len),
        )
    };
    let slot = std::cell::Cell::new(RecordArea::First);
    let step = |start, checkpoint_rows| {
        let records = slot.get();
        let step = GdnStep {
            start,
            checkpoint_rows,
            records,
            lane: None,
        };
        slot.set(step.after());
        [code(start, checkpoint_rows, records, lane)]
    };
    let run = |(state, lanes): (&Buffer, &Buffer),
               conv: &[f32],
               g: &[f32],
               beta: &[f32],
               step: [u32; 1]| {
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
            Some((rows.0, lanes)),
        )
    };

    let (state, lanes) = pool();
    let drafting = (&state, &lanes);
    let rows_of = |full: &[f32], width: usize, r: std::ops::Range<usize>| {
        full[r.start * width..r.end * width].to_vec()
    };
    let prompt = |pool, fresh| {
        run(
            pool,
            &rows_of(&conv_full, conv_dim, 0..3),
            &rows_of(&g_full, nv, 0..3),
            &rows_of(&beta_full, nv, 0..3),
            fresh,
        )
    };
    let Some(_) = prompt(drafting, step(GdnStart::Fresh, CheckpointRows::NONE)) else {
        return;
    };
    let Some(o_verify) = run(
        drafting,
        &verify(&conv_full, conv_dim),
        &verify(&g_full, nv),
        &verify(&beta_full, nv),
        step(GdnStart::Slot, rows),
    ) else {
        return;
    };
    assert_close(
        &o_verify[..2 * value_dim],
        &o_full(&[2])[3 * value_dim..5 * value_dim],
        "verify kept rows",
        1e-3,
    );
    let next = |pool, code| {
        run(
            pool,
            &conv_full[5 * conv_dim..],
            &g_full[5 * nv..],
            &beta_full[5 * nv..],
            code,
        )
    };
    let Some(o_next) = next(
        drafting,
        step(GdnStart::Checkpoint(1), CheckpointRows::NONE),
    ) else {
        return;
    };
    assert_close(
        &o_next,
        &o_full(&[2, 4])[5 * value_dim..],
        "resumed row",
        1e-3,
    );
    let slot0 = read_f16(&state, entry_len);
    assert!(
        slot0.iter().all(|&v| v == 0.0),
        "slot 0's entry was written"
    );
    let lane0 = read_f32(&lanes, lane_len);
    assert!(
        lane0.iter().all(|&v| v == 0.0),
        "lane 0's records were written"
    );

    // The same rows without drafts: prompt, the kept rows, the next row.
    let plain_code = |start| [code(start, CheckpointRows::NONE, RecordArea::First, None)];
    let (plain, plain_lanes) = pool();
    let plain = (&plain, &plain_lanes);
    let kept = |pool| {
        run(
            pool,
            &rows_of(&conv_full, conv_dim, 3..5),
            &rows_of(&g_full, nv, 3..5),
            &rows_of(&beta_full, nv, 3..5),
            plain_code(GdnStart::Slot),
        )
    };
    let (Some(_), Some(_), Some(o_plain)) = (
        prompt(plain, plain_code(GdnStart::Fresh)),
        kept(plain),
        next(plain, plain_code(GdnStart::Slot)),
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
    let lane_len = 2 * rows.rows() * (conv_dim + 2 * nv);
    let lane = DraftLanes::new(rows, 2).and_then(|l| l.lanes().nth(1));
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
                ConstantValue::uint(8, u32::from(rows.0)),
            ],
        ),
    )
    .expect("gdn_scan_simd_f32");
    let (alog_buf, dt_buf) = (buf_f32(&device, &a_log), buf_f32(&device, &dt_bias));
    // Slot 1 of a two-slot pool, lane 1 of two.
    let entry = [GdnSeqSlots::plain(GdnSlot::new(1))];
    let entry_buf = buf_seqs(&device, &entry);
    let unread_skip = buf_zero_f32(&device, 1);
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
    // A pool: its slots' state entries and its draft lanes.
    type Pool<'a> = (&'a Buffer, &'a Buffer);
    let run =
        |scan: Scan, (state, lanes): Pool, conv: &[f32], a: &[f32], b: &[f32], code: [u32; 1]| {
            let t = a.len() / nv;
            match scan {
                Scan::Serial => {
                    let (g, beta) = gating(a, b);
                    let cu = [0, t as i32];
                    dispatch_scan(
                        &device,
                        &cache,
                        conv,
                        &g,
                        &beta,
                        state,
                        &cu,
                        &entry,
                        &code,
                        nk,
                        nv,
                        hk,
                        hv,
                        t,
                        Some((rows.0, lanes)),
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
                        &o,
                        &conv_buf,
                        &a_buf,
                        &b_buf,
                        state,
                        &cu_buf,
                        &entry_buf,
                        &code_buf,
                        &alog_buf,
                        &dt_buf,
                        &unread_skip,
                        lanes,
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
    let pool = || {
        (
            buf_zero_f16(&device, 2 * entry_len),
            buf_zero_f32(&device, 2 * lane_len),
        )
    };
    let prompt = |scan, pool: Pool, code| {
        run(
            scan,
            pool,
            &at(&conv, conv_dim, 0..3),
            &at(&a, nv, 0..3),
            &at(&b, nv, 0..3),
            code,
        )
    };
    let next = |scan, pool: Pool, code| {
        run(
            scan,
            pool,
            &at(&conv, conv_dim, 5..6),
            &at(&a, nv, 5..6),
            &at(&b, nv, 5..6),
            code,
        )
    };
    let plain_code = |start| [code(start, CheckpointRows::NONE, RecordArea::First, None)];

    // The simd scan's rows without drafts: prompt, the kept rows, the next row.
    let (plain, plain_lanes) = pool();
    let plain = (&plain, &plain_lanes);
    let (Some(_), Some(o_kept), Some(o_plain)) = (
        prompt(Scan::Simd, plain, plain_code(GdnStart::Fresh)),
        run(
            Scan::Simd,
            plain,
            &at(&conv, conv_dim, 3..5),
            &at(&a, nv, 3..5),
            &at(&b, nv, 3..5),
            plain_code(GdnStart::Slot),
        ),
        next(Scan::Simd, plain, plain_code(GdnStart::Slot)),
    ) else {
        return;
    };
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    // The two scans' states cross the slot's f16 entry at the same boundaries; their registers
    // agree to the scans' f32 rounding, and the store rounds each to the same half — except
    // where the two straddle a rounding boundary, a half ULP apart.
    let close = |what: &str, got: &[f32], want: &[f32]| {
        let scale = want.iter().fold(0f32, |m, w| m.max(w.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= 1e-3 * scale, "{what}[{i}]: {g} vs {w}");
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
                lane: None,
            };
            slot.set(step.after());
            [code(start, checkpoint_rows, records, lane)]
        };
        let (state, lanes) = pool();
        let drafting = (&state, &lanes);
        prompt(
            verifier,
            drafting,
            step(GdnStart::Fresh, CheckpointRows::NONE),
        )?;
        let o_verify = run(
            verifier,
            drafting,
            &verify(&conv, conv_dim),
            &verify(&a, nv),
            &verify(&b, nv),
            step(GdnStart::Slot, rows),
        )?;
        let o_next = next(resumer, drafting, step(start, CheckpointRows::NONE))?;
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

// ─── Prefix-cache snapshots (`shaders/gdn_state.h`) ────────────────────────────────────────────

/// The pools [`check_snapshots`] runs on: slot 0 no sequence's, [`LIVE`] the sequence's, the
/// [`SNAPS`] its snapshots.
const POOL_SLOTS: usize = 4;
const LIVE: i32 = 1;
const SNAPS: [i32; 2] = [2, 3];

/// The record of [`second_of_two`]'s first sequence, which has no rows: a kernel that ran it would
/// restore from the last snapshot and save into slot 0.
fn idle() -> GdnSeqSlots {
    GdnSeqSlots::plain(GdnSlot::new(0))
        .restoring(GdnSlot::new(SNAPS[1] as usize))
        .saving([save(1, 0), save(1, 0)])
}

/// A save into slot `snapshot` after `rows` rows.
fn save(rows: usize, snapshot: i32) -> Option<GdnSave> {
    Some(GdnSave {
        rows_through: std::num::NonZeroU32::new(rows as u32).expect("rows"),
        snapshot: GdnSlot::new(snapshot as usize),
    })
}

/// A pool's slots, for [`check_snapshots`]: `len` entries a slot, its state entry (a drafting
/// model's checkpoints and records live in its draft lanes, apart) — f32 for a conv pool, f16
/// for an ssm pool, the dtype its kernels store.
#[derive(Clone, Copy, Debug)]
struct Slots {
    len: usize,
    half: bool,
}

impl Slots {
    /// A conv pool's: a window of `conv_dim * (kernel - 1)` a slot.
    fn conv(conv_dim: usize, kernel: usize) -> Self {
        let dims = GdnStateDims {
            conv_dim,
            conv_kernel: kernel,
            num_v_heads: 0,
            head_v_dim: 0,
            head_k_dim: 0,
        };
        Self {
            len: dims.conv_entry_len(),
            half: false,
        }
    }

    /// An ssm pool's, as `GdnStateDims::ssm_entry_len` lays a slot out.
    fn ssm(nk: usize, nv: usize, hk: usize, hv: usize) -> Self {
        let dims = GdnStateDims {
            conv_dim: 2 * nk * hk + nv * hv,
            conv_kernel: 4,
            num_v_heads: nv,
            head_v_dim: hv,
            head_k_dim: hk,
        };
        Self {
            len: dims.ssm_entry_len(),
            half: true,
        }
    }
}

/// One step of a kernel under [`check_snapshots`]: rows `rows` of the test's inputs as a sequence
/// with record `seq` starting from `start` ([`second_of_two`]), on pools holding `pools` (one a
/// [`Slots`]). Its output and the pools after, or `None` with no MTL4 queue.
type SnapshotStep<'a> = &'a dyn Fn(
    &[Vec<f32>],
    std::ops::Range<usize>,
    GdnSeqSlots,
    GdnStart,
) -> Option<(Vec<f32>, Vec<Vec<f32>>)>;

/// The step code of a step without drafts from `start`.
fn plain_code(start: GdnStart) -> u32 {
    GdnStep {
        start,
        checkpoint_rows: CheckpointRows::NONE,
        records: RecordArea::First,
        lane: None,
    }
    .encode()
}

/// `cu_seqlens`, `state_indices` and step codes of a step of `n` rows whose one sequence with rows
/// is its second — `seq`, from `start`. The first has none and the record [`idle`]: every check
/// reads the second record, at its stride, past one no kernel may act on.
fn second_of_two(device: &Device, n: usize, seq: GdnSeqSlots, start: GdnStart) -> [Buffer; 3] {
    [
        buf_i32(device, &[0, 0, n as i32]),
        buf_seqs(device, &[idle(), seq]),
        buf_u32(device, &[plain_code(GdnStart::Slot), plain_code(start)]),
    ]
}

/// Rows `r` of `[rows, width]` `x`.
fn row_range(x: &[f32], width: usize, r: std::ops::Range<usize>) -> Vec<f32> {
    x[r.start * width..r.end * width].to_vec()
}

fn dims(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

/// `got` is `want`, bit for bit.
fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let differs = |(g, w): (&f32, &f32)| g.to_bits() != w.to_bits();
    if let Some(i) = got.iter().zip(want).position(differs) {
        let n = got.iter().zip(want).filter(|&p| differs(p)).count();
        panic!(
            "{what}: {n} of {} differ, the first [{i}]: {} vs {}",
            got.len(),
            got[i],
            want[i]
        );
    }
}

/// The snapshot contract (`shaders/gdn_state.h`) for the kernel `step` runs, over a step of `t`
/// rows and a row `mid` inside it, bit for bit — the kernel stores the very registers it carries:
/// * a save stores into its snapshot what the live slot holds after a step of just the rows it
///   runs through, the step's output and live slot as without it — mid-step, at its last row, two
///   at once, two in either order;
/// * a restore (`source`) computes what the same step does on a live slot holding the snapshot,
///   the snapshot untouched; a fresh start ignores its source; a restore saves like a step
///   on that slot;
/// * slot 0, which no sequence with rows names, stays as it was.
fn check_snapshots(what: &str, layouts: &[Slots], t: usize, mid: usize, step: SnapshotStep<'_>) {
    let pools: Vec<Vec<f32>> = (layouts.iter().enumerate())
        .map(|(p, l)| {
            let by = 11 + 7 * p;
            let entries = fill(POOL_SLOTS * l.len + by)[by..].to_vec();
            // An f16 pool's fixtures are the values its kernels store, so a slot no step
            // touches reads back bit for bit.
            l.half.then(|| q16(&entries)).unwrap_or(entries)
        })
        .collect();
    let slot = |pools: &[Vec<f32>], p: usize, s: i32| {
        let len = layouts[p].len;
        pools[p][s as usize * len..][..len].to_vec()
    };
    let entries = |pools: &[Vec<f32>], s: i32| -> Vec<Vec<f32>> {
        (0..layouts.len()).map(|p| slot(pools, p, s)).collect()
    };
    let plain = GdnSeqSlots::plain(GdnSlot::new(LIVE as usize));
    // Slot 0 and the snapshots are as they were, but for each `(snapshot, entries)` stored, whose
    // entries hold those.
    let check_slots = |case: &str, after: &[Vec<f32>], stored: &[(i32, Vec<Vec<f32>>)]| {
        for p in 0..layouts.len() {
            for s in [0, SNAPS[0], SNAPS[1]] {
                let (got, was) = (slot(after, p, s), slot(&pools, p, s));
                let at = format!("{what}, {case}: pool {p} slot {s}");
                match stored.iter().find(|(snap, _)| *snap == s) {
                    Some((_, want)) => assert_bits(&got, &want[p], &at),
                    None => assert_bits(&got, &was, &at),
                }
            }
        }
    };
    let live_slot_is = |case: &str, after: &[Vec<f32>], want: &[Vec<f32>]| {
        for p in 0..layouts.len() {
            let at = format!("{what}, {case}: pool {p} live slot");
            assert_bits(&slot(after, p, LIVE), &slot(want, p, LIVE), &at);
        }
    };

    let Some((out_plain, after_plain)) = step(&pools, 0..t, plain, GdnStart::Slot) else {
        return;
    };
    assert_ne!(
        entries(&after_plain, LIVE),
        entries(&pools, LIVE),
        "{what}: the step left its live slot"
    );
    check_slots("a plain step", &after_plain, &[]);

    // Saves.
    let (s0, s1) = (SNAPS[0], SNAPS[1]);
    for saves in [
        &[(mid, s0)][..],
        &[(t, s0)],
        &[(mid, s0), (t, s1)],
        &[(1, s1), (mid, s0)],
    ] {
        let case = format!("saves {saves:?}");
        let mut stores = [None; GDN_SAVES];
        for (j, &(through, snap)) in saves.iter().enumerate() {
            stores[j] = save(through, snap);
        }
        let seq = plain.saving(stores);
        let Some((out, after)) = step(&pools, 0..t, seq, GdnStart::Slot) else {
            return;
        };
        // A save rounds the carried registers through the entry's f16 where it stores, so a
        // step with saves is the plain steps its save rows chunk — each boundary the store
        // and load a restore resumes from — bit for bit: the outputs, the live slot, and
        // each snapshot the chunked run's live slot at its row.
        let mut chunked_out = Vec::new();
        let mut chunked = pools.clone();
        let mut at = 0usize;
        let mut stored = Vec::new();
        for &(through, snap) in saves {
            if at < through {
                let Some((o, after_c)) = step(&chunked, at..through, plain, GdnStart::Slot) else {
                    return;
                };
                chunked_out.extend(o);
                chunked = after_c;
                at = through;
            }
            stored.push((snap, entries(&chunked, LIVE)));
        }
        if at < t {
            let Some((o, after_c)) = step(&chunked, at..t, plain, GdnStart::Slot) else {
                return;
            };
            chunked_out.extend(o);
            chunked = after_c;
        }
        assert_bits(&out, &chunked_out, &format!("{what}, {case}: output"));
        live_slot_is(&case, &after, &chunked);
        check_slots(&case, &after, &stored);
    }

    // A restore: the step on `pools` with the live slot's entries the snapshot's.
    let restore = plain.restoring(GdnSlot::new(s0 as usize));
    let mut copied = pools.clone();
    for (p, l) in layouts.iter().enumerate() {
        let from = s0 as usize * l.len;
        copied[p].copy_within(from..from + l.len, LIVE as usize * l.len);
    }
    let (Some((out_r, after_r)), Some((out_c, after_c))) = (
        step(&pools, 0..t, restore, GdnStart::Slot),
        step(&copied, 0..t, plain, GdnStart::Slot),
    ) else {
        return;
    };
    assert_bits(&out_r, &out_c, &format!("{what}, restore: output"));
    assert_ne!(out_r, out_plain, "{what}: the restore read the live slot");
    live_slot_is("restore", &after_r, &after_c);
    check_slots("restore", &after_r, &[]);

    // A fresh start ignores the source.
    let (Some((out_f, after_f)), Some((out_f0, after_f0))) = (
        step(&pools, 0..t, restore, GdnStart::Fresh),
        step(&pools, 0..t, plain, GdnStart::Fresh),
    ) else {
        return;
    };
    assert_bits(
        &out_f,
        &out_f0,
        &format!("{what}, fresh with a source: output"),
    );
    for (p, (got, want)) in after_f.iter().zip(&after_f0).enumerate() {
        assert_bits(got, want, &format!("{what}, fresh with a source: pool {p}"));
    }

    // A restore that saves — the save rounds the carried registers through the entry's f16,
    // so the run is the plain steps its save row chunks on the restored state, bit for bit.
    let seq = restore.saving([save(mid, s1), None]);
    let (Some((out_rs, after_rs)), Some((mut out_m, after_m))) = (
        step(&pools, 0..t, seq, GdnStart::Slot),
        step(&copied, 0..mid, plain, GdnStart::Slot),
    ) else {
        return;
    };
    // The snapshot is the chunk's end state — what a restore of s1 would load.
    let stored_s1 = entries(&after_m, LIVE);
    // The continuation restarts from it, as a restore of s1 would; a save at the step's
    // end stores the final state, and nothing follows it.
    let want;
    if mid < t {
        let Some((out_c2, after_c2)) = step(&after_m, mid..t, plain, GdnStart::Slot) else {
            return;
        };
        out_m.extend(out_c2);
        want = after_c2;
    } else {
        want = after_m;
    }
    assert_bits(
        &out_rs,
        &out_m,
        &format!("{what}, restore and save: output"),
    );
    live_slot_is("restore and save", &after_rs, &want);
    check_slots("restore and save", &after_rs, &[(s1, stored_s1)]);
}

/// [`check_snapshots`] for `gdn_conv1d_varlen_f32`, in its GDN form and its Mamba-2 one (a column
/// slice of a wider row, with a bias), without drafts and compiled for a drafting model (its draft
/// lanes bound, a plain step never reading them). The window saved after row 2 still holds one of
/// the start state's inputs.
#[test]
fn gdn_conv1d_varlen_saves_and_restores_snapshots() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (conv_dim, kernel, t) = (64usize, 4usize, 7usize);
    // The Mamba-2 form's row: `conv_dim` channels from column 8 of 96.
    let (row, offset) = (conv_dim + 32, 8usize);
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (x, w, bias) = (
        shifted(t * row, 3),
        shifted(conv_dim * kernel, 5),
        shifted(conv_dim, 7),
    );
    let (w_buf, bias_buf) = (buf_f32(&device, &w), buf_f32(&device, &bias));
    let unread_lanes = buf_zero_f32(&device, 1);
    for (mamba, drafts) in [(false, 0u8), (false, 2), (true, 0), (true, 2)] {
        let mut constants = vec![
            ConstantValue::uint(0, conv_dim as u32),
            ConstantValue::uint(1, kernel as u32),
            ConstantValue::uint(5, u32::from(drafts)),
            // A 3-token block — the lock's minimum (the seed window is 3 entries): the saves
            // land mid-block and on block boundaries alike over the 7 rows' three blocks.
            ConstantValue::uint(6, 3),
        ];
        let width = match mamba {
            false => conv_dim,
            true => {
                constants.extend([
                    ConstantValue::uint(2, row as u32),
                    ConstantValue::uint(3, offset as u32),
                    ConstantValue::boolean(4, true),
                ]);
                row
            }
        };
        let key = PipelineKey::new("gdn_conv1d_varlen", "gdn_conv1d_varlen_f32", constants);
        let pipeline = baked_build(&cache, &key).expect("gdn_conv1d_varlen pipeline");
        let step = |pools: &[Vec<f32>], r: std::ops::Range<usize>, seq, start| {
            let n = r.len();
            let out = buf_zero_f32(&device, n * conv_dim);
            let x_buf = buf_f32(&device, &row_range(&x, width, r));
            let state = buf_f32(&device, &pools[0]);
            let [cu, si, code] = second_of_two(&device, n, seq, start);
            let mut bufs = vec![&out, &x_buf, &w_buf, &state, &cu, &si, &code, &bias_buf];
            bufs.extend((drafts > 0).then_some(&unread_lanes));
            let (grid, threads) = (dims(2, 1, n.div_ceil(3)), dims(1, conv_dim, 1));
            common::dispatch_threadgroups(&device, &pipeline, &bufs, grid, threads).then(|| {
                let after = vec![read_f32(&state, pools[0].len())];
                (read_f32(&out, n * conv_dim), after)
            })
        };
        let what = format!("gdn_conv1d_varlen, mamba {mamba}, drafts {drafts}");
        check_snapshots(&what, &[Slots::conv(conv_dim, kernel)], t, 2, &step);
    }
}

/// [`check_snapshots`] for `gdn_scan_varlen_f32`, the per-thread scan, in its GDN form and its
/// Mamba-2 one (whose save comes before that branch's own end of the row), without drafts and
/// compiled for a drafting model (its draft lanes bound, a plain step never reading them).
#[test]
fn gdn_scan_varlen_saves_and_restores_snapshots() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv, t) = (2usize, 4usize, 16usize, 16usize, 7usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (conv, g, beta) = (fill(t * conv_dim), shifted(t * nv, 3), shifted(t * nv, 7));
    let d_buf = buf_f32(&device, &shifted(nv, 5));
    let unread_lanes = buf_zero_f32(&device, 1);
    for (mamba, drafts) in [(false, 0u8), (false, 2), (true, 0), (true, 2)] {
        let mut constants = vec![
            ConstantValue::uint(0, nk as u32),
            ConstantValue::uint(1, nv as u32),
            ConstantValue::uint(2, hk as u32),
            ConstantValue::uint(3, hv as u32),
            ConstantValue::float(4, (hk as f32).powf(-0.5)),
            ConstantValue::uint(8, u32::from(drafts)),
        ];
        constants.extend(mamba.then(|| ConstantValue::boolean(5, true)));
        let key = PipelineKey::new("gdn_scan_varlen", "gdn_scan_varlen_f32", constants);
        let pipeline = baked_build(&cache, &key).expect("gdn_scan_varlen pipeline");
        let step = |pools: &[Vec<f32>], r: std::ops::Range<usize>, seq, start| {
            let n = r.len();
            let o = buf_zero_f32(&device, n * value_dim);
            let (conv_buf, g_buf, beta_buf) = (
                buf_f32(&device, &row_range(&conv, conv_dim, r.clone())),
                buf_f32(&device, &row_range(&g, nv, r.clone())),
                buf_f32(&device, &row_range(&beta, nv, r)),
            );
            let state = buf_f16(&device, &pools[0]);
            let [cu, si, code] = second_of_two(&device, n, seq, start);
            let mut bufs = vec![
                &o, &conv_buf, &g_buf, &beta_buf, &state, &cu, &si, &code, &d_buf,
            ];
            bufs.extend((drafts > 0).then_some(&unread_lanes));
            let (grid, threads) = (dims(1, nv, 2), dims(hv, 1, 1));
            common::dispatch_threadgroups(&device, &pipeline, &bufs, grid, threads).then(|| {
                let after = vec![read_f16(&state, pools[0].len())];
                (read_f32(&o, n * value_dim), after)
            })
        };
        let what = format!("gdn_scan_varlen, mamba {mamba}, drafts {drafts}");
        check_snapshots(&what, &[Slots::ssm(nk, nv, hk, hv)], t, 3, &step);
    }
}

/// A step's buffers as [`second_of_two`] builds them, but the second sequence's step code `code`.
fn second_of_two_coded(device: &Device, n: usize, seq: GdnSeqSlots, code: u32) -> [Buffer; 3] {
    [
        buf_i32(device, &[0, 0, n as i32]),
        buf_seqs(device, &[idle(), seq]),
        buf_u32(device, &[plain_code(GdnStart::Slot), code]),
    ]
}

/// A verify step's code: from the slot, `drafts` drafts recorded in lane 0.
fn verify_code(drafts: u8) -> u32 {
    GdnStep {
        start: GdnStart::Slot,
        checkpoint_rows: CheckpointRows(drafts),
        records: RecordArea::First,
        lane: DraftLanes::new(CheckpointRows(drafts), 1).and_then(|l| l.lanes().next()),
    }
    .encode()
}

/// [`check_snapshots`] for `gdn_scan_simd_f32`, lanes splitting head_k (two state elements a
/// lane), in its GDN form and its Mamba-2 one, without drafts and compiled for a drafting model
/// (its draft lanes bound, a plain step never reading them) — and compiled as a geometry the
/// pipelined scan covers (`GDN_SCAN_PLAIN_ROWS_PIPELINED`): there a plain step of more than one row
/// is the pipelined scan's, which this scan leaves untouched, and every save it takes is of a
/// one-row step's row, the contract checked over one row.
#[test]
fn gdn_scan_simd_saves_and_restores_snapshots() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv, t) = (2usize, 4usize, 64usize, 8usize, 7usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (conv, a, b) = (fill(t * conv_dim), shifted(t * nv, 3), shifted(t * nv, 7));
    let (alog_buf, dt_buf, d_buf) = (
        buf_f32(&device, &fill(nv)),
        buf_f32(&device, &shifted(nv, 5)),
        buf_f32(&device, &shifted(nv, 9)),
    );
    // One draft lane — two record areas of `drafts + 1` rows (`GdnStateDims::ssm_lane_len`) — for
    // the verify step below; a plain step never reads it.
    let unread_lanes = buf_zero_f32(&device, 2 * 3 * (conv_dim + 2 * nv));
    let cases = [(false, 0u8), (false, 2), (true, 0), (true, 2)];
    for ((mamba, drafts), pipelined) in cases.into_iter().flat_map(|c| [(c, false), (c, true)]) {
        let mut constants = vec![
            ConstantValue::uint(0, nk as u32),
            ConstantValue::uint(1, nv as u32),
            ConstantValue::uint(2, hk as u32),
            ConstantValue::uint(3, hv as u32),
            ConstantValue::float(4, (hk as f32).powf(-0.5)),
            ConstantValue::uint(8, u32::from(drafts)),
        ];
        constants.extend(mamba.then(|| ConstantValue::boolean(5, true)));
        constants.extend(pipelined.then(|| ConstantValue::boolean(9, true)));
        let key = PipelineKey::new("gdn_scan_varlen", "gdn_scan_simd_f32", constants);
        let pipeline = baked_build(&cache, &key).expect("gdn_scan_simd pipeline");
        let coded = |pools: &[Vec<f32>], r: std::ops::Range<usize>, seq, code| {
            let n = r.len();
            let o = buf_zero_f32(&device, n * value_dim);
            let (conv_buf, a_buf, b_buf) = (
                buf_f32(&device, &row_range(&conv, conv_dim, r.clone())),
                buf_f32(&device, &row_range(&a, nv, r.clone())),
                buf_f32(&device, &row_range(&b, nv, r)),
            );
            let state = buf_f16(&device, &pools[0]);
            let [cu, si, code] = second_of_two_coded(&device, n, seq, code);
            let mut bufs = vec![
                &o, &conv_buf, &a_buf, &b_buf, &state, &cu, &si, &code, &alog_buf, &dt_buf, &d_buf,
            ];
            bufs.extend((drafts > 0).then_some(&unread_lanes));
            let (grid, threads) = (dims(1, value_dim / 4, 2), dims(32, 4, 1));
            common::dispatch_threadgroups(&device, &pipeline, &bufs, grid, threads).then(|| {
                let after = vec![read_f16(&state, pools[0].len())];
                (read_f32(&o, n * value_dim), after)
            })
        };
        let step = |pools: &[Vec<f32>], r, seq, start| coded(pools, r, seq, plain_code(start));
        let what = format!("gdn_scan_simd, mamba {mamba}, drafts {drafts}, pipelined {pipelined}");
        let slots = [Slots::ssm(nk, nv, hk, hv)];
        if !pipelined {
            check_snapshots(&what, &slots, t, 3, &step);
            continue;
        }
        check_snapshots(&what, &slots, 1, 1, &step);
        // A plain step of more than one row is the pipelined scan's: no output, no state.
        let pools = vec![q16(&fill(POOL_SLOTS * slots[0].len))];
        let live = GdnSeqSlots::plain(GdnSlot::new(LIVE as usize));
        let Some((out, after)) = step(&pools, 0..t, live, GdnStart::Slot) else {
            return;
        };
        assert!(
            out.iter().all(|&o| o == 0.0),
            "{what}: a plain prefill chunk"
        );
        assert_bits(
            &after[0],
            &pools[0],
            &format!("{what}: a plain prefill chunk"),
        );
        // A verify step's rows are this scan's.
        if drafts > 0 {
            let Some((out, _)) = coded(&pools, 0..3, live, verify_code(2)) else {
                return;
            };
            assert!(out.iter().any(|&o| o != 0.0), "{what}: a verify step");
        }
    }
}

/// [`check_snapshots`] for `gdn_scan_pipelined_f32` (head_k 128, 12-token blocks), over 25 rows —
/// two whole blocks and a row — saving mid-step after row 12 (a whole block's last), 13 (the next
/// block's first) or 18 (inside it), and after rows 1 and 25 (the first and the partial block's
/// one). Its twin beside the simd scan (`GDN_PIPE_BESIDE_SIMD`) keeps the contract on a prefill
/// chunk, and leaves a verify step's rows and a one-row step's to the simd scan.
#[test]
fn gdn_scan_pipelined_saves_and_restores_snapshots() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv, t) = (2usize, 2usize, 128usize, 128usize, 25usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (conv, a, b) = (fill(t * conv_dim), shifted(t * nv, 3), shifted(t * nv, 7));
    let (alog_buf, dt_buf) = (
        buf_f32(&device, &fill(nv)),
        buf_f32(&device, &shifted(nv, 5)),
    );
    let mut plain_twin: Vec<(Vec<f32>, Vec<Vec<f32>>)> = Vec::new();
    for beside in [false, true] {
        let mut constants = vec![
            ConstantValue::uint(0, nk as u32),
            ConstantValue::uint(1, nv as u32),
            ConstantValue::uint(2, hk as u32),
            ConstantValue::uint(3, hv as u32),
            ConstantValue::float(4, (hk as f32).powf(-0.5)),
            ConstantValue::uint(10, 12),
        ];
        constants.extend(beside.then(|| ConstantValue::boolean(11, true)));
        let key = PipelineKey::new("gdn_scan_pipelined", "gdn_scan_pipelined_f32", constants);
        let pipeline = baked_build(&cache, &key).expect("gdn_scan_pipelined pipeline");
        let coded = |pools: &[Vec<f32>], r: std::ops::Range<usize>, seq, code| {
            let n = r.len();
            let o = buf_zero_f32(&device, n * value_dim);
            let (conv_buf, a_buf, b_buf) = (
                buf_f32(&device, &row_range(&conv, conv_dim, r.clone())),
                buf_f32(&device, &row_range(&a, nv, r.clone())),
                buf_f32(&device, &row_range(&b, nv, r)),
            );
            let state = buf_f16(&device, &pools[0]);
            let [cu, si, code] = second_of_two_coded(&device, n, seq, code);
            let bufs = [
                &o, &conv_buf, &a_buf, &b_buf, &state, &cu, &si, &code, &alog_buf, &dt_buf,
            ];
            let (grid, threads) = (dims(value_dim / 32, 1, 2), dims(128, 1, 1));
            common::dispatch_threadgroups(&device, &pipeline, &bufs, grid, threads).then(|| {
                let after = vec![read_f16(&state, pools[0].len())];
                (read_f32(&o, n * value_dim), after)
            })
        };
        let step = |pools: &[Vec<f32>], r, seq, start| coded(pools, r, seq, plain_code(start));
        let slots = [Slots::ssm(nk, nv, hk, hv)];
        let pools = vec![q16(&fill(POOL_SLOTS * slots[0].len))];
        let live = GdnSeqSlots::plain(GdnSlot::new(LIVE as usize));
        // A prefill chunk restoring a snapshot and saving mid-block and at its end: the beside
        // twin's output, live slot and snapshots are the plain twin's, bit for bit.
        let chunk = live
            .restoring(GdnSlot::new(SNAPS[0] as usize))
            .saving([save(18, SNAPS[1]), save(t, 0)]);
        let Some(run) = step(&pools, 0..t, chunk, GdnStart::Slot) else {
            return;
        };
        if !beside {
            for mid in [12, 13, 18] {
                let what = format!("gdn_scan_pipelined, mid {mid}");
                check_snapshots(&what, &slots, t, mid, &step);
            }
            plain_twin.push(run);
            continue;
        }
        let (out, after) = (&run.0, &run.1[0]);
        assert_bits(out, &plain_twin[0].0, "beside: a prefill chunk's output");
        assert_bits(
            after,
            &plain_twin[0].1[0],
            "beside: a prefill chunk's slots",
        );
        // A verify step's rows and a one-row step's are the simd scan's: no output, no state.
        for (rows, code, what) in [
            (3, verify_code(2), "a verify step"),
            (1, plain_code(GdnStart::Slot), "a one-row step"),
        ] {
            let Some((out, after)) = coded(&pools, 0..rows, live, code) else {
                return;
            };
            assert!(out.iter().all(|&o| o == 0.0), "beside: {what}");
            assert_bits(&after[0], &pools[0], &format!("beside: {what}"));
        }
    }
}

/// [`check_snapshots`] for `gdn_decode_f32`, a decode token's whole layer in one command: a step
/// is one row, so a save is of row 1 — the conv window and the scan state both, into the
/// snapshot's conv and ssm entries — and a restore reads both of the snapshot's.
#[test]
fn gdn_decode_saves_and_restores_snapshots() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache =
        SpecializedPipelineCache::new(device.clone(), &[]).expect("compile standard shaders");
    let (nk, nv, hk, hv, kernel) = (2usize, 4usize, 128usize, 128usize, 4usize);
    let (key_dim, value_dim) = (nk * hk, nv * hv);
    let conv_dim = 2 * key_dim + value_dim;
    let shifted = |n: usize, by: usize| fill(n + by)[by..].to_vec();
    let (qkv, z, conv_w) = (
        buf_f32(&device, &shifted(conv_dim, 7)),
        buf_f32(&device, &shifted(value_dim, 13)),
        buf_f32(&device, &shifted(conv_dim * kernel, 19)),
    );
    let (a, b) = (
        buf_f32(&device, &fill(nv)),
        buf_f32(&device, &shifted(nv, 3)),
    );
    let (a_log, dt_bias, norm_w) = (
        buf_f32(&device, &fill(nv)),
        buf_f32(&device, &shifted(nv, 5)),
        buf_f32(&device, &shifted(hv, 17)),
    );
    let constants = vec![
        ConstantValue::uint(0, nk as u32),
        ConstantValue::uint(1, nv as u32),
        ConstantValue::uint(2, hk as u32),
        ConstantValue::uint(3, hv as u32),
        ConstantValue::float(4, (hk as f32).powf(-0.5)),
        ConstantValue::uint(5, kernel as u32),
        ConstantValue::float(6, 1e-6),
    ];
    let key = PipelineKey::new("gdn_decode", "gdn_decode_f32", constants);
    let pipeline = baked_build(&cache, &key).expect("gdn_decode pipeline");
    let step = |pools: &[Vec<f32>], r: std::ops::Range<usize>, seq, start| {
        assert_eq!(r, 0..1, "a decode step is one row");
        let out = buf_zero_f32(&device, value_dim);
        let (conv, ssm) = (buf_f32(&device, &pools[0]), buf_f16(&device, &pools[1]));
        let [cu, si, code] = second_of_two(&device, 1, seq, start);
        let bufs = [
            &out, &qkv, &z, &a, &b, &conv_w, &conv, &ssm, &cu, &si, &code, &a_log, &dt_bias,
            &norm_w,
        ];
        let (grid, threads) = (dims(1, nk, 2), dims(1024, 1, 1));
        common::dispatch_threadgroups(&device, &pipeline, &bufs, grid, threads).then(|| {
            let after = vec![
                read_f32(&conv, pools[0].len()),
                read_f16(&ssm, pools[1].len()),
            ];
            (read_f32(&out, value_dim), after)
        })
    };
    let layouts = [Slots::conv(conv_dim, kernel), Slots::ssm(nk, nv, hk, hv)];
    check_snapshots("gdn_decode", &layouts, 1, 1, &step);
}
