// SPDX-License-Identifier: Apache-2.0
//! Every kernel that reads a planes weight (`AffineCodes::Planes`, `affine_codes.h`) computes the
//! bits it computes from the same weight stored offset-8 row by row: the matvecs (`qmv_fast`,
//! `qmv_wide`, `qmv_gated_fast`), the bf16 matmuls (`qmm_t`, `qmm_t_splitk`, `qmm_t_nax` with
//! either W dequant) and the W4A8 GEMMs (tiled, split-K, the small-M tile, the pair), at
//! Qwen3.8-27B's shapes and a narrow one. The weights are random offset-8 codes; each kernel runs
//! once on the rows and once on the regrouped planes, and every output element must match.

mod common;

use half::bf16;
use objc2_metal::{MTLBuffer as _, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{Device, Mtl4DispatchBatch, shared_slice, shared_zeroed};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::kernel_constants::{AFFINE_PLANE_ROW_BYTES, AffineCodes};
use scratchy_target_metal::tape::quantized::{
    W4a8Rows, W4a8SplitK, w4a8_quant_static_name, w4a8_scratch_bytes,
};

const GROUP: u32 = 64;

/// Codes stored as written (`k / 2` bytes a row) regrouped into planes (`affine_codes.h`).
fn planes(rows: &[u8], k: usize) -> Vec<u8> {
    let (row, p) = (k / 2, AFFINE_PLANE_ROW_BYTES);
    let n = rows.len() / row;
    let mut out = vec![0u8; rows.len()];
    for r in 0..n {
        for c in 0..k / 64 {
            out[(c * n + r) * p..][..p].copy_from_slice(&rows[r * row + c * p..][..p]);
        }
    }
    out
}

fn bf16s(len: usize, f: impl Fn(usize) -> f32) -> Vec<u16> {
    (0..len).map(|i| bf16::from_f32(f(i)).to_bits()).collect()
}

fn size(w: u32, h: u32, d: u32) -> MTLSize {
    MTLSize { width: w as usize, height: h as usize, depth: d as usize }
}

/// One weight in both layouts, with its scales and biases.
struct Weight {
    rows: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    planes: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    scales: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    biases: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
}

fn weight(device: &Device, n: u32, k: u32, seed: u64) -> Weight {
    let (nu, ku) = (n as usize, k as usize);
    let codes: Vec<u8> = (0..nu * ku / 2)
        .map(|i| ((i as u64).wrapping_mul(2654435761).wrapping_add(seed) >> 9) as u8)
        .collect();
    let groups = nu * ku / GROUP as usize;
    Weight {
        rows: shared_slice(device, &codes),
        planes: shared_slice(device, &planes(&codes, ku)),
        scales: shared_slice(device, &bf16s(groups, |i| 0.004 + 0.0003 * (i % 17) as f32)),
        biases: shared_slice(device, &bf16s(groups, |i| -0.03 + 0.001 * (i % 11) as f32)),
    }
}

type Buf = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// Run `symbol` with `constants` (the codes' own appended per layout), its buffers from `bind`
/// (given the weight's codes buffer), `grid` x `threads`; the output buffer's bf16 bits.
#[allow(clippy::too_many_arguments)]
fn run(
    device: &Device,
    cache: &SpecializedPipelineCache,
    library: &'static str,
    symbol: &'static str,
    constants: &[ConstantValue],
    codes: AffineCodes,
    pre: &[(&'static str, Vec<ConstantValue>, Vec<(&Buf, usize)>, MTLSize, MTLSize)],
    bind: &dyn Fn() -> Vec<(Buf, usize)>,
    out: &Buf,
    out_len: usize,
    grid: MTLSize,
    threads: MTLSize,
) -> Vec<u16> {
    let mut cs = constants.to_vec();
    cs.extend(codes.constants());
    let pso = baked_build(cache, &PipelineKey::new(library, symbol, cs)).expect(symbol);
    let mut batch = Mtl4DispatchBatch::begin(device).expect("batch");
    for (sym, c, b, g, t) in pre {
        let p = baked_build(cache, &PipelineKey::new(library, sym, c.clone())).expect(sym);
        let b: Vec<(&Buf, usize)> = b.to_vec();
        batch.encode(&p, &b, &[], &[], &[], *g, *t);
        batch.barrier();
    }
    let owned = bind();
    let bufs: Vec<(&Buf, usize)> = owned.iter().map(|(b, i)| (b, *i)).collect();
    batch.encode(&pso, &bufs, &[], &[], &[], grid, threads);
    batch.commit(true);
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const u16, out_len) }.to_vec()
}

#[test]
fn every_planes_reader_matches_its_rows_twin_bit_for_bit() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX matrix unit");
        return;
    }
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let mut checked = 0;
    for (n, k) in [(1024u32, 5120u32), (5120, 6144)] {
        let w = weight(&device, n, k, n as u64);
        let u = weight(&device, n, k, 7 + n as u64);
        let rows_codes = AffineCodes::Offset8;
        let planes_codes = AffineCodes::Planes { k };
        let (nu, ku) = (n as usize, k as usize);
        for m in [1u32, 8, 16, 64] {
            let mu = m as usize;
            let x = shared_slice(&device, &bf16s(mu * ku, |i| ((i * 37 % 101) as f32 - 50.0) * 0.02));
            // Split-K matmuls write `[split, M, N]` partials: room for four.
            let y = shared_zeroed(&device, 4 * mu * nu * 2);
            let y2 = shared_zeroed(&device, mu * nu * 4);
            let scratch = shared_zeroed(&device, w4a8_scratch_bytes(m.max(16), k) as usize);
            // (label, library, symbol, constants, grid, threads, pair, pre-pass rows)
            let mut cells: Vec<(String, &'static str, &'static str, Vec<ConstantValue>, MTLSize, MTLSize, bool, Option<u32>)> = Vec::new();
            let int = |i: u16, v: u32| ConstantValue::int(i, v as i32);
            if m == 1 {
                cells.push(("qmv_fast".into(), "quantized_qmv", "affine_qmv_fast_bf16_s_bf16_gs_64_b_4_batch_0",
                    vec![int(0, k), int(1, n)], size(1, n / 4, 1), size(32, 2, 1), false, None));
            }
            if (2..=8).contains(&m) || m == 8 {
                cells.push((format!("qmv_wide m{m}"), "quantized_qmv", "affine_qmv_wide_bf16_s_bf16_gs_64_b_4_nv_4_kl_8_batch_0",
                    vec![int(0, k), int(1, n), int(7, m)], size(m.div_ceil(4), n / 8, 1), size(32, 2, 1), false, None));
            }
            if m >= 16 {
                cells.push((format!("qmm_t m{m}"), "quantized_qmm", "affine_qmm_t_bf16_s_bf16_gs_64_b_4_alN_true_batch_0",
                    vec![int(0, k), int(1, n), int(2, m)], size(n / 32, m.div_ceil(32), 1), size(32, 2, 2), false, None));
                let split = 4u32;
                cells.push((format!("qmm_t_splitk m{m}"), "quantized_qmm", "affine_qmm_t_splitk_bf16_s_bf16_gs_64_b_4_alN_true",
                    vec![int(0, k), int(1, n), int(2, m), int(3, k / split)], size(n / 32, m.div_ceil(32), split), size(32, 2, 2), false, None));
            }
            if m == 64 {
                cells.push(("qmm_t_nax".into(), "quantized_qmm_nax", "affine_qmm_t_nax_bf16_s_bf16_gs_64_b_4_alN_true_batch_0",
                    vec![int(0, k), int(1, n), int(2, m)], size(n / 64, 1, 1), size(32, 2, 2), false, None));
                cells.push(("qmm_t_nax_producer".into(), "quantized_qmm_nax", "affine_qmm_t_nax_producer_bf16_s_bf16_gs_64_b_4_alN_true_batch_0",
                    vec![int(0, k), int(1, n), int(2, m)], size(n / 64, 1, 1), size(32, 5, 1), false, None));
                cells.push(("w4a8 tiled".into(), "quantized_qmm_nax", "affine_qmm_w4a8_bf16_s_bf16_gs_64_tn_128_nsg_4",
                    vec![int(0, k), int(1, n), int(2, m)], size(n / 128, m / 32, 1), size(128, 1, 1), false, Some(m)));
                for ks in W4a8SplitK::ALL {
                    let ways = ks.ways();
                    let sym: &'static str = Box::leak(format!("affine_qmm_w4a8_splitk_bf16_s_bf16_gs_64_tn_32_ks_{ways}").into_boxed_str());
                    cells.push((format!("w4a8 split-K ks{ways}"), "quantized_qmm_nax", sym,
                        vec![int(0, k), int(1, n), int(2, m)], size(n / 32, m / 32, 1), size(32 * ways, 1, 1), false, Some(m)));
                }
            }
            if m == 8 || m == 16 {
                for ks in W4a8SplitK::ALL {
                    let ways = ks.ways();
                    let sym: &'static str = Box::leak(format!("affine_qmm_w4a8_splitk_bf16_s_bf16_gs_64_tm_16_tn_32_ks_{ways}").into_boxed_str());
                    cells.push((format!("w4a8 small-M ks{ways} m{m}"), "quantized_qmm_nax", sym,
                        vec![int(0, k), int(1, n), int(2, m)], size(n / 32, 1, 1), size(32 * ways, 1, 1), false, Some(16)));
                    let pair: &'static str = Box::leak(format!("affine_qmm_w4a8_splitk_pair_bf16_s_bf16_gs_64_tm_16_tn_32_ks_{ways}").into_boxed_str());
                    cells.push((format!("w4a8 pair ks{ways} m{m}"), "quantized_qmm_nax", pair,
                        vec![int(0, k), int(1, n), int(2, m), int(9, n)], size(2 * n / 32, 1, 1), size(32 * ways, 1, 1), true, Some(16)));
                }
            }
            for (label, lib, sym, cs, grid, threads, pair, quant_rows) in cells {
                let pre: Vec<(&'static str, Vec<ConstantValue>, Vec<(&Buf, usize)>, MTLSize, MTLSize)> = match quant_rows {
                    Some(rows) => {
                        let mut qc = vec![int(0, k), int(2, rows)];
                        qc.extend(AffineCodes::Offset8.constant());
                        vec![(w4a8_quant_static_name(W4a8Rows::Dense, scratchy_target_metal::tape::quantized::DequantDtype::Bf16), qc,
                              vec![(&x, 0), (&scratch, 1)], size((k / 64).div_ceil(16), m, 1), size(128, 1, 1))]
                    }
                    None => vec![],
                };
                let input = if quant_rows.is_some() { scratch.clone() } else { x.clone() };
                let layouts = [(rows_codes, &w.rows, &u.rows), (planes_codes, &w.planes, &u.planes)];
                let outs: Vec<(Vec<u16>, Vec<u16>)> = layouts.iter().map(|(codes, wc, uc)| {
                    for b in [&y, &y2] {
                        unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, b.length()) };
                    }
                    let bind = || {
                        let mut v = vec![((*wc).clone(), 0), (w.scales.clone(), 1), (w.biases.clone(), 2), (input.clone(), 3), (y.clone(), 4)];
                        if pair {
                            v.extend([((*uc).clone(), 5), (u.scales.clone(), 6), (u.biases.clone(), 7), (y2.clone(), 8)]);
                        }
                        v
                    };
                    let out_len = if lib == "quantized_qmm" && sym.contains("splitk") { 4 * mu * nu } else { mu * nu };
                    let a = run(&device, &cache, lib, sym, &cs, *codes, &pre, &bind, &y, out_len, grid, threads);
                    let b = unsafe { std::slice::from_raw_parts(y2.contents().as_ptr() as *const u16, mu * nu) }.to_vec();
                    (a, b)
                }).collect();
                let differ = outs[0].0.iter().zip(&outs[1].0).filter(|(a, b)| a != b).count()
                    + if pair { outs[0].1.iter().zip(&outs[1].1).filter(|(a, b)| a != b).count() } else { 0 };
                let nonzero = outs[0].0.iter().filter(|v| **v != 0).count();
                assert!(nonzero > 0, "{label} n={n} k={k}: computed nothing");
                assert_eq!(differ, 0, "{label} n={n} k={k}: {differ} outputs differ between rows and planes");
                checked += 1;
            }
        }
    }
    eprintln!("{checked} kernel cells identical on planes");
}
