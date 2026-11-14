// SPDX-License-Identifier: Apache-2.0
//! The grouped NAX expert GEMM (`affine_gather_qmm_t_nax`, both W dequant schedules) writes every
//! live row however many padded rows the bake holds. A grouped bake pads each expert's run to 64
//! rows, so its row bound is `bucket·top_k + 63·experts`: nemotron-3-super's 512 experts put it
//! past 32,767 at any prefill bucket over 64 rows, where the tile's remaining-row count once wrapped
//! as a `short` and the first tiles stored nothing (#1205).

use half::bf16;
use objc2_metal::{MTLBuffer as _, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{Mtl4DispatchBatch, shared_slice};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::kernel_constants::AffineCodes;
use scratchy_target_metal::tape::quantized::NaxDequant;

const GROUP: usize = 64;

#[test]
fn every_live_row_is_written_past_32k_padded_rows() {
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
    // Two experts' runs of 16,400 rows, each padded to 16,448, then a sentinel tile: 32,960 rows.
    let (n, k, counts) = (64usize, 64usize, [16_400usize, 16_400]);
    let experts = counts.len();
    let runs: Vec<usize> = counts.iter().map(|c| c.div_ceil(64) * 64).collect();
    let m = runs.iter().sum::<usize>() + 64;
    assert!(
        m > i16::MAX as usize,
        "the layout must pass a short's range"
    );
    let (mut indices, mut live) = (vec![experts as u32; m], Vec::new());
    let mut at = 0;
    for (e, (&c, &run)) in counts.iter().zip(&runs).enumerate() {
        indices[at..at + c].fill(e as u32);
        live.extend(at..at + c);
        at += run;
    }
    // 4-bit codes as written: each byte two inputs, the even one in the low nibble.
    let code = |e: usize, r: usize, kk: usize| ((e * 7 + r * 13 + kk * 5) % 16) as u8;
    let packed: Vec<u8> = (0..experts * n * k / 2)
        .map(|i| {
            let (e, r, j) = (i / (n * k / 2), i / (k / 2) % n, i % (k / 2));
            code(e, r, 2 * j) | code(e, r, 2 * j + 1) << 4
        })
        .collect();
    let groups = experts * n * k / GROUP;
    let scale = |g: usize| 0.01 + 0.002 * (g % 7) as f32;
    let bias = |g: usize| -0.05 + 0.01 * (g % 5) as f32;
    let bits = |v: f32| bf16::from_f32(v).to_bits();
    let xval = |r: usize, kk: usize| {
        bf16::from_f32(((r * 31 + kk * 17) % 23) as f32 * 0.05 - 0.5).to_f32()
    };
    let w = shared_slice(&device, &packed);
    let s = shared_slice(
        &device,
        &(0..groups).map(|g| bits(scale(g))).collect::<Vec<_>>(),
    );
    let b = shared_slice(
        &device,
        &(0..groups).map(|g| bits(bias(g))).collect::<Vec<_>>(),
    );
    let x = shared_slice(
        &device,
        &(0..m * k)
            .map(|i| bits(xval(i / k, i % k)))
            .collect::<Vec<_>>(),
    );
    let ix = shared_slice(&device, &indices);
    for dequant in NaxDequant::ALL {
        let symbol: &'static str = Box::leak(
            format!(
                "affine_gather_{}_bf16_s_bf16_gs_{GROUP}_b_4_alN_true_batch_0",
                dequant.symbol_stem()
            )
            .into_boxed_str(),
        );
        let mut constants = vec![
            ConstantValue::int(0, k as i32),
            ConstantValue::int(1, n as i32),
            ConstantValue::int(2, m as i32),
            ConstantValue::int(4, experts as i32),
        ];
        constants.extend(AffineCodes::AsWritten.constants());
        let key = PipelineKey::new("quantized_qmm_nax", symbol, constants);
        let pso = baked_build(&cache, &key).expect(symbol);
        // NaN everywhere first: a live row the kernel leaves unwritten stays NaN.
        let y = shared_slice(&device, &vec![0x7fc0u16; m * n]);
        let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4");
        let (tx, ty, tz) = dequant.threads();
        batch.encode(
            &pso,
            &[(&w, 0), (&s, 1), (&b, 2), (&x, 3), (&y, 4), (&ix, 5)],
            &[],
            &[],
            &[],
            MTLSize {
                width: n / 64,
                height: m.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: tx as usize,
                height: ty as usize,
                depth: tz as usize,
            },
        );
        batch.commit(true);
        let got = unsafe { std::slice::from_raw_parts(y.contents().as_ptr() as *const u16, m * n) };
        for &r in &live {
            let e = indices[r] as usize;
            for c in 0..n {
                let want: f32 = (0..k)
                    .map(|kk| {
                        let g = (e * n + c) * (k / GROUP) + kk / GROUP;
                        let wv = bf16::from_f32(scale(g)).to_f32() * f32::from(code(e, c, kk))
                            + bf16::from_f32(bias(g)).to_f32();
                        xval(r, kk) * wv
                    })
                    .sum();
                let g = bf16::from_bits(got[r * n + c]).to_f32();
                assert!(
                    (g - want).abs() <= 0.02 * want.abs().max(1.0),
                    "{symbol}: live row {r} (tile {}) col {c}: got {g}, want {want}",
                    r / 64
                );
            }
        }
    }
}
