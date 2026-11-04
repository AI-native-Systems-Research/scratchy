// SPDX-License-Identifier: Apache-2.0
//! A verify step's lanes at Qwen3.8-27B's full attention (24 q heads, 4 KV heads, head_dim 256,
//! bf16, 16-token pages): the PACKED NAX paged attention (a threadgroup a (KV head, sequence), the
//! group's six heads' 8-row lanes in one Q tile, `attention_verify_pack_q` first) against the
//! existing per-q-head tile (`..._wm4_wn2_bs16`), 4 sequences x 8 rows at 1000 and 4000 keys.
//! Tolerance: both round O to bf16 once from f32 accumulators over the same keys in the same order
//! per row, so they agree to bf16 rounding of the output (2^-8 relative), checked at 2e-2 absolute.
//! Times both, µs a dispatch, median of rounds.
//!
//! Run: cargo test -p scratchy-target-metal --release --test verify_packed_attention_test -- \
//!     --ignored --nocapture

mod common;

use half::bf16;
use objc2_metal::{MTLBuffer, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{
    Mtl4DispatchBatch, read_slice, shared_slice, shared_zeroed,
};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BS: usize = 16;

fn size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

fn pseudo(seed: u64, n: usize, scale: f32) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let v = ((s as u32 & 0x00FF_FFFF) as f32 / (1u32 << 23) as f32 - 1.0) * scale;
            bf16::from_f32(v).to_bits()
        })
        .collect()
}

/// The verify lanes' route (`verify_lane_attention`): pack → the packed tile split 4 ways → the
/// decode combine, against the per-query-head tile (the plain twin it replaces), every lane
/// length 1..8 over contexts from 17 keys to ~4000. Tolerance: the split reassociates each row's
/// softmax sum and P·V over 4 runs of keys in f32 (the combine rescales them by exp of the max
/// gaps), so the f32 outputs differ by a few f32 ULPs before both round to bf16 — at most one bf16
/// ULP of the output (|o| ≤ ~1 here: 2^-7), checked at 2e-2.
#[test]
fn split_verify_attention_matches_the_plain_tile() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX");
        return;
    }
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("cache");
    const SPLITS: usize = 4;
    let cases: [(&[u32], &[usize]); 2] = [
        (
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[17, 100, 1000, 4000, 33, 512, 2049, 64],
        ),
        (&[8, 1, 8, 5], &[4000, 3999, 16, 1500]),
    ];
    for (nqs, ctxs) in cases {
        let seqs = nqs.len();
        let pages = ctxs.iter().max().unwrap().div_ceil(BS);
        let consts = vec![
            ConstantValue::uint(0, HD as u32),
            ConstantValue::uint(1, NQ as u32),
            ConstantValue::uint(2, NKV as u32),
            ConstantValue::float(3, 1.0 / (HD as f32).sqrt()),
            ConstantValue::uint(4, BS as u32),
            ConstantValue::uint(5, pages as u32),
            ConstantValue::uint(6, (seqs * pages) as u32),
            ConstantValue::int(7, 0),
            ConstantValue::uint(99, 0),
        ];
        let nax = |sym: &'static str| {
            baked_build(
                &cache,
                &PipelineKey::new("attention_steel_nax_paged", sym, consts.clone()),
            )
            .expect(sym)
        };
        let plain = nax("attention_steel_nax_paged_bf16_bq64_bk32_bd256_wm4_wn2_bs16");
        let split =
            nax("attention_steel_nax_paged_bf16_bq64_bk32_bd256_wm4_wn2_bs16_packed_split4");
        let pack = nax("attention_verify_pack_q_bf16_bd256");
        let combine = baked_build(
            &cache,
            &PipelineKey::new(
                "attention",
                "attention_via_cache_v2_combine_bf16_specialized",
                vec![
                    ConstantValue::uint(0, HD as u32),
                    ConstantValue::uint(1, NQ as u32),
                    ConstantValue::uint(2, NKV as u32),
                    ConstantValue::float(3, 1.0 / (HD as f32).sqrt()),
                    ConstantValue::uint(18, SPLITS as u32),
                ],
            ),
        )
        .expect("combine");
        let mut cu = vec![0u32];
        for n in nqs {
            cu.push(cu.last().unwrap() + n);
        }
        let tokens = *cu.last().unwrap() as usize;
        let used: Vec<u32> = ctxs.iter().map(|&c| c as u32).collect();
        let bt: Vec<u32> = (0..seqs * pages).map(|i| i as u32).collect();
        let kv_elems = seqs * pages * NKV * BS * HD;
        let k = shared_slice(&device, &pseudo(3, kv_elems, 1.0));
        let v = shared_slice(&device, &pseudo(5, kv_elems, 1.0));
        let k_tab = shared_slice(&device, &[k.gpuAddress()]);
        let v_tab = shared_slice(&device, &[v.gpuAddress()]);
        let q = shared_slice(&device, &pseudo(7, tokens * NQ * HD, 2.0));
        let cu_b = shared_slice(&device, &cu);
        let used_b = shared_slice(&device, &used);
        let bt_b = shared_slice(&device, &bt);
        let qp = shared_zeroed(&device, seqs * NKV * 64 * HD * 2);
        let partials = shared_zeroed(&device, tokens * NQ * SPLITS * (HD + 2) * 4);
        let o_plain = shared_zeroed(&device, tokens * NQ * HD * 2);
        let o_split = shared_zeroed(&device, tokens * NQ * HD * 2);
        let paged = |o, qin| {
            [
                (o, 0),
                (qin, 1),
                (&cu_b, 2),
                (&used_b, 3),
                (&bt_b, 4),
                (&k_tab, 5),
                (&v_tab, 6),
            ]
        };
        let mut b = Mtl4DispatchBatch::begin(&device).expect("batch");
        b.encode(
            &plain,
            &paged(&o_plain, &q),
            &[],
            &[],
            &[&k, &v],
            size(1, NQ, seqs),
            size(256, 1, 1),
        );
        b.encode(
            &pack,
            &[(&qp, 0), (&q, 1), (&cu_b, 2)],
            &[],
            &[],
            &[],
            size(64, NKV, seqs),
            size(HD, 1, 1),
        );
        b.barrier();
        let mut bind = paged(&o_split, &qp).to_vec();
        bind.push((&partials, 10));
        b.encode(
            &split,
            &bind,
            &[],
            &[],
            &[&k, &v],
            size(SPLITS, NKV, seqs),
            size(256, 1, 1),
        );
        b.barrier();
        b.encode(
            &combine,
            &[(&o_split, 0), (&partials, 16)],
            &[],
            &[],
            &[],
            size(tokens, NQ, 1),
            size(128, 1, 1),
        );
        b.commit(true);
        let a: Vec<u16> = read_slice(&o_plain, tokens * NQ * HD);
        let p: Vec<u16> = read_slice(&o_split, tokens * NQ * HD);
        let f = |x: u16| bf16::from_bits(x).to_f32();
        let max_err = a
            .iter()
            .zip(&p)
            .map(|(&x, &y)| (f(x) - f(y)).abs())
            .fold(0f32, f32::max);
        let nonzero = p.iter().filter(|&&x| x & 0x7fff != 0).count();
        println!("lanes {nqs:?} ctx {ctxs:?}: max_err {max_err}");
        assert!(
            nonzero > p.len() / 2,
            "mostly zero output ({nonzero}/{})",
            p.len()
        );
        assert!(max_err < 2e-2, "max_err {max_err}");
    }
}
