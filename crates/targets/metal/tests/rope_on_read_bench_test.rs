// SPDX-License-Identifier: Apache-2.0
//! Rope-on-read PARITY bench (general, model-independent).
//!
//! The perf guard the user mandated: enabling ROPE_ON_READ compiles the
//! decode kernel WITH the per-key flag-load + branch in the K-loop, paid
//! by EVERY spans-enabled model's decode even on non-span requests. This
//! times the SAME kernel (attention_via_cache_v2) ROR-off vs ROR-on with
//! an ALL-ZERO flag buffer (the non-span case) on the real device and
//! asserts the rope-on-read specialization does not regress the hot path.
//! Each side's time is the median GPU time per dispatch over interleaved
//! command buffers of 100 dispatches. Prints the ratio; fails on a gross
//! regression — the bar, classed by GPU family: on M5 (NAX) the kernel-only
//! ratio is stable at ~1.08× and the bar stays tight, but on pre-NAX chips
//! the machinery cost is both larger and machine-state-bimodal (M1 Max:
//! 0.99–1.25× machinery, 2.07–2.68× span — the flag-test overhead hides
//! under memory latency when the GPU is throttled, and shows at full
//! clock), so the bar there must hold every observed state.
//!
//! A test binary of its own, so it times with the GPU to itself: cargo runs test binaries one
//! at a time, and a binary's tests on parallel threads (#795).

mod common;
mod kernel_inputs;

use common::{Buffer, Device, Mtl4DispatchBatch, Pipeline, shared_slice, shared_zeroed};
use half::f16;
use kernel_inputs::{build_cos_sin, pseudo};
use objc2_metal::{MTLBuffer, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

/// `data` rounded to f16, in a shared buffer.
fn buf_f16(device: &Device, data: &[f32]) -> Buffer {
    let h: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    shared_slice(device, &h)
}

#[test]
fn rope_on_read_decode_parity_bench() {
    let Some(di) = detect_device() else {
        eprintln!("skip rope_on_read_decode_parity_bench: no metal device");
        return;
    };
    let device = di.device.clone();
    let (machinery_bar, rot_bar) =
        if scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
            (1.15, 2.0)
        } else {
            (1.35, 3.0)
        };
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");

    let head_dim = 128usize; // Llama/Qwen-class (general, not gemma4-specific)
    let num_heads = 32usize;
    let num_kv = 8usize;
    let block_size = 16usize;
    let kv_len = 8192usize; // long context → flag-load-per-key cost is visible
    let rot_dim = 128usize;
    let pair_off = 64usize;
    let num_blocks = kv_len.div_ceil(block_size);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_elems = num_blocks * num_kv * block_size * head_dim;

    let k_buf = buf_f16(&device, &pseudo(101, kv_elems, 1.0));
    let v_buf = buf_f16(&device, &pseudo(103, kv_elems, 1.0));
    let k_tab = shared_slice(&device, &[k_buf.gpuAddress()]);
    let v_tab = shared_slice(&device, &[v_buf.gpuAddress()]);
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = shared_slice(&device, &block_table);
    // Flagged block_table (every entry bit-31 set) → do_rot=true → EVERY cached
    // key is re-roped on read. This is the REAL span decode cost (launch-claude
    // re-ropes its whole cached span every token), vs the all-clear case which
    // only measures the bit-packing machinery.
    let bt_flagged: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf_flagged = shared_slice(&device, &bt_flagged);
    let q_buf = buf_f16(&device, &pseudo(107, num_heads * head_dim, 1.0));
    let out_buf = shared_zeroed(&device, num_heads * head_dim * 2);
    let seq_used = shared_slice(&device, &[kv_len as u32]);
    // Non-span case: block_table has no bit-31 flags set (bt_buf above is
    // plain physical ids), so do_rot is always false on the ROR-on path.
    let cos_sin = buf_f16(&device, &build_cos_sin(kv_len, rot_dim));

    let base_consts = || {
        vec![
            ConstantValue::uint(0, head_dim as u32),
            ConstantValue::uint(1, num_heads as u32),
            ConstantValue::uint(2, num_kv as u32),
            ConstantValue::float(3, scale),
            ConstantValue::uint(4, block_size as u32),
            ConstantValue::uint(5, num_blocks as u32),
            ConstantValue::uint(6, 0),
            ConstantValue::int(7, 0),
        ]
    };
    let off_pipe = baked_build(
        &cache,
        &PipelineKey::new(
            "attention",
            "attention_via_cache_v2_f16_specialized",
            base_consts(),
        ),
    )
    .expect("ROR-off pipeline");
    let mut on_consts = base_consts();
    on_consts.push(ConstantValue::uint(8, rot_dim as u32));
    on_consts.push(ConstantValue::uint(9, pair_off as u32));
    on_consts.push(ConstantValue::uint(10, 1));
    let on_pipe = baked_build(
        &cache,
        &PipelineKey::new(
            "attention",
            "attention_via_cache_v2_f16_specialized",
            on_consts,
        ),
    )
    .expect("ROR-on pipeline");

    // One sample: PER_CB dispatches of one side in ONE command buffer, a barrier after each, timed
    // to completion — the kernel's GPU time. A command buffer per dispatch would add a new queue,
    // a wired residency set and an event wait to every sample: host costs larger and noisier than
    // the kernel. ROR-on binds cos_sin at slot 6 (the flag rides in block_table bit 31); K/V are
    // reached via raw gpuAddress → resident, not bound.
    const PER_CB: usize = 100;
    let sample = |pipeline: &Pipeline, rope: Option<&Buffer>, bt: &Buffer| -> Option<f64> {
        let mut binds = vec![
            (&out_buf, 0),
            (&q_buf, 1),
            (&seq_used, 2),
            (bt, 3),
            (&k_tab, 4),
            (&v_tab, 5),
        ];
        binds.extend(rope.map(|cos_sin| (cos_sin, 6)));
        let grid = MTLSize {
            width: 1,
            height: num_heads,
            depth: 1,
        };
        let threads = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        let mut batch = Mtl4DispatchBatch::begin(&device)?;
        for _ in 0..PER_CB {
            batch.encode(pipeline, &binds, &[], &[], &[&k_buf, &v_buf], grid, threads);
            batch.barrier();
        }
        let t = std::time::Instant::now();
        batch.commit(true);
        Some(t.elapsed().as_secs_f64() * 1000.0 / PER_CB as f64)
    };
    let sides = [
        (&off_pipe, None, &bt_buf),
        (&on_pipe, Some(&cos_sin), &bt_buf),
        (&on_pipe, Some(&cos_sin), &bt_buf_flagged),
    ];
    // Warm up every side, then rounds in rotating side order so clock ramps and drift land on all
    // three alike; each side's median sample.
    for (pipeline, rope, bt) in sides {
        if sample(pipeline, rope, bt).is_none() {
            return;
        }
    }
    const ROUNDS: usize = 9;
    let rounds: [[f64; 3]; ROUNDS] = std::array::from_fn(|round| {
        let mut ms = [0f64; 3];
        for k in 0..sides.len() {
            let s = (round + k) % sides.len();
            let (pipeline, rope, bt) = sides[s];
            ms[s] = sample(pipeline, rope, bt).expect("MTL4 queue");
        }
        ms
    });
    let [off_ms, on_ms, rot_ms]: [f64; 3] = std::array::from_fn(|s| {
        let mut ms = rounds.map(|round| round[s]);
        ms.sort_by(f64::total_cmp);
        ms[ROUNDS / 2]
    });
    let ratio = on_ms / off_ms;
    let rot_ratio = rot_ms / off_ms;
    eprintln!(
        "ROPE-ON-READ DECODE OVERHEAD (via_cache_v2, hd{head_dim} {num_heads}q/{num_kv}kv \
         kv_len={kv_len}):\n  baseline (rotate-on-write, read-as-is): {off_ms:.4} ms\n  \
         rope-on-read, NO keys rotate (machinery only): {on_ms:.4} ms  ({ratio:.3}x)\n  \
         rope-on-read, ALL keys re-roped (real span cost): {rot_ms:.4} ms  ({rot_ratio:.3}x)"
    );
    assert!(
        ratio < machinery_bar,
        "rope-on-read regressed the non-span decode hot path: {ratio:.3}x (bar {machinery_bar}x)"
    );
    // Re-roping every cached key is the feature's inherent decode cost; bar
    // catches a gross regression in the rotation path (simd_shuffle + rope).
    assert!(
        rot_ratio < rot_bar,
        "rope-on-read rotation path regressed: {rot_ratio:.3}x (bar {rot_bar}x)"
    );
}
