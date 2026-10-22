// SPDX-License-Identifier: Apache-2.0
//! `argmax` and `argmax_split` against a CPU argmax: the largest value, the smallest index among
//! equal ones, NaN never chosen. `argmax_split` reads one vocabulary in two segments — a draft
//! head's prefix and the rest of its target's — and must pick what the argmax over the whole row
//! picks, its suffix's ids continuing past the prefix.

#![cfg(target_os = "macos")]

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{
    Mtl4DispatchBatch, read_slice, shared_slice, shared_zeroed,
};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

/// The argmax's threadgroup (`ARGMAX_DEFAULT_TG_SIZE`).
const THREADS: usize = scratchy_target_metal::argmax::ARGMAX_DEFAULT_TG_SIZE;

/// The id of `row`'s largest value, the smallest among equal ones, NaN never.
fn argmax_ref(row: &[f32]) -> u32 {
    let mut best = (f32::NEG_INFINITY, 0u32);
    for (i, &v) in row.iter().enumerate() {
        if v > best.0 || (v == best.0 && (i as u32) < best.1) {
            best = (v, i as u32);
        }
    }
    best.1
}

fn bits(row: &[f32]) -> Vec<u16> {
    row.iter().map(|&v| bf16::from_f32(v).to_bits()).collect()
}

/// `rows` rows of `width` bf16-representable values, each its row's own pattern.
fn rows(rows: usize, width: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| {
            (0..width)
                .map(|i| bf16::from_f32((((i * 7919 + r * 104_729) % 4093) as f32) * 1e-3).to_f32())
                .collect()
        })
        .collect()
}

/// The rows the cases plant their maxima in: `row[at] = v` for each `(row, at, v)`.
fn plant(rows: &mut [Vec<f32>], marks: &[(usize, usize, f32)]) {
    for &(r, at, v) in marks {
        rows[r][at] = v;
    }
}

/// Run `function` over `rows` split at `prefix` (`argmax` reads the whole row: `prefix` None).
fn run(rows: &[Vec<f32>], prefix: Option<usize>) -> Option<Vec<u32>> {
    let di = detect_device()?;
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let width = rows[0].len();
    let batch = rows.len();
    let segment = |from: usize, to: usize| {
        let data: Vec<u16> = rows.iter().flat_map(|r| bits(&r[from..to])).collect();
        shared_slice(&device, &data)
    };
    let out = shared_zeroed(&device, batch * 4);
    let mut b = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
    let grid = MTLSize {
        width: batch,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: THREADS,
        height: 1,
        depth: 1,
    };
    let (key, inputs) = match prefix {
        None => (
            PipelineKey::new(
                "argmax",
                "argmax_bf16",
                // ARGMAX_THREADS (slot 2): the dispatch's threads.
                vec![
                    ConstantValue::uint(0, width as u32),
                    ConstantValue::uint(2, THREADS as u32),
                ],
            ),
            vec![segment(0, width)],
        ),
        Some(p) => (
            PipelineKey::new(
                "argmax",
                "argmax_split_bf16",
                vec![
                    ConstantValue::uint(0, p as u32),
                    ConstantValue::uint(1, (width - p) as u32),
                    ConstantValue::uint(2, THREADS as u32),
                ],
            ),
            vec![segment(0, p), segment(p, width)],
        ),
    };
    let pso = baked_build(&cache, &key).expect("argmax pipeline");
    let mut binds = vec![(&inputs[0], 0), (&out, 1)];
    if let Some(suffix) = inputs.get(1) {
        binds.push((suffix, 3));
    }
    b.encode(&pso, &binds, &[(batch as u32, 2)], &[], &[], grid, threads);
    b.commit(true);
    Some(read_slice(&out, batch))
}

/// Every row's maximum somewhere distinct: in the prefix, past it, at either edge of the boundary,
/// tied across it (the prefix's smaller id wins), and beside a NaN (never picked).
fn cases(width: usize, prefix: usize) -> Vec<Vec<f32>> {
    let mut r = rows(7, width);
    plant(
        &mut r,
        &[
            (0, 3, 9.0),
            (1, width - 2, 9.0),
            (2, prefix - 1, 9.0),
            (3, prefix, 9.0),
            (4, prefix - 5, 9.0),
            (4, prefix + 5, 9.0),
            (5, prefix + 9, 9.0),
            (5, prefix + 3, f32::NAN),
            (6, 0, f32::NAN),
            (6, width - 1, 9.0),
        ],
    );
    r
}

#[test]
fn argmax_split_picks_the_whole_rows_argmax() {
    // A wide row across many threads' strides, and a prefix that is no multiple of 4.
    for (width, prefix) in [(248_320, 65_536), (10_000, 4_097)] {
        let r = cases(width, prefix);
        let Some(got) = run(&r, Some(prefix)) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let want: Vec<u32> = r.iter().map(|row| argmax_ref(row)).collect();
        assert_eq!(got, want, "width {width}, prefix {prefix}");
    }
}

#[test]
fn argmax_picks_the_rows_argmax() {
    for width in [248_320, 10_001] {
        let r = cases(width, width / 2);
        let Some(got) = run(&r, None) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let want: Vec<u32> = r.iter().map(|row| argmax_ref(row)).collect();
        assert_eq!(got, want, "width {width}");
    }
}
