// SPDX-License-Identifier: Apache-2.0
//! Correctness test for the `ban_after` kernel: a two-token bad word's ban fires at a row whose
//! input token is the word's first, read on the device, and leaves every other logit as it was; a
//! job that bans always fires whatever the input token.
//!
//! Dispatches on the production MTL4 path (see `common::dispatch_threadgroups`).

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::ban_after::{ALWAYS, BAN_AFTER_TG_SIZE, BanAfterKernel, BanJob};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::PipelineKey;
use scratchy_target_metal::tape::ids::LogitsWidth;
use scratchy_target_metal::tape::kernel_constants::BanAfterConstants;

/// Three sampled rows over a 64-token vocab; input tokens `[101, 5, 101]` with the rows at step
/// rows 0, 1 and 2. A `[101 → 40]` ban fires at rows 0 and 2 and not at row 1; an always-ban of
/// token 7 fires at row 1 alone, the job naming it. Nothing else moves.
#[test]
fn ban_after_fires_where_the_input_token_matches() {
    let Some(dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = dev.device;
    let vocab = 64u32;
    let rows = 3usize;
    let key = PipelineKey::new(
        "ban_after",
        "ban_after_bf16",
        BanAfterConstants {
            vocab: LogitsWidth(vocab),
        }
        .into(),
    );
    let baked = scratchy_target_metal::aot::baked_kernels(&[key]);
    let kernels = scratchy_target_metal::off_tape::OffTapePipeline::new(&device, &baked[0])
        .map(|ban| BanAfterKernel { ban })
        .expect("ban_after kernel");

    let host: Vec<bf16> = (0..rows)
        .flat_map(|_| (0..vocab).map(|v| bf16::from_f32(v as f32)))
        .collect();
    let logits = common::shared_slice(&device, &host);
    let input_ids = common::shared_slice(&device, &[101u32, 5, 101]);
    let after = |row, input| BanJob {
        row,
        input,
        after: 101,
        banned: 40,
    };
    let jobs = [
        after(0, 0),
        after(1, 1),
        after(2, 2),
        BanJob {
            row: 1,
            input: 1,
            after: ALWAYS,
            banned: 7,
        },
    ];
    let words: Vec<u32> = std::iter::once(jobs.len() as u32)
        .chain(jobs.iter().flat_map(|j| j.words()))
        .collect();
    let jobs_buf = common::shared_slice(&device, &words);

    if !common::dispatch_threadgroups(
        &device,
        &kernels.ban,
        &[&logits, &input_ids, &jobs_buf],
        MTLSize {
            width: jobs.len().div_ceil(BAN_AFTER_TG_SIZE),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: BAN_AFTER_TG_SIZE,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let got: Vec<f32> = common::read_slice::<bf16>(&logits, rows * vocab as usize)
        .into_iter()
        .map(|x| x.to_f32())
        .collect();
    let banned = [(0, 40), (2, 40), (1, 7)];
    for r in 0..rows {
        for v in 0..vocab as usize {
            let val = got[r * vocab as usize + v];
            match banned.contains(&(r, v)) {
                true => assert!(val == f32::NEG_INFINITY, "row {r} token {v} not banned"),
                false => assert_eq!(val, v as f32, "row {r} token {v} changed"),
            }
        }
    }
}
