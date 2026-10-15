// The grouped MoE path's counting sort (`moe_group.metal`): `moe_group_offsets` histograms a step's
// (token, expert) pairs by expert and lays out each expert's run padded to the grouped GEMM's
// 64-row tile. Its histogram holds every expert of the model — Gemma-4's 128 and Qwen3.5/3.6's 256.
mod common;

use objc2_metal::MTLSize;
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::constants::{ConstSlot, ConstantValue};

const BM: u32 = 64;

/// Deterministic expert ids: `tokens` rows of `top_k` distinct experts in `0..experts`.
fn routes(tokens: u32, top_k: u32, experts: u32) -> Vec<u32> {
    let mut state = 0x5eedu64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    (0..tokens)
        .flat_map(|_| {
            let mut row: Vec<u32> = Vec::new();
            while row.len() < top_k as usize {
                let e = next() % experts;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            row
        })
        .collect()
}

#[test]
fn offsets_count_every_expert() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    for (experts, tokens) in [(128u32, 64u32), (256, 64), (256, 1024)] {
        let top_k = 8;
        let pairs = tokens * top_k;
        let inds = routes(tokens, top_k, experts);
        let constants = vec![
            ConstantValue::int(ConstSlot(0), pairs as i32),
            ConstantValue::int(ConstSlot(1), experts as i32),
            ConstantValue::int(ConstSlot(6), BM as i32),
        ];
        let pso = baked_pipeline(&device, "moe_group", "moe_group_offsets", constants)
            .expect("moe_group_offsets");
        let inds_buf = common::shared_slice(&device, &inds);
        let count = common::shared_zeroed(&device, experts as usize * 4);
        let offset = common::shared_zeroed(&device, experts as usize * 4);
        let total = common::shared_zeroed(&device, 4);
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        assert!(common::dispatch_threadgroups(
            &device,
            &pso,
            &[&inds_buf, &count, &offset, &total],
            one,
            threads,
        ));
        let mut want_count = vec![0u32; experts as usize];
        for &e in &inds {
            want_count[e as usize] += 1;
        }
        let want_offset: Vec<u32> = (want_count.iter())
            .scan(0u32, |acc, &c| {
                let at = *acc;
                *acc += c.div_ceil(BM) * BM;
                Some(at)
            })
            .collect();
        let want_total: u32 = want_count.iter().map(|c| c.div_ceil(BM) * BM).sum();
        let what = format!("{experts} experts, {tokens} tokens");
        let got_count: Vec<u32> = common::read_slice(&count, experts as usize);
        assert_eq!(got_count, want_count, "{what}: counts");
        let got_offset: Vec<u32> = common::read_slice(&offset, experts as usize);
        assert_eq!(got_offset, want_offset, "{what}: offsets");
        assert_eq!(
            common::read_slice::<u32>(&total, 1),
            [want_total],
            "{what}: total"
        );
    }
}
