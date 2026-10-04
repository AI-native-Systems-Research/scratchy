//! Correctness test for the Q/O layout convert kernels
//! (`attention_layout_convert.metal`). Validates the token-major <-> head-major
//! transposes (round-trip) against a CPU reference, through the production bf16
//! kernels (`q_convert_prod_bf16` / `o_convert_prod_bf16`; small integers are
//! exact in bf16).
//!
//! Dispatches on the production MTL4 path (see `common::dispatch_threadgroups`).
//! Kernel binding contract: `buffer(0)=out, buffer(1)=in`, `[Lq, H, D]` compiled
//! in (slots 0 / 1 / 2); grid is one threadgroup per (token, head).

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

const LQ: u32 = 3;
const H: u32 = 2;
const D: u32 = 4;

#[test]
fn layout_convert_roundtrip_matches_reference() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let dev: &common::Device = &device.device;
    let constants = || {
        let dims = [LQ, H, D].into_iter().zip(0u16..);
        dims.map(|(v, slot)| ConstantValue::uint(slot, v)).collect()
    };
    let pipeline = |function| {
        baked_pipeline(dev, "attention_layout_convert", function, constants()).expect(function)
    };

    let n = (LQ * H * D) as usize;
    // token-major input [Lq, H, D]
    let tok: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let tok16: Vec<u16> = tok.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();

    let grid = MTLSize {
        width: LQ as usize,
        height: H as usize,
        depth: 1,
    };
    let threads = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    // q_convert: token-major -> head-major
    let in_buf = common::shared_slice(dev, &tok16);
    let head_buf = common::shared_zeroed(dev, n * std::mem::size_of::<u16>());
    let q_pso = pipeline("q_convert_prod_bf16");
    if !common::dispatch_threadgroups(dev, &q_pso, &[&head_buf, &in_buf], grid, threads) {
        return;
    }
    let head: Vec<u16> = common::read_slice(&head_buf, n);
    for i in 0..LQ {
        for h in 0..H {
            for d in 0..D {
                let got = bf16::from_bits(head[((h * LQ + i) * D + d) as usize]).to_f32();
                let want = tok[((i * H + h) * D + d) as usize];
                assert_eq!(got, want, "q_convert i={i} h={h} d={d}");
            }
        }
    }

    // o_convert: head-major -> token-major (round-trip back to `tok`)
    let back_buf = common::shared_zeroed(dev, n * std::mem::size_of::<u16>());
    let o_pso = pipeline("o_convert_prod_bf16");
    if !common::dispatch_threadgroups(dev, &o_pso, &[&back_buf, &head_buf], grid, threads) {
        return;
    }
    let back: Vec<u16> = common::read_slice(&back_buf, n);
    for k in 0..n {
        let got = bf16::from_bits(back[k]).to_f32();
        assert_eq!(got, tok[k], "o_convert round-trip at {k}");
    }
}
