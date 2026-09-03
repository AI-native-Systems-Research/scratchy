//! Correctness test for the Q/O layout+dtype convert kernels
//! (`attention_layout_convert.metal`). Validates the token-major <-> head-major
//! transposes (round-trip) against a CPU reference. Uses the f16 parity
//! variants for exactness; the bf16 variants share the identical index math.
//!
//! Dispatches on the production MTL4 path (see `common::dispatch_threadgroups`).
//! Kernel binding contract: `buffer(0)=out, buffer(1)=in, buffer(2)=params`
//! (`[Lq, H, D]`); grid is one threadgroup per (token, head).

mod common;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};
use scratchy_target_metal::detect_device;

const LQ: u32 = 3;
const H: u32 = 2;
const D: u32 = 4;

type PipelineOwned = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Build a compute pipeline for `name` from an already-compiled library.
fn pipeline_for(
    dev: &common::Device,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> PipelineOwned {
    let func = library
        .newFunctionWithName(&NSString::from_str(name))
        .unwrap_or_else(|| panic!("{name}"));
    dev.newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline")
}

#[test]
fn layout_convert_roundtrip_matches_reference() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let dev: &common::Device = &device.device;
    let src = include_str!("../shaders/attention_layout_convert.metal");
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = dev
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
        .expect("compile attention_layout_convert.metal");

    let n = (LQ * H * D) as usize;
    // token-major input [Lq, H, D]
    let tok: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let tok16: Vec<u16> = tok.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let params: [u32; 3] = [LQ, H, D];
    let params_buf = common::shared_slice(dev, &params);

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
    let q_pso = pipeline_for(dev, &library, "q_convert_f16_to_f16");
    if !common::dispatch_threadgroups(
        dev,
        &q_pso,
        &[&head_buf, &in_buf, &params_buf],
        grid,
        threads,
    ) {
        return;
    }
    let head: Vec<u16> = common::read_slice(&head_buf, n);
    for i in 0..LQ {
        for h in 0..H {
            for d in 0..D {
                let got = f16::from_bits(head[((h * LQ + i) * D + d) as usize]).to_f32();
                let want = tok[((i * H + h) * D + d) as usize];
                assert_eq!(got, want, "q_convert i={i} h={h} d={d}");
            }
        }
    }

    // o_convert: head-major -> token-major (round-trip back to `tok`)
    let back_buf = common::shared_zeroed(dev, n * std::mem::size_of::<u16>());
    let o_pso = pipeline_for(dev, &library, "o_convert_f16_to_f16");
    if !common::dispatch_threadgroups(
        dev,
        &o_pso,
        &[&back_buf, &head_buf, &params_buf],
        grid,
        threads,
    ) {
        return;
    }
    let back: Vec<u16> = common::read_slice(&back_buf, n);
    for k in 0..n {
        let got = f16::from_bits(back[k]).to_f32();
        assert_eq!(got, tok[k], "o_convert round-trip at {k}");
    }
}
