// SPDX-License-Identifier: Apache-2.0
//! A DFlash2 drafter reads the output of each layer its target captures, after the layer's residual
//! add (Splash: runtime/model/QwenTarget.cpp, `capturedTargetHidden` from the layer's output). A
//! target tape whose fold defers that add into the next layer's norm runs it after the capture: the
//! drafter then reads the layer's input plus its attention, not its output, and accepts a fraction
//! of what Splash does. So no capture of any baked tape is followed by the add-and-norm that
//! finishes the row it copies.
#![cfg(all(feature = "metal", feature = "mtp-heads", feature = "qwen3.6-35b-a3b"))]

use scratchy_target_metal::tape::lowered::KernelId;

#[test]
fn every_capture_copies_a_finished_layer_output() {
    use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b_mlx_affine_b4_g64_qembed as target;
    let mut captures = 0;
    for spec in target::METAL_BUCKETS {
        for classed in spec.tapes {
            let commands = classed.tape.commands_expanded();
            for (i, c) in commands.iter().enumerate() {
                if c.command.kernel != KernelId::CaptureRows {
                    continue;
                }
                captures += 1;
                let next = commands.get(i + 1).map(|n| n.command.kernel);
                assert_ne!(
                    next,
                    Some(KernelId::FusedAddRmsNorm),
                    "m={} {:?}: command {i} captures a row its next command's add finishes",
                    spec.bucket_m,
                    classed.gen_class,
                );
            }
        }
    }
    assert!(captures > 0, "the target captures no rows: build it with a drafter beside it");
}
