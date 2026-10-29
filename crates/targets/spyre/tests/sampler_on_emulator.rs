// SPDX-License-Identifier: Apache-2.0
//! THE TOKEN DRAW, EXECUTED ALONE ON THE EMULATOR — the minted `Sample` node (`sample_fwd.py`)
//! spliced and run through the production session, staged the way the bake's numeric gate and the
//! worker stage it, on a logits row whose first maximum is planted: a greedy draw must answer it.
#![cfg(feature = "spyre-emu")]

use std::borrow::Cow;

use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{SampleRows, SourceShape, SubOp, lower_region, mint_sample};

/// Draw greedily over a `vocab`-wide row whose unique maximum sits at `argmax`; returns the
/// index the emulator's run of the minted sampler picked.
fn greedy_draw(vocab: u32, argmax: usize) -> u32 {
    // The one-request lm-head shape the mint admits: `[1, 1] x [1, vocab]`.
    let input = LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: 1, cols: 1 },
            SourceShape {
                rows: 1,
                cols: vocab,
            },
        ],
        ops: vec![OpDesc {
            op: SubOp::MatmulTile {
                n: vocab,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    };
    let mut ir = lower_region(&input, std::num::NonZeroU32::new(64).unwrap());
    let mint = mint_sample(&mut ir, SampleRows::All);
    let node = ir
        .nodes
        .iter()
        .find(|n| matches!(n.op, SubOp::Sample { .. }))
        .expect("the mint appends a Sample node");
    let spliced = scratchy_triton_splice::lower(node, &ir, false).expect("the sampler splices");
    let k = spliced
        .ktir
        .as_ref()
        .expect("the spliced op carries its program");
    let args: Vec<(
        ktir_core::ir::Ssa,
        scratchy_target_spyre::bundle_code::PlaceId,
    )> = k
        .func
        .arguments
        .iter()
        .map(|(ssa, _)| {
            (
                *ssa,
                scratchy_target_spyre::bundle_code::PlaceId::Act(k.bindings[ssa.slot()].get()),
            )
        })
        .collect();
    let group = scratchy_target_spyre::bundle_code::LaunchGroup {
        kv: Default::default(),
        programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
            func: k.func,
            program: k.program,
            args: Cow::Owned(args),
            lx_pin_window_bytes: Some(0),
        }]),
        init_binary: Cow::Borrowed(&[]),
        job_bin_ptr: 0,
        correction: Cow::Borrowed(&[]),
    };

    // The twenty load-time tables, staged exactly as the gate stages them, and the greedy seat:
    // the seed quadruple's four i32 words at 0 (the gate word's 0 IS the greedy law), inv_t 1.0.
    let staggered = scratchy_target_spyre::lower_subtile_tape_to_ktir::staggered_of_ir(&ir)
        .expect("arrangements");
    let consts =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::load_consts_of_ir(&ir, &staggered)
            .expect("the load-const list derives");
    let mut weights: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)> = consts
        .iter()
        .map(|&(tid, kind)| {
            let (bytes, dt, shape) = kind.build();
            (tid as usize, bytes, dt, shape.to_vec())
        })
        .collect();
    for tid in mint.seed_words {
        weights.push((
            tid.index(),
            0i32.to_le_bytes().to_vec(),
            scratchy_tensors::DType::I32,
            vec![1, 1],
        ));
    }
    weights.push((
        mint.invt.index(),
        1.0f32.to_le_bytes().to_vec(),
        scratchy_tensors::DType::F32,
        vec![1, 1],
    ));

    let idx_tid = node.output.tensor.index() as u64;
    let row_tid = node.inputs[0].tensor.index();
    let row_cols = ir.tensors[row_tid].cols as usize;
    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&[group], &[idx_tid])], weights)
            .expect("the one-program session builds");
    // Distinct small values everywhere, the planted maximum well above them, zeros on the pads
    // (the kernel's own pad law drives the pads out of the contest).
    let logits: Vec<f32> = (0..row_cols)
        .map(|i| {
            if i == argmax {
                0.5
            } else if i < vocab as usize {
                ((i * 7919) % 1009) as f32 * 1e-4
            } else {
                0.0
            }
        })
        .collect();
    let out = session
        .run_step(
            0,
            vec![(row_tid as u64, logits, vec![1, row_cols])],
            &[(idx_tid, 0)],
        )
        .expect("the draw runs");
    out[&idx_tid][0] as u32
}

/// How many map windows the emulator has run as ONE fused Metal kernel so far, where it has the
/// Metal backend (its `cfg(metal)`: macOS); `None` elsewhere.
#[cfg(target_os = "macos")]
fn fused_map_windows() -> Option<usize> {
    Some(ktir_emulator::metal::MAP_REGION_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed))
}

#[cfg(not(target_os = "macos"))]
fn fused_map_windows() -> Option<usize> {
    None
}

#[test]
fn a_greedy_draw_answers_the_planted_maximum() {
    let vocab = 49155u32; // granite's
    let mut report = Vec::new();
    for argmax in [0usize, 7, 8, 9, 15, 16, 39752, 39751, 39744, 49154] {
        let before = fused_map_windows();
        let got = greedy_draw(vocab, argmax);
        // ⭐ THE OFFLOAD PROOF. The row's widen windows are past the fused-map size gate, so where
        // the emulator has Metal they run as fused kernels — the path that drew argmax + 8 when it
        // widened without the card's lane stagger. The interpreter alone already draws right, so
        // a draw that never reached the kernel would pass here and prove nothing.
        if let (Some(before), Some(after)) = (before, fused_map_windows()) {
            assert!(
                after > before,
                "planted {argmax}: the draw ran no map window on the fused Metal kernel"
            );
        }
        println!(
            "  planted {argmax:>6} -> drew {got:>6} (diff {:+})",
            got as i64 - argmax as i64
        );
        if got as usize != argmax {
            report.push(format!("{argmax} -> {got}"));
        }
    }
    assert!(
        report.is_empty(),
        "the greedy draw missed the planted maximum: {report:?}"
    );
}
