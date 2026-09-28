// SPDX-License-Identifier: Apache-2.0
//! The megakernel plan of every baked metal decode tape in scope is well-formed: runs are
//! disjoint and in tape order, units tile each run, and every wait points backward. `--nocapture`
//! prints what the dataflow analysis found.
#![cfg(feature = "metal")]

use scratchy_target_metal::interpreter::metal::MetalBucketSpec;
use scratchy_target_metal::tape::lowered::GenClass;
use scratchy_target_metal::tape::megakernel::{plan_runs, step_flow};

fn check(name: &str, buckets: &[MetalBucketSpec]) {
    let decode = buckets.iter().find(|b| b.bucket_m == 1).expect("a decode bucket");
    let classed = decode
        .tapes
        .iter()
        .find(|t| t.gen_class == GenClass::M5 && !t.chunked)
        .expect("an M5 variant");
    let commands = classed.tape.commands_expanded();
    let runs = plan_runs(&commands);

    let mut next = 0;
    let (mut units, mut chained, mut waits, mut shared) = (0, 0, 0, 0);
    for run in &runs {
        assert!(run.commands.start >= next, "{name}: runs overlap or go backward");
        next = run.commands.end;
        let mut step = 0;
        for (u, unit) in run.plan.units.iter().enumerate() {
            assert_eq!(unit.steps.start, step, "{name}: units must tile the run");
            step = unit.steps.end;
            assert!(unit.waits.iter().all(|w| (w.0 as usize) < u), "{name}: a wait points forward");
            waits += unit.waits.len();
            if unit.steps.len() > 1 {
                chained += unit.steps.len();
            }
        }
        assert_eq!(step as usize, run.commands.len(), "{name}: units must cover the run");
        units += run.plan.units.len();
        shared += run.plan.shared.len();
    }
    let covered: usize = runs.iter().map(|r| r.commands.len()).sum();
    let declared = commands.iter().filter(|c| step_flow(c).is_some()).count();
    assert_eq!(covered, declared, "{name}: every declared command belongs to a run");
    println!(
        "{name}: {} commands, {covered} in {} runs -> {units} units ({chained} commands chained \
         onto one threadgroup), {waits} waits, {shared} coherent locations",
        commands.len(),
        runs.len()
    );
}

#[cfg(feature = "llama-3.2-3b")]
#[test]
fn llama_3_2_3b_decode_plan() {
    check("llama-3.2-3b", scratchy_models::llama::llama_3_2_3b_mlx_affine_b4_g64::METAL_BUCKETS);
}

#[cfg(feature = "gemma-4-26b-a4b-it")]
#[test]
fn gemma_4_26b_a4b_decode_plan() {
    check("gemma-4-26b-a4b-it", scratchy_models::gemma4_moe::gemma_4_26b_a4b_it_mlx_affine_b4_g64::METAL_BUCKETS);
}
