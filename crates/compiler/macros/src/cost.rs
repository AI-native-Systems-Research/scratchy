// SPDX-License-Identifier: Apache-2.0
//! Post-scheduler cost aggregation with per-wave contention.
//!
//! The solver's DP sums per-Impl costs independently — fine for
//! picking decisions (claim-size DESC breaks ties; cost ASC breaks
//! within size). But the reported wall-clock number should reflect
//! what actually happens on the device: Impls in the same wave
//! contend or overlap according to [`ConcurrencyModel`].
//!
//! This module walks the [`Loop`] per workload point, groups
//! Impls by wave, applies `contention_factor`, and returns the
//! adjusted total µs.
//!
//! Ported in spirit from `scratchy-target-metal/src/lowering/cost.rs` —
//! simplified because our assignment doesn't yet carry
//! `CompilationUnitId`/Handoff edges. Those pieces land with their
//! consumers (DeviceCallable impls + megakernel emission).

use std::collections::BTreeMap;

use crate::assignment::{Assignment, WorkloadAssignments};
use crate::concurrency::ConcurrencyModel;
use crate::fuf::Fuf;
use crate::impl_lib::{CostCtx, Implementation, ImplementationLibrary, MatchInfo};
use crate::schedule::{Loop, WorkloadLoops};
use crate::target::TargetProfile;

/// Contention-aware total µs for a single workload bucket's LOOP.
///
/// Walks each wave, computes each Impl's cost with the other Impls
/// in the same wave as concurrent co-runners, sums. Returns
/// `f64::INFINITY` if any wave contains an infeasible combination
/// (two cooperative grids).
pub fn loop_cost_us(
    fuf: &Fuf,
    sfuf: &Assignment,
    loop_ir: &Loop,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    bounds: &BTreeMap<String, u64>,
) -> f64 {
    let concurrency = ConcurrencyModel::new(target);
    let ctx = CostCtx {
        fuf,
        profile: target,
        bounds,
        expected_calls_per_load: CostCtx::default_expected_calls_per_load(target.backend),
    };

    let mut total = 0.0_f64;

    for wave in &loop_ir.waves {
        // Resolve each subgraph in this wave to (Impl, MatchInfo).
        let mut wave_entries: Vec<(&dyn Implementation, MatchInfo)> =
            Vec::with_capacity(wave.subgraphs.len());
        for (sg, imp_id) in &wave.subgraphs {
            let claimed = sfuf.tiles_in_subgraph(*sg);
            let m = MatchInfo {
                claimed_tiles: claimed,
                boundary_inputs: vec![],
                boundary_outputs: vec![],
            };
            wave_entries.push((lib.get(*imp_id), m));
        }

        // For each entry, apply contention_factor(self, others_in_wave).
        for i in 0..wave_entries.len() {
            let (imp, m) = &wave_entries[i];
            let others: Vec<&dyn Implementation> = wave_entries
                .iter()
                .enumerate()
                .filter_map(|(j, (oi, _))| if j == i { None } else { Some(*oi) })
                .collect();
            let factor = concurrency.contention_factor(*imp, &others);
            if !factor.is_finite() {
                return f64::INFINITY;
            }
            total += imp.cost_us(m, &ctx) * factor;
        }
    }

    total
}

/// Overwrite each `Assignment.predicted_us` with the
/// contention-aware aggregate over its paired `Loop`. Called from
/// the macro drive after solve + schedule.
pub fn refresh_predicted_us(
    fuf: &Fuf,
    sfufs: &mut WorkloadAssignments,
    loops: &WorkloadLoops,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    bounds: &BTreeMap<String, u64>,
) {
    let mut scratch = bounds.clone();
    for (wp, sfuf) in sfufs.per_workload.iter_mut() {
        scratch.insert("num_tokens".into(), wp.num_tokens);
        scratch.insert("sk_bucket".into(), wp.sk_bucket);
        let Some(loop_ir) = loops.per_workload.get(wp) else {
            continue;
        };
        sfuf.predicted_us = loop_cost_us(fuf, sfuf, loop_ir, lib, target, &scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::classify::classify;
    use crate::config::{self, ModelParams};
    use crate::fuf::unroll;
    use crate::impl_lib::starter_library;
    use crate::parse::parse_block;
    use crate::schedule::schedule_workloads;
    use crate::shape::infer;
    use crate::solver::solve;
    use crate::target::from_profile_def;
    use std::path::PathBuf;

    fn llama_params(stem: &str) -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            .join("configs")
            .join("llama")
            .join(format!("{stem}.json"));
        config::load_file(&path).unwrap()
    }

    fn l4_target() -> TargetProfile {
        from_profile_def(&scratchy_target_cuda::targets::L4_SM89)
    }

    const LLAMA_BODY: &str = r#"
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            normed = rmsnorm(hidden_states, input_layernorm[layer]);
            q = gemm(normed, self_attn.q_proj[layer]);
            k = gemm(normed, self_attn.k_proj[layer]);
            v = gemm(normed, self_attn.v_proj[layer]);
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
            attn = attention(q, k, v, kv_cache[layer], block_table);
            oproj = gemm(attn, self_attn.o_proj[layer]);
            hidden_states = add(oproj, hidden_states);

            normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
            gate = silu(gemm(normed2, mlp.gate_proj[layer]));
            up = gemm(normed2, mlp.up_proj[layer]);
            down = gemm(gate * up, mlp.down_proj[layer]);
            hidden_states = add(down, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
    "#;

    #[test]
    fn loop_cost_equals_naive_sum_on_serial_chain() {
        // Post-fusion Llama is a pure serial chain: every wave has
        // exactly one subgraph, so contention factor is always 1.0.
        // Consequence: contention-aware total matches the DP's naive
        // per-Impl sum. If this test fails, either the scheduler
        // started grouping independent subgraphs into waves (good —
        // update the test to match) or the concurrency model's
        // factor-1 identity broke.
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        let sfuf = sfufs.get_nt(1).unwrap();
        let loops = schedule_workloads(&fuf, &sfufs);
        let wp1 = crate::assignment::WorkloadPoint::num_tokens_only(1);
        let loop_ir = &loops.per_workload[&wp1];

        let mut scratch = params.bounds.clone();
        scratch.insert("num_tokens".into(), 1);

        let naive = sfuf.predicted_us;
        let aware = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);

        // Every wave has one subgraph → contention factor 1.0
        // everywhere. Numbers must match to within float-noise
        // (both sums include the same Impl cost_us calls).
        assert!(
            (naive - aware).abs() < 1e-6,
            "naive={naive} aware={aware} (expected equal on serial chain)"
        );
    }

    /// `loop_cost_us` must be deterministic across repeated calls on
    /// the same inputs. Non-determinism here (old scratchy-target-metal had
    /// a HashMap-iteration bug that caused this) would invalidate
    /// any future cost-aware optimization pass: two calls in a row
    /// on the same assignment returned different numbers.
    #[test]
    fn loop_cost_us_is_deterministic() {
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let sfufs = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();
        let loops = schedule_workloads(&fuf, &sfufs);

        let mut scratch = params.bounds.clone();
        for &m in &[1u64, 512] {
            scratch.insert("num_tokens".into(), m);
            let wp = crate::assignment::WorkloadPoint::num_tokens_only(m);
            let sfuf = sfufs.get_nt(m).unwrap();
            let loop_ir = &loops.per_workload[&wp];
            let first = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);
            for _ in 0..20 {
                let again = loop_cost_us(&fuf, sfuf, loop_ir, &lib, &target, &scratch);
                assert!(
                    (first - again).abs() < 1e-9,
                    "loop_cost_us returned {first} then {again} at num_tokens={m}"
                );
            }
        }
    }

    #[test]
    fn refresh_predicted_us_overwrites_in_place() {
        let params = llama_params("llama-3.2-1b");
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {LLAMA_BODY} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        let lib = starter_library();
        let target = l4_target();

        let mut sfufs = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();
        let before: BTreeMap<crate::assignment::WorkloadPoint, f64> = sfufs
            .per_workload
            .iter()
            .map(|(wp, sf)| (*wp, sf.predicted_us))
            .collect();
        let loops = schedule_workloads(&fuf, &sfufs);
        refresh_predicted_us(&fuf, &mut sfufs, &loops, &lib, &target, &params.bounds);
        for (wp, sf) in sfufs.per_workload.iter() {
            assert!(sf.predicted_us.is_finite() && sf.predicted_us > 0.0);
            // On serial chain the refresh shouldn't move the number.
            assert!((before[wp] - sf.predicted_us).abs() < 1e-6);
        }
    }
}
