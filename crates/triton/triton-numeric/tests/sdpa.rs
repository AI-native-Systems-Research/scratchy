// SPDX-License-Identifier: Apache-2.0

//! TORCH-SPYRE'S SDPA, EXECUTED. `test-fixtures/sdpa.py` -- the decomposition torch-spyre lowers
//! every fused SDPA to (`decompositions.py:1960`), written as one Triton kernel -- goes `.py` ->
//! ttir -> KTIR -> the emulator at the three plans `cases.rs` records, and is compared against
//! torch's own f32 `scaled_dot_product_attention` under the envelope [`bounds::sdpa`] derives.
//!
//! The executor is `ktir-emulator`'s interpreter (`execute_function_with_latency`), handed the
//! `ktir_core::IRFunction` that `to_ktir_emit::lower` builds in memory -- no KTIR text in between.
//! What it runs is `to_ktir::run`'s output with ONE harness rewrite, `refold_matmul_accumulators`
//! (the door's `matmul` + `addf` folded back into an accumulating `matmul`, semantics-preserving).
//! These grids unroll (`unroll_grid_positions` -- a KV window shared by a GQA group is not a
//! per-core uniform), so the emulator runs each configuration as ONE straight-line program on ONE
//! emulated core (grid `(1, 1, 1)`), and the clone arguments that mints are staged from
//! `spyre.partition_clones`, not bypassed.
//!
//! Per configuration, the three things `executed.rs` owes, and one more:
//!   1. the comparison, `max|err|` against the derived bound, both reported;
//!   2. the BRACKETING control on the configuration's own reference;
//!   3. every plausible port bug that is a different function at this shape EXCEEDS the bound;
//!   4. the f64 truth those mutants are measured against IS torch's SDPA, to one f16 rounding --
//!      so a mutant's distance is a distance from the reference, not from a paraphrase of it.

use triton_numeric::bounds::{self, Envelope, F16_MIN_SUBNORMAL, U_F16};
use triton_numeric::mutants::sdpa::{self, Shape, Variant};
use triton_numeric::{Binding, Comparison, data, mutants};

const CONFIGS: &[&str] = &[
    "sdpa_prefill_causal",
    "sdpa_prefill_causal_scan",
    "sdpa_decode",
];

/// The configuration's geometry, read from its `meta.json` -- the constexprs the data was
/// generated at, so a tolerance cannot be derived at one shape and gate another.
fn shape(f: &data::Fixture) -> Shape {
    let i = |k: &str| f.int(k).unwrap_or_else(|e| panic!("{}: {e}", f.config)) as usize;
    Shape {
        z: i("Z"),
        nqh: i("NQH"),
        nkvh: i("NKVH"),
        hd: i("HD"),
        sq: i("SQ"),
        skv: i("SKV"),
        block_n: i("BLOCK_N"),
        scale: f.float("SCALE").expect("SCALE"),
        causal: f
            .constexprs
            .get("HAS_MASK")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    }
}

fn envelope(s: &Shape) -> Envelope {
    bounds::sdpa(s.hd, s.skv, s.block_n, s.scale)
}

fn decoded(bindings: &[Binding], name: &str) -> Vec<f64> {
    let b = bindings
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no binding `{name}`"));
    data::decode_f64(&b.bytes, b.shape.iter().product(), b.dtype).expect("decodes")
}

#[test]
fn every_sdpa_configuration_matches_torch_under_its_derived_bound() {
    let mut failures = Vec::new();
    for config in CONFIGS {
        let f = data::Fixture::load(config).unwrap_or_else(|e| panic!("{config}: {e}"));
        let bindings = f.bindings().unwrap_or_else(|e| panic!("{config}: {e}"));
        let lowered = triton_numeric::lower(config).unwrap_or_else(|e| panic!("{config}: {e}"));
        let run = triton_numeric::execute(&lowered, &bindings)
            .unwrap_or_else(|e| panic!("{config}: {e}"));
        let got = run.f64s(&f.output.arg).expect("desc_o");
        let reference = f.reference().expect("reference");
        let s = shape(&f);
        let env = envelope(&s);
        let c = Comparison::new(config, &got, &reference, &env).expect("comparison");
        println!("  {}", data::banner(&f));
        println!(
            "  launch grid {:?} -> ktir-emulator grid {:?}: {} op(s), {} clone argument(s) staged",
            lowered.grid,
            lowered.func.grid,
            lowered.func.ops_deep().len(),
            lowered.clones.len()
        );
        println!(
            "  bound   k+ = {:.6e}  k- = {:.6e}  floor = {:.6e}",
            env.k_plus, env.k_minus, env.floor
        );
        println!("  {}", c.report());
        if !c.within_bound() {
            failures.push(c.report());
        }

        // THE BRACKETING CONTROL: 1.5x the largest allowance must exceed, 0.5x must not.
        for (factor, must_exceed) in [(1.5, true), (0.5, false)] {
            let (mutated, at) = mutants::bracket(&reference, &env, factor);
            assert_eq!(
                mutants::exceeds(&reference, &mutated, &env),
                must_exceed,
                "{config}: bracketing control at {factor}x (element {at}) {}",
                if must_exceed {
                    "did not exceed -- the gate is looser than its derivation"
                } else {
                    "exceeded -- the gate is tighter than its derivation"
                },
            );
        }
    }
    assert!(
        failures.is_empty(),
        "outside the derived bound:\n{}",
        failures.join("\n")
    );
}

/// THE EXECUTED PATH CAN FAIL. The same KTIR, run on the scan configuration with its mask staged
/// as zeros -- every key visible, the program now non-causal -- must leave the bound, and must land
/// where the f64 non-causal attention lands. That proves the comparison above sees the executed
/// output (not a staged buffer echoed back), and that the scan applies the LOADED mask tile per KV
/// block rather than ignoring it.
#[test]
fn the_executed_scan_obeys_the_mask_it_is_handed() {
    let config = "sdpa_prefill_causal_scan";
    let f = data::Fixture::load(config).expect("data");
    let mut bindings = f.bindings().expect("bindings");
    let s = shape(&f);
    let env = envelope(&s);
    let mask = bindings
        .iter_mut()
        .find(|b| b.name == "desc_mask")
        .expect("the causal configuration binds a mask");
    mask.bytes.iter_mut().for_each(|x| *x = 0);
    let lowered = triton_numeric::lower(config).expect("lowers");
    let got = triton_numeric::execute(&lowered, &bindings)
        .expect("runs")
        .f64s("desc_o")
        .expect("desc_o");
    let reference = f.reference().expect("reference");
    let ratio = mutants::worst_ratio(&reference, &got, &env);
    println!(
        "  {config} with the mask zeroed: worst ratio {ratio:.2} against the causal reference"
    );
    assert!(
        mutants::exceeds(&reference, &got, &env),
        "{config}: a non-causal run stayed inside the causal reference's bound"
    );
    let non_causal = sdpa::attention(
        &decoded(&bindings, "desc_q"),
        &decoded(&bindings, "desc_k"),
        &decoded(&bindings, "desc_v"),
        &Shape { causal: false, ..s },
        Variant::Truth,
    );
    let c = Comparison::new(config, &got, &non_causal, &env).expect("comparison");
    println!("  {}", c.report());
    assert!(
        c.within_bound(),
        "{config}: the zero-mask run is not the non-causal attention:\n{}",
        c.report()
    );
}

/// The f64 truth the mutants are measured from IS the checked-in reference: torch's f32 SDPA cast
/// once to f16 sits within one f16 rounding of the exact value (half an ULP relative, half the
/// subnormal quantum absolute), plus f32's own accumulation error, which at HD=64 and at most 512
/// keys is below 1e-5 of the output's unit scale. If this fails, the mutant controls below would be
/// measuring distance from the wrong function.
#[test]
fn the_f64_truth_is_torchs_sdpa() {
    for config in CONFIGS {
        let f = data::Fixture::load(config).unwrap_or_else(|e| panic!("{config}: {e}"));
        let b = f.bindings().expect("bindings");
        let s = shape(&f);
        let truth = sdpa::attention(
            &decoded(&b, "desc_q"),
            &decoded(&b, "desc_k"),
            &decoded(&b, "desc_v"),
            &s,
            Variant::Truth,
        );
        let reference = f.reference().expect("reference");
        let (worst, at) = truth
            .iter()
            .zip(&reference)
            .map(|(t, r)| (r - t).abs() / (U_F16 * t.abs() + 0.5 * F16_MIN_SUBNORMAL + 1e-5))
            .enumerate()
            .fold(
                (0.0f64, 0usize),
                |(w, wi), (i, x)| if x > w { (x, i) } else { (w, wi) },
            );
        println!(
            "  {config}: worst |ref - truth| / (u|truth| + half-subnormal + 1e-5) = {worst:.4} at [{at}]"
        );
        assert!(
            worst <= 1.0,
            "{config}: the f64 SDPA is not torch's reference at [{at}]: ref {:+.6e}, truth {:+.6e}",
            reference[at],
            truth[at]
        );
    }
}

/// EVERY PLAUSIBLE PORT BUG THAT IS A DIFFERENT FUNCTION AT THIS SHAPE MUST EXCEED THE BOUND, or the
/// bound cannot tell this kernel from a broken one. Which bugs apply is the shape's: a causal
/// off-by-one needs a mask, the scan's two bugs need more than one KV block.
#[test]
fn every_plausible_port_bug_exceeds_the_bound() {
    let mut missed = Vec::new();
    for config in CONFIGS {
        let f = data::Fixture::load(config).unwrap_or_else(|e| panic!("{config}: {e}"));
        let b = f.bindings().expect("bindings");
        let s = shape(&f);
        let env = envelope(&s);
        let reference = f.reference().expect("reference");
        let (q, k, v) = (
            decoded(&b, "desc_q"),
            decoded(&b, "desc_k"),
            decoded(&b, "desc_v"),
        );
        for variant in sdpa::applicable(&s) {
            let m = sdpa::attention(&q, &k, &v, &s, variant);
            let ratio = mutants::worst_ratio(&reference, &m, &env);
            println!("  {config}: {variant:?} worst ratio {ratio:.2}");
            if !mutants::exceeds(&reference, &m, &env) {
                missed.push(format!("{config}: {variant:?} (worst ratio {ratio:.3})"));
            }
        }
    }
    assert!(
        missed.is_empty(),
        "mutants INSIDE the bound:\n{}",
        missed.join("\n")
    );
}
