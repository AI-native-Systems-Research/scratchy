// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ⭐⭐ `embedding_granite_bm128`: WHICH FAILURE IS IT? Both sides, in one process.
//!
//! This configuration is `embedding_granite` at `BLOCK_M = 128` — the same three buffers, the same
//! `ref_out.bin`, two work items instead of four. With the gather's subscript key in place it stops
//! refusing at the gather and refuses at LX instead:
//!
//! ```text
//!   ArithMulf: LX capacity exceeded on core 0: 2097152 + 1048576 > 2097152
//!              charging %34 tile [128, 4096]
//! ```
//!
//! ⛔ AND FROM OUTSIDE THE EXECUTOR THAT IS INDISTINGUISHABLE FROM A WRONG ANSWER. `track_lx_tile`
//! refuses the program BEFORE it computes anything, so the sweep reports "UNVERIFIED" for a
//! residency model and for a broken kernel in exactly the same words. Separating the two is the
//! whole point of this file, and it does it by running the configuration TWICE:
//!
//! 1. at the modelled 2 MiB, where the refusal must happen and its text must be the recorded one;
//! 2. at `KTIR_LX_CAPACITY_MB=3`, where the comparison against `ref_out.bin` and its bracketing
//!    control are exactly the ones every verified configuration gets.
//!
//! ⚖️ WHY RAISING LX IS NOT LOOSENING A GATE. **LX capacity takes part in no arithmetic.** It is a
//! residency budget checked before an op runs; the tolerance, the per-element envelope, the sign
//! awareness and both bracketing sides below are untouched by it. So raising it can turn a refusal
//! into a number and can never turn a WRONG number into a right one — which is the property that
//! makes this a measurement rather than a concession. The default stays at 2 MiB for everyone else
//! (`ktir-emulator:src/machine_state/memory.rs`, `LX_CAPACITY_MB`), so every recorded LX refusal in
//! the tree keeps its exact text.
//!
//! ⚖️ AND WHAT THE 3 MiB RUN DOES **NOT** SAY. It does not say the device runs this kernel; the
//! device's LX is 2 MB and this program as written needs 3 MiB of it. What it does say is that the
//! overflow is 1 MiB of `tensor.splat` — a splatted CONSTANT, which the device path never
//! materialises (`ktir-superdsc`'s `is_plumbing` lists `OpKind::TensorSplat`, and
//! `Lowering::ScalarMul` lowers `tile * <splatted constant>` to an opspec) — so the device's peak
//! here is the gathered tile plus the product, 2 MiB, exactly LX. The executor charges the splat
//! because `consume_if_last_use` returns early at function top level (`cur_gen == 0`), where
//! upstream's comment says the RESIDENT scheduler's `dies_at`/`forget` owns liveness — and
//! `resident` is `optimizer`-gated and off in this build, so nothing owns it. That is a finding for
//! the emulator's reviewer, recorded in `vendor/PROVENANCE.md`, and deliberately not patched: it
//! would move every recorded top-level LX refusal in the tree at once.
//!
//! ⛔ ONE TEST IN ONE FILE, ON PURPOSE. `KTIR_LX_CAPACITY_MB` is process-global and read at every
//! `SpyreMemoryHierarchy::new`. An integration test FILE is its own binary, so nothing else in this
//! crate can observe the variable this test sets; putting it beside the sweep would let the two race.

use triton_numeric::{bounds, data, mutants, Comparison};

const CONFIG: &str = "embedding_granite_bm128";
/// The recorded refusal, minus the `%34`: an SSA number is not a fact about residency, the three
/// byte counts are. Same substring `executed.rs`'s `EXPECTED_REFUSAL` carries.
const RECORDED: &str = "LX capacity exceeded on core 0: 2097152 + 1048576 > 2097152";

/// Run the configuration with LX set to `mb` MiB per core.
fn run_at(mb: i64) -> triton_numeric::Result<Vec<f64>> {
    std::env::set_var("KTIR_LX_CAPACITY_MB", mb.to_string());
    let f = data::Fixture::load(CONFIG).expect("the bm128 metadata");
    let bindings = f.bindings().expect("the bm128 bindings");
    let lowered = triton_numeric::lower(CONFIG)?;
    let out = triton_numeric::execute(&lowered, &bindings)?;
    out.f64s(&f.output.arg)
}

#[test]
fn bm128_refuses_on_residency_at_2_mib_and_computes_the_right_function_at_3() {
    let f = data::Fixture::load(CONFIG).expect("the bm128 metadata");
    println!("  {}", data::banner(&f));
    // ⭐ THE REFERENCE COMES THROUGH A HASH-VERIFIED SIBLING. bm128's own directory holds only
    // `meta.json` + `sha256.txt`; `Fixture::reference` reads `embedding_granite/ref_out.bin` ONLY
    // because bm128's own meta.json records that file's sha256 and the bytes match it. Two configs
    // with the same shape are one `data_dir` apart, so nothing else in this crate could tell the
    // wrong answer from the right one — see `Fixture::reference_dir`.
    let reference = f.reference().expect("ref_out.bin, through the verified sibling");
    let env = bounds::embedding(f.float("EMB_SCALE").expect("EMB_SCALE"));
    println!(
        "  bound   k+ = {:.6e}  k- = {:.6e}  floor = {:.6e}",
        env.k_plus, env.k_minus, env.floor
    );

    // ---- 1. THE MODELLED 2 MiB: the refusal, and its exact text --------------------------------
    match run_at(2) {
        Ok(_) => panic!(
            "⛔ `{CONFIG}` EXECUTED AT THE MODELLED 2 MiB LX. Then the residency refusal this file \
             and `executed.rs`'s `EXPECTED_REFUSAL` both record is gone, and BOTH should be \
             shortened deliberately rather than left claiming a gap that closed."
        ),
        Err(e) => {
            let why = e.to_string();
            println!("  at LX = 2 MiB (the modelled size): REFUSED  {why}");
            assert!(
                why.contains(RECORDED),
                "the recorded residency refusal has CHANGED. Expected a refusal containing\n  \
                 {RECORDED}\nand got\n  {why}\nA recorded gap that becomes a different gap is a new \
                 defect wearing the old one's name."
            );
            // ⛔ AND IT MUST BE THE MULTIPLY THAT OVERFLOWS, ON A `[128, 4096]` TILE. Without this,
            // any LX overflow anywhere in the program would satisfy the substring above and the
            // diagnosis in this file's header — three co-resident `[128, 4096]` f16 tiles, one of
            // them the splat — would be unsupported.
            for want in ["ArithMulf", "[128, 4096]"] {
                assert!(
                    why.contains(want),
                    "the refusal does not name `{want}`, so it is not the one this file diagnoses: \
                     {why}"
                );
            }
        }
    }

    // ---- 2. LX = 3 MiB: the function itself, gated exactly as every verified config is ---------
    let got = run_at(3).unwrap_or_else(|e| {
        panic!(
            "⛔⛔ `{CONFIG}` REFUSES EVEN WITH LX AT 3 MiB, so the 2 MiB refusal was NOT the whole \
             story and something else is wrong: {e}"
        )
    });
    let c = Comparison::new(CONFIG, &got, &reference, &env).expect("a comparison");
    println!("  at LX = 3 MiB: {}", c.report());

    // THE BRACKETING CONTROL, both sides, on this configuration's own reference. Same rule as
    // `executed.rs::bracket_control`: 1.5x the largest per-element allowance must EXCEED and 0.5x
    // must stay WITHIN, which is what proves the gate sits where the derivation puts it.
    for (factor, must_exceed) in [(1.5, true), (0.5, false)] {
        let (mutated, at) = mutants::bracket(&reference, &env, factor);
        let b = Comparison::new(CONFIG, &mutated, &reference, &env).expect("a bracket comparison");
        assert_eq!(
            b.exceeded > 0,
            must_exceed,
            "{CONFIG}: BRACKETING CONTROL FAILED at {factor}x the per-element allowance (element \
             {at}, ref {:+.6e}). {}",
            reference[at],
            if must_exceed {
                "A perturbation of 1.5x the bound must EXCEED it; that it did not means the gate is \
                 LOOSER than the derivation states and the comparison above is vacuous."
            } else {
                "A perturbation of 0.5x the bound must stay WITHIN it; that it did not means the \
                 gate is TIGHTER than the derivation states and a green result is luck."
            }
        );
    }

    assert!(
        c.within_bound(),
        "⛔⛔ `{CONFIG}` EXCEEDED ITS DERIVED BOUND once it could run. The kernel computes a \
         different function from the fixture's reference at `BLOCK_M = 128` while agreeing at 64 — \
         which would be a defect in the gather's block offset, the one thing the two configurations \
         differ in. REPORT IT; do not widen anything: {}",
        c.report()
    );

    // ⭐ AND THE TWO BLOCKINGS MUST AGREE WITH EACH OTHER, not merely each with the reference.
    // `BLOCK_M` is a TILING knob: 4 items of 64 and 2 of 128 walk the same 256 tokens, so a block
    // offset applied to the wrong axis would show up here as a disagreement even if both somehow sat
    // inside the envelope. Run at 2 MiB, where the 64-high tiles fit.
    std::env::set_var("KTIR_LX_CAPACITY_MB", "2");
    let sib = data::Fixture::load("embedding_granite").expect("the bm64 metadata");
    let l64 = triton_numeric::lower("embedding_granite").expect("bm64 lowers");
    let out64 = triton_numeric::execute(&l64, &sib.bindings().expect("bm64 bindings"))
        .expect("bm64 executes at the modelled LX");
    let got64 = out64.f64s(&sib.output.arg).expect("bm64 read-back");
    let cross = Comparison::new("bm128 vs bm64", &got, &got64, &env).expect("a cross comparison");
    println!("  the two blockings against each other: {}", cross.report());
    assert_eq!(
        cross.max_abs_err, 0.0,
        "the two blockings of the SAME kernel over the SAME bytes disagree. `BLOCK_M` is a tiling \
         knob and the gather is data movement, so their outputs are the same f16 values or one of \
         the two reads the wrong rows: {}",
        cross.report()
    );
}
