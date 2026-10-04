// THE ONE-TILE ATTENTION, EXECUTED BY THE KTIR EMULATOR — the fork instrument.
//
// The card run of `attention_flash_noncausal_1tile` (2026-09-28) exits 0 with the right
// footprint and wrong values: max|err| 5.43 against ref_out, no correlation (~0.01) with any
// candidate formula, no permutation, no transpose. That is the signature of an
// operand-addressing defect, but it sits somewhere between three layers: our `to_ktir` passes,
// the whole-function SuperDSC emission, and the card itself. This example runs ONLY the first
// two legs of the ladder (`.py` -> ttir -> KTIR -> `ktir_core::IRFunction`) and executes the
// result in `ktir-emulator` — the program ABOVE the SuperDSC lowering. Two outcomes, and each
// names its layer:
//
//   * MATCH  -> the KTIR is right and the defect is in the SuperDSC descriptor emission
//     (`whole_function`/`lower_ktir_to_superdsc`) or below it on card.
//   * MISS   -> the defect is in `to_ktir` (the seed fold, the accumulator decomposition, the
//     window folding) or the frontend, and the SuperDSC layer never had a chance.
//
// The comparison is a plain max|err| over the window rows the launch writes, because this is a
// diagnosis and not a gate: the number that matters is whether it is ~1e-2 (f16 softmax
// rounding) or ~5 (garbage).

use triton_numeric::{data, execute, lower};

fn main() {
    let config = "attention_flash_noncausal_1tile";
    let f = data::Fixture::load(config).expect("the staged fixture data loads");
    let lowered = lower(config).expect("the config lowers");
    let bindings = f.bindings().expect("the bindings stage");
    println!("lowered: {} argument(s)", lowered.func.arguments.len());
    let run = execute(&lowered, &bindings).expect("the emulator runs");
    let got = run.f64s("desc_o").expect("desc_o comes back");
    let reference = f.reference().expect("the reference re-reads");
    println!("got {} element(s), reference {}", got.len(), reference.len());

    // The launch is grid [1,1]: query rows 0..64 of the [1024,128] output, HEAD_DIM=128 wide.
    // Compare the window and the rest separately — the rest must be ZERO in both.
    let hd = 128usize;
    let block_m = 64usize;
    let mut max_win = 0.0f64;
    let mut max_rest = 0.0f64;
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        let row = i / hd;
        let e = (g - r).abs();
        if row < block_m {
            max_win = max_win.max(e);
        } else {
            max_rest = max_rest.max(e);
        }
    }
    println!("WINDOW  rows 0..{block_m}: max|err| = {max_win:.6}");
    println!("REST    rows {block_m}..:   max|err| = {max_rest:.6}");
    println!(
        "VERDICT: {}",
        if max_win < 0.05 {
            "KTIR IS RIGHT — the defect is BELOW, in the SuperDSC emission or on card"
        } else {
            "KTIR IS WRONG — the defect is ABOVE, in to_ktir or the frontend"
        }
    );
}
