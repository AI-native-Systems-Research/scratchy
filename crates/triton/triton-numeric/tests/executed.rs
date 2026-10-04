// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE TEN CONFIGURATIONS, EXECUTED AND COMPARED. This is the file the whole crate is for.
//!
//! Per configuration: `.py` source -> ttir -> KTIR -> `ktir_core::IRFunction` -> the KTIR executor
//! -> the fixture's own torch reference. `max|err|` and the DERIVED bound, both reported, never
//! "passes".
//!
//! # ⚠️ WHAT THE ORACLE IS
//!
//! `ktir-emulator` accumulates a CONTRACTION in f32 and rounds it once
//! (`ktir-emulator:src/dialects/linalg.rs:446` -> `blas.rs:42`, one `Tile::compute` at
//! `linalg.rs:463`), and rounds EVERY ELEMENTWISE STEP to the tile's dtype
//! (`ktir-core:src/tile.rs:151`). The device rounds every step to DL16 f16 with round-HALF-UP; this
//! rounds to nearest-EVEN (`ktir-core:src/codec.rs:67,74`). So it is a **CORRECTNESS** oracle and
//! not a fidelity one: it proves the program computes the right FUNCTION -- a wrong contraction, a
//! dropped transpose, a reduction on the wrong axis -- and it does NOT prove the device's rounding
//! matches.
//!
//! ⛔ IT ALSO HAS NO SIGMOID AND NO NEWTON RSQRT. `math.rs`'s registration table has no sigmoid
//! entry at all and `rsqrt` is `1.0 / x.sqrt()` (`math.rs:110`), so `eps_sigmoid` and `eps_rsqrt` --
//! which DOMINATE the pod's device bounds -- are error terms this oracle never commits. That is
//! exactly why [`triton_numeric::bounds`] re-derives rather than copying `test/pod`'s coefficients,
//! and it is not a detail: `rmsnorm`'s stick-drop control has a worst-ratio of 2.18 against the
//! re-derived bound and 0.28 against the device's, i.e. VACUOUS.
//!
//! # WHAT EACH CONFIGURATION OWES
//!
//! 1. **The comparison**, against a bound derived before the run from the arithmetic.
//! 2. **The BRACKETING control** -- the reference perturbed at its largest-bound element by 1.5x
//!    that element's allowance must EXCEED, and by 0.5x must stay WITHIN. One-sided controls prove
//!    only that a test CAN fail; bracketing proves the threshold sits where the derivation puts it.
//! 3. **A discriminating mutant** where one exists for that kernel's algorithm.

use triton_numeric::{bounds, data, mutants, Comparison};

/// Which derived envelope gates a configuration, with the extents read FROM `meta.json` rather than
/// restated here -- so a tolerance derived for one extent cannot come to gate a comparison at
/// another.
fn envelope(f: &data::Fixture) -> bounds::Envelope {
    let i = |k: &str| f.int(k).unwrap_or_else(|e| panic!("{}: {e}", f.config)) as usize;
    match f.fixture.as_str() {
        "rmsnorm" => bounds::rmsnorm(i("D_MODEL")),
        "rope" => bounds::rope(),
        "embedding" => bounds::embedding(f.float("EMB_SCALE").expect("EMB_SCALE")),
        // ⛔ THE BLOCKING IS READ FROM `meta.json` TOO, not just the width. `BLOCK_N`/`BLOCK_K` set
        // the two loops' trip counts, which ARE the MLP's f16 accumulation depths, so a tolerance
        // derived at one blocking must not come to gate a comparison at another -- the same rule
        // this function already applies to D_MODEL.
        "swiglu_mlp" => bounds::swiglu(i("D_MODEL"), i("D_FF"), i("BLOCK_N"), i("BLOCK_K")),
        "decoder_block" => bounds::decoder(
            i("D_MODEL"),
            i("D_FF"),
            // ONE LAYER OR TWO IS THE CONFIGURATION, and the kernel name is where it is stated.
            // `constexprs` does not carry a layer count, so reading it off the kernel is reading it
            // from the program rather than from a second statement that could disagree.
            if f.kernel.contains("two_layers") { 2 } else { 1 },
        ),
        other => panic!("{}: no derived envelope for fixture `{other}`", f.config),
    }
}

/// What became of one configuration. Three OUTCOMES AND NOT TWO, because "it did not run" has two
/// causes that a reader must be able to tell apart: data that is not in the tree (expected, for the
/// one oversized configuration) and a stage that REFUSED (a defect, wherever it is).
enum Outcome {
    Ran(data::Fixture, Comparison),
    /// No bytes to compare against.
    Missing(String),
    /// A stage refused. THIS IS NOT A SKIP -- the configuration is unverified and something is
    /// broken. Collected rather than panicked on, so ONE run reports every refusal instead of the
    /// first; the sweep still fails.
    Refused(String),
}

fn run(config: &str) -> Outcome {
    let f = match data::Fixture::load(config) {
        Ok(f) => f,
        Err(e) => return Outcome::Missing(e.to_string()),
    };
    let bindings = match f.bindings() {
        Ok(b) => b,
        Err(e) => return Outcome::Missing(e.to_string()),
    };
    let lowered = match triton_numeric::lower(config) {
        Ok(l) => l,
        Err(e) => return Outcome::Refused(format!("the chain refused before executing: {e}")),
    };
    let out = match triton_numeric::execute(&lowered, &bindings) {
        Ok(o) => o,
        Err(e) => return Outcome::Refused(e.to_string()),
    };
    let got = match out.f64s(&f.output.arg) {
        Ok(g) => g,
        Err(e) => return Outcome::Refused(e.to_string()),
    };
    let reference = match f.reference() {
        Ok(r) => r,
        Err(e) => return Outcome::Refused(e.to_string()),
    };
    let env = envelope(&f);
    let c = match Comparison::new(config, &got, &reference, &env) {
        Ok(c) => c,
        Err(e) => return Outcome::Refused(e.to_string()),
    };
    println!("  {}", data::banner(&f));
    println!(
        "  bound   k+ = {:.6e}  k- = {:.6e}  floor = {:.6e}",
        env.k_plus, env.k_minus, env.floor
    );
    println!("  {}", c.report());
    Outcome::Ran(f, c)
}

/// THE BRACKETING CONTROL, on the configuration's OWN reference.
///
/// ⛔ A COMPARISON THAT CANNOT FAIL PROVES NOTHING, and a control that only fails proves only that.
/// `bracket` moves the element with the largest upward allowance by `factor` x that allowance; 1.5
/// must exceed and 0.5 must not. If the 1.5 side passed, the gate is not where the algebra puts it;
/// if the 0.5 side failed, the gate is tighter than the algebra allows and every green result below
/// is luck.
fn bracket_control(config: &str, reference: &[f64], env: &bounds::Envelope) {
    for (factor, must_exceed) in [(1.5, true), (0.5, false)] {
        let (mutated, at) = mutants::bracket(reference, env, factor);
        let c = Comparison::new(config, &mutated, reference, env).expect("a bracket comparison");
        assert_eq!(
            c.exceeded > 0,
            must_exceed,
            "{config}: BRACKETING CONTROL FAILED at {factor}x the per-element allowance \
             (element {at}, ref {:+.6e}). {} {}",
            reference[at],
            if must_exceed {
                "A perturbation of 1.5x the bound must EXCEED it; that it did not means the gate is \
                 LOOSER than the derivation states and every comparison in this file is vacuous."
            } else {
                "A perturbation of 0.5x the bound must stay WITHIN it; that it did not means the \
                 gate is TIGHTER than the derivation states and a green result is luck."
            },
            c.report(),
        );
    }
}

/// ⭐ EVERY CONFIGURATION WITH DATA, IN ONE TEST, so the report is one table and a configuration
/// cannot be quietly absent from it.
///
/// ⛔ AND A MISSING CONFIGURATION IS A FAILURE, NOT A SKIP. Eight of the eleven have their bytes in
/// the tree; the three Granite-width SwiGLU configurations share one set of 315,621,376 bytes of
/// densely-read weights that live on the pod, and all three are named here explicitly so that
/// "eight ran" can never be read as "eleven did".
#[test]
fn every_configuration_executes_and_matches_its_fixture_reference() {
    const TEN: &[&str] = &[
        "rmsnorm_granite",
        "rope_q32",
        "rope_kv8",
        "embedding_granite",
        "embedding_granite_bm128",
        // ⭐⭐⭐ THE GATHER AT **ONE INDEX STICK** — the configuration that fits the 32-entry
        // (128-byte) stick dxp fills the L3LU IBR with in a single transfer, and therefore the one that
        // emits a SINGLE op. The two above declare 256 entries and are CUT into one op per stick;
        // all three are exact on card. Its own bytes, not a truncation: N_TOK reaches `inputs()` where
        // BLOCK_M does not, so ids, table and reference are all different (`ref_out.bin` sha
        // `24332ebe…`).
        //
        // ⚖️ THE TWO SIDES MEASURE DIFFERENT THINGS AND BOTH ARE KEPT. The emulator is
        // layout-agnostic, so a green result here says the FUNCTION is right at any entry count; the
        // card says the ADDRESSING is. This row is also the accepting control that keeps the cut a cut:
        // one index stick must still emit one op, and its descriptor is byte-identical to the one the
        // card proved before the cut existed.
        "embedding_granite_m32",
        "swiglu_mlp_flat",
        "swiglu_mlp_granite_flat",
        // ⭐ GRANITE WIDTH AT A BLOCKING THAT FITS LX, which is the one that gets a NUMBER. Same
        // kernel, same width, same bytes as `_flat` -- only BLOCK_N/BLOCK_K differ (64 / 2048
        // against 12800 / 4096), and `cases.rs` records the sweep that shows those are the only
        // settings the executor's LX model admits. It sits BESIDE `_flat` rather than replacing it
        // because `_flat`'s refusal is itself a measurement, asserted below.
        "swiglu_mlp_granite_tiled_k",
        // ⭐ THE MIDDLE BLOCKING (BLOCK_N 64 / BLOCK_K 4096), a RUNNING config since the accumulate
        // form was restored in the straight-line program -- it refused at `ScfFor` from Sep 19 and
        // at `LinalgMatmul` after the Sep 28 decompose; `granite_width_swiglu_at_three_blockings`
        // carries the full attribution and runs it under its own derived envelope.
        "swiglu_mlp_granite",
        "decoder_layer_one_flat",
        "decoder_two_layers_flat",
    ];
    /// The configurations whose inputs are not in the tree. Named, so the day they ARE checked in
    /// this list shrinks deliberately rather than the sweep silently covering one more.
    const STAGED_ONLY: &[&str] = &[
        "swiglu_mlp_granite_flat",
        "swiglu_mlp_granite_tiled_k",
        "swiglu_mlp_granite",
    ];

    // ⭐⭐ THE GATHER REACHES THE EXECUTOR NOW, AND THE GAP THAT WAS RECORDED HERE IS CLOSED. What
    // stood here was a three-part diagnosis ending "SO THE FIX IS UPSTREAM, NOT HERE: `AttrKey` needs
    // a key that can carry an indirect dim's index expression". Right about the mechanism, wrong
    // about the OWNERSHIP: `ktir-core` and `ktir-emulator` are vendored under `rust/vendor/` with an
    // unmerged-patch table beside them, so "upstream" was a directory in this repository. The key is
    // `AttrKey::DimSubs` (`dim_subs`) -- one `AffineMapList`, one map per output dim, domain = the
    // ENUMERATION point, symbols = `intermediate_vars` -- and it is a row in that table.
    //
    // THE THREE MISMATCHES, KEPT, because each has a silently-wrong neighbour and the neighbours are
    // what the controls are aimed at. `to_ktir_emit::indirect_access_tile` crosses all three:
    //
    //   1. `dim_kinds` -- ours is the C++ `per_dim_subscript_kinds`, a boolean list
    //      `StrList(["true", "false"])`; the executor matches the STRINGS `"direct"` /
    //      `"direct_sub"` / `"indirect"` / `"direct_expr"`.
    //   2. THE OPERAND LAYOUT. The C++ op's two operand groups flatten to
    //      `(%table, %row_off, %c0, %ids_view)`; the executor reads `operands[1..]` as index views
    //      ONLY and takes the captured scalars from `intermediate_vars`. That is what
    //      "index_view 0 is Index(0), expected MemRef" was.
    //   3. THE SUBSCRIPT MAPS. Ours are in the captures-first domain
    //      `(c_x0, c_y, d_0, d_1) -> (d0 + d2)`; a subscript is evaluated at the enumeration point,
    //      so the captures become SYMBOLS -- `(d0, d1)[s0, s1] -> s0 + d0`, i.e.
    //      `ids[program_id * BLOCK_M + row]`, which is the address `embedding.py:113` writes.
    //
    // ⛔ AND FIXING (2) ALONE WAS NEVER AVAILABLE, BECAUSE IT IS A WRONG ANSWER AND NOT A PARTIAL
    // ONE. Drop the capture operands and state no subscript: the executor takes its legacy identity
    // path, addresses the index view by the bare enumeration point, and grid item 0 agrees while
    // items 1..3 each re-read `ids[0..64]`. 192 of 256 tokens read the wrong embedding and every
    // shape checks out. `the_gathers_subscript_crosses_into_the_executors_domain` (in `triton-ktir`)
    // pins WHICH subscript crossed rather than that one exists, and `tests/embedding_gather.rs`
    // gates the ANSWER with an off-by-one-row mutant and the sparse table's absent-row property,
    // both on the checked-in bytes.
    //
    // ⛔⛔ `embedding_granite_bm128` STILL REFUSES, FOR A DIFFERENT REASON, AND IT IS A RESIDENCY
    // STATEMENT RATHER THAN AN ARITHMETIC ONE -- the same kind as the two SwiGLU entries below.
    // Same kernel at `BLOCK_M = 128`: a `[128, 4096]` f16 tile is 1,048,576 B and THREE are
    // co-resident at function top level -- the gathered rows, the `tensor.splat` of `EMB_SCALE`, and
    // their product -- against a modelled 2,097,152 B LX.
    //
    // ⚖️ AND THE 1 MiB THAT OVERFLOWS IS THE SPLAT, WHICH THE DEVICE PATH DOES NOT MATERIALISE:
    // `ktir-superdsc`'s `is_plumbing` lists `OpKind::TensorSplat` and `Lowering::ScalarMul` lowers
    // `tile * <splatted constant>` to an opspec, so the device's peak for this kernel is the gathered
    // tile plus the product -- 2 MiB, exactly LX. The executor charges it because
    // `consume_if_last_use` returns early at function top level (`cur_gen == 0`), where upstream's
    // comment says the RESIDENT scheduler owns liveness, and `resident` is `optimizer`-gated and off
    // in this build. So this does NOT say the kernel computes the wrong function at `BLOCK_M = 128`,
    // and `tests/embedding_bm128_lx.rs` measures which of the two it is: it runs this configuration
    // at the default 2 MiB (asserting this very refusal) and again at `KTIR_LX_CAPACITY_MB=3`, where
    // it matches its reference at max|err| = 0 with the bracketing control on both sides.
    //
    // ⭐ RECORDED AS AN EXPECTATION WITH ITS EXACT TEXT, WHICH IS THIS TREE'S OWN IDIOM
    // (`pure_rust_ktir.rs::Expect::DiffersOn`) and the reason the gather's own gap was noticed the
    // day it closed. The substring omits the `%34` because an SSA number is not a fact about
    // residency; the three byte counts are.
    const EXPECTED_REFUSAL: &[(&str, &str)] = &[
        (
            "embedding_granite_bm128",
            "LX capacity exceeded on core 0: 2097152 + 1048576 > 2097152",
        ),
        // ⛔ AND THE FLAT GRANITE BLOCKING'S LX OVERFLOW IS RECORDED THE SAME WAY, so it is asserted
        // by the sweep rather than by a comment. `BLOCK_N = D_FF = 12800` states a `[64, 12800]` f16
        // tile of 1,638,400 B and the body needs TWO live at once -- the gate and the up projection
        // -- before their elementwise product, against a modelled 2,097,152 B LX.
        //
        // ⚖️ THIS IS A RESIDENCY STATEMENT ABOUT THE PROGRAM WE EMIT, NOT ABOUT THE DEVICE. The
        // executor has NO TILER: it runs the program as written. The SuperDSC consumer does tile,
        // and `swiglu_mlp_granite_tiled_k` -- same kernel, same width, same bytes, BLOCK_N 64 and
        // BLOCK_K 2048 -- is in the list above and RUNS, which is what turns this entry from a
        // blocker into a measurement of one blocking.
        (
            "swiglu_mlp_granite_flat",
            "TensorSplat: LX capacity exceeded on core 0: 1638400 + 1638400 > 2097152",
        ),
    ];

    let mut ran = Vec::new();
    let mut missing = Vec::new();
    let mut refused = Vec::new();
    let mut expected_gap = Vec::new();
    let mut failed = Vec::new();
    for config in TEN {
        println!("=== {config} ===");
        match run(config) {
            Outcome::Missing(why) => {
                println!("  MISSING  {why}");
                missing.push(*config);
            }
            Outcome::Refused(why) => {
                println!("  REFUSED  {why}");
                match EXPECTED_REFUSAL.iter().find(|(c, _)| c == config) {
                    // The refusal we already know about, AND ITS TEXT STILL MATCHES. A recorded gap
                    // that quietly becomes a DIFFERENT gap is a second defect wearing the first
                    // one's name, so the substring is asserted rather than the mere fact of a
                    // refusal.
                    Some((_, want)) if why.contains(want) => expected_gap.push(*config),
                    Some((_, want)) => refused.push(format!(
                        "{config}: the recorded gap has CHANGED. Expected a refusal containing\n                           {want}\nand got\n  {why}\nA recorded gap that becomes a different gap is                          a new defect wearing the old one's name."
                    )),
                    None => refused.push(format!("{config}: {why}")),
                }
            }
            Outcome::Ran(f, c) => {
                let env = envelope(&f);
                let reference = f.reference().expect("the reference re-reads");
                bracket_control(config, &reference, &env);
                if !c.within_bound() {
                    failed.push(c.report());
                }
                ran.push(*config);
            }
        }
    }

    println!("\n=== SUMMARY ===\nran {}/{}: {ran:?}", ran.len(), TEN.len());
    if !missing.is_empty() {
        println!("missing: {missing:?}");
    }
    if !expected_gap.is_empty() {
        println!(
            // NOT "the gather encoding" any more: `EXPECTED_REFUSAL` now records two DIFFERENT
            // kinds of gap -- the gather's attribute vocabulary and the flat Granite blocking's LX
            // residency -- and a label that named only one would misattribute the other.
            "UNVERIFIED, by a RECORDED refusal (each entry names its own): {expected_gap:?}"
        );
    }
    if !refused.is_empty() {
        println!("refused: {}", refused.len());
    }
    // ⛔ AND A GAP THAT CLOSES MUST FAIL THIS TEST. Every recorded refusal has to actually happen;
    // otherwise a configuration that started working would keep being reported as unverified, which
    // is the mirror image of reporting an unverified one as green.
    let closed: Vec<&str> = EXPECTED_REFUSAL
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| !expected_gap.contains(c))
        .filter(|c| !missing.contains(c))
        .collect();
    assert!(
        closed.is_empty(),
        "these configurations are on the EXPECTED_REFUSAL list but did NOT refuse: {closed:?}. The \
         gap closed -- delete them from the list and let the comparison gate them."
    );
    // A configuration may be missing ONLY because it is one of the staged-only ones.
    let unexpected: Vec<&&str> = missing.iter().filter(|c| !STAGED_ONLY.contains(c)).collect();
    assert!(
        unexpected.is_empty(),
        "these configurations have no data and are not on the staged-only list, so the sweep \
         covered less than it names: {unexpected:?}"
    );
    assert!(
        !ran.is_empty(),
        "NOTHING RAN. A sweep over an empty list passes, which is the failure this assertion exists \
         to prevent."
    );
    assert!(
        refused.is_empty(),
        "a stage REFUSED for these configurations, so they are UNVERIFIED -- not skipped. Every \
         refusal names its stage; each is a defect somewhere between the fixture and the \
         executor:\n{}",
        refused.join("\n")
    );
    assert!(
        failed.is_empty(),
        "these configurations exceeded their DERIVED bound -- the kernel computes a different \
         function from the fixture's reference, or the derivation is wrong, and either is a \
         finding:\n{}",
        failed.join("\n")
    );
}

/// THE DEFECT THAT PASSED THE DXP GATE FOR HOURS, as a control on the configuration that carried it.
///
/// ⛔ THIS IS THE CONTROL THAT EARNS THE HARNESS. Two decoders compiled to SpyreCode while
/// contracting `q @ k` where the program says `q @ k.T`. `dxp_standalone` could not see it because
/// it executes no arithmetic. So: contract a square projection against B instead of B^T and assert
/// the deviation EXCEEDS the decoder's envelope. If it does not, this file cannot see the very
/// defect it was written for, and that is the loudest thing it could report.
///
/// SQUARE ON PURPOSE. `wq/wk/wv/wo` are `[128, 128]`, so no shape check anywhere can separate the
/// two orientations -- which is exactly why the defect survived, and `whole_function.rs`'s
/// `matmul_weight_is_transpose_b` says the same about Granite's `[4096, 4096]` output projection.
#[test]
fn transposing_a_square_projection_exceeds_the_decoder_bound() {
    let f = data::Fixture::load("decoder_layer_one_flat").expect("the decoder data");
    let d_model = f.int("D_MODEL").expect("D_MODEL") as usize;
    let env = envelope(&f);

    // A unit-variance activation and a `randn/sqrt(d_model)` weight -- the stimulus the fixture's
    // own `layer_weights` draws (`decoder_block.py:500-514`), reproduced here deterministically
    // because a mutant needs the OPERANDS and the checked-in data holds only the block's output.
    let m = 64usize;
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Two uniforms -> one approximately-normal sample; the mutant's margin is orders of
        // magnitude, so the stimulus only has to be non-degenerate.
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let a: Vec<f64> = (0..m * d_model).map(|_| next()).collect();
    let scale = 1.0 / (d_model as f64).sqrt();
    let b: Vec<f64> = (0..d_model * d_model).map(|_| next() * scale).collect();

    let truth = mutants::matmul::truth(&a, &b, m, d_model, d_model);
    let wrong = mutants::matmul::transposed_b(&a, &b, m, d_model, d_model);
    let (max_dev, bound_at) = mutants::max_abs_deviation(&truth, &wrong, &env);
    let ratio = mutants::worst_ratio(&truth, &wrong, &env);
    println!(
        "transposed_b on [{d_model}, {d_model}]: max|dev| = {max_dev:.6e}  bound = {bound_at:.6e}  \
         worst ratio = {ratio:.4}"
    );
    assert!(
        mutants::exceeds(&truth, &wrong, &env),
        "⛔⛔ THE q@k VS q@k.T DEFECT DOES NOT EXCEED THE DECODER'S ENVELOPE. max|dev| = \
         {max_dev:.6e} against a bound of {bound_at:.6e} (worst ratio {ratio:.4}). This is the \
         defect that passed the dxp gate for hours, and if this control cannot see it then neither \
         can the comparison above -- the decoder configurations' green results mean nothing. \
         REPORT THIS LOUDLY rather than loosening anything."
    );
    // A mutant must change essentially every word, not a handful. `MUTANT_MIN_ROW_FRACTION` is the
    // predeclared floor ported from `swiglu_oracle2.py`, and the `md > 0` any-change rule it
    // replaced let a one-word-per-row mutant pass.
    let frac = mutants::changed_row_fraction(&truth, &wrong, d_model);
    assert!(
        frac >= mutants::MUTANT_MIN_ROW_FRACTION,
        "the transposed contraction changes only {frac:.4} of the words in its thinnest row; the \
         predeclared floor is {}",
        mutants::MUTANT_MIN_ROW_FRACTION
    );
}

/// THE SWIGLU OPERAND SWAP, which is the same defect in elementwise clothing.
///
/// `swap_gu` exchanges the gate and the up projection. Ported from `swiglu_oracle2.py`'s mutant
/// list, where it sits beside `no_sigmoid` and `sig_of_u`.
///
/// ⚠️ AND `sig_of_u` IS NOT AN INDEPENDENT FOURTH CONTROL. In exact arithmetic it equals `swap_gu`
/// -- both are `g * u * sigmoid(u)` and only the association differs -- so the pod's four entries
/// are THREE controls plus one informational. Counting four would be overcounting, and
/// `mutants::swiglu` asserts the equality so the duplication cannot be forgotten.
#[test]
fn the_swiglu_operand_swap_exceeds_the_activation_bound() {
    let env = bounds::swiglu_activation();
    let n = 4096usize;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
    };
    let g: Vec<f64> = (0..n).map(|_| next()).collect();
    let u: Vec<f64> = (0..n).map(|_| next()).collect();
    let truth = mutants::swiglu::truth(&g, &u);

    for (name, got) in [
        ("no_sigmoid", mutants::swiglu::no_sigmoid(&g, &u)),
        ("swap_gu", mutants::swiglu::swap_gu(&g, &u)),
        ("half_sigmoid", mutants::swiglu::half_sigmoid(&g, &u)),
    ] {
        let (dev, bound) = mutants::max_abs_deviation(&truth, &got, &env);
        let ratio = mutants::worst_ratio(&truth, &got, &env);
        println!("swiglu/{name:14} max|dev| = {dev:.6e}  bound = {bound:.6e}  ratio = {ratio:.4e}");
        assert!(
            mutants::exceeds(&truth, &got, &env),
            "swiglu/{name} does not exceed the activation envelope: max|dev| = {dev:.6e} against \
             {bound:.6e}. `swap_gu` is the q@k-vs-q@k.T defect in swiglu clothing; a control that \
             cannot see it makes the swiglu comparison vacuous."
        );
    }
}

/// ⭐⭐ GRANITE WIDTH: THE THREE BLOCKINGS OF ONE FUNCTION, OVER ONE SET OF BYTES.
///
/// `swiglu_mlp.py` has two loops -- `for n in tl.range(0, D_FF, BLOCK_N)` and, nested inside it,
/// `for k in tl.range(0, D_MODEL, BLOCK_K)` -- and `BLOCK_N`/`BLOCK_K` are TILING knobs: the
/// computed function is the same at every setting (delta 12). So this is a controlled comparison of
/// three lowerings against ONE answer, at `d_model = 4096, d_ff = 12800`:
///
/// ```text
///   swiglu_mlp_granite_flat      BLOCK_N 12800  BLOCK_K 4096   REFUSES: TensorSplat,  [64, 12800]
///   swiglu_mlp_granite           BLOCK_N    64  BLOCK_K 4096   RUNS  -> its own derived bound
///   swiglu_mlp_granite_tiled_k   BLOCK_N    64  BLOCK_K 2048   RUNS  -> its own derived bound
/// ```
///
/// ⛔⛔ THE `_flat` REFUSAL IS A RESIDENCY STATEMENT ABOUT THE PROGRAM WE EMIT, NOT A DEVICE LIMIT,
/// AND THE ROWS BELOW IT ARE WHAT PROVE IT. The executor has NO TILER: it runs the program as written
/// against a modelled 2 MiB LX, so its charge is about OUR blocking. `_flat` states a `[64, 12800]`
/// f16 tile (1,638,400 B) and needs two live at once -- the gate and the up projection -- before
/// their elementwise product. `swiglu_mlp_granite` narrows BLOCK_N to 64 but leaves
/// `BLOCK_K = D_MODEL`; it ran once the program was STRAIGHT-LINE with the accumulate form restored
/// (see the REFUSE history note below), and it is now gated by its own comparison at its own derived
/// bound (t_k = 1, TIGHTER than tiled_k's t_k = 2 -- never a shared envelope).
/// `swiglu_mlp_granite_tiled_k` makes the inner loop MULTI-trip at the largest BLOCK_K that fits.
/// `cases.rs` carries the full sweep that shows both knobs are measured rather than
/// picked -- BLOCK_N is FORCED to 64 by the `[D_MODEL, BLOCK_N]` down-projection tile, and 2048 is
/// the largest divisor of D_MODEL that fits.
///
/// ⛔ THE MIDDLE ROW'S OWN HISTORY, RECORDED SO NOBODY READS ITS PASSING AS A MODELING GIFT. It
/// refused from Sep 19 at `ScfFor 1589248 + 524288` charging a `[64, 4096]` tile, because a LOOP's
/// yield charges a fresh carry tile beside the live one. Sep 28's `48c6a7954` brought TWO changes
/// at once: `unroll_constant_trip_loops` made the program straight-line (no `scf.for` reaches the
/// executor at this trip count), and `decompose_matmul_accumulators` spelled the accumulation as
/// `splat + matmul + addf`, which held the fresh `[64, 4096]` dot live beside the old accumulator
/// and moved the SAME overflow to `LinalgMatmul`. `refold_matmul_accumulators` (triton-numeric's
/// lowering) folds the decompose's exact shape back to `outs = acc` for the EMULATOR path only;
/// in the straight-line program the consume-on-last-use model frees each accumulator at the matmul
/// that takes it as `outs`, BEFORE the result is charged, so 2 MiB holds the whole chain. The
/// emulator's charge model is UNTOUCHED since Sep 19 (`d49ca38f2`), so the closure is a property of
/// the program shape, not of the instrument.
///
/// ⛔ ONE SET OF BYTES, AND THAT IS DELIBERATE RATHER THAN A SHORTCUT. `gen_numeric_data.py`'s
/// `_swiglu_inputs` reads only M/D_MODEL/D_FF, so the stimulus and the reference are byte-identical
/// across the three -- VERIFIED, not assumed: `swiglu_mlp_granite_flat/sha256.txt` and
/// `swiglu_mlp_granite_tiled_k/sha256.txt` are identical files in the tree, generated independently
/// on the pod under each config's own name, and `the_granite_blockings_share_one_recorded_sha_map`
/// asserts it. So the refusing blockings are executed against the SAME bindings the running one
/// gets -- three lowerings of one program over one input, which is the only arrangement in which
/// "one blocking fits and another does not" is a statement about the blocking.
///
/// # WHAT THIS TEST OWES BEYOND THE NUMBER
///
/// The comparison alone would be one number with nothing behind it, so this also runs, ON THE REAL
/// STAGED STIMULUS AT THE REAL WIDTH:
///
/// * **The BRACKETING control** -- 1.5x the per-element allowance must exceed, 0.5x must not.
/// * **An independent-implementation cross-check** -- `mutants::mlp`'s f64 model of the same
///   function must sit INSIDE the envelope against the torch f16 `ref_out.bin`. Two authored
///   implementations agreeing at 4096/12800 is what makes the reference more than one opinion.
/// * **Three DISCRIMINATING mutants**, each propagated through the REAL down projection:
///   `down_transposed` (the `q@k` vs `q@k.T` defect in the MLP's clothing), and `swap_gu` /
///   `no_sigmoid` reused verbatim from `mutants::swiglu` so the activation controls are the
///   already-validated ones rather than a second copy. Each must EXCEED.
///
/// ⚠️ `sig_of_u` IS NOT A FOURTH CONTROL. In exact arithmetic it is the same function as `swap_gu`
/// -- both are `g * u * sigmoid(u)` -- so counting it would be overcounting; `mutants::swiglu`'s
/// own doc and `swap_gu_is_exact_and_is_the_operand_swap_it_claims_to_be` say so.
///
/// # ⛔⛔ THE LIMIT OF THE CLAIM: THIS BLOCKING DOES NOT BAKE
///
/// `KTIR_WHOLE=1 bake_py swiglu_mlp_granite_tiled_k` REFUSES at `regions` -- the SuperDSC
/// whole-function door walks the function's TOP LEVEL and this blocking's windows are inside an
/// `scf.for` body -- while `swiglu_mlp_granite_flat` bakes (ops=7) and reaches `dxp_standalone`
/// exit 0. The two consumers want OPPOSITE blockings and BOTH are right about their own model;
/// `cases.rs` carries the measurement and the reason. So what this test establishes is that the
/// FUNCTION `swiglu_mlp.py` computes is correct at 4096/12800 -- a property of the `.py`, not of a
/// blocking, which is exactly the claim `_flat` could not make because it cannot execute. It does
/// NOT establish that the baked descriptors read the right bytes; nothing here is a layout check,
/// and `dxp_standalone` executes no arithmetic, so neither result substitutes for the other.
///
/// ⛔ STILL `#[ignore]`d, because the bytes are 315,621,376 B and genuinely are not in the tree --
/// the same reason the sweep lists these configs as staged-only. BUT AN UNSET VARIABLE IS NOW A
/// PANIC AND NOT A SKIP: this test used to `continue` with "SKIPPED", so `--ignored` passed while
/// measuring nothing, which is the vacuous-green failure the whole crate exists to prevent.
///
/// ```bash
/// TRITON_NUMERIC_STAGED_SWIGLU_MLP_GRANITE_TILED_K=<dir> \
/// TRITON_NUMERIC_STAGED_SWIGLU_MLP_GRANITE_FLAT=<dir> \
///   cargo test --offline --release -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the 315 MiB staged weights; set TRITON_NUMERIC_STAGED_SWIGLU_MLP_GRANITE_TILED_K \
            (see test/numeric/swiglu_mlp_granite_tiled_k/meta.json's `staging` note)"]
fn granite_width_swiglu_at_three_blockings() {
    const RUNS: &str = "swiglu_mlp_granite_tiled_k";
    /// The middle blocking, now a RUNNING comparison (see REFUSE's history note below).
    const MIDDLE: &str = "swiglu_mlp_granite";
    /// The blocking that does NOT fit, with the op that charged the overflow and the tile it
    /// charged. `_flat`'s two `[64, 12800]` f16 tiles (gate and up, 1,638,400 B each) exceed LX
    /// before their elementwise product. RECORDED PER CONFIG so the check cannot let one config's
    /// failure quietly become another's.
    ///
    /// `swiglu_mlp_granite` USED TO SIT HERE and no longer does -- it executed on the pod once the
    /// accumulate form was restored in the straight-line program; the test's mirror-guard fired
    /// exactly as designed and the config moved to the running comparisons below, with its own
    /// `test/numeric/swiglu_mlp_granite/` fixture and envelope. The full attribution is in this
    /// test's doc comment.
    const REFUSE: &[(&str, &[&str])] =
        &[("swiglu_mlp_granite_flat", &["TensorSplat", "LX capacity exceeded", "[64, 12800]"])];

    // ---- the configuration that RUNS, and its bytes ----------------------------------------
    let var = format!("TRITON_NUMERIC_STAGED_{}", RUNS.to_uppercase());
    let staged = std::env::var(&var).unwrap_or_else(|_| {
        panic!(
            "⛔ `{var}` is unset, so this test would measure NOTHING. It is `#[ignore]`d precisely \
             because the 315,621,376 B of Granite weights are not in the tree; running it with \
             `--ignored` and no staged directory is the vacuous pass this panic exists to prevent. \
             Stage with `python3 test/numeric/gen_numeric_data.py {RUNS} <dir>` (torch 2.11.0+cpu) \
             or copy from the pod -- see that config's meta.json `staging` note."
        )
    });
    println!("=== {RUNS} <- {staged} ===");

    let f = data::Fixture::load(RUNS).expect("the tiled-k fixture metadata");
    let env = envelope(&f);
    let (m, d_model, d_ff) = (
        f.int("M").expect("M") as usize,
        f.int("D_MODEL").expect("D_MODEL") as usize,
        f.int("D_FF").expect("D_FF") as usize,
    );
    // ⛔ THE WIDTH IS ASSERTED, NOT ASSUMED. The whole point of this test is a number AT GRANITE
    // WIDTH; a staged directory generated at a smaller width would otherwise produce a green
    // result that means nothing, and its extents all agree with each other.
    assert_eq!(
        (m, d_model, d_ff),
        (64, 4096, 12800),
        "this test's claim is a number at REAL Granite width; {RUNS} is staged at {:?}",
        (m, d_model, d_ff)
    );
    assert_eq!(
        (f.int("BLOCK_N").expect("BLOCK_N"), f.int("BLOCK_K").expect("BLOCK_K")),
        (64, 2048),
        "the blocking the envelope was derived for"
    );
    println!(
        "  bound   k+ = {:.6e}  k- = {:.6e}  floor = {:.6e}   (T_k = {}, T_n = {})",
        env.k_plus,
        env.k_minus,
        env.floor,
        d_model / 2048,
        d_ff / 64
    );

    let bindings = f.bindings().expect("the staged bytes");
    let reference = f.reference().expect("the staged reference");

    // ---- the two blockings that REFUSE, on those same bindings ------------------------------
    //
    // The five pointer arguments' extents are M/D_MODEL/D_FF, which all three configurations share,
    // so one set of bindings is the RIGHT input for each of the three lowerings. `execute` checks
    // the count and each buffer's declared shape against the lowered function, so a mismatch would
    // refuse rather than mis-bind.
    for (config, want) in REFUSE {
        let lowered = triton_numeric::lower(config).expect("the chain lowers before executing");
        let why = match triton_numeric::execute(&lowered, &bindings) {
            Err(e) => e.to_string(),
            Ok(_) => panic!(
                "⭐ `{config}` NO LONGER REFUSES. The LX gap closed -- delete it from REFUSE, give \
                 it a `test/numeric/` directory and let the comparison gate it. This is the mirror \
                 of a recorded gap that quietly becomes a different gap."
            ),
        };
        println!("  {config:28} REFUSED  {why}");
        assert!(
            want.iter().all(|w| why.contains(w)),
            "{config}: an UNEXPECTED refusal. The recorded one names {want:?}; got: {why}"
        );
    }

    // ---- the number --------------------------------------------------------------------------
    let lowered = triton_numeric::lower(RUNS).expect("the tiled-k chain lowers");
    let out = triton_numeric::execute(&lowered, &bindings)
        .unwrap_or_else(|e| panic!("⛔ {RUNS} REFUSED at the executor: {e}"));
    let got = out.f64s(&f.output.arg).expect("the output reads back");
    let c = Comparison::new(RUNS, &got, &reference, &env).expect("the comparison");
    println!("  {}", data::banner(&f));
    println!("  {}", c.report());
    bracket_control(RUNS, &reference, &env);
    assert!(
        c.within_bound(),
        "⛔⛔ {RUNS} EXCEEDED its derived bound at Granite width. The K-length accumulation at 4096 \
         and the D_FF walk at 12800 are exactly where a fold defect hides, so this is the most \
         valuable failure in the suite -- diagnose it, do NOT loosen the bound: {}",
        c.report()
    );

    // ---- the middle blocking's number, at ITS OWN envelope ----------------------------------
    //
    // `swiglu_mlp_granite` (BLOCK_K = 4096, t_k = 1) refused at this executor until the accumulate
    // form was restored in the straight-line program -- the doc comment above carries the full
    // attribution. It now runs against the SAME bindings (the three configs' bytes are one set,
    // asserted by `the_granite_blockings_share_one_recorded_sha_map`) but under its OWN derived
    // envelope: `bounds::swiglu` charges the K-trip chain from this config's BLOCK_K, and at t_k=1
    // that is TIGHTER than tiled_k's t_k=2 -- so a shared envelope would gate this comparison more
    // loosely than its own arithmetic allows, which is exactly the arrangement this suite refuses.
    let fm = data::Fixture::load(MIDDLE).expect("the middle blocking's fixture metadata");
    assert_eq!(
        (fm.int("BLOCK_N").expect("BLOCK_N"), fm.int("BLOCK_K").expect("BLOCK_K")),
        (64, 4096),
        "the middle blocking the envelope below was derived for"
    );
    let env_mid = envelope(&fm);
    let lowered_mid = triton_numeric::lower(MIDDLE).expect("the middle chain lowers");
    let out_mid = triton_numeric::execute(&lowered_mid, &bindings)
        .unwrap_or_else(|e| panic!("⛔ {MIDDLE} REFUSED at the executor: {e}"));
    let got_mid = out_mid.f64s(&fm.output.arg).expect("the output reads back");
    let c_mid = Comparison::new(MIDDLE, &got_mid, &reference, &env_mid).expect("the comparison");
    println!("  {}", data::banner(&fm));
    println!("  {}", c_mid.report());
    bracket_control(MIDDLE, &reference, &env_mid);
    assert!(
        c_mid.within_bound(),
        "⛔⛔ {MIDDLE} EXCEEDED its derived bound at Granite width. t_k = 1 is the SINGLE-TRIP \
         chain, so this is the tightest envelope of the three blockings -- a failure here with \
         tiled_k green would localise a defect to the fold's accumulate chaining: {}",
        c_mid.report()
    );

    // ---- the controls, on the real stimulus at the real width --------------------------------
    //
    // The operands come out of the SAME bindings the executor ran on, decoded with the SAME decoder
    // (`data::decode_f64` -> `ktir_core::codec::decode`), so the control and the comparison cannot
    // disagree about what an f16 bit pattern means.
    let operand = |arg: &str| -> Vec<f64> {
        let b = bindings
            .iter()
            .find(|b| b.name == arg)
            .unwrap_or_else(|| panic!("no binding for `{arg}`"));
        data::decode_f64(&b.bytes, b.shape.iter().product(), b.dtype).expect("decodes")
    };
    // `projections` is two thirds of a whole MLP at this width, so it runs ONCE and every mutant
    // reuses it; `wg`/`wu` go out of scope straight after, which is 840 MB of f64 returned.
    let (g, u) = {
        let (x, wg, wu) = (operand("desc_x"), operand("desc_wg"), operand("desc_wu"));
        mutants::mlp::projections(&x, &wg, &wu, m, d_model, d_ff)
    };
    let wd = operand("desc_wd");
    let truth = mutants::mlp::down(&mutants::mlp::silu_gate(&g, &u), &wd, m, d_model, d_ff);

    // ⭐ TWO INDEPENDENT IMPLEMENTATIONS OF THE SAME FUNCTION, AT GRANITE WIDTH. `mutants::mlp` is
    // f64 Rust; `ref_out.bin` is the generator's f32 torch with one f16 cast. If these disagreed,
    // the number above would be gating the emulator against one of two opinions and nothing would
    // say which.
    let cross = Comparison::new("mlp::truth vs ref_out", &truth, &reference, &env)
        .expect("the cross comparison");
    println!("  cross-check (f64 model vs torch reference)\n  {}", cross.report());
    assert!(
        cross.within_bound(),
        "⛔ THE AUTHORED f64 MODEL AND THE AUTHORED TORCH REFERENCE DISAGREE at Granite width, so \
         the emulator's result above is being compared against an answer that is itself in \
         question. This is a finding about the REFERENCE, not about the kernel: {}",
        cross.report()
    );

    for (name, mutated) in [
        // The down projection contracts against Wd instead of Wd^T -- same buffer, same element
        // count, wrong stride. The MLP's form of the defect that passed the dxp gate for hours.
        (
            "down_transposed",
            mutants::mlp::down_transposed(&mutants::mlp::silu_gate(&g, &u), &wd, m, d_model, d_ff),
        ),
        // The two activation mutants, REUSED from `mutants::swiglu` and pushed through the real
        // down projection. Not re-written here: `silu_gate_is_the_swiglu_modules_truth` pins the
        // faithful activation to that module's, so these are the same controls the elementwise
        // harness already validates, now measured at Granite width.
        (
            "swap_gu",
            mutants::mlp::down(&mutants::swiglu::swap_gu(&g, &u), &wd, m, d_model, d_ff),
        ),
        (
            "no_sigmoid",
            mutants::mlp::down(&mutants::swiglu::no_sigmoid(&g, &u), &wd, m, d_model, d_ff),
        ),
    ] {
        let (dev, bound) = mutants::max_abs_deviation(&truth, &mutated, &env);
        let ratio = mutants::worst_ratio(&truth, &mutated, &env);
        let frac = mutants::changed_row_fraction(&truth, &mutated, d_model);
        println!(
            "  mutant {name:16} max|dev| = {dev:.6e}  bound = {bound:.6e}  worst ratio = \
             {ratio:.4e}  changed rows = {frac:.4}"
        );
        assert!(
            mutants::exceeds(&truth, &mutated, &env),
            "⛔⛔ `{name}` DOES NOT EXCEED the Granite-width envelope: max|dev| = {dev:.6e} against \
             {bound:.6e} (worst ratio {ratio:.4e}). The comparison above is then vacuous at this \
             width. REPORT IT rather than loosening anything."
        );
        assert!(
            mutants::discrimination_ok(&truth, &mutated, d_model),
            "`{name}` changes only {frac:.4} of the words in its thinnest row; the predeclared \
             floor is {}",
            mutants::MUTANT_MIN_ROW_FRACTION
        );
    }
}

/// ⛔ THE THREE GRANITE BLOCKINGS' BYTES ARE ONE SET, AND THAT IDENTITY IS CHECKED IN THE TREE.
///
/// `granite_width_swiglu_at_three_blockings` executes three lowerings against ONE set of bindings.
/// That is only sound while the three configurations' stimulus and reference really are the same
/// bytes -- which they are BY CONSTRUCTION (`gen_numeric_data.py`'s `_swiglu_inputs` reads only
/// M/D_MODEL/D_FF; BLOCK_N/BLOCK_K never reach it) and BY MEASUREMENT: the two Granite directories'
/// `sha256.txt` were produced by independent generator runs under each config's own name, on the
/// pod, and are identical files.
///
/// This is the same argument `embedding_granite_bm128` makes with `data_dir`, and it needs its own
/// assertion because `data_dir` cannot express it: that field resolves against a SIBLING IN-TREE
/// directory, and these bytes are not in the tree at all.
///
/// NEEDS NO STAGED DATA -- it compares two checked-in `sha256.txt` files -- so it is not ignored.
#[test]
fn the_granite_blockings_share_one_recorded_sha_map() {
    let read = |config: &str| {
        let p = data::dir(config).join("sha256.txt");
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
    };
    let flat = read("swiglu_mlp_granite_flat");
    let tiled = read("swiglu_mlp_granite_tiled_k");
    let middle = read("swiglu_mlp_granite");
    assert!(!flat.trim().is_empty(), "an empty sha map would make this assertion vacuous");
    assert!(
        flat == tiled && flat == middle,
        "⛔ THE THREE GRANITE BLOCKINGS' RECORDED BYTES DIFFER, so executing them against ONE set of \
         bindings compares at least one of them against another configuration's answer. Either a \
         blocking constexpr now reaches `_swiglu_inputs` -- in which case each config needs its own \
         staged bytes and `granite_width_swiglu_at_three_blockings` must stop sharing -- or one of \
         the maps is stale."
    );
}

/// ⭐ COULD THE TWO DECODER RESULTS BE VACUOUS? THE CROSS-COMPARISON SAYS NO.
///
/// `decoder_layer_one_flat` and `decoder_two_layers_flat` produce the SAME SHAPE -- `[64, 128]` f16,
/// 8192 elements -- from the same stimulus at the same extents, differing only in how many layers ran.
/// So their references are interchangeable as far as every LENGTH and SHAPE check in this crate can
/// tell, which makes them the sharpest available test of whether the comparison is looking at
/// anything: feed each configuration's OUTPUT the OTHER's reference and require it to EXCEED.
///
/// ⛔ THIS IS NOT THE BRACKETING CONTROL AND DOES NOT REPLACE IT. Bracketing proves the threshold sits
/// where the algebra puts it, by moving ONE element by a known multiple of its own allowance. This
/// proves something different and coarser: that the bytes being compared are THIS configuration's and
/// not a plausible neighbour's. A harness that read `ref_out.bin` from the wrong directory -- which is
/// one `data_dir` away, and `embedding_granite_bm128` legitimately uses `data_dir` -- would pass every
/// bracketing control in the file while comparing the wrong answer.
#[test]
fn one_layer_and_two_layers_are_not_interchangeable() {
    let one = data::Fixture::load("decoder_layer_one_flat").expect("the one-layer data");
    let two = data::Fixture::load("decoder_two_layers_flat").expect("the two-layer data");
    let r1 = one.reference().expect("one-layer reference");
    let r2 = two.reference().expect("two-layer reference");
    assert_eq!(
        r1.len(),
        r2.len(),
        "the two references must be the same length for this control to be the sharp one it claims \
         to be; if they are not, a length check already separates them and this test is easier than \
         advertised"
    );

    for (name, got, wrong_ref) in
        [("decoder_layer_one_flat", &r1, &r2), ("decoder_two_layers_flat", &r2, &r1)]
    {
        let f = if name == "decoder_layer_one_flat" { &one } else { &two };
        let env = envelope(f);
        let c = Comparison::new(name, got, wrong_ref, &env).expect("a cross comparison");
        println!("cross-control {name} vs the other configuration's reference:\n  {}", c.report());
        assert!(
            !c.within_bound(),
            "⛔⛔ {name}'s OUTPUT SITS INSIDE ITS BOUND AGAINST THE **OTHER** CONFIGURATION'S \
             REFERENCE. One decoder layer and two are then indistinguishable to this comparison, so \
             its green result says nothing about which program ran. {}",
            c.report()
        );
    }
}
