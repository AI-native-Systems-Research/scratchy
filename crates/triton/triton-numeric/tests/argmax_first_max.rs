// SPDX-License-Identifier: Apache-2.0

//! THE LADDER-FORM SAMPLER, EXECUTED: `argmax_first_max` through the whole numeric chain --
//! `.py` → TTIR → KTIR → `ktir_core::IRFunction` → the ktir emulator -- against the fixture's
//! own torch reference.
//!
//! This is the rung ABOVE the door: `triton-numeric` stops before the SuperDSC descriptor
//! emission (the case row's door kind is `None` and says why), so a green result here states
//! that the KERNEL'S ARITHMETIC -- the five whole-row maxima, the three equality masks,
//! the four staged index tables, the fp32-island combine, the index cast -- computes the right FUNCTION
//! on the device model. It does not state that the descriptors address it right; that is the
//! next rung's gate.
//!
//! # WHAT THE COMPARISON IS: EXACT, AND WHY THAT IS HONEST HERE
//!
//! Every other configuration in this crate compares under a DERIVED ROUNDING ENVELOPE because
//! its output is an f16 tile produced by f16 arithmetic. This kernel's output is an i32 INDEX,
//! and the index is exact by construction: the whole-row maxima combine real f16 values with
//! no arithmetic beyond `maximum`, both equality masks are exact, the two-level ramp products
//! stay in the fp16-exact small-integer range (block < 1024, lane < 64), and the one wide
//! integer (`b·64 + l`, reaching 49154) rides the fp32 island where every integer below 2^24
//! is exact. So the bound is ZERO, stated as
//! equality, and [`Comparison`] is not used: an envelope's ratio machinery over a zero bound
//! reports `inf` or `0` and nothing in between, which would dress an exact comparison as a
//! rounded one. `max|err| = 0` over all 64 lanes is the claim, and every lane is asserted.
//!
//! # WHAT THE CONTROLS ARE
//!
//! `make_range.rs` states the law this file follows: a comparison that cannot fail proves
//! nothing, and for an exact comparison the failing control is a DIFFERING ANSWER, not a
//! bracket. The stimulus carries a DELIBERATE tie at HIGH lanes: the generator pins the
//! row's maximum (one above the natural max -- a 49k-lane fp16 randn row ties its own
//! maximum naturally, so pinning AT it would leave an earlier natural lane winning) at
//! lanes 40017 and 48000, and BOTH placements are load-bearing:
//!
//! 1. **The last-max control**: the LAST maximal index of the same bytes is computed here
//!    from the stimulus itself, not restated -- 48000, not 40017. An implementation that
//!    breaks the tie the wrong way answers the last index and the exact comparison sees
//!    it. Without ties the two answers would coincide and no tie-break defect would be
//!    visible; the control proves the stimulus discriminates.
//! 2. **The f16-degrade control**: the ANSWER 40017 = 625*64 + 17 is not fp16-representable
//!    (ULP 32 at [32768, 65536)), so a kernel whose `b*64 + l` island collapsed out of f32
//!    answers 40032 and the exact comparison sees it. This is not hypothetical: the
//!    collapsed form answered 40032 on a green run before `LegalizeTypes` learned to
//!    protect the fptosi-terminated island (measured, 2026-10-21). A tie at LOW lanes
//!    (the original 100/5000 stimulus) sits inside fp16's exact-integer range and cannot
//!    see this defect at all.
//! 3. **The padoff control**: the row's pads stage 0.0 -- the PRODUCTION shape, the
//!    lm-head's zero-pad columns -- and the row's real lanes sit BELOW ZERO, so a kernel
//!    that maxes the RAW row answers a pad lane (0.0 out-maxes every real lane) and the
//!    exact comparison sees it. Only `x + padoff` (the staged offset row, -inf on the
//!    pads) keeps the answer on the real lanes; a positive-max stimulus cannot see a
//!    missing offset table at all, which is why the fixture shifts its row negative.
//! 4. **The lane-uniformity claim is asserted, not assumed**: every lane of the [64] i32 stick
//!    must hold the same index. The kernel computes one vector and stores it whole; a lane that
//!    disagrees is a stick-addressing defect the single-lane reference cannot see on its own.
//!
//! # WHAT THIS RUNG ALSO MEASURES, AS A SIDE EFFECT
//!
//! `make_range.rs` documents that no executed fixture before it plain-STORES an i32 tile
//! (embedding's `*i32` ids feed `tt.descriptor_gather`, not a store). This kernel's output IS a
//! plain i32 descriptor store, so this is the first executed configuration of that path -- a
//! refusal here may name the store leg and not the sampler's arithmetic, and the stage in the
//! refusal says which.

use triton_numeric::{data, execute, lower};

/// The stimulus's logits row, decoded. Read from the same checked bytes the run binds, so the
/// control and the execution cannot disagree about what the row holds.
fn logits_row(f: &data::Fixture) -> Vec<f64> {
    input_row(f, "logits_ptr")
}

/// Any staged input, decoded — the padoff control reads the offset row the same way the
/// run binds it.
fn input_row(f: &data::Fixture, arg: &str) -> Vec<f64> {
    let t = f
        .inputs
        .iter()
        .find(|t| t.arg == arg)
        .unwrap_or_else(|| panic!("the fixture records a `{arg}` input"));
    let p = data::dir(&f.config).join(&t.file);
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    data::decode_f64(&bytes, t.elements(), t.dtype)
        .unwrap_or_else(|e| panic!("decode {}: {e}", p.display()))
}

/// The FIRST index holding the row's maximum over the real lanes (`0..V`), computed on f64s --
/// exact, since f16 values widen exactly and equality on widened f64s is the same predicate.
fn first_max(row: &[f64]) -> usize {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    row.iter().position(|&x| x == m).expect("a finite maximum")
}

/// The LAST index holding it -- the tie-break mutant.
fn last_max(row: &[f64]) -> usize {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    row.iter().rposition(|&x| x == m).expect("a finite maximum")
}

#[test]
fn the_sampler_executes_first_max_exactly() {
    let f = data::Fixture::load("argmax_first_max").expect("the checked-in data loads");
    let bindings = f.bindings().expect("the bindings stage");
    let lowered = lower("argmax_first_max").expect("the numeric legs lower the kernel");
    let run = execute(&lowered, &bindings).expect("the emulator runs the kernel");

    let got = run.f64s("idx_ptr").expect("idx_ptr comes back");
    let reference = f.reference().expect("the reference re-reads");
    assert_eq!(reference.len(), 1, "the reference is one index");
    let want = reference[0];

    // EVERY LANE, EXACTLY: lane 0 is the answer the consumer reads, and lanes 1..63 must hold
    // the same index -- the kernel computes one [BLOCK] vector and stores the whole stick. A
    // differing lane is a stick defect a single-lane comparison would miss.
    assert_eq!(got.len(), 64, "the output is one whole i32 stick");
    let mut max_abs_err = 0.0f64;
    for (i, &g) in got.iter().enumerate() {
        let err = (g - want).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        assert_eq!(
            g, want,
            "lane {i} answered {g}, the reference is {want}: the index is exact by \
             construction (fp32 exact to 2^24, V = 49155), so any deviation is a defect"
        );
    }
    println!("  argmax_first_max: 64/64 lanes exact, max|err| = {max_abs_err:e}, idx = {want}");
}

/// THE NEGATIVE CONTROL: the stimulus's tie must be able to fail the comparison. The last-max
/// index of these same bytes is 48000 (the tie's other lane); if it equalled the first-max
/// answer the tie would be vacuous and the exact comparison above could not see a tie-break
/// defect -- every wrong-way implementation would pass.
#[test]
fn the_stimulus_tie_discriminates() {
    let f = data::Fixture::load("argmax_first_max").expect("the checked-in data loads");
    let row = logits_row(&f);
    // The real lanes: V from the fixture's own constexprs, not restated.
    let v = f.int("V").expect("V is recorded") as usize;
    assert!(row.len() > v, "the row is padded past V");
    let real = &row[..v];
    // The pad contract, PRODUCTION shape: the pads hold 0.0 (the lm-head's zero-pad
    // columns write 0.0 there every step) and the real lanes sit below zero, so the
    // raw row's max IS a pad lane -- only the staged offset row (-inf past V, added
    // before the kernel's first max) keeps the answer on the real lanes.
    assert!(
        real.iter().all(|&x| x.is_finite()),
        "the real lanes are finite, else the pad control below is vacuous"
    );
    assert!(
        row[v..].iter().all(|&x| x == 0.0),
        "the pad lanes hold 0.0 (the production shape -- the lm-head's zero-pad columns)"
    );
    assert!(
        real.iter().cloned().fold(f64::NEG_INFINITY, f64::max) < 0.0,
        "the real lanes sit below zero, else the padoff control is vacuous (a positive \
         max beats the 0.0 pads with or without the offset row)"
    );
    // THE OFFSET ROW itself: 0.0 on the real lanes (x + 0 exact), -inf on the pads --
    // the host obligation that carries the pad law.
    let padoff = input_row(&f, "padoff_ptr");
    assert_eq!(
        padoff.len(),
        row.len(),
        "the offset row covers the whole padded row"
    );
    assert!(
        padoff[..v].iter().all(|&x| x == 0.0)
            && padoff[v..].iter().all(|&x| x == f64::NEG_INFINITY),
        "the offset row is 0.0 on the real lanes and -inf on the pads (the pad law the \
         kernel adds before its first max)"
    );

    let first = first_max(real);
    let last = last_max(real);
    let reference = f.reference().expect("the reference re-reads")[0] as usize;
    assert_eq!(
        first, reference,
        "the derived first-max agrees with the reference (else the control compares the \
         wrong predicate)"
    );
    assert_ne!(
        first, last,
        "VACUOUS TIE: first-max and last-max coincide, so the exact comparison cannot see a \
         tie-break defect. Regenerate the stimulus (the fixture ties lanes 40017 and
         48000) --\
         this failure means the tie is gone."
    );
    println!(
        "  argmax_first_max control: first-max {first} vs last-max {last} -- the tie discriminates"
    );
}
