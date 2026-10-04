//! DISCRIMINATING CONTROLS: wrong-but-plausible algorithms, and the bracketing control.
//!
//! A tolerance that nothing can fail is not a test. Each mutant here is a WRONG algorithm a
//! plausible bug would produce, and the harness asserts each one EXCEEDS the envelope its true
//! counterpart sits inside. Ported from `third_party/spyre/test/pod/*_oracle2.py` and
//! `*_negctl.py`; the five configs with no pod ancestor (rope x2, embedding x2, decoder x2 —
//! authored here) are marked AUTHORED in their module docs.
//!
//! TWO THINGS THIS FILE REFUSES TO DO, both of them lessons from the pod apparatus:
//!
//!   1. IT DOES NOT HIDE A CONTROL THAT CANNOT DISCRIMINATE. `rmsnorm_oracle2.py:21-24` reports
//!      "the full 64-stick partition HONESTLY: some sticks carry so little energy that omitting
//!      them rounds away below one DL16 ULP (a sub-ULP no-op, numerically undetectable)". The port
//!      of that honesty is [`rmsnorm::stick_drop_materiality`], which returns the material and
//!      immaterial counts so a caller can NAME the weak controls instead of quietly dropping them.
//!      A mutant that cannot exceed the bound is a fact about the bound, not a bug to be tidied.
//!
//!   2. IT DOES NOT PRETEND A ONE-SIDED CONTROL PROVES A THRESHOLD. [`bracket`] is the whole point
//!      of the file: `1.5x` the per-element bound must FAIL and `0.5x` must PASS. A one-sided
//!      control only shows the gate CAN fire; bracketing shows it fires exactly where the
//!      derivation puts it (`fp_negctl.py:13-19`, `swiglu_negctl.py:10-13`).
//!
//! Everything here is a pure function over `f64` slices: no emulator, no IR, no files. The
//! reference builders take the same inputs as their `truth` counterpart and return the MUTANT's
//! output, so a test asserts `env.contains(truth_i, mutant_i)` is false somewhere.

use crate::bounds::Envelope;

/// One control's identity. `informational` marks a mutant that may legitimately sit INSIDE the
/// bound — `half_sigmoid` is the pod's example (`swiglu_negctl.py:20-22`: "SUBTLE ... ->
/// INFORMATIONAL (may be WITHIN the loose bound; Oracle-2's byte-exact model discriminates it)").
/// An informational control is REPORTED, never asserted, and never counted as discrimination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mutant {
    pub name: &'static str,
    pub informational: bool,
}

impl Mutant {
    pub const fn discriminating(name: &'static str) -> Self {
        Mutant {
            name,
            informational: false,
        }
    }
    pub const fn informational(name: &'static str) -> Self {
        Mutant {
            name,
            informational: true,
        }
    }
}

/// The PREDECLARED near-total per-row changed-word floor a mutant must clear to count as a
/// discriminator.
///
/// Ported verbatim from `swiglu_oracle2.py:50-56`, including WHERE THE NUMBER COMES FROM, because
/// a floor chosen after seeing the results would be worthless: "PREDECLARED near-total per-row
/// threshold (Codex M10 B3): the discriminator is only meaningful if each mutant changes
/// ESSENTIALLY every output word of every row. The `md>0` (any-change) test let a
/// one-word-per-row mutant pass. We require the MINIMUM per-row changed fraction across all rows
/// to be >= this threshold. The real candidate mutants change ~98.6-99.9% of each row; 0.90 is a
/// conservative predeclared floor (fixed BEFORE inspecting results, well below the observed ~0.986
/// minimum and far above a token one-word 1/12800 ~= 8e-5 change). A mutant that only flips a
/// handful of words FAILS."
pub const MUTANT_MIN_ROW_FRACTION: f64 = 0.90;

/// The MINIMUM over rows of the changed-element fraction between `a` and `b`.
///
/// The minimum, not the mean: `swiglu_oracle2.py:59-68`'s `mutant_discrimination_ok` gates on
/// `min(per_row_changed) / N` precisely so a mutant that mangles one row and leaves the rest alone
/// is REJECTED as a discriminator. Compare against [`MUTANT_MIN_ROW_FRACTION`].
///
/// "Changed" here is exact `f64` inequality — the emulator-side comparison is a DL16-word compare
/// in the pod (`np.count_nonzero(mut != y_dev[r])`), and on the f64 reference path the analogue is
/// bit inequality. A mutant that produces bit-identical output has changed nothing, whatever its
/// intent.
///
/// # Panics
/// If the slices differ in length, if `row_len` is zero, or if the length is not a whole number of
/// rows — every one of those is a harness wiring bug, and a silently-truncated comparison would
/// under-report the changed fraction.
pub fn changed_row_fraction(a: &[f64], b: &[f64], row_len: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "changed_row_fraction: length mismatch");
    assert!(row_len > 0, "changed_row_fraction: row_len must be positive");
    assert_eq!(
        a.len() % row_len,
        0,
        "changed_row_fraction: {} elements is not a whole number of {row_len}-wide rows",
        a.len()
    );
    let mut worst = f64::INFINITY;
    for (ra, rb) in a.chunks(row_len).zip(b.chunks(row_len)) {
        let changed = ra.iter().zip(rb).filter(|(x, y)| x != y).count();
        let frac = changed as f64 / row_len as f64;
        if frac < worst {
            worst = frac;
        }
    }
    if worst.is_finite() {
        worst
    } else {
        0.0
    }
}

/// True iff the changed-word fraction clears the predeclared floor — the single shared verdict
/// helper, so a test exercises the same decision the harness makes
/// (`swiglu_oracle2.py:59-68`'s reason for having exactly one of these).
pub fn discrimination_ok(a: &[f64], b: &[f64], row_len: usize) -> bool {
    changed_row_fraction(a, b, row_len) >= MUTANT_MIN_ROW_FRACTION
}

/// THE BRACKETING CONTROL. Perturb the element with the LARGEST per-element UPWARD allowance by
/// `factor` times that allowance, and return the mutated vector plus the index it touched.
///
/// `factor = 1.5` must make the comparison FAIL; `factor = 0.5` must leave it PASSING. Ported from
/// `swiglu_negctl.py:59-74` / `rmsnorm_negctl.py:64-67` / `softmax_negctl.py:63-66`, which all
/// pick `argmax(up)` and step by `scale * b_ij`.
///
/// IT PERTURBS AGAINST THE UPWARD ALLOWANCE, NOT `abs_bound`. The band is sign-aware, so for a
/// negative reference the upward allowance is the SMALLER coefficient (`k_minus`); stepping by
/// `1.5 * abs_bound` there would clear the gate for the wrong reason — it would prove the band is
/// asymmetric, not that the threshold is calibrated. Stepping by `1.5 * up_allowance` crosses
/// exactly the edge the derivation placed.
///
/// The chosen index is the one whose allowance is largest, i.e. the element where a fixed relative
/// error is hardest to detect — the pod picks it for the same reason.
///
/// # Panics
/// If `reference` is empty.
pub fn bracket(reference: &[f64], env: &Envelope, factor: f64) -> (Vec<f64>, usize) {
    assert!(!reference.is_empty(), "bracket: empty reference");
    let mut idx = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &r) in reference.iter().enumerate() {
        let up = env.up_allowance(r);
        if up > best {
            best = up;
            idx = i;
        }
    }
    let mut out = reference.to_vec();
    out[idx] = reference[idx] + factor * best;
    (out, idx)
}

/// The largest normalized excursion of `got` against `reference` under `env`: `<= 1.0` means every
/// element is within band, `> 1.0` means at least one exceeded. Report this number WITH the bound
/// it is normalized by — never "passes".
pub fn worst_ratio(reference: &[f64], got: &[f64], env: &Envelope) -> f64 {
    env.worst_ratio(reference, got)
}

/// The worst raw deviation and the bound that applied at that element: `(max|got-ref|, bound)`.
/// The pair, so a report can print `max|err| = X against a bound of Y` instead of a verdict.
pub fn max_abs_deviation(reference: &[f64], got: &[f64], env: &Envelope) -> (f64, f64) {
    let mut worst = 0.0f64;
    let mut bound_there = 0.0f64;
    for (&r, &g) in reference.iter().zip(got.iter()) {
        let d = (g - r).abs();
        if d > worst {
            worst = d;
            bound_there = env.abs_bound(r);
        }
    }
    (worst, bound_there)
}

/// True iff at least one element of `got` falls outside `env`'s band around `reference` — the
/// verdict a discriminating mutant must produce.
pub fn exceeds(reference: &[f64], got: &[f64], env: &Envelope) -> bool {
    reference
        .iter()
        .zip(got.iter())
        .any(|(&r, &g)| !env.contains(r, g))
}

fn sigmoid(z: f64) -> f64 {
    // Numerically stable both ways — swiglu_eps_derivation.py:81-86's `_s_true`.
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// ===========================================================================================
// SwiGLU — the ELEMENTWISE activation. PORTED from swiglu_oracle2.py / swiglu_negctl.py.
// ===========================================================================================

/// `h = silu(g) * u = (g * sigmoid(g)) * u` and its operator/operand mutants.
///
/// PORTED from `swiglu_emitted_model.py:138-165`'s `mutate` arm and the verdicts in
/// `swiglu_negctl.py:76-101`. Gate these against [`crate::bounds::swiglu_activation`] — the elementwise
/// envelope — NOT against [`crate::bounds::swiglu`], which bounds the three-matmul MLP and carries a
/// contraction floor two decades larger.
///
/// SHAPES: `g` and `u` are flat, equal-length, row-major `[M, N]`; every function returns a fresh
/// `Vec<f64>` of the same length. The extents matter only to [`changed_row_fraction`].
pub mod swiglu {
    use super::{sigmoid, Mutant};

    /// The controls this module offers, with `half_sigmoid` marked informational.
    ///
    /// `sig_of_u` is listed because the ported list names it, but see its own doc: on an exact-f64
    /// reference path it is the SAME FUNCTION as `swap_gu`, so these four entries are three
    /// independent controls plus one informational one.
    pub const MUTANTS: [Mutant; 4] = [
        Mutant::discriminating("no_sigmoid"),
        Mutant::discriminating("swap_gu"),
        Mutant::discriminating("sig_of_u"),
        Mutant::informational("half_sigmoid"),
    ];

    /// The faithful algorithm: `swiglu_oracle.py:151` `true_swiglu = (g*sigmoid(g))*u`, which is
    /// also `swiglu_mlp.py`'s delta 8 (`silu(gate) * up`, NOT upstream's `s * (up + 1)`).
    pub fn truth(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| (gv * sigmoid(gv)) * uv)
    }

    /// `no_sigmoid`: the activation is skipped entirely, `out = g * u`
    /// (`swiglu_emitted_model.py:154-155`). Wrong wherever `sigmoid(g) != 1`, i.e. everywhere.
    /// It also violates the silu-magnitude hard gate `|h| <= |g*u|` (`swiglu_oracle.py:35-38`).
    pub fn no_sigmoid(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| gv * uv)
    }

    /// `swap_gu`: the two operands are exchanged, `out = (u * sigmoid(u)) * g`
    /// (`swiglu_emitted_model.py:156-158`).
    ///
    /// THIS IS THE CONTROL THAT EARNS THE HARNESS. It is the `q@k` vs `q@k.T` defect in swiglu
    /// clothing: nothing about the shapes, the dtypes or the magnitudes changes, only WHICH
    /// OPERAND the nonlinearity and the multiply see. A bound that misses this misses an operand
    /// swap anywhere else in the stack, which is the single most common lowering defect in this
    /// backend. It must be exact and it must be exercised.
    pub fn swap_gu(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| (uv * sigmoid(uv)) * gv)
    }

    /// `sig_of_u`: the sigmoid is applied to the WRONG operand, `out = (g * sigmoid(u)) * u`
    /// (`swiglu_emitted_model.py:159-161`).
    ///
    /// WEAK CONTROL, NAMED AS SUCH: IN EXACT ARITHMETIC THIS IS THE SAME FUNCTION AS
    /// [`swap_gu`]. `(g * sigmoid(u)) * u` and `(u * sigmoid(u)) * g` are both
    /// `g * u * sigmoid(u)`; only the ASSOCIATION differs. The pod's two entries are distinct
    /// controls solely because its model interposes a DL16 round between the two multiplies —
    /// `dl16(dl16(g*sigmoid(u)) * u)` vs `dl16(dl16(u*sigmoid(u)) * g)` round different
    /// intermediates and so produce different device WORDS. On this f64 reference path there is no
    /// intermediate to round, so `sig_of_u` adds NO discriminating power over `swap_gu`.
    ///
    /// It is kept because the harness's mutant list names it and because it separates again the
    /// moment an f16 intermediate is modelled (see [`sig_of_u_f16_intermediate`]) — but a report
    /// that counts it as an independent control is overcounting by one.
    pub fn sig_of_u(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| (gv * sigmoid(uv)) * uv)
    }

    /// [`sig_of_u`] with the FIRST multiply rounded to f16, i.e. the pod's
    /// `dl16(dl16(g*sigmoid(u)) * u)` (`swiglu_emitted_model.py:160-161`) with `dl16` replaced by
    /// the emulator's f16 RNE. Paired with [`swap_gu_f16_intermediate`] it recovers the
    /// distinction the exact-arithmetic form loses. Both round only the intermediate, so they are
    /// still comparable against an f64 `truth` through the envelope.
    pub fn sig_of_u_f16_intermediate(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| {
            crate::bounds::round_f16(gv * sigmoid(uv)) * uv
        })
    }

    /// [`swap_gu`] with the first multiply rounded to f16 — the pod's
    /// `dl16(dl16(u*sigmoid(u)) * g)` (`swiglu_emitted_model.py:157-158`).
    pub fn swap_gu_f16_intermediate(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| {
            crate::bounds::round_f16(uv * sigmoid(uv)) * gv
        })
    }

    /// `half_sigmoid`: a BIASED sigmoid, `out = g * (0.5 + 0.5*sigmoid(g)) * u`
    /// (`swiglu_negctl.py:103-104`).
    ///
    /// LABELLED INFORMATIONAL, and that label is the honest report, not a failure being hidden:
    /// the bias is a smooth O(50%) shift on small |g| but shrinks toward zero for large positive
    /// g, so on some stimuli it can sit inside a loose bound. `swiglu_negctl.py:105-107` prints it
    /// without asserting, and says why: "Oracle-2 byte-exact model discriminates". Assert on it
    /// only after measuring that it exceeds on the stimulus at hand.
    pub fn half_sigmoid(g: &[f64], u: &[f64]) -> Vec<f64> {
        pair(g, u, |gv, uv| gv * (0.5 + 0.5 * sigmoid(gv)) * uv)
    }

    fn pair(g: &[f64], u: &[f64], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        assert_eq!(g.len(), u.len(), "swiglu: g and u must be the same length");
        g.iter().zip(u).map(|(&a, &b)| f(a, b)).collect()
    }
}

// ===========================================================================================
// RMSNorm — the reduction-stick omission. PORTED from rmsnorm_oracle2.py.
// ===========================================================================================

/// RMSNorm and the dropped-reduction-stick mutant.
///
/// SHAPES: `x` is flat row-major `[m, d_model]`, `w` is `[d_model]`, and the reduction axis is
/// partitioned into sticks of `lanes` (64 on this device — `rmsnorm_oracle2.py:84-85` asserts
/// `N == STICKS * LANES`).
pub mod rmsnorm {
    use crate::bounds::Envelope;

    /// The faithful algorithm: `y = x * rsqrt(mean(x^2) + eps) * w`, computed in f64.
    /// `rmsnorm.py:186-200`'s reference, without the final f16 cast (the harness's comparison
    /// applies that through the envelope).
    pub fn truth(x: &[f64], w: &[f64], m: usize, d_model: usize, eps: f64) -> Vec<f64> {
        rows(x, w, m, d_model, eps, None, 64)
    }

    /// Drop the reduction stick carrying the LARGEST sum-of-squares energy from the mean — the
    /// worst-case one-stick omission, `rmsnorm_oracle2.py:129-137`. Every OUTPUT element is still
    /// written; only the mean the whole row is normalized by is wrong, which is exactly what a
    /// scheduler that loses one fold group would produce.
    ///
    /// The stick is chosen PER ROW, as the Python does (`stick_energy = sq.reshape(STICKS,
    /// LANES).sum(axis=1); max_stick = int(np.argmax(stick_energy))`).
    ///
    /// MEASURED MARGIN, AND IT IS THE THINNEST IN THE SUITE. On unit-variance data at
    /// D_MODEL = 4096 with 64-wide sticks, dropping one of 64 sticks moves the mean by ~1/64, and
    /// the rsqrt HALVES that, so the output moves ~0.8% — against `bounds::rmsnorm(4096)`'s
    /// k+ = 6.069e-3 that is a worst_ratio of 2.18, i.e. a 2.0x margin. It discriminates, barely.
    ///
    /// AND IT ONLY DISCRIMINATES BECAUSE THE BOUND WAS RE-DERIVED FOR THIS ORACLE. Under the pod's
    /// device-sourced coefficient (`rmsnorm_oracle.py`'s k+ = 0.0477, dominated by
    /// `eps_rsqrt = 8e-3` — an error the emulator does not commit) the same 0.8% deviation would
    /// sit at a ratio of 0.28 and this control would be VACUOUS. That is the concrete cost of
    /// copying a device bound onto an emulator oracle.
    pub fn drop_max_energy_stick(
        x: &[f64],
        w: &[f64],
        m: usize,
        d_model: usize,
        eps: f64,
        lanes: usize,
    ) -> Vec<f64> {
        rows(x, w, m, d_model, eps, Some(usize::MAX), lanes)
    }

    /// Drop one NAMED stick `k` — the sweep `rmsnorm_oracle2.py:140-145` runs over all sticks to
    /// report which ones are material.
    pub fn drop_stick(
        x: &[f64],
        w: &[f64],
        m: usize,
        d_model: usize,
        eps: f64,
        lanes: usize,
        k: usize,
    ) -> Vec<f64> {
        rows(x, w, m, d_model, eps, Some(k), lanes)
    }

    /// PORTING THE HONESTY. Returns `(material, total)`: how many of the `d_model/lanes` sticks,
    /// when dropped, actually push some element of some row OUTSIDE `env` — and therefore how
    /// many CAN discriminate.
    ///
    /// `rmsnorm_oracle2.py:21-24` reports this rather than only the max-energy stick, because
    /// "some sticks carry so little energy that omitting them rounds away below one DL16 ULP (a
    /// sub-ULP no-op, numerically undetectable)". A caller must PRINT the immaterial count. A
    /// control that cannot discriminate is a property of the bound and the stimulus; suppressing
    /// it would turn a known blind spot into an unknown one.
    ///
    /// MEASURED: on unit-variance data at D_MODEL = 4096 / lanes 64 this returns 64/64 — every
    /// stick is material, because the energy is spread evenly and the re-derived bound is tight.
    /// The pod's sub-ULP no-ops appear when the energy is SKEWED (one dominant stick and the rest
    /// negligible), which is the case the unit test builds deliberately. So the honest statement is
    /// that the weak-control regime is stimulus-dependent, not that it has gone away.
    pub fn stick_drop_materiality(
        x: &[f64],
        w: &[f64],
        m: usize,
        d_model: usize,
        eps: f64,
        lanes: usize,
        env: &Envelope,
    ) -> (usize, usize) {
        let total = d_model / lanes;
        let t = truth(x, w, m, d_model, eps);
        let mut material = 0usize;
        for k in 0..total {
            let mutated = drop_stick(x, w, m, d_model, eps, lanes, k);
            if super::exceeds(&t, &mutated, env) {
                material += 1;
            }
        }
        (material, total)
    }

    /// `drop`: `None` = faithful, `Some(usize::MAX)` = the per-row max-energy stick,
    /// `Some(k)` = stick `k`.
    fn rows(
        x: &[f64],
        w: &[f64],
        m: usize,
        d_model: usize,
        eps: f64,
        drop: Option<usize>,
        lanes: usize,
    ) -> Vec<f64> {
        assert_eq!(x.len(), m * d_model, "rmsnorm: x is not [m, d_model]");
        assert_eq!(w.len(), d_model, "rmsnorm: w is not [d_model]");
        if drop.is_some() {
            assert!(lanes > 0 && d_model.is_multiple_of(lanes), "rmsnorm: d_model must be a whole number of {lanes}-wide sticks");
        }
        let mut out = vec![0.0f64; x.len()];
        for r in 0..m {
            let row = &x[r * d_model..(r + 1) * d_model];
            let drop_k = match drop {
                None => None,
                Some(k) if k == usize::MAX => {
                    // the stick with the most sum-of-squares energy in THIS row
                    let sticks = d_model / lanes;
                    let mut best = 0usize;
                    let mut best_e = f64::NEG_INFINITY;
                    for s in 0..sticks {
                        let e: f64 = row[s * lanes..(s + 1) * lanes].iter().map(|v| v * v).sum();
                        if e > best_e {
                            best_e = e;
                            best = s;
                        }
                    }
                    Some(best)
                }
                Some(k) => Some(k),
            };
            let mut ssq = 0.0f64;
            for (i, &v) in row.iter().enumerate() {
                if let Some(k) = drop_k {
                    if i / lanes == k {
                        continue; // the omitted stick contributes nothing to the mean
                    }
                }
                ssq += v * v;
            }
            let inv = 1.0 / (ssq / d_model as f64 + eps).sqrt();
            for i in 0..d_model {
                out[r * d_model + i] = row[i] * inv * w[i];
            }
        }
        out
    }
}

// ===========================================================================================
// Matmul — the q@k vs q@k.T defect. AUTHORED (test/pod has no oracle2 for the MLP/decoder dots).
// ===========================================================================================

/// A plain contraction and the TRANSPOSED-B defect.
///
/// AUTHORED for this harness. The pod's real-fp matmul apparatus (`oracle_fp.py`, `fp_negctl.py`)
/// has no operand-orientation mutant: its controls are `drophalf` (a dropped split-K partition)
/// and the bracketing pair, and it says explicitly that a subtle permute may NOT exceed its loose
/// bound (`fp_negctl.py:9-12`). The orientation defect is what THIS backend keeps producing —
/// see the branch's own history, "a `.T` on a computed value is a relayout, not a label" — so it
/// is the mutant that has to be carried for every dot in the suite.
///
/// SHAPES: `a` is `[m, k]` row-major; `b` holds `k * n` elements and is read as `[k, n]` by
/// [`truth`] and as `[n, k]` by [`transposed_b`]. THE SAME BYTES, read with the other stride —
/// which is precisely the bug: nothing about the buffer, its size or its dtype announces which
/// reading was meant.
pub mod matmul {
    /// `c[i,j] = sum_p a[i,p] * b[p,j]`, B read as `[k, n]`.
    pub fn truth(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        assert_eq!(a.len(), m * k, "matmul: a is not [m, k]");
        assert_eq!(b.len(), k * n, "matmul: b does not hold k*n elements");
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
        c
    }

    /// THE DEFECT: `c[i,j] = sum_p a[i,p] * b[j,p]`, the same buffer read as `[n, k]`.
    ///
    /// This is `q @ k` where `q @ k.T` was meant (or the reverse). It cannot be caught by a shape
    /// check when `k == n`, which is every square projection in the decoder
    /// (`decoder_block.py:506-509`: wq/wk/wv/wo are all `[d_model, d_model]`), and it produces
    /// output of the RIGHT magnitude — so only a numeric gate can see it.
    pub fn transposed_b(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        assert_eq!(a.len(), m * k, "matmul: a is not [m, k]");
        assert_eq!(b.len(), k * n, "matmul: b does not hold n*k elements");
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }
}

// ===========================================================================================
// The three-matmul SwiGLU MLP. AUTHORED — this is what bounds::swiglu bounds.
// ===========================================================================================

/// `o = (silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T` — `swiglu_mlp.py`'s kernel — and its
/// transposed-operand mutant. AUTHORED; gate against [`crate::bounds::swiglu`].
///
/// SHAPES, taken from the kernel's own descriptors (`swiglu_mlp.py:133-146`, restated at
/// `gen_numeric_data.py:244-252`): `x` is `[m, d_model]`, `wg`/`wu` are `[d_ff, d_model]`, `wd` is
/// `[d_model, d_ff]`. Each `@` therefore takes the transpose, which is HF's `x @ W.T`.
///
/// ⭐ THE THREE STAGES ARE `pub` SEPARATELY, AND THAT IS WHAT MAKES A GRANITE-WIDTH CONTROL
/// AFFORDABLE. At `[64, 4096] x [4096, 12800]` one whole `truth` is ~10 GFLOP of scalar f64; four
/// mutants built as four whole `truth`s would be 40. [`projections`] costs two thirds of one and is
/// the SAME for every activation-level mutant, so a control run reuses it and pays only for each
/// mutant's own [`down`]. It also lets the ALREADY-VALIDATED [`super::swiglu`] mutants (`swap_gu`,
/// `no_sigmoid`) be propagated through the real down projection instead of a second,
/// separately-written MLP-level copy of them.
pub mod mlp {
    use super::sigmoid;

    /// The faithful MLP, f64 throughout.
    pub fn truth(
        x: &[f64],
        wg: &[f64],
        wu: &[f64],
        wd: &[f64],
        m: usize,
        d_model: usize,
        d_ff: usize,
    ) -> Vec<f64> {
        let (g, u) = projections(x, wg, wu, m, d_model, d_ff);
        down(&silu_gate(&g, &u), wd, m, d_model, d_ff)
    }

    /// THE MUTANT: the down projection contracts against `Wd` instead of `Wd^T` — `wd` read as
    /// `[d_ff, d_model]` rather than `[d_model, d_ff]`. Same buffer, same element count, wrong
    /// stride; the output is the right shape and the right magnitude and completely wrong.
    pub fn transposed_wd(
        x: &[f64],
        wg: &[f64],
        wu: &[f64],
        wd: &[f64],
        m: usize,
        d_model: usize,
        d_ff: usize,
    ) -> Vec<f64> {
        let (g, u) = projections(x, wg, wu, m, d_model, d_ff);
        down_transposed(&silu_gate(&g, &u), wd, m, d_model, d_ff)
    }

    /// The gate and up projections, `g = x@Wg^T` and `u = x@Wu^T`, both `[m, d_ff]`.
    ///
    /// ⛔ THE TWO CONTRACTIONS SHARE ONE PASS OVER `x`, which is not an optimisation but the reason
    /// a Granite-width control runs at all: at `d_model = 4096, d_ff = 12800` this loop is
    /// `m * d_ff * d_model` multiply-adds twice over, and doing it once per RUN instead of once per
    /// MUTANT is the difference between seconds and minutes.
    pub fn projections(
        x: &[f64],
        wg: &[f64],
        wu: &[f64],
        m: usize,
        d_model: usize,
        d_ff: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(x.len(), m * d_model, "mlp: x is not [m, d_model]");
        assert_eq!(wg.len(), d_ff * d_model, "mlp: wg is not [d_ff, d_model]");
        assert_eq!(wu.len(), d_ff * d_model, "mlp: wu is not [d_ff, d_model]");
        let mut g = vec![0.0f64; m * d_ff];
        let mut u = vec![0.0f64; m * d_ff];
        for i in 0..m {
            for nn in 0..d_ff {
                let mut gv = 0.0f64;
                let mut uv = 0.0f64;
                for p in 0..d_model {
                    let xv = x[i * d_model + p];
                    gv += xv * wg[nn * d_model + p];
                    uv += xv * wu[nn * d_model + p];
                }
                g[i * d_ff + nn] = gv;
                u[i * d_ff + nn] = uv;
            }
        }
        (g, u)
    }

    /// The FAITHFUL activation, `h = silu(g) * u` — `swiglu_oracle.py:151`'s `true_swiglu` form,
    /// which is also [`super::swiglu::truth`]. Stated here as its own function so an MLP-level
    /// control and the elementwise harness cannot disagree about the activation; asserted equal to
    /// that one by `silu_gate_is_the_swiglu_modules_truth`.
    pub fn silu_gate(g: &[f64], u: &[f64]) -> Vec<f64> {
        assert_eq!(g.len(), u.len(), "mlp: g and u must be the same length");
        g.iter().zip(u).map(|(&gv, &uv)| (gv * sigmoid(gv)) * uv).collect()
    }

    /// The down projection, `o[i,j] = sum_n h[i,n] * wd[j,n]` with `wd` as `[d_model, d_ff]`.
    pub fn down(h: &[f64], wd: &[f64], m: usize, d_model: usize, d_ff: usize) -> Vec<f64> {
        assert_eq!(h.len(), m * d_ff, "mlp: h is not [m, d_ff]");
        assert_eq!(wd.len(), d_model * d_ff, "mlp: wd is not [d_model, d_ff]");
        let mut o = vec![0.0f64; m * d_model];
        for i in 0..m {
            for j in 0..d_model {
                let mut acc = 0.0f64;
                for nn in 0..d_ff {
                    acc += h[i * d_ff + nn] * wd[j * d_ff + nn];
                }
                o[i * d_model + j] = acc;
            }
        }
        o
    }

    /// [`down`] with `wd` read in the WRONG orientation, `[d_ff, d_model]`. Same buffer, same
    /// element count, wrong stride.
    pub fn down_transposed(
        h: &[f64],
        wd: &[f64],
        m: usize,
        d_model: usize,
        d_ff: usize,
    ) -> Vec<f64> {
        assert_eq!(h.len(), m * d_ff, "mlp: h is not [m, d_ff]");
        assert_eq!(wd.len(), d_model * d_ff, "mlp: wd is not [d_model, d_ff]");
        let mut o = vec![0.0f64; m * d_model];
        for i in 0..m {
            for j in 0..d_model {
                let mut acc = 0.0f64;
                for nn in 0..d_ff {
                    // wd indexed [nn, j] instead of [j, nn]
                    acc += h[i * d_ff + nn] * wd[nn * d_model + j];
                }
                o[i * d_model + j] = acc;
            }
        }
        o
    }
}

// ===========================================================================================
// RoPE. AUTHORED — test/pod has no rope apparatus at all.
// ===========================================================================================

/// The rotary embedding and two authored mutants.
///
/// AUTHORED. There is no `rope_oracle2.py` / `rope_negctl.py` in `test/pod` — the pod's M7..M12
/// milestones never covered rope — so both mutants are written here in the pod's style: change
/// ONE thing that a plausible lowering bug would change, and nothing else.
///
/// SHAPES: `x`, `cos`, `sin` and the output are flat row-major `[rows, head_dim]` with
/// `head_dim = 2 * half`; the tables are already head-replicated and both halves equal, which
/// `rope.py:377-388` ASSERTS rather than assumes. Only the first `half` columns of `cos`/`sin`
/// are read, matching the kernel's half-width loads (`rope.py:99-100`).
pub mod rope {
    use super::Mutant;

    pub const MUTANTS: [Mutant; 2] = [
        Mutant::discriminating("swap_sin_cos"),
        Mutant::discriminating("rotate_wrong_half"),
    ];

    /// The faithful rotation: `o1 = x1*c - x2*s`, `o2 = x2*c + x1*s` — `rope.py:102-105`, which is
    /// HF's `rotate_half` written per half.
    pub fn truth(x: &[f64], cos: &[f64], sin: &[f64], rows: usize, head_dim: usize) -> Vec<f64> {
        map(x, cos, sin, rows, head_dim, |x1, x2, c, s| {
            (x1 * c - x2 * s, x2 * c + x1 * s)
        })
    }

    /// MUTANT: sin and cos exchanged. The rotation angle becomes `pi/2 - theta` — still a
    /// rotation, still the right magnitude, wrong for every position but position 0 (where
    /// `sin = 0`, `cos = 1`, so the first row is a legitimate blind spot and the control's
    /// discrimination comes from every other row).
    pub fn swap_sin_cos(
        x: &[f64],
        cos: &[f64],
        sin: &[f64],
        rows: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        map(x, cos, sin, rows, head_dim, |x1, x2, c, s| {
            (x1 * s - x2 * c, x2 * s + x1 * c)
        })
    }

    /// MUTANT: the minus sign lands on the WRONG half — `o1 = x1*c + x2*s`,
    /// `o2 = x2*c - x1*s`, i.e. `cat(x2, -x1)` where HF's `rotate_half` is `cat(-x2, x1)`. This is
    /// a rotation by `-theta`: the exact defect a swapped half-descriptor offset produces, and
    /// invisible to any shape or magnitude check.
    pub fn rotate_wrong_half(
        x: &[f64],
        cos: &[f64],
        sin: &[f64],
        rows: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        map(x, cos, sin, rows, head_dim, |x1, x2, c, s| {
            (x1 * c + x2 * s, x2 * c - x1 * s)
        })
    }

    fn map(
        x: &[f64],
        cos: &[f64],
        sin: &[f64],
        rows: usize,
        head_dim: usize,
        f: impl Fn(f64, f64, f64, f64) -> (f64, f64),
    ) -> Vec<f64> {
        assert_eq!(head_dim % 2, 0, "rope: head_dim must be even");
        let half = head_dim / 2;
        assert_eq!(x.len(), rows * head_dim, "rope: x is not [rows, head_dim]");
        assert_eq!(cos.len(), rows * head_dim, "rope: cos is not [rows, head_dim]");
        assert_eq!(sin.len(), rows * head_dim, "rope: sin is not [rows, head_dim]");
        let mut out = vec![0.0f64; x.len()];
        for r in 0..rows {
            let base = r * head_dim;
            for i in 0..half {
                let (o1, o2) = f(
                    x[base + i],
                    x[base + half + i],
                    cos[base + i],
                    sin[base + i],
                );
                out[base + i] = o1;
                out[base + half + i] = o2;
            }
        }
        out
    }
}

// ===========================================================================================
// Embedding. AUTHORED — test/pod has no embedding apparatus.
// ===========================================================================================

/// The scaled table gather and the off-by-one row mutant.
///
/// AUTHORED. SHAPES: `ids` holds `n_tok` row indices into a `[v, d_model]` flat row-major `table`;
/// the output is `[n_tok, d_model]`.
pub mod embedding {
    use super::Mutant;

    pub const MUTANTS: [Mutant; 2] = [
        Mutant::discriminating("off_by_one_row"),
        Mutant::discriminating("index_stick_wrap"),
    ];

    /// The faithful gather: `o[t, :] = table[ids[t], :] * emb_scale` — `embedding.py:113-117`,
    /// which is `GraniteModel.forward`'s first two lines (`embedding.py:162-168`).
    pub fn truth(
        ids: &[i64],
        table: &[f64],
        v: usize,
        d_model: usize,
        emb_scale: f64,
    ) -> Vec<f64> {
        gather(ids, table, v, d_model, emb_scale, 0)
    }

    /// MUTANT: every gathered row index is one too high (wrapping at `v`).
    ///
    /// The gather is the ONLY thing in this kernel that can be wrong — the arithmetic is a single
    /// multiply — and an index that is off by one is what a descriptor whose base or index dtype
    /// is mis-set produces. The output stays the right shape and the right distribution, so
    /// nothing but a per-element comparison against the true row can see it. It is a strong
    /// control precisely because the table rows are independent draws: the deviation is
    /// `emb_scale * (t[i+1] - t[i])`, i.e. `emb_scale * sqrt(2)` in RMS, four decades above the
    /// envelope's `k * |o|`.
    pub fn off_by_one_row(
        ids: &[i64],
        table: &[f64],
        v: usize,
        d_model: usize,
        emb_scale: f64,
    ) -> Vec<f64> {
        gather(ids, table, v, d_model, emb_scale, 1)
    }

    /// ⭐⭐⭐ MUTANT: the gather reads entry `t % stick` instead of entry `t` — THE DEFECT THE CARD
    /// PRODUCED, not an invented one.
    ///
    /// A gather's index reaches the L3LU IBR as ONE stick transfer (`SenUint32`, 32 entries / 128 B)
    /// and each core indexes it at its own work-slice word, so an op whose index is LONGER than one
    /// stick wraps. MEASURED on `embedding_granite` (256 entries, one per token, against a 32-entry
    /// stick): the card returned output row `r` holding the table row `ids[r mod 32]` names, for
    /// 256/256 rows — rows 0..31 exact (`within_2pct` 1.000000 over 131072 elements) and the whole
    /// `[256, 4096]` at 0.130597 ≈ 32/256, with the reference's own rms to 0.2% because a vocabulary's
    /// rows are independent draws. `dxp_standalone` exited 0 and the launch returned rc=0.
    ///
    /// ⛔ SO IT IS EXACTLY THE FAILURE MODE NOTHING ELSE IN THE PIPELINE CAN SEE, and the reason this
    /// mutant belongs beside `off_by_one_row`: the emission side now CUTS a node whose index exceeds
    /// one stick into one op per stick (`assemble_pointwise_broadcast_gather`), and a cut that dropped
    /// a leg's entry base would put every leg on leg 0's 32 entries — the SAME `ids[r mod 32]` shape
    /// this mutant is. So this is the numeric gate for the cut as well as for the ceiling. A `stick` of
    /// 0 or one that already covers every entry is refused rather than returning the truth — a mutant
    /// that mutates nothing passes every bound and reports a discriminator that is not one.
    pub fn index_stick_wrap(
        ids: &[i64],
        table: &[f64],
        v: usize,
        d_model: usize,
        emb_scale: f64,
        stick: usize,
    ) -> Vec<f64> {
        assert!(
            stick > 0 && stick < ids.len(),
            "index_stick_wrap: a {stick}-entry stick over {} entries mutates nothing",
            ids.len()
        );
        let wrapped: Vec<i64> = (0..ids.len()).map(|t| ids[t % stick]).collect();
        gather(&wrapped, table, v, d_model, emb_scale, 0)
    }

    fn gather(
        ids: &[i64],
        table: &[f64],
        v: usize,
        d_model: usize,
        emb_scale: f64,
        skew: i64,
    ) -> Vec<f64> {
        assert_eq!(table.len(), v * d_model, "embedding: table is not [v, d_model]");
        let mut out = vec![0.0f64; ids.len() * d_model];
        for (t, &id) in ids.iter().enumerate() {
            assert!(
                id >= 0 && (id as usize) < v,
                "embedding: id {id} out of range for v={v}"
            );
            let row = ((id + skew).rem_euclid(v as i64)) as usize;
            for j in 0..d_model {
                out[t * d_model + j] = table[row * d_model + j] * emb_scale;
            }
        }
        out
    }
}

// ===========================================================================================
// Decoder. AUTHORED — the mutant is a transposed projection weight.
// ===========================================================================================

/// The decoder block's mutant surface.
///
/// AUTHORED. The block is thirteen dots, two norms, a rope and a softmax; building a second full
/// f64 model of it here would duplicate the harness's own reference and would test that duplicate
/// rather than the emulator. So the decoder's control is the ONE defect the derivation says the
/// bound must catch, applied where it is cheapest to apply and hardest to see: TRANSPOSE A
/// PROJECTION'S B OPERAND.
///
/// Use [`matmul::transposed_b`] against [`matmul::truth`] for the square projections
/// (wq/wk/wv/wo, all `[d_model, d_model]` — `decoder_block.py:506-509`, where `k == n` so no
/// shape check can fire), and [`mlp::transposed_wd`] against [`mlp::truth`] for the MLP's down
/// projection. Both must EXCEED [`crate::bounds::decoder`]. [`transposed_projection_deviation`] states
/// the scale of that deviation so the assertion is a number, not a hope.
pub mod decoder {
    use super::matmul;

    /// The per-element RMS deviation a transposed square projection produces, on the documented
    /// stimulus: for `h` of unit RMS and a weight of row norm 1, `h@W` and `h@W^T` are two
    /// INDEPENDENT unit-RMS Gaussians, so their difference has RMS `sqrt(2)`.
    ///
    /// This is the number the assertion rests on: `sqrt(2)` against a decoder floor two decades
    /// smaller. It is stated rather than measured so a shrinking margin is visible in the report.
    pub const TRANSPOSED_PROJECTION_RMS: f64 = std::f64::consts::SQRT_2;

    /// The realized RMS deviation between the faithful and transposed-B contraction, for a report
    /// that prints a measured number beside [`TRANSPOSED_PROJECTION_RMS`].
    pub fn transposed_projection_deviation(
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> f64 {
        let t = matmul::truth(a, b, m, k, n);
        let x = matmul::transposed_b(a, b, m, k, n);
        let s: f64 = t
            .iter()
            .zip(&x)
            .map(|(&p, &q)| (p - q) * (p - q))
            .sum();
        (s / t.len() as f64).sqrt()
    }
}

// ===========================================================================================
// tests — synthetic data only, no emulator, no fixtures, no files.
// ===========================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds;

    /// A deterministic, dependency-free unit-variance-ish stream: a 64-bit LCG mapped through the
    /// Box-Muller transform. Not torch's generator (the harness owns matching that); good enough
    /// to exercise a mutant against a bound.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn unit(&mut self) -> f64 {
            // (0, 1)
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        /// Standard normal.
        fn normal(&mut self) -> f64 {
            let u1 = self.unit();
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
        fn normals(&mut self, n: usize, scale: f64) -> Vec<f64> {
            (0..n).map(|_| self.normal() * scale).collect()
        }
    }

    // ---- the bracketing control, both sides -------------------------------------------------

    #[test]
    fn bracket_is_a_real_bracket_over_every_envelope() {
        let mut rng = Rng::new(7);
        let reference: Vec<f64> = rng.normals(512, 1.0);
        for env in [
            bounds::rmsnorm(4096),
            bounds::rope(),
            bounds::embedding(12.0),
            bounds::swiglu_activation(),
            // swiglu_mlp_flat's blocking: BLOCK_N = D_FF = 256, BLOCK_K = D_MODEL = 128.
            bounds::swiglu(128, 256, 256, 128),
            // and the Granite blocking that fits LX, whose envelope carries the trip counts.
            bounds::swiglu(4096, 12800, 64, 2048),
            bounds::decoder(128, 256, 1),
        ] {
            let (over, i_over) = bracket(&reference, &env, 1.5);
            let (under, i_under) = bracket(&reference, &env, 0.5);
            assert_eq!(i_over, i_under, "bracket must pick the same element both ways");
            assert!(
                exceeds(&reference, &over, &env),
                "1.5x the per-element bound must EXCEED (env {env:?})"
            );
            assert!(
                !exceeds(&reference, &under, &env),
                "0.5x the per-element bound must stay WITHIN (env {env:?})"
            );
            // and exactly at the edge it is still within: the threshold is where the derivation
            // put it, not a hair above or below.
            let (edge, _) = bracket(&reference, &env, 1.0);
            assert!(!exceeds(&reference, &edge, &env), "1.0x must sit ON the edge");
            // one element only
            let diff = reference
                .iter()
                .zip(&over)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(diff, 1, "bracket must perturb exactly one element");
        }
    }

    #[test]
    fn bracket_picks_the_largest_allowance_and_respects_the_sign_swap() {
        // With k_plus > k_minus, the largest UPWARD allowance for equal magnitudes is at the
        // POSITIVE element, and bracket must pick it rather than the (larger-|.|) negative one.
        let env = Envelope::new(0.10, 0.01, 0.0);
        let reference = [-1.5, 1.0, -1.0];
        let (_, idx) = bracket(&reference, &env, 1.5);
        assert_eq!(idx, 1, "up_allowance(1.0)=0.10 > up_allowance(-1.5)=0.015");
        // Using abs_bound instead would have picked index 0 (0.15) and stepped 1.5*0.15 = 0.225
        // past a reference whose UPWARD allowance is only 0.015 -- exceeding for the wrong reason.
        assert!(env.abs_bound(reference[0]) > env.up_allowance(reference[1]));
    }

    // ---- the changed-word-fraction criterion -----------------------------------------------

    #[test]
    fn changed_row_fraction_is_the_minimum_over_rows() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // row 0 fully changed, row 1 changed in one of three -> the MIN is 1/3, not the mean 2/3.
        let b = [9.0, 9.0, 9.0, 4.0, 5.0, 9.0];
        assert!((changed_row_fraction(&a, &b, 3) - 1.0 / 3.0).abs() < 1e-15);
        assert!(!discrimination_ok(&a, &b, 3));
        // A one-word-per-row mutant must be REJECTED -- the r1 `md>0` rule this floor replaced.
        let one_word: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 3 == 0 { v + 1.0 } else { v })
            .collect();
        assert!(!discrimination_ok(&a, &one_word, 3));
        // A near-total change passes.
        let all: Vec<f64> = a.iter().map(|v| v + 1.0).collect();
        assert_eq!(changed_row_fraction(&a, &all, 3), 1.0);
        assert!(discrimination_ok(&a, &all, 3));
    }

    // ---- SwiGLU: truth inside, mutants outside ---------------------------------------------

    #[test]
    fn swiglu_mutants_exceed_while_truth_sits_inside() {
        let mut rng = Rng::new(11);
        let n = 4096usize;
        let row_len = 256usize;
        let g = rng.normals(n, 1.0);
        let u = rng.normals(n, 1.0);
        let env = bounds::swiglu_activation();
        let t = swiglu::truth(&g, &u);

        // truth against itself: zero deviation, comfortably inside.
        assert!(!exceeds(&t, &t, &env));
        assert_eq!(worst_ratio(&t, &t, &env), 0.0);

        for m in swiglu::MUTANTS.iter().filter(|m| !m.informational) {
            let got = match m.name {
                "no_sigmoid" => swiglu::no_sigmoid(&g, &u),
                "swap_gu" => swiglu::swap_gu(&g, &u),
                // exceeds, but see its doc: not independent of swap_gu on this path
                "sig_of_u" => swiglu::sig_of_u(&g, &u),
                other => panic!("unhandled discriminating mutant {other}"),
            };
            let (dev, bound) = max_abs_deviation(&t, &got, &env);
            assert!(
                exceeds(&t, &got, &env),
                "{}: max|err| = {dev:.6e} did NOT exceed its bound {bound:.6e}",
                m.name
            );
            assert!(
                discrimination_ok(&t, &got, row_len),
                "{}: min per-row changed fraction {:.4} < {MUTANT_MIN_ROW_FRACTION}",
                m.name,
                changed_row_fraction(&t, &got, row_len)
            );
        }

        // half_sigmoid is INFORMATIONAL: we measure it and do not assert a verdict.
        let half = swiglu::half_sigmoid(&g, &u);
        let (dev, bound) = max_abs_deviation(&t, &half, &env);
        assert!(dev.is_finite() && bound.is_finite());
        // it is a real perturbation, whatever the verdict
        assert!(dev > 0.0, "half_sigmoid changed nothing at all");
    }

    #[test]
    fn swap_gu_is_exact_and_is_the_operand_swap_it_claims_to_be() {
        // The control that earns the harness: it must be EXACTLY truth with g and u exchanged,
        // element for element -- not an approximation of one.
        let g = [-3.0, -0.5, 0.0, 0.5, 2.0, 7.0];
        let u = [1.0, -2.0, 4.0, -0.25, 0.5, -1.0];
        let swapped = swiglu::swap_gu(&g, &u);
        let truth_with_operands_exchanged = swiglu::truth(&u, &g);
        assert_eq!(swapped, truth_with_operands_exchanged);
        // and it genuinely differs from truth wherever silu(g)*u != silu(u)*g. `g = 0` is an
        // EXACT fixed point of the swap -- silu(0)*u = 0 = silu(u)*0 -- so one of these six
        // elements legitimately does not move, and the count says so rather than the test
        // pretending otherwise. That fixed point is why the changed-word-fraction floor
        // (MUTANT_MIN_ROW_FRACTION) is 0.90 and not 1.0.
        let t = swiglu::truth(&g, &u);
        let differing = t.iter().zip(&swapped).filter(|(a, b)| a != b).count();
        let fixed_points = g.iter().zip(&u).filter(|(a, b)| **a == 0.0 || **b == 0.0).count();
        assert_eq!(fixed_points, 1, "one g == 0 in this fixture");
        assert_eq!(differing, g.len() - fixed_points, "every non-fixed-point element should move");
        // THE WEAK CONTROL, ASSERTED AS WEAK RATHER THAN ASSUMED STRONG: on an exact-f64 path
        // sig_of_u IS swap_gu -- both are g*u*sigmoid(u), differing only in association. The pod's
        // two entries separate only because its model rounds the intermediate product.
        assert_eq!(
            swiglu::sig_of_u(&g, &u),
            swapped,
            "sig_of_u must be recognised as algebraically identical to swap_gu in exact arithmetic"
        );
        // ... and they separate again once the intermediate IS rounded, which is the form that
        // recovers the distinction if the harness ever models it.
        assert_ne!(
            swiglu::sig_of_u_f16_intermediate(&g, &u),
            swiglu::swap_gu_f16_intermediate(&g, &u)
        );
    }

    #[test]
    fn no_sigmoid_violates_the_silu_magnitude_gate() {
        // swiglu_oracle.py:35-38's hard gate: |h| <= |g*u| because 0 < sigmoid < 1. The
        // no_sigmoid mutant is |g*u| exactly, so it sits ON the gate everywhere and ABOVE truth.
        let g = [1.0, -2.0, 3.0];
        let u = [1.0, 1.0, 1.0];
        let t = swiglu::truth(&g, &u);
        let ns = swiglu::no_sigmoid(&g, &u);
        for (a, b) in t.iter().zip(&ns) {
            assert!(a.abs() <= b.abs() + 1e-15, "silu magnitude gate violated by truth");
        }
    }

    // ---- RMSNorm: the stick drop, and the honest weak-control report ------------------------

    #[test]
    fn rmsnorm_max_energy_stick_drop_exceeds() {
        let mut rng = Rng::new(13);
        let (m, d) = (8usize, 512usize);
        let x = rng.normals(m * d, 1.0);
        let w: Vec<f64> = vec![1.0; d];
        let env = bounds::rmsnorm(d);
        let t = rmsnorm::truth(&x, &w, m, d, 1e-5);
        assert!(!exceeds(&t, &t, &env));

        let dropped = rmsnorm::drop_max_energy_stick(&x, &w, m, d, 1e-5, 64);
        let (dev, bound) = max_abs_deviation(&t, &dropped, &env);
        assert!(
            exceeds(&t, &dropped, &env),
            "max-energy stick drop: max|err| = {dev:.6e} did NOT exceed bound {bound:.6e}"
        );
        // dropping a stick rescales the WHOLE row, so every element moves
        assert_eq!(changed_row_fraction(&t, &dropped, d), 1.0);
    }

    #[test]
    fn rmsnorm_reports_weak_sticks_instead_of_hiding_them() {
        // The honesty port. A row with one huge stick and seven tiny ones: dropping the huge one
        // is material, dropping a tiny one may be a sub-ULP no-op against the derived envelope.
        // The API must SAY how many sticks are material rather than silently keeping the good one.
        let (m, d, lanes) = (1usize, 512usize, 64usize);
        let mut x = vec![0.0f64; m * d];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i < lanes { 1.0 } else { 1e-4 };
        }
        let w = vec![1.0f64; d];
        let env = bounds::rmsnorm(d);
        let (material, total) = rmsnorm::stick_drop_materiality(&x, &w, m, d, 1e-5, lanes, &env);
        assert_eq!(total, d / lanes);
        assert!(material >= 1, "the max-energy stick must at least be material");
        assert!(
            material < total,
            "this stimulus was built so some sticks are sub-ULP no-ops; if all {total} are \
             material the control is stronger than expected and the test should say so"
        );
        // And the max-energy one is among the material ones.
        let t = rmsnorm::truth(&x, &w, m, d, 1e-5);
        let big = rmsnorm::drop_stick(&x, &w, m, d, 1e-5, lanes, 0);
        assert!(exceeds(&t, &big, &env));
    }

    // ---- matmul / MLP: the transposed operand ----------------------------------------------

    #[test]
    fn transposed_b_exceeds_and_is_not_a_shape_error() {
        let mut rng = Rng::new(17);
        let (m, k, n) = (16usize, 128usize, 128usize); // square: no shape check can fire
        let a = rng.normals(m * k, 1.0);
        let b = rng.normals(k * n, 1.0 / (k as f64).sqrt());
        let t = matmul::truth(&a, &b, m, k, n);
        let x = matmul::transposed_b(&a, &b, m, k, n);
        // Both are the right shape and the right magnitude -- that is the point.
        assert_eq!(t.len(), x.len());
        let rms = |v: &[f64]| (v.iter().map(|p| p * p).sum::<f64>() / v.len() as f64).sqrt();
        let ratio = rms(&x) / rms(&t);
        assert!(
            (0.5..2.0).contains(&ratio),
            "the mutant should look plausible: rms ratio {ratio}"
        );
        // and it exceeds a decoder-scale envelope by a wide margin
        let env = bounds::decoder(128, 256, 1);
        let (dev, bound) = max_abs_deviation(&t, &x, &env);
        assert!(
            exceeds(&t, &x, &env),
            "transposed_b: max|err| = {dev:.6e} did NOT exceed bound {bound:.6e}"
        );
        assert!(
            dev > 10.0 * bound,
            "transposed_b margin is only {:.2}x -- report it",
            dev / bound
        );
        // the stated RMS deviation for a fan-in-normalized square projection is sqrt(2)
        let realized = decoder::transposed_projection_deviation(&a, &b, m, k, n);
        assert!(
            (realized / decoder::TRANSPOSED_PROJECTION_RMS - 1.0).abs() < 0.25,
            "realized RMS deviation {realized:.4} vs stated {:.4}",
            decoder::TRANSPOSED_PROJECTION_RMS
        );
        // A transpose is not a no-op even for a square buffer.
        assert_ne!(t, x);
    }

    #[test]
    fn transposed_b_is_a_genuine_no_op_only_for_a_symmetric_b() {
        // Guards the mutant itself: if `transposed_b` accidentally re-implemented `truth`, this
        // would still pass for symmetric B but the previous test would fail. Both are needed.
        let (m, k, n) = (2usize, 3usize, 3usize);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // symmetric 3x3
        let b = [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];
        assert_eq!(
            matmul::truth(&a, &b, m, k, n),
            matmul::transposed_b(&a, &b, m, k, n)
        );
    }

    #[test]
    fn mlp_transposed_down_projection_exceeds() {
        let mut rng = Rng::new(19);
        let (m, d_model, d_ff) = (8usize, 128usize, 256usize);
        let scale = 1.0 / (d_model as f64).sqrt();
        let x = rng.normals(m * d_model, 1.0);
        let wg = rng.normals(d_ff * d_model, scale);
        let wu = rng.normals(d_ff * d_model, scale);
        let wd = rng.normals(d_model * d_ff, scale);
        // The blocking `swiglu_mlp_flat` runs at: both loops single-trip.
        let env = bounds::swiglu(d_model, d_ff, d_ff, d_model);
        let t = mlp::truth(&x, &wg, &wu, &wd, m, d_model, d_ff);
        let mutated = mlp::transposed_wd(&x, &wg, &wu, &wd, m, d_model, d_ff);
        assert!(!exceeds(&t, &t, &env));
        let (dev, bound) = max_abs_deviation(&t, &mutated, &env);
        assert!(
            exceeds(&t, &mutated, &env),
            "transposed_wd: max|err| = {dev:.6e} did NOT exceed bound {bound:.6e}"
        );
        assert!(
            discrimination_ok(&t, &mutated, d_model),
            "transposed_wd: min per-row changed fraction {:.4}",
            changed_row_fraction(&t, &mutated, d_model)
        );
    }

    /// ⛔ ONE STATEMENT OF THE ACTIVATION, NOT TWO. `mlp::silu_gate` exists so a Granite-width
    /// control can reuse [`swiglu`]'s mutants on the real `g`/`u`; that reuse is only sound while
    /// the FAITHFUL activation is the same function on both paths. Two copies of `silu(g)*u` that
    /// drifted would make every MLP-level mutant a comparison against the wrong truth.
    #[test]
    fn silu_gate_is_the_swiglu_modules_truth() {
        let mut rng = Rng::new(23);
        let g = rng.normals(512, 2.0);
        let u = rng.normals(512, 2.0);
        assert_eq!(mlp::silu_gate(&g, &u), swiglu::truth(&g, &u));
    }

    // ---- rope ------------------------------------------------------------------------------

    #[test]
    fn rope_mutants_exceed_while_truth_sits_inside() {
        let mut rng = Rng::new(23);
        let (rows, head_dim) = (32usize, 128usize);
        let half = head_dim / 2;
        let x = rng.normals(rows * head_dim, 1.0);
        // A real angle table: both halves equal, head-replicated per row (rope.py:387-388).
        let mut cos = vec![0.0f64; rows * head_dim];
        let mut sin = vec![0.0f64; rows * head_dim];
        for r in 0..rows {
            for i in 0..half {
                // skip position 0 for sin=0/cos=1 is a REAL blind spot; start the sweep at 1.
                let theta = (r as f64 + 1.0) / 10.0f64.powf(4.0 * i as f64 / half as f64);
                let (c, s) = (theta.cos(), theta.sin());
                cos[r * head_dim + i] = c;
                cos[r * head_dim + half + i] = c;
                sin[r * head_dim + i] = s;
                sin[r * head_dim + half + i] = s;
            }
        }
        let env = bounds::rope();
        let t = rope::truth(&x, &cos, &sin, rows, head_dim);
        assert!(!exceeds(&t, &t, &env));
        // the rotation is norm-preserving per (x1, x2) pair -- a property check on `truth` itself
        for r in 0..rows {
            for i in 0..half {
                let b = r * head_dim;
                let before = x[b + i] * x[b + i] + x[b + half + i] * x[b + half + i];
                let after = t[b + i] * t[b + i] + t[b + half + i] * t[b + half + i];
                assert!((before - after).abs() < 1e-9 * (1.0 + before));
            }
        }
        for name in ["swap_sin_cos", "rotate_wrong_half"] {
            let got = match name {
                "swap_sin_cos" => rope::swap_sin_cos(&x, &cos, &sin, rows, head_dim),
                _ => rope::rotate_wrong_half(&x, &cos, &sin, rows, head_dim),
            };
            let (dev, bound) = max_abs_deviation(&t, &got, &env);
            assert!(
                exceeds(&t, &got, &env),
                "{name}: max|err| = {dev:.6e} did NOT exceed bound {bound:.6e}"
            );
        }
    }

    // ---- embedding -------------------------------------------------------------------------

    #[test]
    fn embedding_off_by_one_row_exceeds() {
        let mut rng = Rng::new(29);
        let (n_tok, v, d_model) = (32usize, 97usize, 128usize);
        let table = rng.normals(v * d_model, 1.0);
        let ids: Vec<i64> = (0..n_tok).map(|i| ((i * 7 + 3) % v) as i64).collect();
        let env = bounds::embedding(12.0);
        let t = embedding::truth(&ids, &table, v, d_model, 12.0);
        assert!(!exceeds(&t, &t, &env));
        let got = embedding::off_by_one_row(&ids, &table, v, d_model, 12.0);
        let (dev, bound) = max_abs_deviation(&t, &got, &env);
        assert!(
            exceeds(&t, &got, &env),
            "off_by_one_row: max|err| = {dev:.6e} did NOT exceed bound {bound:.6e}"
        );
        assert!(
            discrimination_ok(&t, &got, d_model),
            "off_by_one_row: min per-row changed fraction {:.4}",
            changed_row_fraction(&t, &got, d_model)
        );
        // the scale really is applied
        assert!((t[0] / table[(ids[0] as usize) * d_model] - 12.0).abs() < 1e-12);
    }
}
