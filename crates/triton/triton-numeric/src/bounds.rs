//! DERIVED per-element tolerance envelopes for the numeric harness.
//!
//! This is a PORT OF THE METHOD in `third_party/spyre/test/pod/{rmsnorm,softmax,swiglu}_eps_derivation.py`
//! — state the forward-error algebra, compose it, emit a two-sided per-element envelope, outward-ceil
//! the coefficients to a decimal grid — and it is deliberately NOT a port of their NUMBERS. Those
//! numbers are dominated by `eps_sigmoid` (the `sigmoid.smc` opaque at 2 reciprocal refinements,
//! `swiglu_eps_derivation.py`'s 4.0e-3) and `eps_rsqrt` (`rsqrtx1` at 4 Newton iters,
//! `rmsnorm_eps_derivation.py`'s 8.0e-3). THE EXECUTOR HERE COMMITS NEITHER OF THOSE ERRORS, so
//! copying them would yield a bound ~20x too loose to discriminate a mutant.
//!
//! ===========================================================================================
//! WHAT THE ORACLE ACTUALLY ROUNDS — established from the emulator's source, not its docstrings
//! ===========================================================================================
//! Read at /Users/nickm/git/scratchy/.claude/worktrees/ktir-retarget/crates/targets/spyre/ktir
//! (READ-ONLY tree; paths below are relative to that root).
//!
//! Q1. IS A TILE ROUNDED TO f16 WHEN IT IS WRITTEN BACK TO HBM/LX?
//!     YES — and, far more importantly, IT IS ROUNDED AT EVERY OP, NOT ONLY AT THE STORE.
//!       * store: `ops_memory.rs:1004` `store_data` reads the tile through `Tile::as_f32` and
//!         encodes into the memref's dtype at `ops_memory.rs:1523` -> `codec::encode`
//!         (`ktir-core/src/codec.rs:112`), whose f16 arm is `f32_to_f16_bits` / `encode_f16`
//!         (`codec.rs:331`). A value already on the f16 grid encodes losslessly, so the store
//!         itself is only a rounding for a tile that is NOT f16-typed.
//!       * per-op: `ktir-core/src/tile.rs:151` `Tile::compute` calls `codec::round_to_dtype`
//!         (`codec.rs:88`) on EVERY result, and its own comment says so: "Because every op handler
//!         builds its result through this constructor, this gives per-op rounding for free: an f16
//!         compute *chain* rounds after each step the way NumPy does, rather than accumulating in
//!         f32 and rounding only at store" (`tile.rs:145-150`). Every elementwise handler goes
//!         through it (`dialects/arith.rs:889` for `addf/subf/mulf/divf` via `binary_float`,
//!         `dialects/math.rs:335` for the transcendentals via `unary`).
//!     => THE DOMINANT ERROR TERM IS THE PER-OP f16 ROUND, and the envelopes below are a COUNT OF
//!        THOSE ROUNDS along each kernel's data path. That count is only correct where the tile is
//!        f16-typed; every fixture in this suite forces exactly that (`rmsnorm.py` delta 5,
//!        `swiglu_mlp.py` delta 2 "f16 accumulators", `decoder_block.py` delta 3
//!        `out_dtype=tl.float16`). An f32-typed tile makes `round_to_dtype` a NO-OP
//!        (`codec.rs:89-93`), which can only make the true error SMALLER — the band still holds.
//!
//! Q2. DOES `linalg.matmul` ACCUMULATE IN f32 OR f64, AND DOES IT ROUND PER-K-STEP OR ONCE?
//!     f32, AND ONCE AT THE END — never per-K-step, never f64.
//!       * `dialects/linalg.rs:446` `matmul2d` calls `gemm` (`linalg.rs:45`) and wraps the result in
//!         `Tile::compute(data, a.dtype, ...)` at `linalg.rs:463` — ONE rounding of the finished
//!         product.
//!       * `gemm` dispatches to `blas::sgemm_rowmajor` (`blas.rs:28`), whose accumulator is f32:
//!         `let mut c = vec![0.0f32; m * n]` (`blas.rs:42`) and, in the reference path,
//!         `let mut acc = 0.0f32; ... acc += a[..] * b[..]` (`blas.rs:72-76`). `linalg.rs:47` says
//!         it plainly: "Same math, same f32-accumulate ... (the caller still rounds to f16 via
//!         `Tile::compute`)".
//!       * an `outs` operand adds a SECOND rounding: `accumulate_outs` (`linalg.rs:470`) forms
//!         `result + C` in f32 and rounds once via `Tile::compute` (`linalg.rs:494`).
//!     => a `tl.dot(a, b)` costs 1 f16 round; a `tl.dot(a, b, acc)` costs 2. The K-length
//!        contraction itself contributes only the f32 accumulation difference against an f32
//!        reference, which is 2^-24-scale and, as computed below, negligible.
//!
//! Q3. IS THE f16 ROUNDING ROUND-TO-NEAREST-EVEN, OR DL16 ROUND-HALF-UP?
//!     ROUND-TO-NEAREST-EVEN. `codec.rs:7-9` states it ("an inline IEEE half-precision round-trip
//!     (round-to-nearest-even) — no `half` dependency") and `codec.rs:44-81` implements it: the tie
//!     test is `rem > halfway || (rem == halfway && (half_mant & 1) == 1)` at `codec.rs:67` for the
//!     subnormal arm and `codec.rs:74` for the normal arm, with the comment "round-to-nearest-even"
//!     at `codec.rs:64`. The DEVICE is a different rule — `softmax_eps_derivation.py:52` cites
//!     "the DL16 round-half-up gap ... (dataElement.cpp:505, single guard bit)" — SO THE POD'S
//!     ROUNDING MODEL DOES NOT APPLY HERE.
//!     => the unit roundoff is HALF an ULP. f16 keeps 10 explicit mantissa bits (`codec.rs:72`
//!        `mant >> 13` off f32's 23), so ULP(1) = 2^-10 and [`U_F16`] = 2^-11. The pod files pin
//!        `u = 2^-10` (a FULL ULP, e.g. `rmsnorm_eps_derivation.py:117`); that is 2x conservative
//!        for RNE and we do NOT carry the factor, because the whole point of re-deriving is a band
//!        tight enough to discriminate.
//!
//! Q4. ARE `rsqrt` / `exp` / `sigmoid` APPROXIMATED, OR COMPUTED WITH libm?
//!     libm (Rust `std`), in f32, with NO Newton refinement anywhere.
//!       * `dialects/math.rs:110` — `rsqrt` is `unary(op, ctx, "math.rsqrt", |x| 1.0 / x.sqrt())`.
//!         Two correctly-rounded f32 operations. NOT `rsqrtx1.smc`, NOT 4 Newton iters, so
//!         `eps_rsqrt = 8.0e-3` (`rmsnorm_oracle.py:125`) IS NOT AN ERROR THIS ORACLE COMMITS.
//!       * `dialects/math.rs:93` — `exp` is `|x| x.exp()`.
//!       * there is NO sigmoid op in the emulator at all: a grep for `sigmoid` over
//!         `ktir-emulator/src/` finds only a comment naming an activation code
//!         (`metal.rs:3453` "Activation: 0 none, 1 relu, 2 tanh, 3 exp, 4 sigmoid"), and the
//!         dialect registration table (`math.rs:37-77`) has no entry. The fixtures spell the
//!         sigmoid out (`swiglu_mlp.py` delta 5: widen / `tl.exp` / truncate / `tl.fdiv`), so its
//!         cost here is one f32 `exp` plus the f16 rounds of the written-out chain — NOT
//!         `eps_sigmoid = 4.0e-3` (`swiglu_oracle.py:105`).
//!       * NOTE FOR THE HARNESS OWNER, not a bound: `decoder_block.py:283` calls `tl.math.exp2`,
//!         and the emulator's math table (`math.rs:37-77`) registers `MathExp`/`MathLog2` but NO
//!         `exp2`. Either the frontend folds it to `math.exp` (ln2 into the constexpr, which
//!         `QK_SCALE` already carries as `LOG2E`) or the op is unhandled. Both f32 paths are
//!         accurate to `U_F32`, so the envelope is the same either way.
//!
//! ===========================================================================================
//! THE TWO CURRENCIES, AND WHY THE CONTRACTION KERNELS NEED A FLOOR
//! ===========================================================================================
//! For an elementwise chain, every error term is proportional to the element's own value, so a
//! purely MULTIPLICATIVE two-sided band is exact and the additive floor only covers underflow and
//! constant-representation error. `rmsnorm`, `rope`, `embedding` are in that class.
//!
//! A CONTRACTION IS NOT. `c = sum_k a_k b_k` cancels: |c| can be arbitrarily smaller than
//! `sum_k |a_k b_k|`, so no multiple of |c| bounds the error. `fp_negctl.py:4-12` says this out
//! loud for the pod's own matmul gate ("the DL16-accumulation worst case ((1+2^-9)^D-1)*S is
//! LOOSE, so a subtly-wrong matmul ... can sit WITHIN it"), and its bound is written against
//! `|A|.|B|[i,j]`, not against `|C|`. We do the same: the cancellation allowance goes into the
//! ADDITIVE FLOOR.
//!
//! HOW THE FLOOR IS SIZED, STATED PLAINLY BECAUSE IT IS THE ONE NON-WORST-CASE STEP HERE:
//!   * The worst-case (Cauchy-Schwarz) floor is `gain = sqrt(K)` per contraction, i.e. 113x at
//!     `D_FF = 12800`. Composed over the MLP's two contractions that is ~59.0 absolute against a
//!     unit-RMS output — VACUOUS, it would not catch even a transposed operand. We do not ship it.
//!   * We ship a ROOT-SUM-SQUARE propagation instead: an error vector of RMS `e` entering a
//!     contraction against a weight row of L2 norm `w` leaves with RMS `w * e`, and the per-element
//!     ceiling is [`GAUSS_TAIL`] times that RMS. This is a PROBABILISTIC bound over the harness's
//!     documented stimulus (`test/numeric/gen_numeric_data.py:254-260` and
//!     `test/fixtures/decoder_block.py:500-514`: unit-variance activations, weights
//!     `randn * 1/sqrt(d_model)`), NOT a worst case. It is labelled as such at every use.
//!   * It is therefore VALID ONLY FOR THAT STIMULUS. Feed the harness adversarial data and these
//!     floors must be re-derived; feed it the checked-in generator and they hold with the tail
//!     margin stated at [`GAUSS_TAIL`].
//!
//! Every coefficient below is COMPUTED from [`U_F16`], [`U_F32`] and the extents. There are no
//! fitted literals: the only hand-chosen numbers are the four labelled ceilings
//! ([`GAUSS_TAIL`], [`SILU_LIP`], [`SILU_REL_GAIN`], and the fixture constexprs
//! [`DECODER_QK_SCALE`] / [`DECODER_RM`]), each carrying its provenance.

// ===========================================================================================
// unit roundoff and the machine constants
// ===========================================================================================

/// f16 round-to-nearest-even half-ULP.
///
/// 2^-11, NOT 2^-10. f16 carries 10 explicit mantissa bits (`codec.rs:72`, `mant >> 13` from
/// f32's 23), so ULP(1) = 2^-10; the emulator's rounding is round-to-nearest-EVEN
/// (`codec.rs:64-75`), whose worst case is half an ULP. The pod derivations pin `u = 2^-10`
/// (`rmsnorm_eps_derivation.py:117`, `swiglu_eps_derivation.py:171`) — a full ULP, which is a
/// valid but 2x-conservative bound. See Q3 in the module header.
pub const U_F16: f64 = 1.0 / 2048.0;

/// f32 round-to-nearest half-ULP, 2^-24. The emulator's contraction accumulator
/// (`blas.rs:42`, `blas.rs:72`) and its `exp`/`rsqrt` (`math.rs:93`, `math.rs:110`) work here.
pub const U_F32: f64 = 1.0 / 16_777_216.0;

/// The smallest positive f16 (subnormal), 2^-24 — `codec.rs:56-70`'s subnormal arm. Half of it is
/// the absolute rounding error floor for a value too small for a relative bound to mean anything.
pub const F16_MIN_SUBNORMAL: f64 = 1.0 / 16_777_216.0;

/// Smallest NORMAL f16, 2^-14. Below this the grid is the fixed subnormal grid, so relative
/// precision degrades and [`F16_MIN_SUBNORMAL`] carries the error instead.
pub const F16_MIN_NORMAL: f64 = 1.0 / 16_384.0;

/// Per-element tail ceiling on a unit-variance Gaussian stimulus: 6 sigma.
///
/// PROVENANCE AND EXCEEDANCE, so this is a stated risk and not a magic number. Every stimulus in
/// the suite is `torch.randn` (`gen_numeric_data.py:256-260`, `rope.py:333`, `embedding.py:156`,
/// `decoder_block.py:519`). The Mills-ratio tail bound gives P(|Z| > 6) = 1.97e-9; the largest
/// single tile in the suite is rope_q32's `256*32*128 = 1_048_576` elements, so the expected
/// number of 6-sigma elements there is 2.1e-3. At 4.8 sigma (the typical realized max of a
/// million draws) the floor would be 20% smaller; 6 buys margin for the composed stages whose
/// error distribution is only approximately Gaussian.
///
/// LABEL: STIMULUS TAIL CEILING. It is the one place the floors stop being worst-case.
pub const GAUSS_TAIL: f64 = 6.0;

/// Lipschitz ceiling of `silu(z) = z * sigmoid(z)`: `silu'(z) = s(1 + z(1-s))`, whose maximum over
/// the reals is 1.09984... at z = 2.39936 (a stationary point of `silu'`, i.e. `silu'' = 0`).
/// Outward-ceiled to the 1e-3 grid via [`ceil_to`]. ANALYTIC, not measured.
pub const SILU_LIP: f64 = 1.1;

/// Gain applied to a RELATIVE error crossing the silu, i.e. [`SILU_LIP`] divided by the RMS
/// contraction `RMS(silu(Z))/RMS(Z) = 0.57501...` for `Z ~ N(0,1)` — 1.9127, outward-ceiled to
/// the 1e-1 grid. The RMS ratio is a Gaussian moment of the stimulus, so this is the second
/// stimulus-dependent ceiling; the theorem-only alternative is `|silu(z)| <= |z|` (ratio <= 1),
/// which gives gain 1.1 and is TIGHTER but is a bound on the WRONG quantity (it bounds the value,
/// not the RMS the relative error is measured against). We take the conservative 2.0.
pub const SILU_REL_GAIN: f64 = 2.0;

/// `decoder_block.py`'s attention multiplier folded with `log2(e)`, i.e. the `QK_SCALE` constexpr:
/// `0.0078125 * 1.44269504` (`gen_numeric_data.py:166`). Needed because the score error enters the
/// softmax as an ABSOLUTE exponent perturbation, and this constant is what converts a score-scale
/// error into one. Config, not a fit.
pub const DECODER_QK_SCALE: f64 = 0.011271055; // 0.0078125 * 1.44269504, folded — Python's literal, bit-identical; NOT std's LOG2_E product spelling

/// `decoder_block.py`'s residual multiplier, the `RM` constexpr = 0.22 (`gen_numeric_data.py:167`).
/// The residual adds `o * RM` to `x`, so it is what weights the branch error against the skip.
pub const DECODER_RM: f64 = 0.22;

// ===========================================================================================
// the envelope
// ===========================================================================================

/// A two-sided multiplicative envelope plus an additive floor, applied SIGN-AWARELY.
///
/// The gate is `lo <= got <= hi` with the band from [`Envelope::band`]. The sign-awareness is the
/// M9 r1 correction ported verbatim from `rmsnorm_oracle.py:167-183` / `swiglu_oracle.py:160-189`:
/// the derivation bounds the RATIO `R = got/reference` in `[1 - k_minus, 1 + k_plus]`, and the
/// absolute deviation `(R-1)*reference` FLIPS SIGN with `reference`, so the coefficient bounding
/// the UPWARD deviation is `k_plus` for a positive reference and `k_minus` for a negative one.
/// Using `k_plus` upward at every sign (the pre-r1 bug) is simultaneously too loose for negative
/// references on the up side and too strict on the down side — "NOT a rigorous upper bound"
/// (`rmsnorm_oracle.py:180-183`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Envelope {
    /// Bounds the ratio's upward excursion: `R <= 1 + k_plus`.
    pub k_plus: f64,
    /// Bounds the ratio's downward excursion: `R >= 1 - k_minus`.
    pub k_minus: f64,
    /// Additive allowance, in the output's own units. Carries underflow, constant-representation
    /// error, and (for a contraction) the cancellation allowance that no multiple of |reference|
    /// can cover. See the module header.
    pub floor: f64,
}

impl Envelope {
    /// Construct from already-composed coefficients.
    pub const fn new(k_plus: f64, k_minus: f64, floor: f64) -> Self {
        Envelope {
            k_plus,
            k_minus,
            floor,
        }
    }

    /// The permitted band for one reference element. `got` passes iff `lo <= got <= hi`.
    ///
    /// SIGN-AWARE: `k_plus` and `k_minus` swap when `reference < 0`.
    pub fn band(&self, reference: f64) -> (f64, f64) {
        let a = reference.abs();
        // k_up bounds the UPWARD deviation, k_lo the DOWNWARD one; they swap by sign(reference).
        let (k_up, k_lo) = if reference < 0.0 {
            (self.k_minus, self.k_plus)
        } else {
            (self.k_plus, self.k_minus)
        };
        (
            reference - (k_lo * a + self.floor),
            reference + (k_up * a + self.floor),
        )
    }

    /// The upward allowance alone: `hi - reference`. This is what a bracketing control perturbs
    /// against (`swiglu_negctl.py:59-74` picks the element with the largest `up`).
    pub fn up_allowance(&self, reference: f64) -> f64 {
        let (_, hi) = self.band(reference);
        hi - reference
    }

    /// The downward allowance alone: `reference - lo`.
    pub fn down_allowance(&self, reference: f64) -> f64 {
        let (lo, _) = self.band(reference);
        reference - lo
    }

    /// The per-element absolute bound `|got - reference|` may not exceed — the LARGER of the two
    /// one-sided allowances, so a sign-blind absolute comparison is never tighter than the
    /// sign-aware band.
    pub fn abs_bound(&self, reference: f64) -> f64 {
        let a = reference.abs();
        let k = if self.k_plus > self.k_minus {
            self.k_plus
        } else {
            self.k_minus
        };
        k * a + self.floor
    }

    /// True iff `got` sits inside the sign-aware band for `reference`.
    pub fn contains(&self, reference: f64, got: f64) -> bool {
        let (lo, hi) = self.band(reference);
        got >= lo && got <= hi
    }

    /// The worst normalized excursion over a whole vector: `max_i deviation_i / allowance_i`,
    /// on the same side the deviation actually falls. `<= 1.0` iff every element is within band.
    /// Report this WITH the bound, never as "passes".
    pub fn worst_ratio(&self, reference: &[f64], got: &[f64]) -> f64 {
        let mut worst = 0.0f64;
        for (&r, &g) in reference.iter().zip(got.iter()) {
            let d = g - r;
            let allow = if d >= 0.0 {
                self.up_allowance(r)
            } else {
                self.down_allowance(r)
            };
            let ratio = if allow > 0.0 {
                d.abs() / allow
            } else if d == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            if ratio > worst {
                worst = ratio;
            }
        }
        worst
    }

    /// Outward-ceil the coefficients onto decimal grids — the reproducibility step
    /// `swiglu_eps_derivation.py:150-160` applies before pinning a bound.
    ///
    /// TWO grids, because the two quantities live decades apart: the ratio coefficients are
    /// 1e-3-ish and pin to `k_grid` (1e-6, the pod's grid), while a pure-underflow floor is
    /// 1e-8-ish and pinning IT to 1e-6 would inflate it by two decades. `floor_grid` keeps the
    /// pinning outward without coarsening the small floors into significance.
    pub fn pinned(self, k_grid: f64, floor_grid: f64) -> Self {
        Envelope {
            k_plus: ceil_to(self.k_plus, k_grid),
            k_minus: ceil_to(self.k_minus, k_grid),
            floor: ceil_to(self.floor, floor_grid),
        }
    }
}

/// Outward (toward +inf) rounding to a `10^-n` decimal grid, via the EXACT integer reciprocal.
///
/// Ported from `_ceil_to` in `swiglu_eps_derivation.py:127-138`. WHY DIVIDE-THEN-MULTIPLY IS
/// WRONG, in that file's own words: "A decimal grid like 1e-6 is NOT exactly representable, so the
/// divide-then-multiply form lands one binary64 ULP off the intended decimal value (e.g.
/// 0.005961999999999999 != 0.005962); scaling by the exact integer inv=1_000_000 and dividing
/// lands EXACTLY on the decimal-grid double, so the returned value compares == the manifest
/// literal." That reproducibility is the whole point: a pinned coefficient has to compare equal
/// to the number written in the record.
///
/// # Panics
/// If `grid` is not a negative power of ten (the same assertion the Python carries).
pub fn ceil_to(x: f64, grid: f64) -> f64 {
    let inv = (1.0 / grid).round();
    assert!(
        (inv * grid - 1.0).abs() < 1e-9,
        "ceil_to expects a 10^-n grid, got {grid:?}"
    );
    (x * inv).ceil() / inv
}

// ===========================================================================================
// the composable pieces of the algebra
// ===========================================================================================

/// `(1 + U_F16)^depth` — the upper edge of the band a `depth`-deep chain of f16 roundings spans.
///
/// `depth` is the FOLD DEPTH, i.e. the NUMBER OF ROUNDINGS on the path, never the contraction
/// length. Same convention as the pod's `D_sum` / `D_mean` (`rmsnorm_eps_derivation.py:38-44`:
/// "D_sum=73 rounds = 63 stick-fold MAC + 7 PE cross-8 + 3 within-8"), and here the depth comes
/// from the emulator's own reduce: `fast_tree_fold` (`linalg.rs:845`) halves the axis and calls
/// `round_to_dtype` once per level (`linalg.rs:875`).
pub fn band_up(depth: u32) -> f64 {
    (1.0 + U_F16).powi(depth as i32)
}

/// `(1 - U_F16)^depth` — the lower edge. Monotonically decreasing in `depth`.
pub fn band_down(depth: u32) -> f64 {
    (1.0 - U_F16).powi(depth as i32)
}

/// Roundings the emulator's `linalg.reduce` commits folding `n` elements along one axis.
///
/// `ceil(log2(n))`, because `fast_tree_fold` (`linalg.rs:845-881`) is a PAIRWISE HALVING tree —
/// `let half = n / 2; let new_n = n - half;` with `round_to_dtype(&mut next, dtype)` at the bottom
/// of each level (`linalg.rs:875`). n=4096 -> 12 levels, n=128 -> 7, n=64 -> 6.
///
/// WHY THIS IS THE CONSERVATIVE CHOICE AMONG THE LOWERINGS THE EMULATOR CAN TAKE: if the frontend
/// emits the reduce as a matmul against a ones-vector instead (the device's stick-fold MAC shape),
/// the emulator accumulates it in f32 and rounds ONCE or TWICE (Q2 above) — strictly fewer
/// roundings than the tree, so this band still covers it. The one lowering it does NOT cover is a
/// SEQUENTIAL `scf.for` accumulator, whose depth would be `n` rather than `log2(n)`; no fixture in
/// this suite reduces that way (each `tl.sum` is one whole-axis reduce over a resident tile), and
/// if one ever does, its envelope has to be re-derived with `depth = n`.
pub fn fold_depth(n: usize) -> u32 {
    debug_assert!(n >= 1, "fold_depth needs a non-empty axis");
    if n <= 1 {
        return 0;
    }
    // ceil(log2(n)) without floats.
    usize::BITS - (n - 1).leading_zeros()
}

/// The classic f32 forward-error coefficient for summing `n` terms in ANY order:
/// `gamma_n = n*u32 / (1 - n*u32)`. Used only to show the contraction's f32-accumulation
/// difference against an f32 reference is negligible; see [`f32_accum_rel`].
pub fn gamma_f32(n: usize) -> f64 {
    let nu = n as f64 * U_F32;
    debug_assert!(nu < 1.0, "gamma_f32 needs n*u32 < 1");
    nu / (1.0 - nu)
}

/// RMS-relative difference between the emulator's f32 contraction and the reference's f32
/// contraction over `k` terms, on the documented stimulus.
///
/// ALGEBRA. Both sides accumulate in f32 (emulator: `blas.rs:42/72`; reference: torch f32), so
/// only the ORDER differs. For random-sign terms of RMS `sigma_t`, the partial sums of a k-term
/// sum grow as `sqrt(j) * sigma_t`, and stochastic rounding gives an error
/// `~ u32 * sqrt(sum_j j) * sigma_t = u32 * sigma_t * k / sqrt(2)`. With the stimulus'
/// fan-in normalization `sigma_t = sigma_a * sigma_b = 1 * 1/sqrt(k)`, and the result's own scale
/// `sigma_c = sqrt(k) * sigma_t = 1`, the RELATIVE figure is `u32 * sqrt(k/2)`. Doubled for the
/// two independent orders.
///
/// AT THE REAL EXTENTS this is 5.4e-6 (k=4096) and 9.5e-6 (k=12800): two orders of magnitude
/// below one f16 round, which is why the f16 round count — not the contraction length — is what
/// the envelopes are built out of. The WORST-CASE alternative, `2 * gamma_f32(k) * sqrt(k)`
/// (Cauchy-Schwarz), is 3.1e-2 at k=4096 and 1.7e-1 at k=12800; carrying it would make the MLP's
/// floor a third of the output scale for no measurable gain in soundness against this stimulus.
pub fn f32_accum_rel(k: usize) -> f64 {
    2.0 * U_F32 * (k as f64 / 2.0).sqrt()
}

/// Round `x` to the nearest f16 value (round-to-nearest-EVEN), returned widened back to f64 —
/// a self-contained model of `codec.rs:44-81`'s `f32_to_f16_bits` composed with
/// `f16_bits_to_f32`. Used to DERIVE the representation error of a kernel's float constexprs
/// (`EPS`, `EMB_SCALE`) rather than writing that error as a literal.
pub fn round_f16(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    // f16 max finite is 65504; above the rounding boundary 65520 the encode gives inf
    // (`codec.rs:77-81`: "a carry that reaches exp 0x1f naturally yields inf").
    if a >= 65520.0 {
        return sign * f64::INFINITY;
    }
    // Quantum: 2^(exp-10) for a normal, the fixed 2^-24 grid for a subnormal (`codec.rs:56-70`).
    let quantum = if a < F16_MIN_NORMAL {
        F16_MIN_SUBNORMAL
    } else {
        let e = a.log2().floor() as i32;
        (2.0f64).powi(e - 10)
    };
    let n = a / quantum;
    let fl = n.floor();
    let frac = n - fl;
    let r = if frac > 0.5 {
        fl + 1.0
    } else if frac < 0.5 {
        fl
    } else if (fl as i64) % 2 == 0 {
        fl // tie -> even
    } else {
        fl + 1.0
    };
    sign * r * quantum
}

/// True iff `x` is exactly representable in f16 — i.e. materializing it as an f16 constexpr costs
/// no rounding. `12.0 = 1.5 * 2^3` is; `1e-5` is not.
pub fn f16_exact(x: f64) -> bool {
    round_f16(x) == x
}

/// Relative representation error of a float constexpr materialized at f16:
/// `|fl16(c) - c| / |c|`. Zero when [`f16_exact`].
pub fn const_rel_error(c: f64) -> f64 {
    if c == 0.0 {
        return 0.0;
    }
    (round_f16(c) - c).abs() / c.abs()
}

/// Compose a two-sided ratio band into an [`Envelope`].
///
/// `up` / `down` are the extreme values of `R = got/reference` (so `up >= 1 >= down`), and the
/// envelope's coefficients are `k_plus = up - 1`, `k_minus = 1 - down`, exactly the
/// `composed_two_sided` shape of `rmsnorm_eps_derivation.py:105-113` and
/// `swiglu_eps_derivation.py:141-160`. The result is pinned to the 1e-6 grid, as the pod's
/// manifest coefficients are.
fn compose(up: f64, down: f64, floor: f64) -> Envelope {
    Envelope::new(up - 1.0, 1.0 - down, floor).pinned(1e-6, 1e-9)
}

/// Absolute floor from `rounds` f16 roundings of a value that may be too small for a relative
/// bound: each rounding is at worst half the subnormal quantum. The `+1` is the REFERENCE's own
/// final f16 cast — every `reference()` in `test/fixtures` computes in f32 and casts once
/// (`rmsnorm.py:196`, `gen_numeric_data.py:279`), so the stored reference is itself on the f16
/// grid and the comparison inherits that rounding.
fn subnormal_floor(rounds: u32) -> f64 {
    (rounds as f64 + 1.0) * 0.5 * F16_MIN_SUBNORMAL
}

// ===========================================================================================
// the per-kernel derived envelopes
// ===========================================================================================

/// RMSNorm — `test/fixtures/rmsnorm.py:107-119`, config `rmsnorm_granite` (M=64, D_MODEL=4096).
///
/// THE EMITTED PATH AND ITS ROUNDS, op by op, against the f32 reference (`rmsnorm.py:186-200`):
/// ```text
///   sq   = x * x                        1 f16 round   (arith.mulf -> Tile::compute)
///   ssq  = tl.sum(sq, 1)                fold_depth(D) f16 rounds  (fast_tree_fold, linalg.rs:875)
///                                       -- SQUARES, so the terms are NONNEGATIVE: no cancellation,
///                                          the multiplicative band is exact here, unlike a matmul.
///   ms   = ssq * INV_D                  0 rounds when D is a power of two (INV_D = 2^-12 scales
///                                       the exponent only, exact barring underflow), else 1.
///   me   = ms + EPS                     1 f16 round, plus EPS's own representation error (below)
///   r32  = rsqrt(me.to(f32))            f32: sqrt then divide, each correctly rounded -> the
///                                       relative error is 2*U_F32 (the sqrt's error propagates
///                                       1:1 through the reciprocal, plus the divide's own).
///                                       NOT rsqrtx1.smc: no eps_rsqrt = 8e-3 here (Q4).
///   r    = r32.to(f16)                  1 f16 round
///   y    = x * r * w                    2 f16 rounds
///   store                               f16 tile into an f16 memref -> lossless
///   reference .to(f16)                  1 f16 round on the REFERENCE side
/// ```
/// MEAN BAND: `D_mean = 1 + fold_depth(D) + inv_d_rounds + 1`, and `mean_got/mean_true` lies in
/// `[(1-u)^D_mean, (1+u)^D_mean]`. RMSNorm is a PRODUCT, so that band propagates through the
/// rsqrt HALVED — `d(m^-1/2)/dm` has exponent -1/2, so the worst upper factor is the EXACT
/// `(1-u)^(-D_mean/2)` (not a linearization), exactly as `rmsnorm_eps_derivation.py:52-56`
/// composes it.
/// ```text
///   R_max = (1 + eps_rsqrt) * (1-u)^(-D_mean/2) * (1+u)^(R_post + 1)
///   R_min = (1 - eps_rsqrt) * (1+u)^(-D_mean/2) * (1-u)^(R_post + 1)
/// ```
/// with `R_post = 3` (r's cast, `x*r`, `*w`) and the trailing `+1` the reference's cast.
///
/// THE EPS CONSTANT is derived, not asserted: `EPS = 1e-5` is BELOW f16's smallest normal
/// (2^-14 = 6.1e-5), so it lands on the subnormal grid and [`const_rel_error`] computes its
/// relative error from [`round_f16`]. It perturbs `mean` additively, and the worst case is a row
/// whose sum of squares vanishes (`mean -> EPS`), where the whole relative error shows up in
/// `mean`; through the rsqrt it is halved. That term is carried in `k_plus`/`k_minus`, not in the
/// floor, because it scales with the output.
pub fn rmsnorm(d_model: usize) -> Envelope {
    const EPS: f64 = 1e-5; // rmsnorm.py:127 RMS_NORM_EPS / gen_numeric_data.py:147

    let u = U_F16;
    // INV_D is constexpr-folded on the host (rmsnorm.py delta 2). A power-of-two reciprocal is an
    // exponent shift: exact. Anything else materializes at f16 and rounds the product.
    let inv_d_rounds: u32 = if d_model.is_power_of_two() { 0 } else { 1 };
    let d_mean = 1 + fold_depth(d_model) + inv_d_rounds + 1;

    // f32 rsqrt: sqrt (<= U_F32) then reciprocal (<= U_F32).
    let eps_rsqrt = 2.0 * U_F32;
    // EPS mis-representation, halved by the rsqrt. Worst case mean == EPS.
    let eps_const = 0.5 * const_rel_error(EPS);

    let r_post = 3u32; // r's f16 cast, x*r, *w
    let r_ref = 1u32; // the reference's own single f16 cast

    let half = 0.5 * d_mean as f64;
    let up = (1.0 + eps_rsqrt + eps_const)
        * (1.0 - u).powf(-half)
        * band_up(r_post + r_ref);
    let down = (1.0 - eps_rsqrt - eps_const)
        * (1.0 + u).powf(-half)
        * band_down(r_post + r_ref);

    // Floor: pure underflow. Every term above is proportional to |y|, and the reduce has no
    // cancellation, so there is nothing else to cover.
    let floor = subnormal_floor(1 + fold_depth(d_model) + inv_d_rounds + 1 + r_post);
    compose(up, down, floor)
}

/// RoPE — `test/fixtures/rope.py:76-105`, configs `rope_q32` / `rope_kv8` (HEAD_DIM=128).
///
/// THE EMITTED PATH: `o1 = x1*c - x2*s`, `o2 = x2*c + x1*s`, three f16 rounds per output, no fold.
/// The reference is f32 with one cast (`rope.py:389-393`).
///
/// THIS IS THE ONE ELEMENTWISE KERNEL WITH CANCELLATION, and a relative-only band would be WRONG.
/// Writing the rounds out:
/// ```text
///   fl(x1 c) = x1 c (1+d_a),  fl(x2 s) = x2 s (1+d_b),  fl(A - B) = (A - B)(1+d_c)
///   o1_got - o1 = (x1 c d_a - x2 s d_b)(1+d_c) + o1 d_c
///   |o1_got - o1| <= u (1+u) (|x1 c| + |x2 s|)  +  u |o1|
/// ```
/// The first term does NOT scale with `|o1|`: rope is a rotation, so `o1` can pass through zero
/// while both products are O(1). It goes in the FLOOR, bounded by the stimulus:
/// ```text
///   |x1|,|x2| <= GAUSS_TAIL (unit-variance randn, rope.py:333)
///   c^2 + s^2 = 1  (asserted by rope.py:387-388, both halves equal, HF's cat((freqs,freqs)))
///   => |x1 c| + |x2 s| <= GAUSS_TAIL * (|c| + |s|) <= GAUSS_TAIL * sqrt(2)
/// ```
/// The multiplicative part is then just the two roundings that DO scale with the output — the
/// add's own, plus the reference's cast.
pub fn rope() -> Envelope {
    let u = U_F16;
    let term_ceiling = GAUSS_TAIL * 2f64.sqrt(); // |x1 c| + |x2 s|
    let floor = u * (1.0 + u) * term_ceiling;
    let up = band_up(2); // fl(A-B)'s own round + the reference's f16 cast
    let down = band_down(2);
    // NOT pre-ceiled: `compose` pins the floor once, on the 1e-9 grid. Ceiling twice on two grids
    // bumps the value by a grid step for nothing.
    compose(up, down, floor)
}

/// Embedding — `test/fixtures/embedding.py:98-118`, configs `embedding_granite` /
/// `embedding_granite_bm128` (D_MODEL=4096, EMB_SCALE=12.0).
///
/// THE EMITTED PATH IS ONE MULTIPLY: `o = rows * EMB_SCALE` (`embedding.py:117`). The gather is
/// data movement, not arithmetic, so it contributes no error at all — the only way to get it wrong
/// is to gather the WRONG ROW, which is a MUTANT, not a tolerance
/// (`mutants::embedding::off_by_one_row`).
/// ```text
///   emulator: 1 f16 round on the product
///   constexpr: EMB_SCALE materializes at the tile's dtype ("no widening island appears here",
///              embedding.py:116). If it is not f16-exact that costs one more relative round;
///              12.0 = 1.5 * 2^3 IS exact, so at the shipped config it costs nothing —
///              DERIVED via f16_exact(emb_scale), not assumed.
///   reference: F.embedding(...) * emb_scale on an f16 table keeps f16 -> 1 round, the SAME
///              operation. Counted anyway; the two rounds are the honest upper bound for a
///              comparison whose two sides are computed by different code.
/// ```
/// The floor is pure underflow: a table entry small enough that `t * EMB_SCALE` lands on the f16
/// subnormal grid has no relative precision left.
pub fn embedding(emb_scale: f64) -> Envelope {
    let const_rounds: u32 = if f16_exact(emb_scale) { 0 } else { 1 };
    let rounds = 1 + const_rounds + 1; // emulator mul, constexpr, reference mul
    compose(band_up(rounds), band_down(rounds), subnormal_floor(rounds))
}

/// The SwiGLU ACTIVATION alone — `h = silu(g) * u`, the elementwise stage the pod's apparatus
/// covers (`swiglu_oracle.py:151` `true_swiglu = (g*sigmoid(g))*u`). This is the envelope the
/// `mutants::swiglu` controls are gated against; the three-matmul MLP is [`swiglu`].
///
/// K-INVARIANT, exactly as `swiglu_eps_derivation.py:26-33` argues for its own bound: purely
/// elementwise, no reduce, so no `(1+-u)^D` band grows.
/// ```text
///   e   = exp((-g).to(f32)).to(f16)     f32 exp (U_F32, libm — Q4) + 1 f16 round
///   den = one + e                       1 f16 round; den >= 1, and d(s)/d(den) * den/s = -1,
///                                       so its relative error passes to s 1:1
///   s   = fdiv(g, den)                  1 f16 round
///   h   = s * u                         1 f16 round
///   reference .to(f16)                  1 f16 round
/// ```
/// The `exp`'s own f32 error enters as `|de/e| = U_F32` and reaches `s` attenuated by
/// `e/(1+e) < 1`, so it is bounded by `U_F32` and folded in as a relative term. No cancellation
/// anywhere (a product chain), so the floor is underflow only.
pub fn swiglu_activation() -> Envelope {
    let rounds = 4 + 1; // e-cast, one+e, fdiv, s*u, reference cast
    let up = (1.0 + U_F32) * band_up(rounds);
    let down = (1.0 - U_F32) * band_down(rounds);
    compose(up, down, subnormal_floor(rounds))
}

/// The three-matmul SwiGLU MLP — `test/fixtures/swiglu_mlp.py:120-215`, configs
/// `swiglu_mlp_flat` (128 / 256), `swiglu_mlp_granite_flat` (4096 / 12800) and
/// `swiglu_mlp_granite_tiled_k` (4096 / 12800 at BLOCK_N 64, BLOCK_K 2048).
///
/// `o = (silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T`.
///
/// ⛔⛔ THE BLOCKING IS A PARAMETER AND NOT AN OPTION, because it is not recoverable from
/// `(d_model, d_ff)` and it CHANGES THE ROUND COUNT. `swiglu_mlp.py`'s two loops are
/// `for n in tl.range(0, D_FF, BLOCK_N)` and, nested inside it, `for k in tl.range(0, D_MODEL,
/// BLOCK_K)`; the trip counts
/// ```text
///   T_k = ceil(d_model / block_k)     the gate/up contraction's f16 accumulation depth
///   T_n = ceil(d_ff / block_n)        the down projection's f16 accumulation depth
/// ```
/// are what the envelope is built out of. An earlier signature took `(d_model, d_ff)` alone and
/// its own doc said the blocked case "must be re-derived"; taking the knobs makes forgetting that
/// impossible, and [`crate::data::Fixture`] reads BLOCK_N/BLOCK_K out of `meta.json` so the
/// tolerance and the bytes cannot disagree about the blocking any more than about D_MODEL.
///
/// ROUNDS, from Q2: `tl.dot(a, b, acc)` = product round + accumulate round = 2;
/// `tl.dot(a, b)` = 1. `g` and `u` are seeded with `tl.zeros` accumulators
/// (`swiglu_mlp.py:174-175`), so they cost 2 PER K TRIP; `acc` likewise 2 PER N TRIP.
///
/// THE COMPOSITION IS AN ABSOLUTE-RMS PROPAGATION, in the output's own units, because of the
/// cancellation argument in the module header. `sigma_*` are the stages' RMS on the documented
/// stimulus (`gen_numeric_data.py:254-260`: `x = randn`, every weight `randn * 1/sqrt(d_model)`),
/// so a weight row of a `[*, K]` weight has L2 norm `sqrt(K/d_model)` and that is the RMS gain of
/// its contraction:
/// ```text
///   sigma_x  = 1                                (unit-variance activations)
///   sigma_g  = sigma_u = sqrt(d_model/d_model) * sigma_x = 1
///   sigma_s <= sigma_g = 1                      (|silu(z)| <= |z|, a theorem)
///   sigma_h <= sigma_s * sigma_u = 1
///   G_down   = sqrt(d_ff/d_model)               (Wd's fan-in is d_ff, its scale 1/sqrt(d_model))
///   sigma_o  = G_down * sigma_h
///
///   e_g = 2u*sigma_g + u*sigma_g*sqrt(T_k - 1) + f32_accum_rel(d_model)*sigma_g
///   e_s = SILU_REL_GAIN*e_g + 3u*sigma_s        (e-cast, one+e, fdiv)
///   e_h = sigma_u*e_s + sigma_s*e_g + u*sigma_h (product rule + the mul's own round)
///   e_o = G_down*e_h + f32_accum_rel(d_ff)*sigma_o + u*sigma_o*sqrt(T_n - 1)
///                                                     <- goes to the FLOOR (x GAUSS_TAIL)
///         + 2u*|o| (the LAST n-trip's two rounds) + u*|o| (reference cast)  <- k_plus/k_minus
/// ```
///
/// ⭐ THE TWO `sqrt(T - 1)` TERMS ARE THE WHOLE OF THE BLOCKING'S COST, AND WHY THEY HAVE THAT
/// SHAPE. A multi-trip accumulator's roundings do not all act on the same quantity. Trip `t`'s
/// two roundings act on the PARTIAL sum, whose RMS on this stimulus is `sqrt(t/T)` of the
/// finished value's — the partial contracts `t/T` of the terms, and fan-in normalization makes RMS
/// grow as the square root of the term count. Those roundings are independent of one another
/// (different weight slabs, different data), so they compose in ROOT-SUM-SQUARE, which is the
/// method the module header already declares and [`f32_accum_rel`] already uses:
/// ```text
///   sum over t = 1..T-1 of  2 * (u * sqrt(t/T))^2  =  u^2 * (T - 1)   ->   u * sqrt(T - 1)
/// ```
/// and the FINAL trip's two roundings stay in the multiplicative band, where the single-trip
/// derivation already put them, because that trip produces the finished value.
///
/// ⚖️ WHY THAT IS SOUND AND NOT A CONVENIENCE. The band charges the final trip's PRODUCT round as
/// `u*|o|`, and that product is a partial of RMS `sigma_o/sqrt(T_n)` rather than `|o|` — so for an
/// element where `|o|` is far below its RMS the band term under-covers it by at most
/// `u*sigma_o/sqrt(T_n)`, which at the Granite blocking is 6.1e-5 against a FLOOR of ~1.5e-1. The
/// floor covers that gap by three orders of magnitude, and it is stated here rather than waved at.
///
/// ⭐ AND IT REDUCES EXACTLY TO THE SINGLE-BLOCK DERIVATION AT `T_k = T_n = 1`, where both
/// `sqrt(T-1)` terms are ZERO — asserted by
/// `single_trip_blocking_reproduces_the_unblocked_swiglu_envelope`, so `swiglu_mlp_flat`'s shipped
/// coefficients cannot move under this generalization.
///
/// The two rounds of the down-projection's last trip and the reference's cast are the only terms
/// proportional to the element's OWN value, so they are the multiplicative band; everything
/// upstream reaches the output through a contraction and is therefore an ABSOLUTE allowance,
/// ceiled per element by [`GAUSS_TAIL`].
pub fn swiglu(d_model: usize, d_ff: usize, block_n: usize, block_k: usize) -> Envelope {
    assert!(block_n >= 1 && block_k >= 1, "swiglu: a blocking of zero has no trips");
    let u = U_F16;
    let g_down = (d_ff as f64 / d_model as f64).sqrt();
    // The trip counts the kernel's two `tl.range`s actually run.
    let t_k = d_model.div_ceil(block_k) as f64;
    let t_n = d_ff.div_ceil(block_n) as f64;

    let sigma_g = 1.0; // fan-in-normalized projection of a unit-variance activation
    let sigma_u = 1.0; // the up projection, same shape and scale
    let sigma_s = sigma_g; // |silu(z)| <= |z|
    let sigma_h = sigma_s * sigma_u;
    let sigma_o = g_down * sigma_h;

    // The K-trip chain: the last trip's two rounds linearly, the earlier trips' in RSS against
    // each partial's own RMS. Zero extra at T_k = 1.
    let e_g = 2.0 * u * sigma_g + u * sigma_g * (t_k - 1.0).sqrt()
        + f32_accum_rel(d_model) * sigma_g;
    let e_u = e_g; // the up projection is the same computation on the same operand
    let e_s = SILU_REL_GAIN * e_g + 3.0 * u * sigma_s;
    let e_h = sigma_u * e_s + sigma_s * e_u + u * sigma_h;
    // The N-trip chain, same shape, against the OUTPUT's RMS. Zero extra at T_n = 1.
    let e_o_absolute = g_down * e_h
        + f32_accum_rel(d_ff) * sigma_o
        + u * sigma_o * (t_n - 1.0).sqrt();

    let rounds = 2 + 1; // the last n-trip's product + accumulate, then the reference's cast
    compose(band_up(rounds), band_down(rounds), GAUSS_TAIL * e_o_absolute)
}

/// The Granite decoder block — `test/fixtures/decoder_block.py:237-318`, configs
/// `decoder_layer_one_flat` and `decoder_two_layers_flat` (M=64, D_MODEL=128, D_FF=256,
/// BLOCK_N=256, so the MLP loop is single-iteration).
///
/// This is the same absolute-RMS propagation as [`swiglu`], run stage by stage down the block and
/// then once per layer. `r_*` below are RELATIVE-to-stage-RMS errors (dimensionless) and every
/// `sigma` is 1 unless a weight-row norm says otherwise, by the stimulus' fan-in normalization
/// (`decoder_block.py:500-514`: `n1 = n2 = ones`, every 2-D weight `randn * 1/sqrt(d_model)`).
///
/// ```text
///  pre-norm      r_h  = 1.5*r_in + (D_mean/2 + 3)u
///                       D_mean = 1 + fold_depth(d_model) + 0 + 1   (INV_D = 2^-7 exact)
///                       the 1.5 is the direct x factor (1) plus the mean's half-power (0.5);
///                       an RMSNorm ATTENUATES an input error, it does not amplify it, so this
///                       is conservative.
///  q,k proj      r_q  = r_h + u                    4 half-dots, 1 round each, one erroneous operand
///  rope          r_qr = r_q + 3u                   2 muls + 1 add; c^2+s^2 = 1 so the RSS gain is 1
///  scores        r_qk = 2*r_qr + 3u                BOTH operands erroneous; dot + dot-with-outs
///  scale+mask    r_qk += 2u                        mul by QK_SCALE, add of the exact mask tile
///  softmax       the score error reaches p as an ABSOLUTE exponent perturbation:
///                       sigma_arg = sqrt(d_model) * DECODER_QK_SCALE     (RMS of the scaled score)
///                       r_p  = ln2 * sigma_arg * r_qk + 2u
///                       -- THE SOFTMAX ATTENUATES HARD: sigma_arg is 0.128 at d_model=128, so a
///                          1.4e-2 score error becomes a 1.2e-3 probability error.
///                r_l  = r_p + fold_depth(m)*u      the row sum; NONNEGATIVE terms, no cancellation
///                r_v  = r_h + u
///                r_a  = r_p + r_v + u + r_l + u    p@v then the fdiv by l
///  out proj      r_o  = r_a + u
///  residual 1    x' = x + o*RM: absolute errors add over the residual's own scale
///                       sigma_x' = sqrt(sigma_x^2 + (RM*sigma_o)^2)
///                       r_x' = (sigma_x*r_in + RM*sigma_o*r_o)/sigma_x' + 2u
///  post-norm     r_h2 = 1.5*r_x' + (D_mean/2 + 3)u
///  MLP           r_g  = r_h2 + u ; r_s = SILU_REL_GAIN*r_g + 3u ; r_hm = r_s + r_g + u
///                r_acc = r_hm + 2u                 the down-dot carries an outs accumulator
///  residual 2    r_out = (sigma_x'*r_x' + RM*sigma_acc*r_acc)/sigma_out + 2u
///                       sigma_acc = sqrt(d_ff/d_model) ; sigma_out = sqrt(sigma_x'^2 + (RM*sigma_acc)^2)
/// ```
/// Layer 2 re-enters with `r_in = r_out` and `sigma_x = sigma_out` of layer 1 — which is why the
/// two-layer floor GROWS 4.2x rather than 2x: the incoming error is amplified by 1.5 at each of the
/// two norms and by `SILU_REL_GAIN` at the MLP before it is re-weighted by `RM`.
///
/// MEASURED CONSEQUENCE, STATED BECAUSE IT IS THE LOOSEST ENVELOPE IN THE SUITE: at
/// (128, 256, 1) the floor is 8.33e-2 against an output RMS of 1.07 (7.8% of scale); at
/// (128, 256, 2) it is 3.52e-1, i.e. 33% of scale. A transposed projection still exceeds it by
/// ~58x at one layer and ~14x at two, so the controls bite — but a SUBTLE two-layer defect
/// (one dropped fold group, a permuted K) may not, and this envelope must not be cited as
/// evidence against one.
///
/// The FINAL 2 rounds of the residual add plus the reference's f16 cast are proportional to the
/// output element, so they are the multiplicative band; everything else is the absolute floor,
/// ceiled per element by [`GAUSS_TAIL`]. `f32_accum_rel` is added once per contraction at its own
/// K; at these extents (128 / 256) it is ~1e-6 and does not move the answer.
pub fn decoder(d_model: usize, d_ff: usize, layers: usize) -> Envelope {
    let u = U_F16;
    let m = 64usize; // gen_numeric_data.py:165 — the block's row count, the softmax fold length
    let ln2 = std::f64::consts::LN_2;

    // RMSNorm's mean-path depth: square + tree fold + INV_D (exact for a power of two) + eps add.
    let inv_d_rounds: u32 = if d_model.is_power_of_two() { 0 } else { 1 };
    let d_mean = 1 + fold_depth(d_model) + inv_d_rounds + 1;
    let norm_rounds = 0.5 * d_mean as f64 + 3.0; // rsqrt-halved band + r cast + x*r + *n

    let g_ff = (d_ff as f64 / d_model as f64).sqrt(); // Wd's row norm
    let sigma_arg = (d_model as f64).sqrt() * DECODER_QK_SCALE;

    let mut r_in = 0.0f64; // x is the SAME f16 tensor both sides read: no input error
    let mut sigma_x = 1.0f64;

    for _ in 0..layers {
        let r_h = 1.5 * r_in + norm_rounds * u;
        let r_q = r_h + u;
        let r_qr = r_q + 3.0 * u;
        let r_qk = 2.0 * r_qr + 3.0 * u + 2.0 * u + f32_accum_rel(d_model);
        let r_p = ln2 * sigma_arg * r_qk + 2.0 * u;
        let r_l = r_p + fold_depth(m) as f64 * u;
        let r_v = r_h + u;
        let r_a = r_p + r_v + u + r_l + u + f32_accum_rel(m);
        let r_o = r_a + u + f32_accum_rel(d_model);

        // residual 1: the branch's absolute error is RM * sigma_o * r_o, over sigma_x'.
        let sigma_o = 1.0; // wo's fan-in is d_model, its scale 1/sqrt(d_model) -> row norm 1
        let sigma_x1 = (sigma_x * sigma_x + (DECODER_RM * sigma_o).powi(2)).sqrt();
        let r_x1 = (sigma_x * r_in + DECODER_RM * sigma_o * r_o) / sigma_x1 + 2.0 * u;

        let r_h2 = 1.5 * r_x1 + norm_rounds * u;
        let r_g = r_h2 + u + f32_accum_rel(d_model);
        let r_s = SILU_REL_GAIN * r_g + 3.0 * u;
        let r_hm = r_s + r_g + u;
        let r_acc = r_hm + 2.0 * u + f32_accum_rel(d_ff);

        let sigma_acc = g_ff;
        let sigma_out = (sigma_x1 * sigma_x1 + (DECODER_RM * sigma_acc).powi(2)).sqrt();
        // The residual add's own 2 rounds are NOT included here: they are the multiplicative band.
        r_in = (sigma_x1 * r_x1 + DECODER_RM * sigma_acc * r_acc) / sigma_out;
        sigma_x = sigma_out;
    }

    let rounds = 2 + 1; // the final residual's mul + add, then the reference's f16 cast
    compose(band_up(rounds), band_down(rounds), GAUSS_TAIL * r_in * sigma_x)
}

// ===========================================================================================
// tests — the COMPOSITION ARITHMETIC only. No emulator, no data, no fixtures.
// ===========================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u_f16_is_the_rne_half_ulp() {
        // ULP(1) in f16 is 2^-10 (10 explicit mantissa bits, codec.rs:72), so the
        // round-to-nearest-even worst case is half of that.
        let ulp_at_one = round_f16(1.0 + 2f64.powi(-10)) - 1.0;
        assert_eq!(ulp_at_one, 2f64.powi(-10));
        assert_eq!(U_F16, ulp_at_one / 2.0);
        // And the pod's convention is exactly 2x ours.
        assert_eq!(2.0 * U_F16, 2f64.powi(-10));
    }

    #[test]
    fn ceil_to_lands_on_the_decimal_grid() {
        // The exact case from swiglu_eps_derivation.py:130-136.
        assert_eq!(ceil_to(0.005961999999999999, 1e-6), 0.005962);
        // ... and the divide-then-multiply form it warns against does NOT.
        let wrong = (0.005961999999999999f64 / 1e-6).ceil() * 1e-6;
        assert_ne!(wrong, 0.005962);
        // Outward means toward +inf for both signs.
        assert_eq!(ceil_to(0.0000001, 1e-6), 0.000001);
        assert_eq!(ceil_to(-0.0000001, 1e-6), 0.0);
        // Already on the grid: unchanged (no spurious bump).
        assert_eq!(ceil_to(0.005962, 1e-6), 0.005962);
    }

    #[test]
    #[should_panic(expected = "10^-n grid")]
    fn ceil_to_rejects_a_non_decimal_grid() {
        ceil_to(1.0, 0.003);
    }

    #[test]
    fn band_swaps_with_the_sign_of_the_reference() {
        let e = Envelope::new(0.05, 0.01, 0.001);
        let (lo, hi) = e.band(4.0);
        assert!((hi - (4.0 + 0.05 * 4.0 + 0.001)).abs() < 1e-15);
        assert!((lo - (4.0 - 0.01 * 4.0 - 0.001)).abs() < 1e-15);
        // For a NEGATIVE reference the coefficients swap: the upward allowance is k_minus.
        let (lo_n, hi_n) = e.band(-4.0);
        assert!((hi_n - (-4.0 + 0.01 * 4.0 + 0.001)).abs() < 1e-15);
        assert!((lo_n - (-4.0 - 0.05 * 4.0 - 0.001)).abs() < 1e-15);
        // The sign-BLIND value (rmsnorm_oracle.py:180-183's pre-r1 bug) is strictly larger, which
        // is exactly why using it upward for y<0 is not a rigorous bound.
        assert!(e.k_plus * 4.0 + 0.001 > e.up_allowance(-4.0));
        // abs_bound is the larger side, so it never under-reports.
        assert_eq!(e.abs_bound(-4.0), e.k_plus * 4.0 + 0.001);
        assert!(e.abs_bound(-4.0) >= e.up_allowance(-4.0));
        assert!(e.abs_bound(-4.0) >= e.down_allowance(-4.0));
    }

    #[test]
    fn band_is_symmetric_at_zero_and_equals_the_floor() {
        let e = Envelope::new(0.05, 0.01, 0.001);
        let (lo, hi) = e.band(0.0);
        assert_eq!(hi, 0.001);
        assert_eq!(lo, -0.001);
        assert_eq!(e.abs_bound(0.0), 0.001);
    }

    #[test]
    fn contains_and_worst_ratio_agree_on_the_edge() {
        let e = Envelope::new(0.05, 0.01, 0.0);
        let r = [2.0, -2.0];
        let on_edge = [2.0 * 1.05, -2.0 - 0.05 * 2.0];
        assert!(e.contains(r[0], on_edge[0]));
        assert!(e.contains(r[1], on_edge[1]));
        assert!((e.worst_ratio(&r, &on_edge) - 1.0).abs() < 1e-12);
        let over = [2.0 * 1.05 * 1.5, -2.0];
        assert!(!e.contains(r[0], over[0]));
        assert!(e.worst_ratio(&r, &over) > 1.0);
    }

    #[test]
    fn fold_depth_is_ceil_log2() {
        assert_eq!(fold_depth(1), 0);
        assert_eq!(fold_depth(2), 1);
        assert_eq!(fold_depth(3), 2);
        assert_eq!(fold_depth(64), 6);
        assert_eq!(fold_depth(128), 7);
        assert_eq!(fold_depth(4096), 12);
        assert_eq!(fold_depth(12800), 14); // 2^13 = 8192 < 12800 <= 16384
    }

    #[test]
    fn fold_band_is_monotone_in_depth() {
        let mut prev_up = 1.0;
        let mut prev_down = 1.0;
        for d in 1..40u32 {
            let up = band_up(d);
            let down = band_down(d);
            assert!(up > prev_up, "band_up must grow with depth at d={d}");
            assert!(down < prev_down, "band_down must shrink with depth at d={d}");
            assert!(up > 1.0 && down < 1.0);
            prev_up = up;
            prev_down = down;
        }
        // and the deeper fold is a strictly wider envelope for the same kernel shape
        let shallow = rmsnorm(64);
        let deep = rmsnorm(4096);
        assert!(deep.k_plus > shallow.k_plus);
        assert!(deep.k_minus > shallow.k_minus);
    }

    #[test]
    fn round_f16_is_round_to_nearest_even() {
        // exact values round-trip
        for v in [0.0, 1.0, -1.0, 0.5, 12.0, 2048.0, 65504.0] {
            assert_eq!(round_f16(v), v, "{v} must be f16-exact");
        }
        // a tie at 1 + 1.5*2^-10 goes to the EVEN mantissa (1 + 2*2^-10), not away from zero
        let tie = 1.0 + 1.5 * 2f64.powi(-10);
        assert_eq!(round_f16(tie), 1.0 + 2.0 * 2f64.powi(-10));
        // and the tie one step down goes DOWN, to the even neighbour
        let tie_lo = 1.0 + 0.5 * 2f64.powi(-10);
        assert_eq!(round_f16(tie_lo), 1.0);
        // subnormals land on the 2^-24 grid
        assert_eq!(round_f16(1.5 * F16_MIN_SUBNORMAL), 2.0 * F16_MIN_SUBNORMAL);
        // overflow
        assert!(round_f16(70000.0).is_infinite());
        // a rounding carry crosses the binade cleanly (codec.rs:77-81)
        assert_eq!(round_f16(2047.9), 2048.0);
    }

    #[test]
    fn constexpr_representation_errors_are_derived_not_asserted() {
        // 12.0 = 1.5 * 2^3 is f16-exact, so the embedding scale costs no extra round.
        assert!(f16_exact(12.0));
        assert_eq!(const_rel_error(12.0), 0.0);
        // 1e-5 is BELOW f16's smallest normal, so it lands on the subnormal grid and is not exact.
        assert!(1e-5 < F16_MIN_NORMAL);
        assert!(!f16_exact(1e-5));
        let e = const_rel_error(1e-5);
        assert!(e > 1e-4 && e < 2e-3, "eps rel error {e} out of expected decade");
        // A non-representable scale must widen the embedding envelope.
        let exact = embedding(12.0);
        let inexact = embedding(1e-5);
        assert!(inexact.k_plus > exact.k_plus);
    }

    #[test]
    fn every_envelope_is_two_sided_positive_and_pinned() {
        let all = [
            ("rmsnorm4096", rmsnorm(4096)),
            ("rmsnorm128", rmsnorm(128)),
            ("rope", rope()),
            ("embedding", embedding(12.0)),
            ("swiglu_act", swiglu_activation()),
            ("swiglu_small", swiglu(128, 256, 256, 128)),
            ("swiglu_granite_flat", swiglu(4096, 12800, 12800, 4096)),
            ("swiglu_granite_tiled_k", swiglu(4096, 12800, 64, 2048)),
            ("decoder1", decoder(128, 256, 1)),
            ("decoder2", decoder(128, 256, 2)),
        ];
        for (name, e) in all {
            assert!(e.k_plus > 0.0, "{name}: k_plus must be positive");
            assert!(e.k_minus > 0.0, "{name}: k_minus must be positive");
            assert!(e.floor >= 0.0, "{name}: floor must be nonnegative");
            // k_plus >= k_minus always: (1+u)^n - 1 > 1 - (1-u)^n for n >= 1. This is the
            // asymmetry the sign-aware band exists to handle.
            assert!(
                e.k_plus >= e.k_minus,
                "{name}: k_plus {} < k_minus {}",
                e.k_plus,
                e.k_minus
            );
            // pinned: the ratio coefficients on the 1e-6 grid, the floor on the finer 1e-9 grid
            // (see Envelope::pinned for why they differ).
            for (c, inv, grid) in [
                (e.k_plus, 1e6, "1e-6"),
                (e.k_minus, 1e6, "1e-6"),
                (e.floor, 1e9, "1e-9"),
            ] {
                let scaled = c * inv;
                assert!(
                    (scaled - scaled.round()).abs() < 1e-6,
                    "{name}: {c} is not on the {grid} grid"
                );
            }
        }
    }

    #[test]
    fn envelopes_widen_with_the_work_they_bound() {
        // more layers -> strictly more accumulated error
        assert!(decoder(128, 256, 2).floor > decoder(128, 256, 1).floor);
        // a wider MLP -> a larger down-projection gain -> a larger floor
        assert!(swiglu(4096, 12800, 12800, 4096).floor > swiglu(128, 256, 256, 128).floor);
        // and BLOCKING the same width strictly widens it: more trips, more f16 roundings on both
        // accumulators. This is the blocking's whole numeric cost, and it must not be free.
        assert!(
            swiglu(4096, 12800, 64, 2048).floor > swiglu(4096, 12800, 12800, 4096).floor,
            "a 200-trip down projection and a 2-trip contraction must cost MORE than single-trip"
        );
        // the elementwise activation is TIGHTER than the three-matmul MLP (no contraction)
        assert!(swiglu_activation().floor < swiglu(128, 256, 256, 128).floor);
        // rope's floor is the cancellation term, so it dwarfs a pure-underflow floor
        assert!(rope().floor > embedding(12.0).floor);
    }

    #[test]
    fn f32_accumulation_is_negligible_against_one_f16_round() {
        // The claim the derivation rests on: both sides accumulate in f32, so the contraction
        // length does not drive the bound.
        for k in [64usize, 128, 4096, 12800] {
            assert!(
                f32_accum_rel(k) < U_F16 / 10.0,
                "f32_accum_rel({k}) = {} is not negligible vs u = {U_F16}",
                f32_accum_rel(k)
            );
        }
        // and the worst-case Cauchy-Schwarz alternative is NOT negligible — the reason it is
        // documented as rejected rather than silently omitted.
        let cs_worst = 2.0 * gamma_f32(12800) * (12800f64).sqrt();
        assert!(cs_worst > 0.1, "cs worst-case {cs_worst} unexpectedly small");
    }

    #[test]
    fn rmsnorm_band_tracks_the_mean_path_algebra() {
        // Recompute k_plus independently of rmsnorm()'s body, from the stated algebra, and require
        // the shipped coefficient to be the outward-ceiled version of it. This is the guard against
        // a coefficient drifting away from the comment that justifies it.
        let d = 4096usize;
        let d_mean = (1 + fold_depth(d)) + 1;
        assert_eq!(d_mean, 14);
        let eps_rsqrt = 2.0 * U_F32;
        let eps_const = 0.5 * const_rel_error(1e-5);
        let up = (1.0 + eps_rsqrt + eps_const)
            * (1.0 - U_F16).powf(-0.5 * d_mean as f64)
            * (1.0 + U_F16).powi(4);
        let want = ceil_to(up - 1.0, 1e-6);
        assert_eq!(rmsnorm(d).k_plus, want);
    }

    /// ⛔ THE GENERALIZATION MUST NOT MOVE THE SHIPPED NUMBER. `swiglu` grew two arguments so a
    /// blocked configuration's extra f16 roundings are charged for; at a SINGLE-TRIP blocking both
    /// new terms are `sqrt(1 - 1) = 0`, so the envelope has to be bit-identical to what the
    /// two-argument form produced. `swiglu_mlp_flat` is gated by that value and its measured
    /// `max|err|` is quoted against it, so a silent shift here would re-date an existing result.
    #[test]
    fn single_trip_blocking_reproduces_the_unblocked_swiglu_envelope() {
        for (d_model, d_ff) in [(128usize, 256usize), (4096, 12800)] {
            // Recomputed here from the algebra WITHOUT the trip terms, i.e. the pre-generalization
            // body, so this is an independent statement and not a call to the same code.
            let u = U_F16;
            let g_down = (d_ff as f64 / d_model as f64).sqrt();
            let e_g = 2.0 * u + f32_accum_rel(d_model);
            let e_s = SILU_REL_GAIN * e_g + 3.0 * u;
            let e_h = e_s + e_g + u;
            let e_o = g_down * e_h + f32_accum_rel(d_ff) * g_down;
            let want = Envelope::new(band_up(3) - 1.0, 1.0 - band_down(3), GAUSS_TAIL * e_o)
                .pinned(1e-6, 1e-9);
            // BLOCK_N = D_FF and BLOCK_K = D_MODEL is what "flat" means: both loops single-trip.
            assert_eq!(
                swiglu(d_model, d_ff, d_ff, d_model),
                want,
                "the single-trip blocking at ({d_model}, {d_ff}) must reproduce the unblocked \
                 envelope exactly"
            );
        }
        // And the shipped `swiglu_mlp_flat` coefficients, as the report quotes them.
        let flat = swiglu(128, 256, 256, 128);
        assert_eq!(flat.k_plus, ceil_to(band_up(3) - 1.0, 1e-6));
        assert_eq!(flat.k_minus, ceil_to(1.0 - band_down(3), 1e-6));
    }

    /// The GRANITE blocking's floor, recomputed from the stated algebra term by term — the same
    /// guard `rmsnorm_band_tracks_the_mean_path_algebra` is, for the configuration that has no
    /// single-trip fallback to check against.
    #[test]
    fn granite_tiled_k_floor_tracks_the_trip_count_algebra() {
        let (d_model, d_ff, block_n, block_k) = (4096usize, 12800usize, 64usize, 2048usize);
        let (t_k, t_n) = (d_model.div_ceil(block_k) as f64, d_ff.div_ceil(block_n) as f64);
        assert_eq!((t_k, t_n), (2.0, 200.0), "the blocking's trip counts");
        let u = U_F16;
        let g_down = (d_ff as f64 / d_model as f64).sqrt();
        let e_g = 2.0 * u + u * (t_k - 1.0).sqrt() + f32_accum_rel(d_model);
        let e_s = SILU_REL_GAIN * e_g + 3.0 * u;
        let e_h = e_s + e_g + u;
        let e_o = g_down * e_h + f32_accum_rel(d_ff) * g_down + u * g_down * (t_n - 1.0).sqrt();
        let e = swiglu(d_model, d_ff, block_n, block_k);
        assert_eq!(e.floor, ceil_to(GAUSS_TAIL * e_o, 1e-9));
        // ⛔ AND IT MUST STILL DISCRIMINATE. sigma_o = sqrt(d_ff/d_model) = 1.7678, so a floor that
        // reached the output's own RMS would admit any answer of the right magnitude -- which is
        // exactly the vacuous bound the module header rejects the Cauchy-Schwarz floor for. The
        // ceiling is a QUARTER of sigma_o: stated as a number so a future re-derivation that
        // loosened the floor tenfold would fail here rather than quietly pass everything.
        assert!(
            e.floor < 0.25 * g_down,
            "the Granite blocked floor {} has reached {:.2}x the output RMS {g_down:.4} and is no \
             longer discriminating",
            e.floor,
            e.floor / g_down
        );
        // The BLOCKING, not the width, is what the extra allowance buys: the same width flat.
        let flat = swiglu(d_model, d_ff, d_ff, d_model);
        assert!(e.floor > flat.floor);
        assert_eq!(
            (e.k_plus, e.k_minus),
            (flat.k_plus, flat.k_minus),
            "the multiplicative band is the LAST trip's two rounds plus the reference cast at \
             either blocking; only the floor carries the earlier trips"
        );
    }

    #[test]
    fn rope_floor_is_the_cancellation_term_not_underflow() {
        // The floor must be the stimulus-bounded |x1 c| + |x2 s| term; a pure-underflow floor
        // would be ~1e-8 and would make the envelope unsound for an output near a rotation zero.
        let e = rope();
        assert!(e.floor > 1e-3, "rope floor {} is too small to cover a rotation zero", e.floor);
        let want = ceil_to(U_F16 * (1.0 + U_F16) * GAUSS_TAIL * 2f64.sqrt(), 1e-9);
        assert_eq!(e.floor, want);
    }
}
