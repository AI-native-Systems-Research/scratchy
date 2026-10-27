// SPDX-License-Identifier: Apache-2.0
//! THE EXP-BASE LAW, AS TYPES — the softmax recurrence's exponential step states its base,
//! and every spelling of the device op on this path derives from it.
//!
//! The recurrence (the attention contract, design-note §3): `p = base^(score' − m)`. The device
//! has exactly ONE exponential primitive — `OpFunc::Exp`, **e^x**: the corrected fast-exp
//! `e^x = 2^(x·log2e)` (arch graph `spyre_dd2_basic.mlir`: `c1 = 16·log2e`, `c4` the fp32
//! exponent bias in bit position; `unary_pipeline.ddl`'s `opFuncName="exp"` the same scheme at
//! SEN169_FP16 — the separate `"fastexp"` is the UNCORRECTED trick). Determined statically from
//! the device's own op definitions, three deeptools authorities agreeing:
//! `bench_results/attn-compile-restart/2026-10-24/device-exp-base.md`.
//!
//! ⛔ THIS MODULE EXISTS BECAUSE THE BASE WAS SPELLINGS, NOT A FACT:
//!
//! | the wall (as it stood) | the type that closes it |
//! |---|---|
//! | The attention ladder emitted its exp step as a bare `"exp"` `&'static str` at six sites, beside `elementwise_op_func`'s `Elementwise::Exp` arm spelling the same string independently — the key and the base could drift apart, and a `"fastexp"` swap would run the UNCORRECTED trick and report success | [`ExpBase::op_func`] — the one derivation; the ladder's sites and the table's arm take the key from here |
//! | A program may spell the recurrence in base 2 (`tl.math.exp2` over `qk · (SCALE·LOG2E)`, the splice's restated form) — legal ONLY because the frontend decomposes `exp2(x) ↦ exp(x·ln2)` (`triton-ktir`'s `exp2_as_exp`), so the restatement times the decomposition is the identity. Nothing stated that; the bridge spelled `"exp"` and the door checked the restatement, two spellings of one fact that could drift apart | [`ExpBase::LOG2E`] / [`ExpBase::LN2`] — the two factors as one law's consts, the identity pinned by the doc test below; the door's restatement guard reads the law's const, not an inline `std::f32::consts::LOG2_E` |
//!
//! ⚠️ SCOPE: the law owns the op-func KEY (the `assemble_*` call convention, the input to
//! `op_func_from_str` and the `{op_func}_o{tid}` descriptor name). The WIRE spelling
//! (`OpFunc::name`) and the reverse parse door (`op_func_from_str`) are their own vocabularies —
//! key and wire name differ for four ops — and stay where they are.

/// The softmax recurrence's exponential step, carrying its base.
///
/// One variant, because the device offers one exponential: there is no exp2 op to choose, and a
/// `fastexp` is not a base but a MISSING correction. The variant exists so the step is a VALUE
/// the emission sites must hold — the op-func key and the restatement factors derive from it,
/// and there is no second spelling of either left in this crate's attention path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpBase {
    /// `p = e^(score' − m)` — `OpFunc::Exp`, the device's only exponential primitive.
    E,
}

impl ExpBase {
    /// The dxp op-func KEY this base lowers to — the one spelling, derived from the base.
    /// The attention ladder's exp emissions and `elementwise_op_func`'s `Elementwise::Exp` arm
    /// take it from here; a bare `"exp"` string at an exp site is the two-spellings defect
    /// (the key and the base could drift apart, and a `"fastexp"` swap would run the
    /// UNCORRECTED trick and report success).
    ///
    /// ```
    /// use ktir_superdsc::exp_base::ExpBase;
    ///
    /// // ⭐ THE IDENTITY, PINNED: the restatement times the decomposition is one — the
    /// // two spellings of the recurrence (restated-exp2 programs, raw-exp ladders) agree
    /// // IFF it holds. Each factor is the f32 image of its true value, so the product
    /// // holds to the f32 ulp, not exactly. IF IT EVER FAILS, the two spellings have
    /// // drifted: a program restated in base 2 is being evaluated in base e (every
    /// // probability raised to log2e ≈ 1.4427 — a uniform temperature error), or a
    /// // decomposition is being applied to a raw score (the same error, opposite sign).
    /// // That is not a tolerance question; it is a spelling question, and the fix is at
    /// // the mint that moved.
    /// assert!((ExpBase::LOG2E * ExpBase::LN2 - 1.0).abs() < 1e-6);
    ///
    /// // And the base lowers to the device's one exponential, never the uncorrected trick.
    /// assert_eq!(ExpBase::E.op_func(), "exp");
    /// assert_ne!(ExpBase::E.op_func(), "fastexp");
    /// ```
    pub const fn op_func(self) -> &'static str {
        match self {
            ExpBase::E => "exp",
        }
    }

    /// The restatement a BASE-2 spelling of the recurrence folds into its scores
    /// (`qk · (SCALE · LOG2E)` — the splice's exp2-end-to-end form). Legal only together with
    /// [`Self::LN2`]: the door's identity guard compares a program's stated multiplier against
    /// this value (at the f16 image, the only form a program constant carries), so a program
    /// whose scale is genuinely ONE states `log2e`, never `1.0`.
    pub const LOG2E: f32 = std::f32::consts::LOG2_E;

    /// The decomposition the frontend applies to a base-2 op before it reaches this device
    /// (`exp2(x) ↦ exp(x·ln2)`, `triton-ktir`'s `exp2_as_exp` — the base change is explicit in
    /// the IR rather than assumed). One mint: the frontend's pass. An emitter-side spelling of
    /// this factor would be the half-applied decomposition — a program restated in base 2 and
    /// evaluated in base e, a uniform temperature error on every softmax in the model.
    pub const LN2: f32 = std::f32::consts::LN_2;
}
