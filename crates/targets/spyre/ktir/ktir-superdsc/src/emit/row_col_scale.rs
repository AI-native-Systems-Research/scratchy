// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE ROW-THEN-COLUMN SCALE FOLD — `out = (x · r[:, None]) · g[None, :]` as ONE device op.
//!
//! # The pattern and the device op it maps to
//!
//! | pattern (two broadcast multiplies)                   | device op                                   |
//! |------------------------------------------------------|---------------------------------------------|
//! | `t = mul(x, col(r))`, `out = mul(t, mb(g))`, `t` read by the second multiply only | `layernormnorm(x, 0, r, g, 0)` |
//!
//! `layernormnorm.ddl` (its one fp16 bind, `:15`; runtime `OpFuncs::LAYERNORM_NORM`,
//! `updatelibrary.cpp:95`) computes, per element:
//!
//! ```text
//!   d   = FNMS(scale, mean, 0)        PE    −scale·mean, one value per row
//!   t   = FMA16(x, scale, d)          PE    x·scale + d
//!   out = FMA16(t, γ, β)              SFP   t·γ + β
//! ```
//!
//! With `mean = 0` and `β = 0` that is `d = −0` (or `+0`), `t = round(x·r + d)`, `out = round(t·g + 0)`.
//! Adding a signed zero to a product is exact and keeps the product's own sign unless the product is
//! itself a zero of the opposite sign, so `t = round(x·r)` and `out = round(t·g)` — the SAME two
//! roundings, in the SAME order, that the pair of `mul`s it replaces performs (each broadcast `mul` is
//! `MACC(a, b, 0)`, `broadcast_ops.ddl`'s mul arm). The chain IS the op at the program's precision;
//! it is not an approximation of it.
//!
//! # The shape predicate — ONE, here
//!
//! [`RowColScale::of`] is the only place a shape is judged: `x` and `out` are `[rows, cols]` with
//! `cols` a whole number of fp16 sticks, `r` is one value per row (`[rows, stick]`, element 0 of each
//! row's stick — the per-row-scalar form the fp8 quantize writes its scale in), and `g` is one row of
//! `cols` (`[1, cols]`). Nothing here names a model, a layer kind or a kernel: the fold is a law over
//! the two multiplies' operand forms.
//!
//! # What is NOT claimed
//!
//! The interior product `t` must have exactly ONE reader — the second multiply — because the fold
//! never materialises it. The caller states that: it is the producer of `t`, and passes the fold only
//! a `t` it minted and nothing else reads. A chain whose links do not close stays two multiplies.
//!
//! # The zero operands
//!
//! `mean` and `β` are the staged zero rows [`ZERO_ROW_TID`] (`[rows, stick]`) and [`ZERO_COL_TID`]
//! (`[1, cols]`), worker-bound like the fp8 clamp constants. Each fold's read sits inside its
//! placement: the bundle places them at the widest rows / cols any fold in it states.

use super::{EmittedOp, emit_sdsc_tiled, rb};
use crate::placement::BundleLayout;
use crate::reserved_tids::{ZERO_COL_TID, ZERO_ROW_TID};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{DataFormat, Fp16, OpFunc, SdscFoldSet};

/// The fold's one shape predicate, proven: what [`assemble`] lowers.
///
/// Built only by [`RowColScale::of`], so a lowering cannot be handed extents the predicate did not
/// check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowColScale {
    rows: u32,
    cols: u32,
}

impl RowColScale {
    /// ⭐ THE SHAPE PREDICATE — the ONE statement of which `[rows, cols]` this fold covers.
    ///
    /// * `rows ≥ 1`;
    /// * `cols` a positive whole number of fp16 sticks — `layernormnorm.ddl` binds the stick on the
    ///   normalisation dim and its `slice`/`stick` constraints hold every operand to one stick
    ///   layout, so a ragged last stick is not a shape this op states.
    ///
    /// Anything else is `Err` naming the extent, and the caller lowers the two multiplies instead.
    pub fn of(rows: u32, cols: u32) -> Result<Self, String> {
        if rows == 0 {
            return Err("a row-then-column scale over zero rows".to_string());
        }
        if cols == 0 || !cols.is_multiple_of(Fp16::ELEMS_PER_STICK) {
            return Err(format!(
                "a row-then-column scale over {cols} columns — `layernormnorm` needs a whole \
                 number of {}-element fp16 sticks",
                Fp16::ELEMS_PER_STICK
            ));
        }
        Ok(RowColScale { rows, cols })
    }

    pub fn rows(self) -> u32 {
        self.rows
    }

    pub fn cols(self) -> u32 {
        self.cols
    }
}

/// The fold's operands by name — the same spellings the two multiplies it replaces read.
pub struct RowColOperands<'a> {
    /// `x`, `[rows, cols]`.
    pub x: &'a str,
    /// `r`, one value per row of a `[rows, stick]` per-row-scalar tensor, read at element
    /// `row_scale_lane` of each row's stick (0 for the fp8 `ascale`; 8 for the rmsnorm's `rinv`, the
    /// exx2 output's E[x²] half — a lane select, byte-exact only on an LX-pinned operand).
    pub row_scale: &'a str,
    pub row_scale_lane: crate::addr::DevOff,
    /// `g`, one row: `[1, cols]`.
    pub col_scale: &'a str,
    /// `out`, `[rows, cols]`.
    pub out: &'a str,
}

/// `γ`'s and `β`'s positions in `layernormnorm.ddl:15`'s input list — `x, mean, scale, γ, β`.
const GAMMA_ARG: usize = 3;
const BETA_ARG: usize = 4;

/// Lower `(x · r[:, None]) · g[None, :]` to ONE `layernormnorm` descriptor.
///
/// The operands are bound in the DDL's own order — `x`, `mean`, `scale`, `γ`, `β` — through the
/// pointwise family's operand forms: `x` and `out` full, `mean` and `scale` per-row scalars (the
/// `RedStick` broadcast `In::col` states), `γ` and `β` one row broadcast down (`In::mb`). The work
/// division, the mint's bounds checks and the arity check are the family's own.
pub fn assemble<T: crate::collective::Tp>(
    op_name: &str,
    shape: RowColScale,
    o: RowColOperands<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp<T>, SuperDscError> {
    let (rows, cols) = (shape.rows, shape.cols);
    let stk = Fp16::ELEMS_PER_STICK;
    let zero_row = crate::place::act_name(ZERO_ROW_TID);
    let zero_col = crate::place::act_name(ZERO_COL_TID);
    let x = rb(o.x, rows, cols);
    let mean = rb(&zero_row, rows, stk);
    let scale = rb(o.row_scale, rows, stk);
    let gamma = rb(o.col_scale, 1, cols);
    let beta = rb(&zero_col, 1, cols);
    let out = rb(o.out, rows, cols);
    let inputs = [
        super::In::full(&x).ew(),
        super::In::col(&mean).ew(),
        super::In::col_at(&scale, o.row_scale_lane).ew(),
        super::In::mb(&gamma).ew(),
        super::In::mb(&beta).ew(),
    ];
    super::mint_ew_output(
        op_name,
        &out,
        0,
        rows,
        cols,
        crate::superdsc_opspec::Df::Fp16,
        layout,
    )
    .map_err(SuperDscError)?;
    let mut op = super::pointwise_broadcast_opspec_df(
        OpFunc::LayerNormNorm,
        rows,
        cols,
        &inputs,
        out.name(),
        crate::addr::DevOff::ZERO.into(),
        false,
        crate::superdsc_opspec::Df::Fp16,
        layout,
        crate::work::EwDivision::Law,
    )
    .map_err(SuperDscError)?;
    // ⛔ `γ` AND `β` ARE DECLARED OVER THE NORMALISATION DIM ALONE. `layernormnorm.ddl:14` gives
    // `%lnA, %lnB` the global layout `%global_layout_nd = ddl.layout(%nd)` — one dim — and
    // `DdlConversion` (`ddl_conversion.cpp:2410`) refuses a tensor with more non-broadcast dims than
    // its layout names ("Not enough dimensions" → "Impossible to match"). The family's `mb` operand
    // form broadcasts the rows and leaves the size-1 `y` active, so `y` is stated broadcast here, as
    // the vendor's own fixture states it (`sdsc_layernormnorm.json`, lds 3/4 `scale_ [-1, 1, -1]`).
    // `y` is extent 1, so the scale changes no address; a rank-2 (stick-major) form has no `y`.
    for idx in [GAMMA_ARG, BETA_ARG] {
        let has_y = op.args[idx].view().layout.contains(&"y");
        if has_y {
            op.args[idx].set_scale_for_dim("y", crate::superdsc_opspec::Scale::RedNonStick);
        }
    }
    emit_sdsc_tiled(
        op_name,
        &op,
        &SdscFoldSet::new(op.iter.cores_used()),
        sym_id_base,
        layout,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktir_core::codec::round_dlf16;

    /// The two multiplies the fold replaces, element by element, each `MACC(a, b, 0)` landing on the
    /// DLF16 grid: `round(round(x·r)·g)`.
    fn two_muls(x: f32, r: f32, g: f32) -> f32 {
        round_dlf16(round_dlf16(x * r + 0.0) * g + 0.0)
    }

    /// `layernormnorm` at `mean = 0`, `β = 0`, per its DDL: `d = FNMS(r, 0, 0)`, `t = FMA16(x, r, d)`,
    /// `out = FMA16(t, g, 0)` — each a fused multiply-add landing on the DLF16 grid.
    fn layernormnorm(x: f32, mean: f32, r: f32, g: f32, beta: f32) -> f32 {
        let d = round_dlf16(-(r * mean) + 0.0);
        let t = round_dlf16(x.mul_add(r, d));
        round_dlf16(t.mul_add(g, beta))
    }

    /// ⭐ THE FOLD IS BIT-EXACT AGAINST THE TWO MULTIPLIES, over every sign, a decade sweep of
    /// magnitudes, the DLF16 range ends, and exact zeros — the values the chain can carry.
    #[test]
    fn the_fold_is_the_two_multiplies_bit_for_bit() {
        let mut vals: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 448.0, 1.0 / 448.0, 0.22];
        let mut v = 1.0e-9f32;
        while v < 1.0e9 {
            vals.push(round_dlf16(v));
            vals.push(round_dlf16(-v * 1.37));
            v *= 3.1;
        }
        let mut checked = 0u64;
        for &x in &vals {
            for &r in &vals {
                for &g in &vals {
                    let want = two_muls(x, r, g);
                    let got = layernormnorm(x, 0.0, r, g, 0.0);
                    if want.is_nan() {
                        assert!(got.is_nan(), "x={x} r={r} g={g}: want NaN, got {got}");
                    } else {
                        assert_eq!(
                            want.to_bits(),
                            got.to_bits(),
                            "x={x} r={r} g={g}: two muls {want}, layernormnorm {got}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 100_000, "the sweep covers {checked} triples");
    }

    /// A NONZERO mean or β is a DIFFERENT function — the fixture that keeps the zero operands load-
    /// bearing: staging anything but zeros would not be this fold.
    #[test]
    fn a_nonzero_mean_or_beta_is_not_the_two_multiplies() {
        let (x, r, g) = (3.0f32, 0.5, 2.0);
        assert_ne!(two_muls(x, r, g), layernormnorm(x, 1.0, r, g, 0.0));
        assert_ne!(two_muls(x, r, g), layernormnorm(x, 0.0, r, g, 1.0));
    }

    /// ⭐ THE ORDER IS PART OF THE PATTERN: `(x·g)·r` rounds in the other order, and is not the op.
    /// A witness triple where the two orders round differently, so a matcher that accepted either
    /// order would be lowering a different chain.
    #[test]
    fn the_column_first_order_rounds_differently() {
        let mut found = false;
        let vals = [
            1.0f32 + 1.0 / 512.0,
            3.0 + 1.0 / 256.0,
            0.333,
            7.77,
            1.0e-3,
            123.4,
        ];
        'outer: for &x in &vals {
            for &r in &vals {
                for &g in &vals {
                    let row_first = round_dlf16(round_dlf16(x * r) * g);
                    let col_first = round_dlf16(round_dlf16(x * g) * r);
                    if row_first.to_bits() != col_first.to_bits() {
                        found = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(found, "some triple rounds differently in the two orders");
    }

    /// The shape predicate: whole sticks only, at least one row.
    #[test]
    fn the_shape_predicate_admits_whole_sticks_only() {
        assert!(RowColScale::of(1, 64).is_ok());
        assert!(RowColScale::of(32, 2048).is_ok());
        assert!(RowColScale::of(0, 64).is_err(), "zero rows");
        assert!(RowColScale::of(1, 0).is_err(), "zero cols");
        assert!(RowColScale::of(1, 96).is_err(), "a ragged last stick");
    }
}
