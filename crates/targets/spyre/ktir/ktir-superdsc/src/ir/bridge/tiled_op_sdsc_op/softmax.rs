// SPDX-License-Identifier: Apache-2.0
//! Bridge 3 (`TiledOp -> SdscOp`), the row-softmax decomposition.
//!
//! The MoE router's `RouteSoftmax` — `softmax(scores, dim=-1)` over `[rows, experts]` —
//! and `RouteRenorm` (mixtral's `scores / rowsum(scores)`, the same chain minus the
//! exp) decompose into primitives every one of which is already live on card:
//!
//! ```text
//! m  = max(x, dim=-1)          one native `max` reduce  → [rows, stick]
//! z  = x − m                   pointwise `sub`, m out-broadcast over cols
//! e  = exp(z)                  pointwise `exp`
//! d  = sum(e, dim=-1)          one native `sum` reduce  → [rows, stick]
//! y  = e / d                   pointwise `realdiv`, d out-broadcast over cols
//! ```
//!
//! The structure is `assemble_rmsnorm`'s — a per-row `[rows, stick]` reduce, an
//! out-broadcast combine, a pointwise chain, a second reduce, a terminal
//! out-broadcast divide — because a row softmax and a row rms are the same shape
//! of computation: per-row scalars sprayed back over the row.

use super::reduce::assemble_reduce_seeded;
use crate::emit::{EmittedOp, In, pw1, pw2, rb};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{BlockCols, RowCount};
use crate::superdsc_opspec::{DataFormat, Fp16};

/// Assemble the row softmax — `x[rows,cols]` → `out[rows,cols] = softmax(x, dim=-1)`.
///
/// The five-op chain above; the row-max and row-sum live in one-stick synthetics
/// (`RMax`/`RDen`), the shifted scores in `RSub`, exactly as rmsnorm's mean/rinv do.
pub fn assemble_row_softmax(
    prefix: &str,
    rows: u32,
    cols: u32,
    x: &str,
    out_id: crate::place::PlaceId,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    use crate::place::SynthRole as R;
    let stk = Fp16::ELEMS_PER_STICK; // 64 — the per-row reduce/scalar lane (one fp16 stick)
    let mut ops: Vec<EmittedOp> = Vec::new();
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    let rmax = syn(R::RMax); // rowmax(x)   [rows, stk]
    let rsub = syn(R::RSub); // x − rowmax  [rows, cols]
    let rden = syn(R::RDen); // rowsum(exp) [rows, stk]
    if let Some(l) = layout {
        for (r, dims) in [
            (R::RMax, vec![rows, stk]),
            (R::RSub, vec![rows, cols]),
            (R::RDen, vec![rows, stk]),
        ] {
            l.synth(out_id.synth(r), &dims);
        }
    }
    // Same handle kinds as `assemble_rmsnorm`'s — `x` is RowBlocked (stick-major at
    // rows>1), the per-row scalars one-stick-wide, the broadcast the `In::col`
    // out-broadcast the rmsnorm's `rmxn` uses for the same per-row value.
    let x = rb(x, rows, cols);
    let out = rb(out, rows, cols);
    let rmax = rb(&rmax, rows, stk);
    let rsub = rb(&rsub, rows, cols);
    let rden = rb(&rden, rows, stk);
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    // 1. rmax = max(x, dim=-1) — ONE native `max` reduce.
    ops.push(assemble_reduce_seeded(
        &format!("rsmax_{prefix}"),
        "max",
        rows,
        cols,
        &x,
        &rmax,
        sym_id_base,
        layout,
    ));
    // 2. rsub = x − rmax (r per-row, out-broadcast over cols).
    ops.push(pw2(
        &format!("rssub_{prefix}"),
        "sub",
        t_rows,
        f_cols,
        In::full(&x),
        In::col(&rmax),
        &rsub,
        sym_id_base,
        layout,
    ));
    // 3. exp(rsub) — in place over the shifted buffer, then its row sum.
    //    ⛔ THE EXP'S DESTINATION IS `RSub` ITSELF, reused as its own tanh's
    //    destination in `assemble_tanhsoftcap`'s pattern: a fresh `[rows, cols]`
    //    synthetic for a value nothing else reads would double the softmax's
    //    footprint for nothing.
    ops.push(pw1(
        &format!("rsexp_{prefix}"),
        "exp",
        t_rows,
        f_cols,
        In::full(&rsub),
        &rsub,
        sym_id_base,
        layout,
    ));
    // 4. rden = sum(exp(x − rowmax), dim=-1) — ONE native `sum` reduce.
    ops.push(assemble_reduce_seeded(
        &format!("rssum_{prefix}"),
        "sum",
        rows,
        cols,
        &rsub,
        &rden,
        sym_id_base,
        layout,
    ));
    // 5. out = exp / rden (r per-row, out-broadcast over cols).
    ops.push(pw2(
        &format!("rsdiv_{prefix}"),
        "realdiv",
        t_rows,
        f_cols,
        In::full(&rsub),
        In::col(&rden),
        &out,
        sym_id_base,
        layout,
    ));
    ops
}

/// Assemble the row renorm — `x[rows,cols]` → `out[rows,cols] = x / rowsum(x, dim=-1)`.
///
/// Mixtral's `RouteRenorm`, the softmax chain minus the stability subtract and the
/// exp: `d = sum(x)` then `out = x / d`, both broadcasts the softmax's own.
pub fn assemble_row_renorm(
    prefix: &str,
    rows: u32,
    cols: u32,
    x: &str,
    out_id: crate::place::PlaceId,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    use crate::place::SynthRole as R;
    let stk = Fp16::ELEMS_PER_STICK;
    let mut ops: Vec<EmittedOp> = Vec::new();
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    let rden = syn(R::RDen); // rowsum(x)  [rows, stk]
    if let Some(l) = layout {
        l.synth(out_id.synth(R::RDen), &[rows, stk]);
    }
    let x = rb(x, rows, cols);
    let out = rb(out, rows, cols);
    let rden = rb(&rden, rows, stk);
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    // 1. rden = sum(x, dim=-1).
    ops.push(assemble_reduce_seeded(
        &format!("rrsum_{prefix}"),
        "sum",
        rows,
        cols,
        &x,
        &rden,
        sym_id_base,
        layout,
    ));
    // 2. out = x / rden (r per-row, out-broadcast over cols).
    ops.push(pw2(
        &format!("rrdiv_{prefix}"),
        "realdiv",
        t_rows,
        f_cols,
        In::full(&x),
        In::col(&rden),
        &out,
        sym_id_base,
        layout,
    ));
    ops
}
