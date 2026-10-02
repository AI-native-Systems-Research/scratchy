//! Bridge 3 (`TiledOp -> SdscOp`), RmsNorm composite decomposition.
//!
//! Ported 1:1 from torch-spyre's `spyre_rms_norm` (`torch_spyre/_inductor/decompositions.py:434`):
//!
//! ```python
//! mean = torch.mean(input * input, dim=-1, keepdim=True)
//! rsqrt_inp = torch.rsqrt(mean + eps)
//! output = input * rsqrt_inp
//! if weight is not None:
//!     output = output * weight
//! ```
//!
//! ONE formula, unconditional on row count — torch-spyre's decomposition has no `m==1` vs `m>1`
//! branch at all; the same expression is lowered by Inductor for any shape. This is the whole
//! point of porting it: scratchy previously had a SEPARATE, hand-written decode-only (`rows==1`)
//! implementation (an amax-normalized Newton-Raphson rsqrt, built as a scratchy-specific
//! workaround) alongside this exact torch-spyre form for `rows>1` — two independent code paths
//! that could silently diverge, which is exactly the bug class this whole Islands/Bridges
//! restructuring exists to eliminate. There is now ONE decomposition, used for every row count,
//! matching torch-spyre exactly: `mean(x²) → rsqrt(mean+eps) → x·rsqrt → ·gamma`.

use super::reduce::assemble_reduce_seeded;
use crate::emit::{EmittedOp, In, pw1, pw2, rb};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{BlockCols, Lanes, RowCount};
use crate::superdsc_opspec::{DataFormat, Fp16};

/// Assemble torch-spyre's `spyre_rms_norm` decomposition for one node: `x[rows,cols]`,
/// `gamma[1,cols]`, `eps_const` a bound `[1,stick]` config constant → `out[rows,cols]`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_rmsnorm(
    prefix: &str,
    rows: u32,
    cols: u32,
    x: &str,
    gamma: &str,
    out_id: crate::place::PlaceId,
    eps_const: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    use crate::place::SynthRole as R;
    let stk = Fp16::ELEMS_PER_STICK; // 64 — the per-row reduce/scalar lane (one fp16 stick)
    let mut ops: Vec<EmittedOp> = Vec::new();
    // The op's operand SPELLING, rendered from the identity the layout is keyed by.
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    let sq16 = syn(R::Sq16); // x²        [rows, cols]
    let mean = syn(R::Mean); // mean(x²)  [rows, stk]
    let rinv = syn(R::Rinv); // 1/√mean    [rows, stk]
    let xn = syn(R::Xn); // x·r          [rows, cols]
    let meps = syn(R::Meps); // mean+eps   [rows, stk]
    if let Some(l) = layout {
        for (r, dims) in [
            (R::Sq16, [rows, cols]),
            (R::Mean, [rows, stk]),
            (R::Meps, [rows, stk]),
            (R::Rinv, [rows, stk]),
            (R::Xn, [rows, cols]),
        ] {
            l.synth(out_id.synth(r), &dims);
        }
    }
    // ── HANDLE-FLOW ── bind each tensor's typed handle ONCE, then pass the SAME handle to its producer
    // (as output) and its consumer (as input). A wrong kind is caught at cargo build by whichever consumer
    // REQUIRES a fixed kind.
    //
    // `sq16` is RowBlocked, NOT Flat (2026-07-28). It used to be `fl()`, on the reasoning that its only
    // consumer is the mean-reduce and that the reduce could not walk a stick-major activation at
    // rows>1. That reasoning is now STALE, and forcing Flat here was actively wrong for mq>1:
    //   * `assemble_pointwise_broadcast_off` derives ONE `head_major` for the WHOLE op from the OUTPUT
    //     handle's kind. A Flat output forces head_major=true, which (see pointwise.rs's own comment)
    //     "stays rank-3" -- i.e. it also forces the INPUT `x` to be read rank-3 flat. But `x` is the
    //     residual stream, WRITTEN stick-major by the previous layer's rank-2 pointwise. Reading a
    //     stick-major [rows, cols] tensor as rank-3 flat is, in that same comment's words,
    //     "byte-identical at rows==1, scrambled at rows>1 AND cols>64" -- and here rows=mq=31,
    //     cols=hidden=2048. So mean(x^2) was computed over a SCRAMBLED element set (each flat "row"
    //     spans every real row's first stick), giving every row the wrong normalization scale, every
    //     layer, for batched prefill only.
    //   * The reduce's stick-major path is fully implemented and its two historical defects are fixed:
    //     the `vector::_M_range_check` pod crash came from dropping to rank-2 for cols<=64 ops and is
    //     now gated by `cols > Fp16::ELEMS_PER_STICK` (reduce.rs), and the mixed-rank data/accum
    //     failure is fixed by making the accum rank-2 as well in that path.
    // With RowBlocked, `rmsq` and `rmmean` BOTH take the stick-major rank-2 path at rows>1 && cols>64,
    // so `x` is read in the layout it was actually written in and the reduce walks the same layout.
    // At rows==1 (decode) `stickmajor` is false in both the pointwise and the reduce, so the emission
    // is byte-identical to the previous Flat form -- decode is provably unaffected.
    let x = rb(x, rows, cols);
    let gamma = rb(gamma, 1, cols);
    let out = rb(out, rows, cols);
    let sq16 = rb(&sq16, rows, cols);
    let mean = rb(&mean, rows, stk);
    let meps = rb(&meps, rows, stk);
    let eps = rb(eps_const, 1, stk);
    let rinv = rb(&rinv, rows, stk);
    let xn = rb(&xn, rows, cols);
    // The norm's typed extents: `rows` is the residual stream's token rows, `cols` its feature
    // width, and the per-row mean/rsqrt scalars live in ONE-STICK-wide tensors.
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    let s_cols = BlockCols::of_one_stick(Lanes::FP16);
    // torch-spyre `spyre_rms_norm` (decompositions.py:434), EXACTLY:
    //   mean = torch.mean(input*input, dim=-1)  →  rsqrt(mean+eps)  →  input*·  →  ·*weight
    // 1. sq16 = x·x  (the `input*input`).
    ops.push(pw2(
        &format!("rmsq_{prefix}"),
        "mul",
        t_rows,
        f_cols,
        In::full(&x),
        In::full(&x),
        &sq16,
        sym_id_base,
        layout,
    )); // sq16 = x²
    // 2. mean(x²) = ONE native `mean` reduce over cols (folds 1/N into the reduce scale) — this IS
    //    `torch.mean(input*input, dim=-1)`, not a matmul-by-ones substitute + separate ·(1/cols).
    ops.push(assemble_reduce_seeded(
        &format!("rmmean_{prefix}"),
        "mean",
        rows,
        cols,
        &sq16,
        &mean,
        sym_id_base,
        layout,
    ));
    // 3. mean + eps  (the `+ eps` inside torch.rsqrt).
    ops.push(pw2(
        &format!("rmeps_{prefix}"),
        "add",
        t_rows,
        s_cols,
        In::full(&mean),
        In::scalar(&eps),
        &meps,
        sym_id_base,
        layout,
    )); // mean + eps
    // 4. rinv = rsqrt(mean+eps) — ONE native `rsqrt` (torch.rsqrt), not sqrt+reciprocal.
    ops.push(pw1(
        &format!("rmrsqrt_{prefix}"),
        "rsqrt",
        t_rows,
        s_cols,
        In::full(&meps),
        &rinv,
        sym_id_base,
        layout,
    )); // 1/√(mean+eps)
    // 5. apply: xn = x·rinv (r per-row, out-broadcast over cols); out = xn·gamma ([1,cols], mb-broadcast).
    ops.push(pw2(
        &format!("rmxn_{prefix}"),
        "multiply",
        t_rows,
        f_cols,
        In::full(&x),
        In::col(&rinv),
        &xn,
        sym_id_base,
        layout,
    ));
    ops.push(pw2(
        &format!("rmg_{prefix}"),
        "multiply",
        t_rows,
        f_cols,
        In::full(&xn),
        In::mb(&gamma),
        &out,
        sym_id_base,
        layout,
    ));
    ops
}

/// Assemble the GAINLESS rmsnorm — `x[rows,cols]`, `eps_const` a bound `[1,stick]`
/// config constant → `out[rows,cols] = x · rsqrt(mean(x²) + eps)`.
///
/// The first five steps of [`assemble_rmsnorm`] exactly, with the normalising
/// multiply writing `out` directly: the chain's terminal multiply IS the output,
/// which is the whole difference from [`assemble_rmsnorm`] — there, `x·rinv` is an
/// intermediate (`Xn`) that only the gain multiply reads. The host reference is
/// `SubOp::RmsNormUnit` (`subtile_ir.rs`): unit gain, nothing to apply a
/// GainConvention to, same reduce, no scale.
///
/// The synthetic roles are the SHARED ones (`Sq16`/`Mean`/`Meps`/`Rinv`): the unit
/// form mints no `Xn`, because the normalising multiply's result is the output
/// buffer, not a synthetic intermediate.
#[allow(clippy::too_many_arguments)]
pub fn assemble_rmsnorm_unit(
    prefix: &str,
    rows: u32,
    cols: u32,
    x: &str,
    out_id: crate::place::PlaceId,
    eps_const: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    use crate::place::SynthRole as R;
    let stk = Fp16::ELEMS_PER_STICK; // 64 — the per-row reduce/scalar lane (one fp16 stick)
    let mut ops: Vec<EmittedOp> = Vec::new();
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    let sq16 = syn(R::Sq16); // x²        [rows, cols]
    let mean = syn(R::Mean); // mean(x²)  [rows, stk]
    let rinv = syn(R::Rinv); // 1/√mean    [rows, stk]
    let meps = syn(R::Meps); // mean+eps   [rows, stk]
    if let Some(l) = layout {
        for (r, dims) in [
            (R::Sq16, [rows, cols]),
            (R::Mean, [rows, stk]),
            (R::Meps, [rows, stk]),
            (R::Rinv, [rows, stk]),
        ] {
            l.synth(out_id.synth(r), &dims);
        }
    }
    // Same handle kinds as `assemble_rmsnorm`'s — see its comment for why `x` is
    // RowBlocked, not Flat.
    let x = rb(x, rows, cols);
    let out = rb(out, rows, cols);
    let sq16 = rb(&sq16, rows, cols);
    let mean = rb(&mean, rows, stk);
    let meps = rb(&meps, rows, stk);
    let eps = rb(eps_const, 1, stk);
    let rinv = rb(&rinv, rows, stk);
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    let s_cols = BlockCols::of_one_stick(Lanes::FP16);
    // Steps 1–4 are `assemble_rmsnorm` verbatim — the same formula up to the
    // normalising multiply, because a unit-gain norm is the same norm.
    ops.push(pw2(
        &format!("rmsq_{prefix}"),
        "mul",
        t_rows,
        f_cols,
        In::full(&x),
        In::full(&x),
        &sq16,
        sym_id_base,
        layout,
    )); // sq16 = x²
    ops.push(assemble_reduce_seeded(
        &format!("rmmean_{prefix}"),
        "mean",
        rows,
        cols,
        &sq16,
        &mean,
        sym_id_base,
        layout,
    ));
    ops.push(pw2(
        &format!("rmeps_{prefix}"),
        "add",
        t_rows,
        s_cols,
        In::full(&mean),
        In::scalar(&eps),
        &meps,
        sym_id_base,
        layout,
    )); // mean + eps
    ops.push(pw1(
        &format!("rmrsqrt_{prefix}"),
        "rsqrt",
        t_rows,
        s_cols,
        In::full(&meps),
        &rinv,
        sym_id_base,
        layout,
    )); // 1/√(mean+eps)
    // Step 5 is the TERMINAL multiply here, not `Xn`: `out = x·rinv` with the
    // per-row reciprocal out-broadcast over cols — the same operand mode
    // (`In::col`) the gained form's `rmxn` uses for the same value.
    ops.push(pw2(
        &format!("rmxn_{prefix}"),
        "multiply",
        t_rows,
        f_cols,
        In::full(&x),
        In::col(&rinv),
        &out,
        sym_id_base,
        layout,
    ));
    ops
}

/// Assemble the tanh soft cap — `x[rows,cols]`, `cap_const` a bound `[1,1]`
/// config constant → `out[rows,cols] = cap · tanh(x / cap)`.
///
/// THREE pointwise ops, because that IS the function: `realdiv` by the cap,
/// `tanh`, `multiply` by the cap. Both `realdiv` and `tanh` are DDL primitives
/// already live on card (`attn.rs`'s rescale, `rmsnorm`'s rsqrt chain), and the
/// cap const is read through `In::scalar` — the `[1,1]` out-broadcast mode the
/// rmsnorm uses for its epsilon — so the SAME bound constant serves the divide
/// and the multiply.
///
/// ⭐ NO SYNTHETIC ROLE FOR THE SCALED INTERMEDIATE. `x/cap` is `[rows, cols]`
/// like its producer and consumer, but it is a fresh buffer this lowering
/// mints, so it takes a `SynthRole` (`Tanhc`) and is reserved through the
/// layout exactly as `Sq16` is — placed, never aliased.
#[allow(clippy::too_many_arguments)]
pub fn assemble_tanhsoftcap(
    prefix: &str,
    rows: u32,
    cols: u32,
    x: &str,
    out_id: crate::place::PlaceId,
    cap_const: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    use crate::place::SynthRole as R;
    let mut ops: Vec<EmittedOp> = Vec::new();
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let tanhc = crate::placement::syn(layout, out_id.synth(R::Tanhc)); // tanh(x/cap) [rows, cols]
    if let Some(l) = layout {
        l.synth(out_id.synth(R::Tanhc), &[rows, cols]);
    }
    let x = rb(x, rows, cols);
    let out = rb(out, rows, cols);
    let tanhc = rb(&tanhc, rows, cols);
    let cap = rb(cap_const, 1, Fp16::ELEMS_PER_STICK);
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    // 1. scaled = x / cap (the cap `[1,1]` const, out-broadcast — the rmsnorm's
    //    `eps` operand mode).
    ops.push(pw2(
        &format!("tscdiv_{prefix}"),
        "realdiv",
        t_rows,
        f_cols,
        In::full(&x),
        In::scalar(&cap),
        &tanhc,
        sym_id_base,
        layout,
    ));
    // 2. tanh — in place on the scaled buffer's name, into the same synth.
    ops.push(pw1(
        &format!("tscth_{prefix}"),
        "tanh",
        t_rows,
        f_cols,
        In::full(&tanhc),
        &tanhc,
        sym_id_base,
        layout,
    ));
    // 3. out = cap · tanh(x/cap).
    ops.push(pw2(
        &format!("tscmul_{prefix}"),
        "multiply",
        t_rows,
        f_cols,
        In::full(&tanhc),
        In::scalar(&cap),
        &out,
        sym_id_base,
        layout,
    ));
    ops
}
