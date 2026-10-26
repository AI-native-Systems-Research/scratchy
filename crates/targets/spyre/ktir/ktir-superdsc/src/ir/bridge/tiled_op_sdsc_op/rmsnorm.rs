//! Bridge 3 (`TiledOp -> SdscOp`), RmsNorm composite decomposition.
//!
//! The torch-spyre formula (`torch_spyre/_inductor/decompositions.py:999`, `spyre_rms_norm`):
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
//! restructuring exists to eliminate. There is now ONE decomposition, used for every row count.
//!
//! The HEAD is fused one step past the vendor's own RMS shape. torch-spyre lowers `mean(x·x)`
//! as TWO elementary ops (`x·x` pointwise, then `mean` reduce) — but their LAYER-norm
//! decomposition (decompositions.py:1020, `spyre_layer_norm`) proves the device's own fused
//! head: ONE `exx2` reduce squares and accumulates in a single SFP_FMA inside the reduce, so
//! the [rows,cols] x² intermediate never touches HBM at all. This chain borrows that head for
//! RMS in the vendor's own production configuration (`useZeroMean` = 0 — the plain op both
//! moments are computed in, E[x] simply unread), taking the chain from 6 descriptors to 5.
//! See [`OpFunc::Exx2`]'s doc for the full wire contract.

use super::reduce::assemble_reduce_seeded;
use crate::emit::{EmittedOp, In, pw1, pw2, rb};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{BlockCols, Lanes, RowCount};
use crate::superdsc_opspec::{DataFormat, Fp16};

/// Assemble torch-spyre's `spyre_rms_norm` decomposition for one node: `x[rows,cols]`,
/// `gamma[1,cols]`, `eps_const` a bound `[1,stick]` config constant → `out[rows,cols]`.
///
/// ⭐ THE GAINED NORM'S WHOLE FACT SET, ON ONE CARRIER — the eight facts the decomposition reads,
/// grouped exactly as its own doc frames them, deleting the positional run where a name, a gain
/// or a bound const transposes silently (the arrangement this file shares with every sibling the
/// carrier family has already closed). The sink pair (`sym_id_base`/`layout`) stays in the
/// signature BESIDE the carrier, as it does at every door of this family.
pub struct RmsnormFacts<'a> {
    pub prefix: &'a str,
    pub rows: u32,
    pub cols: u32,
    pub x: &'a str,
    pub gamma: &'a str,
    pub out_id: crate::place::PlaceId,
    pub eps_const: &'a str,
    /// A bound `[1,stick]` constant holding 1.0, stating the OnePlusScale convention's gain
    /// offset: when present, ONE `add` ahead of the gain multiply fills the `[1,cols]` `Gw` row
    /// with `gamma + 1`, and the multiply reads THAT row — the same formula the spliced program
    /// states (`rmsnorm_oneplus_fwd`'s `w = load + GAIN_OFFSET`), never the algebraic
    /// `xn + xn·w` rewrite. `None` is the Scale convention: no add, the multiply reads the
    /// stored gamma, byte-identical to the form before the row existed.
    pub ones_const: Option<&'a str>,
}

pub fn assemble_rmsnorm(
    facts: RmsnormFacts<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    // ⭐ THE NORM'S FACTS, DESTRUCTURED ONCE AT THE DOOR — every field restates the local the
    // body below already reads (the same names the parameter list spelled), so the emission is
    // the same op stream on the same values. The sink pair stays in the signature.
    let RmsnormFacts {
        prefix,
        rows,
        cols,
        x,
        gamma,
        out_id,
        eps_const,
        ones_const,
    } = facts;
    use crate::place::SynthRole as R;
    let stk = Fp16::ELEMS_PER_STICK; // 64 — the per-row reduce/scalar lane (one fp16 stick)
    let mut ops: Vec<EmittedOp> = Vec::new();
    // The op's operand SPELLING, rendered from the identity the layout is keyed by.
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    let mean = syn(R::Mean); // (E[x], E[x²])  [rows, stk]
    let rinv = syn(R::Rinv); // 1/√E[x²]   [rows, stk]
    let xn = syn(R::Xn); // x·r          [rows, cols]
    let meps = syn(R::Meps); // E[x²]+eps   [rows, stk]
    // γ+1        [1, cols]  — OnePlusScale only
    let gw = syn(R::Gw);
    if let Some(l) = layout {
        for (r, dims) in [
            (R::Mean, [rows, stk]),
            (R::Meps, [rows, stk]),
            (R::Xn, [rows, cols]),
        ] {
            l.synth(out_id.synth(r), &dims);
        }
        // `rinv` is PINNED-LX, not seg3-HBM (`synth_lx_op_local`): its consumer reads a SUB-STICK lane
        // (element 8 of the exx2 pair — see step 4 below), and a sub-stick read offset survives
        // the compile chain only on the LX path. On HBM the +16B is floored away twice over —
        // the L3 EBR is stick-granular (the reader's allocate carries the offset but the
        // EBR-bound L3 leg rounds to the stick base) and the reader's 2bsplat then reads the
        // landing stick at imm 0, i.e. element 0 = rsqrt(E[x]+eps) ≈ 316: a saturated constant
        // that turned every prior card run into byte-identical "signature" garbage with a GREEN
        // bake. An LX allocate is byte-granular end-to-end (lbrInit_ in bytes → lxlu register →
        // LDSTU address), so writer-at-pin / reader-at-pin+16 is exactly expressible —
        // bake-verified against dxp_standalone (the +16 lands in the compiled reader registers;
        // every HBM leg for the tensor vanishes).
        //
        // The producer side of the pin: LX is per-core private, so `apply_lx_residency`
        // replicates `rmrsqrt` across all cores (each computes the whole [rows, stk] stick into
        // its own scratchpad) and the reader's own slice corner selects its rows there.
        l.synth_lx_op_local(out_id.synth(R::Rinv), &[rows, stk]);
        if ones_const.is_some() {
            l.synth(out_id.synth(R::Gw), &[1, cols]);
        }
    }
    // ── HANDLE-FLOW ── bind each tensor's typed handle ONCE, then pass the SAME handle to its producer
    // (as output) and its consumer (as input). A wrong kind is caught at cargo build by whichever consumer
    // REQUIRES a fixed kind.
    //
    // `x` feeds the reduce DIRECTLY now (the exx2 head squares inside the reduce), so the reduce's data
    // kind is `x`'s own kind — RowBlocked, which is exactly what the stick-major reduce path needs
    // (2026-07-28 history, still load-bearing): `x` is the residual stream, WRITTEN stick-major by the
    // previous layer's rank-2 pointwise. A Flat data handle would force the reduce rank-3 flat, which
    // reads a stick-major [rows, cols] tensor "byte-identical at rows==1, scrambled at rows>1 AND
    // cols>64" — at prefill rows=mq, cols=hidden=2048, so every row would get the wrong normalization
    // scale, every layer. With RowBlocked, the exx2 reduce takes the stick-major rank-2 path at
    // rows>1 && cols>64 and walks `x` in the layout it was actually written in. At rows==1 (decode)
    // `stickmajor` is false in the reduce, so the emission is byte-identical to the flat form — decode
    // is provably unaffected.
    // The norm's typed extents: `rows` is the residual stream's token rows, `cols` its feature
    // width, and the per-row mean/rsqrt scalars live in ONE-STICK-wide tensors.
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    let s_cols = BlockCols::of_one_stick(Lanes::FP16);
    let x = rb(x, t_rows, f_cols);
    let gamma = rb(gamma, RowCount::of_broadcast_row(), f_cols);
    let out = rb(out, t_rows, f_cols);
    let mean = rb(&mean, t_rows, s_cols);
    let meps = rb(&meps, t_rows, s_cols);
    let eps = rb(eps_const, RowCount::of_broadcast_row(), s_cols);
    let rinv = rb(&rinv, t_rows, s_cols);
    let xn = rb(&xn, t_rows, f_cols);
    // The `spyre_rms_norm` formula with the vendor's own LAYER-norm head (their
    // `spyre_layer_norm`, decompositions.py:1020, opens with exactly this reduce):
    //   exx2(1/cols)  →  rsqrt(E[x²]+eps)  →  x·r  →  ·γ
    // 1. E[x²] = ONE native `exx2` reduce reading `x` directly — the square happens INSIDE the
    //    reduce (SFP_FMA x·x+acc, one rounding — the vendor's own fused numerics), so no [rows,cols]
    //    x² intermediate is ever materialized in HBM. The reduce writes TWO half-stick values per
    //    row: elements [0..8) = E[x] (computed and unread — the plain op computes both moments),
    //    elements [8..16) = E[x²]·(1/cols scale) — the vendor's own write AND read-back of the
    //    second value both sit at `stick_replicated_dim_offset_elements=8` (summeanmaxexx2.ddl:625/
    //    :192, and layernormscale_32.ddl:75 reads it there too). The reduce's 1/N rides the op's
    //    `exx2scale` const (same channel the mean reduce's `scaling_factor` used), and
    //    `useZeroMean` stays the RAW 0 — the vendor's production spelling (the =1 rewrite names
    //    an op the runtime smc table cannot resolve) — see OpInfo::Exx2Scale.
    ops.push(assemble_reduce_seeded(
        &format!("rmmean_{prefix}"),
        "exx2",
        rows,
        cols,
        &x,
        &mean,
        sym_id_base,
        layout,
    ));
    // 2. E[x²] + eps  (the `+ eps` inside torch.rsqrt). `mean` holds both half-stick values, so
    //    elements [0..8) compute E[x]+eps and the pad elements compute over stale bytes — both
    //    unread (the tail reads element 8 only), the same computed-but-unread padding class the
    //    elementary chain always had.
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
    )); // E[x²] + eps
    // 3. rinv = rsqrt(E[x²]+eps) — ONE native `rsqrt` (torch.rsqrt), not sqrt+reciprocal.
    ops.push(pw1(
        &format!("rmrsqrt_{prefix}"),
        "rsqrt",
        t_rows,
        s_cols,
        In::full(&meps),
        &rinv,
        sym_id_base,
        layout,
    )); // 1/√(E[x²]+eps)
    // 4. apply: xn = x·r (r per-row, out-broadcast over cols); out = xn·gamma ([1,cols], mb-broadcast).
    //    `rinv`'s element 8 IS E[x²]'s rsqrt — the FIRST element of the exx2 output's second
    //    half-stick ([8..16); the vendor's own consumers read it at exactly this offset,
    //    `stick_replicated_dim_offset_elements=8`) — read via a one-element view step into the
    //    single-stick-wide view (same `col_at` idiom as the attention finalize's per-head scalar).
    //    ⛔ NOT element 1: the first card run of this chain read element 1 (the E[x] half, ≈0 for
    //    a signed activation row), which made the whole norm multiply by rsqrt(E[x]+eps) — a
    //    saturated, degenerate generation. The half-stick map is the DDL's, not a guess.
    //    The +8 elements (= +16 B, one fp16 word past the pair's first slot) rides the reader's
    //    allocate through the ordinary slice-offset machinery — and because `rinv` is LX-pinned,
    //    that byte lands BYTE-exact in the reader's lxlu address (see the `synth_lx_op_local` note above
    //    for why the same offset on HBM is silently floored to element 0).
    ops.push(pw2(
        &format!("rmxn_{prefix}"),
        "multiply",
        t_rows,
        f_cols,
        In::full(&x),
        In::col_at(&rinv, crate::addr::DevOff::from_view_step(8)),
        &xn,
        sym_id_base,
        layout,
    ));
    // 6. OnePlusScale: gw = gamma + 1 — ONE `add` over the stored row with the
    //    `[1,1]`-broadcast offset (the eps operand mode), filling the `[1,cols]`
    //    `Gw` row the gain multiply reads. The stored gamma is never rewritten.
    //    The handles are hoisted OUT of the arm because `In::mb` borrows: the
    //    multiply's operand must outlive the `match` that minted it.
    let gw_ones_h = ones_const.map(|c| {
        (
            rb(
                &gw,
                RowCount::of_broadcast_row(),
                BlockCols::of_feature_cols(cols),
            ),
            rb(
                c,
                RowCount::of_broadcast_row(),
                BlockCols::of_one_stick(Lanes::FP16),
            ),
        )
    });
    let gamma_operand = match &gw_ones_h {
        Some((gw_handle, ones)) => {
            ops.push(pw2(
                &format!("rmga_{prefix}"),
                "add",
                RowCount::of_token_rows(1),
                BlockCols::of_feature_cols(cols),
                In::mb(&gamma),
                In::scalar(ones),
                gw_handle,
                sym_id_base,
                layout,
            )); // gw = γ + 1
            In::mb(gw_handle)
        }
        None => In::mb(&gamma),
    };
    ops.push(pw2(
        &format!("rmg_{prefix}"),
        "multiply",
        t_rows,
        f_cols,
        In::full(&xn),
        gamma_operand,
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
///
/// ⭐ THE UNIT NORM'S FACT SET, ON ONE CARRIER — the gained form's carrier minus the gain facts
/// (`gamma`/`ones_const`), because a unit-gain norm has none to state. The sink pair stays in
/// the signature BESIDE the carrier.
pub struct RmsnormUnitFacts<'a> {
    pub prefix: &'a str,
    pub rows: u32,
    pub cols: u32,
    pub x: &'a str,
    pub out_id: crate::place::PlaceId,
    pub eps_const: &'a str,
}

pub fn assemble_rmsnorm_unit(
    facts: RmsnormUnitFacts<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    // ⭐ THE UNIT NORM'S FACTS, DESTRUCTURED ONCE AT THE DOOR — the same names the parameter
    // list spelled, so the body below is unchanged arithmetic on the same values.
    let RmsnormUnitFacts {
        prefix,
        rows,
        cols,
        x,
        out_id,
        eps_const,
    } = facts;
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
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    let s_cols = BlockCols::of_one_stick(Lanes::FP16);
    let x = rb(x, t_rows, f_cols);
    let out = rb(out, t_rows, f_cols);
    let sq16 = rb(&sq16, t_rows, f_cols);
    let mean = rb(&mean, t_rows, s_cols);
    let meps = rb(&meps, t_rows, s_cols);
    let eps = rb(eps_const, RowCount::of_broadcast_row(), s_cols);
    let rinv = rb(&rinv, t_rows, s_cols);
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
///
/// ⭐ THE SOFTCAP'S FACT SET, ON ONE CARRIER — the unit form's frame with the bound const
/// renamed to what it IS here (the cap, not an epsilon): the names are the safety, so the
/// semantic rename travels with the carrier rather than living in a positional slot. The sink
/// pair stays in the signature BESIDE the carrier.
pub struct TanhsoftcapFacts<'a> {
    pub prefix: &'a str,
    pub rows: u32,
    pub cols: u32,
    pub x: &'a str,
    pub out_id: crate::place::PlaceId,
    pub cap_const: &'a str,
}

pub fn assemble_tanhsoftcap(
    facts: TanhsoftcapFacts<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Vec<EmittedOp> {
    // ⭐ THE SOFTCAP'S FACTS, DESTRUCTURED ONCE AT THE DOOR — the same names the parameter list
    // spelled, so the body below is unchanged arithmetic on the same values.
    let TanhsoftcapFacts {
        prefix,
        rows,
        cols,
        x,
        out_id,
        cap_const,
    } = facts;
    use crate::place::SynthRole as R;
    let mut ops: Vec<EmittedOp> = Vec::new();
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let tanhc = crate::placement::syn(layout, out_id.synth(R::Tanhc)); // tanh(x/cap) [rows, cols]
    if let Some(l) = layout {
        l.synth(out_id.synth(R::Tanhc), &[rows, cols]);
    }
    let t_rows = RowCount::of_token_rows(rows);
    let f_cols = BlockCols::of_feature_cols(cols);
    let x = rb(x, t_rows, f_cols);
    let out = rb(out, t_rows, f_cols);
    let tanhc = rb(&tanhc, t_rows, f_cols);
    let cap = rb(
        cap_const,
        RowCount::of_broadcast_row(),
        BlockCols::of_one_stick(Lanes::FP16),
    );
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
