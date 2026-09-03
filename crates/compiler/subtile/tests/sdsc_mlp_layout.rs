// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "superdsc")]
//! MLP + RESIDUAL BRIDGE PROOF (staged-lowering verification, ≤3-fan-out).
//!
//! The symptom is the residual dominated by a linearly-growing channel (`ch247`), injected by the
//! MLP. The MLP is one bridge in the SDSC lowering: `rmsnorm-out → {gate=norm·Wg, up=norm·Wu} →
//! silu(gate) → silu·up → down=(silu·up)·Wd → residual+down`. This test runs that EXACT emitted
//! op-DAG (the `lower_silumul_node` decomposition: silu then multiply, + the gate/up/down matmuls +
//! the residual add) through the abstract interpreter and requires it equals `mlp_reference`
//! (`hidden + Σ_i silu(Σ n·Wg)·(Σ n·Wu) · Wd`). A divergence localizes the ch247 bug to the MLP bridge.
//!
//! Decode dims (m=1 token): every tensor is `[1, X]`, so `dev_off` collapses to the flat column index
//! (no stick-pad aliasing) — the real on-card decode layout. Run:
//!   cargo test -p scratchy-subtile --features spyre --test sdsc_mlp_layout

use scratchy_subtile::sdsc_abstract::*;

const HIDDEN: usize = 64;
const INTER: usize = 192;
const EPS: f64 = 1e-5;

/// Run the emitted MLP+residual op-DAG (m=1) and return (got_out, want_out) over `[hidden]`.
fn run_mlp() -> (Vec<f64>, Vec<f64>) {
    let (hb, nb, gtb, upb, sib, dnb, ob) =
        (0usize, 1_000, 10_000, 100_000, 200_000, 300_000, 400_000);
    let ht = AbsTensor {
        name: "hidden".into(),
        base: hb,
        dims: vec![1, HIDDEN],
        stick_idx: 1,
    };
    let nt = AbsTensor {
        name: "norm".into(),
        base: nb,
        dims: vec![1, HIDDEN],
        stick_idx: 1,
    };
    let wg = AbsTensor {
        name: "wg".into(),
        base: gtb,
        dims: vec![HIDDEN, INTER],
        stick_idx: 1,
    };
    let wu = AbsTensor {
        name: "wu".into(),
        base: upb,
        dims: vec![HIDDEN, INTER],
        stick_idx: 1,
    };
    let wd = AbsTensor {
        name: "wd".into(),
        base: dnb,
        dims: vec![INTER, HIDDEN],
        stick_idx: 1,
    };
    let gate = AbsTensor {
        name: "gate".into(),
        base: sib,
        dims: vec![1, INTER],
        stick_idx: 1,
    };
    let up = AbsTensor {
        name: "up".into(),
        base: sib + INTER,
        dims: vec![1, INTER],
        stick_idx: 1,
    };
    let silu = AbsTensor {
        name: "silu".into(),
        base: sib + 2 * INTER,
        dims: vec![1, INTER],
        stick_idx: 1,
    };
    let mlp = AbsTensor {
        name: "mlp".into(),
        base: ob,
        dims: vec![1, HIDDEN],
        stick_idx: 1,
    };
    let outh = AbsTensor {
        name: "outh".into(),
        base: ob + HIDDEN,
        dims: vec![1, HIDDEN],
        stick_idx: 1,
    };

    let xf = |c: usize| seed_val(1, c as u64, 0, 0);
    let gam = |c: usize| seed_val(9, c as u64, 0, 0) * 0.05 + 0.04;
    let wgf = |d: usize, i: usize| seed_val(2, d as u64, i as u64, 0);
    let wuf = |d: usize, i: usize| seed_val(3, d as u64, i as u64, 0);
    let wdf = |i: usize, c: usize| seed_val(4, i as u64, c as u64, 0);

    let mut mem = AbsMem::default();
    for c in 0..HIDDEN {
        mem.set(ht.addr(&[0, c]), xf(c));
    }
    // rmsnorm is validated on-card (cos 1.0); seed its OUTPUT from the reference (this bridge is the
    // MLP, not the norm).
    let nrm = rmsnorm_reference(&xf, &gam, HIDDEN, EPS);
    for c in 0..HIDDEN {
        mem.set(nt.addr(&[0, c]), nrm[c]);
    }
    for d in 0..HIDDEN {
        for i in 0..INTER {
            mem.set(wg.addr(&[d, i]), wgf(d, i));
            mem.set(wu.addr(&[d, i]), wuf(d, i));
        }
    }
    for i in 0..INTER {
        for c in 0..HIDDEN {
            mem.set(wd.addr(&[i, c]), wdf(i, c));
        }
    }

    // EXACT emitted op-DAG: gate/up matmuls; lower_silumul_node = silu then multiply; down matmul;
    // residual add. (Each composite step ≤3 fine ops — the ≤3-fan-out rule.)
    let ops = vec![
        AbsOp::Matmul {
            a: nt.clone(),
            w: wg.clone(),
            o: gate.clone(),
        },
        AbsOp::Matmul {
            a: nt.clone(),
            w: wu.clone(),
            o: up.clone(),
        },
        AbsOp::Ew {
            f: EwF::Silu,
            a: gate.clone(),
            ab: Bcast::None,
            b: None,
            bb: Bcast::None,
            o: silu.clone(),
        },
        AbsOp::Ew {
            f: EwF::Mul,
            a: silu.clone(),
            ab: Bcast::None,
            b: Some(up.clone()),
            bb: Bcast::None,
            o: silu.clone(),
        },
        AbsOp::Matmul {
            a: silu.clone(),
            w: wd.clone(),
            o: mlp.clone(),
        },
        AbsOp::Ew {
            f: EwF::Add,
            a: ht.clone(),
            ab: Bcast::None,
            b: Some(mlp.clone()),
            bb: Bcast::None,
            o: outh.clone(),
        },
    ];
    interp(&ops, &mut mem);

    let mlpref = mlp_reference(&|c| nrm[c], &wgf, &wuf, &wdf, HIDDEN, INTER);
    let got: Vec<f64> = (0..HIDDEN).map(|c| mem.get(outh.addr(&[0, c]))).collect();
    let want: Vec<f64> = (0..HIDDEN).map(|c| xf(c) + mlpref[c]).collect();
    (got, want)
}

/// POSITIVE: the emitted MLP+residual op-DAG equals `hidden + mlp_reference(rmsnorm)`.
#[test]
fn emitted_mlp_matches_reference() {
    let (got, want) = run_mlp();
    for c in 0..HIDDEN {
        assert!(
            (got[c] - want[c]).abs() < 1e-9,
            "MLP bridge diverges at out[{c}]: got {} want {} — the ch247-style residual growth is an \
             MLP-decomposition bug (silu/mul/down/residual wiring)",
            got[c],
            want[c]
        );
    }
}

/// NEGATIVE: dropping the residual (down only, no +hidden) provably diverges — proves the residual
/// add is load-bearing and the check is real.
#[test]
fn missing_residual_diverges() {
    let (got, want) = run_mlp();
    // `want` includes +hidden; a forgotten residual would equal `want - hidden`. Confirm that differs.
    let xf = |c: usize| seed_val(1, c as u64, 0, 0);
    let max_abs = (0..HIDDEN)
        .map(|c| (got[c] - (want[c] - xf(c))).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs > 1e-3,
        "residual must be load-bearing; max|diff|={max_abs}"
    );
}

/// A column-block-split intermediate must be WRITTEN through the same nest it is READ through.
/// granite-3.1-8b's 12800-wide MLP intermediate splits into `[0,8192)` and `[8192,12800)`; 2b's 8192
/// never splits, so this path is 8b-only. The raw `cols_start` is a flat element index, but the
/// output is stick-blocked: column `c` of a `[rows, cols]` tensor begins at `(c/64)*(rows*64)`.
///
/// Those are the same number only at rows == 1. So decode wrote the second block correctly and
/// prefill wrote it on top of the first block, leaving `[8192,12800)` unwritten -- the on-card
/// measurement recorded at `pointwise_chunk_out_offset` (`t65[8192..12800] = 0`, ~27% of the MLP).
#[test]
fn split_intermediate_write_offset_is_stick_blocked_not_flat() {
    const STK: u32 = 64;
    for &(rows, cols_start) in &[(1u32, 8192u32), (31, 8192), (64, 8192)] {
        let flat = cols_start;
        let nest = (cols_start / STK) * (rows * STK);
        if rows == 1 {
            assert_eq!(
                flat, nest,
                "decode: the two agree, which is why decode was right"
            );
        } else {
            assert_ne!(flat, nest, "rows={rows}: a flat offset must NOT be used");
            assert_eq!(
                nest,
                cols_start * rows,
                "block base scales with the row count"
            );
        }
    }
}
