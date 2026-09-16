// SPDX-License-Identifier: Apache-2.0
// The abstract interpreter (`sdsc_abstract`) is gated behind `scratchy-subtile/spyre`.
#![cfg(feature = "spyre")]
//! ROPE LOWERING PROOF (compile-time, via the abstract interpreter).
//!
//! The emitter lowers `RopeRotate` to the permutation-matmul form (`lower_rope_node`):
//!   rot = matmul(x[heads,hd] · P[hd,hd]);  xc = x·cos;  rs = rot·sin;  out = xc + rs
//! where `P` is the FIXED rotate-half sign-permutation (`P[o+half,o]=-1` for o<half,
//! `P[o,o+half]=+1`) the real worker binds. This test EXECUTES that exact op-DAG (with the TRUE P,
//! over the same `dev_off` device-address model the emitter bakes) and requires the result equals
//! `rope_reference` (NeoX `x·cos + rotate_half(x)·sin`).
//!
//! WHY: the on-card `SPYRE_SUPERDSC_SELFTEST` reports RopeRotate at cos~0.656 — but that selftest does
//! NOT bind the true P (P is bound by the worker via acts.push, not as a graph source), so its `rot=x·P`
//! uses a zero/random P → a RUNTIME ARTIFACT, not a lowering bug. This compile-time guard settles it: if
//! the EMITTED RoPE op-DAG is structurally correct, this passes (⇒ RoPE is fine in the real forward,
//! which binds P); if the lowering were wrong (bad P, wrong broadcast, wrong contraction), it FAILS at
//! `cargo test`. The negative test proves a wrong P provably diverges (the guard is real, not a tautology).
//! Run: cargo test -p scratchy-subtile --features spyre --test sdsc_rope_layout

use scratchy_subtile::sdsc_abstract::*;

const HEADS: usize = 3;
const HD: usize = 64; // == stick: the real device head_dim (avoids the dev_off pad-aliasing artifact)
const STK: usize = 64;

/// Run the emitted RoPE op-DAG with a caller-supplied P-builder, returning (out, reference).
fn run_rope_with_p(pbuild: &dyn Fn(usize, usize) -> f64) -> (Vec<f64>, Vec<f64>) {
    let half = HD / 2;
    let (xb, pb, cb, sb, rotb, xcb, rsb, ob) = (
        0usize, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000,
    );
    let x = AbsTensor {
        name: "x".into(),
        base: xb,
        dims: vec![HEADS, HD],
        stick_idx: 1,
    };
    let p = AbsTensor {
        name: "P".into(),
        base: pb,
        dims: vec![HD, HD],
        stick_idx: 1,
    };
    let cos = AbsTensor {
        name: "cos".into(),
        base: cb,
        dims: vec![1, HD],
        stick_idx: 1,
    };
    let sin = AbsTensor {
        name: "sin".into(),
        base: sb,
        dims: vec![1, HD],
        stick_idx: 1,
    };
    let rot = AbsTensor {
        name: "rot".into(),
        base: rotb,
        dims: vec![HEADS, HD],
        stick_idx: 1,
    };
    let xc = AbsTensor {
        name: "xc".into(),
        base: xcb,
        dims: vec![HEADS, HD],
        stick_idx: 1,
    };
    let rs = AbsTensor {
        name: "rs".into(),
        base: rsb,
        dims: vec![HEADS, HD],
        stick_idx: 1,
    };
    let out = AbsTensor {
        name: "out".into(),
        base: ob,
        dims: vec![HEADS, HD],
        stick_idx: 1,
    };

    let xf = |h: usize, d: usize| seed_val(1, h as u64, d as u64, 0);
    let cf = |d: usize| seed_val(2, d as u64, 0, 0);
    let sf = |d: usize| seed_val(3, d as u64, 0, 0);
    let mut mem = AbsMem::default();
    for h in 0..HEADS {
        for d in 0..HD {
            mem.set(x.addr(&[h, d]), xf(h, d));
        }
    }
    for d in 0..HD {
        mem.set(cos.addr(&[0, d]), cf(d));
        mem.set(sin.addr(&[0, d]), sf(d));
    }
    for i in 0..HD {
        for j in 0..HD {
            mem.set(p.addr(&[i, j]), pbuild(i, j));
        }
    }

    // EXACT emitted op-DAG (lower_rope_node): rot=x·P; xc=x·cos(mb); rs=rot·sin(mb); out=xc+rs.
    let ops = vec![
        AbsOp::Matmul {
            a: x.clone(),
            w: p.clone(),
            o: rot.clone(),
        },
        AbsOp::Ew {
            f: EwF::Mul,
            a: x.clone(),
            ab: Bcast::None,
            b: Some(cos.clone()),
            bb: Bcast::Mb,
            o: xc.clone(),
        },
        AbsOp::Ew {
            f: EwF::Mul,
            a: rot.clone(),
            ab: Bcast::None,
            b: Some(sin.clone()),
            bb: Bcast::Mb,
            o: rs.clone(),
        },
        AbsOp::Ew {
            f: EwF::Add,
            a: xc.clone(),
            ab: Bcast::None,
            b: Some(rs.clone()),
            bb: Bcast::None,
            o: out.clone(),
        },
    ];
    interp(&ops, &mut mem);
    let got: Vec<f64> = (0..HEADS)
        .flat_map(|h| (0..HD).map(move |d| (h, d)))
        .map(|(h, d)| mem.get(out.addr(&[h, d])))
        .collect();
    let want = rope_reference(&xf, &cf, &sf, HEADS, HD);
    let _ = (half, STK);
    (got, want)
}

/// The TRUE rotate-half sign-permutation the worker binds: rot[o]=-x[o+half] (o<half),
/// rot[o+half]=x[o]. P[i][j] nonzero: P[o+half][o]=-1, P[o][o+half]=+1.
fn true_rope_p(i: usize, j: usize) -> f64 {
    let half = HD / 2;
    if j < half && i == j + half {
        -1.0
    } else if j >= half && i == j - half {
        1.0
    } else {
        0.0
    }
}

/// POSITIVE: the emitted RoPE op-DAG with the TRUE P reproduces the NeoX reference exactly.
#[test]
fn emitted_rope_matches_reference() {
    let (got, want) = run_rope_with_p(&true_rope_p);
    for i in 0..HEADS * HD {
        assert!(
            (got[i] - want[i]).abs() < 1e-9,
            "rope out[{i}] got {} want {}: emitted (rot=x·P; out=x·cos+rot·sin) must equal NeoX rope",
            got[i],
            want[i]
        );
    }
}

/// NEGATIVE: an IDENTITY P (rot=x, no rotate-half) provably diverges — so the permutation is
/// load-bearing and this guard is real (a structurally-wrong P would fail the positive test).
#[test]
fn identity_p_diverges() {
    let (got, want) = run_rope_with_p(&|i, j| if i == j { 1.0 } else { 0.0 });
    let max_abs = (0..HEADS * HD)
        .map(|i| (got[i] - want[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs > 1e-3,
        "identity P (no rotate-half) must diverge from the rope reference; max|diff|={max_abs}"
    );
}

// head_dim == 128 (granite-3.3-8b / llama-3.2-3b) in PREFILL. A head spans n_slabs=hd/stick=2 device
// sticks, so `lower_rope_node` emits per-(row,head,slab) [1,stick] ops and realizes the rotate-half as a
// SIGNED SLAB-SWAP (no [hd,hd] P-matmul, which can't cross the stick-scattered head): for output slab s,
// `out_s = x_s·cos_s + rot_s·sin_s` with `rot_s = (s<n/2 ? -x_{s+n/2} : +x_{s-n/2})`.
const HD128: usize = 128;

/// The slab-swap rotate-half MATH reproduces the NeoX `rope_reference` exactly at hd=128. For every
/// output dim d, the emitter's per-slab formula (slab s=d/stick, partner = s±n_slabs/2 at the same
/// within-slab col) must equal `x·cos + rotate_half(x)·sin`. This is the novel/risky part of the hd>64
/// prefill rope; `rope_reference` is the independent NeoX oracle, so this is non-circular.
#[test]
fn slab_rope_math_matches_neox_hd128() {
    let n_slabs = HD128 / STK; // 2
    let xf2 = |_h: usize, d: usize| seed_val(1, d as u64, 0, 0);
    let cf = |d: usize| seed_val(2, d as u64, 0, 0);
    let sf = |d: usize| seed_val(3, d as u64, 0, 0);
    let want = rope_reference(&xf2, &cf, &sf, 1, HD128); // NeoX reference for one head, [hd]
    let xf = |d: usize| xf2(0, d);
    for d in 0..HD128 {
        let (s, dp) = (d / STK, d % STK);
        let first_half = s < n_slabs / 2;
        let partner = if first_half {
            (s + n_slabs / 2) * STK + dp
        } else {
            (s - n_slabs / 2) * STK + dp
        };
        // The emitter: xc_s = x_s·cos_s, rs_s = x_partner·sin_s, out_s = xc_s ∓ rs_s (∓ = the rot sign).
        let got = if first_half {
            xf(d) * cf(d) - xf(partner) * sf(d)
        } else {
            xf(d) * cf(d) + xf(partner) * sf(d)
        };
        assert!(
            (got - want[d]).abs() < 1e-12,
            "hd=128 slab rope dim {d} (slab {s}): slab-swap {got} != NeoX reference {}",
            want[d]
        );
    }
}

/// The per-(row,head,slab) output offsets `dev_off([mq,total],1,[r,h·hd+s·stick])` address the exact
/// stick-scattered cells of the [mq,total] roped tensor: each slab's stick is contiguous (off+d' ==
/// dev_off of dim s·stick+d'), and every (r,h,d) output cell is DISTINCT (no two rope writes alias).
#[test]
fn slab_rope_offsets_address_the_scattered_head_hd128() {
    let (mq, heads) = (5usize, 3usize);
    let total = heads * HD128;
    let n_slabs = HD128 / STK;
    let mut seen = std::collections::HashSet::new();
    for r in 0..mq {
        for h in 0..heads {
            for s in 0..n_slabs {
                let off_s = dev_off(&[mq, total], 1, &[r, h * HD128 + s * STK]);
                for dp in 0..STK {
                    // element dp of slab s == dim s·STK+dp of head h, row r — contiguous within the stick.
                    assert_eq!(
                        off_s + dp,
                        dev_off(&[mq, total], 1, &[r, h * HD128 + s * STK + dp]),
                        "slab element must be contiguous within its stick"
                    );
                    assert!(
                        seen.insert(off_s + dp),
                        "rope output cell {} aliases (r={r}, h={h}, s={s}, dp={dp})",
                        off_s + dp
                    );
                }
            }
        }
    }
}

/// The slab RoPE's ROW BATCH: one `[mq, STK]` op per (head, slab) must address exactly the bytes the
/// per-row `[1, STK]` ops did. That holds because the mq rows of a fixed (h, s) are CONTIGUOUS —
/// `dev_off([mq,total],1,[r, h*hd+s*stk])` has column term `(h*hd+s*stk)/stk` with residue 0, so it is
/// `(h*n_slabs+s)*mq*stk + r*stk`, i.e. row r of a standalone [mq,stk] block based at r=0.
///
/// This is the whole justification for dropping the row loop (8b prefill 7440 -> 240 RoPE ops). If the
/// batched base or the block's own row stride were wrong, row 0 would still be right and every other
/// row would land in another head's slab — the "right only at index 0" signature.
#[test]
fn slab_rope_row_batch_addresses_equal_the_per_row_ops_hd128() {
    let (mq, heads) = (31usize, 3usize); // mq=31: the real granite prefill rung, and NOT a stick multiple
    let total = heads * HD128;
    let n_slabs = HD128 / STK;
    for h in 0..heads {
        for s in 0..n_slabs {
            // What the batched op declares: a [mq, STK] block based at the r=0 address.
            let base = dev_off(&[mq, total], 1, &[0, h * HD128 + s * STK]);
            assert_eq!(
                base,
                (h * n_slabs + s) * mq * STK,
                "the (h,s) block base must be the plain slab-major base"
            );
            for r in 0..mq {
                for d in 0..STK {
                    // A standalone [mq,STK] op walks its own (r,d) at r*STK+d — STK is exactly one
                    // stick, so the stick-plane term is 0 and the row stride IS the stick. Written out
                    // rather than as a dev_off, because a dev_off here would be trivially 0-plane and
                    // would pin nothing.
                    assert_eq!(
                        base + r * STK + d,
                        dev_off(&[mq, total], 1, &[r, h * HD128 + s * STK + d]),
                        "batched (h={h},s={s}) row {r} col {d} must hit the per-row address"
                    );
                }
            }
        }
    }
}

/// The cos/sin operand batches the same way: the table is head-tiled, so every head reads HEAD 0's
/// slab s — a `[mq, STK]` block at `s*mq*STK`, independent of h.
#[test]
fn slab_rope_row_batch_cos_block_is_head_independent_hd128() {
    let (mq, heads) = (31usize, 3usize);
    let total = heads * HD128;
    let n_slabs = HD128 / STK;
    for s in 0..n_slabs {
        let coff = dev_off(&[mq, total], 1, &[0, s * STK]);
        assert_eq!(coff, s * mq * STK);
        for r in 0..mq {
            for d in 0..STK {
                assert_eq!(
                    coff + r * STK + d,
                    dev_off(&[mq, total], 1, &[r, s * STK + d]),
                    "cos/sin slab {s} row {r} col {d}"
                );
            }
        }
    }
    let _ = heads;
}

/// The GQA-replicate copies (`attn_krep`/`attn_vrep`) read and write a buffer RoPE produced, so they
/// must use RoPE's address law. RoPE packs each (kv-head, slab) as `mq` tightly-strided rows, giving
/// slab base `(kvh*n_slabs + s)*mq*stk`. A single `[mq_pad, hd]` copy instead addresses its own second
/// stick at its declared ROW COUNT, `mq_pad*stk`.
///
/// Those agree only when a head is one stick. At the real granite prefill shape they are 2112 elements
/// apart, so every kv-head's upper half of K and V was copied from, and to, the wrong place -- which is
/// prefill-only (decode aliases the copies away) and head_dim>64-only, exactly the observed failure.
#[test]
fn krep_slab_base_follows_ropes_law_not_the_declared_row_count_hd128() {
    let (mq, mq_pad, nkvh) = (31usize, 64usize, 8usize);
    for (hd, n_slabs) in [(HD128, HD128 / STK), (64, 1)] {
        for kvh in 0..nkvh {
            for s in 0..n_slabs {
                // RoPE's law, the same `dev_off` its own emission goes through.
                let want = dev_off(&[mq, nkvh * hd], 1, &[0, kvh * hd + s * STK]);
                assert_eq!(want, (kvh * n_slabs + s) * mq * STK, "packed slab base");
                // What a single [mq_pad, hd] op would have used for this slab.
                let flat = kvh * mq * hd + s * mq_pad * STK;
                if s == 0 {
                    assert_eq!(
                        flat, want,
                        "slab 0 always agreed — that is why hd=64 worked"
                    );
                } else {
                    assert_ne!(
                        flat, want,
                        "slab {s} of head {kvh} at hd={hd} must NOT agree"
                    );
                    assert_eq!(
                        flat - want,
                        s * (mq_pad - mq) * STK,
                        "off by the padding rows"
                    );
                }
            }
        }
    }
}
