// SPDX-License-Identifier: Apache-2.0
// The abstract interpreter (`sdsc_abstract`) is gated behind `scratchy-subtile/spyre`; without it this
// whole test crate compiles to nothing (run with `--features spyre`).
#![cfg(feature = "superdsc")]
//! K-CACHE PRODUCER/CONSUMER LAYOUT PROOF (the missing guard for the multi-day attention bug).
//!
//! The SuperDSC decode score matmul reads the resident K cache `kc` as a `[hd, cap]` **cap-sticked
//! KERNEL** (out=cap sticked). For the score to be correct the PRODUCER (the shim's `host_kv_write`
//! scatter) must store each `K[slot][d]` at the matching device offset
//! [`kcache_kt_write_offset`] — NOT the old NATURAL slot-major `slot*hd + d`, which differs whenever
//! `cap > stick` and scrambles the prefix-K.
//!
//! These tests EXECUTE the producer write + the real per-head score→softmax→value op-DAG through the
//! abstract interpreter (over the same `dev_off` device-address model the emitter bakes) and compare to
//! an independent `attn_reference`. They are the NON-CIRCULAR proof behind the emitter's per-element
//! build guard: the positive test passes only because the Kᵀ write feeds the matmul exactly; the
//! negative test shows a natural-layout write provably diverges (so the formula is load-bearing, not a
//! tautology the guard could be hand-tuned to satisfy).
//!
//! Lives in `tests/` (a separate binary using only the public API) so it does not co-compile the
//! crate's feature-gated inline test modules, which do not build on a plain `cargo test` here. Run:
//!   cargo test -p scratchy-subtile --test sdsc_kcache_layout

use scratchy_subtile::sdsc_abstract::*;

/// Build + run the per-head attention op-DAG with K written at `producer_off(slot, d)` (element units,
/// within one head). Returns the on-"card" out `[nqh*hd]` and the independent reference, for comparison.
fn run_attention_with_producer(
    nqh: usize,
    hd: usize,
    cap: usize,
    p: usize,
    producer_off: &dyn Fn(usize, usize) -> usize,
) -> (Vec<f64>, Vec<f64>) {
    // Distinct bases (element units), spaced so no two tensors alias.
    let (qb, kb, vb, mb) = (0usize, 1_000, 200_000, 400_000);
    let (spb, exb, ppb, mxb, smb, rcb, ob) = (
        500_000usize,
        600_000,
        700_000,
        800_000,
        810_000,
        820_000,
        900_000,
    );
    let qs = |h: usize| AbsTensor {
        name: "qs".into(),
        base: qb + h * hd,
        dims: vec![1, hd],
        stick_idx: 1,
    };
    // Consumer view: kc is the [hd, cap] cap-sticked KERNEL, per head at h*hd*cap.
    let kc = |h: usize| AbsTensor {
        name: "kc".into(),
        base: kb + h * hd * cap,
        dims: vec![hd, cap],
        stick_idx: 1,
    };
    let vc = |h: usize| AbsTensor {
        name: "vc".into(),
        base: vb + h * cap * hd,
        dims: vec![cap, hd],
        stick_idx: 1,
    };
    let mask = |h: usize| AbsTensor {
        name: "mask".into(),
        base: mb + h * cap,
        dims: vec![1, cap],
        stick_idx: 1,
    };
    let sp = |h: usize| AbsTensor {
        name: "sp".into(),
        base: spb + h * cap,
        dims: vec![1, cap],
        stick_idx: 1,
    };
    let ex = |h: usize| AbsTensor {
        name: "ex".into(),
        base: exb + h * cap,
        dims: vec![1, cap],
        stick_idx: 1,
    };
    let pp = |h: usize| AbsTensor {
        name: "pp".into(),
        base: ppb + h * cap,
        dims: vec![1, cap],
        stick_idx: 1,
    };
    let mx = |h: usize| AbsTensor {
        name: "mx".into(),
        base: mxb + h,
        dims: vec![1, 1],
        stick_idx: 1,
    };
    let sm = |h: usize| AbsTensor {
        name: "sm".into(),
        base: smb + h,
        dims: vec![1, 1],
        stick_idx: 1,
    };
    let rc = |h: usize| AbsTensor {
        name: "rc".into(),
        base: rcb + h,
        dims: vec![1, 1],
        stick_idx: 1,
    };
    let out = |h: usize| AbsTensor {
        name: "out".into(),
        base: ob + h * hd,
        dims: vec![1, hd],
        stick_idx: 1,
    };

    let kf = |h: usize, s: usize, d: usize| seed_val(2, h as u64, s as u64, d as u64);
    let mut mem = AbsMem::default();
    for h in 0..nqh {
        for d in 0..hd {
            mem.set(qs(h).addr(&[0, d]), seed_val(1, h as u64, d as u64, 0));
        }
        // PRODUCER: write K[h][slot][d] at the per-head base + producer_off(slot, d).
        for s in 0..cap {
            for d in 0..hd {
                mem.set(kc(0).base + h * hd * cap + producer_off(s, d), kf(h, s, d));
                mem.set(
                    vc(h).addr(&[s, d]),
                    seed_val(3, h as u64, s as u64, d as u64),
                );
            }
        }
        for s in 0..cap {
            mem.set(mask(h).addr(&[0, s]), if s <= p { 0.0 } else { -1e30 });
        }
    }
    let mut ops = vec![];
    for h in 0..nqh {
        ops.push(AbsOp::Matmul {
            a: qs(h),
            w: kc(h),
            o: sp(h),
        });
        ops.push(AbsOp::Ew {
            f: EwF::Add,
            a: sp(h),
            ab: Bcast::None,
            b: Some(mask(h)),
            bb: Bcast::None,
            o: sp(h),
        });
        ops.push(AbsOp::Reduce {
            f: RedF::Max,
            x: sp(h),
            o: mx(h),
            scale: 1.0,
        });
        ops.push(AbsOp::Ew {
            f: EwF::Sub,
            a: sp(h),
            ab: Bcast::None,
            b: Some(mx(h)),
            bb: Bcast::Out,
            o: sp(h),
        });
        ops.push(AbsOp::Ew {
            f: EwF::Exp,
            a: sp(h),
            ab: Bcast::None,
            b: None,
            bb: Bcast::None,
            o: ex(h),
        });
        ops.push(AbsOp::Reduce {
            f: RedF::Sum,
            x: ex(h),
            o: sm(h),
            scale: 1.0,
        });
        ops.push(AbsOp::Ew {
            f: EwF::Recip,
            a: sm(h),
            ab: Bcast::None,
            b: None,
            bb: Bcast::None,
            o: rc(h),
        });
        ops.push(AbsOp::Ew {
            f: EwF::Mul,
            a: ex(h),
            ab: Bcast::None,
            b: Some(rc(h)),
            bb: Bcast::Out,
            o: pp(h),
        });
        ops.push(AbsOp::Matmul {
            a: pp(h),
            w: vc(h),
            o: out(h),
        });
    }
    interp(&ops, &mut mem);
    let qsf = |h: usize, d: usize| seed_val(1, h as u64, d as u64, 0);
    let vcf = |h: usize, s: usize, d: usize| seed_val(3, h as u64, s as u64, d as u64);
    let refout = attn_reference(&qsf, &kf, &vcf, nqh, hd, p);
    let got: Vec<f64> = (0..nqh)
        .flat_map(|h| (0..hd).map(move |d| (h, d)))
        .map(|(h, d)| mem.get(out(h).addr(&[0, d])))
        .collect();
    (got, refout)
}

// The REAL device geometry: head_dim == stick (64). `dev_off` pads a 2-D tensor's last dim to a full
// 64-stick, so a head's K/V data spans `dim·64` elements; with `hd=64` the per-head spacing `hd·cap`
// matches that span exactly (no aliasing). A smaller `hd` would make heads overlap in this address
// model — an artifact of the test geometry, not the on-card layout (which always has hd=64=STK). `cap`
// is 4 sticks so `p=130` spans 3 stick-tiles, exercising the cross-stick mapping at full strength.
const NQH: usize = 3;
const HD: usize = 64;
const CAP: usize = 256;
const STK: usize = 64;
const P: usize = 130;

/// POSITIVE: writing K at the shared Kᵀ offset feeds the score matmul exactly → attention == reference.
#[test]
fn kcache_kt_write_feeds_the_score() {
    let (got, want) = run_attention_with_producer(NQH, HD, CAP, P, &|slot, d| {
        kcache_kt_write_offset(slot, d, HD, CAP, STK)
    });
    for i in 0..NQH * HD {
        assert!(
            (got[i] - want[i]).abs() < 1e-9,
            "out[{i}] got {} want {}: the Kᵀ producer write must feed the score matmul exactly",
            got[i],
            want[i]
        );
    }
}

/// The exact invariant the emitter's per-element build guard (`lower_attn_node`) asserts: for every
/// K element the PRODUCER offset (`kcache_kt_write_offset`) equals the CONSUMER offset
/// (`dev_off(&[hd,cap],1,&[d,slot])`). Replicated here so a green model build is proven locally (the
/// guard fires at model-build time on the pod; this is the same loop, on the real device dims).
#[test]
fn build_guard_formulas_agree() {
    for &(hd, cap) in &[
        (64usize, 256usize),
        (64, 192),
        (64, 64),
        (128, 256), // granite-3.3-8b / llama-3.2-3b (2 sticks per head)
        (128, 512),
        (512, 256), // gemma global (8 sticks per head)
    ] {
        for slot in 0..cap {
            for d in 0..hd {
                assert_eq!(
                    kcache_kt_write_offset(slot, d, hd, cap, STK),
                    dev_off(&[hd, cap], 1, &[d, slot]),
                    "guard would FIRE at slot={slot} d={d} (hd={hd}, cap={cap})"
                );
            }
        }
    }
}

/// NEGATIVE: the OLD natural slot-major write (`slot*hd + d`) scrambles the prefix-K the score reads,
/// so the result provably diverges from the reference. This is the bug the build guard now forbids;
/// it proves the layout formula is load-bearing (the positive test is not a tautology).
#[test]
fn natural_slot_major_write_diverges() {
    let (got, want) = run_attention_with_producer(NQH, HD, CAP, P, &|slot, d| slot * HD + d);
    let max_abs_diff = (0..NQH * HD)
        .map(|i| (got[i] - want[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs_diff > 1e-3,
        "natural slot-major write must diverge from the reference (cap={CAP}>stk={STK}); \
         max|diff|={max_abs_diff} — if this is ~0 the test no longer distinguishes the layouts"
    );
}

/// PATH B (host_kv_write nuke): NATURAL on-card store + on-card RESTICKIFY produces exactly the Kᵀ
/// layout the score matmul reads. K is written NATURAL flat `[cap,hd]` (`slot*hd+d`, the expressible
/// dense on-card `[1,hd]`-per-slot copy), then restickified (a TRANSPOSE, mirroring torch-spyre
/// `key.transpose(-2,-1).contiguous()`) into the `[hd,cap]` cap-sticked kernel. After the restickify,
/// the cell the score reads for kernel index (in=d, out=slot) must hold K[slot][d]. Chained with
/// `kcache_kt_write_feeds_the_score` (that Kᵀ cell feeds the reference), this proves the full Path B
/// substitute for the host scatter is correct at the device-address level — the restickify OPSPEC must
/// declare input `[cap,hd]` (flat-natural) and output `[hd,cap]` (cap-sticked), NOT the same dim order.
#[test]
fn natural_store_plus_restickify_produces_kt_layout() {
    let (hd, cap) = (HD, CAP);
    let kf = |s: usize, d: usize| seed_val(2, 0, s as u64, d as u64);
    let mut mem = AbsMem::default();
    // NATURAL store: kc_nat `[cap,hd]` FLAT (stick_idx 0 ⇒ row-major `slot*hd+d`), the expressible
    // on-card dense write. kct: the `[hd,cap]` cap-sticked (stick_idx 1) kernel the score reads.
    let kc_nat = AbsTensor {
        name: "kc_nat".into(),
        base: 0,
        dims: vec![cap, hd],
        stick_idx: 0,
    };
    let kct = AbsTensor {
        name: "kct".into(),
        base: 1_000_000,
        dims: vec![hd, cap],
        stick_idx: 1,
    };
    for s in 0..cap {
        for d in 0..hd {
            // The natural store IS flat row-major — the address the on-card dense copy writes.
            assert_eq!(
                kc_nat.addr(&[s, d]),
                s * hd + d,
                "natural store must be flat row-major slot*hd+d"
            );
            mem.set(kc_nat.addr(&[s, d]), kf(s, d));
        }
    }
    interp(
        &[AbsOp::Restickify {
            x: kc_nat.clone(),
            o: kct.clone(),
        }],
        &mut mem,
    );
    // After the transpose, the kt cell the score matmul reads for (in=d, out=slot) holds K[slot][d].
    for s in 0..cap {
        for d in 0..hd {
            let got = mem.get(kct.addr(&[d, s]));
            assert!(
                (got - kf(s, d)).abs() < 1e-12,
                "restickify must place K[{s}][{d}] at the Kᵀ cell the score reads; got {got} want {}",
                kf(s, d)
            );
            // And that cell IS the shared build-guard offset — the restickify output == score read.
            assert_eq!(
                kct.addr(&[d, s]) - kct.base,
                kcache_kt_write_offset(s, d, hd, cap, STK)
            );
        }
    }
}

/// PATH B (V side): NATURAL on-card store + on-card RE-STICK produces exactly the hd-sticked layout
/// the value bmm reads. UNLIKE K this is NOT a transpose — V keeps its `[cap,hd]` logical shape, only
/// the stick axis moves flat→hd. V is written NATURAL flat (`slot*hd+d`), then re-stuck into the
/// `[cap,hd]` hd-sticked KERNEL the value bmm reads (`vcache_write_offset` = `dev_off([cap,hd],1,·)`).
/// After the re-stick the cell the bmm reads for kernel index (k=slot, n=d) must hold V[slot][d].
/// Chained with the value matmul in `run_attention_with_producer` (which reads V hd-sticked), this
/// proves the Path B V substitute. The re-stick is modeled as an identity-logical copy across the two
/// layouts (the interpreter's `Restickify` is a TRANSPOSE, so it does NOT model V — a real re-stick op).
#[test]
fn natural_store_plus_restick_produces_v_hdstick_layout() {
    let (hd, cap) = (HD, CAP);
    let vf = |s: usize, d: usize| seed_val(3, 0, s as u64, d as u64);
    let mut mem = AbsMem::default();
    // NATURAL store: vc_nat `[cap,hd]` FLAT (stick_idx 0 ⇒ `slot*hd+d`). vct: `[cap,hd]` hd-sticked
    // (stick_idx 1 ⇒ `dev_off([cap,hd],1,·)` = vcache_write_offset), the layout the value bmm reads.
    let vc_nat = AbsTensor {
        name: "vc_nat".into(),
        base: 0,
        dims: vec![cap, hd],
        stick_idx: 0,
    };
    let vct = AbsTensor {
        name: "vct".into(),
        base: 2_000_000,
        dims: vec![cap, hd],
        stick_idx: 1,
    };
    for s in 0..cap {
        for d in 0..hd {
            assert_eq!(
                vc_nat.addr(&[s, d]),
                s * hd + d,
                "natural V store must be flat row-major slot*hd+d"
            );
            mem.set(vc_nat.addr(&[s, d]), vf(s, d));
        }
    }
    // RE-STICK: identity-logical copy V[slot][d] from the flat cell to the hd-sticked cell.
    for s in 0..cap {
        for d in 0..hd {
            let v = mem.get(vc_nat.addr(&[s, d]));
            mem.set(vct.addr(&[s, d]), v);
        }
    }
    // After the re-stick, the hd-sticked cell the value bmm reads for (k=slot, n=d) holds V[slot][d].
    for s in 0..cap {
        for d in 0..hd {
            let got = mem.get(vct.addr(&[s, d]));
            assert!(
                (got - vf(s, d)).abs() < 1e-12,
                "re-stick must place V[{s}][{d}] at the hd-sticked cell the value bmm reads; got {got} want {}",
                vf(s, d)
            );
            assert_eq!(
                vct.addr(&[s, d]) - vct.base,
                vcache_write_offset(s, d, hd, cap, STK)
            );
        }
    }
}

// head_dim == 128 (granite-3.3-8b, llama-3.2-3b): 2 fp16 sticks per head. head_dim > stick is the case
// the SLAB-MAJOR store + stick-last K restickify enable (previously a build Err in restickify_kt_opspec_2d).
const HD128: usize = 128;

/// The keystone identity that makes hd>stick work WITHOUT a V restickify and with the DEPLOYED stick-last
/// K restickify input: at ANY head_dim the SLAB-MAJOR on-card store (`vcache_write_offset`) EQUALS the
/// stick-last kernel read `dev_off([cap,hd],1,·)` — which is BOTH the K restickify INPUT and the V
/// value-bmm read. Checked at hd = 64 (granite-3.3-2b), 128 (granite-3.3-8b / llama-3.2-3b), 512 (gemma
/// global) over a multi-stick cap. If these diverged, the stick-last restickify would read scrambled K.
#[test]
fn slab_store_equals_stick_last_kernel_read_any_hd() {
    for &hd in &[64usize, HD128, 512] {
        let cap = 256usize;
        for slot in 0..cap {
            for d in 0..hd {
                assert_eq!(
                    vcache_write_offset(slot, d, hd, cap, STK),
                    dev_off(&[cap, hd], 1, &[slot, d]),
                    "hd={hd} slot={slot} d={d}: slab-major store != stick-last kernel read"
                );
            }
        }
    }
}

/// hd=128 PATH B (the head_dim > stick case this change enables): the SLAB-MAJOR on-card store
/// (`dev_off([cap,hd],1,·)`, the `[hd/64,cap,64]` stick-last layout the cachewr writes) + the on-card K
/// restickify produces exactly the Kᵀ layout the score matmul reads. UNLIKE the hd==64 flat store, the
/// input here is stick-LAST (stick_idx 1) — verbatim the deployed `restickify_kt_opspec_2d` INPUT — so
/// this proves the stick-last input reads the resident cache correctly at hd=128 (2 sticks per head),
/// then transposes into the score kernel. Mirrors `natural_store_plus_restickify_produces_kt_layout`.
#[test]
fn slab_store_plus_restickify_produces_kt_layout_hd128() {
    let (hd, cap) = (HD128, CAP);
    let kf = |s: usize, d: usize| seed_val(2, 0, s as u64, d as u64);
    let mut mem = AbsMem::default();
    // SLAB-MAJOR store: kc_slab `[cap,hd]` stick-LAST (stick_idx 1 ⇒ dev_off([cap,hd],1,·)) — the layout
    // the on-card cachewr writes and the restickify INPUT reads. kct: `[hd,cap]` cap-sticked score kernel.
    let kc_slab = AbsTensor {
        name: "kc_slab".into(),
        base: 0,
        dims: vec![cap, hd],
        stick_idx: 1,
    };
    let kct = AbsTensor {
        name: "kct".into(),
        base: 4_000_000,
        dims: vec![hd, cap],
        stick_idx: 1,
    };
    for s in 0..cap {
        for d in 0..hd {
            // The slab-major store IS dev_off([cap,hd],1,·) = vcache_write_offset (the restickify input).
            assert_eq!(
                kc_slab.addr(&[s, d]),
                vcache_write_offset(s, d, hd, cap, STK),
                "slab-major store must be dev_off([cap,hd],1,·) = vcache_write_offset"
            );
            mem.set(kc_slab.addr(&[s, d]), kf(s, d));
        }
    }
    interp(
        &[AbsOp::Restickify {
            x: kc_slab.clone(),
            o: kct.clone(),
        }],
        &mut mem,
    );
    // After the transpose, the kt cell the score matmul reads for (in=d, out=slot) holds K[slot][d].
    for s in 0..cap {
        for d in 0..hd {
            let got = mem.get(kct.addr(&[d, s]));
            assert!(
                (got - kf(s, d)).abs() < 1e-12,
                "hd=128 restickify must place K[{s}][{d}] at the Kᵀ cell the score reads; got {got} want {}",
                kf(s, d)
            );
            assert_eq!(
                kct.addr(&[d, s]) - kct.base,
                kcache_kt_write_offset(s, d, hd, cap, STK)
            );
        }
    }
}

/// hd=128 PREFILL slab-major layout: the re-stick (kh/vh) and cachewr (kc/vc) place each element at the
/// exact cell its consumer reads. The K restickify + value-bmm read kh/vh as the stick-last `[mqp,hd]`
/// kernel (`dev_off([mqp,hd],1,·)`); the DECODE restickify + value-bmm read kc/vc as the stick-last
/// `[cap,hd]` kernel (`vcache_write_offset`). Both slab-major writes must equal those, and be injective
/// (no element aliases another). Mirrors the emitter's build-time LINK A/B guards on the real device dims.
#[test]
fn prefill_slab_major_layout_hd128() {
    let (mqp, cap, mqu) = (64usize, 256usize, 8usize);
    // LINK A: kh/vh re-stick write == restickify/value-bmm stick-last read; injective over [mqp,hd].
    let mut seen_kh = std::collections::HashSet::new();
    for r in 0..mqp {
        for d in 0..HD128 {
            let (slab, dp) = (d / STK, d % STK);
            let write = slab * mqp * STK + r * STK + dp;
            assert_eq!(
                write,
                dev_off(&[mqp, HD128], 1, &[r, d]),
                "kh/vh LINK A r={r} d={d}"
            );
            assert!(
                seen_kh.insert(write),
                "kh/vh slab write {write} aliases (r={r}, d={d})"
            );
        }
    }
    // LINK B: kc/vc cachewr write == decode restickify read (vcache_write_offset); injective over slots.
    let mut seen_kc = std::collections::HashSet::new();
    for s in 0..mqu {
        for d in 0..HD128 {
            let (slab, dp) = (d / STK, d % STK);
            let write = slab * cap * STK + s * STK + dp;
            assert_eq!(
                write,
                vcache_write_offset(s, d, HD128, cap, STK),
                "kc/vc LINK B s={s} d={d}"
            );
            assert!(
                seen_kc.insert(write),
                "kc/vc slab write {write} aliases (s={s}, d={d})"
            );
        }
    }
}

/// hd=128 END-TO-END: the full per-head attention DAG (score → mask → softmax → value) is correct when K
/// is written in the Kᵀ `[hd,cap]` cap-sticked kernel and V in the hd-sticked `[cap,hd]` kernel — the
/// consumer layouts the slab-major store + K restickify realize. Exercises the multi-stick (hd/64=2)
/// head_dim end-to-end against the independent `attn_reference`, the strongest hd=128 correctness check.
#[test]
fn attention_correct_at_hd128() {
    let (got, want) = run_attention_with_producer(NQH, HD128, CAP, P, &|slot, d| {
        kcache_kt_write_offset(slot, d, HD128, CAP, STK)
    });
    for i in 0..NQH * HD128 {
        assert!(
            (got[i] - want[i]).abs() < 1e-9,
            "hd=128 out[{i}] got {} want {}: the Kᵀ producer write must feed the score matmul exactly",
            got[i],
            want[i]
        );
    }
}
