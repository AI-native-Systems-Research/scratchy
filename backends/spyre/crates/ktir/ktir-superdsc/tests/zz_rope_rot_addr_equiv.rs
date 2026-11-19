// SPDX-License-Identifier: Apache-2.0
//! DO THE ROTATE'S FACTORS COMPUTE THE ROTATE `P` STATES? Element for element, at every head dim the
//! workspace's ropes run — the rule this file exists to obey: the factors the lowering emits
//! (`RotateHalf::of`, read off `rope_p_entry`) are checked against the one statement of the pairing
//! and sign, `x · P`, never against a restatement of it.
//!
//! The lowering emits rotate-half as its factors: whole-stick moves under one sign each (a head whose
//! half is whole sticks — 128, 256, 512) or ONE in-stick lane permutation against P's one-stick block
//! (a one-stick head — 64). Either factorization reproducing `x · P` is the whole premise of a rope
//! with no `[hd, hd]` contraction; a transposed pairing would silently compute `-rotate_half` (P is NOT
//! symmetric — its transpose is the inverse rotation), which is fluent-but-wrong output, not a crash.

use ktir_superdsc::addr::{Feat, Idx, Nest};
use ktir_superdsc::sdsc_abstract::{HeadWidth, Lanes, RotSign, RotateHalf, rope_p_entry};
use ktir_superdsc::superdsc_opspec::Df;

const STK: u32 = 64;
const HEADS: u32 = 32;
const MQ: u32 = 21;

/// `x · P` for one head of `hd`: `rot[o] = Σ_inn x[inn] · P[inn][o]` — the worker's own fill law
/// (`stage_kernel_table(hd, |inn, o| rope_p_entry(hd, inn, o))`), contracted whole.
fn by_p(hd: u32, x: &[i64]) -> Vec<i64> {
    (0..hd)
        .map(|o| {
            (0..hd)
                .map(|inn| {
                    x[inn as usize] * i64::from(rope_p_entry(hd as usize, inn as usize, o as usize))
                })
                .sum()
        })
        .collect()
}

/// The factors applied to one head of `hd`: each stick move copies its sticks lane-for-lane under
/// its sign; a lane factor contracts every stick against P's stick-0 `[lanes, lanes]` block — the
/// block the lowering's one-stick kernel reads out of the staged table.
fn by_factors(hd: u32, x: &[i64]) -> Vec<i64> {
    let r = RotateHalf::of(HeadWidth::of_whole_sticks(hd), Lanes::FP16)
        .unwrap_or_else(|| panic!("hd {hd} has a factored form"));
    let mut rot = vec![i64::MIN; hd as usize];
    for m in r.stick_moves() {
        let sign = match m.sign() {
            RotSign::Plus => 1,
            RotSign::Minus => -1,
        };
        for k in 0..m.sticks() {
            for j in 0..STK {
                let (o, inn) = ((m.dst() + k) * STK + j, (m.src() + k) * STK + j);
                rot[o as usize] = sign * x[inn as usize];
            }
        }
    }
    if let Some(lane) = r.lane() {
        let w = lane.width().get();
        for o in 0..hd {
            let (s, j) = (o / w, o % w);
            rot[o as usize] = (0..w)
                .map(|i| {
                    x[(s * w + i) as usize]
                        * i64::from(rope_p_entry(hd as usize, i as usize, j as usize))
                })
                .sum();
        }
    }
    rot
}

/// ⭐ THE FACTORS COMPUTE `x · P` at every head dim the ropes run, and the non-identity factor is
/// the expected one: the lane factor alone at one stick, two stick moves (one negated) above it.
/// Every output column is written (a move set that left one unwritten shows `i64::MIN`).
#[test]
fn the_factors_compute_the_p_rotate_at_every_head_dim() {
    // (head dim, stick moves, lane factor) — the expectation table, one row per head dim in use.
    for (hd, moves, lane) in [
        (64u32, 0usize, true),
        (128, 2, false),
        (256, 2, false),
        (512, 2, false),
    ] {
        let r = RotateHalf::of(HeadWidth::of_whole_sticks(hd), Lanes::FP16)
            .unwrap_or_else(|| panic!("hd {hd} has a factored form"));
        assert_eq!(r.stick_moves().len(), moves, "hd {hd}: stick moves");
        assert_eq!(r.lane().is_some(), lane, "hd {hd}: lane factor");
        // x[d] distinguishable per head and per dim, so a wrong stick or lane shows up.
        for h in [0u32, 1, HEADS - 1] {
            let x: Vec<i64> = (0..hd).map(|d| i64::from(h * hd + d) + 1).collect();
            assert_eq!(by_factors(hd, &x), by_p(hd, &x), "hd {hd} head {h}");
        }
    }
}

/// ⭐ THE SIGNED TABLE CARRIES THE ROTATE'S SIGN — the 4-op form's whole premise. For a rotate of
/// stick moves, `rotate_half(x) · sin` (the `x · P` contraction times the plain table) equals each
/// move's plain lane-for-lane copy of `x` times `sin'` (`SinTable::sign_heads` of the plain table),
/// column for column; a lane factor's table stays plain (its P carries the sign).
#[test]
fn the_signed_table_carries_the_rotates_sign() {
    for hd in [128u32, 256, 512] {
        let r = RotateHalf::of(HeadWidth::of_whole_sticks(hd), Lanes::FP16)
            .unwrap_or_else(|| panic!("hd {hd} has a factored form"));
        let table = r.sin_table();
        assert!(
            table.is_signed(),
            "hd {hd}: stick moves read the signed table"
        );
        let x: Vec<i64> = (0..hd).map(|d| i64::from(d) + 1).collect();
        let sin: Vec<f32> = (0..hd).map(|d| (d % 13) as f32 + 1.0).collect();
        let mut signed = sin.clone();
        table.sign_heads(&mut signed);
        let rot = by_p(hd, &x);
        for m in r.stick_moves() {
            for k in 0..m.sticks() {
                for j in 0..STK {
                    let o = ((m.dst() + k) * STK + j) as usize;
                    let inn = ((m.src() + k) * STK + j) as usize;
                    assert_eq!(
                        rot[o] as f32 * sin[o],
                        x[inn] as f32 * signed[o],
                        "hd {hd} column {o}"
                    );
                }
            }
        }
    }
    let lane = RotateHalf::of(HeadWidth::of_whole_sticks(STK), Lanes::FP16)
        .unwrap_or_else(|| panic!("hd {STK} has a factored form"));
    assert!(
        !lane.sin_table().is_signed(),
        "a lane factor's P carries the sign; its table is plain"
    );
}

/// A head whose half straddles a stick — an odd stick count above one — keeps neither lanes nor
/// sticks in place: no factored form, so its rope's bake refuses it by name rather than emitting a
/// rotate that computes something else.
#[test]
fn a_half_that_straddles_a_stick_has_no_factored_form() {
    assert!(RotateHalf::of(HeadWidth::of_whole_sticks(192), Lanes::FP16).is_none());
    assert!(!RotateHalf::reads_lane_table(192));
    assert!(RotateHalf::reads_lane_table(64));
    assert!(!RotateHalf::reads_lane_table(128));
}

/// THE BANDS STEP PLANES AND HEADS THROUGH THE TOKEN STREAM, differenced through the nest: in the
/// `[mq, heads·hd]` stream a stick move's band starts one stick PLANE (`mq · 64`) per stick into its
/// head, and the next head's band is `mq · hd` on — the pitch the band op states on `out` (`hd`
/// columns, i.e. `mq · hd` elements of the stick-major plane).
#[test]
fn the_bands_step_planes_and_heads_through_the_token_stream() {
    const HD: u32 = 128;
    let nest = Nest::new(&["row", "feat"], &[MQ, HEADS * HD], Df::Fp16);
    let at = |h: u32, s: u32| {
        nest.view()
            .at(Idx::<Feat>::n(h * HD + s * STK))
            .dev()
            .into_raw_elems()
    };
    for h in [0u32, 1, HEADS - 1] {
        assert_eq!(
            at(h, 1) - at(h, 0),
            MQ * STK,
            "h={h}: one stick is one stick plane"
        );
    }
    for s in 0..HD / STK {
        assert_eq!(at(1, s) - at(0, s), MQ * HD, "s={s}: one head is mq*hd");
    }
    // And the base the band op is given for (head 0, stick s) is exactly `s` planes in.
    for s in 0..HD / STK {
        assert_eq!(at(0, s), s * MQ * STK, "s={s}: the op's own base");
    }
}
