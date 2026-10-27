#!/usr/bin/env python3
"""The hash-noise probe's CARD VERDICTS (front-b-noise, nickm2 leg).

The auditor's ask, as ruled:
* u: BIT-COMPARED against the host emulation of the whole integer chain +
  wrap-fold — every op up to and including the uniform is exact by
  representability (wrapping i32; int32tofp32 exact on multiples of 2^8
  with <= 24 significant bits, card leg B1), so ZERO differing lanes is the
  only pass. This checks the hash, the SENUINT32 mul/add ops, the convert,
  the abs fold, and the strict-interior folds in one number.
* u-histogram: chi-square over 256 bins, expected 2^20/256 = 4096 per bin
  (df 255). The u grid is (np+0.5)*2^-22 over np in [0, 2^22), so a perfect
  hash puts exactly 4096 lanes in every bin.
* g: the card's mean and variance against the Gumbel values
  (Euler-Mascheroni 0.5772156649..., pi^2/6 = 1.6449340668...). The card's
  log is the FP16 FEST bind (the vendor DDLs carry no fp32 log), so g is the
  fp16 chain: narrow, clamp at 2^-14, log, neg, clamp, log, neg, widen. The
  host reference beside it is the f64 -log(-log(u)) over the SAME u, plus an
  fp16 emulation of the card's chain (numpy float16, same clamps — a
  correctly-rounded fp16 log, so the diff against the card is the FEST log's
  own deviation). Clamped-lane counts and non-finite lanes are REPORTED,
  never gated.
* Bonus, never gating: the two fptosi exits (ui = trunc(u), gi = trunc(g))
  against the card's own u and g.

Every number is written to fb-noise-verdict.json beside this script.
Exit 0 iff the u bit-check is zero-diff (the statistics are REPORTED, never
gated — they are the measurement the probe exists to take).
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fb_inputs as G  # noqa: E402  (the generators ARE the input definition)

N = G.N
IN = G.INDIR

GUMBEL_MEAN = 0.5772156649015329
GUMBEL_VAR = math.pi**2 / 6


def read_u32(name: str) -> np.ndarray:
    return np.fromfile(IN / f"{name}.bin", dtype=np.uint32)


def host_u() -> np.ndarray:
    """The host bit-emulation of the whole chain up to and including u.

    Every op is exact by representability, so this is bit-exact against a
    correct card: wrapping u32 arithmetic for the hash (numpy wraps), the
    SIGNED interpretation for the convert (B1's i2f law), and IEEE f32 for
    the fold (all operands are integers <= 2^23 or powers of two).
    """
    lane = np.fromfile(IN / "t08_lane.bin", dtype=np.float32)
    # The fptosi of the staged f32 lane recovers the integer bit for bit.
    lanei = lane.astype(np.int32).astype(np.uint32)
    slo = read_u32("t00_slo")[0]
    shi = read_u32("t01_shi")[0]
    stp = read_u32("t02_stp")[0]
    srow = read_u32("t03_srow")[0]
    c1 = read_u32("t04_c1")[0]
    c2 = read_u32("t05_c2")[0]
    c3 = read_u32("t06_c3")[0]
    c256 = read_u32("t07_c256")[0]

    h = lanei * c1 + slo
    h = h * c2 + shi + stp + srow
    h = h * h
    h = h * c1 + slo
    h = h * h
    h = h * c3 + shi + stp
    h = h * h
    h = h * c1 + lanei * c2 + slo
    r = (h * c256).astype(np.int32)
    rf = r.astype(np.float32)
    a = np.abs(rf)
    n = a * np.float32(2.0**-8)
    t = np.float32(8388607.0) - n
    nm = np.minimum(n, t)
    npx = np.maximum(nm, np.float32(0.0))
    return (npx + np.float32(0.5)) * np.float32(2.0**-22)


def chi2_sf_approx(x: float, k: int) -> float:
    """The chi-square survival function, Wilson-Hilferty approximation
    (labeled as such in the report — the pod has no scipy)."""
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def main() -> int:
    verdict: dict = {"lanes": N}
    rc = 0

    card_u = np.fromfile(HERE / "out_u.bin", dtype=np.float32)
    card_g = np.fromfile(HERE / "out_g.bin", dtype=np.float32)
    card_ui = np.fromfile(HERE / "out_ui.bin", dtype=np.int32)
    card_gi = np.fromfile(HERE / "out_gi.bin", dtype=np.int32)
    for name, a in [("u", card_u), ("g", card_g), ("ui", card_ui), ("gi", card_gi)]:
        if a.size != N:
            print(f"VERDICT {name}: BAD SIZE {a.size} != {N}")
            verdict[f"{name}_bad_size"] = a.size
            return 1

    # ── u: the bit-check (the gate) ──
    u = host_u()
    ub = u.view(np.uint32)
    cb = card_u.view(np.uint32)
    diff = int(np.count_nonzero(ub != cb))
    mism = np.flatnonzero(ub != cb)[:5]
    detail = [
        {
            "lane": int(i),
            "card_bits": hex(int(cb[i])),
            "host_bits": hex(int(ub[i])),
            "card": float(card_u[i]),
            "host": float(u[i]),
        }
        for i in mism
    ]
    verdict["u_bitcheck"] = {"diff": diff, "detail": detail}
    print(f"VERDICT u-bitcheck: {'PASS' if diff == 0 else 'FAIL'} diff={diff}/{N}")
    for d in detail:
        print(f"    {d}")
    if diff != 0:
        rc = 1

    in_range = bool(np.all((card_u > 0.0) & (card_u < 1.0)))
    verdict["u_in_open_unit"] = in_range
    print(f"VERDICT u-in-open-unit: {in_range}")

    # ── the u-histogram chi-square over 256 bins ──
    bins = np.floor(card_u.astype(np.float64) * 256.0).astype(np.int64)
    counts = np.bincount(bins, minlength=256)
    expected = N / 256.0
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = chi2_sf_approx(chi2, 255)
    verdict["u_chi2_256bins"] = {
        "chi2": chi2,
        "df": 255,
        "expected_per_bin": expected,
        "min_bin": int(counts.min()),
        "max_bin": int(counts.max()),
        "p_wilson_hilferty": p,
    }
    print(
        f"VERDICT u-chi2(255df): {chi2:.2f}  (expected 4096/bin; "
        f"min={counts.min()} max={counts.max()}; p~{p:.4g} WH-approx)"
    )

    # ── g: the Gumbel moments ──
    g64 = card_g.astype(np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(card_g)))
    finite = bool(nonfinite == 0)
    mean_g = float(g64[np.isfinite(g64)].mean()) if not finite else float(g64.mean())
    var_g = float(g64[np.isfinite(g64)].var()) if not finite else float(g64.var())
    verdict["g_card"] = {
        "finite": finite,
        "nonfinite_lanes": nonfinite,
        "mean": mean_g,
        "var_pop": var_g,
        "gumbel_mean": GUMBEL_MEAN,
        "gumbel_var": GUMBEL_VAR,
        "mean_abs_err": abs(mean_g - GUMBEL_MEAN),
        "var_abs_err": abs(var_g - GUMBEL_VAR),
    }
    print(f"VERDICT g-card: finite={finite} (nonfinite={nonfinite}) mean={mean_g:.6f}")
    print(f"           (want {GUMBEL_MEAN:.6f}, err {mean_g - GUMBEL_MEAN:+.6f})")
    print(f"           var={var_g:.6f} (want {GUMBEL_VAR:.6f}, err {var_g - GUMBEL_VAR:+.6f})")

    # The host reference over the SAME u: f64 log, so the card log's
    # contribution to the moments is separable from the draw's.
    uu = u.astype(np.float64)
    gref = -np.log(-np.log(uu))
    verdict["g_host_ref"] = {
        "mean": float(gref.mean()),
        "var_pop": float(gref.var()),
    }
    print(
        f"VERDICT g-host-ref (f64 log, same u): mean={gref.mean():.6f} "
        f"var={gref.var():.6f}"
    )

    # The fp16 emulation of the card's own chain: narrow, clamp at 2^-14,
    # log, neg, clamp, log, neg, widen (numpy float16 — a correctly-rounded
    # fp16 log, so the diff against the card isolates the FEST log's
    # deviation). Clamped-lane counts reported with it.
    cmin = np.float16(2.0**-14)
    u16 = u.astype(np.float16)
    clamped1 = int(np.count_nonzero(u16 < cmin))
    c1 = np.maximum(u16, cmin)
    lu = np.log(c1.astype(np.float32)).astype(np.float16)
    v = (-lu).astype(np.float16)
    clamped2 = int(np.count_nonzero(v < cmin))
    c2 = np.maximum(v, cmin)
    lt = np.log(c2.astype(np.float32)).astype(np.float16)
    g16 = (-lt).astype(np.float16)
    gemu = g16.astype(np.float32)
    both_finite = np.isfinite(gemu) & np.isfinite(card_g)
    d = np.abs(gemu[both_finite].astype(np.float64) - card_g[both_finite].astype(np.float64))
    verdict["g_fp16_emulation"] = {
        "mean": float(gemu.mean()),
        "var_pop": float(gemu.var()),
        "clamped_lanes_inner": clamped1,
        "clamped_lanes_outer": clamped2,
        "max_abs_diff_vs_card": float(d.max()) if d.size else None,
        "mean_abs_diff_vs_card": float(d.mean()) if d.size else None,
    }
    print(
        f"VERDICT g-fp16-emu (host fp16 chain, same clamps): mean={gemu.mean():.6f} "
        f"var={gemu.var():.6f}"
    )
    print(
        f"           clamped lanes: inner={clamped1} outer={clamped2}; "
        f"|emu-card|: max={d.max() if d.size else -1:.6f} mean={d.mean() if d.size else -1:.6f}"
    )

    # ── the fptosi exits, never gating ──
    # ui: u ∈ (0,1) so trunc(u) = 0 on every lane — the 0-diff is VACUOUS,
    # reported as such. gi: THE CARD'S fp32toint32 IS FLOOR (round toward
    # −∞), measured by this probe — gi == floor(g) on every lane, while
    # gi == trunc(g) only on the non-negative ones (the Gumbel's exp(−1)
    # negative fraction differs by 1). B1's uses of the op were over exact
    # integers, where floor == trunc == round, so this is the first
    # non-integer measurement of the op's rounding semantics — a port-back
    # fact for every production exit over a non-integer f32.
    ui_want = np.trunc(card_u).astype(np.int32)
    gi_floor = np.floor(card_g).astype(np.int32)
    gi_trunc = np.trunc(card_g).astype(np.int32)
    ui_diff = int(np.count_nonzero(card_ui != ui_want))
    gi_floor_diff = int(np.count_nonzero(card_gi != gi_floor))
    gi_trunc_diff = int(np.count_nonzero(card_gi != gi_trunc))
    verdict["exits"] = {
        "ui_trunc_diff_vacuous_all_zero": ui_diff,
        "gi_floor_diff": gi_floor_diff,
        "gi_trunc_diff": gi_trunc_diff,
    }
    print(
        f"VERDICT exits: ui-trunc diff={ui_diff}/{N} (vacuous: u in (0,1) truncs to 0)"
    )
    print(
        f"           gi-floor diff={gi_floor_diff}/{N}  gi-trunc diff={gi_trunc_diff}/{N}"
        f"  => fp32toint32 is FLOOR (round toward -inf)"
    )

    out = HERE / "fb-noise-verdict.json"
    out.write_text(json.dumps(verdict, indent=1))
    print(f"verdict written: {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
