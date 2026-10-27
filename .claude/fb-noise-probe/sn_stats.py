#!/usr/bin/env python3
"""The sample-noise probe's VERDICT (front-b-noise, nickm2 leg — auditor
ruling (a): g and score off the card).

Reads the six outputs back (UN-SCATTERED from the device's stick-major
layout), runs the host reference (`sn_inputs` — the generators ARE the input
definition), and rules:

  1. `u` BIT-EXACT per row and lane — the whole integer chain plus every
     `[rows, 1]` staging. HARD GATE. On any mismatch the verdict NAMES the
     first wrong value: (row, lane, card bits, host bits).
  2. rows differ — the u streams of two rows with different seeds are
     different arrays. HARD GATE.
  3. `g` LIVE per row — ≥ 2 distinct values and variance > 0.5 (the Gumbel
     variance is 1.6449; the Finding-1 signature is a lane-CONSTANT noise,
     variance 0). HARD GATE.
  4. `g` moments per row — mean within 0.5772 ± 0.05, variance within
     1.6449 ± 0.5 (generous: σ_mean ≈ 0.005 at 65536 lanes). HARD GATE.
  5. `score`'s argmax MOVED — at least one row where argmax(score) differs
     from the logits row's own argmax over the real lanes (the near-tie
     peaks are staged so a live noise moves most rows; a constant noise can
     move none). HARD GATE on b8 (8 rows — all keeping the argmax by chance
     is ~1e-5); reported on b2 (each row keeps its argmax ~24% by chance
     under a live noise — a 2-row hard gate would false-fail ~6% of the time).
  6. `g`/`score` vs the host fp16/f32 emulation — the FEST log is the one
     unemulable op (~2^-11 relative per log), so these are REPORTED (max and
     mean |diff| over the real lanes), never gated bit-for-bit.

The numbers land in sn-verdict-<case>.json beside this script. Exit 0 iff
every hard gate holds.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sn_inputs  # noqa: E402

GUMBEL_MEAN = 0.5772156649015329
GUMBEL_VAR = 1.6449340668482264


def read_out(case: str, name: str, rows: int, dtype=np.float32) -> np.ndarray:
    """One output file, un-scattered from the device's stick-major layout
    back to `[rows, PAD_V]` row-major (f32 and i32 outs share the 32-word
    stick granule)."""
    flat = np.fromfile(HERE / f"out_sn_{case}_{name}.bin", dtype=dtype)
    return sn_inputs.stick_unscatter(flat, rows, sn_inputs.PAD_V, sn_inputs.F32_STICK)


def first_mismatch(card: np.ndarray, host: np.ndarray):
    """The first (row, lane, card bits, host bits) where two f32 tiles
    differ, or None."""
    cb = card.view(np.uint32)
    hb = host.view(np.uint32)
    bad = cb != hb
    if not bad.any():
        return None
    r, j = np.argwhere(bad)[0]
    return {
        "row": int(r),
        "lane": int(j),
        "card_bits": f"0x{int(cb[r, j]):08X}",
        "host_bits": f"0x{int(hb[r, j]):08X}",
        "card_value": float(card[r, j]),
        "host_value": float(host[r, j]),
    }


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else "b8"
    rows = sn_inputs.CASES[case]
    sw = sn_inputs.seed_words(rows)
    x = sn_inputs.logits(rows)
    v = sn_inputs.V

    # ── gate 1: u bit-exact, per row and lane ──
    u_card = read_out(case, "u", rows)
    u_host = np.stack([
        sn_inputs.emulate_u(int(sw["slo"][r]), int(sw["shi"][r]),
                            int(sw["stp"][r]), int(sw["srow"][r]))
        for r in range(rows)
    ])
    u_first = first_mismatch(u_card, u_host)
    u_diff = int((u_card.view(np.uint32) != u_host.view(np.uint32)).sum())

    # ── gate 2: rows differ ──
    rows_differ = all(
        not np.array_equal(u_host[0].view(np.uint32), u_host[r].view(np.uint32))
        for r in range(1, rows)
    )

    # ── gates 3+4: g live, moments near the Gumbel values ──
    g_card = read_out(case, "g", rows)
    g_emu = np.stack([sn_inputs.emulate_g(u_host[r]) for r in range(rows)])
    g_rows = []
    for r in range(rows):
        real = g_card[r, :v]
        g_rows.append({
            "mean": float(real.mean()),
            "var_pop": float(real.var()),
            "distinct": int(np.unique(real).size),
        })
    g_live = all(gr["distinct"] >= 2 and gr["var_pop"] > 0.5 for gr in g_rows)
    g_moments = all(
        abs(gr["mean"] - GUMBEL_MEAN) < 0.05 and abs(gr["var_pop"] - GUMBEL_VAR) < 0.5
        for gr in g_rows
    )

    # ── gate 5: the noise moves the argmax ──
    score_card = read_out(case, "score", rows)
    am_x, am_s = [], []
    for r in range(rows):
        am_x.append(int(np.argmax(x[r, :v].astype(np.float32))))
        am_s.append(int(np.argmax(score_card[r, :v])))
    moved = [r for r in range(rows) if am_x[r] != am_s[r]]
    moved_gate = (len(moved) >= 1) if case == "b8" else True  # b2: reported only

    # ── report 6: g/score vs the host emulation, real lanes only ──
    score_emu = sn_inputs.emulate_score(x, g_emu)
    g_diff = np.abs(g_card[:, :v] - g_emu[:, :v])
    s_diff = np.abs(score_card[:, :v] - score_emu[:, :v])

    # ── the exits: a liveness witness only (truncated toward zero) ──
    gi = read_out(case, "gi", rows, np.int32)
    si = read_out(case, "si", rows, np.int32)

    gates = {
        "u_bit_exact": u_diff == 0,
        "rows_differ": rows_differ,
        "g_live": g_live,
        "g_moments": g_moments,
        "argmax_moved": moved_gate,
    }
    verdict = {
        "probe": "sample-noise",
        "case": case,
        "rows": rows,
        "vocab": v,
        "pad_v": sn_inputs.PAD_V,
        "u_bitcheck": {"diff": u_diff, "first_mismatch": u_first},
        "rows_differ": rows_differ,
        "g_card_rows": g_rows,
        "gumbel": {"mean": GUMBEL_MEAN, "var_pop": GUMBEL_VAR},
        "g_vs_emu": {"max_abs": float(g_diff.max()), "mean_abs": float(g_diff.mean())},
        "score_vs_emu": {"max_abs": float(s_diff.max()), "mean_abs": float(s_diff.mean())},
        "argmax": {
            "logits": am_x,
            "score": am_s,
            "moved_rows": len(moved),
            "moved_row_indices": moved,
        },
        "exits": {
            "gi_distinct": int(np.unique(gi[:, :v]).size),
            "si_distinct": int(np.unique(si[:, :v]).size),
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    out = HERE / f"sn-verdict-{case}.json"
    with open(out, "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"verdict written to {out}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
