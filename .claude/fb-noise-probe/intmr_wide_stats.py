#!/usr/bin/env python3
"""The WIDE TWIN's CARD VERDICT (front-b-noise, wall-9 width leg, auditor order
2026-10-27T02:5xZ).

The re-leg refuted the both-split diagnosis: the fold moved to {mb:1, out:16}
and the card output did not move — the Col operand's word is selected by the
output's stick index (stick s reads row s mod rows, for EVERY output row),
division-invariant. The field-by-field descriptor diff against the card-proven
b2 equality then left ONE variable: PER-CORE SLICE WIDTH — the probe's core
owns 1 stick, the equality's owns 64.

This cell is the same kernel at production width: PAD_V 65536, 2048 out-sticks,
the equality's own fold class. The verdict decides the auditor's fork:
  * PASS  — the Col defect is narrow-slice only; production is safe, and the
    int arm can be admitted at production widths with the narrow form refused
    by name.
  * FAIL  — the batched greedy equality is only passing by luck. Urgent.

The verdict reports the per-stick addend pattern either way, so a failure
names the law it broke (which row each stick read). Exit 0 iff bit-for-bit.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intmr_inputs as G  # noqa: E402  (the generators ARE the input definition)

IN = G.INDIR
WIDE = G.WIDE_N
ROWS = G.ROWS
STICKS = WIDE // 32


def main() -> int:
    a = np.fromfile(IN / "intmr_f32wide_add_t00_a.bin", dtype=np.float32).reshape(ROWS, WIDE)
    b = np.fromfile(IN / "intmr_f32wide_add_t01_b.bin", dtype=np.float32).reshape(ROWS, 32)[:, :1]
    o = np.fromfile(HERE / "out_intmr_f32wide_add_o.bin", dtype=np.float32).reshape(ROWS, WIDE)
    # All staged words and sums are exact integers < 2^24 — any mismatch is
    # addressing, never numerics.
    want = a + b
    diff = o != want
    mismatches = int(diff.sum())

    # The per-stick addend pattern — which row's word each stick read. This is
    # the measurement's own language: the narrow twin read row (s mod 2) for
    # every output row.
    per_row_sticks = []
    for r in range(ROWS):
        words = []
        for s in range(STICKS):
            seg = np.unique(np.round(o[r, s * 32:(s + 1) * 32] - a[r, s * 32:(s + 1) * 32], 3))
            words.append(seg.tolist())
        per_row_sticks.append(words)

    report = {
        "probe": "intmr-f32wide",
        "rows": ROWS,
        "lanes": WIDE,
        "sticks_per_row": STICKS,
        "form": "add, [2,65536] dense IEEE_FP32 + [2,1] Col at production width "
                "(the equality's fold class {mb:1, out:32}, 64 sticks per core)",
        "mismatches": mismatches,
        "pass": mismatches == 0,
        "per_row_per_stick_addends_first8": [w[:8] for w in per_row_sticks],
    }
    if mismatches:
        first_rows = sorted(set(int(r) for r in np.nonzero(diff)[0]))
        report["mismatching_rows"] = first_rows
        report["per_row_mismatch_counts"] = {
            str(r): int(diff[r].sum()) for r in range(ROWS)
        }
        idx = np.nonzero(diff)
        report["first_mismatches"] = [
            {
                "row": int(r),
                "lane": int(j),
                "card": float(o[r, j]),
                "host": float(want[r, j]),
                "a": float(a[r, j]),
            }
            for r, j in zip(idx[0][:10], idx[1][:10])
        ]
        # The full per-stick pattern compressed: for each row, the set of
        # distinct addends over all sticks (the narrow twin's was
        # {4096, -8192} alternating).
        report["per_row_distinct_addends"] = [
            sorted({w for stick in per_row_sticks[r] for w in stick}) for r in range(ROWS)
        ]

    out = HERE / "intmr-wide-verdict.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if mismatches:
        print("INTMRW-FAILED (the wide Col is WRONG — the equality's proof is in question)")
        return 1
    print(f"INTMRW-DONE ({ROWS}x{WIDE}, bit-for-bit at production width)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
