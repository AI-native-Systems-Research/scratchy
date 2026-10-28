#!/usr/bin/env python3
"""The REDUCE-CHAIN TWIN's CARD VERDICT (front-b-noise, wall-9, auditor order
after the wide leg failed identically: "learn the device's own [B,1] layout").

The wide twin proved the HOST-staged [B,1] Col read is row-blind: stick s reads
word (s mod rows) for every row. The auditor's theory: the batched greedy
equality survives because its mm is DEVICE-WRITTEN (the reduce's output) and
the Col read is self-consistent with the device's own write layout.

This cell reproduces the production pair exactly (field-identical emission,
verified against the base b2 dump): max at {mb:2} — one core per row — writing
the [2,1] intermediate at scale [1,-2,1]; the add at {mb:1, out:32} reading it
as the Col operand at scale [1,-2]. The output is the sum cast to i32 through
the sampler's own fptosi sink.

THE READOUT IS THE ADDEND: o - a per (row, stick) names the buffer word the
Col read picked. Row 0's max is 65535, row 1's is 296605, and every other
word of the 256 B intermediate is unwritten (0). So each addend names its
source: 65535 = row 0's max, 296605 = row 1's max, 0 = an unwritten word.
  * PASS (every lane adds its OWN row's max) — the device-written pair is
    self-consistent; the exposure is host-staged tables alone.
  * FAIL — the addend pattern states which words the read picks, i.e. where
    the reduce's write actually lands and whether the equality can see row
    1's max AT ALL. Any pattern where a row reads the other row's max, or 0,
    means the batched greedy equality is wrong on card at B > 1 — urgent.

Exit 0 iff bit-for-bit (every lane adds its own row's max).
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

ROW_MAX = [int(a.max()) for a in (
    np.fromfile(IN / "intmr_f32maxadd_t00_a.bin", dtype=np.float32).reshape(ROWS, WIDE))]


def main() -> int:
    a = np.fromfile(IN / "intmr_f32maxadd_t00_a.bin", dtype=np.float32).reshape(ROWS, WIDE)
    o = np.fromfile(HERE / "out_intmr_f32maxadd_o.bin", dtype=np.int32).reshape(ROWS, WIDE)
    # All staged words, maxima, and sums are exact integers < 2^24 — any
    # mismatch is addressing, never numerics.
    ai = a.astype(np.int64)
    want = ai + np.array(ROW_MAX, dtype=np.int64)[:, None]
    diff = o != want
    mismatches = int(diff.sum())

    # THE ADDEND PATTERN — the measurement's own language. Each addend names
    # the buffer word the Col read picked at that (row, stick).
    addend = o.astype(np.int64) - ai
    per_row_sticks = []
    for r in range(ROWS):
        words = []
        for s in range(STICKS):
            seg = np.unique(addend[r, s * 32:(s + 1) * 32])
            words.append(seg.tolist())
        per_row_sticks.append(words)

    def name(v):
        if v == ROW_MAX[0]:
            return "row0max"
        if v == ROW_MAX[1]:
            return "row1max"
        if v == 0:
            return "UNWRITTEN(0)"
        return f"other({v})"

    report = {
        "probe": "intmr-f32maxadd",
        "rows": ROWS,
        "lanes": WIDE,
        "sticks_per_row": STICKS,
        "row_max": ROW_MAX,
        "form": "max(axis=1) -> expand_dims -> add Col, [2,65536] fp32, the "
                "equality's own writer->reader pair (max {mb:2}, add {mb:1,out:32})",
        "mismatches": mismatches,
        "pass": mismatches == 0,
        "per_row_per_stick_addends_first8": [w[:8] for w in per_row_sticks],
        "per_row_per_stick_addend_names_first8": [
            [[name(v) for v in w] for w in per_row_sticks[r][:8]] for r in range(ROWS)
        ],
        "per_row_distinct_addends": [
            sorted({v for stick in per_row_sticks[r] for v in stick}) for r in range(ROWS)
        ],
        "per_row_distinct_addend_names": [
            sorted({name(v) for stick in per_row_sticks[r] for v in stick}) for r in range(ROWS)
        ],
    }
    if mismatches:
        # Which sticks read which source, compressed: for each row, the set of
        # stick indices per addend name (the pattern's period if regular).
        for r in range(ROWS):
            by_src = {}
            for s in range(STICKS):
                key = name(per_row_sticks[r][s][0]) if per_row_sticks[r][s] else "?"
                by_src.setdefault(key, []).append(s)
            report[f"row{r}_sticks_by_source"] = {
                k: {"count": len(v), "first8": v[:8],
                    "stride_regularity": (v[1] - v[0]) if len(v) > 1 else None}
                for k, v in by_src.items()
            }
        idx = np.nonzero(diff)
        report["first_mismatches"] = [
            {"row": int(r), "lane": int(j), "card": int(o[r, j]), "host": int(want[r, j]),
             "addend": int(addend[r, j]), "addend_is": name(int(addend[r, j]))}
            for r, j in zip(idx[0][:10], idx[1][:10])
        ]

    out = HERE / "intmr-maxadd-verdict.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if mismatches:
        print("INTMRMA-FAILED (the device-written Col read is NOT self-consistent "
              "-- the equality's proof is in question)")
        return 1
    print(f"INTMRMA-DONE ({ROWS}x{WIDE}: every lane added its own row's max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
