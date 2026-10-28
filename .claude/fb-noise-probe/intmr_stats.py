#!/usr/bin/env python3
"""The int multi-row probes' CARD VERDICT (front-b-noise, nickm2 leg).

The auditor's GO (2026-10-27): "The two cells on nickm2, as you designed them:
[2,512] dense + [2,512] dense (the control), then [2,512] + [2,1] Col, for add
AND mul." Wrapping i32 is exact on both sides, so each cell's verdict is ONE
number — the count of lanes where the card's `[2, 512]` result disagrees with
the host's row-and-lane wrapping op. The dense cells isolate the MULTI-ROW
STAGING (B1 measured `[1, 512]` alone); the Col cells measure the operand mode
ON TOP of it — the attribution the two-cell design buys. ANY mismatch is an
addressing defect (a wrong lane, a wrong stride, a sprayed word read from the
wrong row's stick), never numerics. The first mismatching lanes are printed
with both words so the card bytes can be read straight off the log if a cell
fails.

Every cell's result is written to intmr-verdict.json beside this script. Exit 0
iff every cell's mismatch count is zero.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intmr_inputs as G  # noqa: E402  (the generators ARE the input definition)

IN = G.INDIR

CELLS = [
    ("intmr_dense_add", "add", "dense"),
    ("intmr_dense_mul", "mul", "dense"),
    ("intmr_col_add", "add", "col"),
    ("intmr_col_mul", "mul", "col"),
    # The FP32 TWIN (wall-9 diagnosis): the same [2,512]+[2,1] Col form at
    # IEEE_FP32 — the dataFormat is the ONLY variable against intmr_col_add.
    # Its verdict decides whether the measured-wrong addressing is int-specific
    # or the form's own (the emissions are field-identical besides the df).
    ("intmr_f32col_add", "add", "f32col"),
    # The UN-FIXED TWIN (auditor GO 2026-10-27): the PRE-BAKED bundle from
    # before the both-split law (/work/int-mr-f32col-unfixed, {mb:2, out:16}),
    # run in the SAME leg as the fixed one. RE-LEG VERDICT (02:51:39Z): the
    # diagnosis is REFUTED — the fixed twin measured wrong the same way
    # (512/1024, bit-identical bytes), so the work division is not the cause
    # and this cell's failure confirms nothing. It stays in the leg as the
    # division-invariance control: the two twins' read-backs must stay
    # bit-identical, whichever fold runs.
    ("intmr_f32col_add_unfixed", "add", "f32col_unfixed"),
]


def read_u32(name: str) -> np.ndarray:
    return np.fromfile(IN / f"{name}.bin", dtype=np.uint32)


def main() -> int:
    a = read_u32("intmr_dense_add_t00_a").reshape(G.ROWS, G.N)
    a32 = np.fromfile(IN / "intmr_f32col_add_t00_a.bin", dtype=np.float32).reshape(G.ROWS, G.N)
    report = {"probe": "intmr+f32col", "rows": G.ROWS, "lanes": G.N, "cells": {}}
    ok = True
    for cell, op, mode in CELLS:
        if mode in ("f32col", "f32col_unfixed"):
            # The unfixed arm is the DIAGNOSIS CONTROL: its bundle was baked
            # BEFORE the both-split law and states the {mb:2, out:16} fold, so
            # it must MEASURE WRONG. Its pass condition is inverted.
            expect_fail = mode == "f32col_unfixed"
            b = np.fromfile(IN / "intmr_f32col_add_t01_b.bin", dtype=np.float32)
            # The card run's read-back, written beside this script by the leg.
            o = np.fromfile(HERE / f"out_{cell}_o.bin", dtype=np.float32).reshape(G.ROWS, G.N)
            # Each row's own word: word 0 of its own 32-word stick. All staged
            # words and sums are exact integers < 2^24, so the f32 add is exact
            # on both sides — any mismatch is addressing, never numerics.
            bm = b.reshape(G.ROWS, 32)[:, :1]
            want = a32 + bm
            diff = o != want
            mismatches = int(diff.sum())
            if expect_fail:
                cell_report = {
                    "form": "add, the UN-FIXED fp32 twin ({mb:2, out:16}, pre-baked "
                    "before the both-split law) — must MEASURE WRONG",
                    "mismatches": mismatches,
                    "pass": mismatches > 0,
                }
                if mismatches == 0:
                    cell_report["refuted"] = (
                        "the un-fixed fold read the right rows — the both-split "
                        "diagnosis is WRONG"
                    )
                else:
                    idx = np.nonzero(diff)
                    cell_report["first_mismatches"] = [
                        {
                            "row": int(r),
                            "lane": int(j),
                            "card": float(o[r, j]),
                            "host": float(want[r, j]),
                            "a": float(a32[r, j]),
                        }
                        for r, j in zip(idx[0][:10], idx[1][:10])
                    ]
                report["cells"][cell] = cell_report
                ok = ok and mismatches > 0
                # Neutral wording, per the auditor's correction (2026-10-27T02:5xZ): the
                # un-fixed twin measuring wrong CONFIRMS NOTHING once the fixed twin
                # measures wrong the same way — the re-leg refuted the both-split
                # diagnosis, and this label must not claim it back.
                verdict = "MEASURED WRONG (no confirmation — compare the fixed twin)"
                print(f"{cell}: {mismatches} differing lanes of {G.ROWS * G.N} — {verdict}")
                continue
            cell_report = {
                "form": "add, [2,512] dense IEEE_FP32 + [2,1] Col (the intmr_col_add twin)",
                "mismatches": mismatches,
                "pass": mismatches == 0,
            }
            if mismatches:
                idx = np.nonzero(diff)
                cell_report["first_mismatches"] = [
                    {
                        "row": int(r),
                        "lane": int(j),
                        "card": float(o[r, j]),
                        "host": float(want[r, j]),
                        "a": float(a32[r, j]),
                    }
                    for r, j in zip(idx[0][:10], idx[1][:10])
                ]
            report["cells"][cell] = cell_report
            ok = ok and mismatches == 0
            print(f"{cell}: {mismatches} differing lanes of {G.ROWS * G.N}")
            continue
        b = read_u32(f"{cell}_t01_b")
        # The card run's read-back, written beside this script by the leg.
        o = np.fromfile(HERE / f"out_{cell}_o.bin", dtype=np.uint32).reshape(G.ROWS, G.N)
        if mode == "dense":
            bm = b.reshape(G.ROWS, G.N)
        else:
            # Each row's own word: word 0 of its own 32-word stick.
            bm = b.reshape(G.ROWS, 32)[:, :1]
        if op == "add":
            want = (a + bm).astype(np.uint32)
        else:
            want = (a * bm).astype(np.uint32)
        diff = o != want
        mismatches = int(diff.sum())
        cell_report = {
            "form": f"{'addi32toi32' if op == 'add' else 'muli32toi32'}, "
            f"[2,512] dense + [2,{ '512' if mode == 'dense' else '1'}] "
            f"{'dense' if mode == 'dense' else 'Col'}",
            "mismatches": mismatches,
            "pass": mismatches == 0,
        }
        if mismatches:
            idx = np.nonzero(diff)
            cell_report["first_mismatches"] = [
                {
                    "row": int(r),
                    "lane": int(j),
                    "card": f"0x{o[r, j]:08X}",
                    "host": f"0x{want[r, j]:08X}",
                    "a": f"0x{a[r, j]:08X}",
                }
                for r, j in zip(idx[0][:10], idx[1][:10])
            ]
        report["cells"][cell] = cell_report
        ok = ok and mismatches == 0
        print(f"{cell}: {mismatches} differing lanes of {G.ROWS * G.N}")

    out = HERE / "intmr-verdict.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if not ok:
        if report["cells"].get("intmr_f32col_add_unfixed", {}).get("mismatches") == 0:
            print("UNFIXED-TWIN-PASSED — DIAGNOSIS REFUTED (stop; report to the auditor)")
        print("INTMR-FAILED (see cells above)")
        return 1
    print(f"INTMR-DONE ({len(CELLS)} cells, bit-for-bit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
