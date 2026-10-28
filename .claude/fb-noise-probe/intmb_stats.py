#!/usr/bin/env python3
"""The int Mb probes' CARD VERDICT — STICK-SCATTER HARNESS (front-b-noise
wall 10, auditor GO 2026-10-27: "The int Mb leg: [2,512] dense + [1,512] Mb
add and mul, plus the [2,65536] twin, through the scatter harness, on
nickm2. On pass: (a) the band-Mb gate extended to IntBinary, (b) the int
door's Mb arm opened citing the leg.").

The Mb form is the 32-row ceiling rung's own bake refusal: the whole-row
`[1, PAD_V]` i32 `lane` table against the band-typed `[B, PAD_V]` tile —
`sample_fwd.py`'s `h = lane * hc1 + slo` and its re-injection `lane * hc2`.
Two cell families measure it:

  * dense+Mb (`intmb_add` / `intmb_mul` / the wide twins) — isolates the
    SPRAY on top of the wall-9 dense control's staging.
  * Mb×Col (`intmb_col_*` / the wide twins) — the PRODUCTION form: the
    hash's multiply has NO dense operand (the per-lane table against the
    per-row word), so the two operand modes are measured TOGETHER.

This script takes the cell names to verdict from argv (the leg names what it
ran — a verdict is never computed over cells that did not run). For each
cell: un-scatter the staged multi-row inputs and CROSS-CHECK them against
the logical generators (a staging bug must not masquerade as a card
defect), read the single-row Mb tables plainly (scatter fixed points),
un-scatter the card's read-back, and compare against the logical host op.
All staged words, products and sums are wrapping u32, so ANY mismatch is
addressing, never numerics.

Usage: intmb_stats.py <cell> [<cell> ...]
Exit 0 iff every named cell is bit-for-bit.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intmb_inputs as G  # noqa: E402  (the generators ARE the input definition)
import intmr_inputs as W  # noqa: E402  (the shared layout law)

IN = G.INDIR
ROWS = G.ROWS


def unscattered(name: str, rows: int, width: int) -> np.ndarray:
    flat = np.fromfile(IN / f"{name}.bin", dtype=np.uint32)
    return W.stick_unscatter(flat, rows, width)


def out_unscattered(cell: str, rows: int, width: int) -> np.ndarray:
    flat = np.fromfile(HERE / f"out_{cell}_o.bin", dtype=np.uint32)
    return W.stick_unscatter(flat, rows, width)


def check_staged(cell: str, what: str, got: np.ndarray, want: np.ndarray, report: dict) -> bool:
    """The staging round-trip: the un-scattered (or plainly read) .bin must
    BE the generator."""
    if not np.array_equal(got, want):
        report["staging_bug"] = f"{cell}: the staged {what} does not read back " \
                                "to its own generator — harness defect, not a card verdict"
        return False
    return True


def main() -> int:
    cells = sys.argv[1:]
    if not cells:
        print("usage: intmb_stats.py <cell> [<cell> ...]")
        return 2

    # (form, width, op) per cell — the host statement of each kernel.
    FORMS = {
        "intmb_add": ("dense_mb", G.N, "add"),
        "intmb_mul": ("dense_mb", G.N, "mul"),
        "intmb_wide_add": ("dense_mb", G.WIDE_N, "add"),
        "intmb_wide_mul": ("dense_mb", G.WIDE_N, "mul"),
        "intmb_col_add": ("mb_col", G.N, "add"),
        "intmb_col_mul": ("mb_col", G.N, "mul"),
        "intmb_wide_col_add": ("mb_col", G.WIDE_N, "add"),
        "intmb_wide_col_mul": ("mb_col", G.WIDE_N, "mul"),
    }
    for cell in cells:
        if cell not in FORMS:
            print(f"unknown cell {cell}")
            return 2

    report = {"probe": "intmb-scatter", "rows": ROWS, "cells": {}}
    ok = True
    for cell in cells:
        form, width, op = FORMS[cell]
        cell_report: dict = {"width": width, "op": op, "form": form}
        dense = G.a_dense(width)
        mb = G.b_mb(width)
        col = W.b_col()
        if form == "dense_mb":
            a_card = unscattered(f"{cell}_t00_a", ROWS, width)
            if not check_staged(cell, "a", a_card, dense, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            b_card = np.fromfile(IN / f"{cell}_t01_b.bin", dtype=np.uint32).reshape(1, width)
            if not check_staged(cell, "b", b_card, mb, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            b_host = mb  # [1, width] — sprays down the rows
        else:  # mb_col — the production hash form: a the lane table, b the word
            a_card = np.fromfile(IN / f"{cell}_t00_a.bin", dtype=np.uint32).reshape(1, width)
            if not check_staged(cell, "a", a_card, mb, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            b_card = W.stick_unscatter(
                np.fromfile(IN / f"{cell}_t01_b.bin", dtype=np.uint32), ROWS, 32
            )[:, :1]
            if not check_staged(cell, "b", b_card, col[:, :1], cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            b_host = np.broadcast_to(col[:, :1], (ROWS, width))  # the per-row word
        a_host = dense if form == "dense_mb" else np.broadcast_to(mb, (ROWS, width))

        if op == "add":
            want = (a_host + b_host).astype(np.uint32)
        else:
            want = (a_host * b_host).astype(np.uint32)
        o = out_unscattered(cell, ROWS, width)

        diff = o != want
        mismatches = int(diff.sum())
        cell_report["mismatches"] = mismatches
        cell_report["pass"] = mismatches == 0
        if mismatches:
            idx = np.nonzero(diff)
            cell_report["first_mismatches"] = [
                {"row": int(r), "lane": int(j), "card": f"0x{o[r, j]:08X}",
                 "host": f"0x{want[r, j]:08X}"}
                for r, j in zip(idx[0][:10], idx[1][:10])
            ]
        report["cells"][cell] = cell_report
        ok = ok and mismatches == 0
        print(f"{cell}: {mismatches} differing lanes of {ROWS * width}")

    out = HERE / "intmb-scatter-verdict.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if not ok:
        print("INTMB-FAILED (see cells above)")
        return 1
    print(f"INTMB-DONE ({len(cells)} cells, bit-for-bit through the stick-scatter harness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
