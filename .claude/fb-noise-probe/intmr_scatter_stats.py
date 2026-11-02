#!/usr/bin/env python3
"""The int multi-row probes' CARD VERDICT — STICK-SCATTER HARNESS (front-b-noise,
auditor ruling 2026-10-27: "stage every multi-row input through stick_scatter
and un-scatter every output (forward_tape.rs:299 law), then re-leg the col,
maxadd and wide cells. Expect them to pass.").

The first legs staged and read ROW-MAJOR against the device's STICK-MAJOR
layout (stick k of row r at flat stick k*rows + r). A position-wise op
commutes with a consistent permutation, so the DENSE controls passed, while
every row-sensitive cell (Col, reduce) failed with the s-mod-rows signature
that IS the scatter — the device was right, the harness was wrong. The
"[rows,1]" Col tables are fixed points of the scatter (one stick per row),
so the card's per-row Col semantics were never actually measured wrong.

This script takes the cell names to verdict from argv (the leg names what it
ran — a verdict is never computed over cells that did not run). For each
cell: un-scatter the staged inputs and CROSS-CHECK them against the logical
generators (a staging bug must not masquerade as a card defect), un-scatter
the card's read-back, and compare against the logical host op. All staged
words, products, sums and maxima are exact integers < 2^24 (or wrapping u32),
so ANY mismatch is addressing, never numerics.

Usage: intmr_scatter_stats.py <cell> [<cell> ...]
Exit 0 iff every named cell is bit-for-bit.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intmr_inputs as G  # noqa: E402  (the generators ARE the input definition)

IN = G.INDIR
ROWS = G.ROWS


def unscattered(name: str, rows: int, width: int, dtype) -> np.ndarray:
    flat = np.fromfile(IN / f"{name}.bin", dtype=dtype)
    return G.stick_unscatter(flat, rows, width)


def out_unscattered(cell: str, rows: int, width: int, dtype) -> np.ndarray:
    flat = np.fromfile(HERE / f"out_{cell}_o.bin", dtype=dtype)
    return G.stick_unscatter(flat, rows, width)


def check_staged(cell: str, what: str, got: np.ndarray, want: np.ndarray, report: dict):
    """The staging round-trip: the un-scattered .bin must BE the generator."""
    if not np.array_equal(got, want):
        report["staging_bug"] = f"{cell}: the staged {what} does not un-scatter " \
                                "to its own generator — harness defect, not a card verdict"
        return False
    return True


def col_words(table_flat: np.ndarray) -> np.ndarray:
    """The [rows,1] word table: word 0 of each row's own stick (a scatter fixed
    point — the flat buffer IS [row r's stick] in row order)."""
    return table_flat.reshape(ROWS, G.WORDS_PER_STICK)[:, :1]


def main() -> int:
    cells = sys.argv[1:]
    if not cells:
        print("usage: intmr_scatter_stats.py <cell> [<cell> ...]")
        return 2

    a_u32 = G.a_tile()
    b_dense = G.b_dense()
    b_col = G.b_col()
    a_f32 = G.a_f32()
    b_f32 = G.b_f32col()
    a_wide = G.a_f32wide()

    # (dtype, width, op) per cell — the host statement of each kernel.
    FORMS = {
        "intmr_dense_add": ("u32", G.N, "add"),
        "intmr_dense_mul": ("u32", G.N, "mul"),
        "intmr_col_add": ("u32", G.N, "add"),
        "intmr_col_mul": ("u32", G.N, "mul"),
        "intmr_f32col_add": ("f32", G.N, "add"),
        "intmr_f32wide_add": ("f32", G.WIDE_N, "add"),
        "intmr_f32maxadd": ("f32", G.WIDE_N, "maxadd"),
    }
    for cell in cells:
        if cell not in FORMS:
            print(f"unknown cell {cell}")
            return 2

    report = {"probe": "intmr-scatter", "rows": ROWS, "cells": {}}
    ok = True
    for cell in cells:
        df, width, op = FORMS[cell]
        cell_report: dict = {"width": width, "op": op}
        if df == "u32":
            a_host = a_u32
            a_card = unscattered(f"{cell}_t00_a", ROWS, width, np.uint32)
            if not check_staged(cell, "a", a_card, a_host, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            if cell.startswith("intmr_dense"):
                b_host = b_dense
                b_card = unscattered(f"{cell}_t01_b", ROWS, width, np.uint32)
            else:
                b_host = col_words(G.stick_scatter(b_col))  # the fixed point
                b_card = col_words(np.fromfile(IN / f"{cell}_t01_b.bin", dtype=np.uint32))
            if not check_staged(cell, "b", b_card, b_host, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            if op == "add":
                want = (a_host + b_host).astype(np.uint32)
            else:
                want = (a_host * b_host).astype(np.uint32)
            o = out_unscattered(cell, ROWS, width, np.uint32)
            fmt = lambda v: f"0x{v:08X}"  # noqa: E731
        elif cell == "intmr_f32maxadd":
            a_card = unscattered(f"{cell}_t00_a", ROWS, width, np.float32)
            if not check_staged(cell, "a", a_card, a_wide, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            row_max = a_wide.max(axis=1).astype(np.int64)
            want = (a_wide.astype(np.int64) + row_max[:, None]).astype(np.int32)
            o = out_unscattered(cell, ROWS, width, np.int32)
            cell_report["row_max"] = row_max.tolist()
            fmt = lambda v: int(v)  # noqa: E731
        else:
            a_host = a_f32 if width == G.N else a_wide
            a_card = unscattered(f"{cell}_t00_a", ROWS, width, np.float32)
            if not check_staged(cell, "a", a_card, a_host, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            b_host = col_words(G.stick_scatter(b_f32))
            b_card = col_words(np.fromfile(IN / f"{cell}_t01_b.bin", dtype=np.float32))
            if not check_staged(cell, "b", b_card, b_host, cell_report):
                report["cells"][cell] = cell_report
                ok = False
                print(f"{cell}: STAGING BUG (see report)")
                continue
            want = a_host + b_host
            o = out_unscattered(cell, ROWS, width, np.float32)
            fmt = lambda v: float(v)  # noqa: E731

        diff = o != want
        mismatches = int(diff.sum())
        cell_report["mismatches"] = mismatches
        cell_report["pass"] = mismatches == 0
        if mismatches:
            idx = np.nonzero(diff)
            cell_report["first_mismatches"] = [
                {"row": int(r), "lane": int(j), "card": fmt(o[r, j]), "host": fmt(want[r, j])}
                for r, j in zip(idx[0][:10], idx[1][:10])
            ]
        report["cells"][cell] = cell_report
        ok = ok and mismatches == 0
        print(f"{cell}: {mismatches} differing lanes of {ROWS * width}")

    out = HERE / "intmr-scatter-verdict.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if not ok:
        print("INTMRSC-FAILED (see cells above)")
        return 1
    print(f"INTMRSC-DONE ({len(cells)} cells, bit-for-bit through the stick-scatter harness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
