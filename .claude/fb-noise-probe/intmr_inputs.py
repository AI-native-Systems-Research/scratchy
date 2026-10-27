#!/usr/bin/env python3
"""The int multi-row probes' STAGED INPUT TABLES (front-b-noise, nickm2 leg).

Deterministic, no RNG, and EVERY ROW DIFFERENT — the whole point of the
multi-row cells: a broadcast defect that sprays row 0's word across row 1, or
an addressing defect that reads either row at the wrong stick, must land on a
mismatch, never a coincidental pass.

  * `a` — `[2, 512]` u32: row 0 is wrapping multiples of 0x9E3779B1 with the
    boundary words pinned at the head (0, 1, INT_MIN, INT_MAX — the wrap
    cases staged on card); row 1 is wrapping multiples of 0x85EBCA77 with the
    same boundary pins, so the two rows share no lane value.
  * `b_dense` — `[2, 512]` u32 for the control cells: row 0 is wrapping
    multiples of 0xC2B2AE3D, row 1 of 0x9E3779B1 with a stride offset — dense
    operands whose every lane differs from `a`'s.
  * `b_col` — the `[2, 1]` word table at the SENUINT32 one-stick footprint
    (32 words per row, the value in word 0 of each row's own stick — the
    128 B-per-row law the seed table's own rows obey): row 0 carries
    0x85EBCA77, row 1 carries 0xC2B2AE3D, DIFFERENT odd constants with bits in
    both halves so each row's carry pattern is its own.

The stats script IMPORTS this module, so the host compare and the card run
consume the SAME statement of the inputs (the B1 leg's own law: the generators
ARE the input definition).

Files land in ./in/ beside this script (never /tmp), per cell in the kernel's
parameter order: `<cell>_t00_a` and `<cell>_t01_b`.
"""
import pathlib

import numpy as np

N = 512  # PAD_V — the auditor's stated width
ROWS = 2  # B — the multi-row width

HERE = pathlib.Path(__file__).resolve().parent
INDIR = HERE / "in"

# The per-row words for the Col cells — different odd constants per row.
COL_WORDS = [0x85EBCA77, 0xC2B2AE3D]


def a_tile() -> np.ndarray:
    """The [2, 512] dense tile: two rows of wrapping multiples of different
    odd constants, boundary words pinned at each row's head."""
    out = np.zeros((ROWS, N), dtype=np.uint32)
    for r, c in enumerate((0x9E3779B1, 0x85EBCA77)):
        row = (np.arange(N, dtype=np.uint64) * np.uint64(c)) & np.uint64(0xFFFFFFFF)
        row = row.astype(np.uint32)
        for i, v in enumerate([0, 1, 0x80000000, 0x7FFFFFFF]):
            row[i] = np.uint32(v)
        out[r] = row
    return out


def b_dense() -> np.ndarray:
    """The control cells' [2, 512] dense operand: every lane different from
    `a`'s, each row its own constant's multiples."""
    out = np.zeros((ROWS, N), dtype=np.uint32)
    for r, c in enumerate((0xC2B2AE3D, 0x9E3779B1)):
        row = (np.arange(N, dtype=np.uint64) * np.uint64(c) + np.uint64(r + 1)) & np.uint64(0xFFFFFFFF)
        out[r] = row.astype(np.uint32)
    return out


def b_col() -> np.ndarray:
    """The Col cells' [2, 1] word table at its SENUINT32 one-stick footprint:
    32 words per row, the row's own word in word 0 of its own stick."""
    stick = np.zeros((ROWS, 32), dtype=np.uint32)
    for r, w in enumerate(COL_WORDS):
        stick[r, 0] = np.uint32(w)
    return stick


def tables() -> list[tuple[str, np.ndarray]]:
    a = a_tile()
    dense = b_dense()
    col = b_col()
    out: list[tuple[str, np.ndarray]] = []
    for cell in ("intmr_dense_add", "intmr_dense_mul"):
        out.append((f"{cell}_t00_a", a))
        out.append((f"{cell}_t01_b", dense))
    for cell in ("intmr_col_add", "intmr_col_mul"):
        out.append((f"{cell}_t00_a", a))
        out.append((f"{cell}_t01_b", col))
    return out


def main() -> int:
    INDIR.mkdir(exist_ok=True)
    for name, arr in tables():
        arr.tofile(INDIR / f"{name}.bin")
        print(f"{name}.bin {arr.dtype} {arr.shape} {arr.nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
