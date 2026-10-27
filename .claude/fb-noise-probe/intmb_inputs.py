#!/usr/bin/env python3
"""The int Mb probes' STAGED INPUT TABLES (front-b-noise wall 10, nickm2 leg,
auditor GO 2026-10-27).

The same DEVICE LAYOUT LAW as the intmr legs (auditor ruling 2026-10-27,
citing forward_tape.rs:299): the card lays a `[rows, width]` tensor
STICK-MAJOR — stick k of row r at flat stick `k*rows + r` — so every
multi-row input is STAGED through that scatter and every output UN-SCATTERED
from it. A `[1, width]` single-row table is a FIXED POINT of the scatter
(stick k of row 0 at flat stick `k`), so the Mb tables stage PLAINLY — the
same law that left the `[rows,1]` Col tables alone at wall 9.

  * `a_dense` — the dense `[2, 512]` / `[2, 65536]` u32 tile: the intmr legs'
    own generator (wrapping multiples of different odd constants per row,
    boundary words pinned at each row's head), so the dense side of the
    dense+Mb cells is the SAME measured staging the wall-9 controls ran.
  * `b_mb` — the `[1, 512]` / `[1, 65536]` u32 whole-row table: wrapping
    multiples of 0xC2B2AE3D with the boundary pins — every lane different
    from every row of `a_dense`, so a spray that reads the wrong lane (or
    repeats one lane) is a mismatch, never a coincidental pass.
  * `b_col` — the intmr legs' own `[2, 1]` word table (different odd
    constants per row at the SENUINT32 one-stick footprint), for the
    Mb×Col cells: the production hash form `lane * hc1` has NO dense
    operand.

The stats script IMPORTS this module (and intmr_inputs for the shared
generators), so the host compare and the card run consume the SAME statement
of the inputs — the B1 leg's own law: the generators ARE the input
definition. The `.bin` files carry the STICK-SCATTERED bytes for multi-row
tiles and the plain bytes for the single-row tables.

Files land in ./in/ beside this script (never /tmp), per cell in the
kernel's parameter order: `<cell>_t00_a` and `<cell>_t01_b`.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intmr_inputs as G  # noqa: E402  (the shared layout law + generators)

N = G.N        # 512 — the auditor's stated width
WIDE_N = G.WIDE_N  # 65536 — production PAD_V
ROWS = G.ROWS  # 2

INDIR = HERE / "in"


def a_dense(width: int) -> np.ndarray:
    """The dense [2, width] u32 tile: the intmr generator's own pattern at
    any width — wrapping multiples of a different odd constant per row,
    boundary words pinned at each row's head."""
    out = np.zeros((ROWS, width), dtype=np.uint32)
    for r, c in enumerate((0x9E3779B1, 0x85EBCA77)):
        row = (np.arange(width, dtype=np.uint64) * np.uint64(c)) & np.uint64(0xFFFFFFFF)
        row = row.astype(np.uint32)
        for i, v in enumerate([0, 1, 0x80000000, 0x7FFFFFFF]):
            row[i] = np.uint32(v)
        out[r] = row
    return out


def b_mb(width: int) -> np.ndarray:
    """The [1, width] u32 whole-row table: wrapping multiples of a THIRD odd
    constant with the boundary pins — every lane differs from every row of
    `a_dense`, so a wrong-lane or repeated-lane spray is a mismatch."""
    row = (np.arange(width, dtype=np.uint64) * np.uint64(0xC2B2AE3D)) & np.uint64(0xFFFFFFFF)
    row = row.astype(np.uint32)
    for i, v in enumerate([0, 1, 0x80000000, 0x7FFFFFFF]):
        row[i] = np.uint32(v)
    return row.reshape(1, width)


def tables() -> list[tuple[str, np.ndarray]]:
    """The staged files, per cell in the kernel's parameter order. The dense
    tiles ride `stick_scatter`; the single-row Mb tables and the `[rows,1]`
    Col tables are scatter fixed points and stage their own bytes."""
    out: list[tuple[str, np.ndarray]] = []
    # dense + Mb — the ordered cells (a dense pid-cornered, b the lane table).
    for cell in ("intmb_add", "intmb_mul"):
        out.append((f"{cell}_t00_a", G.stick_scatter(a_dense(N))))
        out.append((f"{cell}_t01_b", b_mb(N)))
    for cell in ("intmb_wide_add", "intmb_wide_mul"):
        out.append((f"{cell}_t00_a", G.stick_scatter(a_dense(WIDE_N))))
        out.append((f"{cell}_t01_b", b_mb(WIDE_N)))
    # Mb × Col — the production hash form (a the lane table, b the per-row word).
    for cell in ("intmb_col_add", "intmb_col_mul"):
        out.append((f"{cell}_t00_a", b_mb(N)))
        out.append((f"{cell}_t01_b", G.b_col()))
    for cell in ("intmb_wide_col_add", "intmb_wide_col_mul"):
        out.append((f"{cell}_t00_a", b_mb(WIDE_N)))
        out.append((f"{cell}_t01_b", G.b_col()))
    return out


def main() -> int:
    INDIR.mkdir(exist_ok=True)
    for name, arr in tables():
        arr.tofile(INDIR / f"{name}.bin")
        print(f"{name}.bin {arr.dtype} {arr.shape} {arr.nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
