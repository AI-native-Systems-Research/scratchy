#!/usr/bin/env python3
"""The int-mul-bcast probe's STAGED INPUT TABLES (front-b-noise, nickm2 leg).

Deterministic, no RNG: `a` is a wrapping-i32 pattern that exercises bit
carries across the whole word (odd multiples of the golden-ratio constant,
plus a zero, a one, INT_MIN and INT_MAX so the multiply's wrap cases are on
the card — 0 annihilates, 1 is the identity, INT_MIN*odd wraps to itself
shifted, INT_MAX*odd carries out of the top); `b` is the per-row word,
staged at the SENUINT32 one-stick footprint (32 words, the value in word 0 —
the same 128 B-per-row law the seed table's own rows obey). The stats script
IMPORTS this module, so the host compare and the card run consume the SAME
statement of the inputs (the B1 leg's own law: the generators ARE the input
definition).

Files land in ./in/ beside this script (never /tmp): t00_a (512 u32) and
t01_b (32 u32, one stick), in the kernel's parameter order.
"""
import pathlib

import numpy as np

N = 512  # PAD_V — the auditor's stated width

# The per-row word: an odd constant with bits in both halves (the fold's own
# hc2 spelling), so the product carries into the high byte on most lanes of
# `a` and the operand's broadcast is exercised at every bit position.
B_WORD = 0x85EBCA77

HERE = pathlib.Path(__file__).resolve().parent
INDIR = HERE / "in"


def a_row() -> np.ndarray:
    """The dense row: wrapping-i32 multiples of the golden-ratio constant,
    with the boundary words pinned at the head (0, 1, INT_MIN, INT_MAX) so
    the wrap cases are staged, not just the generic carry case."""
    a = (np.arange(N, dtype=np.uint64) * np.uint64(0x9E3779B1)) & np.uint64(0xFFFFFFFF)
    a = a.astype(np.uint32)
    for i, v in enumerate([0, 1, 0x80000000, 0x7FFFFFFF]):
        a[i] = np.uint32(v)
    return a


def b_stick() -> np.ndarray:
    """The [1, 1] word at its SENUINT32 one-stick footprint: 32 words, the
    value in word 0, the rest zero — the footprint law the Col handle's
    `rb(name, rows, elems_per_stick)` annotation states."""
    stick = np.zeros(32, dtype=np.uint32)
    stick[0] = np.uint32(B_WORD)
    return stick


def tables() -> list[tuple[str, np.ndarray]]:
    return [("t00_a", a_row()), ("t01_b", b_stick())]


def main() -> int:
    INDIR.mkdir(exist_ok=True)
    for name, a in tables():
        a.tofile(INDIR / f"{name}.bin")
        print(f"{name}.bin {a.dtype} {a.size} elems {a.nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
