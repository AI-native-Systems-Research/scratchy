#!/usr/bin/env python3
"""The hash-noise probe's STAGED INPUT TABLES (front-b-noise, nickm2 leg).

Deterministic, no RNG: the seed quad, the hash constants, the lane, and the
fold's five f32 constants, each duplicated down all 2^20 lanes of its
[1, 2^20] table — exactly the kernel's staged-constants law (a splat would
route its consumer to the fp16 scalar-multiply door). The stats script
IMPORTS this module, so the host emulation and the card run consume the
SAME statement of the inputs (the B1 leg's own law: the generators ARE the
input definition).

Files land in ./in/ beside this script (never /tmp): eight u32 tables
(t0..t7), six f32 tables (t8..t13), and one fp16 table (t14, the FEST-domain
clamp), in the kernel's parameter order.
"""
import pathlib

import numpy as np

N = 1 << 20  # PAD_V — the probe's one whole row

# The seed quad: (seed_lo, seed_hi, step, row). Fixed values for the probe
# run — one program, one seed; step=1 and row=0 are the run counters' spellings.
SEED_LO = 0x12345678
SEED_HI = 0x9ABCDEF0
STEP = 1
ROW = 0

# The hash constants — the signed spellings on card are these u32 words:
# 0x9E3779B1 / 0x85EBCA77 / 0xC2B2AE3D, and 2^8 for the wrap-fold shift.
C1 = 0x9E3779B1
C2 = 0x85EBCA77
C3 = 0xC2B2AE3D
C256 = 256

HERE = pathlib.Path(__file__).resolve().parent
INDIR = HERE / "in"


def u32_row(v: int) -> np.ndarray:
    return np.full(N, np.uint32(v), dtype=np.uint32)


def f32_row(v: float) -> np.ndarray:
    return np.full(N, np.float32(v), dtype=np.float32)


def tables() -> list[tuple[str, np.ndarray]]:
    return [
        ("t00_slo", u32_row(SEED_LO)),
        ("t01_shi", u32_row(SEED_HI)),
        ("t02_stp", u32_row(STEP)),
        ("t03_srow", u32_row(ROW)),
        ("t04_c1", u32_row(C1)),
        ("t05_c2", u32_row(C2)),
        ("t06_c3", u32_row(C3)),
        ("t07_c256", u32_row(C256)),
        # The lane, 0..N-1: every value exact in f32 below 2^24.
        ("t08_lane", np.arange(N, dtype=np.float32)),
        # The fold's constants: 2^-8 (window scale), 2^23-1 (reflect),
        # 0.0 (lower clamp), 0.5 and 2^-22 (strict-interior shift/scale).
        ("t09_fm8", f32_row(2.0**-8)),
        ("t10_ftop", f32_row(8388607.0)),
        ("t11_fzero", f32_row(0.0)),
        ("t12_fhalf", f32_row(0.5)),
        ("t13_fm22", f32_row(2.0**-22)),
        # The FEST-domain clamp: 2^-14, fp16's smallest normal, staged as an
        # fp16 row (the card's log is fp16-only, so the draw narrows for each
        # log and clamps its input at the smallest normal — the guard against
        # the convert's and the FEST log's subnormal behavior).
        ("t14_cmin", np.full(N, np.float16(2.0**-14), dtype=np.float16)),
    ]


def main() -> int:
    INDIR.mkdir(exist_ok=True)
    for name, a in tables():
        a.tofile(INDIR / f"{name}.bin")
        print(f"{name}.bin {a.dtype} {a.size} elems {a.nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
