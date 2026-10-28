#!/usr/bin/env python3
"""The int multi-row probes' STAGED INPUT TABLES (front-b-noise, nickm2 leg).

Deterministic, no RNG, and EVERY ROW DIFFERENT — the whole point of the
multi-row cells: a broadcast defect that sprays row 0's word across row 1, or
an addressing defect that reads either row at the wrong stick, must land on a
mismatch, never a coincidental pass.

⭐ THE DEVICE LAYOUT (auditor ruling 2026-10-27, citing forward_tape.rs:299):
the card lays a `[rows, width]` tensor STICK-MAJOR — stick k of row r at flat
stick `k*rows + r` — so every multi-row input is STAGED through that scatter
and every output UN-SCATTERED from it. (The Rust law's 64 counts fp16-sized
slots: 64 halves = 32 fp32/u32 words = ONE 128 B stick, so at fp32/u32 the
granule is 32 words.) The first legs staged and read ROW-MAJOR; a
position-wise op commutes with a consistent permutation, so the DENSE
controls passed, while every row-sensitive cell failed with the s-mod-rows
signature that IS the scatter, not a device defect. The `[rows,1]` Col tables
are fixed points of the scatter (one stick per row), which is why the card's
per-row Col semantics were right all along.

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
  * `a_f32` / `b_f32col` — the FP32 TWIN cell's tables (`intmr_f32col_add`,
    the wall-9 diagnosis leg): the same `[2, 512]`-against-`[2, 1]` Col form
    at IEEE_FP32, in exactly-representable integers (every staged word and
    every sum < 2^24), so the bit-for-bit compare is exact arithmetic on both
    sides. Row 0 of `a_f32` is `j`, row 1 is `100000 + 3j`; the Col words are
    +4096.0 (row 0) and -8192.0 (row 1) at word 0 of each row's own stick,
    distinct signs so a row-swap or a wrong-stick read is a mismatch, never a
    coincidental pass.

The stats script IMPORTS this module, so the host compare and the card run
consume the SAME statement of the inputs (the B1 leg's own law: the generators
ARE the input definition). The `.bin` files carry the STICK-SCATTERED bytes;
`stick_unscatter` recovers the logical tile for the host compare.

Files land in ./in/ beside this script (never /tmp), per cell in the kernel's
parameter order: `<cell>_t00_a` and `<cell>_t01_b`.
"""
import pathlib

import numpy as np

N = 512  # PAD_V — the auditor's stated width
ROWS = 2  # B — the multi-row width

HERE = pathlib.Path(__file__).resolve().parent
INDIR = HERE / "in"

# One 128 B stick at fp32/u32: 32 words (the forward_tape law's 64 counts
# fp16-sized slots — 64 halves = 32 words).
WORDS_PER_STICK = 32


def stick_scatter(arr: np.ndarray) -> np.ndarray:
    """The device's stick-scattered stage of a `[rows, width]` tile
    (forward_tape.rs:299): stick k of row r lands at flat stick `k*rows + r`.
    A `[rows, 1]` table (one stick per row) is its own scatter."""
    rows, width = arr.shape
    sticks = arr.reshape(rows, width // WORDS_PER_STICK, WORDS_PER_STICK)
    return sticks.transpose(1, 0, 2).reshape(-1)


def stick_unscatter(flat: np.ndarray, rows: int, width: int) -> np.ndarray:
    """The inverse: flat stick `k*rows + r` back to `[rows, width]` row-major."""
    sticks = flat.reshape(width // WORDS_PER_STICK, rows, WORDS_PER_STICK)
    return sticks.transpose(1, 0, 2).reshape(rows, width)

# The per-row words for the Col cells — different odd constants per row.
COL_WORDS = [0x85EBCA77, 0xC2B2AE3D]

# The fp32 twin's Col words — distinct signs, exactly representable.
F32_COL_WORDS = [4096.0, -8192.0]


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


def a_f32() -> np.ndarray:
    """The fp32 twin's [2, 512] dense tile: exactly-representable integers,
    each row its own pattern (row 0 = j, row 1 = 100000 + 3j — max 203533,
    every sum with the Col word < 2^24)."""
    out = np.zeros((ROWS, N), dtype=np.float32)
    for r in range(ROWS):
        out[r] = (np.arange(N) * (1 if r == 0 else 3) + (0 if r == 0 else 100000)).astype(np.float32)
    return out


def b_f32col() -> np.ndarray:
    """The fp32 twin's [2, 1] Col table at its IEEE_FP32 one-stick footprint:
    32 words per row, the row's own word in word 0 of its own stick."""
    stick = np.zeros((ROWS, 32), dtype=np.float32)
    for r, w in enumerate(F32_COL_WORDS):
        stick[r, 0] = np.float32(w)
    return stick


# The wide twin's width (auditor order 2026-10-27T02:5xZ): production PAD_V,
# 2048 out-sticks — the per-core slice width the card-proven b2 equality
# states (64 sticks per core at {mb:1, out:32}).
WIDE_N = 65536


def a_f32wide() -> np.ndarray:
    """The wide twin's [2, 65536] dense tile: the same exact-integer patterns
    as `a_f32` at production width (row 0 = j, row 1 = 100000 + 3j — max
    296605, every sum with the Col word < 2^24, exact on both sides)."""
    out = np.zeros((ROWS, WIDE_N), dtype=np.float32)
    for r in range(ROWS):
        out[r] = (np.arange(WIDE_N) * (1 if r == 0 else 3) + (0 if r == 0 else 100000)).astype(np.float32)
    return out


def tables() -> list[tuple[str, np.ndarray]]:
    """The staged files: every multi-row tile through `stick_scatter` (the
    device's own layout); the `[rows,1]` Col tables are scatter fixed points,
    so their bytes are unchanged by the law."""
    a = a_tile()
    dense = b_dense()
    col = b_col()
    out: list[tuple[str, np.ndarray]] = []
    for cell in ("intmr_dense_add", "intmr_dense_mul"):
        out.append((f"{cell}_t00_a", stick_scatter(a)))
        out.append((f"{cell}_t01_b", stick_scatter(dense)))
    for cell in ("intmr_col_add", "intmr_col_mul"):
        out.append((f"{cell}_t00_a", stick_scatter(a)))
        out.append((f"{cell}_t01_b", stick_scatter(col)))
    out.append(("intmr_f32col_add_t00_a", stick_scatter(a_f32())))
    out.append(("intmr_f32col_add_t01_b", stick_scatter(b_f32col())))
    out.append(("intmr_f32wide_add_t00_a", stick_scatter(a_f32wide())))
    out.append(("intmr_f32wide_add_t01_b", stick_scatter(b_f32col())))
    # The reduce-chain twin: the same wide tile (row maxima 65535 / 296605 —
    # DISTINCT, exact), fed to `max -> expand_dims -> add` — the card's own
    # writer->reader pair, now staged in the device's layout.
    out.append(("intmr_f32maxadd_t00_a", stick_scatter(a_f32wide())))
    return out


def main() -> int:
    INDIR.mkdir(exist_ok=True)
    for name, arr in tables():
        arr.tofile(INDIR / f"{name}.bin")
        print(f"{name}.bin {arr.dtype} {arr.shape} {arr.nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
