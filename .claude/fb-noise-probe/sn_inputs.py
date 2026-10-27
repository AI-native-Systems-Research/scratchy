#!/usr/bin/env python3
"""The sample-noise probe's STAGED INPUT TABLES and HOST REFERENCE
(front-b-noise, nickm2 leg — auditor ruling (a): g and score off the card).

The probe kernel (`crates/triton/test-fixtures/sample_noise_probe.py`) is the
PRODUCTION noise-and-score section of `sample_fwd.py` verbatim; this module
states its inputs and the host emulation of everything through `u` (exact by
representability — wrapping i32 and exact f32 scalings) and of `g`/`score`
(fp16/f32 arithmetic; the FEST log alone is unemulable, so `g`/`score`
compares are reported, never gated bit-for-bit).

DETERMINISTIC, NO RNG: every row's seeds, logits and constants are arithmetic
patterns, and this module is IMPORTED by the stats script, so the host compare
and the card run consume the SAME statement of the inputs (the B1 leg's own
law: the generators ARE the input definition).

⭐ THE DEVICE LAYOUT (the intmr kit's measured law, forward_tape.rs:299): a
`[rows, width]` tensor stages STICK-MAJOR — stick k of row r at flat stick
`k*rows + r` — so the logits tile is staged through that scatter and the six
outputs are UN-SCATTERED from it on read. The `[rows, 1]` tables are fixed
points of the scatter (one 128 B stick per row — fp32/u32 32 words, fp16 64
slots — this door's own `synth_footprint` law), each row's word in word 0 of
its own stick; the bake's `placements.json` states every size and the binds
helper verifies each one, so the staging and the door cannot disagree. The
`[1, PAD_V]` shared tables (lane, padoff) are rows=1 — the scatter is the
identity.

⭐ THE GEOMETRY IS GRANITE-3.1-2B'S OWN: V 49155, PAD_V 65536 — the vocab
the chat leg serves, so the probe's noise is the production noise at the
production width. `sgate` is 1 on every row (the SAMPLED spelling — the
defect's own case; a greedy row's gate 0 makes pick == argmax correct).

⭐ THE LOGITS CARRY NEAR-TIE PEAKS: each row's top four lanes sit at
200.0, 199.75, 199.5, 199.25 at a row-distinct position, over a ±76.5
background — all exactly representable fp16. A LIVE Gumbel noise (σ ≈ 1.28)
moves the argmax on most rows; a lane-CONSTANT noise (the Finding-1
signature: zeroed `[rows, 1]` constants collapse the hash) can never move it.
`argmax(score) != argmax(x)` per row is the probe's direct measurement that
the noise is live.

Files land in ./in/ beside this script (never /tmp), per case (`b2` the
intmr-proven control width, `b8` the batched defect geometry), in the kernel's
parameter order.
"""
import pathlib

import numpy as np

V = 49155  # granite-3.1-2b-instruct's vocab — the pad law's real-lane bound
PAD_V = 65536  # sample_pad_vocab(49155)
CASES = {"b2": 2, "b8": 8}

# The stick granules: 128 B per stick — 64 fp16 slots or 32 fp32/u32 words.
F16_STICK = 64
F32_STICK = 32

# The noise section's constants — the same words the bake stages.
HC1 = 0x9E3779B1
HC2 = 0x85EBCA77
HC3 = 0xC2B2AE3D
C256 = 256
FM8 = 2.0**-8
FTOP = 8388607.0
FZERO = 0.0
FHALF = 0.5
FM22 = 2.0**-22
CMIN = 2.0**-14  # fp16's smallest normal — the FEST-domain clamp

HERE = pathlib.Path(__file__).resolve().parent
INDIR = HERE / "in"
OUTDIR = HERE


def stick_scatter(arr: np.ndarray, eps: int) -> np.ndarray:
    """The device's stick-scattered stage of a `[rows, width]` tile
    (forward_tape.rs:299): stick k of row r lands at flat stick `k*rows + r`."""
    rows, width = arr.shape
    sticks = arr.reshape(rows, width // eps, eps)
    return sticks.transpose(1, 0, 2).reshape(-1)


def stick_unscatter(flat: np.ndarray, rows: int, width: int, eps: int) -> np.ndarray:
    """The inverse: flat stick `k*rows + r` back to `[rows, width]` row-major."""
    sticks = flat.reshape(width // eps, rows, eps)
    return sticks.transpose(1, 0, 2).reshape(rows, width)


def seed_words(rows: int) -> dict[str, np.ndarray]:
    """The per-row seed QUINTUPLE — every row DIFFERENT, so a broadcast defect
    that sprays row 0's words across row 1, or reads either row at the wrong
    stick, lands on a u mismatch, never a coincidental pass. `sgate` is 1 on
    every row (the sampled spelling)."""
    r = np.arange(rows, dtype=np.uint64)
    return {
        "slo": ((0x12345678 + 0x01010101 * r) & 0xFFFFFFFF).astype(np.uint32),
        "shi": ((0x9ABCDEF0 - 0x00FEFEFE * r) & 0xFFFFFFFF).astype(np.uint32),
        "stp": (r + 1).astype(np.uint32),
        "sgate": np.full(rows, 1, dtype=np.uint32),
        "srow": r.astype(np.uint32),
    }


def logits(rows: int) -> np.ndarray:
    """The `[rows, PAD_V]` fp16 logits tile: a ±76.5 background in exact 0.25
    steps, each row's top four lanes a near-tie peak (200.0, 199.75, 199.5,
    199.25) at a row-distinct position under V — exact fp16 throughout, so the
    host compare carries no rounding of its own."""
    x = np.zeros((rows, PAD_V), dtype=np.float16)
    j = np.arange(PAD_V)
    for r in range(rows):
        base = (((j * 37 + r * 101) % 613) * 0.25 - 76.5).astype(np.float16)
        p = (r * 7919 + 1234) % 40000
        for k in range(4):
            base[p + k] = np.float16(200.0 - 0.25 * k)
        x[r] = base
    return x


def padoff_row() -> np.ndarray:
    """The pad law's offset row: 0.0 on the real lanes, −FLT_MAX on the pads
    (never IEEE −inf — an exponent-all-ones lane reads as NaN on the card)."""
    row = np.zeros(PAD_V, dtype=np.float32)
    row[V:] = np.float32(np.finfo(np.float32).min)
    return row


def lane_row() -> np.ndarray:
    """The per-lane index 0..PAD_V-1 as f32 — exact below 2^24."""
    return np.arange(PAD_V, dtype=np.float32)


def col_table(words: np.ndarray, dtype) -> np.ndarray:
    """A `[rows, 1]` table at its one-stick-per-row footprint (fp32/u32: 32
    words per row): the row's word in word 0 of its own stick, zeros elsewhere.
    A fixed point of the stick scatter."""
    rows = len(words)
    t = np.zeros((rows, F32_STICK), dtype=dtype)
    t[:, 0] = words.astype(dtype)
    return t


def cmin_table(rows: int) -> np.ndarray:
    """The fp16 `[rows, 1]` FEST clamp at THIS DOOR's fp16 one-stick-per-row
    footprint (synth_footprint_bytes: 64 fp16 slots = 128 B per row — the
    bake's own placements.json states it, and the binds helper verifies): 2^-14
    in word 0 of each row's own stick."""
    t = np.zeros((rows, F16_STICK), dtype=np.float16)
    t[:, 0] = np.float16(CMIN)
    return t


# ── The host reference ──────────────────────────────────────────────────────


def emulate_u(slo: int, shi: int, stp: int, srow: int) -> np.ndarray:
    """The uniform `u`, BIT-EXACT: the counter hash in wrapping u32, the
    wrap-fold, the abs fold and the strict-interior folds — every op exact by
    representability, so numpy f32 at the card's own op sequence reproduces
    the card bit for bit (the free check of the whole integer chain + every
    `[rows, 1]` staging: a wrong stick-head word or a wrong row's seed lands
    on a named mismatch)."""
    with np.errstate(over="ignore"):
        lane = np.arange(PAD_V, dtype=np.uint32)
        u32 = np.uint32
        h = lane * u32(HC1) + u32(slo)
        h = h * u32(HC2) + u32(shi) + u32(stp) + u32(srow)
        h = h * h
        h = h * u32(HC1) + u32(slo)
        h = h * h
        h = h * u32(HC3) + u32(shi) + u32(stp)
        h = h * h
        h = h * u32(HC1) + lane * u32(HC2) + u32(slo)
        r = (h * u32(C256)).view(np.int32)
    rf = r.astype(np.float32)
    a = np.abs(rf)
    n = a * np.float32(FM8)
    t = np.float32(FTOP) - n
    nm = np.minimum(n, t)
    np_ = np.maximum(nm, np.float32(FZERO))
    return (np_ + np.float32(FHALF)) * np.float32(FM22)


def emulate_g(u: np.ndarray) -> np.ndarray:
    """The Gumbel draw `g`, fp16-EMULATED (the card's FEST log is the one
    unemulable op — numpy's log rounded through the same fp16 island; the
    fb-noise probe measured mean |diff| ≈ 4e-3, max ≈ 2.6 on tail lanes).
    Reported, never gated bit-for-bit."""
    u16 = u.astype(np.float16)
    c1 = np.maximum(u16, np.float16(CMIN))
    lu = np.log(c1.astype(np.float32)).astype(np.float16)
    v = -lu
    c2 = np.maximum(v, np.float16(CMIN))
    lt = np.log(c2.astype(np.float32)).astype(np.float16)
    return (-lt).astype(np.float32)


def emulate_score(x: np.ndarray, g: np.ndarray, invt: float = 1.0,
                  gate: float = 1.0) -> np.ndarray:
    """The score, the production's own THREE-rounding sequence: x32*invt,
    noise*gate, the add, then the pad law's +padoff — all f32."""
    x32 = x.astype(np.float32)
    t1 = x32 * np.float32(invt)
    t2 = g * np.float32(gate)
    s = t1 + t2
    return s + padoff_row()


# ── The staged files and the expected-placement table ───────────────────────


def staged(case: str) -> list[tuple[str, np.ndarray]]:
    """The staged files in the kernel's PARAMETER ORDER, each already in the
    device's own layout (scattered tiles, stick-head word tables)."""
    rows = CASES[case]
    sw = seed_words(rows)
    x = logits(rows)
    tables: list[tuple[str, np.ndarray]] = [
        (f"sn_{case}_logits", stick_scatter(x, F16_STICK)),
        (f"sn_{case}_slo", col_table(sw["slo"], np.uint32)),
        (f"sn_{case}_shi", col_table(sw["shi"], np.uint32)),
        (f"sn_{case}_stp", col_table(sw["stp"], np.uint32)),
        (f"sn_{case}_sgate", col_table(sw["sgate"], np.uint32)),
        (f"sn_{case}_srow", col_table(sw["srow"], np.uint32)),
        (f"sn_{case}_invt", col_table(np.full(rows, 1.0), np.float32)),
        (f"sn_{case}_ones", col_table(np.full(rows, 1.0), np.float32)),
        (f"sn_{case}_padoff", padoff_row()),
        (f"sn_{case}_hc1", col_table(np.full(rows, HC1), np.uint32)),
        (f"sn_{case}_hc2", col_table(np.full(rows, HC2), np.uint32)),
        (f"sn_{case}_hc3", col_table(np.full(rows, HC3), np.uint32)),
        (f"sn_{case}_c256", col_table(np.full(rows, C256), np.uint32)),
        (f"sn_{case}_lane", lane_row()),
        (f"sn_{case}_fm8", col_table(np.full(rows, FM8), np.float32)),
        (f"sn_{case}_ftop", col_table(np.full(rows, FTOP), np.float32)),
        (f"sn_{case}_fzero", col_table(np.full(rows, FZERO), np.float32)),
        (f"sn_{case}_fhalf", col_table(np.full(rows, FHALF), np.float32)),
        (f"sn_{case}_fm22", col_table(np.full(rows, FM22), np.float32)),
        (f"sn_{case}_cmin", cmin_table(rows)),
    ]
    return tables


def expected(case: str) -> list[dict]:
    """The kernel's tensors in PARAMETER ORDER with the placement size each
    must carry — the binds helper verifies every one against the bake's own
    placements.json before a single byte is staged, so a door that numbers or
    sizes differently FAILS LOUDLY (never a silent misbind)."""
    rows = CASES[case]

    def ins(name: str, size: int) -> dict:
        return {"name": name, "size": size, "out": False}

    def out(name: str, size: int) -> dict:
        return {"name": name, "size": size, "out": True}

    col = rows * 128  # the [rows, 1] fp32/u32 one-stick-per-row footprint
    wide32 = rows * PAD_V * 4  # a [rows, PAD_V] fp32/i32 row-block tile
    return [
        ins("logits", rows * PAD_V * 2),
        ins("slo", col), ins("shi", col), ins("stp", col),
        ins("sgate", col), ins("srow", col),
        ins("invt", col),
        ins("ones", col),
        ins("padoff", PAD_V * 4),
        ins("hc1", col), ins("hc2", col), ins("hc3", col), ins("c256", col),
        ins("lane", PAD_V * 4),
        ins("fm8", col), ins("ftop", col), ins("fzero", col),
        ins("fhalf", col), ins("fm22", col),
        ins("cmin", rows * 128),  # this door's fp16 [rows, 1] footprint: 1 stick/row
        out("u", wide32), out("g", wide32), out("score", wide32),
        out("ui", wide32), out("gi", wide32), out("si", wide32),
    ]


def main() -> int:
    INDIR.mkdir(exist_ok=True)
    for case in CASES:
        for name, arr in staged(case):
            arr.tofile(INDIR / f"{name}.bin")
            print(f"{name}.bin {arr.dtype} {arr.shape} {arr.nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
