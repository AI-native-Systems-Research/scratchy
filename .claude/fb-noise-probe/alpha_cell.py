#!/usr/bin/env python3
"""The α CELL (front-b-noise convert-door wall, auditor order 2026-10-27):
int32tofp32 over [2, 32] i32 — ONE STICK PER ROW — bit-for-bit against host
`as f32`, through the scatter harness.

THE QUESTION: `convert_opspec` hardcodes `StickExtent::<Fp16>` (a 64-ELEMENT
multiple, citing the dxp scheduler) while its own comment states the slab law
as "a whole stick of the WIDER of the two formats" — 32 elems for the i32→f32
pair. Every convert ever emitted was full-row-width (a 64-multiple at any
basis), so the two never disagreed until the sampler's [B,1] gate chain. The
device's law for a 32-element slab is not decidable from the citations in our
tree — this cell asks the card directly.

THE METHOD (B1's `i2f_g1` recipe, on a base our door CAN bake): bake the
dense i32 add at [2, 32] (`intmr_alpha_base` — the same pid-cornered spelling
as every intmr cell), then mutate the baked SDSC OFF-TREE: opFuncName
`addi32toi32` → `int32tofp32`, inputLabeledDs → Tensor0 alone, Tensor2 →
IEEE_FP32. Folds, start addresses and placements verbatim (B1's g1 lesson:
the minimal mutation is the faithful one). The door cannot bake the convert
at this width — that refusal IS the wall — so the base bakes the geometry and
the mutation swaps only the op.

THE CONTROL: the unmutated base add runs on the card FIRST — same geometry,
same staging — so a dxp refusal or run failure on the i2f cell is the
CONVERT's slab law, never the one-stick-per-row geometry itself.

THE STAGING (the scatter harness's own law, forward_tape.rs:299): a [2, 32]
u32 tile is ONE 128 B stick per row — a fixed point of the stick scatter —
but every input stages through `stick_scatter` and every output un-scatters
through `stick_unscatter` anyway, so the law is stated, not assumed.

THE HOST LAW: the card's int32tofp32 interprets SENUINT32 words SIGNED
(B1's i2f law, measured bit-exact across full-range u32 at w2048):
expected = x.view(int32).astype(float32) — IEEE round-to-nearest on both
sides, so ANY mismatch is a device defect, never numerics.

Inputs: the B1 generator's wraparound classes (near ±2^31, small ints,
~2^30, near −1, full-range random), 64 words, deterministic — the stats
subcommand IMPORTS this module, so the host compare and the card run consume
the SAME statement of the inputs.
"""
import json
import pathlib
import random
import shutil
import struct
import sys

import numpy as np

ROWS = 2
WIDTH = 32  # one 128 B stick per row at u32/f32 — the cell's whole point
WORDS_PER_STICK = 32

HERE = pathlib.Path(__file__).resolve().parent
INDIR = HERE / "in"

# The base bake (the door's own emission) and the mutated cell dir.
BAKE = pathlib.Path("/work/int-alpha-base")
CELL = pathlib.Path("/work/alpha-i2f")


def gen_words(seed: int, n: int) -> np.ndarray:
    """The B1 wraparound classes, deterministic: near ±2^31, small ints,
    ~2^30, near −1, small negative, full-range random — every conversion
    rounding class on card."""
    r = random.Random(seed)
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        m = i % 8
        if m == 0:
            v = 0x7FFFFFF0 + (i & 0xF)       # near INT32_MAX
        elif m == 1:
            v = 0x80000000 + (i & 0xF)       # near INT32_MIN
        elif m == 2:
            v = i & 0xFF                     # small positive
        elif m == 3:
            v = 0xFFFFFF00 + (i & 0xF)       # near -1
        elif m == 4:
            v = 0x40000000 + (i & 0xFFFF)    # ~2^30
        elif m == 5:
            v = (i & 0xF) | 0x80000000       # small negative
        else:
            v = r.getrandbits(32)            # full-range random
        out[i] = np.uint32(v)
    return out


def a_words() -> np.ndarray:
    """The i2f cell's input: row r carries seed (23 + r), so the two rows
    share no word."""
    return np.stack([gen_words(23 + r, WIDTH) for r in range(ROWS)])


def b_words() -> np.ndarray:
    """The control add's second operand: row r carries seed (37 + r)."""
    return np.stack([gen_words(37 + r, WIDTH) for r in range(ROWS)])


def stick_scatter(arr: np.ndarray) -> np.ndarray:
    """The device's stick-scattered stage (forward_tape.rs:299): stick k of
    row r at flat stick k*rows + r. A one-stick-per-row tile is its own
    scatter — stated through the helper regardless."""
    rows, width = arr.shape
    sticks = arr.reshape(rows, width // WORDS_PER_STICK, WORDS_PER_STICK)
    return sticks.transpose(1, 0, 2).reshape(-1)


def stick_unscatter(flat: np.ndarray, rows: int, width: int) -> np.ndarray:
    """The inverse: flat stick k*rows + r back to [rows, width] row-major."""
    sticks = flat.reshape(width // WORDS_PER_STICK, rows, WORDS_PER_STICK)
    return sticks.transpose(1, 0, 2).reshape(rows, width)


def stage() -> int:
    INDIR.mkdir(exist_ok=True)
    for name, arr in (("alpha_t00_a", a_words()), ("alpha_t01_b", b_words())):
        scattered = stick_scatter(arr)
        scattered.tofile(INDIR / f"{name}.bin")
        print(f"{name}.bin {arr.dtype} {arr.shape} -> {scattered.nbytes} B")
    return 0


def inner(doc: dict) -> dict:
    return next(iter(next(iter(doc.values()))["dscs_"][0].values()))


def mutate(bake: str, cell: str) -> int:
    """The B1 i2f_g1 recipe on the [2, 32] base bake: swap the op to
    int32tofp32, drop Tensor1 from the op's input list, restyle Tensor2 to
    IEEE_FP32. Everything else verbatim; every expectation asserted loudly so
    a changed base emission REFUSES instead of mutating the wrong thing."""
    bake, cell = pathlib.Path(bake), pathlib.Path(cell)
    doc = json.loads((bake / "sdsc_0.json").read_text())
    node = inner(doc)
    cop = node["computeOp_"][0]
    assert cop["opFuncName"] == "addi32toi32", cop["opFuncName"]
    assert cop["inputLabeledDs"] == ["Tensor0-idx0", "Tensor1-idx1"], cop["inputLabeledDs"]
    assert cop["attributes_"]["dataFormat_"] == "IEEE_FP32", cop["attributes_"]
    for ds in node["labeledDs_"]:
        assert ds["wordLength"] == 4 and ds["dataFormat_"] == "SENUINT32", ds
    assert node["primaryDsInfo_"]["OUTPUT"]["stickSize_"] == [32], node["primaryDsInfo_"]

    # The mutation — three fields, nothing else.
    cop["opFuncName"] = "int32tofp32"
    cop["inputLabeledDs"] = ["Tensor0-idx0"]
    for ds in node["labeledDs_"]:
        if ds["dsName_"] == "Tensor2":
            ds["dataFormat_"] = "IEEE_FP32"  # wl stays 4 — f32 is wl4 too

    shutil.rmtree(cell, ignore_errors=True)
    shutil.copytree(bake, cell)
    (cell / "sdsc_0.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    # bundle.mlir + placements.json ride the copy verbatim (Tensor1 stays
    # allocated but unread — B1's own spelling).
    print(f"mutated {bake} -> {cell}: int32tofp32, Tensor0 -> Tensor2 @IEEE_FP32")
    return 0


def read_out(name: str) -> np.ndarray:
    flat = np.fromfile(HERE / name, dtype=np.float32)
    return stick_unscatter(flat, ROWS, WIDTH)


def check() -> int:
    a, b = a_words(), b_words()

    # The control: wrapping i32 add, exact on both sides.
    cout = read_out("out_alpha_control.bin").astype(np.int32)
    cexp = (a.astype(np.int32) + b.astype(np.int32)).astype(np.int32)
    cok = bool((cout == cexp).all())
    verdict = {
        "cell": "alpha_i2f",
        "question": "int32tofp32 over [2,32] i32, one stick per row — "
                    "device-legal at a 32-element slab?",
        "geometry": {"rows": ROWS, "width": WIDTH, "bytes_per_row": WIDTH * 4},
        "control": {
            "op": "addi32toi32 (the unmutated base)",
            "pass": cok,
            "mismatches": int((cout != cexp).sum()),
        },
    }

    # The cell: host `as f32`, SIGNED interpretation (B1's i2f law).
    out = read_out("out_alpha_i2f.bin")
    exp = a.view(np.int32).astype(np.float32)
    ok = bool((out == exp).all())
    bad = np.argwhere(out != exp)
    verdict["i2f"] = {
        "op": "int32tofp32 (the mutated cell)",
        "pass": ok,
        "mismatches": int(bad.shape[0]),
        "first": None if ok else [
            {"r": int(r), "j": int(j),
             "in_i32": int(a.view(np.int32)[r, j]),
             "expected_f32": float(exp[r, j]), "got_f32": float(out[r, j]),
             "expected_bits": format(np.asarray(exp[r, j]).view(np.uint32), "08x"),
             "got_bits": format(np.asarray(out[r, j]).view(np.uint32), "08x")}
            for r, j in bad[:8]
        ],
    }
    verdict["verdict"] = "PASS" if (ok and cok) else "FAIL"

    (HERE / "alpha-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["verdict"] == "PASS" else 1


def fail(stage_name: str, evidence: str) -> int:
    """A refusal/failure BEFORE the compare is itself the verdict — recorded,
    never silently skipped."""
    verdict = {
        "cell": "alpha_i2f",
        "question": "int32tofp32 over [2,32] i32, one stick per row — "
                    "device-legal at a 32-element slab?",
        "verdict": "FAIL",
        "failed_at": stage_name,
        "evidence": evidence.strip()[-2000:],
    }
    (HERE / "alpha-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 1


def main(argv: list[str]) -> int:
    if argv[1:] == ["stage"]:
        return stage()
    if len(argv) == 4 and argv[1] == "mutate":
        return mutate(argv[2], argv[3])
    if argv[1:] == ["check"]:
        return check()
    if len(argv) >= 4 and argv[1] == "fail":
        return fail(argv[2], " ".join(argv[3:]))
    print(__doc__)
    print("usage: alpha_cell.py stage | mutate <bake> <cell> | check | "
          "fail <stage> <evidence...>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
