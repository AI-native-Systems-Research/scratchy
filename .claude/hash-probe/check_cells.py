#!/usr/bin/env python3
"""B1 card-run verdicts (nickm4), computed on the pod against the SAME
deterministic inputs build_cells.py generated (imported, not re-stated).

The bar, as ruled:
* mul/add legs (BOTH geometry legs): card u32 vs host wrapping i32
  (* / +) over every one of the 524,288 lanes — ZERO differing lanes is the
  only pass. g1 vs g2 divergence is itself the datum on whether dxp
  recomputes arrays.
* i2f_g1: card f32 bits vs host `as f32` of the signed interpretation —
  counted, zero required.
* control: the verbatim fp16 cell through the original IO path — small-int
  products exact in SEN169 and fp16, zero required. Proves the harness.
* log_g1: max ULP vs host ln() — A NUMBER, not pass/fail.

Every number is also written to cells/B1-verdict.json (a saved artifact).
Exit 0 iff control + mul/add (both legs) + i2f are all zero-diff; log never
gates.
"""
import json
import math
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_cells as B  # noqa: E402  (the generators ARE the input definition)

CELLS = B.CELLS
N = B.N_LANES
MASK = 0xFFFFFFFF


def read_words(path: pathlib.Path, fmt: str):
    data = path.read_bytes()
    n = len(data) // struct.calcsize(fmt)
    return struct.unpack(f"<{n}{fmt}", data[: n * struct.calcsize(fmt)])


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ulp_f32(x: float) -> float:
    bits = f32_bits(x)
    y = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    return max(abs(y - x), 2.0 ** -149)


def sample_mismatches(card, host, k=5):
    out = []
    for i, (c, h) in enumerate(zip(card, host)):
        if c != h:
            out.append((i, c, h))
            if len(out) >= k:
                break
    return out


def main() -> int:
    verdict = {"lanes": N}
    rc = 0

    def gate(name: str, diff: int, detail) -> None:
        nonlocal rc
        verdict[name] = {"diff": diff, "detail": detail}
        status = "PASS" if diff == 0 else "FAIL"
        print(f"VERDICT {name}: {status} diff={diff}/{N}")
        for i, c, h in detail:
            print(f"    lane {i}: card={c} host={h}")
        if diff != 0:
            rc = 1

    # ── control: fp16 small-int mul, bitwise ──
    p = CELLS / "out_control.bin"
    if not p.exists():
        print("VERDICT control: MISSING")
        verdict["control"] = {"missing": True}
        rc = 1
    else:
        a = read_words(CELLS / "in0_control.bin", "e")
        b = read_words(CELLS / "in1_control.bin", "e")
        o = read_words(p, "e")
        host = [x * y for x, y in zip(a, b)]  # ints < 512, exact in both formats
        gate("control", sum(1 for c, h in zip(o, host) if c != h),
             sample_mismatches(o, host))

    # ── mul / add, both geometry legs: wrapping i32, bitwise ──
    for base, fn in (("mul", lambda x, y: (x * y) & MASK),
                     ("add", lambda x, y: (x + y) & MASK)):
        for geo in ("g1", "g2"):
            leg = f"{base}_{geo}"
            p = CELLS / f"out_{leg}.bin"
            if not p.exists():
                print(f"VERDICT {leg}: MISSING")
                verdict[leg] = {"missing": True}
                rc = 1
                continue
            a = read_words(CELLS / f"in0_{leg}.bin", "I")
            b = read_words(CELLS / f"in1_{leg}.bin", "I")
            o = read_words(p, "I")
            host = [fn(x, y) for x, y in zip(a, b)]
            gate(leg, sum(1 for c, h in zip(o, host) if c != h),
                 sample_mismatches(o, host))

    # ── i2f_g1: signed interpretation -> f32, bitwise ──
    p = CELLS / "out_i2f_g1.bin"
    if not p.exists():
        print("VERDICT i2f_g1: MISSING")
        verdict["i2f_g1"] = {"missing": True}
        rc = 1
    else:
        a = read_words(CELLS / "in0_i2f_g1.bin", "I")
        o = read_words(p, "I")  # f32 read back as raw bits
        host = [f32_bits(float(x if x < 2**31 else x - 2**32)) for x in a]
        gate("i2f_g1", sum(1 for c, h in zip(o, host) if c != h),
             sample_mismatches(o, host))

    # ── log_g1: max ULP vs host ln() — a number, never a gate ──
    p = CELLS / "out_log_g1.bin"
    if not p.exists():
        print("VERDICT log_g1: MISSING (reported, not gated)")
        verdict["log_g1"] = {"missing": True}
    else:
        x = read_words(CELLS / "in0_log_g1.bin", "f")
        o = read_words(p, "f")
        max_ulp = 0.0
        exact = 0
        for xi, oi in zip(x, o):
            h = struct.unpack("<f", struct.pack("<f", math.log(xi)))[0]
            if oi == h:
                exact += 1
            u = abs(oi - h) / ulp_f32(h)
            if u > max_ulp:
                max_ulp = u
        verdict["log_g1"] = {"max_ulp": max_ulp, "exact": exact}
        print(f"VERDICT log_g1: max_ulp={max_ulp:.4f} exact={exact}/{N} (a number, not a gate)")

    (CELLS / "B1-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"B1-verdict.json written; rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
