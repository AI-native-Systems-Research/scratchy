#!/usr/bin/env python3
"""The int-bcast probe's CARD VERDICT (front-b-noise, nickm2 leg).

The auditor's ask, as ruled (b): "Compare it against the host bit for bit."
Wrapping i32 is exact on both sides, so the verdict is ONE number — the count
of lanes where the card's `[1, 512]` sum disagrees with the host's
`(a[j] + b[0]) mod 2^32`. ANY mismatch is an addressing defect in the
Col-broadcast operand mode (a wrong lane, a wrong stride, a sprayed word read
from the wrong stick), never numerics. The first mismatching lanes are printed
with both words so the card bytes can be read straight off the log if the leg
fails.

Every number is written to intb-verdict.json beside this script. Exit 0 iff
the mismatch count is zero.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intb_inputs as G  # noqa: E402  (the generators ARE the input definition)

IN = G.INDIR


def read_u32(name: str) -> np.ndarray:
    return np.fromfile(IN / f"{name}.bin", dtype=np.uint32)


def main() -> int:
    a = read_u32("t00_a")
    b_stick = read_u32("t01_b")
    # The card run's read-back, written beside this script by the leg.
    o = np.fromfile(HERE / "out_o.bin", dtype=np.uint32)
    b_word = b_stick[0]

    want = (a + np.uint32(b_word)).astype(np.uint32)
    diff = o != want
    mismatches = int(diff.sum())

    report = {
        "probe": "int_bcast",
        "form": "addi32toi32, [1,512] dense SENUINT32 + [1,1] Col-broadcast",
        "lanes": int(a.size),
        "b_word": f"0x{b_word:08X}",
        "mismatches": mismatches,
        "pass": mismatches == 0,
    }
    if mismatches:
        idx = np.nonzero(diff)[0][:10]
        report["first_mismatches"] = [
            {
                "lane": int(i),
                "card": f"0x{o[i]:08X}",
                "host": f"0x{want[i]:08X}",
                "a": f"0x{a[i]:08X}",
            }
            for i in idx
        ]

    out = HERE / "intb-verdict.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if mismatches:
        print(f"INTB-FAILED ({mismatches} differing lanes of {a.size})")
        return 1
    print(f"INTB-DONE ({a.size} lanes, bit-for-bit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
