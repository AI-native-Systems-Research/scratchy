#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.2
"""First-divergence compare of a card `SCRATCHY_SDSC_TENSORDUMP` against an emulator
`SCRATCHY_EMU_TENSORDUMP` of the SAME prompt — the #717 card-vs-emu localization leg.

Usage: python3 scripts/td_first_divergence.py <card-dump-dir> <emu-dump-dir>

Join key: TENSOR ID. Element count comes from the file size (LE f32 throughout); the `.n`
sidecar is a human cross-check. The prog/seg numbers are NOT comparable across sides (the card's prog index is per-fingerprint,
the emulator's is its own program order), so the join is on tid alone. Values are LE f32 on
both sides (the card converts SEN169→f32 as it dumps; the emulator decodes its dtype).

Verdict law: `rel = |card - emu| / max(|emu|, 1e-6)`, tolerance 2e-2 — the numeric gate's
fp8 slack, so anything past it is a defect class the bake-time gate has convicted before.

Report:
  * every tid whose LATEST card value diverges from the emulator's latest, with the
    card op that first wrote it (timeline attribution via the `op{O}_…` copies) and the
    ops.txt label lines for those ops;
  * tids present on one side only, listed (host-bound/raw card-only tensors are expected
    and marked, not counted).

Accumulators (a tid written by several ops) compare FINAL-vs-FINAL like everything else;
their timeline attribution names the FIRST write, which for a fold/running-max is the op
that seeded the divergence.
"""
import math
import os
import re
import struct
import sys

TOL = 2e-2

EMU_FILE = re.compile(r"prog\d+_seg\d+_t(\d+)\.bin$")
CARD_LATEST = re.compile(r"prog\d+_seg\d+_t(\d+)\.bin$")
CARD_TL = re.compile(r"op(\d+)_prog\d+_seg\d+_t(\d+)\.bin$")


def read_f32(path):
    with open(path, "rb") as f:
        b = f.read()
    return struct.unpack(f"<{len(b) // 4}f", b)


def latest_by_tid(dump_dir, pattern):
    """tid -> (mtime, path) of the most recent matching file."""
    out = {}
    for name in os.listdir(dump_dir):
        m = pattern.match(name)
        if not m:
            continue
        tid = int(m.group(1))
        p = os.path.join(dump_dir, name)
        t = os.path.getmtime(p)
        if tid not in out or t > out[tid][0]:
            out[tid] = (t, p)
    return {tid: p for tid, (_, p) in out.items()}


def first_write_by_tid(dump_dir):
    """tid -> smallest op number that wrote it (timeline files)."""
    out = {}
    for name in os.listdir(dump_dir):
        m = CARD_TL.match(name)
        if not m:
            continue
        opno, tid = int(m.group(1)), int(m.group(2))
        out[tid] = min(opno, out.get(tid, opno))
    return out


def ops_labels(dump_dir):
    """op number -> its ops.txt line (may be absent)."""
    out = {}
    path = os.path.join(dump_dir, "ops.txt")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r"op(\d+)\s", line)
            if m:
                out[int(m.group(1))] = line.rstrip()
    return out


def verdict(card_v, emu_v):
    if len(card_v) != len(emu_v):
        return f"SHAPE card {len(card_v)} el vs emu {len(emu_v)} el"
    bad = 0
    worst = 0.0
    for c, e in zip(card_v, emu_v):
        if math.isnan(c) or math.isnan(e):
            bad += 1
            continue
        d = abs(c - e) / max(abs(e), 1e-6)
        if d > TOL:
            bad += 1
        worst = max(worst, d)
    if bad == 0:
        return None
    return (
        f"{bad}/{len(card_v)} el past rel {TOL} (worst {worst:.4g})\n"
        f"    card[:8] = {['%.5g' % x for x in card_v[:8]]}\n"
        f"    emu [:8] = {['%.5g' % x for x in emu_v[:8]]}"
    )


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    card_dir, emu_dir = sys.argv[1], sys.argv[2]
    card = latest_by_tid(card_dir, CARD_LATEST)
    emu = latest_by_tid(emu_dir, EMU_FILE)
    first_write = first_write_by_tid(card_dir)
    labels = ops_labels(card_dir)

    both = sorted(set(card) & set(emu))
    card_only = sorted(set(card) - set(emu))
    emu_only = sorted(set(emu) - set(card))

    diverged = []
    for tid in both:
        v = verdict(read_f32(card[tid]), read_f32(emu[tid]))
        if v:
            diverged.append((first_write.get(tid, -1), tid, v))
    diverged.sort()

    if not diverged:
        print(f"NO divergence past rel {TOL}: {len(both)} tid(s) compared")
    else:
        print(f"DIVERGENT tid(s): {len(diverged)} of {len(both)} compared, by first card write:")
        for opno, tid, v in diverged[:20]:
            print(f"  t{tid} (first card write op{opno}): {v}")
            if opno in labels:
                print(f"    ops.txt: {labels[opno]}")
        if len(diverged) > 20:
            print(f"  … and {len(diverged) - 20} more")

    if card_only:
        print(f"card-only tid(s) (host-bound/raw — no emu reference, expected): {card_only}")
    if emu_only:
        print(f"emu-only tid(s) (never reached on card): {emu_only}")
    if diverged:
        sys.exit(1)


if __name__ == "__main__":
    main()
