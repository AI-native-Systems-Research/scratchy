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

⛔ A VERDICT OVER NOTHING MUST FAIL: an empty join refuses with exit 2 and a per-side
listing of what each dir actually holds, because "0 compared" is a broken leg, not a pass.
Exit codes: 0 clean, 1 divergence found, 2 refusal (nothing compared / bad args).

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

# ⛔ THE TWO SIDES SPELL THE TID DIFFERENTLY, AND EACH SPELLING IS STRUCTURAL. The card's stem is
# `format!("..._t{id}", …)` where `id` is a `PlaceId` whose Display renders `t{tid}` — the literal
# `t` plus the rendering is DOUBLE-t: `prog{P}_seg{S}_tt{T}.bin`. The emulator writes a plain u64
# tid behind one `t`: `prog{P}_seg{S}_t{T}.bin`. The card's `PlaceId::Synth` renders `t{of}_{role}`
# (so `tt{of}_{role}`) and is card-only — the emulator dumps graph tids — so synth files join
# nothing but are LISTED rather than silently invisible. Synth roles appear both bare and under an
# `op{O}_` timeline prefix (`xn`/`xc`/`rs` measured on the j25 dump), so the synth pattern takes
# both; the LATEST pattern must NOT take the op-prefixed form, or a timeline copy would masquerade
# as a tensor's final value.
EMU_FILE = re.compile(r"prog\d+_seg\d+_t(\d+)\.bin$")
CARD_LATEST = re.compile(r"prog\d+_seg\d+_tt(\d+)\.bin$")
CARD_SYNTH = re.compile(r"(?:op\d+_)?prog\d+_seg\d+_tt(\d+)_[A-Za-z0-9]+\.bin$")
CARD_TL = re.compile(r"op(\d+)_prog\d+_seg\d+_tt(\d+)(?:_[A-Za-z0-9]+)?\.bin$")


def selftest():
    """Parse the EXACT filenames measured on the j25 dumps (nickm2 /work/runhelper-j25), so a
    spelling drift in either writer's format! fails here instead of silently emptying the join."""
    cases = [
        # (pattern, filename, expected capture or None for no-match)
        (CARD_LATEST, "prog0_seg0_tt1127.bin", "1127"),
        (CARD_LATEST, "prog0_seg0_tt4294967271.bin", "4294967271"),
        (CARD_LATEST, "op101_prog0_seg0_tt1127.bin", None),  # timeline copy is not a latest value
        (CARD_LATEST, "prog0_seg0_t1127.bin", None),  # single-t is the EMULATOR's spelling
        (CARD_TL, "op101_prog0_seg0_tt1127.bin", ("101", "1127")),
        (CARD_TL, "op5_prog0_seg3_tt9_xc.bin", ("5", "9")),
        (CARD_TL, "prog0_seg0_tt1127.bin", None),  # latest files carry no op number
        (CARD_SYNTH, "op3_prog0_seg2_tt7_rs.bin", "7"),
        (CARD_SYNTH, "prog0_seg2_tt7_xn.bin", "7"),
        (CARD_SYNTH, "prog0_seg2_tt7.bin", None),  # no role suffix: not a synth
        (EMU_FILE, "prog0_seg0_t1012.bin", "1012"),
        (EMU_FILE, "prog0_seg0_tt1012.bin", None),  # double-t is the CARD's spelling
        (EMU_FILE, "op1_prog0_seg0_t1012.bin", None),
    ]
    bad = 0
    for pat, name, want in cases:
        m = pat.match(name)
        got = None if m is None else (m.groups() if len(m.groups()) > 1 else m.group(1))
        if got != want:
            print(f"FAIL {pat.pattern!r} on {name!r}: got {got!r}, want {want!r}")
            bad += 1
    if bad:
        print(f"SELFTEST: {bad} failure(s)")
        sys.exit(1)
    print(f"SELFTEST: {len(cases)} filename cases parse")


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
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    card_dir, emu_dir = sys.argv[1], sys.argv[2]
    card = latest_by_tid(card_dir, CARD_LATEST)
    emu = latest_by_tid(emu_dir, EMU_FILE)
    card_synth = latest_by_tid(card_dir, CARD_SYNTH)
    first_write = first_write_by_tid(card_dir)
    labels = ops_labels(card_dir)

    both = sorted(set(card) & set(emu))
    card_only = sorted(set(card) - set(emu))
    emu_only = sorted(set(emu) - set(card))

    # ⛔ A VERDICT OVER NOTHING MUST FAIL. An empty join is not "no divergence": it is a leg that
    # compared nothing — a side that dumped no files, or filenames this parser does not read. Refuse
    # with what each side actually holds, so the failure names its own cause instead of passing.
    if not both:
        for what, d, parsed in (("card", card_dir, card), ("emu", emu_dir, emu)):
            entries = sorted(os.listdir(d)) if os.path.isdir(d) else []
            sample = ", ".join(entries[:8]) or "<empty or missing dir>"
            print(
                f"REFUSED: the join is EMPTY — {what} dir {d} parsed {len(parsed)} tensor(s) "
                f"of {len(entries)} entr(y|ies): {sample}"
            )
        print(
            "no tensor was compared, so there is no verdict — fix the dump or the parser, "
            "do not read this as a pass"
        )
        sys.exit(2)

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
    if card_synth:
        names = sorted(os.path.basename(p) for p in card_synth.values())
        print(f"card-only synth(s) (lowering-minted, no tid join by design): {names}")
    if emu_only:
        print(f"emu-only tid(s) (never reached on card): {emu_only}")
    if diverged:
        sys.exit(1)


if __name__ == "__main__":
    main()
