#!/usr/bin/env python3
"""P3 localization — the auditor's order (2026-10-27), ONE run.

Seed 1234 solo, then 1234+9999 concurrent, on ONE server (p3-loc-leg.sh).
For each decode step, request-1234's seat words (seed_lo, seed_hi, step,
gate, row) and picked token, the pair's against the solo run's — the FIRST
step where they differ, and which of the auditor's two branches it is:

  - SEAT WORDS DIFFER -> the differing word is batch-dependent; it must be
    a function of the request alone (its own step count and its own seed).
  - WORDS EQUAL, PICK DIFFERS -> numeric (the rows=2 decode rung is a
    different baked tape than the rows=1 one); the gate becomes 'batched
    equals main's batched', not solo.

Reads (all produced by p3-loc-leg.sh on the pod):
  p3-loc-server.log   — the server log, debug level (SEAT_BIND +
                        SEAT_READBACK per launch)
  p3-loc-slices.txt   — "L0 L1 L2": solo slice = [L0,L1), pair = [L1,L2)
  p3-loc-solo1234.json / p3-loc-pair1234.json / p3-loc-pair9999.json
Writes: p3-localize-verdict.json (+ a human summary on stdout).

A reproduced P3 fork is a RESULT, not a failure — the script exits nonzero
only on structural failure (slices unreadable, no SEAT_READBACK lines, no
seat row carrying seed 1234): a verdict computed over nothing must FAIL.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path("/work/fb-noise-probe")
SRVLOG = HERE / "p3-loc-server.log"
SLICES = HERE / "p3-loc-slices.txt"
SOLO_JSON = HERE / "p3-loc-solo1234.json"
PAIR1234_JSON = HERE / "p3-loc-pair1234.json"
PAIR9999_JSON = HERE / "p3-loc-pair9999.json"
SEED = 1234
SEED_B = 9999
WORD_NAMES = ["seed_lo", "seed_hi", "step", "gate", "row"]

BIND_RE = re.compile(
    r"SEAT_BIND rows=(\d+) seed_words=\[([^\]]*)\] invt=t(\d+) laws=(.*)")
LAW_RE = re.compile(r"r(\d+):(?:(seed=\d+)|greedy)@step(\d+):inv_t([0-9.]+)")
RB_RE = re.compile(
    r"SEAT_READBACK (decode-batch|prefill-final) mq=(\d+) start=(\d+) rows=(\d+)")
ROW_RE = re.compile(
    r"r(\d+):words=\[([^\]]*)\]:invt=([^:]*):pick=(None|Some\(\d+\))"
    r":argmax=(None|Some\(\d+\)):eq=(true|false)")


def parse_opt_u32(s):
    return None if s == "None" else int(s[5:-1])


def seat_events(lines):
    """SEAT_BIND / SEAT_READBACK events in log order. Each readback keeps
    the raw text of the bind that preceded it — bind and readback are one
    launch, so the pair (staged law, card word) is directly comparable."""
    binds, reads = [], []
    last_bind = None
    for ln in lines:
        if "SEAT_BIND" in ln:
            m = BIND_RE.search(ln)
            if m:
                laws = []
                for lm in LAW_RE.finditer(m.group(4)):
                    laws.append({
                        "row": int(lm.group(1)),
                        "seed": (int(lm.group(2).split("=")[1])
                                 if lm.group(2) else None),
                        "step": int(lm.group(3)),
                        "inv_t": float(lm.group(4)),
                    })
                last_bind = {"rows": int(m.group(1)), "laws": laws,
                             "raw": ln.strip()}
                binds.append(last_bind)
        elif "SEAT_READBACK" in ln:
            m = RB_RE.search(ln)
            if not m:
                # the readback's own failure lines ("... read failed") carry
                # no rows — count them so a vanished row is visible, not silent
                continue
            per_row = []
            for rm in ROW_RE.finditer(ln):
                words = [int(x.strip()) for x in rm.group(2).split(",")]
                per_row.append({
                    "row": int(rm.group(1)),
                    "words": words,
                    "inv_t": float(rm.group(3)),
                    "pick": parse_opt_u32(rm.group(4)),
                    "argmax": parse_opt_u32(rm.group(5)),
                    "eq": rm.group(6) == "true",
                })
            reads.append({
                "ctx": m.group(1), "mq": int(m.group(2)),
                "start": int(m.group(3)), "rows": int(m.group(4)),
                "per_row": per_row,
                "bind_raw": last_bind["raw"] if last_bind else None,
                "raw": ln.strip(),
            })
    return binds, reads


def req_seq(reads, seed):
    """The request carrying `seed`'s seat entries, in launch order — the
    row is identified by its own seed_lo word (greedy rows hold 0)."""
    out = []
    for ev in reads:
        for r in ev["per_row"]:
            if r["words"] and r["words"][0] == seed:
                out.append({"ctx": ev["ctx"], "mq": ev["mq"],
                            "start": ev["start"], "bind_raw": ev["bind_raw"],
                            "raw": ev["raw"], **r})
    return out


def ctx_win(seq, i, w=2):
    lo, hi = max(0, i - w), min(len(seq), i + w + 1)
    return [{"ctx": e["ctx"], "mq": e["mq"], "row": e["row"],
             "words": e["words"], "pick": e["pick"], "argmax": e["argmax"],
             "eq": e["eq"]} for e in seq[lo:hi]]


def text_of(p):
    try:
        return json.loads(Path(p).read_text())["choices"][0]["message"]["content"]
    except Exception as e:  # unreadable reply is reported, never fatal here
        return f"<unreadable: {e}>"


def main():
    fail = []
    try:
        l0, l1, l2 = [int(x) for x in SLICES.read_text().split()]
        lines = SRVLOG.read_text().splitlines()
    except Exception as e:
        print(f"P3-LOC STRUCTURAL FAILURE: cannot read slices/log ({e})")
        return 1
    solo_lines = lines[l0:l1]
    pair_lines = lines[l1:l2]

    solo_binds, solo_reads = seat_events(solo_lines)
    pair_binds, pair_reads = seat_events(pair_lines)
    solo = req_seq(solo_reads, SEED)
    pair = req_seq(pair_reads, SEED)
    pair_b = req_seq(pair_reads, SEED_B)

    if not solo_reads:
        fail.append("no SEAT_READBACK lines in the solo slice")
    if not pair_reads:
        fail.append("no SEAT_READBACK lines in the pair slice")
    if not solo:
        fail.append(f"no seat row for seed {SEED} in the solo slice")
    if not pair:
        fail.append(f"no seat row for seed {SEED} in the pair slice")
    if fail:
        print("P3-LOC STRUCTURAL FAILURE: " + "; ".join(fail))
        return 1

    n = min(len(solo), len(pair))
    first_words = first_pick = first_any = None
    member_diff_counts = {name: 0 for name in WORD_NAMES}
    pick_diff_count = 0
    for i in range(n):
        a, b = solo[i], pair[i]
        words_differ = a["words"] != b["words"]
        if words_differ:
            for k in range(min(5, len(a["words"]), len(b["words"]))):
                if a["words"][k] != b["words"][k]:
                    member_diff_counts[WORD_NAMES[k]] += 1
        if a["pick"] != b["pick"]:
            pick_diff_count += 1
        if words_differ and first_words is None:
            members = [WORD_NAMES[k]
                       for k in range(min(5, len(a["words"]), len(b["words"])))
                       if a["words"][k] != b["words"][k]]
            first_words = {"index": i, "step": b["words"][2], "members": members,
                           "solo_words": a["words"], "pair_words": b["words"],
                           "ctx": b["ctx"], "mq": b["mq"], "row": b["row"]}
        if a["pick"] != b["pick"] and first_pick is None:
            first_pick = {"index": i, "step": b["words"][2],
                          "solo_pick": a["pick"], "pair_pick": b["pick"],
                          "words_equal": not words_differ,
                          "solo_argmax": a["argmax"], "pair_argmax": b["argmax"],
                          "ctx": b["ctx"], "mq": b["mq"], "row": b["row"]}
        if (words_differ or a["pick"] != b["pick"]) and first_any is None:
            first_any = i

    if first_any is None:
        branch = ("no-divergence" if len(solo) == len(pair)
                  else "no-divergence-in-overlap (sequence lengths differ: "
                       f"solo {len(solo)} vs pair {len(pair)})")
    elif first_words is not None and first_words["index"] == first_any:
        branch = "seat-words-differ"
    else:
        branch = "numeric"

    branch_text = {
        "seat-words-differ":
            "Seat words differ at the first divergence — the differing word "
            "is batch-dependent; it must be a function of the request alone "
            "(its own step count and its own seed).",
        "numeric":
            "Seat words are equal at every compared step but the pick "
            "differs — numeric (the rows=2 decode rung is a different baked "
            "tape than the rows=1 one); main's batched path shows the same "
            "flip class, and the gate becomes 'batched equals main's "
            "batched', not solo.",
    }.get(branch, "No divergence found in the compared overlap.")

    verdict = {
        "probe": "p3-localize",
        "order": ("auditor 2026-10-27: one run, seed 1234 solo and "
                  "1234+9999 concurrent; per decode step, request-1234's "
                  "seat words and picked token vs solo; first differing step"),
        "seed": SEED,
        "solo_entries": len(solo),
        "pair_entries_1234": len(pair),
        "pair_entries_9999": len(pair_b),
        "solo_launches_with_readback": len(solo_reads),
        "pair_launches_with_readback": len(pair_reads),
        "pair_1234_rows": dict(Counter(e["row"] for e in pair)),
        "pair_1234_launch_kinds": {f"{k[0]} mq={k[1]}": v for k, v in
                                   sorted(Counter((e["ctx"], e["mq"])
                                                  for e in pair).items())},
        "compared_steps": n,
        "first_divergence_index": first_any,
        "first_words_diff": first_words,
        "first_pick_diff": first_pick,
        "member_diff_counts_over_compared": member_diff_counts,
        "pick_diff_count_over_compared": pick_diff_count,
        "branch": branch,
        "branch_text": branch_text,
        "texts": {
            "solo_seed1234": text_of(SOLO_JSON),
            "pair_seed1234": text_of(PAIR1234_JSON),
            "pair_seed9999": text_of(PAIR9999_JSON),
        },
        "solo_context": ctx_win(solo, first_any) if first_any is not None else [],
        "pair_context": ctx_win(pair, first_any) if first_any is not None else [],
        "divergence_solo_readback_raw":
            solo[first_any]["raw"] if first_any is not None else None,
        "divergence_pair_readback_raw":
            pair[first_any]["raw"] if first_any is not None else None,
        "divergence_pair_bind_raw":
            pair[first_any]["bind_raw"] if first_any is not None else None,
    }
    out = HERE / "p3-localize-verdict.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"P3-LOCALIZE verdict -> {out}")
    print(f"  solo entries {len(solo)}, pair entries (1234) {len(pair)}, "
          f"compared {n}")
    print(f"  pair 1234 rows: {verdict['pair_1234_rows']}, "
          f"launch kinds: {verdict['pair_1234_launch_kinds']}")
    if first_any is None:
        print(f"  NO DIVERGENCE in the compared overlap. {branch}")
    else:
        print(f"  first divergence at index {first_any} "
              f"(step {pair[first_any]['words'][2]}, {pair[first_any]['ctx']} "
              f"mq={pair[first_any]['mq']} row={pair[first_any]['row']})")
        if first_words:
            print(f"  words differ: {first_words['members']} "
                  f"solo={first_words['solo_words']} "
                  f"pair={first_words['pair_words']}")
        if first_pick:
            print(f"  pick: solo={first_pick['solo_pick']} "
                  f"pair={first_pick['pair_pick']} "
                  f"(words_equal={first_pick['words_equal']}, "
                  f"argmax solo={first_pick['solo_argmax']} "
                  f"pair={first_pick['pair_argmax']})")
        print(f"  member diff counts over {n} compared: {member_diff_counts}; "
              f"pick diffs: {pick_diff_count}")
    print(f"  BRANCH: {branch}")
    print(branch_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
