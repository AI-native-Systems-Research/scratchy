#!/usr/bin/env python3
"""P3 localization — the auditor's order (2026-10-27), ONE run.

Seed 1234 solo, then 1234+9999 concurrent, on ONE server (p3-loc-leg.sh).
For each decode step, request-1234's seat words (seed_lo, seed_hi, step,
gate, row) and picked token, the pair's against the solo run's — the FIRST
step where they differ, and which of the auditor's two branches it is:

  - SEAT WORDS DIFFER -> the differing word is batch-dependent; it must be
    a function of the request alone (its own step count and its own seed).
  - WORDS EQUAL, PICK DIFFERS -> numeric (the rows=2 decode-batch rung is
    a different baked tape than the rows=1 solo one); the gate becomes
    'batched equals main's batched', not solo.

Launch geometry this log actually carries (measured, first run): a solo
request's decode steps run one launch each, ctx 'solo pos=N' (rows=1); a
concurrent pair runs each request's first decode steps solo, then shared
'decode-batch mq=2' launches — request 1234 at row 0, request 9999 at row
1 (the row word of the second request is 1 in the pair and 0 in any solo
run). Comparison is keyed on the step word (words[2], the request's own
absolute position), never on list index.

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
# ctx is free-form ("decode-batch mq=2 start=69", "prefill-final mq=71
# start=0", "solo pos=67"); the rows count is the first ' rows=N ' after it.
RB_RE = re.compile(r"SEAT_READBACK (.+?) rows=(\d+) ")
MQ_RE = re.compile(r"mq=(\d+)")
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
                # no rows — skipped; an empty sequence fails structurally
                continue
            ctx = m.group(1).strip()
            mq_m = MQ_RE.search(ctx)
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
                "ctx": ctx, "mq": int(mq_m.group(1)) if mq_m else None,
                "rows": int(m.group(2)), "per_row": per_row,
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
                            "bind_raw": ev["bind_raw"], "raw": ev["raw"],
                            **r})
    return out


def by_step(seq):
    """Step word -> entry. A step appearing twice (two launches at one
    position) is kept as a list and flagged by the caller."""
    out = {}
    for e in seq:
        out.setdefault(e["words"][2], []).append(e)
    return out


def ctx_win(seq, i, w=2):
    lo, hi = max(0, i - w), min(len(seq), i + w + 1)
    return [{"ctx": e["ctx"], "row": e["row"], "words": e["words"],
             "pick": e["pick"], "argmax": e["argmax"], "eq": e["eq"]}
            for e in seq[lo:hi]]


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

    solo_steps = by_step(solo)
    pair_steps = by_step(pair)
    dup = ([k for k, v in solo_steps.items() if len(v) > 1]
           + [k for k, v in pair_steps.items() if len(v) > 1])
    common = sorted(set(solo_steps) & set(pair_steps))
    only_solo = sorted(set(solo_steps) - set(pair_steps))
    only_pair = sorted(set(pair_steps) - set(solo_steps))

    first_words = first_pick = first_any = None
    member_diff_counts = {name: 0 for name in WORD_NAMES}
    pick_diff_count = 0
    pick_diff_steps = []
    for st in common:
        a = solo_steps[st][0]
        b = pair_steps[st][0]
        words_differ = a["words"] != b["words"]
        if words_differ:
            for k in range(min(5, len(a["words"]), len(b["words"]))):
                if a["words"][k] != b["words"][k]:
                    member_diff_counts[WORD_NAMES[k]] += 1
        pick_differ = a["pick"] != b["pick"]
        if pick_differ:
            pick_diff_count += 1
            pick_diff_steps.append(st)
        if words_differ and first_words is None:
            members = [WORD_NAMES[k]
                       for k in range(min(5, len(a["words"]), len(b["words"])))
                       if a["words"][k] != b["words"][k]]
            first_words = {"step": st, "members": members,
                           "solo_words": a["words"], "pair_words": b["words"],
                           "ctx": b["ctx"], "row": b["row"]}
        if pick_differ and first_pick is None:
            first_pick = {"step": st, "solo_pick": a["pick"],
                          "pair_pick": b["pick"],
                          "words_equal": not words_differ,
                          "solo_argmax": a["argmax"], "pair_argmax": b["argmax"],
                          "ctx": b["ctx"], "row": b["row"]}
        if (words_differ or pick_differ) and first_any is None:
            first_any = st

    if first_any is None:
        branch = ("no-divergence" if not only_solo and not only_pair
                  else "no-divergence-in-common-steps (step sets differ: "
                       f"solo-only {only_solo}, pair-only {only_pair})")
    elif first_words is not None and first_words["step"] == first_any:
        branch = "seat-words-differ"
    else:
        branch = "numeric"

    branch_text = {
        "seat-words-differ":
            "Seat words differ at the first divergence — the differing word "
            "is batch-dependent; it must be a function of the request alone "
            "(its own step count and its own seed).",
        "numeric":
            "Seat words are equal at every common step but the pick "
            "differs — numeric (the shared decode-batch mq=2 rung is a "
            "different baked tape than the solo rows=1 one); main's batched "
            "path shows the same flip class, and the gate becomes 'batched "
            "equals main's batched', not solo.",
    }.get(branch, "No divergence found in the common steps.")

    # The second request's seat in the pair — its row word vs any solo run's
    # (a solo launch is always row 0; this run's solo slice shows it).
    pair_b_rows = Counter(e["row"] for e in pair_b)
    pair_b_row_words = Counter(e["words"][4] for e in pair_b)

    def solo_index_of(st):
        return next((i for i, e in enumerate(solo) if e["words"][2] == st), None)

    def pair_index_of(st):
        return next((i for i, e in enumerate(pair) if e["words"][2] == st), None)

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
        "pair_1234_launch_kinds": {k: v for k, v in
                                   sorted(Counter(e["ctx"].split()[0]
                                                  for e in pair).items())},
        "pair_9999_rows": dict(pair_b_rows),
        "pair_9999_row_words": dict(pair_b_row_words),
        "common_steps": len(common),
        "solo_only_steps": only_solo,
        "pair_only_steps": only_pair,
        "duplicate_step_words": dup,
        "first_divergence_step": first_any,
        "first_words_diff": first_words,
        "first_pick_diff": first_pick,
        "member_diff_counts_over_common": member_diff_counts,
        "pick_diff_count_over_common": pick_diff_count,
        "pick_diff_steps": pick_diff_steps,
        "branch": branch,
        "branch_text": branch_text,
        "texts": {
            "solo_seed1234": text_of(SOLO_JSON),
            "pair_seed1234": text_of(PAIR1234_JSON),
            "pair_seed9999": text_of(PAIR9999_JSON),
        },
        "solo_context": (ctx_win(solo, solo_index_of(first_any))
                         if first_any is not None else []),
        "pair_context": (ctx_win(pair, pair_index_of(first_any))
                         if first_any is not None else []),
        "divergence_solo_readback_raw":
            solo[solo_index_of(first_any)]["raw"] if first_any is not None else None,
        "divergence_pair_readback_raw":
            pair[pair_index_of(first_any)]["raw"] if first_any is not None else None,
        "divergence_pair_bind_raw":
            pair[pair_index_of(first_any)]["bind_raw"] if first_any is not None else None,
    }
    out = HERE / "p3-localize-verdict.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"P3-LOCALIZE verdict -> {out}")
    print(f"  solo entries {len(solo)}, pair entries (1234) {len(pair)}, "
          f"common steps {len(common)}")
    print(f"  pair 1234 rows: {verdict['pair_1234_rows']}, "
          f"launch kinds: {verdict['pair_1234_launch_kinds']}")
    print(f"  pair 9999 rows: {dict(pair_b_rows)}, row words: {dict(pair_b_row_words)}")
    if only_solo or only_pair or dup:
        print(f"  step-set notes: solo-only {only_solo}, pair-only "
              f"{only_pair}, dup {dup}")
    if first_any is None:
        print(f"  NO DIVERGENCE in the common steps. {branch}")
    else:
        print(f"  first divergence at step {first_any} "
              f"(ctx '{pair[pair_index_of(first_any)]['ctx']}', "
              f"row {pair[pair_index_of(first_any)]['row']})")
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
        print(f"  member diff counts over {len(common)} common steps: "
              f"{member_diff_counts}; pick diffs: {pick_diff_count}")
        if pick_diff_steps:
            print(f"  pick-diff steps: {pick_diff_steps}")
    print(f"  BRANCH: {branch}")
    print(branch_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
