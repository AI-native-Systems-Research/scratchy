#!/usr/bin/env python3
"""Rebuild each capacity sub-campaign's port-remainder.json from the anchors already on the branch.

Run after any promote (driver.sh does). Without it a restart re-ports what already landed: the
schedules are static and crustify has no idea which units are done.

⛔ THIS WRITES ONLY port-remainder.json, A GENERATED ARTIFACT. It never edits Rust, never edits
port.json, and never edits UNITS.tsv — that file must stay byte-stable while a wave is in flight.

⛔ AN ANCHOR IS NOT A PORT. This script drops a unit from the remainder once its `/// Replaces:`
anchor exists, which is the only signal crustify can consume — it is NOT the acceptance gate. The
gate is: zero `todo!` for the arms in scope, a real non-test caller, and execution on a real program.
A unit whose anchor is present but whose body is a stub must be put BACK by hand.

Reads /tmp/filled-capacity.txt — one `eNNN_name` per line, produced by:
    grep -rhoE '/// Replaces: e0(0[2-9]|1[0-9])_[A-Za-z0-9_]+' crates/compiler/deeptools/src \
      | sed 's|/// Replaces: ||' | sort -u > /tmp/filled-capacity.txt
"""
import json
import pathlib

MAX_SYMS = 8  # port batches stay small: crustify promotes nothing until an agent completes

filled = set(open("/tmp/filled-capacity.txt").read().split())
base = pathlib.Path(__file__).resolve().parent

# ⛔ e009_calculate_padded and e010_primaryDimToVal_base_st are MUTUALLY RECURSIVE
# (dsc/dims.cpp:560 -> calculate_padded, dsc/dims.cpp:594 -> primaryDimToVal_base_st). They must
# never be split across two batches: whichever agent gets one alone cannot close its own callee.
INSEPARABLE = [{"e009_calculate_padded", "e010_primaryDimToVal_base_st"}]


def batch(units):
    """Cut `units` into batches of at most MAX_SYMS, never splitting an INSEPARABLE group."""
    out, cur = [], []
    for u in units:
        cur.append(u)
        names = {i["name"] for i in cur}
        pending = any(g & names and not g <= names for g in INSEPARABLE)
        if len(cur) >= MAX_SYMS and not pending:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


for d in [
    "sc1-level0-leaves",
    "sc2-levels1-2-dim-values",
    "sc3-levels3-6-capacity",
]:
    f = base / d / "port.json"
    s = json.load(open(f))
    waves = []
    for w in s["waves"]:
        units = [i for b in w["batches"] for i in b["items"] if i["name"] not in filled]
        if not units:
            continue
        proto = {k: v for k, v in w["batches"][0].items() if k != "items"}
        waves.append({
            "unit_count": len(units),
            "batches": [dict(proto, items=b) for b in batch(units)],
        })
    nb = sum(len(w["batches"]) for w in waves)
    nu = sum(w["unit_count"] for w in waves)
    s["waves"] = waves
    s["summary"].update(batch_count=nb, unit_count=nu, layer_count=len(waves))
    json.dump(s, open(f.with_name("port-remainder.json"), "w"), indent=1)
    print(f"  {d}: {nu} units left, {nb} batches")

print(f"filled anchors accounted for: {len(filled)} (an anchor is NOT the acceptance gate)")
