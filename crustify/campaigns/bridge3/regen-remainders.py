#!/usr/bin/env python3
"""Rebuild each sub-campaign's port-remainder.json from the anchors already filled on the branch.

⭐ LESSON 5. Run after any promote. Without it, a restart re-ports what already landed: the schedules
are static and crustify has no idea which units are done.

Reads /tmp/b3-filled.txt — one `eNNN_name` per line, produced by the driver's regen():
    grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' crates/compiler/deeptools/src \\
      | sed 's|/// Replaces: ||' | sort -u > /tmp/b3-filled.txt
"""
import json
import pathlib

MAX_SYMS = 8
MAX_LOC = 700

filled = set()
p = pathlib.Path("/tmp/b3-filled.txt")
if p.exists():
    filled = set(p.read_text().split())
base = pathlib.Path(__file__).resolve().parent

STAGES = [
    "levels0-1-leaves-and-instruction-builders",
    "levels2-3-computes-and-transfers",
    "levels4-7-uniform-regions-and-the-driver",
]


def chunk(items):
    """Same rule as gen_campaign3.py: batch on BOTH the symbol count and the LoC."""
    out, cur, loc = [], [], 0
    for it in items:
        if cur and (len(cur) >= MAX_SYMS or loc + it.get("loc", 0) > MAX_LOC):
            out.append(cur)
            cur, loc = [], 0
        cur.append(it)
        loc += it.get("loc", 0)
    if cur:
        out.append(cur)
    return out


for d in STAGES:
    f = base / d / "port.json"
    if not f.exists():
        continue
    s = json.load(open(f))
    waves = []
    for w in s["waves"]:
        units = [i for b in w["batches"] for i in b["items"] if i["name"] not in filled]
        if not units:
            continue
        proto = {k: v for k, v in w["batches"][0].items() if k != "items"}
        waves.append({
            "unit_count": len(units),
            "batches": [dict(proto, items=b) for b in chunk(units)],
        })
    nb = sum(len(w["batches"]) for w in waves)
    nu = sum(w["unit_count"] for w in waves)
    s["waves"] = waves
    s["summary"].update(batch_count=nb, unit_count=nu, layer_count=len(waves))
    json.dump(s, open(f.with_name("port-remainder.json"), "w"), indent=2)
    print(f"  {d}: {nu} units left, {nb} batches")

print(f"filled anchors accounted for: {len(filled)}")
