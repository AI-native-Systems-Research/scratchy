#!/usr/bin/env python3
"""Rebuild each sub-campaign's port-remainder.json from the anchors already filled on the branch.

Run after any promote. Without it, a restart re-ports what already landed: the schedules are static
and crustify has no idea which units are done.

Reads /tmp/filled.txt — one `eNNN_name` per line, produced by:
    grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' crates/compiler/deeptools/src \
      | sed 's|/// Replaces: ||' | sort -u > /tmp/filled.txt
"""
import json
import pathlib

filled = set(open("/tmp/filled.txt").read().split())
base = pathlib.Path(__file__).resolve().parent

for d in [
    "level0-accessors-and-leaves",
    "levels1-2-transfer-and-compute",
    "levels3-7-statements-and-passes",
    "levels8-10-pass-drivers",
]:
    f = base / d / "port.json"
    s = json.load(open(f))
    waves = []
    for w in s["waves"]:
        units = [i for b in w["batches"] for i in b["items"] if i["name"] not in filled]
        if not units:
            continue
        proto = {k: v for k, v in w["batches"][0].items() if k != "items"}
        waves.append(
            {
                "unit_count": len(units),
                "batches": [
                    dict(proto, items=units[i : i + 8]) for i in range(0, len(units), 8)
                ],
            }
        )
    nb = sum(len(w["batches"]) for w in waves)
    nu = sum(w["unit_count"] for w in waves)
    s["waves"] = waves
    s["summary"].update(batch_count=nb, unit_count=nu, layer_count=len(waves))
    json.dump(s, open(f.with_name("port-remainder.json"), "w"), indent=2)
    print(f"  {d}: {nu} units left, {nb} batches")

print(f"filled anchors accounted for: {len(filled)}")
