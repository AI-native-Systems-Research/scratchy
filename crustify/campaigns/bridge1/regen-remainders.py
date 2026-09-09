#!/usr/bin/env python3
"""Rebuild each bridge-1 sub-campaign's port-remainder.json from the anchors already on the branch.

Run after any promote (driver.sh does). Without it a restart re-ports what already landed: the
schedules are static and crustify has no idea which units are done.

Reads /tmp/filled-bridge1.txt -- one `eNNN_name` per line, produced by:
    grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' crates/compiler/deeptools/src \
      | sed 's|/// Replaces: ||' | sort -u > /tmp/filled-bridge1.txt
"""
import json
import pathlib

MAX_SYMS = 8  # port batches stay small: crustify promotes nothing until an agent completes

filled = set(open("/tmp/filled-bridge1.txt").read().split())
base = pathlib.Path(__file__).resolve().parent

for d in [
    "levels0-1-leaves-and-helpers",
    "levels2-4-transfer-and-compute",
    "levels5-10-statements-and-drivers",
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
            "batches": [
                dict(proto, items=units[i:i + MAX_SYMS])
                for i in range(0, len(units), MAX_SYMS)
            ],
        })
    nb = sum(len(w["batches"]) for w in waves)
    nu = sum(w["unit_count"] for w in waves)
    s["waves"] = waves
    s["summary"].update(batch_count=nb, unit_count=nu, layer_count=len(waves))
    json.dump(s, open(f.with_name("port-remainder.json"), "w"), indent=2)
    print(f"  {d}: {nu} units left, {nb} batches")

print(f"filled anchors accounted for: {len(filled)}")
