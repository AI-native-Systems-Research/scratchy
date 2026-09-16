#!/usr/bin/env python3
"""Rebuild each types sub-campaign's port-remainder.json from the anchors already on the branch.

Run after any promote (driver.sh does). Without it a restart re-ports what already landed: the
schedules are static and crustify has no idea which units are done.

⛔ THIS WRITES ONLY port-remainder.json, A GENERATED ARTIFACT. It never edits Rust, never edits
port.json, and never edits UNITS.tsv — that file must stay byte-stable while a wave is in flight.

⛔ AN ANCHOR IS NOT A PORT. This script drops a unit from the remainder once its `/// Replaces:`
anchor exists, which is the only signal crustify can consume — it is NOT the acceptance gate. The
gate is: zero `todo!` for the arms in scope, a real non-test caller, and execution on a real program.
A unit whose anchor is present but whose body is a stub must be put BACK by hand.

⛔⛔ AND ON A **TYPES** CAMPAIGN AN ANCHOR IS EVEN WEAKER THAN USUAL. A type unit's anchor says the
struct was touched; it says NOTHING about how many of the C++ type's declared fields it now carries,
which is this campaign's actual burndown (`driver.sh`'s `count` prints it, and DesignSpaceConfig
measured 9 of 49 on 2026-09-16). A struct that gained one field of thirty-nine looks identical to a
finished one here. Read the driver's field line, never this script's unit count.

Reads /tmp/filled-types.txt — one `eNNN_name` per line, written by driver.sh's `remainders()`.
"""
import json
import pathlib

MAX_SYMS = 8  # port batches stay small: crustify promotes nothing until an agent completes

filled = set(open("/tmp/filled-types.txt").read().split())
base = pathlib.Path(__file__).resolve().parent

# ⛔ EMPTY ON PURPOSE, AND NOT A PLACEHOLDER TO FILL WITH GUESSES. On the capacity campaign this held
# {e009_calculate_padded, e010_primaryDimToVal_base_st}, a pair MUTUALLY RECURSIVE in C++
# (dsc/dims.cpp:560 -> calculate_padded, :594 -> primaryDimToVal_base_st) — whichever agent got one
# alone could not close its own callee. `wavefront schedule` already isolates a wide type by itself
# (--min-fields) and keeps a type with its lifecycle primitives, so the type batches need no help.
#
# ⭐ ADD A GROUP HERE THE MOMENT A CONCRETE ONE IS FOUND — a base and derived class whose fields only
# make sense together (dsc2.h's ScheduleNode hierarchy is the candidate), or two types with a
# by-value cycle. NAME THE C++ LINES THAT PROVE IT, exactly as the capacity entry did.
INSEPARABLE: list[set[str]] = []


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
    "sc1-node-hierarchy",
    "sc2-designspaceconfig",
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
    # ⛔ `layer_count` IS THE NUMBER OF DISTINCT `layer` VALUES, **NOT** THE NUMBER OF WAVES.
    # crustify checks `len({item["layer"] for item in items}) == summary["layer_count"]`
    # (wave.py:107-112) and exits with "schedule summary or batched item identities disagree".
    # Bridge 1's regen script writes `len(waves)` and gets away with it only because its waves
    # coincide with its levels. THIS campaign cuts waves by ACTUAL DEPENDENCY inside a level — sc1
    # is two waves at layer 0 alone — so the two numbers differ and every remainder was rejected in
    # under a second. Derive it from the items, never from the wave count.
    layers = {i["layer"] for w in waves for b in w["batches"] for i in b["items"]}
    s["waves"] = waves
    s["summary"].update(batch_count=nb, unit_count=nu, layer_count=len(layers))
    # ⭐ ASSERT THE FOUR THINGS crustify CHECKS, HERE, WHERE THE NUMBER IS WRITTEN. A remainder that
    # disagrees with its own items costs a whole driver sweep to discover (19 seconds, three
    # sub-campaigns, zero units) and the message names none of the five clauses.
    items = [i for w in waves for b in w["batches"] for i in b["items"]]
    ids = [(i["name"], i["defined_in"]) for i in items]
    assert len(set(ids)) == len(ids), f"{d}: duplicate (name, defined_in) identity"
    assert len(items) == s["summary"]["unit_count"], f"{d}: unit_count"
    assert len({i["layer"] for i in items}) == s["summary"]["layer_count"], f"{d}: layer_count"
    assert nb == s["summary"]["batch_count"], f"{d}: batch_count"
    json.dump(s, open(f.with_name("port-remainder.json"), "w"), indent=1)
    print(f"  {d}: {nu} units left, {nb} batches, "
          f"{s['summary']['layer_count']} distinct layer(s) — summary agrees with items")

print(f"filled anchors accounted for: {len(filled)} (an anchor is NOT the acceptance gate)")
