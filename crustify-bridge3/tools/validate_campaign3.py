#!/usr/bin/env python3
"""Cross-check the whole bridge-3 campaign before a single agent runs.

Asserts, and prints CAMPAIGN CONSISTENT only if all of them hold:
  1. every one of the 130 UNITS.tsv rows is scheduled in exactly one port.json wave
  2. every scheduled item's `name` has a `// crustify:todo:` anchor in its declared Rust home
  3. every anchor in the Rust homes corresponds to a UNITS.tsv row (no orphans)
  4. every UNITS.tsv citation resolves: the authority file exists and its cited line holds a
     definition whose name matches
  5. every review.json covers exactly its sub-campaign's units
  6. wave order is monotone in level, and every item's callees are in an earlier or the same wave
  7. no unit name collides with a bridge-2 anchor already on the branch
"""
import json
import pathlib
import re
import sys

W = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3")
SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
CAMP = W / "crustify/campaigns/bridge3"
RUST = W / "crates/compiler/deeptools/src/bridges/sentient_to_progir"

STAGES = [
    "levels0-1-leaves-and-instruction-builders",
    "levels2-3-computes-and-transfers",
    "levels4-7-uniform-regions-and-the-driver",
]

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)


rows = [l.split("\t") for l in (W / "crustify-bridge3/UNITS.tsv").read_text().splitlines()[1:]]
units = {r[0]: {"level": int(r[2]), "loc": int(r[3]), "auth": r[4], "home": r[6],
                "calls": [c for c in r[7].split(",") if c and c != "-"]} for r in rows}
print(f"UNITS.tsv: {len(units)} units")
check(len(units) == 130, f"expected 130 units, got {len(units)}")

# 1 + 5 + 6
scheduled = {}
for st in STAGES:
    p = json.load(open(CAMP / st / "port.json"))
    seen_here = []
    prev_level = -1
    for wi, w in enumerate(p["waves"]):
        lv = {i["layer"] for b in w["batches"] for i in b["items"]}
        check(len(lv) == 1, f"{st} wave {wi} mixes levels {lv}")
        check(min(lv) > prev_level, f"{st} wave {wi} level {lv} not after {prev_level}")
        prev_level = max(lv)
        for b in w["batches"]:
            check(len(b["items"]) <= 8, f"{st} wave {wi} batch of {len(b['items'])} > max_syms 8")
            loc = sum(i["loc"] for i in b["items"])
            check(loc <= 700 or len(b["items"]) == 1,
                  f"{st} wave {wi} batch of {loc} LoC > 700 with {len(b['items'])} items")
            for i in b["items"]:
                check(i["name"] not in scheduled,
                      f"{i['name']} scheduled twice ({scheduled.get(i['name'])} and {st})")
                scheduled[i["name"]] = st
                seen_here.append(i["name"])
                # 6: callees earlier
                for dep in i["deps"]["symbols"]:
                    d = units.get(dep["name"])
                    check(d is not None, f"{i['name']} depends on unknown {dep['name']}")
                    if d:
                        check(d["level"] <= i["layer"],
                              f"{i['name']} (L{i['layer']}) depends on {dep['name']} (L{d['level']})")
    r = json.load(open(CAMP / st / "review.json"))
    rnames = [i["name"] for w in r["waves"] for b in w["batches"] for i in b["items"]]
    check(sorted(rnames) == sorted(seen_here),
          f"{st}: review covers {len(rnames)} units, port covers {len(seen_here)}")
    check(all(len(b["items"]) <= 24 for w in r["waves"] for b in w["batches"]),
          f"{st}: a review batch exceeds max_syms 24")
    print(f"  {st}: {len(seen_here)} port, {len(rnames)} review, "
          f"{p['summary']['batch_count']}+{r['summary']['batch_count']} batches")

check(set(scheduled) == set(units),
      f"scheduled != UNITS.tsv: missing {sorted(set(units) - set(scheduled))[:5]}, "
      f"extra {sorted(set(scheduled) - set(units))[:5]}")

# 2 + 3
anchors = {}
for f in RUST.rglob("*.rs"):
    for m in re.finditer(r"// crustify:todo: (e\d{3}_[A-Za-z0-9_]+)", f.read_text()):
        rel = str(f.relative_to(RUST.parent.parent.parent.parent))
        anchors.setdefault(m.group(1), []).append(rel)
print(f"anchors in the Rust homes: {len(anchors)}")
check(set(anchors) == set(units),
      f"anchors != units: missing {sorted(set(units) - set(anchors))[:5]}, "
      f"orphans {sorted(set(anchors) - set(units))[:5]}")
for name, locs in anchors.items():
    check(len(locs) == 1, f"{name} anchored in {len(locs)} files: {locs}")
    if name in units and len(locs) == 1:
        check(locs[0].endswith(units[name]["home"].split("src/")[1]),
              f"{name} anchored in {locs[0]} but UNITS.tsv says {units[name]['home']}")

# 4: every citation resolves and names the right function
sys.path.insert(0, str(W / "crustify-bridge3/tools"))
import cppscan  # noqa: E402

cache = {}
bad_cite = 0
for name, u in units.items():
    rel, ln = u["auth"].rsplit(":", 1)
    p = SRC / rel
    if not p.exists():
        fail.append(f"{name}: authority file {rel} does not exist")
        continue
    if rel not in cache:
        cache[rel] = {d["head_line"]: d for d in cppscan.find_defs(p.read_text(errors="ignore"))}
    d = cache[rel].get(int(ln))
    cpp = name.split("_", 1)[1].replace("dtor_", "~")
    if d is None or d["name"] != cpp:
        bad_cite += 1
        fail.append(f"{name}: {u['auth']} holds "
                    f"{(d['name'] if d else 'no definition')}, expected {cpp}")
print(f"citations resolving to the named definition: {len(units) - bad_cite}/{len(units)}")

# 7: no collision with a bridge-2 anchor already on the branch
b2 = set()
for f in (W / "crates/compiler/deeptools/src/bridges/dataflow_ir_to_sentient").rglob("*.rs"):
    b2 |= set(re.findall(r"/// Replaces: (e\d{3}_[A-Za-z0-9_]+)", f.read_text()))
clash = set(units) & b2
print(f"bridge-2 filled anchors on the branch: {len(b2)}; collisions with bridge 3: {len(clash)}")
check(not clash, f"anchor namespace collision with bridge 2: {sorted(clash)}")

print()
if fail:
    print(f"⛔ {len(fail)} PROBLEM(S):")
    for f in fail[:40]:
        print("   ", f)
    sys.exit(1)
print("✅ CAMPAIGN CONSISTENT")
