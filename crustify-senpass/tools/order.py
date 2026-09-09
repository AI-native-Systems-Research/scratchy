import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

#!/usr/bin/env python3
"""Dependency order for the Sentient in-place pass campaign.

Levels are computed over the SCC CONDENSATION, never by a plain longest-path fixpoint: bridge 3's
plain fixpoint printed "level 266" on a mutually recursive pair because each member kept raising
the other. Tarjan gives the strongly connected components; every member of a component shares the
component's level, and the level is the longest path in the (acyclic) condensation.

Call edges are resolved by NAME with a same-file preference, because 51 pass classes each define
their own `runOnOperation` and `runOn`. Ambiguous names that cannot be pinned to one unit are
dropped and COUNTED - a dropped edge can only make a level too low, never invent one, and the
count is reported so the loss is visible rather than silent.
"""
import json
import re
import sys
from collections import Counter, defaultdict

import cppscan

SENT = _paths.SENT
WORK = _paths.WORK


def tarjan(nodes, succ):
    """Iterative Tarjan SCC. Returns (comp_of_node, comps) with comps in reverse topological order."""
    index = {}
    low = {}
    onstack = {}
    stack = []
    comp_of = {}
    comps = []
    counter = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ.get(root, ())))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        onstack[root] = True
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    onstack[w] = True
                    work.append((w, iter(succ.get(w, ()))))
                    advanced = True
                    break
                if onstack.get(w):
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                p = work[-1][0]
                low[p] = min(low[p], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    comp.append(w)
                    comp_of[w] = len(comps)
                    if w == v:
                        break
                comps.append(comp)
    return comp_of, comps


def main():
    units = json.load(open(WORK + "/inscope.json"))
    # bodies are re-read here (inscope.json omits them) so the call scan sees the real text
    text_of = {}
    for u in units:
        f = SENT + "/" + u["file"]
        if f not in text_of:
            text_of[f] = open(f, encoding="utf-8", errors="replace").read()
    for u in units:
        t = text_of[SENT + "/" + u["file"]]
        u["body"] = t[u["open_off"]:u["close_off"] + 1]

    # unit ids are assigned only after ordering, so key by (file, head_line) for now
    key = lambda u: (u["file"], u["head_line"])
    by_name = defaultdict(list)
    for u in units:
        by_name[u["name"]].append(u)

    succ = defaultdict(set)
    ambiguous = Counter()
    resolved = 0
    for u in units:
        callees = cppscan.calls_in(u["body"])
        for c in callees:
            cands = by_name.get(c)
            if not cands:
                continue
            same = [x for x in cands if x["file"] == u["file"]]
            picks = []
            if len(same) == 1:
                picks = same
            elif len(cands) == 1:
                picks = cands
            elif same:
                # several in the same file: prefer the one whose class qualifier matches the
                # caller's; if that is still not unique, take ALL of them. A spurious edge can
                # only push a unit to a HIGHER level (ported later, after more of its span);
                # a dropped edge can put a caller before its callee, which is the failure that
                # actually costs an agent its port.
                q = [x for x in same if x["qual"] == u["qual"]]
                picks = q if len(q) == 1 else same
            if not picks:
                ambiguous[c] += 1
                continue
            for pick in picks:
                if key(pick) == key(u):
                    continue
                succ[key(u)].add(key(pick))
                resolved += 1

    nodes = [key(u) for u in units]
    comp_of, comps = tarjan(nodes, succ)
    # condensation edges
    cedge = defaultdict(set)
    for v in nodes:
        for w in succ.get(v, ()):
            if comp_of[v] != comp_of[w]:
                cedge[comp_of[v]].add(comp_of[w])
    # longest path over the DAG; comps from tarjan are in reverse topological order, so a
    # component's successors always have a lower index and are already levelled.
    level = {}
    for ci in range(len(comps)):
        succ_levels = [level[d] for d in cedge.get(ci, ())]
        level[ci] = (max(succ_levels) + 1) if succ_levels else 0

    for u in units:
        u["level"] = level[comp_of[key(u)]]
        u["scc"] = comp_of[key(u)]
        u["calls"] = sorted(succ.get(key(u), ()))

    # order: level, then file, then head_line -> deterministic entry numbering
    units.sort(key=lambda u: (u["level"], u["file"], u["head_line"]))
    for i, u in enumerate(units, 1):
        u["entry"] = i
        u["unit"] = f"e{i:03d}_{u['name']}"

    idx = {key(u): u for u in units}
    for u in units:
        u["call_units"] = sorted(idx[c]["unit"] for c in u["calls"])

    json.dump([{k: v for k, v in u.items() if k != "body"} for u in units],
              open(WORK + "/ordered.json", "w"), indent=0)

    cyc = [c for c in comps if len(c) > 1]
    print(f"units              : {len(units)}")
    print(f"call edges resolved: {resolved}")
    print(f"ambiguous callees  : {sum(ambiguous.values())} over {len(ambiguous)} names")
    print(f"   worst: {ambiguous.most_common(6)}")
    print(f"SCCs               : {len(comps)}  (non-trivial: {len(cyc)}, "
          f"largest {max((len(c) for c in cyc), default=0)})")
    print(f"levels             : 0..{max(u['level'] for u in units)}")
    hist = Counter(u["level"] for u in units)
    print("   " + "  ".join(f"L{k}:{v}" for k, v in sorted(hist.items())))
    if cyc:
        biggest = max(cyc, key=len)
        print(f"largest cycle ({len(biggest)}): "
              + ", ".join(idx[n]["unit"] for n in biggest[:8]))


if __name__ == "__main__":
    main()
