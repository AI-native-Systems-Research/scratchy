#!/usr/bin/env python3
"""Dependency order for the ddc + L3-scheduler campaign.

Levels are computed over the SCC CONDENSATION, never by a plain longest-path fixpoint: bridge 3's
plain fixpoint printed "level 266" on a mutually recursive pair because each member kept raising the
other. Tarjan gives the strongly connected components; every member of a component shares the
component's level, and the level is the longest path in the (acyclic) condensation.

Call edges are resolved by NAME with a same-file, then same-scope, preference. Ambiguous names that
cannot be pinned are counted rather than dropped silently. A spurious edge can only push a unit to a
HIGHER level (ported later); a dropped edge can put a caller before its callee, which is the failure
that actually costs an agent its port.

Also reports CROSS-SCOPE call edges, which is the measurement that decides whether l3 / ddc / ddl
can be run as independent tracks or must share one level ladder.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
from collections import Counter, defaultdict

import cppscan

AUTH = _paths.AUTH_ROOT
WORK = _paths.WORK


def tarjan(nodes, succ):
    """Iterative Tarjan SCC. Returns (comp_of_node, comps), comps in reverse topological order."""
    index, low, onstack = {}, {}, {}
    stack, comp_of, comps = [], {}, []
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
    text_of = {}
    for u in units:
        p = os.path.join(AUTH, u["rel"])
        if p not in text_of:
            text_of[p] = open(p, encoding="utf-8", errors="replace").read()
    for u in units:
        t = text_of[os.path.join(AUTH, u["rel"])]
        u["body"] = t[u["open_off"]:u["close_off"] + 1]

    key = lambda u: (u["rel"], u["head_line"])
    by_name = defaultdict(list)
    for u in units:
        by_name[u["name"]].append(u)

    succ = defaultdict(set)
    ambiguous = Counter()
    resolved = 0
    for u in units:
        for c in cppscan.calls_in(u["body"]):
            cands = by_name.get(c)
            if not cands:
                continue
            same_file = [x for x in cands if x["rel"] == u["rel"]]
            same_scope = [x for x in cands if x["scope"] == u["scope"]]
            picks = []
            if len(same_file) == 1:
                picks = same_file
            elif len(cands) == 1:
                picks = cands
            elif same_file:
                q = [x for x in same_file if x["qual"] == u["qual"]]
                picks = q if len(q) == 1 else same_file
            elif len(same_scope) == 1:
                picks = same_scope
            elif same_scope:
                picks = same_scope
            else:
                picks = cands
            if not picks:
                ambiguous[c] += 1
                continue
            if len(picks) > 1:
                ambiguous[c] += 1
            for pick in picks:
                if key(pick) == key(u):
                    continue
                succ[key(u)].add(key(pick))
                resolved += 1

    nodes = [key(u) for u in units]
    comp_of, comps = tarjan(nodes, succ)
    cedge = defaultdict(set)
    for v in nodes:
        for w in succ.get(v, ()):
            if comp_of[v] != comp_of[w]:
                cedge[comp_of[v]].add(comp_of[w])
    # comps are in reverse topological order, so a component's successors have a lower index and
    # are already levelled.
    level = {}
    for ci in range(len(comps)):
        sl = [level[d] for d in cedge.get(ci, ())]
        level[ci] = (max(sl) + 1) if sl else 0

    for u in units:
        u["level"] = level[comp_of[key(u)]]
        u["scc"] = comp_of[key(u)]
        u["calls"] = sorted(succ.get(key(u), ()))

    # Deterministic entry numbering: level, then scope (l3, ddc, ddl), then file, then line.
    scope_rank = {s: i for i, s in enumerate(_paths.SCOPES)}
    units.sort(key=lambda u: (u["level"], scope_rank[u["scope"]], u["rel"], u["head_line"]))
    for i, u in enumerate(units, 1):
        u["entry"] = i
        u["unit"] = "e%03d_%s" % (i, u["name"].replace("~", "dtor_"))

    idx = {key(u): u for u in units}
    for u in units:
        u["call_units"] = sorted(idx[c]["unit"] for c in u["calls"])

    json.dump([{k: v for k, v in u.items() if k != "body"} for u in units],
              open(WORK + "/ordered.json", "w"), indent=0)

    cyc = [c for c in comps if len(c) > 1]
    print("units              : %d" % len(units))
    print("call edges resolved: %d" % resolved)
    print("ambiguous callees  : %d over %d names" % (sum(ambiguous.values()), len(ambiguous)))
    print("   worst: %s" % ambiguous.most_common(6))
    print("SCCs               : %d  (non-trivial: %d, largest %d)"
          % (len(comps), len(cyc), max((len(c) for c in cyc), default=0)))
    print("levels             : 0..%d" % max(u["level"] for u in units))
    hist = Counter(u["level"] for u in units)
    print("   " + "  ".join("L%d:%d" % (k, v) for k, v in sorted(hist.items())))

    # -- cross-scope edges: the measurement that decides the sub-campaign split ----------------
    cross = Counter()
    for u in units:
        for c in u["calls"]:
            if idx[c]["scope"] != u["scope"]:
                cross[(u["scope"], idx[c]["scope"])] += 1
    print("cross-scope edges  : %d" % sum(cross.values()))
    for (a, b), n in cross.most_common():
        print("   %-4s -> %-4s %d" % (a, b, n))
    if cyc:
        biggest = max(cyc, key=len)
        print("largest cycle (%d): %s"
              % (len(biggest), ", ".join(idx[n]["unit"] for n in biggest[:8])))


if __name__ == "__main__":
    main()
