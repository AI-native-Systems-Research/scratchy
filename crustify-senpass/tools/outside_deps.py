#!/usr/bin/env python3
"""Which in-scope passes cannot be ported without state defined OUTSIDE this campaign's scope.

Scope = dcc/src/Transform/Sentient/*.{cpp,hpp,h} (top level, 32,766 .cpp lines + 1,672 header).
Out of scope, and what this measures:
  * dcc/src/Transform/Sentient/Analyses/          39 files, ~9,900 lines - the analysis RESULTS
  * dcc/src/Transform/Sentient/RegisterInitialization/  12 files, ~1,800 lines - the candidate pipeline

EVIDENCE IS PRECISE, NOT INFERRED FROM CALL NAMES. An earlier version counted any call whose name
appeared anywhere in those subdirectories and reported all 52 passes "blocked" on the strength of
`if(`, `for(`, `push_back(` and `dbgs(`. Two things are counted here instead:
  1. the out-of-scope headers the pass's own files #include - a fact, quotable as a line;
  2. the out-of-scope TYPE names (analysis classes) the pass's unit bodies actually mention.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import re
from collections import Counter, defaultdict

SENT = _paths.SENT
WORK = _paths.WORK
SUBDIRS = ("Analyses", "RegisterInitialization")

TYPE_DECL = re.compile(r"^\s*(?:class|struct)\s+([A-Z][A-Za-z0-9_]*)\b", re.M)


def outside_types():
    """Class/struct names declared in the out-of-scope subdirectories, by owning file."""
    owner = {}
    for s in SUBDIRS:
        for p in sorted(glob.glob(f"{SENT}/{s}/*")):
            if not p.endswith((".h", ".hpp", ".cpp")):
                continue
            t = open(p, encoding="utf-8", errors="replace").read()
            for m in TYPE_DECL.finditer(t):
                owner.setdefault(m.group(1), f"{s}/{os.path.basename(p)}")
    return owner


def main():
    units = json.load(open(WORK + "/units.json"))
    owner = outside_types()
    mods = sorted({u["module"] for u in units})
    stems = defaultdict(set)
    for u in units:
        stems[u["module"]].add(os.path.splitext(u["file"])[0])

    incl = defaultdict(set)
    for mod, ss in stems.items():
        for stem in ss:
            for ext in (".cpp", ".hpp", ".h"):
                p = f"{SENT}/{stem}{ext}"
                if not os.path.exists(p):
                    continue
                for m in re.finditer(r'#\s*include\s*"([^"]+)"',
                                     open(p, encoding="utf-8", errors="replace").read()):
                    h = m.group(1)
                    for s in SUBDIRS:
                        if f"/Sentient/{s}/" in h:
                            incl[mod].add(h.split(f"/Sentient/")[1])

    text_of = {}
    hits = defaultdict(Counter)
    hit_units = defaultdict(set)
    for u in units:
        p = SENT + "/" + u["file"]
        if p not in text_of:
            text_of[p] = open(p, encoding="utf-8", errors="replace").read()
        body = text_of[p][u["open_off"]:u["close_off"] + 1]
        for name, own in owner.items():
            if re.search(r"\b" + re.escape(name) + r"\b", body):
                hits[u["module"]][name] += 1
                hit_units[u["module"]].add(u["unit"])

    blocked = sorted((m for m in mods if incl.get(m) or hits.get(m)),
                     key=lambda m: (-len(hit_units[m]), m))
    clean = [m for m in mods if m not in blocked]
    print(f"out-of-scope type names declared : {len(owner)}")
    print(f"passes needing out-of-scope state: {len(blocked)} of {len(mods)}")
    print(f"passes self-contained            : {len(clean)}")
    print()
    print(f"{'pass module':44s} {'units':>5s} {'need':>5s}  out-of-scope headers / types")
    for m in blocked:
        n = len([u for u in units if u["module"] == m])
        h = sorted(incl.get(m, ()))
        t = [k for k, _ in hits[m].most_common(5)]
        print(f"{m:44s} {n:5d} {len(hit_units[m]):5d}  "
              + (", ".join(os.path.basename(x) for x in h) if h else "-")
              + ("   [" + ", ".join(t) + "]" if t else ""))
    print()
    print("self-contained passes (no out-of-scope include, no out-of-scope type named):")
    for m in clean:
        print(f"   {m}  ({len([u for u in units if u['module'] == m])} units)")
    tot = len(set().union(*hit_units.values())) if hit_units else 0
    print(f"\nunits naming an out-of-scope type: {tot} of {len(units)}")

    import os as _os
    out = _os.path.join(_paths.CAMP, "OUTSIDE-DEPS.tsv")
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("pass_module\tunits\tunits_needing_outside_state\tout_of_scope_headers\t"
                 "out_of_scope_types\n")
        for m in mods:
            n = len([u for u in units if u["module"] == m])
            fh.write("\t".join([
                m, str(n), str(len(hit_units.get(m, ()))),
                ",".join(sorted(incl.get(m, ()))) or "-",
                ",".join(sorted(hits[m])) or "-",
            ]) + "\n")
    with open(_os.path.join(_paths.CAMP, "OUTSIDE-UNITS.tsv"), "w") as fh:
        fh.write("unit\tpass_module\tout_of_scope_types_named\n")
        for u in units:
            p2 = SENT + "/" + u["file"]
            body = text_of[p2][u["open_off"]:u["close_off"] + 1]
            named = sorted(k for k in owner if re.search(r"\b" + re.escape(k) + r"\b", body))
            if named:
                fh.write(f"{u['unit']}\t{u['module']}\t{','.join(named)}\n")
    print("wrote OUTSIDE-DEPS.tsv and OUTSIDE-UNITS.tsv")


if __name__ == "__main__":
    main()
