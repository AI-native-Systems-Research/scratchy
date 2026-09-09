#!/usr/bin/env python3
"""Build the crustify translate schedules for the Sentient in-place pass campaign.

Also usable as the remainder regenerator: `gen_campaign.py --remainder <repo-root>` rewrites every
port-remainder.json with only the units that still have no `/// Replaces:` anchor.

⛔ PROGRESS IS COUNTED UNDER THE CAMPAIGN'S OWN OUTPUT DIRECTORY. Session refs and `eNNN_` unit
names are both repo-wide, so a branch-based or crate-wide count returns another campaign's number.
The only place counted is crates/compiler/deeptools/src/transform/sentient/.

⛔ `layer_count` MUST BE DISTINCT ITEM LAYERS, NOT WAVE COUNT, or every review stage dies.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import re
from collections import Counter, defaultdict

WORK = _paths.WORK
TREE = _paths.TREE
EXTRACT = _paths.EXTRACT
OUTDIR = _paths.OUTDIR
CAMPDIR = "crustify/campaigns/senpass"
WF = "crustify/wavefront/wavefront-config.json"

# ⭐ max_syms 8 port / 24 review. crustify batches are ALL-OR-NOTHING: nothing a batch produces is
# promoted until its agent completes, so bridge 2's first run lost 4 h 46 m of three agents' work
# when the API dropped at 123 turns with 50-unit batches. Small batches bound the loss to one batch.
BUDGETS_PORT = {"max_syms": 8, "max_loc": 320, "max_types": 5, "min_fields": 20}
BUDGETS_REVIEW = {"max_syms": 24, "max_loc": 960, "max_types": 15, "min_fields": 60}

SUBS = [
    ("sc1", "level0-leaves", [0]),
    ("sc2", "level1-helpers", [1]),
    ("sc3", "levels2-3-pass-bodies", [2, 3]),
    ("sc4", "levels4-6-drivers", [4, 5, 6]),
    ("sc5", "levels7-plus-roots", [7, 8, 9, 10, 11, 12]),
]


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def item(u, in_scope):
    return {
        "name": u["unit"],
        "defined_in": EXTRACT,
        "kind": "symbol",
        "source_kind": "function",
        "layer": u["level"],
        "loc": u["body_lines"],
        "deps": {
            "types": [],
            "symbols": [{"name": c, "defined_in": EXTRACT, "scope": "port"}
                        for c in u["call_units"] if c in in_scope],
        },
        "fallback": [],
        "back_fill": [],
        "generates": [],
        "field_anchors": [],
    }


def schedule(units, levels, budgets, wf_path, wf_sha, note):
    """One schedule: a wave per dependency level, batches packed within a single home file."""
    sel = [u for u in units if u["level"] in levels]
    in_scope = {u["unit"] for u in units}
    waves = []
    for lv in sorted(levels):
        lvu = [u for u in sel if u["level"] == lv]
        if not lvu:
            continue
        by_home = defaultdict(list)
        for u in lvu:
            by_home[u["rust_home"]].append(u)
        # Pack whole homes into a batch up to max_syms, so a home file is owned by ONE batch
        # wherever it fits: file locality without starving the batch. A home larger than max_syms
        # is split, and only then do two concurrent agents share a file.
        batches = []
        cur = []
        cap = budgets["max_syms"]
        for home in sorted(by_home):
            us = sorted(by_home[home], key=lambda u: u["entry"])
            if len(us) > cap:
                if cur:
                    batches.append(cur)
                    cur = []
                for i in range(0, len(us), cap):
                    batches.append(us[i:i + cap])
                continue
            if len(cur) + len(us) > cap:
                batches.append(cur)
                cur = []
            cur.extend(us)
        if cur:
            batches.append(cur)
        batches = [{"kind": "symbol", "source_file": EXTRACT,
                    "items": [item(u, in_scope) for u in b]} for b in batches]
        waves.append({"unit_count": len(lvu), "batches": batches})
    layers = sorted({u["level"] for u in sel})
    return {
        "schema_version": 3,
        "oracle_config": {"path": wf_path, "sha256": wf_sha},
        "api_headers_only": False,
        "budgets": budgets,
        "summary": {
            "unit_count": len(sel),
            # ⛔ DISTINCT ITEM LAYERS, not the number of waves.
            "layer_count": len(layers),
            "batch_count": sum(len(w["batches"]) for w in waves),
            "file_count": 1,
        },
        "waves": waves,
        "_comment": note,
    }


def filled_units(root):
    """Units with a landed `/// Replaces:` anchor, counted ONLY under the campaign's output dir."""
    got = set()
    base = os.path.join(root, OUTDIR)
    for dirpath, _, names in os.walk(base):
        for n in names:
            if not n.endswith(".rs"):
                continue
            s = open(os.path.join(dirpath, n), encoding="utf-8", errors="replace").read()
            got |= set(re.findall(r"///\s*Replaces:\s*(e\d+_[A-Za-z0-9_]+)", s))
    return got


def main():
    units = json.load(open(WORK + "/units.json"))
    remainder = "--remainder" in sys.argv
    root = TREE
    if remainder:
        root = sys.argv[sys.argv.index("--remainder") + 1]
    wf_abs = os.path.join(root, WF)
    wf_sha = sha256_file(wf_abs)
    done = filled_units(root) if remainder else set()
    if remainder:
        print(f"filled anchors under {OUTDIR}: {len(done)}")

    note_common = [
        "Batches are packed WITHIN A SINGLE HOME FILE so two concurrent agents rarely touch the same",
        "file; where a home holds more than max_syms units its batches do run concurrently, and the",
        "driver's promote is rebase-then-fast-forward for exactly that reason.",
        "⛔ layer_count is the number of DISTINCT ITEM LAYERS in this schedule, not the wave count.",
        "⛔ OUTSTANDING WORK COMES FROM crustify-senpass/UNITS.tsv, NEVER FROM ANCHORS IN THE TREE.",
        "   A deleted `crustify:todo:` anchor is indistinguishable from a finished unit: bridge 2",
        "   printed CAMPAIGN DRIVER DONE having ported 235 of 384, silently losing 149 — the biggest",
        "   functions in its span. Never delete an anchor you did not port.",
    ]
    made = []
    for tag, name, levels in SUBS:
        d = os.path.join(root, CAMPDIR, f"{tag}-{name}")
        os.makedirs(d, exist_ok=True)
        sel = [u for u in units if u["level"] in levels]
        if not remainder:
            for obj, fn, budg, extra in (
                (None, "port.json", BUDGETS_PORT,
                 [f"PORT wave for {name}: dependency level(s) {levels}, {len(sel)} units."]),
                (None, "review.json", BUDGETS_REVIEW,
                 [f"REVIEW wave for {name}: every unit reviewed by a DIFFERENT agent instance than",
                  "ported it. Independent review is mandatory; the review pass owns citation checks."]),
            ):
                s = schedule(units, levels, budg, WF, wf_sha, extra + note_common)
                open(os.path.join(d, fn), "w").write(json.dumps(s, indent=1) + "\n")
            made.append((tag, name, levels, len(sel)))
        # the remainder is always (re)written, so a resumed stage re-runs only unfilled units
        left = [u for u in units if u["level"] in levels and u["unit"] not in done]
        s = schedule([u for u in units if u["unit"] not in done], levels, BUDGETS_PORT, WF, wf_sha,
                     [f"PORT REMAINDER for {name}: the {len(left)} of {len(sel)} units in level(s) "
                      f"{levels} that still have no `/// Replaces:` anchor under {OUTDIR}."]
                     + note_common)
        open(os.path.join(d, "port-remainder.json"), "w").write(json.dumps(s, indent=1) + "\n")
        if remainder:
            made.append((tag, name, levels, len(left)))

    hist = Counter(u["level"] for u in units)
    print("levels: " + "  ".join(f"L{k}:{v}" for k, v in sorted(hist.items())))
    tot = 0
    for tag, name, levels, n in made:
        d = os.path.join(root, CAMPDIR, f"{tag}-{name}")
        p = os.path.join(d, "port-remainder.json" if remainder else "port.json")
        s = json.load(open(p))
        print(f"   {tag} {name:24s} levels {str(levels):18s} units {n:4d}  "
              f"waves {len(s['waves'])}  batches {s['summary']['batch_count']}  "
              f"layers {s['summary']['layer_count']}")
        tot += n
    print(f"total scheduled: {tot} of {len(units)}")


if __name__ == "__main__":
    main()
