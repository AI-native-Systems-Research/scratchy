#!/usr/bin/env python3
"""Emit each types sub-campaign's port.json and review.json from UNITS.tsv + the vendored headers.

⛔ WHY THIS EXISTS RATHER THAN `wavefront schedule`. The oracle's CodeQL database is built and its T1
entity tables are complete, but THREE T2 edge tables were never extracted by the older tool version
(`macro_expansions`, `macro_generated_types`, `signature_type_uses`), and `compose/reach.py` loads
`macro_expansions.csv` unconditionally. Re-extracting it is an unfiltered `from MacroInvocation mi,
Macro m` over a 56 GB C++ database. If that extraction lands, PREFER `wavefront schedule` and diff its
layering against this file's — a disagreement is a real finding about the type graph, not a nuisance.

⛔ WHAT THIS SCRIPT DOES **NOT** INVENT. Every number below is read from a file:
  - the unit list, levels, callees and homes come from crustify-types/UNITS.tsv;
  - the field names and class spans are re-derived from crustify-types/cpp/*.h on every run, so the
    schedule cannot drift from the authority (those headers are byte-identical to deeptools-src
    @ a0d29abbed, verified with `cmp` at staging).
The only authored inputs are the batching budgets, which match the capacity campaign's.

⛔ AND THE LEVELS IN UNITS.tsv WERE MEASURED, NOT GUESSED. Three edges an earlier draft asserted were
counted and found to be ZERO: DesignSpaceConfig->Metadata, Metadata->FailedAlloc,
ScheduleTree->BlockNode. If you add a unit, count its edges in the class body before assigning a level.
"""
import hashlib
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent.parent          # the worktree root
CPP = ROOT / "crustify-types" / "cpp"
UNITS = ROOT / "crustify-types" / "UNITS.tsv"

# Same budgets as the capacity campaign. `min_fields` closes a type batch once this many declared
# fields have accumulated, which is what isolates a wide type (DesignSpaceConfig 49, Metadata 41).
BUDGETS = {"max_syms": 8, "max_loc": 320, "max_types": 5, "min_fields": 20}

STAGES = {
    # stage dir            -> the unit entries it schedules, in UNITS.tsv order
    "sc1-node-hierarchy": [f"e{n:03d}" for n in range(1, 17)],
    "sc2-designspaceconfig": ["e017", "e018"],
}

FIELD = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*_$")


def cpp_class(header: str, name: str) -> tuple[str, int, int, list[str]]:
    """(source_kind, span_lines, first_line, declared field names) for one class in a header.

    The field rule is the one the driver's `count` uses: a member whose identifier ends in `_` and is
    followed by `;`, `=` or `[`. Nested structs are included, which is deliberate — Metadata's
    Datastage/Constraints/... are part of that unit.
    """
    lines = (CPP / header).read_text().splitlines()
    start = kind = None
    for i, ln in enumerate(lines):
        m = re.match(rf"^(class|struct)\s+{re.escape(name)}(\s|:|$)", ln)
        if m:
            start, kind = i, m.group(1)
            break
    if start is None:
        raise SystemExit(f"⛔ {name} not found in {header} — UNITS.tsv and the vendored header disagree")
    # ⛔⛔ crustify's TranslateAgent TAXONOMY IS C'S: it accepts only struct/union/enum/macro and
    # rejects a whole BATCH with `ValueError: unsupported type kind(s) ['class']`. That killed 14 of
    # sc1's 16 units on the first real wave — every batch that contained one `class` failed wholesale,
    # and 7 of these 18 units are declared `class` (DataStructDims, BlockNode, ScheduleTree,
    # DesignSpaceConfig among them).
    # A C++ `class` differs from `struct` ONLY in default member access, so for a port they are the
    # same aggregate and "struct" is the honest kind to report. The vendored header remains the
    # authority for the actual keyword; nothing about the C++ is restated.
    if kind == "class":
        kind = "struct"
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("};"))
    body = "\n".join(lines[start:end + 1])
    fields = sorted({
        t for t in re.findall(r"\b([a-zA-Z][a-zA-Z0-9]*_)\s*[;=\[]", body) if FIELD.match(t)
    })
    return kind, end - start + 1, start + 1, fields


def rows():
    out = []
    for line in UNITS.read_text().splitlines()[1:]:
        c = line.split("\t")
        entry, level, _loc, authority, _ex, _home, callees = c[0], c[1], c[2], c[3], c[4], c[5], c[6]
        header = authority.split("/")[-1].split(":")[0]
        name = entry.split("_", 1)[1]
        kind, span, first, fields = cpp_class(header, name)
        out.append({
            "entry": entry, "pref": entry.split("_", 1)[0], "layer": int(level),
            "header": header, "name": name,
            "kind": kind, "span": span, "first": first, "fields": fields,
            "deps": [d for d in callees.split(",") if d],
        })
    # Keyed by the eNNN PREFIX: UNITS.tsv's callee column names prefixes ("e001,e006"), while an
    # item's `name` is the full `eNNN_CppName`. Keying by the full name made STAGES' prefix list a
    # KeyError, which is the honest failure — a silent `.get()` would have emitted a schedule with
    # units missing and every assert still passing.
    by = {r["pref"]: r for r in out}

    # ⭐⭐ THE LAYER IS DERIVED FROM THE DEPENDENCY GRAPH, NOT READ FROM UNITS.tsv. A wave IS a
    # barrier, so a unit must sit strictly above every callee; the authored `level` column is
    # documentation and it was WRONG for six of eighteen units on first writing — e010_BlockNode was
    # authored at the same level as the e009_ScheduleNode it inherits, which would have put a base
    # class and its derived class in ONE wave and handed an agent a type whose parent did not exist
    # yet. Deriving it here makes that unrepresentable rather than caught-if-lucky.
    def depth(e, seen=()):
        if e in seen:
            raise SystemExit(f"⛔ dependency cycle through {e}: {' -> '.join(seen + (e,))}")
        return 0 if not by[e]["deps"] else 1 + max(depth(d, seen + (e,)) for d in by[e]["deps"])

    for pref, r in by.items():
        d = depth(pref)
        if d != r["layer"]:
            print(f"  ⚠ {r['entry']}: UNITS.tsv says level {r['layer']}, graph says {d} — using {d}")
        r["layer"] = d
    return by


def batches(units):
    """Group into single-file batches, closing on max_types or min_fields (wide types isolated)."""
    out = []
    for header in dict.fromkeys(u["header"] for u in units):        # stable, first-seen order
        cur, nf = [], 0
        for u in [x for x in units if x["header"] == header]:
            if cur and (len(cur) >= BUDGETS["max_types"] or nf >= BUDGETS["min_fields"]):
                out.append((header, cur)); cur, nf = [], 0
            cur.append(u); nf += len(u["fields"])
        if cur:
            out.append((header, cur))
    return out


def item(u, by_entry):
    return {
        "name": u["entry"],
        "defined_in": f"crustify-types/cpp/{u['header']}",
        "kind": "type",
        "source_kind": u["kind"],
        "layer": u["layer"],
        "loc": u["span"],
        # ⛔ A v3 DEP IS A DICT WITH A `scope`, NOT A BARE NAME. crustify's wave.py:84-90 rejects the
        # whole schedule with "invalid dependency scope" unless every entry is a dict whose `scope` is
        # one of wrap/port/ext — and wavefront's own examples/waves.json predates that field, so
        # copying its `{name, defined_in}` shape is not enough. `capacity` never hit this because
        # every one of its items had EMPTY deps.
        # Every dep here is another unit of this campaign, so the scope is "port" — not "ext", which
        # would say the type is supplied from outside and needs no unit.
        "deps": {
            "types": [
                {
                    "name": by_entry[d]["entry"],
                    "defined_in": f"crustify-types/cpp/{by_entry[d]['header']}",
                    "scope": "port",
                }
                for d in u["deps"]
            ],
            "symbols": [],
        },
        "fallback": [],
        "back_fill": [],
        "generates": [],
        # ⭐ THE POINT OF THE CAMPAIGN: every field the C++ type declares, so a unit's completion is
        # checkable against a list rather than against "the struct was touched".
        "field_anchors": u["fields"],
    }


def schedule(stage, entries, by_entry, objective):
    units = [by_entry[e] for e in entries]
    waves = []
    for layer in sorted({u["layer"] for u in units}):
        at = [u for u in units if u["layer"] == layer]
        bs = [{"kind": "type", "source_file": f"crustify-types/cpp/{h}",
               "items": [item(u, by_entry) for u in g]} for h, g in batches(at)]
        waves.append({"unit_count": len(at), "batches": bs})
    cfg = BASE / stage / "wavefront-config.json"
    items = [i for w in waves for b in w["batches"] for i in b["items"]]
    doc = {
        "schema_version": 3,
        "oracle_config": {
            "path": f"crustify/campaigns/types/{stage}/wavefront-config.json",
            "sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
        },
        "api_headers_only": False,
        "budgets": dict(BUDGETS),
        "summary": {
            "unit_count": len(items),
            "layer_count": len({i["layer"] for i in items}),
            "batch_count": sum(len(w["batches"]) for w in waves),
            "file_count": len({b["source_file"] for w in waves for b in w["batches"]}),
        },
        "waves": waves,
    }
    # ⭐ ASSERT THE FOUR THINGS crustify CHECKS, HERE, WHERE THE NUMBERS ARE WRITTEN. A schedule that
    # disagrees with its own items is rejected in under a second and the message names none of them.
    ids = [(i["name"], i["defined_in"]) for i in items]
    assert len(set(ids)) == len(ids), f"{stage}: duplicate (name, defined_in) identity"
    assert len(items) == doc["summary"]["unit_count"], f"{stage}: unit_count"
    assert len({i["layer"] for i in items}) == doc["summary"]["layer_count"], f"{stage}: layer_count"
    assert doc["summary"]["batch_count"] == sum(len(w["batches"]) for w in waves), f"{stage}: batch_count"
    # A dep must be scheduled in this stage or an EARLIER one, never later — that is what a barrier is.
    seen, order = set(), [i for w in waves for b in w["batches"] for i in b["items"]]
    for w in waves:
        for b in w["batches"]:
            for i in b["items"]:
                for d in [x["name"] for x in i["deps"]["types"]]:
                    assert d in seen or d not in {x["name"] for x in order}, \
                        f"{stage}: {i['name']} depends on {d} scheduled in the same or a later wave"
        seen |= {i["name"] for b in w["batches"] for i in b["items"]}
    (BASE / stage / f"{objective}.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


by_entry = rows()
for stage, entries in STAGES.items():
    for objective in ("port", "review"):
        d = schedule(stage, entries, by_entry, objective)
        s = d["summary"]
        print(f"  {stage}/{objective}.json: {s['unit_count']} units, {len(d['waves'])} wave(s), "
              f"{s['batch_count']} batches, {s['layer_count']} layer(s), {s['file_count']} file(s), "
              f"{sum(len(i['field_anchors']) for w in d['waves'] for b in w['batches'] for i in b['items'])} field anchors")
