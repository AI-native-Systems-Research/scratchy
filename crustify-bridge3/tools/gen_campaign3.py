#!/usr/bin/env python3
"""Generate bridge 3's crustify schedules and the per-sub-campaign wavefront configs."""
import hashlib
import json
import pathlib

OUT = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3")
SRCFILE = "crustify-bridge3/cpp/bridge3.cpp"
TOTAL = 130

STAGES = [
    ("levels0-1-leaves-and-instruction-builders", [0, 1],
     "the leaves and the single-instruction builders: the NOP/return/jump/sync/assign/add/sub "
     "constructors, the operand map, the uniform-instruction and uniform-block accessors, the "
     "register-init accumulation and the reg-def tracker"),
    ("levels2-3-computes-and-transfers", [2, 3],
     "the computes and transfers: FMA, binary, unary and ternary instruction construction, the "
     "L3 and non-L3 load/store instructions, and the Lower* operation handlers that call them"),
    ("levels4-7-uniform-regions-and-the-driver", [4, 5, 6, 7],
     "the region and program level: LowerCommonOperations, LowerForOperation, the uniform-region "
     "pair (LowerUniformOperations and GenerateProgIR are MUTUALLY RECURSIVE and share level 5), "
     "GenerateProgIRForProgramUnit and runOnOperation"),
]

AUTHORITY = [
    "THE ULTIMATE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT:",
    "  /Users/nickm/git/deeptools-src   (repo_info.txt: deeptools|master|a0d29abbedfa2dd44ec7255e59440b06a429118c)",
    "which is exactly the revision every banner cites.",
    "",
    "✅ THIS EXTRACT IS VERIFIED UNTRUNCATED: 130 of 130. An INDEPENDENT verifier",
    "(tools/verify_extract3.py) re-derives each function's end from the authority with its own",
    "character-state-machine brace matcher and compares LINE COUNTS, non-whitespace CHARACTER counts",
    "and then full CONTENT; it never re-slices with the extractor's own (file, line, length), which",
    "is the tautology that hid bridge 2's truncation of 366 of 384 bodies. It also asserts each",
    "body's last non-blank line closes its function and was not brace-padded. TWO NEGATIVE CONTROLS",
    "confirm it bites: a tail cut re-balanced with a bare `}` and a brace-neutral 4-line cut from the",
    "middle of a body were both CAUGHT.",
    "",
    "⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. Do not use it.",
    "⛔ dcc/src/Transform/Sentient/ (32,766 lines, passes D29-D75) IS NOT IN THIS CAMPAIGN. Those",
    "   passes rewrite SentientIR IN PLACE before D76 runs. If a function you port depends on state",
    "   they establish -- register assignments, pinned addresses, rerolled loops -- port the",
    "   conversion AS THE REFERENCE WRITES IT and record the dependency in the commit message. Do",
    "   NOT port the pass, and do NOT invent the state.",
]

MAX_SYMS_PORT = 8
MAX_LOC_PORT = 700          # a batch of eight 485-line functions is not a batch
MAX_SYMS_REVIEW = 24
MAX_LOC_REVIEW = 3000


def load_units():
    rows = [l.split("\t") for l in (OUT / "UNITS.tsv").read_text().splitlines()[1:]]
    units = []
    for unit, entry, level, loc, auth, exl, home, callees in rows:
        units.append({
            "name": unit, "level": int(level), "loc": int(loc), "authority": auth,
            "extract_lines": exl, "home": home,
            "callees": [c for c in callees.split(",") if c and c != "-"],
        })
    return units


def item(u, by_name):
    return {
        "name": u["name"],
        "defined_in": SRCFILE,
        "kind": "symbol",
        "source_kind": "function",
        "layer": u["level"],
        "loc": u["loc"],
        "deps": {
            "types": [],
            "symbols": [{"name": by_name[c], "defined_in": SRCFILE, "scope": "port"}
                        for c in u["callees"] if c in by_name],
        },
        "fallback": [],
        "back_fill": [],
        "generates": [],
        "field_anchors": [],
    }


def chunk(items, max_syms, max_loc):
    """Batch on BOTH counts. `ConstructBinaryInstr` alone is 485 lines; eight of its neighbours in
    one batch is the 30-48-turns-per-function failure mode the budget section exists to stop."""
    out, cur, cur_loc = [], [], 0
    for it in items:
        if cur and (len(cur) >= max_syms or cur_loc + it["loc"] > max_loc):
            out.append(cur)
            cur, cur_loc = [], 0
        cur.append(it)
        cur_loc += it["loc"]
    if cur:
        out.append(cur)
    return out


def schedule(units, levels, max_syms, max_loc, wf_path, wf_sha, per_level=True):
    """`per_level` puts a wave barrier between levels so every producer lands before its consumer.

    PORT needs that. REVIEW does not: a reviewer checks an already-ported function against its
    citation, so splitting review by level only buys extra agents.
    """
    by_name = {}
    for u in units:
        by_name[u["name"].split("_", 1)[1]] = u["name"]
    groups = [[u for u in units if u["level"] == lv] for lv in levels] if per_level else [units]
    waves = []
    for sel in groups:
        if not sel:
            continue
        items = [item(u, by_name) for u in sel]
        waves.append({
            "unit_count": len(items),
            "batches": [{"kind": "symbol", "source_file": SRCFILE, "items": b}
                        for b in chunk(items, max_syms, max_loc)],
        })
    return {
        "schema_version": 3,
        "oracle_config": {"path": wf_path, "sha256": wf_sha},
        "api_headers_only": False,
        "budgets": {"max_syms": max_syms, "max_loc": max_loc, "max_types": 5, "min_fields": 20},
        "summary": {
            "unit_count": sum(w["unit_count"] for w in waves),
            # ⛔ THE NUMBER OF DISTINCT ITEM LAYERS, NOT THE NUMBER OF WAVES. crustify's own
            # consistency check is `len({item["layer"] for item in items}) != layer_count`
            # (wave.py:106-107); for a REVIEW schedule the two differ (one wave, several levels) and
            # every review stage died with "schedule summary or batched item identities disagree".
            "layer_count": len({i["layer"] for w in waves for b in w["batches"]
                                for i in b["items"]}),
            "batch_count": sum(len(w["batches"]) for w in waves),
            "file_count": 1,
        },
        "waves": waves,
    }


def main():
    units = load_units()
    camp = OUT / "campaigns/bridge3"
    camp.mkdir(parents=True, exist_ok=True)
    seen = 0
    for name, levels, blurb in STAGES:
        d = camp / name
        d.mkdir(exist_ok=True)
        sel = [u for u in units if u["level"] in levels]
        seen += len(sel)
        lv_desc = ", ".join(f"L{l}={len([u for u in sel if u['level'] == l])}" for l in levels)
        wf = {
            "_comment": [
                f"Narrow inventory for sub-campaign `{name}`: bridge3.cpp LEVEL(S) {levels} — "
                f"{len(sel)} of the span's {TOTAL} functions ({lv_desc}).",
                blurb,
                "impl_files is the same single translation unit as the campaign-wide config (the "
                "campaign has exactly one); the NARROWING is the level range above, which is what "
                "the schedules beside this file select. Every producer level lands before this one "
                "starts.",
            ],
            "campaign_objective": "port",
            "impl_files": [SRCFILE],
            "api_headers": [],
            "out_of_scope": {"paths": ["crustify-bridge3/cpp/prelude.inc"], "features": []},
            "_comment_authority": AUTHORITY,
            "_comment_levels": levels,
        }
        wf_txt = json.dumps(wf, indent=1)
        (d / "wavefront-config.json").write_text(wf_txt)
        wf_path = f"crustify/campaigns/bridge3/{name}/wavefront-config.json"
        wf_sha = hashlib.sha256(wf_txt.encode()).hexdigest()

        p = schedule(sel, levels, MAX_SYMS_PORT, MAX_LOC_PORT, wf_path, wf_sha)
        r = schedule(sel, levels, MAX_SYMS_REVIEW, MAX_LOC_REVIEW, wf_path, wf_sha, per_level=False)
        json.dump(p, open(d / "port.json", "w"), indent=2)
        json.dump(r, open(d / "review.json", "w"), indent=2)
        print(f"  {name}: {len(sel)} units ({lv_desc}), "
              f"{p['summary']['batch_count']} port batches in {p['summary']['layer_count']} waves, "
              f"{r['summary']['batch_count']} review batches")
    assert seen == len(units) == TOTAL, f"stages cover {seen} of {len(units)} units"
    print(f"  every one of the {TOTAL} units is scheduled exactly once")


if __name__ == "__main__":
    main()
