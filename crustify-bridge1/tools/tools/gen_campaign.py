#!/usr/bin/env python3
"""Generate bridge 1's crustify schedules, the wavefront configs and prelude.inc."""
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cppscan

OUT = pathlib.Path("/tmp/bridge1-setup")
SRCFILE = "crustify-bridge1/cpp/bridge1.cpp"

STAGES = [
    ("levels0-1-leaves-and-helpers", [0, 1],
     "the leaves: the DSC accessors, the shape/extent helpers, the addressing arithmetic and the "
     "ddc constraint system every transfer and compute lowering stands on"),
    ("levels2-4-transfer-and-compute", [2, 3, 4],
     "the transfer and compute statements: the load/store/send shuffles, the bit-stream constants, "
     "the vector chains"),
    ("levels5-10-statements-and-drivers", [5, 6, 7, 8, 9, 10],
     "the control flow, the sync statements and the conversion drivers (runTranslator, convertV3/V4)"),
]

AUTHORITY = [
    "THE ULTIMATE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT:",
    "  /Users/nickm/git/deeptools-src   (repo_info.txt: deeptools|master|a0d29abbedfa2dd44ec7255e59440b06a429118c)",
    "which is exactly the revision every banner cites.",
    "",
    "✅ THIS EXTRACT IS VERIFIED UNTRUNCATED. All 110 bodies were checked by an INDEPENDENT",
    "verifier (tools/verify_extract.py) that re-derives each function's end from the authority by",
    "its own brace matcher and compares LENGTHS and CONTENT -- it never re-slices with the",
    "extractor's own (file, line, length), which is the tautology that hid bridge 2's truncation of",
    "366 of 384 bodies. Two negative controls confirm the check fails when a body IS truncated:",
    "a tail cut re-balanced with a bare `}` (caught: 'extract body never closes') and a",
    "brace-balanced 4-line cut from the middle (caught: LENGTH MISMATCH 85 vs 81 lines).",
    "",
    "⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. Do not use it.",
]


def load_units():
    rows = [l.split("\t") for l in (OUT / "UNITS.tsv").read_text().splitlines()[1:]]
    units = []
    for entry, level, loc, auth, exl, home, callees in rows:
        units.append({
            "name": entry, "level": int(level), "loc": int(loc), "authority": auth,
            "extract_lines": exl, "home": home,
            "callees": [c for c in callees.split(",") if c],
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
            "symbols": [
                {"name": by_name[c], "defined_in": SRCFILE, "scope": "port"}
                for c in u["callees"] if c in by_name
            ],
        },
        "fallback": [],
        "back_fill": [],
        "generates": [],
        "field_anchors": [],
    }


def schedule(units, levels, max_syms, max_loc, wf_path, wf_sha, per_level=True):
    """`per_level` puts a wave barrier between levels, so every producer lands before its consumer.

    PORT needs that. REVIEW does not: a reviewer checks a already-ported function against its
    citation, so splitting review by level only buys extra agents — the last stage's 6 levels would
    become 6 batches of 1-4 units each. One wave, batched by max_syms, is the same work in a third
    of the agents.
    """
    waves = []
    by_name = {}
    for u in units:
        by_name[u["name"].split("_", 1)[1]] = u["name"]
    groups = [[u for u in units if u["level"] == lv] for lv in levels] if per_level else [units]
    for sel in groups:
        if not sel:
            continue
        items = [item(u, by_name) for u in sel]
        batches = [
            {"kind": "symbol", "source_file": SRCFILE, "items": items[i:i + max_syms]}
            for i in range(0, len(items), max_syms)
        ]
        waves.append({"unit_count": len(items), "batches": batches})
    return {
        "schema_version": 3,
        "oracle_config": {"path": wf_path, "sha256": wf_sha},
        "api_headers_only": False,
        "budgets": {"max_syms": max_syms, "max_loc": max_loc, "max_types": 5, "min_fields": 20},
        "summary": {
            "unit_count": sum(w["unit_count"] for w in waves),
            "layer_count": len(waves),
            "batch_count": sum(len(w["batches"]) for w in waves),
            "file_count": 1,
        },
        "waves": waves,
    }


def main():
    units = load_units()
    camp = OUT / "campaigns/bridge1"
    camp.mkdir(parents=True, exist_ok=True)

    for name, levels, blurb in STAGES:
        d = camp / name
        d.mkdir(exist_ok=True)
        sel = [u for u in units if u["level"] in levels]
        lv_desc = ", ".join(f"L{l}={len([u for u in sel if u['level']==l])}" for l in levels)
        wf = {
            "_comment": [
                f"Narrow inventory for sub-campaign `{name}`: bridge1.cpp LEVEL(S) {levels} -- "
                f"{len(sel)} of the span's {len(units)} functions ({lv_desc}).",
                blurb,
                "impl_files is the same single translation unit as the campaign-wide config (the "
                "campaign has exactly one); the NARROWING is the level range above, which is what "
                "the schedules beside this file select. Every producer level lands before this one "
                "starts.",
            ],
            "campaign_objective": "port",
            "impl_files": [SRCFILE],
            "api_headers": [],
            "out_of_scope": {"paths": ["crustify-bridge1/cpp/prelude.inc"], "features": []},
            "_comment_authority": AUTHORITY,
            "_comment_levels": levels,
        }
        wf_txt = json.dumps(wf, indent=1)
        (d / "wavefront-config.json").write_text(wf_txt)
        wf_path = f"crustify/campaigns/bridge1/{name}/wavefront-config.json"
        wf_sha = hashlib.sha256(wf_txt.encode()).hexdigest()

        p = schedule(sel, levels, 8, 320, wf_path, wf_sha)
        r = schedule(sel, levels, 24, 960, wf_path, wf_sha, per_level=False)
        json.dump(p, open(d / "port.json", "w"), indent=2)
        json.dump(r, open(d / "review.json", "w"), indent=2)
        print(f"  {name}: {len(sel)} units, "
              f"{p['summary']['batch_count']} port batches, "
              f"{r['summary']['batch_count']} review batches, "
              f"{p['summary']['layer_count']} waves")

    # ---- prelude.inc: names and shapes only, no behaviour.
    text = (OUT / "cpp/bridge1.cpp").read_text()
    own = {re.match(r"e\d+_(.*)", u["name"]).group(1) for u in units}
    own |= {u["name"] for u in units}
    called = cppscan.calls_in(text)
    external = sorted(c for c in called
                      if c not in own and not re.match(r"^e\d{3}_", c)
                      and c not in {"if", "for", "while", "switch", "return", "sizeof", "catch"})
    scopes = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)::", cppscan.prepared(text))))
    types = sorted(set(re.findall(r"\b([A-Z][A-Za-z_0-9]*)\b", cppscan.prepared(text))))
    pre = [
        "// prelude.inc -- plain-C++ stand-ins for every name bridge1.cpp's bodies reach.",
        "//",
        "// ⛔ THIS IS NOT A PORTING INPUT. It carries NAMES AND SHAPES only, so the translation",
        "// unit stands alone with no MLIR, LLVM or dcc header. The behaviour being ported lives in",
        "// the bodies; nothing here has any.",
        "//",
        "// ⚠ The unit is NOT compile-clean, exactly as bridge 2's was not: these bodies do heavy",
        "// member access on opaque reference types, which no stand-in can satisfy without the real",
        "// headers. The bodies are verbatim and verified untruncated -- that is the load-bearing",
        "// property. Use the banners for WHICH function and WHAT ORDER, and port from the authority.",
        "#pragma once",
        "#include <cstdint>",
        "#include <map>",
        "#include <memory>",
        "#include <optional>",
        "#include <set>",
        "#include <string>",
        "#include <unordered_map>",
        "#include <unordered_set>",
        "#include <utility>",
        "#include <vector>",
        "",
        f"// ---- {len(scopes)} qualified scopes reached by the bodies",
    ]
    for s in scopes[:400]:
        pre.append(f"// scope: {s}")
    pre.append("")
    pre.append(f"// ---- {len(external)} external functions the bodies call")
    for c in external:
        pre.append(f"// extern: {c}")
    (OUT / "cpp/prelude.inc").write_text("\n".join(pre) + "\n")
    print(f"  prelude.inc: {len(scopes)} scopes, {len(external)} external calls catalogued")


if __name__ == "__main__":
    main()
