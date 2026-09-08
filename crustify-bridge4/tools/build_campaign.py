#!/usr/bin/env python3
"""Generate bridge 4's Rust homes, schedules and placement oracle from UNITS.tsv.

All three come from the one table so they cannot disagree — the failure that cost bridge 1 three
relaunches was a crates.json whose homes did not match the files on disk.

⛔ THE PLACEMENT ORACLE GOES AT THE REPO TIER, `crustify/crates.json`, NOT in the campaign directory.
A worktree branched from another campaign's branch INHERITS that campaign's copy, and crustify reads
the repo-tier one — so bridge 1's port stage refused 71 units with "no home .rs on disk" while every
home existed and its UNITS.tsv named them correctly.
"""
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent          # crustify-bridge4/
W = HERE.parent                                                 # the worktree
RUST_DIR = "src/bridges/progir_to_senprog"
TU = "crustify-bridge4/cpp/bridge4.cpp"

rows = [l.split("\t") for l in (HERE / "UNITS.tsv").read_text().splitlines()[1:] if l.strip()]
# entry, unit, qname, authority_file, auth_start, auth_end, loc, extract_start, extract_end, callees
units = []
for r in rows:
    n, unit, qname, af, a0, a1, loc, e0, e1 = r[:9]
    callees = r[9] if len(r) > 9 else ""
    units.append(dict(n=int(n), unit=unit, qname=qname, af=af, a0=a0, loc=int(loc), callees=callees))

# ── one home per authority file, so a unit's module and its .rs agree by construction ──────────────
HOMES = {
    "sys-arch-spec/progir/progir.h": "progir_inline.rs",
    "sys-arch-spec/progir/progir.cpp": "progir_print.rs",
    "sys-arch-spec/dpc/dpc.cpp": "senprog_writer.rs",
    "sys-arch-spec/isa/isa.cpp": "isa_fields.rs",
}
by_home = collections.defaultdict(list)
for u in units:
    by_home[HOMES.get(u["af"], "misc.rs")].append(u)

# ── the Rust homes, each carrying its units' crustify:todo anchors ─────────────────────────────────
dst = W / "crates/compiler/deeptools/src/bridges/progir_to_senprog"
dst.mkdir(parents=True, exist_ok=True)
for home, us in sorted(by_home.items()):
    body = [
        "// SPDX-License-Identifier: Apache-2.0",
        f"//! Ported from `{us[0]['af']}` — {len(us)} unit(s) of bridge 4, `ProgIR -> SenProg`.",
        "//!",
        "//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,",
        "//! spelling and separators. A predicate that decides what to write, without writing it, is not",
        "//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.",
        "",
    ]
    for u in sorted(us, key=lambda x: x["n"]):
        body.append(f"// crustify:todo: {u['unit']}")
        body.append(f"//   authority: {u['af']}:{u['a0']}  ({u['loc']} lines)  `{u['qname']}`")
        body.append("")
    (dst / home).write_text("\n".join(body) + "\n")

mod = [
    "// SPDX-License-Identifier: Apache-2.0",
    "//! `ProgIR -> SenProg` — bridge 4, ported from `sys-arch-spec/{dpc,progir,isa}`.",
    "//!",
    "//! ⭐ THE EMISSION'S CLOSURE IS FOUR FILES, NOT ONE. `dpc.cpp` holds only 3 of the 33 units;",
    "//! `progir.h` holds 18 as IN-CLASS definitions, which a `.cpp`-only scan would miss entirely.",
    "",
]
for home in sorted(by_home):
    mod.append(f"pub mod {home[:-3]};")
(dst / "mod.rs").write_text("\n".join(mod) + "\n")

# ── the placement oracle, repo tier, in the schema crustify reads ──────────────────────────────────
rs = {}
for home, us in sorted(by_home.items()):
    rs[f"{RUST_DIR}/{home}"] = {
        "_comment": f"{len(us)} unit(s) from {us[0]['af']}. Authority: /Users/nickm/git/deeptools-src/<file>:<line> per unit.",
        "tu": TU,
        "headers": sorted({u["af"] for u in us}),
        "members": {"functions": sorted(u["unit"] for u in us)},
    }
oracle = {
    "_comment": [
        "Placement oracle for the bridge-4 port (ProgIR -> SenProg). REPO TIER — crustify reads this",
        "file, not the campaign directory's copy, and this worktree inherited bridge 2's until now.",
    ],
    "crates": {
        "deeptools": {
            "kind": "library",
            "in_tree": True,
            "crate_path": "crates/compiler/deeptools",
            "sys_crate": "crustify/rust/deeptools-sys",
            "depends_on": [],
            "modules": {"progir_to_senprog": {"rust_path": RUST_DIR, "rs": rs}},
        }
    },
}
(W / "crustify/crates.json").write_text(json.dumps(oracle, indent=1) + "\n")

# ── one schedule, batches of 8, callee-before-caller order ─────────────────────────────────────────
ordered = sorted(units, key=lambda u: u["n"])
batches = [
    {
        "kind": "symbol",
        "source_file": TU,
        "items": [
            {"name": u["unit"], "defined_in": TU, "kind": "symbol", "source_kind": "function",
             "layer": 0, "loc": u["loc"], "deps": {"types": [], "symbols": []},
             "fallback": [], "back_fill": []}
            for u in ordered[i:i + 8]
        ],
    }
    for i in range(0, len(ordered), 8)
]
camp = W / "crustify/campaigns/bridge4/progir-to-senprog"
camp.mkdir(parents=True, exist_ok=True)

# ⛔ `oracle_config` IS AN OBJECT WITH A CHECKSUM, NOT A PATH STRING. A string is rejected outright:
# "translate: invalid schedule; expected exactly one of oracle_config or legacy oracle_target".
wf = {
    "_comment": [
        "Narrow inventory for bridge 4, `ProgIR -> SenProg`: all 33 units of the emission's transitive",
        "closure, in one sub-campaign because they sit at a single dependency layer.",
        "impl_files is the one verified translation unit; prelude.inc is not a porting input.",
    ],
    "campaign_objective": "port",
    "impl_files": [TU],
    "api_headers": [],
    "out_of_scope": {"paths": ["crustify-bridge4/cpp/prelude.inc"], "features": []},
    "_comment_authority": [
        "THE ULTIMATE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT:",
        "  /Users/nickm/git/deeptools-src  (repo_info.txt: deeptools|master|a0d29abbed…)",
        "",
        "✅ THIS EXTRACT IS VERIFIED VERBATIM, 33 of 33, by tools/verify_extract.py, which shares no",
        "code with the extractor: it re-derives each body's end with its own scanner and compares",
        "length AND content, asserts brace depth reaches zero at the last character, and never",
        "appends a brace. Both negative controls fail as required. It caught a real defect en route:",
        "24 header in-class bodies had lost their leading indentation.",
        "",
        "⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. Do not use it.",
    ],
}
wf_path = camp / "wavefront-config.json"
wf_path.write_text(json.dumps(wf, indent=1) + "\n")
oracle_ref = {
    "path": "crustify/campaigns/bridge4/progir-to-senprog/wavefront-config.json",
    "sha256": __import__("hashlib").sha256(wf_path.read_bytes()).hexdigest(),
}

sched = {
    "schema_version": 3,
    "oracle_config": oracle_ref,
    "api_headers_only": False,
    "budgets": {"max_syms": 8, "max_loc": 320, "max_types": 5, "min_fields": 20},
    "summary": {"unit_count": len(ordered), "layer_count": 1,
                "batch_count": len(batches), "file_count": 1},
    "waves": [{"unit_count": len(ordered), "batches": batches}],
}
json.dump(sched, open(camp / "port.json", "w"), indent=1)
rev = json.loads(json.dumps(sched))
rev["budgets"]["max_syms"] = 24
rev["waves"][0]["batches"] = [
    {"kind": "symbol", "source_file": TU,
     "items": [it for b in batches for it in b["items"]][i:i + 24]}
    for i in range(0, len(ordered), 24)
]
rev["summary"]["batch_count"] = len(rev["waves"][0]["batches"])
json.dump(rev, open(camp / "review.json", "w"), indent=1)

print(f"homes:      {len(by_home)} files + mod.rs -> {dst.relative_to(W)}")
for h, us in sorted(by_home.items()):
    print(f"   {h:22} {len(us):3} units  ({us[0]['af']})")
print(f"oracle:     crustify/crates.json, {sum(len(v['members']['functions']) for v in rs.values())} units")
print(f"schedules:  port {len(batches)} batches / review {len(rev['waves'][0]['batches'])} batches")
