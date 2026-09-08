#!/usr/bin/env python3
"""Rewrite bridge 1's crates.json into the schema crustify actually reads.

The generated file used a simpler shape — `path` plus `modules.<m>.symbols` — and crustify rejected
the whole port stage with "71 selected item(s) have no home `.rs` on disk (no crates.json home)". The
homes existed and UNITS.tsv named them correctly; the placement oracle just could not see them.

The working schema, taken from the bridge-2 campaign's own crates.json:

    crates.<crate>.crate_path            the crate, repo-relative
    crates.<crate>.in_tree / kind / sys_crate / depends_on
    crates.<crate>.modules.<m>.rust_path the module directory, crate-relative
    crates.<crate>.modules.<m>.rs.<file> keyed by CRATE-RELATIVE path, holding
                                          tu, headers, members.functions

One module per home file, so a unit's module and its `.rs` agree by construction.
"""
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
RUST_DIR = "src/bridges/superdsc_to_dataflow_ir"
TU = "crustify-bridge1/cpp/bridge1.cpp"

rows = [l.split("\t") for l in (HERE / "UNITS.tsv").read_text().splitlines()[1:] if l.strip()]
by_home = collections.defaultdict(list)
authority = collections.defaultdict(set)
levels = collections.defaultdict(set)
for entry, level, loc, auth, lines, home, *rest in rows:
    by_home[home].append(entry)
    authority[home].add(auth.split(":")[0])
    levels[home].add(int(level))

old = json.load(open(HERE / "crates.json"))
rs = {}
for home, units in sorted(by_home.items()):
    rs[f"{RUST_DIR}/{home}"] = {
        "_comment": (
            f"{len(units)} unit(s) from {', '.join(sorted(authority[home]))} "
            f"(levels {sorted(levels[home])}). Authority: /Users/nickm/git/deeptools-src/<file>:<line> per unit."
        ),
        "tu": TU,
        "headers": sorted(authority[home]),
        "members": {"functions": sorted(units)},
    }

out = {
    "_comment": old.get("_comment", []) + [
        "",
        "SCHEMA NOTE: this file was regenerated into the shape crustify reads. The first version used",
        "`path` + `modules.<m>.symbols`, and crustify refused the port stage with '71 selected item(s)",
        "have no home .rs on disk (no crates.json home)' even though every home existed and UNITS.tsv",
        "named it. The placement oracle needs crate_path / rust_path / rs.<crate-relative path> /",
        "members.functions, keyed exactly as the bridge-2 campaign's own crates.json keys them.",
    ],
    "crates": {
        "deeptools": {
            "kind": "library",
            "in_tree": True,
            "crate_path": "crates/compiler/deeptools",
            "sys_crate": "crustify/rust/deeptools-sys",
            "depends_on": [],
            "modules": {
                "superdsc_to_dataflow_ir": {"rust_path": RUST_DIR, "rs": rs}
            },
        }
    },
}
(HERE / "crates.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"crates.json: 1 module, {len(rs)} home files, {sum(len(v['members']['functions']) for v in rs.values())} units")
for k, v in rs.items():
    print(f"  {k.rsplit('/', 1)[1]:24} {len(v['members']['functions']):3} units")
