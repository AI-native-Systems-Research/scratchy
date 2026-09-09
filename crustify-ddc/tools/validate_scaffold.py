#!/usr/bin/env python3
"""Structural check on the generated Rust scaffolding, before cargo ever sees it.

UNITS.tsv column 7 is rust_home for this campaign, as it was for senpass.

* every `pub(crate) mod X;` a module declares has a file on disk
* every unit in UNITS.tsv has exactly one anchor, in its recorded home
* the scaffold contains ONLY comments and module declarations, so `cargo check` cannot fail on it
* lib.rs reaches the new tree
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

TREE = _paths.TREE
OUT = os.path.join(TREE, _paths.OUTDIR)
CAMP = _paths.CAMP

ALLOWED = re.compile(r"^\s*$|^\s*//|^\s*pub\(crate\) mod [a-z][a-z0-9_]*;\s*$")

def main():
    bad = []
    rs = []
    for dirpath, _, names in os.walk(OUT):
        for n in names:
            if n.endswith(".rs"):
                rs.append(os.path.join(dirpath, n))
    for p in rs:
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if not ALLOWED.match(line):
                bad.append(f"{os.path.relpath(p, TREE)}:{i}: not a comment or mod decl: {line[:70]!r}")
        for m in re.finditer(r"^pub\(crate\) mod ([a-z][a-z0-9_]*);", open(p, encoding="utf-8").read(), re.M):
            sub = m.group(1)
            d = os.path.dirname(p)
            if not (os.path.exists(f"{d}/{sub}.rs") or os.path.exists(f"{d}/{sub}/mod.rs")):
                bad.append(f"{os.path.relpath(p, TREE)}: declares `mod {sub};` with no file")

    anchors = {}
    for p in rs:
        s = open(p, encoding="utf-8").read()
        for u in re.findall(r"crustify:todo:\s*(e\d+_[A-Za-z0-9_]+)", s):
            anchors.setdefault(u, []).append(os.path.relpath(p, TREE))
    want = {}
    with open(os.path.join(CAMP, "UNITS.tsv")) as fh:
        next(fh)
        for row in fh:
            f = row.rstrip("\n").split("\t")
            want[f[0]] = f[7]
    for u, home in want.items():
        got = anchors.get(u, [])
        if len(got) != 1:
            bad.append(f"{u}: {len(got)} anchors (expected 1)")
        elif got[0] != home:
            bad.append(f"{u}: anchored in {got[0]}, UNITS.tsv says {home}")
    extra = set(anchors) - set(want)
    for u in sorted(extra):
        bad.append(f"{u}: anchored but not in UNITS.tsv")

    lib = os.path.join(TREE, "crates/compiler/deeptools/src/lib.rs")
    if os.path.exists(lib) and "pub mod schedule;" not in open(lib, encoding="utf-8").read():
        bad.append("lib.rs does not declare `pub mod schedule;` - the new tree is unreachable")

    print(f"scaffold .rs files : {len(rs)}")
    print(f"anchors            : {len(anchors)} distinct, {len(want)} expected")
    if bad:
        print(f"⛔ {len(bad)} PROBLEM(S)")
        for b in bad[:25]:
            print("   " + b)
        sys.exit(1)
    print("✅ scaffolding is structurally sound: comments and mod declarations only, every "
          "declared module present, one anchor per unit in its recorded home")


if __name__ == "__main__":
    main()
