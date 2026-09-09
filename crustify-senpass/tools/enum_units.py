#!/usr/bin/env python3
"""Enumerate every portable function definition in dcc/src/Transform/Sentient.

BY BRACE MATCHING from each signature's opening paren (cppscan.find_defs), never by a signature
regex: a signature regex undercounted bridge 2 by 72%. Headers are scanned too - on bridge 4
progir.h held 18 of 33 units and a .cpp-only scan would have missed every one.

Writes work/inscope.json (in scope) and work/EXCLUSIONS.tsv (out, with a reason each).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import re
from collections import Counter

import cppscan

SENT = _paths.SENT
REL = _paths.REL
OUT = _paths.WORK

PASS_FACTORY = re.compile(r"^create[A-Za-z0-9_]*Pass$")
PASS_META = {
    "getArgument", "getDescription", "getName", "getDependentDialects",
    "getAnalysisUsage", "getArgumentName", "getPassName", "clonePass",
}
POPULATE = re.compile(r"^populate[A-Za-z0-9_]*(Patterns|Passes)$")
ACCESSOR_NAME = re.compile(
    r"^(get|set|is|has|use|are|should|can|with)[A-Z_]"
    r"|^(size|empty|begin|end|front|back|data|count|clear|reset|value|kind|name|str|"
    r"first|second|length|at|find|contains|insert|push_back|emplace)$"
    r"|^operator"
)
RET_ONLY = re.compile(r"^\{\s*(return\b[^;]*;)?\s*\}$", re.S)
ASSIGN_ONLY = re.compile(r"^\{\s*[A-Za-z_][A-Za-z_0-9]*_?\s*=\s*[^;]*;\s*\}$", re.S)


def classify(d):
    """Return (reason, rule) if the def is OUT of scope, else None."""
    n, q, bl = d["name"], d["qual"].rstrip(":"), d["body_lines"]
    body = d["body"].strip()
    if PASS_FACTORY.match(n):
        return ("MLIR pass factory: #[forward] runs the whole pipeline at macro expansion, so no "
                "pass manager, pass instance or factory exists at run time", "mlir_pass_factory")
    if n in PASS_META:
        return ("MLIR pass metadata (command-line name/description/dialect registration) - no "
                "behaviour to port", "mlir_pass_metadata")
    if POPULATE.match(n):
        return ("MLIR pattern/pass registration helper - no rewriter or pattern driver exists at "
                "run time", "mlir_pattern_registration")
    if n == "classof":
        return ("LLVM RTTI `classof` hook - a Rust enum discriminant match, not a function",
                "llvm_rtti_hook")
    cls = q.split("::")[-1] if q else ""
    if bl <= 3 and cls and n in (cls, "~" + cls):
        return (f"trivial ctor/dtor of {cls} ({bl}-line body) - a Rust struct literal",
                "trivial_ctor_dtor")
    if bl <= 3 and (RET_ONLY.match(body) or ASSIGN_ONLY.match(body) or ACCESSOR_NAME.match(n)):
        return (f"{bl}-line field accessor - a struct field in Rust, not a function",
                "field_accessor")
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    files = (sorted(glob.glob(SENT + "/*.cpp")) + sorted(glob.glob(SENT + "/*.hpp"))
             + sorted(glob.glob(SENT + "/*.h")))
    inscope, excluded = [], []
    for f in files:
        base = os.path.basename(f)
        text = open(f, encoding="utf-8", errors="replace").read()
        for d in cppscan.find_defs(text):
            d["file"] = base
            r = classify(d)
            if r:
                d["reason"], d["rule"] = r
                excluded.append(d)
            else:
                inscope.append(d)
    inscope.sort(key=lambda d: (d["file"], d["head_line"]))
    json.dump([{k: v for k, v in d.items() if k not in ("body",)} for d in inscope],
              open(OUT + "/inscope.json", "w"), indent=0)
    with open(os.path.join(_paths.CAMP, "EXCLUSIONS.tsv"), "w") as fh:
        fh.write("symbol\tqualifier\tauthority\tbody_lines\trule\treason\n")
        for d in sorted(excluded, key=lambda d: (d["file"], d["head_line"])):
            fh.write("\t".join([d["name"], d["qual"].rstrip(":"),
                                f"{REL}/{d['file']}:{d['head_line']}",
                                str(d["body_lines"]), d["rule"], d["reason"]]) + "\n")
    print(f"files scanned      : {len(files)}")
    print(f"definitions found  : {len(inscope) + len(excluded)}")
    print(f"IN SCOPE           : {len(inscope)}")
    print(f"EXCLUDED           : {len(excluded)}")
    for k, v in Counter(d["rule"] for d in excluded).most_common():
        print(f"   {k:26s} {v}")
    print(f"in-scope body lines: {sum(d['body_lines'] for d in inscope)}")
    per = Counter(d["file"] for d in inscope)
    print("top files by in-scope units:")
    for k, v in per.most_common(12):
        print(f"   {k:46s} {v}")


if __name__ == "__main__":
    main()
