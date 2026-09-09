#!/usr/bin/env python3
"""Enumerate every portable function definition in the ddc + L3-scheduler scope.

BY BRACE MATCHING from each signature's opening paren (cppscan.find_defs), never by a signature
regex: a signature regex undercounted bridge 2 by 72%. Headers are scanned too -- on bridge 4
progir.h held 18 of 33 units and a .cpp-only scan would have missed every one; here ddc.h is 805
lines and ddl_conversion.h 562, so the same trap is live.

Writes work/inscope.json (in scope) and EXCLUSIONS.tsv (out, with a reason each).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import re
from collections import Counter

import cppscan

AUTH = _paths.AUTH_ROOT
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
ASSIGN_ONLY = re.compile(r"^\{\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^;]*);\s*\}$", re.S)
# `main` never appears (no standalone file is in SCOPE) but guard anyway.
DRIVER = {"main"}


def classify(d, rel):
    """Return (reason, rule) if the def is OUT of scope, else None."""
    n, bl = d["name"], d["body_lines"]
    body = d["body"].strip()

    if (rel, n) in _paths.ALREADY_PORTED:
        return ("ALREADY PORTED on bridge1-campaign -- re-porting it would create a second, "
                "diverging implementation of the same authority citation", "already_ported")
    named = _paths.NAMED_EXCLUSIONS.get((rel, n))
    if named:
        return (named, "named_exclusion")
    if n in DRIVER:
        return ("command-line driver entry point, not part of the runDdc path", "driver_main")
    if PASS_FACTORY.match(n):
        return ("MLIR pass factory: #[forward] runs the whole pipeline at macro expansion, so no "
                "pass manager, pass instance or factory exists at run time", "mlir_pass_factory")
    if n in PASS_META:
        return ("MLIR pass metadata (command-line name/description/dialect registration) -- no "
                "behaviour to port", "mlir_pass_metadata")
    if POPULATE.match(n):
        return ("MLIR pattern/pass registration helper -- no rewriter or pattern driver exists at "
                "run time", "mlir_pattern_registration")
    if n == "classof":
        return ("LLVM RTTI `classof` hook -- a Rust enum discriminant match, not a function",
                "llvm_rtti_hook")

    cls = d["cls"]
    if bl <= 3 and cls and n in (cls, "~" + cls):
        return ("trivial ctor/dtor of %s (%d-line body) -- a Rust struct literal" % (cls, bl),
                "trivial_ctor_dtor")

    if bl <= 3:
        am = ASSIGN_ONLY.match(body)
        # A setter that READS ITS OWN PRIOR VALUE is an accumulate, not an accessor: the
        # constraint join `min_ = min_ ? std::max(*min_, v) : v` in
        # L3DlOpsScheduler.h::Metadata::Datastage::Constraints is exactly that, and the absolute
        # extents it feeds are a CONSTRAINT SYSTEM, not a formula. Those stay IN scope.
        if am and re.search(r"\b" + re.escape(am.group(1)) + r"\b", am.group(2)):
            return None
        if am or RET_ONLY.match(body) or ACCESSOR_NAME.match(n):
            return ("%d-line field accessor -- a struct field in Rust, not a function" % bl,
                    "field_accessor")
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    inscope, excluded = [], []
    missing = []
    for rel, scope, mod, stem in _paths.SCOPE:
        p = os.path.join(AUTH, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        text = open(p, encoding="utf-8", errors="replace").read()
        classes = cppscan.classes_in(text)
        for d in cppscan.find_defs(text):
            q = d["qual"].rstrip(":")
            # A `Class::method` definition names its class; an IN-CLASS definition does not, so the
            # enclosing class/struct is recovered from the brace structure instead.
            d["cls"] = q.split("::")[-1] if q else cppscan.enclosing_class(classes, d["head_off"])
            d["rel"] = rel
            d["scope"] = scope
            d["module"] = mod
            d["stem"] = stem
            d["rust_home"] = _paths.home_of(mod, stem)
            r = classify(d, rel)
            if r:
                d["reason"], d["rule"] = r
                excluded.append(d)
            else:
                inscope.append(d)
    if missing:
        raise SystemExit("MISSING authority file(s): %s" % missing)

    inscope.sort(key=lambda d: (d["rel"], d["head_line"]))
    json.dump([{k: v for k, v in d.items() if k != "body"} for d in inscope],
              open(OUT + "/inscope.json", "w"), indent=0)
    with open(os.path.join(_paths.CAMP, "EXCLUSIONS.tsv"), "w") as fh:
        fh.write("symbol\tqualifier\tauthority\tbody_lines\trule\treason\n")
        fh.write("\t".join(_paths.STAGE1_EXCLUSION) + "\n")
        for d in sorted(excluded, key=lambda d: (d["rel"], d["head_line"])):
            fh.write("\t".join([d["name"], d["qual"].rstrip(":") or d["cls"],
                                "%s:%d" % (d["rel"], d["head_line"]),
                                str(d["body_lines"]), d["rule"], d["reason"]]) + "\n")

    print("files scanned      : %d" % len(_paths.SCOPE))
    print("definitions found  : %d" % (len(inscope) + len(excluded)))
    print("IN SCOPE           : %d" % len(inscope))
    print("EXCLUDED           : %d" % len(excluded))
    for k, v in Counter(d["rule"] for d in excluded).most_common():
        print("   %-26s %d" % (k, v))
    print("in-scope body lines: %d" % sum(d["body_lines"] for d in inscope))
    print("by scope:")
    for k, v in Counter(d["scope"] for d in inscope).most_common():
        print("   %-6s %4d units" % (k, v))
    print("by file:")
    per = Counter(d["rel"] for d in inscope)
    for rel, _s, _m, _st in _paths.SCOPE:
        if per.get(rel):
            print("   %-56s %4d" % (rel, per[rel]))


if __name__ == "__main__":
    main()
