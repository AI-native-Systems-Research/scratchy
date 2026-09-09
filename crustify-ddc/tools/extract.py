#!/usr/bin/env python3
"""Consolidate the campaign's C++ into ONE self-contained TU PER SCOPE, bodies VERBATIM.

Writes campaign/cpp/{l3,ddc,ddl}.cpp (banner + rewritten signature + verbatim body, in dependency
order), campaign/cpp/prelude.inc (plain-C++ stand-ins, no MLIR/LLVM/deeptools header anywhere) and
campaign/UNITS.tsv.

⛔ The signature is rewritten to the unit symbol `eNNN_name` so that symbol == unit name for EVERY
entry. Bridge 2 left 116 of its 384 entries carrying their bare C++ name, which is why its own
crates.json had to warn "ALWAYS resolve a unit through UNITS.tsv". The BODY -- `{` through its
matching `}` -- is copied byte for byte and is never reflowed, re-indented or brace-balanced.

⛔ Entry numbering is GLOBAL across the three TUs (e001..eNNN in one dependency ladder), because
the ladder itself is global: 31 call edges cross scopes. The TU split is for readability and for
crustify's per-batch `source_file`, not a second ordering.
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
WORK = _paths.WORK
CAMP = _paths.CAMP
REV = _paths.REV

RULE = "// " + "-" * 96


def rewrite_head(u, sym):
    """The extract's signature: the original head with the qualified name replaced by `sym`, and
    C++ member specifiers that have no meaning at file scope removed."""
    head = u["head"]
    qual = u["qual"]
    name = u["name"]
    pat = (re.escape(qual) + r"\s*" if qual else "") + re.escape(name) + r"(\s*\()"
    m = None
    for m in re.finditer(pat, head):
        pass
    if m is None:
        raise SystemExit("cannot locate signature name for %s in head: %r"
                         % (u["unit"], head[:160]))
    head = head[:m.start()] + sym + m.group(1) + head[m.end():]
    head = re.sub(r"^\s*(virtual|static|inline|explicit)\s+", "", head)
    head = re.sub(r"\b(override|final)\b", "", head)
    return head.rstrip()


def main():
    units = json.load(open(WORK + "/ordered.json"))
    os.makedirs(CAMP + "/cpp", exist_ok=True)
    text_of = {}
    for u in units:
        p = os.path.join(AUTH, u["rel"])
        if p not in text_of:
            text_of[p] = open(p, encoding="utf-8", errors="replace").read()

    n = len(units)
    by_scope = {s: [] for s in _paths.SCOPES}
    for u in units:
        u["note"] = _paths.NOTES.get((u["rel"], u["name"]), "")
        by_scope[u["scope"]].append(u)

    SCOPE_WHAT = {
        "l3": ("STAGE 1 of runDdc: L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, "
               "verbose).run(sdsc)", "dcg/dcg_fe/scheduler/"),
        "ddc": ("STAGE 2 of runDdc: ddc::Ddc(dscGlobal, ..).run_v1(sdsc)", "ddc/"),
        "ddl": ("STAGE 2's DDL conversion sub-surface, reached from Ddc", "ddc/ddl/"),
        "dcg": ("STAGE 3: DcgManager::runDcgForDlOpsStandalone(sdsc) -- the branch "
                "SchedulerStages.cpp:53-57 takes whenever dscs_ is non-empty (always, for us). "
                "NOT runDcg.", "dcg/dcg_manager/"),
    }

    for scope in _paths.SCOPES:
        us = by_scope[scope]
        what, where = SCOPE_WHAT[scope]
        out = []
        out.append("// " + "=" * 96)
        out.append("// DDC / L3-SCHEDULER CAMPAIGN - consolidated translation unit: %s" % scope)
        out.append("//")
        out.append("// %s" % what)
        out.append("//   dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41 is the call site.")
        out.append("//")
        out.append("// %d of the campaign's %d function bodies, from %s, extracted VERBATIM"
                   % (len(us), n, where))
        out.append("// and emitted in dependency order over the SCC condensation (level 0 calls")
        out.append("// nothing else in the span; level N only calls levels below N).")
        out.append("// Entry numbers are GLOBAL across the campaign's three TUs - one ladder.")
        out.append("//")
        out.append("// AUTHORITY, and the ONLY thing to port from: %s" % AUTH)
        out.append("//   revision %s" % REV)
        out.append("// The other local deeptools checkout is a DIFFERENT revision. Do not use it.")
        out.append("// The pod is not reachable from this host and is not the authority here.")
        out.append("//")
        out.append("// Each entry's banner gives its unit symbol, level, and original file:line. The")
        out.append("// signature is rewritten to the unit symbol so symbol == unit name; the body")
        out.append("// between `{` and its matching `}` is byte-for-byte the authority's.")
        out.append("// " + "=" * 96)
        out.append("")
        out.append('#include "prelude.inc"')
        out.append("")

        for u in us:
            sym = u["unit"]
            head = rewrite_head(u, sym)
            body = text_of[os.path.join(AUTH, u["rel"])][u["open_off"]:u["close_off"] + 1]
            out.append(RULE)
            out.append("// entry %03d/%d   level %d   scc %d   %d body lines"
                       % (u["entry"], n, u["level"], u["scc"], u["body_lines"]))
            out.append("// unit: %s" % sym)
            out.append("// authority: %s:%d" % (u["rel"], u["head_line"]))
            orig = re.sub(r"\s+", " ", u["head"]).strip()
            out.append("// original: %s" % orig[:400])
            if u["cls"]:
                out.append("// class: %s" % u["cls"])
            out.append("// rust home: %s" % u["rust_home"])
            if u["note"]:
                for i, chunk in enumerate(_wrap(u["note"], 92)):
                    out.append("// %s %s" % ("NOTE:" if i == 0 else "     ", chunk))
            out.append(RULE)
            start = len(out) + 1  # 1-based line of the first signature line
            out.extend(head.split("\n"))
            out.extend(body.split("\n"))
            u["extract_lines"] = "%d-%d" % (start, len(out))
            u["extract_file"] = _paths.extract_rel(scope)
            out.append("")

        open(CAMP + "/cpp/%s.cpp" % scope, "w").write("\n".join(out) + "\n")

    # -- prelude: a NAME INVENTORY of every external callee the bodies mention -----------------
    defined = {u["name"] for u in units} | {u["unit"] for u in units}
    ext = Counter()
    for u in units:
        body = text_of[os.path.join(AUTH, u["rel"])][u["open_off"]:u["close_off"] + 1]
        for c in cppscan.calls_in(body):
            if c not in defined:
                ext[c] += 1
    pre = []
    pre.append("// prelude.inc - plain-C++ stand-ins for the names the bodies call.")
    pre.append("//")
    pre.append("// NO MLIR, LLVM OR deeptools HEADER IS INCLUDED ANYWHERE IN THIS CAMPAIGN. This")
    pre.append("// file is a NAME INVENTORY, not a compilable prelude: the stand-ins carry names and")
    pre.append("// arity only, so each consolidated TU stands alone textually. It is declared")
    pre.append("// out_of_scope in wavefront-config.json and NOTHING IN IT IS TO BE PORTED - the")
    pre.append("// behaviour to port lives entirely in the bodies in l3.cpp / ddc.cpp / ddl.cpp.")
    pre.append("//")
    pre.append("// %d distinct external names, %d call sites." % (len(ext), sum(ext.values())))
    pre.append("")
    pre.append("struct Any {")
    pre.append("  template <class... A> Any operator()(A &&...) const;")
    pre.append("  template <class T> operator T() const;")
    pre.append("  Any operator*() const; Any operator->() const; Any operator[](Any) const;")
    pre.append("};")
    pre.append("")
    for name, cnt in sorted(ext.items(), key=lambda kv: (-kv[1], kv[0])):
        pre.append("extern Any %s;  // %d call site(s)" % (name, cnt))
    open(CAMP + "/cpp/prelude.inc", "w").write("\n".join(pre) + "\n")

    # -- UNITS.tsv ----------------------------------------------------------------------------
    with open(CAMP + "/UNITS.tsv", "w") as fh:
        fh.write("unit\tentry\tlevel\tscc\tloc\tauthority (under %s)\textract_lines\trust_home\t"
                 "scope\textract_file\tclass\tcalls\tnote\n" % AUTH)
        for u in units:
            fh.write("\t".join([
                u["unit"], "%03d/%d" % (u["entry"], n), str(u["level"]), str(u["scc"]),
                str(u["body_lines"]), "%s:%d" % (u["rel"], u["head_line"]),
                u["extract_lines"], u["rust_home"], u["scope"], u["extract_file"],
                u["cls"] or "-",
                ",".join(u["call_units"]) if u["call_units"] else "-",
                u["note"] or "-",
            ]) + "\n")

    json.dump(units, open(WORK + "/units.json", "w"), indent=0)

    homes = Counter(u["rust_home"] for u in units)
    print("units              : %d" % n)
    print("prelude names      : %d" % len(ext))
    for scope in _paths.SCOPES:
        p = CAMP + "/cpp/%s.cpp" % scope
        print("   cpp/%-8s %5d units  %7d lines" % (scope + ".cpp", len(by_scope[scope]),
                                                    open(p).read().count("\n") + 1))
    print("rust homes         : %d" % len(homes))
    for k, v in sorted(homes.items(), key=lambda kv: -kv[1]):
        print("   %4d  %s" % (v, k))
    print("units carrying a NOTE: %d" % sum(1 for u in units if u["note"]))


def _wrap(s, w):
    words, line, out = s.split(), "", []
    for x in words:
        if len(line) + len(x) + 1 > w:
            out.append(line)
            line = x
        else:
            line = (line + " " + x) if line else x
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
