#!/usr/bin/env python3
"""Consolidate the campaign's C++ into ONE self-contained TU, bodies VERBATIM.

Writes campaign/cpp/sentient.cpp (banner + rewritten signature + verbatim body, in dependency
order), campaign/cpp/prelude.inc (plain-C++ stand-ins, no MLIR/LLVM/dcc header anywhere) and
campaign/UNITS.tsv.

⛔ The signature is rewritten to the unit symbol `eNNN_name` so that symbol == unit name for EVERY
entry. Bridge 2 left 116 of its 384 entries carrying their bare C++ name, which is why its own
crates.json had to warn "ALWAYS resolve a unit through UNITS.tsv". The BODY - `{` through its
matching `}` - is copied byte for byte and is never reflowed, re-indented or brace-balanced.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import re
from collections import Counter, defaultdict

import cppscan

SENT = _paths.SENT
REL = _paths.REL
WORK = _paths.WORK
CAMP = _paths.CAMP
REV = _paths.REV
AUTH_ROOT = _paths.AUTH_ROOT

RUST_ROOT = _paths.OUTDIR


def snake(n):
    n = n.replace("CFG", "Cfg").replace("XRF", "Xrf").replace("NOP", "Nop")
    n = n.replace("SSA", "Ssa").replace("RE", "Re").replace("HBM", "Hbm")
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", n)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"__+", "_", s).lower().strip("_")


def sym_of(entry, name):
    return "e%03d_%s" % (entry, name.replace("~", "dtor_"))


def home_of(u):
    """`<module>/mod.rs`, or `<module>/<helper class>.rs` for a helper class in the same file.

    Real nested submodules named after the passes - not flat prefixed files.
    """
    stem = os.path.splitext(u["file"])[0]
    mod = snake(stem)
    cls = u["qual"].rstrip(":").split("::")[-1] if u["qual"] else ""
    if not cls:
        return f"{RUST_ROOT}/{mod}/mod.rs", mod, None
    # the pass's own methods live in the pass module's root; a helper class gets a real submodule
    if cls.endswith("Pass") or snake(cls) == mod:
        return f"{RUST_ROOT}/{mod}/mod.rs", mod, None
    sub = snake(cls)
    return f"{RUST_ROOT}/{mod}/{sub}.rs", mod, sub


def rewrite_head(u, sym):
    """Return the extract's signature line(s): the original head with the qualified name replaced
    by `sym` and MLIR/C++ member specifiers that have no meaning at file scope removed."""
    head = u["head"]
    qual = u["qual"]
    name = u["name"]
    pat = (re.escape(qual) + r"\s*" if qual else "") + re.escape(name) + r"(\s*\()"
    m = None
    for m in re.finditer(pat, head):
        pass
    if m is None:
        raise SystemExit(f"cannot locate signature name for {u['unit']} in head: {head[:160]!r}")
    head = head[:m.start()] + sym + m.group(1) + head[m.end():]
    head = re.sub(r"^\s*(virtual|static|inline|explicit)\s+", "", head)
    head = re.sub(r"\b(override|final)\b", "", head)
    return head.rstrip()


def main():
    units = json.load(open(WORK + "/ordered.json"))
    os.makedirs(CAMP + "/cpp", exist_ok=True)
    text_of = {}
    for u in units:
        p = SENT + "/" + u["file"]
        if p not in text_of:
            text_of[p] = open(p, encoding="utf-8", errors="replace").read()

    n = len(units)
    out = []
    out.append("// " + "=" * 96)
    out.append("// SENTIENT IN-PLACE PASS CAMPAIGN - consolidated translation unit")
    out.append("//")
    out.append(f"// {n} function bodies from {REL}/*.{{cpp,hpp,h}}, extracted VERBATIM and emitted")
    out.append("// in dependency order over the SCC condensation (level 0 calls nothing else in the")
    out.append("// span; level N only calls levels below N, or its own strongly connected component).")
    out.append("//")
    out.append(f"// AUTHORITY, and the ONLY thing to port from: {AUTH_ROOT}")
    out.append(f"//   repo_info.txt: deeptools|master|{REV}")
    out.append("// ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. Do not use it. The pod is not")
    out.append("//    the authority for this campaign either - this local tree is the cited revision.")
    out.append("//")
    out.append("// Each entry's banner gives its unit symbol, level, and original file:line. The")
    out.append("// signature is rewritten to the unit symbol so symbol == unit name; the body between")
    out.append("// `{` and its matching `}` is byte-for-byte the authority's.")
    out.append("// " + "=" * 96)
    out.append("")
    out.append('#include "prelude.inc"')
    out.append("")

    for u in units:
        sym = sym_of(u["entry"], u["name"])
        u["unit"] = sym
        head = rewrite_head(u, sym)
        body = text_of[SENT + "/" + u["file"]][u["open_off"]:u["close_off"] + 1]
        home, mod, sub = home_of(u)
        u["rust_home"], u["module"], u["submodule"] = home, mod, sub
        out.append("// " + "-" * 96)
        out.append(f"// entry {u['entry']:03d}/{n}   level {u['level']}   scc {u['scc']}   "
                   f"{u['body_lines']} body lines")
        out.append(f"// unit: {sym}")
        out.append(f"// authority: {REL}/{u['file']}:{u['head_line']}")
        orig = re.sub(r"\s+", " ", u["head"]).strip()
        out.append(f"// original: {orig[:400]}")
        out.append(f"// rust home: {home}")
        out.append("// " + "-" * 96)
        start = len(out) + 1  # 1-based line of the first signature line
        out.extend(head.split("\n"))
        out.extend(body.split("\n"))
        u["extract_lines"] = f"{start}-{len(out)}"
        out.append("")

    open(CAMP + "/cpp/sentient.cpp", "w").write("\n".join(out) + "\n")

    # ---- prelude: a NAME INVENTORY of every external callee/type the bodies mention -----------
    defined = {u["name"] for u in units} | {sym_of(u["entry"], u["name"]) for u in units}
    ext = Counter()
    for u in units:
        body = text_of[SENT + "/" + u["file"]][u["open_off"]:u["close_off"] + 1]
        for c in cppscan.calls_in(body):
            if c not in defined:
                ext[c] += 1
    pre = []
    pre.append("// prelude.inc - plain-C++ stand-ins for the names the bodies call.")
    pre.append("//")
    pre.append("// NO MLIR, LLVM OR dcc HEADER IS INCLUDED ANYWHERE IN THIS CAMPAIGN. This file is a")
    pre.append("// NAME INVENTORY, not a compilable prelude: the stand-ins carry names and arity")
    pre.append("// only, so the consolidated TU stands alone textually. It is declared out_of_scope")
    pre.append("// in wavefront-config.json and NOTHING IN IT IS TO BE PORTED - the behaviour to")
    pre.append("// port lives entirely in the bodies in sentient.cpp.")
    pre.append("//")
    pre.append(f"// {len(ext)} distinct external names, {sum(ext.values())} call sites.")
    pre.append("")
    pre.append("struct Any {")
    pre.append("  template <class... A> Any operator()(A &&...) const;")
    pre.append("  template <class T> operator T() const;")
    pre.append("  Any operator*() const; Any operator->() const; Any operator[](Any) const;")
    pre.append("};")
    pre.append("")
    for name, cnt in sorted(ext.items(), key=lambda kv: (-kv[1], kv[0])):
        pre.append(f"extern Any {name};  // {cnt} call site(s)")
    open(CAMP + "/cpp/prelude.inc", "w").write("\n".join(pre) + "\n")

    # ---- UNITS.tsv --------------------------------------------------------------------------
    with open(CAMP + "/UNITS.tsv", "w") as fh:
        fh.write("unit\tentry\tlevel\tscc\tloc\tauthority (under %s)\textract_lines "
                 "(crustify-senpass/cpp/sentient.cpp)\trust_home\tcalls\n" % AUTH_ROOT)
        for u in units:
            fh.write("\t".join([
                u["unit"], f"{u['entry']:03d}/{n}", str(u["level"]), str(u["scc"]),
                str(u["body_lines"]), f"{REL}/{u['file']}:{u['head_line']}",
                u["extract_lines"], u["rust_home"],
                ",".join(u["call_units"]) if u["call_units"] else "-",
            ]) + "\n")

    json.dump(units, open(WORK + "/units.json", "w"), indent=0)

    homes = Counter(u["rust_home"] for u in units)
    mods = Counter(u["module"] for u in units)
    print(f"extract lines      : {len(out)}")
    print(f"units              : {n}")
    print(f"prelude names      : {len(ext)}")
    print(f"rust homes         : {len(homes)} files in {len(mods)} pass modules")
    print("largest homes:")
    for k, v in homes.most_common(8):
        print(f"   {v:4d}  {k}")


if __name__ == "__main__":
    main()
