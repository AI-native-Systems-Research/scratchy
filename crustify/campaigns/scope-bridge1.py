#!/usr/bin/env python3
"""Scope of bridge 1: SuperDSC -> DataflowIR.

Extracts every function definition in dsc-based-utils/DSC2ToDataflowIR by brace matching (NOT by a
signature regex — that undercounted bridge 2 by 72%), then reports which symbols those bodies call
that are DEFINED in the DDL compiler (ddc/) rather than in the conversion itself. Those calls are the
part of `ddc` the port actually reaches, as opposed to all 18,807 lines of it.
"""
import pathlib
import re
import collections

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
CONV = SRC / "dsc-based-utils/DSC2ToDataflowIR"


def defs_in(path):
    """Function definitions, by brace matching from the opening paren of the signature."""
    text = path.read_text(errors="ignore")
    lines = text.splitlines()
    out = []
    # a plausible definition head: something(...)  followed eventually by '{' at depth 0
    pat = re.compile(r"^[A-Za-z_][\w:<>,&*\s~]*\b([A-Za-z_]\w*)\s*\(")
    i = 0
    while i < len(lines):
        m = pat.match(lines[i])
        if not m or lines[i].lstrip().startswith(("//", "#", "return", "if", "for", "while")):
            i += 1
            continue
        # walk forward to the matching close paren, then require '{'
        depth = 0
        j = i
        seen = False
        while j < len(lines) and j < i + 40:
            depth += lines[j].count("(") - lines[j].count(")")
            if depth == 0 and "(" in "".join(lines[i : j + 1]):
                seen = True
                break
            j += 1
        if not seen:
            i += 1
            continue
        tail = lines[j][lines[j].rfind(")") :] if ")" in lines[j] else ""
        if "{" not in tail and (j + 1 >= len(lines) or "{" not in lines[j + 1][:4]):
            i += 1
            continue
        # brace match the body
        k = j
        bd = 0
        started = False
        while k < len(lines):
            bd += lines[k].count("{") - lines[k].count("}")
            if "{" in lines[k]:
                started = True
            if started and bd <= 0:
                break
            k += 1
        out.append((m.group(1), i + 1, k - i + 1))
        i = k + 1
    return out


conv_files = sorted(list(CONV.glob("*.cpp")) + list(CONV.glob("*.hpp")) + list(CONV.glob("V3/*.cpp")) + list(CONV.glob("V3/*.hpp")))
alldefs = {}
total = 0
for f in conv_files:
    ds = defs_in(f)
    alldefs[f] = ds
    total += sum(d[2] for d in ds)
    print(f"  {f.relative_to(CONV)!s:44} {len(ds):4} defs  {sum(d[2] for d in ds):6} body lines")
n = sum(len(v) for v in alldefs.values())
print(f"\nDSC2ToDataflowIR: {n} function definitions, {total} body lines")

# which ddc symbols do those bodies call?
own = {d[0] for v in alldefs.values() for d in v}
ddc_defs = collections.defaultdict(list)
for f in list((SRC / "ddc").rglob("*.cpp")) + list((SRC / "ddc").rglob("*.hpp")):
    for name, ln, nl in defs_in(f):
        ddc_defs[name].append((f, ln, nl))

called = collections.Counter()
for f in conv_files:
    body = f.read_text(errors="ignore")
    for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", body):
        if name in ddc_defs and name not in own:
            called[name] += 1

reach = sum(ddc_defs[nm][0][2] for nm in called)
print(f"\nddc functions the conversion reaches: {len(called)}, {reach} body lines")
for nm, c in called.most_common(20):
    f, ln, nl = ddc_defs[nm][0]
    print(f"  {nm:46} {nl:5}L  {f.relative_to(SRC)}:{ln}  ({c} call sites)")
print(f"\nBRIDGE 1 TOTAL: {n + len(called)} functions, {total + reach} body lines")
