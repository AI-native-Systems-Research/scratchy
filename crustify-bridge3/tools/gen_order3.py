#!/usr/bin/env python3
"""Emit docs/bridge3-porting-order.md -- the 130 units in dependency order with a PORT and an AUDIT
checkbox each, plus the 36 exclusions with a reason.

⛔⛔ THE METHODOLOGY THIS SERVES: PORT ONE fn -> AUDIT it line by line -> tick BOTH. Never port new
work with an audit outstanding. A predicate is not a port.
"""
import collections
import pathlib

OUT = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3")
rows = [l.split("\t") for l in (OUT / "UNITS.tsv").read_text().splitlines()[1:]]
exl = [l.split("\t") for l in (OUT / "EXCLUSIONS.tsv").read_text().splitlines()[1:]]

by_level = collections.defaultdict(list)
for unit, entry, level, loc, auth, ex, home, calls in rows:
    by_level[int(level)].append((unit, entry, int(loc), auth, home, calls))

L = [
    "# Bridge 3 porting order — `SentientIR -> ProgIR`, dcc pass D76",
    "",
    "130 units from `dcc/src/Conversion/SentientToProgIR/`, 6,879 declaration lines, in dependency",
    "order. Authority: `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.",
    "",
    "## ⛔⛔ The methodology",
    "",
    "**PORT ONE fn → AUDIT it line by line against the citation → tick BOTH.** Never port new work",
    "with an audit outstanding. A predicate is not a port: the op a function emits IS the function.",
    "If the target island cannot express something, ADD IT TO THE ISLAND — deciding a function is",
    "unnecessary is not the porter's call.",
    "",
    "**PORT** = the whole function including its emission, anchored `/// Replaces: eNNN_name`.",
    "**AUDIT** = a different agent instance read the port beside the authority at the cited line and",
    "confirmed the branch order, the early returns, and every attribute name and value.",
    "",
    "## ⛔ Not in this span",
    "",
    "`dcc/src/Transform/Sentient/` — 32,766 lines of D29-D75 passes that rewrite SentientIR *in",
    "place* — is a separate, larger body of work. A ported function that depends on state those",
    "passes establish (register assignments, pinned addresses, rerolled loops) is ported **as the",
    "reference writes it**, with the dependency recorded in the commit message. Do not port the pass,",
    "and do not invent the state.",
    "",
    "## ⛔ One mutually recursive component",
    "",
    "`LowerUniformOperations` and `GenerateProgIR` call each other, so they share level 5 and cannot",
    "be split by a wave barrier. Neither may be ported as if the other were a leaf.",
    "",
]

for lv in sorted(by_level):
    us = by_level[lv]
    L += [f"## Level {lv} — {len(us)} units, {sum(u[2] for u in us)} declaration lines", "",
          "| | PORT | AUDIT | unit | LoC | authority | Rust home |",
          "|---|---|---|---|---|---|---|"]
    for unit, entry, loc, auth, home, calls in us:
        short = auth.replace("dcc/src/Conversion/SentientToProgIR/", "")
        h = home.replace("src/bridges/sentient_to_progir/", "")
        L.append(f"| {entry} | [ ] | [ ] | `{unit}` | {loc} | `{short}` | `{h}` |")
    L.append("")

L += ["## Excluded — 36 definitions, with a reason each", "",
      "The criterion is what a Rust struct field or a `const` already is, NOT what is hard.", "",
      "| definition | authority | LoC | reason |", "|---|---|---|---|"]
for name, auth, loc, why in exl:
    short = auth.replace("dcc/src/Conversion/SentientToProgIR/", "")
    L.append(f"| `{name}` | `{short}` | {loc} | {why} |")
L.append("")
L += ["## Kept despite living in a header", "",
      "`getUnitRegionIndex` (`UniformInstrAndBlock.hpp:208`, 6L) is scheduled: its body is a real",
      "decision rule — a REGULAR block always answers 0, a missing entry answers -1 — and a `-1`",
      "sentinel is an `Option` in this crate, never a negative number.", ""]

(OUT / "bridge3-porting-order.md").write_text("\n".join(L))
print(f"bridge3-porting-order.md: {len(rows)} units, {len(exl)} exclusions, "
      f"{len(L)} lines")
