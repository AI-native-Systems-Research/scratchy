#!/usr/bin/env python3
"""Every NAMED lambda inside ddcv1.cpp, at any nesting depth, with an absolute line.

The brief's three ddc derivations are not all top-level functions: `checkConstraints` is a lambda
at :792 nested in Ddc::exploreAssignDataStages, and `checkConstraintsImpl` is nested inside THAT.
A one-level scan finds only the first.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cppscan

DDCV1 = pathlib.Path("/Users/nickm/git/deeptools-src/ddc/ddcv1.cpp")


def walk(body_text, base_line, enclosing, depth, out):
    if depth > 4:
        return
    for x in cppscan.find_defs(body_text, want_lambdas=True):
        abs_line = base_line + x["head_line"] - 1
        abs_open = base_line + x["open_line"] - 1
        out.append((abs_line, x["name"], x["body_lines"], enclosing, depth))
        walk(x["body"][1:-1], abs_open, x["name"], depth + 1, out)


text = DDCV1.read_text(errors="ignore")
top = cppscan.find_defs(text)
out = []
for d in top:
    walk(d["body"][1:-1], d["open_line"], d["qual"] + d["name"], 1, out)

print(f"{'line':>7}  {'name':30} {'L':>5}  depth  enclosing")
for line, name, bl, enc, depth in sorted(out):
    print(f"  :{line:<5} {name:30} {bl:5}  d{depth}     {enc}")
print(f"\n{len(out)} named lambdas total")

print("\n=== the constraint system (what shape.rs currently invents) ===")
for line, name, bl, enc, depth in sorted(out):
    if "onstraint" in name or "tickSize" in name or "primaryDim" in name or "tageExtent" in name:
        print(f"  :{line:<5} {name:30} {bl:5}L  d{depth}  in {enc}")
