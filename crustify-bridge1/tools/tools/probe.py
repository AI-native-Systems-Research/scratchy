#!/usr/bin/env python3
"""Cross-checks: the brief's body-line convention, the full ddcv1 list, and the nested lambdas."""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cppscan

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
CONV = SRC / "dsc-based-utils/DSC2ToDataflowIR"
DDCV1 = SRC / "ddc/ddcv1.cpp"

IN_SCOPE = [
    "DSC2ToDataflowIR.cpp", "DSC2ToDataflowIRUtils.hpp", "DataflowIRConstructionUtils.hpp",
    "V3/SNComputeLowering.cpp", "V3/SNControlFlowLowering.cpp", "V3/SNDSCLowering.cpp",
    "V3/SNStickMaskLowering.cpp", "V3/SNSyncLowering.cpp", "V3/SNTransferLowering.cpp",
]

print("=== which line convention gives the brief's 7395? ===")
tb = td = 0
for rel in IN_SCOPE:
    defs = cppscan.find_defs((CONV / rel).read_text(errors="ignore"))
    tb += sum(d["body_lines"] for d in defs)
    td += sum(d["decl_lines"] for d in defs)
print(f"  sum(open_line..close_line) = {tb}")
print(f"  sum(head_line..close_line) = {td}   <- brief says 7395")

print("\n=== header inline defs the brief's 106 excludes (accessor check) ===")
for rel in ["DSC2ToDataflowIR.hpp", "V3/SNComputeLowering.hpp", "V3/SNDSCLowering.hpp"]:
    for d in cppscan.find_defs((CONV / rel).read_text(errors="ignore")):
        one = re.sub(r"\s+", " ", d["body"]).strip()
        print(f"  {rel}:{d['head_line']:4} {d['qual']+d['name']:34} {d['body_lines']}L  {one[:64]}")

print("\n=== ddc/ddcv1.cpp: all top-level defs ===")
dd = cppscan.find_defs(DDCV1.read_text(errors="ignore"))
for d in sorted(dd, key=lambda x: x["head_line"]):
    print(f"  :{d['head_line']:<5} {d['qual']+d['name']:44} {d['body_lines']:5}L  "
          f"(ends :{d['close_line']})")
print(f"  {len(dd)} defs, {sum(d['body_lines'] for d in dd)} body lines")

print("\n=== the named derivations: nested lambdas inside those bodies ===")
text = DDCV1.read_text(errors="ignore")
for d in dd:
    inner = cppscan.find_defs(d["body"][1:-1], want_lambdas=True)
    for x in inner:
        if x["name"] in ("checkConstraints", "checkConstraintsImpl", "getCumulativeStickSizes",
                         "primaryDimToVal_st", "getStageExtent"):
            abs_line = d["open_line"] + x["head_line"] - 1
            print(f"  {x['name']:26} ddc/ddcv1.cpp:{abs_line:<5} {x['body_lines']:4}L   "
                  f"nested in {d['qual']+d['name']} (:{d['head_line']})")

print("\n=== grep: how are they written? ===")
lines = text.splitlines()
for pat in ("checkConstraints", "getCumulativeStickSizes"):
    for i, l in enumerate(lines, 1):
        if pat in l and ("=" in l or "auto" in l or "(" in l):
            print(f"  :{i:<5} {l.strip()[:110]}")
