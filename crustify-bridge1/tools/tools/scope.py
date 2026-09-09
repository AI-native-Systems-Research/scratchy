#!/usr/bin/env python3
"""Bridge-1 scope: SuperDSC -> DataflowIR.

Enumerates every function definition by BRACE MATCHING (a signature regex undercounted bridge 2
by 72%) in the two in-scope components, and reports per-file counts against the campaign brief's
measured figures so a disagreement is visible rather than silently adopted.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cppscan

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
CONV = SRC / "dsc-based-utils/DSC2ToDataflowIR"
DDCV1 = SRC / "ddc/ddcv1.cpp"

# what the campaign brief states, measured by the earlier naive-brace scan
BRIEF = {
    "V3/SNTransferLowering.cpp": (29, 2667),
    "V3/SNComputeLowering.cpp": (18, 1549),
    "V3/SNControlFlowLowering.cpp": (14, 1090),
    "DSC2ToDataflowIRUtils.hpp": (11, 647),
    "DSC2ToDataflowIR.cpp": (11, 521),
    "V3/SNDSCLowering.cpp": (12, 466),
    "V3/SNSyncLowering.cpp": (4, 250),
    "DataflowIRConstructionUtils.hpp": (6, 147),
    "V3/SNStickMaskLowering.cpp": (1, 58),
}


def files():
    out = sorted(CONV.glob("*.cpp")) + sorted(CONV.glob("*.hpp"))
    out += sorted((CONV / "V3").glob("*.cpp")) + sorted((CONV / "V3").glob("*.hpp"))
    return out


def main():
    print("=== dsc-based-utils/DSC2ToDataflowIR ===")
    print(f"{'file':44} {'defs':>5} {'bodyL':>7}   brief(defs,bodyL)")
    total_defs = total_lines = 0
    per_file = {}
    for f in files():
        defs = cppscan.find_defs(f.read_text(errors="ignore"))
        bl = sum(d["body_lines"] for d in defs)
        rel = str(f.relative_to(CONV))
        per_file[rel] = defs
        total_defs += len(defs)
        total_lines += bl
        b = BRIEF.get(rel)
        flag = ""
        if b:
            flag = f"   {b}" + ("" if (len(defs), bl) == b else "   <-- DIFFERS")
        elif defs:
            flag = "   (not in brief)"
        print(f"{rel:44} {len(defs):5} {bl:7}{flag}")
    print(f"\nTOTAL: {total_defs} definitions, {total_lines} body lines")
    print(f"brief says: 106 definitions, 7395 body lines, 17 files")

    print("\n=== ddc/ddcv1.cpp (brace-matched, full enumeration) ===")
    dd = cppscan.find_defs(DDCV1.read_text(errors="ignore"))
    print(f"{len(dd)} definitions, {sum(d['body_lines'] for d in dd)} body lines "
          f"of {len(DDCV1.read_text(errors='ignore').splitlines())} file lines")
    for d in sorted(dd, key=lambda x: -x["body_lines"]):
        print(f"  {d['qual']+d['name']:52} {d['body_lines']:5}L  ddc/ddcv1.cpp:{d['head_line']}")

    print("\n=== the three named derivations ===")
    for want in ("checkConstraints", "checkConstraintsImpl", "getCumulativeStickSizes",
                 "createDataConnectMetadata"):
        hits = [d for d in dd if d["name"] == want]
        if hits:
            for h in hits:
                print(f"  FOUND {want:28} ddc/ddcv1.cpp:{h['head_line']}  {h['body_lines']}L")
        else:
            print(f"  MISSING as a top-level def: {want}  (may be a lambda)")


if __name__ == "__main__":
    main()
