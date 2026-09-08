#!/usr/bin/env python3
"""Consolidate bridge 1's C++ into one translation unit, in dependency order.

Emits:
  cpp/bridge1.cpp   -- every in-scope body VERBATIM, each behind a banner giving its entry number
                       and its ORIGINAL <file>:<line>, ordered level 0 first.
  cpp/prelude.inc   -- plain-C++ stand-ins so the unit needs no MLIR/LLVM/dcc header.
  UNITS.tsv         -- entry, level, LoC, authority file:line, extract line range, rust home, callees.

⛔ The bodies are sliced with cppscan's brace matcher. `verify_extract.py` re-derives every end
INDEPENDENTLY and compares LENGTHS -- it must not be made to share this file's slicing.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cppscan

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
OUT = pathlib.Path("/tmp/bridge1-setup")
CONV = "dsc-based-utils/DSC2ToDataflowIR"

# The conversion: the 9 files carrying the 106 in-scope definitions. The remaining .hpp inline
# bodies are 1-3 line field accessors (a struct field in Rust, not a function) -- excluded, with
# the two substantive ones named in EXCLUDED below.
CONV_FILES = [
    f"{CONV}/DSC2ToDataflowIR.cpp",
    f"{CONV}/DSC2ToDataflowIRUtils.hpp",
    f"{CONV}/DataflowIRConstructionUtils.hpp",
    f"{CONV}/V3/SNDSCLowering.cpp",
    f"{CONV}/V3/SNTransferLowering.cpp",
    f"{CONV}/V3/SNComputeLowering.cpp",
    f"{CONV}/V3/SNControlFlowLowering.cpp",
    f"{CONV}/V3/SNSyncLowering.cpp",
    f"{CONV}/V3/SNStickMaskLowering.cpp",
]

# The DDL compiler's derivations. These are NOT reachable from a call graph rooted at the
# conversion -- in the reference they run before a SuperDsc exists -- so they are named.
# `checkConstraints` is a LAMBDA nested in Ddc::exploreAssignDataStages and lexically CONTAINS
# `checkConstraintsImpl`, so one unit covers both.
DDC_LAMBDAS = [("ddc/ddcv1.cpp", "checkConstraints")]
DDC_TOPLEVEL = [
    ("ddc/ddcv1.cpp", "createDataConnectMetadata"),
    # getCumulativeStickSizes is NOT defined in ddcv1.cpp -- :789/:1723 are CALL SITES. Its
    # definition is DesignSpaceConfig::getCumulativeStickSizes, and it delegates to getStickSizes,
    # which carries the actual per-dim stick arithmetic. Both are needed to replace shape.rs.
    ("dsc/dsc2.cpp", "getCumulativeStickSizes"),
    ("dsc/dsc2.cpp", "getStickSizes"),
]

HOME = {
    "DSC2ToDataflowIR.cpp": "driver.rs",
    "DSC2ToDataflowIRUtils.hpp": "utils.rs",
    "DataflowIRConstructionUtils.hpp": "construction.rs",
    "SNDSCLowering.cpp": "dsc_lowering.rs",
    "SNTransferLowering.cpp": "transfer.rs",
    "SNComputeLowering.cpp": "compute.rs",
    "SNControlFlowLowering.cpp": "control_flow.rs",
    "SNSyncLowering.cpp": "sync.rs",
    "SNStickMaskLowering.cpp": "stick_mask.rs",
    "ddcv1.cpp": "shape_constraints.rs",
    "dsc2.cpp": "shape_constraints.rs",
}

EXCLUDED = [
    ("stringifyComputePrecision", "DSC2ToDataflowIR.hpp:54",
     "ComputeOpType -> string table; in this crate a closed set is an enum and never a string"),
    ("cloneSNDSCLoweringObject", "V3/SNDSCLowering.hpp:71",
     "copies the lowering object; the Rust port threads a value, there is no object to clone"),
    ("getFromLatchMap", "V3/SNDSCLowering.hpp:134",
     "map lookup helper; the operand-reaching mechanism the brief allows dropping"),
    ("<16 field accessors>", "V3/SNDSCLowering.hpp, DSC2ToDataflowIR.hpp, SNComputeLowering.hpp",
     "1-3 line getters/setters over a member -- a struct field in Rust, not a function"),
]


def collect():
    units = []
    for rel in CONV_FILES:
        text = (SRC / rel).read_text(errors="ignore")
        for d in cppscan.find_defs(text):
            d["file"] = rel
            units.append(d)
    for rel, name in DDC_TOPLEVEL:
        text = (SRC / rel).read_text(errors="ignore")
        hits = [d for d in cppscan.find_defs(text) if d["name"] == name]
        if not hits:
            print(f"  !! {name} not found in {rel}", file=sys.stderr)
        for d in hits:
            d["file"] = rel
            units.append(d)
    for rel, name in DDC_LAMBDAS:
        text = (SRC / rel).read_text(errors="ignore")
        for d in cppscan.find_defs(text):
            for x in cppscan.find_defs(d["body"][1:-1], want_lambdas=True):
                if x["name"] != name:
                    continue
                base = d["open_line"]
                x["head_line"] += base - 1
                x["open_line"] += base - 1
                x["close_line"] += base - 1
                x["body_lines"] = x["close_line"] - x["open_line"] + 1
                x["decl_lines"] = x["close_line"] - x["head_line"] + 1
                x["file"] = rel
                x["enclosing"] = d["qual"] + d["name"]
                units.append(x)
    return units


def levels(units):
    """Longest-path level over intra-span call edges, by name."""
    by_name = {}
    for u in units:
        by_name.setdefault(u["name"], []).append(u)
    for u in units:
        called = cppscan.calls_in(u["body"])
        u["callees"] = sorted(c for c in called if c in by_name and c != u["name"])
    lvl = {u["name"]: 0 for u in units}
    for _ in range(len(units) + 2):
        changed = False
        for u in units:
            want = max([lvl[c] + 1 for c in u["callees"]] or [0])
            if want > lvl[u["name"]]:
                lvl[u["name"]] = want
                changed = True
        if not changed:
            break
    for u in units:
        u["level"] = lvl[u["name"]]
    return units


def main():
    units = levels(collect())
    units.sort(key=lambda u: (u["level"], u["file"], u["head_line"]))
    for n, u in enumerate(units, 1):
        u["entry"] = n
        u["sym"] = f"e{n:03d}_{u['name']}"

    lines = []
    lines.append("// bridge1.cpp -- IBM Spyre deeptools bridge 1: SuperDSC -> DataflowIR.")
    lines.append("//")
    lines.append("// Every body below is VERBATIM from the authority tree")
    lines.append("//   /Users/nickm/git/deeptools-src   (repo_info.txt: deeptools|master|a0d29abbed...)")
    lines.append("// and each banner gives that body's ORIGINAL <file>:<line>. Port from the authority")
    lines.append("// file at the cited line; this unit tells you WHICH function and in WHAT ORDER.")
    lines.append("//")
    lines.append(f"// {len(units)} units. Member definitions are rewritten as free functions")
    lines.append("// (Class::foo -> eNNN_foo) so the unit needs no class declarations; bodies are untouched.")
    lines.append('#include "prelude.inc"')
    lines.append("")
    cur = None
    index = {}
    for u in units:
        if u["level"] != cur:
            cur = u["level"]
            lines.append("")
            lines.append("// " + "=" * 96)
            lines.append(f"// LEVEL {cur}")
            lines.append("// " + "=" * 96)
            lines.append("")
        rel = u["file"]
        banner = (f"// ---- {u['entry']}/{len(units)}  {u['name']}  --  "
                  f"{rel}:{u['head_line']}  ({u['decl_lines']}L)")
        if u.get("enclosing"):
            banner += f"   [lambda nested in {u['enclosing']}]"
        lines.append(banner)
        head = u["head"]
        # rewrite the declarator: Class::name -> eNNN_name (head only; the body is untouched)
        if u["qual"]:
            head = head.replace(u["qual"] + u["name"], u["sym"], 1)
        else:
            head = re.sub(r"\b" + re.escape(u["name"]) + r"\b", u["sym"], head, count=1)
        if u.get("enclosing"):
            head = f"static auto {u['sym']} = " + head[head.index("["):] if "[" in head else head
        # VERBATIM: the head, then the body from its `{` onward, untouched. Anything that shares
        # the `{`'s line -- a trailing comment, or real code -- must survive. An earlier version
        # rebuilt that line as `head + " {"` and silently dropped it; verify_extract.py caught it.
        start = len(lines) + 1
        lines.extend((head.rstrip() + " " + u["body"]).split("\n"))
        if u.get("enclosing"):
            lines[-1] = lines[-1].rstrip() + ";"
        end = len(lines)
        index[u["sym"]] = (start, end)
        u["extract_start"], u["extract_end"] = start, end
        lines.append("")

    (OUT / "cpp").mkdir(parents=True, exist_ok=True)
    (OUT / "cpp/bridge1.cpp").write_text("\n".join(lines) + "\n")

    tsv = ["\t".join(["entry", "level", "loc", "authority", "extract_lines", "rust_home", "callees"])]
    for u in units:
        home = HOME.get(pathlib.Path(u["file"]).name, "unsorted.rs")
        tsv.append("\t".join([
            u["sym"], str(u["level"]), str(u["decl_lines"]),
            f"{u['file']}:{u['head_line']}",
            f"{u['extract_start']}-{u['extract_end']}",
            home,
            ",".join(u["callees"]),
        ]))
    (OUT / "UNITS.tsv").write_text("\n".join(tsv) + "\n")

    nl = {}
    for u in units:
        nl.setdefault(u["level"], []).append(u)
    print(f"{len(units)} units, {sum(u['decl_lines'] for u in units)} declaration lines")
    print(f"bridge1.cpp: {len(lines)} lines")
    for k in sorted(nl):
        print(f"  level {k}: {len(nl[k])} units, {sum(u['decl_lines'] for u in nl[k])}L")
    homes = {}
    for u in units:
        homes.setdefault(HOME.get(pathlib.Path(u["file"]).name, "unsorted.rs"), 0)
        homes[HOME.get(pathlib.Path(u["file"]).name, "unsorted.rs")] += 1
    print("  rust homes: " + ", ".join(f"{k}={v}" for k, v in sorted(homes.items())))
    return units


if __name__ == "__main__":
    main()
