#!/usr/bin/env python3
"""Does our island emit any op the .td does NOT declare?

The other direction from island-vs-td.py, and the one that matters more. A MISSING op is scheduled
work — the porting agents add ops on demand and the island's exhaustive matches make an addition a
compile error rather than a silent default. An INVENTED op is a defect: it prints plausibly and the
backend rejects it, or worse accepts it and means something else.

Censuses the `<dialect>.<mnemonic>` spellings our island mentions and reports any the .td does not
declare.
"""
import pathlib
import re

TD = pathlib.Path(
    "/Users/nickm/git/deeptools-src/dataflow-scheduler/external/"
    "dataflow-scheduler-dialects/include/dataflow-scheduler/Dialect"
)
ISLAND = pathlib.Path(
    "/Users/nickm/git/scratchy/.claude/worktrees/bridge2/crates/compiler/deeptools/src/"
    "islands/dataflow_ir/dialects"
)
DIALECTS = {
    "dataflow": (TD / "Dataflow/Dataflow.td", "dataflow.rs"),
    "agen": (TD / "Agen/Agen.td", "agen.rs"),
    "vectorchain": (TD / "VectorChain/VectorChain.td", "vectorchain.rs"),
    "uniform": (TD / "Uniform/Uniform.td", "uniform.rs"),
}

bad = 0
for name, (td, fname) in DIALECTS.items():
    if not td.exists():
        print(f"{name}: .td MISSING")
        continue
    declared = set(re.findall(r'_Op<\s*"([a-z_0-9]+)"', td.read_text(errors="ignore")))
    rs = ISLAND / fname
    body = rs.read_text(errors="ignore") if rs.exists() else ""
    mentioned = set(re.findall(rf"\b{name}\.([a-z_0-9]+)", body))
    invented = sorted(mentioned - declared)
    print(f"\n=== {name}: island mentions {len(mentioned)} spellings, .td declares {len(declared)}")
    if invented:
        bad += len(invented)
        print(f"  ⛔ NOT DECLARED BY THE .td ({len(invented)}): {', '.join(invented)}")
    else:
        print("  ✅ every spelling the island mentions is a declared op")

print(f"\n{bad} spelling(s) the island mentions that the dialect does not declare")
