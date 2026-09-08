#!/usr/bin/env python3
"""Is our DataflowIR island ACCURATE against the authoritative .td files?

Volume is not accuracy. The island grew organically — porting agents added ops as their functions
needed them — so it may lack ops, and worse may carry ops the dialect never declares.

⛔ AN EARLIER VERSION OF THIS SCRIPT WAS STRUCTURALLY INCAPABLE OF ANSWERING. It looked for the
mnemonic as a literal next to the dialect name (`"vectorchain.binary"`), but the printers write
`"{} = vectorchain.{} ..."` with the mnemonic coming from `kind.spelling()` — a runtime value. It
therefore reported vectorchain as emitting ZERO of 23 ops, in a file that plainly has fifteen. The fix
is to look for each declared mnemonic as a bare quoted string ANYWHERE in the dialect's module, which
catches both the inline literals and the `spelling()` tables.
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

total_declared = total_ours = 0
for name, (td, fname) in DIALECTS.items():
    if not td.exists():
        print(f"{name}: .td MISSING at {td}")
        continue
    declared = set(re.findall(r'_Op<\s*"([a-z_0-9]+)"', td.read_text(errors="ignore")))
    rs = ISLAND / fname
    body = rs.read_text(errors="ignore") if rs.exists() else ""
    # ⛔ NO ANCHORS. The mnemonic reaches the text three different ways — inline in a format string
    # (`"{} = dataflow.get_unit {{name = ..."`), from a `spelling()` table (`Self::Binary => "binary"`),
    # or via `kind.spelling()` — and two earlier versions of this check each missed two of the three.
    # A plain substring search for `<dialect>.<mnemonic>` catches all of them, because every op is
    # named that way in its own doc comment even when the printer composes it at run time.
    ours = {m for m in declared if f"{name}.{m}" in body}
    missing = sorted(declared - ours)
    total_declared += len(declared)
    total_ours += len(ours)
    print(f"\n=== {name}: .td declares {len(declared)}, ours carries {len(ours)}")
    if missing:
        print(f"  MISSING ({len(missing)}): {', '.join(missing)}")
    else:
        print("  ✅ every declared op is present")

print(f"\nTOTAL: {total_ours} of {total_declared} declared ops present in the island")
