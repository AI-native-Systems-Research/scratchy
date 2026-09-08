#!/usr/bin/env python3
"""Generate src/bridges/superdsc_to_dataflow_ir/ -- the .rs homes with their crustify:todo anchors.

The AGENT-BRIEF promises each agent that its `.rs` home already exists, carries the citation table
for its units in the module doc, and holds one `// crustify:todo: eNNN_name` per scheduled unit. This
writes exactly that, so `grep -c 'crustify:todo:'` is the campaign's open-work count from minute one.
"""
import collections
import pathlib

OUT = pathlib.Path("/tmp/bridge1-setup/rust/superdsc_to_dataflow_ir")

BLURB = {
    "transfer.rs": (
        "THE TRANSFER STATEMENTS — load, store and send, and how each is walked over the AGEN time axis.",
        "Carries the 2B/16B store shuffles, the constant bit streams, the burst and interleave settings.",
    ),
    "compute.rs": (
        "THE VECTOR CHAINS — mac, binary and unary computes, and the precision they run at.",
        "⭐ ROPE IS TWO FMA STAGES, NOT A MULTIPLY: rope.ddl:26-27 declares rope64p1/rope64p2, two chained",
        "FMA16 stages through an intermediate, and the m2 transfer's rotate_num_elements=32 IS the pair swap.",
    ),
    "control_flow.rs": (
        "THE LOOP NEST AND THE CONDITIONALS — DSC loops, blocks and conds becoming the scf/affine nest.",
    ),
    "dsc_lowering.rs": (
        "THE PER-COMPONENT LOWERING — unit, corelet and core identity, and the component handler.",
        "⭐ ONE ProgramUnitOp SPANS THE SET: handles = cores x corelets x num_folds. The component-to-handler",
        "map is CLEARED per unit; the unit-to-value map is module-wide.",
    ),
    "driver.rs": (
        "THE CONVERSION DRIVER — runTranslator, convertV3, convertV4, and the module scaffolding they build.",
    ),
    "utils.rs": (
        "THE SHARED SHAPE AND FOLD HELPERS every lowering below stands on.",
    ),
    "construction.rs": (
        "DATAFLOWIR CONSTRUCTION — building the island's types and attributes.",
    ),
    "sync.rs": (
        "THE SYNC STATEMENTS.",
        "⛔ THE ORDER IS dxp's AND CITED: reordering syncs times the card out (CB state=TimedOut, sync rc=-1).",
    ),
    "stick_mask.rs": (
        "THE STICK MASK — the SAMV set-transfer-mask-state op.",
    ),
    "shape_constraints.rs": (
        "THE DDL COMPILER'S DERIVATIONS — and the reason this campaign exists.",
        "⛔ scratchy currently INVENTS these in islands/.../shape.rs. That invention is the defect being removed.",
        "⛔ ABSOLUTE STAGE EXTENTS ARE A CONSTRAINT SYSTEM, NOT A FORMULA: ratios against a tensor, multiples,",
        "stage-relative bounds, SETs, bare bounds, and a different rule per `ddl.if` arm. TRIP COUNTS need none",
        "of it — a loop's own two stages are dimensionless.",
        "⭐ THE PORT IDENTITY IS `data_connect=`, NOT THE SSA NAME (createDataConnectMetadata).",
    ),
}


def main():
    rows = [l.split("\t") for l in
            pathlib.Path("/tmp/bridge1-setup/UNITS.tsv").read_text().splitlines()[1:]]
    by_home = collections.defaultdict(list)
    for entry, level, loc, auth, exl, home, callees in rows:
        by_home[home].append((entry, int(level), int(loc), auth))

    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(by_home, key=lambda h: -len(by_home[h]))

    mod = [
        "//! BRIDGE 1 — scratchy's SuperDSC meeting DataflowIR's vocabulary.",
        "//!",
        "//! ```text",
        "//! SuperDSC (SdscOp / SdscFolds)  ──here──►  islands::dataflow_ir::Run<A>  ──►  dbo-opt",
        "//! ```",
        "//!",
        "//! ⭐ THE DDL TEMPLATE IS THE SCHEDULE (units, transfers, loop nest, computes) — the same for",
        "//! every op of its op-func, and it knows no extents. THE NODE IS THE SHAPE. Two sides, both",
        "//! needed; a bridge that derives one from the other has invented it.",
        "//!",
        "//! ⛔ THIS REPLACES [`super::subtile_to_dataflow_ir`], which stays in the tree until the",
        "//! acceptance build is green on BOTH granite-3.1-2b and -8b. Switching the call site is a",
        "//! separate, deliberate step — do not edit that module from here.",
        "//!",
        "//! Ported from the authority tree `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`;",
        f"//! {len(rows)} functions, each carrying a `/// Replaces: eNNN_name` anchor citing its original.",
        "",
    ]
    for home in order:
        stem = home[:-3]
        mod.append(f"/// {BLURB.get(home, ('',))[0]}")
        mod.append(f"pub mod {stem};")
        mod.append("")
    (OUT / "mod.rs").write_text("\n".join(mod))

    for home, units in by_home.items():
        units.sort(key=lambda u: (u[1], u[0]))
        lines = []
        for l in BLURB.get(home, ("TODO: describe this module.",)):
            lines.append(f"//! {l}")
        lines.append("//!")
        lines.append(f"//! {len(units)} units. Every citation resolves against the authority tree")
        lines.append("//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.")
        lines.append("//!")
        lines.append("//! | unit | level | LoC | authority |")
        lines.append("//! |---|---|---|---|")
        for entry, level, loc, auth in units:
            lines.append(f"//! | `{entry}` | {level} | {loc} | `{auth}` |")
        lines.append("")
        lines.append("// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function")
        lines.append("// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.")
        lines.append("")
        for entry, level, loc, auth in units:
            lines.append(f"// crustify:todo: {entry}")
        lines.append("")
        (OUT / home).write_text("\n".join(lines))
        print(f"  {home:24} {len(units)} anchors")

    print(f"\n  mod.rs declares {len(order)} submodules")


if __name__ == "__main__":
    main()
