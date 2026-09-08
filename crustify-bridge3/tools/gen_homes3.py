#!/usr/bin/env python3
"""Generate src/bridges/sentient_to_progir/ -- the .rs homes with their crustify:todo anchors.

The AGENT-BRIEF promises each agent that its `.rs` home already exists, carries the citation table
for its units in the module doc, and holds one `// crustify:todo: eNNN_name` per scheduled unit.
This writes exactly that, so `grep -c 'crustify:todo:'` is the campaign's open-work count from
minute one.

⭐ REAL NESTED SUBMODULES: `construct/`, `lower/`, `uniform/` each get a `mod.rs`.
"""
import collections
import pathlib

OUT = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crates/compiler/deeptools/src/bridges/sentient_to_progir")
UNITS = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3/UNITS.tsv")

# One blurb per home. ⛔ 8 DOC LINES PER FUNCTION is the agents' cap; these module headers are the
# campaign's, written once, and carry the traps a porter would otherwise rediscover.
BLURB = {
    "construct/scalar.rs": [
        "THE SCALAR INSTRUCTIONS — the jumps, the compares, the adds and subs, and the sync.",
        "",
        "⛔ THE LDST FAMILY IS ONE OPCODE AND THE COMPONENT DECIDES ITS ROLE (`isa.cpp:952`): the",
        "field names `ldtype`/`consumertag` are picked BY COMPONENT, not by opcode. A misport here",
        "writes the wrong field name into every load/store.",
        "⛔ THE LBR HOLDS AN INDEX, NOT AN ADDRESS — measured on an EBR-matched op-11 diff.",
    ],
    "construct/compute.rs": [
        "THE COMPUTE INSTRUCTIONS — FMA, binary, unary and ternary, and the operand plumbing that",
        "feeds them.",
        "",
        "⛔ THE FMA DESTINATION IS `ResultForwarding` ON THE OP, not a register operand.",
        "⛔ `ConstructBinaryInstr` IS 485 LINES — the single largest unit in the span. Its branch",
        "order and early returns ARE the port; a summary of its decision rule is not.",
    ],
    "construct/transfer.rs": [
        "THE TRANSFER INSTRUCTIONS — the L3 and non-L3 loads and stores, the burst normalisation and",
        "the load/store fusion.",
        "",
        "⛔ TWO BURST DERIVATIONS EXIST IN OUR TREE AND dxp HAS ONE. Port the reference's.",
        "⛔ BOTH STORE-SIDE SYNCS WERE FOUND INVERTED ON AN EBR-MATCHED OP, and that writes zeros.",
    ],
    "construct/mask_and_splat.rs": [
        "THE MASK, SPLAT AND SAMV INSTRUCTIONS — set-dest-mask, set-dest, splat, splat-pad, and the",
        "set/incr mask pair.",
    ],
    "construct/opaque.rs": [
        "THE OPAQUE-TEMPLATE INSTRUCTION — `ConstructOpaqueInstr` and the string trim it uses.",
        "",
        "⭐ THE TEMPLATES ARE DATA, IN `dcc/src/Conversion/SentientToProgIR/opaqueTemplates/` — 38",
        "`.smc` files (exp, gelu, sigmoid, rsqrt, layernormscale, idx32toaddr, …). They are a",
        "declared-data table, not code; this crate's `build.rs` ALREADY reads `.smc` mnemonics and",
        "emits `InstOpCode::…`, so a template naming a non-opcode fails to compile.",
    ],
    "construct/reg_init.rs": [
        "WHAT MUST BE IN A REGISTER BEFORE ANYTHING READS IT — the reg-init accumulation and the",
        "PE/SFP LRF immediate copies.",
        "",
        "⛔ REGISTER-FILE ALLOCATIONS NEED `primaryDimToVal_st`'s rowSplit_/peSfpSplit_ SHARE. That",
        "input was missing once and produced 199 refusals.",
    ],
    "lower/transfer.rs": [
        "THE TRANSFER OPERATION HANDLERS — load_and_send, receive_and_store, load_and_store, the",
        "scalar extracts, load_compute_and_send and copy.",
        "",
        "⭐ THESE ARE FIVE OF THE SEVEN SENTIENT OPS THE REFERENCE ACTUALLY EMITS across all 417",
        "staged programs, so this file is on the hot path for every program.",
    ],
    "lower/compute.rs": [
        "THE COMPUTE OPERATION HANDLERS — mac, binary, unary, ternary, add, sub, splat, opaque, samv",
        "and the mask pair, plus `LowerCommonOperations` that dispatches them.",
    ],
    "lower/control.rs": [
        "THE CONTROL FLOW AND THE UNIFORM REGIONS — for, yield, return, sync, nop, set-send-dest, and",
        "the uniform-operation pair.",
        "",
        "⛔ `LowerUniformOperations` AND `GenerateProgIR` ARE MUTUALLY RECURSIVE (level 5, one",
        "component). Neither can be ported as if the other were a leaf.",
        "⛔ THE SYNC ORDER IS dxp's AND CITED: reordering syncs times the card out (CB state=TimedOut,",
        "sync rc=-1).",
    ],
    "lower/labels_and_regs.rs": [
        "THE LABELS, THE CODE GRAPH AND THE REGISTER IMMEDIATES — AddToLabelsMap,",
        "updateLabelAndAddToCodeGraph, addToRegInit, GetAddressScale, GetOpCodePrefix, getRegImmVals.",
    ],
    "uniform/instr.rs": [
        "ONE INSTRUCTION AND ITS OPERAND MAP — `UniformInstrInfo` and `OperandMap`.",
        "",
        "⭐ `addEntryToOperandMap` HAS TWO OVERLOADS (121L and 43L) and they are two units.",
    ],
    "uniform/block.rs": [
        "THE BLOCKS AND THE REGIONS — `UniformInstrBlock` and `UniformInstrBlocks`: per-unit",
        "instruction lists, the uniformised lists, region indices and the flattened index.",
        "",
        "⛔ A REGION INDEX OF -1 IS \"NOT PRESENT\", and a REGULAR block always answers 0",
        "(`UniformInstrAndBlock.hpp:208`). A sentinel is an `Option` here, never a negative number.",
    ],
    "driver.rs": [
        "THE D76 PASS ITSELF — `runOnOperation`, `GenerateProgIR`, `GenerateProgIRForProgramUnit`,",
        "the register initialisation, the program-length equalisation and the `init.smc` collapse.",
        "",
        "⭐ ONE `ProgramUnitOp` SPANS THE SET: handles = cores x corelets x num_folds.",
        "⛔ THE PROGRAM BOUNDS ARE READ, NOT RESTATED — `max_ibuff_entries(unit)` per component and",
        "per arch, never `kMaxCompIBuff` as if it were any unit's limit.",
    ],
    "reg_def_tracker.rs": [
        "WHICH REGISTERS ARE DEFINED, AND WHETHER ANYTHING READS ONE THAT IS NOT.",
        "",
        "⛔⛔ `checkRegDefs` IS A CHECK, AND THIS CRATE NEVER RUNTIME REFUSES. Port it as a function",
        "that RETURNS THE OFFENDERS, the way `progir::Program::overflowing` does — never as an",
        "`assert!`, a `Result` or a `signalPassFailure`.",
        "⭐ `~UniformRegionContext` IS RAII: real work runs on scope exit (it merges the region's reg",
        "defs). It is unit `e070_dtor_UniformRegionContext`, not a destructor to drop.",
    ],
    "utils.rs": [
        "THE SHARED HELPERS — the on-the-fly conversion check, the proper-consumer walk, the address",
        "wraparound and the fold-mode value.",
        "",
        "⛔ `verifyOnTheFlyConversions` IS A CHECK — see `reg_def_tracker.rs`'s note. Return the",
        "offenders, never refuse.",
    ],
}

MODHEAD = """\
// SPDX-License-Identifier: Apache-2.0
//! `SentientIR -> ProgIR` — dcc pass **D76**, the last MLIR rung, as one lowering over the TYPED
//! program.
//!
//! ```text
//! islands::sentient::Run<A, M, W>  ──here (D76)──►  islands::progir::Program<A, M, W>
//! ```
//!
//! # 🛑 THIS RUNG LEAVES MLIR
//!
//! ⛔⛔ **PROGIR IS NOT AN MLIR DIALECT.** `SentientToProgIR` fills in a plain C++ structure —
//! `std::map<int, ProgramAndStateInfo>` keyed by core id — and the MLIR module it leaves behind
//! holds only [`crate::islands::progir::dialects::init`]'s reference to it. So this bridge produces
//! a VALUE, and the only text it emits is the twelve-line `init.smc` module.
//!
//! ⭐ AND THE OUTPUT IS TINY. `kMaxCompIBuff = 256`, `kMaxCompRegs = 128`; a complete int8 batched
//! matmul compiles to **eight** instructions on one unit.
//!
//! # 🛑 WHAT IS AND IS NOT IN THIS SPAN
//!
//! ⭐ IN SCOPE: the {n} functions of `dcc/src/Conversion/SentientToProgIR/` ({loc} declaration
//! lines), listed with their authority citations in `crustify-bridge3/UNITS.tsv` and their
//! exclusions in `crustify-bridge3/EXCLUSIONS.tsv`.
//!
//! ⛔⛔ **NOT IN SCOPE: `dcc/src/Transform/Sentient/`** — 32,766 lines of D29-D75 passes that
//! rewrite SentientIR *in place* (`RegisterAllocation`, `SmartRegisterAllocation`,
//! `RegisterTypeAssignment`, `AddressPinningAndToggle`, `LiveRangeReduction`, `OpRerolling`,
//! `LoopRolling`, …). They are island-internal transforms, not this conversion. A function here
//! that depends on state they establish — register assignments, pinned addresses, rerolled loops —
//! is ported **as the reference writes it**, with the dependency recorded in the commit message.
//! Do not port the pass, and do not invent the state.
//!
//! Ported from the authority tree `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`;
//! every function carries a `/// Replaces: eNNN_name` anchor citing its original.
"""


def main():
    rows = [l.split("\t") for l in UNITS.read_text().splitlines()[1:]]
    by_home = collections.defaultdict(list)
    total_loc = 0
    for unit, entry, level, loc, auth, exl, home, callees in rows:
        rel = home.split("src/bridges/sentient_to_progir/", 1)[1]
        by_home[rel].append((unit, int(level), int(loc), auth))
        total_loc += int(loc)

    OUT.mkdir(parents=True, exist_ok=True)
    dirs = sorted({h.split("/")[0] for h in by_home if "/" in h})

    # ---- top-level mod.rs
    mod = MODHEAD.format(n=len(rows), loc=total_loc).split("\n")
    for d in dirs:
        n = sum(len(v) for k, v in by_home.items() if k.startswith(d + "/"))
        mod.append("")
        mod.append(f"/// {'`' + d + '`'} — {n} units.")
        mod.append(f"pub mod {d};")
    for h in sorted(k for k in by_home if "/" not in k):
        mod.append("")
        mod.append(f"/// {BLURB[h][0]}")
        mod.append(f"pub mod {h[:-3]};")
    mod.append("")
    (OUT / "mod.rs").write_text("\n".join(mod))

    # ---- per-directory mod.rs
    for d in dirs:
        (OUT / d).mkdir(exist_ok=True)
        kids = sorted(k for k in by_home if k.startswith(d + "/"))
        src = {"construct": "ConstructProgIRHelper.cpp",
               "lower": "LowerSentientHelper.cpp",
               "uniform": "UniformInstrAndBlock.cpp / .hpp"}[d]
        lines = [
            f"//! `{src}` — {sum(len(by_home[k]) for k in kids)} units, one submodule per family.",
            "",
        ]
        for k in kids:
            lines.append(f"/// {BLURB[k][0]}")
            lines.append(f"pub mod {k.split('/')[1][:-3]};")
            lines.append("")
        (OUT / d / "mod.rs").write_text("\n".join(lines))

    # ---- the homes themselves
    for home, units in sorted(by_home.items()):
        units.sort(key=lambda u: (u[1], u[0]))
        lines = [f"//! {l}" if l else "//!" for l in BLURB[home]]
        lines += [
            "//!",
            f"//! {len(units)} units. Every citation resolves against the authority tree",
            "//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.",
            "//!",
            "//! | unit | level | LoC | authority |",
            "//! |---|---|---|---|",
        ]
        for unit, level, loc, auth in units:
            lines.append(f"//! | `{unit}` | {level} | {loc} | `{auth}` |")
        lines += [
            "",
            "// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function",
            "// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.",
            "",
        ]
        lines += [f"// crustify:todo: {u[0]}" for u in units]
        lines.append("")
        (OUT / home).write_text("\n".join(lines))
        print(f"  {home:32} {len(units)} anchors")

    print(f"\n  mod.rs declares {len(dirs)} nested modules and "
          f"{len([k for k in by_home if '/' not in k])} leaf modules")
    print(f"  {sum(len(v) for v in by_home.values())} anchors total")


if __name__ == "__main__":
    main()
