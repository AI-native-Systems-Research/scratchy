#!/usr/bin/env python3
"""Generate the Rust scaffolding: real nested submodules under src/schedule/.

⛔⛔ THIS GENERATOR REFUSES TO OVERWRITE PORTED WORK. Re-running bridge 4's generator destroyed its
finished 2,299-line port, and the first fix -- guarding files that contain `/// Replaces:` -- still
ate a `mod.rs` along with its test fixtures.

So the guard here is not a content sniff. Every file this generator writes is recorded in
scaffold-manifest.json with the sha256 of exactly what was written. On a later run a file is
rewritten ONLY when its current sha256 still equals the recorded one, i.e. nobody has touched it
since. Anything else -- edited, ported, hand-fixed, or simply not in the manifest -- is SKIPPED and
reported. There is no --force.
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
from collections import defaultdict

WORK = _paths.WORK
CAMP = _paths.CAMP
VSTATS = json.load(open(os.path.join(_paths.WORK, "verify-stats.json")))
TREE = _paths.TREE
AUTH = _paths.AUTH_ROOT
REV = _paths.REV
OUTDIR = _paths.OUTDIR
CAMPNAME = _paths.CAMPNAME
MANIFEST = os.path.join(CAMP, "scaffold-manifest.json")

BANNER = r'''// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY ddc / L3-SCHEDULER CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.     ║
// ║ Campaign statement: crustify-ddc/TASK.md   ·   worklist: crustify-ddc/UNITS.tsv              ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       __AUTH__/<file>:<line>     (revision __REV__)
//    Every citation below resolves against that revision. `crustify-ddc/cpp/{l3,ddc,ddl,dcg}.cpp`
//    say WHICH functions are in scope and IN WHAT ORDER; their bodies were verified byte-identical
//    to the authority (__VOK__/__VN__, __VB__ bytes, __VC__ negative controls DETECTED), so either
//    may be read — but the authority file is the one that carries the surrounding declarations you
//    will need. ⛔ The other local deeptools checkout is a DIFFERENT revision; the pod is not
//    reachable from here.
//
// 2. WHAT THIS STAGE IS. `dbo-opt` is the binary scratchy shells out to; its per-program pipeline
//    runs `runDdc` for every program (dbo/src/Transforms/sdsc_bundle/RunSchedulerOnSdsc.cpp:145).
//    ⭐ `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp` — 60 lines — IS THE SPEC FOR THIS WHOLE
//    JOB. READ IT FIRST. Four stages:
//      1.  sbf::doCoreletSplitSdsc     ⛔ EXCLUDED: SchedulerStages.cpp:25 returns early unless
//          numCoreletsPerCore == 2, and scratchy emits numCoreletsUsed_ = 1 in all 313 sampled
//          SuperDSCs. See crustify-ddc/EXCLUSIONS.tsv.
//      2a. L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose).run(sdsc)
//      2b. ddc::Ddc(dscGlobal, ..).run_v1(sdsc)      entry: ddc/ddcv1.cpp:3695
//      3.  DcgManager::runDcgForDlOpsStandalone(sdsc)   dcg/dcg_manager/dcg_manager.cpp:449
//          ⛔ THAT branch, NOT `runDcg`: SchedulerStages.cpp:53-57 picks it whenever `dscs_` is
//          non-empty, always true for scratchy's input. ⚠️ Its body delegates almost entirely to
//          `dcg_fe/pcfg_gen/` and `dcg_be/`, both OUT of scope — `todo!` NAMING the missing
//          translator is correct there; ⛔ do NOT invent a PCFG.
//    `runDdc` raises when DDC finds no mapping ("Scheduler failed to find a suitable op mapping"),
//    so the mapping is not optional.
//
// 3. WHY IT MATTERS. `ddc.run_v1` is what PLACES ADDRESSES, and the L3 scheduler's `run` commits
//    LX allocations then calls `fillAllocationStartAddrAndOffset` — literally "Set start address,
//    offset in allocations" (dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:8000). Our emitted views
//    have printed `start_address = 0` where the reference states a placed base; the backend has
//    refused with `Register initialization out of boundary`; and `src/reginit.rs` (1,569 lines, on
//    the integ branch) HAND-COMPUTES placement from ddc/ddcv1.cpp:132-360. This campaign replaces
//    that guesswork with the real thing.
//
// 4. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For this stage the effect is WHAT IS
//    WRITTEN INTO THE `SuperDsc`: addresses, mappings, symbol definitions, fold state, schedule
//    steps. A hand attempt on bridge 2 extracted each function's decision rule into a documented
//    predicate, omitted the part that changed the IR, and reported it done — nothing called any of
//    it. A PREDICATE IS NOT A PORT. Droppable: only the mechanism for REACHING operands (walking
//    uses, memoising, positioning a builder). ⛔ If a TYPE cannot express a result, EXTEND THE
//    TYPE. Deciding a function is unnecessary is NOT the porter's call.
//
// 5. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 6. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no sanitizers, ❌ no C-vs-Rust harness.
//
// 7. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    NEWTYPES, NEVER RAW SCALARS — an address, an offset, a core index and an element count must
//    not be interchangeable `u64`s; transposing two must be E0308. A closed set is an `enum`, never
//    a string. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED here
//    — and ⛔ never substitute a stand-in op, or a fabricated address, to dodge one. A fabricated
//    placement to avoid a stop is worse than the stop.
//
// 8. ⚠️ THE L3 SCHEDULER MAY OR MAY NOT APPLY TO US — DO NOT DECIDE THIS, PORT IT. Scratchy states
//    its own schedule (our SuperDSC writes `coreIdToDscSchedule`) and L3DlOpsScheduler.cpp:415-425
//    READS that field. That question is the USER'S, not the porter's. Measured while scoping: the
//    field occurs in that file ONLY as a read, never a write, so it is an INPUT to both stages —
//    what the L3 scheduler PRODUCES is the dsc2 schedule tree, the LX buffer type, the committed
//    LX allocations and their start addresses.
//
// 9. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 10. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//     the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 11. ⭐⭐ THE ACCEPTANCE CRITERION — WHAT THESE STAGES PRODUCE. They mutate the `SuperDsc` IN
//     PLACE, and the observable result is that THE `ScheduleNode` TREE GAINS ITS LOOP, TRANSFER,
//     SYNC AND CONDITION NODES. Scratchy's SuperDSC today has only ALLOCATE nodes (one construction
//     site: `crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:5127`) plus a flat
//     `computeOp_` list — which matches torch-spyre's own `generate_sdsc`, and is exactly why
//     `dxp_standalone` works and the Rust `sdscToDataflowIR` port yields nothing.
//     The REPRESENTATIVE MINTING SITE to model is `ddc/ddl/ddl_conversion.cpp:1065`: it mints a
//     `dsc2::LoopNode`, takes its dims from the DDL, names it `loop_ds<num>_ds<den>`, and registers
//     it in `ddlInterface.loop_labels_`.
//     ⛔ A PORT THAT DOCUMENTS THE SCHEDULING DECISION WITHOUT ADDING NODES TO THAT TREE IS NOT A
//     PORT.
//
// 12. ⭐⭐ THE TARGET VOCABULARY IS ALREADY TYPED — a wrong shape must be a COMPILE ERROR, not a
//     judgement call. Emit into these EXISTING types; do not invent any:
//       `src/bridges/superdsc_to_dataflow_ir/driver.rs:467`        `Statement`
//       `driver.rs:1152`                                           `Scheduled`
//       `driver.rs:1756-1790`                                      `Viewed` / `Viewing`
//       `driver.rs:1539`                                           `ScheduleView::roots`
//       `driver.rs:1788`                                           `Dsc`
//     The single function it all plugs into is `Schedule::roots` in
//     `crates/targets/spyre/src/lower_superdsc_to_dataflow_ir.rs` — committed, compiles, and
//     currently yields nothing. Both the component and the DSC are already in hand there.
//
// 13. ALREADY PORTED, DO NOT DUPLICATE — all four in
//     `src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs`, following the same
//     `/// Replaces: eNNN_name` convention so cross-referencing works:
//       `e001_checkConstraints` (ddc/ddcv1.cpp:792)   `e002_createDataConnectMetadata` (:3283)
//       `e041_getStickSizes`    `e071_getCumulativeStickSizes`  (both `DesignSpaceConfig` methods
//       in `dsc/dsc2.cpp`, OUTSIDE this campaign's file list — your units CALL them.)
//     `checkConstraints` is a LAMBDA inside `Ddc::exploreAssignDataStages`, so porting that unit
//     means CALLING the existing port, not writing a second constraint checker. Units carrying such
//     a constraint have a ⛔ NOTE on the anchor itself.
'''.replace("__AUTH__", _paths.AUTH_ROOT).replace("__REV__", REV) \
    .replace("__VOK__", str(VSTATS["units"] - VSTATS["failures"])) \
    .replace("__VN__", str(VSTATS["units"])) \
    .replace("__VB__", "{:,}".format(VSTATS["bytes"])) \
    .replace("__VC__", "%d of %d" % (VSTATS["negative_controls_detected"],
                                     VSTATS["negative_controls"]))


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def table(units):
    out = ["//!", "//! | unit | entry | level | lines | class | authority path:line |",
           "//! |---|---|---|---|---|---|"]
    for u in units:
        out.append("//! | `%s` | %03d | %d | %d | %s | `%s:%d` |"
                   % (u["unit"], u["entry"], u["level"], u["body_lines"],
                      ("`%s`" % u["cls"]) if u["cls"] else "—", u["rel"], u["head_line"]))
    return out


def wrap(s, w):
    words, line, out = s.split(), "", []
    for x in words:
        if len(line) + len(x) + 1 > w:
            out.append(line)
            line = x
        else:
            line = (line + " " + x) if line else x
    if line:
        out.append(line)
    return out


def anchors(units):
    out = []
    for u in units:
        out.append("")
        out.append("// crustify:todo: %s" % u["unit"])
        out.append("//   authority : %s:%d  (%d body lines, level %d)"
                   % (u["rel"], u["head_line"], u["body_lines"], u["level"]))
        if u["cls"]:
            out.append("//   class     : %s" % u["cls"])
        orig = re.sub(r"\s+", " ", u["head"]).strip()
        out.append("//   original  : %s" % orig[:300])
        out.append("//   extract   : %s:%s" % (u["extract_file"], u["extract_lines"]))
        if u["call_units"]:
            out.append("//   calls     : " + ", ".join(u["call_units"][:14])
                       + (" …" if len(u["call_units"]) > 14 else ""))
        if u["note"]:
            for i, chunk in enumerate(wrap(u["note"], 88)):
                out.append("//   %s %s" % ("⛔ NOTE   :" if i == 0 else "           ", chunk))
    return out


SCOPE_BLURB = {
    "l3": ("STAGE 1 of `runDdc`: `L3DlOpsScheduler::run` — the L3 DL-ops scheduler. Builds the "
           "dsc2 schedule tree, picks the LX buffer type, commits LX allocations and fills their "
           "start addresses and offsets."),
    "ddc": ("STAGE 2 of `runDdc`: `ddc::Ddc::run_v1` — the Deep Dataflow Constructor. Places "
            "addresses, assigns data stages, folds coordinates and finalizes ops."),
    "ddl": ("STAGE 2's DDL conversion sub-surface, reached from `Ddc`: dsc2 <-> DDL conversion and "
            "the DDL op dialect. ⭐ `ddl_conversion.cpp:1065` is the REPRESENTATIVE NODE-MINTING "
            "SITE to model: it mints a `dsc2::LoopNode`, takes its dims from the DDL, names it "
            "`loop_ds<num>_ds<den>` and registers it in `ddlInterface.loop_labels_`."),
    "dcg": ("STAGE 3 of `runDdc`'s pipeline: `DcgManager::runDcgForDlOpsStandalone` "
            "(`dcg/dcg_manager/dcg_manager.cpp:449`) — the branch "
            "`SchedulerStages.cpp:53-57` picks whenever `dscs_` is non-empty, which is always true "
            "for scratchy's input. ⛔ NOT `runDcg`. ⚠️ Its body delegates almost entirely to "
            "`dcg_fe/pcfg_gen/` and `dcg_be/`, both OUT of scope, so expect `todo!` naming those "
            "rather than an invented PCFG."),
}


def build():
    units = json.load(open(WORK + "/units.json"))
    by_home = defaultdict(list)
    for u in units:
        by_home[u["rust_home"]].append(u)
    for v in by_home.values():
        v.sort(key=lambda u: u["entry"])

    mods = sorted({u["module"] for u in units})
    subs = defaultdict(set)
    for u in units:
        if u["stem"]:
            subs[u["module"]].add(u["stem"])

    files = {}
    for home, us in by_home.items():
        mod = us[0]["module"]
        stem = us[0]["stem"]
        srcs = sorted({u["rel"] for u in us})
        levels = sorted({u["level"] for u in us})
        lines = [BANNER.rstrip("\n"), ""]
        lines.append("//! `%s` — %d of the campaign's %d units (dependency level(s) %s)."
                     % ("`, `".join(srcs), len(us), len(units), levels))
        lines += table(us)
        lines.append("")
        if stem is None and mod in subs:
            for s in sorted(subs[mod]):
                lines.append("pub(crate) mod %s;" % s)
            lines.append("")
        lines += anchors(us)
        lines.append("")
        files[home] = "\n".join(lines) + "\n"

    # a mod.rs for any module whose units are all in stem files (no mod.rs home of its own)
    for mod in mods:
        root = "%s/%s/mod.rs" % (OUTDIR, mod)
        if root in files:
            continue
        scope = next(u["scope"] for u in units if u["module"] == mod)
        lines = [BANNER.rstrip("\n"), ""]
        lines.append("//! %s" % SCOPE_BLURB[scope])
        lines.append("//!")
        lines.append("//! %d units across %d file(s)."
                     % (sum(1 for u in units if u["module"] == mod), len(subs[mod])))
        lines.append("")
        for s in sorted(subs[mod]):
            lines.append("pub(crate) mod %s;" % s)
        lines.append("")
        files[root] = "\n".join(lines) + "\n"

    root = "%s/mod.rs" % OUTDIR
    lines = [BANNER.rstrip("\n"), ""]
    lines.append("//! The scheduling and address-placement stage of deeptools — %d units across "
                 "%d modules." % (len(units), len(mods)))
    lines.append("//!")
    lines.append("//! `dbo-opt`'s per-program pipeline runs `runDdc` for every program")
    lines.append("//! (`dbo/src/Transforms/sdsc_bundle/RunSchedulerOnSdsc.cpp:145`), and `runDdc` is")
    lines.append("//! exactly two stages (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41`):")
    lines.append("//! the L3 DL-ops scheduler, then the Deep Dataflow Constructor. This module is")
    lines.append("//! both.")
    lines.append("")
    for m in mods:
        lines.append("pub(crate) mod %s;" % m)
    lines.append("")
    files[root] = "\n".join(lines) + "\n"
    return files, units, mods


def main():
    files, units, mods = build()
    old = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    wrote, skipped, unchanged = [], [], []
    for rel, content in sorted(files.items()):
        p = os.path.join(TREE, rel)
        if os.path.exists(p):
            cur = sha(open(p).read())
            if rel not in old:
                skipped.append((rel, "exists and is not in the scaffold manifest"))
                continue
            if cur != old[rel]:
                skipped.append((rel, "has been edited since it was generated (sha differs)"))
                continue
            if cur == sha(content):
                unchanged.append(rel)
                continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)
        wrote.append(rel)
        old[rel] = sha(content)
    json.dump(old, open(MANIFEST, "w"), indent=1, sort_keys=True)

    # lib.rs: surgical, idempotent insertion only
    lib = os.path.join(TREE, "crates/compiler/deeptools/src/lib.rs")
    note = ""
    if os.path.exists(lib):
        s = open(lib).read()
        if "pub mod schedule;" not in s:
            m = re.search(r"^pub mod islands;.*$", s, re.M)
            ins = ("\n/// The scheduling and address-placement stage: `runDdc`'s L3 DL-ops\n"
                   "/// scheduler and Deep Dataflow Constructor.\n"
                   "pub mod schedule;\n")
            if m:
                s = s[:m.end() + 1] + ins + s[m.end() + 1:]
            else:
                s = s + ins
            open(lib, "w").write(s)
            note = "lib.rs: inserted `pub mod schedule;`"
        else:
            note = "lib.rs: `pub mod schedule;` already present"

    print("homes generated    : %d written, %d unchanged, %d SKIPPED"
          % (len(wrote), len(unchanged), len(skipped)))
    print("modules            : %d" % len(mods))
    print("anchors emitted    : %d" % len(units))
    if note:
        print("                     %s" % note)
    for rel, why in skipped:
        print("   ⛔ SKIPPED (ported work preserved): %s — %s" % (rel, why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
