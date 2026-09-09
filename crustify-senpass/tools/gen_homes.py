#!/usr/bin/env python3
"""Generate the Rust scaffolding: real nested submodules under src/transform/sentient/.

⛔⛔ THIS GENERATOR REFUSES TO OVERWRITE PORTED WORK. Re-running bridge 4's generator for its
*schedules* destroyed its finished 2,299-line port, and the first fix - guarding files that
contain `/// Replaces:` - still ate a `mod.rs` along with its test fixtures.

So the guard here is not a content sniff. Every file this generator writes is recorded in
scaffold-manifest.json with the sha256 of exactly what was written. On a later run a file is
rewritten ONLY when its current sha256 still equals the recorded one, i.e. nobody has touched it
since. Anything else - edited, ported, hand-fixed, or simply not in the manifest - is SKIPPED and
reported. There is no --force.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import re
from collections import defaultdict

WORK = _paths.WORK
CAMP = _paths.CAMP
TREE = _paths.TREE  # repo root to write into
MANIFEST = os.path.join(CAMP, "scaffold-manifest.json")
REL = _paths.REL
AUTH = _paths.AUTH_ROOT

BANNER = r'''// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
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
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.
'''


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def table(units):
    out = ["//!", "//! | unit | entry | level | lines | authority path:line |", "//! |---|---|---|---|---|"]
    for u in units:
        out.append(f"//! | `{u['unit']}` | {u['entry']:03d} | {u['level']} | {u['body_lines']} | "
                   f"`{REL}/{u['file']}:{u['head_line']}` |")
    return out


def anchors(units):
    out = []
    for u in units:
        out.append("")
        out.append(f"// crustify:todo: {u['unit']}")
        out.append(f"//   authority : {REL}/{u['file']}:{u['head_line']}  "
                   f"({u['body_lines']} body lines, level {u['level']})")
        orig = re.sub(r"\s+", " ", u["head"]).strip()
        out.append(f"//   original  : {orig[:300]}")
        if u["call_units"]:
            out.append("//   calls     : " + ", ".join(u["call_units"][:14])
                       + (" …" if len(u["call_units"]) > 14 else ""))
    return out


def build():
    units = json.load(open(WORK + "/units.json"))
    by_home = defaultdict(list)
    for u in units:
        by_home[u["rust_home"]].append(u)
    for v in by_home.values():
        v.sort(key=lambda u: u["entry"])

    subs = defaultdict(set)
    for u in units:
        if u["submodule"]:
            subs[u["module"]].add(u["submodule"])
    mods = sorted({u["module"] for u in units})

    files = {}
    for home, us in by_home.items():
        mod = us[0]["module"]
        sub = us[0]["submodule"]
        stems = sorted({os.path.splitext(u["file"])[0] for u in us})
        lines = [BANNER.rstrip("\n"), ""]
        what = f"`{'`, `'.join(s + '.cpp' for s in stems)}`"
        levels = sorted({u["level"] for u in us})
        lines.append(f"//! {what} — {len(us)} of the campaign's {len(units)} units "
                     f"(dependency level(s) {levels}).")
        lines += table(us)
        lines.append("")
        if sub is None and mod in subs:
            for s in sorted(subs[mod]):
                lines.append(f"pub(crate) mod {s};")
            lines.append("")
        lines += anchors(us)
        lines.append("")
        files[home] = "\n".join(lines) + "\n"

    # transform/sentient/mod.rs — the island-extension root
    root = "crates/compiler/deeptools/src/transform/sentient/mod.rs"
    lines = [BANNER.rstrip("\n"), ""]
    lines.append(f"//! The D29–D75 in-place SentientIR passes — {len(units)} units across "
                 f"{len(mods)} passes.")
    lines.append("//!")
    lines.append("//! Input and output are both `crate::islands::sentient`. Every pass in the "
                 "shipped driver")
    lines.append("//! (`dcc/tools/dcc-standalone/dcc-standalone-main.cpp`, between "
                 "`createAgenToSentientPass`")
    lines.append("//! at :271 and `createSentientToProgIRPass` at :755) is added UNCONDITIONALLY — "
                 "none is")
    lines.append("//! behind a flag or an option, so there is no \"just the required ones\" subset.")
    lines.append("")
    for m in mods:
        lines.append(f"pub(crate) mod {m};")
    lines.append("")
    files[root] = "\n".join(lines) + "\n"

    tmod = "crates/compiler/deeptools/src/transform/mod.rs"
    files[tmod] = ("// SPDX-License-Identifier: Apache-2.0\n"
                   "//! In-place IR-to-itself passes, ported from `dcc/src/Transform/`.\n\n"
                   "pub(crate) mod sentient;\n")
    return files, units, mods


def main():
    files, units, mods = build()
    old = {}
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST))
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
        if "pub mod transform;" not in s:
            m = re.search(r"^pub mod islands;.*$", s, re.M)
            ins = ("\n/// The in-place `dcc/src/Transform/` passes: SentientIR rewritten in place.\n"
                   "pub mod transform;\n")
            if m:
                s = s[:m.end() + 1] + ins + s[m.end() + 1:]
            else:
                s = s + ins
            open(lib, "w").write(s)
            note = "lib.rs: inserted `pub mod transform;`"
        else:
            note = "lib.rs: `pub mod transform;` already present"

    print(f"homes generated    : {len(wrote)} written, {len(unchanged)} unchanged, "
          f"{len(skipped)} SKIPPED")
    print(f"pass modules       : {len(mods)}")
    print(f"anchors emitted    : {len(units)}")
    if note:
        print(f"                     {note}")
    for rel, why in skipped:
        print(f"   ⛔ SKIPPED (ported work preserved): {rel} — {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
