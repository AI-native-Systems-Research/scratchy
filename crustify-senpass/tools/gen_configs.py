#!/usr/bin/env python3
"""Write every crustify config this campaign needs.

⛔ THE TRANSLATOR PROMPT NEVER NAMES AGENT-BRIEF.md. It names only crustify/build.json,
crustify/wavefront/wavefront-config.json, crustify/crates.json, the worklist and the worktree.
Anything written only in a brief never reaches an agent: bridge 2's first 144 functions ran with no
budget at all for exactly that reason, and two bridge-3 agents ran a whole batch against a sibling
campaign's inherited config. So the campaign's facts AND the hard caps live in the `_comment`
arrays of build.json and wavefront-config.json, and in the banner of every home .rs.

⛔ crustify reads the REPO-TIER crustify/crates.json. This worktree was branched from bridge2-campaign
and INHERITED bridge 2's copy, which names bridge 2's homes; left alone, crustify refuses the port
stage with "N selected item(s) have no home .rs on disk" while our homes exist. These files replace it.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
from collections import defaultdict

WORK = _paths.WORK
CAMP = _paths.CAMP
TREE = _paths.TREE
REL = _paths.REL
AUTH = _paths.AUTH_ROOT
REV = _paths.REV
EXTRACT = _paths.EXTRACT

CAPS = [
    "⛔ HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock,",
    "89% model time and 586 M cache-read tokens against 5 M output, at 30-48 turns per function; of",
    "the 30,991 lines produced only 5,306 were implementation (12,333 doc comments, 12,575 tests).",
    "PER PORTED FUNCTION:",
    "  * 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, ONE line of what it does, any TRAP.",
    "    No tutorials, no restating the C++ in prose, no design essays.",
    "  * ONE TEST. Two only where the vendor's own case AND a negative both apply.",
    "  * `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, never per function.",
    "  * Do NOT re-verify citations — the review pass owns that.",
    "  * Do NOT grep the crate to discover types; the anchor names what you need.",
    "NOT capped: correctness, and the emission.",
    "⛔ AGENTS MUST NOT run the workspace build, the acceptance build, or clippy over the workspace:",
    "   ~6 GB of target/ per agent worktree and this host is tight. The orchestrator owns that gate.",
]

PORTED_MEANS = [
    "⭐ PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. These passes rewrite SentientIR IN",
    "PLACE — they are NOT a conversion between rungs like bridges 1-4, so input and output are both",
    "`src/islands/sentient/`. For an in-place pass THE EFFECT IS THE PORT: which ops are rewritten,",
    "which attributes are set to what, in what order. A hand attempt on bridge 2 extracted each",
    "function's decision rule into a documented predicate, omitted the part that changed the IR, and",
    "reported it done — nothing called any of it. A PREDICATE IS NOT A PORT.",
    "Droppable: only the mechanism for reaching operands (walking uses, memoising, positioning a",
    "builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND — deciding a function is",
    "unnecessary is not the porter's call.",
]

WHY = [
    "⭐ WHY THIS CAMPAIGN EXISTS. Bridge 2's ported SentientIR emission assigns NO registers",
    "(`index: None` in 39 of 41 sites) and that is FAITHFUL: the reference's own SentientIR at that",
    "point carries `regIndex = -1 : i32` in all 54 occurrences of the committed golden corpus",
    "(crates/compiler/deeptools/tests/sentient_corpus/). THESE passes are what turn -1 into a real",
    "register file and index. Without them ProgIR gets -1 where an instruction needs a register and",
    "the backend refuses with `Register initialization out of boundary` — already observed on",
    "lxsu0:LRF0 and l3lu:LBR2. The register chain specifically: RegisterTypeAssignment,",
    "AddressRegisterPrecisionAssignment, RegisterInitialization, VectorRegisterInitialization,",
    "SmartRegisterAllocation, ReadOnlyRegisterRenumbering, RegisterPacking, PortAssignment,",
    "AddressPinningAndToggle.",
]

SCOPE = [
    "⭐ THE SCOPE IS NOT NEGOTIABLE AND HAS NO SUBSET. Measured on the shipped driver",
    "(dcc/tools/dcc-standalone/dcc-standalone-main.cpp) with a brace/conditional tracker: between",
    "`createAgenToSentientPass` at :271 and `createSentientToProgIRPass` at :755, EVERY ONE of the",
    "48 passes is added UNCONDITIONALLY — none is behind a flag or an option. Do not look for a",
    "\"just the required ones\" subset; there isn't one.",
]

AUTHORITY = [
    f"⭐ THE ULTIMATE AUTHORITY IS THE LOCAL C++ TREE: {AUTH}",
    f"   repo_info.txt: deeptools|master|{REV}  — exactly the revision every banner cites.",
    "⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. Do not use it.",
    "⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the local tree.",
    f"The extract {EXTRACT} says WHICH functions are in scope and IN WHAT ORDER.",
    "Unlike bridge 2's extract (which was truncated at the tail of 366 of 384 bodies), this one was",
    "verified INDEPENDENTLY: crustify-senpass/tools/verify_extract.py re-derives every body's end",
    "with its own character state machine — it does not import the extractor's scanner and never",
    "re-slices with the extractor's own (file, line, length) — and reports 656/656 bodies",
    "byte-identical to the authority, 962,619 bytes compared, with two negative controls (a dropped",
    "last line, a blanked inner brace) both DETECTED. Either file may be read; the authority file is",
    "the one that carries the surrounding declarations you will need.",
]


ANALYSES = [
    "⛔ 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS NOT IN THIS CAMPAIGN'S SCOPE, and 122 of the",
    "656 units name one directly. MEASURED, not guessed — see crustify-senpass/OUTSIDE-DEPS.tsv",
    "(per pass) and OUTSIDE-UNITS.tsv (per unit), built from the passes' own #include lines and the",
    "analysis class names their bodies mention. The scope is the TOP LEVEL of",
    f"{REL}/*.{{cpp,hpp,h}}; these two subdirectories are OUTSIDE it and total ~11,700 more lines:",
    f"   {REL}/Analyses/               (39 files) — Liveness, PropagationAnalysis,",
    "       GraphColoring, ExpressionEvaluatorUtils, InstructionEstimation, CorrelationAnalysis,",
    "       TimeStamps, RegisterPressureAnalysis, AddressPinningScheme, XRFRegisterAnalyzer,",
    "       LoopGraph, BurstUtils, LiveRange, RedundantDefinitionEliminationTree,",
    "       CFGSSentientLevelConditionalTree, CFGDeepMergingConditionalTree, UniformGroupAnalysis",
    f"   {REL}/RegisterInitialization/ (12 files) — Candidate, Collector, Evaluator,",
    "       Selector, Transformer, UniformGrouper: the whole register-init candidate pipeline",
    "",
    "⭐ WHAT TO DO WHEN A UNIT NEEDS ONE. `todo!` NAMING THE MISSING ANALYSIS is the correct action,",
    "and it is allowed in this module. ⛔ DO NOT INVENT THE ANALYSIS, do not inline a guess at what",
    "it would have returned, and do not substitute a constant for its result — a fabricated analysis",
    "result is the exact failure mode that produced 'I golden-hacked the coefficient chain'. Port the",
    "part of the function that is present, `todo!(\"<Analysis>::<method> — out of campaign scope\")`",
    "for the part that is not, and leave the anchor filled so the unit is not lost. Extending",
    "`src/islands/sentient/` is a DIFFERENT case and IS expected: if the island cannot express a",
    "RESULT the function produces (an assigned register, a pinned address, a rolled loop), extend it.",
]


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_crates_json(units):
    by_mod = defaultdict(lambda: defaultdict(list))
    files_of = defaultdict(set)
    for u in units:
        by_mod[u["module"]][u["rust_home"]].append(u["unit"])
        files_of[u["module"]].add(u["file"])
    modules = {}
    for mod in sorted(by_mod):
        rs = {}
        for home in sorted(by_mod[mod]):
            crate_rel = home.split("crates/compiler/deeptools/", 1)[1]
            rs[crate_rel] = {
                "tu": EXTRACT,
                "headers": sorted(f"{REL}/{f}" for f in files_of[mod]),
                "members": {"functions": sorted(by_mod[mod][home])},
            }
        modules[mod] = {
            "rust_path": f"src/transform/sentient/{mod}",
            "rs": rs,
        }
    return {
        "_comment": [
            "Placement oracle for the Sentient in-place pass campaign (the D29-D75 span).",
            "",
            "⛔ THIS FILE REPLACES THE ONE THIS WORKTREE INHERITED FROM bridge2-campaign. crustify reads",
            "the REPO-TIER crustify/crates.json; with bridge 2's copy in place it looks for bridge 2's",
            "homes and refuses the port stage with \"N selected item(s) have no home .rs on disk\" while",
            "these homes exist on disk. Two bridge-3 agents ran a whole batch against a sibling",
            "campaign's inherited config for exactly this reason.",
            "",
            "Every unit lands in the EXISTING scratchy crate `deeptools`, under",
            "src/transform/sentient/<pass>/ — REAL NESTED SUBMODULES named after the passes, one module",
            "per original dcc pass file and a nested submodule per helper class. ⛔ NOT flat prefixed",
            "filenames: bridge 2's agen_*/tf_*/vc_* layout is the thing this campaign does not repeat.",
            "",
            f"UNIT NAMES are `e<NNN>_<cppName>`, NNN being the entry number in {EXTRACT} and in",
            "crustify-senpass/UNITS.tsv. Unlike bridge 2 — where 116 of 384 entries kept their bare C++",
            "name and the oracle had to warn \"ALWAYS resolve a unit through UNITS.tsv\" — EVERY entry's",
            "signature here is rewritten to its unit symbol, so symbol == unit name everywhere.",
            "",
            "`tu` is the extract for every entry because the extract is the campaign's declared",
            "impl_file; `headers` names the ORIGINAL dcc file(s) the .rs homes. There is no -sys",
            "surface: deeptools-sys is an empty placeholder.",
        ] + AUTHORITY + SCOPE + WHY + PORTED_MEANS + ANALYSES + CAPS,
        "crates": {
            "deeptools": {
                "kind": "library",
                "in_tree": True,
                "crate_path": "crates/compiler/deeptools",
                "sys_crate": "crustify/rust/deeptools-sys",
                "depends_on": [],
                "modules": modules,
            }
        },
    }


BUILD_JSON = {
    "_comment": [
        "Sentient in-place pass campaign build manifest. The campaign ports C++ into an EXISTING Rust",
        "crate (crates/compiler/deeptools); there is no C build to configure, no shim to compile and no",
        "FFI boundary, so `configure` is a no-op and the C-side gates in the translator playbook do not",
        "apply. ❌ no bindgen/-sys, ❌ no ffi::/extern \"C\", ❌ no unsafe, ❌ no C-vs-Rust harness:",
        "there is no C to call.",
        "",
        "⛔ TRANSLATOR GATE — run EXACTLY these two, from the repo root, ONCE PER BATCH:",
        "     cargo check -p deeptools",
        "     cargo test  -p deeptools",
        "   Both are seconds-to-a-minute: `deeptools` has one path dependency (sys-arch-spec) and no",
        "   registry deps.",
        "",
        "⛔ CRATE RULES BIND YOU — crates/compiler/deeptools/CLAUDE.md, in full. NEVER RUNTIME REFUSE:",
        "   no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`. Newtypes, never raw",
        "   scalars. No strings for closed sets — a closed set is a generated enum. Arch/Model/Workload",
        "   flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED in this module (the crate's refusal",
        "   ratchet, crates/targets/spyre/tests/dfir_never_runtime_refuses.rs, is scoped to ONE file,",
        "   src/lower_subtile_tape_to_dataflow_ir.rs, and does not cover src/transform/) — but ⛔ NEVER",
        "   substitute a stand-in op to dodge one.",
        "",
        "⛔ THE ACCEPTANCE COMMAND DOES NOT BUILD ON THIS HOST:",
        "     cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel",
        "   `superdsc` on the CLI turns on scratchy-serving-worker/sendnn -> flex-rs, whose build.rs",
        "   cc-compiles a senlib shim against /opt/ibm/spyre/senlib/include — Spyre-host-only headers",
        "   that do not exist on this Mac. The `pipeline` command below is the largest runnable prefix",
        "   that still expands #[forward] and therefore still runs this span over every staged program.",
    ] + CAPS + PORTED_MEANS + ANALYSES + AUTHORITY,
    "version": 1,
    "build_commands": {
        "configure": "true",
        "build": "cargo check -p deeptools",
        "test": "cargo test -p deeptools",
        "pipeline": ("cargo build -p scratchy-models --features superdsc,granite-3.1-2b-instruct,"
                     "scratchy-quantizations/fp8-dynamic-per-channel"),
        "acceptance_on_a_spyre_host": ("cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,"
                                      "quant/fp8-dynamic-per-channel"),
    },
}


def wavefront_config(nunits, nlevels):
    return {
        "_comment": [
            f"Campaign-wide inventory for the Sentient in-place passes: {nunits} units, dependency",
            f"levels 0..{nlevels}, from {REL}/*.{{cpp,hpp,h}} (32,766 lines of .cpp plus 1,672 of",
            "headers). This is the largest single stage of the compiler and the last unported one on",
            "the SuperDSC -> init_binary path; bridges 1-4 are COMPLETE.",
            "",
            "⛔ THERE IS NO CODEQL DATABASE AND NO WAVEFRONT EXTRACTION, and that is a decision:",
            "  1. The C++ surface is ONE self-contained TU whose bodies call names that exist only as",
            "     semantically empty stand-ins in cpp/prelude.inc. A CodeQL call graph over stand-ins",
            "     yields edges through a stub type, so it cannot order this work.",
            "  2. The dependency order is already authoritative and machine-readable: the extract is",
            "     emitted in dependency order computed over the SCC CONDENSATION (Tarjan), so a",
            "     mutually recursive group shares one level instead of inflating — bridge 3's plain",
            "     longest-path fixpoint printed \"level 266\" on a mutually recursive pair. Every entry",
            "     carries its original dcc/src/<file>:<line>; crustify-senpass/UNITS.tsv lists all",
            f"     {nunits} in that order with level, SCC, size, home and callees.",
            "  3. prelude.inc is a NAME INVENTORY, not a compilable prelude, so a database built from",
            "     it would drop bodies anyway. It is out_of_scope and NOTHING IN IT IS TO BE PORTED.",
            "",
            "⭐ ENUMERATION METHOD, because the count is load-bearing: definitions were found BY BRACE",
            "MATCHING from each signature's opening paren, comment- and string-aware, never by a",
            "signature regex — a signature regex undercounted bridge 2 by 72%. Headers were scanned",
            "too: on bridge 4 progir.h held 18 of 33 units and a .cpp-only scan would have missed",
            "every one. 1,029 definitions were found; 373 are excluded with a stated reason each in",
            "crustify-senpass/EXCLUSIONS.tsv (263 one-to-three-line field accessors, 97 MLIR pass",
            "factories, 12 LLVM RTTI `classof` hooks, 1 trivial ctor/dtor) and none for being hard.",
            "",
            "This file is the provenance the schedules record and hash. Keep it byte-stable while a",
            "wave is in flight: `crustify translate` rejects a schedule whose oracle config changed",
            "since planning.",
        ] + SCOPE + WHY + AUTHORITY + PORTED_MEANS + ANALYSES + CAPS,
        "campaign_objective": "port",
        "impl_files": [EXTRACT],
        "api_headers": [],
        "out_of_scope": {"paths": ["crustify-senpass/cpp/prelude.inc"], "features": []},
    }


def main():
    units = json.load(open(WORK + "/units.json"))
    nlev = max(u["level"] for u in units)
    os.makedirs(CAMP, exist_ok=True)
    for d in ("crustify", "crustify/wavefront", "crustify/campaigns/senpass"):
        os.makedirs(os.path.join(TREE, d), exist_ok=True)

    crates = build_crates_json(units)
    wf = wavefront_config(len(units), nlev)

    def w(rel, obj):
        p = os.path.join(TREE, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
        return p

    w("crustify/crates.json", crates)
    w("crustify/build.json", BUILD_JSON)
    w("crustify/wavefront/wavefront-config.json", wf)
    # campaign-local copies, as bridge 2 kept
    w("crustify-senpass/crates.json", crates)
    w("crustify-senpass/build.json", BUILD_JSON)
    w("crustify-senpass/scope-config.json", wf)
    print("configs written:")
    for r in ("crustify/crates.json", "crustify/build.json",
              "crustify/wavefront/wavefront-config.json"):
        p = os.path.join(TREE, r)
        print(f"   {os.path.getsize(p):7d} B  {r}   sha256 {sha256_file(p)[:16]}…")
    print(f"crates.json modules: {len(crates['crates']['deeptools']['modules'])}")


if __name__ == "__main__":
    main()
