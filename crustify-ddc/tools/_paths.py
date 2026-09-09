"""Paths and the SCOPE TABLE shared by the ddc / L3-scheduler campaign tools.

The tools are authored elsewhere and INSTALLED into <worktree>/crustify-ddc/tools/, so nothing
hardcodes the authoring directory: paths derive from this file's own location, with an env override
each. The authority root is read from `authroot.txt` beside this file (or $DDC_AUTH).

Unlike the senpass campaign (one flat directory), this campaign's scope spans several directories
across the TWO stages of `runDdc` (dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41):

    stage 1   L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose).run(sdsc)
              dcg/dcg_fe/scheduler/                                        8,574 lines
    stage 2   ddc::Ddc(dscGlobal, ..).run_v1(sdsc)
              ddc/ + ddc/ddl/ + ddc/ddl/Dialect/ + ddc/transformations/   20,869 lines

So the file list is EXPLICIT (SCOPE below) rather than a glob of one directory, and every file
carries the Rust module and file its units home into. dcg/dcg_be/ and dcg/dcg_fe/pcfg_gen/ are OUT
of scope: the user has ruled the PCFG / data-DSC path off our path.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
CAMP = os.environ.get("DDC_CAMP") or os.path.dirname(_here)
WORK = os.environ.get("DDC_WORK") or os.path.join(CAMP, "work")
TREE = os.environ.get("DDC_TREE") or os.path.dirname(CAMP)


def _authroot():
    p = os.path.join(_here, "authroot.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    raise SystemExit("no authroot.txt beside _paths.py and no $DDC_AUTH")


AUTH_ROOT = os.environ.get("DDC_AUTH") or _authroot()
REV = "a0d29abbed"

OUTDIR = "crates/compiler/deeptools/src/schedule"
CAMPNAME = "ddc"

# -- THE SCOPE TABLE --------------------------------------------------------------------------
# (authority path relative to AUTH_ROOT, scope tag, rust module, rust file stem)
#
# `scope` groups files into the three consolidated C++ translation units, one per sub-campaign
# scope: l3, ddc, ddl.
# `module`/`stem` give the REAL NESTED SUBMODULE the units home into, named after the original
# file: ddc/fold.rs, ddc/transformation.rs, ddl/conversion.rs, l3/dl_ops.rs.
# NOT flat prefixed filenames -- bridge 2's agen_*/tf_*/vc_* layout is what this campaign avoids.
SCOPE = [
    ("dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp", "l3", "l3", "dl_ops"),
    ("dcg/dcg_fe/scheduler/L3DlOpsScheduler.h", "l3", "l3", "dl_ops"),
    ("ddc/ddc.h", "ddc", "ddc", None),
    ("ddc/ddcv1.cpp", "ddc", "ddc", "v1"),
    ("ddc/ddc_fold.cpp", "ddc", "ddc", "fold"),
    ("ddc/ddc_transformation.cpp", "ddc", "ddc", "transformation"),
    ("ddc/ddc_transformation_util.cpp", "ddc", "ddc", "transformation_util"),
    ("ddc/ddc_metadata.h", "ddc", "ddc", "metadata"),
    ("ddc/transformations/automatic_shuffle/shuffle.cpp", "ddc", "ddc", "shuffle"),
    ("ddc/transformations/automatic_shuffle/shuffle.h", "ddc", "ddc", "shuffle"),
    ("ddc/ddl/ddl.cpp", "ddl", "ddl", None),
    ("ddc/ddl/ddl.h", "ddl", "ddl", None),
    ("ddc/ddl/ddl_conversion.cpp", "ddl", "ddl", "conversion"),
    ("ddc/ddl/ddl_conversion.h", "ddl", "ddl", "conversion"),
    ("ddc/ddl/ddl_convert_interface.h", "ddl", "ddl", "convert_interface"),
    ("ddc/ddl/Dialect/DdlOps.cpp", "ddl", "ddl", "ops"),
    ("ddc/ddl/Dialect/DdlOps.hpp", "ddl", "ddl", "ops"),
    # stage 3: the DCG manager. runDcgForDlOpsStandalone (dcg_manager.cpp:449) is the branch
    # SchedulerStages.cpp:53-57 picks whenever dscs_ is non-empty, which is always true for
    # scratchy's input. dcg_be/ and dcg_fe/pcfg_gen/ stay OUT.
    ("dcg/dcg_manager/dcg_manager.cpp", "dcg", "dcg", "manager"),
    ("dcg/dcg_manager/dcg_manager.h", "dcg", "dcg", "manager"),
]

# Standalone main() drivers are NOT in scope: command-line test harnesses around the library, not
# part of the runDdc path. Named so the exclusion is explicit and auditable.
STANDALONE = [
    "ddc/ddc_standalone.cpp",
    "ddc/ddl/ddl_standalone.cpp",
    "ddc/transformations/automatic_shuffle/shuffle_standalone.cpp",
    "dcg/dcg_fe/scheduler/L3DlOpsScheduler_standalone.cpp",
]

# ALREADY PORTED on bridge1-campaign -- DO NOT RE-PORT. Keyed by authority (file, name) so the
# match is by CITATION, not by bare name. All four live in
# crates/compiler/deeptools/src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs and follow the
# `/// Replaces: eNNN_name` convention, so cross-referencing works.
#   e001_checkConstraints            ddc/ddcv1.cpp:792   (a LAMBDA inside exploreAssignDataStages)
#   e002_createDataConnectMetadata   ddc/ddcv1.cpp:3283
#   e041_getStickSizes               dsc/dsc2.cpp:4066   DesignSpaceConfig method -- OUTSIDE this
#   e071_getCumulativeStickSizes     dsc/dsc2.cpp        campaign's file list, so never enumerated;
#                                                       our units CALL these, they do not re-port
#                                                       them.
ALREADY_PORTED = {
    ("ddc/ddcv1.cpp", "checkConstraints"),
    ("ddc/ddcv1.cpp", "createDataConnectMetadata"),
    ("dsc/dsc2.cpp", "getStickSizes"),
    ("dsc/dsc2.cpp", "getCumulativeStickSizes"),
}

# Named exclusions with a stated reason, beyond what the structural rules catch.
NAMED_EXCLUSIONS = {
    ("dcg/dcg_manager/dcg_manager.cpp", "runDcg"):
        "NOT THE BRANCH TAKEN. SchedulerStages.cpp:53-57 picks runDcgForDlOpsStandalone whenever "
        "dscs_ is non-empty, which is always true for scratchy's input; runDcg is the "
        "dataOpdscs_-only PCFG path.",
    ("dcg/dcg_manager/dcg_manager.cpp", "runDcgGeneratePCFG"):
        "The PCFG / data-DSC path is ruled off our path by the user (dcg_fe/pcfg_gen/ is out of "
        "scope).",
}

# Stage 1 of runDdc, recorded as excluded rather than silently absent.
STAGE1_EXCLUSION = (
    "doCoreletSplitSdsc", "sbf",
    "dbo/src/Utils/sdsc_bundle/SdscCoreletSplit.cpp:84", "-", "stage1_not_reached",
    "STAGE 1 of runDdc, and it is SKIPPED for our input: SchedulerStages.cpp:25 returns early "
    "unless numCoreletsPerCore == 2, and scratchy emits numCoreletsUsed_ = 1 in all 313 sampled "
    "SuperDSCs. Recorded here so the skip is a stated decision, not an omission.",
)

SCOPES = ["l3", "ddc", "ddl", "dcg"]


def scope_of(rel):
    for f, sc, mod, stem in SCOPE:
        if f == rel:
            return sc, mod, stem
    raise KeyError(rel)


def extract_rel(scope):
    return "crustify-%s/cpp/%s.cpp" % (CAMPNAME, scope)


def home_of(module, stem):
    if stem is None:
        return "%s/%s/mod.rs" % (OUTDIR, module)
    return "%s/%s/%s.rs" % (OUTDIR, module, stem)


# -- PER-UNIT NOTES, keyed by (authority file, C++ name) --------------------------------------
# These reach the porter in three places: UNITS.tsv, the extract banner, and the home-file anchor.
# They exist because the translator prompt NEVER names a brief -- a fact written only in a brief
# never reaches an agent.
NOTES = {
    ("ddc/ddcv1.cpp", "exploreAssignDataStages"):
        "REUSE, DO NOT RE-IMPLEMENT: this body CONTAINS the `checkConstraints` lambda at "
        "ddc/ddcv1.cpp:792 and its inner `checkConstraintsImpl` at :825, both ALREADY PORTED on "
        "bridge1-campaign as `e001_checkConstraints` in "
        "crates/compiler/deeptools/src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs:439 "
        "(the impl at that file's :518). CALL THOSE. A second copy of the constraint check is the "
        "diverging-implementation failure this campaign's exclusion list exists to prevent.",
    ("ddc/ddcv1.cpp", "allocAllMem"):
        "SUPERSEDES HAND-TRANSCRIBED GUESSWORK: crates/compiler/deeptools/src/reginit.rs (1,569 "
        "lines, on the integ branch) hand-computes placement from ddc/ddcv1.cpp:132-360 -- THIS "
        "function. Port what the authority does, not what reginit.rs guessed; the two will be "
        "reconciled when this campaign and integ meet, and the authority wins.",
    ("ddc/ddcv1.cpp", "minimizeAllocations"):
        "Part of the ddcv1.cpp:30-360 placement span that reginit.rs hand-transcribes. The "
        "authority is this body.",
    ("ddc/ddcv1.cpp", "calculateClStartAddress"):
        "PLACES ADDRESSES. Part of the span reginit.rs hand-transcribes.",
    ("ddc/ddcv1.cpp", "finalizeOps"):
        "Binds register names to allocation start addresses (ddcv1.cpp:3345-3392) -- the "
        "`R + std::to_string(startAddress)` convention our islands already read back.",
    ("dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp", "fillAllocationStartAddrAndOffset"):
        "THIS IS THE `start_address = 0` DEFECT. L3DlOpsScheduler::run calls this as \"Set start "
        "address, offset in allocations\" (L3DlOpsScheduler.cpp:8000). Our emitted views print "
        "start_address = 0 where the reference states a placed base. The effect of this function IS "
        "the port.",
    ("dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp", "allocAllMem"):
        "The L3 scheduler's own allocation commit (distinct from Ddc::allocAllMem at "
        "ddc/ddcv1.cpp:132 -- two different functions with the same name; the unit number "
        "disambiguates).",
}
