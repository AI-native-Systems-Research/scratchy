#!/usr/bin/env python3
"""Consolidate bridge 3's C++ into one translation unit, in dependency order.

Emits, under OUT:
  cpp/bridge3.cpp   -- every in-scope body VERBATIM, each behind a banner giving its entry number
                       and its ORIGINAL <file>:<line>, ordered level 0 first.
  cpp/prelude.inc   -- plain-C++ stand-ins so the unit needs no MLIR/LLVM/dcc header.
  UNITS.tsv         -- entry, level, LoC, authority file:line, extract line range, rust home, callees.
  EXCLUSIONS.tsv    -- every definition in the directory NOT scheduled, with a reason.

⛔ The bodies are sliced with cppscan's brace matcher. `verify_extract3.py` re-derives every end
INDEPENDENTLY and compares LENGTHS -- it must not be made to share this file's slicing.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cppscan

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
OUT = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3")
CONV = "dcc/src/Conversion/SentientToProgIR"

# Every file in the conversion directory that holds a definition. The .hpp files are scanned too,
# so an inline body that does real work is not silently missed -- EXCLUDE decides what is dropped.
CONV_FILES = [
    f"{CONV}/ConstructProgIRHelper.cpp",
    f"{CONV}/LowerSentientHelper.cpp",
    f"{CONV}/SentientToProgIR.cpp",
    f"{CONV}/UniformInstrAndBlock.cpp",
    f"{CONV}/RegDefTracker.cpp",
    f"{CONV}/Utils.cpp",
    f"{CONV}/SentientToProgIR.hpp",
    f"{CONV}/UniformInstrAndBlock.hpp",
    f"{CONV}/RegDefTracker.hpp",
    f"{CONV}/Utils.hpp",
]

# ---------------------------------------------------------------------------- exclusions
# Keyed (file basename, name, head_line) so an overload is excluded individually and a drifting
# line number turns into a LOUD "exclusion did not match" rather than a silent inclusion.
EXCLUDE = {
    ("SentientToProgIR.cpp", "initializeOutputStream", 421):
        "ostream plumbing (copyfmt/rdbuf); this crate emits and never configures a C++ stream",
    ("SentientToProgIR.cpp", "createSentientToProgIRPass", 746):
        "MLIR pass factory -- #[forward] runs the pipeline at expansion, there is no pass manager",
    ("SentientToProgIR.cpp", "createSentientToProgIRPass", 759):
        "MLIR pass-registry factory over a dummy context; same reason",
    ("SentientToProgIR.hpp", "dccExtContext", 107):
        "1-line field accessor -- a struct field in Rust, not a function",
    ("SentientToProgIR.hpp", "GetAddressScale", 438):
        "3-line overload that stringifies its operand and forwards to the .cpp GetAddressScale (e"
        "ntry kept)",
    ("RegDefTracker.hpp", "regDefChecking", 25):
        "`return false;` -- a compile-time debug constant, a `const bool` in Rust",
    ("RegDefTracker.hpp", "enabled", 48): "1-line accessor over two flags",
    ("RegDefTracker.hpp", "dccExtContext", 49): "1-line field accessor",
    ("RegDefTracker.hpp", "progStateInfo", 50): "1-line field accessor",
    ("RegDefTracker.hpp", "regs", 69):
        "context-chain lookup guarded by DT_CHECK_MSG -- the operand-reaching mechanism the brief "
        "allows dropping, and a runtime abort this crate forbids",
    ("RegDefTracker.hpp", "regs", 73): "the const overload of the same",
    ("UniformInstrAndBlock.hpp", "setUniformizedComment", 124):
        "comment bookkeeping; setComment drops the text unless enable_debug_flag (progir.h:322-330)"
        " so a comment is never load-bearing",
    ("UniformInstrAndBlock.hpp", "setCommonComment", 128): "the same, for the common comment",
}
# every remaining .hpp inline of 3 lines or fewer is an accessor -- excluded by rule, not by name
ACCESSOR_MAX_LINES = 3
KEEP_HPP = {("UniformInstrAndBlock.hpp", "getUnitRegionIndex", 208)}

# ---------------------------------------------------------------------------- rust homes
# Assigned by (file, name pattern) so the mapping is mechanical and auditable, not per-unit taste.
# ⭐ REAL NESTED SUBMODULES, NOT `construct_*`/`lower_*` PREFIXED FLAT FILENAMES. Bridge 2's homes
# are flat `agen_*`/`tf_*`/`vc_*` files and that is on the record as the thing to not repeat.
HOME_RULES = [
    # ConstructProgIRHelper.cpp -- 46 units, one file per instruction family
    ("ConstructProgIRHelper.cpp", r"^(setSentientCompute|ConstructFMA|ConstructBinary|"
     r"ConstructUnary|ConstructTernary)", "construct/compute.rs"),
    ("ConstructProgIRHelper.cpp", r"^(normalizeBurstSize|ConstructL3Load|ConstructL3Store|"
     r"ConstructZRAssign|ConstructL3LoadAndStore|ConstructLoadInstr|ConstructLoadCompute|"
     r"ConstructStore|ConstructLRFCopy)", "construct/transfer.rs"),
    ("ConstructProgIRHelper.cpp", r"^(ConstructSetDstMask|ConstructSetDest|ConstructImmCopy|"
     r"ConstructSplat|ConstructSetMask|ConstructIncrMask|ConstructSAMV)",
     "construct/mask_and_splat.rs"),
    ("ConstructProgIRHelper.cpp", r"^(rtrim|ConstructOpaque)", "construct/opaque.rs"),
    ("ConstructProgIRHelper.cpp", r"^(fillImmField|addToRegsToInit|addPESFPLRF)",
     "construct/reg_init.rs"),
    ("ConstructProgIRHelper.cpp", r".", "construct/scalar.rs"),
    # LowerSentientHelper.cpp -- 34 units
    ("LowerSentientHelper.cpp", r"^(LowerLoadAndSend|LowerReceiveAndStore|LowerLoadAndStore|"
     r"LowerLoadAndExtractScalar|LowerReceiveAndExtractScalar|LowerLoadComputeAndSend|"
     r"LowerCopy)", "lower/transfer.rs"),
    ("LowerSentientHelper.cpp", r"^(LowerCommonOperations|LowerBinary|LowerUnary|LowerTernary|"
     r"LowerMAC|LowerSub|LowerAdd|LowerSplat|LowerOpaque|LowerSAMV|LowerSetMask|LowerIncrMask)",
     "lower/compute.rs"),
    ("LowerSentientHelper.cpp", r"^(LowerFor|LowerYield|LowerReturn|LowerSync|LowerNOP|"
     r"LowerSetSendDestination|LowerUniform|fillUnitToIdMap)", "lower/control.rs"),
    ("LowerSentientHelper.cpp", r".", "lower/labels_and_regs.rs"),
    # the rest, one home per file
    ("SentientToProgIR.cpp", r".", "driver.rs"),
    ("UniformInstrAndBlock.cpp", r"^(createJmpInstr|createNOPInstr|OperandMap|addEntryToOperandMap|"
     r"getUniformizedInstr|getRegularInstr|getCommonField|setCommonField|hasCommonField)",
     "uniform/instr.rs"),
    ("UniformInstrAndBlock.cpp", r".", "uniform/block.rs"),
    ("UniformInstrAndBlock.hpp", r".", "uniform/block.rs"),
    ("RegDefTracker.cpp", r".", "reg_def_tracker.rs"),
    ("Utils.cpp", r".", "utils.rs"),
]


def home_of(fn, name):
    for f, pat, home in HOME_RULES:
        if f == fn and re.search(pat, name):
            return home
    return "unsorted.rs"


# ---------------------------------------------------------------------------- collect
def collect():
    units, dropped = [], []
    matched_excludes = set()
    for rel in CONV_FILES:
        p = SRC / rel
        if not p.exists():
            print(f"  !! missing {rel}", file=sys.stderr)
            continue
        fn = pathlib.Path(rel).name
        text = p.read_text(errors="ignore")
        for d in cppscan.find_defs(text):
            key = (fn, d["name"], d["head_line"])
            d["file"] = rel
            if key in EXCLUDE:
                matched_excludes.add(key)
                dropped.append((d, EXCLUDE[key]))
                continue
            if fn.endswith(".hpp") and key not in KEEP_HPP:
                if d["decl_lines"] <= ACCESSOR_MAX_LINES:
                    dropped.append((d, f"{d['decl_lines']}-line header inline over a member -- "
                                       "a struct field in Rust, not a function"))
                    continue
            units.append(d)
    unmatched = set(EXCLUDE) - matched_excludes
    if unmatched:
        print("  !! EXCLUSION DID NOT MATCH (line drift?):", file=sys.stderr)
        for k in sorted(unmatched):
            print(f"     {k}", file=sys.stderr)
        sys.exit(2)
    return units, dropped


def levels(units):
    """Longest-path level over intra-span call edges, by NAME, with cycles collapsed.

    ⚠️ Overloads share a name, so a level is the max over the overload set. That is conservative
    (never schedules a caller before any callee) and is what bridges 1 and 2 did.

    ⛔⛔ THIS SPAN IS MUTUALLY RECURSIVE and a plain longest-path fixpoint RAN AWAY on it -- the
    first run printed "level 266..269", which is a cycle counting itself, not a depth. The lowering
    is recursive by construction: a `for` body holds common operations, and a uniform region holds
    the same operations a regular one does. So the levels are computed over the CONDENSATION: each
    strongly connected component gets one level, and every unit in it gets that level. A cycle
    cannot be split across a wave barrier, and pretending otherwise is what produced 266.
    """
    by_name = {}
    for u in units:
        by_name.setdefault(u["name"], []).append(u)
    for u in units:
        called = cppscan.calls_in(u["body"])
        u["callees"] = sorted(c for c in called if c in by_name and c != u["name"])

    # edges name -> set(callee names), unioned over overloads
    edges = {n: set() for n in by_name}
    for u in units:
        edges[u["name"]].update(u["callees"])

    # Tarjan SCC, iterative
    index, low, on_stack, stack, comp = {}, {}, set(), [], {}
    counter = [0]
    ncomp = [0]
    for root in edges:
        if root in index:
            continue
        work = [(root, iter(sorted(edges[root])))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(sorted(edges[w]))))
                    advanced = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                cid = ncomp[0]
                ncomp[0] += 1
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp[w] = cid
                    if w == v:
                        break

    cedges = {c: set() for c in set(comp.values())}
    for n, outs in edges.items():
        for w in outs:
            if comp[w] != comp[n]:
                cedges[comp[n]].add(comp[w])
    clvl = {c: 0 for c in cedges}
    for _ in range(len(cedges) + 2):
        changed = False
        for c, outs in cedges.items():
            want = max([clvl[d] + 1 for d in outs] or [0])
            if want > clvl[c]:
                clvl[c] = want
                changed = True
        if not changed:
            break

    cyclic = {c for c in cedges if sum(1 for n in comp if comp[n] == c) > 1}
    cyclic |= {comp[n] for n in edges if n in edges[n]}
    for u in units:
        u["level"] = clvl[comp[u["name"]]]
        u["scc"] = comp[u["name"]]
        u["in_cycle"] = comp[u["name"]] in cyclic
    if cyclic:
        print("  mutual recursion (one level per component, cannot be split by a wave barrier):")
        for c in sorted(cyclic, key=lambda c: clvl[c]):
            members = sorted(n for n in comp if comp[n] == c)
            print(f"    level {clvl[c]}: {', '.join(members)}")
    return units


HEADER = """\
// bridge3.cpp -- IBM Spyre deeptools bridge 3: SentientIR -> ProgIR (dcc pass D76).
//
// Every body below is VERBATIM from the authority tree
//   /Users/nickm/git/deeptools-src   (repo_info.txt: deeptools|master|a0d29abbed...)
// and each banner gives that body's ORIGINAL <file>:<line>. PORT FROM THE AUTHORITY FILE AT THE
// CITED LINE; this unit tells you WHICH function and in WHAT ORDER.
//
// {n} units, {loc} declaration lines, from dcc/src/Conversion/SentientToProgIR/.
// Member definitions are rewritten as free functions (Class::foo -> eNNN_foo) so the unit needs no
// class declarations; the BODIES are untouched.
//
// Levels are the longest path over intra-span call edges: level 0 calls nothing else here, level N
// only levels below it.
#include "prelude.inc"
"""


def main():
    units, dropped = collect()
    units = levels(units)
    units.sort(key=lambda u: (u["level"], u["file"], u["head_line"]))
    for n, u in enumerate(units, 1):
        u["entry"] = n
        # ⛔ `~Dtor` IS NOT AN IDENTIFIER. `e070_~UniformRegionContext` is neither legal C++ nor a
        # match for the campaign's own anchor regex `e[0-9]{3}_[A-Za-z0-9_]+`, so the driver's
        # anchor count and regen-remainders.py would both silently skip that unit.
        u["sym"] = f"e{n:03d}_" + u["name"].replace("~", "dtor_")

    total_loc = sum(u["decl_lines"] for u in units)
    lines = HEADER.format(n=len(units), loc=total_loc).split("\n")
    cur = None
    for u in units:
        if u["level"] != cur:
            cur = u["level"]
            lines += ["", "// " + "=" * 96, f"// LEVEL {cur}", "// " + "=" * 96, ""]
        lines.append(f"// ---- {u['entry']}/{len(units)}  {u['name']}  --  "
                     f"{u['file']}:{u['head_line']}  ({u['decl_lines']}L)")
        head = u["head"]
        if u["qual"]:
            # `Class::` may carry the scanner's doubled `::`; rebuild from the raw head instead.
            head = re.sub(r"[A-Za-z_][A-Za-z_0-9:]*::" + re.escape(u["name"]) + r"\b",
                          u["sym"], head, count=1)
        else:
            head = re.sub(r"\b" + re.escape(u["name"]) + r"\b", u["sym"], head, count=1)
        # VERBATIM: the rewritten head, then the body from its `{` onward, untouched. Anything
        # sharing the `{`'s line -- a trailing comment, or real code -- must survive.
        start = len(lines) + 1
        lines.extend((head.rstrip() + " " + u["body"]).split("\n"))
        u["extract_start"], u["extract_end"] = start, len(lines)
        lines.append("")

    (OUT / "cpp").mkdir(parents=True, exist_ok=True)
    (OUT / "cpp/bridge3.cpp").write_text("\n".join(lines) + "\n")

    tsv = ["\t".join(["unit", "entry", "level", "loc",
                      "authority (under /Users/nickm/git/deeptools-src)",
                      "extract_lines (crustify-bridge3/cpp/bridge3.cpp)", "rust_home", "calls"])]
    for u in units:
        tsv.append("\t".join([
            u["sym"], f"{u['entry']:03d}/{len(units)}", str(u["level"]), str(u["decl_lines"]),
            f"{u['file']}:{u['head_line']}",
            f"{u['extract_start']}-{u['extract_end']}",
            f"src/bridges/sentient_to_progir/{home_of(pathlib.Path(u['file']).name, u['name'])}",
            ",".join(u["callees"]) or "-",
        ]))
    (OUT / "UNITS.tsv").write_text("\n".join(tsv) + "\n")

    ex = ["\t".join(["definition", "authority", "loc", "reason"])]
    for d, why in sorted(dropped, key=lambda x: (x[0]["file"], x[0]["head_line"])):
        ex.append("\t".join([f"{d['qual']}{d['name']}" if d["qual"] else d["name"],
                             f"{d['file']}:{d['head_line']}", str(d["decl_lines"]), why]))
    (OUT / "EXCLUSIONS.tsv").write_text("\n".join(ex) + "\n")

    nl, homes = {}, {}
    for u in units:
        nl.setdefault(u["level"], []).append(u)
        h = home_of(pathlib.Path(u["file"]).name, u["name"])
        homes[h] = homes.get(h, 0) + 1
    print(f"{len(units)} units, {total_loc} declaration lines; {len(dropped)} excluded")
    print(f"bridge3.cpp: {len(lines)} lines")
    for k in sorted(nl):
        print(f"  level {k:2d}: {len(nl[k]):3d} units, {sum(u['decl_lines'] for u in nl[k])}L")
    print("  rust homes:")
    for k, v in sorted(homes.items()):
        print(f"    {k:34s} {v}")
    return units


if __name__ == "__main__":
    main()
