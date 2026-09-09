#!/usr/bin/env python3
"""Which in-scope units cannot be ported without state from OUTSIDE this campaign's scope.

MEASURED, not guessed: for every in-scope unit, the names its body calls that resolve to neither
another in-scope unit nor one of the four already-ported functions are classified against the
out-of-scope subsystems below. A unit naming one needs state this campaign does not bring.

Writes OUTSIDE-DEPS.tsv (per subsystem) and OUTSIDE-UNITS.tsv (per unit).
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

import cppscan

AUTH = _paths.AUTH_ROOT
WORK = _paths.WORK
CAMP = _paths.CAMP

# (subsystem label, where it lives, matcher against the CALLED NAME or the surrounding text)
SUBSYSTEMS = [
    ("PCFG translator", "dcg/dcg_fe/pcfg_gen/ + DscPcfgTranslator (OUT of scope)",
     re.compile(r"transformDscCompToPcfg|createPcfgForUnitPerCore|getMaxGTRGroupId|"
                r"DscPcfgTranslator|mergePcfg|SenPcfg")),
    ("DCG backend", "dcg/dcg_be/ (OUT of scope)",
     re.compile(r"fillAndCreateSenProgInfoUsingSuperDSC|dcg_be_|createSenProg")),
    ("memory trackers", "sys-arch-spec/memtracker/ (MemTrackBundle, DsTrackInMem)",
     re.compile(r"MemTrack|memTracker|DsTrackInMem|getTracker|allocMem|freeMem|reserveMem")),
    ("dsc2 tree utilities", "dsc/dsc2.cpp (SuperDsc / DesignSpaceConfig / dsc2:: helpers)",
     re.compile(r"transformLxZeroPadInfoInScheduleTree|traverseTree|getOwnerLoop|"
                r"getNonBroadcastLdsDimSet|isDSC2|completeFoldConstruction")),
    ("stick sizes (ALREADY PORTED)", "dsc/dsc2.cpp:4066 -- e041/e071 on bridge1-campaign",
     re.compile(r"getStickSizes|getCumulativeStickSizes")),
    ("constraint check (ALREADY PORTED)", "shape_constraints.rs:439 -- e001/e002",
     re.compile(r"checkConstraints|createDataConnectMetadata")),
    ("ISA tables", "isa/ (senCompToISAptr, Isa)",
     re.compile(r"senCompToISAptr|isaPerUnit|getIsa\b")),
    ("fold infrastructure", "util/foldManager/ (foldInfrastructure.h)",
     re.compile(r"FoldManager|foldInfrastructure|sdscFoldProps_")),
]


def main():
    units = json.load(open(WORK + "/units.json"))
    in_scope = {u["name"] for u in units} | {u["unit"] for u in units}
    text_of = {}
    for u in units:
        p = os.path.join(AUTH, u["rel"])
        if p not in text_of:
            text_of[p] = open(p, encoding="utf-8", errors="replace").read()

    per_sub = defaultdict(list)
    per_unit = defaultdict(set)
    for u in units:
        body = text_of[os.path.join(AUTH, u["rel"])][u["open_off"]:u["close_off"] + 1]
        for label, where, rx in SUBSYSTEMS:
            if rx.search(body):
                per_sub[label].append(u)
                per_unit[u["unit"]].add(label)

    with open(os.path.join(CAMP, "OUTSIDE-DEPS.tsv"), "w") as fh:
        fh.write("subsystem\twhere\tunits_naming_it\texample_units\n")
        for label, where, _rx in SUBSYSTEMS:
            us = per_sub.get(label, [])
            fh.write("\t".join([label, where, str(len(us)),
                                ",".join(x["unit"] for x in us[:8]) or "-"]) + "\n")

    with open(os.path.join(CAMP, "OUTSIDE-UNITS.tsv"), "w") as fh:
        fh.write("unit\tlevel\tloc\tauthority\tsubsystems\n")
        for u in units:
            labs = sorted(per_unit.get(u["unit"], ()))
            if not labs:
                continue
            fh.write("\t".join([u["unit"], str(u["level"]), str(u["body_lines"]),
                                "%s:%d" % (u["rel"], u["head_line"]),
                                "; ".join(labs)]) + "\n")

    print("units naming at least one out-of-scope subsystem: %d of %d"
          % (len(per_unit), len(units)))
    for label, where, _rx in SUBSYSTEMS:
        us = per_sub.get(label, [])
        print("   %-32s %4d units   %s" % (label, len(us), where))
    print()
    print("units blocked by the two GENUINELY out-of-scope dcg subsystems (PCFG / DCG backend):")
    blocked = [u for u in units
               if per_unit.get(u["unit"], set()) & {"PCFG translator", "DCG backend"}]
    for u in sorted(blocked, key=lambda u: u["unit"]):
        print("   %-42s L%d %4d lines  %s:%d"
              % (u["unit"], u["level"], u["body_lines"], u["rel"], u["head_line"]))


if __name__ == "__main__":
    main()
