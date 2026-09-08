#!/usr/bin/env python3
"""End-to-end consistency of the generated campaign. Any FAIL line means do not launch."""
import json
import pathlib
import re

OUT = pathlib.Path("/tmp/bridge1-setup")
STAGES = ["levels0-1-leaves-and-helpers", "levels2-4-transfer-and-compute",
          "levels5-10-statements-and-drivers"]
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


rows = [l.split("\t") for l in (OUT / "UNITS.tsv").read_text().splitlines()[1:]]
units = {r[0]: {"level": int(r[1]), "loc": int(r[2]), "auth": r[3], "home": r[5],
                "callees": [c for c in r[6].split(",") if c]} for r in rows}
check(len(rows) == 110, f"UNITS.tsv has 110 rows (got {len(rows)})")
check(len(units) == 110, f"all 110 unit names are distinct (got {len(units)})")

# banners in the extract
ex = (OUT / "cpp/bridge1.cpp").read_text()
banners = re.findall(r"^// ---- (\d+)/(\d+)\s+(\S+)\s+--\s+(\S+):(\d+)\s+\((\d+)L\)", ex, re.M)
check(len(banners) == 110, f"extract carries 110 banners (got {len(banners)})")
check(all(int(b[1]) == 110 for b in banners), "every banner's total is 110")
bn = {f"e{int(b[0]):03d}_{b[2]}" for b in banners}
check(bn == set(units), "banner names == UNITS.tsv names")

# dependency order: a callee must sit at a strictly lower level
bad = []
for name, u in units.items():
    for c in u["callees"]:
        for other, o in units.items():
            if other.split("_", 1)[1] == c and o["level"] >= u["level"]:
                bad.append((name, c, u["level"], o["level"]))
check(not bad, f"every callee is at a strictly lower level ({len(bad)} violations)")
for b in bad[:8]:
    print("        ", b)

# schedules
sched_units, review_units = set(), set()
tot_p = tot_r = 0
for st in STAGES:
    d = OUT / "campaigns/bridge1" / st
    wf = (d / "wavefront-config.json").read_text()
    import hashlib
    sha = hashlib.sha256(wf.encode()).hexdigest()
    for kind, acc in (("port.json", sched_units), ("review.json", review_units)):
        s = json.load(open(d / kind))
        got = [i["name"] for w in s["waves"] for b in w["batches"] for i in b["items"]]
        acc.update(got)
        check(s["summary"]["unit_count"] == len(got),
              f"{st}/{kind} summary unit_count matches its items ({len(got)})")
        check(s["oracle_config"]["sha256"] == sha,
              f"{st}/{kind} oracle sha256 matches wavefront-config.json")
        check(all(len(b["items"]) <= s["budgets"]["max_syms"]
                  for w in s["waves"] for b in w["batches"]),
              f"{st}/{kind} every batch is within max_syms={s['budgets']['max_syms']}")
        # PORT needs a wave barrier between levels so producers land first. REVIEW does not.
        if kind == "port.json":
            for w in s["waves"]:
                lv = {i["layer"] for b in w["batches"] for i in b["items"]}
                check(len(lv) == 1,
                      f"{st}/{kind} each wave holds exactly one level (saw {sorted(lv)})")
        else:
            check(len(s["waves"]) == 1,
                  f"{st}/{kind} is one wave (review needs no level barrier)")
        if kind == "port.json":
            tot_p += s["summary"]["batch_count"]
        else:
            tot_r += s["summary"]["batch_count"]

check(sched_units == set(units), f"port schedules cover all 110 units (got {len(sched_units)})")
check(review_units == set(units), f"review schedules cover all 110 units (got {len(review_units)})")

# rust homes
homes = OUT / "rust/superdsc_to_dataflow_ir"
anchors = set()
for f in homes.glob("*.rs"):
    if f.name == "mod.rs":
        continue
    anchors |= set(re.findall(r"crustify:todo: (e\d{3}_[A-Za-z0-9_]+)", f.read_text()))
check(anchors == set(units), f"todo anchors cover all 110 units (got {len(anchors)})")
declared = set(re.findall(r"pub mod (\w+);", (homes / "mod.rs").read_text()))
files = {f.stem for f in homes.glob("*.rs")} - {"mod"}
check(declared == files, f"mod.rs declares exactly the home files ({declared ^ files or 'match'})")
check({u["home"] for u in units.values()} == {f + ".rs" for f in files},
      "every UNITS.tsv home exists as a file")

print(f"\n  port batches: {tot_p}   review batches: {tot_r}")
print(f"\n{'CAMPAIGN CONSISTENT — ready to install' if not fails else str(len(fails)) + ' FAILURES — DO NOT LAUNCH'}")
