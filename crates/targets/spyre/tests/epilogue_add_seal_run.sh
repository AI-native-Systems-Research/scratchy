#!/usr/bin/env bash
# ⭐ THE 8b ATTN-EPILOGUE ADD'S NUMERIC SEAL RUNNER — runs the one cell the dump test wrote
# (`zz_the_attn_epilogue_add_seal_dump.rs`, this directory) on the card and checks it
# element-for-element against a host reference computed FROM THE STATED WINDOWS, never from our
# opspec. Test tooling: nothing here ships.
#
# usage: epilogue_add_seal_run.sh <cell-dir> <tree-with-bundle_run_dev> <log>
#   <cell-dir>  the dump test's cell directory (group_0/, placements.json, cell.json)
#   <tree>      a warm pod checkout with target/release/examples/bundle_run_dev built from
#               the SAME commit the dump test ran at
#   <log>       the verdict log (one verdict line, appended — the run is resumable: a verdict
#               already in the log at the same sha and group_0 md5 is skipped)
#
# THE BAR: the read-back buffer holds the card image of
#   out[r][c] = parent[r][c] + parent[512 + r][c]      for r in 0..512, c in 0..128
# — head plus tail, the row-band windows the program states — with the parent filled by
# ELEMENT-INDEX in two runs (row+1, then col+1, both fp16-exact), so the first wrong element is
# reported with its position. The verdict splits the wrong count by COLUMN HALF: the c0 half is
# the control (every door addresses it right), the c64 half is the discriminator — the finding
# under seal says MAIN binds the c64 piece's head input at tail[c0]'s base there, so MAIN is
# expected wrong on EVERY c64 lane and no c0 lane, and the fix right on every lane. The verdict
# line carries the dump's own record of the c64 head corner (element 32768 at main, 65536 at the
# fix), so one line states both the emission fact and the numeric outcome.
#
# ⛔ THE FENCE PROTOCOL (the page-gather sweep's own): one cell alone — the runner holds
# /work/.spyre-bake.lock for the whole run — and ABORT ON FENCE: after the compile and every
# run, the logs are checked for the card's fence signatures (`RAS::PCI::BusFence`, `0xa35e`,
# `PfMSIMonitor`, senlib's abort), and the first one stops the run with a FENCED verdict and a
# non-zero exit.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: epilogue_add_seal_run.sh <cell-dir> <tree-with-bundle_run_dev> <log>" >&2; exit 2; }
CELL=$(cd "$1" && pwd) || exit 80
TREE=$2
LOG=$3
# CELL/TREE/LOG go in as POSITIONAL args — never spliced into the single-quoted body.
RUNNER_SHA=$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUNNER_MD5=$(md5sum < "$0" 2>/dev/null | cut -c1-32 || echo unknown)
echo "# RUNNER sha=$RUNNER_SHA md5=$RUNNER_MD5 argv=$*" >> "$LOG"
exec flock /work/.spyre-bake.lock bash -c '
set -uo pipefail
CELL=$1; TREE=$2; LOG=$3
export PATH=/work/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
cd "$TREE" || exit 80

# The card NAMED fence signatures — checked after the compile and every run.
fenced() {
  grep -iE "BusFence|0xa35e|PfMSIMonitor" "$1" 2>/dev/null | head -1
}

[ -f "$CELL/cell.json" ] || { echo "SEAL_RUN_INCOMPLETE — no cell.json in $CELL" >&2; exit 1; }
[ -d "$CELL/group_0" ] || { echo "VERDICT epilogue-add-o1190 MISSING-LOWERING (see cell.json) sha=$(python3 -c "import json;print(json.load(open(\"$CELL/cell.json\"))[\"sha\"])") md5=-" | tee -a "$LOG"; exit 1; }

# The stamp: the sha the dump test wrote into cell.json, and an md5 over the dumped group_0
# (the dxp input — bundle.mlir and the sdsc jsons; a compiled spyreCodeDir is excluded).
eval "$(python3 - "$CELL" "$LOG" <<"PY"
import hashlib, json, os, sys
cell, log = sys.argv[1], sys.argv[2]
c = json.load(open(cell + "/cell.json"))
h = hashlib.md5()
for root, dirs, files in os.walk(cell + "/group_0"):
    dirs[:] = [d for d in dirs if d != "spyreCodeDir"]
    for f in sorted(files):
        p = os.path.join(root, f)
        h.update(p[len(cell) + 1:].encode())
        h.update(open(p, "rb").read())
md5 = h.hexdigest()
c["group_0_md5"] = md5
json.dump(c, open(cell + "/cell.json", "w"), indent=1)
skip = "False"
if os.path.exists(log):
    for line in open(log):
        if line.startswith("VERDICT epilogue-add-o1190 ") and "sha=" + str(c["sha"]) in line and "md5=" + md5 in line:
            skip = "True"
print("sha=%s" % c["sha"])
print("md5=%s" % md5)
print("head_c64=%s" % c["head_c64_corner_elems"])
print("parent=%d out=%d" % (c["parent_tid"], c["out_tid"]))
print("skip=%s" % skip)
PY
)"
[ "$skip" = "True" ] && { echo "SKIP epilogue-add-o1190 (already verdicted at sha $sha md5 $md5)"; exit 0; }

# The fills: the parent device image (stick-major [parent_rows, cols]: D = stick·(rows·64) +
# r·64 + lane), one run naming the ROW (+1), one naming the COLUMN (+1) — both fp16-exact, so
# every element of the parent is named by the pair and every sum stays exact.
python3 - "$CELL" <<"PY"
import json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(f"{cell}/cell.json"))
R, C, S = c["parent_rows"], c["cols"], c["stick"]
n = R * C
D = np.arange(n)
stick, rem = np.divmod(D, R * S)
r, lane = np.divmod(rem, S)
col = stick * S + lane
(r + 1).astype(np.float16).tofile(f"{cell}/fill-row.bin")
(col + 1).astype(np.float16).tofile(f"{cell}/fill-col.bin")
PY
[ $? -eq 0 ] || { echo "VERDICT epilogue-add-o1190 NOT-RUN fill-failed sha=$sha md5=$md5" | tee -a "$LOG"; exit 1; }

# Compile the dumped bundle, then run it once per fill, staging the parent verbatim (raw:) and
# reading the output back verbatim.
rm -rf "$CELL/group_0/spyreCodeDir"
timeout 1800 /opt/ibm/spyre/deeptools/bin/dxp_standalone --bundle -d "$CELL/group_0" -b sentient \
  > "$CELL/compile.log" 2>&1
crc=$?
sig=$(fenced "$CELL/compile.log")
if [ -n "$sig" ]; then
  echo "VERDICT epilogue-add-o1190 FENCED at-compile sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
  exit 99
fi
[ $crc -ne 0 ] && { echo "VERDICT epilogue-add-o1190 NOT-RUN compile_rc=$crc sha=$sha md5=$md5" | tee -a "$LOG"; exit 1; }

for fill in row col; do
  ./target/release/examples/bundle_run_dev "$CELL" \
    "t$parent=$CELL/fill-$fill.bin" "raw:t$parent" \
    --out "raw:t$out=$CELL/out-$fill.bin" > "$CELL/run-$fill.log" 2>&1
  rrc=$?
  sig=$(fenced "$CELL/run-$fill.log")
  if [ -n "$sig" ]; then
    echo "VERDICT epilogue-add-o1190 FENCED at-run-$fill sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
    exit 99
  fi
  [ $rrc -ne 0 ] && { echo "VERDICT epilogue-add-o1190 NOT-RUN run-$fill-rc=$rrc sha=$sha md5=$md5" | tee -a "$LOG"; exit 1; }
done

# ⭐ THE DECODE — the host reference FROM THE STATED WINDOWS: out[r][c] = parent[r][c] +
# parent[512+r][c], the parent value named by the fill (row+1 / col+1), the read-back decoded
# at the output stick-major [rows, cols] image of its own. The wrong count is split by column
# half: c0 is the control, c64 the discriminator. The verdict line carries the dump record of
# the c64 head corner, so the emission fact and the numeric outcome are one line.
python3 - "$CELL" >> "$LOG" <<"PY"
import hashlib, json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(f"{cell}/cell.json"))
R, C, S = c["parent_rows"], c["cols"], c["stick"]
rows = c["rows"]
n = rows * C
D = np.arange(n)
stick, rem = np.divmod(D, rows * S)
r, lane = np.divmod(rem, S)
col = stick * S + lane
c0 = col < S
c64 = ~c0
fills = {"row": (r + 1).astype(np.float16), "col": (col + 1).astype(np.float16)}
bad_total = 0
first = None
got = {}
for name, pval in fills.items():
    g = np.fromfile(f"{cell}/out-{name}.bin", dtype=np.float16, count=n)
    if g.size < n:
        print("VERDICT epilogue-add-o1190 NOT-RUN short-readback %d of %d sha=%s md5=%s"
              % (g.size, n, c["sha"], c["group_0_md5"]))
        sys.exit()
    got[name] = g
    # want[r][c] = parent[r][c] + parent[R-rows+r][c] — head band plus tail band, the stated windows
    want = (pval + pval[R - rows: R - rows + rows]).astype(np.float16)
    bad = g != want
    if bad.any() and first is None:
        i = int(bad.nonzero()[0][0])
        first = (name, i, int(r[i]), int(col[i]), float(g[i]), float(want[i]))
    bad_total += int(bad.sum())
    print("  run-%s: wrong-c0=%d/%d wrong-c64=%d/%d first-wrong=%s"
          % (name, int((bad & c0).sum()), int(c0.sum()), int((bad & c64).sum()), int(c64.sum()),
             "none" if not bad.any() else "(r=%d,c=%d got=%.0f want=%.0f)" % (r[bad.nonzero()[0][0]], col[bad.nonzero()[0][0]], g[bad.nonzero()[0][0]], want[bad.nonzero()[0][0]])))
md5_row = hashlib.md5(open(f"{cell}/out-row.bin", "rb").read()).hexdigest()
md5_col = hashlib.md5(open(f"{cell}/out-col.bin", "rb").read()).hexdigest()
if bad_total == 0:
    print("VERDICT epilogue-add-o1190 PASS head_c64_corner=%d sha=%s md5=%s out-row-md5=%s out-col-md5=%s"
          % (c["head_c64_corner_elems"], c["sha"], c["group_0_md5"], md5_row, md5_col))
else:
    name, i, rr, cc, gv, wv = first
    print("VERDICT epilogue-add-o1190 FIRST-WRONG run-%s (r=%d,c=%d) got=%.0f want=%.0f wrong=%d/%d head_c64_corner=%d sha=%s md5=%s out-row-md5=%s out-col-md5=%s"
          % (name, rr, cc, gv, wv, bad_total, 2 * n, c["head_c64_corner_elems"], c["sha"], c["group_0_md5"], md5_row, md5_col))
    sys.exit(1)
PY
echo SEAL_RUN_DONE
' _ "$CELL" "$TREE" "$LOG"
