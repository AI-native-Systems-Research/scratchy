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
# ELEMENT-INDEX in two runs (row+1, then col+1), so the first wrong element is reported with
# its position. The run stages the parent RAW (no IEEE→SEN conversion) and reads the output
# RAW, so BOTH buffers carry DL16 words (SEN169: 1 sign, 6 exponent, 9 mantissa bits), and the
# decode reads them as what they are: every word decoded as DL16, both images gathered to their
# LOGICAL frames through the stick-major law D = (c/S)·R·S + r·S + c%S, compared in 2-D against
# want = head + tail rounded HALF-UP to 10 significant bits (the measured narrow rounding,
# convert probe 2026-11-07). The verdict splits the wrong count by COLUMN HALF: the c0 half is
# the control (every door addresses it right), the c64 half is the discriminator — the finding
# under seal says MAIN binds the c64 piece's head input at tail[c0]'s base there, so MAIN is
# expected wrong on EVERY c64 lane and no c0 lane (the decode also counts the lanes equal to
# that wrong binding, tail[c0] + tail[c64]), and the fix right on every lane. The verdict
# line carries the dump's own record of the c64 head corner (element 32768 at main, 65536 at the
# fix), so one line states both the emission fact and the numeric outcome. A decode failure is
# a NOT-RUN verdict and a non-zero exit — never a traceback that reads as done.
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
# r·64 + lane), one run naming the ROW (+1), one naming the COLUMN (+1). The words are staged
# RAW, so the card holds each 16-bit pattern as a DL16 word — the decode reads the fill back
# as DL16 too, so want is computed from the values the card actually holds.
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

# ⭐ THE DECODE — the DEVICE format, both images, in 2-D. The run stages the parent RAW and
# reads the output RAW, so both buffers carry DL16 words (SEN169: 1 sign, 6 exponent, 9
# mantissa bits): every word is decoded as DL16, both images are gathered to their LOGICAL
# frames through the stick-major law D = (c/S)·R·S + r·S + c%S (the parent at
# [parent_rows, cols], the read-back at [rows, cols]), and want = head + tail is rounded
# HALF-UP to 10 significant bits — the measured narrow rounding (convert probe 2026-11-07).
# The wrong count is split by column half (c0 control, c64 discriminator), and the lanes equal
# to the binding MAIN is suspected of — tail[c0] + tail[c64] — are counted beside it. The
# verdict line carries the dump record of the c64 head corner, so the emission fact and the
# numeric outcome are one line. Any decode failure is a NOT-RUN verdict and a non-zero exit —
# a traceback never reads as done.
python3 - "$CELL" >> "$LOG" <<"PY"
import hashlib, json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(f"{cell}/cell.json"))
R, C, S = c["parent_rows"], c["cols"], c["stick"]
rows = c["rows"]
T = R - rows

def dl16(bits):
    # SEN169: 1 sign, 6 exponent (bias 31), 9 mantissa bits.
    b = bits.astype(np.int64)
    sign = np.where(b >> 15, -1.0, 1.0)
    e = (b >> 9) & 0x3F
    m = b & 0x1FF
    return (sign * (1.0 + m / 512.0) * np.exp2(e - 31.0) * (e != 0)
            + sign * (m / 512.0) * np.exp2(-30.0) * (e == 0))

def round_dl16(v):
    # HALF-UP to 10 significant bits — the measured narrow rounding.
    out = np.zeros_like(v)
    nz = v != 0
    q = np.exp2(np.floor(np.log2(np.abs(v[nz]))) - 9)
    out[nz] = np.sign(v[nz]) * np.floor(np.abs(v[nz]) / q + 0.5) * q
    return out

# The logical frames, gathered out of the stick-major device images: in a [NR, C] frame the
# word at logical (r, c) sits at (c//S)·NR·S + r·S + c%S.
rr, cc = np.meshgrid(np.arange(rows), np.arange(C), indexing="ij")
out_idx = (cc // S) * rows * S + rr * S + cc % S
pr, pc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
par_idx = (pc // S) * R * S + pr * S + pc % S

bad_total = 0
first = None
md5s = {}
try:
    for name in ("row", "col"):
        pf = np.fromfile(f"{cell}/fill-{name}.bin", dtype=np.uint16)
        of = np.fromfile(f"{cell}/out-{name}.bin", dtype=np.uint16)
        if pf.size < R * C:
            raise ValueError("short-fill %d of %d" % (pf.size, R * C))
        if of.size < rows * C:
            raise ValueError("short-readback %d of %d" % (of.size, rows * C))
        P = dl16(pf)[par_idx]
        g = dl16(of)[out_idx]
        want = round_dl16(P[:rows] + P[T:T + rows])
        # what MAIN is measured to emit on the c64 half: tail[c0] + tail[c64]
        alt = want.copy()
        alt[:, S:] = round_dl16(P[T:T + rows, :S] + P[T:T + rows, S:])
        bad = g != want
        if bad.any() and first is None:
            r0, c0 = [int(v) for v in np.argwhere(bad)[0]]
            first = (name, r0, c0, float(g[r0, c0]), float(want[r0, c0]))
        bad_total += int(bad.sum())
        md5s[name] = hashlib.md5(open(f"{cell}/out-{name}.bin", "rb").read()).hexdigest()
        for half, sl in (("c0", np.s_[:, :S]), ("c64", np.s_[:, S:])):
            b = bad[sl]
            a = g[sl] == alt[sl]
            print("  run-%s %s: wrong=%d/%d equal-to-tail[c0]+tail[c64]=%d/%d"
                  % (name, half, int(b.sum()), b.size, int(a.sum()), a.size))
except Exception as ex:
    print("VERDICT epilogue-add-o1190 NOT-RUN decode-failed %s sha=%s md5=%s"
          % (ex, c["sha"], c.get("group_0_md5", "-")))
    sys.exit(3)
if bad_total == 0:
    print("VERDICT epilogue-add-o1190 PASS head_c64_corner=%d sha=%s md5=%s out-row-md5=%s out-col-md5=%s"
          % (c["head_c64_corner_elems"], c["sha"], c["group_0_md5"], md5s["row"], md5s["col"]))
else:
    name, r0, c0, gv, wv = first
    print("VERDICT epilogue-add-o1190 FIRST-WRONG run-%s (r=%d,c=%d) got=%r want=%r wrong=%d/%d head_c64_corner=%d sha=%s md5=%s out-row-md5=%s out-col-md5=%s"
          % (name, r0, c0, gv, wv, bad_total, 2 * rows * C, c["head_c64_corner_elems"], c["sha"], c["group_0_md5"], md5s["row"], md5s["col"]))
    sys.exit(1)
PY
drc=$?
[ $drc -eq 0 ] || exit $drc
echo SEAL_RUN_DONE
' _ "$CELL" "$TREE" "$LOG"
