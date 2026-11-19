#!/usr/bin/env bash
# ⭐ STEP-0 CARD RUNNER for the rope half-swap cells (written by
# `../examples/rope_half_swap_cells.rs`) — runs a cell on the card and checks it EXACT against a
# host model computed FROM THE OP DEFINITION. Test tooling: nothing here ships.
#
# usage: rope_half_swap_cells_run.sh <sweep-root> <tree-with-bundle_run_dev> <log> [cell …]
#   <sweep-root>  the directory of cell dirs (each: group_0/, placements.json, cell.json)
#   <tree>        a warm pod checkout's backends/spyre-hw, with target/release/examples/bundle_run_dev
#                 built from the SAME commit the example writer ran at
#   <log>         the verdict log (one line per cell, appended — resumable: a cell whose verdict
#                 line is already in the log at the same sha AND group_0 md5 is skipped)
#   [cell …]      zero or more cell DIRS to run (default: every cell dir, sorted)
#
# THE BAR, per cell: `out` and `rot` (read back straight from the bundle's own placements —
# the runner resticks and converts, so the read-back is the HOST-ROW-MAJOR [mq, total] image)
# hold EXACTLY
#   rot = rotate_half(x)   per head: rot[h*hd+d] = -x[h*hd+d+hd/2] (d < hd/2), +x[h*hd+d-hd/2]
#   out  = x*cos + rot*sin            (HF rotate_half + the elementwise combine, never our opspec)
# on fp16-exact integer fills (values ≤ 15, cos/sin ≤ 13 — every product and sum an integer
# ≤ 390, exactly representable in fp16 AND in the SEN169 staging, so the compare is EQUALITY).
# Three fills name the element — row in one run, column in the second, a full mix in the third —
# so the first wrong element is reported with its position. `rot` is checked in its own right:
# the half-moves' output is where a wrong gap, stride or sign lands, before the legs smear it.
#
# Both forms stage IDENTICALLY (x, cos, sin AND the baseline's P are placed and bound in every
# cell), so a verdict never depends on what the other form needs.
#
# ⛔ THE FENCE PROTOCOL (the page-gather sweep's ruling): the runner holds /work/.spyre-bake.lock
# for the whole run, so no bake or other sweep interleaves — and ABORT ON FENCE: after every
# compile and every run, the logs are checked for the card's fence signatures
# (`RAS::PCI::BusFence`, `0xa35e`, `PfMSIMonitor`), and the FIRST one stops the sweep with a
# FENCED verdict and a non-zero exit. A fence here is a result, not a line to log past.
set -uo pipefail
[ $# -ge 3 ] || { echo "usage: rope_half_swap_cells_run.sh <sweep-root> <tree-with-bundle_run_dev> <log> [cell …]" >&2; exit 2; }
ROOT=$(cd "$1" && pwd) || exit 80
TREE=$2
LOG=$3
shift 3 || true
# ROOT/TREE/LOG (and the cell list) go in as POSITIONAL args — never spliced into the
# single-quoted body (a splice of only some of them leaves the rest unset inside).
RUNNER_SHA=$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUNNER_MD5=$(md5sum < "$0" 2>/dev/null | cut -c1-32 || echo unknown)
echo "# RUNNER sha=$RUNNER_SHA md5=$RUNNER_MD5 argv=$*" >> "$LOG"
exec flock /work/.spyre-bake.lock bash -c '
set -uo pipefail
ROOT=$1; TREE=$2; LOG=$3; shift 3
CELLS="$*"
export PATH=/work/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
cd "$TREE" || exit 80

# The card NAMED fence signatures — checked after every compile and run; the first one aborts
# the sweep (a bare "fence" matches nothing here: only the signatures the card itself prints).
fenced() {
  grep -iE "BusFence|0xa35e|PfMSIMonitor" "$1" 2>/dev/null | head -1
}

# One cell: stamp sha+md5, fills, compile, three runs, host model, verdict line. A cell is
# SKIPPED only when the log already holds its verdict AT THE SAME sha and group_0 md5 — a
# re-dump or a re-compile changes the stamp and the cell re-runs.
run_cell() {
  local cell=$1 name sha md5 skip; name=$(basename "$cell")
  [ -f "$cell/cell.json" ] || return 0
  if [ ! -d "$cell/group_0" ]; then
    # MISSING-LOWERING: the writer recorded the refusal in cell.json — a fact for the report,
    # never a silent skip. (A halfswap cell cannot reach here: the writer refuses to write a
    # sweep whose new-form cell will not lower, so every MISSING line is a fact about main.)
    python3 - "$cell" >> "$LOG" <<"PY"
import json, os, sys
cell = sys.argv[1]
c = json.load(open(cell + "/cell.json"))
print("VERDICT %s MISSING form=%s mq=%d hd=%d heads=%d refusal=%s sha=%s md5=(no-bundle)"
      % (os.path.basename(cell), c["form"], c["mq"], c["hd"], c["heads"],
         repr(c.get("missing")), c["sha"]))
PY
    return
  fi
  eval "$(python3 - "$cell" "$LOG" <<"PY"
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
name = os.path.basename(cell)
if os.path.exists(log):
    for line in open(log):
        if line.startswith("VERDICT " + name + " ") and "sha=" + str(c["sha"]) in line and "md5=" + md5 in line:
            skip = "True"
print("sha=%s" % c["sha"])
print("md5=%s" % md5)
print("skip=%s" % skip)
PY
)"
  [ "$skip" = "True" ] && { echo "SKIP $name (already verdicted at sha $sha md5 $md5)"; return; }

  # The fills: fp16-exact integers. x names the element — row in one run, column in the second,
  # a per-(row, col) mix in the third; cos/sin are 1 in the first two runs (out = x + rot, the
  # pairing made visible alone) and their own mix in the third. P is the baseline rope-p class,
  # stated from the SAME op definition the host model states (the rope_p_entry law, spelled from
  # the reference rotate_half): P[inn][o] = -1 (o < half, inn = o+half), +1 (o >= half,
  # inn = o-half), 0 otherwise.
  python3 - "$cell" <<"PY"
import json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(f"{cell}/cell.json"))
mq, hd, total = c["mq"], c["hd"], c["total"]
m = np.arange(mq)[:, None]
col = np.arange(total)[None, :]
np.broadcast_to((m % 15) + 1, (mq, total)).astype(np.float16).tofile(f"{cell}/fill-m-x.bin")
np.broadcast_to((col % 15) + 1, (mq, total)).astype(np.float16).tofile(f"{cell}/fill-c-x.bin")
(((m * 7 + col * 3) % 15) + 1).astype(np.float16).tofile(f"{cell}/fill-mix-x.bin")
np.ones((mq, total), dtype=np.float16).tofile(f"{cell}/fill-one-cos.bin")
np.ones((mq, total), dtype=np.float16).tofile(f"{cell}/fill-one-sin.bin")
(((m + col * 5) % 13) + 1).astype(np.float16).tofile(f"{cell}/fill-mix-cos.bin")
(((m * 3 + col) % 13) + 1).astype(np.float16).tofile(f"{cell}/fill-mix-sin.bin")
half = hd // 2
inn = np.arange(hd)[:, None]
o = np.arange(hd)[None, :]
p = np.where(o < half, np.where(inn == o + half, -1.0, 0.0),
             np.where(inn == o - half, 1.0, 0.0)).astype(np.float16)
p.tofile(f"{cell}/p.bin")
PY
  [ $? -eq 0 ] || { echo "VERDICT $name NOT-RUN fill-failed sha=$sha md5=$md5" | tee -a "$LOG"; return; }

  rm -rf "$cell/group_0/spyreCodeDir"
  timeout 1800 /opt/ibm/spyre/deeptools/bin/dxp_standalone --bundle -d "$cell/group_0" -b sentient \
    > "$cell/compile.log" 2>&1
  local crc=$? sig
  sig=$(fenced "$cell/compile.log")
  if [ -n "$sig" ]; then
    echo "VERDICT $name FENCED at-compile sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
    echo "FENCE-ABORT at $name (compile)" >&2
    exit 99
  fi
  [ $crc -ne 0 ] && { echo "VERDICT $name NOT-RUN compile_rc=$crc sha=$sha md5=$md5" | tee -a "$LOG"; return; }

  local x cos sin out p rot run rrc
  eval "$(python3 - "$cell" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("x=%d cos=%d sin=%d out=%d p=%d rot=%s mq=%d hd=%d" % (
    c["x_tid"], c["cos_tid"], c["sin_tid"], c["out_tid"], c["p_tid"], c["rot_synth"], c["mq"], c["hd"]))
PY
)"
  # Three runs, identical staging: x/cos/sin per fill, P always bound (both forms place it —
  # the staging never depends on the form), the stick statements riding the input list (the
  # runner reads them for the OUTPUT restick too).
  for run in m c mix; do
    local cfill=$([ "$run" = mix ] && echo mix || echo one)
    ./target/release/examples/bundle_run_dev "$cell" \
      "t$x=$cell/fill-$run-x.bin" "stick:t$x=$mq" \
      "t$cos=$cell/fill-$cfill-cos.bin" "stick:t$cos=$mq" \
      "t$sin=$cell/fill-$cfill-sin.bin" "stick:t$sin=$mq" \
      "t$p=$cell/p.bin" "stick:t$p=$hd" \
      "stick:t$out=$mq" \
      --out "t$out=$cell/out-$run.bin" \
      --out "synth:$rot@$mq=$cell/rot-$run.bin" > "$cell/run-$run.log" 2>&1
    rrc=$?
    sig=$(fenced "$cell/run-$run.log")
    if [ -n "$sig" ]; then
      echo "VERDICT $name FENCED at-run-$run sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
      echo "FENCE-ABORT at $name (run $run)" >&2
      exit 99
    fi
    [ $rrc -ne 0 ] && { echo "VERDICT $name NOT-RUN run-$run-rc=$rrc sha=$sha md5=$md5" | tee -a "$LOG"; return; }
  done

  python3 - "$cell" "$name" >> "$LOG" <<"PY"
import json, sys
import numpy as np
cell, name = sys.argv[1], sys.argv[2]
c = json.load(open(f"{cell}/cell.json"))
mq, hd, total = c["mq"], c["hd"], c["total"]
n = mq * total
half = hd // 2
stamp = "sha=%s md5=%s" % (c["sha"], c["group_0_md5"])
# ⭐ THE HOST MODEL, FROM THE OP DEFINITION — HF rotate_half + the elementwise combine, never
# our opspec: per head h, position d (< hd/2): rot = -x[h*hd+d+half]; else rot = +x[h*hd+d-half].
cols = np.arange(total)
d = cols % hd
h = cols // hd
src = np.where(d < half, h * hd + d + half, h * hd + d - half)
sign = np.where(d < half, -1.0, 1.0)
def want(run):
    x = np.fromfile(f"{cell}/fill-{run}-x.bin", dtype=np.float16, count=n).reshape(mq, total)
    if run == "mix":
        cos = np.fromfile(f"{cell}/fill-mix-cos.bin", dtype=np.float16, count=n).reshape(mq, total)
        sin = np.fromfile(f"{cell}/fill-mix-sin.bin", dtype=np.float16, count=n).reshape(mq, total)
    else:
        cos = np.ones(n, dtype=np.float16).reshape(mq, total)
        sin = np.ones(n, dtype=np.float16).reshape(mq, total)
    rot = (sign[None, :] * x[:, src]).astype(np.float16)
    out = (x.astype(np.float32) * cos.astype(np.float32)
           + rot.astype(np.float32) * sin.astype(np.float32)).astype(np.float16)
    return rot, out
worst = None
for run in ("m", "c", "mix"):
    wr, wo = want(run)
    gr = np.fromfile(f"{cell}/rot-{run}.bin", dtype=np.float16, count=n)
    go = np.fromfile(f"{cell}/out-{run}.bin", dtype=np.float16, count=n)
    if gr.size < n or go.size < n:
        print(f"VERDICT {name} NOT-RUN short-readback rot={gr.size}/{n} out={go.size}/{n} {stamp}")
        sys.exit()
    gr = gr.reshape(mq, total)
    go = go.reshape(mq, total)
    for tensor, got, w in (("rot", gr, wr), ("out", go, wo)):
        bad = got != w
        if bad.any():
            i = int(bad.nonzero()[0][0])  # first wrong ROW; report its first wrong column too
            row = got[i] != w[i]
            cc = int(row.nonzero()[0][0])
            if worst is None:
                worst = (run, tensor, i, cc, int(got[i, cc]), int(w[i, cc]), int(bad.sum()))
if worst is None:
    print("VERDICT %s PASS form=%s mq=%d hd=%d heads=%d desc=%d %s"
          % (name, c["form"], mq, hd, c["heads"], c["descriptors"], stamp))
else:
    run, tensor, mi, cc, got, w, count = worst
    print("VERDICT %s FIRST-WRONG form=%s mq=%d hd=%d heads=%d desc=%d at=%s[%d,%d] (head %d pos %d) "
          "got=%d want=%d wrong=%d/%d %s"
          % (name, c["form"], mq, hd, c["heads"], c["descriptors"], tensor, mi, cc,
             cc // hd, cc % hd, got, w, count, n, stamp))
PY
}

# ⛔ THE CELL LIST, BUILT ONCE — explicit cells if given, every cell dir sorted otherwise.
if [ -n "$CELLS" ]; then
  cell_list="$CELLS"
else
  [ -d "$ROOT" ] || { echo "ROPE_HALF_SWAP_RUN_INCOMPLETE — not a dir: $ROOT" >&2; exit 1; }
  cell_list=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
  if [ -z "$cell_list" ]; then
    echo "ROPE_HALF_SWAP_RUN_INCOMPLETE — no cell dirs in $ROOT" >&2
    exit 1
  fi
fi
echo "$cell_list" | while read -r c; do run_cell "$c"; done

# ⛔ A VERDICT OVER NOTHING FAILS: no DONE unless every cell run holds a PASS at ITS OWN STAMP.
pass=0; firstwrong=0; notrun=0; fenced=0; missing=0; total=0
for c in $cell_list; do
  total=$((total + 1))
  name=$(basename "$c")
  eval "$(python3 - "$c" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("stamp=sha=%s md5=%s" % (c["sha"], c.get("group_0_md5", "")))
PY
)"
  kind=$(grep "^VERDICT $name " "$LOG" 2>/dev/null | grep -F "$stamp" | tail -1 | awk "{print \$3}")
  case "$kind" in
    PASS) pass=$((pass + 1));;
    FIRST-WRONG) firstwrong=$((firstwrong + 1));;
    NOT-RUN) notrun=$((notrun + 1));;
    FENCED) fenced=$((fenced + 1));;
    MISSING) missing=$((missing + 1));;
  esac
done
echo "PASS=$pass FIRST-WRONG=$firstwrong NOT-RUN=$notrun FENCED=$fenced MISSING=$missing CELLS=$total"
if [ $((pass + missing)) -lt "$total" ]; then
  echo "ROPE_HALF_SWAP_RUN_INCOMPLETE — every cell needs a PASS (or a reported MISSING) at its own stamp" >&2
  exit 1
fi
echo ROPE_HALF_SWAP_RUN_DONE
' _ "$ROOT" "$TREE" "$LOG" "$@"
