#!/usr/bin/env bash
# ⭐ THE PAGE-GATHER CARD SWEEP RUNNER AT THE ONE `index_select` GATHER — runs every cell the dump
# test wrote (`zz_page_gather_sweep.rs`, this directory) on the card and checks it FRAME-EXACT
# against a host model computed FROM THE OP DEFINITION. Test tooling: nothing here ships.
#
# usage: page_gather_sweep_run.sh <sweep-root> <tree-with-bundle_run_dev> <log>
#   <sweep-root>  the directory of cell dirs (each: group_0/, placements.json, cell.json)
#   <tree>        a warm pod checkout with target/release/examples/bundle_run_dev built from
#                 the SAME commit the dump test ran at
#   <log>         the verdict log (one line per cell, appended — the run is resumable: a cell
#                 whose verdict line is already in the log is skipped)
#
# THE BAR, per cell: the read-back buffer holds the card image of
#   pool.index_select(0, ids).reshape(<the frame dims>)
# — `index_select` and `reshape` exactly as the reference states them (the op's definition, never
# our opspec) — AT THE FRAME OWN RANK, per torch-spyre's layout law:
# * rank-2 [T, w]: the one stick-column law (tensors_and_layouts.md:178-189; views.py:674-681) —
#   a stick is 128 contiguous bytes (64 fp16), a device layout is row-major with the stick dim
#   last, so a host [T, w] sits on the device as [w/64, T, 64]:
#     D = (c/64)·(T·64) + t·64 + (c%64).
#   A restickify cell reads the one whole stick permute's destination — the frame's own stick
#   column — and the same law decodes it (the permute's contract IS that law).
# * rank-3 [E, R, C] / rank-4 [E, B, R, C]: the indirectly-indexed entry axis sits at device
#   position 0 — a HARDWARE requirement, not a hint (torch-spyre _inductor/nonstick_dim_order.py:
#   "The indirectly-indexed dimension of a gather value tensor must sit at device position 0") —
#   and each [R, C] block behind it carries the rank-2 law, the blocks one after another:
#     D = e·(B·R·C) + blk·(R·C) + (c/64)·(R·64) + r·64 + (c%64).
#   FOLDING E·R into one row axis (the rank-2 law over [E·R, C]) disagrees with this at E > 1
#   and C > 64 — the decoder must not fold; the emitter's per-block landing is the lawful one.
#
# The pool is staged as its own device image (row-outermost: row-major pages, `raw:` — the fills
# are IEEE fp16 bits moved verbatim, no codec), the landing read back verbatim (`raw:`), and three
# fills name the source element — (page, row) in two runs, col in the third — so the first wrong
# element is reported with its position. The verdicts: PASS, FIRST-WRONG (with the element),
# NOT-RUN (compile or run failure); MISSING-LOWERING is decided by the dump test, not here.
#
# ⛔ THE FENCE PROTOCOL (the ruling for R>1 cells): one cell alone — the runner holds
# /work/.spyre-bake.lock for the whole run, so no bake or other sweep interleaves — and ABORT ON
# FENCE: after every compile and every run, the logs are checked for the card's fence signatures
# (`RAS::PCI::BusFence`, `0xa35e`, `PfMSIMonitor`, senlib's abort), and the FIRST one stops the
# sweep with a FENCED verdict and a non-zero exit. The r32-c128-e2-b0 fence of the main sweep is
# unexplained; a fence here is a result, not a line to log past.
set -uo pipefail
ROOT=$1; TREE=$2; LOG=$3
exec flock /work/.spyre-bake.lock bash -c '
export PATH=/work/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
cd '"$TREE"' || exit 80

# The card NAMED fence signatures — checked after every compile and run; the first one aborts
# the sweep (a bare "fence" matches nothing here: only the signatures the card itself prints).
fenced() {
  grep -iE "BusFence|0xa35e|PfMSIMonitor" "$1" 2>/dev/null | head -1
}

# One cell: stamp sha+md5, fills, compile, three runs, decode, verdict line. A cell is SKIPPED
# only when the log already holds its verdict AT THE SAME sha and group_0 md5 — a re-dump or a
# re-compile changes the stamp and the cell re-runs.
run_cell() {
  local cell=$1 name sha md5 skip; name=$(basename "$cell")
  [ -f "$cell/cell.json" ] || return 0
  [ -d "$cell/group_0" ] || return 0   # MISSING-LOWERING: nothing to run
  # The stamp: the sha the dump test wrote into cell.json, and an md5 over the dumped group_0
  # (the dxp input — bundle.mlir and the sdsc jsons; a compile spyreCodeDir is excluded).
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

  # The fills: the pool device image (row-outermost, row-major pages), one plane per run -
  # (page+1), (row+1), (col+1), all fp16-exact - plus the index tensor words.
  python3 - "$cell" <<"PY"
import json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(f"{cell}/cell.json"))
pages, slab, cols = c["pages"], c["slab"], c["land_cols"]
p = np.arange(pages)[:, None] + 1
r = (np.arange(slab) // cols)[None, :] + 1
cc = (np.arange(slab) % cols)[None, :] + 1
one = lambda v: np.broadcast_to(v, (pages, slab)).astype(np.float16)
one(p).tofile(f"{cell}/fill-pr-e.bin")
one(r).tofile(f"{cell}/fill-pr-f.bin")
one(cc).tofile(f"{cell}/fill-c.bin")
words = np.zeros(c["index_bytes"] // 4, dtype=np.int32)
words[: len(c["index"])] = c["index"]
words.tofile(f"{cell}/index.bin")
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

  local fill rrc pool idx rb
  eval "$(python3 - "$cell" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("pool=%d idx=%d rb=%d" % (c["pool_tid"], c["idx_tid"], c["readback_tid"]))
PY
)"
  for fill in pr-e pr-f c; do
    ./target/release/examples/bundle_run_dev "$cell" \
      "t$pool=$cell/fill-$fill.bin" "raw:t$pool" \
      "t$idx=$cell/index.bin" "raw:t$idx" \
      --out "raw:t$rb=$cell/out-$fill.bin" > "$cell/run-$fill.log" 2>&1
    rrc=$?
    sig=$(fenced "$cell/run-$fill.log")
    if [ -n "$sig" ]; then
      echo "VERDICT $name FENCED at-run-$fill sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
      echo "FENCE-ABORT at $name (run $fill)" >&2
      exit 99
    fi
    [ $rrc -ne 0 ] && { echo "VERDICT $name NOT-RUN run-$fill-rc=$rrc sha=$sha md5=$md5" | tee -a "$LOG"; return; }
  done

  python3 - "$cell" "$name" >> "$LOG" <<"PY"
import json, sys
import numpy as np
cell, name = sys.argv[1], sys.argv[2]
c = json.load(open(f"{cell}/cell.json"))
E, slab = c["entries"], c["slab"]
dims = c["frame"]
idx = np.array(c["index"])
n = E * slab
pr_e = np.fromfile(f"{cell}/out-pr-e.bin", dtype=np.float16, count=n)
pr_f = np.fromfile(f"{cell}/out-pr-f.bin", dtype=np.float16, count=n)
c_e = np.fromfile(f"{cell}/out-c.bin", dtype=np.float16, count=n)
if pr_e.size < n or pr_f.size < n or c_e.size < n:
    print(f"VERDICT {name} NOT-RUN short-readback {pr_e.size}/{pr_f.size}/{c_e.size} of {n} sha={c["sha"]} md5={c["group_0_md5"]}")
    sys.exit()
# ⭐ THE HOST MODEL, FROM THE OP DEFINITION — never from our opspec:
#   v = pool.index_select(0, ids).reshape(<the frame dims>)   (the tape own reshape)
# and the read-back holds v in the card frame AT THE FRAME OWN RANK:
#   rank-2 [T, w]:  D = (c//64)·(T·64) + t·64 + (c%64)
#     (torch-spyre tensors_and_layouts.md:178-189 — the one stick-column law)
#   rank-3 [E, R, C] / rank-4 [E, B, R, C]: the entry axis — the indirectly-indexed
#   dim — sits at device position 0, a hardware requirement (torch-spyre
#   nonstick_dim_order.py), and each [R, C] block behind it carries the rank-2 law,
#   blocks one after another:
#     D = e·(B·R·C) + blk·(R·C) + (c//64)·(R·64) + r·64 + (c%64)
# The decoder maps each device offset D back to the source element it must hold.
D = np.arange(n)
if len(dims) == 2:
    T, C = dims
    stick, rem = np.divmod(D, T * 64)
    t, lane = np.divmod(rem, 64)
    col = stick * 64 + lane
    f = t.astype(np.int64) * C + col          # the frame flat index (reshape row-major law)
    e, o = np.divmod(f, slab)                 # index_select keeps the entry axis whole
else:
    if len(dims) == 3:
        E_, R, C = dims
        B = 1
    else:
        E_, B, R, C = dims
    e, o1 = np.divmod(D, B * R * C)           # the entry axis is device dim 0
    blk, o = np.divmod(o1, R * C)             # then the blocks, one after another
    stick, rem = np.divmod(o, R * 64)         # then the rank-2 law on this block
    r, lane = np.divmod(rem, 64)
    col = stick * 64 + lane
    o = blk * R * C + r * C + col             # the source offset inside the entry page
want = (
    (idx[e] + 1).astype(np.float16),        # the page the entry gathered
    (o // C + 1).astype(np.float16),        # the source row (the pool [slab] read C wide)
    (o % C + 1).astype(np.float16),         # the source col
)
got = (pr_e, pr_f, c_e)
bad = (got[0] != want[0]) | (got[1] != want[1]) | (got[2] != want[2])
if not bad.any():
    print("VERDICT %s PASS %s sha=%s md5=%s" % (name, c["form"], c["sha"], c["group_0_md5"]))
else:
    i = int(bad.nonzero()[0][0])
    print(
        "VERDICT %s FIRST-WRONG %s D=%d (e,o)=(%d,%d) "
        "got=(%.0f,%.0f,%.0f) want=(%.0f,%.0f,%.0f) wrong=%d/%d sha=%s md5=%s"
        % (name, c["form"], i, e[i], o[i],
           got[0][i], got[1][i], got[2][i],
           want[0][i], want[1][i], want[2][i], int(bad.sum()), n,
           c["sha"], c["group_0_md5"])
    )
PY
}

find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r c; do run_cell "$c"; done
echo PAGE_GATHER_SWEEP_RUN_DONE
'
