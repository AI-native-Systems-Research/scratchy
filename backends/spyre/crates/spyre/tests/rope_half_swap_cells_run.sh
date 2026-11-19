#!/usr/bin/env bash
# ⭐ STEP-0 CARD RUNNER for the rope half-swap cells (written by
# `../examples/rope_half_swap_cells.rs`) — runs a cell on the card and checks it EXACT against a
# host model computed FROM THE OP DEFINITION. Test tooling: nothing here ships.
#
# usage: rope_half_swap_cells_run.sh [--noise] <sweep-root> <tree-with-bundle_run_dev> <log> [cell …]
#   <sweep-root>  the directory of cell dirs (each: group_0/, placements.json, cell.json)
#   <tree>        a warm pod checkout's backends/spyre-hw, with target/release/examples/bundle_run_dev
#                 built from the SAME commit the example writer ran at
#   <log>         the verdict log (one line per cell, appended — resumable: a cell whose verdict
#                 line is already in the log at the same sha AND group_0 md5 is skipped)
#   [cell …]      zero or more cell DIRS to run (default: every cell dir, sorted)
#
# THE BAR, per cell: `out` and the form's own probe (read back straight from the bundle's own
# placements — the runner resticks and converts, so the read-back is the HOST-ROW-MAJOR
# [mq, total] image) hold EXACTLY
#   rot = rotate_half(x)          per head: rot[h*hd+d] = -x[h*hd+d+hd/2] (d < hd/2), +x[h*hd+d-hd/2]
#   rs  = rotate_half(x) * sin    (halfswap4's own intermediate — it has no rot tensor)
#   out = x*cos + rotate_half(x)*sin   (HF rotate_half + the elementwise combine, never our opspec)
# on fp16-exact integer fills (values ≤ 15, cos/sin/sin' ≤ 13 — every product and sum an integer
# ≤ 390, exactly representable in fp16 AND in the SEN169 staging, so the compare is EQUALITY).
# Three fills name the element — row in one run, column in the second, a full mix in the third —
# so the first wrong element is reported with its position. The probe is checked in its own
# right: the half-ops' output is where a wrong gap, stride or sign lands, before the legs smear
# it.
#
# All three forms place and stage the SAME six tensors (x, cos, sin, out, P and the signed sin'
# table), so a verdict never depends on what the other forms need; only the BINDING set branches
# on the form (halfswap4 alone binds sin', its own op input) and the probe branches per
# `probe_kind` in cell.json.
#
# ⛔ THE FENCE PROTOCOL (the page-gather sweep's ruling): the runner holds /work/.spyre-bake.lock
# for the whole run, so no bake or other sweep interleaves — and ABORT ON FENCE: after every
# compile and every run, the logs are checked for the card's fence signatures
# (`RAS::PCI::BusFence`, `0xa35e`, `PfMSIMonitor`), and the FIRST one stops the sweep with a
# FENCED verdict and a non-zero exit. A fence here is a result, not a line to log past.
set -uo pipefail
# --noise (FIRST arg, before the positionals) switches to the noise-protocol leg: per shape, the
# five ABBA pairs (three A/A controls — both dirs copies of one bundle — and main against each
# new form) run as FRESH PROCESSES, two independent passes each, 26 rounds (52 timed launches per
# side, no refill), and every pair reports n / median / IQR / min per side. The decision rule —
# a delta is REAL only if it exceeds 3x the A/A floor, the IQRs separate, and the sign repeats
# across both passes — is applied at report time over the logged NOISE lines.
MODE=run
if [ "${1:-}" = "--noise" ]; then MODE=noise; shift; fi
[ $# -ge 3 ] || { echo "usage: rope_half_swap_cells_run.sh [--noise] <sweep-root> <tree-with-bundle_run_dev> <log> [cell …]" >&2; exit 2; }
ROOT=$(cd "$1" && pwd) || exit 80
TREE=$2
LOG=$3
shift 3 || true
# ROOT/TREE/LOG (and the cell list) go in as POSITIONAL args — never spliced into the
# single-quoted body (a splice of only some of them leaves the rest unset inside).
RUNNER_SHA=$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUNNER_MD5=$(md5sum < "$0" 2>/dev/null | cut -c1-32 || echo unknown)
echo "# RUNNER mode=$MODE sha=$RUNNER_SHA md5=$RUNNER_MD5 argv=$*" >> "$LOG"
# MODE rides the environment into the flock body (the body is single-quoted).
export MODE
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
    # never a silent skip. (A new-form cell cannot reach here: the writer refuses to write a
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
            # ONLY a terminal verdict marks the cell done — PASS, FIRST-WRONG or MISSING.
            # NOT-RUN and FENCED are failures to measure, not measurements: the cell re-runs.
            if line.split()[2] in ("PASS", "FIRST-WRONG", "MISSING"):
                skip = "True"
print("sha=%s" % c["sha"])
print("md5=%s" % md5)
print("skip=%s" % skip)
PY
)"
  [ "$skip" = "True" ] && { echo "SKIP $name (already verdicted at sha $sha md5 $md5)"; return; }

  # The fills: fp16-exact integers. x names the element — row in one run, column in the second,
  # a per-(row, col) mix in the third; cos/sin are 1 in the first two runs (out = x + rot, the
  # pairing made visible alone) and their own mix in the third. sin-prime is the SIGNED table,
  # sign[col % hd] * sin, the sign READ from cell.json (the writer read it from rope_p_entry —
  # the one statement of the sign law; this fill never types ±1 itself). P is the baseline
  # rope-p class, stated from the SAME op definition the host model states (the rope_p_entry
  # law, spelled from the reference rotate_half): P[inn][o] = -1 (o < half, inn = o+half), +1
  # (o >= half, inn = o-half), 0 otherwise.
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
sign = np.array(c["sin_sign"], dtype=np.float16)
sgn = np.tile(sign, total // hd)
for name in ("one", "mix"):
    if name == "one":
        sin = np.ones((mq, total), dtype=np.float16)
    else:
        sin = np.fromfile(f"{cell}/fill-mix-sin.bin", dtype=np.float16, count=mq * total).reshape(mq, total)
    ((sgn[None, :] * sin)).astype(np.float16).tofile(f"{cell}/fill-{name}-sins.bin")
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

  local x cos sin out p sins probe kind run rrc sbind
  eval "$(python3 - "$cell" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("x=%d cos=%d sin=%d out=%d p=%d sins=%d probe=%s kind=%s mq=%d hd=%d" % (
    c["x_tid"], c["cos_tid"], c["sin_tid"], c["out_tid"], c["p_tid"], c["sin_tid_signed"],
    c["probe_synth"], c["probe_kind"], c["mq"], c["hd"]))
PY
)"
  # Three runs, identical placement: x/cos/sin per fill, P always bound (every form places it —
  # the staging never depends on the form), sin-prime bound ONLY where the form reads it (its
  # own op input), the stick statements riding the input list (the runner reads them for the
  # OUTPUT restick too).
  for run in m c mix; do
    local cfill=$([ "$run" = mix ] && echo mix || echo one)
    sbind=""
    if [ "$kind" = rs ]; then
      sbind="t$sins=$cell/fill-$cfill-sins.bin stick:t$sins=$mq"
    fi
    ./target/release/examples/bundle_run_dev "$cell" \
      "t$x=$cell/fill-$run-x.bin" "stick:t$x=$mq" \
      "t$cos=$cell/fill-$cfill-cos.bin" "stick:t$cos=$mq" \
      "t$sin=$cell/fill-$cfill-sin.bin" "stick:t$sin=$mq" \
      "t$p=$cell/p.bin" "stick:t$p=$hd" \
      "stick:t$out=$mq" $sbind \
      --out "t$out=$cell/out-$run.bin" \
      --out "synth:$probe@$mq=$cell/probe-$run.bin" > "$cell/run-$run.log" 2>&1
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
mq, hd, total, kind = c["mq"], c["hd"], c["total"], c["probe_kind"]
n = mq * total
half = hd // 2
stamp = "sha=%s md5=%s" % (c["sha"], c["group_0_md5"])
# ⭐ THE HOST MODEL, FROM THE OP DEFINITION — HF rotate_half + the elementwise combine, never
# our opspec: per head h, position d (< hd/2): rot = -x[h*hd+d+half]; else rot = +x[h*hd+d-half].
# The probe is rot itself, or rs = rot * sin when the form has no rot tensor (halfswap4).
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
    probe = (rot.astype(np.float32) * sin.astype(np.float32)).astype(np.float16) if kind == "rs" else rot
    out = (x.astype(np.float32) * cos.astype(np.float32)
           + rot.astype(np.float32) * sin.astype(np.float32)).astype(np.float16)
    return probe, out
worst = None
for run in ("m", "c", "mix"):
    wp, wo = want(run)
    gp = np.fromfile(f"{cell}/probe-{run}.bin", dtype=np.float16, count=n)
    go = np.fromfile(f"{cell}/out-{run}.bin", dtype=np.float16, count=n)
    if gp.size < n or go.size < n:
        print(f"VERDICT {name} NOT-RUN short-readback probe={gp.size}/{n} out={go.size}/{n} {stamp}")
        sys.exit()
    gp = gp.reshape(mq, total)
    go = go.reshape(mq, total)
    for tensor, got, w in ((kind, gp, wp), ("out", go, wo)):
        bad = got != w
        if bad.any():
            i = int(bad.nonzero()[0][0])  # first wrong ROW; report its first wrong column too
            row = got[i] != w[i]
            cc = int(row.nonzero()[0][0])
            if worst is None:
                worst = (run, tensor, i, cc, int(got[i, cc]), int(w[i, cc]), int(bad.sum()))
if worst is None:
    print("VERDICT %s PASS form=%s probe=%s mq=%d hd=%d heads=%d desc=%d %s"
          % (name, c["form"], kind, mq, hd, c["heads"], c["descriptors"], stamp))
else:
    run, tensor, mi, cc, got, w, count = worst
    print("VERDICT %s FIRST-WRONG form=%s probe=%s mq=%d hd=%d heads=%d desc=%d at=%s[%d,%d] (head %d pos %d) "
          "got=%d want=%d wrong=%d/%d %s"
          % (name, c["form"], kind, mq, hd, c["heads"], c["descriptors"], tensor, mi, cc,
             cc // hd, cc % hd, got, w, count, n, stamp))
PY
}

# ⭐ THE NOISE-PROTOCOL LEG (--noise): per shape, five ABBA pairs — the three A/A controls (both
# dirs copies of one bundle; the copies live under <root>/.aa so the two executors never share a
# directory) and main against each new form. Each pair x pass is ONE FRESH PROCESS
# (bundle_run_dev --abba), two independent passes, 26 rounds = 52 timed launches per side, no
# refill (launch_timed). A NOISE line reports n / median / q1 / q3 / IQR / min per side at the
# sha+md5 stamp of both sides; the decision rule (3x the A/A floor, separated IQRs, same sign in
# both passes) is applied at REPORT time over the logged lines. A halfswap4 side binds the
# signed sin-prime table as its own input; main and halfswap bind the same four tensors.
noise_shape() {
  local shape=$1 main=$2 hs=$3 hs4=$4
  local x cos sin out p sins mq hd
  eval "$(python3 - "$main" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("x=%d cos=%d sin=%d out=%d p=%d sins=%d mq=%d hd=%d" % (
    c["x_tid"], c["cos_tid"], c["sin_tid"], c["out_tid"], c["p_tid"], c["sin_tid_signed"],
    c["mq"], c["hd"]))
PY
)"
  # Every form cell needs its bundle compiled (the correctness sweep already did; compile only
  # when the dxp artifacts are missing) and its fills written.
  local c sig crc
  for c in "$main" "$hs" "$hs4"; do
    if [ ! -d "$c/group_0" ]; then
      echo "NOISE shape=$shape pair=ab-hs pass=1 NOT-RUN no-bundle ($c) sha=unknown md5A=(no-bundle) md5B=(no-bundle)" | tee -a "$LOG"
      return
    fi
    if [ ! -d "$c/group_0/spyreCodeDir" ]; then
      timeout 1800 /opt/ibm/spyre/deeptools/bin/dxp_standalone --bundle -d "$c/group_0" -b sentient \
        > "$c/compile.log" 2>&1
      crc=$?
      sig=$(fenced "$c/compile.log")
      if [ -n "$sig" ]; then
        echo "NOISE shape=$shape pair=ab-hs pass=1 FENCED at-compile sig=[$sig]" | tee -a "$LOG"
        echo "FENCE-ABORT at $shape (noise compile)" >&2
        exit 99
      fi
      [ $crc -ne 0 ] && { echo "NOISE shape=$shape pair=ab-hs pass=1 NOT-RUN compile_rc=$crc ($c)" | tee -a "$LOG"; return; }
    fi
    python3 - "$c" <<"PY"
import json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(cell + "/cell.json"))
mq, hd, total = c["mq"], c["hd"], c["total"]
m = np.arange(mq)[:, None]
np.broadcast_to((m % 15) + 1, (mq, total)).astype(np.float16).tofile(cell + "/fill-m-x.bin")
np.ones((mq, total), dtype=np.float16).tofile(cell + "/fill-one-cos.bin")
np.ones((mq, total), dtype=np.float16).tofile(cell + "/fill-one-sin.bin")
sign = np.array(c["sin_sign"], dtype=np.float16)
(np.broadcast_to(np.tile(sign, total // hd)[None, :], (mq, total))).astype(np.float16).tofile(cell + "/fill-one-sins.bin")
half = hd // 2
inn = np.arange(hd)[:, None]
o = np.arange(hd)[None, :]
p = np.where(o < half, np.where(inn == o + half, -1.0, 0.0),
             np.where(inn == o - half, 1.0, 0.0)).astype(np.float16)
p.tofile(cell + "/p.bin")
PY
    [ $? -eq 0 ] || { echo "NOISE shape=$shape pair=ab-hs pass=1 NOT-RUN fill-failed ($c)" | tee -a "$LOG"; return; }
  done
  # Stamps first (md5 of group_0, excluding the dxp artifacts — compile-independent), THEN the
  # A/A copies, so each copy carries its own stamped cell.json.
  local sha_main sha_hs sha_hs4 md5_main md5_hs md5_hs4
  eval "$(python3 - "$main" "$hs" "$hs4" <<"PY"
import hashlib, json, os, sys
def stamp(cell):
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
    return c["sha"], md5
for name, cell in (("main", sys.argv[1]), ("hs", sys.argv[2]), ("hs4", sys.argv[3])):
    sha, md5 = stamp(cell)
    print("sha_%s=%s" % (name, sha))
    print("md5_%s=%s" % (name, md5))
PY
)"
  local aaroot=$ROOT/.aa
  mkdir -p "$aaroot"
  rm -rf "$aaroot/$shape-aa-main" "$aaroot/$shape-aa-hs" "$aaroot/$shape-aa-hs4"
  cp -r "$main" "$aaroot/$shape-aa-main"
  cp -r "$hs" "$aaroot/$shape-aa-hs"
  cp -r "$hs4" "$aaroot/$shape-aa-hs4"
  # The binding lists: main and halfswap bind the same four tensors; halfswap4 adds the signed
  # sin-prime table (its own op input). The A/A copies bind the ORIGINAL cell fills — staging
  # reads the files once at load; the timed launches never touch them again.
  local bind_main bind_hs4
  bind_main="t$x=$main/fill-m-x.bin stick:t$x=$mq t$cos=$main/fill-one-cos.bin stick:t$cos=$mq t$sin=$main/fill-one-sin.bin stick:t$sin=$mq t$p=$main/p.bin stick:t$p=$hd stick:t$out=$mq"
  bind_hs4="t$x=$hs4/fill-m-x.bin stick:t$x=$mq t$cos=$hs4/fill-one-cos.bin stick:t$cos=$mq t$sin=$hs4/fill-one-sin.bin stick:t$sin=$mq t$p=$hs4/p.bin stick:t$p=$hd t$sins=$hs4/fill-one-sins.bin stick:t$sins=$mq stick:t$out=$mq"

  local pass pair dir_a dir_b bind_a bind_b sha_a md5_a md5_b plog rrc
  for pass in 1 2; do
    for pair in aa-main aa-hs aa-hs4 ab-hs ab-hs4; do
      case "$pair" in
        aa-main) dir_a=$main;    dir_b=$aaroot/$shape-aa-main; bind_a=$bind_main;  bind_b=$bind_main;  sha_a=$sha_main; md5_a=$md5_main; md5_b=$md5_main;;
        aa-hs)   dir_a=$hs;      dir_b=$aaroot/$shape-aa-hs;   bind_a=$bind_main;  bind_b=$bind_main;  sha_a=$sha_hs;   md5_a=$md5_hs;   md5_b=$md5_hs;;
        aa-hs4)  dir_a=$hs4;     dir_b=$aaroot/$shape-aa-hs4;  bind_a=$bind_hs4;   bind_b=$bind_hs4;   sha_a=$sha_hs4;  md5_a=$md5_hs4;  md5_b=$md5_hs4;;
        ab-hs)   dir_a=$main;    dir_b=$hs;                    bind_a=$bind_main;  bind_b=$bind_main;  sha_a=$sha_main; md5_a=$md5_main; md5_b=$md5_hs;;
        ab-hs4)  dir_a=$main;    dir_b=$hs4;                   bind_a=$bind_main;  bind_b=$bind_hs4;   sha_a=$sha_main; md5_a=$md5_main; md5_b=$md5_hs4;;
      esac
      if grep "^NOISE shape=$shape pair=$pair pass=$pass " "$LOG" 2>/dev/null \
        | grep -F "sha=$sha_a md5A=$md5_a md5B=$md5_b" | grep -q "medA="; then
        echo "SKIP-NOISE $shape $pair pass=$pass (already logged at sha $sha_a md5A $md5_a md5B $md5_b)"
        continue
      fi
      plog=$aaroot/$shape-$pair-p$pass.log
      ./target/release/examples/bundle_run_dev --abba "$dir_a" "$dir_b" --rounds 26 \
        --a $bind_a --b $bind_b > "$plog" 2>&1
      rrc=$?
      sig=$(fenced "$plog")
      if [ -n "$sig" ]; then
        echo "NOISE shape=$shape pair=$pair pass=$pass FENCED sig=[$sig] sha=$sha_a md5A=$md5_a md5B=$md5_b" | tee -a "$LOG"
        echo "FENCE-ABORT at $shape $pair pass=$pass" >&2
        exit 99
      fi
      [ $rrc -ne 0 ] && { echo "NOISE shape=$shape pair=$pair pass=$pass NOT-RUN rc=$rrc sha=$sha_a md5A=$md5_a md5B=$md5_b" | tee -a "$LOG"; continue; }
      python3 - "$plog" "$shape" "$pair" "$pass" "$sha_a" "$md5_a" "$md5_b" >> "$LOG" <<"PY"
import sys
plog, shape, pair, pass_, sha, md5a, md5b = sys.argv[1:8]
samples = {"A": [], "B": []}
for line in open(plog):
    if line.startswith("TIME-SAMPLE "):
        f = dict(kv.split("=", 1) for kv in line.split()[1:])
        samples[f["side"]].append(int(f["ns"]))
def stats(v):
    v = sorted(v)
    n = len(v)
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
    def pct(p):
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    q1, q3 = pct(0.25), pct(0.75)
    return n, med, q1, q3, v[0]
if not samples["A"] or not samples["B"]:
    print("NOISE shape=%s pair=%s pass=%s NOT-RUN no-samples sha=%s md5A=%s md5B=%s"
          % (shape, pair, pass_, sha, md5a, md5b))
    sys.exit()
nA, medA, q1A, q3A, minA = stats(samples["A"])
nB, medB, q1B, q3B, minB = stats(samples["B"])
print("NOISE shape=%s pair=%s pass=%s nA=%d medA=%s q1A=%s q3A=%s iqrA=%s minA=%s "
      "nB=%d medB=%s q1B=%s q3B=%s iqrB=%s minB=%s sha=%s md5A=%s md5B=%s"
      % (shape, pair, pass_, nA, medA, q1A, q3A, q3A - q1A, minA,
         nB, medB, q1B, q3B, q3B - q1B, minB, sha, md5a, md5b))
PY
    done
  done
}

# ⛔ THE CELL LIST, BUILT ONCE — explicit cells if given, every cell dir sorted otherwise (the
# dot-prefixed scratch dirs — the noise leg A/A copies — are never cells).
if [ -n "$CELLS" ]; then
  cell_list="$CELLS"
else
  [ -d "$ROOT" ] || { echo "ROPE_HALF_SWAP_RUN_INCOMPLETE — not a dir: $ROOT" >&2; exit 1; }
  cell_list=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name ".*" | sort)
  if [ -z "$cell_list" ]; then
    echo "ROPE_HALF_SWAP_RUN_INCOMPLETE — no cell dirs in $ROOT" >&2
    exit 1
  fi
fi
if [ "$MODE" = noise ]; then
  # Group the cells into shapes (all three forms must be present) and run the protocol per
  # shape. The grouping reads every cell.json once; a shape missing a form is a NOT-RUN verdict
  # line, never a silent skip.
  shapes=$(CELLS="$cell_list" python3 - "$ROOT" <<"PY"
import json, os, sys
cells = [c for c in os.environ.get("CELLS", "").split() if c]
if not cells:
    root = sys.argv[1]
    cells = sorted(os.path.join(root, d) for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and not d.startswith("."))
groups = {}
for c in cells:
    try:
        j = json.load(open(os.path.join(c, "cell.json")))
    except Exception:
        continue
    groups.setdefault((j["mq"], j["hd"], j["heads"]), {})[j["form"]] = c
for (mq, hd, heads), g in sorted(groups.items()):
    shape = "mq%d-hd%d-h%d" % (mq, hd, heads)
    if all(f in g for f in ("main", "halfswap", "halfswap4")):
        print(shape, g["main"], g["halfswap"], g["halfswap4"])
    else:
        for f in ("main", "halfswap", "halfswap4"):
            if f not in g:
                print(shape, "MISSING-FORM", f, "", "")
PY
)
  while read -r shape main hs hs4; do
    [ -n "$shape" ] || continue
    if [ "$main" = MISSING-FORM ]; then
      echo "NOISE shape=$shape pair=ab-hs pass=1 NOT-RUN missing-form=$hs sha=unknown md5A=(no-bundle) md5B=(no-bundle)" | tee -a "$LOG"
      continue
    fi
    noise_shape "$shape" "$main" "$hs" "$hs4"
  done <<< "$shapes"
  # ⭐ THE DECISION RULE, at report time: for each shape the A/A floor is the LARGEST same-bundle
  # median delta across the three controls (both passes); the delta of a new form is REAL only if
  # |delta| > 3x that floor in BOTH passes, the IQRs separate in BOTH passes, and the sign
  # repeats across the passes. A missing NOISE line is INCOMPLETE — never a verdict.
  python3 - "$LOG" <<"PY"
import sys
rows = {}
shapes = set()
for line in open(sys.argv[1]):
    if not line.startswith("NOISE shape="):
        continue
    f = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
    shapes.add(f["shape"])
    if "medA" not in f:
        continue
    rows[(f["shape"], f["pair"], f["pass"])] = f
shapes = sorted(shapes)
incomplete = []
verdicts = 0
for s in shapes:
    aa = [rows.get((s, p, q)) for p in ("aa-main", "aa-hs", "aa-hs4") for q in ("1", "2")]
    if any(a is None for a in aa):
        incomplete.append(s + " (aa)")
        continue
    floor = max(abs(float(a["medA"]) - float(a["medB"])) for a in aa)
    for pair in ("ab-hs", "ab-hs4"):
        r1, r2 = rows.get((s, pair, "1")), rows.get((s, pair, "2"))
        if r1 is None or r2 is None:
            incomplete.append("%s (%s)" % (s, pair))
            continue
        d1 = float(r1["medB"]) - float(r1["medA"])
        d2 = float(r2["medB"]) - float(r2["medA"])
        def sep(r):
            return float(r["q3A"]) < float(r["q1B"]) or float(r["q3B"]) < float(r["q1A"])
        real = (abs(d1) > 3 * floor and abs(d2) > 3 * floor
                and sep(r1) and sep(r2) and (d1 > 0) == (d2 > 0))
        print("NOISE-VERDICT shape=%s pair=%s real=%s d1=%+.1f d2=%+.1f floor=%.1f medA1=%s medB1=%s medA2=%s medB2=%s"
              % (s, pair, "yes" if real else "no", d1, d2, floor,
                 r1["medA"], r1["medB"], r2["medA"], r2["medB"]))
        verdicts += 1
print("NOISE-SHAPES=%d VERDICTS=%d INCOMPLETE=%d" % (len(shapes), verdicts, len(incomplete)))
for i in incomplete:
    print("NOISE-INCOMPLETE %s" % i)
sys.exit(1 if incomplete or verdicts == 0 else 0)
PY
  nrc=$?
  [ $nrc -ne 0 ] && { echo "ROPE_HALF_SWAP_NOISE_INCOMPLETE — every shape needs all five pairs, both passes, at its own stamp" >&2; exit 1; }
  echo ROPE_HALF_SWAP_NOISE_DONE
  exit 0
fi

while read -r c; do run_cell "$c"; done <<< "$cell_list"

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
