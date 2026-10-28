#!/bin/bash
# B1 card-run leg: build bundle_run_dev (with the probe-only rawout patch) at
# e63c6f8ac8 in /work/b1-tree, dxp-compile the integer-op cells, run them on the
# card, and compare against host wrapping i32 / host `as f32` / host ln.
# Markers: B1-DONE / B1-FAILED. Logs: /work/b1-cardleg.log
set -u

LOG=/work/b1-cardleg.log
TREE=/work/b1-tree
TDIR=/work/pr916/head/target
PROBE=/work/b1-hashprobe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone
SHA=e63c6f8ac8
UPSTREAM=https://github.com/AI-native-Systems-Research/scratchy.git

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== b1-cardleg start $(date -u +%FT%TZ) ===="

# ── busy check ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo B1-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo B1-FAILED; exit 1
fi

# ── the probe tree: a checkout at e63c6f8ac8, warm target dir reused ──
if [ ! -d "$TREE/.git" ]; then
  git clone --quiet "$UPSTREAM" "$TREE" || { echo CLONE-FAILED; echo B1-FAILED; exit 1; }
fi
cd "$TREE" || { echo "no tree"; echo B1-FAILED; exit 1; }
git fetch --quiet "$UPSTREAM" main || { echo FETCH-FAILED; echo B1-FAILED; exit 1; }
git checkout --quiet -f "$SHA" || { echo "checkout $SHA failed"; echo B1-FAILED; exit 1; }
HEADSHA=$(git rev-parse HEAD)
echo "tree at $HEADSHA (want $SHA)"
case "$HEADSHA" in
  "$SHA"*) ;;
  *) echo SHA-MISMATCH; echo B1-FAILED; exit 1 ;;
esac

# ── the probe-only rawout patch (never in any pushed tree) ──
if ! grep -q rawout crates/targets/spyre/src/bundle_run.rs; then
  git apply "$PROBE/rawout.patch" || { echo PATCH-FAILED; echo B1-FAILED; exit 1; }
fi
echo "rawout patched: $(grep -c rawout crates/targets/spyre/src/bundle_run.rs) mentions"

# ── build (WARM: the pr916/head target dir — one target dir, shared; nothing else runs) ──
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
echo "==== B1-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo B1-BUILD-FAILED
  echo B1-FAILED
  exit 1
fi
echo "==== B1-BUILD-OK $(date -u +%FT%TZ) ===="

# ── cells: build probe dirs at w2048 (control + both geometry legs + i2f + log) ──
echo "==== B1-CELLS $(date -u +%FT%TZ) ===="
python3 "$PROBE/build_cells.py" || { echo CELLS-FAILED; echo B1-FAILED; exit 1; }

# ── dxp compile: bundle_run_dev expects <dir>/spyreCodeDir/ to already exist ──
echo "==== B1-DXP $(date -u +%FT%TZ) ===="
for leg in control mul_g1 mul_g2 add_g1 add_g2 i2f_g1 log_g1; do
  d="$PROBE/cells/$leg"
  if [ ! -f "$d/spyreCodeDir/init_binary.bin" ]; then
    if ! "$DXP" -d "$d" -b sentient > "$d/dxp.log" 2>&1; then
      echo "B1-DXP-FAILED $leg (rc=$?)"; tail -5 "$d/dxp.log"
      if [ "$leg" != "log_g1" ]; then echo B1-FAILED; exit 1; fi
    else
      echo "B1-DXP-OK $leg ($(stat -c%s "$d/spyreCodeDir/init_binary.bin") B image)"
    fi
  fi
done

# ── run every leg on the card ──
echo "==== B1-RUN-START $(date -u +%FT%TZ) ===="
fails=0
logfail=0

# control — the base cell VERBATIM through the ORIGINAL IO path (IEEE->SEN staging,
# SEN->IEEE readback, stick-major map stated per tensor): proves dxp + launch +
# readback on this exact geometry before any integer spelling is judged.
d="$PROBE/cells/control"
echo "===== B1-CELL control ====="
if ! "$BIN" "$d" t0="$PROBE/cells/in0_control.bin" t1="$PROBE/cells/in1_control.bin" \
    stick:t0=256 stick:t1=256 stick:t2=256 \
    --out t2="$PROBE/cells/out_control.bin" > "$PROBE/cells/run_control.log" 2>&1; then
  echo "B1-RUN-FAILED control"; tail -20 "$PROBE/cells/run_control.log"; fails=$((fails+1))
else
  tail -5 "$PROBE/cells/run_control.log"
fi

for leg in mul_g1 mul_g2 add_g1 add_g2 i2f_g1 log_g1; do
  d="$PROBE/cells/$leg"
  echo "===== B1-CELL $leg ====="
  in0="$PROBE/cells/in0_$leg.bin"; in1="$PROBE/cells/in1_$leg.bin"
  if [ -f "$in1" ]; then
    if ! "$BIN" "$d" t0="$in0" raw:t0 t1="$in1" raw:t1 \
        --out t2="$PROBE/cells/out_$leg.bin" rawout:t2 > "$PROBE/cells/run_$leg.log" 2>&1; then
      echo "B1-RUN-FAILED $leg"; tail -20 "$PROBE/cells/run_$leg.log"
      if [ "$leg" = "log_g1" ]; then logfail=1; else fails=$((fails+1)); fi
      continue
    fi
  else
    if ! "$BIN" "$d" t0="$in0" raw:t0 \
        --out t2="$PROBE/cells/out_$leg.bin" rawout:t2 > "$PROBE/cells/run_$leg.log" 2>&1; then
      echo "B1-RUN-FAILED $leg"; tail -20 "$PROBE/cells/run_$leg.log"
      if [ "$leg" = "log_g1" ]; then logfail=1; else fails=$((fails+1)); fi
      continue
    fi
  fi
  tail -5 "$PROBE/cells/run_$leg.log"
done

# ── the verdicts, computed on the pod ──
echo "==== B1-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/check_cells.py"
rc=$?
[ "$logfail" -ne 0 ] && echo "B1-NOTE log_g1 run failed — reported, not gated (the ulp number was the ask)"
echo "==== b1-cardleg end $(date -u +%FT%TZ) fails=$fails logfail=$logfail checkrc=$rc ===="
if [ "$fails" -gt 0 ] || [ "$rc" -ne 0 ]; then echo B1-FAILED; else echo B1-DONE; fi
