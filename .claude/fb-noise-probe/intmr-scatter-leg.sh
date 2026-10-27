#!/bin/bash
# front-b-noise WALL-9 SCATTER LEG (nickm2): the CORRECTED HARNESS re-leg —
# auditor ruling 2026-10-27: "Fix the HARNESS: stage every multi-row input
# through stick_scatter and un-scatter every output (forward_tape.rs:299 law),
# then re-leg the col, maxadd and wide cells. Expect them to pass. If they do,
# the multi-row Col int arm is admissible, citing the corrected legs, and wall
# 9 is closed. Write no emission change."
# The first legs staged/read ROW-MAJOR against the device's STICK-MAJOR layout
# (stick k of row r at flat stick k*rows + r); the s-mod-rows failure signature
# on every row-sensitive cell was the scatter itself. The dense controls passed
# because a position-wise op commutes with a consistent permutation. The
# [rows,1] Col tables are scatter fixed points, so the card's per-row Col
# semantics were never wrong.
# Default cells: everything that bakes at HEAD (the int Col cells need the
# multi-row Col admission re-landed first; they ride the same script via
# CELLS="intmr_col_add intmr_col_mul"). Tree: /work/front-b-noise. Bake: ON
# THE POD, KTIR_WHOLE=1 through the whole-function door. Runner: bundle_run_dev
# at release in the WARM target dir /work/pr916/head/target. Verdict ON THE POD
# (intmr_scatter_stats.py, cells named on the command line); the numbers land
# in intmr-scatter-verdict.json. Markers: INTMRSC-DONE / INTMRSC-FAILED.
# Log: /work/intmr-scatter-leg.log
set -u

LOG=/work/intmr-scatter-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

# The cells this leg runs — POSITIONAL ARGS, else the bakeable set by default
# (the int Col cells need the multi-row Col admission re-landed first; they
# ride the same script as args). The BG re-exec passes "$@" through.
CELLS="${*:-intmr_dense_add intmr_dense_mul intmr_f32col_add intmr_f32wide_add intmr_f32maxadd}"

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" "$@" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intmr-scatter-leg start $(date -u +%FT%TZ) cells: $CELLS ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTMRSC-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTMRSC-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTMRSC-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTMRSC-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== INTMRSC-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTMRSC-BUILD-FAILED; echo INTMRSC-FAILED; exit 1
fi
echo "==== INTMRSC-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged inputs (stick-scattered), ONCE ──
echo "==== INTMRSC-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_inputs.py" || { echo INTMRSC-INPUTS-FAILED; echo INTMRSC-FAILED; exit 1; }

# ── legs 3-5: per cell — bake, dxp, run ──
for CELL in $CELLS; do
  BAKE=/work/int-mr-$CELL
  echo "==== INTMRSC-BAKE-START $CELL $(date -u +%FT%TZ) ===="
  rm -rf "$BAKE"
  if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
    echo "INTMRSC-BAKE-FAILED ($CELL)"; echo INTMRSC-FAILED; exit 1
  fi
  echo "==== INTMRSC-BAKE-OK $CELL ($(grep -o '"numWkSlicesPerDim_":{[^}]*}' "$BAKE"/sdsc_0.json | head -1)) ===="

  echo "==== INTMRSC-DXP $CELL $(date -u +%FT%TZ) ===="
  if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
    echo "INTMRSC-DXP-FAILED ($CELL)"; tail -5 "$BAKE/dxp.log"; echo INTMRSC-FAILED; exit 1
  fi
  echo "INTMRSC-DXP-OK $CELL ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

  echo "==== INTMRSC-RUN-START $CELL $(date -u +%FT%TZ) ===="
  if [ "$CELL" = intmr_f32maxadd ]; then
    SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
      t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
      --out raw:t1="$PROBE/out_${CELL}_o.bin" \
      > "$PROBE/intmr-run-$CELL.log" 2>&1
  else
    SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
      t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
      t1="$PROBE/in/${CELL}_t01_b.bin" raw:t1 \
      --out raw:t2="$PROBE/out_${CELL}_o.bin" \
      > "$PROBE/intmr-run-$CELL.log" 2>&1
  fi
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "INTMRSC-RUN-FAILED ($CELL rc=$rc)"; tail -20 "$PROBE/intmr-run-$CELL.log"; echo INTMRSC-FAILED; exit 1
  fi
  tail -4 "$PROBE/intmr-run-$CELL.log"
done

# ── leg 6: the verdict, on the pod — exactly the cells this leg ran ──
echo "==== INTMRSC-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_scatter_stats.py" $CELLS
rc=$?
echo "==== intmr-scatter-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo INTMRSC-FAILED; exit 1; fi
echo INTMRSC-DONE
