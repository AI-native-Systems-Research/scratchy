#!/bin/bash
# front-b-noise WALL-9 REDUCE-CHAIN LEG (nickm2): the device-written [B,1] —
# auditor order after the wide leg failed identically: "Learn the device's own
# [B,1] layout. In the batched greedy bundle (Front A, b2), read back the
# reduce's [2,1] mm output after a step with two different rows, and find
# where row 1's max lives (byte offset, stick, word)."
# The cell: intmr_f32maxadd — the sampler's own `max -> expand_dims -> add`
# chain at production width, field-identical to the equality's writer->reader
# pair (max {mb:2} scale [1,-2,1]; add {mb:1,out:32} scale [1,-2]; the mm
# intermediate minted at the same 256 B). The output is the sum through the
# sampler's fptosi sink (i32), so the addend per (row, stick) names the buffer
# word the Col read picked. Tree: /work/front-b-noise. Bake: ON THE POD,
# KTIR_WHOLE=1 through the whole-function door. Runner: bundle_run_dev at
# release in the WARM target dir /work/pr916/head/target. Verdict ON THE POD
# (intmr_maxadd_stats.py); the numbers land in intmr-maxadd-verdict.json.
# Markers: INTMRMA-DONE / INTMRMA-FAILED. Log: /work/intmr-maxadd-leg.log
set -u

LOG=/work/intmr-maxadd-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

CELL=intmr_f32maxadd

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intmr-maxadd-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTMRMA-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTMRMA-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTMRMA-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTMRMA-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir; the runner crate is
#    unchanged by this cell — a warm no-op if the tree matches) ──
echo "==== INTMRMA-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTMRMA-BUILD-FAILED; echo INTMRMA-FAILED; exit 1
fi
echo "==== INTMRMA-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged inputs, ONCE (deterministic, no RNG, rows differ) ──
echo "==== INTMRMA-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_inputs.py" || { echo INTMRMA-INPUTS-FAILED; echo INTMRMA-FAILED; exit 1; }

# ── leg 3: bake (the new fixture + cases row ride the rsync) ──
BAKE=/work/int-mr-$CELL
echo "==== INTMRMA-BAKE-START $(date -u +%FT%TZ) ===="
rm -rf "$BAKE"
if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
  echo "INTMRMA-BAKE-FAILED ($CELL)"; echo INTMRMA-FAILED; exit 1
fi
echo "==== INTMRMA-BAKE-OK ($(grep -o '"numWkSlicesPerDim_":{[^}]*}' "$BAKE"/sdsc_0.json | head -1)) ===="

# ── leg 4: dxp ──
echo "==== INTMRMA-DXP $(date -u +%FT%TZ) ===="
if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
  echo "INTMRMA-DXP-FAILED"; tail -5 "$BAKE/dxp.log"; echo INTMRMA-FAILED; exit 1
fi
echo "INTMRMA-DXP-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

# ── leg 5: run (one input, one output; the mm intermediate is minted) ──
echo "==== INTMRMA-RUN-START $(date -u +%FT%TZ) ===="
SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
  t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
  --out raw:t1="$PROBE/out_${CELL}_o.bin" \
  > "$PROBE/intmr-run-$CELL.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "INTMRMA-RUN-FAILED (rc=$rc)"; tail -20 "$PROBE/intmr-run-$CELL.log"; echo INTMRMA-FAILED; exit 1
fi
tail -8 "$PROBE/intmr-run-$CELL.log"

# ── leg 6: the verdict, on the pod ──
echo "==== INTMRMA-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_maxadd_stats.py"
rc=$?
echo "==== intmr-maxadd-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo INTMRMA-FAILED; exit 1; fi
echo INTMRMA-DONE
