#!/bin/bash
# front-b-noise INT MULTI-ROW card leg (nickm2): the multi-row Col admission's
# measurement, four cells for attribution — auditor GO 2026-10-27: "[2,512]
# dense + [2,512] dense (the control), then [2,512] + [2,1] Col, for add AND
# mul", all in the production pid-cornered spelling (grid [B]=2, [1, ·]
# blocks, the per-row loads at [pid, 0]). Tree: /work/front-b-noise — THIS
# session's worktree, rsynced from the Mac (never cloned: nothing is pushed).
# Bake: ON THE POD, from that tree, KTIR_WHOLE=1 through the whole-function
# door, WITH the integer door's multi-row Col admission (the probe measures
# exactly that emission). Runner: bundle_run_dev at release in the WARM target
# dir /work/pr916/head/target (one target dir, shared; nothing else runs).
# Outputs read back VERBATIM through the tree's own `raw:` output spelling.
# Verdict ON THE POD (intmr_stats.py: bit-for-bit, per row and lane, against
# the host wrapping op — dense control vs Col, add vs mul); the numbers land
# in intmr-verdict.json. Markers: INTMR-DONE / INTMR-FAILED. Log:
# /work/intmr-leg.log
set -u

LOG=/work/intmr-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

CELLS="intmr_dense_add intmr_dense_mul intmr_col_add intmr_col_mul"

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intmr-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTMR-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTMR-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTMR-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTMR-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== INTMR-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTMR-BUILD-FAILED; echo INTMR-FAILED; exit 1
fi
echo "==== INTMR-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged inputs, ONCE (deterministic, no RNG, rows differ) ──
echo "==== INTMR-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_inputs.py" || { echo INTMR-INPUTS-FAILED; echo INTMR-FAILED; exit 1; }

# ── legs 3-5, per cell: bake → dxp → run ──
for CELL in $CELLS; do
  BAKE=/work/int-mr-$CELL
  echo "==== INTMR-BAKE-$CELL-START $(date -u +%FT%TZ) ===="
  rm -rf "$BAKE"
  if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
    echo "INTMR-BAKE-FAILED ($CELL)"; echo INTMR-FAILED; exit 1
  fi
  echo "==== INTMR-BAKE-$CELL-OK $(date -u +%FT%TZ) ===="

  echo "==== INTMR-DXP-$CELL $(date -u +%FT%TZ) ===="
  if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
    echo "INTMR-DXP-FAILED ($CELL)"; tail -5 "$BAKE/dxp.log"; echo INTMR-FAILED; exit 1
  fi
  echo "INTMR-DXP-$CELL-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

  echo "==== INTMR-RUN-$CELL-START $(date -u +%FT%TZ) ===="
  SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
    t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
    t1="$PROBE/in/${CELL}_t01_b.bin" raw:t1 \
    --out raw:t2="$PROBE/out_${CELL}_o.bin" \
    > "$PROBE/intmr-run-$CELL.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "INTMR-RUN-FAILED ($CELL rc=$rc)"; tail -20 "$PROBE/intmr-run-$CELL.log"; echo INTMR-FAILED; exit 1
  fi
  tail -8 "$PROBE/intmr-run-$CELL.log"
done

# ── leg 6: the verdict, on the pod — bit for bit, all four cells ──
echo "==== INTMR-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_stats.py"
rc=$?
echo "==== intmr-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo INTMR-FAILED; exit 1; fi
echo INTMR-DONE
