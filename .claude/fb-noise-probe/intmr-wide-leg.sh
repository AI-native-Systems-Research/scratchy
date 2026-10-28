#!/bin/bash
# front-b-noise WALL-9 WIDTH LEG (nickm2): the WIDE fp32 twin, one cell —
# auditor order 2026-10-27T02:5xZ, after the re-leg refuted the both-split
# diagnosis: "Leg the fp32 twin at [2, 65536] (production width, out=2048
# sticks) with a [2,1] Col. If it passes, the defect is narrow-row only, so
# production is safe... If it fails, the batched greedy equality is only
# passing by luck, and that's urgent."
# The cell: intmr_f32wide_add — the SAME intmr_f32col_add kernel at PAD_V
# 65536 (the equality's fold class {mb:1, out:32}, 64 sticks per core against
# the narrow twin's 1). Tree: /work/front-b-noise (the law DROPPED at
# 2f20b0e2aa; the fp32 door needs no admission). Bake: ON THE POD, KTIR_WHOLE=1
# through the whole-function door. Runner: bundle_run_dev at release in the
# WARM target dir /work/pr916/head/target. Verdict ON THE POD
# (intmr_wide_stats.py: bit-for-bit, per row and lane, plus the per-stick
# addend pattern — which row's word each stick read); the numbers land in
# intmr-wide-verdict.json. Markers: INTMRW-DONE / INTMRW-FAILED. Log:
# /work/intmr-wide-leg.log
set -u

LOG=/work/intmr-wide-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

CELL=intmr_f32wide_add

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intmr-wide-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTMRW-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTMRW-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTMRW-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTMRW-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir; the tree changed — the law
#    dropped — so this rebuilds ktir-superdsc and its dependents) ──
echo "==== INTMRW-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTMRW-BUILD-FAILED; echo INTMRW-FAILED; exit 1
fi
echo "==== INTMRW-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged inputs, ONCE (deterministic, no RNG, rows differ) ──
echo "==== INTMRW-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_inputs.py" || { echo INTMRW-INPUTS-FAILED; echo INTMRW-FAILED; exit 1; }

# ── leg 3: bake ──
BAKE=/work/int-mr-$CELL
echo "==== INTMRW-BAKE-START $(date -u +%FT%TZ) ===="
rm -rf "$BAKE"
if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
  echo "INTMRW-BAKE-FAILED ($CELL)"; echo INTMRW-FAILED; exit 1
fi
echo "==== INTMRW-BAKE-OK ($(grep -o '"numWkSlicesPerDim_":{[^}]*}' "$BAKE"/*.json | head -1)) ===="

# ── leg 4: dxp ──
echo "==== INTMRW-DXP $(date -u +%FT%TZ) ===="
if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
  echo "INTMRW-DXP-FAILED"; tail -5 "$BAKE/dxp.log"; echo INTMRW-FAILED; exit 1
fi
echo "INTMRW-DXP-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

# ── leg 5: run ──
echo "==== INTMRW-RUN-START $(date -u +%FT%TZ) ===="
SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
  t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
  t1="$PROBE/in/${CELL}_t01_b.bin" raw:t1 \
  --out raw:t2="$PROBE/out_${CELL}_o.bin" \
  > "$PROBE/intmr-run-$CELL.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "INTMRW-RUN-FAILED (rc=$rc)"; tail -20 "$PROBE/intmr-run-$CELL.log"; echo INTMRW-FAILED; exit 1
fi
tail -8 "$PROBE/intmr-run-$CELL.log"

# ── leg 6: the verdict, on the pod ──
echo "==== INTMRW-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_wide_stats.py"
rc=$?
echo "==== intmr-wide-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo INTMRW-FAILED; exit 1; fi
echo INTMRW-DONE
