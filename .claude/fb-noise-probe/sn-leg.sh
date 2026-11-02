#!/bin/bash
# front-b-noise SAMPLE-NOISE card leg (nickm2) — auditor ruling (a), order
# 2026-10-27: "read g and score off the card in the same rebuild". The probe
# kernel is the PRODUCTION noise-and-score section of sample_fwd.py VERBATIM
# (selection dropped, hash_noise's own island law), staged at
# granite-3.1-2b's own geometry (V 49155, PAD_V 65536) in the production
# [B, 1] stick-head spelling, run at B=2 (the intmr-proven control width) and
# B=8 (the batched defect geometry — the rows whose uniform tables read ZERO
# before the d99c3d750f fix). u/g/score read off the card per row and lane.
# Tree: /work/front-b-noise (this branch). Bake: ON THE POD, from that tree,
# KTIR_WHOLE=1 through the whole-function door. Runner: bundle_run_dev at
# release in front-b's OWN target dir /work/front-b/target (auditor order
# 2026-10-27: never the shared /work/pr916/head/target). Binds VERIFIED
# against the bake's own placements.json (sn_binds.py — a door that numbers
# or sizes differently fails loudly, never a silent misbind). Outputs read
# back VERBATIM through the tree's own `raw:` output spelling. Verdict on
# the pod (sn_stats.py: u bit-exact with first-wrong-value naming, g live +
# Gumbel moments, score's argmax moved, g/score vs the host emulation);
# numbers land in sn-verdict-<case>.json. Markers: SN-DONE / SN-FAILED.
# Log: /work/sn-leg.log
set -u

LOG=/work/sn-leg.log
TREE=/work/front-b-noise
TDIR=/work/front-b/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

CELLS="sample_noise_b2 sample_noise_b8"

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== sn-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo SN-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo SN-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo SN-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo SN-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, front-b's own warm target dir) ──
echo "==== SN-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo SN-BUILD-FAILED; echo SN-FAILED; exit 1
fi
echo "==== SN-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged input tables (deterministic, no RNG, rows differ) ──
echo "==== SN-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/sn_inputs.py" || { echo SN-INPUTS-FAILED; echo SN-FAILED; exit 1; }

# ── legs 3-5, per cell: bake → binds verify → dxp → run ──
for CELL in $CELLS; do
  CASE=${CELL#sample_noise_}
  BAKE=/work/sn-$CELL
  echo "==== SN-BAKE-$CELL-START $(date -u +%FT%TZ) ===="
  rm -rf "$BAKE"
  if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
    echo "SN-BAKE-FAILED ($CELL)"; echo SN-FAILED; exit 1
  fi
  echo "==== SN-BAKE-$CELL-OK ($(grep -o '"numWkSlicesPerDim_":{[^}]*}' "$BAKE"/*.json | head -1)) ===="

  echo "==== SN-BINDS-$CELL $(date -u +%FT%TZ) ===="
  if ! BINDS=$(python3 "$PROBE/sn_binds.py" "$BAKE" "$CASE"); then
    echo "SN-BINDS-FAILED ($CASE)"; echo SN-FAILED; exit 1
  fi
  echo "==== SN-BINDS-$CELL: $BINDS"

  echo "==== SN-DXP-$CELL $(date -u +%FT%TZ) ===="
  if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
    echo "SN-DXP-FAILED ($CELL)"; tail -5 "$BAKE/dxp.log"; echo SN-FAILED; exit 1
  fi
  echo "SN-DXP-$CELL-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

  echo "==== SN-RUN-$CELL-START $(date -u +%FT%TZ) ===="
  # shellcheck disable=SC2086
  SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" $BINDS \
    > "$PROBE/sn-run-$CELL.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SN-RUN-FAILED ($CELL rc=$rc)"; tail -20 "$PROBE/sn-run-$CELL.log"; echo SN-FAILED; exit 1
  fi
  tail -8 "$PROBE/sn-run-$CELL.log"
done

# ── leg 6: the verdicts, on the pod — both cells ──
echo "==== SN-CHECK $(date -u +%FT%TZ) ===="
rc=0
for CELL in $CELLS; do
  CASE=${CELL#sample_noise_}
  python3 "$PROBE/sn_stats.py" "$CASE" || rc=1
done
echo "==== sn-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo SN-FAILED; exit 1; fi
echo SN-DONE
