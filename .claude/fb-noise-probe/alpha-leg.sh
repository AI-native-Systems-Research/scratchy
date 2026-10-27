#!/bin/bash
# front-b-noise α CELL LEG (nickm2, auditor order 2026-10-27): int32tofp32
# over [2, 32] i32 — ONE STICK PER ROW — through the scatter harness,
# bit-for-bit against host `as f32`. Decides the convert door's device law:
# convert_opspec's hardcoded 64-elem StickExtent::<Fp16> basis vs its own
# stated wider-stick law (32 elems for the i32→f32 pair).
#
# Method: bake the dense i32 add at [2, 32] (intmr_alpha_base — the door
# CANNOT bake the convert at this width; that refusal IS the wall), mutate
# the baked SDSC off-tree per B1's i2f_g1 recipe (alpha_cell.py mutate:
# opFuncName -> int32tofp32, unary input list, Tensor2 -> IEEE_FP32), dxp,
# run. The CONTROL runs the unmutated base add first — same geometry, same
# staging — so a failure on the i2f cell is the convert's slab law, never the
# one-stick-per-row geometry. Verdict ON THE POD (alpha_cell.py check ->
# alpha-verdict.json); every pre-compare failure is recorded via `fail`,
# never silently skipped. Markers: ALPHA-DONE / ALPHA-FAILED. Log:
# /work/alpha-leg.log
set -u

LOG=/work/alpha-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

BASE=intmr_alpha_base
BAKE=/work/int-alpha-base
CELL=/work/alpha-i2f

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== alpha-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo ALPHA-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo ALPHA-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo ALPHA-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo ALPHA-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== ALPHA-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo ALPHA-BUILD-FAILED; echo ALPHA-FAILED; exit 1
fi
echo "==== ALPHA-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged inputs (stick-scattered) ──
echo "==== ALPHA-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/alpha_cell.py" stage || { echo ALPHA-INPUTS-FAILED; echo ALPHA-FAILED; exit 1; }

# ── leg 3: the base bake — the door's own emission at [2, 32] ──
echo "==== ALPHA-BAKE-START $(date -u +%FT%TZ) ===="
rm -rf "$BAKE"
if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$BASE" "$BAKE"; then
  python3 "$PROBE/alpha_cell.py" fail bake "the door refused the BASE ADD at [2,32] — see /work/alpha-leg.log"
  echo ALPHA-BAKE-FAILED; echo ALPHA-FAILED; exit 1
fi
echo "==== ALPHA-BAKE-OK ($(grep -o '"numWkSlicesPerDim_":{[^}]*}' "$BAKE"/sdsc_0.json | head -1)) ===="

# ── leg 4: the mutation (B1 i2f_g1 recipe, off-tree) ──
echo "==== ALPHA-MUTATE $(date -u +%FT%TZ) ===="
python3 "$PROBE/alpha_cell.py" mutate "$BAKE" "$CELL" || { echo ALPHA-MUTATE-FAILED; echo ALPHA-FAILED; exit 1; }

# ── leg 5: dxp — the scheduler's own acceptance of the 32-elem slab ──
echo "==== ALPHA-DXP $(date -u +%FT%TZ) ===="
if ! "$DXP" -d "$CELL" -b sentient > "$CELL/dxp.log" 2>&1; then
  echo "ALPHA-DXP-REFUSED (rc=$?) — the 32-elem convert slab:"
  tail -15 "$CELL/dxp.log"
  python3 "$PROBE/alpha_cell.py" fail dxp "$(tail -15 "$CELL/dxp.log")"
  echo ALPHA-FAILED; exit 1
fi
echo "ALPHA-DXP-OK ($(stat -c%s "$CELL/spyreCodeDir/init_binary.bin") B image)"

# ── leg 6: the runs — CONTROL first, then the cell ──
echo "==== ALPHA-RUN-START $(date -u +%FT%TZ) ===="
if ! SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
    t0="$PROBE/in/alpha_t00_a.bin" raw:t0 \
    t1="$PROBE/in/alpha_t01_b.bin" raw:t1 \
    --out raw:t2="$PROBE/out_alpha_control.bin" \
    > "$PROBE/alpha-run-control.log" 2>&1; then
  echo "ALPHA-RUN-FAILED control (rc=$?)"; tail -20 "$PROBE/alpha-run-control.log"
  python3 "$PROBE/alpha_cell.py" fail run-control "$(tail -20 "$PROBE/alpha-run-control.log")"
  echo ALPHA-FAILED; exit 1
fi
tail -4 "$PROBE/alpha-run-control.log"

if ! SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$CELL" \
    t0="$PROBE/in/alpha_t00_a.bin" raw:t0 \
    --out raw:t2="$PROBE/out_alpha_i2f.bin" \
    > "$PROBE/alpha-run-i2f.log" 2>&1; then
  echo "ALPHA-RUN-FAILED i2f (rc=$?)"; tail -20 "$PROBE/alpha-run-i2f.log"
  python3 "$PROBE/alpha_cell.py" fail run-i2f "$(tail -20 "$PROBE/alpha-run-i2f.log")"
  echo ALPHA-FAILED; exit 1
fi
tail -4 "$PROBE/alpha-run-i2f.log"

# ── leg 7: the verdict, on the pod ──
echo "==== ALPHA-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/alpha_cell.py" check
rc=$?
echo "==== alpha-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo ALPHA-FAILED; exit 1; fi
echo ALPHA-DONE
