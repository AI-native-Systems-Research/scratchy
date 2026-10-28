#!/bin/bash
# front-b-noise INT-MUL-BCAST card leg (nickm2): the Col-broadcast integer
# multiply probe end to end — auditor order 2026-10-26, the wall-5 (B) ruling
# ("If (B) still has a muli by a [1,1] broadcast (hc1/hc2/hc3), card-leg that
# one op on nickm2, the same probe form as wall 3, before you admit it").
# Tree: /work/front-b-noise — THIS session's worktree (uncommitted work),
# rsynced from the Mac (never cloned: nothing is pushed). Bake: ON THE POD,
# from that tree, KTIR_WHOLE=1 through the whole-function door, WITH the
# integer door's Mul broadcast admission (the probe measures exactly that
# emission). Runner: bundle_run_dev at release in the WARM target dir
# /work/pr916/head/target (one target dir, shared; nothing else runs).
# Outputs read back VERBATIM through the tree's own `raw:` output spelling.
# Verdict ON THE POD (intm_stats.py: bit-for-bit against the host wrapping
# multiply); the number lands in intm-verdict.json. Markers: INTM-DONE /
# INTM-FAILED. Log: /work/intm-leg.log
set -u

LOG=/work/intm-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone
BAKE=$PROBE/int-mul-bcast

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intm-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTM-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTM-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTM-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTM-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== INTM-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTM-BUILD-FAILED; echo INTM-FAILED; exit 1
fi
echo "==== INTM-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the bake, ON THE POD, from THIS tree, through the whole-function door
#    (with the integer door's Mul+Col broadcast admission — the form under test) ──
echo "==== INTM-BAKE-START $(date -u +%FT%TZ) ===="
rm -rf "$BAKE"
if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- intmul_bcast "$BAKE"; then
  echo INTM-BAKE-FAILED; echo INTM-FAILED; exit 1
fi
echo "==== INTM-BAKE-OK $(date -u +%FT%TZ) ===="

# ── leg 3: the staged inputs (deterministic, no RNG) ──
echo "==== INTM-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intm_inputs.py" || { echo INTM-INPUTS-FAILED; echo INTM-FAILED; exit 1; }

# ── leg 4: dxp compile (bundle_run_dev expects <dir>/spyreCodeDir/ to exist) ──
echo "==== INTM-DXP $(date -u +%FT%TZ) ===="
if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
  echo INTM-DXP-FAILED; tail -5 "$BAKE/dxp.log"; echo INTM-FAILED; exit 1
fi
echo "INTM-DXP-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

# ── leg 5: the card run — a and b staged raw, the product read back raw ──
echo "==== INTM-RUN-START $(date -u +%FT%TZ) ===="
SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
  t0="$PROBE/in/t00_a.bin" raw:t0 \
  t1="$PROBE/in/t01_b.bin" raw:t1 \
  --out raw:t2="$PROBE/out_o.bin" \
  > "$PROBE/intm-run.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "INTM-RUN-FAILED (rc=$rc)"; tail -20 "$PROBE/intm-run.log"; echo INTM-FAILED; exit 1
fi
tail -8 "$PROBE/intm-run.log"

# ── leg 6: the verdict, on the pod — bit for bit against the host wrapping multiply ──
echo "==== INTM-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intm_stats.py"
rc=$?
echo "==== intm-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo INTM-FAILED; exit 1; fi
echo INTM-DONE
