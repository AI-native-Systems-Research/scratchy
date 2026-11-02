#!/bin/bash
# front-b-noise INT-BCAST card leg (nickm2): the Col-broadcast integer add
# probe end to end — auditor order 2026-10-26, ruling (b). Tree:
# /work/front-b-noise — THIS session's worktree (uncommitted work), rsynced
# from the Mac (never cloned: nothing is pushed). Bake: ON THE POD, from that
# tree, KTIR_WHOLE=1 through the whole-function door, WITH the integer door's
# broadcast admission (the probe measures exactly that emission). Runner:
# bundle_run_dev at release in the WARM target dir /work/pr916/head/target
# (one target dir, shared; nothing else runs). Outputs read back VERBATIM
# through the tree's own `raw:` output spelling. Verdict ON THE POD
# (intb_stats.py: bit-for-bit against the host wrapping add); the number
# lands in intb-verdict.json. Markers: INTB-DONE / INTB-FAILED. Log:
# /work/intb-leg.log
set -u

LOG=/work/intb-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone
BAKE=$PROBE/int-bcast

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intb-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTB-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTB-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTB-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTB-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== INTB-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTB-BUILD-FAILED; echo INTB-FAILED; exit 1
fi
echo "==== INTB-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the bake, ON THE POD, from THIS tree, through the whole-function door
#    (with the integer door's Col/Mb broadcast admission — the form under test) ──
echo "==== INTB-BAKE-START $(date -u +%FT%TZ) ===="
rm -rf "$BAKE"
if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- int_bcast "$BAKE"; then
  echo INTB-BAKE-FAILED; echo INTB-FAILED; exit 1
fi
echo "==== INTB-BAKE-OK $(date -u +%FT%TZ) ===="

# ── leg 3: the staged inputs (deterministic, no RNG) ──
echo "==== INTB-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intb_inputs.py" || { echo INTB-INPUTS-FAILED; echo INTB-FAILED; exit 1; }

# ── leg 4: dxp compile (bundle_run_dev expects <dir>/spyreCodeDir/ to exist) ──
echo "==== INTB-DXP $(date -u +%FT%TZ) ===="
if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
  echo INTB-DXP-FAILED; tail -5 "$BAKE/dxp.log"; echo INTB-FAILED; exit 1
fi
echo "INTB-DXP-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

# ── leg 5: the card run — a and b staged raw, the sum read back raw ──
echo "==== INTB-RUN-START $(date -u +%FT%TZ) ===="
SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
  t0="$PROBE/in/t00_a.bin" raw:t0 \
  t1="$PROBE/in/t01_b.bin" raw:t1 \
  --out raw:t2="$PROBE/out_o.bin" \
  > "$PROBE/intb-run.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "INTB-RUN-FAILED (rc=$rc)"; tail -20 "$PROBE/intb-run.log"; echo INTB-FAILED; exit 1
fi
tail -8 "$PROBE/intb-run.log"

# ── leg 6: the verdict, on the pod — bit for bit against the host wrapping add ──
echo "==== INTB-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intb_stats.py"
rc=$?
echo "==== intb-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo INTB-FAILED; exit 1; fi
echo INTB-DONE
