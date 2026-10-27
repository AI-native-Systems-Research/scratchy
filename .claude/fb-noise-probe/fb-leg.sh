#!/bin/bash
# front-b-noise card leg (nickm2): the hash-noise probe end to end.
# Tree: /work/front-b-noise — THIS session's worktree (uncommitted work),
# rsynced from the Mac (never cloned: nothing is pushed). Bake: ON THE POD,
# from that tree, KTIR_WHOLE=1 through the whole-function door. Runner:
# bundle_run_dev at release in the WARM target dir /work/pr916/head/target
# (one target dir, shared; nothing else runs). Outputs read back VERBATIM
# through the tree's own `raw:` output spelling (no fp16 codec). Verdicts
# on the pod (fb_stats.py); every number lands in fb-noise-verdict.json.
# Markers: FB-NOISE-DONE / FB-NOISE-FAILED. Log: /work/fb-noise-leg.log
set -u

LOG=/work/fb-noise-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone
BAKE=$PROBE/hash-noise

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== fb-noise-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo FB-NOISE-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo FB-NOISE-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo FB-NOISE-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo FB-NOISE-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== FB-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo FB-BUILD-FAILED; echo FB-NOISE-FAILED; exit 1
fi
echo "==== FB-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the bake, ON THE POD, from THIS tree, through the whole-function door ──
echo "==== FB-BAKE-START $(date -u +%FT%TZ) ===="
rm -rf "$BAKE"
if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- hash_noise "$BAKE"; then
  echo FB-BAKE-FAILED; echo FB-NOISE-FAILED; exit 1
fi
echo "==== FB-BAKE-OK $(date -u +%FT%TZ) ===="

# ── leg 3: the staged input tables (deterministic, no RNG) ──
echo "==== FB-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/fb_inputs.py" || { echo FB-INPUTS-FAILED; echo FB-NOISE-FAILED; exit 1; }

# ── leg 4: dxp compile (bundle_run_dev expects <dir>/spyreCodeDir/ to exist) ──
echo "==== FB-DXP $(date -u +%FT%TZ) ===="
if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
  echo FB-DXP-FAILED; tail -5 "$BAKE/dxp.log"; echo FB-NOISE-FAILED; exit 1
fi
echo "FB-DXP-OK ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

# ── leg 5: the card run — all 14 tables staged raw, all 4 outs read raw ──
echo "==== FB-RUN-START $(date -u +%FT%TZ) ===="
SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
  t0="$PROBE/in/t00_slo.bin" raw:t0 \
  t1="$PROBE/in/t01_shi.bin" raw:t1 \
  t2="$PROBE/in/t02_stp.bin" raw:t2 \
  t3="$PROBE/in/t03_c1.bin" raw:t3 \
  t4="$PROBE/in/t04_c2.bin" raw:t4 \
  t5="$PROBE/in/t05_c3.bin" raw:t5 \
  t6="$PROBE/in/t06_c256.bin" raw:t6 \
  t7="$PROBE/in/t07_lane.bin" raw:t7 \
  t8="$PROBE/in/t08_fm8.bin" raw:t8 \
  t9="$PROBE/in/t09_ftop.bin" raw:t9 \
  t10="$PROBE/in/t10_fzero.bin" raw:t10 \
  t11="$PROBE/in/t11_fhalf.bin" raw:t11 \
  t12="$PROBE/in/t12_fm22.bin" raw:t12 \
  t13="$PROBE/in/t13_cmin.bin" raw:t13 \
  --out raw:t15="$PROBE/out_u.bin" \
  --out raw:t16="$PROBE/out_g.bin" \
  --out raw:t17="$PROBE/out_ui.bin" \
  --out raw:t18="$PROBE/out_gi.bin" \
  > "$PROBE/run.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "FB-RUN-FAILED (rc=$rc)"; tail -20 "$PROBE/run.log"; echo FB-NOISE-FAILED; exit 1
fi
tail -8 "$PROBE/run.log"

# ── leg 6: the verdicts, on the pod ──
echo "==== FB-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/fb_stats.py"
rc=$?
echo "==== fb-noise-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo FB-NOISE-FAILED; exit 1; fi
echo FB-NOISE-DONE
