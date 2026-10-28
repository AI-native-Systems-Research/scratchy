#!/bin/bash
# Front A compile check: cargo check the sampler tree at /work/front-a-sampling
# (the spyre-device-sampling worktree, rsynced from the Mac), riding the WARM
# target dir /work/pr916/head/target. Markers: FA-CHECK-DONE / FA-CHECK-FAILED.
# Logs: /work/front-a-check.log
set -u

LOG=/work/front-a-check.log
TREE=/work/front-a-sampling
TDIR=/work/pr916/head/target

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== front-a-check start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo FA-CHECK-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo FA-CHECK-FAILED; exit 1
fi

cd "$TREE" || { echo "no tree"; echo FA-CHECK-FAILED; exit 1; }
export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export SKIP_SENDNN_CXX=1
export HF_HOME=/work/.cache/huggingface
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

echo "==== FA-CHECK-START $(date -u +%FT%TZ) (granite-2b fp8; mint on every decode rung; fp32 broadcast arm; Fp16Sticks footprint law) ===="
if cargo check -p scratchy-cli --features spyre-hw,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel \
   && cargo test -p triton-ktir legalize_types \
   && cargo test -p ktir-superdsc \
   && cargo test -p scratchy-target-spyre lower_subtile_tape_to_superdsc; then
  echo "==== FA-CHECK-OK $(date -u +%FT%TZ) ===="
  echo FA-CHECK-DONE
else
  echo "==== FA-CHECK-ERR $(date -u +%FT%TZ) ===="
  echo FA-CHECK-FAILED
fi
