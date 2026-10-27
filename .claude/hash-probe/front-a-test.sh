#!/bin/bash
# Front A: run the ktir-superdsc test suite alone (the fp32 broadcast arm's pass test +
# the whole crate, so the fp16 broadcast path's byte-identity is proven too). This does
# not need the scratchy-cli model build, so it runs while the run-5 wall work continues.
# Markers: FA-TEST-DONE / FA-TEST-FAILED. Logs: /work/front-a-test.log
set -u

LOG=/work/front-a-test.log
TREE=/work/front-a-sampling
TDIR=/work/pr916/head/target

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== front-a-test start $(date -u +%FT%TZ) ===="

if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo FA-TEST-FAILED; exit 1
    fi
  done
fi

cd "$TREE" || { echo "no tree"; echo FA-TEST-FAILED; exit 1; }
export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export SKIP_SENDNN_CXX=1

if cargo test -p ktir-superdsc; then
  echo "==== FA-TEST-OK $(date -u +%FT%TZ) ===="
  echo FA-TEST-DONE
else
  echo "==== FA-TEST-ERR $(date -u +%FT%TZ) ===="
  echo FA-TEST-FAILED
fi
