#!/bin/bash
# Front A legalize byte-identity dump: compile every production kernel (except
# sample_fwd.py) through the splice's own ladder and dump the post-to_ktir module
# text. Run twice (current legalize vs the pre-change file swapped in), then diff
# the two dump dirs externally. Markers: FA-DUMP-DONE / FA-DUMP-FAILED.
# Logs: /work/front-a-dump.log ; dumps: /work/front-a-sampling/target/spyre-ktir-dump/
set -u

LOG=/work/front-a-dump.log
TREE=/work/front-a-sampling
TDIR=/work/pr916/head/target

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== front-a-dump start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo FA-DUMP-FAILED; exit 1
    fi
  done
fi

cd "$TREE" || { echo "no tree"; echo FA-DUMP-FAILED; exit 1; }
export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export SKIP_SENDNN_CXX=1

if cargo test -p scratchy-triton-splice --lib production_ktir_dump -- --ignored; then
  echo "==== FA-DUMP-OK $(date -u +%FT%TZ) ===="
  echo FA-DUMP-DONE
else
  echo "==== FA-DUMP-ERR $(date -u +%FT%TZ) ===="
  echo FA-DUMP-FAILED
fi
