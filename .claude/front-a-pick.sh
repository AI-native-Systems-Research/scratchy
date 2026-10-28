#!/bin/bash
# Front A — the pick flip. cargo check (worker) then the granite 2b fp8 release build.
# Detached runner; truth = the FA-PICK-* terminators in this log.
cd /work/front-a-sampling || { echo "FA-PICK-FAILED no-tree"; exit 1; }
export CARGO_TARGET_DIR=/work/pr916/head/target
echo "FA-PICK-START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
export SKIP_SENDNN_CXX=1
cargo check -p scratchy-serving-worker --features spyre-hw > /work/front-a-pick-check.log 2>&1
rc=$?
echo "FA-PICK-CHECK rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$rc" -ne 0 ]; then
  tail -60 /work/front-a-pick-check.log
  echo "FA-PICK-FAILED check"
  exit 1
fi
unset SKIP_SENDNN_CXX
cargo build --release -p scratchy-cli --features spyre-hw,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel > /work/front-a-pick-build.log 2>&1
rc=$?
echo "FA-PICK-BUILD rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$rc" -ne 0 ]; then
  tail -60 /work/front-a-pick-build.log
  echo "FA-PICK-FAILED build"
  exit 1
fi
echo "FA-PICK-DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
