#!/bin/bash
# Front A — card cells: the device-picked greedy chat, twice, md5 both runs.
cd /work/front-a-sampling || { echo "FA-CARD-FAILED no-tree"; exit 1; }
BIN=/work/pr916/head/target/release/scr
echo "FA-CARD-START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$BIN" chat -m granite-3.1-2b-instruct --device spyre -q "The capital of France is" > /work/fa-card-run1.txt 2>&1
rc1=$?
"$BIN" chat -m granite-3.1-2b-instruct --device spyre -q "The capital of France is" > /work/fa-card-run2.txt 2>&1
rc2=$?
echo "FA-CARD-RUN1 rc=$rc1"
echo "FA-CARD-RUN2 rc=$rc2"
md5sum /work/fa-card-run1.txt /work/fa-card-run2.txt
echo "----- run1 -----"
cat /work/fa-card-run1.txt
echo "----- run2 -----"
cat /work/fa-card-run2.txt
if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
  echo "FA-CARD-FAILED"
  exit 1
fi
echo "FA-CARD-DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
