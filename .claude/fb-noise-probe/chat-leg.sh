#!/bin/bash
# front-b-noise SEEDED CHAT card leg (nickm2) — the auditor's GO, all five
# parts, through `scr serve` so the server log carries the evidence:
#   1. temp 1.0, the SAME seed (1234) twice          -> texts identical.
#   2. a DIFFERENT seed (9999)                       -> text differs.
#   3. 2-CONCURRENT, seeds 1234 + 9999               -> each request's text
#      equals its own solo seeded text (the batched decode rung at rows=2 —
#      wall 9's multi-row Col seat).
#   4. greedy (temp 0)                               -> equals main's greedy
#      text (main-ref-leg's main-greedy.run; the fold must not move greedy).
#   5. 200 seeded single-token draws at temp 1.0     -> each token's frequency
#      vs softmax(main's host logits) — top-10 within sampling error
#      (chat_stats.py reads main's logits-0000.bin).
# The server runs at --log-level debug so SEAT_BIND (the seat binding the
# seeds) and the PREFILL_PATH=batched / decode-batch rung lines are in the
# log. Tree: /work/front-b-noise (this branch). Build: ON THE POD, release,
# WARM target dir /work/pr916/head/target. Server port 8791. Markers:
# CHAT-DONE / CHAT-FAILED. Log: /work/chat-leg.log
set -u

LOG=/work/chat-leg.log
TREE=/work/front-b-noise
# ── front-b's OWN target dir (auditor order 2026-10-27: the shared
#    /work/pr916/head/target is a hazard — another tree's rlibs can be
#    served as fresh). Cold once, then warm. ──
TDIR=/work/front-b/target
PROBE=/work/fb-noise-probe
REF=/work/fb-chat-main
SCR=$TDIR/release/scr
PORT=8791
PROMPT="Write two sentences about the ocean."
DRAWS=200
SEED0=100000

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== chat-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo CHAT-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo CHAT-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo CHAT-FAILED; exit 1; }
[ -s "$REF/main-greedy.run" ] || { echo "no main reference at $REF (run main-ref-leg first)"; echo CHAT-FAILED; exit 1; }
[ -s "$REF/logits-0000.bin" ] || { echo "no main logits dump at $REF (run main-ref-leg first)"; echo CHAT-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo CHAT-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: this branch's chat stack with the server (release, warm dir) ──
echo "==== CHAT-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel; then
  echo CHAT-BUILD-FAILED; echo CHAT-FAILED; exit 1
fi
echo "==== CHAT-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the server, at debug so the seat + rung lines land in the log ──
SRVLOG=$PROBE/chat-server.log
echo "==== CHAT-SERVE-START $(date -u +%FT%TZ) ===="
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-2b-instruct-FP8-dynamic --device spyre \
  --port $PORT --log-level debug > "$SRVLOG" 2>&1 &
SRVPID=$!
# wait for readiness (the health/models endpoint answers)
for i in $(seq 1 120); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  echo "CHAT-SERVE-NEVER-READY"; kill $SRVPID 2>/dev/null; tail -30 "$SRVLOG"; echo CHAT-FAILED; exit 1
fi
echo "==== CHAT-SERVE-READY $(date -u +%FT%TZ) ===="

cleanup() { kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null; }
trap cleanup EXIT

# one request: ask <seed> <temperature> <max-tokens> <outfile>
ask() {  # ask <seed> <temp> <max-tokens> <outfile>
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":$2,\"seed\":$1,\"max_tokens\":$3}" \
    > "$4"
}

# ── part 1: the SAME seed twice at temp 1.0 ──
echo "==== CHAT-P1-START $(date -u +%FT%TZ) ===="
ask 1234 1.0 64 "$PROBE/chat-p1-seed1234-a.json"
ask 1234 1.0 64 "$PROBE/chat-p1-seed1234-b.json"
echo "==== CHAT-P1-DONE $(date -u +%FT%TZ) ===="

# ── part 2: a DIFFERENT seed ──
echo "==== CHAT-P2-START $(date -u +%FT%TZ) ===="
ask 9999 1.0 64 "$PROBE/chat-p2-seed9999.json"
echo "==== CHAT-P2-DONE $(date -u +%FT%TZ) ===="

# ── part 3: 2-CONCURRENT, the two seeds at once ──
echo "==== CHAT-P3-START $(date -u +%FT%TZ) ===="
ask 1234 1.0 64 "$PROBE/chat-p3-seed1234.json" &
P3A=$!
ask 9999 1.0 64 "$PROBE/chat-p3-seed9999.json" &
P3B=$!
wait $P3A $P3B
echo "==== CHAT-P3-DONE $(date -u +%FT%TZ) ===="

# ── part 4: greedy (temp 0) via the server (the in-process twin runs after
#    the server is down — two processes must not hold the card at once) ──
echo "==== CHAT-P4-START $(date -u +%FT%TZ) ===="
ask 0 0.0 64 "$PROBE/chat-p4-greedy.json"
echo "==== CHAT-P4-DONE $(date -u +%FT%TZ) ===="

# ── part 5: 200 seeded single-token draws ──
echo "==== CHAT-P5-START $(date -u +%FT%TZ) ===="
rm -f "$PROBE/chat-p5-draws.txt"
i=0
while [ "$i" -lt "$DRAWS" ]; do
  SEED=$((SEED0 + i))
  ask "$SEED" 1.0 1 "$PROBE/chat-p5-draw.json" || { echo "CHAT-P5-FAILED (draw $i)"; echo CHAT-FAILED; exit 1; }
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r["choices"][0]["message"]["content"]))' \
    "$PROBE/chat-p5-draw.json" >> "$PROBE/chat-p5-draws.txt" || { echo "CHAT-P5-PARSE-FAILED (draw $i)"; cat "$PROBE/chat-p5-draw.json"; echo CHAT-FAILED; exit 1; }
  i=$((i + 1))
done
echo "==== CHAT-P5-DONE ($DRAWS draws, $(date -u +%FT%TZ)) ===="

cleanup
trap - EXIT

# ── part 4b: the in-process greedy twin, AFTER the server released the card ──
echo "==== CHAT-P4B-START $(date -u +%FT%TZ) ===="
"$SCR" chat -m RedHatAI/granite-3.1-2b-instruct-FP8-dynamic --device spyre \
  -q "$PROMPT" --temperature 0.0 --max-tokens 64 \
  > "$PROBE/chat-p4-greedy-chat.run" 2> "$PROBE/chat-p4-greedy-chat.err"
echo "==== CHAT-P4B-RC rc=$? $(date -u +%FT%TZ) ===="

# ── phase B (DIAGNOSTIC, never a gate): the same parts 1 and 3 with prefix
#    caching OFF — separates a cache-path first-token salt difference from a
#    seat-law (row word) one. The verdict reads these as diagnostics only. ──
echo "==== CHAT-B-SERVE-START $(date -u +%FT%TZ) ===="
SRVLOGB=$PROBE/chatb-server.log
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-2b-instruct-FP8-dynamic --device spyre \
  --port $PORT --log-level debug --no-prefix-caching > "$SRVLOGB" 2>&1 &
SRVPID=$!
for i in $(seq 1 120); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  sleep 5
done
if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  echo "==== CHAT-B-SERVE-READY $(date -u +%FT%TZ) ===="
  ask 1234 1.0 64 "$PROBE/chatb-p1-seed1234-a.json"
  ask 1234 1.0 64 "$PROBE/chatb-p1-seed1234-b.json"
  echo "==== CHAT-B-P1-DONE $(date -u +%FT%TZ) ===="
  ask 1234 1.0 64 "$PROBE/chatb-p3-seed1234.json" &
  B3A=$!
  ask 9999 1.0 64 "$PROBE/chatb-p3-seed9999.json" &
  B3B=$!
  wait $B3A $B3B
  echo "==== CHAT-B-P3-DONE $(date -u +%FT%TZ) ===="
  kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null
else
  echo "CHAT-B-SERVE-NEVER-READY (diagnostic skipped)"; kill $SRVPID 2>/dev/null
fi

# ── the verdict, on the pod ──
echo "==== CHAT-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/chat_stats.py"
rc=$?
echo "==== chat-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo CHAT-FAILED; exit 1; fi
echo CHAT-DONE
