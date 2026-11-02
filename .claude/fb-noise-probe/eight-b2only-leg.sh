#!/bin/bash
# front-b-noise 8B FP8 B2-ONLY RELEG (nickm2) — the auditor's order
# 2026-10-27, after the rebase onto 8143cf0ebe (the scatter commit): "rebase
# onto 8143cf0ebe now, rebuild 8b, and rerun B2 only. If B2 matches main's
# batched, it's done. If it's still garbage, it's the 8b-specific l_sticks = 2
# class, and you localize it with argmax-coder's calibrated instrument
# (SEN-decoded, by placement)."
# One part: 2-CONCURRENT greedy (temp 0, 64 tokens, the same prompt twice) —
# the branch's own batched decode against MAIN's own 2-concurrent greedy texts
# (main8-b2-a/b.json, banked by main8-ref-leg.sh; the batched-numerics
# reference, never solo). The pre-rebase b2 verdict (banked, gate FAILED) read
# degenerate texts from the first shared rows=2 rung; this leg answers whether
# the rebased tree still does.
# Tree: /work/front-b-noise (the REBASED tip — rsync before firing). Build:
# ON THE POD, release, front-b's OWN warm target dir /work/front-b/target.
# Server port 8793, debug log for the rung evidence. Markers (kept identical
# so target/main8-watch.sh's grep matches): EIGHT-DONE / EIGHT-FAILED. Log:
# /work/eight-b2only-leg.log — a FRESH log file, so no earlier run's terminal
# markers can stale-cursor a watch.
set -u

LOG=/work/eight-b2only-leg.log
TREE=/work/front-b-noise
TDIR=/work/front-b/target
PROBE=/work/fb-noise-probe
SCR=$TDIR/release/scr
PORT=8793
PROMPT="Write two sentences about the ocean."

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== eight-b2only-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo EIGHT-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr serve|scr chat' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo EIGHT-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo EIGHT-FAILED; exit 1; }
[ -s "$PROBE/main8-b2-a.json" ] || { echo "no main 8b b2 reference (run main8-ref-leg first)"; echo EIGHT-FAILED; exit 1; }
[ -s "$PROBE/main8-b2-b.json" ] || { echo "no main 8b b2 reference b (run main8-ref-leg first)"; echo EIGHT-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo EIGHT-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: the REBASED tree's 8b chat stack (release, warm dir) ──
echo "==== EIGHT-B2-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel; then
  echo EIGHT-B2-BUILD-FAILED; echo EIGHT-FAILED; exit 1
fi
echo "==== EIGHT-B2-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the server, at debug so the rung lines land in the log ──
SRVLOG=$PROBE/eight-b2-server.log
echo "==== EIGHT-B2-SERVE-START $(date -u +%FT%TZ) ===="
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-8b-instruct-FP8-dynamic --device spyre \
  --port $PORT --log-level debug > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 240); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  # A DEAD SERVER ENDS THE LEG — never poll a corpse for 20 minutes.
  if ! kill -0 $SRVPID 2>/dev/null; then break; fi
  sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  if ! kill -0 $SRVPID 2>/dev/null; then
    echo "EIGHT-B2-SERVE-DIED"; tail -30 "$SRVLOG"; echo EIGHT-FAILED; exit 1
  fi
  echo "EIGHT-B2-SERVE-NEVER-READY"; kill $SRVPID 2>/dev/null; tail -30 "$SRVLOG"; echo EIGHT-FAILED; exit 1
fi
echo "==== EIGHT-B2-SERVE-READY $(date -u +%FT%TZ) ===="

cleanup() { kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null; }
trap cleanup EXIT

ask() {  # ask <seed> <temp> <max-tokens> <outfile>
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":$2,\"seed\":$1,\"max_tokens\":$3}" \
    > "$4"
}

# ── the part: 2-CONCURRENT greedy — vs MAIN's own 2-concurrent greedy ──
echo "==== EIGHT-B2-START $(date -u +%FT%TZ) ===="
ask 0 0.0 64 "$PROBE/eight-b2-a.json" &
BA=$!
ask 0 0.0 64 "$PROBE/eight-b2-b.json" &
BB=$!
wait $BA $BB
echo "==== EIGHT-B2-DONE $(date -u +%FT%TZ) ===="

cleanup
trap - EXIT

# ── the verdict, on the pod ──
echo "==== EIGHT-B2-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/eight_b2_stats.py"
rc=$?
echo "==== eight-b2only-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo EIGHT-FAILED; exit 1; fi
echo EIGHT-DONE
