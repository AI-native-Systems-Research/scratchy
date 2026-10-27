#!/bin/bash
# front-b-noise 8B FP8 DEVICE-SAMPLING LEG (nickm2) — the auditor's order
# 2026-10-27 on the combined tree cde6ba9c86: "device sampling at
# granite-3.1-8b fp8. Every claim so far is 2b only, and 8b is where
# l_sticks = 2 and the IBR wrap differ." Parts, all through `scr serve` at
# debug so the rung/seat evidence lands in the server log (no silent rung
# skips — a rung that cannot bake/load is a loud build error, and that is a
# finding):
#   G1. greedy solo (temp 0, 64 tokens)   -> equals MAIN's 8b fp8 greedy solo
#       text (main8-ref-leg.sh, main e63c6f8ac8, same prompt/tokens).
#   P1. temp 1.0, the SAME seed (1234) twice -> texts identical.
#   P2. a DIFFERENT seed (9999)             -> text differs.
#   B2. 2-CONCURRENT greedy (temp 0, 64 tokens, the same prompt twice)
#       -> both texts equal MAIN's own 2-concurrent greedy texts (main's
#       batched decode; the same batched-numerics reference, not solo).
# Tree: /work/front-b-noise (branch tip cde6ba9c86). Build: ON THE POD,
# release, front-b's OWN warm target dir /work/front-b/target. Server port
# 8793. Markers: EIGHT-DONE / EIGHT-FAILED. Log: /work/eight-leg.log
set -u

LOG=/work/eight-leg.log
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

echo "==== eight-leg start $(date -u +%FT%TZ) ===="

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
[ -s "$PROBE/main8-solo-greedy.json" ] || { echo "no main 8b reference (run main8-ref-leg first)"; echo EIGHT-FAILED; exit 1; }
[ -s "$PROBE/main8-b2-a.json" ] || { echo "no main 8b b2 reference (run main8-ref-leg first)"; echo EIGHT-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo EIGHT-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: this branch's 8b chat stack (release, warm dir) ──
echo "==== EIGHT-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel; then
  echo EIGHT-BUILD-FAILED; echo EIGHT-FAILED; exit 1
fi
echo "==== EIGHT-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the server, at debug so the seat + rung lines land in the log ──
SRVLOG=$PROBE/eight-server.log
echo "==== EIGHT-SERVE-START $(date -u +%FT%TZ) ===="
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-8b-instruct-FP8-dynamic --device spyre \
  --port $PORT --log-level debug > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 240); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  # A DEAD SERVER ENDS THE LEG — never poll a corpse for 20 minutes (auditor
  # order 2026-10-27: the watch should have fired when the server died).
  if ! kill -0 $SRVPID 2>/dev/null; then break; fi
  sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  if ! kill -0 $SRVPID 2>/dev/null; then
    echo "EIGHT-SERVE-DIED"; tail -30 "$SRVLOG"; echo EIGHT-FAILED; exit 1
  fi
  echo "EIGHT-SERVE-NEVER-READY"; kill $SRVPID 2>/dev/null; tail -30 "$SRVLOG"; echo EIGHT-FAILED; exit 1
fi
echo "==== EIGHT-SERVE-READY $(date -u +%FT%TZ) ===="

cleanup() { kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null; }
trap cleanup EXIT

ask() {  # ask <seed> <temp> <max-tokens> <outfile>
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":$2,\"seed\":$1,\"max_tokens\":$3}" \
    > "$4"
}

# ── part G1: greedy solo, against main's 8b greedy ──
echo "==== EIGHT-G1-START $(date -u +%FT%TZ) ===="
ask 0 0.0 64 "$PROBE/eight-g1-greedy.json"
echo "==== EIGHT-G1-DONE $(date -u +%FT%TZ) ===="

# ── part P1: the SAME seed twice at temp 1.0 ──
echo "==== EIGHT-P1-START $(date -u +%FT%TZ) ===="
ask 1234 1.0 64 "$PROBE/eight-p1-seed1234-a.json"
ask 1234 1.0 64 "$PROBE/eight-p1-seed1234-b.json"
echo "==== EIGHT-P1-DONE $(date -u +%FT%TZ) ===="

# ── part P2: a DIFFERENT seed ──
echo "==== EIGHT-P2-START $(date -u +%FT%TZ) ===="
ask 9999 1.0 64 "$PROBE/eight-p2-seed9999.json"
echo "==== EIGHT-P2-DONE $(date -u +%FT%TZ) ===="

# ── part B2: 2-CONCURRENT greedy — against MAIN's own 2-concurrent greedy ──
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
echo "==== EIGHT-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/eight_stats.py"
rc=$?
echo "==== eight-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo EIGHT-FAILED; exit 1; fi
echo EIGHT-DONE
