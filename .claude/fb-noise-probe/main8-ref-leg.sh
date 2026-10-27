#!/bin/bash
# front-b-noise 8B FP8 CHAT LEG, PART 0 — MAIN's 8b REFERENCE (nickm2).
# The auditor's 8b order (2026-10-27): "greedy solo against main's 8b fp8 text
# (build main's reference in its own warm tree on nickm2)" and "the same-text
# check at b2 for greedy against main's batched". Main e63c6f8ac8 — the same
# tree that banked the 2b reference — rebuilt at granite-3.1-8b fp8 in its OWN
# warm target dir. Main's spyre worker is always greedy (it never read
# temperature), so every request here is greedy whatever the body says.
# Parts:
#   1. SOLO greedy, 64 tokens (main8-solo-greedy.json).
#   2. 2-CONCURRENT greedy, 64 tokens, the same prompt twice
#      (main8-b2-a.json / main8-b2-b.json) — main's own batched decode.
# Tree: /work/front-b-chatmain (untouched — it carries the never-committed
# logits-dump patch). Build: ON THE POD, release, its own target dir. Server
# port 8792. Markers: MAINREF8B-DONE / MAINREF8B-FAILED. Log:
# /work/main8-ref-leg.log
set -u

LOG=/work/main8-ref-leg.log
TREE=/work/front-b-chatmain
TDIR=/work/front-b-chatmain/target
PROBE=/work/fb-noise-probe
SCR=$TDIR/release/scr
PORT=8792
PROMPT="Write two sentences about the ocean."

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== main8-ref-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo MAINREF8B-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr serve|scr chat' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo MAINREF8B-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no main tree at $TREE"; echo MAINREF8B-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo MAINREF8B-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: main's 8b chat stack (release, its own warm target dir) ──
echo "==== MAINREF8B-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel; then
  echo MAINREF8B-BUILD-FAILED; echo MAINREF8B-FAILED; exit 1
fi
echo "==== MAINREF8B-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: main's server (greedy only — the worker never read temperature) ──
SRVLOG=$PROBE/main8-server.log
echo "==== MAINREF8B-SERVE-START $(date -u +%FT%TZ) ===="
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-8b-instruct-FP8-dynamic --device spyre \
  --port $PORT > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 240); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  echo "MAINREF8B-SERVE-NEVER-READY"; kill $SRVPID 2>/dev/null; tail -30 "$SRVLOG"; echo MAINREF8B-FAILED; exit 1
fi
echo "==== MAINREF8B-SERVE-READY $(date -u +%FT%TZ) ===="

cleanup() { kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null; }
trap cleanup EXIT

ask() {  # ask <outfile>  (greedy; main ignores temperature but the body is well-formed)
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0.0,\"max_tokens\":64}" \
    > "$1"
}

# ── part 1: solo greedy ──
echo "==== MAINREF8B-SOLO-START $(date -u +%FT%TZ) ===="
ask "$PROBE/main8-solo-greedy.json"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(len(r["choices"][0]["message"]["content"]), "chars")' \
  "$PROBE/main8-solo-greedy.json" || { echo "MAINREF8B-SOLO-PARSE-FAILED"; cat "$PROBE/main8-solo-greedy.json"; echo MAINREF8B-FAILED; exit 1; }
echo "==== MAINREF8B-SOLO-DONE $(date -u +%FT%TZ) ===="

# ── part 2: 2-concurrent greedy — main's own batched decode ──
echo "==== MAINREF8B-B2-START $(date -u +%FT%TZ) ===="
ask "$PROBE/main8-b2-a.json" &
PA=$!
ask "$PROBE/main8-b2-b.json" &
PB=$!
wait $PA $PB
for f in a b; do
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(len(r["choices"][0]["message"]["content"]), "chars")' \
    "$PROBE/main8-b2-$f.json" || { echo "MAINREF8B-B2-PARSE-FAILED ($f)"; cat "$PROBE/main8-b2-$f.json"; echo MAINREF8B-FAILED; exit 1; }
done
echo "==== MAINREF8B-B2-DONE $(date -u +%FT%TZ) ===="

cleanup
trap - EXIT
tail -5 "$SRVLOG"
echo "==== main8-ref-leg end $(date -u +%FT%TZ) ===="
echo MAINREF8B-DONE
