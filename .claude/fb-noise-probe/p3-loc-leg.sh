#!/bin/bash
# front-b-noise P3 LOCALIZATION leg (nickm2) — the auditor's order
# 2026-10-27: "Localize, ONE run, with seed 1234 solo and 1234+9999
# concurrent: for each decode step, compare the pair's request-1234 seat
# words (seed_lo, seed_hi, step, gate, row) and its picked token against
# the solo run's, and find the first step where they differ."
#
# ONE server, ONE run: a solo seed-1234 request, then the 1234+9999
# concurrent pair — the same geometry as chat-leg part 3, so the fork
# reproduces under measurement. The server log is sliced by line count
# (solo = [L0,L1), pair = [L1,L2)); p3_localize.py parses the SEAT_BIND +
# SEAT_READBACK lines (per-row words, pick, argmax per launch) and writes
# the verdict. No gates here — this is a localization; a reproduced fork
# is the RESULT. Markers: P3LOC-DONE / P3LOC-FAILED. Log: /work/p3-loc-leg.log
set -u

LOG=/work/p3-loc-leg.log
TREE=/work/front-b-noise
TDIR=/work/front-b/target
PROBE=/work/fb-noise-probe
SCR=$TDIR/release/scr
PORT=8791
PROMPT="Write two sentences about the ocean."

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== p3-loc start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo P3LOC-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo P3LOC-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo P3LOC-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo P3LOC-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── warm build (the tree is at the chat leg's HEAD — no code has moved
#    since; this is a no-op freshness check, never a cold build) ──
echo "==== P3LOC-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel; then
  echo P3LOC-BUILD-FAILED; echo P3LOC-FAILED; exit 1
fi
echo "==== P3LOC-BUILD-OK $(date -u +%FT%TZ) ===="

# ── the server, at debug so SEAT_BIND + SEAT_READBACK land in the log ──
SRVLOG=$PROBE/p3-loc-server.log
: > "$SRVLOG"
echo "==== P3LOC-SERVE-START $(date -u +%FT%TZ) ===="
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-2b-instruct-FP8-dynamic --device spyre \
  --port $PORT --log-level debug > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 120); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  echo "P3LOC-SERVE-NEVER-READY"; kill $SRVPID 2>/dev/null; tail -30 "$SRVLOG"; echo P3LOC-FAILED; exit 1
fi
echo "==== P3LOC-SERVE-READY $(date -u +%FT%TZ) ===="

cleanup() { kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null; }
trap cleanup EXIT

ask() {  # ask <seed> <temp> <max-tokens> <outfile>
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":$2,\"seed\":$1,\"max_tokens\":$3}" \
    > "$4"
}

# ── phase 1: seed 1234 SOLO — the reference sequence ──
L0=$(wc -l < "$SRVLOG")
echo "==== P3LOC-SOLO-START L0=$L0 $(date -u +%FT%TZ) ===="
ask 1234 1.0 64 "$PROBE/p3-loc-solo1234.json"
L1=$(wc -l < "$SRVLOG")
echo "==== P3LOC-SOLO-DONE L1=$L1 $(date -u +%FT%TZ) ===="

# ── phase 2: the 1234+9999 CONCURRENT pair — chat-leg part 3's geometry,
#    1234 fired first exactly as there ──
echo "==== P3LOC-PAIR-START $(date -u +%FT%TZ) ===="
ask 1234 1.0 64 "$PROBE/p3-loc-pair1234.json" &
PA=$!
ask 9999 1.0 64 "$PROBE/p3-loc-pair9999.json" &
PB=$!
wait $PA $PB
L2=$(wc -l < "$SRVLOG")
echo "==== P3LOC-PAIR-DONE L2=$L2 $(date -u +%FT%TZ) ===="

cleanup
trap - EXIT

echo "$L0 $L1 $L2" > "$PROBE/p3-loc-slices.txt"
for f in "$PROBE/p3-loc-solo1234.json" "$PROBE/p3-loc-pair1234.json" "$PROBE/p3-loc-pair9999.json"; do
  [ -s "$f" ] || { echo "P3LOC-REPLY-EMPTY ($f)"; echo P3LOC-FAILED; exit 1; }
done

# ── the verdict, on the pod ──
echo "==== P3LOC-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/p3_localize.py"
rc=$?
echo "==== p3-loc end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo P3LOC-FAILED; exit 1; fi
echo P3LOC-DONE
