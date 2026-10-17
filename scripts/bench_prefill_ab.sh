#!/usr/bin/env bash
# One-shot cold-prefill A/B: alternates two `scr` binaries (base vs candidate)
# on the same model, fresh server per leg, and reports order-balanced TTFT
# deltas per shape. Built to reproduce the prefill-moonshot evidence for
# PR #597 (fused MoE act+quant + token-block-parallel GDN conv1d) but usable
# for any prefill-affecting change.
#
# Shapes (all prefill-bound, max_tokens=1):
#   long   - a ~--long-tokens prompt (default 30000; the real-world case this
#            was built for is Claude Code's ~30.3k-token startup prompt, which
#            is byte-stable across sessions - pass it via --prompt-file)
#   chunk  - a single ~4200-token prompt (one prefill bucket)
#   short  - a ~10-token chat prompt (decode/small-bucket control)
#   batch  - two concurrent ~4200-token prompts at --max-num-seqs 2 (leg value
#            = mean of the two concurrent requests' TTFTs)
#
# Hard-won measurement rules baked in (17 Oct moonshot):
#   - PAIR ORDER BIASES ~±300ms: whichever binary runs second pays server
#     teardown contention, so reps alternate the starting side (odd reps run
#     base first, even reps candidate first). Deltas are per-pair, reported
#     with the median.
#   - FRESH SERVER PER LEG: the in-memory prefix cache makes a warm server
#     answer in ~0.09s - never reuse one across legs.
#   - A NON-200 RESPONSE IS A FAILED LEG, NOT A FAST ONE: the server stays
#     healthy through a request panic, so error pages pose as record TTFTs.
#     Any non-200 fails the run and the server log's panic lines are printed.
#   - The authoritative TTFT is the server log's per-request ttft_ms (single
#     shapes); curl's time_starttransfer is the convenience copy.
#   - The synthetic long prompt's real token count comes from the server log
#     ("prompt tokens: N") - filler is word-approximate, the log is exact.
#
# Usage (both binaries must have the model compiled in, metal backend):
#   cargo build -F metal,model/qwen3.6-35b-a3b,quant/mlx,serve,claude --release
#   scripts/bench_prefill_ab.sh \
#     --base  <worktree-a>/target/release/scr \
#     --cand  <worktree-b>/target/release/scr \
#     [--model mlx-community/Qwen3.6-35B-A3B-4bit] [--reps 3] \
#     [--shapes long,chunk,short,batch] [--prompt-file claude-startup.txt]
#
# Output: per-leg lines on stdout plus a per-shape median; temp files under
# /tmp are removed per leg.

set -euo pipefail

MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
REPS=3
SHAPES=long,chunk,short,batch
BASE=
CAND=
PROMPT_FILE=
LONG_TOKENS=30000
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2 ;;
    --cand) CAND=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --shapes) SHAPES=$2; shift 2 ;;
    --prompt-file) PROMPT_FILE=$2; shift 2 ;;
    --long-tokens) LONG_TOKENS=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$BASE" ] && [ -n "$CAND" ] || { echo "need --base and --cand" >&2; exit 2; }
BASE=$(realpath "$BASE"); CAND=$(realpath "$CAND")
for b in "$BASE" "$CAND"; do
  [ -x "$b" ] || { echo "not executable: $b" >&2; exit 2; }
done

# Refuse to share the GPU with a stray server.
if pgrep -f "scr serve" >/dev/null 2>&1; then
  echo "REFUSED: another 'scr serve' is running; not benching beside it" >&2
  exit 1
fi

# Filler paragraph: 75 words ~ 100 tokens per repetition. Word-approximate on
# purpose; the server log's "prompt tokens:" line is the exact count.
synth_prompt() { # $1 = target tokens
  local para='The quick brown fox jumps over the lazy dog while the server warms its weights and the scheduler queues the first prefill chunk. Every op is a compile-time constant here: the tape was expanded at build time, the buckets were chosen, and the kernels only wait for their barriers. Latency is the only variable left, so we measure it twice, in both orders, because the second server always pays for the first one teardown.'
  local reps=$(( $1 / 95 + 1 ))
  for _ in $(seq 1 "$reps"); do printf '%s ' "$para"; done
}

# request_json <max_tokens> <prompt>  (prompt travels via env, not argv, so
# very long prompts don't hit ARG_MAX)
request_json() {
  MAX_TOK=$1 PROMPT=$2 python3 - <<'PY'
import json, os, sys
body = {
    "model": "scratchy",
    "max_tokens": int(os.environ["MAX_TOK"]),
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
}
sys.stdout.write(json.dumps(body))
PY
}

# run_leg <scr> <shape> -> prints one number: the leg's TTFT metric in
# seconds (server-log ttft_ms for singles; mean curl TTFT for batch).
run_leg() { # $1 scr  $2 shape
  local scr=$1 shape=$2 port slog req srv ok=0 t code metric
  port=$(( 20000 + RANDOM % 20000 ))
  slog=$(mktemp /tmp/pfab.XXXXXX.log)
  req=$(mktemp /tmp/pfab.XXXXXX.json)
  request_json 1 "$PROMPT" > "$req"
  cleanup() { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; rm -f "$slog" "$req"; }
  "$scr" serve "$MODEL" --host 127.0.0.1 --port "$port" \
    --max-num-seqs "$([ "$shape" = batch ] && echo 2 || echo 1)" > "$slog" 2>&1 &
  srv=$!
  trap cleanup RETURN
  for t in $(seq 1 1200); do
    curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/health" && { ok=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 0.5
  done
  if [ "$ok" != 1 ]; then
    echo "SERVER FAILED TO BECOME HEALTHY" >&2; tail -20 "$slog" >&2; return 1
  fi
  local url="http://127.0.0.1:$port/v1/messages"
  local hdr='content-type: application/json'
  if [ "$shape" = batch ]; then
    local c1 c2 tmp1 tmp2
    tmp1=$(mktemp /tmp/pfab.XXXXXX.c1); tmp2=$(mktemp /tmp/pfab.XXXXXX.c2)
    cleanup() {
      kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
      rm -f "$slog" "$req" "$tmp1" "$tmp2"
    }
    curl -s -o /dev/null -w '%{time_starttransfer} %{http_code}\n' -H "$hdr" \
      --data @"$req" "$url" > "$tmp1" &
    local bg=$!
    curl -s -o /dev/null -w '%{time_starttransfer} %{http_code}\n' -H "$hdr" \
      --data @"$req" "$url" > "$tmp2"
    wait "$bg"
    for f in "$tmp1" "$tmp2"; do
      read -r _ code < "$f"
      if [ "$code" != 200 ]; then
        echo "REQUEST FAILED http=$code (fast responses are error pages, not wins)" >&2
        grep -n "panicked\|ERROR" "$slog" | tail -5 >&2
        return 1
      fi
    done
    metric=$(python3 -c "print(f'{(float(open('$tmp1').read().split()[0]) + float(open('$tmp2').read().split()[0])) / 2:.4f}')")
    echo "$metric"
    return 0
  fi
  local out
  out=$(curl -s -o /dev/null -w '%{time_starttransfer} %{http_code}' -H "$hdr" \
    --data @"$req" "$url")
  code=${out##* }
  if [ "$code" != 200 ]; then
    echo "REQUEST FAILED http=$code (fast responses are error pages, not wins)" >&2
    grep -n "panicked\|ERROR" "$slog" | tail -5 >&2
    return 1
  fi
  # authoritative number: the last ttft_ms in the server log
  local ms
  ms=$(grep -o 'ttft_ms[":= ]*[0-9.]*' "$slog" | tail -1 | grep -o '[0-9.]*$')
  [ -n "$ms" ] || { echo "no ttft_ms in server log" >&2; return 1; }
  python3 -c "print(f'{$ms / 1000:.4f}')"
}

prompt_for_shape() { # $1 shape
  case "$1" in
    long)
      if [ -n "$PROMPT_FILE" ]; then cat "$PROMPT_FILE"; else synth_prompt "$LONG_TOKENS"; fi ;;
    chunk|batch) synth_prompt 4200 ;;
    short) echo 'Say hi and nothing else.' ;;
  esac
}

echo "base=$BASE"
echo "cand=$CAND"
echo "model=$MODEL reps=$REPS shapes=$SHAPES"
for shape in ${SHAPES//,/ }; do
  PROMPT=$(prompt_for_shape "$shape")
  echo "--- shape=$shape ---"
  deltas=()
  for i in $(seq 1 "$REPS"); do
    # odd reps: base first; even reps: candidate first (order-balanced pairs)
    if [ $(( i % 2 )) = 1 ]; then first=$BASE second=$CAND fl=base sl=cand
    else first=$CAND second=$BASE fl=cand sl=base; fi
    a=$(run_leg "$first" "$shape") || exit 1
    sleep 3   # let graceful shutdown release GPU residency
    b=$(run_leg "$second" "$shape") || exit 1
    sleep 3
    d=$(python3 -c "print(f'{($b - $a) * 1000:.1f}')")
    deltas+=("$d")
    echo "rep$i: $fl ${a}s $sl ${b}s delta(cand-base)=${d}ms"
  done
  med=$(printf '%s\n' "${deltas[@]}" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  echo "shape=$shape median delta = ${med}ms over $REPS order-balanced pairs"
done
echo 'DONE'
