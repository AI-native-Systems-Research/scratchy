#!/usr/bin/env bash
# One-shot cold-prefill A/B for a PR: fetches the base and candidate refs,
# builds each in its own worktree with its own CARGO_TARGET_DIR, and reports
# order-balanced TTFT deltas per shape. Built for PR #597 (fused MoE act+quant
# + token-block-parallel GDN conv1d) but usable for any prefill-affecting PR.
#
# Shapes (all prefill-bound, max_tokens=1, run at server DEFAULTS — no
# --max-num-seqs, which changes the MTP state reservation):
#   long   - a ~--long-tokens prompt (default 30000; the real-world case is
#            Claude Code's ~30.3k-token startup prompt — pass --prompt-file)
#   short  - a ~10-token chat prompt (decode/small-bucket control)
#   batch  - two concurrent ~4200-token prompts (leg value = mean of the two
#            requests' TTFTs)
#   64|512|1024|2048|4096 - a prompt of exactly that many tokens, one per
#            rung of the default prefill ladder
#
# Measurement rules (review-hardened):
#   - PAIR ORDER BIASES (wired memory released asynchronously after a server
#     exits): legs settle before launch (vm_stat poll back within ~2 GiB of
#     the pre-run baseline, bounded) and reps alternate the starting side,
#     with an EVEN rep count enforced so base-first and candidate-first pairs
#     balance exactly.
#   - THE WHOLE MACHINE IS GATED, per launch: any scr serve/chat/bench,
#     mlx_lm, or cargo/rustc job blocks the next server start (bounded wait).
#   - FRESH SERVER PER LEG: the in-memory prefix cache makes a warm server
#     answer in ~0.09s.
#   - A NON-200 RESPONSE IS A FAILED LEG, NOT A FAST ONE: the server stays
#     healthy through a request panic, so error pages pose as record TTFTs.
#   - The authoritative TTFT is the server log's per-request ttft_ms (single
#     shapes); curl's time_starttransfer is the convenience copy.
#   - Each leg's weights+overhead / kv_budget startup line is printed and
#     compared across legs of a shape — a reservation difference between base
#     and candidate is a confounder, not a result.
#   - The synthetic prompts' real token counts come from the server log
#     ("prompt tokens: N") — filler is word-approximate, the log is exact.
#
# Usage:
#   one-shot (default):   scripts/bench_prefill_ab.sh --pr 597
#     fetches <remote>/main and <remote>/pull/<pr>/head, builds both under
#     .pr-ab-<pr>/ (gitignored), benches, prints the cleanup command.
#   explicit binaries:    scripts/bench_prefill_ab.sh --base <scr> --cand <scr>
#     (both must have the model compiled in; skips the fetch/build)
#   scripts/bench_prefill_ab.sh --pr 597 --reps 4 \
#     [--remote upstream] [--base-ref main] [--cand-ref pull/597/head] \
#     [--model mlx-community/Qwen3.6-35B-A3B-4bit] \
#     [--features 'metal,model/qwen3.6-35b-a3b,quant/mlx,serve,claude'] \
#     [--shapes long,short,batch,64,512,1024,2048,4096] [--prompt-file f]

set -euo pipefail

MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
FEATURES='metal,model/qwen3.6-35b-a3b,quant/mlx,serve,claude'
REPS=4
SHAPES=long,short,batch,64,512,1024,2048,4096
BASE_BIN=
CAND_BIN=
PR=
REMOTE=upstream
BASE_REF=main
PROMPT_FILE=
LONG_TOKENS=30000
while [ $# -gt 0 ]; do
  case "$1" in
    --pr) PR=$2; shift 2 ;;
    --remote) REMOTE=$2; shift 2 ;;
    --base-ref) BASE_REF=$2; shift 2 ;;
    --base) BASE_BIN=$2; shift 2 ;;
    --cand) CAND_BIN=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --features) FEATURES=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --shapes) SHAPES=$2; shift 2 ;;
    --prompt-file) PROMPT_FILE=$2; shift 2 ;;
    --long-tokens) LONG_TOKENS=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# An even rep count is required: odd counts give base-first pairs a majority
# under the order bias, which is not order-balanced.
if [ $(( REPS % 2 )) != 0 ]; then
  echo "--reps must be even (base-first and candidate-first pairs must balance)" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "not run from inside the repo" >&2; exit 2; }
AB="$ROOT/.pr-ab-${PR:-manual}"
TMP="$AB/tmp"
mkdir -p "$TMP"

if [ -n "$BASE_BIN" ] && [ -n "$CAND_BIN" ]; then
  [ -x "$BASE_BIN" ] || { echo "not executable: $BASE_BIN" >&2; exit 2; }
  [ -x "$CAND_BIN" ] || { echo "not executable: $CAND_BIN" >&2; exit 2; }
  BASE_BIN=$(realpath "$BASE_BIN"); CAND_BIN=$(realpath "$CAND_BIN")
elif [ -n "$PR" ]; then
  CAND_REF="pull/$PR/head"
  # fetch + build both refs, each in its own worktree and target dir
  build_ref() { # $1 dir-suffix  $2 git ref
    local sha wt
    sha=$(git -C "$ROOT" fetch --quiet "$REMOTE" "$2" && git -C "$ROOT" rev-parse FETCH_HEAD)
    wt="$AB/$1"
    if [ ! -d "$wt" ]; then
      git -C "$ROOT" worktree add --detach "$wt" "$sha"
    else
      git -C "$wt" fetch --quiet "$REMOTE" "$2" \
        && git -C "$wt" checkout --quiet --detach FETCH_HEAD
    fi
    echo "building $1 at $sha ($2)" >&2
    (cd "$wt" && CARGO_TARGET_DIR="$AB/$1-target" \
       cargo build -F "$FEATURES" --release -p scratchy-cli) >&2
  }
  build_ref base "$BASE_REF"
  build_ref cand "$CAND_REF"
  BASE_BIN="$AB/base-target/release/scr"
  CAND_BIN="$AB/cand-target/release/scr"
  echo "cleanup when done:"
  echo "  git worktree remove --force $AB/base; git worktree remove --force $AB/cand; rm -rf $AB"
else
  echo "need --pr <n>, or both --base and --cand" >&2; exit 2
fi

# --- machine gates -----------------------------------------------------------

wired_bytes() {
  local pages pagesize
  pages=$(vm_stat | awk '/Pages wired down/ {gsub(/\./, "", $4); print $4}')
  pagesize=$(pagesize)
  echo $(( pages * pagesize ))
}
BASELINE_WIRED=$(wired_bytes)

# Wait for every GPU/CPU job that would pollute a leg: scr serve/chat/bench,
# mlx_lm, cargo build/test/clippy (and their rustc children). Bounded.
wait_machine_quiet() {
  local deadline=$(( SECONDS + 600 ))
  while pgrep -f 'scr (serve|chat|bench)' >/dev/null 2>&1 \
     || pgrep -f mlx_lm >/dev/null 2>&1 \
     || pgrep -x cargo >/dev/null 2>&1 \
     || pgrep -x rustc >/dev/null 2>&1; do
    if [ $SECONDS -ge $deadline ]; then
      echo "WARNING: machine still busy after 600s, benching anyway" >&2
      return 0
    fi
    sleep 5
  done
}

# Wired GPU memory is released asynchronously after a server exits: wait for
# it to return within ~2 GiB of the pre-run baseline, bounded at 120s.
wait_wired_settled() {
  local deadline=$(( SECONDS + 120 )) w
  while :; do
    w=$(wired_bytes)
    if [ $(( w - BASELINE_WIRED )) -le $(( 2 * 1024 * 1024 * 1024 )) ]; then
      return 0
    fi
    if [ $SECONDS -ge $deadline ]; then
      echo "WARNING: wired memory not settled ($(( (w - BASELINE_WIRED) / 1024 / 1024 )) MiB over baseline), proceeding" >&2
      return 0
    fi
    sleep 2
  done
}

# --- prompts and requests ----------------------------------------------------

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
# seconds (server-log ttft_ms for singles; mean curl TTFT for batch), and
# appends the leg's memory line to $MEMLOG.
run_leg() { # $1 scr  $2 shape
  local scr=$1 shape=$2 port slog req srv ok=0 t code metric mem
  wait_machine_quiet
  port=$(( 20000 + RANDOM % 20000 ))
  slog=$(mktemp "$TMP/leg.XXXXXX.log")
  req=$(mktemp "$TMP/leg.XXXXXX.json")
  request_json 1 "$PROMPT" > "$req"
  cleanup() { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; rm -f "$slog" "$req"; }
  # server DEFAULTS: no --max-num-seqs (the knob changes the MTP state
  # reservation, so legs would not be comparable)
  "$scr" serve "$MODEL" --host 127.0.0.1 --port "$port" > "$slog" 2>&1 &
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
  # memory confounder: this leg's reservation line, printed and compared
  mem=$(grep -i -m1 'weights+overhead\|kv_budget' "$slog" || true)
  if [ -n "$mem" ]; then
    echo "  mem: $mem" >&2
    # compare on the line without its timestamp, or every leg "differs"
    printf '%s\n' "${mem#*INFO }" >> "$MEMLOG"
  fi
  local url="http://127.0.0.1:$port/v1/messages"
  local hdr='content-type: application/json'
  if [ "$shape" = batch ]; then
    local tmp1 tmp2
    tmp1=$(mktemp "$TMP/b1.XXXXXX"); tmp2=$(mktemp "$TMP/b2.XXXXXX")
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
    short) echo 'Say hi and nothing else.' ;;
    batch) synth_prompt 4200 ;;
    *) synth_prompt "$1" ;;   # rung shapes: 64/512/1024/2048/4096
  esac
}

# --- the sweep ----------------------------------------------------------------

echo "base=$BASE_BIN"
echo "cand=$CAND_BIN"
echo "model=$MODEL reps=$REPS shapes=$SHAPES"
echo "baseline wired: $(( BASELINE_WIRED / 1024 / 1024 )) MiB"
for shape in ${SHAPES//,/ }; do
  PROMPT=$(prompt_for_shape "$shape")
  MEMLOG=$(mktemp "$TMP/mem.$shape.XXXXXX")
  echo "--- shape=$shape ---"
  deltas=()
  for i in $(seq 1 "$REPS"); do
    # odd reps: base first; even reps: candidate first (order-balanced pairs)
    if [ $(( i % 2 )) = 1 ]; then first=$BASE_BIN second=$CAND_BIN fl=base sl=cand
    else first=$CAND_BIN second=$BASE_BIN fl=cand sl=base; fi
    a=$(run_leg "$first" "$shape") || exit 1
    b=$(run_leg "$second" "$shape") || exit 1
    if [ "$fl" = base ]; then
      d=$(python3 -c "print(f'{($b - $a) * 1000:.1f}')")   # cand ran second
    else
      d=$(python3 -c "print(f'{($a - $b) * 1000:.1f}')")   # cand ran first
    fi
    deltas+=("$d")
    echo "rep$i: $fl ${a}s $sl ${b}s delta(cand-base)=${d}ms"
  done
  med=$(printf '%s\n' "${deltas[@]}" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  echo "shape=$shape median delta = ${med}ms over $REPS order-balanced pairs"
  # memory confounder check: every leg of the shape must report the same
  # reservation line, else the comparison is not like-for-like
  if [ "$(sort -u "$MEMLOG" | wc -l | tr -d ' ')" -gt 1 ]; then
    echo "WARNING: memory reservation differs across legs of shape=$shape:" >&2
    sort -u "$MEMLOG" >&2
  fi
  rm -f "$MEMLOG"
done
echo 'DONE'
