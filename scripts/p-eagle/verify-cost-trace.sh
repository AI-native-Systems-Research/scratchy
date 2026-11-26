#!/usr/bin/env bash
# A target's verify cost c(R) / c(1) at R rows, probe-free: for each R, one `scr serve` alone on the
# GPU (the n-gram proposer at R - 1 drafts a step; R = 1 serves without), one greedy request whose
# reply copies its prompt (so the proposer drafts R - 1 on most steps), and an Instruments Metal
# System Trace of the server over the whole reply. A decode step is one command buffer; its GPU
# span is the step's cost. Prints, per R, a histogram of the decode command buffers' GPU spans (the
# R-row steps are the slowest cluster: a step whose n-gram found no match runs fewer rows), the
# reply's drafts a drafting step (from /metrics), then c(R) / c(1) at each R's upper mode.
# Wired GPU memory settles back to its baseline before each leg; one server at a time.
# Usage: verify-cost-trace.sh <scr binary> <target repo> <out dir> [R ...]   (default R: 1 2 4 8)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${1:?scr binary}
model=${2:?target repo}
out=${3:?out dir}
shift 3
rows=("$@")
[ ${#rows[@]} -gt 0 ] || rows=(1 2 4 8)
port=${PORT:-8793}
mkdir -p "$out"

wired() { vm_stat | awk '/Pages wired down/ { gsub("\\.", "", $4); print $4 * 16384 }'; }
base_wired=$(wired)
settle() {
  for _ in $(seq 1 300); do
    [ $(($(wired) - base_wired)) -lt $((2 << 30)) ] && return
    timeout 2 tail -f /dev/null || true
  done
  echo "wired GPU memory never settled"
}
metric() {
  curl -sf "http://127.0.0.1:$port/metrics" |
    awk -v m="$1" '{ n = $1; sub(/\{.*/, "", n) } n == m || n == m "_total" { s += $NF } END { print s + 0 }'
}

para="The river carried the old mill's stones downstream every spring, and the town counted the boats that drifted past the willows while the miller kept his ledgers by lamplight, writing each cargo, each name and each price in a careful hand."
body=$(python3 -I -c 'import json, sys; print(json.dumps({"model": sys.argv[2], "temperature": 0,
  "max_tokens": 2000, "messages": [{"role": "user", "content":
  "Copy this paragraph exactly, sixteen times, one copy per line, nothing else:\n\n" + sys.argv[1]}]}))' \
  "$para" "$model")
for r in "${rows[@]}"; do
  leg="$out/r$r"
  mkdir -p "$leg"
  extra=()
  [ "$r" -gt 1 ] && extra=(--speculative-model ngram --num-speculative-tokens $((r - 1)))
  settle
  "$bin" serve "$model" --port $port --enable-metrics ${extra[@]+"${extra[@]}"} > "$leg/serve.log" 2>&1 &
  pid=$!
  until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
    kill -0 $pid 2> /dev/null || { echo "r$r: server died, see $leg/serve.log"; continue 2; }
    timeout 2 tail -f /dev/null || true
  done
  curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"'"$model"'","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}' \
    > /dev/null
  d0=$(metric vllm_spec_decode_num_drafts) t0=$(metric vllm_spec_decode_num_draft_tokens)
  xcrun xctrace record --template "Metal System Trace" --attach $pid --time-limit 300s \
    --output "$leg/mst.trace" > "$leg/xctrace.log" 2>&1 &
  trace=$!
  until grep -q "Ctrl-C to stop" "$leg/xctrace.log" 2> /dev/null; do
    kill -0 $trace 2> /dev/null || { echo "r$r: xctrace failed, see $leg/xctrace.log"; break; }
    timeout 1 tail -f /dev/null || true
  done
  curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' \
    -d "$body" > "$leg/reply.json"
  kill -INT $trace
  wait $trace
  d1=$(metric vllm_spec_decode_num_drafts) t1=$(metric vllm_spec_decode_num_draft_tokens)
  kill -INT $pid
  wait $pid
  xcrun xctrace export --input "$leg/mst.trace" --xpath \
    '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > "$leg/gpu.xml"
  rm -rf "$leg/mst.trace"
  per=$(python3 -I -c 'import sys; d0, d1, t0, t1 = (float(x or 0) for x in sys.argv[1:5]);
print(f"{(t1 - t0) / (d1 - d0):.2f} over {d1 - d0:.0f} drafting steps" if d1 > d0 else "-")' \
    "${d0:-0}" "${d1:-0}" "${t0:-0}" "${t1:-0}")
  python3 -I "$here/spans.py" "$leg/gpu.xml" "$(basename "$bin")" "$r" "$per" | tee "$leg/row.txt"
done
python3 -I - "$out" "${rows[@]}" <<'PY'
import os, re, sys
out, rows = sys.argv[1], sys.argv[2:]
span = {}
for r in rows:
    p = os.path.join(out, f"r{r}", "row.txt")
    m = re.search(r"upper mode ([0-9.]+) ms", open(p).read()) if os.path.exists(p) else None
    if m:
        span[r] = float(m.group(1))
base = span.get("1")
for r in rows:
    if r in span:
        print(f"R={r}: {span[r]:.2f} ms" + (f"  c(R)/c(1) = {span[r] / base:.2f}" if base else ""))
PY
