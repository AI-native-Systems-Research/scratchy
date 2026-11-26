#!/usr/bin/env bash
# The GPU's timeline while decoding: one `scr serve` of gpt-oss-20b (serve args after the out dir),
# one greedy request of 1500 tokens, an Instruments Metal System Trace of the server for 3 s
# mid-reply. Writes the trace and the reply's usage. Usage: gpu-trace.sh <out dir> [serve args]
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?out dir}
shift
mkdir -p "$out"
port=8795
model=jesusoctavioas/gpt-oss-20b-mlx-4Bit
"${SCR:?binary}" serve "$model" --port $port --enable-metrics "$@" > "$out/serve.log" 2>&1 &
pid=$!
for _ in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && break
  kill -0 $pid 2> /dev/null || { echo "server died"; exit 1; }
  timeout 2 tail -f /dev/null
done
body='{"model":"'$model'","temperature":0,"max_tokens":1500,"messages":[{"role":"user","content":"Write a long, detailed story about a lighthouse keeper and the ships she guides, chapter by chapter."}]}'
curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d "$body" > "$out/reply.json" &
req=$!
timeout 3 tail -f /dev/null
curl -sf "http://127.0.0.1:$port/metrics" | grep -E "spec_decode_num_(drafts|accepted_tokens)(_total)? |generation_tokens_total " > "$out/metrics-0.txt"
xcrun xctrace record --template "Metal System Trace" --attach $pid --time-limit 3s \
  --output "$out/mst.trace" > "$out/xctrace.log" 2>&1
curl -sf "http://127.0.0.1:$port/metrics" | grep -E "spec_decode_num_(drafts|accepted_tokens)(_total)? |generation_tokens_total " > "$out/metrics-1.txt"
wait $req
kill -INT $pid
wait $pid
python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['usage'])" "$out/reply.json"
cat "$out/metrics-0.txt" "$out/metrics-1.txt"
