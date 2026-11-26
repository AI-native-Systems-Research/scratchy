#!/usr/bin/env bash
# One traced bench leg: `scr serve` (serve args after the four positionals), a 16-token warm-up
# request, then an Instruments Metal System Trace while the first N HumanEval prompts stream greedily
# (max_tokens T) one at a time; each request's start, first token and end, then request-trace.py.
# Usage: request-trace.sh <binary> <target repo> <bench dir> <out dir> <N> <T> [serve args]
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${1:?binary}
model=${2:?target}
bench=${3:?bench dir}
out=${4:?out dir}
n=${5:?prompts}
t=${6:?max tokens}
shift 6
port=8799
rm -rf "$out"
mkdir -p "$out"
"$bin" serve "$model" --port $port "$@" > "$out/serve.log" 2>&1 &
pid=$!
until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
  kill -0 $pid 2> /dev/null || { echo "server died, see $out/serve.log"; exit 1; }
  timeout 2 tail -f /dev/null || true
done
curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"'"$model"'","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}' > /dev/null
xcrun xctrace record --template "Metal System Trace" --attach $pid --time-limit 90s \
  --output "$out/mst.trace" > "$out/xctrace.log" 2>&1 &
trace=$!
until grep -q "Ctrl-C to stop" "$out/xctrace.log" 2> /dev/null; do
  kill -0 $trace 2> /dev/null || { echo "xctrace failed, see $out/xctrace.log"; exit 1; }
  timeout 1 tail -f /dev/null || true
done
python3 -I - "$port" "$model" "$bench" "$n" "$t" "$out/requests.json" <<'PY'
import json, os, sys, time, urllib.request
port, model, bench, n, t, path = sys.argv[1:7]
if os.path.exists(f"{bench}/humaneval-rows.json"):
    prompts = [r["row"]["prompt"] for r in json.load(open(f"{bench}/humaneval-rows.json"))["rows"]]
elif os.path.exists(f"{bench}/mtbench.jsonl"):
    prompts = [r["row"]["prompt"] for r in json.load(open(f"{bench}/humaneval.json"))["rows"]]
else:
    prompts = json.load(open(f"{bench}/humaneval.json"))
done = []
for i, p in enumerate(prompts[: int(n)]):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": int(t), "stream": True,
                       "messages": [{"role": "user", "content": p}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    start, first = time.time(), None
    with urllib.request.urlopen(req) as r:
        for line in r:
            if first is None and line.startswith(b"data:"):
                first = time.time()
    done.append({"name": f"humaneval-{i}", "start": start, "first": first, "end": time.time()})
json.dump(done, open(path, "w"))
PY
kill -INT $trace 2> /dev/null
wait $trace
kill -INT $pid
wait $pid
xcrun xctrace export --input "$out/mst.trace" --toc > "$out/toc.xml"
xcrun xctrace export --input "$out/mst.trace" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > "$out/metal-gpu-intervals.xml"
python3 -I "$here/request-trace.py" "$out/toc.xml" "$out/metal-gpu-intervals.xml" "$(basename "$bin")" \
  "$out/requests.json"
