#!/usr/bin/env bash
# A request's first command buffers, probe-free: `scr serve` alone, a warm-up request, then an
# Instruments Metal System Trace around one streamed greedy request. Prints the request's time to
# its first streamed chunk and each of the first command buffers' GPU start (from the request's
# send) and span. Usage: first-steps.sh <scr binary> <target repo> <out dir> [serve args...]
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${1:?binary}
model=${2:?target}
out=${3:?out dir}
shift 3
port=${PORT:-8798}
mkdir -p "$out"
"$bin" serve "$model" --port $port --enable-metrics "$@" > "$out/serve.log" 2>&1 &
pid=$!
until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
  kill -0 $pid 2> /dev/null || { echo "server died, see $out/serve.log"; exit 1; }
  timeout 2 tail -f /dev/null || true
done
curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"'"$model"'","temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}' > /dev/null
xcrun xctrace record --template "Metal System Trace" --attach $pid --time-limit 60s \
  --output "$out/mst.trace" > "$out/xctrace.log" 2>&1 &
trace=$!
until grep -q "Ctrl-C to stop" "$out/xctrace.log" 2> /dev/null; do
  kill -0 $trace 2> /dev/null || { echo "xctrace failed, see $out/xctrace.log"; exit 1; }
  timeout 1 tail -f /dev/null || true
done
timeout 2 tail -f /dev/null
python3 -I - "$port" "$model" "$out/request.json" <<'PY'
import json, sys, time, urllib.request
port, model, path = sys.argv[1:4]
prompt = ("Write a Python function that returns the n-th Fibonacci number, with a docstring "
          "and a short explanation of its complexity.")
body = json.dumps({"model": model, "temperature": 0, "max_tokens": 48, "stream": True,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                             {"content-type": "application/json"})
start, first, chunks = time.time(), None, []
with urllib.request.urlopen(req) as r:
    for line in r:
        if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
            t = time.time()
            first = first or t
            chunks.append(t - start)
json.dump({"start": start, "first": first, "chunks": chunks}, open(path, "w"))
PY
timeout 2 tail -f /dev/null
kill -INT $trace 2> /dev/null
wait $trace
kill -INT $pid
wait $pid
xcrun xctrace export --input "$out/mst.trace" --toc > "$out/toc.xml"
xcrun xctrace export --input "$out/mst.trace" --xpath \
  '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > "$out/gpu.xml"
rm -rf "$out/mst.trace"
python3 -I - "$out" "$(basename "$bin")" <<'PY'
import datetime, json, re, sys
import xml.etree.ElementTree as ET
out, process = sys.argv[1:3]
t0 = datetime.datetime.fromisoformat(
    re.search(r"<start-date>([^<]*)</start-date>", open(f"{out}/toc.xml").read()).group(1)).timestamp()
ids = {}
def value(el):
    if "ref" in el.attrib:
        return ids[el.attrib["ref"]]
    v = (el.text or "", el.attrib.get("fmt", ""), el.tag)
    if "id" in el.attrib:
        ids[el.attrib["id"]] = v
    for c in el:
        value(c)
    return v
gpu = {}
for row in ET.parse(f"{out}/gpu.xml").getroot().iter("row"):
    cols = {}
    for c in row:
        v = value(c)
        cols.setdefault(v[2], v)
    proc, cb = cols.get("process"), cols.get("metal-command-buffer-id")
    depth, chan = cols.get("metal-nesting-level"), cols.get("gpu-channel-name")
    if not (proc and process in proc[1] and cb and depth and depth[0] == "0" and chan and chan[0] == "Compute"):
        continue
    s, d = int(cols["start-time"][0]), int(cols["duration"][0])
    a, b = gpu.get(cb[0], (s, s + d))
    gpu[cb[0]] = (min(a, s), max(b, s + d))
r = json.load(open(f"{out}/request.json"))
# The trace's clock and the host's differ by its start's rounding: the request's first command
# buffer (its prefill) is the trace's first, so times count from its start.
spans = sorted(gpu.values())
first_cb = spans[0][0]
cbs = [((s - first_cb) / 1e6, (e - s) / 1e6) for s, e in spans]
print(f"first chunk {1000 * (r['first'] - r['start']):.0f} ms; chunks at "
      + " ".join(f"{1000 * c:.0f}" for c in r["chunks"][:8]) + " ms")
for at, span in cbs[:12]:
    print(f"  cb at {at:7.1f} ms  span {span:6.2f} ms")
PY
