#!/usr/bin/env bash
# One bench leg: `scr serve` alone on the GPU (wired memory settled first), the first N prompts of
# each of the P-EAGLE model card's benchmarks (bench-prompts.py's lists), greedy, streamed one at a
# time. Per benchmark: decode tok/s (first visible token to last, so prefill is out), the server's
# mean time to first token (its histogram) and the client's median to the first visible chunk, and
# tokens a drafting step from /metrics. Writes each reply for a text check.
# Usage: leg.sh <scr binary> <target repo> <bench dir> <out dir> <N> <max tokens> [serve args...]
set -uo pipefail
bin=${1:?binary}
model=${2:?target}
bench=${3:?bench dir}
out=${4:?out dir}
n=${5:?prompts}
t=${6:?max tokens}
shift 6
port=${PORT:-8796}
mkdir -p "$out"
wired() { vm_stat | awk '/Pages wired down/ { gsub("\\.", "", $4); print $4 * 16384 }'; }
for _ in $(seq 1 150); do
  [ "$(wired)" -lt $((6 << 30)) ] && break
  timeout 2 tail -f /dev/null || true
done
"$bin" serve "$model" --port $port --enable-metrics "$@" > "$out/serve.log" 2>&1 &
pid=$!
until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
  kill -0 $pid 2> /dev/null || { echo "server died, see $out/serve.log"; exit 1; }
  timeout 2 tail -f /dev/null || true
done
python3 -I - "$port" "$model" "$bench" "$out" "$n" "$t" <<'PY'
import json, re, statistics, sys, time, urllib.request
port, model, bench, out, n, t = sys.argv[1:7]
n, t = int(n), int(t)
def counters():
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm_spec_decode_num_{k}(?:_total)?(?:{{[^}}]*}})?\s+(\S+)$", text, re.M))
    ttft = lambda k: sum(float(v) for v in re.findall(rf"^vllm_time_to_first_token_seconds_{k}(?:{{[^}}]*}})?\s+(\S+)$", text, re.M))
    return get("drafts"), get("accepted_tokens"), ttft("sum"), ttft("count")
def stream(prompt):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": t, "stream": True,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    start, first, text, tokens = time.time(), None, [], 0
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            for c in chunk.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning") or "") + (d.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time()
                text.append(piece)
    return start, first, time.time(), tokens, "".join(text)
stream("Hi")  # warm-up
row = []
for name in ("mtbench", "humaneval", "gsm8k"):
    prompts = json.load(open(f"{bench}/{name}.json"))[:n]
    d0, a0, s0, c0 = counters()
    toks, decode, ttfts, texts = 0, 0.0, [], []
    for p in prompts:
        time.sleep(float(__import__("os").environ.get("GAP_S", "0")))
        start, first, end, k, text = stream(p)
        toks += k - 1
        decode += end - first
        ttfts.append(first - start)
        texts.append(text)
    d1, a1, s1, c1 = counters()
    server = f"{1000 * (s1 - s0) / (c1 - c0):4.0f}" if c1 > c0 else "-"
    al = f"{1 + (a1 - a0) / (d1 - d0):.2f}" if d1 > d0 else "-"
    open(f"{out}/{name}.txt", "w").write("\n\x1e\n".join(texts))
    row.append(f"{name} {toks / decode:5.1f} tok/s TTFT {server} ms (visible {1000 * statistics.median(ttfts):4.0f}) AL {al}")
print(" | ".join(row), flush=True)
PY
kill -INT $pid
wait $pid
