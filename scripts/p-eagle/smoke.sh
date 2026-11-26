#!/usr/bin/env bash
# One server alone on the GPU, one greedy request: its reply, decode tokens, and spec-decode
# counters. Usage: smoke.sh <scr binary> <target repo> <out dir> <max tokens> [serve args...]
set -uo pipefail
bin=${1:?scr binary}
model=${2:?target repo}
out=${3:?out dir}
max=${4:?max tokens}
shift 4
port=${PORT:-8794}
mkdir -p "$out"
"$bin" serve "$model" --port $port --enable-metrics "$@" > "$out/serve.log" 2>&1 &
pid=$!
until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
  kill -0 $pid 2> /dev/null || { echo "server died, see $out/serve.log"; tail -20 "$out/serve.log"; exit 1; }
  timeout 2 tail -f /dev/null || true
done
prompt=${PROMPT:-"Write a Python function that returns the n-th Fibonacci number, with a docstring."}
body=$(python3 -I -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "temperature": 0,
  "max_tokens": int(sys.argv[2]), "messages": [{"role": "user", "content": sys.argv[3]}]}))' \
  "$model" "$max" "$prompt")
start=$(python3 -I -c 'import time; print(time.time())')
curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' \
  -d "$body" > "$out/reply.json"
end=$(python3 -I -c 'import time; print(time.time())')
curl -sf "http://127.0.0.1:$port/metrics" | grep -E "^vllm_spec_decode_num" > "$out/metrics.txt"
kill -INT $pid
wait $pid
python3 -I - "$out" "$start" "$end" <<'PY'
import json, re, sys
out, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = json.load(open(f"{out}/reply.json"))
n = r["usage"]["completion_tokens"]
m = {k: float(v) for k, v in re.findall(r"^(\S+?)(?:_total)?(?:\{[^}]*\})?\s+(\S+)$", open(f"{out}/metrics.txt").read(), re.M)}
drafts, toks, acc = (m.get(f"vllm_spec_decode_num_{k}", 0) for k in ("drafts", "draft_tokens", "accepted_tokens"))
al = f"{1 + acc / drafts:.2f}" if drafts else "-"
print(f"{n} tokens in {end - start:.2f} s ({n / (end - start):.1f} tok/s, prefill included); "
      f"drafting steps {drafts:.0f}, drafted {toks:.0f}, accepted {acc:.0f}, tokens/step {al}")
open(f"{out}/text.txt", "w").write(r["choices"][0]["message"].get("content") or "")
PY
