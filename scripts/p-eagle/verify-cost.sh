#!/usr/bin/env bash
# Verify cost c(R)/c(1) on gpt-oss-20b: one `scr serve` with the n-gram proposer (K=7), one greedy
# request whose reply repeats its prompt (so verify steps of 2..8 rows run), per-forward GPU wait
# from SCRATCHY_METAL_TRACE grouped by row count. Usage: verify-cost.sh <out-dir>
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?out dir}
mkdir -p "$out"
port=8791
k=${2:-7}
model=jesusoctavioas/gpt-oss-20b-mlx-4Bit
SCRATCHY_METAL_TRACE=1 "$here/target/release/scr" serve "$model" --port $port \
  --speculative-model ngram --num-speculative-tokens $k > "$out/serve.log" 2>&1 &
pid=$!
up=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && { up=1; break; }
  kill -0 $pid 2> /dev/null || break
  timeout 2 tail -f /dev/null
done
if [ $up -eq 1 ]; then
  para="The quick brown fox jumps over the lazy dog near the riverbank, where the old miller kept his stones and the children counted boats drifting past the willow trees every summer afternoon."
  body=$(python3 -I -c 'import json,sys; p=sys.argv[1]; print(json.dumps({"model":sys.argv[2],"temperature":0,"max_tokens":400,"messages":[{"role":"user","content":"Copy this paragraph exactly, four times, one copy per line, nothing else:\n\n"+p}]}))' "$para" "$model")
  curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' -d "$body" > "$out/reply.json"
  echo "request exit $?"
else
  echo "server never came up"
fi
kill -INT $pid 2> /dev/null
wait $pid
echo "server exit $?"
# GPU wait per forward by row count (num_tokens), steady state only: skip each count's first 2.
grep -o 'num_tokens=[0-9]* .*wait=[0-9.]*[mµn]*s' "$out/serve.log" | python3 -I -c '
import re, sys, statistics as st
by = {}
for line in sys.stdin:
    n = int(re.search(r"num_tokens=(\d+)", line).group(1))
    w = re.search(r"wait=([0-9.]+)(ms|µs|ns|s)", line)
    v = float(w.group(1)) * {"ms": 1, "µs": 1e-3, "ns": 1e-6, "s": 1e3}[w.group(2)]
    by.setdefault(n, []).append(v)
base = None
for n in sorted(by):
    xs = by[n][2:] or by[n]
    if n > 64: continue
    med = st.median(xs)
    base = base or (med if n == 1 else None)
    rel = f"{med / base:.2f}x" if base else "-"
    print(f"rows={n:3d} count={len(by[n]):4d} median_wait={med:7.2f} ms  c(R)/c(1)={rel}")
' | tee "$out/by-rows.txt"
