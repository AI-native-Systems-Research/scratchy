#!/usr/bin/env bash
# GLM-4.5-Air-3bit m≥2 matvec-ends fold + hazard-class fences (#295):
# correctness smoke + decode benchmark. This script is the PR's repro record.
#
#   1. BATCH COHERENCE — the defect this PR fixes. Two fixed prompts at
#      temperature 0.0 through `scr batch` (the m2 bucket, the failing
#      configuration). Before the fence fix, one request garbled
#      ~50% of runs (self-echo loops / token soup, nondeterministic —
#      same inputs, same weights, different outputs). The gate runs the
#      batch N times (default 6) and asserts every run is coherent AND
#      all runs are byte-identical to each other: a race is
#      nondeterministic, so determinism at temp 0 is the oracle.
#   2. FOUR-WAY BATCH (the m4 bucket) — same oracle at a bigger bucket.
#   3. GREEDY PARITY — keyword assertions on `scr chat` transcripts
#      (conc-1, the m1 bucket: must not regress).
#   4. DECODE THROUGHPUT — `scr bench latency` at conc 1/2/4/8 and conc-1
#      ctx 512/2048/4096, plus an mlx-lm same-checkpoint decode number
#      when a `mlx_lm`-capable python is available.
#
# Usage:
#   scripts/bench_glm45_air_m2_fold.sh [--runs N] [--skip-coherence]
#                                      [--skip-parity] [--skip-bench]
#                                      [--mlx-python PATH]
#
# Needs one build first (metal, this checkpoint's model + quant scope):
#   cargo build --release -p scratchy-cli \
#     --features metal,bench,model/glm-4.5-air,quant/mlx-affine-b3-g64-qembed
#
# Results land under bench_results/glm45-air-m2/<timestamp>/.
#
# ⚠️ KNOWN MACHINE EDGE (64 GB M5 Max): conc-4 at out-128 sits AT the
# GPU wired-memory limit and can OOM
# (kIOGPUCommandBufferCallbackErrorOutOfMemory) depending on machine
# state — pre-existing at the base commit too, not introduced here. It
# passes at out-64; the conc-4 point of the A/B table was captured on
# an idle machine.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&>/dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &&>/dev/null && pwd)"
SCR="$ROOT/target/release/scr"

MODEL="mlx-community/GLM-4.5-Air-3bit"
RUNS=6
SKIP_COHERENCE=0
SKIP_PARITY=0
SKIP_BENCH=0
MLX_PYTHON=""
while [ $# -gt 0 ]; do
    case "$1" in
        --runs)           RUNS="$2"; shift 2 ;;
        --skip-coherence) SKIP_COHERENCE=1; shift ;;
        --skip-parity)    SKIP_PARITY=1; shift ;;
        --skip-bench)     SKIP_BENCH=1; shift ;;
        --mlx-python)     MLX_PYTHON="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
[ -z "$MLX_PYTHON" ] && [ -x "$HOME/git/mlx-lm/.venv/bin/python" ] \
    && MLX_PYTHON="$HOME/git/mlx-lm/.venv/bin/python"

[ -x "$SCR" ] || { echo "no $SCR — build with the bench feature first (see header)"; exit 1; }

OUT_DIR="$ROOT/bench_results/glm45-air-m2/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

{
    echo "machine: $(sysctl -n hw.model 2>/dev/null || echo unknown)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || system_profiler SPHardwareDataType 2>/dev/null | grep Chip | head -1)"
    echo "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
    echo "binary: $SCR"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/env.txt"

# ── 1. batch coherence (the m2 bucket) ────────────────────────────────
# temp 0.0: same inputs + same weights must give the SAME output every
# run. Before the fence fix one request garbled ~50% of runs and two
# identical prompts diverged from each other.
if [ "$SKIP_COHERENCE" = 0 ]; then
    echo "== batch coherence (m2 bucket, $RUNS runs)"
    COH="$OUT_DIR/coherence"
    mkdir -p "$COH"
    python3 - "$COH" <<'PY'
import json, sys
out = sys.argv[1]
reqs = [
    ("req-1", "Explain in exactly three sentences why the sky is blue."),
    ("req-2", "Write a haiku about mountains in winter."),
]
with open(f"{out}/reqs.jsonl", "w") as f:
    for cid, prompt in reqs:
        f.write(json.dumps({
            "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "x", "messages": [{"role": "user", "content": prompt}],
                     "temperature": 0.0, "max_tokens": 64},
        }) + "\n")
# The four-way batch: exercises the m4 bucket, which also gained the rows kernel.
c4 = [
    ("c4-1", "Explain in exactly three sentences why the sky is blue."),
    ("c4-2", "Write a haiku about mountains in winter."),
    ("c4-3", "Name three primary colors and one secondary color."),
    ("c4-4", "What is 17 times 23? Answer with just the number."),
]
with open(f"{out}/reqs-c4.jsonl", "w") as f:
    for cid, prompt in c4:
        f.write(json.dumps({
            "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "x", "messages": [{"role": "user", "content": prompt}],
                     "temperature": 0.0, "max_tokens": 64},
        }) + "\n")
PY
    for i in $(seq 1 "$RUNS"); do
        "$SCR" batch -m "$MODEL" --device metal \
            -i "$COH/reqs.jsonl" -o "$COH/out-$i.jsonl" >/dev/null 2>&1
        "$SCR" batch -m "$MODEL" --device metal \
            -i "$COH/reqs-c4.jsonl" -o "$COH/out-c4-$i.jsonl" >/dev/null 2>&1
    done
    python3 - "$COH" "$RUNS" <<'PY'
import json, sys, glob
out, runs = sys.argv[1], int(sys.argv[2])
def texts(pat):
    per_req = {}
    for f in sorted(glob.glob(f"{out}/{pat}")):
        for line in open(f):
            d = json.loads(line)
            per_req.setdefault(d["custom_id"], set()).add(
                d["response"]["body"]["choices"][0]["message"]["content"])
    return per_req
def check(pat, label):
    per_req = texts(pat)
    assert per_req, f"{label}: no outputs"
    bad = {k: len(v) for k, v in per_req.items() if len(v) != 1}
    if bad:
        print(f"  {label}: FAIL — nondeterministic outputs: {bad}")
        sys.exit(1)
    # Coherence: no self-echo loops / soup. Cheap structural oracle: each
    # output must contain at least 8 distinct words (garbled runs collapse
    # to repetition).
    for k, v in per_req.items():
        words = set(next(iter(v)).split())
        if len(words) < 8:
            print(f"  {label}: FAIL — {k} looks like garbage: {next(iter(v))[:80]!r}")
            sys.exit(1)
    print(f"  {label}: {len(per_req)} requests x {runs} runs, deterministic + coherent")
check("out-*.jsonl", "m2 batch")
check("out-c4-*.jsonl", "m4 batch")
PY
fi

# ── 2. greedy parity (the m1 bucket) ──────────────────────────────────
if [ "$SKIP_PARITY" = 0 ]; then
    echo "== greedy parity"
    PARITY="$OUT_DIR/parity.txt"
    : > "$PARITY"
    check() { # prompt expected_keyword
        got=$("$SCR" chat -m "$MODEL" --device metal --temperature 0.0 \
                  --max-tokens 800 -q "$1" 2>/dev/null \
              | grep -v -E '^(2026-|Using model:)' | tr -d '\n')
        echo "prompt: $1" >> "$PARITY"
        echo "output: ${got:0:120}" >> "$PARITY"
        if [[ "$got" == *"$2"* ]]; then
            echo "  PASS: contains '$2'"
        else
            echo "  FAIL: expected '$2' somewhere in the transcript"
            exit 1
        fi
    }
    check "The capital of France is" "Paris"
    check "What is 17*23? Answer with just the number." "391"
    echo "parity: PASS" >> "$PARITY"
fi

# ── 3. decode throughput ──────────────────────────────────────────────
# --gpu-memory-utilization 0.95 keeps ctx 4096 schedulable inside the
# 64 GB machine alongside the 43.6 GB weights.
if [ "$SKIP_BENCH" = 0 ]; then
    echo "== decode throughput"
    BENCH="$OUT_DIR/bench.txt"
    : > "$BENCH"
    for bs in 1 2 4 8; do
        echo "-- conc $bs (in 32 / out 128)" | tee -a "$BENCH"
        "$SCR" bench latency -m "$MODEL" --device metal \
            --num-iters 10 --num-iters-warmup 3 \
            --input-len 32 --output-len 128 --batch-size "$bs" \
            --gpu-memory-utilization 0.95 2>/dev/null \
            | grep -E "Throughput|Avg latency" | tee -a "$BENCH"
    done
    for ctx in 512 2048 4096; do
        echo "-- ctx $ctx (conc 1 / out 128)" | tee -a "$BENCH"
        "$SCR" bench latency -m "$MODEL" --device metal \
            --num-iters 5 --num-iters-warmup 2 \
            --input-len "$ctx" --output-len 128 --batch-size 1 \
            --gpu-memory-utilization 0.95 2>/dev/null \
            | grep -E "Throughput|Avg latency" | tee -a "$BENCH"
    done

    if [ -n "$MLX_PYTHON" ] && "$MLX_PYTHON" -c "import mlx_lm" 2>/dev/null; then
        echo "== mlx-lm comparison" | tee -a "$BENCH"
        "$MLX_PYTHON" - "$MODEL" >> "$BENCH" 2>/dev/null <<'PY'
import sys, time
from mlx_lm import load
from mlx_lm.generate import stream_generate
model, tokenizer = load(sys.argv[1])
prompt = "Write a short paragraph explaining why the sky is blue."
for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=8):
    pass
ts = [time.perf_counter()]
for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=128):
    ts.append(time.perf_counter())
itls = sorted(b - a for a, b in zip(ts, ts[1:]))
med = itls[len(itls) // 2]
print(f"mlx-lm median ITL {med*1000:.2f} ms -> {1/med:.1f} tok/s decode")
PY
    else
        echo "mlx-lm comparison skipped (no mlx_lm python found)" | tee -a "$BENCH"
    fi
fi

echo "results: $OUT_DIR"
