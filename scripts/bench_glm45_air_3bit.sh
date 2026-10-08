#!/usr/bin/env bash
# GLM-4.5-Air-3bit correctness smoke + initial decode benchmark.
#
# The first b3 checkpoint on metal: `mlx-community/GLM-4.5-Air-3bit`
# (mlx-affine-b3-g64-qembed). This script is the PR's repro record:
#
#   1. GREEDY PARITY — three fixed prompts through `scr chat` at
#      --temperature 0.0. Before the fold fix every prompt decoded
#      deterministic garbage from the first step (the M1 bucket dropped
#      the routed experts' contribution); the expected lines pin the
#      coherent behaviour. Requires the checkpoint in the local Hub
#      cache (`scr model pull mlx-community/GLM-4.5-Air-3bit`).
#   2. DECODE THROUGHPUT — `scr bench latency` at conc 1/2/4/8 (short
#      ctx) and conc 1 at ctx 512/2048/4096, plus an mlx-lm
#      same-checkpoint decode number for comparison when a
#      `mlx_lm`-capable python is available (a checkout with a venv,
#      e.g. ~/git/mlx-lm/.venv/bin/python).
#
# Usage:
#   scripts/bench_glm45_air_3bit.sh [--skip-parity] [--skip-bench]
#                                   [--mlx-python PATH]
#
# Needs one build first (metal, this checkpoint's model + quant scope):
#   cargo build --release -p scratchy-cli \
#     --features metal,bench,model/glm-4.5-air,quant/mlx-affine-b3-g64-qembed
#
# Results land under bench_results/glm45-air-3bit/<timestamp>/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&>/dev/null && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." &&>/dev/null && pwd)"
SCR="$ROOT/target/release/scr"

MODEL="mlx-community/GLM-4.5-Air-3bit"
SKIP_PARITY=0
SKIP_BENCH=0
MLX_PYTHON=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-parity) SKIP_PARITY=1; shift ;;
        --skip-bench)  SKIP_BENCH=1; shift ;;
        --mlx-python)  MLX_PYTHON="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
[ -z "$MLX_PYTHON" ] && [ -x "$HOME/git/mlx-lm/.venv/bin/python" ] \
    && MLX_PYTHON="$HOME/git/mlx-lm/.venv/bin/python"

[ -x "$SCR" ] || { echo "no $SCR — build with the bench feature first (see header)"; exit 1; }

OUT_DIR="$ROOT/bench_results/glm45-air-3bit/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

{
    echo "machine: $(sysctl -n hw.model 2>/dev/null || echo unknown)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || system_profiler SPHardwareDataType 2>/dev/null | grep Chip | head -1)"
    echo "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
    echo "binary: $SCR"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/env.txt"

# ── 1. greedy parity smoke ─────────────────────────────────────────────
# GLM-4.5-Air is a reasoning model: `scr chat` prints the thinking
# before the answer, so these are KEYWORD assertions on the full greedy
# transcript, not completion-prefix or token-identity checks. The defect
# this pins is "deterministic garbage from the first decode step" —
# before the fold fix no prompt ever produced its answer.
if [ "$SKIP_PARITY" = 0 ]; then
    echo "== greedy parity"
    PARITY="$OUT_DIR/parity.txt"
    : > "$PARITY"
    check() { # prompt expected_keyword
        # 800 tokens: the thinking trace routinely runs 300-600 tokens
        # before the model commits to the answer.
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
    # Answers recorded 2026-10-08 (M5 Max, 64 GB) at the fix commit,
    # greedy --temperature 0.0.
    check "The capital of France is" "Paris"
    check "What is 17*23? Answer with just the number." "391"
    echo "parity: PASS" >> "$PARITY"
fi

# ── 2. decode throughput ───────────────────────────────────────────────
# `scr bench latency` loads the model in-process (same worker path as
# `scr serve`); --gpu-memory-utilization 0.95 keeps ctx 2048 schedulable
# inside the 64 GB machine alongside the 43.6 GB weights.
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

echo
echo "results: $OUT_DIR"
