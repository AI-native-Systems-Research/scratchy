#!/usr/bin/env bash
# Harness for draft-model speculative decoding.
# Walks three scenarios against a single
# target model and reports TTFT + decode TPS for each:
#
#   1. baseline          — no spec decode
#   2. ngram             — existing prompt-lookup proposer
#   3. draft-model       — (Phase 4) second-model proposer
#
# Each scenario launches `scr serve` in the background, waits for the model
# to load, runs `scr bench serve` against /v1/completions, then tears the
# server down before moving on. Serialized so that the 24 GiB working-set cap
# is respected (one model loader at a time, per machine).
#
# Usage:
#   scripts/bench_spec_decode.sh [--skip-build] [--target REPO] [--draft REPO]
# scratchy-models has no default model scope, so the build below names both
# `llama-3.2-3b` (target) and `llama-3.2-1b` (draft) explicitly.
#
# Outputs land under vllm-rs/bench_results/spec_decode/<timestamp>/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VLLM_RS_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

TARGET_MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
DRAFT_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
INPUT_LEN=512
OUTPUT_LEN=128
NUM_PROMPTS=16
NUM_SPEC_TOKENS=4
PORT=8765
SKIP_BUILD=0
SCENARIOS="baseline,ngram,draft-model"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)            TARGET_MODEL="$2"; shift 2 ;;
        --draft)             DRAFT_MODEL="$2"; shift 2 ;;
        --input-len)         INPUT_LEN="$2"; shift 2 ;;
        --output-len)        OUTPUT_LEN="$2"; shift 2 ;;
        --num-prompts)       NUM_PROMPTS="$2"; shift 2 ;;
        --num-spec-tokens)   NUM_SPEC_TOKENS="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --scenarios)         SCENARIOS="$2"; shift 2 ;;
        --skip-build)        SKIP_BUILD=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

cd "${VLLM_RS_DIR}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${VLLM_RS_DIR}/bench_results/spec_decode/${TS}"
mkdir -p "${OUT_DIR}"
echo "results -> ${OUT_DIR}"

# ---- build (or trust an existing binary) -----------------------------------
VLLM_BIN="${VLLM_RS_DIR}/target/release/scr"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "building scr (release, Fmetal,bench)..."
    cargo build --release --bin scr \
        --features metal,bench,model/llama-3.2-3b,model/llama-3.2-1b
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "scr binary not found at ${VLLM_BIN}" >&2
    exit 1
fi

# ---- helpers ---------------------------------------------------------------
SERVER_PID=""
SERVER_LOG=""

wait_for_server() {
    local url="http://127.0.0.1:${PORT}/v1/models"
    local deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            return 0
        fi
        # Bail fast if the server already died.
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server pid ${SERVER_PID} died before ready; see ${SERVER_LOG}" >&2
            tail -n 60 "${SERVER_LOG}" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "server failed to come up within 300s; see ${SERVER_LOG}" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        # Give it a few seconds to drop the model + release GPU memory.
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

run_scenario() {
    local label="$1"; shift
    local serve_extra=("$@")

    SERVER_LOG="${OUT_DIR}/${label}.serve.log"
    echo
    echo "=== scenario: ${label} ==="
    echo "serve: ${VLLM_BIN} serve ${TARGET_MODEL} --port ${PORT} ${serve_extra[*]+${serve_extra[*]}}"

    # `scr serve` defaults: host=127.0.0.1 is fine (see crates/cli/scr/src/args.rs:133).
    # The ${arr[@]+"${arr[@]}"} idiom expands to nothing when the array is
    # empty — needed because `set -u` treats `${arr[@]}` on an empty array
    # as referencing an unset variable.
    "${VLLM_BIN}" serve "${TARGET_MODEL}" \
        --port "${PORT}" \
        ${serve_extra[@]+"${serve_extra[@]}"} \
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        stop_server
        return 1
    fi

    local bench_json="${OUT_DIR}/${label}.bench.json"
    local bench_log="${OUT_DIR}/${label}.bench.log"
    "${VLLM_BIN}" bench serve \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${TARGET_MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --output-json "${bench_json}" \
        --disable-tqdm \
        2>&1 | tee "${bench_log}"

    stop_server
}

# ---- scenarios -------------------------------------------------------------
IFS=',' read -r -a scenario_arr <<<"${SCENARIOS}"
for sc in "${scenario_arr[@]}"; do
    case "${sc}" in
        baseline)
            run_scenario "baseline"
            ;;
        ngram)
            run_scenario "ngram" \
                --speculative-model ngram \
                --num-speculative-tokens "${NUM_SPEC_TOKENS}"
            ;;
        draft-model)
            # Phase 1+ will teach `--speculative-model <repo>` to load a real
            # draft model. Until then this scenario is expected to fail at
            # startup with a clear "not yet implemented" error; the failure
            # log is preserved next to the baseline/ngram results so the gap
            # is auditable.
            if ! run_scenario "draft-model" \
                --speculative-model "${DRAFT_MODEL}" \
                --num-speculative-tokens "${NUM_SPEC_TOKENS}"; then
                echo "draft-model scenario failed (expected pre-Phase-4); continuing"
            fi
            ;;
        *)
            echo "unknown scenario: ${sc}" >&2
            exit 2
            ;;
    esac
done

# ---- summary ---------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.md"
{
    echo "# Spec-decode bench — ${TS}"
    echo
    echo "- target: ${TARGET_MODEL}"
    echo "- draft:  ${DRAFT_MODEL}"
    echo "- input/output: ${INPUT_LEN}/${OUTPUT_LEN}"
    echo "- prompts: ${NUM_PROMPTS}"
    echo "- num_speculative_tokens: ${NUM_SPEC_TOKENS}"
    echo
    echo "| scenario | request_throughput | output_throughput | mean_ttft_ms | mean_tpot_ms |"
    echo "|----------|--------------------|-------------------|--------------|--------------|"
    for sc in "${scenario_arr[@]}"; do
        bench_json="${OUT_DIR}/${sc}.bench.json"
        if [[ -f "${bench_json}" ]]; then
            python3 - "$sc" "$bench_json" <<'PY'
import json, sys
label, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
def g(k): return d.get(k, "")
print(f"| {label} | {g('request_throughput')} | {g('output_throughput')} | {g('mean_ttft_ms')} | {g('mean_tpot_ms')} |")
PY
        else
            echo "| ${sc} | _no result_ | | | |"
        fi
    done
} > "${SUMMARY}"

echo
echo "summary -> ${SUMMARY}"
cat "${SUMMARY}"
