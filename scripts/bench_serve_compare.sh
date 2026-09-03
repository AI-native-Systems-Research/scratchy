#!/usr/bin/env bash
# Head-to-head `scr bench serve` comparison: scratchy (`scr serve`) vs
# mlx-lm (`mlx_lm.server`), one OpenAI-compat server at a time, common
# bench client (the Rust `scr bench serve`, requires a -Fbench build).
#
# Sweep shape (per backend):
#   - input-lens × output-lens cross product at concurrency 1
#   - concurrency sweep at (--base-input × --base-output)
#   - or an explicit --cells "INxOUTxCONC,..." override
#
# Hard-won rules baked in (see memory feedback_real_benchmarks +
# project_gemma4_port "BENCH LESSONS"):
#   - UNIQUE SEED PER CELL: same --seed ⇒ identical random prompts ⇒ the
#     the scratchy prefix cache serves later cells and TTFT collapses to ~0.
#     Seeds are shared across backends (same prompts ⇒ fair) but unique
#     across cells. Servers are started once per backend, solo.
#   - mlx-lm does NOT honor ignore_eos and under-generates: compare
#     TTFT/TPOT directly; E2E only via the work-normalized
#     E2E* = median_ttft + (output_len-1) * median_tpot. The summary
#     reports mean generated tokens per request so under-generation is
#     visible, and computes E2E* for you.
#   - ONE model resident at a time (24 GiB box): backends run serially,
#     server torn down (and waited on) before the next starts.
#
# Usage:
#   scripts/bench_serve_compare.sh --model mlx-community/Llama-3.2-3B-Instruct-4bit
#   scripts/bench_serve_compare.sh \
#       --model mlx-community/gemma-4-12B-it-4bit \
#       --vllm-bin <gemma4-worktree>/vllm-rs/target/release/scr \
#       --mlx-server "python <gemma4-worktree>/vllm-rs/scripts/gemma4/mlx_lm_server_gemma4.py"
#
# The scratchy binary must be built with the model compiled in — no default
# model scope, name it explicitly (quant defaults are already on for metal):
#   cargo build --release -p scratchy-cli --features metal,bench,model/llama-3.2-3b
#
# Outputs land under vllm-rs/bench_results/serve_compare/<timestamp>_<label>/:
#   <backend>.serve.log, <backend>.<cell>.json, summary.md

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VLLM_RS_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

MODEL=""
VLLM_BIN="${VLLM_RS_DIR}/target/release/scr"
MLX_SERVER="mlx_lm.server"
BACKENDS="scratchy,mlx-lm"
INPUT_LENS="128,512,2048"
OUTPUT_LENS="32,128,512"
CONCURRENCIES="2,4,8"
BASE_INPUT=512
BASE_OUTPUT=128
CELLS=""
NUM_PROMPTS=0          # 0 = auto: max(6, 2*concurrency)
WARMUPS=1
PORT=8731
SEED_BASE=1000
OUT_DIR=""
LABEL=""
SCRATCHY_SERVE_EXTRA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)               MODEL="$2"; shift 2 ;;
        --vllm-bin)            VLLM_BIN="$2"; shift 2 ;;
        --mlx-server)          MLX_SERVER="$2"; shift 2 ;;
        --backends)            BACKENDS="$2"; shift 2 ;;
        --input-lens)          INPUT_LENS="$2"; shift 2 ;;
        --output-lens)         OUTPUT_LENS="$2"; shift 2 ;;
        --concurrencies)       CONCURRENCIES="$2"; shift 2 ;;
        --base-input)          BASE_INPUT="$2"; shift 2 ;;
        --base-output)         BASE_OUTPUT="$2"; shift 2 ;;
        --cells)               CELLS="$2"; shift 2 ;;
        --num-prompts)         NUM_PROMPTS="$2"; shift 2 ;;
        --warmups)             WARMUPS="$2"; shift 2 ;;
        --port)                PORT="$2"; shift 2 ;;
        --seed-base)           SEED_BASE="$2"; shift 2 ;;
        --out-dir)             OUT_DIR="$2"; shift 2 ;;
        --label)               LABEL="$2"; shift 2 ;;
        --scratchy-serve-extra) SCRATCHY_SERVE_EXTRA="$2"; shift 2 ;;
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

if [[ -z "${MODEL}" ]]; then
    echo "--model is required" >&2
    exit 2
fi
if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "scr binary not found/executable: ${VLLM_BIN}" >&2
    exit 1
fi
if ! "${VLLM_BIN}" bench --help >/dev/null 2>&1; then
    echo "${VLLM_BIN} lacks the bench subcommand — rebuild with -Fbench" >&2
    exit 1
fi

# ---- build the cell list ----------------------------------------------------
# Each cell is "INPUTxOUTPUTxCONC".
if [[ -z "${CELLS}" ]]; then
    IFS=',' read -r -a in_arr   <<<"${INPUT_LENS}"
    IFS=',' read -r -a out_arr  <<<"${OUTPUT_LENS}"
    IFS=',' read -r -a conc_arr <<<"${CONCURRENCIES}"
    cells=()
    for il in "${in_arr[@]}"; do
        for ol in "${out_arr[@]}"; do
            cells+=("${il}x${ol}x1")
        done
    done
    for c in "${conc_arr[@]}"; do
        [[ "${c}" == "1" ]] && continue
        cells+=("${BASE_INPUT}x${BASE_OUTPUT}x${c}")
    done
else
    IFS=',' read -r -a cells <<<"${CELLS}"
fi

if [[ -z "${LABEL}" ]]; then
    LABEL="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
fi
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${VLLM_RS_DIR}/bench_results/serve_compare/${TS}_${LABEL}"
fi
mkdir -p "${OUT_DIR}"

{
    echo "model: ${MODEL}"
    echo "vllm_bin: ${VLLM_BIN}"
    echo "mlx_server: ${MLX_SERVER}"
    echo "backends: ${BACKENDS}"
    echo "cells: ${cells[*]}"
    echo "num_prompts: ${NUM_PROMPTS} (0 = max(6, 2*conc))"
    echo "warmups: ${WARMUPS}"
    echo "seed_base: ${SEED_BASE} (seed = base + cell index; shared across backends)"
    echo "scratchy_serve_extra: ${SCRATCHY_SERVE_EXTRA:-<none>}"
} > "${OUT_DIR}/run_config.txt"
echo "results -> ${OUT_DIR}"

# ---- server lifecycle -------------------------------------------------------
SERVER_PID=""
SERVER_LOG=""

wait_for_server() {
    # mlx_lm.server and scratchy `scr serve` both expose /v1/models.
    local url="http://127.0.0.1:${PORT}/v1/models"
    local deadline=$((SECONDS + 600))
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            return 0
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server pid ${SERVER_PID} died before ready; see ${SERVER_LOG}" >&2
            tail -n 60 "${SERVER_LOG}" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "server not ready within 600s; see ${SERVER_LOG}" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        # Wait for the process to actually exit so GPU/RAM is released
        # before the next backend loads (one resident model at a time).
        for _ in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

start_backend() {
    local backend="$1"
    SERVER_LOG="${OUT_DIR}/${backend}.serve.log"

    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "port ${PORT} already serving — refusing to bench a co-resident server" >&2
        return 1
    fi

    case "${backend}" in
        scratchy)
            # shellcheck disable=SC2086  # intentional word-split of extras
            "${VLLM_BIN}" serve "${MODEL}" --port "${PORT}" \
                ${SCRATCHY_SERVE_EXTRA} \
                >"${SERVER_LOG}" 2>&1 &
            ;;
        mlx-lm)
            # shellcheck disable=SC2086  # MLX_SERVER may be "python script.py"
            ${MLX_SERVER} --model "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
                >"${SERVER_LOG}" 2>&1 &
            ;;
        *)
            echo "unknown backend: ${backend}" >&2
            return 1
            ;;
    esac
    SERVER_PID=$!
    echo "[${backend}] server pid ${SERVER_PID}, log ${SERVER_LOG}"
    wait_for_server
}

# ---- bench one cell ---------------------------------------------------------
run_cell() {
    local backend="$1" cell="$2" idx="$3"
    local il ol conc
    IFS='x' read -r il ol conc <<<"${cell}"

    local np="${NUM_PROMPTS}"
    if [[ "${np}" -eq 0 ]]; then
        np=$(( conc * 2 ))
        (( np < 6 )) && np=6
    fi
    local seed=$(( SEED_BASE + idx ))
    local tag="${il}x${ol}x${conc}"
    local json="${OUT_DIR}/${backend}.${tag}.json"
    local log="${OUT_DIR}/${backend}.${tag}.bench.log"

    echo
    echo "=== [${backend}] cell ${tag} (prompts=${np} seed=${seed}) ==="
    "${VLLM_BIN}" bench serve \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${MODEL}" \
        --num-prompts "${np}" \
        --input-len "${il}" \
        --output-len "${ol}" \
        --max-concurrency "${conc}" \
        --seed "${seed}" \
        --num-warmups "${WARMUPS}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,99 \
        --output-json "${json}" \
        --disable-tqdm \
        2>&1 | tee "${log}" | tail -n 4
}

# ---- run --------------------------------------------------------------------
IFS=',' read -r -a backend_arr <<<"${BACKENDS}"
for backend in "${backend_arr[@]}"; do
    echo
    echo "######## backend: ${backend} ########"
    if ! start_backend "${backend}"; then
        stop_server
        echo "[${backend}] failed to start; continuing to next backend" >&2
        continue
    fi
    idx=0
    for cell in "${cells[@]}"; do
        run_cell "${backend}" "${cell}" "${idx}" || \
            echo "[${backend}] cell ${cell} failed; continuing" >&2
        idx=$(( idx + 1 ))
    done
    stop_server
done

# ---- summary ----------------------------------------------------------------
SUMMARY="${OUT_DIR}/summary.md"
python3 - "${OUT_DIR}" "${MODEL}" "${BACKENDS}" "${cells[@]}" <<'PY' > "${SUMMARY}"
import json, os, sys

out_dir, model, backends_csv = sys.argv[1], sys.argv[2], sys.argv[3]
cells = sys.argv[4:]
backends = backends_csv.split(",")

def load(backend, cell):
    path = os.path.join(out_dir, f"{backend}.{cell}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

print(f"# serve compare — {model}")
print()
print(f"results: `{out_dir}`")
print()
print("TTFT/TPOT are per-request medians (ms). gen/req = mean generated")
print("tokens per request — mlx-lm ignores `ignore_eos`, so it under-")
print("generates; E2E* = ttft + (out-1)*tpot is the work-normalized")
print("end-to-end estimate that stays fair under early EOS.")
print()
hdr = ["cell (in×out×conc)"]
for b in backends:
    hdr += [f"{b} ttft", f"{b} tpot", f"{b} gen/req"]
if len(backends) == 2:
    a, b = backends
    hdr += [f"E2E* {b}/{a}"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))

def e2e_star(d, out_len):
    return d["median_ttft_ms"] + (out_len - 1) * d["median_tpot_ms"]

for cell in cells:
    il, ol, conc = (int(x) for x in cell.split("x"))
    row = [cell]
    datas = []
    for bk in backends:
        d = load(bk, cell)
        datas.append(d)
        if d is None:
            row += ["—", "—", "—"]
            continue
        gen = d["total_output_tokens"] / max(d["completed"], 1)
        row += [
            f"{d['median_ttft_ms']:.0f}",
            f"{d['median_tpot_ms']:.1f}",
            f"{gen:.0f}/{ol}",
        ]
    if len(backends) == 2 and all(datas):
        ratio = e2e_star(datas[1], ol) / e2e_star(datas[0], ol)
        row += [f"{ratio:.2f}×"]
    elif len(backends) == 2:
        row += ["—"]
    print("| " + " | ".join(row) + " |")
PY

echo
echo "summary -> ${SUMMARY}"
cat "${SUMMARY}"
